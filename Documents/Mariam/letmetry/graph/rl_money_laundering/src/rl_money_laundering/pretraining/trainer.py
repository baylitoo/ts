"""
Continuous Pretraining Trainer.

Wraps HuggingFace Trainer with AML-specific customizations:
- Domain/Task-adaptive pretraining stages
- Curriculum learning support
- Mixed-source training
- Custom logging and metrics
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from transformers import (
        AutoModelForMaskedLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        PreTrainedModel,
        PreTrainedTokenizer,
        get_scheduler,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. Install with: pip install transformers")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

from .config import (
    PretrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    PretrainingObjective,
    ModelArchitecture,
)
from .collators import (
    AMLTextDataset,
    StreamingAMLDataset,
    MLMDataCollator,
    SpanMLMCollator,
    CLMDataCollator,
    SpanCorruptionCollator,
    create_data_collator,
)


@dataclass
class PretrainingMetrics:
    """Metrics collected during pretraining."""

    epoch: int
    step: int
    loss: float
    perplexity: float
    learning_rate: float
    tokens_seen: int
    samples_seen: int
    throughput_tokens_per_sec: float
    gpu_memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


class PretrainingCallback(TrainerCallback if HAS_TRANSFORMERS else object):
    """Custom callback for pretraining monitoring."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        tokens_per_sample: int = 256,
    ) -> None:
        """Initialize callback.

        Args:
            log_file: Path to log metrics.
            tokens_per_sample: Average tokens per sample for throughput calculation.
        """
        self.log_file = log_file
        self.tokens_per_sample = tokens_per_sample
        self.start_time: Optional[datetime] = None
        self.metrics_history: List[PretrainingMetrics] = []

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        logger.info("Starting pretraining...")

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        """Called when logging metrics."""
        if logs is None:
            return

        loss = logs.get("loss", 0.0)
        perplexity = math.exp(min(loss, 20))  # Cap to avoid overflow

        # Calculate throughput
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 1
        tokens_seen = state.global_step * args.per_device_train_batch_size * self.tokens_per_sample
        throughput = tokens_seen / elapsed

        # Get GPU memory
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        metrics = PretrainingMetrics(
            epoch=int(state.epoch) if state.epoch else 0,
            step=state.global_step,
            loss=loss,
            perplexity=perplexity,
            learning_rate=logs.get("learning_rate", 0.0),
            tokens_seen=tokens_seen,
            samples_seen=state.global_step * args.per_device_train_batch_size,
            throughput_tokens_per_sec=throughput,
            gpu_memory_mb=gpu_memory,
        )

        self.metrics_history.append(metrics)

        # Log to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """Called at the end of training."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"Pretraining completed in {elapsed / 3600:.2f} hours")


class EarlyStoppingCallback(TrainerCallback if HAS_TRANSFORMERS else object):
    """Early stopping based on validation loss or perplexity.

    Supports two stopping criteria:
    1. Validation loss (default): Stop when eval_loss stops improving
    2. Perplexity: Stop when eval perplexity stops improving or exceeds threshold

    For TAPT with small corpora, perplexity monitoring helps detect overfitting
    early by watching for perplexity increases on the validation set.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        metric: str = "loss",  # "loss" or "perplexity"
        max_perplexity: Optional[float] = None,  # Stop if perplexity exceeds this
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of evaluations without improvement before stopping.
            min_delta: Minimum improvement required (absolute for loss, relative for perplexity).
            metric: Metric to monitor ("loss" or "perplexity").
            max_perplexity: Optional perplexity ceiling; stop immediately if exceeded.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.max_perplexity = max_perplexity

        self.best_value = float("inf")
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        metrics: Dict[str, float] = None,
        **kwargs,
    ):
        """Check for improvement after evaluation."""
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss", float("inf"))
        # Compute perplexity from loss (capped to avoid overflow)
        eval_perplexity = math.exp(min(eval_loss, 20))

        # Record history
        self.history.append({
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": eval_loss,
            "perplexity": eval_perplexity,
        })

        # Check perplexity ceiling
        if self.max_perplexity is not None and eval_perplexity > self.max_perplexity:
            logger.warning(
                f"Perplexity {eval_perplexity:.2f} exceeds max threshold "
                f"{self.max_perplexity:.2f}. Stopping training."
            )
            control.should_training_stop = True
            return

        # Select metric to monitor
        if self.metric == "perplexity":
            current_value = eval_perplexity
            metric_name = "perplexity"
        else:
            current_value = eval_loss
            metric_name = "loss"

        # Check for improvement
        if self.metric == "perplexity":
            # For perplexity, use relative improvement
            improvement = (self.best_value - current_value) / max(self.best_value, 1e-8)
            improved = improvement > self.min_delta
        else:
            # For loss, use absolute improvement
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.patience_counter = 0
            logger.info(f"New best validation {metric_name}: {current_value:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement ({self.patience_counter}/{self.patience}). "
                f"Best {metric_name}: {self.best_value:.4f}, Current: {current_value:.4f}"
            )

        if self.patience_counter >= self.patience:
            logger.info(f"Early stopping triggered after {len(self.history)} evaluations")
            control.should_training_stop = True

    def get_history(self) -> List[Dict[str, float]]:
        """Get evaluation history.

        Returns:
            List of evaluation metrics at each step.
        """
        return self.history


class ContinuousPretrainer:
    """Main class for continuous pretraining.

    Handles:
    - Model and tokenizer loading
    - Data preparation
    - Training configuration
    - Checkpointing and resumption
    """

    def __init__(self, config: PretrainingConfig) -> None:
        """Initialize pretrainer.

        Args:
            config: Complete pretraining configuration.
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for pretraining")

        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def setup_model_and_tokenizer(self) -> None:
        """Load and configure model and tokenizer."""
        model_config = self.config.model
        logger.info(f"Loading model: {model_config.model_name_or_path}")

        # Load tokenizer
        tokenizer_name = model_config.tokenizer_name or model_config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens if specified
        if model_config.add_special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": model_config.add_special_tokens
            })

        # Load model based on architecture
        model_kwargs = {}

        # Handle quantization
        if model_config.use_4bit and HAS_BITSANDBYTES:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif model_config.use_8bit and HAS_BITSANDBYTES:
            model_kwargs["load_in_8bit"] = True

        # Load appropriate model type
        if model_config.objective == PretrainingObjective.MLM:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_config.model_name_or_path,
                **model_kwargs,
            )
        elif model_config.objective == PretrainingObjective.CLM:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path,
                **model_kwargs,
            )
        else:
            # Default to MLM
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_config.model_name_or_path,
                **model_kwargs,
            )

        # Resize embeddings if tokens were added
        if model_config.resize_embeddings or model_config.add_special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if specified
        if model_config.use_lora:
            self._apply_lora()

        # Enable gradient checkpointing if specified
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(f"Model loaded. Parameters: {self._count_parameters()}")

    def _apply_lora(self) -> None:
        """Apply LoRA to the model."""
        if not HAS_PEFT:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        model_config = self.config.model

        # Prepare model for k-bit training if quantized
        if model_config.use_4bit or model_config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Determine task type
        if model_config.objective == PretrainingObjective.CLM:
            task_type = TaskType.CAUSAL_LM
        else:
            task_type = TaskType.FEATURE_EXTRACTION  # MLM doesn't have direct task type

        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
            bias="none",
            task_type=task_type,
        )

        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"LoRA applied. Trainable parameters: {self._count_parameters(trainable_only=True)}")

    def _count_parameters(self, trainable_only: bool = False) -> str:
        """Count model parameters."""
        if trainable_only:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            params = sum(p.numel() for p in self.model.parameters())

        if params >= 1e9:
            return f"{params / 1e9:.2f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.2f}M"
        else:
            return f"{params / 1e3:.2f}K"

    def setup_data(self) -> None:
        """Prepare training and validation datasets."""
        data_config = self.config.data

        if data_config.train_file is None and data_config.corpus_dir is None:
            raise ValueError("Either train_file or corpus_dir must be specified")

        # Find data files
        train_files = []
        val_files = []

        if data_config.train_file:
            train_files.append(data_config.train_file)
        if data_config.validation_file:
            val_files.append(data_config.validation_file)

        if data_config.corpus_dir:
            corpus_files = list(data_config.corpus_dir.glob("*.jsonl"))
            # Split 90/10 if no separate validation file
            if not val_files and len(corpus_files) > 1:
                split_idx = int(len(corpus_files) * 0.9)
                train_files.extend(corpus_files[:split_idx])
                val_files.extend(corpus_files[split_idx:])
            else:
                train_files.extend(corpus_files)

        logger.info(f"Training files: {len(train_files)}")
        logger.info(f"Validation files: {len(val_files)}")

        # Create datasets
        if data_config.streaming:
            self.train_dataset = StreamingAMLDataset(
                file_paths=train_files,
                tokenizer=self.tokenizer,
                max_length=data_config.max_seq_length,
                shuffle_buffer_size=data_config.shuffle_buffer_size,
                seed=self.config.training.seed,
            )
            if val_files:
                self.eval_dataset = StreamingAMLDataset(
                    file_paths=val_files,
                    tokenizer=self.tokenizer,
                    max_length=data_config.max_seq_length,
                    seed=self.config.training.seed,
                )
        else:
            # Load all files into single dataset
            if len(train_files) == 1:
                self.train_dataset = AMLTextDataset(
                    file_path=train_files[0],
                    tokenizer=self.tokenizer,
                    max_length=data_config.max_seq_length,
                )
            else:
                # Concatenate multiple files
                all_samples = []
                for f in train_files:
                    ds = AMLTextDataset(
                        file_path=f,
                        tokenizer=self.tokenizer,
                        max_length=data_config.max_seq_length,
                        cache_tokenized=False,
                    )
                    all_samples.extend(ds.samples)

                # Create combined dataset
                self.train_dataset = AMLTextDataset.__new__(AMLTextDataset)
                self.train_dataset.samples = all_samples
                self.train_dataset.tokenizer = self.tokenizer
                self.train_dataset.max_length = data_config.max_seq_length
                self.train_dataset._tokenized_cache = {}
                self.train_dataset.cache_tokenized = True

            if val_files:
                if len(val_files) == 1:
                    self.eval_dataset = AMLTextDataset(
                        file_path=val_files[0],
                        tokenizer=self.tokenizer,
                        max_length=data_config.max_seq_length,
                    )

        logger.info(f"Training samples: {len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else 'streaming'}")

    def setup_trainer(self) -> None:
        """Configure HuggingFace Trainer."""
        training_config = self.config.training
        data_config = self.config.data

        # Create output directory
        training_config.output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(training_config.output_dir),
            run_name=training_config.run_name or f"pretrain_{self.config.stage}",
            num_train_epochs=training_config.num_train_epochs,
            max_steps=training_config.max_steps,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            warmup_steps=training_config.warmup_steps,
            lr_scheduler_type=training_config.lr_scheduler_type,
            optim=training_config.optim,
            adam_beta1=training_config.adam_beta1,
            adam_beta2=training_config.adam_beta2,
            adam_epsilon=training_config.adam_epsilon,
            max_grad_norm=training_config.max_grad_norm,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            tf32=training_config.tf32,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=training_config.load_best_model_at_end,
            eval_strategy=training_config.eval_strategy if self.eval_dataset else "no",
            eval_steps=training_config.eval_steps,
            logging_dir=str(training_config.logging_dir),
            logging_steps=training_config.logging_steps,
            report_to=training_config.report_to,
            dataloader_num_workers=training_config.dataloader_num_workers,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            seed=training_config.seed,
            ddp_find_unused_parameters=training_config.ddp_find_unused_parameters,
            deepspeed=training_config.deepspeed,
        )

        # Create data collator
        if self.config.model.objective == PretrainingObjective.MLM:
            # Check if span MLM should be used (for TAPT)
            if data_config.use_span_mlm:
                data_collator = SpanMLMCollator(
                    tokenizer=self.tokenizer,
                    mlm_probability=data_config.mlm_probability,
                    mean_span_length=data_config.mean_noise_span_length,
                    max_span_length=data_config.max_span_length,
                )
                logger.info(
                    f"Using SpanMLMCollator with mean_span_length={data_config.mean_noise_span_length}"
                )
            else:
                data_collator = MLMDataCollator(
                    tokenizer=self.tokenizer,
                    mlm_probability=data_config.mlm_probability,
                    whole_word_masking=data_config.whole_word_masking,
                )
        elif self.config.model.objective == PretrainingObjective.CLM:
            data_collator = CLMDataCollator(tokenizer=self.tokenizer)
        elif self.config.model.objective == PretrainingObjective.SPAN_MLM:
            # Explicit span MLM objective
            data_collator = SpanMLMCollator(
                tokenizer=self.tokenizer,
                mlm_probability=data_config.mlm_probability,
                mean_span_length=data_config.mean_noise_span_length,
                max_span_length=data_config.max_span_length,
            )
        else:
            data_collator = SpanCorruptionCollator(
                tokenizer=self.tokenizer,
                noise_density=data_config.noise_density,
                mean_noise_span_length=data_config.mean_noise_span_length,
            )

        # Callbacks
        callbacks = [
            PretrainingCallback(
                log_file=training_config.output_dir / "metrics.jsonl",
                tokens_per_sample=data_config.max_seq_length,
            ),
        ]

        # Configure early stopping based on stage
        if self.eval_dataset:
            if self.config.stage == "tapt":
                # For TAPT, use perplexity-based early stopping with stricter settings
                callbacks.append(EarlyStoppingCallback(
                    patience=3,
                    min_delta=0.01,  # 1% relative improvement for perplexity
                    metric="perplexity",
                ))
                logger.info("Using perplexity-based early stopping for TAPT")
            else:
                # For DAPT, use loss-based early stopping
                callbacks.append(EarlyStoppingCallback(patience=3, metric="loss"))

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run pretraining.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Training metrics.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")

        logger.info("Starting training...")

        # Resume from checkpoint if specified
        checkpoint = resume_from_checkpoint
        if checkpoint is None and self.config.continue_from_checkpoint:
            checkpoint = str(self.config.continue_from_checkpoint)

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        logger.info(f"Training complete. Model saved to {self.config.training.output_dir}")

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Evaluation metrics.
        """
        if self.trainer is None or self.eval_dataset is None:
            raise RuntimeError("Trainer or eval dataset not initialized")

        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def run(self) -> Dict[str, Any]:
        """Run complete pretraining pipeline.

        Returns:
            Final training metrics.
        """
        self.setup_model_and_tokenizer()
        self.setup_data()
        self.setup_trainer()

        metrics = self.train()

        if self.eval_dataset:
            eval_metrics = self.evaluate()
            metrics.update(eval_metrics)

        return metrics


def create_pretrainer(
    model_name: str = "roberta-base",
    corpus_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    stage: str = "dapt",
    **kwargs,
) -> ContinuousPretrainer:
    """Create a pretrainer with default settings.

    Args:
        model_name: Base model name or path.
        corpus_path: Path to training corpus.
        output_dir: Output directory for checkpoints.
        stage: Pretraining stage (dapt or tapt).
        **kwargs: Additional configuration options.

    Returns:
        Configured ContinuousPretrainer.
    """
    if stage == "dapt":
        config = PretrainingConfig.for_dapt(
            model_name=model_name,
            corpus_path=Path(corpus_path) if corpus_path else None,
            output_dir=Path(output_dir) if output_dir else None,
        )
    else:
        config = PretrainingConfig.for_tapt(
            base_model=model_name,
            corpus_path=Path(corpus_path) if corpus_path else None,
            output_dir=Path(output_dir) if output_dir else None,
        )

    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)

    return ContinuousPretrainer(config)
