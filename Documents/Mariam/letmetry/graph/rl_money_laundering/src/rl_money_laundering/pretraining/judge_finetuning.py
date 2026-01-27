"""
Judge fine-tuning pipeline for AML decision behavior.

After TAPT, the model has good domain embeddings but needs decision behavior.
This module fine-tunes supervised heads on AML-specific tasks:

1. Typology Classification: Multi-label classification of AML patterns
2. Suspiciousness Scoring: Binary/regression scoring of transaction risk
3. Evidence Selection: Identifying red flag sentences/tokens
4. Pairwise Ranking: Comparative judgment for investigation prioritization

Usage:
    from rl_money_laundering.pretraining.judge_finetuning import (
        JudgeFinetuner,
        JudgeFinetuneConfig,
        run_judge_finetuning,
    )

    # Fine-tune typology classifier
    finetuner = JudgeFinetuner(
        base_model="./models/tapt",
        task="typology",
        output_dir="./models/judge",
    )
    finetuner.train(train_data, eval_data)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .supervised_heads import (
    HeadConfig,
    PoolingStrategy,
    TypologyClassificationHead,
    SuspiciousnessScorer,
    EvidenceSelector,
    PairwiseRanker,
    create_supervised_head,
    AML_TYPOLOGIES,
)
from .ranking_loss import (
    create_ranking_loss,
    compute_ndcg,
    compute_map,
)


class JudgeTask(Enum):
    """Judge fine-tuning tasks."""
    TYPOLOGY = "typology"
    SUSPICIOUSNESS = "suspiciousness"
    EVIDENCE = "evidence"
    RANKING = "ranking"


@dataclass
class JudgeFinetuneConfig:
    """Configuration for judge fine-tuning."""

    # Model
    base_model: str = "./models/tapt"
    tokenizer_name: Optional[str] = None

    # Task
    task: JudgeTask = JudgeTask.TYPOLOGY
    num_labels: int = len(AML_TYPOLOGIES)
    label_names: Optional[List[str]] = None

    # Head configuration
    pooling: PoolingStrategy = PoolingStrategy.CLS
    dropout_prob: float = 0.1
    freeze_encoder: bool = True
    freeze_encoder_layers: Optional[int] = None

    # Training
    output_dir: Path = field(default_factory=lambda: Path("./models/judge"))
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 512

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 100
    metric_for_best_model: str = "f1"

    # Calibration
    use_temperature_scaling: bool = True
    calibration_samples: int = 1000

    # Multi-task (optional)
    multi_task: bool = False
    task_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if isinstance(self.task, str):
            self.task = JudgeTask(self.task)
        if self.label_names is None:
            self.label_names = AML_TYPOLOGIES[:self.num_labels]


@dataclass
class JudgeFinetuneResult:
    """Results from judge fine-tuning."""

    task: str
    best_metric: float
    final_metrics: Dict[str, float]
    training_time_seconds: float
    checkpoint_path: str
    calibration_temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "best_metric": self.best_metric,
            "final_metrics": self.final_metrics,
            "training_time_seconds": self.training_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "calibration_temperature": self.calibration_temperature,
        }


if HAS_TORCH and HAS_TRANSFORMERS:

    class JudgeDataset(Dataset):
        """Dataset for judge fine-tuning tasks."""

        def __init__(
            self,
            data: List[Dict[str, Any]],
            tokenizer: Any,
            task: JudgeTask,
            max_length: int = 512,
            label_names: Optional[List[str]] = None,
        ) -> None:
            """Initialize judge dataset.

            Args:
                data: List of samples with 'text' and 'labels' keys.
                tokenizer: HuggingFace tokenizer.
                task: Judge task type.
                max_length: Maximum sequence length.
                label_names: Names of labels for multi-label tasks.
            """
            self.data = data
            self.tokenizer = tokenizer
            self.task = task
            self.max_length = max_length
            self.label_names = label_names or []

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            sample = self.data[idx]

            # Tokenize text
            text = sample["text"]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            result = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }

            # Handle labels based on task
            if self.task == JudgeTask.TYPOLOGY:
                # Multi-hot encoding
                labels = sample.get("labels", [])
                if isinstance(labels, list):
                    label_vector = torch.zeros(len(self.label_names))
                    for label in labels:
                        if label in self.label_names:
                            label_vector[self.label_names.index(label)] = 1.0
                        elif isinstance(label, int):
                            label_vector[label] = 1.0
                    result["labels"] = label_vector
                else:
                    result["labels"] = torch.tensor(labels)

            elif self.task == JudgeTask.SUSPICIOUSNESS:
                # Single score
                score = sample.get("score", sample.get("labels", 0.0))
                result["labels"] = torch.tensor(float(score))

            elif self.task == JudgeTask.EVIDENCE:
                # Token-level or sentence-level labels
                labels = sample.get("labels", [])
                if isinstance(labels[0] if labels else 0, int):
                    # Token-level BIO tags
                    label_ids = labels + [-100] * (self.max_length - len(labels))
                    result["labels"] = torch.tensor(label_ids[:self.max_length])
                else:
                    # Sentence-level
                    result["labels"] = torch.tensor(labels)

            elif self.task == JudgeTask.RANKING:
                # Pairwise ranking (handled separately)
                pass

            return result

    class PairwiseRankingDataset(Dataset):
        """Dataset for pairwise ranking task."""

        def __init__(
            self,
            data: List[Dict[str, Any]],
            tokenizer: Any,
            max_length: int = 512,
        ) -> None:
            """Initialize pairwise ranking dataset.

            Args:
                data: List of pairs with 'text_a', 'text_b', 'label' keys.
                    label: +1 if A is more suspicious, -1 if B is more suspicious.
                tokenizer: HuggingFace tokenizer.
                max_length: Maximum sequence length.
            """
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            sample = self.data[idx]

            # Tokenize both texts
            encoding_a = self.tokenizer(
                sample["text_a"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoding_b = self.tokenizer(
                sample["text_b"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids_a": encoding_a["input_ids"].squeeze(0),
                "attention_mask_a": encoding_a["attention_mask"].squeeze(0),
                "input_ids_b": encoding_b["input_ids"].squeeze(0),
                "attention_mask_b": encoding_b["attention_mask"].squeeze(0),
                "labels": torch.tensor(sample["label"]),
            }

    class JudgeFinetuner:
        """Fine-tuner for judge supervised heads."""

        def __init__(self, config: JudgeFinetuneConfig) -> None:
            """Initialize judge finetuner.

            Args:
                config: Fine-tuning configuration.
            """
            self.config = config
            self.model: Optional[nn.Module] = None
            self.encoder: Optional[nn.Module] = None
            self.tokenizer: Optional[Any] = None
            self.trainer: Optional[Trainer] = None

        def setup(self) -> None:
            """Set up model, tokenizer, and training components."""
            logger.info(f"Loading base model from {self.config.base_model}")

            # Load tokenizer
            tokenizer_name = self.config.tokenizer_name or self.config.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load encoder
            self.encoder = AutoModel.from_pretrained(self.config.base_model)

            # Create head configuration
            head_config = HeadConfig(
                num_labels=self.config.num_labels,
                dropout_prob=self.config.dropout_prob,
                pooling=self.config.pooling,
                freeze_encoder=self.config.freeze_encoder,
                freeze_encoder_layers=self.config.freeze_encoder_layers,
            )

            # Create supervised head
            if self.config.task == JudgeTask.TYPOLOGY:
                self.model = TypologyClassificationHead(
                    self.encoder,
                    self.config.num_labels,
                    head_config,
                    self.config.label_names,
                )
            elif self.config.task == JudgeTask.SUSPICIOUSNESS:
                self.model = SuspiciousnessScorer(
                    self.encoder,
                    head_config,
                    regression=True,
                )
            elif self.config.task == JudgeTask.EVIDENCE:
                self.model = EvidenceSelector(
                    self.encoder,
                    head_config,
                    token_level=True,
                )
            elif self.config.task == JudgeTask.RANKING:
                self.model = PairwiseRanker(
                    self.encoder,
                    head_config,
                    margin=0.5,
                )

            logger.info(f"Model setup complete. Task: {self.config.task.value}")

        def train(
            self,
            train_data: List[Dict[str, Any]],
            eval_data: Optional[List[Dict[str, Any]]] = None,
        ) -> JudgeFinetuneResult:
            """Train the judge model.

            Args:
                train_data: Training samples.
                eval_data: Evaluation samples.

            Returns:
                Fine-tuning results.
            """
            if self.model is None:
                self.setup()

            start_time = datetime.now()

            # Create datasets
            if self.config.task == JudgeTask.RANKING:
                train_dataset = PairwiseRankingDataset(
                    train_data,
                    self.tokenizer,
                    self.config.max_seq_length,
                )
                eval_dataset = None
                if eval_data:
                    eval_dataset = PairwiseRankingDataset(
                        eval_data,
                        self.tokenizer,
                        self.config.max_seq_length,
                    )
            else:
                train_dataset = JudgeDataset(
                    train_data,
                    self.tokenizer,
                    self.config.task,
                    self.config.max_seq_length,
                    self.config.label_names,
                )
                eval_dataset = None
                if eval_data:
                    eval_dataset = JudgeDataset(
                        eval_data,
                        self.tokenizer,
                        self.config.task,
                        self.config.max_seq_length,
                        self.config.label_names,
                    )

            # Create output directory
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                load_best_model_at_end=eval_dataset is not None,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=True,
                logging_steps=50,
                report_to=["tensorboard"],
            )

            # Custom trainer for different tasks
            if self.config.task == JudgeTask.RANKING:
                trainer = self._create_ranking_trainer(
                    train_dataset,
                    eval_dataset,
                    training_args,
                )
            else:
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=self._get_compute_metrics(),
                )

            # Train
            logger.info("Starting training...")
            trainer.train()

            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)

            # Evaluate
            final_metrics = {}
            if eval_dataset:
                final_metrics = trainer.evaluate()

            # Calibration
            calibration_temp = None
            if self.config.use_temperature_scaling and eval_data:
                calibration_temp = self._calibrate(eval_data[:self.config.calibration_samples])

            elapsed = (datetime.now() - start_time).total_seconds()

            result = JudgeFinetuneResult(
                task=self.config.task.value,
                best_metric=final_metrics.get(f"eval_{self.config.metric_for_best_model}", 0.0),
                final_metrics=final_metrics,
                training_time_seconds=elapsed,
                checkpoint_path=str(self.config.output_dir),
                calibration_temperature=calibration_temp,
            )

            # Save results
            with open(self.config.output_dir / "results.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.info(f"Training complete in {elapsed:.1f}s")
            return result

        def _get_compute_metrics(self) -> Callable:
            """Get metrics computation function for the task."""

            if not HAS_SKLEARN:
                return None

            if self.config.task == JudgeTask.TYPOLOGY:
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

                    # Micro and macro F1
                    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
                    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

                    return {
                        "f1": f1_micro,
                        "f1_micro": f1_micro,
                        "f1_macro": f1_macro,
                    }
                return compute_metrics

            elif self.config.task == JudgeTask.SUSPICIOUSNESS:
                def compute_metrics(eval_pred):
                    scores, labels = eval_pred
                    predictions = (scores > 0.5).astype(int)

                    acc = accuracy_score(labels > 0.5, predictions)
                    f1 = f1_score(labels > 0.5, predictions, zero_division=0)

                    try:
                        auc = roc_auc_score(labels > 0.5, scores)
                    except ValueError:
                        auc = 0.0

                    return {
                        "accuracy": acc,
                        "f1": f1,
                        "auc": auc,
                    }
                return compute_metrics

            elif self.config.task == JudgeTask.EVIDENCE:
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = logits.argmax(axis=-1)

                    # Flatten and remove padding (-100)
                    mask = labels != -100
                    flat_preds = predictions[mask]
                    flat_labels = labels[mask]

                    f1 = f1_score(flat_labels, flat_preds, average="macro", zero_division=0)

                    return {"f1": f1}
                return compute_metrics

            return None

        def _create_ranking_trainer(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset],
            training_args: TrainingArguments,
        ) -> Trainer:
            """Create trainer for pairwise ranking task."""

            class RankingTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    outputs = model(
                        input_ids_a=inputs["input_ids_a"],
                        input_ids_b=inputs["input_ids_b"],
                        attention_mask_a=inputs["attention_mask_a"],
                        attention_mask_b=inputs["attention_mask_b"],
                        labels=inputs["labels"],
                    )
                    loss = outputs["loss"]
                    return (loss, outputs) if return_outputs else loss

            def compute_ranking_metrics(eval_pred):
                # For ranking, track accuracy of pairwise comparisons
                predictions, labels = eval_pred
                correct = (predictions * labels > 0).mean()
                return {"accuracy": correct}

            return RankingTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_ranking_metrics,
            )

        def _calibrate(self, calibration_data: List[Dict[str, Any]]) -> float:
            """Perform temperature scaling calibration.

            Args:
                calibration_data: Data for calibration.

            Returns:
                Optimal temperature.
            """
            logger.info("Calibrating model with temperature scaling...")

            # Get predictions on calibration set
            self.model.eval()
            all_logits = []
            all_labels = []

            dataset = JudgeDataset(
                calibration_data,
                self.tokenizer,
                self.config.task,
                self.config.max_seq_length,
                self.config.label_names,
            )
            loader = DataLoader(dataset, batch_size=self.config.batch_size)

            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()

                    outputs = self.model(input_ids, attention_mask)
                    all_logits.append(outputs["logits"].cpu())
                    all_labels.append(labels)

            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)

            # Grid search for optimal temperature
            best_temp = 1.0
            best_loss = float("inf")

            for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
                scaled_logits = logits / temp

                if self.config.task == JudgeTask.TYPOLOGY:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        scaled_logits, labels.float()
                    ).item()
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        scaled_logits.squeeze(), labels.float()
                    ).item()

                if loss < best_loss:
                    best_loss = loss
                    best_temp = temp

            # Apply calibration to model
            if hasattr(self.model, "config"):
                self.model.config.temperature = best_temp

            logger.info(f"Calibration complete. Optimal temperature: {best_temp}")
            return best_temp

        def predict(
            self,
            texts: List[str],
            batch_size: int = 16,
        ) -> List[Dict[str, Any]]:
            """Run inference on new texts.

            Args:
                texts: List of input texts.
                batch_size: Batch size for inference.

            Returns:
                List of prediction dictionaries.
            """
            if self.model is None:
                raise RuntimeError("Model not initialized. Call setup() or train() first.")

            self.model.eval()
            results = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_seq_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    encodings = {k: v.cuda() for k, v in encodings.items()}

                with torch.no_grad():
                    if self.config.task == JudgeTask.TYPOLOGY:
                        predictions = self.model.predict(
                            encodings["input_ids"],
                            encodings["attention_mask"],
                        )
                        for j in range(len(batch_texts)):
                            results.append({
                                "text": batch_texts[j],
                                "labels": predictions["labels"][j],
                                "probabilities": predictions["probabilities"][j].cpu().tolist(),
                            })

                    elif self.config.task == JudgeTask.SUSPICIOUSNESS:
                        scores = self.model.predict(
                            encodings["input_ids"],
                            encodings["attention_mask"],
                        )
                        for j in range(len(batch_texts)):
                            results.append({
                                "text": batch_texts[j],
                                "score": scores[j].cpu().item(),
                            })

                    elif self.config.task == JudgeTask.EVIDENCE:
                        spans = self.model.predict_evidence_spans(
                            encodings["input_ids"],
                            encodings["attention_mask"],
                        )
                        for j in range(len(batch_texts)):
                            results.append({
                                "text": batch_texts[j],
                                "evidence_spans": spans[j],
                            })

            return results


def run_judge_finetuning(
    base_model: str,
    task: str,
    train_data: List[Dict[str, Any]],
    eval_data: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> JudgeFinetuneResult:
    """Convenience function to run judge fine-tuning.

    Args:
        base_model: Path to TAPT checkpoint or model name.
        task: Task type (typology, suspiciousness, evidence, ranking).
        train_data: Training samples.
        eval_data: Evaluation samples.
        output_dir: Output directory.
        **kwargs: Additional config options.

    Returns:
        Fine-tuning results.
    """
    config = JudgeFinetuneConfig(
        base_model=base_model,
        task=JudgeTask(task),
        output_dir=Path(output_dir) if output_dir else Path("./models/judge"),
        **kwargs,
    )

    finetuner = JudgeFinetuner(config)
    return finetuner.train(train_data, eval_data)


def load_judge_model(
    checkpoint_path: str,
    task: str,
) -> Tuple["nn.Module", Any]:
    """Load a trained judge model.

    Args:
        checkpoint_path: Path to model checkpoint.
        task: Task type.

    Returns:
        Tuple of (model, tokenizer).
    """
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        raise ImportError("torch and transformers required")

    checkpoint_path = Path(checkpoint_path)

    # Load results to get configuration
    results_path = checkpoint_path / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load encoder
    encoder = AutoModel.from_pretrained(checkpoint_path)

    # Create appropriate head
    task_enum = JudgeTask(task)
    head_config = HeadConfig()

    if task_enum == JudgeTask.TYPOLOGY:
        model = TypologyClassificationHead(encoder, config=head_config)
    elif task_enum == JudgeTask.SUSPICIOUSNESS:
        model = SuspiciousnessScorer(encoder, config=head_config)
    elif task_enum == JudgeTask.EVIDENCE:
        model = EvidenceSelector(encoder, config=head_config)
    elif task_enum == JudgeTask.RANKING:
        model = PairwiseRanker(encoder, config=head_config)

    # Load head weights
    head_weights_path = checkpoint_path / "head_weights.pt"
    if head_weights_path.exists():
        head_state = torch.load(head_weights_path, map_location="cpu")
        model.load_state_dict(head_state, strict=False)

    return model, tokenizer
