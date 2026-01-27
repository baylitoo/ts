"""
Configuration classes for continuous pretraining.

Includes hardware estimation based on corpus size and model parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PretrainingObjective(Enum):
    """Pretraining objective types."""
    MLM = "mlm"  # Masked Language Modeling (BERT-style)
    SPAN_MLM = "span_mlm"  # Span MLM with Geometric distribution (for TAPT)
    CLM = "clm"  # Causal Language Modeling (GPT-style)
    SPAN_CORRUPTION = "span"  # Span corruption (T5-style)
    REPLACED_TOKEN = "rtd"  # Replaced token detection (ELECTRA-style)


class ModelArchitecture(Enum):
    """Model architecture types."""
    ENCODER_ONLY = "encoder"  # BERT, RoBERTa
    DECODER_ONLY = "decoder"  # GPT, Llama
    ENCODER_DECODER = "enc_dec"  # T5, BART


# Model parameter counts (approximate, in millions)
MODEL_PARAMS = {
    # Encoder-only
    "bert-base-uncased": 110,
    "bert-large-uncased": 340,
    "roberta-base": 125,
    "roberta-large": 355,
    "deberta-v3-base": 184,
    "deberta-v3-large": 434,
    "finbert": 110,
    "alephbert": 110,
    # Decoder-only
    "gpt2": 124,
    "gpt2-medium": 355,
    "gpt2-large": 774,
    "gpt2-xl": 1500,
    "llama-7b": 7000,
    "llama-13b": 13000,
    "mistral-7b": 7000,
    # Encoder-decoder
    "t5-small": 60,
    "t5-base": 220,
    "t5-large": 770,
    "flan-t5-base": 250,
    "flan-t5-large": 780,
}

# Model architecture mapping
MODEL_ARCHITECTURES = {
    "bert": ModelArchitecture.ENCODER_ONLY,
    "roberta": ModelArchitecture.ENCODER_ONLY,
    "deberta": ModelArchitecture.ENCODER_ONLY,
    "finbert": ModelArchitecture.ENCODER_ONLY,
    "electra": ModelArchitecture.ENCODER_ONLY,
    "gpt2": ModelArchitecture.DECODER_ONLY,
    "gpt": ModelArchitecture.DECODER_ONLY,
    "llama": ModelArchitecture.DECODER_ONLY,
    "mistral": ModelArchitecture.DECODER_ONLY,
    "t5": ModelArchitecture.ENCODER_DECODER,
    "flan": ModelArchitecture.ENCODER_DECODER,
    "bart": ModelArchitecture.ENCODER_DECODER,
}


@dataclass
class ModelConfig:
    """Configuration for the base model."""

    # Model identification
    model_name_or_path: str = "roberta-base"
    tokenizer_name: Optional[str] = None  # If different from model

    # Architecture
    architecture: Optional[ModelArchitecture] = None
    objective: PretrainingObjective = PretrainingObjective.MLM

    # Model modifications
    resize_embeddings: bool = False
    add_special_tokens: Optional[List[str]] = None

    # LoRA/PEFT settings (for large models)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Quantization
    use_4bit: bool = False  # QLoRA
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"

    def __post_init__(self):
        """Infer architecture from model name if not specified."""
        if self.architecture is None:
            model_lower = self.model_name_or_path.lower()
            for key, arch in MODEL_ARCHITECTURES.items():
                if key in model_lower:
                    self.architecture = arch
                    break
            if self.architecture is None:
                self.architecture = ModelArchitecture.ENCODER_ONLY

        # Set appropriate objective based on architecture
        if self.architecture == ModelArchitecture.DECODER_ONLY:
            self.objective = PretrainingObjective.CLM
        elif self.architecture == ModelArchitecture.ENCODER_DECODER:
            self.objective = PretrainingObjective.SPAN_CORRUPTION

        # Default LoRA target modules
        if self.use_lora and self.lora_target_modules is None:
            if "llama" in self.model_name_or_path.lower():
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "gpt" in self.model_name_or_path.lower():
                self.lora_target_modules = ["c_attn", "c_proj"]
            else:
                self.lora_target_modules = ["query", "value"]

    @property
    def param_count_millions(self) -> float:
        """Get approximate parameter count in millions."""
        model_lower = self.model_name_or_path.lower()
        for name, params in MODEL_PARAMS.items():
            if name in model_lower:
                return params
        # Default estimate based on architecture
        if self.architecture == ModelArchitecture.ENCODER_ONLY:
            return 125  # RoBERTa-base size
        elif self.architecture == ModelArchitecture.DECODER_ONLY:
            return 355  # GPT2-medium size
        else:
            return 220  # T5-base size


@dataclass
class DataConfig:
    """Configuration for training data."""

    # Data paths
    train_file: Optional[Path] = None
    validation_file: Optional[Path] = None
    corpus_dir: Optional[Path] = None

    # Text field
    text_column: str = "text"
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4

    # MLM specific
    mlm_probability: float = 0.15
    whole_word_masking: bool = True

    # CLM specific
    block_size: Optional[int] = None  # For concatenating texts

    # Span corruption (T5)
    mean_noise_span_length: float = 3.0
    noise_density: float = 0.15

    # Span MLM (for TAPT)
    use_span_mlm: bool = False  # Use span masking instead of random token masking
    max_span_length: int = 10  # Maximum span length for span MLM

    # Data mixing
    shuffle_buffer_size: int = 10000
    streaming: bool = False  # For large datasets

    def __post_init__(self):
        """Convert paths."""
        if self.train_file:
            self.train_file = Path(self.train_file)
        if self.validation_file:
            self.validation_file = Path(self.validation_file)
        if self.corpus_dir:
            self.corpus_dir = Path(self.corpus_dir)

        if self.block_size is None:
            self.block_size = self.max_seq_length


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./models/pretrained"))
    run_name: Optional[str] = None

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # Override epochs if set

    # Batch sizes
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4

    # Learning rate
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    warmup_steps: int = 0  # Override ratio if set
    lr_scheduler_type: str = "linear"

    # Optimizer
    optim: str = "adamw_torch"  # Or "adamw_8bit" for memory efficiency
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Precision
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True  # For Ampere+ GPUs

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500

    # Logging
    logging_dir: Optional[Path] = None
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Distributed training
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Memory optimization
    gradient_checkpointing: bool = False
    deepspeed: Optional[str] = None  # Path to DeepSpeed config

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Convert paths."""
        self.output_dir = Path(self.output_dir)
        if self.logging_dir:
            self.logging_dir = Path(self.logging_dir)
        else:
            self.logging_dir = self.output_dir / "logs"

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return (
            self.per_device_train_batch_size *
            self.gradient_accumulation_steps
        )


@dataclass
class HardwareProfile:
    """Hardware requirements profile."""

    # GPU requirements
    min_gpu_memory_gb: float
    recommended_gpu_memory_gb: float
    num_gpus: int = 1

    # System memory
    min_ram_gb: float
    recommended_ram_gb: float

    # Storage
    corpus_size_gb: float
    model_checkpoint_size_gb: float
    total_storage_gb: float

    # Time estimates (hours)
    estimated_training_hours: float
    estimated_training_hours_range: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # Cost estimates (USD)
    estimated_cloud_cost_usd: Optional[float] = None

    # Recommendations
    recommended_instance_type: str = ""
    alternative_configs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu": {
                "min_memory_gb": self.min_gpu_memory_gb,
                "recommended_memory_gb": self.recommended_gpu_memory_gb,
                "num_gpus": self.num_gpus,
            },
            "ram": {
                "min_gb": self.min_ram_gb,
                "recommended_gb": self.recommended_ram_gb,
            },
            "storage": {
                "corpus_size_gb": self.corpus_size_gb,
                "model_checkpoint_size_gb": self.model_checkpoint_size_gb,
                "total_gb": self.total_storage_gb,
            },
            "time": {
                "estimated_hours": self.estimated_training_hours,
                "range_hours": self.estimated_training_hours_range,
            },
            "cost": {
                "estimated_usd": self.estimated_cloud_cost_usd,
            },
            "recommendations": {
                "instance_type": self.recommended_instance_type,
                "alternatives": self.alternative_configs,
                "notes": self.notes,
            },
        }


@dataclass
class PretrainingConfig:
    """Complete pretraining configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Pretraining stage
    stage: str = "dapt"  # "dapt" or "tapt"
    continue_from_checkpoint: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": {
                "name": self.model.model_name_or_path,
                "architecture": self.model.architecture.value if self.model.architecture else None,
                "objective": self.model.objective.value,
                "use_lora": self.model.use_lora,
                "param_count_m": self.model.param_count_millions,
            },
            "data": {
                "max_seq_length": self.data.max_seq_length,
                "mlm_probability": self.data.mlm_probability,
            },
            "training": {
                "epochs": self.training.num_train_epochs,
                "batch_size": self.training.effective_batch_size,
                "learning_rate": self.training.learning_rate,
                "fp16": self.training.fp16,
                "bf16": self.training.bf16,
            },
            "stage": self.stage,
        }

    @classmethod
    def for_dapt(
        cls,
        model_name: str = "roberta-base",
        corpus_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "PretrainingConfig":
        """Create configuration for DAPT stage."""
        return cls(
            model=ModelConfig(model_name_or_path=model_name),
            data=DataConfig(
                train_file=corpus_path,
                mlm_probability=0.15,
            ),
            training=TrainingConfig(
                output_dir=output_dir or Path("./models/dapt"),
                num_train_epochs=3,
                learning_rate=5e-5,
            ),
            stage="dapt",
        )

    @classmethod
    def for_tapt(
        cls,
        base_model: str,  # DAPT checkpoint or original model
        corpus_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        use_span_mlm: bool = True,  # Use span masking by default for TAPT
        mean_span_length: float = 3.0,  # Geometric(μ=3) for span lengths
        early_stopping_patience: int = 3,  # Early stopping patience
    ) -> "PretrainingConfig":
        """Create configuration for TAPT stage.

        TAPT (Task-Adaptive Pre-Training) on small corpora requires:
        - Lower learning rate (1e-5) to avoid catastrophic forgetting
        - Higher weight decay (0.05) for regularization
        - Span masking to make the task harder and reduce overfitting
        - Early stopping to prevent overfitting

        Args:
            base_model: DAPT checkpoint or original model name/path.
            corpus_path: Path to TAPT corpus.
            output_dir: Output directory for checkpoints.
            use_span_mlm: Use span masking instead of random token masking.
            mean_span_length: Mean span length for Geometric distribution.
            early_stopping_patience: Number of eval steps without improvement.

        Returns:
            PretrainingConfig optimized for TAPT.
        """
        return cls(
            model=ModelConfig(model_name_or_path=base_model),
            data=DataConfig(
                train_file=corpus_path,
                mlm_probability=0.15,
                use_span_mlm=use_span_mlm,
                mean_noise_span_length=mean_span_length,
            ),
            training=TrainingConfig(
                output_dir=output_dir or Path("./models/tapt"),
                num_train_epochs=10,  # More epochs for smaller TAPT corpus (early stopping will halt)
                learning_rate=1e-5,  # Lower LR to reduce overfitting (was 2e-5)
                weight_decay=0.05,  # Higher weight decay for regularization (was 0.01)
                warmup_ratio=0.1,  # Longer warmup for stability
                eval_strategy="steps",
                eval_steps=100,  # Frequent evaluation for early stopping
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,  # Load best model after early stopping
            ),
            stage="tapt",
        )


def estimate_corpus_size(
    num_samples: int,
    avg_tokens_per_sample: int = 256,
    bytes_per_token: float = 4.5,  # Average for English text
) -> float:
    """Estimate corpus size in GB.

    Args:
        num_samples: Number of text samples.
        avg_tokens_per_sample: Average tokens per sample.
        bytes_per_token: Average bytes per token.

    Returns:
        Estimated size in GB.
    """
    total_bytes = num_samples * avg_tokens_per_sample * bytes_per_token
    return total_bytes / (1024 ** 3)


def estimate_training_tokens(
    num_samples: int,
    avg_tokens_per_sample: int = 256,
    num_epochs: int = 3,
) -> int:
    """Estimate total training tokens.

    Args:
        num_samples: Number of text samples.
        avg_tokens_per_sample: Average tokens per sample.
        num_epochs: Number of training epochs.

    Returns:
        Total training tokens.
    """
    return num_samples * avg_tokens_per_sample * num_epochs


def estimate_hardware_requirements(
    config: PretrainingConfig,
    corpus_samples: int,
    avg_tokens_per_sample: int = 256,
) -> HardwareProfile:
    """Estimate hardware requirements for pretraining.

    Based on:
    - Model size and architecture
    - Corpus size
    - Training configuration (batch size, precision, etc.)

    Args:
        config: Pretraining configuration.
        corpus_samples: Number of samples in corpus.
        avg_tokens_per_sample: Average tokens per sample.

    Returns:
        HardwareProfile with requirements and recommendations.
    """
    # Model parameters
    model_params_m = config.model.param_count_millions
    model_params_b = model_params_m / 1000

    # Corpus size
    corpus_size_gb = estimate_corpus_size(corpus_samples, avg_tokens_per_sample)

    # Total training tokens
    total_tokens = estimate_training_tokens(
        corpus_samples,
        avg_tokens_per_sample,
        config.training.num_train_epochs,
    )
    total_tokens_b = total_tokens / 1e9

    # GPU memory estimation
    # Rule of thumb: Full precision training needs ~4x model size
    # FP16/BF16: ~2x model size
    # With gradient checkpointing: ~0.5x reduction
    # With LoRA: Only ~10-20% of full fine-tuning

    bytes_per_param = 4.0  # FP32
    if config.training.fp16 or config.training.bf16:
        bytes_per_param = 2.0
    if config.model.use_4bit:
        bytes_per_param = 0.5
    elif config.model.use_8bit:
        bytes_per_param = 1.0

    # Base memory for model weights
    model_memory_gb = (model_params_m * 1e6 * bytes_per_param) / (1024 ** 3)

    # Optimizer states (Adam: 2x model size in FP32)
    optimizer_memory_gb = model_memory_gb * 2 if not config.model.use_lora else model_memory_gb * 0.2

    # Gradients
    gradient_memory_gb = model_memory_gb

    # Activations (rough estimate based on batch size and seq length)
    batch_size = config.training.per_device_train_batch_size
    seq_length = config.data.max_seq_length
    hidden_size = 768 if model_params_m < 200 else 1024  # Approximate
    num_layers = 12 if model_params_m < 200 else 24

    activation_memory_gb = (
        batch_size * seq_length * hidden_size * num_layers * 4
    ) / (1024 ** 3)

    if config.training.gradient_checkpointing:
        activation_memory_gb *= 0.3  # ~70% reduction

    if config.model.use_lora:
        # LoRA significantly reduces gradient and optimizer memory
        optimizer_memory_gb *= 0.1
        gradient_memory_gb *= 0.1

    total_gpu_memory_gb = (
        model_memory_gb +
        optimizer_memory_gb +
        gradient_memory_gb +
        activation_memory_gb
    )

    # Add 20% buffer
    min_gpu_memory_gb = total_gpu_memory_gb * 1.2
    recommended_gpu_memory_gb = total_gpu_memory_gb * 1.5

    # RAM requirements (for data loading)
    min_ram_gb = max(16, corpus_size_gb * 2)
    recommended_ram_gb = max(32, corpus_size_gb * 4)

    # Storage requirements
    model_checkpoint_gb = model_memory_gb * 3  # Multiple checkpoints
    total_storage_gb = corpus_size_gb + model_checkpoint_gb + 20  # Buffer

    # Training time estimation
    # Based on typical throughput: ~3000-5000 tokens/second on A100
    # Adjust based on model size
    tokens_per_second = 4000 / math.sqrt(model_params_b + 1)
    if config.training.fp16 or config.training.bf16:
        tokens_per_second *= 1.5
    if config.model.use_lora:
        tokens_per_second *= 1.2

    training_seconds = total_tokens / tokens_per_second
    training_hours = training_seconds / 3600

    # Range based on hardware variation
    training_hours_min = training_hours * 0.7
    training_hours_max = training_hours * 1.5

    # Cloud cost estimation (A100 ~$3/hr, V100 ~$2/hr)
    hourly_rate = 3.0 if recommended_gpu_memory_gb > 24 else 2.0
    estimated_cost = training_hours * hourly_rate * 1.2  # Buffer

    # Recommendations
    notes = []
    alternatives = []

    if min_gpu_memory_gb > 80:
        notes.append("Consider using DeepSpeed ZeRO-3 for model parallelism")
        notes.append("Multi-GPU setup required")
        recommended_instance = "8x A100 80GB or equivalent"
        alternatives = [
            "AWS: p4d.24xlarge (8x A100 40GB)",
            "GCP: a2-megagpu-16g (16x A100 40GB)",
            "Azure: ND96asr_v4 (8x A100 80GB)",
        ]
    elif min_gpu_memory_gb > 40:
        notes.append("Single A100 80GB or 2x A100 40GB recommended")
        recommended_instance = "A100 80GB"
        alternatives = [
            "AWS: p4d.24xlarge (8x A100 40GB) - use 2 GPUs",
            "GCP: a2-highgpu-2g (2x A100 40GB)",
            "Lambda Labs: 1x A100 80GB (~$1.29/hr)",
        ]
    elif min_gpu_memory_gb > 24:
        notes.append("A100 40GB or RTX 4090 recommended")
        recommended_instance = "A100 40GB"
        alternatives = [
            "AWS: p4d.24xlarge (use 1 GPU)",
            "GCP: a2-highgpu-1g",
            "Consumer: RTX 4090 24GB",
            "Colab Pro+: A100 40GB",
        ]
    elif min_gpu_memory_gb > 16:
        notes.append("V100 32GB or RTX 3090/4090 suitable")
        recommended_instance = "V100 32GB or RTX 4090"
        alternatives = [
            "AWS: p3.2xlarge (V100 16GB) with gradient accumulation",
            "GCP: n1-highmem-8 + V100",
            "Consumer: RTX 3090 24GB, RTX 4090 24GB",
        ]
    elif min_gpu_memory_gb > 8:
        notes.append("T4 16GB or RTX 3080 suitable with FP16")
        recommended_instance = "T4 16GB"
        alternatives = [
            "AWS: g4dn.xlarge (T4 16GB)",
            "GCP: n1-standard-4 + T4",
            "Colab Free/Pro: T4 16GB",
            "Consumer: RTX 3080 10GB with gradient checkpointing",
        ]
    else:
        notes.append("Most modern GPUs should work")
        recommended_instance = "Any GPU with 8GB+ VRAM"
        alternatives = [
            "Colab Free: T4 16GB",
            "Consumer: RTX 3060 12GB, RTX 3070 8GB",
        ]

    # Add precision recommendations
    if not (config.training.fp16 or config.training.bf16):
        notes.append("Enable FP16/BF16 to reduce memory by ~50%")

    if not config.training.gradient_checkpointing and min_gpu_memory_gb > 16:
        notes.append("Enable gradient_checkpointing to reduce memory by ~30%")

    if model_params_m > 1000 and not config.model.use_lora:
        notes.append("Consider LoRA/QLoRA for large models to reduce memory by ~90%")

    return HardwareProfile(
        min_gpu_memory_gb=round(min_gpu_memory_gb, 1),
        recommended_gpu_memory_gb=round(recommended_gpu_memory_gb, 1),
        num_gpus=max(1, int(math.ceil(min_gpu_memory_gb / 80))),
        min_ram_gb=round(min_ram_gb, 0),
        recommended_ram_gb=round(recommended_ram_gb, 0),
        corpus_size_gb=round(corpus_size_gb, 2),
        model_checkpoint_size_gb=round(model_checkpoint_gb, 1),
        total_storage_gb=round(total_storage_gb, 0),
        estimated_training_hours=round(training_hours, 1),
        estimated_training_hours_range=(round(training_hours_min, 1), round(training_hours_max, 1)),
        estimated_cloud_cost_usd=round(estimated_cost, 0),
        recommended_instance_type=recommended_instance,
        alternative_configs=alternatives,
        notes=notes,
    )


# Preset configurations for common scenarios
PRESET_CONFIGS = {
    "roberta-base-dapt": PretrainingConfig.for_dapt("roberta-base"),
    "roberta-large-dapt": PretrainingConfig.for_dapt("roberta-large"),
    "deberta-v3-base-dapt": PretrainingConfig.for_dapt("microsoft/deberta-v3-base"),
    "finbert-dapt": PretrainingConfig.for_dapt("ProsusAI/finbert"),
    "llama-7b-lora": PretrainingConfig(
        model=ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            use_lora=True,
            use_4bit=True,
        ),
        training=TrainingConfig(
            gradient_checkpointing=True,
            fp16=True,
        ),
    ),
}
