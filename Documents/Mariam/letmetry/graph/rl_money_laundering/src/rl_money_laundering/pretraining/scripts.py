"""
Pretraining scripts for DAPT and TAPT stages.

Provides convenient functions and CLI entry points for running
domain-adaptive and task-adaptive pretraining.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import (
    PretrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    HardwareProfile,
    estimate_hardware_requirements,
    estimate_corpus_size,
)
from .trainer import ContinuousPretrainer, create_pretrainer

logger = logging.getLogger(__name__)


def run_dapt(
    corpus_path: Union[str, Path],
    base_model: str = "roberta-base",
    output_dir: Optional[Union[str, Path]] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_seq_length: int = 512,
    fp16: bool = True,
    gradient_checkpointing: bool = False,
    resume_from: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run Domain-Adaptive Pretraining (DAPT).

    DAPT continues pretraining on domain-specific text (financial/legal)
    before task-specific fine-tuning. This stage uses a large, diverse
    corpus to teach the model domain vocabulary and patterns.

    Args:
        corpus_path: Path to training corpus (JSONL file).
        base_model: Base model to continue from.
        output_dir: Output directory for checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Initial learning rate.
        max_seq_length: Maximum sequence length.
        fp16: Use FP16 mixed precision.
        gradient_checkpointing: Enable gradient checkpointing.
        resume_from: Path to checkpoint to resume from.
        **kwargs: Additional configuration options.

    Returns:
        Training metrics dictionary.

    Example:
        ```python
        from rl_money_laundering.pretraining import run_dapt

        metrics = run_dapt(
            corpus_path="./data/processed/aml_corpus.jsonl",
            base_model="roberta-base",
            output_dir="./models/aml-roberta-dapt",
            num_epochs=3,
            fp16=True,
        )
        print(f"Final loss: {metrics['train_loss']:.4f}")
        ```
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir) if output_dir else Path(f"./models/{base_model.split('/')[-1]}-dapt")

    logger.info(f"Starting DAPT from {base_model}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {output_dir}")

    # Create configuration
    config = PretrainingConfig(
        model=ModelConfig(
            model_name_or_path=base_model,
        ),
        data=DataConfig(
            train_file=corpus_path,
            max_seq_length=max_seq_length,
            mlm_probability=0.15,
            whole_word_masking=True,
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            run_name=f"dapt_{base_model.split('/')[-1]}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=0.06,
            save_steps=1000,
            logging_steps=100,
        ),
        stage="dapt",
    )

    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)

    # Run pretraining
    pretrainer = ContinuousPretrainer(config)
    metrics = pretrainer.run()

    logger.info(f"DAPT complete. Model saved to {output_dir}")
    return metrics


def run_tapt(
    corpus_path: Union[str, Path],
    base_model: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 512,
    fp16: bool = True,
    gradient_checkpointing: bool = False,
    resume_from: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run Task-Adaptive Pretraining (TAPT).

    TAPT continues pretraining on task-specific text (SARs, typologies)
    after DAPT. This uses a smaller, highly relevant corpus to specialize
    the model for AML detection.

    Args:
        corpus_path: Path to training corpus (JSONL file).
        base_model: DAPT checkpoint or base model to continue from.
        output_dir: Output directory for checkpoints.
        num_epochs: Number of training epochs (typically more than DAPT).
        batch_size: Per-device batch size.
        learning_rate: Initial learning rate (typically lower than DAPT).
        max_seq_length: Maximum sequence length.
        fp16: Use FP16 mixed precision.
        gradient_checkpointing: Enable gradient checkpointing.
        resume_from: Path to checkpoint to resume from.
        **kwargs: Additional configuration options.

    Returns:
        Training metrics dictionary.

    Example:
        ```python
        from rl_money_laundering.pretraining import run_tapt

        # Start from DAPT checkpoint
        metrics = run_tapt(
            corpus_path="./data/processed/aml_sar_corpus.jsonl",
            base_model="./models/aml-roberta-dapt",
            output_dir="./models/aml-roberta-tapt",
            num_epochs=5,
            learning_rate=2e-5,
        )
        ```
    """
    corpus_path = Path(corpus_path)
    base_model_str = str(base_model)
    output_dir = Path(output_dir) if output_dir else Path(f"{base_model_str}-tapt")

    logger.info(f"Starting TAPT from {base_model}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {output_dir}")

    # Create configuration
    config = PretrainingConfig(
        model=ModelConfig(
            model_name_or_path=base_model_str,
        ),
        data=DataConfig(
            train_file=corpus_path,
            max_seq_length=max_seq_length,
            mlm_probability=0.15,
            whole_word_masking=True,
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            run_name=f"tapt_{Path(base_model_str).name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=0.1,  # Higher warmup for smaller corpus
            save_steps=500,
            logging_steps=50,
        ),
        stage="tapt",
    )

    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)

    # Run pretraining
    pretrainer = ContinuousPretrainer(config)
    metrics = pretrainer.run()

    logger.info(f"TAPT complete. Model saved to {output_dir}")
    return metrics


def run_full_pretraining(
    dapt_corpus: Union[str, Path],
    tapt_corpus: Union[str, Path],
    base_model: str = "roberta-base",
    output_dir: Optional[Union[str, Path]] = None,
    dapt_epochs: int = 3,
    tapt_epochs: int = 5,
    batch_size: int = 8,
    fp16: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run complete two-stage pretraining (DAPT + TAPT).

    Executes both domain-adaptive and task-adaptive pretraining in sequence.
    The DAPT stage uses a larger corpus of general financial/legal text,
    while TAPT uses a smaller, more specific AML corpus.

    Args:
        dapt_corpus: Path to DAPT corpus (broad domain text).
        tapt_corpus: Path to TAPT corpus (AML-specific text).
        base_model: Initial base model.
        output_dir: Base output directory.
        dapt_epochs: Epochs for DAPT stage.
        tapt_epochs: Epochs for TAPT stage.
        batch_size: Per-device batch size.
        fp16: Use FP16 mixed precision.
        **kwargs: Additional configuration options.

    Returns:
        Combined metrics from both stages.

    Example:
        ```python
        from rl_money_laundering.pretraining import run_full_pretraining

        metrics = run_full_pretraining(
            dapt_corpus="./data/processed/financial_corpus.jsonl",
            tapt_corpus="./data/processed/aml_sar_corpus.jsonl",
            base_model="roberta-base",
            output_dir="./models/aml-roberta",
        )
        ```
    """
    output_dir = Path(output_dir) if output_dir else Path(f"./models/{base_model.split('/')[-1]}-aml")

    dapt_output = output_dir / "dapt"
    tapt_output = output_dir / "tapt"

    logger.info("=" * 60)
    logger.info("STAGE 1: Domain-Adaptive Pretraining (DAPT)")
    logger.info("=" * 60)

    dapt_metrics = run_dapt(
        corpus_path=dapt_corpus,
        base_model=base_model,
        output_dir=dapt_output,
        num_epochs=dapt_epochs,
        batch_size=batch_size,
        fp16=fp16,
        **kwargs,
    )

    logger.info("=" * 60)
    logger.info("STAGE 2: Task-Adaptive Pretraining (TAPT)")
    logger.info("=" * 60)

    tapt_metrics = run_tapt(
        corpus_path=tapt_corpus,
        base_model=dapt_output,
        output_dir=tapt_output,
        num_epochs=tapt_epochs,
        batch_size=batch_size,
        fp16=fp16,
        learning_rate=2e-5,  # Lower LR for TAPT
        **kwargs,
    )

    # Combine metrics
    combined_metrics = {
        "dapt": dapt_metrics,
        "tapt": tapt_metrics,
        "final_model": str(tapt_output),
    }

    # Save combined metrics
    metrics_path = output_dir / "full_pretraining_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Full pretraining complete. Final model: {tapt_output}")
    logger.info("=" * 60)

    return combined_metrics


def estimate_requirements(
    corpus_path: Union[str, Path],
    model_name: str = "roberta-base",
    num_epochs: int = 3,
    batch_size: int = 8,
    fp16: bool = True,
) -> HardwareProfile:
    """Estimate hardware requirements for pretraining.

    Args:
        corpus_path: Path to corpus file.
        model_name: Model to pretrain.
        num_epochs: Number of epochs.
        batch_size: Per-device batch size.
        fp16: Whether FP16 will be used.

    Returns:
        HardwareProfile with requirements.
    """
    corpus_path = Path(corpus_path)

    # Count samples
    num_samples = 0
    total_tokens = 0

    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    num_samples += 1
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        # Rough token count (words * 1.3)
                        total_tokens += len(text.split()) * 1.3
                    except json.JSONDecodeError:
                        pass

    avg_tokens = int(total_tokens / num_samples) if num_samples > 0 else 256

    # Create config
    config = PretrainingConfig(
        model=ModelConfig(model_name_or_path=model_name),
        training=TrainingConfig(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            fp16=fp16,
        ),
    )

    profile = estimate_hardware_requirements(
        config=config,
        corpus_samples=num_samples,
        avg_tokens_per_sample=avg_tokens,
    )

    return profile


def print_requirements(profile: HardwareProfile) -> None:
    """Print hardware requirements in readable format."""
    print("\n" + "=" * 60)
    print("HARDWARE REQUIREMENTS")
    print("=" * 60)

    print("\nGPU Memory:")
    print(f"  Minimum:      {profile.min_gpu_memory_gb:.1f} GB")
    print(f"  Recommended:  {profile.recommended_gpu_memory_gb:.1f} GB")
    print(f"  GPUs needed:  {profile.num_gpus}")

    print("\nSystem RAM:")
    print(f"  Minimum:      {profile.min_ram_gb:.0f} GB")
    print(f"  Recommended:  {profile.recommended_ram_gb:.0f} GB")

    print("\nStorage:")
    print(f"  Corpus size:  {profile.corpus_size_gb:.2f} GB")
    print(f"  Checkpoints:  {profile.model_checkpoint_size_gb:.1f} GB")
    print(f"  Total needed: {profile.total_storage_gb:.0f} GB")

    print("\nTime Estimate:")
    print(f"  Expected:     {profile.estimated_training_hours:.1f} hours")
    print(f"  Range:        {profile.estimated_training_hours_range[0]:.1f} - {profile.estimated_training_hours_range[1]:.1f} hours")

    if profile.estimated_cloud_cost_usd:
        print(f"\nEstimated Cloud Cost: ${profile.estimated_cloud_cost_usd:.0f} USD")

    print(f"\nRecommended Instance: {profile.recommended_instance_type}")

    if profile.alternative_configs:
        print("\nAlternative Configurations:")
        for alt in profile.alternative_configs:
            print(f"  - {alt}")

    if profile.notes:
        print("\nNotes:")
        for note in profile.notes:
            print(f"  - {note}")

    print("=" * 60 + "\n")


# CLI entry point
def main():
    """Command-line interface for pretraining."""
    parser = argparse.ArgumentParser(
        description="AML Continuous Pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DAPT
  python -m rl_money_laundering.pretraining.scripts dapt \\
      --corpus ./data/aml_corpus.jsonl \\
      --model roberta-base \\
      --output ./models/aml-roberta-dapt

  # Run TAPT on DAPT checkpoint
  python -m rl_money_laundering.pretraining.scripts tapt \\
      --corpus ./data/sar_corpus.jsonl \\
      --model ./models/aml-roberta-dapt \\
      --output ./models/aml-roberta-tapt

  # Estimate hardware requirements
  python -m rl_money_laundering.pretraining.scripts estimate \\
      --corpus ./data/aml_corpus.jsonl \\
      --model roberta-base
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # DAPT command
    dapt_parser = subparsers.add_parser("dapt", help="Run Domain-Adaptive Pretraining")
    dapt_parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus")
    dapt_parser.add_argument("--model", type=str, default="roberta-base", help="Base model")
    dapt_parser.add_argument("--output", type=Path, help="Output directory")
    dapt_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dapt_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    dapt_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    dapt_parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    dapt_parser.add_argument("--gradient-checkpointing", action="store_true")
    dapt_parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # TAPT command
    tapt_parser = subparsers.add_parser("tapt", help="Run Task-Adaptive Pretraining")
    tapt_parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus")
    tapt_parser.add_argument("--model", type=str, required=True, help="Base model or DAPT checkpoint")
    tapt_parser.add_argument("--output", type=Path, help="Output directory")
    tapt_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    tapt_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    tapt_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    tapt_parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    tapt_parser.add_argument("--gradient-checkpointing", action="store_true")
    tapt_parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Full pretraining command
    full_parser = subparsers.add_parser("full", help="Run full DAPT + TAPT")
    full_parser.add_argument("--dapt-corpus", type=Path, required=True, help="DAPT corpus")
    full_parser.add_argument("--tapt-corpus", type=Path, required=True, help="TAPT corpus")
    full_parser.add_argument("--model", type=str, default="roberta-base", help="Base model")
    full_parser.add_argument("--output", type=Path, help="Output directory")
    full_parser.add_argument("--dapt-epochs", type=int, default=3)
    full_parser.add_argument("--tapt-epochs", type=int, default=5)
    full_parser.add_argument("--batch-size", type=int, default=8)
    full_parser.add_argument("--no-fp16", action="store_true")

    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate hardware requirements")
    est_parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus")
    est_parser.add_argument("--model", type=str, default="roberta-base", help="Model to use")
    est_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    est_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    est_parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "dapt":
        run_dapt(
            corpus_path=args.corpus,
            base_model=args.model,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=not args.no_fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            resume_from=args.resume,
        )

    elif args.command == "tapt":
        run_tapt(
            corpus_path=args.corpus,
            base_model=args.model,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=not args.no_fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            resume_from=args.resume,
        )

    elif args.command == "full":
        run_full_pretraining(
            dapt_corpus=args.dapt_corpus,
            tapt_corpus=args.tapt_corpus,
            base_model=args.model,
            output_dir=args.output,
            dapt_epochs=args.dapt_epochs,
            tapt_epochs=args.tapt_epochs,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
        )

    elif args.command == "estimate":
        profile = estimate_requirements(
            corpus_path=args.corpus,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
        )
        print_requirements(profile)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
