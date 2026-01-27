# Pretraining

## Overview
Continuous pretraining utilities for AML-domain language models. The modules
support task-adaptive pretraining, span corruption, and supervised heads that
feed the learned judge and other downstream components.

## Key modules
- `config.py`: Dataclasses for model, data, and training configuration, plus
  hardware sizing heuristics.
- `trainer.py`: AML-specific wrapper around HuggingFace `Trainer` with custom
  callbacks, metrics, and early stopping.
- `collators.py`: Dataset and data-collator implementations for MLM, span-MLM,
  CLM, and span corruption objectives.
- `supervised_heads.py`: Supervised heads for classification/regression on top
  of pretrained encoders.
- `ranking_loss.py`: Pairwise/listwise ranking losses for judge tuning.
- `judge_finetuning.py`: Fine-tuning pipeline for the learned judge model.
- `pretokenize.py`: Tokenization utilities for preprocessing large corpora.
- `scripts.py`: Scriptable helpers for batch runs and experiment automation.
- `HARDWARE_REQUIREMENTS.md`: Guidance for hardware sizing.

## Usage notes
- The trainer is compatible with LoRA/PEFT and optional 4-bit/8-bit
  quantization.
- Use `PretrainingConfig` to define objectives (MLM/CLM/span) and data paths
  before launching training.
