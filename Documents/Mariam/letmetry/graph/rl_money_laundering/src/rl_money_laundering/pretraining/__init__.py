"""
Continuous Pretraining module for AML domain adaptation.

This module provides tools for domain-adaptive and task-adaptive pretraining
of language models on AML-specific text corpora.

Pretraining Stages:
1. DAPT (Domain-Adaptive Pretraining): Broad financial/legal text
2. TAPT (Task-Adaptive Pretraining): AML-specific documents (SARs, typologies)

Supported Models:
- Encoder-only: BERT, RoBERTa, DeBERTa, FinBERT
- Encoder-decoder: T5, FLAN-T5
- Decoder-only: GPT-2, Llama (with LoRA/QLoRA)

Usage:
    from rl_money_laundering.pretraining import (
        PretrainingConfig,
        ContinuousPretrainer,
        run_dapt,
        run_tapt,
    )

    # Quick DAPT
    run_dapt(
        corpus_path="./data/processed/aml_corpus.jsonl",
        base_model="roberta-base",
        output_dir="./models/aml-roberta-dapt",
    )
"""

from .config import (
    PretrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    HardwareProfile,
    estimate_hardware_requirements,
)
from .collators import (
    MLMDataCollator,
    SpanMLMCollator,
    CLMDataCollator,
    SpanCorruptionCollator,
    AMLTextDataset,
    StreamingAMLDataset,
    create_data_collator,
)
from .trainer import (
    ContinuousPretrainer,
    PretrainingCallback,
    EarlyStoppingCallback,
)
from .scripts import (
    run_dapt,
    run_tapt,
    run_full_pretraining,
)
from .pretokenize import (
    PreTokenizeConfig,
    PreTokenizeStats,
    PreTokenizer,
    PreTokenizedDataset,
    PackedDataset,
    StreamingPreTokenizedDataset,
    pretokenize_corpus,
)
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
    MarginRankingLoss,
    ContrastiveLoss,
    TripletMarginLoss,
    ListNetLoss,
    LambdaRankLoss,
    create_ranking_loss,
    compute_ndcg,
    compute_map,
)
from .judge_finetuning import (
    JudgeTask,
    JudgeFinetuneConfig,
    JudgeFinetuneResult,
    JudgeFinetuner,
    run_judge_finetuning,
    load_judge_model,
)

__all__ = [
    # Config
    "PretrainingConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "HardwareProfile",
    "estimate_hardware_requirements",
    # Data collators
    "MLMDataCollator",
    "SpanMLMCollator",
    "CLMDataCollator",
    "SpanCorruptionCollator",
    "AMLTextDataset",
    "StreamingAMLDataset",
    "create_data_collator",
    # Training
    "ContinuousPretrainer",
    "PretrainingCallback",
    "EarlyStoppingCallback",
    # Scripts
    "run_dapt",
    "run_tapt",
    "run_full_pretraining",
    # Pre-tokenization
    "PreTokenizeConfig",
    "PreTokenizeStats",
    "PreTokenizer",
    "PreTokenizedDataset",
    "PackedDataset",
    "StreamingPreTokenizedDataset",
    "pretokenize_corpus",
    # Supervised heads
    "HeadConfig",
    "PoolingStrategy",
    "TypologyClassificationHead",
    "SuspiciousnessScorer",
    "EvidenceSelector",
    "PairwiseRanker",
    "create_supervised_head",
    "AML_TYPOLOGIES",
    # Ranking losses
    "MarginRankingLoss",
    "ContrastiveLoss",
    "TripletMarginLoss",
    "ListNetLoss",
    "LambdaRankLoss",
    "create_ranking_loss",
    "compute_ndcg",
    "compute_map",
    # Judge fine-tuning
    "JudgeTask",
    "JudgeFinetuneConfig",
    "JudgeFinetuneResult",
    "JudgeFinetuner",
    "run_judge_finetuning",
    "load_judge_model",
]
