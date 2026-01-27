# Hardware Requirements for AML Continuous Pretraining

This document outlines hardware requirements for domain-adaptive (DAPT) and task-adaptive (TAPT) pretraining on AML text corpora.

## Table of Contents
- [Dataset Size Estimates](#dataset-size-estimates)
- [Hardware Tiers](#hardware-tiers)
- [Model-Specific Requirements](#model-specific-requirements)
- [Cloud Provider Recommendations](#cloud-provider-recommendations)
- [Cost Estimates](#cost-estimates)
- [Memory Optimization Techniques](#memory-optimization-techniques)

---

## Dataset Size Estimates

Based on the data sources integrated in the dataprep module:

### DAPT Corpus (Domain-Adaptive)

| Source | Est. Samples | Est. Tokens | Size (GB) |
|--------|-------------|-------------|-----------|
| FinCEN Files (SARs) | ~2,100 | ~1.5M | ~0.01 |
| SEC EDGAR Fraud | ~15,000 | ~50M | ~0.25 |
| OpenSanctions | ~500,000 | ~100M | ~0.5 |
| GDELT News (filtered) | ~100,000 | ~150M | ~0.75 |
| FATF Typologies | ~500 | ~2M | ~0.01 |
| **Total DAPT** | **~618,000** | **~304M** | **~1.5 GB** |

With additional web-scraped financial/legal text (optional):
- Financial news: +500M tokens (~2.5 GB)
- Legal documents: +200M tokens (~1 GB)
- **Extended DAPT Total**: ~1B tokens (~5 GB)

### TAPT Corpus (Task-Adaptive)

| Source | Est. Samples | Est. Tokens | Size (GB) |
|--------|-------------|-------------|-----------|
| FinCEN SARs (high-value) | ~1,000 | ~700K | ~0.004 |
| FATF Case Studies | ~200 | ~400K | ~0.002 |
| FraudNLP Sequences | ~5,000 | ~500K | ~0.003 |
| Venmo Notes (labeled) | ~10,000 | ~100K | ~0.001 |
| **Total TAPT** | **~16,200** | **~1.7M** | **~0.01 GB** |

### Token Counts for Pretraining

| Stage | Tokens | Epochs | Total Training Tokens |
|-------|--------|--------|----------------------|
| DAPT (base) | ~304M | 3 | ~912M |
| DAPT (extended) | ~1B | 3 | ~3B |
| TAPT | ~1.7M | 5 | ~8.5M |

---

## Hardware Tiers

### Tier 1: Minimal (Research/Prototyping)

**Configuration:**
- GPU: NVIDIA T4 16GB or RTX 3060 12GB
- RAM: 16 GB
- Storage: 50 GB SSD

**Suitable for:**
- Small models (BERT-base, RoBERTa-base, ~125M params)
- Batch size: 4-8 with gradient accumulation
- TAPT on small corpus
- Development and debugging

**Estimated Training Time:**
- DAPT (base corpus): 8-12 hours
- TAPT: 30 minutes - 1 hour

**Requirements:**
- FP16 mixed precision: **Required**
- Gradient checkpointing: Recommended
- Gradient accumulation: 4-8 steps

---

### Tier 2: Standard (Production-Ready)

**Configuration:**
- GPU: NVIDIA V100 32GB or RTX 4090 24GB
- RAM: 32 GB
- Storage: 100 GB SSD

**Suitable for:**
- Medium models (RoBERTa-large, DeBERTa-v3, ~350M params)
- Batch size: 8-16
- Full DAPT + TAPT pipeline

**Estimated Training Time:**
- DAPT (base corpus): 4-8 hours
- DAPT (extended): 15-25 hours
- TAPT: 20-40 minutes

**Requirements:**
- FP16/BF16: Recommended
- Gradient checkpointing: Optional

---

### Tier 3: Advanced (Large Models)

**Configuration:**
- GPU: NVIDIA A100 40GB/80GB
- RAM: 64 GB
- Storage: 200 GB SSD

**Suitable for:**
- Large models (GPT2-XL, Llama-7B with LoRA)
- Batch size: 16-32
- Extended corpora

**Estimated Training Time:**
- DAPT (extended): 8-15 hours
- TAPT: 30 minutes - 1 hour

**Requirements:**
- BF16 preferred (Ampere architecture)
- TF32 enabled
- Optional: DeepSpeed ZeRO-2

---

### Tier 4: Enterprise (Very Large Models)

**Configuration:**
- GPU: 4-8x NVIDIA A100 80GB
- RAM: 256 GB+
- Storage: 1 TB NVMe

**Suitable for:**
- Very large models (Llama-13B+, full fine-tuning)
- Multi-billion token corpora
- Production training at scale

**Requirements:**
- DeepSpeed ZeRO-3 or FSDP
- Model parallelism
- NVLink interconnect

---

## Model-Specific Requirements

### Encoder-Only Models (MLM)

| Model | Parameters | Min GPU Memory | Recommended | Batch Size* |
|-------|------------|----------------|-------------|-------------|
| BERT-base | 110M | 8 GB | 12 GB | 16 |
| RoBERTa-base | 125M | 8 GB | 12 GB | 16 |
| FinBERT | 110M | 8 GB | 12 GB | 16 |
| RoBERTa-large | 355M | 16 GB | 24 GB | 8 |
| DeBERTa-v3-base | 184M | 12 GB | 16 GB | 12 |
| DeBERTa-v3-large | 434M | 24 GB | 32 GB | 4 |

*Batch size with FP16 and seq_length=512

### Decoder-Only Models (CLM)

| Model | Parameters | Min GPU Memory | With LoRA | With QLoRA |
|-------|------------|----------------|-----------|------------|
| GPT2 | 124M | 8 GB | 6 GB | 4 GB |
| GPT2-medium | 355M | 16 GB | 10 GB | 6 GB |
| GPT2-large | 774M | 24 GB | 14 GB | 8 GB |
| GPT2-XL | 1.5B | 48 GB | 20 GB | 10 GB |
| Llama-7B | 7B | 140 GB* | 24 GB | 12 GB |
| Llama-13B | 13B | 260 GB* | 40 GB | 20 GB |

*Full fine-tuning; impractical without model parallelism

### Encoder-Decoder Models (Span Corruption)

| Model | Parameters | Min GPU Memory | Recommended |
|-------|------------|----------------|-------------|
| T5-small | 60M | 6 GB | 8 GB |
| T5-base | 220M | 12 GB | 16 GB |
| T5-large | 770M | 32 GB | 48 GB |
| FLAN-T5-base | 250M | 14 GB | 20 GB |

---

## Cloud Provider Recommendations

### AWS

| Use Case | Instance | GPU | Cost/hr | Notes |
|----------|----------|-----|---------|-------|
| Development | g4dn.xlarge | T4 16GB | ~$0.53 | Good for small models |
| Standard | p3.2xlarge | V100 16GB | ~$3.06 | Popular choice |
| Production | p4d.24xlarge | 8x A100 40GB | ~$32.77 | Use 1-2 GPUs for most cases |
| Large Models | p5.48xlarge | 8x H100 | ~$98.32 | For 70B+ models |

**Spot Instances:** Save 60-90% with spot pricing (if available)

### Google Cloud

| Use Case | Instance | GPU | Cost/hr | Notes |
|----------|----------|-----|---------|-------|
| Development | n1-standard-4 + T4 | T4 16GB | ~$0.45 | Cheapest option |
| Standard | n1-highmem-8 + V100 | V100 16GB | ~$2.48 | Good balance |
| Production | a2-highgpu-1g | A100 40GB | ~$3.67 | Single A100 |
| Large | a2-megagpu-16g | 16x A100 40GB | ~$58.72 | For parallel training |

### Azure

| Use Case | Instance | GPU | Cost/hr | Notes |
|----------|----------|-----|---------|-------|
| Development | NC4as_T4_v3 | T4 16GB | ~$0.53 | Entry level |
| Standard | NC6s_v3 | V100 16GB | ~$3.06 | Standard ML |
| Production | ND96asr_v4 | 8x A100 80GB | ~$27.20 | Premium |

### Lambda Labs

| GPU | Cost/hr | Notes |
|-----|---------|-------|
| RTX 3090 24GB | ~$0.50 | Budget option |
| A100 40GB | ~$1.10 | Great value |
| A100 80GB | ~$1.29 | Best for large models |
| H100 80GB | ~$1.99 | Newest generation |

### RunPod / Vast.ai

Community GPU marketplace with variable pricing:
- RTX 3090: $0.20-0.40/hr
- A100 40GB: $0.80-1.50/hr
- A100 80GB: $1.00-2.00/hr

---

## Cost Estimates

### DAPT Training (Base Corpus: ~300M tokens, 3 epochs)

| Hardware Tier | Training Time | Cloud Cost |
|---------------|---------------|------------|
| T4 16GB | 10-15 hours | $5-8 |
| V100 32GB | 5-8 hours | $15-25 |
| A100 40GB | 3-5 hours | $10-18 |
| A100 80GB | 2-4 hours | $8-15 |

### DAPT Training (Extended Corpus: ~1B tokens, 3 epochs)

| Hardware Tier | Training Time | Cloud Cost |
|---------------|---------------|------------|
| V100 32GB | 20-30 hours | $60-90 |
| A100 40GB | 10-15 hours | $35-55 |
| A100 80GB | 8-12 hours | $25-45 |
| 4x A100 80GB | 3-5 hours | $35-60 |

### TAPT Training (~2M tokens, 5 epochs)

| Hardware Tier | Training Time | Cloud Cost |
|---------------|---------------|------------|
| T4 16GB | 30-60 min | $0.50-1 |
| V100 32GB | 15-30 min | $0.75-1.50 |
| A100 40GB | 10-20 min | $0.60-1.20 |

### Full Pipeline (DAPT + TAPT)

| Configuration | Total Time | Total Cost |
|---------------|------------|------------|
| Minimal (T4) | 12-18 hours | $6-10 |
| Standard (V100) | 6-10 hours | $20-30 |
| Recommended (A100) | 4-6 hours | $15-25 |

---

## Memory Optimization Techniques

### 1. Mixed Precision Training (FP16/BF16)

Reduces memory by ~50%, speeds up training by ~1.5x.

```python
training_args = TrainingArguments(
    fp16=True,  # For V100, T4
    bf16=True,  # For A100, H100 (preferred)
)
```

### 2. Gradient Checkpointing

Trades compute for memory (~30% memory reduction, ~20% slower).

```python
model.gradient_checkpointing_enable()
```

### 3. Gradient Accumulation

Simulates larger batch sizes with limited memory.

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
)
```

### 4. LoRA (Low-Rank Adaptation)

For large models, trains only ~1-10% of parameters.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "value"],
)
model = get_peft_model(model, lora_config)
```

Memory reduction: ~90% for gradients and optimizer states.

### 5. QLoRA (Quantized LoRA)

4-bit quantization + LoRA for very large models.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Enables Llama-7B fine-tuning on 12GB GPU.

### 6. DeepSpeed ZeRO

Distributed training with optimizer state sharding.

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

ZeRO stages:
- Stage 1: Optimizer state partitioning (~4x memory reduction)
- Stage 2: + Gradient partitioning (~8x reduction)
- Stage 3: + Parameter partitioning (train any model size)

### 7. CPU Offloading

Move optimizer states to CPU when GPU memory is tight.

```python
training_args = TrainingArguments(
    optim="adamw_8bit",  # 8-bit optimizer
    deepspeed="zero2_offload.json",  # CPU offload
)
```

---

## Quick Reference: Recommended Configurations

### For RoBERTa-base (125M params) on ~300M token corpus:

```python
config = PretrainingConfig(
    model=ModelConfig(model_name_or_path="roberta-base"),
    training=TrainingConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        num_train_epochs=3,
    ),
)
# Requires: 12GB GPU, 16GB RAM, ~5 hours on V100
```

### For DeBERTa-v3-large (434M params) with LoRA:

```python
config = PretrainingConfig(
    model=ModelConfig(
        model_name_or_path="microsoft/deberta-v3-large",
        use_lora=True,
        lora_r=16,
    ),
    training=TrainingConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
    ),
)
# Requires: 16GB GPU, 32GB RAM, ~8 hours on V100
```

### For Llama-7B with QLoRA:

```python
config = PretrainingConfig(
    model=ModelConfig(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        use_lora=True,
        use_4bit=True,
        lora_r=64,
    ),
    training=TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        bf16=True,
        gradient_checkpointing=True,
    ),
)
# Requires: 16GB GPU, 32GB RAM, ~15 hours on V100
```

---

## Estimating Your Requirements

Use the built-in estimation tool:

```python
from rl_money_laundering.pretraining import estimate_hardware_requirements

profile = estimate_hardware_requirements(
    config=your_config,
    corpus_samples=100000,
    avg_tokens_per_sample=256,
)

print(f"GPU Memory: {profile.min_gpu_memory_gb} - {profile.recommended_gpu_memory_gb} GB")
print(f"Training Time: {profile.estimated_training_hours} hours")
print(f"Estimated Cost: ${profile.estimated_cloud_cost_usd}")
print(f"Recommended: {profile.recommended_instance_type}")
```

Or via CLI:

```bash
python -m rl_money_laundering.pretraining.scripts estimate \
    --corpus ./data/aml_corpus.jsonl \
    --model roberta-base \
    --epochs 3
```

---

## Summary

| Scenario | Min GPU | Recommended | Est. Cost | Training Time |
|----------|---------|-------------|-----------|---------------|
| Quick prototype (TAPT only) | 8 GB | 12 GB | $1-2 | 30 min |
| Standard DAPT+TAPT | 12 GB | 24 GB | $15-30 | 5-10 hrs |
| Extended corpus | 24 GB | 40 GB | $30-60 | 15-25 hrs |
| Large model (LoRA) | 16 GB | 32 GB | $25-50 | 10-20 hrs |
| Production scale | 80 GB+ | Multi-GPU | $100+ | Variable |

For most AML use cases, a **single V100 or A100** with the base corpus provides excellent results at reasonable cost.
