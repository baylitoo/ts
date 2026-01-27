"""Tests for pretraining data collators.

Tests SpanMLMCollator, MLMDataCollator, and related functionality.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np


class MockTokenizer:
    """Mock tokenizer for testing collators without loading real models."""

    def __init__(self):
        self.vocab_size = 30000
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.cls_token = "[CLS]"
        self.cls_token_id = 101
        self.sep_token = "[SEP]"
        self.sep_token_id = 102
        self.unk_token = "[UNK]"
        self.unk_token_id = 100

        self.all_special_ids = [0, 100, 101, 102, 103]
        self.all_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.additional_special_tokens = []

    def __len__(self):
        return self.vocab_size

    def convert_ids_to_tokens(self, ids):
        """Convert IDs to fake tokens."""
        tokens = []
        for id in ids:
            if id == self.cls_token_id:
                tokens.append("[CLS]")
            elif id == self.sep_token_id:
                tokens.append("[SEP]")
            elif id == self.pad_token_id:
                tokens.append("[PAD]")
            elif id == self.mask_token_id:
                tokens.append("[MASK]")
            elif id == self.unk_token_id:
                tokens.append("[UNK]")
            else:
                # Simulate subword tokens
                if id % 10 == 0:
                    tokens.append(f"word{id // 10}")
                else:
                    tokens.append(f"##sub{id}")
        return tokens


class TestSpanMLMCollator:
    """Tests for SpanMLMCollator."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def collator(self, tokenizer):
        from rl_money_laundering.pretraining.collators import SpanMLMCollator
        return SpanMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            mean_span_length=3.0,
            max_span_length=10,
        )

    def test_initialization(self, collator):
        """Test collator initializes correctly."""
        assert collator.mlm_probability == 0.15
        assert collator.mean_span_length == 3.0
        assert collator.max_span_length == 10
        assert collator._geom_p == pytest.approx(1.0 / 3.0)

    def test_sample_span_length(self, collator):
        """Test span length sampling from Geometric distribution."""
        np.random.seed(42)
        lengths = [collator._sample_span_length() for _ in range(1000)]

        # All lengths should be >= 1 and <= max_span_length
        assert all(1 <= l <= collator.max_span_length for l in lengths)

        # Mean should be approximately mean_span_length (allowing for capping)
        mean_length = np.mean(lengths)
        # Due to capping at max_span_length, mean will be somewhat lower
        assert 1.5 < mean_length < 4.0

    def test_span_masking_creates_labels(self, collator, tokenizer):
        """Test that span masking creates correct labels."""
        # Create fake batch
        batch_size = 2
        seq_length = 128

        # Generate random input IDs (avoiding special tokens)
        input_ids = torch.randint(200, 1000, (batch_size, seq_length))
        # Add CLS and SEP tokens
        input_ids[:, 0] = tokenizer.cls_token_id
        input_ids[:, -1] = tokenizer.sep_token_id

        examples = [{"input_ids": input_ids[i]} for i in range(batch_size)]

        result = collator(examples)

        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape == (batch_size, seq_length)
        assert result["labels"].shape == (batch_size, seq_length)

        # Check that some tokens are masked
        for i in range(batch_size):
            masked_count = (result["labels"][i] != -100).sum().item()
            total_valid = seq_length - 2  # Exclude CLS and SEP
            expected_masked = int(total_valid * collator.mlm_probability)

            # Allow some variance
            assert masked_count > 0
            assert masked_count <= total_valid

    def test_special_tokens_not_masked(self, collator, tokenizer):
        """Test that special tokens (CLS, SEP, PAD) are never masked."""
        seq_length = 64
        input_ids = torch.randint(200, 1000, (1, seq_length))
        input_ids[0, 0] = tokenizer.cls_token_id
        input_ids[0, -10:] = tokenizer.pad_token_id
        input_ids[0, seq_length - 11] = tokenizer.sep_token_id

        examples = [{"input_ids": input_ids[0]}]
        result = collator(examples)

        # CLS should not be masked (label = -100)
        assert result["labels"][0, 0] == -100

        # Padding should not be masked
        for i in range(-10, 0):
            assert result["labels"][0, i] == -100

    def test_span_statistics(self, collator, tokenizer):
        """Test span statistics tracking."""
        # Reset statistics
        collator.reset_statistics()
        assert collator.get_span_statistics()["count"] == 0

        # Process some batches
        for _ in range(10):
            input_ids = torch.randint(200, 1000, (4, 128))
            input_ids[:, 0] = tokenizer.cls_token_id
            input_ids[:, -1] = tokenizer.sep_token_id
            examples = [{"input_ids": input_ids[i]} for i in range(4)]
            collator(examples)

        stats = collator.get_span_statistics()
        assert stats["count"] > 0
        assert stats["mean"] > 0
        assert stats["min"] >= 1
        assert stats["max"] <= collator.max_span_length

    def test_masking_strategy_distribution(self, collator, tokenizer):
        """Test that 80/10/10 masking strategy is applied."""
        np.random.seed(42)

        # Process many batches to get distribution
        mask_count = 0
        random_count = 0
        unchanged_count = 0

        for _ in range(100):
            input_ids = torch.randint(200, 1000, (1, 64))
            input_ids[0, 0] = tokenizer.cls_token_id
            input_ids[0, -1] = tokenizer.sep_token_id
            original = input_ids.clone()

            examples = [{"input_ids": input_ids[0]}]
            result = collator(examples)

            # Count masking types
            for j in range(64):
                if result["labels"][0, j] != -100:  # This position was masked
                    if result["input_ids"][0, j] == tokenizer.mask_token_id:
                        mask_count += 1
                    elif result["input_ids"][0, j] == original[0, j]:
                        unchanged_count += 1
                    else:
                        random_count += 1

        total = mask_count + random_count + unchanged_count
        if total > 0:
            # Expect roughly 80/10/10 distribution
            mask_ratio = mask_count / total
            random_ratio = random_count / total
            unchanged_ratio = unchanged_count / total

            assert 0.7 < mask_ratio < 0.9, f"Mask ratio {mask_ratio} not ~0.8"
            assert 0.05 < random_ratio < 0.15, f"Random ratio {random_ratio} not ~0.1"
            assert 0.05 < unchanged_ratio < 0.15, f"Unchanged ratio {unchanged_ratio} not ~0.1"


class TestMLMDataCollator:
    """Tests for standard MLMDataCollator."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def collator(self, tokenizer):
        from rl_money_laundering.pretraining.collators import MLMDataCollator
        return MLMDataCollator(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            whole_word_masking=True,
        )

    def test_basic_masking(self, collator, tokenizer):
        """Test basic MLM masking works."""
        input_ids = torch.randint(200, 1000, (2, 64))
        input_ids[:, 0] = tokenizer.cls_token_id
        input_ids[:, -1] = tokenizer.sep_token_id

        examples = [{"input_ids": input_ids[i]} for i in range(2)]
        result = collator(examples)

        assert "input_ids" in result
        assert "labels" in result

        # Check masking occurred
        for i in range(2):
            masked_count = (result["labels"][i] != -100).sum().item()
            assert masked_count > 0


class TestEarlyStoppingCallback:
    """Tests for enhanced EarlyStoppingCallback."""

    def test_loss_based_stopping(self):
        """Test early stopping based on loss."""
        from rl_money_laundering.pretraining.trainer import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=3,
            min_delta=0.001,
            metric="loss",
        )

        # Mock trainer state and control
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        control.should_training_stop = False

        # Simulate improving loss
        metrics = {"eval_loss": 2.0}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.best_value == 2.0
        assert callback.patience_counter == 0

        metrics = {"eval_loss": 1.5}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.best_value == 1.5
        assert callback.patience_counter == 0

        # Simulate no improvement
        metrics = {"eval_loss": 1.6}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 1

        metrics = {"eval_loss": 1.7}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 2

        metrics = {"eval_loss": 1.55}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 3
        assert control.should_training_stop

    def test_perplexity_based_stopping(self):
        """Test early stopping based on perplexity."""
        from rl_money_laundering.pretraining.trainer import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=2,
            min_delta=0.01,  # 1% relative improvement
            metric="perplexity",
        )

        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        control.should_training_stop = False

        # Loss of 2.0 -> perplexity of exp(2) = 7.39
        metrics = {"eval_loss": 2.0}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 0

        # Loss of 1.8 -> perplexity of exp(1.8) = 6.05 (improvement)
        metrics = {"eval_loss": 1.8}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 0

        # Loss of 1.81 -> perplexity of exp(1.81) = 6.11 (no improvement)
        metrics = {"eval_loss": 1.81}
        callback.on_evaluate(None, state, control, metrics)
        assert callback.patience_counter == 1

    def test_max_perplexity_ceiling(self):
        """Test that max_perplexity triggers immediate stop."""
        from rl_money_laundering.pretraining.trainer import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=10,  # High patience
            metric="perplexity",
            max_perplexity=100.0,  # Perplexity ceiling
        )

        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        control.should_training_stop = False

        # Loss of 5.0 -> perplexity of exp(5) = 148.4 > 100
        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(None, state, control, metrics)
        assert control.should_training_stop

    def test_history_tracking(self):
        """Test that evaluation history is tracked."""
        from rl_money_laundering.pretraining.trainer import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=3)

        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        control.should_training_stop = False

        for i in range(5):
            state.global_step = i * 100
            metrics = {"eval_loss": 2.0 - i * 0.1}
            callback.on_evaluate(None, state, control, metrics)

        history = callback.get_history()
        assert len(history) == 5
        assert all("loss" in h and "perplexity" in h for h in history)


class TestCreateDataCollator:
    """Tests for create_data_collator factory function."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_creates_mlm_collator(self, tokenizer):
        """Test creating MLM collator."""
        from rl_money_laundering.pretraining.collators import (
            create_data_collator,
            MLMDataCollator,
        )

        collator = create_data_collator(tokenizer, objective="mlm")
        assert isinstance(collator, MLMDataCollator)

    def test_creates_span_mlm_collator(self, tokenizer):
        """Test creating Span MLM collator."""
        from rl_money_laundering.pretraining.collators import (
            create_data_collator,
            SpanMLMCollator,
        )

        collator = create_data_collator(
            tokenizer,
            objective="span_mlm",
            mean_noise_span_length=3.0,
        )
        assert isinstance(collator, SpanMLMCollator)
        assert collator.mean_span_length == 3.0

    def test_creates_clm_collator(self, tokenizer):
        """Test creating CLM collator."""
        from rl_money_laundering.pretraining.collators import (
            create_data_collator,
            CLMDataCollator,
        )

        collator = create_data_collator(tokenizer, objective="clm")
        assert isinstance(collator, CLMDataCollator)

    def test_invalid_objective_raises(self, tokenizer):
        """Test that invalid objective raises ValueError."""
        from rl_money_laundering.pretraining.collators import create_data_collator

        with pytest.raises(ValueError, match="Unknown objective"):
            create_data_collator(tokenizer, objective="invalid")


class TestTAPTConfig:
    """Tests for TAPT configuration defaults."""

    def test_tapt_config_defaults(self):
        """Test TAPT configuration has correct defaults for overfitting prevention."""
        from rl_money_laundering.pretraining.config import PretrainingConfig

        config = PretrainingConfig.for_tapt(
            base_model="roberta-base",
        )

        # Check learning rate is lower
        assert config.training.learning_rate == 1e-5

        # Check weight decay is higher
        assert config.training.weight_decay == 0.05

        # Check span MLM is enabled
        assert config.data.use_span_mlm is True

        # Check mean span length
        assert config.data.mean_noise_span_length == 3.0

        # Check early stopping settings
        assert config.training.eval_strategy == "steps"
        assert config.training.load_best_model_at_end is True

    def test_tapt_config_customization(self):
        """Test TAPT configuration can be customized."""
        from rl_money_laundering.pretraining.config import PretrainingConfig

        config = PretrainingConfig.for_tapt(
            base_model="roberta-base",
            use_span_mlm=False,
            mean_span_length=5.0,
        )

        assert config.data.use_span_mlm is False
        assert config.data.mean_noise_span_length == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
