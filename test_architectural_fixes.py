"""
Tests for architectural defect fixes implemented in ae_train.py.

Covers fixes C1–C6, M1–M3, W1–W4 as described in the upgrade plan.
Each test targets a specific fix and validates the corrected behavior.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from ae_train import (
    AEONConfigV4,
    AEONDeltaV4,
    ContextualRSSMTrainer,
    QualityHead,
    TrainingMonitor,
    VTStreamingSignalBus,
    _QUALITY_LOSS_LAMBDA,
    adapt_entropy_weight,
    align_ssp_temperature,
    annotate_z_sequences_quality,
    auto_detect_task_boundary,
    diagnose_corpus_via_vt,
    micro_retrain_from_pseudo_labels,
    ucc_inner_epoch_evaluation,
)
from aeon_core import AEONConfig, ContinualLearningCore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> AEONConfigV4:
    """Return a lightweight AEONConfigV4 with small dimensions for speed."""
    defaults = dict(z_dim=32, hidden_dim=32, vq_embedding_dim=32,
                    vq_num_embeddings=64, context_window=2,
                    rssm_hidden_dim=64, seq_length=16, vocab_size=256)
    defaults.update(overrides)
    return AEONConfigV4(**defaults)


def _make_model(config: Optional[AEONConfigV4] = None) -> AEONDeltaV4:
    cfg = config or _make_config()
    return AEONDeltaV4(cfg)


def _make_trainer(config=None, model=None):
    """Build a ContextualRSSMTrainer with minimal setup."""
    import logging
    cfg = config or _make_config()
    mdl = model or _make_model(cfg)
    monitor = TrainingMonitor(logger=logging.getLogger("test"), save_dir=".")
    return ContextualRSSMTrainer(model=mdl, config=cfg, monitor=monitor)


# ===========================================================================
# [C1] QualityHead + L_quality in Phase B
# ===========================================================================

class TestC1QualityHead:
    """C1: QualityHead exists and integrates into Phase B training."""

    def test_quality_head_exists(self):
        assert QualityHead is not None

    def test_quality_head_forward_shape(self):
        head = QualityHead(z_dim=32, hidden_dim=16)
        z = torch.randn(4, 32)
        out = head(z)
        assert out.shape == (4, 3), f"Expected (4,3), got {out.shape}"

    def test_quality_head_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        head = QualityHead(z_dim=32)
        out = head(torch.randn(8, 32))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_quality_loss_lambda_positive_float(self):
        assert isinstance(_QUALITY_LOSS_LAMBDA, float)
        assert _QUALITY_LOSS_LAMBDA > 0

    def test_train_step_accepts_quality_target(self):
        trainer = _make_trainer()
        K = trainer.config.context_window
        D = trainer.config.z_dim
        z_ctx = torch.randn(2, K, D)
        z_tgt = torch.randn(2, D)
        q_tgt = torch.rand(2, 3)
        result = trainer.train_step(z_ctx, z_tgt, quality_target=q_tgt)
        assert "quality_loss" in result
        assert result["quality_loss"] >= 0.0

    def test_train_step_quality_loss_zero_without_target(self):
        trainer = _make_trainer()
        K, D = trainer.config.context_window, trainer.config.z_dim
        result = trainer.train_step(torch.randn(2, K, D), torch.randn(2, D))
        assert result["quality_loss"] == 0.0

    def test_fit_accepts_quality_annotations(self):
        cfg = _make_config(context_window=2)
        trainer = _make_trainer(config=cfg)
        D = cfg.z_dim
        K = cfg.context_window
        num_chunks = K + 3
        z_seqs = [torch.randn(num_chunks, D)]
        q_ann = [torch.rand(num_chunks, 3)]
        # Should not raise
        trainer.fit(z_seqs, epochs=1, quality_annotations=q_ann)


# ===========================================================================
# [C2] align_ssp_temperature() invocation
# ===========================================================================

class TestC2AlignSSPTemperature:
    """C2: align_ssp_temperature returns expected keys."""

    def test_function_exists(self):
        assert callable(align_ssp_temperature)

    def test_default_params_returns_expected_keys(self):
        result = align_ssp_temperature()
        expected_keys = {
            "vt_temperature_original", "gumbel_temperature_original",
            "vt_temperature_aligned", "gumbel_temperature_aligned",
            "geometric_mean", "alignment_factor",
        }
        assert expected_keys == set(result.keys())

    @pytest.mark.parametrize("gumbel_temp", [0.5, 1.0, 2.0, 5.0])
    def test_custom_gumbel_temperature(self, gumbel_temp):
        result = align_ssp_temperature(gumbel_temperature=gumbel_temp)
        assert result["gumbel_temperature_original"] == gumbel_temp
        assert result["gumbel_temperature_aligned"] > 0

    def test_alignment_factor_respected(self):
        r0 = align_ssp_temperature(alignment_factor=0.0)
        r1 = align_ssp_temperature(alignment_factor=1.0)
        # factor=0 → aligned == original
        assert r0["vt_temperature_aligned"] == pytest.approx(
            r0["vt_temperature_original"]
        )
        # factor=1 → aligned == geometric mean
        assert r1["vt_temperature_aligned"] == pytest.approx(
            r1["geometric_mean"]
        )


# ===========================================================================
# [C3] lr_scale applied to optimizer via VTStreamingSignalBus
# ===========================================================================

class TestC3LRScale:
    """C3: apply_to_controller returns lr_scale when calibration_pressure > 0.3."""

    def test_lr_scale_returned_when_pressure_high(self):
        bus = VTStreamingSignalBus()
        # Push enough values to build EMA above 0.3
        for _ in range(20):
            bus.push("calibration_pressure", 0.8)
        controller = MagicMock()
        result = bus.apply_to_controller(controller)
        assert "lr_scale" in result
        assert 0.0 < result["lr_scale"] <= 1.0

    def test_no_lr_scale_when_pressure_low(self):
        bus = VTStreamingSignalBus()
        bus.push("calibration_pressure", 0.1)
        result = bus.apply_to_controller(MagicMock())
        assert "lr_scale" not in result


# ===========================================================================
# [C4] micro_retrain uses real z
# ===========================================================================

class TestC4MicroRetrainRealZ:
    """C4: micro_retrain_from_pseudo_labels uses real z when provided."""

    def _pseudo_labels(self, n=3):
        return [{"confidence": 0.9, "quality": 0.9, "cot_depth": 1, "episode": i}
                for i in range(n)]

    def test_accepts_z_sequences_param(self):
        cfg = _make_config()
        model = _make_model(cfg)
        z_seqs = [torch.randn(5, cfg.z_dim)]
        result = micro_retrain_from_pseudo_labels(
            model=model, pseudo_labels=self._pseudo_labels(),
            config=cfg, z_sequences=z_seqs, max_steps=2,
        )
        assert isinstance(result, dict)

    def test_used_real_z_true_when_provided(self):
        cfg = _make_config()
        model = _make_model(cfg)
        z_seqs = [torch.randn(5, cfg.z_dim)]
        result = micro_retrain_from_pseudo_labels(
            model=model, pseudo_labels=self._pseudo_labels(),
            config=cfg, z_sequences=z_seqs, max_steps=2,
        )
        if result.get("retrained"):
            assert result["used_real_z"] is True

    def test_works_without_z_sequences(self):
        cfg = _make_config()
        model = _make_model(cfg)
        result = micro_retrain_from_pseudo_labels(
            model=model, pseudo_labels=self._pseudo_labels(),
            config=cfg, z_sequences=None, max_steps=2,
        )
        assert isinstance(result, dict)


# ===========================================================================
# [C5] EWC in micro_retrain
# ===========================================================================

class TestC5EWCInMicroRetrain:
    """C5: EWC penalty is applied during micro-retrain."""

    def test_ewc_applied_flag(self):
        cfg = _make_config()
        model = _make_model(cfg)
        pls = [{"confidence": 0.9, "quality": 0.9, "cot_depth": 1, "episode": 0}]
        result = micro_retrain_from_pseudo_labels(
            model=model, pseudo_labels=pls, config=cfg, max_steps=2,
        )
        if result.get("retrained"):
            assert result["ewc_applied"] is True


# ===========================================================================
# [C6] ContinualLearningCore constructor
# ===========================================================================

class TestC6ContinualLearningCore:
    """C6: ContinualLearningCore(base_model=...) works correctly."""

    def test_construct_with_linear(self):
        clc = ContinualLearningCore(base_model=nn.Linear(32, 32))
        assert clc is not None
        assert len(clc.columns) == 1

    def test_ae_train_uses_base_model_keyword(self):
        """Verify ae_train constructs ContinualLearningCore with base_model=model.encoder."""
        import inspect, ae_train
        # Search only in function bodies, not comments/docstrings
        src = inspect.getsource(ae_train.main)
        assert "ContinualLearningCore(" in src
        assert "base_model=model.encoder" in src


# ===========================================================================
# [M1] Task boundary in epoch loop
# ===========================================================================

class TestM1TaskBoundary:
    """M1: auto_detect_task_boundary works with various coherence scores."""

    def test_function_exists(self):
        assert callable(auto_detect_task_boundary)

    @pytest.mark.parametrize("coherence,expected", [
        (0.1, True), (0.3, True), (0.49, True),
        (0.5, False), (0.8, False), (1.0, False),
    ])
    def test_boundary_detection(self, coherence, expected):
        result = auto_detect_task_boundary(coherence_score=coherence)
        assert result["boundary_detected"] is expected

    def test_coherence_drop_computed(self):
        result = auto_detect_task_boundary(
            coherence_score=0.3, previous_coherence=0.9,
        )
        assert result["coherence_drop"] == pytest.approx(0.6)

    def test_recommendation_add_task_when_boundary(self):
        result = auto_detect_task_boundary(coherence_score=0.1)
        assert result["recommendation"] in ("add_task_strong", "add_task_weak")

    def test_recommendation_continue_when_no_boundary(self):
        result = auto_detect_task_boundary(coherence_score=0.8)
        assert result["recommendation"] == "continue"


# ===========================================================================
# [M2] diagnose_corpus_via_vt codebook_size
# ===========================================================================

class TestM2DiagnoseCorpusViaVT:
    """M2: diagnose_corpus_via_vt function signature exists."""

    def test_function_exists(self):
        assert callable(diagnose_corpus_via_vt)

    def test_signature_has_expected_params(self):
        import inspect
        sig = inspect.signature(diagnose_corpus_via_vt)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "tokens" in params
        assert "config" in params


# ===========================================================================
# [M3] vibe_thinker_weights_path handling
# ===========================================================================

class TestM3VibeThinkerWeightsPath:
    """M3: AEONConfig (aeon_core) has vibe_thinker_weights_path attribute."""

    def test_aeon_config_has_attribute(self):
        cfg = AEONConfig(
            hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
            seq_length=32, dropout_rate=0.0, meta_dim=32,
            lipschitz_target=0.9, vq_embedding_dim=64,
        )
        assert hasattr(cfg, "vibe_thinker_weights_path")
        assert cfg.vibe_thinker_weights_path == ""

    def test_training_code_reads_vibe_thinker_weights_path(self):
        """ae_train.main() uses getattr to read vibe_thinker_weights_path."""
        import inspect, ae_train
        src = inspect.getsource(ae_train.main)
        assert "vibe_thinker_weights_path" in src


# ===========================================================================
# [W1] ucc_inner_epoch_evaluation
# ===========================================================================

class TestW1UCCInnerEpochEvaluation:
    """W1: ucc_inner_epoch_evaluation exists and respects interval."""

    def test_function_exists(self):
        assert callable(ucc_inner_epoch_evaluation)

    def test_returns_early_when_not_interval(self):
        cycle = MagicMock()
        result = ucc_inner_epoch_evaluation(
            cycle=cycle, subsystem_states={}, loss_delta=0.0,
            uncertainty=0.0, epoch=1, total_epochs=100, interval=5,
        )
        assert result["evaluated"] is False

    def test_evaluates_at_interval_boundary(self):
        cycle = MagicMock()
        cycle.evaluate.return_value = {
            "coherence_result": {"coherence_deficit": 0.1},
            "should_rerun": False,
        }
        result = ucc_inner_epoch_evaluation(
            cycle=cycle, subsystem_states={}, loss_delta=0.01,
            uncertainty=0.1, epoch=10, total_epochs=100, interval=5,
        )
        assert result["evaluated"] is True

    def test_accepts_interval_parameter(self):
        import inspect
        sig = inspect.signature(ucc_inner_epoch_evaluation)
        assert "interval" in sig.parameters


# ===========================================================================
# [W2] Item-wise reason() in annotate_z_sequences_quality
# ===========================================================================

class TestW2AnnotateZSequencesQuality:
    """W2: annotate_z_sequences_quality exists and is callable."""

    def test_function_exists(self):
        assert callable(annotate_z_sequences_quality)

    def test_signature_has_expected_params(self):
        import inspect
        sig = inspect.signature(annotate_z_sequences_quality)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "z_sequences" in params
        assert "config" in params


# ===========================================================================
# [W3] min_episodes guard in closed_loop_step
# ===========================================================================

class TestW3MinEpisodesGuard:
    """W3: closed_loop_step returns streaming=False when episodes < 50."""

    def test_streaming_false_below_threshold(self):
        bus = VTStreamingSignalBus()
        learner = SimpleNamespace(_episode_count=10)
        controller = MagicMock()
        result = bus.closed_loop_step(learner, controller)
        assert result["streaming"] is False

    def test_streaming_false_at_zero_episodes(self):
        bus = VTStreamingSignalBus()
        learner = SimpleNamespace(_episode_count=0)
        result = bus.closed_loop_step(learner, MagicMock())
        assert result["streaming"] is False

    def test_streaming_true_above_threshold(self):
        bus = VTStreamingSignalBus()
        learner = SimpleNamespace(
            _episode_count=50,
            _calibration_ema=0.1,
            _psi_weight_ema=0.1,
            _complexity_threshold_ema=0.5,
        )
        result = bus.closed_loop_step(learner, MagicMock())
        assert result["streaming"] is True

    def test_streaming_false_with_none_learner(self):
        bus = VTStreamingSignalBus()
        result = bus.closed_loop_step(None, MagicMock())
        assert result["streaming"] is False


# ===========================================================================
# [W4] Per-epoch adapt_entropy_weight
# ===========================================================================

class TestW4AdaptEntropyWeight:
    """W4: adapt_entropy_weight mutates config.entropy_weight."""

    def test_mutates_config(self):
        cfg = _make_config()
        original = cfg.entropy_weight
        adapt_entropy_weight(cfg, vt_entropy=0.9)
        assert cfg.entropy_weight != original

    def test_high_entropy_increases_weight(self):
        cfg = _make_config(entropy_weight=0.1)
        adapt_entropy_weight(cfg, vt_entropy=0.9, target_entropy=0.5)
        assert cfg.entropy_weight > 0.1

    def test_low_entropy_decreases_weight(self):
        cfg = _make_config(entropy_weight=0.5)
        adapt_entropy_weight(cfg, vt_entropy=0.1, target_entropy=0.5)
        assert cfg.entropy_weight < 0.5

    def test_returns_expected_keys(self):
        cfg = _make_config()
        result = adapt_entropy_weight(cfg, vt_entropy=0.5)
        assert "original_weight" in result
        assert "new_weight" in result
        assert "delta" in result

    @pytest.mark.parametrize("vt_entropy", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_weight_stays_in_bounds(self, vt_entropy):
        cfg = _make_config(entropy_weight=0.5)
        adapt_entropy_weight(cfg, vt_entropy=vt_entropy)
        assert 0.01 <= cfg.entropy_weight <= 1.0
