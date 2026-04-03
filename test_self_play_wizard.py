"""
Tests for AEON Self-Play Wizard Architecture (§SP).

Tests cover:
  - GenerationMode enum
  - LatentWorldGenerator (§SP.1): SMOOTH / STRUCTURED / ADVERSARIAL
  - AdaptiveCurriculumManager (§SP.3): ALP, promotion, regression
  - CorrectiveSynthesizer (§SP.4): failure-mode correction
  - bootstrap_codebook_embeddings (§SP.5): corpus-free codebook init
  - VibeThinkerMetaSignaler (§SP.6): λ_cos adaptation
  - run_self_play_wizard: full orchestrator
  - run_wizard backward compatibility (tokens ignored)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
#  Minimal stubs
# ===========================================================================

class _StubConfig:
    z_dim: int = 32
    hidden_dim: int = 32
    vq_embedding_dim: int = 32
    context_window: int = 3
    vq_num_embeddings: int = 16
    codebook_size: int = 16
    seed: int = 42
    document_aware: bool = True
    seq_length: int = 128
    vocab_size: int = 30522
    grad_clip_norm: float = 1.0
    entropy_weight: float = 0.01
    vq_reset_threshold: float = 0.1
    warmup_steps: int = 100
    action_dim: int = 16


class _StubEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, z_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class _StubVQ(nn.Module):
    def __init__(self, num_embeddings: int = 16, z_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, z_dim)
        self.ema_w = nn.Parameter(
            torch.randn(num_embeddings, z_dim), requires_grad=False,
        )
        self.ema_cluster_size = nn.Parameter(
            torch.ones(num_embeddings), requires_grad=False,
        )


class _StubModel(nn.Module):
    def __init__(self, z_dim: int = 32, num_embeddings: int = 16):
        super().__init__()
        self.encoder = _StubEncoder(128, z_dim)
        self.vq = _StubVQ(num_embeddings, z_dim)
        self.vibe_thinker_adapter = None
        self.vibe_thinker_kernel = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ===========================================================================
#  GenerationMode Tests
# ===========================================================================

class TestGenerationMode:
    """Tests for GenerationMode enum."""

    def test_enum_values(self):
        from aeon_wizard import GenerationMode
        assert GenerationMode.SMOOTH.value == "smooth"
        assert GenerationMode.STRUCTURED.value == "structured"
        assert GenerationMode.ADVERSARIAL.value == "adversarial"

    def test_enum_members(self):
        from aeon_wizard import GenerationMode
        assert len(GenerationMode) == 3


# ===========================================================================
#  LatentWorldGenerator Tests
# ===========================================================================

class TestLatentWorldGenerator:
    """Tests for LatentWorldGenerator (§SP.1)."""

    def test_init(self):
        from aeon_wizard import LatentWorldGenerator
        gen = LatentWorldGenerator(latent_dim=32, action_dim=16)
        assert gen.latent_dim == 32
        assert gen.action_dim == 16
        assert not gen._initialised

    def test_generate_smooth(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        result = gen.generate(
            mode=GenerationMode.SMOOTH,
            batch_size=8,
            config=config,
        )
        assert "scenarios" in result
        assert result["scenarios"].shape == (8, 32)
        assert "complexity" in result
        assert result["complexity"].shape == (8,)
        assert "intrinsic_reward" in result
        assert result["mode"] == "smooth"

    def test_generate_structured(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        result = gen.generate(
            mode=GenerationMode.STRUCTURED,
            batch_size=4,
            config=config,
        )
        assert result["scenarios"].shape == (4, 32)
        assert result["mode"] == "structured"

    def test_generate_adversarial(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        result = gen.generate(
            mode=GenerationMode.ADVERSARIAL,
            batch_size=4,
            config=config,
        )
        assert result["scenarios"].shape == (4, 32)
        assert result["mode"] == "adversarial"

    def test_generate_with_base_z(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        base_z = torch.randn(4, 32)
        result = gen.generate(
            mode=GenerationMode.SMOOTH,
            batch_size=4,
            config=config,
            base_z=base_z,
        )
        assert result["scenarios"].shape == (4, 32)

    def test_scenarios_normalized(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        result = gen.generate(
            mode=GenerationMode.SMOOTH,
            batch_size=8,
            config=config,
        )
        norms = result["scenarios"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_get_summary(self):
        from aeon_wizard import LatentWorldGenerator
        gen = LatentWorldGenerator(latent_dim=64, action_dim=32)
        summary = gen.get_summary()
        assert summary["latent_dim"] == 64
        assert summary["action_dim"] == 32
        assert summary["initialised"] is False

    def test_lazy_init_without_aeon_core(self):
        from aeon_wizard import LatentWorldGenerator
        gen = LatentWorldGenerator(latent_dim=32)
        config = _StubConfig()
        gen._lazy_init(config)
        assert gen._initialised is True
        # Without aeon_core in path, dynamics may or may not be available
        summary = gen.get_summary()
        assert isinstance(summary["has_dynamics"], bool)

    def test_different_batch_sizes(self):
        from aeon_wizard import LatentWorldGenerator, GenerationMode
        gen = LatentWorldGenerator(latent_dim=16)
        config = _StubConfig()
        for bs in [1, 4, 16, 64]:
            result = gen.generate(
                mode=GenerationMode.SMOOTH,
                batch_size=bs,
                config=config,
            )
            assert result["scenarios"].shape[0] == bs


# ===========================================================================
#  AdaptiveCurriculumManager Tests
# ===========================================================================

class TestAdaptiveCurriculumManager:
    """Tests for AdaptiveCurriculumManager (§SP.3)."""

    def test_init(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager()
        assert mgr.current_level == 0
        assert mgr.success_rate == 0.0

    def test_generation_mode_mapping(self):
        from aeon_wizard import AdaptiveCurriculumManager, GenerationMode
        mgr = AdaptiveCurriculumManager()
        mgr.current_level = 0
        assert mgr.generation_mode == GenerationMode.SMOOTH
        mgr.current_level = 1
        assert mgr.generation_mode == GenerationMode.SMOOTH
        mgr.current_level = 2
        assert mgr.generation_mode == GenerationMode.STRUCTURED
        mgr.current_level = 3
        assert mgr.generation_mode == GenerationMode.STRUCTURED
        mgr.current_level = 4
        assert mgr.generation_mode == GenerationMode.ADVERSARIAL

    def test_record_outcome(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager()
        result = mgr.record_outcome(success=True)
        assert result["level"] == 0
        assert result["success_rate"] == 1.0

    def test_success_rate(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager()
        for _ in range(3):
            mgr.record_outcome(success=True)
        for _ in range(1):
            mgr.record_outcome(success=False)
        assert abs(mgr.success_rate - 0.75) < 0.01

    def test_promotion(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager(
            alp_window=4,
            alp_epsilon=0.1,
        )
        # Fill with successes: SR ≥ 0.75 and ALP → 0
        for _ in range(20):
            mgr.record_outcome(success=True)
        # After enough consistent successes, level should increase
        assert mgr.current_level >= 1

    def test_regression(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager(alp_window=4)
        mgr.current_level = 2
        mgr._level_successes.clear()
        # Add failures: SR < 0.35
        for _ in range(10):
            mgr.record_outcome(success=False)
        # Level should decrease
        assert mgr.current_level < 2

    def test_no_regression_at_level_0(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager(alp_window=4)
        mgr.current_level = 0
        for _ in range(10):
            mgr.record_outcome(success=False)
        assert mgr.current_level == 0

    def test_max_level_cap(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager(max_level=2, alp_window=4, alp_epsilon=0.5)
        mgr.current_level = 2
        mgr._level_successes.clear()
        for _ in range(20):
            mgr.record_outcome(success=True)
        assert mgr.current_level <= 2

    def test_get_summary(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager()
        mgr.record_outcome(success=True)
        summary = mgr.get_summary()
        assert "current_level" in summary
        assert "generation_mode" in summary
        assert "success_rate" in summary
        assert "alp" in summary
        assert "total_episodes" in summary
        assert summary["total_episodes"] == 1

    def test_alp_with_few_samples(self):
        from aeon_wizard import AdaptiveCurriculumManager
        mgr = AdaptiveCurriculumManager()
        assert mgr.absolute_learning_progress == 1.0  # high uncertainty
        mgr.record_outcome(success=True)
        assert mgr.absolute_learning_progress == 1.0  # still few samples


# ===========================================================================
#  CorrectiveSynthesizer Tests
# ===========================================================================

class TestCorrectiveSynthesizer:
    """Tests for CorrectiveSynthesizer (§SP.4)."""

    def test_init(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32)
        assert synth.latent_dim == 32
        assert synth.num_corrective == 16

    def test_synthesize_direction(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=8)
        z = torch.randn(32)
        result = synth.synthesize(failure_z=z, failure_mode="direction")
        assert result["corrective_scenarios"].shape == (8, 32)
        assert result["failure_mode"] == "direction"
        assert result["strategy"] == "direction"

    def test_synthesize_magnitude(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=4)
        z = torch.randn(32)
        result = synth.synthesize(failure_z=z, failure_mode="magnitude")
        assert result["corrective_scenarios"].shape == (4, 32)
        assert result["failure_mode"] == "magnitude"

    def test_synthesize_oscillation(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=8)
        z = torch.randn(32)
        result = synth.synthesize(failure_z=z, failure_mode="oscillation")
        assert result["corrective_scenarios"].shape == (8, 32)
        assert result["failure_mode"] == "oscillation"

    def test_synthesize_collapse(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=8)
        z = torch.randn(32)
        result = synth.synthesize(failure_z=z, failure_mode="collapse")
        assert result["corrective_scenarios"].shape == (8, 32)
        assert result["failure_mode"] == "collapse"

    def test_synthesize_unknown_mode(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=4)
        z = torch.randn(32)
        result = synth.synthesize(failure_z=z, failure_mode="unknown")
        assert result["corrective_scenarios"].shape == (4, 32)

    def test_classify_failure_direction(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32)
        z1 = torch.randn(32)
        z2 = -z1  # opposite direction
        mode = synth.classify_failure(z1, z2)
        assert mode == "direction"

    def test_classify_failure_magnitude(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32)
        z_target = torch.randn(32)
        z_pred = z_target * 2.0  # same direction, double magnitude
        mode = synth.classify_failure(z_pred, z_target)
        assert mode in ("magnitude", "direction", "oscillation")

    def test_classify_failure_collapse(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32)
        z_target = torch.randn(32)
        z_pred = torch.zeros(32)  # collapsed
        mode = synth.classify_failure(z_pred, z_target)
        assert mode in ("direction", "collapse")

    def test_batch_failure_z(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=4)
        z = torch.randn(1, 32)  # batch dim
        result = synth.synthesize(failure_z=z, failure_mode="direction")
        assert result["corrective_scenarios"].shape == (4, 32)

    def test_get_summary(self):
        from aeon_wizard import CorrectiveSynthesizer
        synth = CorrectiveSynthesizer(latent_dim=64, num_corrective=32)
        summary = synth.get_summary()
        assert summary["latent_dim"] == 64
        assert summary["num_corrective"] == 32


# ===========================================================================
#  Bootstrap Codebook Tests
# ===========================================================================

class TestBootstrapCodebookEmbeddings:
    """Tests for bootstrap_codebook_embeddings (§SP.5)."""

    def test_basic_bootstrap(self):
        from aeon_wizard import bootstrap_codebook_embeddings
        model = _StubModel(z_dim=32, num_embeddings=16)
        config = _StubConfig()
        config.vq_num_embeddings = 16
        result = bootstrap_codebook_embeddings(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_samples=128,
        )
        assert result["initialized"] is True
        assert result["method"] == "bootstrap_synthetic_kmeans"
        assert result["num_embeddings"] == 16

    def test_codebook_weights_updated(self):
        from aeon_wizard import bootstrap_codebook_embeddings
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        old_weights = model.vq.embedding.weight.clone()
        result = bootstrap_codebook_embeddings(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_samples=64,
        )
        if result["initialized"]:
            new_weights = model.vq.embedding.weight
            # Weights should have changed
            assert not torch.allclose(old_weights, new_weights)

    def test_temperature_parameter(self):
        from aeon_wizard import bootstrap_codebook_embeddings
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        result = bootstrap_codebook_embeddings(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_samples=64,
            temperature=2.0,
        )
        assert result.get("temperature", 1.5) == 2.0

    def test_inertia_finite(self):
        from aeon_wizard import bootstrap_codebook_embeddings
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        result = bootstrap_codebook_embeddings(
            model=model,
            config=config,
            num_samples=64,
        )
        if result["initialized"]:
            assert result["inertia"] >= 0
            assert np.isfinite(result["inertia"])


# ===========================================================================
#  VibeThinkerMetaSignaler Tests
# ===========================================================================

class TestVibeThinkerMetaSignaler:
    """Tests for VibeThinkerMetaSignaler (§SP.6)."""

    def test_init(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        assert sig.lambda_cos == 0.1
        assert sig._learner is None

    def test_update_without_learner(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        result = sig.update()
        assert "lambda_cos" in result
        assert "calibration_ema" in result
        assert result["calibration_ema"] == 0.0

    def test_update_with_learner(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler(base_lambda_cos=0.1)
        learner = MagicMock()
        learner._calibration_ema = 0.3
        sig.attach_learner(learner)
        result = sig.update()
        # λ_cos = 0.1 * (1 + 2*0.3) = 0.16
        assert abs(result["lambda_cos"] - 0.16) < 0.01

    def test_lambda_cos_bounds(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler(
            base_lambda_cos=0.1,
            lambda_cos_min=0.05,
            lambda_cos_max=0.3,
        )
        learner = MagicMock()
        learner._calibration_ema = 10.0  # extreme
        sig.attach_learner(learner)
        result = sig.update()
        assert result["lambda_cos"] <= 0.3

        learner._calibration_ema = -5.0  # extreme negative
        result = sig.update()
        assert result["lambda_cos"] >= 0.05

    def test_compute_loss(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        z_pred = torch.randn(4, 32)
        z_target = torch.randn(4, 32)
        loss = sig.compute_loss(z_pred, z_target)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_compute_loss_identical(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        z = torch.randn(4, 32)
        loss = sig.compute_loss(z, z)
        # Should be very small (both MSE and cosine terms → 0)
        assert loss.item() < 0.01

    def test_get_summary(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        summary = sig.get_summary()
        assert "lambda_cos" in summary
        assert "has_learner" in summary
        assert summary["has_learner"] is False

    def test_history_bounded(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        for _ in range(600):
            sig.update()
        assert len(sig._history) <= 500


# ===========================================================================
#  run_self_play_wizard Tests
# ===========================================================================

class TestRunSelfPlayWizard:
    """Tests for run_self_play_wizard orchestrator."""

    def test_basic_run(self):
        from aeon_wizard import run_self_play_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        result = run_self_play_wizard(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_episodes=4,
        )
        assert result["wizard_completed"] is True
        assert "overall_status" in result
        assert "total_duration_s" in result
        assert result["architecture"] == "self_play_synthetic_curriculum"

    def test_self_play_results(self):
        from aeon_wizard import run_self_play_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        result = run_self_play_wizard(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_episodes=8,
        )
        if "self_play" in result and "error" not in result["self_play"]:
            sp = result["self_play"]
            assert sp["episodes"] == 8
            assert "final_level" in sp
            assert "curriculum_summary" in sp
            assert "meta_signaler" in sp

    def test_codebook_init_in_wizard(self):
        from aeon_wizard import run_self_play_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel(z_dim=32, num_embeddings=16)
        config = _StubConfig()
        config.vq_num_embeddings = 16
        result = run_self_play_wizard(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_episodes=2,
        )
        assert "codebook_init" in result

    def test_wizard_state_updated(self):
        from aeon_wizard import (
            run_self_play_wizard, get_wizard_state, reset_wizard_state,
        )
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        run_self_play_wizard(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_episodes=2,
        )
        state = get_wizard_state()
        assert state.overall_status in ("completed", "completed_with_warnings")
        assert state.finished_at is not None


# ===========================================================================
#  run_wizard Backward Compatibility Tests
# ===========================================================================

class TestRunWizardBackwardCompat:
    """Tests for run_wizard backward-compatible alias."""

    def test_run_wizard_with_tokens(self):
        from aeon_wizard import run_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        tokens = torch.randint(0, 100, (50, 128))
        result = run_wizard(
            model=model,
            tokens=tokens,
            config=config,
            device=torch.device("cpu"),
        )
        assert result["wizard_completed"] is True
        assert "overall_status" in result

    def test_run_wizard_without_tokens(self):
        from aeon_wizard import run_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        result = run_wizard(
            model=model,
            config=config,
            device=torch.device("cpu"),
        )
        assert result["wizard_completed"] is True

    def test_run_wizard_returns_architecture(self):
        from aeon_wizard import run_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        tokens = torch.randint(0, 100, (20, 128))
        result = run_wizard(
            model=model,
            tokens=tokens,
            config=config,
            device=torch.device("cpu"),
        )
        assert result.get("architecture") == "self_play_synthetic_curriculum"

    def test_run_wizard_signature_compat(self):
        """run_wizard should accept both old and new calling conventions."""
        from aeon_wizard import run_wizard, reset_wizard_state
        reset_wizard_state()
        model = _StubModel()
        config = _StubConfig()
        # Old-style call with positional tokens
        result = run_wizard(model, None, config)
        assert result["wizard_completed"] is True


# ===========================================================================
#  Self-Play Diagnostics Tests
# ===========================================================================

class TestSelfPlayDiagnostics:
    """Tests for run_self_play_diagnostics (§SP.2)."""

    def test_diagnostics_without_tokens(self):
        from aeon_wizard import run_self_play_diagnostics
        model = _StubModel()
        config = _StubConfig()
        result = run_self_play_diagnostics(
            model=model,
            config=config,
            device=torch.device("cpu"),
            num_synthetic=32,
        )
        # Either diagnosed or fell back gracefully
        assert "diagnosed" in result

    def test_legacy_corpus_diagnostics_delegates(self):
        from aeon_wizard import run_corpus_diagnostics
        model = _StubModel()
        config = _StubConfig()
        tokens = torch.randint(0, 100, (20, 128))
        result = run_corpus_diagnostics(
            model=model,
            tokens=tokens,
            config=config,
            device=torch.device("cpu"),
        )
        assert "diagnosed" in result

    def test_diagnostics_source_field(self):
        from aeon_wizard import run_self_play_diagnostics
        model = _StubModel()
        config = _StubConfig()
        result = run_self_play_diagnostics(
            model=model,
            config=config,
            num_synthetic=16,
        )
        if result.get("diagnosed"):
            assert result.get("source") == "self_play_synthetic"


# ===========================================================================
#  Initialize Codebook Updated Tests
# ===========================================================================

class TestInitializeCodebookUpdated:
    """Tests for updated initialize_codebook with synthetic bootstrap."""

    def test_codebook_init_no_tokens(self):
        from aeon_wizard import initialize_codebook
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        result = initialize_codebook(
            model=model,
            tokens=None,
            config=config,
            device=torch.device("cpu"),
        )
        assert "initialized" in result

    def test_codebook_init_with_tokens_ignored(self):
        from aeon_wizard import initialize_codebook
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        tokens = torch.randn(100, 128)
        result = initialize_codebook(
            model=model,
            tokens=tokens,
            config=config,
        )
        assert "initialized" in result

    def test_codebook_method_is_bootstrap(self):
        from aeon_wizard import initialize_codebook
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        result = initialize_codebook(
            model=model,
            config=config,
        )
        if result["initialized"]:
            assert "bootstrap" in result["method"] or "kmeans" in result["method"]


# ===========================================================================
#  Integration: Full Curriculum Cycle Test
# ===========================================================================

class TestFullCurriculumCycle:
    """Integration test for a full curriculum cycle."""

    def test_world_gen_plus_curriculum(self):
        from aeon_wizard import (
            LatentWorldGenerator,
            AdaptiveCurriculumManager,
            CorrectiveSynthesizer,
            GenerationMode,
        )
        gen = LatentWorldGenerator(latent_dim=32)
        mgr = AdaptiveCurriculumManager()
        synth = CorrectiveSynthesizer(latent_dim=32, num_corrective=4)
        config = _StubConfig()

        for _ in range(10):
            result = gen.generate(
                mode=mgr.generation_mode,
                batch_size=1,
                config=config,
            )
            success = result["complexity"].mean().item() > 0
            outcome = mgr.record_outcome(success=success)

            if not success:
                z = result["scenarios"].squeeze(0)
                corr = synth.synthesize(z, "direction")
                assert corr["num_generated"] == 4

        assert mgr.get_summary()["total_episodes"] == 10

    def test_meta_signaler_in_loop(self):
        from aeon_wizard import VibeThinkerMetaSignaler
        sig = VibeThinkerMetaSignaler()
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        for _ in range(5):
            sig.update()
            loss = sig.compute_loss(z1, z2)
            assert loss.item() > 0
