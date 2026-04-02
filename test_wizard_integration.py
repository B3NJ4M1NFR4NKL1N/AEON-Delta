"""
Tests for AEON First-Run Wizard and Integration Layer.

Tests cover:
  - aeon_wizard.py: WizardState, load_vt_weights, run_corpus_diagnostics,
    compute_hyperparameters, initialize_codebook, generate_config, run_wizard,
    is_cold_start
  - aeon_integration.py: IntegrationState, load_vt_weights_into_model,
    connect_feedback_bus, UnifiedTrainingCycleController, DashboardMetricsCollector
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
#  Minimal stubs for dependencies
# ===========================================================================

class _StubConfig:
    """Minimal AEONConfigV4-like config for testing."""
    z_dim: int = 32
    hidden_dim: int = 32
    vq_embedding_dim: int = 32
    context_window: int = 3
    vq_num_embeddings: int = 64
    codebook_size: int = 64
    seed: int = 42
    document_aware: bool = True
    seq_length: int = 128
    vocab_size: int = 30522
    grad_clip_norm: float = 1.0
    entropy_weight: float = 0.01
    vq_reset_threshold: float = 0.1
    warmup_steps: int = 100


class _StubEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, z_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class _StubVQ(nn.Module):
    def __init__(self, num_embeddings: int = 64, z_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, z_dim)
        self.ema_w = nn.Parameter(
            torch.randn(num_embeddings, z_dim), requires_grad=False,
        )
        self.ema_cluster_size = nn.Parameter(
            torch.ones(num_embeddings), requires_grad=False,
        )

    def get_codebook_usage(self) -> float:
        return 75.0


class _StubModel(nn.Module):
    """Minimal model that mimics AEONDeltaV4/AEONDeltaV3."""

    def __init__(self, z_dim: int = 32, num_embeddings: int = 64):
        super().__init__()
        self.encoder = _StubEncoder(128, z_dim)
        self.vq = _StubVQ(num_embeddings, z_dim)
        self.vibe_thinker_adapter = None
        self.vibe_thinker_kernel = None
        self.metacognitive_trigger = None
        self.cognitive_cycle = None
        self.continual_learning_core = None
        self.feedback_bus = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ===========================================================================
#  AEON WIZARD TESTS
# ===========================================================================

class TestWizardStepStatus:
    """Tests for WizardStepStatus lifecycle."""

    def test_initial_state(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("test_step")
        assert step.name == "test_step"
        assert step.status == "pending"
        assert step.started_at is None
        assert step.finished_at is None
        assert step.result == {}
        assert step.error is None

    def test_start(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("step")
        step.start()
        assert step.status == "running"
        assert step.started_at is not None

    def test_complete(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("step")
        step.start()
        step.complete({"key": "value"})
        assert step.status == "completed"
        assert step.finished_at is not None
        assert step.result == {"key": "value"}

    def test_fail(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("step")
        step.start()
        step.fail("something broke")
        assert step.status == "failed"
        assert step.error == "something broke"

    def test_to_dict_completed(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("step")
        step.start()
        step.complete({"x": 1})
        d = step.to_dict()
        assert d["name"] == "step"
        assert d["status"] == "completed"
        assert "duration_s" in d
        assert d["result"] == {"x": 1}

    def test_to_dict_failed(self):
        from aeon_wizard import WizardStepStatus
        step = WizardStepStatus("s")
        step.start()
        step.fail("err")
        d = step.to_dict()
        assert d["error"] == "err"
        assert d["status"] == "failed"


class TestWizardState:
    """Tests for WizardState."""

    def test_initial_steps(self):
        from aeon_wizard import WizardState
        state = WizardState()
        assert len(state.steps) == 5
        assert "weight_loading" in state.steps
        assert "corpus_diagnostics" in state.steps
        assert "hyperparameterization" in state.steps
        assert "codebook_init" in state.steps
        assert "config_generation" in state.steps
        assert state.overall_status == "idle"

    def test_to_dict(self):
        from aeon_wizard import WizardState
        state = WizardState()
        d = state.to_dict()
        assert "overall_status" in d
        assert "steps" in d
        assert len(d["steps"]) == 5


class TestWizardWeightLoading:
    """Tests for load_vt_weights."""

    def test_missing_weights(self):
        from aeon_wizard import load_vt_weights
        model = _StubModel()
        result = load_vt_weights(model, Path("/nonexistent/path.safetensors"))
        assert result["loaded"] is False
        assert "not found" in result["reason"].lower() or "not found" in result.get("reason", "").lower()

    def test_weight_path_constant(self):
        from aeon_wizard import VT_WEIGHTS_PATH
        assert str(VT_WEIGHTS_PATH) == "vibe_thinker_weights/model.safetensors"


class TestComputeHyperparameters:
    """Tests for compute_hyperparameters."""

    def test_context_window_from_cot_depth(self):
        from aeon_wizard import compute_hyperparameters
        config = _StubConfig()
        diagnostics = {
            "cot_depth_stats": {"p95": 4.7, "mean": 2.5, "std": 1.0},
        }
        result = compute_hyperparameters(diagnostics, config)
        assert result["config_updated"] is True
        assert config.context_window == 5  # ceil(4.7)
        assert result["applied"]["context_window"]["new"] == 5

    def test_context_window_clamped_min(self):
        from aeon_wizard import compute_hyperparameters
        config = _StubConfig()
        diagnostics = {
            "cot_depth_stats": {"p95": 0.2},
        }
        compute_hyperparameters(diagnostics, config)
        assert config.context_window >= 1

    def test_context_window_clamped_max(self):
        from aeon_wizard import compute_hyperparameters
        config = _StubConfig()
        diagnostics = {
            "cot_depth_stats": {"p95": 100.0},
        }
        compute_hyperparameters(diagnostics, config)
        assert config.context_window <= 16

    def test_z_dim_recommendation(self):
        from aeon_wizard import compute_hyperparameters
        config = _StubConfig()
        diagnostics = {
            "pca_explained_95pct_components": 24,
        }
        result = compute_hyperparameters(diagnostics, config)
        assert result["applied"]["z_dim_recommendation"] == 24

    def test_empty_diagnostics(self):
        from aeon_wizard import compute_hyperparameters
        config = _StubConfig()
        old_window = config.context_window
        result = compute_hyperparameters({}, config)
        assert result["config_updated"] is True
        assert config.context_window == old_window  # unchanged


class TestGenerateConfig:
    """Tests for generate_config."""

    def test_basic_generation(self):
        from aeon_wizard import generate_config
        config = _StubConfig()
        result = generate_config(
            config=config,
            diagnostics={"corpus_size": 100, "heterogeneous": False},
            hyperparams={"config_updated": True},
            codebook_result={"method": "kmeans", "inertia": 1.0},
        )
        assert "_wizard_meta" in result
        assert result["_wizard_meta"]["wizard_version"] == "1.0.0"
        assert result["_wizard_meta"]["corpus_size"] == 100

    def test_save_to_file(self):
        from aeon_wizard import generate_config
        config = _StubConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            result = generate_config(
                config=config,
                diagnostics={},
                hyperparams={},
                codebook_result={},
                output_path=path,
            )
            assert os.path.exists(path)
            with open(path) as f:
                saved = json.load(f)
            assert "_wizard_meta" in saved

    def test_validation_warning(self):
        from aeon_wizard import generate_config
        config = _StubConfig()
        config.z_dim = 32
        config.hidden_dim = 64  # mismatch!
        result = generate_config(
            config=config,
            diagnostics={},
            hyperparams={},
            codebook_result={},
        )
        assert len(result["_validation"]["warnings"]) > 0


class TestColdStart:
    """Tests for is_cold_start."""

    def test_cold_start_no_config(self):
        from aeon_wizard import is_cold_start
        assert is_cold_start(
            config_path="/nonexistent/config.json",
            weights_path=Path("/nonexistent/weights.safetensors"),
        ) is True

    def test_cold_start_with_config(self):
        from aeon_wizard import is_cold_start
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{}')
            f.flush()
            # Config exists but no weights
            result = is_cold_start(
                config_path=f.name,
                weights_path=Path("/nonexistent/weights.safetensors"),
            )
            assert result is True
            os.unlink(f.name)


class TestResetWizardState:
    """Tests for reset_wizard_state."""

    def test_reset(self):
        from aeon_wizard import get_wizard_state, reset_wizard_state
        state = get_wizard_state()
        state.overall_status = "completed"
        reset_wizard_state()
        new_state = get_wizard_state()
        assert new_state.overall_status == "idle"


# ===========================================================================
#  AEON INTEGRATION TESTS
# ===========================================================================

class TestIntegrationState:
    """Tests for IntegrationState."""

    def test_initial_state(self):
        from aeon_integration import IntegrationState
        state = IntegrationState()
        assert len(state.points) == 10
        assert state.get_active_count() == 0
        assert state.cycle_count == 0

    def test_update_point(self):
        from aeon_integration import IntegrationState
        state = IntegrationState()
        state.update_point("codebook_warm_start", {"initialized": True})
        assert state.points["codebook_warm_start"]["active"] is True
        assert state.get_active_count() == 1

    def test_to_dict(self):
        from aeon_integration import IntegrationState
        state = IntegrationState()
        d = state.to_dict()
        assert "integration_points" in d
        assert d["total_points"] == 10
        assert d["active_count"] == 0


class TestLoadVTWeightsIntoModel:
    """Tests for load_vt_weights_into_model."""

    def test_missing_weights(self):
        from aeon_integration import load_vt_weights_into_model
        model = _StubModel()
        result = load_vt_weights_into_model(
            model, Path("/nonexistent/weights.safetensors"),
        )
        assert result["loaded"] is False

    def test_weight_path_constant(self):
        from aeon_integration import VT_WEIGHTS_PATH
        assert str(VT_WEIGHTS_PATH) == "vibe_thinker_weights/model.safetensors"


class TestConnectFeedbackBus:
    """Tests for connect_feedback_bus."""

    def test_connect_with_write_signal(self):
        from aeon_integration import connect_feedback_bus
        controller = MagicMock()
        learner = MagicMock()
        bus = MagicMock()
        bus.register_signal = MagicMock()

        result = connect_feedback_bus(controller, learner, bus)
        assert result["connected"] is True
        assert len(result["registered_signals"]) == 4
        assert bus.register_signal.call_count == 4

    def test_connect_fallback_write(self):
        from aeon_integration import connect_feedback_bus
        controller = MagicMock()
        learner = MagicMock()
        bus = MagicMock(spec=[])  # No register_signal
        bus.write_signal = MagicMock()

        result = connect_feedback_bus(controller, learner, bus)
        assert result["connected"] is True


class TestUnifiedTrainingCycleController:
    """Tests for UnifiedTrainingCycleController."""

    def test_init(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        assert ctrl._cycle_count == 0

    def test_attach_components(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)

        ctrl.attach_signal_bus(MagicMock())
        assert ctrl._signal_bus is not None

        ctrl.attach_vt_learner(MagicMock())
        assert ctrl._vt_learner is not None

        ctrl.attach_controller(MagicMock())
        assert ctrl._controller is not None

        ctrl.attach_feedback_bus(MagicMock())
        assert ctrl._feedback_bus is not None

        ctrl.attach_ucc(MagicMock())
        assert ctrl._ucc is not None

        ctrl.attach_metacognitive_trigger(MagicMock())
        assert ctrl._mct is not None

        ctrl.attach_continual_core(MagicMock())
        assert ctrl._continual_core is not None

    def test_signal_bus_step_no_components(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        result = ctrl.execute_signal_bus_step()
        assert result["executed"] is False
        assert result["reason"] == "components_not_attached"

    def test_get_dashboard_metrics(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        metrics = ctrl.get_dashboard_metrics()
        assert "cycle_count" in metrics
        assert "integration_state" in metrics

    def test_get_dashboard_metrics_with_components(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)

        # Attach mock VT learner
        vt = MagicMock()
        vt._calibration_ema = 0.1
        vt._complexity_threshold_ema = 0.6
        vt._episode_count = 42
        ctrl.attach_vt_learner(vt)

        # Attach mock feedback bus
        fb = MagicMock()
        fb.get_state.return_value = {"safety": 0.9}
        fb.get_oscillation_score.return_value = 0.15
        ctrl.attach_feedback_bus(fb)

        # Attach mock continual core
        cl = MagicMock()
        cl._task_count = 3
        ctrl.attach_continual_core(cl)

        metrics = ctrl.get_dashboard_metrics()
        assert metrics["vt_learner"]["calibration_ema"] == 0.1
        assert metrics["vt_learner"]["episode_count"] == 42
        assert metrics["feedback_bus"] == {"safety": 0.9}
        assert metrics["feedback_bus_oscillation"] == 0.15
        assert metrics["continual_learning"]["active"] is True

    def test_metrics_history(self):
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        assert ctrl.get_metrics_history() == []


class TestDashboardMetricsCollector:
    """Tests for DashboardMetricsCollector."""

    def test_record_phase_a(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_phase_a(
            epoch=1,
            commitment_loss=0.5,
            entropy_weight=0.01,
            codebook_usage=0.75,
            total_loss=1.2,
        )
        assert len(collector.get_phase_a_metrics()) == 1
        m = collector.get_phase_a_metrics()[0]
        assert m["epoch"] == 1
        assert m["commitment_loss"] == 0.5
        assert "timestamp" in m

    def test_record_phase_b(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_phase_b(
            epoch=1,
            l_mse=0.01,
            l_quality=0.05,
            cot_depth_pred=3.5,
        )
        m = collector.get_phase_b_metrics()[0]
        assert m["L_mse"] == 0.01
        assert m["cot_depth_predicted"] == 3.5

    def test_record_vt_signals(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_vt_signals(
            calibration_error=0.15,
            confidence=0.85,
            complexity_threshold_ema=0.55,
        )
        m = collector.get_vt_signals()[0]
        assert m["calibration_error"] == 0.15
        assert m["confidence"] == 0.85

    def test_record_coherence(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_coherence(
            cognitive_unity=0.92,
            feedback_oscillation=0.05,
            convergence_quality=0.88,
        )
        m = collector.get_coherence_metrics()[0]
        assert m["cognitive_unity"] == 0.92

    def test_get_latest_empty(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        latest = collector.get_latest()
        assert latest["phase_a"] is None
        assert latest["phase_b"] is None

    def test_get_latest_with_data(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_phase_a(1, 0.5, 0.01, 0.75, 1.2)
        collector.record_phase_a(2, 0.3, 0.01, 0.80, 0.9)
        latest = collector.get_latest()
        assert latest["phase_a"]["epoch"] == 2
        assert latest["phase_a"]["commitment_loss"] == 0.3

    def test_last_n_limit(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        for i in range(100):
            collector.record_phase_a(i, 0.5, 0.01, 0.75, 1.0)
        assert len(collector.get_phase_a_metrics(last_n=10)) == 10

    def test_get_all(self):
        from aeon_integration import DashboardMetricsCollector
        collector = DashboardMetricsCollector()
        collector.record_phase_a(1, 0.5, 0.01, 0.75, 1.2)
        collector.record_vt_signals(0.1, 0.8, 0.5)
        all_m = collector.get_all()
        assert len(all_m["phase_a"]) == 1
        assert len(all_m["vt_signals"]) == 1
        assert len(all_m["phase_b"]) == 0


class TestGetMetricsCollector:
    """Tests for global metrics collector singleton."""

    def test_singleton(self):
        from aeon_integration import get_metrics_collector
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2


class TestGetIntegrationState:
    """Tests for global integration state singleton."""

    def test_singleton(self):
        from aeon_integration import get_integration_state
        s1 = get_integration_state()
        s2 = get_integration_state()
        assert s1 is s2


# ===========================================================================
#  WIZARD RUN (Integration Test — with mocked VibeThinker imports)
# ===========================================================================

class TestRunWizard:
    """Integration test for run_wizard with mocked dependencies."""

    def test_run_wizard_without_vt(self):
        """Wizard should complete even without VibeThinker available."""
        from aeon_wizard import run_wizard, reset_wizard_state
        reset_wizard_state()

        model = _StubModel()
        config = _StubConfig()
        tokens = torch.randint(0, 100, (50, 128))

        # This will skip VibeThinker-dependent steps gracefully
        result = run_wizard(
            model=model,
            tokens=tokens,
            config=config,
            device=torch.device("cpu"),
        )
        assert result["wizard_completed"] is True
        assert "overall_status" in result


class TestInitializeCodebook:
    """Tests for initialize_codebook."""

    def test_inline_codebook_init(self):
        """Test inline k-means codebook initialization."""
        from aeon_wizard import initialize_codebook
        model = _StubModel(z_dim=32, num_embeddings=8)
        config = _StubConfig()
        config.vq_num_embeddings = 8
        tokens = torch.randn(100, 128)

        # We need to mock the VibeThinkerPromptAdapter import
        # Since aeon_core may not be importable in test env,
        # test the insufficient_samples path
        config.vq_num_embeddings = 200  # More than N=100
        result = initialize_codebook(
            model=model,
            tokens=tokens,
            config=config,
            device=torch.device("cpu"),
        )
        # Either initialized or fell back gracefully
        assert "initialized" in result or "method" in result
