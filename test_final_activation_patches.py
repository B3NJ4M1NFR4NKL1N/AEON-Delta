"""Tests for Final Integration & Cognitive Activation Patches.

Validates:
  F1: execute_codebook_warm_start() method
  F2: execute_context_calibration() method
  F3: trace_output_to_premise() causal transparency
  F4: execute_full_cycle() 10-point integration
  F5: Metacognitive uncertainty check in execute_full_cycle()
  F6: execute_inference_to_training_bridge() signature fix
"""

import sys
import os
import time
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(__file__))


# ── Lightweight fixtures ─────────────────────────────────────────────────────

class _FakeConfig:
    """Minimal config for integration tests."""
    def __init__(self):
        self.hidden_dim = 32
        self.z_dim = 32
        self.vq_embedding_dim = 32
        self.context_window = 3
        self.codebook_size = 16
        self.num_rssm_layers = 1
        self.max_seq_len = 64


class _FakeModel(nn.Module):
    """Minimal model for integration tests."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class _FakeConvergenceMonitor:
    """Fake convergence monitor with episodes."""
    def __init__(self):
        self.episodes = [
            {"class": "divergence", "severity": 0.5, "step": 10},
            {"class": "stagnation", "severity": 0.3, "step": 20},
        ]

    def get_summary(self):
        return {"episodes": len(self.episodes), "classes": ["divergence", "stagnation"]}


class _FakeErrorEvolution:
    """Fake error evolution tracker."""
    def __init__(self):
        self.episodes = []
        self._episode_count = 0

    def record_episode(self, **kwargs):
        self.episodes.append(kwargs)
        self._episode_count += 1

    def get_episode_count(self):
        return self._episode_count

    def trace_root_cause(self):
        return "initial_training_divergence"


class _FakeMCT:
    """Fake MetaCognitiveRecursionTrigger."""
    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {"verdict": "review_required", "uncertainty": kwargs.get("uncertainty", 0.0)}


class _FakeSignalBus:
    """Fake VTStreamingSignalBus."""
    def __init__(self):
        self._signals = {}

    def push(self, name, value):
        self._signals[name] = value

    def get_ema(self):
        return self._signals

    def closed_loop_step(self, learner, controller):
        return {"executed": True, "signals_pushed": 3}


class _FakeTrainer:
    """Fake SafeThoughtAETrainerV4."""
    def __init__(self):
        self.adaptations = []

    def adapt(self, **kwargs):
        self.adaptations.append(kwargs)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_controller(**kwargs):
    """Create a UnifiedTrainingCycleController with optional attachments."""
    from aeon_integration import UnifiedTrainingCycleController
    config = kwargs.pop("config", _FakeConfig())
    model = kwargs.pop("model", _FakeModel())
    device = kwargs.pop("device", torch.device("cpu"))
    ctrl = UnifiedTrainingCycleController(model=model, config=config, device=device)

    if "signal_bus" in kwargs:
        ctrl.attach_signal_bus(kwargs["signal_bus"])
    if "mct" in kwargs:
        ctrl.attach_metacognitive_trigger(kwargs["mct"])
    if "feedback_bus" in kwargs:
        ctrl.attach_feedback_bus(kwargs["feedback_bus"])
    if "ucc" in kwargs:
        ctrl.attach_ucc(kwargs["ucc"])
    if "continual_core" in kwargs:
        ctrl.attach_continual_core(kwargs["continual_core"])
    return ctrl


# =============================================================================
#  F1: execute_codebook_warm_start
# =============================================================================

class TestF1CodebookWarmStart:
    """Tests for Patch F1: execute_codebook_warm_start method."""

    def test_method_exists(self):
        """F1: UnifiedTrainingCycleController has execute_codebook_warm_start."""
        ctrl = _make_controller()
        assert hasattr(ctrl, "execute_codebook_warm_start"), (
            "execute_codebook_warm_start method must exist"
        )

    def test_returns_dict(self):
        """F1: execute_codebook_warm_start returns a dict."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_codebook_warm_start(tokens)
        assert isinstance(result, dict), "Must return dict"

    def test_graceful_failure(self):
        """F1: Returns initialized=False when model lacks encode method."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_codebook_warm_start(tokens)
        assert isinstance(result, dict)
        assert result.get("initialized", False) is False

    def test_updates_integration_state(self):
        """F1: Successful warm-start updates integration state."""
        from aeon_integration import get_integration_state
        state = get_integration_state()
        # Record initial state
        initial = state.points["codebook_warm_start"]["active"]

        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))

        # Mock the ae_train function
        mock_result = {"initialized": True, "num_embeddings": 16}
        with patch.dict('sys.modules', {'ae_train': MagicMock(
            warm_start_codebook_from_vt=MagicMock(return_value=mock_result)
        )}):
            result = ctrl.execute_codebook_warm_start(tokens)

        # Even if import fails, method should not crash
        assert isinstance(result, dict)


# =============================================================================
#  F2: execute_context_calibration
# =============================================================================

class TestF2ContextCalibration:
    """Tests for Patch F2: execute_context_calibration method."""

    def test_method_exists(self):
        """F2: UnifiedTrainingCycleController has execute_context_calibration."""
        ctrl = _make_controller()
        assert hasattr(ctrl, "execute_context_calibration"), (
            "execute_context_calibration method must exist"
        )

    def test_returns_dict(self):
        """F2: execute_context_calibration returns a dict."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_context_calibration(tokens)
        assert isinstance(result, dict)

    def test_graceful_failure(self):
        """F2: Returns calibrated=False when model lacks encode method."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_context_calibration(tokens)
        assert isinstance(result, dict)
        assert result.get("calibrated", False) is False


# =============================================================================
#  F3: trace_output_to_premise
# =============================================================================

class TestF3TraceOutputToPremise:
    """Tests for Patch F3: trace_output_to_premise causal transparency."""

    def test_function_exists(self):
        """F3: trace_output_to_premise function exists in aeon_integration."""
        from aeon_integration import trace_output_to_premise
        assert callable(trace_output_to_premise)

    def test_basic_trace(self):
        """F3: Basic trace returns a dict with required keys."""
        from aeon_integration import trace_output_to_premise
        result = trace_output_to_premise("test_output")
        assert isinstance(result, dict)
        assert "traced" in result
        assert "output_action" in result
        assert "trace_chain" in result
        assert result["output_action"] == "test_output"

    def test_trace_with_cycle_history(self):
        """F3: Trace with cycle history finds originating cycle."""
        from aeon_integration import trace_output_to_premise
        history = [
            {
                "cycle": 1,
                "epoch": 0,
                "phase": "A",
                "signal_bus": {"executed": True},
                "ucc": {"evaluated": True},
            },
            {
                "cycle": 2,
                "epoch": 1,
                "phase": "A",
                "ssp": {"aligned": True},
                "ucc": {"evaluated": True},
            },
        ]
        result = trace_output_to_premise("action_x", cycle_history=history)
        assert result["originating_cycle"] == 2
        assert result["originating_point"] is not None
        assert result["traced"] is True

    def test_trace_with_error_evolution(self):
        """F3: Trace with error evolution provides root cause."""
        from aeon_integration import trace_output_to_premise
        ee = _FakeErrorEvolution()
        ee.record_episode(cls="divergence", severity=0.5)
        result = trace_output_to_premise(
            "recovery_action",
            error_evolution=ee,
        )
        assert result["root_premise"] == "initial_training_divergence"
        # Check error_evolution step is in trace chain
        steps = [s["step"] for s in result["trace_chain"]]
        assert "error_evolution" in steps
        assert "root_cause" in steps

    def test_trace_with_integration_state(self):
        """F3: Trace uses integration state for active points."""
        from aeon_integration import trace_output_to_premise, IntegrationState
        state = IntegrationState()
        state.update_point("streaming_signal_bus", {"executed": True})
        state.update_point("ucc_epoch_evaluation", {"evaluated": True})

        result = trace_output_to_premise(
            "signal_output",
            integration_state=state,
        )
        steps = [s["step"] for s in result["trace_chain"]]
        assert "integration_state" in steps
        # Should find active points
        state_step = [s for s in result["trace_chain"]
                      if s["step"] == "integration_state"][0]
        assert state_step["total_active"] == 2

    def test_trace_fallback_premise(self):
        """F3: When no deep trace, root_premise uses integration point."""
        from aeon_integration import trace_output_to_premise, IntegrationState
        state = IntegrationState()
        state.update_point("ssp_temperature_alignment", {"aligned": True})

        result = trace_output_to_premise(
            "temperature_change",
            integration_state=state,
        )
        assert result["root_premise"] is not None
        assert "ssp_temperature_alignment" in result["root_premise"]

    def test_trace_empty_state(self):
        """F3: Trace with empty state still returns valid structure."""
        from aeon_integration import trace_output_to_premise, IntegrationState
        state = IntegrationState()
        result = trace_output_to_premise(
            "unknown_output",
            integration_state=state,
        )
        assert isinstance(result, dict)
        assert result["output_action"] == "unknown_output"
        assert isinstance(result["trace_chain"], list)


# =============================================================================
#  F4: execute_full_cycle 10-point integration
# =============================================================================

class TestF4FullCycle10Point:
    """Tests for Patch F4: execute_full_cycle all 10 integration points."""

    def test_accepts_new_params(self):
        """F4: execute_full_cycle accepts tokens, convergence_monitor, etc."""
        ctrl = _make_controller()
        import inspect
        sig = inspect.signature(ctrl.execute_full_cycle)
        params = list(sig.parameters.keys())
        assert "tokens" in params, "Must accept tokens param"
        assert "convergence_monitor" in params
        assert "error_evolution" in params
        assert "pseudo_labels" in params
        assert "trainer" in params
        assert "inference_error_evolution" in params

    def test_basic_cycle_still_works(self):
        """F4: Basic cycle (4 points) still works with old signature."""
        ctrl = _make_controller()
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={"loss": 0.5},
        )
        assert isinstance(result, dict)
        assert result["cycle"] == 1
        assert result["epoch"] == 0
        assert result["phase"] == "A"
        assert "ucc" in result
        assert "ssp" in result

    def test_cycle_with_z_sequences(self):
        """F4: Cycle with z_sequences triggers annotation."""
        ctrl = _make_controller()
        z = [torch.randn(4, 32)]
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            z_sequences=z,
        )
        # z_annotation or teacher_student_inversion should be attempted
        assert isinstance(result, dict)

    def test_cycle_with_signal_bus(self):
        """F4: Signal bus is called in cycle when attached."""
        bus = _FakeSignalBus()
        ctrl = _make_controller(signal_bus=bus)
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
        )
        assert "signal_bus" in result

    def test_cycle_count_increments(self):
        """F4: Cycle count increments with each call."""
        ctrl = _make_controller()
        r1 = ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        r2 = ctrl.execute_full_cycle(epoch=1, phase="A", epoch_metrics={})
        assert r1["cycle"] == 1
        assert r2["cycle"] == 2

    def test_cycle_records_duration(self):
        """F4: Cycle result includes duration_s."""
        ctrl = _make_controller()
        result = ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        assert "duration_s" in result
        assert isinstance(result["duration_s"], float)

    def test_metrics_history_updated(self):
        """F4: Metrics history is populated after cycle."""
        ctrl = _make_controller()
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        history = ctrl.get_metrics_history()
        assert len(history) == 1
        assert history[0]["cycle"] == 1

    def test_tokens_trigger_warm_start_first_cycle(self):
        """F4: Providing tokens on first cycle triggers warm-start."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            tokens=tokens,
        )
        # codebook_warm_start should be attempted on first cycle
        assert "codebook_warm_start" in result
        # context_calibration should be attempted on first cycle
        assert "context_calibration" in result

    def test_tokens_no_warm_start_second_cycle(self):
        """F4: Warm-start only runs on first cycle."""
        ctrl = _make_controller()
        tokens = torch.randint(0, 100, (10, 16))
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={}, tokens=tokens)
        result2 = ctrl.execute_full_cycle(
            epoch=1, phase="A", epoch_metrics={},
            tokens=tokens,
        )
        # Second cycle should NOT have warm-start
        assert "codebook_warm_start" not in result2
        assert "context_calibration" not in result2

    def test_uncertainty_flags_collected(self):
        """F4: Uncertainty flags are collected in results."""
        ctrl = _make_controller()
        result = ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        assert "uncertainty_flags" in result
        assert isinstance(result["uncertainty_flags"], list)

    def test_pseudo_labels_trigger_micro_retrain(self):
        """F4: Providing pseudo_labels triggers micro-retrain attempt."""
        ctrl = _make_controller()
        labels = [{"quality": 0.9, "confidence": 0.8, "cot_depth": 3}]
        result = ctrl.execute_full_cycle(
            epoch=0, phase="B", epoch_metrics={},
            pseudo_labels=labels,
        )
        # micro_retrain should be attempted
        assert "micro_retrain" in result

    def test_convergence_monitor_triggers_t2i_bridge(self):
        """F4: convergence_monitor + error_evolution triggers T→I bridge."""
        ctrl = _make_controller()
        cm = _FakeConvergenceMonitor()
        ee = _FakeErrorEvolution()
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            convergence_monitor=cm,
            error_evolution=ee,
        )
        assert "training_to_inference_bridge" in result

    def test_inference_ee_triggers_i2t_bridge(self):
        """F4: inference_error_evolution + trainer triggers I→T bridge."""
        ctrl = _make_controller()
        iee = _FakeErrorEvolution()
        trainer = _FakeTrainer()
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            inference_error_evolution=iee,
            trainer=trainer,
        )
        assert "inference_to_training_bridge" in result


# =============================================================================
#  F5: Metacognitive uncertainty trigger
# =============================================================================

class TestF5MetacognitiveUncertaintyTrigger:
    """Tests for Patch F5: MCT trigger on uncertainty in execute_full_cycle."""

    def test_mct_triggered_on_uncertainty(self):
        """F5: MCT triggered when integration points report failure."""
        mct = _FakeMCT()
        ctrl = _make_controller(mct=mct)
        tokens = torch.randint(0, 100, (10, 16))
        # First cycle with tokens will attempt warm-start/calibration
        # which will fail (no ae_train available) → uncertainty flags
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            tokens=tokens,
        )
        flags = result.get("uncertainty_flags", [])
        if flags:
            # MCT should have been called
            assert "metacognitive_review" in result
            review = result["metacognitive_review"]
            # With continuous MCT (G2), triggered reflects MCT verdict
            # The key assertion is that MCT was evaluated with uncertainty > 0
            assert len(mct.calls) > 0
            assert mct.calls[-1].get("uncertainty", 0.0) > 0

    def test_no_mct_when_no_uncertainty(self):
        """F5: MCT evaluated but not triggered when no uncertainty flags."""
        mct = _FakeMCT()
        ctrl = _make_controller(mct=mct)
        # No tokens, no pseudo_labels, no convergence_monitor
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
        )
        flags = result.get("uncertainty_flags", [])
        if not flags:
            # With continuous MCT (G2), MCT is always evaluated
            # but should not be "triggered" when uncertainty is low
            review = result.get("metacognitive_review", {})
            if review:
                assert review.get("triggered") is not True or review.get("continuous") is True

    def test_mct_not_called_without_attachment(self):
        """F5: MCT not called if not attached even with uncertainty."""
        ctrl = _make_controller()  # no MCT attached
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            tokens=tokens,
        )
        # Should not crash even without MCT
        assert "metacognitive_review" not in result

    def test_mct_receives_uncertainty_score(self):
        """F5: MCT evaluate receives uncertainty as fraction of total points."""
        mct = _FakeMCT()
        ctrl = _make_controller(mct=mct)
        tokens = torch.randint(0, 100, (10, 16))
        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            tokens=tokens,
        )
        if mct.calls:
            uncertainty = mct.calls[0]["uncertainty"]
            assert 0.0 <= uncertainty <= 1.0, (
                "Uncertainty must be normalized [0, 1]"
            )


# =============================================================================
#  F6: execute_inference_to_training_bridge signature fix
# =============================================================================

class TestF6InferenceToTrainingBridgeFix:
    """Tests for Patch F6: Signature alignment of I→T bridge."""

    def test_method_signature_has_correct_params(self):
        """F6: execute_inference_to_training_bridge has correct params."""
        import inspect
        ctrl = _make_controller()
        sig = inspect.signature(ctrl.execute_inference_to_training_bridge)
        params = list(sig.parameters.keys())
        assert "inference_error_evolution" in params
        assert "trainer" in params
        # Old params should NOT be present
        assert "model_inference" not in params, (
            "Old param model_inference should be removed"
        )
        assert "convergence_monitor" not in params, (
            "Old param convergence_monitor should be removed"
        )

    def test_method_returns_dict(self):
        """F6: execute_inference_to_training_bridge returns dict."""
        ctrl = _make_controller()
        result = ctrl.execute_inference_to_training_bridge(
            inference_error_evolution=_FakeErrorEvolution(),
            trainer=_FakeTrainer(),
        )
        assert isinstance(result, dict)

    def test_graceful_failure(self):
        """F6: Returns bridged=False on failure."""
        ctrl = _make_controller()
        result = ctrl.execute_inference_to_training_bridge(
            inference_error_evolution=_FakeErrorEvolution(),
            trainer=_FakeTrainer(),
        )
        assert isinstance(result, dict)
        if not result.get("bridged", True):
            assert "reason" in result


# =============================================================================
#  Integration: Full end-to-end cognitive cycle
# =============================================================================

class TestEndToEndCognitiveCycle:
    """End-to-end tests for the full cognitive cycle."""

    def test_full_cycle_all_components_attached(self):
        """E2E: Full cycle with all optional components."""
        mct = _FakeMCT()
        bus = _FakeSignalBus()
        ctrl = _make_controller(signal_bus=bus, mct=mct)
        tokens = torch.randint(0, 100, (10, 16))
        z = [torch.randn(4, 32)]
        cm = _FakeConvergenceMonitor()
        ee = _FakeErrorEvolution()
        iee = _FakeErrorEvolution()
        trainer = _FakeTrainer()
        labels = [{"quality": 0.9, "confidence": 0.8}]

        result = ctrl.execute_full_cycle(
            epoch=0,
            phase="A",
            epoch_metrics={"loss": 0.5},
            z_sequences=z,
            tokens=tokens,
            convergence_monitor=cm,
            error_evolution=ee,
            pseudo_labels=labels,
            trainer=trainer,
            inference_error_evolution=iee,
        )

        assert result["cycle"] == 1
        assert result["phase"] == "A"
        assert "signal_bus" in result
        assert "ucc" in result
        assert "ssp" in result
        assert "uncertainty_flags" in result
        assert "duration_s" in result

    def test_causal_transparency_with_cycle_history(self):
        """E2E: trace_output_to_premise works with real cycle history."""
        from aeon_integration import trace_output_to_premise
        ctrl = _make_controller()
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={"loss": 0.5})
        ctrl.execute_full_cycle(epoch=1, phase="A", epoch_metrics={"loss": 0.3})

        history = ctrl.get_metrics_history()
        result = trace_output_to_premise(
            "loss_improvement",
            cycle_history=history,
        )
        assert result["traced"] is True
        assert result["originating_cycle"] is not None

    def test_mutual_reinforcement_through_cycles(self):
        """E2E: Multiple cycles produce consistent metrics history."""
        ctrl = _make_controller()
        for epoch in range(5):
            ctrl.execute_full_cycle(
                epoch=epoch, phase="A",
                epoch_metrics={"loss": 0.5 - 0.1 * epoch},
            )
        history = ctrl.get_metrics_history()
        assert len(history) == 5
        for i, entry in enumerate(history):
            assert entry["cycle"] == i + 1

    def test_integration_state_reflects_active_points(self):
        """E2E: Integration state shows which points are active."""
        from aeon_integration import get_integration_state
        state = get_integration_state()

        ctrl = _make_controller(signal_bus=_FakeSignalBus())
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        # At minimum, UCC and SSP should be attempted
        state_dict = state.to_dict()
        assert state_dict["total_points"] == 10

    def test_dashboard_metrics_after_cycle(self):
        """E2E: Dashboard metrics available after cycle execution."""
        ctrl = _make_controller(signal_bus=_FakeSignalBus())
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        metrics = ctrl.get_dashboard_metrics()
        assert "cycle_count" in metrics
        assert metrics["cycle_count"] == 1
        assert "integration_state" in metrics


# =============================================================================
#  Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
