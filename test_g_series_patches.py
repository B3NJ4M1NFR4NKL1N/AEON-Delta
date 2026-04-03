"""Tests for G-series integration patches.

G1: Bridge failure recording to error_evolution
G2: Continuous MCT evaluation (always, not just on failure)
G3: Feedback bus cycle synchronisation
G4: UCC/SSP failure uncertainty flags
G5: Integration-point failure escalation to error_evolution
G6: Periodic mutual reinforcement via verify_and_reinforce
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ── Lightweight stubs ────────────────────────────────────────────────────────

class _StubConfig:
    """Minimal config stub with required attributes."""
    reinforce_interval = 2  # trigger every 2 cycles for testing


class _StubErrorEvolution:
    """Records episodes for testing."""

    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(
        self,
        error_class: str = "",
        strategy_used: str = "",
        success: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.episodes.append({
            "error_class": error_class,
            "strategy_used": strategy_used,
            "success": success,
            "metadata": metadata or {},
        })

    def get_episode_count(self) -> int:
        return len(self.episodes)

    def get_error_summary(self) -> Dict[str, Any]:
        return {"total_episodes": len(self.episodes)}


class _StubMCT:
    """Lightweight MCT that records evaluations."""

    def __init__(self) -> None:
        self.evaluations: List[Dict[str, Any]] = []

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        self.evaluations.append(kwargs)
        score = kwargs.get("uncertainty", 0.0)
        return {
            "should_trigger": score > 0.3,
            "trigger_score": score,
            "tightened_threshold": 0.5,
            "extra_iterations": 1 if score > 0.3 else 0,
            "triggers_active": ["uncertainty"] if score > 0.3 else [],
            "signal_weights": {},
            "dominant_module": None,
            "dominant_module_signal": None,
        }


class _StubFeedbackBus:
    """Records signal writes for testing."""

    def __init__(self) -> None:
        self.signals: Dict[str, float] = {}

    def write_signal(self, name: str, value: float) -> None:
        self.signals[name] = value

    def get_state(self) -> Dict[str, float]:
        return dict(self.signals)

    def get_oscillation_score(self) -> float:
        return 0.0


class _StubModel(nn.Module):
    """Minimal model with verify_and_reinforce."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.reinforce_calls = 0

    def verify_and_reinforce(self) -> Dict[str, Any]:
        self.reinforce_calls += 1
        return {"coherence_score": 0.95, "reinforced": True}


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ucc_controller():
    """Create a UnifiedTrainingCycleController with all stubs attached."""
    from aeon_integration import UnifiedTrainingCycleController
    model = _StubModel()
    cfg = _StubConfig()
    ctrl = UnifiedTrainingCycleController(model, cfg)
    return ctrl


# ═══════════════════════════════════════════════════════════════════════════
#  G1: Bridge failure recording to error_evolution
# ═══════════════════════════════════════════════════════════════════════════

class TestG1BridgeFailureRecording:
    """Bridge failures must be recorded to error_evolution."""

    def test_record_failure_episode_helper(self, ucc_controller):
        """_record_failure_episode writes to error_evolution."""
        ee = _StubErrorEvolution()
        ucc_controller._record_failure_episode(
            ee, "test_bridge_failure", "retry",
            {"reason": "unit test"},
        )
        assert len(ee.episodes) == 1
        assert ee.episodes[0]["error_class"] == "test_bridge_failure"
        assert ee.episodes[0]["success"] is False

    def test_record_failure_episode_none_noop(self, ucc_controller):
        """_record_failure_episode is a no-op when error_evolution is None."""
        # Should not raise
        ucc_controller._record_failure_episode(
            None, "test_class", "test_strategy",
        )

    def test_t2i_bridge_failure_recorded(self, ucc_controller):
        """Training→Inference bridge failure records to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Provide convergence_monitor + error_evolution so bridge is attempted
        mock_cm = MagicMock()
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            convergence_monitor=mock_cm,
            error_evolution=ee,
        )
        # The bridge will fail (import ae_train fails with missing module
        # or bridge function raises) — check that episode was recorded
        flags = result.get("uncertainty_flags", [])
        if "training_to_inference_bridge_failed" in flags:
            bridge_eps = [
                e for e in ee.episodes
                if "bridge" in e["error_class"]
            ]
            assert len(bridge_eps) >= 1, (
                "Bridge failure should record an episode"
            )

    def test_i2t_bridge_failure_recorded(self, ucc_controller):
        """Inference→Training bridge failure records to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        mock_trainer = MagicMock()
        mock_inf_ee = MagicMock()
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.5},
            error_evolution=ee,
            inference_error_evolution=mock_inf_ee,
            trainer=mock_trainer,
        )
        flags = result.get("uncertainty_flags", [])
        if "inference_to_training_bridge_failed" in flags:
            bridge_eps = [
                e for e in ee.episodes
                if "bridge" in e["error_class"]
            ]
            assert len(bridge_eps) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  G2: Continuous MCT evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TestG2ContinuousMCT:
    """MCT must be evaluated every cycle, not just on failures."""

    def test_mct_evaluated_on_successful_cycle(self, ucc_controller):
        """MCT.evaluate() is called even when no uncertainty flags exist."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Minimal cycle — no tokens, no z_sequences, no pseudo_labels
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.1},
        )

        # MCT should have been called regardless of failure count
        assert len(mct.evaluations) >= 1, (
            "MCT must be evaluated even on healthy cycles"
        )
        # Verify MCT was called with a numeric uncertainty value
        last_eval = mct.evaluations[-1]
        assert "uncertainty" in last_eval
        assert isinstance(last_eval["uncertainty"], float)

    def test_mct_result_includes_continuous_flag(self, ucc_controller):
        """Result includes continuous=True in metacognitive_review."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.2},
        )

        review = result.get("metacognitive_review", {})
        assert review.get("continuous") is True

    def test_mct_triggered_on_high_uncertainty(self, ucc_controller):
        """MCT trigger fires when many integration points fail."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Provide tokens to trigger points 2 & 3 (which will fail)
        tokens = torch.randn(4, 16)
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.9},
            tokens=tokens,
        )

        flags = result.get("uncertainty_flags", [])
        review = result.get("metacognitive_review", {})
        # If enough flags accumulated, MCT should trigger
        if len(flags) >= 4:
            assert review.get("triggered") is True

    def test_mct_not_triggered_on_zero_uncertainty(self, ucc_controller):
        """MCT should NOT trigger when everything is healthy."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.05},
        )

        review = result.get("metacognitive_review", {})
        # With uncertainty=0.0, our stub MCT won't trigger (threshold > 0.3)
        assert review.get("triggered") is False


# ═══════════════════════════════════════════════════════════════════════════
#  G3: Feedback bus cycle sync
# ═══════════════════════════════════════════════════════════════════════════

class TestG3FeedbackBusSync:
    """Cycle health must be written to feedback bus after each cycle."""

    def test_integration_health_written(self, ucc_controller):
        """integration_health signal is written to feedback bus."""
        fb = _StubFeedbackBus()
        mct = _StubMCT()
        ucc_controller.attach_feedback_bus(fb)
        ucc_controller.attach_metacognitive_trigger(mct)

        ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.3},
        )

        assert "integration_health" in fb.signals
        assert 0.0 <= fb.signals["integration_health"] <= 1.0

    def test_integration_failure_rate_written(self, ucc_controller):
        """integration_failure_rate signal is written."""
        fb = _StubFeedbackBus()
        mct = _StubMCT()
        ucc_controller.attach_feedback_bus(fb)
        ucc_controller.attach_metacognitive_trigger(mct)

        ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.3},
        )

        assert "integration_failure_rate" in fb.signals

    def test_ucc_evaluation_ok_written(self, ucc_controller):
        """ucc_evaluation_ok signal is written after UCC runs."""
        fb = _StubFeedbackBus()
        mct = _StubMCT()
        ucc_controller.attach_feedback_bus(fb)
        ucc_controller.attach_metacognitive_trigger(mct)

        ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.3},
        )

        assert "ucc_evaluation_ok" in fb.signals

    def test_no_feedback_bus_no_error(self, ucc_controller):
        """Cycle runs cleanly without feedback bus attached."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Should not raise
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.3},
        )
        assert "uncertainty_flags" in result

    def test_healthy_cycle_health_is_high(self, ucc_controller):
        """Healthy cycle should produce health close to 1.0."""
        fb = _StubFeedbackBus()
        mct = _StubMCT()
        ucc_controller.attach_feedback_bus(fb)
        ucc_controller.attach_metacognitive_trigger(mct)

        ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.1},
        )

        # No tokens or z_sequences → fewer integration points attempted
        # UCC/SSP may fail but health should be reasonable
        assert fb.signals["integration_health"] >= 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  G4: UCC/SSP failure uncertainty flags
# ═══════════════════════════════════════════════════════════════════════════

class TestG4UCCSSPFailureFlags:
    """UCC and SSP failures must appear as uncertainty flags."""

    def test_ucc_failure_flag_present(self, ucc_controller):
        """UCC evaluation failure creates an uncertainty flag."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.5},
        )

        ucc = result.get("ucc", {})
        flags = result.get("uncertainty_flags", [])
        # If UCC failed (evaluated=False), flag should be present
        if not ucc.get("evaluated", True):
            assert "ucc_evaluation_failed" in flags

    def test_ssp_failure_flag_present(self, ucc_controller):
        """SSP alignment failure creates an uncertainty flag."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
        )

        ssp = result.get("ssp", {})
        flags = result.get("uncertainty_flags", [])
        # If SSP failed (aligned=False), flag should be present
        if not ssp.get("aligned", True):
            assert "ssp_alignment_failed" in flags

    def test_ucc_failure_recorded_to_ee(self, ucc_controller):
        """UCC failure records an episode to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.5},
            error_evolution=ee,
        )

        ucc = result.get("ucc", {})
        if not ucc.get("evaluated", True):
            ucc_eps = [
                e for e in ee.episodes
                if "ucc" in e["error_class"]
            ]
            assert len(ucc_eps) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  G5: Integration-point failure escalation
# ═══════════════════════════════════════════════════════════════════════════

class TestG5IntegrationPointFailureEscalation:
    """Integration-point failures must be recorded to error_evolution."""

    def test_codebook_failure_recorded(self, ucc_controller):
        """Codebook warm-start failure records to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        tokens = torch.randn(4, 16)
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            tokens=tokens,
            error_evolution=ee,
        )

        if not result.get("codebook_warm_start", {}).get("initialized", True):
            codebook_eps = [
                e for e in ee.episodes
                if "codebook" in e["error_class"]
            ]
            assert len(codebook_eps) >= 1

    def test_context_calibration_failure_recorded(self, ucc_controller):
        """Context calibration failure records to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        tokens = torch.randn(4, 16)
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            tokens=tokens,
            error_evolution=ee,
        )

        if not result.get("context_calibration", {}).get("calibrated", True):
            cal_eps = [
                e for e in ee.episodes
                if "calibration" in e["error_class"]
            ]
            assert len(cal_eps) >= 1

    def test_teacher_student_failure_recorded(self, ucc_controller):
        """Teacher-student inversion failure records to error_evolution."""
        ee = _StubErrorEvolution()
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        z = [torch.randn(8, 64)]
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            z_sequences=z,
            error_evolution=ee,
        )

        if not result.get("teacher_student_inversion", {}).get(
            "inverted", True,
        ):
            inv_eps = [
                e for e in ee.episodes
                if "teacher_student" in e["error_class"]
            ]
            assert len(inv_eps) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  G6: Periodic mutual reinforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestG6MutualReinforcement:
    """verify_and_reinforce must be called periodically."""

    def test_reinforce_called_at_interval(self, ucc_controller):
        """verify_and_reinforce called every reinforce_interval cycles."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        model = ucc_controller.model
        assert isinstance(model, _StubModel)

        # reinforce_interval=2, so should trigger on cycle 2, 4, 6...
        for i in range(4):
            result = ucc_controller.execute_full_cycle(
                epoch=i + 1, phase="B",
                epoch_metrics={"loss": 0.3},
            )

        # After 4 cycles, should have been called twice (cycle 2 and 4)
        assert model.reinforce_calls == 2

    def test_reinforce_result_in_cycle_results(self, ucc_controller):
        """mutual_reinforcement key present when verify_and_reinforce runs."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Run 2 cycles (interval=2)
        for _ in range(2):
            result = ucc_controller.execute_full_cycle(
                epoch=1, phase="B",
                epoch_metrics={"loss": 0.3},
            )

        reinforce = result.get("mutual_reinforcement", {})
        assert reinforce.get("executed") is True

    def test_reinforce_not_called_when_not_due(self, ucc_controller):
        """verify_and_reinforce NOT called on non-interval cycles."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        model = ucc_controller.model
        result = ucc_controller.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.3},
        )

        assert model.reinforce_calls == 0
        assert "mutual_reinforcement" not in result

    def test_model_without_reinforce_no_error(self):
        """Models without verify_and_reinforce don't cause errors."""
        from aeon_integration import UnifiedTrainingCycleController

        plain_model = nn.Linear(4, 4)
        cfg = _StubConfig()
        ctrl = UnifiedTrainingCycleController(plain_model, cfg)
        mct = _StubMCT()
        ctrl.attach_metacognitive_trigger(mct)

        # Should not raise even though model has no verify_and_reinforce
        for _ in range(3):
            result = ctrl.execute_full_cycle(
                epoch=1, phase="B",
                epoch_metrics={"loss": 0.3},
            )
        assert "mutual_reinforcement" not in result

    def test_reinforce_failure_recorded(self, ucc_controller):
        """Failed verify_and_reinforce is captured in cycle_results."""
        mct = _StubMCT()
        ucc_controller.attach_metacognitive_trigger(mct)

        # Make verify_and_reinforce raise
        model = ucc_controller.model
        model.verify_and_reinforce = lambda: (_ for _ in ()).throw(
            RuntimeError("test failure"),
        )

        # Run 2 cycles to hit interval
        for _ in range(2):
            result = ucc_controller.execute_full_cycle(
                epoch=1, phase="B",
                epoch_metrics={"loss": 0.3},
            )

        reinforce = result.get("mutual_reinforcement", {})
        assert reinforce.get("executed") is False
        assert "test failure" in reinforce.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
#  End-to-End: Full cognitive cycle with all patches
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndCognitiveActivation:
    """Verify that all patches work together in a full cycle."""

    def test_full_cycle_with_all_patches(self):
        """Execute a full cycle with MCT, feedback bus, error evolution."""
        from aeon_integration import UnifiedTrainingCycleController

        model = _StubModel()
        cfg = _StubConfig()
        cfg.reinforce_interval = 1  # every cycle

        ctrl = UnifiedTrainingCycleController(model, cfg)
        mct = _StubMCT()
        fb = _StubFeedbackBus()
        ee = _StubErrorEvolution()

        ctrl.attach_metacognitive_trigger(mct)
        ctrl.attach_feedback_bus(fb)

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.3},
            error_evolution=ee,
        )

        # G2: MCT was evaluated continuously
        assert len(mct.evaluations) >= 1
        assert "metacognitive_review" in result
        assert result["metacognitive_review"]["continuous"] is True

        # G3: Feedback bus was synced
        assert "integration_health" in fb.signals

        # G6: Mutual reinforcement was called (interval=1)
        assert model.reinforce_calls >= 1
        assert result.get("mutual_reinforcement", {}).get("executed")

    def test_causal_traceability(self):
        """Verify causal trace from output back to premise."""
        from aeon_integration import (
            UnifiedTrainingCycleController,
            trace_output_to_premise,
        )

        model = _StubModel()
        cfg = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, cfg)
        mct = _StubMCT()
        ee = _StubErrorEvolution()

        ctrl.attach_metacognitive_trigger(mct)

        result = ctrl.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.2},
            error_evolution=ee,
        )

        # Trace any output back
        trace = trace_output_to_premise(
            "test_output",
            cycle_history=ctrl.get_metrics_history(),
            error_evolution=ee,
        )
        assert trace["traced"] is True
        assert len(trace["trace_chain"]) >= 2

    def test_multiple_cycles_cumulative(self):
        """Multiple cycles accumulate MCT evaluations and reinforcements."""
        from aeon_integration import UnifiedTrainingCycleController

        model = _StubModel()
        cfg = _StubConfig()
        cfg.reinforce_interval = 2

        ctrl = UnifiedTrainingCycleController(model, cfg)
        mct = _StubMCT()
        fb = _StubFeedbackBus()
        ee = _StubErrorEvolution()

        ctrl.attach_metacognitive_trigger(mct)
        ctrl.attach_feedback_bus(fb)

        for epoch in range(1, 7):
            ctrl.execute_full_cycle(
                epoch=epoch, phase="B",
                epoch_metrics={"loss": max(0.01, 0.5 - epoch * 0.05)},
                error_evolution=ee,
            )

        # G2: MCT evaluated 6 times
        assert len(mct.evaluations) == 6

        # G6: Reinforcement called 3 times (cycles 2, 4, 6)
        assert model.reinforce_calls == 3

        # Metrics history has 6 entries
        assert len(ctrl.get_metrics_history()) == 6
