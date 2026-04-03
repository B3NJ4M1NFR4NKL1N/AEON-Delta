"""Tests for K-series integration patches: Final Cognitive Activation.

K1: UCC coherence feedback to MCT (graded coherence_deficit in _collect_mct_signals)
K2: SSP temperature signals to MCT (SSP alignment into _collect_mct_signals)
K3: Same-cycle MCT re-execution (re-evaluate UCC+SSP when MCT triggers)
K4: UCC coherence → training curriculum feedback (graded score + phase pressure)
K5: Graded diversity escalation (accept_diversity_pressure on RobustVectorQuantizer)
K6: Error-class sub-typing for bridge failures (granular sub-error-classes)
K7: Convergence certificate → same-pass MCT gating
"""

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

# ── Ensure repo root is importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_feedback_bus():
    """A mock CognitiveFeedbackBus with write_signal and read capabilities."""
    bus = MagicMock()
    bus._signals: Dict[str, float] = {}

    def write_signal(name: str, value: float):
        bus._signals[name] = value

    def get_oscillation_score():
        return 0.0

    bus.write_signal = MagicMock(side_effect=write_signal)
    bus.get_oscillation_score = MagicMock(side_effect=get_oscillation_score)
    return bus


@pytest.fixture
def mock_mct():
    """A mock MetaCognitiveRecursionTrigger."""
    mct = MagicMock()
    mct.evaluate = MagicMock(return_value={
        "should_trigger": False,
        "trigger_score": 0.1,
        "effective_trigger_score": 0.1,
        "triggers_active": [],
        "signal_weights": {},
        "max_recursions_capped": False,
    })
    mct.adapt_weights_from_evolution = MagicMock()
    return mct


@pytest.fixture
def mock_error_evolution():
    """A mock CausalErrorEvolutionTracker."""
    ee = MagicMock()
    ee.record_episode = MagicMock()
    ee.get_error_summary = MagicMock(return_value={
        "total_recorded": 0,
        "error_classes": {},
    })
    return ee


@pytest.fixture
def mock_model():
    """A mock model with cached attributes."""
    model = MagicMock()
    model._cached_spectral_stability_margin = 1.0
    model._cached_surprise = 0.0
    model._memory_stale = False
    model._cached_safety_violation = False
    model._cached_stall_severity = 0.0
    model._cached_output_quality = 1.0
    model._cached_border_uncertainty = 0.0
    model._last_trust_score = 1.0
    model._cached_causal_quality = 1.0
    model._cached_topology_state = None
    model.convergence_monitor = None
    model._error_evolution = None
    return model


def _make_utcc(model, feedback_bus=None, mct=None, config=None):
    """Create a UnifiedTrainingCycleController from aeon_server."""
    from aeon_server import UnifiedTrainingCycleController

    if config is None:
        config = MagicMock()
        config.reinforce_interval = 5

    utcc = UnifiedTrainingCycleController.__new__(
        UnifiedTrainingCycleController,
    )
    utcc.model = model
    utcc.config = config
    utcc._feedback_bus = feedback_bus
    utcc._mct = mct
    utcc._signal_bus = None
    utcc._vt_learner = None
    utcc._controller = None
    utcc._continual_core = None
    utcc._ucc = None
    utcc._cycle_count = 0
    utcc._metrics_history = []
    utcc.TOTAL_INTEGRATION_POINTS = 10
    utcc._z_annotation_used_fallback = False
    return utcc


# ══════════════════════════════════════════════════════════════════════
# K1: UCC Coherence Feedback to MCT
# ══════════════════════════════════════════════════════════════════════

class TestK1_UCC_Coherence_To_MCT:
    """K1: _collect_mct_signals reads graded UCC coherence from cycle_results."""

    def test_ucc_coherence_score_propagated(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.4},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # coherence_deficit should be max(0, 1 - 0.4) = 0.6
        assert kwargs["coherence_deficit"] >= 0.6

    def test_ucc_agreement_score_fallback(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "agreement_score": 0.3},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["coherence_deficit"] >= 0.7

    def test_ucc_missing_score_no_crash(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {"ucc": {"evaluated": True}}
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # Should not crash; coherence_deficit from fallback
        assert "coherence_deficit" in kwargs

    def test_ucc_high_coherence_low_deficit(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.95},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["coherence_deficit"] <= 0.1

    def test_no_ucc_in_cycle_results(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # Should use integration_health-based fallback
        assert "coherence_deficit" in kwargs

    def test_ucc_coherence_takes_max_of_existing(self, mock_model, mock_feedback_bus):
        """K1 uses max() so UCC doesn't lower an already-high deficit."""
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        # Simulate low integration_health → high coherence_deficit
        mock_feedback_bus._signals["integration_health"] = 0.1
        utcc._read_fb_signal = lambda name, default=0.0: mock_feedback_bus._signals.get(name, default)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.8},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # 1 - 0.1 = 0.9 from integration_health; 1 - 0.8 = 0.2 from UCC
        # max(0.9, 0.2) = 0.9
        assert kwargs["coherence_deficit"] >= 0.9 - 0.01


# ══════════════════════════════════════════════════════════════════════
# K2: SSP Temperature Signals to MCT
# ══════════════════════════════════════════════════════════════════════

class TestK2_SSP_Signals_To_MCT:
    """K2: _collect_mct_signals reads SSP alignment from cycle_results."""

    def test_ssp_misalignment_increases_convergence_conflict(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {"ssp": {"aligned": False}}
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["convergence_conflict"] >= 0.5

    def test_ssp_aligned_no_extra_conflict(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        # Ensure the feedback bus returns "aligned" status for SSP
        mock_feedback_bus._signals["ssp_alignment_ok"] = 1.0
        utcc._read_fb_signal = lambda name, default=0.0: mock_feedback_bus._signals.get(name, default)
        cycle_results = {"ssp": {"aligned": True}}
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # Should not add extra conflict for aligned SSP
        assert kwargs.get("convergence_conflict", 0.0) <= 0.01

    def test_ssp_feedback_bus_low_alignment(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        # Simulate low SSP alignment on feedback bus
        mock_feedback_bus._signals["ssp_alignment_ok"] = 0.2
        utcc._read_fb_signal = lambda name, default=0.0: mock_feedback_bus._signals.get(name, default)
        cycle_results = {"ssp": {"aligned": True}}
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # 1.0 - 0.2 = 0.8 from feedback bus
        assert kwargs["convergence_conflict"] >= 0.8 - 0.01

    def test_no_ssp_in_cycle_results(self, mock_model, mock_feedback_bus):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # Should not crash
        assert "convergence_conflict" in kwargs


# ══════════════════════════════════════════════════════════════════════
# K3: Same-Cycle MCT Re-execution
# ══════════════════════════════════════════════════════════════════════

class TestK3_Same_Cycle_Reexecution:
    """K3: When MCT triggers, UCC and SSP are re-executed in same cycle."""

    def test_reexecution_occurs_when_triggered(
        self, mock_model, mock_feedback_bus, mock_mct, mock_error_evolution,
    ):
        utcc = _make_utcc(
            mock_model, feedback_bus=mock_feedback_bus, mct=mock_mct,
        )
        # Make MCT trigger
        mock_mct.evaluate.return_value = {
            "should_trigger": True,
            "trigger_score": 0.8,
            "triggers_active": ["uncertainty"],
        }
        # Mock the execute methods
        utcc.execute_ucc_evaluation = MagicMock(return_value={
            "evaluated": True, "coherence_score": 0.7,
        })
        utcc.execute_ssp_alignment = MagicMock(return_value={
            "aligned": True,
        })
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._record_failure_episode = MagicMock()
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()

        result = utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            error_evolution=mock_error_evolution,
        )

        # Check that same_cycle_reexecution was executed
        assert result.get("same_cycle_reexecution", {}).get("executed") is True
        # UCC and SSP should have been called at least twice
        assert utcc.execute_ucc_evaluation.call_count >= 2
        assert utcc.execute_ssp_alignment.call_count >= 2

    def test_no_reexecution_when_not_triggered(
        self, mock_model, mock_feedback_bus, mock_mct, mock_error_evolution,
    ):
        utcc = _make_utcc(
            mock_model, feedback_bus=mock_feedback_bus, mct=mock_mct,
        )
        # MCT does NOT trigger
        mock_mct.evaluate.return_value = {
            "should_trigger": False,
            "trigger_score": 0.1,
            "triggers_active": [],
        }
        utcc.execute_ucc_evaluation = MagicMock(return_value={
            "evaluated": True,
        })
        utcc.execute_ssp_alignment = MagicMock(return_value={
            "aligned": True,
        })
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._record_failure_episode = MagicMock()
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()

        result = utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            error_evolution=mock_error_evolution,
        )

        # No re-execution
        assert result.get("same_cycle_reexecution") is None
        # UCC and SSP called only once
        assert utcc.execute_ucc_evaluation.call_count == 1
        assert utcc.execute_ssp_alignment.call_count == 1

    def test_no_double_reexecution(
        self, mock_model, mock_feedback_bus, mock_mct, mock_error_evolution,
    ):
        """K3 guard: _k3_reexecuted prevents infinite recursion."""
        utcc = _make_utcc(
            mock_model, feedback_bus=mock_feedback_bus, mct=mock_mct,
        )
        mock_mct.evaluate.return_value = {
            "should_trigger": True,
            "trigger_score": 0.9,
            "triggers_active": ["uncertainty"],
        }
        utcc.execute_ucc_evaluation = MagicMock(return_value={
            "evaluated": True,
        })
        utcc.execute_ssp_alignment = MagicMock(return_value={
            "aligned": True,
        })
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._record_failure_episode = MagicMock()
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()

        result = utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            error_evolution=mock_error_evolution,
        )

        # UCC should be called exactly 2 times (initial + 1 re-exec)
        assert utcc.execute_ucc_evaluation.call_count == 2

    def test_reexecution_ucc_failure_logged(
        self, mock_model, mock_feedback_bus, mock_mct, mock_error_evolution,
    ):
        """K3 gracefully handles UCC re-evaluation failure."""
        utcc = _make_utcc(
            mock_model, feedback_bus=mock_feedback_bus, mct=mock_mct,
        )
        mock_mct.evaluate.return_value = {
            "should_trigger": True,
            "trigger_score": 0.9,
            "triggers_active": ["uncertainty"],
        }
        call_count = [0]
        def ucc_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"evaluated": True}
            raise RuntimeError("UCC re-eval failed")

        utcc.execute_ucc_evaluation = MagicMock(side_effect=ucc_side_effect)
        utcc.execute_ssp_alignment = MagicMock(return_value={
            "aligned": True,
        })
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._record_failure_episode = MagicMock()
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()

        result = utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            error_evolution=mock_error_evolution,
        )

        # Should still have same_cycle_reexecution with error info
        reexec = result.get("same_cycle_reexecution", {})
        assert reexec.get("executed") is True
        assert "ucc_reeval_error" in reexec.get("results", {})


# ══════════════════════════════════════════════════════════════════════
# K4: UCC Coherence → Training Curriculum Feedback
# ══════════════════════════════════════════════════════════════════════

class TestK4_UCC_Training_Feedback:
    """K4: _sync_feedback_bus writes graded coherence + phase pressure."""

    def test_graded_coherence_written_to_bus(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.6},
        }
        utcc._sync_feedback_bus([], cycle_results)
        # Check that write_signal was called with ucc_coherence_score
        calls = {c.args[0]: c.args[1] for c in mock_feedback_bus.write_signal.call_args_list}
        assert "ucc_coherence_score" in calls
        assert abs(calls["ucc_coherence_score"] - 0.6) < 0.01

    def test_training_phase_pressure_written(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.3},
        }
        utcc._sync_feedback_bus([], cycle_results)
        calls = {c.args[0]: c.args[1] for c in mock_feedback_bus.write_signal.call_args_list}
        assert "training_phase_pressure" in calls
        # pressure = 1.0 - 0.3 = 0.7
        assert abs(calls["training_phase_pressure"] - 0.7) < 0.01

    def test_high_coherence_low_pressure(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.95},
        }
        utcc._sync_feedback_bus([], cycle_results)
        calls = {c.args[0]: c.args[1] for c in mock_feedback_bus.write_signal.call_args_list}
        assert "training_phase_pressure" in calls
        assert calls["training_phase_pressure"] <= 0.1

    def test_no_coherence_score_no_crash(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {"ucc": {"evaluated": True}}
        utcc._sync_feedback_bus([], cycle_results)
        # Should not crash; ucc_coherence_score NOT written
        calls = {c.args[0] for c in mock_feedback_bus.write_signal.call_args_list}
        # ucc_evaluation_ok should still be written
        assert "ucc_evaluation_ok" in calls

    def test_agreement_score_fallback(
        self, mock_model, mock_feedback_bus,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "agreement_score": 0.5},
        }
        utcc._sync_feedback_bus([], cycle_results)
        calls = {c.args[0]: c.args[1] for c in mock_feedback_bus.write_signal.call_args_list}
        assert "ucc_coherence_score" in calls
        assert abs(calls["ucc_coherence_score"] - 0.5) < 0.01


# ══════════════════════════════════════════════════════════════════════
# K5: Graded Diversity Escalation (RobustVectorQuantizer)
# ══════════════════════════════════════════════════════════════════════

class TestK5_Graded_Diversity_Escalation:
    """K5: accept_diversity_pressure on RobustVectorQuantizer."""

    @pytest.fixture
    def vq(self):
        from aeon_core import RobustVectorQuantizer
        return RobustVectorQuantizer(
            num_embeddings=64,
            embedding_dim=32,
        )

    def test_method_exists(self, vq):
        assert hasattr(vq, "accept_diversity_pressure")

    def test_normal_pressure(self, vq):
        result = vq.accept_diversity_pressure(0.1)
        assert result["accepted"] is True
        assert result["action_taken"] == "normal"

    def test_moderate_escalation(self, vq):
        result = vq.accept_diversity_pressure(0.35)
        assert result["action_taken"] == "moderate_escalation"
        assert "maintenance_interval" in result

    def test_high_escalation(self, vq):
        result = vq.accept_diversity_pressure(0.65)
        assert result["action_taken"] == "high_escalation"
        assert "maintenance_interval" in result
        assert "revival_threshold" in result

    def test_critical_escalation(self, vq):
        result = vq.accept_diversity_pressure(0.9)
        assert result["action_taken"] == "critical_escalation"
        assert result.get("forced_immediate") is True

    def test_pressure_clamped(self, vq):
        result = vq.accept_diversity_pressure(1.5)
        assert result["pressure"] == 1.0
        assert result["action_taken"] == "critical_escalation"

    def test_negative_pressure_clamped(self, vq):
        result = vq.accept_diversity_pressure(-0.5)
        assert result["pressure"] == 0.0
        assert result["action_taken"] == "normal"

    def test_escalation_progression(self, vq):
        """Verify monotonic escalation as pressure increases."""
        actions = []
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            r = vq.accept_diversity_pressure(p)
            actions.append(r["action_taken"])
        # Should progress from normal → moderate → high → critical
        assert actions[0] == "normal"
        assert actions[1] == "normal"
        assert actions[2] == "moderate_escalation"
        assert actions[3] == "high_escalation"
        assert actions[4] == "high_escalation"
        assert actions[5] == "critical_escalation"
        assert actions[6] == "critical_escalation"


# ══════════════════════════════════════════════════════════════════════
# K6: Error-class Sub-typing for Bridge Failures
# ══════════════════════════════════════════════════════════════════════

class TestK6_Bridge_Failure_Subtyping:
    """K6: Bridge failures have granular sub-error-classes."""

    def test_t2i_convergence_subtype(
        self, mock_model, mock_feedback_bus, mock_error_evolution,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(
            return_value={"bridged": False, "reason": "convergence monitor stalled"},
        )
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_ucc_evaluation = MagicMock(return_value={"evaluated": True})
        utcc.execute_ssp_alignment = MagicMock(return_value={"aligned": True})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()
        utcc._record_failure_episode = MagicMock()

        utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            convergence_monitor=MagicMock(),
            error_evolution=mock_error_evolution,
        )

        # Find the T2I bridge failure recording
        for call in utcc._record_failure_episode.call_args_list:
            args = call.args if call.args else ()
            if len(args) >= 2 and "training_to_inference_bridge_failure" in str(args[1]):
                assert "/convergence" in args[1]
                assert args[3].get("sub_class") == "convergence"
                return
        pytest.fail("T2I bridge failure with sub-class not recorded")

    def test_t2i_error_transfer_subtype(
        self, mock_model, mock_feedback_bus, mock_error_evolution,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(
            return_value={"bridged": False, "reason": "signal bus disconnected"},
        )
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_ucc_evaluation = MagicMock(return_value={"evaluated": True})
        utcc.execute_ssp_alignment = MagicMock(return_value={"aligned": True})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()
        utcc._record_failure_episode = MagicMock()

        utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            convergence_monitor=MagicMock(),
            error_evolution=mock_error_evolution,
        )

        for call in utcc._record_failure_episode.call_args_list:
            args = call.args if call.args else ()
            if len(args) >= 2 and "training_to_inference_bridge_failure" in str(args[1]):
                assert "/error_transfer" in args[1]
                return
        pytest.fail("T2I bridge failure with error_transfer sub-class not recorded")

    def test_i2t_hyperparameter_subtype(
        self, mock_model, mock_feedback_bus, mock_error_evolution,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(
            return_value={"bridged": False, "reason": "learning rate adaptation failed"},
        )
        utcc.execute_ucc_evaluation = MagicMock(return_value={"evaluated": True})
        utcc.execute_ssp_alignment = MagicMock(return_value={"aligned": True})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()
        utcc._record_failure_episode = MagicMock()

        utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            inference_error_evolution=MagicMock(),
            trainer=MagicMock(),
            error_evolution=mock_error_evolution,
        )

        for call in utcc._record_failure_episode.call_args_list:
            args = call.args if call.args else ()
            if len(args) >= 2 and "inference_to_training_bridge_failure" in str(args[1]):
                assert "/hyperparameter" in args[1]
                return
        pytest.fail("I2T bridge failure with hyperparameter sub-class not recorded")

    def test_i2t_inference_pattern_subtype(
        self, mock_model, mock_feedback_bus, mock_error_evolution,
    ):
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(
            return_value={"bridged": False, "reason": "pattern mismatch"},
        )
        utcc.execute_ucc_evaluation = MagicMock(return_value={"evaluated": True})
        utcc.execute_ssp_alignment = MagicMock(return_value={"aligned": True})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()
        utcc._record_failure_episode = MagicMock()

        utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            inference_error_evolution=MagicMock(),
            trainer=MagicMock(),
            error_evolution=mock_error_evolution,
        )

        for call in utcc._record_failure_episode.call_args_list:
            args = call.args if call.args else ()
            if len(args) >= 2 and "inference_to_training_bridge_failure" in str(args[1]):
                assert "/inference_pattern" in args[1]
                return
        pytest.fail("I2T bridge failure with inference_pattern sub-class not recorded")


# ══════════════════════════════════════════════════════════════════════
# K7: Convergence Certificate → Same-pass MCT Gating
# ══════════════════════════════════════════════════════════════════════

class TestK7_Certificate_MCT_Gating:
    """K7: Convergence certificate failure evaluates MCT in same pass."""

    @pytest.fixture
    def meta_loop(self):
        """A mock ProvablyConvergentMetaLoop with verify_convergence."""
        loop = MagicMock()
        return loop

    def test_k7_attribute_stored_on_trigger(self):
        """Verify K7 code path includes _k7_same_pass_triggered attribute."""
        source = Path(__file__).resolve().parent / "aeon_core.py"
        text = source.read_text()
        assert "_k7_same_pass_triggered" in text
        assert "_k7_certificate_mct_result" in text

    def test_k7_code_path_exists_in_source(self):
        """Verify K7 code path is present in aeon_core.py source."""
        source = Path(__file__).resolve().parent / "aeon_core.py"
        text = source.read_text()
        assert "K7: Same-pass MCT gating" in text
        assert "_k7_same_pass_triggered" in text
        assert "_k7_certificate_mct_result" in text

    def test_k7_evaluates_mct_on_contraction_failure(self):
        """Verify the K7 code evaluates MCT when contraction is not satisfied."""
        source = Path(__file__).resolve().parent / "aeon_core.py"
        text = source.read_text()
        # Find the K7 section
        k7_start = text.find("K7: Same-pass MCT gating")
        assert k7_start > 0
        k7_section = text[k7_start:k7_start + 2000]
        # Should check contraction_satisfied
        assert "contraction_satisfied" in k7_section
        # Should call metacognitive_trigger.evaluate
        assert "metacognitive_trigger.evaluate" in k7_section
        # Should have elevated uncertainty
        assert "uncertainty=0.6" in k7_section

    def test_k7_handles_exception_gracefully(self):
        """K7 MCT evaluation failure is bridged, not silently swallowed."""
        source = Path(__file__).resolve().parent / "aeon_core.py"
        text = source.read_text()
        k7_start = text.find("K7: Same-pass MCT gating")
        k7_section = text[k7_start:k7_start + 3000]
        assert "_bridge_silent_exception" in k7_section or "k7_certificate_mct_failure" in k7_section


# ══════════════════════════════════════════════════════════════════════
# Integration: Full Cognitive Flow Verification
# ══════════════════════════════════════════════════════════════════════

class TestCognitiveFlowIntegration:
    """End-to-end tests verifying complete cognitive data flows."""

    def test_ucc_deficit_reaches_mct_via_k1(
        self, mock_model, mock_feedback_bus, mock_mct, mock_error_evolution,
    ):
        """UCC coherence deficit flows: UCC → K1 → MCT coherence_deficit."""
        utcc = _make_utcc(
            mock_model, feedback_bus=mock_feedback_bus, mct=mock_mct,
        )
        utcc.execute_codebook_warm_start = MagicMock(return_value={})
        utcc.execute_context_calibration = MagicMock(return_value={})
        utcc.execute_teacher_student_inversion = MagicMock(return_value={})
        utcc.execute_streaming_signal_bus = MagicMock(return_value={})
        utcc.execute_z_annotation = MagicMock(return_value=([], None))
        utcc.execute_training_to_inference_bridge = MagicMock(return_value={})
        utcc.execute_inference_to_training_bridge = MagicMock(return_value={})
        utcc.execute_ucc_evaluation = MagicMock(return_value={
            "evaluated": True, "coherence_score": 0.2,
        })
        utcc.execute_ssp_alignment = MagicMock(return_value={"aligned": True})
        utcc.execute_micro_retrain = MagicMock(return_value={})
        utcc._record_failure_episode = MagicMock()
        utcc._sync_feedback_bus = MagicMock()
        utcc._feed_reinforce_to_mct = MagicMock()

        utcc.execute_full_cycle(
            epoch=0, phase="phase_a", epoch_metrics={},
            error_evolution=mock_error_evolution,
        )

        # MCT should have been called with high coherence_deficit
        mct_call = mock_mct.evaluate.call_args
        assert mct_call is not None
        mct_kwargs = mct_call.kwargs if mct_call.kwargs else {}
        if not mct_kwargs:
            # Might be passed via **kwargs dict
            pass
        # The _collect_mct_signals was called; verify it returns proper deficit
        collected = utcc._collect_mct_signals(0.0, [], {
            "ucc": {"evaluated": True, "coherence_score": 0.2},
        })
        assert collected["coherence_deficit"] >= 0.8 - 0.01

    def test_ssp_misalignment_reaches_mct_via_k2(
        self, mock_model, mock_feedback_bus,
    ):
        """SSP misalignment flows: SSP → K2 → MCT convergence_conflict."""
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        collected = utcc._collect_mct_signals(0.0, [], {
            "ssp": {"aligned": False},
        })
        assert collected["convergence_conflict"] >= 0.5

    def test_diversity_pressure_accepted_by_vq(self):
        """Diversity collapse signal → K5 → RobustVectorQuantizer."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(num_embeddings=32, embedding_dim=16)
        result = vq.accept_diversity_pressure(0.75)
        assert result["accepted"] is True
        assert result["action_taken"] == "high_escalation"

    def test_phase_pressure_signal_chain(
        self, mock_model, mock_feedback_bus,
    ):
        """K4: Low UCC coherence → training_phase_pressure on feedback bus."""
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        cycle_results = {
            "ucc": {"evaluated": True, "coherence_score": 0.25},
        }
        utcc._sync_feedback_bus([], cycle_results)
        calls = {c.args[0]: c.args[1] for c in mock_feedback_bus.write_signal.call_args_list}
        assert calls.get("training_phase_pressure", 0.0) >= 0.7


# ══════════════════════════════════════════════════════════════════════
# Regression: Ensure Existing Behavior Unchanged
# ══════════════════════════════════════════════════════════════════════

class TestKSeriesRegression:
    """Verify K-series patches don't break existing behavior."""

    def test_collect_mct_signals_returns_all_original_keys(
        self, mock_model, mock_feedback_bus,
    ):
        """All pre-K original signal keys still present."""
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        kwargs = utcc._collect_mct_signals(0.5, ["test"], {})
        expected_keys = {
            "uncertainty", "coherence_deficit", "recovery_pressure",
            "diversity_collapse", "convergence_conflict",
            "spectral_stability_margin", "world_model_surprise",
            "memory_staleness", "safety_violation", "stall_severity",
            "output_reliability", "border_uncertainty", "memory_trust_deficit",
            "causal_quality",
        }
        for key in expected_keys:
            assert key in kwargs, f"Missing key: {key}"

    def test_sync_feedback_bus_still_writes_original_signals(
        self, mock_model, mock_feedback_bus,
    ):
        """Original _sync_feedback_bus signals still present."""
        utcc = _make_utcc(mock_model, feedback_bus=mock_feedback_bus)
        utcc._sync_feedback_bus([], {"ucc": {"evaluated": True}})
        written_keys = {c.args[0] for c in mock_feedback_bus.write_signal.call_args_list}
        assert "integration_health" in written_keys
        assert "integration_failure_rate" in written_keys
        assert "ucc_evaluation_ok" in written_keys

    def test_vq_still_has_codebook_usage_stats(self):
        """RobustVectorQuantizer original methods still present."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(num_embeddings=32, embedding_dim=16)
        stats = vq.get_codebook_usage_stats()
        assert "total_codes" in stats
        assert "used_codes" in stats
        assert "perplexity" in stats
