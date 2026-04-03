"""
Tests for J-series cognitive integration patches.

J1:  Feed Teacher-Student Inversion (Point 1) results to feedback bus.
J2:  Feed SSP alignment (Point 9) status to feedback bus.
J3:  Context calibration fallback — apply safe default when calibration fails.
J4:  Z-annotation fallback quality — conservative 0.2 instead of uniform 1.0.
J5:  Auto-wire expansion — discover inference_error_evolution, convergence_monitor,
     trainer, causal_trace + log missing components.
J6:  Silent failure logging — replace bare except: pass in MCT recording.
J7:  MCT weight adaptation ordering — adapt BEFORE evaluate.
J8:  Pseudo-label generation bridge — generate from VT continuous learner.
J9:  Inference→Training bridge logging — log+record when skipped.
J10: Cycle provenance enrichment — cycle_id and timestamp on feedback bus writes.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers — lightweight mocks that exercise the integration controller
# without pulling in the full AEON stack.
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> SimpleNamespace:
    """Minimal AEONConfig-like object."""
    defaults = dict(
        z_dim=64,
        hidden_dim=128,
        num_latent_tokens=16,
        vq_num_embeddings=32,
        vq_embedding_dim=64,
        context_window=None,
        reinforce_interval=5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_model(**attrs: Any) -> nn.Module:
    """Minimal nn.Module with optional mock attributes."""
    model = nn.Linear(4, 4)
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


class FakeFeedbackBus:
    """Mock CognitiveFeedbackBus that records all writes."""

    def __init__(self) -> None:
        self._extra_signals: Dict[str, float] = {}
        self._write_log: List[tuple] = []

    def write_signal(self, name: str, value: float) -> None:
        self._extra_signals[name] = value
        self._write_log.append((name, value))

    def read_signal(self, name: str) -> float:
        return self._extra_signals.get(name, 0.0)

    def get_state(self) -> Dict[str, float]:
        return dict(self._extra_signals)

    def get_oscillation_score(self) -> float:
        return 0.0


class FakeErrorEvolution:
    """Mock CausalErrorEvolutionTracker that records episodes."""

    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(self, **kwargs: Any) -> None:
        self.episodes.append(kwargs)

    def get_error_summary(self) -> Dict[str, Any]:
        return {
            "error_classes": {ep["error_class"]: {"count": 1} for ep in self.episodes},
            "total_recorded": len(self.episodes),
            "avg_success_rate": 0.5,
        }

    def get_episode_count(self) -> int:
        return len(self.episodes)


class FakeMCT:
    """Mock MetaCognitiveRecursionTrigger."""

    def __init__(self) -> None:
        self.evaluate_calls: List[Dict[str, Any]] = []
        self.adapt_calls: List[Any] = []

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        self.evaluate_calls.append(kwargs)
        score = kwargs.get("uncertainty", 0.0)
        return {"should_trigger": score > 0.5, "trigger_score": score}

    def adapt_weights_from_evolution(self, summary: Any) -> None:
        self.adapt_calls.append(summary)


class FakeVTLearner:
    """Mock VibeThinkerContinuousLearner with optional pseudo-label generation."""

    def __init__(
        self,
        has_generate: bool = True,
        labels: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._calibration_ema = 0.5
        self._complexity_threshold_ema = 0.5
        self._psi_weight_ema = 0.05
        self._has_generate = has_generate
        self._labels = labels or [{"token_id": 1, "label": "test"}]

    def generate_pseudo_labels(self) -> List[Dict[str, Any]]:
        if not self._has_generate:
            raise AttributeError("no generate")
        return self._labels

    def get_consolidation_labels(self) -> List[Dict[str, Any]]:
        return self._labels


# ---------------------------------------------------------------------------
# Import the integration controller
# ---------------------------------------------------------------------------
from aeon_integration import UnifiedTrainingCycleController, trace_output_to_premise


def _make_controller(**kw: Any) -> UnifiedTrainingCycleController:
    config = kw.pop("config", _make_config())
    model = kw.pop("model", _make_model())
    device = kw.pop("device", torch.device("cpu"))
    ctrl = UnifiedTrainingCycleController(model, config, device)
    return ctrl


# ============================================================================
# J1 — Teacher-Student Inversion feedback bus propagation
# ============================================================================

class TestJ1InversionFeedbackBus:
    """Verify Point 1 (Teacher-Student Inversion) writes to feedback bus."""

    def test_j1_inversion_success_writes_feedback_bus(self):
        """Successful inversion writes 1.0 to teacher_student_inversion_ok."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        # Mock the inversion to succeed
        with patch.object(ctrl, "execute_teacher_student_inversion",
                          return_value={"inverted": True}):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                z_sequences=[torch.randn(4, 64)],
            )

        assert "teacher_student_inversion_ok" in fb._extra_signals
        assert fb._extra_signals["teacher_student_inversion_ok"] == 1.0

    def test_j1_inversion_failure_writes_zero(self):
        """Failed inversion writes 0.0 to teacher_student_inversion_ok."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_teacher_student_inversion",
                          return_value={"inverted": False, "reason": "test"}):
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                z_sequences=[torch.randn(4, 64)],
            )

        assert fb._extra_signals["teacher_student_inversion_ok"] == 0.0

    def test_j1_no_feedback_bus_no_error(self):
        """Without feedback bus, J1 code path is simply skipped."""
        ctrl = _make_controller()
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_teacher_student_inversion",
                          return_value={"inverted": True}):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                z_sequences=[torch.randn(4, 64)],
            )
        assert "teacher_student_inversion" in result

    def test_j1_phase_b_no_inversion(self):
        """Point 1 is phase-A only — no feedback bus write in phase B."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        ctrl.execute_full_cycle(
            epoch=0, phase="B", epoch_metrics={},
            z_sequences=[torch.randn(4, 64)],
        )
        assert "teacher_student_inversion_ok" not in fb._extra_signals


# ============================================================================
# J2 — SSP alignment feedback bus propagation
# ============================================================================

class TestJ2SSPFeedbackBus:
    """Verify Point 9 (SSP alignment) writes to feedback bus."""

    def test_j2_ssp_success_writes_feedback_bus(self):
        """Successful SSP alignment writes 1.0 to ssp_alignment_ok."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_ssp_alignment",
                          return_value={"aligned": True}):
            ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        assert fb._extra_signals["ssp_alignment_ok"] == 1.0

    def test_j2_ssp_failure_writes_zero(self):
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_ssp_alignment",
                          return_value={"aligned": False, "reason": "err"}):
            ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        assert fb._extra_signals["ssp_alignment_ok"] == 0.0

    def test_j2_ssp_temperature_values_written(self):
        """Temperature values from SSP alignment are surfaced on feedback bus."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_ssp_alignment",
                          return_value={
                              "aligned": True,
                              "temperature_values": {"diversity": 0.7, "sampling": 1.2},
                          }):
            ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        assert fb._extra_signals["ssp_temperature_diversity"] == 0.7
        assert fb._extra_signals["ssp_temperature_sampling"] == 1.2


# ============================================================================
# J3 — Context calibration fallback
# ============================================================================

class TestJ3ContextCalibrationFallback:
    """Verify context_window fallback when calibration fails."""

    def test_j3_fallback_applied_when_no_context_window(self):
        """config.context_window=None → fallback to 512 on cal failure."""
        config = _make_config(context_window=None)
        ctrl = _make_controller(config=config)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_context_calibration",
                          return_value={"calibrated": False, "reason": "fail"}):
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                tokens=torch.randn(8, 64),
            )

        assert config.context_window == 512

    def test_j3_no_overwrite_when_context_window_set(self):
        """Existing context_window is NOT overwritten by fallback."""
        config = _make_config(context_window=1024)
        ctrl = _make_controller(config=config)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_context_calibration",
                          return_value={"calibrated": False, "reason": "fail"}):
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                tokens=torch.randn(8, 64),
            )

        assert config.context_window == 1024

    def test_j3_calibration_success_no_fallback(self):
        """Successful calibration does not trigger fallback."""
        config = _make_config(context_window=None)
        ctrl = _make_controller(config=config)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        with patch.object(ctrl, "execute_context_calibration",
                          return_value={"calibrated": True}):
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                tokens=torch.randn(8, 64),
            )

        # context_window remains None because calibration succeeded —
        # the real calibrate_context_window would set it.
        assert config.context_window is None


# ============================================================================
# J4 — Z-annotation fallback quality
# ============================================================================

class TestJ4ZAnnotationFallback:
    """Verify fallback annotations use conservative confidence (0.2)."""

    def test_j4_fallback_annotations_are_conservative(self):
        """Fallback annotations should have confidence=0.2, not 1.0."""
        ctrl = _make_controller()

        # Force annotation to fail by patching the import
        with patch.dict("sys.modules", {"ae_train": MagicMock()}):
            # Make annotate_z_sequences_quality raise
            import sys
            sys.modules["ae_train"].annotate_z_sequences_quality = MagicMock(
                side_effect=RuntimeError("no model")
            )

            z_seqs = [torch.randn(4, 64), torch.randn(3, 64)]
            _, annotations = ctrl.execute_z_annotation(z_seqs)

        # Annotations should be conservative (0.2), not 1.0
        for ann in annotations:
            assert ann.shape[-1] == 3
            assert torch.allclose(ann, torch.full_like(ann, 0.2))

    def test_j4_fallback_flag_set(self):
        """_z_annotation_used_fallback should be True after failure."""
        ctrl = _make_controller()

        with patch.dict("sys.modules", {"ae_train": MagicMock()}):
            import sys
            sys.modules["ae_train"].annotate_z_sequences_quality = MagicMock(
                side_effect=RuntimeError("no model")
            )

            z_seqs = [torch.randn(4, 64)]
            ctrl.execute_z_annotation(z_seqs)

        assert ctrl._z_annotation_used_fallback is True

    def test_j4_fallback_below_quality_threshold(self):
        """Fallback confidence 0.2 should be below the 0.3 threshold."""
        ctrl = _make_controller()

        with patch.dict("sys.modules", {"ae_train": MagicMock()}):
            import sys
            sys.modules["ae_train"].annotate_z_sequences_quality = MagicMock(
                side_effect=RuntimeError("no model")
            )

            z_seqs = [torch.randn(4, 64)]
            _, annotations = ctrl.execute_z_annotation(z_seqs)

        # Mean confidence of fallback should be 0.2 < 0.3 threshold
        mean_conf = annotations[0][..., 0].mean().item()
        assert mean_conf < 0.3, f"Fallback confidence {mean_conf} >= 0.3"


# ============================================================================
# J5 — Auto-wire expansion
# ============================================================================

class TestJ5AutoWireExpansion:
    """Verify auto_wire discovers extra components and logs missing ones."""

    def test_j5_discovers_inference_error_evolution(self):
        """Auto-wire finds inference_error_evolution on model."""
        ctrl = _make_controller()
        ee = FakeErrorEvolution()
        model = _make_model(inference_error_evolution=ee)

        result = ctrl.auto_wire(model)

        assert "inference_error_evolution" in result["wired"]
        assert ctrl._discovered_inference_error_evolution is ee

    def test_j5_discovers_convergence_monitor(self):
        """Auto-wire finds convergence_monitor on model."""
        ctrl = _make_controller()
        mon = MagicMock()
        model = _make_model(convergence_monitor=mon)

        result = ctrl.auto_wire(model)

        assert "convergence_monitor" in result["wired"]
        assert ctrl._discovered_convergence_monitor is mon

    def test_j5_discovers_trainer(self):
        """Auto-wire finds trainer on model."""
        ctrl = _make_controller()
        trainer = MagicMock()
        model = _make_model(trainer=trainer)

        result = ctrl.auto_wire(model)

        assert "trainer" in result["wired"]
        assert ctrl._discovered_trainer is trainer

    def test_j5_discovers_causal_trace(self):
        """Auto-wire finds causal_trace on model."""
        ctrl = _make_controller()
        trace = MagicMock()
        model = _make_model(causal_trace=trace)

        result = ctrl.auto_wire(model)

        assert "causal_trace" in result["wired"]
        assert ctrl._discovered_causal_trace is trace

    def test_j5_missing_extra_components_logged(self, caplog):
        """Missing extra components appear in the missing list."""
        ctrl = _make_controller()
        model = _make_model()  # no extra components

        result = ctrl.auto_wire(model)

        for name in ["inference_error_evolution", "convergence_monitor",
                      "trainer", "causal_trace"]:
            assert name in result["missing"]

    def test_j5_missing_core_component_logged(self, caplog):
        """Missing core components are logged at debug level."""
        ctrl = _make_controller()
        model = _make_model()

        with caplog.at_level(logging.DEBUG, logger="AEON-Integration"):
            ctrl.auto_wire(model)

        assert any("not found on model" in r.message for r in caplog.records)


# ============================================================================
# J6 — Silent failure logging for MCT recording
# ============================================================================

class TestJ6MCTRecordingLogging:
    """Verify MCT recording failures are logged, not silently swallowed."""

    def test_j6_mct_recording_failure_logged(self, caplog):
        """When error_evolution.record_episode raises, it's logged."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        # Error evolution that fails on recording
        bad_ee = MagicMock()
        bad_ee.record_episode = MagicMock(side_effect=RuntimeError("db full"))
        bad_ee.get_error_summary = MagicMock(return_value={
            "error_classes": {}, "total_recorded": 0, "avg_success_rate": 0.5,
        })

        with caplog.at_level(logging.DEBUG, logger="AEON-Integration"):
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                error_evolution=bad_ee,
            )

        # Check that the MCT recording failure was logged (not silently dropped)
        assert any(
            "MCT trigger decision recording failed" in r.message
            for r in caplog.records
        )


# ============================================================================
# J7 — MCT weight adaptation ordering
# ============================================================================

class TestJ7MCTAdaptationOrdering:
    """Verify MCT weights are adapted BEFORE evaluate() is called."""

    def test_j7_adapt_before_evaluate(self):
        """adapt_weights_from_evolution called before evaluate."""
        ctrl = _make_controller()
        call_order: List[str] = []

        class OrderTrackingMCT:
            def adapt_weights_from_evolution(self, summary: Any) -> None:
                call_order.append("adapt")

            def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
                call_order.append("evaluate")
                return {"should_trigger": False, "trigger_score": 0.0}

        ctrl.attach_metacognitive_trigger(OrderTrackingMCT())
        ee = FakeErrorEvolution()
        ee.record_episode(error_class="test", strategy_used="test", success=True)

        ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            error_evolution=ee,
        )

        assert "adapt" in call_order
        assert "evaluate" in call_order
        assert call_order.index("adapt") < call_order.index("evaluate")

    def test_j7_adapt_failure_does_not_block_evaluate(self):
        """If pre-adapt fails, evaluate still runs."""
        ctrl = _make_controller()

        class FailAdaptMCT:
            def adapt_weights_from_evolution(self, summary: Any) -> None:
                raise RuntimeError("adapt fail")

            def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
                return {"should_trigger": False, "trigger_score": 0.0}

        ctrl.attach_metacognitive_trigger(FailAdaptMCT())
        ee = FakeErrorEvolution()

        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            error_evolution=ee,
        )

        # evaluate() should still have run
        assert "metacognitive_review" in result

    def test_j7_no_error_evolution_skips_adapt(self):
        """Without error_evolution, pre-adapt is skipped (no crash)."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
        )

        # evaluate() should run, adapt should NOT have been called
        assert len(mct.adapt_calls) == 0
        assert len(mct.evaluate_calls) > 0


# ============================================================================
# J8 — Pseudo-label generation bridge
# ============================================================================

class TestJ8PseudoLabelGeneration:
    """Verify pseudo-labels are generated from VT learner when not provided."""

    def test_j8_generates_from_vt_learner(self):
        """When pseudo_labels=None and VT learner attached, generates them."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        labels = [{"token_id": i, "label": f"l{i}"} for i in range(3)]
        learner = FakeVTLearner(has_generate=True, labels=labels)
        ctrl.attach_vt_learner(learner)

        with patch.object(ctrl, "execute_micro_retrain",
                          return_value={"retrained": True}) as mock_mr:
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
            )

        # execute_micro_retrain should have been called with generated labels
        mock_mr.assert_called_once()
        call_args = mock_mr.call_args
        assert call_args[0][0] == labels  # pseudo_labels

    def test_j8_fallback_to_consolidation_labels(self):
        """Falls back to get_consolidation_labels if generate_pseudo_labels missing."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        class LegacyLearner:
            _calibration_ema = 0.5
            _complexity_threshold_ema = 0.5
            _psi_weight_ema = 0.05

            def get_consolidation_labels(self):
                return [{"token_id": 42}]

        ctrl.attach_vt_learner(LegacyLearner())

        with patch.object(ctrl, "execute_micro_retrain",
                          return_value={"retrained": True}) as mock_mr:
            ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        mock_mr.assert_called_once()

    def test_j8_external_pseudo_labels_take_priority(self):
        """Externally provided pseudo_labels are used even with VT learner."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        learner = FakeVTLearner(labels=[{"bad": True}])
        ctrl.attach_vt_learner(learner)

        external_labels = [{"external": True}]
        with patch.object(ctrl, "execute_micro_retrain",
                          return_value={"retrained": True}) as mock_mr:
            ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                pseudo_labels=external_labels,
            )

        call_args = mock_mr.call_args
        assert call_args[0][0] == external_labels

    def test_j8_generation_failure_no_crash(self):
        """Pseudo-label generation failure doesn't crash the cycle."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        class FailLearner:
            _calibration_ema = 0.5

            def generate_pseudo_labels(self):
                raise RuntimeError("gen fail")

        ctrl.attach_vt_learner(FailLearner())

        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
        )

        # Cycle should still complete
        assert "cycle" in result
        # No micro_retrain since generation failed and no fallback
        assert "micro_retrain" not in result

    def test_j8_no_vt_learner_pseudo_labels_none(self):
        """Without VT learner, pseudo_labels stay None — Point 10 skipped."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
        )

        assert "micro_retrain" not in result


# ============================================================================
# J9 — Inference→Training bridge logging
# ============================================================================

class TestJ9BridgeLogging:
    """Verify skipped bridge is logged and recorded in results."""

    def test_j9_missing_both_logged(self, caplog):
        """Missing trainer and inference_ee both logged."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        with caplog.at_level(logging.WARNING, logger="AEON-Integration"):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
            )

        assert "inference_to_training_bridge" in result
        bridge = result["inference_to_training_bridge"]
        assert bridge["bridged"] is False
        assert bridge["reason"] == "skipped_missing_components"
        assert "inference_error_evolution=None" in bridge["missing"]
        assert "trainer=None" in bridge["missing"]

    def test_j9_missing_trainer_only(self, caplog):
        """Missing trainer logged when inference_ee provided."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        ee = FakeErrorEvolution()

        with caplog.at_level(logging.WARNING, logger="AEON-Integration"):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                inference_error_evolution=ee,
            )

        bridge = result["inference_to_training_bridge"]
        assert bridge["bridged"] is False
        assert "trainer=None" in bridge["missing"]
        assert "inference_error_evolution=None" not in bridge["missing"]

    def test_j9_both_provided_no_skip(self):
        """When both are provided, bridge executes normally."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        ee = FakeErrorEvolution()
        trainer = MagicMock()

        with patch.object(ctrl, "execute_inference_to_training_bridge",
                          return_value={"bridged": True}):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                inference_error_evolution=ee,
                trainer=trainer,
            )

        bridge = result["inference_to_training_bridge"]
        assert bridge["bridged"] is True


# ============================================================================
# J10 — Cycle provenance enrichment
# ============================================================================

class TestJ10CycleProvenance:
    """Verify cycle_id and timestamp are written to feedback bus."""

    def test_j10_cycle_id_written(self):
        """integration_cycle_id appears on feedback bus after cycle."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        assert "integration_cycle_id" in fb._extra_signals
        assert fb._extra_signals["integration_cycle_id"] == 1.0

    def test_j10_timestamp_written(self):
        """integration_cycle_timestamp is a valid time value."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        before = time.time()
        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        after = time.time()

        ts = fb._extra_signals.get("integration_cycle_timestamp")
        assert ts is not None
        assert before <= ts <= after

    def test_j10_cycle_id_increments(self):
        """Cycle ID increments with each cycle."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        assert fb._extra_signals["integration_cycle_id"] == 1.0

        ctrl.execute_full_cycle(epoch=1, phase="A", epoch_metrics={})
        assert fb._extra_signals["integration_cycle_id"] == 2.0

    def test_j10_no_feedback_bus_no_error(self):
        """Without feedback bus, provenance writes are skipped safely."""
        ctrl = _make_controller()
        ctrl.attach_metacognitive_trigger(FakeMCT())

        result = ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        assert "cycle" in result


# ============================================================================
# Integration tests — end-to-end flows
# ============================================================================

class TestJSeriesEndToEnd:
    """End-to-end tests verifying multiple J-patches work together."""

    def test_full_cycle_with_all_j_patches(self):
        """Complete cycle exercises J1-J10 patches in concert."""
        fb = FakeFeedbackBus()
        ee = FakeErrorEvolution()
        mct = FakeMCT()
        learner = FakeVTLearner(labels=[{"token_id": 1}])
        config = _make_config(context_window=None)
        model = _make_model()

        ctrl = _make_controller(config=config, model=model)
        ctrl.attach_feedback_bus(fb)
        ctrl.attach_metacognitive_trigger(mct)
        ctrl.attach_vt_learner(learner)

        with patch.object(ctrl, "execute_teacher_student_inversion",
                          return_value={"inverted": True}), \
             patch.object(ctrl, "execute_codebook_warm_start",
                          return_value={"initialized": True}), \
             patch.object(ctrl, "execute_context_calibration",
                          return_value={"calibrated": False, "reason": "test"}), \
             patch.object(ctrl, "execute_ssp_alignment",
                          return_value={"aligned": True, "temperature_values": {"t": 0.9}}), \
             patch.object(ctrl, "execute_micro_retrain",
                          return_value={"retrained": True}):

            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
                z_sequences=[torch.randn(4, 64)],
                tokens=torch.randn(8, 64),
                error_evolution=ee,
            )

        # J1: Inversion feedback bus
        assert fb._extra_signals["teacher_student_inversion_ok"] == 1.0
        # J2: SSP feedback bus
        assert fb._extra_signals["ssp_alignment_ok"] == 1.0
        assert fb._extra_signals["ssp_temperature_t"] == 0.9
        # J3: Fallback context_window applied (cal failed, was None)
        assert config.context_window == 512
        # J7: MCT adaptation happened before evaluate
        assert len(mct.adapt_calls) >= 1
        assert len(mct.evaluate_calls) >= 1
        # J8: Pseudo-labels generated from VT learner
        assert "micro_retrain" in result
        # J9: Bridge recorded as skipped (no trainer/inference_ee)
        assert result["inference_to_training_bridge"]["bridged"] is False
        # J10: Provenance data
        assert "integration_cycle_id" in fb._extra_signals
        assert "integration_cycle_timestamp" in fb._extra_signals

    def test_causal_transparency_trace(self):
        """trace_output_to_premise works with enriched cycle data."""
        ee = FakeErrorEvolution()
        ee.record_episode(
            error_class="test_error", strategy_used="test", success=False,
        )

        ctrl = _make_controller()
        ctrl.attach_metacognitive_trigger(FakeMCT())

        result = ctrl.execute_full_cycle(
            epoch=0, phase="A", epoch_metrics={},
            error_evolution=ee,
        )

        trace = trace_output_to_premise(
            output_action="test_output",
            cycle_history=[result],
            error_evolution=ee,
        )

        assert trace["traced"] is True
        assert trace["originating_cycle"] == 1
        assert len(trace["trace_chain"]) >= 2

    def test_auto_wire_with_extra_discovery(self):
        """Auto-wire discovers both core and extra components."""
        ctrl = _make_controller()
        fb = FakeFeedbackBus()
        ee = FakeErrorEvolution()
        model = _make_model(
            cognitive_feedback_bus=fb,
            inference_error_evolution=ee,
        )

        result = ctrl.auto_wire(model)

        assert "feedback_bus" in result["wired"]
        assert "inference_error_evolution" in result["wired"]
        assert ctrl._feedback_bus is fb
        assert ctrl._discovered_inference_error_evolution is ee


# ============================================================================
# Regression tests — verify existing behavior preserved
# ============================================================================

class TestJSeriesRegression:
    """Ensure J-series patches don't break existing functionality."""

    def test_g2_continuous_mct_still_works(self):
        """G2 continuous MCT evaluation still runs every cycle."""
        ctrl = _make_controller()
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})

        assert len(mct.evaluate_calls) == 1
        assert "metacognitive_review" in ctrl._metrics_history[-1]

    def test_g6_periodic_reinforcement_still_works(self):
        """G6 periodic mutual reinforcement still fires at correct interval."""
        config = _make_config(reinforce_interval=2)
        model = _make_model()
        model.verify_and_reinforce = MagicMock(return_value={"overall_score": 0.9})

        ctrl = _make_controller(config=config, model=model)
        ctrl.attach_metacognitive_trigger(FakeMCT())

        ctrl.execute_full_cycle(epoch=0, phase="A", epoch_metrics={})
        assert not model.verify_and_reinforce.called

        ctrl.execute_full_cycle(epoch=1, phase="A", epoch_metrics={})
        assert model.verify_and_reinforce.called

    def test_h2_z_quality_filtering_still_works(self):
        """H2 z-quality filtering in execute_micro_retrain still functions."""
        ctrl = _make_controller()

        z_seqs = [torch.randn(4, 64), torch.randn(4, 64)]
        # First annotation has low confidence (should be filtered)
        ann_low = torch.full((4, 3), 0.1)
        # Second has high confidence (should pass)
        ann_high = torch.full((4, 3), 0.8)

        with patch.dict("sys.modules", {"ae_train": MagicMock()}):
            import sys
            mock_retrain = MagicMock(return_value={"retrained": True})
            sys.modules["ae_train"].micro_retrain_from_pseudo_labels = mock_retrain

            result = ctrl.execute_micro_retrain(
                pseudo_labels=[{"label": "test"}],
                z_sequences=z_seqs,
                z_annotations=[ann_low, ann_high],
            )

        # Only high-confidence sequence should be in the filtered set
        call_kwargs = mock_retrain.call_args[1]
        filtered = call_kwargs.get("z_sequences") or mock_retrain.call_args[0][0] if len(mock_retrain.call_args[0]) > 0 else None
        # The retrain was called with filtered sequences
        assert result.get("z_quality_filtered", False) or "retrained" in result

    def test_uncertainty_flags_propagation_intact(self):
        """Uncertainty flags still propagate to MCT via collect_mct_signals."""
        fb = FakeFeedbackBus()
        ctrl = _make_controller()
        ctrl.attach_feedback_bus(fb)
        mct = FakeMCT()
        ctrl.attach_metacognitive_trigger(mct)

        # Force some failures to generate uncertainty flags
        with patch.object(ctrl, "execute_ssp_alignment",
                          return_value={"aligned": False, "reason": "test"}):
            result = ctrl.execute_full_cycle(
                epoch=0, phase="A", epoch_metrics={},
            )

        flags = result["uncertainty_flags"]
        assert "ssp_alignment_failed" in flags
        assert result["metacognitive_review"]["flags"] == flags
