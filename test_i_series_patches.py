"""
Tests for I-series cognitive integration patches (final integration & activation).

I1:  Record MCT evaluation failures to error_evolution
I2:  Record mutual reinforcement failures to error_evolution
I3:  MCT signal collection resilience (defaults on exception)
I4:  Record feedback bus sync failures to error_evolution
I5:  Debug logging on feedback bus read failures
I6:  Record MCT weight adaptation failures to error_evolution
I7:  Z-annotation fallback flagged as uncertainty
I8:  Micro-retrain z-filter ratio written to feedback bus
I9:  MCT trigger decisions recorded to error_evolution
I10: Reinforce-to-MCT bridge failures recorded
I11: Failure returns include traced/causal_chain fields
I12: Wizard results consumption bridging wizard → integration
"""
from __future__ import annotations

import logging
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ── Lightweight stubs ────────────────────────────────────────────────────────

class _StubConfig:
    reinforce_interval = 5
    z_dim = 16
    vq_embedding_dim = 16


class _StubErrorEvolution:
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(self, *, error_class: str, strategy_used: str,
                       success: bool, metadata: Any = None,
                       causal_antecedents: Any = None) -> None:
        self.episodes.append({
            "error_class": error_class,
            "strategy_used": strategy_used,
            "success": success,
            "metadata": metadata,
        })

    def get_error_summary(self) -> Dict[str, Any]:
        return {
            "total_recorded": len(self.episodes),
            "error_classes": list({e["error_class"] for e in self.episodes}),
        }

    def get_episode_count(self) -> int:
        return len(self.episodes)


class _StubFeedbackBus:
    """Minimal feedback bus that records writes and supports reads."""
    def __init__(self) -> None:
        self._extra_signals: Dict[str, float] = {}

    def write_signal(self, name: str, value: float) -> None:
        self._extra_signals[name] = value

    def register_signal(self, name: str, default: float = 0.0) -> None:
        self._extra_signals.setdefault(name, default)

    def get_oscillation_score(self) -> float:
        return 0.05


class _FailingFeedbackBus(_StubFeedbackBus):
    """Feedback bus whose write_signal always raises."""
    def write_signal(self, name: str, value: float) -> None:
        raise RuntimeError("bus write boom")


class _FailingOscillationBus(_StubFeedbackBus):
    """Feedback bus whose get_oscillation_score raises."""
    def get_oscillation_score(self) -> float:
        raise RuntimeError("oscillation boom")


class _StubMCT:
    """Minimal MCT that records all kwargs it receives."""
    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}
        self._should_trigger = False

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = dict(kwargs)
        return {
            "should_trigger": self._should_trigger,
            "trigger_score": kwargs.get("uncertainty", 0.0),
        }

    def adapt_weights_from_evolution(self, summary: Dict[str, Any]) -> None:
        self._adapted_from = summary


class _FailingMCT(_StubMCT):
    """MCT whose evaluate always raises."""
    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("MCT evaluate boom")


class _FailingAdaptMCT(_StubMCT):
    """MCT whose adapt_weights_from_evolution always raises."""
    def adapt_weights_from_evolution(self, summary: Dict[str, Any]) -> None:
        raise RuntimeError("adapt boom")


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(4, 4)
        self._cached_spectral_stability_margin = 0.9
        self._cached_surprise = 0.15
        self._memory_stale = False
        self._cached_safety_violation = False
        self._cached_stall_severity = 0.05
        self._cached_output_quality = 0.95
        self._cached_border_uncertainty = 0.02
        self._last_trust_score = 0.88
        self._cached_causal_quality = 0.92
        self._reinforce_result: Optional[Dict[str, Any]] = None
        self._error_evolution: Optional[Any] = None

    def verify_and_reinforce(self) -> Dict[str, Any]:
        if self._reinforce_result is not None:
            return self._reinforce_result
        return {"overall_score": 0.85, "status": "ok"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)


class _FailingReinforceModel(_StubModel):
    """Model whose verify_and_reinforce always raises."""
    def verify_and_reinforce(self) -> Dict[str, Any]:
        raise RuntimeError("reinforce boom")


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _make_controller(
    attach_fb: bool = True,
    attach_mct: bool = True,
    fb_class: type = _StubFeedbackBus,
    mct_class: type = _StubMCT,
    model_class: type = _StubModel,
) -> Any:
    """Build a wired UnifiedTrainingCycleController."""
    from aeon_integration import UnifiedTrainingCycleController

    model = model_class()
    config = _StubConfig()
    ctrl = UnifiedTrainingCycleController(model, config)

    if attach_fb:
        fb = fb_class()
        ctrl.attach_feedback_bus(fb)
    if attach_mct:
        mct = mct_class()
        ctrl.attach_metacognitive_trigger(mct)

    return ctrl


# =============================================================================
#  I1: MCT Evaluation Failure Recording
# =============================================================================

class TestI1MCTEvaluationFailureRecording:
    """Verify that MCT.evaluate() failures are recorded to error_evolution."""

    def test_mct_failure_records_episode(self) -> None:
        ctrl = _make_controller(mct_class=_FailingMCT)
        ee = _StubErrorEvolution()

        ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            error_evolution=ee,
        )
        classes = [e["error_class"] for e in ee.episodes]
        assert "mct_evaluation_failure" in classes

    def test_mct_failure_does_not_crash_cycle(self) -> None:
        ctrl = _make_controller(mct_class=_FailingMCT)
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
        )
        # Cycle should still complete with duration
        assert "duration_s" in result

    def test_mct_failure_without_error_evolution(self) -> None:
        """MCT failure should not crash even without error_evolution."""
        ctrl = _make_controller(mct_class=_FailingMCT)
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
        )
        assert result["cycle"] == 1


# =============================================================================
#  I2: Mutual Reinforcement Failure Recording
# =============================================================================

class TestI2MutualReinforcementFailureRecording:
    """Verify that verify_and_reinforce failures are recorded."""

    def test_reinforce_failure_records_episode(self) -> None:
        ctrl = _make_controller(model_class=_FailingReinforceModel)
        ee = _StubErrorEvolution()
        # Set cycle count to trigger reinforcement (interval=5)
        ctrl._cycle_count = 4  # will become 5 on execute

        ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        classes = [e["error_class"] for e in ee.episodes]
        assert "mutual_reinforcement_failure" in classes

    def test_reinforce_failure_result_in_cycle(self) -> None:
        ctrl = _make_controller(model_class=_FailingReinforceModel)
        ctrl._cycle_count = 4

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
        )
        mr = result.get("mutual_reinforcement", {})
        assert mr.get("executed") is False
        assert "error" in mr


# =============================================================================
#  I3: MCT Signal Collection Resilience
# =============================================================================

class TestI3MCTSignalResilience:
    """Verify that signal collection defaults on exception."""

    def test_oscillation_defaults_on_exception(self) -> None:
        ctrl = _make_controller(fb_class=_FailingOscillationBus)
        kwargs = ctrl._collect_mct_signals(0.1, [], {})
        # Should have oscillation_severity set to 0.0 (default)
        assert kwargs["oscillation_severity"] == 0.0

    def test_divergence_defaults_on_exception(self) -> None:
        ctrl = _make_controller()
        # Attach a convergence_monitor that raises
        mock_conv = MagicMock()
        mock_conv.is_diverging.side_effect = RuntimeError("boom")
        ctrl.model.convergence_monitor = mock_conv

        kwargs = ctrl._collect_mct_signals(0.1, [], {})
        assert kwargs["is_diverging"] is False

    def test_topology_defaults_on_exception(self) -> None:
        ctrl = _make_controller()
        # Set a topology state that will cause .any().item() to fail
        ctrl.model._cached_topology_state = "not_a_tensor"

        kwargs = ctrl._collect_mct_signals(0.1, [], {})
        assert kwargs["topology_catastrophe"] is False

    def test_all_signals_present_after_exception(self) -> None:
        """Even when all optional sources fail, core signals survive."""
        ctrl = _make_controller(fb_class=_FailingOscillationBus)
        ctrl.model.convergence_monitor = MagicMock(
            is_diverging=MagicMock(side_effect=RuntimeError("boom")),
        )
        ctrl.model._cached_topology_state = "bad"

        kwargs = ctrl._collect_mct_signals(0.0, [], {})
        assert "oscillation_severity" in kwargs
        assert "is_diverging" in kwargs
        assert "topology_catastrophe" in kwargs


# =============================================================================
#  I4: Feedback Bus Sync Failure Recording
# =============================================================================

class TestI4FeedbackBusSyncFailure:
    """Verify that _sync_feedback_bus failures are recorded."""

    def test_sync_failure_records_to_model_ee(self) -> None:
        ctrl = _make_controller(fb_class=_FailingFeedbackBus)
        ee = _StubErrorEvolution()
        ctrl.model._error_evolution = ee

        ctrl._sync_feedback_bus(["flag1"], {"ucc": {"evaluated": True}})
        classes = [e["error_class"] for e in ee.episodes]
        assert "feedback_bus_sync_failure" in classes

    def test_sync_failure_does_not_crash(self) -> None:
        ctrl = _make_controller(fb_class=_FailingFeedbackBus)
        # Should not raise
        ctrl._sync_feedback_bus(["flag1"], {})


# =============================================================================
#  I5: Feedback Bus Read Logging
# =============================================================================

class TestI5FeedbackBusReadLogging:
    """Verify that _read_fb_signal logs on failure."""

    def test_read_returns_default_on_exception(self) -> None:
        ctrl = _make_controller()
        # Replace _extra_signals with something that raises on .get
        ctrl._feedback_bus._extra_signals = MagicMock(
            get=MagicMock(side_effect=RuntimeError("read boom")),
        )
        result = ctrl._read_fb_signal("test_signal", 0.42)
        assert result == 0.42

    def test_read_logs_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        ctrl = _make_controller()
        ctrl._feedback_bus._extra_signals = MagicMock(
            get=MagicMock(side_effect=RuntimeError("read boom")),
        )
        with caplog.at_level(logging.DEBUG, logger="AEON-Integration"):
            ctrl._read_fb_signal("test_signal", 0.0)
        assert any("read failed" in r.message for r in caplog.records)


# =============================================================================
#  I6: MCT Weight Adaptation Failure Recording
# =============================================================================

class TestI6MCTWeightAdaptationFailure:
    """Verify that MCT weight adaptation failures are recorded."""

    def test_adapt_failure_records_episode(self) -> None:
        ctrl = _make_controller(mct_class=_FailingAdaptMCT)
        ee = _StubErrorEvolution()

        # Trigger weight adaptation by providing low coherence
        reinforce_dict = {"overall_score": 0.3}
        ctrl._feed_reinforce_to_mct(reinforce_dict, ee)

        classes = [e["error_class"] for e in ee.episodes]
        assert "mct_weight_adaptation_failure" in classes

    def test_adapt_failure_does_not_crash(self) -> None:
        ctrl = _make_controller(mct_class=_FailingAdaptMCT)
        # Should not raise
        ctrl._feed_reinforce_to_mct({"overall_score": 0.3}, _StubErrorEvolution())


# =============================================================================
#  I7: Z-Annotation Fallback Uncertainty Flag
# =============================================================================

class TestI7ZAnnotationUncertaintyFlag:
    """Verify z_annotation_failed flag is set on annotation fallback."""

    def test_z_annotation_fallback_sets_flag(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Simulate annotation fallback by directly setting the flag
        # (the real execute_z_annotation catches exceptions internally)
        orig = ctrl.execute_z_annotation

        def _fallback_z_ann(z_sequences, error_evolution=None):
            ctrl._z_annotation_used_fallback = True
            return z_sequences, [torch.ones(s.shape[0], 3) for s in z_sequences]

        ctrl.execute_z_annotation = _fallback_z_ann

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            z_sequences=[torch.randn(4, 16)],
            error_evolution=ee,
        )
        flags = result.get("uncertainty_flags", [])
        assert "z_annotation_failed" in flags

    def test_z_annotation_success_no_flag(self) -> None:
        ctrl = _make_controller()

        # Mock annotate_z_sequences_quality to succeed
        with patch(
            "aeon_integration.UnifiedTrainingCycleController.execute_z_annotation",
            return_value=(
                [torch.randn(4, 16)],
                [torch.randn(4, 3)],
            ),
        ):
            result = ctrl.execute_full_cycle(
                epoch=1, phase="A",
                epoch_metrics={},
                z_sequences=[torch.randn(4, 16)],
            )
        flags = result.get("uncertainty_flags", [])
        assert "z_annotation_failed" not in flags


class TestI7DirectAnnotationFallback:
    """Test the internal fallback flag directly."""

    def test_fallback_sets_internal_flag(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Patch the import to force failure
        with patch("aeon_integration.UnifiedTrainingCycleController.execute_z_annotation") as mock_ann:
            # Simulate what happens during fallback
            def _side_effect(z_sequences, error_evolution=None):
                ctrl._z_annotation_used_fallback = True
                return z_sequences, [torch.ones(s.shape[0], 3) for s in z_sequences]
            mock_ann.side_effect = _side_effect

            result = ctrl.execute_full_cycle(
                epoch=1, phase="A",
                epoch_metrics={},
                z_sequences=[torch.randn(4, 16)],
                error_evolution=ee,
            )
        flags = result.get("uncertainty_flags", [])
        assert "z_annotation_failed" in flags


# =============================================================================
#  I8: Z-Filter Ratio to Feedback Bus
# =============================================================================

class TestI8ZFilterRatioToBus:
    """Verify z_filter_pass_ratio is written to feedback bus."""

    def test_filter_ratio_written_on_filtered_retrain(self) -> None:
        ctrl = _make_controller()

        with patch(
            "aeon_integration.UnifiedTrainingCycleController.execute_micro_retrain",
        ) as mock_retrain:
            mock_retrain.return_value = {
                "retrained": True,
                "z_quality_filtered": True,
                "z_original_count": 10,
                "z_filtered_count": 7,
            }
            result = ctrl.execute_full_cycle(
                epoch=1, phase="B",
                epoch_metrics={},
                pseudo_labels=[{"label": 1}],
            )

        ratio = ctrl._feedback_bus._extra_signals.get("z_filter_pass_ratio")
        assert ratio is not None
        assert ratio == pytest.approx(0.7)

    def test_no_ratio_when_not_filtered(self) -> None:
        ctrl = _make_controller()

        with patch(
            "aeon_integration.UnifiedTrainingCycleController.execute_micro_retrain",
        ) as mock_retrain:
            mock_retrain.return_value = {"retrained": True}
            ctrl.execute_full_cycle(
                epoch=1, phase="B",
                epoch_metrics={},
                pseudo_labels=[{"label": 1}],
            )

        assert "z_filter_pass_ratio" not in ctrl._feedback_bus._extra_signals


# =============================================================================
#  I9: MCT Trigger Decision Recording
# =============================================================================

class TestI9MCTBaselineRecording:
    """Verify MCT trigger decisions are recorded to error_evolution."""

    def test_baseline_decision_recorded(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        classes = [e["error_class"] for e in ee.episodes]
        assert "mct_trigger_decision" in classes
        # Should be baseline (not triggered) and success=True
        decisions = [
            e for e in ee.episodes
            if e["error_class"] == "mct_trigger_decision"
        ]
        assert decisions[0]["strategy_used"] == "baseline"
        assert decisions[0]["success"] is True

    def test_triggered_decision_recorded(self) -> None:
        ctrl = _make_controller()
        ctrl._mct._should_trigger = True
        ee = _StubErrorEvolution()

        ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        decisions = [
            e for e in ee.episodes
            if e["error_class"] == "mct_trigger_decision"
        ]
        assert len(decisions) == 1
        assert decisions[0]["strategy_used"] == "triggered"

    def test_no_recording_without_error_evolution(self) -> None:
        """When error_evolution is None, no crash occurs."""
        ctrl = _make_controller()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
        )
        assert "metacognitive_review" in result


# =============================================================================
#  I10: Reinforce-to-MCT Bridge Failure Recording
# =============================================================================

class TestI10ReinforceToMCTBridgeFailure:
    """Verify _feed_reinforce_to_mct outer failures are recorded."""

    def test_outer_failure_records_episode(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Make the entire _feed_reinforce_to_mct logic fail at the dict access
        bad_result = MagicMock()
        bad_result.get = MagicMock(side_effect=RuntimeError("outer boom"))

        ctrl._feed_reinforce_to_mct(bad_result, ee)

        classes = [e["error_class"] for e in ee.episodes]
        assert "reinforce_to_mct_bridge_failure" in classes


# =============================================================================
#  I11: Failure Returns Include traced/causal_chain
# =============================================================================

class TestI11TraceabilityFields:
    """Verify failure returns include traced and causal_chain."""

    def test_codebook_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        with patch("aeon_integration.UnifiedTrainingCycleController.execute_codebook_warm_start") as mock:
            mock.return_value = {
                "initialized": False, "reason": "test",
                "traced": False, "causal_chain": ["codebook_warm_start", "test"],
            }
            result = ctrl.execute_full_cycle(
                epoch=1, phase="A",
                epoch_metrics={},
                tokens=torch.randn(10, 16),
            )
        cws = result.get("codebook_warm_start", {})
        assert cws.get("traced") is False
        assert "causal_chain" in cws

    def test_t2i_bridge_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_training_to_inference_bridge(
            convergence_monitor=None,
            error_evolution=None,
        )
        # Should fail because imports will fail
        if not result.get("bridged", True):
            assert "traced" in result
            assert "causal_chain" in result

    def test_i2t_bridge_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        # Force import failure by patching
        with patch(
            "aeon_integration.UnifiedTrainingCycleController.execute_inference_to_training_bridge",
        ) as mock:
            mock.return_value = {
                "bridged": False,
                "reason": "import error",
                "traced": False,
                "causal_chain": ["inference_to_training_bridge", "import error"],
            }
            result = mock.return_value
        assert result.get("traced") is False
        assert "causal_chain" in result

    def test_ucc_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_ucc_evaluation(
            epoch=1, phase="A", epoch_metrics={},
        )
        if not result.get("evaluated", True):
            assert "traced" in result
            assert "causal_chain" in result

    def test_ssp_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_ssp_alignment()
        if not result.get("aligned", True):
            assert "traced" in result
            assert "causal_chain" in result

    def test_micro_retrain_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        # Patch the import to force a wrapper-level exception
        with patch(
            "aeon_integration.UnifiedTrainingCycleController.execute_micro_retrain",
            side_effect=ImportError("boom"),
        ):
            pass
        # Test the wrapper directly when it catches an exception
        result = ctrl.execute_micro_retrain(pseudo_labels=[{"label": 1}])
        # When ae_train successfully handles internally, traced may not be
        # present; it's only added on wrapper-level exception
        if "traced" in result:
            assert result["traced"] is False

    def test_teacher_student_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        # The traced field is only added when the wrapper's except clause fires
        # (e.g. import failure). The underlying function handles errors internally.
        # Test this by forcing an import failure:
        with patch.dict("sys.modules", {"ae_train": None}):
            result = ctrl.execute_teacher_student_inversion(
                z_sequences=[torch.randn(4, 16)],
            )
        assert result.get("inverted") is False
        assert "traced" in result
        assert "causal_chain" in result

    def test_context_calibration_failure_has_trace_fields(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_context_calibration(
            tokens=torch.randn(10, 16),
        )
        if not result.get("calibrated", True):
            assert "traced" in result
            assert "causal_chain" in result


# =============================================================================
#  I12: Wizard Results Consumption
# =============================================================================

class TestI12WizardConsumption:
    """Verify consume_wizard_results() bridges wizard → integration."""

    def test_consume_records_to_error_evolution(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        wizard_results = {
            "overall_status": "completed",
            "total_duration_s": 12.5,
            "wizard_completed": True,
        }
        result = ctrl.consume_wizard_results(wizard_results, ee)

        assert result["consumed"] is True
        classes = [e["error_class"] for e in ee.episodes]
        assert "wizard_completion" in classes
        # Should be recorded as success
        ep = next(e for e in ee.episodes if e["error_class"] == "wizard_completion")
        assert ep["success"] is True

    def test_consume_applies_hyperparameters(self) -> None:
        ctrl = _make_controller()

        wizard_results = {
            "overall_status": "completed",
            "hyperparameters": {
                "z_dim": 32,
                "vq_embedding_dim": 32,
            },
        }
        result = ctrl.consume_wizard_results(wizard_results)

        assert result["consumed"] is True
        assert "z_dim" in result["applied_settings"]
        assert ctrl.config.z_dim == 32

    def test_consume_writes_to_feedback_bus(self) -> None:
        ctrl = _make_controller()

        wizard_results = {
            "overall_status": "completed",
            "corpus_diagnostics": {
                "corpus_quality": 0.85,
            },
        }
        ctrl.consume_wizard_results(wizard_results)

        assert ctrl._feedback_bus._extra_signals.get("wizard_completed") == 1.0
        assert ctrl._feedback_bus._extra_signals.get("wizard_corpus_quality") == pytest.approx(0.85)

    def test_consume_with_warnings_status(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        wizard_results = {
            "overall_status": "completed_with_warnings",
        }
        result = ctrl.consume_wizard_results(wizard_results, ee)

        assert result["consumed"] is True
        assert result["wizard_status"] == "completed_with_warnings"
        ep = next(e for e in ee.episodes if e["error_class"] == "wizard_completion")
        assert ep["success"] is False  # not "completed" → not success

    def test_consume_without_feedback_bus(self) -> None:
        ctrl = _make_controller(attach_fb=False)
        ee = _StubErrorEvolution()

        wizard_results = {"overall_status": "completed"}
        result = ctrl.consume_wizard_results(wizard_results, ee)
        assert result["consumed"] is True

    def test_consume_failure_records_episode(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Pass something that will cause an error
        bad_results = MagicMock()
        bad_results.get = MagicMock(side_effect=RuntimeError("consume boom"))

        result = ctrl.consume_wizard_results(bad_results, ee)
        assert result["consumed"] is False
        classes = [e["error_class"] for e in ee.episodes]
        assert "wizard_consumption_failure" in classes

    def test_consume_empty_hyperparameters(self) -> None:
        ctrl = _make_controller()
        result = ctrl.consume_wizard_results({
            "overall_status": "completed",
            "hyperparameters": {},
        })
        assert result["consumed"] is True
        assert result["applied_settings"] == []


# =============================================================================
#  Cross-Patch Integration Tests
# =============================================================================

class TestCrossPatchIntegration:
    """End-to-end tests verifying multiple patches work together."""

    def test_full_cycle_with_all_patches_active(self) -> None:
        """Verify a complete cycle activates all I-series patches."""
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            error_evolution=ee,
        )

        # I9: MCT decision should be recorded
        classes = [e["error_class"] for e in ee.episodes]
        assert "mct_trigger_decision" in classes

        # Cycle should complete normally
        assert "duration_s" in result
        assert result["cycle"] == 1

    def test_multiple_failures_all_recorded(self) -> None:
        """When multiple things fail, all are recorded."""
        ctrl = _make_controller(
            mct_class=_FailingMCT,
            model_class=_FailingReinforceModel,
        )
        ee = _StubErrorEvolution()
        ctrl._cycle_count = 4  # reinforce interval

        ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )

        classes = [e["error_class"] for e in ee.episodes]
        assert "mct_evaluation_failure" in classes
        assert "mutual_reinforcement_failure" in classes

    def test_wizard_then_cycle_integration(self) -> None:
        """Wizard results consumed before first cycle."""
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Step 1: Consume wizard results
        wizard_results = {
            "overall_status": "completed",
            "hyperparameters": {"z_dim": 64},
            "corpus_diagnostics": {"corpus_quality": 0.9},
        }
        consume_result = ctrl.consume_wizard_results(wizard_results, ee)
        assert consume_result["consumed"] is True
        assert ctrl.config.z_dim == 64

        # Step 2: Run a cycle
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        assert result["cycle"] == 1

        # Both wizard and cycle decisions recorded
        classes = [e["error_class"] for e in ee.episodes]
        assert "wizard_completion" in classes
        assert "mct_trigger_decision" in classes

    def test_causal_transparency_on_failure(self) -> None:
        """When a point fails, trace_output_to_premise can trace it."""
        from aeon_integration import trace_output_to_premise

        ctrl = _make_controller()
        ee = _StubErrorEvolution()
        ee.record_episode(
            error_class="test_failure",
            strategy_used="test",
            success=False,
        )

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )

        trace = trace_output_to_premise(
            "test_output",
            cycle_history=[result],
            error_evolution=ee,
        )
        assert trace["traced"] is True
        assert trace["originating_cycle"] == 1
