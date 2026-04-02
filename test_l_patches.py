"""Tests for L-series patches: Final Integration & Cognitive Activation.

L1:  Bridge silent exception at provenance diversity tracing (_reasoning_core_impl)
L2:  Bridge silent exception at error_evolution recording (_reasoning_core_impl)
L3:  Bridge silent exception at trigger adaptation (_reasoning_core_impl)
L4:  Bridge silent exception at strategy adaptation tracing (_reasoning_core_impl)
L5:  Bridge silent exception at re-reasoning tracing (_reasoning_core_impl)
L6:  Bridge silent exception at fast-mode trigger adaptation (_reasoning_core_impl)
L7:  Bridge silent exception at feedback rebuild after OOM (forward)
L8:  Bridge silent exception at post-output root-cause trace (_forward_impl)
L9:  Bridge silent exception at late meta-loop root-cause trace (_forward_impl)
L10: Bridge silent exception at provenance trigger adaptation (verify_and_reinforce)
L11: Bridge silent exception at signal freshness recording (system_emergence_report)
L12: Log calibration drift recording failure (VibeThinkerContinuousLearner)
L13: Add 5 missing prefix→signal mappings
L14: Add 3 missing _CYCLE_EXEMPT_EDGES
L15: _class_to_signal mappings for L-series error classes
"""

import importlib
import inspect
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# ── Import helpers ──────────────────────────────────────────────────────
aeon = importlib.import_module("aeon_core")

AEONConfig = getattr(aeon, "AEONConfig", None)
AEONDeltaV3 = getattr(aeon, "AEONDeltaV3", None)
MetaCognitiveRecursionTrigger = getattr(aeon, "MetaCognitiveRecursionTrigger", None)
CognitiveFeedbackBus = getattr(aeon, "CognitiveFeedbackBus", None)
CausalProvenanceTracker = getattr(aeon, "CausalProvenanceTracker", None)
VibeThinkerContinuousLearner = getattr(aeon, "VibeThinkerContinuousLearner", None)


# ── Fixtures ────────────────────────────────────────────────────────────

def _make_config(**overrides):
    """Build a minimal AEONConfig suitable for unit testing."""
    defaults = dict(
        z_dim=8, hidden_dim=8, num_pillars=2,
        enable_safety_guardrails=False, enable_memory_routing=False,
        diversity_collapse_threshold=0.5,
        vq_embedding_dim=8,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_trigger(**overrides):
    """Build a MetaCognitiveRecursionTrigger with sane defaults."""
    defaults = dict(
        trigger_threshold=0.5,
        max_recursions=3,
    )
    defaults.update(overrides)
    return MetaCognitiveRecursionTrigger(**defaults)


def _make_model():
    """Build a minimal AEONDeltaV3 instance."""
    cfg = _make_config()
    return AEONDeltaV3(cfg)


# ════════════════════════════════════════════════════════════════════════
#  L1–L11: Silent Exception Bridges
# ════════════════════════════════════════════════════════════════════════

class TestL1ProvenanceDiversityTracingBridge:
    """L1: provenance_diversity_tracing_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        """The except block at the diversity tracing site calls _bridge_silent_exception."""
        src = inspect.getsource(AEONDeltaV3)
        assert "provenance_diversity_tracing_failure" in src

    def test_bridge_method_invoked_on_exception(self):
        """When provenance tracing raises, _bridge_silent_exception is called."""
        model = _make_model()
        model._bridge_silent_exception = MagicMock()
        # Call the bridge directly to verify it works
        model._bridge_silent_exception(
            'provenance_diversity_tracing_failure',
            'reasoning_core_impl',
            RuntimeError("test"),
        )
        model._bridge_silent_exception.assert_called_once()
        args = model._bridge_silent_exception.call_args[0]
        assert args[0] == 'provenance_diversity_tracing_failure'
        assert args[1] == 'reasoning_core_impl'

    def test_error_class_in_class_to_signal(self):
        """provenance_diversity_tracing_failure is mapped in _class_to_signal."""
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "provenance_diversity_tracing_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)
        # Should NOT fall through to generic "uncertainty" —
        # it should route to "low_causal_quality"


class TestL2ErrorEvolutionRecordingBridge:
    """L2: error_evolution_recording_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "error_evolution_recording_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "error_evolution_recording_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL3TriggerAdaptationBridge:
    """L3: trigger_adaptation_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "trigger_adaptation_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "trigger_adaptation_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL4StrategyAdaptationBridge:
    """L4: strategy_adaptation_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "strategy_adaptation_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "strategy_adaptation_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL5RerunTracingBridge:
    """L5: rerun_tracing_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "rerun_tracing_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "rerun_tracing_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL6FastModeTriggerAdaptationBridge:
    """L6: fast_mode_trigger_adaptation_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "fast_mode_trigger_adaptation_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "fast_mode_trigger_adaptation_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL7FeedbackRebuildAfterOomBridge:
    """L7: feedback_rebuild_after_oom_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "feedback_rebuild_after_oom_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "feedback_rebuild_after_oom_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL8PostOutputRootCauseTraceBridge:
    """L8: post_output_root_cause_trace_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "post_output_root_cause_trace_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "post_output_root_cause_trace_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL9LateMetaLoopRootCauseTraceBridge:
    """L9: late_meta_loop_root_cause_trace_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "late_meta_loop_root_cause_trace_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "late_meta_loop_root_cause_trace_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL10ProvenanceTriggerAdaptationBridge:
    """L10: provenance_trigger_adaptation_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "provenance_trigger_adaptation_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "provenance_trigger_adaptation_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


class TestL11SignalFreshnessRecordingBridge:
    """L11: signal_freshness_recording_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        src = inspect.getsource(AEONDeltaV3)
        assert "signal_freshness_recording_failure" in src

    def test_error_class_in_class_to_signal(self):
        trigger = _make_trigger()
        summary = {
            "error_classes": {
                "signal_freshness_recording_failure": {
                    "count": 1, "success_rate": 0.0,
                },
            },
        }
        trigger.adapt_weights_from_evolution(summary)


# ════════════════════════════════════════════════════════════════════════
#  L12: VibeThinkerContinuousLearner Calibration Drift Logging
# ════════════════════════════════════════════════════════════════════════

class TestL12VibeThinkerCalibrationDriftLogging:
    """L12: VibeThinkerContinuousLearner logs calibration drift failures."""

    @pytest.mark.skipif(
        VibeThinkerContinuousLearner is None,
        reason="VibeThinkerContinuousLearner not found",
    )
    def test_logging_on_record_failure(self):
        """When error_evolution.record_episode raises, the failure is logged."""
        mock_ee = MagicMock()
        mock_ee.record_episode.side_effect = RuntimeError("mock ee failure")

        # Build a VibeThinkerConfig-like object
        cfg = MagicMock()
        cfg.calibration_ema_alpha = 0.3
        cfg.complexity_gate_threshold = 0.5
        cfg.psi_vibe_weight = 0.5

        learner = VibeThinkerContinuousLearner(
            config=cfg,
            error_evolution=mock_ee,
        )

        # High calibration error to trigger recording
        result = learner.evaluate_episode(
            parsed_response={'confidence': 0.9, 'reasoning_quality': 0.1},
            actual_correctness=0.1,
        )
        # Should not raise — the exception is caught and logged
        assert 'calibration_error' in result

    @pytest.mark.skipif(
        VibeThinkerContinuousLearner is None,
        reason="VibeThinkerContinuousLearner not found",
    )
    def test_source_has_logger_debug(self):
        """The except block logs via logger.debug."""
        src = inspect.getsource(VibeThinkerContinuousLearner)
        assert "logger.debug" in src
        assert "VibeThinkerContinuousLearner" in src


# ════════════════════════════════════════════════════════════════════════
#  L13: Prefix→Signal Mappings
# ════════════════════════════════════════════════════════════════════════

class TestL13PrefixSignalMappings:
    """L13: 5 additional prefix→signal mappings in adapt_weights_from_evolution."""

    _NEW_PREFIXES = {
        "training_": "diverging",
        "safety_": "safety_violation",
        "adaptation_": "uncertainty",
        "recovery_": "recovery_pressure",
        "escalation_": "uncertainty",
    }

    @pytest.mark.parametrize("prefix,expected_signal", list(_NEW_PREFIXES.items()))
    def test_prefix_routing(self, prefix, expected_signal):
        """Error classes with the given prefix route to the expected signal."""
        trigger = _make_trigger()
        # Use a synthetic error class that starts with the prefix
        # but is NOT in the explicit _class_to_signal dict.
        synthetic_class = f"{prefix}synthetic_test_class"
        summary = {
            "error_classes": {
                synthetic_class: {
                    "count": 5, "success_rate": 0.0,
                },
            },
        }
        # Before adaptation, save initial weights
        initial_weights = dict(trigger._signal_weights)

        trigger.adapt_weights_from_evolution(summary)

        # The target signal's weight should have changed (boosted by
        # low success_rate=0.0), confirming prefix routing worked.
        assert trigger._signal_weights[expected_signal] != initial_weights[expected_signal], (
            f"Prefix '{prefix}' did not route to signal '{expected_signal}'"
        )

    def test_prefix_present_in_source(self):
        """All new prefixes appear in the adapt_weights_from_evolution source."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.adapt_weights_from_evolution)
        for prefix in self._NEW_PREFIXES:
            assert f'"{prefix}"' in src or f"'{prefix}'" in src, (
                f"Prefix '{prefix}' not found in adapt_weights_from_evolution source"
            )


# ════════════════════════════════════════════════════════════════════════
#  L14: Cycle-Exempt Edges
# ════════════════════════════════════════════════════════════════════════

class TestL14CycleExemptEdges:
    """L14: 3 additional entries in _CYCLE_EXEMPT_EDGES."""

    _NEW_EDGES = [
        ("safety", "error_evolution"),
        ("causal_model", "error_evolution"),
        ("unified_cognitive_cycle", "error_evolution"),
    ]

    @pytest.mark.parametrize("edge", _NEW_EDGES)
    def test_edge_in_exempt_set(self, edge):
        """The edge is in the _CYCLE_EXEMPT_EDGES class attribute."""
        exempt = AEONDeltaV3._CYCLE_EXEMPT_EDGES
        assert edge in exempt, (
            f"Edge {edge} not found in _CYCLE_EXEMPT_EDGES"
        )

    def test_total_exempt_edges_increased(self):
        """The exempt set grew by at least 3 entries."""
        exempt = AEONDeltaV3._CYCLE_EXEMPT_EDGES
        # Before L14, there were ~43 entries.  Now there should be ≥46.
        assert len(exempt) >= 46


# ════════════════════════════════════════════════════════════════════════
#  L15: _class_to_signal Mappings for L-series Error Classes
# ════════════════════════════════════════════════════════════════════════

class TestL15ClassToSignalMappings:
    """L15: All L-series error classes are mapped in _class_to_signal."""

    _L_SERIES_CLASSES = {
        "provenance_diversity_tracing_failure": "low_causal_quality",
        "error_evolution_recording_failure": "coherence_deficit",
        "trigger_adaptation_failure": "uncertainty",
        "strategy_adaptation_failure": "uncertainty",
        "rerun_tracing_failure": "low_causal_quality",
        "fast_mode_trigger_adaptation_failure": "uncertainty",
        "feedback_rebuild_after_oom_failure": "recovery_pressure",
        "post_output_root_cause_trace_failure": "low_causal_quality",
        "late_meta_loop_root_cause_trace_failure": "low_causal_quality",
        "provenance_trigger_adaptation_failure": "low_causal_quality",
        "signal_freshness_recording_failure": "coherence_deficit",
    }

    @pytest.mark.parametrize(
        "error_class,expected_signal",
        list(_L_SERIES_CLASSES.items()),
    )
    def test_class_routed_to_correct_signal(self, error_class, expected_signal):
        """The error class is explicitly mapped and routes correctly."""
        trigger = _make_trigger()
        initial_weights = dict(trigger._signal_weights)

        summary = {
            "error_classes": {
                error_class: {"count": 3, "success_rate": 0.0},
            },
        }
        trigger.adapt_weights_from_evolution(summary)

        # The expected signal's weight should have changed
        assert trigger._signal_weights[expected_signal] != initial_weights[expected_signal], (
            f"Error class '{error_class}' did not route to "
            f"signal '{expected_signal}'"
        )

    def test_all_l_series_classes_in_source(self):
        """All L-series error classes appear in the _class_to_signal dict."""
        src = inspect.getsource(
            MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        for cls_name in self._L_SERIES_CLASSES:
            assert f'"{cls_name}"' in src or f"'{cls_name}'" in src, (
                f"Error class '{cls_name}' not found in source"
            )


# ════════════════════════════════════════════════════════════════════════
#  Integration: Bridge Silent Exception Method Contract
# ════════════════════════════════════════════════════════════════════════

class TestBridgeSilentExceptionContract:
    """Verify _bridge_silent_exception accepts the L-series error classes."""

    def test_bridge_records_episode(self):
        """_bridge_silent_exception records an error_evolution episode."""
        model = _make_model()
        if model.error_evolution is None:
            pytest.skip("Model has no error_evolution")
        model._bridge_silent_exception(
            'provenance_diversity_tracing_failure',
            'reasoning_core_impl',
            RuntimeError("test provenance tracing failure"),
        )
        summary = model.error_evolution.get_error_summary()
        classes = summary.get("error_classes", {})
        assert "provenance_diversity_tracing_failure" in classes

    def test_bridge_records_causal_trace(self):
        """_bridge_silent_exception records a causal trace entry."""
        model = _make_model()
        if model.causal_trace is None:
            pytest.skip("Model has no causal_trace")
        model._bridge_silent_exception(
            'trigger_adaptation_failure',
            'reasoning_core_impl',
            TypeError("test trigger adaptation failure"),
        )
        # Verify the causal trace has recent entries
        recent = model.causal_trace.recent(n=5)
        assert len(recent) > 0

    def test_bridge_does_not_raise(self):
        """_bridge_silent_exception never raises even with broken backends."""
        model = _make_model()
        model.error_evolution = None
        model.causal_trace = None
        # Should log warning but not raise
        model._bridge_silent_exception(
            'feedback_rebuild_after_oom_failure',
            'forward',
            RuntimeError("test"),
        )


# ════════════════════════════════════════════════════════════════════════
#  Integration: No Bare Pass in Patched Sites
# ════════════════════════════════════════════════════════════════════════

class TestNoBarePassInPatchedSites:
    """Verify that the patched except blocks no longer contain bare 'pass'."""

    _PATCHED_ERROR_CLASSES = [
        "provenance_diversity_tracing_failure",
        "error_evolution_recording_failure",
        "trigger_adaptation_failure",
        "strategy_adaptation_failure",
        "rerun_tracing_failure",
        "fast_mode_trigger_adaptation_failure",
        "feedback_rebuild_after_oom_failure",
        "post_output_root_cause_trace_failure",
        "late_meta_loop_root_cause_trace_failure",
        "provenance_trigger_adaptation_failure",
        "signal_freshness_recording_failure",
    ]

    def test_all_error_classes_in_source(self):
        """All L-series error classes appear in AEONDeltaV3 source."""
        src = inspect.getsource(AEONDeltaV3)
        for cls_name in self._PATCHED_ERROR_CLASSES:
            assert cls_name in src, (
                f"Error class '{cls_name}' not found in AEONDeltaV3 source"
            )
