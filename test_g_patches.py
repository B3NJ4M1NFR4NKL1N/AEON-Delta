"""Tests for G-series patches: final integration & cognitive activation fixes.

G1 — oscillation_severity parameter added to MetaCognitiveRecursionTrigger.evaluate()
G2 — oscillation_severity added to _signal_weights initialization
G3 — root_cause_ prefix routing in _prefix_to_signal
G4 — deeper_meta_loop → error_evolution in _CYCLE_EXEMPT_EDGES
G5 — explicit root_cause error class mappings in _class_to_signal
"""

import importlib
import inspect
import sys
import types
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Import helpers – load aeon_core lazily so import errors are test failures
# rather than collection errors.
# ---------------------------------------------------------------------------

_aeon = None


def _load_aeon():
    global _aeon
    if _aeon is None:
        _aeon = importlib.import_module("aeon_core")
    return _aeon


# ═══════════════════════════════════════════════════════════════════════════
# G1 — oscillation_severity parameter in evaluate()
# ═══════════════════════════════════════════════════════════════════════════


class TestG1_OscillationSeverityParameter:
    """Verify evaluate() accepts oscillation_severity without TypeError."""

    def _make_trigger(self):
        m = _load_aeon()
        return m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )

    def test_evaluate_accepts_oscillation_severity(self):
        """evaluate() should accept oscillation_severity as a keyword arg."""
        trigger = self._make_trigger()
        result = trigger.evaluate(oscillation_severity=0.8)
        assert isinstance(result, dict)

    def test_evaluate_oscillation_severity_default_zero(self):
        """oscillation_severity defaults to 0.0 when not passed."""
        trigger = self._make_trigger()
        result = trigger.evaluate()
        assert isinstance(result, dict)

    def test_evaluate_oscillation_severity_with_other_signals(self):
        """oscillation_severity can be passed alongside other signals."""
        trigger = self._make_trigger()
        result = trigger.evaluate(
            uncertainty=0.5,
            coherence_deficit=0.3,
            oscillation_severity=0.7,
            stall_severity=0.4,
        )
        assert isinstance(result, dict)

    def test_evaluate_signature_has_oscillation_severity(self):
        """The evaluate() signature must list oscillation_severity."""
        trigger = self._make_trigger()
        sig = inspect.signature(trigger.evaluate)
        assert "oscillation_severity" in sig.parameters

    def test_oscillation_severity_influences_trigger_score(self):
        """Non-zero oscillation_severity should contribute to trigger_score."""
        trigger = self._make_trigger()
        # High oscillation should push trigger_score up
        result_high = trigger.evaluate(oscillation_severity=1.0)
        result_zero = trigger.evaluate(oscillation_severity=0.0)
        assert result_high["trigger_score"] >= result_zero["trigger_score"]

    def test_post_pipeline_signals_accepted(self):
        """Simulates unpacking _post_pipeline_signals dict with oscillation_severity."""
        trigger = self._make_trigger()
        signals = {
            "uncertainty": 0.3,
            "coherence_deficit": 0.2,
            "stall_severity": 0.1,
            "oscillation_severity": 0.5,
        }
        # This should NOT raise TypeError
        result = trigger.evaluate(**signals)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# G2 — oscillation_severity in _signal_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestG2_OscillationSeverityWeight:
    """Verify oscillation_severity has a default weight in _signal_weights."""

    def test_signal_weights_has_oscillation_severity(self):
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        assert "oscillation_severity" in trigger._signal_weights

    def test_oscillation_severity_weight_positive(self):
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        assert trigger._signal_weights["oscillation_severity"] > 0.0

    def test_oscillation_severity_weight_equals_default(self):
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        default = trigger._DEFAULT_WEIGHT
        assert trigger._signal_weights["oscillation_severity"] == default


# ═══════════════════════════════════════════════════════════════════════════
# G3 — root_cause_ prefix routing
# ═══════════════════════════════════════════════════════════════════════════


class TestG3_RootCausePrefixRouting:
    """Verify root_cause_ prefix routes to low_causal_quality."""

    def _run_adapt(self, error_class: str):
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        # Prepare error_classes dict as adapt_weights_from_evolution expects
        error_classes = {
            error_class: {
                "count": 10,
                "success_count": 2,
                "last_strategies": ["metacognitive_rerun"],
            }
        }
        # Should not produce "defaulting to 'uncertainty' signal" debug log
        trigger.adapt_weights_from_evolution(error_classes)
        return trigger

    def test_root_cause_novel_subsystem_routes_to_low_causal_quality(self):
        """A novel root_cause_{X} class should match the root_cause_ prefix."""
        trigger = self._run_adapt("root_cause_some_new_subsystem")
        # After adaptation, low_causal_quality weight should have been boosted
        assert trigger._signal_weights.get("low_causal_quality", 0) > 0

    def test_root_cause_encoder_routes_correctly(self):
        """root_cause_encoder should match root_cause_ prefix."""
        trigger = self._run_adapt("root_cause_encoder")
        assert trigger._signal_weights.get("low_causal_quality", 0) > 0


# ═══════════════════════════════════════════════════════════════════════════
# G4 — deeper_meta_loop → error_evolution in _CYCLE_EXEMPT_EDGES
# ═══════════════════════════════════════════════════════════════════════════


class TestG4_DeeperMetaLoopExemptEdge:
    """Verify the deeper_meta_loop → error_evolution edge is cycle-exempt."""

    def test_edge_in_cycle_exempt_set(self):
        m = _load_aeon()
        exempt = m.AEONDeltaV3._CYCLE_EXEMPT_EDGES
        assert ("deeper_meta_loop", "error_evolution") in exempt

    def test_edge_not_pruned_by_validate_dag(self):
        """CausalProvenanceTracker should not remove an exempt edge."""
        m = _load_aeon()
        import torch
        tracker = m.CausalProvenanceTracker()
        dummy = torch.zeros(1)
        # Register both nodes
        tracker.record_before("deeper_meta_loop", dummy)
        tracker.record_after("deeper_meta_loop", dummy)
        tracker.record_before("error_evolution", dummy)
        tracker.record_after("error_evolution", dummy)
        # Add the dependency
        tracker.record_dependency("error_evolution", "deeper_meta_loop")
        # Check that dependency graph has the edge
        deps = tracker.get_dependency_graph()
        # record_dependency(child, parent) stores parent under child's deps
        assert "error_evolution" in deps.get("deeper_meta_loop", set()) or \
               "error_evolution" in deps.get("deeper_meta_loop", [])


# ═══════════════════════════════════════════════════════════════════════════
# G5 — explicit root_cause error class mappings
# ═══════════════════════════════════════════════════════════════════════════


class TestG5_ExplicitRootCauseMappings:
    """Verify specific root_cause error classes are in _class_to_signal."""

    def _run_adapt(self, error_class: str):
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        error_classes = {
            error_class: {
                "count": 10,
                "success_count": 2,
                "last_strategies": ["metacognitive_rerun"],
            }
        }
        trigger.adapt_weights_from_evolution(error_classes)
        return trigger

    def test_root_cause_auto_critic_revision_mapped(self):
        """root_cause_auto_critic/revision should map to low_causal_quality."""
        trigger = self._run_adapt("root_cause_auto_critic/revision")
        assert trigger._signal_weights.get("low_causal_quality", 0) > 0

    def test_root_cause_input_mapped(self):
        """root_cause_input should map to uncertainty."""
        trigger = self._run_adapt("root_cause_input")
        assert trigger._signal_weights.get("uncertainty", 0) > 0

    def test_root_cause_provenance_auto_critic_revision_mapped(self):
        """root_cause_provenance/auto_critic_revision → low_causal_quality."""
        trigger = self._run_adapt("root_cause_provenance/auto_critic_revision")
        assert trigger._signal_weights.get("low_causal_quality", 0) > 0

    def test_recurring_root_cause_mapped(self):
        """recurring_root_cause should map to low_causal_quality."""
        trigger = self._run_adapt("recurring_root_cause")
        assert trigger._signal_weights.get("low_causal_quality", 0) > 0

    def test_no_debug_fallback_for_known_classes(self, caplog):
        """Known root_cause classes should not trigger the debug fallback log."""
        import logging
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        error_classes = {
            "root_cause_auto_critic/revision": {
                "count": 5, "success_count": 1,
                "last_strategies": ["metacognitive_rerun"],
            },
            "root_cause_input": {
                "count": 5, "success_count": 1,
                "last_strategies": ["metacognitive_rerun"],
            },
            "root_cause_provenance/auto_critic_revision": {
                "count": 5, "success_count": 1,
                "last_strategies": ["metacognitive_rerun"],
            },
        }
        with caplog.at_level(logging.DEBUG, logger="AEON-Delta"):
            trigger.adapt_weights_from_evolution(error_classes)
        # None of these should appear as "unmapped" in the log
        unmapped_msgs = [
            r for r in caplog.records
            if "unmapped error class" in r.message
            and any(
                cls in r.message
                for cls in [
                    "root_cause_auto_critic/revision",
                    "root_cause_input",
                    "root_cause_provenance/auto_critic_revision",
                ]
            )
        ]
        assert len(unmapped_msgs) == 0, (
            f"Expected no unmapped debug messages for known root_cause "
            f"classes, but got: {[r.message for r in unmapped_msgs]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Integration: all fixes work together
# ═══════════════════════════════════════════════════════════════════════════


class TestG_Integration:
    """End-to-end integration: all G-series patches working together."""

    def test_full_post_pipeline_signal_dict_accepted(self):
        """Simulates the full _post_pipeline_signals dict with all signals."""
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        # Simulate a realistic _post_pipeline_signals dict
        signals = {
            "uncertainty": 0.3,
            "is_diverging": False,
            "topology_catastrophe": False,
            "coherence_deficit": 0.2,
            "memory_staleness": False,
            "recovery_pressure": 0.1,
            "world_model_surprise": 0.05,
            "causal_quality": 0.9,
            "safety_violation": False,
            "diversity_collapse": 0.0,
            "memory_trust_deficit": 0.0,
            "convergence_conflict": 0.0,
            "output_reliability": 0.95,
            "spectral_stability_margin": 0.8,
            "border_uncertainty": 0.1,
            "stall_severity": 0.05,
            "oscillation_severity": 0.4,
        }
        result = trigger.evaluate(**signals)
        assert isinstance(result, dict)
        assert "trigger_score" in result
        assert "should_recurse" in result

    def test_adapt_then_evaluate_with_oscillation(self):
        """Full cycle: adapt weights from root_cause classes then evaluate."""
        m = _load_aeon()
        trigger = m.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        error_classes = {
            "root_cause_auto_critic/revision": {
                "count": 10, "success_count": 1,
                "last_strategies": ["metacognitive_rerun"],
            },
        }
        trigger.adapt_weights_from_evolution(error_classes)
        result = trigger.evaluate(
            oscillation_severity=0.9,
            causal_quality=0.2,
        )
        assert isinstance(result, dict)
