"""
Tests for C-series cognitive integration patches:
- C1: stall_severity signal propagation to all secondary evaluate() calls
- C2: stall_severity in _post_pipeline_signals dict
- C5: Evolved strategy modulation of metacognitive re-reasoning depth

Validates that the identified dead-end signal gaps are closed and that
error_evolution-learned strategies now influence actual re-reasoning behavior.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import importlib

aeon = importlib.import_module("aeon_core")


def _make_config(**overrides):
    """Create a minimal AEONConfig with sensible test defaults."""
    defaults = dict(
        hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
        seq_length=32, dropout_rate=0.0, meta_dim=32,
        lipschitz_target=0.9, vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


def _get_reasoning_core_source():
    """Return source of _reasoning_core_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)


def _get_forward_impl_source():
    """Return source of _forward_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._forward_impl)


def _get_emergence_source():
    """Return source of system_emergence_report."""
    return inspect.getsource(aeon.AEONDeltaV3.system_emergence_report)


def _get_ucc_evaluate_source():
    """Return source of UnifiedCognitiveCycle.evaluate."""
    return inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)


def _get_generate_source():
    """Return source of generate."""
    return inspect.getsource(aeon.AEONDeltaV3.generate)


# ═══════════════════════════════════════════════════════════════════════
# C1a: stall_severity in fast-mode deferred evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1aStallSeverityFastMode:
    """Verify stall_severity is passed in the fast-mode deferred evaluate()."""

    def test_fast_mode_evaluate_includes_stall_severity(self):
        src = _get_reasoning_core_source()
        # The fast-mode evaluate call should include stall_severity
        # Look for the pattern near the B1 border_uncertainty patch
        idx_b1 = src.find("Patch B1: Add border_uncertainty to fast-mode")
        assert idx_b1 > 0, "B1 patch marker not found"
        # stall_severity should appear after B1 in the same evaluate call
        idx_stall = src.find("stall_severity=self._cached_stall_severity", idx_b1)
        assert idx_stall > 0, (
            "stall_severity not found in fast-mode deferred evaluate()"
        )
        # Should be within ~200 chars of the B1 marker
        assert idx_stall - idx_b1 < 500, (
            "stall_severity too far from B1 marker — may be in wrong call"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1b: stall_severity in output reliability evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1bStallSeverityOutputReliability:
    """Verify stall_severity in the output reliability evaluate() call."""

    def test_output_reliability_evaluate_includes_stall_severity(self):
        src = _get_reasoning_core_source()
        # The C1b patch should be present
        assert "Patch C1b: stall_severity for output reliability" in src, (
            "C1b patch marker not found in _reasoning_core_impl"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1c: stall_severity in VQ evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1cStallSeverityVQ:
    """Verify stall_severity in the VQ codebook recovery evaluate() call."""

    def test_vq_evaluate_includes_stall_severity(self):
        src = _get_forward_impl_source()
        assert "Patch C1c: stall_severity for VQ eval" in src, (
            "C1c patch marker not found in _forward_impl"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1d: stall_severity in cache validity evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1dStallSeverityCache:
    """Verify stall_severity in the cache validity evaluate() call."""

    def test_cache_evaluate_includes_stall_severity(self):
        src = _get_forward_impl_source()
        assert "Patch C1d: stall_severity for cache eval" in src, (
            "C1d patch marker not found in _forward_impl"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1e: stall_severity in high-uncertainty evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1eStallSeverityHighUnc:
    """Verify stall_severity in the high-uncertainty evaluate() call."""

    def test_high_unc_evaluate_includes_stall_severity(self):
        src = _get_forward_impl_source()
        assert "Patch C1e: stall_severity for high-unc" in src, (
            "C1e patch marker not found in _forward_impl"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1f: stall_severity in moderate-uncertainty evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestC1fStallSeverityModerate:
    """Verify stall_severity in the moderate-uncertainty evaluate() call."""

    def test_moderate_evaluate_includes_stall_severity(self):
        src = _get_forward_impl_source()
        assert "Patch C1f: stall_severity for moderate path" in src, (
            "C1f patch marker not found in _forward_impl"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1g: stall_severity in critical-patch evaluate() (emergence report)
# ═══════════════════════════════════════════════════════════════════════

class TestC1gStallSeverityCriticalPatch:
    """Verify stall_severity in the critical-patch evaluate() call."""

    def test_critical_patch_evaluate_includes_stall_severity(self):
        src = _get_emergence_source()
        idx_b8 = src.find("Patch B8: Complete signal set for critical patch")
        assert idx_b8 > 0, "B8 patch marker not found"
        idx_stall = src.find(
            "stall_severity=self._cached_stall_severity", idx_b8,
        )
        assert idx_stall > 0, (
            "stall_severity not found in critical-patch evaluate()"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1h: stall_severity forwarded in UCC evaluate() → metacognitive trigger
# ═══════════════════════════════════════════════════════════════════════

class TestC1hStallSeverityUCCSignature:
    """Verify stall_severity is in the UCC evaluate() signature."""

    def test_ucc_evaluate_accepts_stall_severity(self):
        sig = inspect.signature(aeon.UnifiedCognitiveCycle.evaluate)
        assert 'stall_severity' in sig.parameters, (
            "stall_severity not in UCC evaluate() signature"
        )
        # Default should be 0.0
        param = sig.parameters['stall_severity']
        assert param.default == 0.0

    def test_ucc_evaluate_forwards_to_trigger(self):
        src = _get_ucc_evaluate_source()
        # Should forward stall_severity to metacognitive_trigger.evaluate()
        assert "stall_severity=stall_severity" in src, (
            "UCC evaluate() does not forward stall_severity to trigger"
        )


# ═══════════════════════════════════════════════════════════════════════
# C1i-k: stall_severity in main, fast-mode, and generate UCC calls
# ═══════════════════════════════════════════════════════════════════════

class TestC1iMainUCCCall:
    """Verify stall_severity in the main UCC evaluate() call."""

    def test_main_ucc_call_includes_stall_severity(self):
        src = _get_reasoning_core_source()
        # Find the main unified_cognitive_cycle.evaluate call
        idx = src.find("unified_cognitive_cycle.evaluate(")
        assert idx > 0
        # stall_severity should appear in the same call
        idx_stall = src.find("stall_severity=self._cached_stall_severity", idx)
        assert idx_stall > 0, (
            "stall_severity not found in main UCC evaluate() call"
        )


class TestC1jFastUCCCall:
    """Verify stall_severity in the fast-mode UCC evaluate() call."""

    def test_fast_ucc_call_includes_stall_severity(self):
        src = _get_reasoning_core_source()
        idx = src.find("Fast-mode cognitive signal bridge")
        assert idx > 0, "Fast-mode UCC bridge marker not found"
        idx_stall = src.find("stall_severity=self._cached_stall_severity", idx)
        assert idx_stall > 0, (
            "stall_severity not found in fast-mode UCC evaluate() call"
        )


class TestC1kGenerateUCCCall:
    """Verify stall_severity in the generate UCC evaluate() call."""

    def test_generate_ucc_call_includes_stall_severity(self):
        src = _get_generate_source()
        assert "Patch C1k: stall_severity in generate UCC" in src, (
            "C1k patch marker not found in generate()"
        )


# ═══════════════════════════════════════════════════════════════════════
# C2: stall_severity in _post_pipeline_signals dict
# ═══════════════════════════════════════════════════════════════════════

class TestC2PostPipelineStallSeverity:
    """Verify stall_severity is in the _post_pipeline_signals dict."""

    def test_post_pipeline_signals_includes_stall_severity(self):
        src = _get_forward_impl_source()
        idx = src.find("_post_pipeline_signals = {")
        assert idx > 0, "_post_pipeline_signals dict not found"
        # Find the closing brace of the dict
        idx_close = src.find("}", idx)
        assert idx_close > 0
        dict_text = src[idx:idx_close]
        assert "'stall_severity'" in dict_text, (
            "stall_severity key not found in _post_pipeline_signals dict"
        )


# ═══════════════════════════════════════════════════════════════════════
# C5: Evolved strategy modulation of re-reasoning depth
# ═══════════════════════════════════════════════════════════════════════

class TestC5EvolvedStrategyModulation:
    """Verify that evolved strategy from get_best_strategy() influences
    metacognitive re-reasoning depth and threshold."""

    def test_evolved_strategy_modulates_extra_iters(self):
        src = _get_reasoning_core_source()
        # The evolved strategy modulation should be near the _extra_iters
        idx = src.find("Patch C5: evolved strategy modulation")
        assert idx > 0, "C5 patch marker not found"

    def test_deeper_strategy_adds_iterations(self):
        src = _get_reasoning_core_source()
        idx = src.find("Patch C5: evolved strategy modulation")
        assert idx > 0
        # Verify the deeper branch adds iterations
        region = src[idx:idx + 1500]
        assert "deeper" in region, (
            "Deeper strategy keyword check not found"
        )
        assert "_extra_iters += 2" in region, (
            "Deeper strategy iteration boost not found"
        )

    def test_relax_strategy_adjusts_threshold(self):
        src = _get_reasoning_core_source()
        idx = src.find("Patch C5: evolved strategy modulation")
        assert idx > 0
        region = src[idx:idx + 1500]
        assert "relax" in region, (
            "Relax strategy keyword check not found"
        )
        assert "_tight_threshold" in region, (
            "Relax strategy threshold adjustment not found"
        )


# ═══════════════════════════════════════════════════════════════════════
# Structural: All evaluate() calls include stall_severity
# ═══════════════════════════════════════════════════════════════════════

class TestStallSeverityCompleteness:
    """Verify stall_severity is present at all key evaluate() call sites."""

    def test_stall_severity_in_signal_weights(self):
        """Verify stall_severity is a key in the _signal_weights dict."""
        config = _make_config()
        trigger = aeon.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
            tightening_factor=0.8,
            extra_iterations=2,
        )
        assert "stall_severity" in trigger._signal_weights, (
            "stall_severity not in _signal_weights"
        )

    def test_evaluate_accepts_stall_severity_param(self):
        """Verify evaluate() accepts stall_severity as a parameter."""
        sig = inspect.signature(aeon.MetaCognitiveRecursionTrigger.evaluate)
        assert 'stall_severity' in sig.parameters, (
            "stall_severity not in evaluate() signature"
        )

    def test_evaluate_uses_stall_severity_in_signal_values(self):
        """Verify stall_severity contributes to the signal_values dict."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger.evaluate)
        assert '"stall_severity"' in src, (
            "stall_severity not computed in evaluate() signal_values"
        )

    def test_cached_stall_severity_initialized(self):
        """Verify _cached_stall_severity is initialized in AEONDeltaV3."""
        src = inspect.getsource(aeon.AEONDeltaV3.__init__)
        assert "_cached_stall_severity" in src, (
            "_cached_stall_severity not initialized in __init__"
        )

    def test_stall_severity_count_in_reasoning_core(self):
        """Verify stall_severity appears in at least 3 evaluate() calls
        within _reasoning_core_impl."""
        src = _get_reasoning_core_source()
        count = src.count("stall_severity=self._cached_stall_severity")
        assert count >= 3, (
            f"stall_severity passed in only {count} evaluate() calls "
            f"in _reasoning_core_impl, expected >= 3"
        )

    def test_stall_severity_count_in_forward_impl(self):
        """Verify stall_severity appears in at least 4 evaluate() calls
        within _forward_impl."""
        src = _get_forward_impl_source()
        count = src.count("stall_severity=self._cached_stall_severity")
        assert count >= 4, (
            f"stall_severity passed in only {count} evaluate() calls "
            f"in _forward_impl, expected >= 4"
        )


# ═══════════════════════════════════════════════════════════════════════
# Functional: MetaCognitiveRecursionTrigger.evaluate() with stall_severity
# ═══════════════════════════════════════════════════════════════════════

class TestStallSeverityFunctional:
    """Verify that stall_severity actually contributes to trigger score."""

    def test_stall_severity_increases_trigger_score(self):
        """When stall_severity is high, trigger_score should increase."""
        trigger = aeon.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
            tightening_factor=0.8,
            extra_iterations=2,
        )
        result_no_stall = trigger.evaluate(
            uncertainty=0.3,
            stall_severity=0.0,
        )
        # Reset recursion count for second evaluation
        trigger._recursion_count = 0
        result_with_stall = trigger.evaluate(
            uncertainty=0.3,
            stall_severity=0.9,
        )
        assert result_with_stall['trigger_score'] > result_no_stall['trigger_score'], (
            "High stall_severity should increase trigger_score"
        )

    def test_stall_severity_activates_trigger(self):
        """When only stall_severity is high, it should appear in
        triggers_active."""
        trigger = aeon.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
            tightening_factor=0.8,
            extra_iterations=2,
        )
        result = trigger.evaluate(stall_severity=0.95)
        assert "stall_severity" in result.get('triggers_active', []), (
            "High stall_severity should be in triggers_active"
        )
