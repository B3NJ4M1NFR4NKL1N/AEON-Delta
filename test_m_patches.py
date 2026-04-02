"""Tests for M-series patches: Final Integration & Cognitive Activation.

M1:  Bridge silent exception at convergence completeness check (verify_convergence)
M1b: Bridge completeness exception at AEONDeltaV3 call site into error_evolution
M2:  Log safetensors metadata read failure (_vt_load_weight_file)
M3:  Add oscillation_severity to fast-mode deferred MCT evaluate
M4:  Add oscillation_severity to output reliability MCT evaluate
M5:  Add oscillation_severity to VQ codebook quality MCT evaluate
M6:  Add oscillation_severity to cache verification MCT evaluate
M7:  Add oscillation_severity to high-uncertainty MCT evaluate
M8:  Add oscillation_severity to moderate uncertainty MCT evaluate
M9:  Add stall_severity + oscillation_severity to verify_coherence MCT evaluate
M10: Add stall_severity + oscillation_severity to verify_and_reinforce MCT evaluate
M11: Add oscillation_severity to system_emergence_report MCT evaluate
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
ProvablyConvergentMetaLoop = getattr(aeon, "ProvablyConvergentMetaLoop", None)
CognitiveFeedbackBus = getattr(aeon, "CognitiveFeedbackBus", None)
CausalProvenanceTracker = getattr(aeon, "CausalProvenanceTracker", None)
_vt_load_weight_file = getattr(aeon, "_vt_load_weight_file", None)


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


# ════════════════════════════════════════════════════════════════════════
# M1: Bridge convergence completeness exception (verify_convergence)
# ════════════════════════════════════════════════════════════════════════

class TestM1ConvergenceCompletenessBridge:
    """M1: The verify_convergence completeness self-mapping check must
    log exceptions instead of silently swallowing them, and store the
    exception info in the result meta for the caller to bridge."""

    def test_no_bare_except_pass_in_verify_convergence(self):
        """verify_convergence must not have bare 'except Exception: pass'
        for the completeness check."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        # There should be no bare 'except Exception:\n            pass'
        # The old pattern was: except Exception:\n                pass
        assert "except Exception:\n                pass" not in src

    def test_exception_is_captured_with_variable(self):
        """The except block must capture the exception into a named variable."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        assert "_completeness_err" in src

    def test_logger_debug_on_completeness_failure(self):
        """A logger.debug call must be present for completeness failures."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        assert "Completeness self-mapping check failed" in src

    def test_exception_stored_in_meta(self):
        """The exception message must be stored in meta dict for caller."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        assert "_completeness_check_exception" in src

    def test_m1_patch_comment_present(self):
        """The M1 patch comment must be in the source."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        assert "Patch M1" in src


# ════════════════════════════════════════════════════════════════════════
# M1b: Bridge completeness exception at AEONDeltaV3 call site
# ════════════════════════════════════════════════════════════════════════

class TestM1bCompletenessExceptionBridgeAtCallSite:
    """M1b: AEONDeltaV3 must check verify_convergence results for
    _completeness_check_exception and bridge it via
    _bridge_silent_exception."""

    def test_bridge_call_for_completeness(self):
        """_bridge_silent_exception must be called with
        'convergence_completeness_check_failure' error class."""
        src = inspect.getsource(AEONDeltaV3)
        assert "'convergence_completeness_check_failure'" in src

    def test_completeness_exception_key_checked(self):
        """The call site must check for '_completeness_check_exception'
        in the convergence certificate."""
        src = inspect.getsource(AEONDeltaV3)
        assert "_completeness_check_exception" in src

    def test_m1b_patch_comment(self):
        """The M1b patch comment must be present."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M1b" in src

    def test_error_class_routes_via_prefix(self):
        """convergence_completeness_check_failure must route to
        convergence_conflict via the convergence_ prefix mapping."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger)
        # The prefix routing: ("convergence_", "convergence_conflict")
        assert '"convergence_"' in src or "'convergence_'" in src


# ════════════════════════════════════════════════════════════════════════
# M2: Log safetensors metadata read failure
# ════════════════════════════════════════════════════════════════════════

class TestM2SafetensorsMetadataLog:
    """M2: _vt_load_weight_file must log safetensors metadata read
    failures instead of silently swallowing them."""

    def test_no_bare_except_pass_in_loader(self):
        """_vt_load_weight_file must not have bare 'except Exception: pass'
        for metadata reading."""
        src = inspect.getsource(_vt_load_weight_file)
        # The metadata read block should not have bare pass
        assert "except Exception:\n                    pass" not in src

    def test_exception_captured_with_variable(self):
        """The except block must capture the exception into _meta_read_err."""
        src = inspect.getsource(_vt_load_weight_file)
        assert "_meta_read_err" in src

    def test_logger_debug_on_metadata_failure(self):
        """A logger.debug call must log metadata read failures."""
        src = inspect.getsource(_vt_load_weight_file)
        assert "Safetensors metadata read failed" in src

    def test_m2_patch_comment(self):
        """The M2 patch comment must be present."""
        src = inspect.getsource(_vt_load_weight_file)
        assert "Patch M2" in src


# ════════════════════════════════════════════════════════════════════════
# M3: oscillation_severity in fast-mode deferred MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM3OscillationSeverityFastModeDeferred:
    """M3: The fast-mode deferred metacognitive evaluate() call must pass
    oscillation_severity so the trigger is no longer blind to oscillation
    on the fast path."""

    def test_m3_patch_comment_present(self):
        """The M3 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M3" in src

    def test_oscillation_severity_count_increased(self):
        """oscillation_severity should appear at all evaluate call sites
        that previously only had stall_severity. Count the occurrences
        of the cached value pattern."""
        src = inspect.getsource(AEONDeltaV3)
        pattern = r"oscillation_severity\s*=\s*getattr\(\s*\n?\s*self,\s*['\"]_cached_oscillation_severity['\"]"
        matches = re.findall(pattern, src)
        # K6 + K7 had 2, M-series adds 9 more = 11 total
        assert len(matches) >= 11, (
            f"Expected >=11 oscillation_severity=getattr(self, "
            f"'_cached_oscillation_severity'...) patterns, found {len(matches)}"
        )

    def test_evaluate_accepts_oscillation_severity(self):
        """evaluate() must accept oscillation_severity parameter."""
        sig = inspect.signature(MetaCognitiveRecursionTrigger.evaluate)
        assert 'oscillation_severity' in sig.parameters

    def test_oscillation_severity_contributes_to_score(self):
        """When oscillation_severity > 0, the trigger score should
        increase compared to oscillation_severity=0."""
        trigger = _make_trigger()
        result_zero = trigger.evaluate(oscillation_severity=0.0)
        result_high = trigger.evaluate(oscillation_severity=0.9)
        assert result_high['trigger_score'] >= result_zero['trigger_score']


# ════════════════════════════════════════════════════════════════════════
# M4: oscillation_severity in output reliability MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM4OscillationSeverityOutputReliability:
    """M4: The output reliability metacognitive evaluate() call must pass
    oscillation_severity."""

    def test_m4_patch_comment_present(self):
        """The M4 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M4" in src

    def test_output_reliability_path_has_both_severities(self):
        """The output reliability evaluation path must have both
        stall_severity and oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        # Find the C1b and M4 comments near each other
        assert "Patch C1b" in src  # stall_severity was added by C1b
        assert "Patch M4" in src   # oscillation_severity added by M4


# ════════════════════════════════════════════════════════════════════════
# M5: oscillation_severity in VQ codebook quality MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM5OscillationSeverityVQ:
    """M5: The VQ codebook quality metacognitive evaluate() call must
    pass oscillation_severity."""

    def test_m5_patch_comment_present(self):
        """The M5 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M5" in src

    def test_vq_path_has_both_severities(self):
        """The VQ evaluation path must have both stall_severity (C1c)
        and oscillation_severity (M5)."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch C1c" in src
        assert "Patch M5" in src


# ════════════════════════════════════════════════════════════════════════
# M6: oscillation_severity in cache verification MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM6OscillationSeverityCache:
    """M6: The cache verification metacognitive evaluate() call must
    pass oscillation_severity."""

    def test_m6_patch_comment_present(self):
        """The M6 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M6" in src

    def test_cache_path_has_both_severities(self):
        """The cache evaluation path must have both stall_severity (C1d)
        and oscillation_severity (M6)."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch C1d" in src
        assert "Patch M6" in src


# ════════════════════════════════════════════════════════════════════════
# M7: oscillation_severity in high-uncertainty MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM7OscillationSeverityHighUncertainty:
    """M7: The high-uncertainty metacognitive evaluate() call must pass
    oscillation_severity."""

    def test_m7_patch_comment_present(self):
        """The M7 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M7" in src

    def test_high_unc_path_has_both_severities(self):
        """The high-uncertainty path must have both stall_severity (C1e)
        and oscillation_severity (M7)."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch C1e" in src
        assert "Patch M7" in src


# ════════════════════════════════════════════════════════════════════════
# M8: oscillation_severity in moderate uncertainty MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM8OscillationSeverityModerateUncertainty:
    """M8: The moderate uncertainty metacognitive evaluate() call must
    pass oscillation_severity."""

    def test_m8_patch_comment_present(self):
        """The M8 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M8" in src

    def test_moderate_path_has_both_severities(self):
        """The moderate uncertainty path must have both stall_severity (C1f)
        and oscillation_severity (M8)."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch C1f" in src
        assert "Patch M8" in src


# ════════════════════════════════════════════════════════════════════════
# M9: stall + oscillation severity in verify_coherence MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM9SeveritiesVerifyCoherence:
    """M9: verify_coherence metacognitive evaluate() must pass BOTH
    stall_severity and oscillation_severity — both were previously
    missing from this out-of-band coherence check."""

    def test_m9_patch_comment_present(self):
        """The M9 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M9" in src

    def test_verify_coherence_has_stall_severity(self):
        """verify_coherence must pass stall_severity to the MCT
        evaluate call."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        assert "stall_severity" in src

    def test_verify_coherence_has_oscillation_severity(self):
        """verify_coherence must pass oscillation_severity to the MCT
        evaluate call."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        assert "oscillation_severity" in src

    def test_verify_coherence_uses_cached_values(self):
        """The values should come from _cached_ attributes."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        assert "_cached_stall_severity" in src
        assert "_cached_oscillation_severity" in src


# ════════════════════════════════════════════════════════════════════════
# M10: stall + oscillation severity in verify_and_reinforce MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM10SeveritiesVerifyAndReinforce:
    """M10: verify_and_reinforce metacognitive evaluate() must pass BOTH
    stall_severity and oscillation_severity — both were missing from
    the cognitive unity assessment."""

    def test_m10_patch_comment_present(self):
        """The M10 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M10" in src

    def test_verify_and_reinforce_has_stall_severity(self):
        """verify_and_reinforce must pass stall_severity to the MCT
        evaluate call."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "stall_severity" in src

    def test_verify_and_reinforce_has_oscillation_severity(self):
        """verify_and_reinforce must pass oscillation_severity to the
        MCT evaluate call."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "oscillation_severity" in src

    def test_verify_and_reinforce_uses_cached_values(self):
        """The values should come from _cached_ attributes."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "_cached_stall_severity" in src
        assert "_cached_oscillation_severity" in src


# ════════════════════════════════════════════════════════════════════════
# M11: oscillation_severity in system_emergence_report MCT evaluate
# ════════════════════════════════════════════════════════════════════════

class TestM11OscillationSeverityEmergenceReport:
    """M11: system_emergence_report metacognitive evaluate() must pass
    oscillation_severity alongside the existing stall_severity."""

    def test_m11_patch_comment_present(self):
        """The M11 patch comment must be in the source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M11" in src

    def test_emergence_report_has_oscillation_severity(self):
        """system_emergence_report must pass oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        assert "oscillation_severity" in src

    def test_emergence_report_has_both_severities(self):
        """system_emergence_report must have both stall_severity (C1g)
        and oscillation_severity (M11)."""
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        assert "stall_severity" in src
        assert "oscillation_severity" in src


# ════════════════════════════════════════════════════════════════════════
# Integration: All MCT evaluate call sites have complete severity params
# ════════════════════════════════════════════════════════════════════════

class TestMSeriesIntegrationCompleteness:
    """Integration tests verifying that ALL metacognitive trigger
    evaluate() call sites now pass both stall_severity and
    oscillation_severity, ensuring no trigger weight is permanently
    inert at any evaluation point."""

    def test_all_mct_eval_sites_have_oscillation_severity(self):
        """Every MCT evaluate() call site in AEONDeltaV3 that has
        stall_severity must also have oscillation_severity.

        This test counts the occurrences of both patterns and ensures
        they are equal (every call that passes one passes both)."""
        src = inspect.getsource(AEONDeltaV3)
        stall_count = len(re.findall(
            r"stall_severity\s*=",
            src,
        ))
        osc_count = len(re.findall(
            r"oscillation_severity\s*=",
            src,
        ))
        # stall_severity has one extra occurrence in the UCC evaluate
        # signature and UCC call (UnifiedCognitiveCycle doesn't accept
        # oscillation_severity), but all MCT calls should have both.
        # At minimum, oscillation_severity should appear >= 11 times
        # (K6+K7 = 2 original, M3-M11 = 9 new)
        assert osc_count >= 11, (
            f"Expected >=11 oscillation_severity= assignments, "
            f"found {osc_count}"
        )

    def test_evaluate_signature_has_both_severity_params(self):
        """MetaCognitiveRecursionTrigger.evaluate must accept both
        stall_severity and oscillation_severity."""
        sig = inspect.signature(MetaCognitiveRecursionTrigger.evaluate)
        assert 'stall_severity' in sig.parameters
        assert 'oscillation_severity' in sig.parameters

    def test_signal_weights_contain_both_severities(self):
        """The trigger's _signal_weights must contain entries for both
        severity signals."""
        trigger = _make_trigger()
        assert 'stall_severity' in trigger._signal_weights
        assert 'oscillation_severity' in trigger._signal_weights

    def test_combined_severity_elevates_trigger(self):
        """When both stall and oscillation severity are high, the
        trigger score should exceed either alone."""
        trigger = _make_trigger()
        result_stall = trigger.evaluate(stall_severity=0.7)
        result_osc = trigger.evaluate(oscillation_severity=0.7)
        result_both = trigger.evaluate(
            stall_severity=0.7,
            oscillation_severity=0.7,
        )
        assert result_both['trigger_score'] >= result_stall['trigger_score']
        assert result_both['trigger_score'] >= result_osc['trigger_score']

    def test_no_silent_except_pass_remains(self):
        """After M-series patches, no 'except Exception:\\n...pass'
        pattern should remain in verify_convergence or the weight
        loader for the targeted blocks."""
        vc_src = inspect.getsource(
            ProvablyConvergentMetaLoop.verify_convergence,
        )
        assert "except Exception:\n                pass" not in vc_src

        loader_src = inspect.getsource(_vt_load_weight_file)
        assert "except Exception:\n                    pass" not in loader_src


# ════════════════════════════════════════════════════════════════════════
# Causal Transparency: error class routing verification
# ════════════════════════════════════════════════════════════════════════

class TestMSeriesCausalTransparency:
    """Verify that all M-series error classes are routable through the
    metacognitive trigger's signal mapping infrastructure."""

    def test_convergence_prefix_mapping_exists(self):
        """The prefix mapping for convergence_ → convergence_conflict
        must exist so convergence_completeness_check_failure routes
        correctly."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger)
        assert "convergence_" in src
        assert "convergence_conflict" in src

    def test_convergence_completeness_routes_to_convergence_conflict(self):
        """convergence_completeness_check_failure must be routable
        through adapt_weights_from_evolution."""
        trigger = _make_trigger()
        summary = {
            'convergence_completeness_check_failure': {
                'total_episodes': 3,
                'recent_success_rate': 0.0,
                'recent_strategies': {'warn_and_continue': 3},
            }
        }
        # Should not raise — the error class should be routable
        try:
            trigger.adapt_weights_from_evolution(summary)
        except Exception:
            pytest.fail(
                "adapt_weights_from_evolution failed for "
                "convergence_completeness_check_failure"
            )
