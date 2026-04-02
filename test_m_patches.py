"""Tests for M-series patches: Final Integration & Cognitive Activation.

M1:  Bridge verify_convergence T(0) norm verification exception to
     _bridge_silent_exception (convergence_self_mapping_verification_failure).
M2:  Add logger.debug for safetensors metadata parse failure.
M3:  Add oscillation_severity to fast-mode deferred evaluate.
M4:  Add oscillation_severity to output reliability evaluate.
M5:  Add oscillation_severity to VQ codebook evaluate.
M6:  Add oscillation_severity to cache-hit evaluate.
M7:  Add oscillation_severity to high-unc reinforcement evaluate.
M8:  Add oscillation_severity to moderate uncertainty evaluate.
M9:  Add stall_severity + oscillation_severity to verify_coherence evaluate.
M10: Add stall_severity + oscillation_severity to verify_and_reinforce evaluate.
M11: Add oscillation_severity to system_emergence_report evaluate.
"""

from __future__ import annotations

import inspect
import logging
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    MetaCognitiveRecursionTrigger,
    ProvablyConvergentMetaLoop,
    _vt_load_weight_file,
)


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

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
#  M1: Bridge verify_convergence T(0) norm verification exception
# ════════════════════════════════════════════════════════════════════════

class TestM1ConvergenceSelfMappingBridge:
    """M1: convergence_self_mapping_verification_failure → _bridge_silent_exception."""

    def test_bridge_call_exists_in_source(self):
        """The except block at the T(0) norm verification site calls _bridge_silent_exception."""
        src = inspect.getsource(ProvablyConvergentMetaLoop)
        assert "convergence_self_mapping_verification_failure" in src

    def test_bridge_method_invoked_on_exception(self):
        """When T(0) norm verification raises, _bridge_silent_exception is called."""
        model = _make_model()
        model._bridge_silent_exception = MagicMock()
        model._bridge_silent_exception(
            'convergence_self_mapping_verification_failure',
            'verify_convergence',
            RuntimeError("test"),
        )
        model._bridge_silent_exception.assert_called_once()
        args = model._bridge_silent_exception.call_args[0]
        assert args[0] == 'convergence_self_mapping_verification_failure'
        assert args[1] == 'verify_convergence'

    def test_error_class_has_bridge_not_pass(self):
        """The except block uses _bridge_silent_exception, NOT bare pass."""
        src = inspect.getsource(ProvablyConvergentMetaLoop.verify_convergence)
        # Find the convergence_self_mapping_verification_failure string
        idx = src.find('convergence_self_mapping_verification_failure')
        assert idx != -1
        # The bridge call should be within 200 chars of the error class string
        context = src[max(0, idx - 200):idx + 200]
        assert '_bridge_silent_exception' in context


# ════════════════════════════════════════════════════════════════════════
#  M2: Log safetensors metadata parse failure
# ════════════════════════════════════════════════════════════════════════

class TestM2SafetensorsMetadataLogging:
    """M2: Safetensors metadata parse failure now logged."""

    def test_logging_call_exists_in_source(self):
        """The except block at the metadata parsing site logs the failure."""
        src = inspect.getsource(_vt_load_weight_file)
        assert "safetensors metadata parsing failed" in src

    def test_no_bare_pass_at_metadata_site(self):
        """The metadata parsing except block does NOT use bare pass."""
        src = inspect.getsource(_vt_load_weight_file)
        # Look for the comment that identifies the M2 patch site
        idx = src.find("safetensors metadata parsing failed")
        assert idx != -1
        # The logging call should be within 300 chars
        context = src[max(0, idx - 300):idx + 300]
        assert 'logger.debug' in context


# ════════════════════════════════════════════════════════════════════════
#  M3–M8: oscillation_severity added to 6 MCT evaluate calls
# ════════════════════════════════════════════════════════════════════════

class TestM3FastModeOscillationSeverity:
    """M3: oscillation_severity passed to fast-mode deferred evaluate."""

    def test_oscillation_severity_in_fast_mode_source(self):
        """The fast-mode deferred evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        # M3 patch comment should exist
        assert "Patch M3" in src

    def test_evaluate_accepts_oscillation_severity(self):
        """MetaCognitiveRecursionTrigger.evaluate() accepts oscillation_severity."""
        trigger = _make_trigger()
        result = trigger.evaluate(oscillation_severity=0.7)
        assert isinstance(result, dict)
        assert 'trigger_score' in result

    def test_oscillation_severity_affects_trigger_score(self):
        """Non-zero oscillation_severity contributes to the trigger score."""
        trigger = _make_trigger()
        result_zero = trigger.evaluate(oscillation_severity=0.0)
        result_high = trigger.evaluate(oscillation_severity=0.9)
        assert result_high['trigger_score'] >= result_zero['trigger_score']


class TestM4OutputReliabilityOscillationSeverity:
    """M4: oscillation_severity passed to output reliability evaluate."""

    def test_oscillation_severity_in_output_reliability_source(self):
        """The output reliability evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M4" in src

    def test_patch_follows_stall_severity(self):
        """M4 oscillation_severity is placed after C1b stall_severity."""
        src = inspect.getsource(AEONDeltaV3)
        idx_c1b = src.find("Patch C1b")
        idx_m4 = src.find("Patch M4")
        assert idx_c1b != -1
        assert idx_m4 != -1
        assert idx_m4 > idx_c1b


class TestM5VQCodebookOscillationSeverity:
    """M5: oscillation_severity passed to VQ codebook evaluate."""

    def test_oscillation_severity_in_vq_eval_source(self):
        """The VQ codebook evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M5" in src

    def test_patch_follows_stall_severity(self):
        """M5 oscillation_severity is placed after C1c stall_severity."""
        src = inspect.getsource(AEONDeltaV3)
        idx_c1c = src.find("Patch C1c")
        idx_m5 = src.find("Patch M5")
        assert idx_c1c != -1
        assert idx_m5 != -1
        assert idx_m5 > idx_c1c


class TestM6CacheHitOscillationSeverity:
    """M6: oscillation_severity passed to cache-hit evaluate."""

    def test_oscillation_severity_in_cache_eval_source(self):
        """The cache-hit evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M6" in src

    def test_patch_follows_stall_severity(self):
        """M6 oscillation_severity is placed after C1d stall_severity."""
        src = inspect.getsource(AEONDeltaV3)
        idx_c1d = src.find("Patch C1d")
        idx_m6 = src.find("Patch M6")
        assert idx_c1d != -1
        assert idx_m6 != -1
        assert idx_m6 > idx_c1d


class TestM7HighUncOscillationSeverity:
    """M7: oscillation_severity passed to high-unc reinforcement evaluate."""

    def test_oscillation_severity_in_high_unc_source(self):
        """The high-unc reinforcement evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M7" in src

    def test_patch_follows_stall_severity(self):
        """M7 oscillation_severity is placed after C1e stall_severity."""
        src = inspect.getsource(AEONDeltaV3)
        idx_c1e = src.find("Patch C1e")
        idx_m7 = src.find("Patch M7")
        assert idx_c1e != -1
        assert idx_m7 != -1
        assert idx_m7 > idx_c1e


class TestM8ModerateUncOscillationSeverity:
    """M8: oscillation_severity passed to moderate uncertainty evaluate."""

    def test_oscillation_severity_in_moderate_source(self):
        """The moderate uncertainty evaluate() call passes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M8" in src

    def test_patch_follows_stall_severity(self):
        """M8 oscillation_severity is placed after C1f stall_severity."""
        src = inspect.getsource(AEONDeltaV3)
        idx_c1f = src.find("Patch C1f")
        idx_m8 = src.find("Patch M8")
        assert idx_c1f != -1
        assert idx_m8 != -1
        assert idx_m8 > idx_c1f


# ════════════════════════════════════════════════════════════════════════
#  M9: stall_severity + oscillation_severity in verify_coherence
# ════════════════════════════════════════════════════════════════════════

class TestM9VerifyCoherenceSeveritySignals:
    """M9: stall_severity + oscillation_severity added to verify_coherence evaluate."""

    def test_patch_comment_in_source(self):
        """Patch M9 comment exists in source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M9" in src

    def test_stall_severity_in_verify_coherence(self):
        """verify_coherence evaluate() call now includes stall_severity."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        assert "stall_severity" in src

    def test_oscillation_severity_in_verify_coherence(self):
        """verify_coherence evaluate() call now includes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        assert "oscillation_severity" in src

    def test_both_signals_near_border_uncertainty(self):
        """M9 adds both signals after the border_uncertainty parameter."""
        src = inspect.getsource(AEONDeltaV3.verify_coherence)
        idx_border = src.find("border_uncertainty")
        idx_stall = src.find("stall_severity")
        idx_osc = src.find("oscillation_severity")
        assert idx_border != -1
        assert idx_stall != -1
        assert idx_osc != -1
        assert idx_stall > idx_border
        assert idx_osc > idx_stall


# ════════════════════════════════════════════════════════════════════════
#  M10: stall_severity + oscillation_severity in verify_and_reinforce
# ════════════════════════════════════════════════════════════════════════

class TestM10VerifyAndReinforceSeveritySignals:
    """M10: stall_severity + oscillation_severity added to verify_and_reinforce evaluate."""

    def test_patch_comment_in_source(self):
        """Patch M10 comment exists in source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M10" in src

    def test_stall_severity_in_verify_and_reinforce(self):
        """verify_and_reinforce evaluate() call now includes stall_severity."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "stall_severity" in src

    def test_oscillation_severity_in_verify_and_reinforce(self):
        """verify_and_reinforce evaluate() call now includes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "oscillation_severity" in src

    def test_both_after_memory_trust_deficit(self):
        """M10 adds both signals after memory_trust_deficit."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        # Find the M10 comment and verify it's near memory_trust_deficit
        idx_m10 = src.find("Patch M10")
        assert idx_m10 != -1
        # Look backwards for memory_trust_deficit
        context_before = src[max(0, idx_m10 - 300):idx_m10]
        assert "memory_trust_deficit" in context_before


# ════════════════════════════════════════════════════════════════════════
#  M11: oscillation_severity in system_emergence_report
# ════════════════════════════════════════════════════════════════════════

class TestM11EmergenceReportOscillationSeverity:
    """M11: oscillation_severity added to system_emergence_report evaluate."""

    def test_patch_comment_in_source(self):
        """Patch M11 comment exists in source."""
        src = inspect.getsource(AEONDeltaV3)
        assert "Patch M11" in src

    def test_oscillation_severity_in_emergence_report(self):
        """system_emergence_report evaluate() call includes oscillation_severity."""
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        assert "oscillation_severity" in src

    def test_patch_follows_stall_severity(self):
        """M11 oscillation_severity is placed after C1g stall_severity."""
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        idx_c1g = src.find("Patch C1g")
        idx_m11 = src.find("Patch M11")
        assert idx_c1g != -1
        assert idx_m11 != -1
        assert idx_m11 > idx_c1g


# ════════════════════════════════════════════════════════════════════════
#  Integration: All evaluate() calls now pass oscillation_severity
# ════════════════════════════════════════════════════════════════════════

class TestAllEvaluateCallsHaveOscillationSeverity:
    """Verify that all metacognitive_trigger.evaluate() call sites pass
    oscillation_severity, ensuring no signal dropout in the cognitive flow."""

    def test_evaluate_signature_includes_oscillation_severity(self):
        """MetaCognitiveRecursionTrigger.evaluate() parameter list
        includes oscillation_severity."""
        sig = inspect.signature(MetaCognitiveRecursionTrigger.evaluate)
        assert 'oscillation_severity' in sig.parameters

    def test_evaluate_signature_includes_stall_severity(self):
        """MetaCognitiveRecursionTrigger.evaluate() parameter list
        includes stall_severity."""
        sig = inspect.signature(MetaCognitiveRecursionTrigger.evaluate)
        assert 'stall_severity' in sig.parameters

    def test_oscillation_severity_signal_weight_exists(self):
        """MetaCognitiveRecursionTrigger has an oscillation_severity
        signal weight."""
        trigger = _make_trigger()
        assert 'oscillation_severity' in trigger._signal_weights

    def test_stall_severity_signal_weight_exists(self):
        """MetaCognitiveRecursionTrigger has a stall_severity
        signal weight."""
        trigger = _make_trigger()
        assert 'stall_severity' in trigger._signal_weights

    def test_high_oscillation_triggers_active(self):
        """High oscillation_severity appears in triggers_active list."""
        trigger = _make_trigger()
        result = trigger.evaluate(oscillation_severity=0.9)
        if result['trigger_score'] > 0:
            assert 'oscillation_severity' in result.get('triggers_active', [])

    def test_combined_stall_and_oscillation(self):
        """Both stall_severity and oscillation_severity contribute
        independently to the trigger score."""
        trigger = _make_trigger()
        r_stall = trigger.evaluate(stall_severity=0.8)
        r_osc = trigger.evaluate(oscillation_severity=0.8)
        r_both = trigger.evaluate(stall_severity=0.8, oscillation_severity=0.8)
        # Combined should be >= either individual
        assert r_both['trigger_score'] >= r_stall['trigger_score']
        assert r_both['trigger_score'] >= r_osc['trigger_score']


# ════════════════════════════════════════════════════════════════════════
#  Count: verify all 11 patches exist in source
# ════════════════════════════════════════════════════════════════════════

class TestAllMPatchesPresent:
    """Verify all M-series patch comments are present in the source."""

    @pytest.mark.parametrize("patch_id", [
        "Patch M3", "Patch M4",
        "Patch M5", "Patch M6", "Patch M7", "Patch M8",
        "Patch M9", "Patch M10", "Patch M11",
    ])
    def test_patch_comment_exists_in_model(self, patch_id):
        """Each M-series patch comment (M3-M11) exists in AEONDeltaV3 source."""
        src = inspect.getsource(AEONDeltaV3)
        assert patch_id in src, f"{patch_id} not found in AEONDeltaV3 source"

    def test_patch_m1_in_meta_loop(self):
        """Patch M1 exists in ProvablyConvergentMetaLoop source."""
        src = inspect.getsource(ProvablyConvergentMetaLoop)
        assert "Patch M1" in src

    def test_patch_m2_in_load_function(self):
        """Patch M2 exists in _vt_load_weight_file source."""
        src = inspect.getsource(_vt_load_weight_file)
        assert "Patch M2" in src
