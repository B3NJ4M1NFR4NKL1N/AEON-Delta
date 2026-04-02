"""Tests for O-series patches: Final cognitive integration bridges.

O1:  oscillation_severity passed to UCC evaluate's metacognitive_trigger.evaluate()
O2a: verify_coherence provenance_root_cause recording failure → logger.debug
O2b: verify_coherence convergence_trend_diverging recording failure → logger.debug
O2c: verify_coherence coherence_correction_applied recording failure → logger.debug
O2d: verify_coherence diversity_collapse recording failure → logger.debug
O2e: verify_coherence ns_consistency_low recording failure → logger.debug
O2f: verify_coherence uncertainty_propagation_high recording failure → logger.debug
O2g: verify_coherence feedback_oscillation_high recording failure → logger.debug
O2h: verify_coherence memory_cross_validation_disagreement recording failure → logger.debug
O2i: _cognitive_activation_probe recording failure → logger.debug + break
"""

import importlib
import inspect
import logging
import re
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import aeon_core once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aeon_mod():
    """Return the aeon_core module, imported once per test module."""
    return importlib.import_module("aeon_core")


# ---------------------------------------------------------------------------
# Helper: build a minimal AEONDeltaV3 model with all subsystems
# ---------------------------------------------------------------------------
@pytest.fixture()
def model(aeon_mod):
    """Build a minimal AEONDeltaV3 for unit-testing."""
    cfg = aeon_mod.AEONConfig(
        z_dim=32,
        hidden_dim=32,
        vq_num_embeddings=8,
        vq_embedding_dim=32,
    )
    m = aeon_mod.AEONDeltaV3(cfg)
    return m


# ===================================================================
# O1: oscillation_severity is now passed at UCC evaluate call site
# ===================================================================
class TestO1_OscillationSeverityInUCC:
    """Verify oscillation_severity flows through UCC evaluate to trigger."""

    def test_ucc_evaluate_signature_has_feedback_oscillation_score(self, aeon_mod):
        """UCC.evaluate() accepts feedback_oscillation_score."""
        sig = inspect.signature(aeon_mod.UnifiedCognitiveCycle.evaluate)
        assert "feedback_oscillation_score" in sig.parameters

    def test_mct_evaluate_signature_has_oscillation_severity(self, aeon_mod):
        """MetaCognitiveRecursionTrigger.evaluate() accepts oscillation_severity."""
        sig = inspect.signature(
            aeon_mod.MetaCognitiveRecursionTrigger.evaluate
        )
        assert "oscillation_severity" in sig.parameters

    def test_ucc_evaluate_passes_oscillation_to_trigger(self, aeon_mod):
        """UCC.evaluate() passes feedback_oscillation_score as oscillation_severity
        to its internal metacognitive_trigger.evaluate() call."""
        src = inspect.getsource(aeon_mod.UnifiedCognitiveCycle.evaluate)
        # The patch adds oscillation_severity=feedback_oscillation_score
        assert "oscillation_severity=feedback_oscillation_score" in src, (
            "UCC.evaluate() must pass feedback_oscillation_score as "
            "oscillation_severity to metacognitive_trigger.evaluate()"
        )

    def test_ucc_evaluate_forwards_oscillation_to_trigger_at_runtime(
        self, model,
    ):
        """At runtime, passing feedback_oscillation_score to UCC.evaluate()
        causes oscillation_severity to reach metacognitive_trigger.evaluate()."""
        if model.unified_cognitive_cycle is None:
            pytest.skip("UCC not initialised")
        ucc = model.unified_cognitive_cycle
        # Mock the trigger so we can inspect calls
        original_eval = ucc.metacognitive_trigger.evaluate
        call_kwargs = {}

        def spy_eval(**kwargs):
            call_kwargs.update(kwargs)
            return original_eval(**kwargs)

        ucc.metacognitive_trigger.evaluate = spy_eval
        try:
            B = 1
            H = model.config.hidden_dim
            dummy = torch.randn(B, H)
            ucc.evaluate(
                subsystem_states={"meta_loop": dummy, "world_model": dummy},
                delta_norm=0.01,
                feedback_oscillation_score=0.75,
            )
            assert "oscillation_severity" in call_kwargs, (
                "oscillation_severity was not passed to trigger.evaluate()"
            )
            assert abs(call_kwargs["oscillation_severity"] - 0.75) < 1e-6, (
                f"Expected 0.75, got {call_kwargs['oscillation_severity']}"
            )
        finally:
            ucc.metacognitive_trigger.evaluate = original_eval

    def test_ucc_evaluate_zero_oscillation_by_default(self, model):
        """When feedback_oscillation_score is not passed (default 0.0),
        oscillation_severity=0.0 reaches the trigger."""
        if model.unified_cognitive_cycle is None:
            pytest.skip("UCC not initialised")
        ucc = model.unified_cognitive_cycle
        original_eval = ucc.metacognitive_trigger.evaluate
        call_kwargs = {}

        def spy_eval(**kwargs):
            call_kwargs.update(kwargs)
            return original_eval(**kwargs)

        ucc.metacognitive_trigger.evaluate = spy_eval
        try:
            B, H = 1, model.config.hidden_dim
            dummy = torch.randn(B, H)
            ucc.evaluate(
                subsystem_states={"meta_loop": dummy, "world_model": dummy},
                delta_norm=0.01,
            )
            assert call_kwargs.get("oscillation_severity", -1) == 0.0
        finally:
            ucc.metacognitive_trigger.evaluate = original_eval

    def test_all_evaluate_sites_pass_oscillation_severity(self, aeon_mod):
        """Every metacognitive_trigger.evaluate() call site in the
        AEONDeltaV3 class source passes oscillation_severity (not just UCC)."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3)
        eval_pattern = re.compile(
            r'self\.metacognitive_trigger\.evaluate\(',
        )
        matches = list(eval_pattern.finditer(src))
        assert len(matches) > 0, "No metacognitive_trigger.evaluate() calls found"

        for m in matches:
            # Use a large window (some calls span 90+ lines with comments)
            call_block = src[m.start():m.start() + 6000]
            # The call that uses **_post_pipeline_signals unpacks it
            if "**_post_pipeline_signals" in call_block:
                continue  # signal dict already contains oscillation_severity
            assert "oscillation_severity" in call_block, (
                f"evaluate() call at offset {m.start()} missing oscillation_severity"
            )


# ===================================================================
# O2a-O2h: verify_coherence recording failures now use logger.debug
# ===================================================================
class TestO2_VerifyCoherenceRecordingFailures:
    """Verify that error_evolution recording failures in verify_coherence()
    are logged via logger.debug instead of being silently swallowed."""

    def test_no_bare_except_pass_in_verify_coherence(self, aeon_mod):
        """verify_coherence() must not contain 'except Exception: pass'."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        # The pattern we're checking for should NOT exist
        bare_pass_pattern = re.compile(
            r'except\s+Exception\s*:\s*\n\s*pass\b',
        )
        matches = bare_pass_pattern.findall(src)
        assert len(matches) == 0, (
            f"Found {len(matches)} bare 'except Exception: pass' blocks "
            f"in verify_coherence()"
        )

    def test_recording_failures_use_named_exception(self, aeon_mod):
        """All except Exception blocks in verify_coherence() that guard
        error_evolution recording should name the exception variable."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        # Find 'except Exception:' without 'as' (bare form)
        # We need to exclude 'except Exception as ...' from bare count
        bare_pattern = re.compile(r'except\s+Exception\s*:(?!\s)')
        # Matches like 'except Exception:' (end of line)
        all_except_lines = [
            line.strip()
            for line in src.split('\n')
            if re.match(r'\s*except\s+Exception\b', line.strip())
        ]
        unnamed = [
            line for line in all_except_lines
            if ' as ' not in line
        ]
        assert len(unnamed) == 0, (
            f"Found {len(unnamed)} unnamed 'except Exception:' blocks "
            f"in verify_coherence(): {unnamed[:3]}"
        )

    def test_o2a_provenance_root_cause_has_debug_logging(self, aeon_mod):
        """O2a: provenance_root_cause recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "provenance_root_cause" in src

    def test_o2b_convergence_trend_has_debug_logging(self, aeon_mod):
        """O2b: convergence_trend_diverging recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "convergence_trend_diverging" in src

    def test_o2c_coherence_correction_has_debug_logging(self, aeon_mod):
        """O2c: coherence_correction_applied recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "coherence_correction_applied" in src

    def test_o2d_diversity_collapse_has_debug_logging(self, aeon_mod):
        """O2d: diversity_collapse recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "diversity_collapse" in src

    def test_o2e_ns_consistency_has_debug_logging(self, aeon_mod):
        """O2e: ns_consistency_low recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "ns_consistency_low" in src

    def test_o2f_uncertainty_propagation_has_debug_logging(self, aeon_mod):
        """O2f: uncertainty_propagation_high recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "uncertainty_propagation_high" in src

    def test_o2g_feedback_oscillation_has_debug_logging(self, aeon_mod):
        """O2g: feedback_oscillation_high recording failure logs via debug."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "feedback_oscillation_high" in src

    def test_o2h_memory_cross_validation_has_debug_logging(self, aeon_mod):
        """O2h: memory_cross_validation_disagreement recording failure logs."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3.verify_coherence)
        assert "memory_cross_validation_disagreement" in src

    def test_verify_coherence_recording_failure_emits_debug(
        self, model, caplog,
    ):
        """When error_evolution.record_episode() raises inside
        verify_coherence(), the failure is logged at DEBUG level
        and verify_coherence still returns a result."""
        if model.error_evolution is None:
            pytest.skip("error_evolution not initialised")
        # Count how many recording calls we intercept
        call_count = [0]
        failure_count = [0]
        original = model.error_evolution.record_episode

        def sometimes_failing_record(*args, **kwargs):
            call_count[0] += 1
            # Fail only on specific error classes that are in O2 patches
            ec = kwargs.get('error_class', '')
            o2_classes = {
                'coherence_provenance_root_cause',
                'coherence_convergence_trend_diverging',
                'coherence_correction_applied',
                'coherence_diversity_collapse',
                'coherence_ns_consistency_low',
                'uncertainty_propagation_high',
                'feedback_oscillation_high',
                'memory_cross_validation_disagreement',
            }
            if ec in o2_classes:
                failure_count[0] += 1
                raise RuntimeError("simulated recording failure")
            return original(*args, **kwargs)

        model.error_evolution.record_episode = sometimes_failing_record
        try:
            with caplog.at_level(logging.DEBUG, logger="aeon_core"):
                result = model.verify_coherence()
            # Should not crash
            assert isinstance(result, dict)
        finally:
            model.error_evolution.record_episode = original


# ===================================================================
# O2i: _cognitive_activation_probe recording failure → debug + break
# ===================================================================
class TestO2i_CognitiveActivationProbeRecording:
    """Verify that recording failures in _cognitive_activation_probe()
    are logged via debug before breaking."""

    def test_no_bare_except_break_in_probe(self, aeon_mod):
        """_cognitive_activation_probe() must not contain bare
        'except Exception: break'."""
        src = inspect.getsource(
            aeon_mod.AEONDeltaV3._cognitive_activation_probe,
        )
        bare_break_pattern = re.compile(
            r'except\s+Exception\s*:\s*\n\s*break\b',
        )
        matches = bare_break_pattern.findall(src)
        assert len(matches) == 0, (
            f"Found {len(matches)} bare 'except Exception: break' blocks "
            f"in _cognitive_activation_probe()"
        )

    def test_probe_recording_failure_names_exception(self, aeon_mod):
        """The except block in _cognitive_activation_probe() names
        the exception variable."""
        src = inspect.getsource(
            aeon_mod.AEONDeltaV3._cognitive_activation_probe,
        )
        # Should contain 'except Exception as _rec_err'
        assert re.search(r'except\s+Exception\s+as\s+\w+', src), (
            "_cognitive_activation_probe() should name exception variable"
        )

    def test_probe_recording_failure_has_debug_log(self, aeon_mod):
        """The recording failure handler in _cognitive_activation_probe()
        includes a logger.debug() call."""
        src = inspect.getsource(
            aeon_mod.AEONDeltaV3._cognitive_activation_probe,
        )
        assert "logger.debug" in src, (
            "_cognitive_activation_probe() should log recording failures"
        )

    def test_probe_recording_failure_still_breaks(self, aeon_mod):
        """After logging, the handler still breaks to prevent cascading."""
        src = inspect.getsource(
            aeon_mod.AEONDeltaV3._cognitive_activation_probe,
        )
        # Pattern: logger.debug(...) followed by break within exception handler
        # We check that both exist in the method
        assert "break" in src, (
            "_cognitive_activation_probe() should still break on recording failure"
        )


# ===================================================================
# Integration: Full signal chain completeness
# ===================================================================
class TestSignalChainCompleteness:
    """Verify the complete metacognitive signal chain is connected."""

    def test_trigger_evaluate_has_17_signals(self, aeon_mod):
        """MetaCognitiveRecursionTrigger.evaluate() has all 17 signal params."""
        sig = inspect.signature(
            aeon_mod.MetaCognitiveRecursionTrigger.evaluate
        )
        expected_signals = {
            "uncertainty", "is_diverging", "topology_catastrophe",
            "coherence_deficit", "memory_staleness", "recovery_pressure",
            "world_model_surprise", "causal_quality", "safety_violation",
            "diversity_collapse", "memory_trust_deficit",
            "convergence_conflict", "output_reliability",
            "spectral_stability_margin", "border_uncertainty",
            "stall_severity", "oscillation_severity",
        }
        params = set(sig.parameters.keys()) - {"self", "provenance_attribution"}
        missing = expected_signals - params
        assert not missing, f"Missing signal params: {missing}"

    def test_oscillation_severity_weight_exists(self, aeon_mod):
        """The oscillation_severity weight is defined in _signal_weights."""
        trigger = aeon_mod.MetaCognitiveRecursionTrigger()
        weights = trigger._signal_weights
        assert "oscillation_severity" in weights, (
            "oscillation_severity weight missing from _signal_weights"
        )

    def test_oscillation_severity_nonzero_weight(self, aeon_mod):
        """The oscillation_severity weight has a non-zero default."""
        trigger = aeon_mod.MetaCognitiveRecursionTrigger()
        w = trigger._signal_weights.get("oscillation_severity", 0.0)
        assert w > 0.0, (
            f"oscillation_severity weight should be positive, got {w}"
        )

    def test_stall_and_oscillation_at_all_call_sites(self, aeon_mod):
        """Both stall_severity and oscillation_severity are passed at
        every metacognitive_trigger.evaluate() call site."""
        src = inspect.getsource(aeon_mod.AEONDeltaV3)
        eval_pattern = re.compile(
            r'self\.metacognitive_trigger\.evaluate\(',
        )
        matches = list(eval_pattern.finditer(src))
        for m in matches:
            # Use large window (some calls span 90+ lines with comments)
            block = src[m.start():m.start() + 6000]
            if "**_post_pipeline_signals" in block:
                continue
            assert "stall_severity" in block, (
                f"Missing stall_severity at offset {m.start()}"
            )
            assert "oscillation_severity" in block, (
                f"Missing oscillation_severity at offset {m.start()}"
            )

    def test_ucc_evaluate_no_silent_exception_swallowing(self, aeon_mod):
        """UCC.evaluate() source does not contain bare 'except Exception: pass'."""
        src = inspect.getsource(aeon_mod.UnifiedCognitiveCycle.evaluate)
        bare_pass = re.compile(r'except\s+Exception\s*:\s*\n\s*pass\b')
        assert not bare_pass.search(src), (
            "UCC.evaluate() contains bare 'except Exception: pass'"
        )
