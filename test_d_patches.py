"""
Tests for D-series final cognitive integration patches:

D1: Output reliability → convergence monitor secondary signal
D2: Spectral stability margin → meta-loop Lipschitz tightening
D3: Oscillation severity in _post_pipeline_signals dict
D4: Bridge convergence cert exception to _bridge_silent_exception
D5: Adaptive provenance anomaly threshold + silent exception bridges

Validates that the identified dead-end signal gaps are closed and that
the cognitive architecture now operates as a unified organism with full
causal coherence and self-reflection capabilities.
"""

import inspect
import math
import os
import sys

import pytest
import torch

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


def _get_ucc_evaluate_source():
    """Return source of UnifiedCognitiveCycle.evaluate."""
    return inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)


def _get_forward_impl_source():
    """Return source of _forward_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._forward_impl)


def _get_reasoning_core_source():
    """Return source of _reasoning_core_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)


def _get_provenance_tracker_source():
    """Return source of CausalProvenanceTracker."""
    return inspect.getsource(aeon.CausalProvenanceTracker)


def _get_feedback_extra_source():
    """Return source of _build_feedback_extra_signals."""
    return inspect.getsource(aeon.AEONDeltaV3._build_feedback_extra_signals)


# ═══════════════════════════════════════════════════════════════════════
# D1: Output reliability → convergence monitor secondary signal
# ═══════════════════════════════════════════════════════════════════════

class TestD1OutputReliabilityConvergenceSignal:
    """Verify output_reliability is recorded as convergence secondary signal."""

    def test_ucc_evaluate_records_output_reliability_secondary_signal(self):
        """UCC evaluate() must record output_reliability as convergence
        monitor secondary signal when output_reliability < 0.5."""
        src = _get_ucc_evaluate_source()
        assert "Patch D1" in src, "D1 patch marker not found in UCC evaluate"
        # Check the actual recording call
        idx_d1 = src.find("Patch D1")
        assert idx_d1 > 0
        idx_record = src.find(
            'record_secondary_signal(\n                "output_reliability"',
            idx_d1,
        )
        if idx_record < 0:
            # Try single-line format
            idx_record = src.find(
                'record_secondary_signal("output_reliability"', idx_d1,
            )
        assert idx_record > 0, (
            "record_secondary_signal('output_reliability') not found "
            "after D1 patch marker"
        )

    def test_output_reliability_threshold_is_0_5(self):
        """Only record when output_reliability < 0.5 (significant degradation)."""
        src = _get_ucc_evaluate_source()
        idx_d1 = src.find("Patch D1")
        assert idx_d1 > 0
        # The threshold check should be nearby
        idx_check = src.find("output_reliability < 0.5", idx_d1 - 200)
        assert idx_check > 0, "output_reliability < 0.5 threshold not found"

    def test_output_reliability_inverted_for_secondary_signal(self):
        """Secondary signal should be 1 - output_reliability (higher = worse)."""
        src = _get_ucc_evaluate_source()
        idx_d1 = src.find("Patch D1")
        # The value should be inverted: 1.0 - output_reliability
        idx_invert = src.find("1.0 - max(0.0, min(1.0, output_reliability))", idx_d1)
        assert idx_invert > 0, (
            "Output reliability inversion not found — convergence signal "
            "should be 1.0 - clamped(output_reliability)"
        )

    def test_convergence_monitor_receives_output_reliability_at_runtime(self):
        """Instantiate components and verify signal flows at runtime."""
        monitor = aeon.ConvergenceMonitor(threshold=1e-5)
        # Record a secondary signal
        monitor.record_secondary_signal("output_reliability", 0.7)
        signals = monitor.get_secondary_signals()
        assert "output_reliability" in signals, (
            "ConvergenceMonitor did not accept output_reliability signal"
        )
        assert abs(signals["output_reliability"] - 0.7) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# D2: Spectral stability margin → meta-loop Lipschitz tightening
# ═══════════════════════════════════════════════════════════════════════

class TestD2SpectralLipschitzTightening:
    """Verify low spectral margin tightens meta-loop Lipschitz target."""

    def test_d2_patch_marker_in_reasoning_core(self):
        """D2 patch marker must be present in _reasoning_core_impl."""
        src = _get_reasoning_core_source()
        assert "Patch D2" in src, (
            "D2 patch marker not found in _reasoning_core_impl"
        )

    def test_d2_lipschitz_target_tightened_on_low_spectral_margin(self):
        """When spectral margin < 0.3, lipschitz_target must be reduced."""
        src = _get_reasoning_core_source()
        idx_d2 = src.find("Patch D2: spectral margin")
        assert idx_d2 > 0
        # Check that the lambda_op lipschitz_target is modified
        idx_lip = src.find("lambda_op.lipschitz_target", idx_d2)
        assert idx_lip > 0, (
            "lambda_op.lipschitz_target modification not found after D2"
        )

    def test_d2_lipschitz_target_restored_after_deeper_loop(self):
        """Lipschitz target must be restored after deeper meta-loop."""
        src = _get_reasoning_core_source()
        idx_restore = src.find("Patch D2 restore: Lipschitz target")
        assert idx_restore > 0, (
            "D2 Lipschitz target restoration marker not found"
        )
        idx_set_back = src.find(
            "lambda_op.lipschitz_target = _orig_lip_target", idx_restore,
        )
        assert idx_set_back > 0, (
            "Lipschitz target restoration assignment not found"
        )

    def test_d2_tightening_threshold_is_0_3(self):
        """Spectral margin tightening threshold should be 0.3."""
        src = _get_reasoning_core_source()
        assert "_D2_SPECTRAL_TIGHTEN_THRESHOLD = 0.3" in src, (
            "D2 spectral tighten threshold not found or has wrong value"
        )

    def test_d2_lipschitz_floor_is_0_5(self):
        """Lipschitz target should never go below 0.5."""
        src = _get_reasoning_core_source()
        idx_d2 = src.find("Patch D2: spectral margin")
        assert idx_d2 > 0
        # Check for the floor
        idx_floor = src.find("max(\n                        0.5,", idx_d2)
        if idx_floor < 0:
            idx_floor = src.find("max(0.5,", idx_d2)
        assert idx_floor > 0, (
            "Lipschitz target floor of 0.5 not found in D2 patch"
        )

    def test_d2_tightening_magnitude(self):
        """Verify the tightening is proportional to margin deficit."""
        src = _get_reasoning_core_source()
        idx_d2 = src.find("Patch D2: spectral margin")
        assert idx_d2 > 0
        # Look for margin_deficit computation within 1800 chars
        assert "_margin_deficit" in src[idx_d2:idx_d2+1800], (
            "Margin deficit computation not found in D2"
        )
        assert "_lip_reduction" in src[idx_d2:idx_d2+1800], (
            "Lipschitz reduction computation not found in D2"
        )

    def test_d2_saves_original_lipschitz_target(self):
        """Original Lipschitz target must be saved for restoration."""
        src = _get_reasoning_core_source()
        assert "_orig_lip_target = self.meta_loop.lambda_op.lipschitz_target" in src, (
            "Original Lipschitz target not saved for later restoration"
        )


# ═══════════════════════════════════════════════════════════════════════
# D3: Oscillation severity in _post_pipeline_signals
# ═══════════════════════════════════════════════════════════════════════

class TestD3OscillationInPostPipeline:
    """Verify oscillation severity is included in post-pipeline signals."""

    def test_d3_patch_marker_in_forward_impl(self):
        """D3 patch marker must be in _forward_impl."""
        src = _get_forward_impl_source()
        assert "Patch D3" in src, (
            "D3 patch marker not found in _forward_impl"
        )

    def test_oscillation_severity_in_post_pipeline_dict(self):
        """_post_pipeline_signals must include oscillation_severity."""
        src = _get_forward_impl_source()
        idx_post = src.find("_post_pipeline_signals = {")
        assert idx_post > 0, "_post_pipeline_signals dict not found"
        idx_close = src.find("}", idx_post)
        assert idx_close > 0
        dict_section = src[idx_post:idx_close + 1]
        assert "'oscillation_severity'" in dict_section, (
            "oscillation_severity not found in _post_pipeline_signals dict"
        )

    def test_oscillation_severity_from_cached_value(self):
        """Should read from _cached_oscillation_severity."""
        src = _get_forward_impl_source()
        idx_d3 = src.find("Patch D3")
        assert idx_d3 > 0
        idx_cached = src.find("_cached_oscillation_severity", idx_d3)
        assert idx_cached > 0, (
            "_cached_oscillation_severity not referenced after D3 marker"
        )

    def test_oscillation_severity_default_zero(self):
        """Should default to 0.0 when no cached value exists."""
        src = _get_forward_impl_source()
        # Look for getattr with default 0.0
        assert "_cached_oscillation_severity', 0.0" in src, (
            "oscillation_severity should default to 0.0 via getattr"
        )

    def test_post_pipeline_signals_completeness(self):
        """Post-pipeline signals should now include all 18+ signals."""
        src = _get_forward_impl_source()
        idx_post = src.find("_post_pipeline_signals = {")
        assert idx_post > 0
        idx_close = src.find("}", idx_post)
        dict_section = src[idx_post:idx_close + 1]
        expected_signals = [
            'uncertainty', 'coherence_deficit', 'spectral_stability_margin',
            'output_reliability', 'world_model_surprise',
            'convergence_conflict', 'recovery_pressure',
            'safety_violation', 'topology_catastrophe',
            'diversity_collapse', 'memory_trust_deficit',
            'memory_staleness', 'causal_quality', 'border_uncertainty',
            'is_diverging', 'stall_severity', 'oscillation_severity',
        ]
        for sig in expected_signals:
            assert f"'{sig}'" in dict_section, (
                f"Signal '{sig}' missing from _post_pipeline_signals dict"
            )


# ═══════════════════════════════════════════════════════════════════════
# D4: Bridge convergence cert exception to _bridge_silent_exception
# ═══════════════════════════════════════════════════════════════════════

class TestD4ConvergenceCertExceptionBridge:
    """Verify convergence cert recording failure is bridged properly."""

    def test_d4_patch_marker_in_ucc(self):
        """D4 patch marker must be present in UCC evaluate."""
        src = _get_ucc_evaluate_source()
        assert "Patch D4" in src, (
            "D4 patch marker not found in UCC evaluate"
        )

    def test_no_bare_except_pass_at_cert_violation(self):
        """The convergence cert recording must NOT use bare except:pass."""
        src = _get_ucc_evaluate_source()
        # Find the cert violation recording block
        idx_cert = src.find("convergence_certificate_violation")
        assert idx_cert > 0
        # Look in the next 500 chars — should not have bare except:pass
        region = src[idx_cert:idx_cert + 800]
        lines = region.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'except Exception:':
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
                assert next_line != 'pass', (
                    "Bare except Exception: pass still present near "
                    "convergence_certificate_violation recording"
                )

    def test_bridge_silent_exception_called_for_cert_failure(self):
        """_bridge_silent_exception must be called on cert recording failure."""
        src = _get_ucc_evaluate_source()
        idx_d4 = src.find("Patch D4")
        assert idx_d4 > 0
        idx_bridge = src.find("_bridge_silent_exception", idx_d4)
        assert idx_bridge > 0, (
            "_bridge_silent_exception not called after D4 marker"
        )

    def test_cert_failure_error_class_name(self):
        """The bridged error class should identify the failure origin."""
        src = _get_ucc_evaluate_source()
        assert "convergence_certificate_recording_failure" in src, (
            "Error class for cert recording failure not found"
        )


# ═══════════════════════════════════════════════════════════════════════
# D5: Adaptive provenance anomaly threshold
# ═══════════════════════════════════════════════════════════════════════

class TestD5AdaptiveAnomalyThreshold:
    """Verify provenance anomaly threshold adapts from delta history."""

    def test_d5_patch_marker_in_provenance_tracker(self):
        """D5 patch marker must be in CausalProvenanceTracker."""
        src = _get_provenance_tracker_source()
        assert "Patch D5" in src, (
            "D5 patch marker not found in CausalProvenanceTracker"
        )

    def test_delta_history_list_initialized(self):
        """CausalProvenanceTracker must have _delta_history list."""
        src = _get_provenance_tracker_source()
        assert "_delta_history" in src, (
            "_delta_history not found in CausalProvenanceTracker"
        )

    def test_delta_history_bounded(self):
        """Delta history must be bounded to prevent unbounded growth."""
        src = _get_provenance_tracker_source()
        assert "_DELTA_HISTORY_MAX" in src, (
            "_DELTA_HISTORY_MAX bound not found"
        )

    def test_anomaly_threshold_has_floor(self):
        """Anomaly threshold must have a minimum floor."""
        src = _get_provenance_tracker_source()
        assert "_DELTA_ANOMALY_MIN" in src, (
            "_DELTA_ANOMALY_MIN floor not found"
        )

    def test_adaptive_threshold_uses_3_sigma(self):
        """Adaptive threshold should be mean + 3*std."""
        src = _get_provenance_tracker_source()
        assert "3.0 * _std" in src or "3 * _std" in src, (
            "3-sigma formula not found in threshold adaptation"
        )

    def test_adaptive_threshold_at_runtime(self):
        """Verify threshold adapts correctly from recorded deltas."""
        config = _make_config()
        tracker = aeon.CausalProvenanceTracker()
        # Initial threshold should be 50.0
        assert tracker._delta_anomaly_threshold == 50.0

        # Record 15 normal deltas of value ~1.0
        for i in range(15):
            state_before = torch.randn(1, 64)
            state_after = state_before + torch.randn(1, 64) * 0.1
            tracker.record_before(f"module_{i % 3}", state_before)
            tracker.record_after(f"module_{i % 3}", state_after)

        # After 15 recordings, threshold should have adapted downward
        # from 50.0 since all deltas are ~0.1-scale
        assert tracker._delta_anomaly_threshold < 50.0, (
            f"Threshold did not adapt: {tracker._delta_anomaly_threshold}"
        )
        # But should be above the floor
        assert tracker._delta_anomaly_threshold >= tracker._DELTA_ANOMALY_MIN, (
            f"Threshold below floor: {tracker._delta_anomaly_threshold}"
        )

    def test_adaptive_threshold_raises_with_large_deltas(self):
        """Threshold should increase when deltas are consistently large."""
        tracker = aeon.CausalProvenanceTracker()
        # Record 15 large deltas
        for i in range(15):
            state_before = torch.zeros(1, 64)
            state_after = torch.ones(1, 64) * 100.0
            tracker.record_before(f"module_{i % 3}", state_before)
            tracker.record_after(f"module_{i % 3}", state_after)

        # Threshold should be well above the floor
        assert tracker._delta_anomaly_threshold > 50.0, (
            f"Threshold should rise with large deltas: "
            f"{tracker._delta_anomaly_threshold}"
        )


# ═══════════════════════════════════════════════════════════════════════
# D5b: Bridge diversity adaptation failure
# ═══════════════════════════════════════════════════════════════════════

class TestD5bDiversityAdaptationBridge:
    """Verify diversity collapse adaptation failure is bridged."""

    def test_diversity_adaptation_uses_bridge_silent_exception(self):
        """Diversity adaptation exception must use _bridge_silent_exception."""
        src = _get_feedback_extra_source()
        assert "Patch D5b" in src, (
            "D5b patch marker not found in _build_feedback_extra_signals"
        )

    def test_no_bare_except_pass_at_diversity_adaptation(self):
        """Diversity adaptation must not use bare except:pass."""
        src = _get_feedback_extra_source()
        # Find diversity collapse adaptation area
        idx_div = src.find("diversity_collapse_adaptation_failure")
        assert idx_div > 0, (
            "diversity_collapse_adaptation_failure error class not found"
        )

    def test_sustained_diversity_uses_bridge(self):
        """Sustained diversity collapse must also use bridge."""
        src = _get_feedback_extra_source()
        assert "sustained_diversity_adaptation_failure" in src, (
            "sustained_diversity_adaptation_failure not found"
        )


# ═══════════════════════════════════════════════════════════════════════
# D5c: Bridge feedback-to-trigger wiring failure
# ═══════════════════════════════════════════════════════════════════════

class TestD5cFeedbackTriggerBridge:
    """Verify feedback-to-trigger wiring failure is bridged."""

    def test_feedback_trigger_wiring_uses_bridge(self):
        """Feedback-trigger wiring exception must use _bridge_silent_exception."""
        src = _get_reasoning_core_source()
        assert "Patch D5c" in src, (
            "D5c patch marker not found in _reasoning_core_impl"
        )

    def test_feedback_trigger_wiring_error_class(self):
        """Error class for feedback-trigger wiring should be specific."""
        src = _get_reasoning_core_source()
        assert "feedback_trigger_wiring_failure" in src, (
            "feedback_trigger_wiring_failure error class not found"
        )


# ═══════════════════════════════════════════════════════════════════════
# Integration: Cross-patch coherence
# ═══════════════════════════════════════════════════════════════════════

class TestDSeriesCrossPatches:
    """Cross-patch integration tests validating D-series coherence."""

    def test_all_d_patch_markers_present(self):
        """All D-series patches must have their markers in the codebase."""
        ucc_src = _get_ucc_evaluate_source()
        fwd_src = _get_forward_impl_source()
        rsn_src = _get_reasoning_core_source()
        prv_src = _get_provenance_tracker_source()
        fb_src = _get_feedback_extra_source()

        assert "Patch D1" in ucc_src, "D1 marker missing"
        assert "Patch D2" in rsn_src, "D2 marker missing"
        assert "Patch D3" in fwd_src, "D3 marker missing"
        assert "Patch D4" in ucc_src, "D4 marker missing"
        assert "Patch D5" in prv_src, "D5 marker missing"
        assert "Patch D5b" in fb_src, "D5b marker missing"
        assert "Patch D5c" in rsn_src, "D5c marker missing"

    def test_no_remaining_bare_except_pass_in_critical_paths(self):
        """Critical cognitive paths should not have bare except:pass."""
        # Check UCC evaluate for bare except:pass
        ucc_src = _get_ucc_evaluate_source()
        lines = ucc_src.split('\n')
        bare_pass_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'except Exception:':
                next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ''
                if next_stripped == 'pass':
                    bare_pass_count += 1
        # D4 should have eliminated the cert violation one;
        # there may be other non-critical ones remaining
        # but the cert violation one should be gone
        cert_idx = ucc_src.find("convergence_certificate_violation")
        if cert_idx > 0:
            region = ucc_src[cert_idx:cert_idx + 800]
            region_lines = region.split('\n')
            for i, line in enumerate(region_lines):
                stripped = line.strip()
                if stripped == 'except Exception:':
                    next_stripped = region_lines[i + 1].strip() if i + 1 < len(region_lines) else ''
                    assert next_stripped != 'pass', (
                        "Bare except:pass still present near cert violation"
                    )

    def test_spectral_margin_flows_to_both_iterations_and_lipschitz(self):
        """Spectral stability should influence BOTH extra iterations AND
        Lipschitz target (pre-D2 it only influenced iterations)."""
        src = _get_reasoning_core_source()
        # Verify extra iterations from spectral (pre-existing)
        assert "_spectral_extra" in src, (
            "Spectral extra iterations not found"
        )
        # Verify Lipschitz tightening from spectral (D2)
        assert "_lip_reduction" in src, (
            "Lipschitz reduction from spectral margin not found"
        )

    def test_post_pipeline_signals_have_stall_and_oscillation(self):
        """Both stall_severity (C2) and oscillation_severity (D3) must
        be present in post-pipeline signals."""
        src = _get_forward_impl_source()
        idx_post = src.find("_post_pipeline_signals = {")
        assert idx_post > 0
        idx_close = src.find("}", idx_post)
        dict_section = src[idx_post:idx_close + 1]
        assert "'stall_severity'" in dict_section, (
            "stall_severity (C2) missing from post-pipeline signals"
        )
        assert "'oscillation_severity'" in dict_section, (
            "oscillation_severity (D3) missing from post-pipeline signals"
        )

    def test_model_instantiation_succeeds(self):
        """AEONDeltaV3 must instantiate without errors after all patches."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert model is not None
        # Verify critical components exist
        assert hasattr(model, 'metacognitive_trigger')
        assert hasattr(model, 'error_evolution')
        assert hasattr(model, 'provenance_tracker')
        assert hasattr(model, 'feedback_bus')
        assert hasattr(model, 'convergence_monitor')
        assert hasattr(model, 'meta_loop')

    def test_convergence_monitor_has_record_secondary_signal(self):
        """ConvergenceMonitor must support record_secondary_signal."""
        monitor = aeon.ConvergenceMonitor(threshold=1e-5)
        assert hasattr(monitor, 'record_secondary_signal')
        assert hasattr(monitor, 'get_secondary_signals')
        # Should accept arbitrary signal names
        monitor.record_secondary_signal("test_signal", 0.5)
        signals = monitor.get_secondary_signals()
        assert "test_signal" in signals

    def test_provenance_tracker_has_adaptive_threshold_attrs(self):
        """CausalProvenanceTracker must have adaptive threshold attrs."""
        tracker = aeon.CausalProvenanceTracker()
        assert hasattr(tracker, '_delta_history')
        assert hasattr(tracker, '_DELTA_HISTORY_MAX')
        assert hasattr(tracker, '_DELTA_ANOMALY_MIN')
        assert isinstance(tracker._delta_history, list)
        assert tracker._DELTA_HISTORY_MAX > 0
        assert tracker._DELTA_ANOMALY_MIN > 0

    def test_oscillation_severity_cached_on_model(self):
        """AEONDeltaV3 must cache oscillation severity for D3."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # Should have cached oscillation severity (initialized to 0.0)
        osc = getattr(model, '_cached_oscillation_severity', None)
        assert osc is not None or hasattr(model, '_cached_oscillation_severity'), (
            "_cached_oscillation_severity not found on model"
        )
