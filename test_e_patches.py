"""Tests for E-series cognitive integration patches.

E1: Verified prediction error → feedback bus cross-pass surfacing
E2: Complexity gate corruption flag → feedback bus pressure signal
E3: VibeThinker calibration Phase 1 & Phase 2 → error_evolution bridging
E4: Trace completeness deficit → feedback bus signal
E5: Non-deeper error_evolution strategies → uncertainty escalation
E6: Cross-model prediction divergence detection (WM vs HWM)
"""

import inspect
import os
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import aeon_core as aeon


# ── Config factory ────────────────────────────────────────────────────
def _make_config(**overrides):
    """Create AEONConfig with minimal test defaults."""
    defaults = dict(
        hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
        seq_length=32, dropout_rate=0.0, meta_dim=32,
        lipschitz_target=0.9, vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


# ── Source helpers ────────────────────────────────────────────────────
def _get_reasoning_core_source():
    return inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)


def _get_feedback_extra_source():
    return inspect.getsource(aeon.AEONDeltaV3._build_feedback_extra_signals)


def _get_forward_impl_source():
    return inspect.getsource(aeon.AEONDeltaV3._forward_impl)


def _get_vt_calibration_source():
    return inspect.getsource(
        aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
    )


def _get_init_source():
    return inspect.getsource(aeon.AEONDeltaV3.__init__)


# ══════════════════════════════════════════════════════════════════════
# E1: Verified prediction error → feedback bus cross-pass surfacing
# ══════════════════════════════════════════════════════════════════════

class TestE1VerifiedPredictionErrorCache:
    """E1: _cached_verified_prediction_error must be stored after
    computation and surfaced in _build_feedback_extra_signals."""

    def test_cached_vpe_attribute_exists_in_init(self):
        """__init__ must declare _cached_verified_prediction_error."""
        src = _get_init_source()
        assert "_cached_verified_prediction_error" in src, (
            "_cached_verified_prediction_error not initialized in __init__"
        )

    def test_cached_vpe_stored_in_reasoning_core(self):
        """_reasoning_core_impl must cache the VPE after computation."""
        src = _get_reasoning_core_source()
        assert "self._cached_verified_prediction_error" in src, (
            "VPE not cached in _reasoning_core_impl"
        )
        idx = src.find("_verified_prediction_error")
        assert idx >= 0
        idx_cache = src.find(
            "self._cached_verified_prediction_error", idx,
        )
        assert idx_cache >= 0, (
            "VPE cache write not found after VPE computation"
        )

    def test_vpe_pressure_in_feedback_signals(self):
        """_build_feedback_extra_signals must emit
        verified_prediction_error_pressure."""
        src = _get_feedback_extra_source()
        assert "verified_prediction_error_pressure" in src, (
            "verified_prediction_error_pressure not in feedback signals"
        )

    def test_vpe_registered_in_feedback_bus(self):
        """The signal must be registered in the feedback bus at init."""
        src = _get_init_source()
        assert "verified_prediction_error_pressure" in src, (
            "verified_prediction_error_pressure not registered in __init__"
        )

    def test_model_has_cached_vpe(self):
        """Instantiated model must have _cached_verified_prediction_error."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_cached_verified_prediction_error')
        assert model._cached_verified_prediction_error == 0.0

    def test_feedback_bus_has_vpe_signal(self):
        """Feedback bus must have verified_prediction_error_pressure."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        signals = model.feedback_bus._extra_signals
        assert "verified_prediction_error_pressure" in signals, (
            "Signal not registered in feedback bus"
        )

    def test_vpe_surfaces_when_high(self):
        """When _cached_verified_prediction_error > 0.1, the feedback bus
        signal must be emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_verified_prediction_error = 0.8
        extra = model._build_feedback_extra_signals()
        assert "verified_prediction_error_pressure" in extra, (
            "VPE pressure not emitted when VPE=0.8"
        )
        assert 0.0 < extra["verified_prediction_error_pressure"] <= 1.0

    def test_vpe_not_surfaced_when_low(self):
        """When _cached_verified_prediction_error <= 0.1, no signal."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_verified_prediction_error = 0.05
        extra = model._build_feedback_extra_signals()
        assert "verified_prediction_error_pressure" not in extra

    def test_vpe_reset_per_pass(self):
        """_forward_impl must reset _cached_verified_prediction_error."""
        src = _get_forward_impl_source()
        assert "_cached_verified_prediction_error = 0.0" in src, (
            "VPE not reset in _forward_impl per-pass reset block"
        )


# ══════════════════════════════════════════════════════════════════════
# E2: Complexity gate corruption → feedback bus pressure signal
# ══════════════════════════════════════════════════════════════════════

class TestE2GateCorruptionPressure:
    """E2: Non-finite complexity gates must set a flag that the feedback
    bus surfaces as gate_corruption_pressure."""

    def test_cached_flag_exists_in_init(self):
        """__init__ must declare _cached_gate_corruption_flag."""
        src = _get_init_source()
        assert "_cached_gate_corruption_flag" in src

    def test_flag_set_on_corruption(self):
        """_reasoning_core_impl must set the flag on NaN/Inf gates."""
        src = _get_reasoning_core_source()
        assert "self._cached_gate_corruption_flag = True" in src, (
            "Gate corruption flag not set in _reasoning_core_impl"
        )

    def test_pressure_in_feedback_signals(self):
        """_build_feedback_extra_signals must emit gate_corruption_pressure."""
        src = _get_feedback_extra_source()
        assert "gate_corruption_pressure" in src

    def test_signal_registered(self):
        """gate_corruption_pressure must be registered in feedback bus."""
        src = _get_init_source()
        assert "gate_corruption_pressure" in src

    def test_model_has_flag(self):
        """Instantiated model must have _cached_gate_corruption_flag."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_cached_gate_corruption_flag')
        assert model._cached_gate_corruption_flag is False

    def test_pressure_emitted_when_flag_set(self):
        """When flag is True, feedback bus emits gate_corruption_pressure."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_gate_corruption_flag = True
        extra = model._build_feedback_extra_signals()
        assert "gate_corruption_pressure" in extra
        assert extra["gate_corruption_pressure"] == 1.0

    def test_flag_reset_after_consumption(self):
        """The flag must be reset after the feedback bus consumes it."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_gate_corruption_flag = True
        model._build_feedback_extra_signals()
        assert model._cached_gate_corruption_flag is False, (
            "Gate corruption flag not reset after consumption"
        )

    def test_no_pressure_when_flag_clear(self):
        """When flag is False, no gate_corruption_pressure emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        extra = model._build_feedback_extra_signals()
        assert "gate_corruption_pressure" not in extra

    def test_error_class_bridge_mapping(self):
        """non_finite_complexity_gate → gate_corruption_pressure in bridge map."""
        src = _get_feedback_extra_source()
        assert "non_finite_complexity_gate" in src
        assert "gate_corruption_pressure" in src


# ══════════════════════════════════════════════════════════════════════
# E3: VibeThinker calibration anomaly bridging
# ══════════════════════════════════════════════════════════════════════

class TestE3VibeThinkerCalibrationBridging:
    """E3: VibeThinker calibration Phase 1 & Phase 2 must bridge
    anomalies to error_evolution."""

    def test_phase1_anomaly_detection_exists(self):
        """Phase 1 (adapter warm-up) must check for non-finite embeddings."""
        src = _get_vt_calibration_source()
        assert "_warmup_anomaly" in src, (
            "Phase 1 anomaly detection not implemented"
        )

    def test_phase1_records_error_evolution(self):
        """Phase 1 anomalies must record vibe_thinker_warmup_anomaly."""
        src = _get_vt_calibration_source()
        assert "vibe_thinker_warmup_anomaly" in src, (
            "Phase 1 error_evolution episode not recorded"
        )

    def test_phase2_anomaly_detection_exists(self):
        """Phase 2 (VQ seeding) must check for non-finite seed vectors."""
        src = _get_vt_calibration_source()
        assert "vibe_thinker_vq_seeding_anomaly" in src, (
            "Phase 2 VQ seeding anomaly detection not implemented"
        )

    def test_phase2_checks_seeded_data(self):
        """Phase 2 must validate seeded embeddings after writing."""
        src = _get_vt_calibration_source()
        assert "torch.isfinite(_seeded_data)" in src, (
            "Phase 2 does not validate seeded codebook entries"
        )

    def test_phase1_checks_magnitude(self):
        """Phase 1 must detect extreme-magnitude embeddings (> 100.0)."""
        src = _get_vt_calibration_source()
        assert "_emb.abs().max().item() > 100.0" in src, (
            "Phase 1 magnitude threshold check not present"
        )

    def test_error_class_bridge_mapping_warmup(self):
        """vibe_thinker_warmup_anomaly must be in the error class bridge."""
        src = _get_feedback_extra_source()
        assert "vibe_thinker_warmup_anomaly" in src

    def test_error_class_bridge_mapping_vq(self):
        """vibe_thinker_vq_seeding_anomaly must be in the error class bridge."""
        src = _get_feedback_extra_source()
        assert "vibe_thinker_vq_seeding_anomaly" in src


# ══════════════════════════════════════════════════════════════════════
# E4: Trace completeness deficit → feedback bus signal
# ══════════════════════════════════════════════════════════════════════

class TestE4TraceCompletenessPressure:
    """E4: Provenance trace completeness must be cached and surfaced
    in the feedback bus as trace_completeness_pressure."""

    def test_cached_ratio_exists_in_init(self):
        """__init__ must declare _cached_trace_completeness_ratio."""
        src = _get_init_source()
        assert "_cached_trace_completeness_ratio" in src

    def test_ratio_cached_in_reasoning_core(self):
        """_reasoning_core_impl must cache the trace completeness ratio."""
        src = _get_reasoning_core_source()
        assert "self._cached_trace_completeness_ratio" in src, (
            "Trace completeness ratio not cached"
        )

    def test_pressure_in_feedback_signals(self):
        """_build_feedback_extra_signals must emit trace_completeness_pressure."""
        src = _get_feedback_extra_source()
        assert "trace_completeness_pressure" in src

    def test_signal_registered(self):
        """trace_completeness_pressure must be registered in feedback bus."""
        src = _get_init_source()
        assert "trace_completeness_pressure" in src

    def test_model_has_cached_ratio(self):
        """Instantiated model must have _cached_trace_completeness_ratio."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_cached_trace_completeness_ratio')
        assert model._cached_trace_completeness_ratio == 1.0

    def test_pressure_emitted_when_low(self):
        """When ratio < 0.9, trace_completeness_pressure is emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_trace_completeness_ratio = 0.6
        extra = model._build_feedback_extra_signals()
        assert "trace_completeness_pressure" in extra
        expected = 1.0 - 0.6  # = 0.4
        assert abs(extra["trace_completeness_pressure"] - expected) < 0.01

    def test_no_pressure_when_complete(self):
        """When ratio >= 0.9, no trace_completeness_pressure emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_trace_completeness_ratio = 0.95
        extra = model._build_feedback_extra_signals()
        assert "trace_completeness_pressure" not in extra

    def test_feedback_bus_has_signal(self):
        """Feedback bus must have trace_completeness_pressure registered."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        signals = model.feedback_bus._extra_signals
        assert "trace_completeness_pressure" in signals


# ══════════════════════════════════════════════════════════════════════
# E5: Non-deeper strategies → uncertainty escalation
# ══════════════════════════════════════════════════════════════════════

class TestE5NonDeeperStrategyEscalation:
    """E5: When get_best_strategy returns a non-deeper strategy,
    uncertainty must still be escalated proportionally."""

    def test_elif_branch_exists(self):
        """_reasoning_core_impl must have the elif branch for non-deeper
        strategies that escalates uncertainty."""
        src = _get_reasoning_core_source()
        assert "evolved_strategy_guidance" in src, (
            "Non-deeper strategy uncertainty escalation not implemented"
        )

    def test_audit_record_for_escalation(self):
        """An audit record must be created for proactive uncertainty escalation."""
        src = _get_reasoning_core_source()
        assert "proactive_uncertainty_escalation" in src, (
            "Audit record for non-deeper escalation not found"
        )

    def test_uncertainty_sources_key(self):
        """The uncertainty_sources dict must record the evolved_strategy_guidance
        key when the non-deeper path fires."""
        src = _get_reasoning_core_source()
        idx = src.find("evolved_strategy_guidance")
        assert idx >= 0, "evolved_strategy_guidance not found in source"
        # Check it's in an uncertainty_sources assignment context
        context = src[max(0, idx - 200):idx + 200]
        assert "uncertainty_sources" in context, (
            "evolved_strategy_guidance not tied to uncertainty_sources"
        )

    def test_high_uncertainty_flag_set(self):
        """After escalation, high_uncertainty must be re-evaluated."""
        src = _get_reasoning_core_source()
        idx = src.find("evolved_strategy_guidance")
        assert idx >= 0, "evolved_strategy_guidance not found in source"
        context = src[idx:idx + 300]
        assert "high_uncertainty" in context, (
            "high_uncertainty not re-evaluated after non-deeper escalation"
        )


# ══════════════════════════════════════════════════════════════════════
# E6: Cross-model prediction divergence detection
# ══════════════════════════════════════════════════════════════════════

class TestE6CrossModelDivergence:
    """E6: When WM and HWM both produce verified prediction errors,
    their divergence must be cached and surfaced."""

    def test_cached_divergence_exists_in_init(self):
        """__init__ must declare _cached_cross_model_divergence."""
        src = _get_init_source()
        assert "_cached_cross_model_divergence" in src

    def test_divergence_computed_in_reasoning_core(self):
        """_reasoning_core_impl must compute cross-model divergence."""
        src = _get_reasoning_core_source()
        assert "cross_model_prediction_divergence" in src, (
            "Cross-model divergence not computed"
        )
        assert "self._cached_cross_model_divergence" in src, (
            "Cross-model divergence not cached"
        )

    def test_divergence_threshold_is_0_3(self):
        """Only record error_evolution when divergence > 0.3."""
        src = _get_reasoning_core_source()
        assert "_cross_div > 0.3" in src, (
            "0.3 threshold for cross-model divergence not found"
        )

    def test_divergence_error_evolution_episode(self):
        """Cross-model divergence must record an error_evolution episode."""
        src = _get_reasoning_core_source()
        assert "cross_model_prediction_divergence" in src

    def test_pressure_in_feedback_signals(self):
        """_build_feedback_extra_signals must emit
        cross_model_divergence_pressure."""
        src = _get_feedback_extra_source()
        assert "cross_model_divergence_pressure" in src

    def test_signal_registered(self):
        """cross_model_divergence_pressure must be registered."""
        src = _get_init_source()
        assert "cross_model_divergence_pressure" in src

    def test_model_has_cached_divergence(self):
        """Instantiated model must have _cached_cross_model_divergence."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_cached_cross_model_divergence')
        assert model._cached_cross_model_divergence == 0.0

    def test_divergence_surfaces_when_high(self):
        """When _cached_cross_model_divergence > 0.1, pressure emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_cross_model_divergence = 0.5
        extra = model._build_feedback_extra_signals()
        assert "cross_model_divergence_pressure" in extra
        assert abs(extra["cross_model_divergence_pressure"] - 0.5) < 0.01

    def test_divergence_not_surfaced_when_low(self):
        """When divergence <= 0.1, no pressure emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_cross_model_divergence = 0.05
        extra = model._build_feedback_extra_signals()
        assert "cross_model_divergence_pressure" not in extra

    def test_divergence_reset_per_pass(self):
        """_forward_impl must reset _cached_cross_model_divergence."""
        src = _get_forward_impl_source()
        assert "_cached_cross_model_divergence = 0.0" in src

    def test_error_class_bridge_mapping(self):
        """cross_model_prediction_divergence → cross_model_divergence_pressure."""
        src = _get_feedback_extra_source()
        assert "cross_model_prediction_divergence" in src


# ══════════════════════════════════════════════════════════════════════
# Integration tests: Full model with E-series patches
# ══════════════════════════════════════════════════════════════════════

class TestESeriesIntegration:
    """End-to-end tests verifying E-series patches work together."""

    def test_model_instantiation_with_all_patches(self):
        """AEONDeltaV3 must instantiate without errors after E patches."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert model is not None
        assert hasattr(model, '_cached_verified_prediction_error')
        assert hasattr(model, '_cached_gate_corruption_flag')
        assert hasattr(model, '_cached_trace_completeness_ratio')
        assert hasattr(model, '_cached_cross_model_divergence')

    def test_feedback_bus_e_series_signals_registered(self):
        """All four E-series feedback bus signals must be registered."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        signals = model.feedback_bus._extra_signals
        e_signals = [
            "verified_prediction_error_pressure",
            "gate_corruption_pressure",
            "trace_completeness_pressure",
            "cross_model_divergence_pressure",
        ]
        for sig in e_signals:
            assert sig in signals, f"E-series signal '{sig}' not registered"

    def test_feedback_bus_returns_all_e_signals_when_active(self):
        """When all E-series cached values are active, all signals emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        model._cached_verified_prediction_error = 0.7
        model._cached_gate_corruption_flag = True
        model._cached_trace_completeness_ratio = 0.5
        model._cached_cross_model_divergence = 0.4
        extra = model._build_feedback_extra_signals()
        assert "verified_prediction_error_pressure" in extra
        assert "gate_corruption_pressure" in extra
        assert "trace_completeness_pressure" in extra
        assert "cross_model_divergence_pressure" in extra

    def test_feedback_bus_returns_no_e_signals_when_healthy(self):
        """When all values are in healthy range, no E signals emitted."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # All defaults should be healthy
        extra = model._build_feedback_extra_signals()
        e_signals = [
            "verified_prediction_error_pressure",
            "gate_corruption_pressure",
            "trace_completeness_pressure",
            "cross_model_divergence_pressure",
        ]
        for sig in e_signals:
            assert sig not in extra, (
                f"E-series signal '{sig}' emitted in healthy state"
            )

    def test_error_evolution_has_e_series_bridge_mappings(self):
        """Error class bridge map must include all E-series error classes."""
        src = _get_feedback_extra_source()
        e_error_classes = [
            "non_finite_complexity_gate",
            "cross_model_prediction_divergence",
            "vibe_thinker_warmup_anomaly",
            "vibe_thinker_vq_seeding_anomaly",
        ]
        for cls in e_error_classes:
            assert cls in src, (
                f"E-series error class '{cls}' not in bridge map"
            )

    def test_e_series_all_pressure_values_clamped(self):
        """All E-series pressure signals must be clamped to [0, 1]."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # Set extreme values
        model._cached_verified_prediction_error = 5.0
        model._cached_gate_corruption_flag = True
        model._cached_trace_completeness_ratio = -1.0
        model._cached_cross_model_divergence = 10.0
        extra = model._build_feedback_extra_signals()
        for key in [
            "verified_prediction_error_pressure",
            "gate_corruption_pressure",
            "trace_completeness_pressure",
            "cross_model_divergence_pressure",
        ]:
            if key in extra:
                assert 0.0 <= extra[key] <= 1.0, (
                    f"{key}={extra[key]} not clamped to [0, 1]"
                )
