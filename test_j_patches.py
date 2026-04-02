"""Tests for J-series cognitive integration patches.

J-series patches close the remaining signal-flow gaps between high-level
cognition and low-level execution after patches A through I:

J1  – Integrity health degradation → error_evolution in verify_coherence
J2  – DAG cycle detection → error_evolution in verify_coherence
J4  – Cognitive unity deficit → error_evolution in verify_and_reinforce
J6  – Stall severity → feedback bus (_build_feedback_extra_signals)
J7  – Convergence quality deficit → feedback bus
J8a – coherence_check_provenance_dag_cycle → _class_to_signal mapping
J8b – critical_architectural_health → _class_to_signal mapping
J8c – cognitive_unity_deficit → _class_to_signal mapping
"""

import importlib
import inspect
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def _load_aeon():
    """Import aeon_core lazily."""
    return importlib.import_module("aeon_core")


def _make_config(**overrides):
    """Create a minimal AEONConfig with sensible test defaults."""
    aeon = _load_aeon()
    defaults = dict(
        hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
        seq_length=32, dropout_rate=0.0, meta_dim=32,
        lipschitz_target=0.9, vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


def _make_model(**config_overrides):
    """Create a minimal AEONDeltaV3 instance for testing."""
    import torch
    aeon = _load_aeon()
    cfg = _make_config(**config_overrides)
    with patch.object(
        aeon.AEONDeltaV3,
        "_vibe_thinker_first_start_calibration",
        lambda self: None,
    ):
        model = aeon.AEONDeltaV3(cfg)
    model.eval()
    return model


def _build_trigger():
    """Instantiate a MetaCognitiveRecursionTrigger for unit tests."""
    aeon = _load_aeon()
    return aeon.MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=2,
    )


# ===========================================================================
# J1: Integrity health degradation → error_evolution in verify_coherence
# ===========================================================================

class TestJ1IntegrityHealthEpisode:
    """Patch J1: When integrity health drops below threshold,
    an error_evolution episode must be recorded in verify_coherence."""

    def test_j1_patch_marker_in_source(self):
        """Patch J1 marker must exist in verify_coherence."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        assert "Patch J1" in src, (
            "Patch J1 marker not found in verify_coherence"
        )

    def test_j1_records_error_evolution(self):
        """verify_coherence must call error_evolution.record_episode for
        'critical_architectural_health'."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        assert "critical_architectural_health" in src, (
            "J1: error_class 'critical_architectural_health' must appear "
            "in verify_coherence source"
        )

    def test_j1_records_on_low_health(self):
        """When integrity health < 0.5, episode is recorded."""
        model = _make_model()

        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kw):
                self.episodes.append(kw)
            def get_recent_episodes(self, **kw):
                return self.episodes[-10:]
            def get_escalation_summary(self, **kw):
                return {"recurring_classes": {}, "escalation_count": 0,
                        "most_common_class": None, "unique_classes": set(),
                        "total_episodes": len(self.episodes)}

        ee = MockEE()
        model.error_evolution = ee

        # Patch integrity monitor to return low health
        model.integrity_monitor.get_global_health = lambda: 0.2

        try:
            model.verify_coherence()
        except Exception:
            pass

        critical_eps = [
            e for e in ee.episodes
            if e.get("error_class") == "critical_architectural_health"
        ]
        assert len(critical_eps) >= 1, (
            "J1: verify_coherence should record "
            "'critical_architectural_health' when integrity health < 0.5. "
            f"Got classes: {[e.get('error_class') for e in ee.episodes]}"
        )

    def test_j1_metadata_contains_health(self):
        """Episode metadata must contain integrity_health."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        # Find J1 region
        idx = src.index("Patch J1")
        region = src[idx:idx + 1000]
        assert "integrity_health" in region, (
            "J1: Episode metadata must include 'integrity_health'"
        )

    def test_j1_has_causal_antecedents(self):
        """Episode must include causal antecedents."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        idx = src.index("Patch J1")
        region = src[idx:idx + 1500]
        assert "causal_antecedents" in region, (
            "J1: Episode must include causal_antecedents"
        )


# ===========================================================================
# J2: DAG cycle detection → error_evolution in verify_coherence
# ===========================================================================

class TestJ2DagCycleEpisode:
    """Patch J2: When DAG validation finds cycles in verify_pipeline_wiring,
    an error_evolution episode must be recorded."""

    def test_j2_patch_marker_in_source(self):
        """Patch J2 marker must exist in verify_pipeline_wiring."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        assert "Patch J2" in src, (
            "Patch J2 marker not found in verify_pipeline_wiring"
        )

    def test_j2_records_dag_cycle_episode(self):
        """verify_pipeline_wiring must record
        'coherence_check_provenance_dag_cycle'."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        assert "coherence_check_provenance_dag_cycle" in src, (
            "J2: 'coherence_check_provenance_dag_cycle' must appear in "
            "verify_pipeline_wiring source"
        )

    def test_j2_checks_is_acyclic(self):
        """J2 must check dag_validation 'is_acyclic' field."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        idx = src.index("Patch J2")
        region = src[idx:idx + 800]
        assert "is_acyclic" in region, (
            "J2: Must check dag_validation['is_acyclic']"
        )

    def test_j2_has_causal_antecedents(self):
        """DAG cycle episode must include causal antecedents."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        idx = src.index("Patch J2")
        region = src[idx:idx + 1500]
        assert "causal_antecedents" in region

    def test_j2_records_cycles_found_metadata(self):
        """Episode metadata must contain cycles_found count."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        idx = src.index("Patch J2")
        region = src[idx:idx + 1500]
        assert "cycles_found" in region


# ===========================================================================
# J4: Cognitive unity deficit → error_evolution in verify_and_reinforce
# ===========================================================================

class TestJ4CognitiveUnityDeficit:
    """Patch J4: When cognitive_unity_score < 0.8, verify_and_reinforce
    must record a 'cognitive_unity_deficit' episode."""

    def test_j4_patch_marker_in_source(self):
        """Patch J4 marker must exist in verify_and_reinforce."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        assert "Patch J4" in src, (
            "Patch J4 marker not found in verify_and_reinforce"
        )

    def test_j4_records_unity_deficit(self):
        """verify_and_reinforce must record 'cognitive_unity_deficit'."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        assert "cognitive_unity_deficit" in src, (
            "J4: 'cognitive_unity_deficit' must appear in "
            "verify_and_reinforce source"
        )

    def test_j4_checks_threshold(self):
        """J4 must check if cognitive_unity_score < 0.8."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.index("Patch J4")
        region = src[idx:idx + 500]
        assert "0.8" in region, (
            "J4: Must check cognitive_unity_score against 0.8 threshold"
        )

    def test_j4_metadata_contains_score(self):
        """Episode metadata must include cognitive_unity_score."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.index("Patch J4")
        region = src[idx:idx + 500]
        assert "cognitive_unity_score" in region

    def test_j4_has_causal_antecedents(self):
        """Unity deficit episode must include causal antecedents."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.index("Patch J4")
        region = src[idx:idx + 1500]
        assert "causal_antecedents" in region


# ===========================================================================
# J6: Stall severity → feedback bus
# ===========================================================================

class TestJ6StallSeverityFeedbackBus:
    """Patch J6: _build_feedback_extra_signals must surface
    _cached_stall_severity as 'stall_severity_pressure'."""

    def test_j6_patch_marker_in_source(self):
        """Patch J6 marker must exist in _build_feedback_extra_signals."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "Patch J6" in src, (
            "Patch J6 marker not found in _build_feedback_extra_signals"
        )

    def test_j6_emits_stall_severity_pressure(self):
        """Source must emit 'stall_severity_pressure'."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "stall_severity_pressure" in src

    def test_j6_reads_cached_stall_severity(self):
        """J6 must read _cached_stall_severity."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        idx = src.index("Patch J6")
        region = src[idx:idx + 400]
        assert "_cached_stall_severity" in region

    def test_j6_stall_present_when_nonzero(self):
        """When _cached_stall_severity > 0, signal appears in extra."""
        model = _make_model()
        model._cached_stall_severity = 0.6
        extra = model._build_feedback_extra_signals()
        assert "stall_severity_pressure" in extra, (
            "J6: 'stall_severity_pressure' must be present when "
            "_cached_stall_severity > 0"
        )
        assert abs(extra["stall_severity_pressure"] - 0.6) < 0.01

    def test_j6_stall_absent_when_zero(self):
        """When _cached_stall_severity is 0, no pressure signal emitted."""
        model = _make_model()
        model._cached_stall_severity = 0.0
        extra = model._build_feedback_extra_signals()
        assert "stall_severity_pressure" not in extra

    def test_j6_stall_clamped(self):
        """Stall severity pressure is clamped to [0, 1]."""
        model = _make_model()
        model._cached_stall_severity = 1.5
        extra = model._build_feedback_extra_signals()
        assert "stall_severity_pressure" in extra
        assert 0.0 <= extra["stall_severity_pressure"] <= 1.0


# ===========================================================================
# J7: Convergence quality deficit → feedback bus
# ===========================================================================

class TestJ7ConvergenceQualityDeficitFeedbackBus:
    """Patch J7: _build_feedback_extra_signals must surface
    convergence quality deficit when quality < 0.9."""

    def test_j7_patch_marker_in_source(self):
        """Patch J7 marker must exist in _build_feedback_extra_signals."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "Patch J7" in src

    def test_j7_emits_convergence_quality_deficit(self):
        """Source must emit 'convergence_quality_deficit'."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "convergence_quality_deficit" in src

    def test_j7_deficit_present_when_quality_low(self):
        """When _cached_convergence_quality < 0.9, deficit appears."""
        model = _make_model()
        model._cached_convergence_quality = 0.3
        extra = model._build_feedback_extra_signals()
        assert "convergence_quality_deficit" in extra
        expected = 1.0 - 0.3  # = 0.7
        assert abs(extra["convergence_quality_deficit"] - expected) < 0.01

    def test_j7_deficit_absent_when_quality_high(self):
        """When _cached_convergence_quality >= 0.9, no deficit signal."""
        model = _make_model()
        model._cached_convergence_quality = 0.95
        extra = model._build_feedback_extra_signals()
        assert "convergence_quality_deficit" not in extra

    def test_j7_deficit_absent_at_threshold(self):
        """Exactly 0.9 should NOT produce deficit signal."""
        model = _make_model()
        model._cached_convergence_quality = 0.9
        extra = model._build_feedback_extra_signals()
        assert "convergence_quality_deficit" not in extra

    def test_j7_deficit_clamped(self):
        """Convergence quality deficit is clamped to [0, 1]."""
        model = _make_model()
        model._cached_convergence_quality = -0.5
        extra = model._build_feedback_extra_signals()
        if "convergence_quality_deficit" in extra:
            assert 0.0 <= extra["convergence_quality_deficit"] <= 1.0


# ===========================================================================
# J8: _class_to_signal mappings
# ===========================================================================

class TestJ8ClassToSignalMappings:
    """Patch J8: New error classes from J-series patches must have
    explicit _class_to_signal mappings."""

    def test_j8a_dag_cycle_mapping_in_source(self):
        """'coherence_check_provenance_dag_cycle' must appear in
        adapt_weights_from_evolution source."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "coherence_check_provenance_dag_cycle" in src, (
            "J8a: 'coherence_check_provenance_dag_cycle' must be in "
            "adapt_weights_from_evolution source"
        )

    def test_j8a_maps_to_low_causal_quality(self):
        """'coherence_check_provenance_dag_cycle' → 'low_causal_quality'."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        # Check that the mapping is to the correct signal
        idx = src.index("coherence_check_provenance_dag_cycle")
        region = src[idx:idx + 500]
        assert "low_causal_quality" in region, (
            "J8a: Must map to 'low_causal_quality'"
        )

    def test_j8b_critical_health_mapping_in_source(self):
        """'critical_architectural_health' must appear in
        adapt_weights_from_evolution source."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "critical_architectural_health" in src

    def test_j8b_maps_to_coherence_deficit(self):
        """'critical_architectural_health' → 'coherence_deficit'."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        idx = src.index('"critical_architectural_health"')
        region = src[idx:idx + 100]
        assert "coherence_deficit" in region

    def test_j8c_unity_deficit_mapping_in_source(self):
        """'cognitive_unity_deficit' must appear in
        adapt_weights_from_evolution source."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "cognitive_unity_deficit" in src

    def test_j8c_maps_to_coherence_deficit(self):
        """'cognitive_unity_deficit' → 'coherence_deficit'."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        # Find the J8c mapping specifically
        idx = src.index("Patch J8c")
        region = src[idx:idx + 400]
        assert "coherence_deficit" in region


# ===========================================================================
# Cross-cutting integration tests
# ===========================================================================

class TestJSeriesCrossCutting:
    """End-to-end tests ensuring J-series patches work together."""

    def test_all_j_error_classes_in_adapt_weights(self):
        """Every error class introduced by J-series patches appears in
        adapt_weights_from_evolution source for _class_to_signal routing."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        j_error_classes = [
            "critical_architectural_health",
            "coherence_check_provenance_dag_cycle",
            "cognitive_unity_deficit",
        ]
        for ec in j_error_classes:
            assert ec in src, (
                f"Error class '{ec}' must appear in "
                f"adapt_weights_from_evolution source"
            )

    def test_feedback_bus_j6_j7_both_present(self):
        """_build_feedback_extra_signals produces both J6 and J7 signals
        when their respective caches are set."""
        model = _make_model()
        model._cached_stall_severity = 0.5
        model._cached_convergence_quality = 0.3

        extra = model._build_feedback_extra_signals()
        assert "stall_severity_pressure" in extra
        assert "convergence_quality_deficit" in extra

    def test_verify_coherence_j1_source_integrity(self):
        """J1 patch must be in verify_coherence and reference error_evolution."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        idx = src.index("Patch J1")
        region = src[idx:idx + 1000]
        assert "error_evolution" in region
        assert "record_episode" in region

    def test_verify_pipeline_wiring_j2_source_integrity(self):
        """J2 patch must be in verify_pipeline_wiring and reference
        error_evolution."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_pipeline_wiring)
        idx = src.index("Patch J2")
        region = src[idx:idx + 1500]
        assert "error_evolution" in region
        assert "record_episode" in region

    def test_verify_and_reinforce_j4_source_integrity(self):
        """J4 patch must be in verify_and_reinforce and reference
        error_evolution."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.index("Patch J4")
        region = src[idx:idx + 1500]
        assert "error_evolution" in region
        assert "record_episode" in region


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
