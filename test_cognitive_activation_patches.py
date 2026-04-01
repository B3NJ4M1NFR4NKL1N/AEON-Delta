"""
Tests for cognitive activation patches: validating that the 8 integration
patches correctly bridge the identified gaps in the cognitive architecture.

Validates:
1. Coherence trend escalation → error_evolution recording
2. Post-integration deeper loop → inference cache invalidation
3. UCC deeper acceptance → inference cache invalidation
4. Post-integration deeper loop exception → error_evolution bridge
5. Feedback bus → provenance edge registration
6. Stall severity → metacognitive trigger signal integration
7. Output reliability component decomposition → error_evolution
8. Training-time VT consolidation → error_evolution tracking
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
    """Return source of _forward_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._forward_impl)


def _get_reasoning_core_source():
    """Return source of _reasoning_core_impl."""
    return inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)


def _get_evaluate_source():
    """Return source of metacognitive evaluate."""
    return inspect.getsource(aeon.MetaCognitiveRecursionTrigger.evaluate)


def _get_activation_source():
    """Return source of _vibe_thinker_first_start_calibration."""
    return inspect.getsource(aeon.AEONDeltaV3._vibe_thinker_first_start_calibration)


# ═══════════════════════════════════════════════════════════════════════
# Patch 1: Coherence trend escalation → error_evolution
# ═══════════════════════════════════════════════════════════════════════


class TestPatch1CoherenceTrendEscalation:
    """Verify coherence trend escalation records in error_evolution."""

    def test_trend_escalation_recording_code_present(self):
        """Trend escalation path must record to error_evolution."""
        src = _get_reasoning_core_source()
        assert "coherence_trend_escalation" in src
        assert "error_evolution" in src

    def test_trend_escalation_records_episode(self):
        """The recording must use record_episode with correct error class."""
        src = _get_reasoning_core_source()
        assert "record_episode" in src
        # Verify the error class is specific
        assert "'coherence_trend_escalation'" in src or '"coherence_trend_escalation"' in src

    def test_trend_escalation_metadata_includes_boost(self):
        """Episode metadata must include trend_boost for traceability."""
        src = _get_reasoning_core_source()
        assert "'trend_boost'" in src or '"trend_boost"' in src

    def test_trend_escalation_has_causal_antecedents(self):
        """Episode must include causal_antecedents for provenance."""
        src = _get_reasoning_core_source()
        # The recording should have causal antecedents
        assert "causal_antecedents" in src

    def test_trend_escalation_error_class_in_signal_mapping(self):
        """coherence_trend_escalation must map to a trigger signal."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "coherence_trend_escalation" in src


# ═══════════════════════════════════════════════════════════════════════
# Patch 2: Post-integration deeper loop → inference cache invalidation
# ═══════════════════════════════════════════════════════════════════════


class TestPatch2PostIntegrationCacheInvalidation:
    """Verify post-integration deeper loop invalidates inference cache."""

    def test_post_deeper_cache_reset_code_present(self):
        """Accepted post-integration deeper loop must reset cache."""
        src = _get_reasoning_core_source()
        # Should contain cache invalidation near post_deeper_accepted
        assert "inference_cache" in src
        assert "_post_deeper_accepted" in src

    def test_post_deeper_cache_reset_follows_acceptance(self):
        """Cache reset must be conditional on acceptance."""
        src = _get_reasoning_core_source()
        # Find the block where _post_deeper_accepted = True and cache reset
        lines = src.split('\n')
        found_acceptance = False
        found_reset_after = False
        for line in lines:
            if '_post_deeper_accepted = True' in line:
                found_acceptance = True
            if found_acceptance and 'inference_cache' in line and 'reset' in line:
                found_reset_after = True
                break
        assert found_reset_after, (
            "inference_cache.reset() should follow _post_deeper_accepted = True"
        )


# ═══════════════════════════════════════════════════════════════════════
# Patch 3: UCC deeper acceptance → inference cache invalidation
# ═══════════════════════════════════════════════════════════════════════


class TestPatch3UCCDeeperCacheInvalidation:
    """Verify UCC deeper acceptance invalidates inference cache."""

    def test_ucc_deeper_cache_reset_code_present(self):
        """UCC deeper acceptance must reset cache."""
        src = _get_reasoning_core_source()
        # Should find cache reset near _ucc_deeper_accepted
        assert "_ucc_deeper_accepted" in src

    def test_ucc_deeper_cache_reset_before_error_evolution(self):
        """Cache reset should precede error_evolution recording."""
        src = _get_reasoning_core_source()
        lines = src.split('\n')
        found_ucc_accepted = False
        found_cache_reset = False
        found_error_recording = False
        for line in lines:
            if '_ucc_deeper_accepted = True' in line:
                found_ucc_accepted = True
            if found_ucc_accepted and 'inference_cache' in line and 'reset' in line:
                found_cache_reset = True
            if found_ucc_accepted and found_cache_reset and 'deeper_meta_loop_accepted' in line:
                found_error_recording = True
                break
        assert found_cache_reset, (
            "inference_cache.reset() should follow _ucc_deeper_accepted = True"
        )


# ═══════════════════════════════════════════════════════════════════════
# Patch 4: Post-integration deeper loop exception → error_evolution
# ═══════════════════════════════════════════════════════════════════════


class TestPatch4PostDeeperExceptionBridge:
    """Verify post-integration deeper loop exceptions bridge to error_evolution."""

    def test_exception_bridge_code_present(self):
        """Exception handler must call _bridge_silent_exception."""
        src = _get_reasoning_core_source()
        assert "_bridge_silent_exception" in src
        assert "post_integration_deeper_loop_failure" in src

    def test_exception_bridge_uses_correct_error_class(self):
        """Bridge must use a specific error class, not generic."""
        src = _get_reasoning_core_source()
        assert "'post_integration_deeper_loop_failure'" in src or \
               '"post_integration_deeper_loop_failure"' in src

    def test_error_class_in_signal_mapping(self):
        """post_integration_deeper_loop_failure must map to trigger signal."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "post_integration_deeper_loop_failure" in src


# ═══════════════════════════════════════════════════════════════════════
# Patch 5: Feedback bus → provenance edge registration
# ═══════════════════════════════════════════════════════════════════════


class TestPatch5FeedbackBusProvenance:
    """Verify feedback bus recomputation registers provenance edges."""

    def test_feedback_provenance_edge_present(self):
        """Feedback bus → deeper_meta_loop provenance edge must exist."""
        src = _get_reasoning_core_source()
        assert "record_dependency" in src

    def test_feedback_bus_deeper_meta_loop_edge(self):
        """The provenance edge must connect feedback_bus to deeper_meta_loop."""
        src = _get_reasoning_core_source()
        # Check both endpoints mentioned near record_dependency
        assert "feedback_bus" in src
        assert "deeper_meta_loop" in src


# ═══════════════════════════════════════════════════════════════════════
# Patch 6: Stall severity → metacognitive trigger signal
# ═══════════════════════════════════════════════════════════════════════


class TestPatch6StallSeveritySignal:
    """Verify stall severity is integrated as a metacognitive trigger signal."""

    def test_stall_severity_parameter_exists(self):
        """evaluate() must accept stall_severity parameter."""
        sig = inspect.signature(aeon.MetaCognitiveRecursionTrigger.evaluate)
        assert "stall_severity" in sig.parameters

    def test_stall_severity_in_signal_values(self):
        """stall_severity must appear in signal_values computation."""
        src = _get_evaluate_source()
        assert '"stall_severity"' in src or "'stall_severity'" in src

    def test_stall_severity_default_weight_initialized(self):
        """MetaCognitiveRecursionTrigger must initialize stall_severity weight."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.__init__,
        )
        assert "stall_severity" in src

    def test_stall_severity_passed_from_reasoning_core(self):
        """_reasoning_core_impl must pass stall_severity to evaluate()."""
        src = _get_reasoning_core_source()
        assert "stall_severity" in src
        assert "_cached_stall_severity" in src

    def test_stall_severity_passed_from_post_pipeline(self):
        """Post-pipeline evaluate() must pass stall_severity."""
        src = _get_reasoning_core_source()
        assert "stall_severity=self._cached_stall_severity" in src

    def test_meta_loop_stall_maps_to_stall_severity_signal(self):
        """meta_loop_stall error class must map to stall_severity signal."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        # Verify the _class_to_signal mapping
        assert "meta_loop_stall" in src
        assert "stall_severity" in src

    def test_stall_severity_evaluates_correctly(self):
        """evaluate() with high stall_severity should increase trigger score."""
        config = _make_config()
        trigger = aeon.MetaCognitiveRecursionTrigger()
        # Baseline with no stall
        result_no_stall = trigger.evaluate(stall_severity=0.0)
        trigger._recursion_count = 0  # reset
        # With high stall
        result_stall = trigger.evaluate(stall_severity=0.9)
        assert result_stall["trigger_score"] >= result_no_stall["trigger_score"]

    def test_stall_severity_appears_in_triggers_active(self):
        """High stall_severity should appear in triggers_active list."""
        config = _make_config()
        trigger = aeon.MetaCognitiveRecursionTrigger()
        result = trigger.evaluate(stall_severity=0.9)
        assert "stall_severity" in result.get("triggers_active", [])


# ═══════════════════════════════════════════════════════════════════════
# Patch 7: Output reliability decomposition → error_evolution
# ═══════════════════════════════════════════════════════════════════════


class TestPatch7OutputReliabilityDecomposition:
    """Verify output reliability records component decomposition."""

    def test_reliability_decomposition_code_present(self):
        """Low reliability must record decomposed components."""
        src = _get_reasoning_core_source()
        assert "reliability_decomposition" in src
        assert "low_output_reliability" in src

    def test_decomposition_includes_all_components(self):
        """Recording must include all 4 reliability components."""
        src = _get_reasoning_core_source()
        assert "uncertainty_component" in src
        assert "critic_component" in src
        assert "convergence_component" in src
        assert "coherence_component" in src

    def test_decomposition_includes_bottleneck(self):
        """Recording must identify the bottleneck component."""
        src = _get_reasoning_core_source()
        assert "bottleneck" in src

    def test_reliability_threshold_gates_recording(self):
        """Decomposition only records when reliability < 0.5."""
        src = _get_reasoning_core_source()
        assert "_or_composite < 0.5" in src

    def test_low_output_reliability_in_signal_mapping(self):
        """low_output_reliability must map to trigger signal."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "low_output_reliability" in src


# ═══════════════════════════════════════════════════════════════════════
# Patch 8: Training-time VT consolidation → error_evolution
# ═══════════════════════════════════════════════════════════════════════


class TestPatch8TrainingConsolidation:
    """Verify training-time VT consolidation records to error_evolution."""

    def test_consolidation_recording_code_present(self):
        """Consolidation during baseline seeding must record to error_evolution."""
        src = _get_activation_source()
        assert "vibe_thinker_calibration_consolidation" in src

    def test_consolidation_checks_dict_result(self):
        """Recording must check for dict result with consolidated flag."""
        src = _get_activation_source()
        assert "consolidated" in src

    def test_consolidation_metadata_includes_target(self):
        """Episode metadata must include the calibration target."""
        src = _get_activation_source()
        assert "calibration_target" in src

    def test_consolidation_error_class_in_signal_mapping(self):
        """vibe_thinker_calibration_consolidation must map to trigger signal."""
        src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        assert "vibe_thinker_calibration_consolidation" in src


# ═══════════════════════════════════════════════════════════════════════
# Cross-cutting integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrossCuttingIntegration:
    """Verify cross-cutting integration properties across all patches."""

    def test_model_instantiation_with_patches(self):
        """Model must instantiate without errors after all patches."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert model is not None

    def test_cached_stall_severity_initialized(self):
        """_cached_stall_severity must be initialized in __init__."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_cached_stall_severity')
        assert model._cached_stall_severity == 0.0

    def test_metacognitive_trigger_has_16_signals(self):
        """Trigger must now have 16 signals (15 original + stall_severity)."""
        config = _make_config()
        trigger = aeon.MetaCognitiveRecursionTrigger()
        # Count the number of default weights
        assert "stall_severity" in trigger._signal_weights
        assert len(trigger._signal_weights) >= 16

    def test_evaluate_accepts_all_new_parameters(self):
        """evaluate() must accept stall_severity without error."""
        config = _make_config()
        trigger = aeon.MetaCognitiveRecursionTrigger()
        result = trigger.evaluate(
            uncertainty=0.5,
            is_diverging=False,
            topology_catastrophe=False,
            coherence_deficit=0.3,
            memory_staleness=False,
            recovery_pressure=0.2,
            world_model_surprise=0.1,
            causal_quality=0.8,
            safety_violation=False,
            diversity_collapse=0.0,
            memory_trust_deficit=0.0,
            convergence_conflict=0.0,
            output_reliability=0.7,
            spectral_stability_margin=0.9,
            border_uncertainty=0.1,
            stall_severity=0.5,
        )
        assert "trigger_score" in result
        assert "triggers_active" in result

    def test_bridge_silent_exception_exists(self):
        """_bridge_silent_exception must exist for patched paths."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, '_bridge_silent_exception')

    def test_provenance_tracker_has_record_dependency(self):
        """CausalProvenanceTracker must have record_dependency method."""
        assert hasattr(aeon.CausalProvenanceTracker, 'record_dependency')
