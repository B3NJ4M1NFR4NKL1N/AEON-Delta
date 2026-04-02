"""Tests for F-series cognitive integration patches.

F1:  DAG revision exception → _bridge_silent_exception()
F2:  DAG auto-critic exception → _bridge_silent_exception()
F3:  UCC evaluate() verdict → error_evolution recording
F4:  Post-pipeline metacognitive verdict → error_evolution recording
F5:  Verify-coherence metacognitive verdict → error_evolution recording
F6:  Cognitive unity non-triggered path → error_evolution recording
F7:  Cognitive unity meta-evaluation exception → _bridge_silent_exception()
F8:  Provenance record_dependency() for coherence_verifier, provenance
     tracker, and chain_validator subsystem interactions
F9:  Provenance adapt_thresholds exception → _bridge_silent_exception()
F10: UCC verdict + post-pipeline verdict → feedback_bus signals
"""

import inspect
import os
import sys

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


def _get_forward_impl_source():
    return inspect.getsource(aeon.AEONDeltaV3._forward_impl)


def _get_ucc_source():
    return inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)


def _get_verify_coherence_source():
    return inspect.getsource(aeon.AEONDeltaV3.verify_coherence)


def _get_verify_reinforce_source():
    return inspect.getsource(
        aeon.AEONDeltaV3.verify_and_reinforce,
    )


# ══════════════════════════════════════════════════════════════════════
# F1: DAG revision exception → _bridge_silent_exception
# ══════════════════════════════════════════════════════════════════════

class TestF1DagRevisionExceptionBridge:
    """F1: Silent DAG revision exception must be bridged to
    error_evolution and causal_trace via _bridge_silent_exception."""

    def test_bridge_call_present_in_reasoning_core(self):
        """_reasoning_core_impl must call _bridge_silent_exception
        for dag_revision_consensus_failure."""
        src = _get_reasoning_core_source()
        assert "_bridge_silent_exception" in src, (
            "_bridge_silent_exception not found in _reasoning_core_impl"
        )
        assert "dag_revision_consensus_failure" in src, (
            "dag_revision_consensus_failure error_class not in source"
        )

    def test_bridge_references_auto_critic_subsystem(self):
        """The bridge call must reference 'auto_critic' as subsystem."""
        src = _get_reasoning_core_source()
        # Find the dag_revision block
        idx = src.find("dag_revision_consensus_failure")
        assert idx >= 0
        # The subsystem parameter should be nearby
        context = src[max(0, idx - 200):idx + 200]
        assert "'auto_critic'" in context, (
            "auto_critic subsystem not referenced near dag_revision bridge"
        )

    def test_bridge_receives_exception_variable(self):
        """The _dag_rev_err variable must be passed to the bridge."""
        src = _get_reasoning_core_source()
        idx = src.find("dag_revision_consensus_failure")
        assert idx >= 0
        context = src[max(0, idx - 100):idx + 200]
        assert "_dag_rev_err" in context, (
            "_dag_rev_err not passed to _bridge_silent_exception"
        )

    def test_runtime_bridge_records_episode(self):
        """Calling _bridge_silent_exception must record in error_evolution."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        initial_count = len(model.error_evolution._episodes)
        model._bridge_silent_exception(
            'dag_revision_consensus_failure',
            'auto_critic',
            RuntimeError("test dag revision failure"),
        )
        assert len(model.error_evolution._episodes) > initial_count, (
            "_bridge_silent_exception did not record episode"
        )

    def test_runtime_bridge_records_causal_trace(self):
        """Calling _bridge_silent_exception must record in causal_trace."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        initial_entries = len(model.causal_trace._entries)
        model._bridge_silent_exception(
            'dag_revision_consensus_failure',
            'auto_critic',
            RuntimeError("test dag revision failure"),
        )
        assert len(model.causal_trace._entries) > initial_entries, (
            "_bridge_silent_exception did not record causal trace"
        )


# ══════════════════════════════════════════════════════════════════════
# F2: DAG auto-critic exception → _bridge_silent_exception
# ══════════════════════════════════════════════════════════════════════

class TestF2DagAutoCriticExceptionBridge:
    """F2: Silent DAG auto-critic exception must be bridged."""

    def test_bridge_call_present_for_auto_critic(self):
        """_reasoning_core_impl must call _bridge_silent_exception
        for dag_auto_critic_failure."""
        src = _get_reasoning_core_source()
        assert "dag_auto_critic_failure" in src, (
            "dag_auto_critic_failure error_class not in source"
        )

    def test_bridge_references_auto_critic_subsystem(self):
        """The bridge call must reference 'auto_critic' as subsystem."""
        src = _get_reasoning_core_source()
        idx = src.find("dag_auto_critic_failure")
        assert idx >= 0
        context = src[max(0, idx - 200):idx + 200]
        assert "'auto_critic'" in context, (
            "auto_critic subsystem not near dag_auto_critic_failure bridge"
        )

    def test_bridge_receives_exception_variable(self):
        """The _dag_ac_err variable must be passed to the bridge."""
        src = _get_reasoning_core_source()
        idx = src.find("dag_auto_critic_failure")
        assert idx >= 0
        context = src[max(0, idx - 100):idx + 200]
        assert "_dag_ac_err" in context, (
            "_dag_ac_err not passed to _bridge_silent_exception"
        )


# ══════════════════════════════════════════════════════════════════════
# F3: UCC evaluate() verdict → error_evolution recording
# ══════════════════════════════════════════════════════════════════════

class TestF3UccVerdictRecording:
    """F3: The UCC evaluate() verdict must be recorded in
    error_evolution so the metacognitive trigger can learn."""

    def test_ucc_verdict_episode_in_reasoning_core(self):
        """_reasoning_core_impl must record 'ucc_evaluation_verdict'."""
        src = _get_reasoning_core_source()
        assert "ucc_evaluation_verdict" in src, (
            "ucc_evaluation_verdict not recorded in _reasoning_core_impl"
        )

    def test_ucc_verdict_records_should_rerun(self):
        """The recorded episode metadata must include should_rerun."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_evaluation_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "should_rerun" in context, (
            "should_rerun not in UCC verdict metadata"
        )

    def test_ucc_verdict_records_trigger_score(self):
        """The recorded episode metadata must include trigger_score."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_evaluation_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "trigger_score" in context, (
            "trigger_score not in UCC verdict metadata"
        )

    def test_ucc_verdict_records_coherence_deficit(self):
        """The recorded episode metadata must include coherence_deficit."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_evaluation_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "coherence_deficit" in context, (
            "coherence_deficit not in UCC verdict metadata"
        )

    def test_ucc_verdict_has_causal_antecedents(self):
        """The recording must include causal_antecedents for tracing."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_evaluation_verdict")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "causal_antecedents" in context, (
            "causal_antecedents not in UCC verdict recording"
        )
        assert "unified_cognitive_cycle" in context, (
            "unified_cognitive_cycle not in causal_antecedents"
        )

    def test_ucc_verdict_strategy(self):
        """Strategy must be 'unified_cognitive_cycle'."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_evaluation_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "unified_cognitive_cycle" in context


# ══════════════════════════════════════════════════════════════════════
# F4: Post-pipeline metacognitive verdict → error_evolution
# ══════════════════════════════════════════════════════════════════════

class TestF4PostPipelineVerdictRecording:
    """F4: Post-pipeline metacognitive evaluation verdict must be
    recorded in error_evolution."""

    def test_post_pipeline_verdict_in_forward_impl(self):
        """_forward_impl must record 'post_pipeline_metacog_verdict'."""
        src = _get_forward_impl_source()
        assert "post_pipeline_metacog_verdict" in src, (
            "post_pipeline_metacog_verdict not in _forward_impl"
        )

    def test_post_pipeline_verdict_records_trigger_score(self):
        """Metadata must include trigger_score."""
        src = _get_forward_impl_source()
        idx = src.find("post_pipeline_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 500]
        assert "trigger_score" in context

    def test_post_pipeline_verdict_records_triggers_active(self):
        """Metadata must include triggers_active list."""
        src = _get_forward_impl_source()
        idx = src.find("post_pipeline_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 500]
        assert "triggers_active" in context

    def test_post_pipeline_verdict_has_antecedents(self):
        """Recording must include causal_antecedents."""
        src = _get_forward_impl_source()
        idx = src.find("post_pipeline_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "causal_antecedents" in context
        assert "post_pipeline_gate" in context

    def test_post_pipeline_verdict_records_correction_strategy(self):
        """Metadata must include correction_strategy."""
        src = _get_forward_impl_source()
        idx = src.find("post_pipeline_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "correction_strategy" in context


# ══════════════════════════════════════════════════════════════════════
# F5: Verify-coherence metacognitive verdict → error_evolution
# ══════════════════════════════════════════════════════════════════════

class TestF5VerifyCoherenceVerdictRecording:
    """F5: The verify_coherence metacognitive trigger verdict must
    be recorded in error_evolution."""

    def test_verify_coherence_verdict_in_source(self):
        """verify_coherence must record
        'verify_coherence_metacog_verdict'."""
        src = _get_verify_coherence_source()
        assert "verify_coherence_metacog_verdict" in src, (
            "verify_coherence_metacog_verdict not recorded"
        )

    def test_verify_coherence_records_trigger_score(self):
        """Metadata must include trigger_score."""
        src = _get_verify_coherence_source()
        idx = src.find("verify_coherence_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "trigger_score" in context

    def test_verify_coherence_records_coherence_deficit(self):
        """Metadata must include coherence_deficit."""
        src = _get_verify_coherence_source()
        idx = src.find("verify_coherence_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "coherence_deficit" in context

    def test_verify_coherence_has_antecedents(self):
        """Recording must include causal_antecedents."""
        src = _get_verify_coherence_source()
        idx = src.find("verify_coherence_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "causal_antecedents" in context
        assert "verify_coherence" in context
        assert "metacognitive_trigger" in context

    def test_verify_coherence_strategy(self):
        """Strategy must be 'out_of_band_coherence_check'."""
        src = _get_verify_coherence_source()
        idx = src.find("verify_coherence_metacog_verdict")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "out_of_band_coherence_check" in context


# ══════════════════════════════════════════════════════════════════════
# F6: Cognitive unity non-triggered path → error_evolution
# ══════════════════════════════════════════════════════════════════════

class TestF6CognitiveUnityPassedRecording:
    """F6: When cognitive unity check passes (should_recurse=False),
    the success must be recorded in error_evolution."""

    def test_unity_passed_episode_in_source(self):
        """verify_and_reinforce must record 'cognitive_unity_check'."""
        src = _get_verify_reinforce_source()
        assert "cognitive_unity_check" in src, (
            "cognitive_unity_check episode not in verify_and_reinforce"
        )

    def test_unity_passed_records_success_true(self):
        """The success=True path must exist for non-triggered checks."""
        src = _get_verify_reinforce_source()
        idx = src.find("cognitive_unity_check")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "success=True" in context, (
            "success=True not found for cognitive_unity_check"
        )

    def test_unity_passed_records_overall_score(self):
        """Metadata must include overall_score."""
        src = _get_verify_reinforce_source()
        idx = src.find("cognitive_unity_check")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert "overall_score" in context

    def test_unity_passed_has_antecedents(self):
        """Recording must include causal_antecedents."""
        src = _get_verify_reinforce_source()
        idx = src.find("cognitive_unity_check")
        assert idx >= 0
        context = src[idx:idx + 800]
        assert "causal_antecedents" in context
        assert "cognitive_unity_passed" in context

    def test_unity_both_paths_covered(self):
        """Both triggered and non-triggered paths must record episodes."""
        src = _get_verify_reinforce_source()
        assert "cognitive_unity_deficit" in src, (
            "Triggered path (cognitive_unity_deficit) missing"
        )
        assert "cognitive_unity_check" in src, (
            "Non-triggered path (cognitive_unity_check) missing"
        )


# ══════════════════════════════════════════════════════════════════════
# F7: Cognitive unity exception → _bridge_silent_exception
# ══════════════════════════════════════════════════════════════════════

class TestF7CognitiveUnityExceptionBridge:
    """F7: Cognitive unity meta-evaluation exception must be bridged
    via _bridge_silent_exception for full causal coverage."""

    def test_bridge_call_in_verify_reinforce(self):
        """verify_and_reinforce must call _bridge_silent_exception
        for cognitive_unity_meta_evaluation_failure."""
        src = _get_verify_reinforce_source()
        assert "_bridge_silent_exception" in src, (
            "_bridge_silent_exception not found in verify_and_reinforce"
        )
        assert "cognitive_unity_meta_evaluation_failure" in src, (
            "cognitive_unity_meta_evaluation_failure not in source"
        )

    def test_bridge_references_verify_and_reinforce_subsystem(self):
        """The bridge call must reference 'verify_and_reinforce'."""
        src = _get_verify_reinforce_source()
        idx = src.find("cognitive_unity_meta_evaluation_failure")
        assert idx >= 0
        context = src[max(0, idx - 200):idx + 200]
        assert "'verify_and_reinforce'" in context, (
            "verify_and_reinforce subsystem not in bridge call"
        )

    def test_bridge_receives_exception_variable(self):
        """The _unity_meta_err variable must be passed to the bridge."""
        src = _get_verify_reinforce_source()
        idx = src.find("cognitive_unity_meta_evaluation_failure")
        assert idx >= 0
        context = src[max(0, idx - 100):idx + 200]
        assert "_unity_meta_err" in context, (
            "_unity_meta_err not passed to _bridge_silent_exception"
        )


# ══════════════════════════════════════════════════════════════════════
# F8: Provenance record_dependency for subsystem interactions
# ══════════════════════════════════════════════════════════════════════

class TestF8ProvenanceDependencyRecording:
    """F8: Subsystem interactions within the UCC must record
    provenance dependencies for causal traceability."""

    def test_coherence_verifier_dependency_in_ucc(self):
        """UCC evaluate must record coherence_verifier→ucc dependency."""
        src = _get_ucc_source()
        assert "record_dependency" in src, (
            "record_dependency not called in UCC evaluate"
        )
        # Check for the specific dependency
        assert "coherence_verifier" in src, (
            "coherence_verifier dependency not recorded"
        )

    def test_provenance_tracker_pipeline_dependency(self):
        """UCC evaluate must record provenance_tracker→ucc dependency."""
        src = _get_ucc_source()
        assert "ucc_pipeline_validation" in src, (
            "ucc_pipeline_validation dependency not recorded"
        )

    def test_chain_validator_dependency(self):
        """UCC evaluate must record chain_validator→ucc dependency."""
        src = _get_ucc_source()
        assert "provenance_chain_validator" in src, (
            "provenance_chain_validator dependency not recorded"
        )

    def test_runtime_record_dependency_creates_edge(self):
        """record_dependency must create an edge in the dependency graph."""
        tracker = aeon.CausalProvenanceTracker()
        tracker.record_dependency("coherence_verifier", "ucc")
        deps = tracker._dependencies
        assert "ucc" in deps and "coherence_verifier" in deps["ucc"], (
            "record_dependency did not create expected edge"
        )

    def test_multiple_dependencies_recorded(self):
        """Multiple record_dependency calls must accumulate."""
        tracker = aeon.CausalProvenanceTracker()
        tracker.record_dependency("coherence_verifier", "ucc")
        tracker.record_dependency("provenance_tracker", "ucc_pipeline_validation")
        tracker.record_dependency("provenance_chain_validator", "ucc")
        # The tracker should have at least 3 dependency targets
        assert len(tracker._dependencies) >= 2, (
            "Not all dependencies accumulated"
        )


# ══════════════════════════════════════════════════════════════════════
# F9: Provenance adapt_thresholds exception → _bridge_silent_exception
# ══════════════════════════════════════════════════════════════════════

class TestF9ProvenanceAdaptExceptionBridge:
    """F9: Provenance adapt_thresholds failures must be bridged to
    the metacognitive feedback loop."""

    def test_bridge_call_in_ucc(self):
        """UCC evaluate must call _bridge_silent_exception for
        provenance_adapt_thresholds_failure."""
        src = _get_ucc_source()
        assert "provenance_adapt_thresholds_failure" in src, (
            "provenance_adapt_thresholds_failure not in UCC source"
        )

    def test_bridge_references_provenance_subsystem(self):
        """The bridge call must reference 'provenance_tracker'."""
        src = _get_ucc_source()
        idx = src.find("provenance_adapt_thresholds_failure")
        assert idx >= 0
        context = src[max(0, idx - 200):idx + 200]
        assert "'provenance_tracker'" in context, (
            "provenance_tracker subsystem not in bridge call"
        )

    def test_bridge_receives_exception_variable(self):
        """The _prov_adapt_err variable must be passed."""
        src = _get_ucc_source()
        idx = src.find("provenance_adapt_thresholds_failure")
        assert idx >= 0
        context = src[max(0, idx - 100):idx + 200]
        assert "_prov_adapt_err" in context, (
            "_prov_adapt_err not passed to _bridge_silent_exception"
        )


# ══════════════════════════════════════════════════════════════════════
# F10: Feedback bus signals for UCC and post-pipeline verdicts
# ══════════════════════════════════════════════════════════════════════

class TestF10FeedbackBusVerdictSignals:
    """F10: UCC evaluation verdict and post-pipeline verdict must
    emit feedback_bus signals for cross-pass conditioning."""

    def test_ucc_verdict_pressure_in_reasoning_core(self):
        """_reasoning_core_impl must write ucc_verdict_pressure."""
        src = _get_reasoning_core_source()
        assert "ucc_verdict_pressure" in src, (
            "ucc_verdict_pressure signal not in _reasoning_core_impl"
        )

    def test_ucc_verdict_uses_write_signal(self):
        """UCC verdict must use feedback_bus.write_signal()."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_verdict_pressure")
        assert idx >= 0
        context = src[max(0, idx - 200):idx + 200]
        assert "write_signal" in context, (
            "write_signal not called for ucc_verdict_pressure"
        )

    def test_post_pipeline_verdict_pressure_in_forward_impl(self):
        """_forward_impl must write post_pipeline_verdict_pressure."""
        src = _get_forward_impl_source()
        assert "post_pipeline_verdict_pressure" in src, (
            "post_pipeline_verdict_pressure not in _forward_impl"
        )

    def test_post_pipeline_verdict_uses_write_signal(self):
        """Post-pipeline verdict must use feedback_bus.write_signal()."""
        src = _get_forward_impl_source()
        idx = src.find("post_pipeline_verdict_pressure")
        assert idx >= 0
        context = src[max(0, idx - 200):idx + 200]
        assert "write_signal" in context, (
            "write_signal not called for post_pipeline_verdict_pressure"
        )

    def test_runtime_feedback_bus_write_signal(self):
        """Feedback bus write_signal must accept new signal names."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # Write a test signal
        model.feedback_bus.write_signal('ucc_verdict_pressure', 0.42)
        # Verify it was written in _extra_signals
        assert 'ucc_verdict_pressure' in model.feedback_bus._extra_signals, (
            "write_signal did not store the signal in _extra_signals"
        )
        assert abs(model.feedback_bus._extra_signals['ucc_verdict_pressure'] - 0.42) < 1e-6

    def test_ucc_verdict_trigger_score_propagated(self):
        """UCC verdict must propagate the trigger_score value."""
        src = _get_reasoning_core_source()
        idx = src.find("ucc_verdict_pressure")
        assert idx >= 0
        context = src[max(0, idx - 300):idx + 200]
        assert "trigger_score" in context, (
            "trigger_score not used for ucc_verdict_pressure value"
        )


# ══════════════════════════════════════════════════════════════════════
# Integration: Cross-patch coherence tests
# ══════════════════════════════════════════════════════════════════════

class TestFSeriesIntegration:
    """Integration tests verifying that F-series patches create a
    coherent cognitive feedback loop."""

    def test_error_evolution_has_record_episode(self):
        """ErrorEvolutionTracker must have record_episode method."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model.error_evolution, 'record_episode'), (
            "error_evolution missing record_episode method"
        )

    def test_bridge_silent_exception_exists(self):
        """AEONDeltaV3 must have _bridge_silent_exception method."""
        assert hasattr(aeon.AEONDeltaV3, '_bridge_silent_exception'), (
            "_bridge_silent_exception method missing"
        )

    def test_bridge_records_both_error_and_trace(self):
        """_bridge_silent_exception must write to both error_evolution
        and causal_trace for full coverage."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        initial_episodes = len(model.error_evolution._episodes)
        initial_traces = len(model.causal_trace._entries)
        model._bridge_silent_exception(
            'test_error_class',
            'test_subsystem',
            RuntimeError("test"),
        )
        assert len(model.error_evolution._episodes) > initial_episodes
        assert len(model.causal_trace._entries) > initial_traces

    def test_provenance_tracker_has_record_dependency(self):
        """CausalProvenanceTracker must have record_dependency method."""
        tracker = aeon.CausalProvenanceTracker()
        assert hasattr(tracker, 'record_dependency'), (
            "CausalProvenanceTracker missing record_dependency method"
        )

    def test_feedback_bus_has_write_signal(self):
        """CognitiveFeedbackBus must have write_signal method."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model.feedback_bus, 'write_signal'), (
            "feedback_bus missing write_signal method"
        )

    def test_all_f_patches_present(self):
        """Verify all F-series patch markers are present in source."""
        src_rc = _get_reasoning_core_source()
        src_fi = _get_forward_impl_source()
        src_vc = _get_verify_coherence_source()
        src_vr = _get_verify_reinforce_source()
        src_ucc = _get_ucc_source()

        # F1: DAG revision bridge
        assert "dag_revision_consensus_failure" in src_rc
        # F2: DAG auto-critic bridge
        assert "dag_auto_critic_failure" in src_rc
        # F3: UCC verdict recording
        assert "ucc_evaluation_verdict" in src_rc
        # F4: Post-pipeline verdict recording
        assert "post_pipeline_metacog_verdict" in src_fi
        # F5: Verify-coherence verdict recording
        assert "verify_coherence_metacog_verdict" in src_vc
        # F6: Unity passed recording
        assert "cognitive_unity_check" in src_vr
        # F7: Unity exception bridge
        assert (
            "_bridge_silent_exception" in src_vr
            and "cognitive_unity_meta_evaluation_failure" in src_vr
        )
        # F8: Provenance dependencies
        assert "record_dependency" in src_ucc
        # F9: Provenance adapt bridge
        assert "provenance_adapt_thresholds_failure" in src_ucc
        # F10a: UCC verdict → feedback bus
        assert "ucc_verdict_pressure" in src_rc
        # F10b: Post-pipeline verdict → feedback bus
        assert "post_pipeline_verdict_pressure" in src_fi
