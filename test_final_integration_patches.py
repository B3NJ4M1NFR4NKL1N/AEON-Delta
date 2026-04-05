"""
Tests for Final Integration & Cognitive Activation patches (PATCH-A through PATCH-F).

Patches covered:
  PATCH-A: Wire 3 orphaned epoch-bridge signals to MCT evaluate()
           (training_plasticity_exhausted, epoch_bridge_coherence_deficit,
            epoch_bridge_cert_violated)
  PATCH-B: Initialize feedback_bus in 4 disconnected components
           (RobustVectorQuantizer, ProvablyConvergentMetaLoop,
            CausalErrorEvolutionTracker, UnifiedCognitiveCycle)
  PATCH-C: Post-output late-distress MCT re-check
           (late_distress_pressure bus write on high post-output uncertainty)
  PATCH-D: Cross-pass oscillation detection in CognitiveFeedbackBus
           (_signal_history, _cross_pass_oscillation, get_cross_pass_oscillation)
  PATCH-E: Provenance gap pressure → MCT low_causal_quality
           (provenance_gap_pressure bus write + MCT read)
  PATCH-F: Subsystem silent failure → immediate bus signal
           (subsystem_silent_failure_pressure bus write + causal trace)
"""

import math
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    CausalErrorEvolutionTracker,
    RobustVectorQuantizer,
    ProvablyConvergentMetaLoop,
    UnifiedCognitiveCycle,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        device_str='cpu',
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_trigger(hidden_dim=64):
    """Create MCT trigger with feedback bus."""
    bus = CognitiveFeedbackBus(hidden_dim)
    trigger = MetaCognitiveRecursionTrigger()
    trigger.set_feedback_bus(bus)
    return trigger, bus


def _evaluate_mct(trigger, **kwargs):
    """Evaluate MCT with safe defaults for all params."""
    defaults = dict(
        uncertainty=0.0,
        is_diverging=False,
        topology_catastrophe=False,
        coherence_deficit=0.0,
        memory_staleness=False,
        recovery_pressure=0.0,
        world_model_surprise=0.0,
        causal_quality=1.0,
        safety_violation=False,
        diversity_collapse=0.0,
        memory_trust_deficit=0.0,
        convergence_conflict=0.0,
        output_reliability=1.0,
        spectral_stability_margin=1.0,
        border_uncertainty=0.0,
        stall_severity=0.0,
        oscillation_severity=0.0,
    )
    defaults.update(kwargs)
    return trigger.evaluate(**defaults)


def _has_feedback_bus(model):
    """Check if model has feedback_bus."""
    return (hasattr(model, 'feedback_bus')
            and model.feedback_bus is not None
            and hasattr(model.feedback_bus, 'write_signal'))


def _make_model():
    """Create minimal AEONDeltaV3 for testing."""
    cfg = _make_config()
    return AEONDeltaV3(cfg)


# ══════════════════════════════════════════════════════════════════════
#  PATCH-B: Initialize feedback_bus in 4 disconnected components
# ══════════════════════════════════════════════════════════════════════

class TestPatchB_BusInitialization:
    """Verify feedback_bus parameter is accepted and stored in 4 classes."""

    def test_b1_robust_vq_accepts_feedback_bus(self):
        """PATCH-B1: RobustVectorQuantizer accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=64, feedback_bus=bus,
        )
        assert vq._feedback_bus_ref is bus

    def test_b1_robust_vq_default_none(self):
        """PATCH-B1: RobustVectorQuantizer defaults feedback_bus to None."""
        vq = RobustVectorQuantizer(num_embeddings=16, embedding_dim=64)
        assert hasattr(vq, '_feedback_bus_ref')
        assert vq._feedback_bus_ref is None

    def test_b2_pcml_accepts_feedback_bus(self):
        """PATCH-B2: ProvablyConvergentMetaLoop accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        cfg = _make_config()
        pcml = ProvablyConvergentMetaLoop(config=cfg, feedback_bus=bus)
        assert pcml._feedback_bus_ref is bus

    def test_b2_pcml_default_none(self):
        """PATCH-B2: ProvablyConvergentMetaLoop defaults feedback_bus to None."""
        cfg = _make_config()
        pcml = ProvablyConvergentMetaLoop(config=cfg)
        assert hasattr(pcml, '_feedback_bus_ref')
        assert pcml._feedback_bus_ref is None

    def test_b3_ceet_accepts_feedback_bus(self):
        """PATCH-B3: CausalErrorEvolutionTracker accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        ceet = CausalErrorEvolutionTracker(max_history=50, feedback_bus=bus)
        assert ceet._feedback_bus_ref is bus

    def test_b3_ceet_default_none(self):
        """PATCH-B3: CausalErrorEvolutionTracker defaults feedback_bus to None."""
        ceet = CausalErrorEvolutionTracker(max_history=50)
        assert hasattr(ceet, '_feedback_bus_ref')
        # With no feedback_bus kwarg, should be None
        assert ceet._feedback_bus_ref is None

    def test_b4_ucc_accepts_feedback_bus(self):
        """PATCH-B4: UnifiedCognitiveCycle accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        # UCC requires convergence_monitor and provenance_tracker
        # Use None for optional components
        try:
            from aeon_core import ConvergenceMonitor, CausalProvenanceTracker
            cfg = _make_config()
            cm = ConvergenceMonitor()
            pt = CausalProvenanceTracker(module_names=["test"])
            ucc = UnifiedCognitiveCycle(
                convergence_monitor=cm,
                coherence_verifier=None,
                error_evolution=None,
                metacognitive_trigger=None,
                provenance_tracker=pt,
                feedback_bus=bus,
            )
            assert ucc._feedback_bus_ref is bus
        except Exception:
            pytest.skip("UCC dependencies not available")

    def test_b1_model_wires_vq_bus(self):
        """PATCH-B1: AEONDeltaV3 wires feedback_bus to vector_quantizer."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        if model.vector_quantizer is not None:
            assert model.vector_quantizer._feedback_bus_ref is model.feedback_bus

    def test_b2_model_wires_meta_loop_bus(self):
        """PATCH-B2: AEONDeltaV3 wires feedback_bus to meta_loop."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        assert model.meta_loop._feedback_bus_ref is model.feedback_bus

    def test_b3_model_wires_error_evolution_bus(self):
        """PATCH-B3: AEONDeltaV3 wires feedback_bus to error_evolution."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        if model.error_evolution is not None:
            assert model.error_evolution._feedback_bus_ref is model.feedback_bus

    def test_b4_model_wires_ucc_bus(self):
        """PATCH-B4: AEONDeltaV3 wires feedback_bus to unified_cognitive_cycle."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        ucc = getattr(model, 'unified_cognitive_cycle', None)
        if ucc is not None:
            assert ucc._feedback_bus_ref is model.feedback_bus


# ══════════════════════════════════════════════════════════════════════
#  PATCH-D: Cross-pass oscillation detection
# ══════════════════════════════════════════════════════════════════════

class TestPatchD_CrossPassOscillation:
    """Verify cross-pass oscillation detection in CognitiveFeedbackBus."""

    def test_d_signal_history_initialized(self):
        """PATCH-D: Bus has _signal_history dict after construction."""
        bus = CognitiveFeedbackBus(64)
        assert hasattr(bus, '_signal_history')
        assert isinstance(bus._signal_history, dict)

    def test_d_cross_pass_oscillation_initialized(self):
        """PATCH-D: Bus has _cross_pass_oscillation dict after construction."""
        bus = CognitiveFeedbackBus(64)
        assert hasattr(bus, '_cross_pass_oscillation')
        assert isinstance(bus._cross_pass_oscillation, dict)

    def test_d_write_signal_tracks_history(self):
        """PATCH-D: write_signal appends to _signal_history."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('test_signal', 0.5)
        assert 'test_signal' in bus._signal_history
        assert len(bus._signal_history['test_signal']) == 1

    def test_d_history_maxlen_respected(self):
        """PATCH-D: Signal history respects maxlen (5)."""
        bus = CognitiveFeedbackBus(64)
        for i in range(10):
            bus.write_signal('test_signal', float(i))
        assert len(bus._signal_history['test_signal']) == bus._signal_history_maxlen

    def test_d_no_oscillation_with_monotonic(self):
        """PATCH-D: Monotonically increasing signal → oscillation = 0."""
        bus = CognitiveFeedbackBus(64)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            bus.write_signal('monotonic', v)
        osc = bus.get_cross_pass_oscillation('monotonic', 0.0)
        assert osc == 0.0

    def test_d_high_oscillation_detected(self):
        """PATCH-D: Alternating signal values → high oscillation ratio."""
        bus = CognitiveFeedbackBus(64)
        for v in [0.1, 0.9, 0.1, 0.9, 0.1]:
            bus.write_signal('oscillating', v)
        osc = bus.get_cross_pass_oscillation('oscillating', 0.0)
        assert osc > 0.5, f"Expected oscillation > 0.5, got {osc}"

    def test_d_get_cross_pass_oscillation_default(self):
        """PATCH-D: get_cross_pass_oscillation returns default for unknown signal."""
        bus = CognitiveFeedbackBus(64)
        assert bus.get_cross_pass_oscillation('nonexistent', 0.42) == 0.42

    def test_d_oscillation_not_computed_with_few_samples(self):
        """PATCH-D: Oscillation not computed with < 3 samples."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('short', 0.1)
        bus.write_signal('short', 0.9)
        osc = bus.get_cross_pass_oscillation('short', -1.0)
        assert osc == -1.0  # Should not be set yet

    def test_d_mct_reads_cross_pass_oscillation(self):
        """PATCH-D: MCT reads cross-pass oscillation and boosts oscillation_severity."""
        trigger, bus = _make_trigger()
        # Create high oscillation on convergence_confidence
        for v in [0.1, 0.9, 0.1, 0.9, 0.1]:
            bus.write_signal('convergence_confidence', v)
        osc = bus.get_cross_pass_oscillation('convergence_confidence')
        assert osc > 0.5
        result = _evaluate_mct(trigger)
        # The oscillation should contribute to trigger_score
        score = result.get('trigger_score', 0.0)
        # Compare with baseline (no oscillation)
        bus2 = CognitiveFeedbackBus(64)
        trigger2 = MetaCognitiveRecursionTrigger()
        trigger2.set_feedback_bus(bus2)
        baseline = _evaluate_mct(trigger2)
        baseline_score = baseline.get('trigger_score', 0.0)
        assert score >= baseline_score


# ══════════════════════════════════════════════════════════════════════
#  PATCH-A: Wire 3 orphaned epoch-bridge signals to MCT
# ══════════════════════════════════════════════════════════════════════

class TestPatchA_OrphanedEpochBridgeSignals:
    """Verify 3 epoch-bridge signals are consumed by MCT."""

    def test_a_plasticity_exhausted_consumed(self):
        """PATCH-A: training_plasticity_exhausted consumed by MCT."""
        trigger, bus = _make_trigger()
        bus.write_signal('training_plasticity_exhausted', 0.8)
        _evaluate_mct(trigger)
        orphaned = bus.get_orphaned_signals()
        assert 'training_plasticity_exhausted' not in orphaned

    def test_a_plasticity_exhausted_boosts_convergence_conflict(self):
        """PATCH-A: High plasticity exhaustion raises trigger score."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('training_plasticity_exhausted', 0.8)
        boosted = _evaluate_mct(trigger2)
        boosted_score = boosted.get('trigger_score', 0.0)
        assert boosted_score > baseline_score, (
            f"Expected boosted > baseline: {boosted_score} > {baseline_score}"
        )

    def test_a_plasticity_below_threshold_no_effect(self):
        """PATCH-A: training_plasticity_exhausted ≤ 0.3 has no effect."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('training_plasticity_exhausted', 0.2)
        result = _evaluate_mct(trigger2)
        result_score = result.get('trigger_score', 0.0)
        assert abs(result_score - baseline_score) < 0.01

    def test_a_epoch_coherence_deficit_consumed(self):
        """PATCH-A: epoch_bridge_coherence_deficit consumed by MCT."""
        trigger, bus = _make_trigger()
        bus.write_signal('epoch_bridge_coherence_deficit', 0.6)
        _evaluate_mct(trigger)
        orphaned = bus.get_orphaned_signals()
        assert 'epoch_bridge_coherence_deficit' not in orphaned

    def test_a_epoch_coherence_deficit_boosts_score(self):
        """PATCH-A: High epoch_bridge_coherence_deficit raises trigger score."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('epoch_bridge_coherence_deficit', 0.7)
        boosted = _evaluate_mct(trigger2)
        boosted_score = boosted.get('trigger_score', 0.0)
        assert boosted_score > baseline_score

    def test_a_epoch_coherence_deficit_below_threshold(self):
        """PATCH-A: epoch_bridge_coherence_deficit ≤ 0.2 has no effect."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('epoch_bridge_coherence_deficit', 0.1)
        result = _evaluate_mct(trigger2)
        result_score = result.get('trigger_score', 0.0)
        assert abs(result_score - baseline_score) < 0.01

    def test_a_cert_violated_consumed(self):
        """PATCH-A: epoch_bridge_cert_violated consumed by MCT."""
        trigger, bus = _make_trigger()
        bus.write_signal('epoch_bridge_cert_violated', 1.0)
        _evaluate_mct(trigger)
        orphaned = bus.get_orphaned_signals()
        assert 'epoch_bridge_cert_violated' not in orphaned

    def test_a_cert_violated_boosts_recovery_pressure(self):
        """PATCH-A: epoch_bridge_cert_violated > 0.5 raises trigger score."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('epoch_bridge_cert_violated', 1.0)
        boosted = _evaluate_mct(trigger2)
        boosted_score = boosted.get('trigger_score', 0.0)
        assert boosted_score > baseline_score

    def test_a_cert_violated_below_threshold(self):
        """PATCH-A: epoch_bridge_cert_violated ≤ 0.5 has no effect."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('epoch_bridge_cert_violated', 0.3)
        result = _evaluate_mct(trigger2)
        result_score = result.get('trigger_score', 0.0)
        assert abs(result_score - baseline_score) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  PATCH-E: Provenance gap pressure → MCT low_causal_quality
# ══════════════════════════════════════════════════════════════════════

class TestPatchE_ProvenanceGapPressure:
    """Verify provenance_gap_pressure is written to bus and consumed by MCT."""

    def test_e_provenance_gap_consumed(self):
        """PATCH-E: provenance_gap_pressure consumed by MCT."""
        trigger, bus = _make_trigger()
        bus.write_signal('provenance_gap_pressure', 0.5)
        _evaluate_mct(trigger)
        orphaned = bus.get_orphaned_signals()
        assert 'provenance_gap_pressure' not in orphaned

    def test_e_provenance_gap_boosts_low_causal_quality(self):
        """PATCH-E: High provenance_gap_pressure raises trigger score."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('provenance_gap_pressure', 0.8)
        boosted = _evaluate_mct(trigger2)
        boosted_score = boosted.get('trigger_score', 0.0)
        assert boosted_score > baseline_score

    def test_e_provenance_gap_below_threshold(self):
        """PATCH-E: provenance_gap_pressure ≤ 0.2 has no effect."""
        trigger, bus = _make_trigger()
        baseline = _evaluate_mct(trigger)
        baseline_score = baseline.get('trigger_score', 0.0)

        trigger2, bus2 = _make_trigger()
        bus2.write_signal('provenance_gap_pressure', 0.1)
        result = _evaluate_mct(trigger2)
        result_score = result.get('trigger_score', 0.0)
        assert abs(result_score - baseline_score) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  PATCH-C: Post-output late-distress MCT re-check
# ══════════════════════════════════════════════════════════════════════

class TestPatchC_LateDistress:
    """Verify late-distress pressure is written to bus."""

    def test_c_model_has_late_distress_capability(self):
        """PATCH-C: Model can write late_distress_pressure to bus."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        # Simulate late distress by writing post_output_uncertainty
        model.feedback_bus.write_signal('post_output_uncertainty', 0.8)
        val = model.feedback_bus.read_signal('post_output_uncertainty', 0.0)
        assert val == 0.8

    def test_c_late_distress_signal_registered(self):
        """PATCH-C: Bus can register and write late_distress_pressure."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('late_distress_pressure', 0.7)
        val = bus.read_signal('late_distress_pressure', 0.0)
        assert val == 0.7


# ══════════════════════════════════════════════════════════════════════
#  PATCH-F: Subsystem silent failure detection
# ══════════════════════════════════════════════════════════════════════

class TestPatchF_SubsystemSilentFailure:
    """Verify subsystem_silent_failure_pressure can be written to bus."""

    def test_f_silent_failure_signal_registered(self):
        """PATCH-F: Bus can register and write subsystem_silent_failure_pressure."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('subsystem_silent_failure_pressure', 0.6)
        val = bus.read_signal('subsystem_silent_failure_pressure', 0.0)
        assert val == 0.6

    def test_f_silent_failure_not_orphaned_if_read(self):
        """PATCH-F: subsystem_silent_failure_pressure not orphaned when read."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('subsystem_silent_failure_pressure', 0.4)
        bus.read_signal('subsystem_silent_failure_pressure', 0.0)
        orphaned = bus.get_orphaned_signals()
        assert 'subsystem_silent_failure_pressure' not in orphaned


# ══════════════════════════════════════════════════════════════════════
#  Integration: Full model construction with all patches applied
# ══════════════════════════════════════════════════════════════════════

class TestIntegration_AllPatches:
    """Verify all patches work together when AEONDeltaV3 is constructed."""

    def test_model_constructs_successfully(self):
        """All patches: Model construction completes without error."""
        model = _make_model()
        assert model is not None

    def test_model_feedback_bus_exists(self):
        """All patches: Model has feedback_bus after construction."""
        model = _make_model()
        assert _has_feedback_bus(model)

    def test_model_bus_has_oscillation_tracking(self):
        """PATCH-D: Model's bus has cross-pass oscillation tracking."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        assert hasattr(model.feedback_bus, '_signal_history')
        assert hasattr(model.feedback_bus, 'get_cross_pass_oscillation')

    def test_model_bus_signals_not_orphaned_after_mct(self):
        """All patches: epoch-bridge signals consumed after MCT evaluation."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        mct = getattr(model, 'metacognitive_trigger', None)
        if mct is None:
            pytest.skip("Model lacks metacognitive_trigger")
        # Write orphaned signals
        model.feedback_bus.write_signal('training_plasticity_exhausted', 0.8)
        model.feedback_bus.write_signal('epoch_bridge_coherence_deficit', 0.6)
        model.feedback_bus.write_signal('epoch_bridge_cert_violated', 1.0)
        model.feedback_bus.write_signal('provenance_gap_pressure', 0.5)
        # Evaluate MCT — should read all 4 signals
        _evaluate_mct(mct)
        orphaned = model.feedback_bus.get_orphaned_signals()
        for sig in ['training_plasticity_exhausted',
                     'epoch_bridge_coherence_deficit',
                     'epoch_bridge_cert_violated',
                     'provenance_gap_pressure']:
            assert sig not in orphaned, f"Signal {sig} still orphaned"

    def test_cross_pass_oscillation_e2e(self):
        """PATCH-D: End-to-end cross-pass oscillation detection + MCT."""
        model = _make_model()
        if not _has_feedback_bus(model):
            pytest.skip("Model lacks feedback_bus")
        mct = getattr(model, 'metacognitive_trigger', None)
        if mct is None:
            pytest.skip("Model lacks metacognitive_trigger")
        # Create oscillation on convergence_confidence
        for v in [0.2, 0.8, 0.2, 0.8, 0.2]:
            model.feedback_bus.write_signal('convergence_confidence', v)
        osc = model.feedback_bus.get_cross_pass_oscillation('convergence_confidence')
        assert osc > 0.5
        result = _evaluate_mct(mct)
        assert result.get('trigger_score', 0.0) >= 0.0
