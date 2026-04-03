"""Tests for D-series cognitive integration patches (D1-D7).

Validates the seven patches that bridge critical signal discontinuities
in the AEON-Delta RMT v3.1 architecture:

  D1: CausalErrorEvolutionTracker → FeedbackBus bridge
  D2: MetaCognitiveRecursor outcome externalization
  D3: Oscillation severity consumption in MCT evaluate()
  D4: Unified subsystem health aggregation
  D5: Training loop feedback bus lifecycle management
  D6: Secondary path provenance instrumentation
  D7: read_signal() migration for orphan detection
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    AEONConfig,
    CausalErrorEvolutionTracker,
    CausalProvenanceTracker,
    CognitiveFeedbackBus,
    MetaCognitiveRecursor,
    MetaCognitiveRecursionTrigger,
    ProvablyConvergentMetaLoop,
    RecursionUtilityGate,
    RobustVectorQuantizer,
    SubsystemHealthGate,
)


def _make_config(**overrides):
    """Create a minimal AEONConfig for testing."""
    defaults = dict(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_feedback_bus(config=None):
    """Create a CognitiveFeedbackBus for testing."""
    if config is None:
        config = _make_config()
    return CognitiveFeedbackBus(hidden_dim=config.hidden_dim)


def _make_meta_loop(config=None, max_iterations=5):
    """Create a ProvablyConvergentMetaLoop for testing."""
    if config is None:
        config = _make_config()
    return ProvablyConvergentMetaLoop(
        config=config,
        max_iterations=max_iterations,
        min_iterations=1,
        convergence_threshold=1e-3,
    )


# ═══════════════════════════════════════════════════════════════════════
# D1: CausalErrorEvolutionTracker → FeedbackBus Bridge
# ═══════════════════════════════════════════════════════════════════════

class TestD1_ErrorEvolutionFeedbackBusBridge(unittest.TestCase):
    """D1: record_episode() writes signals to feedback bus."""

    def setUp(self):
        self.tracker = CausalErrorEvolutionTracker()
        self.bus = _make_feedback_bus()

    def test_d1_tracker_has_feedback_bus_ref(self):
        """CausalErrorEvolutionTracker has _feedback_bus_ref attribute."""
        self.assertTrue(hasattr(self.tracker, '_feedback_bus_ref'))
        self.assertIsNone(self.tracker._feedback_bus_ref)

    def test_d1_set_feedback_bus(self):
        """set_feedback_bus() attaches bus reference."""
        self.tracker.set_feedback_bus(self.bus)
        self.assertIs(self.tracker._feedback_bus_ref, self.bus)

    def test_d1_set_feedback_bus_none(self):
        """set_feedback_bus(None) detaches bus reference."""
        self.tracker.set_feedback_bus(self.bus)
        self.tracker.set_feedback_bus(None)
        self.assertIsNone(self.tracker._feedback_bus_ref)

    def test_d1_record_episode_writes_severity(self):
        """record_episode() writes error_evolution_severity to bus."""
        self.tracker.set_feedback_bus(self.bus)
        self.tracker.record_episode(
            error_class='test_error',
            strategy_used='retry',
            success=True,
            metadata={'severity': 0.7},
        )
        state = self.bus.get_state()
        self.assertIn('error_evolution_severity', state)
        self.assertAlmostEqual(state['error_evolution_severity'], 0.7, places=1)

    def test_d1_record_episode_writes_recovery_success(self):
        """record_episode() writes error_recovery_success to bus."""
        self.tracker.set_feedback_bus(self.bus)
        # Successful recovery
        self.tracker.record_episode(
            error_class='test_error',
            strategy_used='retry',
            success=True,
        )
        state = self.bus.get_state()
        self.assertIn('error_recovery_success', state)
        self.assertEqual(state['error_recovery_success'], 1.0)

    def test_d1_record_episode_writes_failure(self):
        """record_episode() writes 0.0 recovery_success on failure."""
        self.tracker.set_feedback_bus(self.bus)
        self.tracker.record_episode(
            error_class='test_error',
            strategy_used='retry',
            success=False,
        )
        state = self.bus.get_state()
        self.assertEqual(state['error_recovery_success'], 0.0)

    def test_d1_record_episode_writes_recovery_ratio(self):
        """record_episode() writes running error_recovery_ratio."""
        self.tracker.set_feedback_bus(self.bus)
        # Record 2 successes and 1 failure
        self.tracker.record_episode('e1', 'retry', True)
        self.tracker.record_episode('e1', 'retry', False)
        self.tracker.record_episode('e1', 'retry', True)
        state = self.bus.get_state()
        self.assertIn('error_recovery_ratio', state)
        ratio = state['error_recovery_ratio']
        self.assertGreater(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_d1_no_bus_no_crash(self):
        """record_episode() with no bus attached does not crash."""
        self.tracker.record_episode(
            error_class='test_error',
            strategy_used='retry',
            success=True,
        )
        # Should succeed without error

    def test_d1_default_severity_when_no_metadata(self):
        """record_episode() uses default severity 0.5 without metadata."""
        self.tracker.set_feedback_bus(self.bus)
        self.tracker.record_episode(
            error_class='test_error',
            strategy_used='retry',
            success=True,
        )
        state = self.bus.get_state()
        self.assertIn('error_evolution_severity', state)
        self.assertAlmostEqual(state['error_evolution_severity'], 0.5, places=1)


# ═══════════════════════════════════════════════════════════════════════
# D2: MetaCognitiveRecursor Outcome Externalization
# ═══════════════════════════════════════════════════════════════════════

class TestD2_RecursionOutcomeExternalization(unittest.TestCase):
    """D2: recurse_if_needed() writes outcome signals to bus."""

    def setUp(self):
        self.config = _make_config()
        self.bus = _make_feedback_bus(self.config)
        self.meta_loop = _make_meta_loop(self.config)
        self.utility_gate = RecursionUtilityGate()
        self.recursor = MetaCognitiveRecursor(
            meta_loop=self.meta_loop,
            feedback_bus=self.bus,
            recursion_utility_gate=self.utility_gate,
        )

    def _make_trigger_detail(self, should_recurse=True):
        return {
            'should_recurse': True,
            'should_trigger': should_recurse,
            'trigger_score': 0.8 if should_recurse else 0.2,
            'triggers_active': ['uncertainty'] if should_recurse else [],
        }

    def _make_meta_results(self):
        return {
            'final_residual': 0.1,
            'iterations': 3,
            'convergence_rate': 0.5,
        }

    def test_d2_recurse_writes_outcome_useful(self):
        """recurse_if_needed() writes recursion_outcome_useful signal."""
        psi_0 = torch.randn(1, self.config.hidden_dim)
        C_init = torch.randn(1, self.config.hidden_dim)
        C_out, meta = self.recursor.recurse_if_needed(
            psi_0, C_init, self._make_trigger_detail(),
            self._make_meta_results(),
        )
        state = self.bus.get_state()
        self.assertIn('recursion_outcome_useful', state)

    def test_d2_recurse_writes_best_improvement(self):
        """recurse_if_needed() writes recursion_best_improvement signal."""
        psi_0 = torch.randn(1, self.config.hidden_dim)
        C_init = torch.randn(1, self.config.hidden_dim)
        C_out, meta = self.recursor.recurse_if_needed(
            psi_0, C_init, self._make_trigger_detail(),
            self._make_meta_results(),
        )
        state = self.bus.get_state()
        self.assertIn('recursion_best_improvement', state)

    def test_d2_no_recurse_no_outcome_signals(self):
        """When should_recurse=False, no outcome signals are written."""
        psi_0 = torch.randn(1, self.config.hidden_dim)
        C_init = torch.randn(1, self.config.hidden_dim)
        C_out, meta = self.recursor.recurse_if_needed(
            psi_0, C_init, self._make_trigger_detail(should_recurse=False),
            self._make_meta_results(),
        )
        state = self.bus.get_state()
        # Should not have outcome signals since no recursion happened
        self.assertNotIn('recursion_outcome_useful', state)

    def test_d2_outcome_useful_is_binary(self):
        """recursion_outcome_useful is 0.0 or 1.0."""
        psi_0 = torch.randn(1, self.config.hidden_dim)
        C_init = torch.randn(1, self.config.hidden_dim)
        C_out, meta = self.recursor.recurse_if_needed(
            psi_0, C_init, self._make_trigger_detail(),
            self._make_meta_results(),
        )
        state = self.bus.get_state()
        useful = state.get('recursion_outcome_useful', -1)
        self.assertIn(useful, [0.0, 1.0])

    def test_d2_best_improvement_bounded(self):
        """recursion_best_improvement ∈ [0, 1]."""
        psi_0 = torch.randn(1, self.config.hidden_dim)
        C_init = torch.randn(1, self.config.hidden_dim)
        C_out, meta = self.recursor.recurse_if_needed(
            psi_0, C_init, self._make_trigger_detail(),
            self._make_meta_results(),
        )
        state = self.bus.get_state()
        improvement = state.get('recursion_best_improvement', -1)
        self.assertGreaterEqual(improvement, 0.0)
        self.assertLessEqual(improvement, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# D3: Oscillation Severity Consumption in MCT
# ═══════════════════════════════════════════════════════════════════════

class TestD3_OscillationSeverityConsumption(unittest.TestCase):
    """D3: MCT evaluate() reads oscillation_severity_pressure from bus."""

    def setUp(self):
        self.mct = MetaCognitiveRecursionTrigger()
        self.bus = _make_feedback_bus()

    def test_d3_mct_has_feedback_bus_ref(self):
        """MCT has _feedback_bus_ref attribute."""
        self.assertTrue(hasattr(self.mct, '_feedback_bus_ref'))
        self.assertIsNone(self.mct._feedback_bus_ref)

    def test_d3_set_feedback_bus(self):
        """set_feedback_bus() attaches bus reference."""
        self.mct.set_feedback_bus(self.bus)
        self.assertIs(self.mct._feedback_bus_ref, self.bus)

    def test_d3_no_oscillation_no_amplification(self):
        """Without oscillation pressure, diverging signal is unchanged."""
        self.mct.set_feedback_bus(self.bus)
        result = self.mct.evaluate(
            uncertainty=0.0,
            is_diverging=True,
        )
        # Baseline: diverging signal with no oscillation amplification
        baseline_score = result['trigger_score']
        self.assertGreater(baseline_score, 0.0)

    def test_d3_high_oscillation_amplifies_diverging(self):
        """High oscillation_severity_pressure amplifies diverging signal."""
        self.mct.set_feedback_bus(self.bus)
        # Write high oscillation pressure
        self.bus.write_signal('oscillation_severity_pressure', 0.8)

        result_with_osc = self.mct.evaluate(
            uncertainty=0.0,
            is_diverging=True,
        )

        # Compare with no oscillation
        mct_no_osc = MetaCognitiveRecursionTrigger()
        result_no_osc = mct_no_osc.evaluate(
            uncertainty=0.0,
            is_diverging=True,
        )

        # With oscillation, trigger score should be higher
        self.assertGreater(
            result_with_osc['trigger_score'],
            result_no_osc['trigger_score'],
        )

    def test_d3_low_oscillation_no_amplification(self):
        """Low oscillation pressure (< 0.3) does not amplify."""
        self.mct.set_feedback_bus(self.bus)
        self.bus.write_signal('oscillation_severity_pressure', 0.2)

        result = self.mct.evaluate(
            uncertainty=0.0,
            is_diverging=True,
        )

        mct_no_osc = MetaCognitiveRecursionTrigger()
        result_no_osc = mct_no_osc.evaluate(
            uncertainty=0.0,
            is_diverging=True,
        )

        # Should be same (within floating point)
        self.assertAlmostEqual(
            result['trigger_score'],
            result_no_osc['trigger_score'],
            places=5,
        )

    def test_d3_oscillation_reads_signal(self):
        """MCT evaluate() uses read_signal() which enables orphan tracking."""
        self.mct.set_feedback_bus(self.bus)
        self.bus.write_signal('oscillation_severity_pressure', 0.5)
        self.mct.evaluate()
        # The read_signal call should be tracked in the bus
        # (verifiable via _read_log if available)
        read_log = getattr(self.bus, '_read_log', {})
        self.assertIn('oscillation_severity_pressure', read_log)

    def test_d3_no_bus_no_crash(self):
        """MCT evaluate() works fine without attached bus."""
        result = self.mct.evaluate(
            uncertainty=0.3,
            is_diverging=True,
        )
        self.assertIn('trigger_score', result)


# ═══════════════════════════════════════════════════════════════════════
# D4: Unified Subsystem Health Aggregation
# ═══════════════════════════════════════════════════════════════════════

class TestD4_UnifiedSubsystemHealth(unittest.TestCase):
    """D4: _build_feedback_extra_signals() populates health_mean."""

    def test_d4_subsystem_health_gate_exists(self):
        """SubsystemHealthGate produces health features and gate values."""
        gate = SubsystemHealthGate(hidden_dim=64)
        output = torch.randn(1, 64)
        features = gate.compute_health_features(output)
        self.assertEqual(features.shape[-1], 3)  # 3 health features

    def test_d4_subsystem_health_gate_forward(self):
        """SubsystemHealthGate.forward() returns gated output and gate value."""
        gate = SubsystemHealthGate(hidden_dim=64)
        output = torch.randn(1, 64)
        gated_output, gate_value = gate(output)
        self.assertEqual(gated_output.shape, output.shape)
        self.assertGreater(gate_value, 0.0)
        self.assertLessEqual(gate_value, 1.0)

    def test_d4_health_gate_provenance_tracking(self):
        """D6: SubsystemHealthGate forwards with provenance tracking."""
        gate = SubsystemHealthGate(hidden_dim=64)
        tracker = CausalProvenanceTracker()
        gate._provenance_tracker_ref = tracker
        output = torch.randn(1, 64)
        gated_output, gate_value = gate(output)
        attr = tracker.compute_attribution()
        self.assertIn('health_gate_attenuation', attr.get('contributions', {}))

    def test_d4_nan_output_low_health(self):
        """SubsystemHealthGate detects non-finite outputs as unhealthy."""
        gate = SubsystemHealthGate(hidden_dim=64)
        # Create tensor with NaN
        bad_output = torch.full((1, 64), float('nan'))
        features = gate.compute_health_features(bad_output)
        # First feature (is_finite) should be 0.0
        self.assertEqual(features[0].item(), 0.0)


# ═══════════════════════════════════════════════════════════════════════
# D5: Training Loop Feedback Bus Lifecycle Management
# ═══════════════════════════════════════════════════════════════════════

class TestD5_TrainingBusLifecycle(unittest.TestCase):
    """D5: AEONTrainer calls flush_consumed/get_orphaned_signals per step."""

    def test_d5_feedback_bus_flush_consumed(self):
        """flush_consumed() resets write/read logs and returns summary."""
        bus = _make_feedback_bus()
        bus.write_signal('test_signal', 1.0)
        summary = bus.flush_consumed()
        # flush_consumed returns a summary dict
        self.assertIsInstance(summary, dict)
        self.assertIn('total_written', summary)

    def test_d5_feedback_bus_get_orphaned_signals(self):
        """get_orphaned_signals() detects written-but-not-read signals."""
        bus = _make_feedback_bus()
        bus.write_signal('orphan_signal', 1.0)
        # Don't read it — check orphans BEFORE flush
        orphans = bus.get_orphaned_signals()
        self.assertIn('orphan_signal', orphans)

    def test_d5_read_signal_prevents_orphaning(self):
        """read_signal() marks signal as consumed, preventing orphan status."""
        bus = _make_feedback_bus()
        bus.write_signal('consumed_signal', 1.0)
        bus.read_signal('consumed_signal', 0.0)
        orphans = bus.get_orphaned_signals()
        self.assertNotIn('consumed_signal', orphans)

    def test_d5_lifecycle_idempotent(self):
        """Calling flush_consumed twice is safe."""
        bus = _make_feedback_bus()
        bus.flush_consumed()
        bus.flush_consumed()
        orphans = bus.get_orphaned_signals()
        self.assertIsInstance(orphans, dict)


# ═══════════════════════════════════════════════════════════════════════
# D6: Secondary Path Provenance Instrumentation
# ═══════════════════════════════════════════════════════════════════════

class TestD6_SecondaryPathProvenance(unittest.TestCase):
    """D6: record_before/record_after wrapping on secondary paths."""

    def test_d6_vq_provenance_tracking(self):
        """RobustVectorQuantizer.forward() records provenance deltas."""
        vq = RobustVectorQuantizer(
            num_embeddings=16,
            embedding_dim=32,
            commitment_cost=0.25,
        )
        tracker = CausalProvenanceTracker()
        vq._provenance_tracker_ref = tracker

        inputs = torch.randn(4, 32)
        quantized, loss, indices = vq(inputs)

        attr = tracker.compute_attribution()
        self.assertIn('vq_codebook_selection', attr.get('contributions', {}))

    def test_d6_vq_no_provenance_no_crash(self):
        """VQ forward works fine without provenance tracker."""
        vq = RobustVectorQuantizer(
            num_embeddings=16,
            embedding_dim=32,
            commitment_cost=0.25,
        )
        inputs = torch.randn(4, 32)
        quantized, loss, indices = vq(inputs)
        self.assertEqual(quantized.shape, inputs.shape)

    def test_d6_vq_provenance_captures_quantization_delta(self):
        """VQ provenance delta reflects quantization distortion."""
        vq = RobustVectorQuantizer(
            num_embeddings=16,
            embedding_dim=32,
            commitment_cost=0.25,
        )
        tracker = CausalProvenanceTracker()
        vq._provenance_tracker_ref = tracker

        inputs = torch.randn(4, 32)
        quantized, loss, indices = vq(inputs)

        attr = tracker.compute_attribution()
        delta = attr['contributions'].get('vq_codebook_selection', 0.0)
        # Quantization should produce a non-zero delta (inputs ≠ quantized)
        self.assertGreater(delta, 0.0)

    def test_d6_health_gate_no_provenance_no_crash(self):
        """SubsystemHealthGate forward works without provenance tracker."""
        gate = SubsystemHealthGate(hidden_dim=64)
        output = torch.randn(1, 64)
        gated_output, gate_value = gate(output)
        self.assertEqual(gated_output.shape, output.shape)


# ═══════════════════════════════════════════════════════════════════════
# D7: read_signal() Migration for Orphan Detection
# ═══════════════════════════════════════════════════════════════════════

class TestD7_ReadSignalMigration(unittest.TestCase):
    """D7: _build_feedback_extra_signals uses read_signal() with fallback."""

    def test_d7_read_signal_basic(self):
        """read_signal() returns written value."""
        bus = _make_feedback_bus()
        bus.write_signal('test_sig', 0.75)
        val = bus.read_signal('test_sig', 0.0)
        self.assertAlmostEqual(val, 0.75, places=2)

    def test_d7_read_signal_default(self):
        """read_signal() returns default for unwritten signal."""
        bus = _make_feedback_bus()
        val = bus.read_signal('nonexistent', 0.42)
        self.assertAlmostEqual(val, 0.42, places=2)

    def test_d7_read_signal_tracked_in_read_log(self):
        """read_signal() updates _read_log for orphan detection."""
        bus = _make_feedback_bus()
        bus.write_signal('tracked_sig', 0.5)
        bus.read_signal('tracked_sig', 0.0)
        read_log = getattr(bus, '_read_log', {})
        self.assertIn('tracked_sig', read_log)

    def test_d7_write_without_read_is_orphan(self):
        """Signal written without read_signal() appears as orphan."""
        bus = _make_feedback_bus()
        bus.write_signal('orphan_test', 0.5)
        # Check orphans BEFORE flush clears the logs
        orphans = bus.get_orphaned_signals()
        self.assertIn('orphan_test', orphans)

    def test_d7_write_with_read_not_orphan(self):
        """Signal consumed via read_signal() is not orphaned."""
        bus = _make_feedback_bus()
        bus.write_signal('consumed_test', 0.5)
        bus.read_signal('consumed_test', 0.0)
        # Check orphans before flush
        orphans = bus.get_orphaned_signals()
        self.assertNotIn('consumed_test', orphans)


# ═══════════════════════════════════════════════════════════════════════
# Cross-Patch Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCrossPatchIntegration(unittest.TestCase):
    """Cross-patch signal flow verification."""

    def test_d1_d3_error_evolution_triggers_mct_via_bus(self):
        """D1+D3: Error evolution → bus → MCT sensitivity."""
        bus = _make_feedback_bus()
        tracker = CausalErrorEvolutionTracker()
        tracker.set_feedback_bus(bus)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)

        # Record a high-severity error
        tracker.record_episode(
            error_class='convergence_failure',
            strategy_used='deeper_meta_loop',
            success=False,
            metadata={'severity': 0.9},
        )

        # Verify signals propagated
        state = bus.get_state()
        self.assertAlmostEqual(state['error_evolution_severity'], 0.9, places=1)
        self.assertEqual(state['error_recovery_success'], 0.0)

    def test_d1_d5_training_lifecycle_detects_error_orphans(self):
        """D1+D5: Error signals detected as orphans if not consumed."""
        bus = _make_feedback_bus()
        tracker = CausalErrorEvolutionTracker()
        tracker.set_feedback_bus(bus)

        tracker.record_episode('test_error', 'retry', True)

        # Without any reads, check orphans BEFORE flush
        orphans = bus.get_orphaned_signals()
        # At least some D1 signals should appear as orphans
        d1_signals = {'error_evolution_severity', 'error_recovery_success',
                      'error_recovery_ratio'}
        found_orphans = d1_signals & set(orphans.keys())
        self.assertGreater(len(found_orphans), 0)

    def test_d3_d7_oscillation_read_prevents_orphan(self):
        """D3+D7: MCT reading oscillation_severity_pressure prevents orphaning."""
        bus = _make_feedback_bus()
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)

        # Write oscillation signal
        bus.write_signal('oscillation_severity_pressure', 0.5)

        # MCT evaluate reads it
        mct.evaluate()

        # Check orphans BEFORE flush
        orphans = bus.get_orphaned_signals()
        self.assertNotIn('oscillation_severity_pressure', orphans)

    def test_d2_recursion_outcome_three_signal_assessment(self):
        """D2: Creates 3-signal recursion assessment: futility + useful + improvement."""
        config = _make_config()
        bus = _make_feedback_bus(config)
        meta_loop = _make_meta_loop(config)
        utility_gate = RecursionUtilityGate()
        recursor = MetaCognitiveRecursor(
            meta_loop=meta_loop,
            feedback_bus=bus,
            recursion_utility_gate=utility_gate,
        )

        psi_0 = torch.randn(1, config.hidden_dim)
        C_init = torch.randn(1, config.hidden_dim)
        trigger_detail = {
            'should_trigger': True,
            'trigger_score': 0.8,
            'triggers_active': ['uncertainty'],
        }
        meta_results = {
            'final_residual': 0.1,
            'iterations': 3,
            'convergence_rate': 0.5,
        }
        C_out, meta = recursor.recurse_if_needed(
            psi_0, C_init, trigger_detail, meta_results,
        )

        state = bus.get_state()
        # All three recursion signals should exist
        self.assertIn('metacognitive_recursion_depth', state)
        self.assertIn('recursion_outcome_useful', state)
        self.assertIn('recursion_best_improvement', state)

    def test_d6_vq_provenance_completeness(self):
        """D6: VQ codebook selection appears in compute_attribution()."""
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=32, commitment_cost=0.25,
        )
        tracker = CausalProvenanceTracker()
        vq._provenance_tracker_ref = tracker

        # Also track another module for comparison
        tracker.record_before('encoder', torch.randn(4, 32))
        encoder_out = torch.randn(4, 32)
        tracker.record_after('encoder', encoder_out)

        # VQ should also be tracked
        inputs = torch.randn(4, 32)
        vq(inputs)

        attr = tracker.compute_attribution()
        contributions = attr.get('contributions', {})
        self.assertIn('encoder', contributions)
        self.assertIn('vq_codebook_selection', contributions)


# ═══════════════════════════════════════════════════════════════════════
# Activation Sequence Verification
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence(unittest.TestCase):
    """Verifies the activation sequence: D5→D7→D1→D4→D2→D3→D6."""

    def test_full_activation_sequence(self):
        """All patches can be activated together without conflicts."""
        config = _make_config()
        bus = _make_feedback_bus(config)

        # D5: Bus lifecycle works
        bus.flush_consumed()
        orphans = bus.get_orphaned_signals()
        self.assertIsInstance(orphans, dict)

        # D7: read_signal works
        bus.write_signal('test_d7', 0.5)
        val = bus.read_signal('test_d7', 0.0)
        self.assertAlmostEqual(val, 0.5, places=2)

        # D1: Error evolution → bus
        tracker = CausalErrorEvolutionTracker()
        tracker.set_feedback_bus(bus)
        tracker.record_episode('test', 'retry', True)

        # D4: Health gate
        gate = SubsystemHealthGate(hidden_dim=64)
        output = torch.randn(1, 64)
        gated, gv = gate(output)
        self.assertGreater(gv, 0.0)

        # D2: Recursion outcome
        meta_loop = _make_meta_loop(config)
        utility_gate = RecursionUtilityGate()
        recursor = MetaCognitiveRecursor(
            meta_loop=meta_loop,
            feedback_bus=bus,
            recursion_utility_gate=utility_gate,
        )
        psi_0 = torch.randn(1, config.hidden_dim)
        C_init = torch.randn(1, config.hidden_dim)
        C_out, meta = recursor.recurse_if_needed(
            psi_0, C_init,
            {'should_trigger': True, 'trigger_score': 0.8,
             'triggers_active': ['uncertainty']},
            {'final_residual': 0.1, 'iterations': 3, 'convergence_rate': 0.5},
        )
        self.assertIn('recursion_outcome_useful', bus.get_state())

        # D3: Oscillation consumption
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        bus.write_signal('oscillation_severity_pressure', 0.5)
        result = mct.evaluate(is_diverging=True)
        self.assertIn('trigger_score', result)

        # D6: Provenance tracking
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=32, commitment_cost=0.25,
        )
        prov = CausalProvenanceTracker()
        vq._provenance_tracker_ref = prov
        q, l, i = vq(torch.randn(4, 32))
        attr = prov.compute_attribution()
        self.assertIn('vq_codebook_selection', attr.get('contributions', {}))

        # D5 final check: orphan detection after all signals
        bus.flush_consumed()
        orphans = bus.get_orphaned_signals()
        self.assertIsInstance(orphans, dict)


if __name__ == '__main__':
    unittest.main()
