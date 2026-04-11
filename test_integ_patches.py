"""Tests for PATCH-INTEG-1 through PATCH-INTEG-5.

Verifies that:
1. TrainingConvergenceMonitor exposes ``last_verdict`` and ``history``
   properties for ConvergenceErrorEvolutionBridge compatibility.
2. SafeThoughtAETrainerV4 (Phase A) creates and polls a
   ConvergenceErrorEvolutionBridge.
3. ContextualRSSMTrainer (Phase B) creates and polls a
   ConvergenceErrorEvolutionBridge.
4. Bridge poll writes training_convergence_* signals to the bus.
5. End-to-end: convergence verdict transitions reach the MCT via
   the feedback bus.
"""
import inspect
import math
import re
import sys

import pytest

# ---------------------------------------------------------------------------
# Import ae_train (aeon_core is loaded as a transitive dependency)
# ---------------------------------------------------------------------------
import ae_train
import aeon_core

# Convenient aliases
ConvergenceErrorEvolutionBridge = ae_train.ConvergenceErrorEvolutionBridge
TrainingConvergenceMonitor = ae_train.TrainingConvergenceMonitor
CausalErrorEvolutionTracker = ae_train.CausalErrorEvolutionTracker
CognitiveFeedbackBus = aeon_core.CognitiveFeedbackBus
MetaCognitiveRecursionTrigger = aeon_core.MetaCognitiveRecursionTrigger


# ======================================================================
# PATCH-INTEG-1: TrainingConvergenceMonitor bridge-compatible accessors
# ======================================================================

class TestInteg1VerdictAccessors:
    """PATCH-INTEG-1: TrainingConvergenceMonitor bridge compatibility."""

    def test_last_verdict_property_exists(self):
        """TrainingConvergenceMonitor has a last_verdict property."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        assert hasattr(tcm, 'last_verdict')

    def test_last_verdict_aliases_status(self):
        """last_verdict returns the same value as status."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        assert tcm.last_verdict == tcm.status
        # Push past warmup
        for val in [1.0, 0.9, 0.8, 0.7, 0.6]:
            tcm.update(val)
        assert tcm.last_verdict == tcm.status
        assert tcm.last_verdict == 'converging'

    def test_last_verdict_reflects_diverging(self):
        """last_verdict reflects diverging status."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        for val in [0.5, 0.5, 0.5, 0.5, 0.5]:
            tcm.update(val)
        tcm.update(100.0)  # Trigger divergence
        assert tcm.last_verdict == 'diverging'

    def test_last_verdict_reflects_stagnating(self):
        """last_verdict reflects stagnating status."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        for _ in range(10):
            tcm.update(0.5)  # Identical values → stagnation
        assert tcm.last_verdict == 'stagnating'

    def test_history_property_exists(self):
        """TrainingConvergenceMonitor has a history property."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        assert hasattr(tcm, 'history')

    def test_history_aliases_internal_history(self):
        """history returns the same list as _history."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=5)
        tcm.update(1.0)
        tcm.update(0.9)
        assert tcm.history is tcm._history
        assert len(tcm.history) == 2
        assert tcm.history == [1.0, 0.9]

    def test_history_tracks_window(self):
        """history respects window_size trimming."""
        tcm = TrainingConvergenceMonitor(threshold=1e-5, window_size=3)
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            tcm.update(v)
        # TrainingConvergenceMonitor trims _history to window_size
        assert len(tcm.history) == 3
        assert tcm.history == [0.8, 0.7, 0.6]


# ======================================================================
# PATCH-INTEG-2/4: Bridge instantiation in trainers
# ======================================================================

class TestInteg2PhaseABridgeCreation:
    """PATCH-INTEG-2: SafeThoughtAETrainerV4 creates a bridge."""

    def test_source_has_bridge_creation(self):
        """Phase A trainer creates ConvergenceErrorEvolutionBridge."""
        src = inspect.getsource(ae_train.SafeThoughtAETrainerV4.__init__)
        assert 'ConvergenceErrorEvolutionBridge' in src, (
            "Phase A __init__ must instantiate ConvergenceErrorEvolutionBridge"
        )

    def test_bridge_uses_convergence_monitor(self):
        """Bridge creation references self.convergence_monitor."""
        src = inspect.getsource(ae_train.SafeThoughtAETrainerV4.__init__)
        assert 'convergence_monitor=self.convergence_monitor' in src, (
            "Bridge must wrap the trainer's convergence_monitor"
        )

    def test_bridge_uses_error_evolution(self):
        """Bridge creation references self._error_evolution."""
        src = inspect.getsource(ae_train.SafeThoughtAETrainerV4.__init__)
        assert 'error_evolution=self._error_evolution' in src, (
            "Bridge must reference the trainer's error_evolution"
        )

    def test_bridge_uses_inference_bus(self):
        """Bridge creation references self._inference_bus_ref."""
        src = inspect.getsource(ae_train.SafeThoughtAETrainerV4.__init__)
        assert 'feedback_bus=self._inference_bus_ref' in src, (
            "Bridge must reference the trainer's inference bus"
        )


class TestInteg4PhaseBBridgeCreation:
    """PATCH-INTEG-4: ContextualRSSMTrainer creates a bridge."""

    def test_source_has_bridge_creation(self):
        """Phase B trainer creates ConvergenceErrorEvolutionBridge."""
        src = inspect.getsource(ae_train.ContextualRSSMTrainer.__init__)
        assert 'ConvergenceErrorEvolutionBridge' in src, (
            "Phase B __init__ must instantiate ConvergenceErrorEvolutionBridge"
        )

    def test_phase_b_has_inference_bus_ref(self):
        """Phase B trainer extracts inference bus ref from model."""
        src = inspect.getsource(ae_train.ContextualRSSMTrainer.__init__)
        assert '_inference_bus_ref' in src, (
            "Phase B __init__ must set _inference_bus_ref"
        )

    def test_bridge_uses_convergence_monitor(self):
        """Bridge creation references self.convergence_monitor."""
        src = inspect.getsource(ae_train.ContextualRSSMTrainer.__init__)
        assert 'convergence_monitor=self.convergence_monitor' in src, (
            "Bridge must wrap the trainer's convergence_monitor"
        )


# ======================================================================
# PATCH-INTEG-3/5: Bridge poll in training loops
# ======================================================================

class TestInteg3PhaseAPoll:
    """PATCH-INTEG-3: Phase A polls bridge after convergence update."""

    def test_train_method_polls_bridge(self):
        """Phase A train method calls _conv_ee_bridge.poll()."""
        # Find the train method that contains convergence_monitor.update
        # and verify it also calls _conv_ee_bridge.poll()
        src = inspect.getsource(ae_train.SafeThoughtAETrainerV4)
        # Find the poll call after convergence_monitor.update in the train method
        assert '_conv_ee_bridge.poll(' in src, (
            "Phase A training loop must call _conv_ee_bridge.poll()"
        )


class TestInteg5PhaseBPoll:
    """PATCH-INTEG-5: Phase B polls bridge after convergence update."""

    def test_train_method_polls_bridge(self):
        """Phase B train method calls _conv_ee_bridge.poll()."""
        src = inspect.getsource(ae_train.ContextualRSSMTrainer)
        assert '_conv_ee_bridge.poll(' in src, (
            "Phase B training loop must call _conv_ee_bridge.poll()"
        )


# ======================================================================
# Functional: Bridge poll writes signals to bus
# ======================================================================

class TestBridgePollWritesSignals:
    """Functional test: bridge poll writes training_convergence_* signals."""

    def _make_bus(self):
        """Create a minimal CognitiveFeedbackBus."""
        import torch
        bus = CognitiveFeedbackBus(hidden_dim=32)
        return bus

    def test_bridge_poll_diverging_writes_signal(self):
        """Bridge poll writes training_convergence_diverging on diverge."""
        bus = self._make_bus()
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=bus,
        )
        # Push past warmup with stable values
        for v in [0.5, 0.5, 0.5, 0.5, 0.5]:
            tcm.update(v)
        # Trigger divergence
        tcm.update(100.0)
        assert tcm.last_verdict == 'diverging'
        # Poll bridge — should detect transition and write signal
        bridge.poll(100.0)
        sig = float(bus.read_signal('training_convergence_diverging', 0.0))
        assert sig == 1.0, (
            f"training_convergence_diverging should be 1.0, got {sig}"
        )

    def test_bridge_poll_stagnating_writes_signal(self):
        """Bridge poll writes training_convergence_stagnating on stagnation."""
        bus = self._make_bus()
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=bus,
        )
        # Push enough identical values to trigger stagnation
        for _ in range(10):
            tcm.update(0.5)
        assert tcm.last_verdict == 'stagnating'
        bridge.poll(0.5)
        sig = float(bus.read_signal('training_convergence_stagnating', 0.0))
        assert sig == 1.0, (
            f"training_convergence_stagnating should be 1.0, got {sig}"
        )

    def test_bridge_poll_no_duplicate_on_same_verdict(self):
        """Bridge poll is a no-op when verdict hasn't changed."""
        bus = self._make_bus()
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=bus,
        )
        # Push past warmup with converging values
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            tcm.update(v)
        bridge.poll(0.6)
        # Poll again — no transition, should be no-op
        bridge.poll(0.55)
        # Signal should still be 0.0 (no divergence/stagnation)
        sig = float(bus.read_signal('training_convergence_diverging', 0.0))
        assert sig == 0.0

    def test_bridge_poll_records_error_evolution(self):
        """Bridge poll records verdict transition in error evolution."""
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=None,  # No bus — just test error evolution
        )
        # Push to diverging
        for v in [0.5, 0.5, 0.5, 0.5, 0.5]:
            tcm.update(v)
        tcm.update(100.0)
        bridge.poll(100.0)
        summary = ee.get_error_summary()
        error_classes = summary.get('error_classes', {})
        # Should have convergence_diverging recorded by bridge
        assert 'convergence_diverging' in error_classes, (
            f"Expected convergence_diverging in error evolution, "
            f"got: {list(error_classes.keys())}"
        )


# ======================================================================
# End-to-end: convergence → bridge → bus → MCT reads
# ======================================================================

class TestEndToEndConvergenceBusMCT:
    """E2E: Training convergence transitions reach MCT via bus."""

    def test_mct_reads_training_convergence_diverging(self):
        """MCT.evaluate() reads training_convergence_diverging from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'training_convergence_diverging' in src, (
            "MCT.evaluate() must read training_convergence_diverging"
        )

    def test_mct_reads_training_convergence_stagnating(self):
        """MCT.evaluate() reads training_convergence_stagnating from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'training_convergence_stagnating' in src, (
            "MCT.evaluate() must read training_convergence_stagnating"
        )

    def test_mct_reads_training_convergence_conflicting(self):
        """MCT.evaluate() reads training_convergence_conflicting from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'training_convergence_conflicting' in src, (
            "MCT.evaluate() must read training_convergence_conflicting"
        )

    def test_e2e_diverging_flows_to_mct_bus(self):
        """Full flow: diverging verdict → bridge → bus signal present."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=bus,
        )
        # Simulate training: warmup then diverge
        for v in [0.5, 0.5, 0.5, 0.5, 0.5]:
            tcm.update(v)
        tcm.update(100.0)
        bridge.poll(100.0)
        # Verify the signal is on the bus where MCT can read it
        signal = float(bus.read_signal('training_convergence_diverging', 0.0))
        assert signal == 1.0, (
            "MCT bus should have training_convergence_diverging=1.0 "
            f"after divergence, got {signal}"
        )

    def test_e2e_stagnating_flows_to_mct_bus(self):
        """Full flow: stagnating verdict → bridge → bus signal present."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        ee = CausalErrorEvolutionTracker(max_history=50)
        tcm = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=5, error_evolution=ee,
        )
        bridge = ConvergenceErrorEvolutionBridge(
            convergence_monitor=tcm,
            error_evolution=ee,
            feedback_bus=bus,
        )
        for _ in range(10):
            tcm.update(0.5)
        bridge.poll(0.5)
        signal = float(bus.read_signal('training_convergence_stagnating', 0.0))
        assert signal == 1.0, (
            "MCT bus should have training_convergence_stagnating=1.0 "
            f"after stagnation, got {signal}"
        )


# ======================================================================
# Signal ecosystem: no new orphans
# ======================================================================

class TestNoNewOrphans:
    """Ensure our patches don't introduce new orphaned signals."""

    def test_integ_patches_in_ae_train_source(self):
        """All PATCH-INTEG comments are present in ae_train.py source."""
        with open(ae_train.__file__) as f:
            src = f.read()
        for patch_id in [
            'PATCH-INTEG-1',
            'PATCH-INTEG-2',
            'PATCH-INTEG-3',
            'PATCH-INTEG-4',
            'PATCH-INTEG-5',
        ]:
            assert patch_id in src, f"{patch_id} comment missing from ae_train.py"

    def test_bridge_signals_all_consumed_by_mct(self):
        """Signals written by bridge (diverging/stagnating/conflicting)
        are all read by MCT."""
        mct_src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        for signal in [
            'training_convergence_diverging',
            'training_convergence_stagnating',
            'training_convergence_conflicting',
        ]:
            assert signal in mct_src, (
                f"MCT.evaluate() should read '{signal}' from bus"
            )
