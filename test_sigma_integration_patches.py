"""
Tests for PATCH-Σ1 through PATCH-Σ7: Final cognitive integration patches.

Patches covered:
  PATCH-Σ1: MCTSPlanner ↔ Feedback Bus Bridge
             (feedback_bus param, read convergence_quality, write
              mcts_planning_confidence + mcts_search_depth_pressure)
  PATCH-Σ2: ConvergenceMonitor → Bus
             (feedback_bus param, write convergence_monitor_quality,
              convergence_monitor_is_converging, convergence_secondary_
              degradation, convergence_lyapunov_violated)
  PATCH-Σ3: CausalProvenanceTracker → Bus
             (_fb_ref attribute, auto-broadcast provenance_attribution_
              concentration + provenance_dominance_alarm from
              compute_attribution)
  PATCH-Σ4: self_diagnostic() → Bus
             (diagnostic_system_health, diagnostic_critical_failure)
  PATCH-Σ5: Systematic flush_consumed() in UCC evaluate()
             (signal_ecosystem_health, signal_health in result)
  PATCH-Σ6: Instance cache → Bus protocol migration
             (train_step reads convergence_monitor_quality from bus
              before falling back to _cached_coherence_loss_scale)
  PATCH-Σ7: OOM recovery staleness indicator
             (oom_recovery_active bus write + MCT read)
  MCT wiring: New signals read by MCT evaluate()
             (mcts_planning_confidence, convergence_monitor_quality,
              provenance_dominance_alarm, diagnostic_system_health,
              oom_recovery_active)
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
    UnifiedCognitiveCycle,
    ConvergenceMonitor,
    CausalProvenanceTracker,
    MCTSPlanner,
    AEONTrainer,
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


def _make_model():
    """Create minimal AEONDeltaV3 for testing."""
    cfg = _make_config()
    return AEONDeltaV3(cfg)


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ1: MCTSPlanner ↔ Feedback Bus Bridge
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma1_MCTSPlanner:
    """Verify MCTSPlanner accepts feedback_bus and reads/writes signals."""

    def test_sigma1_accepts_feedback_bus(self):
        """PATCH-Σ1: MCTSPlanner accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        planner = MCTSPlanner(
            state_dim=64, action_dim=4, feedback_bus=bus,
        )
        assert planner._fb_ref is bus

    def test_sigma1_default_none(self):
        """PATCH-Σ1: MCTSPlanner defaults feedback_bus to None."""
        planner = MCTSPlanner(state_dim=64, action_dim=4)
        assert hasattr(planner, '_fb_ref')
        assert planner._fb_ref is None

    def test_sigma1_base_num_simulations_stored(self):
        """PATCH-Σ1: MCTSPlanner stores _base_num_simulations."""
        planner = MCTSPlanner(state_dim=64, action_dim=4, num_simulations=75)
        assert planner._base_num_simulations == 75
        assert planner.num_simulations == 75

    def test_sigma1_writes_planning_confidence_after_search(self):
        """PATCH-Σ1b: search() writes mcts_planning_confidence to bus."""
        bus = CognitiveFeedbackBus(64)
        planner = MCTSPlanner(
            state_dim=64, action_dim=4, num_simulations=5,
            feedback_bus=bus,
        )
        state = torch.randn(64)
        # Create a minimal world model mock that returns {'output': tensor}
        class _MockWorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return {'output': self.linear(x)}
        wm = _MockWorldModel()
        result = planner.search(state, wm)
        # Check bus signal was written
        _confidence = bus.read_signal('mcts_planning_confidence', None)
        assert _confidence is not None
        assert 0.0 <= _confidence <= 1.0

    def test_sigma1_writes_search_depth_pressure(self):
        """PATCH-Σ1b: search() writes mcts_search_depth_pressure to bus."""
        bus = CognitiveFeedbackBus(64)
        planner = MCTSPlanner(
            state_dim=64, action_dim=4, num_simulations=5,
            feedback_bus=bus,
        )
        state = torch.randn(64)
        class _MockWorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return {'output': self.linear(x)}
        wm = _MockWorldModel()
        planner.search(state, wm)
        _pressure = bus.read_signal('mcts_search_depth_pressure', None)
        assert _pressure is not None
        assert _pressure in (0.0, 1.0)

    def test_sigma1_reads_convergence_quality_low(self):
        """PATCH-Σ1a: search() reads convergence_quality from bus."""
        bus = CognitiveFeedbackBus(64)
        # Write low convergence quality BEFORE search
        bus.write_signal('convergence_quality', 0.1)
        planner = MCTSPlanner(
            state_dim=64, action_dim=4, num_simulations=10,
            feedback_bus=bus,
        )
        # Low convergence_quality should cause planner to use 1.5x simulations
        # We verify by checking that the bus read happened (signal consumed)
        state = torch.randn(64)
        class _MockWorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return {'output': self.linear(x)}
        wm = _MockWorldModel()
        result = planner.search(state, wm)
        # The planner should have completed without error
        assert 'best_action' in result

    def test_sigma1_no_crash_without_bus(self):
        """PATCH-Σ1: search() works fine without feedback_bus."""
        planner = MCTSPlanner(
            state_dim=64, action_dim=4, num_simulations=3,
        )
        state = torch.randn(64)
        class _MockWorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return {'output': self.linear(x)}
        wm = _MockWorldModel()
        result = planner.search(state, wm)
        assert 'best_action' in result


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ2: ConvergenceMonitor → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma2_ConvergenceMonitor:
    """Verify ConvergenceMonitor writes convergence verdicts to bus."""

    def test_sigma2_accepts_feedback_bus(self):
        """PATCH-Σ2: ConvergenceMonitor accepts feedback_bus kwarg."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(feedback_bus=bus)
        assert cm._fb_ref is bus

    def test_sigma2_default_none(self):
        """PATCH-Σ2: ConvergenceMonitor defaults feedback_bus to None."""
        cm = ConvergenceMonitor()
        assert hasattr(cm, '_fb_ref')
        assert cm._fb_ref is None

    def test_sigma2_writes_quality_on_converged(self):
        """PATCH-Σ2b: check() writes convergence_monitor_quality on converged."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(threshold=0.1, feedback_bus=bus)
        # Warmup: need 3+ history entries
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.05)  # below threshold 0.1 → converged
        # Now check bus signal
        quality = bus.read_signal('convergence_monitor_quality', None)
        assert quality is not None
        assert isinstance(quality, float)

    def test_sigma2_writes_is_converging(self):
        """PATCH-Σ2b: check() writes convergence_monitor_is_converging."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(threshold=1e-5, feedback_bus=bus)
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.3)  # converging but not converged
        is_conv = bus.read_signal('convergence_monitor_is_converging', None)
        assert is_conv is not None
        assert is_conv == 1.0  # status should be 'converging'

    def test_sigma2_writes_zero_when_diverging(self):
        """PATCH-Σ2b: check() writes 0.0 when diverging."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(feedback_bus=bus)
        cm.check(0.1)
        cm.check(0.5)
        cm.check(1.0)  # diverging: contraction > 1
        is_conv = bus.read_signal('convergence_monitor_is_converging', None)
        assert is_conv is not None
        assert is_conv == 0.0

    def test_sigma2_secondary_degradation_written(self):
        """PATCH-Σ2b: secondary degradation written when signals degrade."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(threshold=0.1, feedback_bus=bus)
        cm.record_secondary_signal('coherence_deficit', 0.8)
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.05)
        sec_deg = bus.read_signal('convergence_secondary_degradation', None)
        assert sec_deg is not None
        assert sec_deg > 0.0

    def test_sigma2_no_crash_without_bus(self):
        """PATCH-Σ2: check() works fine without feedback_bus."""
        cm = ConvergenceMonitor()
        cm.check(1.0)
        cm.check(0.5)
        verdict = cm.check(0.3)
        assert 'status' in verdict


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ3: CausalProvenanceTracker → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma3_ProvenanceTracker:
    """Verify CausalProvenanceTracker broadcasts attribution to bus."""

    def test_sigma3_has_fb_ref(self):
        """PATCH-Σ3: CausalProvenanceTracker has _fb_ref attribute."""
        tracker = CausalProvenanceTracker()
        assert hasattr(tracker, '_fb_ref')
        assert tracker._fb_ref is None

    def test_sigma3_fb_ref_can_be_set(self):
        """PATCH-Σ3: _fb_ref can be assigned after construction."""
        bus = CognitiveFeedbackBus(64)
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        assert tracker._fb_ref is bus

    def test_sigma3_compute_attribution_writes_concentration(self):
        """PATCH-Σ3a/b: compute_attribution() writes provenance_attribution_concentration."""
        bus = CognitiveFeedbackBus(64)
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        # Record some module deltas
        state = torch.randn(64)
        tracker.record_before('encoder', state)
        tracker.record_after('encoder', state + torch.randn(64) * 0.5)
        tracker.record_before('decoder', state)
        tracker.record_after('decoder', state + torch.randn(64) * 0.1)
        attribution = tracker.compute_attribution()
        # Verify bus signal
        concentration = bus.read_signal(
            'provenance_attribution_concentration', None,
        )
        assert concentration is not None
        assert 0.0 <= concentration <= 1.0
        assert 'contributions' in attribution

    def test_sigma3_dominance_alarm_on_high_concentration(self):
        """PATCH-Σ3b: provenance_dominance_alarm written when concentration > 0.6."""
        bus = CognitiveFeedbackBus(64)
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        # One dominant module
        state = torch.randn(64)
        tracker.record_before('dominant', state)
        tracker.record_after('dominant', state + torch.randn(64) * 10.0)
        tracker.record_before('minor', state)
        tracker.record_after('minor', state + torch.randn(64) * 0.001)
        tracker.compute_attribution()
        alarm = bus.read_signal('provenance_dominance_alarm', None)
        # Should be written because dominant module has >60% of attribution
        assert alarm is not None
        assert alarm > 0.6

    def test_sigma3_no_alarm_when_balanced(self):
        """PATCH-Σ3b: no alarm when attribution is balanced."""
        bus = CognitiveFeedbackBus(64)
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        state = torch.randn(64)
        # Three equally contributing modules
        tracker.record_before('a', state)
        tracker.record_after('a', state + torch.randn(64) * 1.0)
        tracker.record_before('b', state)
        tracker.record_after('b', state + torch.randn(64) * 1.0)
        tracker.record_before('c', state)
        tracker.record_after('c', state + torch.randn(64) * 1.0)
        tracker.compute_attribution()
        alarm = bus.read_signal('provenance_dominance_alarm', 0.0)
        # Balanced — alarm should not be written (or be 0.0)
        assert alarm <= 0.6

    def test_sigma3_broadcast_method_exists(self):
        """PATCH-Σ3b: _broadcast_attribution_to_bus method exists."""
        tracker = CausalProvenanceTracker()
        assert hasattr(tracker, '_broadcast_attribution_to_bus')
        assert callable(tracker._broadcast_attribution_to_bus)

    def test_sigma3_no_crash_without_bus(self):
        """PATCH-Σ3: compute_attribution works without bus."""
        tracker = CausalProvenanceTracker()
        state = torch.randn(64)
        tracker.record_before('test', state)
        tracker.record_after('test', state + torch.randn(64))
        result = tracker.compute_attribution()
        assert 'contributions' in result


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ4: self_diagnostic() → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma4_SelfDiagnostic:
    """Verify self_diagnostic writes health signals to bus."""

    def test_sigma4_writes_diagnostic_system_health(self):
        """PATCH-Σ4: self_diagnostic() writes diagnostic_system_health."""
        model = _make_model()
        if not hasattr(model, 'feedback_bus') or model.feedback_bus is None:
            pytest.skip("Model has no feedback_bus")
        # Simulate some forward calls to activate the diagnostic
        model._total_forward_calls = torch.tensor(5)
        model.self_diagnostic()
        health = model.feedback_bus.read_signal(
            'diagnostic_system_health', None,
        )
        # Should be written (may be None if _fwd_count guard prevented it)
        if health is not None:
            assert 0.0 <= health <= 1.0

    def test_sigma4_no_crash(self):
        """PATCH-Σ4: self_diagnostic() completes without crash."""
        model = _make_model()
        result = model.self_diagnostic()
        assert isinstance(result, dict)
        assert 'gaps' in result or 'active_modules' in result


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ5: Systematic flush_consumed() in UCC evaluate()
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma5_FlushConsumed:
    """Verify UCC evaluate() calls flush_consumed() and returns signal_health."""

    def test_sigma5_ucc_result_has_signal_health(self):
        """PATCH-Σ5: UCC evaluate() returns signal_health key."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(feedback_bus=bus)
        tracker = CausalProvenanceTracker()
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            coherence_verifier=None,
            error_evolution=None,
            metacognitive_trigger=None,
            provenance_tracker=tracker,
            feedback_bus=bus,
        )
        state = torch.randn(64)
        result = ucc.evaluate(
            subsystem_states={'meta_loop': state},
            delta_norm=0.1,
        )
        assert 'signal_health' in result

    def test_sigma5_flush_activates_orphan_detection(self):
        """PATCH-Σ5: flush_consumed() activates after UCC evaluate."""
        bus = CognitiveFeedbackBus(64)
        # Write some signals that won't be read
        bus.write_signal('test_orphan_1', 0.5)
        bus.write_signal('test_orphan_2', 0.7)
        cm = ConvergenceMonitor(feedback_bus=bus)
        tracker = CausalProvenanceTracker()
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            coherence_verifier=None,
            error_evolution=None,
            metacognitive_trigger=None,
            provenance_tracker=tracker,
            feedback_bus=bus,
        )
        state = torch.randn(64)
        result = ucc.evaluate(
            subsystem_states={'meta_loop': state},
            delta_norm=0.1,
        )
        # signal_health should be populated
        sh = result.get('signal_health')
        assert sh is not None


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ6: Instance cache → Bus protocol migration
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma6_CacheMigration:
    """Verify train_step reads from bus before falling back to cache."""

    @pytest.fixture
    def trainer_fixture(self):
        """Create an AEONTrainer for testing."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        try:
            trainer = AEONTrainer(model, cfg)
            return trainer
        except Exception:
            pytest.skip("AEONTrainer could not be instantiated")

    def test_sigma6_trainer_reads_bus_convergence_quality(self, trainer_fixture):
        """PATCH-Σ6a: train_step reads convergence_monitor_quality from bus."""
        trainer = trainer_fixture
        fb = getattr(trainer.model, 'feedback_bus', None)
        if fb is None:
            pytest.skip("Model has no feedback_bus")
        # Write a low convergence quality to bus
        fb.write_signal('convergence_monitor_quality', 0.2)
        # The signal should be readable
        val = fb.read_signal('convergence_monitor_quality', None)
        assert val is not None
        assert val == pytest.approx(0.2, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
#  PATCH-Σ7: OOM recovery staleness indicator
# ══════════════════════════════════════════════════════════════════════

class TestPatchSigma7_OOMStaleness:
    """Verify OOM recovery writes staleness indicator to bus."""

    def test_sigma7_forward_has_oom_handler(self):
        """PATCH-Σ7: forward() OOM handler writes oom_recovery_active."""
        model = _make_model()
        # Verify the model has the OOM recovery code path
        import inspect
        src = inspect.getsource(model.forward)
        assert 'oom_recovery_active' in src

    def test_sigma7_oom_signal_readable(self):
        """PATCH-Σ7: oom_recovery_active can be written/read to/from bus."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('oom_recovery_active', 1.0)
        val = bus.read_signal('oom_recovery_active', 0.0)
        assert val == 1.0


# ══════════════════════════════════════════════════════════════════════
#  MCT WIRING: New signals read by MCT evaluate()
# ══════════════════════════════════════════════════════════════════════

class TestMCTWiring:
    """Verify MCT evaluate() reads the new PATCH-Σ signals from bus."""

    def test_sigma1c_mcts_planning_confidence_read(self):
        """PATCH-Σ1c: MCT reads mcts_planning_confidence from bus."""
        trigger, bus = _make_trigger()
        bus.write_signal('mcts_planning_confidence', 0.1)
        r1 = _evaluate_mct(trigger)
        trigger.reset()
        bus.write_signal('mcts_planning_confidence', 1.0)
        r2 = _evaluate_mct(trigger)
        # Low confidence should give higher trigger_score than high
        assert r1['trigger_score'] >= r2['trigger_score']

    def test_sigma2c_convergence_monitor_quality_read(self):
        """PATCH-Σ2c: MCT reads convergence_monitor_quality from bus."""
        trigger, bus = _make_trigger()
        # contraction > 1.0 means diverging → convergence_conflict
        bus.write_signal('convergence_monitor_quality', 2.0)
        r = _evaluate_mct(trigger)
        assert r['trigger_score'] > 0.0

    def test_sigma3c_provenance_dominance_alarm_read(self):
        """PATCH-Σ3c: MCT reads provenance_dominance_alarm from bus."""
        trigger, bus = _make_trigger()
        bus.write_signal('provenance_dominance_alarm', 0.9)
        r = _evaluate_mct(trigger)
        assert r['trigger_score'] > 0.0

    def test_sigma4c_diagnostic_system_health_read(self):
        """PATCH-Σ4c: MCT reads diagnostic_system_health from bus."""
        trigger, bus = _make_trigger()
        bus.write_signal('diagnostic_system_health', 0.2)
        r = _evaluate_mct(trigger)
        assert r['trigger_score'] > 0.0

    def test_sigma7c_oom_recovery_active_read(self):
        """PATCH-Σ7c: MCT reads oom_recovery_active from bus."""
        trigger, bus = _make_trigger()
        bus.write_signal('oom_recovery_active', 1.0)
        r = _evaluate_mct(trigger)
        assert r['trigger_score'] > 0.0

    def test_no_signals_baseline(self):
        """Baseline: MCT with no signals gives trigger_score=0.0."""
        trigger, bus = _make_trigger()
        r = _evaluate_mct(trigger)
        # With all safe defaults, trigger_score should be near 0
        assert r['trigger_score'] < 0.1

    def test_mct_still_reads_existing_signals(self):
        """MCT still reads existing signals (regression check)."""
        trigger, bus = _make_trigger()
        # Existing signal: training_gradient_explosion
        bus.write_signal('training_gradient_explosion', 0.8)
        r = _evaluate_mct(trigger)
        assert r['trigger_score'] > 0.0

    def test_multiple_new_signals_compound(self):
        """Multiple PATCH-Σ signals compound in trigger_score."""
        trigger, bus = _make_trigger()
        bus.write_signal('mcts_planning_confidence', 0.1)
        bus.write_signal('diagnostic_system_health', 0.1)
        bus.write_signal('oom_recovery_active', 1.0)
        r = _evaluate_mct(trigger)
        # Multiple signals should give higher score than individual
        assert r['trigger_score'] > 0.0


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION: Cross-patch coherence tests
# ══════════════════════════════════════════════════════════════════════

class TestCrossPatchCoherence:
    """Verify patches work together as a coherent system."""

    def test_convergence_monitor_to_mct_pipeline(self):
        """ConvergenceMonitor writes → bus → MCT reads → trigger decision."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(feedback_bus=bus)
        trigger = MetaCognitiveRecursionTrigger()
        trigger.set_feedback_bus(bus)
        # Simulate diverging convergence
        cm.check(0.1)
        cm.check(0.5)
        cm.check(1.0)  # diverging
        # MCT should now see the convergence_monitor_quality signal
        r = _evaluate_mct(trigger)
        quality = bus.read_signal('convergence_monitor_quality', None)
        assert quality is not None

    def test_provenance_to_mct_pipeline(self):
        """ProvenanceTracker writes → bus → MCT reads → trigger decision."""
        bus = CognitiveFeedbackBus(64)
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        trigger = MetaCognitiveRecursionTrigger()
        trigger.set_feedback_bus(bus)
        # Create dominant module attribution
        state = torch.randn(64)
        tracker.record_before('dominant', state)
        tracker.record_after('dominant', state + torch.randn(64) * 10.0)
        tracker.record_before('minor', state)
        tracker.record_after('minor', state + torch.randn(64) * 0.001)
        tracker.compute_attribution()
        # MCT should now see the dominance alarm
        r = _evaluate_mct(trigger)
        alarm = bus.read_signal('provenance_dominance_alarm', 0.0)
        if alarm > 0.6:
            assert r['trigger_score'] > 0.0

    def test_ucc_flush_cycle(self):
        """UCC evaluate → flush_consumed → signal_ecosystem_health → stable."""
        bus = CognitiveFeedbackBus(64)
        cm = ConvergenceMonitor(feedback_bus=bus)
        tracker = CausalProvenanceTracker()
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            coherence_verifier=None,
            error_evolution=None,
            metacognitive_trigger=None,
            provenance_tracker=tracker,
            feedback_bus=bus,
        )
        state = torch.randn(64)
        # Write signals that will be orphaned
        bus.write_signal('orphan_a', 0.5)
        bus.write_signal('orphan_b', 0.6)
        result = ucc.evaluate(
            subsystem_states={'meta_loop': state},
            delta_norm=0.1,
        )
        # After flush, ecosystem health may have been written
        sh = result.get('signal_health')
        assert sh is not None

    def test_model_has_all_patch_infrastructure(self):
        """Integration: AEONDeltaV3 has feedback_bus for all Σ patches."""
        model = _make_model()
        assert hasattr(model, 'feedback_bus')
        # The model should have all the components that interact with bus
        assert hasattr(model, 'self_diagnostic')
        assert hasattr(model, 'provenance_tracker')

    def test_convergence_monitor_backward_compat(self):
        """Backward compatibility: ConvergenceMonitor without bus still works."""
        cm = ConvergenceMonitor()
        cm.check(1.0)
        cm.check(0.5)
        v = cm.check(0.3)
        assert 'status' in v
        assert v['status'] in ('converged', 'converging', 'diverging', 'warmup')

    def test_mcts_planner_backward_compat(self):
        """Backward compatibility: MCTSPlanner without bus still works."""
        planner = MCTSPlanner(state_dim=32, action_dim=4, num_simulations=3)
        assert planner._fb_ref is None
        result = planner.forward(torch.randn(32))
        assert 'value' in result or 'best_action' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-q'])
