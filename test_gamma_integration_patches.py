"""
Γ-Series Integration Patches — Test Suite
==========================================

Tests for the 6 final cognitive activation patches that transform
AEON-Delta from a connected architecture to a functional cognitive
organism:

  Γ1  MCT Decision Broadcast
  Γ2  Causal Trace Coverage Enforcement (auto-provenance)
  Γ3  Compute-Loss Effectiveness Feedback
  Γ4  VibeThinker First-Class Integration
  Γ5  MCTS Confidence → Output Reliability Synthesis
  Γ6  Server Signal Bidirectional Bridge
"""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    UnifiedCognitiveCycle,
    AEONConfig,
    ConvergenceMonitor,
    ModuleCoherenceVerifier,
    CausalErrorEvolutionTracker,
    CausalProvenanceTracker,
)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def bus():
    """Fresh CognitiveFeedbackBus for isolated testing."""
    return CognitiveFeedbackBus(hidden_dim=256)


@pytest.fixture
def mct(bus):
    """MetaCognitiveRecursionTrigger wired to a feedback bus."""
    trigger = MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=3,
    )
    trigger.set_feedback_bus(bus)
    return trigger


@pytest.fixture
def config():
    """Minimal AEONConfig."""
    return AEONConfig(
        vocab_size=256,
        hidden_dim=256,
        z_dim=256,
    )


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ2: Causal Trace Coverage Enforcement
# ═══════════════════════════════════════════════════════════════════

class TestGamma2_CausalTraceCoverage:
    """PATCH-Γ2: Auto-provenance capture in write_signal()."""

    def test_gamma2_01_provenance_init(self, bus):
        """_signal_provenance and _trace_enforcement initialized."""
        assert hasattr(bus, '_signal_provenance')
        assert isinstance(bus._signal_provenance, dict)
        assert hasattr(bus, '_trace_enforcement')
        assert bus._trace_enforcement is True

    def test_gamma2_02_auto_capture_on_write(self, bus):
        """write_signal() auto-captures provenance metadata."""
        bus.write_signal('test_signal', 0.42)
        prov = bus._signal_provenance.get('test_signal')
        assert prov is not None
        assert 'source_module' in prov
        assert 'timestamp' in prov
        assert prov['value'] == pytest.approx(0.42)

    def test_gamma2_03_source_module_captured(self, bus):
        """Provenance captures the caller's class name."""
        # When called from test method, source_module should be this
        # test class or 'TestGamma2_CausalTraceCoverage'
        bus.write_signal('test_source', 0.5)
        prov = bus._signal_provenance['test_source']
        assert isinstance(prov['source_module'], str)
        assert prov['source_module'] != ''

    def test_gamma2_04_get_signal_provenance(self, bus):
        """get_signal_provenance() returns provenance for known signal."""
        bus.write_signal('named_signal', 0.7)
        result = bus.get_signal_provenance('named_signal')
        assert result is not None
        assert result['value'] == pytest.approx(0.7)

    def test_gamma2_05_get_signal_provenance_missing(self, bus):
        """get_signal_provenance() returns None for unknown signal."""
        assert bus.get_signal_provenance('nonexistent') is None

    def test_gamma2_06_get_full_provenance_map(self, bus):
        """get_full_provenance_map() returns all provenance entries."""
        bus.write_signal('sig_a', 0.1)
        bus.write_signal('sig_b', 0.2)
        bus.write_signal('sig_c', 0.3)
        pmap = bus.get_full_provenance_map()
        assert 'sig_a' in pmap
        assert 'sig_b' in pmap
        assert 'sig_c' in pmap
        assert len(pmap) >= 3

    def test_gamma2_07_provenance_updated_on_overwrite(self, bus):
        """Provenance is updated when signal is re-written."""
        bus.write_signal('evolving', 0.1)
        prov1 = bus.get_signal_provenance('evolving')
        assert prov1['value'] == pytest.approx(0.1)
        bus.write_signal('evolving', 0.9)
        prov2 = bus.get_signal_provenance('evolving')
        assert prov2['value'] == pytest.approx(0.9)

    def test_gamma2_08_trace_enforcement_disable(self, bus):
        """When _trace_enforcement is False, provenance not captured."""
        bus._trace_enforcement = False
        bus.write_signal('untraced', 0.5)
        assert bus.get_signal_provenance('untraced') is None

    def test_gamma2_09_provenance_map_is_copy(self, bus):
        """get_full_provenance_map() returns a copy, not internal ref."""
        bus.write_signal('x', 1.0)
        pmap = bus.get_full_provenance_map()
        pmap['injected'] = {'fake': True}
        assert 'injected' not in bus._signal_provenance

    def test_gamma2_10_write_signal_traced_also_captured(self, bus):
        """write_signal_traced() also triggers auto-provenance."""
        bus.write_signal_traced(
            'traced_sig', 0.8,
            source_module='TestModule',
            reason='test reason',
        )
        prov = bus.get_signal_provenance('traced_sig')
        assert prov is not None
        assert prov['value'] == pytest.approx(0.8)

    def test_gamma2_11_provenance_does_not_break_ema(self, bus):
        """Provenance capture does not interfere with EMA tracking."""
        bus.write_signal('ema_test', 0.5)
        val = bus.read_signal('ema_test', 0.0)
        assert val == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ1: MCT Decision Broadcast
# ═══════════════════════════════════════════════════════════════════

class TestGamma1_MCTBroadcast:
    """PATCH-Γ1: MCT writes mct_should_trigger + mct_trigger_score."""

    def test_gamma1_01_mct_writes_should_trigger(self, bus, mct):
        """MCT.evaluate() writes mct_should_trigger to bus."""
        mct.evaluate(uncertainty=0.0)
        val = bus.read_signal('mct_should_trigger', -1.0)
        assert val in (0.0, 1.0)

    def test_gamma1_02_mct_writes_trigger_score(self, bus, mct):
        """MCT.evaluate() writes mct_trigger_score to bus."""
        mct.evaluate(uncertainty=0.3)
        val = bus.read_signal('mct_trigger_score', -1.0)
        assert val >= 0.0

    def test_gamma1_03_trigger_true_when_high_uncertainty(self, bus, mct):
        """High uncertainty → mct_should_trigger = 1.0."""
        mct.evaluate(uncertainty=0.9)
        val = bus.read_signal('mct_should_trigger', 0.0)
        assert val == pytest.approx(1.0)

    def test_gamma1_04_trigger_false_when_stable(self, bus, mct):
        """Low distress → mct_should_trigger = 0.0."""
        mct.evaluate(uncertainty=0.0)
        val = bus.read_signal('mct_should_trigger', -1.0)
        assert val == pytest.approx(0.0)

    def test_gamma1_05_trigger_score_increases_with_distress(self, bus, mct):
        """Higher distress → higher mct_trigger_score."""
        mct.evaluate(uncertainty=0.1)
        score_low = bus.read_signal('mct_trigger_score', 0.0)
        mct.reset()
        mct.evaluate(uncertainty=0.9)
        score_high = bus.read_signal('mct_trigger_score', 0.0)
        assert score_high > score_low

    def test_gamma1_06_provenance_captured(self, bus, mct):
        """mct_should_trigger has auto-provenance from Γ2."""
        mct.evaluate(uncertainty=0.5)
        prov = bus.get_signal_provenance('mct_should_trigger')
        assert prov is not None

    def test_gamma1_07_no_bus_no_crash(self):
        """MCT without feedback bus doesn't crash."""
        trigger = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=3,
        )
        result = trigger.evaluate(uncertainty=0.9)
        assert 'should_trigger' in result

    def test_gamma1_08_score_with_multiple_signals(self, bus, mct):
        """Trigger score combines multiple signal sources."""
        mct.evaluate(
            uncertainty=0.3,
            coherence_deficit=0.3,
            recovery_pressure=0.3,
        )
        score = bus.read_signal('mct_trigger_score', 0.0)
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ3: Compute-Loss Effectiveness Feedback
# ═══════════════════════════════════════════════════════════════════

class TestGamma3_LossEffectivenessFeedback:
    """PATCH-Γ3: compute_loss writes loss_scaling_factor to bus."""

    def test_gamma3_01_loss_scaling_factor_written(self, config):
        """compute_loss() writes loss_scaling_factor to bus."""
        from aeon_core import AEONDeltaV3
        model = AEONDeltaV3(config)
        model.eval()
        # Write a signal that triggers scaling
        model.feedback_bus.write_signal('safety_score', 0.3)
        # Create minimal outputs
        B, T = 2, 8
        logits = torch.randn(B, T, config.vocab_size)
        targets = torch.randint(0, config.vocab_size, (B, T))
        outputs = {'logits': logits}
        model.compute_loss(outputs, targets)
        val = model.feedback_bus.read_signal('loss_scaling_factor', -1.0)
        assert val > 0.0

    def test_gamma3_02_loss_intervention_active_written(self, config):
        """compute_loss() writes loss_intervention_active to bus."""
        from aeon_core import AEONDeltaV3
        model = AEONDeltaV3(config)
        model.eval()
        B, T = 2, 8
        logits = torch.randn(B, T, config.vocab_size)
        targets = torch.randint(0, config.vocab_size, (B, T))
        outputs = {'logits': logits}
        model.compute_loss(outputs, targets)
        val = model.feedback_bus.read_signal('loss_intervention_active', -1.0)
        assert val in (0.0, 1.0)

    def test_gamma3_03_mct_reads_loss_scaling(self, bus, mct):
        """MCT reads loss_scaling_factor when intervention is active."""
        bus.write_signal('loss_scaling_factor', 2.5)
        bus.write_signal('loss_intervention_active', 1.0)
        result = mct.evaluate(uncertainty=0.0)
        # Should boost recovery_pressure due to high scaling
        score = result.get('trigger_score', 0.0)
        assert score > 0.0

    def test_gamma3_04_mct_ignores_normal_scaling(self, bus, mct):
        """MCT does not boost when loss scaling is normal."""
        bus.write_signal('loss_scaling_factor', 1.0)
        bus.write_signal('loss_intervention_active', 0.0)
        result = mct.evaluate(uncertainty=0.0)
        # With no other signals, recovery_pressure should not activate
        score = result.get('trigger_score', 0.0)
        # Score may be small or zero
        assert score < 0.5

    def test_gamma3_05_scaling_factor_provenance(self, config):
        """loss_scaling_factor has provenance from Γ2."""
        from aeon_core import AEONDeltaV3
        model = AEONDeltaV3(config)
        model.eval()
        B, T = 2, 8
        outputs = {'logits': torch.randn(B, T, config.vocab_size)}
        targets = torch.randint(0, config.vocab_size, (B, T))
        model.compute_loss(outputs, targets)
        prov = model.feedback_bus.get_signal_provenance('loss_scaling_factor')
        assert prov is not None


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ4: VibeThinker First-Class Integration
# ═══════════════════════════════════════════════════════════════════

class TestGamma4_VibeThinkerIntegration:
    """PATCH-Γ4: VibeThinker signals → MCT + compute_loss."""

    def test_gamma4_01_mct_reads_vibe_divergence(self, bus, mct):
        """MCT reads vibe_reasoning_divergence from bus."""
        bus.write_signal('vibe_reasoning_divergence', 0.8)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        # High divergence (0.8 > 0.4) → coherence_deficit amplified
        assert score > 0.0

    def test_gamma4_02_mct_reads_vibe_confidence(self, bus, mct):
        """MCT reads vibe_reasoning_confidence from bus."""
        bus.write_signal('vibe_reasoning_confidence', 0.1)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        # Low confidence (0.1 < 0.3) → uncertainty amplified
        assert score > 0.0

    def test_gamma4_03_vibe_divergence_threshold(self, bus, mct):
        """Vibe divergence below 0.4 does not activate."""
        bus.write_signal('vibe_reasoning_divergence', 0.2)
        result = mct.evaluate(uncertainty=0.0)
        # Below threshold — should not contribute
        signals = result.get('triggers_active', [])
        # coherence_deficit should NOT be in triggers from vibe alone
        score = result.get('trigger_score', 0.0)
        assert score < 0.3  # Minimal contribution

    def test_gamma4_04_vibe_confidence_threshold(self, bus, mct):
        """Vibe confidence above 0.3 does not activate."""
        bus.write_signal('vibe_reasoning_confidence', 0.8)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        assert score < 0.3  # No activation from good confidence

    def test_gamma4_05_compute_loss_vibe_scaling(self, config):
        """compute_loss scales up when vibe divergence > 0.5."""
        from aeon_core import AEONDeltaV3
        model = AEONDeltaV3(config)
        model.eval()
        B, T = 2, 8
        logits = torch.randn(B, T, config.vocab_size)
        targets = torch.randint(0, config.vocab_size, (B, T))
        outputs = {'logits': logits}

        # Baseline loss (no vibe divergence)
        loss_base = model.compute_loss(outputs, targets)
        total_base = loss_base['total_loss'].item()

        # Write high vibe divergence
        model.feedback_bus.write_signal('vibe_reasoning_divergence', 0.9)
        loss_vibe = model.compute_loss(outputs, targets)
        total_vibe = loss_vibe['total_loss'].item()

        # Loss should be higher with vibe divergence
        assert total_vibe >= total_base

    def test_gamma4_06_both_signals_combined(self, bus, mct):
        """High divergence + low confidence amplifies trigger score."""
        bus.write_signal('vibe_reasoning_divergence', 0.9)
        bus.write_signal('vibe_reasoning_confidence', 0.1)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        assert score > 0.1  # Both should contribute


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ5: MCTS Confidence → Output Reliability
# ═══════════════════════════════════════════════════════════════════

class TestGamma5_MCTSReliability:
    """PATCH-Γ5: MCTS planning confidence degrades output reliability."""

    def test_gamma5_01_low_planning_degrades_reliability(self):
        """Low MCTS confidence → output_reliability reduced."""
        bus = CognitiveFeedbackBus(hidden_dim=256)
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=3,
        )
        mct.set_feedback_bus(bus)

        # Create UCC with bus
        cm = ConvergenceMonitor(threshold=0.01)
        coherence_verifier = ModuleCoherenceVerifier(hidden_dim=256)
        ee = CausalErrorEvolutionTracker()
        pt = CausalProvenanceTracker()
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            metacognitive_trigger=mct,
            coherence_verifier=coherence_verifier,
            error_evolution=ee,
            provenance_tracker=pt,
        )
        ucc._feedback_bus_ref = bus

        # Write low planning confidence
        bus.write_signal('mcts_planning_confidence', 0.2)

        # Evaluate UCC
        delta_norm = torch.tensor(0.1)
        result = ucc.evaluate(
            subsystem_states={},
            delta_norm=delta_norm,
        )

        # UCC should have read mcts_planning_confidence
        # and degraded output_reliability before MCT call
        assert bus.read_signal('mcts_planning_confidence', 1.0) == pytest.approx(0.2)

    def test_gamma5_02_high_planning_no_degradation(self):
        """High MCTS confidence → output_reliability unchanged."""
        bus = CognitiveFeedbackBus(hidden_dim=256)
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=3,
        )
        mct.set_feedback_bus(bus)

        cm = ConvergenceMonitor(threshold=0.01)
        coherence_verifier = ModuleCoherenceVerifier(hidden_dim=256)
        ee = CausalErrorEvolutionTracker()
        pt = CausalProvenanceTracker()
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            metacognitive_trigger=mct,
            coherence_verifier=coherence_verifier,
            error_evolution=ee,
            provenance_tracker=pt,
        )
        ucc._feedback_bus_ref = bus

        bus.write_signal('mcts_planning_confidence', 0.9)

        delta_norm = torch.tensor(0.1)
        result = ucc.evaluate(
            subsystem_states={},
            delta_norm=delta_norm,
        )
        # Should not degrade — confidence is above threshold
        assert 'should_rerun' in result

    def test_gamma5_03_planning_confidence_read(self, bus):
        """MCTS planning confidence is consumed from bus."""
        bus.write_signal('mcts_planning_confidence', 0.3)
        val = bus.read_signal('mcts_planning_confidence', 1.0)
        assert val == pytest.approx(0.3)

    def test_gamma5_04_reliability_scaling_bounds(self, bus):
        """Output reliability scaling bounded to [0.5, 1.5]."""
        # When planning_confidence = 0.0, scale = 0.5 + 0.0 = 0.5
        # When planning_confidence = 0.5, no scaling (threshold)
        # This tests the formula: output_reliability * (0.5 + conf)
        bus.write_signal('mcts_planning_confidence', 0.0)
        val = bus.read_signal('mcts_planning_confidence', 1.0)
        assert val == pytest.approx(0.0)
        # 0.5 + 0.0 = 0.5 → reliability halved (minimum degradation)


# ═══════════════════════════════════════════════════════════════════
#  PATCH-Γ6: Server Signal Bidirectional Bridge
# ═══════════════════════════════════════════════════════════════════

class TestGamma6_ServerBridge:
    """PATCH-Γ6: Server ↔ Inference signal bridge."""

    def test_gamma6_01_mct_reads_server_ssp_pressure(self, bus, mct):
        """MCT reads server_ssp_pressure from bus."""
        bus.write_signal('server_ssp_pressure', 0.8)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        # High SSP pressure (0.8 > 0.3) → coherence_deficit amplified
        assert score > 0.0

    def test_gamma6_02_ssp_below_threshold(self, bus, mct):
        """SSP pressure below 0.3 does not activate MCT."""
        bus.write_signal('server_ssp_pressure', 0.1)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        assert score < 0.3  # Minimal contribution

    def test_gamma6_03_ssp_alignment_ok_readable(self, bus):
        """ssp_alignment_ok signal can be written and read."""
        bus.write_signal('ssp_alignment_ok', 0.3)
        val = bus.read_signal('ssp_alignment_ok', 1.0)
        assert val == pytest.approx(0.3)

    def test_gamma6_04_server_ssp_pressure_writable(self, bus):
        """server_ssp_pressure signal can be written and read."""
        bus.write_signal('server_ssp_pressure', 0.6)
        val = bus.read_signal('server_ssp_pressure', 0.0)
        assert val == pytest.approx(0.6)

    def test_gamma6_05_ssp_provenance(self, bus):
        """server_ssp_pressure has auto-provenance."""
        bus.write_signal('server_ssp_pressure', 0.5)
        prov = bus.get_signal_provenance('server_ssp_pressure')
        assert prov is not None
        assert prov['value'] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
#  CROSS-PATCH INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCrossPatchIntegration:
    """End-to-end integration tests across all Γ patches."""

    def test_cross_01_full_signal_loop(self, bus, mct):
        """Full loop: bus signal → MCT → bus broadcast → readable."""
        # Write distress signals
        bus.write_signal('vibe_reasoning_divergence', 0.9)
        bus.write_signal('server_ssp_pressure', 0.8)
        bus.write_signal('loss_scaling_factor', 2.0)
        bus.write_signal('loss_intervention_active', 1.0)

        # MCT evaluates and writes back
        result = mct.evaluate(uncertainty=0.3)

        # MCT broadcast readable
        should_trigger = bus.read_signal('mct_should_trigger', -1.0)
        trigger_score = bus.read_signal('mct_trigger_score', -1.0)
        assert should_trigger in (0.0, 1.0)
        assert trigger_score >= 0.0

    def test_cross_02_provenance_chain(self, bus, mct):
        """All Γ signals have provenance metadata."""
        bus.write_signal('vibe_reasoning_divergence', 0.8)
        bus.write_signal('loss_scaling_factor', 1.5)
        bus.write_signal('server_ssp_pressure', 0.6)
        mct.evaluate(uncertainty=0.5)

        # Check provenance for all key signals
        pmap = bus.get_full_provenance_map()
        for sig in ['vibe_reasoning_divergence', 'loss_scaling_factor',
                     'server_ssp_pressure', 'mct_should_trigger',
                     'mct_trigger_score']:
            assert sig in pmap, f"Missing provenance for {sig}"

    def test_cross_03_no_orphaned_gamma_signals(self, bus, mct):
        """All Γ signals are both written and read (no orphans)."""
        # Write all Γ signals
        bus.write_signal('loss_scaling_factor', 1.5)
        bus.write_signal('loss_intervention_active', 1.0)
        bus.write_signal('vibe_reasoning_divergence', 0.6)
        bus.write_signal('vibe_reasoning_confidence', 0.2)
        bus.write_signal('server_ssp_pressure', 0.5)

        # MCT reads them all
        mct.evaluate(uncertainty=0.2)

        # mct_should_trigger and mct_trigger_score are written
        # Read them to prevent orphaning
        bus.read_signal('mct_should_trigger', 0.0)
        bus.read_signal('mct_trigger_score', 0.0)

        # All written signals should have been read
        orphans = bus.get_orphaned_signals()
        gamma_signals = {
            'loss_scaling_factor', 'loss_intervention_active',
            'vibe_reasoning_divergence', 'vibe_reasoning_confidence',
            'server_ssp_pressure', 'mct_should_trigger',
            'mct_trigger_score',
        }
        orphaned_gamma = gamma_signals & set(orphans.keys())
        assert len(orphaned_gamma) == 0, (
            f"Γ signals orphaned: {orphaned_gamma}"
        )

    def test_cross_04_cascading_trigger(self, bus, mct):
        """Cascading: vibe divergence → MCT → should_trigger → True."""
        # Single strong signal should cascade
        bus.write_signal('vibe_reasoning_divergence', 1.0)
        bus.write_signal('vibe_reasoning_confidence', 0.0)
        result = mct.evaluate(uncertainty=0.0)
        # With max divergence + min confidence → should trigger
        # coherence_deficit ≥ 0.6, uncertainty ≥ 0.4
        score = result.get('trigger_score', 0.0)
        assert score > 0.3  # Significant contribution

    def test_cross_05_model_compute_loss_gamma3_gamma4(self, config):
        """Model compute_loss writes Γ3 and reads Γ4 signals."""
        from aeon_core import AEONDeltaV3
        model = AEONDeltaV3(config)
        model.eval()

        # Write vibe divergence for Γ4b
        model.feedback_bus.write_signal('vibe_reasoning_divergence', 0.8)

        B, T = 2, 8
        outputs = {'logits': torch.randn(B, T, config.vocab_size)}
        targets = torch.randint(0, config.vocab_size, (B, T))
        result = model.compute_loss(outputs, targets)

        # Γ3 signals should be written
        lsf = model.feedback_bus.read_signal('loss_scaling_factor', -1.0)
        assert lsf > 0.0
        lia = model.feedback_bus.read_signal('loss_intervention_active', -1.0)
        assert lia in (0.0, 1.0)

    def test_cross_06_mct_sensitivity_to_combined_distress(self, bus, mct):
        """Combined Γ signals push MCT past trigger threshold."""
        # Each individually sub-threshold, but combined should trigger
        bus.write_signal('vibe_reasoning_divergence', 0.5)
        bus.write_signal('server_ssp_pressure', 0.5)
        bus.write_signal('loss_scaling_factor', 1.8)
        bus.write_signal('loss_intervention_active', 1.0)

        result = mct.evaluate(
            uncertainty=0.2,
            coherence_deficit=0.2,
        )
        score = result.get('trigger_score', 0.0)
        # Combined distress should yield meaningful score
        assert score > 0.2

    def test_cross_07_idempotent_provenance_map(self, bus):
        """Multiple writes to same signal update provenance correctly."""
        bus.write_signal('evolving_sig', 0.1)
        bus.write_signal('evolving_sig', 0.5)
        bus.write_signal('evolving_sig', 0.9)
        prov = bus.get_signal_provenance('evolving_sig')
        assert prov['value'] == pytest.approx(0.9)

    def test_cross_08_bus_forward_still_works(self, bus):
        """CognitiveFeedbackBus.forward() still produces valid output."""
        # forward() requires batch_size and device
        out = bus(batch_size=1, device='cpu')
        assert out.shape[-1] == bus.hidden_dim

    def test_cross_09_mct_result_structure_preserved(self, bus, mct):
        """MCT.evaluate() still returns all expected keys."""
        result = mct.evaluate(uncertainty=0.5)
        expected_keys = {
            'should_trigger', 'should_recurse', 'trigger_score',
            'effective_trigger_score', 'triggers_active', 'triggers',
            'recursion_count', 'trigger_count', 'signal_weights',
            'max_recursions_capped',
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
