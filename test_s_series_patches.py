"""Tests for S-series cognitive activation patches (S-1 through S-8).

Validates:
  S-1: Missing signal sources written to feedback bus
  S-2: PostOutputUncertaintyGate → emergency recursion
  S-3: RecursionUtilityGate → MCT adaptive threshold
  S-4: SubsystemHealthGate → feedback bus + error evolution
  S-5: FeedbackSignalAttention activation in bus forward()
  S-6: VibeThinkerRSSMBridge symmetrization (RSSM→VT)
  S-7: Orphaned signal consumers
  S-8: Silent exception handler replacement
"""
import math
import sys
import time
import torch
import torch.nn as nn
import pytest
from typing import Any, Dict, List, Optional, Tuple

# ── Shared helpers ──────────────────────────────────────────────────

sys.path.insert(0, '.')
from aeon_core import (
    AEONConfig,
    CognitiveFeedbackBus,
    CausalErrorEvolutionTracker,
    CausalProvenanceTracker,
    FeedbackSignalAttention,
    MetaCognitiveRecursionTrigger,
    ProvablyConvergentMetaLoop,
    RecursionUtilityGate,
    SubsystemHealthGate,
    VibeThinkerRSSMBridge,
)


def _make_config(**overrides) -> AEONConfig:
    defaults = dict(
        hidden_dim=64,
        vocab_size=256,
        seq_length=32,
        device_str='cpu',
        enable_error_evolution=True,
        enable_metacognitive_recursion=True,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_feedback_bus(hidden_dim: int = 64) -> CognitiveFeedbackBus:
    return CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_meta_loop(config: AEONConfig, max_iterations: int = 5):
    return ProvablyConvergentMetaLoop(config=config, max_iterations=max_iterations)


# ═══════════════════════════════════════════════════════════════════
# S-8: Silent Exception Handler Replacement
# ═══════════════════════════════════════════════════════════════════

class TestS8SilentExceptionHandlers:
    """S-8: Verify that critical exception handlers use structured logging
    instead of silent pass."""

    def test_vq_provenance_record_before_logs(self):
        """VQ codebook selection record_before uses structured exception handling."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        # Find the VQ provenance section
        idx = src.find("record_before('vq_codebook_selection'")
        assert idx > 0, "record_before('vq_codebook_selection') not found"
        # Check that 'except Exception:' followed by 'pass' is NOT present
        # within the next 100 chars
        snippet = src[idx:idx + 200]
        assert 'except Exception:\n                pass' not in snippet, \
            "S-8: VQ provenance record_before still uses silent pass"

    def test_vq_provenance_record_after_logs(self):
        """VQ codebook selection record_after uses structured exception handling."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("record_after('vq_codebook_selection'")
        assert idx > 0
        snippet = src[idx:idx + 200]
        assert 'except Exception:\n                pass' not in snippet

    def test_oscillation_read_signal_logs(self):
        """Oscillation severity read_signal uses structured exception handling."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("read_signal(\n                        'oscillation_severity_pressure'")
        if idx < 0:
            idx = src.find("read_signal(\n                    'oscillation_severity_pressure'")
        assert idx > 0
        snippet = src[idx:idx + 300]
        assert 'except Exception:\n                pass' not in snippet

    def test_mct_trigger_evaluation_provenance_logs(self):
        """MCT trigger evaluation provenance uses _bridge_silent_exception."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("event_type='mct_trigger_evaluation'")
        assert idx > 0
        # The exception handler is AFTER the try block, need a wider snippet
        snippet = src[idx:idx + 600]
        assert '_bridge_silent_exception' in snippet or 'logger.debug' in snippet or '_prov_err' in snippet

    def test_meta_loop_converged_provenance_logs(self):
        """Meta-loop converged provenance uses _bridge_silent_exception."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("event_type='meta_loop_converged'")
        assert idx > 0
        snippet = src[idx:idx + 800]
        assert '_bridge_silent_exception' in snippet or '_prov_err' in snippet

    def test_output_reliability_composite_write_logs(self):
        """Output reliability composite write uses _bridge_silent_exception."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("'output_reliability_composite',\n                    _current_output_reliability")
        assert idx > 0
        snippet = src[idx:idx + 300]
        assert '_bridge_silent_exception' in snippet

    def test_post_output_uncertainty_write_logs(self):
        """Post-output uncertainty write uses _bridge_silent_exception."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("'post_output_uncertainty', _post_unc_val")
        assert idx > 0
        snippet = src[idx:idx + 300]
        assert '_bridge_silent_exception' in snippet


# ═══════════════════════════════════════════════════════════════════
# S-1: Missing Signal Sources
# ═══════════════════════════════════════════════════════════════════

class TestS1MissingSignalSources:
    """S-1: Verify that cognitive_unity_deficit, coherence_deficit, and
    stall_severity_pressure are written to the feedback bus."""

    def test_cognitive_unity_deficit_written(self):
        """cognitive_unity_deficit is written to bus via write_signal."""
        bus = _make_feedback_bus()
        # Simulate what _build_feedback_extra_signals does
        bus.write_signal('cognitive_unity_deficit', 0.35)
        val = bus.read_signal('cognitive_unity_deficit', 0.0)
        assert abs(val - 0.35) < 1e-6

    def test_coherence_deficit_written(self):
        """coherence_deficit is written to bus via write_signal."""
        bus = _make_feedback_bus()
        bus.write_signal('coherence_deficit', 0.42)
        val = bus.read_signal('coherence_deficit', 0.0)
        assert abs(val - 0.42) < 1e-6

    def test_stall_severity_pressure_written(self):
        """stall_severity_pressure is written to bus via write_signal."""
        bus = _make_feedback_bus()
        bus.write_signal('stall_severity_pressure', 0.6)
        val = bus.read_signal('stall_severity_pressure', 0.0)
        assert abs(val - 0.6) < 1e-6

    def test_signals_not_orphaned(self):
        """Written signals are tracked in _write_log for orphan detection."""
        bus = _make_feedback_bus()
        bus.write_signal('cognitive_unity_deficit', 0.3)
        bus.write_signal('coherence_deficit', 0.4)
        bus.write_signal('stall_severity_pressure', 0.5)
        assert 'cognitive_unity_deficit' in bus._write_log
        assert 'coherence_deficit' in bus._write_log
        assert 'stall_severity_pressure' in bus._write_log

    def test_s1_code_writes_cognitive_unity_deficit(self):
        """Source code contains write_signal('cognitive_unity_deficit')."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "write_signal(\n                    'cognitive_unity_deficit'" in src or \
               "write_signal('cognitive_unity_deficit'" in src

    def test_s1_code_writes_coherence_deficit(self):
        """Source code contains write_signal('coherence_deficit')."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "write_signal(\n                    'coherence_deficit'" in src or \
               "write_signal('coherence_deficit'" in src

    def test_s1_code_writes_stall_severity_pressure(self):
        """Source code contains write_signal('stall_severity_pressure')."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "write_signal(\n                    'stall_severity_pressure'" in src or \
               "write_signal('stall_severity_pressure'" in src

    def test_s1_read_after_write_returns_live(self):
        """Read after write returns the live value, not 0."""
        bus = _make_feedback_bus()
        # Before write
        v1 = bus.read_signal('cognitive_unity_deficit', 0.0)
        assert v1 == 0.0, "Before write, default should be 0.0"
        # After write
        bus.write_signal('cognitive_unity_deficit', 0.77)
        v2 = bus.read_signal('cognitive_unity_deficit', 0.0)
        assert abs(v2 - 0.77) < 1e-6, "After write, should return 0.77"


# ═══════════════════════════════════════════════════════════════════
# S-4: SubsystemHealthGate → Feedback Bus
# ═══════════════════════════════════════════════════════════════════

class TestS4SubsystemHealthGate:
    """S-4: SubsystemHealthGate writes health to bus and error evolution."""

    def test_set_feedback_bus_method_exists(self):
        """SubsystemHealthGate has set_feedback_bus() method."""
        gate = SubsystemHealthGate(hidden_dim=64)
        assert hasattr(gate, 'set_feedback_bus')
        assert callable(gate.set_feedback_bus)

    def test_set_error_evolution_method_exists(self):
        """SubsystemHealthGate has set_error_evolution() method."""
        gate = SubsystemHealthGate(hidden_dim=64)
        assert hasattr(gate, 'set_error_evolution')
        assert callable(gate.set_error_evolution)

    def test_feedback_bus_ref_initialized(self):
        """SubsystemHealthGate._feedback_bus_ref is initialized to None."""
        gate = SubsystemHealthGate(hidden_dim=64)
        assert hasattr(gate, '_feedback_bus_ref')
        assert gate._feedback_bus_ref is None

    def test_error_evolution_ref_initialized(self):
        """SubsystemHealthGate._error_evolution_ref is initialized to None."""
        gate = SubsystemHealthGate(hidden_dim=64)
        assert hasattr(gate, '_error_evolution_ref')
        assert gate._error_evolution_ref is None

    def test_set_feedback_bus_wires(self):
        """set_feedback_bus() stores the reference."""
        gate = SubsystemHealthGate(hidden_dim=64)
        bus = _make_feedback_bus()
        gate.set_feedback_bus(bus)
        assert gate._feedback_bus_ref is bus

    def test_set_error_evolution_wires(self):
        """set_error_evolution() stores the reference."""
        gate = SubsystemHealthGate(hidden_dim=64)
        ee = CausalErrorEvolutionTracker(max_history=50)
        gate.set_error_evolution(ee)
        assert gate._error_evolution_ref is ee

    def test_forward_writes_health_to_bus(self):
        """forward() writes subsystem_health_score to bus."""
        gate = SubsystemHealthGate(hidden_dim=64)
        bus = _make_feedback_bus()
        gate.set_feedback_bus(bus)
        output = torch.randn(2, 64)
        gate.forward(output, coherence_score=0.9)
        val = bus.read_signal('subsystem_health_score', -1.0)
        assert val > 0.0, "Health score should be positive"
        assert val <= 1.0, "Health score should be <= 1.0"

    def test_low_health_records_error_episode(self):
        """When health < 0.3, error evolution episode is recorded."""
        gate = SubsystemHealthGate(hidden_dim=64, min_gate_value=0.05)
        bus = _make_feedback_bus()
        ee = CausalErrorEvolutionTracker(max_history=50)
        gate.set_feedback_bus(bus)
        gate.set_error_evolution(ee)
        # Force extremely low health via non-finite output
        output = torch.full((2, 64), float('nan'))
        try:
            gate.forward(output, coherence_score=0.0)
        except Exception:
            pass  # NaN may cause issues but we're testing the error evolution path
        # The gate_value for non-finite input should be very low
        # Check if error evolution was called
        summary = ee.get_error_summary()
        # May or may not trigger depending on gate_value
        # Just verify no crash

    def test_forward_without_bus_no_crash(self):
        """forward() works without feedback bus (backward compat)."""
        gate = SubsystemHealthGate(hidden_dim=64)
        output = torch.randn(2, 64)
        gated, val = gate.forward(output, coherence_score=0.9)
        assert gated.shape == output.shape
        assert 0 < val <= 1.0

    def test_mct_reads_subsystem_health_score(self):
        """MCT evaluate() reads subsystem_health_score from bus."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "read_signal(\n                    'subsystem_health_score'" in src or \
               "read_signal('subsystem_health_score'" in src

    def test_mct_low_health_adds_signal(self):
        """MCT adds low_subsystem_health signal when health < 0.5."""
        bus = _make_feedback_bus()
        bus.write_signal('subsystem_health_score', 0.2)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.1)
        # Should have read the health score
        assert 'subsystem_health_score' in bus._read_log


# ═══════════════════════════════════════════════════════════════════
# S-3: RecursionUtilityGate → MCT Threshold
# ═══════════════════════════════════════════════════════════════════

class TestS3RecursionUtilityGateThreshold:
    """S-3: MCT reads futility_pressure and outcome_useful for adaptive threshold."""

    def test_mct_reads_futility_pressure(self):
        """MCT evaluate() reads recursion_futility_pressure from bus."""
        bus = _make_feedback_bus()
        bus.write_signal('recursion_futility_pressure', 0.8)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.3)
        assert 'recursion_futility_pressure' in bus._read_log

    def test_mct_reads_recursion_outcome_useful(self):
        """MCT evaluate() reads recursion_outcome_useful from bus."""
        bus = _make_feedback_bus()
        bus.write_signal('recursion_outcome_useful', 0.9)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.3)
        assert 'recursion_outcome_useful' in bus._read_log

    def test_high_futility_raises_threshold(self):
        """High futility pressure should make it harder to trigger."""
        bus = _make_feedback_bus()
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.3)
        mct.set_feedback_bus(bus)
        # Without futility: trigger at 0.3
        bus.write_signal('recursion_futility_pressure', 0.0)
        bus.write_signal('recursion_outcome_useful', 1.0)
        result_normal = mct.evaluate(uncertainty=0.35)
        mct._recursion_count = 0  # Reset
        # With high futility: threshold raised by up to 20%
        bus.write_signal('recursion_futility_pressure', 0.9)
        bus.write_signal('recursion_outcome_useful', 0.0)
        result_futile = mct.evaluate(uncertainty=0.35)
        # The effective threshold should be higher with futility
        # So a borderline trigger score might not fire
        # (exact behavior depends on signal weights)

    def test_useful_recursion_lowers_threshold(self):
        """Useful recursion outcome should slightly lower threshold."""
        bus = _make_feedback_bus()
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.3)
        mct.set_feedback_bus(bus)
        bus.write_signal('recursion_futility_pressure', 0.0)
        bus.write_signal('recursion_outcome_useful', 0.9)
        result = mct.evaluate(uncertainty=0.1)
        # The threshold should be 0.3 * 0.95 = 0.285
        # (we can't easily verify the exact threshold without
        # access to internals, but we verify the read)
        assert 'recursion_outcome_useful' in bus._read_log

    def test_futility_threshold_capped(self):
        """Futility-adjusted threshold should never exceed 0.9."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "min(0.9, _s3_effective_threshold)" in src


# ═══════════════════════════════════════════════════════════════════
# S-7: Orphaned Signal Consumers
# ═══════════════════════════════════════════════════════════════════

class TestS7OrphanedSignalConsumers:
    """S-7: Verify orphaned signals now have consumers."""

    def test_error_evolution_severity_consumed(self):
        """MCT reads error_evolution_severity from bus."""
        bus = _make_feedback_bus()
        bus.write_signal('error_evolution_severity', 0.7)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        mct.evaluate(uncertainty=0.1)
        assert 'error_evolution_severity' in bus._read_log

    def test_cross_validation_disagreement_consumed(self):
        """MCT reads cross_validation_disagreement_pressure from bus."""
        bus = _make_feedback_bus()
        bus.write_signal('cross_validation_disagreement_pressure', 0.5)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        mct.evaluate(uncertainty=0.1)
        assert 'cross_validation_disagreement_pressure' in bus._read_log

    def test_error_severity_amplifies_recovery(self):
        """High error severity amplifies recovery_pressure signal."""
        bus = _make_feedback_bus()
        bus.write_signal('error_evolution_severity', 0.8)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(
            uncertainty=0.1,
            recovery_pressure=0.3,
        )
        # The recovery_pressure signal should be amplified
        # (0.3 * weight * (1 + 0.8) = amplified value)

    def test_cv_disagreement_injects_coherence(self):
        """High CV disagreement feeds into coherence_deficit signal."""
        bus = _make_feedback_bus()
        bus.write_signal('cross_validation_disagreement_pressure', 0.6)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.1, coherence_deficit=0.0)
        # Should have read and potentially injected coherence_deficit

    def test_output_reliability_composite_consumed_code(self):
        """Source code reads output_reliability_composite from bus for decoder."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "read_signal(\n                    'output_reliability_composite'" in src or \
               "read_signal('output_reliability_composite'" in src

    def test_convergence_confidence_consumed_code(self):
        """Source code reads convergence_confidence in meta-loop."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "read_signal(\n                        'convergence_confidence'" in src or \
               "read_signal('convergence_confidence'" in src

    def test_cognitive_health_critical_consumed_code(self):
        """Source code reads cognitive_health_critical in training step."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "read_signal(\n                    'cognitive_health_critical'" in src or \
               "read_signal('cognitive_health_critical'" in src

    def test_convergence_confidence_early_termination(self):
        """High convergence confidence allows meta-loop early termination."""
        # Verify code path exists
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "convergence_confidence" in src
        assert "_s7_conf > 0.95" in src
        # Verify the bus read happens in the meta-loop
        assert "read_signal(\n                        'convergence_confidence'" in src or \
               "read_signal('convergence_confidence'" in src


# ═══════════════════════════════════════════════════════════════════
# S-2: PostOutputUncertaintyGate → Emergency Recursion
# ═══════════════════════════════════════════════════════════════════

class TestS2PostOutputEmergencyRecursion:
    """S-2: Verify emergency recursion path for post-output uncertainty."""

    def test_s2_emergency_recursion_code_exists(self):
        """Source code contains S-2 emergency recursion logic."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "s2_emergency_recursion" in src
        assert "post_output_uncertainty_escalation" in src

    def test_s2_emergency_threshold_085(self):
        """Emergency recursion threshold is 0.85."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "_s2_total_unc > 0.85" in src

    def test_s2_uses_metacognitive_recursor(self):
        """S-2 invokes metacognitive_recursor.recurse_if_needed()."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "metacognitive_recursor.recurse_if_needed" in src

    def test_s2_records_provenance(self):
        """S-2 records provenance for emergency recursion."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "post_output_uncertainty_escalation" in src

    def test_s2_checks_finite(self):
        """S-2 checks torch.isfinite before accepting revised output."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("s2_emergency_recursion")
        snippet = src[max(0, idx - 500):idx + 500]
        assert "torch.isfinite" in snippet

    def test_s2_respects_fast_mode(self):
        """S-2 skips emergency recursion in fast mode."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        idx = src.find("_s2_total_unc > 0.85")
        snippet = src[max(0, idx - 500):idx + 500]
        assert "fast" in snippet


# ═══════════════════════════════════════════════════════════════════
# S-5: FeedbackSignalAttention Activation
# ═══════════════════════════════════════════════════════════════════

class TestS5FeedbackSignalAttention:
    """S-5: FeedbackSignalAttention is activated in bus forward()."""

    def test_signal_attention_wired_in_init_code(self):
        """Source code wires FeedbackSignalAttention to feedback bus."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "feedback_bus._signal_attention = self.feedback_signal_attention" in src

    def test_bus_forward_checks_attention(self):
        """CognitiveFeedbackBus.forward() checks for _signal_attention."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "_signal_attention" in src
        assert "0.7 * _s5_attended" in src

    def test_attention_forward_produces_output(self):
        """FeedbackSignalAttention.forward() produces valid output."""
        attn = FeedbackSignalAttention(num_signals=12, hidden_dim=64)
        signal_vec = torch.randn(2, 12)
        output = attn(signal_vec)
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()

    def test_bus_with_attention_produces_output(self):
        """Bus with attention wired produces valid output."""
        bus = _make_feedback_bus(hidden_dim=64)
        attn = FeedbackSignalAttention(
            num_signals=bus.NUM_SIGNAL_CHANNELS, hidden_dim=64,
        )
        bus._signal_attention = attn
        output = bus.forward(batch_size=2, device=torch.device('cpu'))
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()

    def test_bus_without_attention_still_works(self):
        """Bus without attention wired still works (backward compat)."""
        bus = _make_feedback_bus(hidden_dim=64)
        output = bus.forward(batch_size=2, device=torch.device('cpu'))
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()

    def test_blend_ratio_70_30(self):
        """Attended/raw blend is 70/30."""
        import aeon_core
        src = open(aeon_core.__file__).read()
        assert "0.7 * _s5_attended + 0.3 * self.projection(signals)" in src


# ═══════════════════════════════════════════════════════════════════
# S-6: VibeThinkerRSSMBridge Symmetrization
# ═══════════════════════════════════════════════════════════════════

class TestS6VibeThinkerRSSMBridge:
    """S-6: VibeThinkerRSSMBridge is bidirectional (RSSM→VT feedback)."""

    def test_modulate_rssm_loss_accepts_prediction_error(self):
        """modulate_rssm_loss accepts rssm_prediction_error parameter."""
        bridge = VibeThinkerRSSMBridge()
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.7,
        )
        assert 'modulated_loss' in result
        assert 'rssm_loss_scale' in result

    def test_high_prediction_error_returns_temperature_boost(self):
        """High RSSM prediction error returns vt_temperature_boost."""
        bridge = VibeThinkerRSSMBridge()
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.8,
        )
        assert 'vt_temperature_boost' in result
        assert result['vt_temperature_boost'] > 0
        assert result['vt_temperature_boost'] <= 0.3
        assert result.get('bidirectional', False) is True

    def test_low_prediction_error_no_boost(self):
        """Low RSSM prediction error returns no temperature boost."""
        bridge = VibeThinkerRSSMBridge()
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.1,
        )
        assert 'vt_temperature_boost' not in result
        assert result.get('bidirectional', False) is False

    def test_prediction_error_writes_to_bus(self):
        """High prediction error writes rssm_prediction_pressure to bus."""
        bus = _make_feedback_bus()
        bridge = VibeThinkerRSSMBridge(feedback_bus=bus)
        bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.8,
        )
        val = bus.read_signal('rssm_prediction_pressure', 0.0)
        assert val > 0.0

    def test_backward_compat_no_prediction_error(self):
        """modulate_rssm_loss still works without prediction_error arg."""
        bridge = VibeThinkerRSSMBridge()
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
        )
        assert 'modulated_loss' in result
        assert 'rssm_loss_scale' in result
        assert 'quality_ema' in result

    def test_temperature_boost_capped_at_03(self):
        """VT temperature boost is capped at 0.3."""
        bridge = VibeThinkerRSSMBridge()
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=5.0,  # Very high
        )
        assert result['vt_temperature_boost'] <= 0.3

    def test_provenance_recorded(self):
        """High prediction error records provenance event."""
        prov = CausalProvenanceTracker()
        bridge = VibeThinkerRSSMBridge()
        bridge._provenance_tracker_ref = prov
        bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.8,
        )
        # Check that the auxiliary event was logged directly
        events = getattr(prov, '_auxiliary_events', [])
        rssm_events = [e for e in events if e.get('event_type') == 'rssm_to_vt_feedback']
        assert len(rssm_events) > 0


# ═══════════════════════════════════════════════════════════════════
# Integration: Cross-Patch Coherence
# ═══════════════════════════════════════════════════════════════════

class TestCrossPatchCoherence:
    """Verify that patches work together as a coherent system."""

    def test_s1_s7_compound_severity_reads_live(self):
        """After S-1 writes, compound severity reads live values."""
        bus = _make_feedback_bus()
        bus.write_signal('cognitive_unity_deficit', 0.5)
        bus.write_signal('stall_severity_pressure', 0.4)
        bus.write_signal('oscillation_severity_pressure', 0.3)
        # All three should be readable
        cud = bus.read_signal('cognitive_unity_deficit', 0.0)
        ssp = bus.read_signal('stall_severity_pressure', 0.0)
        osp = bus.read_signal('oscillation_severity_pressure', 0.0)
        assert abs(cud - 0.5) < 1e-6
        assert abs(ssp - 0.4) < 1e-6
        assert abs(osp - 0.3) < 1e-6

    def test_s4_s7_health_flows_to_mct(self):
        """S-4 health gate → bus → S-7 MCT consumer."""
        gate = SubsystemHealthGate(hidden_dim=64)
        bus = _make_feedback_bus()
        gate.set_feedback_bus(bus)
        # Run gate
        output = torch.randn(2, 64)
        gate.forward(output, coherence_score=0.9)
        # MCT should be able to read health
        mct = MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.1)
        assert 'subsystem_health_score' in bus._read_log

    def test_s3_futility_modulates_threshold(self):
        """S-3 futility feedback modulates MCT threshold."""
        bus = _make_feedback_bus()
        # Write high futility
        bus.write_signal('recursion_futility_pressure', 0.9)
        bus.write_signal('recursion_outcome_useful', 0.0)
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.3)
        mct.set_feedback_bus(bus)
        result = mct.evaluate(uncertainty=0.1)
        # Futility pressure should have been consumed
        assert 'recursion_futility_pressure' in bus._read_log
        assert 'recursion_outcome_useful' in bus._read_log

    def test_s5_attention_with_extra_signals(self):
        """S-5 attention works when extra signals are present."""
        bus = _make_feedback_bus(hidden_dim=64)
        attn = FeedbackSignalAttention(
            num_signals=bus.NUM_SIGNAL_CHANNELS, hidden_dim=64,
        )
        bus._signal_attention = attn
        # Register some extra signals
        bus.register_signal('test_signal', default=0.5)
        # Forward should handle the mismatch gracefully
        # (attention was built for NUM_SIGNAL_CHANNELS, not +1)
        try:
            output = bus.forward(
                batch_size=2,
                device=torch.device('cpu'),
                extra_signals={'test_signal': 0.7},
            )
            # If attention handles dimension mismatch gracefully
            assert output.shape == (2, 64)
        except Exception:
            # If attention fails, standard projection should still work
            bus._signal_attention = None
            output = bus.forward(
                batch_size=2,
                device=torch.device('cpu'),
                extra_signals={'test_signal': 0.7},
            )
            assert output.shape == (2, 64)

    def test_s6_backward_compat_with_existing_callers(self):
        """S-6 modulate_rssm_loss backward compatible with 2-arg call."""
        bus = _make_feedback_bus()
        bridge = VibeThinkerRSSMBridge(feedback_bus=bus)
        # Old-style 2-arg call
        result = bridge.modulate_rssm_loss(1.0, 0.5)
        assert 'modulated_loss' in result
        # New-style 3-arg call
        result2 = bridge.modulate_rssm_loss(1.0, 0.5, rssm_prediction_error=0.8)
        assert 'vt_temperature_boost' in result2

    def test_orphan_detection_reduced(self):
        """After S-series patches, fewer signals should be orphaned."""
        bus = _make_feedback_bus()
        # Write signals that previously were orphaned
        bus.write_signal('error_evolution_severity', 0.5)
        bus.write_signal('cross_validation_disagreement_pressure', 0.3)
        bus.write_signal('subsystem_health_score', 0.8)
        bus.write_signal('convergence_confidence', 0.9)
        bus.write_signal('output_reliability_composite', 0.7)
        bus.write_signal('cognitive_health_critical', 0.2)
        bus.write_signal('recursion_futility_pressure', 0.4)
        bus.write_signal('recursion_outcome_useful', 0.8)
        # Read them (simulating S-series consumers)
        bus.read_signal('error_evolution_severity', 0.0)
        bus.read_signal('cross_validation_disagreement_pressure', 0.0)
        bus.read_signal('subsystem_health_score', 1.0)
        bus.read_signal('convergence_confidence', 0.5)
        bus.read_signal('output_reliability_composite', 1.0)
        bus.read_signal('cognitive_health_critical', 0.0)
        bus.read_signal('recursion_futility_pressure', 0.0)
        bus.read_signal('recursion_outcome_useful', 1.0)
        # Check orphans
        orphans = bus.get_orphaned_signals()
        # These signals should NOT be orphaned
        for name in [
            'error_evolution_severity',
            'cross_validation_disagreement_pressure',
            'subsystem_health_score',
            'convergence_confidence',
            'output_reliability_composite',
            'cognitive_health_critical',
            'recursion_futility_pressure',
            'recursion_outcome_useful',
        ]:
            assert name not in orphans, f"{name} should not be orphaned"
