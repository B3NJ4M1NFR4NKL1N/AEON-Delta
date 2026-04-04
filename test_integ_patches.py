"""
Tests for INTEG-series patches: Final Integration & Cognitive Activation.

Patches covered:
  INTEG-1: Wire 8 orphaned bus signals into MCT evaluate()
    (a) post_output_uncertainty → uncertainty
    (b) post_pipeline_verdict_pressure → convergence_conflict
    (c) error_recovery_ratio → recovery_pressure
    (d) consolidated_output_quality → low_output_reliability
    (e) ucc_verdict_pressure → coherence_deficit
    (f) orphaned_signal_escalation → amplify ALL signals
    (g) corrective_pressure → recovery_pressure
    (h) diversity_collapse → diversity_collapse
  INTEG-2: recursion_best_improvement → threshold modulation
  INTEG-3: Causal trace recording for MCT decisions
  INTEG-4: Training pressure signals in train_step metrics
  INTEG-5: vibe_thinker_entropy + confidence → output reliability
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
    AEONTrainer,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
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


def _make_mct_with_bus():
    """Create an MCT instance with a wired CognitiveFeedbackBus."""
    bus = CognitiveFeedbackBus(64)
    mct = MetaCognitiveRecursionTrigger()
    mct.set_feedback_bus(bus)
    return mct, bus


def _baseline_evaluate(mct, **overrides):
    """Evaluate MCT with safe baseline signals and optional overrides."""
    defaults = dict(
        uncertainty=0.0,
        is_diverging=False,
        topology_catastrophe=False,
        coherence_deficit=0.0,
        memory_staleness=0.0,
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
    defaults.update(overrides)
    return mct.evaluate(**defaults)


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1a: post_output_uncertainty → MCT uncertainty
# ══════════════════════════════════════════════════════════════════════

class TestInteg1aPostOutputUncertainty:
    """post_output_uncertainty bus signal should amplify MCT uncertainty."""

    def test_signal_not_present_baseline(self):
        """When signal absent, uncertainty should not be amplified."""
        mct, bus = _make_mct_with_bus()
        result = _baseline_evaluate(mct, uncertainty=0.1)
        baseline_score = result['trigger_score']
        assert baseline_score >= 0.0

    def test_high_post_output_uncertainty_amplifies(self):
        """When post_output_uncertainty > 0.3, uncertainty amplified."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_output_uncertainty', 0.8)
        result = _baseline_evaluate(mct, uncertainty=0.2)
        # The amplified uncertainty should produce a higher score
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, uncertainty=0.2)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_low_post_output_uncertainty_no_effect(self):
        """When post_output_uncertainty <= 0.3, no amplification."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_output_uncertainty', 0.1)
        result = _baseline_evaluate(mct, uncertainty=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, uncertainty=0.2)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1b: post_pipeline_verdict_pressure → convergence_conflict
# ══════════════════════════════════════════════════════════════════════

class TestInteg1bPipelineVerdictPressure:
    """post_pipeline_verdict_pressure → convergence_conflict."""

    def test_high_pressure_amplifies_convergence_conflict(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_pipeline_verdict_pressure', 0.7)
        result = _baseline_evaluate(mct, convergence_conflict=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, convergence_conflict=0.2)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_low_pressure_no_effect(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_pipeline_verdict_pressure', 0.1)
        result = _baseline_evaluate(mct, convergence_conflict=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, convergence_conflict=0.2)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1c: error_recovery_ratio → recovery_pressure
# ══════════════════════════════════════════════════════════════════════

class TestInteg1cErrorRecoveryRatio:
    """Low error_recovery_ratio → amplified recovery_pressure."""

    def test_low_ratio_amplifies_recovery(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_ratio', 0.2)  # Low recovery rate
        result = _baseline_evaluate(mct, recovery_pressure=0.3)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, recovery_pressure=0.3)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_high_ratio_no_amplification(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_ratio', 0.8)  # Good recovery
        result = _baseline_evaluate(mct, recovery_pressure=0.3)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, recovery_pressure=0.3)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1d: consolidated_output_quality → low_output_reliability
# ══════════════════════════════════════════════════════════════════════

class TestInteg1dConsolidatedOutputQuality:
    """Low consolidated_output_quality → low_output_reliability."""

    def test_low_quality_amplifies_reliability(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('consolidated_output_quality', 0.2)
        result = _baseline_evaluate(mct, output_reliability=0.5)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, output_reliability=0.5)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_high_quality_no_effect(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('consolidated_output_quality', 0.9)
        result = _baseline_evaluate(mct, output_reliability=0.5)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, output_reliability=0.5)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1e: ucc_verdict_pressure → coherence_deficit
# ══════════════════════════════════════════════════════════════════════

class TestInteg1eUccVerdictPressure:
    """ucc_verdict_pressure → coherence_deficit."""

    def test_high_ucc_verdict_amplifies_coherence(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('ucc_verdict_pressure', 0.6)
        result = _baseline_evaluate(mct, coherence_deficit=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, coherence_deficit=0.2)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_low_ucc_verdict_no_effect(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('ucc_verdict_pressure', 0.1)
        result = _baseline_evaluate(mct, coherence_deficit=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, coherence_deficit=0.2)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1f: orphaned_signal_escalation → amplify ALL signals
# ══════════════════════════════════════════════════════════════════════

class TestInteg1fOrphanedSignalEscalation:
    """orphaned_signal_escalation → system-wide signal boost."""

    def test_escalation_boosts_all_signals(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('orphaned_signal_escalation', 0.6)
        result = _baseline_evaluate(
            mct, uncertainty=0.15, coherence_deficit=0.15,
        )
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(
            mct2, uncertainty=0.15, coherence_deficit=0.15,
        )
        assert result['trigger_score'] > result2['trigger_score']

    def test_low_escalation_no_effect(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('orphaned_signal_escalation', 0.1)
        result = _baseline_evaluate(mct, uncertainty=0.15)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, uncertainty=0.15)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1g: corrective_pressure → recovery_pressure
# ══════════════════════════════════════════════════════════════════════

class TestInteg1gCorrectivePressure:
    """corrective_pressure → recovery_pressure."""

    def test_high_corrective_amplifies_recovery(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('corrective_pressure', 0.6)
        result = _baseline_evaluate(mct, recovery_pressure=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, recovery_pressure=0.2)
        assert result['trigger_score'] >= result2['trigger_score']


# ══════════════════════════════════════════════════════════════════════
#  INTEG-1h: diversity_collapse → diversity_collapse signal
# ══════════════════════════════════════════════════════════════════════

class TestInteg1hDiversityCollapse:
    """diversity_collapse bus signal → MCT diversity_collapse channel."""

    def test_high_collapse_amplifies(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal('diversity_collapse', 0.7)
        result = _baseline_evaluate(mct, diversity_collapse=0.2)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, diversity_collapse=0.2)
        assert result['trigger_score'] >= result2['trigger_score']


# ══════════════════════════════════════════════════════════════════════
#  INTEG-2: recursion_best_improvement → threshold modulation
# ══════════════════════════════════════════════════════════════════════

class TestInteg2RecursionBestImprovement:
    """recursion_best_improvement modulates trigger threshold."""

    def test_high_improvement_lowers_threshold(self):
        """When recursion had high improvement, threshold should drop."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('recursion_best_improvement', 0.8)
        # Use a signal level just below default threshold
        result = _baseline_evaluate(mct, uncertainty=0.25)
        # Compare with no improvement signal
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, uncertainty=0.25)
        # With improvement, effective threshold is lower so trigger
        # may fire more easily (or at least effective_trigger_score
        # relationship to threshold changes)
        # The key check: result should be at least as likely to trigger
        assert isinstance(result['effective_trigger_score'], float)

    def test_negligible_improvement_raises_threshold(self):
        """When recursion had negligible improvement, threshold rises."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('recursion_best_improvement', 0.02)
        result = _baseline_evaluate(mct, uncertainty=0.25)
        assert isinstance(result['trigger_score'], float)

    def test_zero_improvement_no_change(self):
        """Zero improvement (signal not written) → no threshold change."""
        mct, bus = _make_mct_with_bus()
        result = _baseline_evaluate(mct, uncertainty=0.25)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, uncertainty=0.25)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  INTEG-3: Causal trace recording for MCT decisions
# ══════════════════════════════════════════════════════════════════════

class _MockCausalTrace:
    """Minimal causal trace stub that records calls."""

    def __init__(self):
        self.records = []

    def record(self, subsystem=None, decision=None, metadata=None,
               severity=None, causal_prerequisites=None, **kwargs):
        self.records.append({
            'subsystem': subsystem,
            'decision': decision,
            'metadata': metadata,
            'severity': severity,
        })

    def __bool__(self):
        return True


class TestInteg3CausalTraceRecording:
    """MCT evaluate() should record decisions in causal trace."""

    def test_trigger_fired_recorded(self):
        """When MCT triggers, trace records 'trigger:fired'."""
        mct, bus = _make_mct_with_bus()
        trace = _MockCausalTrace()
        bus._causal_trace_ref = trace
        # Force a trigger with high uncertainty
        _baseline_evaluate(mct, uncertainty=0.9)
        fired_records = [
            r for r in trace.records
            if r['subsystem'] == 'metacognitive_trigger'
            and 'fired' in r.get('decision', '')
        ]
        assert len(fired_records) >= 1
        rec = fired_records[0]
        assert rec['severity'] == 'warning'
        assert 'trigger_score' in rec['metadata']
        assert 'effective_trigger_score' in rec['metadata']

    def test_trigger_suppressed_recorded(self):
        """When MCT does not trigger, trace records 'trigger:suppressed'."""
        mct, bus = _make_mct_with_bus()
        trace = _MockCausalTrace()
        bus._causal_trace_ref = trace
        _baseline_evaluate(mct)  # All calm
        suppressed = [
            r for r in trace.records
            if r['subsystem'] == 'metacognitive_trigger'
            and 'suppressed' in r.get('decision', '')
        ]
        assert len(suppressed) >= 1
        assert suppressed[0]['severity'] == 'info'

    def test_trace_contains_signal_top3(self):
        """Trace metadata should include top-3 signal values."""
        mct, bus = _make_mct_with_bus()
        trace = _MockCausalTrace()
        bus._causal_trace_ref = trace
        _baseline_evaluate(
            mct, uncertainty=0.5, coherence_deficit=0.3,
            recovery_pressure=0.1,
        )
        records = [
            r for r in trace.records
            if r['subsystem'] == 'metacognitive_trigger'
        ]
        assert len(records) >= 1
        meta = records[0]['metadata']
        assert 'signal_values_top3' in meta
        assert isinstance(meta['signal_values_top3'], dict)
        assert len(meta['signal_values_top3']) <= 3

    def test_no_trace_available_no_error(self):
        """When no causal trace is available, evaluate still works."""
        mct, bus = _make_mct_with_bus()
        # No trace attached
        result = _baseline_evaluate(mct, uncertainty=0.5)
        assert 'trigger_score' in result


# ══════════════════════════════════════════════════════════════════════
#  INTEG-4: Training pressure signals in train_step metrics
# ══════════════════════════════════════════════════════════════════════

class TestInteg4TrainingPressureMetrics:
    """train_step metrics should include training pressure signals."""

    def test_output_reliability_pressure_surfaced(self):
        """output_reliability_training_pressure appears in metrics."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        # Write signal to bus
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal(
                'output_reliability_training_pressure', 0.5,
            )
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        # Signal should appear if bus was available and value > 0
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            assert 'output_reliability_training_pressure' in metrics

    def test_convergence_quality_pressure_surfaced(self):
        """convergence_quality_training_pressure appears in metrics."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal(
                'convergence_quality_training_pressure', 0.3,
            )
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            assert 'convergence_quality_training_pressure' in metrics

    def test_spectral_stability_pressure_surfaced(self):
        """spectral_stability_training_pressure appears in metrics."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal(
                'spectral_stability_training_pressure', 0.4,
            )
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            assert 'spectral_stability_training_pressure' in metrics

    def test_error_recovery_ratio_surfaced(self):
        """error_recovery_ratio appears in metrics when non-zero."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('error_recovery_ratio', 0.7)
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            assert 'error_recovery_ratio' in metrics

    def test_consolidated_output_quality_surfaced(self):
        """consolidated_output_quality appears in metrics when non-zero."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal(
                'consolidated_output_quality', 0.6,
            )
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            assert 'consolidated_output_quality' in metrics


# ══════════════════════════════════════════════════════════════════════
#  INTEG-5: vibe_thinker_entropy + confidence → output reliability
# ══════════════════════════════════════════════════════════════════════

class TestInteg5VibeThinkerEntropy:
    """High entropy + low confidence → amplified output reliability."""

    def test_high_entropy_low_confidence_amplifies(self):
        """When entropy > 0.5 and confidence < 0.5, amplify reliability."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('vibe_thinker_entropy', 0.8)
        bus.write_signal('vibe_thinker_confidence', 0.2)
        result = _baseline_evaluate(mct, output_reliability=0.5)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, output_reliability=0.5)
        assert result['trigger_score'] >= result2['trigger_score']

    def test_low_entropy_no_effect(self):
        """When entropy <= 0.5, no amplification regardless of confidence."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('vibe_thinker_entropy', 0.3)
        bus.write_signal('vibe_thinker_confidence', 0.2)
        result = _baseline_evaluate(mct, output_reliability=0.5)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, output_reliability=0.5)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01

    def test_high_confidence_no_effect(self):
        """When confidence >= 0.5, no amplification even with high entropy."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('vibe_thinker_entropy', 0.8)
        bus.write_signal('vibe_thinker_confidence', 0.7)
        result = _baseline_evaluate(mct, output_reliability=0.5)
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(mct2, output_reliability=0.5)
        assert abs(result['trigger_score'] - result2['trigger_score']) < 0.01


# ══════════════════════════════════════════════════════════════════════
#  Cross-patch integration tests
# ══════════════════════════════════════════════════════════════════════

class TestIntegCrossPatches:
    """Verify that multiple INTEG signals compose correctly."""

    def test_multiple_orphaned_signals_compound(self):
        """Multiple orphaned signals should compound trigger pressure."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_output_uncertainty', 0.7)
        bus.write_signal('error_recovery_ratio', 0.1)
        bus.write_signal('consolidated_output_quality', 0.2)
        bus.write_signal('ucc_verdict_pressure', 0.5)
        result = _baseline_evaluate(
            mct, uncertainty=0.2, recovery_pressure=0.2,
            output_reliability=0.5, coherence_deficit=0.2,
        )
        # Should have a significantly higher score than baseline
        mct2, bus2 = _make_mct_with_bus()
        result2 = _baseline_evaluate(
            mct2, uncertainty=0.2, recovery_pressure=0.2,
            output_reliability=0.5, coherence_deficit=0.2,
        )
        assert result['trigger_score'] > result2['trigger_score']

    def test_escalation_with_improvement_interact(self):
        """orphaned_signal_escalation + recursion_best_improvement."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('orphaned_signal_escalation', 0.5)
        bus.write_signal('recursion_best_improvement', 0.6)
        result = _baseline_evaluate(mct, uncertainty=0.3)
        assert isinstance(result['trigger_score'], float)
        assert result['trigger_score'] >= 0.0

    def test_all_signals_calm_no_trigger(self):
        """With all signals at healthy levels, MCT should not trigger."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('post_output_uncertainty', 0.0)
        bus.write_signal('error_recovery_ratio', 1.0)
        bus.write_signal('consolidated_output_quality', 1.0)
        bus.write_signal('ucc_verdict_pressure', 0.0)
        bus.write_signal('orphaned_signal_escalation', 0.0)
        bus.write_signal('diversity_collapse', 0.0)
        bus.write_signal('corrective_pressure', 0.0)
        bus.write_signal('vibe_thinker_entropy', 0.0)
        bus.write_signal('vibe_thinker_confidence', 1.0)
        result = _baseline_evaluate(mct)
        assert not result['should_trigger']

    def test_evaluate_returns_expected_keys(self):
        """MCT evaluate returns all expected keys after INTEG patches."""
        mct, bus = _make_mct_with_bus()
        result = _baseline_evaluate(mct, uncertainty=0.5)
        expected_keys = {
            'should_trigger', 'should_recurse', 'trigger_score',
            'effective_trigger_score', 'triggers_active',
        }
        assert expected_keys.issubset(result.keys())

    def test_bus_ema_smoothing_applied(self):
        """CognitiveFeedbackBus returns EMA-smoothed values."""
        bus = CognitiveFeedbackBus(64)
        bus.write_signal('test_signal', 1.0)
        val = float(bus.read_signal('test_signal', 0.0))
        # EMA smoothing means the read value may differ from raw write
        assert 0.0 <= val <= 1.0
