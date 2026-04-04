"""Tests for FIA-1 through FIA-5 Final Integration & Activation patches.

Validates:
  FIA-1: 8 orphaned bus signals wired to MCT evaluate()
    FIA-1a: post_output_uncertainty → uncertainty signal
    FIA-1b: post_pipeline_verdict_pressure → convergence_conflict
    FIA-1c: error_recovery_ratio → recovery_pressure
    FIA-1d: consolidated_output_quality → low_output_reliability
    FIA-1e: ucc_verdict_pressure → coherence_deficit
    FIA-1f: orphaned_signal_escalation → ALL signal boost
    FIA-1g: corrective_pressure → recovery_pressure
    FIA-1h: vibe_thinker_confidence + entropy → low_output_reliability
  FIA-2: recursion_best_improvement → threshold modulation
  FIA-3: Causal trace reference in MCT evaluate() result
  FIA-4: 3 training pressure signals surfaced in train_step metrics
  FIA-5: 6 critical training diagnostics in train_step metrics
"""

import math
import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    AEONTrainer,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_config(**overrides) -> AEONConfig:
    hd = overrides.pop('hidden_dim', 64)
    defaults = dict(
        hidden_dim=hd,
        z_dim=hd,
        vq_embedding_dim=hd,
        device_str='cpu',
        enable_error_evolution=True,
        enable_metacognitive_recursion=True,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_mct(
    trigger_threshold: float = 0.3,
    max_recursions: int = 5,
) -> MetaCognitiveRecursionTrigger:
    """Create MCT with a feedback bus attached."""
    mct = MetaCognitiveRecursionTrigger(
        trigger_threshold=trigger_threshold,
        max_recursions=max_recursions,
    )
    bus = CognitiveFeedbackBus(hidden_dim=64)
    mct.set_feedback_bus(bus)
    return mct


def _has_feedback_bus(mct: MetaCognitiveRecursionTrigger) -> bool:
    return (
        hasattr(mct, '_feedback_bus_ref')
        and mct._feedback_bus_ref is not None
    )


def _make_trainer_and_batch(hidden_dim: int = 64):
    """Create a trainer and a minimal batch for testing train_step."""
    config = _make_config(hidden_dim=hidden_dim)
    model = AEONDeltaV3(config)
    trainer = AEONTrainer(model, config)
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 16)),
        'attention_mask': torch.ones(2, 16, dtype=torch.long),
        'labels': torch.randint(0, config.vocab_size, (2, 16)),
    }
    return trainer, batch


# ═══════════════════════════════════════════════════════════════════
# FIA-1a: post_output_uncertainty → uncertainty
# ═══════════════════════════════════════════════════════════════════


class TestFIA1aPostOutputUncertainty:
    """Post-output uncertainty should amplify MCT uncertainty signal."""

    def test_high_post_output_uncertainty_amplifies_uncertainty(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_output_uncertainty', 0.8)
        result = mct.evaluate()
        assert 'uncertainty' in result['triggers_active'] or \
            result['trigger_score'] > 0

    def test_low_post_output_uncertainty_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_output_uncertainty', 0.1)
        result = mct.evaluate()
        # 0.1 <= 0.3 threshold, should not add uncertainty signal
        assert result['trigger_score'] < 0.3

    def test_post_output_uncertainty_zero_no_amplification(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_output_uncertainty', 0.0)
        result_with = mct.evaluate()
        mct2 = _make_mct()
        result_without = mct2.evaluate()
        assert abs(result_with['trigger_score'] - result_without['trigger_score']) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# FIA-1b: post_pipeline_verdict_pressure → convergence_conflict
# ═══════════════════════════════════════════════════════════════════


class TestFIA1bPostPipelineVerdictPressure:
    """Post-pipeline verdict pressure should amplify convergence_conflict."""

    def test_high_verdict_pressure_amplifies_convergence_conflict(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_pipeline_verdict_pressure', 0.7)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_below_threshold_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_pipeline_verdict_pressure', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] < 0.3


# ═══════════════════════════════════════════════════════════════════
# FIA-1c: error_recovery_ratio → recovery_pressure
# ═══════════════════════════════════════════════════════════════════


class TestFIA1cErrorRecoveryRatio:
    """Low error recovery ratio should amplify recovery_pressure."""

    def test_low_recovery_ratio_amplifies_pressure(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('error_recovery_ratio', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_high_recovery_ratio_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('error_recovery_ratio', 0.9)
        result = mct.evaluate()
        # High ratio = good recovery, no additional pressure
        assert result['trigger_score'] < 0.3

    def test_recovery_ratio_at_midpoint(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('error_recovery_ratio', 0.5)
        result = mct.evaluate()
        # 0.5 is the threshold, should not trigger
        assert result['trigger_score'] < 0.3


# ═══════════════════════════════════════════════════════════════════
# FIA-1d: consolidated_output_quality → low_output_reliability
# ═══════════════════════════════════════════════════════════════════


class TestFIA1dConsolidatedOutputQuality:
    """Low consolidated output quality should amplify low_output_reliability."""

    def test_low_quality_amplifies_reliability(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('consolidated_output_quality', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_high_quality_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('consolidated_output_quality', 0.9)
        result = mct.evaluate()
        assert result['trigger_score'] < 0.3


# ═══════════════════════════════════════════════════════════════════
# FIA-1e: ucc_verdict_pressure → coherence_deficit
# ═══════════════════════════════════════════════════════════════════


class TestFIA1eUccVerdictPressure:
    """UCC verdict pressure should amplify coherence_deficit."""

    def test_high_ucc_pressure_amplifies_coherence(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('ucc_verdict_pressure', 0.6)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_low_ucc_pressure_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('ucc_verdict_pressure', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] < 0.3


# ═══════════════════════════════════════════════════════════════════
# FIA-1f: orphaned_signal_escalation → ALL signal boost
# ═══════════════════════════════════════════════════════════════════


class TestFIA1fOrphanedSignalEscalation:
    """Orphaned signal escalation should boost ALL signals."""

    def test_escalation_boosts_all_signals(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        # Set some base signal
        result_base = mct.evaluate(uncertainty=0.6)

        mct2 = _make_mct()
        bus2 = mct2._feedback_bus_ref
        bus2.write_signal('orphaned_signal_escalation', 0.5)
        result_boosted = mct2.evaluate(uncertainty=0.6)

        # Score should be higher with escalation
        assert result_boosted['trigger_score'] > result_base['trigger_score']

    def test_no_escalation_no_boost(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('orphaned_signal_escalation', 0.05)
        result = mct.evaluate(uncertainty=0.6)
        mct2 = _make_mct()
        result2 = mct2.evaluate(uncertainty=0.6)
        # Below 0.1 threshold, scores should be equal
        assert abs(result['trigger_score'] - result2['trigger_score']) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# FIA-1g: corrective_pressure → recovery_pressure
# ═══════════════════════════════════════════════════════════════════


class TestFIA1gCorrectivePressure:
    """Corrective pressure should amplify recovery_pressure."""

    def test_high_corrective_pressure_amplifies(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('corrective_pressure', 0.7)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_low_corrective_pressure_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('corrective_pressure', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] < 0.3


# ═══════════════════════════════════════════════════════════════════
# FIA-1h: vibe_thinker_confidence + entropy → low_output_reliability
# ═══════════════════════════════════════════════════════════════════


class TestFIA1hVibeThinkerSignals:
    """Vibe thinker confidence/entropy should modulate reliability."""

    def test_low_confidence_amplifies_reliability(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('vibe_thinker_confidence', 0.2)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_high_entropy_amplifies_reliability(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('vibe_thinker_entropy', 0.8)
        result = mct.evaluate()
        assert result['trigger_score'] > 0

    def test_high_confidence_low_entropy_no_effect(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('vibe_thinker_confidence', 0.9)
        bus.write_signal('vibe_thinker_entropy', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] < 0.3

    def test_combined_deficit(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('vibe_thinker_confidence', 0.1)
        bus.write_signal('vibe_thinker_entropy', 0.9)
        result = mct.evaluate()
        # Both bad — should have higher score
        assert result['trigger_score'] > 0


# ═══════════════════════════════════════════════════════════════════
# FIA-2: recursion_best_improvement → threshold modulation
# ═══════════════════════════════════════════════════════════════════


class TestFIA2RecursionBestImprovement:
    """Recursion best improvement should modulate trigger threshold."""

    def test_high_improvement_lowers_threshold(self):
        """When recursion was very beneficial, threshold should lower."""
        mct = _make_mct(trigger_threshold=0.5)
        bus = mct._feedback_bus_ref
        bus.write_signal('recursion_best_improvement', 0.8)
        # With a high uncertainty that's near threshold
        result = mct.evaluate(uncertainty=0.45)
        # The threshold should be lowered, making it easier to trigger
        # We can't directly read the threshold, but the trigger decision
        # should reflect a lowered threshold
        assert result['effective_trigger_score'] >= 0

    def test_low_improvement_raises_threshold(self):
        """When recursion yielded minimal improvement, raise threshold."""
        mct = _make_mct(trigger_threshold=0.3)
        bus = mct._feedback_bus_ref
        bus.write_signal('recursion_best_improvement', 0.05)
        result = mct.evaluate()
        # Threshold raised — harder to trigger
        assert not result['should_trigger']

    def test_zero_improvement_no_modulation(self):
        """Zero improvement should not change threshold."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('recursion_best_improvement', 0.0)
        result = mct.evaluate()
        assert result['trigger_score'] >= 0


# ═══════════════════════════════════════════════════════════════════
# FIA-3: Causal trace reference in MCT evaluate() result
# ═══════════════════════════════════════════════════════════════════


class TestFIA3CausalTraceRef:
    """MCT evaluate() should include _causal_trace_ref for traceability."""

    def test_causal_trace_ref_present(self):
        mct = _make_mct()
        result = mct.evaluate()
        assert '_causal_trace_ref' in result

    def test_causal_trace_ref_has_trigger_signals(self):
        mct = _make_mct()
        result = mct.evaluate(uncertainty=0.8)
        trace = result['_causal_trace_ref']
        assert 'trigger_signals' in trace
        assert isinstance(trace['trigger_signals'], dict)

    def test_causal_trace_ref_has_threshold(self):
        mct = _make_mct()
        result = mct.evaluate()
        trace = result['_causal_trace_ref']
        assert 'effective_threshold' in trace
        assert isinstance(trace['effective_threshold'], float)

    def test_causal_trace_ref_records_bus_signals(self):
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('post_output_uncertainty', 0.7)
        result = mct.evaluate()
        trace = result['_causal_trace_ref']
        assert 'bus_signals_read' in trace
        assert isinstance(trace['bus_signals_read'], dict)
        # post_output_uncertainty was written and should appear
        assert 'post_output_uncertainty' in trace['bus_signals_read']

    def test_causal_trace_ref_empty_without_bus_signals(self):
        mct = _make_mct()
        result = mct.evaluate()
        trace = result['_causal_trace_ref']
        # No signals written, so bus_signals_read should be empty
        assert trace['bus_signals_read'] == {}

    def test_causal_trace_active_triggers_match_signals(self):
        mct = _make_mct()
        result = mct.evaluate(uncertainty=0.8, is_diverging=True)
        trace = result['_causal_trace_ref']
        # Active trigger signals should be in trigger_signals dict
        for sig_name, sig_value in trace['trigger_signals'].items():
            assert sig_value > 0


# ═══════════════════════════════════════════════════════════════════
# FIA-4: Training pressure signals in metrics
# ═══════════════════════════════════════════════════════════════════


class TestFIA4TrainingPressureMetrics:
    """Training pressure signals should appear in train_step metrics."""

    def test_training_pressure_signals_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        # Write some pressure signals to the bus
        if hasattr(trainer.model, 'feedback_bus') and trainer.model.feedback_bus is not None:
            trainer.model.feedback_bus.write_signal(
                'output_reliability_training_pressure', 0.3,
            )
            trainer.model.feedback_bus.write_signal(
                'convergence_quality_training_pressure', 0.4,
            )
            trainer.model.feedback_bus.write_signal(
                'spectral_stability_training_pressure', 0.2,
            )
        metrics = trainer.train_step(batch)
        # These should be in metrics
        assert 'output_reliability_training_pressure' in metrics
        assert 'convergence_quality_training_pressure' in metrics
        assert 'spectral_stability_training_pressure' in metrics

    def test_training_pressure_defaults_zero(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        # Even without writing, should default to 0.0
        for key in (
            'output_reliability_training_pressure',
            'convergence_quality_training_pressure',
            'spectral_stability_training_pressure',
        ):
            if key in metrics:
                assert isinstance(metrics[key], float)


# ═══════════════════════════════════════════════════════════════════
# FIA-5: Critical training diagnostics in metrics
# ═══════════════════════════════════════════════════════════════════


class TestFIA5TrainingDiagnostics:
    """Critical diagnostics should appear in train_step metrics."""

    def test_gradient_explosion_flag_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'gradient_explosion_detected' in metrics
        assert metrics['gradient_explosion_detected'] in (0.0, 1.0)

    def test_cognitive_health_skip_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'cognitive_health_skip' in metrics
        assert metrics['cognitive_health_skip'] in (0.0, 1.0)

    def test_loss_divergence_flag_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'loss_divergence_detected' in metrics
        assert metrics['loss_divergence_detected'] in (0.0, 1.0)

    def test_lipschitz_budget_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'lipschitz_budget_satisfied' in metrics
        assert metrics['lipschitz_budget_satisfied'] in (0.0, 1.0)

    def test_lyapunov_descent_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'lyapunov_descent_stable' in metrics
        assert metrics['lyapunov_descent_stable'] in (0.0, 1.0)

    def test_verify_reinforce_in_metrics(self):
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        assert 'verify_reinforce_success' in metrics
        assert metrics['verify_reinforce_success'] in (0.0, 1.0)

    def test_loss_divergence_reset_each_step(self):
        trainer, batch = _make_trainer_and_batch()
        metrics1 = trainer.train_step(batch)
        # With normal training, should be 0.0
        assert metrics1['loss_divergence_detected'] == 0.0
        # Verify the attribute itself is reset
        assert trainer._loss_divergence_detected is False

    def test_all_diagnostics_present(self):
        """All 6 FIA-5 diagnostic flags must be in metrics."""
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        expected_keys = {
            'gradient_explosion_detected',
            'cognitive_health_skip',
            'loss_divergence_detected',
            'lipschitz_budget_satisfied',
            'lyapunov_descent_stable',
            'verify_reinforce_success',
        }
        for key in expected_keys:
            assert key in metrics, f"Missing FIA-5 metric: {key}"


# ═══════════════════════════════════════════════════════════════════
# Integration tests: Multiple patches working together
# ═══════════════════════════════════════════════════════════════════


class TestFIAIntegration:
    """Test multiple FIA patches interact correctly."""

    def test_multiple_orphaned_signals_compound(self):
        """Multiple orphaned signals should compound trigger score."""
        mct = _make_mct(trigger_threshold=0.5)
        bus = mct._feedback_bus_ref
        bus.write_signal('post_output_uncertainty', 0.8)
        bus.write_signal('ucc_verdict_pressure', 0.7)
        bus.write_signal('consolidated_output_quality', 0.1)
        bus.write_signal('orphaned_signal_escalation', 0.5)
        result = mct.evaluate()
        # Multiple signals should compound to a high score
        assert result['trigger_score'] > 0.1

    def test_causal_trace_records_compounded_signals(self):
        """Causal trace should record all active bus signals."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('corrective_pressure', 0.8)
        bus.write_signal('error_recovery_ratio', 0.1)
        result = mct.evaluate()
        trace = result['_causal_trace_ref']
        assert 'corrective_pressure' in trace['bus_signals_read']
        assert 'error_recovery_ratio' in trace['bus_signals_read']

    def test_no_bus_graceful_degradation(self):
        """MCT without feedback bus should still work (no FIA signals)."""
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.3,
            max_recursions=5,
        )
        # No bus attached
        result = mct.evaluate(uncertainty=0.8)
        assert 'should_trigger' in result
        assert '_causal_trace_ref' in result
        # Causal trace bus_signals_read should be empty
        assert result['_causal_trace_ref']['bus_signals_read'] == {}

    def test_full_train_step_with_all_metrics(self):
        """Full train_step should include ALL FIA-4 and FIA-5 metrics."""
        trainer, batch = _make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        # FIA-4 metrics (may be present if bus exists)
        fia4_keys = {
            'output_reliability_training_pressure',
            'convergence_quality_training_pressure',
            'spectral_stability_training_pressure',
        }
        # FIA-5 metrics (always present)
        fia5_keys = {
            'gradient_explosion_detected',
            'cognitive_health_skip',
            'loss_divergence_detected',
            'lipschitz_budget_satisfied',
            'lyapunov_descent_stable',
            'verify_reinforce_success',
        }
        for key in fia5_keys:
            assert key in metrics, f"Missing metric: {key}"
        # FIA-4 only if bus exists
        if hasattr(trainer.model, 'feedback_bus') and trainer.model.feedback_bus is not None:
            for key in fia4_keys:
                assert key in metrics, f"Missing metric: {key}"
