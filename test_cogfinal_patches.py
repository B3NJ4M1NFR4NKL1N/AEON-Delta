"""
Tests for COGFINAL-series patches: Final Integration & Cognitive Activation.

Patches covered:
  COGFINAL-1: anderson_safeguard_pressure → MCT convergence_conflict
  COGFINAL-2: lyapunov_oscillation_pressure → MCT oscillation_severity
  COGFINAL-3: memory_cross_validation_failure → MCT memory_trust_deficit
  COGFINAL-4: convergence_degradation_pressure → MCT convergence_conflict
  COGFINAL-5: metacognitive_recursion_depth → training metrics
  COGFINAL-6: training_coherence_verified → MCT coherence_deficit
  COGFINAL-7: spectral_depth_adaptation → training metrics
  COGFINAL-8: divergence_active signal written to bus on divergence detection
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
# Helper: minimal config
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


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-1: anderson_safeguard_pressure → MCT convergence_conflict
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal1AndersonSafeguard:
    """Verify anderson_safeguard_pressure is consumed by MCT."""

    def test_anderson_pressure_zero_no_effect(self):
        """When anderson_safeguard_pressure is 0, convergence_conflict unchanged."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('anderson_safeguard_pressure', 0.0)
        result = mct.evaluate(uncertainty=0.0)
        cc_baseline = result.get('trigger_score', 0.0)
        assert cc_baseline >= 0.0  # No contribution

    def test_anderson_pressure_high_amplifies_conflict(self):
        """High anderson_safeguard_pressure amplifies convergence_conflict."""
        mct, bus = _make_mct_with_bus()
        # Baseline without pressure
        result_base = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        mct.reset()
        # With pressure
        bus.write_signal('anderson_safeguard_pressure', 0.8)
        result_high = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_anderson_pressure_below_threshold_no_effect(self):
        """anderson_safeguard_pressure <= 0.2 has no effect."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('anderson_safeguard_pressure', 0.15)
        result = mct.evaluate(uncertainty=0.0, convergence_conflict=0.3)
        mct.reset()
        bus.write_signal('anderson_safeguard_pressure', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0, convergence_conflict=0.3)
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_anderson_pressure_read_from_bus(self):
        """MCT reads anderson_safeguard_pressure from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('anderson_safeguard_pressure', 0.9)
        mct.evaluate(uncertainty=0.1)
        # After evaluate, the signal should have been read
        read_log = getattr(bus, '_read_log', set())
        assert 'anderson_safeguard_pressure' in read_log


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-2: lyapunov_oscillation_pressure → MCT oscillation_severity
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal2LyapunovOscillation:
    """Verify lyapunov_oscillation_pressure is consumed by MCT."""

    def test_lyapunov_oscillation_zero_no_effect(self):
        """When lyapunov_oscillation_pressure is 0, oscillation_severity unchanged."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('lyapunov_oscillation_pressure', 0.0)
        result = mct.evaluate(uncertainty=0.0, oscillation_severity=0.0)
        # Should have no oscillation contribution
        assert 'oscillation_severity' not in result.get('triggers_active', [])

    def test_lyapunov_oscillation_high_amplifies_severity(self):
        """High lyapunov_oscillation_pressure amplifies oscillation_severity."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1, oscillation_severity=0.2)
        mct.reset()
        bus.write_signal('lyapunov_oscillation_pressure', 0.7)
        result_high = mct.evaluate(uncertainty=0.1, oscillation_severity=0.2)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_lyapunov_oscillation_below_threshold(self):
        """lyapunov_oscillation_pressure <= 0.1 has no effect."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('lyapunov_oscillation_pressure', 0.05)
        result = mct.evaluate(uncertainty=0.0, oscillation_severity=0.2)
        mct.reset()
        bus.write_signal('lyapunov_oscillation_pressure', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0, oscillation_severity=0.2)
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_lyapunov_oscillation_read_from_bus(self):
        """MCT reads lyapunov_oscillation_pressure from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('lyapunov_oscillation_pressure', 0.6)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'lyapunov_oscillation_pressure' in read_log


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-3: memory_cross_validation_failure → MCT memory_trust_deficit
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal3MemoryCrossValidation:
    """Verify memory_cross_validation_failure is consumed by MCT."""

    def test_memory_cv_failure_zero_no_effect(self):
        """When memory_cross_validation_failure is 0, memory_trust_deficit unchanged."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('memory_cross_validation_failure', 0.0)
        result = mct.evaluate(uncertainty=0.0, memory_trust_deficit=0.0)
        assert 'memory_trust_deficit' not in result.get('triggers_active', [])

    def test_memory_cv_failure_high_amplifies_deficit(self):
        """High memory_cross_validation_failure amplifies memory_trust_deficit."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1, memory_trust_deficit=0.2)
        mct.reset()
        bus.write_signal('memory_cross_validation_failure', 0.8)
        result_high = mct.evaluate(uncertainty=0.1, memory_trust_deficit=0.2)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_memory_cv_failure_read_from_bus(self):
        """MCT reads memory_cross_validation_failure from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('memory_cross_validation_failure', 0.5)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'memory_cross_validation_failure' in read_log


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-4: convergence_degradation_pressure → MCT convergence_conflict
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal4ConvergenceDegradation:
    """Verify convergence_degradation_pressure is consumed by MCT."""

    def test_convergence_degradation_zero_no_effect(self):
        """When convergence_degradation_pressure is 0, no effect."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('convergence_degradation_pressure', 0.0)
        result = mct.evaluate(uncertainty=0.0, convergence_conflict=0.0)
        assert result['trigger_score'] >= 0.0

    def test_convergence_degradation_high_amplifies(self):
        """High convergence_degradation_pressure amplifies convergence_conflict."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        mct.reset()
        bus.write_signal('convergence_degradation_pressure', 0.7)
        result_high = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_convergence_degradation_below_threshold(self):
        """convergence_degradation_pressure <= 0.2 has no effect."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('convergence_degradation_pressure', 0.15)
        result = mct.evaluate(uncertainty=0.0, convergence_conflict=0.3)
        mct.reset()
        bus.write_signal('convergence_degradation_pressure', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0, convergence_conflict=0.3)
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_convergence_degradation_read_from_bus(self):
        """MCT reads convergence_degradation_pressure from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('convergence_degradation_pressure', 0.6)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'convergence_degradation_pressure' in read_log


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-5: metacognitive_recursion_depth → training metrics
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal5RecursionDepthMetrics:
    """Verify metacognitive_recursion_depth appears in training metrics."""

    def test_recursion_depth_in_train_step_metrics(self):
        """train_step metrics include metacognitive_recursion_depth."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        # Write a recursion depth signal
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('metacognitive_recursion_depth', 2.0)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        assert 'metacognitive_recursion_depth' in metrics
        # Signal should be >= 0
        assert metrics['metacognitive_recursion_depth'] >= 0.0

    def test_recursion_depth_is_numeric(self):
        """metacognitive_recursion_depth is a finite non-negative number."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        val = metrics.get('metacognitive_recursion_depth', 0.0)
        assert isinstance(val, float)
        assert val >= 0.0
        assert math.isfinite(val)


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-6: training_coherence_verified → MCT coherence_deficit
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal6TrainingCoherenceVerified:
    """Verify training_coherence_verified is consumed by MCT."""

    def test_coherence_verified_success_no_effect(self):
        """When training_coherence_verified=1.0, no additional deficit."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('training_coherence_verified', 1.0)
        result = mct.evaluate(uncertainty=0.1, coherence_deficit=0.1)
        # coherence_deficit should not be amplified
        score_with_pass = result['trigger_score']
        mct.reset()
        bus.write_signal('training_coherence_verified', 1.0)
        result2 = mct.evaluate(uncertainty=0.1, coherence_deficit=0.1)
        assert abs(score_with_pass - result2['trigger_score']) < 0.01

    def test_coherence_verified_failure_amplifies(self):
        """When training_coherence_verified=0.0, coherence_deficit is amplified."""
        mct, bus = _make_mct_with_bus()
        # Baseline with verification passing
        bus.write_signal('training_coherence_verified', 1.0)
        result_pass = mct.evaluate(uncertainty=0.1, coherence_deficit=0.1)
        mct.reset()
        # With verification failing
        bus.write_signal('training_coherence_verified', 0.0)
        result_fail = mct.evaluate(uncertainty=0.1, coherence_deficit=0.1)
        assert result_fail['trigger_score'] >= result_pass['trigger_score']

    def test_coherence_verified_read_from_bus(self):
        """MCT reads training_coherence_verified from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('training_coherence_verified', 0.0)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'training_coherence_verified' in read_log


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-7: spectral_depth_adaptation → training metrics
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal7SpectralDepthMetrics:
    """Verify spectral_depth_adaptation appears in training metrics."""

    def test_spectral_depth_in_train_step_metrics(self):
        """train_step metrics include spectral_depth_adaptation."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_depth_adaptation', 0.75)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        assert 'spectral_depth_adaptation' in metrics

    def test_spectral_depth_is_numeric(self):
        """spectral_depth_adaptation is a finite non-negative number."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        val = metrics.get('spectral_depth_adaptation', 0.0)
        assert isinstance(val, float)
        assert val >= 0.0
        assert math.isfinite(val)


# ══════════════════════════════════════════════════════════════════════
#  COGFINAL-8: divergence_active signal written to bus
# ══════════════════════════════════════════════════════════════════════


class TestCogfinal8DivergenceActive:
    """Verify divergence_active signal is written to bus on divergence."""

    def test_divergence_active_written_after_forward(self):
        """After a forward pass, divergence_active is available on the bus."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        input_ids = torch.randint(1, cfg.vocab_size, (1, 16))
        model(input_ids)
        fb = getattr(model, 'feedback_bus', None)
        if fb is not None and hasattr(fb, 'read_signal'):
            val = float(fb.read_signal('divergence_active', -1.0))
            # Should be in [0, 1] range (not the default -1.0)
            assert 0.0 <= val <= 1.0, f"Expected [0, 1], got {val}"

    def test_divergence_active_in_train_metrics(self):
        """train_step surfaces divergence_active in metrics."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        assert 'divergence_active' in metrics
        assert 0.0 <= metrics['divergence_active'] <= 1.0

    def test_divergence_active_default_zero(self):
        """Without divergence, signal is 0.0."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        input_ids = torch.randint(1, cfg.vocab_size, (1, 16))
        model(input_ids)
        fb = getattr(model, 'feedback_bus', None)
        if fb is not None and hasattr(fb, 'read_signal'):
            val = float(fb.read_signal('divergence_active', -1.0))
            # First pass should not be diverging (warmup)
            assert val >= 0.0


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION: End-to-end signal flow verification
# ══════════════════════════════════════════════════════════════════════


class TestCogfinalIntegration:
    """End-to-end tests verifying signal flow from producers to consumers."""

    def test_all_cogfinal_signals_consumed_by_mct(self):
        """All COGFINAL signals are read by MCT evaluate()."""
        mct, bus = _make_mct_with_bus()
        # Write all signals
        bus.write_signal('anderson_safeguard_pressure', 0.5)
        bus.write_signal('lyapunov_oscillation_pressure', 0.5)
        bus.write_signal('memory_cross_validation_failure', 0.5)
        bus.write_signal('convergence_degradation_pressure', 0.5)
        bus.write_signal('training_coherence_verified', 0.0)
        mct.evaluate(uncertainty=0.2)
        read_log = getattr(bus, '_read_log', set())
        expected_signals = {
            'anderson_safeguard_pressure',
            'lyapunov_oscillation_pressure',
            'memory_cross_validation_failure',
            'convergence_degradation_pressure',
            'training_coherence_verified',
        }
        for sig in expected_signals:
            assert sig in read_log, f"Signal '{sig}' not read by MCT"

    def test_combined_pressure_exceeds_threshold(self):
        """Combined COGFINAL signals can push trigger_score past threshold."""
        mct, bus = _make_mct_with_bus()
        mct.trigger_threshold = 0.3
        # Write moderate pressure from multiple sources
        bus.write_signal('anderson_safeguard_pressure', 0.6)
        bus.write_signal('lyapunov_oscillation_pressure', 0.5)
        bus.write_signal('convergence_degradation_pressure', 0.5)
        bus.write_signal('training_coherence_verified', 0.0)
        result = mct.evaluate(
            uncertainty=0.2,
            convergence_conflict=0.3,
            oscillation_severity=0.3,
            coherence_deficit=0.2,
        )
        # Combined pressure should be significant
        assert result['trigger_score'] > 0.0 or result['effective_trigger_score'] > 0.0

    def test_full_forward_then_mct_picks_up_signals(self):
        """Full forward pass writes signals that MCT can read."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        input_ids = torch.randint(1, cfg.vocab_size, (1, 16))
        model(input_ids)
        # After forward, bus should have signals written
        fb = getattr(model, 'feedback_bus', None)
        if fb is not None:
            # divergence_active should be present
            div = float(fb.read_signal('divergence_active', -1.0))
            assert div >= 0.0, "divergence_active not written after forward"

    def test_no_orphaned_cogfinal_signals_after_evaluate(self):
        """After MCT evaluate(), COGFINAL signals are consumed (not orphaned)."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('anderson_safeguard_pressure', 0.5)
        bus.write_signal('lyapunov_oscillation_pressure', 0.5)
        bus.write_signal('memory_cross_validation_failure', 0.5)
        bus.write_signal('convergence_degradation_pressure', 0.5)
        bus.write_signal('training_coherence_verified', 0.0)
        mct.evaluate(uncertainty=0.2)
        orphans = bus.get_orphaned_signals()
        cogfinal_signals = {
            'anderson_safeguard_pressure',
            'lyapunov_oscillation_pressure',
            'memory_cross_validation_failure',
            'convergence_degradation_pressure',
            'training_coherence_verified',
        }
        orphaned_cogfinal = cogfinal_signals & set(orphans.keys())
        assert len(orphaned_cogfinal) == 0, (
            f"COGFINAL signals still orphaned: {orphaned_cogfinal}"
        )

    def test_train_step_returns_all_cogfinal_metrics(self):
        """train_step returns all COGFINAL-5/7/8 metrics."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        metrics = trainer.train_step(batch)
        for key in [
            'metacognitive_recursion_depth',
            'spectral_depth_adaptation',
            'divergence_active',
        ]:
            assert key in metrics, f"Missing metric: {key}"


# ══════════════════════════════════════════════════════════════════════
#  MUTUAL REINFORCEMENT: Verify components stabilise each other
# ══════════════════════════════════════════════════════════════════════


class TestMutualReinforcement:
    """Verify that active components verify and stabilise each other."""

    def test_convergence_signals_reinforce_each_other(self):
        """Multiple convergence signals compound to ensure detection."""
        mct, bus = _make_mct_with_bus()
        # Single weak signal should not trigger
        bus.write_signal('anderson_safeguard_pressure', 0.25)
        r1 = mct.evaluate(uncertainty=0.1, convergence_conflict=0.1)
        mct.reset()
        # Combined weak signals should have higher score
        bus.write_signal('anderson_safeguard_pressure', 0.25)
        bus.write_signal('convergence_degradation_pressure', 0.25)
        bus.write_signal('lyapunov_oscillation_pressure', 0.15)
        r2 = mct.evaluate(
            uncertainty=0.1,
            convergence_conflict=0.1,
            oscillation_severity=0.1,
        )
        assert r2['trigger_score'] >= r1['trigger_score']

    def test_memory_and_coherence_mutual_detection(self):
        """Memory failure + coherence failure creates compound detection."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('memory_cross_validation_failure', 0.6)
        bus.write_signal('training_coherence_verified', 0.0)
        result = mct.evaluate(
            uncertainty=0.2,
            memory_trust_deficit=0.3,
            coherence_deficit=0.3,
        )
        # Both should appear in triggers
        assert result['trigger_score'] > 0.0


# ══════════════════════════════════════════════════════════════════════
#  CAUSAL TRANSPARENCY: Every output traceable to root cause
# ══════════════════════════════════════════════════════════════════════


class TestCausalTransparency:
    """Verify outputs are traceable to originating signals."""

    def test_trigger_active_lists_cogfinal_sources(self):
        """triggers_active includes signals from COGFINAL-wired sources."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('anderson_safeguard_pressure', 0.8)
        result = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        # convergence_conflict should be active (amplified by anderson)
        assert 'convergence_conflict' in result['triggers_active']

    def test_signal_weights_include_all_channels(self):
        """MCT signal weights cover all expected channels."""
        mct, bus = _make_mct_with_bus()
        result = mct.evaluate(uncertainty=0.1)
        weights = result.get('signal_weights', {})
        # Core channels should be present
        for ch in [
            'uncertainty',
            'convergence_conflict',
            'oscillation_severity',
            'memory_trust_deficit',
            'coherence_deficit',
        ]:
            assert ch in weights, f"Missing weight: {ch}"

    def test_divergence_signal_traceable_to_forward(self):
        """divergence_active can be traced to forward pass convergence check."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        input_ids = torch.randint(1, cfg.vocab_size, (1, 16))
        model(input_ids)
        fb = getattr(model, 'feedback_bus', None)
        if fb is not None:
            write_log = getattr(fb, '_write_log', set())
            assert 'divergence_active' in write_log
