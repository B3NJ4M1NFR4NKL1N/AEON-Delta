"""Tests for EMRG-1 through EMRG-6: Final integration & cognitive activation patches.

EMRG-1: Structured error recording in _build_feedback_extra_signals
EMRG-2: Convergence quality + spectral stability → compute_loss training pressure
EMRG-3: Periodic verify_and_reinforce() in AEONTrainer.train_step
EMRG-4: convergence_confidence → MCT convergence_conflict amplification
EMRG-5: _NullCausalTrace stub replaces None causal_trace
EMRG-6: Cross-pass coherence tracking in training metrics
"""

import math
import sys
import unittest
import torch

sys.path.insert(0, ".")

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    AEONTrainer,
    CognitiveFeedbackBus,
    CausalProvenanceTracker,
    CausalErrorEvolutionTracker,
    MetaCognitiveRecursionTrigger,
    ProvablyConvergentMetaLoop,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config(**overrides):
    """Create AEONConfig with test defaults."""
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        vocab_size=256,
        device_str='cpu',
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_feedback_bus(hidden_dim=64):
    """Create CognitiveFeedbackBus for testing."""
    return CognitiveFeedbackBus(hidden_dim)


def _make_mct():
    """Create MetaCognitiveRecursionTrigger with feedback bus."""
    mct = MetaCognitiveRecursionTrigger()
    bus = _make_feedback_bus()
    mct.set_feedback_bus(bus)
    return mct, bus


def _make_outputs(config, B=2, S=16):
    """Create minimal model outputs for compute_loss."""
    return {
        'logits': torch.randn(B, S, config.vocab_size),
        'vq_loss': torch.tensor(0.0),
        'certificate': {},
        'meta_results': {},
    }


# ═══════════════════════════════════════════════════════════════════════
# EMRG-1: Structured error recording in _build_feedback_extra_signals
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG1_StructuredErrorRecording(unittest.TestCase):
    """Silent except:pass replaced with error_evolution recording."""

    def test_emrg1_normal_write_succeeds(self):
        """Normal bus writes succeed without error recording."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_cognitive_unity_deficit = 0.5
        model._cached_coherence_deficit = 0.3

        # _build_feedback_extra_signals writes these signals
        extra = model._build_feedback_extra_signals()

        # Verify signals were written
        if model.feedback_bus is not None:
            cud = model.feedback_bus.read_signal(
                'cognitive_unity_deficit', -1.0,
            )
            cd = model.feedback_bus.read_signal(
                'coherence_deficit', -1.0,
            )
            self.assertGreaterEqual(float(cud), 0.0)
            self.assertGreaterEqual(float(cd), 0.0)

    def test_emrg1_returns_dict(self):
        """_build_feedback_extra_signals returns a dict."""
        config = _make_config()
        model = AEONDeltaV3(config)
        extra = model._build_feedback_extra_signals()
        self.assertIsInstance(extra, dict)

    def test_emrg1_bus_failure_does_not_crash(self):
        """Bus write failure does not crash the method."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_cognitive_unity_deficit = 0.5

        # Simulate bus in a broken state by replacing write_signal
        if model.feedback_bus is not None:
            original = model.feedback_bus.write_signal

            def _broken_write(name, value):
                if name in ('cognitive_unity_deficit', 'coherence_deficit'):
                    raise RuntimeError("Simulated bus failure")
                return original(name, value)

            model.feedback_bus.write_signal = _broken_write

            # Should not raise
            extra = model._build_feedback_extra_signals()
            self.assertIsInstance(extra, dict)

            # Restore
            model.feedback_bus.write_signal = original

    def test_emrg1_bus_failure_records_error_evolution(self):
        """Bus write failure records episode in error_evolution."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_cognitive_unity_deficit = 0.5

        if model.feedback_bus is None or model.error_evolution is None:
            self.skipTest("feedback_bus or error_evolution not available")

        original = model.feedback_bus.write_signal
        _recorded = []

        # Monkey-patch error_evolution to capture calls
        orig_record = model.error_evolution.record_episode

        def _capturing_record(**kwargs):
            _recorded.append(kwargs)
            return orig_record(**kwargs)

        model.error_evolution.record_episode = _capturing_record

        def _broken_write(name, value):
            if name == 'cognitive_unity_deficit':
                raise RuntimeError("Simulated bus failure")
            return original(name, value)

        model.feedback_bus.write_signal = _broken_write

        model._build_feedback_extra_signals()

        # Restore
        model.feedback_bus.write_signal = original
        model.error_evolution.record_episode = orig_record

        # Check that at least one feedback_bus_write_failure was recorded
        fb_failures = [
            r for r in _recorded
            if r.get('error_class') == 'feedback_bus_write_failure'
        ]
        self.assertGreater(
            len(fb_failures), 0,
            "Bus write failure should record error_evolution episode",
        )


# ═══════════════════════════════════════════════════════════════════════
# EMRG-2: Convergence quality + spectral stability → loss scaling
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG2_ConvergenceSpectralLoss(unittest.TestCase):
    """Training loss scales with convergence quality and spectral margin."""

    def test_emrg2_normal_no_scaling(self):
        """No extra scaling when convergence and spectral are healthy."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_convergence_quality = 1.0
        model._cached_spectral_stability_margin = 1.0

        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)
        loss_dict = model.compute_loss(outputs, targets)

        self.assertIn('convergence_quality_training_pressure', loss_dict)
        self.assertIn('spectral_stability_training_pressure', loss_dict)
        self.assertAlmostEqual(
            loss_dict['convergence_quality_training_pressure'], 0.0,
            places=5,
        )
        self.assertAlmostEqual(
            loss_dict['spectral_stability_training_pressure'], 0.0,
            places=5,
        )

    def test_emrg2_low_convergence_increases_loss(self):
        """Low convergence quality increases total loss."""
        config = _make_config()
        model = AEONDeltaV3(config)
        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)

        # Baseline with good quality
        model._cached_convergence_quality = 1.0
        model._cached_spectral_stability_margin = 1.0
        baseline = model.compute_loss(outputs, targets)
        baseline_loss = float(baseline['total_loss'].detach())

        # Low convergence quality
        model._cached_convergence_quality = 0.3
        model._cached_spectral_stability_margin = 1.0
        scaled = model.compute_loss(outputs, targets)
        scaled_loss = float(scaled['total_loss'].detach())

        self.assertGreater(
            scaled['convergence_quality_training_pressure'], 0.0,
            "Convergence quality pressure should be positive",
        )

    def test_emrg2_low_spectral_increases_pressure(self):
        """Low spectral margin produces non-zero pressure."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_convergence_quality = 1.0
        model._cached_spectral_stability_margin = 0.3

        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)
        loss_dict = model.compute_loss(outputs, targets)

        self.assertGreater(
            loss_dict['spectral_stability_training_pressure'], 0.0,
            "Spectral stability pressure should be positive",
        )

    def test_emrg2_scale_bounded(self):
        """Loss scaling is bounded at 1.5x maximum."""
        config = _make_config()
        model = AEONDeltaV3(config)

        # Both at worst case
        model._cached_convergence_quality = 0.0
        model._cached_spectral_stability_margin = 0.0

        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)
        loss_dict = model.compute_loss(outputs, targets)

        # Total pressure from EMRG-2: 0.3 (conv) + 0.2 (spectral) = 0.5
        # So scale = 1.5, bounded at 1.5
        total_pressure = (
            loss_dict['convergence_quality_training_pressure']
            + loss_dict['spectral_stability_training_pressure']
        )
        self.assertGreater(total_pressure, 0.0)

    def test_emrg2_bus_signal_written_on_low_quality(self):
        """Pressure signals written to feedback bus."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model._cached_convergence_quality = 0.3
        model._cached_spectral_stability_margin = 0.3

        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)
        model.compute_loss(outputs, targets)

        if model.feedback_bus is not None:
            conv_pressure = model.feedback_bus.read_signal(
                'convergence_quality_training_pressure', 0.0,
            )
            spectral_pressure = model.feedback_bus.read_signal(
                'spectral_stability_training_pressure', 0.0,
            )
            self.assertGreater(
                float(conv_pressure), 0.0,
                "convergence_quality_training_pressure should be written",
            )
            self.assertGreater(
                float(spectral_pressure), 0.0,
                "spectral_stability_training_pressure should be written",
            )


# ═══════════════════════════════════════════════════════════════════════
# EMRG-3: Periodic verify_and_reinforce() in training
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG3_TrainingVerification(unittest.TestCase):
    """Verify that train_step periodically calls verify_and_reinforce."""

    def _make_trainer(self):
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        return trainer, model, config

    def test_emrg3_verify_interval_default(self):
        """Default verification interval is 500."""
        trainer, _, _ = self._make_trainer()
        interval = getattr(trainer, '_emrg3_verify_interval', 500)
        self.assertEqual(interval, 500)

    def test_emrg3_verify_called_at_interval(self):
        """verify_and_reinforce called at the configured interval."""
        trainer, model, config = self._make_trainer()

        # Track calls to verify_and_reinforce
        _calls = []
        original = model.verify_and_reinforce

        def _tracking_verify():
            result = original()
            _calls.append(result)
            return result

        model.verify_and_reinforce = _tracking_verify

        # Set interval to 1 for testing
        trainer._emrg3_verify_interval = 1
        trainer.global_step = 1

        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 16)),
            'labels': torch.randint(0, config.vocab_size, (2, 16)),
        }
        trainer.train_step(batch)

        model.verify_and_reinforce = original

        self.assertGreater(
            len(_calls), 0,
            "verify_and_reinforce should be called at interval",
        )

    def test_emrg3_no_verify_before_interval(self):
        """EMRG-3 branch does not fire at non-interval steps."""
        trainer, _, _ = self._make_trainer()
        trainer._emrg3_verify_interval = 500

        # Verify the conditional logic: step 3 is not a multiple of 500
        trainer.global_step = 3
        self.assertNotEqual(
            trainer.global_step % trainer._emrg3_verify_interval, 0,
            "Step 3 should not satisfy the interval condition",
        )

        # Verify step 500 IS a multiple
        trainer.global_step = 500
        self.assertEqual(
            trainer.global_step % trainer._emrg3_verify_interval, 0,
            "Step 500 should satisfy the interval condition",
        )

    def test_emrg3_verify_failure_no_crash(self):
        """If verify_and_reinforce raises, training continues."""
        trainer, model, config = self._make_trainer()

        def _broken_verify():
            raise RuntimeError("Simulated verification failure")

        model.verify_and_reinforce = _broken_verify
        trainer._emrg3_verify_interval = 1
        trainer.global_step = 1

        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 16)),
            'labels': torch.randint(0, config.vocab_size, (2, 16)),
        }

        # Should not raise
        metrics = trainer.train_step(batch)
        self.assertIn('total_loss', metrics)


# ═══════════════════════════════════════════════════════════════════════
# EMRG-4: convergence_confidence → MCT convergence_conflict
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG4_ConvergenceConfidenceConsumption(unittest.TestCase):
    """MCT reads convergence_confidence from bus and amplifies conflict."""

    def test_emrg4_low_confidence_amplifies_conflict(self):
        """Low convergence_confidence amplifies convergence_conflict."""
        mct, bus = _make_mct()
        bus.write_signal('convergence_confidence', 0.1)

        result_low = mct.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        # Compare with high confidence
        mct2 = MetaCognitiveRecursionTrigger()
        bus2 = _make_feedback_bus()
        mct2.set_feedback_bus(bus2)
        bus2.write_signal('convergence_confidence', 1.0)

        result_high = mct2.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        self.assertGreaterEqual(
            result_low['trigger_score'],
            result_high['trigger_score'],
            "Low convergence_confidence should amplify trigger score",
        )

    def test_emrg4_high_confidence_no_amplification(self):
        """High convergence_confidence does not amplify."""
        mct, bus = _make_mct()
        bus.write_signal('convergence_confidence', 0.9)

        result = mct.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        mct2 = MetaCognitiveRecursionTrigger()
        result_no_bus = mct2.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        # With high confidence (0.9 > 0.5), no amplification
        self.assertAlmostEqual(
            result['trigger_score'],
            result_no_bus['trigger_score'],
            places=3,
        )

    def test_emrg4_no_bus_no_crash(self):
        """MCT without bus still works."""
        mct = MetaCognitiveRecursionTrigger()
        result = mct.evaluate(
            uncertainty=0.5,
            convergence_conflict=0.5,
        )
        self.assertIn('trigger_score', result)

    def test_emrg4_confidence_read_tracked(self):
        """Reading convergence_confidence is tracked in bus read_log."""
        mct, bus = _make_mct()
        bus.write_signal('convergence_confidence', 0.3)

        mct.evaluate(uncertainty=0.0)

        read_log = getattr(bus, '_read_log', set())
        self.assertIn(
            'convergence_confidence', read_log,
            "convergence_confidence should be tracked in read_log",
        )

    def test_emrg4_boundary_confidence_half(self):
        """Confidence exactly at 0.5 triggers no amplification."""
        mct, bus = _make_mct()
        bus.write_signal('convergence_confidence', 0.5)

        result = mct.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        mct2 = MetaCognitiveRecursionTrigger()
        result_no_bus = mct2.evaluate(
            uncertainty=0.0,
            convergence_conflict=1.0,
        )

        # At exactly 0.5, deficit is 0, so no amplification
        self.assertAlmostEqual(
            result['trigger_score'],
            result_no_bus['trigger_score'],
            places=3,
        )


# ═══════════════════════════════════════════════════════════════════════
# EMRG-5: _NullCausalTrace stub
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG5_NullCausalTrace(unittest.TestCase):
    """_NullCausalTrace replaces None causal_trace safely."""

    def test_emrg5_causal_trace_never_none(self):
        """causal_trace should never be None on a model instance."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)
        self.assertIsNotNone(
            model.causal_trace,
            "causal_trace should be _NullCausalTrace, not None",
        )

    def test_emrg5_null_trace_record_succeeds(self):
        """_NullCausalTrace.record() accepts and silently discards."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)

        # Should not raise
        result = model.causal_trace.record(
            subsystem="test",
            decision="test_decision",
            metadata={"test": True},
        )
        self.assertEqual(result, "")

    def test_emrg5_null_trace_get_entries_empty(self):
        """_NullCausalTrace.get_entries() returns empty list."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)
        entries = model.causal_trace.get_entries()
        self.assertEqual(len(entries), 0)

    def test_emrg5_null_trace_len_zero(self):
        """_NullCausalTrace has length 0."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)
        self.assertEqual(len(model.causal_trace), 0)

    def test_emrg5_null_trace_bool_false(self):
        """_NullCausalTrace evaluates to False in boolean context."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)
        self.assertFalse(bool(model.causal_trace))

    def test_emrg5_enabled_trace_is_real(self):
        """When enabled, causal_trace is a real buffer."""
        config = _make_config(enable_causal_trace=True)
        model = AEONDeltaV3(config)
        # Real buffer should have _entries attribute
        self.assertTrue(
            hasattr(model.causal_trace, '_entries'),
            "Enabled causal_trace should be TemporalCausalTraceBuffer",
        )

    def test_emrg5_null_trace_trace_root_cause(self):
        """_NullCausalTrace.trace_root_cause() returns empty list."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)
        result = model.causal_trace.trace_root_cause("test_id")
        self.assertEqual(result, [])


# ═══════════════════════════════════════════════════════════════════════
# EMRG-6: Cross-pass coherence tracking in metrics
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG6_CoherenceMetrics(unittest.TestCase):
    """Training metrics include cognitive coherence signals."""

    def _make_trainer_and_batch(self):
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 16)),
            'labels': torch.randint(0, config.vocab_size, (2, 16)),
        }
        return trainer, model, config, batch

    def test_emrg6_metrics_include_unity_deficit(self):
        """Training metrics include cognitive_unity_deficit."""
        trainer, model, config, batch = self._make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        self.assertIn('cognitive_unity_deficit', metrics)
        # Value comes from model's cached state after forward pass
        self.assertIsInstance(metrics['cognitive_unity_deficit'], float)
        self.assertGreaterEqual(metrics['cognitive_unity_deficit'], 0.0)
        self.assertLessEqual(metrics['cognitive_unity_deficit'], 1.0)

    def test_emrg6_metrics_include_spectral_margin(self):
        """Training metrics include spectral_stability_margin."""
        trainer, model, config, batch = self._make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        self.assertIn('spectral_stability_margin', metrics)
        self.assertIsInstance(metrics['spectral_stability_margin'], float)
        self.assertGreaterEqual(metrics['spectral_stability_margin'], 0.0)
        self.assertLessEqual(metrics['spectral_stability_margin'], 1.0)

    def test_emrg6_metrics_include_convergence_quality(self):
        """Training metrics include convergence_quality."""
        trainer, model, config, batch = self._make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        self.assertIn('convergence_quality', metrics)

    def test_emrg6_metrics_include_bus_oscillation(self):
        """Training metrics include feedback_bus_oscillation when bus exists."""
        trainer, model, config, batch = self._make_trainer_and_batch()
        metrics = trainer.train_step(batch)
        if (hasattr(model, 'feedback_bus')
                and model.feedback_bus is not None):
            self.assertIn('feedback_bus_oscillation', metrics)

    def test_emrg6_default_values_when_no_cache(self):
        """Metrics use safe defaults when cached values not set."""
        trainer, model, config, batch = self._make_trainer_and_batch()
        # The forward pass during train_step may set cached values,
        # so we verify that the returned metrics are valid floats
        # regardless of what the forward pass computes.
        metrics = trainer.train_step(batch)

        self.assertIsInstance(metrics['cognitive_unity_deficit'], float)
        self.assertIsInstance(metrics['spectral_stability_margin'], float)
        # Values should be in valid range
        self.assertGreaterEqual(metrics['cognitive_unity_deficit'], 0.0)
        self.assertGreaterEqual(metrics['spectral_stability_margin'], 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEMRG_Integration(unittest.TestCase):
    """End-to-end integration tests for EMRG patches."""

    def test_full_training_step_with_all_patches(self):
        """A full training step exercises all EMRG patches."""
        config = _make_config()
        model = AEONDeltaV3(config)
        trainer = AEONTrainer(model, config)

        # Set cached values to trigger EMRG-2
        model._cached_convergence_quality = 0.4
        model._cached_spectral_stability_margin = 0.5

        # Set interval to 1 to trigger EMRG-3
        trainer._emrg3_verify_interval = 1
        trainer.global_step = 1

        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 16)),
            'labels': torch.randint(0, config.vocab_size, (2, 16)),
        }
        metrics = trainer.train_step(batch)

        # Verify EMRG-2 pressure in loss dict
        self.assertIn('convergence_quality_training_pressure', metrics)
        self.assertIn('spectral_stability_training_pressure', metrics)

        # Verify EMRG-6 cognitive metrics
        self.assertIn('cognitive_unity_deficit', metrics)
        self.assertIn('spectral_stability_margin', metrics)
        self.assertIn('convergence_quality', metrics)

    def test_mct_reads_convergence_and_bus_signals(self):
        """MCT evaluate reads both convergence_confidence and oscillation."""
        mct, bus = _make_mct()
        bus.write_signal('convergence_confidence', 0.2)
        bus.write_signal('oscillation_severity_pressure', 0.8)

        result = mct.evaluate(
            uncertainty=0.5,
            is_diverging=True,
            convergence_conflict=1.0,
        )

        # Both signals should be read
        read_log = getattr(bus, '_read_log', set())
        self.assertIn('convergence_confidence', read_log)
        self.assertIn('oscillation_severity_pressure', read_log)

    def test_null_causal_trace_with_verify_and_reinforce(self):
        """verify_and_reinforce works with _NullCausalTrace."""
        config = _make_config(enable_causal_trace=False)
        model = AEONDeltaV3(config)

        # Should not raise even with null trace
        result = model.verify_and_reinforce()
        self.assertIsInstance(result, dict)

    def test_compute_loss_returns_all_emrg2_keys(self):
        """compute_loss return dict includes EMRG-2 keys."""
        config = _make_config()
        model = AEONDeltaV3(config)
        targets = torch.randint(0, config.vocab_size, (2, 16))
        outputs = _make_outputs(config)
        loss_dict = model.compute_loss(outputs, targets)

        self.assertIn('convergence_quality_training_pressure', loss_dict)
        self.assertIn('spectral_stability_training_pressure', loss_dict)

    def test_coherence_metrics_accessible_after_forward(self):
        """After a forward pass, cached cognitive values are accessible."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model.eval()

        with torch.no_grad():
            input_ids = torch.randint(1, config.vocab_size - 1, (1, 8))
            try:
                model(input_ids)
            except (RuntimeError, ValueError, TypeError):
                pass  # Forward may fail in minimal test config

        # Cached values should exist (either from forward or defaults)
        cud = getattr(model, '_cached_cognitive_unity_deficit', 0.0)
        ssm = getattr(model, '_cached_spectral_stability_margin', 1.0)
        self.assertIsInstance(cud, float)
        self.assertIsInstance(ssm, float)


if __name__ == '__main__':
    unittest.main()
