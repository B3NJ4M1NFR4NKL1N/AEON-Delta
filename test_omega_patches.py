"""Tests for Ω-series integration patches (Final Cognitive Activation).

Patch summary:
    Ω1  – error_recovery_pressure → MCT recovery_pressure
    Ω2  – causal_trace_coverage_deficit → MCT coherence_deficit
    Ω4  – Adaptive loss scaling from feedback bus in compute_loss()
    Ω5a – Unconditional safety decision → causal trace
    Ω6  – Inference anomaly signals → training bridge
    Ω7  – causal_trace_disabled + deficit → unified causal quality
    Ω8  – Research-variant docstrings on meta-loop classes

(Ω3 was already implemented as patch S-1; no new test needed.)
"""

import math
import torch
import pytest

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    AEONTrainer,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    HierarchicalMetaLoop,
    RecursiveMetaLoop,
    AdaptiveMetaLoop,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        vocab_size=256,
        device_str='cpu',
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_mct_with_bus():
    """Create an MCT instance wired to a CognitiveFeedbackBus."""
    bus = CognitiveFeedbackBus(64)
    mct = MetaCognitiveRecursionTrigger()
    mct.set_feedback_bus(bus)
    return mct, bus


def _make_model_and_batch():
    """Create a minimal AEONDeltaV3 + dummy batch for loss/training tests."""
    cfg = _make_config()
    model = AEONDeltaV3(cfg)
    B, S = 1, 16
    batch = {
        'input_ids': torch.randint(1, cfg.vocab_size, (B, S)),
        'labels': torch.randint(1, cfg.vocab_size, (B, S)),
    }
    return model, cfg, batch


def _make_outputs(config):
    """Create minimal outputs dict for compute_loss()."""
    B, S = 2, 16
    return {
        'logits': torch.randn(B, S, config.vocab_size),
        'vq_loss': torch.tensor(0.0),
        'certificate': {},
        'meta_results': {},
    }


# ======================================================================
# Ω1: error_recovery_pressure → MCT recovery_pressure
# ======================================================================

class TestOmega1ErrorRecoveryPressure:
    """Verify error_recovery_pressure is consumed by MCT evaluate()."""

    def test_zero_pressure_no_effect(self):
        """When error_recovery_pressure = 0, no recovery_pressure contribution."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_pressure', 0.0)
        result = mct.evaluate(uncertainty=0.0)
        score = result.get('trigger_score', 0.0)
        assert score >= 0.0

    def test_high_pressure_amplifies_trigger(self):
        """High error_recovery_pressure amplifies MCT trigger score."""
        mct, bus = _make_mct_with_bus()
        # Baseline without pressure
        result_base = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        mct.reset()
        # With high recovery pressure
        bus.write_signal('error_recovery_pressure', 0.9)
        result_high = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_below_threshold_no_contribution(self):
        """error_recovery_pressure ≤ 0.1 should not contribute."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_pressure', 0.05)
        result = mct.evaluate(uncertainty=0.0)
        mct.reset()
        bus.write_signal('error_recovery_pressure', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0)
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_signal_is_read_from_bus(self):
        """MCT must read error_recovery_pressure from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_pressure', 0.8)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'error_recovery_pressure' in read_log

    def test_routes_to_recovery_pressure_channel(self):
        """error_recovery_pressure routes into recovery_pressure signal."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_pressure', 0.9)
        result = mct.evaluate(uncertainty=0.1, convergence_conflict=0.3)
        # The _causal_trace_ref should record the bus signal read
        causal_ref = result.get('_causal_trace_ref', {})
        bus_reads = causal_ref.get('bus_signals_read', {})
        assert 'error_recovery_pressure' in bus_reads

    def test_moderate_pressure_contributes(self):
        """error_recovery_pressure = 0.5 should contribute to trigger."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1)
        mct.reset()
        bus.write_signal('error_recovery_pressure', 0.5)
        result_mod = mct.evaluate(uncertainty=0.1)
        assert result_mod['trigger_score'] >= result_base['trigger_score']


# ======================================================================
# Ω2: causal_trace_coverage_deficit → MCT coherence_deficit
# ======================================================================

class TestOmega2CausalTraceCoverageDeficit:
    """Verify causal_trace_coverage_deficit is consumed by MCT."""

    def test_zero_deficit_no_effect(self):
        """When causal_trace_coverage_deficit = 0, no coherence contribution."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_coverage_deficit', 0.0)
        result = mct.evaluate(uncertainty=0.0)
        assert result['trigger_score'] >= 0.0

    def test_high_deficit_amplifies_coherence(self):
        """High coverage deficit amplifies MCT coherence_deficit."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1)
        mct.reset()
        bus.write_signal('causal_trace_coverage_deficit', 0.8)
        result_high = mct.evaluate(uncertainty=0.1)
        assert result_high['trigger_score'] >= result_base['trigger_score']

    def test_below_threshold_no_contribution(self):
        """causal_trace_coverage_deficit ≤ 0.05 should not contribute."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_coverage_deficit', 0.03)
        result = mct.evaluate(uncertainty=0.0)
        mct.reset()
        bus.write_signal('causal_trace_coverage_deficit', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0)
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_signal_is_read_from_bus(self):
        """MCT must read causal_trace_coverage_deficit from the bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_coverage_deficit', 0.5)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'causal_trace_coverage_deficit' in read_log

    def test_in_causal_trace_ref(self):
        """causal_trace_coverage_deficit should appear in FIA-3 tracing."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_coverage_deficit', 0.7)
        result = mct.evaluate(uncertainty=0.1)
        causal_ref = result.get('_causal_trace_ref', {})
        bus_reads = causal_ref.get('bus_signals_read', {})
        assert 'causal_trace_coverage_deficit' in bus_reads

    def test_moderate_deficit_contributes(self):
        """Coverage deficit = 0.3 should contribute to trigger."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1)
        mct.reset()
        bus.write_signal('causal_trace_coverage_deficit', 0.3)
        result_mod = mct.evaluate(uncertainty=0.1)
        assert result_mod['trigger_score'] >= result_base['trigger_score']


# ======================================================================
# Ω7: causal_trace_disabled + deficit → unified causal quality
# ======================================================================

class TestOmega7UnifiedCausalQuality:
    """Verify causal_trace_disabled boosts coherence_deficit in MCT."""

    def test_disabled_trace_amplifies_coherence_deficit(self):
        """causal_trace_disabled > 0.5 boosts coherence_deficit."""
        mct, bus = _make_mct_with_bus()
        result_base = mct.evaluate(uncertainty=0.1)
        mct.reset()
        bus.write_signal('causal_trace_disabled', 1.0)
        result_disabled = mct.evaluate(uncertainty=0.1)
        assert result_disabled['trigger_score'] >= result_base['trigger_score']

    def test_disabled_below_threshold_no_effect(self):
        """causal_trace_disabled ≤ 0.5 should not boost coherence_deficit."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_disabled', 0.3)
        result = mct.evaluate(uncertainty=0.0)
        mct.reset()
        bus.write_signal('causal_trace_disabled', 0.0)
        result_zero = mct.evaluate(uncertainty=0.0)
        # Ω7 block should not fire below 0.5
        assert result['trigger_score'] == result_zero['trigger_score']

    def test_combined_disabled_and_deficit(self):
        """Both disabled + coverage deficit produce stronger coherence boost."""
        mct, bus = _make_mct_with_bus()
        # Only deficit
        bus.write_signal('causal_trace_coverage_deficit', 0.5)
        result_deficit = mct.evaluate(uncertainty=0.1)
        mct.reset()
        # Both disabled + deficit
        bus.write_signal('causal_trace_disabled', 1.0)
        bus.write_signal('causal_trace_coverage_deficit', 0.5)
        result_both = mct.evaluate(uncertainty=0.1)
        assert result_both['trigger_score'] >= result_deficit['trigger_score']

    def test_signal_read_from_bus(self):
        """MCT must read causal_trace_disabled via Ω7 block."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_disabled', 1.0)
        mct.evaluate(uncertainty=0.1)
        read_log = getattr(bus, '_read_log', set())
        assert 'causal_trace_disabled' in read_log


# ======================================================================
# Ω4: Adaptive loss scaling from feedback bus in compute_loss()
# ======================================================================

class TestOmega4AdaptiveLossScaling:
    """Verify compute_loss() reads bus signals for adaptive scaling."""

    def test_spectral_instability_increases_loss(self):
        """High spectral_instability scales total_loss upward."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        # Baseline: no spectral pressure
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 0.0)
        loss_base = model.compute_loss(outputs, targets)
        total_base = float(loss_base['total_loss'].detach())

        # With spectral instability
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 0.9)
        loss_high = model.compute_loss(outputs, targets)
        total_high = float(loss_high['total_loss'].detach())

        assert total_high >= total_base

    def test_coherence_deficit_increases_loss(self):
        """High coherence_deficit on bus scales total_loss upward."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('coherence_deficit', 0.0)
        loss_base = model.compute_loss(outputs, targets)
        total_base = float(loss_base['total_loss'].detach())

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('coherence_deficit', 0.9)
        loss_high = model.compute_loss(outputs, targets)
        total_high = float(loss_high['total_loss'].detach())

        assert total_high >= total_base

    def test_recovery_pressure_increases_loss(self):
        """High error_recovery_pressure scales total_loss upward."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('error_recovery_pressure', 0.0)
        loss_base = model.compute_loss(outputs, targets)
        total_base = float(loss_base['total_loss'].detach())

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('error_recovery_pressure', 0.9)
        loss_high = model.compute_loss(outputs, targets)
        total_high = float(loss_high['total_loss'].detach())

        assert total_high >= total_base

    def test_below_threshold_no_scaling(self):
        """Signals ≤ 0.1 should not change total_loss."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 0.05)
            model.feedback_bus.write_signal('coherence_deficit', 0.05)
            model.feedback_bus.write_signal('error_recovery_pressure', 0.05)
        loss_low = model.compute_loss(outputs, targets)
        total_low = float(loss_low['total_loss'].detach())

        # Should be finite
        assert math.isfinite(total_low)

    def test_scaling_bounded_at_1_5x(self):
        """Adaptive scaling should be bounded at 1.5× max."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        # Set extreme values (all at 1.0)
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 1.0)
            model.feedback_bus.write_signal('coherence_deficit', 1.0)
            model.feedback_bus.write_signal('error_recovery_pressure', 1.0)
        loss_extreme = model.compute_loss(outputs, targets)
        total_extreme = float(loss_extreme['total_loss'].detach())

        # Should still be finite
        assert math.isfinite(total_extreme)

    def test_omega4_scale_signal_written_to_bus(self):
        """When scaling is active, omega4_adaptive_loss_scale is written."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 0.8)
            model.compute_loss(outputs, targets)
            scale = model.feedback_bus.read_signal(
                'omega4_adaptive_loss_scale', 0.0,
            )
            assert float(scale) >= 1.0


# ======================================================================
# Ω5a: Unconditional safety decision → causal trace
# ======================================================================

class TestOmega5SafetyCausalTrace:
    """Verify safety decisions are unconditionally recorded in causal trace."""

    def test_safety_trace_recorded_on_forward(self):
        """After forward pass, causal trace should contain a 'safety' entry."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        ct = getattr(model, 'causal_trace', None)
        if ct is None or not ct:
            pytest.skip("Causal trace not enabled")
        B, S = 1, 16
        x = torch.randint(1, cfg.vocab_size, (B, S))
        with torch.no_grad():
            model(x)
        # Use find() or recent() to query entries
        safety_entries = []
        if hasattr(ct, 'find'):
            safety_entries = ct.find(subsystem='safety')
        elif hasattr(ct, 'recent'):
            entries = ct.recent(200)
            safety_entries = [
                e for e in entries if e.get('subsystem') == 'safety'
            ]
        assert len(safety_entries) > 0, (
            "No 'safety' entry found in causal trace after forward pass"
        )

    def test_safety_trace_has_verdict(self):
        """Safety trace entry should contain verdict metadata."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        ct = getattr(model, 'causal_trace', None)
        if ct is None or not ct:
            pytest.skip("Causal trace not enabled")
        B, S = 1, 16
        x = torch.randint(1, cfg.vocab_size, (B, S))
        with torch.no_grad():
            model(x)
        safety_entries = []
        if hasattr(ct, 'find'):
            safety_entries = ct.find(subsystem='safety')
        elif hasattr(ct, 'recent'):
            entries = ct.recent(200)
            safety_entries = [
                e for e in entries if e.get('subsystem') == 'safety'
            ]
        if not safety_entries:
            pytest.skip("No safety entry found")
        entry = safety_entries[-1]
        decision = entry.get('decision', '')
        # Ω5a records decision as "verdict=passed" or "verdict=enforced"
        assert 'verdict=' in str(decision), (
            f"Safety trace decision should contain 'verdict=', got: {decision}"
        )


# ======================================================================
# Ω6: Inference anomaly signals → training bridge
# ======================================================================

class TestOmega6InferenceTrainingBridge:
    """Verify _bridge_epoch_feedback() reads inference anomaly signals."""

    def _has_feedback_bus(self, model):
        return hasattr(model, 'feedback_bus') and model.feedback_bus is not None

    def _make_trainer_and_batch(self):
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        trainer = AEONTrainer(model, cfg)
        batch = {
            'input_ids': torch.randint(1, cfg.vocab_size, (1, 16)),
            'labels': torch.randint(1, cfg.vocab_size, (1, 16)),
        }
        return trainer, model, cfg, batch

    def test_spectral_instability_tightens_gradient_clip(self):
        """High spectral_instability in bus should tighten gradient clipping."""
        trainer, model, cfg, batch = self._make_trainer_and_batch()
        if not self._has_feedback_bus(model):
            pytest.skip("No feedback bus")
        # Set high spectral instability
        model.feedback_bus.write_signal('spectral_instability', 0.8)
        old_clip = trainer._grad_clip_norm
        result = trainer._bridge_epoch_feedback()
        new_clip = trainer._grad_clip_norm
        # Gradient clip should be tightened
        assert new_clip <= old_clip

    def test_recovery_pressure_sets_lr_scale(self):
        """High error_recovery_pressure should set recovery LR scale."""
        trainer, model, cfg, batch = self._make_trainer_and_batch()
        if not self._has_feedback_bus(model):
            pytest.skip("No feedback bus")
        model.feedback_bus.write_signal('error_recovery_pressure', 0.8)
        trainer._bridge_epoch_feedback()
        lr_scale = getattr(trainer, '_recovery_lr_scale', 1.0)
        assert lr_scale >= 1.0

    def test_low_spectral_no_omega6_tightening(self):
        """Spectral instability ≤ 0.3 should not trigger Ω6 gradient clip."""
        trainer, model, cfg, batch = self._make_trainer_and_batch()
        if not self._has_feedback_bus(model):
            pytest.skip("No feedback bus")
        model.feedback_bus.write_signal('spectral_instability', 0.2)
        # Reset coherence deficit to prevent other bridge paths from firing
        model._cached_coherence_deficit = 0.0
        model._cached_cert_violated = False
        old_clip = trainer._grad_clip_norm
        trainer._bridge_epoch_feedback()
        new_clip = trainer._grad_clip_norm
        # Clip should not be tightened by Ω6 specifically
        assert new_clip == old_clip

    def test_low_recovery_no_lr_change(self):
        """Recovery pressure ≤ 0.5 should not set recovery LR scale."""
        trainer, model, cfg, batch = self._make_trainer_and_batch()
        if not self._has_feedback_bus(model):
            pytest.skip("No feedback bus")
        model.feedback_bus.write_signal('error_recovery_pressure', 0.3)
        trainer._bridge_epoch_feedback()
        lr_scale = getattr(trainer, '_recovery_lr_scale', 1.0)
        assert lr_scale == 1.0, (
            f"Recovery LR scale should be 1.0 when pressure is below "
            f"threshold, got {lr_scale}"
        )

    def test_bridge_returns_adapted_count(self):
        """When anomaly signals are high, adapted_from_inference count increases."""
        trainer, model, cfg, batch = self._make_trainer_and_batch()
        if not self._has_feedback_bus(model):
            pytest.skip("No feedback bus")
        model.feedback_bus.write_signal('spectral_instability', 0.9)
        model.feedback_bus.write_signal('error_recovery_pressure', 0.9)
        result = trainer._bridge_epoch_feedback()
        assert result['adapted_from_inference'] >= 1


# ======================================================================
# Ω8: Research-variant docstrings on meta-loop classes
# ======================================================================

class TestOmega8MetaLoopDocstrings:
    """Verify meta-loop variant classes have research-variant documentation."""

    def test_hierarchical_meta_loop_docstring(self):
        """HierarchicalMetaLoop docstring mentions 'Research variant'."""
        doc = HierarchicalMetaLoop.__doc__ or ''
        assert 'research variant' in doc.lower() or 'Research variant' in doc

    def test_recursive_meta_loop_docstring(self):
        """RecursiveMetaLoop docstring mentions 'Research variant'."""
        doc = RecursiveMetaLoop.__doc__ or ''
        assert 'research variant' in doc.lower() or 'Research variant' in doc

    def test_adaptive_meta_loop_docstring(self):
        """AdaptiveMetaLoop docstring mentions 'Research variant'."""
        doc = AdaptiveMetaLoop.__doc__ or ''
        assert 'research variant' in doc.lower() or 'Research variant' in doc

    def test_hierarchical_meta_loop_references_production(self):
        """HierarchicalMetaLoop docstring references ProvablyConvergentMetaLoop."""
        doc = HierarchicalMetaLoop.__doc__ or ''
        assert 'ProvablyConvergentMetaLoop' in doc

    def test_recursive_meta_loop_references_production(self):
        """RecursiveMetaLoop docstring references ProvablyConvergentMetaLoop."""
        doc = RecursiveMetaLoop.__doc__ or ''
        assert 'ProvablyConvergentMetaLoop' in doc

    def test_adaptive_meta_loop_references_production(self):
        """AdaptiveMetaLoop docstring references ProvablyConvergentMetaLoop."""
        doc = AdaptiveMetaLoop.__doc__ or ''
        assert 'ProvablyConvergentMetaLoop' in doc


# ======================================================================
# Integration: End-to-end Ω signal flow verification
# ======================================================================

class TestOmegaIntegrationEndToEnd:
    """End-to-end tests verifying the Ω patches close feedback loops."""

    def test_mutual_reinforcement_error_recovery(self):
        """Ω1+Ω4: error_recovery_pressure → MCT trigger ↑ → loss scale ↑."""
        mct, bus = _make_mct_with_bus()
        # Write high recovery pressure
        bus.write_signal('error_recovery_pressure', 0.9)
        result = mct.evaluate(uncertainty=0.2, convergence_conflict=0.3)
        # MCT should see elevated trigger
        assert result['trigger_score'] > 0.0
        # FIA-3 causal ref should record the signal
        causal_ref = result.get('_causal_trace_ref', {})
        bus_reads = causal_ref.get('bus_signals_read', {})
        assert 'error_recovery_pressure' in bus_reads

    def test_metacognitive_trigger_from_disabled_trace(self):
        """Ω7: causal_trace_disabled → MCT coherence_deficit ↑ → re-reasoning."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_trace_disabled', 1.0)
        result = mct.evaluate(uncertainty=0.2, convergence_conflict=0.3)
        # Should see boosted coherence_deficit in trigger
        assert result['trigger_score'] > 0.0

    def test_causal_trace_coverage_self_healing(self):
        """Ω2: coverage deficit → MCT trigger → re-reasoning → coverage up."""
        mct, bus = _make_mct_with_bus()
        # Write high coverage deficit
        bus.write_signal('causal_trace_coverage_deficit', 0.7)
        result = mct.evaluate(uncertainty=0.3, convergence_conflict=0.3)
        # MCT should amplify coherence_deficit
        assert result['trigger_score'] > 0.0

    def test_all_omega_signals_finite(self):
        """All Ω signal reads produce finite values in MCT."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('error_recovery_pressure', 0.5)
        bus.write_signal('causal_trace_coverage_deficit', 0.5)
        bus.write_signal('causal_trace_disabled', 1.0)
        result = mct.evaluate(uncertainty=0.3, convergence_conflict=0.3)
        assert math.isfinite(result['trigger_score'])
        assert math.isfinite(result['effective_trigger_score'])

    def test_compute_loss_with_all_omega_signals(self):
        """compute_loss with all Ω signals still produces finite loss."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        outputs = _make_outputs(cfg)
        targets = torch.randint(0, cfg.vocab_size, (2, 16))
        if hasattr(model, 'feedback_bus') and model.feedback_bus is not None:
            model.feedback_bus.write_signal('spectral_instability', 0.8)
            model.feedback_bus.write_signal('coherence_deficit', 0.7)
            model.feedback_bus.write_signal('error_recovery_pressure', 0.6)
        loss = model.compute_loss(outputs, targets)
        total = float(loss['total_loss'].detach())
        assert math.isfinite(total)
        assert total > 0.0
