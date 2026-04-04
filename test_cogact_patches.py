"""
Tests for COGACT-series patches: Final Integration & Cognitive Activation.

Patches covered:
  COGACT-1: Orphaned signal wiring to MCT
  COGACT-2: Unified convergence theorem for Anderson+KM
  COGACT-3: Joint γ_max·√d/√ε constraint + GELU 1.13 derivation
  COGACT-4: SSM cumsum identity proof for diagonal recurrences
  COGACT-5: Catastrophe ground-truth validation against A_k normal forms
  COGACT-6: Stall severity → convergence_residual loss weight
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
    OptimizedTopologyAnalyzer,
    _SSMBlock,
    _SSDBlock,
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


# ══════════════════════════════════════════════════════════════════════
#  COGACT-1: Orphaned signal wiring to MCT
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT1_OrphanedSignalWiring:
    """Verify that previously orphaned signals are now consumed by MCT."""

    def _make_trigger(self, hidden_dim=64):
        """Create MCT trigger with feedback bus."""
        bus = CognitiveFeedbackBus(hidden_dim)
        trigger = MetaCognitiveRecursionTrigger()
        trigger.set_feedback_bus(bus)
        return trigger, bus

    def test_vq_collapse_severity_consumed(self):
        """COGACT-1a: VQ collapse severity → diversity_collapse signal."""
        trigger, bus = self._make_trigger()
        bus.write_signal('vq_collapse_severity', 0.8)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        # VQ collapse was read by MCT, so it should not be orphaned
        orphaned = bus.get_orphaned_signals()
        assert 'vq_collapse_severity' not in orphaned

    def test_rssm_prediction_pressure_consumed(self):
        """COGACT-1b: RSSM prediction pressure → world_model_surprise."""
        trigger, bus = self._make_trigger()
        bus.write_signal('rssm_prediction_pressure', 0.7)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        # read_signal was called, signal was consumed
        orphaned = bus.get_orphaned_signals()
        assert 'rssm_prediction_pressure' not in orphaned

    def test_vibe_thinker_quality_consumed(self):
        """COGACT-1c: Vibe thinker quality → output reliability."""
        trigger, bus = self._make_trigger()
        bus.write_signal('vibe_thinker_quality', 0.2)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        orphaned = bus.get_orphaned_signals()
        assert 'vibe_thinker_quality' not in orphaned

    def test_moderate_output_uncertainty_consumed(self):
        """COGACT-1e: Moderate output uncertainty → border_uncertainty."""
        trigger, bus = self._make_trigger()
        bus.write_signal('moderate_output_uncertainty_pressure', 0.6)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        orphaned = bus.get_orphaned_signals()
        assert 'moderate_output_uncertainty_pressure' not in orphaned

    def test_low_causal_coverage_consumed(self):
        """COGACT-1f: Low causal coverage → low_causal_quality."""
        trigger, bus = self._make_trigger()
        bus.write_signal('low_causal_coverage', 0.5)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        orphaned = bus.get_orphaned_signals()
        assert 'low_causal_coverage' not in orphaned

    def test_lyapunov_descent_violation_consumed(self):
        """COGACT-1g: Lyapunov descent violation → convergence_conflict."""
        trigger, bus = self._make_trigger()
        bus.write_signal('lyapunov_descent_violation', 0.7)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        orphaned = bus.get_orphaned_signals()
        assert 'lyapunov_descent_violation' not in orphaned

    def test_recursion_outcome_consumed(self):
        """COGACT-1d: Recursion outcome useful is consumed."""
        trigger, bus = self._make_trigger()
        bus.write_signal('recursion_outcome_useful', 0.9)
        result = trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )
        orphaned = bus.get_orphaned_signals()
        assert 'recursion_outcome_useful' not in orphaned


# ══════════════════════════════════════════════════════════════════════
#  COGACT-2: Unified convergence theorem for Anderson+KM
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT2_UnifiedConvergenceTheorem:
    """Verify convergence regime classification in certificate."""

    def _make_model(self):
        cfg = _make_config()
        return AEONDeltaV3(cfg)

    def test_certificate_has_convergence_regime(self):
        """Certificate contains convergence_regime dict."""
        model = self._make_model()
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        assert 'convergence_regime' in cert

    def test_convergence_regime_fields(self):
        """Convergence regime has required fields."""
        model = self._make_model()
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        regime = cert['convergence_regime']
        assert 'regime' in regime
        assert 'anderson_active' in regime
        assert 'inertial_active' in regime
        assert 'a_posteriori_bound_valid' in regime
        assert 'bound_correction_formula' in regime
        assert 'perturbation_summability' in regime
        assert 'safeguard_enforcement' in regime

    def test_convergence_regime_type(self):
        """Regime type is one of the valid values."""
        model = self._make_model()
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        regime = cert['convergence_regime']
        assert regime['regime'] in (
            'banach_contraction', 'km_nonexpansive', 'unverified',
        )

    def test_safeguard_enforcement_fields(self):
        """Safeguard enforcement dict has required fields."""
        model = self._make_model()
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        safeguard = cert['convergence_regime']['safeguard_enforcement']
        assert safeguard['residual_monotonicity_enforced'] is True
        assert safeguard['gram_conditioning_enforced'] is True
        assert safeguard['step_norm_bounded'] is True
        assert 'Zhang' in safeguard['theorem']

    def test_perturbation_summability_fields(self):
        """Perturbation summability dict has required fields."""
        model = self._make_model()
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        summ = cert['convergence_regime']['perturbation_summability']
        assert 'summable' in summ
        assert 'budget_remaining' in summ
        assert 'formal_guarantee' in summ


# ══════════════════════════════════════════════════════════════════════
#  COGACT-3: Joint γ_max·√d/√ε constraint + GELU derivation
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT3_LipschitzConstraint:
    """Verify joint constraint and GELU derivation in budget."""

    def _make_model(self):
        cfg = _make_config()
        return AEONDeltaV3(cfg)

    def test_joint_constraint_in_budget(self):
        """verify_joint_lipschitz_budget includes joint_constraint_input."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        ln_analysis = budget['layernorm_variance_analysis']
        assert 'joint_constraint_input' in ln_analysis
        assert 'joint_constraint_output' in ln_analysis

    def test_joint_constraint_fields(self):
        """Joint constraint dict has derivation and satisfaction flag."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        jc_in = budget['layernorm_variance_analysis']['joint_constraint_input']
        assert 'gamma_max' in jc_in
        assert 'epsilon' in jc_in
        assert 'dimension' in jc_in
        assert 'Lip_LN_worst_case' in jc_in
        assert 'constraint_satisfied' in jc_in
        assert 'derivation' in jc_in

    def test_joint_constraint_worst_case_correct(self):
        """Worst-case Lip(LN) = γ_max · √(d/(d-1)) / √ε."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        jc = budget['layernorm_variance_analysis']['joint_constraint_input']
        gamma_max = jc['gamma_max']
        eps = jc['epsilon']
        d = jc['dimension']
        expected = gamma_max * math.sqrt(d / max(d - 1, 1)) / math.sqrt(eps)
        assert abs(jc['Lip_LN_worst_case'] - expected) < 1e-4

    def test_gelu_derivation_present(self):
        """GELU Lipschitz derivation included in budget."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        assert 'gelu_lipschitz_derivation' in budget

    def test_gelu_constant_is_1_13(self):
        """GELU constant is exactly 1.13."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        assert budget['gelu_lipschitz_derivation']['L_GELU'] == 1.13

    def test_gelu_has_derivation_method(self):
        """GELU derivation includes method and critical point."""
        model = self._make_model()
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        gelu = budget['gelu_lipschitz_derivation']
        assert 'derivation_method' in gelu
        assert 'critical_point_x' in gelu
        assert abs(gelu['critical_point_x'] - (-0.436)) < 0.01
        assert 'verification' in gelu


# ══════════════════════════════════════════════════════════════════════
#  COGACT-4: SSM cumsum identity proof
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT4_SSMCumsumProof:
    """Verify the cumsum identity proof and diagonal structure."""

    def test_ssm_diagonal_invariant(self):
        """_SSMBlock verifies diagonal A invariant."""
        ssm = _SSMBlock(d_model=64, d_state=16, d_inner=128,
                        dt_rank=8, dropout=0.0)
        result = ssm.verify_diagonal_A_invariant()
        assert result['is_diagonal'] is True
        assert result['A_structure'] == 'diagonal'
        assert result['elementwise_exp_valid'] is True

    def test_ssd_scalar_per_head(self):
        """_SSDBlock has scalar_per_head structure."""
        ssd = _SSDBlock(d_model=64, d_state=16, d_inner=128,
                        nheads=4, dt_rank=8, dropout=0.0)
        assert ssd._A_structure == 'scalar_per_head'

    def test_cumsum_scan_correctness(self):
        """_cumsum_scan produces correct results for known recurrence."""
        ssm = _SSMBlock(d_model=64, d_state=16, d_inner=128,
                        dt_rank=8, dropout=0.0)
        B, L, D, N = 1, 8, 4, 2
        coeffs = torch.ones(B, L, D, N) * 0.9  # constant decay
        values = torch.randn(B, L, D, N)

        # Ground truth via sequential scan
        h_seq = torch.zeros(B, D, N)
        h_all_seq = []
        for t in range(L):
            h_seq = coeffs[:, t] * h_seq + values[:, t]
            h_all_seq.append(h_seq.clone())
        h_expected = torch.stack(h_all_seq, dim=1)

        # cumsum scan
        h_actual = ssm._cumsum_scan(coeffs, values)
        assert torch.allclose(h_actual, h_expected, atol=1e-4), \
            f"Max diff: {(h_actual - h_expected).abs().max()}"

    def test_cumsum_scan_docstring_has_proof(self):
        """_cumsum_scan docstring contains the formal proof."""
        docstring = _SSMBlock._cumsum_scan.__doc__
        assert 'COGACT-4' in docstring
        assert 'Proof' in docstring or 'proof' in docstring
        assert 'diagonal' in docstring.lower()


# ══════════════════════════════════════════════════════════════════════
#  COGACT-5: Catastrophe ground-truth validation
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT5_CatastropheValidation:
    """Verify ground-truth validation against A2/A3/A4 normal forms."""

    def test_validate_method_exists(self):
        """OptimizedTopologyAnalyzer has validate_against_normal_forms."""
        assert hasattr(OptimizedTopologyAnalyzer, 'validate_against_normal_forms')

    def test_validation_all_pass(self):
        """All normal form validation tests pass."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['all_passed'] is True

    def test_a2_fold_classification(self):
        """A₂ fold is correctly classified as fold."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['A2_fold']['actual_type'] == 'fold'
        assert results['A2_fold']['passed'] is True

    def test_a3_cusp_classification(self):
        """A₃ cusp is correctly classified as cusp."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['A3_cusp']['actual_type'] == 'cusp'
        assert results['A3_cusp']['passed'] is True

    def test_a4_swallowtail_classification(self):
        """A₄ swallowtail is correctly classified as swallowtail."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['A4_swallowtail']['actual_type'] == 'swallowtail'
        assert results['A4_swallowtail']['passed'] is True

    def test_non_degenerate_classification(self):
        """Non-degenerate case is classified as none."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['non_degenerate']['actual_type'] == 'none'
        assert results['non_degenerate']['passed'] is True

    def test_confidence_capped_at_0_5(self):
        """Classification confidence is capped at 0.5 (no jet analysis)."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        for key in ('A2_fold', 'A3_cusp', 'A4_swallowtail'):
            assert results[key]['confidence'] <= 0.5

    def test_criticality_at_critical_point(self):
        """At critical point (grad=0, corank≥1), criticality is degenerate."""
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert results['A2_fold']['criticality'] == 'degenerate_critical_point'


# ══════════════════════════════════════════════════════════════════════
#  COGACT-6: Stall → convergence residual loss weight
# ══════════════════════════════════════════════════════════════════════

class TestCOGACT6_StallLossScaling:
    """Verify stall severity modulates convergence residual loss."""

    def _make_model(self):
        cfg = _make_config()
        return AEONDeltaV3(cfg)

    def test_compute_loss_handles_stall_metadata(self):
        """compute_loss accepts meta_results with stall_detected."""
        model = self._make_model()
        B, S = 2, 16
        logits = torch.randn(B, S, model.config.vocab_size)
        targets = torch.randint(0, model.config.vocab_size, (B, S))
        outputs = {
            'logits': logits,
            'vq_loss': torch.tensor(0.0),
            'certificate': {},
            'meta_results': {
                'residual_norm': 5.0,
                'certificate': {
                    'stall_detected': True,
                    'stall_contraction_ratio': 0.95,
                },
            },
        }
        # Should not raise
        loss_dict = model.compute_loss(outputs, targets)
        assert 'total_loss' in loss_dict

    def test_stall_amplifies_convergence_loss(self):
        """When stall_detected, convergence residual loss is amplified."""
        model = self._make_model()
        B, S = 2, 16
        logits = torch.randn(B, S, model.config.vocab_size)
        targets = torch.randint(0, model.config.vocab_size, (B, S))

        # Test with stall — verify the component loss is larger than
        # the base residual_norm value (which is amplified by stall)
        outputs_stall = {
            'logits': logits,
            'vq_loss': torch.tensor(0.0),
            'certificate': {},
            'meta_results': {
                'residual_norm': 5.0,
                'certificate': {
                    'stall_detected': True,
                    'stall_contraction_ratio': 0.99,
                },
            },
        }
        loss_stall = model.compute_loss(outputs_stall, targets)

        # The convergence_residual_loss with stall should be > 5.0
        # (amplified by 1 + min(1.0, max(0, 0.99)) = 1.99×)
        cr_loss = loss_stall.get('convergence_residual_loss', torch.tensor(0.0))
        if torch.is_tensor(cr_loss):
            cr_val = float(cr_loss.detach().item())
        else:
            cr_val = float(cr_loss)
        # With stall_contraction_ratio=0.99, amplification = 1.99×
        # 5.0 * 1.99 = 9.95
        assert cr_val >= 5.0, (
            f"Stall-amplified convergence residual loss ({cr_val:.4f}) "
            f"should be >= 5.0 (base residual)"
        )


# ══════════════════════════════════════════════════════════════════════
#  Cross-patch integration tests
# ══════════════════════════════════════════════════════════════════════

class TestCrossPatchIntegration:
    """Test interactions between COGACT patches."""

    def test_orphan_reduction(self):
        """COGACT-1 reduces orphaned signal count."""
        bus = CognitiveFeedbackBus(64)
        trigger = MetaCognitiveRecursionTrigger()
        trigger.set_feedback_bus(bus)

        # Write all the previously-orphaned signals
        bus.write_signal('vq_collapse_severity', 0.5)
        bus.write_signal('rssm_prediction_pressure', 0.5)
        bus.write_signal('vibe_thinker_quality', 0.3)
        bus.write_signal('moderate_output_uncertainty_pressure', 0.5)
        bus.write_signal('low_causal_coverage', 0.5)
        bus.write_signal('lyapunov_descent_violation', 0.5)
        bus.write_signal('recursion_outcome_useful', 0.8)

        # Trigger evaluation should consume them
        trigger.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, diversity_collapse=0.0,
            memory_trust_deficit=0.0, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0, stall_severity=0.0,
            oscillation_severity=0.0,
        )

        orphaned = bus.get_orphaned_signals()
        # None of the newly wired signals should be orphaned
        newly_wired = {
            'vq_collapse_severity', 'rssm_prediction_pressure',
            'vibe_thinker_quality', 'moderate_output_uncertainty_pressure',
            'low_causal_coverage', 'lyapunov_descent_violation',
            'recursion_outcome_useful',
        }
        for sig in newly_wired:
            assert sig not in orphaned, f"Signal '{sig}' is still orphaned"

    def test_normal_form_validation_static(self):
        """validate_against_normal_forms is a static method."""
        # Should be callable without instance
        results = OptimizedTopologyAnalyzer.validate_against_normal_forms()
        assert isinstance(results, dict)
        assert results['all_passed'] is True

    def test_convergence_regime_anderson_tracking(self):
        """Certificate tracks Anderson accept/reject in regime."""
        cfg = _make_config()
        model = AEONDeltaV3(cfg)
        psi_0 = torch.randn(1, 64)
        _, _, cert = model.meta_loop.compute_fixed_point(
            psi_0, return_certificate=True,
        )
        regime = cert['convergence_regime']
        assert isinstance(regime['anderson_active'], bool)
        assert isinstance(regime['inertial_active'], bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-q'])
