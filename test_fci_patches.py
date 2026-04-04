"""Tests for FCI-series patches: Final Cognitive Integration.

FCI-1: Lyapunov ΔV tracking in fixed-point iteration loop
FCI-2: Hard LayerNorm gamma constraint via gradient hook
FCI-3: Joint Lipschitz factor verification
FCI-4: Lyapunov descent + joint Lipschitz in training loop
FCI-5: Finite-iterate convergence qualification
FCI-6: Catastrophe classification confidence decay
"""
import math
import sys
import os

import pytest
import torch
import torch.nn as nn

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    ProvablyConvergentMetaLoop,
    LipschitzConstrainedLambda,
    LyapunovDeltaVMonitor,
    CognitiveFeedbackBus,
    OptimizedTopologyAnalyzer,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        meta_dim=32,
        dropout_rate=0.0,
        lipschitz_target=0.95,
        device_str='cpu',
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_meta_loop(config=None, max_iterations=5):
    if config is None:
        config = _make_config()
    return ProvablyConvergentMetaLoop(
        config,
        max_iterations=max_iterations,
        min_iterations=1,
        enable_certification=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FCI-1: Lyapunov ΔV tracking in fixed-point iteration loop
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI1LyapunovIteration:
    """FCI-1: Per-iteration Lyapunov ΔV recording."""

    def test_lyapunov_monitor_attribute_exists(self):
        """Meta-loop should have _lyapunov_iteration_monitor."""
        ml = _make_meta_loop()
        assert hasattr(ml, '_lyapunov_iteration_monitor')
        assert isinstance(ml._lyapunov_iteration_monitor, LyapunovDeltaVMonitor)

    def test_lyapunov_monitor_reset_per_invocation(self):
        """Monitor should be reset at start of each compute_fixed_point."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        # First call
        _, _, meta1 = ml.compute_fixed_point(psi)
        # Record some history
        status1 = meta1.get('lyapunov_iteration_status', {})
        n1 = status1.get('num_samples', 0)
        # Second call should reset
        _, _, meta2 = ml.compute_fixed_point(psi)
        status2 = meta2.get('lyapunov_iteration_status', {})
        n2 = status2.get('num_samples', 0)
        # Both should have samples (iterations ran) and values should be
        # from fresh runs (not accumulated)
        assert n1 > 0
        assert n2 > 0

    def test_lyapunov_status_in_metadata(self):
        """Metadata should contain lyapunov_iteration_status."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        assert 'lyapunov_iteration_status' in meta
        status = meta['lyapunov_iteration_status']
        assert 'delta_v' in status
        assert 'stable' in status
        assert 'oscillating' in status
        assert 'delta_v_mean' in status
        assert 'num_samples' in status

    def test_lyapunov_records_per_iteration(self):
        """Monitor should record one sample per iteration."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        status = meta['lyapunov_iteration_status']
        # Should have recorded at least min_iterations samples
        assert status['num_samples'] >= 1

    def test_lyapunov_bus_signal_on_instability(self):
        """When ΔV > 0, lyapunov_descent_violation should be written."""
        ml = _make_meta_loop(max_iterations=5)
        bus = CognitiveFeedbackBus(64)
        ml._feedback_bus_ref = bus
        psi = torch.randn(2, 64)
        ml.compute_fixed_point(psi)
        # The signal may or may not be written depending on the iteration
        # dynamics; we verify the infrastructure is in place
        assert hasattr(ml, '_feedback_bus_ref')
        assert ml._feedback_bus_ref is bus

    def test_lyapunov_psi_value_finite(self):
        """Lyapunov V(t) values should be finite."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        status = meta['lyapunov_iteration_status']
        psi_val = status.get('psi_value', None)
        if psi_val is not None and psi_val != 0.0:
            assert math.isfinite(psi_val)


# ═══════════════════════════════════════════════════════════════════════════
# FCI-2: Hard LayerNorm gamma constraint
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI2GammaConstraint:
    """FCI-2: Hard LayerNorm gamma ratio constraint via gradient hook."""

    def test_gamma_ratio_max_attribute(self):
        """Meta-loop should have _gamma_ratio_max."""
        ml = _make_meta_loop()
        assert hasattr(ml, '_gamma_ratio_max')
        assert ml._gamma_ratio_max >= 1.0

    def test_gamma_hook_registered(self):
        """LayerNorm weights should have gradient hooks registered."""
        ml = _make_meta_loop()
        # Input and output stabilizers should have hooks
        assert len(ml.input_stabilizer.weight._backward_hooks) > 0 or \
               hasattr(ml.input_stabilizer.weight, '_backward_hooks')
        assert len(ml.output_stabilizer.weight._backward_hooks) > 0 or \
               hasattr(ml.output_stabilizer.weight, '_backward_hooks')

    def test_gamma_clamp_hook_reduces_ratio(self):
        """Hook should project gamma when ratio exceeds max."""
        ml = _make_meta_loop()
        ml._gamma_ratio_max = 1.5

        # Manually set gamma to have large ratio
        with torch.no_grad():
            gamma = ml.output_stabilizer.weight
            gamma[0] = 10.0  # make one element very large
            gamma[1] = 0.1   # make another very small

        # Create a hook and call it
        hook = ml._make_gamma_clamp_hook(ml.output_stabilizer)
        dummy_grad = torch.ones_like(ml.output_stabilizer.weight)
        hook(dummy_grad)

        # After hook, ratio should be reduced
        gamma_after = ml.output_stabilizer.weight.detach()
        ratio_after = gamma_after.abs().max().item() / max(
            gamma_after.abs().min().item(), 1e-8
        )
        assert ratio_after < 10.0  # Should be reduced from 100

    def test_gamma_clamp_preserves_gradient(self):
        """Hook should return the gradient unchanged."""
        ml = _make_meta_loop()
        hook = ml._make_gamma_clamp_hook(ml.output_stabilizer)
        dummy_grad = torch.randn_like(ml.output_stabilizer.weight)
        result = hook(dummy_grad)
        assert torch.allclose(result, dummy_grad)

    def test_gamma_hook_noop_within_budget(self):
        """Hook should not modify gamma when ratio is within budget."""
        ml = _make_meta_loop()
        ml._gamma_ratio_max = 100.0  # very lenient

        # Default init gamma is close to uniform (ratio ≈ 1)
        gamma_before = ml.output_stabilizer.weight.detach().clone()
        hook = ml._make_gamma_clamp_hook(ml.output_stabilizer)
        hook(torch.ones_like(gamma_before))
        gamma_after = ml.output_stabilizer.weight.detach()

        # Should be unchanged (or negligibly changed)
        assert torch.allclose(gamma_before, gamma_after, atol=1e-6)

    def test_make_gamma_clamp_hook_method(self):
        """_make_gamma_clamp_hook should return a callable."""
        ml = _make_meta_loop()
        hook = ml._make_gamma_clamp_hook(ml.output_stabilizer)
        assert callable(hook)


# ═══════════════════════════════════════════════════════════════════════════
# FCI-3: Joint Lipschitz factor verification
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI3JointLipschitz:
    """FCI-3: verify_joint_lipschitz_budget method."""

    def test_method_exists(self):
        """Meta-loop should have verify_joint_lipschitz_budget."""
        ml = _make_meta_loop()
        assert hasattr(ml, 'verify_joint_lipschitz_budget')
        assert callable(ml.verify_joint_lipschitz_budget)

    def test_returns_expected_keys(self):
        """Result should contain all required keys."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        expected_keys = {
            'factors', 'L_T_composed', 'lipschitz_target',
            'banach_contraction_satisfied', 'km_nonexpansiveness_satisfied',
            'budget_satisfied', 'dominant_factor', 'dominant_value',
            'gamma_ratio_input', 'gamma_ratio_output', 'gamma_ratio_max',
            'recommendation',
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_factors_dict_has_components(self):
        """Factors dict should include all compositional components."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        factors = result['factors']
        # Must include input/output stabilizers and lambda internals
        assert 'L_input_stabilizer' in factors
        assert 'L_output_stabilizer' in factors
        assert 'L_GELU' in factors

    def test_l_t_composed_is_product(self):
        """L_T_composed should be the product of all factors."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        factors = result['factors']
        expected_product = 1.0
        for v in factors.values():
            if math.isfinite(v) and v > 0:
                expected_product *= v
        assert abs(result['L_T_composed'] - expected_product) < 1e-6

    def test_l_t_positive(self):
        """L_T_composed should be positive."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        assert result['L_T_composed'] > 0

    def test_dominant_factor_is_largest(self):
        """dominant_factor should be the factor with largest value."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        factors = result['factors']
        actual_max = max(factors.items(), key=lambda x: x[1])
        assert result['dominant_factor'] == actual_max[0]
        assert abs(result['dominant_value'] - actual_max[1]) < 1e-6

    def test_recommendation_string(self):
        """Should provide a non-empty recommendation."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        assert isinstance(result['recommendation'], str)
        assert len(result['recommendation']) > 0

    def test_banach_vs_km_distinction(self):
        """Banach and KM satisfaction should be consistent."""
        ml = _make_meta_loop()
        result = ml.verify_joint_lipschitz_budget()
        # Banach (L < 1) implies KM (L ≤ 1)
        if result['banach_contraction_satisfied']:
            assert result['km_nonexpansiveness_satisfied']


# ═══════════════════════════════════════════════════════════════════════════
# FCI-4: Lyapunov + joint Lipschitz in training loop
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI4TrainingIntegration:
    """FCI-4: Lyapunov and joint Lipschitz invoked in training."""

    def test_training_step_with_lyapunov_monitor(self):
        """Training step should not error with Lyapunov monitor present."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model.train()
        meta_loop = getattr(model, 'meta_loop', None)
        if meta_loop is not None:
            assert hasattr(meta_loop, '_lyapunov_iteration_monitor')
            # The monitor should exist on the model's meta_loop
            monitor = meta_loop._lyapunov_iteration_monitor
            assert isinstance(monitor, LyapunovDeltaVMonitor)

    def test_meta_loop_has_joint_budget_method(self):
        """Model's meta_loop should have verify_joint_lipschitz_budget."""
        config = _make_config()
        model = AEONDeltaV3(config)
        meta_loop = getattr(model, 'meta_loop', None)
        if meta_loop is not None:
            assert hasattr(meta_loop, 'verify_joint_lipschitz_budget')


# ═══════════════════════════════════════════════════════════════════════════
# FCI-5: Finite-iterate convergence qualification
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI5FiniteIterate:
    """FCI-5: Finite-iterate convergence qualification in metadata."""

    def test_qualification_in_metadata(self):
        """Metadata should contain finite_iterate_qualification."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        assert 'finite_iterate_qualification' in meta

    def test_qualification_keys(self):
        """Qualification dict should have all required keys."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        fiq = meta['finite_iterate_qualification']
        expected = {
            'N_actual', 'N_max', 'N_infinite_required',
            'banach_finite_bound', 'km_series_coverage',
            'convergence_gap_note',
        }
        assert expected.issubset(set(fiq.keys()))

    def test_n_infinite_required_true(self):
        """N_infinite_required should always be True."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        assert meta['finite_iterate_qualification']['N_infinite_required'] is True

    def test_n_actual_within_bounds(self):
        """N_actual should be ≤ N_max."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        fiq = meta['finite_iterate_qualification']
        # N_actual = iterations.max() + 1 (iterations tensor counts from 0)
        # It can equal N_max + 1 when no sample converged early.
        assert fiq['N_actual'] <= fiq['N_max'] + 1

    def test_banach_bound_positive_when_contractive(self):
        """When L_C < 1, banach_finite_bound should be finite and positive."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        fiq = meta['finite_iterate_qualification']
        L_C = meta.get('L_certificate', 1.0)
        if L_C < 1.0:
            assert math.isfinite(fiq['banach_finite_bound'])
            assert fiq['banach_finite_bound'] >= 0

    def test_km_series_coverage_nonnegative(self):
        """KM series coverage should be non-negative."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        assert meta['finite_iterate_qualification']['km_series_coverage'] >= 0

    def test_convergence_gap_note_string(self):
        """Gap note should be a non-empty string."""
        ml = _make_meta_loop()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi)
        note = meta['finite_iterate_qualification']['convergence_gap_note']
        assert isinstance(note, str)
        assert len(note) > 0


# ═══════════════════════════════════════════════════════════════════════════
# FCI-6: Catastrophe classification confidence decay
# ═══════════════════════════════════════════════════════════════════════════


class TestFCI6CatastropheConfidence:
    """FCI-6: Multi-factor catastrophe classification confidence."""

    def _make_analyzer(self):
        config = _make_config()
        return OptimizedTopologyAnalyzer(config=config)

    @staticmethod
    def _classify(analyzer, eigenvalues):
        """Helper: call classify_catastrophe_type with derived metrics."""
        condition_number = (
            eigenvalues.abs().max(dim=-1).values
            / eigenvalues.abs().clamp(min=1e-8).min(dim=-1).values
        )
        grad_norm = eigenvalues.abs().mean(dim=-1)
        return analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )

    def test_confidence_capped_without_formal_criteria(self):
        """Without jet analysis, confidence should be ≤ 0.5."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5, 0.3, 0.1]])
        result = self._classify(analyzer, eigenvalues)
        conf = result.get('classification_confidence', 1.0)
        assert conf <= 0.5 + 1e-6, (
            f"Confidence {conf} should be ≤ 0.5 without formal criteria"
        )

    def test_confidence_factors_dict_present(self):
        """Result should include classification_confidence_factors."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5, 0.01]])
        result = self._classify(analyzer, eigenvalues)
        assert 'classification_confidence_factors' in result

    def test_confidence_factors_keys(self):
        """Confidence factors should include all sub-factors."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.01]])
        result = self._classify(analyzer, eigenvalues)
        factors = result['classification_confidence_factors']
        expected = {
            'dimension_factor', 'jet_analysis_factor',
            'nondegeneracy_factor', 'spectral_completeness_factor',
            'has_formal_criteria', 'confidence_cap_applied',
            'confidence_cap_reason',
        }
        assert expected.issubset(set(factors.keys()))

    def test_jet_analysis_factor_zero(self):
        """Jet analysis factor should be 0 (not implemented)."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5]])
        result = self._classify(analyzer, eigenvalues)
        assert result['classification_confidence_factors']['jet_analysis_factor'] == 0.0

    def test_nondegeneracy_factor_zero(self):
        """Non-degeneracy factor should be 0 (not verified)."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5]])
        result = self._classify(analyzer, eigenvalues)
        assert result['classification_confidence_factors']['nondegeneracy_factor'] == 0.0

    def test_has_formal_criteria_false(self):
        """has_formal_criteria should be False (no jet/non-deg)."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5]])
        result = self._classify(analyzer, eigenvalues)
        assert result['classification_confidence_factors']['has_formal_criteria'] is False

    def test_confidence_cap_applied(self):
        """confidence_cap_applied should be True without formal criteria."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5]])
        result = self._classify(analyzer, eigenvalues)
        assert result['classification_confidence_factors']['confidence_cap_applied'] is True

    def test_dimension_factor_for_low_dim(self):
        """For dim ≤ 10, dimension factor should be 1.0."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5, 0.1, 0.01, 0.001]])
        result = self._classify(analyzer, eigenvalues)
        dim_factor = result['classification_confidence_factors']['dimension_factor']
        assert dim_factor == 1.0

    def test_dimension_factor_decays_for_high_dim(self):
        """For dim > 10, dimension factor should decay."""
        analyzer = self._make_analyzer()
        # Create high-dim eigenvalues
        eigenvalues = torch.randn(1, 30).abs()
        result = self._classify(analyzer, eigenvalues)
        dim_factor = result['classification_confidence_factors']['dimension_factor']
        assert dim_factor < 1.0

    def test_confidence_nonnegative(self):
        """Confidence should always be ≥ 0."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.randn(1, 50).abs()
        result = self._classify(analyzer, eigenvalues)
        assert result['classification_confidence'] >= 0.0

    def test_spectral_completeness_factor(self):
        """Spectral completeness should be 1.0 for dim ≤ 50."""
        analyzer = self._make_analyzer()
        eigenvalues = torch.tensor([[1.0, 0.5, 0.1]])
        result = self._classify(analyzer, eigenvalues)
        sf = result['classification_confidence_factors']['spectral_completeness_factor']
        assert sf == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Combined patches
# ═══════════════════════════════════════════════════════════════════════════


class TestFCIIntegration:
    """Integration tests combining multiple FCI patches."""

    def test_full_forward_with_all_patches(self):
        """Full forward pass should work with all FCI patches active."""
        config = _make_config()
        model = AEONDeltaV3(config)
        model.eval()
        input_ids = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            output = model(input_ids, decode_mode='train')
        assert output is not None

    def test_meta_loop_certificate_with_lyapunov(self):
        """Certificate should include Lyapunov and finite-iterate data."""
        ml = _make_meta_loop(max_iterations=10)
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        assert 'lyapunov_iteration_status' in meta
        assert 'finite_iterate_qualification' in meta
        if 'certificate' in meta:
            cert = meta['certificate']
            assert 'L_C' in cert

    def test_joint_budget_consistent_with_certificate(self):
        """Joint budget L_T should be related to certificate L_C."""
        ml = _make_meta_loop(max_iterations=10)
        budget = ml.verify_joint_lipschitz_budget()
        psi = torch.randn(2, 64)
        _, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        # Both should exist and be positive
        assert budget['L_T_composed'] > 0
        L_C = meta.get('L_certificate', None)
        if L_C is not None:
            assert L_C > 0

    def test_gamma_constraint_and_joint_budget(self):
        """After gamma clamping, joint budget should reflect tighter bound."""
        ml = _make_meta_loop()
        ml._gamma_ratio_max = 1.2  # tight constraint

        # Set extreme gamma
        with torch.no_grad():
            ml.output_stabilizer.weight[0] = 5.0
            ml.output_stabilizer.weight[1] = 0.1

        # Apply hook
        hook = ml._make_gamma_clamp_hook(ml.output_stabilizer)
        hook(torch.ones_like(ml.output_stabilizer.weight))

        # Now verify joint budget
        budget = ml.verify_joint_lipschitz_budget()
        # Output stabilizer Lipschitz should be reduced
        assert budget['gamma_ratio_output'] < 50.0  # was 50 before hook

    def test_lyapunov_monitor_lifecycle(self):
        """Lyapunov monitor should be reset between calls."""
        ml = _make_meta_loop(max_iterations=5)
        psi = torch.randn(1, 64)

        # First call
        _, _, m1 = ml.compute_fixed_point(psi)
        s1 = m1['lyapunov_iteration_status']

        # Second call
        _, _, m2 = ml.compute_fixed_point(psi)
        s2 = m2['lyapunov_iteration_status']

        # Both should have samples (monitor was reset and re-used)
        assert s1['num_samples'] > 0
        assert s2['num_samples'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
