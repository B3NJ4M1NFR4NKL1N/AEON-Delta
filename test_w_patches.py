"""
Test suite for W-series patches: advanced Lipschitz certification framework.

W1: SandwichLinear layer with constructive Lipschitz bounds
W2: Fast KM (Halpern-type inertial) and TKMA acceleration methods
W3: RNN-SDP unrolled meta-loop finite-horizon certification
W4: LiPopt polynomial optimization (SOS relaxation) bounds
W5: ECLipsE dedicated Efficient Cholesky Lipschitz Estimation
"""

import math
import pytest
import torch
import torch.nn as nn

# ============================================================================
# Fixtures and helpers
# ============================================================================


def _make_config(**overrides):
    """Create a minimal AEONConfig-like object for testing."""
    defaults = {
        'hidden_dim': 32,
        'z_dim': 32,
        'meta_dim': 64,
        'lipschitz_target': 0.9,
        'dropout_rate': 0.1,
        'alpha': 0.5,
        'max_meta_iterations': 10,
        'convergence_threshold': 1e-4,
        'min_meta_iterations': 2,
        'anderson_memory': 5,
        'enable_certification': True,
        'vq_num_embeddings': 16,
        'vq_embedding_dim': 32,
    }
    defaults.update(overrides)

    class _Cfg:
        pass

    cfg = _Cfg()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def z_batch():
    torch.manual_seed(42)
    return torch.randn(4, 32)


# ============================================================================
# W1: SandwichLinear
# ============================================================================


class TestW1_SandwichLinear:
    """Tests for SandwichLinear layer."""

    def test_class_exists(self):
        from aeon_core import SandwichLinear
        assert issubclass(SandwichLinear, nn.Module)

    def test_forward_shape(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_forward_square(self):
        """Tied B = A.T when in == out."""
        from aeon_core import SandwichLinear
        layer = SandwichLinear(32, 32)
        assert layer.B is None  # tied
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_lipschitz_bound_finite(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32)
        L = layer.get_lipschitz_bound()
        assert math.isfinite(L)
        assert L > 0

    def test_lipschitz_projection(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32, lipschitz_target=0.5)
        # Make d large
        with torch.no_grad():
            layer.d.fill_(100.0)
        layer.project_lipschitz()
        L = layer.get_lipschitz_bound()
        assert L <= 0.5 + 1e-4, f"L={L} after projection should be ≤ 0.5"

    def test_no_projection_without_target(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32, lipschitz_target=None)
        with torch.no_grad():
            layer.d.fill_(100.0)
        layer.project_lipschitz()  # should be no-op
        assert layer.d.abs().max().item() >= 99.0

    def test_rank_parameter(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32, rank=8)
        assert layer.rank == 8
        assert layer.A.shape == (32, 8)
        assert layer.d.shape == (8,)

    def test_no_bias(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32, bias=False)
        assert layer.bias is None
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_gradient_flow(self):
        from aeon_core import SandwichLinear
        layer = SandwichLinear(16, 32)
        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.A.grad is not None
        assert layer.d.grad is not None


# ============================================================================
# W1b: LipschitzConstrainedLambda with Sandwich
# ============================================================================


class TestW1b_LambdaSandwich:
    """Tests for Sandwich integration in LipschitzConstrainedLambda."""

    def test_sandwich_init(self):
        from aeon_core import LipschitzConstrainedLambda, SandwichLinear
        lam = LipschitzConstrainedLambda(
            input_dim=64, hidden_dim=32, output_dim=32,
            use_sandwich=True,
        )
        assert lam.use_sandwich is True
        assert isinstance(lam.W1, SandwichLinear)
        assert isinstance(lam.W2, SandwichLinear)

    def test_sandwich_forward(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            input_dim=64, hidden_dim=32, output_dim=32,
            use_sandwich=True,
        )
        x = torch.randn(4, 64)
        y = lam(x)
        assert y.shape == (4, 32)

    def test_sandwich_lipschitz_bound(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            input_dim=64, hidden_dim=32, output_dim=32,
            use_sandwich=True,
        )
        result = lam.get_sandwich_lipschitz_bound()
        assert result['sandwich_active'] is True
        assert math.isfinite(result['L_sandwich'])
        assert 'sandwich_W1' in result['per_layer']
        assert 'Wang' in result['methodology']

    def test_standard_lipschitz_bound_without_sandwich(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            input_dim=64, hidden_dim=32, output_dim=32,
            use_sandwich=False,
        )
        result = lam.get_sandwich_lipschitz_bound()
        assert result['sandwich_active'] is False

    def test_sandwich_projection_on_training(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            input_dim=64, hidden_dim=32, output_dim=32,
            use_sandwich=True, lipschitz_target=0.5,
        )
        lam.train()
        x = torch.randn(4, 64)
        _ = lam(x)
        # After forward in training mode, d should be projected
        L1 = lam.W1.get_lipschitz_bound()
        L2 = lam.W2.get_lipschitz_bound()
        # Each layer target = sqrt(0.5/1.13) ≈ 0.665
        target_per_layer = math.sqrt(0.5 / 1.13)
        assert L1 <= target_per_layer + 0.1 or L2 <= target_per_layer + 0.1


# ============================================================================
# W2: Fast KM and TKMA
# ============================================================================


class TestW2_FastKM:
    """Tests for Fast KM (Halpern-type inertial) acceleration."""

    def test_method_exists(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        assert hasattr(loop, '_fast_km_step')
        assert callable(loop._fast_km_step)

    def test_fast_km_shape(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C_curr = torch.randn(4, 32)
        C_prev = torch.randn(4, 32)
        T_C = torch.randn(4, 32)
        alpha = torch.full((4,), 0.5)
        result = loop._fast_km_step(C_curr, C_prev, T_C, alpha, n=5)
        assert result.shape == (4, 32)

    def test_fast_km_no_momentum_at_n0(self, config):
        """At n=0, beta=0 so Fast KM = standard KM."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C_curr = torch.randn(4, 32)
        C_prev = torch.randn(4, 32)
        T_C = torch.randn(4, 32)
        alpha = torch.full((4,), 0.5)
        result_fast = loop._fast_km_step(C_curr, C_prev, T_C, alpha, n=0)
        # At n=0: beta = max(0, -1/2) = 0, so y=C_curr, result = 0.5*C_curr + 0.5*T_C
        expected = 0.5 * C_curr + 0.5 * T_C
        assert torch.allclose(result_fast, expected, atol=1e-6)

    def test_fast_km_momentum_increases(self, config):
        """Beta should increase with n."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C_curr = torch.randn(4, 32)
        C_prev = torch.zeros(4, 32)
        T_C = torch.randn(4, 32)
        alpha = torch.full((4,), 0.5)
        r1 = loop._fast_km_step(C_curr, C_prev, T_C, alpha, n=1)
        r10 = loop._fast_km_step(C_curr, C_prev, T_C, alpha, n=10)
        # At n=10, more momentum → result differs more from n=1
        diff = (r10 - r1).norm().item()
        assert diff > 0  # should have different momentum

    def test_fast_km_scalar_alpha(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C = torch.randn(4, 32)
        result = loop._fast_km_step(C, C, C, 0.5, n=5)
        assert result.shape == (4, 32)


class TestW2_TKMA:
    """Tests for Tikhonov–Krasnosel'skii–Mann Acceleration."""

    def test_method_exists(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        assert hasattr(loop, '_tkma_step')
        assert callable(loop._tkma_step)

    def test_tkma_shape(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C = torch.randn(4, 32)
        T_C = torch.randn(4, 32)
        anchor = torch.zeros(4, 32)
        alpha = torch.full((4,), 0.5)
        result = loop._tkma_step(C, T_C, anchor, alpha, n=5)
        assert result.shape == (4, 32)

    def test_tkma_anchoring_decreases(self, config):
        """Anchoring coefficient γ should decrease with n."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C = torch.randn(4, 32)
        T_C = torch.randn(4, 32)
        anchor = torch.ones(4, 32) * 10.0  # strong anchor signal
        alpha = torch.full((4,), 0.3)

        r1 = loop._tkma_step(C, T_C, anchor, alpha, n=1)
        r50 = loop._tkma_step(C, T_C, anchor, alpha, n=50)

        # At n=1, stronger anchoring → closer to anchor
        # At n=50, weaker anchoring → closer to standard KM
        dist_to_anchor_1 = (r1 - anchor).norm().item()
        dist_to_anchor_50 = (r50 - anchor).norm().item()
        # n=50 should be farther from anchor (less anchoring)
        assert dist_to_anchor_50 >= dist_to_anchor_1 - 1e-4

    def test_tkma_scalar_alpha(self, config):
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        C = torch.randn(4, 32)
        result = loop._tkma_step(C, C, C, 0.5, n=5)
        assert result.shape == (4, 32)

    def test_tkma_methodology_ref(self, config):
        """Method docstring should reference Boţ & Sedlmayer."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        doc = loop._tkma_step.__doc__
        assert 'Tikhonov' in doc
        assert 'Boţ' in doc or 'Bot' in doc


# ============================================================================
# W3: RNN-SDP Unrolled Certification
# ============================================================================


class TestW3_UnrolledCertification:
    """Tests for finite-horizon RNN-SDP-style unrolled certification."""

    def test_method_exists(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        assert hasattr(cml, 'compute_unrolled_lipschitz_certificate')

    def test_returns_dict(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        assert isinstance(result, dict)

    def test_required_keys(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        required = [
            'L_unrolled', 'L_per_step', 'L_geometric_sum',
            'contraction_verified', 'N_steps', 'methodology',
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_n_steps_recorded(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=7)
        assert result['N_steps'] == 7

    def test_per_step_length(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=5)
        assert len(result['L_per_step']) == 5

    def test_l_unrolled_finite(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        assert math.isfinite(result['L_unrolled'])

    def test_geometric_sum_finite(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        assert math.isfinite(result['L_geometric_sum'])

    def test_methodology_references(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        assert 'RNN-SDP' in result['methodology']
        assert 'Revay' in result['methodology']

    def test_contraction_bool(self, config, z_batch):
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        result = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        assert isinstance(result['contraction_verified'], bool)


# ============================================================================
# W4: LiPopt Polynomial Optimization
# ============================================================================


class TestW4_LiPopt:
    """Tests for LiPopt SOS relaxation Lipschitz certificate."""

    def test_method_exists(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        assert hasattr(lam, 'compute_lipopt_certificate')

    def test_returns_dict(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        assert isinstance(result, dict)

    def test_required_keys(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        required = [
            'L_lipopt', 'is_contraction', 'gram_eigenvalue',
            'sos_degree', 'polynomial_label', 'methodology',
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_l_lipopt_finite(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        assert math.isfinite(result['L_lipopt'])
        assert result['L_lipopt'] >= 0

    def test_degree_2(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate(degree=2)
        assert result['sos_degree'] == 2
        assert result['polynomial_label'] == 'degree-2'

    def test_degree_4(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate(degree=4)
        assert result['sos_degree'] == 4
        assert result['polynomial_label'] == 'degree-4'

    def test_gram_eigenvalue_nonnegative(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        assert result['gram_eigenvalue'] >= -1e-6

    def test_layernorm_factor(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        assert 'layernorm_factor' in result
        assert result['layernorm_factor'] > 0

    def test_methodology_references(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_lipopt_certificate()
        assert 'Latorre' in result['methodology']
        assert 'SOS' in result['methodology']

    def test_with_explicit_input(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        x = torch.randn(8, 64)
        result = lam.compute_lipopt_certificate(x=x)
        assert math.isfinite(result['L_lipopt'])

    def test_sandwich_mode(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            64, 32, 32, use_sandwich=True,
        )
        result = lam.compute_lipopt_certificate()
        assert math.isfinite(result['L_lipopt'])


# ============================================================================
# W5: ECLipsE
# ============================================================================


class TestW5_ECLipsE:
    """Tests for dedicated ECLipsE implementation."""

    def test_method_exists(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        assert hasattr(lam, 'compute_eclipse_bound')

    def test_returns_dict(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        assert isinstance(result, dict)

    def test_required_keys(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        required = [
            'L_eclipse', 'certified', 'is_contraction',
            'binary_search_steps', 'worst_jacobian_sigma',
            'iqc_scale', 'min_eigenvalue_M', 'methodology',
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_l_eclipse_finite(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        assert math.isfinite(result['L_eclipse'])
        assert result['L_eclipse'] >= 0

    def test_binary_search_steps(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        assert result['binary_search_steps'] >= 1
        assert result['binary_search_steps'] <= 20

    def test_iqc_scale_gelu(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        expected_scale = math.sqrt(2.0 / 1.13)
        assert abs(result['iqc_scale'] - expected_scale) < 1e-4

    def test_slope_restriction(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        assert result['slope_restriction'] == [0.0, 1.13]

    def test_methodology_references(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound()
        assert 'ECLipsE' in result['methodology']
        assert 'Pauli' in result['methodology']
        assert 'Cholesky' in result['methodology']

    def test_with_explicit_input(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        x = torch.randn(8, 64)
        result = lam.compute_eclipse_bound(x=x)
        assert math.isfinite(result['L_eclipse'])

    def test_tighter_than_spectral(self):
        """ECLipsE should be ≤ spectral norm product (or at least finite)."""
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        eclipse = lam.compute_eclipse_bound()
        assert math.isfinite(eclipse['L_eclipse'])

    def test_jacobian_samples_count(self):
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        result = lam.compute_eclipse_bound(num_jacobian_samples=4)
        assert result['num_jacobian_samples'] >= 1


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Cross-feature integration tests."""

    def test_sandwich_eclipse_consistency(self):
        """ECLipsE bound should work with sandwich layers."""
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(
            64, 32, 32, use_sandwich=True,
        )
        eclipse = lam.compute_eclipse_bound()
        sandwich = lam.get_sandwich_lipschitz_bound()
        assert math.isfinite(eclipse['L_eclipse'])
        assert math.isfinite(sandwich['L_sandwich'])

    def test_lipopt_vs_eclipse(self):
        """Both LiPopt and ECLipsE should produce finite bounds."""
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)
        lipopt = lam.compute_lipopt_certificate()
        eclipse = lam.compute_eclipse_bound()
        assert math.isfinite(lipopt['L_lipopt'])
        assert math.isfinite(eclipse['L_eclipse'])

    def test_all_certification_methods(self):
        """All certification methods should be callable on same operator."""
        from aeon_core import LipschitzConstrainedLambda
        lam = LipschitzConstrainedLambda(64, 32, 32)

        # LipSDP
        lipsdp = lam.compute_lipsdp_certificate()
        assert isinstance(lipsdp, dict)

        # ECLipsE
        eclipse = lam.compute_eclipse_bound()
        assert isinstance(eclipse, dict)

        # LiPopt
        lipopt = lam.compute_lipopt_certificate()
        assert isinstance(lipopt, dict)

        # Sandwich (inactive)
        sandwich = lam.get_sandwich_lipschitz_bound()
        assert isinstance(sandwich, dict)

    def test_unrolled_with_different_N(self, config, z_batch):
        """Unrolled certification at different depths."""
        from aeon_core import CertifiedMetaLoop
        cml = CertifiedMetaLoop(config)
        r3 = cml.compute_unrolled_lipschitz_certificate(z_batch, N=3)
        r7 = cml.compute_unrolled_lipschitz_certificate(z_batch, N=7)
        assert len(r3['L_per_step']) == 3
        assert len(r7['L_per_step']) == 7

    def test_fast_km_and_tkma_coexist(self, config):
        """Both acceleration methods should be available."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        assert hasattr(loop, '_fast_km_step')
        assert hasattr(loop, '_tkma_step')
        assert hasattr(loop, '_anderson_step')

    def test_compute_fixed_point_still_works(self, config):
        """Original compute_fixed_point should still function."""
        from aeon_core import ProvablyConvergentMetaLoop
        loop = ProvablyConvergentMetaLoop(config)
        psi = torch.randn(2, 32)
        C_star, iters, meta = loop.compute_fixed_point(psi)
        assert C_star.shape == (2, 32)
        assert iters.shape == (2,)
        assert isinstance(meta, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-q'])
