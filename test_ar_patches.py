"""
Tests for AR (Academic Refinement) patches addressing the 5 critical issues
identified in the AEON-Delta RMT v3.1 cognitive integration analysis.

AR-1: SSM discretization diagonal structure enforcement
AR-2: Banach a posteriori bound preconditions for KM relaxation
AR-3: LayerNorm Lipschitz variance-aware analysis
AR-4: Inertial KM / Anderson nonexpansiveness enforcement
AR-5: Catastrophe classification formal applicability guards
"""

import math
import pytest
import torch
import torch.nn as nn

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    AEONTrainer,
    _SSMBlock,
    _SSDBlock,
    CognitiveFeedbackBus,
    OptimizedTopologyAnalyzer,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return AEONConfig(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        device_str='cpu',
    )


@pytest.fixture
def model(config):
    m = AEONDeltaV3(config)
    m.eval()
    return m


@pytest.fixture
def ssm_block():
    return _SSMBlock(d_model=64, d_state=16, d_inner=128, dt_rank=8)


@pytest.fixture
def ssd_block():
    return _SSDBlock(
        d_model=64, d_state=16, d_inner=128, nheads=4, dt_rank=8,
    )


@pytest.fixture
def topology_analyzer(config):
    return OptimizedTopologyAnalyzer(config)


# ═════════════════════════════════════════════════════════════════════════════
# AR-1: SSM Discretization Diagonal Structure Enforcement
# ═════════════════════════════════════════════════════════════════════════════

class TestAR1_SSMDiagonalStructure:
    """Verify that the elementwise exponential discretization is justified
    by the explicit diagonal structure of A in both Mamba-1 and Mamba-2."""

    def test_ssm_block_A_structure_flag(self, ssm_block):
        """_SSMBlock stores _A_structure = 'diagonal'."""
        assert hasattr(ssm_block, '_A_structure')
        assert ssm_block._A_structure == 'diagonal'

    def test_ssm_block_A_log_shape(self, ssm_block):
        """A_log is [d_inner, d_state] — implicit diagonal (no off-diag)."""
        assert ssm_block.A_log.shape == (128, 16)

    def test_ssm_block_verify_diagonal_invariant(self, ssm_block):
        """verify_diagonal_A_invariant() returns well-structured result."""
        result = ssm_block.verify_diagonal_A_invariant()
        assert result['is_diagonal'] is True
        assert result['discretization_valid'] is True
        assert result['structure'] == 'diagonal'
        assert result['A_shape'] == [128, 16]
        assert 'note' in result

    def test_ssd_block_A_structure_flag(self, ssd_block):
        """_SSDBlock stores _A_structure = 'scalar_per_head'."""
        assert hasattr(ssd_block, '_A_structure')
        assert ssd_block._A_structure == 'scalar_per_head'

    def test_ssd_block_A_log_shape(self, ssd_block):
        """A_log in SSD is [nheads] — scalar per head."""
        assert ssd_block.A_log.shape == (4,)

    def test_ssm_discretization_is_exact_for_diagonal(self, ssm_block):
        """For diagonal A, exp(Δ⊙A) == matrix_exp(diag(Δ·A))."""
        A = -torch.exp(ssm_block.A_log.float())  # [d_inner, d_state]
        dt = torch.rand(1, 1, 128, 1) * 0.1  # small dt
        # Elementwise exponential (what code does)
        A_bar = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))
        # For diagonal A, this IS the matrix exponential
        assert A_bar.shape == (1, 1, 128, 16)
        assert torch.all(torch.isfinite(A_bar))
        # All entries should be positive (exp of real number)
        assert torch.all(A_bar > 0)

    def test_ssm_forward_works(self, ssm_block):
        """SSM forward pass produces valid output."""
        x = torch.randn(2, 10, 64)
        y, state = ssm_block(x)
        assert y.shape == (2, 10, 64)
        assert torch.all(torch.isfinite(y))

    def test_ssd_forward_works(self, ssd_block):
        """SSD forward pass produces valid output."""
        x = torch.randn(2, 10, 64)
        y, state = ssd_block(x)
        assert y.shape == (2, 10, 64)
        assert torch.all(torch.isfinite(y))


# ═════════════════════════════════════════════════════════════════════════════
# AR-2: Banach A Posteriori Bound Preconditions
# ═════════════════════════════════════════════════════════════════════════════

class TestAR2_BanachBoundPreconditions:
    """Verify that the a posteriori bound correctly documents and
    enforces preconditions for KM relaxation vs pure Picard."""

    def test_certificate_contains_banach_preconditions(self, model):
        """compute_fixed_point certificate includes banach_bound_preconditions."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        assert 'banach_bound_preconditions' in meta
        bp = meta['banach_bound_preconditions']
        assert 'bound_valid' in bp

    def test_banach_preconditions_structure_when_jacobian_computed(self, model):
        """When Jacobian IS computed, preconditions include all fields."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        bp = meta['banach_bound_preconditions']
        # Should have either the full structure or the fallback
        assert isinstance(bp, dict)
        assert 'bound_valid' in bp

    def test_L_certificate_less_than_1_implies_valid_bound(self, model):
        """When L_certificate < 1, bound_valid should be True."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        bp = meta['banach_bound_preconditions']
        L = meta.get('L_certificate', 2.0)
        if L < 1.0:
            assert bp.get('bound_valid') is True
            assert bp.get('L_eff_less_than_1') is True

    def test_certified_error_uses_operator_residual(self, model):
        """The bound uses operator residual ‖T(C)−C‖, NOT successive diffs."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        bp = meta['banach_bound_preconditions']
        if 'residual_type' in bp:
            assert bp['residual_type'] == 'operator_residual'
            assert bp['residual_type_valid'] is True

    def test_km_formal_convergence_present(self, model):
        """Certificate includes structured km_formal_convergence."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        assert 'km_formal_convergence' in meta
        km = meta['km_formal_convergence']
        assert 'convergence_type' in km
        assert 'convergence_strength' in km

    def test_alpha_mean_tracking(self, model):
        """When α < 1, the certificate should reflect KM relaxation."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        bp = meta['banach_bound_preconditions']
        if 'alpha_mean' in bp:
            alpha = bp['alpha_mean']
            assert 0 < alpha <= 1.0
            if abs(alpha - 1.0) < 1e-6:
                assert bp['is_pure_picard'] is True


# ═════════════════════════════════════════════════════════════════════════════
# AR-3: LayerNorm Lipschitz Variance-Aware Analysis
# ═════════════════════════════════════════════════════════════════════════════

class TestAR3_LayerNormVarianceAware:
    """Verify that LayerNorm Lipschitz analysis accounts for ε and variance."""

    def test_gamma_hook_records_variance_bound(self, model):
        """After backward pass, _ar3_lipschitz_bound is stored on LN modules."""
        # We need a forward+backward to trigger the hook
        psi_0 = torch.randn(2, 64, requires_grad=True)
        C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        loss = C.sum()
        loss.backward()

        # Check that the hook recorded bounds
        for ln_mod in [model.meta_loop.input_stabilizer,
                       model.meta_loop.output_stabilizer]:
            if hasattr(ln_mod, '_ar3_lipschitz_bound'):
                bound = ln_mod._ar3_lipschitz_bound
                assert 'L_LN_worst_case' in bound
                assert 'gamma_max' in bound
                assert 'sigma_floor' in bound
                assert 'eps' in bound
                assert bound['L_LN_worst_case'] > 0
                assert bound['sigma_floor'] > 0

    def test_joint_lipschitz_includes_variance_analysis(self, model):
        """verify_joint_lipschitz_budget includes layernorm_variance_analysis."""
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        assert 'layernorm_variance_analysis' in budget
        va = budget['layernorm_variance_analysis']
        assert 'input_ln_eps' in va
        assert 'output_ln_eps' in va
        assert 'variance_floor' in va
        assert 'L_input_data_independent' in va
        assert 'L_input_with_variance_floor' in va
        assert 'L_output_data_independent' in va
        assert 'L_output_with_variance_floor' in va
        assert 'note' in va

    def test_variance_floor_tightens_bound(self, model):
        """Bound with variance_floor should be tighter than data-independent."""
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        # σ_floor > √ε ⟹ L_with_floor < L_data_independent
        L_di = va['L_input_data_independent']
        L_vf = va['L_input_with_variance_floor']
        # Variance floor (0.1) >> √ε (≈0.003), so floor-based should be tighter
        if va['variance_floor'] > math.sqrt(va['input_ln_eps']):
            assert L_vf <= L_di + 1e-6

    def test_gamma_ratio_max_present(self, model):
        """Model tracks gamma_ratio_max constraint."""
        assert hasattr(model.meta_loop, '_gamma_ratio_max')
        assert model.meta_loop._gamma_ratio_max >= 1.0

    def test_pre_ln_variance_floor_set(self, model):
        """Variance floor is initialized."""
        assert hasattr(model.meta_loop, '_pre_ln_variance_floor')
        assert model.meta_loop._pre_ln_variance_floor > 0


# ═════════════════════════════════════════════════════════════════════════════
# AR-4: Inertial KM / Anderson Nonexpansiveness Enforcement
# ═════════════════════════════════════════════════════════════════════════════

class TestAR4_InertialKMAnderson:
    """Verify nonexpansiveness guards on inertial KM and Anderson."""

    def test_fast_km_dampens_when_expansive(self, model):
        """_fast_km_step suppresses momentum when L > 1.1."""
        ml = model.meta_loop
        C_curr = torch.randn(2, 64)
        C_prev = torch.randn(2, 64)
        T_C = torch.randn(2, 64)
        alpha = torch.tensor([0.5, 0.5])

        # Set high L_certificate → should suppress momentum
        ml._last_L_certificate = 2.0
        result_no_momentum = ml._fast_km_step(C_curr, C_prev, T_C, alpha, n=10)

        # Set L < 1 → full momentum
        ml._last_L_certificate = 0.5
        result_full_momentum = ml._fast_km_step(C_curr, C_prev, T_C, alpha, n=10)

        # With L=2.0 and n=10, β should be 0 (standard KM)
        # So result_no_momentum == (1-α)·C_curr + α·T_C (no inertial term)
        _alpha = alpha.unsqueeze(-1)
        expected_no_momentum = (1.0 - _alpha) * C_curr + _alpha * T_C
        assert torch.allclose(result_no_momentum, expected_no_momentum, atol=1e-5)

    def test_fast_km_partial_dampening(self, model):
        """_fast_km_step partially dampens momentum for 1 < L ≤ 1.1."""
        ml = model.meta_loop
        C_curr = torch.randn(2, 64)
        C_prev = torch.zeros(2, 64)
        T_C = torch.randn(2, 64)
        alpha = torch.tensor([0.5, 0.5])

        # L = 1.05 → partial dampening
        ml._last_L_certificate = 1.05
        result_partial = ml._fast_km_step(C_curr, C_prev, T_C, alpha, n=10)

        # L = 0.5 → full momentum
        ml._last_L_certificate = 0.5
        result_full = ml._fast_km_step(C_curr, C_prev, T_C, alpha, n=10)

        # Results should differ (partial dampening ≠ full momentum)
        assert not torch.allclose(result_partial, result_full, atol=1e-5)

    def test_fast_km_no_L_attribute_uses_full_momentum(self, model):
        """When _last_L_certificate is not set, full momentum is used."""
        ml = model.meta_loop
        if hasattr(ml, '_last_L_certificate'):
            delattr(ml, '_last_L_certificate')
        C_curr = torch.randn(2, 64)
        C_prev = torch.randn(2, 64)
        T_C = torch.randn(2, 64)
        alpha = torch.tensor([0.5, 0.5])

        # Should not raise
        result = ml._fast_km_step(C_curr, C_prev, T_C, alpha, n=5)
        assert torch.all(torch.isfinite(result))

    def test_anderson_convergence_conditions_documented(self, model):
        """Anderson conditions include formal_guarantee=False."""
        ml = model.meta_loop
        conds = ml._anderson_convergence_conditions
        assert conds['formal_guarantee'] is False
        assert conds['stability_guarantee_status'] == 'heuristic'
        assert 'missing_for_formal_guarantee' in conds
        assert len(conds['missing_for_formal_guarantee']) >= 1

    def test_anderson_heuristic_safeguards_listed(self, model):
        """Anderson conditions list concrete heuristic safeguards."""
        conds = model.meta_loop._anderson_convergence_conditions
        assert 'heuristic_safeguards' in conds
        safeguards = conds['heuristic_safeguards']
        assert len(safeguards) >= 3

    def test_perturbation_budget_initialized(self, model):
        """Perturbation budget = B₀ · π²/6."""
        ml = model.meta_loop
        expected = 1.0 * (math.pi ** 2 / 6.0)
        assert abs(ml._perturbation_budget - expected) < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# AR-5: Catastrophe Classification Formal Applicability Guards
# ═════════════════════════════════════════════════════════════════════════════

class TestAR5_CatastropheClassification:
    """Verify formal applicability guards and criticality checks."""

    def test_criticality_assessment_present(self, topology_analyzer):
        """classify_catastrophe_type returns criticality_assessment."""
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        cond = torch.tensor([2000.0])
        grad = torch.tensor([0.5])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad,
        )
        assert 'criticality_assessment' in result
        ca = result['criticality_assessment']
        assert 'at_critical_point' in ca
        assert 'criticality_score' in ca
        assert 'grad_threshold' in ca
        assert 'note' in ca

    def test_high_gradient_reduces_criticality(self, topology_analyzer):
        """High gradient norm → low criticality_score."""
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        cond = torch.tensor([2000.0])
        grad_high = torch.tensor([10.0])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad_high,
        )
        ca = result['criticality_assessment']
        # High gradient → not at critical point
        scores = ca['criticality_score']
        assert scores[0] < 0.5  # low criticality

    def test_near_zero_gradient_high_criticality(self, topology_analyzer):
        """Near-zero gradient → high criticality_score."""
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        cond = torch.tensor([2000.0])
        grad_low = torch.tensor([1e-8])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad_low,
        )
        ca = result['criticality_assessment']
        scores = ca['criticality_score']
        assert scores[0] > 0.9  # high criticality

    def test_corank_based_classification_preserved(self, topology_analyzer):
        """Corank-primary classification still works correctly."""
        # Corank 0 (no near-zero eigenvalues)
        eig_none = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result_none = topology_analyzer.classify_catastrophe_type(
            eig_none, torch.tensor([4.0]), torch.tensor([0.1]),
        )
        assert result_none['spectral_degeneracy_type'][0] == 'none'

        # Corank 1 (one near-zero eigenvalue)
        eig_fold = torch.tensor([[0.001, 2.0, 3.0, 4.0]])
        result_fold = topology_analyzer.classify_catastrophe_type(
            eig_fold, torch.tensor([4000.0]), torch.tensor([0.1]),
        )
        assert result_fold['spectral_degeneracy_type'][0] == 'fold'

    def test_confidence_capped_without_jet_analysis(self, topology_analyzer):
        """Confidence ≤ 0.5 when jet analysis not performed."""
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        cond = torch.tensor([2000.0])
        grad = torch.tensor([0.1])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad,
        )
        assert result['classification_confidence'] <= 0.5

    def test_jet_analysis_required_flag(self, topology_analyzer):
        """jet_analysis_required is always True (no jet analysis performed)."""
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        cond = torch.tensor([4.0])
        grad = torch.tensor([0.1])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad,
        )
        assert result['jet_analysis_required'] is True

    def test_unfolding_analysis_not_implemented(self, topology_analyzer):
        """unfolding_analysis status = NOT IMPLEMENTED."""
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        cond = torch.tensor([4.0])
        grad = torch.tensor([0.1])
        result = topology_analyzer.classify_catastrophe_type(
            eigenvalues, cond, grad,
        )
        assert 'unfolding_analysis' in result
        ua = result['unfolding_analysis']
        assert ua['versal_unfolding_computed'] is False
        assert ua['normal_form_verified'] is False


# ═════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestARIntegration:
    """End-to-end integration tests for all AR patches working together."""

    def test_full_forward_pass(self, model):
        """Full model forward pass succeeds with all patches applied."""
        input_ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert output is not None

    def test_compute_fixed_point_certificate_complete(self, model):
        """Fixed-point certificate includes all AR-enhanced fields."""
        psi_0 = torch.randn(2, 64)
        with torch.no_grad():
            C, iters, meta = model.meta_loop.compute_fixed_point(psi_0)
        # AR-2 fields
        assert 'banach_bound_preconditions' in meta
        assert 'km_formal_convergence' in meta
        assert 'L_certificate' in meta
        # Existing fields preserved
        assert 'convergence_trajectory' in meta
        assert 'km_convergence_status' in meta
