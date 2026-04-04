"""Tests for RIGOR-1 through RIGOR-5 patches.

RIGOR-1: SSM diagonal structure enforcement (verify_diagonal_A_invariant, _A_structure)
RIGOR-2: Banach bound preconditions (iteration_mode, picard vs km_relaxed)
RIGOR-3: LayerNorm ε-aware variance analysis (layernorm_variance_analysis)
RIGOR-4: Inertial KM stability guarantee (stability_guarantee, _last_inertial_km_metadata)
RIGOR-5: Catastrophe criticality assessment (criticality_assessment, gradient-vanishing)
"""
import math
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    _SSMBlock,
    _SSDBlock,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return AEONConfig(device_str='cpu')


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
    return _SSDBlock(d_model=64, d_state=16, d_inner=128, nheads=4, dt_rank=8)


# ──────────────────────────────────────────────────────────────────────────────
# RIGOR-1: SSM diagonal structure enforcement
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGOR1_SSMDiagonalStructure:
    """Tests for RIGOR-1: diagonal structure invariant enforcement."""

    def test_ssm_block_has_A_structure(self, ssm_block):
        assert hasattr(ssm_block, '_A_structure')
        assert ssm_block._A_structure == 'diagonal'

    def test_ssd_block_has_A_structure(self, ssd_block):
        assert hasattr(ssd_block, '_A_structure')
        assert ssd_block._A_structure == 'scalar_per_head'

    def test_ssm_verify_diagonal_A_invariant_exists(self, ssm_block):
        assert hasattr(ssm_block, 'verify_diagonal_A_invariant')
        assert callable(ssm_block.verify_diagonal_A_invariant)

    def test_ssm_verify_diagonal_A_invariant_returns_dict(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert isinstance(result, dict)

    def test_ssm_invariant_is_diagonal(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert result['is_diagonal'] is True
        assert result['A_structure'] == 'diagonal'

    def test_ssm_invariant_elementwise_exp_valid(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert result['elementwise_exp_valid'] is True

    def test_ssm_invariant_commutativity_satisfied(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert result['commutativity_satisfied'] is True

    def test_ssm_invariant_a_log_shape(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert result['A_log_shape'] == (128, 16)  # d_inner, d_state

    def test_ssm_invariant_mathematical_justification(self, ssm_block):
        result = ssm_block.verify_diagonal_A_invariant()
        assert 'mathematical_justification' in result
        assert 'diag' in result['mathematical_justification'].lower()
        assert 'exact' in result['mathematical_justification'].lower()

    def test_ssm_A_log_is_2d(self, ssm_block):
        """A_log must be 2-D [d_inner, d_state] for diagonal validity."""
        assert ssm_block.A_log.ndim == 2

    def test_ssd_A_log_is_1d_scalar_per_head(self, ssd_block):
        """SSD A_log must be 1-D [nheads] for scalar-per-head validity."""
        assert ssd_block.A_log.ndim == 1
        assert ssd_block.A_log.shape[0] == 4  # nheads=4


# ──────────────────────────────────────────────────────────────────────────────
# RIGOR-2: Banach bound preconditions
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGOR2_BanachBoundPreconditions:
    """Tests for RIGOR-2: iteration_mode and Banach bound validity."""

    def test_certificate_has_banach_bound_preconditions(self, model):
        """When Banach contraction holds, preconditions dict must be present."""
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            a_post = convergence.get('a_posteriori_validity', {})
            assert 'banach_bound_preconditions' in a_post

    def test_banach_preconditions_has_iteration_mode(self, model):
        """Preconditions must distinguish picard vs km_relaxed."""
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            assert 'iteration_mode' in preconds
            assert preconds['iteration_mode'] in ('picard', 'km_relaxed')

    def test_banach_preconditions_has_alpha_n(self, model):
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            assert 'alpha_n' in preconds

    def test_banach_successive_diff_bound_picard_only(self, model):
        """successive_difference_bound_valid should be True only for picard."""
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            if preconds['iteration_mode'] == 'picard':
                assert preconds['successive_difference_bound_valid'] is True
            else:
                assert preconds['successive_difference_bound_valid'] is False

    def test_banach_preconditions_has_L_effective(self, model):
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            assert 'L_effective' in preconds
            assert 'L_bare' in preconds

    def test_banach_preconditions_has_bound_formula(self, model):
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            assert 'bound_formula' in preconds

    def test_banach_preconditions_has_inertial_caveat(self, model):
        psi_0 = torch.randn(1, model.config.hidden_dim)
        _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        convergence = meta.get('km_formal_convergence', {})
        if convergence.get('convergence_type') == 'banach_contraction':
            preconds = convergence['a_posteriori_validity']['banach_bound_preconditions']
            assert 'inertial_bound_caveat' in preconds
            assert 'summable' in preconds['inertial_bound_caveat'].lower()


# ──────────────────────────────────────────────────────────────────────────────
# RIGOR-3: LayerNorm ε-aware variance analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGOR3_LayerNormVarianceAnalysis:
    """Tests for RIGOR-3: ε-aware Lipschitz bound for LayerNorm."""

    def test_joint_budget_has_variance_analysis(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        assert 'layernorm_variance_analysis' in budget

    def test_variance_analysis_has_epsilon(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        assert 'epsilon_input' in va
        assert 'epsilon_output' in va
        assert va['epsilon_input'] > 0
        assert va['epsilon_output'] > 0

    def test_variance_analysis_worst_case_L(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        assert 'worst_case_L_input' in va
        assert 'worst_case_L_output' in va
        # Worst case should be finite and positive
        assert math.isfinite(va['worst_case_L_input'])
        assert math.isfinite(va['worst_case_L_output'])
        assert va['worst_case_L_input'] > 0
        assert va['worst_case_L_output'] > 0

    def test_variance_analysis_floor_sufficiency(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        assert 'variance_floor_sufficient' in va
        assert isinstance(va['variance_floor_sufficient'], bool)

    def test_variance_analysis_gamma_hook_limitation(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        assert 'gamma_hook_limitation' in va
        assert 'γ_max' in va['gamma_hook_limitation']

    def test_variance_analysis_formal_bound_reference(self, model):
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        assert 'formal_bound_reference' in va
        # Should reference the formal bound Lip(LN) = ...
        assert 'Lip' in va['formal_bound_reference']

    def test_worst_case_exceeds_average_case(self, model):
        """Worst-case L (σ→ε) should exceed the average-case heuristic bound."""
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        va = budget['layernorm_variance_analysis']
        factors = budget['factors']
        # Worst case when variance → 0 should be ≥ the heuristic bound
        assert va['worst_case_L_input'] >= factors['L_input_stabilizer']


# ──────────────────────────────────────────────────────────────────────────────
# RIGOR-4: Inertial KM stability guarantee
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGOR4_InertialKMStability:
    """Tests for RIGOR-4: stability_guarantee in inertial KM step."""

    def test_fast_km_step_exists(self, model):
        assert hasattr(model.meta_loop, '_fast_km_step')

    def test_fast_km_step_records_metadata(self, model):
        """After calling _fast_km_step, metadata should be recorded."""
        C_curr = torch.randn(1, model.config.hidden_dim)
        C_prev = torch.randn(1, model.config.hidden_dim)
        T_C = torch.randn(1, model.config.hidden_dim)
        alpha_n = torch.tensor([0.5])

        model.meta_loop._fast_km_step(C_curr, C_prev, T_C, alpha_n, n=5)
        assert hasattr(model.meta_loop, '_last_inertial_km_metadata')

    def test_inertial_metadata_has_stability_guarantee(self, model):
        C_curr = torch.randn(1, model.config.hidden_dim)
        C_prev = torch.randn(1, model.config.hidden_dim)
        T_C = torch.randn(1, model.config.hidden_dim)
        alpha_n = torch.tensor([0.5])

        model.meta_loop._fast_km_step(C_curr, C_prev, T_C, alpha_n, n=5)
        meta = model.meta_loop._last_inertial_km_metadata
        assert 'stability_guarantee' in meta
        assert meta['stability_guarantee'] == 'conditional'

    def test_inertial_metadata_has_conditions(self, model):
        C_curr = torch.randn(1, model.config.hidden_dim)
        C_prev = torch.randn(1, model.config.hidden_dim)
        T_C = torch.randn(1, model.config.hidden_dim)
        alpha_n = torch.tensor([0.5])

        model.meta_loop._fast_km_step(C_curr, C_prev, T_C, alpha_n, n=5)
        meta = model.meta_loop._last_inertial_km_metadata
        assert 'stability_conditions' in meta
        conds = meta['stability_conditions']
        assert conds['requires_theta_averaged_T'] is True
        assert conds['theta_averaged_verified'] is False
        assert conds['nonexpansive_sufficient'] is True

    def test_inertial_metadata_has_beta_alpha(self, model):
        C_curr = torch.randn(1, model.config.hidden_dim)
        C_prev = torch.randn(1, model.config.hidden_dim)
        T_C = torch.randn(1, model.config.hidden_dim)
        alpha_n = torch.tensor([0.5])

        model.meta_loop._fast_km_step(C_curr, C_prev, T_C, alpha_n, n=5)
        meta = model.meta_loop._last_inertial_km_metadata
        assert 'beta_n' in meta
        assert 'alpha_n' in meta
        assert meta['beta_n'] == max(0.0, (5 - 1) / (5 + 2))

    def test_inertial_metadata_has_formal_limitation(self, model):
        C_curr = torch.randn(1, model.config.hidden_dim)
        C_prev = torch.randn(1, model.config.hidden_dim)
        T_C = torch.randn(1, model.config.hidden_dim)
        alpha_n = torch.tensor([0.5])

        model.meta_loop._fast_km_step(C_curr, C_prev, T_C, alpha_n, n=5)
        meta = model.meta_loop._last_inertial_km_metadata
        assert 'formal_limitation' in meta
        assert 'θ-averaged' in meta['formal_limitation']

    def test_anderson_convergence_conditions_conditional(self, model):
        """Anderson convergence guarantee should be 'conditional'."""
        assert hasattr(model.meta_loop, '_anderson_convergence_conditions')
        conds = model.meta_loop._anderson_convergence_conditions
        assert conds['convergence_guarantee'] == 'conditional'

    def test_anderson_formal_guarantee_false(self, model):
        """Dropout training reconciliation should have formal_guarantee=False."""
        assert hasattr(model.meta_loop, '_dropout_training_reconciliation')
        recon = model.meta_loop._dropout_training_reconciliation
        assert recon['formal_guarantee'] is False


# ──────────────────────────────────────────────────────────────────────────────
# RIGOR-5: Catastrophe criticality assessment
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGOR5_CatastropheCriticality:
    """Tests for RIGOR-5: criticality_assessment in classify_catastrophe_type."""

    def _get_analyzer(self, model):
        """Get the topology analyzer from the model."""
        return model.topology_analyzer

    def test_classify_has_criticality_assessment(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        condition_number = torch.tensor([2000.0])
        grad_norm = torch.tensor([0.1])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        assert 'criticality_assessment' in result

    def test_criticality_has_criticality_level(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        condition_number = torch.tensor([2000.0])
        grad_norm = torch.tensor([0.1])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert 'criticality_level' in ca
        assert len(ca['criticality_level']) == 1

    def test_near_catastrophe_when_corank_nonzero_grad_nonzero(self, model):
        """Corank≥1 but ∇E≠0 → near_catastrophe (not at critical point)."""
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[0.001, 0.5, 1.0, 2.0]])
        condition_number = torch.tensor([2000.0])
        grad_norm = torch.tensor([0.1])  # gradient NOT vanishing
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert ca['criticality_level'][0] == 'near_catastrophe'

    def test_degenerate_critical_when_corank_and_grad_vanishing(self, model):
        """Corank≥1 AND ∇E=0 → degenerate_critical_point."""
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[0.0001, 0.5, 1.0, 2.0]])
        condition_number = torch.tensor([20000.0])
        grad_norm = torch.tensor([0.0001])  # gradient vanishing
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert ca['criticality_level'][0] == 'degenerate_critical_point'

    def test_regular_point_when_no_degeneracy(self, model):
        """Corank=0 and ∇E≠0 → regular_point."""
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert ca['criticality_level'][0] == 'regular_point'

    def test_nondegenerate_critical_when_grad_zero_corank_zero(self, model):
        """Corank=0 and ∇E=0 → nondegenerate_critical_point (local min/max)."""
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([0.0001])  # gradient vanishing
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert ca['criticality_level'][0] == 'nondegenerate_critical_point'

    def test_criticality_has_gradient_threshold(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert 'gradient_vanishing_threshold' in ca
        assert ca['gradient_vanishing_threshold'] > 0

    def test_criticality_has_formal_classification_gap(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert 'formal_classification_gap' in ca
        gap = ca['formal_classification_gap']
        assert 'hessian_diagonal_approximation' in gap
        assert 'thom_arnold_gap' in gap
        assert 'missing_for_formal_classification' in gap

    def test_classification_gap_mentions_jet_space(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        gap = result['criticality_assessment']['formal_classification_gap']
        assert 'jet' in gap['thom_arnold_gap'].lower()

    def test_classification_gap_lists_missing_steps(self, model):
        analyzer = self._get_analyzer(model)
        eigenvalues = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        condition_number = torch.tensor([4.0])
        grad_norm = torch.tensor([1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        gap = result['criticality_assessment']['formal_classification_gap']
        missing = gap['missing_for_formal_classification']
        assert len(missing) >= 4  # At least 4 missing steps

    def test_gradient_vanishing_check_list(self, model):
        analyzer = self._get_analyzer(model)
        # Batch of 2
        eigenvalues = torch.tensor([
            [0.001, 0.5, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        condition_number = torch.tensor([2000.0, 4.0])
        grad_norm = torch.tensor([0.0001, 1.0])
        result = analyzer.classify_catastrophe_type(
            eigenvalues, condition_number, grad_norm,
        )
        ca = result['criticality_assessment']
        assert len(ca['gradient_vanishing_check']) == 2
        assert ca['gradient_vanishing_check'][0] is True  # 0.0001 < threshold
        assert ca['gradient_vanishing_check'][1] is False  # 1.0 > threshold


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRIGORIntegration:
    """Cross-patch integration tests."""

    def test_full_forward_with_rigor_patches(self, model):
        """Ensure model forward pass succeeds with all RIGOR patches."""
        x = torch.randint(0, model.config.vocab_size, (1, 16))
        with torch.no_grad():
            out = model(x)
        assert out is not None

    def test_ssm_forward_with_diagonal_invariant(self, ssm_block):
        """SSM block forward should work and invariant should hold."""
        x = torch.randn(1, 8, 64)  # [B, L, D]
        with torch.no_grad():
            y, state = ssm_block(x)
        assert y.shape == (1, 8, 64)
        inv = ssm_block.verify_diagonal_A_invariant()
        assert inv['is_diagonal'] is True

    def test_meta_loop_certificate_with_rigor2(self, model):
        """Meta-loop certificate should include RIGOR-2 preconditions."""
        psi_0 = torch.randn(1, model.config.hidden_dim)
        with torch.no_grad():
            _, _, meta = model.meta_loop.compute_fixed_point(psi_0)
        # Certificate should have convergence info
        assert 'km_formal_convergence' in meta

    def test_lipschitz_budget_with_rigor3(self, model):
        """Joint Lipschitz budget should include RIGOR-3 variance analysis."""
        budget = model.meta_loop.verify_joint_lipschitz_budget()
        assert 'layernorm_variance_analysis' in budget
        assert 'factors' in budget
        assert 'L_T_composed' in budget
