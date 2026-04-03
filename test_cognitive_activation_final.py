"""
Tests for Final Integration & Cognitive Activation patches.

Covers the 6 critical theoretical gaps addressed by the cognitive activation:
1. LayerNorm Lipschitz consistency (LiPopt fix)
2. Feedback gate dg/dC Lipschitz (product rule)
3. IQC certificate adversarial refinement & formal distinction
4. Dropout training-time Lipschitz reconciliation
5. Anderson acceleration formal convergence conditions
6. VQ EMA normalization derivation & stability analysis

Each test class validates structural correctness, numerical consistency,
and theoretical documentation of the fixes.
"""

import math
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeon_core import (
    AEONConfig,
    LipschitzConstrainedLambda,
    ProvablyConvergentMetaLoop,
    RobustVectorQuantizer,
)

# ── Module-level constants ──────────────────────────────────────────────────
STRICT_TOL = 1e-5
MODERATE_TOL = 1e-3


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Standard AEONConfig for testing."""
    return AEONConfig(device_str='cpu')


@pytest.fixture
def meta_loop(config):
    """ProvablyConvergentMetaLoop in eval mode."""
    ml = ProvablyConvergentMetaLoop(config)
    ml.eval()
    return ml, config


@pytest.fixture
def lambda_op(config):
    """LipschitzConstrainedLambda in eval mode."""
    lam = LipschitzConstrainedLambda(
        input_dim=config.hidden_dim * 2,
        hidden_dim=config.meta_dim,
        output_dim=config.hidden_dim,
        lipschitz_target=config.lipschitz_target,
        use_spectral_norm=True,
        dropout=config.dropout_rate,
    )
    lam.eval()
    return lam, config


@pytest.fixture
def vq():
    """RobustVectorQuantizer for VQ EMA tests."""
    return RobustVectorQuantizer(
        num_embeddings=64,
        embedding_dim=32,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1: LayerNorm Lipschitz Consistency (LiPopt)
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerNormLipschitzConsistency:
    """Validate that the LiPopt LayerNorm bound is now consistent
    with the sandwich parameterization (√d · γ_ratio)."""

    def test_lipopt_layernorm_factor_uses_sqrt_d(self, lambda_op):
        """LiPopt L_LN must include √d factor, not just max|γ|."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        _ln_gamma = lam.layer_norm.weight.detach()
        d = float(_ln_gamma.numel())
        gamma_max = _ln_gamma.abs().max().item()
        gamma_min = max(_ln_gamma.abs().min().item(), 1e-8)
        expected_L_LN = math.sqrt(d) * gamma_max / gamma_min
        assert abs(result['layernorm_factor'] - expected_L_LN) < STRICT_TOL

    def test_lipopt_has_layernorm_details(self, lambda_op):
        """LiPopt return must include layernorm_details with derivation."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        assert 'layernorm_details' in result
        details = result['layernorm_details']
        required_fields = [
            'dimension', 'gamma_max', 'gamma_min', 'gamma_ratio',
            'structural_bound', 'epsilon', 'epsilon_aware_worst_case',
            'derivation',
        ]
        for field in required_fields:
            assert field in details, f"Missing layernorm_details.{field}"

    def test_lipopt_layernorm_gamma_ratio_positive(self, lambda_op):
        """gamma_ratio must be ≥ 1.0."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        assert result['layernorm_details']['gamma_ratio'] >= 1.0

    def test_lipopt_epsilon_aware_bound_finite(self, lambda_op):
        """The epsilon-aware worst-case bound must be finite and larger."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        details = result['layernorm_details']
        assert math.isfinite(details['epsilon_aware_worst_case'])
        # Epsilon-aware bound >= structural bound (since epsilon is small)
        assert details['epsilon_aware_worst_case'] >= details['structural_bound']

    def test_lipopt_methodology_mentions_layernorm_consistency(self, lambda_op):
        """Methodology must mention LayerNorm and Behrmann reference."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        meth = result['methodology'].lower()
        assert 'layernorm' in meth
        assert 'behrmann' in meth

    def test_lipopt_bound_multiplies_L_LN_correctly(self, lambda_op):
        """L_lipopt must equal √λ_max(H_gram) · L_LN."""
        lam, config = lambda_op
        result = lam.compute_lipopt_certificate()
        L_sqrt_eig = math.sqrt(max(result['gram_eigenvalue'], 0.0))
        L_LN = result['layernorm_factor']
        expected = L_sqrt_eig * L_LN
        assert abs(result['L_lipopt'] - expected) < STRICT_TOL


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2: Feedback Gate dg/dC Lipschitz
# ═══════════════════════════════════════════════════════════════════════════

class TestFeedbackGateLipschitz:
    """Validate the product-rule Lipschitz analysis of the feedback gate."""

    def test_verify_uniform_contraction_has_gate_analysis(self, lambda_op):
        """Return dict must include feedback_gate_analysis."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        assert 'feedback_gate_analysis' in result

    def test_gate_analysis_has_required_fields(self, lambda_op):
        """feedback_gate_analysis must have dg_dC_lipschitz and product_rule_bound."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        ga = result['feedback_gate_analysis']
        required = [
            'gate_type', 'attenuation_factor', 'dg_dC_lipschitz',
            'W_gate_C_spectral_norm', 'amplification_risk',
            'product_rule_bound', 'analysis_note',
        ]
        for field in required:
            assert field in ga, f"Missing feedback_gate_analysis.{field}"

    def test_dg_dC_lipschitz_is_025_times_spectral_norm(self, lambda_op):
        """dg/dC Lipschitz = 0.25 · ‖W_gate_C‖₂."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        ga = result['feedback_gate_analysis']
        expected = 0.25 * ga['W_gate_C_spectral_norm']
        assert abs(ga['dg_dC_lipschitz'] - expected) < STRICT_TOL

    def test_per_component_includes_dg_dC(self, lambda_op):
        """per_component dict must include feedback_gate_dg_dC."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        assert 'feedback_gate_dg_dC' in result['per_component']

    def test_gate_analysis_note_mentions_product_rule(self, lambda_op):
        """Analysis note must explain the product-rule decomposition."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        note = result['feedback_gate_analysis']['analysis_note']
        note_normalized = note.lower().replace('\u2202', 'd')
        assert 'product' in note_normalized or 'dg/dc' in note_normalized

    def test_no_gate_means_zero_dg_dC(self, lambda_op):
        """When feedback_gate is None, dg/dC should be 0."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=None, alpha=0.5,
        )
        ga = result['feedback_gate_analysis']
        assert ga['dg_dC_lipschitz'] == 0.0
        assert ga['amplification_risk'] is False

    def test_amplification_risk_flag(self, lambda_op):
        """amplification_risk should be True when dg/dC > 0.1."""
        lam, config = lambda_op
        psi = torch.randn(2, config.z_dim)
        # Create gate with large weights to trigger amplification risk
        gate = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid())
        # Scale up weights to make spectral norm large
        with torch.no_grad():
            for m in gate.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(10.0)
        result = lam.verify_uniform_contraction(
            psi, num_samples=8, feedback_gate=gate, alpha=0.5,
        )
        ga = result['feedback_gate_analysis']
        # With 10× weight scaling, dg/dC should be > 0.1
        assert ga['dg_dC_lipschitz'] > 0.1
        assert ga['amplification_risk'] is True


# ═══════════════════════════════════════════════════════════════════════════
# Fix 3: IQC Certificate Adversarial Refinement & Formal Distinction
# ═══════════════════════════════════════════════════════════════════════════

class TestIQCCertificateAdversarial:
    """Validate adversarial refinement and IQC methodology distinction."""

    def test_certificate_has_iqc_distinction(self, meta_loop):
        """Certificate must include iqc_distinction dict."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'iqc_distinction' in cert

    def test_iqc_distinction_marks_not_standard(self, meta_loop):
        """IQC distinction must clearly state this is NOT standard IQC."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        iqc = cert['iqc_distinction']
        assert iqc['NOT_standard_iqc'] is True
        assert iqc['method_type'] == 'sample_based_lipschitz_surrogate'

    def test_iqc_distinction_has_required_fields(self, meta_loop):
        """IQC distinction must document what we actually do vs standard IQC."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        iqc = cert['iqc_distinction']
        required = [
            'method_type', 'NOT_standard_iqc', 'standard_iqc_requirements',
            'what_we_actually_do', 'gelu_slope_restriction_valid',
            'gelu_sector_bound', 'contraction_claim_strength',
        ]
        for field in required:
            assert field in iqc, f"Missing iqc_distinction.{field}"

    def test_contraction_claim_strength_is_local(self, meta_loop):
        """Contraction claim must be 'local_empirical', not 'global'."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['iqc_distinction']['contraction_claim_strength'] == 'local_empirical'

    def test_adversarial_refinement_in_coverage(self, meta_loop):
        """Sample coverage must report adversarial refinement steps."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert 'adversarial_refinement_steps' in sc
        assert sc['adversarial_refinement_steps'] >= 0

    def test_adversarial_worst_sigma_reported(self, meta_loop):
        """Sample coverage must include adversarial_worst_sigma."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert 'adversarial_worst_sigma' in sc
        assert sc['adversarial_worst_sigma'] >= 0.0

    def test_methodology_mentions_not_standard_iqc(self, meta_loop):
        """Methodology string must mention this is NOT standard IQC."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        meth = cert['methodology'].lower()
        assert 'not a standard iqc' in meth or 'not standard iqc' in meth or (
            'megretski' in meth and 'not' in meth
        )

    def test_gelu_sector_bound_valid(self, meta_loop):
        """GELU sector bound [0, 1.13] must be valid in IQC distinction."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        iqc = cert['iqc_distinction']
        assert iqc['gelu_slope_restriction_valid'] is True
        assert iqc['gelu_sector_bound'] == [0.0, 1.13]

    def test_coverage_type_with_adversarial(self, meta_loop):
        """Coverage type must reflect adversarial refinement."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'adversarial' in cert['sample_coverage']['coverage_type']


# ═══════════════════════════════════════════════════════════════════════════
# Fix 4: Dropout Training-Time Lipschitz Reconciliation
# ═══════════════════════════════════════════════════════════════════════════

class TestDropoutTrainingReconciliation:
    """Validate the dropout training-time contraction reconciliation."""

    def test_has_dropout_training_reconciliation(self, meta_loop):
        """Meta-loop must have _dropout_training_reconciliation dict."""
        ml, config = meta_loop
        assert hasattr(ml, '_dropout_training_reconciliation')
        rec = ml._dropout_training_reconciliation
        assert isinstance(rec, dict)

    def test_reconciliation_has_required_fields(self, meta_loop):
        """Reconciliation dict must document dropout rate, inflation, and mitigation."""
        ml, config = meta_loop
        rec = ml._dropout_training_reconciliation
        required = [
            'dropout_rate', 'lipschitz_inflation_factor',
            'contraction_violated', 'mitigation_strategy',
            'formal_guarantee', 'note',
        ]
        for field in required:
            assert field in rec, f"Missing _dropout_training_reconciliation.{field}"

    def test_formal_guarantee_is_false(self, meta_loop):
        """formal_guarantee must be False (honest limitation)."""
        ml, config = meta_loop
        assert ml._dropout_training_reconciliation['formal_guarantee'] is False

    def test_inflation_factor_matches_dropout_rate(self, meta_loop):
        """Inflation factor must equal 1/(1-p)."""
        ml, config = meta_loop
        rec = ml._dropout_training_reconciliation
        p = rec['dropout_rate']
        if p < 1.0:
            expected = 1.0 / (1.0 - p)
            assert abs(rec['lipschitz_inflation_factor'] - expected) < STRICT_TOL

    def test_contraction_violated_when_dropout_nonzero(self, meta_loop):
        """contraction_violated must be True when dropout_rate > 0."""
        ml, config = meta_loop
        rec = ml._dropout_training_reconciliation
        if rec['dropout_rate'] > 0:
            assert rec['contraction_violated'] is True
        else:
            assert rec['contraction_violated'] is False

    def test_has_dropout_training_lipschitz_inflation(self, meta_loop):
        """Meta-loop must expose _dropout_training_lipschitz_inflation."""
        ml, config = meta_loop
        assert hasattr(ml, '_dropout_training_lipschitz_inflation')
        assert ml._dropout_training_lipschitz_inflation >= 1.0

    def test_mitigation_strategy_covers_three_approaches(self, meta_loop):
        """Mitigation strategy must mention eval, KM, and spectral."""
        ml, config = meta_loop
        strat = ml._dropout_training_reconciliation['mitigation_strategy']
        assert 'eval' in strat.lower() or 'certification' in strat.lower()
        assert 'km' in strat.lower() or 'averaging' in strat.lower()
        assert 'spectral' in strat.lower() or 'penalty' in strat.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Fix 5: Anderson Acceleration Convergence Conditions
# ═══════════════════════════════════════════════════════════════════════════

class TestAndersonConvergenceConditions:
    """Validate formal convergence conditions documentation for Anderson."""

    def test_has_convergence_conditions(self, meta_loop):
        """Meta-loop must have _anderson_convergence_conditions."""
        ml, config = meta_loop
        assert hasattr(ml, '_anderson_convergence_conditions')
        conds = ml._anderson_convergence_conditions
        assert isinstance(conds, dict)

    def test_convergence_conditions_required_fields(self, meta_loop):
        """Conditions dict must have all required fields."""
        ml, config = meta_loop
        conds = ml._anderson_convergence_conditions
        required = [
            'requires_nonexpansive_T', 'type_I_safeguard',
            'summable_perturbation_budget', 'required_decay_rate',
            'adaptive_memory_depth', 'convergence_guarantee',
            'divergence_risk', 'theorem_reference',
        ]
        for field in required:
            assert field in conds, f"Missing convergence condition: {field}"

    def test_convergence_guarantee_is_conditional(self, meta_loop):
        """Convergence guarantee must be 'conditional', not 'unconditional'."""
        ml, config = meta_loop
        assert ml._anderson_convergence_conditions['convergence_guarantee'] == 'conditional'

    def test_requires_nonexpansive_T(self, meta_loop):
        """Must document that nonexpansive T is required."""
        ml, config = meta_loop
        assert ml._anderson_convergence_conditions['requires_nonexpansive_T'] is True

    def test_type_I_safeguard_enabled(self, meta_loop):
        """Type-I safeguard (Zhang et al., 2020) must be enabled."""
        ml, config = meta_loop
        assert ml._anderson_convergence_conditions['type_I_safeguard'] is True

    def test_summable_budget_matches_pi_squared(self, meta_loop):
        """Perturbation budget must be B₀ · π²/6."""
        ml, config = meta_loop
        expected_budget = 1.0 * (math.pi ** 2 / 6.0)
        actual_budget = ml._anderson_convergence_conditions['summable_perturbation_budget']
        assert abs(actual_budget - expected_budget) < STRICT_TOL

    def test_divergence_risk_mentions_key_risks(self, meta_loop):
        """Divergence risk must mention nonexpansiveness failure and summability."""
        ml, config = meta_loop
        risk = ml._anderson_convergence_conditions['divergence_risk'].lower()
        assert 'nonexpansive' in risk or 'diverge' in risk
        assert 'summable' in risk or 'budget' in risk

    def test_theorem_reference_cites_zhang(self, meta_loop):
        """Theorem reference must cite Zhang et al. (2020)."""
        ml, config = meta_loop
        ref = ml._anderson_convergence_conditions['theorem_reference']
        assert 'zhang' in ref.lower() or 'siam' in ref.lower()

    def test_decay_rate_specified(self, meta_loop):
        """Required decay rate must specify asymptotic bound O(1/n^{1+δ})."""
        ml, config = meta_loop
        rate = ml._anderson_convergence_conditions['required_decay_rate']
        assert 'O(1/n' in rate or '1/n^{1+' in rate


# ═══════════════════════════════════════════════════════════════════════════
# Fix 6: VQ EMA Normalization & Stability Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestVQEMAStabilityAnalysis:
    """Validate VQ EMA stability analysis method and docstring."""

    def test_ema_update_docstring_has_equations(self, vq):
        """_ema_update docstring must reference Eqs. 31-34."""
        doc = vq._ema_update.__doc__
        assert doc is not None
        doc_lower = doc.lower()
        assert 'eq. 31' in doc_lower or 'eq 31' in doc_lower
        assert 'eq. 32' in doc_lower or 'eq 32' in doc_lower
        assert 'eq. 33' in doc_lower or 'eq 33' in doc_lower
        assert 'eq. 34' in doc_lower or 'eq 34' in doc_lower

    def test_ema_update_docstring_mentions_laplace(self, vq):
        """Docstring must explain the Laplace smoothing derivation."""
        doc = vq._ema_update.__doc__
        assert 'laplace' in doc.lower() or 'dirichlet' in doc.lower()

    def test_ema_update_docstring_has_bias_analysis(self, vq):
        """Docstring must include bias properties section."""
        doc = vq._ema_update.__doc__
        doc_lower = doc.lower()
        assert 'bias' in doc_lower

    def test_has_get_ema_stability_analysis(self, vq):
        """VQ must have get_ema_stability_analysis method."""
        assert hasattr(vq, 'get_ema_stability_analysis')
        result = vq.get_ema_stability_analysis()
        assert isinstance(result, dict)

    def test_stability_analysis_has_required_fields(self, vq):
        """Stability analysis must return all required fields."""
        result = vq.get_ema_stability_analysis()
        required = [
            'decay', 'epsilon', 'effective_window_steps',
            'total_ema_count', 'laplace_smoothing_bias',
            'count_conservation', 'dead_code_analysis',
            'stability_verdict',
        ]
        for field in required:
            assert field in result, f"Missing stability analysis field: {field}"

    def test_effective_window_matches_decay(self, vq):
        """Effective window must equal 1/(1-decay)."""
        result = vq.get_ema_stability_analysis()
        expected = 1.0 / (1.0 - vq.decay)
        assert abs(result['effective_window_steps'] - expected) < STRICT_TOL

    def test_count_conservation_holds_after_update(self, vq):
        """After an EMA update, count conservation must hold."""
        vq.train()
        inputs = torch.randn(16, 32)
        vq(inputs)  # trigger EMA update
        result = vq.get_ema_stability_analysis()
        assert result['count_conservation']['verified'] is True

    def test_laplace_bias_has_derivation(self, vq):
        """Laplace smoothing bias dict must include derivation."""
        result = vq.get_ema_stability_analysis()
        bias = result['laplace_smoothing_bias']
        assert 'derivation' in bias
        assert len(bias['derivation']) > 50

    def test_dead_code_analysis_sane_at_init(self, vq):
        """At initialization, dead code count should be ≤ total codes."""
        result = vq.get_ema_stability_analysis()
        dca = result['dead_code_analysis']
        assert dca['dead_codes'] + dca['alive_codes'] == vq.num_embeddings
        assert 0.0 <= dca['dead_fraction'] <= 1.0

    def test_stability_verdict_at_init(self, vq):
        """At initialization (no updates), stability verdict should be reasonable."""
        result = vq.get_ema_stability_analysis()
        verdict = result['stability_verdict']
        assert 'numerically_stable' in verdict
        assert 'bias_negligible' in verdict
        assert 'conservation_holds' in verdict

    def test_laplace_smoothing_preserves_sum(self, vq):
        """Numerically verify Σ Ñ_k ≈ Σ N_k after updates."""
        vq.train()
        for _ in range(5):
            inputs = torch.randn(32, 32)
            vq(inputs)
        result = vq.get_ema_stability_analysis()
        assert result['count_conservation']['error'] < 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCognitiveActivationIntegration:
    """End-to-end integration tests for the complete cognitive activation."""

    def test_meta_loop_full_forward_with_all_fixes(self, meta_loop):
        """Meta-loop forward pass must work with all fixes applied."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, iters, info = ml(psi)
        assert C.shape == (2, config.hidden_dim)
        assert torch.isfinite(C).all()

    def test_composite_cert_with_all_enhancements(self, meta_loop):
        """Composite certificate must include all new fields."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        # Check all new fields exist
        assert 'iqc_distinction' in cert
        assert 'sample_coverage' in cert
        # Check adversarial refinement
        assert 'adversarial_refinement_steps' in cert['sample_coverage']

    def test_uniform_contraction_with_gate_and_dropout(self, meta_loop):
        """Uniform contraction must work with gate analysis and dropout info."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        result = ml.lambda_op.verify_uniform_contraction(
            psi, num_samples=8,
            feedback_gate=ml.feedback_gate,
            alpha=0.5,
        )
        # Gate analysis must be present
        assert 'feedback_gate_analysis' in result
        # Per-component must include dg/dC
        assert 'feedback_gate_dg_dC' in result['per_component']

    def test_all_theoretical_patches_coexist(self, meta_loop):
        """All six patches must coexist without conflict."""
        ml, config = meta_loop
        # Fix 4: dropout reconciliation
        assert hasattr(ml, '_dropout_training_reconciliation')
        # Fix 5: Anderson conditions
        assert hasattr(ml, '_anderson_convergence_conditions')
        # Fix 1+2+3: verified via certificate computation
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert is not None

    def test_vq_stability_with_codebook_stats(self, vq):
        """VQ stability analysis must be consistent with usage stats."""
        vq.train()
        inputs = torch.randn(32, 32)
        vq(inputs)
        stability = vq.get_ema_stability_analysis()
        usage = vq.get_codebook_usage_stats()
        # Dead code count should be consistent
        assert stability['dead_code_analysis']['dead_codes'] <= usage['total_codes']
