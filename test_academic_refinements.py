"""Tests for academic-standard refinements addressing five theoretical concerns.

Refinement 1: KM iteration — dropout exclusion, averaged-map, monotone operator refs.
Refinement 2: Banach contraction — constructive a-posteriori bound validity.
Refinement 3: IQC certificate — coverage quantification and confidence qualification.
Refinement 4: Catastrophe analysis — unfolding analysis framework with normal forms.
Refinement 5: Causal factorization — causal assumptions, identifiability assessment.

All tests validate that mathematical claims are appropriately scoped, that
documentation accurately reflects formal guarantees, and that new analytical
structures are complete and internally consistent.
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
    ProvablyConvergentMetaLoop,
    OptimizedTopologyAnalyzer,
    CausalFactorExtractor,
    NeuralCausalModel,
    NOTEARSCausalModel,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )


@pytest.fixture
def meta_loop(config):
    ml = ProvablyConvergentMetaLoop(config, max_iterations=10, min_iterations=2)
    ml.eval()
    return ml, config


@pytest.fixture
def topology_analyzer():
    cfg = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=True,
        enable_safety_guardrails=False,
    )
    return OptimizedTopologyAnalyzer(cfg)


@pytest.fixture
def causal_extractor():
    return CausalFactorExtractor(hidden_dim=256, num_factors=8)


@pytest.fixture
def neural_causal():
    return NeuralCausalModel(num_vars=6, hidden_dim=32)


@pytest.fixture
def notears_causal():
    return NOTEARSCausalModel(num_vars=6, hidden_dim=32)


# ============================================================================
# Refinement 1: KM Iteration — Dropout Exclusion & Monotone Operator
# ============================================================================

class TestKMDropoutExclusion:
    """Validate dropout exclusion and monotone operator documentation."""

    def test_dropout_excluded_flag_exists(self, meta_loop):
        """FixedPointOperator must have _dropout_excluded_during_certification."""
        ml, _ = meta_loop
        assert hasattr(ml, '_dropout_excluded_during_certification')
        assert ml._dropout_excluded_during_certification is True

    def test_monotone_operator_note_exists(self, meta_loop):
        """FixedPointOperator must reference monotone operator alternatives."""
        ml, _ = meta_loop
        assert hasattr(ml, '_monotone_operator_alternative_note')
        note = ml._monotone_operator_alternative_note
        assert isinstance(note, str)
        assert len(note) > 50

    def test_monotone_operator_note_mentions_lben(self, meta_loop):
        """Monotone operator note must reference LBEN (Revay et al.)."""
        ml, _ = meta_loop
        note = ml._monotone_operator_alternative_note
        assert 'LBEN' in note or 'Revay' in note

    def test_monotone_operator_note_mentions_nemon(self, meta_loop):
        """Monotone operator note must reference NEMON (Winston & Kolter)."""
        ml, _ = meta_loop
        note = ml._monotone_operator_alternative_note
        assert 'NEMON' in note or 'Winston' in note or 'Kolter' in note

    def test_monotone_operator_note_mentions_non_euclidean(self, meta_loop):
        """Note must mention non-Euclidean metrics for constructive guarantees."""
        ml, _ = meta_loop
        note = ml._monotone_operator_alternative_note
        assert 'non-Euclidean' in note or 'weighted' in note

    def test_km_formal_convergence_has_certification_caveat(self, meta_loop):
        """Banach convergence must include certification caveat about dropout."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            caveat = km.get('certification_caveat', '')
            assert 'dropout' in caveat.lower()

    def test_km_convergence_references_layernorm_lipschitz(self, meta_loop):
        """KM convergence caveat must mention LayerNorm Lipschitz data-dependence."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') in ('banach_contraction', 'krasnoselskii_mann'):
            all_text = str(km)
            assert 'LayerNorm' in all_text or 'LN' in all_text


# ============================================================================
# Refinement 2: Banach Contraction — A-Posteriori Bound Validity
# ============================================================================

class TestBanachAPosterioriValidity:
    """Validate constructive a-posteriori error bound verification."""

    def test_a_posteriori_validity_present_when_contraction(self, meta_loop):
        """Banach certificate must include a_posteriori_validity dict."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            assert 'a_posteriori_validity' in km
            apv = km['a_posteriori_validity']
            assert isinstance(apv, dict)

    def test_a_posteriori_has_L_C(self, meta_loop):
        """a_posteriori_validity must contain L_C."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            assert 'L_C' in apv
            assert isinstance(apv['L_C'], float)
            assert apv['L_C'] < 1.0

    def test_a_posteriori_has_one_minus_L_C(self, meta_loop):
        """a_posteriori_validity must track 1 − L_C (contraction margin)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            assert 'one_minus_L_C' in apv
            assert apv['one_minus_L_C'] > 0
            assert abs(apv['one_minus_L_C'] - (1.0 - apv['L_C'])) < 1e-10

    def test_a_posteriori_denominator_positive(self, meta_loop):
        """Denominator 1 − L_C must be constructively positive (> 1e-10)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            assert 'denominator_positive' in apv
            assert isinstance(apv['denominator_positive'], bool)

    def test_a_posteriori_bound_status(self, meta_loop):
        """bound_status must be 'valid' or 'degenerate'."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            assert 'bound_status' in apv
            assert apv['bound_status'] in ('valid', 'degenerate — L_C too close to 1')

    def test_a_posteriori_constructive_note_mentions_equation_7(self, meta_loop):
        """Constructive note must reference Equation 7 context."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            note = apv.get('constructive_note', '')
            assert 'Equation 7' in note or 'a-posteriori' in note

    def test_a_posteriori_mentions_gelu_lipschitz(self, meta_loop):
        """Constructive note must mention L_GELU ≈ 1.13."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            note = apv.get('constructive_note', '')
            assert '1.13' in note

    def test_error_amplification_factor_finite(self, meta_loop):
        """Error amplification factor L_C/(1−L_C) must be finite when valid."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            if apv['denominator_positive']:
                assert math.isfinite(apv['error_amplification_factor'])
                assert apv['error_amplification_factor'] >= 0

    def test_contraction_margin_positive(self, meta_loop):
        """Contraction margin must be positive for valid Banach certificate."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        km = meta.get('km_formal_convergence', {})
        if km.get('convergence_type') == 'banach_contraction':
            apv = km['a_posteriori_validity']
            assert 'contraction_margin' in apv
            assert apv['contraction_margin'] > 0


# ============================================================================
# Refinement 3: IQC Certificate — Coverage Quantification
# ============================================================================

class TestIQCCoverageQuantification:
    """Validate IQC certificate coverage metrics and confidence qualification."""

    def test_sample_coverage_present(self, meta_loop):
        """Composite T certificate must include sample_coverage dict."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'sample_coverage' in cert
        sc = cert['sample_coverage']
        assert isinstance(sc, dict)

    def test_sample_coverage_num_samples(self, meta_loop):
        """Coverage must report the number of samples used."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=8)
        sc = cert['sample_coverage']
        assert sc['num_samples'] == 8

    def test_sample_coverage_type_is_local(self, meta_loop):
        """Coverage type must be 'local_empirical', not 'global'."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert sc['coverage_type'] == 'local_empirical'
        assert 'global' not in sc['coverage_type']

    def test_sample_coverage_note_not_iff(self, meta_loop):
        """Coverage note must clarify the test is NOT an 'iff' condition."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        note = sc.get('coverage_note', '')
        assert 'iff' in note.lower() or 'lower bound' in note.lower()

    def test_sample_coverage_has_confidence_qualification(self, meta_loop):
        """Coverage must include confidence qualification disclaiming stats."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert 'confidence_qualification' in sc
        cq = sc['confidence_qualification']
        assert 'statistical' in cq.lower() or 'confidence interval' in cq.lower()

    def test_sample_coverage_has_recommendations(self, meta_loop):
        """Coverage must include improvement recommendations."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert 'improvement_recommendations' in sc
        recs = sc['improvement_recommendations']
        assert isinstance(recs, list)
        assert len(recs) >= 3

    def test_sample_coverage_recommendations_mention_adversarial(self, meta_loop):
        """Recommendations must mention adversarial input search."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        recs = cert['sample_coverage']['improvement_recommendations']
        recs_text = ' '.join(recs).lower()
        assert 'adversarial' in recs_text

    def test_sample_coverage_recommendations_mention_monotone(self, meta_loop):
        """Recommendations must mention monotone operator frameworks."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        recs = cert['sample_coverage']['improvement_recommendations']
        recs_text = ' '.join(recs).lower()
        assert 'monotone' in recs_text or 'lben' in recs_text or 'nemon' in recs_text

    def test_sample_coverage_sampling_distribution(self, meta_loop):
        """Coverage must document the sampling distribution."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        sc = cert['sample_coverage']
        assert 'sampling_distribution' in sc
        assert 'N(0' in sc['sampling_distribution'] or 'normal' in sc['sampling_distribution'].lower()


# ============================================================================
# Refinement 4: Catastrophe Analysis — Unfolding Framework
# ============================================================================

class TestCatastropheUnfoldingAnalysis:
    """Validate formal unfolding analysis framework in catastrophe classification."""

    def _get_classification(self, topology_analyzer, corank_target=2):
        """Helper to get a catastrophe classification result."""
        P = 8
        eigs = torch.ones(2, P) * 0.5
        # Make some eigenvalues near zero to trigger the target corank
        for i in range(min(corank_target, P)):
            eigs[:, i] = 1e-5
        kappa = torch.tensor([100.0, 100.0])
        grad = torch.tensor([0.01, 0.01])
        return topology_analyzer.classify_catastrophe_type(
            eigs, kappa, grad
        )

    def test_unfolding_analysis_present(self, topology_analyzer):
        """Classification must include unfolding_analysis dict."""
        result = self._get_classification(topology_analyzer)
        assert 'unfolding_analysis' in result
        ua = result['unfolding_analysis']
        assert isinstance(ua, dict)

    def test_unfolding_not_computed(self, topology_analyzer):
        """Versal unfolding must be marked as NOT computed."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert ua['versal_unfolding_computed'] is False

    def test_normal_form_not_verified(self, topology_analyzer):
        """Normal form must be marked as NOT verified."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert ua['normal_form_verified'] is False

    def test_control_parameters_not_identified(self, topology_analyzer):
        """Control parameters must be marked as NOT identified."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert ua['control_parameters_identified'] is False

    def test_codimension_not_verified(self, topology_analyzer):
        """Codimension must be marked as NOT verified."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert ua['codimension_verified'] is False

    def test_unfolding_status_mentions_not_implemented(self, topology_analyzer):
        """Status must state unfolding is not implemented."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert 'NOT IMPLEMENTED' in ua['status']

    def test_formal_requirements_has_5_steps(self, topology_analyzer):
        """Formal classification requirements must specify 5 steps."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        reqs = ua['requirements_for_formal_classification']
        assert isinstance(reqs, dict)
        assert len(reqs) == 5

    def test_step1_kernel_extraction(self, topology_analyzer):
        """Step 1 must describe kernel extraction."""
        result = self._get_classification(topology_analyzer)
        reqs = result['unfolding_analysis']['requirements_for_formal_classification']
        assert 'step_1_kernel_extraction' in reqs
        assert 'kernel' in reqs['step_1_kernel_extraction'].lower()

    def test_step2_jet_projection(self, topology_analyzer):
        """Step 2 must describe jet projection."""
        result = self._get_classification(topology_analyzer)
        reqs = result['unfolding_analysis']['requirements_for_formal_classification']
        assert 'step_2_jet_projection' in reqs
        text = reqs['step_2_jet_projection']
        assert 'jet' in text.lower()
        # Must reference specific jet orders for each ADE type
        assert '3-jet' in text or 'order' in text.lower()

    def test_step3_normal_form_verification(self, topology_analyzer):
        """Step 3 must describe normal form verification with canonical forms."""
        result = self._get_classification(topology_analyzer)
        reqs = result['unfolding_analysis']['requirements_for_formal_classification']
        assert 'step_3_normal_form_verification' in reqs
        text = reqs['step_3_normal_form_verification']
        # Must include canonical normal forms
        assert 'A₂' in text or 'A_2' in text or 'fold' in text.lower()
        assert 'A₃' in text or 'A_3' in text or 'cusp' in text.lower()

    def test_step4_nondegeneracy_conditions(self, topology_analyzer):
        """Step 4 must describe non-degeneracy conditions."""
        result = self._get_classification(topology_analyzer)
        reqs = result['unfolding_analysis']['requirements_for_formal_classification']
        assert 'step_4_nondegeneracy_conditions' in reqs
        text = reqs['step_4_nondegeneracy_conditions']
        assert 'non-degeneracy' in text.lower() or 'nondegeneracy' in text.lower()
        # Must warn about mislabeling
        assert 'mislabel' in text.lower() or 'ill-conditioning' in text.lower()

    def test_step5_versal_unfolding(self, topology_analyzer):
        """Step 5 must describe versal unfolding construction."""
        result = self._get_classification(topology_analyzer)
        reqs = result['unfolding_analysis']['requirements_for_formal_classification']
        assert 'step_5_versal_unfolding' in reqs
        text = reqs['step_5_versal_unfolding']
        assert 'versal' in text.lower() or 'universal' in text.lower()
        # Must include codimension values for ADE types
        assert 'A₂' in text or 'A_2' in text
        assert 'codimension' in text.lower()

    def test_alternative_interpretation_present(self, topology_analyzer):
        """Must offer alternative interpretation as spectral ill-conditioning alerts."""
        result = self._get_classification(topology_analyzer)
        ua = result['unfolding_analysis']
        assert 'alternative_interpretation' in ua
        alt = ua['alternative_interpretation']
        assert 'anomaly detection' in alt.lower() or 'ill-conditioning' in alt.lower()

    def test_alternative_mentions_severity_levels(self, topology_analyzer):
        """Alternative interpretation must mention severity levels."""
        result = self._get_classification(topology_analyzer)
        alt = result['unfolding_analysis']['alternative_interpretation']
        assert 'severity' in alt.lower() or 'corank=1' in alt

    def test_unfolding_with_no_degeneracy(self, topology_analyzer):
        """Unfolding analysis must still be present when corank=0."""
        P = 8
        eigs = torch.ones(2, P) * 0.5  # no near-zero eigenvalues
        kappa = torch.tensor([2.0, 2.0])
        grad = torch.tensor([0.5, 0.5])
        result = topology_analyzer.classify_catastrophe_type(
            eigs, kappa, grad
        )
        assert 'unfolding_analysis' in result
        assert result['unfolding_analysis']['versal_unfolding_computed'] is False


# ============================================================================
# Refinement 5: Causal Factorization — Assumptions & Identifiability
# ============================================================================

class TestCausalAssumptions:
    """Validate causal assumption documentation in CausalFactorExtractor."""

    def test_causal_assumptions_in_output(self, causal_extractor):
        """CausalFactorExtractor output must include causal_assumptions dict."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        assert 'causal_assumptions' in result
        ca = result['causal_assumptions']
        assert isinstance(ca, dict)

    def test_causal_sufficiency_documented(self, causal_extractor):
        """Causal sufficiency assumption must be documented."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'causal_sufficiency' in ca
        text = ca['causal_sufficiency']
        assert 'confound' in text.lower() or 'hidden' in text.lower()

    def test_markov_property_documented(self, causal_extractor):
        """Markov property must be documented."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'markov_property' in ca
        text = ca['markov_property']
        assert 'independent' in text.lower() or 'Markov' in text

    def test_faithfulness_documented(self, causal_extractor):
        """Faithfulness assumption must be documented."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'faithfulness' in ca
        text = ca['faithfulness']
        assert 'd-separation' in text or 'separation' in text.lower()

    def test_identifiability_documented(self, causal_extractor):
        """Identifiability conditions must be documented."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'identifiability' in ca
        text = ca['identifiability']
        assert 'Markov equivalence' in text or 'identif' in text.lower()

    def test_d_separation_flag(self, causal_extractor):
        """d_separation_verified flag must be False (not implemented)."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'd_separation_verified' in ca
        assert ca['d_separation_verified'] is False

    def test_do_calculus_note(self, causal_extractor):
        """do-calculus applicability note must be present."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        assert 'do_calculus_applicable' in ca
        text = ca['do_calculus_applicable']
        assert 'Pearl' in text
        # Must clarify this is NOT a causal model in Pearl's sense
        assert 'not a causal model' in text.lower() or 'NOT a causal model' in text

    def test_causal_assumptions_note_not_verified(self, causal_extractor):
        """Assumptions must be marked as ASSUMED, not verified."""
        x = torch.randn(2, 256)
        result = causal_extractor(x)
        ca = result['causal_assumptions']
        # At least sufficiency and faithfulness should say 'ASSUMED'
        assert 'ASSUMED' in ca['causal_sufficiency']
        assert 'ASSUMED' in ca['faithfulness']


class TestNeuralCausalModelDocstring:
    """Validate enhanced NeuralCausalModel docstring with causal assumptions."""

    def test_docstring_mentions_causal_sufficiency(self, neural_causal):
        """Docstring must mention causal sufficiency."""
        doc = NeuralCausalModel.__doc__
        assert 'Causal Sufficiency' in doc or 'causal sufficiency' in doc.lower()

    def test_docstring_mentions_markov(self, neural_causal):
        """Docstring must mention Causal Markov Property."""
        doc = NeuralCausalModel.__doc__
        assert 'Markov' in doc

    def test_docstring_mentions_faithfulness(self, neural_causal):
        """Docstring must mention faithfulness."""
        doc = NeuralCausalModel.__doc__
        assert 'Faithfulness' in doc or 'faithfulness' in doc.lower()

    def test_docstring_mentions_exogeneity(self, neural_causal):
        """Docstring must mention exogeneity of noise."""
        doc = NeuralCausalModel.__doc__
        assert 'Exogeneity' in doc or 'exogenous' in doc.lower()

    def test_docstring_mentions_identifiability(self, neural_causal):
        """Docstring must mention identifiability."""
        doc = NeuralCausalModel.__doc__
        assert 'Identifiability' in doc or 'identifiab' in doc.lower()

    def test_docstring_references_spirtes(self, neural_causal):
        """Docstring must reference Spirtes et al. or Pearl."""
        doc = NeuralCausalModel.__doc__
        assert 'Spirtes' in doc or 'Pearl' in doc

    def test_docstring_mentions_verma_pearl(self, neural_causal):
        """Docstring must reference Verma & Pearl (1991) equivalence class."""
        doc = NeuralCausalModel.__doc__
        assert 'Verma' in doc or 'equivalence class' in doc.lower()


class TestNeuralCausalIdentifiabilityAssessment:
    """Validate identifiability_assessment() method on NeuralCausalModel."""

    def test_method_exists(self, neural_causal):
        """NeuralCausalModel must have identifiability_assessment method."""
        assert hasattr(neural_causal, 'identifiability_assessment')
        assert callable(neural_causal.identifiability_assessment)

    def test_returns_dict(self, neural_causal):
        """Method must return a dict."""
        result = neural_causal.identifiability_assessment()
        assert isinstance(result, dict)

    def test_assumptions_documented(self, neural_causal):
        """Must document all assumed conditions."""
        result = neural_causal.identifiability_assessment()
        assert result['causal_sufficiency_assumed'] is True
        assert result['markov_property_assumed'] is True
        assert result['faithfulness_assumed'] is True
        assert result['exogeneity_assumed'] is True
        assert result['fixed_ordering'] is True

    def test_identifiability_class(self, neural_causal):
        """Must report identifiability class."""
        result = neural_causal.identifiability_assessment()
        assert 'identifiability_class' in result
        assert 'point_identified' in result['identifiability_class']

    def test_markov_equivalence_note(self, neural_causal):
        """Must include Markov equivalence class note."""
        result = neural_causal.identifiability_assessment()
        assert 'markov_equivalence_note' in result
        note = result['markov_equivalence_note']
        assert 'Markov equivalence' in note or 'equivalence class' in note.lower()

    def test_edge_count(self, neural_causal):
        """Must report edge count of learned graph."""
        result = neural_causal.identifiability_assessment()
        assert 'edge_count' in result
        assert isinstance(result['edge_count'], (int, float))

    def test_graph_sparsity(self, neural_causal):
        """Must report graph sparsity."""
        result = neural_causal.identifiability_assessment()
        assert 'graph_sparsity' in result
        assert 0 <= result['graph_sparsity'] <= 1.0

    def test_unverified_assumptions(self, neural_causal):
        """Must list assumptions not verified against data."""
        result = neural_causal.identifiability_assessment()
        assert 'assumptions_not_verified' in result
        unverified = result['assumptions_not_verified']
        assert isinstance(unverified, list)
        assert len(unverified) >= 3
        # Must mention confounders
        unverified_text = ' '.join(unverified).lower()
        assert 'confounder' in unverified_text or 'sufficiency' in unverified_text

    def test_recommendations(self, neural_causal):
        """Must include validation recommendations."""
        result = neural_causal.identifiability_assessment()
        assert 'recommendations' in result
        recs = result['recommendations']
        assert isinstance(recs, list)
        assert len(recs) >= 4

    def test_recommendations_mention_notears(self, neural_causal):
        """Recommendations must mention NOTEARS as alternative."""
        result = neural_causal.identifiability_assessment()
        recs_text = ' '.join(result['recommendations']).lower()
        assert 'notears' in recs_text

    def test_recommendations_mention_do_calculus(self, neural_causal):
        """Recommendations must mention do-calculus identifiability."""
        result = neural_causal.identifiability_assessment()
        recs_text = ' '.join(result['recommendations'])
        assert 'do-calculus' in recs_text or 'Pearl' in recs_text


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Cross-cutting integration tests for all 5 refinements."""

    def test_full_fixed_point_with_all_refinements(self, meta_loop):
        """Full fixed-point computation must include all new fields."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        C, _, meta = ml.compute_fixed_point(psi, return_certificate=True)
        # Must have km_formal_convergence
        assert 'km_formal_convergence' in meta
        km = meta['km_formal_convergence']
        # If Banach, must have a_posteriori_validity
        if km['convergence_type'] == 'banach_contraction':
            assert 'a_posteriori_validity' in km
        # Dropout exclusion must be tracked on the meta-loop
        assert ml._dropout_excluded_during_certification is True

    def test_composite_cert_has_coverage_and_limitations(self, meta_loop):
        """Composite certificate must have both coverage and limitations."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'sample_coverage' in cert
        assert 'certificate_limitations' in cert
        # Coverage type must be local
        assert cert['sample_coverage']['coverage_type'] == 'local_empirical'

    def test_catastrophe_classification_complete(self, topology_analyzer):
        """Catastrophe classification must have both existing and new fields."""
        P = 8
        eigs = torch.ones(2, P) * 0.5
        eigs[:, 0] = 1e-5  # corank=1 (fold)
        kappa = torch.tensor([100.0, 100.0])
        grad = torch.tensor([0.01, 0.01])
        result = topology_analyzer.classify_catastrophe_type(
            eigs, kappa, grad
        )
        # Existing fields
        assert 'classification_basis' in result
        assert result['classification_basis'] == 'corank'
        assert result['jet_analysis_required'] is True
        # New fields
        assert 'unfolding_analysis' in result
        assert result['unfolding_analysis']['versal_unfolding_computed'] is False

    def test_causal_model_pipeline(self, neural_causal):
        """Full causal model pipeline with identifiability check."""
        # Forward pass
        exo = torch.randn(4, 6)
        output = neural_causal(exo)
        assert output.shape == (4, 6)
        # Identifiability assessment
        ident = neural_causal.identifiability_assessment()
        assert ident['fixed_ordering'] is True
        assert len(ident['assumptions_not_verified']) >= 3
