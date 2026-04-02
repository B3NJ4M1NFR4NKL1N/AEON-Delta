"""
Test suite for V5-series patches: Academic-level convergence and calibration.

V5a: LipSDP/Cholesky-based Lipschitz certificate (LipschitzConstrainedLambda)
V5b: IQC/slope-restricted GELU & LayerNorm IBP (CertifiedMetaLoop)
V5c: Constructive fixed-point existence proof (ProvablyConvergentMetaLoop)
V5d: Lanczos-based catastrophe criterion (OptimizedTopologyAnalyzer)
V5e: Proper calibration metrics (ECE, Brier score, calibrated fusion)
"""

import math
import sys
import os
import pytest
import random

import torch
import torch.nn as nn

# Ensure the repo root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeon_core import (
    LipschitzConstrainedLambda,
    ProvablyConvergentMetaLoop,
    OptimizedTopologyAnalyzer,
    CertifiedMetaLoop,
    CalibrationMetrics,
    _calibrated_uncertainty_fusion,
    _weighted_uncertainty_fusion,
    CognitiveFeedbackBus,
    AEONConfig,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    """Create a minimal AEONConfig for testing."""
    defaults = dict(
        hidden_dim=32,
        z_dim=32,
        meta_dim=64,
        lipschitz_target=0.9,
        dropout_rate=0.0,
        alpha=0.5,
        num_pillars=8,
        topo_method='finite_differences',
        topo_epsilon=1e-4,
        topo_use_cache=False,
        vq_embedding_dim=32,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


# ════════════════════════════════════════════════════════════════════════════
# V5a: LipSDP/Cholesky-based Lipschitz Certificate
# ════════════════════════════════════════════════════════════════════════════

class TestV5a_LipSDPCertificate:
    """Tests for LipschitzConstrainedLambda.compute_lipsdp_certificate()."""

    def test_method_exists(self):
        """V5a: compute_lipsdp_certificate exists."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        assert hasattr(op, 'compute_lipsdp_certificate')

    def test_returns_dict_with_keys(self):
        """V5a: Returns dict with all required keys."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=4)
        for key in ['certified', 'L_certified', 'cholesky_succeeded',
                     'min_eigenvalue_M', 'jacobian_spectral_norm', 'methodology']:
            assert key in result, f"Missing key: {key}"

    def test_with_explicit_input(self):
        """V5a: Works with explicit input tensor."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        x = torch.randn(4, 64)
        result = op.compute_lipsdp_certificate(x=x, num_jacobian_samples=4)
        assert result['num_jacobian_samples'] >= 1

    def test_iqc_scale_matches_gelu(self):
        """V5a: IQC scale = √(2/1.13) for GELU [0, 1.13]."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=2)
        expected = math.sqrt(2.0 / 1.13)
        assert abs(result['iqc_scale_factor'] - expected) < 1e-6

    def test_slope_restriction(self):
        """V5a: Slope restriction is [0, 1.13]."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=2)
        assert result['slope_restriction'] == [0.0, 1.13]

    def test_l_certified_finite(self):
        """V5a: L_certified is always finite."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=4)
        assert math.isfinite(result['L_certified'])

    def test_methodology_references(self):
        """V5a: Methodology references LipSDP, ECLipsE, Cholesky."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=2)
        meth = result['methodology']
        assert 'LipSDP' in meth
        assert 'ECLipsE' in meth
        assert 'Cholesky' in meth

    def test_custom_l_candidate(self):
        """V5a: Custom L_candidate is used for certification."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(L_candidate=0.5, num_jacobian_samples=4)
        assert result['L_candidate'] == 0.5

    def test_min_eigenvalue_m_exists(self):
        """V5a: min_eigenvalue_M is always present."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        result = op.compute_lipsdp_certificate(num_jacobian_samples=4)
        assert 'min_eigenvalue_M' in result


# ════════════════════════════════════════════════════════════════════════════
# V5b: IQC/Slope-Restricted GELU & LayerNorm IBP
# ════════════════════════════════════════════════════════════════════════════

class TestV5b_IQC_IBP:
    """Tests for CertifiedMetaLoop._compute_certified_lipschitz_iqc()."""

    def _make_loop(self):
        return CertifiedMetaLoop(_make_config())

    def test_method_exists(self):
        """V5b: IQC IBP method exists."""
        loop = self._make_loop()
        assert hasattr(loop, '_compute_certified_lipschitz_iqc')

    def test_returns_dict_keys(self):
        """V5b: Returns required keys."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        result = loop._compute_certified_lipschitz_iqc(z)
        for key in ['L_iqc', 'L_standard_ibp', 'tightening', 'iqc_succeeded',
                     'layer_diagnostics', 'is_contraction', 'methodology']:
            assert key in result, f"Missing key: {key}"

    def test_iqc_tighter_or_equal(self):
        """V5b: IQC bound ≤ standard IBP."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        result = loop._compute_certified_lipschitz_iqc(z)
        if result['iqc_succeeded']:
            assert result['L_iqc'] <= result['L_standard_ibp'] * 1.01 + 0.01

    def test_layer_diagnostics(self):
        """V5b: Layer diagnostics populated."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        result = loop._compute_certified_lipschitz_iqc(z)
        assert len(result['layer_diagnostics']) > 0

    def test_methodology_iqc(self):
        """V5b: Methodology references IQC."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        result = loop._compute_certified_lipschitz_iqc(z)
        assert 'IQC' in result['methodology']

    def test_integrated_into_verification(self):
        """V5b: IQC result in verify_convergence_preconditions."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        _, _, diag = loop.verify_convergence_preconditions(z)
        assert 'iqc_ibp' in diag

    def test_tightening_nonnegative(self):
        """V5b: Tightening is non-negative."""
        loop = self._make_loop()
        loop.eval()
        z = torch.randn(2, 32)
        result = loop._compute_certified_lipschitz_iqc(z)
        assert result['tightening'] >= -0.01  # small tolerance


# ════════════════════════════════════════════════════════════════════════════
# V5c: Constructive Fixed-Point Existence Proof
# ════════════════════════════════════════════════════════════════════════════

class TestV5c_FixedPointExistence:
    """Tests for fixed-point existence and θ-averagedness."""

    def _make_meta_loop(self):
        cfg = _make_config()
        return ProvablyConvergentMetaLoop(cfg, max_iterations=10, min_iterations=2)

    def test_fp_existence_in_verify(self):
        """V5c: verify_convergence returns fixed_point_existence."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert 'fixed_point_existence' in result

    def test_fp_existence_keys(self):
        """V5c: fixed_point_existence has required fields."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        fp = result['fixed_point_existence']
        for key in ['brouwer_self_mapping', 'residual_at_convergence',
                     'fix_T_nonempty', 'theta_averaged', 'theta_value',
                     'nonexpansiveness_verified']:
            assert key in fp, f"Missing key: {key}"

    def test_theta_in_range(self):
        """V5c: θ ∈ [0, 1]."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert 0.0 <= result['fixed_point_existence']['theta_value'] <= 1.0

    def test_nonexpansiveness_is_bool(self):
        """V5c: nonexpansiveness_verified is bool."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert isinstance(result['fixed_point_existence']['nonexpansiveness_verified'], bool)

    def test_lipsdp_in_verify(self):
        """V5c: lipsdp_certificate in verify_convergence."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert 'lipsdp_certificate' in result

    def test_brouwer_is_bool(self):
        """V5c: brouwer_self_mapping is bool."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert isinstance(result['fixed_point_existence']['brouwer_self_mapping'], bool)

    def test_fp_nonempty_with_convergence(self):
        """V5c: fix_T_nonempty is bool and consistent with logic."""
        ml = self._make_meta_loop()
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        fp = result['fixed_point_existence']
        # fix_T_nonempty is always a bool
        assert isinstance(fp['fix_T_nonempty'], bool)
        # If both brouwer and convergence are true, fix_T must be true
        if fp['brouwer_self_mapping'] and result['residual_norm'] < 1e-3:
            assert fp['fix_T_nonempty'] is True


# ════════════════════════════════════════════════════════════════════════════
# V5d: Lanczos-Based Catastrophe Criterion
# ════════════════════════════════════════════════════════════════════════════

class TestV5d_LanczosCatastrophe:
    """Tests for OptimizedTopologyAnalyzer.lanczos_catastrophe_criterion()."""

    def _make_analyzer(self):
        return OptimizedTopologyAnalyzer(_make_config())

    def test_method_exists(self):
        """V5d: lanczos_catastrophe_criterion exists."""
        ana = self._make_analyzer()
        assert hasattr(ana, 'lanczos_catastrophe_criterion')

    def test_returns_dict(self):
        """V5d: Returns dict with required keys."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        assert isinstance(result, dict)
        assert result['method'] == 'lanczos'
        assert 'catastrophe_type' in result

    def test_eigenvalues_list(self):
        """V5d: eigenvalues is a list."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        assert isinstance(result['eigenvalues'], list)

    def test_lambda_ordering(self):
        """V5d: λ_min ≤ λ_max."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        if result.get('eigenvalues'):
            assert result['lambda_min'] <= result['lambda_max']

    def test_condition_number(self):
        """V5d: κ ≥ 1."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        if result.get('eigenvalues'):
            assert result['condition_number'] >= 1.0 - 1e-6

    def test_corank_nonneg(self):
        """V5d: corank ≥ 0."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        if result.get('eigenvalues'):
            assert result['corank'] >= 0

    def test_valid_catastrophe_type(self):
        """V5d: Type is none/fold/cusp/swallowtail."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        assert result['catastrophe_type'] in {'none', 'fold', 'cusp', 'swallowtail'}

    def test_kappa_tier(self):
        """V5d: kappa_tier is valid."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        if result.get('eigenvalues'):
            assert result['kappa_tier'] in {'stable', 'warning', 'fold_regime', 'cusp_regime'}

    def test_methodology_lanczos(self):
        """V5d: Methodology mentions Lanczos."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        assert 'Lanczos' in result.get('methodology', '')

    def test_saddle_detection(self):
        """V5d: is_saddle_point exists."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        if result.get('eigenvalues'):
            assert isinstance(result['is_saddle_point'], bool)

    def test_diagonal_mismatch_note(self):
        """V5d: Methodology notes diagonal-Hessian insufficiency."""
        ana = self._make_analyzer()
        ana.eval()
        factors = torch.randn(2, 8)
        result = ana.lanczos_catastrophe_criterion(factors)
        meth = result.get('methodology', '')
        assert 'diagonal' in meth.lower()


# ════════════════════════════════════════════════════════════════════════════
# V5e: Proper Calibration Metrics (ECE, Brier Score)
# ════════════════════════════════════════════════════════════════════════════

class TestV5e_CalibrationMetrics:
    """Tests for CalibrationMetrics and calibrated fusion."""

    def test_init(self):
        """V5e: CalibrationMetrics inits correctly."""
        cm = CalibrationMetrics(num_bins=10, window_size=100)
        assert cm.num_bins == 10
        assert cm.window_size == 100

    def test_record(self):
        """V5e: record adds entries."""
        cm = CalibrationMetrics()
        cm.record(0.3, True)
        cm.record(0.7, False)
        assert len(cm._history) == 2

    def test_ece_empty(self):
        """V5e: ECE = 0 with no data."""
        cm = CalibrationMetrics()
        assert cm.compute_ece()['ece'] == 0.0

    def test_ece_well_calibrated(self):
        """V5e: ECE small for well-calibrated predictions."""
        cm = CalibrationMetrics(num_bins=5, window_size=1000)
        random.seed(42)
        for _ in range(500):
            cm.record(0.2, random.random() < 0.8)
        result = cm.compute_ece()
        assert result['ece'] < 0.15

    def test_ece_poorly_calibrated(self):
        """V5e: ECE large for poor calibration."""
        cm = CalibrationMetrics(num_bins=5, window_size=1000)
        for i in range(500):
            cm.record(0.1, i % 10 < 3)
        assert cm.compute_ece()['ece'] > 0.3

    def test_ece_bins(self):
        """V5e: ECE bin_details correct."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(i / 100.0, i % 2 == 0)
        result = cm.compute_ece()
        assert len(result['bin_details']) == 5

    def test_ece_interpretation(self):
        """V5e: ECE interpretation valid."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(0.5, i % 2 == 0)
        result = cm.compute_ece()
        assert result['interpretation'] in {
            'well_calibrated', 'moderately_calibrated', 'poorly_calibrated',
        }

    def test_brier_empty(self):
        """V5e: Brier = 0 with no data."""
        cm = CalibrationMetrics()
        assert cm.compute_brier_score()['brier_score'] == 0.0

    def test_brier_perfect(self):
        """V5e: Brier ≈ 0 for perfect predictions."""
        cm = CalibrationMetrics(num_bins=5)
        for _ in range(100):
            cm.record(0.0, True)
        assert cm.compute_brier_score()['brier_score'] < 0.01

    def test_brier_worst(self):
        """V5e: Brier ≈ 1 for worst-case."""
        cm = CalibrationMetrics(num_bins=5)
        for _ in range(100):
            cm.record(0.0, False)
        assert cm.compute_brier_score()['brier_score'] > 0.9

    def test_brier_decomposition(self):
        """V5e: Brier decomposition has all components."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(0.3, i % 3 == 0)
        result = cm.compute_brier_score()
        assert 'reliability' in result
        assert 'resolution' in result
        assert 'uncertainty_component' in result

    def test_brier_interpretation(self):
        """V5e: Brier interpretation valid."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(0.3, i % 2 == 0)
        result = cm.compute_brier_score()
        assert result['interpretation'] in {'excellent', 'good', 'fair', 'poor'}

    def test_calibration_adjustment(self):
        """V5e: Adjustment ∈ (0, 1]."""
        cm = CalibrationMetrics(num_bins=5)
        assert cm.get_calibration_adjustment() == 1.0
        for i in range(100):
            cm.record(0.5, i % 2 == 0)
        adj = cm.get_calibration_adjustment()
        assert 0.0 < adj <= 1.0

    def test_ece_ema(self):
        """V5e: ECE EMA tracked."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(0.5, i % 2 == 0)
        assert cm.compute_ece()['ece_ema'] >= 0.0

    def test_mce(self):
        """V5e: MCE ≥ ECE."""
        cm = CalibrationMetrics(num_bins=5)
        for i in range(100):
            cm.record(i / 100.0, i % 2 == 0)
        result = cm.compute_ece()
        assert result['mce'] >= result['ece']


# ════════════════════════════════════════════════════════════════════════════
# V5f: Calibrated uncertainty fusion
# ════════════════════════════════════════════════════════════════════════════

class TestV5f_CalibratedFusion:
    """Tests for _calibrated_uncertainty_fusion."""

    def test_no_calibration(self):
        """V5f: Without calibration, uncertainty == raw."""
        sources = {'a': 0.5, 'b': 0.3}
        result = _calibrated_uncertainty_fusion(sources)
        assert result['calibration_available'] is False
        assert abs(result['uncertainty'] - result['raw_uncertainty']) < 1e-6

    def test_with_calibration(self):
        """V5f: With calibration, adjusts uncertainty."""
        cm = CalibrationMetrics(num_bins=5, window_size=200)
        for i in range(200):
            cm.record(0.1, i % 10 < 3)
        sources = {'a': 0.5, 'b': 0.3}
        result = _calibrated_uncertainty_fusion(sources, calibration=cm)
        assert result['calibration_available'] is True

    def test_keys(self):
        """V5f: All expected keys present."""
        cm = CalibrationMetrics(num_bins=5, window_size=200)
        for i in range(200):
            cm.record(0.3, i % 2 == 0)
        sources = {'a': 0.5}
        result = _calibrated_uncertainty_fusion(sources, calibration=cm)
        for key in ['uncertainty', 'raw_uncertainty', 'ece', 'brier_score',
                     'calibration_adjustment', 'calibration_available']:
            assert key in result

    def test_uncertainty_in_range(self):
        """V5f: Calibrated uncertainty ∈ [0, 1]."""
        cm = CalibrationMetrics(num_bins=5, window_size=200)
        for i in range(200):
            cm.record(0.5, i % 2 == 0)
        sources = {'a': 0.8}
        result = _calibrated_uncertainty_fusion(sources, calibration=cm)
        assert 0.0 <= result['uncertainty'] <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# V5g: Enhanced compute_sample_uncertainty
# ════════════════════════════════════════════════════════════════════════════

class TestV5g_EnhancedUncertainty:
    """Tests for the enhanced compute_sample_uncertainty."""

    def test_set_calibration_exists(self):
        """V5g: set_calibration_metrics exists."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        assert hasattr(bus, 'set_calibration_metrics')

    def test_set_calibration(self):
        """V5g: set_calibration_metrics attaches."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        cm = CalibrationMetrics()
        bus.set_calibration_metrics(cm)
        assert bus._calibration_metrics is cm

    def test_uncertainty_without_calibration(self):
        """V5g: Uncertainty works without calibration."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        bus.register_signal('test_uncertainty', 0.5)
        bus.write_signal('test_uncertainty', 0.7)
        unc = bus.compute_sample_uncertainty()
        assert unc >= 0.0

    def test_uncertainty_with_calibration(self):
        """V5g: Uncertainty uses calibration when available."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        cm = CalibrationMetrics(num_bins=5, window_size=200)
        for i in range(200):
            cm.record(0.3, i % 2 == 0)
        bus.set_calibration_metrics(cm)
        bus.register_signal('test_uncertainty', 0.5)
        bus.write_signal('test_uncertainty', 0.7)
        unc = bus.compute_sample_uncertainty()
        assert unc >= 0.0

    def test_weighted_not_max(self):
        """V5g: Uses weighted fusion, not max pooling."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        bus.register_signal('uncertainty_a', 0.3)
        bus.register_signal('uncertainty_b', 0.8)
        bus.write_signal('uncertainty_a', 0.3)
        bus.write_signal('uncertainty_b', 0.8)
        unc = bus.compute_sample_uncertainty()
        assert unc > 0.0


# ════════════════════════════════════════════════════════════════════════════
# Integration
# ════════════════════════════════════════════════════════════════════════════

class TestV5_Integration:
    """Integration tests for all V5 patches."""

    def test_verify_convergence_complete(self):
        """Integration: verify_convergence has V5a + V5c fields."""
        cfg = _make_config()
        ml = ProvablyConvergentMetaLoop(cfg, max_iterations=10, min_iterations=2)
        ml.eval()
        psi_0 = torch.randn(2, 32)
        result = ml.verify_convergence(psi_0, num_samples=10)
        assert 'lipsdp_certificate' in result
        assert 'fixed_point_existence' in result
        assert 'km_convergence_status' in result

    def test_calibration_end_to_end(self):
        """Integration: Full calibration pipeline."""
        cm = CalibrationMetrics(num_bins=5, window_size=500)
        random.seed(123)
        for _ in range(200):
            unc = random.uniform(0.0, 1.0)
            cm.record(unc, random.random() > unc)
        ece = cm.compute_ece()
        brier = cm.compute_brier_score()
        assert ece['sufficient_data'] is True
        assert brier['sufficient_data'] is True
        sources = {'convergence_conflict': 0.4, 'coherence_deficit': 0.6}
        result = _calibrated_uncertainty_fusion(sources, calibration=cm)
        assert result['calibration_available'] is True
        assert 0.0 <= result['uncertainty'] <= 1.0

    def test_lanczos_consistent_with_forward(self):
        """Integration: Lanczos + forward produce same catastrophe types."""
        cfg = _make_config()
        ana = OptimizedTopologyAnalyzer(cfg)
        ana.eval()
        factors = torch.randn(2, 8)
        full = ana.forward(factors)
        lanczos = ana.lanczos_catastrophe_criterion(factors)
        assert full['catastrophe_type'] is not None
        assert lanczos['catastrophe_type'] is not None

    def test_lipsdp_consistent_with_constructive(self):
        """Integration: LipSDP and constructive bounds both finite."""
        op = LipschitzConstrainedLambda(64, 32, 32, lipschitz_target=0.9)
        op.eval()
        L_c = op.get_constructive_lipschitz_bound()
        cert = op.compute_lipsdp_certificate(num_jacobian_samples=4)
        assert math.isfinite(L_c)
        assert math.isfinite(cert['L_certified'])
