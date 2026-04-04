"""Tests for FCA-1 through FCA-6 formal certification patches.

Validates:
  FCA-1: Convergence guarantee level classification in certificates
  FCA-2: Krasnoselskii-Mann per-iterate bound (Opial–Fejér)
  FCA-3: Structured IQC interconnection decomposition
  FCA-4: Lyapunov formal verification conditions
  FCA-5: LayerNorm-gate sensitivity analysis
  FCA-6: Convergence degradation reporting
"""

import math
import sys
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from aeon_core import (
    AEONConfig,
    CognitiveFeedbackBus,
    OptimizedTopologyAnalyzer,
    ProvablyConvergentMetaLoop,
    RecursionUtilityGate,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_config(**overrides) -> AEONConfig:
    hd = overrides.pop('hidden_dim', 64)
    defaults = dict(
        hidden_dim=hd,
        z_dim=hd,
        vq_embedding_dim=hd,
        device_str='cpu',
        enable_error_evolution=True,
        enable_metacognitive_recursion=True,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_meta_loop(config=None, max_iterations=5, min_iterations=2):
    if config is None:
        config = _make_config()
    ml = ProvablyConvergentMetaLoop(
        config=config,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
    )
    ml.eval()
    return ml


def _run_fixed_point(hidden_dim=64):
    """Run compute_fixed_point and return (C_star, iterations, metadata)."""
    config = _make_config(hidden_dim=hidden_dim)
    ml = _make_meta_loop(config)
    psi_0 = torch.randn(2, hidden_dim)
    with torch.no_grad():
        return ml.compute_fixed_point(psi_0, return_certificate=True)


def _get_certificate(hidden_dim=64):
    """Return the certificate dict from compute_fixed_point metadata."""
    _, _, metadata = _run_fixed_point(hidden_dim)
    return metadata['certificate']


def _get_km_convergence(hidden_dim=64):
    """Return the km_formal_convergence dict from metadata."""
    _, _, metadata = _run_fixed_point(hidden_dim)
    return metadata.get('km_formal_convergence', {})


def _get_composite_T_cert(hidden_dim=64):
    """Return the composite T certificate (contains structured_interconnection)."""
    config = _make_config(hidden_dim=hidden_dim)
    ml = _make_meta_loop(config)
    psi_0 = torch.randn(2, hidden_dim)
    with torch.no_grad():
        return ml.compute_composite_T_certificate(psi_0, num_jacobian_samples=3)


def _get_lyapunov_result(hidden_dim=64):
    """Return verify_lyapunov_descent result with T_T_z for full analysis."""
    config = _make_config(hidden_dim=hidden_dim)
    analyzer = OptimizedTopologyAnalyzer(config)
    z = torch.randn(4, hidden_dim)
    T_z = z * 0.9 + torch.randn_like(z) * 0.01
    T_T_z = T_z * 0.9 + torch.randn_like(T_z) * 0.01
    return analyzer.verify_lyapunov_descent(z, T_z, T_T_z)


def _get_uniform_contraction(hidden_dim=64):
    """Return verify_uniform_contraction result from lambda_op."""
    config = _make_config(hidden_dim=hidden_dim)
    ml = _make_meta_loop(config)
    psi_0 = torch.randn(2, hidden_dim)
    with torch.no_grad():
        return ml.lambda_op.verify_uniform_contraction(psi_0)


def _make_utility_gate():
    return RecursionUtilityGate()


# ═══════════════════════════════════════════════════════════════════
# FCA-1: Convergence Guarantee Level
# ═══════════════════════════════════════════════════════════════════


class TestFCA1_ConvergenceGuaranteeLevel:
    """FCA-1: Verify convergence_guarantee_level classification in certificate."""

    def test_certificate_contains_convergence_guarantee_level(self):
        cert = _get_certificate()
        assert 'convergence_guarantee_level' in cert

    def test_certificate_contains_banach_bound_valid(self):
        cert = _get_certificate()
        assert 'banach_bound_valid' in cert

    def test_certificate_contains_L_C(self):
        cert = _get_certificate()
        assert 'L_C' in cert

    def test_certificate_contains_L_C_source(self):
        cert = _get_certificate()
        assert 'L_C_source' in cert
        assert cert['L_C_source'] in (
            'analytical_compositional',
            'empirical_jacobian',
        )

    def test_guarantee_level_is_valid_enum(self):
        cert = _get_certificate()
        allowed = {
            'certified_contraction',
            'empirical_contraction',
            'nonexpansive_heuristic',
            'no_guarantee',
        }
        assert cert['convergence_guarantee_level'] in allowed

    def test_banach_bound_valid_type(self):
        cert = _get_certificate()
        assert isinstance(cert['banach_bound_valid'], bool)

    def test_L_C_source_valid_values(self):
        cert = _get_certificate()
        assert cert['L_C_source'] in (
            'analytical_compositional',
            'empirical_jacobian',
        )


# ═══════════════════════════════════════════════════════════════════
# FCA-2: KM Per-Iterate Bound
# ═══════════════════════════════════════════════════════════════════


class TestFCA2_KMPerIterateBound:
    """FCA-2: Verify km_formal_convergence and per-iterate bound structure."""

    def test_km_formal_convergence_exists(self):
        km = _get_km_convergence()
        assert 'convergence_type' in km

    def test_km_convergence_type_valid(self):
        km = _get_km_convergence()
        assert km['convergence_type'] in (
            'banach_contraction',
            'krasnoselskii_mann',
            'unverified',
        )

    def test_km_convergence_strength_valid(self):
        km = _get_km_convergence()
        assert km.get('convergence_strength') in (
            'strong',
            'weak',
            'none',
        )

    def test_km_per_iterate_bound_in_convergence_dict(self):
        """When KM convergence applies, per_iterate_bound dict exists."""
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            assert 'per_iterate_bound' in km

    def test_per_iterate_bound_formula(self):
        """Verify the Opial inequality formula string when KM applies."""
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            pib = km['per_iterate_bound']
            assert 'formula' in pib
            assert '‖x' in pib['formula'] or 'x_' in pib['formula']

    def test_per_iterate_bound_reference(self):
        """Verify the reference field when KM applies."""
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            pib = km['per_iterate_bound']
            assert 'reference' in pib

    def test_per_iterate_bound_residual_energy_sum_nonnegative(self):
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            pib = km['per_iterate_bound']
            assert pib.get('residual_energy_sum', 0.0) >= 0.0

    def test_per_iterate_bound_status_values(self):
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            pib = km['per_iterate_bound']
            status = pib.get('bound_status', '')
            assert status == 'valid' or status.startswith('conditional')

    def test_per_iterate_bound_has_caveat(self):
        km = _get_km_convergence()
        if km.get('convergence_type') == 'krasnoselskii_mann':
            pib = km['per_iterate_bound']
            assert 'caveat' in pib
            assert isinstance(pib['caveat'], str)
            assert len(pib['caveat']) > 0

    def test_banach_error_bound_formula(self):
        """When Banach contraction, verify error_bound_formula present."""
        km = _get_km_convergence()
        if km.get('convergence_type') == 'banach_contraction':
            assert 'error_bound_formula' in km
            assert 'certified_error_bound' in km


# ═══════════════════════════════════════════════════════════════════
# FCA-3: Structured IQC Interconnection
# ═══════════════════════════════════════════════════════════════════


class TestFCA3_StructuredIQC:
    """FCA-3: Verify structured_interconnection decomposition in IQC cert."""

    def test_structured_interconnection_exists(self):
        cert = _get_composite_T_cert()
        assert 'structured_interconnection' in cert

    def test_structured_interconnection_has_decomposition(self):
        cert = _get_composite_T_cert()
        si = cert['structured_interconnection']
        assert 'decomposition' in si
        decomp = si['decomposition']
        expected_blocks = [
            'block_1_input_ln',
            'block_2_linear_W1',
            'block_3_gelu',
            'block_4_linear_W2',
            'block_5_output_ln',
            'block_6_feedback_gate',
        ]
        for block in expected_blocks:
            assert block in decomp, f"Missing block: {block}"

    def test_block_3_gelu_sector(self):
        cert = _get_composite_T_cert()
        gelu = cert['structured_interconnection']['decomposition']['block_3_gelu']
        assert 'sector' in gelu
        assert '[0, 1.13]' in gelu['sector']

    def test_structured_matrix_inequality_documented(self):
        cert = _get_composite_T_cert()
        si = cert['structured_interconnection']
        assert 'structured_matrix_inequality' in si
        assert isinstance(si['structured_matrix_inequality'], str)
        assert len(si['structured_matrix_inequality']) > 0

    def test_missing_steps_for_full_iqc(self):
        cert = _get_composite_T_cert()
        si = cert['structured_interconnection']
        assert 'missing_steps_for_full_iqc' in si
        steps = si['missing_steps_for_full_iqc']
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_actionable_path_exists(self):
        cert = _get_composite_T_cert()
        si = cert['structured_interconnection']
        assert 'actionable_path' in si
        path = si['actionable_path']
        assert isinstance(path, str)
        path_lower = path.lower()
        assert 'cvxpy' in path_lower or 'sdp' in path_lower

    def test_iqc_distinction_still_present(self):
        cert = _get_composite_T_cert()
        assert 'iqc_distinction' in cert

    def test_block_types_are_valid(self):
        cert = _get_composite_T_cert()
        decomp = cert['structured_interconnection']['decomposition']
        for block_name, block_info in decomp.items():
            assert 'type' in block_info, f"{block_name} missing 'type'"
            assert block_info['type'] in ('linear', 'nonlinear'), (
                f"{block_name} has invalid type: {block_info['type']}"
            )


# ═══════════════════════════════════════════════════════════════════
# FCA-4: Lyapunov Formal Verification
# ═══════════════════════════════════════════════════════════════════


class TestFCA4_LyapunovFormalVerification:
    """FCA-4: Verify lyapunov_formal_verification conditions."""

    def test_lyapunov_formal_verification_exists(self):
        result = _get_lyapunov_result()
        assert 'lyapunov_formal_verification' in result

    def test_condition_i_zero_at_fixpoint(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        cond = lfv['condition_i_zero_at_fixpoint']
        assert 'status' in cond
        assert isinstance(cond['status'], bool)
        assert 'V_min' in cond

    def test_condition_ii_positive_definiteness(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        cond = lfv['condition_ii_positive_definiteness']
        assert 'status' in cond
        assert isinstance(cond['status'], bool)
        assert 'V_mean' in cond
        assert 'V_min' in cond

    def test_condition_iii_strict_descent(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        cond = lfv['condition_iii_strict_descent']
        assert 'status' in cond
        assert 'exponential_stability' in cond
        assert isinstance(cond['exponential_stability'], bool)

    def test_radial_unboundedness(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        ru = lfv['radial_unboundedness']
        assert 'status' in ru
        assert isinstance(ru['status'], str)

    def test_overall_lyapunov_status_values(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        assert lfv['overall_lyapunov_status'] in (
            'certified',
            'conditional',
            'not_verified',
        )

    def test_formal_limitations_documented(self):
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        assert 'formal_limitations' in lfv
        assert isinstance(lfv['formal_limitations'], str)
        assert len(lfv['formal_limitations']) > 0

    def test_spectral_descent_rate(self):
        """When contraction_ratio < 1, spectral_descent_rate should be > 0."""
        result = _get_lyapunov_result()
        cr = result.get('contraction_ratio', 1.0)
        lfv = result['lyapunov_formal_verification']
        cond_iii = lfv['condition_iii_strict_descent']
        if cr < 1.0:
            assert cond_iii['spectral_descent_rate'] is not None
            assert cond_iii['spectral_descent_rate'] > 0.0
        else:
            # When not contractive, spectral_descent_rate may be None
            assert cond_iii['spectral_descent_rate'] is None or cr >= 1.0

    def test_contraction_implies_certified_lyapunov(self):
        """When descent_holds and contraction_ratio < 1, status is 'certified'."""
        result = _get_lyapunov_result()
        lfv = result['lyapunov_formal_verification']
        status = lfv['overall_lyapunov_status']
        descent = result.get('descent_holds', False)
        cr = result.get('contraction_ratio', 1.0)
        if descent and cr < 1.0:
            assert status == 'certified'
        else:
            # If conditions aren't met, status must still be valid
            assert status in ('certified', 'conditional', 'not_verified')


# ═══════════════════════════════════════════════════════════════════
# FCA-5: LayerNorm-Gate Sensitivity
# ═══════════════════════════════════════════════════════════════════


class TestFCA5_LayerNormGateSensitivity:
    """FCA-5: Verify layernorm_gate_sensitivity in uniform contraction result."""

    def test_layernorm_gate_sensitivity_exists(self):
        result = _get_uniform_contraction()
        assert 'layernorm_gate_sensitivity' in result

    def test_sensitivity_has_description(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        assert 'description' in lgs
        assert isinstance(lgs['description'], str)
        assert len(lgs['description']) > 0

    def test_current_gamma_ratio_input(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        val = lgs['current_gamma_ratio_input']
        assert isinstance(val, float)
        assert math.isfinite(val)
        assert val >= 0.0

    def test_current_gamma_ratio_output(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        val = lgs['current_gamma_ratio_output']
        assert isinstance(val, float)
        assert math.isfinite(val)
        assert val >= 0.0

    def test_sensitivity_dLT_dGammaRatio_positive(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        assert lgs['sensitivity_dLT_dGammaRatio_input'] >= 0.0
        assert lgs['sensitivity_dLT_dGammaRatio_output'] >= 0.0

    def test_target_gamma_ratio_finite(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        val = lgs['target_gamma_ratio_for_contraction']
        assert isinstance(val, float)
        # May be inf when alpha == 0; otherwise should be finite positive
        assert val > 0.0 or val == float('inf')

    def test_gamma_recommendations_present(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        assert 'gamma_recommendations' in lgs
        assert isinstance(lgs['gamma_recommendations'], str)
        assert len(lgs['gamma_recommendations']) > 0

    def test_current_bottleneck_valid(self):
        result = _get_uniform_contraction()
        lgs = result['layernorm_gate_sensitivity']
        assert 'current_bottleneck' in lgs
        assert lgs['current_bottleneck'] in (
            'layernorm_input',
            'layernorm_output',
            'feedback_gate',
            'within_budget',
        )


# ═══════════════════════════════════════════════════════════════════
# FCA-6: Degradation Reporting
# ═══════════════════════════════════════════════════════════════════


class TestFCA6_DegradationReporting:
    """FCA-6: Verify convergence_degradation reporting in RecursionUtilityGate."""

    def test_banach_bound_valid_true_when_L_less_1(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=0.8,
        )
        assert result['banach_bound_valid'] is True

    def test_banach_bound_valid_false_when_L_geq_1(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.1,
        )
        assert result['banach_bound_valid'] is False

    def test_convergence_degradation_present_when_L_geq_1(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.1,
        )
        assert 'convergence_degradation' in result

    def test_convergence_degradation_has_level(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.1,
        )
        deg = result['convergence_degradation']
        assert deg['degradation_level'] in ('moderate', 'severe', 'critical')

    def test_degradation_level_moderate(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.05,
        )
        assert result['convergence_degradation']['degradation_level'] == 'moderate'

    def test_degradation_level_severe(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.3,
        )
        assert result['convergence_degradation']['degradation_level'] == 'severe'

    def test_degradation_level_critical(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.6,
        )
        assert result['convergence_degradation']['degradation_level'] == 'critical'

    def test_no_degradation_when_contractive(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=0.9,
        )
        assert 'convergence_degradation' not in result

    def test_feedback_bus_receives_degradation_signal(self):
        gate = _make_utility_gate()
        bus = CognitiveFeedbackBus(hidden_dim=64)
        lc = 1.2
        gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=lc, feedback_bus=bus,
        )
        signal = bus.read_signal('convergence_degradation_pressure', default=-1.0)
        assert signal > 0.0
        # Signal should be proportional to (L_C - 1.0), scaled by
        # DEGRADATION_SIGNAL_SCALE = 5.0 (matches aeon_core FCA-6 constant)
        DEGRADATION_SIGNAL_SCALE = 5.0
        expected = min(1.0, (lc - 1.0) * DEGRADATION_SIGNAL_SCALE)
        assert signal == pytest.approx(expected)

    def test_elevated_threshold_in_degradation(self):
        gate = _make_utility_gate()
        result = gate.evaluate_recursion_utility(
            1.0, 0.5, lipschitz_constant=1.1,
        )
        deg = result['convergence_degradation']
        expected = gate.improvement_threshold * 1.5
        assert deg['elevated_threshold'] == pytest.approx(expected)
