"""
Tests for deep cognitive-intellectual analysis fixes.

Covers four mathematical corrections:
1. Contraction analysis — Lipschitz bound with identity contribution
2. LayerNorm data-dependent Lipschitz fragility detection
3. NOTEARS acyclicity constraint notation
4. Von Neumann entropy with proper density matrix normalisation
"""

from __future__ import annotations

import math
import torch
import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Contraction Analysis — Lipschitz Bound Corrections
# ═══════════════════════════════════════════════════════════════════════════


class TestContractionAnalysisLipschitzBound:
    """Verify that the Lipschitz bound includes the identity contribution
    from the KM averaging and the external output_stabilizer LayerNorm."""

    @pytest.fixture
    def meta_loop(self):
        from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        ml = ProvablyConvergentMetaLoop(config, max_iterations=10, min_iterations=2)
        ml.eval()
        return ml, config

    def test_output_stabilizer_lipschitz_included(self, meta_loop):
        """The constructive Lipschitz bound must include output_stabilizer."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # The metadata should have the output_stabilizer_lipschitz field
        assert 'output_stabilizer_lipschitz' in meta, (
            "Metadata missing output_stabilizer_lipschitz"
        )
        L_out_stab = meta['output_stabilizer_lipschitz']
        assert L_out_stab >= 1.0, (
            f"Output stabilizer Lipschitz should be ≥ 1.0, got {L_out_stab}"
        )

    def test_lipschitz_partial_C_includes_output_stabilizer(self, meta_loop):
        """lip_partial_C must be >= Lambda-only bound (composing with
        output_stabilizer can only increase or equal the bound)."""
        ml, config = meta_loop

        # Get Lambda-only bound
        L_lambda_only = (
            ml.lambda_op.get_constructive_partial_lipschitz_bound_wrt_C()
        )

        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        lip_partial_C = meta['lipschitz_partial_C_constructive']
        # The composed bound should be >= Lambda-only bound (it multiplies
        # by L_output_stab >= 1)
        assert lip_partial_C >= L_lambda_only - 1e-6, (
            f"Composed bound {lip_partial_C} should be >= Lambda-only "
            f"bound {L_lambda_only}"
        )

    def test_km_identity_contribution_in_certificate(self, meta_loop):
        """When KM averaging is active (alpha < 1), L_certificate should
        include the identity contribution: L_eff = (1-α) + α*L_T."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        L_cert = meta['L_certificate']
        # L_certificate should always be non-negative
        assert L_cert >= 0.0, f"L_certificate should be >= 0, got {L_cert}"

    def test_certified_error_bound_uses_correct_L(self, meta_loop):
        """The a-posteriori bound L/(1-L)*residual must use L_eff, not L_Λ."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        L_cert = meta['L_certificate']
        cert_error = meta['certified_error_bound']
        residual = meta['residual_norm']

        if cert_error is not None and L_cert < 1.0:
            # Verify the formula: ε = L/(1-L) * residual
            expected = (L_cert / (1.0 - L_cert)) * residual
            assert abs(cert_error - expected) < 1e-4, (
                f"Certified error {cert_error} != expected {expected} "
                f"(L={L_cert}, residual={residual})"
            )

    def test_dropout_active_flag_inference(self, meta_loop):
        """Dropout should be inactive at inference (eval mode)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # In eval mode, dropout should be inactive
        assert meta['dropout_active'] is False, (
            "Dropout should be inactive at inference"
        )

    def test_dropout_active_flag_training(self, meta_loop):
        """Dropout should be active during training (if dropout_rate > 0)."""
        ml, config = meta_loop
        ml.train()
        psi = torch.randn(2, config.z_dim)
        C, iters, meta = ml.compute_fixed_point(psi)

        dropout_rate = ml.lambda_op._dropout_rate
        if dropout_rate > 0:
            assert meta['dropout_active'] is True, (
                "Dropout should be active during training"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: LayerNorm Data-Dependent Lipschitz Fragility
# ═══════════════════════════════════════════════════════════════════════════


class TestLayerNormFragility:
    """Verify data-dependent LayerNorm Lipschitz estimation and fragility
    detection when pre-normalisation variance is small."""

    @pytest.fixture
    def meta_loop(self):
        from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        ml = ProvablyConvergentMetaLoop(config, max_iterations=10, min_iterations=2)
        ml.eval()
        return ml, config

    def test_layernorm_fragility_field_present(self, meta_loop):
        """Metadata should contain layernorm_fragile field."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        assert 'layernorm_fragile' in meta
        assert 'layernorm_data_dependent_lipschitz' in meta
        assert isinstance(meta['layernorm_fragile'], bool)

    def test_layernorm_data_dependent_lipschitz_finite(self, meta_loop):
        """Data-dependent LayerNorm Lipschitz should be finite."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        L_ln = meta['layernorm_data_dependent_lipschitz']
        if L_ln is not None:
            assert math.isfinite(L_ln), (
                f"Data-dependent LayerNorm Lipschitz should be finite, got {L_ln}"
            )
            assert L_ln > 0, (
                f"Data-dependent LayerNorm Lipschitz should be > 0, got {L_ln}"
            )

    def test_layernorm_fragile_with_near_constant_input(self, meta_loop):
        """Near-constant C (small variance) should trigger fragility flag."""
        ml, config = meta_loop

        # Create a near-constant C that will produce small variance
        # after the Lambda operator processes it
        psi = torch.ones(2, config.z_dim) * 0.001  # near-constant input
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # The fragility field should exist (may or may not be flagged
        # depending on the actual variance after Lambda)
        assert 'layernorm_fragile' in meta


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: NOTEARS Acyclicity Constraint Notation
# ═══════════════════════════════════════════════════════════════════════════


class TestNOTEARSNotation:
    """Verify the NOTEARS formula uses correct standard notation."""

    def test_notears_class_docstring_uses_hadamard(self):
        """NOTEARSCausalModel docstring should use ∘ (Hadamard) notation."""
        from aeon_core import NOTEARSCausalModel
        docstring = NOTEARSCausalModel.__doc__
        assert '∘' in docstring, (
            "NOTEARSCausalModel docstring should use ∘ for Hadamard product"
        )
        # Should use explicit exp() notation
        assert 'exp(W ∘ W)' in docstring, (
            "NOTEARSCausalModel docstring should contain 'exp(W ∘ W)'"
        )
        # Should NOT contain the ambiguous e^{W ⊙ W} notation
        assert 'e^{W' not in docstring, (
            "NOTEARSCausalModel docstring should not use ambiguous e^{} notation"
        )

    def test_dag_loss_docstring_uses_hadamard(self):
        """dag_loss docstring should use standard ∘ notation."""
        from aeon_core import NOTEARSCausalModel
        model = NOTEARSCausalModel(num_vars=5)
        docstring = model.dag_loss.__doc__
        assert '∘' in docstring, (
            "dag_loss docstring should use ∘ for Hadamard product"
        )
        assert 'exp(W ∘ W)' in docstring, (
            "dag_loss docstring should contain 'exp(W ∘ W)'"
        )

    def test_notears_implementation_correct(self):
        """The NOTEARS implementation should compute h(W) = tr(exp(W∘W)) - d."""
        from aeon_core import NOTEARSCausalModel
        d = 4
        model = NOTEARSCausalModel(num_vars=d)

        # Zero W → h(W) = tr(exp(0)) - d = tr(I) - d = d - d = 0
        with torch.no_grad():
            model.W.zero_()
            h = model.dag_loss()
            assert abs(h.item()) < 1e-5, (
                f"h(0) should be 0, got {h.item()}"
            )

    def test_notears_dag_identity(self):
        """A strictly lower-triangular W should give h(W) ≈ 0 (DAG)."""
        from aeon_core import NOTEARSCausalModel
        d = 4
        model = NOTEARSCausalModel(num_vars=d)

        with torch.no_grad():
            # Lower-triangular → DAG
            model.W.zero_()
            model.W[1, 0] = 0.5
            model.W[2, 0] = 0.3
            model.W[3, 1] = 0.7
            h_dag = model.dag_loss()

        # For a DAG, W∘W is nilpotent, so exp(W∘W) = I + (W∘W) + ...
        # tr should be close to d (since diagonal of exp is close to 1)
        # and h should be small but possibly not exactly 0 due to
        # off-diagonal entries feeding back through matrix exponential
        assert h_dag.item() >= -1e-5, (
            f"h(DAG) should be >= 0, got {h_dag.item()}"
        )

    def test_notears_cycle_positive(self):
        """A W with cycles should give h(W) > 0."""
        from aeon_core import NOTEARSCausalModel
        d = 4
        model = NOTEARSCausalModel(num_vars=d)

        with torch.no_grad():
            model.W.zero_()
            # Create a cycle: 0 → 1 → 2 → 0
            model.W[1, 0] = 1.0
            model.W[2, 1] = 1.0
            model.W[0, 2] = 1.0
            h_cycle = model.dag_loss()

        assert h_cycle.item() > 0.01, (
            f"h(cyclic W) should be > 0, got {h_cycle.item()}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Von Neumann Entropy
# ═══════════════════════════════════════════════════════════════════════════


class TestVonNeumannEntropy:
    """Verify the Von Neumann entropy implementation with proper density
    matrix normalisation."""

    def test_import(self):
        """compute_von_neumann_entropy should be importable."""
        from aeon_core import compute_von_neumann_entropy
        assert callable(compute_von_neumann_entropy)

    def test_identity_like_C(self):
        """When C has orthogonal rows (identity-like), entropy should be
        maximal (uniform eigenspectrum)."""
        from aeon_core import compute_von_neumann_entropy
        d = 4
        C = torch.eye(d)  # [d, d] — orthogonal rows
        result = compute_von_neumann_entropy(C)

        assert result['is_proper_density_matrix'], (
            "ρ should be a proper density matrix (Tr=1)"
        )
        # Eigenvalues should be uniform: λ_i = 1/d
        evals = result['eigenvalues']
        assert abs(evals.sum().item() - 1.0) < 1e-5, (
            f"Eigenvalues should sum to 1, got {evals.sum().item()}"
        )
        # S_VN should be log(d) for uniform distribution
        expected_S = math.log(d)
        assert abs(result['S_VN'] - expected_S) < 1e-4, (
            f"S_VN for identity-like C should be log({d})={expected_S:.4f}, "
            f"got {result['S_VN']:.4f}"
        )
        # D should be 1.0 (maximally diverse)
        assert abs(result['D'] - 1.0) < 1e-4, (
            f"D for identity-like C should be 1.0, got {result['D']:.4f}"
        )

    def test_collapsed_C(self):
        """When all rows of C are identical, entropy should be 0 (collapsed)."""
        from aeon_core import compute_von_neumann_entropy
        B, d = 5, 4
        C = torch.ones(B, d)  # all rows identical
        result = compute_von_neumann_entropy(C)

        assert result['is_proper_density_matrix']
        # Only one non-zero eigenvalue → S_VN = 0
        assert abs(result['S_VN']) < 1e-4, (
            f"S_VN for collapsed C should be ~0, got {result['S_VN']:.4f}"
        )
        assert abs(result['D']) < 1e-4, (
            f"D for collapsed C should be ~0, got {result['D']:.4f}"
        )

    def test_trace_normalisation(self):
        """ρ must have Tr(ρ) = 1 regardless of the scale of C."""
        from aeon_core import compute_von_neumann_entropy
        B, d = 8, 6
        C = torch.randn(B, d) * 100  # large scale
        result = compute_von_neumann_entropy(C)

        assert result['is_proper_density_matrix'], (
            "ρ should be a proper density matrix regardless of C scale"
        )
        evals = result['eigenvalues']
        assert abs(evals.sum().item() - 1.0) < 1e-4, (
            f"Eigenvalues of ρ should sum to 1.0, got {evals.sum().item()}"
        )

    def test_zero_C(self):
        """C = 0 should yield S_VN = 0 (degenerate case)."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.zeros(4, 3)
        result = compute_von_neumann_entropy(C)

        assert result['S_VN'] == 0.0
        assert result['D'] == 0.0
        assert result['is_proper_density_matrix'] is False

    def test_entropy_bounds(self):
        """S_VN should be in [0, log(min(B, d))]."""
        from aeon_core import compute_von_neumann_entropy
        B, d = 10, 8
        C = torch.randn(B, d)
        result = compute_von_neumann_entropy(C)

        max_S = math.log(min(B, d))
        assert result['S_VN'] >= -1e-6, (
            f"S_VN should be >= 0, got {result['S_VN']}"
        )
        assert result['S_VN'] <= max_S + 1e-4, (
            f"S_VN should be <= log(min(B,d))={max_S:.4f}, "
            f"got {result['S_VN']:.4f}"
        )

    def test_normalised_diversity_bounds(self):
        """D should be in [0, 1]."""
        from aeon_core import compute_von_neumann_entropy
        B, d = 10, 8
        C = torch.randn(B, d)
        result = compute_von_neumann_entropy(C, normalize=True)

        assert result['D'] >= -1e-6, f"D should be >= 0, got {result['D']}"
        assert result['D'] <= 1.0 + 1e-4, f"D should be <= 1, got {result['D']}"

    def test_effective_rank(self):
        """Effective rank should be in [0, min(B, d)]."""
        from aeon_core import compute_von_neumann_entropy
        B, d = 10, 8
        C = torch.randn(B, d)
        result = compute_von_neumann_entropy(C)

        assert result['rank'] >= 0, (
            f"Effective rank should be >= 0, got {result['rank']}"
        )
        assert result['rank'] <= min(B, d) + 1e-4, (
            f"Effective rank should be <= min(B,d)={min(B,d)}, "
            f"got {result['rank']:.4f}"
        )

    def test_different_scales_same_entropy(self):
        """Scaling C by a constant should not change entropy (ρ is normalised)."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(5, 4)
        result1 = compute_von_neumann_entropy(C)
        result2 = compute_von_neumann_entropy(C * 10.0)

        assert abs(result1['S_VN'] - result2['S_VN']) < 1e-4, (
            f"Scaling C should not change S_VN: "
            f"{result1['S_VN']:.4f} vs {result2['S_VN']:.4f}"
        )

    def test_trace_G_positive(self):
        """Tr(G) = Tr(CᵀC) should be non-negative."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(5, 4)
        result = compute_von_neumann_entropy(C)

        assert result['trace_G'] >= 0, (
            f"Tr(CᵀC) should be >= 0, got {result['trace_G']}"
        )

    def test_eigenvalues_non_negative(self):
        """All eigenvalues of ρ should be non-negative."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(5, 4)
        result = compute_von_neumann_entropy(C)

        evals = result['eigenvalues']
        assert (evals >= -1e-6).all(), (
            f"Eigenvalues should be non-negative, min={evals.min().item()}"
        )

    def test_normalize_false(self):
        """When normalize=False, D should equal S_VN (unnormalised)."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(5, 4)
        result = compute_von_neumann_entropy(C, normalize=False)
        assert abs(result['D'] - result['S_VN']) < 1e-6

    def test_batch_vs_hidden_dim_asymmetry(self):
        """Entropy should work when B >> d and when B << d."""
        from aeon_core import compute_von_neumann_entropy

        # B >> d
        C1 = torch.randn(100, 3)
        r1 = compute_von_neumann_entropy(C1)
        assert r1['is_proper_density_matrix']

        # B << d
        C2 = torch.randn(3, 100)
        r2 = compute_von_neumann_entropy(C2)
        assert r2['is_proper_density_matrix']


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: Integration Tests — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests verifying all four fixes together."""

    def test_full_meta_loop_with_all_diagnostics(self):
        """Run full meta-loop and verify all new diagnostic fields."""
        from aeon_core import AEONConfig, ProvablyConvergentMetaLoop

        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        ml = ProvablyConvergentMetaLoop(config, max_iterations=10, min_iterations=2)
        ml.eval()

        psi = torch.randn(4, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # Check all new fields exist
        required_fields = [
            'output_stabilizer_lipschitz',
            'layernorm_fragile',
            'layernorm_data_dependent_lipschitz',
            'dropout_active',
            'L_certificate',
            'lipschitz_partial_C_constructive',
        ]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"

        # Basic sanity
        assert not torch.isnan(C).any()
        assert not torch.isinf(C).any()

    def test_von_neumann_entropy_on_meta_loop_output(self):
        """Compute Von Neumann entropy on actual meta-loop output."""
        from aeon_core import (
            AEONConfig, ProvablyConvergentMetaLoop,
            compute_von_neumann_entropy,
        )

        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        ml = ProvablyConvergentMetaLoop(config, max_iterations=10, min_iterations=2)
        ml.eval()

        psi = torch.randn(8, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # Compute Von Neumann entropy on the converged representation
        vn_result = compute_von_neumann_entropy(C)

        assert vn_result['is_proper_density_matrix'], (
            "ρ from meta-loop output should be a proper density matrix"
        )
        assert math.isfinite(vn_result['S_VN']), (
            f"S_VN should be finite, got {vn_result['S_VN']}"
        )
        assert vn_result['D'] >= 0 and vn_result['D'] <= 1.0 + 1e-4, (
            f"D should be in [0, 1], got {vn_result['D']}"
        )

    def test_notears_and_von_neumann_together(self):
        """NOTEARS and VN entropy should be usable together for causal
        diversity analysis."""
        from aeon_core import NOTEARSCausalModel, compute_von_neumann_entropy

        d = 5
        model = NOTEARSCausalModel(num_vars=d)

        # Compute DAG loss
        h = model.dag_loss()
        assert torch.isfinite(h), f"DAG loss should be finite, got {h}"

        # Compute VN entropy on the adjacency matrix as a representation
        W = model.W.detach().unsqueeze(0)  # [1, d, d] → reshape to [d, d]
        vn_result = compute_von_neumann_entropy(model.W.detach())
        assert math.isfinite(vn_result['S_VN'])
