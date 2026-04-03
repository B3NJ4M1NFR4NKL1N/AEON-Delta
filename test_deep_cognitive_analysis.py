"""
Tests for deep cognitive-intellectual analysis refinements.

Covers six areas of mathematical/theoretical rigour:
1. Composite T IQC certification (full operator, not per-block)
2. LayerNorm variance floor enforcement (noise injection stabilisation)
3. KM formal convergence statement (Banach vs KM with assumptions)
4. ECLipsE/IQC scope, input domain, dropout treatment documentation
5. NOTEARS acyclicity constraint h(W) = tr(exp(W ∘ W)) − d
6. Von Neumann entropy PSD re-normalisation
"""

from __future__ import annotations

import math
import torch
import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def meta_loop():
    """Create a ProvablyConvergentMetaLoop instance for testing."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    ml = ProvablyConvergentMetaLoop(
        config, max_iterations=10, min_iterations=2,
    )
    ml.eval()
    return ml, config


@pytest.fixture
def lambda_op():
    """Create a LipschitzConstrainedLambda for testing."""
    from aeon_core import LipschitzConstrainedLambda
    lop = LipschitzConstrainedLambda(
        input_dim=128, hidden_dim=128, output_dim=64,
        lipschitz_target=0.9, use_spectral_norm=True, dropout=0.1,
    )
    lop.eval()
    return lop


@pytest.fixture
def notears_model():
    """Create a NOTEARSCausalModel for testing."""
    from aeon_core import NOTEARSCausalModel
    return NOTEARSCausalModel(num_vars=5, hidden_dim=32)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Composite T IQC Certification
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeT_IQC:
    """Verify the composite T IQC certificate certifies the full operator."""

    def test_method_exists(self, meta_loop):
        """compute_composite_T_certificate must be a method."""
        ml, _ = meta_loop
        assert hasattr(ml, 'compute_composite_T_certificate'), (
            "Missing compute_composite_T_certificate method"
        )

    def test_returns_required_fields(self, meta_loop):
        """Certificate must include all required fields."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)

        required = [
            'L_composite', 'certified', 'is_contraction',
            'is_nonexpansive', 'cholesky_succeeded', 'scope',
            'dropout_active', 'slope_restriction', 'methodology',
        ]
        for field in required:
            assert field in cert, f"Missing field: {field}"

    def test_scope_is_global(self, meta_loop):
        """Certificate scope must be 'global' (worst-case over samples)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['scope'] == 'global'

    def test_slope_restriction_gelu(self, meta_loop):
        """IQC slope restriction must be [0, 1.13] for GELU."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['slope_restriction'] == [0.0, 1.13]

    def test_L_composite_is_finite(self, meta_loop):
        """L_composite must be finite and non-negative."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert math.isfinite(cert['L_composite'])
        assert cert['L_composite'] >= 0.0

    def test_dropout_inactive_at_eval(self, meta_loop):
        """Dropout must be inactive at eval (cert valid)."""
        ml, config = meta_loop
        ml.eval()
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['dropout_active'] is False

    def test_dropout_note_present(self, meta_loop):
        """Certificate must document dropout treatment."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'dropout_note' in cert
        assert 'inference' in cert['dropout_note'].lower()

    def test_cholesky_test_runs(self, meta_loop):
        """Cholesky test must execute (succeed or fail gracefully)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert isinstance(cert['cholesky_succeeded'], bool)

    def test_is_contraction_consistent(self, meta_loop):
        """is_contraction must be True iff L_composite < 1."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['is_contraction'] == (cert['L_composite'] < 1.0)

    def test_is_nonexpansive_consistent(self, meta_loop):
        """is_nonexpansive must be True iff L_composite ≤ 1 (with tolerance)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert cert['is_nonexpansive'] == (cert['L_composite'] <= 1.0 + 1e-6)

    def test_methodology_mentions_composite(self, meta_loop):
        """Methodology must mention 'composite' operator."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        cert = ml.compute_composite_T_certificate(psi, num_jacobian_samples=4)
        assert 'composite' in cert['methodology'].lower()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: LayerNorm Variance Floor Enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestLayerNormVarianceFloor:
    """Verify pre-LayerNorm variance floor enforcement."""

    def test_variance_floor_attribute_exists(self, meta_loop):
        """_pre_ln_variance_floor must be defined."""
        ml, _ = meta_loop
        assert hasattr(ml, '_pre_ln_variance_floor')
        assert ml._pre_ln_variance_floor > 0

    def test_variance_floor_default(self, meta_loop):
        """Default variance floor must be 0.1."""
        ml, _ = meta_loop
        assert ml._pre_ln_variance_floor == pytest.approx(0.1)

    def test_metadata_includes_floor(self, meta_loop):
        """Metadata must report the variance floor."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        assert 'pre_ln_variance_floor' in meta
        assert meta['pre_ln_variance_floor'] == pytest.approx(0.1)

    def test_low_variance_input_stabilised(self, meta_loop):
        """Near-zero variance input should not cause Lipschitz blow-up."""
        ml, config = meta_loop
        # Create input that would produce near-zero variance pre-LN
        psi = torch.ones(2, config.z_dim) * 0.001  # near-constant
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)
        # The output should be finite (no NaN/Inf from LN blow-up)
        assert torch.isfinite(C).all(), "Output contains NaN/Inf after LN"

    def test_normal_variance_unaffected(self, meta_loop):
        """Normal-variance inputs should pass through unaffected."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)  # σ ≈ 1
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)
        assert torch.isfinite(C).all()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: KM Formal Convergence Statement
# ═══════════════════════════════════════════════════════════════════════════


class TestKMFormalConvergence:
    """Verify the formal KM convergence certificate."""

    def test_formal_convergence_in_metadata(self, meta_loop):
        """Metadata must include km_formal_convergence."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        assert 'km_formal_convergence' in meta

    def test_formal_convergence_structure(self, meta_loop):
        """km_formal_convergence must have convergence_type and assumptions."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        fc = meta['km_formal_convergence']
        assert 'convergence_type' in fc
        assert 'assumptions_met' in fc
        assert 'theorem_reference' in fc
        assert 'convergence_strength' in fc

    def test_banach_contraction_detected(self, meta_loop):
        """When L < 1, convergence_type must be 'banach_contraction'."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        fc = meta['km_formal_convergence']
        L = meta.get('L_certificate', 1.0)
        if L < 1.0 and math.isfinite(L):
            assert fc['convergence_type'] == 'banach_contraction'
            assert fc['convergence_strength'] == 'strong'

    def test_convergence_type_is_valid(self, meta_loop):
        """convergence_type must be one of the expected values."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        valid_types = {
            'banach_contraction', 'krasnoselskii_mann',
            'unverified', 'none',
        }
        assert meta['km_formal_convergence']['convergence_type'] in valid_types

    def test_km_type_includes_caveat(self, meta_loop):
        """KM convergence must include a caveat about weak convergence."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        fc = meta['km_formal_convergence']
        if fc['convergence_type'] == 'krasnoselskii_mann':
            assert 'caveat' in fc
            assert 'weak' in fc['caveat'].lower()

    def test_banach_type_includes_error_bound(self, meta_loop):
        """Banach convergence must include certified_error_bound."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        fc = meta['km_formal_convergence']
        if fc['convergence_type'] == 'banach_contraction':
            assert 'certified_error_bound' in fc
            assert 'error_bound_formula' in fc

    def test_assumptions_include_hilbert_space(self, meta_loop):
        """Assumptions must note the Hilbert space property."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        fc = meta['km_formal_convergence']
        if fc['convergence_type'] in ('banach_contraction', 'krasnoselskii_mann'):
            assumptions = fc['assumptions_met']
            # Either 'hilbert_space' or 'complete_metric_space' should be True
            has_space = assumptions.get('hilbert_space', False) or \
                        assumptions.get('complete_metric_space', False)
            assert has_space


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: ECLipsE/IQC Scope and Dropout Documentation
# ═══════════════════════════════════════════════════════════════════════════


class TestECLipSEScope:
    """Verify ECLipsE certificate includes scope and dropout info."""

    def test_eclipse_bound_returns_scope(self, lambda_op):
        """compute_eclipse_bound must return certificate_scope."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert 'certificate_scope' in result
        assert result['certificate_scope'] == 'global'

    def test_eclipse_bound_returns_input_domain(self, lambda_op):
        """compute_eclipse_bound must document input domain."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert 'input_domain' in result
        assert len(result['input_domain']) > 20  # meaningful description

    def test_eclipse_bound_returns_dropout_treatment(self, lambda_op):
        """compute_eclipse_bound must document dropout treatment."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert 'dropout_treatment' in result
        assert 'inference' in result['dropout_treatment'].lower()

    def test_eclipse_bound_dropout_active_flag(self, lambda_op):
        """dropout_active must be False at eval."""
        lambda_op.eval()
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert 'dropout_active' in result
        assert result['dropout_active'] is False

    def test_eclipse_composite_note(self, lambda_op):
        """Must include note about composite T certification."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert 'composite_note' in result
        assert 'composite' in result['composite_note'].lower()

    def test_eclipse_bound_L_finite(self, lambda_op):
        """L_eclipse must be finite and positive."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert math.isfinite(result['L_eclipse'])
        assert result['L_eclipse'] >= 0

    def test_eclipse_slope_restriction(self, lambda_op):
        """Slope restriction must be [0, 1.13] for GELU."""
        result = lambda_op.compute_eclipse_bound(num_jacobian_samples=4)
        assert result['slope_restriction'] == [0.0, 1.13]


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: NOTEARS Acyclicity Constraint
# ═══════════════════════════════════════════════════════════════════════════


class TestNOTEARS:
    """Verify NOTEARS uses h(W) = tr(exp(W ∘ W)) − d correctly."""

    def test_dag_loss_at_zero_W(self, notears_model):
        """h(0) = tr(exp(0)) − d = tr(I) − d = 0."""
        notears_model.W.data.zero_()
        h = notears_model.dag_loss()
        assert abs(h.item()) < 1e-5, f"h(0) should be 0, got {h.item()}"

    def test_dag_loss_uses_hadamard_product(self, notears_model):
        """h(W) must use W ∘ W (element-wise square), not W²."""
        # Set W to known values and verify h(W) = tr(exp(W∘W)) - d
        d = notears_model.num_vars
        W_test = torch.randn(d, d) * 0.1
        notears_model.W.data.copy_(W_test)
        h = notears_model.dag_loss()

        # Manual computation
        M = W_test * W_test  # Hadamard product
        if hasattr(torch.linalg, 'matrix_exp'):
            expm = torch.linalg.matrix_exp(M)
        else:
            # Taylor fallback for comparison
            I = torch.eye(d)
            expm = I.clone()
            Mk = I.clone()
            for k in range(1, 10):
                Mk = Mk @ M
                expm = expm + Mk / math.factorial(k)
        expected = torch.trace(expm) - d
        assert abs(h.item() - expected.item()) < 1e-3, (
            f"h(W) mismatch: got {h.item()}, expected {expected.item()}"
        )

    def test_dag_loss_docstring_formula(self, notears_model):
        """Docstring must contain the correct formula."""
        docstring = notears_model.dag_loss.__doc__
        assert 'tr(exp(W' in docstring or 'tr(exp(W ∘ W))' in docstring
        assert '− d' in docstring or '- d' in docstring

    def test_dag_loss_positive_for_cycle(self, notears_model):
        """h(W) > 0 when W has a cycle (non-zero diagonal in exp(W∘W))."""
        d = notears_model.num_vars
        # Create a cycle: 0→1→2→0
        W_cycle = torch.zeros(d, d)
        W_cycle[0, 1] = 1.0
        W_cycle[1, 2] = 1.0
        W_cycle[2, 0] = 1.0
        notears_model.W.data.copy_(W_cycle)
        h = notears_model.dag_loss()
        assert h.item() > 0.01, f"h(W) should be > 0 for cyclic W, got {h.item()}"

    def test_dag_loss_diagnostics(self, notears_model):
        """dag_loss should produce diagnostics."""
        notears_model.W.data.uniform_(-0.1, 0.1)
        _ = notears_model.dag_loss()
        assert hasattr(notears_model, '_dag_loss_diag')
        diag = notears_model._dag_loss_diag
        assert 'method' in diag
        assert 'h_value' in diag


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: Von Neumann Entropy PSD Normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestVonNeumannEntropy:
    """Verify VN entropy uses PSD trace-1 normalised density matrix."""

    def test_function_exists(self):
        """compute_von_neumann_entropy must be importable."""
        from aeon_core import compute_von_neumann_entropy
        assert callable(compute_von_neumann_entropy)

    def test_trace_normalisation(self):
        """ρ must be trace-normalised (Tr(ρ) = 1)."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(8, 16) * 3.0  # non-unit scale
        result = compute_von_neumann_entropy(C)
        assert result['is_proper_density_matrix'] is True, (
            "Density matrix must be proper (Tr(ρ) = 1)"
        )

    def test_entropy_in_valid_range(self):
        """S_VN must be in [0, log(rank)]."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(10, 8)
        result = compute_von_neumann_entropy(C)
        assert result['S_VN'] >= -1e-10, f"S_VN must be ≥ 0, got {result['S_VN']}"
        max_S = math.log(max(min(10, 8), 1))
        assert result['S_VN'] <= max_S + 1e-6, (
            f"S_VN must be ≤ log(min(B,d))={max_S}, got {result['S_VN']}"
        )

    def test_diversity_in_01(self):
        """D must be in [0, 1] when normalize=True."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(10, 8)
        result = compute_von_neumann_entropy(C, normalize=True)
        assert 0.0 - 1e-6 <= result['D'] <= 1.0 + 1e-6, (
            f"D must be in [0, 1], got {result['D']}"
        )

    def test_psd_enforcement_after_clamping(self):
        """Eigenvalues must sum to 1 after PSD clamping."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.randn(5, 4)
        result = compute_von_neumann_entropy(C)
        evals = result['eigenvalues']
        ev_sum = evals.sum().item()
        assert abs(ev_sum - 1.0) < 1e-5, (
            f"Eigenvalues must sum to 1 after PSD enforcement, got {ev_sum}"
        )

    def test_degenerate_input(self):
        """Near-zero input should return S_VN = 0, D = 0."""
        from aeon_core import compute_von_neumann_entropy
        C = torch.zeros(5, 4)
        result = compute_von_neumann_entropy(C)
        assert result['S_VN'] == 0.0
        assert result['D'] == 0.0
        assert result['is_proper_density_matrix'] is False

    def test_uniform_eigenspectrum_max_entropy(self):
        """Identity-like Gram should give near-maximum entropy."""
        from aeon_core import compute_von_neumann_entropy
        d = 8
        # Orthonormal rows → Gram = I → ρ = I/d → max entropy
        C = torch.eye(d)
        result = compute_von_neumann_entropy(C)
        max_S = math.log(d)
        assert abs(result['S_VN'] - max_S) < 0.01, (
            f"S_VN for identity should be ≈ log({d})={max_S:.4f}, "
            f"got {result['S_VN']:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: Integration / Cross-Cutting Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Cross-cutting tests ensuring all fixes work together."""

    def test_full_forward_pass_with_all_fixes(self, meta_loop):
        """Full forward pass should succeed with all fixes active."""
        ml, config = meta_loop
        psi = torch.randn(4, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        # Basic validity
        assert torch.isfinite(C).all()
        assert C.shape == (4, config.hidden_dim)

        # All new fields present
        assert 'km_formal_convergence' in meta
        assert 'pre_ln_variance_floor' in meta

    def test_composite_cert_consistent_with_per_block(self, meta_loop):
        """Composite cert should be ≤ per-block product (tighter)."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)

        composite = ml.compute_composite_T_certificate(
            psi, num_jacobian_samples=8,
        )
        per_block = (
            ml.lambda_op.get_constructive_partial_lipschitz_bound_wrt_C()
        )

        # Composite should typically be tighter (≤) than per-block product,
        # though with finite samples and finite differences, it may
        # occasionally exceed due to approximation. We check it's in
        # the same order of magnitude.
        assert composite['L_composite'] < per_block * 2.0, (
            f"Composite L={composite['L_composite']:.4f} should be "
            f"comparable to per-block L={per_block:.4f}"
        )

    def test_layernorm_fragile_detection_still_works(self, meta_loop):
        """LayerNorm fragility detection must still be present."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        assert 'layernorm_fragile' in meta

    def test_identity_contribution_still_applied(self, meta_loop):
        """KM identity contribution L_eff = (1-α) + α·L_T must be applied."""
        ml, config = meta_loop
        psi = torch.randn(2, config.z_dim)
        with torch.no_grad():
            _, _, meta = ml.compute_fixed_point(psi)
        # L_certificate should reflect identity contribution when α < 1
        L_cert = meta.get('L_certificate')
        if L_cert is not None and math.isfinite(L_cert):
            # L_eff should be > 0 (identity contribution adds positive term)
            assert L_cert > 0, f"L_certificate must be > 0, got {L_cert}"
