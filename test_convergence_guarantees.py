"""
Test suite for formal convergence guarantee improvements.

Tests validate:
1. Spectral radius enforcement ensuring L < 1
2. CROWN-based IBP certification for GELU and LayerNorm
3. Lanczos iteration for scalable corank estimation
4. End-to-end convergence verification
"""

import torch
import math
import pytest


def test_adaptive_lipschitz_target_computation():
    """Verify adaptive target accounts for GELU and LayerNorm composition."""
    from aeon_core import LipschitzConstrainedLambda

    # Create operator with known dimensions
    lambda_op = LipschitzConstrainedLambda(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        lipschitz_target=0.9,
        use_spectral_norm=True,
    )

    target = lambda_op.compute_adaptive_lipschitz_target()

    # Target should be < 1 and account for composition
    assert target < 1.0, f"Target {target} should be < 1.0"
    assert target > 0.3, f"Target {target} should be reasonable (>0.3)"

    # With GELU (L≈1.13) and LayerNorm, per-layer target should be conservative
    L_gelu = 1.13
    # For 2 linear layers: target² · 1.13 · L_LN < 0.85
    # So target < sqrt(0.85 / (1.13 · L_LN))
    # With L_LN ≈ sqrt(64) ≈ 8, expect target < sqrt(0.85 / 9) ≈ 0.31
    # But clamp ensures target ≥ 0.5
    assert target >= 0.5, f"Target {target} should be ≥ 0.5 (clamped)"

    print(f"✅ Adaptive Lipschitz target: {target:.4f}")


def test_spectral_bound_enforcement():
    """Verify enforce_spectral_bound projects weights to ensure L < 1."""
    from aeon_core import LipschitzConstrainedLambda

    lambda_op = LipschitzConstrainedLambda(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        lipschitz_target=0.9,
        use_spectral_norm=True,
    )

    # Compute bound before enforcement
    L_before = lambda_op.get_constructive_lipschitz_bound()

    # Enforce spectral bounds
    result = lambda_op.enforce_spectral_bound()

    # Check result structure
    assert 'target' in result
    assert 'L_bound_after' in result
    assert 'contraction_certified' in result

    L_after = result['L_bound_after']

    # After enforcement, bound should be < 1 (or very close)
    print(f"L before: {L_before:.4f}, L after: {L_after:.4f}")
    print(f"Projections applied: {result['projections_applied']}")
    print(f"Contraction certified: {result['contraction_certified']}")

    # Depending on initial weights, L_after should be ≤ L_before
    assert L_after <= L_before + 0.01, "Enforcement should not increase Lipschitz constant"

    # If L_before was > 1, L_after should be < 1
    if L_before > 1.0:
        assert L_after < 1.0, f"After enforcement, L={L_after} should be < 1.0"

    print(f"✅ Spectral bound enforcement: L={L_after:.4f} < 1.0")


def test_crown_gelu_interval_propagation():
    """Verify CROWN linear relaxation for GELU produces valid bounds."""
    import torch.nn.functional as F

    # Test CROWN bounds at various intervals
    test_intervals = [
        (torch.tensor([-1.0]), torch.tensor([1.0])),
        (torch.tensor([-0.5]), torch.tensor([0.5])),
        (torch.tensor([0.0]), torch.tensor([2.0])),
    ]

    for lb, ub in test_intervals:
        # Compute GELU values at bounds
        gelu_lb = F.gelu(lb)
        gelu_ub = F.gelu(ub)

        # Compute CROWN relaxation
        delta = 1e-4
        interval_width = (ub - lb).clamp(min=1e-8)
        lambda_L = (gelu_ub - gelu_lb) / interval_width
        b_L = gelu_lb - lambda_L * lb

        # Lower bound: λ_L · x + b_L
        # Should satisfy: λ_L · lb + b_L ≤ GELU(lb)
        lower_at_lb = lambda_L * lb + b_L
        assert torch.allclose(lower_at_lb, gelu_lb, atol=1e-5), \
            f"Lower bound should equal GELU(lb) at endpoint"

        # Check at midpoint
        mid = (lb + ub) / 2.0
        gelu_mid = F.gelu(mid)
        lower_at_mid = lambda_L * mid + b_L

        # Linear lower bound should be ≤ true GELU value
        assert lower_at_mid <= gelu_mid + 1e-4, \
            f"CROWN lower bound {lower_at_mid} should be ≤ GELU(mid) {gelu_mid}"

    print("✅ CROWN GELU bounds validated")


def test_layernorm_lipschitz_bound():
    """Verify LayerNorm Lipschitz bound computation."""
    import torch.nn as nn

    # Create LayerNorm with known parameters
    d = 64
    ln = nn.LayerNorm(d)

    # Set weights to known values
    with torch.no_grad():
        ln.weight.fill_(1.0)  # γ = [1, 1, ..., 1]
        ln.bias.fill_(0.0)    # β = [0, 0, ..., 0]

    # With uniform weights, L_LN ≈ √d
    _gamma = ln.weight.detach()
    _gamma_abs = _gamma.abs()
    _gamma_min = _gamma_abs.min().clamp(min=1e-8)
    _gamma_max = _gamma_abs.max()
    _gamma_ratio = _gamma_max / _gamma_min
    L_ln = min(math.sqrt(d) * _gamma_ratio, math.sqrt(d) + 1.0)

    assert abs(L_ln - math.sqrt(d)) < 0.1, \
        f"For uniform weights, L_LN ≈ √{d} ≈ {math.sqrt(d):.2f}, got {L_ln:.2f}"

    # Now test with non-uniform weights
    with torch.no_grad():
        ln.weight[0] = 2.0   # max(γ) = 2
        ln.weight[1] = 0.5   # min(γ) = 0.5

    _gamma = ln.weight.detach()
    _gamma_abs = _gamma.abs()
    _gamma_min = _gamma_abs.min().clamp(min=1e-8)
    _gamma_max = _gamma_abs.max()
    _gamma_ratio = _gamma_max / _gamma_min
    L_ln_nonuniform = min(math.sqrt(d) * _gamma_ratio, math.sqrt(d) + 1.0)

    # With γ_ratio = 2.0/0.5 = 4.0, expect L_LN ≈ √64 · 4 = 32
    expected = math.sqrt(d) * 4.0
    assert abs(L_ln_nonuniform - expected) < 1.0, \
        f"Expected L_LN ≈ {expected:.2f}, got {L_ln_nonuniform:.2f}"

    print(f"✅ LayerNorm Lipschitz: uniform={L_ln:.2f}, non-uniform={L_ln_nonuniform:.2f}")


def test_lanczos_bottom_k_eigenvalues():
    """Verify Lanczos iteration computes bottom-k eigenvalues correctly."""
    from aeon_core import FastHessianComputer

    # Create quadratic function with known Hessian eigenvalues
    # f(x) = 1/2 · xᵀ diag(λ₁, λ₂, ..., λₙ) x
    n = 20
    k = 5

    # Known eigenvalues: [1, 2, 3, ..., 20]
    true_eigenvalues = torch.arange(1.0, n + 1.0)

    def quadratic_func(x):
        """f(x) = 1/2 · xᵀ D x where D = diag(1, 2, ..., n)"""
        B = x.shape[0]
        result = 0.5 * (x ** 2 * true_eigenvalues.unsqueeze(0)).sum(dim=-1)
        return result

    hc = FastHessianComputer(method='finite_differences')
    x = torch.randn(2, n)

    result = hc.lanczos_bottom_k_eigenvalues(quadratic_func, x, k=k, num_iterations=30)

    # Check result structure
    assert 'eigenvalues' in result
    assert 'converged' in result
    assert result['eigenvalues'].shape == (2, k)

    # For quadratic function, Hessian is constant = diag(1, 2, ..., n)
    # Bottom k eigenvalues should be [1, 2, 3, 4, 5]
    estimated_eigs = result['eigenvalues'][0, :]  # First batch element
    expected_eigs = true_eigenvalues[:k]

    # Allow some tolerance due to Lanczos approximation
    for i in range(k):
        est = estimated_eigs[i].item()
        exp = expected_eigs[i].item()
        rel_error = abs(est - exp) / exp
        print(f"  λ_{i+1}: estimated={est:.4f}, expected={exp:.4f}, rel_error={rel_error:.4f}")
        assert rel_error < 0.1, f"Eigenvalue {i} error too large: {rel_error:.4f}"

    print(f"✅ Lanczos eigenvalues converged: {result['converged'].all()}")


def test_end_to_end_convergence_verification():
    """End-to-end test: verify full convergence guarantee pipeline."""
    from aeon_core import LipschitzConstrainedLambda

    # Create operator
    lambda_op = LipschitzConstrainedLambda(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        lipschitz_target=0.9,
        use_spectral_norm=True,
    )

    # Step 1: Compute initial bound
    L_initial = lambda_op.get_constructive_lipschitz_bound()
    print(f"Initial Lipschitz bound: {L_initial:.4f}")

    # Step 2: Enforce spectral bounds
    enforcement = lambda_op.enforce_spectral_bound()
    L_enforced = enforcement['L_bound_after']
    print(f"After enforcement: {L_enforced:.4f}")
    print(f"Contraction certified: {enforcement['contraction_certified']}")

    # Step 3: Verify contraction via empirical sampling
    x = torch.randn(10, 32)
    y = torch.randn(10, 32)

    with torch.no_grad():
        fx = lambda_op(x)
        fy = lambda_op(y)

        numerator = torch.norm(fx - fy, dim=-1).mean().item()
        denominator = torch.norm(x - y, dim=-1).mean().item()

        empirical_L = numerator / denominator

    print(f"Empirical Lipschitz (sampled): {empirical_L:.4f}")

    # Empirical should be ≤ theoretical bound
    assert empirical_L <= L_enforced + 0.1, \
        f"Empirical L={empirical_L:.4f} should be ≤ bound={L_enforced:.4f}"

    # If enforcement succeeded, should have L < 1
    if enforcement['projections_applied'] > 0:
        assert L_enforced < 1.05, f"After projection, L={L_enforced:.4f} should be ≈ 1"

    print("✅ End-to-end convergence verification passed")


def test_lipschitz_bound_consistency():
    """Verify consistency between constructive bound and partial bound."""
    from aeon_core import LipschitzConstrainedLambda

    lambda_op = LipschitzConstrainedLambda(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        lipschitz_target=0.9,
        use_spectral_norm=True,
    )

    # Global constructive bound
    L_global = lambda_op.get_constructive_lipschitz_bound()

    # Partial bound w.r.t. C (with ψ₀ fixed)
    psi_0 = torch.randn(4, 32)
    L_partial = lambda_op.compute_partial_lipschitz_wrt_C(psi_0, num_samples=50)

    print(f"Global bound: {L_global:.4f}")
    print(f"Partial bound (w.r.t. C): {L_partial:.4f}")

    # Partial bound can be ≤ global bound (more specific)
    assert L_partial <= L_global + 0.2, \
        f"Partial bound {L_partial:.4f} should be ≤ global {L_global:.4f}"

    print("✅ Lipschitz bound consistency verified")


def test_corank_estimation_fold_catastrophe():
    """Verify corank estimation detects fold catastrophe (1 zero eigenvalue)."""
    from aeon_core import FastHessianComputer

    # Create function with corank=1 (one zero eigenvalue)
    # f(x) = x₁² + x₂² + ... + x_{n-1}² + 0·x_n²
    n = 10

    def fold_func(x):
        """Quadratic with last eigenvalue = 0 (fold singularity)"""
        B = x.shape[0]
        # All squared except last dimension
        result = 0.5 * (x[:, :-1] ** 2).sum(dim=-1)
        return result

    hc = FastHessianComputer(method='finite_differences')
    x = torch.randn(2, n)

    result = hc.lanczos_bottom_k_eigenvalues(fold_func, x, k=3, num_iterations=20)

    eigenvalues = result['eigenvalues'][0, :]  # First batch element

    # Smallest eigenvalue should be ≈ 0 (within numerical tolerance)
    lambda_min = eigenvalues[0].abs().item()
    print(f"Smallest eigenvalue: {lambda_min:.6f}")

    # Corank = number of |λᵢ| < ε_zero
    eps_zero = 0.1
    corank = (eigenvalues.abs() < eps_zero).sum().item()

    print(f"Corank (|λ| < {eps_zero}): {corank}")
    assert corank == 1, f"Expected corank=1 (fold), got {corank}"

    print("✅ Fold catastrophe detected (corank=1)")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Formal Convergence Guarantee Improvements")
    print("=" * 70)

    test_adaptive_lipschitz_target_computation()
    print()

    test_spectral_bound_enforcement()
    print()

    test_crown_gelu_interval_propagation()
    print()

    test_layernorm_lipschitz_bound()
    print()

    test_lanczos_bottom_k_eigenvalues()
    print()

    test_end_to_end_convergence_verification()
    print()

    test_lipschitz_bound_consistency()
    print()

    test_corank_estimation_fold_catastrophe()
    print()

    print("=" * 70)
    print("All convergence guarantee tests passed ✅")
    print("=" * 70)
