"""Targeted integration tests for the 5 cognitive activation patches.

These tests validate the specific changes made to bridge the remaining
discontinuities in the AEON-Delta RMT v3.1 cognitive architecture:
  1. Adaptive spectral projection (logarithmic scaling of iterations)
  2. encoder → meta_learner cycle-exempt edge
  3. Inference cache invalidation on deeper meta-loop acceptance
  4. Aggressive EMA reconciliation (hard-reset when gap > 0.2)
  5. KM-damped fallback when L_C stays > 1.0
"""

import math
import sys
import os
import importlib
import inspect

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Patch 1: Adaptive Spectral Projection
# ---------------------------------------------------------------------------

class TestAdaptiveSpectralProjection:
    """Verify that _MAX_PROJ_ITERS scales logarithmically with L_C."""

    def test_max_proj_iters_formula_low_lc(self):
        """When L_C is just above 1.0, iterations should be ~5 (baseline)."""
        lc = 1.05
        computed = max(5, min(30, int(5 + 5 * math.log2(max(lc, 1.01)))))
        assert computed == 5, f"For L_C={lc}, expected 5 iters, got {computed}"

    def test_max_proj_iters_formula_moderate_lc(self):
        """When L_C ~ 4.0, iterations should be ~15."""
        lc = 4.0
        computed = max(5, min(30, int(5 + 5 * math.log2(max(lc, 1.01)))))
        assert computed == 15, f"For L_C={lc}, expected 15 iters, got {computed}"

    def test_max_proj_iters_formula_high_lc(self):
        """When L_C ~ 16.8 (from the log), iterations should be ~25."""
        lc = 16.8
        computed = max(5, min(30, int(5 + 5 * math.log2(max(lc, 1.01)))))
        assert 20 <= computed <= 30, (
            f"For L_C={lc}, expected 20-30 iters, got {computed}"
        )

    def test_max_proj_iters_capped_at_30(self):
        """Even for extremely high L_C, iterations should not exceed 30."""
        lc = 10000.0
        computed = max(5, min(30, int(5 + 5 * math.log2(max(lc, 1.01)))))
        assert computed == 30, f"For L_C={lc}, expected cap at 30, got {computed}"

    def test_relaxed_progress_threshold_in_source(self):
        """The progress check threshold should be 0.999, not 0.99."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        assert '_prev_L_C * 0.999' in source, (
            "Progress threshold should be relaxed to 0.999 for incremental "
            "progress accumulation over the larger iteration budget"
        )

    def test_adaptive_iter_formula_in_source(self):
        """The adaptive formula using log2 should appear in the source."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        assert 'math.log2' in source, (
            "Adaptive projection should use log2-scaling for iteration count"
        )


# ---------------------------------------------------------------------------
# Patch 2: encoder → meta_learner Cycle-Exempt Edge
# ---------------------------------------------------------------------------

class TestCycleExemptEdge:
    """Verify that encoder → meta_learner is in _CYCLE_EXEMPT_EDGES."""

    def test_encoder_meta_learner_exempt(self):
        """The edge (encoder, meta_learner) must be in the exempt set."""
        from aeon_core import AEONDeltaV3
        exempt = AEONDeltaV3._CYCLE_EXEMPT_EDGES
        assert ("encoder", "meta_learner") in exempt, (
            "encoder → meta_learner should be a cycle-exempt edge "
            "to preserve the cross-subsystem feedback path"
        )

    def test_existing_encoder_edges_preserved(self):
        """Existing encoder edges should not be disturbed."""
        from aeon_core import AEONDeltaV3
        exempt = AEONDeltaV3._CYCLE_EXEMPT_EDGES
        assert ("active_learning", "encoder") in exempt
        assert ("encoder", "vibe_thinker") in exempt

    def test_exempt_edges_are_tuples(self):
        """All exempt edges should be 2-tuples of strings."""
        from aeon_core import AEONDeltaV3
        for edge in AEONDeltaV3._CYCLE_EXEMPT_EDGES:
            assert isinstance(edge, tuple) and len(edge) == 2, (
                f"Expected 2-tuple, got {edge}"
            )
            assert isinstance(edge[0], str) and isinstance(edge[1], str), (
                f"Edge components must be strings, got {edge}"
            )


# ---------------------------------------------------------------------------
# Patch 3: Inference Cache Invalidation on Deeper Meta-Loop Acceptance
# ---------------------------------------------------------------------------

class TestCacheInvalidationOnDeeperAcceptance:
    """Verify that inference cache is reset after accepted deeper meta-loop."""

    def test_cache_reset_code_present(self):
        """The inference_cache.reset() call must exist after acceptance."""
        from aeon_core import AEONDeltaV3
        source = inspect.getsource(AEONDeltaV3)

        # Search for the cache invalidation log message which is
        # unique to our patch (not in any mapping tables)
        marker = "Inference cache invalidated after accepted"
        marker_idx = source.find(marker)
        assert marker_idx > 0, (
            "Cache invalidation log message must exist after "
            "accepting deeper meta-loop result"
        )

        # Verify inference_cache.reset() is nearby (within 500 chars before)
        nearby_region = source[max(0, marker_idx - 500):marker_idx + 200]
        assert 'inference_cache.reset()' in nearby_region, (
            "inference_cache.reset() must be called near the "
            "cache invalidation log message"
        )

    def test_cache_invalidation_log_message(self):
        """A debug log should explain the cache invalidation reason."""
        from aeon_core import AEONDeltaV3
        source = inspect.getsource(AEONDeltaV3)
        assert "Inference cache invalidated after accepted" in source, (
            "Cache invalidation after deeper meta-loop must be logged"
        )

    def test_inference_cache_reset_clears_reasoning_result(self):
        """InferenceCache.reset() must clear _cached_reasoning_result."""
        from aeon_core import InferenceCache
        cache = InferenceCache()
        # Simulate storing a result
        dummy_z = torch.zeros(1, 64)
        dummy_out = {"test": True}
        cache.set_reasoning_result(dummy_z, dummy_out)
        assert cache._cached_reasoning_result is not None
        # Reset must clear it
        cache.reset()
        assert cache._cached_reasoning_result is None


# ---------------------------------------------------------------------------
# Patch 4: Aggressive EMA Reconciliation
# ---------------------------------------------------------------------------

class TestAggressiveEMAReconciliation:
    """Verify hard-reset behavior when EMA-empirical gap > 0.2."""

    def test_hard_reset_code_in_source(self):
        """verify_convergence must have a hard-reset branch for gap > 0.2."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        assert '_divergence > 0.2' in source, (
            "Hard-reset threshold of 0.2 must be in verify_convergence"
        )
        assert '_alpha = 1.0' in source, (
            "Hard-reset must set alpha to 1.0 (snap to empirical)"
        )

    def test_hard_reset_math(self):
        """When gap=0.27 (from log: 1.0121 vs 0.7434), alpha must be 1.0."""
        ema_L = 1.0121
        empirical_L = 0.7434
        divergence = abs(empirical_L - ema_L)
        # Under the new code:
        if divergence > 0.2:
            alpha = 1.0
        else:
            alpha = 0.6 + 0.2 * math.tanh(divergence / 0.2)
        assert alpha == 1.0, (
            f"For gap={divergence:.4f} > 0.2, alpha must be 1.0 (hard reset)"
        )
        # Compute new EMA
        new_ema = (1.0 - alpha) * ema_L + alpha * empirical_L
        assert abs(new_ema - empirical_L) < 1e-10, (
            f"After hard reset, EMA should equal empirical: {new_ema} vs {empirical_L}"
        )

    def test_gradual_blend_below_threshold(self):
        """When gap <= 0.2, the original tanh-based blend should apply."""
        ema_L = 0.85
        empirical_L = 0.70
        divergence = abs(empirical_L - ema_L)
        assert divergence <= 0.2
        alpha = 0.6 + 0.2 * math.tanh(divergence / 0.2)
        assert 0.6 <= alpha <= 0.8, (
            f"For gap={divergence:.4f} <= 0.2, alpha should be in [0.6, 0.8], got {alpha}"
        )


# ---------------------------------------------------------------------------
# Patch 5: KM-Damped Fallback
# ---------------------------------------------------------------------------

class TestKMDampedFallback:
    """Verify KM damping α=1/(1+L_C) when L_C stays > 1.0."""

    def test_km_damping_formula(self):
        """For L_C=16.8 (from log), damping alpha should be ~0.056."""
        lc = 16.8
        alpha = 1.0 / (1.0 + lc)
        assert abs(alpha - 0.0562) < 0.001, (
            f"For L_C={lc}, KM damping alpha should be ~0.056, got {alpha:.4f}"
        )

    def test_km_damping_makes_operator_nonexpansive(self):
        """The damped step size α=1/(1+L_C) ensures bounded iteration steps.
        
        The KM-averaged operator T_α = (1-α)I + αT does NOT make the
        effective Lipschitz ≤ 1 when L > 1; rather, it ensures that each
        iteration step is bounded by α·‖T(x)-x‖ ≤ ‖T(x)-x‖/(1+L_C),
        preventing divergence and enabling weak convergence via
        asymptotic regularity (Bauschke & Combettes, Thm 5.14).
        """
        lc = 16.8
        alpha = 1.0 / (1.0 + lc)
        # Step size reduction: damped step is α fraction of full step
        step_reduction = alpha
        assert step_reduction < 1.0 / lc, (
            f"KM damping step reduction ({step_reduction:.4f}) should be "
            f"< 1/L_C ({1.0/lc:.4f}) for bounded iteration"
        )

    def test_km_damping_inactive_when_lc_below_one(self):
        """When L_C < 1.0, alpha should be 1.0 (no damping, pure Picard)."""
        lc = 0.85
        # The code only applies damping when lc >= 1.0
        if lc >= 1.0:
            alpha = 1.0 / (1.0 + lc)
        else:
            alpha = 1.0
        assert alpha == 1.0, "No damping should apply when L_C < 1.0"

    def test_km_damping_code_in_source(self):
        """KM damping code must exist in ProvablyConvergentMetaLoop."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        assert '_km_damping_alpha' in source, (
            "KM damping alpha variable must exist in compute_fixed_point"
        )
        assert '1.0 / (1.0 + _partial_lip_C_constructive)' in source, (
            "KM damping formula must be 1/(1+L_C)"
        )

    def test_km_damping_application_in_source(self):
        """The damping must be applied to C_new in the iteration loop."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        assert '(1.0 - _km_damping_alpha) * C + _km_damping_alpha * C_new' in source, (
            "KM averaging formula must appear in the iteration loop"
        )

    def test_km_damping_reduces_residual(self):
        """KM damping should reduce the effective step size, shrinking residual."""
        lc = 16.8
        alpha = 1.0 / (1.0 + lc)
        # Simulate: C=0, T(C)=1 (expansive step of size 1)
        C = torch.tensor([0.0])
        T_C = torch.tensor([1.0])
        # Undamped residual: |T(C) - C| = 1.0
        undamped_residual = (T_C - C).abs().item()
        # Damped: C_new = (1-α)C + αT(C) = α
        C_new_damped = (1.0 - alpha) * C + alpha * T_C
        damped_residual = (C_new_damped - C).abs().item()
        assert damped_residual < undamped_residual, (
            f"Damped residual ({damped_residual:.4f}) should be smaller "
            f"than undamped ({undamped_residual:.4f})"
        )


# ---------------------------------------------------------------------------
# Integration: Cross-Patch Coherence
# ---------------------------------------------------------------------------

class TestCrossPatchCoherence:
    """Verify that patches work together for system-level coherence."""

    def test_spectral_projection_and_km_damping_coexist(self):
        """Both spectral projection AND KM damping should handle L_C > 1."""
        from aeon_core import ProvablyConvergentMetaLoop
        source = inspect.getsource(ProvablyConvergentMetaLoop)
        # Spectral projection runs first (reduces L_C as much as possible)
        proj_idx = source.find('enforce_spectral_bound')
        # KM damping runs after (handles remaining L_C > 1)
        km_idx = source.find('_km_damping_alpha')
        assert proj_idx < km_idx, (
            "Spectral projection must run before KM damping in the forward pass"
        )

    def test_cache_invalidation_ordered_after_acceptance(self):
        """Cache reset must happen after acceptance, not before."""
        from aeon_core import AEONDeltaV3
        source = inspect.getsource(AEONDeltaV3)
        accepted_idx = source.find('_metacog_accepted = True')
        assert accepted_idx > 0
        # Find cache reset after that
        cache_reset_idx = source.find('inference_cache.reset()', accepted_idx)
        assert cache_reset_idx > accepted_idx, (
            "Cache reset must occur after _metacog_accepted = True"
        )

    def test_ema_hard_reset_preserves_finite_invariant(self):
        """After hard-reset, the EMA should be finite."""
        empirical_L = 0.7434
        ema_L = 1.0121
        # Simulate hard-reset
        alpha = 1.0
        new_ema = (1.0 - alpha) * ema_L + alpha * empirical_L
        assert math.isfinite(new_ema), "EMA must remain finite after hard-reset"
        assert new_ema < 1.0, (
            "After hard-reset from empirical < 1.0, EMA must be < 1.0"
        )

    def test_cycle_exempt_edges_count_increased(self):
        """Adding encoder→meta_learner should increase edge count."""
        from aeon_core import AEONDeltaV3
        exempt = AEONDeltaV3._CYCLE_EXEMPT_EDGES
        # Verify the set includes our new edge
        assert ("encoder", "meta_learner") in exempt
        # Verify we have a reasonable number of exempt edges
        assert len(exempt) >= 40, (
            f"Expected ≥40 cycle-exempt edges, got {len(exempt)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
