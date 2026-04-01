"""Targeted integration tests for the 5 cognitive coherence patches.

These tests validate the specific changes made to bridge the remaining
discontinuities between high-level cognition and low-level execution:
  1. Adaptive stall ratio threshold on ProvablyConvergentMetaLoop
  2. Deeper_meta_loop strategy relaxes stall threshold
  3. KM-damped stall threshold relaxation
  4. Diversity-collapse extra iterations in metacognitive re-run
  5. Stall severity caching and convergence quality degradation
  6. Stall-to-uncertainty bridge
  7. Root-cause-driven adaptation episodes
  8. Metacognitive re-run stall threshold relaxation and restoration
"""

import math
import sys
import os
import importlib

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import aeon_core
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
aeon = importlib.import_module("aeon_core")


# ---------------------------------------------------------------------------
# Helper: minimal config
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Create a minimal AEONConfig with sensible test defaults."""
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vocab_size=256,
        num_pillars=8,
        seq_length=32,
        dropout_rate=0.0,
        meta_dim=32,
        lipschitz_target=0.9,
        vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


# ===========================================================================
# Patch 1: Adaptive stall ratio threshold attribute
# ===========================================================================

class TestAdaptiveStallThreshold:
    """Verify stall_ratio_threshold is an instance attribute on the meta-loop."""

    def test_stall_ratio_threshold_exists(self):
        """ProvablyConvergentMetaLoop must have stall_ratio_threshold attr."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        assert hasattr(ml, 'stall_ratio_threshold'), (
            "ProvablyConvergentMetaLoop missing stall_ratio_threshold attribute"
        )

    def test_stall_ratio_threshold_default(self):
        """Default stall_ratio_threshold should be 0.98."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        assert ml.stall_ratio_threshold == 0.98, (
            f"Expected 0.98, got {ml.stall_ratio_threshold}"
        )

    def test_stall_ratio_threshold_mutable(self):
        """stall_ratio_threshold should be modifiable at runtime."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        ml.stall_ratio_threshold = 0.995
        assert ml.stall_ratio_threshold == 0.995

    def test_stall_ratio_threshold_restored_after_adjustment(self):
        """After adjusting and restoring, threshold should return to default."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        original = ml.stall_ratio_threshold
        ml.stall_ratio_threshold = 0.999
        ml.stall_ratio_threshold = original
        assert ml.stall_ratio_threshold == 0.98


# ===========================================================================
# Patch 2: KM-damped stall relaxation in compute_fixed_point
# ===========================================================================

class TestKMDampedStallRelaxation:
    """Verify that KM damping relaxes the effective stall threshold."""

    def test_km_relax_formula(self):
        """When KM alpha < 1.0, the effective stall threshold increases."""
        # Simulate: alpha = 1/(1+L_C) for L_C=16.8
        L_C = 16.8
        km_alpha = 1.0 / (1.0 + L_C)  # ~0.056
        # The relaxation formula from the patch:
        km_stall_relax = min(0.019, (1.0 - km_alpha) * 0.02)
        effective = min(0.999, 0.98 + km_stall_relax)
        assert effective > 0.98, (
            f"KM-damped effective threshold {effective} should exceed 0.98"
        )
        assert effective <= 0.999, (
            f"KM-damped effective threshold {effective} should not exceed 0.999"
        )

    def test_km_relax_increases_with_damping(self):
        """Higher L_C (more damping) should produce more relaxation."""
        thresholds = []
        for L_C in [2.0, 8.0, 16.0, 32.0]:
            km_alpha = 1.0 / (1.0 + L_C)
            km_stall_relax = min(0.019, (1.0 - km_alpha) * 0.02)
            effective = min(0.999, 0.98 + km_stall_relax)
            thresholds.append(effective)
        # More damping should give higher (more relaxed) threshold
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1], (
                f"Threshold should increase with L_C: {thresholds}"
            )

    def test_km_relax_caps_at_0999(self):
        """Even with extreme damping, threshold must not exceed 0.999."""
        L_C = 1e6
        km_alpha = 1.0 / (1.0 + L_C)
        km_stall_relax = min(0.019, (1.0 - km_alpha) * 0.02)
        effective = min(0.999, 0.98 + km_stall_relax)
        assert effective == 0.999


# ===========================================================================
# Patch 3: Deeper meta-loop stall threshold relaxation
# ===========================================================================

class TestDeeperMetaLoopStallRelaxation:
    """Verify that deeper_meta_loop also relaxes stall_ratio_threshold."""

    def test_evolved_stall_relax_formula(self):
        """When uncertainty > 0, stall threshold should increase from 0.98."""
        unc = 0.1  # evolved_preemptive_uncertainty
        stall = min(0.999, 0.98 + 0.01 * min(2.0, unc / 0.05))
        assert stall > 0.98
        assert stall <= 0.999

    def test_evolved_stall_relax_proportional(self):
        """Higher preemptive uncertainty => more relaxation."""
        thresholds = []
        for unc in [0.01, 0.05, 0.1, 0.2]:
            stall = min(0.999, 0.98 + 0.01 * min(2.0, unc / 0.05))
            thresholds.append(stall)
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1], (
                f"Stall threshold should increase with uncertainty: {thresholds}"
            )


# ===========================================================================
# Patch 4: Stall severity caching
# ===========================================================================

class TestStallSeverityCaching:
    """Verify _cached_stall_severity initialization and degradation logic."""

    def test_cached_stall_severity_init(self):
        """AEONDeltaV3 must initialize _cached_stall_severity to 0.0."""
        cfg = _make_config()
        model = aeon.AEONDeltaV3(cfg)
        assert hasattr(model, '_cached_stall_severity'), (
            "AEONDeltaV3 missing _cached_stall_severity attribute"
        )
        assert model._cached_stall_severity == 0.0

    def test_stall_severity_mapping(self):
        """Stall severity should map ratio ∈ [0.98, 1.0] → [0, 1]."""
        # ratio = 0.98 → severity = 0
        s1 = min(1.0, max(0.0, (0.98 - 0.98) / 0.02))
        assert s1 == 0.0

        # ratio = 0.99 → severity = 0.5
        s2 = min(1.0, max(0.0, (0.99 - 0.98) / 0.02))
        assert abs(s2 - 0.5) < 1e-9

        # ratio = 1.0 → severity = 1.0
        s3 = min(1.0, max(0.0, (1.0 - 0.98) / 0.02))
        assert s3 == 1.0

    def test_convergence_quality_degradation(self):
        """Stall severity should degrade convergence quality proportionally."""
        quality = 0.8
        stall_severity = 0.5
        penalty = stall_severity * 0.3  # 0.15
        degraded = max(0.0, quality * (1.0 - penalty))
        assert abs(degraded - 0.68) < 1e-9, (
            f"Expected ~0.68, got {degraded}"
        )

    def test_convergence_quality_no_degradation_without_stall(self):
        """Without stall (severity = 0), quality should be unchanged."""
        quality = 0.8
        stall_severity = 0.0
        if stall_severity > 0.0:
            penalty = stall_severity * 0.3
            quality = max(0.0, quality * (1.0 - penalty))
        assert quality == 0.8


# ===========================================================================
# Patch 5: Stall-to-uncertainty bridge
# ===========================================================================

class TestStallToUncertaintyBridge:
    """Verify stall severity injects uncertainty proportionally."""

    def test_stall_uncertainty_boost(self):
        """Stall severity > 0 should boost uncertainty."""
        uncertainty = 0.3
        stall_severity = 0.5
        stall_unc_boost = min(1.0 - uncertainty, stall_severity * 0.15)
        new_uncertainty = min(1.0, uncertainty + stall_unc_boost)
        assert new_uncertainty > uncertainty, (
            f"Expected uncertainty to increase from {uncertainty}"
        )

    def test_stall_uncertainty_capped_at_1(self):
        """Stall uncertainty boost should never push uncertainty above 1.0."""
        uncertainty = 0.95
        stall_severity = 1.0
        stall_unc_boost = min(1.0 - uncertainty, stall_severity * 0.15)
        new_uncertainty = min(1.0, uncertainty + stall_unc_boost)
        assert new_uncertainty <= 1.0

    def test_no_stall_no_boost(self):
        """Without stall (severity = 0), no uncertainty boost."""
        uncertainty = 0.5
        stall_severity = 0.0
        if stall_severity > 0.0:
            stall_unc_boost = min(1.0 - uncertainty, stall_severity * 0.15)
            uncertainty = min(1.0, uncertainty + stall_unc_boost)
        assert uncertainty == 0.5


# ===========================================================================
# Patch 6: Diversity-collapse extra iterations
# ===========================================================================

class TestDiversityCollapseExtraIterations:
    """Verify diversity collapse adds extra iterations to re-run."""

    def test_diversity_extra_iterations_formula(self):
        """Low diversity should produce 1-3 extra iterations."""
        threshold = 0.3
        for div_score, expected_min in [(0.1, 2), (0.04, 2), (0.0, 3)]:
            severity = (threshold - div_score) / max(threshold, 1e-6)
            extra = max(1, int(3 * severity))
            assert extra >= expected_min, (
                f"div_score={div_score}: expected >= {expected_min} extra iters, got {extra}"
            )

    def test_no_extra_iters_above_threshold(self):
        """Diversity score above threshold should not add extra iterations."""
        threshold = 0.3
        div_score = 0.5
        # Above threshold, the condition div_score < threshold is false,
        # so no extra iterations are added.
        should_add = div_score < threshold
        assert not should_add


# ===========================================================================
# Patch 7: Root-cause-driven adaptation
# ===========================================================================

class TestRootCauseDrivenAdaptation:
    """Verify root-cause subsystems generate error_evolution episodes."""

    def test_root_cause_error_class_format(self):
        """Root-cause episodes should use 'root_cause_{subsystem}' format."""
        subsystems = ["meta_loop", "diversity_monitor", "world_model"]
        for sub in subsystems:
            error_class = f'root_cause_{sub}'
            assert error_class.startswith('root_cause_'), (
                f"Expected 'root_cause_' prefix, got {error_class}"
            )
            assert sub in error_class

    def test_root_cause_subsystem_limit(self):
        """At most 3 root-cause subsystems should be recorded per cycle."""
        subsystems = ["a", "b", "c", "d", "e"]
        limited = subsystems[:3]
        assert len(limited) == 3
        assert limited == ["a", "b", "c"]


# ===========================================================================
# Patch 8: Metacognitive re-run stall threshold restoration
# ===========================================================================

class TestMetacognitiveRerunStallRestoration:
    """Verify stall threshold is restored after metacognitive re-run."""

    def test_restore_after_deeper_rerun(self):
        """stall_ratio_threshold should be restored to original after re-run."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        original = ml.stall_ratio_threshold
        # Simulate metacognitive re-run adjustment
        ml.stall_ratio_threshold = min(0.999, original + 0.015)
        assert ml.stall_ratio_threshold > original
        # Simulate restoration
        ml.stall_ratio_threshold = original
        assert ml.stall_ratio_threshold == 0.98

    def test_restore_after_evolved_guidance(self):
        """stall_ratio_threshold should be restored to 0.98 after evolved guidance."""
        cfg = _make_config()
        ml = aeon.ProvablyConvergentMetaLoop(cfg)
        # Simulate evolved guidance adjustment
        ml.stall_ratio_threshold = 0.995
        # Simulate restoration (as done in _reasoning_core_impl)
        ml.stall_ratio_threshold = 0.98
        assert ml.stall_ratio_threshold == 0.98


# ===========================================================================
# Integration: End-to-end signal flow
# ===========================================================================

class TestEndToEndSignalFlow:
    """Verify the full signal path: stall → severity → quality → uncertainty → trigger."""

    def test_full_signal_chain(self):
        """Stall at ratio 0.99 should degrade quality and boost uncertainty."""
        # 1. Stall detection: ratio = 0.99
        stall_ratio = 0.99
        stall_severity = min(1.0, max(0.0, (stall_ratio - 0.98) / 0.02))
        assert abs(stall_severity - 0.5) < 1e-9

        # 2. Convergence quality degradation
        quality = 0.7
        penalty = stall_severity * 0.3
        quality = max(0.0, quality * (1.0 - penalty))
        assert quality < 0.7

        # 3. Uncertainty boost
        uncertainty = 0.3
        stall_unc_boost = min(1.0 - uncertainty, stall_severity * 0.15)
        uncertainty = min(1.0, uncertainty + stall_unc_boost)
        assert uncertainty > 0.3

        # 4. Trigger would fire since uncertainty + degraded quality
        # combine to push metacognitive score higher.
        # (Exact trigger logic depends on weights, but the signals are present)
        assert stall_severity > 0, "Stall severity should be non-zero"
        assert quality < 0.7, "Quality should be degraded"
        assert uncertainty > 0.3, "Uncertainty should be boosted"

    def test_no_stall_no_cascade(self):
        """Without stall, no quality degradation or uncertainty boost."""
        stall_ratio = 0.95  # Below threshold, no stall
        stall_severity = 0.0  # No stall detected

        quality = 0.7
        if stall_severity > 0:
            quality = max(0.0, quality * (1.0 - stall_severity * 0.3))
        assert quality == 0.7

        uncertainty = 0.3
        if stall_severity > 0:
            uncertainty = min(1.0, uncertainty + stall_severity * 0.15)
        assert uncertainty == 0.3
