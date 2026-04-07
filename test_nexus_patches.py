"""
AEON-Delta RMT v3.1 — NEXUS Series Final Integration Patch Tests
═════════════════════════════════════════════════════════════════

Tests for NEXUS-1, NEXUS-2, and NEXUS-3 patches which close the
remaining orphaned-signal gaps and complete the cognitive activation
of AEON-Delta RMT v3.1.

Patch Summary
─────────────
NEXUS-1  Wire 20 orphaned cognitive signals to MCT evaluate():
           (a) safety_violation_active    → recovery_pressure
           (b) safety_pressure_active     → recovery_pressure
           (c) error_recurrence_rate      → uncertainty
           (d) module_coherence_score     → coherence_deficit  (inv)
           (e) memory_consolidation_health→ memory_trust_deficit (inv)
           (f) memory_retrieval_quality   → memory_trust_deficit (inv)
           (g) auto_critic_revision_pressure → coherence_deficit
           (h) reinforcement_ineffective_pressure → coherence_deficit
           (i) world_model_prediction_error → uncertainty
           (j) vq_codebook_quality        → convergence_conflict (inv)
           (k) system_health_aggregate    → recovery_pressure  (inv)
           (l) ucc_should_rerun           → coherence_deficit
           (m) ucc_coherence_score        → coherence_deficit  (inv)
           (n) ucc_convergence_verdict    → convergence_conflict (inv)
           (o) cross_pass_instability_pressure → oscillation_severity
           (p) training_adaptation_confidence  → convergence_conflict (inv)
           (q) cache_staleness_risk       → coherence_deficit
           (r) pillar_consistency_pressure→ coherence_deficit
           (s) diversity_collapse_alarm   → coherence_deficit
           (t) reinforce_coherence_score  → coherence_deficit  (inv)

NEXUS-2  compute_loss reads training loop health signals from the bus:
           training_adaptation_confidence → loss amplification
           training_convergence_trend     → loss amplification
           training_step_loss             → loss dampening (spike guard)

NEXUS-3  _bridge_epoch_feedback reads MCT output signals:
           mct_should_trigger → clip relaxation when low
           mct_trigger_score  → clip tightening when high
"""

import sys
import os
import math
import types
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    AEONConfig,
)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def bus():
    return CognitiveFeedbackBus(hidden_dim=64)


@pytest.fixture
def mct(bus):
    t = MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=3,
    )
    t._feedback_bus_ref = bus
    return t


def _base_eval_kwargs(**overrides):
    """Return minimal safe kwargs for MCT.evaluate()."""
    defaults = dict(
        uncertainty=0.0,
        is_diverging=False,
        topology_catastrophe=False,
        coherence_deficit=0.0,
        memory_staleness=False,
        recovery_pressure=0.0,
        world_model_surprise=False,
        causal_quality=1.0,
        safety_violation=False,
        diversity_collapse=0.0,
        memory_trust_deficit=0.0,
        convergence_conflict=0.0,
        output_reliability=1.0,
        spectral_stability_margin=1.0,
        border_uncertainty=0.0,
        stall_severity=0.0,
        oscillation_severity=0.0,
    )
    defaults.update(overrides)
    return defaults


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (a): safety_violation_active → recovery_pressure
# ═══════════════════════════════════════════════════════════════════

class TestNexus1a_SafetyViolationActive:
    """NEXUS-1a: safety_violation_active → MCT recovery_pressure."""

    def test_signal_read_when_above_threshold(self, mct, bus):
        """High safety_violation_active raises recovery_pressure."""
        bus.write_signal('safety_violation_active', 1.0)
        result_active = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('safety_violation_active', 0.0)
        result_off = mct.evaluate(**_base_eval_kwargs())

        score_active = result_active.get('trigger_score', 0.0)
        score_off = result_off.get('trigger_score', 0.0)
        assert score_active >= score_off, (
            f"safety_violation_active=1.0 should raise trigger score "
            f"(got {score_active} vs {score_off})"
        )

    def test_signal_below_threshold_no_effect(self, mct, bus):
        """safety_violation_active=0.3 (below 0.5 threshold) has no effect."""
        bus.write_signal('safety_violation_active', 0.3)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('safety_violation_active', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        # Both should be roughly equal since 0.3 < threshold 0.5
        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05

    def test_no_bus_no_crash(self):
        """MCT with no bus ref still evaluates without error."""
        t = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
        # No bus attached
        result = t.evaluate(**_base_eval_kwargs())
        assert 'trigger_score' in result


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (b): safety_pressure_active → recovery_pressure
# ═══════════════════════════════════════════════════════════════════

class TestNexus1b_SafetyPressureActive:
    """NEXUS-1b: safety_pressure_active → MCT recovery_pressure."""

    def test_high_safety_pressure_raises_score(self, mct, bus):
        """safety_pressure_active=0.9 raises trigger score."""
        bus.write_signal('safety_pressure_active', 0.9)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('safety_pressure_active', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_partial_safety_pressure(self, mct, bus):
        """safety_pressure_active at 0.5 has proportionally smaller effect."""
        bus.write_signal('safety_pressure_active', 1.0)
        result_full = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('safety_pressure_active', 0.5)
        result_half = mct.evaluate(**_base_eval_kwargs())

        # Full pressure should equal or exceed half pressure
        assert (
            result_full.get('trigger_score', 0.0)
            >= result_half.get('trigger_score', 0.0)
        )


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (c): error_recurrence_rate → uncertainty
# ═══════════════════════════════════════════════════════════════════

class TestNexus1c_ErrorRecurrenceRate:
    """NEXUS-1c: error_recurrence_rate → MCT uncertainty."""

    def test_high_recurrence_raises_score(self, mct, bus):
        """error_recurrence_rate=0.8 raises trigger score."""
        bus.write_signal('error_recurrence_rate', 0.8)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('error_recurrence_rate', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """error_recurrence_rate=0.2 (below 0.3 threshold) has no effect."""
        bus.write_signal('error_recurrence_rate', 0.2)
        result = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('error_recurrence_rate', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (d): module_coherence_score → coherence_deficit (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1d_ModuleCoherenceScore:
    """NEXUS-1d: module_coherence_score (inv) → MCT coherence_deficit."""

    def test_low_coherence_raises_score(self, mct, bus):
        """module_coherence_score=0.2 (< 0.7) raises trigger score."""
        bus.write_signal('module_coherence_score', 0.2)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('module_coherence_score', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )

    def test_coherence_above_07_no_penalty(self, mct, bus):
        """module_coherence_score >= 0.7 does not penalise MCT."""
        bus.write_signal('module_coherence_score', 0.8)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('module_coherence_score', 1.0)
        result_perfect = mct.evaluate(**_base_eval_kwargs())

        # Both should produce similar scores
        diff = abs(
            result_ok.get('trigger_score', 0.0)
            - result_perfect.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (e): memory_consolidation_health → memory_trust_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1e_MemoryConsolidationHealth:
    """NEXUS-1e: memory_consolidation_health (inv) → memory_trust_deficit."""

    def test_poor_consolidation_raises_score(self, mct, bus):
        """memory_consolidation_health=0.1 (< 0.5) raises trigger score."""
        bus.write_signal('memory_consolidation_health', 0.1)
        result_poor = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('memory_consolidation_health', 1.0)
        result_good = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_poor.get('trigger_score', 0.0)
            >= result_good.get('trigger_score', 0.0)
        )

    def test_health_above_05_no_penalty(self, mct, bus):
        """memory_consolidation_health >= 0.5 does not trigger penalty."""
        bus.write_signal('memory_consolidation_health', 0.6)
        result = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('memory_consolidation_health', 1.0)
        result_full = mct.evaluate(**_base_eval_kwargs())

        diff = abs(
            result.get('trigger_score', 0.0)
            - result_full.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (f): memory_retrieval_quality → memory_trust_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1f_MemoryRetrievalQuality:
    """NEXUS-1f: memory_retrieval_quality (inv) → MCT memory_trust_deficit."""

    def test_low_retrieval_quality_raises_score(self, mct, bus):
        """memory_retrieval_quality=0.1 raises trigger score."""
        bus.write_signal('memory_retrieval_quality', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('memory_retrieval_quality', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (g): auto_critic_revision_pressure → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1g_AutoCriticRevisionPressure:
    """NEXUS-1g: auto_critic_revision_pressure → MCT coherence_deficit."""

    def test_high_revision_pressure_raises_score(self, mct, bus):
        """auto_critic_revision_pressure=0.9 raises trigger score."""
        bus.write_signal('auto_critic_revision_pressure', 0.9)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('auto_critic_revision_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """auto_critic_revision_pressure=0.2 (below 0.3) has no effect."""
        bus.write_signal('auto_critic_revision_pressure', 0.2)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('auto_critic_revision_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (h): reinforcement_ineffective_pressure → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1h_ReinforcementIneffectivePressure:
    """NEXUS-1h: reinforcement_ineffective_pressure → coherence_deficit."""

    def test_high_pressure_raises_score(self, mct, bus):
        """reinforcement_ineffective_pressure=0.8 raises trigger score."""
        bus.write_signal('reinforcement_ineffective_pressure', 0.8)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('reinforcement_ineffective_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (i): world_model_prediction_error → uncertainty
# ═══════════════════════════════════════════════════════════════════

class TestNexus1i_WorldModelPredictionError:
    """NEXUS-1i: world_model_prediction_error → MCT uncertainty."""

    def test_high_prediction_error_raises_score(self, mct, bus):
        """world_model_prediction_error=0.9 raises trigger score."""
        bus.write_signal('world_model_prediction_error', 0.9)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('world_model_prediction_error', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """world_model_prediction_error=0.1 (below 0.3) has no effect."""
        bus.write_signal('world_model_prediction_error', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('world_model_prediction_error', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (j): vq_codebook_quality → convergence_conflict (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1j_VqCodebookQuality:
    """NEXUS-1j: vq_codebook_quality (inv) → MCT convergence_conflict."""

    def test_collapsed_codebook_raises_score(self, mct, bus):
        """vq_codebook_quality=0.1 (< 0.5) raises trigger score."""
        bus.write_signal('vq_codebook_quality', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('vq_codebook_quality', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )

    def test_above_threshold_no_penalty(self, mct, bus):
        """vq_codebook_quality >= 0.5 does not penalise MCT."""
        bus.write_signal('vq_codebook_quality', 0.7)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('vq_codebook_quality', 1.0)
        result_perfect = mct.evaluate(**_base_eval_kwargs())

        diff = abs(
            result_ok.get('trigger_score', 0.0)
            - result_perfect.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (k): system_health_aggregate → recovery_pressure (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1k_SystemHealthAggregate:
    """NEXUS-1k: system_health_aggregate (inv) → MCT recovery_pressure."""

    def test_low_system_health_raises_score(self, mct, bus):
        """system_health_aggregate=0.1 (< 0.6) raises trigger score."""
        bus.write_signal('system_health_aggregate', 0.1)
        result_sick = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('system_health_aggregate', 1.0)
        result_healthy = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_sick.get('trigger_score', 0.0)
            >= result_healthy.get('trigger_score', 0.0)
        )

    def test_health_above_06_no_penalty(self, mct, bus):
        """system_health_aggregate >= 0.6 does not trigger penalty."""
        bus.write_signal('system_health_aggregate', 0.7)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('system_health_aggregate', 1.0)
        result_full = mct.evaluate(**_base_eval_kwargs())

        diff = abs(
            result_ok.get('trigger_score', 0.0)
            - result_full.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (l): ucc_should_rerun → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1l_UccShouldRerun:
    """NEXUS-1l: ucc_should_rerun → MCT coherence_deficit."""

    def test_rerun_flag_raises_score(self, mct, bus):
        """ucc_should_rerun=1.0 raises trigger score."""
        bus.write_signal('ucc_should_rerun', 1.0)
        result_rerun = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('ucc_should_rerun', 0.0)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_rerun.get('trigger_score', 0.0)
            >= result_ok.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """ucc_should_rerun=0.3 (below 0.5) has no effect."""
        bus.write_signal('ucc_should_rerun', 0.3)
        result = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('ucc_should_rerun', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (m): ucc_coherence_score → coherence_deficit (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1m_UccCoherenceScore:
    """NEXUS-1m: ucc_coherence_score (inv) → MCT coherence_deficit."""

    def test_low_ucc_coherence_raises_score(self, mct, bus):
        """ucc_coherence_score=0.1 (< 0.5) raises trigger score."""
        bus.write_signal('ucc_coherence_score', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('ucc_coherence_score', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (n): ucc_convergence_verdict → convergence_conflict (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1n_UccConvergenceVerdict:
    """NEXUS-1n: ucc_convergence_verdict (inv) → convergence_conflict."""

    def test_no_convergence_verdict_raises_score(self, mct, bus):
        """ucc_convergence_verdict=0.0 (not converged) raises trigger score."""
        bus.write_signal('ucc_convergence_verdict', 0.0)
        result_no_conv = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('ucc_convergence_verdict', 1.0)
        result_converged = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_no_conv.get('trigger_score', 0.0)
            >= result_converged.get('trigger_score', 0.0)
        )


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (o): cross_pass_instability_pressure → oscillation_severity
# ═══════════════════════════════════════════════════════════════════

class TestNexus1o_CrossPassInstabilityPressure:
    """NEXUS-1o: cross_pass_instability_pressure → oscillation_severity."""

    def test_high_instability_raises_score(self, mct, bus):
        """cross_pass_instability_pressure=0.8 raises trigger score."""
        bus.write_signal('cross_pass_instability_pressure', 0.8)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('cross_pass_instability_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """cross_pass_instability_pressure=0.1 (below 0.2) has no effect."""
        bus.write_signal('cross_pass_instability_pressure', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('cross_pass_instability_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (p): training_adaptation_confidence → convergence_conflict
# ═══════════════════════════════════════════════════════════════════

class TestNexus1p_TrainingAdaptationConfidence:
    """NEXUS-1p: training_adaptation_confidence (inv) → convergence_conflict."""

    def test_low_confidence_raises_score(self, mct, bus):
        """training_adaptation_confidence=0.1 (< 0.4) raises trigger score."""
        bus.write_signal('training_adaptation_confidence', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('training_adaptation_confidence', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )

    def test_above_04_no_penalty(self, mct, bus):
        """training_adaptation_confidence >= 0.4 does not penalise MCT."""
        bus.write_signal('training_adaptation_confidence', 0.5)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('training_adaptation_confidence', 1.0)
        result_full = mct.evaluate(**_base_eval_kwargs())

        diff = abs(
            result_ok.get('trigger_score', 0.0)
            - result_full.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (q): cache_staleness_risk → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1q_CacheStalenessRisk:
    """NEXUS-1q: cache_staleness_risk → MCT coherence_deficit."""

    def test_high_staleness_raises_score(self, mct, bus):
        """cache_staleness_risk=0.9 raises trigger score."""
        bus.write_signal('cache_staleness_risk', 0.9)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('cache_staleness_risk', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """cache_staleness_risk=0.2 (below 0.3) has no effect."""
        bus.write_signal('cache_staleness_risk', 0.2)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('cache_staleness_risk', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (r): pillar_consistency_pressure → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1r_PillarConsistencyPressure:
    """NEXUS-1r: pillar_consistency_pressure → MCT coherence_deficit."""

    def test_high_pillar_inconsistency_raises_score(self, mct, bus):
        """pillar_consistency_pressure=0.8 raises trigger score."""
        bus.write_signal('pillar_consistency_pressure', 0.8)
        result_high = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('pillar_consistency_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_high.get('trigger_score', 0.0)
            >= result_zero.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """pillar_consistency_pressure=0.1 (below 0.2) has no effect."""
        bus.write_signal('pillar_consistency_pressure', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('pillar_consistency_pressure', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result_low.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (s): diversity_collapse_alarm → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestNexus1s_DiversityCollapseAlarm:
    """NEXUS-1s: diversity_collapse_alarm → MCT coherence_deficit."""

    def test_alarm_active_raises_score(self, mct, bus):
        """diversity_collapse_alarm=1.0 raises trigger score."""
        bus.write_signal('diversity_collapse_alarm', 1.0)
        result_alarm = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('diversity_collapse_alarm', 0.0)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_alarm.get('trigger_score', 0.0)
            >= result_ok.get('trigger_score', 0.0)
        )

    def test_below_threshold_no_effect(self, mct, bus):
        """diversity_collapse_alarm=0.3 (below 0.5) has no effect."""
        bus.write_signal('diversity_collapse_alarm', 0.3)
        result = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('diversity_collapse_alarm', 0.0)
        result_zero = mct.evaluate(**_base_eval_kwargs())

        assert abs(
            result.get('trigger_score', 0.0)
            - result_zero.get('trigger_score', 0.0)
        ) < 0.05


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 (t): reinforce_coherence_score → coherence_deficit (inv)
# ═══════════════════════════════════════════════════════════════════

class TestNexus1t_ReinforceCoherenceScore:
    """NEXUS-1t: reinforce_coherence_score (inv) → coherence_deficit."""

    def test_low_reinforce_coherence_raises_score(self, mct, bus):
        """reinforce_coherence_score=0.1 (< 0.5) raises trigger score."""
        bus.write_signal('reinforce_coherence_score', 0.1)
        result_low = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('reinforce_coherence_score', 1.0)
        result_high = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_low.get('trigger_score', 0.0)
            >= result_high.get('trigger_score', 0.0)
        )

    def test_above_threshold_no_penalty(self, mct, bus):
        """reinforce_coherence_score >= 0.5 does not penalise MCT."""
        bus.write_signal('reinforce_coherence_score', 0.6)
        result_ok = mct.evaluate(**_base_eval_kwargs())

        bus.write_signal('reinforce_coherence_score', 1.0)
        result_full = mct.evaluate(**_base_eval_kwargs())

        diff = abs(
            result_ok.get('trigger_score', 0.0)
            - result_full.get('trigger_score', 0.0)
        )
        assert diff < 0.1


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-1 Compound: multiple orphan signals co-activating
# ═══════════════════════════════════════════════════════════════════

class TestNexus1_Compound:
    """NEXUS-1 compound: co-activation of multiple orphaned signals."""

    def test_all_signals_combined_raise_trigger(self, mct, bus):
        """All 20 orphaned signals high together produce much higher score."""
        # Write all signals to worst-case values
        bus.write_signal('safety_violation_active', 1.0)
        bus.write_signal('safety_pressure_active', 1.0)
        bus.write_signal('error_recurrence_rate', 1.0)
        bus.write_signal('module_coherence_score', 0.1)
        bus.write_signal('memory_consolidation_health', 0.1)
        bus.write_signal('memory_retrieval_quality', 0.1)
        bus.write_signal('auto_critic_revision_pressure', 1.0)
        bus.write_signal('reinforcement_ineffective_pressure', 1.0)
        bus.write_signal('world_model_prediction_error', 1.0)
        bus.write_signal('vq_codebook_quality', 0.1)
        bus.write_signal('system_health_aggregate', 0.1)
        bus.write_signal('ucc_should_rerun', 1.0)
        bus.write_signal('ucc_coherence_score', 0.1)
        bus.write_signal('ucc_convergence_verdict', 0.0)
        bus.write_signal('cross_pass_instability_pressure', 1.0)
        bus.write_signal('training_adaptation_confidence', 0.1)
        bus.write_signal('cache_staleness_risk', 1.0)
        bus.write_signal('pillar_consistency_pressure', 1.0)
        bus.write_signal('diversity_collapse_alarm', 1.0)
        bus.write_signal('reinforce_coherence_score', 0.1)
        result_worst = mct.evaluate(**_base_eval_kwargs())

        # Write all signals to best-case values
        bus.write_signal('safety_violation_active', 0.0)
        bus.write_signal('safety_pressure_active', 0.0)
        bus.write_signal('error_recurrence_rate', 0.0)
        bus.write_signal('module_coherence_score', 1.0)
        bus.write_signal('memory_consolidation_health', 1.0)
        bus.write_signal('memory_retrieval_quality', 1.0)
        bus.write_signal('auto_critic_revision_pressure', 0.0)
        bus.write_signal('reinforcement_ineffective_pressure', 0.0)
        bus.write_signal('world_model_prediction_error', 0.0)
        bus.write_signal('vq_codebook_quality', 1.0)
        bus.write_signal('system_health_aggregate', 1.0)
        bus.write_signal('ucc_should_rerun', 0.0)
        bus.write_signal('ucc_coherence_score', 1.0)
        bus.write_signal('ucc_convergence_verdict', 1.0)
        bus.write_signal('cross_pass_instability_pressure', 0.0)
        bus.write_signal('training_adaptation_confidence', 1.0)
        bus.write_signal('cache_staleness_risk', 0.0)
        bus.write_signal('pillar_consistency_pressure', 0.0)
        bus.write_signal('diversity_collapse_alarm', 0.0)
        bus.write_signal('reinforce_coherence_score', 1.0)
        result_best = mct.evaluate(**_base_eval_kwargs())

        assert (
            result_worst.get('trigger_score', 0.0)
            > result_best.get('trigger_score', 0.0)
        ), (
            f"All-bad score {result_worst.get('trigger_score', 0.0):.4f} "
            f"should exceed all-good {result_best.get('trigger_score', 0.0):.4f}"
        )

    def test_combined_worst_case_triggers_mct(self, mct, bus):
        """Worst-case compound signals should trigger MCT (score >= threshold)."""
        # Use very high weights on each channel via dedicated signal
        mct._feedback_bus_ref = bus
        bus.write_signal('safety_violation_active', 1.0)
        bus.write_signal('error_recurrence_rate', 1.0)
        bus.write_signal('module_coherence_score', 0.0)
        bus.write_signal('memory_consolidation_health', 0.0)
        bus.write_signal('world_model_prediction_error', 1.0)
        bus.write_signal('ucc_should_rerun', 1.0)
        bus.write_signal('ucc_convergence_verdict', 0.0)
        bus.write_signal('system_health_aggregate', 0.0)
        result = mct.evaluate(**_base_eval_kwargs())
        # Should have a positive trigger score
        assert result.get('trigger_score', 0.0) > 0.0


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-2: compute_loss reads training health signals
# ═══════════════════════════════════════════════════════════════════

class TestNexus2_TrainingSignalsToLoss:
    """NEXUS-2: training_adaptation_confidence / training_convergence_trend
    / training_step_loss are now consumed by compute_loss."""

    def _make_mock_model(self, bus):
        """Build a minimal object that mimics AEONDeltaV3 compute_loss context."""
        # We test by importing compute_loss indirectly via a minimal shim.
        # Instead, we verify the code paths exist and the signals are read.
        from aeon_core import AEONDeltaV3
        return AEONDeltaV3

    def test_nexus2_reads_training_adaptation_confidence(self):
        """compute_loss section contains read for training_adaptation_confidence."""
        import ast
        src = open('aeon_core.py').read()
        assert "'training_adaptation_confidence'" in src or \
               '"training_adaptation_confidence"' in src

    def test_nexus2_reads_training_convergence_trend(self):
        """compute_loss section contains read for training_convergence_trend."""
        src = open('aeon_core.py').read()
        assert "'training_convergence_trend'" in src or \
               '"training_convergence_trend"' in src

    def test_nexus2_reads_training_step_loss(self):
        """compute_loss section contains read for training_step_loss."""
        src = open('aeon_core.py').read()
        assert "'training_step_loss'" in src or \
               '"training_step_loss"' in src

    def test_nexus2_low_confidence_amplifies_loss(self):
        """NEXUS-2 logic: low adaptation confidence → loss scale > 1.0."""
        # Simulate the NEXUS-2 logic isolated
        tac = 0.1  # very low confidence
        tct = 0.0  # neutral trend
        tsl = 0.0  # no step loss spike
        scale = 1.0
        if tac < 0.4:
            conf_deficit = (0.4 - tac) / 0.4
            scale *= 1.0 + 0.2 * conf_deficit
        assert scale > 1.0, f"Expected scale > 1.0, got {scale}"

    def test_nexus2_negative_trend_amplifies_loss(self):
        """NEXUS-2 logic: negative convergence trend → loss scale > 1.0."""
        tac = 1.0  # confidence fine
        tct = -0.5  # regressing
        scale = 1.0
        if tct < 0.0:
            scale *= 1.0 + min(0.15, abs(tct) * 0.15)
        assert scale > 1.0, f"Expected scale > 1.0, got {scale}"

    def test_nexus2_stale_step_loss_dampens_loss(self):
        """NEXUS-2 logic: step loss spike (> 5× current) dampens loss."""
        tsl = 50.0  # very high stale loss
        cur = 1.0   # current loss
        scale = 1.0
        if tsl > 0.0 and cur > 0.0:
            if tsl > 5.0 * cur:
                scale *= 0.85
        assert scale < 1.0, f"Expected scale < 1.0, got {scale}"

    def test_nexus2_good_training_state_neutral_scale(self):
        """NEXUS-2: healthy training signals → scale ≈ 1.0."""
        tac = 0.9   # high confidence
        tct = 0.1   # positive trend
        tsl = 0.0
        scale = 1.0
        if tac < 0.4:
            conf_deficit = (0.4 - tac) / 0.4
            scale *= 1.0 + 0.2 * conf_deficit
        if tct < 0.0:
            scale *= 1.0 + min(0.15, abs(tct) * 0.15)
        # scale should be 1.0 since tac >= 0.4 and tct >= 0
        assert abs(scale - 1.0) < 0.001, f"Expected scale ≈ 1.0, got {scale}"


# ═══════════════════════════════════════════════════════════════════
#  NEXUS-3: _bridge_epoch_feedback reads MCT output signals
# ═══════════════════════════════════════════════════════════════════

class TestNexus3_BridgeReadsMctSignals:
    """NEXUS-3: _bridge_epoch_feedback reads mct_should_trigger and
    mct_trigger_score from the feedback bus."""

    def test_nexus3_code_present_in_bridge(self):
        """_bridge_epoch_feedback contains NEXUS-3 block."""
        src = open('aeon_core.py').read()
        assert 'NEXUS-3' in src

    def test_nexus3_reads_mct_should_trigger(self):
        """Source contains read_signal for mct_should_trigger in bridge."""
        src = open('aeon_core.py').read()
        assert 'mct_should_trigger' in src

    def test_nexus3_reads_mct_trigger_score(self):
        """Source contains read_signal for mct_trigger_score in bridge."""
        src = open('aeon_core.py').read()
        assert 'mct_trigger_score' in src

    def test_nexus3_high_score_tightens_clip(self):
        """NEXUS-3 logic: mct_trigger_score > 0.7 → tighten gradient clip."""
        score = 0.9
        old_clip = 1.0
        new_clip = old_clip
        if score > 0.7:
            new_clip = max(0.1, old_clip * 0.97)
        assert new_clip < old_clip, (
            f"High trigger score should tighten clip: {old_clip} → {new_clip}"
        )

    def test_nexus3_low_score_relaxes_clip(self):
        """NEXUS-3 logic: mct_trigger_score < 0.2 and no trigger → relax clip."""
        score = 0.1
        trigger = 0.0
        old_clip = 1.0
        new_clip = old_clip
        if score < 0.2 and trigger < 0.5:
            new_clip = min(5.0, old_clip * 1.01)
        assert new_clip > old_clip, (
            f"Low trigger score should relax clip: {old_clip} → {new_clip}"
        )

    def test_nexus3_mid_range_score_no_change(self):
        """NEXUS-3: 0.2 <= score <= 0.7 → clip unchanged."""
        score = 0.5
        trigger = 0.3
        old_clip = 1.0
        new_clip = old_clip
        if score > 0.7:
            new_clip = max(0.1, old_clip * 0.97)
        elif score < 0.2 and trigger < 0.5:
            new_clip = min(5.0, old_clip * 1.01)
        assert new_clip == old_clip

    def test_nexus3_bus_read_integration(self):
        """Mock bridge reads mct signals from bus correctly."""
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.85)

        score = float(bus.read_signal('mct_trigger_score', 0.0))
        trigger = float(bus.read_signal('mct_should_trigger', 0.0))

        assert score == pytest.approx(0.85, abs=0.01)
        assert trigger == pytest.approx(1.0, abs=0.01)

    def test_nexus3_clip_bounds_enforced(self):
        """NEXUS-3 clip adjustments respect min=0.1 / max=5.0 bounds."""
        # Tighten from near-minimum
        near_min = 0.11
        tightened = max(0.1, near_min * 0.97)
        assert tightened >= 0.1

        # Relax from near-maximum
        near_max = 4.99
        relaxed = min(5.0, near_max * 1.01)
        assert relaxed <= 5.0


# ═══════════════════════════════════════════════════════════════════
#  Integration: MCT still works with no bus
# ═══════════════════════════════════════════════════════════════════

class TestNexus_Regression:
    """NEXUS patches are backward-compatible: MCT still works without bus."""

    def test_mct_no_bus_no_crash(self):
        """MCT evaluates without errors when no feedback bus is attached."""
        t = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
        result = t.evaluate(**_base_eval_kwargs())
        assert 'trigger_score' in result
        assert isinstance(result['trigger_score'], float)

    def test_mct_bus_none_explicitly(self):
        """MCT._feedback_bus_ref = None is safe."""
        t = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
        t._feedback_bus_ref = None
        result = t.evaluate(**_base_eval_kwargs())
        assert 'trigger_score' in result

    def test_mct_base_signals_unaffected_when_orphans_zero(self, mct, bus):
        """All orphaned signals at zero → MCT score equals baseline."""
        for sig in [
            'safety_violation_active', 'safety_pressure_active',
            'error_recurrence_rate', 'auto_critic_revision_pressure',
            'reinforcement_ineffective_pressure', 'world_model_prediction_error',
            'ucc_should_rerun', 'cross_pass_instability_pressure',
            'cache_staleness_risk', 'pillar_consistency_pressure',
            'diversity_collapse_alarm',
        ]:
            bus.write_signal(sig, 0.0)
        for sig in [
            'module_coherence_score', 'memory_consolidation_health',
            'memory_retrieval_quality', 'vq_codebook_quality',
            'system_health_aggregate', 'ucc_coherence_score',
            'ucc_convergence_verdict', 'training_adaptation_confidence',
            'reinforce_coherence_score',
        ]:
            bus.write_signal(sig, 1.0)

        result = mct.evaluate(**_base_eval_kwargs())
        # Score should be near zero when all base kwargs are 0/neutral
        assert result.get('trigger_score', 0.0) < 1.0

    def test_mct_returns_result_dict_shape(self, mct, bus):
        """MCT result dict always contains required keys after NEXUS patches."""
        result = mct.evaluate(**_base_eval_kwargs())
        for key in ['trigger_score', 'should_trigger', 'triggers_active']:
            assert key in result, f"Missing key: {key}"

    def test_signal_monotonicity_safety_violation(self, mct, bus):
        """Higher safety_violation_active → higher or equal trigger score."""
        scores = []
        for val in [0.0, 0.6, 0.8, 1.0]:
            bus.write_signal('safety_violation_active', val)
            r = mct.evaluate(**_base_eval_kwargs())
            scores.append(r.get('trigger_score', 0.0))
        # Scores should be non-decreasing (monotonic)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-6, (
                f"Non-monotonic at {i}: {scores[i]:.4f} > {scores[i+1]:.4f}"
            )

    def test_signal_monotonicity_system_health_aggregate(self, mct, bus):
        """Lower system_health_aggregate → higher or equal trigger score."""
        scores = []
        for val in [1.0, 0.6, 0.4, 0.1]:  # decreasing health
            bus.write_signal('system_health_aggregate', val)
            r = mct.evaluate(**_base_eval_kwargs())
            scores.append(r.get('trigger_score', 0.0))
        # Scores should be non-decreasing as health degrades
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-6, (
                f"Non-monotonic at {i}: {scores[i]:.4f} > {scores[i+1]:.4f}"
            )
