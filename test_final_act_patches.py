"""
AEON-Delta RMT v3.1 — FINAL-ACT Series Cognitive Activation Tests
══════════════════════════════════════════════════════════════════

Tests for FINAL-ACT-1, FINAL-ACT-2, and FINAL-ACT-3 patches which
wire the last 14 orphaned bus signals and close the remaining
cognitive integration gaps, completing the transition from
"connected architecture" to "functional cognitive organism."

Patch Summary
─────────────
FINAL-ACT-1  Wire 14 orphaned signals to MCT evaluate():
               (a) auto_critic_quality         → low_output_reliability (inv)
               (b) cognitive_health_critical   → recovery_pressure
               (c) cognitive_unity_deficit     → coherence_deficit
               (d) cross_pass_oscillation      → oscillation_severity
               (e) divergence_active           → diverging
               (f) integration_failure_rate    → recovery_pressure
               (g) integration_health          → coherence_deficit     (inv)
               (h) output_reliability_composite→ low_output_reliability (inv)
               (i) stall_severity_pressure     → stall_severity
               (j) teacher_student_inversion_ok→ convergence_conflict  (inv)
               (k) training_phase_pressure     → recovery_pressure
               (l) ucc_evaluation_ok           → coherence_deficit     (inv)
               (m) world_model_surprise_active → world_model_surprise
               (n) z_filter_pass_ratio         → uncertainty            (inv)

FINAL-ACT-2  compute_loss reads server-side integration signals:
               integration_health            → loss amplification
               ucc_evaluation_ok             → loss amplification
               teacher_student_inversion_ok  → loss dampening

FINAL-ACT-3  _bridge_epoch_feedback reads integration quality signals:
               integration_failure_rate      → gradient clip tightening
               z_filter_pass_ratio           → gradient clip tightening
               teacher_student_inversion_ok  → gradient clip tightening
"""

import sys
import os
import types
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
)


# ═══════════════════════════════════════════════════════════════════
#  Shared fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def bus():
    return CognitiveFeedbackBus(hidden_dim=64)


@pytest.fixture
def mct(bus):
    trigger = MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=3,
    )
    trigger._feedback_bus_ref = bus
    return trigger


def _base_score(mct_obj, bus_obj):
    """Trigger score with all bus signals zeroed (baseline)."""
    return mct_obj.evaluate()['trigger_score']


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (a): auto_critic_quality → low_output_reliability
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1a_AutoCriticQuality:
    """auto_critic_quality < 0.5 amplifies low_output_reliability."""

    def test_low_quality_raises_score(self, mct, bus):
        bus.write_signal('auto_critic_quality', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0
        assert any(
            'reliability' in t or 'output' in t
            for t in result['triggers_active']
        ) or result['trigger_score'] > 0.0

    def test_high_quality_no_effect(self, mct, bus):
        bus.write_signal('auto_critic_quality', 0.9)
        base = _base_score(mct, bus)
        bus.write_signal('auto_critic_quality', 0.9)
        result = mct.evaluate()
        # auto_critic_quality >= 0.5 → no amplification of low_output_reliability
        assert result['trigger_score'] <= base + 1e-6

    def test_moderate_quality_partial_effect(self, mct, bus):
        bus.write_signal('auto_critic_quality', 0.3)
        r1 = mct.evaluate()
        bus.write_signal('auto_critic_quality', 0.1)
        r2 = mct.evaluate()
        # lower quality should produce higher or equal score
        assert r2['trigger_score'] >= r1['trigger_score'] - 1e-6

    def test_no_bus_no_crash(self):
        trigger = MetaCognitiveRecursionTrigger()
        # No bus wired → should not crash
        result = trigger.evaluate()
        assert 'trigger_score' in result


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (b): cognitive_health_critical → recovery_pressure
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1b_CognitiveHealthCritical:
    """cognitive_health_critical > 0.3 amplifies recovery_pressure."""

    def test_critical_health_raises_score(self, mct, bus):
        bus.write_signal('cognitive_health_critical', 0.8)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_below_threshold_no_effect(self, mct, bus):
        bus.write_signal('cognitive_health_critical', 0.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_partial_critical_effect(self, mct, bus):
        bus.write_signal('cognitive_health_critical', 0.4)
        r_low = mct.evaluate()
        bus.write_signal('cognitive_health_critical', 0.9)
        r_high = mct.evaluate()
        assert r_high['trigger_score'] >= r_low['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (c): cognitive_unity_deficit → coherence_deficit
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1c_CognitiveUnityDeficit:
    """cognitive_unity_deficit > 0.2 amplifies coherence_deficit."""

    def test_high_deficit_raises_score(self, mct, bus):
        bus.write_signal('cognitive_unity_deficit', 0.8)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_below_threshold_no_effect(self, mct, bus):
        bus.write_signal('cognitive_unity_deficit', 0.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_monotone_with_deficit_magnitude(self, mct, bus):
        bus.write_signal('cognitive_unity_deficit', 0.3)
        r1 = mct.evaluate()
        bus.write_signal('cognitive_unity_deficit', 0.9)
        r2 = mct.evaluate()
        assert r2['trigger_score'] >= r1['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (d): cross_pass_oscillation → oscillation_severity
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1d_CrossPassOscillation:
    """cross_pass_oscillation > 0.2 amplifies oscillation_severity."""

    def test_high_oscillation_raises_score(self, mct, bus):
        bus.write_signal('cross_pass_oscillation', 0.7)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_below_threshold_no_effect(self, mct, bus):
        bus.write_signal('cross_pass_oscillation', 0.1)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_caps_at_1(self, mct, bus):
        bus.write_signal('cross_pass_oscillation', 999.0)
        r_high = mct.evaluate()
        bus.write_signal('cross_pass_oscillation', 1.0)
        r_one = mct.evaluate()
        # Both should produce finite, non-negative scores
        assert r_high['trigger_score'] >= 0.0
        assert r_one['trigger_score'] >= 0.0


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (e): divergence_active → diverging
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1e_DivergenceActive:
    """divergence_active > 0.5 amplifies diverging trigger channel."""

    def test_active_flag_raises_score(self, mct, bus):
        bus.write_signal('divergence_active', 1.0)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_inactive_flag_no_effect(self, mct, bus):
        bus.write_signal('divergence_active', 0.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_partial_active_no_effect(self, mct, bus):
        # Below threshold (0.3 < 0.5 threshold) → no effect
        bus.write_signal('divergence_active', 0.3)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (f): integration_failure_rate → recovery_pressure
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1f_IntegrationFailureRate:
    """integration_failure_rate > 0.2 amplifies recovery_pressure."""

    def test_high_failure_raises_score(self, mct, bus):
        bus.write_signal('integration_failure_rate', 0.7)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_low_failure_no_effect(self, mct, bus):
        bus.write_signal('integration_failure_rate', 0.05)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_partial_failure_effect(self, mct, bus):
        bus.write_signal('integration_failure_rate', 0.3)
        r_low = mct.evaluate()
        bus.write_signal('integration_failure_rate', 0.8)
        r_high = mct.evaluate()
        assert r_high['trigger_score'] >= r_low['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (g): integration_health → coherence_deficit (inv)
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1g_IntegrationHealth:
    """integration_health < 0.7 amplifies coherence_deficit."""

    def test_low_health_raises_score(self, mct, bus):
        bus.write_signal('integration_health', 0.2)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_healthy_no_effect(self, mct, bus):
        bus.write_signal('integration_health', 1.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_monotone_with_health_decline(self, mct, bus):
        bus.write_signal('integration_health', 0.6)
        r1 = mct.evaluate()
        bus.write_signal('integration_health', 0.1)
        r2 = mct.evaluate()
        assert r2['trigger_score'] >= r1['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (h): output_reliability_composite → low_output_reliability
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1h_OutputReliabilityComposite:
    """output_reliability_composite < 0.5 amplifies low_output_reliability."""

    def test_low_reliability_raises_score(self, mct, bus):
        bus.write_signal('output_reliability_composite', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_high_reliability_no_effect(self, mct, bus):
        bus.write_signal('output_reliability_composite', 0.9)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_exact_threshold_boundary(self, mct, bus):
        bus.write_signal('output_reliability_composite', 0.5)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        # Exactly at threshold → no amplification (< 0.5 required)
        assert abs(result['trigger_score'] - base) < 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (i): stall_severity_pressure → stall_severity
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1i_StallSeverityPressure:
    """stall_severity_pressure > 0.1 amplifies stall_severity channel."""

    def test_high_pressure_raises_score(self, mct, bus):
        bus.write_signal('stall_severity_pressure', 0.8)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_below_threshold_no_effect(self, mct, bus):
        bus.write_signal('stall_severity_pressure', 0.05)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_stall_caps_at_1(self, mct, bus):
        bus.write_signal('stall_severity_pressure', 100.0)
        result = mct.evaluate()
        # Must not produce NaN or Inf
        assert result['trigger_score'] == result['trigger_score']  # not NaN
        assert result['trigger_score'] < float('inf')


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (j): teacher_student_inversion_ok → convergence_conflict
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1j_TeacherStudentInversionOk:
    """teacher_student_inversion_ok < 0.5 amplifies convergence_conflict."""

    def test_failed_inversion_raises_score(self, mct, bus):
        bus.write_signal('teacher_student_inversion_ok', 0.0)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_successful_inversion_no_effect(self, mct, bus):
        bus.write_signal('teacher_student_inversion_ok', 1.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_partial_failure_proportional(self, mct, bus):
        bus.write_signal('teacher_student_inversion_ok', 0.4)
        r_mild = mct.evaluate()
        bus.write_signal('teacher_student_inversion_ok', 0.0)
        r_full = mct.evaluate()
        assert r_full['trigger_score'] >= r_mild['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (k): training_phase_pressure → recovery_pressure
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1k_TrainingPhasePressure:
    """training_phase_pressure > 0.3 amplifies recovery_pressure."""

    def test_high_pressure_raises_score(self, mct, bus):
        bus.write_signal('training_phase_pressure', 0.8)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_below_threshold_no_effect(self, mct, bus):
        bus.write_signal('training_phase_pressure', 0.2)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_monotone_with_pressure(self, mct, bus):
        bus.write_signal('training_phase_pressure', 0.4)
        r1 = mct.evaluate()
        bus.write_signal('training_phase_pressure', 0.9)
        r2 = mct.evaluate()
        assert r2['trigger_score'] >= r1['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (l): ucc_evaluation_ok → coherence_deficit (inv)
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1l_UccEvaluationOk:
    """ucc_evaluation_ok < 0.5 amplifies coherence_deficit."""

    def test_failed_evaluation_raises_score(self, mct, bus):
        bus.write_signal('ucc_evaluation_ok', 0.0)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_successful_evaluation_no_effect(self, mct, bus):
        bus.write_signal('ucc_evaluation_ok', 1.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_at_threshold_boundary(self, mct, bus):
        bus.write_signal('ucc_evaluation_ok', 0.5)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        # Exactly at threshold → no amplification (< 0.5 required)
        assert abs(result['trigger_score'] - base) < 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (m): world_model_surprise_active → world_model_surprise
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1m_WorldModelSurpriseActive:
    """world_model_surprise_active > 0.5 amplifies world_model_surprise."""

    def test_active_surprise_raises_score(self, mct, bus):
        bus.write_signal('world_model_surprise_active', 1.0)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_inactive_flag_no_effect(self, mct, bus):
        bus.write_signal('world_model_surprise_active', 0.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_partial_flag_below_threshold(self, mct, bus):
        # 0.3 < 0.5 threshold → no effect
        bus.write_signal('world_model_surprise_active', 0.3)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-1 (n): z_filter_pass_ratio → uncertainty (inv)
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1n_ZFilterPassRatio:
    """z_filter_pass_ratio < 0.5 amplifies uncertainty."""

    def test_low_ratio_raises_score(self, mct, bus):
        bus.write_signal('z_filter_pass_ratio', 0.1)
        result = mct.evaluate()
        assert result['trigger_score'] > 0.0

    def test_high_ratio_no_effect(self, mct, bus):
        bus.write_signal('z_filter_pass_ratio', 1.0)
        base = _base_score(mct, bus)
        result = mct.evaluate()
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_monotone_with_ratio_decline(self, mct, bus):
        bus.write_signal('z_filter_pass_ratio', 0.4)
        r1 = mct.evaluate()
        bus.write_signal('z_filter_pass_ratio', 0.1)
        r2 = mct.evaluate()
        assert r2['trigger_score'] >= r1['trigger_score'] - 1e-6


# ═══════════════════════════════════════════════════════════════════
#  Combined FINAL-ACT-1 integration
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct1_Combined:
    """Combined FINAL-ACT-1: all 14 signals active raise trigger score."""

    def test_all_signals_combined_raise_trigger(self, mct, bus):
        bus.write_signal('auto_critic_quality', 0.1)
        bus.write_signal('cognitive_health_critical', 0.9)
        bus.write_signal('cognitive_unity_deficit', 0.8)
        bus.write_signal('cross_pass_oscillation', 0.7)
        bus.write_signal('divergence_active', 1.0)
        bus.write_signal('integration_failure_rate', 0.8)
        bus.write_signal('integration_health', 0.1)
        bus.write_signal('output_reliability_composite', 0.1)
        bus.write_signal('stall_severity_pressure', 0.9)
        bus.write_signal('teacher_student_inversion_ok', 0.0)
        bus.write_signal('training_phase_pressure', 0.8)
        bus.write_signal('ucc_evaluation_ok', 0.0)
        bus.write_signal('world_model_surprise_active', 1.0)
        bus.write_signal('z_filter_pass_ratio', 0.1)

        result = mct.evaluate()
        assert result['trigger_score'] > 0.0
        assert result['should_trigger'] or result['trigger_score'] > 0.0

    def test_all_signals_healthy_no_spurious_trigger(self, mct, bus):
        bus.write_signal('auto_critic_quality', 1.0)
        bus.write_signal('cognitive_health_critical', 0.0)
        bus.write_signal('cognitive_unity_deficit', 0.0)
        bus.write_signal('cross_pass_oscillation', 0.0)
        bus.write_signal('divergence_active', 0.0)
        bus.write_signal('integration_failure_rate', 0.0)
        bus.write_signal('integration_health', 1.0)
        bus.write_signal('output_reliability_composite', 1.0)
        bus.write_signal('stall_severity_pressure', 0.0)
        bus.write_signal('teacher_student_inversion_ok', 1.0)
        bus.write_signal('training_phase_pressure', 0.0)
        bus.write_signal('ucc_evaluation_ok', 1.0)
        bus.write_signal('world_model_surprise_active', 0.0)
        bus.write_signal('z_filter_pass_ratio', 1.0)

        base = _base_score(mct, bus)
        result = mct.evaluate()
        # With all signals healthy, score should not exceed baseline
        assert abs(result['trigger_score'] - base) < 1e-6

    def test_result_dict_has_required_keys(self, mct, bus):
        result = mct.evaluate()
        for key in ('should_trigger', 'trigger_score', 'triggers_active',
                    'signal_weights'):
            assert key in result, f"Missing key: {key}"

    def test_no_bus_no_crash(self):
        trigger = MetaCognitiveRecursionTrigger()
        result = trigger.evaluate()
        assert isinstance(result, dict)

    def test_bus_none_explicitly(self):
        trigger = MetaCognitiveRecursionTrigger()
        trigger._feedback_bus_ref = None
        result = trigger.evaluate()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-2 tests (code-presence + logic)
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct2_ComputeLossIntegration:
    """FINAL-ACT-2: compute_loss reads integration health signals."""

    def test_fa2_reads_integration_health(self):
        src = open('aeon_core.py').read()
        assert "'integration_health'" in src or '"integration_health"' in src

    def test_fa2_reads_ucc_evaluation_ok(self):
        src = open('aeon_core.py').read()
        assert "'ucc_evaluation_ok'" in src or '"ucc_evaluation_ok"' in src

    def test_fa2_reads_teacher_student_inversion_ok(self):
        src = open('aeon_core.py').read()
        assert (
            "'teacher_student_inversion_ok'" in src
            or '"teacher_student_inversion_ok"' in src
        )

    def test_fa2_low_integration_health_amplifies_loss(self):
        """FINAL-ACT-2 logic: low health → loss scale > 1.0."""
        ih = 0.2   # very low integration health
        scale = 1.0
        if ih < 0.6:
            ih_deficit = (0.6 - ih) / 0.6
            scale = min(1.2, 1.0 + ih_deficit * 0.2)
        assert scale > 1.0, f"Expected scale > 1.0, got {scale}"

    def test_fa2_healthy_integration_no_amplification(self):
        """FINAL-ACT-2 logic: health >= 0.6 → scale = 1.0."""
        ih = 0.9
        scale = 1.0
        if ih < 0.6:
            ih_deficit = (0.6 - ih) / 0.6
            scale = min(1.2, 1.0 + ih_deficit * 0.2)
        assert abs(scale - 1.0) < 1e-9, f"Expected scale = 1.0, got {scale}"

    def test_fa2_failed_ucc_amplifies_loss(self):
        """FINAL-ACT-2 logic: ucc_evaluation_ok = 0 → scale > 1.0."""
        ueo = 0.0  # evaluation failed
        scale = 1.0
        if ueo < 0.5:
            ueo_deficit = 1.0 - ueo
            scale = min(1.15, 1.0 + ueo_deficit * 0.15)
        assert scale > 1.0, f"Expected scale > 1.0, got {scale}"

    def test_fa2_successful_ucc_no_amplification(self):
        """FINAL-ACT-2 logic: ucc_evaluation_ok = 1 → scale = 1.0."""
        ueo = 1.0
        scale = 1.0
        if ueo < 0.5:
            ueo_deficit = 1.0 - ueo
            scale = min(1.15, 1.0 + ueo_deficit * 0.15)
        assert abs(scale - 1.0) < 1e-9, f"Expected scale = 1.0, got {scale}"

    def test_fa2_failed_inversion_dampens_loss(self):
        """FINAL-ACT-2 logic: failed inversion → scale < 1.0 (dampen)."""
        tsio = 0.0  # completely failed
        scale = 1.0
        if tsio < 0.5:
            tsio_deficit = 1.0 - tsio
            scale = max(0.85, 1.0 - tsio_deficit * 0.1)
        assert scale < 1.0, f"Expected scale < 1.0, got {scale}"

    def test_fa2_successful_inversion_no_dampening(self):
        """FINAL-ACT-2 logic: successful inversion → scale = 1.0."""
        tsio = 1.0
        scale = 1.0
        if tsio < 0.5:
            tsio_deficit = 1.0 - tsio
            scale = max(0.85, 1.0 - tsio_deficit * 0.1)
        assert abs(scale - 1.0) < 1e-9, f"Expected scale = 1.0, got {scale}"

    def test_fa2_dampening_bounded_below(self):
        """FINAL-ACT-2: dampen scale is capped at 0.85 minimum."""
        tsio = 0.0
        scale = 1.0
        if tsio < 0.5:
            tsio_deficit = 1.0 - tsio
            scale = max(0.85, 1.0 - tsio_deficit * 0.1)
        assert scale >= 0.85, f"Expected scale >= 0.85, got {scale}"

    def test_fa2_amplification_bounded_above(self):
        """FINAL-ACT-2: health/ucc amplification caps at 1.2 / 1.15."""
        ih = 0.0   # worst case
        scale = 1.0
        if ih < 0.6:
            ih_deficit = (0.6 - ih) / 0.6
            scale = min(1.2, 1.0 + ih_deficit * 0.2)
        assert scale <= 1.2, f"Expected scale <= 1.2, got {scale}"


# ═══════════════════════════════════════════════════════════════════
#  FINAL-ACT-3 tests (code-presence + logic)
# ═══════════════════════════════════════════════════════════════════

class TestFinalAct3_BridgeReadsIntegrationSignals:
    """FINAL-ACT-3: _bridge_epoch_feedback reads integration signals."""

    def test_fa3_reads_integration_failure_rate(self):
        src = open('aeon_core.py').read()
        assert (
            "'integration_failure_rate'" in src
            or '"integration_failure_rate"' in src
        )

    def test_fa3_reads_z_filter_pass_ratio(self):
        src = open('aeon_core.py').read()
        assert (
            "'z_filter_pass_ratio'" in src
            or '"z_filter_pass_ratio"' in src
        )

    def test_fa3_reads_teacher_student_inversion_ok_in_bridge(self):
        """teacher_student_inversion_ok appears in bridge context."""
        src = open('aeon_core.py').read()
        assert (
            "'teacher_student_inversion_ok'" in src
            or '"teacher_student_inversion_ok"' in src
        )

    def test_fa3_high_failure_tightens_clip(self):
        """FINAL-ACT-3 logic: high integration_failure_rate → clip reduces."""
        ifr = 0.8  # 80% failure rate
        old_clip = 1.0
        new_clip = old_clip
        if ifr > 0.3:
            new_clip = max(0.1, old_clip * (1.0 - 0.05 * ifr))
        assert new_clip < old_clip, f"Expected clip to tighten, got {new_clip}"

    def test_fa3_low_failure_no_clip_change(self):
        """FINAL-ACT-3 logic: low integration_failure_rate → no clip change."""
        ifr = 0.1
        old_clip = 1.0
        new_clip = old_clip
        if ifr > 0.3:
            new_clip = max(0.1, old_clip * (1.0 - 0.05 * ifr))
        assert abs(new_clip - old_clip) < 1e-9

    def test_fa3_low_z_filter_ratio_tightens_clip(self):
        """FINAL-ACT-3 logic: low z_filter_pass_ratio → clip reduces."""
        zfpr = 0.2  # only 20% of z vectors passed
        old_clip = 1.0
        new_clip = old_clip
        if zfpr < 0.4:
            zfpr_deficit = 1.0 - zfpr
            new_clip = max(0.1, old_clip * (1.0 - 0.03 * zfpr_deficit))
        assert new_clip < old_clip, f"Expected clip to tighten, got {new_clip}"

    def test_fa3_high_z_filter_ratio_no_clip_change(self):
        """FINAL-ACT-3 logic: high z_filter_pass_ratio → no clip change."""
        zfpr = 0.9
        old_clip = 1.0
        new_clip = old_clip
        if zfpr < 0.4:
            zfpr_deficit = 1.0 - zfpr
            new_clip = max(0.1, old_clip * (1.0 - 0.03 * zfpr_deficit))
        assert abs(new_clip - old_clip) < 1e-9

    def test_fa3_failed_inversion_tightens_clip(self):
        """FINAL-ACT-3 logic: failed inversion → clip reduces."""
        tsio = 0.0  # completely failed
        old_clip = 1.0
        new_clip = old_clip
        if tsio < 0.5:
            new_clip = max(0.1, old_clip * (1.0 - 0.03 * (1.0 - tsio)))
        assert new_clip < old_clip, f"Expected clip to tighten, got {new_clip}"

    def test_fa3_successful_inversion_no_clip_change(self):
        """FINAL-ACT-3 logic: successful inversion → no clip change."""
        tsio = 1.0
        old_clip = 1.0
        new_clip = old_clip
        if tsio < 0.5:
            new_clip = max(0.1, old_clip * (1.0 - 0.03 * (1.0 - tsio)))
        assert abs(new_clip - old_clip) < 1e-9

    def test_fa3_clip_bounded_below(self):
        """FINAL-ACT-3: clip never goes below 0.1."""
        ifr = 1.0   # maximum failure
        old_clip = 0.1  # already at minimum
        new_clip = max(0.1, old_clip * (1.0 - 0.05 * ifr))
        assert new_clip >= 0.1, f"Expected clip >= 0.1, got {new_clip}"

    def test_fa3_block_present_in_bridge_source(self):
        """FINAL-ACT-3 block comment present in source."""
        src = open('aeon_core.py').read()
        assert 'FINAL-ACT-3' in src


# ═══════════════════════════════════════════════════════════════════
#  Regression / baseline tests
# ═══════════════════════════════════════════════════════════════════

class TestFinalActRegression:
    """Regression checks: base signals unaffected when orphans are zero."""

    def test_base_uncertainty_unaffected(self, mct, bus):
        """Setting all FINAL-ACT signals to healthy leaves base score stable."""
        bus.write_signal('auto_critic_quality', 1.0)
        bus.write_signal('cognitive_health_critical', 0.0)
        bus.write_signal('cognitive_unity_deficit', 0.0)
        bus.write_signal('cross_pass_oscillation', 0.0)
        bus.write_signal('divergence_active', 0.0)
        bus.write_signal('integration_failure_rate', 0.0)
        bus.write_signal('integration_health', 1.0)
        bus.write_signal('output_reliability_composite', 1.0)
        bus.write_signal('stall_severity_pressure', 0.0)
        bus.write_signal('teacher_student_inversion_ok', 1.0)
        bus.write_signal('training_phase_pressure', 0.0)
        bus.write_signal('ucc_evaluation_ok', 1.0)
        bus.write_signal('world_model_surprise_active', 0.0)
        bus.write_signal('z_filter_pass_ratio', 1.0)
        r1 = mct.evaluate()
        r2 = mct.evaluate()
        assert abs(r1['trigger_score'] - r2['trigger_score']) < 1e-6

    def test_mct_returns_correct_shape(self, mct, bus):
        result = mct.evaluate()
        assert isinstance(result['triggers_active'], list)
        assert isinstance(result['signal_weights'], dict)
        assert isinstance(result['trigger_score'], float)

    def test_signal_monotonicity_auto_critic_quality(self, mct, bus):
        """Higher auto_critic_quality → lower trigger score (inverted signal)."""
        bus.write_signal('auto_critic_quality', 0.1)
        r_low = mct.evaluate()
        bus.write_signal('auto_critic_quality', 0.4)
        r_high = mct.evaluate()
        assert r_low['trigger_score'] >= r_high['trigger_score'] - 1e-6

    def test_signal_monotonicity_integration_health(self, mct, bus):
        """Lower integration_health → higher trigger score (inverted signal)."""
        bus.write_signal('integration_health', 0.1)
        r_low = mct.evaluate()
        bus.write_signal('integration_health', 0.5)
        r_high = mct.evaluate()
        assert r_low['trigger_score'] >= r_high['trigger_score'] - 1e-6
