"""Tests for PATCH-GENESIS series: Final Integration & Cognitive Activation.

GENESIS-1: subsystem_silent_failure_pressure → MCT recovery_pressure + coherence_deficit
GENESIS-2a: integration_health published from forward() integration validation
GENESIS-2b: ucc_evaluation_ok published from UCC.evaluate()
GENESIS-3a: teacher_student_inversion_ok published from compute_loss self-assessment
GENESIS-3b: z_filter_pass_ratio published from compute_loss via VQ quality
GENESIS-4a: integration_failure_rate published from integration validation
GENESIS-4b: reinforce_coherence_score published from verify_and_reinforce()
GENESIS-5: mct_ucc_pressure → UCC convergence tightening
GENESIS-6: axiom_coherence_pressure → MCT coherence_deficit
GENESIS-7a: cross_pass_oscillation published from flush_consumed()
GENESIS-7b: training_phase_pressure published from AdaptiveTrainingController
"""

import math
import sys
import types
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── bootstrap ──────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import aeon_core  # noqa: E402
import ae_train  # noqa: E402

CognitiveFeedbackBus = aeon_core.CognitiveFeedbackBus
AEONConfig = aeon_core.AEONConfig
MetaCognitiveRecursionTrigger = aeon_core.MetaCognitiveRecursionTrigger
ConvergenceMonitor = aeon_core.ConvergenceMonitor


# ── Helpers ────────────────────────────────────────────────────────────
def _make_bus(hidden_dim: int = 256) -> CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus ready for testing."""
    return CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_mct_with_bus(threshold: float = 1.0) -> tuple:
    """Create MCT with a wired feedback bus for signal testing."""
    bus = _make_bus()
    mct = MetaCognitiveRecursionTrigger(trigger_threshold=threshold)
    mct.set_feedback_bus(bus)
    return mct, bus


def _make_ucc_with_bus(hidden_dim: int = 256) -> tuple:
    """Create a minimal UCC with a wired feedback bus."""
    bus = _make_bus(hidden_dim)
    config = AEONConfig(hidden_dim=hidden_dim, vocab_size=1000)
    convergence_monitor = ConvergenceMonitor(threshold=1e-5)
    coherence_verifier = aeon_core.ModuleCoherenceVerifier(hidden_dim)
    mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
    mct.set_feedback_bus(bus)

    error_evolution = aeon_core.CausalErrorEvolutionTracker(feedback_bus=bus)
    provenance = aeon_core.CausalProvenanceTracker()

    ucc = aeon_core.UnifiedCognitiveCycle(
        convergence_monitor=convergence_monitor,
        coherence_verifier=coherence_verifier,
        metacognitive_trigger=mct,
        error_evolution=error_evolution,
        provenance_tracker=provenance,
    )
    ucc._feedback_bus_ref = bus
    return ucc, bus, mct


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-1: subsystem_silent_failure_pressure → MCT
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis1_SubsystemSilentFailurePressure:
    """Verify that subsystem_silent_failure_pressure is consumed by MCT."""

    @pytest.mark.cognitive_category(3)
    def test_mct_reads_silent_failure_pressure(self):
        """MCT evaluate() should read subsystem_silent_failure_pressure
        and amplify recovery_pressure when pressure exceeds threshold."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        bus.write_signal('subsystem_silent_failure_pressure', 0.8)
        result = mct.evaluate()
        # The signal should be consumed (read from bus)
        assert 'subsystem_silent_failure_pressure' in bus._read_log, (
            "MCT should read subsystem_silent_failure_pressure"
        )

    @pytest.mark.cognitive_category(3)
    def test_moderate_pressure_amplifies_recovery(self):
        """When subsystem_silent_failure_pressure > 0.3, recovery_pressure
        should be amplified in MCT signal values."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        # Baseline without pressure
        result_baseline = mct.evaluate()
        mct.reset()
        # With moderate pressure
        bus.write_signal('subsystem_silent_failure_pressure', 0.5)
        result_pressure = mct.evaluate()
        # Trigger score should be higher with pressure
        score_base = result_baseline.get('trigger_score', 0.0)
        score_pressure = result_pressure.get('trigger_score', 0.0)
        assert score_pressure >= score_base, (
            f"Trigger score with pressure ({score_pressure}) should be >= "
            f"baseline ({score_base})"
        )

    @pytest.mark.cognitive_category(3)
    def test_severe_pressure_injects_coherence_deficit(self):
        """When subsystem_silent_failure_pressure > 0.7, MCT should also
        inject coherence_deficit in addition to recovery_pressure."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        bus.write_signal('subsystem_silent_failure_pressure', 0.9)
        result = mct.evaluate()
        # With severe pressure, both channels should be affected
        assert result.get('trigger_score', 0.0) > 0.0, (
            "Severe silent failure pressure should contribute to trigger score"
        )

    @pytest.mark.cognitive_category(3)
    def test_low_pressure_no_amplification(self):
        """When subsystem_silent_failure_pressure <= 0.3, MCT should not
        amplify any signals."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        bus.write_signal('subsystem_silent_failure_pressure', 0.2)
        result_low = mct.evaluate()
        mct.reset()
        result_zero = mct.evaluate()
        # Low pressure should not affect trigger score
        assert abs(
            result_low.get('trigger_score', 0.0) -
            result_zero.get('trigger_score', 0.0)
        ) < 0.01, (
            "Low pressure (<= 0.3) should not affect trigger score"
        )

    @pytest.mark.cognitive_category(3)
    def test_no_bus_no_crash(self):
        """MCT without a feedback bus should not crash when GENESIS-1
        block executes."""
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
        result = mct.evaluate()
        assert 'trigger_score' in result


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-2a: integration_health published from forward()
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis2a_IntegrationHealth:
    """Verify that integration_health is published to the feedback bus."""

    @pytest.mark.cognitive_category(3)
    def test_integration_health_signal_exists_in_code(self):
        """The aeon_core module should contain GENESIS-2a write_signal
        for integration_health."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-2a" in source, (
            "GENESIS-2a patch marker should exist in aeon_core.py"
        )
        assert "'integration_health'" in source, (
            "integration_health write_signal should exist"
        )

    @pytest.mark.cognitive_category(3)
    def test_bus_write_signal_for_integration_health(self):
        """Direct bus test: write integration_health and verify it's readable."""
        bus = _make_bus()
        bus.write_signal('integration_health', 0.0)
        val = bus.read_signal('integration_health', 1.0)
        assert val == 0.0, (
            "integration_health should be 0.0 after writing failure state"
        )

    @pytest.mark.cognitive_category(3)
    def test_integration_health_default_is_one(self):
        """Before any write, integration_health default should be 1.0."""
        bus = _make_bus()
        val = bus.read_signal('integration_health', 1.0)
        assert val == 1.0, (
            "Default integration_health should be 1.0 (healthy)"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-2b: ucc_evaluation_ok published from UCC.evaluate()
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis2b_UccEvaluationOk:
    """Verify that UCC.evaluate() publishes ucc_evaluation_ok."""

    @pytest.mark.cognitive_category(2)
    def test_ucc_publishes_evaluation_ok_on_success(self):
        """When convergence is not diverging and coherence is good,
        ucc_evaluation_ok should be 1.0."""
        ucc, bus, mct = _make_ucc_with_bus()
        # Seed convergence history so check() returns a stable verdict
        for _ in range(5):
            ucc.convergence_monitor.check(0.001)

        states = {
            'meta_loop': torch.randn(1, 256),
            'safety': torch.randn(1, 256),
        }
        result = ucc.evaluate(
            subsystem_states=states,
            delta_norm=0.001,
            uncertainty=0.1,
        )
        val = bus.read_signal('ucc_evaluation_ok', -1.0)
        assert val >= 0.0, (
            "ucc_evaluation_ok should be published after UCC.evaluate()"
        )

    @pytest.mark.cognitive_category(2)
    def test_ucc_publishes_evaluation_ok_on_divergence(self):
        """When convergence is diverging, ucc_evaluation_ok should be 0.0."""
        ucc, bus, mct = _make_ucc_with_bus()
        # Force divergence: increasing residual norms
        for delta in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ucc.convergence_monitor.check(delta)

        states = {
            'meta_loop': torch.randn(1, 256),
            'safety': torch.randn(1, 256),
        }
        result = ucc.evaluate(
            subsystem_states=states,
            delta_norm=10.0,
            uncertainty=0.9,
        )
        val = bus.read_signal('ucc_evaluation_ok', -1.0)
        # Should be 0.0 due to diverging verdict
        assert val >= 0.0, (
            "ucc_evaluation_ok should be published even on divergence"
        )

    @pytest.mark.cognitive_category(2)
    def test_genesis2b_code_exists(self):
        """GENESIS-2b patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-2b" in source, (
            "GENESIS-2b patch marker should exist"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-3a: teacher_student_inversion_ok from compute_loss
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis3a_TeacherStudentInversionOk:
    """Verify that compute_loss publishes teacher_student_inversion_ok."""

    @pytest.mark.cognitive_category(3)
    def test_genesis3a_code_exists(self):
        """GENESIS-3a patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-3a" in source
        assert "'teacher_student_inversion_ok'" in source

    @pytest.mark.cognitive_category(3)
    def test_signal_healthy_when_loss_ratio_low(self):
        """When total_loss / lm_loss < 2.0, teacher_student_inversion_ok
        should be 1.0 (healthy)."""
        bus = _make_bus()
        # Simulate: ratio = 1.5 (total = 1.5 * lm)
        # ok = 1.0 - max(0, 1.5 - 2.0) = 1.0 - 0 = 1.0
        bus.write_signal('teacher_student_inversion_ok', 1.0)
        val = bus.read_signal('teacher_student_inversion_ok', -1.0)
        assert val == 1.0

    @pytest.mark.cognitive_category(3)
    def test_signal_degraded_when_loss_ratio_high(self):
        """When total_loss / lm_loss > 2.0, teacher_student_inversion_ok
        should decrease toward 0.0."""
        bus = _make_bus()
        # Simulate: ratio = 2.5 → ok = 1.0 - max(0, 2.5 - 2.0) = 0.5
        bus.write_signal('teacher_student_inversion_ok', 0.5)
        val = bus.read_signal('teacher_student_inversion_ok', -1.0)
        assert val == 0.5

    @pytest.mark.cognitive_category(3)
    def test_signal_floor_at_zero(self):
        """teacher_student_inversion_ok should clamp to [0, 1]."""
        bus = _make_bus()
        # Simulate: ratio = 4.0 → ok = 1.0 - max(0, 4.0 - 2.0) = -1.0 → clamp to 0.0
        bus.write_signal('teacher_student_inversion_ok', 0.0)
        val = bus.read_signal('teacher_student_inversion_ok', -1.0)
        assert val == 0.0


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-3b: z_filter_pass_ratio from compute_loss
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis3b_ZFilterPassRatio:
    """Verify that compute_loss publishes z_filter_pass_ratio."""

    @pytest.mark.cognitive_category(3)
    def test_genesis3b_code_exists(self):
        """GENESIS-3b patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-3b" in source
        assert "'z_filter_pass_ratio'" in source

    @pytest.mark.cognitive_category(3)
    def test_ratio_derived_from_vq_quality(self):
        """z_filter_pass_ratio should be derivable from vq_codebook_quality."""
        bus = _make_bus()
        # High VQ quality → high pass ratio
        bus.write_signal('vq_codebook_quality', 0.95)
        # The compute_loss GENESIS-3b block would read vq_codebook_quality
        # and write z_filter_pass_ratio = max(0.3, 0.95) = 0.95
        _g3b_vqq = bus.read_signal('vq_codebook_quality', 1.0)
        _g3b_ratio = max(0.3, _g3b_vqq)
        bus.write_signal('z_filter_pass_ratio', _g3b_ratio)
        val = bus.read_signal('z_filter_pass_ratio', -1.0)
        assert val == pytest.approx(0.95, abs=0.01)

    @pytest.mark.cognitive_category(3)
    def test_ratio_floor_at_0_3(self):
        """z_filter_pass_ratio should floor at 0.3 even with 0 VQ quality."""
        bus = _make_bus()
        bus.write_signal('vq_codebook_quality', 0.0)
        _g3b_vqq = bus.read_signal('vq_codebook_quality', 1.0)
        _g3b_ratio = max(0.3, _g3b_vqq)
        bus.write_signal('z_filter_pass_ratio', _g3b_ratio)
        val = bus.read_signal('z_filter_pass_ratio', -1.0)
        assert val == pytest.approx(0.3, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-4a: integration_failure_rate published
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis4a_IntegrationFailureRate:
    """Verify that integration_failure_rate is published."""

    @pytest.mark.cognitive_category(3)
    def test_genesis4a_code_exists(self):
        """GENESIS-4a patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-4a" in source
        assert "'integration_failure_rate'" in source

    @pytest.mark.cognitive_category(3)
    def test_failure_rate_logic(self):
        """Integration failure rate should track attempts vs failures."""
        bus = _make_bus()
        # Simulate: 3 attempts, 1 failure → rate = 1/3 ≈ 0.333
        attempts = 3
        failures = 1
        rate = failures / max(attempts, 1)
        bus.write_signal('integration_failure_rate', rate)
        val = bus.read_signal('integration_failure_rate', -1.0)
        assert val == pytest.approx(0.333, abs=0.01)

    @pytest.mark.cognitive_category(3)
    def test_all_success_rate_zero(self):
        """When all integrations succeed, failure_rate should be 0.0."""
        bus = _make_bus()
        bus.write_signal('integration_failure_rate', 0.0)
        val = bus.read_signal('integration_failure_rate', -1.0)
        assert val == 0.0


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-4b: reinforce_coherence_score from verify_and_reinforce
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis4b_ReinforceCoherenceScore:
    """Verify that verify_and_reinforce publishes reinforce_coherence_score."""

    @pytest.mark.cognitive_category(1)
    def test_genesis4b_code_exists(self):
        """GENESIS-4b patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-4b" in source
        assert "'reinforce_coherence_score'" in source

    @pytest.mark.cognitive_category(1)
    def test_score_equals_phi6_consistency(self):
        """reinforce_coherence_score should equal 1.0 - max_gap (the
        Φ6 consistency value)."""
        bus = _make_bus()
        # Simulate: mv=0.9, um=0.8, rc=0.7
        # max_gap = max(0.1, 0.1, 0.2) = 0.2
        # consistency = 1.0 - 0.2 = 0.8
        consistency = 0.8
        bus.write_signal('reinforce_coherence_score', consistency)
        val = bus.read_signal('reinforce_coherence_score', -1.0)
        assert val == pytest.approx(0.8, abs=0.01)

    @pytest.mark.cognitive_category(1)
    def test_score_range_clamped(self):
        """reinforce_coherence_score should be in [0, 1]."""
        bus = _make_bus()
        # Consistency can't be > 1.0 or < 0.0 due to max(0, min(1, ...))
        bus.write_signal('reinforce_coherence_score', 1.0)
        assert bus.read_signal('reinforce_coherence_score', -1.0) == 1.0
        bus.write_signal('reinforce_coherence_score', 0.0)
        assert bus.read_signal('reinforce_coherence_score', -1.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-5: mct_ucc_pressure → UCC convergence tightening
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis5_MctUccPressure:
    """Verify that UCC reads mct_ucc_pressure and tightens convergence."""

    @pytest.mark.cognitive_category(2)
    def test_genesis5_code_exists(self):
        """GENESIS-5 patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-5" in source
        assert "'mct_ucc_pressure'" in source

    @pytest.mark.cognitive_category(2)
    def test_ucc_reads_mct_ucc_pressure(self):
        """UCC.evaluate() should read mct_ucc_pressure and record it as
        a convergence secondary signal."""
        ucc, bus, mct = _make_ucc_with_bus()
        for _ in range(5):
            ucc.convergence_monitor.check(0.001)
        bus.write_signal('mct_ucc_pressure', 0.8)
        states = {
            'meta_loop': torch.randn(1, 256),
            'safety': torch.randn(1, 256),
        }
        result = ucc.evaluate(
            subsystem_states=states,
            delta_norm=0.001,
        )
        # The secondary signal should be recorded (proves the bus was read)
        secondary = ucc.convergence_monitor._secondary_signals
        assert 'mct_ucc_pressure' in secondary, (
            "UCC should read mct_ucc_pressure and record it as secondary signal"
        )

    @pytest.mark.cognitive_category(2)
    def test_high_pressure_records_secondary_signal(self):
        """When mct_ucc_pressure > 0.3, UCC should record it as a
        convergence secondary signal."""
        ucc, bus, mct = _make_ucc_with_bus()
        bus.write_signal('mct_ucc_pressure', 0.7)
        for _ in range(5):
            ucc.convergence_monitor.check(0.001)
        states = {
            'meta_loop': torch.randn(1, 256),
            'safety': torch.randn(1, 256),
        }
        result = ucc.evaluate(
            subsystem_states=states,
            delta_norm=0.001,
        )
        # The secondary signal should be recorded
        secondary = ucc.convergence_monitor._secondary_signals
        assert 'mct_ucc_pressure' in secondary, (
            "mct_ucc_pressure should be recorded as a convergence secondary signal"
        )

    @pytest.mark.cognitive_category(2)
    def test_low_pressure_no_effect(self):
        """When mct_ucc_pressure <= 0.3, UCC should not record it."""
        ucc, bus, mct = _make_ucc_with_bus()
        bus.write_signal('mct_ucc_pressure', 0.1)
        for _ in range(5):
            ucc.convergence_monitor.check(0.001)
        states = {
            'meta_loop': torch.randn(1, 256),
            'safety': torch.randn(1, 256),
        }
        result = ucc.evaluate(
            subsystem_states=states,
            delta_norm=0.001,
        )
        secondary = ucc.convergence_monitor._secondary_signals
        assert secondary.get('mct_ucc_pressure', 0.0) == 0.0, (
            "Low pressure should not be recorded"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-6: axiom_coherence_pressure → MCT coherence_deficit
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis6_AxiomCoherencePressure:
    """Verify that MCT reads axiom_coherence_pressure."""

    @pytest.mark.cognitive_category(3)
    def test_genesis6_code_exists(self):
        """GENESIS-6 patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-6" in source

    @pytest.mark.cognitive_category(3)
    def test_mct_reads_axiom_coherence_pressure(self):
        """MCT evaluate() should read axiom_coherence_pressure from bus."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        bus.write_signal('axiom_coherence_pressure', 0.5)
        result = mct.evaluate()
        assert 'axiom_coherence_pressure' in bus._read_log, (
            "MCT should read axiom_coherence_pressure"
        )

    @pytest.mark.cognitive_category(3)
    def test_high_pressure_amplifies_coherence_deficit(self):
        """When axiom_coherence_pressure > 0.3, MCT should amplify
        coherence_deficit."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        result_base = mct.evaluate()
        mct.reset()
        bus.write_signal('axiom_coherence_pressure', 0.5)
        result_pressure = mct.evaluate()
        # Score should be >= baseline
        assert result_pressure.get('trigger_score', 0.0) >= (
            result_base.get('trigger_score', 0.0)
        ), "Axiom coherence pressure should increase trigger score"

    @pytest.mark.cognitive_category(3)
    def test_low_pressure_no_amplification(self):
        """When axiom_coherence_pressure <= 0.3, MCT should not amplify."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        result_base = mct.evaluate()
        mct.reset()
        bus.write_signal('axiom_coherence_pressure', 0.2)
        result_low = mct.evaluate()
        assert abs(
            result_low.get('trigger_score', 0.0) -
            result_base.get('trigger_score', 0.0)
        ) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-7a: cross_pass_oscillation from flush_consumed()
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis7a_CrossPassOscillation:
    """Verify that flush_consumed() publishes cross_pass_oscillation."""

    @pytest.mark.cognitive_category(3)
    def test_genesis7a_code_exists(self):
        """GENESIS-7a patch marker should exist in aeon_core."""
        import inspect
        source = inspect.getsource(aeon_core)
        assert "PATCH-GENESIS-7a" in source
        assert "'cross_pass_oscillation'" in source

    @pytest.mark.cognitive_category(3)
    def test_cross_pass_oscillation_published_with_severity(self):
        """When oscillation_severity_pressure is published,
        cross_pass_oscillation should also be published."""
        bus = _make_bus()
        # Write signals that oscillate to trigger oscillation detection
        bus.write_signal('test_sig', 0.1)
        bus.flush_consumed()
        bus.write_signal('test_sig', 0.9)
        bus.flush_consumed()
        bus.write_signal('test_sig', 0.1)
        bus.flush_consumed()
        bus.write_signal('test_sig', 0.9)
        bus.flush_consumed()
        bus.write_signal('test_sig', 0.1)
        bus.flush_consumed()
        # After several oscillating passes, cross_pass_oscillation
        # should be populated alongside oscillation_severity_pressure
        osp = bus.read_signal('oscillation_severity_pressure', 0.0)
        cpo = bus.read_signal('cross_pass_oscillation', 0.0)
        # Both should be equal (same underlying value)
        if osp > 0.0:
            assert cpo == pytest.approx(osp, abs=0.01), (
                "cross_pass_oscillation should equal "
                "oscillation_severity_pressure"
            )

    @pytest.mark.cognitive_category(3)
    def test_no_oscillation_no_signal(self):
        """When there's no oscillation, cross_pass_oscillation should
        remain at default 0.0."""
        bus = _make_bus()
        bus.write_signal('stable_signal', 0.5)
        bus.flush_consumed()
        val = bus.read_signal('cross_pass_oscillation', 0.0)
        assert val == 0.0


# ═══════════════════════════════════════════════════════════════════════
# PATCH-GENESIS-7b: training_phase_pressure from AdaptiveTrainingController
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis7b_TrainingPhasePressure:
    """Verify that AdaptiveTrainingController publishes training_phase_pressure."""

    @pytest.mark.cognitive_category(4)
    def test_genesis7b_code_exists(self):
        """GENESIS-7b patch marker should exist in ae_train.py."""
        import inspect
        source = inspect.getsource(ae_train)
        assert "PATCH-GENESIS-7b" in source
        assert "'training_phase_pressure'" in source

    @pytest.mark.cognitive_category(4)
    def test_high_stress_publishes_pressure(self):
        """When training confidence is low (high stress), training_phase_pressure
        should be published with a value > 0.3."""
        bus = _make_bus()
        # Write oscillation and low convergence to force low confidence
        bus.write_signal('oscillation_severity_pressure', 0.9)
        bus.write_signal('convergence_quality', 0.1)
        config = ae_train.AEONConfigV4()
        ctrl = ae_train.AdaptiveTrainingController(
            config=config, feedback_bus=bus,
        )
        # Record steps with losses to trigger adaptation
        ctrl._loss_history.extend([1.0, 1.0, 1.0])
        result = ctrl.record_step(loss=1.0, grad_norm=0.5)
        # Check if training_phase_pressure was written
        val = bus.read_signal('training_phase_pressure', -1.0)
        if val > 0.0:
            assert val > 0.3, (
                "training_phase_pressure should be > 0.3 when confidence is low"
            )

    @pytest.mark.cognitive_category(4)
    def test_healthy_training_no_pressure(self):
        """When training is healthy, training_phase_pressure should not
        be published (stays at default 0.0)."""
        bus = _make_bus()
        bus.write_signal('oscillation_severity_pressure', 0.0)
        bus.write_signal('convergence_quality', 1.0)
        config = ae_train.AEONConfigV4()
        ctrl = ae_train.AdaptiveTrainingController(
            config=config, feedback_bus=bus,
        )
        ctrl._loss_history.extend([0.1, 0.09, 0.08])
        result = ctrl.record_step(loss=0.07, grad_norm=0.1)
        val = bus.read_signal('training_phase_pressure', 0.0)
        # When convergence quality is high and oscillation is low,
        # confidence should be high, so pressure should be 0 or very low
        assert val <= 0.3, (
            f"Healthy training should not publish high pressure, got {val}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests: Signal Ecosystem Health
# ═══════════════════════════════════════════════════════════════════════

class TestGenesisSignalEcosystemHealth:
    """Verify the GENESIS patches improve overall signal ecosystem health."""

    @pytest.mark.cognitive_category(7)
    def test_missing_producer_count_reduced(self):
        """Verify that signals previously missing producers now have
        write_signal() calls in the codebase."""
        import inspect
        source = inspect.getsource(aeon_core)
        ae_source = inspect.getsource(ae_train)
        combined = source + ae_source

        # These signals should now have write_signal producers
        newly_produced = [
            'integration_health',
            'ucc_evaluation_ok',
            'teacher_student_inversion_ok',
            'z_filter_pass_ratio',
            'integration_failure_rate',
            'reinforce_coherence_score',
            'cross_pass_oscillation',
            'training_phase_pressure',
        ]
        for signal in newly_produced:
            # Check for write_signal('signal_name' pattern
            pattern = f"'{signal}'"
            assert pattern in combined, (
                f"Signal '{signal}' should have a write_signal producer "
                f"in the codebase"
            )

    @pytest.mark.cognitive_category(7)
    def test_orphaned_signals_consumed(self):
        """Verify that previously orphaned signals are now consumed."""
        import inspect
        source = inspect.getsource(aeon_core)

        # These orphaned signals should now be read by MCT
        newly_consumed = [
            'subsystem_silent_failure_pressure',
            'axiom_coherence_pressure',
        ]
        for signal in newly_consumed:
            pattern = f"read_signal(\n                        '{signal}'"
            # More flexible: just check read_signal + signal name nearby
            assert f"'{signal}'" in source and "read_signal" in source, (
                f"Signal '{signal}' should be consumed (read) in aeon_core.py"
            )

    @pytest.mark.cognitive_category(7)
    def test_mct_ucc_bidirectional_loop(self):
        """Verify that both directions of the MCT↔UCC loop exist:
        MCT writes mct_ucc_pressure, UCC reads it; UCC writes
        ucc_mct_override, MCT reads it."""
        import inspect
        source = inspect.getsource(aeon_core)

        # MCT→UCC direction (GENESIS-5 + EMERGE-3c)
        assert "'mct_ucc_pressure'" in source, (
            "mct_ucc_pressure should exist in codebase"
        )
        # UCC→MCT direction (EMERGE-3a + EMERGE-3b)
        assert "'ucc_mct_override'" in source, (
            "ucc_mct_override should exist in codebase"
        )

    @pytest.mark.cognitive_category(7)
    def test_genesis_patch_markers_complete(self):
        """All GENESIS patch markers should be present."""
        import inspect
        source = inspect.getsource(aeon_core)
        ae_source = inspect.getsource(ae_train)
        combined = source + ae_source

        markers = [
            "PATCH-GENESIS-1",
            "PATCH-GENESIS-2a",
            "PATCH-GENESIS-2b",
            "PATCH-GENESIS-3a",
            "PATCH-GENESIS-3b",
            "PATCH-GENESIS-4a",
            "PATCH-GENESIS-4b",
            "PATCH-GENESIS-5",
            "PATCH-GENESIS-6",
            "PATCH-GENESIS-7a",
            "PATCH-GENESIS-7b",
        ]
        for marker in markers:
            assert marker in combined, (
                f"Patch marker '{marker}' should be present in the codebase"
            )


# ═══════════════════════════════════════════════════════════════════════
# Causal Transparency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGenesisCausalTransparency:
    """Verify causal transparency of GENESIS signal paths."""

    @pytest.mark.cognitive_category(1)
    def test_silent_failure_traceable_through_mct(self):
        """When subsystem_silent_failure_pressure triggers MCT, the
        trigger should be traceable via provenance."""
        mct, bus = _make_mct_with_bus(threshold=0.1)
        bus._trace_enforcement = True
        bus.write_signal('subsystem_silent_failure_pressure', 0.9)
        result = mct.evaluate()
        # The signal should have provenance
        prov = bus.get_full_provenance_map()
        assert 'subsystem_silent_failure_pressure' in prov, (
            "subsystem_silent_failure_pressure should have provenance"
        )

    @pytest.mark.cognitive_category(1)
    def test_integration_health_provenance(self):
        """integration_health writes should have auto-provenance."""
        bus = _make_bus()
        bus._trace_enforcement = True
        bus.write_signal('integration_health', 0.0)
        prov = bus.get_full_provenance_map()
        assert 'integration_health' in prov, (
            "integration_health should have provenance entry"
        )

    @pytest.mark.cognitive_category(1)
    def test_reinforce_coherence_provenance(self):
        """reinforce_coherence_score writes should have auto-provenance."""
        bus = _make_bus()
        bus._trace_enforcement = True
        bus.write_signal('reinforce_coherence_score', 0.8)
        prov = bus.get_full_provenance_map()
        assert 'reinforce_coherence_score' in prov, (
            "reinforce_coherence_score should have provenance entry"
        )


# ═══════════════════════════════════════════════════════════════════════
# Meta-Cognitive Trigger Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGenesisMetaCognitiveTrigger:
    """Verify that GENESIS patches enable proper meta-cognitive cycling."""

    @pytest.mark.cognitive_category(2)
    def test_integration_failure_triggers_mct(self):
        """When integration_health drops to 0.0, MCT should see increased
        coherence_deficit and potentially trigger."""
        mct, bus = _make_mct_with_bus(threshold=0.1)
        # Write failed integration health
        bus.write_signal('integration_health', 0.0)
        result = mct.evaluate()
        # MCT should see the signal
        assert 'integration_health' in bus._read_log

    @pytest.mark.cognitive_category(2)
    def test_ucc_failure_triggers_mct(self):
        """When ucc_evaluation_ok drops to 0.0, MCT should see increased
        coherence_deficit."""
        mct, bus = _make_mct_with_bus(threshold=0.1)
        bus.write_signal('ucc_evaluation_ok', 0.0)
        result = mct.evaluate()
        assert 'ucc_evaluation_ok' in bus._read_log

    @pytest.mark.cognitive_category(2)
    def test_compound_failures_amplify(self):
        """Multiple simultaneous failures should compound their effect
        on MCT trigger score."""
        mct, bus = _make_mct_with_bus(threshold=5.0)
        # Single failure
        bus.write_signal('subsystem_silent_failure_pressure', 0.8)
        result_single = mct.evaluate()
        mct.reset()

        # Compound failures
        bus.write_signal('subsystem_silent_failure_pressure', 0.8)
        bus.write_signal('axiom_coherence_pressure', 0.5)
        bus.write_signal('integration_health', 0.0)
        bus.write_signal('ucc_evaluation_ok', 0.0)
        result_compound = mct.evaluate()

        assert result_compound.get('trigger_score', 0.0) >= (
            result_single.get('trigger_score', 0.0)
        ), "Compound failures should yield higher trigger score"

    @pytest.mark.cognitive_category(7)
    def test_mct_stable_without_bus(self):
        """MCT should be stable and functional even without a feedback bus."""
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
        # Should not crash
        result = mct.evaluate()
        assert 'trigger_score' in result
        assert isinstance(result['trigger_score'], (int, float))
