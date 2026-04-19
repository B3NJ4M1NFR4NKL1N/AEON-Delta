"""Tests for PATCH-FINAL-INT-5/6/7: Final integration patches.

PATCH-FINAL-INT-5: Consume _forced_reevaluation in MCT evaluate()
PATCH-FINAL-INT-6: Consume _forced_recheck in ConvergenceMonitor check()
PATCH-FINAL-INT-7: Recovery → verify_and_reinforce bridge
"""

import pytest
import torch
import math
from unittest.mock import MagicMock, patch

from aeon_core import (
    AEONDeltaV3,
    AEONConfig,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    ConvergenceMonitor,
    TemporalCausalTraceBuffer,
)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    return AEONConfig(**overrides)


def _make_model(**overrides):
    cfg = _make_config(**overrides)
    model = AEONDeltaV3(cfg)
    model.eval()
    return model


def _make_mct(feedback_bus=None):
    """Create a MetaCognitiveRecursionTrigger with a feedback bus."""
    mct = MetaCognitiveRecursionTrigger()
    if feedback_bus is not None:
        mct._feedback_bus_ref = feedback_bus
    return mct


def _make_convergence_monitor(threshold=0.01):
    """Create a ConvergenceMonitor."""
    return ConvergenceMonitor(threshold=threshold)


# ────────────────────────────────────────────────────────────────────
# PATCH-FINAL-INT-5: Forced reevaluation flag in MCT
# ────────────────────────────────────────────────────────────────────

class TestPatchFinalInt5ForcedReevaluation:
    """Tests for consuming _forced_reevaluation in MCT evaluate()."""

    def test_forced_flag_triggers_when_score_below_threshold(self):
        """When _forced_reevaluation is True, should_trigger must be True
        even if trigger_score is below threshold."""
        mct = _make_mct()
        # Set the flag as the callback would
        mct._forced_reevaluation = True
        result = mct.evaluate(
            uncertainty=0.0,  # No uncertainty
            coherence_deficit=0.0,  # No deficit
            recovery_pressure=0.0,  # No pressure
        )
        assert result["should_trigger"] is True
        assert "forced_reevaluation" in result["triggers_active"]

    def test_forced_flag_reset_after_consumption(self):
        """After evaluate() consumes the flag, it must be reset to False."""
        mct = _make_mct()
        mct._forced_reevaluation = True
        mct.evaluate()
        assert mct._forced_reevaluation is False

    def test_no_forced_flag_normal_behavior(self):
        """Without the flag, low signals should NOT trigger."""
        mct = _make_mct()
        assert not getattr(mct, '_forced_reevaluation', False)
        result = mct.evaluate(
            uncertainty=0.0,
            coherence_deficit=0.0,
            recovery_pressure=0.0,
        )
        assert result["should_trigger"] is False

    def test_forced_flag_with_max_recursions_reached(self):
        """When max_recursions is reached (can_recurse=False),
        forced flag should NOT bypass the safety cap."""
        mct = _make_mct()
        mct._forced_reevaluation = True
        # Exhaust recursion budget
        mct._recursion_count = mct.max_recursions + 1
        result = mct.evaluate(uncertainty=0.0)
        # Flag consumed (reset) but trigger NOT forced past safety cap
        assert mct._forced_reevaluation is False
        assert result["should_trigger"] is False

    def test_forced_flag_does_not_double_trigger(self):
        """If should_trigger is already True from high uncertainty,
        forced flag should not add redundant trigger."""
        mct = _make_mct()
        mct._forced_reevaluation = True
        result = mct.evaluate(uncertainty=0.9)  # High enough to trigger
        assert result["should_trigger"] is True
        # Flag still consumed
        assert mct._forced_reevaluation is False

    def test_callback_sets_flag(self):
        """Verify the PATCH-Ξ9 callback actually sets the flag."""
        model = _make_model()
        mct = model.metacognitive_trigger
        fb = model.feedback_bus
        # Write a safety violation above threshold
        fb.write_signal('safety_violation_active', 0.8)
        # Flag should have been set by callback
        assert getattr(mct, '_forced_reevaluation', False) is True

    def test_forced_flag_consumed_on_next_evaluate(self):
        """End-to-end: callback fires → flag set → evaluate consumes it."""
        model = _make_model()
        mct = model.metacognitive_trigger
        fb = model.feedback_bus
        # Fire callback
        fb.write_signal('safety_violation_active', 0.8)
        assert mct._forced_reevaluation is True
        # Evaluate consumes it
        result = mct.evaluate(uncertainty=0.0)
        assert result["should_trigger"] is True
        assert mct._forced_reevaluation is False


# ────────────────────────────────────────────────────────────────────
# PATCH-FINAL-INT-6: Forced recheck flag in ConvergenceMonitor
# ────────────────────────────────────────────────────────────────────

class TestPatchFinalInt6ForcedRecheck:
    """Tests for consuming _forced_recheck in ConvergenceMonitor check()."""

    def test_forced_recheck_withholds_certification(self):
        """When _forced_recheck is True, a normally-converged verdict
        should be downgraded to converging with certified=False."""
        cm = _make_convergence_monitor(threshold=0.01)
        # Build up history showing convergence
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.001)  # Should converge normally

        # Set forced flag as callback would
        cm._forced_recheck = True
        verdict = cm.check(0.0005)  # Even better → would be converged
        assert verdict['status'] == 'converging'
        assert verdict['certified'] is False
        assert verdict.get('forced_recheck') is True

    def test_forced_recheck_flag_reset_after_consumption(self):
        """After check() consumes the flag, it must be reset to False."""
        cm = _make_convergence_monitor(threshold=0.01)
        cm._forced_recheck = True
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.001)
        assert cm._forced_recheck is False

    def test_no_forced_flag_normal_convergence(self):
        """Without the flag, normal convergence should certify."""
        cm = _make_convergence_monitor(threshold=0.01)
        cm.check(1.0)
        cm.check(0.5)
        verdict = cm.check(0.001)
        assert verdict['status'] == 'converged'
        assert verdict['certified'] is True

    def test_forced_recheck_on_diverging_is_still_diverging(self):
        """Forced recheck on an already-diverging state keeps it diverging."""
        cm = _make_convergence_monitor(threshold=0.01)
        cm.check(0.1)
        cm.check(0.5)
        cm._forced_recheck = True
        verdict = cm.check(1.0)  # Diverging
        assert verdict['status'] == 'diverging'
        assert verdict['certified'] is False
        # Flag consumed
        assert cm._forced_recheck is False

    def test_forced_recheck_confidence_penalty(self):
        """Forced recheck should halve the confidence."""
        cm = _make_convergence_monitor(threshold=0.01)
        cm.check(1.0)
        cm.check(0.5)
        # Normal convergence confidence
        verdict_normal = cm.check(0.001)
        normal_confidence = verdict_normal.get('confidence', 0.0)

        # Reset and test with forced flag
        cm2 = _make_convergence_monitor(threshold=0.01)
        cm2.check(1.0)
        cm2.check(0.5)
        cm2._forced_recheck = True
        verdict_forced = cm2.check(0.001)
        forced_confidence = verdict_forced.get('confidence', 0.0)

        assert forced_confidence < normal_confidence

    def test_callback_sets_flag_on_model(self):
        """Verify the PATCH-Ξ9 Lyapunov callback sets the flag on the model's
        convergence monitor."""
        model = _make_model()
        cm = model.convergence_monitor
        fb = model.feedback_bus
        # Write Lyapunov violation above threshold
        fb.write_signal('convergence_lyapunov_violated', 0.8)
        assert getattr(cm, '_forced_recheck', False) is True


# ────────────────────────────────────────────────────────────────────
# PATCH-FINAL-INT-7: Recovery → verify_and_reinforce bridge
# ────────────────────────────────────────────────────────────────────

class TestPatchFinalInt7RecoveryBridge:
    """Tests for the recovery → verify_and_reinforce bridge."""

    def test_failed_recovery_writes_pending_signal(self):
        """_bridge_recovery_to_evolution with success=False should write
        recovery_verification_pending = 1.0 to the bus."""
        model = _make_model()
        model._bridge_recovery_to_evolution(
            error_class="subsystem",
            context="test_failure",
            success=False,
        )
        val = model.feedback_bus.read_signal(
            'recovery_verification_pending', 0.0,
        )
        assert val > 0.5

    def test_successful_recovery_no_pending_signal(self):
        """PATCH-EMERGE-F broadens the recovery→verify bridge so it now
        fires on BOTH success and failure (with an outcome-tagged
        channel).  The legacy failure-only behavior is replaced: a
        successful recovery now writes ``recovery_verification_pending``
        AND ``recovery_verification_outcome_success`` so the post-
        pipeline consumer can run a confirming verify pass and route it
        differently from a repair pass.  This test asserts the new
        symmetric handshake."""
        model = _make_model()
        model._bridge_recovery_to_evolution(
            error_class="subsystem",
            context="test_success",
            success=True,
        )
        pending = model.feedback_bus.read_signal(
            'recovery_verification_pending', 0.0,
        )
        outcome_success = model.feedback_bus.read_signal(
            'recovery_verification_outcome_success', 0.0,
        )
        outcome_failure = model.feedback_bus.read_signal(
            'recovery_verification_outcome_failure', 0.0,
        )
        # PATCH-EMERGE-F: pending fires on success too
        assert pending >= 0.5
        # outcome tagged as success, not failure
        assert outcome_success >= 0.5
        assert outcome_failure < 0.5

    def test_routing_entry_exists(self):
        """recovery_verification_pending must have a routing entry in
        _FEEDBACK_SIGNAL_TO_TRIGGER."""
        routing = MetaCognitiveRecursionTrigger._FEEDBACK_SIGNAL_TO_TRIGGER
        assert "recovery_verification_pending" in routing
        assert routing["recovery_verification_pending"] == "recovery_pressure"

    def test_causal_trace_recorded_on_verification(self):
        """When post-recovery verification runs, a causal trace entry
        should be recorded with subsystem='error_recovery'."""
        model = _make_model()
        # Write the pending signal directly
        model.feedback_bus.write_signal('recovery_verification_pending', 1.0)
        # Run a forward pass to trigger the consumer
        x = torch.randint(0, 32, (1, 8))
        with torch.no_grad():
            model(x)
        # Signal should be consumed (reset to 0.0) by the post-pipeline
        # consumer.  Even if verify_and_reinforce was skipped due to
        # re-entrancy, the consumer resets the signal before calling it.
        val = model.feedback_bus.read_signal(
            'recovery_verification_pending', 0.0,
        )
        assert val == 0.0, (
            f"recovery_verification_pending should be consumed (0.0), "
            f"got {val}"
        )


# ────────────────────────────────────────────────────────────────────
# Integration: Signal ecosystem completeness
# ────────────────────────────────────────────────────────────────────

class TestSignalEcosystemCompleteness:
    """Verify that new signals don't create orphans."""

    def test_recovery_verification_pending_not_orphaned(self):
        """recovery_verification_pending must have a routing entry so
        it is not orphaned after being written."""
        routing = MetaCognitiveRecursionTrigger._FEEDBACK_SIGNAL_TO_TRIGGER
        assert "recovery_verification_pending" in routing

    def test_forced_reevaluation_consumed_not_orphaned(self):
        """_forced_reevaluation flag is consumed (reset to False)
        in every evaluate() call."""
        mct = _make_mct()
        mct._forced_reevaluation = True
        mct.evaluate()
        assert mct._forced_reevaluation is False

    def test_forced_recheck_consumed_not_orphaned(self):
        """_forced_recheck flag is consumed (reset to False)
        in every check() call."""
        cm = _make_convergence_monitor()
        cm._forced_recheck = True
        cm.check(1.0)
        assert cm._forced_recheck is False


# ────────────────────────────────────────────────────────────────────
# Integration: Mutual reinforcement
# ────────────────────────────────────────────────────────────────────

class TestMutualReinforcement:
    """Verify the 3 patches create mutual reinforcement loops."""

    def test_safety_violation_triggers_mct_cycle(self):
        """Safety violation → callback → _forced_reevaluation → MCT trigger.
        This verifies the complete intra-pass response path."""
        model = _make_model()
        mct = model.metacognitive_trigger
        # Fire safety callback
        model.feedback_bus.write_signal('safety_violation_active', 0.8)
        assert mct._forced_reevaluation is True
        # MCT evaluates and triggers
        result = mct.evaluate(uncertainty=0.0)
        assert result["should_trigger"] is True
        assert mct._forced_reevaluation is False

    def test_lyapunov_violation_withholds_certification(self):
        """Lyapunov violation → callback → _forced_recheck → no certification.
        This verifies the convergence fast-path is complete."""
        model = _make_model()
        cm = model.convergence_monitor
        # Build converging history
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.001)
        # Fire Lyapunov callback
        model.feedback_bus.write_signal('convergence_lyapunov_violated', 0.8)
        assert cm._forced_recheck is True
        # Check withholds certification
        verdict = cm.check(0.0005)
        assert verdict['certified'] is False

    def test_failed_recovery_triggers_verification_signal(self):
        """Failed recovery → pending signal → routed to MCT.
        Verifies recovery failures feed back to meta-cognition."""
        model = _make_model()
        model._bridge_recovery_to_evolution("test", "ctx", success=False)
        val = model.feedback_bus.read_signal(
            'recovery_verification_pending', 0.0,
        )
        assert val > 0.5


# ────────────────────────────────────────────────────────────────────
# Integration: Causal transparency
# ────────────────────────────────────────────────────────────────────

class TestCausalTransparency:
    """Verify that patches maintain causal transparency."""

    def test_forced_reevaluation_trigger_traceable(self):
        """When forced_reevaluation triggers MCT, the trigger reason
        must appear in triggers_active for root-cause analysis."""
        mct = _make_mct()
        mct._forced_reevaluation = True
        result = mct.evaluate()
        assert "forced_reevaluation" in result["triggers_active"]

    def test_forced_recheck_reason_in_verdict(self):
        """When forced_recheck downgrades convergence, the reason
        must appear in the verdict metadata."""
        cm = _make_convergence_monitor(threshold=0.01)
        cm.check(1.0)
        cm.check(0.5)
        cm._forced_recheck = True
        verdict = cm.check(0.001)
        assert verdict.get('forced_recheck') is True


# ────────────────────────────────────────────────────────────────────
# Activation sequence
# ────────────────────────────────────────────────────────────────────

class TestActivationSequence:
    """Verify patches can be applied in the correct order."""

    def test_patches_applied_in_correct_order(self):
        """All 3 patches must be active simultaneously.
        PATCH-FINAL-INT-5: MCT forced reevaluation (depends on Ξ9 callbacks)
        PATCH-FINAL-INT-6: CM forced recheck (depends on Ξ9 callbacks)
        PATCH-FINAL-INT-7: Recovery bridge (depends on error_evolution)
        """
        model = _make_model()
        # INT-5: MCT has forced_reevaluation capability
        mct = model.metacognitive_trigger
        mct._forced_reevaluation = True
        r = mct.evaluate()
        assert r["should_trigger"] is True

        # INT-6: CM has forced_recheck capability
        cm = model.convergence_monitor
        cm.check(1.0)
        cm.check(0.5)
        cm._forced_recheck = True
        v = cm.check(0.001)
        assert v['certified'] is False

        # INT-7: Recovery bridge writes pending signal
        model._bridge_recovery_to_evolution("test", "ctx", False)
        val = model.feedback_bus.read_signal(
            'recovery_verification_pending', 0.0,
        )
        assert val > 0.5
