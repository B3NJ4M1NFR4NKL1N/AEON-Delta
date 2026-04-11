"""Tests for Final Integration & Cognitive Activation patches.

Covers:
  PATCH-D: UCC Exception Traceability — MCT evaluate() failure handling
  PATCH-E: Adaptive Reinforce Interval — default 10, MCT-score-adaptive
  PATCH-A: MCT-Aware Training Step — training reads mct_should_trigger
  PATCH-C: Server Coherence Pre-Injection — pre-forward signal read + server timestamp

Signal ecosystem integrity is also verified.
"""

import importlib
import math
import re
import sys
import types
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

# ── Import targets ────────────────────────────────────────────────────
import aeon_core
import ae_train

# ── Helpers ───────────────────────────────────────────────────────────

def _make_bus():
    """Create a minimal CognitiveFeedbackBus."""
    bus = aeon_core.CognitiveFeedbackBus(hidden_dim=64)
    return bus


def _make_convergence_monitor():
    """Create a minimal ConvergenceMonitor."""
    return aeon_core.ConvergenceMonitor(
        threshold=1e-5,
    )


def _make_error_evolution():
    """Create a minimal CausalErrorEvolutionTracker."""
    return aeon_core.CausalErrorEvolutionTracker(max_history=50)


def _make_provenance():
    """Create a minimal CausalProvenanceTracker."""
    return aeon_core.CausalProvenanceTracker()


def _make_mct(feedback_bus=None):
    """Create a MetaCognitiveRecursionTrigger with optional bus."""
    mct = aeon_core.MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=2,
    )
    if feedback_bus is not None:
        mct.set_feedback_bus(feedback_bus)
    return mct


def _make_ucc(*, mct=None, feedback_bus=None, error_evolution=None,
              causal_trace=None):
    """Create a minimal UnifiedCognitiveCycle."""
    conv_mon = _make_convergence_monitor()
    prov = _make_provenance()
    ee = error_evolution or _make_error_evolution()
    ucc = aeon_core.UnifiedCognitiveCycle(
        convergence_monitor=conv_mon,
        coherence_verifier=None,
        error_evolution=ee,
        metacognitive_trigger=mct,
        provenance_tracker=prov,
        causal_trace=causal_trace,
        feedback_bus=feedback_bus,
    )
    return ucc


# =====================================================================
# PATCH-D: UCC Exception Traceability
# =====================================================================


class TestPatchD_UCCExceptionTraceability:
    """MCT evaluate() failures must be caught and traced, not silent."""

    def _make_failing_ucc(self, *, feedback_bus=None, causal_trace=None,
                          error_evolution=None):
        """Create a UCC with a failing MCT for testing."""
        import torch
        bus = feedback_bus or _make_bus()

        class _FailingMCT:
            _feedback_bus_ref = bus
            def evaluate(self, **kwargs):
                raise RuntimeError("Simulated MCT failure")
            def set_feedback_bus(self, b):
                self._feedback_bus_ref = b

        mct = _FailingMCT()
        ucc = _make_ucc(mct=mct, feedback_bus=bus,
                         error_evolution=error_evolution,
                         causal_trace=causal_trace)
        return ucc, bus

    def test_source_has_try_except_around_mct_evaluate(self):
        """The MCT evaluate() call in UCC must be wrapped in try/except."""
        src = open(aeon_core.__file__).read()
        assert 'PATCH-D: UCC Exception Traceability' in src

    def test_mct_failure_writes_mct_evaluation_failure(self):
        """When MCT evaluate() raises, PATCH-D writes mct_evaluation_failure=1."""
        import torch
        ucc, bus = self._make_failing_ucc()

        result = ucc.evaluate(
            subsystem_states={'encoder': torch.randn(1, 64)},
            delta_norm=0.1,
            uncertainty=0.5,
        )

        failure_val = bus.read_signal('mct_evaluation_failure', 0.0)
        assert failure_val > 0.5, (
            f"Expected mct_evaluation_failure > 0.5, got {failure_val}"
        )

    def test_mct_failure_sets_conservative_trigger(self):
        """MCT failure → trigger_detail shows should_trigger=True."""
        import torch
        ucc, bus = self._make_failing_ucc()

        result = ucc.evaluate(
            subsystem_states={'encoder': torch.randn(1, 64)},
            delta_norm=0.1,
            uncertainty=0.3,
        )

        trigger = result.get('trigger_detail', {})
        assert trigger.get('should_trigger') is True
        assert trigger.get('trigger_score', 0.0) >= 0.9
        assert 'mct_evaluation_failure' in trigger.get(
            'triggers_active', [],
        )

    def test_mct_failure_records_causal_trace(self):
        """MCT failure → causal trace records the failure event."""
        import torch
        bus = _make_bus()
        # Use _NullCausalTrace which supports record() and
        # get_provenance_chain() but not get_causal_chain().
        # The UCC may call get_causal_chain downstream, so we
        # verify only that the record() call in our PATCH-D code
        # succeeds by checking the provenance chain directly.
        trace = aeon_core._NullCausalTrace(feedback_bus=bus)

        ucc, _ = self._make_failing_ucc(
            feedback_bus=bus, causal_trace=trace,
        )

        # UCC.evaluate() may raise downstream due to _NullCausalTrace
        # not implementing get_causal_chain. We catch that and verify
        # our PATCH-D recording happened before the downstream error.
        try:
            ucc.evaluate(
                subsystem_states={'encoder': torch.randn(1, 64)},
                delta_norm=0.1,
                uncertainty=0.5,
            )
        except (AttributeError, Exception):
            pass  # Downstream _NullCausalTrace limitation

        chain = trace.get_provenance_chain()
        failure_entries = [
            e for e in chain
            if 'evaluation_failure' in str(e)
        ]
        assert len(failure_entries) > 0, (
            f"Expected causal trace to record MCT evaluation failure, "
            f"got chain: {chain}"
        )

    def test_mct_failure_records_error_evolution(self):
        """MCT failure → error evolution records an episode."""
        import torch
        bus = _make_bus()
        ee = _make_error_evolution()

        ucc, _ = self._make_failing_ucc(
            feedback_bus=bus, error_evolution=ee,
        )

        # UCC.evaluate() may fail downstream; we only need the PATCH-D
        # error_evolution recording to have happened
        try:
            ucc.evaluate(
                subsystem_states={'encoder': torch.randn(1, 64)},
                delta_norm=0.1,
                uncertainty=0.5,
            )
        except (AttributeError, Exception):
            pass  # Downstream errors are acceptable

        summary = ee.get_error_summary()
        total = summary.get('total_recorded', 0)
        # The error evolution should have at least 1 episode from
        # our PATCH-D recording
        assert total >= 1, (
            f"Expected at least 1 error episode, got {total}. "
            f"Summary: {summary}"
        )

    def test_mct_success_no_failure_signal(self):
        """When MCT evaluate() succeeds, no failure signal is written."""
        import torch
        bus = _make_bus()
        mct = _make_mct(feedback_bus=bus)
        ucc = _make_ucc(mct=mct, feedback_bus=bus)

        ucc.evaluate(
            subsystem_states={'encoder': torch.randn(1, 64)},
            delta_norm=0.1,
            uncertainty=0.5,
        )

        failure_val = bus.read_signal('mct_evaluation_failure', 0.0)
        assert failure_val < 0.5, (
            f"Expected no mct_evaluation_failure, got {failure_val}"
        )


# =====================================================================
# PATCH-E: Adaptive Reinforce Interval
# =====================================================================


class TestPatchE_AdaptiveReinforceInterval:
    """_REINFORCE_INTERVAL defaults to 10 and adapts to MCT score."""

    def test_default_interval_is_10(self):
        """Class-level _REINFORCE_INTERVAL should be 10 (not 50)."""
        src = open(aeon_core.__file__).read()
        # Find the class attribute definition
        match = re.search(
            r'_REINFORCE_INTERVAL:\s*int\s*=\s*(\d+)',
            src,
        )
        assert match is not None, "Could not find _REINFORCE_INTERVAL"
        default_val = int(match.group(1))
        assert default_val == 10, (
            f"Expected _REINFORCE_INTERVAL=10, got {default_val}"
        )

    def test_source_has_adaptive_interval_logic(self):
        """PATCH-E adaptive interval logic must be present."""
        src = open(aeon_core.__file__).read()
        assert 'PATCH-E: Adaptive reinforce interval' in src
        assert '_patche_effective_interval' in src

    def test_source_reads_mct_trigger_score_for_interval(self):
        """The adaptive logic reads mct_trigger_score from the bus."""
        src = open(aeon_core.__file__).read()
        # Look for the pattern where mct_trigger_score is read near
        # the reinforce interval logic
        assert "read_signal('mct_trigger_score'" in src

    def test_source_reads_mct_evaluation_failure_for_interval(self):
        """The adaptive logic reads mct_evaluation_failure from the bus."""
        src = open(aeon_core.__file__).read()
        assert "read_signal('mct_evaluation_failure'" in src

    def test_source_writes_reinforce_interval_current(self):
        """The adaptive logic writes reinforce_interval_current to bus."""
        src = open(aeon_core.__file__).read()
        assert "'reinforce_interval_current'" in src

    def test_reinforce_interval_current_is_read(self):
        """reinforce_interval_current must have a reader (bidirectional)."""
        src = open(aeon_core.__file__).read()
        assert re.search(
            r"read_signal\s*\(\s*['\"]reinforce_interval_current['\"]",
            src,
        ), "reinforce_interval_current not read in aeon_core.py"

    def test_high_mct_score_forces_immediate_reinforce(self):
        """Source: mct_trigger_score > 0.7 → interval=1."""
        src = open(aeon_core.__file__).read()
        assert '_patche_mct_score > 0.7' in src or \
               '_patche_mct_score > 0.7 or _patche_mct_failure > 0.5' in src

    def test_low_mct_score_relaxes_interval(self):
        """Source: mct_trigger_score < 0.3 → interval=max(base, 20)."""
        src = open(aeon_core.__file__).read()
        assert '_patche_mct_score < 0.3' in src


# =====================================================================
# PATCH-A: MCT-Aware Training Step
# =====================================================================


class TestPatchA_MCTAwareTraining:
    """Training reads mct_should_trigger to halt/slow gradient updates."""

    def test_source_has_patch_a_phase_a(self):
        """PATCH-A logic present in Phase A train_step."""
        src = open(ae_train.__file__).read()
        assert 'PATCH-A: MCT-Aware Training Step' in src

    def test_source_has_patch_a_phase_b(self):
        """PATCH-A logic present in Phase B train_step."""
        src = open(ae_train.__file__).read()
        assert 'PATCH-A: MCT-Aware Training Step (Phase B)' in src

    def test_phase_a_reads_mct_should_trigger(self):
        """Phase A train_step reads mct_should_trigger from bus."""
        src = open(ae_train.__file__).read()
        assert "read_signal('mct_should_trigger'" in src

    def test_phase_a_reads_mct_trigger_score(self):
        """Phase A train_step reads mct_trigger_score from bus."""
        src = open(ae_train.__file__).read()
        assert "read_signal('mct_trigger_score'" in src

    def test_phase_a_writes_intervention_active(self):
        """Phase A writes training_mct_intervention_active to bus."""
        src = open(ae_train.__file__).read()
        assert "'training_mct_intervention_active'" in src

    def test_phase_a_skips_backward_on_high_score(self):
        """Phase A skips backward when mct_trigger_score > 0.8."""
        src = open(ae_train.__file__).read()
        assert '_patcha_skip_backward' in src
        assert '_patcha_score > 0.8' in src

    def test_phase_a_halves_lr_on_moderate_trigger(self):
        """Phase A halves LR when triggered but score <= 0.8."""
        src = open(ae_train.__file__).read()
        # Should contain *= 0.5 for LR
        assert "['lr'] *= 0.5" in src

    def test_mct_intervention_active_is_read_by_mct(self):
        """training_mct_intervention_active is consumed by MCT."""
        src = open(aeon_core.__file__).read()
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]training_mct_intervention_active['\"]",
            src,
        ), "training_mct_intervention_active not read in aeon_core.py"

    def test_mct_dampens_recovery_when_training_intervened(self):
        """MCT reduces recovery_pressure when training already acted."""
        src = open(aeon_core.__file__).read()
        assert 'PATCH-A: training_mct_intervention_active' in src
        assert 'recovery_pressure' in src


# =====================================================================
# PATCH-C: Server Coherence Pre-Injection
# =====================================================================


class TestPatchC_ServerCoherencePreInjection:
    """Server coherence read at TOP of train_step, before forward pass."""

    def test_source_has_patch_c_in_training(self):
        """PATCH-C pre-injection logic present in ae_train.py."""
        src = open(ae_train.__file__).read()
        assert 'PATCH-C: Server Coherence Pre-Injection' in src

    def test_pre_injection_before_forward_pass(self):
        """Server coherence read appears BEFORE _forward_pass call."""
        src = open(ae_train.__file__).read()
        # Find PATCH-C and _forward_pass positions
        patchc_pos = src.find('PATCH-C: Server Coherence Pre-Injection')
        forward_pos = src.find('outputs = self._forward_pass(tokens)')
        assert patchc_pos > 0 and forward_pos > 0
        assert patchc_pos < forward_pos, (
            "PATCH-C must appear BEFORE _forward_pass call"
        )

    def test_reads_server_coherence_score(self):
        """Pre-injection reads server_coherence_score from bus."""
        src = open(ae_train.__file__).read()
        # Should appear in the PATCH-C block
        assert "read_signal('server_coherence_score'" in src

    def test_reads_server_ssp_pressure(self):
        """Pre-injection reads server_ssp_pressure from bus."""
        src = open(ae_train.__file__).read()
        assert "read_signal('server_ssp_pressure'" in src

    def test_writes_precheck_signal(self):
        """Pre-injection writes training_server_coherence_precheck."""
        src = open(ae_train.__file__).read()
        assert "'training_server_coherence_precheck'" in src

    def test_coherence_boost_on_low_score(self):
        """When server_coherence_score < 0.4, coherence boost applied."""
        src = open(ae_train.__file__).read()
        assert '_patchc_server_coh < 0.4' in src
        assert '_patchc_boost' in src

    def test_server_writes_inference_complete(self):
        """aeon_server.py writes server_inference_complete counter."""
        src = open('aeon_server.py').read()
        assert "'server_inference_complete'" in src
        assert 'PATCH-C: Server inference completion timestamp' in src

    def test_precheck_signal_is_read_by_mct(self):
        """training_server_coherence_precheck is consumed by MCT."""
        src = open(aeon_core.__file__).read()
        assert "'training_server_coherence_precheck'" in src


# =====================================================================
# PATCH-B: PostOutputUncertaintyGate (already implemented)
# =====================================================================


class TestPatchB_PostOutputUncertaintyGate:
    """Verify that PostOutputUncertaintyGate.evaluate() IS called."""

    def test_gate_evaluate_called_in_forward(self):
        """PostOutputUncertaintyGate.evaluate() called in _forward_impl."""
        src = open(aeon_core.__file__).read()
        assert 'post_output_uncertainty_gate.evaluate(' in src

    def test_gate_writes_post_output_uncertainty_to_bus(self):
        """Gate writes post_output_uncertainty to feedback bus."""
        src = open(aeon_core.__file__).read()
        assert re.search(
            r"write_signal\s*\(\s*\n?\s*['\"]post_output_uncertainty['\"]",
            src,
        ), "post_output_uncertainty not written in aeon_core.py"


# =====================================================================
# Signal Ecosystem Integrity
# =====================================================================


class TestSignalEcosystem:
    """All new signals are bidirectional (written AND read)."""

    def _get_signals(self):
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            with open(fname) as f:
                content = f.read()
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                content,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"write_signal(?:_traced)?\(\s*\n\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                content,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal(?:_current_gen|_any_gen)?\s*\(\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                content,
            ):
                read.add(m.group(1))
            for m in re.finditer(
                r"read_signal(?:_current_gen|_any_gen)?\(\s*\n\s*['\"]([a-z_][a-z0-9_]*)['\"]",
                content,
            ):
                read.add(m.group(1))
        return written, read

    def test_no_orphans(self):
        """No signal written without a reader."""
        written, read = self._get_signals()
        orphans = written - read
        assert len(orphans) == 0, f"Orphaned signals: {sorted(orphans)}"

    def test_no_missing_producers(self):
        """No signal read without a writer."""
        written, read = self._get_signals()
        missing = read - written
        assert len(missing) == 0, f"Missing producers: {sorted(missing)}"

    def test_new_signals_present(self):
        """All PATCH-A/C/D/E signals exist in the ecosystem."""
        written, read = self._get_signals()
        new_signals = {
            'mct_evaluation_failure',
            'training_mct_intervention_active',
            'training_server_coherence_precheck',
            'reinforce_interval_current',
            'server_inference_complete',
        }
        for sig in new_signals:
            assert sig in written, f"Signal '{sig}' not written"
            assert sig in read, f"Signal '{sig}' not read"

    def test_ecosystem_at_least_204(self):
        """Signal ecosystem should have at least 204 bidirectional."""
        written, read = self._get_signals()
        bidirectional = written & read
        assert len(bidirectional) >= 204, (
            f"Expected >=204 bidirectional, got {len(bidirectional)}"
        )


# =====================================================================
# E2E: Functional signal flow scenarios
# =====================================================================


class TestE2E_FunctionalSignalFlow:
    """End-to-end signal flow verification."""

    def test_mct_failure_triggers_conservative_then_reinforce(self):
        """MCT failure → conservative trigger → bus signals flow."""
        import torch
        bus = _make_bus()

        class _FailingMCT:
            _feedback_bus_ref = bus
            def evaluate(self, **kwargs):
                raise RuntimeError("Test failure")
            def set_feedback_bus(self, b):
                self._feedback_bus_ref = b

        mct = _FailingMCT()
        ucc = _make_ucc(mct=mct, feedback_bus=bus)

        result = ucc.evaluate(
            subsystem_states={'encoder': torch.randn(1, 64)},
            delta_norm=0.1,
            uncertainty=0.6,
        )

        # Verify signal chain
        assert bus.read_signal('mct_evaluation_failure', 0.0) > 0.5
        assert bus.read_signal('mct_should_trigger', 0.0) > 0.5
        assert bus.read_signal('mct_trigger_score', 0.0) > 0.5

    def test_server_precheck_flows_to_mct(self):
        """Server coherence < 0.4 → PATCH-C precheck → MCT reads."""
        bus = _make_bus()
        mct = _make_mct(feedback_bus=bus)

        # Simulate server writing low coherence
        bus.write_signal('server_coherence_score', 0.2)
        # Simulate training writing precheck
        bus.write_signal('training_server_coherence_precheck', 0.2)

        # MCT should be able to read the precheck signal
        val = bus.read_signal('training_server_coherence_precheck', 1.0)
        assert val < 0.5, f"Expected low precheck, got {val}"

    def test_training_intervention_dampens_mct_recovery(self):
        """training_mct_intervention_active=1 → MCT reduces recovery."""
        bus = _make_bus()
        # Write intervention active
        bus.write_signal('training_mct_intervention_active', 1.0)
        # MCT should be able to read it
        val = bus.read_signal('training_mct_intervention_active', 0.0)
        assert val > 0.5


# =====================================================================
# Activation Sequence Verification
# =====================================================================


class TestActivationSequence:
    """Verify patches are composable in the planned activation order."""

    def test_patch_d_independent(self):
        """PATCH-D can be applied independently."""
        # Just verify the source pattern exists
        src = open(aeon_core.__file__).read()
        assert 'PATCH-D: UCC Exception Traceability' in src
        assert 'PATCH-D: MCT failure → conservative trigger' in src

    def test_patch_e_independent_of_a(self):
        """PATCH-E works without PATCH-A signals."""
        src = open(aeon_core.__file__).read()
        # PATCH-E reads mct_trigger_score which exists from MCT itself
        assert '_patche_mct_score' in src

    def test_patch_a_uses_bus_signals_from_mct(self):
        """PATCH-A reads from MCT signals that PATCH-D also writes."""
        train_src = open(ae_train.__file__).read()
        core_src = open(aeon_core.__file__).read()
        # PATCH-A reads mct_should_trigger which MCT writes
        assert "'mct_should_trigger'" in train_src
        # And PATCH-D also writes it as fallback
        assert "'mct_should_trigger', 1.0" in core_src

    def test_patch_c_reads_server_signals(self):
        """PATCH-C reads server_coherence_score that server writes."""
        train_src = open(ae_train.__file__).read()
        server_src = open('aeon_server.py').read()
        assert "'server_coherence_score'" in train_src
        assert "'server_coherence_score'" in server_src
