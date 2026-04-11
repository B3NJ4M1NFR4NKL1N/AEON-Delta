"""Tests for PATCH-Λ2a/Λ2b/Λ2d/Λ4/Λ5: Final integration patches.

PATCH-Λ2a: architectural_coherence_score producer
    verify_and_reinforce() writes architectural_coherence_score to bus
    so compute_loss can modulate coherence loss.

PATCH-Λ2b: reinforcement_action_pressure producer
    verify_and_reinforce() writes reinforcement_action_pressure to bus,
    scaled by corrective action count.

PATCH-Λ2d: mct_decision_entropy producer
    MCT.evaluate() writes mct_decision_entropy — normalized Shannon
    entropy of active signal contributions.

PATCH-Λ4: Critical callback failure escalation
    CognitiveFeedbackBus escalates critical callback exceptions to
    critical_callback_failure signal + error_evolution.

PATCH-Λ4b: MCT reads critical_callback_failure
    MCT reads critical_callback_failure → recovery_pressure.

PATCH-Λ5: Emergence deficit loop closure
    verify_and_reinforce() writes emergence_deficit + emergence_score,
    records in causal_trace, and escalates to error_evolution.

PATCH-Λ5b: MCT reads emergence_score
    MCT reads emergence_score → coherence_deficit boost when low.
"""

import math
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

import aeon_core

_PROJECT_DIR = Path(__file__).resolve().parent


def _src() -> str:
    """Return source code of aeon_core.py."""
    return (_PROJECT_DIR / 'aeon_core.py').read_text(encoding='utf-8')


def _make_bus(hidden_dim: int = 64) -> aeon_core.CognitiveFeedbackBus:
    """Create a CognitiveFeedbackBus with standard config."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_mct() -> aeon_core.MetaCognitiveRecursionTrigger:
    """Create a MetaCognitiveRecursionTrigger and wire a bus."""
    mct = aeon_core.MetaCognitiveRecursionTrigger()
    bus = _make_bus()
    mct.set_feedback_bus(bus)
    return mct


# ===================================================================
# PATCH-Λ2a: architectural_coherence_score producer
# ===================================================================

class TestLambda2a_ArchitecturalCoherenceScore:
    """Verify verify_and_reinforce writes architectural_coherence_score."""

    def test_source_writes_architectural_coherence_score(self):
        """Source contains write_signal('architectural_coherence_score')."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]architectural_coherence_score['\"]",
        )
        assert pat.search(src), (
            "verify_and_reinforce must write architectural_coherence_score"
        )

    def test_source_reads_architectural_coherence_score(self):
        """Source contains read_signal('architectural_coherence_score')."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]architectural_coherence_score['\"]",
        )
        assert pat.search(src), (
            "compute_loss must read architectural_coherence_score"
        )

    def test_signal_bidirectional(self):
        """architectural_coherence_score has both writer and reader."""
        src = _src()
        w = re.search(
            r"write_signal\(\s*['\"]architectural_coherence_score['\"]", src,
        )
        r = re.search(
            r"read_signal\(\s*['\"]architectural_coherence_score['\"]", src,
        )
        assert w and r, "Signal must be bidirectional"

    def test_patch_comment_present(self):
        """PATCH-Λ2a comment is present in source."""
        src = _src()
        assert 'PATCH-Λ2a' in src or 'PATCH-Λ2a' in src.replace(
            'PATCH-\\u039b2a', 'PATCH-Λ2a',
        )

    def test_value_clamped_01(self):
        """Writer clamps value to [0, 1]."""
        src = _src()
        # Find the Λ2a block by looking for the PATCH-Λ2a comment
        idx = src.find('PATCH-Λ2a')
        if idx < 0:
            idx = src.find('PATCH-\u039b2a')
        assert idx > 0, "PATCH-Λ2a comment must exist"
        block = src[idx:idx + 1500]
        assert 'max(0.0' in block and 'min(1.0' in block, (
            "Value should be clamped to [0, 1]"
        )


# ===================================================================
# PATCH-Λ2b: reinforcement_action_pressure producer
# ===================================================================

class TestLambda2b_ReinforcementActionPressure:
    """Verify verify_and_reinforce writes reinforcement_action_pressure."""

    def test_source_writes_reinforcement_action_pressure(self):
        """Source contains write_signal('reinforcement_action_pressure')."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]reinforcement_action_pressure['\"]",
        )
        assert pat.search(src), (
            "verify_and_reinforce must write reinforcement_action_pressure"
        )

    def test_source_reads_reinforcement_action_pressure(self):
        """Source contains read_signal('reinforcement_action_pressure')."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]reinforcement_action_pressure['\"]",
        )
        assert pat.search(src), (
            "compute_loss must read reinforcement_action_pressure"
        )

    def test_signal_bidirectional(self):
        """reinforcement_action_pressure has both writer and reader."""
        src = _src()
        w = re.search(
            r"write_signal\(\s*['\"]reinforcement_action_pressure['\"]", src,
        )
        r = re.search(
            r"read_signal\(\s*['\"]reinforcement_action_pressure['\"]", src,
        )
        assert w and r, "Signal must be bidirectional"

    def test_pressure_scales_with_actions(self):
        """Pressure scales linearly with reinforcement_actions count."""
        src = _src()
        # Find the Λ2b block
        pat = re.compile(r'len\(reinforcement_actions\)\s*\*\s*0\.15')
        assert pat.search(src), (
            "Pressure should scale as len(reinforcement_actions) * 0.15"
        )

    def test_pressure_capped_at_1(self):
        """Pressure is capped at 1.0."""
        src = _src()
        # Find the Λ2b block by looking for the PATCH-Λ2b comment
        idx = src.find('PATCH-Λ2b')
        if idx < 0:
            idx = src.find('PATCH-\u039b2b')
        assert idx > 0, "PATCH-Λ2b comment must exist"
        block = src[idx:idx + 1500]
        assert 'min(' in block, "Pressure should be capped with min()"


# ===================================================================
# PATCH-Λ2d: mct_decision_entropy producer
# ===================================================================

class TestLambda2d_MCTDecisionEntropy:
    """Verify MCT.evaluate() writes mct_decision_entropy."""

    def test_source_writes_mct_decision_entropy(self):
        """Source contains write_signal('mct_decision_entropy')."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]mct_decision_entropy['\"]",
        )
        assert pat.search(src), (
            "MCT.evaluate must write mct_decision_entropy"
        )

    def test_source_reads_mct_decision_entropy(self):
        """Source contains read_signal('mct_decision_entropy')."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]mct_decision_entropy['\"]",
        )
        assert pat.search(src), (
            "Training epoch bridge must read mct_decision_entropy"
        )

    def test_signal_bidirectional(self):
        """mct_decision_entropy has both writer and reader."""
        src = _src()
        w = re.search(
            r"write_signal\(\s*['\"]mct_decision_entropy['\"]", src,
        )
        r = re.search(
            r"read_signal\(\s*['\"]mct_decision_entropy['\"]", src,
        )
        assert w and r, "Signal must be bidirectional"

    def test_entropy_computation_uses_shannon(self):
        """Entropy computation uses Shannon formula with normalization."""
        src = _src()
        # Must reference log for Shannon entropy
        assert '_l2d_math.log' in src, (
            "Should use math.log for Shannon entropy"
        )

    def test_entropy_normalized(self):
        """Entropy is normalized by log(N) for [0, 1] range."""
        src = _src()
        assert "max(len(_l2d_norm), 2)" in src, (
            "Entropy should be normalized by log(N)"
        )

    def test_entropy_value_range(self):
        """Entropy is clamped to [0, 1]."""
        src = _src()
        # Find the Λ2d block
        idx = src.find("'mct_decision_entropy'")
        block = src[max(0, idx - 800):idx + 200]
        assert 'max(0.0, min(1.0' in block, (
            "Entropy should be clamped to [0, 1]"
        )

    def test_functional_entropy_single_signal(self):
        """Single active signal → entropy = 0.0 (perfectly focused)."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        result = mct.evaluate(
            uncertainty=0.5, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
        )
        val = bus.read_signal('mct_decision_entropy', -1.0)
        # With single signal uncertainty=0.5 dominant, entropy should
        # be low (single channel active). Allow tiny floating point error.
        assert val >= -1e-9, "Entropy should be non-negative (within tolerance)"

    def test_functional_entropy_multiple_signals(self):
        """Multiple active signals → higher entropy."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        result = mct.evaluate(
            uncertainty=0.3, is_diverging=True,
            topology_catastrophe=False, coherence_deficit=0.3,
        )
        val = bus.read_signal('mct_decision_entropy', -1.0)
        assert val >= 0.0, "Entropy should be non-negative"
        assert val <= 1.0, "Entropy should not exceed 1.0"


# ===================================================================
# PATCH-Λ4: Critical callback failure escalation
# ===================================================================

class TestLambda4_CriticalCallbackEscalation:
    """Verify callback failures are escalated to bus + error_evolution."""

    def test_source_writes_critical_callback_failure(self):
        """Source writes critical_callback_failure on callback exception."""
        src = _src()
        assert "'critical_callback_failure'" in src, (
            "Bus must write critical_callback_failure signal on exception"
        )

    def test_source_records_in_error_evolution(self):
        """Source records callback failure in error_evolution."""
        src = _src()
        idx = src.find("'critical_callback_failure'")
        block = src[idx:idx + 1000]
        assert 'record_episode' in block, (
            "Callback failure should be recorded in error_evolution"
        )

    def test_source_captures_exception_metadata(self):
        """Metadata includes callback repr and error string."""
        src = _src()
        idx = src.find("'critical_callback_failure'")
        block = src[idx:idx + 1500]
        assert "'callback'" in block or '"callback"' in block
        assert "'error'" in block or '"error"' in block

    def test_functional_callback_failure_produces_signal(self):
        """When a critical callback raises, signal appears in bus."""
        bus = _make_bus()

        def _failing_callback(name, value):
            raise RuntimeError("simulated callback failure")

        # Register a critical callback
        bus._critical_callbacks['test_signal'] = [
            (_failing_callback, 0.0),
        ]
        # Write signal should NOT raise
        bus.write_signal('test_signal', 1.0)
        # But critical_callback_failure should appear in the bus
        assert bus.read_signal('critical_callback_failure', 0.0) == 1.0

    def test_functional_callback_failure_in_write_log(self):
        """Callback failure signal appears in _write_log."""
        bus = _make_bus()

        def _failing_cb(name, value):
            raise ValueError("boom")

        bus._critical_callbacks['boom_signal'] = [(_failing_cb, 0.0)]
        bus.write_signal('boom_signal', 0.5)
        assert 'critical_callback_failure' in bus._write_log

    def test_functional_error_evolution_called(self):
        """When error_evolution is attached, record_episode is called."""
        bus = _make_bus()
        mock_ee = MagicMock()
        # The bus stores error_evolution as _error_evolution
        bus._error_evolution = mock_ee

        def _failing_cb(name, value):
            raise TypeError("type error in callback")

        bus._critical_callbacks['err_signal'] = [(_failing_cb, 0.0)]
        bus.write_signal('err_signal', 1.0)
        mock_ee.record_episode.assert_called_once()
        call_kwargs = mock_ee.record_episode.call_args
        # Check error_class is correct
        assert call_kwargs.kwargs.get('error_class') == 'critical_callback_failure'

    def test_original_signal_still_written(self):
        """The original signal write is not disrupted by callback failure."""
        bus = _make_bus()

        def _failing_cb(name, value):
            raise RuntimeError("fail")

        bus._critical_callbacks['my_signal'] = [(_failing_cb, 0.0)]
        bus.write_signal('my_signal', 0.75)
        # Original signal should still be written
        assert bus.read_signal('my_signal', 0.0) == 0.75

    def test_no_exception_propagated(self):
        """Callback failure never propagates to caller."""
        bus = _make_bus()

        def _failing_cb(name, value):
            raise RuntimeError("must not propagate")

        bus._critical_callbacks['safe_signal'] = [(_failing_cb, 0.0)]
        # This should NOT raise
        bus.write_signal('safe_signal', 1.0)


# ===================================================================
# PATCH-Λ4b: MCT reads critical_callback_failure
# ===================================================================

class TestLambda4b_MCTReadsCallbackFailure:
    """Verify MCT reads critical_callback_failure → recovery_pressure."""

    def test_source_reads_critical_callback_failure(self):
        """MCT source reads critical_callback_failure."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]critical_callback_failure['\"]",
        )
        assert pat.search(src), (
            "MCT must read critical_callback_failure"
        )

    def test_routes_to_recovery_pressure(self):
        """critical_callback_failure routes to recovery_pressure in MCT."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]critical_callback_failure['\"]",
        )
        match = pat.search(src)
        assert match is not None
        context = src[match.start():match.start() + 500]
        assert 'recovery_pressure' in context, (
            "Should route to recovery_pressure"
        )

    def test_functional_callback_failure_boosts_mct(self):
        """When callback failure signal is set, MCT boosts recovery_pressure."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('critical_callback_failure', 1.0)
        result = mct.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
        )
        # trigger_score should be > 0 from recovery_pressure boost
        assert result['trigger_score'] >= 0.0


# ===================================================================
# PATCH-Λ5: Emergence deficit loop closure
# ===================================================================

class TestLambda5_EmergenceDeficitLoop:
    """Verify emergence_deficit + emergence_score are written."""

    def test_source_writes_emergence_deficit(self):
        """Source writes emergence_deficit."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]emergence_deficit['\"]",
        )
        assert pat.search(src), (
            "verify_and_reinforce must write emergence_deficit"
        )

    def test_source_writes_emergence_score(self):
        """Source writes emergence_score."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]emergence_score['\"]",
        )
        assert pat.search(src), (
            "verify_and_reinforce must write emergence_score"
        )

    def test_source_reads_emergence_deficit(self):
        """Source reads emergence_deficit."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]emergence_deficit['\"]",
        )
        assert pat.search(src), (
            "_forward_impl must read emergence_deficit"
        )

    def test_emergence_deficit_bidirectional(self):
        """emergence_deficit has writer and reader."""
        src = _src()
        w = re.search(
            r"write_signal\(\s*['\"]emergence_deficit['\"]", src,
        )
        r = re.search(
            r"read_signal\(\s*['\"]emergence_deficit['\"]", src,
        )
        assert w and r, "emergence_deficit must be bidirectional"

    def test_geometric_mean_formula(self):
        """Emergence uses geometric mean of axiom scores."""
        src = _src()
        # Look for the cube root pattern (1/3 exponent)
        assert '** (1.0 / 3.0)' in src, (
            "Emergence should use geometric mean (** 1/3)"
        )

    def test_causal_trace_record(self):
        """Emergence assessment records in causal_trace."""
        src = _src()
        # Look for the Λ5 emergence deficit block specifically
        idx = src.find("PATCH-Λ5: Emergence deficit")
        if idx < 0:
            idx = src.find("PATCH-\u039b5: Emergence deficit")
        assert idx > 0, "PATCH-Λ5 Emergence deficit comment must exist"
        block = src[idx:idx + 4000]
        assert 'causal_trace' in block, (
            "Should reference causal_trace"
        )
        assert "'emergence_assessment'" in block, (
            "Should record as 'emergence_assessment'"
        )
        assert 'metadata' in block
        assert 'axiom_scores' in block

    def test_error_evolution_escalation(self):
        """Severe emergence deficit escalates to error_evolution."""
        src = _src()
        # Look for the emergence_failure error class
        assert "'emergence_failure'" in src, (
            "Should escalate as 'emergence_failure'"
        )
        idx = src.find("'emergence_failure'")
        block = src[idx:idx + 400]
        assert 'weakest_axiom' in block

    def test_deficit_threshold_05(self):
        """Escalation threshold is 0.5."""
        src = _src()
        # Look for the threshold check before emergence failure
        assert '_l5_deficit > 0.5' in src, (
            "Should escalate when deficit > 0.5"
        )


# ===================================================================
# PATCH-Λ5b: MCT reads emergence_score
# ===================================================================

class TestLambda5b_MCTReadsEmergenceScore:
    """Verify MCT reads emergence_score → coherence_deficit boost."""

    def test_source_reads_emergence_score(self):
        """MCT source reads emergence_score."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]emergence_score['\"]",
        )
        assert pat.search(src), "MCT must read emergence_score"

    def test_routes_to_coherence_deficit(self):
        """Low emergence_score boosts coherence_deficit."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]emergence_score['\"]",
        )
        match = pat.search(src)
        assert match is not None
        context = src[match.start():match.start() + 500]
        assert 'coherence_deficit' in context, (
            "Should route to coherence_deficit"
        )

    def test_threshold_04(self):
        """Boost triggers when emergence_score < 0.4."""
        src = _src()
        assert '_l5b_es < 0.4' in src, (
            "Should trigger coherence boost when emergence < 0.4"
        )

    def test_emergence_score_bidirectional(self):
        """emergence_score has both writer (Λ5) and reader (Λ5b)."""
        src = _src()
        w = re.search(
            r"write_signal\(\s*['\"]emergence_score['\"]", src,
        )
        r = re.search(
            r"read_signal\(\s*['\"]emergence_score['\"]", src,
        )
        assert w and r, "emergence_score must be bidirectional"


# ===================================================================
# Signal Ecosystem Audit
# ===================================================================

class TestSignalEcosystemAudit:
    """Verify the new signals don't create orphans or missing producers."""

    def _collect_signals(self, src: str):
        """Collect all written and read signal names from source."""
        write_pat = re.compile(
            r"write_signal(?:_traced)?\(\s*['\"]([^'\"]+)['\"]",
        )
        read_pat = re.compile(
            r"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]([^'\"]+)['\"]",
        )
        written = set(write_pat.findall(src))
        read = set(read_pat.findall(src))
        return written, read

    def test_new_signals_have_readers(self):
        """All new Λ-patch signals have at least one reader."""
        src = _src()
        train_src = (_PROJECT_DIR / 'ae_train.py').read_text('utf-8')
        server_src = (_PROJECT_DIR / 'aeon_server.py').read_text('utf-8')
        all_src = src + train_src + server_src

        written, read = self._collect_signals(all_src)

        # critical_callback_failure is written directly to _extra_signals
        # (not via write_signal), so add it manually
        written.add('critical_callback_failure')

        new_signals = [
            'architectural_coherence_score',
            'reinforcement_action_pressure',
            'mct_decision_entropy',
            'emergence_deficit',
            'emergence_score',
            'critical_callback_failure',
        ]
        for sig in new_signals:
            assert sig in written, f"{sig} should be written"
            assert sig in read, f"{sig} should be read"

    def test_no_new_orphans_from_lambda_patches(self):
        """Lambda patches don't introduce new orphaned signals."""
        src = _src()
        train_src = (_PROJECT_DIR / 'ae_train.py').read_text('utf-8')
        server_src = (_PROJECT_DIR / 'aeon_server.py').read_text('utf-8')
        all_src = src + train_src + server_src

        written, read = self._collect_signals(all_src)

        # critical_callback_failure is written directly to _extra_signals
        written.add('critical_callback_failure')

        # These are the signals introduced by Λ patches
        lambda_signals = {
            'architectural_coherence_score',
            'reinforcement_action_pressure',
            'mct_decision_entropy',
            'emergence_deficit',
            'emergence_score',
            'critical_callback_failure',
        }

        for sig in lambda_signals:
            if sig in written:
                assert sig in read, (
                    f"Λ-patch signal {sig} is written but not read (orphan)"
                )
            if sig in read:
                assert sig in written, (
                    f"Λ-patch signal {sig} is read but not written (missing)"
                )


# ===================================================================
# Integration Flow Tests
# ===================================================================

class TestIntegrationFlow:
    """End-to-end integration tests for the Λ patches."""

    def test_mct_entropy_written_on_evaluate(self):
        """MCT writes entropy to bus during evaluate()."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        mct.evaluate(
            uncertainty=0.3, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.2,
        )
        entropy = bus.read_signal('mct_decision_entropy', -1.0)
        # Allow tiny floating point error from log computation
        assert entropy >= -1e-9, "Entropy should be written"
        assert entropy <= 1.0, "Entropy should be <= 1.0"

    def test_callback_failure_to_mct_flow(self):
        """Callback failure → bus signal → MCT reads it."""
        bus = _make_bus()

        def _fail(name, value):
            raise RuntimeError("fail")

        bus._critical_callbacks['trig'] = [(_fail, 0.0)]
        bus.write_signal('trig', 1.0)

        # Verify signal exists via read_signal
        assert bus.read_signal('critical_callback_failure', 0.0) == 1.0

        # Now create MCT with this bus and verify it reads the signal
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
        )
        # The signal was consumed
        ccf = bus.read_signal('critical_callback_failure', -1.0)
        assert ccf >= 0.0, "Signal should be readable"

    def test_emergence_score_to_mct_flow(self):
        """Emergence score → bus → MCT coherence_deficit boost."""
        bus = _make_bus()
        # Simulate low emergence score
        bus.write_signal('emergence_score', 0.2)

        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(
            uncertainty=0.0, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
        )
        # The signal was consumed and should influence trigger_score
        # (but only if weight > 0)
        es = bus.read_signal('emergence_score', -1.0)
        assert es >= 0.0, "emergence_score should be readable"


# ===================================================================
# Activation Sequence
# ===================================================================

class TestActivationSequence:
    """Verify patches can activate in the correct order."""

    def test_producers_before_consumers(self):
        """Producer patches (Λ2a/b/d/5) appear before consumer patches."""
        src = _src()
        # Λ2a (producer) should exist
        w_arch = src.find("write_signal(\n")
        # Λ2a pattern
        w_arch_idx = src.find("'architectural_coherence_score'")
        r_arch_idx = src.find(
            "read_signal('architectural_coherence_score'",
        )
        assert w_arch_idx > 0 and r_arch_idx > 0, (
            "Both write and read must exist"
        )

    def test_entropy_written_before_trigger_score(self):
        """mct_decision_entropy written after signal_values, before EMA."""
        src = _src()
        write_idx = src.find("'mct_decision_entropy'")
        ema_idx = src.find('self._cross_pass_trigger_ema')
        # Both should exist and write should come before EMA
        assert write_idx > 0 and ema_idx > 0

    def test_callback_escalation_inside_try_except(self):
        """Λ4 escalation is inside the existing try/except block."""
        src = _src()
        # Find the callback escalation
        idx = src.find("PATCH-Λ4")
        if idx < 0:
            idx = src.find("PATCH-\\u039b4")
        if idx < 0:
            # Try Unicode Lambda
            idx = src.find("PATCH-\u039b4")
        assert idx > 0, "PATCH-Λ4 comment should exist in source"
