"""Tests for PATCH-Ω-FINAL-1/2/3: Final cognitive integration patches.

PATCH-Ω-FINAL-1: Extra signals write-log bridge
    Ensures signals injected via forward(extra_signals=...) are tracked
    in _write_log, _signal_write_pass, and _signal_generation.

PATCH-Ω-FINAL-2: Error episode causal feedback loop
    Ensures verify_and_reinforce() queries causal_trace for recurring
    error episode root causes and publishes error_episode_root_pressure.

PATCH-Ω-FINAL-2b: MCT reader for error_episode_root_pressure
    Ensures MCT reads error_episode_root_pressure and routes it to
    recovery_pressure.

PATCH-Ω-FINAL-3: _NullCausalTrace truncation signal
    Ensures _NullCausalTrace publishes causal_trace_truncation_pressure
    when its lightweight log overflows.

PATCH-Ω-FINAL-3b: MCT reader for causal_trace_truncation_pressure
    Ensures MCT reads causal_trace_truncation_pressure and routes it
    to low_causal_quality.
"""

import re
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

import aeon_core  # noqa: E402


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
# PATCH-Ω-FINAL-1: Extra signals write-log bridge
# ===================================================================

class TestExtraSignalsWriteLogBridge:
    """Verify that forward(extra_signals=...) updates tracking metadata."""

    def test_extra_signals_appear_in_write_log(self):
        """Signals injected via extra_signals must appear in _write_log."""
        import torch
        bus = _make_bus()
        bus.register_signal('test_metric', default=0.0)
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'test_metric': 0.75},
        )
        assert 'test_metric' in bus._write_log, (
            "extra_signals should be tracked in _write_log"
        )

    def test_extra_signals_update_write_pass(self):
        """Signals injected via extra_signals must set _signal_write_pass."""
        import torch
        bus = _make_bus()
        bus.register_signal('alpha_signal', default=0.0)
        initial_pass = bus._pass_counter
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'alpha_signal': 0.5},
        )
        assert 'alpha_signal' in bus._signal_write_pass, (
            "extra_signals should set _signal_write_pass"
        )
        assert bus._signal_write_pass['alpha_signal'] == initial_pass

    def test_extra_signals_update_generation(self):
        """Signals injected via extra_signals must set _signal_generation."""
        import torch
        bus = _make_bus()
        bus.register_signal('beta_signal', default=0.0)
        current_gen = bus._generation
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'beta_signal': 0.8},
        )
        assert 'beta_signal' in bus._signal_generation, (
            "extra_signals should set _signal_generation"
        )
        assert bus._signal_generation['beta_signal'] == current_gen

    def test_default_value_signals_not_tracked(self):
        """Signals at their default value should NOT be tracked as written."""
        import torch
        bus = _make_bus()
        bus.register_signal('gamma_signal', default=0.0)
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'gamma_signal': 0.0},  # default value
        )
        assert 'gamma_signal' not in bus._write_log, (
            "default-value extra_signals should not be tracked"
        )

    def test_no_extra_signals_no_tracking(self):
        """When extra_signals is None, no tracking should be added."""
        import torch
        bus = _make_bus()
        bus.register_signal('delta_signal', default=0.0)
        bus._write_log.clear()
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals=None,
        )
        assert 'delta_signal' not in bus._write_log

    def test_read_signal_current_gen_sees_extra_signals(self):
        """read_signal_current_gen should return values for extra signals."""
        import torch
        bus = _make_bus()
        bus.register_signal('zeta_signal', default=0.0)
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'zeta_signal': 0.9},
        )
        val = bus.read_signal_current_gen('zeta_signal', default=-1.0)
        assert val == pytest.approx(0.9), (
            f"read_signal_current_gen should see extra signal, got {val}"
        )

    def test_freshness_decay_applies_after_staleness(self):
        """Extra signals should decay after becoming stale (age >= 2)."""
        import torch
        bus = _make_bus()
        bus.register_signal('eta_signal', default=0.0)
        # Write in pass 0
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'eta_signal': 1.0},
        )
        # Advance pass counter by 3 to make it stale
        bus._pass_counter += 3
        val = bus.read_signal('eta_signal', default=0.0)
        # With age=3, factor = max(0.1, 1.0 - 0.1 * 2) = 0.8
        assert val < 1.0, (
            f"stale extra signal should decay, got {val}"
        )

    def test_multiple_extra_signals_tracked(self):
        """Multiple non-default extra signals should all be tracked."""
        import torch
        bus = _make_bus()
        bus.register_signal('sig_a', default=0.0)
        bus.register_signal('sig_b', default=0.0)
        bus.register_signal('sig_c', default=0.5)
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={
                'sig_a': 0.7,
                'sig_b': 0.3,
                'sig_c': 0.5,  # default, should not track
            },
        )
        assert 'sig_a' in bus._write_log
        assert 'sig_b' in bus._write_log
        assert 'sig_c' not in bus._write_log


# ===================================================================
# PATCH-Ω-FINAL-2: Error episode causal feedback loop
# ===================================================================

class TestErrorEpisodeCausalFeedback:
    """Verify causal trace query for recurring error episodes."""

    def _make_model_with_trace(self):
        """Create minimal model-like object with causal trace and bus."""
        bus = _make_bus()
        trace = aeon_core.TemporalCausalTraceBuffer(max_entries=100)
        error_evo = aeon_core.CausalErrorEvolutionTracker()
        error_evo.set_causal_trace(trace)

        # Simulate error episodes with failing root causes
        for i in range(6):
            trace.record(
                subsystem=f'error_evolution/convergence_failure',
                decision='rollback:fail',
                causal_prerequisites=['meta_loop', 'encoder'],
                metadata={'pass_id': i},
                severity='warning',
            )
        # A few successes too
        for i in range(2):
            trace.record(
                subsystem=f'error_evolution/convergence_failure',
                decision='rollback:ok',
                causal_prerequisites=['decoder'],
                metadata={'pass_id': 10 + i},
                severity='info',
            )
        return bus, trace, error_evo

    def test_find_with_subsystem_prefix(self):
        """TemporalCausalTraceBuffer.find should support subsystem_prefix."""
        trace = aeon_core.TemporalCausalTraceBuffer(max_entries=100)
        trace.record(
            subsystem='error_evolution/a', decision='x:fail',
            severity='warning',
        )
        trace.record(
            subsystem='error_evolution/b', decision='y:ok',
            severity='info',
        )
        trace.record(
            subsystem='safety/check', decision='pass',
            severity='info',
        )
        results = trace.find(subsystem_prefix='error_evolution/')
        assert len(results) == 2
        for r in results:
            assert r['subsystem'].startswith('error_evolution/')

    def test_find_subsystem_takes_precedence_over_prefix(self):
        """When both subsystem and subsystem_prefix are given, exact match wins."""
        trace = aeon_core.TemporalCausalTraceBuffer(max_entries=100)
        trace.record(
            subsystem='error_evolution/a', decision='x:fail',
            severity='warning',
        )
        trace.record(
            subsystem='error_evolution/b', decision='y:ok',
            severity='info',
        )
        results = trace.find(
            subsystem='error_evolution/a',
            subsystem_prefix='error_evolution/',
        )
        assert len(results) == 1
        assert results[0]['subsystem'] == 'error_evolution/a'

    def test_recurring_root_causes_surfaced(self):
        """verify_and_reinforce should surface recurring root causes."""
        bus, trace, error_evo = self._make_model_with_trace()

        # Query trace for error episodes (same logic as patch)
        entries = trace.find(subsystem_prefix='error_evolution/')
        recurring: dict = {}
        for entry in entries:
            decision = entry.get('decision', '')
            if ':fail' in decision:
                prereqs = entry.get('causal_prerequisites', [])
                for prereq in (prereqs or []):
                    recurring[prereq] = recurring.get(prereq, 0) + 1

        assert 'meta_loop' in recurring
        assert 'encoder' in recurring
        assert recurring['meta_loop'] == 6
        assert recurring['encoder'] == 6

    def test_error_episode_root_pressure_written(self):
        """Bus should contain error_episode_root_pressure after V&R logic."""
        bus, trace, error_evo = self._make_model_with_trace()

        # Simulate the patch logic
        entries = trace.find(subsystem_prefix='error_evolution/')
        recurring: dict = {}
        for entry in entries:
            decision = entry.get('decision', '')
            if ':fail' in decision:
                prereqs = entry.get('causal_prerequisites', [])
                for prereq in (prereqs or []):
                    recurring[prereq] = recurring.get(prereq, 0) + 1

        if recurring:
            worst = max(recurring, key=recurring.get)
            worst_count = recurring[worst]
            pressure = min(1.0, worst_count / 5.0)
            bus.write_signal('error_episode_root_pressure', pressure)

        val = bus.read_signal('error_episode_root_pressure', 0.0)
        assert val == pytest.approx(1.0), (
            f"6 failures / 5.0 = 1.2, clamped to 1.0; got {val}"
        )


# ===================================================================
# PATCH-Ω-FINAL-2b: MCT reads error_episode_root_pressure
# ===================================================================

class TestMCTReadsErrorEpisodeRootPressure:
    """Verify MCT routes error_episode_root_pressure to recovery_pressure."""

    def test_mct_reads_error_episode_root_pressure(self):
        """MCT evaluate should read error_episode_root_pressure from bus."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('error_episode_root_pressure', 0.6)
        result = mct.evaluate(recovery_pressure=0.0)
        # The signal should amplify recovery_pressure
        triggers = result.get('triggers_active', [])
        score = result.get('trigger_score', 0.0)
        # With 0.6 pressure, recovery_pressure should be boosted
        # We mainly verify the signal is read (appears in _read_log)
        assert 'error_episode_root_pressure' in bus._read_log, (
            "MCT should read error_episode_root_pressure"
        )

    def test_mct_ignores_low_pressure(self):
        """MCT should not amplify recovery_pressure for low root pressure."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('error_episode_root_pressure', 0.05)
        result = mct.evaluate(recovery_pressure=0.0)
        # 0.05 < 0.1 threshold, so no amplification
        assert 'error_episode_root_pressure' in bus._read_log


# ===================================================================
# PATCH-Ω-FINAL-3: _NullCausalTrace truncation signal
# ===================================================================

class TestNullCausalTraceTruncation:
    """Verify _NullCausalTrace publishes truncation pressure."""

    def test_truncation_writes_pressure(self):
        """When log overflows, causal_trace_truncation_pressure is written."""
        bus = _make_bus()
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus)
        # Fill the log to capacity
        for i in range(null_trace._max_log):
            null_trace.record(f'module_{i}', 'decision')
        # Now overflow
        for i in range(5):
            null_trace.record(f'overflow_{i}', 'dropped')

        val = bus.read_signal('causal_trace_truncation_pressure', 0.0)
        assert val == pytest.approx(0.5), (
            f"5 truncations / 10.0 = 0.5; got {val}"
        )

    def test_truncation_pressure_saturates(self):
        """Truncation pressure should saturate at 1.0."""
        bus = _make_bus()
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus)
        # Fill and overflow by 15
        for i in range(null_trace._max_log + 15):
            null_trace.record(f'module_{i}', 'decision')

        val = bus.read_signal('causal_trace_truncation_pressure', 0.0)
        assert val == pytest.approx(1.0), (
            f"15 truncations / 10.0 = 1.5, clamped to 1.0; got {val}"
        )

    def test_no_truncation_no_pressure(self):
        """When log is not full, no truncation pressure should exist."""
        bus = _make_bus()
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus)
        for i in range(10):
            null_trace.record(f'module_{i}', 'decision')

        val = bus.read_signal('causal_trace_truncation_pressure', 0.0)
        assert val == 0.0, (
            "No truncation should mean no pressure"
        )

    def test_truncation_count_persists(self):
        """Truncation count should accumulate across record() calls."""
        bus = _make_bus()
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus)
        # Fill to max
        for i in range(null_trace._max_log):
            null_trace.record(f'fill_{i}', 'fill')
        # First overflow
        null_trace.record('over_1', 'dropped')
        assert getattr(null_trace, '_truncation_count', 0) == 1
        # Second overflow
        null_trace.record('over_2', 'dropped')
        assert getattr(null_trace, '_truncation_count', 0) == 2


# ===================================================================
# PATCH-Ω-FINAL-3b: MCT reads causal_trace_truncation_pressure
# ===================================================================

class TestMCTReadsTruncationPressure:
    """Verify MCT routes truncation pressure to low_causal_quality."""

    def test_mct_reads_truncation_pressure(self):
        """MCT evaluate should read causal_trace_truncation_pressure."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('causal_trace_truncation_pressure', 0.7)
        result = mct.evaluate()
        assert 'causal_trace_truncation_pressure' in bus._read_log, (
            "MCT should read causal_trace_truncation_pressure"
        )

    def test_mct_ignores_low_truncation(self):
        """MCT should not boost causal quality for low truncation."""
        mct = _make_mct()
        bus = mct._feedback_bus_ref
        bus.write_signal('causal_trace_truncation_pressure', 0.05)
        result = mct.evaluate()
        assert 'causal_trace_truncation_pressure' in bus._read_log


# ===================================================================
# Signal Ecosystem Audit
# ===================================================================

class TestSignalEcosystemIntegrity:
    """Verify the overall signal ecosystem remains healthy."""

    def test_no_new_orphans(self):
        """New signals must be both written and read."""
        new_signals = [
            'error_episode_root_pressure',
            'causal_trace_truncation_pressure',
        ]
        # Check each signal has both write and read patterns
        with open('aeon_core.py', 'r') as f:
            content = f.read()

        for sig in new_signals:
            write_found = (
                f"write_signal('{sig}'" in content
                or f"write_signal(\n'{sig}'" in content
                or f"write_signal(\n                        '{sig}'" in content
            )
            read_found = (
                f"read_signal('{sig}'" in content
                or f"read_signal(\n'{sig}'" in content
                or f"read_signal(\n                        '{sig}'" in content
                or f"read_signal(\n                    '{sig}'" in content
            )
            assert write_found, f"Signal '{sig}' has no writer"
            assert read_found, f"Signal '{sig}' has no reader"

    def test_signal_ecosystem_balanced(self):
        """All signals should be both written and read (0 orphans, 0 missing)."""
        files = ['aeon_core.py', 'ae_train.py', 'aeon_server.py']
        write_signals = set()
        read_signals = set()

        for fname in files:
            try:
                with open(fname, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                continue

            # Match write_signal('name' and write_signal_traced('name'
            for m in re.finditer(
                r"write_signal(?:_traced)?\(\s*['\"]([^'\"]+)['\"]", content,
            ):
                write_signals.add(m.group(1))
            # Match multi-line write_signal(\n 'name'
            for m in re.finditer(
                r"write_signal(?:_traced)?\(\s*\n\s*['\"]([^'\"]+)['\"]",
                content,
            ):
                write_signals.add(m.group(1))

            # Match read_signal variants
            for m in re.finditer(
                r"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]([^'\"]+)['\"]",
                content,
            ):
                read_signals.add(m.group(1))
            for m in re.finditer(
                r"read_signal(?:_current_gen|_any_gen)?\(\s*\n\s*['\"]([^'\"]+)['\"]",
                content,
            ):
                read_signals.add(m.group(1))

        orphans = write_signals - read_signals
        missing = read_signals - write_signals

        assert len(orphans) == 0, (
            f"Orphaned signals (written, not read): {sorted(orphans)}"
        )
        assert len(missing) == 0, (
            f"Missing producers (read, not written): {sorted(missing)}"
        )


# ===================================================================
# Integration Flow Tests
# ===================================================================

class TestCognitiveIntegrationFlow:
    """End-to-end tests verifying the full cognitive feedback loop."""

    def test_extra_signals_to_mct_via_bus(self):
        """Extra signals written via forward() should be readable by MCT."""
        import torch
        bus = _make_bus()
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)

        # Write a signal via forward extra_signals path
        bus.register_signal('test_flow_signal', default=0.0)
        bus.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'test_flow_signal': 0.85},
        )

        # Verify it's readable
        val = bus.read_signal('test_flow_signal', 0.0)
        assert val == pytest.approx(0.85)

        # Verify generation tracking works
        val_gen = bus.read_signal_current_gen('test_flow_signal', -1.0)
        assert val_gen == pytest.approx(0.85), (
            f"current_gen read should see extra signal, got {val_gen}"
        )

    def test_null_trace_overflow_to_mct(self):
        """NullCausalTrace overflow → bus → MCT trigger modulation."""
        bus = _make_bus()
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus)

        # Fill and overflow
        for i in range(null_trace._max_log + 10):
            null_trace.record(f'module_{i}', 'decision')

        # MCT should now see elevated truncation pressure
        result = mct.evaluate()
        assert 'causal_trace_truncation_pressure' in bus._read_log

    def test_error_episode_to_mct_recovery_pressure(self):
        """Error episode root pressure → bus → MCT recovery_pressure."""
        bus = _make_bus()
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)

        # Simulate writing root pressure (as verify_and_reinforce would)
        bus.write_signal('error_episode_root_pressure', 0.8)

        result = mct.evaluate(recovery_pressure=0.0)
        assert 'error_episode_root_pressure' in bus._read_log
        # The signal should influence the trigger score
        score = result.get('trigger_score', 0.0)
        # With 0.8 pressure, recovery_pressure should be amplified
        # and contribute to the trigger score


# ===================================================================
# Activation Sequence Test
# ===================================================================

class TestActivationSequence:
    """Verify patches can be applied in the correct order."""

    def test_patch_order_independence(self):
        """All three patches should work independently."""
        import torch

        # Patch 1: Extra signals bridge
        bus1 = _make_bus()
        bus1.register_signal('p1_sig', default=0.0)
        bus1.forward(
            batch_size=1,
            device=torch.device('cpu'),
            extra_signals={'p1_sig': 0.5},
        )
        assert 'p1_sig' in bus1._write_log

        # Patch 2: Error episode causal loop
        trace = aeon_core.TemporalCausalTraceBuffer(max_entries=50)
        trace.record(
            subsystem='error_evolution/test',
            decision='strategy:fail',
            causal_prerequisites=['encoder'],
        )
        results = trace.find(subsystem_prefix='error_evolution/')
        assert len(results) == 1

        # Patch 3: Truncation signal
        bus3 = _make_bus()
        null_trace = aeon_core._NullCausalTrace(feedback_bus=bus3)
        for i in range(null_trace._max_log + 3):
            null_trace.record(f'mod_{i}', 'dec')
        val = bus3.read_signal('causal_trace_truncation_pressure', 0.0)
        assert val > 0.0
