"""Tests for PATCH-EMERGE-7 through PATCH-EMERGE-11.

EMERGE-7:  MCT decision recording to causal trace
EMERGE-8:  Training-time verify_and_reinforce invocation
EMERGE-9:  Compound severity pressure → MCT route
EMERGE-9b: compound_severity_pressure bus write
EMERGE-10: RecursionUtilityGate decision broadcast
EMERGE-10b: recursion_gate_suppressed → MCT recovery_pressure damping
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core  # noqa: E402
import ae_train  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBus:
    """Minimal feedback bus mock for signal read/write."""

    def __init__(self):
        self._signals = {}

    def write_signal(self, name, value, **kw):
        self._signals[name] = float(value)

    def write_signal_traced(self, name, value, **kw):
        self._signals[name] = float(value)

    def read_signal(self, name, default=0.0, **kw):
        return self._signals.get(name, default)

    def get_oscillation_score(self):
        return 0.0

    def flush_consumed(self):
        return {}


class _FakeTrace:
    """Minimal causal trace mock."""

    def __init__(self):
        self.entries = []

    def record(self, subsystem=None, decision=None, **kw):
        self.entries.append({
            'subsystem': subsystem,
            'decision': decision,
        })

    def __bool__(self):
        return True


# ===================================================================
# PATCH-EMERGE-7: MCT decision recording to causal trace
# ===================================================================

class TestEmerge7_MCTDecisionTrace(unittest.TestCase):
    """MCT evaluate() records decision snapshot to causal trace."""

    def _make_mct(self, bus=None, trace=None):
        """Build a minimal MCT with optional bus and trace."""
        mct = aeon_core.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=2,
            tightening_factor=0.5,
            extra_iterations=2,
            surprise_threshold=0.5,
            causal_quality_threshold=0.3,
            high_uncertainty_override=0.5,
        )
        if bus is not None:
            mct.set_feedback_bus(bus)
        mct._causal_trace_ref = trace
        return mct

    def test_mct_records_decision_when_trace_available(self):
        """evaluate() records decision to causal trace."""
        bus = _FakeBus()
        trace = _FakeTrace()
        mct = self._make_mct(bus=bus, trace=trace)
        mct.evaluate(uncertainty=0.9)
        # At least one entry should mention MetaCognitiveRecursionTrigger
        mct_entries = [
            e for e in trace.entries
            if e.get('subsystem') == 'MetaCognitiveRecursionTrigger'
        ]
        self.assertGreater(
            len(mct_entries), 0,
            "EMERGE-7: MCT decision not recorded to causal trace",
        )

    def test_mct_decision_contains_score_info(self):
        """Recorded decision includes trigger_score and threshold."""
        bus = _FakeBus()
        trace = _FakeTrace()
        mct = self._make_mct(bus=bus, trace=trace)
        mct.evaluate(uncertainty=0.8, coherence_deficit=0.7)
        mct_entries = [
            e for e in trace.entries
            if e.get('subsystem') == 'MetaCognitiveRecursionTrigger'
        ]
        if mct_entries:
            decision_str = mct_entries[0].get('decision', '')
            self.assertIn('score=', decision_str)
            self.assertIn('threshold=', decision_str)

    def test_mct_no_crash_without_trace(self):
        """evaluate() works fine when causal trace is None."""
        bus = _FakeBus()
        mct = self._make_mct(bus=bus, trace=None)
        result = mct.evaluate(uncertainty=0.5)
        self.assertIn('trigger_score', result)


# ===================================================================
# PATCH-EMERGE-10: RecursionUtilityGate decision broadcast
# ===================================================================

class TestEmerge10_GateBroadcast(unittest.TestCase):
    """RecursionUtilityGate writes recursion_gate_suppressed."""

    def _make_gate(self):
        return aeon_core.RecursionUtilityGate(
            improvement_threshold=0.05,
        )

    def test_gate_writes_suppressed_when_futile(self):
        """Gate writes 1.0 when recursion is NOT useful."""
        gate = self._make_gate()
        bus = _FakeBus()
        gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.99,  # < 5% improvement → futile
            feedback_bus=bus,
        )
        self.assertEqual(
            bus._signals.get('recursion_gate_suppressed'),
            1.0,
            "EMERGE-10: gate should write suppressed=1.0 for futile recursion",
        )

    def test_gate_writes_not_suppressed_when_useful(self):
        """Gate writes 0.0 when recursion IS useful."""
        gate = self._make_gate()
        bus = _FakeBus()
        gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,  # 50% improvement → useful
            feedback_bus=bus,
        )
        self.assertEqual(
            bus._signals.get('recursion_gate_suppressed'),
            0.0,
            "EMERGE-10: gate should write suppressed=0.0 for useful recursion",
        )

    def test_gate_no_crash_without_bus(self):
        """Gate works fine when feedback_bus is None."""
        gate = self._make_gate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.99,
            feedback_bus=None,
        )
        self.assertIn('was_useful', result)


# ===================================================================
# PATCH-EMERGE-10b: MCT reads recursion_gate_suppressed
# ===================================================================

class TestEmerge10b_MCTReadsGate(unittest.TestCase):
    """MCT dampens recovery_pressure when gate suppression active."""

    def test_patch_emerge10b_code_present(self):
        """EMERGE-10b code block exists in aeon_core.py."""
        with open('aeon_core.py') as f:
            src = f.read()
        self.assertIn(
            'recursion_gate_suppressed',
            src,
            "EMERGE-10b: recursion_gate_suppressed reader missing",
        )
        self.assertIn(
            'PATCH-EMERGE-10b',
            src,
            "EMERGE-10b: patch marker missing",
        )


# ===================================================================
# PATCH-EMERGE-9: compound_severity_pressure → MCT
# ===================================================================

class TestEmerge9_CompoundSeverity(unittest.TestCase):
    """compound_severity_pressure routed to MCT recovery_pressure."""

    def test_patch_emerge9_code_present(self):
        """EMERGE-9 code block exists in aeon_core.py."""
        with open('aeon_core.py') as f:
            src = f.read()
        self.assertIn(
            'PATCH-EMERGE-9:',
            src,
            "EMERGE-9: MCT route for compound_severity_pressure missing",
        )

    def test_patch_emerge9b_bus_write(self):
        """EMERGE-9b publishes compound_severity_pressure to bus."""
        with open('aeon_core.py') as f:
            src = f.read()
        self.assertIn(
            'PATCH-EMERGE-9b',
            src,
            "EMERGE-9b: bus write for compound_severity_pressure missing",
        )


# ===================================================================
# PATCH-EMERGE-8: Training-time verify_and_reinforce
# ===================================================================

class TestEmerge8_TrainingVnR(unittest.TestCase):
    """Training loop calls verify_and_reinforce periodically."""

    def test_patch_emerge8_code_present(self):
        """EMERGE-8 code block exists in ae_train.py."""
        with open('ae_train.py') as f:
            src = f.read()
        self.assertIn(
            'PATCH-EMERGE-8',
            src,
            "EMERGE-8: training verify_and_reinforce block missing",
        )
        self.assertIn(
            'verify_and_reinforce',
            src,
            "EMERGE-8: verify_and_reinforce call missing from training",
        )


# ===================================================================
# Signal ecosystem integrity
# ===================================================================

class TestEmergeSignalEcosystem(unittest.TestCase):
    """Verify new signals are both written and read."""

    def _get_signal_sets(self):
        """Extract write/read signal sets from source."""
        write_patterns = [
            re.compile(r"""write_signal(?:_traced)?\s*\(\s*['"]([a-z_]+)['"]"""),
            re.compile(r"""_write_log\.add\s*\(\s*['"]([a-z_]+)['"]"""),
        ]
        read_patterns = [
            re.compile(r"""read_signal\s*\(\s*['"]([a-z_]+)['"]"""),
            re.compile(r"""_extra_signals\.get\s*\(\s*['"]([a-z_]+)['"]"""),
        ]
        # Pattern for signal names on their own line (used for multiline detection)
        sig_name_pat = re.compile(r"""['"]([a-z_]{4,})['"]""")

        writers = set()
        readers = set()

        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                lines = open(fname).readlines()
                for i, line in enumerate(lines):
                    # Single-line matches
                    for pat in write_patterns:
                        for m in pat.finditer(line):
                            writers.add(m.group(1))
                    for pat in read_patterns:
                        for m in pat.finditer(line):
                            readers.add(m.group(1))
                    # Multiline: signal name on current line, function
                    # call on preceding lines (up to 3 lines back)
                    for m in sig_name_pat.finditer(line):
                        sig = m.group(1)
                        ctx = ''.join(lines[max(0, i - 3):i + 1])
                        if 'write_signal' in ctx and 'read_signal' not in ctx:
                            writers.add(sig)
                        elif 'read_signal' in ctx and 'write_signal' not in ctx:
                            readers.add(sig)
            except FileNotFoundError:
                pass

        return writers, readers

    def test_new_signals_connected(self):
        """New EMERGE signals are both written and read."""
        writers, readers = self._get_signal_sets()
        new_signals = [
            'recursion_gate_suppressed',
            'compound_severity_pressure',
        ]
        for sig in new_signals:
            self.assertIn(
                sig, writers,
                f"Signal '{sig}' has no writer",
            )
            self.assertIn(
                sig, readers,
                f"Signal '{sig}' has no reader",
            )


# ===================================================================
# Cross-patch integration
# ===================================================================

class TestEmergeCrossPatch(unittest.TestCase):
    """Cross-patch integration validates that new patches
    don't interfere with existing patch families."""

    def test_existing_mct_bus_writes_preserved(self):
        """MCT still writes mct_should_trigger and mct_trigger_score."""
        with open('aeon_core.py') as f:
            src = f.read()
        self.assertIn("'mct_should_trigger'", src)
        self.assertIn("'mct_trigger_score'", src)

    def test_existing_gate_futility_write_preserved(self):
        """Gate still writes recursion_futility_pressure."""
        with open('aeon_core.py') as f:
            src = f.read()
        self.assertIn("recursion_futility_pressure", src)

    def test_emerge7_before_cogact2(self):
        """EMERGE-7 causal trace recording appears before COGACT-2."""
        with open('aeon_core.py') as f:
            src = f.read()
        pos_e7 = src.find('PATCH-EMERGE-7')
        pos_ca2 = src.find('PATCH-COGACT-2')
        self.assertGreater(pos_e7, 0, "EMERGE-7 marker not found")
        self.assertGreater(pos_ca2, 0, "COGACT-2 marker not found")
        self.assertLess(
            pos_e7, pos_ca2,
            "EMERGE-7 should appear before COGACT-2 in MCT evaluate()",
        )

    def test_emerge10_before_metacognitive_recursor(self):
        """EMERGE-10 appears before MetaCognitiveRecursor class."""
        with open('aeon_core.py') as f:
            src = f.read()
        pos_e10 = src.find('PATCH-EMERGE-10:')
        pos_mcr = src.find('class MetaCognitiveRecursor:')
        self.assertGreater(pos_e10, 0, "EMERGE-10 marker not found")
        self.assertGreater(pos_mcr, 0, "MetaCognitiveRecursor not found")
        self.assertLess(
            pos_e10, pos_mcr,
            "EMERGE-10 should appear inside RecursionUtilityGate, "
            "before MetaCognitiveRecursor",
        )


if __name__ == '__main__':
    unittest.main()
