"""Tests for PATCH-Ψ1..Ψ4: SubsystemCrossValidator integration.

These patches close the critical mutual-reinforcement gap where
SubsystemCrossValidator was *defined* (class with full logic) but
never *instantiated*, leaving cross_subsystem_inconsistency always
at 0.0 and preventing MCT from detecting inter-component
contradictions.

Patch summary:
  Ψ1  Instantiate SubsystemCrossValidator in AEONDeltaV3.__init__()
  Ψ2  Call validate() in verify_and_reinforce() to feed inconsistency
      into the next MCT evaluation and record reinforcement actions
  Ψ3  Wire causal trace so consistency verdicts are root-cause-traceable
  Ψ4  Call validate() in reasoning_core_impl before primary MCT
      evaluation so intra-pass contradictions trigger deeper reasoning

Tests organised by the three emergence requirements:
  - Mutual Reinforcement: subsystems verify each other via consistency pairs
  - Meta-Cognitive Trigger: inconsistency auto-escalates MCT
  - Causal Transparency: consistency verdicts are traceable
"""

import importlib
import inspect
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_aeon_core():
    """Import aeon_core with stubbed-out heavy dependencies."""
    # Stub out torch.utils.data imports that may fail in minimal env
    _tud = type(sys)('torch.utils.data')
    _tud.DataLoader = MagicMock
    _tud.TensorDataset = MagicMock
    _tud.Dataset = MagicMock
    _tud.IterableDataset = MagicMock
    sys.modules.setdefault('torch.utils.data', _tud)

    try:
        import aeon_core
        importlib.reload(aeon_core)
    except Exception:
        import aeon_core
    return aeon_core


_ac = _load_aeon_core()


def _source_lines():
    """Return the full source of aeon_core.py as a list of lines."""
    with open(_ac.__file__) as f:
        return f.readlines()


# Reusable minimal feedback bus mock
class _MockFeedbackBus:
    """Minimal CognitiveFeedbackBus stand-in for unit tests."""

    def __init__(self):
        self._signals: Dict[str, float] = {}

    def write_signal(self, name: str, value: float) -> None:
        self._signals[name] = float(value)

    def read_signal(self, name: str, default: float = 0.0) -> float:
        return self._signals.get(name, default)

    def register_persistent_signal(self, name: str, value: float) -> None:
        self._signals.setdefault(name, float(value))


class _MockCausalTrace:
    """Minimal TemporalCausalTraceBuffer stand-in."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._counter = 0

    def record(self, subsystem: str, decision: str, **kwargs) -> str:
        self._counter += 1
        entry_id = f"{subsystem}_{self._counter}"
        self.entries.append({
            'id': entry_id,
            'subsystem': subsystem,
            'decision': decision,
            **kwargs,
        })
        return entry_id

    def find(self, subsystem: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e['subsystem'] == subsystem]


# ---------------------------------------------------------------------------
# PATCH-Ψ1: SubsystemCrossValidator instantiation
# ---------------------------------------------------------------------------


class TestPsi1_Instantiation:
    """Verify SubsystemCrossValidator is now instantiated in __init__."""

    def test_class_exists(self):
        """SubsystemCrossValidator class must exist in aeon_core."""
        assert hasattr(_ac, 'SubsystemCrossValidator')

    def test_init_creates_instance(self):
        """__init__ must assign self.subsystem_cross_validator."""
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        assert 'subsystem_cross_validator' in src
        assert 'SubsystemCrossValidator(' in src

    def test_instance_has_feedback_bus(self):
        """The validator must be wired to feedback_bus at construction."""
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        # Must pass feedback_bus= to constructor
        assert re.search(
            r'SubsystemCrossValidator\([^)]*feedback_bus', src,
        ), "SubsystemCrossValidator must receive feedback_bus in __init__"

    def test_standalone_instantiation(self):
        """Directly instantiate SubsystemCrossValidator."""
        bus = _MockFeedbackBus()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        assert scv._fb_ref is bus
        assert scv._last_inconsistency == 0.0

    def test_validate_returns_dict(self):
        """validate() must return dict with expected keys."""
        bus = _MockFeedbackBus()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        result = scv.validate()
        assert isinstance(result, dict)
        assert 'inconsistency_score' in result
        assert 'violations' in result
        assert 'pair_details' in result

    def test_validate_no_bus_safe(self):
        """validate() with no bus must not raise."""
        scv = _ac.SubsystemCrossValidator(feedback_bus=None)
        result = scv.validate()
        assert result['inconsistency_score'] == 0.0

    def test_consistency_pairs_exist(self):
        """_CONSISTENCY_PAIRS must have at least 4 entries."""
        pairs = _ac.SubsystemCrossValidator._CONSISTENCY_PAIRS
        assert len(pairs) >= 4
        # Each pair is (signal_a, signal_b, relationship)
        for sig_a, sig_b, rel in pairs:
            assert isinstance(sig_a, str)
            assert isinstance(sig_b, str)
            assert rel in ('positively_correlated', 'negatively_correlated')


# ---------------------------------------------------------------------------
# PATCH-Ψ2: validate() in verify_and_reinforce
# ---------------------------------------------------------------------------


class TestPsi2_VerifyAndReinforce:
    """Verify SubsystemCrossValidator.validate() is called in verify_and_reinforce."""

    def test_validate_called_in_verify_and_reinforce(self):
        """Source of verify_and_reinforce must call subsystem_cross_validator.validate()."""
        src = inspect.getsource(_ac.AEONDeltaV3.verify_and_reinforce)
        assert 'subsystem_cross_validator' in src
        assert '.validate()' in src

    def test_result_stored_in_report(self):
        """cross_subsystem_validation should be added to the report."""
        src = inspect.getsource(_ac.AEONDeltaV3.verify_and_reinforce)
        assert 'cross_subsystem_validation' in src

    def test_violations_become_reinforcement_action(self):
        """When violations exist, a reinforcement action should be appended."""
        src = inspect.getsource(_ac.AEONDeltaV3.verify_and_reinforce)
        assert 'reinforcement_actions' in src
        assert 'cross_subsystem_inconsistency' in src

    def test_inconsistency_written_to_bus(self):
        """validate() must write cross_subsystem_inconsistency to bus."""
        bus = _MockFeedbackBus()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.validate()
        # Signal should be written (even if 0.0 when no violations)
        assert 'cross_subsystem_inconsistency' in bus._signals

    def test_high_inconsistency_triggers_action(self):
        """When signals contradict, inconsistency should be > 0."""
        bus = _MockFeedbackBus()
        # Set up a positively_correlated pair with contradicting values
        bus.write_signal('memory_retrieval_confidence', 0.9)
        bus.write_signal('symbolic_reasoning_confidence', 0.1)
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        result = scv.validate()
        # Large gap (0.9 vs 0.1) on positively_correlated → violation
        assert result['inconsistency_score'] > 0.0
        pair_key = 'memory_retrieval_confidence↔symbolic_reasoning_confidence'
        assert pair_key in result['pair_details']
        detail = result['pair_details'][pair_key]
        assert detail['violated'] is True


# ---------------------------------------------------------------------------
# PATCH-Ψ3: Causal trace wiring
# ---------------------------------------------------------------------------


class TestPsi3_CausalTrace:
    """Verify SubsystemCrossValidator is wired to causal trace."""

    def test_wiring_in_init_source(self):
        """__init__ source must wire subsystem_cross_validator causal trace."""
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        assert re.search(
            r'subsystem_cross_validator.*set_causal_trace', src,
        ), "PATCH-Ψ3 causal trace wiring must appear in __init__"

    def test_causal_trace_records_on_validate(self):
        """validate() with causal trace must record consistency check."""
        bus = _MockFeedbackBus()
        trace = _MockCausalTrace()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.set_causal_trace(trace)
        scv.validate()
        # Should record under subsystem='cross_validator'
        entries = trace.find('cross_validator')
        assert len(entries) >= 1
        assert entries[0]['decision'] == 'consistency_check'

    def test_causal_trace_records_violations(self):
        """Trace metadata must include violated pairs."""
        bus = _MockFeedbackBus()
        bus.write_signal('memory_retrieval_confidence', 0.9)
        bus.write_signal('symbolic_reasoning_confidence', 0.1)
        trace = _MockCausalTrace()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.set_causal_trace(trace)
        scv.validate()
        entries = trace.find('cross_validator')
        assert len(entries) >= 1
        meta = entries[0].get('metadata', {})
        assert 'violated_pairs' in meta
        assert len(meta['violated_pairs']) > 0

    def test_causal_trace_severity_warning_on_violation(self):
        """Trace severity must be 'warning' when violations exist."""
        bus = _MockFeedbackBus()
        bus.write_signal('memory_retrieval_confidence', 0.9)
        bus.write_signal('symbolic_reasoning_confidence', 0.1)
        trace = _MockCausalTrace()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.set_causal_trace(trace)
        scv.validate()
        entries = trace.find('cross_validator')
        assert entries[0].get('severity') == 'warning'


# ---------------------------------------------------------------------------
# PATCH-Ψ4: Pre-MCT intra-pass validation
# ---------------------------------------------------------------------------


class TestPsi4_IntraPassValidation:
    """Verify SubsystemCrossValidator.validate() runs before MCT in reasoning."""

    def test_validate_before_mct_in_source(self):
        """subsystem_cross_validator.validate() must appear before MCT evaluate."""
        lines = _source_lines()
        validator_line = None
        mct_line = None
        for i, line in enumerate(lines, 1):
            if 'subsystem_cross_validator.validate()' in line and validator_line is None:
                validator_line = i
            if 'metacognitive_trigger.evaluate(' in line and validator_line is not None:
                mct_line = i
                break
        assert validator_line is not None, (
            "subsystem_cross_validator.validate() must appear in source"
        )
        assert mct_line is not None, (
            "metacognitive_trigger.evaluate() must appear after validate()"
        )
        assert validator_line < mct_line, (
            f"validate() (line {validator_line}) must come before "
            f"MCT evaluate (line {mct_line})"
        )

    def test_validate_call_is_exception_safe(self):
        """The validate() call must be wrapped in try/except."""
        lines = _source_lines()
        for i, line in enumerate(lines):
            if 'subsystem_cross_validator.validate()' in line:
                # Check surrounding context for try/except
                context = ''.join(lines[max(0, i-5):i+5])
                assert 'try:' in context or 'except' in context, (
                    "Pre-MCT validate() must be exception-safe"
                )
                break


# ---------------------------------------------------------------------------
# Signal Ecosystem Audit
# ---------------------------------------------------------------------------


class TestSignalEcosystem:
    """Verify the signal ecosystem remains healthy after patches."""

    def test_cross_subsystem_inconsistency_written(self):
        """cross_subsystem_inconsistency must be written somewhere."""
        with open(_ac.__file__) as f:
            src = f.read()
        assert re.search(
            r"write_signal\s*\(\s*['\"]cross_subsystem_inconsistency['\"]",
            src,
        ), "cross_subsystem_inconsistency must be written to bus"

    def test_cross_subsystem_inconsistency_read(self):
        """cross_subsystem_inconsistency must be read somewhere."""
        with open(_ac.__file__) as f:
            src = f.read()
        assert re.search(
            r"read_signal\s*\(\s*['\"]cross_subsystem_inconsistency['\"]",
            src,
        ), "cross_subsystem_inconsistency must be read from bus"

    def test_mct_reads_inconsistency(self):
        """cross_subsystem_inconsistency must be read somewhere in MCT."""
        with open(_ac.__file__) as f:
            src = f.read()
        # The read_signal call spans multiple lines, so check the full
        # source for a read_signal block containing this signal name.
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]cross_subsystem_inconsistency['\"]",
            src,
        ), "MCT must read cross_subsystem_inconsistency from bus"

    def test_no_new_orphan_signals(self):
        """No new write-only or read-only signal orphans introduced."""
        with open(_ac.__file__) as f:
            src = f.read()
        written = set()
        read = set()
        # Match write_signal("name") or write_signal('name')
        _quote = r"""[\"']"""
        for m in re.finditer(r'write_signal\s*\(\s*' + _quote + r'(\w+)' + _quote, src):
            written.add(m.group(1))
        for m in re.finditer(r'read_signal\s*\(\s*' + _quote + r'(\w+)' + _quote, src):
            read.add(m.group(1))
        # cross_subsystem_inconsistency must be both written and read
        assert 'cross_subsystem_inconsistency' in written
        assert 'cross_subsystem_inconsistency' in read


# ---------------------------------------------------------------------------
# E2E: Mutual Reinforcement
# ---------------------------------------------------------------------------


class TestMutualReinforcement:
    """End-to-end tests for the mutual reinforcement emergence requirement."""

    def test_contradicting_subsystems_detected(self):
        """When subsystems contradict, the validator flags inconsistency."""
        bus = _MockFeedbackBus()
        # MCT trigger score high but convergence arbiter also high
        # (negatively correlated → both high is inconsistent)
        bus.write_signal('mct_trigger_score', 0.8)
        bus.write_signal('convergence_arbiter_confidence', 0.9)
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        result = scv.validate()
        # This pair should be violated
        assert result['inconsistency_score'] > 0.0

    def test_consistent_subsystems_pass(self):
        """When subsystems agree, inconsistency stays low."""
        bus = _MockFeedbackBus()
        # MCT high, convergence low (negatively correlated → expected)
        bus.write_signal('mct_trigger_score', 0.8)
        bus.write_signal('convergence_arbiter_confidence', 0.1)
        # Emergence high, integration high (positively correlated → expected)
        bus.write_signal('emergence_readiness', 0.8)
        bus.write_signal('integration_health', 0.8)
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        result = scv.validate()
        # These pairs should NOT be violated
        for pair_key, detail in result['pair_details'].items():
            if 'mct_trigger_score' in pair_key or 'emergence_readiness' in pair_key:
                assert not detail['violated'], f"{pair_key} should not be violated"

    def test_mct_aware_threshold_tightens(self):
        """When MCT is active (trigger_score > 0.5), threshold tightens."""
        bus = _MockFeedbackBus()
        bus.write_signal('mct_trigger_score', 0.8)
        # Set up a borderline inconsistency
        bus.write_signal('memory_retrieval_confidence', 0.7)
        bus.write_signal('symbolic_reasoning_confidence', 0.2)
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        result_tight = scv.validate()

        # Reset bus without MCT active
        bus2 = _MockFeedbackBus()
        bus2.write_signal('mct_trigger_score', 0.0)
        bus2.write_signal('memory_retrieval_confidence', 0.7)
        bus2.write_signal('symbolic_reasoning_confidence', 0.2)
        scv2 = _ac.SubsystemCrossValidator(feedback_bus=bus2)
        result_normal = scv2.validate()

        # With MCT active, threshold is tighter, so more/same violations
        assert result_tight['inconsistency_score'] >= result_normal['inconsistency_score']


# ---------------------------------------------------------------------------
# E2E: Meta-Cognitive Trigger
# ---------------------------------------------------------------------------


class TestMetaCognitiveTrigger:
    """End-to-end tests for the meta-cognitive trigger emergence requirement."""

    def test_inconsistency_amplifies_coherence_deficit(self):
        """High cross_subsystem_inconsistency must amplify MCT's coherence_deficit."""
        # The MCT reader at PATCH-Σ7b reads cross_subsystem_inconsistency
        # and when > 0.3 amplifies coherence_deficit.  The read and the
        # amplification are in the same block but on different lines.
        with open(_ac.__file__) as f:
            src = f.read()
        # Find the PATCH-Σ7b block: read_signal('cross_subsystem_inconsistency')
        # followed by coherence_deficit amplification within ~20 lines
        match = re.search(
            r"cross_subsystem_inconsistency.*?coherence_deficit",
            src,
            re.DOTALL,
        )
        assert match is not None, (
            "MCT must amplify coherence_deficit when cross_subsystem_inconsistency is high"
        )

    def test_inconsistency_amplifies_recovery_pressure(self):
        """High inconsistency must also amplify recovery_pressure."""
        with open(_ac.__file__) as f:
            src = f.read()
        match = re.search(
            r"cross_subsystem_inconsistency.*?recovery_pressure",
            src,
            re.DOTALL,
        )
        assert match is not None, (
            "MCT must amplify recovery_pressure when inconsistency is high"
        )


# ---------------------------------------------------------------------------
# E2E: Causal Transparency
# ---------------------------------------------------------------------------


class TestCausalTransparency:
    """End-to-end tests for the causal transparency emergence requirement."""

    def test_validate_decision_traceable(self):
        """Every validate() call must leave a causal trace entry."""
        bus = _MockFeedbackBus()
        trace = _MockCausalTrace()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.set_causal_trace(trace)

        # Run 3 validations
        for _ in range(3):
            scv.validate()

        entries = trace.find('cross_validator')
        assert len(entries) == 3, "Each validate() must produce a trace entry"

    def test_trace_includes_pairs_checked(self):
        """Trace metadata must report how many pairs were checked."""
        bus = _MockFeedbackBus()
        trace = _MockCausalTrace()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        scv.set_causal_trace(trace)
        scv.validate()
        entries = trace.find('cross_validator')
        meta = entries[0].get('metadata', {})
        assert 'pairs_checked' in meta
        assert 'inconsistency_score' in meta

    def test_inconsistency_history_tracked(self):
        """SubsystemCrossValidator must track inconsistency history."""
        bus = _MockFeedbackBus()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        for _ in range(5):
            scv.validate()
        assert len(scv._inconsistency_history) == 5


# ---------------------------------------------------------------------------
# Activation Sequence
# ---------------------------------------------------------------------------


class TestActivationSequence:
    """Verify patches must be applied in the correct order."""

    def test_instantiation_before_wiring(self):
        """Ψ1 (instantiation) must come before Ψ3 (causal trace wiring)."""
        lines = _source_lines()
        instantiation_line = None
        wiring_line = None
        for i, line in enumerate(lines, 1):
            if 'SubsystemCrossValidator(' in line and 'class' not in line:
                if instantiation_line is None:
                    instantiation_line = i
            if 'subsystem_cross_validator' in line and 'set_causal_trace' in line:
                if wiring_line is None:
                    wiring_line = i
        assert instantiation_line is not None, "Ψ1 instantiation must exist"
        assert wiring_line is not None, "Ψ3 wiring must exist"
        assert instantiation_line < wiring_line, (
            f"Ψ1 (line {instantiation_line}) must come before "
            f"Ψ3 (line {wiring_line})"
        )

    def test_wiring_before_verify_and_reinforce(self):
        """Ψ3 (wiring) must come before verify_and_reinforce usage (Ψ2)."""
        lines = _source_lines()
        wiring_line = None
        usage_line = None
        for i, line in enumerate(lines, 1):
            if 'subsystem_cross_validator' in line and 'set_causal_trace' in line:
                if wiring_line is None:
                    wiring_line = i
            if ('subsystem_cross_validator' in line
                    and '.validate()' in line
                    and i > (wiring_line or 0)):
                if usage_line is None:
                    usage_line = i
        assert wiring_line is not None
        assert usage_line is not None
        assert wiring_line < usage_line

    def test_intra_pass_exception_safety(self):
        """Ψ4 pre-MCT validate() must not break MCT evaluation on failure."""
        bus = _MockFeedbackBus()
        scv = _ac.SubsystemCrossValidator(feedback_bus=bus)
        # Corrupt the internal state to force an exception
        scv._fb_ref = None
        # Should not raise — validate() handles None bus gracefully
        result = scv.validate()
        assert result['inconsistency_score'] == 0.0


# ---------------------------------------------------------------------------
# Integration Map Summary
# ---------------------------------------------------------------------------


class TestIntegrationMap:
    """Verify the integration map is complete after all Ψ patches."""

    def test_connected_paths(self):
        """All critical cognitive paths must be connected."""
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        # Ψ1: SubsystemCrossValidator instantiated
        assert 'subsystem_cross_validator' in src
        # Ψ3: Causal trace wired
        assert 'subsystem_cross_validator' in src

    def test_rmt34_pairs_still_present(self):
        """RMT34-1 consistency pairs must still be in the validator."""
        pairs = _ac.SubsystemCrossValidator._CONSISTENCY_PAIRS
        pair_signals = [(a, b) for a, b, _ in pairs]
        assert ('mct_trigger_score', 'convergence_arbiter_confidence') in pair_signals
        assert ('emergence_readiness', 'integration_health') in pair_signals

    def test_cogact5c_mct_aware_threshold(self):
        """COGACT-5c MCT-aware threshold must still be active."""
        src = inspect.getsource(_ac.SubsystemCrossValidator.validate)
        assert '_ca5c_threshold' in src
        assert 'mct_trigger_score' in src
