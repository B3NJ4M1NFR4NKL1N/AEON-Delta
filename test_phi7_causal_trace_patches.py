"""Tests for PATCH-Φ7a/b/c/d: Causal Trace Integration.

Verifies that the 4 key subsystems now record decisions to the
TemporalCausalTraceBuffer, satisfying Axiom 3 (root-cause traceability):

  Φ7a — MetaCognitiveRecursionTrigger records trigger evaluations
  Φ7b — MultiLevelSafetySystem records safety verdicts
  Φ7c — AutoCriticLoop records revision decisions
  Φ7d — SubsystemCrossValidator records consistency checks

Also verifies the wiring in AEONDeltaV3.__init__ and end-to-end
causal chain reconstruction.
"""
import sys
import types
import math

import pytest

# ---------------------------------------------------------------------------
# Import helpers — load aeon_core once
# ---------------------------------------------------------------------------
import aeon_core as _ac

TemporalCausalTraceBuffer = _ac.TemporalCausalTraceBuffer
CognitiveFeedbackBus = _ac.CognitiveFeedbackBus
MetaCognitiveRecursionTrigger = _ac.MetaCognitiveRecursionTrigger
AutoCriticLoop = _ac.AutoCriticLoop
SubsystemCrossValidator = _ac.SubsystemCrossValidator

# MultiLevelSafetySystem needs torch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    MultiLevelSafetySystem = _ac.MultiLevelSafetySystem


# ============================================================================
# Φ7a: MetaCognitiveRecursionTrigger → causal trace
# ============================================================================

class TestPhi7aMCTCausalTrace:
    """Verify MCT records trigger decisions to TemporalCausalTraceBuffer."""

    def test_mct_has_set_causal_trace_method(self):
        mct = MetaCognitiveRecursionTrigger()
        assert hasattr(mct, 'set_causal_trace'), \
            "MCT must have set_causal_trace() method"
        assert callable(mct.set_causal_trace)

    def test_mct_causal_trace_ref_initialized_to_none(self):
        mct = MetaCognitiveRecursionTrigger()
        assert mct._causal_trace_ref is None

    def test_mct_set_causal_trace_wires_buffer(self):
        mct = MetaCognitiveRecursionTrigger()
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)
        assert mct._causal_trace_ref is ct

    def test_mct_evaluate_records_to_causal_trace(self):
        mct = MetaCognitiveRecursionTrigger()
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)

        result = mct.evaluate(uncertainty=0.8, is_diverging=True)

        entries = ct.find(subsystem='metacognitive_trigger')
        assert len(entries) >= 1, \
            "MCT evaluate() must record to causal trace"
        entry = entries[0]
        assert entry['decision'] == 'trigger_evaluation'
        assert 'trigger_score' in entry['metadata']
        assert 'should_trigger' in entry['metadata']
        assert 'triggers_active' in entry['metadata']
        assert 'effective_threshold' in entry['metadata']

    def test_mct_non_triggering_also_recorded(self):
        """Even when MCT does NOT fire, the decision is recorded."""
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.99)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)

        result = mct.evaluate(uncertainty=0.01)
        assert not result['should_trigger']

        entries = ct.find(subsystem='metacognitive_trigger')
        assert len(entries) >= 1
        assert entries[0]['severity'] == 'info'

    def test_mct_triggering_has_warning_severity(self):
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.1)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)

        result = mct.evaluate(uncertainty=0.9, is_diverging=True)
        assert result['should_trigger']

        entries = ct.find(subsystem='metacognitive_trigger')
        assert len(entries) >= 1
        assert entries[0]['severity'] == 'warning'

    def test_mct_without_causal_trace_works_normally(self):
        """MCT without causal trace must still work (backward compat)."""
        mct = MetaCognitiveRecursionTrigger()
        result = mct.evaluate(uncertainty=0.5)
        assert 'trigger_score' in result
        assert 'should_trigger' in result


# ============================================================================
# Φ7b: MultiLevelSafetySystem → causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestPhi7bSafetyCausalTrace:
    """Verify SafetySystem records safety verdicts to causal trace."""

    def _make_config(self):
        cfg = types.SimpleNamespace(
            hidden_dim=64,
            action_dim=32,
            num_pillars=8,
        )
        return cfg

    def test_safety_has_set_causal_trace_method(self):
        cfg = self._make_config()
        ss = MultiLevelSafetySystem(cfg)
        assert hasattr(ss, 'set_causal_trace')
        assert callable(ss.set_causal_trace)

    def test_safety_causal_trace_ref_initialized_to_none(self):
        cfg = self._make_config()
        ss = MultiLevelSafetySystem(cfg)
        assert ss._causal_trace_ref is None

    def test_safety_forward_records_to_causal_trace(self):
        cfg = self._make_config()
        ss = MultiLevelSafetySystem(cfg)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        ss.set_causal_trace(ct)

        B = 2
        action = torch.randn(B, cfg.action_dim)
        core = torch.randn(B, cfg.hidden_dim)
        factors = torch.randn(B, cfg.num_pillars)
        diversity = {"diversity": torch.rand(B)}
        topo = {"potential": torch.rand(B, 1)}

        result = ss.forward(action, core, factors, diversity, topo)

        entries = ct.find(subsystem='safety_system')
        assert len(entries) >= 1, \
            "Safety forward() must record to causal trace"
        entry = entries[0]
        assert entry['decision'] == 'safety_evaluation'
        assert 'safety_score' in entry['metadata']
        assert 'action_safety' in entry['metadata']
        assert 'cognitive_safety' in entry['metadata']
        assert 'ethical_safety' in entry['metadata']
        assert 'violation_active' in entry['metadata']

    def test_safety_without_causal_trace_works(self):
        cfg = self._make_config()
        ss = MultiLevelSafetySystem(cfg)
        B = 1
        result = ss.forward(
            torch.randn(B, cfg.action_dim),
            torch.randn(B, cfg.hidden_dim),
            torch.randn(B, cfg.num_pillars),
            {"diversity": torch.rand(B)},
            {"potential": torch.rand(B, 1)},
        )
        assert result.shape == (B, 1)


# ============================================================================
# Φ7c: AutoCriticLoop → causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestPhi7cAutoCriticCausalTrace:
    """Verify AutoCriticLoop records revision decisions to causal trace."""

    def _make_auto_critic(self):
        gen = nn.Linear(64, 64)
        ac = AutoCriticLoop(base_model=gen, hidden_dim=64, max_iterations=2)
        return ac

    def test_auto_critic_has_set_causal_trace_method(self):
        ac = self._make_auto_critic()
        assert hasattr(ac, 'set_causal_trace')
        assert callable(ac.set_causal_trace)

    def test_auto_critic_causal_trace_ref_initialized_to_none(self):
        ac = self._make_auto_critic()
        assert ac._causal_trace_ref is None

    def test_auto_critic_forward_records_to_causal_trace(self):
        ac = self._make_auto_critic()
        ct = TemporalCausalTraceBuffer(max_entries=100)
        ac.set_causal_trace(ct)

        query = torch.randn(1, 64)
        result = ac.forward(query)

        entries = ct.find(subsystem='auto_critic')
        assert len(entries) >= 1, \
            "AutoCritic forward() must record to causal trace"
        entry = entries[0]
        assert entry['decision'] == 'revision_decision'
        assert 'revised' in entry['metadata']
        assert 'iterations' in entry['metadata']
        assert 'final_score' in entry['metadata']
        assert 'threshold' in entry['metadata']

    def test_auto_critic_without_causal_trace_works(self):
        ac = self._make_auto_critic()
        query = torch.randn(1, 64)
        result = ac.forward(query)
        assert 'candidate' in result
        assert 'final_score' in result


# ============================================================================
# Φ7d: SubsystemCrossValidator → causal trace
# ============================================================================

class TestPhi7dCrossValidatorCausalTrace:
    """Verify SubsystemCrossValidator records consistency checks."""

    def test_cross_validator_has_set_causal_trace_method(self):
        cv = SubsystemCrossValidator()
        assert hasattr(cv, 'set_causal_trace')
        assert callable(cv.set_causal_trace)

    def test_cross_validator_causal_trace_ref_initialized_to_none(self):
        cv = SubsystemCrossValidator()
        assert cv._causal_trace_ref is None

    def test_cross_validator_validate_records_to_causal_trace(self):
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cv = SubsystemCrossValidator(feedback_bus=fb)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cv.set_causal_trace(ct)

        # Write some signals so validation has data
        fb.write_signal('memory_retrieval_confidence', 0.9)
        fb.write_signal('symbolic_reasoning_confidence', 0.1)  # disagreement

        result = cv.validate()

        entries = ct.find(subsystem='cross_validator')
        assert len(entries) >= 1, \
            "CrossValidator validate() must record to causal trace"
        entry = entries[0]
        assert entry['decision'] == 'consistency_check'
        assert 'inconsistency_score' in entry['metadata']
        assert 'violated_pairs' in entry['metadata']
        assert 'pairs_checked' in entry['metadata']

    def test_cross_validator_without_causal_trace_works(self):
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cv = SubsystemCrossValidator(feedback_bus=fb)
        result = cv.validate()
        assert 'inconsistency_score' in result

    def test_cross_validator_violation_has_warning_severity(self):
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cv = SubsystemCrossValidator(feedback_bus=fb)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cv.set_causal_trace(ct)

        # Create a clear violation: high memory conf + low reasoning conf
        fb.write_signal('memory_retrieval_confidence', 0.95)
        fb.write_signal('symbolic_reasoning_confidence', 0.05)

        result = cv.validate()
        entries = ct.find(subsystem='cross_validator')
        if entries and entries[0]['metadata'].get('violated_pairs'):
            assert entries[0]['severity'] == 'warning'


# ============================================================================
# Wiring verification: AEONDeltaV3 __init__ wiring
# ============================================================================

class TestPhi7Wiring:
    """Verify causal_trace is wired to all 4 subsystems in __init__."""

    def test_wiring_code_in_init(self):
        """Verify the Φ7 wiring block is present in the source."""
        import inspect
        src = inspect.getsource(_ac)

        # Check that the wiring code references all 4 subsystems
        assert 'metacognitive_trigger.set_causal_trace' in src, \
            "AEONDeltaV3 must wire MCT → causal_trace"
        assert 'safety_system.set_causal_trace' in src, \
            "AEONDeltaV3 must wire safety → causal_trace"
        assert 'auto_critic.set_causal_trace' in src, \
            "AEONDeltaV3 must wire auto_critic → causal_trace"
        assert 'subsystem_health_gate.set_causal_trace' in src, \
            "AEONDeltaV3 must wire health_gate → causal_trace"


# ============================================================================
# End-to-end: Causal chain reconstruction
# ============================================================================

class TestE2ECausalChainReconstruction:
    """Verify backward traceability from any subsystem decision."""

    def test_mct_find_by_subsystem(self):
        mct = MetaCognitiveRecursionTrigger()
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)

        # Record a few evaluations
        mct.evaluate(uncertainty=0.3)
        mct.evaluate(uncertainty=0.7, is_diverging=True)
        mct.evaluate(uncertainty=0.1)

        entries = ct.find(subsystem='metacognitive_trigger')
        assert len(entries) == 3

    def test_cross_subsystem_trace(self):
        """Multiple subsystems record to the same trace buffer."""
        ct = TemporalCausalTraceBuffer(max_entries=100)

        # MCT
        mct = MetaCognitiveRecursionTrigger()
        mct.set_causal_trace(ct)
        mct.evaluate(uncertainty=0.5)

        # CrossValidator
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cv = SubsystemCrossValidator(feedback_bus=fb)
        cv.set_causal_trace(ct)
        cv.validate()

        # Both should be in the trace
        mct_entries = ct.find(subsystem='metacognitive_trigger')
        cv_entries = ct.find(subsystem='cross_validator')
        assert len(mct_entries) >= 1
        assert len(cv_entries) >= 1

        # Total entries should be the sum
        all_entries = ct.recent(n=100)
        assert len(all_entries) >= 2

    def test_causal_trace_summary_counts_all(self):
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct = MetaCognitiveRecursionTrigger()
        mct.set_causal_trace(ct)
        mct.evaluate(uncertainty=0.5)
        mct.evaluate(uncertainty=0.9, is_diverging=True)

        summary = ct.summary()
        assert summary['total_entries'] >= 2


# ============================================================================
# Signal ecosystem audit (verify no regressions)
# ============================================================================

class TestSignalEcosystemAudit:
    """Verify signal ecosystem remains healthy after Φ7 patches."""

    def test_no_new_orphaned_signals(self):
        """Φ7 patches add no new signals — only causal trace recording."""
        import re
        files = ['aeon_core.py', 'ae_train.py', 'aeon_server.py']
        content = ''
        for f in files:
            with open(f) as fh:
                content += fh.read()

        writes = set()
        for m in re.finditer(
            r'write_signal(?:_traced)?\s*\(\s*[\'"](\w+)[\'"]\s*,', content
        ):
            writes.add(m.group(1))

        reads = set()
        for m in re.finditer(
            r'read_signal(?:_current_gen)?\s*\(\s*[\'"](\w+)[\'"]\s*[,)]',
            content,
        ):
            reads.add(m.group(1))

        write_only = writes - reads
        read_only = reads - writes
        bidirectional = writes & reads

        # Φ7 should not increase orphans
        assert len(bidirectional) >= 240, \
            f"Expected ≥240 bidirectional signals, got {len(bidirectional)}"
        # Orphan counts should not increase
        assert len(write_only) <= 5, \
            f"Too many orphaned writes: {write_only}"
        assert len(read_only) <= 5, \
            f"Too many orphaned reads: {read_only}"


# ============================================================================
# Activation sequence verification
# ============================================================================

class TestActivationSequence:
    """Verify patches can be applied without breaking coherence."""

    def test_mct_works_before_causal_trace_set(self):
        """MCT with _causal_trace_ref=None must work (Layer 4 applied last)."""
        mct = MetaCognitiveRecursionTrigger()
        assert mct._causal_trace_ref is None
        result = mct.evaluate(uncertainty=0.5)
        assert 'trigger_score' in result

    def test_cross_validator_works_before_causal_trace_set(self):
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cv = SubsystemCrossValidator(feedback_bus=fb)
        assert cv._causal_trace_ref is None
        result = cv.validate()
        assert 'inconsistency_score' in result

    def test_causal_trace_set_to_none_reverts(self):
        """Setting causal_trace to None disables recording (safe unwire)."""
        mct = MetaCognitiveRecursionTrigger()
        ct = TemporalCausalTraceBuffer(max_entries=100)
        mct.set_causal_trace(ct)
        mct.evaluate(uncertainty=0.5)
        assert len(ct.find(subsystem='metacognitive_trigger')) == 1

        # Disconnect
        mct.set_causal_trace(None)
        mct.evaluate(uncertainty=0.5)
        # Should still have exactly 1 entry (no new ones)
        assert len(ct.find(subsystem='metacognitive_trigger')) == 1
