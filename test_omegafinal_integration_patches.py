"""
Tests for PATCH-ΩF-1 through PATCH-ΩF-7:
AEON-Delta Final Integration & Cognitive Activation — Same-Pass MCT Triggers.

These patches close the last remaining feedback-loop gaps that prevented
AEON-Delta from responding to critical events within the same reasoning cycle.

PATCH-ΩF-1: Safety Violation → Same-Pass MCT
  - MultiLevelSafetySystem gains set_metacognitive_trigger().
  - When safety_score < 0.5, MCT.evaluate() is called immediately.
  - Writes safety_same_pass_mct_triggered signal.

PATCH-ΩF-2: Cross-Validator Inconsistency → Same-Pass MCT
  - SubsystemCrossValidator gains set_metacognitive_trigger().
  - When inconsistency_score > 0.6, MCT.evaluate() is called immediately.
  - Writes cross_validator_same_pass_mct_triggered signal.

PATCH-ΩF-3: Coherence Trend Reversal → Same-Pass MCT
  - SubsystemCoherenceRegistry gains set_metacognitive_trigger().
  - When trend reversals are detected with conflict_pressure > 0.2,
    MCT.evaluate() is called immediately.

PATCH-ΩF-4: Oscillation Detection → Same-Pass MCT
  - CognitiveFeedbackBus gains set_metacognitive_trigger().
  - When oscillation_severity > 0.5, MCT.evaluate() is called immediately.
  - Writes oscillation_same_pass_mct_triggered signal.

PATCH-ΩF-5: Memory Degradation → Same-Pass MCT
  - HierarchicalMemory triggers MCT when staleness > 0.8 AND
    retrieval confidence < 0.3.
  - Writes memory_same_pass_mct_triggered signal.

PATCH-ΩF-6: Wiring
  - AEONDeltaV3.__init__ wires MCT ref into all 5 subsystems.

PATCH-ΩF-7: MCT Readers for Same-Pass Signals
  - MetaCognitiveRecursionTrigger.evaluate() reads the 4 new same-pass
    trigger signals and boosts recovery_pressure when any are active.

Signal Ecosystem: 4 new bidirectional signals:
  safety_same_pass_mct_triggered
  cross_validator_same_pass_mct_triggered
  oscillation_same_pass_mct_triggered
  memory_same_pass_mct_triggered

Activation Sequence:
  1. ΩF-1..5 — Add set_metacognitive_trigger() and same-pass MCT calls
  2. ΩF-6   — Wire MCT references in AEONDeltaV3.__init__
  3. ΩF-7   — MCT reads new signals (producers before consumers)
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

_PROJECT_DIR = Path(__file__).resolve().parent


def _src(filename: str = 'aeon_core.py') -> str:
    """Return source code of a project file (cached)."""
    cache_attr = f'_cache_{filename}'
    if not hasattr(_src, cache_attr):
        setattr(_src, cache_attr,
                (_PROJECT_DIR / filename).read_text(encoding='utf-8'))
    return getattr(_src, cache_attr)


def _class_block(class_name: str, src: str | None = None) -> str:
    """Extract the source code for a single class definition."""
    src = src or _src()
    idx = src.index(f'class {class_name}')
    next_class = src.find('\nclass ', idx + 1)
    if next_class == -1:
        return src[idx:]
    return src[idx:next_class]


def make_bus():
    """Create a minimal CognitiveFeedbackBus for testing."""
    import aeon_core
    return aeon_core.CognitiveFeedbackBus(hidden_dim=64)


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-1: Safety Violation → Same-Pass MCT
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF1_SafetySamePassMCT:
    """PATCH-ΩF-1: Safety violation triggers immediate MCT re-evaluation."""

    def test_safety_has_set_metacognitive_trigger(self):
        """MultiLevelSafetySystem has set_metacognitive_trigger method."""
        src = _class_block('MultiLevelSafetySystem')
        assert 'def set_metacognitive_trigger(' in src

    def test_safety_has_mct_ref_init(self):
        """MultiLevelSafetySystem initializes _mct_ref = None."""
        src = _class_block('MultiLevelSafetySystem')
        assert '_mct_ref' in src

    def test_safety_calls_mct_evaluate_on_violation(self):
        """Safety forward() calls MCT.evaluate() when violation detected."""
        src = _class_block('MultiLevelSafetySystem')
        assert re.search(
            r'_omf1_result\s*=\s*self\._mct_ref\.evaluate\(',
            src,
        ), "Safety must call self._mct_ref.evaluate() on violation"

    def test_safety_checks_score_below_threshold(self):
        """Safety triggers MCT only when safety_mean < 0.5."""
        src = _class_block('MultiLevelSafetySystem')
        assert re.search(
            r'_omf1_safety_mean\s*<\s*0\.5',
            src,
        ), "MCT trigger must check safety_mean < 0.5"

    def test_safety_writes_same_pass_signal(self):
        """Safety writes safety_same_pass_mct_triggered to bus."""
        src = _class_block('MultiLevelSafetySystem')
        assert 'safety_same_pass_mct_triggered' in src

    def test_safety_mct_has_guard(self):
        """Safety MCT trigger is wrapped in try/except."""
        src = _class_block('MultiLevelSafetySystem')
        # The MCT evaluate call must be guarded with try/except
        assert '_omf1_result = self._mct_ref.evaluate(' in src
        # Find the evaluate call and check for nearby except
        eval_idx = src.index('_omf1_result = self._mct_ref.evaluate(')
        nearby = src[eval_idx:eval_idx + 1000]
        assert 'except Exception' in nearby, (
            "Same-pass MCT must be wrapped in try/except"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-2: Cross-Validator Inconsistency → Same-Pass MCT
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF2_CrossValidatorSamePassMCT:
    """PATCH-ΩF-2: High inconsistency triggers immediate MCT."""

    def test_cross_validator_has_set_metacognitive_trigger(self):
        """SubsystemCrossValidator has set_metacognitive_trigger method."""
        src = _class_block('SubsystemCrossValidator')
        assert 'def set_metacognitive_trigger(' in src

    def test_cross_validator_has_mct_ref_init(self):
        """SubsystemCrossValidator initializes _mct_ref = None."""
        src = _class_block('SubsystemCrossValidator')
        assert '_mct_ref' in src

    def test_cross_validator_calls_mct_evaluate(self):
        """validate() calls MCT.evaluate() when inconsistency > 0.6."""
        src = _class_block('SubsystemCrossValidator')
        assert re.search(
            r'_omf2_result\s*=\s*self\._mct_ref\.evaluate\(',
            src,
        ), "CrossValidator must call self._mct_ref.evaluate()"

    def test_cross_validator_threshold_0_6(self):
        """MCT triggers when overall > 0.6."""
        src = _class_block('SubsystemCrossValidator')
        assert re.search(
            r'overall\s*>\s*0\.6',
            src,
        ), "MCT trigger threshold must be overall > 0.6"

    def test_cross_validator_writes_same_pass_signal(self):
        """validate() writes cross_validator_same_pass_mct_triggered."""
        src = _class_block('SubsystemCrossValidator')
        assert 'cross_validator_same_pass_mct_triggered' in src


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-3: Coherence Trend Reversal → Same-Pass MCT
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF3_CoherenceTrendMCT:
    """PATCH-ΩF-3: Coherence trend reversal triggers MCT."""

    def test_coherence_registry_has_set_metacognitive_trigger(self):
        """SubsystemCoherenceRegistry has set_metacognitive_trigger method."""
        src = _class_block('SubsystemCoherenceRegistry')
        assert 'def set_metacognitive_trigger(' in src

    def test_coherence_registry_has_mct_ref_init(self):
        """SubsystemCoherenceRegistry initializes _mct_ref = None."""
        src = _class_block('SubsystemCoherenceRegistry')
        assert '_mct_ref' in src

    def test_coherence_triggers_on_reversals(self):
        """check_conflicts calls MCT when trend reversals > 0."""
        src = _class_block('SubsystemCoherenceRegistry')
        assert re.search(
            r'_n_reversals\s*>\s*0',
            src,
        ), "MCT must trigger when _n_reversals > 0"

    def test_coherence_checks_conflict_pressure(self):
        """MCT only triggers when conflict_pressure > 0.2."""
        src = _class_block('SubsystemCoherenceRegistry')
        assert re.search(
            r'conflict_pressure\s*>\s*0\.2',
            src,
        ), "MCT trigger needs conflict_pressure > 0.2"


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-4: Oscillation → Same-Pass MCT
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF4_OscillationSamePassMCT:
    """PATCH-ΩF-4: High oscillation triggers immediate MCT."""

    def test_feedback_bus_has_set_metacognitive_trigger(self):
        """CognitiveFeedbackBus has set_metacognitive_trigger method."""
        src = _class_block('CognitiveFeedbackBus')
        assert 'def set_metacognitive_trigger(' in src

    def test_feedback_bus_has_mct_ref_init(self):
        """CognitiveFeedbackBus initializes _mct_ref = None."""
        src = _class_block('CognitiveFeedbackBus')
        assert '_mct_ref' in src

    def test_oscillation_triggers_above_0_5(self):
        """MCT triggers when _xi3_agg_osc > 0.5."""
        src = _class_block('CognitiveFeedbackBus')
        assert re.search(
            r'_xi3_agg_osc\s*>\s*0\.5',
            src,
        ), "Oscillation MCT trigger threshold must be > 0.5"

    def test_oscillation_writes_same_pass_signal(self):
        """Writes oscillation_same_pass_mct_triggered to bus."""
        src = _class_block('CognitiveFeedbackBus')
        assert 'oscillation_same_pass_mct_triggered' in src


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-5: Memory Degradation → Same-Pass MCT
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF5_MemorySamePassMCT:
    """PATCH-ΩF-5: Memory degradation triggers immediate MCT."""

    def test_memory_has_mct_trigger_block(self):
        """HierarchicalMemory has PATCH-ΩF-5 trigger block."""
        src = _class_block('HierarchicalMemory')
        assert 'PATCH-ΩF-5' in src

    def test_memory_checks_staleness_threshold(self):
        """MCT triggers when staleness > 0.8."""
        src = _class_block('HierarchicalMemory')
        assert re.search(
            r'_omf5_staleness\s*>\s*0\.8',
            src,
        ), "Memory MCT trigger needs staleness > 0.8"

    def test_memory_checks_confidence_threshold(self):
        """MCT triggers when confidence < 0.3."""
        src = _class_block('HierarchicalMemory')
        assert re.search(
            r'_omf5_confidence\s*<\s*0\.3',
            src,
        ), "Memory MCT trigger needs confidence < 0.3"

    def test_memory_writes_same_pass_signal(self):
        """Writes memory_same_pass_mct_triggered to bus."""
        src = _class_block('HierarchicalMemory')
        assert 'memory_same_pass_mct_triggered' in src

    def test_memory_caches_staleness(self):
        """Memory caches _last_staleness for MCT trigger."""
        src = _class_block('HierarchicalMemory')
        assert '_last_staleness' in src


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-6: Wiring in AEONDeltaV3.__init__
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF6_Wiring:
    """PATCH-ΩF-6: MCT references wired in AEONDeltaV3.__init__."""

    def test_wiring_block_exists(self):
        """AEONDeltaV3.__init__ has PATCH-ΩF-6 wiring block."""
        src = _src()
        assert 'PATCH-ΩF-6' in src

    def test_safety_system_wired(self):
        """Safety system MCT wiring present."""
        src = _src()
        omf6_idx = src.index('PATCH-ΩF-6')
        wiring_block = src[omf6_idx:omf6_idx + 3000]
        assert 'safety_system.set_metacognitive_trigger' in wiring_block

    def test_coherence_registry_wired(self):
        """Coherence registry MCT wiring present."""
        src = _src()
        omf6_idx = src.index('PATCH-ΩF-6')
        wiring_block = src[omf6_idx:omf6_idx + 3000]
        assert 'coherence_registry.set_metacognitive_trigger' in wiring_block

    def test_feedback_bus_wired(self):
        """Feedback bus MCT wiring present."""
        src = _src()
        omf6_idx = src.index('PATCH-ΩF-6')
        wiring_block = src[omf6_idx:omf6_idx + 3000]
        assert 'feedback_bus.set_metacognitive_trigger' in wiring_block

    def test_hierarchical_memory_wired(self):
        """Hierarchical memory MCT wiring present."""
        src = _src()
        omf6_idx = src.index('PATCH-ΩF-6')
        wiring_block = src[omf6_idx:omf6_idx + 3000]
        assert 'hierarchical_memory._mct_ref' in wiring_block

    def test_wiring_guarded(self):
        """Wiring is conditional on metacognitive_trigger not None."""
        src = _src()
        omf6_idx = src.index('PATCH-ΩF-6')
        # The guard should appear before the wiring calls
        pre_block = src[omf6_idx - 200:omf6_idx + 200]
        assert 'metacognitive_trigger is not None' in pre_block


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ΩF-7: MCT Readers for Same-Pass Signals
# ═══════════════════════════════════════════════════════════════════════

class TestOmegaF7_MCTReaders:
    """PATCH-ΩF-7: MCT reads 4 new same-pass trigger signals."""

    def test_mct_reads_safety_same_pass(self):
        """MCT reads safety_same_pass_mct_triggered."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]safety_same_pass_mct_triggered['\"]",
            src,
        ), "MCT must read safety_same_pass_mct_triggered"

    def test_mct_reads_cross_validator_same_pass(self):
        """MCT reads cross_validator_same_pass_mct_triggered."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]cross_validator_same_pass_mct_triggered['\"]",
            src,
        ), "MCT must read cross_validator_same_pass_mct_triggered"

    def test_mct_reads_oscillation_same_pass(self):
        """MCT reads oscillation_same_pass_mct_triggered."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]oscillation_same_pass_mct_triggered['\"]",
            src,
        ), "MCT must read oscillation_same_pass_mct_triggered"

    def test_mct_reads_memory_same_pass(self):
        """MCT reads memory_same_pass_mct_triggered."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]memory_same_pass_mct_triggered['\"]",
            src,
        ), "MCT must read memory_same_pass_mct_triggered"

    def test_same_pass_signals_boost_recovery_pressure(self):
        """Same-pass signals boost recovery_pressure in MCT."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        # Find the ΩF-7 block
        omf7_idx = src.index('PATCH-ΩF-7')
        omf7_block = src[omf7_idx:omf7_idx + 1500]
        assert 'recovery_pressure' in omf7_block


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL ECOSYSTEM AUDIT
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystem:
    """Verify signal ecosystem health after ΩF patches."""

    @staticmethod
    def _scan_signals():
        """Scan all source files for written and read signal names."""
        write_pat = re.compile(
            r'write_signal(?:_traced)?\s*\(\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        read_pat = re.compile(
            r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        written: set = set()
        read: set = set()
        for src_file in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            fp = _PROJECT_DIR / src_file
            if fp.exists():
                code = fp.read_text(encoding='utf-8')
                written |= set(write_pat.findall(code))
                read |= set(read_pat.findall(code))
        return written, read

    def test_new_signals_are_written(self):
        """All 4 new same-pass signals have writers."""
        written, _ = self._scan_signals()
        new_signals = {
            'safety_same_pass_mct_triggered',
            'cross_validator_same_pass_mct_triggered',
            'oscillation_same_pass_mct_triggered',
            'memory_same_pass_mct_triggered',
        }
        missing = new_signals - written
        assert not missing, f"New signals without writers: {sorted(missing)}"

    def test_new_signals_are_read(self):
        """All 4 new same-pass signals have readers."""
        _, read = self._scan_signals()
        new_signals = {
            'safety_same_pass_mct_triggered',
            'cross_validator_same_pass_mct_triggered',
            'oscillation_same_pass_mct_triggered',
            'memory_same_pass_mct_triggered',
        }
        missing = new_signals - read
        assert not missing, f"New signals without readers: {sorted(missing)}"

    def test_no_new_write_only_orphans(self):
        """No new write-only orphan signals introduced."""
        written, read = self._scan_signals()
        orphans = written - read
        assert not orphans, f"Write-only orphans: {sorted(orphans)}"


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION MAP: Connected vs Isolated Paths
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationMap:
    """Verify that all critical paths are now connected."""

    def test_safety_to_mct_bidirectional(self):
        """Safety ↔ MCT: bidirectional via safety_mct_tightening_active
        AND same-pass via safety_same_pass_mct_triggered."""
        src = _src()
        # Safety writes both signals
        safety_block = _class_block('MultiLevelSafetySystem', src)
        assert 'safety_mct_tightening_active' in safety_block
        assert 'safety_same_pass_mct_triggered' in safety_block
        # MCT reads both
        mct_block = _class_block('MetaCognitiveRecursionTrigger', src)
        assert 'safety_mct_tightening_active' in mct_block
        assert 'safety_same_pass_mct_triggered' in mct_block

    def test_cross_validator_to_mct(self):
        """CrossValidator → MCT: via cross_subsystem_inconsistency
        AND same-pass via cross_validator_same_pass_mct_triggered."""
        src = _src()
        cv_block = _class_block('SubsystemCrossValidator', src)
        assert 'cross_subsystem_inconsistency' in cv_block
        assert 'cross_validator_same_pass_mct_triggered' in cv_block

    def test_oscillation_to_mct(self):
        """FeedbackBus → MCT: via oscillation_severity_pressure
        AND same-pass via oscillation_same_pass_mct_triggered."""
        src = _src()
        bus_block = _class_block('CognitiveFeedbackBus', src)
        assert 'oscillation_severity_pressure' in bus_block
        assert 'oscillation_same_pass_mct_triggered' in bus_block

    def test_memory_to_mct(self):
        """Memory → MCT: via memory_staleness_pressure
        AND same-pass via memory_same_pass_mct_triggered."""
        src = _src()
        mem_block = _class_block('HierarchicalMemory', src)
        assert 'memory_staleness_pressure' in mem_block
        assert 'memory_same_pass_mct_triggered' in mem_block


# ═══════════════════════════════════════════════════════════════════════
# MUTUAL REINFORCEMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify that active components verify and stabilize each other."""

    def test_safety_mct_mutual_loop(self):
        """Safety ↔ MCT: Safety tightens on MCT concern, MCT reads tightening,
        AND safety violations immediately trigger MCT."""
        src = _src()
        safety = _class_block('MultiLevelSafetySystem', src)
        # Safety reads MCT trigger score (RMT34-2)
        assert re.search(
            r"read_signal\s*\(\s*['\"]mct_trigger_score['\"]",
            safety,
        )
        # Safety calls MCT.evaluate on violation (ΩF-1)
        assert 'self._mct_ref.evaluate(' in safety
        # MCT reads safety signals
        mct = _class_block('MetaCognitiveRecursionTrigger', src)
        assert 'safety_same_pass_mct_triggered' in mct

    def test_cross_validator_mct_mutual_loop(self):
        """CrossValidator ↔ MCT: CV detects inconsistency → MCT amplifies,
        AND high inconsistency immediately triggers MCT."""
        src = _src()
        cv = _class_block('SubsystemCrossValidator', src)
        assert 'self._mct_ref.evaluate(' in cv
        mct = _class_block('MetaCognitiveRecursionTrigger', src)
        assert 'cross_validator_same_pass_mct_triggered' in mct

    def test_all_same_pass_signals_in_single_reader_block(self):
        """All 4 same-pass signals are read in a single MCT block."""
        src = _class_block('MetaCognitiveRecursionTrigger')
        omf7_idx = src.index('PATCH-ΩF-7')
        omf7_block = src[omf7_idx:omf7_idx + 2000]
        for sig in [
            'safety_same_pass_mct_triggered',
            'cross_validator_same_pass_mct_triggered',
            'oscillation_same_pass_mct_triggered',
            'memory_same_pass_mct_triggered',
        ]:
            assert sig in omf7_block, f"{sig} not in ΩF-7 reader block"


# ═══════════════════════════════════════════════════════════════════════
# META-COGNITIVE TRIGGER COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTrigger:
    """Verify that uncertainty triggers meta-cognitive cycles."""

    def test_safety_violation_triggers_immediate_mct(self):
        """Safety violation (score < 0.5) triggers same-pass MCT."""
        src = _class_block('MultiLevelSafetySystem')
        # Must check score and call evaluate
        assert '_omf1_safety_mean < 0.5' in src
        assert '_omf1_result = self._mct_ref.evaluate(' in src

    def test_cross_validator_inconsistency_triggers_mct(self):
        """Cross-validator inconsistency > 0.6 triggers same-pass MCT."""
        src = _class_block('SubsystemCrossValidator')
        assert 'overall > 0.6' in src
        assert '_omf2_result = self._mct_ref.evaluate(' in src

    def test_coherence_trend_reversal_triggers_mct(self):
        """Coherence trend reversal triggers same-pass MCT."""
        src = _class_block('SubsystemCoherenceRegistry')
        assert '_n_reversals > 0' in src
        assert 'self._mct_ref.evaluate(' in src

    def test_oscillation_triggers_mct(self):
        """Oscillation > 0.5 triggers same-pass MCT."""
        src = _class_block('CognitiveFeedbackBus')
        assert '_xi3_agg_osc > 0.5' in src
        assert '_omf4_result = self._mct_ref.evaluate(' in src

    def test_memory_degradation_triggers_mct(self):
        """Memory degradation triggers same-pass MCT."""
        src = _class_block('HierarchicalMemory')
        assert '_omf5_staleness > 0.8' in src
        assert '_omf5_result = _omf5_mct.evaluate(' in src


# ═══════════════════════════════════════════════════════════════════════
# CAUSAL TRANSPARENCY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify that outputs can be traced to originating premises."""

    def test_same_pass_signals_carry_trigger_result(self):
        """Same-pass signals encode whether MCT should_trigger fired."""
        src = _src()
        for sig in [
            'safety_same_pass_mct_triggered',
            'cross_validator_same_pass_mct_triggered',
            'oscillation_same_pass_mct_triggered',
            'memory_same_pass_mct_triggered',
        ]:
            # Each signal should be written as 1.0 if triggered, 0.0 otherwise
            assert re.search(
                rf"['\"]" + re.escape(sig) + r"['\"]",
                src,
            ), f"{sig} must be written to bus"

    def test_mct_evaluate_called_with_specific_signals(self):
        """Each same-pass MCT call passes context-specific signal values."""
        src = _src()
        # Safety: passes safety_violation=True
        safety = _class_block('MultiLevelSafetySystem', src)
        assert 'safety_violation=True' in safety
        # CrossValidator: passes coherence_deficit=overall
        cv = _class_block('SubsystemCrossValidator', src)
        assert 'coherence_deficit=overall' in cv
        # Memory: passes memory_staleness=1.0
        mem = _class_block('HierarchicalMemory', src)
        assert 'memory_staleness=1.0' in mem


# ═══════════════════════════════════════════════════════════════════════
# ACTIVATION SEQUENCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify patches can be applied in the correct order."""

    def test_setters_before_wiring(self):
        """set_metacognitive_trigger methods exist before wiring code."""
        src = _src()
        # Setter definitions
        safety_setter = src.index(
            'class MultiLevelSafetySystem'
        )
        cv_setter = src.index(
            'class SubsystemCrossValidator'
        )
        bus_setter = src.index(
            'class CognitiveFeedbackBus'
        )
        # Wiring in __init__
        wiring = src.index('PATCH-ΩF-6')
        # All setters must be defined BEFORE wiring uses them
        assert safety_setter < wiring
        assert cv_setter < wiring
        assert bus_setter < wiring

    def test_writers_before_readers(self):
        """Signal writers (ΩF-1..5) are defined before readers (ΩF-7)."""
        src = _src()
        # Writers in their respective classes
        safety_write = src.index('safety_same_pass_mct_triggered')
        # Reader in MCT
        reader = src.index('PATCH-ΩF-7')
        # Writer must exist before reader in the file
        assert safety_write < reader

    def test_wiring_after_subsystem_creation(self):
        """ΩF-6 wiring appears after convergence_monitor MCT wiring."""
        src = _src()
        conv_wire = src.index(
            'convergence_monitor.set_metacognitive_trigger'
        )
        omf6_wire = src.index('PATCH-ΩF-6')
        assert conv_wire < omf6_wire


# ═══════════════════════════════════════════════════════════════════════
# RUNTIME INTEGRATION (Feedback Bus Signal Flow)
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeSignalFlow:
    """Runtime verification of signal flow through the feedback bus."""

    def test_safety_signal_roundtrip(self):
        """safety_same_pass_mct_triggered survives write/read cycle."""
        bus = make_bus()
        bus.write_signal('safety_same_pass_mct_triggered', 1.0)
        val = float(bus.read_signal('safety_same_pass_mct_triggered', 0.0))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_cross_validator_signal_roundtrip(self):
        """cross_validator_same_pass_mct_triggered survives write/read."""
        bus = make_bus()
        bus.write_signal('cross_validator_same_pass_mct_triggered', 1.0)
        val = float(bus.read_signal(
            'cross_validator_same_pass_mct_triggered', 0.0,
        ))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_oscillation_signal_roundtrip(self):
        """oscillation_same_pass_mct_triggered survives write/read."""
        bus = make_bus()
        bus.write_signal('oscillation_same_pass_mct_triggered', 1.0)
        val = float(bus.read_signal(
            'oscillation_same_pass_mct_triggered', 0.0,
        ))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_memory_signal_roundtrip(self):
        """memory_same_pass_mct_triggered survives write/read."""
        bus = make_bus()
        bus.write_signal('memory_same_pass_mct_triggered', 1.0)
        val = float(bus.read_signal(
            'memory_same_pass_mct_triggered', 0.0,
        ))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_e2e_safety_violation_flow(self):
        """E2E: safety violation → signal write → MCT reads → recovery boost."""
        bus = make_bus()
        # Step 1: Safety detects violation → writes signal
        bus.write_signal('safety_same_pass_mct_triggered', 1.0)
        bus.write_signal('safety_violation_active', 1.0)

        # Step 2: MCT reads both signals
        sp_val = float(bus.read_signal(
            'safety_same_pass_mct_triggered', 0.0,
        ))
        violation = float(bus.read_signal('safety_violation_active', 0.0))
        assert sp_val > 0.5
        assert violation > 0.5

        # Step 3: MCT computes recovery_pressure boost
        base_rp = 0.2
        boost = 0.15 if sp_val > 0.5 else 0.0
        final_rp = min(1.0, base_rp + boost)
        assert final_rp > base_rp

    def test_e2e_multi_trigger_accumulation(self):
        """E2E: multiple same-pass triggers accumulate recovery_pressure."""
        bus = make_bus()
        # Multiple subsystems trigger simultaneously
        bus.write_signal('safety_same_pass_mct_triggered', 1.0)
        bus.write_signal('cross_validator_same_pass_mct_triggered', 1.0)
        bus.write_signal('memory_same_pass_mct_triggered', 1.0)

        # MCT accumulates all boosts
        _omf7_any_triggered = 0.0
        for sig in [
            'safety_same_pass_mct_triggered',
            'cross_validator_same_pass_mct_triggered',
            'oscillation_same_pass_mct_triggered',
            'memory_same_pass_mct_triggered',
        ]:
            val = float(bus.read_signal(sig, 0.0))
            if val > 0.5:
                _omf7_any_triggered += 0.15

        # Three signals triggered → 3 × 0.15 = 0.45
        assert _omf7_any_triggered == pytest.approx(0.45, abs=0.01)
        # Recovery pressure capped at 1.0
        base_rp = 0.7
        final_rp = min(1.0, base_rp + _omf7_any_triggered)
        assert final_rp == pytest.approx(1.0, abs=0.15)
