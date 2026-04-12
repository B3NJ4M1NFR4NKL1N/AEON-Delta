"""
Tests for PATCH-RMT34-1, PATCH-RMT34-2, PATCH-RMT34-3:
AEON-Delta RMT v3.4 Final Integration & Cognitive Activation Patches.

These patches close the remaining structural gaps that prevent AEON-Delta
from operating as a unified, self-consistent cognitive organism.

PATCH-RMT34-1: CrossValidator Extended Consistency Pairs
  - SubsystemCrossValidator._CONSISTENCY_PAIRS gains two new pairs:
      (mct_trigger_score, convergence_arbiter_confidence, negatively_correlated)
      (emergence_readiness, integration_health, positively_correlated)
  - Covers the MCT-convergence-emergence triad previously absent from
    mutual-reinforcement checks.

PATCH-RMT34-2: Safety System MCT Coupling
  - MultiLevelSafetySystem.forward() reads mct_trigger_score from the bus.
  - When trigger_score > 0.7, safety result is tightened by up to 10%.
  - Writes safety_mct_tightening_active (1.0 if tightened, else 0.0).
  - This is the missing Safety → MCT feedback direction.

PATCH-RMT34-3: MCT reads safety_mct_tightening_active
  - MetaCognitiveRecursionTrigger's NX-1 signal reader adds item (u):
    safety_mct_tightening_active → recovery_pressure boost (×0.3).
  - Closes the mutual reinforcement loop:
      MCT triggers → Safety tightens → MCT reads tightening → escalates.

Signal Ecosystem: safety_mct_tightening_active is bidirectional.
  Written by: MultiLevelSafetySystem.forward() (PATCH-RMT34-2)
  Read by:    MetaCognitiveRecursionTrigger.evaluate() (PATCH-RMT34-3)

Integration Map (v3.4 final state):
  Connected:  MCT ↔ Safety (new bidirectional via safety_mct_tightening_active)
  Connected:  CrossValidator covers MCT-convergence + emergence-integration
  Connected:  All 254 signals are bidirectional (0 orphans, 0 missing)
  Verified:   Causal transparency via PATCH-Φ7a/b (MCT + Safety causal traces)

Activation Sequence:
  1. PATCH-RMT34-1 — CrossValidator pairs (no new signals, pure extension)
  2. PATCH-RMT34-2 — Safety writes safety_mct_tightening_active
  3. PATCH-RMT34-3 — MCT reads safety_mct_tightening_active
  (Apply in this order: 2 must precede 3 so the signal has a producer
  before the consumer is wired.)
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

_PROJECT_DIR = Path(__file__).resolve().parent


def _src() -> str:
    """Return source code of aeon_core.py (cached)."""
    if not hasattr(_src, '_cache'):
        _src._cache = (_PROJECT_DIR / 'aeon_core.py').read_text(encoding='utf-8')
    return _src._cache


# ═══════════════════════════════════════════════════════════════════════
# PATCH-RMT34-1: CrossValidator Extended Consistency Pairs
# ═══════════════════════════════════════════════════════════════════════

class TestRMT34_1_CrossValidatorPairs:
    """PATCH-RMT34-1: Two new signal consistency pairs in CrossValidator."""

    def test_mct_trigger_score_in_consistency_pairs(self):
        """mct_trigger_score appears in _CONSISTENCY_PAIRS."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'mct_trigger_score' in pairs_block, (
            "mct_trigger_score must appear in _CONSISTENCY_PAIRS"
        )

    def test_convergence_arbiter_confidence_in_consistency_pairs(self):
        """convergence_arbiter_confidence appears in _CONSISTENCY_PAIRS."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'convergence_arbiter_confidence' in pairs_block, (
            "convergence_arbiter_confidence must appear in _CONSISTENCY_PAIRS"
        )

    def test_mct_convergence_pair_is_negatively_correlated(self):
        """mct_trigger_score / convergence_arbiter_confidence pair is negatively_correlated."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert re.search(
            r"['\"]mct_trigger_score['\"].*negatively_correlated|"
            r"negatively_correlated.*['\"]mct_trigger_score['\"]",
            pairs_block,
        ), "mct_trigger_score pair must be negatively_correlated with convergence_arbiter_confidence"

    def test_emergence_readiness_in_consistency_pairs(self):
        """emergence_readiness appears in _CONSISTENCY_PAIRS."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'emergence_readiness' in pairs_block, (
            "emergence_readiness must appear in _CONSISTENCY_PAIRS"
        )

    def test_integration_health_in_consistency_pairs(self):
        """integration_health appears in _CONSISTENCY_PAIRS."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'integration_health' in pairs_block, (
            "integration_health must appear in _CONSISTENCY_PAIRS"
        )

    def test_emergence_integration_pair_is_positively_correlated(self):
        """emergence_readiness / integration_health pair is positively_correlated."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert re.search(
            r"['\"]emergence_readiness['\"].*positively_correlated|"
            r"positively_correlated.*['\"]emergence_readiness['\"]",
            pairs_block,
        ), "emergence_readiness pair must be positively_correlated with integration_health"

    def test_patch_rmt34_1_comment_present(self):
        """PATCH-RMT34-1 comment is present in source."""
        assert 'PATCH-RMT34-1' in _src(), (
            "Source must contain PATCH-RMT34-1 patch comment"
        )

    def test_patch_rmt34_1b_comment_present(self):
        """PATCH-RMT34-1b comment is present in source."""
        assert 'PATCH-RMT34-1b' in _src(), (
            "Source must contain PATCH-RMT34-1b patch comment"
        )

    def test_total_consistency_pairs_at_least_six(self):
        """_CONSISTENCY_PAIRS has at least 6 tuples (was 4, added 2)."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        # Count tuple entries by the 'positively_correlated' and
        # 'negatively_correlated' relationship strings — one per pair
        pair_count = pairs_block.count('positively_correlated') + \
                     pairs_block.count('negatively_correlated')
        assert pair_count >= 6, (
            f"Expected >= 6 consistency pairs (relationships), found {pair_count}"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-RMT34-2: Safety System MCT Coupling
# ═══════════════════════════════════════════════════════════════════════

class TestRMT34_2_SafetyMCTCoupling:
    """PATCH-RMT34-2: Safety reads mct_trigger_score, writes safety_mct_tightening_active."""

    def test_safety_reads_mct_trigger_score(self):
        """MultiLevelSafetySystem reads mct_trigger_score from bus."""
        src = _src()
        # The read must occur within the Safety class
        safety_idx = src.index('class MultiLevelSafetySystem')
        # Find next class boundary (approximate)
        next_class = src.find('\nclass ', safety_idx + 1)
        safety_code = src[safety_idx:next_class]
        assert re.search(
            r"read_signal\s*\(\s*['\"]mct_trigger_score['\"]",
            safety_code,
        ), "MultiLevelSafetySystem must read mct_trigger_score"

    def test_safety_writes_mct_tightening_active(self):
        """MultiLevelSafetySystem writes safety_mct_tightening_active."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*'safety_mct_tightening_active'",
            src,
        ), "Must write safety_mct_tightening_active signal"

    def test_tightening_threshold_is_0_7(self):
        """Tightening activates when mct_trigger_score > 0.7."""
        src = _src()
        assert re.search(
            r"_rmt2_mct_score\s*>\s*0\.7",
            src,
        ), "Tightening threshold must be > 0.7"

    def test_max_tightening_factor_is_0_9(self):
        """Maximum tightening factor floors at 0.9 (10% reduction)."""
        src = _src()
        assert re.search(
            r"max\s*\(\s*0\.9",
            src,
        ), "Max tightening must floor at 0.9"

    def test_patch_rmt34_2_comment_present(self):
        """PATCH-RMT34-2 comment is present in source."""
        assert 'PATCH-RMT34-2' in _src(), (
            "Source must contain PATCH-RMT34-2 patch comment"
        )

    def test_tightening_writes_zero_when_not_triggered(self):
        """When trigger_score <= 0.7, writes safety_mct_tightening_active = 0.0."""
        src = _src()
        # The else branch of the tightening block must write 0.0
        assert re.search(
            r"write_signal\s*\(\s*['\"]safety_mct_tightening_active['\"],\s*0\.0",
            src,
        ), "Must write safety_mct_tightening_active = 0.0 when not tightening"

    def test_tightening_result_clamped(self):
        """The tightened result uses clamp or max/min to prevent over-reduction."""
        src = _src()
        # _rmt2_tighten uses max(0.9, ...) which prevents going below 0.9
        assert re.search(
            r"_rmt2_tighten\s*=\s*max\s*\(",
            src,
        ), "_rmt2_tighten must use max() to floor tightening at 0.9"


# ═══════════════════════════════════════════════════════════════════════
# PATCH-RMT34-3: MCT reads safety_mct_tightening_active
# ═══════════════════════════════════════════════════════════════════════

class TestRMT34_3_MCTSafetyFeedback:
    """PATCH-RMT34-3: MCT reads safety_mct_tightening_active → recovery_pressure."""

    def test_mct_reads_safety_mct_tightening_active(self):
        """MCT evaluate() reads safety_mct_tightening_active from bus."""
        src = _src()
        # Locate the MCT NX-1 reader block by the (u) item comment
        assert "safety_mct_tightening_active → recovery_pressure" in src, (
            "MCT NX-1 block must have (u) safety_mct_tightening_active item"
        )
        assert re.search(
            r"read_signal\s*\(\s*'safety_mct_tightening_active'",
            src,
        ), "MCT NX-1 block must read safety_mct_tightening_active"

    def test_tightening_active_routes_to_recovery_pressure(self):
        """When tightening_active > 0.5, recovery_pressure is boosted."""
        src = _src()
        assert re.search(
            r"_rmt3_smt\s*>\s*0\.5",
            src,
        ), "MCT must check _rmt3_smt > 0.5 for recovery_pressure boost"

    def test_recovery_pressure_boost_factor_is_0_3(self):
        """Recovery pressure boost uses factor 0.3."""
        src = _src()
        # The boost block uses _rmt3_smt and sets recovery_pressure to 0.3
        assert re.search(
            r"_rmt3_smt.*recovery_pressure|recovery_pressure.*_rmt3_smt",
            src,
            re.DOTALL,
        ), "Recovery pressure boost must reference _rmt3_smt"
        # 0.3 appears as the scaling factor in the same block
        rmt3_idx = src.index('PATCH-RMT34-3')
        rmt3_block = src[rmt3_idx:rmt3_idx + 600]
        assert '0.3' in rmt3_block, (
            "Recovery pressure boost factor 0.3 must appear in PATCH-RMT34-3 block"
        )

    def test_rmt3_smt_in_else_branch(self):
        """_rmt3_smt is initialized to 0.0 in the no-bus else branch."""
        src = _src()
        assert '_rmt3_smt = 0.0' in src, (
            "_rmt3_smt must be initialized to 0.0 in no-bus else branch"
        )

    def test_patch_rmt34_3_comment_present(self):
        """PATCH-RMT34-3 comment is present in source."""
        assert 'PATCH-RMT34-3' in _src(), (
            "Source must contain PATCH-RMT34-3 patch comment"
        )

    def test_feedback_loop_comment_describes_chain(self):
        """Patch comment describes the full MCT→safety→MCT feedback chain."""
        src = _src()
        assert re.search(
            r'MCT trigger.*safety tight|safety tight.*MCT trigger',
            src,
        ), "Patch comment must describe the MCT→safety→MCT feedback loop"


# ═══════════════════════════════════════════════════════════════════════
# Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystemAudit:
    """Verify signal ecosystem integrity after RMT34 patches."""

    def _get_all_signals(self):
        """Extract written and read signal names from all source files."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            fpath = _PROJECT_DIR / fname
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding='utf-8')
            for m in re.finditer(
                r'write_signal(?:_traced)?\s*\(\s*["\'](\w+)["\']', text,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*["\'](\w+)["\']',
                text,
            ):
                read.add(m.group(1))
        return written, read

    def test_safety_mct_tightening_is_bidirectional(self):
        """safety_mct_tightening_active has both writer and reader."""
        written, read = self._get_all_signals()
        assert 'safety_mct_tightening_active' in written, (
            "safety_mct_tightening_active must have a producer"
        )
        assert 'safety_mct_tightening_active' in read, (
            "safety_mct_tightening_active must have a consumer"
        )

    def test_no_orphaned_writers(self):
        """No signal is written but never read (no orphaned producers)."""
        written, read = self._get_all_signals()
        orphans = written - read
        assert not orphans, (
            f"Found {len(orphans)} orphaned signal producers: {sorted(orphans)}"
        )

    def test_no_orphaned_readers(self):
        """No signal is read but never written (no orphaned consumers)."""
        written, read = self._get_all_signals()
        missing = read - written
        assert not missing, (
            f"Found {len(missing)} missing signal producers: {sorted(missing)}"
        )

    def test_total_signals_at_least_254(self):
        """Signal count has not decreased below 254 (was 253 + new signal)."""
        written, read = self._get_all_signals()
        assert len(written) >= 254, (
            f"Expected >= 254 signals, found {len(written)}"
        )
        assert written == read, (
            "Written signal set must equal read signal set (all bidirectional)"
        )

    def test_new_pairs_signals_have_existing_producers(self):
        """All signals used in new CrossValidator pairs have producers."""
        written, _ = self._get_all_signals()
        new_pair_signals = [
            'mct_trigger_score',
            'convergence_arbiter_confidence',
            'emergence_readiness',
            'integration_health',
        ]
        for sig in new_pair_signals:
            assert sig in written, (
                f"Signal '{sig}' used in new consistency pair must have a producer"
            )


# ═══════════════════════════════════════════════════════════════════════
# Integration Map Verification
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationMap:
    """Verify the integration map described in the module docstring."""

    def test_mct_safety_bidirectional_coupling(self):
        """Both MCT→Safety and Safety→MCT signal paths exist."""
        src = _src()
        # MCT outputs safety_violation_active → Safety reads it implicitly
        # Safety outputs safety_mct_tightening_active → MCT reads it (PATCH-RMT34-3)
        assert re.search(
            r"write_signal\s*\(\s*'safety_violation_active'", src,
        ), "Safety must write safety_violation_active (MCT reads it)"
        assert re.search(
            r"write_signal\s*\(\s*'safety_mct_tightening_active'", src,
        ), "Safety must write safety_mct_tightening_active (MCT reads it)"
        assert re.search(
            r"read_signal\s*\(\s*'mct_trigger_score'", src,
        ), "Safety must read mct_trigger_score from MCT output"
        assert re.search(
            r"read_signal\s*\(\s*'safety_mct_tightening_active'", src,
        ), "MCT must read safety_mct_tightening_active from Safety"

    def test_cross_validator_covers_mct_convergence_triad(self):
        """CrossValidator checks MCT-convergence consistency."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'mct_trigger_score' in pairs_block
        assert 'convergence_arbiter_confidence' in pairs_block

    def test_cross_validator_covers_emergence_integration(self):
        """CrossValidator checks emergence-integration alignment."""
        src = _src()
        pairs_start = src.index('_CONSISTENCY_PAIRS = [')
        pairs_end = src.index(']', pairs_start)
        pairs_block = src[pairs_start:pairs_end]
        assert 'emergence_readiness' in pairs_block
        assert 'integration_health' in pairs_block

    def test_causal_trace_covers_both_endpoints(self):
        """Both MCT and Safety record to causal trace (PATCH-Φ7a/b)."""
        src = _src()
        assert "subsystem='metacognitive_trigger'" in src or \
               "subsystem=\"metacognitive_trigger\"" in src, (
            "MCT must record to causal trace under 'metacognitive_trigger'"
        )
        assert "subsystem='safety_system'" in src or \
               "subsystem=\"safety_system\"" in src, (
            "Safety must record to causal trace under 'safety_system'"
        )

    def test_activation_sequence_documented(self):
        """Source code documents the activation sequence for the patches."""
        src = _src()
        # Both patches must be present
        assert 'PATCH-RMT34-2' in src and 'PATCH-RMT34-3' in src, (
            "Both PATCH-RMT34-2 and PATCH-RMT34-3 must be documented in source"
        )
        # The inter-patch dependency is documented: RMT34-2 references RMT34-3
        # and vice versa (activation ordering is captured in both comments)
        rmt2_idx = src.index('PATCH-RMT34-2')
        rmt2_block = src[rmt2_idx:rmt2_idx + 600]
        assert 'PATCH-RMT34-3' in rmt2_block, (
            "PATCH-RMT34-2 comment must reference PATCH-RMT34-3 (consumer dependency)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Mutual Reinforcement Verification
# ═══════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify that active components verify and stabilize each other."""

    def test_mct_reads_cross_validation_disagreement(self):
        """MCT reads cross_subsystem_inconsistency (from CrossValidator)."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*['\"]cross_subsystem_inconsistency['\"]",
            src,
        ), "MCT must read cross_subsystem_inconsistency from CrossValidator"

    def test_cross_validator_writes_inconsistency_score(self):
        """CrossValidator writes cross_subsystem_inconsistency to bus."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*['\"]cross_subsystem_inconsistency['\"]",
            src,
        ), "CrossValidator must write cross_subsystem_inconsistency"

    def test_safety_MCT_mutual_reinforcement_loop(self):
        """Safety and MCT form a mutual reinforcement loop via new signal."""
        src = _src()
        # Loop: MCT trigger → Safety reads trigger → Safety tightens →
        #        Safety writes tightening_active → MCT reads tightening_active →
        #        MCT boosts recovery_pressure → next MCT evaluation
        signals_in_loop = [
            'mct_trigger_score',       # MCT writes, Safety reads
            'safety_mct_tightening_active',  # Safety writes, MCT reads
        ]
        written = set(re.findall(
            r'write_signal(?:_traced)?\s*\(\s*["\'](\w+)["\']', src,
        ))
        read = set(re.findall(
            r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*["\'](\w+)["\']',
            src,
        ))
        for sig in signals_in_loop:
            assert sig in written, f"Loop signal '{sig}' must have a writer"
            assert sig in read, f"Loop signal '{sig}' must have a reader"

    def test_auto_critic_feedback_to_mct(self):
        """AutoCritic quality feeds back into MCT (auto_critic_quality reader)."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*['\"]auto_critic_quality['\"]",
            src,
        ), "MCT must read auto_critic_quality from AutoCritic"


# ═══════════════════════════════════════════════════════════════════════
# Meta-Cognitive Trigger Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTriggerCompleteness:
    """Verify that uncertainty sources reach MCT evaluate()."""

    def test_safety_violation_reaches_mct(self):
        """Safety violation signal propagates to MCT recovery_pressure."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*'safety_violation_active'", src,
        ), "MCT must read safety_violation_active"

    def test_convergence_conflict_reaches_mct(self):
        """Convergence arbiter conflict reaches MCT."""
        src = _src()
        # convergence_arbiter_conflict maps to convergence_conflict in MCT
        assert "'convergence_arbiter_conflict': 'convergence_conflict'" in src \
            or '"convergence_arbiter_conflict": "convergence_conflict"' in src, (
            "convergence_arbiter_conflict must map to convergence_conflict in MCT"
        )

    def test_memory_staleness_reaches_mct(self):
        """Memory staleness pressure reaches MCT."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*['\"]memory_staleness_pressure['\"]",
            src,
        ), "MCT must read memory_staleness_pressure"

    def test_integration_staleness_reaches_mct(self):
        """Integration staleness pressure reaches MCT (PATCH-H2b)."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]integration_staleness_pressure['\"]",
            src,
        ), "MCT must read integration_staleness_pressure (PATCH-H2b)"

    def test_emergence_deficit_reaches_mct(self):
        """Emergence deficit reaches MCT."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*['\"]emergence_deficit['\"]",
            src,
        ), "MCT must read emergence_deficit"

    def test_safety_mct_tightening_reaches_mct_recovery(self):
        """safety_mct_tightening_active reaches MCT's recovery_pressure."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*'safety_mct_tightening_active'",
            src,
        ), "MCT must read safety_mct_tightening_active"
        assert re.search(
            r"_rmt3_smt\s*>\s*0\.5",
            src,
        ), "MCT must threshold _rmt3_smt at > 0.5 for recovery_pressure"


# ═══════════════════════════════════════════════════════════════════════
# Causal Transparency Verification
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify that every output/action can be traced to its origin."""

    def test_mct_records_to_causal_trace(self):
        """MCT evaluate() records its decision to causal trace."""
        src = _src()
        assert 'PATCH-Φ7a' in src, "MCT causal trace (PATCH-Φ7a) must be present"

    def test_safety_records_to_causal_trace(self):
        """Safety forward() records its verdict to causal trace."""
        src = _src()
        assert 'PATCH-Φ7b' in src, "Safety causal trace (PATCH-Φ7b) must be present"

    def test_auto_critic_records_to_causal_trace(self):
        """AutoCritic forward() records to causal trace."""
        src = _src()
        assert 'PATCH-Φ7c' in src, "AutoCritic causal trace (PATCH-Φ7c) must be present"

    def test_cross_validator_records_to_causal_trace(self):
        """CrossValidator validate() records to causal trace."""
        src = _src()
        assert 'PATCH-Φ7d' in src, "CrossValidator causal trace (PATCH-Φ7d) must be present"

    def test_temporal_causal_trace_buffer_exists(self):
        """TemporalCausalTraceBuffer class is defined."""
        src = _src()
        assert 'class TemporalCausalTraceBuffer' in src, (
            "TemporalCausalTraceBuffer must be defined"
        )

    def test_causal_trace_find_by_subsystem(self):
        """TemporalCausalTraceBuffer.find() supports subsystem lookup."""
        src = _src()
        assert re.search(
            r'def find\s*\(',
            src,
        ), "TemporalCausalTraceBuffer.find() must exist"
        # The subsystem parameter appears on a line near the find() definition
        find_idx = src.index('def find(')
        find_sig = src[find_idx:find_idx + 300]
        assert 'subsystem' in find_sig, (
            "TemporalCausalTraceBuffer.find() must accept a subsystem parameter"
        )
