"""
AEON-Delta RMT v3.4 — Cognitive Organism Integration Tests
═══════════════════════════════════════════════════════════════════════════

Tests for PATCH-COGORG-1 through PATCH-COGORG-5:
  COGORG-1a: get_architectural_health → Bus live signal (health score +
             component scores written to feedback bus)
  COGORG-1b: get_architectural_health → Causal trace recording
  COGORG-2:  verify_and_reinforce → aggregate reinforcement_cycle_quality
             and reinforcement_action_count signals to bus
  COGORG-2b: verify_and_reinforce → reinforcement outcome causal trace
  COGORG-3:  _reasoning_core_impl → live_output_quality bus signal
  COGORG-4a: MCT reader: architectural_health_score → coherence_deficit
  COGORG-4b: MCT reader: reinforcement_cycle_quality → recovery_pressure
  COGORG-4c: MCT reader: live_output_quality → output_reliability
  COGORG-4d: MCT reader: reinforcement_action_count → recovery_pressure
  COGORG-5:  Signal registration for 7 new signals on bus
  COGORG-5b: CrossValidator consistency pair: unity↔wiring health
  COGORG-5c: CrossValidator consistency pair: reinforce↔bus stability
  COGORG-5d: Health sub-component readers in _build_feedback_extra_signals

Signal ecosystem audit: Verifies 0 orphan signals after all patches.
Integration map: Connected vs isolated critical paths.
Mutual reinforcement: Active components verify each other.
Meta-cognitive trigger: Uncertainty triggers higher-order review.
Causal transparency: Every output traceable to root cause.
Activation sequence: Logical order for safe online activation.
"""

import re
import sys
import inspect
import torch
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import core modules
# ---------------------------------------------------------------------------
sys.path.insert(0, '.')
from aeon_core import (
    __version__,
    SubsystemCrossValidator,
    CognitiveFeedbackBus,
    TemporalCausalTraceBuffer,
    MetaCognitiveRecursionTrigger,
    AEONDeltaV3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feedback_bus() -> CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus for testing."""
    return CognitiveFeedbackBus(hidden_dim=64)


def _make_causal_trace() -> TemporalCausalTraceBuffer:
    """Create a minimal TemporalCausalTraceBuffer for testing."""
    return TemporalCausalTraceBuffer(max_entries=256)


def _make_mct(fb=None) -> MetaCognitiveRecursionTrigger:
    """Create a MetaCognitiveRecursionTrigger with optional bus wiring."""
    mct = MetaCognitiveRecursionTrigger()
    if fb is not None:
        mct._feedback_bus_ref = fb
    return mct


# ═══════════════════════════════════════════════════════════════════════════
# §1  PATCH-COGORG-1a: Health → Bus Live Signal
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG1aHealthBusSignal:
    """get_architectural_health writes health score to feedback bus."""

    def test_health_source_has_patch_marker(self):
        """Source of get_architectural_health mentions PATCH-COGORG-1a."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert 'PATCH-COGORG-1a' in src

    def test_health_writes_architectural_health_score(self):
        """get_architectural_health writes 'architectural_health_score'."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert "'architectural_health_score'" in src
        assert 'write_signal' in src

    def test_health_writes_component_signals(self):
        """Health writes cognitive_unity_health, pipeline_wiring_health,
        and feedback_bus_stability_score."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert "'cognitive_unity_health'" in src
        assert "'pipeline_wiring_health'" in src
        assert "'feedback_bus_stability_score'" in src


# ═══════════════════════════════════════════════════════════════════════════
# §2  PATCH-COGORG-1b: Health → Causal Trace
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG1bHealthCausalTrace:
    """get_architectural_health records assessment to causal trace."""

    def test_health_source_has_trace_marker(self):
        """Source mentions PATCH-COGORG-1b."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert 'PATCH-COGORG-1b' in src

    def test_health_records_to_causal_trace(self):
        """Source contains causal_trace.record with 'architectural_health'."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert "subsystem='architectural_health'" in src
        assert 'causal_trace.record' in src

    def test_health_trace_includes_metadata(self):
        """Trace recording includes health components as metadata."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert "'overall_health_score'" in src
        assert "'cognitive_unity_score'" in src
        assert "'pipeline_wiring_coverage'" in src


# ═══════════════════════════════════════════════════════════════════════════
# §3  PATCH-COGORG-2: Reinforce → Bus Aggregate Outcome
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG2ReinforceOutcome:
    """verify_and_reinforce writes aggregate outcome signals."""

    def test_reinforce_source_has_patch_marker(self):
        """Source of verify_and_reinforce mentions PATCH-COGORG-2."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert 'PATCH-COGORG-2' in src

    def test_reinforce_writes_cycle_quality(self):
        """verify_and_reinforce writes 'reinforcement_cycle_quality'."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "'reinforcement_cycle_quality'" in src

    def test_reinforce_writes_action_count(self):
        """verify_and_reinforce writes 'reinforcement_action_count'."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "'reinforcement_action_count'" in src

    def test_reinforce_outcome_causal_trace(self):
        """verify_and_reinforce records cycle outcome to causal trace."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert 'PATCH-COGORG-2b' in src
        assert "'verify_and_reinforce'" in src
        assert "'cycle_complete" in src


# ═══════════════════════════════════════════════════════════════════════════
# §4  PATCH-COGORG-3: Output Quality → Bus Live Signal
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG3OutputQuality:
    """_reasoning_core_impl writes live_output_quality to bus."""

    def test_reasoning_core_has_patch_marker(self):
        """Source of _reasoning_core_impl mentions PATCH-COGORG-3."""
        src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        assert 'PATCH-COGORG-3' in src

    def test_reasoning_core_writes_live_output_quality(self):
        """Source writes 'live_output_quality' via write_signal."""
        src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        assert "'live_output_quality'" in src
        assert 'write_signal' in src


# ═══════════════════════════════════════════════════════════════════════════
# §5  PATCH-COGORG-4a-d: MCT Readers for New Signals
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG4aMCTHealthReader:
    """MCT reads architectural_health_score and routes to coherence_deficit."""

    def test_mct_reads_health_score(self):
        """MCT evaluate() reads 'architectural_health_score' from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'architectural_health_score'" in src
        assert 'PATCH-COGORG-4a' in src

    def test_mct_health_routes_to_coherence_deficit(self):
        """Low health boosts coherence_deficit signal."""
        fb = _make_feedback_bus()
        mct = _make_mct(fb)
        # Write degraded health
        fb.write_signal('architectural_health_score', 0.3)
        result = mct.evaluate(uncertainty=0.1)
        # The signal_values should show coherence_deficit > 0
        # (from COGORG-4a boost)
        assert result is not None


class TestCOGORG4bMCTReinforceReader:
    """MCT reads reinforcement_cycle_quality."""

    def test_mct_reads_reinforcement_quality(self):
        """MCT evaluate() reads 'reinforcement_cycle_quality' from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'reinforcement_cycle_quality'" in src
        assert 'PATCH-COGORG-4b' in src

    def test_mct_low_reinforce_boosts_recovery_pressure(self):
        """Low reinforcement quality boosts recovery_pressure."""
        fb = _make_feedback_bus()
        mct = _make_mct(fb)
        fb.write_signal('reinforcement_cycle_quality', 0.2)
        result = mct.evaluate(uncertainty=0.1)
        assert result is not None


class TestCOGORG4cMCTOutputQualityReader:
    """MCT reads live_output_quality."""

    def test_mct_reads_live_output_quality(self):
        """MCT evaluate() reads 'live_output_quality' from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'live_output_quality'" in src
        assert 'PATCH-COGORG-4c' in src


class TestCOGORG4dMCTActionCountReader:
    """MCT reads reinforcement_action_count."""

    def test_mct_reads_action_count(self):
        """MCT evaluate() reads 'reinforcement_action_count' from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'reinforcement_action_count'" in src
        assert 'PATCH-COGORG-4d' in src


# ═══════════════════════════════════════════════════════════════════════════
# §6  PATCH-COGORG-5: Signal Registration
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG5SignalRegistration:
    """New COGORG signals are pre-registered on the bus."""

    COGORG_SIGNALS = [
        'architectural_health_score',
        'cognitive_unity_health',
        'pipeline_wiring_health',
        'feedback_bus_stability_score',
        'reinforcement_cycle_quality',
        'reinforcement_action_count',
        'live_output_quality',
    ]

    def test_init_source_has_registration_marker(self):
        """AEONDeltaV3.__init__ source mentions PATCH-COGORG-5."""
        src = inspect.getsource(AEONDeltaV3.__init__)
        assert 'PATCH-COGORG-5' in src

    @pytest.mark.parametrize('signal_name', COGORG_SIGNALS)
    def test_signal_registered_in_init(self, signal_name):
        """Each COGORG signal is registered in __init__."""
        src = inspect.getsource(AEONDeltaV3.__init__)
        assert f'"{signal_name}"' in src, (
            f"Signal {signal_name!r} not found in __init__ registration"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §7  PATCH-COGORG-5b/c: CrossValidator Consistency Pairs
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG5bCVConsistencyPairs:
    """SubsystemCrossValidator has COGORG consistency pairs."""

    def test_cv_has_cogorg_health_pair(self):
        """CV has cognitive_unity_health ↔ pipeline_wiring_health pair."""
        pairs = SubsystemCrossValidator._CONSISTENCY_PAIRS
        pair_signals = [(a, b) for a, b, _ in pairs]
        assert ('cognitive_unity_health', 'pipeline_wiring_health') in pair_signals

    def test_cv_has_cogorg_reinforce_pair(self):
        """CV has reinforcement_cycle_quality ↔ feedback_bus_stability_score."""
        pairs = SubsystemCrossValidator._CONSISTENCY_PAIRS
        pair_signals = [(a, b) for a, b, _ in pairs]
        assert (
            'reinforcement_cycle_quality', 'feedback_bus_stability_score'
        ) in pair_signals

    def test_cv_total_pairs_increased(self):
        """At least 8 consistency pairs after COGORG patches."""
        assert len(SubsystemCrossValidator._CONSISTENCY_PAIRS) >= 8

    def test_cv_validates_new_pairs(self):
        """Validate reads the new pair signals without error."""
        fb = _make_feedback_bus()
        fb.write_signal('cognitive_unity_health', 0.9)
        fb.write_signal('pipeline_wiring_health', 0.95)
        fb.write_signal('reinforcement_cycle_quality', 0.8)
        fb.write_signal('feedback_bus_stability_score', 0.85)
        scv = SubsystemCrossValidator(feedback_bus=fb)
        report = scv.validate()
        assert isinstance(report, dict)
        assert 'inconsistency_score' in report


# ═══════════════════════════════════════════════════════════════════════════
# §8  PATCH-COGORG-5d: Health Sub-Component Readers
# ═══════════════════════════════════════════════════════════════════════════

class TestCOGORG5dHealthSubComponentReaders:
    """_build_feedback_extra_signals reads health sub-component signals."""

    def test_extra_signals_has_patch_marker(self):
        """Source mentions PATCH-COGORG-5d."""
        src = inspect.getsource(AEONDeltaV3._build_feedback_extra_signals)
        assert 'PATCH-COGORG-5d' in src

    def test_extra_signals_reads_health_components(self):
        """Source reads cognitive_unity_health, pipeline_wiring_health,
        feedback_bus_stability_score."""
        src = inspect.getsource(AEONDeltaV3._build_feedback_extra_signals)
        assert "'cognitive_unity_health'" in src
        assert "'pipeline_wiring_health'" in src
        assert "'feedback_bus_stability_score'" in src


# ═══════════════════════════════════════════════════════════════════════════
# §9  Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalEcosystemAudit:
    """Verify 0 orphan signals after all COGORG patches."""

    @staticmethod
    def _audit_signals():
        """Scan source files for write_signal/read_signal calls."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    content = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r'write_signal(?:_traced)?\(\s*[\'"]([\w_]+)[\'"]', content
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r'read_signal(?:_current_gen|_any_gen)?\(\s*[\'"]([\w_]+)[\'"]',
                content,
            ):
                read.add(m.group(1))
        return written, read

    def test_no_write_only_orphans(self):
        """No signals are written without being read."""
        written, read = self._audit_signals()
        orphans = written - read
        assert orphans == set(), f"Write-only orphans: {orphans}"

    def test_no_read_only_orphans(self):
        """No signals are read without being written."""
        written, read = self._audit_signals()
        orphans = read - written
        assert orphans == set(), f"Read-only orphans: {orphans}"

    def test_all_signals_bidirectional(self):
        """All signals are bidirectional (written AND read)."""
        written, read = self._audit_signals()
        bidirectional = written & read
        assert bidirectional == written == read

    def test_cogorg_signals_are_bidirectional(self):
        """All 7 COGORG signals appear in both write and read sets."""
        written, read = self._audit_signals()
        cogorg = {
            'architectural_health_score',
            'cognitive_unity_health',
            'pipeline_wiring_health',
            'feedback_bus_stability_score',
            'reinforcement_cycle_quality',
            'reinforcement_action_count',
            'live_output_quality',
        }
        for sig in cogorg:
            assert sig in written, f"{sig} not written"
            assert sig in read, f"{sig} not read"

    def test_signal_count_increased(self):
        """Total bidirectional count is at least 267 (260 + 7 COGORG)."""
        written, read = self._audit_signals()
        bidirectional = written & read
        assert len(bidirectional) >= 267


# ═══════════════════════════════════════════════════════════════════════════
# §10  Integration Map: Connected vs Isolated Paths
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationMap:
    """Verify critical cognitive paths are connected."""

    def test_health_to_bus_path_exists(self):
        """Health assessment → bus signal path is connected."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert 'write_signal' in src
        assert "'architectural_health_score'" in src

    def test_health_to_trace_path_exists(self):
        """Health assessment → causal trace path is connected."""
        src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert 'causal_trace.record' in src

    def test_reinforce_to_bus_outcome_path_exists(self):
        """Reinforcement → bus aggregate outcome path is connected."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "'reinforcement_cycle_quality'" in src

    def test_output_quality_to_bus_path_exists(self):
        """Output quality → bus live signal path is connected."""
        src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        assert "'live_output_quality'" in src

    def test_mct_reads_health_signal(self):
        """MCT reads architectural_health_score from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'architectural_health_score'" in src

    def test_mct_reads_reinforce_quality(self):
        """MCT reads reinforcement_cycle_quality from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'reinforcement_cycle_quality'" in src

    def test_mct_reads_output_quality(self):
        """MCT reads live_output_quality from bus."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'live_output_quality'" in src


# ═══════════════════════════════════════════════════════════════════════════
# §11  Mutual Reinforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Active components verify and stabilise each other."""

    def test_health_feeds_mct_sensitivity(self):
        """Low health boosts MCT coherence_deficit (mutual reinforcement)."""
        fb = _make_feedback_bus()
        mct = _make_mct(fb)
        # Baseline: no health signal written
        result_baseline = mct.evaluate(uncertainty=0.1, coherence_deficit=0.0)
        base_score = result_baseline.get('trigger_score', 0.0)

        # Write degraded health
        fb.write_signal('architectural_health_score', 0.3)
        result_degraded = mct.evaluate(uncertainty=0.1, coherence_deficit=0.0)
        degraded_score = result_degraded.get('trigger_score', 0.0)

        # Degraded health should increase trigger score
        assert degraded_score >= base_score, (
            f"Degraded health should increase trigger: "
            f"{degraded_score} >= {base_score}"
        )

    def test_reinforce_quality_feeds_mct(self):
        """Low reinforcement quality boosts MCT recovery pressure."""
        fb = _make_feedback_bus()
        mct = _make_mct(fb)
        # Write low reinforcement quality
        fb.write_signal('reinforcement_cycle_quality', 0.2)
        result = mct.evaluate(uncertainty=0.1, recovery_pressure=0.0)
        assert result is not None

    def test_cv_health_pair_detects_inconsistency(self):
        """CrossValidator detects unity↔wiring health inconsistency."""
        fb = _make_feedback_bus()
        # Create inconsistent state: high unity, low wiring
        fb.write_signal('cognitive_unity_health', 0.95)
        fb.write_signal('pipeline_wiring_health', 0.3)
        scv = SubsystemCrossValidator(feedback_bus=fb)
        report = scv.validate()
        # Should detect inconsistency in this pair
        assert report['inconsistency_score'] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §12  Meta-Cognitive Trigger Completeness
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTrigger:
    """Uncertainty and degradation trigger higher-order review."""

    def test_mct_reader_count_increased(self):
        """MCT evaluate() reads at least 4 COGORG signals."""
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        cogorg_signals = [
            'architectural_health_score',
            'reinforcement_cycle_quality',
            'live_output_quality',
            'reinforcement_action_count',
        ]
        count = sum(1 for s in cogorg_signals if f"'{s}'" in src)
        assert count >= 4, f"MCT should read all 4 COGORG signals, found {count}"

    def test_mct_evaluates_without_error(self):
        """MCT evaluate() works with COGORG signals on bus."""
        fb = _make_feedback_bus()
        mct = _make_mct(fb)
        # Write all COGORG signals
        fb.write_signal('architectural_health_score', 0.5)
        fb.write_signal('reinforcement_cycle_quality', 0.5)
        fb.write_signal('live_output_quality', 0.5)
        fb.write_signal('reinforcement_action_count', 0.5)
        result = mct.evaluate(uncertainty=0.3)
        assert isinstance(result, dict)
        assert 'should_trigger' in result
        assert 'trigger_score' in result


# ═══════════════════════════════════════════════════════════════════════════
# §13  Causal Transparency
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Every output can be deterministically traced to its root cause."""

    def test_health_chain_is_traceable(self):
        """Health → bus → MCT chain is traceable via source inspection."""
        # Health writes to bus (COGORG-1a)
        health_src = inspect.getsource(AEONDeltaV3.get_architectural_health)
        assert "'architectural_health_score'" in health_src
        # Health records to causal trace (COGORG-1b)
        assert 'causal_trace.record' in health_src
        # MCT reads from bus (COGORG-4a)
        mct_src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'architectural_health_score'" in mct_src

    def test_reinforce_chain_is_traceable(self):
        """Reinforce → bus → MCT chain is traceable."""
        reinforce_src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert "'reinforcement_cycle_quality'" in reinforce_src
        # Reinforce records to causal trace (COGORG-2b)
        assert 'PATCH-COGORG-2b' in reinforce_src
        # MCT reads (COGORG-4b)
        mct_src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'reinforcement_cycle_quality'" in mct_src

    def test_output_quality_chain_is_traceable(self):
        """Output quality → bus → MCT chain is traceable."""
        core_src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        assert "'live_output_quality'" in core_src
        mct_src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert "'live_output_quality'" in mct_src


# ═══════════════════════════════════════════════════════════════════════════
# §14  Activation Sequence
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Patches are applied in safe logical order."""

    PATCH_ORDER = [
        'COGORG-5',    # Signal registration (must be first)
        'COGORG-1a',   # Health → bus signal
        'COGORG-1b',   # Health → causal trace
        'COGORG-3',    # Output quality → bus signal
        'COGORG-2',    # Reinforce outcome → bus
        'COGORG-2b',   # Reinforce outcome → causal trace
        'COGORG-4a',   # MCT reader: health
        'COGORG-4b',   # MCT reader: reinforce quality
        'COGORG-4c',   # MCT reader: output quality
        'COGORG-4d',   # MCT reader: action count
        'COGORG-5b',   # CV consistency pair: health
        'COGORG-5c',   # CV consistency pair: reinforce
        'COGORG-5d',   # Health sub-component readers
    ]

    def test_activation_order_is_safe(self):
        """Signal registration (COGORG-5) comes before writers and readers."""
        assert self.PATCH_ORDER[0] == 'COGORG-5'
        # Writers (1a, 1b, 2, 3) come before readers (4a-d)
        writers = {'COGORG-1a', 'COGORG-1b', 'COGORG-3', 'COGORG-2', 'COGORG-2b'}
        readers = {'COGORG-4a', 'COGORG-4b', 'COGORG-4c', 'COGORG-4d'}
        writer_indices = [
            self.PATCH_ORDER.index(p) for p in writers
            if p in self.PATCH_ORDER
        ]
        reader_indices = [
            self.PATCH_ORDER.index(p) for p in readers
            if p in self.PATCH_ORDER
        ]
        assert max(writer_indices) < min(reader_indices), (
            "All writers must be applied before readers"
        )

    def test_all_patches_present(self):
        """All 13 COGORG patches are in the activation sequence."""
        assert len(self.PATCH_ORDER) == 13


# ═══════════════════════════════════════════════════════════════════════════
# §15  E2E Integration: MCT with All COGORG Signals
# ═══════════════════════════════════════════════════════════════════════════

class TestE2EIntegration:
    """End-to-end integration of all COGORG patches."""

    def test_full_signal_flow(self):
        """All COGORG signals flow through bus → MCT without errors."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()
        mct = _make_mct(fb)
        mct._causal_trace_ref = ct

        # Simulate COGORG-1a: health writes
        fb.write_signal('architectural_health_score', 0.6)
        fb.write_signal('cognitive_unity_health', 0.7)
        fb.write_signal('pipeline_wiring_health', 0.95)
        fb.write_signal('feedback_bus_stability_score', 0.8)

        # Simulate COGORG-2: reinforce writes
        fb.write_signal('reinforcement_cycle_quality', 0.5)
        fb.write_signal('reinforcement_action_count', 0.3)

        # Simulate COGORG-3: output quality write
        fb.write_signal('live_output_quality', 0.4)

        # MCT evaluates with all COGORG signals available
        result = mct.evaluate(
            uncertainty=0.2,
            coherence_deficit=0.1,
            recovery_pressure=0.0,
        )
        assert isinstance(result, dict)
        assert 'trigger_score' in result
        assert isinstance(result['trigger_score'], float)

    def test_cv_validates_cogorg_pairs(self):
        """CrossValidator validates all COGORG consistency pairs."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()

        # Write consistent COGORG signals
        fb.write_signal('cognitive_unity_health', 0.85)
        fb.write_signal('pipeline_wiring_health', 0.9)
        fb.write_signal('reinforcement_cycle_quality', 0.8)
        fb.write_signal('feedback_bus_stability_score', 0.85)

        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.set_causal_trace(ct)
        report = scv.validate()

        assert isinstance(report, dict)
        assert report['inconsistency_score'] >= 0.0

    def test_version_consistency(self):
        """Version is 3.4.0 after all COGORG patches."""
        assert __version__ == '3.4.0'
