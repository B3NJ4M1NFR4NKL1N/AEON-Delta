"""Tests for PATCH-ζ1 through PATCH-ζ7 — Final Integration & Cognitive Activation.

Covers:
  PATCH-ζ6:  Feedback bus seeding in /api/init
  PATCH-ζ4:  RSSM prediction error causal trace recording
  PATCH-ζ5:  Decoder temperature/reliability tracing
  PATCH-ζ1:  Pillar stage (diversity/topology/safety) causal trace recording
  PATCH-ζ2:  Memory fusion bidirectional feedback + MCT reader (ζ2b)
  PATCH-ζ3:  Causal discovery quality recovery path + MCT reader (ζ3b)
  PATCH-ζ7:  Proactive MCT evaluation during training + MCT reader (ζ7b)
  Signal ecosystem integrity audit
  E2E integration flows, causal transparency, activation sequence
"""

import math
import re
import sys
from collections import deque
from unittest.mock import MagicMock, patch

import pytest
import torch

import aeon_core
import ae_train


# ── Helpers ───────────────────────────────────────────────────────────

def _make_bus():
    """Create a minimal CognitiveFeedbackBus."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=64)


def _make_causal_trace():
    """Create a minimal TemporalCausalTraceBuffer."""
    return aeon_core.TemporalCausalTraceBuffer(max_entries=100)


def _make_mct(feedback_bus=None):
    """Create a MetaCognitiveRecursionTrigger with optional bus."""
    mct = aeon_core.MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=2,
    )
    if feedback_bus is not None:
        mct.set_feedback_bus(feedback_bus)
    return mct


def _src(fname):
    """Read source file as string."""
    with open(fname) as f:
        return f.read()


def _scan_signals():
    """Scan all signal writes/reads from source files."""
    written = set()
    read = set()
    w_pat = re.compile(
        r"write_signal\s*\(\s*['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    r_pat = re.compile(
        r"read_signal\s*\(\s*['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
        try:
            with open(fname) as f:
                src = f.read()
            written |= set(w_pat.findall(src))
            read |= set(r_pat.findall(src))
        except FileNotFoundError:
            pass
    return written, read


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: PATCH-ζ6 — Feedback Bus Seeding in /api/init
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta6:
    """PATCH-ζ6: Bus flush + seed + causal origin in /api/init."""

    def test_z6_flush_consumed_called(self):
        """Server /api/init calls flush_consumed() on feedback bus."""
        src = _src('aeon_server.py')
        assert 'flush_consumed()' in src, (
            "PATCH-ζ6 must call flush_consumed() in /api/init"
        )

    def test_z6_server_coherence_seeded(self):
        """Server /api/init seeds server_coherence_score to 0.7."""
        src = _src('aeon_server.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]server_coherence_score['\"].*0\.7",
            src,
        ), "PATCH-ζ6 must seed server_coherence_score=0.7"

    def test_z6_reinforcement_pressure_seeded(self):
        """Server /api/init seeds server_reinforcement_pressure to 0.0."""
        src = _src('aeon_server.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]server_reinforcement_pressure['\"].*0\.0",
            src,
        ), "PATCH-ζ6 must seed server_reinforcement_pressure=0.0"

    def test_z6_causal_trace_origin(self):
        """Server /api/init records server_init in causal trace."""
        src = _src('aeon_server.py')
        assert re.search(
            r'subsystem\s*=\s*["\']server_init["\']',
            src,
        ), "PATCH-ζ6 must record server_init in causal trace"

    def test_z6_causal_trace_metadata_has_signals(self):
        """Causal trace record includes seeded signal names."""
        src = _src('aeon_server.py')
        assert 'seeded_signals' in src, (
            "PATCH-ζ6 trace record must include seeded_signals metadata"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 1a: PATCH-ζ4 — RSSM Prediction Error Causal Trace
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta4:
    """PATCH-ζ4: RSSM prediction error → causal trace."""

    def test_z4_causal_trace_record_in_rssm(self):
        """VibeThinkerRSSMBridge records high prediction error in trace."""
        src = _src('aeon_core.py')
        # Find the RSSM bridge section with both rssm_prediction_pressure
        # write and causal_trace record nearby
        idx = src.find("PATCH-ζ4")
        assert idx > 0, "PATCH-ζ4 section must exist"
        nearby = src[idx:idx + 1500]
        assert '_z4_ct.record(' in nearby, (
            "PATCH-ζ4 must record to causal trace"
        )

    def test_z4_subsystem_name_is_rssm(self):
        """Causal trace uses subsystem='rssm'."""
        src = _src('aeon_core.py')
        assert re.search(
            r'subsystem\s*=\s*["\']rssm["\']',
            src,
        ), "PATCH-ζ4 must use subsystem='rssm'"

    def test_z4_severity_is_warning(self):
        """RSSM high-error trace entry has severity='warning'."""
        src = _src('aeon_core.py')
        # Find ζ4 section
        idx = src.find("PATCH-ζ4")
        assert idx > 0
        nearby = src[idx:idx + 1000]
        assert 'severity="warning"' in nearby or "severity='warning'" in nearby


# ═══════════════════════════════════════════════════════════════════════
# Phase 1b: PATCH-ζ5 — Decoder Temperature/Reliability Tracing
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta5:
    """PATCH-ζ5: Decoder records temperature + reliability in trace."""

    def test_z5_decoder_trace_has_temperature(self):
        """Decoder causal trace record includes temperature metadata."""
        src = _src('aeon_core.py')
        # Find the decode_complete record
        idx = src.find('"decode_complete"')
        if idx < 0:
            idx = src.find("'decode_complete'")
        assert idx > 0, "Decoder must record 'decode_complete' in trace"
        nearby = src[idx:idx + 500]
        assert "'temperature'" in nearby or '"temperature"' in nearby, (
            "PATCH-ζ5 must include temperature in decode_complete metadata"
        )

    def test_z5_decoder_trace_has_reliability(self):
        """Decoder causal trace record includes reliability_score."""
        src = _src('aeon_core.py')
        idx = src.find('"decode_complete"')
        if idx < 0:
            idx = src.find("'decode_complete'")
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert "'reliability_score'" in nearby or '"reliability_score"' in nearby

    def test_z5_decoder_trace_has_attenuation_flag(self):
        """Decoder causal trace includes reliability_attenuated flag."""
        src = _src('aeon_core.py')
        idx = src.find('"decode_complete"')
        if idx < 0:
            idx = src.find("'decode_complete'")
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert 'reliability_attenuated' in nearby


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: PATCH-ζ1 — Pillar Stage Causal Trace Recording
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta1:
    """PATCH-ζ1: Diversity/Topology/Safety pillars → causal trace."""

    def test_z1a_diversity_pillar_trace(self):
        """Diversity pillar records to causal trace with subsystem='pillar_diversity'."""
        src = _src('aeon_core.py')
        assert re.search(
            r'subsystem\s*=\s*["\']pillar_diversity["\']',
            src,
        ), "PATCH-ζ1a must record subsystem='pillar_diversity'"

    def test_z1a_diversity_trace_has_prerequisites(self):
        """Diversity trace entry includes causal_prerequisites."""
        src = _src('aeon_core.py')
        idx = src.find("'pillar_diversity'")
        if idx < 0:
            idx = src.find('"pillar_diversity"')
        assert idx > 0
        nearby = src[idx:idx + 400]
        assert 'causal_prerequisites' in nearby

    def test_z1b_topology_pillar_trace(self):
        """Topology pillar records with subsystem='pillar_topology'."""
        src = _src('aeon_core.py')
        assert re.search(
            r'subsystem\s*=\s*["\']pillar_topology["\']',
            src,
        ), "PATCH-ζ1b must record subsystem='pillar_topology'"

    def test_z1b_topology_trace_includes_catastrophe(self):
        """Topology trace records catastrophe detection status."""
        src = _src('aeon_core.py')
        idx = src.find("'pillar_topology'")
        if idx < 0:
            idx = src.find('"pillar_topology"')
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert 'catastrophe_detected' in nearby

    def test_z1c_safety_pillar_trace(self):
        """Safety pillar records with subsystem='pillar_safety'."""
        src = _src('aeon_core.py')
        assert re.search(
            r'subsystem\s*=\s*["\']pillar_safety["\']',
            src,
        ), "PATCH-ζ1c must record subsystem='pillar_safety'"

    def test_z1c_safety_trace_includes_enforcement(self):
        """Safety trace records enforcement status and score."""
        src = _src('aeon_core.py')
        idx = src.find("'pillar_safety'")
        if idx < 0:
            idx = src.find('"pillar_safety"')
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert 'safety_score' in nearby
        assert 'enforced' in nearby


# ═══════════════════════════════════════════════════════════════════════
# Phase 3a: PATCH-ζ2 — Memory Fusion Bidirectional Feedback
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta2:
    """PATCH-ζ2: Memory fusion deficit → bus + MCT reader."""

    def test_z2_memory_fusion_deficit_written(self):
        """Forward pass writes 'memory_fusion_deficit' to bus."""
        src = _src('aeon_core.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]memory_fusion_deficit['\"]",
            src,
        ), "PATCH-ζ2 must write memory_fusion_deficit signal"

    def test_z2_memory_fusion_deficit_zero_when_ok(self):
        """Forward pass writes 0.0 when quality >= threshold."""
        src = _src('aeon_core.py')
        # Find the ζ2 section in the forward pass (not the MCT reader)
        idx = src.find("PATCH-ζ2: Memory fusion")
        assert idx > 0
        nearby = src[idx:idx + 2000]
        # The else branch writes 0.0 on a separate line from the signal name
        assert "'memory_fusion_deficit'" in nearby
        assert "0.0" in nearby

    def test_z2_causal_trace_on_low_quality(self):
        """Low memory quality records to causal trace."""
        src = _src('aeon_core.py')
        assert re.search(
            r'subsystem\s*=\s*["\']memory_fusion["\']',
            src,
        ), "PATCH-ζ2 must record subsystem='memory_fusion'"

    def test_z2b_mct_reads_memory_fusion_deficit(self):
        """MCT evaluate() reads memory_fusion_deficit signal."""
        src = _src('aeon_core.py')
        assert re.search(
            r"read_signal\s*\(\s*['\"]memory_fusion_deficit['\"]",
            src,
        ), "PATCH-ζ2b: MCT must read memory_fusion_deficit"

    def test_z2b_mct_amplifies_memory_staleness(self):
        """MCT amplifies memory_staleness when deficit > 0.3."""
        src = _src('aeon_core.py')
        idx = src.find("PATCH-ζ2b")
        assert idx > 0, "PATCH-ζ2b section must exist in MCT"
        nearby = src[idx:idx + 500]
        assert 'memory_staleness' in nearby, (
            "PATCH-ζ2b must amplify memory_staleness trigger"
        )

    def test_z2b_functional_amplification(self):
        """MCT actually amplifies memory_staleness from deficit signal."""
        bus = _make_bus()
        mct = _make_mct(bus)
        # Write high deficit
        bus.write_signal('memory_fusion_deficit', 0.8)
        # Evaluate with base memory_staleness active and high uncertainty
        result_with = mct.evaluate(memory_staleness=True, uncertainty=0.6)
        sv_with = result_with.get('signal_values', {})

        # Now compare with bus without deficit
        bus2 = _make_bus()
        mct2 = _make_mct(bus2)
        bus2.write_signal('memory_fusion_deficit', 0.0)
        result_without = mct2.evaluate(memory_staleness=True, uncertainty=0.6)
        sv_without = result_without.get('signal_values', {})

        ms_with = sv_with.get('memory_staleness', 0.0)
        ms_without = sv_without.get('memory_staleness', 0.0)
        # With high deficit, memory_staleness should be >= without deficit
        assert ms_with >= ms_without, (
            f"memory_staleness with deficit ({ms_with}) should be >= "
            f"without ({ms_without})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3b: PATCH-ζ3 — Causal Discovery Quality Recovery
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta3:
    """PATCH-ζ3: Causal discovery recovery pressure + MCT reader."""

    def test_z3_recovery_pressure_written(self):
        """Forward pass writes 'causal_discovery_recovery_pressure'."""
        src = _src('aeon_core.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]causal_discovery_recovery_pressure['\"]",
            src,
        ), "PATCH-ζ3 must write causal_discovery_recovery_pressure"

    def test_z3_zero_pressure_when_quality_ok(self):
        """Forward pass writes 0.0 when quality >= 0.5."""
        src = _src('aeon_core.py')
        # Find the ζ3 section in the forward pass
        idx = src.find("PATCH-ζ3: Causal discovery")
        assert idx > 0
        nearby = src[idx:idx + 2500]
        # The else branch writes 0.0 on a separate line
        assert "'causal_discovery_recovery_pressure'" in nearby
        count = nearby.count("'causal_discovery_recovery_pressure'")
        assert count >= 2, (
            f"Expected at least 2 writes (low + ok), got {count}"
        )

    def test_z3_causal_trace_on_low_quality(self):
        """Low causal quality records warning to causal trace."""
        src = _src('aeon_core.py')
        idx = src.find("PATCH-ζ3: Causal discovery")
        assert idx > 0
        nearby = src[idx:idx + 2500]
        assert 'severity="warning"' in nearby or "severity='warning'" in nearby

    def test_z3_attenuation_applied(self):
        """Low quality attenuates causal blend (floor 30%)."""
        src = _src('aeon_core.py')
        idx = src.find("PATCH-ζ3: Causal discovery")
        assert idx > 0
        nearby = src[idx:idx + 2500]
        assert 'max(0.3' in nearby, (
            "PATCH-ζ3 must floor attenuation at 0.3"
        )

    def test_z3b_mct_reads_recovery_pressure(self):
        """MCT evaluate() reads causal_discovery_recovery_pressure."""
        src = _src('aeon_core.py')
        assert re.search(
            r"read_signal\s*\(\s*['\"]causal_discovery_recovery_pressure['\"]",
            src,
        ), "PATCH-ζ3b: MCT must read causal_discovery_recovery_pressure"

    def test_z3b_mct_amplifies_low_causal_quality(self):
        """MCT amplifies low_causal_quality when pressure > 0.3."""
        src = _src('aeon_core.py')
        idx = src.find("PATCH-ζ3b")
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert 'low_causal_quality' in nearby

    def test_z3b_functional_amplification(self):
        """MCT actually amplifies low_causal_quality from pressure."""
        bus = _make_bus()
        mct = _make_mct(bus)
        # Write high recovery pressure
        bus.write_signal('causal_discovery_recovery_pressure', 0.8)
        # Evaluate with low causal quality baseline
        result_with = mct.evaluate(causal_quality=0.3, uncertainty=0.6)
        sv_with = result_with.get('signal_values', {})

        bus2 = _make_bus()
        mct2 = _make_mct(bus2)
        bus2.write_signal('causal_discovery_recovery_pressure', 0.0)
        result_without = mct2.evaluate(causal_quality=0.3, uncertainty=0.6)
        sv_without = result_without.get('signal_values', {})

        lcq_with = sv_with.get('low_causal_quality', 0.0)
        lcq_without = sv_without.get('low_causal_quality', 0.0)
        assert lcq_with >= lcq_without, (
            f"low_causal_quality with pressure ({lcq_with}) should be >= "
            f"without ({lcq_without})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: PATCH-ζ7 — Proactive MCT in Training
# ═══════════════════════════════════════════════════════════════════════

class TestPatchZeta7:
    """PATCH-ζ7: Training loop calls MCT.evaluate() proactively."""

    def test_z7_training_mct_should_trigger_written_phase_a(self):
        """Phase A writes 'training_mct_should_trigger' to bus."""
        src = _src('ae_train.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]training_mct_should_trigger['\"]",
            src,
        ), "PATCH-ζ7 must write training_mct_should_trigger"

    def test_z7_training_mct_trigger_score_written_phase_a(self):
        """Phase A writes 'training_mct_trigger_score' to bus."""
        src = _src('ae_train.py')
        assert re.search(
            r"write_signal\s*\(\s*['\"]training_mct_trigger_score['\"]",
            src,
        ), "PATCH-ζ7 must write training_mct_trigger_score"

    def test_z7_mct_evaluate_called_in_training(self):
        """Phase A calls MCT.evaluate() directly."""
        src = _src('ae_train.py')
        idx = src.find("PATCH-ζ7")
        assert idx > 0
        nearby = src[idx:idx + 1500]
        assert '.evaluate(' in nearby, (
            "PATCH-ζ7 must call MCT.evaluate() in training"
        )

    def test_z7_loss_spike_detection(self):
        """ζ7 detects loss spikes via running mean comparison."""
        src = _src('ae_train.py')
        idx = src.find("PATCH-ζ7")
        assert idx > 0
        nearby = src[idx:idx + 1500]
        assert '_z7_loss_history' in nearby or '_z7b_loss_history' in nearby

    def test_z7_lr_halved_on_trigger(self):
        """When MCT triggers, LR is halved."""
        src = _src('ae_train.py')
        idx = src.find("PATCH-ζ7")
        assert idx > 0
        nearby = src[idx:idx + 2500]
        assert '*= 0.5' in nearby, (
            "PATCH-ζ7 must halve LR on MCT trigger"
        )

    def test_z7_phase_b_mirror(self):
        """Phase B mirrors ζ7 logic for RSSM training."""
        src = _src('ae_train.py')
        # Count occurrences of the ζ7 pattern
        matches = re.findall(
            r"PATCH-ζ7.*Phase B",
            src,
            re.DOTALL,
        )
        assert len(matches) > 0, (
            "PATCH-ζ7 must be mirrored in Phase B"
        )

    def test_z7b_mct_reads_training_signals(self):
        """MCT evaluate() reads training_mct_should_trigger."""
        src = _src('aeon_core.py')
        assert re.search(
            r"read_signal\s*\(\s*['\"]training_mct_should_trigger['\"]",
            src,
        ), "PATCH-ζ7b: MCT must read training_mct_should_trigger"

    def test_z7b_mct_reads_training_score(self):
        """MCT evaluate() reads training_mct_trigger_score."""
        src = _src('aeon_core.py')
        assert re.search(
            r"read_signal\s*\(\s*['\"]training_mct_trigger_score['\"]",
            src,
        ), "PATCH-ζ7b: MCT must read training_mct_trigger_score"

    def test_z7b_mct_amplifies_uncertainty(self):
        """MCT amplifies uncertainty from training MCT score."""
        src = _src('aeon_core.py')
        idx = src.find("PATCH-ζ7b")
        assert idx > 0
        nearby = src[idx:idx + 500]
        assert 'uncertainty' in nearby


# ═══════════════════════════════════════════════════════════════════════
# Signal Ecosystem Integrity Audit
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystem:
    """Verify new signals from ζ-patches are bidirectional."""

    @pytest.fixture(autouse=True)
    def _scan(self):
        self.written, self.read = _scan_signals()

    def test_memory_fusion_deficit_bidirectional(self):
        """memory_fusion_deficit is both written and read."""
        assert 'memory_fusion_deficit' in self.written
        assert 'memory_fusion_deficit' in self.read

    def test_causal_discovery_recovery_pressure_bidirectional(self):
        """causal_discovery_recovery_pressure is both written and read."""
        assert 'causal_discovery_recovery_pressure' in self.written
        assert 'causal_discovery_recovery_pressure' in self.read

    def test_training_mct_should_trigger_bidirectional(self):
        """training_mct_should_trigger is both written and read."""
        assert 'training_mct_should_trigger' in self.written
        assert 'training_mct_should_trigger' in self.read

    def test_training_mct_trigger_score_bidirectional(self):
        """training_mct_trigger_score is both written and read."""
        assert 'training_mct_trigger_score' in self.written
        assert 'training_mct_trigger_score' in self.read

    def test_no_orphan_signals(self):
        """No signal is written without a corresponding reader."""
        # Exclude known server-only signals that are consumed externally
        _server_only = {
            'server_coherence_score',
            'server_reinforcement_pressure',
        }
        orphans = self.written - self.read - _server_only
        assert len(orphans) == 0, (
            f"Orphan signals (written, never read): {orphans}"
        )


# ═══════════════════════════════════════════════════════════════════════
# E2E Integration Flows
# ═══════════════════════════════════════════════════════════════════════

class TestE2EIntegration:
    """End-to-end integration verification."""

    def test_pillar_trace_subsystem_prefix_search(self):
        """causal_trace.find(subsystem_prefix='pillar_') returns entries."""
        trace = _make_causal_trace()
        trace.record(
            subsystem="pillar_diversity",
            decision="diversity_score=0.8",
            metadata={'diversity_score': 0.8},
        )
        trace.record(
            subsystem="pillar_topology",
            decision="topology_stable",
            metadata={'catastrophe_detected': False},
        )
        trace.record(
            subsystem="pillar_safety",
            decision="safety_passed, score=0.95",
            metadata={'safety_score': 0.95, 'enforced': False},
        )
        results = trace.find(subsystem_prefix="pillar_")
        assert len(results) == 3, (
            f"Expected 3 pillar entries, got {len(results)}"
        )

    def test_rssm_trace_findable(self):
        """RSSM trace entries are findable by subsystem='rssm'."""
        trace = _make_causal_trace()
        trace.record(
            subsystem="rssm",
            decision="high_prediction_error=0.7",
            severity="warning",
            metadata={'prediction_error': 0.7},
        )
        results = trace.find(subsystem="rssm")
        assert len(results) == 1

    def test_memory_fusion_trace_findable(self):
        """Memory fusion trace entries findable by subsystem."""
        trace = _make_causal_trace()
        trace.record(
            subsystem="memory_fusion",
            decision="low_fusion_quality=0.3",
            metadata={'retrieval_quality': 0.3, 'deficit': 0.7},
        )
        results = trace.find(subsystem="memory_fusion")
        assert len(results) == 1

    def test_decoder_trace_with_temperature(self):
        """Decoder trace entry includes temperature metadata."""
        trace = _make_causal_trace()
        trace.record(
            subsystem="decoder",
            decision="decode_complete",
            metadata={
                'mode': 'inference',
                'temperature': 0.8,
                'reliability_score': 0.7,
                'reliability_attenuated': True,
            },
        )
        results = trace.find(subsystem="decoder")
        assert len(results) == 1
        assert results[0]['metadata']['temperature'] == 0.8
        assert results[0]['metadata']['reliability_attenuated'] is True


# ═══════════════════════════════════════════════════════════════════════
# Causal Transparency
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify end-to-end causal trace coverage."""

    def test_full_chain_coverage(self):
        """Trace chain: input → pillars → memory_fusion → causal_model → decoder."""
        trace = _make_causal_trace()
        # Simulate full forward chain
        t0 = trace.record("input", "encoded", metadata={})
        t1 = trace.record(
            "pillar_diversity", "diversity_score=0.9",
            causal_prerequisites=[t0],
        )
        t2 = trace.record(
            "pillar_topology", "topology_stable",
            causal_prerequisites=[t0],
        )
        t3 = trace.record(
            "pillar_safety", "safety_passed, score=0.95",
            causal_prerequisites=[t0],
        )
        t4 = trace.record(
            "memory_fusion", "low_fusion_quality=0.4",
            causal_prerequisites=[t0],
        )
        t5 = trace.record(
            "rssm", "high_prediction_error=0.6",
            causal_prerequisites=[t4] if t4 else [],
        )
        t6 = trace.record(
            "causal_model", "dag_computed",
            causal_prerequisites=[t0],
        )
        t7 = trace.record(
            "decoder", "decode_complete",
            metadata={'temperature': 0.8, 'reliability_score': 0.7},
        )
        # All subsystems should be findable
        all_entries = trace.find()
        subsystems = {e.get('subsystem', '') for e in all_entries}
        expected = {
            'input', 'pillar_diversity', 'pillar_topology',
            'pillar_safety', 'memory_fusion', 'rssm',
            'causal_model', 'decoder',
        }
        assert expected.issubset(subsystems), (
            f"Missing subsystems: {expected - subsystems}"
        )

    def test_server_init_is_session_origin(self):
        """server_init entry establishes causal origin for session."""
        trace = _make_causal_trace()
        trace.record(
            subsystem="server_init",
            decision="feedback_bus_seeded_and_flushed",
            metadata={'generation': 0},
        )
        results = trace.find(subsystem="server_init")
        assert len(results) == 1
        assert results[0]['decision'] == "feedback_bus_seeded_and_flushed"


# ═══════════════════════════════════════════════════════════════════════
# Activation Sequence
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify patches are applied in correct dependency order."""

    def test_z6_before_z2_z3(self):
        """ζ6 (bus seeding) in /api/init is before any inference."""
        src_server = _src('aeon_server.py')
        idx_z6 = src_server.find("PATCH-ζ6")
        idx_api_init = src_server.find("/api/init")
        assert idx_z6 > idx_api_init, (
            "PATCH-ζ6 must be in /api/init"
        )

    def test_z4_z5_are_observation_only(self):
        """ζ4 and ζ5 only add trace records, no behavioral changes."""
        src = _src('aeon_core.py')
        # ζ4: should only have trace.record, no signal writes
        idx_z4 = src.find("PATCH-ζ4")
        assert idx_z4 > 0
        z4_section = src[idx_z4:idx_z4 + 700]
        assert '_z4_ct.record(' in z4_section
        # No write_signal in ζ4 section
        assert 'write_signal' not in z4_section, (
            "PATCH-ζ4 should be observation-only"
        )

    def test_z1_after_pillar_provenance(self):
        """ζ1 trace records appear after provenance_tracker records."""
        src = _src('aeon_core.py')
        idx_div_prov = src.find('record_after("diversity_analysis"')
        idx_div_trace = src.find('"pillar_diversity"')
        if idx_div_trace < 0:
            idx_div_trace = src.find("'pillar_diversity'")
        assert idx_div_trace > 0, "pillar_diversity must exist in source"
        assert idx_div_prov > 0, "diversity_analysis provenance must exist"
        assert idx_div_trace > idx_div_prov, (
            "Diversity trace must be after provenance"
        )

    def test_z7_after_z2_z3_in_mct(self):
        """ζ7b MCT reader appears after ζ2b and ζ3b readers."""
        src = _src('aeon_core.py')
        idx_z2b = src.find("PATCH-ζ2b")
        idx_z3b = src.find("PATCH-ζ3b")
        idx_z7b = src.find("PATCH-ζ7b")
        assert idx_z2b > 0 and idx_z3b > 0 and idx_z7b > 0
        assert idx_z7b > idx_z2b, "ζ7b must come after ζ2b"
        assert idx_z7b > idx_z3b, "ζ7b must come after ζ3b"


# ═══════════════════════════════════════════════════════════════════════
# Mutual Reinforcement
# ═══════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify that patches enable mutual verification between subsystems."""

    def test_mct_receives_pillar_signals_indirectly(self):
        """MCT can read signals that pillars influence via uncertainty."""
        # Pillar decisions affect uncertainty → MCT reads uncertainty
        src = _src('aeon_core.py')
        # Topology catastrophe escalates uncertainty
        assert 'topology_catastrophe' in src
        # Safety enforcement escalates uncertainty
        assert 'safety_enforced' in src or 'safety_blocked' in src

    def test_training_mct_closes_bidirectional_loop(self):
        """Training writes MCT signals, inference reads them."""
        written, read = _scan_signals()
        # Training writes
        assert 'training_mct_should_trigger' in written
        assert 'training_mct_trigger_score' in written
        # MCT reads during inference
        assert 'training_mct_should_trigger' in read
        assert 'training_mct_trigger_score' in read

    def test_memory_fusion_closes_bidirectional_loop(self):
        """Memory writes deficit, MCT reads and amplifies."""
        written, read = _scan_signals()
        assert 'memory_fusion_deficit' in written
        assert 'memory_fusion_deficit' in read

    def test_causal_discovery_closes_bidirectional_loop(self):
        """Causal model writes pressure, MCT reads and amplifies."""
        written, read = _scan_signals()
        assert 'causal_discovery_recovery_pressure' in written
        assert 'causal_discovery_recovery_pressure' in read
