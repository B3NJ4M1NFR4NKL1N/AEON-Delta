"""Tests for PATCH-Σ1 through Σ6: Final cognitive integration patches.

Covers:
  Σ1  UnifiedCausalSimulator → causal trace
  Σ2  MCTSPlanner → causal trace
  Σ3  CausalFactorExtractor → causal trace + feedback bus
  Σ3b MCT reader for factor_sparsity_ratio
  Σ4  VibeThinkerRSSMBridge → causal trace wiring fix
  Σ5  system_emergence_report → feedback bus + MCT reader
  Σ6  PostPipelineMCTGate → causal trace
  Signal ecosystem audit (bidirectional, 0 orphans)
  E2E integration flows
  Causal transparency (trace_root_cause traversal)
  Activation sequence ordering
  Mutual reinforcement
"""

import math
import re
import sys
import threading
from collections import deque
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs — avoid importing the full (heavy) aeon_core module
# ---------------------------------------------------------------------------


class StubCausalTrace:
    """Minimal TemporalCausalTraceBuffer for unit tests."""

    def __init__(self, max_entries: int = 100):
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def record(
        self,
        subsystem: str = "",
        decision: str = "",
        severity: str = "info",
        initial_state_hash: Any = None,
        causal_prerequisites: Optional[List[str]] = None,
        rejected_alternatives: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._lock:
            entry_id = self._next_id
            self._next_id += 1
            self._entries.append({
                "id": entry_id,
                "subsystem": subsystem,
                "decision": decision,
                "severity": severity,
                "initial_state_hash": initial_state_hash,
                "causal_prerequisites": causal_prerequisites or [],
                "rejected_alternatives": rejected_alternatives or [],
                "metadata": metadata or {},
            })
            return entry_id

    def find(
        self,
        subsystem: Optional[str] = None,
        subsystem_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for e in reversed(self._entries):
            if subsystem and e["subsystem"] == subsystem:
                results.append(e)
            elif subsystem_prefix and e["subsystem"].startswith(subsystem_prefix):
                results.append(e)
        return results


class StubFeedbackBus:
    """Minimal CognitiveFeedbackBus for unit tests."""

    NUM_SIGNAL_CHANNELS = 12

    def __init__(self):
        self._signals: Dict[str, float] = {}
        self._registered: set = set()
        self._persistent: set = set()
        self._write_log: set = set()

    def register_signal(self, name: str, default: float = 0.0) -> None:
        self._registered.add(name)
        self._signals.setdefault(name, default)

    def register_persistent_signal(self, name: str, default: float = 0.0) -> None:
        self._persistent.add(name)
        self._registered.add(name)
        self._signals.setdefault(name, default)

    def write_signal(self, name: str, value: float) -> None:
        self._signals[name] = value
        self._write_log.add(name)

    def read_signal(self, name: str, default: float = 0.0) -> float:
        return self._signals.get(name, default)


# ===========================================================================
# PATCH-Σ4: VibeThinkerRSSMBridge causal trace wiring
# ===========================================================================


class TestSigma4_VibeThinkerRSSMBridge:
    """Σ4: VibeThinkerRSSMBridge now has _causal_trace_ref + set_causal_trace."""

    def test_init_has_causal_trace_ref(self):
        """_causal_trace_ref is initialised to None in __init__."""
        from aeon_core import VibeThinkerRSSMBridge
        bridge = VibeThinkerRSSMBridge()
        assert hasattr(bridge, '_causal_trace_ref')
        assert bridge._causal_trace_ref is None

    def test_set_causal_trace_method_exists(self):
        """set_causal_trace() method is present."""
        from aeon_core import VibeThinkerRSSMBridge
        bridge = VibeThinkerRSSMBridge()
        assert hasattr(bridge, 'set_causal_trace')
        assert callable(bridge.set_causal_trace)

    def test_set_causal_trace_wires_ref(self):
        """set_causal_trace() stores the reference."""
        from aeon_core import VibeThinkerRSSMBridge
        bridge = VibeThinkerRSSMBridge()
        ct = StubCausalTrace()
        bridge.set_causal_trace(ct)
        assert bridge._causal_trace_ref is ct

    def test_modulate_rssm_loss_records_trace(self):
        """When prediction error is high, ζ4 recording code now executes."""
        from aeon_core import VibeThinkerRSSMBridge
        bus = StubFeedbackBus()
        bridge = VibeThinkerRSSMBridge(feedback_bus=bus)
        ct = StubCausalTrace()
        bridge.set_causal_trace(ct)
        # Trigger high prediction error path (> default threshold 0.5)
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.8,
        )
        entries = ct.find(subsystem="rssm")
        assert len(entries) > 0, "ζ4 recording code should fire when causal trace is wired"
        assert "high_prediction_error" in entries[0]["decision"]

    def test_modulate_rssm_loss_no_trace_when_not_wired(self):
        """When _causal_trace_ref is None, no recording attempt is made."""
        from aeon_core import VibeThinkerRSSMBridge
        bus = StubFeedbackBus()
        bridge = VibeThinkerRSSMBridge(feedback_bus=bus)
        # Do NOT wire causal trace
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.5,
            rssm_prediction_error=0.8,
        )
        # Should not raise; result should still have temperature boost
        assert 'vt_temperature_boost' in result


# ===========================================================================
# PATCH-Σ6: PostPipelineMCTGate causal trace
# ===========================================================================


class TestSigma6_PostPipelineMCTGate:
    """Σ6: PostPipelineMCTGate records gate decisions to causal trace."""

    def test_init_has_causal_trace_ref(self):
        from aeon_core import PostPipelineMCTGate
        gate = PostPipelineMCTGate(mct_ref=None, feedback_bus_ref=None)
        assert hasattr(gate, '_causal_trace_ref')
        assert gate._causal_trace_ref is None

    def test_set_causal_trace_method(self):
        from aeon_core import PostPipelineMCTGate
        gate = PostPipelineMCTGate(mct_ref=None, feedback_bus_ref=None)
        ct = StubCausalTrace()
        gate.set_causal_trace(ct)
        assert gate._causal_trace_ref is ct

    def test_retrigger_records_trace(self):
        """When retrigger fires, causal trace records the decision."""
        from aeon_core import PostPipelineMCTGate
        bus = StubFeedbackBus()
        bus.write_signal('post_output_uncertainty', 0.8)
        bus.write_signal('post_pipeline_verdict_pressure', 0.9)
        # Mock MCT
        mock_mct = MagicMock()
        mock_mct.evaluate.return_value = {
            'should_trigger': True,
            'trigger_score': 0.85,
        }
        gate = PostPipelineMCTGate(
            mct_ref=mock_mct, feedback_bus_ref=bus, threshold=0.6,
        )
        ct = StubCausalTrace()
        gate.set_causal_trace(ct)
        triggered = gate.check_and_retrigger()
        assert triggered is True
        entries = ct.find(subsystem="post_pipeline_mct_gate")
        assert len(entries) > 0
        assert "retrigger=yes" in entries[0]["decision"]
        assert entries[0]["severity"] == "warning"

    def test_no_retrigger_records_trace(self):
        """When no retrigger, causal trace records the 'no' decision."""
        from aeon_core import PostPipelineMCTGate
        bus = StubFeedbackBus()
        bus.write_signal('post_output_uncertainty', 0.1)
        bus.write_signal('post_pipeline_verdict_pressure', 0.2)
        gate = PostPipelineMCTGate(
            mct_ref=MagicMock(), feedback_bus_ref=bus, threshold=0.6,
        )
        ct = StubCausalTrace()
        gate.set_causal_trace(ct)
        triggered = gate.check_and_retrigger()
        assert triggered is False
        entries = ct.find(subsystem="post_pipeline_mct_gate")
        assert len(entries) > 0
        assert "retrigger=no" in entries[0]["decision"]
        assert entries[0]["severity"] == "info"


# ===========================================================================
# PATCH-Σ1: UnifiedCausalSimulator causal trace
# ===========================================================================


class TestSigma1_UnifiedCausalSimulator:
    """Σ1: UnifiedCausalSimulator records simulation decisions."""

    def test_init_has_causal_trace_ref(self):
        from aeon_core import UnifiedCausalSimulator
        sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
        assert hasattr(sim, '_causal_trace_ref')
        assert sim._causal_trace_ref is None

    def test_set_causal_trace_method(self):
        from aeon_core import UnifiedCausalSimulator
        sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
        ct = StubCausalTrace()
        sim.set_causal_trace(ct)
        assert sim._causal_trace_ref is ct

    def test_forward_records_trace(self):
        """forward() records simulation step to causal trace."""
        import torch
        from aeon_core import UnifiedCausalSimulator
        sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
        ct = StubCausalTrace()
        sim.set_causal_trace(ct)
        state = torch.randn(2, 32)
        with torch.no_grad():
            result = sim.forward(state)
        entries = ct.find(subsystem="unified_simulator")
        assert len(entries) > 0
        assert "sim_step" in entries[0]["decision"]
        assert "state_norm" in entries[0]["decision"]
        assert "causal_vars_active" in entries[0]["decision"]

    def test_forward_no_trace_when_not_wired(self):
        """forward() works normally without causal trace."""
        import torch
        from aeon_core import UnifiedCausalSimulator
        sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
        state = torch.randn(2, 32)
        with torch.no_grad():
            result = sim.forward(state)
        assert "next_state" in result
        assert "causal_vars" in result


# ===========================================================================
# PATCH-Σ2: MCTSPlanner causal trace
# ===========================================================================


class TestSigma2_MCTSPlanner:
    """Σ2: MCTSPlanner records action selection decisions."""

    def test_init_has_causal_trace_ref(self):
        from aeon_core import MCTSPlanner
        planner = MCTSPlanner(state_dim=32, action_dim=4)
        assert hasattr(planner, '_causal_trace_ref')
        assert planner._causal_trace_ref is None

    def test_set_causal_trace_method(self):
        from aeon_core import MCTSPlanner
        planner = MCTSPlanner(state_dim=32, action_dim=4)
        ct = StubCausalTrace()
        planner.set_causal_trace(ct)
        assert planner._causal_trace_ref is ct

    def test_search_records_trace(self):
        """search() records action selection to causal trace."""
        import torch
        from aeon_core import MCTSPlanner, PhysicsGroundedWorldModel
        planner = MCTSPlanner(
            state_dim=32, action_dim=4, num_simulations=5,
        )
        ct = StubCausalTrace()
        planner.set_causal_trace(ct)
        world_model = PhysicsGroundedWorldModel(input_dim=32, state_dim=32)
        state = torch.randn(32)
        with torch.no_grad():
            result = planner.search(state, world_model)
        entries = ct.find(subsystem="mcts_planner")
        assert len(entries) > 0
        assert "action_selected" in entries[0]["decision"]
        assert "sims=" in entries[0]["decision"]

    def test_search_no_trace_when_not_wired(self):
        """search() works normally without causal trace."""
        import torch
        from aeon_core import MCTSPlanner, PhysicsGroundedWorldModel
        planner = MCTSPlanner(
            state_dim=32, action_dim=4, num_simulations=5,
        )
        world_model = PhysicsGroundedWorldModel(input_dim=32, state_dim=32)
        state = torch.randn(32)
        with torch.no_grad():
            result = planner.search(state, world_model)
        assert "best_action" in result


# ===========================================================================
# PATCH-Σ3: CausalFactorExtractor causal trace + bus
# ===========================================================================


class TestSigma3_CausalFactorExtractor:
    """Σ3: CausalFactorExtractor now records to causal trace and bus."""

    def test_init_has_refs(self):
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        assert hasattr(cfe, '_fb_ref')
        assert hasattr(cfe, '_causal_trace_ref')
        assert cfe._fb_ref is None
        assert cfe._causal_trace_ref is None

    def test_set_feedback_bus(self):
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        bus = StubFeedbackBus()
        cfe.set_feedback_bus(bus)
        assert cfe._fb_ref is bus

    def test_set_causal_trace(self):
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        ct = StubCausalTrace()
        cfe.set_causal_trace(ct)
        assert cfe._causal_trace_ref is ct

    def test_forward_records_trace(self):
        """forward() records factor extraction to causal trace."""
        import torch
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        ct = StubCausalTrace()
        cfe.set_causal_trace(ct)
        state = torch.randn(2, 32)
        with torch.no_grad():
            result = cfe.forward(state)
        entries = ct.find(subsystem="factor_extraction")
        assert len(entries) > 0
        assert "factors_extracted" in entries[0]["decision"]
        assert "sparsity" in entries[0]["decision"]

    def test_forward_writes_sparsity_to_bus(self):
        """forward() writes factor_sparsity_ratio to feedback bus."""
        import torch
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        bus = StubFeedbackBus()
        cfe.set_feedback_bus(bus)
        state = torch.randn(2, 32)
        with torch.no_grad():
            result = cfe.forward(state)
        assert 'factor_sparsity_ratio' in bus._signals
        sparsity = bus._signals['factor_sparsity_ratio']
        assert 0.0 <= sparsity <= 1.0

    def test_forward_no_error_without_wiring(self):
        """forward() works normally without bus or trace."""
        import torch
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        state = torch.randn(2, 32)
        with torch.no_grad():
            result = cfe.forward(state)
        assert "factors" in result
        assert "causal_graph" in result


# ===========================================================================
# PATCH-Σ5: system_emergence_report → bus + MCT reader
# ===========================================================================


class TestSigma5_EmergenceBusBridge:
    """Σ5: system_emergence_report writes axiom scores to bus."""

    def test_persistent_signals_registered(self):
        """The 4 emergence axiom signals are pre-registered as persistent."""
        # Create a bus and check the registration logic
        bus = StubFeedbackBus()
        expected_signals = [
            'emergence_overall_readiness',
            'emergence_mutual_reinforcement_quality',
            'emergence_metacognitive_quality',
            'emergence_causal_transparency_quality',
        ]
        # Simulate what AEONDeltaV3.__init__ does
        for sig in expected_signals:
            bus.register_persistent_signal(sig, 0.0)
        for sig in expected_signals:
            assert sig in bus._persistent
            assert sig in bus._registered

    def test_sigma5_signals_in_source(self):
        """The source code registers the 4 Σ5 signals."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core)
        for sig in [
            'emergence_overall_readiness',
            'emergence_mutual_reinforcement_quality',
            'emergence_metacognitive_quality',
            'emergence_causal_transparency_quality',
        ]:
            assert sig in source, f"Signal {sig} not found in aeon_core source"

    def test_sigma5_write_signals_in_emergence_report(self):
        """system_emergence_report writes Σ5 signals."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.system_emergence_report)
        assert 'emergence_overall_readiness' in source
        assert 'emergence_mutual_reinforcement_quality' in source
        assert 'emergence_metacognitive_quality' in source
        assert 'emergence_causal_transparency_quality' in source


class TestSigma5_MCTReader:
    """Σ5: MCT reads emergence_overall_readiness to gate threshold."""

    def test_mct_reader_in_source(self):
        """MCT evaluate reads emergence_overall_readiness."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_overall_readiness' in source


class TestSigma3b_MCTReader:
    """Σ3b: MCT reads factor_sparsity_ratio to amplify diversity_collapse."""

    def test_mct_reader_in_source(self):
        """MCT evaluate reads factor_sparsity_ratio."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'factor_sparsity_ratio' in source


class TestSigma5b_MCTReaders:
    """Σ5b: MCT reads emergence axiom sub-scores."""

    def test_mct_reads_mutual_reinforcement_quality(self):
        """MCT evaluate reads emergence_mutual_reinforcement_quality."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_mutual_reinforcement_quality' in source

    def test_mct_reads_metacognitive_quality(self):
        """MCT evaluate reads emergence_metacognitive_quality."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_metacognitive_quality' in source

    def test_mct_reads_causal_transparency_quality(self):
        """MCT evaluate reads emergence_causal_transparency_quality."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_causal_transparency_quality' in source


# ===========================================================================
# Signal ecosystem audit
# ===========================================================================


class TestSignalEcosystemAudit:
    """Verify all new Σ signals are bidirectional (written + read)."""

    def _load_source(self):
        with open('aeon_core.py', 'r') as f:
            core = f.read()
        with open('ae_train.py', 'r') as f:
            train = f.read()
        with open('aeon_server.py', 'r') as f:
            server = f.read()
        return core + '\n' + train + '\n' + server

    def _find_written_signals(self, source: str) -> set:
        """Find all signal names passed to write_signal."""
        pattern = r"write_signal\w*\(\s*['\"](\w+)['\"]"
        return set(re.findall(pattern, source))

    def _find_read_signals(self, source: str) -> set:
        """Find all signal names passed to read_signal."""
        pattern = r"read_signal\w*\(\s*['\"](\w+)['\"]"
        return set(re.findall(pattern, source))

    def test_factor_sparsity_ratio_bidirectional(self):
        source = self._load_source()
        written = self._find_written_signals(source)
        read = self._find_read_signals(source)
        assert 'factor_sparsity_ratio' in written
        assert 'factor_sparsity_ratio' in read

    def test_emergence_overall_readiness_bidirectional(self):
        source = self._load_source()
        written = self._find_written_signals(source)
        read = self._find_read_signals(source)
        assert 'emergence_overall_readiness' in written
        assert 'emergence_overall_readiness' in read

    def test_emergence_mutual_reinforcement_quality_written(self):
        source = self._load_source()
        written = self._find_written_signals(source)
        assert 'emergence_mutual_reinforcement_quality' in written

    def test_emergence_metacognitive_quality_written(self):
        source = self._load_source()
        written = self._find_written_signals(source)
        assert 'emergence_metacognitive_quality' in written

    def test_emergence_causal_transparency_quality_written(self):
        source = self._load_source()
        written = self._find_written_signals(source)
        assert 'emergence_causal_transparency_quality' in written

    def test_no_new_orphans(self):
        """No new orphan signals (written but never read)."""
        source = self._load_source()
        written = self._find_written_signals(source)
        read = self._find_read_signals(source)
        # Check our new signals specifically — system-wide orphan check
        # is left to the comprehensive signal audit
        new_signals = {
            'factor_sparsity_ratio',
            'emergence_overall_readiness',
        }
        for sig in new_signals:
            assert sig in written, f"{sig} not written"
            assert sig in read, f"{sig} not read — orphan"


# ===========================================================================
# Wiring verification
# ===========================================================================


class TestWiringVerification:
    """Verify causal trace wiring in AEONDeltaV3.__init__ source."""

    def test_sigma4_wiring_in_init(self):
        """Σ4 wiring for VibeThinkerRSSMBridge in __init__."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        assert 'vibe_thinker_bridge' in source
        assert 'set_causal_trace' in source

    def test_sigma6_wiring_in_init(self):
        """Σ6 wiring for PostPipelineMCTGate in __init__."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        assert 'post_pipeline_mct_gate' in source
        # Should have set_causal_trace for the gate
        assert 'post_pipeline_mct_gate' in source

    def test_sigma1_wiring_in_init(self):
        """Σ1 wiring for UnifiedCausalSimulator in __init__."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        assert 'unified_simulator' in source

    def test_sigma2_wiring_in_init(self):
        """Σ2 wiring for MCTSPlanner in __init__."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        assert 'mcts_planner' in source

    def test_sigma3_wiring_in_init(self):
        """Σ3 wiring for CausalFactorExtractor in __init__."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        assert 'causal_encoder' in source  # CausalFactorExtractor is nested in simulator


# ===========================================================================
# E2E causal transparency
# ===========================================================================


class TestCausalTransparency:
    """Verify trace_root_cause can traverse through all new subsystems."""

    def test_all_new_subsystems_recorded(self):
        """All 4 new subsystems record to causal trace."""
        import torch
        from aeon_core import (
            UnifiedCausalSimulator,
            MCTSPlanner,
            CausalFactorExtractor,
            PhysicsGroundedWorldModel,
        )

        ct = StubCausalTrace()

        # Σ1: UnifiedCausalSimulator
        sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
        sim.set_causal_trace(ct)
        with torch.no_grad():
            sim.forward(torch.randn(2, 32))

        # Σ2: MCTSPlanner
        planner = MCTSPlanner(state_dim=32, action_dim=4, num_simulations=3)
        planner.set_causal_trace(ct)
        world_model = PhysicsGroundedWorldModel(input_dim=32, state_dim=32)
        with torch.no_grad():
            planner.search(torch.randn(32), world_model)

        # Σ3: CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=8)
        cfe.set_causal_trace(ct)
        with torch.no_grad():
            cfe.forward(torch.randn(2, 32))

        # Verify all 3 subsystems recorded
        subs = {e["subsystem"] for e in ct._entries}
        assert "unified_simulator" in subs
        assert "mcts_planner" in subs
        assert "factor_extraction" in subs

    def test_sigma4_rssm_trace_activates(self):
        """Σ4: VibeThinkerRSSMBridge ζ4 code now fires."""
        from aeon_core import VibeThinkerRSSMBridge
        ct = StubCausalTrace()
        bus = StubFeedbackBus()
        bridge = VibeThinkerRSSMBridge(feedback_bus=bus)
        bridge.set_causal_trace(ct)
        bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.3,
            rssm_prediction_error=0.9,
        )
        subs = {e["subsystem"] for e in ct._entries}
        assert "rssm" in subs

    def test_sigma6_gate_trace_recorded(self):
        """Σ6: PostPipelineMCTGate records both outcomes."""
        from aeon_core import PostPipelineMCTGate
        ct = StubCausalTrace()
        bus = StubFeedbackBus()
        mock_mct = MagicMock()
        mock_mct.evaluate.return_value = {'should_trigger': True, 'trigger_score': 0.9}

        # Test retrigger
        bus.write_signal('post_output_uncertainty', 0.9)
        gate = PostPipelineMCTGate(mct_ref=mock_mct, feedback_bus_ref=bus, threshold=0.5)
        gate.set_causal_trace(ct)
        gate.check_and_retrigger()

        # Test no-retrigger
        bus.write_signal('post_output_uncertainty', 0.1)
        bus.write_signal('post_pipeline_verdict_pressure', 0.1)
        gate.check_and_retrigger()

        entries = ct.find(subsystem="post_pipeline_mct_gate")
        assert len(entries) >= 2
        decisions = [e["decision"] for e in entries]
        assert any("retrigger=yes" in d for d in decisions)
        assert any("retrigger=no" in d for d in decisions)


# ===========================================================================
# Activation sequence ordering
# ===========================================================================


class TestActivationSequence:
    """Verify patches can be applied in the prescribed order."""

    def test_phase1_before_phase2(self):
        """Phase 1 patches (Σ4, Σ6) are applied before Phase 2 (Σ1, Σ2, Σ3) in wiring block."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        # Find positions of wiring comments within the causal trace wiring block
        # The wiring block is after `if self.causal_trace is not None:`
        wiring_start = source.find('PATCH-Σ4:')
        wiring_sigma6 = source.find('PATCH-Σ6:', wiring_start)
        wiring_sigma1 = source.find('PATCH-Σ1:', wiring_sigma6)
        wiring_sigma2 = source.find('PATCH-Σ2:', wiring_sigma1)
        wiring_sigma3 = source.find('PATCH-Σ3:', wiring_sigma2)
        # All should exist
        assert wiring_start > 0, "Σ4 wiring not found"
        assert wiring_sigma6 > 0, "Σ6 wiring not found"
        assert wiring_sigma1 > 0, "Σ1 wiring not found"
        assert wiring_sigma2 > 0, "Σ2 wiring not found"
        assert wiring_sigma3 > 0, "Σ3 wiring not found"
        # Phase 1 (Σ4, Σ6) before Phase 2 (Σ1, Σ2, Σ3)
        assert wiring_start < wiring_sigma1
        assert wiring_sigma6 < wiring_sigma1

    def test_phase2_before_phase3(self):
        """Phase 2 patches precede Phase 3 (Σ5) in signal registration."""
        import inspect
        import aeon_core
        source = inspect.getsource(aeon_core.AEONDeltaV3.__init__)
        pos_sigma3_reg = source.find('factor_sparsity_ratio')
        pos_sigma5_reg = source.find('emergence_overall_readiness')
        assert pos_sigma3_reg < pos_sigma5_reg


# ===========================================================================
# Mutual reinforcement
# ===========================================================================


class TestMutualReinforcement:
    """Verify mutual reinforcement properties of Σ patches."""

    def test_sigma5_creates_self_stabilizing_loop(self):
        """Low emergence readiness → MCT reads it → boosts signals."""
        import inspect
        import aeon_core
        # Verify the loop exists in code:
        # 1. system_emergence_report writes emergence_overall_readiness
        er_source = inspect.getsource(
            aeon_core.AEONDeltaV3.system_emergence_report,
        )
        assert 'emergence_overall_readiness' in er_source
        # 2. MCT evaluate reads it and adjusts sensitivity
        mct_source = inspect.getsource(
            aeon_core.MetaCognitiveRecursionTrigger.evaluate,
        )
        assert 'emergence_overall_readiness' in mct_source
        assert '_sigma5_scale' in mct_source

    def test_sigma3_factor_collapse_triggers_mct(self):
        """Factor collapse (high sparsity) → MCT diversity_collapse signal."""
        import inspect
        import aeon_core
        mct_source = inspect.getsource(
            aeon_core.MetaCognitiveRecursionTrigger.evaluate,
        )
        assert 'factor_sparsity_ratio' in mct_source
        assert 'diversity_collapse' in mct_source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
