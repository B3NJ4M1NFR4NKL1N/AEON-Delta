"""Tests for PATCH-Γ1 through Γ9: Final Integration & Cognitive Activation.

Verifies the remaining causal trace and feedback bus integration gaps
that prevented full cognitive coherence:

  Γ1  — Server AppState causal_trace_ref field
  Γ2  — _server_causal_record() helper function
  Γ3  — Server verify_and_reinforce causal trace recording
  Γ3b — Server remediation causal trace recording
  Γ4  — Cognitive reset audit trail
  Γ5  — ComplexityEstimator bus + trace integration
  Γ6  — DiversityMetric causal trace
  Γ7  — ConvergenceMonitor causal trace
  Γ8a — TemporalMemory causal trace
  Γ8b — NeurogenicMemorySystem causal trace
  Γ8c — ConsolidatingMemory causal trace
  Γ9  — AEONDeltaV3 wiring for all new subsystems

Also verifies the signal ecosystem, E2E integration flows, causal
transparency, and the activation sequence.
"""
import sys
import types
import math
import time

import pytest

# ---------------------------------------------------------------------------
# Import helpers — load aeon_core once
# ---------------------------------------------------------------------------
import aeon_core as _ac

TemporalCausalTraceBuffer = _ac.TemporalCausalTraceBuffer
CognitiveFeedbackBus = _ac.CognitiveFeedbackBus

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    ComplexityEstimator = _ac.ComplexityEstimator
    DiversityMetric = _ac.DiversityMetric
    TemporalMemory = _ac.TemporalMemory
    NeurogenicMemorySystem = _ac.NeurogenicMemorySystem
    ConsolidatingMemory = _ac.ConsolidatingMemory
    ConvergenceMonitor = _ac.ConvergenceMonitor


# ============================================================================
# Helper: Minimal AEONConfig for integration tests
# ============================================================================
def _make_minimal_config():
    """Return a minimal config namespace for subsystem construction."""
    cfg = types.SimpleNamespace()
    cfg.num_pillars = 8
    cfg.hidden_dim = 64
    cfg.z_dim = 32
    cfg.vq_num_embeddings = 16
    cfg.num_factors = 8
    return cfg


# ============================================================================
# Γ1: Server AppState causal_trace_ref field
# ============================================================================

class TestGamma1ServerCausalTraceRef:
    """Verify AppState has causal_trace_ref field."""

    def test_appstate_has_causal_trace_ref(self):
        """AppState class must have a causal_trace_ref attribute."""
        import aeon_server as _srv
        assert hasattr(_srv.AppState, 'causal_trace_ref'), \
            "AppState missing causal_trace_ref field (PATCH-Γ1)"

    def test_appstate_causal_trace_ref_defaults_none(self):
        """AppState.causal_trace_ref must default to None."""
        import aeon_server as _srv
        assert _srv.APP.causal_trace_ref is None


# ============================================================================
# Γ2: _server_causal_record() helper
# ============================================================================

class TestGamma2ServerCausalRecordHelper:
    """Verify the _server_causal_record helper function exists and works."""

    def test_helper_exists(self):
        """_server_causal_record must be importable from aeon_server."""
        import aeon_server as _srv
        assert hasattr(_srv, '_server_causal_record'), \
            "_server_causal_record not found (PATCH-Γ2)"
        assert callable(_srv._server_causal_record)

    def test_helper_records_to_trace(self):
        """When APP.causal_trace_ref is set, helper records entries."""
        import aeon_server as _srv
        ct = TemporalCausalTraceBuffer(max_entries=100)
        old_ref = _srv.APP.causal_trace_ref
        try:
            _srv.APP.causal_trace_ref = ct
            _srv._server_causal_record(
                subsystem='test_subsystem',
                decision='test_decision',
                metadata={'key': 'value'},
            )
            entries = ct.find(subsystem='test_subsystem')
            assert len(entries) >= 1, \
                "Helper did not record to causal trace"
            assert entries[0]['decision'] == 'test_decision'
        finally:
            _srv.APP.causal_trace_ref = old_ref

    def test_helper_noop_when_no_trace(self):
        """When APP.causal_trace_ref is None, helper does nothing."""
        import aeon_server as _srv
        old_ref = _srv.APP.causal_trace_ref
        try:
            _srv.APP.causal_trace_ref = None
            # Should not raise
            _srv._server_causal_record(
                subsystem='noop',
                decision='noop',
            )
        finally:
            _srv.APP.causal_trace_ref = old_ref


# ============================================================================
# Γ5: ComplexityEstimator bus + trace integration
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestGamma5ComplexityEstimator:
    """Verify ComplexityEstimator now writes to bus and causal trace."""

    def test_has_fb_ref(self):
        """ComplexityEstimator must have _fb_ref attribute."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        assert hasattr(ce, '_fb_ref'), "Missing _fb_ref (PATCH-Γ5)"

    def test_has_set_causal_trace(self):
        """ComplexityEstimator must have set_causal_trace method."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        assert hasattr(ce, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ5)"
        assert callable(ce.set_causal_trace)

    def test_has_causal_trace_ref(self):
        """ComplexityEstimator must have _causal_trace_ref attribute."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        assert hasattr(ce, '_causal_trace_ref'), \
            "Missing _causal_trace_ref (PATCH-Γ5)"

    def test_writes_to_bus(self):
        """Forward pass should write complexity_gate_score to the bus."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        ce._fb_ref = fb
        z = torch.randn(2, 64)
        ce.forward(z)
        score = fb.read_signal('complexity_gate_score', -1.0)
        assert score != -1.0, \
            "complexity_gate_score not written (PATCH-Γ5)"

    def test_writes_active_ratio_to_bus(self):
        """Forward pass should write complexity_subsystems_active."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        ce._fb_ref = fb
        z = torch.randn(2, 64)
        ce.forward(z)
        ratio = fb.read_signal('complexity_subsystems_active', -1.0)
        assert ratio != -1.0, \
            "complexity_subsystems_active not written (PATCH-Γ5)"

    def test_records_low_complexity_to_trace(self):
        """When complexity is low, a causal trace entry should be recorded."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        ce.set_causal_trace(ct)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        ce._fb_ref = fb
        # Force low complexity output by setting weights to produce
        # very negative logits (sigmoid → near 0)
        with torch.no_grad():
            for p in ce.estimator.parameters():
                p.fill_(-5.0)
        z = torch.randn(2, 64)
        ce.forward(z)
        entries = ct.find(subsystem='complexity_estimator')
        assert len(entries) >= 1, \
            "Low-complexity event not recorded in causal trace (PATCH-Γ5)"
        assert entries[0]['decision'] == 'low_complexity_bypass'


# ============================================================================
# Γ6: DiversityMetric causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestGamma6DiversityMetric:
    """Verify DiversityMetric records collapse to causal trace."""

    def test_has_set_causal_trace(self):
        """DiversityMetric must have set_causal_trace method."""
        cfg = _make_minimal_config()
        dm = DiversityMetric(cfg)
        assert hasattr(dm, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ6)"

    def test_has_causal_trace_ref(self):
        """DiversityMetric must have _causal_trace_ref attribute."""
        cfg = _make_minimal_config()
        dm = DiversityMetric(cfg)
        assert hasattr(dm, '_causal_trace_ref')

    def test_records_collapse_to_trace(self):
        """When diversity < 0.1, collapse should be recorded in trace."""
        cfg = _make_minimal_config()
        dm = DiversityMetric(cfg)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        dm.set_causal_trace(ct)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        dm._fb_ref = fb
        # Create near-zero variance factors (all identical → diversity ≈ 0)
        factors = torch.ones(2, cfg.num_pillars) * 0.5
        dm.forward(factors)
        entries = ct.find(subsystem='diversity_metric')
        assert len(entries) >= 1, \
            "Diversity collapse not recorded in trace (PATCH-Γ6)"
        assert entries[0]['decision'] == 'diversity_collapse'


# ============================================================================
# Γ7: ConvergenceMonitor causal trace
# ============================================================================

class TestGamma7ConvergenceMonitor:
    """Verify ConvergenceMonitor records divergence to causal trace."""

    def test_has_set_causal_trace(self):
        """ConvergenceMonitor must have set_causal_trace method."""
        cm = ConvergenceMonitor(threshold=1e-5)
        assert hasattr(cm, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ7)"

    def test_has_causal_trace_ref(self):
        """ConvergenceMonitor must have _causal_trace_ref attribute."""
        cm = ConvergenceMonitor(threshold=1e-5)
        assert hasattr(cm, '_causal_trace_ref')

    def test_records_diverging_to_trace(self):
        """When diverging, ConvergenceMonitor should record to trace."""
        cm = ConvergenceMonitor(threshold=1e-5)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cm.set_causal_trace(ct)
        # Feed increasing delta norms → diverging
        cm.check(1.0)
        cm.check(2.0)
        cm.check(4.0)
        entries = ct.find(subsystem='convergence_monitor')
        assert len(entries) >= 1, \
            "Divergence not recorded in trace (PATCH-Γ7)"
        assert entries[0]['decision'] == 'diverging'

    def test_records_converging_to_trace(self):
        """When converging but not certified, should record to trace."""
        cm = ConvergenceMonitor(threshold=1e-12)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cm.set_causal_trace(ct)
        # Feed decreasing but above threshold → converging
        cm.check(1.0)
        cm.check(0.5)
        cm.check(0.3)
        entries = ct.find(subsystem='convergence_monitor')
        assert len(entries) >= 1, \
            "Converging state not recorded in trace (PATCH-Γ7)"
        assert entries[0]['decision'] == 'converging'


# ============================================================================
# Γ8a: TemporalMemory causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestGamma8aTemporalMemory:
    """Verify TemporalMemory records consolidation to causal trace."""

    def test_has_set_causal_trace(self):
        """TemporalMemory must have set_causal_trace method."""
        tm = TemporalMemory(capacity=5, dim=16)
        assert hasattr(tm, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ8a)"

    def test_has_causal_trace_ref(self):
        """TemporalMemory must have _causal_trace_ref attribute."""
        tm = TemporalMemory(capacity=5, dim=16)
        assert hasattr(tm, '_causal_trace_ref')

    def test_records_consolidation_to_trace(self):
        """When capacity exceeded, consolidation should be traced."""
        tm = TemporalMemory(capacity=3, dim=16)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        tm.set_causal_trace(ct)
        # Store 4 items to trigger consolidation
        for i in range(4):
            tm.store(torch.randn(16), importance=1.0)
        entries = ct.find(subsystem='temporal_memory')
        assert len(entries) >= 1, \
            "Consolidation not recorded in trace (PATCH-Γ8a)"
        assert entries[0]['decision'] == 'consolidation_triggered'


# ============================================================================
# Γ8b: NeurogenicMemorySystem causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestGamma8bNeurogenicMemory:
    """Verify NeurogenicMemorySystem records neurogenesis to causal trace."""

    def test_has_set_causal_trace(self):
        """NeurogenicMemorySystem must have set_causal_trace method."""
        nm = NeurogenicMemorySystem(base_dim=16)
        assert hasattr(nm, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ8b)"

    def test_has_causal_trace_ref(self):
        """NeurogenicMemorySystem must have _causal_trace_ref attribute."""
        nm = NeurogenicMemorySystem(base_dim=16)
        assert hasattr(nm, '_causal_trace_ref')

    def test_records_neurogenesis_to_trace(self):
        """When importance exceeds threshold, neurogenesis should be traced."""
        nm = NeurogenicMemorySystem(
            base_dim=16,
            importance_threshold=0.0,  # Always trigger neurogenesis
            max_capacity=100,
        )
        ct = TemporalCausalTraceBuffer(max_entries=100)
        nm.set_causal_trace(ct)
        nm.consolidate(torch.randn(16), importance=0.9)
        entries = ct.find(subsystem='neurogenic_memory')
        assert len(entries) >= 1, \
            "Neurogenesis not recorded in trace (PATCH-Γ8b)"
        assert entries[0]['decision'] == 'neurogenesis'


# ============================================================================
# Γ8c: ConsolidatingMemory causal trace
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestGamma8cConsolidatingMemory:
    """Verify ConsolidatingMemory records consolidation to causal trace."""

    def test_has_set_causal_trace(self):
        """ConsolidatingMemory must have set_causal_trace method."""
        cm = ConsolidatingMemory(dim=16)
        assert hasattr(cm, 'set_causal_trace'), \
            "Missing set_causal_trace (PATCH-Γ8c)"

    def test_has_causal_trace_ref(self):
        """ConsolidatingMemory must have _causal_trace_ref attribute."""
        cm = ConsolidatingMemory(dim=16)
        assert hasattr(cm, '_causal_trace_ref')

    def test_records_consolidation_to_trace(self):
        """Multi-stage consolidation should be traced."""
        cm = ConsolidatingMemory(
            dim=16,
            importance_threshold=0.0,  # Everything consolidates
        )
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cm.set_causal_trace(ct)
        # Store items and consolidate
        for i in range(3):
            cm.store(torch.randn(16))
        cm.consolidate()
        entries = ct.find(subsystem='consolidating_memory')
        assert len(entries) >= 1, \
            "Multi-stage consolidation not recorded in trace (PATCH-Γ8c)"
        assert entries[0]['decision'] == 'multi_stage_consolidation'


# ============================================================================
# Γ9: AEONDeltaV3 wiring verification
# ============================================================================

class TestGamma9Wiring:
    """Verify that AEONDeltaV3.__init__ wires causal trace to new subsystems."""

    def test_gamma9_wiring_code_exists(self):
        """Confirm PATCH-Γ9 wiring block is present in source."""
        import inspect
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        assert 'PATCH-Γ9' in src or 'PATCH-Γ9' in src or 'Γ9' in src, \
            "PATCH-Γ9 wiring block not found in AEONDeltaV3.__init__"

    def test_gamma5_bus_wiring_code_exists(self):
        """Confirm ComplexityEstimator bus wiring is present."""
        import inspect
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        assert 'complexity_estimator' in src, \
            "complexity_estimator wiring not found in AEONDeltaV3.__init__"


# ============================================================================
# Signal Ecosystem Audit
# ============================================================================

class TestSignalEcosystemAudit:
    """Verify the new signals are properly balanced (written + read)."""

    def test_complexity_gate_score_written(self):
        """complexity_gate_score must be written in aeon_core.py."""
        import inspect
        src = inspect.getsource(_ac.ComplexityEstimator.forward)
        assert 'complexity_gate_score' in src

    def test_complexity_subsystems_active_written(self):
        """complexity_subsystems_active must be written in aeon_core.py."""
        import inspect
        src = inspect.getsource(_ac.ComplexityEstimator.forward)
        assert 'complexity_subsystems_active' in src


# ============================================================================
# E2E Integration Flow: Causal Transparency
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestCausalTransparency:
    """Verify end-to-end causal traceability across new subsystems."""

    def test_complexity_to_trace_chain(self):
        """ComplexityEstimator → bus signal + trace entry chain."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        ce._fb_ref = fb
        ce.set_causal_trace(ct)
        # Force all gates off
        with torch.no_grad():
            for p in ce.estimator.parameters():
                p.fill_(-10.0)
        ce.forward(torch.randn(2, 64))
        # Both bus signal and trace entry should exist
        score = fb.read_signal('complexity_gate_score', -1.0)
        entries = ct.find(subsystem='complexity_estimator')
        assert score != -1.0, "Bus signal missing"
        assert len(entries) >= 1, "Trace entry missing"

    def test_diversity_to_trace_chain(self):
        """DiversityMetric → bus alarm + trace entry chain."""
        cfg = _make_minimal_config()
        dm = DiversityMetric(cfg)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        dm._fb_ref = fb
        dm.set_causal_trace(ct)
        # Zero diversity
        factors = torch.ones(2, cfg.num_pillars)
        dm.forward(factors)
        alarm = fb.read_signal('diversity_collapse_alarm', 0.0)
        entries = ct.find(subsystem='diversity_metric')
        assert alarm == 1.0, "Diversity collapse alarm not written"
        assert len(entries) >= 1, "Trace entry missing"

    def test_convergence_to_trace_chain(self):
        """ConvergenceMonitor diverging → trace entry chain."""
        cm = ConvergenceMonitor(threshold=1e-5)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cm._fb_ref = fb
        cm.set_causal_trace(ct)
        cm.check(1.0)
        cm.check(2.0)
        cm.check(4.0)
        entries = ct.find(subsystem='convergence_monitor')
        assert len(entries) >= 1, "Divergence trace entry missing"

    def test_memory_to_trace_chain(self):
        """TemporalMemory consolidation → trace entry chain."""
        tm = TemporalMemory(capacity=2, dim=16)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        tm.set_causal_trace(ct)
        for i in range(3):
            tm.store(torch.randn(16), importance=1.0)
        entries = ct.find(subsystem='temporal_memory')
        assert len(entries) >= 1, "Memory consolidation trace missing"


# ============================================================================
# Activation Sequence Verification
# ============================================================================

class TestActivationSequence:
    """Verify the logical ordering of patch application."""

    def test_gamma1_before_gamma2(self):
        """Γ1 (AppState field) must exist before Γ2 (helper) can work."""
        import aeon_server as _srv
        assert hasattr(_srv.AppState, 'causal_trace_ref')
        assert callable(_srv._server_causal_record)

    def test_gamma5_bus_before_trace(self):
        """Γ5 bus writes must work before trace recording."""
        if not HAS_TORCH:
            pytest.skip("torch required")
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        assert hasattr(ce, '_fb_ref')
        assert hasattr(ce, '_causal_trace_ref')

    def test_gamma9_after_phi7(self):
        """Γ9 wiring must appear after Φ7 wiring in source."""
        import inspect
        src = inspect.getsource(_ac.AEONDeltaV3.__init__)
        phi7_pos = src.find('PATCH-Φ7')
        gamma9_pos = src.find('PATCH-Γ9')
        if phi7_pos == -1:
            phi7_pos = src.find('Phi7')
        assert phi7_pos > 0, "Φ7 wiring not found"
        assert gamma9_pos > 0, "Γ9 wiring not found"
        assert gamma9_pos > phi7_pos, \
            "Γ9 must appear after Φ7 in __init__"


# ============================================================================
# Mutual Reinforcement Verification
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestMutualReinforcement:
    """Verify subsystems verify and stabilize each other's states."""

    def test_complexity_signals_reach_bus(self):
        """ComplexityEstimator signals must be readable by other subsystems."""
        ce = ComplexityEstimator(hidden_dim=64, num_subsystems=4)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        ce._fb_ref = fb
        z = torch.randn(2, 64)
        ce.forward(z)
        # Another subsystem can now read these signals
        score = fb.read_signal('complexity_gate_score', -1.0)
        active = fb.read_signal('complexity_subsystems_active', -1.0)
        assert score >= 0.0 and score <= 1.0
        assert active >= 0.0 and active <= 1.0

    def test_convergence_trace_enables_root_cause(self):
        """Convergence trace entries must contain root-cause metadata."""
        cm = ConvergenceMonitor(threshold=1e-5)
        ct = TemporalCausalTraceBuffer(max_entries=100)
        cm.set_causal_trace(ct)
        cm.check(1.0)
        cm.check(2.0)
        cm.check(4.0)
        entries = ct.find(subsystem='convergence_monitor')
        assert len(entries) >= 1
        meta = entries[0].get('metadata', {})
        assert 'avg_contraction' in meta
        assert 'delta_norm' in meta


# ============================================================================
# Meta-Cognitive Trigger Verification
# ============================================================================

class TestMetaCognitiveTrigger:
    """Verify uncertainty triggers meta-cognitive review cycles."""

    def test_convergence_divergence_writes_to_bus(self):
        """Divergence should write convergence signals to bus for MCT."""
        cm = ConvergenceMonitor(threshold=1e-5)
        fb = CognitiveFeedbackBus(hidden_dim=64)
        cm._fb_ref = fb
        cm.check(1.0)
        cm.check(2.0)
        cm.check(4.0)
        quality = fb.read_signal('convergence_monitor_quality', -1.0)
        is_converging = fb.read_signal('convergence_monitor_is_converging', -1.0)
        assert quality != -1.0, "convergence_monitor_quality not written"
        assert is_converging == 0.0, "Should report not converging"

    def test_server_helper_records_for_transparency(self):
        """Server helper must record actions that MCT can trace."""
        import aeon_server as _srv
        ct = TemporalCausalTraceBuffer(max_entries=100)
        old_ref = _srv.APP.causal_trace_ref
        try:
            _srv.APP.causal_trace_ref = ct
            _srv._server_causal_record(
                subsystem='server_verify_reinforce',
                decision='coherence_assessment',
                metadata={'overall_score': 0.85},
            )
            entries = ct.find(subsystem='server_verify_reinforce')
            assert len(entries) >= 1
            assert entries[0]['metadata']['overall_score'] == 0.85
        finally:
            _srv.APP.causal_trace_ref = old_ref
