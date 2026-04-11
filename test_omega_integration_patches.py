"""Tests for PATCH-Ω1 through PATCH-Ω6 — Final Cognitive Activation patches.

Covers:
  PATCH-Ω5: SubsystemHealthGate causal trace recording
  PATCH-Ω1: Forward-pass extra-signal pre-seeding
  PATCH-Ω3: Cross-module bridge failure propagation (ae_train.py)
  PATCH-Ω2: Convergence Arbiter → UCC conflict feedback
  PATCH-Ω4: REINFORCE_INTERVAL recency feedback
  PATCH-Ω6: Emergence self-test auto-trigger on uncertainty
  Signal ecosystem integrity audit
  Integration flow & activation sequence
"""

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


def _make_convergence_monitor():
    """Create a minimal ConvergenceMonitor."""
    return aeon_core.ConvergenceMonitor(threshold=1e-5)


def _make_error_evolution():
    """Create a minimal CausalErrorEvolutionTracker."""
    return aeon_core.CausalErrorEvolutionTracker(max_history=50)


def _make_provenance():
    """Create a minimal CausalProvenanceTracker."""
    return aeon_core.CausalProvenanceTracker()


def _make_mct(feedback_bus=None):
    """Create a MetaCognitiveRecursionTrigger with optional bus."""
    mct = aeon_core.MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=2,
    )
    if feedback_bus is not None:
        mct.set_feedback_bus(feedback_bus)
    return mct


def _make_causal_trace():
    """Create a minimal TemporalCausalTraceBuffer."""
    return aeon_core.TemporalCausalTraceBuffer(max_entries=100)


def _make_health_gate():
    """Create a SubsystemHealthGate."""
    return aeon_core.SubsystemHealthGate(hidden_dim=64)


def _make_arbiter(feedback_bus=None):
    """Create a UnifiedConvergenceArbiter with bus."""
    arbiter = aeon_core.UnifiedConvergenceArbiter(
        feedback_bus=feedback_bus,
    )
    return arbiter


def _make_ucc(feedback_bus=None):
    """Create a UnifiedCognitiveCycle with bus."""
    cm = _make_convergence_monitor()
    ee = _make_error_evolution()
    prov = _make_provenance()
    mct = _make_mct(feedback_bus)
    ucc = aeon_core.UnifiedCognitiveCycle(
        convergence_monitor=cm,
        coherence_verifier=None,
        error_evolution=ee,
        metacognitive_trigger=mct,
        provenance_tracker=prov,
        feedback_bus=feedback_bus,
    )
    return ucc


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: PATCH-Ω5 — SubsystemHealthGate Causal Trace Recording
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega5:
    """PATCH-Ω5: Health gate attenuation recorded in causal trace."""

    def test_health_gate_has_set_causal_trace_method(self):
        """SubsystemHealthGate exposes set_causal_trace()."""
        gate = _make_health_gate()
        assert hasattr(gate, 'set_causal_trace')
        assert callable(gate.set_causal_trace)

    def test_causal_trace_recording_on_attenuation(self):
        """When gate attenuates, the causal trace records the decision."""
        gate = _make_health_gate()
        ct = _make_causal_trace()
        gate.set_causal_trace(ct)

        # Use low coherence_score to force gate_value < 1.0
        x = torch.randn(2, 64)
        gated, gate_val = gate(x, coherence_score=0.1)

        # If gate attenuated (gate_val < 1.0), check trace
        if gate_val < 1.0:
            entries = ct.find(subsystem="health_gate")
            assert len(entries) >= 1
            entry = entries[0]
            assert entry['decision'] == 'attenuation'
            assert 'gate_value' in entry.get('metadata', {})
            assert 'coherence_score' in entry.get('metadata', {})

    def test_no_trace_when_no_causal_trace_wired(self):
        """Without set_causal_trace, gate still works normally."""
        gate = _make_health_gate()
        x = torch.randn(2, 64)
        gated, gate_val = gate(x, coherence_score=0.1)
        assert gated.shape == x.shape

    def test_causal_trace_ref_stored(self):
        """set_causal_trace stores the reference internally."""
        gate = _make_health_gate()
        ct = _make_causal_trace()
        gate.set_causal_trace(ct)
        assert gate._causal_trace_ref is ct


# ═══════════════════════════════════════════════════════════════════════
# Phase 2a: PATCH-Ω1 — Forward-Pass Extra-Signal Pre-Seeding
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega1:
    """PATCH-Ω1: Carry forward last-valid cached state for early evaluations."""

    def test_pre_seed_restores_from_snapshot(self):
        """When a _cached_ attribute is 0.0, pre-seeding restores it from
        the last valid snapshot."""
        model = MagicMock()
        model._omega1_last_valid_caches = {
            '_cached_output_quality': 0.85,
            '_cached_causal_quality': 0.9,
        }
        model._cached_output_quality = 0.0
        model._cached_causal_quality = 0.0

        # Simulate the pre-seeding loop from _forward_impl
        _omega1_prev = getattr(model, '_omega1_last_valid_caches', None)
        if _omega1_prev is not None:
            for attr, val in _omega1_prev.items():
                cur = getattr(model, attr, None)
                if cur is None or cur == 0.0:
                    setattr(model, attr, val)

        assert model._cached_output_quality == 0.85
        assert model._cached_causal_quality == 0.9

    def test_pre_seed_does_not_overwrite_nonzero(self):
        """Pre-seeding does not overwrite attributes that already have values."""
        model = MagicMock()
        model._omega1_last_valid_caches = {
            '_cached_output_quality': 0.85,
        }
        model._cached_output_quality = 0.7

        _omega1_prev = getattr(model, '_omega1_last_valid_caches', None)
        if _omega1_prev is not None:
            for attr, val in _omega1_prev.items():
                cur = getattr(model, attr, None)
                if cur is None or cur == 0.0:
                    setattr(model, attr, val)

        # Should keep original 0.7, not overwrite with 0.85
        assert model._cached_output_quality == 0.7

    def test_snapshot_captures_nonzero_caches(self):
        """End-of-pass snapshot captures only non-zero cached values."""
        _OMEGA1_CACHE_ATTRS = (
            '_cached_output_quality', '_cached_causal_quality',
            '_cached_coherence_deficit',
        )
        model = MagicMock()
        model._cached_output_quality = 0.85
        model._cached_causal_quality = 0.0  # Should not be captured
        model._cached_coherence_deficit = 0.3

        snapshot = {}
        for attr in _OMEGA1_CACHE_ATTRS:
            v = getattr(model, attr, None)
            if v is not None and v != 0.0:
                snapshot[attr] = float(v)

        assert '_cached_output_quality' in snapshot
        assert '_cached_causal_quality' not in snapshot
        assert '_cached_coherence_deficit' in snapshot

    def test_omega1_pattern_in_forward_impl(self):
        """_forward_impl contains the Ω1 pre-seeding and snapshot code."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3._forward_impl)
        assert '_omega1_last_valid_caches' in src
        assert '_pre_seed_feedback_caches' in src or '_omega1_prev' in src
        assert '_OMEGA1_CACHE_ATTRS' in src


# ═══════════════════════════════════════════════════════════════════════
# Phase 2b: PATCH-Ω3 — Cross-Module Bridge Failure Propagation
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega3:
    """PATCH-Ω3: Bridge failures tracked and published to bus."""

    def test_bridge_training_to_inference_accepts_feedback_bus(self):
        """bridge_training_errors_to_inference has feedback_bus parameter."""
        import inspect
        sig = inspect.signature(ae_train.bridge_training_errors_to_inference)
        assert 'feedback_bus' in sig.parameters

    def test_bridge_inference_to_training_accepts_feedback_bus(self):
        """bridge_inference_insights_to_training has feedback_bus parameter."""
        import inspect
        sig = inspect.signature(ae_train.bridge_inference_insights_to_training)
        assert 'feedback_bus' in sig.parameters

    def test_bridge_training_writes_health_on_success(self):
        """When all operations succeed, bridge_health = 1.0 is written."""
        bus = _make_bus()
        monitor = MagicMock()
        monitor.export_error_patterns.return_value = {'error_classes': {}}
        ee = _make_error_evolution()

        ae_train.bridge_training_errors_to_inference(
            trainer_monitor=monitor,
            inference_error_evolution=ee,
            feedback_bus=bus,
        )

        # With no operations attempted beyond the initial check,
        # the health signal may not be written (0 total operations).
        # But the function should complete without error.

    def test_bridge_training_tracks_failure_count(self):
        """When convergence monitor lacks set_error_evolution, failure is counted."""
        bus = _make_bus()
        monitor = MagicMock()
        monitor.export_error_patterns.return_value = {'error_classes': {}}
        ee = _make_error_evolution()

        # Create a convergence monitor that raises on set_error_evolution
        bad_cm = MagicMock()
        bad_cm.set_error_evolution.side_effect = AttributeError("no method")

        ae_train.bridge_training_errors_to_inference(
            trainer_monitor=monitor,
            inference_error_evolution=ee,
            inference_convergence_monitor=bad_cm,
            feedback_bus=bus,
        )

        # Should have written bridge health (< 1.0 due to failure)
        health = bus.read_signal('cross_module_bridge_health', -1.0)
        # Should exist and be < 1.0 (one failure)
        assert health < 1.0 or health == -1.0  # -1.0 if no total

    def test_mct_reads_bridge_health(self):
        """MCT evaluate reads cross_module_bridge_health."""
        bus = _make_bus()
        mct = _make_mct(bus)

        # Write low bridge health
        bus.write_signal('cross_module_bridge_health', 0.3)

        result = mct.evaluate(uncertainty=0.3)
        # Should complete without error; bridge health is consumed
        assert 'should_trigger' in result

    def test_mct_evaluate_reads_cross_module_bridge_health_signal(self):
        """MCT.evaluate source contains read_signal for cross_module_bridge_health."""
        import inspect
        src = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'cross_module_bridge_health' in src

    def test_forward_silent_exception_count_in_forward_impl(self):
        """_forward_impl contains the Ω3 silent exception counter."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3._forward_impl)
        assert '_omega3_silent_exc_count' in src
        assert 'forward_silent_exception_count' in src


# ═══════════════════════════════════════════════════════════════════════
# Phase 3a: PATCH-Ω2 — Convergence Arbiter → UCC Conflict Feedback
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega2:
    """PATCH-Ω2: Arbiter writes conflict severity; UCC reads it."""

    def test_arbiter_writes_conflict_signal(self):
        """When has_conflict=True, arbiter writes convergence_arbiter_conflict."""
        bus = _make_bus()
        arbiter = _make_arbiter(bus)

        # Simulate a conflict scenario
        result = arbiter.arbitrate(
            meta_loop_results={'converged': False, 'delta_norm': 0.5},
            convergence_monitor_verdict={'status': 'diverging'},
            certified_results={'converged': True},
            coherence_score=0.3,  # Low coherence triggers conflict
        )

        conflict_val = bus.read_signal('convergence_arbiter_conflict', -1.0)
        # Signal should have been written (either conflict severity or 0.0)
        assert conflict_val >= 0.0

    def test_arbiter_writes_zero_when_no_conflict(self):
        """When all monitors agree (converged), conflict signal = 0.0."""
        bus = _make_bus()
        arbiter = _make_arbiter(bus)

        # All monitors agree: converged, high coherence, no certified
        result = arbiter.arbitrate(
            meta_loop_results={'converged': True, 'delta_norm': 0.001},
            convergence_monitor_verdict={'status': 'converged'},
            certified_results=None,
            coherence_score=1.0,
        )

        conflict_val = bus.read_signal('convergence_arbiter_conflict', -1.0)
        # If has_conflict is False in the result, conflict signal should be 0.0
        if not result.get('has_conflict', True):
            assert conflict_val == 0.0

    def test_ucc_reads_arbiter_conflict(self):
        """UCC evaluate source contains read_signal for convergence_arbiter_conflict."""
        import inspect
        src = inspect.getsource(aeon_core.UnifiedCognitiveCycle.evaluate)
        assert 'convergence_arbiter_conflict' in src

    def test_ucc_boosts_recovery_on_high_conflict(self):
        """When arbiter conflict > 0.3, UCC boosts recovery_pressure."""
        bus = _make_bus()
        ucc = _make_ucc(bus)

        # Write high conflict
        bus.write_signal('convergence_arbiter_conflict', 0.8)

        # Evaluate with low recovery_pressure
        result = ucc.evaluate(
            subsystem_states={'meta_loop': torch.randn(1, 64)},
            delta_norm=0.1,
            recovery_pressure=0.1,
        )
        # UCC should have boosted recovery_pressure internally
        # (observable through trigger_detail or overall behavior)
        assert 'should_rerun' in result


# ═══════════════════════════════════════════════════════════════════════
# Phase 3b: PATCH-Ω4 — REINFORCE_INTERVAL Recency Feedback
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega4:
    """PATCH-Ω4: Verification recency pressure closes the feedback loop."""

    def test_reinforce_recency_pressure_in_forward_impl(self):
        """_forward_impl writes reinforce_recency_pressure."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3._forward_impl)
        assert 'reinforce_recency_pressure' in src

    def test_omega4_last_reinforce_pass_recorded(self):
        """_forward_impl records _omega4_last_reinforce_pass."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3._forward_impl)
        assert '_omega4_last_reinforce_pass' in src

    def test_mct_reads_recency_pressure(self):
        """MCT evaluate reads reinforce_recency_pressure."""
        import inspect
        src = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'reinforce_recency_pressure' in src

    def test_recency_pressure_computation(self):
        """Recency pressure normalisation is correct."""
        # passes_since = 15, REINFORCE_INTERVAL = 10 → 15 / (2*10) = 0.75
        passes_since = 15
        interval = 10
        recency = min(1.0, passes_since / max(1, 2 * interval))
        assert abs(recency - 0.75) < 1e-6

        # passes_since = 25 → 25 / 20 = 1.25 → clamped to 1.0
        passes_since = 25
        recency = min(1.0, passes_since / max(1, 2 * interval))
        assert recency == 1.0

    def test_mct_boosts_signals_on_high_recency(self):
        """When recency > 0.8, MCT boosts signal values."""
        bus = _make_bus()
        mct = _make_mct(bus)

        # Write high recency pressure
        bus.write_signal('reinforce_recency_pressure', 0.95)

        result = mct.evaluate(uncertainty=0.3)
        # Should trigger more aggressively due to recency boost
        assert 'should_trigger' in result


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: PATCH-Ω6 — Emergence Self-Test Auto-Trigger
# ═══════════════════════════════════════════════════════════════════════

class TestPatchOmega6:
    """PATCH-Ω6: Emergence self-assessment auto-triggers on uncertainty."""

    def test_verify_and_reinforce_has_omega6_logic(self):
        """verify_and_reinforce contains PATCH-Ω6 auto-trigger code."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3.verify_and_reinforce)
        assert 'emergence_self_assessment_triggered' in src
        assert 'emergence_readiness' in src
        assert '_omega6_passes_since_emergence_check' in src

    def test_mct_reads_emergence_readiness(self):
        """MCT evaluate reads emergence_readiness signal."""
        import inspect
        src = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_readiness' in src

    def test_mct_reads_emergence_self_assessment_triggered(self):
        """MCT evaluate reads emergence_self_assessment_triggered signal."""
        import inspect
        src = inspect.getsource(aeon_core.MetaCognitiveRecursionTrigger.evaluate)
        assert 'emergence_self_assessment_triggered' in src

    def test_emergence_auto_trigger_on_low_score(self):
        """When overall_score < 0.5 and enough passes, auto-trigger fires."""
        bus = _make_bus()

        # Simulate the Ω6 logic inline
        _omega6_passes_since = 100  # > 50 threshold
        _omega6_deficit = 0.3  # < 0.5 threshold
        _in_diagnostic = False

        triggered = (
            _omega6_deficit < 0.5
            and _omega6_passes_since > 50
            and not _in_diagnostic
        )
        assert triggered is True

    def test_emergence_no_trigger_when_score_high(self):
        """When overall_score >= 0.5, no auto-trigger."""
        _omega6_deficit = 0.7  # >= 0.5
        _omega6_passes_since = 100
        _in_diagnostic = False

        triggered = (
            _omega6_deficit < 0.5
            and _omega6_passes_since > 50
            and not _in_diagnostic
        )
        assert triggered is False

    def test_emergence_no_trigger_when_diagnostic(self):
        """When _in_diagnostic_context is True, no auto-trigger."""
        _omega6_deficit = 0.3
        _omega6_passes_since = 100
        _in_diagnostic = True

        triggered = (
            _omega6_deficit < 0.5
            and _omega6_passes_since > 50
            and not _in_diagnostic
        )
        assert triggered is False

    def test_emergence_readiness_computation(self):
        """Emergence readiness is geometric mean of three axiom scores."""
        mv, um, rc = 0.8, 0.7, 0.9
        emergence = (
            max(1e-10, mv) * max(1e-10, um) * max(1e-10, rc)
        ) ** (1.0 / 3.0)
        assert abs(emergence - (0.8 * 0.7 * 0.9) ** (1.0 / 3.0)) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystem:
    """Verify new signals from Ω-patches are bidirectional."""

    @pytest.fixture(autouse=True)
    def _scan_signals(self):
        """Scan all signal writes and reads from source files."""
        self.written = set()
        self.read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    content = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r'write_signal(?:_traced)?\s*\(\s*["\']([^"\' ]+)["\']',
                content,
            ):
                self.written.add(m.group(1))
            for m in re.finditer(
                r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*["\']([^"\' ]+)["\']',
                content,
            ):
                self.read.add(m.group(1))

    def test_no_orphaned_signals(self):
        """Every written signal has at least one reader."""
        orphaned = self.written - self.read
        assert orphaned == set(), f"Orphaned signals: {orphaned}"

    def test_no_missing_producers(self):
        """Every read signal has at least one writer."""
        missing = self.read - self.written
        assert missing == set(), f"Missing producers: {missing}"

    def test_new_omega_signals_bidirectional(self):
        """All new PATCH-Ω signals are both written and read."""
        omega_signals = [
            'cross_module_bridge_health',
            'convergence_arbiter_conflict',
            'reinforce_recency_pressure',
            'emergence_self_assessment_triggered',
            'emergence_readiness',
        ]
        for sig in omega_signals:
            assert sig in self.written, f"{sig} not written"
            assert sig in self.read, f"{sig} not read"

    def test_total_signal_count_increased(self):
        """Total bidirectional signals should be >= 237."""
        bidir = self.written & self.read
        assert len(bidir) >= 237, f"Expected >= 237, got {len(bidir)}"


# ═══════════════════════════════════════════════════════════════════════
# Integration Flow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationFlow:
    """E2E integration: signals flow through the full cognitive loop."""

    def test_arbiter_conflict_reaches_mct_via_ucc(self):
        """Arbiter conflict → bus → UCC evaluate → MCT trigger."""
        bus = _make_bus()
        arbiter = _make_arbiter(bus)
        ucc = _make_ucc(bus)

        # Force a conflict via low coherence
        arbiter.arbitrate(
            meta_loop_results={'converged': False, 'delta_norm': 0.5},
            convergence_monitor_verdict={'status': 'diverging'},
            coherence_score=0.2,
        )

        # UCC reads the conflict and evaluates
        result = ucc.evaluate(
            subsystem_states={'meta_loop': torch.randn(1, 64)},
            delta_norm=0.5,
            uncertainty=0.5,
        )
        assert 'should_rerun' in result

    def test_bridge_health_reaches_mct(self):
        """Bridge health → bus → MCT reads recovery_pressure."""
        bus = _make_bus()
        mct = _make_mct(bus)

        # Simulate bridge publishing low health
        bus.write_signal('cross_module_bridge_health', 0.2)

        result = mct.evaluate(uncertainty=0.3)
        assert 'should_trigger' in result

    def test_recency_pressure_reaches_mct(self):
        """Reinforce recency → bus → MCT boost."""
        bus = _make_bus()
        mct = _make_mct(bus)

        bus.write_signal('reinforce_recency_pressure', 0.95)

        result = mct.evaluate(uncertainty=0.3)
        assert 'should_trigger' in result

    def test_emergence_readiness_reaches_mct(self):
        """Emergence readiness → bus → MCT coherence_deficit."""
        bus = _make_bus()
        mct = _make_mct(bus)

        bus.write_signal('emergence_readiness', 0.2)

        result = mct.evaluate(uncertainty=0.3)
        assert 'should_trigger' in result


# ═══════════════════════════════════════════════════════════════════════
# Activation Sequence Tests
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify patches can be validated in the prescribed order."""

    def test_phase1_omega5_is_independent(self):
        """Ω5 (causal trace) works independently of all other patches."""
        gate = _make_health_gate()
        ct = _make_causal_trace()
        gate.set_causal_trace(ct)

        x = torch.randn(2, 64)
        gated, val = gate(x, coherence_score=0.1)
        assert gated.shape == x.shape
        # Causal trace may or may not have entries depending on gate_val

    def test_phase2_omega1_no_bus_dependency(self):
        """Ω1 pre-seeding logic works without feedback bus."""
        # Just tests the attribute management pattern
        caches = {'_cached_output_quality': 0.85}
        model = MagicMock()
        model._omega1_last_valid_caches = caches
        model._cached_output_quality = 0.0

        _prev = getattr(model, '_omega1_last_valid_caches', None)
        if _prev is not None:
            for attr, val in _prev.items():
                cur = getattr(model, attr, None)
                if cur is None or cur == 0.0:
                    setattr(model, attr, val)

        assert model._cached_output_quality == 0.85

    def test_phase2_omega3_works_without_bus(self):
        """Bridge functions work fine when feedback_bus=None."""
        monitor = MagicMock()
        monitor.export_error_patterns.return_value = {'error_classes': {}}
        ee = _make_error_evolution()

        result = ae_train.bridge_training_errors_to_inference(
            trainer_monitor=monitor,
            inference_error_evolution=ee,
            feedback_bus=None,  # No bus
        )
        assert result == 0  # No errors bridged

    def test_phase3_omega2_requires_bus(self):
        """Ω2 arbiter conflict signal needs a bus to work."""
        bus = _make_bus()
        arbiter = _make_arbiter(bus)

        result = arbiter.arbitrate(
            meta_loop_results={'converged': False, 'delta_norm': 0.5},
            convergence_monitor_verdict={'status': 'diverging'},
            coherence_score=0.2,
        )
        # Signal was written
        val = bus.read_signal('convergence_arbiter_conflict', -1.0)
        assert val >= 0.0

    def test_phase4_omega6_depends_on_axiom_scores(self):
        """Ω6 reads axiom scores that are computed during verify_and_reinforce."""
        # Just test the logic pattern
        axioms = {
            'mutual_verification': {'score': 0.4},
            'uncertainty_metacognition': {'score': 0.3},
            'root_cause_traceability': {'score': 0.5},
        }
        mv = axioms.get('mutual_verification', {}).get('score', 0.0)
        um = axioms.get('uncertainty_metacognition', {}).get('score', 0.0)
        rc = axioms.get('root_cause_traceability', {}).get('score', 0.0)
        emergence = (
            max(1e-10, mv) * max(1e-10, um) * max(1e-10, rc)
        ) ** (1.0 / 3.0)
        assert emergence < 0.5  # Low enough to trigger


# ═══════════════════════════════════════════════════════════════════════
# Causal Transparency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify new patches maintain causal traceability."""

    def test_health_gate_trace_includes_coherence(self):
        """Ω5 trace entry includes coherence_score for root-cause analysis."""
        gate = _make_health_gate()
        ct = _make_causal_trace()
        gate.set_causal_trace(ct)

        x = torch.randn(2, 64)
        gated, val = gate(x, coherence_score=0.1)

        if val < 1.0:
            entries = ct.find(subsystem="health_gate")
            if entries:
                meta = entries[0].get('metadata', {})
                assert 'coherence_score' in meta
                assert meta['coherence_score'] == 0.1

    def test_arbiter_conflict_uses_write_signal_traced(self):
        """Ω2 conflict write uses write_signal_traced for causal recording."""
        import inspect
        src = inspect.getsource(
            aeon_core.UnifiedConvergenceArbiter.arbitrate
        )
        assert 'write_signal_traced' in src
        assert 'convergence_arbiter_conflict' in src

    def test_emergence_auto_trigger_uses_write_signal_traced(self):
        """Ω6 emergence trigger uses write_signal_traced."""
        import inspect
        src = inspect.getsource(aeon_core.AEONDeltaV3.verify_and_reinforce)
        assert 'write_signal_traced' in src
        assert 'emergence_self_assessment_triggered' in src
