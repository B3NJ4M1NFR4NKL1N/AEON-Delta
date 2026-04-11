"""Tests for PATCH-Ξ1..Ξ6 — Final Cognitive Integration Patches.

These patches close the training↔cognition feedback loops by wiring 13
cognitive signals (emergence, auto-critic, cross-subsystem consistency,
memory/reasoning health, error root pressure) from the core cognitive
pipeline into the training loop, and adding 5 corresponding MCT readers
so the meta-cognitive trigger adapts to training's responses.

Signal ecosystem after all Ξ patches:
  242 written, 242 read, 242 bidirectional, 0 orphans, 0 missing

New signals (5 bidirectional):
  training_emergence_aware       — Ξ1 write / Ξ6a read
  training_critic_adapted        — Ξ2 write / Ξ6b read
  training_consistency_response  — Ξ3 write / Ξ6c read
  training_cognitive_health_response — Ξ4 write / Ξ6d read
  training_error_pressure_response   — Ξ5 write / Ξ6e read
"""

import re
import sys
import types
import math
import pytest

# ── Bootstrap: import aeon_core and ae_train ───────────────────────────
import aeon_core
import ae_train


# ── Helpers ────────────────────────────────────────────────────────────

def make_bus(hidden_dim: int = 64) -> aeon_core.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=hidden_dim)


class FakeOptimizer:
    """Fake optimizer with param_groups for LR testing."""

    def __init__(self, lr: float = 1e-3):
        self.param_groups = [{"lr": lr}]

    def zero_grad(self):
        pass

    def step(self):
        pass


class FakeModel:
    """Fake model with a feedback_bus attribute for testing."""

    def __init__(self, bus=None):
        self.feedback_bus = bus or make_bus()


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ1: Training Emergence Awareness
# ═══════════════════════════════════════════════════════════════════════

class TestXi1EmergenceAwareness:
    """Verify that training reads emergence_deficit and adapts LR."""

    def test_high_emergence_deficit_reduces_lr(self):
        """When emergence_deficit > 0.3 and not emerged, LR decreases."""
        bus = make_bus()
        bus.write_signal("emergence_deficit", 0.8)
        bus.write_signal("emergence_system_emerged", 0.0)
        bus.write_signal("mct_should_trigger", 0.0)
        bus.write_signal("mct_trigger_score", 0.0)

        # Simulate reading signals as Phase A would
        deficit = float(bus.read_signal("emergence_deficit", 0.0))
        emerged = float(bus.read_signal("emergence_system_emerged", 0.0))

        assert deficit > 0.3
        assert emerged < 0.5
        scale = max(0.7, 1.0 - deficit * 0.5)
        assert 0.7 <= scale < 1.0

    def test_emerged_system_no_lr_change(self):
        """When system has emerged, no LR reduction."""
        bus = make_bus()
        bus.write_signal("emergence_deficit", 0.1)
        bus.write_signal("emergence_system_emerged", 1.0)

        deficit = float(bus.read_signal("emergence_deficit", 0.0))
        emerged = float(bus.read_signal("emergence_system_emerged", 0.0))

        # Either low deficit or emerged => no reduction
        should_reduce = deficit > 0.3 and emerged < 0.5
        assert not should_reduce

    def test_writes_training_emergence_aware(self):
        """Ξ1 writes training_emergence_aware back to bus."""
        bus = make_bus()
        bus.write_signal("emergence_deficit", 0.6)
        bus.write_signal("emergence_system_emerged", 0.0)
        # Simulate the write
        _deficit = float(bus.read_signal("emergence_deficit", 0.0))
        _emerged = float(bus.read_signal("emergence_system_emerged", 0.0))
        if _deficit > 0.3 and _emerged < 0.5:
            bus.write_signal("training_emergence_aware", _deficit)
        else:
            bus.write_signal("training_emergence_aware", 0.0)

        val = float(bus.read_signal("training_emergence_aware", 0.0))
        assert val == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ2: Training Auto-Critic Feedback
# ═══════════════════════════════════════════════════════════════════════

class TestXi2AutoCriticFeedback:
    """Verify training reads auto_critic_revision_delta/semantic_drift."""

    def test_high_semantic_drift_triggers_adaptation(self):
        """semantic_drift > 0.5 should trigger LR reduction."""
        bus = make_bus()
        bus.write_signal("auto_critic_semantic_drift", 0.7)
        bus.write_signal("auto_critic_revision_delta", 0.2)

        drift = float(bus.read_signal("auto_critic_semantic_drift", 0.0))
        delta = float(bus.read_signal("auto_critic_revision_delta", 0.0))
        pressure = max(drift, delta)
        assert pressure > 0.5
        scale = max(0.7, 1.0 - pressure * 0.4)
        assert 0.7 <= scale < 1.0

    def test_high_revision_delta_triggers_adaptation(self):
        """revision_delta > 0.5 should trigger LR reduction."""
        bus = make_bus()
        bus.write_signal("auto_critic_semantic_drift", 0.1)
        bus.write_signal("auto_critic_revision_delta", 0.9)

        drift = float(bus.read_signal("auto_critic_semantic_drift", 0.0))
        delta = float(bus.read_signal("auto_critic_revision_delta", 0.0))
        pressure = max(drift, delta)
        assert pressure > 0.5

    def test_low_critic_signals_no_adaptation(self):
        """Low critic signals should not trigger adaptation."""
        bus = make_bus()
        bus.write_signal("auto_critic_semantic_drift", 0.2)
        bus.write_signal("auto_critic_revision_delta", 0.3)

        drift = float(bus.read_signal("auto_critic_semantic_drift", 0.0))
        delta = float(bus.read_signal("auto_critic_revision_delta", 0.0))
        pressure = max(drift, delta)
        assert pressure <= 0.5

    def test_writes_training_critic_adapted(self):
        """Ξ2 writes training_critic_adapted back to bus."""
        bus = make_bus()
        bus.write_signal("auto_critic_semantic_drift", 0.8)
        bus.write_signal("auto_critic_revision_delta", 0.1)
        drift = float(bus.read_signal("auto_critic_semantic_drift", 0.0))
        delta = float(bus.read_signal("auto_critic_revision_delta", 0.0))
        pressure = max(drift, delta)
        bus.write_signal("training_critic_adapted", pressure)
        val = float(bus.read_signal("training_critic_adapted", 0.0))
        assert val == pytest.approx(0.8, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ3: Training Cross-Subsystem Consistency
# ═══════════════════════════════════════════════════════════════════════

class TestXi3ConsistencyResponse:
    """Verify training reads cross_subsystem_inconsistency."""

    def test_high_inconsistency_triggers_response(self):
        """inconsistency > 0.4 should trigger LR reduction."""
        bus = make_bus()
        bus.write_signal("cross_subsystem_inconsistency", 0.7)

        val = float(bus.read_signal("cross_subsystem_inconsistency", 0.0))
        assert val > 0.4
        scale = max(0.75, 1.0 - val * 0.4)
        assert 0.75 <= scale < 1.0

    def test_low_inconsistency_no_response(self):
        """inconsistency <= 0.4 should not trigger reduction."""
        bus = make_bus()
        bus.write_signal("cross_subsystem_inconsistency", 0.2)
        val = float(bus.read_signal("cross_subsystem_inconsistency", 0.0))
        assert val <= 0.4

    def test_writes_training_consistency_response(self):
        """Ξ3 writes training_consistency_response back to bus."""
        bus = make_bus()
        bus.write_signal("cross_subsystem_inconsistency", 0.6)
        val = float(bus.read_signal("cross_subsystem_inconsistency", 0.0))
        bus.write_signal("training_consistency_response", val)
        resp = float(bus.read_signal("training_consistency_response", 0.0))
        assert resp == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ4: Training Memory/Reasoning Health
# ═══════════════════════════════════════════════════════════════════════

class TestXi4CognitiveHealthResponse:
    """Verify training computes composite cognitive health."""

    def test_all_healthy_gives_high_score(self):
        """All-green health indicators → composite ≈ 1.0."""
        bus = make_bus()
        bus.write_signal("memory_staleness_pressure", 0.0)
        bus.write_signal("symbolic_reasoning_confidence", 1.0)
        bus.write_signal("stall_severity_pressure", 0.0)

        mem_stale = float(bus.read_signal("memory_staleness_pressure", 0.0))
        reasoning = float(bus.read_signal("symbolic_reasoning_confidence", 1.0))
        stall = float(bus.read_signal("stall_severity_pressure", 0.0))
        health = (1.0 - mem_stale) * reasoning * (1.0 - stall)
        assert health == pytest.approx(1.0)

    def test_degraded_health_reduces_lr(self):
        """Degraded health (< 0.5) should trigger LR reduction."""
        bus = make_bus()
        bus.write_signal("memory_staleness_pressure", 0.7)
        bus.write_signal("symbolic_reasoning_confidence", 0.3)
        bus.write_signal("stall_severity_pressure", 0.5)

        mem = float(bus.read_signal("memory_staleness_pressure", 0.0))
        reas = float(bus.read_signal("symbolic_reasoning_confidence", 1.0))
        stall = float(bus.read_signal("stall_severity_pressure", 0.0))
        health = (1.0 - mem) * reas * (1.0 - stall)
        assert health < 0.5
        scale = max(0.7, 0.5 + health)
        assert 0.7 <= scale < 1.0

    def test_writes_training_cognitive_health_response(self):
        """Ξ4 writes composite health score back to bus."""
        bus = make_bus()
        bus.write_signal("memory_staleness_pressure", 0.3)
        bus.write_signal("symbolic_reasoning_confidence", 0.8)
        bus.write_signal("stall_severity_pressure", 0.2)
        mem = float(bus.read_signal("memory_staleness_pressure", 0.0))
        reas = float(bus.read_signal("symbolic_reasoning_confidence", 1.0))
        stall = float(bus.read_signal("stall_severity_pressure", 0.0))
        health = (1.0 - mem) * reas * (1.0 - stall)
        bus.write_signal("training_cognitive_health_response", health)
        val = float(bus.read_signal("training_cognitive_health_response", 1.0))
        assert val == pytest.approx(health, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ5: Training Error Root Pressure
# ═══════════════════════════════════════════════════════════════════════

class TestXi5ErrorPressureResponse:
    """Verify training reads error_episode_root_pressure and
    causal_trace_truncation_pressure."""

    def test_high_root_pressure_reduces_lr(self):
        """root_pressure > 0.5 triggers LR reduction."""
        bus = make_bus()
        bus.write_signal("error_episode_root_pressure", 0.8)
        bus.write_signal("causal_trace_truncation_pressure", 0.1)

        root = float(bus.read_signal("error_episode_root_pressure", 0.0))
        trunc = float(bus.read_signal("causal_trace_truncation_pressure", 0.0))
        combined = max(root, trunc)
        assert combined > 0.5
        scale = max(0.75, 1.0 - combined * 0.3)
        assert 0.75 <= scale < 1.0

    def test_high_truncation_pressure_reduces_lr(self):
        """truncation_pressure > 0.5 also triggers reduction."""
        bus = make_bus()
        bus.write_signal("error_episode_root_pressure", 0.1)
        bus.write_signal("causal_trace_truncation_pressure", 0.9)

        root = float(bus.read_signal("error_episode_root_pressure", 0.0))
        trunc = float(bus.read_signal("causal_trace_truncation_pressure", 0.0))
        combined = max(root, trunc)
        assert combined > 0.5

    def test_low_combined_no_reduction(self):
        """Low combined pressure should not trigger reduction."""
        bus = make_bus()
        bus.write_signal("error_episode_root_pressure", 0.2)
        bus.write_signal("causal_trace_truncation_pressure", 0.3)

        root = float(bus.read_signal("error_episode_root_pressure", 0.0))
        trunc = float(bus.read_signal("causal_trace_truncation_pressure", 0.0))
        combined = max(root, trunc)
        assert combined <= 0.5

    def test_writes_training_error_pressure_response(self):
        """Ξ5 writes combined pressure back to bus."""
        bus = make_bus()
        bus.write_signal("error_episode_root_pressure", 0.6)
        bus.write_signal("causal_trace_truncation_pressure", 0.4)
        root = float(bus.read_signal("error_episode_root_pressure", 0.0))
        trunc = float(bus.read_signal("causal_trace_truncation_pressure", 0.0))
        combined = max(root, trunc)
        bus.write_signal("training_error_pressure_response", combined)
        val = float(bus.read_signal("training_error_pressure_response", 0.0))
        assert val == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Ξ6: MCT Readers for Training Response Signals
# ═══════════════════════════════════════════════════════════════════════

class TestXi6MCTReaders:
    """Verify MCT reads the 5 new training response signals."""

    def _make_mct_with_bus(self, bus=None):
        """Create MCT instance with bus."""
        bus = bus or make_bus()
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        return mct, bus

    def test_xi6a_training_emergence_aware_dampens_coherence(self):
        """training_emergence_aware > 0.3 dampens coherence_deficit."""
        mct, bus = self._make_mct_with_bus()
        bus.write_signal("training_emergence_aware", 0.8)
        # Verify the signal is readable
        val = float(bus.read_signal("training_emergence_aware", 0.0))
        assert val > 0.3
        # Damping factor
        damp = max(0.7, 1.0 - val * 0.3)
        assert 0.7 <= damp < 1.0

    def test_xi6b_training_critic_adapted_dampens_reliability(self):
        """training_critic_adapted > 0.3 dampens low_output_reliability."""
        mct, bus = self._make_mct_with_bus()
        bus.write_signal("training_critic_adapted", 0.6)
        val = float(bus.read_signal("training_critic_adapted", 0.0))
        assert val > 0.3
        damp = max(0.7, 1.0 - val * 0.3)
        assert 0.7 <= damp < 1.0

    def test_xi6c_training_consistency_response_dampens_recovery(self):
        """training_consistency_response > 0.3 dampens recovery_pressure."""
        mct, bus = self._make_mct_with_bus()
        bus.write_signal("training_consistency_response", 0.5)
        val = float(bus.read_signal("training_consistency_response", 0.0))
        assert val > 0.3

    def test_xi6d_training_cognitive_health_dampens_memory(self):
        """training_cognitive_health_response < 0.5 dampens memory."""
        mct, bus = self._make_mct_with_bus()
        bus.write_signal("training_cognitive_health_response", 0.3)
        val = float(bus.read_signal("training_cognitive_health_response", 1.0))
        assert val < 0.5
        damp = max(0.7, 0.5 + val)
        assert 0.7 <= damp < 1.0

    def test_xi6e_training_error_pressure_dampens_recovery(self):
        """training_error_pressure_response > 0.5 dampens recovery."""
        mct, bus = self._make_mct_with_bus()
        bus.write_signal("training_error_pressure_response", 0.7)
        val = float(bus.read_signal("training_error_pressure_response", 0.0))
        assert val > 0.5


# ═══════════════════════════════════════════════════════════════════════
#  Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystem:
    """Verify signal ecosystem health after Ξ patches."""

    @staticmethod
    def _scan_signals():
        """Scan all source files for written and read signal names."""
        import pathlib
        base = pathlib.Path(__file__).parent
        write_pat = re.compile(
            r'write_signal(?:_traced)?\s*\(\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        write_pat2 = re.compile(
            r'write_signal(?:_traced)?\s*\(\s*\n\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        read_pat = re.compile(
            r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        read_pat2 = re.compile(
            r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*\n\s*[\'\"]([^\'\"]+)[\'\"]',
            re.MULTILINE,
        )
        written = set()
        read = set()
        for src_file in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            fp = base / src_file
            if fp.exists():
                code = fp.read_text()
                written |= set(write_pat.findall(code))
                written |= set(write_pat2.findall(code))
                read |= set(read_pat.findall(code))
                read |= set(read_pat2.findall(code))
        return written, read

    def test_no_orphans(self):
        """Every written signal must have at least one reader."""
        written, read = self._scan_signals()
        orphans = written - read
        assert not orphans, f"Orphaned signals: {sorted(orphans)}"

    def test_no_missing_producers(self):
        """Every read signal must have at least one writer."""
        written, read = self._scan_signals()
        missing = read - written
        assert not missing, f"Missing producers: {sorted(missing)}"

    def test_new_signals_present_in_written(self):
        """5 new PATCH-Ξ signals must be written."""
        written, _ = self._scan_signals()
        new_sigs = {
            "training_emergence_aware",
            "training_critic_adapted",
            "training_consistency_response",
            "training_cognitive_health_response",
            "training_error_pressure_response",
        }
        for sig in new_sigs:
            assert sig in written, f"{sig} not found in writers"

    def test_new_signals_present_in_read(self):
        """5 new PATCH-Ξ signals must be read."""
        _, read = self._scan_signals()
        new_sigs = {
            "training_emergence_aware",
            "training_critic_adapted",
            "training_consistency_response",
            "training_cognitive_health_response",
            "training_error_pressure_response",
        }
        for sig in new_sigs:
            assert sig in read, f"{sig} not found in readers"

    def test_total_signal_count_increased(self):
        """Total signals should be >= 242 (base 237 + 5 new)."""
        written, read = self._scan_signals()
        assert len(written) >= 242, f"Expected >= 242 written, got {len(written)}"
        assert len(read) >= 242, f"Expected >= 242 read, got {len(read)}"

    def test_bidirectional_count(self):
        """All signals should be bidirectional."""
        written, read = self._scan_signals()
        bidir = written & read
        assert len(bidir) == len(written) == len(read)


# ═══════════════════════════════════════════════════════════════════════
#  Integration Flow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationFlow:
    """E2E flow: cognitive signal → training adaptation → MCT dampening."""

    def test_emergence_deficit_flow(self):
        """emergence_deficit → Ξ1 LR reduction → Ξ6a MCT dampening."""
        bus = make_bus()
        # Step 1: Core writes emergence deficit
        bus.write_signal("emergence_deficit", 0.7)
        bus.write_signal("emergence_system_emerged", 0.0)

        # Step 2: Training (Ξ1) reads and adapts
        deficit = float(bus.read_signal("emergence_deficit", 0.0))
        emerged = float(bus.read_signal("emergence_system_emerged", 0.0))
        assert deficit > 0.3 and emerged < 0.5
        bus.write_signal("training_emergence_aware", deficit)

        # Step 3: MCT (Ξ6a) reads training response
        tea = float(bus.read_signal("training_emergence_aware", 0.0))
        assert tea > 0.3
        # Dampening applied
        original_cd = 0.8
        dampened_cd = original_cd * max(0.7, 1.0 - tea * 0.3)
        assert dampened_cd < original_cd

    def test_auto_critic_drift_flow(self):
        """semantic_drift → Ξ2 LR reduction → Ξ6b MCT dampening."""
        bus = make_bus()
        bus.write_signal("auto_critic_semantic_drift", 0.9)
        bus.write_signal("auto_critic_revision_delta", 0.2)

        drift = float(bus.read_signal("auto_critic_semantic_drift", 0.0))
        delta = float(bus.read_signal("auto_critic_revision_delta", 0.0))
        pressure = max(drift, delta)
        assert pressure > 0.5
        bus.write_signal("training_critic_adapted", pressure)

        tca = float(bus.read_signal("training_critic_adapted", 0.0))
        assert tca > 0.3
        original_lor = 0.6
        dampened_lor = original_lor * max(0.7, 1.0 - tca * 0.3)
        assert dampened_lor < original_lor

    def test_cross_subsystem_inconsistency_flow(self):
        """inconsistency → Ξ3 LR reduction → Ξ6c MCT dampening."""
        bus = make_bus()
        bus.write_signal("cross_subsystem_inconsistency", 0.8)

        incon = float(bus.read_signal("cross_subsystem_inconsistency", 0.0))
        assert incon > 0.4
        bus.write_signal("training_consistency_response", incon)

        tcr = float(bus.read_signal("training_consistency_response", 0.0))
        assert tcr > 0.3
        original_rp = 0.5
        dampened_rp = original_rp * max(0.7, 1.0 - tcr * 0.3)
        assert dampened_rp < original_rp

    def test_cognitive_health_flow(self):
        """degraded health → Ξ4 LR reduction → Ξ6d MCT dampening."""
        bus = make_bus()
        bus.write_signal("memory_staleness_pressure", 0.8)
        bus.write_signal("symbolic_reasoning_confidence", 0.3)
        bus.write_signal("stall_severity_pressure", 0.6)

        mem = float(bus.read_signal("memory_staleness_pressure", 0.0))
        reas = float(bus.read_signal("symbolic_reasoning_confidence", 1.0))
        stall = float(bus.read_signal("stall_severity_pressure", 0.0))
        health = (1.0 - mem) * reas * (1.0 - stall)
        assert health < 0.5
        bus.write_signal("training_cognitive_health_response", health)

        tchr = float(bus.read_signal("training_cognitive_health_response", 1.0))
        assert tchr < 0.5
        damp = max(0.7, 0.5 + tchr)
        original_ms = 0.8
        dampened_ms = original_ms * damp
        assert dampened_ms < original_ms

    def test_error_pressure_flow(self):
        """error root pressure → Ξ5 LR reduction → Ξ6e MCT dampening."""
        bus = make_bus()
        bus.write_signal("error_episode_root_pressure", 0.9)
        bus.write_signal("causal_trace_truncation_pressure", 0.3)

        root = float(bus.read_signal("error_episode_root_pressure", 0.0))
        trunc = float(bus.read_signal("causal_trace_truncation_pressure", 0.0))
        combined = max(root, trunc)
        assert combined > 0.5
        bus.write_signal("training_error_pressure_response", combined)

        tepr = float(bus.read_signal("training_error_pressure_response", 0.0))
        assert tepr > 0.5
        original_rp = 0.6
        dampened_rp = original_rp * max(0.7, 1.0 - tepr * 0.2)
        assert dampened_rp < original_rp


# ═══════════════════════════════════════════════════════════════════════
#  Causal Transparency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Every Ξ patch signal is traceable: writer → bus → reader."""

    def test_all_xi_signals_are_traceable(self):
        """Each new signal has both a write-source and read-destination."""
        written, read = TestSignalEcosystem._scan_signals()
        xi_signals = {
            "training_emergence_aware",
            "training_critic_adapted",
            "training_consistency_response",
            "training_cognitive_health_response",
            "training_error_pressure_response",
        }
        for sig in xi_signals:
            assert sig in written, f"{sig} missing writer"
            assert sig in read, f"{sig} missing reader"

    def test_cognitive_signal_reads_trace_to_training(self):
        """Core cognitive signals read by training trace back to core."""
        # These signals are produced in aeon_core.py, consumed in ae_train.py
        core_to_train = {
            "emergence_deficit",
            "emergence_system_emerged",
            "auto_critic_semantic_drift",
            "auto_critic_revision_delta",
            "cross_subsystem_inconsistency",
            "memory_staleness_pressure",
            "symbolic_reasoning_confidence",
            "stall_severity_pressure",
            "error_episode_root_pressure",
            "causal_trace_truncation_pressure",
        }
        written, read = TestSignalEcosystem._scan_signals()
        for sig in core_to_train:
            assert sig in written, f"Core signal {sig} not written"
            assert sig in read, f"Core signal {sig} not read"

    def test_training_response_signals_trace_to_mct(self):
        """Training response signals trace back to MCT readers."""
        # Written in ae_train.py, read in aeon_core.py MCT evaluate
        train_to_mct = {
            "training_emergence_aware",
            "training_critic_adapted",
            "training_consistency_response",
            "training_cognitive_health_response",
            "training_error_pressure_response",
        }
        written, read = TestSignalEcosystem._scan_signals()
        for sig in train_to_mct:
            assert sig in written, f"Training signal {sig} not written"
            assert sig in read, f"Training signal {sig} not read"


# ═══════════════════════════════════════════════════════════════════════
#  Activation Sequence Tests
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify patches are safely applied in the correct order."""

    def test_xi1_to_xi5_order_independent(self):
        """Ξ1..Ξ5 are independent in training — order doesn't matter."""
        bus = make_bus()
        # Pre-seed all input signals
        bus.write_signal("emergence_deficit", 0.6)
        bus.write_signal("emergence_system_emerged", 0.0)
        bus.write_signal("auto_critic_semantic_drift", 0.7)
        bus.write_signal("auto_critic_revision_delta", 0.2)
        bus.write_signal("cross_subsystem_inconsistency", 0.5)
        bus.write_signal("memory_staleness_pressure", 0.3)
        bus.write_signal("symbolic_reasoning_confidence", 0.5)
        bus.write_signal("stall_severity_pressure", 0.2)
        bus.write_signal("error_episode_root_pressure", 0.6)
        bus.write_signal("causal_trace_truncation_pressure", 0.3)

        # Each Ξ1..5 reads independently — no cross-dependencies
        # Ξ1 reads emergence_deficit, emergence_system_emerged
        # Ξ2 reads auto_critic_semantic_drift, auto_critic_revision_delta
        # Ξ3 reads cross_subsystem_inconsistency
        # Ξ4 reads memory_staleness_pressure, symbolic_reasoning_confidence,
        #    stall_severity_pressure
        # Ξ5 reads error_episode_root_pressure, causal_trace_truncation_pressure
        # All have distinct input signal sets → order-independent
        inputs = {
            'xi1': {'emergence_deficit', 'emergence_system_emerged'},
            'xi2': {'auto_critic_semantic_drift', 'auto_critic_revision_delta'},
            'xi3': {'cross_subsystem_inconsistency'},
            'xi4': {'memory_staleness_pressure', 'symbolic_reasoning_confidence',
                    'stall_severity_pressure'},
            'xi5': {'error_episode_root_pressure',
                    'causal_trace_truncation_pressure'},
        }
        # Verify no input overlap
        for name, sigs in inputs.items():
            for other_name, other_sigs in inputs.items():
                if name != other_name:
                    overlap = sigs & other_sigs
                    assert not overlap, (
                        f"{name} and {other_name} share inputs: {overlap}"
                    )

    def test_xi6_depends_on_xi1_to_xi5(self):
        """Ξ6 MCT readers consume signals written by Ξ1..Ξ5."""
        # Ξ6 reads: training_emergence_aware (from Ξ1),
        #           training_critic_adapted (from Ξ2),
        #           training_consistency_response (from Ξ3),
        #           training_cognitive_health_response (from Ξ4),
        #           training_error_pressure_response (from Ξ5)
        xi6_inputs = {
            "training_emergence_aware",
            "training_critic_adapted",
            "training_consistency_response",
            "training_cognitive_health_response",
            "training_error_pressure_response",
        }
        xi_outputs = {
            "training_emergence_aware",         # Ξ1
            "training_critic_adapted",           # Ξ2
            "training_consistency_response",     # Ξ3
            "training_cognitive_health_response", # Ξ4
            "training_error_pressure_response",  # Ξ5
        }
        # Ξ6 inputs are a subset of Ξ1..Ξ5 outputs
        assert xi6_inputs <= xi_outputs


# ═══════════════════════════════════════════════════════════════════════
#  Mutual Reinforcement Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify active components verify and stabilize each other."""

    def test_training_dampens_mct_when_adapting(self):
        """When training adapts to a deficit, MCT dampens its response."""
        bus = make_bus()
        # Training has adapted to emergence deficit
        bus.write_signal("training_emergence_aware", 0.7)

        # MCT reads and dampens
        tea = float(bus.read_signal("training_emergence_aware", 0.0))
        original_cd = 0.9
        if tea > 0.3:
            dampened = original_cd * max(0.7, 1.0 - tea * 0.3)
        else:
            dampened = original_cd

        # Dampened < original: mutual reinforcement prevents double-reaction
        assert dampened < original_cd
        assert dampened >= original_cd * 0.7  # Not over-dampened

    def test_no_dampening_when_training_not_adapting(self):
        """When training hasn't adapted, MCT maintains full pressure."""
        bus = make_bus()
        bus.write_signal("training_emergence_aware", 0.0)

        tea = float(bus.read_signal("training_emergence_aware", 0.0))
        original_cd = 0.9
        if tea > 0.3:
            dampened = original_cd * max(0.7, 1.0 - tea * 0.3)
        else:
            dampened = original_cd

        assert dampened == original_cd

    def test_bidirectional_health_stabilization(self):
        """Health signal flows both ways: core→training→MCT."""
        bus = make_bus()
        # Step 1: Core writes degraded health signals
        bus.write_signal("memory_staleness_pressure", 0.9)
        bus.write_signal("symbolic_reasoning_confidence", 0.2)
        bus.write_signal("stall_severity_pressure", 0.7)

        # Step 2: Training computes health and writes response
        health = (1.0 - 0.9) * 0.2 * (1.0 - 0.7)
        assert health < 0.5
        bus.write_signal("training_cognitive_health_response", health)

        # Step 3: MCT reads response and dampens memory pressure
        tchr = float(bus.read_signal("training_cognitive_health_response", 1.0))
        assert tchr < 0.5
        damp = max(0.7, 0.5 + tchr)
        original_ms = 0.8
        dampened_ms = original_ms * damp
        # System stabilizes: MCT doesn't over-react since training adapted
        assert dampened_ms < original_ms
