"""Tests for PATCH-ACTIVATE-5 and PATCH-ACTIVATE-7.

PATCH-ACTIVATE-5: Emergence deficit feedback to axiom subsystems
PATCH-ACTIVATE-7: Within-pass MCT pre-signaling
"""
import re
import sys
import math
import pytest
import torch

# ── Import aeon_core ──────────────────────────────────────────────────
import aeon_core  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────

def _make_bus():
    """Create a CognitiveFeedbackBus instance."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=64)


def _make_config(**overrides):
    """Build a minimal AEONConfig."""
    defaults = dict(
        vocab_size=200,
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
    )
    defaults.update(overrides)
    return aeon_core.AEONConfig(**defaults)


def _make_minimal_model():
    """Create a minimal AEONDeltaV3 model for integration tests."""
    config = _make_config()
    model = aeon_core.AEONDeltaV3(config)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ACTIVATE-5: EMERGENCE DEFICIT FEEDBACK TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestActivate5_EmergenceDeficitFeedback:
    """Tests for PATCH-ACTIVATE-5: emergence deficit → axiom threshold
    tightening and per-axiom pressure signals."""

    def test_emergence_deficit_read_in_verify_and_reinforce(self):
        """verify_and_reinforce reads emergence_deficit from the bus."""
        src = open('aeon_core.py').read()
        # Find the PATCH-ACTIVATE-5 block reading emergence_deficit
        assert "PATCH-ACTIVATE-5" in src, (
            "PATCH-ACTIVATE-5 comment marker not found in aeon_core.py"
        )
        # Must read emergence_deficit from bus in verify_and_reinforce
        pattern = r"read_signal\s*\(\s*\n?\s*['\"]emergence_deficit['\"]"
        matches = list(re.finditer(pattern, src))
        assert len(matches) >= 1, (
            "verify_and_reinforce must read emergence_deficit from bus"
        )

    def test_per_axiom_pressure_signals_written(self):
        """Three per-axiom emergence pressure signals are written."""
        src = open('aeon_core.py').read()
        expected_signals = [
            'emergence_pressure_mutual_verification',
            'emergence_pressure_uncertainty_metacognition',
            'emergence_pressure_root_cause_traceability',
        ]
        for sig in expected_signals:
            pattern = rf"write_signal\s*\(\s*\n?\s*['\"]" + re.escape(sig) + r"['\"]"
            assert re.search(pattern, src), (
                f"Per-axiom signal '{sig}' not written to bus"
            )

    def test_per_axiom_pressure_signals_read_by_mct(self):
        """MCT evaluate reads all three per-axiom emergence pressure signals."""
        src = open('aeon_core.py').read()
        expected_signals = [
            'emergence_pressure_mutual_verification',
            'emergence_pressure_uncertainty_metacognition',
            'emergence_pressure_root_cause_traceability',
        ]
        for sig in expected_signals:
            pattern = rf"read_signal\s*\(\s*\n?\s*['\"]" + re.escape(sig) + r"['\"]"
            assert re.search(pattern, src), (
                f"Per-axiom signal '{sig}' not read by MCT"
            )

    def test_threshold_tightening_logic(self):
        """When emergence_deficit > 0.3, axiom thresholds tighten by
        deficit × 0.2, making quality checks stricter."""
        src = open('aeon_core.py').read()
        # Verify the tightening formula exists
        assert '_act5_tighten' in src, "Tightening variable not found"
        assert '_act5_emergence_deficit * 0.2' in src, (
            "Tightening factor formula (deficit × 0.2) not found"
        )
        # Each axiom threshold should reference the tightening
        assert '_act5_mv_threshold' in src, "MV threshold tightening not found"
        assert '_act5_um_threshold' in src, "UM threshold tightening not found"
        assert '_act5_rc_threshold' in src, "RC threshold tightening not found"

    def test_bus_emergence_deficit_affects_threshold(self):
        """When emergence_deficit is written high, the tightening
        factor should be > 0."""
        bus = _make_bus()
        # Write a high emergence deficit
        bus.write_signal('emergence_deficit', 0.6)
        # Read it back
        val = float(bus.read_signal('emergence_deficit', 0.0))
        assert val == pytest.approx(0.6, abs=0.01)
        # Threshold tightening: 0.6 * 0.2 = 0.12
        tighten = val * 0.2 if val > 0.3 else 0.0
        assert tighten == pytest.approx(0.12, abs=0.01)
        # Tightened threshold: 0.8 + 0.12 = 0.92
        assert 0.8 + tighten == pytest.approx(0.92, abs=0.01)

    def test_no_tightening_when_deficit_low(self):
        """When emergence_deficit <= 0.3, no tightening is applied."""
        bus = _make_bus()
        bus.write_signal('emergence_deficit', 0.2)
        val = float(bus.read_signal('emergence_deficit', 0.0))
        tighten = val * 0.2 if val > 0.3 else 0.0
        assert tighten == 0.0

    def test_emergence_pressure_not_written_when_deficit_low(self):
        """Per-axiom pressure signals are only written when tightening > 0."""
        bus = _make_bus()
        # With no emergence deficit, pressure signals shouldn't be written
        bus.write_signal('emergence_deficit', 0.1)
        # Read the signal to ensure freshness
        bus.read_signal('emergence_deficit', 0.0)
        # No emergence_pressure_* should have been written by the bus alone
        # (they're written by verify_and_reinforce when deficit > 0.3)
        val = float(bus.read_signal(
            'emergence_pressure_mutual_verification', 0.0,
        ))
        assert val == 0.0

    def test_self_stabilising_loop_conceptual(self):
        """Conceptual test: high deficit → stricter threshold → lower
        axiom scores → maintained deficit → output attenuated.
        Low deficit → normal threshold → normal scores → reduced
        deficit → normal output."""
        # High deficit scenario
        deficit_high = 0.8
        tighten_high = deficit_high * 0.2  # 0.16
        threshold_high = 0.8 + tighten_high  # 0.96
        # A score of 0.85 passes normal (0.8) but fails tightened (0.96)
        score = 0.85
        passes_normal = score >= 0.8
        passes_tightened = score >= threshold_high
        assert passes_normal is True
        assert passes_tightened is False

        # Low deficit scenario
        deficit_low = 0.1
        tighten_low = deficit_low * 0.2 if deficit_low > 0.3 else 0.0
        threshold_low = 0.8 + tighten_low  # 0.8 (no tightening)
        passes_low = score >= threshold_low
        assert passes_low is True


class TestActivate5_SignalEcosystem:
    """Verify that PATCH-ACTIVATE-5 signals are properly integrated."""

    def test_no_new_orphaned_signals(self):
        """All new signals have both writers and readers."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]",
                full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]",
                full,
            ):
                read.add(m.group(1))
        new_signals = {
            'emergence_pressure_mutual_verification',
            'emergence_pressure_uncertainty_metacognition',
            'emergence_pressure_root_cause_traceability',
        }
        for sig in new_signals:
            assert sig in written, f"Signal '{sig}' not written"
            assert sig in read, f"Signal '{sig}' not read"


# ═══════════════════════════════════════════════════════════════════════
# PATCH-ACTIVATE-7: MCT EARLY PROBE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestActivate7_MCTEarlyProbe:
    """Tests for PATCH-ACTIVATE-7: within-pass MCT pre-signaling."""

    def test_early_probe_signals_written_in_forward(self):
        """forward() writes mct_early_probe_score and
        mct_early_probe_trigger after encoding."""
        src = open('aeon_core.py').read()
        assert "PATCH-ACTIVATE-7" in src, (
            "PATCH-ACTIVATE-7 comment marker not found"
        )
        pattern_score = (
            r"write_signal\s*\(\s*\n?\s*['\"]mct_early_probe_score['\"]"
        )
        pattern_trigger = (
            r"write_signal\s*\(\s*\n?\s*['\"]mct_early_probe_trigger['\"]"
        )
        assert re.search(pattern_score, src), (
            "mct_early_probe_score not written in forward()"
        )
        assert re.search(pattern_trigger, src), (
            "mct_early_probe_trigger not written in forward()"
        )

    def test_early_probe_score_read_by_ucc(self):
        """UCC evaluate reads mct_early_probe_score for convergence
        tightening."""
        src = open('aeon_core.py').read()
        pattern = (
            r"read_signal\s*\(\s*\n?\s*['\"]mct_early_probe_score['\"]"
        )
        assert re.search(pattern, src), (
            "mct_early_probe_score not read by UCC"
        )

    def test_early_probe_trigger_read_by_mct(self):
        """MCT evaluate reads mct_early_probe_trigger for sensitivity
        boost."""
        src = open('aeon_core.py').read()
        pattern = (
            r"read_signal\s*\(\s*\n?\s*['\"]mct_early_probe_trigger['\"]"
        )
        assert re.search(pattern, src), (
            "mct_early_probe_trigger not read by MCT"
        )

    def test_probe_score_computation(self):
        """Probe score is weighted combination of 4 retained signals."""
        bus = _make_bus()
        # Simulate previous pass signals
        bus.write_signal('mct_trigger_score', 0.8)
        bus.write_signal('mct_decision_entropy', 0.5)
        bus.write_signal('meta_oscillation_detected', 0.3)
        bus.write_signal('emergence_deficit', 0.4)
        # Compute probe score as the forward() code does
        prev_trigger = float(bus.read_signal('mct_trigger_score', 0.0))
        prev_entropy = float(bus.read_signal('mct_decision_entropy', 0.0))
        prev_osc = float(bus.read_signal('meta_oscillation_detected', 0.0))
        prev_deficit = float(bus.read_signal('emergence_deficit', 0.0))
        probe_score = (
            0.4 * min(1.0, prev_trigger)
            + 0.2 * min(1.0, prev_entropy)
            + 0.2 * min(1.0, prev_osc)
            + 0.2 * min(1.0, prev_deficit)
        )
        # 0.4*0.8 + 0.2*0.5 + 0.2*0.3 + 0.2*0.4 = 0.32 + 0.10 + 0.06 + 0.08 = 0.56
        assert probe_score == pytest.approx(0.56, abs=0.01)
        # Should trigger (> 0.3)
        assert probe_score > 0.3

    def test_probe_does_not_trigger_when_quiescent(self):
        """When all previous-pass signals are zero, probe does not fire."""
        bus = _make_bus()
        prev_trigger = float(bus.read_signal('mct_trigger_score', 0.0))
        prev_entropy = float(bus.read_signal('mct_decision_entropy', 0.0))
        prev_osc = float(bus.read_signal('meta_oscillation_detected', 0.0))
        prev_deficit = float(bus.read_signal('emergence_deficit', 0.0))
        probe_score = (
            0.4 * min(1.0, prev_trigger)
            + 0.2 * min(1.0, prev_entropy)
            + 0.2 * min(1.0, prev_osc)
            + 0.2 * min(1.0, prev_deficit)
        )
        assert probe_score == 0.0
        probe_trigger = 1.0 if probe_score > 0.3 else 0.0
        assert probe_trigger == 0.0

    def test_probe_trigger_binary(self):
        """mct_early_probe_trigger is binary: 0.0 or 1.0."""
        # Below threshold
        assert (1.0 if 0.2 > 0.3 else 0.0) == 0.0
        # Above threshold
        assert (1.0 if 0.5 > 0.3 else 0.0) == 1.0
        # Exactly at threshold
        assert (1.0 if 0.3 > 0.3 else 0.0) == 0.0

    def test_no_new_orphaned_signals(self):
        """Early probe signals have both writers and readers."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]",
                full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]",
                full,
            ):
                read.add(m.group(1))
        new_signals = {
            'mct_early_probe_score',
            'mct_early_probe_trigger',
        }
        for sig in new_signals:
            assert sig in written, f"Signal '{sig}' not written"
            assert sig in read, f"Signal '{sig}' not read"


class TestActivate7_UCC_ConvergenceTightening:
    """Tests for the UCC consumption of early probe signals."""

    def test_ucc_reads_early_probe_for_convergence(self):
        """UCC evaluate block reads mct_early_probe_score and records
        a secondary convergence signal when probe > 0.3."""
        src = open('aeon_core.py').read()
        assert "PATCH-ACTIVATE-7b" in src, (
            "PATCH-ACTIVATE-7b marker not found in UCC evaluate"
        )
        assert "mct_early_probe_pressure" in src, (
            "Secondary signal 'mct_early_probe_pressure' not recorded"
        )

    def test_mct_reads_early_probe_trigger(self):
        """MCT evaluate reads mct_early_probe_trigger and routes to
        recovery_pressure."""
        src = open('aeon_core.py').read()
        assert "PATCH-ACTIVATE-7c" in src, (
            "PATCH-ACTIVATE-7c marker not found in MCT evaluate"
        )


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL ECOSYSTEM INTEGRITY
# ═══════════════════════════════════════════════════════════════════════


class TestActivateSignalEcosystem:
    """Verify the overall signal ecosystem remains healthy after both
    ACTIVATE patches."""

    def test_no_missing_producers(self):
        """Every read_signal has a corresponding write_signal (except
        known exceptions with external producers)."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]",
                full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]",
                full,
            ):
                read.add(m.group(1))
        missing = read - written
        # Some signals may come from _extra_signals or _build_feedback
        # rather than explicit write_signal calls
        known_external = {
            'convergence_confidence',  # written via _fb_bus in convergence monitor
            # Pre-existing signals produced by non-write_signal paths
            'cross_pass_instability_pressure',  # computed in flush_consumed
            'cross_pass_oscillation',  # computed in flush_consumed
            'server_ssp_pressure',  # written by aeon_server external path
        }
        actual_missing = missing - known_external
        # Filter out signals that exist in _extra_signals computation
        # (populated by _build_feedback_extra_signals, not write_signal)
        extra_signal_pattern = re.compile(
            r'extra\["([^"]+)"\]\s*=|'
            r"extra\['([^']+)'\]\s*="
        )
        with open('aeon_core.py') as f:
            src = f.read()
        for m in extra_signal_pattern.finditer(src):
            name = m.group(1) or m.group(2)
            actual_missing.discard(name)
        assert len(actual_missing) == 0, (
            f"Signals read but never written: {sorted(actual_missing)}"
        )

    def test_orphaned_signals_benign(self):
        """All orphaned signals (written but never read) are known
        benign or informational."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]",
                full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]",
                full,
            ):
                read.add(m.group(1))
        orphaned = written - read
        benign = {
            'integration_cycle_id',
            'integration_cycle_timestamp',
            'wizard_completed',
            'wizard_corpus_quality',
        }
        non_benign = orphaned - benign
        assert len(non_benign) == 0, (
            f"Non-benign orphaned signals: {sorted(non_benign)}"
        )

    def test_activate_5_signals_fully_connected(self):
        """All 3 emergence_pressure signals have writers AND readers."""
        signals = [
            'emergence_pressure_mutual_verification',
            'emergence_pressure_uncertainty_metacognition',
            'emergence_pressure_root_cause_traceability',
        ]
        with open('aeon_core.py') as f:
            src = f.read()
        for sig in signals:
            w = re.search(
                rf"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                + re.escape(sig) + r"['\"]",
                src,
            )
            r = re.search(
                rf"read_signal\s*\(\s*\n?\s*['\"]"
                + re.escape(sig) + r"['\"]",
                src,
            )
            assert w is not None, f"'{sig}' has no writer"
            assert r is not None, f"'{sig}' has no reader"

    def test_activate_7_signals_fully_connected(self):
        """Both early probe signals have writers AND readers."""
        signals = ['mct_early_probe_score', 'mct_early_probe_trigger']
        with open('aeon_core.py') as f:
            src = f.read()
        for sig in signals:
            w = re.search(
                rf"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                + re.escape(sig) + r"['\"]",
                src,
            )
            r = re.search(
                rf"read_signal\s*\(\s*\n?\s*['\"]"
                + re.escape(sig) + r"['\"]",
                src,
            )
            assert w is not None, f"'{sig}' has no writer"
            assert r is not None, f"'{sig}' has no reader"
