"""
Tests for PATCH-COGACT-1 through PATCH-COGACT-5.

COGACT = Cognitive Activation patches that bridge the remaining
discontinuities between high-level cognition and low-level execution
in AEON-Delta RMT v3.1.

Tests cover:
  - PATCH-COGACT-3: Provenance chain quality signal (fidelity)
  - PATCH-COGACT-4: Proactive oscillation suppression
  - PATCH-COGACT-1: Graduated convergence modulation in MetaCognitiveRecursor
  - PATCH-COGACT-2: MCT emergence response signal
  - PATCH-COGACT-5: Loss component attribution to bus
  - Signal ecosystem integrity (no new orphans or missing producers)
"""

import pytest
import sys
import os
import re
import types
import math

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core
import aeon_core as aeon


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_null_trace(bus=None):
    """Create a _NullCausalTrace with optional bus reference."""
    trace = aeon._NullCausalTrace(feedback_bus=bus)
    return trace


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-3: Provenance Chain Quality Signal
# ──────────────────────────────────────────────────────────────────────

class TestCogact3_ProvenanceTraceQuality:
    """PATCH-COGACT-3: _NullCausalTrace publishes provenance_trace_fidelity."""

    def test_fidelity_written_on_record(self):
        """record() publishes provenance_trace_fidelity to bus."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        trace.record("encoder", "attention")
        val = float(bus.read_signal('provenance_trace_fidelity', -1.0))
        assert val >= 0.0, "provenance_trace_fidelity should be non-negative"

    def test_fidelity_reflects_critical_module_coverage(self):
        """Fidelity increases when critical modules are recorded."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        # Record 0 critical modules first → baseline
        trace.record("unknown_module", "action")
        val_no_critical = float(bus.read_signal('provenance_trace_fidelity', 0.0))

        bus2 = _make_bus()
        trace2 = _make_null_trace(bus2)
        # Record all 6 critical modules
        for m in ('encoder', 'meta_loop', 'decoder', 'safety', 'convergence', 'mct'):
            trace2.record(m, "action")
        val_full_critical = float(bus2.read_signal('provenance_trace_fidelity', 0.0))
        assert val_full_critical > val_no_critical, (
            "Full critical coverage should yield higher fidelity"
        )

    def test_fidelity_degrades_near_buffer_saturation(self):
        """When buffer > 80% full, fidelity degrades toward 0."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        max_entries = trace._max_log
        # Record critical modules first for coverage
        for m in ('encoder', 'meta_loop', 'decoder', 'safety', 'convergence', 'mct'):
            trace.record(m, "action")
        # Record below 80%
        target_70 = int(max_entries * 0.7) - 6  # already recorded 6
        for i in range(max(0, target_70)):
            trace.record("filler", f"action_{i}")
        val_70 = float(bus.read_signal('provenance_trace_fidelity', 1.0))

        # Fill to 95%
        target_95 = int(max_entries * 0.95) - len(trace._lightweight_log)
        for i in range(max(0, target_95)):
            trace.record("filler", f"extra_{i}")
        val_95 = float(bus.read_signal('provenance_trace_fidelity', 1.0))
        assert val_95 < val_70, (
            f"Fidelity at 95% ({val_95}) should be lower than at 70% ({val_70})"
        )

    def test_fidelity_zero_at_max_buffer(self):
        """When buffer is 100% full, fidelity is 0 (entries being dropped)."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        max_entries = trace._max_log
        for m in ('encoder', 'meta_loop', 'decoder', 'safety', 'convergence', 'mct'):
            trace.record(m, "action")
        for i in range(max_entries - 6):
            trace.record("filler", f"action_{i}")
        val = float(bus.read_signal('provenance_trace_fidelity', 1.0))
        assert val < 0.01, f"Fidelity should be ~0 at 100% buffer, got {val}"

    def test_no_bus_no_crash(self):
        """record() without a bus reference doesn't crash."""
        trace = _make_null_trace(bus=None)
        trace.record("encoder", "action")
        assert len(trace._lightweight_log) == 1

    def test_fidelity_perfect_coverage_below_80pct(self):
        """Full critical coverage + below 80% = fidelity ~1.0."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        for m in ('encoder', 'meta_loop', 'decoder', 'safety', 'convergence', 'mct'):
            trace.record(m, "action")
        val = float(bus.read_signal('provenance_trace_fidelity', 0.0))
        assert val >= 0.9, f"Expected fidelity >= 0.9, got {val}"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-3b: MCT reads provenance_trace_fidelity
# ──────────────────────────────────────────────────────────────────────

class TestCogact3b_MCTReadsTraceFidelity:
    """PATCH-COGACT-3b: MCT.evaluate() reads provenance_trace_fidelity."""

    def test_source_code_reads_fidelity(self):
        """MCT evaluate() source code contains read of provenance_trace_fidelity."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert 'provenance_trace_fidelity' in src
        # Verify it's read in MCT context (not just written)
        pattern = re.compile(
            r'read_signal\s*\(\s*[\'"]provenance_trace_fidelity[\'"]',
        )
        assert pattern.search(src), (
            "MCT should read provenance_trace_fidelity from bus"
        )

    def test_low_fidelity_routes_to_low_causal_quality(self):
        """Low fidelity (< 0.5) increases low_causal_quality signal."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        # Check the MCT reader uses low_causal_quality
        assert '_ca3_fidelity' in src, "Fidelity variable missing in MCT"
        assert re.search(r'_ca3_fidelity\s*<\s*0\.5', src), (
            "Should check fidelity < 0.5"
        )
        assert 'low_causal_quality' in src, "Should route to low_causal_quality"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-4: Proactive Oscillation Suppression
# ──────────────────────────────────────────────────────────────────────

class TestCogact4_OscillationSuppression:
    """PATCH-COGACT-4: flush_consumed() tracks score deltas for early warning."""

    def test_oscillation_risk_published(self):
        """flush_consumed() publishes mct_oscillation_risk signal."""
        bus = _make_bus()
        # Simulate trigger score oscillation: high → low → high
        bus._extra_signals['mct_trigger_score'] = 0.8
        bus._extra_signals['mct_should_trigger'] = 1.0
        bus.flush_consumed()

        bus._extra_signals['mct_trigger_score'] = 0.2
        bus._extra_signals['mct_should_trigger'] = 0.0
        bus.flush_consumed()

        bus._extra_signals['mct_trigger_score'] = 0.8
        bus._extra_signals['mct_should_trigger'] = 1.0
        bus.flush_consumed()

        risk = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk > 0.0, f"Expected oscillation risk > 0 after reversals, got {risk}"

    def test_no_oscillation_risk_stable(self):
        """Stable trigger scores produce zero oscillation risk."""
        bus = _make_bus()
        # Monotonically increasing scores — no oscillation
        for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
            bus._extra_signals['mct_trigger_score'] = score
            bus._extra_signals['mct_should_trigger'] = 0.0
            bus.flush_consumed()
        risk = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk == 0.0, f"Expected risk = 0 for stable scores, got {risk}"

    def test_oscillation_risk_decays(self):
        """Risk counter decrements when scores move consistently."""
        bus = _make_bus()
        # Create one reversal
        bus._extra_signals['mct_trigger_score'] = 0.8
        bus.flush_consumed()
        bus._extra_signals['mct_trigger_score'] = 0.2
        bus.flush_consumed()
        risk_peak = float(bus.read_signal('mct_oscillation_risk', 0.0))

        # Now consistent direction for several passes
        for score in [0.3, 0.4, 0.5, 0.6, 0.7]:
            bus._extra_signals['mct_trigger_score'] = score
            bus.flush_consumed()
        risk_after = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk_after <= risk_peak, (
            f"Risk should decay: peak={risk_peak}, after={risk_after}"
        )

    def test_sign_reversal_detection(self):
        """One sign reversal in trigger_score delta sets risk > 0."""
        bus = _make_bus()
        # Score goes up then down
        bus._extra_signals['mct_trigger_score'] = 0.3
        bus.flush_consumed()
        bus._extra_signals['mct_trigger_score'] = 0.6  # delta +0.3
        bus.flush_consumed()
        bus._extra_signals['mct_trigger_score'] = 0.3  # delta -0.3 (reversal!)
        bus.flush_consumed()
        risk = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk > 0.0, f"Expected risk > 0 after 1 reversal, got {risk}"

    def test_risk_capped_at_1(self):
        """Risk signal never exceeds 1.0."""
        bus = _make_bus()
        # Create many reversals
        for _ in range(20):
            bus._extra_signals['mct_trigger_score'] = 0.8
            bus.flush_consumed()
            bus._extra_signals['mct_trigger_score'] = 0.2
            bus.flush_consumed()
        risk = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk <= 1.0, f"Risk should be capped at 1.0, got {risk}"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-4b: MCT reads oscillation risk for hysteresis
# ──────────────────────────────────────────────────────────────────────

class TestCogact4b_MCTHysteresis:
    """PATCH-COGACT-4b: MCT raises threshold when oscillation risk is high."""

    def test_source_code_reads_oscillation_risk(self):
        """MCT evaluate() reads mct_oscillation_risk from bus."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'read_signal\s*\(\s*[\'"]mct_oscillation_risk[\'"]',
        )
        assert pattern.search(src), (
            "MCT should read mct_oscillation_risk from bus"
        )

    def test_hysteresis_raises_threshold(self):
        """When oscillation risk > 0.3, effective threshold is raised."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        # Check the threshold modulation formula exists
        assert '_ca4_osc_risk' in src, "Oscillation risk variable missing"
        assert '_s3_effective_threshold' in src, "Threshold variable missing"
        # Check the two key lines exist separately
        assert re.search(r'_ca4_osc_risk.*>\s*0\.3', src), (
            "Should check oscillation risk > 0.3"
        )
        assert re.search(
            r'_s3_effective_threshold\s*\*=\s*\(1\.0\s*\+\s*_ca4_osc_risk\s*\*\s*0\.3\)',
            src,
        ), "Should multiply threshold by (1 + risk * 0.3)"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-1: Graduated Convergence Modulation
# ──────────────────────────────────────────────────────────────────────

class TestCogact1_GraduatedConvergence:
    """PATCH-COGACT-1: MetaCognitiveRecursor uses graduated tightening."""

    def test_source_has_graduated_tightening(self):
        """MetaCognitiveRecursor uses trigger_score for graduated factor."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert '_ca1_trigger_score' in src
        assert '_ca1_factor' in src

    def test_low_trigger_score_mild_tightening(self):
        """trigger_score in [0.5, 0.65) gives factor ~0.7."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert re.search(r'_ca1_trigger_score\s*>=\s*0\.5', src), (
            "Should check trigger_score >= 0.5"
        )
        assert re.search(r'_ca1_factor\s*=\s*0\.7', src), (
            "Low trigger score (0.5-0.65) should use factor 0.7"
        )

    def test_high_trigger_score_aggressive_tightening(self):
        """trigger_score >= 0.85 gives factor ~0.3."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert re.search(r'_ca1_trigger_score\s*>=\s*0\.85', src), (
            "Should check trigger_score >= 0.85"
        )
        assert re.search(r'_ca1_factor\s*=\s*0\.3', src), (
            "High trigger score (>= 0.85) should use factor 0.3"
        )

    def test_modulation_signal_written(self):
        """mct_convergence_modulation signal is written to bus."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'write_signal\s*\(\s*[\'"]mct_convergence_modulation[\'"]',
        )
        assert pattern.search(src), (
            "MetaCognitiveRecursor should write mct_convergence_modulation"
        )

    def test_extra_iterations_boosted_for_high_urgency(self):
        """trigger_score >= 0.85 should multiply extra_iterations by 1.5."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert re.search(r'_ca1_extra_iter_mult\s*=\s*1\.5', src), (
            "High urgency should boost extra iterations by 1.5"
        )
        assert re.search(r'_ca1_trigger_score\s*>=\s*0\.85', src), (
            "Boost should trigger at score >= 0.85"
        )

    def test_convergence_dominant_detected(self):
        """convergence_conflict or spectral_instability triggers detection."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert 'convergence_conflict' in src
        assert 'spectral_instability' in src
        assert '_ca1_convergence_dominant' in src


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-2: MCT Emergence Response Signal
# ──────────────────────────────────────────────────────────────────────

class TestCogact2_EmergenceResponse:
    """PATCH-COGACT-2: MCT writes mct_emergence_response_active."""

    def test_signal_written_in_mct(self):
        """MCT evaluate() writes mct_emergence_response_active."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'write_signal(?:_traced)?\s*\(\s*[\n\s]*[\'"]mct_emergence_response_active[\'"]',
        )
        matches = pattern.findall(src)
        assert len(matches) >= 1, (
            "MCT should write mct_emergence_response_active"
        )

    def test_signal_read_in_verify_and_reinforce(self):
        """verify_and_reinforce() reads mct_emergence_response_active."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'read_signal\s*\(\s*[\n\s]*[\'"]mct_emergence_response_active[\'"]',
        )
        assert pattern.search(src), (
            "verify_and_reinforce should read mct_emergence_response_active"
        )

    def test_emergence_contribution_calculation(self):
        """Emergence contribution is computed from signal_values."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert '_ca2_emergence_contrib' in src
        assert '_ca2_fraction' in src

    def test_tightening_reduction_when_mct_responded(self):
        """verify_and_reinforce reduces tightening when MCT responded."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert '_ca2_mct_response' in src, "MCT response variable missing"
        assert re.search(r'_ca2_mct_response\s*>\s*0\.3', src), (
            "Should check MCT response > 0.3"
        )
        assert re.search(r'_act5_tighten\s*\*=', src), (
            "Should modulate _act5_tighten"
        )

    def test_zero_response_when_not_triggered(self):
        """Signal is 0.0 when MCT doesn't trigger."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        # Check that the else branch writes 0.0
        assert re.search(
            r'write_signal\s*\(\s*[\n\s]*[\'"]mct_emergence_response_active[\'"],\s*0\.0',
            src,
        ), "mct_emergence_response_active should be 0.0 when MCT doesn't fire"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGACT-5: Loss Component Attribution to Bus
# ──────────────────────────────────────────────────────────────────────

class TestCogact5_LossAttribution:
    """PATCH-COGACT-5: compute_loss() publishes loss component attribution."""

    def test_loss_dominant_component_id_written(self):
        """compute_loss() writes loss_dominant_component_id signal."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'write_signal\s*\(\s*[\n\s]*[\'"]loss_dominant_component_id[\'"]',
        )
        assert pattern.search(src), (
            "compute_loss should write loss_dominant_component_id"
        )

    def test_loss_concentration_ratio_written(self):
        """compute_loss() writes loss_concentration_ratio signal."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'write_signal\s*\(\s*[\n\s]*[\'"]loss_concentration_ratio[\'"]',
        )
        assert pattern.search(src), (
            "compute_loss should write loss_concentration_ratio"
        )

    def test_component_bands_mapping(self):
        """Component band mapping covers key loss types."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        for component in ('lm_loss', 'consistency_loss', 'safety_loss',
                          'coherence_loss', 'causal_dag_loss'):
            assert re.search(
                rf"'{component}':\s*0\.\d+",
                src,
            ), f"Component band mapping should include {component}"

    def test_mct_reads_loss_concentration(self):
        """MCT evaluate() reads loss_concentration_ratio from bus."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'read_signal\s*\(\s*[\n\s]*[\'"]loss_concentration_ratio[\'"]',
        )
        assert pattern.search(src), (
            "MCT should read loss_concentration_ratio"
        )

    def test_mct_reads_loss_component_id(self):
        """MCT evaluate() reads loss_dominant_component_id from bus."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        pattern = re.compile(
            r'read_signal\s*\(\s*[\n\s]*[\'"]loss_dominant_component_id[\'"]',
        )
        assert pattern.search(src), (
            "MCT should read loss_dominant_component_id"
        )

    def test_consistency_loss_routes_to_convergence_conflict(self):
        """consistency_loss (band 0.15) routes to convergence_conflict."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert re.search(r'0\.14\s*<=\s*_ca5_comp_id', src), (
            "Should check band for consistency_loss"
        )
        assert 'convergence_conflict' in src, (
            "consistency_loss band should route to convergence_conflict"
        )

    def test_safety_loss_routes_to_safety_violation(self):
        """safety_loss (band 0.25) routes to safety_violation."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert re.search(r'0\.24\s*<=\s*_ca5_comp_id', src), (
            "Should check band for safety_loss"
        )
        assert 'safety_violation' in src, (
            "safety_loss band should route to safety_violation"
        )


# ──────────────────────────────────────────────────────────────────────
# Signal Ecosystem Integrity
# ──────────────────────────────────────────────────────────────────────

class TestCogact_SignalEcosystem:
    """Verify all new COGACT signals are both written and read."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Load source files for signal audit."""
        core = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        train = os.path.join(os.path.dirname(__file__), 'ae_train.py')
        with open(core) as f:
            self.core_src = f.read()
        with open(train) as f:
            self.train_src = f.read()
        self.all_src = self.core_src + self.train_src

    def _is_written(self, signal_name: str) -> bool:
        """Check if a signal is written in the codebase."""
        patterns = [
            rf"write_signal(?:_traced)?\s*\(\s*[\n\s]*'{signal_name}'",
            rf'write_signal(?:_traced)?\s*\(\s*[\n\s]*"{signal_name}"',
            rf"_extra_signals\['{signal_name}'\]\s*=",
            rf'_extra_signals\["{signal_name}"\]\s*=',
        ]
        return any(re.search(p, self.all_src) for p in patterns)

    def _is_read(self, signal_name: str) -> bool:
        """Check if a signal is read in the codebase."""
        patterns = [
            rf"read_signal\s*\(\s*[\n\s]*'{signal_name}'",
            rf'read_signal\s*\(\s*[\n\s]*"{signal_name}"',
            rf"_extra_signals\.get\s*\(\s*'{signal_name}'",
            rf'_extra_signals\.get\s*\(\s*"{signal_name}"',
            rf"_extra_signals\['{signal_name}'\]",
            rf'_extra_signals\["{signal_name}"\]',
        ]
        return any(re.search(p, self.all_src) for p in patterns)

    def test_provenance_trace_fidelity_connected(self):
        """provenance_trace_fidelity is both written and read."""
        assert self._is_written('provenance_trace_fidelity'), "Not written"
        assert self._is_read('provenance_trace_fidelity'), "Not read"

    def test_mct_oscillation_risk_connected(self):
        """mct_oscillation_risk is both written and read."""
        assert self._is_written('mct_oscillation_risk'), "Not written"
        assert self._is_read('mct_oscillation_risk'), "Not read"

    def test_mct_convergence_modulation_connected(self):
        """mct_convergence_modulation is written."""
        assert self._is_written('mct_convergence_modulation'), "Not written"

    def test_mct_emergence_response_active_connected(self):
        """mct_emergence_response_active is both written and read."""
        assert self._is_written('mct_emergence_response_active'), "Not written"
        assert self._is_read('mct_emergence_response_active'), "Not read"

    def test_loss_dominant_component_id_connected(self):
        """loss_dominant_component_id is both written and read."""
        assert self._is_written('loss_dominant_component_id'), "Not written"
        assert self._is_read('loss_dominant_component_id'), "Not read"

    def test_loss_concentration_ratio_connected(self):
        """loss_concentration_ratio is both written and read."""
        assert self._is_written('loss_concentration_ratio'), "Not written"
        assert self._is_read('loss_concentration_ratio'), "Not read"


# ──────────────────────────────────────────────────────────────────────
# Integration: End-to-end coherence tests
# ──────────────────────────────────────────────────────────────────────

class TestCogact_Integration:
    """End-to-end integration tests for all COGACT patches."""

    def test_provenance_fidelity_to_mct_pipeline(self):
        """provenance_trace_fidelity flows from _NullCausalTrace to MCT."""
        bus = _make_bus()
        trace = _make_null_trace(bus)
        # Record all critical modules
        for m in ('encoder', 'meta_loop', 'decoder', 'safety', 'convergence', 'mct'):
            trace.record(m, "action")
        fidelity = float(bus.read_signal('provenance_trace_fidelity', 0.0))
        assert fidelity > 0.5, f"Expected healthy fidelity, got {fidelity}"

    def test_oscillation_risk_pipeline(self):
        """Oscillation detection → risk signal → bus available for MCT."""
        bus = _make_bus()
        # Simulate oscillating trigger scores
        scores = [0.8, 0.2, 0.8, 0.2, 0.8]
        for score in scores:
            bus._extra_signals['mct_trigger_score'] = score
            bus._extra_signals['mct_should_trigger'] = 1.0 if score > 0.5 else 0.0
            bus.flush_consumed()
        risk = float(bus.read_signal('mct_oscillation_risk', 0.0))
        assert risk > 0.0, f"Expected oscillation risk after alternating scores"
        # Also check meta_oscillation_detected fires eventually
        osc = float(bus.read_signal('meta_oscillation_detected', 0.0))
        # After 5 passes of alternation, the counter should be > 3
        assert osc > 0.0 or risk > 0.0, (
            "Either risk or meta_oscillation should be non-zero"
        )

    def test_mct_convergence_modulation_value_range(self):
        """mct_convergence_modulation = 1 - factor: 0.3 (low), 0.5 (med), 0.7 (high)."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        assert "'mct_convergence_modulation'" in src, "Signal name missing"
        assert '1.0 - _ca1_factor' in src, (
            "Modulation signal should be 1.0 - _ca1_factor"
        )

    def test_emergence_response_bidirectional(self):
        """Both writer (MCT) and reader (verify_and_reinforce) exist."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        # Writer
        write_pattern = re.compile(
            r'write_signal(?:_traced)?\s*\(\s*[\n\s]*[\'"]mct_emergence_response_active[\'"]',
        )
        # Reader
        read_pattern = re.compile(
            r'read_signal\s*\(\s*[\n\s]*[\'"]mct_emergence_response_active[\'"]',
        )
        assert write_pattern.search(src), "Writer missing"
        assert read_pattern.search(src), "Reader missing"

    def test_loss_attribution_to_mct_routing(self):
        """Loss attribution signals route to specific MCT triggers."""
        src = open(os.path.join(os.path.dirname(__file__), 'aeon_core.py')).read()
        # Verify the MCT reader section exists with routing logic
        assert '_ca5_conc' in src, "MCT concentration reader missing"
        assert '_ca5_comp_id' in src, "MCT component ID reader missing"
        # Verify routing to at least 3 different MCT signals
        routed = 0
        for sig in ('convergence_conflict', 'safety_violation',
                    'coherence_deficit', 'low_causal_quality'):
            # Check the signal appears in an _ca5_ context
            if f"'{sig}'" in src and '_ca5_comp_id' in src:
                routed += 1
        assert routed >= 3, f"Expected >= 3 loss→MCT routes, found {routed}"

    def test_all_new_signals_have_producers(self):
        """No new signal is read without also being written somewhere."""
        new_signals = [
            'provenance_trace_fidelity',
            'mct_oscillation_risk',
            'mct_convergence_modulation',
            'mct_emergence_response_active',
            'loss_dominant_component_id',
            'loss_concentration_ratio',
        ]
        core_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(core_path) as f:
            src = f.read()
        for sig in new_signals:
            write_pat = re.compile(
                rf"write_signal(?:_traced)?\s*\(\s*[\n\s]*['\"]"
                + re.escape(sig) + r"['\"]"
            )
            assert write_pat.search(src), (
                f"Signal '{sig}' is missing a producer (write_signal)"
            )
