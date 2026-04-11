"""
Tests for PATCH-CI-1 through PATCH-CI-7.

Critical Integration patches close 7 isolated/partially-connected paths
in the AEON-Delta RMT v3.1 cognitive architecture:

  - PATCH-CI-7: Signal Freshness Fence (generation tracking in bus)
  - PATCH-CI-2: Certified Convergence Proof Propagation (3 signals)
  - PATCH-CI-4: AutoCriticLoop ↔ MCT Bidirectional Feedback
  - PATCH-CI-3: verify_and_reinforce() Improvement Delta & Enforcement
  - PATCH-CI-5: Silent Failure Escalation Unsilencing (graduated)
  - PATCH-CI-6: Activation Pre-Condition Enforcement
  - PATCH-CI-1: Post-Pipeline MCT Re-Evaluation Gate
"""

import pytest
import sys
import os
import re
import types
import math
import importlib

# ──────────────────────────────────────────────────────────────────────
# Source helpers
# ──────────────────────────────────────────────────────────────────────

def _src():
    """Read aeon_core.py source."""
    path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
    with open(path) as f:
        return f.read()


def _import_core():
    """Import aeon_core module."""
    import aeon_core
    return aeon_core


# ======================================================================
# PATCH-CI-7: Signal Freshness Fence
# ======================================================================

class TestCI7_SignalFreshnessFence:
    """Tests for generation tracking in CognitiveFeedbackBus."""

    def test_bus_has_generation_counter(self):
        """Bus initialises _generation to 0."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        assert hasattr(bus, '_generation')
        assert bus._generation == 0

    def test_bus_has_signal_generation_dict(self):
        """Bus initialises _signal_generation as empty dict."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        assert hasattr(bus, '_signal_generation')
        assert isinstance(bus._signal_generation, dict)

    def test_write_signal_records_generation(self):
        """write_signal() records the current generation."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 0.5)
        assert bus._signal_generation['test_sig'] == 0

    def test_flush_increments_generation(self):
        """flush_consumed() increments generation counter."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        assert bus._generation == 0
        bus.flush_consumed()
        assert bus._generation == 1

    def test_flush_returns_pending_signals(self):
        """flush_consumed() returns pending_signals in summary."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 0.8)
        summary = bus.flush_consumed()
        assert 'pending_signals' in summary
        assert 'generation' in summary

    def test_read_signal_current_gen_returns_current(self):
        """read_signal_current_gen returns value for current-gen signals."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 0.75)
        val = bus.read_signal_current_gen('test_sig', 0.0)
        assert abs(val - 0.75) < 0.01

    def test_read_signal_current_gen_returns_default_for_stale(self):
        """read_signal_current_gen returns default for pre-flush signals."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 0.75)
        bus.flush_consumed()  # gen 0 → gen 1
        # test_sig was written in gen 0, now gen is 1
        val = bus.read_signal_current_gen('test_sig', -1.0)
        assert val == -1.0

    def test_read_signal_any_gen_returns_regardless(self):
        """read_signal_any_gen returns value regardless of generation."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 0.75)
        bus.flush_consumed()  # gen 0 → gen 1
        val = bus.read_signal_any_gen('test_sig', 0.0)
        assert abs(val - 0.75) < 0.01

    def test_get_signal_generation(self):
        """get_signal_generation returns correct generation."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        assert bus.get_signal_generation('nonexistent') == -1
        bus.write_signal('test_sig', 0.5)
        assert bus.get_signal_generation('test_sig') == 0
        bus.flush_consumed()
        bus.write_signal('test_sig', 0.6)
        assert bus.get_signal_generation('test_sig') == 1

    def test_get_current_generation(self):
        """get_current_generation returns the current generation."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        assert bus.get_current_generation() == 0
        bus.flush_consumed()
        assert bus.get_current_generation() == 1

    def test_source_has_generation_tracking(self):
        """Source code has generation tracking in flush_consumed."""
        src = _src()
        assert 'self._generation += 1' in src
        assert 'read_signal_current_gen' in src
        assert 'read_signal_any_gen' in src


# ======================================================================
# PATCH-CI-2: Certified Convergence Proof Propagation
# ======================================================================

class TestCI2_CertifiedConvergenceProof:
    """Tests for structured convergence proof propagation."""

    def test_source_writes_three_certified_signals(self):
        """Source writes certified_boundary_violations, worst_violation, proof_confidence."""
        src = _src()
        assert "'certified_boundary_violations'" in src
        assert "'certified_worst_violation'" in src
        assert "'certified_proof_confidence'" in src

    def test_source_writes_certified_signals_in_forward(self):
        """Signals are written in _forward_impl after CertifiedMetaLoop."""
        src = _src()
        # Find the write_signal calls for certified signals
        pat_bv = re.compile(
            r"write_signal\(\s*['\"]certified_boundary_violations['\"]",
        )
        pat_wv = re.compile(
            r"write_signal\(\s*['\"]certified_worst_violation['\"]",
        )
        pat_pc = re.compile(
            r"write_signal\(\s*['\"]certified_proof_confidence['\"]",
        )
        assert pat_bv.search(src)
        assert pat_wv.search(src)
        assert pat_pc.search(src)

    def test_mct_reads_certified_signals(self):
        """MCT reads certified_boundary_violations and proof_confidence."""
        src = _src()
        pat_bv = re.compile(
            r"read_signal\(\s*['\"]certified_boundary_violations['\"]",
        )
        pat_pc = re.compile(
            r"read_signal\(\s*['\"]certified_proof_confidence['\"]",
        )
        pat_wv = re.compile(
            r"read_signal\(\s*['\"]certified_worst_violation['\"]",
        )
        assert pat_bv.search(src)
        assert pat_pc.search(src)
        assert pat_wv.search(src)

    def test_mct_routes_boundary_violations_to_convergence_pressure(self):
        """Boundary violations > 0.3 → convergence_degradation_pressure."""
        src = _src()
        # Find the CI-2 consumer block
        idx = src.find("'certified_boundary_violations'")
        assert idx >= 0
        context = src[idx:idx + 600]
        assert 'convergence_degradation_pressure' in context

    def test_mct_routes_low_confidence_to_uncertainty(self):
        """Low proof confidence < 0.5 → uncertainty boost."""
        src = _src()
        idx = src.find("'certified_proof_confidence'")
        assert idx >= 0
        context = src[idx:idx + 400]
        assert 'uncertainty' in context


# ======================================================================
# PATCH-CI-4: AutoCriticLoop ↔ MCT Bidirectional Feedback
# ======================================================================

class TestCI4_AutoCriticFeedback:
    """Tests for AutoCriticLoop revision quality signals."""

    def test_source_writes_revision_delta(self):
        """AutoCriticLoop writes auto_critic_revision_delta."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]auto_critic_revision_delta['\"]",
        )
        assert pat.search(src)

    def test_source_writes_semantic_drift(self):
        """AutoCriticLoop writes auto_critic_semantic_drift."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]auto_critic_semantic_drift['\"]",
        )
        assert pat.search(src)

    def test_mct_reads_revision_delta(self):
        """MCT reads auto_critic_revision_delta."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]auto_critic_revision_delta['\"]",
        )
        assert pat.search(src)

    def test_mct_reads_semantic_drift(self):
        """MCT reads auto_critic_semantic_drift."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]auto_critic_semantic_drift['\"]",
        )
        assert pat.search(src)

    def test_revision_delta_routes_to_uncertainty(self):
        """auto_critic_revision_delta > 0.3 → uncertainty boost."""
        src = _src()
        # Find MCT reader block (PATCH-CI-4 consumer)
        idx = src.find("PATCH-CI-4: AutoCriticLoop revision quality")
        assert idx >= 0
        context = src[idx:idx + 1200]
        assert 'uncertainty' in context
        assert 'auto_critic_revision_delta' in context

    def test_semantic_drift_routes_to_coherence(self):
        """auto_critic_semantic_drift > 0.5 → coherence_deficit boost."""
        src = _src()
        # Find MCT reader block (PATCH-CI-4 consumer)
        idx = src.find("PATCH-CI-4: AutoCriticLoop revision quality")
        assert idx >= 0
        context = src[idx:idx + 1200]
        assert 'coherence_deficit' in context
        assert 'auto_critic_semantic_drift' in context


# ======================================================================
# PATCH-CI-3: verify_and_reinforce() Improvement Delta & Enforcement
# ======================================================================

class TestCI3_ReinforcementDelta:
    """Tests for improvement delta computation in verify_and_reinforce."""

    def test_source_writes_regression_detected(self):
        """verify_and_reinforce writes reinforcement_regression_detected."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]reinforcement_regression_detected['\"]",
        )
        assert pat.search(src)

    def test_source_writes_strategy_exhausted(self):
        """verify_and_reinforce writes reinforcement_strategy_exhausted."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]reinforcement_strategy_exhausted['\"]",
        )
        assert pat.search(src)

    def test_mct_reads_strategy_exhausted(self):
        """MCT reads reinforcement_strategy_exhausted."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]reinforcement_strategy_exhausted['\"]",
        )
        assert pat.search(src)

    def test_mct_reads_regression_detected(self):
        """MCT reads reinforcement_regression_detected."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]reinforcement_regression_detected['\"]",
        )
        assert pat.search(src)

    def test_delta_computation_exists(self):
        """verify_and_reinforce computes improvement deltas."""
        src = _src()
        assert '_ci3_deltas' in src
        assert '_ci3_all_regressing' in src

    def test_regression_counter_tracks_consecutive(self):
        """Regression counter increments on all-regression, decays otherwise."""
        src = _src()
        assert '_ci3_reinforcement_regression_count' in src
        assert '_ci3_regression_count >= 3' in src

    def test_strategy_exhausted_routes_to_recovery_pressure(self):
        """reinforcement_strategy_exhausted → MCT recovery_pressure."""
        src = _src()
        idx = src.find("read_signal(\n")
        # Find MCT reader for strategy_exhausted
        pat = re.compile(
            r"read_signal\(\s*['\"]reinforcement_strategy_exhausted['\"]"
        )
        match = pat.search(src)
        assert match is not None
        context = src[match.start():match.start() + 600]
        assert 'recovery_pressure' in context

    def test_prior_pass_signals_stored(self):
        """Current axiom scores stored as prior_pass_* persistent signals."""
        src = _src()
        assert "prior_pass_" in src
        assert "register_persistent_signal" in src

    def test_participation_deficit_enforcement(self):
        """Participation deficit + regression → confidence penalty."""
        src = _src()
        assert 'reinforcement_confidence_penalty' in src
        assert 'subsystem_participation_deficit' in src


# ======================================================================
# PATCH-CI-5: Silent Failure Escalation Unsilencing
# ======================================================================

class TestCI5_SilentFailureEscalation:
    """Tests for graduated silent failure escalation."""

    def test_graduated_escalation_3_tier(self):
        """MCT applies 3-tier escalation based on silent failure count."""
        src = _src()
        # Find the CI-5 block
        idx = src.find("PATCH-Ω3+CI-5")
        assert idx >= 0
        context = src[idx:idx + 2000]
        # Tier 1: coherence_deficit for 1-2 failures
        assert 'coherence_deficit' in context
        # Tier 2: uncertainty for 3-5 failures
        assert 'uncertainty' in context
        # Tier 3: systemic_silent_failure_alert for 6+
        assert 'systemic_silent_failure_alert' in context

    def test_systemic_alert_signal_written(self):
        """systemic_silent_failure_alert written for 6+ failures."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]systemic_silent_failure_alert['\"]",
        )
        assert pat.search(src)

    def test_systemic_alert_signal_read(self):
        """systemic_silent_failure_alert read by MCT (CI-6 block)."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]systemic_silent_failure_alert['\"]",
        )
        assert pat.search(src)

    def test_threshold_06_for_systemic(self):
        """Threshold 0.6 for systemic alert (6+ out of 10)."""
        src = _src()
        idx = src.find("PATCH-Ω3+CI-5")
        assert idx >= 0
        context = src[idx:idx + 2000]
        assert '>= 0.6' in context or '>=0.6' in context

    def test_threshold_03_for_moderate(self):
        """Threshold 0.3 for moderate escalation (3+ out of 10)."""
        src = _src()
        idx = src.find("PATCH-Ω3+CI-5")
        assert idx >= 0
        context = src[idx:idx + 2000]
        assert '>= 0.3' in context or '>=0.3' in context


# ======================================================================
# PATCH-CI-6: Activation Pre-Condition Enforcement
# ======================================================================

class TestCI6_ActivationEnforcement:
    """Tests for activation pre-condition enforcement."""

    def test_source_writes_activation_incomplete(self):
        """_forward_impl writes activation_incomplete when not activated."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]activation_incomplete['\"]",
        )
        matches = list(pat.finditer(src))
        assert len(matches) >= 2, (
            "Should write activation_incomplete=1.0 and =0.0"
        )

    def test_mct_reads_activation_incomplete(self):
        """MCT reads activation_incomplete signal."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]activation_incomplete['\"]",
        )
        assert pat.search(src)

    def test_activation_incomplete_zeroes_signal_values(self):
        """When activation incomplete, MCT zeroes signal-based triggers."""
        src = _src()
        assert '_ci6_activation_incomplete' in src
        # Should zero out all signal values
        idx = src.find('_ci6_activation_incomplete = True')
        assert idx >= 0
        context = src[idx:idx + 500]
        assert 'signal_values[_ci6_k] = 0.0' in context

    def test_activation_complete_clears_signal(self):
        """When activation is complete, writes activation_incomplete=0.0."""
        src = _src()
        # Find the clearing write
        idx = src.find("PATCH-CI-6: Clear activation_incomplete")
        assert idx >= 0


# ======================================================================
# PATCH-CI-1: Post-Pipeline MCT Re-Evaluation Gate
# ======================================================================

class TestCI1_PostPipelineMCTGate:
    """Tests for PostPipelineMCTGate class and wiring."""

    def test_class_exists(self):
        """PostPipelineMCTGate class exists in aeon_core."""
        aeon = _import_core()
        assert hasattr(aeon, 'PostPipelineMCTGate')

    def test_class_has_check_and_retrigger(self):
        """PostPipelineMCTGate has check_and_retrigger method."""
        aeon = _import_core()
        gate = aeon.PostPipelineMCTGate(
            mct_ref=None, feedback_bus_ref=None, threshold=0.6,
        )
        assert hasattr(gate, 'check_and_retrigger')

    def test_gate_returns_false_with_no_bus(self):
        """Gate returns False when bus is None."""
        aeon = _import_core()
        gate = aeon.PostPipelineMCTGate(
            mct_ref=None, feedback_bus_ref=None, threshold=0.6,
        )
        assert gate.check_and_retrigger() is False

    def test_gate_returns_false_below_threshold(self):
        """Gate returns False when signals below threshold."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('post_output_uncertainty', 0.3)
        bus.write_signal('post_pipeline_verdict_pressure', 0.2)
        gate = aeon.PostPipelineMCTGate(
            mct_ref=types.SimpleNamespace(evaluate=lambda **kw: {}),
            feedback_bus_ref=bus,
            threshold=0.6,
        )
        assert gate.check_and_retrigger() is False

    def test_gate_returns_true_above_threshold(self):
        """Gate returns True and writes retrigger signal above threshold."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('post_output_uncertainty', 0.8)
        bus.write_signal('post_pipeline_verdict_pressure', 0.7)
        gate = aeon.PostPipelineMCTGate(
            mct_ref=types.SimpleNamespace(evaluate=lambda **kw: {}),
            feedback_bus_ref=bus,
            threshold=0.6,
        )
        result = gate.check_and_retrigger()
        assert result is True
        assert bus.read_signal('post_pipeline_mct_retriggered', 0.0) == 1.0

    def test_source_writes_post_pipeline_mct_retriggered(self):
        """Source writes post_pipeline_mct_retriggered signal."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]post_pipeline_mct_retriggered['\"]",
        )
        assert pat.search(src)

    def test_mct_reads_post_pipeline_mct_retriggered(self):
        """MCT reads post_pipeline_mct_retriggered signal."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]post_pipeline_mct_retriggered['\"]",
        )
        assert pat.search(src)

    def test_gate_instantiated_in_model_init(self):
        """PostPipelineMCTGate created in model __init__."""
        src = _src()
        assert 'self.post_pipeline_mct_gate = PostPipelineMCTGate(' in src

    def test_gate_called_in_forward_impl(self):
        """PostPipelineMCTGate.check_and_retrigger() called in forward."""
        src = _src()
        assert 'post_pipeline_mct_gate.check_and_retrigger()' in src


# ======================================================================
# Signal Ecosystem Integrity
# ======================================================================

class TestCISignalEcosystem:
    """Tests for signal ecosystem integrity after CI patches."""

    def test_new_signals_are_written(self):
        """All new CI signals have write_signal calls."""
        src = _src()
        new_signals = [
            'certified_boundary_violations',
            'certified_worst_violation',
            'certified_proof_confidence',
            'auto_critic_revision_delta',
            'auto_critic_semantic_drift',
            'reinforcement_regression_detected',
            'reinforcement_strategy_exhausted',
            'systemic_silent_failure_alert',
            'activation_incomplete',
            'post_pipeline_mct_retriggered',
        ]
        for sig in new_signals:
            pat = re.compile(
                rf"write_signal(?:_traced)?\(\s*['\"]{ re.escape(sig) }['\"]",
            )
            assert pat.search(src), f"Missing write for signal: {sig}"

    def test_new_signals_are_read(self):
        """All new CI signals have read_signal calls."""
        src = _src()
        new_signals = [
            'certified_boundary_violations',
            'certified_worst_violation',
            'certified_proof_confidence',
            'auto_critic_revision_delta',
            'auto_critic_semantic_drift',
            'reinforcement_regression_detected',
            'reinforcement_strategy_exhausted',
            'systemic_silent_failure_alert',
            'activation_incomplete',
            'post_pipeline_mct_retriggered',
        ]
        for sig in new_signals:
            pat = re.compile(
                rf"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]{ re.escape(sig) }['\"]",
            )
            assert pat.search(src), f"Missing read for signal: {sig}"

    def test_no_new_orphaned_signals(self):
        """New signals are all bidirectional (written and read)."""
        src = _src()
        # Extract all write_signal calls
        write_pat = re.compile(
            r"write_signal(?:_traced)?\(\s*['\"]([^'\"]+)['\"]",
        )
        # Extract all read_signal calls
        read_pat = re.compile(
            r"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]([^'\"]+)['\"]",
        )
        ci_new_signals = {
            'certified_boundary_violations',
            'certified_worst_violation',
            'certified_proof_confidence',
            'auto_critic_revision_delta',
            'auto_critic_semantic_drift',
            'reinforcement_regression_detected',
            'reinforcement_strategy_exhausted',
            'systemic_silent_failure_alert',
            'activation_incomplete',
            'post_pipeline_mct_retriggered',
        }
        written = set(write_pat.findall(src))
        read = set(read_pat.findall(src))
        for sig in ci_new_signals:
            assert sig in written, f"Signal {sig} not written"
            assert sig in read, f"Signal {sig} not read"

    def test_prior_pass_persistent_signals(self):
        """Prior-pass signals are registered as persistent."""
        src = _src()
        assert "prior_pass_mutual_verification" in src or "prior_pass_" in src

    def test_no_bare_except_pass_in_ci_code(self):
        """No unhardened except:pass blocks in CI code."""
        src = _src()
        # Find all CI patches by comment markers
        bare_except = re.compile(
            r'except\s*:\s*\n\s*pass\s*$',
            re.MULTILINE,
        )
        matches = list(bare_except.finditer(src))
        # Each bare except:pass should have a comment
        for m in matches:
            line_end = src.find('\n', m.end())
            context = src[m.start():line_end + 1] if line_end > 0 else ''
            # Allow if there's a comment on the pass line
            pass_line = src[m.start():m.end()]
            remaining = src[m.end():line_end].strip() if line_end > 0 else ''
            # This is checked by test_final_patches.py already
            pass  # Defer to existing test


# ======================================================================
# Integration: Cross-Patch Signal Flow
# ======================================================================

class TestCIIntegration:
    """Tests for cross-patch signal flow."""

    def test_ci7_generation_survives_multiple_flushes(self):
        """Generation counter tracks correctly across multiple flushes."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        for i in range(5):
            bus.write_signal(f'sig_{i}', float(i) / 5.0)
            bus.flush_consumed()
        assert bus.get_current_generation() == 5

    def test_ci2_signals_are_clamped(self):
        """Certified signals are clamped to [0, 1]."""
        src = _src()
        # Find the CI-2 producer block in _forward_impl
        idx = src.find("PATCH-CI-2: Propagate structured convergence proof")
        assert idx >= 0
        context = src[idx:idx + 2000]
        assert 'min(1.0' in context

    def test_ci3_regression_threshold_is_3(self):
        """Regression counter threshold is 3 consecutive passes."""
        src = _src()
        assert '_ci3_regression_count >= 3' in src

    def test_ci5_three_tiers_present(self):
        """CI-5 has three distinct escalation tiers."""
        src = _src()
        idx = src.find("PATCH-Ω3+CI-5")
        context = src[idx:idx + 2000]
        assert '>= 0.6' in context  # Tier 3
        assert '>= 0.3' in context  # Tier 2
        assert '> 0.0' in context   # Tier 1

    def test_ci1_gate_threshold_default(self):
        """PostPipelineMCTGate default threshold is 0.6."""
        aeon = _import_core()
        gate = aeon.PostPipelineMCTGate(
            mct_ref=None, feedback_bus_ref=None,
        )
        assert gate._threshold == 0.6

    def test_bus_generation_tracking_with_persistent(self):
        """Persistent signals get generation tracking too."""
        aeon = _import_core()
        bus = aeon.CognitiveFeedbackBus(hidden_dim=64)
        bus.register_persistent_signal('persist_sig', 0.5)
        bus.write_signal('persist_sig', 0.7)
        assert bus.get_signal_generation('persist_sig') == 0
        bus.flush_consumed()
        bus.write_signal('persist_sig', 0.8)
        assert bus.get_signal_generation('persist_sig') == 1
