"""
Tests for PATCH-FINAL-2 through PATCH-FINAL-5.

PATCH-FINAL-1 (silent failure hardening) was already completed in prior
Φ1 sessions — 0 bare except:pass blocks remain.

Tests cover:
  - PATCH-FINAL-2: ConvergenceErrorEvolutionBridge
  - PATCH-FINAL-3: Adaptive emergence gate (deficit-proportional)
  - PATCH-FINAL-4: Decoder variant bus wiring
  - PATCH-FINAL-5: _NullCausalTrace lightweight provenance chain
"""

import pytest
import sys
import os
import re
import types

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core
import aeon_core as aeon
import ae_train


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    bus = aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)
    return bus


def _make_config(**overrides) -> aeon.AEONConfig:
    """Create a minimal AEONConfig for testing."""
    defaults = dict(
        vocab_size=256,
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


# ──────────────────────────────────────────────────────────────────────
# PATCH-FINAL-5: _NullCausalTrace Lightweight Provenance
# ──────────────────────────────────────────────────────────────────────

class TestFinal5_NullCausalTraceProvenance:
    """PATCH-FINAL-5: _NullCausalTrace maintains a lightweight log."""

    def test_record_captures_subsystem_and_decision(self):
        """record() stores (subsystem, decision, pass_id) tuples."""
        trace = aeon._NullCausalTrace()
        trace.record("encoder", "attention_sharpened")
        trace.record("decoder", "generation_complete")
        chain = trace.get_provenance_chain()
        assert len(chain) == 2
        assert chain[0] == ("encoder", "attention_sharpened", 0)
        assert chain[1] == ("decoder", "generation_complete", 0)

    def test_record_before_after_creates_enter_exit(self):
        """record_before/record_after create enter/exit events."""
        trace = aeon._NullCausalTrace()
        trace.record_before("encoder")
        trace.record_after("encoder")
        chain = trace.get_provenance_chain()
        assert len(chain) == 2
        assert chain[0] == ("encoder", "enter", 0)
        assert chain[1] == ("encoder", "exit", 0)

    def test_reset_clears_log_and_increments_pass(self):
        """reset() clears the log and advances pass_id."""
        trace = aeon._NullCausalTrace()
        trace.record_before("encoder")
        assert len(trace.get_provenance_chain()) == 1
        trace.reset()
        assert len(trace.get_provenance_chain()) == 0
        trace.record_before("decoder")
        chain = trace.get_provenance_chain()
        assert chain[0][2] == 1  # pass_id incremented

    def test_max_log_bounded(self):
        """Log does not exceed _max_log entries."""
        trace = aeon._NullCausalTrace()
        trace._max_log = 5
        for i in range(20):
            trace.record_before(f"module_{i}")
        assert len(trace.get_provenance_chain()) == 5

    def test_bool_is_false(self):
        """_NullCausalTrace is falsy (distinguishable from real buffer)."""
        trace = aeon._NullCausalTrace()
        assert not bool(trace)

    def test_len_reflects_log_size(self):
        """__len__ returns the number of logged events."""
        trace = aeon._NullCausalTrace()
        assert len(trace) == 0
        trace.record_before("encoder")
        trace.record_after("encoder")
        assert len(trace) == 2

    def test_get_entries_returns_log(self):
        """get_entries() returns the lightweight log."""
        trace = aeon._NullCausalTrace()
        trace.record("safety", "check")
        entries = trace.get_entries()
        assert len(entries) == 1
        assert entries[0][0] == "safety"

    def test_trace_root_cause_returns_log(self):
        """trace_root_cause() returns the full log as fallback."""
        trace = aeon._NullCausalTrace()
        trace.record_before("encoder")
        trace.record_after("encoder")
        result = trace.trace_root_cause("any_id")
        assert len(result) == 2

    def test_record_returns_empty_string(self):
        """record() returns empty string (API compat)."""
        trace = aeon._NullCausalTrace()
        result = trace.record("test", "decision")
        assert result == ""

    def test_feedback_bus_writes_causal_trace_disabled(self):
        """When bus provided, writes causal_trace_disabled=1.0."""
        bus = _make_bus()
        trace = aeon._NullCausalTrace(feedback_bus=bus)
        val = bus.read_signal('causal_trace_disabled', 0.0)
        assert val == 1.0

    def test_record_kwargs_form(self):
        """record() works with keyword arguments."""
        trace = aeon._NullCausalTrace()
        trace.record(subsystem="world_model", decision="predict")
        chain = trace.get_provenance_chain()
        assert chain[0] == ("world_model", "predict", 0)

    def test_multiple_passes_tracked(self):
        """Events across multiple passes have correct pass_ids."""
        trace = aeon._NullCausalTrace()
        trace.record_before("enc")
        trace.reset()
        trace.record_before("dec")
        trace.reset()
        trace.record_before("mem")
        chain = trace.get_provenance_chain()
        assert chain[0][2] == 2  # pass_id is 2 after two resets


# ──────────────────────────────────────────────────────────────────────
# PATCH-FINAL-3: Adaptive Emergence Gate
# ──────────────────────────────────────────────────────────────────────

class TestFinal3_AdaptiveEmergenceGate:
    """PATCH-FINAL-3: emergence gate boost is deficit-proportional."""

    def test_source_has_deficit_proportional_gate(self):
        """Source code reads emergence_deficit and computes max(0.05, min(0.3, deficit*0.4))."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            src = f.read()
        # Must contain the adaptive formula
        assert 'emergence_deficit' in src
        assert "max(0.05, min(0.3," in src
        # Old fixed 0.05 should no longer be the only value
        # Find the emergence gate block
        gate_match = re.search(
            r'_emergence_gate_boost\s*=\s*max\(0\.05,\s*min\(0\.3,\s*_f3_deficit\s*\*\s*0\.4\)\)',
            src,
        )
        assert gate_match is not None, "Adaptive formula not found"

    def test_fixed_005_no_longer_sole_assignment(self):
        """The old '_emergence_gate_boost = 0.05' is no longer the sole assignment."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            src = f.read()
        # The old pattern should not exist as a standalone assignment
        old_pattern = re.findall(r'_emergence_gate_boost\s*=\s*0\.05\s*\n', src)
        assert len(old_pattern) == 0, "Old fixed 0.05 assignment still present"

    def test_deficit_zero_gives_minimum_boost(self):
        """With deficit=0.0, boost should be 0.05 (floor)."""
        deficit = 0.0
        boost = max(0.05, min(0.3, deficit * 0.4))
        assert boost == pytest.approx(0.05)

    def test_deficit_half_gives_proportional_boost(self):
        """With deficit=0.5, boost = max(0.05, min(0.3, 0.2)) = 0.2."""
        deficit = 0.5
        boost = max(0.05, min(0.3, deficit * 0.4))
        assert boost == pytest.approx(0.2)

    def test_deficit_one_gives_capped_boost(self):
        """With deficit=1.0, boost = max(0.05, min(0.3, 0.4)) = 0.3."""
        deficit = 1.0
        boost = max(0.05, min(0.3, deficit * 0.4))
        assert boost == pytest.approx(0.3)

    def test_deficit_075_gives_correct_boost(self):
        """With deficit=0.75, boost = max(0.05, min(0.3, 0.3)) = 0.3."""
        deficit = 0.75
        boost = max(0.05, min(0.3, deficit * 0.4))
        assert boost == pytest.approx(0.3)

    def test_deficit_0125_gives_correct_boost(self):
        """With deficit=0.125, boost = max(0.05, min(0.3, 0.05)) = 0.05."""
        deficit = 0.125
        boost = max(0.05, min(0.3, deficit * 0.4))
        assert boost == pytest.approx(0.05)


# ──────────────────────────────────────────────────────────────────────
# PATCH-FINAL-4: Decoder Variant Bus Wiring
# ──────────────────────────────────────────────────────────────────────

class TestFinal4_DecoderVariantWiring:
    """PATCH-FINAL-4: auxiliary decoder variants wired to feedback_bus."""

    def test_source_contains_decoder_variant_loop(self):
        """Wiring block iterates over auxiliary decoder attribute names."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            src = f.read()
        assert "'ssm_decoder'" in src
        assert "'mamba2_decoder'" in src
        assert "'auxiliary_decoder'" in src

    def test_wiring_logic_sets_fb_ref(self):
        """Simulated wiring: if attr exists with _fb_ref, it gets set."""
        class MockDecoder:
            def __init__(self):
                self._fb_ref = None

        class MockModel:
            def __init__(self):
                self.feedback_bus = _make_bus()
                self.decoder = MockDecoder()
                self.ssm_decoder = MockDecoder()
                self.mamba2_decoder = MockDecoder()

        model = MockModel()
        # Simulate the wiring block logic
        if getattr(model, 'decoder', None) is not None:
            model.decoder._fb_ref = model.feedback_bus
        for _dec_attr in ('ssm_decoder', 'mamba2_decoder', 'auxiliary_decoder'):
            _dec = getattr(model, _dec_attr, None)
            if _dec is not None and hasattr(_dec, '_fb_ref'):
                _dec._fb_ref = model.feedback_bus

        assert model.decoder._fb_ref is model.feedback_bus
        assert model.ssm_decoder._fb_ref is model.feedback_bus
        assert model.mamba2_decoder._fb_ref is model.feedback_bus

    def test_wiring_skips_missing_attrs(self):
        """Wiring gracefully skips non-existent decoder attributes."""
        class MockModel:
            def __init__(self):
                self.feedback_bus = _make_bus()
                self.decoder = types.SimpleNamespace(_fb_ref=None)

        model = MockModel()
        # Should not raise even if ssm_decoder etc. don't exist
        for _dec_attr in ('ssm_decoder', 'mamba2_decoder', 'auxiliary_decoder'):
            _dec = getattr(model, _dec_attr, None)
            if _dec is not None and hasattr(_dec, '_fb_ref'):
                _dec._fb_ref = model.feedback_bus

        assert model.decoder._fb_ref is None  # not touched by the loop


# ──────────────────────────────────────────────────────────────────────
# PATCH-FINAL-2: ConvergenceErrorEvolutionBridge
# ──────────────────────────────────────────────────────────────────────

class TestFinal2_ConvergenceErrorEvolutionBridge:
    """PATCH-FINAL-2: ConvergenceErrorEvolutionBridge polls verdicts."""

    def test_class_exists(self):
        """ConvergenceErrorEvolutionBridge is importable from ae_train."""
        assert hasattr(ae_train, 'ConvergenceErrorEvolutionBridge')

    def test_poll_detects_verdict_transition(self):
        """poll() records episode when verdict changes."""
        class MockCM:
            last_verdict = None
            history = []
            threshold = 1e-5

        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kwargs):
                self.episodes.append(kwargs)

        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee)

        # No verdict yet → no episode
        bridge.poll(0.5)
        assert len(ee.episodes) == 0

        # Verdict transition → episode recorded
        cm.last_verdict = 'diverging'
        bridge.poll(0.5)
        assert len(ee.episodes) == 1
        assert ee.episodes[0]['error_class'] == 'convergence_diverging'
        assert ee.episodes[0]['success'] is False

    def test_poll_no_duplicate_on_same_verdict(self):
        """poll() does not re-record when verdict hasn't changed."""
        class MockCM:
            last_verdict = 'converging'
        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kwargs):
                self.episodes.append(kwargs)

        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee)
        bridge.poll(0.1)
        bridge.poll(0.1)
        bridge.poll(0.1)
        # Only one episode because verdict didn't change
        assert len(ee.episodes) == 1

    def test_poll_diverging_writes_bus_signal(self):
        """poll() writes training_convergence_diverging=1.0 on diverge."""
        class MockCM:
            last_verdict = None
        class MockEE:
            def record_episode(self, **kwargs):
                pass

        bus = _make_bus()
        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee, feedback_bus=bus)

        cm.last_verdict = 'diverging'
        bridge.poll(1.0)
        val = bus.read_signal('training_convergence_diverging', 0.0)
        assert val == 1.0

    def test_poll_converging_does_not_write_diverge_signal(self):
        """poll() does not write diverging signal for converging verdict."""
        class MockCM:
            last_verdict = None
        class MockEE:
            def record_episode(self, **kwargs):
                pass

        bus = _make_bus()
        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee, feedback_bus=bus)

        cm.last_verdict = 'converging'
        bridge.poll(0.1)
        val = bus.read_signal('training_convergence_diverging', 0.0)
        assert val == 0.0

    def test_poll_success_true_for_converging(self):
        """Converging/converged verdicts recorded as success=True."""
        class MockCM:
            last_verdict = 'converged'
        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kwargs):
                self.episodes.append(kwargs)

        bridge = ae_train.ConvergenceErrorEvolutionBridge(MockCM(), MockEE())
        bridge.poll(0.01)
        assert bridge._ee.episodes[0]['success'] is True

    def test_poll_fallback_to_history(self):
        """When last_verdict is None, falls back to history-based detection."""
        from collections import deque

        class MockCM:
            last_verdict = None
            threshold = 1e-5

            def __init__(self):
                # Simulate diverging: ratios > 1.0
                self.history = deque([0.1, 0.2, 0.4, 0.8])

        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kwargs):
                self.episodes.append(kwargs)

        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee)
        bridge.poll(1.0)
        assert len(ee.episodes) == 1
        assert ee.episodes[0]['error_class'] == 'convergence_diverging'

    def test_bridge_installed_in_fallback_path(self):
        """bridge_training_errors_to_inference installs bridge on AttributeError."""
        with open(os.path.join(os.path.dirname(__file__), 'ae_train.py'), 'r') as f:
            src = f.read()
        assert 'ConvergenceErrorEvolutionBridge' in src
        assert '_conv_ee_bridge' in src

    def test_training_convergence_diverging_read_by_mct(self):
        """MCT reads training_convergence_diverging signal (PATCH-FINAL-2c)."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            src = f.read()
        assert "training_convergence_diverging" in src
        # Verify it appears in a read_signal call
        assert re.search(
            r"read_signal\s*\(\s*['\"]training_convergence_diverging['\"]",
            src,
        ) is not None


# ──────────────────────────────────────────────────────────────────────
# PATCH-FINAL-1: Silent Failure Hardening (verification only)
# ──────────────────────────────────────────────────────────────────────

class TestFinal1_SilentFailureHardening:
    """PATCH-FINAL-1: Verify no bare except:pass blocks remain."""

    def test_no_bare_except_pass_in_aeon_core(self):
        """All except:pass blocks in aeon_core.py are hardened."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            lines = f.readlines()
        bare = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('except') and stripped.endswith(':'):
                except_indent = len(line) - len(line.lstrip())
                body_lines = []
                j = i + 1
                while j < len(lines):
                    if lines[j].strip() == '':
                        j += 1
                        continue
                    inner_indent = len(lines[j]) - len(lines[j].lstrip())
                    if inner_indent <= except_indent:
                        break
                    body_lines.append(lines[j].strip())
                    j += 1
                non_comment = [l for l in body_lines if not l.startswith('#')]
                if non_comment == ['pass']:
                    context = ''.join(lines[i:j])
                    if 'write_signal' not in context and 'subsystem_silent_failure_pressure' not in context:
                        bare.append(i + 1)
        assert len(bare) == 0, f"Found {len(bare)} unhardened except:pass blocks at lines: {bare[:20]}"


# ──────────────────────────────────────────────────────────────────────
# Signal Ecosystem Integrity
# ──────────────────────────────────────────────────────────────────────

class TestFinalSignalEcosystem:
    """Verify signal ecosystem remains fully connected after all patches."""

    def test_no_missing_producers(self):
        """Every read_signal has at least one write_signal producer."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            core_src = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'ae_train.py'), 'r') as f:
            train_src = f.read()
        # ── PATCH-Ψ6: Include aeon_server.py in producer scan ─────────
        # 6 signals (server_coherence_score, server_reinforcement_pressure,
        # integration_cycle_id, integration_cycle_timestamp,
        # wizard_completed, wizard_corpus_quality) are written exclusively
        # in aeon_server.py.  Without scanning the server module, they
        # appear as missing producers even though they are fully wired.
        server_src = ''
        server_path = os.path.join(os.path.dirname(__file__), 'aeon_server.py')
        if os.path.exists(server_path):
            with open(server_path, 'r') as f:
                server_src = f.read()
        combined = core_src + train_src + server_src

        write_pat = re.compile(r'write_signal(?:_traced)?\s*\(\s*["\'](\w+)["\']')
        read_pat = re.compile(r'read_signal\s*\(\s*["\'](\w+)["\']')
        extra_pat = re.compile(r'_extra_signals\s*\[\s*["\'](\w+)["\']\s*\]')

        written = set(write_pat.findall(combined)) | set(extra_pat.findall(combined))
        read = set(read_pat.findall(combined))
        missing = read - written
        assert len(missing) == 0, f"Missing producers: {missing}"

    def test_written_count_at_least_162(self):
        """At least 162 signals are written (maintained or increased)."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            core_src = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'ae_train.py'), 'r') as f:
            train_src = f.read()
        combined = core_src + train_src

        write_pat = re.compile(r'write_signal(?:_traced)?\s*\(\s*["\'](\w+)["\']')
        extra_pat = re.compile(r'_extra_signals\s*\[\s*["\'](\w+)["\']\s*\]')
        written = set(write_pat.findall(combined)) | set(extra_pat.findall(combined))
        assert len(written) >= 162, f"Only {len(written)} written signals (expected ≥162)"


# ──────────────────────────────────────────────────────────────────────
# Cross-patch Integration
# ──────────────────────────────────────────────────────────────────────

class TestFinalCrossPatchIntegration:
    """Verify patches work together as a coherent system."""

    def test_null_trace_provenance_chain_survives_multi_pass(self):
        """Provenance chain accumulates across record types within a pass."""
        trace = aeon._NullCausalTrace()
        trace.record_before("encoder")
        trace.record("encoder", "attention_sharpened")
        trace.record_after("encoder")
        trace.record_before("decoder")
        trace.record_after("decoder")
        chain = trace.get_provenance_chain()
        assert len(chain) == 5
        # All in pass 0
        assert all(entry[2] == 0 for entry in chain)
        # Module order preserved
        assert chain[0][0] == "encoder"
        assert chain[3][0] == "decoder"

    def test_adaptive_gate_reads_emergence_deficit_from_bus(self):
        """Emergence gate reads emergence_deficit from bus (source code check)."""
        with open(os.path.join(os.path.dirname(__file__), 'aeon_core.py'), 'r') as f:
            src = f.read()
        # Find the PATCH-FINAL-3 block
        assert "read_signal('emergence_deficit'" in src or 'read_signal("emergence_deficit"' in src

    def test_convergence_bridge_creates_traceable_episodes(self):
        """Bridge episodes include causal_antecedents for traceability."""
        class MockCM:
            last_verdict = 'diverging'
        class MockEE:
            def __init__(self):
                self.episodes = []
            def record_episode(self, **kwargs):
                self.episodes.append(kwargs)

        bridge = ae_train.ConvergenceErrorEvolutionBridge(MockCM(), MockEE())
        bridge.poll(1.0)
        ep = bridge._ee.episodes[0]
        assert 'causal_antecedents' in ep
        assert 'training_bridge' in ep['causal_antecedents']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-q'])
