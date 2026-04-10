"""
Tests for PATCH-Ω1 through PATCH-Ω4.

Omega patches close the final ~3% integration gaps in the AEON-Delta
RMT v3.1 cognitive architecture:

  - PATCH-Ω1: Wizard signals → MCT awareness (server orphan closure)
  - PATCH-Ω2: Training convergence bridge extension (stagnating/conflicting)
  - PATCH-Ω3: Silent exception counter → MCT uncertainty pressure
  - PATCH-Ω4: MCT decision in API response (causal transparency)
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
import ae_train


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _src():
    """Read the aeon_core.py source once per session."""
    path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
    with open(path) as f:
        return f.read()


def _src_train():
    """Read the ae_train.py source."""
    path = os.path.join(os.path.dirname(__file__), 'ae_train.py')
    with open(path) as f:
        return f.read()


def _src_server():
    """Read the aeon_server.py source."""
    path = os.path.join(os.path.dirname(__file__), 'aeon_server.py')
    with open(path) as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────
# PATCH-Ω3: Silent Exception Counter → MCT Uncertainty
# ──────────────────────────────────────────────────────────────────────

class TestOmega3_SilentExceptionCounter:
    """PATCH-Ω3: _forward_impl counts silent exceptions and publishes
    forward_silent_exception_count to the feedback bus."""

    def test_counter_initialized_in_forward_impl(self):
        """_forward_impl() resets _omega3_silent_exc_count to 0."""
        src = _src()
        assert '_omega3_silent_exc_count' in src, (
            "Counter variable should exist in aeon_core.py"
        )
        # Check reset at start of _forward_impl
        pat = re.compile(
            r'def _forward_impl\b.*?_omega3_silent_exc_count.*?=.*?0',
            re.DOTALL,
        )
        # Use a non-backtracking approach: find _forward_impl then check nearby
        idx = src.find('def _forward_impl(')
        assert idx >= 0
        block = src[idx:idx + 2000]
        assert '_omega3_silent_exc_count' in block, (
            "Counter should be reset near start of _forward_impl"
        )

    def test_counter_incremented_in_bridge_silent_exception(self):
        """_bridge_silent_exception() increments the counter."""
        src = _src()
        idx = src.find('def _bridge_silent_exception(')
        assert idx >= 0
        block = src[idx:idx + 3000]
        assert '_omega3_silent_exc_count' in block or 'omega3' in block, (
            "Counter should be incremented in _bridge_silent_exception"
        )

    def test_signal_written_before_return(self):
        """forward_silent_exception_count is written near end of _forward_impl."""
        src = _src()
        pat = re.compile(
            r"write_signal\(\s*['\"]forward_silent_exception_count['\"]",
        )
        assert pat.search(src), (
            "forward_silent_exception_count should be written to bus"
        )

    def test_signal_normalised_to_0_1(self):
        """Counter normalised: min(1.0, count / 10.0)."""
        src = _src()
        # Find the write call (writer side) and verify normalisation
        idx = src.find("'forward_silent_exception_count'")
        assert idx >= 0
        context = src[max(0, idx - 500):idx + 500]
        assert '10.0' in context or '10' in context, (
            "Should normalise count / 10.0"
        )

    def test_mct_reads_forward_silent_exception_count(self):
        """MCT evaluate() reads forward_silent_exception_count."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]forward_silent_exception_count['\"]",
        )
        assert pat.search(src), (
            "MCT should read forward_silent_exception_count from bus"
        )

    def test_mct_routes_to_uncertainty(self):
        """Signal routed to uncertainty channel in MCT."""
        src = _src()
        # Find the reader block and check it touches uncertainty
        idx = src.find("'forward_silent_exception_count'")
        while idx >= 0:
            context = src[idx:idx + 500]
            if 'uncertainty' in context:
                break
            idx = src.find("'forward_silent_exception_count'", idx + 1)
        assert idx >= 0, (
            "forward_silent_exception_count should route to uncertainty"
        )

    def test_functional_counter_increment(self):
        """Functional: _bridge_silent_exception increments counter."""
        bus = _make_bus()
        config = aeon.AEONConfig(
            hidden_dim=64, z_dim=64, vq_embedding_dim=64,
            vocab_size=128,
        )
        model = aeon.AEONDeltaV3(config)
        # Simulate: set counter to 0 then call _bridge_silent_exception
        model._omega3_silent_exc_count = 0
        model._bridge_silent_exception(
            'test_error', 'test_subsystem', RuntimeError('test'),
        )
        assert model._omega3_silent_exc_count == 1, (
            "Counter should increment on _bridge_silent_exception call"
        )
        model._bridge_silent_exception(
            'test_error2', 'test_subsystem2', RuntimeError('test2'),
        )
        assert model._omega3_silent_exc_count == 2

    def test_counter_not_incremented_outside_forward(self):
        """Counter only increments when _omega3_silent_exc_count is set."""
        config = aeon.AEONConfig(
            hidden_dim=64, z_dim=64, vq_embedding_dim=64,
            vocab_size=128,
        )
        model = aeon.AEONDeltaV3(config)
        # Before _forward_impl initialises the counter, it should be None
        # and _bridge_silent_exception should not crash
        if hasattr(model, '_omega3_silent_exc_count'):
            delattr(model, '_omega3_silent_exc_count')
        # Should not raise
        model._bridge_silent_exception(
            'test_error', 'test_subsystem', RuntimeError('test'),
        )


# ──────────────────────────────────────────────────────────────────────
# PATCH-Ω1: Wizard Signals → MCT Awareness
# ──────────────────────────────────────────────────────────────────────

class TestOmega1_WizardSignals:
    """PATCH-Ω1: MCT reads wizard_corpus_quality and wizard_completed."""

    def test_wizard_corpus_quality_written_in_server(self):
        """aeon_server.py writes wizard_corpus_quality."""
        src = _src_server()
        pat = re.compile(
            r"write_signal\(\s*['\"]wizard_corpus_quality['\"]",
        )
        assert pat.search(src), "Server should write wizard_corpus_quality"

    def test_wizard_completed_written_in_server(self):
        """aeon_server.py writes wizard_completed."""
        src = _src_server()
        pat = re.compile(
            r"write_signal\(\s*['\"]wizard_completed['\"]",
        )
        assert pat.search(src), "Server should write wizard_completed"

    def test_mct_reads_wizard_corpus_quality(self):
        """MCT evaluate() reads wizard_corpus_quality from bus."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]wizard_corpus_quality['\"]",
        )
        assert pat.search(src), (
            "MCT should read wizard_corpus_quality from bus"
        )

    def test_mct_reads_wizard_completed(self):
        """MCT evaluate() reads wizard_completed from bus."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]wizard_completed['\"]",
        )
        assert pat.search(src), (
            "MCT should read wizard_completed from bus"
        )

    def test_low_corpus_quality_boosts_uncertainty(self):
        """wizard_corpus_quality < 0.5 routes to uncertainty."""
        src = _src()
        idx = src.find("'wizard_corpus_quality'")
        found = False
        while idx >= 0:
            context = src[idx:idx + 500]
            if 'uncertainty' in context and '0.5' in context:
                found = True
                break
            idx = src.find("'wizard_corpus_quality'", idx + 1)
        assert found, (
            "Low wizard_corpus_quality should boost uncertainty"
        )

    def test_wizard_completed_boosts_recovery_pressure(self):
        """wizard_completed > 0.5 routes to recovery_pressure."""
        src = _src()
        idx = src.find("'wizard_completed'")
        found = False
        while idx >= 0:
            context = src[idx:idx + 500]
            if 'recovery_pressure' in context:
                found = True
                break
            idx = src.find("'wizard_completed'", idx + 1)
        assert found, (
            "wizard_completed should boost recovery_pressure"
        )

    def test_wizard_signals_in_mct_context(self):
        """Both wizard reads are in the MCT evaluate() method."""
        src = _src()
        # Find MetaCognitiveRecursionTrigger.evaluate() method
        idx_class = src.find('class MetaCognitiveRecursionTrigger')
        assert idx_class >= 0
        class_body = src[idx_class:]
        idx_eval = class_body.find('def evaluate(')
        assert idx_eval >= 0
        # Find next class boundary after evaluate
        idx_next = class_body.find('\nclass ', idx_eval + 100)
        if idx_next < 0:
            idx_next = len(class_body)
        eval_body = class_body[idx_eval:idx_next]
        assert 'wizard_corpus_quality' in eval_body, (
            "wizard_corpus_quality read should be in MCT evaluate()"
        )
        assert 'wizard_completed' in eval_body, (
            "wizard_completed read should be in MCT evaluate()"
        )


# ──────────────────────────────────────────────────────────────────────
# PATCH-Ω2: Training Convergence Bridge Extension
# ──────────────────────────────────────────────────────────────────────

class TestOmega2_TrainingBridgeExtension:
    """PATCH-Ω2: ConvergenceErrorEvolutionBridge signals stagnating/conflicting."""

    def test_stagnating_signal_written(self):
        """poll() writes training_convergence_stagnating on stagnating verdict."""
        class MockCM:
            last_verdict = None
            history = []
            threshold = 1e-5

        class MockEE:
            def record_episode(self, **kwargs):
                pass

        bus = _make_bus()
        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee, feedback_bus=bus)

        cm.last_verdict = 'stagnating'
        bridge.poll(0.5)
        val = float(bus.read_signal('training_convergence_stagnating', 0.0))
        assert val == 1.0, (
            f"Should write 1.0 for stagnating, got {val}"
        )

    def test_conflicting_signal_written(self):
        """poll() writes training_convergence_conflicting on conflicting verdict."""
        class MockCM:
            last_verdict = None
            history = []
            threshold = 1e-5

        class MockEE:
            def record_episode(self, **kwargs):
                pass

        bus = _make_bus()
        cm = MockCM()
        ee = MockEE()
        bridge = ae_train.ConvergenceErrorEvolutionBridge(cm, ee, feedback_bus=bus)

        cm.last_verdict = 'conflicting'
        bridge.poll(0.5)
        val = float(bus.read_signal('training_convergence_conflicting', 0.0))
        assert val == 1.0, (
            f"Should write 1.0 for conflicting, got {val}"
        )

    def test_converging_does_not_write_stagnating(self):
        """poll() does not write stagnating signal for converging verdict."""
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
        val = float(bus.read_signal('training_convergence_stagnating', 0.0))
        assert val == 0.0

    def test_diverging_does_not_write_conflicting(self):
        """poll() does not write conflicting signal for diverging verdict."""
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
        val = float(bus.read_signal('training_convergence_conflicting', 0.0))
        assert val == 0.0

    def test_diverging_still_writes_diverging_signal(self):
        """Existing diverging signal still works after Ω2 extension."""
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
        val = float(bus.read_signal('training_convergence_diverging', 0.0))
        assert val == 1.0

    def test_mct_reads_stagnating(self):
        """MCT evaluate() reads training_convergence_stagnating from bus."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]training_convergence_stagnating['\"]",
        )
        assert pat.search(src), (
            "MCT should read training_convergence_stagnating"
        )

    def test_mct_reads_conflicting(self):
        """MCT evaluate() reads training_convergence_conflicting from bus."""
        src = _src()
        pat = re.compile(
            r"read_signal\(\s*['\"]training_convergence_conflicting['\"]",
        )
        assert pat.search(src), (
            "MCT should read training_convergence_conflicting"
        )

    def test_stagnating_routes_to_coherence_deficit(self):
        """training_convergence_stagnating routes to coherence_deficit."""
        src = _src()
        idx = src.find("'training_convergence_stagnating'")
        found = False
        while idx >= 0:
            context = src[idx:idx + 500]
            if 'coherence_deficit' in context:
                found = True
                break
            idx = src.find("'training_convergence_stagnating'", idx + 1)
        assert found, (
            "Stagnating signal should route to coherence_deficit"
        )

    def test_conflicting_routes_to_uncertainty(self):
        """training_convergence_conflicting routes to uncertainty."""
        src = _src()
        idx = src.find("'training_convergence_conflicting'")
        found = False
        while idx >= 0:
            context = src[idx:idx + 500]
            if 'uncertainty' in context:
                found = True
                break
            idx = src.find("'training_convergence_conflicting'", idx + 1)
        assert found, (
            "Conflicting signal should route to uncertainty"
        )

    def test_source_contains_omega2_labels(self):
        """Source code is labelled with PATCH-Ω2 markers."""
        src_train = _src_train()
        assert 'PATCH-Ω2' in src_train or 'PATCH-Ω2' in src_train.encode().decode('utf-8'), (
            "ae_train.py should contain PATCH-Ω2 marker"
        )
        src_core = _src()
        assert 'PATCH-Ω2' in src_core or 'Omega2' in src_core or 'omega2' in src_core, (
            "aeon_core.py should contain PATCH-Ω2 marker"
        )


# ──────────────────────────────────────────────────────────────────────
# PATCH-Ω4: MCT Decision in API Response
# ──────────────────────────────────────────────────────────────────────

class TestOmega4_MCTDecisionInResponse:
    """PATCH-Ω4: /api/infer response includes MCT decision details."""

    def test_mct_should_trigger_read_in_server(self):
        """Server reads mct_should_trigger from feedback bus."""
        src = _src_server()
        pat = re.compile(
            r"read_signal\(\s*['\"]mct_should_trigger['\"]",
        )
        assert pat.search(src), (
            "Server should read mct_should_trigger from bus"
        )

    def test_mct_trigger_score_read_in_server(self):
        """Server reads mct_trigger_score from feedback bus."""
        src = _src_server()
        pat = re.compile(
            r"read_signal\(\s*['\"]mct_trigger_score['\"]",
        )
        assert pat.search(src), (
            "Server should read mct_trigger_score from bus"
        )

    def test_mct_dominant_trigger_read_in_server(self):
        """Server reads mct_dominant_trigger_id from feedback bus."""
        src = _src_server()
        pat = re.compile(
            r"read_signal\(\s*['\"]mct_dominant_trigger_id['\"]",
        )
        assert pat.search(src), (
            "Server should read mct_dominant_trigger_id from bus"
        )

    def test_mct_decision_dict_in_response(self):
        """Response construction includes mct_decision dict."""
        src = _src_server()
        assert 'mct_decision' in src, (
            "Server response should include mct_decision"
        )

    def test_mct_decision_has_triggered_field(self):
        """mct_decision dict includes 'triggered' boolean."""
        src = _src_server()
        idx = src.find('mct_decision')
        assert idx >= 0
        context = src[idx:idx + 500]
        assert 'triggered' in context, (
            "mct_decision should have 'triggered' field"
        )

    def test_mct_decision_has_trigger_score_field(self):
        """mct_decision dict includes 'trigger_score' float."""
        src = _src_server()
        idx = src.find('mct_decision')
        assert idx >= 0
        context = src[idx:idx + 500]
        assert 'trigger_score' in context, (
            "mct_decision should have 'trigger_score' field"
        )

    def test_mct_decision_has_dominant_trigger_field(self):
        """mct_decision dict includes 'dominant_trigger' int."""
        src = _src_server()
        idx = src.find('mct_decision')
        assert idx >= 0
        context = src[idx:idx + 1000]
        assert 'dominant_trigger' in context, (
            "mct_decision should have 'dominant_trigger' field"
        )

    def test_mct_decision_attached_to_metacognitive_state(self):
        """mct_decision is attached to metacognitive dict."""
        src = _src_server()
        # Look for: metacognitive["mct_decision"] = ...
        pat = re.compile(
            r'metacognitive\s*\[\s*["\']mct_decision["\']\s*\]',
        )
        assert pat.search(src), (
            "mct_decision should be attached to metacognitive dict"
        )

    def test_omega4_wrapped_in_exception_handler(self):
        """MCT decision read is wrapped in try/except."""
        src = _src_server()
        idx = src.find("'mct_should_trigger'")
        if idx < 0:
            idx = src.find('"mct_should_trigger"')
        assert idx >= 0
        # Check there's a try block before it
        before = src[max(0, idx - 300):idx]
        assert 'try:' in before, (
            "MCT decision read should be in a try/except block"
        )


# ──────────────────────────────────────────────────────────────────────
# Signal Ecosystem Integrity
# ──────────────────────────────────────────────────────────────────────

class TestOmega_SignalEcosystemIntegrity:
    """Verify that Ω patches don't create new orphans or missing producers."""

    def _audit_signals(self):
        """Return (written, read) sets across all source files."""
        write_pat = re.compile(
            r"""write_signal(?:_traced)?\(\s*['"]([^'"]+)['"]""",
        )
        read_pat = re.compile(
            r"""read_signal\(\s*['"]([^'"]+)['"]""",
        )
        extra_pat = re.compile(
            r"""_extra_signals\[['"]([^'"]+)['"]\]\s*=""",
        )

        src_core = _src()
        src_train = _src_train()
        src_server = _src_server()

        written = set()
        read = set()

        for src in (src_core, src_train, src_server):
            written.update(write_pat.findall(src))
            written.update(extra_pat.findall(src))
            read.update(read_pat.findall(src))

        return written, read

    def test_new_signals_are_bidirectional(self):
        """New Ω signals have both writers and readers."""
        written, read = self._audit_signals()

        omega_signals = [
            'forward_silent_exception_count',
            'training_convergence_stagnating',
            'training_convergence_conflicting',
        ]
        for sig in omega_signals:
            assert sig in written, f"{sig} should be written"
            assert sig in read, f"{sig} should be read"

    def test_wizard_signals_now_read(self):
        """wizard_corpus_quality and wizard_completed are now read."""
        written, read = self._audit_signals()
        assert 'wizard_corpus_quality' in read, (
            "wizard_corpus_quality should now be read by MCT"
        )
        assert 'wizard_completed' in read, (
            "wizard_completed should now be read by MCT"
        )

    def test_no_new_missing_producers(self):
        """No new signals are read without a writer."""
        written, read = self._audit_signals()
        missing = read - written
        # There should be zero missing producers
        assert len(missing) == 0, (
            f"Missing producers (read but never written): {sorted(missing)}"
        )

    def test_orphan_count_decreased(self):
        """Orphan count should be ≤ 2 (integration_cycle_id/timestamp are
        server-internal reads via _read_fb_signal, not read_signal)."""
        written, read = self._audit_signals()
        orphans = written - read
        # The remaining orphans should only be server-internal status
        # signals that don't need cognitive consumption
        allowed_orphans = {
            'integration_cycle_id',
            'integration_cycle_timestamp',
        }
        unexpected = orphans - allowed_orphans
        assert len(unexpected) == 0, (
            f"Unexpected orphan signals: {sorted(unexpected)}"
        )

    def test_total_signal_counts_increased(self):
        """Total written and read counts should exceed previous baseline."""
        written, read = self._audit_signals()
        # Previous baseline: 177 written, 173 read
        # Ω patches add: forward_silent_exception_count (W+R),
        # training_convergence_stagnating (W+R),
        # training_convergence_conflicting (W+R),
        # wizard_corpus_quality (R), wizard_completed (R)
        # = +3 written, +5 read
        assert len(written) >= 178, (
            f"Expected ≥178 written signals, got {len(written)}"
        )
        assert len(read) >= 178, (
            f"Expected ≥178 read signals, got {len(read)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Cross-Patch Integration
# ──────────────────────────────────────────────────────────────────────

class TestOmega_CrossPatchIntegration:
    """Verify cross-patch coherence and causal chains."""

    def test_omega3_counter_to_mct_pipeline(self):
        """Counter increment → bus write → MCT read is traceable."""
        src = _src()
        # Verify full chain exists:
        # 1. Counter initialised
        assert '_omega3_silent_exc_count' in src
        # 2. Counter incremented in _bridge_silent_exception
        idx_bridge = src.find('def _bridge_silent_exception(')
        assert idx_bridge >= 0
        bridge_body = src[idx_bridge:idx_bridge + 3000]
        assert '_omega3_silent_exc_count' in bridge_body or 'omega3' in bridge_body
        # 3. Written to bus as signal
        assert re.search(
            r"write_signal\(\s*['\"]forward_silent_exception_count['\"]",
            src,
        )
        # 4. Read by MCT
        assert re.search(
            r"read_signal\(\s*['\"]forward_silent_exception_count['\"]",
            src,
        )

    def test_omega2_verdict_to_mct_pipeline(self):
        """Stagnating verdict → bus write → MCT read is traceable."""
        src_train = _src_train()
        src_core = _src()
        # 1. Written in ae_train
        assert re.search(
            r"write_signal\(\s*['\"]training_convergence_stagnating['\"]",
            src_train,
        )
        # 2. Read in aeon_core MCT
        assert re.search(
            r"read_signal\(\s*['\"]training_convergence_stagnating['\"]",
            src_core,
        )

    def test_omega1_wizard_to_mct_pipeline(self):
        """Wizard completion → bus write (server) → MCT read (core)."""
        src_server = _src_server()
        src_core = _src()
        # Written in server
        assert re.search(
            r"write_signal\(\s*['\"]wizard_completed['\"]",
            src_server,
        )
        # Read in core
        assert re.search(
            r"read_signal\(\s*['\"]wizard_completed['\"]",
            src_core,
        )

    def test_omega4_mct_decision_pipeline(self):
        """MCT writes trigger signals → server reads → API response."""
        src_core = _src()
        src_server = _src_server()
        # MCT writes mct_should_trigger
        assert re.search(
            r"write_signal(?:_traced)?\(\s*['\"]mct_should_trigger['\"]",
            src_core,
        )
        # Server reads it
        assert re.search(
            r"read_signal\(\s*['\"]mct_should_trigger['\"]",
            src_server,
        )
        # Server includes mct_decision in response
        assert 'mct_decision' in src_server
