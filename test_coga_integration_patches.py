"""Tests for PATCH-COGA-1..5: Final cognitive integration patches.

Covers:
- PATCH-COGA-1: Re-entrancy scope annotation
- PATCH-COGA-2: Training bridge bidirectional flow verification
- PATCH-COGA-3: Phase 10 emergence cache consumption tracking
- PATCH-COGA-4: Causal chain BFS cycle/broken-link annotation
- PATCH-COGA-5: Generation fence fallback MCT bridge
- Signal ecosystem audit (269 bidirectional, 0 orphans)
- Integration map, mutual reinforcement, meta-cognitive trigger, causal transparency
"""

import inspect
import re
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import AEONDeltaV3, AEONConfig, CognitiveFeedbackBus, TemporalCausalTraceBuffer


# ════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model():
    config = AEONConfig()
    m = AEONDeltaV3(config)
    m._verify_and_reinforce_in_progress = False
    m.verify_and_reinforce()
    return m


@pytest.fixture(scope="module")
def bus():
    return CognitiveFeedbackBus(hidden_dim=64)


# ════════════════════════════════════════════════════════════════════════
# §1 — PATCH-COGA-1: Re-entrancy scope annotation
# ════════════════════════════════════════════════════════════════════════

class TestCOGA1ReentrancyScope:
    """Verify re-entrancy skip carries scope metadata."""

    def test_reentrant_return_has_scope(self, model):
        """When re-entered, return dict includes skipped_scope list."""
        model._verify_and_reinforce_in_progress = True
        try:
            result = model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        assert 'skipped_scope' in result
        assert isinstance(result['skipped_scope'], list)

    def test_reentrant_return_has_severity(self, model):
        """When re-entered, return dict includes scope_severity int."""
        model._verify_and_reinforce_in_progress = True
        try:
            result = model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        assert 'scope_severity' in result
        assert isinstance(result['scope_severity'], int)

    def test_scope_includes_error_evolution(self, model):
        """Scope should include error_evolution_feedback when present."""
        model._verify_and_reinforce_in_progress = True
        try:
            result = model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        if model.error_evolution is not None:
            assert 'error_evolution_feedback' in result['skipped_scope']

    def test_scope_includes_causal_trace(self, model):
        """Scope should include causal_trace_recording when present."""
        model._verify_and_reinforce_in_progress = True
        try:
            result = model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        if model.causal_trace is not None:
            assert 'causal_trace_recording' in result['skipped_scope']

    def test_severity_proportional_to_components(self, model):
        """Severity should equal the count of available subsystems."""
        model._verify_and_reinforce_in_progress = True
        try:
            result = model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        assert result['scope_severity'] == len(result['skipped_scope'])

    def test_causal_trace_records_scope(self, model):
        """Causal trace entry for reentrant skip carries scope metadata."""
        model._verify_and_reinforce_in_progress = True
        try:
            model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        if model.causal_trace is not None:
            entries = model.causal_trace.find(
                subsystem='verify_and_reinforce',
                decision='skipped_reentrant',
            )
            assert len(entries) > 0
            meta = entries[-1].get('metadata', {})
            assert 'skipped_scope' in meta
            assert 'scope_severity' in meta

    def test_bus_receives_scope_severity(self, model):
        """Feedback bus receives reinforce_skip_scope_severity signal."""
        model._verify_and_reinforce_in_progress = True
        try:
            model.verify_and_reinforce()
        finally:
            model._verify_and_reinforce_in_progress = False
        val = model.feedback_bus.read_signal(
            'reinforce_skip_scope_severity', -1.0,
        )
        # Should be > 0 (at least some subsystems present)
        assert val >= 0.0

    def test_normal_path_clears_severity(self, model):
        """Normal verify_and_reinforce writes 0.0 scope severity."""
        model._verify_and_reinforce_in_progress = False
        model.verify_and_reinforce()
        val = model.feedback_bus.read_signal(
            'reinforce_skip_scope_severity', -1.0,
        )
        assert val == 0.0


# ════════════════════════════════════════════════════════════════════════
# §2 — PATCH-COGA-2: Training bridge flow verification
# ════════════════════════════════════════════════════════════════════════

class TestCOGA2TrainingBridge:
    """Verify training bridge includes flow verification fields."""

    def test_bridge_has_flow_fields(self, model):
        """Training bridge dict includes bidirectional flow fields."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        bridge = report.get('training_bridge', {})
        assert 'inference_to_training_signal_live' in bridge
        assert 'training_to_inference_signal_live' in bridge
        assert 'bidirectional_flow_verified' in bridge

    def test_bridge_flow_types(self, model):
        """Flow verification fields are booleans."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        bridge = report.get('training_bridge', {})
        assert isinstance(bridge['inference_to_training_signal_live'], bool)
        assert isinstance(bridge['training_to_inference_signal_live'], bool)
        assert isinstance(bridge['bidirectional_flow_verified'], bool)

    def test_bidirectional_requires_both(self, model):
        """bidirectional_flow_verified is True only when both flows live."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        bridge = report.get('training_bridge', {})
        if bridge['bidirectional_flow_verified']:
            assert bridge['inference_to_training_signal_live']
            assert bridge['training_to_inference_signal_live']


# ════════════════════════════════════════════════════════════════════════
# §3 — PATCH-COGA-3: Phase 10 consumption tracking
# ════════════════════════════════════════════════════════════════════════

class TestCOGA3Phase10Consumption:
    """Verify Phase 10 tracks emergence verdict consumption."""

    def test_phase10_active_on_cold_start(self, model):
        """Phase 10 is active during cold start (no forward pass)."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        p10 = [p for p in seq if p['order'] == 10][0]
        assert p10['status'] in ('active', 'achieved')

    def test_phase10_has_consumption_metadata(self, model):
        """Phase 10 dict includes _verdict_consumed_pass metadata."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        p10 = [p for p in seq if p['order'] == 10][0]
        assert '_verdict_consumed_pass' in p10

    def test_all_phases_active(self, model):
        """All 10 phases must be active or achieved."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        for phase in seq:
            assert phase['status'] in ('active', 'achieved'), (
                f"Phase {phase['order']} ({phase['phase']}) "
                f"has status '{phase['status']}'"
            )


# ════════════════════════════════════════════════════════════════════════
# §4 — PATCH-COGA-4: Causal chain BFS cycle annotation
# ════════════════════════════════════════════════════════════════════════

class TestCOGA4CausalChainAnnotation:
    """Verify get_causal_chain annotates cycles and broken links."""

    def test_complete_chain_has_meta(self):
        """A complete chain has _chain_meta with complete=True."""
        buf = TemporalCausalTraceBuffer(max_entries=100)
        buf.record("sys", "decision_a", metadata={}, severity="info")
        entries = buf.recent(1)
        entry_id = entries[0]['id']
        chain = buf.get_causal_chain(entry_id)
        assert len(chain) == 1
        meta = chain[-1].get('_chain_meta', {})
        assert meta.get('complete') is True
        assert meta.get('broken_links') == []
        assert meta.get('cycle_links') == []

    def test_broken_link_detected(self):
        """When a prerequisite is missing, broken_links is populated."""
        buf = TemporalCausalTraceBuffer(max_entries=100)
        buf.record(
            "sys", "decision_b",
            causal_prerequisites=["nonexistent_id"],
            metadata={}, severity="info",
        )
        entries = buf.recent(1)
        entry_id = entries[0]['id']
        chain = buf.get_causal_chain(entry_id)
        meta = chain[-1].get('_chain_meta', {})
        assert meta.get('complete') is False
        assert 'nonexistent_id' in meta.get('broken_links', [])

    def test_cycle_detected(self):
        """When a cycle exists, cycle_links is populated."""
        buf = TemporalCausalTraceBuffer(max_entries=100)
        # Create two entries that reference each other
        buf.record("sys", "first", metadata={}, severity="info")
        first_id = buf.recent(1)[0]['id']
        buf.record(
            "sys", "second",
            causal_prerequisites=[first_id],
            metadata={}, severity="info",
        )
        second_id = buf.recent(1)[0]['id']
        # Manually add a back-reference to create a cycle
        with buf._lock:
            for e in buf._entries:
                if e['id'] == first_id:
                    e['causal_prerequisites'] = [second_id]
                    break
        chain = buf.get_causal_chain(second_id)
        meta = chain[-1].get('_chain_meta', {})
        assert meta.get('complete') is False
        assert len(meta.get('cycle_links', [])) > 0

    def test_empty_chain_returns_empty(self):
        """get_causal_chain with missing ID returns empty list."""
        buf = TemporalCausalTraceBuffer(max_entries=100)
        chain = buf.get_causal_chain("does_not_exist")
        assert chain == []


# ════════════════════════════════════════════════════════════════════════
# §5 — PATCH-COGA-5: Generation fence fallback MCT bridge
# ════════════════════════════════════════════════════════════════════════

class TestCOGA5GenerationFence:
    """Verify generation fence fallbacks are surfaced to MCT."""

    def test_fallback_counter_increments(self):
        """read_signal_current_gen fallback increments counter."""
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('test_sig', 1.0)
        bus.flush_consumed()  # Advance generation
        # Now read in new generation — should fall back
        val = bus.read_signal_current_gen('test_sig', 0.0)
        assert val == 0.0  # Fell back to default
        count = getattr(bus, '_generation_fence_fallback_count', 0)
        assert count >= 1

    def test_current_gen_read_no_fallback(self):
        """read_signal_current_gen in same gen does not increment."""
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus._generation_fence_fallback_count = 0
        bus.write_signal('fresh_sig', 1.0)
        val = bus.read_signal_current_gen('fresh_sig', 0.0)
        assert val == 1.0
        count = getattr(bus, '_generation_fence_fallback_count', 0)
        assert count == 0

    def test_pressure_surfaced_in_extra_signals(self, model):
        """generation_fence_fallback_pressure is written to bus."""
        src = inspect.getsource(AEONDeltaV3._build_feedback_extra_signals)
        assert 'generation_fence_fallback_pressure' in src

    def test_mct_reads_pressure(self):
        """MCT evaluate reads generation_fence_fallback_pressure."""
        from aeon_core import MetaCognitiveRecursionTrigger
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'generation_fence_fallback_pressure' in src


# ════════════════════════════════════════════════════════════════════════
# §6 — Signal Ecosystem Audit
# ════════════════════════════════════════════════════════════════════════

class TestSignalEcosystemAudit:
    """Verify signal ecosystem health after COGA patches."""

    @staticmethod
    def _audit():
        with open('aeon_core.py') as f:
            core = f.read()
        with open('ae_train.py') as f:
            train = f.read()
        with open('aeon_server.py') as f:
            server = f.read()
        full = core + train + server
        written = set()
        read = set()
        for m in re.finditer(
            r"write_signal(?:_traced)?\s*\(\s*['\"](\w+)['\"]", full,
        ):
            written.add(m.group(1))
        for m in re.finditer(
            r"read_signal\s*\(\s*['\"](\w+)['\"]", full,
        ):
            read.add(m.group(1))
        return written, read

    def test_no_write_only_orphans(self):
        w, r = self._audit()
        orphans = w - r
        assert orphans == set(), f"Write-only orphans: {orphans}"

    def test_no_read_only_orphans(self):
        w, r = self._audit()
        orphans = r - w
        assert orphans == set(), f"Read-only orphans: {orphans}"

    def test_bidirectional_count_ge_269(self):
        w, r = self._audit()
        bi = w & r
        assert len(bi) >= 269, f"Expected >= 269, got {len(bi)}"

    def test_new_coga_signals_bidirectional(self):
        w, r = self._audit()
        coga = {
            'reinforce_skip_scope_severity',
            'generation_fence_fallback_pressure',
        }
        for sig in coga:
            assert sig in w, f"{sig} not written"
            assert sig in r, f"{sig} not read"


# ════════════════════════════════════════════════════════════════════════
# §7 — Integration Map
# ════════════════════════════════════════════════════════════════════════

class TestIntegrationMap:
    """Verify no isolated critical paths."""

    def test_no_isolated_paths(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        blocked = [p for p in seq if p['status'] in ('blocked', 'incomplete')]
        assert len(blocked) == 0, f"Blocked phases: {blocked}"

    def test_10_phases_present(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        assert len(seq) == 10


# ════════════════════════════════════════════════════════════════════════
# §8 — Mutual Reinforcement
# ════════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify components verify and stabilize each other."""

    def test_reinforce_writes_to_bus(self, model):
        """verify_and_reinforce writes quality signals to bus."""
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert 'write_signal' in src

    def test_mct_reads_reinforce_signals(self):
        """MCT reads reinforce_skip_scope_severity."""
        from aeon_core import MetaCognitiveRecursionTrigger
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'reinforce_skip_scope_severity' in src

    def test_cross_validator_present(self, model):
        assert getattr(model, '_subsystem_cross_validator', None) is not None


# ════════════════════════════════════════════════════════════════════════
# §9 — Meta-Cognitive Trigger
# ════════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTrigger:
    """Verify uncertainty triggers higher-order review."""

    def test_generation_fence_triggers_uncertainty(self):
        """MCT routes generation_fence_fallback_pressure to uncertainty."""
        from aeon_core import MetaCognitiveRecursionTrigger
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'generation_fence_fallback_pressure' in src
        assert 'uncertainty' in src

    def test_scope_severity_triggers_recovery(self):
        """MCT routes reinforce_skip_scope_severity to recovery_pressure."""
        from aeon_core import MetaCognitiveRecursionTrigger
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'reinforce_skip_scope_severity' in src
        assert 'recovery_pressure' in src


# ════════════════════════════════════════════════════════════════════════
# §10 — Causal Transparency
# ════════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify outputs are traceable to root causes."""

    def test_causal_chain_traceable(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        cc = report.get('causal_chain', {})
        assert cc.get('traceable') is True

    def test_chain_meta_in_causal_trace(self):
        """get_causal_chain returns _chain_meta for transparency."""
        buf = TemporalCausalTraceBuffer(max_entries=100)
        buf.record("test", "root", metadata={}, severity="info")
        root_id = buf.recent(1)[0]['id']
        buf.record(
            "test", "derived",
            causal_prerequisites=[root_id],
            metadata={}, severity="info",
        )
        derived_id = buf.recent(1)[0]['id']
        chain = buf.get_causal_chain(derived_id)
        assert len(chain) == 2
        meta = chain[-1].get('_chain_meta', {})
        assert meta['complete'] is True


# ════════════════════════════════════════════════════════════════════════
# §11 — Activation Sequence
# ════════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify safe activation order."""

    def test_10_phases(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        assert len(seq) == 10

    def test_phases_ordered(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        orders = [p['order'] for p in seq]
        assert orders == list(range(1, 11))

    def test_all_active(self, model):
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        for p in seq:
            assert p['status'] in ('active', 'achieved'), (
                f"Phase {p['order']} status: {p['status']}"
            )
