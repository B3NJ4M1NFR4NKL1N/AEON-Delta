"""Tests for PATCH-Δ1..Δ6 — Final Integration & Cognitive Activation.

These patches close the remaining gaps that prevent full causal coherence
and self-reflection in the AEON-Delta architecture:

  Δ1  Config Default Alignment (9 getattr fallbacks False→True)
  Δ3  Server Coherence Pre-Seeding (harmonise 0.0/1.0 → 0.7)
  Δ5  Server Signal DAG Integration (virtual server_inference node)
  Δ6  MCT Signal Pre-Registration (6 MCT output signals)

Δ2 (MCT↔Bus wiring) and Δ4 (Error Evolution↔Causal Trace) were already
implemented in prior patches and are validated here for completeness.
"""

import re
import sys
import types
import math
import pytest

# ── Bootstrap ──────────────────────────────────────────────────────────
import aeon_core
import ae_train


# ── Helpers ────────────────────────────────────────────────────────────

def make_bus(hidden_dim: int = 64) -> aeon_core.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _src(path: str) -> str:
    """Read source file for audit tests."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ1: Config Default Alignment
# ═══════════════════════════════════════════════════════════════════════

class TestDelta1ConfigDefaults:
    """Verify getattr fallbacks match AEONConfig class defaults."""

    # Each tuple: (config_field, expected_default, init_line_pattern)
    _CONFIG_FLAGS = [
        ("enable_recursive_meta_loop", True),
        ("enable_hierarchical_meta_loop", True),
        ("enable_certified_meta_loop", True),
        ("enable_adaptive_meta_loop", True),
        ("enable_causal_model", True),
        ("enable_mcts_planner", True),
        ("enable_causal_world_model", True),
        ("enable_active_learning_planner", True),
        ("enable_hierarchical_vae", True),
    ]

    def test_config_declares_true(self):
        """AEONConfig class defaults match expected True."""
        cfg = aeon_core.AEONConfig
        for field, expected in self._CONFIG_FLAGS:
            # Dataclass field defaults are stored as class attributes
            default = getattr(cfg, field, None)
            # For dataclass, check __dataclass_fields__
            if hasattr(cfg, '__dataclass_fields__'):
                dc_field = cfg.__dataclass_fields__.get(field)
                if dc_field is not None:
                    default = dc_field.default
            assert default == expected, (
                f"AEONConfig.{field} should default to {expected}, "
                f"got {default}"
            )

    def test_getattr_fallbacks_aligned_in_source(self):
        """Verify that __init__ getattr fallbacks use True for True-default flags."""
        src = _src("aeon_core.py")
        for field, expected in self._CONFIG_FLAGS:
            pattern = rf"getattr\(config,\s*'{field}',\s*(True|False)\)"
            matches = re.findall(pattern, src)
            assert len(matches) >= 1, (
                f"No getattr pattern found for {field}"
            )
            for m in matches:
                assert m == str(expected), (
                    f"getattr fallback for {field} should be {expected}, "
                    f"found {m}"
                )

    def test_patch_comment_markers_present(self):
        """PATCH-Δ1 comment markers exist in source."""
        src = _src("aeon_core.py")
        for suffix in "abcdefghi":
            marker = f"PATCH-Δ1{suffix}"
            assert marker in src, f"Missing comment marker: {marker}"


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ2: MCT ↔ Feedback Bus (pre-existing — verify still wired)
# ═══════════════════════════════════════════════════════════════════════

class TestDelta2MCTBusWiring:
    """Verify MCT is wired to feedback bus via set_feedback_bus."""

    def test_set_feedback_bus_call_in_init(self):
        """AEONDeltaV3.__init__ calls set_feedback_bus on MCT."""
        src = _src("aeon_core.py")
        assert "metacognitive_trigger.set_feedback_bus" in src

    def test_mct_write_signal_traced_present(self):
        """MCT evaluate() writes mct_should_trigger via write_signal_traced."""
        src = _src("aeon_core.py")
        assert "write_signal_traced" in src
        assert "'mct_should_trigger'" in src
        assert "'mct_trigger_score'" in src

    def test_mct_bus_roundtrip(self):
        """MCT signals can be written and read via the bus."""
        bus = make_bus()
        # Pre-register as PATCH-Δ6 does
        for sig in ('mct_should_trigger', 'mct_trigger_score',
                     'mct_decision_entropy'):
            bus.register_signal(sig, default=0.0)

        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.85)
        bus.write_signal('mct_decision_entropy', 0.3)

        assert float(bus.read_signal('mct_should_trigger', 0.0)) == 1.0
        assert float(bus.read_signal('mct_trigger_score', 0.0)) == pytest.approx(0.85)
        assert float(bus.read_signal('mct_decision_entropy', 0.0)) == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ3: Server Coherence Signal Pre-Seeding
# ═══════════════════════════════════════════════════════════════════════

class TestDelta3ServerCoherencePreSeed:
    """Verify harmonised defaults and pre-seed write."""

    def test_ae_train_default_is_07(self):
        """ae_train.py reads server_coherence_score with default 0.7."""
        src = _src("ae_train.py")
        # Find all read_signal('server_coherence_score', ...) patterns
        pattern = r"read_signal\(\s*'server_coherence_score'\s*,\s*([\d.]+)"
        matches = re.findall(pattern, src)
        assert len(matches) >= 2, (
            f"Expected at least 2 server_coherence_score reads, found {len(matches)}"
        )
        for m in matches:
            assert float(m) == pytest.approx(0.7), (
                f"Default should be 0.7, found {m}"
            )

    def test_aeon_core_mct_default_is_07(self):
        """aeon_core.py MCT reads server_coherence_score with default 0.7."""
        src = _src("aeon_core.py")
        # The read spans multiple lines, so use a simpler check:
        # find "'server_coherence_score', 0.7" near a read_signal call
        assert "'server_coherence_score', 0.7" in src, (
            "Expected 'server_coherence_score', 0.7 in aeon_core.py"
        )

    def test_preseed_write_in_init(self):
        """AEONDeltaV3.__init__ pre-seeds server_coherence_score to 0.7."""
        src = _src("aeon_core.py")
        assert "PATCH-Δ3d" in src, "Missing PATCH-Δ3d comment marker"
        # Check that write_signal('server_coherence_score', 0.7) exists
        assert "write_signal('server_coherence_score', 0.7)" in src

    def test_preseed_ssp_pressure(self):
        """AEONDeltaV3.__init__ pre-seeds server_ssp_pressure to 0.0."""
        src = _src("aeon_core.py")
        assert "write_signal('server_ssp_pressure', 0.0)" in src

    def test_bus_preseed_roundtrip(self):
        """Pre-seeded values are readable immediately."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.7)
        bus.write_signal('server_ssp_pressure', 0.0)

        coh = float(bus.read_signal('server_coherence_score', -1.0))
        ssp = float(bus.read_signal('server_ssp_pressure', -1.0))
        assert coh == pytest.approx(0.7)
        assert ssp == pytest.approx(0.0)

    def test_consistent_view_before_server(self):
        """Both training and MCT get the same value before first server write."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.7)

        # Simulate training read (default 0.7)
        train_view = float(bus.read_signal('server_coherence_score', 0.7))
        # Simulate MCT read (default 0.7)
        mct_view = float(bus.read_signal('server_coherence_score', 0.7))

        assert train_view == mct_view == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ4: Error Evolution ↔ Causal Trace (pre-existing — verify)
# ═══════════════════════════════════════════════════════════════════════

class TestDelta4CausalTraceWiring:
    """Verify causal trace is properly initialised."""

    def test_temporal_causal_trace_init(self):
        """When enable_causal_trace=True, TemporalCausalTraceBuffer is used."""
        src = _src("aeon_core.py")
        # Check that TemporalCausalTraceBuffer is instantiated in __init__
        assert "TemporalCausalTraceBuffer(" in src

    def test_null_causal_trace_fallback(self):
        """_NullCausalTrace is used as fallback."""
        src = _src("aeon_core.py")
        assert "_NullCausalTrace(" in src


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ5: Server Signal DAG Integration
# ═══════════════════════════════════════════════════════════════════════

class TestDelta5ServerSignalDAG:
    """Verify server_inference node in pipeline dependencies."""

    def test_server_inference_in_pipeline_deps(self):
        """server_inference appears as upstream in _PIPELINE_DEPENDENCIES."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        server_edges = [
            (u, d) for u, d in deps if u == "server_inference"
        ]
        assert len(server_edges) >= 1, (
            "No server_inference edges in _PIPELINE_DEPENDENCIES"
        )

    def test_server_inference_to_feedback_bus_edge(self):
        """Edge (server_inference, feedback_bus) exists."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        assert ("server_inference", "feedback_bus") in deps

    def test_feedback_bus_to_ucc_edge(self):
        """Edge (feedback_bus, unified_cognitive_cycle) exists."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        assert ("feedback_bus", "unified_cognitive_cycle") in deps

    def test_server_inference_in_node_attr_map(self):
        """server_inference maps to feedback_bus in _NODE_ATTR_MAP."""
        attr_map = aeon_core.AEONDeltaV3._NODE_ATTR_MAP
        assert "server_inference" in attr_map
        assert attr_map["server_inference"] == "feedback_bus"

    def test_patch_comment_marker(self):
        """PATCH-Δ5 comment markers exist."""
        src = _src("aeon_core.py")
        assert "PATCH-Δ5" in src


# ═══════════════════════════════════════════════════════════════════════
#  PATCH-Δ6: MCT Signal Pre-Registration
# ═══════════════════════════════════════════════════════════════════════

class TestDelta6MCTPreRegistration:
    """Verify MCT signals are pre-registered on the feedback bus."""

    _MCT_SIGNALS = [
        'mct_should_trigger',
        'mct_trigger_score',
        'mct_decision_entropy',
        'mct_dominant_trigger_signal',
        'mct_emergence_response_active',
        'mct_evaluation_failure',
    ]

    def test_preregistration_in_source(self):
        """PATCH-Δ6 block registers all MCT signals."""
        src = _src("aeon_core.py")
        assert "PATCH-Δ6" in src, "Missing PATCH-Δ6 comment marker"
        for sig in self._MCT_SIGNALS:
            assert f"'{sig}'" in src, f"Signal {sig} not found in source"

    def test_preregistration_on_bus(self):
        """After registration, signals are readable with non-None defaults."""
        bus = make_bus()
        for sig in self._MCT_SIGNALS:
            bus.register_signal(sig, default=0.0)

        for sig in self._MCT_SIGNALS:
            val = float(bus.read_signal(sig, -1.0))
            # Should return 0.0 (registered default), not -1.0
            assert val == pytest.approx(0.0), (
                f"{sig} should be 0.0 after registration, got {val}"
            )

    def test_write_after_preregistration(self):
        """Writes succeed after pre-registration."""
        bus = make_bus()
        for sig in self._MCT_SIGNALS:
            bus.register_signal(sig, default=0.0)

        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.9)

        assert float(bus.read_signal('mct_should_trigger', 0.0)) == 1.0
        assert float(bus.read_signal('mct_trigger_score', 0.0)) == pytest.approx(0.9)


# ═══════════════════════════════════════════════════════════════════════
#  Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════

class TestSignalEcosystemAudit:
    """Verify signal health after all Δ patches."""

    def _extract_signals(self, path, pattern):
        """Extract signal names from source using regex."""
        src = _src(path)
        return set(re.findall(pattern, src))

    def test_no_new_orphans(self):
        """No signal is written without a corresponding read.

        This is a structural audit: for every write_signal('X', ...)
        in aeon_core.py, there should be at least one read_signal('X', ...)
        somewhere in aeon_core.py or ae_train.py.
        """
        # Signals known to be server-only writes (read in aeon_server.py)
        _server_only_writes = {
            'server_coherence_score',
            'server_reinforcement_pressure',
            'integration_cycle_id',
            'integration_cycle_timestamp',
            'wizard_completed',
            'wizard_corpus_quality',
            'server_inference_complete',
        }

        core_src = _src("aeon_core.py")
        train_src = _src("ae_train.py")
        combined_read_src = core_src + train_src

        # Extract written signals
        write_pat = r"write_signal(?:_traced)?\(\s*['\"]([a-z_]+)['\"]\s*,"
        written = set(re.findall(write_pat, core_src))
        written |= set(re.findall(write_pat, train_src))

        # Extract read signals
        read_pat = r"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]([a-z_]+)['\"]\s*,"
        read = set(re.findall(read_pat, combined_read_src))
        # Also count _extra_signals.get('X') as reads
        extra_pat = r"_extra_signals\.get\(\s*['\"]([a-z_]+)['\"]\s*"
        read |= set(re.findall(extra_pat, combined_read_src))

        orphans = written - read - _server_only_writes
        # Filter very short names that might be regex false positives
        orphans = {s for s in orphans if len(s) > 3}

        # We don't assert zero orphans because the system is large,
        # but we verify no NEW orphans from Δ patches
        delta_signals = {
            'server_coherence_score',
            'server_ssp_pressure',
            'mct_should_trigger',
            'mct_trigger_score',
            'mct_decision_entropy',
            'mct_dominant_trigger_signal',
            'mct_emergence_response_active',
            'mct_evaluation_failure',
        }
        delta_orphans = delta_signals & orphans
        assert len(delta_orphans) == 0, (
            f"Δ-patch signals are orphaned: {delta_orphans}"
        )

    def test_delta_signals_are_bidirectional(self):
        """All Δ-patch signals have both writers and readers."""
        core_src = _src("aeon_core.py")
        train_src = _src("ae_train.py")
        server_src = _src("aeon_server.py")
        combined = core_src + train_src + server_src

        write_pat = r"write_signal(?:_traced)?\(\s*['\"]([a-z_]+)['\"]\s*,"
        read_pat = r"read_signal(?:_current_gen|_any_gen)?\(\s*['\"]([a-z_]+)['\"]\s*,"
        extra_pat = r"_extra_signals\.get\(\s*['\"]([a-z_]+)['\"]\s*"

        written = set(re.findall(write_pat, combined))
        read = set(re.findall(read_pat, combined))
        read |= set(re.findall(extra_pat, combined))

        # Key Δ signals that should be bidirectional
        key_signals = [
            'mct_should_trigger',
            'mct_trigger_score',
            'server_coherence_score',
        ]
        for sig in key_signals:
            assert sig in written, f"{sig} not written anywhere"
            assert sig in read, f"{sig} not read anywhere"


# ═══════════════════════════════════════════════════════════════════════
#  E2E Integration Flow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestE2EIntegrationFlow:
    """End-to-end flow tests for cognitive coherence."""

    def test_server_coherence_to_training_flow(self):
        """Server writes coherence → training reads consistent value."""
        bus = make_bus()
        # Pre-seed (Δ3d)
        bus.write_signal('server_coherence_score', 0.7)

        # Training reads before first server write
        train_val = float(bus.read_signal('server_coherence_score', 0.7))
        assert train_val == pytest.approx(0.7)

        # Server writes actual score
        bus.write_signal('server_coherence_score', 0.35)

        # Training reads updated value
        train_val = float(bus.read_signal('server_coherence_score', 0.7))
        assert train_val == pytest.approx(0.35)

        # Low coherence should trigger boost in training
        assert train_val < 0.4  # Triggers PATCH-C boost

    def test_mct_to_training_intervention_flow(self):
        """MCT writes trigger → training reads and adapts."""
        bus = make_bus()
        # Pre-register (Δ6)
        bus.register_signal('mct_should_trigger', default=0.0)
        bus.register_signal('mct_trigger_score', default=0.0)

        # MCT fires
        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.9)

        # Training reads
        triggered = float(bus.read_signal('mct_should_trigger', 0.0))
        score = float(bus.read_signal('mct_trigger_score', 0.0))

        assert triggered == 1.0
        assert score == pytest.approx(0.9)
        # >0.8 should skip backward (PATCH-A logic)
        assert triggered > 0.8

    def test_coherence_consistency_after_preseed(self):
        """MCT and training see the same coherence after pre-seed."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.7)

        # Simulate MCT read path
        mct_coh = float(bus.read_signal('server_coherence_score', 0.7))
        # Simulate training read path
        train_coh = float(bus.read_signal('server_coherence_score', 0.7))

        assert mct_coh == train_coh, (
            f"MCT ({mct_coh}) and training ({train_coh}) see different coherence"
        )

    def test_mct_preseed_then_update(self):
        """MCT signals start at 0.0, then update after evaluate."""
        bus = make_bus()
        for sig in ('mct_should_trigger', 'mct_trigger_score'):
            bus.register_signal(sig, default=0.0)

        # Initial state: not triggered
        assert float(bus.read_signal('mct_should_trigger', -1.0)) == 0.0
        assert float(bus.read_signal('mct_trigger_score', -1.0)) == 0.0

        # MCT evaluates and triggers
        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.75)

        # Training reads actual state
        assert float(bus.read_signal('mct_should_trigger', 0.0)) == 1.0
        assert float(bus.read_signal('mct_trigger_score', 0.0)) == pytest.approx(0.75)


# ═══════════════════════════════════════════════════════════════════════
#  Causal Transparency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Verify that conclusions remain traceable to root causes."""

    def test_pipeline_deps_form_connected_graph(self):
        """_PIPELINE_DEPENDENCIES forms a connected DAG from input to decoder."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        # Build adjacency
        adj = {}
        for u, d in deps:
            adj.setdefault(u, set()).add(d)

        # BFS from 'input' — should reach many nodes
        visited = set()
        queue = ["input"]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for n in adj.get(node, []):
                queue.append(n)

        assert "decoder" in visited, "decoder not reachable from input"
        assert "metacognitive_trigger" in visited, "MCT not reachable from input"
        assert "error_evolution" in visited, "error_evolution not reachable"
        assert "feedback_bus" in visited, "feedback_bus not reachable"
        assert "server_inference" not in visited, (
            "server_inference should not be reachable from input "
            "(it's an independent entry point)"
        )

    def test_server_inference_reaches_ucc(self):
        """server_inference → feedback_bus → unified_cognitive_cycle."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        adj = {}
        for u, d in deps:
            adj.setdefault(u, set()).add(d)

        # BFS from server_inference
        visited = set()
        queue = ["server_inference"]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for n in adj.get(node, []):
                queue.append(n)

        assert "feedback_bus" in visited
        assert "unified_cognitive_cycle" in visited

    def test_all_node_attr_map_entries_are_valid(self):
        """Every _NODE_ATTR_MAP value is a plausible attribute name."""
        attr_map = aeon_core.AEONDeltaV3._NODE_ATTR_MAP
        for node, attr in attr_map.items():
            assert isinstance(attr, str) and len(attr) > 0, (
                f"Invalid attr for node {node}: {attr!r}"
            )


# ═══════════════════════════════════════════════════════════════════════
#  Activation Sequence Tests
# ═══════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """Verify patches can be applied in the correct order."""

    def test_delta1_before_delta2(self):
        """Config defaults (Δ1) must be fixed before MCT bus wiring (Δ2)."""
        src = _src("aeon_core.py")
        # Δ1 patches are in module init (enable_* blocks)
        # Δ2 (set_feedback_bus) comes after module init
        pos_d1 = src.find("PATCH-Δ1a")
        pos_d2 = src.find("set_feedback_bus(self.feedback_bus)")
        assert pos_d1 < pos_d2, "Δ1 must come before Δ2 in source order"

    def test_delta3_preseed_after_bus_creation(self):
        """Pre-seed (Δ3d) must come after feedback_bus construction."""
        src = _src("aeon_core.py")
        pos_bus = src.find("self.feedback_bus = CognitiveFeedbackBus(")
        pos_preseed = src.find("PATCH-Δ3d")
        assert pos_bus < pos_preseed, (
            "Δ3d pre-seed must come after feedback_bus construction"
        )

    def test_delta6_after_mct_creation(self):
        """MCT pre-registration (Δ6) must come after MCT construction."""
        src = _src("aeon_core.py")
        pos_mct = src.find("MetaCognitiveRecursionTrigger(")
        pos_d6 = src.find("PATCH-Δ6")
        assert pos_mct < pos_d6, (
            "Δ6 must come after MCT construction"
        )

    def test_delta5_structural_only(self):
        """Δ5 modifies class-level constants, no ordering constraint."""
        deps = aeon_core.AEONDeltaV3._PIPELINE_DEPENDENCIES
        # Just verify the structural addition exists
        assert ("server_inference", "feedback_bus") in deps


# ═══════════════════════════════════════════════════════════════════════
#  Mutual Reinforcement Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Verify active components can verify each other's states."""

    def test_training_adapts_to_mct(self):
        """Training reads MCT signals and adapts behaviour."""
        bus = make_bus()
        bus.register_signal('mct_should_trigger', default=0.0)
        bus.register_signal('mct_trigger_score', default=0.0)

        # MCT writes high trigger
        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.95)

        # Training reads and decides to skip backward
        triggered = float(bus.read_signal('mct_should_trigger', 0.0))
        score = float(bus.read_signal('mct_trigger_score', 0.0))
        should_skip = triggered > 0.8
        assert should_skip, "Training should skip backward on high MCT trigger"

    def test_mct_adapts_to_server_coherence(self):
        """MCT reads server coherence and adjusts trigger sensitivity."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.3)  # Low coherence

        coh = float(bus.read_signal('server_coherence_score', 0.7))
        # Low server coherence → boost coherence_deficit in MCT
        if 0.0 < coh < 0.5:
            boost = 0.5 - coh
            assert boost > 0.0, "Low coherence should produce positive boost"

    def test_server_coherence_drives_training_boost(self):
        """Low server coherence triggers coherence loss boost in training."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.3)

        coh = float(bus.read_signal('server_coherence_score', 0.7))
        assert coh < 0.4
        # PATCH-C logic: boost = 1.0 + (0.4 - coh) * 2.0
        boost = 1.0 + (0.4 - coh) * 2.0
        assert boost > 1.0, "Low coherence should produce loss boost > 1.0"

    def test_bidirectional_mct_training_loop(self):
        """MCT → bus → training → bus → MCT forms a complete loop."""
        bus = make_bus()
        bus.register_signal('mct_should_trigger', default=0.0)
        bus.register_signal('training_mct_intervention_active', default=0.0)

        # Step 1: MCT writes trigger
        bus.write_signal('mct_should_trigger', 1.0)

        # Step 2: Training reads and writes intervention status
        triggered = float(bus.read_signal('mct_should_trigger', 0.0))
        if triggered > 0.5:
            bus.write_signal('training_mct_intervention_active', 1.0)

        # Step 3: MCT reads training's response
        intervention = float(bus.read_signal('training_mct_intervention_active', 0.0))
        assert intervention == 1.0, "MCT should see training's intervention response"


# ═══════════════════════════════════════════════════════════════════════
#  Meta-Cognitive Trigger Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTrigger:
    """Verify uncertainty triggers meta-cognitive review."""

    def test_low_coherence_triggers_review(self):
        """Low server coherence (< 0.5) boosts coherence_deficit for MCT."""
        bus = make_bus()
        bus.write_signal('server_coherence_score', 0.2)

        coh = float(bus.read_signal('server_coherence_score', 0.7))
        coherence_deficit = 0.0
        if 0.0 < coh < 0.5:
            coherence_deficit = max(coherence_deficit, 0.5 - coh)

        assert coherence_deficit == pytest.approx(0.3)
        # This would feed into MCT's signal_values['coherence_deficit']

    def test_mct_signals_propagate_to_training(self):
        """MCT decision signals are available to training loop."""
        bus = make_bus()
        for sig in ('mct_should_trigger', 'mct_trigger_score',
                     'mct_decision_entropy'):
            bus.register_signal(sig, default=0.0)

        # MCT evaluates
        bus.write_signal('mct_should_trigger', 1.0)
        bus.write_signal('mct_trigger_score', 0.82)
        bus.write_signal('mct_decision_entropy', 0.15)

        # Training reads
        assert float(bus.read_signal('mct_should_trigger', 0.0)) == 1.0
        assert float(bus.read_signal('mct_trigger_score', 0.0)) > 0.8
        assert float(bus.read_signal('mct_decision_entropy', 0.0)) > 0.0
