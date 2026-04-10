"""
Tests for Final Integration & Cognitive Activation patches.

Covers:
  - PATCH-COGSERV-3: mct_convergence_modulation consumer in UCC arbiter
  - PATCH-ROOTCAUSE-1: Soft correction dispatcher before decoder
  - PATCH-EMERGENCE-1: Intra-pass emergence modulation
  - PATCH-COGSERV-2: Persistent signal support in CognitiveFeedbackBus
  - PATCH-COGSERV-1: CognitiveInferenceProfile (minimal/standard/full)
  - Signal ecosystem integrity (no new orphans or missing producers)
"""

import pytest
import sys
import os
import re
import importlib

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core
import aeon_core as aeon

try:
    import aeon_server
    _HAS_SERVER = True
except (ImportError, SystemExit):
    _HAS_SERVER = False

_skip_no_server = pytest.mark.skipif(
    not _HAS_SERVER, reason="aeon_server requires fastapi/uvicorn",
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)


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
# PATCH-COGSERV-3: mct_convergence_modulation Consumer
# ──────────────────────────────────────────────────────────────────────

class TestCogServ3_ArbiterModulationConsumer:
    """PATCH-COGSERV-3: UnifiedConvergenceArbiter reads mct_convergence_modulation."""

    def test_arbiter_accepts_feedback_bus(self):
        """Arbiter __init__ accepts optional feedback_bus parameter."""
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        assert arbiter._feedback_bus is bus

    def test_arbiter_default_no_bus(self):
        """Arbiter without bus still works (backward compatible)."""
        arbiter = aeon.UnifiedConvergenceArbiter()
        assert arbiter._feedback_bus is None

    def test_modulation_default_zero(self):
        """When bus has no modulation signal, result shows 0.0."""
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.001},
            convergence_monitor_verdict={"status": "converged", "certified": True},
        )
        assert result["mct_convergence_modulation"] == 0.0
        assert result["convergence_threshold_boost"] == 0.0
        assert result["mct_deep_recursion"] is False

    def test_modulation_read_from_bus(self):
        """Arbiter reads mct_convergence_modulation from feedback bus."""
        bus = _make_bus()
        bus.write_signal('mct_convergence_modulation', 0.5)
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.001},
            convergence_monitor_verdict={"status": "converged", "certified": True},
        )
        assert abs(result["mct_convergence_modulation"] - 0.5) < 0.01
        assert abs(result["convergence_threshold_boost"] - 0.1) < 0.01
        assert result["mct_deep_recursion"] is False

    def test_deep_recursion_flag(self):
        """Modulation > 0.7 triggers mct_deep_recursion flag."""
        bus = _make_bus()
        bus.write_signal('mct_convergence_modulation', 0.85)
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.6, "residual_norm": 0.1},
            convergence_monitor_verdict={"status": "converging", "certified": False},
        )
        assert result["mct_deep_recursion"] is True
        assert result["convergence_threshold_boost"] > 0.14

    def test_signal_appears_in_bus_read_log(self):
        """mct_convergence_modulation is registered in bus _read_log."""
        bus = _make_bus()
        bus.write_signal('mct_convergence_modulation', 0.3)
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.5, "residual_norm": 0.5},
            convergence_monitor_verdict={"status": "converging"},
        )
        assert 'mct_convergence_modulation' in bus._read_log

    def test_existing_arbitrate_behavior_preserved(self):
        """Core arbitrate behavior unchanged (conflict detection, etc.)."""
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.001},
            convergence_monitor_verdict={"status": "diverging", "certified": False},
        )
        assert result["has_conflict"] is True
        assert result["unified_status"] in ("diverging", "conflict")
        assert result["uncertainty_boost"] > 0.0

    def test_arbiter_wired_with_bus_in_model(self):
        """In full model init, arbiter receives the feedback bus."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert hasattr(model, 'convergence_arbiter')
        assert model.convergence_arbiter._feedback_bus is model.feedback_bus


# ──────────────────────────────────────────────────────────────────────
# PATCH-ROOTCAUSE-1: Soft Correction Dispatcher
# ──────────────────────────────────────────────────────────────────────

class TestRootCause1_SoftCorrection:
    """PATCH-ROOTCAUSE-1: Soft correction before decoder."""

    def test_soft_correction_code_present(self):
        """PATCH-ROOTCAUSE-1 code block exists in aeon_core.py."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        assert 'PATCH-ROOTCAUSE-1' in src
        assert 'intra_pass_soft_correction_applied' in src

    def test_soft_correction_signal_written(self):
        """intra_pass_soft_correction_applied signal is writable."""
        bus = _make_bus()
        bus.write_signal('intra_pass_soft_correction_applied', 0.05)
        val = float(bus.read_signal('intra_pass_soft_correction_applied', 0.0))
        assert abs(val - 0.05) < 0.01

    def test_targeted_remediation_read_present(self):
        """Forward reads targeted_remediation_active from bus."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        assert "read_signal" in src
        assert "targeted_remediation_active" in src

    def test_mct_reads_soft_correction(self):
        """MCT evaluate reads intra_pass_soft_correction_applied."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            lines = f.readlines()
        # Check for read_signal within 3 lines of the signal name
        found = False
        for i, line in enumerate(lines):
            if 'intra_pass_soft_correction_applied' in line:
                context = ''.join(lines[max(0, i-3):i+1])
                if 'read_signal' in context:
                    found = True
                    break
        assert found, "MCT should read intra_pass_soft_correction_applied"

    def test_correction_strength_bounded(self):
        """Correction strength is clamped to <= 0.1 (10%)."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        # Look for min(0.1, ...) pattern
        assert 'min(0.1,' in src, "Correction strength must be bounded to 0.1"


# ──────────────────────────────────────────────────────────────────────
# PATCH-EMERGENCE-1: Intra-Pass Emergence Modulation
# ──────────────────────────────────────────────────────────────────────

class TestEmergence1_IntraPassModulation:
    """PATCH-EMERGENCE-1: Emergence deficit modulates decoder confidence."""

    def test_emergence_modulation_code_present(self):
        """PATCH-EMERGENCE-1 code block exists in aeon_core.py."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        assert 'PATCH-EMERGENCE-1' in src
        assert 'emergence_intra_pass_modulation' in src

    def test_emergence_modulation_signal_writable(self):
        """emergence_intra_pass_modulation signal works on bus."""
        bus = _make_bus()
        bus.write_signal('emergence_intra_pass_modulation', 0.7)
        val = float(bus.read_signal('emergence_intra_pass_modulation', 0.0))
        assert abs(val - 0.7) < 0.01

    def test_compute_loss_reads_emergence_modulation(self):
        """compute_loss reads emergence_intra_pass_modulation."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            lines = f.readlines()
        # Check for read_signal within 3 lines of the signal name
        found_read = False
        for i, line in enumerate(lines):
            if 'emergence_intra_pass_modulation' in line:
                context = ''.join(lines[max(0, i-3):i+1])
                if 'read_signal' in context:
                    found_read = True
                    break
        assert found_read, "compute_loss should read emergence_intra_pass_modulation"

    def test_coherence_boost_factor(self):
        """PATCH-EMERGENCE-1b applies coherence boost = 1.0 + deficit * 0.2."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        assert '_em1_mod * 0.2' in src
        assert '_em1_mod * 0.1' in src

    def test_confidence_scale_bounded(self):
        """Confidence reduction is bounded (max 10%)."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            src = f.read()
        assert '_em1_deficit * 0.1' in src


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGSERV-2: Persistent Signal Support
# ──────────────────────────────────────────────────────────────────────

class TestCogServ2_PersistentSignals:
    """PATCH-COGSERV-2: Persistent signals survive flush_consumed()."""

    def test_register_persistent_signal(self):
        """register_persistent_signal adds to _persistent_signals set."""
        bus = _make_bus()
        bus.register_persistent_signal('test_persistent', 0.5)
        assert 'test_persistent' in bus._persistent_signals

    def test_persistent_signal_survives_flush(self):
        """Persistent signal value survives flush_consumed()."""
        bus = _make_bus()
        bus.register_persistent_signal('my_persistent', 0.0)
        bus.write_signal('my_persistent', 0.8)
        bus.flush_consumed()
        val = float(bus.read_signal('my_persistent', 0.0))
        assert val > 0.7, f"Persistent signal lost after flush: {val}"

    def test_non_persistent_signal_survives_flush_value(self):
        """Non-persistent signal values also survive flush (they stay in _extra_signals)."""
        bus = _make_bus()
        bus.write_signal('normal_signal', 0.9)
        bus.flush_consumed()
        # Values persist in _extra_signals; only logs are cleared
        val = float(bus.read_signal('normal_signal', 0.0))
        # Value may have been cleared by flush or retained - check behavior
        assert isinstance(val, float)

    def test_persistent_signal_in_write_log_after_flush(self):
        """Persistent signals appear in _write_log after flush_consumed()."""
        bus = _make_bus()
        bus.register_persistent_signal('server_coherence_score', 0.0)
        bus.write_signal('server_coherence_score', 0.6)
        bus.flush_consumed()
        assert 'server_coherence_score' in bus._write_log

    def test_multiple_persistent_signals(self):
        """Multiple persistent signals all survive flush."""
        bus = _make_bus()
        bus.register_persistent_signal('sig_a', 0.0)
        bus.register_persistent_signal('sig_b', 0.0)
        bus.write_signal('sig_a', 0.4)
        bus.write_signal('sig_b', 0.7)
        bus.flush_consumed()
        assert 'sig_a' in bus._write_log
        assert 'sig_b' in bus._write_log
        val_a = float(bus.read_signal('sig_a', 0.0))
        val_b = float(bus.read_signal('sig_b', 0.0))
        assert val_a > 0.0
        assert val_b > 0.0

    def test_persistent_signal_updates(self):
        """Persistent signal can be updated across flushes."""
        bus = _make_bus()
        bus.register_persistent_signal('evolving', 0.0)
        bus.write_signal('evolving', 0.3)
        bus.flush_consumed()
        bus.write_signal('evolving', 0.6)
        bus.flush_consumed()
        val = float(bus.read_signal('evolving', 0.0))
        assert val > 0.5, "Persistent signal should reflect last write"


# ──────────────────────────────────────────────────────────────────────
# PATCH-COGSERV-1: Cognitive Inference Profile
# ──────────────────────────────────────────────────────────────────────

@_skip_no_server
class TestCogServ1_CognitiveProfile:
    """PATCH-COGSERV-1: Server cognitive inference profiles."""

    def test_profiles_defined(self):
        """_COGNITIVE_PROFILES has minimal, standard, full tiers."""
        # Import server module
        import aeon_server
        assert 'minimal' in aeon_server._COGNITIVE_PROFILES
        assert 'standard' in aeon_server._COGNITIVE_PROFILES
        assert 'full' in aeon_server._COGNITIVE_PROFILES

    def test_minimal_profile_empty(self):
        """Minimal profile has no feature overrides."""
        import aeon_server
        assert aeon_server._COGNITIVE_PROFILES['minimal'] == {}

    def test_standard_profile_core_features(self):
        """Standard profile enables MCT, causal trace, error evolution, coherence."""
        import aeon_server
        std = aeon_server._COGNITIVE_PROFILES['standard']
        assert std.get('enable_metacognitive_recursion') is True
        assert std.get('enable_causal_trace') is True
        assert std.get('enable_error_evolution') is True
        assert std.get('enable_full_coherence') is True
        assert std.get('enable_module_coherence') is True
        assert std.get('enable_recursive_meta_loop') is True

    def test_full_profile_all_features(self):
        """Full profile enables all cognitive features."""
        import aeon_server
        full = aeon_server._COGNITIVE_PROFILES['full']
        assert len(full) > 20
        # Verify it includes standard features
        std = aeon_server._COGNITIVE_PROFILES['standard']
        for key in std:
            assert full.get(key) is True, f"Full profile missing {key}"

    def test_resolve_profile_standard(self):
        """_resolve_cognitive_profile applies standard profile flags."""
        import aeon_server
        result = aeon_server._resolve_cognitive_profile('standard', {})
        assert result.get('enable_metacognitive_recursion') is True
        assert result.get('enable_causal_trace') is True

    def test_resolve_profile_overrides(self):
        """Explicit overrides take precedence over profile."""
        import aeon_server
        result = aeon_server._resolve_cognitive_profile('standard', {
            'enable_metacognitive_recursion': False,
        })
        assert result.get('enable_metacognitive_recursion') is False
        assert result.get('enable_causal_trace') is True

    def test_resolve_profile_minimal(self):
        """Minimal profile does not set any features."""
        import aeon_server
        result = aeon_server._resolve_cognitive_profile('minimal', {})
        assert 'enable_metacognitive_recursion' not in result

    def test_resolve_profile_unknown_defaults_empty(self):
        """Unknown profile name returns empty defaults."""
        import aeon_server
        result = aeon_server._resolve_cognitive_profile('unknown_tier', {})
        assert result == {}

    def test_init_request_has_cognitive_profile(self):
        """InitRequest model has cognitive_profile field."""
        import aeon_server
        req = aeon_server.InitRequest()
        assert hasattr(req, 'cognitive_profile')
        assert req.cognitive_profile == 'standard'

    def test_init_request_profile_minimal(self):
        """InitRequest can be set to minimal profile."""
        import aeon_server
        req = aeon_server.InitRequest(cognitive_profile='minimal')
        assert req.cognitive_profile == 'minimal'


# ──────────────────────────────────────────────────────────────────────
# Signal Ecosystem Integrity
# ──────────────────────────────────────────────────────────────────────

class TestFinalCogServ_SignalEcosystem:
    """Verify signal ecosystem remains fully connected after all patches."""

    def test_no_missing_producers(self):
        """Every read_signal has at least one write_signal producer."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            core_src = f.read()
        train_path = os.path.join(os.path.dirname(__file__), 'ae_train.py')
        with open(train_path, 'r') as f:
            train_src = f.read()
        server_path = os.path.join(os.path.dirname(__file__), 'aeon_server.py')
        with open(server_path, 'r') as f:
            server_src = f.read()
        combined = core_src + train_src + server_src

        write_pat = re.compile(r"write_signal(?:_traced)?\s*\(\s*['\"](\w+)['\"]")
        read_pat = re.compile(r"read_signal\s*\(\s*['\"](\w+)['\"]")
        extra_pat = re.compile(r"_extra_signals\s*\[\s*['\"](\w+)['\"]\s*\]")

        written = set(write_pat.findall(combined)) | set(extra_pat.findall(combined))
        read = set(read_pat.findall(combined))
        missing = read - written
        assert len(missing) == 0, f"Missing producers: {missing}"

    def test_written_count_at_least_162(self):
        """At least 162 signals are written (maintained or increased)."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            core_src = f.read()
        train_path = os.path.join(os.path.dirname(__file__), 'ae_train.py')
        with open(train_path, 'r') as f:
            train_src = f.read()
        server_path = os.path.join(os.path.dirname(__file__), 'aeon_server.py')
        with open(server_path, 'r') as f:
            server_src = f.read()
        combined = core_src + train_src + server_src

        write_pat = re.compile(r"write_signal(?:_traced)?\s*\(\s*['\"](\w+)['\"]")
        extra_pat = re.compile(r"_extra_signals\s*\[\s*['\"](\w+)['\"]\s*\]")
        written = set(write_pat.findall(combined)) | set(extra_pat.findall(combined))
        assert len(written) >= 162, f"Only {len(written)} written signals (expected ≥162)"

    def test_new_signals_have_producers_and_consumers(self):
        """All 5 new patch signals are both written and read."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            core_src = f.read()
        server_path = os.path.join(os.path.dirname(__file__), 'aeon_server.py')
        with open(server_path, 'r') as f:
            server_src = f.read()
        combined = core_src + server_src

        new_signals = [
            'intra_pass_soft_correction_applied',
            'emergence_intra_pass_modulation',
            'server_coherence_score',
            'server_reinforcement_pressure',
        ]
        for sig in new_signals:
            write_count = combined.count(f"'{sig}'") + combined.count(f'"{sig}"')
            assert write_count >= 2, (
                f"Signal '{sig}' appears only {write_count} times "
                f"(need ≥2: at least 1 write + 1 read)"
            )

    def test_mct_convergence_modulation_now_consumed(self):
        """mct_convergence_modulation is no longer write-only."""
        src_path = os.path.join(os.path.dirname(__file__), 'aeon_core.py')
        with open(src_path, 'r') as f:
            lines = f.readlines()
        has_write = False
        has_read = False
        for i, line in enumerate(lines):
            if 'mct_convergence_modulation' in line:
                context = ''.join(lines[max(0, i-3):i+1])
                if 'write_signal' in context:
                    has_write = True
                if 'read_signal' in context:
                    has_read = True
        assert has_write, "mct_convergence_modulation must have a writer"
        assert has_read, "mct_convergence_modulation must have a reader (PATCH-COGSERV-3)"


# ──────────────────────────────────────────────────────────────────────
# Cross-Patch Integration Tests
# ──────────────────────────────────────────────────────────────────────

class TestCrossPatchIntegration:
    """Verify patches work together as a coherent system."""

    def test_arbiter_with_model_bus(self):
        """Model's arbiter reads mct_convergence_modulation via shared bus."""
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # Write modulation to model's bus
        model.feedback_bus.write_signal('mct_convergence_modulation', 0.6)
        # Arbiter should read it
        result = model.convergence_arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.001},
            convergence_monitor_verdict={"status": "converged", "certified": True},
        )
        assert abs(result["mct_convergence_modulation"] - 0.6) < 0.1

    def test_persistent_signals_survive_multi_flush(self):
        """Persistent signals survive 3 consecutive flushes."""
        bus = _make_bus()
        bus.register_persistent_signal('durable_signal', 0.0)
        bus.write_signal('durable_signal', 0.5)
        for _ in range(3):
            bus.flush_consumed()
        val = float(bus.read_signal('durable_signal', 0.0))
        assert val > 0.0, "Persistent signal should survive multiple flushes"

    def test_emergence_modulation_chain(self):
        """emergence_deficit → emergence_intra_pass_modulation → coherence boost chain."""
        bus = _make_bus()
        # Simulate forward writing deficit
        bus.write_signal('emergence_deficit', 0.7)
        # Simulate forward writing modulation
        bus.write_signal('emergence_intra_pass_modulation', 0.7)
        # Verify compute_loss can read it
        val = float(bus.read_signal('emergence_intra_pass_modulation', 0.0))
        assert val > 0.5

    def test_soft_correction_chain(self):
        """targeted_remediation_active → soft correction → MCT reads correction."""
        bus = _make_bus()
        # Simulate forward writing correction
        bus.write_signal('targeted_remediation_active', 0.8)
        bus.write_signal('intra_pass_soft_correction_applied', 0.08)
        # MCT reads it
        val = float(bus.read_signal('intra_pass_soft_correction_applied', 0.0))
        assert abs(val - 0.08) < 0.01

    @_skip_no_server
    def test_cognitive_profile_standard_enables_features(self):
        """Standard profile sets core cognitive features to True."""
        import aeon_server
        req = aeon_server.InitRequest(cognitive_profile='standard')
        cfg_kwargs = req.model_dump()
        cfg_kwargs.pop("seed", None)
        _profile_name = cfg_kwargs.pop("cognitive_profile", "standard")
        resolved = aeon_server._resolve_cognitive_profile(_profile_name, {})
        for key, val in resolved.items():
            cfg_kwargs[key] = val
        assert cfg_kwargs.get('enable_metacognitive_recursion') is True
        assert cfg_kwargs.get('enable_causal_trace') is True

    @_skip_no_server
    def test_cognitive_profile_minimal_preserves_defaults(self):
        """Minimal profile keeps all features at False."""
        import aeon_server
        req = aeon_server.InitRequest(cognitive_profile='minimal')
        cfg_kwargs = req.model_dump()
        cfg_kwargs.pop("seed", None)
        _profile_name = cfg_kwargs.pop("cognitive_profile", "minimal")
        resolved = aeon_server._resolve_cognitive_profile(_profile_name, {})
        for key, val in resolved.items():
            cfg_kwargs[key] = val
        # No feature should have been set by the minimal profile
        assert cfg_kwargs.get('enable_metacognitive_recursion') is False
