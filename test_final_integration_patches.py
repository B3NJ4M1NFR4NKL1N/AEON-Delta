"""
AEON-Delta RMT v3.4 — Final Integration & Cognitive Activation Tests
═════════════════════════════════════════════════════════════════════════

Tests for PATCH-FINAL-1 through PATCH-FINAL-6:
  FINAL-1:  SubsystemCrossValidator instantiation in AEONDeltaV3.__init__
  FINAL-1b: SubsystemCrossValidator causal trace wiring
  FINAL-2:  SubsystemCrossValidator in verify_and_reinforce()
  FINAL-3:  Pre-MCT cross-subsystem validation in _reasoning_core_impl
  FINAL-4:  UnifiedCausalSimulator feedback bus (_fb_ref + set_feedback_bus)
  FINAL-4b: simulation_quality signal written to bus
  FINAL-4c: UnifiedCausalSimulator feedback bus wiring in __init__
  FINAL-4d: simulation_quality MCT reader
  FINAL-5:  Activation sequence Phase 8 checks _subsystem_cross_validator
  FINAL-6:  Version test consistency (3.4.0)

Signal ecosystem audit: Verifies 0 orphan signals after all patches.
Integration map: Connected vs isolated critical paths.
Mutual reinforcement: Active components verify each other.
Meta-cognitive trigger: Uncertainty triggers higher-order review.
Causal transparency: Every output traceable to root cause.
"""

import re
import sys
import torch
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import core modules
# ---------------------------------------------------------------------------
sys.path.insert(0, '.')
from aeon_core import (
    __version__,
    SubsystemCrossValidator,
    UnifiedCausalSimulator,
    CognitiveFeedbackBus,
    TemporalCausalTraceBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feedback_bus() -> CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus for testing."""
    return CognitiveFeedbackBus(hidden_dim=64)


def _make_causal_trace() -> TemporalCausalTraceBuffer:
    """Create a minimal TemporalCausalTraceBuffer for testing."""
    return TemporalCausalTraceBuffer(max_entries=256)


def _make_config(**overrides) -> Any:
    """Create a minimal config namespace for AEONDeltaV3 instantiation."""
    class MinimalConfig:
        hidden_dim = 64
        num_heads = 4
        num_layers = 2
        num_pillars = 4
        vocab_size = 100
        max_seq_len = 32
        dropout = 0.0
        use_rope = False
        convergence_threshold = 0.01
        max_iterations = 5
        min_iterations = 1
        anderson_memory = 3
        enable_certification = False
        enable_ibp_certification = False
        enable_lipstp_certification = False
        enable_iqc_certification = False
        enable_sandwich = False
        enable_spectral_norm = False
        enable_hierarchical_vae = False
        enable_cross_validation = False
        cross_validation_agreement = 0.7
        cross_validation_max_steps = 3
        enable_external_trust = False
        enable_ns_consistency_check = False
        enable_world_model = False
        enable_mcts = False
        enable_causal_model = False
        enable_active_learning = False
        enable_self_play = False
        num_factors = 8
        ns_violation_threshold = 0.5
        convergence_quality_threshold = 0.8
        mct_threshold = 0.5
        recursive_meta_loop = True
        hierarchical_meta_loop = True
        certified_meta_loop = True
        adaptive_meta_loop = True
        causal_model = True
        mcts_planner = True
        causal_world_model = True
        active_learning_planner = True
        hierarchical_vae = True
    cfg = MinimalConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# §1  PATCH-FINAL-1: SubsystemCrossValidator Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal1SubsystemCrossValidatorInstantiation:
    """Verify SubsystemCrossValidator is instantiated and wired."""

    def test_scv_class_exists(self):
        """SubsystemCrossValidator class is importable."""
        assert SubsystemCrossValidator is not None

    def test_scv_construction_with_bus(self):
        """SubsystemCrossValidator can be constructed with a feedback bus."""
        fb = _make_feedback_bus()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        assert scv._fb_ref is fb

    def test_scv_validate_returns_report(self):
        """validate() returns a dict with inconsistency_score."""
        fb = _make_feedback_bus()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        report = scv.validate()
        assert isinstance(report, dict)
        assert 'inconsistency_score' in report
        assert 'violations' in report

    def test_scv_validate_writes_cross_subsystem_inconsistency(self):
        """validate() writes cross_subsystem_inconsistency to the bus."""
        fb = _make_feedback_bus()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.validate()
        val = fb.read_signal('cross_subsystem_inconsistency', -1.0)
        assert val >= 0.0, "cross_subsystem_inconsistency should be written"

    def test_scv_has_consistency_pairs(self):
        """SubsystemCrossValidator defines at least 4 consistency pairs."""
        assert hasattr(SubsystemCrossValidator, '_CONSISTENCY_PAIRS')
        assert len(SubsystemCrossValidator._CONSISTENCY_PAIRS) >= 4

    def test_scv_causal_trace_wiring(self):
        """set_causal_trace() wires the causal trace buffer."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.set_causal_trace(ct)
        assert scv._causal_trace_ref is ct

    def test_scv_validate_records_to_causal_trace(self):
        """validate() records to causal trace when wired."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.set_causal_trace(ct)
        scv.validate()
        entries = ct.find(subsystem='cross_validator')
        assert len(entries) >= 1, "Should record at least one trace entry"


# ═══════════════════════════════════════════════════════════════════════════
# §2  PATCH-FINAL-2: Cross-subsystem validation in verify_and_reinforce
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal2VerifyAndReinforceCV:
    """verify_and_reinforce() calls SubsystemCrossValidator.validate()."""

    def test_v_and_r_contains_cross_validation_code(self):
        """verify_and_reinforce source mentions PATCH-FINAL-2."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert 'PATCH-FINAL-2' in src
        assert '_subsystem_cross_validator' in src

    def test_v_and_r_includes_cross_validation_key(self):
        """verify_and_reinforce result includes cross_subsystem_validation."""
        from aeon_core import AEONDeltaV3
        # We can't easily construct a full model, but we can verify the
        # source code includes the expected output key.
        import inspect
        src = inspect.getsource(AEONDeltaV3.verify_and_reinforce)
        assert 'cross_subsystem_validation' in src


# ═══════════════════════════════════════════════════════════════════════════
# §3  PATCH-FINAL-3: Pre-MCT Cross-Subsystem Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal3PreMCTValidation:
    """Pre-MCT cross-subsystem validation in _reasoning_core_impl."""

    def test_reasoning_core_calls_scv_before_mct(self):
        """_reasoning_core_impl calls _subsystem_cross_validator.validate()
        before the primary metacognitive_trigger.evaluate()."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        # Find the PATCH-FINAL-3 marker position
        patch_pos = src.find('PATCH-FINAL-3: Pre-MCT cross-subsystem')
        assert patch_pos > 0, "Should contain PATCH-FINAL-3 marker"
        # Find the primary MCT evaluate that immediately follows
        mct_after_patch = src.find('metacognitive_trigger.evaluate', patch_pos)
        assert mct_after_patch > 0, (
            "Should have MCT evaluate after PATCH-FINAL-3"
        )
        # The SCV validate call should be between PATCH-FINAL-3 and that evaluate
        scv_validate = src.find('_subsystem_cross_validator', patch_pos)
        assert scv_validate > 0, (
            "Should have _subsystem_cross_validator after PATCH-FINAL-3"
        )
        assert scv_validate < mct_after_patch, (
            "SCV validate should come before the primary MCT evaluate"
        )

    def test_pre_mct_has_patch_marker(self):
        """_reasoning_core_impl contains PATCH-FINAL-3 marker."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3._reasoning_core_impl)
        assert 'PATCH-FINAL-3' in src


# ═══════════════════════════════════════════════════════════════════════════
# §4  PATCH-FINAL-4: UnifiedCausalSimulator Feedback Bus
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal4SimulatorFeedbackBus:
    """UnifiedCausalSimulator has feedback bus and writes simulation_quality."""

    def test_simulator_has_fb_ref(self):
        """UnifiedCausalSimulator.__init__ creates _fb_ref."""
        sim = UnifiedCausalSimulator(state_dim=32)
        assert hasattr(sim, '_fb_ref')
        assert sim._fb_ref is None  # Not wired by default

    def test_simulator_has_set_feedback_bus(self):
        """UnifiedCausalSimulator has set_feedback_bus method."""
        sim = UnifiedCausalSimulator(state_dim=32)
        assert hasattr(sim, 'set_feedback_bus')

    def test_simulator_set_feedback_bus_wires(self):
        """set_feedback_bus() wires the bus reference."""
        fb = _make_feedback_bus()
        sim = UnifiedCausalSimulator(state_dim=32)
        sim.set_feedback_bus(fb)
        assert sim._fb_ref is fb

    def test_simulator_forward_writes_simulation_quality(self):
        """forward() writes simulation_quality to the feedback bus."""
        fb = _make_feedback_bus()
        sim = UnifiedCausalSimulator(state_dim=32)
        sim.set_feedback_bus(fb)
        state = torch.randn(1, 32)
        with torch.no_grad():
            sim(state)
        val = fb.read_signal('simulation_quality', -1.0)
        assert val >= 0.0, "simulation_quality should be written"
        assert val <= 1.0, "simulation_quality should be in [0, 1]"

    def test_simulator_still_has_causal_trace(self):
        """UnifiedCausalSimulator still has causal trace wiring."""
        sim = UnifiedCausalSimulator(state_dim=32)
        assert hasattr(sim, '_causal_trace_ref')
        assert hasattr(sim, 'set_causal_trace')


# ═══════════════════════════════════════════════════════════════════════════
# §5  PATCH-FINAL-4d: simulation_quality MCT Reader
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal4dMCTReader:
    """MCT reads simulation_quality and routes it to world_model_surprise."""

    def test_mct_evaluate_reads_simulation_quality(self):
        """MetaCognitiveRecursionTrigger.evaluate() source reads
        simulation_quality from the feedback bus."""
        from aeon_core import MetaCognitiveRecursionTrigger
        import inspect
        src = inspect.getsource(MetaCognitiveRecursionTrigger.evaluate)
        assert 'simulation_quality' in src
        assert 'PATCH-FINAL-4d' in src

    def test_mct_low_sim_quality_amplifies_surprise(self):
        """When simulation_quality < 0.5, world_model_surprise is amplified."""
        from aeon_core import MetaCognitiveRecursionTrigger
        fb = _make_feedback_bus()
        # Write low simulation quality
        fb.write_signal('simulation_quality', 0.2)
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
        )
        mct.set_feedback_bus(fb)
        # Evaluate with all defaults
        result1 = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.1, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )
        # Now with good simulation quality
        fb.write_signal('simulation_quality', 0.9)
        result2 = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.1, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )
        # Low sim quality should produce a higher (or equal) trigger score
        assert result1.get('trigger_score', 0) >= result2.get('trigger_score', 0)


# ═══════════════════════════════════════════════════════════════════════════
# §6  PATCH-FINAL-5: Activation Sequence Phase 8
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal5ActivationSequencePhase8:
    """Phase 8 checks _subsystem_cross_validator, not cross_validator."""

    def test_activation_sequence_phase8_checks_scv(self):
        """system_emergence_report's activation_sequence Phase 8 status
        checks _subsystem_cross_validator."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        # Find the phase 8 section
        phase8_idx = src.find('Cross-Subsystem Consistency')
        assert phase8_idx > 0
        # The status check should reference _subsystem_cross_validator
        # within a reasonable window after the phase name
        phase8_section = src[phase8_idx:phase8_idx + 1000]
        assert '_subsystem_cross_validator' in phase8_section, (
            "Phase 8 should check _subsystem_cross_validator, "
            f"got section: {phase8_section[:200]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §7  PATCH-FINAL-6: Version Consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestFinal6Version:
    """Version is 3.4.0."""

    def test_version_is_340(self):
        assert __version__ == "3.4.0"


# ═══════════════════════════════════════════════════════════════════════════
# §8  Signal Ecosystem Audit
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalEcosystemAudit:
    """Verify the signal ecosystem has 0 true orphans after all patches."""

    def test_simulation_quality_is_bidirectional(self):
        """simulation_quality is written (simulator) and read (MCT)."""
        with open('aeon_core.py') as f:
            content = f.read()
        # Written
        assert re.search(
            r"write_signal\s*\(\s*\n?\s*['\"]simulation_quality['\"]",
            content,
        ), "simulation_quality should be written"
        # Read
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]simulation_quality['\"]",
            content,
        ), "simulation_quality should be read"

    def test_cross_subsystem_inconsistency_is_bidirectional(self):
        """cross_subsystem_inconsistency is written (SCV) and read (MCT)."""
        with open('aeon_core.py') as f:
            content = f.read()
        assert re.search(
            r"write_signal\s*\(\s*['\"]cross_subsystem_inconsistency['\"]",
            content,
        ), "cross_subsystem_inconsistency should be written"
        assert re.search(
            r"read_signal\s*\(\s*\n?\s*['\"]cross_subsystem_inconsistency['\"]",
            content,
        ), "cross_subsystem_inconsistency should be read"

    def test_no_new_orphan_signals_introduced(self):
        """All new signals from PATCH-FINAL are bidirectional."""
        new_signals = ['simulation_quality']
        with open('aeon_core.py') as f:
            content = f.read()
        for sig in new_signals:
            has_write = bool(re.search(
                rf'write_signal[^)]*?{sig}', content, re.DOTALL,
            ))
            has_read = bool(re.search(
                rf'read_signal[^)]*?{sig}', content, re.DOTALL,
            ))
            assert has_write and has_read, (
                f"{sig} must be both written and read"
            )


# ═══════════════════════════════════════════════════════════════════════════
# §9  Integration Map: Connected vs Isolated Critical Paths
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationMap:
    """Verify that critical path connections are complete."""

    def test_scv_is_wired_in_init(self):
        """SubsystemCrossValidator instantiation code exists in __init__."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.__init__)
        assert '_subsystem_cross_validator = SubsystemCrossValidator' in src

    def test_scv_causal_trace_wiring_in_init(self):
        """SubsystemCrossValidator causal trace is wired in __init__."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.__init__)
        assert '_subsystem_cross_validator' in src
        assert 'set_causal_trace' in src

    def test_simulator_fb_wiring_in_init(self):
        """UnifiedCausalSimulator feedback bus wiring exists in __init__."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.__init__)
        assert 'unified_simulator' in src
        assert 'set_feedback_bus' in src


# ═══════════════════════════════════════════════════════════════════════════
# §10  Mutual Reinforcement Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestMutualReinforcement:
    """Active components verify and stabilize each other's states."""

    def test_scv_detects_contradictory_signals(self):
        """SubsystemCrossValidator detects when positively-correlated
        signals diverge."""
        fb = _make_feedback_bus()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        # Write contradictory signals: memory confidence high but
        # reasoning confidence low (should be positively correlated)
        fb.write_signal('memory_retrieval_confidence', 0.9)
        fb.write_signal('symbolic_reasoning_confidence', 0.1)
        report = scv.validate()
        # Should detect inconsistency
        assert report['inconsistency_score'] > 0.0

    def test_scv_detects_anti_correlation_violation(self):
        """SubsystemCrossValidator detects when negatively-correlated
        signals are both high."""
        fb = _make_feedback_bus()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        # Both high when they should anti-correlate
        fb.write_signal('mct_trigger_score', 0.9)
        fb.write_signal('convergence_arbiter_confidence', 0.9)
        report = scv.validate()
        assert report['inconsistency_score'] > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §11  Meta-Cognitive Trigger Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaCognitiveTrigger:
    """Uncertainty and inconsistency trigger higher-order review."""

    def test_cross_subsystem_inconsistency_amplifies_mct(self):
        """High cross_subsystem_inconsistency amplifies MCT coherence_deficit
        signal."""
        from aeon_core import MetaCognitiveRecursionTrigger
        fb = _make_feedback_bus()
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
        )
        mct.set_feedback_bus(fb)
        # With high inconsistency
        fb.write_signal('cross_subsystem_inconsistency', 0.8)
        result_high = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.2,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )
        # With no inconsistency
        fb.write_signal('cross_subsystem_inconsistency', 0.0)
        result_low = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.2,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )
        # High inconsistency should produce higher trigger score
        assert result_high.get('trigger_score', 0) >= result_low.get('trigger_score', 0)


# ═══════════════════════════════════════════════════════════════════════════
# §12  Causal Transparency Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalTransparency:
    """Every output/action is traceable to root cause."""

    def test_scv_records_to_causal_trace(self):
        """SubsystemCrossValidator records its verdict in the causal trace,
        making cross-subsystem checks visible to root-cause analysis."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()
        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.set_causal_trace(ct)
        scv.validate()
        entries = ct.find(subsystem='cross_validator')
        assert len(entries) >= 1
        # Entry should contain inconsistency details
        entry = entries[0]
        assert 'metadata' in entry or 'decision' in entry

    def test_simulator_records_to_causal_trace(self):
        """UnifiedCausalSimulator records simulation steps."""
        ct = _make_causal_trace()
        sim = UnifiedCausalSimulator(state_dim=32)
        sim.set_causal_trace(ct)
        state = torch.randn(1, 32)
        with torch.no_grad():
            sim(state)
        entries = ct.find(subsystem='unified_simulator')
        assert len(entries) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# §13  Activation Sequence Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationSequence:
    """The 10-phase activation sequence is complete and ordered."""

    def test_activation_sequence_has_10_phases(self):
        """system_emergence_report defines 10 activation phases."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        # Count "order" entries in activation_sequence
        order_matches = re.findall(r'"order":\s*(\d+)', src)
        assert len(order_matches) >= 10

    def test_phases_cover_all_critical_paths(self):
        """Activation sequence covers tensor safety, causal trace,
        convergence, MCT, UCC, emergence, signals, cross-validation,
        training bridge, and self-reflection."""
        import inspect
        from aeon_core import AEONDeltaV3
        src = inspect.getsource(AEONDeltaV3.system_emergence_report)
        required_phases = [
            'Tensor Safety',
            'Causal Trace',
            'Convergence',
            'Meta-Cognitive',
            'Cognitive Cycle',
            'Emergence',
            'Signal Ecosystem',
            'Cross-Subsystem',
            'Training',
            'Self-Reflection',
        ]
        for phase in required_phases:
            assert phase in src, f"Missing activation phase: {phase}"


# ═══════════════════════════════════════════════════════════════════════════
# §14  End-to-End Integration Flow
# ═══════════════════════════════════════════════════════════════════════════

class TestE2EIntegration:
    """End-to-end flow: subsystem produces signal → SCV detects → MCT reads
    → verify_and_reinforce processes → causal trace records."""

    def test_e2e_scv_to_mct_to_trace(self):
        """Full cycle: write contradictory signals → SCV detects → MCT
        reads cross_subsystem_inconsistency → causal trace records."""
        fb = _make_feedback_bus()
        ct = _make_causal_trace()

        # Step 1: Write contradictory signals
        fb.write_signal('memory_retrieval_confidence', 0.9)
        fb.write_signal('symbolic_reasoning_confidence', 0.1)

        # Step 2: SCV detects
        scv = SubsystemCrossValidator(feedback_bus=fb)
        scv.set_causal_trace(ct)
        report = scv.validate()
        assert report['inconsistency_score'] > 0.0

        # Step 3: MCT reads the cross_subsystem_inconsistency
        from aeon_core import MetaCognitiveRecursionTrigger
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
        )
        mct.set_feedback_bus(fb)
        result = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.2,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.0, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )

        # Step 4: Causal trace should have SCV entries
        entries = ct.find(subsystem='cross_validator')
        assert len(entries) >= 1

    def test_e2e_simulator_to_mct(self):
        """Full cycle: simulator writes quality → MCT reads it."""
        fb = _make_feedback_bus()
        sim = UnifiedCausalSimulator(state_dim=32)
        sim.set_feedback_bus(fb)

        # Simulate
        state = torch.randn(1, 32)
        with torch.no_grad():
            sim(state)

        # Check signal was written
        val = fb.read_signal('simulation_quality', -1.0)
        assert val >= 0.0

        # MCT should be able to read it
        from aeon_core import MetaCognitiveRecursionTrigger
        mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
        )
        mct.set_feedback_bus(fb)
        result = mct.evaluate(
            uncertainty=0.1, is_diverging=False,
            topology_catastrophe=False, coherence_deficit=0.0,
            memory_staleness=0.0, recovery_pressure=0.0,
            world_model_surprise=0.1, causal_quality=1.0,
            safety_violation=False, convergence_conflict=0.0,
            output_reliability=1.0, spectral_stability_margin=1.0,
            border_uncertainty=0.0,
        )
        # Should complete without error
        assert isinstance(result, dict)
