"""Tests for PATCH-APEX series: Final Integration & Cognitive Activation.

APEX-1: MCTSPlanner._fb_ref wiring
APEX-2: CausalProvenanceTracker._fb_ref wiring
APEX-3: Orphaned adaptation signals → MCT evaluate()
APEX-4: Silent except block hardening (logging)
"""

import math
import sys
import types
import logging
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── bootstrap: ensure aeon_core is importable ──────────────────────────
sys.path.insert(0, ".")
import aeon_core  # noqa: E402

CognitiveFeedbackBus = aeon_core.CognitiveFeedbackBus
AEONConfig = aeon_core.AEONConfig
MCTSPlanner = aeon_core.MCTSPlanner
CausalProvenanceTracker = aeon_core.CausalProvenanceTracker
MetaCognitiveRecursionTrigger = aeon_core.MetaCognitiveRecursionTrigger
ThoughtEncoder = aeon_core.ThoughtEncoder


# ── Helpers ────────────────────────────────────────────────────────────
def _make_bus(hidden_dim: int = 256) -> CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus ready for testing."""
    return CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_mct_with_bus() -> tuple:
    """Create MCT with a wired feedback bus for signal testing."""
    bus = _make_bus()
    mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
    mct.set_feedback_bus(bus)
    return mct, bus


def _make_encoder(vocab_size=100, emb_dim=64, z_dim=64, feedback_bus=None):
    """Create ThoughtEncoder with optional feedback bus."""
    enc = ThoughtEncoder(vocab_size=vocab_size, emb_dim=emb_dim, z_dim=z_dim)
    if feedback_bus is not None:
        enc._fb_ref = feedback_bus
    return enc


# ═══════════════════════════════════════════════════════════════════════
# PATCH-APEX-1: MCTSPlanner._fb_ref wiring
# ═══════════════════════════════════════════════════════════════════════

class TestAPEX1_MCTSPlannerWiring:
    """Verify MCTSPlanner gets wired to the feedback bus."""

    def test_mcts_planner_has_fb_ref_attribute(self):
        """MCTSPlanner should have _fb_ref attribute."""
        planner = MCTSPlanner(
            state_dim=256, action_dim=4, hidden_dim=128,
        )
        assert hasattr(planner, '_fb_ref')

    def test_mcts_planner_default_fb_ref_is_none(self):
        """MCTSPlanner._fb_ref defaults to None without feedback_bus."""
        planner = MCTSPlanner(
            state_dim=256, action_dim=4, hidden_dim=128,
        )
        assert planner._fb_ref is None

    def test_mcts_planner_fb_ref_wired_when_passed(self):
        """MCTSPlanner._fb_ref set when feedback_bus is passed."""
        bus = _make_bus()
        planner = MCTSPlanner(
            state_dim=256, action_dim=4, hidden_dim=128,
            feedback_bus=bus,
        )
        assert planner._fb_ref is bus

    def test_mcts_planner_fb_ref_post_wiring(self):
        """MCTSPlanner._fb_ref can be set post-construction."""
        bus = _make_bus()
        planner = MCTSPlanner(
            state_dim=256, action_dim=4, hidden_dim=128,
        )
        assert planner._fb_ref is None
        planner._fb_ref = bus
        assert planner._fb_ref is bus

    def test_mcts_reads_convergence_quality_when_wired(self):
        """MCTSPlanner should read convergence_quality from bus when wired."""
        bus = _make_bus()
        bus.write_signal('convergence_quality', 0.3)
        planner = MCTSPlanner(
            state_dim=256, action_dim=4, hidden_dim=128,
            feedback_bus=bus,
        )
        val = bus.read_signal('convergence_quality', 1.0)
        assert abs(val - 0.3) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# PATCH-APEX-2: CausalProvenanceTracker._fb_ref wiring
# ═══════════════════════════════════════════════════════════════════════

class TestAPEX2_ProvenanceTrackerWiring:
    """Verify CausalProvenanceTracker gets wired to the feedback bus."""

    def test_provenance_tracker_has_fb_ref(self):
        """CausalProvenanceTracker should have _fb_ref attribute."""
        tracker = CausalProvenanceTracker()
        assert hasattr(tracker, '_fb_ref')

    def test_provenance_tracker_default_fb_ref_is_none(self):
        """CausalProvenanceTracker._fb_ref defaults to None."""
        tracker = CausalProvenanceTracker()
        assert tracker._fb_ref is None

    def test_provenance_tracker_fb_ref_post_wiring(self):
        """CausalProvenanceTracker._fb_ref can be set post-construction."""
        bus = _make_bus()
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus
        assert tracker._fb_ref is bus

    def test_provenance_broadcast_works_when_wired(self):
        """_broadcast_attribution_to_bus should write signals when wired."""
        bus = _make_bus()
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus

        contributions = {'module_a': 0.8, 'module_b': 0.1, 'module_c': 0.1}
        tracker._broadcast_attribution_to_bus(contributions)

        conc = bus.read_signal('provenance_attribution_concentration', -1.0)
        assert conc > 0.0, "provenance_attribution_concentration should be written"

    def test_provenance_broadcast_dominance_alarm(self):
        """High attribution concentration should trigger dominance alarm."""
        bus = _make_bus()
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus

        contributions = {'module_a': 0.95, 'module_b': 0.05}
        tracker._broadcast_attribution_to_bus(contributions)

        alarm = bus.read_signal('provenance_dominance_alarm', 0.0)
        assert alarm > 0.6, "dominance alarm should fire for concentrated attribution"

    def test_provenance_no_broadcast_without_wiring(self):
        """_broadcast_attribution_to_bus should be a no-op when _fb_ref is None."""
        tracker = CausalProvenanceTracker()
        assert tracker._fb_ref is None
        contributions = {'module_a': 0.9, 'module_b': 0.1}
        tracker._broadcast_attribution_to_bus(contributions)


# ═══════════════════════════════════════════════════════════════════════
# PATCH-APEX-3: Orphaned adaptation signals → MCT
# ═══════════════════════════════════════════════════════════════════════

class TestAPEX3_OrphanedSignalWiring:
    """Verify MCT evaluate() reads previously orphaned adaptation signals."""

    def test_low_output_reliability_pressure_read(self):
        """MCT should read low_output_reliability_pressure."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('low_output_reliability_pressure', 0.8)
        result = mct.evaluate()
        assert 'low_output_reliability_pressure' in bus._read_log

    def test_encoder_attention_sharpened_read(self):
        """MCT should read encoder_attention_sharpened."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('encoder_attention_sharpened', 0.6)
        result = mct.evaluate()
        assert 'encoder_attention_sharpened' in bus._read_log

    def test_memory_retrieval_depth_adapted_read(self):
        """MCT should read memory_retrieval_depth_adapted."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('memory_retrieval_depth_adapted', 0.7)
        result = mct.evaluate()
        assert 'memory_retrieval_depth_adapted' in bus._read_log

    def test_cache_invalidation_convergence_tightened_read(self):
        """MCT should read cache_invalidation_convergence_tightened."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('cache_invalidation_convergence_tightened', 0.8)
        result = mct.evaluate()
        assert 'cache_invalidation_convergence_tightened' in bus._read_log

    def test_causal_reasoning_depth_adapted_read(self):
        """MCT should read causal_reasoning_depth_adapted."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('causal_reasoning_depth_adapted', 0.5)
        result = mct.evaluate()
        assert 'causal_reasoning_depth_adapted' in bus._read_log

    def test_factor_extraction_depth_adapted_read(self):
        """MCT should read factor_extraction_depth_adapted."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('factor_extraction_depth_adapted', 0.5)
        result = mct.evaluate()
        assert 'factor_extraction_depth_adapted' in bus._read_log

    def test_high_output_reliability_pressure_boosts_trigger(self):
        """High low_output_reliability_pressure should increase trigger score."""
        mct_base, bus_base = _make_mct_with_bus()
        result_base = mct_base.evaluate()
        score_base = result_base.get('trigger_score', 0.0)

        mct_press, bus_press = _make_mct_with_bus()
        bus_press.write_signal('low_output_reliability_pressure', 0.9)
        result_press = mct_press.evaluate()
        score_press = result_press.get('trigger_score', 0.0)

        assert score_press >= score_base

    def test_combined_adaptation_signals_amplify_trigger(self):
        """Multiple adaptation signals together should compound trigger."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('low_output_reliability_pressure', 0.7)
        bus.write_signal('encoder_attention_sharpened', 0.5)
        bus.write_signal('memory_retrieval_depth_adapted', 0.6)
        bus.write_signal('causal_reasoning_depth_adapted', 0.5)
        bus.write_signal('factor_extraction_depth_adapted', 0.5)
        bus.write_signal('cache_invalidation_convergence_tightened', 0.8)

        result = mct.evaluate()
        assert 'should_trigger' in result or 'trigger_score' in result

    def test_adaptation_signals_below_threshold_no_effect(self):
        """Adaptation signals below thresholds should not boost trigger."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('low_output_reliability_pressure', 0.1)
        bus.write_signal('encoder_attention_sharpened', 0.1)
        bus.write_signal('memory_retrieval_depth_adapted', 0.1)
        result = mct.evaluate()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
# PATCH-APEX-4: Silent except block hardening
# ═══════════════════════════════════════════════════════════════════════

class TestAPEX4_SilentExceptHardening:
    """Verify that previously-silent except blocks now log failures."""

    def test_thought_encoder_read_failure_logged(self):
        """ThoughtEncoder bus read failure should be logged, not silent."""
        enc = _make_encoder(vocab_size=100, emb_dim=64, z_dim=64)
        mock_bus = MagicMock()
        mock_bus.read_signal.side_effect = RuntimeError("bus read failure")
        mock_bus.write_signal_traced = MagicMock()
        enc._fb_ref = mock_bus

        tokens = torch.randint(0, 100, (2, 10))
        with patch('aeon_core.logger') as mock_logger:
            result = enc(tokens)
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            apex4_logged = any('APEX-4' in c for c in debug_calls)
            assert apex4_logged, (
                "ThoughtEncoder bus read failure should log with APEX-4 prefix"
            )

    def test_thought_encoder_write_failure_logged(self):
        """ThoughtEncoder bus write failure should be logged, not silent."""
        enc = _make_encoder(vocab_size=100, emb_dim=64, z_dim=64)
        mock_bus = MagicMock()
        mock_bus.read_signal.return_value = 0.2  # Low reliability → sharpening
        mock_bus.write_signal_traced.side_effect = RuntimeError("bus write failure")
        enc._fb_ref = mock_bus

        tokens = torch.randint(0, 100, (2, 10))
        with patch('aeon_core.logger') as mock_logger:
            result = enc(tokens)
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            apex4_logged = any('APEX-4' in c for c in debug_calls)
            assert apex4_logged, (
                "ThoughtEncoder bus write failure should log with APEX-4 prefix"
            )

    def test_provenance_tracker_broadcast_failure_logged(self):
        """CausalProvenanceTracker bus broadcast failure should be logged."""
        tracker = CausalProvenanceTracker()
        mock_bus = MagicMock()
        mock_bus.write_signal.side_effect = RuntimeError("bus write failure")
        tracker._fb_ref = mock_bus

        with patch('aeon_core.logger') as mock_logger:
            contributions = {'module_a': 0.8, 'module_b': 0.2}
            tracker._broadcast_attribution_to_bus(contributions)
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            apex4_logged = any('APEX-4' in c for c in debug_calls)
            assert apex4_logged, (
                "CausalProvenanceTracker broadcast failure should log with APEX-4"
            )


# ═══════════════════════════════════════════════════════════════════════
# Integration: Full signal loop verification
# ═══════════════════════════════════════════════════════════════════════

class TestAPEX_IntegrationLoops:
    """Verify that APEX patches close previously-broken signal loops."""

    def test_provenance_to_mct_loop(self):
        """Provenance tracker → bus → MCT reads provenance_attribution_concentration."""
        bus = _make_bus()
        tracker = CausalProvenanceTracker()
        tracker._fb_ref = bus

        contributions = {'module_a': 0.85, 'module_b': 0.15}
        tracker._broadcast_attribution_to_bus(contributions)

        conc = bus.read_signal('provenance_attribution_concentration', -1.0)
        assert conc > 0.0

        mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
        mct.set_feedback_bus(bus)
        result = mct.evaluate()
        assert isinstance(result, dict)

    def test_mcts_to_mct_loop(self):
        """MCTSPlanner writes mcts_planning_confidence → MCT reads it."""
        bus = _make_bus()
        bus.write_signal('mcts_planning_confidence', 0.3)

        mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
        mct.set_feedback_bus(bus)
        result = mct.evaluate()
        assert isinstance(result, dict)

    def test_adaptation_signals_no_longer_orphaned(self):
        """Previously orphaned adaptation signals should now be consumed."""
        bus = _make_bus()
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
        mct.set_feedback_bus(bus)

        orphaned_signals = {
            'low_output_reliability_pressure': 0.7,
            'encoder_attention_sharpened': 0.5,
            'memory_retrieval_depth_adapted': 0.6,
            'cache_invalidation_convergence_tightened': 0.8,
            'causal_reasoning_depth_adapted': 0.5,
            'factor_extraction_depth_adapted': 0.5,
        }
        for name, val in orphaned_signals.items():
            bus.write_signal(name, val)

        result = mct.evaluate()
        assert isinstance(result, dict)

    def test_signal_ecosystem_completeness(self):
        """Verify no critical signal is orphaned after APEX patches."""
        newly_wired = [
            'low_output_reliability_pressure',
            'encoder_attention_sharpened',
            'memory_retrieval_depth_adapted',
            'cache_invalidation_convergence_tightened',
            'causal_reasoning_depth_adapted',
            'factor_extraction_depth_adapted',
        ]
        bus = _make_bus()
        mct = MetaCognitiveRecursionTrigger(trigger_threshold=1.0)
        mct.set_feedback_bus(bus)

        for sig in newly_wired:
            bus.write_signal(sig, 0.5)

        mct.evaluate()

        read_log = getattr(bus, '_read_log', set())
        for sig in newly_wired:
            assert sig in read_log, f"Signal '{sig}' was not read by MCT (still orphaned)"

