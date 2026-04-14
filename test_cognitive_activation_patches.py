"""
Tests for the three critical cognitive integration patches that
transition AEON-Delta from connected architecture to functional
cognitive organism.

Patch 1 (PATCH-ACTIV-1): Fix axioms UnboundLocalError in verify_and_reinforce
Patch 2 (PATCH-ACTIV-2): Add bridge_active key to training_bridge dict
Patch 3 (PATCH-ACTIV-3): Extend cold-start warmup guard to init-time episodes
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import AEONDeltaV3, AEONConfig


@pytest.fixture(scope="module")
def model():
    """Shared model instance for all tests."""
    config = AEONConfig()
    return AEONDeltaV3(config)


# ============================================================
# §1 — PATCH-ACTIV-1: verify_and_reinforce axioms fix
# ============================================================

class TestPatchActiv1VerifyAndReinforce:
    """Verify that verify_and_reinforce no longer crashes with
    UnboundLocalError on `axioms`."""

    def test_verify_and_reinforce_does_not_crash(self, model):
        """The method must complete without exception."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert isinstance(result, dict)

    def test_verify_and_reinforce_not_skipped_reentrant(self, model):
        """After resetting the guard, the method must not skip."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert result.get('skipped_reentrant') is not True

    def test_verify_and_reinforce_returns_actions(self, model):
        """The method must produce reinforcement actions."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert 'reinforcement_actions' in result
        assert isinstance(result['reinforcement_actions'], list)

    def test_verify_and_reinforce_has_overall_score(self, model):
        """The method must return a meaningful overall score."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert 'overall_score' in result
        assert isinstance(result['overall_score'], (int, float))
        assert result['overall_score'] > 0

    def test_reentrant_guard_resets_after_success(self, model):
        """After successful completion, the guard must be False."""
        model._verify_and_reinforce_in_progress = False
        model.verify_and_reinforce()
        assert model._verify_and_reinforce_in_progress is False


# ============================================================
# §2 — PATCH-ACTIV-2: Training bridge bridge_active key
# ============================================================

class TestPatchActiv2TrainingBridge:
    """Verify that training_bridge dict includes bridge_active."""

    def test_training_bridge_has_bridge_active_key(self, model):
        """The training_bridge dict must include bridge_active."""
        unity = model.verify_cognitive_unity()
        tb = unity.get('training_bridge', {})
        assert 'bridge_active' in tb

    def test_bridge_active_is_true_when_ready(self, model):
        """bridge_active should be True when all components are present."""
        unity = model.verify_cognitive_unity()
        tb = unity.get('training_bridge', {})
        # When ready is True, bridge_active should also be True
        if tb.get('ready'):
            assert tb['bridge_active'] is True

    def test_bridge_active_consistent_with_ready(self, model):
        """bridge_active should be consistent with ready."""
        unity = model.verify_cognitive_unity()
        tb = unity.get('training_bridge', {})
        assert tb.get('bridge_active') == tb.get('ready')

    def test_phase_9_reads_bridge_active(self, model):
        """Phase 9 of the activation sequence must now be active."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        phase_9 = next(
            (p for p in seq if p.get('order') == 9), None,
        )
        assert phase_9 is not None
        assert phase_9['status'] == 'active'


# ============================================================
# §3 — PATCH-ACTIV-3: Cold-start trend warmup guard
# ============================================================

class TestPatchActiv3ColdStartWarmup:
    """Verify that init-time diagnostic episodes don't block emergence."""

    def test_ee_trend_healthy_at_cold_start(self, model):
        """Error evolution trend should be healthy at init (0 fwd calls)."""
        unity = model.verify_cognitive_unity()
        ee_eff = unity.get('error_evolution_effectiveness', {})
        assert ee_eff.get('trend_healthy') is True

    def test_is_unified_at_cold_start(self, model):
        """verify_cognitive_unity should return unified=True at init."""
        unity = model.verify_cognitive_unity()
        assert unity.get('unified') is True

    def test_degrading_classes_exist_but_suppressed(self, model):
        """Degrading classes may exist but should not block unity."""
        unity = model.verify_cognitive_unity()
        ee_eff = unity.get('error_evolution_effectiveness', {})
        # Degrading classes can exist (from init seeding) but
        # trend_healthy is True because warmup guard suppresses them
        assert ee_eff.get('trend_healthy') is True

    def test_cognitive_unity_score_is_high(self, model):
        """Cognitive unity score should be high (>= 0.9) at init."""
        unity = model.verify_cognitive_unity()
        assert unity.get('cognitive_unity_score', 0) >= 0.9


# ============================================================
# §4 — System Emergence: Full integration verification
# ============================================================

class TestSystemEmergence:
    """Verify that the three patches together enable system emergence."""

    def test_system_emerged(self, model):
        """The system must report emerged=True."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        es = report.get('system_emergence_status', {})
        assert es.get('emerged') is True

    def test_all_three_axioms_met(self, model):
        """All three core AGI axioms must be satisfied."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        es = report.get('system_emergence_status', {})
        assert es.get('mutual_reinforcement_met') is True
        assert es.get('meta_cognitive_trigger_met') is True
        assert es.get('causal_transparency_met') is True

    def test_all_nine_conditions_met(self, model):
        """All 9 emergence conditions must be satisfied."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        es = report.get('system_emergence_status', {})
        assert es.get('conditions_met') == es.get('conditions_total')

    def test_initial_emergence_has_no_diagnostic_gaps(self):
        """On a fresh model, diagnostic gaps should be zero."""
        config = AEONConfig()
        fresh = AEONDeltaV3(config)
        report = fresh.system_emergence_report()
        es = report.get('system_emergence_status', {})
        assert es.get('diagnostic_gaps_ok') is True

    def test_cognitive_unity_unified(self, model):
        """Cognitive unity must report unified=True."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        es = report.get('system_emergence_status', {})
        assert es.get('cognitive_unity_unified') is True

    def test_emergence_status_string(self, model):
        """emergence_status must be 'emerged'."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        assert report.get('emergence_status') == 'emerged'

    def test_system_unified(self, model):
        """system_unified must be True."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        assert report.get('system_unified') is True


# ============================================================
# §5 — Activation Sequence: All 10 phases active
# ============================================================

class TestActivationSequence:
    """Verify all 10 activation phases are active/achieved."""

    def test_ten_phases_present(self, model):
        """Activation sequence must have exactly 10 phases."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        assert len(seq) == 10

    def test_all_phases_active_or_achieved(self, model):
        """Every phase must be active or achieved (not incomplete/pending)."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        seq = report.get('activation_sequence', [])
        for phase in seq:
            assert phase['status'] in ('active', 'achieved'), (
                f"Phase {phase['order']} ({phase['phase']}) "
                f"has status '{phase['status']}'"
            )


# ============================================================
# §6 — Causal Transparency: Full chain traceability
# ============================================================

class TestCausalTransparency:
    """Verify causal chain is fully traceable."""

    def test_causal_chain_traceable(self, model):
        """Causal chain must be fully traceable."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        cc = report.get('causal_chain', {})
        assert cc.get('traceable') is True

    def test_few_untraced_subsystems(self, model):
        """Most subsystems should be traced (coverage >= 0.7)."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        cc = report.get('causal_chain', {})
        assert cc.get('coverage', 0) >= 0.7

    def test_causal_chain_connected(self, model):
        """Causal chain must be connected."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        cc = report.get('causal_chain', {})
        assert cc.get('chain_connected') is True


# ============================================================
# §7 — Integration Map: No isolated nodes
# ============================================================

class TestIntegrationMap:
    """Verify integration map shows full connectivity."""

    def test_no_isolated_paths(self, model):
        """No isolated paths should remain."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        imap = report.get('integration_map', {})
        assert imap.get('isolated_paths', 0) == 0

    def test_full_wiring_coverage(self, model):
        """Wiring coverage must be 1.0."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        imap = report.get('integration_map', {})
        assert imap.get('wiring_coverage', 0) == 1.0

    def test_dag_acyclic(self, model):
        """Pipeline DAG must be acyclic."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        imap = report.get('integration_map', {})
        assert imap.get('dag_acyclic') is True

    def test_no_missing_edges(self, model):
        """No missing edges in the pipeline wiring."""
        model._verify_and_reinforce_in_progress = False
        report = model.system_emergence_report()
        imap = report.get('integration_map', {})
        missing = imap.get('missing_edges', [])
        assert len(missing) == 0


# ============================================================
# §8 — Mutual Reinforcement: verify_and_reinforce works
# ============================================================

class TestMutualReinforcement:
    """Verify mutual reinforcement is functional."""

    def test_reinforcement_produces_actions(self, model):
        """verify_and_reinforce must produce concrete actions."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert len(result.get('reinforcement_actions', [])) > 0

    def test_reinforcement_success(self, model):
        """Reinforcement must report success."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert result.get('reinforcement_success') is True

    def test_reinforcement_high_score(self, model):
        """Reinforcement overall score should be reasonable (> 0.7)."""
        model._verify_and_reinforce_in_progress = False
        result = model.verify_and_reinforce()
        assert result.get('overall_score', 0) > 0.7
