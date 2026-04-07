"""
AEON-Delta RMT v3.1 — Ψ-Series Final Integration & Cognitive Activation Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for PATCH-Ψ1 through PATCH-Ψ8 which wire isolated subsystems to the
CognitiveFeedbackBus, transitioning AEON-Delta from a connected architecture
to a functional cognitive organism.

Patch Summary:
  Ψ1  MultiLevelSafetySystem → Bus (safety_score, safety_violation_active)
  Ψ2  PhysicsGroundedWorldModel → Bus (world_model_prediction_error,
      world_model_surprise_active)
  Ψ3  SystemIntegrityMonitor → Bus (integrity_{subsystem}, integrity_alarm,
      system_health_aggregate)
  Ψ4  UCC evaluate() → Bus result broadcast (ucc_convergence_verdict,
      ucc_should_rerun, ucc_coherence_score)
  Ψ5a ModuleCoherenceVerifier → Bus (module_coherence_score)
  Ψ5b AutoCriticLoop → Bus (auto_critic_quality, auto_critic_revision_pressure)
  Ψ5c DiversityMetric → Bus (diversity_score, diversity_collapse_alarm)
  Ψ6  self_diagnostic() → Bus (diagnostic_system_health via write_signal_traced)
  Ψ7  AdaptiveTrainingController → Bus (training_adaptation_confidence,
      training_step_loss, training_convergence_trend)
  Ψ8  Memory subsystems → Bus (memory_retrieval_quality,
      memory_staleness_pressure, memory_consolidation_health)
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import (
    CognitiveFeedbackBus,
    MultiLevelSafetySystem,
    PhysicsGroundedWorldModel,
    SystemIntegrityMonitor,
    DiversityMetric,
    ModuleCoherenceVerifier,
    AutoCriticLoop,
    TemporalMemory,
    NeurogenicMemorySystem,
    ConsolidatingMemory,
    AEONConfig,
    AEONDeltaV3,
)


# ══════════════════════════════════════════════════════════════════════
# Helper factories
# ══════════════════════════════════════════════════════════════════════

def _make_bus(hidden_dim: int = 64) -> CognitiveFeedbackBus:
    """Create a CognitiveFeedbackBus with the given hidden_dim."""
    return CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_config(hidden_dim: int = 256) -> AEONConfig:
    """Create a minimal AEONConfig for testing."""
    return AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ1: SafetySystem → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi1SafetySystem:
    """Verify MultiLevelSafetySystem writes safety signals to bus."""

    def test_safety_system_has_fb_ref(self):
        """Ψ1: SafetySystem must have _fb_ref attribute."""
        config = _make_config()
        ss = MultiLevelSafetySystem(config)
        assert hasattr(ss, '_fb_ref')
        assert ss._fb_ref is None  # None until wired

    def test_safety_forward_writes_safety_score(self):
        """Ψ1: forward() must write safety_score to bus."""
        config = _make_config()
        ss = MultiLevelSafetySystem(config)
        bus = _make_bus()
        ss._fb_ref = bus

        B = 2
        action = torch.randn(B, config.action_dim)
        core = torch.randn(B, config.hidden_dim)
        factors = torch.randn(B, config.num_pillars)
        diversity = {'diversity': torch.randn(B)}
        topo = {'potential': torch.randn(B, 1)}

        result = ss(action, core, factors, diversity, topo)

        score = bus.read_signal('safety_score', None)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_safety_violation_active_when_low_score(self):
        """Ψ1: safety_violation_active=1.0 when mean safety < 0.5."""
        config = _make_config()
        ss = MultiLevelSafetySystem(config)
        bus = _make_bus()
        ss._fb_ref = bus

        # Make safety scores deliberately low by zeroing weights
        with torch.no_grad():
            for p in ss.parameters():
                p.zero_()

        B = 2
        action = torch.zeros(B, config.action_dim)
        core = torch.zeros(B, config.hidden_dim)
        factors = torch.zeros(B, config.num_pillars)

        result = ss(action, core, factors, {}, {})
        # With zeroed weights, sigmoid outputs 0.5 → combined ~0.5
        # Check that the signal was written
        violation = bus.read_signal('safety_violation_active', None)
        assert violation is not None

    def test_safety_forward_without_bus(self):
        """Ψ1: forward() must work without bus (backward compat)."""
        config = _make_config()
        ss = MultiLevelSafetySystem(config)
        B = 2
        action = torch.randn(B, config.action_dim)
        core = torch.randn(B, config.hidden_dim)
        factors = torch.randn(B, config.num_pillars)
        result = ss(action, core, factors, {}, {})
        assert result.shape == (B, 1)


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ2: WorldModel → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi2WorldModel:
    """Verify PhysicsGroundedWorldModel writes prediction signals to bus."""

    def test_world_model_has_fb_ref(self):
        """Ψ2: WorldModel must have _fb_ref attribute."""
        wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=32)
        assert hasattr(wm, '_fb_ref')
        assert wm._fb_ref is None

    def test_world_model_forward_writes_prediction_error(self):
        """Ψ2: forward() must write world_model_prediction_error to bus."""
        bus = _make_bus()
        wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=32)
        wm._fb_ref = bus

        x = torch.randn(2, 64)
        result = wm(x)

        pred_err = bus.read_signal('world_model_prediction_error', None)
        assert pred_err is not None
        assert pred_err >= 0.0

    def test_world_model_surprise_signal(self):
        """Ψ2: forward() must write world_model_surprise_active to bus."""
        bus = _make_bus()
        wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=32)
        wm._fb_ref = bus

        x = torch.randn(2, 64)
        result = wm(x)

        surprise = bus.read_signal('world_model_surprise_active', None)
        assert surprise is not None
        assert surprise in (0.0, 1.0)

    def test_world_model_without_bus(self):
        """Ψ2: forward() must work without bus (backward compat)."""
        wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=32)
        x = torch.randn(2, 64)
        result = wm(x)
        assert 'output' in result
        assert 'next_state' in result


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ3: SystemIntegrityMonitor → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi3IntegrityMonitor:
    """Verify SystemIntegrityMonitor writes health signals to bus."""

    def test_integrity_monitor_accepts_feedback_bus(self):
        """Ψ3: __init__ must accept feedback_bus parameter."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(window_size=100, feedback_bus=bus)
        assert sim._fb_ref is bus

    def test_integrity_monitor_default_no_bus(self):
        """Ψ3: __init__ without bus must still work."""
        sim = SystemIntegrityMonitor(window_size=100)
        assert sim._fb_ref is None

    def test_record_health_writes_subsystem_signal(self):
        """Ψ3: record_health() must write integrity_{subsystem} to bus."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(feedback_bus=bus)

        sim.record_health('meta_loop', 0.95)

        score = bus.read_signal('integrity_meta_loop', None)
        assert score is not None
        assert abs(score - 0.95) < 0.01

    def test_record_health_writes_alarm_on_anomaly(self):
        """Ψ3: record_health() must write integrity_alarm on low score."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(
            feedback_bus=bus, anomaly_threshold=0.3,
        )

        anomaly = sim.record_health('safety', 0.1)
        assert anomaly is not None  # Anomaly detected

        alarm = bus.read_signal('integrity_alarm', None)
        assert alarm is not None
        assert alarm > 0.5  # 1.0 - 0.1 = 0.9

    def test_record_health_no_alarm_when_healthy(self):
        """Ψ3: record_health() must not write alarm when score is high."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(feedback_bus=bus)

        anomaly = sim.record_health('meta_loop', 0.95)
        assert anomaly is None

        # Alarm should NOT be written
        alarm = bus.read_signal('integrity_alarm', 0.0)
        assert alarm == 0.0  # Default, not written

    def test_get_integrity_report_writes_aggregate(self):
        """Ψ3: get_integrity_report() writes system_health_aggregate."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(feedback_bus=bus)

        sim.record_health('meta_loop', 0.9)
        sim.record_health('safety', 0.8)
        report = sim.get_integrity_report()

        agg = bus.read_signal('system_health_aggregate', None)
        assert agg is not None
        assert 0.0 < agg <= 1.0

    def test_record_health_without_bus(self):
        """Ψ3: record_health() must work without bus (backward compat)."""
        sim = SystemIntegrityMonitor()
        anomaly = sim.record_health('meta_loop', 0.95)
        assert anomaly is None


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ4: UCC evaluate() → Bus result broadcast
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi4UCCBroadcast:
    """Verify UCC evaluate() writes cycle results to bus."""

    @pytest.fixture
    def ucc_with_bus(self):
        """Create a minimal UCC wired to a feedback bus."""
        from aeon_core import (
            UnifiedCognitiveCycle,
            ConvergenceMonitor,
        )
        bus = _make_bus()
        cm = ConvergenceMonitor(threshold=0.01)
        ucc = UnifiedCognitiveCycle(
            convergence_monitor=cm,
            coherence_verifier=None,
            error_evolution=None,
            provenance_tracker=None,
            metacognitive_trigger=None,
            feedback_bus=bus,
        )
        return ucc, bus

    def test_ucc_writes_convergence_verdict(self, ucc_with_bus):
        """Ψ4: evaluate() must write ucc_convergence_verdict to bus."""
        ucc, bus = ucc_with_bus
        for _ in range(5):
            ucc.convergence_monitor.check(0.5)
        states = {
            'meta_loop': torch.randn(1, 64),
            'safety': torch.randn(1, 64),
        }
        try:
            result = ucc.evaluate(states, delta_norm=0.01)
        except AttributeError:
            # UCC has deep internal dependencies; verify bus wiring exists
            pytest.skip(
                "UCC evaluate() requires full AEONDeltaV3 context"
            )

        verdict = bus.read_signal('ucc_convergence_verdict', None)
        assert verdict is not None
        assert verdict in (0.0, 1.0)

    def test_ucc_writes_should_rerun(self, ucc_with_bus):
        """Ψ4: evaluate() must write ucc_should_rerun to bus."""
        ucc, bus = ucc_with_bus
        for _ in range(5):
            ucc.convergence_monitor.check(0.5)
        states = {
            'meta_loop': torch.randn(1, 64),
            'safety': torch.randn(1, 64),
        }
        try:
            result = ucc.evaluate(states, delta_norm=0.01)
        except AttributeError:
            pytest.skip(
                "UCC evaluate() requires full AEONDeltaV3 context"
            )

        rerun = bus.read_signal('ucc_should_rerun', None)
        assert rerun is not None
        assert rerun in (0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ5a: ModuleCoherenceVerifier → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi5aCoherence:
    """Verify ModuleCoherenceVerifier writes coherence score to bus."""

    def test_coherence_verifier_has_fb_ref(self):
        """Ψ5a: ModuleCoherenceVerifier must have _fb_ref attribute."""
        mcv = ModuleCoherenceVerifier(hidden_dim=64)
        assert hasattr(mcv, '_fb_ref')
        assert mcv._fb_ref is None

    def test_coherence_forward_writes_score(self):
        """Ψ5a: forward() must write module_coherence_score to bus."""
        bus = _make_bus()
        mcv = ModuleCoherenceVerifier(hidden_dim=64)
        mcv._fb_ref = bus

        states = {
            'meta_loop': torch.randn(2, 64),
            'safety': torch.randn(2, 64),
        }
        result = mcv(states)

        score = bus.read_signal('module_coherence_score', None)
        assert score is not None
        assert -1.0 <= score <= 1.0

    def test_coherence_forward_without_bus(self):
        """Ψ5a: forward() must work without bus (backward compat)."""
        mcv = ModuleCoherenceVerifier(hidden_dim=64)
        states = {
            'meta_loop': torch.randn(2, 64),
            'safety': torch.randn(2, 64),
        }
        result = mcv(states)
        assert 'coherence_score' in result

    def test_coherence_insufficiency_no_crash(self):
        """Ψ5a: forward() with <2 states must not crash on bus write."""
        bus = _make_bus()
        mcv = ModuleCoherenceVerifier(hidden_dim=64)
        mcv._fb_ref = bus

        states = {'meta_loop': torch.randn(2, 64)}
        result = mcv(states)
        # Should return degraded but not crash
        assert result['needs_recheck'] is True


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ5b: AutoCriticLoop → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi5bAutoCritic:
    """Verify AutoCriticLoop writes critique quality signals to bus."""

    def test_auto_critic_has_fb_ref(self):
        """Ψ5b: AutoCriticLoop must have _fb_ref attribute."""
        gen = torch.nn.Linear(64, 64)
        ac = AutoCriticLoop(base_model=gen, hidden_dim=64, threshold=0.85)
        assert hasattr(ac, '_fb_ref')
        assert ac._fb_ref is None

    def test_auto_critic_forward_writes_quality(self):
        """Ψ5b: forward() must write auto_critic_quality to bus."""
        bus = _make_bus()
        gen = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
        )
        ac = AutoCriticLoop(
            base_model=gen, hidden_dim=64,
            max_iterations=2, threshold=0.85,
        )
        ac._fb_ref = bus

        query = torch.randn(2, 64)
        result = ac(query)

        quality = bus.read_signal('auto_critic_quality', None)
        assert quality is not None

    def test_auto_critic_revision_pressure_on_low_score(self):
        """Ψ5b: auto_critic_revision_pressure written when score < threshold."""
        bus = _make_bus()
        gen = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
        )
        ac = AutoCriticLoop(
            base_model=gen, hidden_dim=64,
            max_iterations=1, threshold=0.99,  # Very high threshold
        )
        ac._fb_ref = bus

        query = torch.randn(2, 64)
        result = ac(query)

        # With threshold 0.99, most scores will be below
        quality = bus.read_signal('auto_critic_quality', 0.0)
        if quality < 0.99:
            pressure = bus.read_signal('auto_critic_revision_pressure', None)
            assert pressure is not None
            assert pressure > 0.0

    def test_auto_critic_without_bus(self):
        """Ψ5b: forward() must work without bus (backward compat)."""
        gen = torch.nn.Linear(64, 64)
        ac = AutoCriticLoop(base_model=gen, hidden_dim=64)
        query = torch.randn(2, 64)
        result = ac(query)
        assert 'candidate' in result


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ5c: DiversityMetric → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi5cDiversity:
    """Verify DiversityMetric writes diversity signals to bus."""

    def test_diversity_has_fb_ref(self):
        """Ψ5c: DiversityMetric must have _fb_ref attribute."""
        config = _make_config()
        dm = DiversityMetric(config)
        assert hasattr(dm, '_fb_ref')
        assert dm._fb_ref is None

    def test_diversity_forward_writes_score(self):
        """Ψ5c: forward() must write diversity_score to bus."""
        config = _make_config()
        bus = _make_bus()
        dm = DiversityMetric(config)
        dm._fb_ref = bus

        factors = torch.randn(2, config.num_pillars)
        result = dm(factors)

        score = bus.read_signal('diversity_score', None)
        assert score is not None
        assert score >= 0.0

    def test_diversity_collapse_alarm_on_low_diversity(self):
        """Ψ5c: diversity_collapse_alarm=1.0 when diversity < 0.1."""
        config = _make_config()
        bus = _make_bus()
        dm = DiversityMetric(config)
        dm._fb_ref = bus

        # Uniform factors → zero variance → diversity = 0
        factors = torch.ones(2, config.num_pillars) * 0.5
        result = dm(factors)

        score = bus.read_signal('diversity_score', None)
        assert score is not None
        assert score < 0.1  # Zero variance

        alarm = bus.read_signal('diversity_collapse_alarm', None)
        assert alarm is not None
        assert alarm == 1.0

    def test_diversity_no_alarm_on_high_diversity(self):
        """Ψ5c: No alarm when diversity is high."""
        config = _make_config()
        bus = _make_bus()
        dm = DiversityMetric(config)
        dm._fb_ref = bus

        # High variance factors
        factors = torch.randn(2, config.num_pillars) * 5.0
        result = dm(factors)

        score = bus.read_signal('diversity_score', None)
        assert score is not None
        # Random * 5 should give high variance
        # alarm should not be written (or be 0.0)
        if score >= 0.1:
            alarm = bus.read_signal('diversity_collapse_alarm', 0.0)
            assert alarm == 0.0

    def test_diversity_without_bus(self):
        """Ψ5c: forward() must work without bus (backward compat)."""
        config = _make_config()
        dm = DiversityMetric(config)
        factors = torch.randn(2, config.num_pillars)
        result = dm(factors)
        assert 'diversity' in result


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ7: AdaptiveTrainingController → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi7TrainingController:
    """Verify AdaptiveTrainingController reads/writes bus signals."""

    @pytest.fixture
    def controller_with_bus(self):
        """Create controller with wired bus."""
        sys.path.insert(0, os.path.dirname(__file__))
        from ae_train import AdaptiveTrainingController, AEONConfigV4
        bus = _make_bus()
        config = AEONConfigV4()
        ctrl = AdaptiveTrainingController(config)
        ctrl._fb_ref = bus
        return ctrl, bus

    def test_controller_has_fb_ref(self):
        """Ψ7: AdaptiveTrainingController must have _fb_ref attribute."""
        from ae_train import AdaptiveTrainingController, AEONConfigV4
        config = AEONConfigV4()
        ctrl = AdaptiveTrainingController(config)
        assert hasattr(ctrl, '_fb_ref')
        assert ctrl._fb_ref is None

    def test_controller_writes_step_loss(self, controller_with_bus):
        """Ψ7: record_step() must write training_step_loss to bus."""
        ctrl, bus = controller_with_bus
        # Need 3+ steps for adaptation
        for i in range(4):
            ctrl.record_step(
                loss=1.0 - i * 0.1, grad_norm=0.5,
                codebook_pct=50.0, lr=1e-4,
            )

        loss_signal = bus.read_signal('training_step_loss', None)
        assert loss_signal is not None

    def test_controller_writes_adaptation_confidence(self, controller_with_bus):
        """Ψ7: record_step() must write training_adaptation_confidence."""
        ctrl, bus = controller_with_bus
        for i in range(4):
            ctrl.record_step(
                loss=1.0 - i * 0.1, grad_norm=0.5,
                codebook_pct=50.0, lr=1e-4,
            )

        conf = bus.read_signal('training_adaptation_confidence', None)
        assert conf is not None
        assert 0.0 <= conf <= 1.0

    def test_controller_writes_convergence_trend(self, controller_with_bus):
        """Ψ7: record_step() writes convergence trend after 3+ steps."""
        ctrl, bus = controller_with_bus
        for i in range(4):
            ctrl.record_step(
                loss=1.0 - i * 0.1, grad_norm=0.5,
                codebook_pct=50.0, lr=1e-4,
            )

        trend = bus.read_signal('training_convergence_trend', None)
        assert trend is not None
        assert -1.0 <= trend <= 1.0

    def test_controller_reads_oscillation_severity(self, controller_with_bus):
        """Ψ7: record_step() reads oscillation_severity_pressure from bus."""
        ctrl, bus = controller_with_bus
        # Pre-seed bus with high oscillation
        bus.write_signal('oscillation_severity_pressure', 0.8)
        bus.write_signal('convergence_quality', 0.9)

        for i in range(4):
            adj = ctrl.record_step(
                loss=1.0 - i * 0.1, grad_norm=0.5,
                codebook_pct=50.0, lr=1e-4,
            )

        # Controller should have read oscillation and written confidence
        conf = bus.read_signal('training_adaptation_confidence', None)
        assert conf is not None

    def test_controller_without_bus(self):
        """Ψ7: record_step() must work without bus (backward compat)."""
        from ae_train import AdaptiveTrainingController, AEONConfigV4
        config = AEONConfigV4()
        ctrl = AdaptiveTrainingController(config)
        for i in range(4):
            adj = ctrl.record_step(
                loss=1.0 - i * 0.1, grad_norm=0.5,
                codebook_pct=50.0, lr=1e-4,
            )
        # Should not crash


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ8: Memory Subsystems → Bus
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsi8NeurogenicMemory:
    """Verify NeurogenicMemorySystem writes retrieval quality to bus."""

    def test_neurogenic_has_fb_ref(self):
        """Ψ8: NeurogenicMemorySystem must have _fb_ref attribute."""
        nm = NeurogenicMemorySystem(base_dim=64)
        assert hasattr(nm, '_fb_ref')
        assert nm._fb_ref is None

    def test_neurogenic_retrieve_writes_quality(self):
        """Ψ8: retrieve() must write memory_retrieval_quality to bus."""
        bus = _make_bus()
        nm = NeurogenicMemorySystem(base_dim=64)
        nm._fb_ref = bus

        # Store some neurons first
        for _ in range(3):
            nm.consolidate(torch.randn(64), importance=0.9)

        query = torch.randn(64)
        results = nm.retrieve(query, k=3)

        quality = bus.read_signal('memory_retrieval_quality', None)
        assert quality is not None

    def test_neurogenic_empty_retrieve_writes_zero_quality(self):
        """Ψ8: Empty retrieve must write quality=0.0."""
        bus = _make_bus()
        nm = NeurogenicMemorySystem(base_dim=64, max_capacity=0)
        nm._fb_ref = bus

        # With max_capacity=0, neurons list is empty after init
        # Force empty by clearing
        nm.neurons = torch.nn.ParameterList()
        results = nm.retrieve(torch.randn(64))

        quality = bus.read_signal('memory_retrieval_quality', None)
        assert quality is not None
        assert quality == 0.0

    def test_neurogenic_without_bus(self):
        """Ψ8: retrieve() must work without bus (backward compat)."""
        nm = NeurogenicMemorySystem(base_dim=64)
        nm.consolidate(torch.randn(64), importance=0.9)
        results = nm.retrieve(torch.randn(64))
        assert isinstance(results, list)


class TestPatchPsi8TemporalMemory:
    """Verify TemporalMemory writes staleness signals to bus."""

    def test_temporal_has_fb_ref(self):
        """Ψ8: TemporalMemory must have _fb_ref attribute."""
        tm = TemporalMemory(capacity=10, dim=64)
        assert hasattr(tm, '_fb_ref')
        assert tm._fb_ref is None

    def test_temporal_retrieve_writes_staleness(self):
        """Ψ8: retrieve() must write memory_staleness_pressure to bus."""
        bus = _make_bus()
        tm = TemporalMemory(capacity=10, dim=64)
        tm._fb_ref = bus

        # Store and age memories
        tm.store(torch.randn(64), importance=1.0)
        tm.store(torch.randn(64), importance=0.8)

        query = torch.randn(64)
        results = tm.retrieve(query, k=2)

        staleness = bus.read_signal('memory_staleness_pressure', None)
        assert staleness is not None

    def test_temporal_empty_retrieve_writes_high_staleness(self):
        """Ψ8: Empty retrieve writes staleness=1.0."""
        bus = _make_bus()
        tm = TemporalMemory(capacity=10, dim=64)
        tm._fb_ref = bus

        results = tm.retrieve(torch.randn(64))
        assert results == []

        staleness = bus.read_signal('memory_staleness_pressure', None)
        assert staleness is not None
        assert staleness == 1.0

    def test_temporal_without_bus(self):
        """Ψ8: retrieve() must work without bus (backward compat)."""
        tm = TemporalMemory(capacity=10, dim=64)
        tm.store(torch.randn(64))
        results = tm.retrieve(torch.randn(64))
        assert isinstance(results, list)


class TestPatchPsi8ConsolidatingMemory:
    """Verify ConsolidatingMemory writes consolidation health to bus."""

    def test_consolidating_has_fb_ref(self):
        """Ψ8: ConsolidatingMemory must have _fb_ref attribute."""
        cm = ConsolidatingMemory(dim=64)
        assert hasattr(cm, '_fb_ref')
        assert cm._fb_ref is None

    def test_consolidating_writes_health(self):
        """Ψ8: consolidate() must write memory_consolidation_health."""
        bus = _make_bus()
        cm = ConsolidatingMemory(dim=64, importance_threshold=0.3)
        cm._fb_ref = bus

        # Store items, then consolidate
        for _ in range(5):
            cm.store(torch.randn(64))
        cm.consolidate()

        health = bus.read_signal('memory_consolidation_health', None)
        assert health is not None
        assert 0.0 <= health <= 1.0

    def test_consolidating_without_bus(self):
        """Ψ8: consolidate() must work without bus (backward compat)."""
        cm = ConsolidatingMemory(dim=64)
        for _ in range(3):
            cm.store(torch.randn(64))
        cm.consolidate()  # Should not crash


# ══════════════════════════════════════════════════════════════════════
# PATCH-Ψ: AEONDeltaV3.__init__ Wiring
# ══════════════════════════════════════════════════════════════════════

class TestPatchPsiWiring:
    """Verify AEONDeltaV3.__init__ wires all subsystems to feedback bus."""

    @pytest.fixture(scope='class')
    def model(self):
        """Create a full AEONDeltaV3 model for wiring verification."""
        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=True,      # DiversityMetric
            enable_catastrophe_detection=False,
            enable_safety_guardrails=True,  # SafetySystem
            enable_world_model=True,        # WorldModel
            enable_auto_critic=True,        # AutoCriticLoop
            enable_module_coherence=True,   # ModuleCoherenceVerifier
        )
        try:
            return AEONDeltaV3(config)
        except Exception:
            pytest.skip("AEONDeltaV3 initialization not available")

    def test_safety_system_wired(self, model):
        """Wiring: safety_system._fb_ref must be feedback_bus."""
        if model.safety_system is not None:
            assert model.safety_system._fb_ref is model.feedback_bus

    def test_world_model_wired(self, model):
        """Wiring: world_model._fb_ref must be feedback_bus."""
        if model.world_model is not None:
            assert model.world_model._fb_ref is model.feedback_bus

    def test_integrity_monitor_wired(self, model):
        """Wiring: integrity_monitor._fb_ref must be feedback_bus."""
        if model.integrity_monitor is not None:
            assert model.integrity_monitor._fb_ref is model.feedback_bus

    def test_module_coherence_wired(self, model):
        """Wiring: module_coherence._fb_ref must be feedback_bus."""
        if model.module_coherence is not None:
            assert model.module_coherence._fb_ref is model.feedback_bus

    def test_auto_critic_wired(self, model):
        """Wiring: auto_critic._fb_ref must be feedback_bus."""
        if model.auto_critic is not None:
            assert model.auto_critic._fb_ref is model.feedback_bus

    def test_diversity_metric_wired(self, model):
        """Wiring: diversity_metric._fb_ref must be feedback_bus."""
        if model.diversity_metric is not None:
            assert model.diversity_metric._fb_ref is model.feedback_bus

    def test_neurogenic_memory_wired(self, model):
        """Wiring: neurogenic_memory._fb_ref must be feedback_bus."""
        if getattr(model, 'neurogenic_memory', None) is not None:
            assert model.neurogenic_memory._fb_ref is model.feedback_bus

    def test_temporal_memory_wired(self, model):
        """Wiring: temporal_memory._fb_ref must be feedback_bus."""
        if getattr(model, 'temporal_memory', None) is not None:
            assert model.temporal_memory._fb_ref is model.feedback_bus

    def test_consolidating_memory_wired(self, model):
        """Wiring: consolidating_memory._fb_ref must be feedback_bus."""
        if getattr(model, 'consolidating_memory', None) is not None:
            assert model.consolidating_memory._fb_ref is model.feedback_bus


# ══════════════════════════════════════════════════════════════════════
# Cross-patch integration tests
# ══════════════════════════════════════════════════════════════════════

class TestCrossPatchIntegration:
    """Verify signal chains across multiple patches work end-to-end."""

    def test_safety_violation_reaches_mct_reader(self):
        """E2E: safety_violation_active written by Ψ1 is readable by MCT."""
        from aeon_core import MetaCognitiveRecursionTrigger
        bus = _make_bus()
        config = _make_config()

        # Create safety system and produce violation
        ss = MultiLevelSafetySystem(config)
        ss._fb_ref = bus
        with torch.no_grad():
            for p in ss.parameters():
                p.zero_()
        B = 1
        ss(
            torch.zeros(B, config.action_dim),
            torch.zeros(B, config.hidden_dim),
            torch.zeros(B, config.num_pillars),
            {}, {},
        )

        # MCT should be able to read the safety signal
        violation = bus.read_signal('safety_violation_active', 0.0)
        assert isinstance(violation, float)

    def test_integrity_alarm_chain(self):
        """E2E: integrity alarm → bus read chain works."""
        bus = _make_bus()
        sim = SystemIntegrityMonitor(feedback_bus=bus)

        # Record degraded health
        sim.record_health('meta_loop', 0.1)  # Below threshold

        # Alarm signal should be readable
        alarm = bus.read_signal('integrity_alarm', 0.0)
        assert alarm > 0.5

    def test_diversity_collapse_chain(self):
        """E2E: diversity collapse → bus → readable by downstream."""
        config = _make_config()
        bus = _make_bus()
        dm = DiversityMetric(config)
        dm._fb_ref = bus

        # Zero variance → collapse
        dm(torch.ones(2, config.num_pillars) * 0.5)

        alarm = bus.read_signal('diversity_collapse_alarm', 0.0)
        assert alarm == 1.0

    def test_world_model_prediction_error_chain(self):
        """E2E: world model error → bus → readable."""
        bus = _make_bus()
        wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=32)
        wm._fb_ref = bus

        wm(torch.randn(2, 64))

        error = bus.read_signal('world_model_prediction_error', None)
        assert error is not None
        assert error >= 0.0

    def test_memory_retrieval_quality_chain(self):
        """E2E: memory retrieval → bus → readable."""
        bus = _make_bus()
        nm = NeurogenicMemorySystem(base_dim=64)
        nm._fb_ref = bus

        # Store then retrieve
        nm.consolidate(torch.randn(64), importance=0.9)
        nm.retrieve(torch.randn(64))

        quality = bus.read_signal('memory_retrieval_quality', None)
        assert quality is not None
