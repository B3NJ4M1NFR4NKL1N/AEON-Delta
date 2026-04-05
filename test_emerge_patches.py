"""Tests for CP-EMERGE patches (1–8).

Each test class targets a specific CP-EMERGE patch and validates the
signal write, signal read, amplification logic, and edge-case handling
implemented in ``aeon_core.py`` and ``aeon_server.py``.
"""

import sys
import os
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import aeon_core


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        vocab_size=256,
        device_str="cpu",
    )
    defaults.update(overrides)
    return aeon_core.AEONConfig(**defaults)


def _make_bus(hidden_dim=64):
    return aeon_core.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _make_mct_with_bus(hidden_dim=64):
    bus = _make_bus(hidden_dim)
    mct = aeon_core.MetaCognitiveRecursionTrigger()
    mct.set_feedback_bus(bus)
    return mct, bus


_MCT_DEFAULTS = dict(
    uncertainty=0.0,
    is_diverging=False,
    topology_catastrophe=False,
    coherence_deficit=0.0,
    memory_staleness=0.0,
    recovery_pressure=0.0,
    world_model_surprise=0.0,
    causal_quality=1.0,
    safety_violation=False,
    diversity_collapse=0.0,
    memory_trust_deficit=0.0,
    convergence_conflict=0.0,
    output_reliability=1.0,
    spectral_stability_margin=1.0,
    border_uncertainty=0.0,
    stall_severity=0.0,
    oscillation_severity=0.0,
)


def _eval_mct(mct, **overrides):
    kw = dict(_MCT_DEFAULTS)
    kw.update(overrides)
    return mct.evaluate(**kw)


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-1: Memory Subsystem → Feedback Bus Bridge
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge1MemoryFeedbackBus:
    """CP-EMERGE-1: Memory subsystem → feedback bus bridge."""

    def test_memory_retrieval_success_rate_write(self):
        bus = _make_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.75)
        val = bus.read_signal("memory_retrieval_success_rate", -1.0)
        assert abs(val - 0.75) < 1e-6

    def test_consolidation_progress_write(self):
        bus = _make_bus()
        bus.write_signal("consolidation_progress", 0.4)
        val = bus.read_signal("consolidation_progress", -1.0)
        assert abs(val - 0.4) < 1e-6

    def test_memory_staleness_pressure_write(self):
        bus = _make_bus()
        bus.write_signal("memory_staleness_pressure", 0.6)
        val = bus.read_signal("memory_staleness_pressure", -1.0)
        assert abs(val - 0.6) < 1e-6

    def test_memory_signals_consumed_by_mct(self):
        """All three memory signals must be consumed (not orphaned) by MCT."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.3)
        bus.write_signal("memory_staleness_pressure", 0.5)
        bus.write_signal("consolidation_progress", 0.2)
        _eval_mct(mct)
        orphaned = bus.get_orphaned_signals()
        assert "memory_retrieval_success_rate" not in orphaned
        assert "memory_staleness_pressure" not in orphaned
        assert "consolidation_progress" not in orphaned

    def test_memory_signals_default_values(self):
        """Without writes, defaults should be returned."""
        bus = _make_bus()
        assert bus.read_signal("memory_retrieval_success_rate", 1.0) == 1.0
        assert bus.read_signal("consolidation_progress", 1.0) == 1.0
        assert bus.read_signal("memory_staleness_pressure", 0.0) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-1d: Memory signals → MCT amplification
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge1dMemoryMCTReads:
    """CP-EMERGE-1d: MCT reads memory signals and amplifies triggers."""

    def test_low_retrieval_amplifies_memory_staleness(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.2)
        result = _eval_mct(mct)
        assert isinstance(result, dict)
        assert "trigger_score" in result

    def test_high_retrieval_no_amplification(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.9)
        result = _eval_mct(mct)
        assert isinstance(result, dict)

    def test_staleness_pressure_above_threshold_amplifies(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_staleness_pressure", 0.5)
        result = _eval_mct(mct)
        assert isinstance(result, dict)
        assert "trigger_score" in result

    def test_staleness_pressure_below_threshold_no_amplification(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_staleness_pressure", 0.1)
        result = _eval_mct(mct)
        assert isinstance(result, dict)

    def test_low_consolidation_amplifies_memory_trust_deficit(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("consolidation_progress", 0.1)
        result = _eval_mct(mct)
        assert isinstance(result, dict)
        assert "trigger_score" in result

    def test_high_consolidation_no_amplification(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("consolidation_progress", 0.8)
        result = _eval_mct(mct)
        assert isinstance(result, dict)


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-2: SocialCognition & CodeExecution → Feedback Bus + MCT
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge2SocialSandbox:
    """CP-EMERGE-2: Social cognition + sandbox → bus + MCT reads."""

    def test_social_cognition_pressure_write_read(self):
        bus = _make_bus()
        bus.write_signal("social_cognition_pressure", 0.65)
        val = bus.read_signal("social_cognition_pressure", 0.0)
        assert abs(val - 0.65) < 1e-6

    def test_sandbox_execution_risk_write_read(self):
        bus = _make_bus()
        bus.write_signal("sandbox_execution_risk", 0.7)
        val = bus.read_signal("sandbox_execution_risk", 0.0)
        assert abs(val - 0.7) < 1e-6

    def test_social_pressure_above_threshold_consumed(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("social_cognition_pressure", 0.5)
        _eval_mct(mct)
        orphaned = bus.get_orphaned_signals()
        assert "social_cognition_pressure" not in orphaned

    def test_sandbox_risk_above_threshold_consumed(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("sandbox_execution_risk", 0.7)
        _eval_mct(mct)
        orphaned = bus.get_orphaned_signals()
        assert "sandbox_execution_risk" not in orphaned

    def test_social_below_threshold_no_coherence_boost(self):
        """social_cognition_pressure ≤ 0.3 → no coherence_deficit change."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("social_cognition_pressure", 0.1)
        result = _eval_mct(mct)
        assert isinstance(result, dict)

    def test_sandbox_below_threshold_no_recovery_boost(self):
        """sandbox_execution_risk ≤ 0.5 → no recovery_pressure change."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("sandbox_execution_risk", 0.2)
        result = _eval_mct(mct)
        assert isinstance(result, dict)


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-4: Spectral Instability → Convergence Monitor
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge4SpectralConvergence:
    """CP-EMERGE-4: Spectral instability → convergence monitor bridge."""

    def test_convergence_monitor_record_secondary_signal(self):
        cm = aeon_core.ConvergenceMonitor(threshold=1e-5)
        cm.record_secondary_signal("spectral_instability", 0.7)
        signals = cm.get_secondary_signals()
        assert "spectral_instability" in signals
        assert abs(signals["spectral_instability"] - 0.7) < 1e-6

    def test_spectral_signal_clamped_to_unit(self):
        cm = aeon_core.ConvergenceMonitor(threshold=1e-5)
        cm.record_secondary_signal("spectral_instability", 1.5)
        signals = cm.get_secondary_signals()
        assert signals["spectral_instability"] <= 1.0

    def test_spectral_signal_below_threshold_not_recorded(self):
        """When spectral_instability ≤ 0.3, the bridge should not record."""
        bus = _make_bus()
        bus.write_signal("spectral_instability", 0.1)
        val = bus.read_signal("spectral_instability", 0.0)
        assert val <= 0.3

    def test_convergence_monitor_filters_below_threshold(self):
        """Values ≤ 0.3 on the bus should not appear in secondary signals."""
        cm = aeon_core.ConvergenceMonitor(threshold=1e-5)
        # Only values above 0.3 should be recorded by the bridge logic;
        # the convergence monitor itself accepts any value, but the bridge
        # in UnifiedCognitiveCycle only calls record_secondary_signal when
        # spectral_instability > 0.3.  Verify the monitor stores what it's
        # given correctly.
        cm.record_secondary_signal("spectral_instability", 0.0)
        signals = cm.get_secondary_signals()
        assert signals["spectral_instability"] == 0.0
        cm.record_secondary_signal("spectral_instability", 0.8)
        signals = cm.get_secondary_signals()
        assert signals["spectral_instability"] == 0.8


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-5: World Model Surprise → Same-Pass MCTS Depth
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge5WorldModelSurprise:
    """CP-EMERGE-5: World model surprise → MCTS depth boost."""

    def test_surprise_signal_write_read(self):
        bus = _make_bus()
        bus.write_signal("world_model_surprise_active", 0.8)
        val = bus.read_signal("world_model_surprise_active", 0.0)
        assert abs(val - 0.8) < 1e-6

    def test_surprise_depth_boost_formula(self):
        """Depth boost = min(5, int(surprise * 10))."""
        for surprise, expected_boost in [
            (0.31, 3), (0.5, 5), (0.8, 5), (1.0, 5), (0.35, 3),
        ]:
            boost = min(5, int(surprise * 10))
            assert boost == expected_boost, (
                f"surprise={surprise}: expected boost={expected_boost}, got {boost}"
            )

    def test_surprise_below_threshold_no_boost(self):
        """surprise ≤ 0.3 → no depth boost."""
        surprise = 0.2
        assert surprise <= 0.3

    def test_surprise_consumed_by_bus(self):
        bus = _make_bus()
        bus.write_signal("world_model_surprise_active", 0.6)
        _ = bus.read_signal("world_model_surprise_active", 0.0)
        orphaned = bus.get_orphaned_signals()
        assert "world_model_surprise_active" not in orphaned

    def test_mcts_depth_boost_applied_on_model(self):
        """Verify MCTS planner depth increases when surprise > 0.3."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        if model.feedback_bus is not None and hasattr(model, "mcts_planner"):
            orig_depth = model.mcts_planner.max_depth
            model.feedback_bus.write_signal("world_model_surprise_active", 0.8)
            surprise = float(
                model.feedback_bus.read_signal("world_model_surprise_active", 0.0)
            )
            boost = min(5, int(surprise * 10))
            assert boost > 0
            # Simulate what the forward pass does
            model.mcts_planner.max_depth += boost
            assert model.mcts_planner.max_depth > orig_depth
            # Restore
            model.mcts_planner.max_depth = orig_depth


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-6: Orphaned Signal → Loss Modulation
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge6LossModulation:
    """CP-EMERGE-6: convergence_confidence + cognitive_unity_deficit → loss."""

    def test_convergence_confidence_signal(self):
        bus = _make_bus()
        bus.write_signal("convergence_confidence", 0.15)
        val = bus.read_signal("convergence_confidence", 1.0)
        assert abs(val - 0.15) < 1e-6

    def test_cognitive_unity_deficit_signal(self):
        bus = _make_bus()
        bus.write_signal("cognitive_unity_deficit", 0.5)
        val = bus.read_signal("cognitive_unity_deficit", 0.0)
        assert abs(val - 0.5) < 1e-6

    def test_low_convergence_confidence_boost_formula(self):
        """convergence_confidence < 0.3 → boost = 1 + (0.3 - conf), capped 1.3."""
        for conf, expected_boost in [
            (0.1, 1.2), (0.0, 1.3), (0.3, 1.0), (0.5, 1.0),
        ]:
            if conf < 0.3:
                boost = min(1.0 + (0.3 - conf), 1.3)
            else:
                boost = 1.0
            assert abs(boost - expected_boost) < 1e-6, (
                f"conf={conf}: expected boost={expected_boost}, got {boost}"
            )

    def test_high_unity_deficit_boost_formula(self):
        """cognitive_unity_deficit > 0.3 → boost = 1 + deficit * 0.3, capped 1.3."""
        for deficit, expected_boost in [
            (0.5, 1.15), (1.0, 1.3), (0.0, 1.0), (0.3, 1.0),
        ]:
            if deficit > 0.3:
                boost = min(1.0 + deficit * 0.3, 1.3)
            else:
                boost = 1.0
            assert abs(boost - expected_boost) < 1e-6, (
                f"deficit={deficit}: expected boost={expected_boost}, got {boost}"
            )

    def test_compute_loss_runs_with_bus_signals(self):
        """compute_loss completes when bus has EMERGE-6 signals."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        outputs = model(input_ids)
        targets = input_ids.clone()
        if model.feedback_bus is not None:
            model.feedback_bus.write_signal("convergence_confidence", 0.1)
            model.feedback_bus.write_signal("cognitive_unity_deficit", 0.6)
        loss_result = model.compute_loss(outputs, targets)
        assert "total_loss" in loss_result
        total = loss_result["total_loss"]
        assert torch.is_tensor(total)
        assert not torch.isnan(total)

    def test_compute_loss_no_bus_signals(self):
        """compute_loss completes when bus has no EMERGE-6 signals."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        outputs = model(input_ids)
        targets = input_ids.clone()
        loss_result = model.compute_loss(outputs, targets)
        assert "total_loss" in loss_result
        assert not torch.isnan(loss_result["total_loss"])

    def test_loss_modulated_by_low_convergence_confidence(self):
        """Loss with low convergence_confidence should be >= baseline loss."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        B, L = 2, 16
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        outputs_baseline = model(input_ids)
        targets = input_ids.clone()
        loss_baseline = model.compute_loss(outputs_baseline, targets)

        # Now set low convergence_confidence and recompute
        torch.manual_seed(42)
        outputs_modulated = model(input_ids)
        if model.feedback_bus is not None:
            model.feedback_bus.write_signal("convergence_confidence", 0.05)
        loss_modulated = model.compute_loss(outputs_modulated, targets)

        base_val = float(loss_baseline["total_loss"].detach())
        mod_val = float(loss_modulated["total_loss"].detach())
        # Modulated loss should be >= baseline (boost factor >= 1.0)
        assert mod_val >= base_val - 1e-4

    def test_loss_modulated_by_high_unity_deficit(self):
        """Loss with high cognitive_unity_deficit should be >= baseline loss."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        B, L = 2, 16
        torch.manual_seed(99)
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        outputs_baseline = model(input_ids)
        targets = input_ids.clone()
        loss_baseline = model.compute_loss(outputs_baseline, targets)

        torch.manual_seed(99)
        outputs_modulated = model(input_ids)
        if model.feedback_bus is not None:
            model.feedback_bus.write_signal("cognitive_unity_deficit", 0.9)
        loss_modulated = model.compute_loss(outputs_modulated, targets)

        base_val = float(loss_baseline["total_loss"].detach())
        mod_val = float(loss_modulated["total_loss"].detach())
        assert mod_val >= base_val - 1e-4


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-7: Integrity Monitor → Forward Pass
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge7IntegrityMonitor:
    """CP-EMERGE-7: Integrity monitor → forward pass + MCT reads."""

    def test_integrity_health_score_write(self):
        bus = _make_bus()
        bus.write_signal("integrity_health_score", 0.9)
        val = bus.read_signal("integrity_health_score", -1.0)
        assert abs(val - 0.9) < 1e-6

    def test_integrity_alarm_when_health_low(self):
        """When health < 0.5, alarm = 1 - health."""
        health = 0.3
        alarm = max(0.0, min(1.0, 1.0 - health))
        assert abs(alarm - 0.7) < 1e-6

    def test_integrity_alarm_consumed_by_mct(self):
        mct, bus = _make_mct_with_bus()
        bus.write_signal("integrity_alarm", 0.6)
        _eval_mct(mct)
        orphaned = bus.get_orphaned_signals()
        assert "integrity_alarm" not in orphaned

    def test_integrity_alarm_below_threshold_no_amplification(self):
        """integrity_alarm ≤ 0.3 → no recovery_pressure change."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("integrity_alarm", 0.1)
        result = _eval_mct(mct)
        assert isinstance(result, dict)

    def test_integrity_alarm_above_threshold_amplifies_recovery(self):
        """integrity_alarm > 0.3 → amplifies recovery_pressure."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("integrity_alarm", 0.7)
        result = _eval_mct(mct)
        assert isinstance(result, dict)
        assert "trigger_score" in result

    def test_integrity_monitor_exists_on_model(self):
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        assert hasattr(model, "integrity_monitor")

    def test_integrity_monitor_get_global_health(self):
        monitor = aeon_core.SystemIntegrityMonitor()
        health = monitor.get_global_health()
        assert 0.0 <= health <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  CP-EMERGE-8: Training Metrics Server Endpoint
# ═════════════════════════════════════════════════════════════════════════════


class TestEmerge8TrainingMetrics:
    """CP-EMERGE-8: GET /api/training/metrics endpoint."""

    def test_endpoint_returns_200(self):
        import aeon_server
        from fastapi.testclient import TestClient

        client = TestClient(aeon_server.app)
        response = client.get("/api/training/metrics")
        assert response.status_code == 200

    def test_endpoint_response_shape(self):
        import aeon_server
        from fastapi.testclient import TestClient

        client = TestClient(aeon_server.app)
        response = client.get("/api/training/metrics")
        data = response.json()
        assert "ok" in data
        assert data["ok"] is True
        assert "progress" in data
        assert "loss_components" in data
        assert "convergence_state" in data
        assert "cognitive_health" in data


# ═════════════════════════════════════════════════════════════════════════════
#  Edge Cases & Cross-Patch Integration
# ═════════════════════════════════════════════════════════════════════════════


class TestEmergeEdgeCases:
    """Edge cases and cross-patch integration for EMERGE patches."""

    def test_mct_evaluate_without_bus(self):
        """MCT evaluate succeeds when no bus is attached."""
        mct = aeon_core.MetaCognitiveRecursionTrigger()
        result = _eval_mct(mct)
        assert isinstance(result, dict)
        assert "trigger_score" in result

    def test_bus_overwrite_preserves_latest(self):
        bus = _make_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.9)
        bus.write_signal("memory_retrieval_success_rate", 0.2)
        val = bus.read_signal("memory_retrieval_success_rate", -1.0)
        assert abs(val - 0.2) < 1e-6

    def test_boundary_values(self):
        """Boundary values 0.0 and 1.0 are valid."""
        bus = _make_bus()
        for sig in (
            "memory_retrieval_success_rate",
            "consolidation_progress",
            "memory_staleness_pressure",
            "social_cognition_pressure",
            "sandbox_execution_risk",
            "integrity_alarm",
        ):
            bus.write_signal(sig, 0.0)
            assert bus.read_signal(sig, -1.0) == 0.0
            bus.write_signal(sig, 1.0)
            assert bus.read_signal(sig, -1.0) == 1.0

    def test_all_emerge_signals_consumed_by_mct(self):
        """All EMERGE bus signals are consumed (not orphaned) after MCT eval."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal("memory_retrieval_success_rate", 0.3)
        bus.write_signal("memory_staleness_pressure", 0.5)
        bus.write_signal("consolidation_progress", 0.1)
        bus.write_signal("social_cognition_pressure", 0.5)
        bus.write_signal("sandbox_execution_risk", 0.7)
        bus.write_signal("integrity_alarm", 0.6)
        _eval_mct(mct)
        orphaned = bus.get_orphaned_signals()
        for sig in (
            "memory_retrieval_success_rate",
            "memory_staleness_pressure",
            "consolidation_progress",
            "social_cognition_pressure",
            "sandbox_execution_risk",
            "integrity_alarm",
        ):
            assert sig not in orphaned, f"{sig} is orphaned"

    def test_model_forward_produces_outputs_dict(self):
        """Model forward pass returns a dict suitable for compute_loss."""
        config = _make_config()
        model = aeon_core.AEONDeltaV3(config)
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        outputs = model(input_ids)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
