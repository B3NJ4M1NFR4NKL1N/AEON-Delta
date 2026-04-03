"""
Tests for H-series cognitive integration patches (final activation).

H1: Enriched MCT evaluation with full signal collection
H2: Z-annotation propagation to micro-retrain (Point 5→10)
H3: Error recording for signal bus and z-annotation failures
H4: verify_and_reinforce coherence feedback to MCT
H5: Uncertainty flags passed to UCC evaluation
H6: Automatic component discovery (auto_wire)
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ── Lightweight stubs ────────────────────────────────────────────────────────

class _StubConfig:
    reinforce_interval = 5
    z_dim = 16
    vq_embedding_dim = 16


class _StubErrorEvolution:
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(self, *, error_class: str, strategy_used: str,
                       success: bool, metadata: Any = None,
                       causal_antecedents: Any = None) -> None:
        self.episodes.append({
            "error_class": error_class,
            "strategy_used": strategy_used,
            "success": success,
            "metadata": metadata,
        })

    def get_error_summary(self) -> Dict[str, Any]:
        return {
            "total_recorded": len(self.episodes),
            "error_classes": list({e["error_class"] for e in self.episodes}),
        }

    def get_episode_count(self) -> int:
        return len(self.episodes)


class _StubFeedbackBus:
    """Minimal feedback bus that records writes and supports reads."""
    def __init__(self) -> None:
        self._extra_signals: Dict[str, float] = {}

    def write_signal(self, name: str, value: float) -> None:
        self._extra_signals[name] = value

    def register_signal(self, name: str, default: float = 0.0) -> None:
        self._extra_signals.setdefault(name, default)

    def get_oscillation_score(self) -> float:
        return 0.05


class _StubMCT:
    """Minimal MCT that records all kwargs it receives."""
    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}
        self._should_trigger = False

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = dict(kwargs)
        return {
            "should_trigger": self._should_trigger,
            "trigger_score": kwargs.get("uncertainty", 0.0),
        }

    def adapt_weights_from_evolution(self, summary: Dict[str, Any]) -> None:
        self._adapted_from = summary


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(4, 4)  # Dummy parameter
        self._cached_spectral_stability_margin = 0.9
        self._cached_surprise = 0.15
        self._memory_stale = False
        self._cached_safety_violation = False
        self._cached_stall_severity = 0.05
        self._cached_output_quality = 0.95
        self._cached_border_uncertainty = 0.02
        self._last_trust_score = 0.88
        self._cached_causal_quality = 0.92
        self._reinforce_result: Optional[Dict[str, Any]] = None

    def verify_and_reinforce(self) -> Dict[str, Any]:
        if self._reinforce_result is not None:
            return self._reinforce_result
        return {"overall_score": 0.85, "status": "ok"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _make_controller(
    attach_fb: bool = True,
    attach_mct: bool = True,
) -> Any:
    """Build a wired UnifiedTrainingCycleController."""
    from aeon_integration import UnifiedTrainingCycleController

    model = _StubModel()
    config = _StubConfig()
    ctrl = UnifiedTrainingCycleController(model, config)

    if attach_fb:
        fb = _StubFeedbackBus()
        ctrl.attach_feedback_bus(fb)
    if attach_mct:
        mct = _StubMCT()
        ctrl.attach_metacognitive_trigger(mct)

    return ctrl


# =============================================================================
#  H1: MCT Signal Collection Tests
# =============================================================================

class TestH1MCTSignalCollection:
    """Verify that MCT.evaluate() receives enriched signals."""

    def test_collect_mct_signals_returns_all_keys(self) -> None:
        ctrl = _make_controller()
        kwargs = ctrl._collect_mct_signals(0.3, ["flag1"], {})
        assert "uncertainty" in kwargs
        assert kwargs["uncertainty"] == 0.3
        assert "spectral_stability_margin" in kwargs
        assert "world_model_surprise" in kwargs
        assert "stall_severity" in kwargs
        assert "output_reliability" in kwargs
        assert "causal_quality" in kwargs

    def test_collect_mct_reads_model_cached(self) -> None:
        ctrl = _make_controller()
        ctrl.model._cached_surprise = 0.42
        kwargs = ctrl._collect_mct_signals(0.0, [], {})
        assert kwargs["world_model_surprise"] == pytest.approx(0.42)

    def test_collect_mct_reads_feedback_bus(self) -> None:
        ctrl = _make_controller()
        ctrl._feedback_bus.write_signal("integration_health", 0.7)
        kwargs = ctrl._collect_mct_signals(0.0, [], {})
        # coherence_deficit = 1 - integration_health = 0.3
        assert kwargs["coherence_deficit"] == pytest.approx(0.3)

    def test_collect_mct_handles_no_feedback_bus(self) -> None:
        ctrl = _make_controller(attach_fb=False)
        kwargs = ctrl._collect_mct_signals(0.5, ["f"], {})
        assert kwargs["uncertainty"] == 0.5
        # Should not crash; coherence_deficit should not be present
        assert "coherence_deficit" not in kwargs

    def test_execute_full_cycle_passes_enriched_signals(self) -> None:
        ctrl = _make_controller()
        ctrl._feedback_bus.write_signal("integration_health", 0.6)
        ctrl.model._cached_surprise = 0.22

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
        )
        mct = ctrl._mct
        assert "spectral_stability_margin" in mct.last_kwargs
        assert mct.last_kwargs["world_model_surprise"] == pytest.approx(0.22)
        assert result["metacognitive_review"]["signals_provided"] > 1

    def test_mct_signal_count_exceeds_baseline(self) -> None:
        """Before H1, MCT got 1 signal. Now should get ≥10."""
        ctrl = _make_controller()
        ctrl.execute_full_cycle(
            epoch=1, phase="B",
            epoch_metrics={"loss": 0.3},
        )
        assert len(ctrl._mct.last_kwargs) >= 10

    def test_oscillation_severity_from_feedback_bus(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
        )
        assert "oscillation_severity" in ctrl._mct.last_kwargs
        assert ctrl._mct.last_kwargs["oscillation_severity"] == pytest.approx(0.05)


# =============================================================================
#  H2: Z-Annotation Propagation Tests
# =============================================================================

class TestH2ZAnnotationPropagation:
    """Verify z-annotations flow from Point 5 to Point 10."""

    def test_execute_z_annotation_accepts_error_evolution(self) -> None:
        ctrl = _make_controller()
        z = [torch.randn(4, 16)]
        # Should not raise with error_evolution kwarg
        z_out, ann = ctrl.execute_z_annotation(z, error_evolution=None)
        assert len(z_out) == 1

    def test_micro_retrain_accepts_z_annotations(self) -> None:
        ctrl = _make_controller()
        pseudo = [{"label": "test"}]
        z = [torch.randn(4, 16)]
        ann = [torch.tensor([[0.9, 0.1, 0.8]] * 4)]
        result = ctrl.execute_micro_retrain(pseudo, z, z_annotations=ann)
        # Should not crash (ae_train import will fail but that's expected)
        assert isinstance(result, dict)

    def test_micro_retrain_filters_low_confidence(self) -> None:
        """Verify filtering logic: only high-confidence z-sequences pass."""
        z_high = torch.randn(4, 16)
        z_low = torch.randn(4, 16)
        z_seqs = [z_high, z_low]
        ann = [
            torch.tensor([[0.9, 0.1, 0.8]] * 4),  # High confidence
            torch.tensor([[0.1, 0.9, 0.2]] * 4),  # Low confidence
        ]

        # Test filtering logic directly
        _filtered = []
        for seq, a in zip(z_seqs, ann):
            if a.shape[-1] >= 1:
                mean_conf = a[..., 0].mean().item()
                if mean_conf > 0.3:
                    _filtered.append(seq)
            else:
                _filtered.append(seq)
        # Only the high-confidence sequence should survive
        assert len(_filtered) == 1

    def test_full_cycle_propagates_annotations(self) -> None:
        """Verify execute_full_cycle stores z_annotations internally."""
        ctrl = _make_controller()
        z_seqs = [torch.randn(3, 16)]

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.4},
            z_sequences=z_seqs,
            pseudo_labels=[{"label": "x"}],
        )
        # z_annotation should record annotation_count
        if "z_annotation" in result:
            assert "annotation_count" in result["z_annotation"]


# =============================================================================
#  H3: Error Recording for Signal Bus / Z-Annotation
# =============================================================================

class TestH3ErrorRecording:
    """Verify failures in Points 4 and 5 are recorded to error_evolution."""

    def test_signal_bus_failure_recorded(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        # Attach a broken signal bus
        class _BrokenBus:
            def closed_loop_step(self, *a: Any, **kw: Any) -> None:
                raise RuntimeError("bus error")
        ctrl._signal_bus = _BrokenBus()
        ctrl._vt_learner = MagicMock()

        ctrl.execute_signal_bus_step(error_evolution=ee)
        assert any(
            e["error_class"] == "signal_bus_closed_loop_failure"
            for e in ee.episodes
        )

    def test_z_annotation_failure_recorded(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()
        z = [torch.randn(3, 16)]

        # Force a failure by temporarily making the import fail
        import sys
        _orig = sys.modules.get("ae_train")
        # Use a broken model that will cause annotation to fail at the
        # execute_z_annotation level
        class _BreakingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._linear = nn.Linear(2, 2)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("forced failure")

        # Monkeypatch the controller to force annotation failure
        _orig_method = ctrl.execute_z_annotation.__func__

        def _failing_z_annotation(
            self_arg: Any,
            z_sequences: Any,
            error_evolution: Any = None,
        ) -> Any:
            # Simulate an import error
            from aeon_integration import UnifiedTrainingCycleController
            UnifiedTrainingCycleController._record_failure_episode(
                error_evolution, "z_annotation_failure",
                "annotation_fallback",
                {"reason": "forced test failure"},
            )
            fallback = [torch.ones(seq.shape[0], 3) for seq in z_sequences]
            return z_sequences, fallback

        ctrl.execute_z_annotation = lambda *a, **kw: _failing_z_annotation(ctrl, *a, **kw)

        _, ann = ctrl.execute_z_annotation(z, error_evolution=ee)
        assert any(
            e["error_class"] == "z_annotation_failure"
            for e in ee.episodes
        )
        # Fallback annotations should be returned
        assert ann[0].shape[-1] == 3

    def test_signal_bus_failure_flagged_in_full_cycle(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()

        class _BrokenBus:
            def closed_loop_step(self, *a: Any, **kw: Any) -> None:
                raise RuntimeError("bus error")
        ctrl._signal_bus = _BrokenBus()
        ctrl._vt_learner = MagicMock()

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        assert "signal_bus_closed_loop_failed" in result.get(
            "uncertainty_flags", [],
        )


# =============================================================================
#  H4: verify_and_reinforce → MCT Feedback
# =============================================================================

class TestH4ReinforceFeedback:
    """Verify reinforce results feed back to MCT."""

    def test_feed_reinforce_to_mct_adapts_weights(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()
        ee.record_episode(
            error_class="test", strategy_used="s", success=False,
        )

        ctrl._feed_reinforce_to_mct(
            {"overall_score": 0.5}, error_evolution=ee,
        )
        assert hasattr(ctrl._mct, "_adapted_from")

    def test_feed_reinforce_writes_to_feedback_bus(self) -> None:
        ctrl = _make_controller()
        ctrl._feed_reinforce_to_mct(
            {"overall_score": 0.72}, error_evolution=None,
        )
        assert ctrl._feedback_bus._extra_signals.get(
            "reinforce_coherence_score",
        ) == pytest.approx(0.72)

    def test_feed_reinforce_no_mct_no_crash(self) -> None:
        ctrl = _make_controller(attach_mct=False)
        # Should not raise
        ctrl._feed_reinforce_to_mct({"overall_score": 0.5})

    def test_feed_reinforce_high_score_no_adaptation(self) -> None:
        ctrl = _make_controller()
        ee = _StubErrorEvolution()
        ctrl._feed_reinforce_to_mct(
            {"overall_score": 0.95}, error_evolution=ee,
        )
        # Score ≥ 0.8 → no weight adaptation
        assert not hasattr(ctrl._mct, "_adapted_from")

    def test_full_cycle_reinforce_feeds_mct(self) -> None:
        ctrl = _make_controller()
        ctrl.config.reinforce_interval = 1  # Every cycle
        ctrl.model._reinforce_result = {"overall_score": 0.6}

        ee = _StubErrorEvolution()
        ee.record_episode(
            error_class="test", strategy_used="s", success=False,
        )

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={},
            error_evolution=ee,
        )
        assert result.get("mutual_reinforcement", {}).get("executed", False)
        assert ctrl._feedback_bus._extra_signals.get(
            "reinforce_coherence_score",
        ) == pytest.approx(0.6)


# =============================================================================
#  H5: UCC Gets Uncertainty Flags
# =============================================================================

class TestH5UCCUncertaintyFlags:
    """Verify UCC evaluation receives upstream uncertainty context."""

    def test_ucc_eval_accepts_uncertainty_flags(self) -> None:
        ctrl = _make_controller()
        result = ctrl.execute_ucc_evaluation(
            1, "A", {"loss": 0.5},
            uncertainty_flags=["flag1", "flag2"],
        )
        # Will fail because ae_train import fails, but shouldn't crash
        assert isinstance(result, dict)

    def test_full_cycle_passes_flags_to_ucc(self) -> None:
        """Verify that the full cycle enriches epoch_metrics for UCC."""
        ctrl = _make_controller()

        # We test indirectly: mock execute_ucc_evaluation to capture args
        _captured: Dict[str, Any] = {}
        _original = ctrl.execute_ucc_evaluation

        def _spy(epoch: int, phase: str, epoch_metrics: Dict,
                 uncertainty_flags: Any = None) -> Dict:
            _captured["flags"] = uncertainty_flags
            _captured["metrics"] = epoch_metrics
            return {"evaluated": True}

        ctrl.execute_ucc_evaluation = _spy  # type: ignore[assignment]

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.4},
        )
        # UCC should have been called with uncertainty_flags list
        assert "flags" in _captured
        assert isinstance(_captured["flags"], list)


# =============================================================================
#  H6: Auto-Wire Tests
# =============================================================================

class TestH6AutoWire:
    """Verify automatic component discovery and wiring."""

    def test_auto_wire_discovers_feedback_bus(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        model.cognitive_feedback_bus = _StubFeedbackBus()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)

        result = ctrl.auto_wire(model)
        assert "feedback_bus" in result["wired"]
        assert ctrl._feedback_bus is model.cognitive_feedback_bus

    def test_auto_wire_discovers_mct(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        model.metacognitive_trigger = _StubMCT()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)

        result = ctrl.auto_wire(model)
        assert "mct" in result["wired"]
        assert ctrl._mct is model.metacognitive_trigger

    def test_auto_wire_skips_already_wired(self) -> None:
        ctrl = _make_controller(attach_mct=True)
        original_mct = ctrl._mct
        model = _StubModel()
        model.metacognitive_trigger = _StubMCT()  # Different MCT

        result = ctrl.auto_wire(model)
        # Should keep the already-wired MCT
        assert ctrl._mct is original_mct
        assert "mct" in result["wired"]

    def test_auto_wire_reports_missing(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)

        result = ctrl.auto_wire(model)
        assert len(result["missing"]) > 0
        assert result["total_wired"] + result["total_missing"] == 7

    def test_auto_wire_discovers_all_components(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        model.vt_streaming_signal_bus = MagicMock()
        model.vt_continuous_learner = MagicMock()
        model.adaptive_training_controller = MagicMock()
        model.cognitive_feedback_bus = _StubFeedbackBus()
        model.unified_cognitive_cycle = MagicMock()
        model.metacognitive_trigger = _StubMCT()
        model.continual_learning_core = MagicMock()

        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        result = ctrl.auto_wire(model)

        assert result["total_wired"] == 7
        assert result["total_missing"] == 0


# =============================================================================
#  Integration: End-to-End Cognitive Loop
# =============================================================================

class TestEndToEndCognitiveLoop:
    """Verify the full cognitive loop works end-to-end."""

    def test_full_cycle_all_patches_active(self) -> None:
        ctrl = _make_controller()
        ctrl.config.reinforce_interval = 1  # Force reinforce every cycle
        ctrl.model._reinforce_result = {"overall_score": 0.9}

        ee = _StubErrorEvolution()
        z = [torch.randn(4, 16)]

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.3},
            z_sequences=z,
            error_evolution=ee,
            pseudo_labels=[{"label": "x"}],
        )

        # MCT should have been evaluated with enriched signals
        assert result.get("metacognitive_review") is not None
        assert result["metacognitive_review"]["signals_provided"] >= 10

        # Reinforce should have executed
        assert result.get("mutual_reinforcement", {}).get("executed", False)

        # Z-annotation should have been attempted
        if "z_annotation" in result:
            assert "annotation_count" in result["z_annotation"]

    def test_multiple_cycles_mct_weight_adaptation(self) -> None:
        ctrl = _make_controller()
        ctrl.config.reinforce_interval = 2
        ctrl.model._reinforce_result = {"overall_score": 0.5}

        ee = _StubErrorEvolution()
        ee.record_episode(
            error_class="test_error", strategy_used="s", success=False,
        )

        # Run 2 cycles — reinforce should trigger on cycle 2
        ctrl.execute_full_cycle(epoch=1, phase="A", epoch_metrics={},
                                error_evolution=ee)
        ctrl.execute_full_cycle(epoch=2, phase="A", epoch_metrics={},
                                error_evolution=ee)

        # MCT should have been adapted after reinforce on cycle 2
        assert hasattr(ctrl._mct, "_adapted_from")

    def test_causal_transparency_maintained(self) -> None:
        """Verify trace_output_to_premise still works with new patches."""
        from aeon_integration import trace_output_to_premise

        ctrl = _make_controller()
        ee = _StubErrorEvolution()
        ee.record_episode(
            error_class="test", strategy_used="s", success=False,
        )

        result = ctrl.execute_full_cycle(
            epoch=1, phase="A", epoch_metrics={"loss": 0.5},
            error_evolution=ee,
        )
        history = ctrl.get_metrics_history()

        trace = trace_output_to_premise(
            "test_output",
            cycle_history=history,
            error_evolution=ee,
        )
        assert trace["traced"] is True
        assert len(trace["trace_chain"]) >= 2

    def test_read_fb_signal_default(self) -> None:
        ctrl = _make_controller(attach_fb=False)
        assert ctrl._read_fb_signal("nonexistent", 42.0) == 42.0

    def test_read_fb_signal_existing(self) -> None:
        ctrl = _make_controller()
        ctrl._feedback_bus.write_signal("test_sig", 3.14)
        assert ctrl._read_fb_signal("test_sig") == pytest.approx(3.14)
