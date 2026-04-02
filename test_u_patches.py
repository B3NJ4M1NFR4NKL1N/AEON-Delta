"""
Tests for U-series patches — Final Integration & Cognitive Activation.

U1: Auto-wire component discovery (UnifiedTrainingCycleController.auto_wire)
U2: Wizard→cognitive bridge (seed_from_wizard)
U3: Dynamic feedback signal registration in _cognitive_activation_probe
U4: Bidirectional training↔inference bridge in execute_full_cycle
U4b: Meta-cognitive uncertainty check in execute_full_cycle
U5: Causal transparency — trace_output_to_premise
"""

from __future__ import annotations

import inspect
import logging
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────
#  Lightweight stubs — avoid importing heavyweight aeon_core
# ────────────────────────────────────────────────────────────────────────

class _StubErrorEvolution:
    """Minimal error-evolution tracker for testing."""

    def __init__(self) -> None:
        self._episodes: List[Dict[str, Any]] = []

    def record_episode(
        self,
        error_class: str = "",
        strategy_used: str = "",
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._episodes.append({
            "error_class": error_class,
            "strategy_used": strategy_used,
            "success": success,
            "metadata": metadata or {},
        })

    def get_error_summary(self) -> Dict[str, Any]:
        classes: Dict[str, int] = {}
        for ep in self._episodes:
            c = ep["error_class"]
            classes[c] = classes.get(c, 0) + 1
        return {
            "total_recorded": len(self._episodes),
            "error_classes": classes,
        }


class _StubMCT:
    """Minimal metacognitive trigger stub."""

    def __init__(self) -> None:
        self._adapted = False
        self._trigger_history: List[Dict] = []

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        unc = kwargs.get("uncertainty", 0.0)
        should = unc > 0.5
        result = {"should_trigger": should, "score": unc}
        self._trigger_history.append(result)
        return result

    def adapt_weights_from_evolution(self, summary: Dict[str, Any]) -> None:
        self._adapted = True


class _StubFeedbackBus:
    """Minimal feedback bus with register_signal support."""

    _CHANNEL_NAMES: List[str] = ["uncertainty", "coherence_deficit"]

    def __init__(self) -> None:
        self._extra_signals: Dict[str, float] = {}
        self._extra_defaults: Dict[str, float] = {}

    def register_signal(self, name: str, default: float = 0.0) -> None:
        self._extra_signals[name] = default
        self._extra_defaults[name] = default

    def write_signal(self, name: str, value: float) -> None:
        self._extra_signals[name] = value

    def get_state(self) -> Dict[str, float]:
        return dict(self._extra_signals)

    def get_oscillation_score(self) -> float:
        return 0.0


class _StubProvenanceTracker:
    """Minimal provenance tracker for trace testing."""

    def __init__(self) -> None:
        self._deps: Dict[str, List[str]] = {}

    def record_dependency(self, upstream: str, downstream: str) -> None:
        self._deps.setdefault(downstream, []).append(upstream)

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        return dict(self._deps)

    def trace_root_cause(self, output_id: str) -> Dict[str, Any]:
        return {"root_module": "encoder", "chain_length": 3}

    def get_trace_completeness_ratio(self) -> float:
        return 0.85


class _StubCausalTrace:
    """Minimal causal trace for trace testing."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = [
            {"step": 0, "module": "init"},
        ]

    def get_recent_entries(self, limit: int = 5) -> List[Dict]:
        return self._entries[-limit:]


class _StubModel(nn.Module):
    """Model with all cognitive subsystem stubs attached."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.error_evolution = _StubErrorEvolution()
        self.metacognitive_trigger = _StubMCT()
        self.feedback_bus = _StubFeedbackBus()
        self.provenance_tracker = _StubProvenanceTracker()
        self.causal_trace = _StubCausalTrace()
        self.streaming_signal_bus = MagicMock()
        self.streaming_signal_bus.get_ema = MagicMock(return_value={})
        self.vt_continuous_learner = MagicMock()
        self.adaptive_training_controller = MagicMock()
        self.ucc = MagicMock()
        self.continual_learning_core = MagicMock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _StubConfig:
    hidden_dim = 32
    z_dim = 32


# ════════════════════════════════════════════════════════════════════════
#  U1: auto_wire
# ════════════════════════════════════════════════════════════════════════

class TestU1AutoWire:
    """Verify UnifiedTrainingCycleController.auto_wire() discovers components."""

    def _make_controller(self) -> Any:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        return UnifiedTrainingCycleController(model, config), model

    def test_auto_wire_exists(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        assert hasattr(UnifiedTrainingCycleController, "auto_wire")

    def test_auto_wire_discovers_all_components(self) -> None:
        ctrl, model = self._make_controller()
        wired = ctrl.auto_wire(model)
        assert isinstance(wired, dict)
        # Should find at least the components we set on _StubModel
        found = sum(1 for v in wired.values() if v)
        assert found >= 4, f"Expected ≥4 wired components, got {found}"

    def test_auto_wire_attaches_feedback_bus(self) -> None:
        ctrl, model = self._make_controller()
        assert ctrl._feedback_bus is None
        ctrl.auto_wire(model)
        assert ctrl._feedback_bus is model.feedback_bus

    def test_auto_wire_attaches_mct(self) -> None:
        ctrl, model = self._make_controller()
        assert ctrl._mct is None
        ctrl.auto_wire(model)
        assert ctrl._mct is model.metacognitive_trigger

    def test_auto_wire_attaches_ucc(self) -> None:
        ctrl, model = self._make_controller()
        assert ctrl._ucc is None
        ctrl.auto_wire(model)
        assert ctrl._ucc is model.ucc

    def test_auto_wire_attaches_signal_bus(self) -> None:
        ctrl, model = self._make_controller()
        assert ctrl._signal_bus is None
        ctrl.auto_wire(model)
        assert ctrl._signal_bus is model.streaming_signal_bus

    def test_auto_wire_handles_missing_components(self) -> None:
        """If a component is None, auto_wire marks it as False."""
        ctrl, model = self._make_controller()
        model.streaming_signal_bus = None
        wired = ctrl.auto_wire(model)
        assert wired.get("VTStreamingSignalBus") is False

    def test_auto_wire_returns_dict(self) -> None:
        ctrl, model = self._make_controller()
        result = ctrl.auto_wire(model)
        assert isinstance(result, dict)
        assert len(result) >= 5

    def test_auto_wire_idempotent(self) -> None:
        ctrl, model = self._make_controller()
        w1 = ctrl.auto_wire(model)
        w2 = ctrl.auto_wire(model)
        assert w1 == w2


# ════════════════════════════════════════════════════════════════════════
#  U2: seed_from_wizard
# ════════════════════════════════════════════════════════════════════════

class TestU2SeedFromWizard:
    """Verify seed_from_wizard bridges wizard results to error evolution."""

    def test_seed_from_wizard_exists(self) -> None:
        from aeon_wizard import seed_from_wizard
        assert callable(seed_from_wizard)

    def test_seed_from_wizard_seeds_episodes(self) -> None:
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        wizard_results = {
            "corpus_diagnostics": {
                "heterogeneous": True,
                "corpus_size": 1000,
            },
            "codebook_init": {
                "initialized": True,
                "method": "kmeans",
                "inertia": 42.5,
            },
            "weight_loading": {
                "loaded": True,
                "format": "aeon_safetensors",
            },
            "overall_status": "completed",
            "total_duration_s": 5.0,
        }
        result = seed_from_wizard(model, wizard_results)
        assert result["seeded"] >= 3
        assert len(model.error_evolution._episodes) >= 3

    def test_seed_from_wizard_adapts_mct(self) -> None:
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        wizard_results = {
            "corpus_diagnostics": {"heterogeneous": True, "corpus_size": 100},
            "codebook_init": {"initialized": True},
            "weight_loading": {"loaded": True},
            "overall_status": "completed",
            "total_duration_s": 1.0,
        }
        seed_from_wizard(model, wizard_results)
        assert model.metacognitive_trigger._adapted is True

    def test_seed_from_wizard_no_error_evolution(self) -> None:
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        model.error_evolution = None
        result = seed_from_wizard(model, {"overall_status": "completed"})
        assert result["seeded"] == 0

    def test_seed_from_wizard_partial_results(self) -> None:
        """Only completed steps are seeded."""
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        wizard_results = {
            "corpus_diagnostics": {"heterogeneous": False},
            "weight_loading": {"loaded": False},
            "overall_status": "completed_with_warnings",
            "total_duration_s": 2.0,
        }
        result = seed_from_wizard(model, wizard_results)
        # Should seed at least weight_loading (even if failed) + completion
        assert result["seeded"] >= 2

    def test_seed_from_wizard_empty_results(self) -> None:
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        result = seed_from_wizard(model, {})
        # Should still seed wizard_completion with status=unknown
        assert result["seeded"] >= 1

    def test_seed_from_wizard_records_source_metadata(self) -> None:
        from aeon_wizard import seed_from_wizard
        model = _StubModel()
        wizard_results = {
            "weight_loading": {"loaded": True, "format": "raw"},
            "overall_status": "completed",
            "total_duration_s": 1.0,
        }
        seed_from_wizard(model, wizard_results)
        # Verify that source=wizard appears in metadata
        sources = [
            ep["metadata"].get("source")
            for ep in model.error_evolution._episodes
            if ep["metadata"].get("source") == "wizard"
        ]
        assert len(sources) >= 1


# ════════════════════════════════════════════════════════════════════════
#  U3: Dynamic feedback signal registration
# ════════════════════════════════════════════════════════════════════════

class TestU3DynamicSignalRegistration:
    """Verify Patch U3 pre-registers dynamic signals at activation time."""

    def test_u3_patch_text_present(self) -> None:
        with open("aeon_core.py", "r") as f:
            src = f.read()
        assert "Patch U3: Register dynamic extra signals" in src

    def test_u3_dynamic_signals_listed(self) -> None:
        with open("aeon_core.py", "r") as f:
            src = f.read()
        for sig in [
            "provenance_root_pressure",
            "causal_chain_coverage_deficit",
            "trace_incomplete_pressure",
            "decoder_quality_pressure",
            "decoder_variance_pressure",
            "decoder_provenance_pressure",
            "cross_pass_root_pressure",
        ]:
            assert sig in src, f"Dynamic signal {sig} not in aeon_core.py"

    def test_u3_uses_register_signal(self) -> None:
        """Patch U3 calls feedback_bus.register_signal for each dynamic signal."""
        with open("aeon_core.py", "r") as f:
            src = f.read()
        # Find the U3 block
        idx = src.index("Patch U3: Register dynamic extra signals")
        block = src[idx:idx + 2000]
        assert "register_signal" in block
        assert "_registered_dynamic" in block

    def test_u3_counter_logged(self) -> None:
        """Patch U3 logs the number of registered signals."""
        with open("aeon_core.py", "r") as f:
            src = f.read()
        idx = src.index("Patch U3: Register dynamic extra signals")
        block = src[idx:idx + 3000]
        assert "pre-registered" in block
        assert "dynamic feedback signals" in block


# ════════════════════════════════════════════════════════════════════════
#  U4: Bidirectional bridge in execute_full_cycle
# ════════════════════════════════════════════════════════════════════════

class TestU4BidirectionalBridge:
    """Verify execute_full_cycle invokes training↔inference bridges."""

    def _make_controller(self) -> Any:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        return ctrl, model

    def test_execute_full_cycle_accepts_bridge_params(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        sig = inspect.signature(
            UnifiedTrainingCycleController.execute_full_cycle,
        )
        params = list(sig.parameters.keys())
        assert "convergence_monitor" in params
        assert "error_evolution" in params

    def test_execute_full_cycle_basic_still_works(self) -> None:
        """Backward compatibility: cycle works without bridge params."""
        ctrl, model = self._make_controller()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A", epoch_metrics={"loss": 0.5},
        )
        assert result["epoch"] == 1
        assert result["phase"] == "A"
        assert "duration_s" in result

    def test_execute_full_cycle_with_bridges(self) -> None:
        """When bridge params are given, training_bridge key appears."""
        ctrl, model = self._make_controller()
        ee = _StubErrorEvolution()
        cm = MagicMock()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"loss": 0.5},
            convergence_monitor=cm,
            error_evolution=ee,
        )
        assert "training_bridge" in result

    def test_execute_full_cycle_increments_cycle(self) -> None:
        ctrl, model = self._make_controller()
        r1 = ctrl.execute_full_cycle(
            epoch=1, phase="A", epoch_metrics={},
        )
        r2 = ctrl.execute_full_cycle(
            epoch=2, phase="A", epoch_metrics={},
        )
        assert r2["cycle"] == r1["cycle"] + 1


# ════════════════════════════════════════════════════════════════════════
#  U4b: Meta-cognitive uncertainty check
# ════════════════════════════════════════════════════════════════════════

class TestU4bMetaCognitiveCheck:
    """Verify MCT check fires during unified cycle when MCT is attached."""

    def _make_controller_with_mct(self) -> Any:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        ctrl.attach_metacognitive_trigger(model.metacognitive_trigger)
        return ctrl, model

    def test_mct_check_appears_in_results(self) -> None:
        ctrl, model = self._make_controller_with_mct()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"uncertainty": 0.1},
        )
        assert "metacognitive_check" in result
        assert result["metacognitive_check"]["triggered"] is False

    def test_mct_triggers_on_high_uncertainty(self) -> None:
        ctrl, model = self._make_controller_with_mct()
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"uncertainty": 0.9},
        )
        assert result["metacognitive_check"]["triggered"] is True

    def test_mct_absent_no_check(self) -> None:
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        # Don't attach MCT
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"uncertainty": 0.9},
        )
        assert "metacognitive_check" not in result


# ════════════════════════════════════════════════════════════════════════
#  U5: Causal transparency — trace_output_to_premise
# ════════════════════════════════════════════════════════════════════════

class TestU5TraceOutputToPremise:
    """Verify trace_output_to_premise walks the full causal chain."""

    def test_trace_function_exists(self) -> None:
        from aeon_integration import trace_output_to_premise
        assert callable(trace_output_to_premise)

    def test_trace_returns_chain(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        assert "trace_chain" in result
        assert isinstance(result["trace_chain"], list)
        assert len(result["trace_chain"]) >= 2

    def test_trace_finds_root_cause(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        assert result["root_cause"] is not None
        assert result["trace_complete"] is True

    def test_trace_includes_provenance_layer(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "provenance" in layers

    def test_trace_includes_error_evolution_layer(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "error_evolution" in layers

    def test_trace_includes_causal_trace_layer(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "causal_trace" in layers

    def test_trace_includes_mct_layer(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "metacognitive_trigger" in layers

    def test_trace_with_output_id(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model, output_id="test_output_42")
        assert result["output_id"] == "test_output_42"

    def test_trace_without_provenance(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        model.provenance_tracker = None
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "provenance" not in layers

    def test_trace_without_error_evolution(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        model.error_evolution = None
        result = trace_output_to_premise(model)
        layers = [step["layer"] for step in result["trace_chain"]]
        assert "error_evolution" not in layers

    def test_trace_provenance_completeness(self) -> None:
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        result = trace_output_to_premise(model)
        assert "provenance_completeness" in result
        assert result["provenance_completeness"] == 0.85

    def test_trace_empty_model(self) -> None:
        """Model with no cognitive subsystems still returns valid structure."""
        from aeon_integration import trace_output_to_premise
        model = nn.Linear(4, 4)
        result = trace_output_to_premise(model)
        assert result["trace_chain"] == []
        assert result["root_cause"] is None
        assert result["trace_complete"] is False


# ════════════════════════════════════════════════════════════════════════
#  Cross-patch coherence
# ════════════════════════════════════════════════════════════════════════

class TestCrossPatchCoherence:
    """Verify U-series patches work together as a unified system."""

    def test_auto_wire_then_full_cycle_with_mct(self) -> None:
        """U1 + U4b: auto_wire discovers MCT, then full_cycle uses it."""
        from aeon_integration import UnifiedTrainingCycleController
        model = _StubModel()
        config = _StubConfig()
        ctrl = UnifiedTrainingCycleController(model, config)
        ctrl.auto_wire(model)
        assert ctrl._mct is not None
        result = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"uncertainty": 0.8},
        )
        assert result["metacognitive_check"]["triggered"] is True

    def test_wizard_seed_then_trace(self) -> None:
        """U2 + U5: wizard seeds error evolution, then trace finds it."""
        from aeon_wizard import seed_from_wizard
        from aeon_integration import trace_output_to_premise
        model = _StubModel()
        wizard_results = {
            "overall_status": "completed",
            "total_duration_s": 1.0,
            "weight_loading": {"loaded": True, "format": "raw"},
        }
        seed_from_wizard(model, wizard_results)
        trace = trace_output_to_premise(model)
        # Error evolution layer should show the seeded episodes
        ee_layer = [
            s for s in trace["trace_chain"]
            if s["layer"] == "error_evolution"
        ]
        assert len(ee_layer) == 1
        assert ee_layer[0]["total_episodes"] >= 2

    def test_full_lifecycle(self) -> None:
        """U1 + U2 + U4 + U5: Complete lifecycle from wizard to trace."""
        from aeon_wizard import seed_from_wizard
        from aeon_integration import (
            UnifiedTrainingCycleController,
            trace_output_to_premise,
        )
        model = _StubModel()
        config = _StubConfig()

        # Step 1: Wizard seeds
        wizard_results = {
            "overall_status": "completed",
            "total_duration_s": 1.0,
            "corpus_diagnostics": {"heterogeneous": True, "corpus_size": 500},
            "weight_loading": {"loaded": True},
            "codebook_init": {"initialized": True, "method": "kmeans"},
        }
        seed_result = seed_from_wizard(model, wizard_results)
        assert seed_result["seeded"] >= 3

        # Step 2: Controller auto-wires
        ctrl = UnifiedTrainingCycleController(model, config)
        wired = ctrl.auto_wire(model)
        assert sum(v for v in wired.values()) >= 4

        # Step 3: Execute a unified cycle
        cycle = ctrl.execute_full_cycle(
            epoch=1, phase="A",
            epoch_metrics={"uncertainty": 0.3, "loss": 0.5},
        )
        assert cycle["cycle"] == 1

        # Step 4: Trace output to premise
        trace = trace_output_to_premise(model)
        assert trace["trace_complete"] is True
        assert trace["root_cause"] is not None

    def test_u_patches_in_source(self) -> None:
        """Verify all U-series patch markers exist in source files."""
        with open("aeon_core.py", "r") as f:
            core_src = f.read()
        with open("aeon_integration.py", "r") as f:
            integ_src = f.read()
        with open("aeon_wizard.py", "r") as f:
            wizard_src = f.read()

        assert "Patch U3" in core_src
        assert "Patch U1" in integ_src or "auto_wire" in integ_src
        assert "Patch U4" in integ_src
        assert "Patch U5" in integ_src or "trace_output_to_premise" in integ_src
        assert "Patch U2" in wizard_src or "seed_from_wizard" in wizard_src
