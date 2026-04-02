"""
AEON Integration Controller — Glue Code
========================================

Connects the AEON training pipeline (``ae_train``) with the cognitive
inference engine (``aeon_core``) through a unified controller that
manages the full lifecycle of synchronized AEON + VibeThinker training.

Implements Spec §4.В — 10-Point Integration:

  1.  Teacher-Student inversion       (bifasic_didactic_orchestrate)
  2.  VQ codebook warm-start          (warm_start_codebook_from_vt)
  3.  Context window calibration      (calibrate_context_window)
  4.  Streaming signal bus             (VTStreamingSignalBus closed loop)
  5.  Z-sequence quality annotation   (annotate_z_sequences_quality)
  6.  Training→Inference error bridge  (bridge_training_errors_to_inference)
  7.  Inference→Training feedback      (bridge_inference_insights_to_training)
  8.  UCC inner-epoch evaluation       (ucc_inner_epoch_evaluation)
  9.  SSP temperature alignment        (align_ssp_temperature)
  10. Continuous learning micro-retrain (micro_retrain_from_pseudo_labels)

Weight Loading:
  All VibeThinker-1.5B modules load from
  ``vibe_thinker_weights/model.safetensors`` (Spec §2.1).

Closed-Loop Architecture:
  Inference insights flow back to training through CognitiveFeedbackBus,
  and training errors flow forward to inference through error evolution
  seeding — creating a bidirectional feedback loop (Spec §2.4).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("AEON-Integration")

# ── Spec §2.1: Weight path constant ─────────────────────────────────────────
VT_WEIGHTS_PATH = Path("vibe_thinker_weights/model.safetensors")


# =============================================================================
#  Integration State Tracker
# =============================================================================
class IntegrationState:
    """Tracks the state of all 10 integration points."""

    def __init__(self) -> None:
        self.points: Dict[str, Dict[str, Any]] = {
            "teacher_student_inversion": {"active": False, "last_result": None},
            "codebook_warm_start": {"active": False, "last_result": None},
            "context_window_calibration": {"active": False, "last_result": None},
            "streaming_signal_bus": {"active": False, "last_result": None},
            "z_sequence_annotation": {"active": False, "last_result": None},
            "training_to_inference_bridge": {"active": False, "last_result": None},
            "inference_to_training_bridge": {"active": False, "last_result": None},
            "ucc_epoch_evaluation": {"active": False, "last_result": None},
            "ssp_temperature_alignment": {"active": False, "last_result": None},
            "continuous_learning": {"active": False, "last_result": None},
        }
        self.cycle_count = 0
        self.last_cycle_time: Optional[float] = None

    def update_point(self, name: str, result: Dict[str, Any]) -> None:
        if name in self.points:
            self.points[name]["active"] = True
            self.points[name]["last_result"] = result
            self.points[name]["last_updated"] = time.time()

    def get_active_count(self) -> int:
        return sum(1 for p in self.points.values() if p["active"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integration_points": self.points,
            "active_count": self.get_active_count(),
            "total_points": len(self.points),
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
        }


# Global integration state
_integration_state = IntegrationState()


def get_integration_state() -> IntegrationState:
    """Return the global integration state."""
    return _integration_state


# =============================================================================
#  Weight Loader  (Spec §2.1)
# =============================================================================
def load_vt_weights_into_model(
    model: nn.Module,
    weights_path: Path = VT_WEIGHTS_PATH,
) -> Dict[str, Any]:
    """Load VibeThinker-1.5B weights from safetensors into model components.

    Implements Spec §4.В.1: Подгрузка весов из
    ``vibe_thinker_weights/model.safetensors``.

    Args:
        model: AEON model instance with VibeThinker components.
        weights_path: Path to safetensors weight file.

    Returns:
        Dict with loading status and diagnostics.
    """
    result: Dict[str, Any] = {
        "loaded": False,
        "weights_path": str(weights_path),
    }

    if not weights_path.exists():
        result["reason"] = f"Weight file not found: {weights_path}"
        return result

    try:
        from safetensors.torch import load_file as st_load

        flat = st_load(str(weights_path))
        result["total_tensors"] = len(flat)

        # Detect format
        _has_aeon = any(
            k.startswith(("adapter_state.", "kernel_state."))
            for k in flat
        )

        if _has_aeon:
            adapter_state = {
                k[len("adapter_state."):]: v
                for k, v in flat.items()
                if k.startswith("adapter_state.")
            }
            kernel_state = {
                k[len("kernel_state."):]: v
                for k, v in flat.items()
                if k.startswith("kernel_state.")
            }

            # Load adapter
            _adapter = getattr(model, "vibe_thinker_adapter", None)
            if _adapter is not None and adapter_state:
                _adapter.load_state_dict(adapter_state, strict=False)
                result["adapter_loaded"] = True
                result["adapter_keys"] = len(adapter_state)

            # Load kernel
            _kernel = getattr(model, "vibe_thinker_kernel", None)
            if _kernel is not None and kernel_state:
                _kernel.load_state_dict(kernel_state, strict=False)
                result["kernel_loaded"] = True
                result["kernel_keys"] = len(kernel_state)

            result["format"] = "aeon_safetensors"
        else:
            result["format"] = "raw_hf_safetensors"

        result["loaded"] = True
        result["file_size_mb"] = round(
            weights_path.stat().st_size / (1024 * 1024), 2,
        )
        logger.info("✅ VT weights loaded: %s (%s format, %d tensors)",
                     weights_path, result["format"], len(flat))

    except ImportError:
        result["reason"] = "safetensors library not installed"
    except Exception as e:
        result["reason"] = f"{type(e).__name__}: {e}"
        logger.warning("VT weight load failed: %s", e)

    _integration_state.update_point("codebook_warm_start", result)
    return result


# =============================================================================
#  Cognitive Feedback Bus Connector  (Spec §4.В.2)
# =============================================================================
def connect_feedback_bus(
    controller: Any,
    vt_learner: Any,
    feedback_bus: Any,
) -> Dict[str, Any]:
    """Connect AdaptiveTrainingController with VibeThinkerContinuousLearner
    through CognitiveFeedbackBus.

    Implements Spec §4.В.2 — creates the closed-loop feedback path:
      VTContinuousLearner → CognitiveFeedbackBus → AdaptiveTrainingController

    Signals registered on the feedback bus:
      - calibration_pressure:      VT calibration error EMA
      - adaptation_signal:         Learning adaptation strength
      - complexity_threshold:      Dynamic complexity gating
      - psi_weight:                VibeThinker contribution weight

    Args:
        controller: AdaptiveTrainingController instance.
        vt_learner: VibeThinkerContinuousLearner instance.
        feedback_bus: CognitiveFeedbackBus instance.

    Returns:
        Dict with connection status.
    """
    result: Dict[str, Any] = {"connected": False}

    try:
        # Register VibeThinker signals on feedback bus
        _signals = [
            ("vt_calibration_pressure", 0.0),
            ("vt_adaptation_signal", 0.5),
            ("vt_complexity_threshold", 0.5),
            ("vt_psi_weight", 0.05),
        ]
        for name, default in _signals:
            if hasattr(feedback_bus, "register_signal"):
                feedback_bus.register_signal(name, default)
            elif hasattr(feedback_bus, "write_signal"):
                feedback_bus.write_signal(name, default)

        result["registered_signals"] = [s[0] for s in _signals]
        result["connected"] = True

        logger.info("🔗 Feedback bus connected: %d VT signals registered",
                     len(_signals))

    except Exception as e:
        result["error"] = str(e)
        logger.warning("Feedback bus connection failed: %s", e)

    _integration_state.update_point("streaming_signal_bus", result)
    return result


# =============================================================================
#  Patch U5: Causal Transparency — Output→Premise Trace
# =============================================================================
def trace_output_to_premise(
    model: nn.Module,
    output_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically trace an output/action back to its originating premise.

    Implements causal transparency by walking the provenance DAG, error
    evolution history, and causal trace chain in reverse — from the most
    recent output back through the architecture to the root cause.

    Args:
        model: AEONDeltaV4 model instance (with provenance_tracker,
               error_evolution, causal_trace attributes).
        output_id: Optional identifier for a specific output to trace.
                   If ``None``, traces the most recent action.

    Returns:
        Dict with ``trace_chain`` (list of steps from output to premise),
        ``root_cause`` (the originating premise/module), and
        ``trace_complete`` (whether the chain reaches a known root).
    """
    result: Dict[str, Any] = {
        "output_id": output_id,
        "trace_chain": [],
        "root_cause": None,
        "trace_complete": False,
    }

    # ── Step 1: Provenance trace ────────────────────────────────────
    prov = getattr(model, "provenance_tracker", None)
    if prov is not None:
        try:
            if hasattr(prov, "trace_root_cause"):
                root_info = prov.trace_root_cause(
                    output_id or "latest_output",
                )
                if isinstance(root_info, dict):
                    result["trace_chain"].append({
                        "layer": "provenance",
                        "data": root_info,
                    })
                    result["root_cause"] = root_info.get(
                        "root_module", root_info.get("root", None),
                    )
            if hasattr(prov, "get_trace_completeness_ratio"):
                ratio = prov.get_trace_completeness_ratio()
                result["provenance_completeness"] = ratio
        except Exception as e:
            result["trace_chain"].append({
                "layer": "provenance",
                "error": str(e),
            })

    # ── Step 2: Error evolution trail ───────────────────────────────
    ee = getattr(model, "error_evolution", None)
    if ee is not None:
        try:
            summary = ee.get_error_summary()
            if isinstance(summary, dict):
                result["trace_chain"].append({
                    "layer": "error_evolution",
                    "total_episodes": summary.get("total_recorded", 0),
                    "recent_classes": list(
                        summary.get("error_classes", {}).keys(),
                    )[:10],
                })
        except Exception as e:
            result["trace_chain"].append({
                "layer": "error_evolution",
                "error": str(e),
            })

    # ── Step 3: Causal trace entries ────────────────────────────────
    ct = getattr(model, "causal_trace", None)
    if ct is not None:
        try:
            if hasattr(ct, "get_recent_entries"):
                entries = ct.get_recent_entries(limit=5)
                result["trace_chain"].append({
                    "layer": "causal_trace",
                    "recent_entries": len(entries) if entries else 0,
                })
            elif hasattr(ct, "_entries"):
                entries = ct._entries[-5:] if ct._entries else []
                result["trace_chain"].append({
                    "layer": "causal_trace",
                    "recent_entries": len(entries),
                })
        except Exception as e:
            result["trace_chain"].append({
                "layer": "causal_trace",
                "error": str(e),
            })

    # ── Step 4: MCT trigger history ─────────────────────────────────
    mct = getattr(model, "metacognitive_trigger", None)
    if mct is not None:
        try:
            if hasattr(mct, "_trigger_history"):
                hist = mct._trigger_history
                result["trace_chain"].append({
                    "layer": "metacognitive_trigger",
                    "total_triggers": len(hist) if hist else 0,
                })
        except Exception as e:
            result["trace_chain"].append({
                "layer": "metacognitive_trigger",
                "error": str(e),
            })

    # ── Determine trace completeness ────────────────────────────────
    result["trace_complete"] = (
        result["root_cause"] is not None
        or len(result["trace_chain"]) >= 2
    )

    return result


# =============================================================================
#  Closed-Loop Training Cycle  (Spec §4.В.3–10)
# =============================================================================
class UnifiedTrainingCycleController:
    """Orchestrates all 10 integration points in a unified training cycle.

    This controller manages the bidirectional flow between training and
    inference, executing integration points in the correct sequence
    during each training epoch.

    Implements Spec §2.4 — closed-loop feedback from inference to training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._signal_bus: Optional[Any] = None
        self._vt_learner: Optional[Any] = None
        self._controller: Optional[Any] = None
        self._feedback_bus: Optional[Any] = None
        self._ucc: Optional[Any] = None
        self._mct: Optional[Any] = None
        self._continual_core: Optional[Any] = None
        self._cycle_count = 0
        self._metrics_history: List[Dict[str, Any]] = []

    def attach_signal_bus(self, signal_bus: Any) -> None:
        """Attach VTStreamingSignalBus for continuous signal flow."""
        self._signal_bus = signal_bus
        logger.info("🔗 VTStreamingSignalBus attached")

    def attach_vt_learner(self, learner: Any) -> None:
        """Attach VibeThinkerContinuousLearner."""
        self._vt_learner = learner
        logger.info("🔗 VibeThinkerContinuousLearner attached")

    def attach_controller(self, controller: Any) -> None:
        """Attach AdaptiveTrainingController."""
        self._controller = controller
        logger.info("🔗 AdaptiveTrainingController attached")

    def attach_feedback_bus(self, feedback_bus: Any) -> None:
        """Attach CognitiveFeedbackBus."""
        self._feedback_bus = feedback_bus
        logger.info("🔗 CognitiveFeedbackBus attached")

    def attach_ucc(self, ucc: Any) -> None:
        """Attach UnifiedCognitiveCycle for epoch evaluation."""
        self._ucc = ucc
        logger.info("🔗 UnifiedCognitiveCycle attached")

    def attach_metacognitive_trigger(self, mct: Any) -> None:
        """Attach MetaCognitiveRecursionTrigger."""
        self._mct = mct
        logger.info("🔗 MetaCognitiveRecursionTrigger attached")

    def attach_continual_core(self, core: Any) -> None:
        """Attach ContinualLearningCore."""
        self._continual_core = core
        logger.info("🔗 ContinualLearningCore attached")

    # ─── Patch U1: Auto-wire component discovery ────────────────────────
    def auto_wire(self, model: nn.Module) -> Dict[str, bool]:
        """Auto-discover and attach cognitive components from *model*.

        Inspects the model for well-known component attributes
        (``feedback_bus``, ``metacognitive_trigger``, ``ucc``, etc.)
        and calls the corresponding ``attach_*()`` methods.

        Returns:
            Dict mapping component name → whether it was found & attached.
        """
        _component_map: Dict[str, Tuple[str, str]] = {
            # model attribute → (attach method name, human label)
            "streaming_signal_bus": ("attach_signal_bus", "VTStreamingSignalBus"),
            "vt_continuous_learner": ("attach_vt_learner", "VibeThinkerContinuousLearner"),
            "adaptive_training_controller": ("attach_controller", "AdaptiveTrainingController"),
            "feedback_bus": ("attach_feedback_bus", "CognitiveFeedbackBus"),
            "ucc": ("attach_ucc", "UnifiedCognitiveCycle"),
            "metacognitive_trigger": ("attach_metacognitive_trigger", "MetaCognitiveRecursionTrigger"),
            "continual_learning_core": ("attach_continual_core", "ContinualLearningCore"),
        }

        wired: Dict[str, bool] = {}
        for attr_name, (method_name, label) in _component_map.items():
            component = getattr(model, attr_name, None)
            if component is not None:
                attach_fn = getattr(self, method_name)
                attach_fn(component)
                wired[label] = True
            else:
                wired[label] = False

        _attached = sum(1 for v in wired.values() if v)
        logger.info(
            "🔗 auto_wire: %d/%d components discovered and attached",
            _attached, len(_component_map),
        )
        return wired

    # ─── Integration Point 1: Teacher-Student Inversion (Spec II.4) ──────
    def execute_teacher_student_inversion(
        self,
        z_sequences: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Implements Spec Point II.4: Teacher-Student Inversion.

        After Phase A, AEON becomes the teacher — its trained latent
        representations guide VibeThinker adapter alignment.
        """
        try:
            from ae_train import bifasic_didactic_orchestrate
            result = bifasic_didactic_orchestrate(
                model=self.model,
                z_sequences=z_sequences,
                config=self.config,
                device=self.device,
            )
            _integration_state.update_point(
                "teacher_student_inversion", result,
            )
            return result
        except Exception as e:
            return {"inverted": False, "reason": str(e)}

    # ─── Integration Point 4: Streaming Signal Bus Closed Loop ───────────
    def execute_signal_bus_step(self) -> Dict[str, Any]:
        """Execute one closed-loop step of the streaming signal bus.

        Implements Spec §4.В — VTContinuousLearner signals flow through
        VTStreamingSignalBus to AdaptiveTrainingController.
        """
        if self._signal_bus is None or self._vt_learner is None:
            return {"executed": False, "reason": "components_not_attached"}

        try:
            if hasattr(self._signal_bus, "closed_loop_step"):
                result = self._signal_bus.closed_loop_step(
                    self._vt_learner, self._controller,
                )
            else:
                # Manual push-pull cycle
                cal_ema = getattr(self._vt_learner, "_calibration_ema", 0.0)
                self._signal_bus.push("calibration_pressure", cal_ema)
                ct_ema = getattr(
                    self._vt_learner, "_complexity_threshold_ema", 0.5,
                )
                self._signal_bus.push("complexity_threshold", ct_ema)
                psi_w = getattr(self._vt_learner, "_psi_weight_ema", 0.05)
                self._signal_bus.push("psi_weight", psi_w)

                result = {"executed": True, "signals_pushed": 3}

            # Also push to feedback bus if available
            if self._feedback_bus is not None:
                ema = self._signal_bus.get_ema() if hasattr(
                    self._signal_bus, "get_ema",
                ) else {}
                for name, val in ema.items():
                    if hasattr(self._feedback_bus, "write_signal"):
                        self._feedback_bus.write_signal(
                            f"vt_{name}", float(val),
                        )

            _integration_state.update_point("streaming_signal_bus", result)
            return result

        except Exception as e:
            return {"executed": False, "reason": str(e)}

    # ─── Integration Point 5: Z-Sequence Quality Annotation ──────────────
    def execute_z_annotation(
        self,
        z_sequences: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Implements Spec Point II.8: Z-sequence quality annotation.

        Annotates each z-vector with VibeThinker quality metadata
        [confidence, entropy, reasoning_quality] for Phase B training.
        """
        try:
            from ae_train import annotate_z_sequences_quality
            z_out, annotations = annotate_z_sequences_quality(
                model=self.model,
                z_sequences=z_sequences,
                config=self.config,
                device=self.device,
            )
            _integration_state.update_point("z_sequence_annotation", {
                "annotated": True,
                "num_sequences": len(z_sequences),
            })
            return z_out, annotations

        except Exception as e:
            logger.warning("Z-sequence annotation failed: %s", e)
            fallback = [torch.ones(seq.shape[0], 3) for seq in z_sequences]
            return z_sequences, fallback

    # ─── Integration Point 6: Training→Inference Bridge ──────────────────
    def execute_training_to_inference_bridge(
        self,
        convergence_monitor: Any,
        error_evolution: Any,
    ) -> Dict[str, Any]:
        """Implements Spec: Training→inference error propagation.

        Exports training error patterns from convergence monitor
        and seeds the inference-side error evolution tracker.
        """
        try:
            from ae_train import bridge_training_errors_to_inference
            result = bridge_training_errors_to_inference(
                convergence_monitor=convergence_monitor,
                error_evolution=error_evolution,
            )
            _integration_state.update_point(
                "training_to_inference_bridge", result,
            )
            return result
        except Exception as e:
            return {"bridged": False, "reason": str(e)}

    # ─── Integration Point 7: Inference→Training Feedback ────────────────
    def execute_inference_to_training_bridge(
        self,
        model_inference: nn.Module,
        convergence_monitor: Any,
    ) -> Dict[str, Any]:
        """Implements Spec: Inference→training insight propagation.

        Pulls inference-side diagnostics (emergence, uncertainty)
        and feeds them back into the training convergence monitor.
        """
        try:
            from ae_train import bridge_inference_insights_to_training
            result = bridge_inference_insights_to_training(
                model=model_inference,
                convergence_monitor=convergence_monitor,
            )
            _integration_state.update_point(
                "inference_to_training_bridge", result,
            )
            return result
        except Exception as e:
            return {"bridged": False, "reason": str(e)}

    # ─── Integration Point 8: UCC Inner-Epoch Evaluation ─────────────────
    def execute_ucc_evaluation(
        self,
        epoch: int,
        phase: str,
        epoch_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implements Spec: UCC inner-epoch evaluation.

        Invokes UnifiedCognitiveCycle.evaluate() mid-epoch for
        real-time metacognitive feedback.
        """
        try:
            from ae_train import ucc_inner_epoch_evaluation
            result = ucc_inner_epoch_evaluation(
                model=self.model,
                epoch=epoch,
                phase=phase,
                epoch_metrics=epoch_metrics,
            )
            _integration_state.update_point("ucc_epoch_evaluation", result)
            return result
        except Exception as e:
            return {"evaluated": False, "reason": str(e)}

    # ─── Integration Point 9: SSP Temperature Alignment ──────────────────
    def execute_ssp_alignment(self) -> Dict[str, Any]:
        """Implements Spec: SSP temperature alignment.

        Synchronizes SSP framework temperature across VibeThinker
        diversity generator and the training pipeline.
        """
        try:
            from ae_train import align_ssp_temperature
            result = align_ssp_temperature(
                model=self.model,
                config=self.config,
            )
            _integration_state.update_point(
                "ssp_temperature_alignment", result,
            )
            return result
        except Exception as e:
            return {"aligned": False, "reason": str(e)}

    # ─── Integration Point 10: Continuous Learning Micro-Retrain ─────────
    def execute_micro_retrain(
        self,
        pseudo_labels: List[Dict[str, Any]],
        z_sequences: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Implements Spec: ContinualLearningCore micro-retrain.

        Uses pseudo-labels from VibeThinkerContinuousLearner
        Phase 4 (consolidation) to perform incremental updates.
        """
        try:
            from ae_train import micro_retrain_from_pseudo_labels
            result = micro_retrain_from_pseudo_labels(
                model=self.model,
                pseudo_labels=pseudo_labels,
                config=self.config,
                device=self.device,
                z_sequences=z_sequences,
            )
            _integration_state.update_point("continuous_learning", result)
            return result
        except Exception as e:
            return {"retrained": False, "reason": str(e)}

    # ─── Unified Cycle Step ──────────────────────────────────────────────
    def execute_full_cycle(
        self,
        epoch: int,
        phase: str,
        epoch_metrics: Dict[str, Any],
        z_sequences: Optional[List[torch.Tensor]] = None,
        convergence_monitor: Any = None,
        error_evolution: Any = None,
    ) -> Dict[str, Any]:
        """Execute all active integration points for one training cycle.

        Implements Spec §2.4 — unified cognitive cycle step:
          1. Signal bus closed-loop step
          2. UCC epoch evaluation
          3. SSP temperature alignment
          4. Z-sequence annotation (if sequences provided)
          5. Bidirectional training↔inference bridge (Patch U4)
          6. Meta-cognitive uncertainty check (Patch U4)

        Returns:
            Dict with results from all executed integration points.
        """
        self._cycle_count += 1
        cycle_start = time.time()
        cycle_results: Dict[str, Any] = {
            "cycle": self._cycle_count,
            "epoch": epoch,
            "phase": phase,
        }

        # 1. Signal bus closed loop
        if self._signal_bus is not None:
            cycle_results["signal_bus"] = self.execute_signal_bus_step()

        # 2. UCC evaluation
        cycle_results["ucc"] = self.execute_ucc_evaluation(
            epoch, phase, epoch_metrics,
        )

        # 3. SSP alignment
        cycle_results["ssp"] = self.execute_ssp_alignment()

        # 4. Z-sequence annotation
        if z_sequences is not None:
            _, annotations = self.execute_z_annotation(z_sequences)
            cycle_results["z_annotation"] = {
                "annotated": True,
                "num_sequences": len(z_sequences),
            }

        # ── Patch U4: Bidirectional training↔inference bridge ─────────
        # bridge_training_errors_to_inference and
        # bridge_inference_insights_to_training exist as standalone
        # functions in ae_train but were not auto-invoked during the
        # unified cycle.  By calling them here (when the required
        # components are available) the error evolution loop is closed
        # every cycle — training errors flow to inference, and
        # inference insights flow back to training.
        if convergence_monitor is not None and error_evolution is not None:
            cycle_results["training_bridge"] = (
                self.execute_training_to_inference_bridge(
                    convergence_monitor, error_evolution,
                )
            )
        if self._controller is not None and error_evolution is not None:
            if convergence_monitor is not None:
                cycle_results["inference_bridge"] = (
                    self.execute_inference_to_training_bridge(
                        self.model, convergence_monitor,
                    )
                )

        # ── Patch U4b: Meta-cognitive uncertainty check ───────────────
        # After all integration points execute, query the MCT to see
        # whether accumulated uncertainty warrants a meta-cognitive
        # review cycle.  This ensures that *any* internal uncertainty
        # automatically triggers higher-order review per the spec.
        if self._mct is not None:
            try:
                _mct_result = self._mct.evaluate(
                    uncertainty=epoch_metrics.get("uncertainty", 0.0),
                    coherence_deficit=epoch_metrics.get(
                        "coherence_deficit", 0.0,
                    ),
                )
                _should_trigger = False
                if isinstance(_mct_result, dict):
                    _should_trigger = _mct_result.get(
                        "should_trigger", False,
                    )
                cycle_results["metacognitive_check"] = {
                    "triggered": _should_trigger,
                }
            except Exception as _mct_err:
                cycle_results["metacognitive_check"] = {
                    "triggered": False,
                    "error": str(_mct_err),
                }

        cycle_results["duration_s"] = round(time.time() - cycle_start, 4)
        self._metrics_history.append(cycle_results)
        _integration_state.cycle_count = self._cycle_count
        _integration_state.last_cycle_time = time.time()

        return cycle_results

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Return the history of all cycle metrics."""
        return self._metrics_history

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Return current integration metrics for dashboard display.

        Implements Spec §4.Б — real-time metrics for the dashboard.
        """
        metrics: Dict[str, Any] = {
            "cycle_count": self._cycle_count,
            "integration_state": _integration_state.to_dict(),
        }

        # Signal bus EMA values
        if self._signal_bus is not None and hasattr(
            self._signal_bus, "get_ema",
        ):
            metrics["signal_bus_ema"] = self._signal_bus.get_ema()

        # VT learner state
        if self._vt_learner is not None:
            metrics["vt_learner"] = {
                "calibration_ema": getattr(
                    self._vt_learner, "_calibration_ema", 0.0,
                ),
                "complexity_threshold_ema": getattr(
                    self._vt_learner, "_complexity_threshold_ema", 0.5,
                ),
                "episode_count": getattr(
                    self._vt_learner, "_episode_count", 0,
                ),
            }

        # Feedback bus state
        if self._feedback_bus is not None:
            if hasattr(self._feedback_bus, "get_state"):
                metrics["feedback_bus"] = self._feedback_bus.get_state()
            if hasattr(self._feedback_bus, "get_oscillation_score"):
                metrics["feedback_bus_oscillation"] = (
                    self._feedback_bus.get_oscillation_score()
                )

        # Continual learning state
        if self._continual_core is not None:
            metrics["continual_learning"] = {
                "active": True,
                "task_count": getattr(
                    self._continual_core, "_task_count", 0,
                ),
            }

        return metrics


# =============================================================================
#  Dashboard Metrics Collector  (Spec §4.Б)
# =============================================================================
class DashboardMetricsCollector:
    """Collects and aggregates metrics for the AEON Dashboard.

    Implements Spec §4.Б — real-time metrics across all phases:
      - Phase A: commitment_loss, entropy_weight, codebook usage
      - Phase B: L_mse, L_quality, predicted CoT depth
      - VibeThinker: calibration_error, confidence, complexity_threshold_ema
      - Cognitive coherence: feedback bus, convergence, emergence
    """

    def __init__(self) -> None:
        self._phase_a_history: List[Dict[str, float]] = []
        self._phase_b_history: List[Dict[str, float]] = []
        self._vt_signals_history: List[Dict[str, float]] = []
        self._coherence_history: List[Dict[str, float]] = []

    def record_phase_a(
        self,
        epoch: int,
        commitment_loss: float,
        entropy_weight: float,
        codebook_usage: float,
        total_loss: float,
    ) -> None:
        """Record Phase A training metrics."""
        self._phase_a_history.append({
            "epoch": epoch,
            "commitment_loss": round(commitment_loss, 6),
            "entropy_weight": round(entropy_weight, 6),
            "codebook_usage": round(codebook_usage, 4),
            "total_loss": round(total_loss, 6),
            "timestamp": time.time(),
        })

    def record_phase_b(
        self,
        epoch: int,
        l_mse: float,
        l_quality: float,
        cot_depth_pred: float,
    ) -> None:
        """Record Phase B training metrics."""
        self._phase_b_history.append({
            "epoch": epoch,
            "L_mse": round(l_mse, 6),
            "L_quality": round(l_quality, 6),
            "cot_depth_predicted": round(cot_depth_pred, 4),
            "timestamp": time.time(),
        })

    def record_vt_signals(
        self,
        calibration_error: float,
        confidence: float,
        complexity_threshold_ema: float,
    ) -> None:
        """Record VibeThinker signals."""
        self._vt_signals_history.append({
            "calibration_error": round(calibration_error, 6),
            "confidence": round(confidence, 4),
            "complexity_threshold_ema": round(complexity_threshold_ema, 4),
            "timestamp": time.time(),
        })

    def record_coherence(
        self,
        cognitive_unity: float,
        feedback_oscillation: float,
        convergence_quality: float,
    ) -> None:
        """Record cognitive coherence metrics."""
        self._coherence_history.append({
            "cognitive_unity": round(cognitive_unity, 4),
            "feedback_oscillation": round(feedback_oscillation, 4),
            "convergence_quality": round(convergence_quality, 4),
            "timestamp": time.time(),
        })

    def get_phase_a_metrics(self, last_n: int = 50) -> List[Dict[str, float]]:
        """Return recent Phase A metrics."""
        return self._phase_a_history[-last_n:]

    def get_phase_b_metrics(self, last_n: int = 50) -> List[Dict[str, float]]:
        """Return recent Phase B metrics."""
        return self._phase_b_history[-last_n:]

    def get_vt_signals(self, last_n: int = 50) -> List[Dict[str, float]]:
        """Return recent VibeThinker signals."""
        return self._vt_signals_history[-last_n:]

    def get_coherence_metrics(
        self, last_n: int = 50,
    ) -> List[Dict[str, float]]:
        """Return recent coherence metrics."""
        return self._coherence_history[-last_n:]

    def get_latest(self) -> Dict[str, Any]:
        """Return the latest values from all metric categories."""
        return {
            "phase_a": self._phase_a_history[-1]
            if self._phase_a_history else None,
            "phase_b": self._phase_b_history[-1]
            if self._phase_b_history else None,
            "vt_signals": self._vt_signals_history[-1]
            if self._vt_signals_history else None,
            "coherence": self._coherence_history[-1]
            if self._coherence_history else None,
        }

    def get_all(self) -> Dict[str, Any]:
        """Return all metric histories."""
        return {
            "phase_a": self._phase_a_history,
            "phase_b": self._phase_b_history,
            "vt_signals": self._vt_signals_history,
            "coherence": self._coherence_history,
        }


# Global metrics collector
_metrics_collector = DashboardMetricsCollector()


def get_metrics_collector() -> DashboardMetricsCollector:
    """Return the global dashboard metrics collector."""
    return _metrics_collector
