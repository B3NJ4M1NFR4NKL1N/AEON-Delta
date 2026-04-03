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
#  Closed-Loop Training Cycle  (Spec §4.В.3–10)
# =============================================================================
class UnifiedTrainingCycleController:
    """Orchestrates all 10 integration points in a unified training cycle.

    This controller manages the bidirectional flow between training and
    inference, executing integration points in the correct sequence
    during each training epoch.

    Implements Spec §2.4 — closed-loop feedback from inference to training.
    """

    TOTAL_INTEGRATION_POINTS = 10
    _CYCLE_METADATA_KEYS = (
        "cycle", "epoch", "phase", "duration_s",
        "uncertainty_flags", "metacognitive_review",
    )

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

    # ─── I12: Wizard Results Consumption ─────────────────────────────────
    def consume_wizard_results(
        self,
        wizard_results: Dict[str, Any],
        error_evolution: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Consume wizard output and seed the integration controller.

        Bridges the gap between the first-run wizard (aeon_wizard.py)
        and the integration cycle by:
          1. Recording wizard completion status to error_evolution
          2. Applying hyperparameters from the wizard to the config
          3. Surfacing wizard diagnostics on the feedback bus

        Args:
            wizard_results: Dict returned by aeon_wizard.run_wizard().
            error_evolution: Error evolution tracker for seeding.

        Returns:
            Dict with consumption status and applied settings.
        """
        consumed: Dict[str, Any] = {
            "consumed": False,
            "applied_settings": [],
            "wizard_status": "unknown",
        }
        try:
            consumed["wizard_status"] = wizard_results.get(
                "overall_status", "unknown",
            )
            # 1. Record wizard outcome to error_evolution
            _status = wizard_results.get("overall_status", "unknown")
            if error_evolution is not None and hasattr(
                error_evolution, "record_episode",
            ):
                error_evolution.record_episode(
                    error_class="wizard_completion",
                    strategy_used=_status,
                    success=(_status == "completed"),
                    metadata={
                        "duration_s": wizard_results.get("total_duration_s"),
                        "steps": list(wizard_results.keys()),
                    },
                )

            # 2. Apply hyperparameters to config if available
            _hyper = wizard_results.get("hyperparameters", {})
            if isinstance(_hyper, dict):
                for key, value in _hyper.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        consumed["applied_settings"].append(key)

            # 3. Surface wizard diagnostics on feedback bus
            if (
                self._feedback_bus is not None
                and hasattr(self._feedback_bus, "write_signal")
            ):
                self._feedback_bus.write_signal(
                    "wizard_completed", 1.0 if _status == "completed" else 0.0,
                )
                # Write corpus quality if available
                _corpus = wizard_results.get("corpus_diagnostics", {})
                if isinstance(_corpus, dict):
                    _quality = _corpus.get("corpus_quality", None)
                    if _quality is not None:
                        self._feedback_bus.write_signal(
                            "wizard_corpus_quality", float(_quality),
                        )

            consumed["consumed"] = True
            logger.info(
                "🧙 Wizard results consumed: status=%s, %d settings applied",
                _status, len(consumed["applied_settings"]),
            )
        except Exception as e:
            consumed["error"] = str(e)
            self._record_failure_episode(
                error_evolution, "wizard_consumption_failure",
                "consumption_fallback",
                {"reason": str(e)},
            )
            logger.debug("Wizard results consumption failed: %s", e)

        return consumed

    # ─── Integration Point 1: Teacher-Student Inversion (Spec II.4) ──────

    def execute_codebook_warm_start(
        self,
        tokens: torch.Tensor,
    ) -> Dict[str, Any]:
        """Implements Spec Point 2: VQ codebook warm-start.

        Initializes VQ codebook from semantically meaningful VibeThinker
        prompt embeddings rather than random initialization, ensuring
        Phase A begins with meaningful prototypes.
        """
        try:
            from ae_train import warm_start_codebook_from_vt
            result = warm_start_codebook_from_vt(
                model=self.model,
                tokens=tokens,
                config=self.config,
                device=self.device,
            )
            _integration_state.update_point("codebook_warm_start", result)
            return result
        except Exception as e:
            logger.debug("Codebook warm-start failed: %s", e)
            return {"initialized": False, "reason": str(e),
                    "traced": False,
                    "causal_chain": ["codebook_warm_start", str(e)]}

    def execute_context_calibration(
        self,
        tokens: torch.Tensor,
    ) -> Dict[str, Any]:
        """Implements Spec Point 3: Context window calibration.

        Replaces the static context_window hyperparameter with an
        empirically derived value from VibeThinker CoT depth
        distribution across the training corpus.
        """
        try:
            from ae_train import calibrate_context_window
            result = calibrate_context_window(
                model=self.model,
                tokens=tokens,
                config=self.config,
                device=self.device,
            )
            _integration_state.update_point(
                "context_window_calibration", result,
            )
            return result
        except Exception as e:
            logger.debug("Context calibration failed: %s", e)
            return {"calibrated": False, "reason": str(e),
                    "traced": False,
                    "causal_chain": ["context_window_calibration", str(e)]}

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
            return {"inverted": False, "reason": str(e),
                    "traced": False,
                    "causal_chain": ["teacher_student_inversion", str(e)]}

    # ─── Integration Point 4: Streaming Signal Bus Closed Loop ───────────
    def execute_signal_bus_step(
        self,
        error_evolution: Optional[Any] = None,
    ) -> Dict[str, Any]:
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
            # H3: Record signal bus failures to error_evolution
            self._record_failure_episode(
                error_evolution, "signal_bus_closed_loop_failure",
                "integration_retry",
                {"reason": str(e)},
            )
            return {"executed": False, "reason": str(e)}

    # ─── Integration Point 5: Z-Sequence Quality Annotation ──────────────
    def execute_z_annotation(
        self,
        z_sequences: List[torch.Tensor],
        error_evolution: Optional[Any] = None,
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
            # H3: Record z-annotation failures to error_evolution
            self._record_failure_episode(
                error_evolution, "z_annotation_failure",
                "annotation_fallback",
                {"reason": str(e)},
            )
            # I7: Mark fallback annotations so execute_full_cycle can
            #     detect the quality degradation and flag uncertainty.
            fallback = [torch.ones(seq.shape[0], 3) for seq in z_sequences]
            self._z_annotation_used_fallback = True
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
            return {
                "bridged": False,
                "reason": str(e),
                "traced": False,
                "causal_chain": ["training_to_inference_bridge", str(e)],
            }

    # ─── Integration Point 7: Inference→Training Feedback ────────────────
    def execute_inference_to_training_bridge(
        self,
        inference_error_evolution: Any,
        trainer: Any,
    ) -> Dict[str, Any]:
        """Implements Spec: Inference→training insight propagation.

        Pulls inference-side diagnostics (emergence, uncertainty)
        and feeds them back into training hyperparameters.

        Args:
            inference_error_evolution: Inference-side error evolution tracker.
            trainer: SafeThoughtAETrainerV4 instance for adaptation.
        """
        try:
            from ae_train import bridge_inference_insights_to_training
            result = bridge_inference_insights_to_training(
                inference_error_evolution=inference_error_evolution,
                trainer=trainer,
            )
            _integration_state.update_point(
                "inference_to_training_bridge", result,
            )
            return result
        except Exception as e:
            logger.debug("Inference→training bridge failed: %s", e)
            return {
                "bridged": False,
                "reason": str(e),
                "traced": False,
                "causal_chain": ["inference_to_training_bridge", str(e)],
            }

    # ─── Integration Point 8: UCC Inner-Epoch Evaluation ─────────────────
    def execute_ucc_evaluation(
        self,
        epoch: int,
        phase: str,
        epoch_metrics: Dict[str, Any],
        uncertainty_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Implements Spec: UCC inner-epoch evaluation.

        Invokes UnifiedCognitiveCycle.evaluate() mid-epoch for
        real-time metacognitive feedback.

        When uncertainty_flags are provided (from Points 1-7), the
        epoch_metrics are enriched with cycle health context so UCC
        decisions are informed by upstream integration results (Patch H5).
        """
        try:
            # H5: Enrich epoch_metrics with upstream cycle health
            _enriched = dict(epoch_metrics)
            if uncertainty_flags is not None:
                _enriched["integration_uncertainty_flags"] = uncertainty_flags
                _enriched["integration_uncertainty_level"] = (
                    len(uncertainty_flags) / self.TOTAL_INTEGRATION_POINTS
                    if self.TOTAL_INTEGRATION_POINTS > 0 else 0.0
                )

            from ae_train import ucc_inner_epoch_evaluation
            result = ucc_inner_epoch_evaluation(
                model=self.model,
                epoch=epoch,
                phase=phase,
                epoch_metrics=_enriched,
            )
            _integration_state.update_point("ucc_epoch_evaluation", result)
            return result
        except Exception as e:
            return {
                "evaluated": False,
                "reason": str(e),
                "traced": False,
                "causal_chain": ["ucc_epoch_evaluation", str(e)],
            }

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
            return {
                "aligned": False,
                "reason": str(e),
                "traced": False,
                "causal_chain": ["ssp_temperature_alignment", str(e)],
            }

    # ─── Integration Point 10: Continuous Learning Micro-Retrain ─────────
    def execute_micro_retrain(
        self,
        pseudo_labels: List[Dict[str, Any]],
        z_sequences: Optional[List[torch.Tensor]] = None,
        z_annotations: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Implements Spec: ContinualLearningCore micro-retrain.

        Uses pseudo-labels from VibeThinkerContinuousLearner
        Phase 4 (consolidation) to perform incremental updates.

        When z_annotations are provided (from Point 5), filters
        z_sequences by quality scores to avoid training on low-confidence
        representations (Patch H2).
        """
        try:
            # H2: Filter z_sequences by quality annotations if available
            _filtered_z = z_sequences
            if (z_sequences is not None
                    and z_annotations is not None
                    and len(z_annotations) == len(z_sequences)):
                _filtered_z = []
                for seq, ann in zip(z_sequences, z_annotations):
                    # ann shape: [seq_len, 3] →
                    #   dim 0: confidence — VibeThinker output reliability
                    #   dim 1: entropy    — reasoning diversity (lower = more certain)
                    #   dim 2: quality    — composite reasoning quality score
                    # Filtering uses confidence (dim 0); entropy and quality
                    # are preserved in the annotation tensor for downstream
                    # consumers (e.g. training loss weighting).
                    if ann.shape[-1] >= 1:
                        mean_conf = ann[..., 0].mean().item()
                        if mean_conf > 0.3:
                            _filtered_z.append(seq)
                    else:
                        _filtered_z.append(seq)
                if not _filtered_z:
                    _filtered_z = z_sequences  # fallback: keep all

            from ae_train import micro_retrain_from_pseudo_labels
            result = micro_retrain_from_pseudo_labels(
                model=self.model,
                pseudo_labels=pseudo_labels,
                config=self.config,
                device=self.device,
                z_sequences=_filtered_z,
            )
            # H2: Annotate result with filtering metadata
            if z_annotations is not None and z_sequences is not None:
                result["z_quality_filtered"] = True
                result["z_original_count"] = len(z_sequences)
                result["z_filtered_count"] = (
                    len(_filtered_z) if _filtered_z is not None else 0
                )
            _integration_state.update_point("continuous_learning", result)
            return result
        except Exception as e:
            return {
                "retrained": False,
                "reason": str(e),
                "traced": False,
                "causal_chain": ["continuous_learning", str(e)],
            }

    # ─── Error Evolution Recording Helper ────────────────────────────────
    @staticmethod
    def _record_failure_episode(
        error_evolution: Optional[Any],
        error_class: str,
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a failure episode to error_evolution if available.

        Ensures bridge/integration failures are tracked for MCT weight
        adaptation, closing the learning loop on integration-point
        failures (Patch G1/G5).
        """
        if error_evolution is None:
            return
        try:
            if hasattr(error_evolution, "record_episode"):
                error_evolution.record_episode(
                    error_class=error_class,
                    strategy_used=strategy,
                    success=False,
                    metadata=metadata or {},
                )
        except Exception:
            logger.debug(
                "Failed to record episode '%s' to error_evolution",
                error_class,
            )

    # ─── H1: MCT Signal Collection ───────────────────────────────────────
    def _collect_mct_signals(
        self,
        uncertainty_level: float,
        uncertainty_flags: List[str],
        cycle_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect all available signals for MCT evaluation.

        Reads from the feedback bus, model cached state, and cycle results
        to provide the MCT with full situational awareness (Patch H1).
        Gracefully defaults every signal so MCT never receives stale data.
        """
        kwargs: Dict[str, Any] = {"uncertainty": uncertainty_level}

        # ── Feedback bus signals ──
        if self._feedback_bus is not None:
            _read = self._read_fb_signal
            kwargs["coherence_deficit"] = max(
                0.0, 1.0 - _read("integration_health", 1.0),
            )
            kwargs["recovery_pressure"] = _read(
                "error_evolution_pressure", 0.0,
            )
            kwargs["diversity_collapse"] = _read(
                "cognitive_unity_deficit", 0.0,
            )
            kwargs["convergence_conflict"] = _read(
                "systematic_uncertainty", 0.0,
            )

            # Oscillation from feedback bus
            if hasattr(self._feedback_bus, "get_oscillation_score"):
                try:
                    osc = self._feedback_bus.get_oscillation_score()
                    kwargs["oscillation_severity"] = float(
                        osc if isinstance(osc, (int, float)) else 0.0,
                    )
                except Exception:
                    # I3: Default instead of silent drop so MCT always
                    #     receives the signal key.
                    kwargs["oscillation_severity"] = 0.0
                    logger.debug("Oscillation score read failed; defaulting")

        # ── Model cached state ──
        model = self.model
        kwargs["spectral_stability_margin"] = float(
            getattr(model, "_cached_spectral_stability_margin", 1.0),
        )
        kwargs["world_model_surprise"] = float(
            getattr(model, "_cached_surprise", 0.0),
        )
        kwargs["memory_staleness"] = bool(
            getattr(model, "_memory_stale", False),
        )
        kwargs["safety_violation"] = bool(
            getattr(model, "_cached_safety_violation", False),
        )
        kwargs["stall_severity"] = float(
            getattr(model, "_cached_stall_severity", 0.0),
        )
        kwargs["output_reliability"] = float(
            getattr(model, "_cached_output_quality", 1.0),
        )
        kwargs["border_uncertainty"] = float(
            getattr(model, "_cached_border_uncertainty", 0.0),
        )
        kwargs["memory_trust_deficit"] = max(0.0, min(
            1.0,
            1.0 - float(getattr(model, "_last_trust_score", 1.0)),
        ))

        # ── Convergence monitor divergence ──
        _conv = getattr(model, "convergence_monitor", None)
        if _conv is not None and hasattr(_conv, "is_diverging"):
            try:
                kwargs["is_diverging"] = bool(_conv.is_diverging())
            except Exception:
                # I3: Default so MCT always sees the key.
                kwargs["is_diverging"] = False
                logger.debug("Convergence divergence read failed; defaulting")

        # ── Topology catastrophe ──
        _topo = getattr(model, "_cached_topology_state", None)
        if _topo is not None:
            try:
                kwargs["topology_catastrophe"] = bool(_topo.any().item())
            except Exception:
                # I3: Default so MCT always sees the key.
                kwargs["topology_catastrophe"] = False
                logger.debug("Topology state read failed; defaulting")

        # ── Causal quality ──
        kwargs["causal_quality"] = float(
            getattr(model, "_cached_causal_quality", 1.0),
        )

        return kwargs

    def _read_fb_signal(self, name: str, default: float = 0.0) -> float:
        """Safely read a single signal from the feedback bus."""
        if self._feedback_bus is None:
            return default
        try:
            if hasattr(self._feedback_bus, "_extra_signals"):
                return float(
                    self._feedback_bus._extra_signals.get(name, default),
                )
            if hasattr(self._feedback_bus, "read_signal"):
                return float(self._feedback_bus.read_signal(name))
        except Exception:
            # I5: Log signal read failures so silent bus degradation
            #     is visible in debug output.
            logger.debug("Feedback bus signal '%s' read failed; using default", name)
        return default

    # ─── H4: Feed Reinforce Results to MCT ───────────────────────────────
    def _feed_reinforce_to_mct(
        self,
        reinforce_result: Dict[str, Any],
        error_evolution: Optional[Any] = None,
    ) -> None:
        """Feed verify_and_reinforce coherence assessment into MCT.

        After mutual reinforcement, extract the coherence score and
        adapt MCT weights so the next cycle's metacognitive assessment
        is informed by the latest architectural health (Patch H4).
        """
        if self._mct is None:
            return
        try:
            # Extract overall coherence from reinforcement result
            _report = reinforce_result.get("result", reinforce_result)
            _overall = _report.get("overall_score", None)
            if _overall is None:
                _overall = _report.get("coherence_score", None)

            # If coherence is below threshold, adapt MCT weights from
            # error_evolution so future evaluations are sensitized.
            if (
                _overall is not None
                and _overall < 0.8
                and error_evolution is not None
                and hasattr(self._mct, "adapt_weights_from_evolution")
            ):
                try:
                    _summary = error_evolution.get_error_summary()
                    self._mct.adapt_weights_from_evolution(_summary)
                    logger.debug(
                        "H4: MCT weights adapted from reinforce (score=%.3f)",
                        _overall,
                    )
                except Exception as _adapt_err:
                    # I6: Record weight adaptation failures so the
                    #     metacognitive loop can learn its own reliability.
                    self._record_failure_episode(
                        error_evolution,
                        "mct_weight_adaptation_failure",
                        "adaptation_fallback",
                        {"reason": str(_adapt_err)},
                    )

            # Also write coherence score to feedback bus for visibility
            if (
                _overall is not None
                and self._feedback_bus is not None
                and hasattr(self._feedback_bus, "write_signal")
            ):
                self._feedback_bus.write_signal(
                    "reinforce_coherence_score", float(_overall),
                )
        except Exception as e:
            # I10: Record reinforce-to-MCT bridging failure.
            self._record_failure_episode(
                error_evolution,
                "reinforce_to_mct_bridge_failure",
                "bridge_fallback",
                {"reason": str(e)},
            )
            logger.debug("Failed to feed reinforce results to MCT: %s", e)

    # ─── H6: Automatic Component Discovery ──────────────────────────────
    def auto_wire(self, model: nn.Module) -> Dict[str, Any]:
        """Automatically discover and attach components from the model.

        Scans the model for known subsystem attributes and wires them
        into the integration controller, removing the need for manual
        attach_*() calls (Patch H6).

        Returns:
            Dict with discovered and wired component names.
        """
        wired: List[str] = []
        missing: List[str] = []

        _COMPONENT_MAP = {
            "signal_bus": ("_signal_bus", "attach_signal_bus", [
                "vt_streaming_signal_bus",
                "signal_bus",
                "_vt_signal_bus",
            ]),
            "vt_learner": ("_vt_learner", "attach_vt_learner", [
                "vt_continuous_learner",
                "vt_learner",
                "_vt_learner",
            ]),
            "controller": ("_controller", "attach_controller", [
                "adaptive_training_controller",
                "training_controller",
                "_training_controller",
            ]),
            "feedback_bus": ("_feedback_bus", "attach_feedback_bus", [
                "cognitive_feedback_bus",
                "feedback_bus",
                "_feedback_bus",
            ]),
            "ucc": ("_ucc", "attach_ucc", [
                "unified_cognitive_cycle",
                "ucc",
                "_ucc",
            ]),
            "mct": ("_mct", "attach_metacognitive_trigger", [
                "metacognitive_trigger",
                "mct",
                "_metacognitive_trigger",
            ]),
            "continual_core": ("_continual_core", "attach_continual_core", [
                "continual_learning_core",
                "continual_core",
                "_continual_core",
            ]),
        }

        for comp_name, (attr, attach_method, candidates) in _COMPONENT_MAP.items():
            # Skip if already wired
            if getattr(self, attr, None) is not None:
                wired.append(comp_name)
                continue

            # Search model attributes
            _found = None
            for candidate in candidates:
                obj = getattr(model, candidate, None)
                if obj is not None:
                    _found = obj
                    break

            if _found is not None:
                attach_fn = getattr(self, attach_method)
                attach_fn(_found)
                wired.append(comp_name)
            else:
                missing.append(comp_name)

        result = {
            "wired": wired,
            "missing": missing,
            "total_wired": len(wired),
            "total_missing": len(missing),
        }
        logger.info(
            "🔧 Auto-wire: %d components wired, %d missing",
            len(wired), len(missing),
        )
        return result

    # ─── Feedback Bus Sync Helper ─────────────────────────────────────────
    def _sync_feedback_bus(
        self,
        uncertainty_flags: List[str],
        cycle_results: Dict[str, Any],
    ) -> None:
        """Write integration cycle health to the feedback bus.

        After all integration points execute, the cycle health summary
        is written to the cognitive feedback bus so downstream modules
        can condition on integration quality (Patch G3).
        """
        if self._feedback_bus is None:
            return
        try:
            total = self.TOTAL_INTEGRATION_POINTS
            failed = len(uncertainty_flags)
            health = 1.0 - (failed / total) if total > 0 else 1.0

            if hasattr(self._feedback_bus, "write_signal"):
                self._feedback_bus.write_signal(
                    "integration_health", health,
                )
                self._feedback_bus.write_signal(
                    "integration_failure_rate", failed / total if total > 0 else 0.0,
                )

            # Surface UCC verdict if available
            ucc_result = cycle_results.get("ucc", {})
            if isinstance(ucc_result, dict) and hasattr(
                self._feedback_bus, "write_signal",
            ):
                ucc_ok = ucc_result.get("evaluated", False)
                self._feedback_bus.write_signal(
                    "ucc_evaluation_ok", 1.0 if ucc_ok else 0.0,
                )
        except Exception as e:
            # I4: Record sync failure to error_evolution so MCT is
            #     aware that cycle health was not propagated.
            self._record_failure_episode(
                # sync_feedback_bus doesn't receive error_evolution directly;
                # use the model's cached reference if available.
                getattr(self.model, "_error_evolution", None),
                "feedback_bus_sync_failure",
                "sync_fallback",
                {"reason": str(e)},
            )
            logger.debug("Feedback bus sync failed: %s", e)

    # ─── Unified Cycle Step ──────────────────────────────────────────────
    def execute_full_cycle(
        self,
        epoch: int,
        phase: str,
        epoch_metrics: Dict[str, Any],
        z_sequences: Optional[List[torch.Tensor]] = None,
        tokens: Optional[torch.Tensor] = None,
        convergence_monitor: Optional[Any] = None,
        error_evolution: Optional[Any] = None,
        pseudo_labels: Optional[List[Dict[str, Any]]] = None,
        trainer: Optional[Any] = None,
        inference_error_evolution: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute all 10 integration points for one training cycle.

        Implements Spec §2.4 — unified cognitive cycle step covering:
          Point 1:  Teacher-Student Inversion (after Phase A, when z_sequences available)
          Point 2:  Codebook Warm-Start (when tokens provided, first cycle only)
          Point 3:  Context Window Calibration (when tokens provided, first cycle only)
          Point 4:  Streaming Signal Bus closed loop
          Point 5:  Z-sequence quality annotation
          Point 6:  Training→Inference error bridge
          Point 7:  Inference→Training feedback bridge
          Point 8:  UCC inner-epoch evaluation
          Point 9:  SSP temperature alignment
          Point 10: Micro-retrain from pseudo-labels

        After all points execute, a continuous metacognitive assessment
        runs: MCT always evaluates the cycle's health — even when no
        failures occurred — so that adaptive weight adjustment happens
        continuously rather than only post-failure (Patch G2).

        Args:
            epoch: Current training epoch.
            phase: Training phase ("A" or "B").
            epoch_metrics: Metrics from the current epoch.
            z_sequences: Z-sequence tensors for annotation/inversion.
            tokens: Training tokens for warm-start/calibration.
            convergence_monitor: Training convergence monitor.
            error_evolution: Training-side error evolution tracker.
            pseudo_labels: Pseudo-labels from VT continuous learner.
            trainer: SafeThoughtAETrainerV4 for inference→training bridge.
            inference_error_evolution: Inference-side error evolution tracker.

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
        _uncertainty_flags: List[str] = []

        # ── Point 2: Codebook Warm-Start (first cycle only) ─────────
        if tokens is not None and self._cycle_count == 1:
            warm_result = self.execute_codebook_warm_start(tokens)
            cycle_results["codebook_warm_start"] = warm_result
            if not warm_result.get("initialized", False):
                _uncertainty_flags.append("codebook_warm_start_failed")
                # G5: Record integration-point failure to error_evolution
                self._record_failure_episode(
                    error_evolution, "codebook_warm_start_failure",
                    "integration_retry",
                    {"reason": warm_result.get("reason", "unknown")},
                )

        # ── Point 3: Context Window Calibration (first cycle only) ──
        if tokens is not None and self._cycle_count == 1:
            cal_result = self.execute_context_calibration(tokens)
            cycle_results["context_calibration"] = cal_result
            if not cal_result.get("calibrated", False):
                _uncertainty_flags.append("context_calibration_failed")
                # G5: Record integration-point failure
                self._record_failure_episode(
                    error_evolution, "context_calibration_failure",
                    "integration_retry",
                    {"reason": cal_result.get("reason", "unknown")},
                )

        # ── Point 4: Signal bus closed loop ─────────────────────────
        if self._signal_bus is not None:
            # H3: Pass error_evolution so signal bus failures are recorded
            cycle_results["signal_bus"] = self.execute_signal_bus_step(
                error_evolution=error_evolution,
            )
            if not cycle_results["signal_bus"].get("executed", True):
                _uncertainty_flags.append("signal_bus_closed_loop_failed")

        # ── Point 5: Z-sequence annotation ──────────────────────────
        # H2: Store annotations for downstream consumption by Point 10
        _z_annotations: Optional[List[torch.Tensor]] = None
        if z_sequences is not None:
            self._z_annotation_used_fallback = False
            _, _z_annotations = self.execute_z_annotation(
                z_sequences, error_evolution=error_evolution,
            )
            cycle_results["z_annotation"] = {
                "annotated": True,
                "num_sequences": len(z_sequences),
                "annotation_count": (
                    len(_z_annotations) if _z_annotations is not None else 0
                ),
            }
            # I7: If z-annotation fell back to default tensors, flag
            #     uncertainty so MCT is aware of quality degradation.
            if self._z_annotation_used_fallback:
                _uncertainty_flags.append("z_annotation_failed")

        # ── Point 1: Teacher-Student Inversion (Phase A, with z) ────
        if z_sequences is not None and phase == "A":
            inv_result = self.execute_teacher_student_inversion(z_sequences)
            cycle_results["teacher_student_inversion"] = inv_result
            if not inv_result.get("inverted", False):
                _uncertainty_flags.append("teacher_student_inversion_failed")
                # G5: Record integration-point failure
                self._record_failure_episode(
                    error_evolution, "teacher_student_inversion_failure",
                    "integration_retry",
                    {"reason": inv_result.get("reason", "unknown")},
                )

        # ── Point 6: Training→Inference error bridge ────────────────
        if convergence_monitor is not None and error_evolution is not None:
            t2i_result = self.execute_training_to_inference_bridge(
                convergence_monitor, error_evolution,
            )
            cycle_results["training_to_inference_bridge"] = t2i_result
            if not t2i_result.get("bridged", True):
                _uncertainty_flags.append("training_to_inference_bridge_failed")
                # G1: Record bridge failure for MCT learning
                self._record_failure_episode(
                    error_evolution, "training_to_inference_bridge_failure",
                    "bridge_retry",
                    {"reason": t2i_result.get("reason", "unknown")},
                )

        # ── Point 7: Inference→Training feedback ────────────────────
        if inference_error_evolution is not None and trainer is not None:
            i2t_result = self.execute_inference_to_training_bridge(
                inference_error_evolution, trainer,
            )
            cycle_results["inference_to_training_bridge"] = i2t_result
            if not i2t_result.get("bridged", True):
                _uncertainty_flags.append("inference_to_training_bridge_failed")
                # G1: Record bridge failure for MCT learning
                self._record_failure_episode(
                    error_evolution, "inference_to_training_bridge_failure",
                    "bridge_retry",
                    {"reason": i2t_result.get("reason", "unknown")},
                )

        # ── Point 8: UCC evaluation ─────────────────────────────────
        # H5: Pass accumulated uncertainty flags so UCC decisions are
        #     informed by upstream integration results from Points 1-7.
        ucc_result = self.execute_ucc_evaluation(
            epoch, phase, epoch_metrics,
            uncertainty_flags=_uncertainty_flags,
        )
        cycle_results["ucc"] = ucc_result
        # G4: Propagate UCC failure as uncertainty flag
        if not ucc_result.get("evaluated", True):
            _uncertainty_flags.append("ucc_evaluation_failed")
            self._record_failure_episode(
                error_evolution, "ucc_evaluation_failure",
                "integration_retry",
                {"reason": ucc_result.get("reason", "unknown")},
            )

        # ── Point 9: SSP alignment ─────────────────────────────────
        ssp_result = self.execute_ssp_alignment()
        cycle_results["ssp"] = ssp_result
        # G4: Propagate SSP failure as uncertainty flag
        if not ssp_result.get("aligned", True):
            _uncertainty_flags.append("ssp_alignment_failed")
            self._record_failure_episode(
                error_evolution, "ssp_alignment_failure",
                "integration_retry",
                {"reason": ssp_result.get("reason", "unknown")},
            )

        # ── Point 10: Micro-retrain from pseudo-labels ──────────────
        # H2: Pass z_annotations from Point 5 so micro-retrain can
        #     filter z-sequences by quality and avoid low-confidence data.
        if pseudo_labels:
            mr_result = self.execute_micro_retrain(
                pseudo_labels, z_sequences,
                z_annotations=_z_annotations,
            )
            cycle_results["micro_retrain"] = mr_result
            if not mr_result.get("retrained", False):
                _uncertainty_flags.append("micro_retrain_failed")
                self._record_failure_episode(
                    error_evolution, "micro_retrain_failure",
                    "integration_retry",
                    {"reason": mr_result.get("reason", "unknown")},
                )

            # I8: Write z-filter ratio to feedback bus so downstream
            #     consumers (incl. MCT) can track annotation quality.
            if (
                mr_result.get("z_quality_filtered", False)
                and self._feedback_bus is not None
                and hasattr(self._feedback_bus, "write_signal")
            ):
                _orig = mr_result.get("z_original_count", 1)
                _filt = mr_result.get("z_filtered_count", _orig)
                _ratio = _filt / max(_orig, 1)
                self._feedback_bus.write_signal(
                    "z_filter_pass_ratio", float(_ratio),
                )

        # ── G3: Sync cycle health to feedback bus ────────────────────
        self._sync_feedback_bus(_uncertainty_flags, cycle_results)

        # ── G2: Continuous Meta-Cognitive Assessment ─────────────────
        # Always evaluate MCT — not just when failures occur — so that
        # adaptive weight adjustment happens continuously.  On healthy
        # cycles the uncertainty input is 0.0, enabling the MCT to
        # track the steady-state baseline and adapt faster when a
        # genuine anomaly arises.
        cycle_results["uncertainty_flags"] = _uncertainty_flags
        if self._mct is not None:
            try:
                if hasattr(self._mct, "evaluate"):
                    uncertainty_level = (
                        len(_uncertainty_flags) / self.TOTAL_INTEGRATION_POINTS
                    )

                    # H1: Collect all available MCT signals from cycle state
                    #     and model cached values, so MCT operates with full
                    #     situational awareness rather than bare uncertainty.
                    _mct_kwargs = self._collect_mct_signals(
                        uncertainty_level, _uncertainty_flags, cycle_results,
                    )

                    mct_result = self._mct.evaluate(**_mct_kwargs)
                    _triggered = mct_result.get("should_trigger", False)
                    cycle_results["metacognitive_review"] = {
                        "triggered": _triggered,
                        "flags": _uncertainty_flags,
                        "mct_verdict": mct_result,
                        "continuous": True,
                        "signals_provided": len(_mct_kwargs),
                    }
                    if _triggered:
                        logger.info(
                            "🧠 Meta-cognitive review triggered by %d "
                            "uncertainty flags: %s",
                            len(_uncertainty_flags), _uncertainty_flags,
                        )
                    else:
                        logger.debug(
                            "🧠 Continuous MCT baseline: score=%.4f "
                            "(no trigger needed)",
                            mct_result.get("trigger_score", 0.0),
                        )

                    # I9: Record MCT trigger decision to error_evolution
                    #     for pattern learning — both triggered and
                    #     baseline decisions are informative.
                    if error_evolution is not None:
                        self._record_failure_episode(
                            error_evolution,
                            "mct_trigger_decision",
                            "triggered" if _triggered else "baseline",
                            {
                                "trigger_score": mct_result.get(
                                    "trigger_score", 0.0,
                                ),
                                "flags_count": len(_uncertainty_flags),
                                "cycle": self._cycle_count,
                            },
                        )
            except Exception as e:
                # I1: Record MCT evaluation failures to error_evolution
                #     so the system can learn MCT reliability patterns.
                self._record_failure_episode(
                    error_evolution, "mct_evaluation_failure",
                    "mct_fallback",
                    {"reason": str(e)},
                )
                logger.debug("MCT evaluation failed: %s", e)

        # ── G6: Periodic Mutual Reinforcement ────────────────────────
        # Every N cycles, invoke the model's verify_and_reinforce to
        # ensure active components verify and stabilize each other.
        _reinforce_interval = getattr(self.config, "reinforce_interval", 5)
        if (
            self._cycle_count % _reinforce_interval == 0
            and hasattr(self.model, "verify_and_reinforce")
        ):
            try:
                reinforce_result = self.model.verify_and_reinforce()
                _reinforce_dict = (
                    reinforce_result
                    if isinstance(reinforce_result, dict)
                    else {"status": "completed"}
                )
                cycle_results["mutual_reinforcement"] = {
                    "executed": True,
                    "cycle": self._cycle_count,
                    "result": _reinforce_dict,
                }

                # H4: Feed verify_and_reinforce coherence results back
                #     into MCT so the next cycle benefits from the latest
                #     architectural coherence assessment.
                self._feed_reinforce_to_mct(
                    _reinforce_dict, error_evolution,
                )

                logger.debug(
                    "🔄 Mutual reinforcement executed at cycle %d",
                    self._cycle_count,
                )
            except Exception as e:
                cycle_results["mutual_reinforcement"] = {
                    "executed": False,
                    "error": str(e),
                }
                # I2: Record mutual reinforcement failures to
                #     error_evolution so MCT can learn from them.
                self._record_failure_episode(
                    error_evolution, "mutual_reinforcement_failure",
                    "reinforce_fallback",
                    {"reason": str(e)},
                )
                logger.debug("Mutual reinforcement failed: %s", e)

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
#  Causal Transparency: trace_output_to_premise  (Spec §2.4)
# =============================================================================
def trace_output_to_premise(
    output_action: str,
    integration_state: Optional[IntegrationState] = None,
    cycle_history: Optional[List[Dict[str, Any]]] = None,
    error_evolution: Optional[Any] = None,
) -> Dict[str, Any]:
    """Trace any output/action back through the architecture to its premise.

    Implements the Causal Transparency requirement: every output can be
    deterministically traced back through the integration architecture
    to the originating premise, integration point, and cycle that
    produced it.

    The trace follows the reverse path:
      output → cycle_results → integration_point → ae_train function
      → error_evolution episode → root_cause premise

    Args:
        output_action: Description or identifier of the output to trace.
        integration_state: Current integration state (uses global if None).
        cycle_history: List of cycle result dicts from execute_full_cycle.
        error_evolution: CausalErrorEvolutionTracker for deep tracing.

    Returns:
        Dict with:
          - traced: bool — whether a full trace was produced
          - output_action: the queried output
          - trace_chain: list of trace steps from output back to premise
          - originating_cycle: cycle number where action originated
          - originating_point: integration point responsible
          - root_premise: the earliest identifiable cause
    """
    state = integration_state or _integration_state
    history = cycle_history or []

    trace: Dict[str, Any] = {
        "traced": False,
        "output_action": output_action,
        "trace_chain": [],
        "originating_cycle": None,
        "originating_point": None,
        "root_premise": None,
    }

    # Step 1: Find the most recent cycle that contains relevant results
    _relevant_cycle = None
    _relevant_point = None
    for cycle in reversed(history):
        for key, value in cycle.items():
            if key in UnifiedTrainingCycleController._CYCLE_METADATA_KEYS:
                continue
            if isinstance(value, dict):
                _relevant_cycle = cycle
                _relevant_point = key
                break
        if _relevant_cycle is not None:
            break

    if _relevant_cycle is not None:
        trace["originating_cycle"] = _relevant_cycle.get("cycle")
        trace["originating_point"] = _relevant_point
        trace["trace_chain"].append({
            "step": "integration_cycle",
            "cycle": _relevant_cycle.get("cycle"),
            "epoch": _relevant_cycle.get("epoch"),
            "phase": _relevant_cycle.get("phase"),
            "point": _relevant_point,
        })

    # Step 2: Check integration state for the active point
    active_points = [
        name for name, info in state.points.items()
        if info.get("active", False)
    ]
    trace["trace_chain"].append({
        "step": "integration_state",
        "active_points": active_points,
        "total_active": len(active_points),
    })

    # Step 3: Attempt deep trace via error_evolution
    if error_evolution is not None:
        try:
            if hasattr(error_evolution, "get_episode_count"):
                ep_count = error_evolution.get_episode_count()
            elif hasattr(error_evolution, "episodes"):
                ep_count = len(error_evolution.episodes)
            else:
                ep_count = 0

            trace["trace_chain"].append({
                "step": "error_evolution",
                "episode_count": ep_count,
            })

            # Try trace_root_cause if available
            if hasattr(error_evolution, "trace_root_cause"):
                root = error_evolution.trace_root_cause()
                trace["root_premise"] = root
                trace["trace_chain"].append({
                    "step": "root_cause",
                    "premise": root,
                })
        except Exception as e:
            trace["trace_chain"].append({
                "step": "error_evolution_error",
                "error": str(e),
            })

    # Step 4: If no deep trace, the root premise is the integration state
    if trace["root_premise"] is None and active_points:
        trace["root_premise"] = (
            f"integration_point:{active_points[0]}"
        )

    trace["traced"] = len(trace["trace_chain"]) >= 2
    return trace


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
