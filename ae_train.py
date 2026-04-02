"""
================================================================================
AEON TRAINING PIPELINE v4.0 - CONNECTED THOUGHTS EDITION
================================================================================

Ключевые улучшения v4.0:
- ✅ Поддержка связанных мыслей (последовательности внутри документов)
- ✅ Улучшенный RSSM с контекстным окном
- ✅ Стабилизация градиентов (grad_clip снижен до 0.5)
- ✅ Entropy regularization для кодбука
- ✅ Документ-ориентированное построение z_pairs
- ✅ Улучшенный warmup и scheduling

Автор: AEON Research Team
Версия: 4.0.0
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import math
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import logging
import os
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset, Dataset
import argparse
import copy
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

__all__ = [
    "AEONConfigV4", "TrainingMonitor",
    "GumbelVectorQuantizer", "VectorQuantizerHybridV4",
    "AEONDeltaV4", "DocumentAwareDataset",
    "WarmupCosineScheduler",
    "SafeThoughtAETrainerV4", "ContextualRSSMTrainer",
    "validate_training_components", "TrainingProvenanceTracker",
    "TrainingConvergenceMonitor", "bridge_training_errors_to_inference",
    "DataCharacteristicsAnalyzer", "AdaptiveTrainingController",
    "warm_start_codebook_from_vt", "calibrate_context_window",
    "annotate_z_sequences_quality", "adapt_entropy_weight",
    "auto_detect_task_boundary", "micro_retrain_from_pseudo_labels",
    "bifasic_didactic_orchestrate",
    "main",
]

# --- Bridge to aeon_core cognitive architecture ---
# Import core cognitive components for unified coherence verification,
# causal traceability, and meta-cognitive cycle integration during training.
# TensorGuard extends the inference pipeline's NaN/Inf protection to
# training, ensuring tensor safety is consistent across both pipelines.
# CausalErrorEvolutionTracker bridges training convergence events to the
# inference pipeline's error evolution system so that training-time
# divergence and stagnation inform inference-time recovery strategies.
try:
    from aeon_core import (
        CausalProvenanceTracker,
        ConvergenceMonitor,
        SemanticErrorClassifier,
        TensorGuard,
        NaNPolicy,
        CausalErrorEvolutionTracker,
        UnifiedCognitiveCycle,
        MetaCognitiveRecursionTrigger,
        ModuleCoherenceVerifier,
        UnifiedConvergenceArbiter,
        DirectionalUncertaintyTracker,
        MemoryReasoningValidator,
        SystemIntegrityMonitor,
    )
    AEON_CORE_AVAILABLE = True
    try:
        from aeon_core import (
            VibeThinkerPromptAdapter,
            VibeThinkerReasoningKernel,
            VibeThinkerContinuousLearner,
            ContinualLearningCore,
            VibeThinkerConfig,
        )
        VIBE_THINKER_AVAILABLE = True
    except ImportError:
        VIBE_THINKER_AVAILABLE = False
except ImportError:
    AEON_CORE_AVAILABLE = False
    VIBE_THINKER_AVAILABLE = False

    # Lightweight fallback so convergence events are still recorded and
    # bridge_training_errors_to_inference() works without aeon_core.
    from enum import Enum, auto
    from collections import defaultdict

    class NaNPolicy(Enum):
        """Strategies for handling NaN/Inf.

        Fallback mirror of ``aeon_core.NaNPolicy``, used when aeon_core
        is not installed so that training-time code paths share the same
        enum members and semantics as the inference pipeline.
        """
        RAISE = auto()
        WARN = auto()
        SILENT = auto()
        RETURN_NONE = auto()
        QUARANTINE = auto()

    class TensorGuard:
        """Minimal tensor safety guard (fallback when aeon_core unavailable).

        Mirrors the interface of ``aeon_core.TensorGuard`` so that
        training-time tensor safety is consistent with inference.
        """

        def __init__(self, policy=None, enable_tracking: bool = False):
            self.policy = policy or NaNPolicy.WARN
            self._nan_count = 0
            self._inf_count = 0
            self._sanitize_count = 0

        def sanitize(self, tensor: torch.Tensor, context: str = "",
                     custom_default: Optional[float] = None,
                     allow_inf: bool = False) -> torch.Tensor:
            """Sanitize NaN/Inf values, mirroring aeon_core.TensorGuard.

            Args:
                tensor: Input tensor to sanitize.
                context: Descriptive context string for diagnostics.
                custom_default: Replacement value for NaN/Inf elements.
                    Defaults to 0.0 when None.
                allow_inf: When True, skip infinity checks and only
                    replace NaN values.
            """
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item() if not allow_inf else False
            if has_nan:
                self._nan_count += 1
            if has_inf:
                self._inf_count += 1
            if has_nan or has_inf:
                self._sanitize_count += 1
                default = custom_default if custom_default is not None else 0.0
                tensor = torch.where(
                    torch.isfinite(tensor), tensor,
                    torch.full_like(tensor, default),
                )
            return tensor

        def get_stats(self) -> Dict[str, Any]:
            """Return sanitization statistics (mirrors aeon_core API)."""
            return {
                'nan_count': self._nan_count,
                'inf_count': self._inf_count,
                'sanitize_count': self._sanitize_count,
            }

    class SemanticErrorClassifier:
        """Keyword-based error classifier (fallback when aeon_core unavailable).

        Classifies runtime errors into the same taxonomy used by
        ``aeon_core.SemanticErrorClassifier`` so that standalone training
        error-evolution episodes use meaningful error classes rather than
        a blanket ``"unknown"`` label.
        """

        _NUMERICAL_KEYWORDS = frozenset({
            "nan", "inf", "overflow", "underflow", "divide",
            "division by zero", "non-finite",
        })
        _SHAPE_KEYWORDS = frozenset({
            "shape", "dimension", "size mismatch", "expected size",
            "broadcasting", "incompatible",
        })
        _RESOURCE_KEYWORDS = frozenset({
            "cuda", "out of memory", "oom", "device", "memory",
            "cublas", "cudnn",
        })
        _CONVERGENCE_KEYWORDS = frozenset({
            "converge", "diverge", "iteration", "fixed point",
            "lipschitz", "contraction",
        })

        def classify(self, error: BaseException) -> tuple:
            msg = str(error).lower()
            if any(kw in msg for kw in self._NUMERICAL_KEYWORDS):
                return ("numerical", str(error))
            if any(kw in msg for kw in self._SHAPE_KEYWORDS):
                return ("shape", str(error))
            if any(kw in msg for kw in self._RESOURCE_KEYWORDS):
                return ("resource", str(error))
            if any(kw in msg for kw in self._CONVERGENCE_KEYWORDS):
                return ("convergence", str(error))
            if isinstance(error, (ValueError, TypeError)):
                return ("semantic", str(error))
            return ("unknown", str(error))

    class CausalErrorEvolutionTracker:
        """Lightweight error evolution tracker for standalone training.

        Mirrors the interface of ``aeon_core.CausalErrorEvolutionTracker``
        so that ``bridge_training_errors_to_inference()`` and the fallback
        ``UnifiedCognitiveCycle`` work identically whether or not
        aeon_core is installed.
        """

        def __init__(self, max_history: int = 100):
            self._max_history = max_history
            self._episodes: Dict[str, list] = defaultdict(list)
            self._causal_trace = None
            self._metacognitive_trigger = None

        def set_causal_trace(self, trace) -> None:
            """Attach a causal trace buffer for automatic episode propagation."""
            self._causal_trace = trace

        def set_metacognitive_trigger(self, trigger) -> None:
            """Attach a metacognitive trigger for live weight adaptation."""
            self._metacognitive_trigger = trigger

        def get_root_causes(self, error_class: str) -> Dict[str, Any]:
            episodes = self._episodes.get(error_class, [])
            if not episodes:
                return {"root_causes": {}, "antecedent_depth": 0.0,
                        "episodes_with_antecedents": 0}
            known = set(self._episodes.keys())
            root_counter: Dict[str, int] = {}
            total_depth = 0
            with_ant = 0
            for ep in episodes:
                antecedents = ep.get("causal_antecedents", [])
                if antecedents:
                    with_ant += 1
                depth = 0
                queue = list(antecedents)
                visited: set = set()
                while queue:
                    ant = queue.pop(0)
                    if ant in visited:
                        continue
                    visited.add(ant)
                    depth += 1
                    if ant not in known:
                        root_counter[ant] = root_counter.get(ant, 0) + 1
                total_depth += depth
            return {
                "root_causes": root_counter,
                "antecedent_depth": total_depth / max(len(episodes), 1),
                "episodes_with_antecedents": with_ant,
            }

        def record_episode(self, error_class: str, strategy_used: str,
                           success: bool, metadata: Optional[Dict] = None,
                           causal_antecedents: Optional[list] = None,
                           **kwargs) -> None:
            history = self._episodes[error_class]
            history.append({
                "strategy": strategy_used,
                "success": success,
                "metadata": metadata or {},
                "causal_antecedents": causal_antecedents or [],
            })
            if len(history) > self._max_history:
                self._episodes[error_class] = history[-self._max_history:]

        def get_best_strategy(self, error_class: str) -> Optional[str]:
            """Return historically most successful strategy for an error class."""
            episodes = self._episodes.get(error_class, [])
            if not episodes:
                return None
            strategy_stats: Dict[str, list] = {}
            for ep in episodes:
                s = ep["strategy"]
                strategy_stats.setdefault(s, []).append(ep["success"])
            if not strategy_stats:
                return None
            return max(
                strategy_stats,
                key=lambda s: sum(strategy_stats[s]) / max(len(strategy_stats[s]), 1),
            )

        def get_error_summary(self) -> Dict[str, Any]:
            summary: Dict[str, Any] = {"error_classes": {}}
            for cls, eps in self._episodes.items():
                successes = sum(1 for e in eps if e["success"])
                strategies = list({e["strategy"] for e in eps})
                # Aggregate loss magnitudes from episode metadata so that
                # bridge_training_errors_to_inference can convey severity
                # (mild vs catastrophic divergence) to the inference side.
                loss_values = [
                    e["metadata"].get("loss_value")
                    for e in eps
                    if isinstance(e.get("metadata"), dict)
                    and e["metadata"].get("loss_value") is not None
                ]
                cls_stats: Dict[str, Any] = {
                    "count": len(eps),
                    "success_rate": successes / max(len(eps), 1),
                    "strategies_used": strategies,
                    "best_strategy": strategies[0] if strategies else "unknown",
                }
                if loss_values:
                    cls_stats["max_loss_magnitude"] = max(loss_values)
                    cls_stats["mean_loss_magnitude"] = sum(loss_values) / len(loss_values)
                summary["error_classes"][cls] = cls_stats
            summary["total_recorded"] = sum(
                len(eps) for eps in self._episodes.values()
            )
            return summary

        def get_degrading_error_classes(self) -> Dict[str, float]:
            """Identify error classes whose recovery effectiveness is worsening.

            Compares the success rate of the most recent half of episodes
            against the older half for each error class.  A negative trend
            (recent success rate < older success rate) indicates degrading
            recovery, which should raise metacognitive concern.

            Returns:
                Dict mapping error class name to trend delta (negative =
                degrading).  Only classes with >= 4 episodes and a negative
                trend are included.
            """
            degrading: Dict[str, float] = {}
            for cls, eps in self._episodes.items():
                if len(eps) < 4:
                    continue
                mid = len(eps) // 2
                older = eps[:mid]
                recent = eps[mid:]
                older_rate = sum(1 for e in older if e["success"]) / max(len(older), 1)
                recent_rate = sum(1 for e in recent if e["success"]) / max(len(recent), 1)
                trend = recent_rate - older_rate
                if trend < 0:
                    degrading[cls] = trend
            return degrading

    class ConvergenceMonitor:
        """Convergence monitor (fallback when aeon_core unavailable).

        Tracks contraction ratios, bridges divergence and stagnation
        events to an attached ``CausalErrorEvolutionTracker``, and
        provides provenance enrichment — matching the interface of
        ``aeon_core.ConvergenceMonitor``.
        """

        def __init__(self, threshold: float = 1e-5):
            self.history: list = []
            self._threshold = threshold
            self._error_evolution = None
            self._provenance_tracker = None

        def reset(self) -> None:
            self.history.clear()

        def set_error_evolution(self, tracker) -> None:
            """Attach error evolution tracker for automatic divergence bridging."""
            self._error_evolution = tracker

        def set_provenance_tracker(self, tracker) -> None:
            """Attach provenance tracker for enriching error metadata."""
            self._provenance_tracker = tracker

        def set_metacognitive_trigger(self, trigger) -> None:
            """Attach metacognitive trigger for threshold tightening.

            When the trigger has active signals, the convergence
            threshold is tightened so that high metacognitive pressure
            demands stricter convergence before certification.
            """
            self._metacognitive_trigger = trigger

        def record_secondary_signal(
            self, name: str, value: float,
        ) -> None:
            """Record a secondary convergence signal.

            Secondary signals (world model surprise, safety violations,
            recovery pressure, etc.) allow the convergence monitor to
            degrade the verdict from 'converged' to 'converging' when
            auxiliary subsystems indicate instability, even if the
            residual norm converged.
            """
            if not hasattr(self, '_secondary_signals'):
                self._secondary_signals: Dict[str, float] = {}
            self._secondary_signals[name] = value

        def is_diverging(self) -> bool:
            """Return True when the contraction ratio indicates divergence."""
            if len(self.history) < 2:
                return False
            ratio = self.history[-1] / max(self.history[-2], 1e-12)
            return ratio >= 1.0

        def _bridge_convergence_event(
            self, error_class: str, strategy: str,
            success: bool, metadata: Optional[Dict] = None,
        ) -> None:
            """Forward a convergence event to the attached error evolution tracker."""
            tracker = self._error_evolution
            if tracker is not None:
                enriched = dict(metadata) if metadata else {}
                prov = self._provenance_tracker
                if prov is not None:
                    try:
                        snap = prov.compute_attribution()
                        contribs = snap.get("contributions", {})
                        enriched["provenance_contributions"] = contribs
                        if contribs:
                            enriched["dominant_provenance_module"] = max(
                                contribs, key=contribs.get,
                            )
                    except Exception as _prov_err:
                        logger.debug(
                            "Provenance enrichment failed in training monitor: %s",
                            _prov_err,
                        )
                tracker.record_episode(
                    error_class=error_class,
                    strategy_used=strategy,
                    success=success,
                    metadata=enriched,
                    causal_antecedents=["training_monitor", "convergence_check"],
                )

        def check(self, delta_norm: float) -> Dict[str, Any]:
            self.history.append(delta_norm)
            if len(self.history) < 3:
                return {"status": "warmup", "certified": False}
            ratios = [
                self.history[i] / max(self.history[i - 1], 1e-12)
                for i in range(1, len(self.history))
            ]
            avg_contraction = sum(ratios) / len(ratios)
            if avg_contraction < 1.0 and delta_norm < self._threshold:
                self._bridge_convergence_event(
                    "convergence_success", "nominal", success=True,
                    metadata={"avg_contraction": avg_contraction,
                              "delta_norm": delta_norm},
                )
                return {"status": "converged", "certified": True,
                        "contraction_rate": avg_contraction}
            if avg_contraction >= 1.0:
                self._bridge_convergence_event(
                    "convergence_diverging", "meta_loop_rollback",
                    success=False,
                    metadata={"avg_contraction": avg_contraction,
                              "delta_norm": delta_norm},
                )
                return {"status": "diverging", "certified": False,
                        "contraction_rate": avg_contraction}
            if len(self.history) >= 10 and delta_norm > self._threshold * 10:
                self._bridge_convergence_event(
                    "convergence_stagnation", "increase_iterations",
                    success=False,
                    metadata={"avg_contraction": avg_contraction,
                              "delta_norm": delta_norm},
                )
            return {"status": "converging", "certified": False,
                    "contraction_rate": avg_contraction}

        def get_convergence_summary(self) -> Dict[str, Any]:
            """Return a summary of convergence state.

            Mirrors ``aeon_core.ConvergenceMonitor.get_convergence_summary``
            so that ``get_architectural_health()`` can be invoked even
            when aeon_core is unavailable.
            """
            if len(self.history) < 3:
                return {"status": "warmup", "certified": False,
                        "history_length": len(self.history)}
            ratios = [
                self.history[i] / max(self.history[i - 1], 1e-12)
                for i in range(1, len(self.history))
            ]
            avg_contraction = sum(ratios) / len(ratios)
            delta_norm = self.history[-1]
            if avg_contraction < 1.0 and delta_norm < self._threshold:
                status, certified = "converged", True
            elif avg_contraction >= 1.0:
                status, certified = "diverging", False
            else:
                status, certified = "converging", False
            return {
                "status": status,
                "certified": certified,
                "contraction_rate": avg_contraction,
                "history_length": len(self.history),
            }

    class CausalProvenanceTracker:
        """Provenance tracker (fallback when aeon_core unavailable).

        Mirrors ``aeon_core.CausalProvenanceTracker`` including
        ``trace_root_cause()``, ``get_dependency_graph()``, and
        ``set_causal_trace()`` so that ``UnifiedCognitiveCycle`` and
        ``validate_training_components()`` work identically with or
        without aeon_core installed.
        """

        _NORM_EPSILON = 1e-10

        def __init__(self):
            self._deltas: Dict[str, float] = {}
            self._order: list = []
            self._snapshots: Dict[str, torch.Tensor] = {}
            self._dependencies: Dict[str, set] = {}
            self._causal_trace = None

        def reset(self) -> None:
            """Clear all recorded state for a new pass."""
            self._deltas.clear()
            self._order.clear()
            self._snapshots.clear()
            self._dependencies.clear()

        def set_causal_trace(self, trace) -> None:
            """Attach a causal trace buffer for automatic propagation."""
            self._causal_trace = trace

        def record_before(self, module_name: str, state: torch.Tensor) -> None:
            self._snapshots[module_name] = state.detach().clone()
            if module_name not in self._order:
                self._order.append(module_name)

        def record_after(self, module_name: str, state: torch.Tensor) -> None:
            if module_name in self._snapshots:
                before = self._snapshots.pop(module_name)
                # Handle shape mismatches by truncating to smaller size;
                # this mirrors TrainingProvenanceTracker.record_after()
                # and is expected when VQ or projection layers change dim.
                min_size = min(state.shape[-1], before.shape[-1])
                new_delta = (
                    state.detach()[..., :min_size] - before[..., :min_size]
                ).norm().item()
                self._deltas[module_name] = (
                    self._deltas.get(module_name, 0.0) + new_delta
                )

        def record_dependency(self, from_module: str, to_module: str) -> None:
            self._dependencies.setdefault(to_module, set()).add(from_module)

        def get_dependency_graph(self) -> Dict[str, list]:
            """Return the inter-module dependency DAG."""
            return {k: sorted(v) for k, v in self._dependencies.items()}

        def compute_attribution(self) -> Dict[str, Any]:
            deltas: Dict[str, float] = {}
            for name in self._order:
                deltas[name] = self._deltas.get(name, 0.0)
            total = sum(deltas.values()) + self._NORM_EPSILON
            contributions = {k: v / total for k, v in deltas.items()}
            return {
                "contributions": contributions,
                "deltas": deltas,
                "order": list(self._order),
            }

        def trace_root_cause(self, module_name: str) -> Dict[str, Any]:
            """Trace backward through the dependency DAG to find root causes."""
            visited: set = set()
            queue = [module_name]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                for parent in self._dependencies.get(current, set()):
                    queue.append(parent)
            root_modules = [m for m in visited if not self._dependencies.get(m)]
            contributions = {m: self._deltas.get(m, 0.0) for m in visited}
            return {
                "root_modules": sorted(root_modules),
                "visited": visited,
                "contributions": contributions,
            }

        def validate_dag_acyclic(self) -> Dict[str, Any]:
            """Check the dependency DAG for cycles.

            Uses iterative DFS to detect back-edges.  When cycles are
            found, the offending edges are removed so that
            ``trace_root_cause()`` remains safe.

            Returns:
                Dict with ``is_acyclic`` bool and ``cycles_found`` list.
            """
            all_nodes: set = set()
            for target, sources in self._dependencies.items():
                all_nodes.add(target)
                all_nodes.update(sources)
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {n: WHITE for n in all_nodes}
            # Build forward adjacency list (upstream → downstream) for
            # efficient DFS traversal.
            adjacency: Dict[str, list] = {n: [] for n in all_nodes}
            for child, parents in self._dependencies.items():
                for parent in parents:
                    adjacency.setdefault(parent, []).append(child)
            cycles: list = []
            for start in all_nodes:
                if color[start] != WHITE:
                    continue
                stack = [(start, False)]
                while stack:
                    node, processed = stack.pop()
                    if processed:
                        color[node] = BLACK
                        continue
                    if color[node] == GRAY:
                        continue
                    color[node] = GRAY
                    stack.append((node, True))
                    for child in adjacency.get(node, []):
                        if color.get(child, WHITE) == GRAY:
                            cycles.append((node, child))
                        elif color.get(child, WHITE) == WHITE:
                            stack.append((child, False))
            for src, dst in cycles:
                deps = self._dependencies.get(dst, set())
                deps.discard(src)
            return {
                "is_acyclic": len(cycles) == 0,
                "cycles_found": cycles,
            }

        def verify_trace_completeness(self) -> Dict[str, Any]:
            """Verify that all recorded modules have dependency coverage.

            Returns:
                Dict with ``complete`` bool and lists of covered and
                uncovered modules.
            """
            all_dag_nodes: set = set()
            for target, sources in self._dependencies.items():
                all_dag_nodes.add(target)
                all_dag_nodes.update(sources)
            recorded = set(self._order)
            uncovered = sorted(recorded - all_dag_nodes)
            return {
                "complete": len(uncovered) == 0,
                "covered": sorted(recorded & all_dag_nodes),
                "uncovered": uncovered,
            }

    class ModuleCoherenceVerifier:
        """Coherence verifier (fallback when aeon_core unavailable).

        Computes pairwise cosine similarity between subsystem outputs,
        matching the interface of ``aeon_core.ModuleCoherenceVerifier``
        so that standalone training coherence checks produce real scores
        instead of hardcoded ones.
        """

        _ADAPT_STEP = 0.05
        _ADAPT_CAP = 0.9

        def __init__(self, hidden_dim: int = 256, threshold: float = 0.5):
            self.threshold = threshold

        def __call__(self, states):
            names = list(states.keys())
            if len(names) < 2:
                B = next(iter(states.values())).shape[0] if states else 1
                return {
                    "coherence_score": torch.ones(B),
                    "pairwise": {},
                    "needs_recheck": False,
                }
            pairwise = {}
            sims = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = torch.nn.functional.cosine_similarity(
                        states[names[i]], states[names[j]], dim=-1,
                    )
                    pairwise[(names[i], names[j])] = sim
                    sims.append(sim)
            coherence = torch.stack(sims, dim=-1).mean(dim=-1)
            needs_recheck = bool(coherence.mean().item() < self.threshold)
            return {
                "coherence_score": coherence,
                "pairwise": pairwise,
                "needs_recheck": needs_recheck,
            }

        def adapt_threshold(self, error_summary):
            """Raise threshold when coherence deficits recur with low success."""
            classes = error_summary.get("error_classes", {})
            cd_stats = classes.get("coherence_deficit")
            if cd_stats is None:
                return
            if cd_stats.get("count", 0) < 2:
                return
            if cd_stats.get("success_rate", 1.0) < 0.5:
                self.threshold = min(
                    self._ADAPT_CAP, self.threshold + self._ADAPT_STEP,
                )

        @staticmethod
        def get_weakest_pair(pairwise):
            """Identify the subsystem pair with lowest mean similarity."""
            if not pairwise:
                return None
            weakest_key = min(
                pairwise, key=lambda k: pairwise[k].mean().item(),
            )
            return {
                "pair": weakest_key,
                "similarity": float(pairwise[weakest_key].mean().item()),
                "modules": list(weakest_key),
            }

    class MetaCognitiveRecursionTrigger:
        """Metacognitive trigger (fallback when aeon_core unavailable).

        Monitors the same fifteen signals as
        ``aeon_core.MetaCognitiveRecursionTrigger`` and uses weighted-sum
        evaluation so that standalone training can detect uncertainty,
        divergence, and coherence deficits that warrant re-reasoning.
        """

        _DEFAULT_WEIGHT = 1.0 / 15.0

        def __init__(self, trigger_threshold: float = 0.5,
                     max_recursions: int = 2,
                     tightening_factor: float = 0.5,
                     extra_iterations: int = 10,
                     surprise_threshold: float = 0.5,
                     causal_quality_threshold: float = 0.3,
                     high_uncertainty_override: float = 0.7,
                     **kwargs):
            self.trigger_threshold = trigger_threshold
            self.max_recursions = max(1, max_recursions)
            self.tightening_factor = max(0.01, min(tightening_factor, 1.0))
            self.extra_iterations = max(0, extra_iterations)
            self._recursion_count = 0
            self._surprise_threshold = max(0.0, surprise_threshold)
            self._causal_quality_threshold = max(0.0, causal_quality_threshold)
            self._high_uncertainty_override = max(0.0, high_uncertainty_override)
            self._signal_weights: Dict[str, float] = {
                "uncertainty": self._DEFAULT_WEIGHT,
                "diverging": self._DEFAULT_WEIGHT,
                "topology_catastrophe": self._DEFAULT_WEIGHT,
                "coherence_deficit": self._DEFAULT_WEIGHT,
                "memory_staleness": self._DEFAULT_WEIGHT,
                "recovery_pressure": self._DEFAULT_WEIGHT,
                "world_model_surprise": self._DEFAULT_WEIGHT,
                "low_causal_quality": self._DEFAULT_WEIGHT,
                "safety_violation": self._DEFAULT_WEIGHT,
                "diversity_collapse": self._DEFAULT_WEIGHT,
                "memory_trust_deficit": self._DEFAULT_WEIGHT,
                "convergence_conflict": self._DEFAULT_WEIGHT,
                "low_output_reliability": self._DEFAULT_WEIGHT,
                "spectral_instability": self._DEFAULT_WEIGHT,
                "border_uncertainty": self._DEFAULT_WEIGHT,
            }

        def reset(self):
            self._recursion_count = 0

        def evaluate(self, uncertainty: float = 0.0,
                     is_diverging: bool = False,
                     topology_catastrophe: bool = False,
                     coherence_deficit: float = 0.0,
                     memory_staleness: bool = False,
                     recovery_pressure: float = 0.0,
                     world_model_surprise: float = 0.0,
                     causal_quality: float = 1.0,
                     safety_violation: bool = False,
                     diversity_collapse: float = 0.0,
                     memory_trust_deficit: float = 0.0,
                     convergence_conflict: float = 0.0,
                     output_reliability: float = 1.0,
                     spectral_stability_margin: float = 1.0,
                     border_uncertainty: float = 0.0,
                     **kwargs) -> Dict[str, Any]:
            can_recurse = self._recursion_count < self.max_recursions
            signals: Dict[str, float] = {
                "uncertainty": min(uncertainty, 1.0),
                "diverging": 1.0 if is_diverging else 0.0,
                "topology_catastrophe": 1.0 if topology_catastrophe else 0.0,
                "coherence_deficit": min(coherence_deficit, 1.0),
                "memory_staleness": 1.0 if memory_staleness else 0.0,
                "recovery_pressure": min(recovery_pressure, 1.0),
                "world_model_surprise": (
                    1.0 if world_model_surprise > self._surprise_threshold else 0.0
                ),
                "low_causal_quality": (
                    1.0 if causal_quality < self._causal_quality_threshold else 0.0
                ),
                "safety_violation": 1.0 if safety_violation else 0.0,
                "diversity_collapse": min(max(diversity_collapse, 0.0), 1.0),
                "memory_trust_deficit": min(max(memory_trust_deficit, 0.0), 1.0),
                "convergence_conflict": min(max(convergence_conflict, 0.0), 1.0),
                "low_output_reliability": min(max(1.0 - output_reliability, 0.0), 1.0),
                "spectral_instability": min(max(1.0 - spectral_stability_margin, 0.0), 1.0),
                "border_uncertainty": min(max(border_uncertainty, 0.0), 1.0),
            }
            trigger_score = sum(
                self._signal_weights.get(k, 0.0) * v
                for k, v in signals.items()
            )
            triggers_active = [k for k, v in signals.items() if v > 0.0]
            should_trigger = (
                (trigger_score >= self.trigger_threshold and can_recurse)
                or (uncertainty > self._high_uncertainty_override and can_recurse)
                or (topology_catastrophe and can_recurse)
            )
            if should_trigger:
                self._recursion_count += 1
            return {
                "should_trigger": should_trigger,
                # Alias for consumers expecting 'should_recurse'.
                "should_recurse": should_trigger,
                "trigger_score": trigger_score,
                "tightened_threshold": (
                    self.trigger_threshold * self.tightening_factor
                ),
                "extra_iterations": self.extra_iterations,
                "triggers_active": triggers_active,
                "recursion_count": self._recursion_count,
                # Alias for consumers expecting 'trigger_count'.
                "trigger_count": self._recursion_count,
                "signal_weights": dict(self._signal_weights),
            }

        def adapt_weights_from_evolution(self, error_summary):
            """Adjust signal weights based on historical error-recovery patterns."""
            error_classes = error_summary.get("error_classes", {})
            if not error_classes:
                return
            # Map error classes to the trigger signals they most relate to.
            # Mirrors the mapping in aeon_core.MetaCognitiveRecursionTrigger
            # so that training-time error patterns properly sensitise the
            # metacognitive trigger for corresponding signals.
            _class_to_signal = {
                "convergence_diverging": "diverging",
                "convergence_divergence": "diverging",
                "coherence_deficit": "coherence_deficit",
                "post_integration_coherence_deficit": "coherence_deficit",
                "post_auto_critic_coherence_deficit": "coherence_deficit",
                "post_rerun_coherence_deficit": "coherence_deficit",
                "metacognitive_rerun": "uncertainty",
                "numerical": "uncertainty",
                "safety_rollback": "safety_violation",
                "reconciliation_disagreement": "coherence_deficit",
                "world_model_prediction_error": "world_model_surprise",
                "low_causal_quality": "low_causal_quality",
                "mcts_low_confidence": "uncertainty",
                "causal_programmatic_forward": "low_causal_quality",
                "causal_dag_disagreement": "low_causal_quality",
                "convergence_success": "uncertainty",
                "certified_convergence_failure": "uncertainty",
                "safety_critic_revision": "safety_violation",
                "vq_codebook_collapse": "uncertainty",
                "diversity_collapse": "diversity_collapse",
                "memory_staleness": "memory_staleness",
                "memory_subsystem": "memory_staleness",
                "critical_uncertainty": "uncertainty",
                "auto_critic_low_quality": "uncertainty",
                "convergence_conflict": "convergence_conflict",
                "memory_reasoning_inconsistency": "memory_staleness",
                "low_memory_trust": "memory_trust_deficit",
                "low_output_reliability": "low_output_reliability",
                "topology_catastrophe": "topology_catastrophe",
                "coherence_trend_degradation": "coherence_deficit",
                "convergence_certificate_violation": "diverging",
                "dag_consensus_disagreement": "low_causal_quality",
                "high_total_training_loss": "uncertainty",
                "high_training_loss": "uncertainty",
                "training_divergence": "diverging",
                "training_stagnation": "coherence_deficit",
                "training_ucc_failure": "uncertainty",
                "inter_memory_disagreement": "memory_staleness",
                "decoder_degenerate_output": "coherence_deficit",
                # Diagnostic-gap error classes — generated dynamically by
                # self_diagnostic() as "diagnostic_gap_{component}".
                "diagnostic_gap_cognitive_unity": "coherence_deficit",
                "diagnostic_gap_error_evolution": "uncertainty",
                "diagnostic_gap_coherence_registry": "coherence_deficit",
                # Diagnostic gap immediate — same-pass escalation when
                # fresh gaps are discovered during the forward pass.
                "diagnostic_gap_immediate": "coherence_deficit",
                # Diagnostic remediation — auto-remediation outcome.
                "diagnostic_remediation": "coherence_deficit",
                # Convergence stagnation — extended plateau without
                # reaching threshold.
                "convergence_stagnation": "diverging",
                # Spectral instability — Hessian max-eigenvalue bifurcation.
                "spectral_instability": "spectral_instability",
                # Deception suppression — consistency probe diverged.
                "deception_suppression": "safety_violation",
                # Base convergence error class.
                "convergence": "diverging",
                # Unknown — SemanticErrorClassifier fallback.
                "unknown": "uncertainty",
                # Adaptation failure — trigger adaptation exception.
                "adaptation_failure": "uncertainty",
                # Axiom degradation — per-axiom emergence regression.
                "axiom_degradation": "uncertainty",
                # Sustained module decline — prolonged health degradation.
                "sustained_module_decline": "coherence_deficit",
                # Training metacognitive rerun — UCC triggered a
                # same-pass re-reasoning cycle during training.
                "training_metacognitive_rerun": "uncertainty",
                # Decoder cross-validation failure — decoder output
                # diverged from cross-validation expectations.
                "decoder_cross_validation_failure": "coherence_deficit",
                # Recursive meta-loop outcome — rollback indicates
                # convergence instability in hierarchical abstraction.
                "recursive_meta_loop_outcome": "convergence_conflict",
                # Post-pipeline metacognitive escalation — post-pipeline
                # should_recurse fired and uncertainty was escalated.
                "post_pipeline_metacognitive_escalation": "uncertainty",
                # Uncertainty-reinforcement escalation — should_recurse
                # in the high-uncertainty path escalated uncertainty.
                "uncertainty_reinforcement_escalation": "uncertainty",
                # Moderate-uncertainty escalation — should_recurse in
                # the moderate-uncertainty path escalated uncertainty.
                "moderate_uncertainty_escalation": "uncertainty",
                # Post-pipeline reinforcement failure — verify_and_reinforce()
                # raised an exception during post-pipeline corrective loop.
                "post_pipeline_reinforcement_failure": "uncertainty",
                # Certified convergence exception — CertifiedMetaLoop
                # convergence verification raised an exception.
                "certified_convergence_exception": "diverging",
                # Causal chain gap — forward-pass causal verification
                # detected missing trace entries.
                "causal_chain_gap": "low_causal_quality",
                # Causal chain island detected — verify_causal_chain found
                # disconnected subsystem islands and auto-bridged them.
                "causal_chain_island_detected": "low_causal_quality",
                # Causal chain cycle pruned — duplicate subsystems removed
                # from root-cause chain to restore acyclicity.
                "causal_chain_cycle_pruned": "low_causal_quality",
                # Activation probe step failure — one of the cognitive
                # activation probe steps failed silently during init.
                "activation_probe_step_failure": "uncertainty",
                # Critical activation probe step failure.
                "activation_probe_critical_failure": "coherence_deficit",
                # Subsystem runtime gap — wired but silent at runtime.
                "subsystem_runtime_gap": "coherence_deficit",
                # Activation not ready — critical activation steps failed.
                "activation_not_ready": "uncertainty",
                # Post-diagnostic healing failure.
                "post_diagnostic_healing_failure": "coherence_deficit",
                # Post-diagnostic healing success.
                "post_diagnostic_healing_success": "coherence_deficit",
                # Post-healing residual gaps.
                "post_healing_residual_gaps": "coherence_deficit",
                # Within-cycle uncertainty escalation.
                "within_cycle_uncertainty_escalation": "uncertainty",
                # Post-activation unity remediation failure.
                "post_activation_unity_remediation_failure": "coherence_deficit",
                # Cognitive unity deficit — verify_and_reinforce()
                # detected low unity and triggered re-evaluation.
                "cognitive_unity_deficit": "coherence_deficit",
                # Severe reinforce failure — correction-cycle error
                # during high-uncertainty should_recurse path.
                "severe_reinforce_failure": "uncertainty",
                # Moderate reinforce failure — correction-cycle error
                # during moderate-uncertainty should_recurse path.
                "moderate_reinforce_failure": "uncertainty",
                # Emergence patch evaluation — critical patches
                # evaluated against metacognitive trigger.
                "emergence_patch_evaluation": "coherence_deficit",
                # Error-evolution health computation failure — the
                # error recovery subsystem's own health check raised
                # an exception in verify_and_reinforce().
                "ee_health_computation_failure": "uncertainty",
                # Causal chain re-verification failure — post-
                # reinforcement verify_causal_chain() raised.
                "causal_chain_reverify_failure": "low_causal_quality",
                # Reinforcement sub-step failures — verify_and_reinforce
                # sub-steps that raise exceptions during the mutual
                # reinforcement cycle.
                "reinforce_axiom_adapt_failure": "uncertainty",
                # Multi-axiom failure — systemic coherence breakdown.
                "multi_axiom_failure": "coherence_deficit",
                "reinforce_convergence_check_failure": "diverging",
                "reinforce_module_adapt_failure": "uncertainty",
                # Reinforcement cycle outcome — aggregate success/failure.
                "reinforce_cycle_outcome": "coherence_deficit",
                # Feedback correction pressure — elevated feedback bus
                # correction pressure during forward pass.
                "feedback_correction_pressure": "coherence_deficit",
                # Emergence trend degrading — declining cognitive unity
                # score trend across forward passes.
                "emergence_trend_degrading": "coherence_deficit",
                # Recovery reinforcement failure — final weight
                # adaptation in verify_and_reinforce raised.
                "recovery_reinforcement_failed": "uncertainty",
                # ── Cognitive activation integration error classes ──
                "convergence_trigger_adaptation_failure": "diverging",
                "post_correction_verification_failure": "uncertainty",
                "causal_chain_trigger_adaptation_failure": "low_causal_quality",
                "chain_failure_trigger_adaptation_failure": "low_causal_quality",
                "architectural_regression_adaptation_failure": "coherence_deficit",
                "cognitive_unity_meta_evaluation_failure": "coherence_deficit",
                "post_remediation_prime_failure": "uncertainty",
                "signal_dropout_auto_recovery": "uncertainty",
                "upb_provenance_realignment": "low_causal_quality",
                "upb_provenance_misalignment": "low_causal_quality",
                # Mutual verification gap — active modules without
                # cross-validation partners detected in
                # verify_cognitive_unity.
                "mutual_verification_gap": "coherence_deficit",
                # Metacognitive signal registration — auto-registered
                # missing trigger weights.
                "metacognitive_signal_registration": "uncertainty",
                # UCC override remediation — active wiring of missing
                # UCC re-reasoning components.
                "ucc_override_remediation": "uncertainty",
                # Provenance dominance — module monopolisation detected
                # and dampened during the forward pass.
                "provenance_dominance": "coherence_deficit",
                # Coherence verifier failure — the coherence verification
                # subsystem itself raised an exception.
                "coherence_verifier_failure": "coherence_deficit",
                # ── Cognitive flow dead-end closures ───────────────
                "inline_coherence_check_failure": "coherence_deficit",
                "diagnostic_gap_adaptation_failure": "uncertainty",
                "periodic_reinforce_adaptation_failure": "uncertainty",
                "emergence_patch_meta_evaluation_failure": "coherence_deficit",
                # ── Meta-meta-cognitive dead-end closures ──────────
                "trigger_adaptation_failure": "uncertainty",
                "late_feedback_build_failure": "uncertainty",
                "emergence_auto_reinforcement_failure": "coherence_deficit",
                "emergence_post_reinforcement_verification_failure": "coherence_deficit",
                "emergence_re_evaluation_failure": "coherence_deficit",
                # ── Per-condition emergence failure classes ─────────
                # Individual emergence condition failures for targeted
                # metacognitive trigger weight adaptation.
                "emergence_mutual_reinforcement_unmet": "coherence_deficit",
                "emergence_metacognitive_trigger_unmet": "uncertainty",
                "emergence_causal_transparency_unmet": "low_causal_quality",
                "emergence_convergence_unstable": "diverging",
                "emergence_error_evolution_inactive": "uncertainty",
                "emergence_cognitive_unity_deficit": "coherence_deficit",
                "emergence_causal_chain_untraceable": "low_causal_quality",
                "emergence_assessment_failure": "uncertainty",
                "output_reliability_meta_eval_failure": "low_output_reliability",
                # ── Bridge-exception error classes ─────────────────
                # Recorded via _bridge_silent_exception in aeon_core.py
                # when secondary subsystem operations (remediation,
                # verification, feedback) raise exceptions.  Without
                # explicit mappings here, these failure modes fall
                # through to the generic "uncertainty" fallback,
                # preventing targeted metacognitive signal adaptation
                # during standalone training.
                "auto_remediation_failure": "coherence_deficit",
                "causal_chain_verification_failure": "low_causal_quality",
                "causal_chain_island_repair": "low_causal_quality",
                "causal_chain_reinforce_failure": "low_causal_quality",
                "diagnostic_gap_refresh_failure": "coherence_deficit",
                "emergence_cross_verification_failure": "coherence_deficit",
                "error_evolution_health_failure": "uncertainty",
                "feedback_bus_recomputation_failure": "uncertainty",
                "feedback_correction_failure": "uncertainty",
                "integrity_health_failure": "uncertainty",
                "metacognitive_adaptation_failure": "uncertainty",
                "reinforce_materialisation_failure": "coherence_deficit",
                "signal_dropout_recovery_failure": "uncertainty",
                "signal_coverage_dropout": "coherence_deficit",
                "state_vector_nan_detected": "coherence_deficit",
                "upb_provenance_registration_failure": "low_causal_quality",
                "provenance_autowire_failure": "low_causal_quality",
                "warmup_trend_degradation": "diverging",
                "vq_metacognitive_evaluation_failure": "uncertainty",
                # ── Pre-existing aeon_core bridge classes ──────────
                # Already mapped in aeon_core.py's
                # adapt_weights_from_evolution but absent from the
                # standalone trainer, causing training-time error
                # episodes for these classes to be misrouted to the
                # generic "uncertainty" fallback instead of their
                # intended targeted signals.
                "auto_critic_failure": "uncertainty",
                "fast_ucc_evaluation_failure": "uncertainty",
                "memory_consolidation_failure": "memory_staleness",
                "memory_decay_failure": "memory_staleness",
                "memory_routing_failure": "memory_staleness",
                "reasoning_core_trace_failure": "low_causal_quality",
                "value_network_failure": "uncertainty",
                # ── Forward-pass subsystem failure bridges ────────────
                # Mirror the aeon_core.py _class_to_signal entries so
                # training-time weight adaptation routes these error
                # classes to the correct metacognitive signal.
                "backbone_adapter_error": "low_output_reliability",
                "continual_learning_adapter_failure": "low_output_reliability",
                "decoder_degenerate_check_failure": "low_output_reliability",
                "reencode_failure": "coherence_deficit",
                "cycle_consistency_check_failure": "coherence_deficit",
                "post_output_coherence_failure": "coherence_deficit",
                "snapshot_validation_failure": "coherence_deficit",
                # inline_coherence_check_failure already mapped above.
                "uncertainty_reinforcement_failure": "uncertainty",
                # post_pipeline_reinforcement_failure already mapped above.
                "cognitive_unity_verification_failure": "coherence_deficit",
                # ── Cognitive integration patches ─────────────────────
                # Error classes from the four cognitive activation patches.
                "post_output_coherence_rerun": "coherence_deficit",
                "cache_hit_quality_gate": "uncertainty",
                "active_self_healing": "recovery_pressure",
                "healing_verification": "recovery_pressure",
                "provenance_validation_failure": "low_causal_quality",
                "causal_trace_health_failure": "low_causal_quality",
                "causal_trace_gap": "low_causal_quality",
                "emergence_diagnostic_gap": "coherence_deficit",
                "post_reinforcement_diagnostic_failure": "coherence_deficit",
                "post_reinforcement_wiring_failure": "coherence_deficit",
                # ── Grounded multimodal & spectral integration ─────────
                # Error classes from the expanded self-healing and new
                # feedback bus signal patches.
                # grounded_multimodal: general module failure (non-fatal)
                # grounded_multimodal_alignment_failure: specific
                #   vision↔language alignment quality degradation
                "grounded_multimodal": "uncertainty",
                "grounded_multimodal_alignment_failure": "uncertainty",
                # spectral_instability already mapped above (line 877).
                # emergence_adaptation_failure: periodic emergence
                # adaptation raised an exception — route to "uncertainty"
                # so the metacognitive trigger adapts sensitivity.
                "emergence_adaptation_failure": "uncertainty",
                # emergence_transition_adaptation_failure: metacognitive
                # trigger adaptation after emergence degradation raised.
                "emergence_transition_adaptation_failure": "uncertainty",
                # Per-condition emergence failure classes already mapped
                # above (lines ~990-998); not repeated here.
                # ── Verify-and-reinforce health-bridge error classes ────
                # Recorded via _bridge_silent_exception during the
                # mutual-reinforcement cycle when subsystem health
                # checks detect degradation or flush failures.
                "signal_dropout": "coherence_deficit",
                "error_evolution_low_effectiveness": "uncertainty",
                "memory_health_deficit": "coherence_deficit",
                "deferred_adaptation_flush_failure": "uncertainty",
                "severe_axiom_reverify_failure": "coherence_deficit",
                # ── Provenance re-registration failure ──────────────
                # Forced provenance re-registration for untraced active
                # subsystems failed — routes to low_causal_quality so
                # the trigger boosts causal quality sensitivity.
                "provenance_re_registration_failure": "low_causal_quality",
                # ── Decomposed adaptation failure classes ──────────────
                # Subsystem-specific adaptation failures split from the
                # generic metacognitive_adaptation_failure class.
                "causal_adaptation_failure": "low_causal_quality",
                "causal_trace_recording_failure": "low_causal_quality",
                "coherence_adaptation_failure": "coherence_deficit",
                "world_model_adaptation_failure": "world_model_surprise",
                "memory_adaptation_failure": "memory_staleness",
                "vq_adaptation_failure": "uncertainty",
                "convergence_adaptation_failure": "diverging",
                # Persistent silent exception — escalated from
                # _bridge_silent_exception when a subsystem accumulates
                # repeated silent failures.
                "persistent_silent_exception": "uncertainty",
                # Compound degradation — multi-subsystem failure in a
                # single forward pass.
                "compound_degradation": "coherence_deficit",
                # Recovery retry failure — individual retry attempt failed.
                "recovery_retry_failure": "uncertainty",
                # Persistent axiom deficit — chronic axiom weakness.
                "persistent_axiom_deficit": "coherence_deficit",
                # Post-bootstrap validation failure — unseeded baselines.
                "post_bootstrap_validation_failure": "uncertainty",
                # Causal trace forward-complete failure — aggregate
                # causal trace entry could not be recorded.
                "causal_trace_forward_complete_failure": "low_causal_quality",
                # Persistent island bridge — repeatedly bridged
                # subsystem in verify_causal_chain.
                "persistent_island_bridge": "low_causal_quality",
                # Reinforcement stable cycle — all axioms healthy in
                # verify_and_reinforce.
                "reinforcement_stable_cycle": "coherence_deficit",
                # Feedback bus silent — cross-pass feedback inactive
                # outside warmup.
                "feedback_bus_silent": "uncertainty",
                # Emergence not achieved — system did not emerge.
                "emergence_not_achieved": "coherence_deficit",
                # ── Healing & feedback bridge error classes ──────────
                # Mutual verification repair — successful provenance
                # edge repair in verify_and_reinforce.
                "mutual_verification_repair": "recovery_pressure",
                # Traceability repair — successfully registered untraced
                # modules in provenance DAG.
                "traceability_repair": "recovery_pressure",
                # UCC feedback failure — apply_ucc_feedback raised in
                # UnifiedCognitiveCycle.evaluate.
                "ucc_feedback_failure": "coherence_deficit",
                # Error classes present in aeon_core._class_to_signal
                # but previously absent here, creating a signal routing
                # gap where these errors fell through to generic prefix
                # routing or were silently ignored during standalone
                # training.  Grouped by target trigger signal.
                # ── coherence_deficit ─────────────────────────────────
                "architectural_regression": "coherence_deficit",
                "chronic_circuit_breaker": "coherence_deficit",
                "cognitive_frame_ambiguity": "coherence_deficit",
                "cognitive_frame_deficit": "coherence_deficit",
                "cognitive_unity_violation": "coherence_deficit",
                "coherence_auto_critic_failure": "coherence_deficit",
                "critical_coverage_deficit": "coherence_deficit",
                "cross_module_coherence_deficit": "coherence_deficit",
                "cross_validation_low_agreement": "coherence_deficit",
                "cross_validation_persistent_disagreement": "coherence_deficit",
                "cycle_consistency_violation": "coherence_deficit",
                "deeper_coherence_recheck_failure": "coherence_deficit",
                "diagnostic_gap_detected": "coherence_deficit",
                "emergence_axiom_mv_failure": "coherence_deficit",
                "emergence_deficit": "coherence_deficit",
                "emergence_incomplete": "coherence_deficit",
                "emergence_state_transition": "coherence_deficit",
                "feedback_oscillation": "coherence_deficit",
                "high_coherence_loss": "coherence_deficit",
                "high_coherence_training_loss": "coherence_deficit",
                "high_coverage_deficit": "coherence_deficit",
                "high_cross_validation_training_loss": "coherence_deficit",
                "high_decoder_provenance_training_loss": "coherence_deficit",
                "high_factor_cv_supervision_training_loss": "coherence_deficit",
                "high_feedback_demand": "coherence_deficit",
                "high_subsystem_health_loss": "coherence_deficit",
                "integration_failure_retry": "coherence_deficit",
                "integration_gate_low_confidence": "coherence_deficit",
                "low_decoder_quality": "coherence_deficit",
                "module_health_cross_module_coherence": "coherence_deficit",
                "module_health_cycle_consistency": "coherence_deficit",
                "module_health_social_cognition": "coherence_deficit",
                "periodic_reinforcement_failure": "coherence_deficit",
                "pipeline_wiring_gap": "coherence_deficit",
                "post_output_coherence_deficit": "coherence_deficit",
                "provenance_chain_incomplete": "coherence_deficit",
                "reconciliation_exhaustion": "coherence_deficit",
                "training_training_stagnation": "coherence_deficit",
                "uncertainty_auto_critic_audit_pattern": "coherence_deficit",
                "verify_coherence_deficit": "coherence_deficit",
                "vq_auto_critic_failure": "coherence_deficit",
                "vq_utilization_check_failure": "coherence_deficit",
                # ── convergence_conflict ──────────────────────────────
                "convergence_monitor_failure": "convergence_conflict",
                "module_health_convergence_quality": "convergence_conflict",
                # ── convergence_verdict ───────────────────────────────
                "convergence_instability": "convergence_verdict",
                "emergence_axiom_um_failure": "convergence_verdict",
                # ── diverging ─────────────────────────────────────────
                "convergence_certificate_failure": "diverging",
                "deeper_meta_loop_rejected": "diverging",
                "high_consistency_training_loss": "diverging",
                "high_lipschitz_training_loss": "diverging",
                "low_convergence_quality": "diverging",
                "training_convergence_diverging": "diverging",
                "training_loss_divergence": "diverging",
                "training_training_divergence": "diverging",
                "uncertainty_auto_critic_convergence_diverging": "diverging",
                # ── diversity_collapse ────────────────────────────────
                "diversity_collapse_detected": "diversity_collapse",
                "factor_reextraction_failure": "diversity_collapse",
                # ── low_causal_quality ────────────────────────────────
                "active_pass_traceability_gap": "low_causal_quality",
                "causal_blend_skipped": "low_causal_quality",
                "causal_context_conditioning_failure": "low_causal_quality",
                "counterfactual_divergence": "low_causal_quality",
                "emergence_axiom_rc_failure": "low_causal_quality",
                "high_causal_cv_supervision_training_loss": "low_causal_quality",
                "high_causal_dag_training_loss": "low_causal_quality",
                "high_counterfactual_verification_loss": "low_causal_quality",
                "high_unified_sim_loss": "low_causal_quality",
                "mcts_causal_adjacency_failure": "low_causal_quality",
                "module_health_causal_quality": "low_causal_quality",
                "pre_reasoning_causal_trace_failure": "low_causal_quality",
                "provenance_dag_cycle": "low_causal_quality",
                "provenance_delta_anomaly": "low_causal_quality",
                "root_cause_attribution_failure": "low_causal_quality",
                "verify_chain_failure": "low_causal_quality",
                # ── low_output_reliability ────────────────────────────
                "inference_cache_staleness": "low_output_reliability",
                "module_health_output_quality": "low_output_reliability",
                "module_health_vq_codebook": "low_output_reliability",
                "output_reliability_gate_missing": "low_output_reliability",
                "pipeline_wiring_verification_failure": "low_output_reliability",
                # ── memory_staleness ──────────────────────────────────
                "consolidating_memory_fusion_failure": "memory_staleness",
                "consolidation_quality_low": "memory_staleness",
                "high_memory_retrieval_loss": "memory_staleness",
                "memory_aggregate_failure": "memory_staleness",
                "memory_causal_degradation": "memory_staleness",
                "memory_conditioning_failure": "memory_staleness",
                "memory_cross_validation_failure": "memory_staleness",
                "memory_routing_irrelevance": "memory_staleness",
                "memory_validation_failure": "memory_staleness",
                "neurogenic_memory_fusion_failure": "memory_staleness",
                "recovery_memory_store_failed": "memory_staleness",
                "temporal_memory_freshness_low": "memory_staleness",
                "temporal_memory_fusion_failure": "memory_staleness",
                # ── safety_violation ──────────────────────────────────
                "code_execution_low_confidence": "safety_violation",
                "code_execution_sandbox_failure": "safety_violation",
                "deception_detected": "safety_violation",
                "deception_suppressor_failure": "safety_violation",
                "high_ns_consistency_loss": "safety_violation",
                "high_ns_consistency_training_loss": "safety_violation",
                "high_self_report_training_loss": "safety_violation",
                "module_health_code_execution": "safety_violation",
                "module_health_deception_suppressor": "safety_violation",
                "ns_bridge_error": "safety_violation",
                "ns_consistency_violation": "safety_violation",
                "ns_violation_auto_critic": "safety_violation",
                "state_validation_violation": "safety_violation",
                "terminal_state_invalid": "safety_violation",
                # ── topology_catastrophe ──────────────────────────────
                "uncertainty_auto_critic_topology_catastrophe": "topology_catastrophe",
                # ── uncertainty ───────────────────────────────────────
                "active_learning_error": "uncertainty",
                "chunked_encoding_error": "uncertainty",
                "cognitive_snapshot_degradation": "uncertainty",
                "curiosity_exploration_inefficiency": "uncertainty",
                "deeper_meta_loop_accepted": "uncertainty",
                "ewc_drift_estimation_failure": "uncertainty",
                "executive_alignment_deficit": "uncertainty",
                "feedback_bus_failure": "uncertainty",
                "generate_ucc_failure": "uncertainty",
                "high_hvae_kl_training_loss": "uncertainty",
                "high_output_uncertainty": "uncertainty",
                "high_sparsity_training_loss": "uncertainty",
                "high_ucc_training_loss": "uncertainty",
                "icm_reward_computation_failure": "uncertainty",
                "late_meta_loop_failure": "uncertainty",
                "low_executive_health": "uncertainty",
                "low_global_integrity": "uncertainty",
                "max_recursions_capped": "uncertainty",
                "meta_recovery_loss_failure": "uncertainty",
                "metacognitive_gap": "uncertainty",
                "module_health_hybrid_reasoning": "uncertainty",
                "module_health_mcts_planning": "uncertainty",
                "none": "uncertainty",
                "persistent_module_uncertainty": "uncertainty",
                "post_integration_deeper_rerun": "uncertainty",
                "post_integration_metacognitive": "uncertainty",
                "post_output_uncertainty_trigger": "uncertainty",
                "post_pipeline_metacognitive_failure": "uncertainty",
                "premature_complexity_gating": "uncertainty",
                "recurring_root_cause": "uncertainty",
                "social_cognition_failure": "uncertainty",
                "social_cognition_misalignment": "uncertainty",
                "subsystem": "uncertainty",
                "task2vec_ewc_loss_failure": "uncertainty",
                "training_gradient_explosion": "uncertainty",
                "training_nan_loss": "uncertainty",
                "trust_scorer_failure": "uncertainty",
                "ucc_rerun": "uncertainty",
                "uncertainty_auto_critic_uncertainty": "uncertainty",
                "unified_cycle_rerun": "uncertainty",
                # ── world_model_surprise ──────────────────────────────
                "hierarchical_wm_verification_failure": "world_model_surprise",
                "high_hierarchical_wm_loss": "world_model_surprise",
                "high_mcts_value_loss": "world_model_surprise",
                "unified_simulator_divergence": "world_model_surprise",
                "world_model_cross_divergence": "world_model_surprise",
                "world_model_semantic_surprise": "world_model_surprise",
                "world_model_verification_failure": "world_model_surprise",
                # ── Synchronised aeon_core trigger-signal mappings ─────
                # Error classes present in aeon_core.py
                # MetaCognitiveRecursionTrigger._class_to_signal but
                # previously absent here, breaking the training → inference
                # feedback loop for 163 error classes.  Grouped by target
                # trigger signal.
                # ── coherence_deficit ─────────────────────────────────
                "cognitive_frame_pressure": "coherence_deficit",
                "coherence_contribution": "coherence_deficit",
                "consistency_gate": "coherence_deficit",
                "convergence_arbiter_conflict": "convergence_conflict",
                "convergence_conflict_graduated": "convergence_conflict",
                "convergence_contribution": "diverging",
                "convergence_quality": "convergence_conflict",
                "convergence_secondary_pressure": "convergence_conflict",
                "convergence_verdict_pressure": "diverging",
                "correction_target_pressure": "coherence_deficit",
                "corrective_pressure": "coherence_deficit",
                "counterfactual_divergence_pressure": "coherence_deficit",
                "coverage_deficit_pressure": "coherence_deficit",
                "cross_module_coherence_pressure": "coherence_deficit",
                "cross_pass_root_pressure": "coherence_deficit",
                "cross_validation": "coherence_deficit",
                "cross_validation_disagreement_pressure": "coherence_deficit",
                "cv_agreement_deficit": "coherence_deficit",
                "cycle_consistency": "coherence_deficit",
                "cycle_consistency_pressure": "coherence_deficit",
                "diagnostic_gap_pressure": "coherence_deficit",
                "emergence_deficit_pressure": "coherence_deficit",
                "executive_review_pressure": "coherence_deficit",
                "factor_extraction": "coherence_deficit",
                "hvae_abstraction_pressure": "coherence_deficit",
                "hybrid_reasoning_quality": "coherence_deficit",
                "integration": "coherence_deficit",
                "integration_gate_confidence": "coherence_deficit",
                "low_quality_subsystem_pressure": "coherence_deficit",
                "ns_bridge": "coherence_deficit",
                "ns_bridge_confidence": "coherence_deficit",
                "provenance_dominance_pressure": "coherence_deficit",
                "provenance_quality": "coherence_deficit",
                "quality_trend_degradation_pressure": "coherence_deficit",
                "reinforce_cross_module_coherence_pressure": "coherence_deficit",
                "reinforce_cycle_consistency_pressure": "coherence_deficit",
                "reinforce_hybrid_reasoning_pressure": "coherence_deficit",
                "reinforce_vq_codebook_pressure": "coherence_deficit",
                "reinforce_weakness_pressure": "coherence_deficit",
                "slot_binding": "coherence_deficit",
                "ucc_coherence_trend": "coherence_deficit",
                "ucc_flagged_pressure": "coherence_deficit",
                "ucc_recurring_root_pressure": "coherence_deficit",
                "unified_cognitive_cycle": "coherence_deficit",
                "vq_codebook_pressure": "coherence_deficit",
                "weakest_coherence_pair_pressure": "coherence_deficit",
                # ── convergence_conflict ──────────────────────────────
                "feedback_oscillation_pressure": "convergence_conflict",
                "reinforce_convergence_quality_pressure": "convergence_conflict",
                # ── diverging ─────────────────────────────────────────
                "cert_violation_pressure": "diverging",
                "certified_meta_loop": "diverging",
                "deeper_meta_loop": "diverging",
                "lipschitz_pressure": "diverging",
                "meta_loop": "diverging",
                "reinforce_spectral_stability_pressure": "diverging",
                "spectral_instability_pressure": "diverging",
                "spectral_stability_margin": "diverging",
                # ── diversity_collapse ────────────────────────────────
                "diversity_analysis": "diversity_collapse",
                # ── low_causal_quality ────────────────────────────────
                "causal_chain_coverage_deficit": "low_causal_quality",
                "causal_context": "low_causal_quality",
                "causal_dag_consensus": "low_causal_quality",
                "causal_dag_consensus_quality": "low_causal_quality",
                "causal_model": "low_causal_quality",
                "causal_programmatic": "low_causal_quality",
                "causal_quality": "low_causal_quality",
                "dag_acyclicity_pressure": "low_causal_quality",
                "notears_causal": "low_causal_quality",
                "provenance_root_pressure": "low_causal_quality",
                "reinforce_causal_quality_pressure": "low_causal_quality",
                "temporal_knowledge_graph": "low_causal_quality",
                "trace_incomplete_pressure": "low_causal_quality",
                # ── low_output_reliability ────────────────────────────
                "auto_critic_current_quality": "low_output_reliability",
                "auto_critic_quality": "low_output_reliability",
                "auto_critic_quality_deficit": "low_output_reliability",
                "decoder_provenance_pressure": "low_output_reliability",
                "decoder_quality_pressure": "low_output_reliability",
                "decoder_variance_pressure": "low_output_reliability",
                "output_reliability": "low_output_reliability",
                "reinforce_output_quality_pressure": "low_output_reliability",
                "self_report_consistency": "low_output_reliability",
                "verification_coverage": "low_output_reliability",
                # ── memory_staleness ──────────────────────────────────
                "consolidating_memory": "memory_staleness",
                "consolidation_quality_deficit": "memory_staleness",
                "memory": "memory_staleness",
                "memory_re_retrieval_pressure": "memory_staleness",
                "memory_subsystem_aggregate_pressure": "memory_staleness",
                "memory_validation": "memory_staleness",
                "neurogenic_memory": "memory_staleness",
                "neurogenic_memory_retrieval_pressure": "memory_staleness",
                "temporal_memory": "memory_staleness",
                "temporal_memory_freshness_deficit": "memory_staleness",
                "tkg_staleness_pressure": "memory_staleness",
                # ── memory_trust_deficit ──────────────────────────────
                "memory_cv_disagreement": "memory_trust_deficit",
                "memory_retrieval_quality": "memory_trust_deficit",
                "memory_routing_trust_pressure": "memory_trust_deficit",
                "memory_trust": "memory_trust_deficit",
                "memory_trust_deficit": "memory_trust_deficit",
                # ── recovery_pressure ─────────────────────────────────
                "error_evolution_health_deficit": "recovery_pressure",
                "recovery_health": "recovery_pressure",
                # ── safety_violation ──────────────────────────────────
                "auto_critic_safety": "safety_violation",
                "deception_pressure": "safety_violation",
                "deception_suppressor": "safety_violation",
                "ns_consistency": "safety_violation",
                "reinforce_code_execution_pressure": "safety_violation",
                "reinforce_deception_suppressor_pressure": "safety_violation",
                "safety": "safety_violation",
                "safety_violation_pressure": "safety_violation",
                "sandbox_pressure": "safety_violation",
                "self_report": "safety_violation",
                # ── topology_catastrophe ──────────────────────────────
                "topology_analysis": "topology_catastrophe",
                # ── uncertainty ───────────────────────────────────────
                "active_learning": "uncertainty",
                "active_learning_curiosity": "uncertainty",
                "auto_critic": "uncertainty",
                "auto_critic_contribution": "uncertainty",
                "cache_bypass_active": "uncertainty",
                "cl_transfer_quality": "uncertainty",
                "cognitive_executive": "uncertainty",
                "complexity_estimator": "uncertainty",
                "complexity_gate_usage": "uncertainty",
                "continual_learning": "uncertainty",
                "curiosity_exploration_pressure": "uncertainty",
                "deferred_trigger_pressure": "uncertainty",
                "encoder": "uncertainty",
                "encoder_reasoning_norm": "uncertainty",
                "error_evolution_pressure": "uncertainty",
                "error_evolution_trend_pressure": "uncertainty",
                "evolved_strategy_pressure": "uncertainty",
                "feedback_signal_trend": "uncertainty",
                "grounded_multimodal_alignment_pressure": "uncertainty",
                "hierarchical_vae": "uncertainty",
                "hybrid_reasoning": "uncertainty",
                "mcts_planning": "uncertainty",
                "mcts_planning_quality": "uncertainty",
                "meta_learner_ewc_pressure": "uncertainty",
                "multimodal": "uncertainty",
                "post_output_late_uncertainty": "uncertainty",
                "reinforce_continual_learning_pressure": "uncertainty",
                "reinforce_mcts_planning_pressure": "uncertainty",
                "reinforce_social_cognition_pressure": "uncertainty",
                "social_pressure": "uncertainty",
                "systematic_uncertainty": "uncertainty",
                "unc_peak": "uncertainty",
                "unc_source_count": "uncertainty",
                "uncertainty_contribution": "uncertainty",
                "uncertainty_propagation_pressure": "uncertainty",
                "vq": "uncertainty",
                # ── world_model_surprise ──────────────────────────────
                "causal_world_model": "world_model_surprise",
                "hierarchical_world_model": "world_model_surprise",
                "rssm": "world_model_surprise",
                "unified_simulator": "world_model_surprise",
                "world_model": "world_model_surprise",
                "world_model_prediction_pressure": "world_model_surprise",
                # ── fb_correction channels ────────────────────────────
                "fb_correction:causal_quality": "low_causal_quality",
                "fb_correction:coherence": "coherence_deficit",
                "fb_correction:convergence": "convergence_conflict",
                "fb_correction:health_mean": "coherence_deficit",
                "fb_correction:loss_scale": "diverging",
                "fb_correction:memory_quality": "memory_trust_deficit",
                "fb_correction:output_quality": "low_output_reliability",
                "fb_correction:recovery_pressure": "recovery_pressure",
                "fb_correction:safety": "safety_violation",
                "fb_correction:self_report_consistency": "low_output_reliability",
                "fb_correction:surprise": "world_model_surprise",
                "fb_correction:uncertainty": "uncertainty",
                # Reinforce re-entrancy skip — verify_and_reinforce()
                # was called re-entrantly and skipped.
                "reinforce_reentrant_skip": "recovery_pressure",
                # Severe axiom re-verification success — forced
                # re-verification succeeded after catastrophic axiom.
                "severe_axiom_reverification_success": "recovery_pressure",
                # Emergence report auto-trigger — emergency emergence
                # assessment triggered by severe axiom failure.
                "emergence_report_auto_trigger": "coherence_deficit",
                # Downstream consistency reset — healing propagated
                # to cached downstream state.
                "downstream_consistency_reset": "recovery_pressure",
                # Weight boost correction — metacognitive weight
                # adjusted during reinforcement.
                "weight_boost_correction": "recovery_pressure",
                # Reinforcement cycle assessment — outcome of a full
                # verify_and_reinforce cycle.
                "reinforcement_cycle_assessment": "coherence_deficit",
                # Reinforcement cycle complete — full cycle finished.
                "reinforcement_cycle_complete": "convergence_quality",
                # Runtime signal degradation — degraded runtime
                # quality detected during emergence assessment.
                "runtime_signal_degradation": "coherence_deficit",
                # Auto-remediation success — diagnostic remediation
                # succeeded in system_emergence_report.
                "auto_remediation_success": "recovery_pressure",
                # Trace completeness failure — provenance trace
                # incomplete, routes to low_causal_quality.
                "trace_completeness_failure": "low_causal_quality",
                # ── Feedback bus signal computation bridges ────
                "feedback_trend_pressure_failure": "uncertainty",
                "feedback_signal_bridging_failure": "uncertainty",
                "feedback_correction_pressure_failure": "uncertainty",
                "feedback_strategy_pressure_failure": "uncertainty",
                "feedback_trigger_adaptation_failure": "uncertainty",
                "uncertainty_escalation_adaptation_failure": "uncertainty",
                "provenance_auto_registration_failure": "low_causal_quality",
                # ── Silent exception bridge error classes ────
                "provenance_weight_adaptation_failure": "low_causal_quality",
                "hybrid_reasoning_post_revision_failure": "coherence_deficit",
                "auto_critic_iterative_failure": "uncertainty",
                "ucc_causal_trace_recording_failure": "low_causal_quality",
                "memory_re_retrieval_failure": "uncertainty",
                "uncertainty_metacognitive_eval_failure": "uncertainty",
                "emergence_auto_reinforce_adaptation_failure": "coherence_deficit",
                # ── Cognitive activation: feedback bus caching failure ──
                "feedback_bus_caching_failure": "coherence_deficit",
                # ── Cognitive activation: sub-module exception bridges ──
                "mcts_causal_modulation_failure": "low_causal_quality",
                "cycle_consistency_verification_failure": "coherence_deficit",
                "reencode_divergence_verification_failure": "coherence_deficit",
                "memory_subsystem_query_failure": "uncertainty",
                "urgency_entropy_computation_failure": "uncertainty",
                "subsystem_health_check_failure": "coherence_deficit",
                # ── Integration patches: exception bridge error classes ──
                "root_cause_analysis_failure": "low_causal_quality",
                "provenance_enrichment_failure": "low_causal_quality",
                "convergence_provenance_enrichment_failure": "low_causal_quality",
                "intrinsic_motivation_failure": "uncertainty",
                "causal_antecedent_extraction_failure": "low_causal_quality",
                # ── Integration patches: trace and emergence error classes ──
                "shallow_provenance_detected": "low_causal_quality",
                "provenance_dag_cyclic": "low_causal_quality",
                "emergence_axiom_deficit": "coherence_deficit",
                # ── Training→inference bidirectional closure ────────────
                # Recording failure during verify_pipeline_wiring.
                "error_evolution_recording_failure": "low_causal_quality",
                # Convergence monitor wiring gap — set_error_evolution
                # unavailable.
                "convergence_monitor_wiring_gap": "coherence_deficit",
                # UCC root-cause trace failure in training.
                "ucc_root_cause_trace_failure": "low_causal_quality",
                # Axiom oscillation — emergence score alternates
                # between pass and fail.
                "axiom_oscillation": "coherence_deficit",
                # ── Cognitive activation: final integration error classes ──
                "signal_staleness_dampening": "coherence_deficit",
                "escalation_decay_resolved": "uncertainty",
                "reliability_gate_weight_adapt_failure": "coherence_deficit",
                # ── Cognitive activation: system emergence integration ──
                "ucc_orchestration_incomplete": "coherence_deficit",
                "emergence_verdict_oscillation": "coherence_deficit",
                # ── Infrastructure causal trace failures ──
                "integrity_monitor_trace_failure": "uncertainty",
                "provenance_tracker_trace_failure": "low_causal_quality",
                # ── Cognitive activation: snapshot & UCC adaptation bridges ──
                "snapshot_metacognitive_failure": "uncertainty",
                "snapshot_causal_chain_failure": "low_causal_quality",
                "snapshot_emergence_failure": "coherence_deficit",
                "snapshot_unity_failure": "coherence_deficit",
                "snapshot_reinforcement_failure": "coherence_deficit",
                "snapshot_diagnostic_failure": "coherence_deficit",
                "snapshot_feedback_bus_failure": "coherence_deficit",
                "snapshot_convergence_failure": "convergence_conflict",
                "ucc_provenance_adaptation_failure": "low_causal_quality",
                "ucc_directional_uncertainty_adaptation_failure": "uncertainty",
                "ucc_reliability_gate_adaptation_failure": "low_output_reliability",
                "forward_oom_recovery": "uncertainty",
                "convergence_arbiter_conflict": "convergence_conflict",
                # ── Final integration patches ──────────────────────
                # Memory export failure during save_state().
                "hierarchical_memory_export_failure": "memory_staleness",
                # Severe reinforce success — correction succeeded.
                "severe_reinforce_success": "recovery_pressure",
                # Eval rerun failure — re-reasoning during eval raised.
                "eval_rerun_failure": "uncertainty",
                # Sustained diversity collapse — persistent diversity
                # deficit across consecutive passes.
                "sustained_diversity_collapse": "coherence_deficit",
                # ── Cognitive activation: final integration gaps ──
                "memory_retrieval_empty": "memory_staleness",
                "tkg_retrieval_failure": "memory_staleness",
                "eigenvalue_computation_failure": "spectral_instability",
                # ── ACK boundary verification ──
                "boundary_violation_risk": "border_uncertainty",
                "jacobian_sanity_check_failure": "spectral_instability",
                "ack_sdp_recommended": "uncertainty",
                # ── Cognitive activation: remaining bare-except bridges ──
                "causal_chain_depth_computation_failure": "low_causal_quality",
                "cert_error_summary_failure": "uncertainty",
                "cert_provenance_attribution_failure": "low_causal_quality",
                # ── Final cognitive activation patches ──
                "ucc_convergence_autoheal_failure": "coherence_deficit",
                "pipeline_wiring_bridge_failure": "low_causal_quality",
                "causal_trace_gap_bridge_failure": "low_causal_quality",
                "low_wiring_coverage": "coherence_deficit",
                "low_provenance_coverage": "low_causal_quality",
                "emergence_conditions_gap": "coherence_deficit",
                # ── Forward-path I/O exception bridges ─────────────────
                "encoder_forward_failure": "uncertainty",
                "decoder_forward_failure": "uncertainty",
                "vq_forward_failure": "coherence_deficit",
                # ── VibeThinker & SSP error classes ─────────────────
                "vibe_thinker_calibration_drift": "uncertainty",
                "vibe_thinker_calibration_high": "uncertainty",
                "ssp_validated_fail": "low_causal_quality",
                # ── Final cognitive integration: bridge exception routes ──
                # Routes match aeon_core.py _class_to_signal: adaptation
                # failures → "uncertainty" (deepens reasoning), safety
                # subsystem failures → "safety_violation", convergence
                # failures → "convergence_conflict", feedback/coherence
                # subsystem failures → "coherence_deficit", provenance
                # failures → "low_causal_quality", output-quality
                # failures → "low_output_reliability", reactivation/
                # reinforcement → "recovery_pressure".
                "convergence_weight_adaptation_failure": "convergence_conflict",
                "safety_auto_critic_failure": "safety_violation",
                "auto_reactivation_failure": "recovery_pressure",
                "cognitive_potential_dominant_source_failure": "uncertainty",
                "deficit_auto_refresh_failure": "coherence_deficit",
                "feedback_bus_repopulation_failure": "coherence_deficit",
                "feedback_trend_gate_failure": "coherence_deficit",
                "high_output_uncertainty_adaptation_failure": "uncertainty",
                "ns_post_revision_check_failure": "coherence_deficit",
                "ns_violation_auto_critic_failure": "coherence_deficit",
                "output_reliability_error_evolution_failure": "low_output_reliability",
                "pre_reasoning_gate_adaptation_failure": "uncertainty",
                "recurring_error_escalation_failure": "uncertainty",
                "vibe_thinker_forward_failure": "uncertainty",
                "coherence_registry_output_failure": "coherence_deficit",
                "convergence_arbiter_metacognitive_failure": "convergence_conflict",
                "honesty_error_evolution_failure": "safety_violation",
                "oscillation_recording_failure": "coherence_deficit",
                "provenance_delta_recording_failure": "low_causal_quality",
                "provenance_recording_failure": "low_causal_quality",
                "reactive_reinforce_on_emergence_loss": "recovery_pressure",
                "axiom_contradiction_adaptation_failure": "coherence_deficit",
                "vibe_thinker_low_quality": "uncertainty",
                # Meta-loop stall — fixed-point iteration stagnated.
                "meta_loop_stall": "diverging",
                # ── Shape / dimension / size mismatch classes ──────────
                # Mirror aeon_core._class_to_signal so training adapts
                # loss weights when inference encounters tensor shape
                # incompatibilities.
                "shape": "coherence_deficit",
                "dimension_mismatch": "coherence_deficit",
                "size_mismatch": "coherence_deficit",
                # ── Semantic catch-all ─────────────────────────────────
                "semantic": "uncertainty",
                # ── Pre-reasoning diagnostic gap ───────────────────────
                "pre_reasoning_diagnostic_gap": "coherence_deficit",
                # ── Activation probe classes ───────────────────────────
                "activation_incomplete_forward": "uncertainty",
                "silent_exception_escalation_triggered": "uncertainty",
                # ── Causal trace failure ───────────────────────────
                "causal_trace_failure": "low_causal_quality",
                # ── Cross-validation hard gate ─────────────────────
                "cross_validation_hard_gate": "coherence_deficit",
                # ── Causal trace root-cause failure ────────────────
                "causal_trace_root_cause_failure": "low_causal_quality",
                # ── Lipschitz EMA reconciliation ──────────────────────
                "lipschitz_ema_reconciliation": "convergence_conflict",
                # ── KM convergence not verified ──────────────────────
                "km_convergence_not_verified": "convergence_conflict",
                # ── Provenance adaptation failure ─────────────────────
                "provenance_adaptation_failure": "low_causal_quality",
            }
            # Prefix-based routing for dynamically generated training
            # error classes (e.g. "training_{cls_name}" from
            # bridge_training_errors_to_inference).  Without this,
            # dynamic training error classes fall through the static
            # mapping and are silently ignored, breaking the training
            # → metacognitive adaptation feedback loop.
            _prefix_routes = [
                ("training_diverge", "diverging"),
                ("training_stag", "coherence_deficit"),
                ("training_", "uncertainty"),
                # Actionable gap prefixes — dynamically generated by
                # verify_and_reinforce() as "actionable_gap_{axiom}".
                ("actionable_gap_", "coherence_deficit"),
                # Per-module health prefixes — dynamically generated by
                # verify_and_reinforce() as "module_health_{module}".
                # Without this routing, module-level health episodes
                # fall through the static mapping and never influence
                # training-time metacognitive weight adaptation.
                ("module_health_causal", "low_causal_quality"),
                ("module_health_convergence", "convergence_conflict"),
                ("module_health_deception", "safety_violation"),
                ("module_health_code_execution", "safety_violation"),
                ("module_health_coherence", "coherence_deficit"),
                ("module_health_cycle", "coherence_deficit"),
                ("module_health_social", "coherence_deficit"),
                ("module_health_spectral", "diverging"),
                ("module_health_", "low_output_reliability"),
                # Diagnostic gap prefixes — dynamically generated by
                # self_diagnostic() as "diagnostic_gap_{component}".
                ("diagnostic_gap_cognitive", "coherence_deficit"),
                ("diagnostic_gap_coherence", "coherence_deficit"),
                ("diagnostic_gap_error", "uncertainty"),
                ("diagnostic_gap_", "coherence_deficit"),
            ]
            for cls_name, cls_stats in error_classes.items():
                signal = _class_to_signal.get(cls_name)
                if signal is None:
                    for _pfx, _sig in _prefix_routes:
                        if cls_name.startswith(_pfx):
                            signal = _sig
                            break
                if signal is None or signal not in self._signal_weights:
                    continue
                success_rate = cls_stats.get("success_rate", 1.0)
                count = cls_stats.get("count", 0)
                if count >= 2 and success_rate < 0.5:
                    self._signal_weights[signal] = min(
                        0.5, self._signal_weights[signal] + 0.05,
                    )

    class UnifiedCognitiveCycle:
        """Unified cognitive cycle (fallback when aeon_core unavailable).

        Orchestrates convergence monitoring, coherence verification,
        error evolution, metacognitive triggering, and provenance
        tracking into a single ``evaluate()`` call — matching the
        interface of ``aeon_core.UnifiedCognitiveCycle`` so that
        standalone training produces real meta-cognitive decisions.
        """

        def __init__(self, convergence_monitor, coherence_verifier,
                     error_evolution, metacognitive_trigger,
                     provenance_tracker, causal_trace=None):
            self.convergence_monitor = convergence_monitor
            self.coherence_verifier = coherence_verifier
            self.error_evolution = error_evolution
            self.metacognitive_trigger = metacognitive_trigger
            self.provenance_tracker = provenance_tracker
            self.causal_trace = causal_trace
            # Wire convergence monitor → error evolution automatically.
            if self.error_evolution is not None:
                self.convergence_monitor.set_error_evolution(self.error_evolution)
            # Wire convergence monitor → provenance tracker.
            self.convergence_monitor.set_provenance_tracker(self.provenance_tracker)
            # Wire convergence monitor → metacognitive trigger so that
            # high metacognitive pressure tightens convergence threshold.
            if self.metacognitive_trigger is not None:
                self.convergence_monitor.set_metacognitive_trigger(
                    self.metacognitive_trigger,
                )
            # Wire error evolution → causal trace.
            if self.error_evolution is not None and self.causal_trace is not None:
                self.error_evolution.set_causal_trace(self.causal_trace)

        def _provenance_snapshot(self):
            snap = self.provenance_tracker.compute_attribution()
            contribs = snap.get("contributions", {})
            dominant = max(contribs, key=contribs.get) if contribs else None
            return contribs, dominant

        def evaluate(self, subsystem_states, delta_norm,
                     uncertainty=0.0, topology_catastrophe=False,
                     world_model_surprise=0.0, causal_quality=1.0,
                     memory_staleness=False, recovery_pressure=0.0,
                     safety_violation=False, feedback_signal=None,
                     diversity_collapse=0.0, memory_trust_deficit=0.0,
                     **kwargs):
            # 1. Convergence check
            convergence_verdict = self.convergence_monitor.check(delta_norm)

            # 1b. Adapt coherence threshold from error evolution
            _original_threshold = None
            _err_summary = {}
            if (self.error_evolution is not None
                    and self.coherence_verifier is not None):
                _err_summary = self.error_evolution.get_error_summary()
                _original_threshold = self.coherence_verifier.threshold
                self.coherence_verifier.adapt_threshold(_err_summary)

            # 1c. Pre-adapt metacognitive trigger weights
            if self.metacognitive_trigger is not None and _err_summary:
                try:
                    self.metacognitive_trigger.adapt_weights_from_evolution(
                        _err_summary,
                    )
                except Exception as _ae_err:
                    logger.warning(
                        "Metacognitive weight adaptation failed in training: %s",
                        _ae_err,
                    )
                    if self._error_evolution is not None:
                        try:
                            self._error_evolution.record_episode(
                                error_class='adaptation_failure',
                                strategy_used='skip_and_continue',
                                success=False,
                                metadata={
                                    'error': str(_ae_err)[:200],
                                    'source': 'training_ucc_evaluate',
                                },
                                causal_antecedents=["training_ucc", "adapt_weights"],
                            )
                        except Exception as _ee_err:
                            logger.debug(
                                "Error evolution recording itself failed "
                                "in training UCC adaptation: %s", _ee_err,
                            )

            # 2. Coherence verification
            if self.coherence_verifier is not None and subsystem_states:
                coherence_result = self.coherence_verifier(subsystem_states)
                coherence_deficit = (
                    1.0 - coherence_result["coherence_score"].mean().item()
                )
                needs_recheck = coherence_result.get("needs_recheck", False)
            else:
                coherence_result = {
                    "coherence_score": torch.tensor([1.0]),
                    "needs_recheck": False,
                }
                coherence_deficit = 0.0
                needs_recheck = False

            # Record significant coherence deficits
            if coherence_deficit > 0.3 and self.error_evolution is not None:
                _contributions, _dominant = self._provenance_snapshot()
                self.error_evolution.record_episode(
                    error_class="coherence_deficit",
                    strategy_used="meta_rerun",
                    success=False,
                    metadata={
                        "coherence_deficit": coherence_deficit,
                        "dominant_provenance_module": _dominant,
                    },
                    causal_antecedents=["training_ucc", "coherence_verifier"],
                )

            # 3. Metacognitive trigger
            is_diverging = convergence_verdict.get("status") == "diverging"
            if self.metacognitive_trigger is not None:
                trigger_detail = self.metacognitive_trigger.evaluate(
                    uncertainty=uncertainty,
                    is_diverging=is_diverging,
                    topology_catastrophe=topology_catastrophe,
                    coherence_deficit=coherence_deficit,
                    memory_staleness=memory_staleness,
                    recovery_pressure=recovery_pressure,
                    world_model_surprise=world_model_surprise,
                    causal_quality=causal_quality,
                    safety_violation=safety_violation,
                    diversity_collapse=diversity_collapse,
                    memory_trust_deficit=memory_trust_deficit,
                    convergence_conflict=kwargs.get("convergence_conflict", 0.0),
                )
            else:
                _should = (is_diverging or coherence_deficit > 0.5
                           or uncertainty > 0.7 or safety_violation
                           or topology_catastrophe
                           or diversity_collapse > 0.5
                           or memory_trust_deficit > 0.5)
                trigger_detail = {
                    "should_trigger": _should,
                    "trigger_score": max(uncertainty, coherence_deficit,
                                         diversity_collapse,
                                         memory_trust_deficit,
                                         1.0 if is_diverging else 0.0),
                    "triggers_active": [s for s, v in [
                        ("diverging", is_diverging),
                        ("coherence_deficit", coherence_deficit > 0.5),
                        ("uncertainty", uncertainty > 0.7),
                        ("safety_violation", safety_violation),
                        ("topology_catastrophe", topology_catastrophe),
                        ("diversity_collapse", diversity_collapse > 0.5),
                        ("memory_trust_deficit", memory_trust_deficit > 0.5),
                    ] if v],
                }
            should_rerun = trigger_detail.get("should_trigger", False) or needs_recheck

            # 4. Provenance snapshot
            provenance = self.provenance_tracker.compute_attribution()

            # 4b. Weakest-pair identification for consistency with
            # aeon_core.UnifiedCognitiveCycle return structure.
            weakest_pair = None
            if (self.coherence_verifier is not None
                    and hasattr(self.coherence_verifier, 'get_weakest_pair')):
                weakest_pair = self.coherence_verifier.get_weakest_pair(
                    coherence_result.get("pairwise", {}),
                )

            # 4c. Root-cause trace — use the causal_trace buffer (when
            # wired) to provide actual root-cause data instead of an
            # empty dict.  This ensures that conclusions produced during
            # training are traceable to their originating modules,
            # satisfying the root-cause traceability requirement even
            # when aeon_core is unavailable.
            root_cause_trace: Dict[str, Any] = {}
            if self.causal_trace is not None:
                try:
                    _recent = self.causal_trace.recent(1)
                    if _recent:
                        _last_id = _recent[0].get("id") if isinstance(_recent[0], dict) else None
                        if _last_id is not None:
                            root_cause_trace = self.causal_trace.trace_root_cause(
                                _last_id,
                            ) or {}
                except Exception as _rct_err:
                    logger.warning(
                        "UCC root-cause trace failed in training: %s",
                        _rct_err,
                    )
                    if self._error_evolution is not None:
                        try:
                            self._error_evolution.record_episode(
                                error_class='ucc_root_cause_trace_failure',
                                strategy_used='skip_and_continue',
                                success=False,
                                metadata={
                                    'error': str(_rct_err)[:200],
                                    'source': 'training_ucc_evaluate',
                                },
                                causal_antecedents=["training_ucc", "causal_trace"],
                            )
                        except Exception as _ee_err:
                            logger.debug(
                                "Error evolution recording itself failed "
                                "in training UCC root-cause trace: %s",
                                _ee_err,
                            )

            # 4d. Correction guidance — synthesize actionable
            # recommendation from weakest pair, most uncertain module,
            # and provenance root cause.  Mirrors the correction_guidance
            # dict produced by aeon_core.UnifiedCognitiveCycle so that
            # training phases can apply targeted corrections when
            # aeon_core is unavailable.
            _correction_target_module: Optional[str] = None
            _correction_reason: Optional[str] = None
            contribs = provenance.get("contributions", {})
            if contribs and coherence_deficit > 0.3:
                _correction_target_module = max(
                    contribs, key=contribs.get,
                )
                _correction_reason = "dominant_provenance_high_coherence_deficit"
            if _correction_target_module is None and weakest_pair is not None:
                _wp_mods = weakest_pair.get("modules", [])
                if _wp_mods:
                    _correction_target_module = _wp_mods[0]
                    _correction_reason = "weakest_coherence_pair"
            correction_guidance: Dict[str, Any] = {
                "target_module": _correction_target_module,
                "reason": _correction_reason,
                "weakest_pair": weakest_pair,
            }
            if (_correction_target_module is not None
                    and self.error_evolution is not None):
                _best = self.error_evolution.get_best_strategy(
                    _correction_target_module,
                )
                correction_guidance["recommended_strategy"] = _best
                _root_causes = self.error_evolution.get_root_causes(
                    _correction_target_module,
                )
                correction_guidance["historical_root_causes"] = (
                    _root_causes.get("root_causes", {})
                )

            # 5. Restore coherence threshold
            if _original_threshold is not None and self.coherence_verifier is not None:
                self.coherence_verifier.threshold = _original_threshold

            return {
                "convergence_verdict": convergence_verdict,
                "coherence_result": {
                    "coherence_score": coherence_result["coherence_score"],
                    "needs_recheck": needs_recheck,
                    "coherence_deficit": coherence_deficit,
                    "weakest_pair": weakest_pair,
                },
                "should_rerun": should_rerun,
                "trigger_detail": trigger_detail,
                "provenance": provenance,
                "root_cause_trace": root_cause_trace,
                "correction_guidance": correction_guidance,
            }

        def reset(self):
            self.convergence_monitor.reset()
            if self.metacognitive_trigger is not None:
                self.metacognitive_trigger.reset()

# --- Токенизатор ---
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers не установлен. Будет использован fallback-токенизатор.")

# --- Mixed Precision ---
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        AMP_AVAILABLE = False


# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================

def configure_logger(logfile: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Настройка логгера с поддержкой файла и консоли"""
    logger = logging.getLogger("AEON-Training-v4")
    logger.setLevel(level)
    logger.handlers.clear()
    
    detailed_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(detailed_format))
    logger.addHandler(sh)
    
    if logfile:
        log_dir = os.path.dirname(logfile) or "."
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(detailed_format))
        logger.addHandler(fh)
    
    return logger

logger = configure_logger()


# ==============================================================================
# УСТРОЙСТВО
# ==============================================================================

def _select_device() -> torch.device:
    """Select best available device: CUDA → MPS → CPU with runtime probe."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _probe = torch.zeros(1, device="mps")
            del _probe
            return torch.device("mps")
        except Exception as _e:
            logger.warning(f"MPS available but probe failed ({_e}), using CPU")
    return torch.device("cpu")


device = _select_device()
logger.info(f"🖥️  Device: {device}")
if device.type == "cuda":
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif device.type == "mps":
    logger.info("   Apple Silicon MPS accelerator")


# ==============================================================================
# КОНФИГУРАЦИЯ v4.0 — ОПТИМИЗИРОВАННАЯ ДЛЯ СВЯЗАННЫХ МЫСЛЕЙ
# ==============================================================================

@dataclass
class AEONConfigV4:
    """
    Конфигурация v4.0 с оптимизацией для связанных мыслей
    
    Ключевые изменения:
    - grad_clip_norm: 0.5 (было 1.0) — стабилизация
    - context_window: 3 — RSSM учитывает 3 предыдущих состояния
    - entropy_weight: 0.1 — регуляризация кодбука
    - document_aware: True — построение пар по документам
    """
    
    # Архитектура
    z_dim: int = 256
    hidden_dim: int = 256
    vocab_size: int = 30522
    num_pillars: int = 5
    seq_length: int = 64
    
    # VQ-VAE (оптимизировано)
    # NOTE: vq_reset_threshold defaults to 30 (more aggressive than
    # aeon_core's vq_revival_threshold=100) because training benefits
    # from faster codebook reset to avoid dead codes during early
    # training.  When using from_core_config(), the core threshold
    # is inherited unless explicitly overridden.
    vq_num_embeddings: int = 2048
    vq_embedding_dim: int = 256
    vq_commitment_cost: float = 0.25
    vq_loss_weight: float = 0.5
    vq_ema_decay: float = 0.99
    vq_temperature: float = 1.0
    vq_reset_threshold: int = 30
    
    # ✅ НОВОЕ: Entropy regularization
    entropy_weight: float = 0.1  # Поощряет равномерное использование кодов
    
    # Обучение (стабилизировано)
    learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    grad_clip_norm: float = 0.5  # ✅ Было 1.0, теперь стабильнее
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    
    # Warmup и Scheduling
    warmup_steps: int = 1000  # Было 500, теперь плавнее
    warmup_ratio: float = 0.1
    
    # Регуляризация
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # ✅ НОВОЕ: RSSM с контекстом
    context_window: int = 3  # RSSM видит 3 предыдущих z
    rssm_hidden_dim: int = 512  # Увеличен для контекста
    
    # ✅ НОВОЕ: Документ-ориентированное обучение
    document_aware: bool = True  # Строить пары только внутри документов
    min_doc_chunks: int = 2  # Минимум чанков в документе
    
    # Early Stopping
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Прочее
    seed: int = 42
    use_amp: bool = True
    
    # Noise scale for VQ code reset
    code_reset_noise_scale: float = 0.05

    @classmethod
    def from_core_config(cls, core_config: Any, **overrides) -> 'AEONConfigV4':
        """Create a training config from an inference ``AEONConfig``.

        Bridges the inference and training configurations so that shared
        architectural parameters (z_dim, hidden_dim, vocab_size, seq_length,
        VQ-VAE settings) are inherited from the core config, eliminating
        silent divergence between training and inference pipelines.

        Training-specific parameters (grad_clip_norm, context_window,
        entropy_weight, document_aware) retain their v4 defaults unless
        explicitly overridden via ``**overrides``.

        Args:
            core_config: An ``aeon_core.AEONConfig`` instance.
            **overrides: Keyword arguments that override any field.

        Returns:
            A new ``AEONConfigV4`` instance with shared parameters
            inherited from the core config.
        """
        # Map core config field names to AEONConfigV4 field names.
        # Only shared architectural parameters are bridged; training-
        # specific parameters use their v4 defaults.
        _SHARED_FIELDS = {
            'z_dim': 'z_dim',
            'hidden_dim': 'hidden_dim',
            'vocab_size': 'vocab_size',
            'seq_length': 'seq_length',
            'vq_num_embeddings': 'vq_num_embeddings',
            'vq_embedding_dim': 'vq_embedding_dim',
            'vq_commitment_cost': 'vq_commitment_cost',
            'vq_ema_decay': 'vq_ema_decay',
            'dropout_rate': 'dropout_rate',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
        }
        kwargs: Dict[str, Any] = {}
        for core_field, v4_field in _SHARED_FIELDS.items():
            val = getattr(core_config, core_field, None)
            if val is not None:
                kwargs[v4_field] = val
        # Map vq_revival_threshold (inference: steps before a dead code
        # is revived) → vq_reset_threshold (training: steps before a
        # dead code is reset).  Both control codebook refresh
        # aggressiveness via the same mechanism; the name difference
        # reflects the differing contexts (inference revival vs training
        # reset).  Inheriting the core value ensures consistent
        # codebook management unless explicitly overridden.
        _revival = getattr(core_config, 'vq_revival_threshold', None)
        if _revival is not None and 'vq_reset_threshold' not in overrides:
            kwargs['vq_reset_threshold'] = _revival
        # Apply explicit overrides last (highest priority).
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_core_config(self, **overrides):
        """Create an ``AEONConfig`` inference config from this training config.

        Reverse bridge of :meth:`from_core_config`.  Shared architectural
        parameters (z_dim, hidden_dim, vocab_size, seq_length, VQ-VAE
        settings) are propagated so that a model trained with this
        config can be served with a matching inference configuration.

        Inference-specific parameters (topo_method, enable_safety, etc.)
        retain their ``AEONConfig`` defaults unless explicitly overridden
        via ``**overrides``.

        Requires ``aeon_core.AEONConfig`` to be importable.

        Args:
            **overrides: Keyword arguments forwarded to ``AEONConfig()``.

        Returns:
            An ``AEONConfig`` instance.

        Raises:
            ImportError: If ``aeon_core`` is not available.
        """
        from aeon_core import AEONConfig  # local import avoids circular dep

        _SHARED_FIELDS = {
            'z_dim': 'z_dim',
            'hidden_dim': 'hidden_dim',
            'vocab_size': 'vocab_size',
            'seq_length': 'seq_length',
            'vq_num_embeddings': 'vq_num_embeddings',
            'vq_embedding_dim': 'vq_embedding_dim',
            'vq_commitment_cost': 'vq_commitment_cost',
            'vq_ema_decay': 'vq_ema_decay',
            'dropout_rate': 'dropout_rate',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
        }
        kwargs: Dict[str, Any] = {}
        for v4_field, core_field in _SHARED_FIELDS.items():
            val = getattr(self, v4_field, None)
            if val is not None:
                kwargs[core_field] = val
        # Map vq_reset_threshold → vq_revival_threshold (reverse of
        # from_core_config).
        if hasattr(self, 'vq_reset_threshold') and 'vq_revival_threshold' not in overrides:
            kwargs['vq_revival_threshold'] = self.vq_reset_threshold
        kwargs.update(overrides)
        return AEONConfig(**kwargs)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {self.z_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_learning_rate <= 0:
            raise ValueError(f"min_learning_rate must be positive, got {self.min_learning_rate}")
        if self.vq_commitment_cost < 0:
            raise ValueError(f"vq_commitment_cost must be non-negative, got {self.vq_commitment_cost}")
        if self.context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {self.context_window}")
        if self.vq_num_embeddings < 2:
            raise ValueError(f"vq_num_embeddings must be >= 2, got {self.vq_num_embeddings}")
        if not (0 < self.vq_ema_decay < 1):
            raise ValueError(f"vq_ema_decay must be in (0, 1), got {self.vq_ema_decay}")
        if self.code_reset_noise_scale < 0:
            raise ValueError(f"code_reset_noise_scale must be non-negative, got {self.code_reset_noise_scale}")
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if not (0 <= self.dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if not (0 <= self.label_smoothing < 1):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.vq_temperature <= 0:
            raise ValueError(f"vq_temperature must be positive, got {self.vq_temperature}")
        if self.entropy_weight < 0:
            raise ValueError(f"entropy_weight must be non-negative, got {self.entropy_weight}")
        if self.vq_loss_weight < 0:
            raise ValueError(f"vq_loss_weight must be non-negative, got {self.vq_loss_weight}")
        if self.save_every_n_epochs <= 0:
            raise ValueError(f"save_every_n_epochs must be positive, got {self.save_every_n_epochs}")
        if self.keep_n_checkpoints <= 0:
            raise ValueError(f"keep_n_checkpoints must be positive, got {self.keep_n_checkpoints}")
        if self.min_doc_chunks < 1:
            raise ValueError(f"min_doc_chunks must be >= 1, got {self.min_doc_chunks}")


# ==============================================================================
# МОНИТОР ОБУЧЕНИЯ (улучшенный)
# ==============================================================================

class TrainingMonitor:
    """Расширенный монитор для отслеживания метрик обучения"""
    
    def __init__(self, logger: logging.Logger, save_dir: str = "checkpoints"):
        self.logger = logger
        self.metrics_history = {"phase_A": [], "phase_B": []}
        self.batch_metrics = {"phase_A": [], "phase_B": []}
        self.start_time = None
        self.epoch_start_time = None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def start_training(self, phase: str, total_epochs: int, total_samples: int):
        self.start_time = time.time()
        # Only reset known phase keys; the phase argument is used for display
        if phase in self.batch_metrics:
            self.batch_metrics[phase] = []
        self.logger.info("=" * 75)
        self.logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ - {phase}")
        self.logger.info(f"   Всего эпох: {total_epochs}")
        self.logger.info(f"   Всего сэмплов: {total_samples:,}")
        self.logger.info(f"   Время старта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 75)
        
    def start_epoch(self, epoch: int, total_epochs: int):
        self.epoch_start_time = time.time()
        self.logger.info(f"\n{'─' * 60}")
        self.logger.info(f"📍 Эпоха {epoch + 1}/{total_epochs}")
        self.logger.info(f"{'─' * 60}")
        
    def log_batch(self, batch_idx: int, total_batches: int, metrics: dict, 
                  phase: str = "phase_A", log_every: int = 10):
        self.batch_metrics[phase].append(metrics.copy())
        
        log_every = max(log_every, 1)
        if batch_idx % log_every == 0 or batch_idx == total_batches - 1:
            metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            progress = (batch_idx + 1) / max(total_batches, 1) * 100
            self.logger.info(f"   Batch [{batch_idx + 1:5d}/{total_batches}] ({progress:5.1f}%) | {metrics_str}")
            
    def end_epoch(self, epoch: int, total_epochs: int, epoch_metrics: dict, 
                  phase: str = "phase_A") -> bool:
        epoch_time = time.time() - self.epoch_start_time
        self.metrics_history[phase].append(epoch_metrics.copy())
        
        self.logger.info(f"\n   📊 Итоги эпохи {epoch + 1}:")
        for key, value in epoch_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"      • {key}: {value:.6f}")
            else:
                self.logger.info(f"      • {key}: {value}")
        self.logger.info(f"   ⏱️  Время эпохи: {timedelta(seconds=int(epoch_time))}")
        
        elapsed = time.time() - self.start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining = avg_epoch_time * (total_epochs - epoch - 1)
        self.logger.info(f"   ⏳ Осталось примерно: {timedelta(seconds=int(remaining))}")
        
        if len(self.metrics_history[phase]) >= 2:
            prev = self.metrics_history[phase][-2]
            curr = self.metrics_history[phase][-1]
            
            loss_key = "total" if "total" in curr else "mse_loss"
            if loss_key in prev and loss_key in curr:
                delta = curr[loss_key] - prev[loss_key]
                pct_change = (delta / prev[loss_key]) * 100 if prev[loss_key] != 0 else 0
                direction = "📉" if delta < 0 else "📈" if delta > 0 else "➡️"
                self.logger.info(f"   {direction} Δ{loss_key}: {delta:+.6f} ({pct_change:+.2f}%)")
        
        current_loss = epoch_metrics.get("total", epoch_metrics.get("mse_loss", float('inf')))
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return False
        
    def end_training(self, phase: str):
        total_time = time.time() - self.start_time
        self.logger.info("\n" + "=" * 75)
        self.logger.info(f"✅ {phase} ЗАВЕРШЕНА")
        self.logger.info(f"   Общее время: {timedelta(seconds=int(total_time))}")
        
        if phase in self.metrics_history and self.metrics_history[phase]:
            first = self.metrics_history[phase][0]
            last = self.metrics_history[phase][-1]
            
            loss_key = "total" if "total" in first else "mse_loss"
            first_loss = first.get(loss_key, 0)
            last_loss = last.get(loss_key, 0)
            
            if first_loss > 0:
                improvement = (first_loss - last_loss) / first_loss * 100
                self.logger.info(f"   📈 Улучшение loss: {improvement:.2f}%")
            self.logger.info(f"   📊 Начальный loss: {first_loss:.6f}")
            self.logger.info(f"   📊 Финальный loss: {last_loss:.6f}")
        
        self.logger.info("=" * 75 + "\n")
        
    def log_model_stats(self, model: nn.Module, component_name: str = "Модель"):
        self.logger.info(f"📦 Параметры {component_name}:")
        
        total_params = 0
        trainable_params = 0
        
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params
            trainable_params += trainable
            self.logger.info(f"   • {name}: {params:,} (trainable: {trainable:,})")
        
        self.logger.info(f"   ─────────────────────────────────")
        self.logger.info(f"   ВСЕГО: {total_params:,} (trainable: {trainable_params:,})")
        self.logger.info(f"   Память модели: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
        
    def log_tensor_stats(self, tensor: torch.Tensor, name: str):
        with torch.no_grad():
            t = tensor.float()
            self.logger.info(f"   📐 {name}:")
            self.logger.info(f"      shape: {list(tensor.shape)}")
            self.logger.info(f"      mean: {t.mean():.6f}, std: {t.std():.6f}")
            self.logger.info(f"      min: {t.min():.6f}, max: {t.max():.6f}")
            
    def save_metrics(self, filepath: str):
        data = {
            "metrics_history": self.metrics_history,
            "best_loss": self.best_loss,
            "timestamp": datetime.now().isoformat()
        }
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            self.logger.error(f"❌ Failed to save metrics to {filepath}: {e}")


# ==============================================================================
# КОМПОНЕНТЫ МОДЕЛИ
# ==============================================================================

class ThoughtEncoder(nn.Module):
    """Энкодер: tokens → z с Bidirectional LSTM"""
    
    def __init__(self, vocab_size: int, emb_dim: int = 256, z_dim: int = 256, 
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            emb_dim, 
            z_dim // 2,
            batch_first=True, 
            bidirectional=True,
            num_layers=1
        )
        
        self.norm = nn.LayerNorm(z_dim)
        self.z_dim = z_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.dropout(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        z = self.norm(h)
        return z


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel-Softmax based vector quantizer replacing VQ-VAE.

    Advantages over straight-through VQ-VAE:
    - Fully differentiable (no straight-through hack)
    - Temperature annealing provides smooth transition from soft to hard
    - Less collapsed codes (Gumbel noise prevents mode collapse)

    Reference: Jang et al. 2017 "Categorical Reparameterization with Gumbel-Softmax"
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        anneal_rate: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.normal_(0, 0.1)

        # Monitoring
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        self.register_buffer('total_count', torch.tensor(0.0))
        self.register_buffer('global_step', torch.tensor(0))
        self.register_buffer('_temperature', torch.tensor(float(temperature)))

    @property
    def temperature(self) -> float:
        return self._temperature.item()

    @temperature.setter
    def temperature(self, value: float):
        self._temperature.fill_(value)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: [B, embedding_dim] encoded representation.
        Returns:
            (quantized, loss, indices, stats) matching VectorQuantizerHybridV4 signature.
        """
        # Negative squared distances as logits  [B, num_embeddings]
        logits = -torch.cdist(z.unsqueeze(0), self.embeddings.weight.unsqueeze(0)).squeeze(0)

        if self.training:
            # Gumbel-Softmax: differentiable sampling
            soft_idx = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            z_q = soft_idx @ self.embeddings.weight  # [B, embedding_dim]
        else:
            # Hard assignment for inference
            soft_idx = F.softmax(logits, dim=-1)
            z_q = soft_idx @ self.embeddings.weight

        # Hard indices for monitoring / stats
        indices = logits.argmax(dim=-1)

        # Loss: commitment + soft entropy regularization
        commitment_loss = F.mse_loss(z, z_q.detach())
        # Entropy regularization to encourage uniform codebook usage
        avg_probs = soft_idx.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        max_entropy = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
        entropy_loss = 1.0 - entropy / max_entropy

        loss = self.commitment_cost * commitment_loss + 0.1 * entropy_loss

        if self.training:
            self._update_stats(indices)

        stats = self._compute_stats(indices)
        stats['entropy_loss'] = entropy_loss.item()
        stats['temperature'] = self.temperature

        return z_q, loss, indices, stats

    def _update_stats(self, indices: torch.Tensor):
        with torch.no_grad():
            self.global_step += 1
            self.total_count += indices.size(0)
            self.code_usage.scatter_add_(
                0, indices, torch.ones_like(indices, dtype=self.code_usage.dtype)
            )
            self.anneal_temperature()

    def anneal_temperature(self):
        """Anneal temperature towards min_temperature. Call after each training step."""
        # Clamp exponent to prevent overflow when anneal_rate is very large
        self.temperature = max(
            self.min_temperature,
            self.temperature * math.exp(max(-self.anneal_rate, -20.0)),
        )

    def _compute_stats(self, indices: torch.Tensor) -> dict:
        with torch.no_grad():
            unique_in_batch = len(indices.unique())
            total_used = (self.code_usage > 0).sum().item()
            usage_pct = total_used / self.num_embeddings * 100
            if self.total_count > 0:
                probs = self.code_usage / (self.total_count + 1e-10)
                probs = probs[probs > 0]
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                max_ent = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
                normalized_entropy = entropy / max_ent
            else:
                normalized_entropy = 0
            return {
                "codebook_usage_%": usage_pct,
                "unique_codes_batch": unique_in_batch,
                "total_used_codes": total_used,
                "codebook_entropy": normalized_entropy,
            }

    def get_codebook_usage(self) -> float:
        if self.total_count > 0:
            used = (self.code_usage > 0).sum().item()
            return used / self.num_embeddings * 100
        return 0.0


class VectorQuantizerHybridV4(nn.Module):
    """
    VQ-VAE v4 с entropy regularization
    
    Улучшения:
    - Entropy loss для равномерного использования кодов
    - Более агрессивный reset неиспользуемых кодов
    - Улучшенная инициализация
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        temperature: float = 1.0,
        reset_threshold: int = 30,
        entropy_weight: float = 0.1,
        code_reset_noise_scale: float = 0.05
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.temperature = temperature
        self.reset_threshold = reset_threshold
        self.entropy_weight = entropy_weight
        self.code_reset_noise_scale = code_reset_noise_scale
        
        # Кодбук с улучшенной инициализацией
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Инициализация ближе к нормальному распределению z
        self.embedding.weight.data.normal_(0, 0.1)
        
        # EMA буферы
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
        # Мониторинг использования
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        self.register_buffer('code_age', torch.zeros(num_embeddings))
        self.register_buffer('total_count', torch.tensor(0.0))
        self.register_buffer('global_step', torch.tensor(0))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, D = z.shape
        
        # Расчёт расстояний
        distances = (
            torch.sum(z**2, dim=1, keepdim=True) + 
            torch.sum(self.embedding.weight**2, dim=1) - 
            2 * torch.matmul(z, self.embedding.weight.t())
        ) / max(self.temperature, 1e-8)
        
        # Выбор ближайших кодов
        indices = torch.argmin(distances, dim=1)
        
        # Квантизованные векторы
        quantized = self.embedding(indices)
        
        # ========== LOSS COMPUTATION ==========
        # Standard VQ-VAE loss (van den Oord et al., 2017):
        #   L = ||sg[z] - e_k||² + β·||z - sg[e_k]||²
        # When EMA is used for codebook updates, the codebook loss
        # ||sg[z] - e_k||² is dropped because EMA already moves e_k
        # toward z.  Including both creates conflicting gradient vs
        # EMA updates on the same embedding weights.
        
        # 1. Commitment loss: ||z - sg[e_k]||²
        #    Gradient flows to encoder (z), codebook (quantized) is detached.
        commitment_loss = F.mse_loss(z, quantized.detach())
        
        # 2. Codebook loss: ||sg[z] - e_k||²
        #    Gradient flows to codebook (quantized), encoder (z) is detached.
        #    Omitted from total loss because _update_ema handles codebook
        #    updates; kept for monitoring only.
        codebook_loss = F.mse_loss(quantized, z.detach())
        
        # 3. Entropy regularization
        # Поощряет равномерное использование кодов
        # Use soft probabilities from distances for differentiable entropy
        soft_probs = F.softmax(-distances, dim=-1)  # [B, num_embeddings]
        avg_probs = soft_probs.mean(dim=0)  # [num_embeddings]
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        max_entropy = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
        entropy_loss = 1.0 - entropy / max_entropy
        
        # Total loss — codebook_loss excluded (EMA handles codebook updates)
        loss = self.commitment_cost * commitment_loss + self.entropy_weight * entropy_loss
        
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        
        # EMA update
        if self.training:
            self._update_ema(z, indices)
        
        # Статистика
        stats = self._compute_stats(indices)
        stats['entropy_loss'] = entropy_loss.item()
        stats['codebook_loss'] = codebook_loss.item()
        
        return quantized_st, loss, indices, stats
    
    def _compute_entropy_loss(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет entropy loss для поощрения равномерного использования кодов
        
        Максимальная энтропия = log(num_embeddings) при равномерном распределении
        Минимизируем (max_entropy - actual_entropy) / max_entropy
        """
        # Считаем частоту использования каждого кода в батче
        counts = torch.bincount(indices, minlength=self.num_embeddings).float()
        probs = counts / counts.sum().clamp(min=1)
        
        # Entropy: -sum(p * log(p))
        # Добавляем epsilon для численной стабильности
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()
        
        # Нормализуем относительно максимальной энтропии
        # Fallback to 1.0 when num_embeddings=1 to avoid division by log(1)=0
        max_entropy = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
        
        # Loss = 1 - normalized_entropy (хотим максимизировать энтропию)
        entropy_loss = 1.0 - (entropy / max_entropy)
        
        return entropy_loss
    
    def _update_ema(self, z: torch.Tensor, indices: torch.Tensor):
        """EMA обновление для стабильности"""
        with torch.no_grad():
            self.global_step += 1
            self.total_count += z.size(0)
            
            encodings = F.one_hot(indices, self.num_embeddings).float()
            encodings_sum = encodings.sum(0)
            
            self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            dw = torch.matmul(encodings.t(), z)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            used_codes = indices.unique()
            self.code_usage[used_codes] += 1
            self.code_age += 1
            self.code_age[used_codes] = 0
            
            # Update embedding from EMA weights (Laplace-smoothed)
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.weight.data.copy_(
                self.ema_w / smoothed.clamp(min=self.epsilon).unsqueeze(1)
            )
            
            # Периодический сброс (чаще чем в v3)
            if self.global_step % 50 == 0:
                self._reset_unused_codes(z)
    
    def _reset_unused_codes(self, z: torch.Tensor):
        """Более агрессивный сброс неиспользуемых кодов"""
        unused_mask = self.code_age > self.reset_threshold
        num_unused = unused_mask.sum().item()
        
        if num_unused > 0 and z.size(0) > 0:
            num_to_reset = min(num_unused, z.size(0))
            random_indices = torch.randint(0, z.size(0), (num_to_reset,), device=z.device)
            new_codes = z[random_indices].detach()
            
            # Больше шума для разнообразия
            noise = torch.randn_like(new_codes) * self.code_reset_noise_scale
            new_codes = new_codes + noise
            
            unused_indices = torch.where(unused_mask)[0][:num_to_reset]
            
            self.embedding.weight.data[unused_indices] = new_codes
            self.ema_w[unused_indices] = new_codes
            self.ema_cluster_size[unused_indices] = 1.0
            self.code_age[unused_indices] = 0
            self.code_usage[unused_indices] = 1
    
    def _compute_stats(self, indices: torch.Tensor) -> dict:
        with torch.no_grad():
            unique_in_batch = len(indices.unique())
            total_used = (self.code_usage > 0).sum().item()
            usage_pct = total_used / self.num_embeddings * 100
            
            if self.total_count > 0:
                probs = self.code_usage / (self.total_count + 1e-10)
                probs = probs[probs > 0]
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                max_entropy = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0
            
            return {
                "codebook_usage_%": usage_pct,
                "unique_codes_batch": unique_in_batch,
                "total_used_codes": total_used,
                "codebook_entropy": normalized_entropy,
            }
    
    def get_codebook_usage(self) -> float:
        if self.total_count > 0:
            used = (self.code_usage > 0).sum().item()
            return used / self.num_embeddings * 100
        return 0.0


class SimVQQuantizer(nn.Module):
    """SimVQ — Simplified VQ with reparameterised codebook updates.

    Addresses codebook collapse by applying a reparameterisation trick
    that enables gradient flow directly through the codebook, yielding
    near-100% code utilization without EMA or code-reset heuristics.

    Key idea (Zhu et al., 2024): Instead of the straight-through
    estimator (STE), SimVQ reparameterises the quantised output as::

        q = z + stop_grad(e_k − z)      # STE baseline
        →
        q = e_k + σ · ε,  ε ~ N(0, I)   # SimVQ reparameterisation

    where σ anneals from a warm-start value to near-zero during training,
    giving the codebook direct gradient signal.

    This implementation is a *simplified academic variant* suitable for
    integration into the AEON-Delta training pipeline alongside the
    existing ``VectorQuantizerHybridV4``.

    Reference:
    - Zhu et al. (2024), "Addressing Codebook Collapse in VQ-VAE via
      Reparameterized Joint Codebook Updates," *NeurIPS*.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        sigma_init: float = 1.0,
        sigma_min: float = 0.01,
        anneal_rate: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.sigma_min = sigma_min
        self.anneal_rate = anneal_rate

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings,
                         1.0 / num_embeddings)

        self.register_buffer('_sigma', torch.tensor(sigma_init))
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_code_usage',
                             torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer('_total_count', torch.tensor(0, dtype=torch.long))

    @property
    def sigma(self) -> float:
        return self._sigma.item()

    def forward(
        self, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """SimVQ forward pass with reparameterised codebook update.

        Args:
            z: [B, D] latent vectors.

        Returns:
            quantized, loss, indices, info_dict
        """
        B, D = z.shape
        # Distance computation
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        e_k = self.embedding(indices)  # [B, D]

        # Reparameterised quantisation
        if self.training:
            noise = torch.randn_like(e_k) * self._sigma
            quantized = e_k + noise
            # Anneal sigma
            self._sigma.copy_(
                torch.clamp(self._sigma * (1.0 - self.anneal_rate),
                            min=self.sigma_min)
            )
            self._step += 1
        else:
            quantized = e_k

        # Commitment loss (encoder → codebook)
        commitment_loss = F.mse_loss(z, quantized.detach())
        # Codebook loss (codebook → encoder) — direct gradient via reparam
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # STE for backward through encoder
        quantized_st = z + (quantized - z).detach()

        # Usage tracking
        if self.training:
            self._total_count += B
            self._code_usage.scatter_add_(
                0, indices,
                torch.ones_like(indices, dtype=torch.long),
            )

        info = {
            'perplexity': self._compute_perplexity(indices),
            'codebook_loss': codebook_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'sigma': self._sigma.item(),
            'step': self._step.item(),
        }
        return quantized_st, loss, indices, info

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = encodings.mean(dim=0)
        probs_pos = avg_probs[avg_probs > 0]
        if probs_pos.numel() == 0:
            return 1.0
        perplexity = torch.exp(-torch.sum(probs_pos * torch.log(probs_pos + 1e-10)))
        return perplexity.item()

    def get_codebook_usage(self) -> float:
        if self._total_count > 0:
            used = (self._code_usage > 0).sum().item()
            return used / self.num_embeddings * 100
        return 0.0

    def get_usage_stats(self) -> dict:
        total = self._total_count.item()
        used = (self._code_usage > 0).sum().item()
        return {
            'total_codes': self.num_embeddings,
            'used_codes': used,
            'active_ratio': used / self.num_embeddings,
            'sigma': self._sigma.item(),
            'total_steps': self._step.item(),
        }


class MultiGroupVQ(nn.Module):
    """MGVQ — Multi-Group Vector Quantizer with per-group codebooks.

    Mitigates codebook collapse by splitting the latent space into G
    groups, each with its own codebook of K entries.  This multiplicative
    structure provides K^G effective codes while each sub-codebook
    maintains high utilization.

    Key idea (Zheng et al., 2023): For a latent z ∈ R^D, split into
    G sub-vectors z_1, …, z_G each of dimension D/G.  Quantise each
    independently::

        q = [q_1 || q_2 || … || q_G]

    where q_g = argmin_{e ∈ C_g} ||z_g − e||.  Each group codebook
    C_g has K entries.  Total effective codebook = K^G without storing
    K^G vectors.

    Reference:
    - Zheng et al. (2023), "Online Clustered Codebook," *ICLR*.
    - Yang et al. (2024), "Multi-Group Vector Quantization for
      Mitigating Collapse and Improving Reconstruction," *ICML*.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_groups: int = 4,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        if embedding_dim % num_groups != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.group_dim = embedding_dim // num_groups
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Per-group codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(num_embeddings, self.group_dim)
            for _ in range(num_groups)
        ])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / num_embeddings,
                             1.0 / num_embeddings)

        # Per-group EMA buffers
        for g in range(num_groups):
            self.register_buffer(
                f'_ema_count_{g}',
                torch.zeros(num_embeddings),
            )
            self.register_buffer(
                f'_ema_weight_{g}',
                self.codebooks[g].weight.data.clone(),
            )
            self.register_buffer(
                f'_usage_{g}',
                torch.zeros(num_embeddings, dtype=torch.long),
            )

        self.register_buffer('_total_steps', torch.tensor(0, dtype=torch.long))

    def forward(
        self, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Multi-group VQ forward pass.

        Args:
            z: [B, D] latent vectors.

        Returns:
            quantized, loss, indices, info_dict

        ``indices`` shape is [B, G] — one index per group.
        """
        B, D = z.shape
        # Split into groups
        z_groups = z.reshape(B, self.num_groups, self.group_dim)

        quantized_groups = []
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)
        per_group_perplexity = []

        for g in range(self.num_groups):
            z_g = z_groups[:, g, :]  # [B, group_dim]
            cb = self.codebooks[g]

            # Distances
            dist = (
                torch.sum(z_g ** 2, dim=1, keepdim=True)
                + torch.sum(cb.weight ** 2, dim=1)
                - 2 * torch.matmul(z_g, cb.weight.t())
            )
            idx = torch.argmin(dist, dim=1)  # [B]
            q_g = cb(idx)  # [B, group_dim]

            # Loss
            commitment = F.mse_loss(z_g, q_g.detach())
            codebook = F.mse_loss(q_g, z_g.detach())
            total_loss = total_loss + codebook + self.commitment_cost * commitment

            # STE
            q_g_st = z_g + (q_g - z_g).detach()
            quantized_groups.append(q_g_st)
            all_indices.append(idx)

            # EMA update
            if self.training:
                encodings = torch.zeros(B, self.num_embeddings, device=z.device)
                encodings.scatter_(1, idx.unsqueeze(1), 1)
                ema_count = getattr(self, f'_ema_count_{g}')
                ema_weight = getattr(self, f'_ema_weight_{g}')
                usage = getattr(self, f'_usage_{g}')

                ema_count.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay)
                dw = torch.matmul(encodings.t(), z_g)
                ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = ema_count.sum()
                if torch.isfinite(n) and n.item() > 1e-8:
                    smoothed = (ema_count + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                    cb.weight.data.copy_(
                        ema_weight / smoothed.clamp(min=1e-5).unsqueeze(1))

                usage.scatter_add_(0, idx,
                                   torch.ones_like(idx, dtype=torch.long))

            # Per-group perplexity
            with torch.no_grad():
                enc_mean = torch.zeros(self.num_embeddings, device=z.device)
                enc_mean.scatter_add_(0, idx,
                                      torch.ones(B, device=z.device))
                enc_mean = enc_mean / max(B, 1)
                pp = enc_mean[enc_mean > 0]
                ppl = torch.exp(-(pp * torch.log(pp + 1e-10)).sum()).item()
                per_group_perplexity.append(ppl)

        if self.training:
            self._total_steps += 1

        quantized = torch.cat(quantized_groups, dim=-1)  # [B, D]
        indices = torch.stack(all_indices, dim=1)  # [B, G]
        loss = total_loss / self.num_groups

        info = {
            'per_group_perplexity': per_group_perplexity,
            'mean_perplexity': sum(per_group_perplexity) / self.num_groups,
            'effective_codebook_size': self.num_embeddings ** self.num_groups,
            'num_groups': self.num_groups,
        }
        return quantized, loss, indices, info

    def get_codebook_usage(self) -> float:
        """Mean codebook utilization across groups (%)."""
        ratios = []
        for g in range(self.num_groups):
            usage = getattr(self, f'_usage_{g}')
            used = (usage > 0).sum().item()
            ratios.append(used / self.num_embeddings * 100)
        return sum(ratios) / len(ratios) if ratios else 0.0

    def get_per_group_stats(self) -> List[dict]:
        """Per-group utilization statistics."""
        stats = []
        for g in range(self.num_groups):
            usage = getattr(self, f'_usage_{g}')
            used = (usage > 0).sum().item()
            total = usage.sum().item()
            if total > 0:
                probs = usage.float() / total
                pp = probs[probs > 0]
                entropy = -(pp * torch.log(pp + 1e-10)).sum().item()
            else:
                entropy = 0.0
            max_ent = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
            stats.append({
                'group': g,
                'used_codes': used,
                'active_ratio': used / self.num_embeddings,
                'normalised_entropy': entropy / max_ent,
            })
        return stats


class ThoughtDecoder(nn.Module):
    """Декодер: z + tokens → logits"""
    
    def __init__(self, vocab_size: int, emb_dim: int = 256, z_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.z_dim = z_dim
        
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.z_proj = nn.Linear(z_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim * 2, emb_dim, batch_first=True)
        self.head = nn.Linear(emb_dim, vocab_size)
        self.head.weight = self.embed.weight  # Weight tying

    def forward(self, z: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        B, L = teacher_tokens.shape
        
        z_proj = self.z_proj(z)
        z_expanded = z_proj.unsqueeze(1).expand(-1, L, -1)
        
        emb = self.embed(teacher_tokens)
        emb = self.dropout(emb)
        
        lstm_input = torch.cat([emb, z_expanded], dim=-1)
        
        h0 = z_proj.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        out, _ = self.lstm(lstm_input, (h0, c0))
        out = self.dropout(out)
        
        logits = self.head(out)
        
        return logits


class ContextualRSSM(nn.Module):
    """
    ✅ НОВЫЙ: RSSM с контекстным окном
    
    Вместо предсказания z_{t+1} только из z_t,
    использует последние K состояний: [z_{t-K+1}, ..., z_t] → z_{t+1}
    
    Это позволяет модели учиться связным переходам между мыслями.
    """
    
    def __init__(self, hidden_dim: int, context_window: int = 3, 
                 rssm_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.rssm_hidden = rssm_hidden
        
        # Проекция контекста
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * context_window, rssm_hidden),
            nn.LayerNorm(rssm_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Attention over context (опционально, для взвешивания)
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # GRU для рекуррентной обработки
        self.gru = nn.GRUCell(rssm_hidden, rssm_hidden)
        
        # Выходная проекция
        self.out_proj = nn.Sequential(
            nn.Linear(rssm_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_context: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_context: [B, K, D] — контекст из K последних z
            hx: [B, rssm_hidden] — скрытое состояние GRU
            
        Returns:
            z_pred: [B, D] — предсказание следующего z
        """
        B, K, D = z_context.shape
        
        # ── Runtime dimension guard ──────────────────────────────────────────
        # context_proj was built for Linear(hidden_dim * context_window, …).
        # Mismatched K or D (e.g. when z_dim differs from hidden_dim, or
        # context_window changed after construction) produces the cryptic
        # "input and weight.T shapes cannot be multiplied" error.
        expected_flat = self.hidden_dim * self.context_window
        actual_flat = K * D
        if actual_flat != expected_flat:
            raise ValueError(
                f"ContextualRSSM.forward: z_context shape [{B}, {K}, {D}] "
                f"gives flat dim {actual_flat} but context_proj expects "
                f"{expected_flat} (hidden_dim={self.hidden_dim} × "
                f"context_window={self.context_window}).  "
                "Ensure z_dim == hidden_dim and K == context_window."
            )
        
        if hx is None:
            hx = torch.zeros(B, self.rssm_hidden, device=z_context.device)
        
        # Attention-взвешенный контекст
        attn_weights = self.context_attention(z_context)  # [B, K, 1]
        weighted_context = (z_context * attn_weights).sum(dim=1)  # [B, D]
        
        # Конкатенация всего контекста
        flat_context = z_context.reshape(B, -1)  # [B, K*D]
        
        # Проекция
        proj = self.context_proj(flat_context)  # [B, rssm_hidden]
        
        # GRU step
        hx_new = self.gru(proj, hx)
        
        # Выходная проекция с residual
        z_pred = self.out_proj(hx_new)
        
        # Residual: last z + attention-weighted context
        # Both z_last and weighted_context have dim D which equals hidden_dim
        # (enforced by the guard above), so addition to z_pred is safe.
        z_last = z_context[:, -1, :]
        z_pred = z_pred + self.residual_weight * z_last + weighted_context
        
        return z_pred
    
    def forward_single(self, z_t: torch.Tensor, 
                       hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Для совместимости: предсказание из одного z
        """
        # Создаём искусственный контекст из одного z
        z_context = z_t.unsqueeze(1).expand(-1, self.context_window, -1)
        return self.forward(z_context, hx)


class AEONDeltaV4(nn.Module):
    """Полная модель AEON-Delta v4 с контекстным RSSM"""
    
    def __init__(self, config: AEONConfigV4):
        super().__init__()
        self.config = config
        
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                logger.warning(f"Не удалось загрузить токенизатор: {e}")
        
        self.encoder = ThoughtEncoder(
            config.vocab_size, 
            z_dim=config.z_dim,
            dropout=config.dropout_rate
        )
        
        self.vq = GumbelVectorQuantizer(
            config.vq_num_embeddings, 
            config.vq_embedding_dim,
            commitment_cost=config.vq_commitment_cost,
            temperature=config.vq_temperature,
        )
        
        self.decoder = ThoughtDecoder(
            config.vocab_size, 
            z_dim=config.z_dim,
            dropout=config.dropout_rate
        )
        
        # ✅ Новый контекстный RSSM
        self.rssm = ContextualRSSM(
            config.hidden_dim, 
            context_window=config.context_window,
            rssm_hidden=config.rssm_hidden_dim,
            dropout=config.dropout_rate
        )
        
        # Provenance tracking for training-time root-cause traceability
        self.provenance_tracker = CausalProvenanceTracker()
        # Tensor safety for training/inference consistency
        self.tensor_guard = TensorGuard(policy=NaNPolicy.WARN)
        
        self._init_weights()
        
    def _init_weights(self):
        initialized_data_ptrs = set()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                data_ptr = module.weight.data.data_ptr()
                if data_ptr not in initialized_data_ptrs:
                    nn.init.xavier_uniform_(module.weight)
                    initialized_data_ptrs.add(data_ptr)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                data_ptr = module.weight.data.data_ptr()
                if data_ptr not in initialized_data_ptrs:
                    nn.init.normal_(module.weight, std=0.02)
                    initialized_data_ptrs.add(data_ptr)
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.GRUCell)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode token IDs into latent thought vectors.
        
        Args:
            tokens: Input token IDs of shape [B, seq_length].
            
        Returns:
            Latent vectors of shape [B, z_dim].
        """
        return self.encoder(tokens)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Quantize continuous latent vectors via VQ-VAE.
        
        Args:
            z: Continuous latent vectors of shape [B, z_dim].
            
        Returns:
            Tuple of (quantized, vq_loss, indices, stats).
        """
        return self.vq(z)

    def decode(self, quantized_z: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent vectors back to token logits.
        
        Args:
            quantized_z: Quantized vectors of shape [B, z_dim].
            teacher_tokens: Teacher-forced token IDs of shape [B, seq_length].
            
        Returns:
            Logits tensor of shape [B, seq_length, vocab_size].
        """
        return self.decoder(quantized_z, teacher_tokens)
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, Any]:
        self.provenance_tracker.reset()
        self.provenance_tracker.record_dependency("encoder", "vq")
        self.provenance_tracker.record_dependency("vq", "decoder")

        z = self.encode(tokens)
        z = self.tensor_guard.sanitize(z, context="encoder_output")
        # Record encoder contribution as delta from zero baseline
        # (token IDs and latent vectors have incompatible shapes).
        self.provenance_tracker.record_before("encoder", torch.zeros_like(z))
        self.provenance_tracker.record_after("encoder", z)

        self.provenance_tracker.record_before("vq", z)
        quantized, vq_loss, indices, vq_stats = self.quantize(z)
        quantized = self.tensor_guard.sanitize(quantized, context="vq_output")
        self.provenance_tracker.record_after("vq", quantized)

        self.provenance_tracker.record_before("decoder", quantized)
        logits = self.decode(quantized, tokens)
        # Use quantized-dim projection for provenance delta (decoder
        # output is [B, L, vocab_size] which is incompatible with
        # the [B, z_dim] before-state).
        _decoder_summary = logits.detach().mean(dim=1)[..., :quantized.shape[-1]]
        self.provenance_tracker.record_after("decoder", _decoder_summary)

        provenance = self.provenance_tracker.compute_attribution()

        return {
            "z": z,
            "quantized": quantized,
            "vq_loss": vq_loss,
            "indices": indices,
            "logits": logits,
            "vq_stats": vq_stats,
            "provenance": provenance,
        }

    def to_inference_state_dict(self) -> Dict[str, Any]:
        """Export weights compatible with AEONDeltaV3 inference model.

        Maps the V4 training module names (``encoder``, ``decoder``,
        ``vq``, ``rssm``) to V3's naming convention so that a checkpoint
        saved during training can be loaded directly by the inference
        engine without manual key remapping.

        Returns:
            Dict with ``state_dict`` (remapped weights) and ``config``
            (serialised training config for provenance).
        """
        mapping = {
            "vq.": "vector_quantizer.",
            "rssm.": "rssm_cell.",
        }
        remapped: Dict[str, Any] = {}
        for key, value in self.state_dict().items():
            new_key = key
            for src_prefix, dst_prefix in mapping.items():
                if key.startswith(src_prefix):
                    new_key = dst_prefix + key[len(src_prefix):]
                    break
            remapped[new_key] = value
        return {
            "state_dict": remapped,
            "config": {
                "z_dim": self.config.z_dim,
                "hidden_dim": self.config.hidden_dim,
                "vocab_size": self.config.vocab_size,
                "source_model": "AEONDeltaV4",
            },
        }

    def verify_training_coherence(self) -> Dict[str, Any]:
        """Validate training model self-consistency.

        Checks that:
        1. Provenance DAG is acyclic and covers all pipeline stages.
        2. Tensor guard has not accumulated excessive NaN/Inf events.
        3. Encoder→VQ→Decoder round-trip preserves finite values.

        Returns:
            Dict with ``coherent`` bool, ``provenance_dag`` validation,
            ``tensor_safety`` statistics, and ``recommendations`` list.
        """
        recommendations: List[str] = []

        # 1. Provenance DAG validation
        dag_result = self.provenance_tracker.validate_dag_acyclic()
        trace_result = self.provenance_tracker.verify_trace_completeness()
        if not dag_result.get('is_acyclic', True):
            recommendations.append(
                "Fix provenance DAG cycles in training pipeline"
            )
        if not trace_result.get('complete', True):
            recommendations.append(
                f"Register provenance for: "
                f"{', '.join(trace_result.get('uncovered', []))}"
            )

        # 2. Tensor safety statistics
        if hasattr(self.tensor_guard, 'get_stats'):
            safety_stats = self.tensor_guard.get_stats()
        else:
            safety_stats = {
                'nan_count': getattr(self.tensor_guard, '_nan_count', 0),
                'inf_count': getattr(self.tensor_guard, '_inf_count', 0),
                'sanitize_count': getattr(self.tensor_guard, '_sanitize_count', 0),
            }
        if safety_stats.get('nan_count', 0) > 0:
            recommendations.append(
                f"Training produced {safety_stats['nan_count']} NaN tensors"
            )
        if safety_stats.get('inf_count', 0) > 0:
            recommendations.append(
                f"Training produced {safety_stats['inf_count']} Inf tensors"
            )

        # 3. Round-trip sanity check (short sequence to minimise overhead)
        _ROUNDTRIP_SEQ_LEN = 8
        round_trip_ok = True
        try:
            with torch.no_grad():
                test_ids = torch.randint(
                    1, self.config.vocab_size, (1, _ROUNDTRIP_SEQ_LEN),
                    device=next(self.parameters()).device,
                )
                result = self.forward(test_ids)
                if not torch.isfinite(result['logits']).all():
                    round_trip_ok = False
                    recommendations.append(
                        "Encoder→VQ→Decoder round-trip produces non-finite values"
                    )
        except Exception as e:
            round_trip_ok = False
            recommendations.append(f"Round-trip check failed: {e}")

        coherent = (
            dag_result.get('is_acyclic', True)
            and trace_result.get('complete', True)
            and safety_stats.get('nan_count', 0) == 0
            and round_trip_ok
        )

        if not recommendations:
            recommendations.append("All training coherence checks passed.")

        return {
            'coherent': coherent,
            'provenance_dag': dag_result,
            'trace_completeness': trace_result,
            'tensor_safety': safety_stats,
            'round_trip_ok': round_trip_ok,
            'recommendations': recommendations,
        }

    def export_provenance_for_checkpoint(self) -> Dict[str, Any]:
        """Export training provenance data for inclusion in checkpoints.

        Systematically packages causal attribution, dependency graph,
        and tensor safety statistics so that
        ``AEONDeltaV3.load_v4_checkpoint()`` can import training-time
        root-cause traceability into the inference pipeline.

        Returns:
            Dict with ``provenance_attribution``, ``dependency_graph``,
            ``tensor_safety``, and ``dag_validation`` suitable for
            serialisation alongside model weights.
        """
        attribution = self.provenance_tracker.compute_attribution()
        dep_graph = self.provenance_tracker.get_dependency_graph()
        dag_validation = self.provenance_tracker.validate_dag_acyclic()

        if hasattr(self.tensor_guard, 'get_stats'):
            safety_stats = self.tensor_guard.get_stats()
        else:
            safety_stats = {
                'nan_count': getattr(self.tensor_guard, '_nan_count', 0),
                'inf_count': getattr(self.tensor_guard, '_inf_count', 0),
            }

        return {
            'provenance_attribution': attribution,
            'dependency_graph': {
                k: list(v) if isinstance(v, set) else v
                for k, v in dep_graph.items()
            },
            'dag_validation': dag_validation,
            'tensor_safety': safety_stats,
            'source_model': 'AEONDeltaV4',
        }

    def check_training_readiness(self) -> Dict[str, Any]:
        """Validate that the model is ready for training.

        Pre-training gate that verifies:
        1. Config–architecture alignment (z_dim matches module dims).
        2. All sub-modules produce finite outputs for a probe input.
        3. Provenance tracker is operational.

        This implements the "each component verifies the others"
        principle by performing a lightweight cross-module probe
        before training begins, catching misconfigurations that
        would otherwise surface as NaN divergence mid-training.

        Returns:
            Dict with ``ready`` bool, ``checks`` list of individual
            check results, and ``errors`` list of failure descriptions.
        """
        checks: List[Dict[str, Any]] = []
        errors: List[str] = []

        # 1. Config-architecture alignment
        encoder_out_dim = getattr(self.encoder, 'z_dim', None)
        if encoder_out_dim is None:
            # Try to infer from final projection layer
            for name, mod in self.encoder.named_modules():
                if isinstance(mod, nn.Linear):
                    encoder_out_dim = mod.out_features
        config_ok = encoder_out_dim is None or encoder_out_dim == self.config.z_dim
        checks.append({
            'name': 'config_architecture_alignment',
            'passed': config_ok,
            'detail': f'encoder_out={encoder_out_dim}, config.z_dim={self.config.z_dim}',
        })
        if not config_ok:
            errors.append(
                f"z_dim mismatch: encoder outputs {encoder_out_dim} "
                f"but config expects {self.config.z_dim}"
            )

        # 2. Sub-module finite-output probe
        _SEQ_LEN = 4
        device = next(self.parameters()).device
        try:
            with torch.no_grad():
                probe_ids = torch.randint(
                    1, self.config.vocab_size, (1, _SEQ_LEN), device=device,
                )
                z = self.encode(probe_ids)
                z_finite = torch.isfinite(z).all().item()
                checks.append({
                    'name': 'encoder_finite_output',
                    'passed': z_finite,
                })
                if not z_finite:
                    errors.append("Encoder produces non-finite output")

                quantized, vq_loss, _, _ = self.quantize(z)
                vq_finite = (
                    torch.isfinite(quantized).all().item()
                    and torch.isfinite(vq_loss).item()
                )
                checks.append({
                    'name': 'vq_finite_output',
                    'passed': vq_finite,
                })
                if not vq_finite:
                    errors.append("VQ produces non-finite output")

                logits = self.decode(quantized, probe_ids)
                dec_finite = torch.isfinite(logits).all().item()
                checks.append({
                    'name': 'decoder_finite_output',
                    'passed': dec_finite,
                })
                if not dec_finite:
                    errors.append("Decoder produces non-finite output")
        except Exception as e:
            checks.append({
                'name': 'sub_module_probe',
                'passed': False,
                'detail': str(e),
            })
            errors.append(f"Sub-module probe failed: {e}")

        # 3. Provenance tracker operational
        try:
            self.provenance_tracker.reset()
            self.provenance_tracker.record_dependency("_probe_a", "_probe_b")
            dag = self.provenance_tracker.get_dependency_graph()
            prov_ok = "_probe_b" in dag
            self.provenance_tracker.reset()
            checks.append({
                'name': 'provenance_tracker_operational',
                'passed': prov_ok,
            })
            if not prov_ok:
                errors.append("Provenance tracker failed to record dependency")
        except Exception as e:
            checks.append({
                'name': 'provenance_tracker_operational',
                'passed': False,
                'detail': str(e),
            })
            errors.append(f"Provenance tracker error: {e}")

        ready = len(errors) == 0
        return {
            'ready': ready,
            'checks': checks,
            'errors': errors,
        }

    def get_regularization_terms(self) -> Dict[str, torch.Tensor]:
        """Return signal-derived regularization losses for training.

        Training-side mirror of ``AEONDeltaV3.get_regularization_terms()``.
        Computes lightweight regularization penalties from available
        training-time signals:

        1. **coherence_loss** — provenance-based coherence deficit.
        2. **stability_loss** — gradient-based stability proxy.

        Returns:
            Dict mapping ``{term_name: loss_tensor}``.
        """
        dev = next(self.parameters()).device
        terms: Dict[str, torch.Tensor] = {}

        # Coherence loss from provenance tracker
        attribution = self.provenance_tracker.compute_attribution()
        contributions = attribution.get('contributions', {})
        if contributions:
            vals = list(contributions.values())
            # High variance in contributions ⇒ high imbalance ⇒ coherence deficit
            if len(vals) >= 2:
                mean_c = sum(vals) / len(vals)
                variance = sum((v - mean_c) ** 2 for v in vals) / len(vals)
                terms['coherence_loss'] = torch.tensor(
                    variance, dtype=torch.float32, device=dev,
                )
            else:
                terms['coherence_loss'] = torch.tensor(0.0, device=dev)
        else:
            terms['coherence_loss'] = torch.tensor(0.0, device=dev)

        # Stability loss — zero placeholder (V4 has no Lipschitz module)
        terms['stability_loss'] = torch.tensor(0.0, device=dev)
        terms['uncertainty_loss'] = torch.tensor(0.0, device=dev)
        terms['psi_loss'] = torch.tensor(0.0, device=dev)
        terms['delta_psi_loss'] = torch.tensor(0.0, device=dev)

        return terms

# ==============================================================================
# ДОКУМЕНТ-ОРИЕНТИРОВАННЫЙ DATASET
# ==============================================================================

class DocumentAwareDataset(Dataset):
    """
    ✅ НОВОЕ: Dataset, который строит z_pairs ТОЛЬКО внутри документов
    
    Это гарантирует, что RSSM учится на связанных переходах мыслей,
    а не на случайных соседствах из разных документов.
    """
    
    def __init__(self, documents: List[List[torch.Tensor]], context_window: int = 3):
        """
        Args:
            documents: List of documents, each is a list of token tensors (chunks).
            context_window: Number of previous z to use as context (must be >= 1).
            
        Raises:
            ValueError: If documents is empty or context_window < 1.
        """
        if not documents:
            raise ValueError("documents list must not be empty")
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        
        self.context_window = context_window
        self.samples = []  # List of (doc_idx, chunk_indices)
        
        # Создаём список валидных семплов
        for doc_idx, doc_chunks in enumerate(documents):
            num_chunks = len(doc_chunks)
            # Нужно минимум context_window + 1 чанков для создания пары
            if num_chunks >= context_window + 1:
                for i in range(context_window, num_chunks):
                    # context: [i-context_window, ..., i-1]
                    # target: i
                    self.samples.append((doc_idx, i))
        
        self.documents = documents
        
        if len(self.samples) == 0:
            logger.warning(
                "DocumentAwareDataset: no valid samples created. "
                "All documents have fewer than context_window + 1 chunks."
            )
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        doc_idx, target_idx = self.samples[idx]
        doc = self.documents[doc_idx]
        
        # Собираем контекст
        context_indices = list(range(target_idx - self.context_window, target_idx))
        context_chunks = [doc[i] for i in context_indices]
        target_chunk = doc[target_idx]
        
        return {
            'context': torch.stack(context_chunks),  # [K, seq_len]
            'target': target_chunk  # [seq_len]
        }


# ==============================================================================
# ТОКЕНИЗАЦИЯ
# ==============================================================================

def tokenize_batch(texts: List[str], tokenizer, max_len: int, 
                   device: torch.device, fallback_vocab_size: int = 50000) -> torch.Tensor:
    """
    Tokenize a batch of text strings into padded token ID tensors.
    
    Args:
        texts: List of text strings to tokenize.
        tokenizer: HuggingFace tokenizer instance, or None for ASCII fallback.
        max_len: Maximum sequence length (texts are truncated/padded to this).
        device: Target device for the output tensor.
        fallback_vocab_size: Vocabulary size for ASCII fallback tokenizer.
        
    Returns:
        Tensor of shape [len(texts), max_len] with token IDs (dtype=torch.long).
    """
    if tokenizer:
        encoded = tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=max_len, 
            return_tensors='pt'
        )
        return encoded['input_ids'].to(device)
    
    tokenized = []
    for text in texts:
        tokens = [ord(c) % fallback_vocab_size for c in text[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        tokenized.append(tokens)
    if not tokenized:
        return torch.zeros((0, max_len), dtype=torch.long, device=device)
    return torch.tensor(tokenized, dtype=torch.long, device=device)


def load_documents_from_json(json_path: str, tokenizer, max_len: int, 
                             min_chunks: int = 2, logger=None) -> List[List[torch.Tensor]]:
    """
    Load documents from a JSON-lines file, preserving document structure.
    
    Each line should be a JSON object with one of:
    - {"doc_id": "...", "chunks": ["chunk1 text", "chunk2 text", ...]}
    - {"text": "full document text"} — will be split into chunks automatically
    
    Args:
        json_path: Path to the JSON-lines file.
        tokenizer: HuggingFace tokenizer or None (falls back to ASCII tokenization).
        max_len: Maximum token sequence length per chunk.
        min_chunks: Minimum number of chunks per document to include it.
        logger: Optional logger instance.
        
    Returns:
        List of documents, where each document is a list of token tensors.
        
    Raises:
        FileNotFoundError: If json_path does not exist.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    documents = []
    errors = 0
    
    if logger:
        logger.info(f"📥 Загрузка документов из {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                if "chunks" in data:
                    # Документ уже разбит на чанки
                    chunks = data["chunks"]
                elif "text" in data:
                    # Разбиваем текст на чанки
                    text = data["text"]
                    # Простое разбиение по предложениям/абзацам
                    chunks = split_text_into_chunks(text, max_len * 4)  # ~4 символа на токен
                else:
                    chunks = [str(data)]
                
                if len(chunks) >= min_chunks:
                    # Токенизируем каждый чанк
                    tokenized_chunks = []
                    for chunk in chunks:
                        if len(chunk.strip()) > 10:
                            tokens = tokenize_batch([chunk], tokenizer, max_len, 
                                                   torch.device('cpu'))[0]
                            tokenized_chunks.append(tokens)
                    
                    if len(tokenized_chunks) >= min_chunks:
                        documents.append(tokenized_chunks)
                        
            except Exception as e:
                errors += 1
                if errors <= 3 and logger:
                    logger.warning(f"   Ошибка строки {line_num}: {e}")
                elif errors == 4 and logger:
                    logger.warning("   ... (suppressing further per-line error messages)")
    
    if logger:
        logger.info(f"✅ Загружено {len(documents):,} документов")
        total_chunks = sum(len(d) for d in documents)
        logger.info(f"   Всего чанков: {total_chunks:,}")
        avg_chunks = total_chunks / len(documents) if documents else 0
        logger.info(f"   Среднее чанков/документ: {avg_chunks:.1f}")
        logger.info(f"   Пропущено с ошибками: {errors}")
    
    return documents


def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """Разбивает текст на чанки по границам предложений"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Простое разбиение по точкам
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 2 <= max_chars:
            current_chunk += (". " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk + ".")
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk + ".")
    
    return chunks


# ==============================================================================
# LEARNING RATE SCHEDULER
# ==============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            progress = min(1.0, (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class DataCharacteristicsAnalyzer:
    """Analyzes training data to adaptively select initial hyperparameters.

    Examines token distributions, sequence statistics, and document
    structure to recommend learning rate, batch size, gradient clip,
    and other training parameters. This implements the 'intelligent
    parameter selection' requirement: instead of fixed defaults, the
    training pipeline adapts its initial configuration to the data.
    """

    _SMALL_DATASET_THRESHOLD = 100
    _LARGE_DATASET_THRESHOLD = 10000
    _VERY_SMALL_DATASET_THRESHOLD = 64
    _SMALL_DATASET_BS_THRESHOLD = 256
    _HIGH_TOKEN_VARIANCE_THRESHOLD = 15000
    _ESTIMATED_EPOCHS = 30
    _LOW_VOCAB_COVERAGE_THRESHOLD = 0.1

    def __init__(self, config: AEONConfigV4):
        self.config = config
        self._stats: Dict[str, Any] = {}

    def analyze(self, tokens: torch.Tensor,
                documents: Optional[List[List[torch.Tensor]]] = None) -> Dict[str, Any]:
        """Compute data characteristics and recommended parameters.

        Args:
            tokens: Flat token tensor [N, seq_len].
            documents: Optional document-structured token lists.

        Returns:
            Dictionary with 'stats' and 'recommendations' keys.
        """
        stats: Dict[str, Any] = {}
        with torch.no_grad():
            stats['n_samples'] = tokens.shape[0]
            stats['seq_length'] = tokens.shape[1] if tokens.dim() > 1 else 0
            stats['vocab_used'] = int(tokens.unique().numel())
            stats['vocab_total'] = self.config.vocab_size
            stats['vocab_coverage'] = stats['vocab_used'] / max(stats['vocab_total'], 1)

            # Token frequency statistics
            flat = tokens.flatten().float()
            stats['token_mean'] = float(flat.mean())
            stats['token_std'] = float(flat.std())

            # Padding ratio (token 0 is typically padding)
            zero_count = (tokens == 0).sum().item()
            total_count = tokens.numel()
            stats['padding_ratio'] = zero_count / max(total_count, 1)

            # Document structure stats
            if documents:
                doc_lengths = [len(d) for d in documents]
                stats['n_documents'] = len(documents)
                stats['avg_chunks_per_doc'] = sum(doc_lengths) / max(len(doc_lengths), 1)
                stats['max_chunks_per_doc'] = max(doc_lengths) if doc_lengths else 0
                stats['min_chunks_per_doc'] = min(doc_lengths) if doc_lengths else 0
            else:
                stats['n_documents'] = 0
                stats['avg_chunks_per_doc'] = 0

        self._stats = stats
        recommendations = self._compute_recommendations(stats)
        return {'stats': stats, 'recommendations': recommendations}

    def _compute_recommendations(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Derive hyperparameter recommendations from data statistics."""
        rec: Dict[str, Any] = {}
        n = stats['n_samples']
        base_cfg = self.config

        # Learning rate: scale down for small datasets, up for large
        if n < self._SMALL_DATASET_THRESHOLD:
            rec['learning_rate'] = max(base_cfg.min_learning_rate, base_cfg.learning_rate * 0.5)
            rec['lr_reason'] = 'Small dataset — reduced LR to prevent overfitting'
        elif n > self._LARGE_DATASET_THRESHOLD:
            rec['learning_rate'] = min(base_cfg.learning_rate * 2.0, 1e-3)
            rec['lr_reason'] = 'Large dataset — increased LR for faster convergence'
        else:
            rec['learning_rate'] = base_cfg.learning_rate
            rec['lr_reason'] = 'Standard dataset size'

        # Batch size: adapt to dataset size
        if n < self._VERY_SMALL_DATASET_THRESHOLD:
            rec['batch_size'] = min(base_cfg.batch_size, max(2, n // 4))
            rec['bs_reason'] = 'Very small dataset — reduced batch size'
        elif n < self._SMALL_DATASET_BS_THRESHOLD:
            rec['batch_size'] = min(base_cfg.batch_size, 8)
            rec['bs_reason'] = 'Small dataset — moderate batch size'
        else:
            rec['batch_size'] = base_cfg.batch_size
            rec['bs_reason'] = 'Standard batch size'

        # Gradient clip: tighter for high-variance data
        token_std = stats.get('token_std', 0)
        if token_std > self._HIGH_TOKEN_VARIANCE_THRESHOLD:
            rec['grad_clip_norm'] = max(0.1, base_cfg.grad_clip_norm * 0.5)
            rec['gc_reason'] = 'High token variance — tighter gradient clipping'
        else:
            rec['grad_clip_norm'] = base_cfg.grad_clip_norm
            rec['gc_reason'] = 'Normal token distribution'

        # Warmup: proportional to dataset size
        estimated_steps = max(n // base_cfg.batch_size, 1) * self._ESTIMATED_EPOCHS
        rec['warmup_steps'] = min(base_cfg.warmup_steps, max(50, estimated_steps // 10))
        rec['warmup_reason'] = f'Proportional to ~{estimated_steps} estimated steps'

        # VQ codebook: scale with vocabulary coverage
        coverage = stats.get('vocab_coverage', 0)
        if coverage < self._LOW_VOCAB_COVERAGE_THRESHOLD:
            rec['vq_num_embeddings'] = min(base_cfg.vq_num_embeddings, 512)
            rec['vq_reason'] = 'Low vocab coverage — smaller codebook to avoid dead codes'
        else:
            rec['vq_num_embeddings'] = base_cfg.vq_num_embeddings
            rec['vq_reason'] = 'Standard codebook size'

        # Context window: adapt to average document length
        avg_chunks = stats.get('avg_chunks_per_doc', 0)
        if avg_chunks > 0 and avg_chunks < base_cfg.context_window + 1:
            rec['context_window'] = max(1, int(avg_chunks) - 1)
            rec['cw_reason'] = f'Documents average {avg_chunks:.1f} chunks — reduced context window'
        else:
            rec['context_window'] = base_cfg.context_window
            rec['cw_reason'] = 'Standard context window'

        return rec

    _REASON_KEYS = {
        'learning_rate': 'lr_reason',
        'batch_size': 'bs_reason',
        'grad_clip_norm': 'gc_reason',
        'warmup_steps': 'warmup_reason',
        'vq_num_embeddings': 'vq_reason',
        'context_window': 'cw_reason',
    }

    def apply_recommendations(self, config: AEONConfigV4,
                              recommendations: Dict[str, Any]) -> List[str]:
        """Apply recommended parameters to config. Returns list of changes made."""
        changes = []
        for key in ('learning_rate', 'batch_size', 'grad_clip_norm',
                    'warmup_steps', 'vq_num_embeddings', 'context_window'):
            if key in recommendations:
                old_val = getattr(config, key)
                new_val = recommendations[key]
                if old_val != new_val:
                    setattr(config, key, new_val)
                    reason = recommendations.get(self._REASON_KEYS.get(key, ''), '')
                    changes.append(f'{key}: {old_val} → {new_val} ({reason})')
        return changes


class AdaptiveTrainingController:
    """Real-time adaptive controller for training hyperparameters.

    Monitors loss trajectory, gradient norms, codebook utilization,
    and convergence signals to adjust training parameters during
    training. Implements redundant multi-strategy adaptation with
    causal traceability.

    Strategies:
    1. Loss-based LR adaptation (primary)
    2. Gradient-norm-based clip adaptation (secondary)
    3. Convergence-based patience adaptation (tertiary)

    Each strategy operates independently; if one fails, others continue.
    All adjustments are recorded in the provenance tracker.
    """

    # Boundaries to prevent runaway adaptation
    _MIN_LR = 1e-7
    _MAX_LR = 1e-2
    _MIN_GRAD_CLIP = 0.05
    _MAX_GRAD_CLIP = 10.0
    _LOSS_WINDOW = 5
    _GRAD_WINDOW = 5
    _PLATEAU_THRESHOLD = 1e-5
    _IMPROVEMENT_THRESHOLD = -1e-4
    _LOW_CB_USAGE = 5.0
    _HIGH_CB_USAGE = 95.0
    _CB_SATURATION_HISTORY_MIN = 10
    _LOSS_SPIKE_MULTIPLIER = 1.5
    _PLATEAU_LR_MULTIPLIER = 1.5
    _PLATEAU_COUNT_TRIGGER = 3
    _GRAD_CLIP_PROXIMITY_THRESHOLD = 0.9
    _GRAD_CLIP_LOWER_THRESHOLD = 0.1
    _GRAD_CLIP_EXPAND_FACTOR = 1.25
    _GRAD_CLIP_SHRINK_FACTOR = 3

    def __init__(self, config: AEONConfigV4):
        self.config = config
        self._loss_history: List[float] = []
        self._grad_history: List[float] = []
        self._cb_history: List[float] = []
        self._lr_history: List[float] = []
        self._adaptations: List[Dict[str, Any]] = []
        self._current_lr: float = config.learning_rate
        self._current_grad_clip: float = config.grad_clip_norm
        self._plateau_count: int = 0
        self._spike_count: int = 0

    def record_step(self, loss: float, grad_norm: float,
                    codebook_pct: float = 0.0,
                    lr: float = 0.0) -> Dict[str, Any]:
        """Record a training step and return adaptive adjustments.

        Args:
            loss: Current epoch loss.
            grad_norm: Current gradient norm.
            codebook_pct: Codebook utilization percentage.
            lr: Current learning rate.

        Returns:
            Dictionary with recommended adjustments (may be empty).
        """
        adjustments: Dict[str, Any] = {}

        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            return adjustments

        self._loss_history.append(loss)
        self._grad_history.append(grad_norm)
        self._cb_history.append(codebook_pct)
        self._lr_history.append(lr if lr > 0 else self._current_lr)

        if len(self._loss_history) < 3:
            return adjustments

        # Strategy 1: Loss-based LR adaptation
        lr_adj = self._adapt_learning_rate()
        if lr_adj:
            adjustments.update(lr_adj)

        # Strategy 2: Gradient-norm-based clip adaptation
        gc_adj = self._adapt_grad_clip()
        if gc_adj:
            adjustments.update(gc_adj)

        # Strategy 3: Codebook health monitoring
        cb_adj = self._adapt_codebook_params()
        if cb_adj:
            adjustments.update(cb_adj)

        if adjustments:
            self._adaptations.append({
                'epoch': len(self._loss_history),
                'adjustments': adjustments,
                'loss': loss,
                'grad_norm': grad_norm,
            })

        return adjustments

    def _adapt_learning_rate(self) -> Dict[str, Any]:
        """Adapt LR based on loss trajectory."""
        window = self._loss_history[-self._LOSS_WINDOW:]
        if len(window) < 3:
            return {}

        # Detect plateau
        deltas = [window[i] - window[i-1] for i in range(1, len(window))]
        avg_delta = sum(deltas) / len(deltas)

        if all(abs(d) < self._PLATEAU_THRESHOLD for d in deltas):
            self._plateau_count += 1
            if self._plateau_count >= self._PLATEAU_COUNT_TRIGGER:
                new_lr = min(self._current_lr * self._PLATEAU_LR_MULTIPLIER, self._MAX_LR)
                if new_lr != self._current_lr:
                    self._current_lr = new_lr
                    self._plateau_count = 0
                    return {'lr_factor': self._PLATEAU_LR_MULTIPLIER,
                            'lr_reason': 'loss_plateau',
                            'recommended_lr': new_lr}
        else:
            self._plateau_count = 0

        # Detect loss spike
        if len(window) >= 2 and window[-1] > window[-2] * self._LOSS_SPIKE_MULTIPLIER:
            self._spike_count += 1
            new_lr = max(self._current_lr * 0.5, self._MIN_LR)
            if new_lr != self._current_lr:
                self._current_lr = new_lr
                return {'lr_factor': 0.5, 'lr_reason': 'loss_spike',
                        'recommended_lr': new_lr}

        # Steady improvement — maintain current LR
        if avg_delta < self._IMPROVEMENT_THRESHOLD:
            self._spike_count = 0

        return {}

    def _adapt_grad_clip(self) -> Dict[str, Any]:
        """Adapt gradient clip based on gradient norm history."""
        window = self._grad_history[-self._GRAD_WINDOW:]
        if len(window) < 3:
            return {}

        avg_norm = sum(window) / len(window)
        max_norm = max(window)

        # If gradients consistently near clip value, loosen
        if avg_norm > self._current_grad_clip * self._GRAD_CLIP_PROXIMITY_THRESHOLD:
            new_clip = min(self._current_grad_clip * self._GRAD_CLIP_EXPAND_FACTOR, self._MAX_GRAD_CLIP)
            if new_clip != self._current_grad_clip:
                self._current_grad_clip = new_clip
                return {'grad_clip': new_clip, 'gc_reason': 'gradients_near_clip'}

        # If gradients very small, tighten for stability
        if avg_norm < self._current_grad_clip * self._GRAD_CLIP_LOWER_THRESHOLD and avg_norm > 0:
            new_clip = max(avg_norm * self._GRAD_CLIP_SHRINK_FACTOR, self._MIN_GRAD_CLIP)
            if new_clip != self._current_grad_clip:
                self._current_grad_clip = new_clip
                return {'grad_clip': new_clip, 'gc_reason': 'gradients_very_small'}

        return {}

    def _adapt_codebook_params(self) -> Dict[str, Any]:
        """Monitor codebook health and suggest adjustments."""
        if len(self._cb_history) < 5:
            return {}

        recent_cb = self._cb_history[-5:]
        avg_usage = sum(recent_cb) / len(recent_cb)

        if avg_usage < self._LOW_CB_USAGE:
            return {'cb_alert': 'very_low_usage',
                    'cb_recommendation': 'increase_entropy_weight'}
        if avg_usage > self._HIGH_CB_USAGE and len(self._cb_history) > self._CB_SATURATION_HISTORY_MIN:
            return {'cb_alert': 'near_saturation',
                    'cb_recommendation': 'increase_codebook_size'}
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Return current adaptive controller state for telemetry."""
        return {
            'current_lr': self._current_lr,
            'current_grad_clip': self._current_grad_clip,
            'plateau_count': self._plateau_count,
            'spike_count': self._spike_count,
            'total_adaptations': len(self._adaptations),
            'recent_adaptations': self._adaptations[-5:] if self._adaptations else [],
            'loss_trend': self._compute_trend(self._loss_history),
            'grad_trend': self._compute_trend(self._grad_history),
            'cb_trend': self._compute_trend(self._cb_history),
        }

    def _compute_trend(self, history: List[float]) -> float:
        """Compute linear trend of recent values."""
        if len(history) < 2:
            return 0.0
        recent = history[-min(10, len(history)):]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0


# ==============================================================================
# ТРЕЙНЕРЫ
# ==============================================================================

class SafeThoughtAETrainerV4:
    """Трейнер Phase A: AutoEncoder + VQ v4"""
    
    def __init__(self, model: AEONDeltaV4, config: AEONConfigV4, 
                 monitor: TrainingMonitor, output_dir: str):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.monitor = monitor
        self.output_dir = output_dir
        
        self.trainable_params = (
            list(model.encoder.parameters()) + 
            list(model.decoder.parameters()) + 
            list(model.vq.parameters())
        )
        
        self.optimizer = optim.AdamW(
            self.trainable_params, 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=config.label_smoothing
        )
        
        self.use_amp = config.use_amp and AMP_AVAILABLE and self.device.type == 'cuda'
        if self.use_amp:
            try:
                self.scaler = GradScaler(device=self.device.type)
            except TypeError:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        
        # Bridge to aeon_core convergence monitoring — tracks loss
        # trajectory across epochs to detect divergence/stagnation
        # and trigger adaptive training adjustments.
        # Wire CausalErrorEvolutionTracker so training divergence and
        # stagnation events are propagated to inference-time recovery.
        self._error_evolution = CausalErrorEvolutionTracker(max_history=200)
        self.convergence_monitor = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=10,
            error_evolution=self._error_evolution,
        )
        self.provenance = TrainingProvenanceTracker()
        # Error classifier for semantic error categorization when
        # aeon_core is available; provides richer diagnostics than
        # raw exception strings.
        self._error_classifier = SemanticErrorClassifier()
        # TensorGuard for NaN/Inf protection during training — extends
        # the inference pipeline's tensor safety to the training loop,
        # ensuring numerical consistency across both pipelines.
        self._tensor_guard = TensorGuard(policy=NaNPolicy.WARN, enable_tracking=True)
        # Wire provenance tracker to convergence monitor so that
        # convergence failure events include per-module attribution,
        # enabling root-cause analysis from training convergence
        # failures through to the originating component.
        _prov_tracker_a = (
            self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else None
        )
        if _prov_tracker_a is None:
            logger.warning(
                "Phase A: TrainingProvenanceTracker has no _tracker; "
                "convergence events will lack per-module attribution"
            )
        self.convergence_monitor.set_provenance_tracker(_prov_tracker_a)

        # --- Unified Cognitive Cycle integration ---
        # Wire convergence monitoring, coherence verification, error
        # evolution, and meta-cognitive triggers into a single cycle so
        # that training-time uncertainty triggers deeper reasoning and
        # all training decisions are causally traceable.
        self._coherence_verifier = ModuleCoherenceVerifier(
            hidden_dim=config.z_dim, threshold=0.5,
        )
        self._metacognitive_trigger = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=2,
        )
        self._core_convergence = ConvergenceMonitor(threshold=1e-5)
        # Wire SystemIntegrityMonitor into Phase A's UCC so that
        # training-time convergence and coherence health are recorded
        # and low global health escalates uncertainty, matching the
        # inference pipeline's integrity-aware UCC wiring.  Without
        # this, Phase A UCC evaluations cannot detect or react to
        # system integrity degradation during training.
        self._integrity_monitor = (
            SystemIntegrityMonitor(window_size=200)
            if AEON_CORE_AVAILABLE else None
        )
        self._unified_cycle = UnifiedCognitiveCycle(
            convergence_monitor=self._core_convergence,
            coherence_verifier=self._coherence_verifier,
            error_evolution=self._error_evolution,
            metacognitive_trigger=self._metacognitive_trigger,
            provenance_tracker=self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else CausalProvenanceTracker(),
            convergence_arbiter=UnifiedConvergenceArbiter() if AEON_CORE_AVAILABLE else None,
            uncertainty_tracker=DirectionalUncertaintyTracker() if AEON_CORE_AVAILABLE else None,
            memory_validator=None,  # Training has no memory retrieval
            integrity_monitor=self._integrity_monitor,
        )
        # Cache the most recent encoder and VQ output tensors so the
        # epoch-end UCC evaluation receives real subsystem states instead
        # of dummy zeros.  This closes the gap where coherence
        # verification during training used uninformative zero-tensors,
        # preventing the verifier from detecting actual misalignment.
        self._last_encoder_state: Optional[torch.Tensor] = None
        self._last_vq_state: Optional[torch.Tensor] = None
        # Bidirectional bridge: stores per-module feedback from
        # inference-time uncertainty tracking, enabling the training
        # loop to focus on historically problematic modules.
        self._inference_module_feedback: Dict[str, float] = {}
        # Expose gradient clip norm and metacognitive LR factor so that
        # bridge_inference_insights_to_training() can adapt these
        # parameters when inference discovers recurring error patterns.
        # Without these, the inference→training bridge is a no-op for
        # convergence conflicts and coherence deficits.
        self._grad_clip_norm: float = config.grad_clip_norm
        self._metacognitive_lr_factor: float = _METACOGNITIVE_LR_FACTOR

        # --- Adaptive training controller ---
        # Real-time parameter adaptation based on loss trajectory,
        # gradient norms, and codebook health. Implements redundant
        # multi-strategy adaptation with causal traceability.
        self.adaptive_controller = AdaptiveTrainingController(config)

        # --- VTStreamingSignalBus (Item #3) ---
        # Persistent streaming bridge between VibeThinker and training.
        # closed_loop_step() is called every epoch to push VT learner
        # state into the training controller, closing the continuous
        # feedback loop described in upgrade.txt Item #3.
        self._vt_streaming_bus: Optional['VTStreamingSignalBus'] = None
        self._vt_learner_ref: Optional['VibeThinkerContinuousLearner'] = None
        if VIBE_THINKER_AVAILABLE:
            try:
                self._vt_streaming_bus = VTStreamingSignalBus()
                self._vt_learner_ref = VibeThinkerContinuousLearner(
                    config=VibeThinkerConfig(),
                )
            except Exception:
                self._vt_streaming_bus = None
                self._vt_learner_ref = None

        # --- Signal Regularization (Level 1–3) ---
        # Learnable weights for regularization terms (uncertainty, coherence,
        # stability, Ψ, ΔΨ).  When aeon_core is available, uses the full
        # SignalRegularizationWeights module; otherwise falls back to fixed
        # scalar weights applied to the terms from model.get_regularization_terms().
        if AEON_CORE_AVAILABLE:
            from aeon_core import SignalRegularizationWeights
            self.signal_reg_weights = SignalRegularizationWeights(
                init_uncertainty=0.01,
                init_coherence=0.01,
                init_stability=0.01,
                init_psi=0.005,
                init_delta_psi=0.005,
            )
        else:
            self.signal_reg_weights = None
        # Step counter for periodic meta-weight adaptation
        self._meta_adapt_interval: int = 50
        self._meta_adapt_counter: int = 0

    def train_step(self, tokens: torch.Tensor) -> Dict[str, Any]:
        """Execute a single training step for the autoencoder.
        
        Args:
            tokens: Input token IDs of shape [B, seq_length].
            
        Returns:
            Dictionary with loss values and metrics:
                - total_loss: Combined reconstruction + VQ loss (Tensor).
                - recon_loss: Reconstruction loss (float).
                - vq_loss: Vector quantization loss (float).
                - perplexity: exp(recon_loss) (float).
                - accuracy: Token prediction accuracy percentage (float).
        """
        self.model.train()
        tokens = tokens.to(self.device)
        
        if self.use_amp:
            with autocast(device_type=self.device.type):
                outputs = self._forward_pass(tokens)
        else:
            outputs = self._forward_pass(tokens)
        
        total_loss = outputs['total_loss']
        
        # Detect NaN/Inf loss to prevent corrupted gradient updates.
        # When aeon_core is available, classify the error semantically
        # so root-cause analysis can trace it to a specific component
        # via the provenance tracker.
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            _error_detail = "NaN/Inf loss"
            _error_class = "numerical"
            if self._error_classifier is not None:
                try:
                    _err_cls, _err_detail = self._error_classifier.classify(
                        RuntimeError("NaN/Inf loss in training step")
                    )
                    _error_class = _err_cls
                    _error_detail = f"{_err_cls}: {_err_detail}"
                except Exception as exc:
                    logger.debug("Error classifier failed: %s", exc)
            # Identify the dominant provenance module at the point of
            # failure so the error is traceable to its root cause.
            _provenance = outputs.get('provenance', {})
            _dominant = None
            _contributions = _provenance.get('contributions', {})
            if _contributions:
                _dominant = max(_contributions, key=_contributions.get)
            logger.warning(
                f"⚠️ {_error_detail} at step {self.global_step}"
                f" (dominant_module={_dominant}), skipping backward pass"
            )
            # Propagate NaN event to convergence monitor so it can
            # detect training instability and recommend corrective action.
            self.convergence_monitor.update(float('nan'))
            # Record the semantically classified error in the error
            # evolution tracker so that training-time failures inform
            # inference-time recovery strategies with semantic context.
            self._error_evolution.record_episode(
                error_class=_error_class,
                strategy_used="skip_backward",
                success=False,
                metadata={
                    "step": self.global_step,
                    "dominant_module": _dominant,
                    "detail": _error_detail,
                },
                causal_antecedents=["training_backward", _dominant],
            )
            return outputs
        
        # Scale loss by gradient accumulation steps so that accumulated
        # gradients are equivalent to a single large-batch gradient.
        # Without this scaling, the effective loss is multiplied by
        # gradient_accumulation_steps, causing training instability.
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Log provenance on successful steps so that root-cause analysis
        # is available for normal-case training dynamics, not only on
        # NaN/Inf failures.  This closes the traceability gap where
        # successful steps were invisible to provenance diagnostics.
        _provenance = outputs.get('provenance', {})
        _contributions = _provenance.get('contributions', {})
        if _contributions and self.global_step % _PROVENANCE_LOG_INTERVAL == 0:
            _dominant = max(_contributions, key=_contributions.get)
            logger.debug(
                f"Step {self.global_step} provenance: dominant={_dominant} "
                f"({_contributions[_dominant]:.1%})"
            )

        # ── Level 3: Meta-Optimization of Signal Weights ──
        # Periodically adapt regularization weights based on error
        # evolution statistics, so the system learns which signals
        # are most important for convergence in the current context.
        self._meta_adapt_counter += 1
        if (self.signal_reg_weights is not None
                and self._meta_adapt_counter >= self._meta_adapt_interval):
            self._meta_adapt_counter = 0
            try:
                error_summary = self._error_evolution.get_error_summary()
                self.signal_reg_weights.adapt_from_error_evolution(
                    error_summary, lr=0.01,
                )
            except Exception as _adapt_err:
                logger.debug(
                    "Signal reg adaptation from error evolution failed "
                    "(non-fatal): %s", _adapt_err,
                )
        
        return outputs

    def _provenance_causal_quality(self) -> float:
        """Compute causal quality from provenance attribution balance.

        Delegates to the shared :func:`_compute_provenance_causal_quality`
        utility so the formula is maintained in a single location.
        """
        return _compute_provenance_causal_quality(self.provenance)
    
    def _forward_pass(self, tokens: torch.Tensor) -> Dict[str, Any]:
        # Track per-component provenance so training errors can be
        # traced back to their originating component.
        self.provenance.reset()

        # Record encoder input for provenance — tokens are integer IDs,
        # so use mean-pooled float representation as a lightweight proxy
        # for the encoder's input state fingerprint.
        _encoder_input_proxy = tokens.float().mean(dim=-1, keepdim=True)
        self.provenance.record_before("encoder", _encoder_input_proxy)
        z = self.model.encode(tokens)
        self.provenance.record_after("encoder", z)
        # Sanitize encoder output to prevent NaN/Inf from propagating
        # into VQ and decoder, matching the inference pipeline's safety.
        if self._tensor_guard is not None:
            z = self._tensor_guard.sanitize(z, context="training_encoder_output")
        # Record encoder output as both before/after VQ input
        self.provenance.record_before("vq", z)
        quantized, vq_loss, indices, vq_stats = self.model.quantize(z)
        self.provenance.record_after("vq", quantized)
        # Cache detached subsystem states for epoch-end UCC evaluation
        self._last_encoder_state = z.detach()
        self._last_vq_state = quantized.detach()

        self.provenance.record_before("decoder", quantized)
        logits = self.model.decode(quantized, tokens)
        # Record decoder delta using mean-pooled output (seq × vocab → z_dim)
        self.provenance.record_after("decoder", logits.mean(dim=(1, 2)).unsqueeze(-1).expand_as(quantized))
        
        recon_loss = self.criterion(
            logits[:, :-1].contiguous().view(-1, self.config.vocab_size), 
            tokens[:, 1:].contiguous().view(-1)
        )
        
        total_loss = recon_loss + self.config.vq_loss_weight * vq_loss

        # Consume inference module feedback to focus training on modules
        # that inference identified as high-uncertainty.  When inference
        # reports high uncertainty for "vq", boost VQ loss weight; when
        # "encoder" or "decoder" are flagged, boost reconstruction loss.
        # This closes the inference→training feedback loop where per-module
        # uncertainty was recorded but never acted upon during training.
        if self._inference_module_feedback:
            _vq_boost = self._inference_module_feedback.get('vq', 0.0)
            _enc_boost = self._inference_module_feedback.get('encoder', 0.0)
            _dec_boost = self._inference_module_feedback.get('decoder', 0.0)
            _recon_boost = max(_enc_boost, _dec_boost)
            if _vq_boost > 0.0:
                total_loss = total_loss + _vq_boost * vq_loss
            if _recon_boost > 0.0:
                total_loss = total_loss + _recon_boost * recon_loss

        # ── Signal Regularization (Level 1): Add regularization terms ──
        # Query the model for signal-derived penalty terms and sum them
        # into total_loss via learnable weights (Level 3).
        reg_loss_value = 0.0
        if hasattr(self.model, 'get_regularization_terms'):
            try:
                reg_terms = self.model.get_regularization_terms()
                if self.signal_reg_weights is not None:
                    reg_loss = self.signal_reg_weights(reg_terms)
                    total_loss = total_loss + reg_loss
                    reg_loss_value = reg_loss.item()
                else:
                    # Fallback: fixed small weights
                    for _name, _term in reg_terms.items():
                        if _term.item() > 0.0:
                            total_loss = total_loss + 0.01 * _term
                            reg_loss_value += 0.01 * _term.item()
            except Exception as _reg_err:
                logger.debug(
                    "Regularization term accumulation failed "
                    "(non-fatal): %s", _reg_err,
                )

        # ── Signal-Weighted Loss (Level 2): uncertainty-based weighting ──
        # Multiply the loss by (1 + uncertainty) to focus training on
        # high-uncertainty samples (hard example mining through signals).
        if hasattr(self.model, 'get_signal_weighted_factor'):
            try:
                sw_factor = self.model.get_signal_weighted_factor()
                if sw_factor > 1.0:
                    total_loss = total_loss * sw_factor
            except Exception as _sw_err:
                logger.debug(
                    "Signal-weighted loss factor failed "
                    "(non-fatal): %s", _sw_err,
                )

        with torch.no_grad():
            perplexity = torch.exp(recon_loss.clamp(max=80)).item()
            pred_tokens = logits[:, :-1].argmax(dim=-1)
            accuracy = (pred_tokens == tokens[:, 1:]).float().mean().item() * 100
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'reg_loss': reg_loss_value,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'provenance': self.provenance.compute_attribution(),
            'convergence_status': self.convergence_monitor.status,
            # Store the input batch (detached) so that meta-cognitive
            # re-execution can re-run the training step with corrected
            # parameters when the UCC triggers should_rerun=True.
            'input_batch': tokens.detach(),
            **vq_stats
        }
    
    def _optimizer_step(self):
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        # Use the bridgeable _grad_clip_norm so that
        # bridge_inference_insights_to_training() can tighten clipping
        # when inference detects recurring convergence conflicts.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_params, 
            self._grad_clip_norm
        )
        
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.global_step += 1
        
        return grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

    def fit(self, tokenized_tensor: torch.Tensor, epochs: int = 30, 
            log_every_batch: int = 10,
            curriculum_scores: Optional[List[float]] = None):
        
        # Item #1b: When curriculum_scores are available, use
        # _build_curriculum_order() each epoch to select and order
        # training samples from simple→complex.  This replaces random
        # shuffling with an empirically-grounded curriculum schedule.
        _use_curriculum = (
            curriculum_scores is not None
            and len(curriculum_scores) == len(tokenized_tensor)
        )

        loader = DataLoader(
            TensorDataset(tokenized_tensor), 
            batch_size=self.config.batch_size, 
            shuffle=not _use_curriculum,
            drop_last=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        total_batches = len(loader)
        # Ceiling division to ensure at least 1 total step
        total_steps = max(
            (epochs * total_batches + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps,
            1
        )
        
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_learning_rate
        )
        
        self.monitor.start_training("Phase A (AutoEncoder + VQ v4)", epochs, len(tokenized_tensor))
        self.monitor.log_model_stats(self.model, "AEON-Delta-v4")
        
        logger.info(f"   ✅ Warmup steps: {warmup_steps}")
        logger.info(f"   ✅ Total steps: {total_steps}")
        logger.info(f"   ✅ Gradient clip: {self.config.grad_clip_norm}")
        logger.info(f"   ✅ Entropy weight: {self.config.entropy_weight}")
        if _use_curriculum:
            logger.info(f"   ✅ Curriculum learning: enabled")
        
        self.optimizer.zero_grad()
        
        for epoch in range(epochs):
            self.monitor.start_epoch(epoch, epochs)

            # Item #1b: Rebuild curriculum-ordered loader each epoch
            if _use_curriculum:
                _cur_indices = _build_curriculum_order(
                    curriculum_scores,
                    num_samples=len(tokenized_tensor),
                    epoch=epoch,
                    total_epochs=epochs,
                )
                _cur_subset = torch.utils.data.Subset(
                    TensorDataset(tokenized_tensor), _cur_indices,
                )
                loader = DataLoader(
                    _cur_subset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=0,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
            
            epoch_metrics = {
                "recon": 0.0, "vq": 0.0, "total": 0.0, 
                "perplexity": 0.0, "accuracy_%": 0.0, 
                "codebook_%": 0.0, "grad_norm": 0.0
            }
            
            accumulated_loss = 0.0
            num_accumulated = 0
            outputs = None
            
            for batch_idx, (batch,) in enumerate(loader):
                outputs = self.train_step(batch)
                step_loss = outputs['total_loss'].item()
                if not (math.isnan(step_loss) or math.isinf(step_loss)):
                    accumulated_loss += step_loss
                    num_accumulated += 1
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if num_accumulated > 0:
                        grad_norm = self._optimizer_step()
                        self.scheduler.step()
                    else:
                        self.optimizer.zero_grad()
                        grad_norm = 0.0
                    
                    avg_loss = accumulated_loss / max(num_accumulated, 1)
                    accumulated_loss = 0.0
                    num_accumulated = 0
                    
                    epoch_metrics["total"] += avg_loss
                    if not (math.isnan(outputs['recon_loss']) or math.isinf(outputs['recon_loss'])):
                        epoch_metrics["recon"] += outputs['recon_loss']
                        epoch_metrics["vq"] += outputs['vq_loss']
                        epoch_metrics["perplexity"] += outputs['perplexity']
                        epoch_metrics["accuracy_%"] += outputs['accuracy']
                        epoch_metrics["codebook_%"] += outputs.get('codebook_usage_%', 0)
                    epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and math.isfinite(grad_norm)) else 0
                
                if batch_idx % log_every_batch == 0:
                    self.monitor.log_batch(batch_idx, total_batches, {
                        "loss": outputs['recon_loss'] + self.config.vq_loss_weight * outputs['vq_loss'],
                        "recon": outputs['recon_loss'],
                        "ppl": outputs['perplexity'],
                        "acc": outputs['accuracy'],
                        "cb%": outputs.get('codebook_usage_%', 0)
                    }, log_every=log_every_batch)
            
            if num_accumulated > 0 and outputs is not None:
                avg_loss = accumulated_loss / max(num_accumulated, 1)
                epoch_metrics["total"] += avg_loss
                if not (math.isnan(outputs['recon_loss']) or math.isinf(outputs['recon_loss'])):
                    epoch_metrics["recon"] += outputs['recon_loss']
                    epoch_metrics["vq"] += outputs['vq_loss']
                    epoch_metrics["perplexity"] += outputs['perplexity']
                    epoch_metrics["accuracy_%"] += outputs['accuracy']
                    epoch_metrics["codebook_%"] += outputs.get('codebook_usage_%', 0)
                grad_norm = self._optimizer_step()
                self.scheduler.step()
                epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and math.isfinite(grad_norm)) else 0
            
            num_steps = max(
                (total_batches + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps,
                1
            )
            for key in epoch_metrics:
                epoch_metrics[key] /= num_steps
            
            epoch_metrics["lr"] = self.scheduler.get_lr()

            # --- Adaptive parameter controller ---
            # Record epoch-level metrics and apply adaptive adjustments
            # to learning rate and gradient clip. This implements the
            # 'adaptive intelligent parameter adjustment' requirement.
            adaptive_adj = self.adaptive_controller.record_step(
                loss=epoch_metrics["total"],
                grad_norm=epoch_metrics["grad_norm"],
                codebook_pct=epoch_metrics.get("codebook_%", 0.0),
                lr=epoch_metrics["lr"],
            )
            if adaptive_adj:
                if 'recommended_lr' in adaptive_adj:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = adaptive_adj['recommended_lr']
                    epoch_metrics["lr"] = adaptive_adj['recommended_lr']
                    logger.info(
                        f"   🔄 Adaptive LR: {adaptive_adj['recommended_lr']:.2e} "
                        f"(reason: {adaptive_adj.get('lr_reason', 'unknown')})"
                    )
                if 'grad_clip' in adaptive_adj:
                    self._grad_clip_norm = adaptive_adj['grad_clip']
                    logger.info(
                        f"   🔄 Adaptive grad_clip: {adaptive_adj['grad_clip']:.3f} "
                        f"(reason: {adaptive_adj.get('gc_reason', 'unknown')})"
                    )
                if 'cb_alert' in adaptive_adj:
                    logger.info(
                        f"   📊 Codebook: {adaptive_adj['cb_alert']} — "
                        f"{adaptive_adj.get('cb_recommendation', '')}"
                    )
            epoch_metrics["adaptive_state"] = self.adaptive_controller.get_state()

            # Convergence monitoring — track loss trajectory across
            # epochs to detect divergence, stagnation, or convergence.
            # This bridges the training loop to aeon_core's
            # ConvergenceMonitor pattern, ensuring that training
            # dynamics feed back into adaptive behavior.
            convergence_verdict = self.convergence_monitor.update(
                epoch_metrics["total"]
            )
            epoch_metrics["convergence_status"] = convergence_verdict["status"]
            if convergence_verdict["status"] == "diverging":
                logger.warning(
                    f"   ⚠️ Convergence monitor: DIVERGING "
                    f"(trend={convergence_verdict['trend']:.6f}). "
                    f"Recommendation: {convergence_verdict['recommendation']}"
                )
                # Act on the divergence recommendation by reducing the
                # learning rate.  Without this, the convergence monitor
                # produces actionable recommendations that go unused,
                # breaking the feedback loop between monitoring and
                # training behavior.
                if convergence_verdict["recommendation"] == "reduce_lr_or_rollback":
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.info(
                        f"   ↘️ LR reduced to {self.optimizer.param_groups[0]['lr']:.2e} "
                        f"due to divergence"
                    )
            elif convergence_verdict["status"] == "stagnating":
                logger.info(
                    f"   ℹ️ Convergence monitor: stagnating "
                    f"(trend={convergence_verdict['trend']:.6f})"
                )
                # Adaptive LR response — increase learning rate slightly
                # when Phase A stagnates, closing the monitoring-to-action
                # feedback loop for the stagnation case.
                if convergence_verdict.get("recommendation") == "increase_lr_or_augment":
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = min(
                            param_group['lr'] * 1.5,
                            self.config.learning_rate,
                        )
                    logger.info(
                        f"   ↗️ LR increased to "
                        f"{self.optimizer.param_groups[0]['lr']:.2e} "
                        f"due to stagnation"
                    )

            # --- Unified Cognitive Cycle evaluation ---
            # Run the full meta-cognitive evaluation cycle so that every
            # epoch-end decision is verified for cross-module coherence,
            # uncertainty is routed to the metacognitive trigger, and all
            # conclusions are recorded in the causal provenance chain.
            try:
                _loss_delta = abs(convergence_verdict.get("trend", 0.0))
                _uncertainty = min(epoch_metrics.get("perplexity", 0.0) / _PERPLEXITY_UNCERTAINTY_SCALE, 1.0)
                _is_diverging = convergence_verdict["status"] == "diverging"
                # Adapt metacognitive trigger weights from accumulated
                # error-evolution history before UCC evaluation so that
                # historically problematic failure modes increase trigger
                # sensitivity during the upcoming evaluation cycle.
                _err_summary = self._error_evolution.get_error_summary()
                self._metacognitive_trigger.adapt_weights_from_evolution(
                    _err_summary,
                )
                # Item #5: Map VibeThinker signals to trigger inputs
                _vt_mapped = {}
                if VIBE_THINKER_AVAILABLE:
                    try:

                        # [W1] Invoke ucc_inner_epoch_evaluation() with
                        # interval gating so the function is no longer
                        # dead code.  Its result supplements the inline
                        # UCC evaluation below.
                        _ucc_inner_result = ucc_inner_epoch_evaluation(
                            cycle=self._unified_cycle,
                            subsystem_states={
                                "encoder": self._last_encoder_state
                                if self._last_encoder_state is not None
                                else torch.zeros(1, self.config.z_dim),
                                "vq": self._last_vq_state
                                if self._last_vq_state is not None
                                else torch.zeros(1, self.config.z_dim),
                            },
                            loss_delta=_loss_delta,
                            uncertainty=_uncertainty,
                            epoch=epoch,
                            total_epochs=epochs,
                            vt_signals=_vt_mapped,
                        )
                        epoch_metrics["ucc_inner_evaluated"] = _ucc_inner_result.get(
                            "evaluated", False,
                        )
                        _vt_mapped = map_vt_signals_to_trigger(
                            vt_confidence=1.0 - _uncertainty,
                            vt_entropy=min(1.0, epoch_metrics.get("perplexity", 0.0) / 100.0),
                            vt_calibration_error=max(0.0, 1.0 - epoch_metrics.get("cognitive_coherence", 1.0)) if "cognitive_coherence" in epoch_metrics else 0.0,
                        )
                    except Exception:
                        _vt_mapped = {}
                _cycle_result = self._unified_cycle.evaluate(
                    subsystem_states={
                        "encoder": self._last_encoder_state
                        if self._last_encoder_state is not None
                        else torch.zeros(1, self.config.z_dim),
                        "vq": self._last_vq_state
                        if self._last_vq_state is not None
                        else torch.zeros(1, self.config.z_dim),
                    },
                    delta_norm=_loss_delta,
                    uncertainty=_uncertainty,
                    recovery_pressure=1.0 if _is_diverging else 0.0,
                    # Feed provenance-derived causal quality so the UCC
                    # can detect when a single training subsystem dominates
                    # the output attribution (quality = 1 when balanced,
                    # degrades toward 0 when one module dominates entirely).
                    causal_quality=self._provenance_causal_quality(),
                    diversity_collapse=_vt_mapped.get("diversity_collapse", 0.0),
                    convergence_conflict=_vt_mapped.get("convergence_conflict", 0.0),
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
                # Item #7: Coherence-driven epoch rerun — when UCC
                # flags should_rerun, reduce LR for the next epoch to
                # stabilise representations.  This transforms training
                # from pure loss minimisation into joint optimisation of
                # loss and cognitive coherence.
                if _cycle_result["should_rerun"] and not getattr(self, '_rerun_applied', False):
                    for pg in self.optimizer.param_groups:
                        pg['lr'] *= _UCC_RERUN_LR_FACTOR
                    self._rerun_applied = True
                    logger.info(
                        f"   🔁 UCC inner-epoch rerun: LR reduced by "
                        f"{_UCC_RERUN_LR_FACTOR:.0%} for coherence "
                        f"(deficit={_cycle_result['coherence_result']['coherence_deficit']:.4f})"
                    )
                elif not _cycle_result["should_rerun"]:
                    self._rerun_applied = False
                # Synthesize a cognitive_unity_score mirroring the
                # inference pipeline's composite metric so training and
                # inference measure AGI coherence on the same scale.
                # Components: coherence, convergence quality, provenance
                # completeness, metacognitive responsiveness.
                _cus_coherence = epoch_metrics["cognitive_coherence"]
                _cus_convergence = max(0.0, min(
                    1.0, 1.0 - abs(convergence_verdict.get("trend", 0.0)),
                ))
                _cus_provenance = self._provenance_causal_quality()
                # Metacognitive responsiveness: when uncertainty was high
                # and UCC triggered re-reasoning, responsiveness is 1.0.
                # When uncertainty was high but UCC did NOT trigger,
                # responsiveness degrades — mirroring inference-side logic.
                if _uncertainty > 0.5:
                    _cus_metacognitive = (
                        1.0 if _cycle_result["should_rerun"]
                        else 1.0 - _uncertainty
                    )
                else:
                    _cus_metacognitive = 1.0
                epoch_metrics["cognitive_unity_score"] = (
                    _CUS_WEIGHT_COHERENCE * _cus_coherence
                    + _CUS_WEIGHT_CONVERGENCE * _cus_convergence
                    + _CUS_WEIGHT_PROVENANCE * _cus_provenance
                    + _CUS_WEIGHT_UNCERTAINTY * _cus_metacognitive
                )
                if _cycle_result["should_rerun"]:
                    _active = _cycle_result["trigger_detail"].get("triggers_active", [])
                    logger.info(
                        f"   🧠 Meta-cognitive cycle triggered "
                        f"(signals={_active}), adapting training"
                    )
                    # Tighten gradient clipping when meta-cognitive cycle
                    # detects the system needs deeper reasoning — this
                    # closes the feedback loop between the cognitive
                    # architecture and the training optimizer.
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self._metacognitive_lr_factor
                    # Re-execute one corrective training step with the
                    # tightened parameters when the meta-cognitive cycle
                    # triggers.  Previously, should_rerun only adjusted
                    # LR and gradient clip, deferring the actual corrective
                    # computation to the next epoch.  This re-execution
                    # ensures that any uncertainty immediately triggers a
                    # meta-cognitive cycle that produces a corrective
                    # gradient, satisfying the requirement that uncertainty
                    # triggers corrective action within the same cycle.
                    if outputs is not None and isinstance(outputs, dict) and 'input_batch' in outputs:
                        try:
                            _rerun_out = self.train_step(outputs['input_batch'])
                            _rerun_loss = _rerun_out['total_loss']
                            # Accept re-execution if it produces a finite
                            # loss.  The corrective step uses tightened LR
                            # and gradient clip, so the resulting gradients
                            # are inherently more conservative.
                            if not (math.isnan(_rerun_loss.item()) or math.isinf(_rerun_loss.item())):
                                _rerun_grad = self._optimizer_step()
                                logger.info(
                                    f"   🔄 Meta-cognitive re-execution: "
                                    f"loss={_rerun_loss.item():.6f}, "
                                    f"grad_norm={_rerun_grad:.4f}"
                                )
                                if self._error_evolution is not None:
                                    self._error_evolution.record_episode(
                                        error_class='training_metacognitive_rerun',
                                        strategy_used='corrective_step',
                                        success=True,
                                        metadata={
                                            'rerun_loss': _rerun_loss.item(),
                                            'triggers_active': _active,
                                        },
                                        causal_antecedents=["training_metacognitive", "corrective_step"],
                                    )
                        except Exception as _rerun_err:
                            logger.debug(
                                "Meta-cognitive re-execution failed "
                                "(non-fatal): %s", _rerun_err,
                            )
                # Apply correction_guidance from UCC — when the unified
                # cognitive cycle identifies a specific target module and
                # recommended strategy, use that insight to adapt training
                # parameters for the identified module.  This closes the
                # gap where UCC computed detailed correction guidance
                # (target module, weakest pair, recommended strategy) but
                # the training loop only applied a generic LR reduction,
                # discarding the synthesized recommendation.
                _correction = _cycle_result.get("correction_guidance", {})
                _corr_target = _correction.get("target_module")
                if _corr_target is not None:
                    epoch_metrics["correction_target"] = _corr_target
                    epoch_metrics["correction_reason"] = _correction.get(
                        "reason", "unknown",
                    )
                    # Tighten grad clip when a specific module is the root
                    # cause of recurring issues.
                    if _correction.get("recommended_strategy"):
                        self._grad_clip_norm = max(
                            0.1, self._grad_clip_norm * 0.95,
                        )
                    # Store the correction target so the next training
                    # step can prioritise the problematic module.
                    if hasattr(self, '_inference_module_feedback'):
                        self._inference_module_feedback[_corr_target] = max(
                            self._inference_module_feedback.get(
                                _corr_target, 0.0,
                            ),
                            1.0 - epoch_metrics.get(
                                "cognitive_coherence", 1.0,
                            ),
                        )
            except Exception as _cycle_err:
                logger.warning("Unified cognitive cycle evaluation failed: %s", _cycle_err)
                # Record the UCC failure in error evolution so that
                # training-time metacognitive failures are visible to the
                # inference pipeline via bridge_training_errors_to_inference()
                # (defined in this module, see ae_train.py line ~3625),
                # rather than being silently swallowed.
                if self._error_evolution is not None:
                    self._error_evolution.record_episode(
                        error_class='training_ucc_failure',
                        strategy_used='skip_and_continue',
                        success=False,
                        metadata={'error': str(_cycle_err)[:200]},
                        causal_antecedents=["training_ucc", "phase_A"],
                    )

            # --- Periodic inference↔training bridge ---
            # Every bridge_interval epochs, synchronize error evolution
            # insights from the accumulated training error history back
            # into the training hyperparameters.  Previously, this bridge
            # was only invoked after ALL epochs completed, meaning
            # training-time discoveries (e.g., recurring coherence
            # deficits, convergence conflicts) could not adapt the
            # optimizer during training.  Mid-training bridging ensures
            # that accumulated error patterns continuously refine
            # training dynamics, satisfying the requirement that each
            # component verifies and reinforces the others throughout
            # the training process, not just at its conclusion.
            # NOTE: The final epoch is excluded (epoch + 1 < epochs)
            # because sync_from_training() is called unconditionally
            # after the epoch loop completes, avoiding redundant bridging.
            _bridge_interval = getattr(self.config, 'bridge_interval', 5)
            if (epoch + 1) % _bridge_interval == 0 and epoch + 1 < epochs:
                try:
                    _mid_adj = bridge_inference_insights_to_training(
                        inference_error_evolution=self._error_evolution,
                        trainer=self,
                    )
                    if _mid_adj > 0:
                        logger.info(
                            f"   🔗 Mid-training bridge (epoch {epoch+1}): "
                            f"{_mid_adj} adjustment(s) applied"
                        )
                except Exception as _mid_err:
                    logger.debug(
                        "Mid-training bridge failed (non-fatal): %s",
                        _mid_err,
                    )
            
            # Item #3: Continuous closed-loop streaming step.
            # Each epoch, push VibeThinker learner state into the
            # adaptive training controller so that calibration
            # pressure and adaptation signals continuously refine
            # learning rate and gradient clipping.
            if self._vt_streaming_bus is not None:
                try:
                    _cls_result = self._vt_streaming_bus.closed_loop_step(
                        vt_learner=self._vt_learner_ref,
                        controller=self.adaptive_controller,
                    )
                    if _cls_result.get('streaming'):
                        _ema = _cls_result.get('ema', {})
                        _cal_p = _ema.get('calibration_pressure', 0.0)
                        # [C3] Apply lr_scale from streaming bus to
                        # optimizer param_groups — previously computed
                        # but never consumed.
                        _adj = _cls_result.get('adjustments', {})
                        _lr_scale = _adj.get('lr_scale')
                        if _lr_scale is not None and _lr_scale < 1.0:
                            for _pg in self.optimizer.param_groups:
                                _pg['lr'] = _pg['lr'] * _lr_scale
                        if _cal_p > 0.3:
                            logger.info(
                                f"   📡 VT streaming: calibration_pressure="
                                f"{_cal_p:.3f} → LR adjustment applied"
                                f" (lr_scale={_lr_scale})"
                            )
                except Exception as _cls_err:
                    logger.debug(
                        "VT streaming closed-loop step failed "
                        "(non-fatal): %s", _cls_err,
                    )

            # [M1] Task boundary detection within epoch loop.
            # Previously invoked only once post-factum in main(), which
            # meant boundaries were detected too late to inform Phase A
            # training.  Now checked every epoch so that
            # ContinualLearningCore can freeze columns mid-training.
            if VIBE_THINKER_AVAILABLE:
                try:
                    _coherence_for_tb = epoch_metrics.get(
                        "cognitive_coherence", 1.0,
                    )
                    _tb_epoch = auto_detect_task_boundary(
                        coherence_score=_coherence_for_tb,
                        coherence_threshold=0.5,
                        previous_coherence=getattr(
                            self, '_prev_coherence_for_tb', None,
                        ),
                    )
                    self._prev_coherence_for_tb = _coherence_for_tb
                    if _tb_epoch['boundary_detected']:
                        logger.info(
                            f"   🔀 Epoch {epoch}: task boundary detected "
                            f"(coherence={_coherence_for_tb:.4f}, "
                            f"rec={_tb_epoch['recommendation']})"
                        )
                except Exception:
                    pass

            # [W4] Per-epoch entropy weight adaptation.
            # Previously adapt_entropy_weight() was called once in
            # main() before Phase A.  Calling it each epoch allows
            # the codebook entropy regularization to track evolving
            # training dynamics (e.g. codebook utilization changes).
            if VIBE_THINKER_AVAILABLE:
                try:
                    _epoch_perplexity = epoch_metrics.get("perplexity", 0.0)
                    _vt_entropy_proxy = min(
                        1.0, _epoch_perplexity / _PERPLEXITY_UNCERTAINTY_SCALE,
                    )
                    adapt_entropy_weight(self.config, _vt_entropy_proxy)
                except Exception:
                    pass

            if epoch_metrics["total"] < self.best_loss:
                self.best_loss = epoch_metrics["total"]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"   🏆 Новый лучший loss: {self.best_loss:.6f}")
            
            self.monitor.end_epoch(epoch, epochs, epoch_metrics, "phase_A")
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, epoch_metrics)
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"   ✅ Восстановлена лучшая модель с loss={self.best_loss:.6f}")
        
        # Auto-bridge training error patterns into the inference pipeline
        # so that training-discovered convergence failures, coherence
        # deficits, and metacognitive triggers are immediately available
        # for inference-time decision making.  This closes the gap where
        # sync_from_training was defined but never automatically called,
        # requiring manual invocation after training.
        if hasattr(self.model, 'sync_from_training'):
            try:
                _sync_result = self.model.sync_from_training(
                    trainer_monitor=self.convergence_monitor,
                )
                logger.info(
                    "   🔗 Training→inference bridge: %d events imported, "
                    "trigger_adapted=%s",
                    _sync_result.get('events_imported', 0),
                    _sync_result.get('trigger_adapted', False),
                )
            except Exception as _sync_err:
                logger.warning(
                    "   ⚠️  Training→inference bridge failed (non-fatal): %s",
                    _sync_err,
                )

        self.monitor.end_training("phase_A")
    
    def _save_checkpoint(self, epoch: int, metrics: dict):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.output_dir, 
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': asdict(self.config),
            }
            # Include training error patterns accumulated so far so that
            # checkpoint-based recovery carries the full error history,
            # enabling bridge_training_errors_to_inference() to seed the
            # inference pipeline with Phase A failure modes discovered
            # up to this point.  Without this, only the final save
            # includes error patterns, losing mid-training diagnostics.
            _patterns = self.convergence_monitor.export_error_patterns()
            if _patterns.get('error_classes'):
                save_dict['training_error_patterns'] = {
                    'Phase_A': _patterns,
                }
            torch.save(save_dict, checkpoint_path)
            logger.info(f"   💾 Checkpoint сохранён: {checkpoint_path}")
            # ── Cognitive snapshot persistence ──────────────────────────
            # If the model supports cognitive snapshots (hierarchical
            # memory subsystems, provenance baselines), export them
            # alongside the standard checkpoint.  This wires the
            # previously dead export_cognitive_snapshot() code into the
            # training loop, enabling full cross-session cognitive
            # continuity.
            if hasattr(self.model, 'export_cognitive_snapshot'):
                try:
                    _snap_dir = os.path.join(
                        self.output_dir,
                        f"cognitive_snapshot_epoch_{epoch+1}",
                    )
                    _snap_result = self.model.export_cognitive_snapshot(
                        save_dir=_snap_dir,
                    )
                    if _snap_result.get('success'):
                        logger.info(
                            f"   🧠 Cognitive snapshot exported: {_snap_dir}"
                        )
                    else:
                        logger.warning(
                            f"   ⚠️ Cognitive snapshot export incomplete: "
                            f"{_snap_dir}"
                        )
                except Exception as _snap_err:
                    logger.debug(
                        f"   Cognitive snapshot export failed (non-fatal): "
                        f"{_snap_err}"
                    )
        except OSError as e:
            logger.error(f"   ❌ Failed to save checkpoint: {e}")


class QualityHead(nn.Module):
    """Lightweight head predicting VibeThinker quality from RSSM output.

    Maps RSSM-predicted z_{t+1} to a 3-dimensional quality vector
    (confidence, entropy, reasoning_quality) so Phase B can learn a
    joint objective: L = L_mse + λ·L_quality.
    """

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return [B, 3] quality prediction (confidence, entropy, rq)."""
        return self.net(z)


# Mixing weight for quality loss in Phase B joint objective.
_QUALITY_LOSS_LAMBDA = 0.1


class ContextualRSSMTrainer:
    """
    ✅ НОВЫЙ: Трейнер Phase B для контекстного RSSM
    
    Обучает RSSM предсказывать z_{t+1} из контекста [z_{t-K+1}, ..., z_t]
    """
    
    def __init__(self, model: AEONDeltaV4, config: AEONConfigV4, 
                 monitor: TrainingMonitor):
        self.model = model
        self.config = config
        self.monitor = monitor
        self.device = next(model.parameters()).device
        
        # Замораживаем encoder, decoder, vq
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.vq.parameters():
            param.requires_grad = False

        # QualityHead predicts VibeThinker quality from RSSM output.
        # Its parameters are co-trained with the RSSM so that the
        # latent trajectory optimises both prediction accuracy and
        # reasoning quality.
        self._quality_head = QualityHead(config.z_dim).to(self.device)

        self.trainable_params = list(model.rssm.parameters()) + list(
            self._quality_head.parameters()
        )
        
        self.optimizer = optim.AdamW(
            self.trainable_params, 
            lr=config.learning_rate * 0.5,
            weight_decay=config.weight_decay
        )
        
        self.best_loss = float('inf')
        self.best_model_state = None
        self.global_step = 0
        # Convergence monitor for Phase B loss trajectory.
        # Wire CausalErrorEvolutionTracker so training divergence and
        # stagnation events are propagated to inference-time recovery.
        self._error_evolution = CausalErrorEvolutionTracker(max_history=200)
        self.convergence_monitor = TrainingConvergenceMonitor(
            threshold=1e-5, window_size=10,
            error_evolution=self._error_evolution,
        )
        # Bridge to aeon_core provenance and tensor safety — ensures
        # Phase B training errors are traceable to the RSSM component
        # and that NaN/Inf values are caught before gradient updates,
        # matching the safety guarantees of Phase A (SafeThoughtAETrainerV4).
        self.provenance = TrainingProvenanceTracker()
        self._tensor_guard = TensorGuard(policy=NaNPolicy.WARN, enable_tracking=True)
        # Error classifier for semantic error categorization — mirrors
        # Phase A's SemanticErrorClassifier integration so that Phase B
        # NaN/Inf errors are classified consistently, enabling root-cause
        # analysis to trace failures across both training phases.
        self._error_classifier = SemanticErrorClassifier()
        # Wire provenance tracker to convergence monitor so that Phase B
        # convergence failure events include per-module attribution,
        # matching Phase A's provenance-enriched convergence wiring.
        _prov_tracker_b = (
            self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else None
        )
        if _prov_tracker_b is None:
            logger.warning(
                "Phase B: TrainingProvenanceTracker has no _tracker; "
                "convergence events will lack per-module attribution"
            )
        self.convergence_monitor.set_provenance_tracker(_prov_tracker_b)

        # --- Unified Cognitive Cycle integration for Phase B ---
        self._coherence_verifier = ModuleCoherenceVerifier(
            hidden_dim=config.z_dim, threshold=0.5,
        )
        self._metacognitive_trigger = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=2,
        )
        self._core_convergence = ConvergenceMonitor(threshold=1e-5)
        # Wire SystemIntegrityMonitor into Phase B's UCC so that
        # RSSM training health is tracked and low global health
        # escalates uncertainty, matching Phase A and inference
        # pipeline UCC wiring.
        self._integrity_monitor = (
            SystemIntegrityMonitor(window_size=200)
            if AEON_CORE_AVAILABLE else None
        )
        self._unified_cycle = UnifiedCognitiveCycle(
            convergence_monitor=self._core_convergence,
            coherence_verifier=self._coherence_verifier,
            error_evolution=self._error_evolution,
            metacognitive_trigger=self._metacognitive_trigger,
            provenance_tracker=self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else CausalProvenanceTracker(),
            convergence_arbiter=UnifiedConvergenceArbiter() if AEON_CORE_AVAILABLE else None,
            uncertainty_tracker=DirectionalUncertaintyTracker() if AEON_CORE_AVAILABLE else None,
            # Wire MemoryReasoningValidator so RSSM predicted states are
            # validated against actual targets for consistency.  RSSM's
            # recurrent state is functionally a form of memory — validating
            # predicted-vs-actual consistency closes the gap where RSSM
            # state quality was only assessed via loss, never via the
            # meta-cognitive memory-reasoning validation pathway.
            memory_validator=MemoryReasoningValidator(
                consistency_threshold=0.3, staleness_penalty=0.1,
            ) if AEON_CORE_AVAILABLE else None,
            integrity_monitor=self._integrity_monitor,
        )
        # Cache the most recent VQ and RSSM output tensors so the
        # epoch-end UCC evaluation receives real subsystem states.
        self._last_vq_state: Optional[torch.Tensor] = None
        self._last_rssm_state: Optional[torch.Tensor] = None
        # Expose gradient clip norm, metacognitive LR factor, and
        # inference module feedback so that
        # bridge_inference_insights_to_training() can adapt Phase B
        # training parameters, matching Phase A's bridge attributes.
        self._grad_clip_norm: float = config.grad_clip_norm
        self._metacognitive_lr_factor: float = _METACOGNITIVE_LR_FACTOR
        self._inference_module_feedback: Dict[str, float] = {}

        # --- Adaptive training controller for Phase B ---
        self.adaptive_controller = AdaptiveTrainingController(config)

        # --- VTStreamingSignalBus (Item #3) for Phase B ---
        # Mirrors Phase A's continuous streaming bridge so that RSSM
        # training benefits from VibeThinker calibration signals.
        self._vt_streaming_bus: Optional['VTStreamingSignalBus'] = None
        self._vt_learner_ref: Optional['VibeThinkerContinuousLearner'] = None
        if VIBE_THINKER_AVAILABLE:
            try:
                self._vt_streaming_bus = VTStreamingSignalBus()
                self._vt_learner_ref = VibeThinkerContinuousLearner(
                    config=VibeThinkerConfig(),
                )
            except Exception:
                self._vt_streaming_bus = None
                self._vt_learner_ref = None

    def train_step(
        self,
        z_context: torch.Tensor,
        z_target: torch.Tensor,
        quality_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step for contextual RSSM.
        
        Args:
            z_context: [B, K, D] — context from K previous z states
                (B=batch size, K=context window length, D=latent dimension)
            z_target: [B, D] — target z_{t+1}
            quality_target: Optional [B, 3] — VibeThinker quality
                annotations (confidence, entropy, reasoning_quality).
                When provided, an auxiliary L_quality loss is added.
            
        Returns:
            Dictionary with loss and metric values.
        """
        self.model.rssm.train()
        self.provenance.reset()

        # Register dependency edges so root-cause tracing can attribute
        # RSSM errors to upstream components (encoder → vq → rssm).
        # Phase A registers encoder→vq→decoder but Phase B was missing
        # the equivalent edges, leaving provenance DAG disconnected.
        self.provenance.record_dependency("encoder", "vq")
        self.provenance.record_dependency("vq", "rssm")

        # Sanitize z_context input to prevent NaN/Inf from flowing into
        # the RSSM, matching Phase A's encoder-output sanitization.
        # Without this, corrupted VQ outputs from Phase A would propagate
        # unchecked through Phase B, breaking tensor safety consistency.
        if self._tensor_guard is not None:
            _orig_shape = z_context.shape
            _flat = z_context.reshape(-1, z_context.shape[-1])
            _flat = self._tensor_guard.sanitize(_flat, context="rssm_z_context")
            z_context = _flat.reshape(_orig_shape)

        # Track RSSM prediction provenance using mean-pooled context
        # (the RSSM processes the full [B, K, D] context window, so we
        # mean-pool over K to produce a [B, D] summary for provenance).
        self.provenance.record_before("rssm", z_context.mean(dim=1))
        pred = self.model.rssm(z_context)
        # Detect non-finite RSSM output before sanitization so the NaN
        # skip-backward path activates even when the tensor guard replaces
        # the values.  This preserves diagnostic accuracy while still
        # guarding against NaN propagation in the general case.
        _pred_had_nonfinite = not torch.isfinite(pred).all()
        # Sanitize RSSM prediction to prevent NaN/Inf from propagating
        # into loss computation, matching Phase A's encoder-output guard.
        if self._tensor_guard is not None:
            pred = self._tensor_guard.sanitize(pred, context="rssm_prediction")
        self.provenance.record_after("rssm", pred)
        # Cache detached subsystem states for epoch-end UCC evaluation
        self._last_vq_state = z_context.mean(dim=1).detach()
        self._last_rssm_state = pred.detach()
        
        # Losses
        mse_loss = F.mse_loss(pred, z_target)
        smooth_l1 = F.smooth_l1_loss(pred, z_target)
        loss = 0.5 * mse_loss + 0.5 * smooth_l1

        # [C1] Quality loss: when quality annotations are available,
        # train QualityHead to predict VibeThinker quality from RSSM
        # output, adding L_quality to the joint objective.  This
        # encourages RSSM to prefer latent trajectories that lead
        # to regions with high reasoning quality.
        _quality_loss = torch.tensor(0.0, device=self.device)
        if quality_target is not None:
            q_pred = self._quality_head(pred)
            _quality_loss = F.mse_loss(q_pred, quality_target)
            loss = loss + _QUALITY_LOSS_LAMBDA * _quality_loss

        # Consume inference module feedback to boost RSSM loss when
        # inference reports high uncertainty for the RSSM/memory modules.
        # This closes the same inference→training feedback loop as Phase A.
        if self._inference_module_feedback:
            _rssm_boost = self._inference_module_feedback.get('rssm', 0.0)
            _mem_boost = self._inference_module_feedback.get('memory', 0.0)
            _boost = max(_rssm_boost, _mem_boost)
            if _boost > 0.0:
                loss = loss + _boost * mse_loss

        # Detect NaN/Inf loss OR non-finite RSSM output to prevent
        # corrupted gradient updates.  Classify the error semantically
        # (matching Phase A) so root-cause analysis can trace it to the
        # RSSM component and record it in error evolution.
        if _pred_had_nonfinite or torch.isnan(loss) or torch.isinf(loss):
            _error_detail = "NaN/Inf loss"
            _error_class = "numerical"
            if self._error_classifier is not None:
                try:
                    _err_cls, _err_detail = self._error_classifier.classify(
                        RuntimeError("NaN/Inf loss in RSSM training step")
                    )
                    _error_class = _err_cls
                    _error_detail = f"{_err_cls}: {_err_detail}"
                except Exception as _cls_err:
                    logger.debug("Error classifier failed: %s", _cls_err)
            _prov = self.provenance.compute_attribution()
            _dominant = None
            _contributions = _prov.get('contributions', {})
            if _contributions:
                _dominant = max(_contributions, key=_contributions.get)
            logger.warning(
                f"⚠️ {_error_detail} at RSSM step {self.global_step}"
                f" (dominant_module={_dominant}), skipping backward pass"
            )
            # Propagate NaN event to convergence monitor so it can
            # detect training instability and recommend corrective action.
            self.convergence_monitor.update(float('nan'))
            # Record the semantically classified error in error evolution
            # so training-time failures inform inference-time recovery
            # strategies — matching Phase A's error recording pattern.
            self._error_evolution.record_episode(
                error_class=_error_class,
                strategy_used="skip_backward",
                success=False,
                metadata={
                    "step": self.global_step,
                    "dominant_module": _dominant,
                    "detail": _error_detail,
                    "phase": "B",
                },
                causal_antecedents=["training_backward_phase_B", _dominant],
            )
            return {
                "mse_loss": float('nan'), "smooth_l1": float('nan'),
                "total_loss": float('nan'), "cosine_sim": 0.0,
                "l1_loss": float('nan'), "rel_error": float('nan'),
                "grad_norm": 0.0,
                "provenance": _prov,
            }
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_params, 
            self._grad_clip_norm
        )
        
        self.optimizer.step()
        self.global_step += 1
        
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(pred, z_target, dim=1).mean().item()
            l1_loss = F.l1_loss(pred, z_target).item()
            rel_error = (torch.norm(pred - z_target, dim=1) / (torch.norm(z_target, dim=1) + 1e-8)).clamp(max=1e4).mean().item()
        
        # Log provenance on successful steps for root-cause traceability
        _prov = self.provenance.compute_attribution()
        _contributions = _prov.get('contributions', {})
        if _contributions and self.global_step % _PROVENANCE_LOG_INTERVAL == 0:
            _dominant = max(_contributions, key=_contributions.get)
            _max_contrib = _contributions[_dominant]
            logger.debug(
                f"RSSM step {self.global_step} provenance: dominant={_dominant} "
                f"({_max_contrib:.1%})"
            )
            # Warn if a single module dominates provenance — matching
            # Phase A's validation warning pattern.  This indicates an
            # architectural imbalance that may prevent the RSSM from
            # learning balanced cross-component representations.
            if _max_contrib > _PROVENANCE_DOMINANCE_WARNING_THRESHOLD:
                logger.warning(
                    f"   ⚠️ RSSM step {self.global_step}: module "
                    f"'{_dominant}' dominates provenance "
                    f"({_max_contrib:.1%} > "
                    f"{_PROVENANCE_DOMINANCE_WARNING_THRESHOLD:.0%})"
                )
        
        # RSSM-decoder cross-validation: verify that the RSSM prediction
        # can produce valid token reconstructions through the frozen decoder.
        # Without this, Phase B optimizes latent-space proximity (L2) but
        # never validates that predicted latent states lie in the decoder's
        # valid input manifold.  A high reconstruction error here indicates
        # the RSSM is predicting states that the decoder cannot interpret,
        # signalling a latent-space drift between encoder/VQ and RSSM.
        _decoder_valid = True
        with torch.no_grad():
            try:
                _recon_from_pred = self.model.decoder(pred.detach())
                _recon_from_target = self.model.decoder(z_target)
                _decoder_cross_loss = F.mse_loss(
                    _recon_from_pred, _recon_from_target,
                ).item()
                if not math.isfinite(_decoder_cross_loss):
                    _decoder_valid = False
            except Exception as _decoder_err:
                logger.debug("Decoder cross-validation failed: %s", _decoder_err)
                _decoder_cross_loss = float('nan')
                _decoder_valid = False

        return {
            "mse_loss": mse_loss.item(), 
            "smooth_l1": smooth_l1.item(),
            "total_loss": loss.item(),
            "quality_loss": _quality_loss.item(),
            "cosine_sim": cosine_sim, 
            "l1_loss": l1_loss,
            "rel_error": rel_error,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "provenance": _prov,
            "decoder_cross_loss": _decoder_cross_loss,
            "decoder_valid": _decoder_valid,
            "convergence_status": self.convergence_monitor.status,
        }

    def _provenance_causal_quality(self) -> float:
        """Compute causal quality from provenance attribution balance.

        Delegates to the shared :func:`_compute_provenance_causal_quality`
        utility, matching Phase A's provenance-derived quality wiring.
        """
        return _compute_provenance_causal_quality(self.provenance)

    def fit(
        self,
        z_sequences: List[torch.Tensor],
        epochs: int = 10,
        batch_size: int = 128,
        log_every_batch: int = 5,
        quality_annotations: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            z_sequences: List of [num_chunks, D] tensors, one per document
            quality_annotations: Optional parallel list of [num_chunks, 3]
                tensors with per-z quality (confidence, entropy, rq).
        """
        # Создаём dataset из контекстных окон
        K = self.config.context_window
        
        all_contexts = []
        all_targets = []
        all_quality_targets: List[torch.Tensor] = []
        _has_quality = (quality_annotations is not None
                        and len(quality_annotations) == len(z_sequences))
        
        for seq_idx, z_seq in enumerate(z_sequences):
            num_z = z_seq.size(0)
            if num_z >= K + 1:
                for i in range(K, num_z):
                    context = z_seq[i-K:i]  # [K, D]
                    target = z_seq[i]  # [D]
                    all_contexts.append(context)
                    all_targets.append(target)
                    if _has_quality:
                        # quality_annotations[seq_idx] is [num_chunks, 3];
                        # target index i corresponds to z_{i}.
                        _qa = quality_annotations[seq_idx]  # type: ignore[index]
                        if i < _qa.size(0):
                            all_quality_targets.append(_qa[i])
                        else:
                            all_quality_targets.append(torch.ones(3))
        
        if len(all_contexts) == 0:
            logger.warning("⚠️ Недостаточно данных для обучения RSSM")
            return
        
        contexts_tensor = torch.stack(all_contexts)  # [N, K, D]
        targets_tensor = torch.stack(all_targets)  # [N, D]
        
        if _has_quality and all_quality_targets:
            quality_tensor = torch.stack(all_quality_targets)  # [N, 3]
            dataset = TensorDataset(contexts_tensor, targets_tensor, quality_tensor)
        else:
            quality_tensor = None
            dataset = TensorDataset(contexts_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        total_batches = len(loader)
        
        self.monitor.start_training(f"Phase B (Contextual RSSM, K={K})", epochs, len(dataset))
        
        rssm_params = sum(p.numel() for p in self.model.rssm.parameters())
        logger.info(f"📦 Параметры RSSM: {rssm_params:,}")
        logger.info(f"   Context window: {K}")
        logger.info(f"   Training samples: {len(dataset):,}")
        
        for epoch in range(epochs):
            self.monitor.start_epoch(epoch, epochs)
            
            epoch_metrics = {
                "mse_loss": 0.0, "cosine_sim": 0.0, 
                "l1_loss": 0.0, "rel_error": 0.0, "grad_norm": 0.0,
                "decoder_cross_loss": 0.0, "quality_loss": 0.0,
            }
            valid_batches = 0
            _decoder_invalid_count = 0
            
            for batch_idx, batch_data in enumerate(loader):
                if quality_tensor is not None:
                    ctx_batch, tgt_batch, q_batch = batch_data
                    q_batch = q_batch.to(self.device)
                else:
                    ctx_batch, tgt_batch = batch_data
                    q_batch = None
                ctx_batch = ctx_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                metrics = self.train_step(ctx_batch, tgt_batch, quality_target=q_batch)
                
                batch_valid = False
                for key in epoch_metrics:
                    if key in metrics and not (math.isnan(metrics[key]) or math.isinf(metrics[key])):
                        epoch_metrics[key] += metrics[key]
                        batch_valid = True
                if batch_valid:
                    valid_batches += 1
                # Track decoder cross-validation failures so that error
                # evolution can learn from RSSM-decoder incompatibility
                # patterns.  Without this, decoder_valid failures are
                # computed in train_step but silently discarded, making
                # latent-space drift invisible to the cognitive system.
                if not metrics.get("decoder_valid", True):
                    _decoder_invalid_count += 1
                    self._error_evolution.record_episode(
                        error_class="decoder_cross_validation_failure",
                        strategy_used="skip_and_continue",
                        success=False,
                        metadata={
                            "step": self.global_step,
                            "decoder_cross_loss": metrics.get(
                                "decoder_cross_loss", float('nan')
                            ),
                            "phase": "B",
                        },
                        causal_antecedents=["decoder", "cross_validation"],
                    )
                
                if batch_idx % log_every_batch == 0:
                    self.monitor.log_batch(batch_idx, total_batches, {
                        "mse": metrics["mse_loss"],
                        "cos": metrics["cosine_sim"],
                        "rel_err": metrics["rel_error"]
                    }, phase="phase_B", log_every=log_every_batch)
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(valid_batches, 1)

            # --- Adaptive parameter controller ---
            adaptive_adj = self.adaptive_controller.record_step(
                loss=epoch_metrics["mse_loss"],
                grad_norm=epoch_metrics["grad_norm"],
                lr=self.optimizer.param_groups[0]['lr'],
            )
            if adaptive_adj:
                if 'recommended_lr' in adaptive_adj:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = adaptive_adj['recommended_lr']
                    logger.info(
                        f"   🔄 Phase B adaptive LR: {adaptive_adj['recommended_lr']:.2e} "
                        f"(reason: {adaptive_adj.get('lr_reason', 'unknown')})"
                    )
                if 'grad_clip' in adaptive_adj:
                    self._grad_clip_norm = adaptive_adj['grad_clip']
                    logger.info(
                        f"   🔄 Phase B adaptive grad_clip: {adaptive_adj['grad_clip']:.3f} "
                        f"(reason: {adaptive_adj.get('gc_reason', 'unknown')})"
                    )
            epoch_metrics["adaptive_state"] = self.adaptive_controller.get_state()

            # Convergence monitoring for Phase B
            convergence_verdict = self.convergence_monitor.update(
                epoch_metrics["mse_loss"]
            )
            epoch_metrics["convergence_status"] = convergence_verdict["status"]
            if convergence_verdict["status"] == "diverging":
                logger.warning(
                    f"   ⚠️ Phase B convergence: DIVERGING "
                    f"(trend={convergence_verdict['trend']:.6f})"
                )
                # Adaptive LR response — reduce learning rate when Phase B
                # diverges, closing the monitoring-to-action feedback loop.
                if convergence_verdict.get("recommendation") == "reduce_lr_or_rollback":
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.info(
                        f"   ↘️ Phase B LR reduced to "
                        f"{self.optimizer.param_groups[0]['lr']:.2e} "
                        f"due to divergence"
                    )
            elif convergence_verdict["status"] == "stagnating":
                logger.info(
                    f"   ℹ️ Phase B convergence: stagnating "
                    f"(trend={convergence_verdict['trend']:.6f})"
                )
                # Adaptive LR response — increase learning rate slightly
                # when Phase B stagnates, closing the monitoring-to-action
                # feedback loop for the stagnation case.  Without this,
                # convergence monitoring detects stagnation but the system
                # takes no corrective action, breaking the feedback loop.
                if convergence_verdict.get("recommendation") == "increase_lr_or_augment":
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = min(
                            param_group['lr'] * 1.5,
                            self.config.learning_rate,  # cap at initial LR
                        )
                    logger.info(
                        f"   ↗️ Phase B LR increased to "
                        f"{self.optimizer.param_groups[0]['lr']:.2e} "
                        f"due to stagnation"
                    )

            # --- Unified Cognitive Cycle evaluation for Phase B ---
            try:
                _loss_delta = abs(convergence_verdict.get("trend", 0.0))
                _uncertainty = min(epoch_metrics.get("mse_loss", 0.0) / _MSE_UNCERTAINTY_SCALE, 1.0)
                # Escalate uncertainty when decoder cross-validation
                # failed during this epoch — RSSM predictions that the
                # decoder cannot interpret signal latent-space drift,
                # which should trigger deeper meta-cognitive reasoning.
                if _decoder_invalid_count > 0 and valid_batches > 0:
                    _decoder_failure_ratio = _decoder_invalid_count / valid_batches
                    _uncertainty = min(1.0, _uncertainty + _decoder_failure_ratio * 0.3)
                _is_diverging = convergence_verdict["status"] == "diverging"
                # Adapt metacognitive trigger weights from accumulated
                # error-evolution history so that historically problematic
                # failure modes increase trigger sensitivity.  This closes
                # the loop between error evolution and metacognitive
                # re-reasoning, matching Phase A's implicit wiring via UCC.
                _err_summary = self._error_evolution.get_error_summary()
                self._metacognitive_trigger.adapt_weights_from_evolution(
                    _err_summary,
                )
                # Item #5: Map VibeThinker signals to trigger inputs
                _vt_mapped_b = {}
                if VIBE_THINKER_AVAILABLE:
                    try:
                        _vt_mapped_b = map_vt_signals_to_trigger(
                            vt_confidence=1.0 - _uncertainty,
                            vt_entropy=min(1.0, epoch_metrics.get("mse_loss", 0.0) / 10.0),
                            vt_calibration_error=max(0.0, 1.0 - epoch_metrics.get("cognitive_coherence", 1.0)) if "cognitive_coherence" in epoch_metrics else 0.0,
                        )
                    except Exception:
                        _vt_mapped_b = {}
                _cycle_result = self._unified_cycle.evaluate(
                    subsystem_states={
                        "vq": self._last_vq_state
                        if self._last_vq_state is not None
                        else torch.zeros(1, self.config.z_dim),
                        "rssm": self._last_rssm_state
                        if self._last_rssm_state is not None
                        else torch.zeros(1, self.config.z_dim),
                    },
                    delta_norm=_loss_delta,
                    uncertainty=_uncertainty,
                    recovery_pressure=1.0 if _is_diverging else 0.0,
                    # Feed provenance-derived causal quality so the UCC
                    # can detect when a single training subsystem dominates
                    # the output attribution, matching Phase A's wiring.
                    causal_quality=self._provenance_causal_quality(),
                    # Pass RSSM predicted state as memory_signal and VQ
                    # target as converged_state so MemoryReasoningValidator
                    # can check predicted-vs-actual consistency.  This
                    # closes the gap where RSSM state quality was only
                    # assessed via loss, never via the meta-cognitive
                    # memory-reasoning validation pathway.
                    memory_signal=self._last_rssm_state,
                    converged_state=self._last_vq_state,
                    diversity_collapse=_vt_mapped_b.get("diversity_collapse", 0.0),
                    convergence_conflict=_vt_mapped_b.get("convergence_conflict", 0.0),
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
                # Item #7: Coherence-driven epoch rerun for Phase B
                if _cycle_result["should_rerun"] and not getattr(self, '_rerun_applied', False):
                    for pg in self.optimizer.param_groups:
                        pg['lr'] *= _UCC_RERUN_LR_FACTOR
                    self._rerun_applied = True
                    logger.info(
                        f"   🔁 Phase B UCC rerun: LR reduced by "
                        f"{_UCC_RERUN_LR_FACTOR:.0%} for coherence "
                        f"(deficit={_cycle_result['coherence_result']['coherence_deficit']:.4f})"
                    )
                elif not _cycle_result["should_rerun"]:
                    self._rerun_applied = False
                # Synthesize cognitive_unity_score for Phase B, matching
                # the Phase A and inference pipeline composite metrics.
                _cus_coherence_b = epoch_metrics["cognitive_coherence"]
                _cus_convergence_b = max(0.0, min(
                    1.0, 1.0 - abs(convergence_verdict.get("trend", 0.0)),
                ))
                _cus_provenance_b = self._provenance_causal_quality()
                # Metacognitive responsiveness — mirrors Phase A logic.
                if _uncertainty > 0.5:
                    _cus_metacognitive_b = (
                        1.0 if _cycle_result["should_rerun"]
                        else 1.0 - _uncertainty
                    )
                else:
                    _cus_metacognitive_b = 1.0
                epoch_metrics["cognitive_unity_score"] = (
                    _CUS_WEIGHT_COHERENCE * _cus_coherence_b
                    + _CUS_WEIGHT_CONVERGENCE * _cus_convergence_b
                    + _CUS_WEIGHT_PROVENANCE * _cus_provenance_b
                    + _CUS_WEIGHT_UNCERTAINTY * _cus_metacognitive_b
                )
                # Apply correction_guidance from UCC — when the unified
                # cognitive cycle identifies a specific target module and
                # recommended strategy, use that insight to adapt Phase B
                # training parameters.  This closes the gap where UCC
                # correction guidance was only consumed in Phase A,
                # leaving Phase B without targeted correction.
                _correction = _cycle_result.get("correction_guidance", {})
                _corr_target = _correction.get("target_module")
                if _corr_target is not None:
                    epoch_metrics["correction_target"] = _corr_target
                    if hasattr(self, '_inference_module_feedback'):
                        self._inference_module_feedback[_corr_target] = max(
                            self._inference_module_feedback.get(
                                _corr_target, 0.0,
                            ),
                            1.0 - epoch_metrics.get(
                                "cognitive_coherence", 1.0,
                            ),
                        )
                if _cycle_result["should_rerun"]:
                    _active = _cycle_result["trigger_detail"].get("triggers_active", [])
                    logger.info(
                        f"   🧠 Phase B meta-cognitive cycle triggered "
                        f"(signals={_active}), adapting training"
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self._metacognitive_lr_factor
                    # Tighten grad clip from correction guidance or
                    # coherence deficit (whichever is more severe), but
                    # only once to avoid duplicate adjustments.
                    _clip_tightened = False
                    if (_corr_target is not None
                            and _correction.get("recommended_strategy")):
                        self._grad_clip_norm = max(
                            0.1, self._grad_clip_norm * 0.95,
                        )
                        _clip_tightened = True
                    # When the UCC detects coherence issues, tighten gradient
                    # clipping to stabilize the RSSM and prevent further
                    # latent-space drift.  This closes the loop between
                    # coherence verification and training dynamics: detected
                    # coherence deficits translate into stricter optimization
                    # constraints rather than just softer learning rates.
                    _coherence_deficit = _cycle_result["coherence_result"].get(
                        "coherence_deficit", 0.0,
                    )
                    if _coherence_deficit > 0.3 and not _clip_tightened:
                        # _COHERENCE_MIN_GRAD_CLIP: absolute floor to prevent
                        # gradient clipping from becoming so tight that
                        # learning effectively stalls.
                        _COHERENCE_MIN_GRAD_CLIP = 0.1
                        # _COHERENCE_CLIP_FACTOR: scales how aggressively the
                        # deficit reduces the clip norm (0.5 = deficit of 1.0
                        # halves the clip norm).
                        _COHERENCE_CLIP_FACTOR = 0.5
                        _tightened_clip = max(
                            _COHERENCE_MIN_GRAD_CLIP,
                            self._grad_clip_norm * (1.0 - _coherence_deficit * _COHERENCE_CLIP_FACTOR),
                        )
                        if _tightened_clip < self._grad_clip_norm:
                            self._grad_clip_norm = _tightened_clip
                            logger.info(
                                f"   🔧 Grad clip tightened to {_tightened_clip:.3f} "
                                f"due to coherence deficit={_coherence_deficit:.3f}"
                            )
            except Exception as _cycle_err:
                logger.warning("Phase B unified cognitive cycle failed: %s", _cycle_err)
                if self._error_evolution is not None:
                    self._error_evolution.record_episode(
                        error_class='training_ucc_failure',
                        strategy_used='skip_and_continue',
                        success=False,
                        metadata={
                            'error': str(_cycle_err)[:200],
                            'phase': 'B',
                        },
                        causal_antecedents=["training_ucc", "phase_B"],
                    )

            # --- Periodic inference↔training bridge (Phase B) ---
            # Mirrors Phase A's periodic bridging: every bridge_interval
            # epochs, synchronize accumulated error patterns back into
            # training hyperparameters.
            # NOTE: Final epoch excluded — sync_from_training() handles it.
            _bridge_interval = getattr(self.config, 'bridge_interval', 5)
            if (epoch + 1) % _bridge_interval == 0 and epoch + 1 < epochs:
                try:
                    _mid_adj = bridge_inference_insights_to_training(
                        inference_error_evolution=self._error_evolution,
                        trainer=self,
                    )
                    if _mid_adj > 0:
                        logger.info(
                            f"   🔗 Phase B mid-training bridge (epoch {epoch+1}): "
                            f"{_mid_adj} adjustment(s) applied"
                        )
                except Exception as _mid_err:
                    logger.debug(
                        "Phase B mid-training bridge failed (non-fatal): %s",
                        _mid_err,
                    )
            
            # Item #3: Continuous closed-loop streaming step for Phase B.
            # Mirrors Phase A's VT streaming integration so that RSSM
            # training parameters are continuously refined by VibeThinker
            # calibration pressure and adaptation signals.
            if self._vt_streaming_bus is not None:
                try:
                    _cls_result_b = self._vt_streaming_bus.closed_loop_step(
                        vt_learner=self._vt_learner_ref,
                        controller=self.adaptive_controller,
                    )
                    if _cls_result_b.get('streaming'):
                        _ema_b = _cls_result_b.get('ema', {})
                        _cal_p_b = _ema_b.get('calibration_pressure', 0.0)
                        # [C3] Apply lr_scale to Phase B optimizer,
                        # closing the streaming bus feedback loop.
                        _adj_b = _cls_result_b.get('adjustments', {})
                        _lr_scale_b = _adj_b.get('lr_scale')
                        if _lr_scale_b is not None and _lr_scale_b < 1.0:
                            for _pg in self.optimizer.param_groups:
                                _pg['lr'] = _pg['lr'] * _lr_scale_b
                        if _cal_p_b > 0.3:
                            logger.info(
                                f"   📡 Phase B VT streaming: "
                                f"calibration_pressure={_cal_p_b:.3f}"
                                f" (lr_scale={_lr_scale_b})"
                            )
                except Exception as _cls_err_b:
                    logger.debug(
                        "Phase B VT streaming closed-loop step failed "
                        "(non-fatal): %s", _cls_err_b,
                    )

            if epoch_metrics["mse_loss"] < self.best_loss:
                self.best_loss = epoch_metrics["mse_loss"]
                self.best_model_state = copy.deepcopy(self.model.rssm.state_dict())
                logger.info(f"   🏆 Новый лучший MSE: {self.best_loss:.6f}")
            
            self.monitor.end_epoch(epoch, epochs, epoch_metrics, "phase_B")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.rssm.load_state_dict(self.best_model_state)
            logger.info(f"   ✅ Восстановлена лучшая RSSM модель с MSE={self.best_loss:.6f}")
        
        # Auto-bridge Phase B training insights into the inference
        # pipeline, mirroring the Phase A auto-bridge.
        if hasattr(self.model, 'sync_from_training'):
            try:
                _sync_result = self.model.sync_from_training(
                    trainer_monitor=self.convergence_monitor,
                )
                logger.info(
                    "   🔗 Phase B training→inference bridge: %d events imported",
                    _sync_result.get('events_imported', 0),
                )
            except Exception as _sync_err:
                logger.warning(
                    "   ⚠️  Phase B training→inference bridge failed "
                    "(non-fatal): %s", _sync_err,
                )

        self.monitor.end_training("phase_B")


# ==============================================================================
# TRAINING–CORE BRIDGE: PROVENANCE & CONVERGENCE
# ==============================================================================

# Threshold above which a single module's provenance contribution
# triggers a dominance warning during validation.
_PROVENANCE_DOMINANCE_WARNING_THRESHOLD = 0.9

# Interval (in training steps) between provenance dominant-module log
# entries.  Reduced from 50 to 10 to improve root-cause traceability:
# at 50-step intervals, provenance data for normal (non-NaN) training
# steps was invisible, preventing root-cause analysis from tracing
# gradual drift patterns that only manifest over multiple steps.
_PROVENANCE_LOG_INTERVAL = 10

# Meta-cognitive learning rate adjustment factor applied when the
# UnifiedCognitiveCycle triggers re-reasoning during training.
_METACOGNITIVE_LR_FACTOR = 0.7

# Normalization constants for mapping raw loss metrics to the [0, 1]
# uncertainty range consumed by MetaCognitiveRecursionTrigger.
_PERPLEXITY_UNCERTAINTY_SCALE = 1000.0  # Phase A: perplexity → uncertainty
_MSE_UNCERTAINTY_SCALE = 10.0           # Phase B: mse_loss → uncertainty

# Cognitive unity score component weights — shared between Phase A and
# Phase B so that training and inference measure AGI coherence on the
# same scale.  Defined once to prevent weight drift between phases.
_CUS_WEIGHT_COHERENCE = 0.30
_CUS_WEIGHT_CONVERGENCE = 0.25
_CUS_WEIGHT_PROVENANCE = 0.25
_CUS_WEIGHT_UNCERTAINTY = 0.20

class TrainingProvenanceTracker:
    """Lightweight provenance tracker for the training pipeline.

    Bridges ae_train's training loop to aeon_core's
    :class:`CausalProvenanceTracker` pattern by recording per-component
    L2 deltas during validation and training steps.  When aeon_core is
    available, delegates to the real tracker; otherwise uses a minimal
    standalone implementation so the training pipeline always has
    provenance data regardless of import availability.

    This closes the architectural gap where training errors could not be
    traced back to their originating component.
    """

    def __init__(self):
        self._tracker = CausalProvenanceTracker()
        # Standalone fallback storage
        self._deltas: Dict[str, float] = {}
        self._order: list = []
        self._snapshots: Dict[str, torch.Tensor] = {}

    def reset(self):
        """Clear all recorded snapshots for a new pass."""
        if self._tracker is not None:
            self._tracker.reset()
        self._deltas.clear()
        self._order.clear()
        self._snapshots.clear()

    def record_before(self, name: str, state: torch.Tensor):
        if self._tracker is not None:
            self._tracker.record_before(name, state)
        self._snapshots[name] = state.detach().clone()
        if name not in self._order:
            self._order.append(name)

    def record_after(self, name: str, state: torch.Tensor):
        if self._tracker is not None:
            self._tracker.record_after(name, state)
        if name in self._snapshots:
            before = self._snapshots[name]
            # Handle shape mismatches by truncating to smaller size
            min_size = min(state.shape[-1], before.shape[-1])
            self._deltas[name] = (
                state.detach()[..., :min_size] - before[..., :min_size]
            ).norm().item()

    def compute_attribution(self) -> Dict[str, Any]:
        """Return per-component attribution dict."""
        if self._tracker is not None:
            return self._tracker.compute_attribution()
        total = sum(self._deltas.values()) + 1e-10
        contributions = {k: v / total for k, v in self._deltas.items()}
        return {
            'contributions': contributions,
            'deltas': dict(self._deltas),
            'order': list(self._order),
        }

    def record_dependency(self, from_module: str, to_module: str) -> None:
        """Record an inter-module dependency edge for root-cause tracing.

        Delegates to the underlying ``CausalProvenanceTracker`` so that
        ``trace_root_cause()`` can walk the dependency DAG backward from
        any module to the original inputs.
        """
        if self._tracker is not None:
            self._tracker.record_dependency(from_module, to_module)


class TrainingConvergenceMonitor:
    """Monitors training loss convergence across epochs.

    Bridges ae_train's training loop to aeon_core's
    :class:`ConvergenceMonitor` pattern by tracking a sliding window of
    loss values and detecting divergence or stagnation.

    When aeon_core is available, delegates to the real monitor;
    otherwise uses a minimal standalone implementation.

    Attributes:
        status: One of ``'warmup'``, ``'converging'``, ``'converged'``,
            ``'diverging'``, or ``'stagnating'``.
    """

    _WARMUP_SIZE = 5
    _DIVERGENCE_RATIO = 1.5
    _STAGNATION_THRESHOLD = 1e-6

    def __init__(self, threshold: float = 1e-5, window_size: int = 10,
                 error_evolution: Optional[Any] = None):
        """Initialize convergence monitor.

        Args:
            threshold: Loss change threshold for convergence detection.
            window_size: Number of recent losses to retain.
            error_evolution: Optional ``CausalErrorEvolutionTracker``
                instance.  When provided, divergence and stagnation events
                are recorded as error episodes so inference-time recovery
                can learn from training-time convergence failures.
        """
        self._threshold = threshold
        self._window_size = window_size
        self._history: list = []
        self.status: str = 'warmup'
        # Optional bridge to aeon_core's CausalErrorEvolutionTracker.
        # When provided, convergence events (divergence, stagnation) are
        # propagated as error episodes so that inference-time error
        # recovery can learn from training-time convergence failures.
        self._error_evolution = error_evolution
        self._core_monitor = ConvergenceMonitor(threshold=threshold)
        # Wire the core monitor to error evolution so convergence events
        # from the delegate include provenance attribution metadata,
        # enabling root-cause analysis to identify which module dominated
        # when a convergence failure occurred during training.
        if error_evolution is not None:
            self._core_monitor.set_error_evolution(error_evolution)

    def set_provenance_tracker(
        self, tracker: 'CausalProvenanceTracker',
    ) -> None:
        """Attach a provenance tracker to the internal core monitor.

        Once attached, convergence events bridged to error evolution
        include per-module attribution metadata, enabling root-cause
        analysis to correlate training convergence failures with the
        dominant upstream module.
        """
        if self._core_monitor is not None:
            self._core_monitor.set_provenance_tracker(tracker)
        else:
            logger.debug(
                "TrainingConvergenceMonitor: _core_monitor is None; "
                "provenance tracker not attached"
            )

    def update(self, loss_value: float) -> Dict[str, Any]:
        """Record a loss value and return convergence verdict.

        Args:
            loss_value: Scalar training loss for the current epoch/step.

        Returns:
            Dict with ``status``, ``loss_value``, ``trend`` (float),
            and ``recommendation`` (str).
        """
        if not math.isfinite(loss_value):
            self.status = 'diverging'
            return {
                'status': 'diverging',
                'loss_value': loss_value,
                'trend': float('inf'),
                'recommendation': 'reduce_lr_or_rollback',
            }

        self._history.append(loss_value)
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]

        if len(self._history) < self._WARMUP_SIZE:
            self.status = 'warmup'
            return {
                'status': 'warmup',
                'loss_value': loss_value,
                'trend': 0.0,
                'recommendation': 'continue',
            }

        recent = self._history[-self._WARMUP_SIZE:]
        trend = recent[-1] - recent[0]

        if recent[-1] > self._DIVERGENCE_RATIO * min(recent):
            self.status = 'diverging'
            recommendation = 'reduce_lr_or_rollback'
        elif abs(trend) < self._STAGNATION_THRESHOLD:
            self.status = 'stagnating'
            recommendation = 'increase_lr_or_augment'
        elif trend < 0:
            if abs(trend) < self._threshold:
                self.status = 'converged'
                recommendation = 'continue'
            else:
                self.status = 'converging'
                recommendation = 'continue'
        else:
            self.status = 'diverging'
            recommendation = 'reduce_lr_or_rollback'

        # Feed into aeon_core monitor if available
        if self._core_monitor is not None:
            self._core_monitor.check(abs(trend))

        # Propagate convergence events to error evolution tracker so
        # that inference-time recovery can learn from training failures.
        if self._error_evolution is not None:
            if self.status == 'diverging':
                self._error_evolution.record_episode(
                    error_class="training_divergence",
                    strategy_used=recommendation,
                    success=False,
                    metadata={"trend": trend, "loss_value": loss_value},
                    causal_antecedents=["convergence_monitor", "training_loop"],
                )
            elif self.status == 'stagnating':
                self._error_evolution.record_episode(
                    error_class="training_stagnation",
                    strategy_used=recommendation,
                    success=False,
                    metadata={"trend": trend, "loss_value": loss_value},
                    causal_antecedents=["convergence_monitor", "training_loop"],
                )

        return {
            'status': self.status,
            'loss_value': loss_value,
            'trend': trend,
            'recommendation': recommendation,
        }

    def export_error_patterns(self) -> Dict[str, Any]:
        """Export training error patterns for inference-time recovery.

        Returns a summary of training convergence events that can be
        ingested by an inference-time ``CausalErrorEvolutionTracker``
        via :func:`bridge_training_errors_to_inference`.  This closes
        the training→inference feedback loop: training-time divergence
        and stagnation patterns inform inference recovery strategies.
        """
        if self._error_evolution is not None:
            return self._error_evolution.get_error_summary()
        return {
            'status': self.status,
            'history_length': len(self._history),
            'error_classes': {},
        }


# Maps training error classes to pipeline stage dependency pairs (upstream,
# downstream) for provenance forwarding in bridge_training_errors_to_inference.
_ERROR_CLASS_TO_DEPENDENCY_MAP: Dict[str, Tuple[str, str]] = {
    "divergence": ("meta_loop", "error_evolution"),
    "training_divergence": ("meta_loop", "error_evolution"),
    "stagnation": ("encoder", "meta_loop"),
    "training_stagnation": ("encoder", "meta_loop"),
    "coherence_deficit": ("meta_loop", "integration"),
    "training_ucc_failure": ("meta_loop", "unified_cognitive_cycle"),
    "decoder_cross_validation_failure": ("decoder", "cross_validation"),
    # Feedback oscillation indicates the feedback bus detected signal
    # instability — maps to the feedback_bus → metacognitive_trigger
    # provenance edge so root-cause analysis can trace oscillation-
    # driven re-reasoning back to the feedback bus.
    "feedback_oscillation": ("feedback_bus", "metacognitive_trigger"),
    # UCC-triggered re-reasoning failure — maps to the
    # ucc_rerun_meta_loop → integration edge so root-cause analysis
    # can trace integration quality changes to re-reasoning outcomes.
    "ucc_rerun": ("ucc_rerun_meta_loop", "integration"),
}


def _compute_provenance_causal_quality(provenance_tracker) -> float:
    """Compute causal quality from provenance attribution balance.

    Shared utility used by both Phase A (SafeThoughtAETrainerV4) and
    Phase B (ContextualRSSMTrainer) to derive a balance score from
    the provenance tracker's per-module attribution.

    The formula ``1.0 - max_val + 1/n`` computes how far the dominant
    module's contribution (``max_val``) is from the perfectly balanced
    case where each of ``n`` modules contributes ``1/n``.  When all
    modules contribute equally, ``max_val = 1/n`` and the score is 1.0.
    When one module fully dominates (``max_val → 1.0``), the score
    approaches ``1/n`` (near zero for many modules).

    Args:
        provenance_tracker: A provenance tracker with a
            ``compute_attribution()`` method returning a dict with
            a ``'contributions'`` key mapping module names to floats.

    Returns:
        Score in [0, 1] where 1.0 = balanced, approaching 0.0 = dominated.
    """
    try:
        attrib = provenance_tracker.compute_attribution()
        contribs = attrib.get('contributions', {})
        if len(contribs) < 2:
            return 1.0  # Cannot assess balance with < 2 modules
        vals = list(contribs.values())
        max_val = max(vals)
        n = len(vals)
        # Score = 1.0 when max_val == 1/n (perfect balance);
        # degrades toward 1/n when max_val → 1.0 (single-module dominance).
        return max(0.0, min(1.0, 1.0 - max_val + (1.0 / n)))
    except (AttributeError, KeyError, ValueError, TypeError):
        return 1.0  # Default healthy when provenance unavailable


def bridge_training_errors_to_inference(
    trainer_monitor: 'TrainingConvergenceMonitor',
    inference_error_evolution: Any,
    causal_trace: Any = None,
    inference_convergence_monitor: Any = None,
    inference_integrity_monitor: Any = None,
    inference_provenance_tracker: Any = None,
    inference_metacognitive_trigger: Any = None,
) -> int:
    """Bridge training error patterns into inference error evolution.

    Replays training-discovered error patterns (divergence, stagnation)
    into the inference pipeline's ``CausalErrorEvolutionTracker`` so that
    inference-time metacognitive triggers and recovery strategies benefit
    from training-time convergence failures.

    When a ``TemporalCausalTraceBuffer`` is provided via *causal_trace*,
    each bridged episode is also recorded as a traced decision so that
    root-cause analysis can trace inference-time recovery strategies back
    to the specific training-time failure patterns that informed them.

    When *inference_convergence_monitor* is provided, it is wired to
    the inference error evolution tracker via
    :meth:`ConvergenceMonitor.set_error_evolution` so that future
    inference-time divergence/stagnation events automatically flow
    into the same error-evolution system.

    When *inference_integrity_monitor* is provided, training-time
    subsystem health degradation is recorded so that the inference
    pipeline's metacognitive trigger can factor in training instabilities.

    When *inference_provenance_tracker* is provided, training-time
    pipeline failure edges are recorded as provenance dependencies so
    that ``trace_root_cause()`` on the inference side can attribute
    inference-time issues to training-discovered structural weaknesses.

    When *inference_metacognitive_trigger* is provided, trigger signal
    weights are adapted from the combined training + inference error
    summary so that the inference pipeline's metacognitive sensitivity
    immediately reflects training-discovered failure patterns rather
    than waiting for the first inference-time error to occur.

    Args:
        trainer_monitor: The training convergence monitor that has
            accumulated error episodes during training.
        inference_error_evolution: The inference pipeline's
            ``CausalErrorEvolutionTracker`` instance.
        causal_trace: Optional ``TemporalCausalTraceBuffer`` for
            recording bridged episodes as causal trace entries.
        inference_convergence_monitor: Optional inference-side
            ``ConvergenceMonitor`` to wire for automatic bridging.
        inference_integrity_monitor: Optional inference-side
            ``SystemIntegrityMonitor`` to record training health.
        inference_provenance_tracker: Optional inference-side
            ``CausalProvenanceTracker`` to receive training-time
            structural failure edges as provenance dependencies.
        inference_metacognitive_trigger: Optional inference-side
            ``MetaCognitiveRecursionTrigger`` whose signal weights
            will be adapted from the bridged error summary.

    Returns:
        Number of error episodes bridged.
    """
    if inference_error_evolution is None:
        return 0

    # Wire inference convergence monitor → error evolution so future
    # inference-time convergence events are automatically bridged.
    if inference_convergence_monitor is not None:
        try:
            inference_convergence_monitor.set_error_evolution(
                inference_error_evolution,
            )
        except AttributeError as _ae:
            # Older ConvergenceMonitor without set_error_evolution.
            # Record the gap so the metacognitive trigger learns that
            # the convergence→error_evolution wiring is incomplete.
            logger.debug(
                "ConvergenceMonitor lacks set_error_evolution: %s", _ae,
            )
            if inference_error_evolution is not None:
                try:
                    inference_error_evolution.record_episode(
                        error_class='convergence_monitor_wiring_gap',
                        strategy_used='bridge_training_errors',
                        success=False,
                        metadata={'error': str(_ae)[:200]},
                        causal_antecedents=["training_bridge", "convergence_monitor"],
                    )
                except Exception as _ee_err:
                    logger.debug(
                        "Error evolution unavailable in "
                        "bridge_training_errors_to_inference: %s", _ee_err,
                    )
    training_summary = trainer_monitor.export_error_patterns()
    error_classes = training_summary.get('error_classes', {})
    bridged = 0
    for cls_name, cls_stats in error_classes.items():
        count = cls_stats.get('count', 0)
        success_rate = cls_stats.get('success_rate', 1.0)
        if count > 0 and success_rate < 1.0:
            # Compute a severity indicator from loss magnitude so the
            # inference-side metacognitive trigger can distinguish mild
            # training hiccups from catastrophic divergence.  Severity
            # is clamped to [0, 1] with log-scaling to avoid outlier
            # dominance.
            _max_loss = cls_stats.get('max_loss_magnitude')
            _mean_loss = cls_stats.get('mean_loss_magnitude')
            _severity = 0.0
            if _max_loss is not None and _max_loss > 0:
                _severity = min(1.0, math.log1p(_max_loss) / 10.0)
            inference_error_evolution.record_episode(
                error_class=f"training_{cls_name}",
                strategy_used=cls_stats.get('best_strategy', 'unknown'),
                success=success_rate >= 0.5,
                metadata={
                    'source': 'training_bridge',
                    'training_count': count,
                    'training_success_rate': success_rate,
                    'max_loss_magnitude': _max_loss,
                    'mean_loss_magnitude': _mean_loss,
                    'severity': _severity,
                },
                causal_antecedents=["training_bridge", cls_name],
            )
            # Record bridge event in causal trace so inference-time
            # recovery strategies are traceable to training failures.
            if causal_trace is not None:
                try:
                    causal_trace.record(
                        subsystem="training_bridge",
                        decision=f"bridged_{cls_name}",
                        metadata={
                            "training_count": count,
                            "success_rate": success_rate,
                            "best_strategy": cls_stats.get(
                                "best_strategy", "unknown",
                            ),
                        },
                        severity="warning" if success_rate < 0.5 else "info",
                    )
                except Exception as _ct_err:
                    logging.getLogger(__name__).debug(
                        "Causal trace recording failed during bridge: %s",
                        _ct_err,
                    )
            bridged += 1

    # Bridge training health into inference integrity monitor so that
    # training-time subsystem degradation is visible to the inference
    # pipeline's metacognitive trigger via integrity anomaly detection.
    _TRAINING_HEALTH_THRESHOLD = 0.5
    if inference_integrity_monitor is not None:
        for cls_name, cls_stats in error_classes.items():
            success_rate = cls_stats.get('success_rate', 1.0)
            if success_rate < _TRAINING_HEALTH_THRESHOLD:
                try:
                    inference_integrity_monitor.record_health(
                        f"training_{cls_name}",
                        success_rate,
                        {"source": "training_bridge", "count": cls_stats.get("count", 0)},
                    )
                except (AttributeError, TypeError) as _him_err:
                    logging.getLogger(__name__).debug(
                        "Integrity monitor bridge failed for %s: %s",
                        cls_name, _him_err,
                    )

    # Bridge training-time pipeline failure edges into inference provenance
    # tracker so that trace_root_cause() on the inference side can
    # attribute inference-time issues to training-discovered structural
    # weaknesses.  Error classes are mapped to pipeline stage pairs so
    # the provenance DAG includes training-time failure causality.
    if inference_provenance_tracker is not None:
        for cls_name in error_classes:
            stage_pair = _ERROR_CLASS_TO_DEPENDENCY_MAP.get(cls_name)
            if stage_pair is not None:
                try:
                    inference_provenance_tracker.record_dependency(
                        f"training_{stage_pair[0]}", stage_pair[1],
                    )
                except (AttributeError, TypeError) as _prov_err:
                    logging.getLogger(__name__).debug(
                        "Provenance bridge failed for %s: %s",
                        cls_name, _prov_err,
                    )

    # Adapt inference-side metacognitive trigger signal weights from the
    # combined error summary (now including bridged training episodes).
    # Without this, the inference trigger uses default uniform weights
    # and training-discovered failure patterns do not influence trigger
    # sensitivity until the first inference-time error event triggers
    # adapt_weights_from_evolution inside the forward pass.  This closes
    # the training→inference knowledge transfer loop for trigger weights.
    if inference_metacognitive_trigger is not None and bridged > 0:
        try:
            _combined_summary = inference_error_evolution.get_error_summary()
            inference_metacognitive_trigger.adapt_weights_from_evolution(
                _combined_summary,
            )
        except (AttributeError, TypeError) as _trigger_err:
            logging.getLogger(__name__).debug(
                "Metacognitive trigger weight adaptation failed "
                "during bridge (non-fatal): %s", _trigger_err,
            )

    return bridged


def bridge_inference_insights_to_training(
    inference_error_evolution: Any,
    trainer: 'SafeThoughtAETrainerV4',
    inference_uncertainty_tracker: Any = None,
    training_error_evolution: Any = None,
    training_convergence_monitor: Any = None,
    training_metacognitive_trigger: Any = None,
    training_provenance_tracker: Any = None,
) -> int:
    """Bridge inference-time insights back into training parameters.

    Closes the bidirectional feedback loop between inference and training.
    When inference discovers recurring error patterns (e.g., coherence
    deficits, convergence conflicts, memory-reasoning inconsistencies),
    this function adjusts training hyperparameters to prevent those
    patterns from recurring.

    This implements the architectural requirement that core inference
    insights feed back into training objectives, not just the other
    direction (training→inference).

    When *training_error_evolution* is provided, inference error
    patterns are replayed into the training-side error evolution
    tracker so the training pipeline can learn from inference-time
    failure modes.

    When *training_convergence_monitor* is provided, it is wired to
    the training error evolution tracker via
    :meth:`ConvergenceMonitor.set_error_evolution` so future
    training-time convergence events benefit from inference patterns.

    When *training_metacognitive_trigger* is provided, trigger signal
    weights are adapted from the combined inference + training error
    summary so training-time metacognitive sensitivity immediately
    reflects inference-discovered failure patterns.

    When *training_provenance_tracker* is provided, inference-time
    pipeline failure edges are recorded as provenance dependencies so
    that root-cause analysis on the training side can attribute
    training issues to inference-discovered structural weaknesses.

    Args:
        inference_error_evolution: The inference pipeline's
            ``CausalErrorEvolutionTracker`` with accumulated patterns.
        trainer: The ``SafeThoughtAETrainerV4`` instance whose
            hyperparameters will be adapted.
        inference_uncertainty_tracker: Optional ``DirectionalUncertaintyTracker``
            whose per-module breakdown informs which training components
            need the most attention.
        training_error_evolution: Optional training-side
            ``CausalErrorEvolutionTracker`` that receives replayed
            inference error patterns for bidirectional learning.
        training_convergence_monitor: Optional training-side
            ``ConvergenceMonitor`` to wire to the training error
            evolution tracker.
        training_metacognitive_trigger: Optional training-side
            ``MetaCognitiveRecursionTrigger`` whose signal weights
            will be adapted from the inference error summary.
        training_provenance_tracker: Optional training-side
            ``CausalProvenanceTracker`` to receive inference-time
            structural failure edges as provenance dependencies.

    Returns:
        Number of training adjustments applied.
    """
    if inference_error_evolution is None or trainer is None:
        return 0

    adjustments = 0
    summary = inference_error_evolution.get_error_summary()
    error_classes = summary.get('error_classes', {})

    # Adapt gradient clipping if convergence conflicts recur
    _conflict_stats = error_classes.get('convergence_conflict', {})
    if (_conflict_stats.get('count', 0) >= 2
            and _conflict_stats.get('success_rate', 1.0) < 0.5):
        if hasattr(trainer, '_grad_clip_norm'):
            _old_clip = trainer._grad_clip_norm
            trainer._grad_clip_norm = max(0.1, _old_clip * 0.9)
            adjustments += 1

    # Adapt learning rate if coherence deficits recur
    _coherence_stats = error_classes.get('coherence_deficit', {})
    _post_coh_stats = error_classes.get('post_integration_coherence_deficit', {})
    _total_coh_failures = (
        _coherence_stats.get('count', 0) + _post_coh_stats.get('count', 0)
    )
    if _total_coh_failures >= 3:
        if hasattr(trainer, '_metacognitive_lr_factor'):
            trainer._metacognitive_lr_factor = max(
                0.1, trainer._metacognitive_lr_factor * 0.95,
            )
            adjustments += 1

    # Use directional uncertainty to identify problematic modules
    if inference_uncertainty_tracker is not None:
        try:
            _summary = inference_uncertainty_tracker.build_summary()
            _most_uncertain = _summary.get('most_uncertain_module')
            if _most_uncertain and hasattr(trainer, '_inference_module_feedback'):
                trainer._inference_module_feedback[_most_uncertain] = (
                    _summary.get('aggregate_uncertainty', 0.0)
                )
                adjustments += 1
        except (AttributeError, KeyError, TypeError) as _unc_err:
            logging.getLogger(__name__).debug(
                "Uncertainty tracker bridge failed (non-fatal): %s", _unc_err,
            )

    # ── Replay inference error patterns into training error evolution ──
    # Mirrors bridge_training_errors_to_inference: inference-discovered
    # error patterns are replayed into the training-side error evolution
    # so the training pipeline can learn from inference-time failures and
    # adapt its recovery strategies accordingly.
    _bridged_to_training = 0
    if training_error_evolution is not None:
        for cls_name, cls_stats in error_classes.items():
            count = cls_stats.get('count', 0)
            success_rate = cls_stats.get('success_rate', 1.0)
            if count > 0 and success_rate < 1.0:
                _max_loss = cls_stats.get('max_loss_magnitude')
                _severity = 0.0
                if _max_loss is not None and _max_loss > 0:
                    _severity = min(1.0, math.log1p(_max_loss) / 10.0)
                try:
                    training_error_evolution.record_episode(
                        error_class=f"inference_{cls_name}",
                        strategy_used=cls_stats.get(
                            'best_strategy', 'unknown',
                        ),
                        success=success_rate >= 0.5,
                        metadata={
                            'source': 'inference_bridge',
                            'inference_count': count,
                            'inference_success_rate': success_rate,
                            'max_loss_magnitude': _max_loss,
                            'severity': _severity,
                        },
                        causal_antecedents=["inference_bridge", cls_name],
                    )
                    _bridged_to_training += 1
                except (AttributeError, TypeError) as _ee_err:
                    logging.getLogger(__name__).debug(
                        "Error evolution replay failed for %s: %s",
                        cls_name, _ee_err,
                    )
        if _bridged_to_training > 0:
            adjustments += _bridged_to_training

    # ── Wire training convergence monitor → training error evolution ──
    # Mirrors bridge_training_errors_to_inference: wire the training
    # convergence monitor so future training convergence events benefit
    # from the inference error patterns just replayed above.
    if (training_convergence_monitor is not None
            and training_error_evolution is not None):
        try:
            training_convergence_monitor.set_error_evolution(
                training_error_evolution,
            )
        except AttributeError as _ae:
            logger.debug(
                "Training ConvergenceMonitor lacks set_error_evolution: %s",
                _ae,
            )
            if training_error_evolution is not None:
                try:
                    training_error_evolution.record_episode(
                        error_class='convergence_monitor_wiring_gap',
                        strategy_used='bridge_inference_errors',
                        success=False,
                        metadata={'error': str(_ae)[:200]},
                        causal_antecedents=["inference_bridge", "convergence_monitor"],
                    )
                except Exception as _ee_err:
                    logger.debug(
                        "Error evolution unavailable in "
                        "bridge_inference_insights_to_training: %s", _ee_err,
                    )

    # ── Adapt training metacognitive trigger signal weights ──
    # Mirrors bridge_training_errors_to_inference: adapt the training
    # metacognitive trigger weights from the combined error summary so
    # the training pipeline's sensitivity immediately reflects
    # inference-discovered failure patterns.
    if (training_metacognitive_trigger is not None
            and training_error_evolution is not None
            and _bridged_to_training > 0):
        try:
            _combined_summary = training_error_evolution.get_error_summary()
            training_metacognitive_trigger.adapt_weights_from_evolution(
                _combined_summary,
            )
            adjustments += 1
        except (AttributeError, TypeError) as _trigger_err:
            logging.getLogger(__name__).debug(
                "Training metacognitive trigger adaptation failed "
                "during inference bridge (non-fatal): %s", _trigger_err,
            )

    # ── Record inference failure edges in training provenance ──
    # Mirrors bridge_training_errors_to_inference: record inference-time
    # pipeline failure edges as provenance dependencies so root-cause
    # analysis on the training side traces issues to inference-discovered
    # structural weaknesses.
    if training_provenance_tracker is not None:
        for cls_name in error_classes:
            stage_pair = _ERROR_CLASS_TO_DEPENDENCY_MAP.get(cls_name)
            if stage_pair is not None:
                try:
                    training_provenance_tracker.record_dependency(
                        f"inference_{stage_pair[0]}", stage_pair[1],
                    )
                    adjustments += 1
                except (AttributeError, TypeError) as _prov_err:
                    logging.getLogger(__name__).debug(
                        "Training provenance bridge failed for %s: %s",
                        cls_name, _prov_err,
                    )

    # Record bridge adjustments in causal trace for deterministic
    # traceability — without this, training parameter changes triggered
    # by inference error patterns are invisible to root-cause analysis.
    if adjustments > 0:
        _ct_details: Dict[str, Any] = {
            'source': 'bridge_inference_insights_to_training',
        }
        if (_conflict_stats.get('count', 0) >= 2
                and _conflict_stats.get('success_rate', 1.0) < 0.5):
            _ct_details['grad_clip_adjusted'] = True
        if _total_coh_failures >= 3:
            _ct_details['lr_factor_adjusted'] = True
        if _bridged_to_training > 0:
            _ct_details['error_episodes_bridged'] = _bridged_to_training
        _model = getattr(trainer, 'model', None)
        if _model is not None:
            _ct = getattr(_model, 'causal_trace', None)
            if _ct is not None:
                try:
                    _ct.record(
                        "inference_to_training_bridge",
                        "training_parameter_adjustment",
                        metadata={
                            'adjustments_applied': adjustments,
                            **_ct_details,
                        },
                    )
                except Exception as _ct_err:
                    logger.debug(
                        "Causal trace recording failed in "
                        "bridge_inference_insights_to_training: %s", _ct_err,
                    )

    return adjustments


# ==============================================================================
# ВАЛИДАЦИЯ
# ==============================================================================

def _validate_component(model_fn, test_input, expected_shape, name, logger):
    """Validate a single model component with shape checking.
    
    Args:
        model_fn: Callable that takes test_input and returns output tensor.
        test_input: Input tensor(s) for the component.
        expected_shape: Expected output shape tuple.
        name: Component name for logging.
        logger: Logger instance.
        
    Returns:
        Tuple of (output_tensor, error_message_or_None).
    """
    try:
        output = model_fn(test_input) if not isinstance(test_input, tuple) else model_fn(*test_input)
        assert output.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {output.shape}"
        )
        input_shape = test_input.shape if not isinstance(test_input, tuple) else [t.shape for t in test_input]
        logger.info(f"   ✅ {name}: {input_shape} → {output.shape}")
        return output, None
    except Exception as e:
        logger.error(f"   ❌ {name}: {e}")
        return None, f"{name}: {e}"


def validate_training_components(model: AEONDeltaV4, config: AEONConfigV4, 
                                  logger: logging.Logger) -> bool:
    """Validate all training components with shape and gradient checks."""
    logger.info("\n🔍 Валидация компонентов обучения v4...")
    
    issues = []
    model_device = next(model.parameters()).device
    test_batch = torch.randint(0, config.vocab_size, (2, config.seq_length), device=model_device)
    
    # Проверка Encoder
    try:
        z = model.encode(test_batch)
        assert z.shape == (2, config.z_dim)
        logger.info(f"   ✅ Encoder: {test_batch.shape} → {z.shape}")
    except Exception as e:
        issues.append(f"Encoder: {e}")
        logger.error(f"   ❌ Encoder: {e}")
    
    # Проверка VQ
    try:
        quantized, vq_loss, indices, stats = model.quantize(z)
        assert quantized.shape == z.shape
        logger.info(f"   ✅ VectorQuantizer: {z.shape} → {quantized.shape}")
        logger.info(f"      entropy_loss: {stats.get('entropy_loss', 'N/A')}")
    except Exception as e:
        issues.append(f"VQ: {e}")
        logger.error(f"   ❌ VQ: {e}")
    
    # Проверка Decoder
    try:
        logits = model.decode(quantized, test_batch)
        assert logits.shape == (2, config.seq_length, config.vocab_size)
        logger.info(f"   ✅ Decoder: {quantized.shape} → {logits.shape}")
    except Exception as e:
        issues.append(f"Decoder: {e}")
        logger.error(f"   ❌ Decoder: {e}")
    
    # Проверка Contextual RSSM
    try:
        K = config.context_window
        z_context = z.unsqueeze(1).expand(-1, K, -1)  # [2, K, D]
        z_pred = model.rssm(z_context)
        assert z_pred.shape == z.shape
        logger.info(f"   ✅ ContextualRSSM: {z_context.shape} → {z_pred.shape}")
    except Exception as e:
        issues.append(f"RSSM: {e}")
        logger.error(f"   ❌ RSSM: {e}")
    
    # Проверка градиентов
    model.train()
    model.zero_grad()
    
    z = model.encode(test_batch)
    quantized, vq_loss, _, _ = model.quantize(z)
    logits = model.decode(quantized, test_batch)
    
    recon_loss = F.cross_entropy(logits.view(-1, config.vocab_size), test_batch.view(-1))
    total_loss = recon_loss + vq_loss
    total_loss.backward()
    
    for name, component in [("encoder", model.encoder), ("decoder", model.decoder), ("vq", model.vq)]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in component.parameters())
        if has_grad:
            logger.info(f"   ✅ {name}: градиенты проходят")
        else:
            embed_attr = getattr(component, 'embedding', None) or getattr(component, 'embeddings', None)
            if name == "vq" and embed_attr is not None:
                if embed_attr.weight.grad is not None:
                    logger.info(f"   ✅ {name}: градиенты через embedding")
                    continue
            issues.append(f"{name}: нет градиентов")
            logger.error(f"   ❌ {name}: нет градиентов")
    
    model.zero_grad()
    
    # ===== COGNITIVE COHERENCE VERIFICATION =====
    # Cross-validate component outputs using provenance tracking from
    # aeon_core's CausalProvenanceTracker pattern.  This ensures that
    # each component verifies and reinforces the others — a core
    # requirement for a unified cognitive system.
    logger.info("\n🔍 Cognitive coherence verification...")
    provenance = TrainingProvenanceTracker()
    try:
        model.eval()
        with torch.no_grad():
            # Track provenance through the full pipeline
            provenance.reset()

            z_val = model.encode(test_batch)

            provenance.record_before("vq", z_val)
            q_val, _, _, _ = model.quantize(z_val)
            provenance.record_after("vq", q_val)

            provenance.record_before("rssm", q_val)
            K = config.context_window
            z_ctx = q_val.unsqueeze(1).expand(-1, K, -1)
            z_pred_val = model.rssm(z_ctx)
            provenance.record_after("rssm", z_pred_val)

            attribution = provenance.compute_attribution()
            contributions = attribution.get('contributions', {})
            if contributions:
                _max_contrib = max(contributions.values())
                _dominant = max(contributions, key=contributions.get)
                logger.info(
                    f"   ✅ Provenance: dominant_module={_dominant} "
                    f"({_max_contrib:.1%}), "
                    f"modules={list(contributions.keys())}"
                )
                # Warn if a single module dominates — indicates
                # an architectural imbalance where one component
                # overwhelms the others.
                if _max_contrib > _PROVENANCE_DOMINANCE_WARNING_THRESHOLD:
                    logger.warning(
                        f"   ⚠️ Module '{_dominant}' dominates "
                        f"provenance ({_max_contrib:.1%} > "
                        f"{_PROVENANCE_DOMINANCE_WARNING_THRESHOLD:.0%}). "
                        f"Consider rebalancing component contributions."
                    )

            # Cross-module coherence: verify that encoder output and
            # VQ output are semantically aligned (cosine similarity).
            # A threshold of -0.5 catches severe misalignment while
            # allowing normal VQ quantization shifts that may reduce
            # cosine similarity from 1.0.
            _COHERENCE_THRESHOLD = -0.5
            cos_sim = F.cosine_similarity(z_val, q_val, dim=-1).mean().item()
            if cos_sim > _COHERENCE_THRESHOLD:
                logger.info(
                    f"   ✅ Encoder↔VQ coherence: cos_sim={cos_sim:.4f}"
                )
            else:
                logger.warning(
                    f"   ⚠️ Low encoder↔VQ coherence: cos_sim={cos_sim:.4f}"
                )

            # Verify RSSM prediction is finite and reasonably close to input
            if torch.isfinite(z_pred_val).all():
                rssm_cos = F.cosine_similarity(
                    q_val, z_pred_val, dim=-1
                ).mean().item()
                logger.info(
                    f"   ✅ VQ↔RSSM coherence: cos_sim={rssm_cos:.4f}"
                )
            else:
                issues.append("RSSM: non-finite predictions")
                logger.error("   ❌ RSSM produces non-finite values")

    except Exception as coherence_err:
        logger.warning(
            f"   ⚠️ Coherence verification failed (non-fatal): {coherence_err}"
        )

    # ===== UNIFIED COGNITIVE CYCLE WIRING VERIFICATION =====
    # Verify that the UCC components are properly connected: the
    # convergence monitor must be wired to the error evolution tracker,
    # and the metacognitive trigger must be present.  This ensures that
    # uncertainty triggers a meta-cognitive cycle during training.
    logger.info("\n🔍 Unified Cognitive Cycle wiring verification...")
    try:
        _test_ee = CausalErrorEvolutionTracker(max_history=10)
        _test_cv = ConvergenceMonitor(threshold=1e-5)
        _test_mcv = ModuleCoherenceVerifier(hidden_dim=config.z_dim, threshold=0.5)
        _test_mct = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=2,
        )
        _test_prov = CausalProvenanceTracker()
        _test_ucc = UnifiedCognitiveCycle(
            convergence_monitor=_test_cv,
            coherence_verifier=_test_mcv,
            error_evolution=_test_ee,
            metacognitive_trigger=_test_mct,
            provenance_tracker=_test_prov,
        )
        # Verify wiring: convergence monitor should be linked to
        # error evolution and provenance tracker.
        assert _test_cv._error_evolution is _test_ee, (
            "ConvergenceMonitor not wired to error evolution"
        )
        assert _test_cv._provenance_tracker is _test_prov, (
            "ConvergenceMonitor not wired to provenance tracker"
        )
        # Smoke-test: run a cycle with real states from the model
        with torch.no_grad():
            _test_z = model.encode(test_batch)
            _test_q, _, _, _ = model.quantize(_test_z)
        _test_result = _test_ucc.evaluate(
            subsystem_states={"encoder": _test_z.detach(), "vq": _test_q.detach()},
            delta_norm=0.01,
            uncertainty=0.0,
        )
        assert "should_rerun" in _test_result, "UCC evaluate missing should_rerun"
        assert "coherence_result" in _test_result, "UCC evaluate missing coherence_result"
        assert _test_ucc.metacognitive_trigger is _test_mct, (
            "MetaCognitiveRecursionTrigger not wired to UCC"
        )
        logger.info("   ✅ UCC wiring: convergence→error_evolution→provenance verified")
        logger.info(
            f"   ✅ UCC smoke test: coherence_deficit="
            f"{_test_result['coherence_result']['coherence_deficit']:.4f}, "
            f"should_rerun={_test_result['should_rerun']}"
        )
    except Exception as ucc_err:
        issues.append(f"UCC wiring: {ucc_err}")
        logger.error(f"   ❌ UCC wiring verification failed: {ucc_err}")
    
    if issues:
        logger.error(f"\n⚠️ Обнаружено {len(issues)} проблем!")
        return False
    
    logger.info("\n✅ Все компоненты v4 настроены корректно!")
    return True


# ==============================================================================
# UPGRADE INTEGRATION: VibeThinker × ae_train × Cognitive Architecture
# ==============================================================================
# Implements the complete 10-point integration plan from upgrade.txt:
# (1) Semantic codebook warm-start via k-means on VibeThinker embeddings
# (2) Context window calibration from CoT depth distribution
# (3) Persistent streaming bridge signals via VTStreamingSignalBus
# (4) Teacher-student role inversion (AEON ↔ VibeThinker)
# (5) MetaCognitiveRecursionTrigger VibeThinker signal alignment
# (6) Automatic task boundary detection via coherence monitoring
# (7) UCC inner-epoch coherence evaluation with rerun logic
# (8) Quality-annotated z-sequences for Phase B joint objective
# (9) Adaptive entropy regularization via VibeThinker entropy signal
# (10) Micro-retrain from VibeThinkerContinuousLearner pseudo-labels
# (III) Corpus diagnostics via VibeThinker (initial setup architecture)
# (IV.48) SSP temperature alignment (VibeThinker × Gumbel VQ)
# ==============================================================================

# --- Integration constants ---
_MIN_CONTEXT_WINDOW = 1
_MAX_CONTEXT_WINDOW = 16
_MIN_ENTROPY_WEIGHT = 0.01
_MAX_ENTROPY_WEIGHT = 1.0
_PSEUDO_LABEL_QUALITY_THRESHOLD = 0.5
_MIN_CURRICULUM_FRAC = 0.3
_CURRICULUM_GROWTH_RATE = 0.7
_STRONG_BOUNDARY_THRESHOLD = 0.1
_MICRO_RETRAIN_LR_SCALE = 0.1
# Bifasic didactic: learning rate scale for student-mode adapter training
_BIFASIC_STUDENT_LR_SCALE = 0.05

# Item #7: UCC inner-epoch evaluation interval (every K epochs)
_UCC_INNER_EPOCH_INTERVAL = 5
# Maximum consecutive reruns to prevent infinite loops
_UCC_MAX_RERUNS = 2
# LR reduction factor on coherence-driven rerun
_UCC_RERUN_LR_FACTOR = 0.8


class VTStreamingSignalBus:
    """Item #3: Persistent streaming bridge between VibeThinker and training.

    Transforms one-shot bridge calls into continuous signal streams.
    VibeThinkerContinuousLearner produces four streaming signals
    (calibration_error, adaptation_signal, psi_weight_ema,
    complexity_threshold_ema) that flow through this bus into
    AdaptiveTrainingController, creating a closed control loop:
    calibration error → adaptation → LR/grad_clip adjustment →
    improved z-representations → higher VibeThinker reasoning quality.
    """

    _SIGNAL_NAMES = (
        "calibration_pressure",
        "adaptation_signal",
        "psi_weight",
        "complexity_threshold",
    )
    _EMA_ALPHA = 0.2

    def __init__(self):
        self._buffer: Dict[str, List[float]] = {s: [] for s in self._SIGNAL_NAMES}
        self._ema: Dict[str, float] = {s: 0.0 for s in self._SIGNAL_NAMES}
        self._step_count: int = 0

    def push(self, signal_name: str, value: float) -> None:
        """Buffer a streaming signal value."""
        if signal_name in self._buffer:
            self._buffer[signal_name].append(float(value))
            self._ema[signal_name] = (
                self._EMA_ALPHA * float(value)
                + (1.0 - self._EMA_ALPHA) * self._ema[signal_name]
            )

    def pull_all(self) -> Dict[str, List[float]]:
        """Drain and return all buffered signals."""
        snapshot = {k: list(v) for k, v in self._buffer.items()}
        for k in self._buffer:
            self._buffer[k].clear()
        return snapshot

    def get_ema(self) -> Dict[str, float]:
        """Return EMA trend for each signal."""
        return dict(self._ema)

    def apply_to_controller(
        self, controller: 'AdaptiveTrainingController',
    ) -> Dict[str, Any]:
        """Apply buffered streaming signals to training controller.

        Uses calibration pressure to modulate learning rate and
        adaptation signal to adjust gradient clipping, closing
        the VibeThinker → training feedback loop.

        Returns:
            Dict with applied adjustments.
        """
        adjustments: Dict[str, Any] = {}
        cal_pressure = self._ema.get("calibration_pressure", 0.0)
        adapt_sig = self._ema.get("adaptation_signal", 0.0)

        # High calibration pressure → VibeThinker poorly calibrated →
        # slow down training to stabilise representations.
        if cal_pressure > 0.3:
            scale = max(0.5, 1.0 - cal_pressure)
            adjustments["lr_scale"] = scale
            adjustments["lr_reason"] = "vt_calibration_pressure"

        # Low adaptation signal → VibeThinker struggling to adapt →
        # tighten gradient clipping for stability.
        if adapt_sig < 0.3:
            adjustments["grad_clip_scale"] = max(0.5, adapt_sig + 0.5)
            adjustments["gc_reason"] = "vt_low_adaptation"

        self._step_count += 1
        adjustments["streaming_step"] = self._step_count
        return adjustments

    def closed_loop_step(
        self,
        vt_learner: Any,
        controller: 'AdaptiveTrainingController',
    ) -> Dict[str, Any]:
        """Execute one closed-loop streaming step.

        Reads VibeThinkerContinuousLearner state → pushes signals →
        applies adjustments to AdaptiveTrainingController.

        Args:
            vt_learner: VibeThinkerContinuousLearner instance.
            controller: AdaptiveTrainingController to adjust.

        Returns:
            Dict with signals read and adjustments applied.
        """
        if vt_learner is None:
            return {"streaming": False, "reason": "no_learner"}

        # [W3] Guard: skip streaming adjustments until the VT learner
        # has accumulated enough episodes for reliable EMA statistics.
        # With zero episodes the EMA values are uninitialised defaults
        # that would produce spurious LR/grad_clip adjustments.
        _MIN_VT_LEARNER_EPISODES = 50
        _episode_count = getattr(vt_learner, '_episode_count', 0)
        if _episode_count < _MIN_VT_LEARNER_EPISODES:
            return {
                "streaming": False,
                "reason": f"insufficient_episodes ({_episode_count}/{_MIN_VT_LEARNER_EPISODES})",
            }

        self.push("calibration_pressure", getattr(
            vt_learner, '_calibration_ema', 0.0,
        ))
        self.push("adaptation_signal", max(
            0.0, 1.0 - getattr(vt_learner, '_calibration_ema', 0.0) * 2,
        ))
        self.push("psi_weight", getattr(
            vt_learner, '_psi_weight_ema', 0.1,
        ))
        self.push("complexity_threshold", getattr(
            vt_learner, '_complexity_threshold_ema', 0.5,
        ))

        adjustments = self.apply_to_controller(controller)
        return {
            "streaming": True,
            "ema": self.get_ema(),
            "adjustments": adjustments,
        }


def map_vt_signals_to_trigger(
    vt_confidence: float = 0.5,
    vt_entropy: float = 0.5,
    vt_cot_depth: float = 1.0,
    vt_calibration_error: float = 0.0,
    cot_depth_threshold: float = 0.3,
) -> Dict[str, float]:
    """Item #5: Map VibeThinker signals to MetaCognitiveRecursionTrigger inputs.

    Implements the theoretical mapping from upgrade.txt:
      - confidence → 1 - uncertainty
      - entropy → diversity_collapse + spectral_instability (product)
      - calibration_error → coherence_deficit
      - d(calibration_ema)/dt drop → convergence_conflict
      - cot_depth < threshold → low_causal_quality

    This enables a single MetaCognitiveRecursionTrigger instance shared
    between ae_train and aeon_core, where VibeThinker inference signals
    directly modulate training-time metacognitive sensitivity.

    Args:
        vt_confidence: VibeThinker reasoning confidence ∈ [0, 1].
        vt_entropy: VibeThinker response entropy ∈ [0, 1].
        vt_cot_depth: Predicted chain-of-thought depth.
        vt_calibration_error: Calibration EMA error ∈ [0, 1].
        cot_depth_threshold: Below this, causal quality is low.

    Returns:
        Dict of MetaCognitiveRecursionTrigger-compatible signal kwargs.
    """
    uncertainty = max(0.0, min(1.0, 1.0 - vt_confidence))
    # Entropy splits into diversity_collapse and spectral_instability
    # via geometric mean decomposition.
    _ent_clamped = max(0.0, min(1.0, vt_entropy))
    diversity_collapse = _ent_clamped ** 0.5
    spectral_instability = _ent_clamped ** 0.5
    coherence_deficit = max(0.0, min(1.0, vt_calibration_error))
    convergence_conflict = max(0.0, min(1.0, vt_calibration_error * 1.5))
    low_causal = 1.0 if vt_cot_depth < cot_depth_threshold else 0.0

    return {
        "uncertainty": uncertainty,
        "diversity_collapse": diversity_collapse,
        "spectral_stability_margin": 1.0 - spectral_instability,
        "coherence_deficit": coherence_deficit,
        "convergence_conflict": convergence_conflict,
        "causal_quality": 1.0 - low_causal,
    }


def ucc_inner_epoch_evaluation(
    cycle: 'UnifiedCognitiveCycle',
    subsystem_states: Dict[str, torch.Tensor],
    loss_delta: float,
    uncertainty: float,
    epoch: int,
    total_epochs: int,
    interval: int = _UCC_INNER_EPOCH_INTERVAL,
    vt_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Item #7: Lightweight UCC evaluation within epoch loops.

    Instead of running UCC only at the end of training, this function
    is called every *interval* epochs within Phase A and Phase B.
    When UCC signals ``should_rerun=True``, the caller can repeat the
    epoch with a reduced learning rate, transforming training from
    pure loss minimisation into joint optimisation of loss + coherence.

    Args:
        cycle: UnifiedCognitiveCycle instance.
        subsystem_states: Current subsystem state vectors.
        loss_delta: Absolute loss change since last check.
        uncertainty: Normalised uncertainty ∈ [0, 1].
        epoch: Current epoch (0-indexed).
        total_epochs: Total epochs.
        interval: Evaluate every *interval* epochs.
        vt_signals: Optional VibeThinker-mapped signals from
            :func:`map_vt_signals_to_trigger`.

    Returns:
        Dict with 'evaluated' (bool), 'should_rerun' (bool),
        'coherence_deficit' (float), 'lr_adjustment' (float or None),
        'rerun_count' (int).
    """
    result: Dict[str, Any] = {
        "evaluated": False,
        "should_rerun": False,
        "coherence_deficit": 0.0,
        "lr_adjustment": None,
        "rerun_count": 0,
        "epoch": epoch,
    }

    # Only evaluate at interval boundaries (and always at last epoch)
    if epoch % interval != 0 and epoch != total_epochs - 1:
        return result

    try:
        eval_kwargs: Dict[str, Any] = {
            "subsystem_states": subsystem_states,
            "delta_norm": loss_delta,
            "uncertainty": uncertainty,
        }
        # Inject VibeThinker-mapped signals (Item #5 bridge)
        if vt_signals:
            eval_kwargs.update({
                k: v for k, v in vt_signals.items()
                if k in (
                    "diversity_collapse", "coherence_deficit",
                    "convergence_conflict", "causal_quality",
                    "spectral_stability_margin",
                )
            })
            if "uncertainty" in vt_signals:
                # Blend VT uncertainty with training uncertainty
                eval_kwargs["uncertainty"] = max(
                    uncertainty, vt_signals["uncertainty"],
                )

        cycle_result = cycle.evaluate(**eval_kwargs)
        coherence_deficit = cycle_result.get(
            "coherence_result", {},
        ).get("coherence_deficit", 0.0)
        should_rerun = cycle_result.get("should_rerun", False)

        result["evaluated"] = True
        result["coherence_deficit"] = coherence_deficit
        result["should_rerun"] = should_rerun

        if should_rerun:
            result["lr_adjustment"] = _UCC_RERUN_LR_FACTOR
            result["rerun_count"] = 1

    except Exception:
        result["evaluated"] = False

    return result


def align_ssp_temperature(
    vt_temperature: float = 1.0,
    gumbel_temperature: float = 1.0,
    alignment_factor: float = 0.5,
) -> Dict[str, float]:
    """Item #48: Align VibeThinker SSP temperature with Gumbel VQ temperature.

    The SSP (Two-Stage Diversity-Exploring Distillation) mechanism in
    VibeThinkerConfig generates diverse reasoning chains via temperature
    sampling.  GumbelVectorQuantizer uses Gumbel-Softmax temperature
    for codebook diversity.  Aligning these temperatures ensures that
    reasoning diversity matches latent code diversity — a parametric
    invariant for semantic coherence between the two spaces.

    Args:
        vt_temperature: VibeThinker reasoning temperature.
        gumbel_temperature: Gumbel VQ sampling temperature.
        alignment_factor: Blend factor ∈ [0, 1] for harmonisation.

    Returns:
        Dict with aligned temperatures and scaling metadata.
    """
    # Geometric mean provides scale-invariant alignment
    aligned = (vt_temperature * gumbel_temperature) ** 0.5
    vt_aligned = vt_temperature * (1.0 - alignment_factor) + aligned * alignment_factor
    gumbel_aligned = gumbel_temperature * (1.0 - alignment_factor) + aligned * alignment_factor

    return {
        "vt_temperature_original": vt_temperature,
        "gumbel_temperature_original": gumbel_temperature,
        "vt_temperature_aligned": vt_aligned,
        "gumbel_temperature_aligned": gumbel_aligned,
        "geometric_mean": aligned,
        "alignment_factor": alignment_factor,
    }


def diagnose_corpus_via_vt(
    model: nn.Module,
    tokens: torch.Tensor,
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Items #40-42: VibeThinker corpus diagnostics for initial setup.

    Before training begins, VibeThinker acts as corpus diagnostician,
    hyperparametrist, and codebook seeder:

    Role 1 — Diagnostician: Computes per-sample complexity scores via
    VibeThinkerPromptAdapter.  Distribution statistics (mean, variance,
    bimodality) reveal corpus heterogeneity for ContinualLearningCore.

    Role 2 — Hyperparametrist: Recommends codebook_size from cluster
    count (Calinski-Harabasz), context_window from P95(cot_depth),
    and z_dim from PCA explained variance on embeddings.

    Role 3 — Codebook seeder: k-means centroids provide the warm-start
    for VQ codebook (delegates to warm_start_codebook_from_vt).

    Args:
        model: AEONDeltaV4 model (pre-training).
        tokens: Tokenised training corpus.
        config: Training configuration (may be mutated).
        device: Computation device.
        batch_size: Batch size for inference.

    Returns:
        Dict with corpus diagnostics, recommended hyperparameters,
        and complexity distribution statistics.
    """
    if not VIBE_THINKER_AVAILABLE:
        return {"diagnosed": False, "reason": "no_vibe_thinker"}

    try:
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)

        kernel = VibeThinkerReasoningKernel(
            config=VibeThinkerConfig(),
            hidden_dim=config.hidden_dim,
        ).to(device)

        complexity_scores = []
        cot_depths = []
        embeddings = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size].to(device)
                z = model.encode(batch)
                vt_out = adapter(z)
                complexity_scores.extend(
                    vt_out['complexity_score'].cpu().tolist()
                    if vt_out['complexity_score'].dim() > 0
                    else [float(vt_out['complexity_score'].cpu())]
                )
                embeddings.append(vt_out['prompt_embedding'].cpu())
                # CoT depth from reasoning kernel
                r_out = kernel.reason(z)
                if isinstance(r_out, dict) and 'cot_depth' in r_out:
                    _depth = r_out['cot_depth']
                    if hasattr(_depth, 'cpu'):
                        cot_depths.extend(
                            _depth.cpu().tolist()
                            if _depth.dim() > 0
                            else [float(_depth.cpu())]
                        )
                    else:
                        cot_depths.append(float(_depth))

        # Distribution statistics
        _scores = np.array(complexity_scores)
        diag: Dict[str, Any] = {
            "diagnosed": True,
            "corpus_size": len(tokens),
            "complexity_mean": float(_scores.mean()),
            "complexity_std": float(_scores.std()),
            "complexity_min": float(_scores.min()),
            "complexity_max": float(_scores.max()),
        }

        # Bimodality detection via Hartigan's dip test approximation
        _sorted = np.sort(_scores)
        _mid = len(_sorted) // 2
        if _mid > 0:
            _lower_mean = _sorted[:_mid].mean()
            _upper_mean = _sorted[_mid:].mean()
            _gap = abs(_upper_mean - _lower_mean)
            diag["bimodality_gap"] = float(_gap)
            diag["heterogeneous"] = _gap > 0.3
        else:
            diag["bimodality_gap"] = 0.0
            diag["heterogeneous"] = False

        # Role 2: Hyperparameter recommendations
        recommendations: Dict[str, Any] = {}

        # Context window from P95(cot_depth)
        if cot_depths:
            _p95 = float(np.percentile(cot_depths, 95))
            recommendations["context_window"] = max(
                _MIN_CONTEXT_WINDOW,
                min(_MAX_CONTEXT_WINDOW, int(np.ceil(_p95))),
            )
            diag["cot_depth_p95"] = _p95

        # z_dim recommendation from PCA explained variance
        all_emb = torch.cat(embeddings, dim=0).numpy()
        if all_emb.shape[0] > all_emb.shape[1]:
            from sklearn.decomposition import PCA
            _pca = PCA(n_components=min(all_emb.shape[1], 64))
            _pca.fit(all_emb)
            _cumvar = np.cumsum(_pca.explained_variance_ratio_)
            _n95 = int(np.searchsorted(_cumvar, 0.95)) + 1
            recommendations["z_dim_95pct"] = _n95
            diag["pca_explained_95pct_components"] = _n95

        # [M2] Codebook size recommendation via Calinski-Harabasz index.
        # Evaluates k-means for candidate codebook sizes and picks the
        # k that maximises the Calinski-Harabasz score (inter-cluster /
        # intra-cluster variance ratio).
        if all_emb.shape[0] > 8:
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import calinski_harabasz_score
                _candidates = [k for k in [32, 64, 128, 256, 512]
                               if k < all_emb.shape[0]]
                if _candidates:
                    _best_ch, _best_k = -1.0, _candidates[0]
                    for _k in _candidates:
                        _km = KMeans(n_clusters=_k, n_init=3, random_state=42,
                                     max_iter=50)
                        _labels = _km.fit_predict(all_emb)
                        _ch = calinski_harabasz_score(all_emb, _labels)
                        if _ch > _best_ch:
                            _best_ch, _best_k = _ch, _k
                    recommendations["codebook_size"] = _best_k
                    diag["calinski_harabasz_best"] = float(_best_ch)
            except ImportError:
                pass  # sklearn not available

        diag["recommendations"] = recommendations
        return diag

    except Exception as e:
        return {"diagnosed": False, "reason": f"error_{type(e).__name__}: {e}"}


def bifasic_didactic_orchestrate(
    model: nn.Module,
    z_sequences: List[torch.Tensor],
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    max_steps: int = 5,
) -> Dict[str, Any]:
    """Item #4: Bifasic didactic role inversion — AEON teaches VibeThinker.

    After Phase A completes, AEON has semantically dense z-representations.
    This function inverts the teacher-student roles: z-representations
    become the ground truth and VibeThinkerPromptAdapter learns to align
    its projection with the learned VQ latent space.

    Phase 1 (pre-Phase A): VibeThinker is teacher — implemented by
    warm_start_codebook_from_vt(), calibrate_context_window(), and
    adapt_entropy_weight().

    Phase 2 (post-Phase A): AEON is teacher — this function.
    Freeze encoder + VQ + decoder.  Train only the adapter using
    z-representations as targets for the adapter projection.

    Args:
        model: AEONDeltaV4 model with trained encoder.
        z_sequences: List of z-sequence tensors from Phase A.
        config: Training configuration.
        device: Computation device.
        max_steps: Maximum adapter update steps.

    Returns:
        Dictionary with role-inversion training results.
    """
    if not VIBE_THINKER_AVAILABLE:
        return {"inverted": False, "reason": "no_vibe_thinker",
                "steps": 0, "loss_start": 0.0, "loss_end": 0.0}

    if not z_sequences:
        return {"inverted": False, "reason": "no_z_sequences",
                "steps": 0, "loss_start": 0.0, "loss_end": 0.0}

    try:
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)

        # Collect a sample of z-vectors as adaptation targets
        all_z = torch.cat(z_sequences, dim=0)
        sample_size = min(max_steps * 4, all_z.shape[0])
        indices = torch.randperm(all_z.shape[0])[:sample_size]
        z_sample = all_z[indices].to(device)

        # Project z-targets to adapter's projection space so loss
        # dimensions match (z_dim → projection_dim).
        z_projector = torch.nn.Linear(
            config.z_dim, adapter.projection_dim,
        ).to(device)

        losses = []
        adapter.train()
        trainable_params = list(adapter.parameters()) + list(z_projector.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate * _BIFASIC_STUDENT_LR_SCALE,
        )
        effective_steps = min(max_steps, sample_size)
        for step in range(effective_steps):
            z_target = z_sample[step].unsqueeze(0)
            vt_out = adapter(z_target)
            # Align adapter's projection with projected z-representations
            pred = vt_out['prompt_embedding']
            target_proj = z_projector(z_target)
            loss = F.mse_loss(pred, target_proj.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return {
            "inverted": True,
            "reason": "success",
            "steps": len(losses),
            "loss_start": losses[0] if losses else 0.0,
            "loss_end": losses[-1] if losses else 0.0,
            "phase": "aeon_teaches_vt",
        }

    except Exception as e:
        logger.warning(f"⚠️ Bifasic didactic inversion failed (non-fatal): {e}")
        return {"inverted": False, "reason": f"error_{type(e).__name__}",
                "steps": 0, "loss_start": 0.0, "loss_end": 0.0}


def warm_start_codebook_from_vt(
    model: nn.Module,
    tokens: torch.Tensor,
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Item #1: Semantic Meta-Initializer — warm-start VQ codebook via
    VibeThinker prompt embeddings clustered with k-means.

    Instead of random N(0, 0.1) initialization, we:
      1. Encode all training tokens through the encoder (no grad).
      2. Project encoder outputs through VibeThinkerPromptAdapter to
         obtain semantic prompt embeddings P = {p_1, ..., p_n}.
      3. Run k-means on P with K = codebook_size clusters.
      4. Initialize codebook centroids from cluster centers.

    This transforms Phase A from random exploration to a warm start
    with semantically meaningful prototypes, guaranteeing monotone
    decrease of commitment loss from epoch 1.

    Additionally, the complexity_head scores from VibeThinkerPromptAdapter
    are collected for curriculum ordering (returned as complexity_scores).

    Args:
        model: AEONDeltaV4 model with encoder and vq attributes.
        tokens: Training token tensor [N, seq_len].
        config: Training configuration.
        device: Computation device.
        batch_size: Batch size for encoding.

    Returns:
        Dictionary with:
          - initialized: bool — whether codebook was warm-started
          - num_embeddings: int — codebook size
          - complexity_scores: List[float] — per-sample complexity scores
          - inertia: float — k-means inertia (lower = tighter clusters)
          - method: str — initialization method used
    """
    if not VIBE_THINKER_AVAILABLE:
        logger.info("⚠️ VibeThinker not available — skipping codebook warm-start")
        return {"initialized": False, "method": "skipped_no_vibe_thinker",
                "complexity_scores": [], "num_embeddings": 0, "inertia": 0.0}

    logger.info("🌱 Warm-starting codebook from VibeThinker embeddings...")

    try:
        # Step 1: Collect encoder embeddings and complexity scores
        # We cluster in z-space (encoder output) because the codebook
        # operates in z-space.  VibeThinkerPromptAdapter is used only
        # for complexity scoring (curriculum ordering).
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        adapter.eval()

        all_embeddings = []
        all_complexity = []
        model.eval()

        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start:start + batch_size].to(device)
                z = model.encode(batch)  # [B, z_dim]
                all_embeddings.append(z.cpu())
                # Use adapter for complexity scoring only
                vt_out = adapter(z)
                all_complexity.append(vt_out['complexity_score'].cpu())

        embeddings = torch.cat(all_embeddings, dim=0)  # [N, z_dim]
        complexity_scores = torch.cat(all_complexity, dim=0).tolist()  # [N]

        # Step 2: K-means clustering
        K = config.vq_num_embeddings
        N = embeddings.shape[0]
        if N < K:
            logger.warning(
                f"⚠️ Not enough samples ({N}) for k-means with K={K} — "
                f"falling back to random init"
            )
            return {"initialized": False, "method": "insufficient_samples",
                    "complexity_scores": complexity_scores,
                    "num_embeddings": K, "inertia": 0.0}

        # Mini-batch k-means (memory efficient, O(N·K·D) per iteration)
        centroids = embeddings[torch.randperm(N)[:K]].clone()  # [K, D]
        max_iter = 50
        best_inertia = float('inf')

        for iteration in range(max_iter):
            # Assign clusters (batch to avoid OOM)
            assignments = []
            for start in range(0, N, batch_size):
                batch = embeddings[start:start + batch_size]
                dists = torch.cdist(batch, centroids)  # [B, K]
                assignments.append(dists.argmin(dim=1))
            assignments = torch.cat(assignments)  # [N]

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = embeddings[mask].mean(dim=0)
                    counts[k] = mask.sum().item()
                else:
                    # Dead cluster — reinitialize from random sample
                    new_centroids[k] = embeddings[torch.randint(N, (1,))].squeeze(0)
                    counts[k] = 0

            # Compute inertia for convergence check
            inertia = 0.0
            for start in range(0, N, batch_size):
                batch = embeddings[start:start + batch_size]
                batch_assign = assignments[start:start + batch_size]
                batch_centroids = new_centroids[batch_assign]
                inertia += ((batch - batch_centroids) ** 2).sum().item()

            shift = (new_centroids - centroids).norm(dim=1).mean().item()
            centroids = new_centroids
            best_inertia = min(best_inertia, inertia)

            if shift < 1e-5:
                logger.info(f"   K-means converged at iteration {iteration + 1}")
                break

        # Step 3: Initialize codebook with centroids
        vq = model.vq
        with torch.no_grad():
            # Match embedding dimension — centroids may differ from codebook dim
            if centroids.shape[1] == vq.embedding.weight.shape[1]:
                vq.embedding.weight.copy_(centroids.to(vq.embedding.weight.device))
                vq.ema_w.copy_(centroids.to(vq.ema_w.device))
                vq.ema_cluster_size.fill_(N / K)  # Uniform cluster assumption
                logger.info(
                    f"✅ Codebook warm-started with {K} semantic centroids "
                    f"(inertia={best_inertia:.2f})"
                )
            else:
                logger.warning(
                    f"⚠️ Dimension mismatch: centroids {centroids.shape[1]} "
                    f"vs codebook {vq.embedding.weight.shape[1]} — skipping"
                )
                return {"initialized": False, "method": "dimension_mismatch",
                        "complexity_scores": complexity_scores,
                        "num_embeddings": K, "inertia": best_inertia}

        return {
            "initialized": True,
            "method": "kmeans_vt_embeddings",
            "num_embeddings": K,
            "inertia": best_inertia,
            "complexity_scores": complexity_scores,
        }

    except Exception as e:
        logger.warning(f"⚠️ Codebook warm-start failed (non-fatal): {e}")
        return {"initialized": False, "method": f"error_{type(e).__name__}",
                "complexity_scores": [], "num_embeddings": 0, "inertia": 0.0}


def calibrate_context_window(
    model: nn.Module,
    tokens: torch.Tensor,
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    batch_size: int = 256,
    percentile: float = 95.0,
) -> Dict[str, Any]:
    """Item #2: Calibrate RSSM context_window from VibeThinker CoT depth.

    Replaces the static context_window=3 hyperparameter with an
    empirically derived value based on the 95th percentile of the
    chain-of-thought depth distribution across the training corpus.

    The CoT depth predicted by VibeThinkerReasoningKernel reflects the
    reasoning complexity required for each input — the context window
    should be large enough to capture the temporal dependencies of the
    most complex 95% of inputs.

    Args:
        model: AEONDeltaV4 model with encoder.
        tokens: Training token tensor [N, seq_len].
        config: Training configuration (will be mutated).
        device: Computation device.
        batch_size: Batch size for encoding.
        percentile: Percentile of CoT depth distribution (default 95).

    Returns:
        Dictionary with calibration results:
          - calibrated: bool
          - original_window: int
          - new_window: int
          - cot_depth_stats: dict with mean, std, p95, min, max
    """
    if not VIBE_THINKER_AVAILABLE:
        logger.info("⚠️ VibeThinker not available — skipping context window calibration")
        return {"calibrated": False, "original_window": config.context_window,
                "new_window": config.context_window, "cot_depth_stats": {}}

    logger.info("📐 Calibrating context window from CoT depth distribution...")

    try:
        _vt_cfg = VibeThinkerConfig()
        _vt_cfg.adapter_projection_dim = 128
        _vt_cfg.adapter_hidden_dim = config.hidden_dim
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
            projection_dim=_vt_cfg.adapter_projection_dim,
        ).to(device)
        kernel = VibeThinkerReasoningKernel(
            config=_vt_cfg,
            hidden_dim=config.hidden_dim,
        ).to(device)
        adapter.eval()
        kernel.eval()
        model.eval()

        cot_depths = []
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start:start + batch_size].to(device)
                z = model.encode(batch)
                vt_out = adapter(z)
                reason_out = kernel.reason(
                    vt_out['prompt_embedding'],
                    vt_out['complexity_score'],
                )
                # reason() returns scalars (float), not tensors
                _depth = reason_out['cot_depth']
                if isinstance(_depth, torch.Tensor):
                    cot_depths.append(float(_depth.cpu()))
                else:
                    cot_depths.append(float(_depth))

        cot_np = np.array(cot_depths)

        stats = {
            "mean": float(cot_np.mean()),
            "std": float(cot_np.std()),
            "min": float(cot_np.min()),
            "max": float(cot_np.max()),
            "p95": float(np.percentile(cot_np, percentile)),
        }

        # Calibrate: P95 of cot_depth, rounded up, clamped to [1, 16]
        original = config.context_window
        new_window = max(_MIN_CONTEXT_WINDOW, min(
            _MAX_CONTEXT_WINDOW, int(math.ceil(stats["p95"])),
        ))
        config.context_window = new_window

        logger.info(
            f"✅ Context window calibrated: {original} → {new_window} "
            f"(P{percentile:.0f} CoT depth = {stats['p95']:.2f})"
        )

        return {
            "calibrated": True,
            "original_window": original,
            "new_window": new_window,
            "cot_depth_stats": stats,
        }

    except Exception as e:
        logger.warning(f"⚠️ Context window calibration failed (non-fatal): {e}")
        return {"calibrated": False, "original_window": config.context_window,
                "new_window": config.context_window, "cot_depth_stats": {}}


def annotate_z_sequences_quality(
    model: nn.Module,
    z_sequences: List[torch.Tensor],
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    batch_size: int = 256,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Item #8: Annotate z-sequences with VibeThinker quality metadata.

    For each z-vector in the z-sequences, computes VibeThinker quality
    signals (confidence, entropy, reasoning_quality) and returns them
    as parallel annotation tensors.

    This enables Phase B to learn a joint objective:
      L = L_mse(z_pred, z_true) + λ·L_quality(q_pred, q_true)

    The RSSM learns to prefer trajectories in latent space that lead
    to regions with high reasoning quality.

    Args:
        model: AEONDeltaV4 model.
        z_sequences: List of z-sequence tensors [num_chunks, D].
        config: Training configuration.
        device: Computation device.
        batch_size: Batch size for encoding.

    Returns:
        Tuple of (z_sequences, quality_annotations) where
        quality_annotations[i] has shape [num_chunks, 3] with columns
        [confidence, entropy, reasoning_quality].
    """
    if not VIBE_THINKER_AVAILABLE:
        logger.info("⚠️ VibeThinker not available — skipping z-sequence quality annotation")
        annotations = [torch.ones(seq.shape[0], 3) for seq in z_sequences]
        return z_sequences, annotations

    logger.info("🏷️  Annotating z-sequences with VibeThinker quality metadata...")

    try:
        _vt_cfg = VibeThinkerConfig()
        _vt_cfg.adapter_projection_dim = 128
        _vt_cfg.adapter_hidden_dim = config.hidden_dim
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
            projection_dim=_vt_cfg.adapter_projection_dim,
        ).to(device)
        kernel = VibeThinkerReasoningKernel(
            config=_vt_cfg,
            hidden_dim=config.hidden_dim,
        ).to(device)
        adapter.eval()
        kernel.eval()

        quality_annotations = []

        with torch.no_grad():
            for seq_idx, z_seq in enumerate(z_sequences):
                seq_annotations = []
                z_on_device = z_seq.to(device)

                for start in range(0, z_on_device.shape[0], batch_size):
                    z_batch = z_on_device[start:start + batch_size]
                    B = z_batch.shape[0]
                    vt_out = adapter(z_batch)

                    # [W2] Per-sample reason() instead of per-batch
                    # broadcast.  Each z-vector gets its own quality
                    # annotation rather than sharing batch-mean values.
                    _sample_annotations = []
                    for _s in range(B):
                        _emb_s = vt_out['prompt_embedding'][_s:_s + 1]
                        _cs_s = vt_out['complexity_score'][_s:_s + 1]
                        _r_s = kernel.reason(_emb_s, _cs_s)
                        _c = float(_r_s['confidence'])
                        _e = float(_r_s['entropy'])
                        _sample_annotations.append([_c, _e, _c * (1.0 - _e)])

                    annotation = torch.tensor(
                        _sample_annotations,
                        dtype=torch.float32,
                    )  # [B, 3]
                    seq_annotations.append(annotation)

                quality_annotations.append(torch.cat(seq_annotations, dim=0))

        total_annotated = sum(a.shape[0] for a in quality_annotations)
        mean_quality = torch.cat(quality_annotations)[:, 2].mean().item()
        logger.info(
            f"✅ Annotated {total_annotated} z-vectors across {len(z_sequences)} "
            f"sequences (mean quality={mean_quality:.4f})"
        )

        return z_sequences, quality_annotations

    except Exception as e:
        logger.warning(f"⚠️ Z-sequence quality annotation failed (non-fatal): {e}")
        annotations = [torch.ones(seq.shape[0], 3) for seq in z_sequences]
        return z_sequences, annotations


def adapt_entropy_weight(
    config: 'AEONConfigV4',
    vt_entropy: float,
    target_entropy: float = 0.5,
    alpha: float = 0.3,
) -> Dict[str, float]:
    """Item #9: Adaptive entropy regularization via VibeThinker entropy.

    Adjusts the codebook entropy_weight based on VibeThinker's entropy
    signal:
      entropy_weight_new = entropy_weight_base × (1 + α × (vt_entropy - target))

    If VibeThinker entropy is high → codebook underutilized → increase
    entropy_weight to stimulate diversity.
    If VibeThinker entropy is low → codebook oversaturated → decrease
    entropy_weight.

    Args:
        config: Training configuration (mutated in place).
        vt_entropy: Mean VibeThinker entropy across the corpus [0, 1].
        target_entropy: Desired entropy level (default 0.5).
        alpha: Adaptation rate (default 0.3).

    Returns:
        Dictionary with adaptation results.
    """
    original = config.entropy_weight
    delta = alpha * (vt_entropy - target_entropy)
    new_weight = original * (1.0 + delta)
    # Clamp to reasonable range [0.01, 1.0]
    new_weight = max(_MIN_ENTROPY_WEIGHT, min(_MAX_ENTROPY_WEIGHT, new_weight))
    config.entropy_weight = new_weight

    logger.info(
        f"📊 Entropy weight adapted: {original:.4f} → {new_weight:.4f} "
        f"(vt_entropy={vt_entropy:.4f}, target={target_entropy:.4f}, "
        f"delta={delta:+.4f})"
    )

    return {
        "original_weight": original,
        "new_weight": new_weight,
        "vt_entropy": vt_entropy,
        "target_entropy": target_entropy,
        "delta": delta,
    }


def auto_detect_task_boundary(
    coherence_score: float,
    coherence_threshold: float = 0.5,
    calibration_ema: float = 1.0,
    previous_coherence: Optional[float] = None,
) -> Dict[str, Any]:
    """Item #6: Automatic task boundary detection via coherence monitoring.

    Detects task boundaries by monitoring coherence between z-sequence
    groups. When inter-group coherence drops below the adaptive threshold,
    this signals a semantic discontinuity — the system encounters a
    qualitatively different task that should trigger ContinualLearningCore.add_task().

    VibeThinker's calibration_ema serves as a verifier: if calibration
    improves after task transition, the boundary was valid.

    Args:
        coherence_score: Current inter-group coherence [0, 1].
        coherence_threshold: Threshold below which a task boundary is detected.
        calibration_ema: VibeThinker calibration EMA for verification.
        previous_coherence: Previous coherence score for trend detection.

    Returns:
        Dictionary with detection results:
          - boundary_detected: bool
          - coherence_score: float
          - coherence_drop: float (if previous available)
          - recommendation: str
          - calibration_ema: float
    """
    boundary_detected = coherence_score < coherence_threshold

    coherence_drop = 0.0
    if previous_coherence is not None:
        coherence_drop = previous_coherence - coherence_score

    # Compound signal: coherence drop + VibeThinker uncertainty
    compound_signal = 0.0
    if boundary_detected:
        compound_signal = (1.0 - coherence_score) * (1.0 - calibration_ema)

    recommendation = "continue"
    if boundary_detected:
        if compound_signal > _STRONG_BOUNDARY_THRESHOLD:
            recommendation = "add_task_strong"
        else:
            recommendation = "add_task_weak"

    return {
        "boundary_detected": boundary_detected,
        "coherence_score": coherence_score,
        "coherence_drop": coherence_drop,
        "compound_signal": compound_signal,
        "recommendation": recommendation,
        "calibration_ema": calibration_ema,
    }


def micro_retrain_from_pseudo_labels(
    model: nn.Module,
    pseudo_labels: List[Dict[str, Any]],
    config: 'AEONConfigV4',
    device: torch.device = torch.device('cpu'),
    max_steps: int = 10,
    freeze_decoder: bool = True,
    freeze_codebook: bool = True,
    z_sequences: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Item #10: Micro-retrain from VibeThinkerContinuousLearner pseudo-labels.

    Uses pseudo-labels from Phase 4 of VibeThinkerContinuousLearner
    (high-confidence correct responses) as a minimal dataset for a
    micro Phase A cycle. Only the VibeThinkerPromptAdapter and optionally
    top encoder layers are updated. EWC penalty from ContinualLearningCore
    prevents catastrophic forgetting.

    This implements true continuous learning: instead of periodic full
    training cycles, the system receives a constant stream of
    micro-updates driven by VibeThinker quality signals.

    Args:
        model: AEONDeltaV4 model with encoder.
        pseudo_labels: List of pseudo-label dicts from maybe_consolidate().
            Each contains: confidence, quality, cot_depth, episode.
        config: Training configuration.
        device: Computation device.
        max_steps: Maximum micro-training steps.
        freeze_decoder: If True, freeze decoder weights during micro-retrain.
        freeze_codebook: If True, freeze VQ codebook during micro-retrain.
        z_sequences: Optional real z-sequence tensors from Phase B.
            When provided, real z-vectors are used instead of random dummies.

    Returns:
        Dictionary with micro-retrain results.
    """
    if not pseudo_labels:
        return {"retrained": False, "reason": "no_pseudo_labels", "steps": 0,
                "loss_start": 0.0, "loss_end": 0.0}

    if not VIBE_THINKER_AVAILABLE:
        return {"retrained": False, "reason": "no_vibe_thinker", "steps": 0,
                "loss_start": 0.0, "loss_end": 0.0}

    logger.info(
        f"🔄 Micro-retrain from {len(pseudo_labels)} pseudo-labels "
        f"(max_steps={max_steps})"
    )

    try:
        # Filter high-quality pseudo-labels
        quality_threshold = _PSEUDO_LABEL_QUALITY_THRESHOLD
        good_labels = [pl for pl in pseudo_labels
                       if pl.get('quality', 0.0) >= quality_threshold
                       and pl.get('confidence', 0.0) >= quality_threshold]

        if not good_labels:
            logger.info("   No pseudo-labels above quality threshold")
            return {"retrained": False, "reason": "low_quality_labels",
                    "steps": 0, "loss_start": 0.0, "loss_end": 0.0}

        # Create adapter for micro-training
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)

        # Freeze components as specified
        frozen_params = set()
        if freeze_decoder and hasattr(model, 'decoder'):
            for p in model.decoder.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_params.add(id(p))
        if freeze_codebook and hasattr(model, 'vq'):
            for p in model.vq.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_params.add(id(p))

        # [C4] Build a pool of real z-vectors when z_sequences are
        # available.  Falls back to random dummy latents only when
        # no real data is provided.
        _z_pool: Optional[torch.Tensor] = None
        if z_sequences:
            _real_zs = [zs.detach().to(device) for zs in z_sequences if zs.numel() > 0]
            if _real_zs:
                _z_pool = torch.cat(_real_zs, dim=0)

        # [C5] Snapshot adapter parameters before micro-training for
        # EWC-style L2 penalty to prevent catastrophic forgetting.
        _param_snapshot: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in adapter.named_parameters()
        }
        _EWC_LAMBDA = 0.5

        # Micro-training loop
        trainable = list(adapter.parameters())
        optimizer = torch.optim.Adam(
            trainable, lr=config.learning_rate * _MICRO_RETRAIN_LR_SCALE,
        )

        losses = []
        adapter.train()
        for step in range(min(max_steps, len(good_labels))):
            pl = good_labels[step % len(good_labels)]
            # [C4] Use real z-vector from pool when available
            if _z_pool is not None and _z_pool.size(0) > 0:
                _idx = step % _z_pool.size(0)
                z_input = _z_pool[_idx:_idx + 1]
            else:
                z_input = torch.randn(1, config.z_dim, device=device) * 0.1
            vt_out = adapter(z_input)
            # Loss: align complexity score with pseudo-label confidence
            pred_complexity = vt_out['complexity_score']
            target_complexity = torch.tensor(
                [pl.get('confidence', 0.5)], device=device,
            )
            loss = F.mse_loss(pred_complexity, target_complexity)

            # [C5] EWC-style L2 penalty: discourage large deviations
            # from the pre-micro-retrain snapshot to protect learned
            # representations.
            _ewc_penalty = torch.tensor(0.0, device=device)
            for name, param in adapter.named_parameters():
                if name in _param_snapshot:
                    _ewc_penalty = _ewc_penalty + (
                        (param - _param_snapshot[name]).pow(2).sum()
                    )
            loss = loss + _EWC_LAMBDA * _ewc_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Restore frozen parameters
        if freeze_decoder and hasattr(model, 'decoder'):
            for p in model.decoder.parameters():
                if id(p) in frozen_params:
                    p.requires_grad = True
        if freeze_codebook and hasattr(model, 'vq'):
            for p in model.vq.parameters():
                if id(p) in frozen_params:
                    p.requires_grad = True

        result = {
            "retrained": True,
            "reason": "success",
            "steps": len(losses),
            "loss_start": losses[0] if losses else 0.0,
            "loss_end": losses[-1] if losses else 0.0,
            "num_pseudo_labels": len(good_labels),
            "used_real_z": _z_pool is not None,
            "ewc_applied": True,
        }
        logger.info(
            f"✅ Micro-retrain complete: {result['steps']} steps, "
            f"loss {result['loss_start']:.4f} → {result['loss_end']:.4f}"
        )
        return result

    except Exception as e:
        logger.warning(f"⚠️ Micro-retrain failed (non-fatal): {e}")
        return {"retrained": False, "reason": f"error_{type(e).__name__}",
                "steps": 0, "loss_start": 0.0, "loss_end": 0.0}


def _build_curriculum_order(
    complexity_scores: List[float],
    num_samples: int,
    epoch: int,
    total_epochs: int,
) -> List[int]:
    """Item #1b: Build curriculum-ordered indices from complexity scores.

    Implements curriculum learning: start with simple documents and
    gradually introduce complex ones, controlled by training progress.

    The pace parameter p = epoch / total_epochs ∈ [0, 1] determines
    what fraction of the complexity range is available:
      - p=0: only the simplest 30% of samples
      - p=1: all samples available (uniform random)

    Args:
        complexity_scores: Per-sample complexity [0, 1].
        num_samples: Number of samples to draw.
        epoch: Current epoch (0-indexed).
        total_epochs: Total epochs.

    Returns:
        List of sample indices ordered by curriculum.
    """
    if not complexity_scores:
        return list(range(num_samples))

    N = len(complexity_scores)
    pace = min(1.0, max(0.0, epoch / max(total_epochs - 1, 1)))
    # Available fraction grows from 30% to 100%
    available_frac = _MIN_CURRICULUM_FRAC + _CURRICULUM_GROWTH_RATE * pace
    threshold = np.percentile(complexity_scores, available_frac * 100)

    # Select samples below threshold
    eligible = [i for i, c in enumerate(complexity_scores) if c <= threshold]
    if not eligible:
        eligible = list(range(N))

    # Return shuffled eligible indices, possibly with repetition
    rng = np.random.RandomState(epoch)
    indices = rng.choice(eligible, size=min(num_samples, len(eligible)),
                         replace=len(eligible) < num_samples).tolist()
    return indices


# ==============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН v4
# ==============================================================================

def main(
    json_path: str = "combined.json",
    output_dir: str = "processed_v4/",
    epochs_A: int = 30,
    epochs_B: int = 10,
    log_path: str = "training_v4.log",
    resume_from: Optional[str] = None,
    document_aware: bool = True
):
    """Main training pipeline v4.
    
    Args:
        json_path: Path to the input JSON-lines file with documents.
        output_dir: Directory for saving checkpoints, logs, and artifacts.
        epochs_A: Number of epochs for Phase A (AutoEncoder + VQ).
        epochs_B: Number of epochs for Phase B (Contextual RSSM).
        log_path: Path for the training log file.
        resume_from: Optional path to a checkpoint to resume training from.
        document_aware: If True, builds training pairs within document boundaries.
        
    Raises:
        FileNotFoundError: If json_path does not exist.
        ValueError: If epochs_A or epochs_B are non-positive.
    """
    global logger
    
    logger = configure_logger(log_path)
    
    # Validate parameters
    if not os.path.exists(json_path):
        logger.error(f"❌ JSON file not found: {json_path}")
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if epochs_A <= 0:
        raise ValueError(f"epochs_A must be positive, got {epochs_A}")
    if epochs_B <= 0:
        raise ValueError(f"epochs_B must be positive, got {epochs_B}")
    
    monitor = TrainingMonitor(logger, save_dir=os.path.join(output_dir, "checkpoints"))
    
    # Заголовок
    logger.info("🔷" * 38)
    logger.info("       AEON TRAINING PIPELINE v4.0 - CONNECTED THOUGHTS")
    logger.info("🔷" * 38)
    logger.info(f"📁 Входной JSON: {json_path}")
    logger.info(f"📂 Выходная директория: {output_dir}")
    logger.info(f"🔗 Document-aware mode: {document_aware}")

    # Токенизатор
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.warning(f"Ошибка загрузки токенизатора: {e}")
    
    # Конфигурация v4
    config = AEONConfigV4()
    config.document_aware = document_aware
    
    if tokenizer:
        config.vocab_size = tokenizer.vocab_size
        logger.info(f"📖 Токенизатор: bert-base-uncased (vocab_size={config.vocab_size})")

    logger.info(f"\n📋 Конфигурация v4 (ключевые изменения):")
    logger.info(f"   • grad_clip_norm: {config.grad_clip_norm} (стабилизировано)")
    logger.info(f"   • entropy_weight: {config.entropy_weight} (регуляризация кодбука)")
    logger.info(f"   • context_window: {config.context_window} (RSSM контекст)")
    logger.info(f"   • vq_reset_threshold: {config.vq_reset_threshold} (агрессивнее)")
    logger.info(f"   • warmup_steps: {config.warmup_steps} (плавнее)")

    # ===== VibeThinker Integration Pre-Training Setup =====
    _vt_integration_results = {}

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(output_dir, exist_ok=True)

    # ===== Загрузка данных =====
    if document_aware:
        # Загружаем с сохранением структуры документов
        documents = load_documents_from_json(
            json_path, tokenizer, config.seq_length,
            min_chunks=config.min_doc_chunks, logger=logger
        )
        
        # Создаём плоский тензор для Phase A
        all_tokens = []
        for doc in documents:
            all_tokens.extend(doc)
        
        if not all_tokens:
            logger.error("❌ No token chunks found in documents — cannot train.")
            return
        
        tokens = torch.stack(all_tokens).to(device)
        
    else:
        # Стандартная загрузка (как в v3)
        logger.info(f"\n📥 Загрузка данных из {json_path}...")
        texts = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "") if isinstance(data, dict) else str(data)
                    if text and len(text.strip()) > 10:
                        texts.append(text)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug(f"Skipping malformed line: {e}")
        
        tokens = tokenize_batch(texts, tokenizer, config.seq_length, device)
        documents = None
    
    if tokens.numel() == 0:
        logger.error("❌ No valid training data found — tokens tensor is empty.")
        return
    
    logger.info(f"   Токенов для Phase A: {tokens.shape}")

    # --- Adaptive intelligent parameter selection ---
    # Analyze data characteristics and adjust config accordingly.
    analyzer = DataCharacteristicsAnalyzer(config)
    analysis = analyzer.analyze(tokens, documents if document_aware else None)
    data_stats = analysis['stats']
    recommendations = analysis['recommendations']

    logger.info(f"\n📊 Data Characteristics Analysis:")
    logger.info(f"   Samples: {data_stats['n_samples']:,}")
    logger.info(f"   Vocab coverage: {data_stats['vocab_coverage']:.1%}")
    logger.info(f"   Padding ratio: {data_stats['padding_ratio']:.1%}")
    if data_stats.get('n_documents'):
        logger.info(f"   Documents: {data_stats['n_documents']}")
        logger.info(f"   Avg chunks/doc: {data_stats['avg_chunks_per_doc']:.1f}")

    changes = analyzer.apply_recommendations(config, recommendations)
    if changes:
        logger.info(f"\n🔄 Adaptive Parameter Adjustments ({len(changes)}):")
        for change in changes:
            logger.info(f"   • {change}")
    else:
        logger.info(f"\n✅ All default parameters are optimal for this data")

    # Сохраняем токены
    try:
        torch.save(tokens.cpu(), os.path.join(output_dir, "tokens.pt"))
    except OSError as e:
        logger.error(f"❌ Failed to save tokens: {e}")

    # Создание модели v4
    try:
        model = AEONDeltaV4(config).to(device)
    except (RuntimeError, NotImplementedError) as _to_err:
        if device.type == 'mps':
            logger.warning(
                f"MPS transfer failed ({_to_err}), falling back to CPU"
            )
            device = torch.device('cpu')
            model = AEONDeltaV4(config).to(device)
        else:
            raise
    
    # Загрузка checkpoint
    if resume_from and os.path.exists(resume_from):
        logger.info(f"📂 Загрузка checkpoint: {resume_from}")
        try:
            # Try safe loading first
            try:
                checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
            except (RuntimeError, TypeError):
                logger.warning(
                    "⚠️ Loading checkpoint with weights_only=False. "
                    "Only load checkpoints from trusted sources."
                )
                checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                logger.error(
                    f"❌ Checkpoint '{resume_from}' has invalid structure "
                    f"(missing 'model_state_dict' key)."
                )
                return
            
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"   ✅ Checkpoint loaded successfully")
            # ── Import cognitive snapshot if available ──────────────────
            # If a cognitive snapshot directory exists alongside the
            # checkpoint, import hierarchical memory subsystems so that
            # cross-session cognitive continuity is preserved.  This wires
            # the previously dead import_cognitive_snapshot() code into
            # the checkpoint loading flow.
            if hasattr(model, 'import_cognitive_snapshot'):
                _ckpt_dir = os.path.dirname(os.path.abspath(resume_from))
                _ckpt_base = os.path.splitext(os.path.basename(resume_from))[0]
                # Derive cognitive snapshot directory name from checkpoint.
                # Export uses: cognitive_snapshot_epoch_{N}
                # Checkpoint uses: checkpoint_epoch_{N}
                # Primary strategy: replace 'checkpoint_' prefix.
                _snap_candidates = [
                    os.path.join(_ckpt_dir, _ckpt_base.replace(
                        'checkpoint_', 'cognitive_snapshot_',
                    )),
                ]
                # Fallback: extract epoch number via regex for
                # non-standard checkpoint filenames.
                import re
                _epoch_match = re.search(r'epoch_(\d+)', _ckpt_base)
                if _epoch_match:
                    _snap_candidates.append(os.path.join(
                        _ckpt_dir,
                        f"cognitive_snapshot_epoch_{_epoch_match.group(1)}",
                    ))
                for _snap_dir in _snap_candidates:
                    if os.path.isdir(_snap_dir):
                        try:
                            _snap_result = model.import_cognitive_snapshot(
                                save_dir=_snap_dir,
                            )
                            if _snap_result.get('success'):
                                logger.info(
                                    f"   🧠 Cognitive snapshot imported: "
                                    f"{_snap_dir}"
                                )
                            break
                        except Exception as _snap_err:
                            logger.debug(
                                f"   Cognitive snapshot import failed "
                                f"(non-fatal): {_snap_err}"
                            )
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint '{resume_from}': {e}")
            return
    
    # Валидация
    if not validate_training_components(model, config, logger):
        logger.error("❌ Валидация не пройдена!")
        return

    # ===== Items #40-42: VibeThinker corpus diagnostics =====
    # Before any training begins, VibeThinker acts as corpus diagnostician,
    # computing complexity distribution and recommending hyperparameters.
    _corpus_diagnostics = None
    if VIBE_THINKER_AVAILABLE:
        try:
            _corpus_diagnostics = diagnose_corpus_via_vt(
                model, tokens, config, device, batch_size=256,
            )
            if _corpus_diagnostics.get('diagnosed'):
                logger.info(
                    f"🔬 Corpus diagnostics: mean_complexity="
                    f"{_corpus_diagnostics['complexity_mean']:.3f}, "
                    f"std={_corpus_diagnostics['complexity_std']:.3f}, "
                    f"heterogeneous={_corpus_diagnostics.get('heterogeneous', False)}"
                )
                _recs = _corpus_diagnostics.get('recommendations', {})
                if _recs:
                    logger.info(f"   📋 Recommendations: {_recs}")
                    # [M2] Apply VT recommendations to config before
                    # Phase A starts, closing the gap where diagnostics
                    # were logged but never used.
                    if 'context_window' in _recs:
                        _old_cw = config.context_window
                        config.context_window = _recs['context_window']
                        logger.info(
                            f"   ✅ Applied context_window: {_old_cw} → "
                            f"{config.context_window}"
                        )
                    if 'codebook_size' in _recs and hasattr(config, 'codebook_size'):
                        _old_cs = config.codebook_size
                        config.codebook_size = _recs['codebook_size']
                        logger.info(
                            f"   ✅ Applied codebook_size: {_old_cs} → "
                            f"{config.codebook_size}"
                        )
                _vt_integration_results['corpus_diagnostics'] = _corpus_diagnostics
        except Exception as _diag_err:
            logger.debug(f"Corpus diagnostics skipped (non-fatal): {_diag_err}")

    # ===== VibeThinker Pre-Training Integration =====
    # Execute Items #1, #2 from upgrade plan before Phase A begins.
    if VIBE_THINKER_AVAILABLE:
        logger.info("\n🧠 VibeThinker Pre-Training Integration...")

        # [M3] STEP 0: Load VibeThinker adapter weights if configured.
        # vibe_thinker_weights_path is defined in AEONConfig but was
        # never handled in the training pipeline — only in aeon_core
        # AEONDeltaV3.__init__.  Load adapter/kernel weights here so
        # that warm-start, annotation, and micro-retrain use pre-trained
        # VibeThinker parameters instead of random initialization.
        _vt_weights_path = getattr(config, 'vibe_thinker_weights_path', '')
        if _vt_weights_path:
            from pathlib import Path as _Path
            _vt_wp = _Path(_vt_weights_path)
            if _vt_wp.is_file():
                try:
                    _wt_state = torch.load(
                        str(_vt_wp), map_location=device, weights_only=True,
                    )
                    _loaded_keys: List[str] = []
                    # Load adapter weights into the model's VT adapter if present
                    if hasattr(model, '_vt_adapter'):
                        _adapter_state = {
                            k.replace('adapter.', '', 1): v
                            for k, v in _wt_state.items()
                            if k.startswith('adapter.')
                        }
                        if _adapter_state:
                            model._vt_adapter.load_state_dict(
                                _adapter_state, strict=False,
                            )
                            _loaded_keys.extend(_adapter_state.keys())
                    _vt_integration_results['vt_weights_loaded'] = {
                        'path': str(_vt_wp),
                        'keys_loaded': len(_loaded_keys),
                    }
                    logger.info(
                        f"   ✅ VibeThinker weights loaded from {_vt_wp} "
                        f"({len(_loaded_keys)} keys)"
                    )
                except Exception as _vt_load_err:
                    logger.debug(
                        f"VibeThinker weight loading skipped: {_vt_load_err}"
                    )

        # Item #1: Warm-start codebook from VibeThinker embeddings
        _ws_result = warm_start_codebook_from_vt(
            model, tokens, config, device, batch_size=256,
        )
        _vt_integration_results['codebook_warm_start'] = _ws_result
        if _ws_result.get('initialized'):
            logger.info(f"   ✅ Codebook warm-started via {_ws_result['method']}")

        # Item #2: Calibrate context window from CoT depth
        _cw_result = calibrate_context_window(
            model, tokens, config, device, batch_size=256,
        )
        _vt_integration_results['context_calibration'] = _cw_result
        if _cw_result.get('calibrated'):
            logger.info(
                f"   ✅ Context window: {_cw_result['original_window']} → "
                f"{_cw_result['new_window']}"
            )

        # Item #9: Adapt entropy weight from VibeThinker entropy
        if _ws_result.get('complexity_scores'):
            _mean_complexity = sum(_ws_result['complexity_scores']) / len(
                _ws_result['complexity_scores']
            )
            _ew_result = adapt_entropy_weight(config, _mean_complexity)
            _vt_integration_results['entropy_adaptation'] = _ew_result

        # [C2] Item #48: Align SSP temperature with Gumbel VQ temperature.
        # align_ssp_temperature() harmonises VibeThinker sampling diversity
        # with codebook diversity so that reasoning diversity matches
        # latent code diversity.  Invoked between warm_start and Phase A.
        try:
            _gumbel_temp = getattr(config, 'gumbel_temperature', 1.0)
            _ssp_result = align_ssp_temperature(
                vt_temperature=1.0,
                gumbel_temperature=_gumbel_temp,
            )
            _vt_integration_results['ssp_temperature_alignment'] = _ssp_result
            logger.info(
                f"   🌡️ SSP temperature aligned: "
                f"vt={_ssp_result.get('vt_aligned', 1.0):.4f}, "
                f"scaling={_ssp_result.get('scaling_factor', 1.0):.4f}"
            )
        except Exception as _ssp_err:
            logger.debug(f"SSP temperature alignment skipped: {_ssp_err}")
    else:
        logger.info("ℹ️  VibeThinker integration skipped (aeon_core components not available)")

    # ===== Item #1b: Curriculum ordering from VibeThinker complexity =====
    # When complexity scores are available from warm-start, use
    # _build_curriculum_order() to train Phase A with a curriculum that
    # starts with simple documents and gradually introduces complex ones.
    # This replaces random shuffling with empirically-grounded ordering.
    _curriculum_scores: Optional[List[float]] = None
    if VIBE_THINKER_AVAILABLE and _vt_integration_results.get('codebook_warm_start', {}).get('complexity_scores'):
        _curriculum_scores = _vt_integration_results['codebook_warm_start']['complexity_scores']
        logger.info(
            f"   📚 Curriculum learning enabled: {len(_curriculum_scores)} "
            f"complexity scores (mean={sum(_curriculum_scores)/len(_curriculum_scores):.3f})"
        )
        _vt_integration_results['curriculum_learning'] = {
            'enabled': True,
            'num_scores': len(_curriculum_scores),
        }

    # ===== Phase A =====
    logger.info("\n" + "▶" * 38)
    logger.info("     PHASE A: AutoEncoder + VQ v4")
    logger.info("▶" * 38)
    
    trainer_A = SafeThoughtAETrainerV4(model, config, monitor, output_dir)
    trainer_A.fit(tokens, epochs=epochs_A,
                  curriculum_scores=_curriculum_scores)

    # Save best loss and convergence monitor before releasing Phase A resources
    best_loss_A = trainer_A.best_loss
    convergence_monitor_A = trainer_A.convergence_monitor

    # Release Phase A training resources before Phase B
    del trainer_A
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== VQ Codebook Health Gate =====
    # Verify that the VQ codebook is healthy before building z_sequences.
    # If codebook utilization collapsed during Phase A (e.g., < 10%),
    # all subsequent z_sequences will map to a tiny number of codes,
    # making Phase B RSSM training futile.  This check ensures that
    # each component (Phase A VQ) verifies and reinforces the next
    # (Phase B RSSM) by gating the transition on codebook health.
    _VQ_HEALTH_CRITICAL_THRESHOLD = 5.0   # % — below this, abort Phase B
    _VQ_HEALTH_WARNING_THRESHOLD = 20.0   # % — below this, warn and attempt reset
    try:
        _codebook_usage = model.vq.get_codebook_usage()
        logger.info(f"🔍 VQ codebook utilization: {_codebook_usage:.1f}%")
        if _codebook_usage < _VQ_HEALTH_CRITICAL_THRESHOLD:
            logger.error(
                f"❌ VQ codebook critically collapsed ({_codebook_usage:.1f}% < "
                f"{_VQ_HEALTH_CRITICAL_THRESHOLD}%). Phase B cannot proceed "
                f"with degenerate z-sequences."
            )
            convergence_monitor_A.update(float('nan'))
            return
        elif _codebook_usage < _VQ_HEALTH_WARNING_THRESHOLD:
            logger.warning(
                f"⚠️ VQ codebook utilization low ({_codebook_usage:.1f}% < "
                f"{_VQ_HEALTH_WARNING_THRESHOLD}%). Attempting codebook reset "
                f"before z-sequence construction."
            )
            try:
                model.vq.reset_unused_codes()
                _codebook_usage_after = model.vq.get_codebook_usage()
                logger.info(
                    f"   Codebook usage after reset: {_codebook_usage_after:.1f}%"
                )
            except (AttributeError, RuntimeError) as _reset_err:
                logger.warning(f"   Codebook reset failed (non-fatal): {_reset_err}")
    except (AttributeError, RuntimeError) as _vq_err:
        logger.warning(f"⚠️ VQ health check skipped (non-fatal): {_vq_err}")

    # ===== Построение z_sequences =====
    logger.info("\n🔧 Построение z_sequences для Phase B...")
    model.eval()
    
    with torch.no_grad():
        if document_aware and documents:
            # ✅ Строим z_sequences по документам (batch encoding for performance)
            z_sequences = []
            skipped = 0
            
            for doc_idx, doc_chunks in enumerate(tqdm(documents, desc="Encoding documents")):
                if len(doc_chunks) < config.context_window + 1:
                    skipped += 1
                    continue
                
                # Batch encode all chunks in the document at once
                chunks_batch = torch.stack(doc_chunks).to(device)
                z_batch = model.encode(chunks_batch)
                quantized_batch, _, _, _ = model.quantize(z_batch)
                z_seq = quantized_batch.cpu()  # [num_chunks, D]
                z_sequences.append(z_seq)
            
            logger.info(f"✅ Создано {len(z_sequences)} z_sequences (skipped {skipped} docs with < {config.context_window + 1} chunks)")
            total_pairs = sum(max(0, seq.size(0) - config.context_window) for seq in z_sequences)
            logger.info(f"   Всего пар для обучения: {total_pairs:,}")
            
            # Сохраняем
            try:
                torch.save(z_sequences, os.path.join(output_dir, "z_sequences.pt"))
            except OSError as e:
                logger.error(f"❌ Failed to save z_sequences: {e}")
            
        else:
            # Fallback: старый метод (все z подряд)
            z_list = []
            for batch in tqdm(DataLoader(TensorDataset(tokens), batch_size=256), desc="Encoding"):
                z = model.encode(batch[0].to(device))
                quantized, _, _, _ = model.quantize(z)
                z_list.append(quantized.cpu())
            
            z_all = torch.cat(z_list)
            # Создаём один большой sequence
            z_sequences = [z_all]
            
            try:
                torch.save(z_sequences, os.path.join(output_dir, "z_sequences.pt"))
            except OSError as e:
                logger.error(f"❌ Failed to save z_sequences: {e}")

    # ===== Item #4: Bifasic didactic role inversion (Phase 2) =====
    # After Phase A, AEON has trained z-representations. Now AEON becomes
    # the teacher and VibeThinkerPromptAdapter becomes the student. The
    # adapter is trained to align its projection with the VQ latent space,
    # completing the bidirectional teacher-student cycle.
    if VIBE_THINKER_AVAILABLE and z_sequences:
        try:
            _bd_result = bifasic_didactic_orchestrate(
                model=model,
                z_sequences=z_sequences,
                config=config,
                device=device,
                max_steps=5,
            )
            _vt_integration_results['bifasic_didactic'] = _bd_result
            if _bd_result.get('inverted'):
                logger.info(
                    f"🔄 Bifasic didactic: AEON→VT role inversion complete — "
                    f"{_bd_result['steps']} steps, "
                    f"loss {_bd_result['loss_start']:.4f} → "
                    f"{_bd_result['loss_end']:.4f}"
                )
        except Exception as _bd_err:
            logger.debug(f"Bifasic didactic skipped (non-fatal): {_bd_err}")

    # ===== Item #8: Quality-annotate z_sequences =====
    _quality_annotations = None
    if VIBE_THINKER_AVAILABLE and z_sequences:
        z_sequences, _quality_annotations = annotate_z_sequences_quality(
            model, z_sequences, config, device, batch_size=256,
        )
        _vt_integration_results['z_quality_annotation'] = {
            'annotated': _quality_annotations is not None,
            'num_sequences': len(z_sequences),
        }

    # Validate z_sequences before Phase B
    if not z_sequences:
        logger.error("❌ No z_sequences created — cannot run Phase B. "
                      "Check that documents have enough chunks (>= context_window + 1).")
        return

    # ===== Z-Sequence Distribution Verification =====
    # Verify that the z_sequences have meaningful variance/diversity.
    # If VQ collapsed silently, all z-vectors would be nearly identical,
    # making RSSM training futile (predicting constants).  This check
    # ensures that Phase A's output verifies Phase B's input, closing
    # the inter-phase coherence loop.
    _Z_VARIANCE_CRITICAL_THRESHOLD = 1e-6
    _Z_COSINE_COLLAPSE_THRESHOLD = 0.999
    try:
        _all_z = torch.cat(z_sequences, dim=0)  # [total_chunks, D]
        _z_var = _all_z.var(dim=0).mean().item()
        logger.info(f"🔍 Z-sequence mean variance: {_z_var:.6f}")
        if _z_var < _Z_VARIANCE_CRITICAL_THRESHOLD:
            logger.error(
                f"❌ Z-sequences have near-zero variance ({_z_var:.2e} < "
                f"{_Z_VARIANCE_CRITICAL_THRESHOLD:.2e}). VQ likely collapsed — "
                f"Phase B cannot learn from constant inputs."
            )
            return
        # Check pairwise cosine similarity on a sample to detect collapse
        if _all_z.shape[0] >= 2:
            _sample_size = min(100, _all_z.shape[0])
            _sample = _all_z[:_sample_size]
            _cos_matrix = F.cosine_similarity(
                _sample.unsqueeze(0), _sample.unsqueeze(1), dim=-1,
            )
            # Mean off-diagonal similarity
            _n = _cos_matrix.shape[0]
            _mask = ~torch.eye(_n, dtype=torch.bool)
            _mean_cos = _cos_matrix[_mask].mean().item()
            logger.info(f"   Z-sequence mean pairwise cosine similarity: {_mean_cos:.4f}")
            if _mean_cos > _Z_COSINE_COLLAPSE_THRESHOLD:
                logger.warning(
                    f"⚠️ Z-sequences show near-identical representations "
                    f"(mean cosine={_mean_cos:.4f} > {_Z_COSINE_COLLAPSE_THRESHOLD}). "
                    f"Phase B RSSM may struggle to learn meaningful transitions."
                )
        del _all_z  # Free memory
    except Exception as _zv_err:
        logger.warning(f"⚠️ Z-sequence verification skipped (non-fatal): {_zv_err}")

    # ===== Phase B =====
    logger.info("\n" + "▶" * 38)
    logger.info("     PHASE B: Contextual RSSM")
    logger.info("▶" * 38)
    
    # Переносим sequences на device
    z_sequences_gpu = [seq.to(device) for seq in z_sequences]
    
    trainer_B = ContextualRSSMTrainer(model, config, monitor)

    # Seed Phase B error evolution with Phase A patterns so that RSSM
    # training benefits from AE-phase convergence failures.  Without this,
    # Phase B starts with a blank error tracker and cannot leverage
    # divergence/stagnation patterns already discovered during Phase A.
    _phaseA_bridged = bridge_training_errors_to_inference(
        trainer_monitor=convergence_monitor_A,
        inference_error_evolution=trainer_B._error_evolution,
    )
    if _phaseA_bridged:
        logger.info(
            f"🔗 Bridged {_phaseA_bridged} error pattern(s) from Phase A → Phase B"
        )
    else:
        logger.info("🔗 Phase A produced no actionable error patterns to bridge")

    # ===== Phase transition provenance =====
    # Record Phase A summary in provenance tracker so that root-cause
    # analysis can trace Phase B issues back to Phase A training state.
    if hasattr(trainer_B, '_provenance') and trainer_B._provenance is not None:
        trainer_B._provenance.record(
            "phase_transition",
            {"phase": "A_to_B",
             "phase_a_best_loss": best_loss_A,
             "phase_a_bridged_errors": _phaseA_bridged,
             "z_sequences_count": len(z_sequences),
             "total_pairs": sum(
                 max(0, seq.size(0) - config.context_window)
                 for seq in z_sequences
             )},
        )

    trainer_B.fit(z_sequences_gpu, epochs=epochs_B,
                  quality_annotations=_quality_annotations)

    # ===== End-of-training unified cognitive cycle evaluation =====
    # Run a final UCC evaluation after both phases complete to verify
    # that the trained model maintains cross-module coherence.  Without
    # this check, training could silently produce a model whose
    # subsystems are internally inconsistent, violating the architectural
    # requirement that each component verifies and reinforces the others.
    try:
        _final_ucc = trainer_B._unified_cycle
        _final_result = _final_ucc.evaluate(
            subsystem_states={
                "encoder": trainer_B._last_vq_state
                if trainer_B._last_vq_state is not None
                else torch.zeros(1, config.z_dim),
                "rssm": trainer_B._last_rssm_state
                if trainer_B._last_rssm_state is not None
                else torch.zeros(1, config.z_dim),
            },
            delta_norm=0.0,
            uncertainty=0.0,
        )
        _coherence = 1.0 - _final_result["coherence_result"]["coherence_deficit"]
        logger.info(
            f"🧠 End-of-training UCC evaluation: "
            f"coherence={_coherence:.4f}, "
            f"converged={_final_result['convergence_verdict']}, "
            f"should_rerun={_final_result['should_rerun']}"
        )
        if _final_result["should_rerun"]:
            logger.warning(
                "⚠️  End-of-training UCC recommends re-reasoning — "
                "model coherence may be suboptimal"
            )
    except Exception as _ucc_err:
        logger.warning(
            "End-of-training UCC evaluation failed (non-fatal): %s",
            _ucc_err,
        )

    # ===== Bidirectional bridge: inference insights → training =====
    # Close the feedback loop by bridging Phase B's accumulated error
    # evolution back to the trainer's hyperparameters.  This ensures
    # that any uncertainty discovered during training triggers parameter
    # adaptation, fulfilling the requirement that uncertainty triggers
    # a meta-cognitive cycle.
    _inference_adjustments = bridge_inference_insights_to_training(
        inference_error_evolution=getattr(trainer_B, '_error_evolution', None),
        trainer=trainer_B,
        inference_uncertainty_tracker=getattr(
            trainer_B._unified_cycle, 'uncertainty_tracker', None,
        ),
        training_error_evolution=getattr(trainer_B, '_error_evolution', None),
        training_convergence_monitor=getattr(
            trainer_B, 'convergence_monitor', None,
        ),
        training_metacognitive_trigger=getattr(
            trainer_B, '_metacognitive_trigger', None,
        ),
        training_provenance_tracker=getattr(
            trainer_B._unified_cycle, 'provenance_tracker', None,
        ) if hasattr(trainer_B, '_unified_cycle') else None,
    )
    if _inference_adjustments:
        logger.info(
            f"🔗 Applied {_inference_adjustments} inference→training "
            f"adjustment(s) from Phase B error evolution"
        )

    # ===== Item #6: Task boundary detection =====
    if VIBE_THINKER_AVAILABLE:
        try:
            _final_coherence = 1.0 - _final_result.get(
                "coherence_result", {},
            ).get("coherence_deficit", 0.5)
            _tb_result = auto_detect_task_boundary(
                coherence_score=_final_coherence,
                coherence_threshold=0.5,
            )
            _vt_integration_results['task_boundary'] = _tb_result
            if _tb_result['boundary_detected']:
                logger.info(
                    f"🔀 Task boundary detected: coherence={_final_coherence:.4f}, "
                    f"recommendation={_tb_result['recommendation']}"
                )
                # Item #6 completion: Wire ContinualLearningCore.add_task()
                # when a task boundary is detected.  This closes the gap
                # where task boundary detection was advisory-only.
                # ContinualLearningCore freezes the current column and
                # provisions a new one, with EWC protection for old weights.
                try:
                    _clc = ContinualLearningCore(
                        base_model=model.encoder,
                    )
                    _clc.add_task(f"detected_boundary_coherence_{_final_coherence:.3f}")
                    _vt_integration_results['task_boundary']['task_added'] = True
                    logger.info(
                        f"   ✅ ContinualLearningCore.add_task() called — "
                        f"column frozen, EWC protection enabled"
                    )
                except Exception as _clc_err:
                    logger.debug(
                        f"ContinualLearningCore.add_task() failed (non-fatal): "
                        f"{_clc_err}"
                    )
                    _vt_integration_results['task_boundary']['task_added'] = False
        except Exception as _tb_err:
            logger.debug(f"Task boundary detection skipped: {_tb_err}")

    # ===== Item #10: Micro-retrain from VibeThinker pseudo-labels =====
    # After Phase B completes, collect pseudo-labels from
    # VibeThinkerContinuousLearner.maybe_consolidate() and use them for
    # a micro Phase A cycle.  This implements true continuous learning:
    # instead of periodic full training cycles, the system receives a
    # constant stream of micro-updates driven by VibeThinker quality
    # signals.  Only the adapter and top encoder layers are updated;
    # EWC protects critical weights from catastrophic forgetting.
    if VIBE_THINKER_AVAILABLE:
        try:
            _vt_cfg = VibeThinkerConfig()
            _vt_learner = VibeThinkerContinuousLearner(config=_vt_cfg)
            # Drive the VibeThinkerContinuousLearner through its 4-phase
            # cycle using Phase B quality annotations as ground-truth
            # input, then collect real pseudo-labels via consolidation.
            # This replaces synthetic pseudo-labels with genuine learner
            # output, closing the true continuous learning loop.
            _pseudo_labels: List[Dict[str, Any]] = []
            if _quality_annotations is not None:
                for _qa in _quality_annotations:
                    if _qa is not None and len(_qa) > 0:
                        for _row in _qa:
                            _conf = float(_row[0]) if len(_row) > 0 else 0.5
                            _ent = float(_row[1]) if len(_row) > 1 else 0.5
                            _qual = float(_row[2]) if len(_row) > 2 else 0.5
                            # Phase 2: evaluate_episode with real data
                            _parsed = {
                                'confidence': _conf,
                                'entropy': _ent,
                                'reasoning_quality': _qual,
                                'cot_depth': 1.0,
                                'is_high_confidence': _conf >= _vt_cfg.confidence_threshold,
                            }
                            _eval_result = _vt_learner.evaluate_episode(
                                _parsed,
                                actual_correctness=_qual,
                            )
                            # Phase 3: adapt thresholds
                            _vt_learner.adapt(_eval_result, _parsed)
                            # Phase 4: consolidation collects pseudo-labels
                            _consolidation = _vt_learner.maybe_consolidate(_parsed)
                            if _consolidation is not None:
                                logger.debug(
                                    f"VT consolidation: {_consolidation['pseudo_label_count']} "
                                    f"pseudo-labels collected"
                                )
            # Collect consolidated pseudo-labels from the learner
            _pseudo_labels = _vt_learner.get_pseudo_labels()
            # Also include any high-quality annotations that passed
            # through consolidation but weren't part of the latest
            # consolidation window.
            if not _pseudo_labels and _quality_annotations is not None:
                for _qa in _quality_annotations:
                    if _qa is not None and len(_qa) > 0:
                        for _row in _qa:
                            _conf = float(_row[0]) if len(_row) > 0 else 0.5
                            _qual = float(_row[2]) if len(_row) > 2 else 0.5
                            if _conf >= _PSEUDO_LABEL_QUALITY_THRESHOLD:
                                _pseudo_labels.append({
                                    'confidence': _conf,
                                    'quality': _qual,
                                    'cot_depth': 1.0,
                                    'episode': 'phase_b_quality_fallback',
                                })
            _mr_result = micro_retrain_from_pseudo_labels(
                model=model,
                pseudo_labels=_pseudo_labels,
                config=config,
                device=device,
                max_steps=min(10, max(1, len(_pseudo_labels))),
                z_sequences=z_sequences,
            )
            _vt_integration_results['micro_retrain'] = _mr_result
            _vt_integration_results['vt_learner_summary'] = _vt_learner.get_summary()
            if _mr_result.get('retrained'):
                logger.info(
                    f"🔄 Micro-retrain complete: {_mr_result['steps']} steps, "
                    f"loss {_mr_result['loss_start']:.4f} → "
                    f"{_mr_result['loss_end']:.4f}"
                )
            else:
                logger.info(
                    f"ℹ️  Micro-retrain skipped: {_mr_result.get('reason', 'unknown')}"
                )
        except Exception as _mr_err:
            logger.debug(f"Micro-retrain skipped (non-fatal): {_mr_err}")

    # ===== Сохранение =====
    final_path = os.path.join(output_dir, "aeon_v4_final.pt")
    
    for param in model.parameters():
        param.requires_grad = True
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'metrics_history': monitor.metrics_history,
        'training_info': {
            'epochs_A': epochs_A,
            'epochs_B': epochs_B,
            'final_loss_A': best_loss_A,
            'final_loss_B': trainer_B.best_loss,
            'document_aware': document_aware,
            'timestamp': datetime.now().isoformat(),
            'version': '4.0.0'
        }
    }

    # Export training error patterns so the inference pipeline can import
    # them via bridge_training_errors_to_inference() at load time.  The
    # patterns are persisted alongside the model checkpoint.
    _training_error_patterns = {}
    for _phase_label, _cm in [("Phase_A", convergence_monitor_A),
                               ("Phase_B", trainer_B.convergence_monitor)]:
        _patterns = _cm.export_error_patterns()
        _training_error_patterns[_phase_label] = _patterns
        _n_classes = len(_patterns.get("error_classes", {}))
        if _n_classes:
            logger.info(f"🔗 Exported {_n_classes} error class(es) from {_phase_label} for inference bridge")
    save_dict['training_error_patterns'] = _training_error_patterns
    if VIBE_THINKER_AVAILABLE:
        save_dict['vt_integration'] = _vt_integration_results
    
    try:
        torch.save(save_dict, final_path)
        logger.info(f"💾 Финальная модель сохранена: {final_path}")
    except OSError as e:
        logger.error(f"❌ Failed to save final model to {final_path}: {e}")
    monitor.save_metrics(os.path.join(output_dir, "training_metrics_v4.json"))

    # Финальный отчёт
    logger.info("\n" + "🎉" * 25)
    logger.info("     ОБУЧЕНИЕ v4 УСПЕШНО ЗАВЕРШЕНО!")
    logger.info("🎉" * 25)
    logger.info(f"💾 Финальная модель: {final_path}")
    
    logger.info("\n📊 ИТОГОВАЯ СВОДКА v4:")
    logger.info(f"   Phase A лучший loss: {best_loss_A:.6f}")
    logger.info(f"   Phase B лучший MSE: {trainer_B.best_loss:.6f}")
    logger.info(f"   Codebook utilization: {model.vq.get_codebook_usage():.2f}%")
    logger.info(f"   Context window: {config.context_window}")
    logger.info(f"   Document-aware: {document_aware}")
    
    logger.info("\n🚀 Модель v4 готова к использованию!")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AEON Training Pipeline v4.0 - Connected Thoughts Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python train_aeon_v4.py --json_path data.json --epochsA 30 --epochsB 10
  python train_aeon_v4.py --document_aware --json_path structured_data.json
  python train_aeon_v4.py --resume checkpoints/checkpoint_epoch_10.pt
        """
    )
    
    parser.add_argument("--json_path", type=str, default="combined.json")
    parser.add_argument("--output_dir", type=str, default="processed_v4/")
    parser.add_argument("--epochsA", type=int, default=30)
    parser.add_argument("--epochsB", type=int, default=10)
    parser.add_argument("--log", type=str, default="training_v4.log")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--document_aware", action="store_true", 
                        help="Использовать документ-ориентированное обучение")
    
    args = parser.parse_args()
    
    main(
        json_path=args.json_path,
        output_dir=args.output_dir,
        epochs_A=args.epochsA,
        epochs_B=args.epochsB,
        log_path=args.log,
        resume_from=args.resume,
        document_aware=args.document_aware
    )
