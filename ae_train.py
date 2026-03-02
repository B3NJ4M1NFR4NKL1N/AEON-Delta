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
except ImportError:
    AEON_CORE_AVAILABLE = False

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

        def set_causal_trace(self, trace) -> None:
            """Attach a causal trace buffer for automatic episode propagation."""
            self._causal_trace = trace

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
            return summary

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

        Monitors the same twelve signals as
        ``aeon_core.MetaCognitiveRecursionTrigger`` and uses weighted-sum
        evaluation so that standalone training can detect uncertainty,
        divergence, and coherence deficits that warrant re-reasoning.
        """

        _DEFAULT_WEIGHT = 1.0 / 12.0

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
                "trigger_score": trigger_score,
                "tightened_threshold": (
                    self.trigger_threshold * self.tightening_factor
                ),
                "extra_iterations": self.extra_iterations,
                "triggers_active": triggers_active,
                "recursion_count": self._recursion_count,
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
                "low_output_reliability": "uncertainty",
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
            }
            for cls_name, cls_stats in error_classes.items():
                signal = _class_to_signal.get(cls_name)
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
                    logger.debug(
                        "Metacognitive weight adaptation failed in training: %s",
                        _ae_err,
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
                "root_cause_trace": {},
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
        
        # 1. Commitment loss
        commitment_loss = F.mse_loss(z, quantized.detach())
        
        # 2. Codebook loss
        codebook_loss = F.mse_loss(quantized, z.detach())
        
        # 3. ✅ НОВОЕ: Entropy regularization
        # Поощряет равномерное использование кодов
        # Use soft probabilities from distances for differentiable entropy
        soft_probs = F.softmax(-distances, dim=-1)  # [B, num_embeddings]
        avg_probs = soft_probs.mean(dim=0)  # [num_embeddings]
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        max_entropy = math.log(self.num_embeddings) if self.num_embeddings > 1 else 1.0
        entropy_loss = 1.0 - entropy / max_entropy
        
        # Общий loss
        loss = codebook_loss + self.commitment_cost * commitment_loss + self.entropy_weight * entropy_loss
        
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        
        # EMA update
        if self.training:
            self._update_ema(z, indices)
        
        # Статистика
        stats = self._compute_stats(indices)
        stats['entropy_loss'] = entropy_loss.item()
        
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

        with torch.no_grad():
            perplexity = torch.exp(recon_loss.clamp(max=80)).item()
            pred_tokens = logits[:, :-1].argmax(dim=-1)
            accuracy = (pred_tokens == tokens[:, 1:]).float().mean().item() * 100
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
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
            log_every_batch: int = 10):
        
        loader = DataLoader(
            TensorDataset(tokenized_tensor), 
            batch_size=self.config.batch_size, 
            shuffle=True,
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
        
        self.optimizer.zero_grad()
        
        for epoch in range(epochs):
            self.monitor.start_epoch(epoch, epochs)
            
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
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
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
                    if outputs is not None and hasattr(outputs, '__contains__') and 'input_batch' in outputs:
                        try:
                            _rerun_out = self.train_step(outputs['input_batch'])
                            _rerun_loss = _rerun_out['total_loss']
                            if (not (math.isnan(_rerun_loss.item()) or math.isinf(_rerun_loss.item()))
                                    and _rerun_loss.item() < epoch_metrics["total"] * num_steps):
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
        except OSError as e:
            logger.error(f"   ❌ Failed to save checkpoint: {e}")


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
            
        self.trainable_params = list(model.rssm.parameters())
        
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

    def train_step(self, z_context: torch.Tensor, z_target: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for contextual RSSM.
        
        Args:
            z_context: [B, K, D] — context from K previous z states
                (B=batch size, K=context window length, D=latent dimension)
            z_target: [B, D] — target z_{t+1}
            
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

    def fit(self, z_sequences: List[torch.Tensor], epochs: int = 10, 
            batch_size: int = 128, log_every_batch: int = 5):
        """
        Args:
            z_sequences: List of [num_chunks, D] tensors, one per document
        """
        # Создаём dataset из контекстных окон
        K = self.config.context_window
        
        all_contexts = []
        all_targets = []
        
        for z_seq in z_sequences:
            num_z = z_seq.size(0)
            if num_z >= K + 1:
                for i in range(K, num_z):
                    context = z_seq[i-K:i]  # [K, D]
                    target = z_seq[i]  # [D]
                    all_contexts.append(context)
                    all_targets.append(target)
        
        if len(all_contexts) == 0:
            logger.warning("⚠️ Недостаточно данных для обучения RSSM")
            return
        
        contexts_tensor = torch.stack(all_contexts)  # [N, K, D]
        targets_tensor = torch.stack(all_targets)  # [N, D]
        
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
                "decoder_cross_loss": 0.0,
            }
            valid_batches = 0
            _decoder_invalid_count = 0
            
            for batch_idx, (ctx_batch, tgt_batch) in enumerate(loader):
                ctx_batch = ctx_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                metrics = self.train_step(ctx_batch, tgt_batch)
                
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
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
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
                    )

            # --- Periodic inference↔training bridge (Phase B) ---
            # Mirrors Phase A's periodic bridging: every bridge_interval
            # epochs, synchronize accumulated error patterns back into
            # training hyperparameters.
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
                )
            elif self.status == 'stagnating':
                self._error_evolution.record_episode(
                    error_class="training_stagnation",
                    strategy_used=recommendation,
                    success=False,
                    metadata={"trend": trend, "loss_value": loss_value},
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
        except AttributeError:
            pass  # Older ConvergenceMonitor without set_error_evolution

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

    Args:
        inference_error_evolution: The inference pipeline's
            ``CausalErrorEvolutionTracker`` with accumulated patterns.
        trainer: The ``SafeThoughtAETrainerV4`` instance whose
            hyperparameters will be adapted.
        inference_uncertainty_tracker: Optional ``DirectionalUncertaintyTracker``
            whose per-module breakdown informs which training components
            need the most attention.

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
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint '{resume_from}': {e}")
            return
    
    # Валидация
    if not validate_training_components(model, config, logger):
        logger.error("❌ Валидация не пройдена!")
        return
    
    # ===== Phase A =====
    logger.info("\n" + "▶" * 38)
    logger.info("     PHASE A: AutoEncoder + VQ v4")
    logger.info("▶" * 38)
    
    trainer_A = SafeThoughtAETrainerV4(model, config, monitor, output_dir)
    trainer_A.fit(tokens, epochs=epochs_A)

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

    trainer_B.fit(z_sequences_gpu, epochs=epochs_B)

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
    )
    if _inference_adjustments:
        logger.info(
            f"🔗 Applied {_inference_adjustments} inference→training "
            f"adjustment(s) from Phase B error evolution"
        )

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
