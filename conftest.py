"""
AEON-Delta RMT v3.1 — Cognitive Test Architecture
═══════════════════════════════════════════════════

Pytest plugin implementing the 7-category cognitive test organisation system
with comprehensive per-test metrics collection, structured JSON output, and
ErrorEvolutionTracker integration for continuous improvement analytics.

Categories
──────────
 1. AGI Axioms Verification          (mutual verification, metacognitive responsiveness,
                                       root-cause traceability, convergence quality)
 2. Cognitive Activation System       (metacognitive trigger, UCC, cognitive cycle,
                                       activation endpoints, signal routing)
 3. Component Integration             (cross-module bridges, pipeline deps, provenance,
                                       feedback bus, causal trace, coherence registry)
 4. Learning & Adaptation             (meta-loop, error evolution, weight adaptation,
                                       continual learning, curriculum, exploration)
 5. Security & Reliability            (safety guards, quarantine, circuit breaker,
                                       NaN policy, tensor guard, sanitisation)
 6. Performance & Scalability         (benchmarks, latency, throughput, memory,
                                       parallel scan, chunked processing)
 7. Special Scenarios & Stress Tests  (degraded mode, chaos, edge cases, concurrency,
                                       thread safety, reactivation, fallback paths)

Output
──────
Each test produces a structured record conforming to the AEON cognitive analysis
schema, written to ``cognitive_test_report.json`` after the session completes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ─── Try importing psutil (available in requirements.txt) ────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ─── Try importing torch for GPU detection ───────────────────────────────────
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY DEFINITIONS — 7 Cognitive Test Categories
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    1: "AGI Axioms Verification",
    2: "Cognitive Activation System",
    3: "Component Integration",
    4: "Learning & Adaptation",
    5: "Security & Reliability",
    6: "Performance & Scalability",
    7: "Special Scenarios & Stress Tests",
}

# Keywords → category mapping (longest-match wins, same as _KEYWORD_NODE_MAP)
# Each tuple: (keyword, category_number, phase_hint)
#   phase_hint ∈ {init, probe, verify, emerge}
_CATEGORY_KEYWORDS: List[Tuple[str, int, str]] = [
    # ── Category 1: AGI Axioms ────────────────────────────────────────────
    ("axiom",                      1, "verify"),
    ("mutual_verification",        1, "verify"),
    ("root_cause_traceability",    1, "verify"),
    ("metacognitive_responsiveness", 1, "verify"),
    ("convergence_quality",        1, "verify"),
    ("cognitive_unity",            1, "verify"),
    ("unity_score",                1, "verify"),
    ("coherence_deficit",          1, "verify"),
    ("three_axiom",                1, "verify"),
    ("emergence_summary",          1, "emerge"),
    ("emergence_trace",            1, "emerge"),
    ("emergence_axiom",            1, "emerge"),
    ("emergence_report",           1, "emerge"),
    ("emergence_state",            1, "emerge"),
    ("emergence_signal",           1, "emerge"),
    ("emergence_verdict",          1, "emerge"),
    ("emergence_eval",             1, "emerge"),
    ("emergence_patch",            1, "emerge"),
    ("convergence_arbiter",        1, "verify"),
    ("convergence_monitor",        1, "verify"),
    ("km_convergence",             1, "verify"),
    ("km_alpha",                   1, "verify"),
    ("anderson_acceleration",      1, "verify"),
    ("anderson_perturbation",      1, "verify"),
    ("lipschitz",                  1, "verify"),
    ("contraction",                1, "verify"),
    ("verify_convergence",         1, "verify"),
    ("coherence_verifier",         1, "verify"),
    ("coherence_registry",         1, "verify"),
    ("module_coherence",           1, "verify"),
    ("coherence_score",            1, "verify"),
    ("coherence_repair",           1, "verify"),
    ("verify_and_reinforce",       1, "verify"),
    ("cross_module_check",         1, "verify"),

    # ── Category 2: Cognitive Activation ──────────────────────────────────
    ("metacognitive_trigger",      2, "probe"),
    ("metacognitive_state",        2, "probe"),
    ("metacognitive_causal",       2, "probe"),
    ("metacognitive_adaptation",   2, "probe"),
    ("cognitive_cycle",            2, "probe"),
    ("unified_cognitive",          2, "probe"),
    ("ucc",                        2, "probe"),
    ("cognitive_executive",        2, "probe"),
    ("deeper_meta_loop",           2, "probe"),
    ("deeper_meta",                2, "probe"),
    ("activation_endpoint",        2, "probe"),
    ("cognitive_potential",         2, "probe"),
    ("cognitive_frame",            2, "probe"),
    ("trigger_score",              2, "probe"),
    ("trigger_weight",             2, "probe"),
    ("adapt_weights",              2, "probe"),
    ("signal_mapping",             2, "probe"),
    ("signal_routing",             2, "probe"),
    ("signal_to_lambda",           2, "probe"),
    ("class_to_signal",            2, "probe"),
    ("error_class_to_lambda",      2, "probe"),
    ("signal_parity",              2, "probe"),
    ("signal_freshness",           2, "probe"),
    ("post_output_uncertainty",    2, "probe"),
    ("post_output_reliability",    2, "probe"),
    ("output_reliability_gate",    2, "verify"),
    ("ucc_rerun",                  2, "probe"),
    ("rerun_meta_loop",            2, "probe"),
    ("auto_critic",                2, "probe"),
    ("self_report",                2, "probe"),
    ("honesty",                    2, "probe"),

    # ── Category 3: Component Integration ─────────────────────────────────
    ("integration",                3, "verify"),
    ("pipeline_dep",               3, "verify"),
    ("pipeline_trace",             3, "verify"),
    ("cross_pipeline",             3, "verify"),
    ("bridge",                     3, "verify"),
    ("bidirectional",              3, "verify"),
    ("provenance",                 3, "verify"),
    ("causal_trace",               3, "verify"),
    ("causal_chain",               3, "verify"),
    ("causal_decision",            3, "verify"),
    ("feedback_bus",               3, "verify"),
    ("feedback_loop",              3, "verify"),
    ("feedback_signal",            3, "verify"),
    ("wiring",                     3, "init"),
    ("cross_module",               3, "verify"),
    ("subsystem_health",           3, "verify"),
    ("health_gate",                3, "verify"),
    ("cycle_consistency",          3, "verify"),
    ("config_round_trip",          3, "init"),
    ("to_core_config",             3, "init"),
    ("training_readiness",         3, "init"),
    ("sync_from_training",         3, "verify"),
    ("export_provenance",          3, "verify"),
    ("checkpoint",                 3, "init"),
    ("config",                     3, "init"),
    ("version",                    3, "init"),
    ("set_seed",                   3, "init"),
    ("dashboard",                  3, "verify"),
    ("server",                     3, "verify"),
    ("architecture",               3, "verify"),
    ("dag_sink",                   3, "verify"),

    # ── Category 4: Learning & Adaptation ─────────────────────────────────
    ("meta_loop",                  4, "probe"),
    ("meta_learner",               4, "probe"),
    ("certified_meta",             4, "probe"),
    ("error_evolution",            4, "probe"),
    ("error_recovery",             4, "probe"),
    ("error_class",                4, "probe"),
    ("recovery_pressure",          4, "probe"),
    ("recurring_root",             4, "probe"),
    ("semantic_error",             4, "probe"),
    ("continual_learning",         4, "probe"),
    ("train",                      4, "probe"),
    ("fit",                        4, "probe"),
    ("epoch",                      4, "probe"),
    ("curriculum",                 4, "probe"),
    ("active_learning",            4, "probe"),
    ("task2vec",                   4, "probe"),
    ("icm",                        4, "probe"),
    ("curiosity",                  4, "probe"),
    ("exploration",                4, "probe"),
    ("slot_binding",               4, "probe"),
    ("factor_extract",             4, "probe"),
    ("sparse_factor",              4, "probe"),
    ("world_model",                4, "probe"),
    ("hierarchical_world",         4, "probe"),
    ("causal_world",               4, "probe"),
    ("temporal_knowledge",         4, "probe"),
    ("causal_model",               4, "probe"),
    ("neural_causal",              4, "probe"),
    ("notears",                    4, "probe"),
    ("dag_constraint",             4, "probe"),
    ("causal_programmatic",        4, "probe"),
    ("dag_consensus",              4, "probe"),
    ("unified_sim",                4, "probe"),
    ("hybrid_reason",              4, "probe"),
    ("ns_bridge",                  4, "probe"),
    ("hierarchical_vae",           4, "probe"),
    ("causal_context",             4, "probe"),
    ("rssm",                       4, "probe"),
    ("loss",                       4, "probe"),
    ("optimizer",                  4, "probe"),
    ("warmup",                     4, "probe"),
    ("gradscaler",                 4, "probe"),
    ("compute_loss",               4, "probe"),
    ("batch_metric",               4, "probe"),
    ("hessian",                    4, "probe"),
    ("blend_weakest",              4, "probe"),
    ("lambda_meta",                4, "probe"),
    ("vibe_thinker",               4, "probe"),
    ("reasoning_kernel",           4, "probe"),
    ("prompt_adapter",             4, "probe"),
    ("response_parser",            4, "probe"),
    ("continuous_learner",         4, "probe"),

    # ── Category 5: Security & Reliability ────────────────────────────────
    ("safety",                     5, "verify"),
    ("quarantine",                 5, "verify"),
    ("circuit_breaker",            5, "verify"),
    ("tensor_guard",               5, "verify"),
    ("nan_policy",                 5, "verify"),
    ("sanitiz",                    5, "verify"),
    ("deception_suppressor",       5, "verify"),
    ("deception",                  5, "verify"),
    ("hash",                       5, "verify"),
    ("tensor_hash",                5, "verify"),
    ("audit",                      5, "verify"),
    ("decision_audit",             5, "verify"),
    ("memory_trust",               5, "verify"),
    ("memory_validation",          5, "verify"),
    ("memory_cross",               5, "verify"),
    ("counterfactual",             5, "verify"),
    ("reliability",                5, "verify"),
    ("consistency_gate",           5, "verify"),
    ("cross_validation",           5, "verify"),
    ("complexity_estimator",       5, "verify"),
    ("topology",                   5, "verify"),
    ("diversity",                  5, "verify"),

    # ── Category 6: Performance & Scalability ─────────────────────────────
    ("benchmark",                  6, "probe"),
    ("latency",                    6, "probe"),
    ("throughput",                 6, "probe"),
    ("memory_usage",               6, "probe"),
    ("speed",                      6, "probe"),
    ("scalab",                     6, "probe"),
    ("parallel_scan",              6, "probe"),
    ("chunked",                    6, "probe"),
    ("inference_cache",            6, "probe"),
    ("batch_generation",           6, "probe"),
    ("generate",                   6, "probe"),
    ("poly_feature",               6, "probe"),
    ("encoder",                    6, "probe"),
    ("ssm",                        6, "probe"),
    ("ssmv2",                      6, "probe"),
    ("mamba",                      6, "probe"),
    ("linear_attention",           6, "probe"),
    ("decoder",                    6, "probe"),
    ("multimodal",                 6, "probe"),
    ("grounded_multimodal",        6, "probe"),
    ("document_aware",             6, "probe"),

    # ── Category 7: Special Scenarios & Stress Tests ──────────────────────
    ("stress",                     7, "probe"),
    ("chaos",                      7, "probe"),
    ("edge_case",                  7, "probe"),
    ("concurrent",                 7, "probe"),
    ("thread",                     7, "probe"),
    ("degrad",                     7, "probe"),
    ("fallback",                   7, "probe"),
    ("graceful",                   7, "probe"),
    ("reactivation",               7, "probe"),
    ("healing",                    7, "probe"),
    ("repair",                     7, "probe"),
    ("recovery",                   7, "probe"),
    ("oscillat",                   7, "probe"),
    ("staleness",                  7, "probe"),
    ("partial",                    7, "probe"),
    ("overflow",                   7, "probe"),
    ("nan",                        7, "verify"),
    ("inf_guard",                  7, "verify"),
    ("zero_div",                   7, "verify"),
    ("division_by_zero",           7, "verify"),
    ("save_load",                  7, "probe"),
    ("mcts",                       7, "probe"),

    # ── Category 4 — Memory subsystems ────────────────────────────────────
    ("memory",                     4, "probe"),
    ("temporal_memory",            4, "probe"),
    ("neurogenic",                 4, "probe"),
    ("consolidat",                 4, "probe"),
    ("memory_routing",             4, "probe"),
    ("routing_policy",             4, "probe"),
]

# Pre-sort by keyword length descending so longest match wins
_CATEGORY_KEYWORDS.sort(key=lambda t: len(t[0]), reverse=True)


def _classify_test(test_name: str) -> Tuple[int, str]:
    """Classify a test into one of 7 cognitive categories.

    Returns (category_number, phase_hint).
    Uses longest-keyword-match strategy for deterministic classification.
    """
    name_lower = test_name.lower().replace("test_", "", 1)
    for keyword, cat, phase in _CATEGORY_KEYWORDS:
        if keyword in name_lower:
            return cat, phase
    # Default: Component Integration (category 3) for uncategorised tests
    return 3, "verify"


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE NODE MAPPING (mirrors test_fixes.py _ARCH_NODES)
# ═══════════════════════════════════════════════════════════════════════════════

_ARCH_NODES = [
    'input', 'encoder', 'continual_learning', 'vq', 'encoder_reasoning_norm',
    'meta_loop', 'certified_meta_loop', 'convergence_arbiter', 'slot_binding',
    'factor_extraction', 'topology_analysis', 'diversity_analysis',
    'complexity_estimator', 'consistency_gate', 'cross_validation',
    'cross_validation_correction',
    'self_report', 'deception_suppressor', 'safety', 'cognitive_executive',
    'deeper_meta_loop',
    'world_model', 'hierarchical_world_model', 'causal_world_model',
    'memory', 'temporal_memory', 'neurogenic_memory', 'consolidating_memory',
    'memory_trust', 'memory_validation', 'memory_cross_validation',
    'mcts_planning', 'active_learning', 'icm_curiosity', 'causal_model',
    'notears_causal', 'causal_programmatic', 'causal_dag_consensus',
    'unified_simulator', 'hybrid_reasoning', 'ns_bridge',
    'temporal_knowledge_graph', 'hierarchical_vae', 'causal_context',
    'rssm', 'multimodal', 'grounded_multimodal', 'integration',
    'auto_critic', 'output_reliability', 'metacognitive_trigger',
    'error_evolution', 'unified_cognitive_cycle', 'decoder',
    'memory_routing', 'counterfactual_verification',
    'subsystem_health_gate',
    'feedback_bus', 'cycle_consistency', 'output_reliability_gate',
    'ucc_rerun_meta_loop', 'post_output_uncertainty_gate',
    'vibe_thinker',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  COGNITIVE TEST METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveTestMetrics:
    """Collects comprehensive metrics for a single test execution.

    Produces a structured record conforming to the AEON cognitive analysis
    schema with five metric groups: performance, quality, coverage, error,
    and execution context.
    """

    __slots__ = (
        "_test_id", "_test_name", "_category", "_phase",
        "_t_start", "_t_end",
        "_mem_start", "_mem_peak", "_mem_end",
        "_cpu_start", "_cpu_end",
        "_status", "_error_class", "_error_severity",
        "_recovery_success", "_fallback_triggered",
        "_subsystems_covered", "_axioms_verified",
        "_stdout", "_stderr", "_traceback_text",
        "_arch_node", "_doc",
    )

    def __init__(self, test_name: str, category: int, phase: str) -> None:
        self._test_id = f"aeon_{uuid.uuid4().hex[:12]}"
        self._test_name = test_name
        self._category = category
        self._phase = phase
        self._status = "pass"
        self._error_class = ""
        self._error_severity = "info"
        self._recovery_success = True
        self._fallback_triggered = False
        self._subsystems_covered: List[str] = []
        self._axioms_verified: List[str] = []
        self._stdout = ""
        self._stderr = ""
        self._traceback_text = ""
        self._arch_node = ""
        self._doc = ""

        # Timing
        self._t_start = 0.0
        self._t_end = 0.0

        # Memory (bytes)
        self._mem_start = 0
        self._mem_peak = 0
        self._mem_end = 0

        # CPU times (user + system, seconds)
        self._cpu_start = 0.0
        self._cpu_end = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Record initial resource snapshot."""
        gc.collect()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        self._mem_start = tracemalloc.get_traced_memory()[0]
        self._cpu_start = self._cpu_time()
        self._t_start = time.perf_counter()

    def stop(self, status: str = "pass") -> None:
        """Record final resource snapshot and status."""
        self._t_end = time.perf_counter()
        self._cpu_end = self._cpu_time()
        if tracemalloc.is_tracing():
            _, self._mem_peak = tracemalloc.get_traced_memory()
            self._mem_end = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
        self._status = status

    # ── Setters ───────────────────────────────────────────────────────────

    def set_error(self, error_class: str, severity: str,
                  recovery: bool = False, fallback: bool = False) -> None:
        self._error_class = error_class
        self._error_severity = severity
        self._recovery_success = recovery
        self._fallback_triggered = fallback

    def set_subsystems(self, nodes: List[str]) -> None:
        self._subsystems_covered = nodes

    def set_axioms(self, axioms: List[str]) -> None:
        self._axioms_verified = axioms

    def set_output(self, stdout: str, stderr: str, tb: str = "") -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._traceback_text = tb

    def set_arch_node(self, node: str) -> None:
        self._arch_node = node

    def set_doc(self, doc: str) -> None:
        self._doc = doc

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _cpu_time() -> float:
        """Return cumulative CPU time (user + system) for current process."""
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    @staticmethod
    def _detect_environment() -> str:
        """Detect execution environment (cpu / gpu / tpu)."""
        if _HAS_TORCH and _torch.cuda.is_available():
            return "gpu"
        return "cpu"

    @staticmethod
    def _model_config_hash() -> str:
        """Deterministic hash of AEONConfig defaults for provenance."""
        try:
            from aeon_core import AEONConfig
            cfg = AEONConfig(device_str="cpu")
            sig = f"{cfg.d_model}_{cfg.n_layers}_{cfg.vocab_size}"
            return hashlib.sha256(sig.encode()).hexdigest()[:16]
        except Exception:
            return "unavailable"

    # ── Quality metrics (derived from stdout analysis) ────────────────────

    def _extract_quality_metrics(self) -> Dict[str, float]:
        """Parse test stdout for cognitive quality indicators.

        Tests that print metrics like 'unity_score=0.85' or
        'coherence_deficit=0.12' will have those values captured.
        """
        quality: Dict[str, float] = {
            "cognitive_unity_score": -1.0,
            "coherence_deficit": -1.0,
            "uncertainty_level": -1.0,
            "convergence_quality": -1.0,
            "output_reliability": -1.0,
        }
        # Scan stdout for known metric patterns
        for line in self._stdout.splitlines():
            ll = line.lower()
            for key in quality:
                # Match patterns like 'unity_score=0.85' or 'unity_score: 0.85'
                for sep in ("=", ": ", ":", " "):
                    tag = key.replace("cognitive_", "") + sep
                    if tag in ll:
                        try:
                            val_str = ll.split(tag, 1)[1].strip().split()[0]
                            val_str = val_str.rstrip(",;)")
                            quality[key] = round(float(val_str), 6)
                        except (ValueError, IndexError):
                            pass
        return quality

    # ── Build structured record ───────────────────────────────────────────

    def to_record(self) -> Dict[str, Any]:
        """Build the full structured test record per AEON cognitive schema."""
        elapsed_ms = round((self._t_end - self._t_start) * 1000, 3)
        cpu_delta = max(self._cpu_end - self._cpu_start, 1e-9)
        wall_delta = max(self._t_end - self._t_start, 1e-9)
        cpu_pct = round(min(cpu_delta / wall_delta * 100, 100.0), 2)

        mem_peak_mb = round(self._mem_peak / (1024 * 1024), 4)
        mem_avg_mb = round(
            (self._mem_start + self._mem_end) / 2 / (1024 * 1024), 4
        )
        cache_hit = -1.0  # Placeholder; populated when cache subsystem is probed

        quality = self._extract_quality_metrics()

        # Provenance completeness: fraction of subsystems with provenance data
        prov_complete = (
            round(len(self._subsystems_covered) / max(len(_ARCH_NODES), 1), 4)
            if self._subsystems_covered else 0.0
        )

        # Audit entries count (from stdout lines containing 'audit' or 'provenance')
        audit_count = sum(
            1 for l in self._stdout.lower().splitlines()
            if "audit" in l or "provenance" in l
        )

        return {
            "test_id": self._test_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": str(self._category),
            "category_name": CATEGORIES.get(self._category, "Unknown"),
            "test_name": self._test_name,
            "test_doc": self._doc,
            "arch_node": self._arch_node,
            "metrics": {
                "performance": {
                    "execution_time_ms": elapsed_ms,
                    "memory_usage_mb": {
                        "peak": mem_peak_mb,
                        "average": mem_avg_mb,
                    },
                    "cpu_utilization_pct": cpu_pct,
                    "cache_hit_rate": cache_hit,
                },
                "quality": quality,
                "coverage": {
                    "subsystems_covered": self._subsystems_covered,
                    "axioms_verified": self._axioms_verified,
                    "provenance_completeness": prov_complete,
                    "audit_log_entries": audit_count,
                },
                "error": {
                    "error_class": self._error_class,
                    "error_severity": self._error_severity,
                    "recovery_success": self._recovery_success,
                    "fallback_triggered": self._fallback_triggered,
                },
            },
            "result": self._status,
            "cognitive_state": {
                "phase": self._phase,
                "category": self._category,
                "environment": self._detect_environment(),
                "model_config_hash": self._model_config_hash(),
            },
            "provenance": {
                "subsystems": self._subsystems_covered,
                "axioms": self._axioms_verified,
                "completeness": prov_complete,
            },
            "audit_entries": self._stdout.splitlines()[-5:]
            if self._stdout else [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  COGNITIVE TEST REPORT — session-level aggregation
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveTestReport:
    """Session-level aggregator for cognitive test records.

    Provides category-level summaries, architecture coverage, and
    ErrorEvolutionTracker-compatible output.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.category_stats: Dict[int, Dict[str, Any]] = {
            c: {"total": 0, "passed": 0, "failed": 0, "skipped": 0,
                "errors": 0, "total_time_ms": 0.0, "tests": []}
            for c in CATEGORIES
        }
        self.node_coverage: Dict[str, int] = defaultdict(int)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def add(self, record: Dict[str, Any]) -> None:
        """Add a test record to the report."""
        self.records.append(record)
        cat = int(record.get("category", 3))
        cs = self.category_stats.get(cat)
        if cs is None:
            return
        cs["total"] += 1
        result = record.get("result", "fail")
        if result == "pass":
            cs["passed"] += 1
        elif result == "skip":
            cs["skipped"] += 1
        elif result == "fail":
            cs["failed"] += 1
        else:
            cs["errors"] += 1
        elapsed = record.get("metrics", {}).get(
            "performance", {}).get("execution_time_ms", 0.0)
        cs["total_time_ms"] += elapsed
        cs["tests"].append(record["test_name"])

        # Track architecture node coverage
        node = record.get("arch_node", "")
        if node:
            self.node_coverage[node] += 1

    def summary(self) -> Dict[str, Any]:
        """Build session-level summary."""
        total = len(self.records)
        passed = sum(1 for r in self.records if r["result"] == "pass")
        failed = sum(1 for r in self.records if r["result"] == "fail")
        skipped = sum(1 for r in self.records if r["result"] == "skip")
        errors = total - passed - failed - skipped
        total_time = sum(
            r.get("metrics", {}).get("performance", {}).get(
                "execution_time_ms", 0.0)
            for r in self.records
        )

        covered = len([n for n in _ARCH_NODES if n in self.node_coverage])

        # Performance percentiles
        times = sorted(
            r.get("metrics", {}).get("performance", {}).get(
                "execution_time_ms", 0.0)
            for r in self.records
        )
        p50 = times[len(times) // 2] if times else 0.0
        p95 = times[int(len(times) * 0.95)] if times else 0.0
        p99 = times[int(len(times) * 0.99)] if times else 0.0

        # Category coverage
        cat_summary = {}
        for c, info in self.category_stats.items():
            cat_summary[str(c)] = {
                "name": CATEGORIES[c],
                "total": info["total"],
                "passed": info["passed"],
                "failed": info["failed"],
                "skipped": info["skipped"],
                "errors": info["errors"],
                "pass_rate": round(
                    info["passed"] / max(info["total"], 1) * 100, 2),
                "total_time_ms": round(info["total_time_ms"], 2),
            }

        return {
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "python": platform.python_version(),
                "os": platform.system(),
                "arch": platform.machine(),
                "environment": CognitiveTestMetrics._detect_environment(),
            },
            "totals": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "pass_rate": round(passed / max(total, 1) * 100, 2),
                "total_time_ms": round(total_time, 2),
            },
            "performance_percentiles": {
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
            },
            "architecture_coverage": {
                "covered_nodes": covered,
                "total_nodes": len(_ARCH_NODES),
                "coverage_pct": round(
                    covered / max(len(_ARCH_NODES), 1) * 100, 2),
                "node_distribution": dict(self.node_coverage),
            },
            "category_breakdown": cat_summary,
            "error_evolution_data": self._error_evolution_data(),
        }

    def _error_evolution_data(self) -> Dict[str, Any]:
        """Extract error-evolution-compatible data for continuous improvement.

        Maps failed tests to error classes and severities for integration
        with CausalErrorEvolutionTracker.
        """
        error_classes: Dict[str, int] = defaultdict(int)
        severity_dist: Dict[str, int] = defaultdict(int)
        recovery_stats = {"attempted": 0, "succeeded": 0}

        for r in self.records:
            err = r.get("metrics", {}).get("error", {})
            ec = err.get("error_class", "")
            if ec:
                error_classes[ec] += 1
                severity_dist[err.get("error_severity", "info")] += 1
                if err.get("fallback_triggered"):
                    recovery_stats["attempted"] += 1
                    if err.get("recovery_success"):
                        recovery_stats["succeeded"] += 1

        return {
            "error_classes": dict(error_classes),
            "severity_distribution": dict(severity_dist),
            "recovery_stats": recovery_stats,
            "total_errors_recorded": sum(error_classes.values()),
        }

    def export(self) -> Dict[str, Any]:
        """Full export with all records and summary."""
        return {
            "summary": self.summary(),
            "records": self.records,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SUBSYSTEM DETECTION — automatic detection of tested subsystems
# ═══════════════════════════════════════════════════════════════════════════════

# Map test keywords to architecture nodes for subsystem coverage tracking
_KEYWORD_NODE_MAP = {
    'encoder': 'encoder', 'ssm': 'encoder', 'ssmv2': 'encoder',
    'mamba': 'encoder', 'linear_attention': 'encoder',
    'continual_learning': 'continual_learning',
    'vq': 'vq', 'codebook': 'vq', 'gumbel': 'vq', 'vector_quant': 'vq',
    'meta_loop': 'meta_loop', 'meta_learner': 'meta_loop',
    'certified': 'certified_meta_loop',
    'convergence_arbiter': 'convergence_arbiter',
    'slot_binding': 'slot_binding', 'factor_extract': 'factor_extraction',
    'topology': 'topology_analysis', 'diversity': 'diversity_analysis',
    'complexity_estimator': 'complexity_estimator',
    'consistency_gate': 'consistency_gate',
    'cross_validation': 'cross_validation',
    'self_report': 'self_report', 'honesty': 'self_report',
    'deception': 'deception_suppressor',
    'safety': 'safety', 'quarantine': 'safety', 'tensor_guard': 'safety',
    'cognitive_executive': 'cognitive_executive',
    'deeper_meta': 'deeper_meta_loop',
    'world_model': 'world_model',
    'memory': 'memory', 'temporal_memory': 'temporal_memory',
    'neurogenic': 'neurogenic_memory', 'consolidat': 'consolidating_memory',
    'mcts': 'mcts_planning', 'active_learning': 'active_learning',
    'icm': 'icm_curiosity', 'curiosity': 'icm_curiosity',
    'causal_model': 'causal_model', 'notears': 'notears_causal',
    'causal_programmatic': 'causal_programmatic',
    'dag_consensus': 'causal_dag_consensus',
    'unified_sim': 'unified_simulator', 'hybrid_reason': 'hybrid_reasoning',
    'ns_bridge': 'ns_bridge', 'temporal_knowledge': 'temporal_knowledge_graph',
    'hierarchical_vae': 'hierarchical_vae', 'causal_context': 'causal_context',
    'rssm': 'rssm', 'multimodal': 'multimodal',
    'auto_critic': 'auto_critic', 'output_reliability': 'output_reliability',
    'metacognitive_trigger': 'metacognitive_trigger',
    'error_evolution': 'error_evolution',
    'unified_cognitive': 'unified_cognitive_cycle', 'ucc': 'unified_cognitive_cycle',
    'decoder': 'decoder', 'memory_routing': 'memory_routing',
    'counterfactual': 'counterfactual_verification',
    'subsystem_health': 'subsystem_health_gate',
    'feedback_bus': 'feedback_bus', 'cycle_consistency': 'cycle_consistency',
    'output_reliability_gate': 'output_reliability_gate',
    'ucc_rerun': 'ucc_rerun_meta_loop',
    'post_output_uncertainty': 'post_output_uncertainty_gate',
    'vibe_thinker': 'vibe_thinker', 'reasoning_kernel': 'vibe_thinker',
    'coherence': 'unified_cognitive_cycle',
    'provenance': 'integration', 'causal_trace': 'integration',
    'bridge': 'integration', 'pipeline': 'integration',
}

# Axiom detection keywords
_AXIOM_KEYWORDS = {
    "mutual_verification": ["mutual_verification", "verify_and_reinforce",
                            "cross_module_check", "cross_module"],
    "metacognitive_responsiveness": ["metacognitive_trigger", "metacognitive_adaptation",
                                     "trigger_score", "adapt_weights",
                                     "deeper_meta", "cognitive_executive"],
    "root_cause_traceability": ["causal_trace", "provenance", "root_cause",
                                "causal_chain", "causal_decision", "dag_sink"],
    "convergence_quality": ["convergence", "km_convergence", "anderson",
                            "lipschitz", "contraction", "arbiter"],
}


def _detect_subsystems(test_name: str) -> List[str]:
    """Detect which architecture subsystems a test covers."""
    name_lower = test_name.lower().replace("test_", "", 1)
    nodes = set()
    for keyword, node in _KEYWORD_NODE_MAP.items():
        if keyword in name_lower:
            nodes.add(node)
    return sorted(nodes) if nodes else ["integration"]


def _detect_axioms(test_name: str) -> List[str]:
    """Detect which AGI axioms a test verifies."""
    name_lower = test_name.lower().replace("test_", "", 1)
    axioms = []
    for axiom_name, keywords in _AXIOM_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            axioms.append(axiom_name)
    return axioms


def _map_test_to_node(test_name: str) -> str:
    """Map a test function name to its primary architecture node."""
    name_lower = test_name.lower().replace("test_", "", 1)
    best_match = ""
    best_node = "integration"
    for keyword, node in _KEYWORD_NODE_MAP.items():
        if keyword in name_lower and len(keyword) > len(best_match):
            best_match = keyword
            best_node = node
    return best_node


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTEST HOOKS — transparent integration with pytest execution
# ═══════════════════════════════════════════════════════════════════════════════

# Session-level report
_report = CognitiveTestReport()

# Per-item metrics (stash key)
_METRICS_KEY = pytest.StashKey["CognitiveTestMetrics"]()


def pytest_configure(config: pytest.Config) -> None:
    """Register cognitive test markers and configure the report."""
    for cat_num, cat_name in CATEGORIES.items():
        config.addinivalue_line(
            "markers",
            f"category_{cat_num}: {cat_name}",
        )
    _report.start_time = time.perf_counter()


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    """Automatically add category markers to all collected test items."""
    for item in items:
        cat, phase = _classify_test(item.name)
        # Add the category marker dynamically
        item.add_marker(getattr(pytest.mark, f"category_{cat}"))
        # Store classification in item stash for later retrieval
        item.stash[_METRICS_KEY] = CognitiveTestMetrics(
            test_name=item.name,
            category=cat,
            phase=phase,
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """Wrap test execution to collect per-test cognitive metrics."""
    metrics: Optional[CognitiveTestMetrics] = item.stash.get(
        _METRICS_KEY, None)

    if metrics is not None:
        # Set metadata
        doc = (item.function.__doc__ or "").strip().split("\n")[0][:120]
        metrics.set_doc(doc)
        node = _map_test_to_node(item.name)
        metrics.set_arch_node(node)
        metrics.set_subsystems(_detect_subsystems(item.name))
        metrics.set_axioms(_detect_axioms(item.name))
        metrics.start()

    outcome = yield

    if metrics is not None:
        exc = outcome.excinfo if outcome is not None else None
        if exc is not None:
            _, exc_value, exc_tb = exc
            error_type = type(exc_value).__name__
            if isinstance(exc_value, AssertionError):
                metrics.stop("fail")
                metrics.set_error(
                    error_class=error_type,
                    severity="error",
                    recovery=False,
                    fallback=False,
                )
            elif isinstance(exc_value, (ImportError, ModuleNotFoundError)):
                metrics.stop("skip")
                metrics.set_error(
                    error_class=error_type,
                    severity="warning",
                    recovery=False,
                    fallback=True,
                )
            else:
                metrics.stop("fail")
                metrics.set_error(
                    error_class=error_type,
                    severity="critical",
                    recovery=False,
                    fallback=False,
                )
        else:
            metrics.stop("pass")


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """After each test phase, record the cognitive metrics."""
    if call.when != "call":
        return
    metrics: Optional[CognitiveTestMetrics] = item.stash.get(
        _METRICS_KEY, None)
    if metrics is None:
        return

    # Capture any printed output from the capsys-free tests
    # (tests print to real stdout, which we can't capture here —
    #  the TestDataCollector in test_fixes.py already handles this)
    record = metrics.to_record()
    _report.add(record)


def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int
) -> None:
    """Write the cognitive test report after session completion."""
    _report.end_time = time.perf_counter()

    report_data = _report.export()
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cognitive_test_report.json",
    )

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False,
                      default=str)
    except OSError:
        pass  # Non-fatal: don't block test suite on report write failure


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    """Print cognitive architecture summary to terminal."""
    summary = _report.summary()
    totals = summary.get("totals", {})
    cats = summary.get("category_breakdown", {})
    arch = summary.get("architecture_coverage", {})
    perfs = summary.get("performance_percentiles", {})
    evo = summary.get("error_evolution_data", {})

    lines = [
        "",
        "═" * 72,
        "  AEON-Delta Cognitive Test Architecture — Session Report",
        "═" * 72,
        "",
        f"  Total: {totals.get('total', 0)}  "
        f"Passed: {totals.get('passed', 0)}  "
        f"Failed: {totals.get('failed', 0)}  "
        f"Skipped: {totals.get('skipped', 0)}  "
        f"Errors: {totals.get('errors', 0)}",
        f"  Pass Rate: {totals.get('pass_rate', 0)}%  "
        f"Total Time: {totals.get('total_time_ms', 0):.0f}ms",
        "",
        "  Performance Percentiles:",
        f"    p50: {perfs.get('p50_ms', 0):.1f}ms  "
        f"p95: {perfs.get('p95_ms', 0):.1f}ms  "
        f"p99: {perfs.get('p99_ms', 0):.1f}ms",
        "",
        f"  Architecture Coverage: {arch.get('covered_nodes', 0)}"
        f"/{arch.get('total_nodes', 0)} nodes "
        f"({arch.get('coverage_pct', 0)}%)",
        "",
        "  Category Breakdown:",
    ]

    for cnum in sorted(cats.keys(), key=int):
        c = cats[cnum]
        lines.append(
            f"    [{cnum}] {c['name']}: "
            f"{c['passed']}/{c['total']} passed "
            f"({c['pass_rate']}%) "
            f"[{c['total_time_ms']:.0f}ms]"
        )

    if evo.get("total_errors_recorded", 0) > 0:
        lines.extend([
            "",
            "  Error Evolution Data:",
            f"    Total errors recorded: {evo['total_errors_recorded']}",
            f"    Error classes: {evo.get('error_classes', {})}",
            f"    Severity distribution: {evo.get('severity_distribution', {})}",
            f"    Recovery: {evo.get('recovery_stats', {})}",
        ])

    lines.extend(["", "═" * 72, ""])

    for line in lines:
        terminalreporter.write_line(line)
