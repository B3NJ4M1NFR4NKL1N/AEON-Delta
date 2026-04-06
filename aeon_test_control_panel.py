"""
AEON-Delta RMT v3.2 — Test Control Panel
═════════════════════════════════════════

Academic-grade test execution management system for the AEON-Delta
cognitive architecture.  Provides structured, selective, and auditable
control over the unified test corpus (7 443 tests across 66 sections).

Architecture
────────────
The control panel operates on three conceptual layers:

  Layer 1 — Registry         Declarative catalogue of every test suite,
                              its cognitive category, source lineage, and
                              dependency graph.

  Layer 2 — Execution Engine  Translates user intent (run category X,
                              exclude series Y, stress-only) into precise
                              pytest invocations with correct markers,
                              filters, and parallelism settings.

  Layer 3 — Analytics         Aggregates per-test metrics from conftest.py,
                              computes suite-level health indicators, and
                              exports structured reports for the AEON
                              Dashboard.

Usage
─────
  # Run everything
  python aeon_test_control_panel.py --all

  # Run a single series
  python aeon_test_control_panel.py --series c

  # Run by cognitive category (1-7)
  python aeon_test_control_panel.py --category 3

  # Run integration tests only
  python aeon_test_control_panel.py --integration

  # Run wizard tests only
  python aeon_test_control_panel.py --wizard

  # Dry-run (show what would execute)
  python aeon_test_control_panel.py --series v5 --dry-run

  # Generate report from last run
  python aeon_test_control_panel.py --report

  # List all suites
  python aeon_test_control_panel.py --list

  # Run with verbose output
  python aeon_test_control_panel.py --category 1 -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — COGNITIVE CATEGORY TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveCategory(IntEnum):
    """Seven-category cognitive test taxonomy (mirrors conftest.py)."""
    AGI_AXIOMS          = 1   # Mutual verification, metacognitive responsiveness
    COGNITIVE_ACTIVATION = 2  # MCT, UCC, cognitive cycle, signal routing
    COMPONENT_INTEGRATION = 3 # Cross-module bridges, provenance, feedback bus
    LEARNING_ADAPTATION  = 4  # Meta-loop, error evolution, weight adaptation
    SECURITY_RELIABILITY = 5  # Safety guards, quarantine, NaN policy
    PERFORMANCE_SCALE    = 6  # Benchmarks, latency, throughput, memory
    STRESS_SCENARIOS     = 7  # Degraded mode, chaos, edge cases, concurrency

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]

    @classmethod
    def from_name(cls, name: str) -> "CognitiveCategory":
        """Resolve a category from name fragment (case-insensitive)."""
        key = name.upper().replace(" ", "_").replace("-", "_")
        for member in cls:
            if key in member.name:
                return member
        raise ValueError(f"Unknown category: {name!r}")


_CATEGORY_LABELS: Dict[CognitiveCategory, str] = {
    CognitiveCategory.AGI_AXIOMS:           "AGI Axioms Verification",
    CognitiveCategory.COGNITIVE_ACTIVATION: "Cognitive Activation System",
    CognitiveCategory.COMPONENT_INTEGRATION: "Component Integration",
    CognitiveCategory.LEARNING_ADAPTATION:  "Learning & Adaptation",
    CognitiveCategory.SECURITY_RELIABILITY: "Security & Reliability",
    CognitiveCategory.PERFORMANCE_SCALE:    "Performance & Scalability",
    CognitiveCategory.STRESS_SCENARIOS:     "Special Scenarios & Stress Tests",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — TEST SUITE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TestSuite:
    """Declarative descriptor for a logical test suite.

    Attributes
    ----------
    id : str
        Short unique identifier (e.g. "c_patches", "v5_series").
    name : str
        Human-readable label.
    section : int
        Section number inside test_aeon_unified.py (1-33).
    source : str
        Original filename before consolidation.
    test_count : int
        Number of ``def test_*`` functions in the suite.
    categories : FrozenSet[CognitiveCategory]
        Primary cognitive categories this suite exercises.
    series : str
        Alphabetic series code (e.g. "c", "v5", "core").
    tags : FrozenSet[str]
        Free-form tags for advanced filtering.
    """
    id: str
    name: str
    section: int
    source: str
    test_count: int
    categories: FrozenSet[CognitiveCategory]
    series: str
    tags: FrozenSet[str] = field(default_factory=frozenset)


def _fs(*cats: CognitiveCategory) -> FrozenSet[CognitiveCategory]:
    return frozenset(cats)


def _ft(*tags: str) -> FrozenSet[str]:
    return frozenset(tags)


# ── Complete registry of all 33 suites ──────────────────────────────────────

SUITE_REGISTRY: Tuple[TestSuite, ...] = (
    TestSuite(
        id="core",
        name="Core Component Tests",
        section=1,
        source="test_aeon_unified.py",
        test_count=4780,
        categories=_fs(
            CognitiveCategory.AGI_AXIOMS,
            CognitiveCategory.COGNITIVE_ACTIVATION,
            CognitiveCategory.COMPONENT_INTEGRATION,
            CognitiveCategory.LEARNING_ADAPTATION,
            CognitiveCategory.SECURITY_RELIABILITY,
            CognitiveCategory.PERFORMANCE_SCALE,
            CognitiveCategory.STRESS_SCENARIOS,
        ),
        series="core",
        tags=_ft("aeon_core", "ae_train", "comprehensive"),
    ),
    TestSuite(
        id="c_patches",
        name="C-Series Patches — Stall Severity & UCC",
        section=2,
        source="test_aeon_unified.py",
        test_count=24,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION),
        series="c",
        tags=_ft("stall_severity", "ucc"),
    ),
    TestSuite(
        id="cognitive_patches",
        name="Cognitive Patches — Spectral Projection & Cache",
        section=3,
        source="test_aeon_unified.py",
        test_count=25,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="cognitive",
        tags=_ft("spectral", "cache", "ema"),
    ),
    TestSuite(
        id="cognitive_activation",
        name="Cognitive Activation — Signal Routing",
        section=4,
        source="test_aeon_unified.py",
        test_count=37,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION),
        series="activation",
        tags=_ft("signal_routing", "coherence_trend", "feedback_bus"),
    ),
    TestSuite(
        id="d_patches",
        name="D-Series — Convergence & Oscillation",
        section=5,
        source="test_aeon_unified.py",
        test_count=40,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="d",
        tags=_ft("convergence", "oscillation", "anomaly"),
    ),
    TestSuite(
        id="e_patches",
        name="E-Series — Prediction Error & Calibration",
        section=6,
        source="test_aeon_unified.py",
        test_count=54,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="e",
        tags=_ft("prediction_error", "calibration", "divergence"),
    ),
    TestSuite(
        id="f_patches",
        name="F-Series — DAG & Verdict Recording",
        section=7,
        source="test_aeon_unified.py",
        test_count=52,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.AGI_AXIOMS),
        series="f",
        tags=_ft("dag", "verdict", "provenance"),
    ),
    TestSuite(
        id="g_patches",
        name="G-Series — Oscillation & Root Cause",
        section=8,
        source="test_aeon_unified.py",
        test_count=20,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION),
        series="g",
        tags=_ft("oscillation", "root_cause"),
    ),
    TestSuite(
        id="h_patches",
        name="H-Series — VT & Provenance Bridges",
        section=9,
        source="test_aeon_unified.py",
        test_count=41,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION),
        series="h",
        tags=_ft("vibe_thinker", "provenance", "bridge"),
    ),
    TestSuite(
        id="i_patches",
        name="I-Series — Metacognitive Verdict",
        section=10,
        source="test_aeon_unified.py",
        test_count=15,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.AGI_AXIOMS),
        series="i",
        tags=_ft("metacognitive", "verdict"),
    ),
    TestSuite(
        id="j_patches",
        name="J-Series — Integrity & Unity Deficit",
        section=11,
        source="test_aeon_unified.py",
        test_count=38,
        categories=_fs(CognitiveCategory.AGI_AXIOMS, CognitiveCategory.COMPONENT_INTEGRATION),
        series="j",
        tags=_ft("integrity", "unity_deficit", "feedback_bus"),
    ),
    TestSuite(
        id="k_patches",
        name="K-Series — Participation & Replay",
        section=12,
        source="test_aeon_unified.py",
        test_count=39,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="k",
        tags=_ft("participation", "replay", "oscillation"),
    ),
    TestSuite(
        id="l_patches",
        name="L-Series — Provenance & Tracing",
        section=13,
        source="test_aeon_unified.py",
        test_count=35,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION),
        series="l",
        tags=_ft("provenance", "tracing", "root_cause"),
    ),
    TestSuite(
        id="m_patches",
        name="M-Series — Oscillation Severity Propagation",
        section=14,
        source="test_aeon_unified.py",
        test_count=38,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION),
        series="m",
        tags=_ft("oscillation", "severity", "propagation"),
    ),
    TestSuite(
        id="n_patches",
        name="N-Series — Error Evolution Recording",
        section=15,
        source="test_aeon_unified.py",
        test_count=66,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="n",
        tags=_ft("error_evolution", "recording", "recovery"),
    ),
    TestSuite(
        id="o_patches",
        name="O-Series — UCC & Verify Coherence",
        section=16,
        source="test_aeon_unified.py",
        test_count=26,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION),
        series="o",
        tags=_ft("ucc", "coherence"),
    ),
    TestSuite(
        id="p_patches",
        name="P-Series — Task Boundary & Entropy",
        section=17,
        source="test_aeon_unified.py",
        test_count=34,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="p",
        tags=_ft("task_boundary", "entropy"),
    ),
    TestSuite(
        id="q_patches",
        name="Q-Series — Severity Compounding",
        section=18,
        source="test_aeon_unified.py",
        test_count=45,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.AGI_AXIOMS),
        series="q",
        tags=_ft("severity", "compounding", "trend"),
    ),
    TestSuite(
        id="r_patches",
        name="R-Series — Checkpoint & Validation",
        section=19,
        source="test_aeon_unified.py",
        test_count=38,
        categories=_fs(CognitiveCategory.SECURITY_RELIABILITY, CognitiveCategory.COMPONENT_INTEGRATION),
        series="r",
        tags=_ft("checkpoint", "validation", "wiring"),
    ),
    TestSuite(
        id="s_patches",
        name="S-Series — VibeThinker Weight Management",
        section=20,
        source="test_aeon_unified.py",
        test_count=30,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION),
        series="s",
        tags=_ft("vibe_thinker", "weights", "safetensors"),
    ),
    TestSuite(
        id="v_patches",
        name="V-Series — Lipschitz & IBP Certification",
        section=21,
        source="test_aeon_unified.py",
        test_count=38,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.SECURITY_RELIABILITY),
        series="v",
        tags=_ft("lipschitz", "ibp", "crown", "hessian"),
    ),
    TestSuite(
        id="v5_patches",
        name="V5-Series — LipSDP & IQC Certification",
        section=22,
        source="test_aeon_unified.py",
        test_count=62,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.SECURITY_RELIABILITY),
        series="v5",
        tags=_ft("lipsdp", "iqc", "calibration", "lanczos"),
    ),
    TestSuite(
        id="w_patches",
        name="W-Series — Sandwich Linear & KM Iteration",
        section=23,
        source="test_aeon_unified.py",
        test_count=61,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION),
        series="w",
        tags=_ft("sandwich", "km_iteration", "tikhonov", "eclipse"),
    ),
    TestSuite(
        id="integration_patches",
        name="Integration Patches — Signal Flow & Stall",
        section=24,
        source="test_aeon_unified.py",
        test_count=24,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="integration",
        tags=_ft("signal_flow", "stall"),
    ),
    TestSuite(
        id="final_integration",
        name="Final Integration — Cross-Patch Coherence",
        section=25,
        source="test_aeon_unified.py",
        test_count=40,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.AGI_AXIOMS),
        series="final",
        tags=_ft("cross_patch", "coherence"),
    ),
    TestSuite(
        id="final_activation",
        name="Final Activation — Full Cycle",
        section=26,
        source="test_aeon_unified.py",
        test_count=39,
        categories=_fs(
            CognitiveCategory.COMPONENT_INTEGRATION,
            CognitiveCategory.COGNITIVE_ACTIVATION,
            CognitiveCategory.LEARNING_ADAPTATION,
        ),
        series="final_activation",
        tags=_ft("full_cycle", "codebook", "calibration", "mct"),
    ),
    TestSuite(
        id="g_series_integration",
        name="G-Series Integration — Bridge & MCT",
        section=27,
        source="test_aeon_unified.py",
        test_count=27,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="g_int",
        tags=_ft("bridge", "mct", "feedback_bus"),
    ),
    TestSuite(
        id="h_series_integration",
        name="H-Series Integration — MCT Signals & Auto-Wire",
        section=28,
        source="test_aeon_unified.py",
        test_count=31,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="h_int",
        tags=_ft("mct_signals", "auto_wire", "z_annotation"),
    ),
    TestSuite(
        id="i_series_integration",
        name="I-Series Integration — Failure Recording",
        section=29,
        source="test_aeon_unified.py",
        test_count=43,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.SECURITY_RELIABILITY),
        series="i_int",
        tags=_ft("failure_recording", "resilience", "traceability"),
    ),
    TestSuite(
        id="j_series_integration",
        name="J-Series Integration — Feedback Bus & Provenance",
        section=30,
        source="test_aeon_unified.py",
        test_count=42,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION),
        series="j_int",
        tags=_ft("feedback_bus", "provenance", "pseudo_labels"),
    ),
    TestSuite(
        id="architectural_fixes",
        name="Architectural Fixes — Quality & Weight Paths",
        section=31,
        source="test_aeon_unified.py",
        test_count=43,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="arch",
        tags=_ft("quality_head", "weight_paths", "ae_train"),
    ),
    TestSuite(
        id="wizard_integration",
        name="Wizard Integration — State & Metrics",
        section=32,
        source="test_aeon_unified.py",
        test_count=46,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="wizard",
        tags=_ft("wizard", "state", "metrics", "codebook"),
    ),
    TestSuite(
        id="self_play_wizard",
        name="Self-Play Wizard — Curriculum & Synthesis",
        section=33,
        source="test_aeon_unified.py",
        test_count=64,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.STRESS_SCENARIOS),
        series="wizard",
        tags=_ft("self_play", "curriculum", "synthesis", "latent"),
    ),

    # ── Sections 34–66: Consolidated Patch Series (1,406 tests) ─────────────

    TestSuite(
        id="academic_refinements",
        name="Academic Refinements — KM, Banach, IQC, Catastrophe",
        section=34,
        source="test_aeon_unified.py",
        test_count=70,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="acadref",
        tags=_ft("km_iteration", "banach", "iqc", "catastrophe", "causal"),
    ),
    TestSuite(
        id="act_patches",
        name="ACT-Series — Oscillation, Spectral, Certificate, Criticality",
        section=35,
        source="test_aeon_unified.py",
        test_count=29,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="act",
        tags=_ft("oscillation", "spectral", "certificate", "criticality", "feedback_bus"),
    ),
    TestSuite(
        id="ca_patches",
        name="CA-Series — Meta-Cognitive Recursion, Spectral Bifurcation",
        section=36,
        source="test_aeon_unified.py",
        test_count=56,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="ca",
        tags=_ft("meta_cognitive", "recursion", "spectral", "bifurcation", "rssm"),
    ),
    TestSuite(
        id="cact_patches",
        name="CACT-Series — Orphaned Signals, Bidirectional Bridges",
        section=37,
        source="test_aeon_unified.py",
        test_count=44,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="cact",
        tags=_ft("orphaned_signals", "bidirectional", "bridge", "compute_loss"),
    ),
    TestSuite(
        id="cogact_patches",
        name="COGACT-Series — Unified Convergence, Lipschitz, SSM",
        section=38,
        source="test_aeon_unified.py",
        test_count=35,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="cogact",
        tags=_ft("convergence", "lipschitz", "ssm", "iqc", "km_convergence"),
    ),
    TestSuite(
        id="cogfinal_patches",
        name="COGFINAL — Anderson Safeguard, Lyapunov, Memory",
        section=39,
        source="test_aeon_unified.py",
        test_count=35,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.SECURITY_RELIABILITY),
        series="cogfinal",
        tags=_ft("anderson", "lyapunov", "memory_validation", "training_coherence"),
    ),
    TestSuite(
        id="cognitive_activation_final",
        name="Cognitive Activation Final — LayerNorm, Gates, IQC",
        section=40,
        source="test_aeon_unified.py",
        test_count=54,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="cafinal",
        tags=_ft("layernorm", "lipschitz", "feedback_gate", "iqc", "dropout"),
    ),
    TestSuite(
        id="cognitive_analysis_fixes",
        name="Cognitive Analysis — Contraction, NOTEARS, Von Neumann",
        section=41,
        source="test_aeon_unified.py",
        test_count=30,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="cafix",
        tags=_ft("contraction", "layernorm", "notears", "von_neumann"),
    ),
    TestSuite(
        id="cp_integration_patches",
        name="CP-Integration — Causal Trace, Training Bus",
        section=42,
        source="test_aeon_unified.py",
        test_count=44,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="cpint",
        tags=_ft("causal_trace", "training_bus", "feedback_bus", "diagnostic"),
    ),
    TestSuite(
        id="cp_patches",
        name="CP-Series — Catastrophe MCT, Curriculum, Diversity",
        section=43,
        source="test_aeon_unified.py",
        test_count=43,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="cp",
        tags=_ft("catastrophe", "mct", "curriculum", "diversity", "consistency"),
    ),
    TestSuite(
        id="d_series_patches",
        name="D-Series Patches — Error Evolution, Recursion, Oscillation",
        section=44,
        source="test_aeon_unified.py",
        test_count=44,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="dseries",
        tags=_ft("error_evolution", "recursion", "oscillation", "training_bus"),
    ),
    TestSuite(
        id="deep_cognitive_analysis",
        name="Deep Cognitive Analysis — T-IQC, LayerNorm, KM",
        section=45,
        source="test_aeon_unified.py",
        test_count=46,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="deepca",
        tags=_ft("t_iqc", "layernorm", "km_convergence", "eclipse", "hessian"),
    ),
    TestSuite(
        id="emerge_patches",
        name="EMERGE-Series — Memory Bus, Social/Sandbox, World Model",
        section=46,
        source="test_aeon_unified.py",
        test_count=48,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="emerge",
        tags=_ft("memory_bus", "social", "sandbox", "world_model", "integrity"),
    ),
    TestSuite(
        id="emrg_patches",
        name="EMRG-Series — Error Recording, Convergence Spectral",
        section=47,
        source="test_aeon_unified.py",
        test_count=35,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.LEARNING_ADAPTATION),
        series="emrg",
        tags=_ft("error_recording", "convergence", "spectral", "verification"),
    ),
    TestSuite(
        id="fca_patches",
        name="FCA-Series — Convergence Guarantees, KM Bounds, IQC",
        section=48,
        source="test_aeon_unified.py",
        test_count=52,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="fca",
        tags=_ft("convergence", "km_bounds", "iqc", "lyapunov", "spectral"),
    ),
    TestSuite(
        id="fci_patches",
        name="FCI-Series — Lyapunov Iteration, Gamma, Joint Lipschitz",
        section=49,
        source="test_aeon_unified.py",
        test_count=45,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="fci",
        tags=_ft("lyapunov", "gamma", "lipschitz", "finite_iterate", "eclipse"),
    ),
    TestSuite(
        id="fia_patches",
        name="FIA-Series — Post-Output Uncertainty, Verdict, Recovery",
        section=50,
        source="test_aeon_unified.py",
        test_count=43,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="fia",
        tags=_ft("post_output", "uncertainty", "verdict", "recovery", "causal_trace"),
    ),
    TestSuite(
        id="final_cognitive_activation",
        name="Final Cognitive Activation — Bus, MCT, Causal",
        section=51,
        source="test_aeon_unified.py",
        test_count=37,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="fincogact",
        tags=_ft("bus_consumption", "mct", "causal_verification", "adaptive_lr"),
    ),
    TestSuite(
        id="final_integration_v2",
        name="Final Integration — Spectral Gates, Anderson",
        section=52,
        source="test_aeon_unified.py",
        test_count=28,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="finint",
        tags=_ft("spectral_gates", "anderson", "anti_collapse", "sandwich"),
    ),
    TestSuite(
        id="final_int_cognitive_activation",
        name="Final Integration Activation — Sandwich, Spectral",
        section=53,
        source="test_aeon_unified.py",
        test_count=34,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="fintcogact",
        tags=_ft("sandwich", "residual", "theta_history", "spectral", "banach"),
    ),
    TestSuite(
        id="final_integration_patches_v2",
        name="Final Integration Patches — Bus Init, Cross-Pass",
        section=54,
        source="test_aeon_unified.py",
        test_count=41,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="finintpatch",
        tags=_ft("bus_init", "cross_pass", "oscillation", "provenance", "silent_failure"),
    ),
    TestSuite(
        id="final_patches_v2",
        name="Final Patches — Error Recovery, Orphaned, MCT Loss",
        section=55,
        source="test_aeon_unified.py",
        test_count=39,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="final",
        tags=_ft("error_recovery", "orphaned_signals", "mct_loss", "dampening"),
    ),
    TestSuite(
        id="gap_patches",
        name="GAP-Series — Oscillation MCT, Anderson, Decoder",
        section=56,
        source="test_aeon_unified.py",
        test_count=50,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="gap",
        tags=_ft("oscillation", "mct", "anderson", "decoder", "cross_validator"),
    ),
    TestSuite(
        id="integration_patches_v2",
        name="Integration Patches — Spectral, Feedback Gate",
        section=57,
        source="test_aeon_unified.py",
        test_count=46,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION, CognitiveCategory.COGNITIVE_ACTIVATION),
        series="intpatch",
        tags=_ft("spectral", "feedback_gate", "output_reliability", "subsystem_health"),
    ),
    TestSuite(
        id="k_series_patches",
        name="K-Series — UCC Coherence, SSP, Reexecution",
        section=58,
        source="test_aeon_unified.py",
        test_count=41,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="kseries",
        tags=_ft("ucc", "coherence", "ssp", "reexecution", "diversity"),
    ),
    TestSuite(
        id="p_series_patches",
        name="P-Series — Wizard Bridge, Causal Trace, Provenance",
        section=59,
        source="test_aeon_unified.py",
        test_count=40,
        categories=_fs(CognitiveCategory.COMPONENT_INTEGRATION),
        series="pseries",
        tags=_ft("wizard", "bridge", "causal_trace", "provenance"),
    ),
    TestSuite(
        id="r_series_patches",
        name="R-Series — Lipschitz, Contraction, Recursion",
        section=60,
        source="test_aeon_unified.py",
        test_count=40,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="rseries",
        tags=_ft("lipschitz", "contraction", "recursion", "catastrophe", "spectral"),
    ),
    TestSuite(
        id="rigor_patches",
        name="RIGOR-Series — SSM Diagonal, Banach, LayerNorm, KM",
        section=61,
        source="test_aeon_unified.py",
        test_count=48,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="rigor",
        tags=_ft("ssm", "banach", "layernorm", "km_iteration", "hessian"),
    ),
    TestSuite(
        id="s_series_patches",
        name="S-Series — Silent Exceptions, Missing Signals, Orphans",
        section=62,
        source="test_aeon_unified.py",
        test_count=64,
        categories=_fs(CognitiveCategory.SECURITY_RELIABILITY, CognitiveCategory.COMPONENT_INTEGRATION),
        series="sseries",
        tags=_ft("silent_exceptions", "missing_signals", "orphaned_signals", "feedback_bus"),
    ),
    TestSuite(
        id="sigma_integration_patches",
        name="SIGMA-Integration — MCT Planner, Convergence, OOM",
        section=63,
        source="test_aeon_unified.py",
        test_count=42,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="sigmaint",
        tags=_ft("mcts", "convergence_monitor", "provenance", "oom", "cache"),
    ),
    TestSuite(
        id="sigma_patches",
        name="SIGMA-Series — Seven Sigma Patches (Σ1–Σ7)",
        section=64,
        source="test_aeon_unified.py",
        test_count=26,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="sigma",
        tags=_ft("sigma", "mcts", "convergence", "provenance", "diagnostic"),
    ),
    TestSuite(
        id="syn_patches",
        name="SYN-Series — Stall Severity, Axiom Bus, Error Recovery",
        section=65,
        source="test_aeon_unified.py",
        test_count=39,
        categories=_fs(CognitiveCategory.COGNITIVE_ACTIVATION, CognitiveCategory.COMPONENT_INTEGRATION),
        series="syn",
        tags=_ft("stall_severity", "axiom", "error_recovery", "null_causal_trace"),
    ),
    TestSuite(
        id="theoretical_rigor",
        name="Theoretical Rigor — IQC, Catastrophe, DAG, KM",
        section=66,
        source="test_aeon_unified.py",
        test_count=38,
        categories=_fs(CognitiveCategory.LEARNING_ADAPTATION, CognitiveCategory.AGI_AXIOMS),
        series="theorig",
        tags=_ft("iqc", "catastrophe", "dag", "km_convergence"),
    ),
)

# ── Derived indices for O(1) lookup ─────────────────────────────────────────

SUITE_BY_ID: Dict[str, TestSuite] = {s.id: s for s in SUITE_REGISTRY}
SUITE_BY_SERIES: Dict[str, List[TestSuite]] = defaultdict(list)
SUITE_BY_CATEGORY: Dict[CognitiveCategory, List[TestSuite]] = defaultdict(list)
SUITE_BY_TAG: Dict[str, List[TestSuite]] = defaultdict(list)

for _s in SUITE_REGISTRY:
    SUITE_BY_SERIES[_s.series].append(_s)
    for _c in _s.categories:
        SUITE_BY_CATEGORY[_c].append(_s)
    for _t in _s.tags:
        SUITE_BY_TAG[_t].append(_s)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FILTER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFilter:
    """Composable filter that resolves a set of suites from user intent.

    Filters are combined conjunctively (intersection) when multiple
    criteria are specified, except ``series`` and ``suites`` which are
    additive (union) within their group.
    """
    series: List[str] = field(default_factory=list)
    categories: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    suites: List[str] = field(default_factory=list)
    exclude_series: List[str] = field(default_factory=list)
    exclude_suites: List[str] = field(default_factory=list)
    integration_only: bool = False
    wizard_only: bool = False
    core_only: bool = False

    def resolve(self) -> List[TestSuite]:
        """Resolve filter to an ordered list of TestSuite objects."""
        candidates: Optional[Set[str]] = None

        # Additive: series union
        if self.series:
            ids: Set[str] = set()
            for s in self.series:
                for suite in SUITE_BY_SERIES.get(s, []):
                    ids.add(suite.id)
            candidates = ids if candidates is None else candidates & ids

        # Additive: explicit suite ids
        if self.suites:
            ids = {sid for sid in self.suites if sid in SUITE_BY_ID}
            candidates = ids if candidates is None else candidates | ids

        # Restrictive: categories
        if self.categories:
            ids = set()
            for cat_num in self.categories:
                cat = CognitiveCategory(cat_num)
                for suite in SUITE_BY_CATEGORY.get(cat, []):
                    ids.add(suite.id)
            candidates = ids if candidates is None else candidates & ids

        # Restrictive: tags
        if self.tags:
            ids = set()
            for tag in self.tags:
                for suite in SUITE_BY_TAG.get(tag, []):
                    ids.add(suite.id)
            candidates = ids if candidates is None else candidates & ids

        # Shortcut filters
        if self.integration_only:
            ids = {s.id for s in SUITE_REGISTRY if "integration" in s.id or "final" in s.id or s.series.endswith("_int")}
            candidates = ids if candidates is None else candidates & ids

        if self.wizard_only:
            ids = {s.id for s in SUITE_REGISTRY if "wizard" in s.id}
            candidates = ids if candidates is None else candidates & ids

        if self.core_only:
            ids = {"core"}
            candidates = ids if candidates is None else candidates & ids

        # Default: all
        if candidates is None:
            candidates = {s.id for s in SUITE_REGISTRY}

        # Exclusions
        for s in self.exclude_series:
            for suite in SUITE_BY_SERIES.get(s, []):
                candidates.discard(suite.id)
        for sid in self.exclude_suites:
            candidates.discard(sid)

        # Maintain registry order
        return [s for s in SUITE_REGISTRY if s.id in candidates]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PYTEST COMMAND BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

UNIFIED_TEST_FILE = "test_aeon_unified.py"


def _test_class_patterns(suite: TestSuite) -> List[str]:
    """Derive pytest -k patterns from the suite's source file.

    We match test classes that originated from the source file.  For the
    core suite (test_fixes.py), all top-level ``def test_*`` functions
    are included by running without a -k filter.
    """
    if suite.id == "core":
        return []  # No filtering needed; runs everything in section 01

    # Map source file to the test class names within it.
    # This is derived from the known class structure.
    return [f"Section{suite.section:02d}"]


@dataclass
class PytestCommand:
    """Structured representation of a pytest invocation."""
    file: str = UNIFIED_TEST_FILE
    keywords: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    verbose: bool = False
    dry_run: bool = False
    parallel: int = 0  # 0 = sequential

    def to_argv(self) -> List[str]:
        """Build the full argv list for subprocess or pytest.main()."""
        argv = [sys.executable, "-m", "pytest", self.file]

        if self.keywords:
            # Join with OR for multiple suite patterns
            argv.extend(["-k", " or ".join(self.keywords)])

        for m in self.markers:
            argv.extend(["-m", m])

        if self.verbose:
            argv.append("-v")

        argv.extend(["--tb=short", "-q"])

        if self.parallel > 1:
            argv.extend(["-n", str(self.parallel)])

        argv.extend(self.extra_args)

        return argv

    def to_display(self) -> str:
        """Human-readable command string."""
        parts = self.to_argv()
        return " ".join(parts)


def build_command(
    suites: List[TestSuite],
    verbose: bool = False,
    parallel: int = 0,
    extra_args: Optional[List[str]] = None,
) -> PytestCommand:
    """Build a PytestCommand from a list of suites.

    If all suites are selected, no keyword filter is applied (run everything).
    Otherwise, keyword expressions target specific test classes by matching
    class names from each suite's original source file.
    """
    all_selected = len(suites) == len(SUITE_REGISTRY)

    keywords: List[str] = []
    if not all_selected:
        for suite in suites:
            # Build keyword expression based on known test class names
            keywords.append(suite.id)

    return PytestCommand(
        keywords=keywords if not all_selected else [],
        verbose=verbose,
        parallel=parallel,
        extra_args=extra_args or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Captures the outcome of a test run."""
    suites_requested: int
    tests_expected: int
    return_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def passed(self) -> bool:
        return self.return_code == 0


def execute(
    filt: TestFilter,
    verbose: bool = False,
    dry_run: bool = False,
    parallel: int = 0,
    extra_args: Optional[List[str]] = None,
) -> Optional[ExecutionResult]:
    """Execute tests matching the given filter.

    Parameters
    ----------
    filt : TestFilter
        Describes which suites to include/exclude.
    verbose : bool
        Pass ``-v`` to pytest.
    dry_run : bool
        Print the command without running it.
    parallel : int
        Number of parallel workers (requires pytest-xdist).
    extra_args : list[str] | None
        Additional arguments forwarded to pytest.

    Returns
    -------
    ExecutionResult | None
        Result of the run, or ``None`` if dry-run.
    """
    suites = filt.resolve()
    if not suites:
        print("⚠  No suites matched the filter criteria.")
        return None

    total_tests = sum(s.test_count for s in suites)

    cmd = build_command(suites, verbose=verbose, parallel=parallel, extra_args=extra_args)

    # ── Display plan ────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AEON-Delta Test Control Panel — Execution Plan             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Suites:  {len(suites):>4d} / {len(SUITE_REGISTRY):<4d}"
          f"{'':>36s}║")
    print(f"║  Tests:   {total_tests:>5d}"
          f"{'':>43s}║")
    print("╠══════════════════════════════════════════════════════════════╣")

    for s in suites:
        cats = ", ".join(str(c.value) for c in sorted(s.categories))
        print(f"║  [{s.section:02d}] {s.name[:42]:<42s} ({s.test_count:>4d}) C{cats}  ║"[:63] + "║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Command: {cmd.to_display()[:49]:<49s}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    if dry_run:
        print("  (dry-run — no tests executed)")
        return None

    # ── Execute ─────────────────────────────────────────────────────────
    start = time.monotonic()
    argv = cmd.to_argv()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.monotonic() - start

    result = ExecutionResult(
        suites_requested=len(suites),
        tests_expected=total_tests,
        return_code=proc.returncode,
        duration_seconds=round(elapsed, 2),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )

    # ── Print summary ───────────────────────────────────────────────────
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"  {status}  ({result.duration_seconds}s, rc={result.return_code})")

    if not result.passed:
        # Show failing output
        lines = result.stdout.strip().split("\n")
        # Print last 50 lines (usually the summary)
        for line in lines[-50:]:
            print(f"  │ {line}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — REPORT & ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_FILE = "cognitive_test_report.json"


@dataclass
class SuiteHealth:
    """Health metrics for a single test suite."""
    suite_id: str
    suite_name: str
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: float
    pass_rate: float
    category_coverage: List[int]

    def status_icon(self) -> str:
        if self.failed > 0 or self.errors > 0:
            return "❌"
        if self.skipped > 0:
            return "⚠️"
        return "✅"


def load_report(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the cognitive test report JSON from the last run."""
    report_path = Path(path or REPORT_FILE)
    if not report_path.exists():
        print(f"  ⚠  Report not found: {report_path}")
        return None
    with report_path.open() as f:
        return json.load(f)


def generate_report_summary(report: Optional[Dict[str, Any]] = None) -> None:
    """Print a structured summary of the last test run.

    Reads ``cognitive_test_report.json`` (produced by conftest.py) and
    presents a per-category, per-suite breakdown with health indicators.
    """
    if report is None:
        report = load_report()
    if report is None:
        return

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AEON-Delta — Cognitive Test Report Summary                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    duration = summary.get("duration_seconds", 0)

    print(f"║  Total: {total:>5d}  Passed: {passed:>5d}  "
          f"Failed: {failed:>5d}  Skipped: {skipped:>5d}  ║")
    print(f"║  Duration: {duration:>7.1f}s  "
          f"Pass Rate: {passed/max(total,1)*100:>5.1f}%"
          f"{'':>21s}║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # Per-category breakdown
    categories = report.get("categories", {})
    for cat_num in range(1, 8):
        cat = CognitiveCategory(cat_num)
        cat_data = categories.get(str(cat_num), {})
        cat_total = cat_data.get("total", 0)
        cat_passed = cat_data.get("passed", 0)
        icon = "✅" if cat_passed == cat_total and cat_total > 0 else "❌" if cat_total > 0 else "⬜"
        print(f"║  {icon} Cat {cat_num}: {cat.label[:45]:<45s} "
              f"{cat_passed}/{cat_total:>3d}  ║"[:63] + "║")

    print("╠══════════════════════════════════════════════════════════════╣")

    # Error evolution data
    ee = report.get("error_evolution", {})
    if ee:
        total_errors = ee.get("total_errors_recorded", 0)
        error_classes = ee.get("error_classes", {})
        print(f"║  Error Evolution: {total_errors} recorded"
              f"{'':>35s}║"[:63] + "║")
        for ec, count in sorted(error_classes.items(), key=lambda x: -x[1])[:5]:
            print(f"║    {ec[:40]:<40s} ×{count:>3d}"
                  f"{'':>14s}║"[:63] + "║")

    print("╚══════════════════════════════════════════════════════════════╝")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — LISTING & INTROSPECTION
# ═══════════════════════════════════════════════════════════════════════════════

def list_suites(filt: Optional[TestFilter] = None) -> None:
    """Print a formatted table of all registered test suites."""
    suites = filt.resolve() if filt else list(SUITE_REGISTRY)

    print()
    print("╔════╦═══════════════════════════════════════════════╦═══════╦════════════╗")
    print("║ §  ║ Suite Name                                    ║ Tests ║ Categories ║")
    print("╠════╬═══════════════════════════════════════════════╬═══════╬════════════╣")

    total = 0
    for s in suites:
        cats = ",".join(str(c.value) for c in sorted(s.categories))
        print(f"║ {s.section:>2d} ║ {s.name[:45]:<45s} ║ {s.test_count:>5d} ║ {cats:<10s} ║")
        total += s.test_count

    print("╠════╬═══════════════════════════════════════════════╬═══════╬════════════╣")
    print(f"║    ║ {'TOTAL':<45s} ║ {total:>5d} ║            ║")
    print("╚════╩═══════════════════════════════════════════════╩═══════╩════════════╝")
    print()


def list_categories() -> None:
    """Print category definitions with suite counts."""
    print()
    print("╔════╦══════════════════════════════════════╦════════╦═══════╗")
    print("║ #  ║ Category                             ║ Suites ║ Tests ║")
    print("╠════╬══════════════════════════════════════╬════════╬═══════╣")

    for cat in CognitiveCategory:
        suites = SUITE_BY_CATEGORY.get(cat, [])
        total = sum(s.test_count for s in suites)
        print(f"║ {cat.value:>2d} ║ {cat.label[:36]:<36s} ║ {len(suites):>6d} ║ {total:>5d} ║")

    print("╚════╩══════════════════════════════════════╩════════╩═══════╝")
    print()


def list_series() -> None:
    """Print available series codes with suite counts."""
    print()
    print("╔═══════════╦════════╦═══════╗")
    print("║ Series    ║ Suites ║ Tests ║")
    print("╠═══════════╬════════╬═══════╣")

    for series in sorted(SUITE_BY_SERIES.keys()):
        suites = SUITE_BY_SERIES[series]
        total = sum(s.test_count for s in suites)
        print(f"║ {series:<9s} ║ {len(suites):>6d} ║ {total:>5d} ║")

    print("╚═══════════╩════════╩═══════╝")
    print()


def list_tags() -> None:
    """Print available tags with suite counts."""
    print()
    print("╔═════════════════════════╦════════╗")
    print("║ Tag                     ║ Suites ║")
    print("╠═════════════════════════╬════════╣")

    for tag in sorted(SUITE_BY_TAG.keys()):
        suites = SUITE_BY_TAG[tag]
        print(f"║ {tag:<23s} ║ {len(suites):>6d} ║")

    print("╚═════════════════════════╩════════╝")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="aeon_test_control_panel",
        description="AEON-Delta Test Control Panel — Selective test execution "
                    "and analytics for the unified test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --all                  Run all 6037 tests\n"
            "  %(prog)s --series c              Run C-series patches\n"
            "  %(prog)s --series v --series v5   Run V and V5 series\n"
            "  %(prog)s --category 1            Run AGI Axioms tests\n"
            "  %(prog)s --tag lipschitz         Run Lipschitz-tagged tests\n"
            "  %(prog)s --integration           Run integration tests\n"
            "  %(prog)s --wizard                Run wizard tests\n"
            "  %(prog)s --list                  Show all suites\n"
            "  %(prog)s --list-categories       Show category breakdown\n"
            "  %(prog)s --list-series           Show series breakdown\n"
            "  %(prog)s --list-tags             Show available tags\n"
            "  %(prog)s --report                Show last run report\n"
            "  %(prog)s --series d --dry-run    Dry-run D-series\n"
        ),
    )

    # ── Selection group ─────────────────────────────────────────────────
    sel = p.add_argument_group("Test selection")
    sel.add_argument(
        "--all", action="store_true",
        help="Run ALL tests (default if no filter specified)",
    )
    sel.add_argument(
        "--series", action="append", default=[], metavar="CODE",
        help="Run suites from the given series (repeatable)",
    )
    sel.add_argument(
        "--category", action="append", type=int, default=[], metavar="N",
        help="Run suites from cognitive category N (1-7, repeatable)",
    )
    sel.add_argument(
        "--tag", action="append", default=[], metavar="TAG",
        help="Run suites with the given tag (repeatable)",
    )
    sel.add_argument(
        "--suite", action="append", default=[], metavar="ID",
        help="Run a specific suite by ID (repeatable)",
    )
    sel.add_argument(
        "--integration", action="store_true",
        help="Run integration test suites only",
    )
    sel.add_argument(
        "--wizard", action="store_true",
        help="Run wizard test suites only",
    )
    sel.add_argument(
        "--core", action="store_true",
        help="Run core component tests only",
    )

    # ── Exclusion group ─────────────────────────────────────────────────
    exc = p.add_argument_group("Exclusions")
    exc.add_argument(
        "--exclude-series", action="append", default=[], metavar="CODE",
        help="Exclude suites from the given series",
    )
    exc.add_argument(
        "--exclude-suite", action="append", default=[], metavar="ID",
        help="Exclude a specific suite by ID",
    )

    # ── Execution options ───────────────────────────────────────────────
    exe = p.add_argument_group("Execution options")
    exe.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose pytest output",
    )
    exe.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running tests",
    )
    exe.add_argument(
        "-j", "--parallel", type=int, default=0, metavar="N",
        help="Number of parallel workers (requires pytest-xdist)",
    )

    # ── Listing & reporting ─────────────────────────────────────────────
    info = p.add_argument_group("Information & reporting")
    info.add_argument(
        "--list", action="store_true", dest="list_suites",
        help="List all registered test suites",
    )
    info.add_argument(
        "--list-categories", action="store_true",
        help="Show cognitive category breakdown",
    )
    info.add_argument(
        "--list-series", action="store_true",
        help="Show series breakdown",
    )
    info.add_argument(
        "--list-tags", action="store_true",
        help="Show available tags",
    )
    info.add_argument(
        "--report", action="store_true",
        help="Generate report summary from last run",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Information commands (no execution) ──────────────────────────────
    if args.list_suites:
        list_suites()
        return 0
    if args.list_categories:
        list_categories()
        return 0
    if args.list_series:
        list_series()
        return 0
    if args.list_tags:
        list_tags()
        return 0
    if args.report:
        generate_report_summary()
        return 0

    # ── Build filter ────────────────────────────────────────────────────
    filt = TestFilter(
        series=args.series,
        categories=args.category,
        tags=args.tag,
        suites=args.suite,
        exclude_series=args.exclude_series,
        exclude_suites=args.exclude_suite,
        integration_only=args.integration,
        wizard_only=args.wizard,
        core_only=args.core,
    )

    # ── Execute ─────────────────────────────────────────────────────────
    result = execute(
        filt,
        verbose=args.verbose,
        dry_run=args.dry_run,
        parallel=args.parallel,
    )

    if result is None:
        return 0
    return result.return_code


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(main())
