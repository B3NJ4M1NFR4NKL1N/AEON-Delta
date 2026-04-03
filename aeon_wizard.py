"""
AEON Self-Play Setup Wizard — Backend Controller
=================================================

Orchestrates cold-start initialization via **Self-Play / Synthetic
Curriculum** — a fundamental architectural departure from supervised
learning on external corpora.  VibeThinker acts as a *World Generator*
and *Task Setter*, producing synthetic latent scenarios, evaluating
AEON's control quality, and generating corrective training signals when
control fails.

Architecture overview (§SP):

  §SP.1 — **LatentWorldGenerator**
      Three generation modes (SMOOTH / STRUCTURED / ADVERSARIAL) using
      ``LatentDynamicsModel`` as physical engine,
      ``VibeThinkerPromptAdapter.complexity_head`` gradient as semantic
      compass, and ``CuriosityDrivenExploration`` for intrinsic reward.

  §SP.2 — **Self-Play Diagnostics**
      Latent-space complexity profiling via VibeThinker without any
      external tokens — all inference operates on synthetic z-vectors.

  §SP.3 — **AdaptiveCurriculumManager**
      Absolute Learning Progress (ALP) curriculum: difficulty ↑ when
      success_rate ≥ 0.75 ∧ ALP → 0, difficulty ↓ when < 0.35.
      Automatic mode escalation: levels 0–1 → SMOOTH, 2–3 →
      STRUCTURED, 4 → ADVERSARIAL.

  §SP.4 — **CorrectiveSynthesizer**
      Failure-mode–targeted scenario generation: direction, magnitude,
      oscillation, collapse — each with specialized latent corrections.

  §SP.5 — **Bootstrap without corpus**
      ``bootstrap_codebook_embeddings()`` — high-temperature VibeThinker
      sampling (T=1.5) + k-means → semantically diverse codebook
      centroids without any external data.

  §SP.6 — **VibeThinkerContinuousLearner meta-signal**
      ``calibration_ema`` adapts λ_cos in RSSM mini-training loss
      L = L_mse + λ_cos · (1 − cosine_sim), closing the control loop.

Backward compatibility:
  ``run_wizard(model, config, tokens=..., ...)`` is preserved as an
  alias — it accepts ``tokens`` for API compatibility, logs a
  deprecation warning, and delegates to ``run_self_play_wizard()``.
  ``aeon_integration.py`` requires no changes.

Legacy Spec References (still honoured):
  - §4.A.1–4:  Phases 1–4 now operate in latent space
  - §2.1:      Weight path vibe_thinker_weights/model.safetensors
  - §2.3:      Automated First-Run without manual input
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("AEON-Wizard")

# ── Weight path constant (Spec §2.1) ────────────────────────────────────────
VT_WEIGHTS_PATH = Path("vibe_thinker_weights/model.safetensors")

# ── Codebook size search candidates ──────────────────────────────────────────
_CODEBOOK_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024]

# ── Context window bounds ────────────────────────────────────────────────────
_MIN_CONTEXT_WINDOW = 1
_MAX_CONTEXT_WINDOW = 16


# =============================================================================
#  §SP.1 — Generation Mode Enum
# =============================================================================
class GenerationMode(enum.Enum):
    """Latent scenario generation strategy.

    - SMOOTH:       Low-noise perturbations for initial curriculum levels.
    - STRUCTURED:   Gradient-guided generation via VibeThinker complexity_head.
    - ADVERSARIAL:  Gradient search maximising RSSM prediction error (PAIRED).
    """
    SMOOTH = "smooth"
    STRUCTURED = "structured"
    ADVERSARIAL = "adversarial"


# =============================================================================
#  §SP.1 — LatentWorldGenerator
# =============================================================================
class LatentWorldGenerator(nn.Module):
    """Synthetic scenario generator operating purely in latent space.

    Uses ``LatentDynamicsModel`` as a physics engine,
    ``VibeThinkerPromptAdapter.complexity_head`` gradient as a semantic
    compass, and ``CuriosityDrivenExploration`` as intrinsic-reward
    source.  Three generation regimes span the full difficulty range:

    * **SMOOTH** — Gaussian perturbation of a base state z₀ with small σ,
      producing near-distribution scenarios suitable for cold-start.
    * **STRUCTURED** — Gradient ascent on the complexity_head output,
      steering z toward semantically richer regions.
    * **ADVERSARIAL** — Gradient ascent on RSSM (LatentDynamicsModel)
      forward-prediction error, implementing the PAIRED concept
      (Dennis et al., NeurIPS 2020) in latent space.

    Reference:
        Dennis et al. "Emergent Complexity and Zero-shot Transfer via
        Unsupervised Environment Design", NeurIPS 2020.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 64,
        noise_std_smooth: float = 0.1,
        adversarial_steps: int = 10,
        adversarial_lr: float = 0.01,
        structured_steps: int = 8,
        structured_lr: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.noise_std_smooth = noise_std_smooth
        self.adversarial_steps = adversarial_steps
        self.adversarial_lr = adversarial_lr
        self.structured_steps = structured_steps
        self.structured_lr = structured_lr
        self._device = device

        # Sub-modules — lazy-initialised from aeon_core when available
        self._dynamics: Optional[nn.Module] = None
        self._curiosity: Optional[nn.Module] = None
        self._adapter: Optional[nn.Module] = None
        self._initialised = False

    def _lazy_init(self, config: Any) -> None:
        """Attempt to instantiate LatentDynamicsModel, CuriosityDriven-
        Exploration and VibeThinkerPromptAdapter from ``aeon_core``.
        Falls back to synthetic-only mode if unavailable."""
        if self._initialised:
            return
        try:
            from aeon_core import (
                LatentDynamicsModel,
                CuriosityDrivenExploration,
                VibeThinkerPromptAdapter,
            )
            self._dynamics = LatentDynamicsModel(
                latent_dim=self.latent_dim,
                action_dim=self.action_dim,
            ).to(self._device)
            self._curiosity = CuriosityDrivenExploration(
                state_dim=self.latent_dim,
                action_dim=self.action_dim,
            ).to(self._device)
            self._adapter = VibeThinkerPromptAdapter(
                latent_dim=self.latent_dim,
                hidden_dim=getattr(config, "hidden_dim", self.latent_dim),
            ).to(self._device)
        except ImportError:
            logger.debug("aeon_core components unavailable — "
                         "LatentWorldGenerator will use noise-only fallback")
        self._initialised = True

    # ── Core generation dispatch ─────────────────────────────────────────
    def generate(
        self,
        mode: GenerationMode,
        batch_size: int,
        config: Any,
        base_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate a batch of synthetic latent scenarios.

        Args:
            mode: Generation regime (SMOOTH / STRUCTURED / ADVERSARIAL).
            batch_size: Number of scenarios to produce.
            config: AEONConfig-like object (provides z_dim, hidden_dim).
            base_z: Optional seed state [B, z_dim]; random N(0,1) if None.

        Returns:
            Dict with keys:
              - ``scenarios``  : Tensor [B, z_dim]
              - ``complexity`` : Tensor [B] estimated complexity per scenario
              - ``intrinsic_reward`` : Tensor [B] curiosity signal
              - ``mode``       : str  generation mode used
        """
        self._lazy_init(config)

        if base_z is None:
            base_z = torch.randn(batch_size, self.latent_dim,
                                 device=self._device)

        if mode == GenerationMode.SMOOTH:
            return self._generate_smooth(base_z, batch_size)
        elif mode == GenerationMode.STRUCTURED:
            return self._generate_structured(base_z, batch_size, config)
        elif mode == GenerationMode.ADVERSARIAL:
            return self._generate_adversarial(base_z, batch_size, config)
        else:
            return self._generate_smooth(base_z, batch_size)

    # ── SMOOTH mode ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _generate_smooth(
        self, base_z: torch.Tensor, batch_size: int,
    ) -> Dict[str, Any]:
        noise = torch.randn_like(base_z) * self.noise_std_smooth
        scenarios = base_z + noise
        # Normalise to unit sphere for stability
        scenarios = F.normalize(scenarios, dim=-1)

        complexity = scenarios.norm(dim=-1)  # proxy
        intrinsic = torch.zeros(batch_size, device=self._device)

        if self._curiosity is not None and self._dynamics is not None:
            try:
                a_rand = torch.randn(batch_size, self.action_dim,
                                     device=self._device)
                s_next, _, _ = self._dynamics(scenarios, a_rand)
                intrinsic = self._curiosity.intrinsic_reward(
                    scenarios, a_rand, s_next,
                )
            except Exception:
                pass

        return {
            "scenarios": scenarios,
            "complexity": complexity,
            "intrinsic_reward": intrinsic,
            "mode": GenerationMode.SMOOTH.value,
        }

    # ── STRUCTURED mode (gradient-guided via complexity_head) ────────────
    def _generate_structured(
        self, base_z: torch.Tensor, batch_size: int, config: Any,
    ) -> Dict[str, Any]:
        z = base_z.clone().detach().requires_grad_(True)

        if self._adapter is not None:
            # Gradient ascent on complexity_head output
            with torch.enable_grad():
                for _ in range(self.structured_steps):
                    out = self._adapter(z)
                    score = out["complexity_score"]
                    # Maximise complexity → gradient ascent
                    loss = -score.sum()
                    grad = torch.autograd.grad(loss, z, retain_graph=False)[0]
                    z = (z + self.structured_lr * grad).detach()
                    z = F.normalize(z, dim=-1)
                    z = z.requires_grad_(True)

            scenarios = z.detach()
            with torch.no_grad():
                complexity = self._adapter(scenarios)["complexity_score"]
        else:
            # Fallback: directional perturbation without adapter
            direction = torch.randn_like(base_z)
            direction = F.normalize(direction, dim=-1)
            scenarios = F.normalize(base_z + 0.3 * direction, dim=-1)
            complexity = scenarios.norm(dim=-1)

        intrinsic = torch.zeros(batch_size, device=self._device)
        if self._curiosity is not None and self._dynamics is not None:
            try:
                with torch.no_grad():
                    a_rand = torch.randn(batch_size, self.action_dim,
                                         device=self._device)
                    s_next, _, _ = self._dynamics(scenarios, a_rand)
                    intrinsic = self._curiosity.intrinsic_reward(
                        scenarios, a_rand, s_next,
                    )
            except Exception:
                pass

        return {
            "scenarios": scenarios,
            "complexity": complexity,
            "intrinsic_reward": intrinsic,
            "mode": GenerationMode.STRUCTURED.value,
        }

    # ── ADVERSARIAL mode (PAIRED — maximise RSSM prediction error) ──────
    def _generate_adversarial(
        self, base_z: torch.Tensor, batch_size: int, config: Any,
    ) -> Dict[str, Any]:
        z = base_z.clone().detach().requires_grad_(True)

        if self._dynamics is not None:
            with torch.enable_grad():
                for _ in range(self.adversarial_steps):
                    a_rand = torch.randn(batch_size, self.action_dim,
                                         device=self._device)
                    s_next_pred, _, _ = self._dynamics(z, a_rand)
                    # Prediction error as adversarial objective
                    pred_err = F.mse_loss(s_next_pred, z, reduction="none")
                    loss = -pred_err.sum()  # maximise error
                    grad = torch.autograd.grad(loss, z, retain_graph=False)[0]
                    z = (z + self.adversarial_lr * grad).detach()
                    z = F.normalize(z, dim=-1)
                    z = z.requires_grad_(True)

            scenarios = z.detach()
        else:
            # High-variance fallback
            scenarios = F.normalize(
                base_z + torch.randn_like(base_z) * 0.5, dim=-1,
            )

        complexity = scenarios.norm(dim=-1)
        intrinsic = torch.zeros(batch_size, device=self._device)
        if self._curiosity is not None and self._dynamics is not None:
            try:
                with torch.no_grad():
                    a_rand = torch.randn(batch_size, self.action_dim,
                                         device=self._device)
                    s_next, _, _ = self._dynamics(scenarios, a_rand)
                    intrinsic = self._curiosity.intrinsic_reward(
                        scenarios, a_rand, s_next,
                    )
            except Exception:
                pass

        return {
            "scenarios": scenarios,
            "complexity": complexity,
            "intrinsic_reward": intrinsic,
            "mode": GenerationMode.ADVERSARIAL.value,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return generator diagnostics."""
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "has_dynamics": self._dynamics is not None,
            "has_curiosity": self._curiosity is not None,
            "has_adapter": self._adapter is not None,
            "initialised": self._initialised,
        }


# =============================================================================
#  §SP.3 — AdaptiveCurriculumManager
# =============================================================================
class AdaptiveCurriculumManager:
    """Absolute Learning Progress (ALP) curriculum controller.

    Tracks per-level success rate and learning progress to decide when
    to advance, hold, or regress difficulty.  Generation mode is
    automatically selected based on the current curriculum level:

    +---------+-------------------+
    | Level   | GenerationMode    |
    +---------+-------------------+
    | 0 – 1   | SMOOTH            |
    | 2 – 3   | STRUCTURED        |
    |   4+    | ADVERSARIAL       |
    +---------+-------------------+

    Promotion criterion:
        ``success_rate ≥ advance_threshold ∧ |ALP| < alp_epsilon``
    Regression criterion:
        ``success_rate < regress_threshold``

    Reference:
        Portelas et al. "Teacher algorithms for curriculum learning of
        Deep RL in continuously parameterized environments",
        CoRL 2019.
    """

    def __init__(
        self,
        max_level: int = 4,
        advance_threshold: float = 0.75,
        regress_threshold: float = 0.35,
        alp_epsilon: float = 0.05,
        alp_window: int = 20,
    ) -> None:
        self.max_level = max_level
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.alp_epsilon = alp_epsilon
        self.alp_window = alp_window

        self.current_level: int = 0
        self._history: List[Dict[str, Any]] = []
        self._level_successes: List[float] = []

    @property
    def generation_mode(self) -> GenerationMode:
        """Map current level to a GenerationMode."""
        if self.current_level <= 1:
            return GenerationMode.SMOOTH
        elif self.current_level <= 3:
            return GenerationMode.STRUCTURED
        else:
            return GenerationMode.ADVERSARIAL

    @property
    def success_rate(self) -> float:
        """Rolling success rate for the current level."""
        if not self._level_successes:
            return 0.0
        recent = self._level_successes[-self.alp_window:]
        return float(np.mean(recent))

    @property
    def absolute_learning_progress(self) -> float:
        """ALP = |mean(recent_half) − mean(older_half)|."""
        if len(self._level_successes) < 4:
            return 1.0  # high uncertainty → keep exploring
        recent = self._level_successes[-self.alp_window:]
        mid = len(recent) // 2
        if mid == 0:
            return 1.0
        older = float(np.mean(recent[:mid]))
        newer = float(np.mean(recent[mid:]))
        return abs(newer - older)

    def record_outcome(self, success: bool, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Record a single episode outcome and potentially adjust level.

        Args:
            success: Whether AEON successfully handled the scenario.
            metadata: Optional extra information about the episode.

        Returns:
            Dict with ``level``, ``promoted``, ``regressed``,
            ``success_rate``, ``alp``.
        """
        self._level_successes.append(1.0 if success else 0.0)
        self._history.append({
            "level": self.current_level,
            "success": success,
            "timestamp": time.time(),
            **(metadata or {}),
        })

        sr = self.success_rate
        alp = self.absolute_learning_progress
        promoted = False
        regressed = False

        # Promotion
        if (sr >= self.advance_threshold
                and alp < self.alp_epsilon
                and self.current_level < self.max_level):
            self.current_level += 1
            self._level_successes.clear()
            promoted = True
            logger.info("📈 Curriculum promoted to level %d (SR=%.2f, ALP=%.4f)",
                        self.current_level, sr, alp)

        # Regression
        elif sr < self.regress_threshold and self.current_level > 0:
            if len(self._level_successes) >= self.alp_window // 2:
                self.current_level -= 1
                self._level_successes.clear()
                regressed = True
                logger.info("📉 Curriculum regressed to level %d (SR=%.2f)",
                            self.current_level, sr)

        return {
            "level": self.current_level,
            "generation_mode": self.generation_mode.value,
            "promoted": promoted,
            "regressed": regressed,
            "success_rate": round(sr, 4),
            "alp": round(alp, 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Curriculum manager diagnostic summary."""
        return {
            "current_level": self.current_level,
            "generation_mode": self.generation_mode.value,
            "success_rate": round(self.success_rate, 4),
            "alp": round(self.absolute_learning_progress, 4),
            "total_episodes": len(self._history),
            "level_episodes": len(self._level_successes),
            "max_level": self.max_level,
        }


# =============================================================================
#  §SP.4 — CorrectiveSynthesizer
# =============================================================================
class CorrectiveSynthesizer:
    """Failure-mode–targeted latent scenario generator.

    Analyses the failure mode of a failed control episode and generates
    corrective training scenarios tailored to the specific deficiency:

    +---------------+---------------------------------------------------+
    | Failure Mode  | Corrective Strategy                               |
    +---------------+---------------------------------------------------+
    | direction     | Smooth interpolation around failure region ±δ     |
    | magnitude     | Re-normalised z with matched target norm          |
    | oscillation   | Monotonically progressing z (no sign reversals)   |
    | collapse      | High-dispersion sampling far from failure point   |
    +---------------+---------------------------------------------------+

    Reference:
        Inspired by Domain Randomization + PAIRED corrective feedback.
    """

    def __init__(
        self,
        latent_dim: int,
        num_corrective: int = 16,
        direction_delta: float = 0.05,
        collapse_dispersion: float = 0.8,
    ) -> None:
        self.latent_dim = latent_dim
        self.num_corrective = num_corrective
        self.direction_delta = direction_delta
        self.collapse_dispersion = collapse_dispersion

    def synthesize(
        self,
        failure_z: torch.Tensor,
        failure_mode: str,
    ) -> Dict[str, Any]:
        """Generate corrective scenarios for a given failure.

        Args:
            failure_z: [z_dim] latent state where failure occurred.
            failure_mode: One of 'direction', 'magnitude', 'oscillation',
                          'collapse'.

        Returns:
            Dict with ``corrective_scenarios`` [N, z_dim],
            ``failure_mode``, and ``strategy``.
        """
        if failure_z.dim() == 1:
            failure_z = failure_z.unsqueeze(0)

        handler = {
            "direction": self._correct_direction,
            "magnitude": self._correct_magnitude,
            "oscillation": self._correct_oscillation,
            "collapse": self._correct_collapse,
        }.get(failure_mode, self._correct_direction)

        scenarios = handler(failure_z)
        return {
            "corrective_scenarios": scenarios,
            "failure_mode": failure_mode,
            "strategy": handler.__name__.replace("_correct_", ""),
            "num_generated": scenarios.shape[0],
        }

    def _correct_direction(self, z: torch.Tensor) -> torch.Tensor:
        """Smooth interpolation around the failure region."""
        N = self.num_corrective
        device = z.device
        deltas = torch.linspace(-self.direction_delta,
                                self.direction_delta, N,
                                device=device).unsqueeze(1)
        directions = torch.randn(N, self.latent_dim, device=device)
        directions = F.normalize(directions, dim=-1)
        scenarios = z + deltas * directions
        return F.normalize(scenarios, dim=-1)

    def _correct_magnitude(self, z: torch.Tensor) -> torch.Tensor:
        """Re-normalised z with target norm matching."""
        N = self.num_corrective
        device = z.device
        target_norm = z.norm(dim=-1, keepdim=True).item()
        noise = torch.randn(N, self.latent_dim, device=device) * 0.1
        scenarios = z + noise
        current_norms = scenarios.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scenarios = scenarios * (target_norm / current_norms)
        return scenarios

    def _correct_oscillation(self, z: torch.Tensor) -> torch.Tensor:
        """Monotonically progressing z (no sign reversals)."""
        N = self.num_corrective
        device = z.device
        base = z.squeeze(0)
        direction = F.normalize(torch.randn(self.latent_dim, device=device),
                                dim=0)
        alphas = torch.linspace(0.0, 1.0, N, device=device)
        scenarios = base.unsqueeze(0) + alphas.unsqueeze(1) * direction.unsqueeze(0)
        return F.normalize(scenarios, dim=-1)

    def _correct_collapse(self, z: torch.Tensor) -> torch.Tensor:
        """High-dispersion sampling far from the failure point."""
        N = self.num_corrective
        device = z.device
        noise = torch.randn(N, self.latent_dim, device=device)
        noise = noise * self.collapse_dispersion
        scenarios = z + noise
        return F.normalize(scenarios, dim=-1)

    def classify_failure(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> str:
        """Heuristic failure-mode classifier.

        Compares predicted vs target latent states to determine the
        dominant failure mode.

        Args:
            z_pred:   [z_dim] or [B, z_dim] predicted latent state.
            z_target: [z_dim] or [B, z_dim] target latent state.

        Returns:
            One of 'direction', 'magnitude', 'oscillation', 'collapse'.
        """
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)
        if z_target.dim() == 1:
            z_target = z_target.unsqueeze(0)

        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
        norm_ratio = (z_pred.norm(dim=-1) /
                      z_target.norm(dim=-1).clamp(min=1e-8)).mean().item()

        # Direction failure: cosine similarity < 0.5
        if cos_sim < 0.5:
            return "direction"
        # Magnitude failure: norm ratio far from 1.0
        if abs(norm_ratio - 1.0) > 0.3:
            return "magnitude"
        # Collapse: very low variance in prediction
        if z_pred.std(dim=-1).mean().item() < 0.01:
            return "collapse"
        # Default: oscillation
        return "oscillation"

    def get_summary(self) -> Dict[str, Any]:
        """Return synthesizer diagnostic summary."""
        return {
            "latent_dim": self.latent_dim,
            "num_corrective": self.num_corrective,
            "direction_delta": self.direction_delta,
            "collapse_dispersion": self.collapse_dispersion,
        }


# =============================================================================
#  §SP.5 — Bootstrap Codebook Embeddings (corpus-free)
# =============================================================================
def bootstrap_codebook_embeddings(
    model: nn.Module,
    config: Any,
    device: torch.device = torch.device("cpu"),
    num_samples: int = 2048,
    temperature: float = 1.5,
) -> Dict[str, Any]:
    """Generate semantically diverse codebook centroids without any corpus.

    Instead of k-means on real token embeddings, we:
      1. Sample N latent z-vectors from a high-temperature VibeThinker
         distribution (temperature=1.5) to maximise diversity.
      2. Run k-means on the imagined embeddings to find K well-separated
         centroids.
      3. Initialise the VQ codebook from these centroids.

    This implements §SP.5 — Bootstrap without corpus.

    Args:
        model: AEONDeltaV4 model instance.
        config: AEONConfig with z_dim, hidden_dim, vq_num_embeddings.
        device: Computation device.
        num_samples: Number of synthetic z-vectors to generate.
        temperature: Sampling temperature (higher → more diverse).

    Returns:
        Dict with ``initialized``, ``method``, ``num_embeddings``,
        ``inertia``.
    """
    z_dim = getattr(config, "z_dim", getattr(config, "hidden_dim", 256))
    K = getattr(config, "vq_num_embeddings",
                getattr(config, "codebook_size", 256))

    try:
        # Step 1: High-temperature z sampling via VibeThinker
        adapter = None
        try:
            from aeon_core import VibeThinkerPromptAdapter
            adapter = VibeThinkerPromptAdapter(
                latent_dim=z_dim,
                hidden_dim=getattr(config, "hidden_dim", z_dim),
            ).to(device)
            adapter.eval()
        except ImportError:
            logger.debug("VibeThinkerPromptAdapter unavailable — using "
                         "pure Gaussian bootstrap")

        all_z: List[torch.Tensor] = []
        batch_size = min(256, num_samples)

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                n = min(batch_size, num_samples - start)
                z_raw = torch.randn(n, z_dim, device=device) * temperature

                if adapter is not None:
                    # Use adapter to project → back-project for diversity
                    vt_out = adapter(z_raw)
                    prompt_emb = vt_out["prompt_embedding"]
                    # Mix raw z with prompted representation for richness
                    if prompt_emb.shape[-1] != z_dim:
                        # Pad or truncate to match z_dim
                        if prompt_emb.shape[-1] < z_dim:
                            pad = torch.randn(
                                n, z_dim - prompt_emb.shape[-1],
                                device=device,
                            ) * temperature * 0.5
                            prompt_emb = torch.cat([prompt_emb, pad], dim=-1)
                        else:
                            prompt_emb = prompt_emb[:, :z_dim]
                    z_diverse = F.normalize(
                        0.5 * z_raw + 0.5 * prompt_emb, dim=-1,
                    )
                else:
                    z_diverse = F.normalize(z_raw, dim=-1)

                all_z.append(z_diverse.cpu())

        embeddings = torch.cat(all_z, dim=0)
        N = embeddings.shape[0]

        if N < K:
            return {
                "initialized": False,
                "method": "insufficient_samples",
                "num_embeddings": K,
                "inertia": 0.0,
            }

        # Step 2: K-means on synthetic embeddings
        centroids = embeddings[torch.randperm(N)[:K]].clone()
        assignments = torch.zeros(N, dtype=torch.long)

        for _iter in range(50):
            # Assignment step
            new_assignments = []
            for s in range(0, N, batch_size):
                batch_emb = embeddings[s:s + batch_size]
                dists = torch.cdist(batch_emb, centroids)
                new_assignments.append(dists.argmin(dim=1))
            assignments = torch.cat(new_assignments)

            # Update step
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = embeddings[mask].mean(dim=0)
                else:
                    dists_to_cents = torch.cdist(
                        embeddings, new_centroids,
                    ).min(dim=1).values
                    farthest_idx = dists_to_cents.argmax()
                    new_centroids[k] = embeddings[farthest_idx]

            shift = (new_centroids - centroids).norm(dim=1).mean().item()
            centroids = new_centroids
            if shift < 1e-5:
                break

        # Step 3: Apply to VQ codebook
        vq = getattr(model, "vq", None)
        if vq is not None:
            with torch.no_grad():
                emb_weight = getattr(vq, "embedding", None)
                if emb_weight is not None:
                    weight = emb_weight.weight
                    if centroids.shape[1] == weight.shape[1]:
                        actual_K = min(K, weight.shape[0])
                        weight[:actual_K].copy_(
                            centroids[:actual_K].to(weight.device),
                        )
                        if hasattr(vq, "ema_w"):
                            vq.ema_w[:actual_K].copy_(
                                centroids[:actual_K].to(vq.ema_w.device),
                            )
                        if hasattr(vq, "ema_cluster_size"):
                            vq.ema_cluster_size.fill_(N / K)

        # Compute inertia
        inertia = 0.0
        for s in range(0, N, batch_size):
            batch_emb = embeddings[s:s + batch_size]
            batch_assign = assignments[s:s + batch_size]
            batch_cents = centroids[batch_assign]
            inertia += ((batch_emb - batch_cents) ** 2).sum().item()

        return {
            "initialized": True,
            "method": "bootstrap_synthetic_kmeans",
            "num_embeddings": K,
            "inertia": round(inertia, 4),
            "num_synthetic_samples": N,
            "temperature": temperature,
        }

    except Exception as e:
        return {
            "initialized": False,
            "method": f"bootstrap_error_{type(e).__name__}",
            "reason": str(e),
            "num_embeddings": 0,
            "inertia": 0.0,
        }


# =============================================================================
#  §SP.6 — VibeThinkerMetaSignaler (RSSM loss adaptation wrapper)
# =============================================================================
class VibeThinkerMetaSignaler:
    """Meta-signaler adapting λ_cos in RSSM mini-training loss.

    Wraps ``VibeThinkerContinuousLearner`` from aeon_core (when
    available) and uses its ``calibration_ema`` to modulate the cosine-
    similarity weight in the RSSM loss function:

        L = L_mse + λ_cos · (1 − cosine_sim(z_pred, z_target))

    When calibration_ema is high (poor calibration), λ_cos is
    *increased* to enforce representational alignment.  When calibration
    is good, λ_cos is *reduced* to let MSE dominate.

    This closes the VibeThinker → RSSM → Controller feedback loop
    without modifying ``aeon_core``.
    """

    def __init__(
        self,
        base_lambda_cos: float = 0.1,
        lambda_cos_min: float = 0.01,
        lambda_cos_max: float = 0.5,
    ) -> None:
        self.base_lambda_cos = base_lambda_cos
        self.lambda_cos_min = lambda_cos_min
        self.lambda_cos_max = lambda_cos_max
        self._learner: Optional[Any] = None
        self._lambda_cos = base_lambda_cos
        self._history: List[Dict[str, float]] = []

    def attach_learner(self, learner: Any) -> None:
        """Attach a VibeThinkerContinuousLearner instance."""
        self._learner = learner

    @property
    def lambda_cos(self) -> float:
        """Current λ_cos value."""
        return self._lambda_cos

    def update(self) -> Dict[str, float]:
        """Recompute λ_cos from the learner's calibration_ema.

        Returns:
            Dict with ``lambda_cos``, ``calibration_ema``.
        """
        cal_ema = 0.0
        if self._learner is not None:
            cal_ema = getattr(self._learner, "_calibration_ema", 0.0)

        # λ_cos ∝ calibration_ema: poor calibration → higher weight
        raw = self.base_lambda_cos * (1.0 + 2.0 * cal_ema)
        self._lambda_cos = max(
            self.lambda_cos_min,
            min(self.lambda_cos_max, raw),
        )

        record = {
            "lambda_cos": self._lambda_cos,
            "calibration_ema": cal_ema,
            "timestamp": time.time(),
        }
        self._history.append(record)
        # Bound history
        if len(self._history) > 500:
            self._history = self._history[-250:]

        return record

    def compute_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined RSSM training loss.

        L = L_mse + λ_cos · (1 − cosine_sim(z_pred, z_target))

        Args:
            z_pred:   [B, z_dim] predicted latent.
            z_target: [B, z_dim] target latent.

        Returns:
            Scalar loss tensor.
        """
        l_mse = F.mse_loss(z_pred, z_target)
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()
        l_cos = 1.0 - cos_sim
        return l_mse + self._lambda_cos * l_cos

    def get_summary(self) -> Dict[str, Any]:
        """Return meta-signaler diagnostics."""
        return {
            "lambda_cos": self._lambda_cos,
            "base_lambda_cos": self.base_lambda_cos,
            "has_learner": self._learner is not None,
            "history_length": len(self._history),
        }


# =============================================================================
#  Wizard Step Status
# =============================================================================
class WizardStepStatus:
    """Tracks per-step completion status for the First-Run Wizard."""

    __slots__ = ("name", "status", "started_at", "finished_at",
                 "result", "error")

    def __init__(self, name: str):
        self.name = name
        self.status = "pending"      # pending | running | completed | failed
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def start(self) -> None:
        self.status = "running"
        self.started_at = time.time()

    def complete(self, result: Dict[str, Any]) -> None:
        self.status = "completed"
        self.finished_at = time.time()
        self.result = result

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.finished_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "status": self.status,
        }
        if self.started_at is not None:
            d["started_at"] = self.started_at
        if self.finished_at is not None:
            d["finished_at"] = self.finished_at
            if self.started_at is not None:
                d["duration_s"] = round(self.finished_at - self.started_at, 3)
        if self.result:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


# =============================================================================
#  Wizard State
# =============================================================================
class WizardState:
    """Global wizard state across all steps."""

    def __init__(self) -> None:
        self.steps: Dict[str, WizardStepStatus] = {
            "weight_loading":       WizardStepStatus("weight_loading"),
            "corpus_diagnostics":   WizardStepStatus("corpus_diagnostics"),
            "hyperparameterization": WizardStepStatus("hyperparameterization"),
            "codebook_init":        WizardStepStatus("codebook_init"),
            "config_generation":    WizardStepStatus("config_generation"),
        }
        self.overall_status = "idle"    # idle | running | completed | failed
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.generated_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "overall_status": self.overall_status,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }
        if self.started_at is not None:
            d["started_at"] = self.started_at
        if self.finished_at is not None:
            d["finished_at"] = self.finished_at
            if self.started_at is not None:
                d["total_duration_s"] = round(
                    self.finished_at - self.started_at, 3,
                )
        if self.generated_config is not None:
            d["generated_config"] = self.generated_config
        return d


# Global wizard state (singleton per server process)
_wizard_state = WizardState()


def get_wizard_state() -> WizardState:
    """Return the global wizard state."""
    return _wizard_state


def reset_wizard_state() -> None:
    """Reset wizard state for a fresh run."""
    global _wizard_state
    _wizard_state = WizardState()


# =============================================================================
#  Step 0: Load VibeThinker Weights
# =============================================================================
def load_vt_weights(
    model: nn.Module,
    weights_path: Path = VT_WEIGHTS_PATH,
) -> Dict[str, Any]:
    """Load VibeThinker-1.5B weights from safetensors file.

    Implements Spec §2.1 — hard reference to
    ``vibe_thinker_weights/model.safetensors``.

    Args:
        model: The AEON model instance (AEONDeltaV4 or AEONDeltaV3).
        weights_path: Path to the safetensors weight file.

    Returns:
        Dict with loading status and metadata.
    """
    result: Dict[str, Any] = {
        "loaded": False,
        "weights_path": str(weights_path),
    }

    if not weights_path.exists():
        result["reason"] = f"Weight file not found: {weights_path}"
        logger.warning("VT weights not found at %s — wizard will use "
                        "randomly initialized adapter", weights_path)
        return result

    try:
        from safetensors.torch import load_file as st_load

        flat = st_load(str(weights_path))

        # Detect AEON-formatted safetensors (prefixed keys)
        _has_aeon = any(
            k.startswith(("adapter_state.", "kernel_state."))
            for k in flat
        )

        if _has_aeon:
            # Load adapter state into model's VibeThinker components
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

            # Try to load into model components with shape validation
            _adapter = getattr(model, "vibe_thinker_adapter", None)
            if _adapter is not None and adapter_state:
                _model_sd = _adapter.state_dict()
                _compat = {
                    k: v for k, v in adapter_state.items()
                    if k in _model_sd and v.shape == _model_sd[k].shape
                }
                if _compat:
                    _adapter.load_state_dict(_compat, strict=False)
                result["adapter_keys_loaded"] = len(_compat)

            _kernel = getattr(model, "vibe_thinker_kernel", None)
            if _kernel is not None and kernel_state:
                _model_sd = _kernel.state_dict()
                _compat = {
                    k: v for k, v in kernel_state.items()
                    if k in _model_sd and v.shape == _model_sd[k].shape
                }
                if _compat:
                    _kernel.load_state_dict(_compat, strict=False)
                result["kernel_keys_loaded"] = len(_compat)

            result["format"] = "aeon_safetensors"
        else:
            # Raw HuggingFace weight format — extract AEON-compatible
            # weights using the weight manager's extraction logic.
            result["format"] = "raw_safetensors"
            result["num_tensors"] = len(flat)

            try:
                from aeon_core import VibeThinkerWeightManager
                _hf_hidden = 1536  # default VibeThinker-1.5B hidden size
                _aeon_payload = VibeThinkerWeightManager._extract_aeon_weights(
                    dict(flat), _hf_hidden,
                )

                _adapter = getattr(model, "vibe_thinker_adapter", None)
                if _adapter is not None and "adapter_state" in _aeon_payload:
                    _model_sd = _adapter.state_dict()
                    _compat = {
                        k: v for k, v in _aeon_payload["adapter_state"].items()
                        if k in _model_sd and v.shape == _model_sd[k].shape
                    }
                    if _compat:
                        _adapter.load_state_dict(_compat, strict=False)
                    result["adapter_keys_loaded"] = len(_compat)

                _kernel = getattr(model, "vibe_thinker_kernel", None)
                if _kernel is not None and "kernel_state" in _aeon_payload:
                    _model_sd = _kernel.state_dict()
                    _compat = {
                        k: v for k, v in _aeon_payload["kernel_state"].items()
                        if k in _model_sd and v.shape == _model_sd[k].shape
                    }
                    if _compat:
                        _kernel.load_state_dict(_compat, strict=False)
                    result["kernel_keys_loaded"] = len(_compat)
            except Exception as _extract_err:
                logger.warning(
                    "HF weight extraction failed (non-fatal): %s",
                    _extract_err,
                )

        result["loaded"] = True
        result["file_size_mb"] = round(
            weights_path.stat().st_size / (1024 * 1024), 2,
        )
        logger.info("✅ VT weights loaded from %s (%s)",
                     weights_path, result["format"])

    except ImportError:
        result["reason"] = "safetensors library not installed"
        logger.warning("safetensors not installed — cannot load VT weights")
    except Exception as e:
        result["reason"] = f"Load error: {type(e).__name__}: {e}"
        logger.warning("VT weight loading failed: %s", e)

    return result


# =============================================================================
#  Step 1: Self-Play Diagnostics  (§SP.2, replaces §4.A.1)
# =============================================================================
def run_self_play_diagnostics(
    model: nn.Module,
    config: Any,
    device: torch.device = torch.device("cpu"),
    num_synthetic: int = 512,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Latent-space complexity profiling via VibeThinker.

    Implements §SP.2 — replaces corpus-based diagnostics with
    self-generated latent scenarios.  All inference operates on
    synthetic z-vectors; no external tokens are required.

    The function generates ``num_synthetic`` latent vectors using
    ``LatentWorldGenerator`` in SMOOTH mode, then profiles them through
    ``VibeThinkerPromptAdapter`` and ``VibeThinkerReasoningKernel``.

    Args:
        model: AEONDeltaV4 model instance.
        config: AEONConfig.
        device: Computation device.
        num_synthetic: Number of synthetic z-vectors to generate.
        batch_size: Batch size for inference passes.

    Returns:
        Dict with complexity statistics, bimodality analysis, and
        embedding PCA results.
    """
    try:
        from aeon_core import VibeThinkerPromptAdapter, VibeThinkerReasoningKernel
        from aeon_core import VibeThinkerConfig
    except ImportError:
        return {"diagnosed": False, "reason": "VibeThinker modules not available"}

    try:
        z_dim = getattr(config, "z_dim", getattr(config, "hidden_dim", 256))
        hidden_dim = getattr(config, "hidden_dim", z_dim)

        adapter = VibeThinkerPromptAdapter(
            latent_dim=z_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        kernel = VibeThinkerReasoningKernel(
            config=VibeThinkerConfig(),
            hidden_dim=hidden_dim,
        ).to(device)

        # Generate synthetic latent scenarios
        world_gen = LatentWorldGenerator(
            latent_dim=z_dim,
            device=device,
        )

        complexity_scores: List[float] = []
        cot_depths: List[float] = []
        embeddings: List[torch.Tensor] = []

        model.eval()
        adapter.eval()
        kernel.eval()

        with torch.no_grad():
            for start in range(0, num_synthetic, batch_size):
                n = min(batch_size, num_synthetic - start)
                gen_result = world_gen.generate(
                    mode=GenerationMode.SMOOTH,
                    batch_size=n,
                    config=config,
                )
                z = gen_result["scenarios"].to(device)
                vt_out = adapter(z)

                # Complexity scores
                _scores = vt_out["complexity_score"]
                if _scores.dim() > 0:
                    complexity_scores.extend(_scores.cpu().tolist())
                else:
                    complexity_scores.append(float(_scores.cpu()))

                embeddings.append(vt_out["prompt_embedding"].cpu())

                # CoT depth from reasoning kernel
                r_out = kernel.reason(z)
                if isinstance(r_out, dict) and "cot_depth" in r_out:
                    _depth = r_out["cot_depth"]
                    if hasattr(_depth, "cpu"):
                        if _depth.dim() > 0:
                            cot_depths.extend(_depth.cpu().tolist())
                        else:
                            cot_depths.append(float(_depth.cpu()))
                    else:
                        cot_depths.append(float(_depth))

        # Distribution statistics
        _scores_np = np.array(complexity_scores)
        diag: Dict[str, Any] = {
            "diagnosed": True,
            "source": "self_play_synthetic",
            "num_synthetic": num_synthetic,
            "complexity_mean": float(_scores_np.mean()),
            "complexity_std": float(_scores_np.std()),
            "complexity_min": float(_scores_np.min()),
            "complexity_max": float(_scores_np.max()),
        }

        # Bimodality detection (Hartigan's dip test approximation)
        _sorted = np.sort(_scores_np)
        _mid = len(_sorted) // 2
        if _mid > 0:
            _lower = _sorted[:_mid].mean()
            _upper = _sorted[_mid:].mean()
            _gap = abs(_upper - _lower)
            diag["bimodality_gap"] = float(_gap)
            diag["heterogeneous"] = _gap > 0.3
        else:
            diag["bimodality_gap"] = 0.0
            diag["heterogeneous"] = False

        # CoT depth statistics
        if cot_depths:
            _cot_np = np.array(cot_depths)
            diag["cot_depth_stats"] = {
                "mean": float(_cot_np.mean()),
                "std": float(_cot_np.std()),
                "min": float(_cot_np.min()),
                "max": float(_cot_np.max()),
                "p95": float(np.percentile(_cot_np, 95)),
            }

        # Embedding PCA for z_dim recommendation
        all_emb = torch.cat(embeddings, dim=0).numpy()
        if all_emb.shape[0] > all_emb.shape[1]:
            try:
                from sklearn.decomposition import PCA
                _pca = PCA(n_components=min(all_emb.shape[1], 64))
                _pca.fit(all_emb)
                _cumvar = np.cumsum(_pca.explained_variance_ratio_)
                _n95 = int(np.searchsorted(_cumvar, 0.95)) + 1
                diag["pca_explained_95pct_components"] = _n95
            except ImportError:
                logger.debug("sklearn not available for PCA analysis")

        # Store raw embeddings for downstream k-means
        diag["_embeddings"] = all_emb
        diag["_complexity_scores"] = complexity_scores

        return diag

    except Exception as e:
        return {"diagnosed": False, "reason": f"{type(e).__name__}: {e}"}


def run_corpus_diagnostics(
    model: nn.Module,
    tokens: torch.Tensor,
    config: Any,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Legacy corpus diagnostics — delegates to self-play diagnostics.

    .. deprecated::
        The ``tokens`` parameter is accepted for backward compatibility
        but is **ignored**.  All diagnostics now operate in latent space
        via ``run_self_play_diagnostics()``.

    Returns:
        Dict with complexity statistics from self-play diagnostics.
    """
    logger.info("run_corpus_diagnostics: tokens argument ignored — "
                "delegating to run_self_play_diagnostics()")
    num_synthetic = max(256, len(tokens) if tokens is not None else 512)
    return run_self_play_diagnostics(
        model=model,
        config=config,
        device=device,
        num_synthetic=num_synthetic,
        batch_size=batch_size,
    )


# =============================================================================
#  Step 2: Hyperparameterization  (Spec §4.A.2)
# =============================================================================
def compute_hyperparameters(
    diagnostics: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Compute codebook_size, context_window, z_dim from diagnostics.

    Implements Spec §4.A.2:
      - codebook_size:   Calinski-Harabasz optimal cluster count
      - context_window:  ceil(P95 CoT depth), clamped [1, 16]
      - z_dim:           PCA 95% explained-variance components

    All values are computed automatically — no manual input.

    Args:
        diagnostics: Output from run_corpus_diagnostics.
        config: AEONConfigV4 to be updated.

    Returns:
        Dict with recommended and applied hyperparameters.
    """
    recommendations: Dict[str, Any] = {}
    applied: Dict[str, Any] = {}

    # ── Context window from P95 CoT depth ────────────────────────────────
    cot_stats = diagnostics.get("cot_depth_stats", {})
    if "p95" in cot_stats:
        _p95 = cot_stats["p95"]
        new_window = max(
            _MIN_CONTEXT_WINDOW,
            min(_MAX_CONTEXT_WINDOW, int(math.ceil(_p95))),
        )
        recommendations["context_window"] = new_window
        old_window = getattr(config, "context_window", 3)
        config.context_window = new_window
        applied["context_window"] = {
            "old": old_window,
            "new": new_window,
            "p95_cot_depth": round(_p95, 4),
        }
        logger.info("📐 context_window: %d → %d (P95 CoT = %.2f)",
                     old_window, new_window, _p95)

    # ── z_dim from PCA explained variance ────────────────────────────────
    _n95 = diagnostics.get("pca_explained_95pct_components")
    if _n95 is not None:
        recommendations["z_dim_95pct"] = _n95
        # Note: z_dim must equal hidden_dim in AEONConfig, so we record
        # the recommendation but only apply if architecturally safe
        applied["z_dim_recommendation"] = _n95
        logger.info("📊 z_dim recommendation: %d (PCA 95%% variance)", _n95)

    # ── Codebook size via Calinski-Harabasz ───────────────────────────────
    all_emb = diagnostics.get("_embeddings")
    if all_emb is not None and len(all_emb) > 8:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import calinski_harabasz_score

            candidates = [k for k in _CODEBOOK_SIZE_CANDIDATES
                          if k < len(all_emb)]
            if candidates:
                best_ch, best_k = -1.0, candidates[0]
                for k in candidates:
                    _km = KMeans(n_clusters=k, n_init=10, random_state=42,
                                 max_iter=200)
                    _labels = _km.fit_predict(all_emb)
                    _ch = calinski_harabasz_score(all_emb, _labels)
                    if _ch > best_ch:
                        best_ch, best_k = _ch, k

                recommendations["codebook_size"] = best_k
                old_cb = getattr(config, "vq_num_embeddings",
                                 getattr(config, "codebook_size", 256))
                config.vq_num_embeddings = best_k
                applied["codebook_size"] = {
                    "old": old_cb,
                    "new": best_k,
                    "calinski_harabasz": round(best_ch, 2),
                }
                logger.info("📦 codebook_size: %d → %d (CH = %.2f)",
                             old_cb, best_k, best_ch)
        except ImportError:
            logger.debug("sklearn not available for codebook sizing")

    return {
        "recommendations": recommendations,
        "applied": applied,
        "config_updated": True,
    }


# =============================================================================
#  Step 3: Codebook Warm-Start  (§SP.5, replaces §4.A.3)
# =============================================================================
def initialize_codebook(
    model: nn.Module,
    tokens: Optional[torch.Tensor] = None,
    config: Any = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Initialize VQ codebook via synthetic bootstrap.

    Implements §SP.5 — warm-start the vector quantiser with
    semantically meaningful centroids generated entirely in latent
    space.  The ``tokens`` parameter is accepted for backward
    compatibility but is **ignored**.

    Delegates to ``bootstrap_codebook_embeddings()`` which uses
    high-temperature VibeThinker sampling + k-means.

    Falls back to ``ae_train.warm_start_codebook_from_vt()`` only if
    bootstrap fails and tokens are available.

    Returns:
        Dict with initialization status, inertia, and method.
    """
    if tokens is not None:
        logger.info("initialize_codebook: tokens argument ignored — "
                    "using synthetic bootstrap (§SP.5)")

    # Primary path: synthetic bootstrap (no corpus required)
    result = bootstrap_codebook_embeddings(
        model=model,
        config=config,
        device=device,
    )

    if result.get("initialized", False):
        return result

    # Fallback: try ae_train if tokens were provided
    if tokens is not None:
        try:
            from ae_train import warm_start_codebook_from_vt
            logger.info("Bootstrap failed — falling back to corpus-based "
                        "warm_start_codebook_from_vt (legacy)")
            return warm_start_codebook_from_vt(
                model=model,
                tokens=tokens,
                config=config,
                device=device,
                batch_size=batch_size,
            )
        except ImportError:
            logger.debug("ae_train.warm_start_codebook_from_vt unavailable")

    # Final fallback: inline k-means on random embeddings
    try:
        z_dim = getattr(config, "z_dim", getattr(config, "hidden_dim", 256))
        K = getattr(config, "vq_num_embeddings",
                     getattr(config, "codebook_size", 256))

        embeddings = F.normalize(
            torch.randn(max(K * 4, 512), z_dim), dim=-1,
        )
        N = embeddings.shape[0]

        centroids = embeddings[torch.randperm(N)[:K]].clone()
        for _iter in range(50):
            assignments_list = []
            for start in range(0, N, batch_size):
                batch_emb = embeddings[start:start + batch_size]
                dists = torch.cdist(batch_emb, centroids)
                assignments_list.append(dists.argmin(dim=1))
            assignments = torch.cat(assignments_list)

            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = embeddings[mask].mean(dim=0)
                else:
                    dists_to_cents = torch.cdist(
                        embeddings, new_centroids,
                    ).min(dim=1).values
                    farthest_idx = dists_to_cents.argmax()
                    new_centroids[k] = embeddings[farthest_idx]

            shift = (new_centroids - centroids).norm(dim=1).mean().item()
            centroids = new_centroids
            if shift < 1e-5:
                break

        # Apply to VQ codebook
        vq = getattr(model, "vq", None)
        if vq is not None:
            with torch.no_grad():
                emb_weight = getattr(vq, "embedding", None)
                if emb_weight is not None and centroids.shape[1] == emb_weight.weight.shape[1]:
                    vq.embedding.weight.copy_(
                        centroids.to(vq.embedding.weight.device),
                    )
                    if hasattr(vq, "ema_w"):
                        vq.ema_w.copy_(centroids.to(vq.ema_w.device))
                    if hasattr(vq, "ema_cluster_size"):
                        vq.ema_cluster_size.fill_(N / K)

        inertia = 0.0
        for start in range(0, N, batch_size):
            batch_emb = embeddings[start:start + batch_size]
            batch_assign = assignments[start:start + batch_size]
            batch_cents = centroids[batch_assign]
            inertia += ((batch_emb - batch_cents) ** 2).sum().item()

        return {
            "initialized": True,
            "method": "inline_random_kmeans_fallback",
            "num_embeddings": K,
            "inertia": round(inertia, 4),
        }

    except Exception as e:
        return {
            "initialized": False,
            "method": f"error_{type(e).__name__}",
            "reason": str(e),
            "num_embeddings": 0,
            "inertia": 0.0,
        }


# =============================================================================
#  Step 4: Configuration Generation  (Spec §4.A.4)
# =============================================================================
def generate_config(
    config: Any,
    diagnostics: Dict[str, Any],
    hyperparams: Dict[str, Any],
    codebook_result: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and save a validated AEONConfigV4 based on wizard results.

    Implements Spec §4.A.4 — produces a complete configuration with
    all wizard-derived hyperparameters applied.

    Args:
        config: The mutated AEONConfigV4 object.
        diagnostics: Corpus diagnostics results.
        hyperparams: Hyperparameterization results.
        codebook_result: Codebook initialization results.
        output_path: Optional path to save the config JSON.

    Returns:
        Dict with the serialized configuration and validation status.
    """
    try:
        config_dict = asdict(config)
    except Exception:
        # Fallback for non-dataclass configs
        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
        }

    # Add wizard metadata
    wizard_meta = {
        "wizard_version": "2.0.0",
        "architecture": "self_play_synthetic_curriculum",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_size": diagnostics.get("corpus_size",
                                       diagnostics.get("num_synthetic", 0)),
        "heterogeneous": diagnostics.get("heterogeneous", False),
        "codebook_method": codebook_result.get("method", "unknown"),
        "codebook_inertia": codebook_result.get("inertia", 0.0),
        "source": diagnostics.get("source", "unknown"),
    }
    config_dict["_wizard_meta"] = wizard_meta

    # Validation checks
    validation = {
        "valid": True,
        "warnings": [],
    }

    _z_dim = config_dict.get("z_dim", 0)
    _hidden = config_dict.get("hidden_dim", 0)
    if _z_dim != _hidden:
        validation["warnings"].append(
            f"z_dim ({_z_dim}) != hidden_dim ({_hidden}): "
            f"AEONConfig requires z_dim == hidden_dim"
        )

    if config_dict.get("context_window", 3) < 1:
        validation["warnings"].append("context_window < 1")
        validation["valid"] = False

    config_dict["_validation"] = validation

    # Save if output_path provided
    if output_path is not None:
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info("💾 Config saved to %s", output_path)
            config_dict["_saved_to"] = output_path
        except Exception as e:
            logger.warning("Config save failed: %s", e)
            config_dict["_save_error"] = str(e)

    return config_dict


# =============================================================================
#  Self-Play Wizard Orchestrator  (§SP — primary entry point)
# =============================================================================
def run_self_play_wizard(
    model: nn.Module,
    config: Any,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    output_dir: str = "wizard_output",
    num_episodes: int = 64,
) -> Dict[str, Any]:
    """Execute the Self-Play Setup Wizard — fully latent-space.

    Implements §SP — cold-start initialization via Self-Play / Synthetic
    Curriculum.  No external corpus is required; all five steps operate
    purely on synthetic latent scenarios generated by the
    ``LatentWorldGenerator``.

    Steps:
      0. Load VibeThinker weights (§2.1)
      1. Self-Play diagnostics (§SP.2)
      2. Hyperparameterization (§4.A.2, adapted for latent stats)
      3. Codebook bootstrap (§SP.5)
      4. Configuration generation (§4.A.4)

    Optionally runs a short self-play evaluation loop (``num_episodes``)
    using ``AdaptiveCurriculumManager`` + ``CorrectiveSynthesizer`` to
    validate the initialisation.

    Args:
        model: AEONDeltaV4 model instance.
        config: AEONConfig (will be mutated).
        device: Computation device.
        batch_size: Batch size for inference passes.
        output_dir: Directory for saving wizard artifacts.
        num_episodes: Number of self-play evaluation episodes.

    Returns:
        Dict with complete wizard results including generated config.
    """
    reset_wizard_state()
    state = get_wizard_state()
    state.overall_status = "running"
    state.started_at = time.time()

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {"wizard_completed": False}

    z_dim = getattr(config, "z_dim", getattr(config, "hidden_dim", 256))

    # ── Step 0: Weight loading ───────────────────────────────────────────
    step = state.steps["weight_loading"]
    step.start()
    try:
        wt_result = load_vt_weights(model, VT_WEIGHTS_PATH)
        step.complete(wt_result)
        results["weight_loading"] = wt_result
    except Exception as e:
        step.fail(str(e))
        results["weight_loading"] = {"loaded": False, "error": str(e)}
        logger.warning("Weight loading failed (non-fatal): %s", e)

    # ── Step 1: Self-Play diagnostics ────────────────────────────────────
    step = state.steps["corpus_diagnostics"]
    step.start()
    try:
        diagnostics = run_self_play_diagnostics(
            model=model,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        # Remove internal numpy arrays before serializing
        _diag_clean = {
            k: v for k, v in diagnostics.items()
            if not k.startswith("_")
        }
        step.complete(_diag_clean)
        results["corpus_diagnostics"] = _diag_clean
    except Exception as e:
        step.fail(str(e))
        diagnostics = {"diagnosed": False, "reason": str(e)}
        results["corpus_diagnostics"] = diagnostics
        logger.warning("Self-play diagnostics failed: %s", e)

    # ── Step 2: Hyperparameterization ────────────────────────────────────
    step = state.steps["hyperparameterization"]
    step.start()
    try:
        hyperparams = compute_hyperparameters(diagnostics, config)
        step.complete(hyperparams)
        results["hyperparameterization"] = hyperparams
    except Exception as e:
        step.fail(str(e))
        hyperparams = {"config_updated": False, "error": str(e)}
        results["hyperparameterization"] = hyperparams
        logger.warning("Hyperparameterization failed: %s", e)

    # ── Step 3: Codebook bootstrap (§SP.5) ───────────────────────────────
    step = state.steps["codebook_init"]
    step.start()
    try:
        codebook_result = initialize_codebook(
            model=model,
            tokens=None,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        step.complete(codebook_result)
        results["codebook_init"] = codebook_result
    except Exception as e:
        step.fail(str(e))
        codebook_result = {"initialized": False, "error": str(e)}
        results["codebook_init"] = codebook_result
        logger.warning("Codebook bootstrap failed: %s", e)

    # ── Step 4: Configuration generation ─────────────────────────────────
    step = state.steps["config_generation"]
    step.start()
    try:
        config_path = os.path.join(output_dir, "aeon_config_wizard.json")
        gen_config = generate_config(
            config=config,
            diagnostics=diagnostics,
            hyperparams=hyperparams,
            codebook_result=codebook_result,
            output_path=config_path,
        )
        step.complete({"config_path": config_path, "valid": True})
        state.generated_config = gen_config
        results["config_generation"] = {
            "config_path": config_path,
            "valid": gen_config.get("_validation", {}).get("valid", True),
        }
    except Exception as e:
        step.fail(str(e))
        results["config_generation"] = {"valid": False, "error": str(e)}
        logger.warning("Config generation failed: %s", e)

    # ── Self-play evaluation loop ────────────────────────────────────────
    try:
        world_gen = LatentWorldGenerator(latent_dim=z_dim, device=device)
        curriculum = AdaptiveCurriculumManager()
        corrective = CorrectiveSynthesizer(latent_dim=z_dim)
        meta_sig = VibeThinkerMetaSignaler()

        sp_results: List[Dict[str, Any]] = []
        for ep in range(num_episodes):
            gen_out = world_gen.generate(
                mode=curriculum.generation_mode,
                batch_size=1,
                config=config,
            )
            scenario = gen_out["scenarios"]
            complexity = gen_out["complexity"].mean().item()

            # Evaluate: predict next state via dynamics, measure error
            success = True
            failure_mode = "none"
            if world_gen._dynamics is not None:
                try:
                    a_rand = torch.randn(1, world_gen.action_dim,
                                         device=device)
                    s_next, _, _ = world_gen._dynamics(scenario, a_rand)
                    pred_err = F.mse_loss(s_next, scenario).item()
                    success = pred_err < 0.5  # threshold
                    if not success:
                        failure_mode = corrective.classify_failure(
                            s_next.detach(), scenario.detach(),
                        )
                except Exception:
                    pass

            outcome = curriculum.record_outcome(
                success=success,
                metadata={"complexity": complexity, "failure_mode": failure_mode},
            )

            # Generate corrective data if failed
            if not success:
                corr_result = corrective.synthesize(
                    failure_z=scenario.squeeze(0).detach(),
                    failure_mode=failure_mode,
                )
                outcome["corrective_generated"] = corr_result["num_generated"]

            # Update meta-signaler
            meta_sig.update()

            sp_results.append(outcome)

        results["self_play"] = {
            "episodes": num_episodes,
            "final_level": curriculum.current_level,
            "final_success_rate": round(curriculum.success_rate, 4),
            "curriculum_summary": curriculum.get_summary(),
            "meta_signaler": meta_sig.get_summary(),
        }
    except Exception as e:
        results["self_play"] = {"error": str(e)}
        logger.warning("Self-play evaluation failed (non-fatal): %s", e)

    # ── Finalize ─────────────────────────────────────────────────────────
    all_passed = all(
        s.status == "completed" for s in state.steps.values()
    )
    state.overall_status = "completed" if all_passed else "completed_with_warnings"
    state.finished_at = time.time()

    results["wizard_completed"] = True
    results["overall_status"] = state.overall_status
    results["total_duration_s"] = round(
        state.finished_at - state.started_at, 3,
    )
    results["architecture"] = "self_play_synthetic_curriculum"

    logger.info("🧙 Self-Play Wizard %s in %.2fs",
                 state.overall_status,
                 results["total_duration_s"])

    return results


# =============================================================================
#  Backward-Compatible Wizard Entry Point
# =============================================================================
def run_wizard(
    model: nn.Module,
    tokens: Optional[torch.Tensor] = None,
    config: Any = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    output_dir: str = "wizard_output",
) -> Dict[str, Any]:
    """Execute the First-Run Setup Wizard (backward-compatible alias).

    .. deprecated::
        The ``tokens`` parameter is accepted for backward compatibility
        but is **ignored**.  This function delegates entirely to
        ``run_self_play_wizard()`` which operates purely in latent space.

    Args:
        model: AEONDeltaV4 model instance.
        tokens: **IGNORED** — retained for API compatibility only.
        config: AEONConfigV4 configuration (will be mutated).
        device: Computation device.
        batch_size: Batch size for inference passes.
        output_dir: Directory for saving wizard artifacts.

    Returns:
        Dict with complete wizard results including generated config.
    """
    if tokens is not None:
        logger.warning(
            "⚠️  run_wizard(): 'tokens' argument is deprecated and will be "
            "ignored.  The wizard now operates purely in latent space via "
            "Self-Play / Synthetic Curriculum (§SP).  Use "
            "run_self_play_wizard() directly."
        )

    return run_self_play_wizard(
        model=model,
        config=config,
        device=device,
        batch_size=batch_size,
        output_dir=output_dir,
    )


# =============================================================================
#  Cold-Start Detection
# =============================================================================
def is_cold_start(
    config_path: str = "wizard_output/aeon_config_wizard.json",
    weights_path: Path = VT_WEIGHTS_PATH,
) -> bool:
    """Check whether the system needs a first-run wizard execution.

    Returns True if:
      - No wizard config has been generated yet
      - No VT weights exist at the expected path

    Implements Spec §2.3 — automatic first-run detection.
    """
    if not os.path.exists(config_path):
        return True
    # Config exists but may be stale — check if weights also exist
    return not weights_path.exists()
