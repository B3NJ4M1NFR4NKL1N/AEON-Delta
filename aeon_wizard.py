"""
AEON First-Run Setup Wizard — Backend Controller
=================================================

Orchestrates the complete cold-start initialization sequence for the
AEON training pipeline.  When no prior configuration or weights exist,
the Wizard automates:

  Phase 1 — Corpus Diagnostics       (VibeThinkerPromptAdapter inference)
  Phase 2 — Hyperparameterization    (codebook_size, context_window, z_dim)
  Phase 3 — Codebook Warm-Start      (k-means on prompt embeddings)
  Phase 4 — Configuration Generation (validated AEONConfigV4)

All hyperparameters are computed automatically — zero manual input
required.  The Wizard uses VibeThinker-1.5B weights loaded from
``vibe_thinker_weights/model.safetensors``.

Spec References:
  - §4.A.1: Corpus diagnostics via VibeThinkerPromptAdapter
  - §4.A.2: Hyperparameterization via 95th-percentile CoT depth
  - §4.A.3: Codebook initialization via k-means clustering
  - §4.A.4: Configuration generation for AEONConfigV4
  - §2.1:   Weight path vibe_thinker_weights/model.safetensors
  - §2.3:   Automated First-Run without manual input
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("AEON-Wizard")

# ── Weight path constant (Spec §2.1) ────────────────────────────────────────
VT_WEIGHTS_PATH = Path("vibe_thinker_weights/model.safetensors")

# ── Codebook size search candidates ──────────────────────────────────────────
_CODEBOOK_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024]

# ── Context window bounds ────────────────────────────────────────────────────
_MIN_CONTEXT_WINDOW = 1
_MAX_CONTEXT_WINDOW = 16


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
#  Step 1: Corpus Diagnostics  (Spec §4.A.1)
# =============================================================================
def run_corpus_diagnostics(
    model: nn.Module,
    tokens: torch.Tensor,
    config: Any,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Run VibeThinkerPromptAdapter inference to assess corpus complexity.

    Implements Spec §4.A.1 — evaluates data complexity and heterogeneity
    using complexity_head scores and CoT-depth distribution.

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
        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        kernel = VibeThinkerReasoningKernel(
            config=VibeThinkerConfig(),
            hidden_dim=config.hidden_dim,
        ).to(device)

        complexity_scores: List[float] = []
        cot_depths: List[float] = []
        embeddings: List[torch.Tensor] = []

        model.eval()
        adapter.eval()
        kernel.eval()

        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size].to(device)
                z = model.encode(batch)
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
            "corpus_size": len(tokens),
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
#  Step 3: Codebook Warm-Start  (Spec §4.A.3)
# =============================================================================
def initialize_codebook(
    model: nn.Module,
    tokens: torch.Tensor,
    config: Any,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Initialize VQ codebook via k-means clustering on prompt embeddings.

    Implements Spec §4.A.3 — warm-start the vector quantizer with
    semantically meaningful centroids instead of random N(0, 0.1).

    Delegates to ``ae_train.warm_start_codebook_from_vt`` if available,
    otherwise performs inline k-means.

    Returns:
        Dict with initialization status, inertia, and method.
    """
    try:
        from ae_train import warm_start_codebook_from_vt
        return warm_start_codebook_from_vt(
            model=model,
            tokens=tokens,
            config=config,
            device=device,
            batch_size=batch_size,
        )
    except ImportError:
        logger.debug("ae_train.warm_start_codebook_from_vt not available, "
                      "using inline implementation")

    # Inline fallback: k-means on encoder outputs
    try:
        from aeon_core import VibeThinkerPromptAdapter

        adapter = VibeThinkerPromptAdapter(
            latent_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        adapter.eval()
        model.eval()

        all_z: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start:start + batch_size].to(device)
                z = model.encode(batch)
                all_z.append(z.cpu())

        embeddings = torch.cat(all_z, dim=0)
        K = getattr(config, "vq_num_embeddings",
                     getattr(config, "codebook_size", 256))
        N = embeddings.shape[0]

        if N < K:
            return {
                "initialized": False,
                "method": "insufficient_samples",
                "num_embeddings": K,
                "inertia": 0.0,
            }

        # Mini-batch k-means
        centroids = embeddings[torch.randperm(N)[:K]].clone()
        for _iter in range(50):
            assignments = []
            for start in range(0, N, batch_size):
                batch_emb = embeddings[start:start + batch_size]
                dists = torch.cdist(batch_emb, centroids)
                assignments.append(dists.argmin(dim=1))
            assignments = torch.cat(assignments)

            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = embeddings[mask].mean(dim=0)
                else:
                    # Dead cluster — reinitialize from farthest point
                    # to maximize inter-cluster separation
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
        vq = model.vq
        with torch.no_grad():
            if centroids.shape[1] == vq.embedding.weight.shape[1]:
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
            "method": "inline_kmeans",
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
        "wizard_version": "1.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_size": diagnostics.get("corpus_size", 0),
        "heterogeneous": diagnostics.get("heterogeneous", False),
        "codebook_method": codebook_result.get("method", "unknown"),
        "codebook_inertia": codebook_result.get("inertia", 0.0),
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
#  Main Wizard Orchestrator
# =============================================================================
def run_wizard(
    model: nn.Module,
    tokens: torch.Tensor,
    config: Any,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    output_dir: str = "wizard_output",
) -> Dict[str, Any]:
    """Execute the complete First-Run Setup Wizard.

    Implements Spec §4.A — full cold-start initialization:
      1. Load VibeThinker weights (§2.1)
      2. Corpus diagnostics (§4.A.1)
      3. Hyperparameterization (§4.A.2)
      4. Codebook warm-start (§4.A.3)
      5. Config generation (§4.A.4)

    All steps are fully automated — no manual hyperparameter input.

    Args:
        model: AEONDeltaV4 model instance.
        tokens: Training corpus token tensor [N, seq_len].
        config: AEONConfigV4 configuration (will be mutated).
        device: Computation device.
        batch_size: Batch size for inference passes.
        output_dir: Directory for saving wizard artifacts.

    Returns:
        Dict with complete wizard results including generated config.
    """
    reset_wizard_state()
    state = get_wizard_state()
    state.overall_status = "running"
    state.started_at = time.time()

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {"wizard_completed": False}

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

    # ── Step 1: Corpus diagnostics ───────────────────────────────────────
    step = state.steps["corpus_diagnostics"]
    step.start()
    try:
        diagnostics = run_corpus_diagnostics(
            model=model,
            tokens=tokens,
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
        logger.warning("Corpus diagnostics failed: %s", e)

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

    # ── Step 3: Codebook initialization ──────────────────────────────────
    step = state.steps["codebook_init"]
    step.start()
    try:
        codebook_result = initialize_codebook(
            model=model,
            tokens=tokens,
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
        logger.warning("Codebook initialization failed: %s", e)

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

    logger.info("🧙 Wizard %s in %.2fs",
                 state.overall_status,
                 results["total_duration_s"])

    return results


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
