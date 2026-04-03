"""
AEON Self-Play Setup Wizard — Backward Compatibility Wrapper
=============================================================

All wizard classes and functions have been merged into
``aeon_server.py`` (v4.0.0).  This module re-exports them for
backward compatibility so existing imports continue to work:

    from aeon_wizard import LatentWorldGenerator
    from aeon_wizard import run_self_play_wizard

No code changes required in downstream consumers.
"""

import sys as _sys
import torch as _torch
from typing import Any as _Any, Dict as _Dict, Optional as _Optional

import torch.nn as _nn

from aeon_server import (  # noqa: F401
    GenerationMode,
    LatentWorldGenerator,
    AdaptiveCurriculumManager,
    CorrectiveSynthesizer,
    VibeThinkerMetaSignaler,
    WizardStepStatus,
    WizardState,
    VT_WEIGHTS_PATH,
    get_wizard_state,
    reset_wizard_state,
    load_vt_weights,
    run_self_play_diagnostics,
    run_corpus_diagnostics,
    compute_hyperparameters,
    bootstrap_codebook_embeddings,
    generate_config,
    run_self_play_wizard,
    run_wizard_func as run_wizard,
    is_cold_start,
)

# ── Patchable initialize_codebook ────────────────────────────────────────────
# Tests mock aeon_wizard.bootstrap_codebook_embeddings and expect
# initialize_codebook (called from within this module) to see the mock.
# We re-define a thin wrapper that resolves bootstrap_codebook_embeddings
# through the module dict so unittest.mock.patch works correctly.

import aeon_server as _srv

def initialize_codebook(
    model: _nn.Module,
    tokens: _Optional[_torch.Tensor] = None,
    config: _Any = None,
    device: _torch.device = _torch.device("cpu"),
    batch_size: int = 256,
) -> _Dict[str, _Any]:
    """Initialize VQ codebook — patchable wrapper.

    Resolves ``bootstrap_codebook_embeddings`` via the **aeon_wizard** module
    namespace so that ``unittest.mock.patch("aeon_wizard.bootstrap_codebook_embeddings", ...)``
    correctly intercepts the call.
    """
    _this = _sys.modules[__name__]
    _bootstrap = getattr(_this, "bootstrap_codebook_embeddings")
    return _srv._initialize_codebook_impl(
        model=model, tokens=tokens, config=config,
        device=device, batch_size=batch_size,
        bootstrap_fn=_bootstrap,
    )


__all__ = [
    "GenerationMode",
    "LatentWorldGenerator",
    "AdaptiveCurriculumManager",
    "CorrectiveSynthesizer",
    "VibeThinkerMetaSignaler",
    "WizardStepStatus",
    "WizardState",
    "VT_WEIGHTS_PATH",
    "get_wizard_state",
    "reset_wizard_state",
    "load_vt_weights",
    "run_self_play_diagnostics",
    "run_corpus_diagnostics",
    "compute_hyperparameters",
    "initialize_codebook",
    "bootstrap_codebook_embeddings",
    "generate_config",
    "run_self_play_wizard",
    "run_wizard",
    "is_cold_start",
]
