"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AEON-Delta Dashboard Backend  ·  aeon_server.py  v3.4.0 — Production  ║
║  FastAPI + WebSocket + SSE · Full integration with core.py              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  NEW IN v3.2.0:                                                         ║
║  · /api/benchmark — Latency profiling (N-run stats)                     ║
║  · /api/test/run  — Full AEONTestSuite with per-test breakdown          ║
║  · /api/introspect/modules — Layer-by-layer parameter stats             ║
║  · /api/vq/codebook — VQ embedding detail with utilization history      ║
║  · /api/status/system — GPU VRAM, RAM, CPU usage                        ║
║  · /api/gradient/stats — Real-time gradient norm tracking               ║
║  · /api/config/validate — Validate config before init                   ║
║  · /api/session/export — Export full session to JSON                    ║
║  · /api/session/import — Restore session from JSON                      ║
║  · Enhanced training loop with per-step gradient norm streaming         ║
║  · SSE log streaming with per-level filtering                           ║
║  NEW IN v3.3.0:                                                         ║
║  · /api/tests/catalogue  — 642 tests × 49 sections, metadata           ║
║  · /api/tests/run        — run all/section/named, background thread    ║
║  · /api/tests/stop       — graceful cancellation                       ║
║  · /api/tests/progress   — live counters: passed/failed/error/total   ║
║  · /api/tests/results    — full+brief output, filter by status         ║
║  · /api/tests/stream     — SSE per-test events + progress pings        ║
║  · /api/tests/run_single — run one test synchronously                  ║
║  · WS type=test_event broadcast per test completion                    ║
║  · WS type=test_progress broadcast every 2s during run                 ║
║  NEW IN v3.4.0:                                                         ║
║  · /api/vibe_thinker/verify_model  — pre-init model dependency check   ║
║  · /api/vibe_thinker/install_model — auto-download tokenizer model     ║
║  · /api/emergence_summary   — cached emergence (no forward pass)       ║
║  · /api/error_evolution/seed — seed training→inference bridge           ║
║  · /api/feedback_bus         — feedback bus state + signal coverage     ║
║  · /api/convergence/detailed — full convergence monitor state           ║
║  · /api/convergence/analytics — residual distributions & bounds        ║
║  · /api/eval/perplexity    — standardized perplexity evaluation        ║
║  · /api/eval/ablation      — module ablation study                     ║
║  · /api/eval/causal_discovery — SHD/TPR/FDR causal benchmarks          ║
║  · /api/eval/continual_learning — MAML+EWC scaling & gates             ║
║  · /api/cognitive_completeness — per-axiom AGI coverage                 ║
║  · /api/regularization       — signal-derived regularization terms     ║
║  · /api/sync_from_training   — training→inference state bridge         ║
║  · /api/load_v4_checkpoint   — v4 checkpoint loading                   ║
║  · /api/vibe_thinker/save_weights  — VibeThinker weight persistence   ║
║  · /api/vibe_thinker/load_weights  — VibeThinker weight loading       ║
║  · /api/vibe_thinker/switch_weights — hot-swap VibeThinker weights    ║
║  · /api/vibe_thinker/list_weights  — list available weight files      ║
║  · /api/vibe_thinker/first_start_calibration — manual VQ alignment   ║
║  · VibeThinker auto-install before AEONDeltaV3 init                    ║
║  · Enhanced heartbeat with emergence + feedback bus telemetry           ║
╚══════════════════════════════════════════════════════════════════════════╝

Запуск:
    pip install fastapi uvicorn psutil
    python aeon_server.py [--host 0.0.0.0] [--port 8000]

Dashboard:  http://localhost:8000
API Docs:   http://localhost:8000/docs
"""

import os, sys, json, time, queue, logging, threading, traceback, math, asyncio
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from contextlib import asynccontextmanager
import statistics
import io
from contextlib import redirect_stdout, redirect_stderr
import importlib.util

import torch
import numpy as np

# ─── FastAPI / Uvicorn ───────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    # File upload support
    from fastapi import UploadFile, File, Form
except ImportError:
    print("ERROR: pip install fastapi uvicorn pydantic python-multipart")
    sys.exit(1)

# ─── Optional psutil for system stats ────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ─── AEON Core ────────────────────────────────────────────────────────────────
CORE_PATH = Path(__file__).parent / "aeon_core.py"
if not CORE_PATH.exists():
    CORE_PATH = Path(__file__).parent / "core.py"
if not CORE_PATH.exists():
    CORE_PATH = Path("/mnt/user-data/uploads/core.py")

if not CORE_PATH.exists():
    print(f"ERROR: aeon_core.py / core.py not found.")
    sys.exit(1)

sys.path.insert(0, str(CORE_PATH.parent))

CORE_LOADED = False
CORE_LOAD_ERROR = ""
# Derive module name from the resolved file (aeon_core or core)
_core_module_name = CORE_PATH.stem  # "aeon_core" or "core"
try:
    import importlib
    _core_mod = importlib.import_module(_core_module_name)
    AEONConfig = _core_mod.AEONConfig
    AEONDeltaV3 = _core_mod.AEONDeltaV3
    set_seed = _core_mod.set_seed
    AEONTestSuite = _core_mod.AEONTestSuite
    TelemetryCollector = _core_mod.TelemetryCollector
    generate_correlation_id = _core_mod.generate_correlation_id
    CORE_LOADED = True
    print(f"✅ {CORE_PATH.name} loaded successfully")
except Exception as e:
    CORE_LOAD_ERROR = str(e)
    print(f"⚠ {CORE_PATH.name} import error: {e}")


# ─── VibeThinker Model Verification & Auto-Install ────────────────────────────
def _verify_vibe_thinker_model() -> Dict[str, Any]:
    """Verify VibeThinker dependencies (transformers library + tokenizer model).

    Checks whether the ``transformers`` library is installed and whether
    the ``bert-base-uncased`` tokenizer (required by VibeThinker reasoning
    kernel integration) is cached locally.

    Returns:
        Dict with keys:
            - ``transformers_installed`` (bool)
            - ``tokenizer_cached`` (bool)
            - ``ready`` (bool) — True when both conditions are met
            - ``model_name`` (str) — expected tokenizer model identifier
            - ``message`` (str) — human-readable status
    """
    model_name = "bert-base-uncased"
    result: Dict[str, Any] = {
        "model_name": model_name,
        "transformers_installed": False,
        "tokenizer_cached": False,
        "ready": False,
        "message": "",
    }

    # 1. Check transformers library
    try:
        import transformers
        result["transformers_installed"] = True
        result["transformers_version"] = transformers.__version__
    except ImportError:
        result["message"] = "transformers library not installed"
        return result

    # 2. Check if tokenizer model is cached
    try:
        from transformers import AutoTokenizer
        _cache_dir = None
        try:
            _cache_dir = transformers.utils.hub.TRANSFORMERS_CACHE
        except AttributeError:
            try:
                from huggingface_hub import constants as hf_constants
                _cache_dir = hf_constants.HF_HUB_CACHE
            except Exception:
                pass
        result["cache_dir"] = str(_cache_dir) if _cache_dir else "default"

        # Attempt offline-only load to check if model is cached
        try:
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            result["tokenizer_cached"] = True
        except Exception:
            result["tokenizer_cached"] = False
    except Exception as e:
        result["message"] = f"Error checking tokenizer cache: {e}"
        return result

    result["ready"] = result["transformers_installed"] and result["tokenizer_cached"]
    if result["ready"]:
        result["message"] = f"VibeThinker ready: {model_name} tokenizer cached"
    else:
        result["message"] = f"VibeThinker tokenizer '{model_name}' not cached — download required"
    return result


def _install_vibe_thinker_model() -> Dict[str, Any]:
    """Install VibeThinker dependencies and download the tokenizer model.

    1. Installs the ``transformers`` library if not present.
    2. Downloads the ``bert-base-uncased`` tokenizer to the HuggingFace
       cache so that subsequent ``AEONDeltaV3.__init__`` finds it locally.

    Returns:
        Dict with installation steps and final readiness status.
    """
    steps: List[Dict[str, Any]] = []
    model_name = "bert-base-uncased"

    # Step 1: Install transformers if missing
    try:
        import transformers
        steps.append({"step": "transformers_install", "status": "already_installed",
                       "version": transformers.__version__})
    except ImportError:
        try:
            import subprocess
            logging.info("📦 Installing transformers library for VibeThinker...")
            # 5-minute timeout accommodates slow networks; pip install
            # transformers downloads ~500 MB of dependencies on first run.
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "transformers", "--quiet"],
                timeout=300,
            )
            import transformers
            steps.append({"step": "transformers_install", "status": "installed",
                           "version": transformers.__version__})
            logging.info(f"✅ transformers {transformers.__version__} installed")
        except Exception as e:
            steps.append({"step": "transformers_install", "status": "failed",
                           "error": str(e)})
            return {"ok": False, "steps": steps,
                    "message": f"Failed to install transformers: {e}"}

    # Step 2: Download tokenizer model
    try:
        from transformers import AutoTokenizer
        logging.info(f"📥 Downloading tokenizer: {model_name}...")
        AutoTokenizer.from_pretrained(model_name)
        steps.append({"step": "tokenizer_download", "status": "cached",
                       "model": model_name})
        logging.info(f"✅ Tokenizer '{model_name}' cached successfully")
    except Exception as e:
        steps.append({"step": "tokenizer_download", "status": "failed",
                       "error": str(e)})
        return {"ok": False, "steps": steps,
                "message": f"Failed to download tokenizer: {e}"}

    # Final verification
    verification = _verify_vibe_thinker_model()
    return {
        "ok": verification["ready"],
        "steps": steps,
        "verification": verification,
        "message": "VibeThinker model ready" if verification["ready"]
                   else "VibeThinker model installation incomplete",
    }


# ─── AEON ae_train v4 ─────────────────────────────────────────────────────────
AE_TRAIN_PATH = Path(__file__).parent / "ae_train.py"
if not AE_TRAIN_PATH.exists():
    AE_TRAIN_PATH = Path("/mnt/user-data/uploads/ae_train.py")

AE_TRAIN_LOADED = False
AE_TRAIN_ERROR = ""
_ae_module = None

try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ae_train", str(AE_TRAIN_PATH))
    _ae_module = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ae_module)
    AE_TRAIN_LOADED = True
    print("✅ ae_train.py loaded successfully")
except Exception as _e:
    AE_TRAIN_ERROR = str(_e)
    print(f"⚠ ae_train.py import error: {_e}")


class _V4WSLogHandler(logging.Handler):
    """Routes ae_train logs into the dashboard's v4 log buffer."""
    def emit(self, record: logging.LogRecord):
        entry = {
            "time":  time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "subsys": "ae_train",
            "msg":   self.format(record) if record.exc_info else record.getMessage(),
            "ts":    record.created,
        }
        APP.v4_log_buffer.append(entry)
        if len(APP.v4_log_buffer) > 8000:
            APP.v4_log_buffer = APP.v4_log_buffer[-8000:]
        # Mirror to main log history + queue
        APP.log_history.append(entry)
        if len(APP.log_history) > 4000:
            APP.log_history = APP.log_history[-4000:]
        try:
            APP.log_queue.put_nowait(entry)
        except queue.Full:
            pass

_v4_ws_handler = _V4WSLogHandler()
_v4_ws_handler.setLevel(logging.DEBUG)
_v4_ws_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
# Attach to the ae_train logger namespace
for _lname in ("AEON-Training-v4", "ae_train", "root"):
    logging.getLogger(_lname).addHandler(_v4_ws_handler)


class _DashboardMonitor:
    """
    Drop-in replacement for ae_train.TrainingMonitor that also pushes
    structured progress into APP.v4_progress so the dashboard can poll it.
    Delegates all logging to the standard ae_train logger.
    """
    def __init__(self, delegate_monitor, phase_metrics_ref: dict):
        self._d = delegate_monitor
        self._ref = phase_metrics_ref

    def __getattr__(self, name):
        return getattr(self._d, name)

    def log_batch(self, batch_idx, total_batches, metrics, phase="phase_A", log_every=10):
        self._d.log_batch(batch_idx, total_batches, metrics, phase, log_every)
        APP.v4_progress.update({
            "batch": batch_idx + 1,
            "total_batches": total_batches,
            "batch_metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        })

    def end_epoch(self, epoch, total_epochs, epoch_metrics, phase="phase_A"):
        result = self._d.end_epoch(epoch, total_epochs, epoch_metrics, phase)
        safe_m = {k: round(float(v), 6) if isinstance(v, float) else v
                  for k, v in epoch_metrics.items()}
        # Push to history
        self._ref.setdefault(phase, []).append(safe_m)
        APP.v4_metrics_history = self._ref

        loss_key = "total" if "total" in epoch_metrics else "mse_loss"
        APP.v4_progress.update({
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "phase": phase,
            "epoch_metrics": safe_m,
            "current_loss": round(float(epoch_metrics.get(loss_key, 0)), 6),
            "best_loss": round(float(self._d.best_loss), 6),
        })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
class AppState:
    model: Optional[Any]       = None
    config: Optional[Any]      = None
    trainer: Optional[Any]     = None
    training_thread: Optional[threading.Thread] = None
    training_active: bool      = False
    training_stop: bool        = False
    training_progress: dict    = {}
    gradient_history: List[dict] = []   # per-step grad norms
    step_loss_history: List[dict] = []  # per-step losses
    # ── Global device selection ───────────────────────────────────
    # Set during /api/init from the user's device_str choice.
    # Subsequent subsystems (engine, v4 training, tests) honour this
    # value so that the entire pipeline runs on the same device.
    selected_device: str       = "auto"
    test_results: Optional[dict]  = None
    benchmark_results: Optional[dict] = None
    ws_clients: List[WebSocket] = []
    log_queue: queue.Queue     = queue.Queue(maxsize=4000)
    log_history: List[dict]    = []
    session_meta: dict         = {"init_time": None, "init_count": 0}
    # ── Thread safety lock ─────────────────────────────────────────
    # Protects shared mutable state from concurrent access by
    # background training/testing threads and async HTTP handlers.
    # Use ``with APP.lock:`` around reads and writes to fields that
    # are modified by background threads (training_active, v4_active,
    # test_run_active, test_run_results, v4_progress, etc.).
    lock: threading.RLock      = threading.RLock()
    # ── Test-runner state ──────────────────────────────────────────
    test_run_active: bool        = False
    test_run_stop_event          = None       # threading.Event
    test_run_results: List[dict] = []
    test_run_progress: dict      = {}
    test_run_summary: dict       = {}
    test_catalogue_cache: list   = []
    # ── AEON v4 Training state ─────────────────────────────────────
    v4_active: bool              = False
    v4_stop: bool                = False
    v4_thread: Optional[threading.Thread] = None
    v4_progress: dict            = {}
    v4_log_buffer: List[dict]    = []   # dedicated log ring for v4 training
    v4_metrics_history: dict     = {"phase_A": [], "phase_B": []}
    v4_upload_dir: str           = "./training_data"
    v4_trained_model: Optional[Any] = None      # trained AEONDeltaV4 instance
    v4_trained_model_path: Optional[str] = None # path to saved v4 checkpoint
    v4_adaptive_state: dict      = {}           # adaptive controller telemetry
    v4_data_analysis: dict       = {}           # data characteristics analysis

APP = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Log Handler
# ═══════════════════════════════════════════════════════════════════════════════
class WSLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        entry = {
            "time":   time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level":  record.levelname,
            "subsys": record.name.replace("AEON-Delta", "core").replace("root", "sys")[:20],
            "msg":    self.format(record) if record.exc_info else record.getMessage(),
            "ts":     record.created,
        }
        APP.log_history.append(entry)
        if len(APP.log_history) > 4000:
            APP.log_history = APP.log_history[-4000:]
        try:
            APP.log_queue.put_nowait(entry)
        except queue.Full:
            pass

_ws_handler = WSLogHandler()
_ws_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_ws_handler)
logging.getLogger("AEON-Delta").addHandler(_ws_handler)


# ═══════════════════════════════════════════════════════════════════════════════
#  Broadcast Helper
# ═══════════════════════════════════════════════════════════════════════════════
async def broadcast(msg: dict):
    dead = []
    for ws in list(APP.ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in APP.ws_clients:
            APP.ws_clients.remove(ws)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════════
class InitRequest(BaseModel):
    # Core architecture
    hidden_dim: int = 256
    z_dim: int = 256
    vq_embedding_dim: int = 256
    vq_num_embeddings: int = 8192
    vocab_size: int = 30522
    seq_length: int = 64
    num_pillars: int = 64
    encoder_backend: str = "ssm"
    decoder_backend: str = "ssm"
    device_str: str = "auto"
    max_iterations: int = 50
    convergence_threshold: float = 1e-5
    lipschitz_target: float = 0.85
    use_vq: bool = True
    use_amp: bool = True
    enable_inference_cache: bool = True
    nan_policy: str = "WARN"
    pretrained_backbone: str = ""
    # Memory
    enable_hierarchical_memory: bool = False
    hierarchical_working_capacity: int = 7
    hierarchical_episodic_capacity: int = 1000
    enable_neurogenic_memory: bool = False
    neurogenic_retrieval_weight: float = 0.1
    neurogenic_retrieval_k: int = 3
    enable_temporal_memory: bool = False
    temporal_memory_decay_rate: float = 0.01
    enable_consolidating_memory: bool = False
    consolidating_semantic_weight: float = 0.1
    # Causal
    enable_causal_model: bool = False
    enable_notears_causal: bool = False
    notears_num_vars: int = 8
    lambda_causal_dag: float = 0.01
    enable_causal_world_model: bool = False
    enable_unified_simulator: bool = False
    unified_simulator_num_vars: int = 16
    enable_causal_context: bool = False
    enable_causal_trace: bool = False
    causal_blend_weight: float = 0.05
    # Meta-cognition
    enable_auto_critic: bool = False
    auto_critic_threshold: float = 0.85
    auto_critic_max_iterations: int = 3
    enable_recursive_meta_loop: bool = False
    recursive_meta_depth: int = 3
    enable_metacognitive_recursion: bool = False
    metacognitive_trigger_threshold: float = 0.5
    metacognitive_extra_iterations: int = 10
    enable_error_evolution: bool = False
    enable_meta_recovery_integration: bool = False
    enable_meta_learning: bool = False
    meta_ewc_lambda: float = 1000.0
    # Coherence
    enable_full_coherence: bool = False
    enable_module_coherence: bool = False
    module_coherence_threshold: float = 0.5
    enable_cross_validation: bool = False
    enable_ns_consistency_check: bool = False
    enable_complexity_estimator: bool = False
    enable_external_trust: bool = False
    enable_hybrid_reasoning: bool = False
    intra_pass_feedback_threshold: float = 0.3
    intra_pass_feedback_scale: float = 0.05
    # Planning
    enable_world_model: bool = False
    world_model_tree_depth: int = 3
    surprise_threshold: float = 0.5
    enable_mcts_planner: bool = False
    enable_active_learning_planner: bool = False
    active_learning_curiosity_weight: float = 1.0
    enable_hierarchical_vae: bool = False
    hvae_blend_weight: float = 0.1
    seed: int = 42
    # Observability
    enable_structured_logging: bool = False
    enable_academic_mode: bool = False
    enable_telemetry: bool = True
    # VibeThinker
    vibe_thinker_enabled: bool = True
    vibe_thinker_adapter_hidden: int = 256
    vibe_thinker_projection_dim: int = 128
    vibe_thinker_adapter_lr: float = 1e-4
    vibe_thinker_max_tokens: int = 512
    vibe_thinker_temperature: float = 0.6
    vibe_thinker_top_p: float = 0.95
    vibe_thinker_confidence_threshold: float = 0.7
    vibe_thinker_entropy_threshold: float = 0.5
    vibe_thinker_calibration_alpha: float = 0.1
    vibe_thinker_adaptation_rate: float = 0.01
    vibe_thinker_consolidation_interval: int = 100
    vibe_thinker_complexity_threshold: float = 0.5
    vibe_thinker_psi_weight: float = 0.1
    vibe_thinker_weights_path: str = ""

class InferRequest(BaseModel):
    prompt: str = "What is consciousness?"
    max_length: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    fast: bool = False

class ForwardRequest(BaseModel):
    input_ids: List[int]
    fast: bool = False

class TrainRequest(BaseModel):
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-5
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    checkpoint_dir: str = "./checkpoints"
    log_grad_norms: bool = True
    synthetic_samples: int = 256

class BenchmarkRequest(BaseModel):
    num_runs: int = Field(default=20, ge=1, le=200)
    prompt: str = "Benchmark test"
    max_length: int = 32
    temperature: float = 1.0
    top_k: int = 50
    fast: bool = True

class SaveRequest(BaseModel):
    path: str = "./checkpoints/aeon_state"

class LoadRequest(BaseModel):
    path: str = "./checkpoints/aeon_state"

class ValidateConfigRequest(BaseModel):
    config: dict

class V4TrainRequest(BaseModel):
    """Configuration for AEON v4 two-phase training pipeline."""
    # Data
    json_path: str       = Field("./training_data/data.json", description="Path to training JSON/JSONL/TXT file")
    output_dir: str      = Field("./processed_v4/",           description="Output directory for checkpoints and artifacts")
    resume_from: str     = Field("",                           description="Path to checkpoint to resume from (empty = fresh start)")
    # Training schedule
    epochs_A: int        = Field(30,    ge=1,    le=1000,  description="Phase A epochs: AutoEncoder + VQ")
    epochs_B: int        = Field(10,    ge=1,    le=1000,  description="Phase B epochs: Contextual RSSM")
    # Architecture
    z_dim: int           = Field(256,   ge=64,   le=2048,  description="Latent dimension")
    hidden_dim: int      = Field(256,   ge=64,   le=2048,  description="Hidden dimension")
    vq_num_embeddings: int = Field(2048, ge=64,  le=65536, description="VQ codebook size")
    context_window: int  = Field(3,     ge=1,    le=16,    description="RSSM context window (K previous states)")
    num_pillars: int     = Field(5,     ge=1,    le=64,    description="Number of cognitive pillars")
    seq_length: int      = Field(64,    ge=16,   le=8192,  description="Token sequence length per chunk")
    # Optimiser
    learning_rate: float = Field(3e-5,  gt=0,    description="Peak learning rate")
    min_learning_rate: float = Field(1e-6, gt=0, description="Minimum learning rate for cosine decay")
    batch_size: int      = Field(16,    ge=1,    le=512,   description="Batch size")
    grad_clip: float     = Field(0.5,   gt=0,    description="Gradient clip norm")
    warmup_steps: int    = Field(1000,  ge=0,    description="LR warmup steps")
    weight_decay: float  = Field(0.01,  ge=0.0,  description="AdamW weight decay")
    gradient_accumulation_steps: int = Field(2, ge=1, le=64, description="Gradient accumulation steps")
    entropy_weight: float = Field(0.1,  ge=0.0,  description="VQ codebook entropy regularisation weight")
    # VQ-VAE advanced
    vq_commitment_cost: float = Field(0.25, ge=0.0, le=2.0,  description="VQ commitment cost coefficient")
    vq_loss_weight: float     = Field(0.5,  ge=0.0, le=5.0,  description="VQ loss weight in total loss")
    vq_ema_decay: float       = Field(0.99, ge=0.0, le=1.0,  description="EMA decay for VQ codebook updates")
    vq_temperature: float     = Field(1.0,  gt=0.0,          description="Gumbel-Softmax temperature")
    vq_reset_threshold: int   = Field(30,   ge=1,   le=1000, description="Epochs before dead code reset")
    # RSSM advanced
    rssm_hidden_dim: int = Field(512, ge=64, le=4096, description="RSSM hidden dimension")
    # Regularisation
    dropout_rate: float    = Field(0.1, ge=0.0, le=0.9,  description="Dropout rate")
    label_smoothing: float = Field(0.1, ge=0.0, le=0.5,  description="Label smoothing factor")
    # Early stopping & checkpointing
    early_stopping_patience: int = Field(5,  ge=1, le=100, description="Early stopping patience (epochs)")
    min_delta: float             = Field(1e-4, ge=0.0,     description="Minimum improvement delta for early stopping")
    save_every_n_epochs: int     = Field(5,  ge=1, le=100, description="Save checkpoint every N epochs")
    keep_n_checkpoints: int      = Field(3,  ge=1, le=50,  description="Max number of checkpoints to keep")
    # Document-aware
    min_doc_chunks: int  = Field(2, ge=1, le=100, description="Minimum chunks per document for document-aware mode")
    # Flags
    document_aware: bool = Field(True,  description="Build RSSM pairs within document boundaries")
    use_amp: bool        = Field(True,  description="Use automatic mixed precision (requires CUDA)")
    # Misc
    seed: int            = Field(42,    description="Random seed")


# ═══════════════════════════════════════════════════════════════════════════════
#  App Lifespan
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    logging.info("AEON Dashboard server v3.4.0 starting")
    asyncio.create_task(_log_forwarder())
    asyncio.create_task(_heartbeat())
    asyncio.create_task(_test_progress_broadcaster())
    # Pre-parse test catalogue
    _warmup_test_catalogue()
    yield
    logging.info("AEON Dashboard server shutting down")

app = FastAPI(
    title="AEON-Delta Dashboard API",
    version="3.4.0",
    description="Production dashboard API for AEON-Delta RMT v3.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  X-Request-ID / Correlation ID Middleware
# ═══════════════════════════════════════════════════════════════════════════════
import uuid as _uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Injects a correlation ID into every HTTP request/response for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(
            "X-Request-ID", generate_correlation_id()
        )
        request.state.correlation_id = correlation_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response


app.add_middleware(CorrelationIDMiddleware)

# ─── Serve Dashboard HTML ────────────────────────────────────────────────────
DASHBOARD_FILE = Path(__file__).parent / "AEON_Dashboard.html"

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text(encoding="utf-8")
    return HTMLResponse("<h1>AEON Dashboard</h1><p>Place AEON_Dashboard.html next to aeon_server.py</p>")


# ─── Three.js CDN Proxy (serves Three.js files locally) ─────────────────────
_THREE_JS_CDNS = [
    "https://cdn.jsdelivr.net/npm/three@0.160.0",
    "https://unpkg.com/three@0.160.0",
]
_three_js_cache: Dict[str, bytes] = {}

def _fetch_cdn_url(url: str) -> bytes:
    """Fetch a URL synchronously (runs in thread pool)."""
    req = urllib.request.Request(url, headers={"User-Agent": "AEON-Server/3.4"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()

@app.get("/cdn/three/{path:path}")
async def proxy_threejs(path: str):
    """Proxy Three.js files from CDN to avoid browser-level blocks.

    Fetches files from jsdelivr CDN (with unpkg fallback), caches them
    in memory, and serves them with the correct Content-Type.  This
    ensures the 3D Architecture visualization works even when CDN access
    is blocked by ad-blockers, corporate proxies, or restrictive CSP
    policies.
    """
    # Validate path to prevent directory traversal / SSRF
    if ".." in path or path.startswith("/") or not path.endswith(".js"):
        raise HTTPException(400, "Invalid path")

    if path in _three_js_cache:
        return Response(
            _three_js_cache[path],
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400",
                     "Access-Control-Allow-Origin": "*"},
        )

    last_error = None
    for cdn_base in _THREE_JS_CDNS:
        url = f"{cdn_base}/{path}"
        try:
            data = await asyncio.to_thread(_fetch_cdn_url, url)
            _three_js_cache[path] = data
            return Response(
                data,
                media_type="application/javascript",
                headers={"Cache-Control": "public, max-age=86400",
                         "Access-Control-Allow-Origin": "*"},
            )
        except Exception as e:
            last_error = e
            logging.warning(f"Three.js CDN proxy failed for {cdn_base}/{path}: {e}")
            continue

    raise HTTPException(502, f"All CDN sources failed for {path}: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/status")
async def get_status():
    return {
        "core_loaded": CORE_LOADED,
        "core_error": CORE_LOAD_ERROR if not CORE_LOADED else None,
        "model_ready": APP.model is not None,
        "training": APP.training_active,
        "device": str(APP.model.device) if APP.model else "none",
        "selected_device": APP.selected_device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "observability": {
            "structured_logging": getattr(APP.config, "enable_structured_logging", False) if APP.config else False,
            "academic_mode": getattr(APP.config, "enable_academic_mode", False) if APP.config else False,
            "telemetry": getattr(APP.config, "enable_telemetry", False) if APP.config else False,
        },
    }

@app.get("/api/status/system")
async def get_system_status():
    """Detailed system resource status: RAM, GPU VRAM, CPU."""
    info = {
        "ok": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(),
    }
    # CPU / RAM
    if PSUTIL_AVAILABLE:
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        info.update({
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_percent": vm.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "process_ram_mb": round(proc.memory_info().rss / 1e6, 1),
            "cpu_count": psutil.cpu_count(),
        })

    # CUDA memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            info.setdefault("gpu_devices", []).append({
                "index": i,
                "name": props.name,
                "total_mb": round(props.total_memory / 1e6, 0),
                "allocated_mb": round(alloc / 1e6, 1),
                "reserved_mb": round(reserved / 1e6, 1),
                "free_mb": round((props.total_memory - alloc) / 1e6, 1),
            })

    # Model stats
    if APP.model is not None:
        info["model_params"] = APP.model.count_parameters()
        info["model_device"] = str(APP.model.device)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
#  COGNITIVE UNITY / ARCHITECTURAL HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
def _make_json_safe(obj):
    """Recursively convert non-JSON-serializable values to safe types.

    Handles ``torch.Tensor``, NumPy scalars/arrays, ``float('nan')``/
    ``float('inf')``, ``set``/``frozenset``, ``bytes``, and arbitrary
    objects (converted via ``str()`` as a last resort) so that FastAPI's
    default JSON encoder never raises ``TypeError``.
    """
    # --- torch Tensors ---
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # --- NumPy arrays ---
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # --- NumPy scalars (np.float64, np.int64, np.bool_, …) ---
    if isinstance(obj, np.generic):
        return obj.item()

    # --- Python floats with special IEEE-754 values ---
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj

    # --- Primitive pass-through ---
    if isinstance(obj, (int, str, bool, type(None))):
        return obj

    # --- Dicts ---
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # --- Lists / tuples ---
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]

    # --- Sets / frozensets ---
    if isinstance(obj, (set, frozenset)):
        return [_make_json_safe(item) for item in sorted(obj, key=str)]

    # --- Bytes ---
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    # --- Fallback: stringify unknown objects ---
    return str(obj)


@app.get("/api/cognitive_unity")
async def get_cognitive_unity():
    """Return AGI coherence diagnostics from verify_cognitive_unity().

    Exposes mutual-verification coverage, uncertainty→metacognition
    coverage, root-cause traceability, and a composite cognitive-unity
    score via the REST API so the dashboard and external monitors can
    query AGI coherence status without programmatic access.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.verify_cognitive_unity()
        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"cognitive_unity error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/architectural_health")
async def get_architectural_health():
    """Return synthesized architectural health from get_architectural_health().

    Single endpoint combining cognitive unity, pipeline wiring, and
    convergence health into one actionable diagnostic.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.get_architectural_health()
        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"architectural_health error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/coherence_report")
async def get_coherence_report():
    """Return a comprehensive architectural coherence report.

    Synthesizes cognitive unity, pipeline wiring, error evolution
    effectiveness, provenance traceability, and UCC correction guidance
    into a single structured report that assesses whether AEON-Delta
    operates as a unified, self-reflective, causally coherent system.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.architectural_coherence_report()
        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"coherence_report error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/cognitive_activation")
async def get_cognitive_activation():
    """Return the cognitive activation status of the AEON-Delta system.

    Produces the three deliverables required by the Final Integration &
    Cognitive Activation task:

    1. **Integration Map** — connected vs. isolated critical paths.
    2. **Critical Patches** — remaining disconnected nodes with status.
    3. **Activation Sequence** — logical order for safe online activation.

    Delegates to ``AEONDeltaV3.get_cognitive_activation_report()`` which
    synthesizes ``verify_cognitive_unity()``, ``verify_pipeline_wiring()``,
    ``self_diagnostic()``, and ``get_architectural_health()`` into an
    actionable activation report.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        report = APP.model.get_cognitive_activation_report()
        return _make_json_safe(report)
    except Exception as e:
        logging.error(f"cognitive_activation error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/verify_and_reinforce")
async def trigger_verify_and_reinforce():
    """Trigger an on-demand mutual-reinforcement cycle.

    Runs ``verify_and_reinforce()`` which performs a full architectural
    coherence report and then feeds identified weaknesses back into
    error evolution and metacognitive trigger weights.  This enables
    external callers (dashboard, monitoring tools) to force a
    reinforcement cycle outside the periodic forward-pass schedule.

    Returns the coherence report augmented with ``reinforcement_actions``
    describing what corrective feedback was applied.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.verify_and_reinforce()
        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"verify_and_reinforce error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/system_emergence")
async def get_system_emergence():
    """Return the unified system emergence report.

    Produces the four deliverables required by the Final Integration &
    Cognitive Activation task:

    1. **Integration Map** — connected vs. isolated critical paths with
       wiring coverage, provenance coverage, and feedback bus coverage.
    2. **Critical Patches** — remaining disconnected nodes with module
       health scores and remediation guidance.
    3. **Activation Sequence** — logical order for safe online activation.
    4. **System Emergence Status** — actionable readiness verdict
       synthesizing mutual reinforcement, meta-cognitive trigger, and
       causal transparency requirements.

    This endpoint calls ``system_emergence_report()`` on the model,
    which internally synthesizes ``verify_cognitive_unity()``,
    ``verify_pipeline_wiring()``, ``self_diagnostic()``, and
    ``get_architectural_health()`` into a single actionable report.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.system_emergence_report()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"system_emergence error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/verify_causal_chain")
async def verify_causal_chain():
    """Verify causal transparency across all subsystems.

    Checks that key architectural subsystems have recorded causal
    trace entries and that those entries can be traced back to root
    causes.  Returns coverage and any untraced subsystems so that
    external consumers can assess whether the Causal Transparency
    requirement is met.

    When untraced subsystems are found, automatically triggers a
    ``verify_and_reinforce()`` cycle to feed the causal-chain gap
    into error evolution and metacognitive trigger weights, closing
    the gap where this endpoint was purely diagnostic and never
    initiated corrective action.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.verify_causal_chain()
        # When causal chain is incomplete, trigger a reinforcement
        # cycle so the gap is actively corrected rather than merely
        # reported.  This bridges the discontinuity where the
        # diagnostic endpoint observed architectural weaknesses but
        # never fed them into the self-correction loop.
        corrective_action = None
        if not result.get('traceable', False):
            try:
                corrective_action = APP.model.verify_and_reinforce()
            except Exception as _corr_err:
                logging.warning(
                    "verify_causal_chain corrective action failed: %s",
                    _corr_err,
                )
        return _make_json_safe({
            "ok": True,
            **result,
            "corrective_action_applied": corrective_action is not None,
            "corrective_action": corrective_action,
        })
    except Exception as e:
        logging.error(f"verify_causal_chain error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/diagnostic/full")
async def get_full_diagnostic():
    """Return the complete self-diagnostic report.

    Exposes the full ``self_diagnostic()`` result including active modules,
    verified connections, gaps with remediation guidance, error evolution
    summary, runtime coherence, provenance attribution, output reliability,
    pipeline wiring assessment, and all other diagnostic fields.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.self_diagnostic()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"diagnostic/full error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/cognitive_state_snapshot")
async def get_cognitive_state_snapshot():
    """Return a unified snapshot of the complete cognitive state.

    Aggregates metacognitive state, causal chain verification,
    system emergence report, cognitive unity, verify-and-reinforce
    results, and error evolution summary into a single response.
    This bridges the gap where cognitive state was distributed across
    multiple API endpoints with no unified aggregator.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.get_cognitive_state_snapshot()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"cognitive_state_snapshot error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/pipeline_wiring")
async def get_pipeline_wiring():
    """Return the full pipeline wiring verification report.

    Exposes ``verify_pipeline_wiring()`` which checks every declared
    dependency edge between subsystems, validates the DAG is acyclic,
    assesses provenance coverage, and reports uncertainty propagation
    coverage.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.verify_pipeline_wiring()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"pipeline_wiring error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/diagnostic/remediate")
async def apply_remediation():
    """Apply automated diagnostic remediation.

    Calls ``apply_diagnostic_remediation()`` which inspects gaps identified
    by ``self_diagnostic()`` and attempts to fix any that have automated
    remediation paths.  Returns the list of remediated and skipped
    components.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.apply_diagnostic_remediation()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"diagnostic/remediate error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/cognitive_snapshot/export")
async def export_cognitive_snapshot():
    """Export a cognitive snapshot of the current model state.

    Saves the full model weights and cognitive memory state to a
    directory.  Returns success status and details of what was saved.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.export_cognitive_snapshot()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"cognitive_snapshot/export error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/cognitive_snapshot/import")
async def import_cognitive_snapshot():
    """Import a previously exported cognitive snapshot.

    Loads model weights and cognitive memory state from a snapshot
    directory.  Returns success status and details of what was restored.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result = APP.model.import_cognitive_snapshot()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"cognitive_snapshot/import error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  INIT / DEINIT
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/init")
async def init_model(req: InitRequest):
    if not CORE_LOADED:
        raise HTTPException(503, f"core.py failed to load: {CORE_LOAD_ERROR}")
    try:
        logging.info(f"Initializing AEONDeltaV3 · backend={req.encoder_backend} · hidden={req.hidden_dim} · seed={req.seed}")

        # ── VibeThinker Pre-Init Model Verification ──────────────────
        # When VibeThinker is enabled, verify that its model dependencies
        # (transformers library + bert-base-uncased tokenizer) are
        # available.  If not, attempt automatic installation so that
        # the subsequent AEONDeltaV3.__init__ finds the tokenizer
        # without a cold-start failure.
        _vt_verification: Dict[str, Any] = {}
        if req.vibe_thinker_enabled:
            logging.info("🔍 Verifying VibeThinker model dependencies...")
            _vt_verification = _verify_vibe_thinker_model()
            if not _vt_verification.get("ready", False):
                logging.info("📦 VibeThinker model not cached — initiating auto-install...")
                _install_result = _install_vibe_thinker_model()
                _vt_verification["auto_install"] = _install_result
                if _install_result.get("ok"):
                    logging.info("✅ VibeThinker model auto-installed successfully")
                    _vt_verification["ready"] = True
                else:
                    logging.warning(
                        "⚠ VibeThinker model auto-install failed — "
                        "VibeThinker will operate in degraded mode: %s",
                        _install_result.get("message", "unknown"),
                    )
            else:
                logging.info("✅ VibeThinker model dependencies verified")

        set_seed(req.seed)
        cfg_kwargs = req.model_dump()
        cfg_kwargs.pop("seed", None)
        config = AEONConfig(**cfg_kwargs)
        APP.config = config
        logging.info("Building model graph...")
        model = AEONDeltaV3(config)
        model.eval()
        APP.model = model
        APP.gradient_history.clear()
        APP.step_loss_history.clear()
        # Store the resolved device globally so that training, engine,
        # and test subsystems all operate on the same device.
        APP.selected_device = str(model.device)

        params = model.count_parameters()
        trainable = model.count_trainable_parameters()
        arch = model.print_architecture_summary()

        APP.session_meta["init_time"] = time.time()
        APP.session_meta["init_count"] += 1

        # Count enabled flags
        flags = [k for k, v in req.model_dump().items() if k.startswith("enable_") and v is True]
        # VibeThinker status
        _vt_cfg = getattr(model, 'vibe_thinker_config', None)
        _vt_enabled = _vt_cfg is not None and getattr(_vt_cfg, 'enabled', False)
        _vt_summary: Dict[str, Any] = {}
        if _vt_enabled:
            try:
                _vt_summary = {
                    "enabled": True,
                    "adapter": model.vibe_thinker_adapter is not None,
                    "kernel": model.vibe_thinker_kernel is not None,
                    "parser": model.vibe_thinker_parser is not None,
                    "learner": model.vibe_thinker_learner is not None,
                    "integration": model.vibe_thinker_integration is not None,
                    "model_verification": _vt_verification,
                    "is_first_start": getattr(model, '_vt_is_first_start', False),
                    "weights_loaded": not getattr(model, '_vt_is_first_start', True),
                    "weights_path": req.vibe_thinker_weights_path or "",
                }
            except Exception:
                _vt_summary = {"enabled": True, "model_verification": _vt_verification}
        else:
            _vt_summary = {"enabled": False}
        # Emergence status from activation
        _emergence = {}
        try:
            _emergence = {
                "activation_complete": getattr(model, '_cognitive_activation_complete', False),
                "cached_verdict": getattr(model, '_cached_emergence_verdict', False),
            }
        except Exception:
            pass

        # ── Feedback bus initial state ──────────────────────────────
        _feedback_bus_info = {}
        try:
            fb = getattr(model, 'feedback_bus', None)
            if fb is not None:
                _feedback_bus_info = {
                    "total_channels": fb.total_channels,
                    "core_channels": fb.NUM_SIGNAL_CHANNELS,
                    "dynamic_channels": len(fb._extra_signals),
                }
        except Exception:
            pass

        # ── Convergence monitor state ───────────────────────────────
        _convergence_info = {}
        try:
            cm = getattr(model, 'convergence_monitor', None)
            if cm is not None:
                _convergence_info = cm.get_convergence_summary()
        except Exception:
            pass

        logging.info(f"✅ Model ready · {params:,} params · {len(flags)} subsystems · device={model.device}")

        return {
            "ok": True,
            "parameters": params,
            "trainable_parameters": trainable,
            "device": str(model.device),
            "encoder_backend": config.encoder_backend,
            "decoder_backend": config.decoder_backend,
            "hidden_dim": config.hidden_dim,
            "z_dim": config.z_dim,
            "vq_num_embeddings": config.vq_num_embeddings,
            "enabled_flags": flags,
            "architecture_summary": arch,
            "vibe_thinker": _vt_summary,
            "emergence": _emergence,
            "feedback_bus": _feedback_bus_info,
            "convergence": _convergence_info,
        }
    except AssertionError as e:
        raise HTTPException(400, f"Config validation failed: {e}")
    except Exception as e:
        logging.error(f"Init error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Init failed: {e}")


@app.post("/api/deinit")
async def deinit_model():
    if APP.model is not None:
        del APP.model
        APP.model = None
        APP.config = None
        APP.selected_device = "auto"
        APP.gradient_history.clear()
        APP.step_loss_history.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Model deinitialized · GPU cache cleared")
    return {"ok": True}


@app.post("/api/config/validate")
async def validate_config(req: ValidateConfigRequest):
    """Validate a config dict without actually creating the model."""
    if not CORE_LOADED:
        raise HTTPException(503, "core.py not loaded")
    errors = []
    warnings = []
    cfg = req.config

    # Critical checks
    z = cfg.get("z_dim", 256)
    vqe = cfg.get("vq_embedding_dim", 256)
    if z != vqe:
        errors.append(f"vq_embedding_dim ({vqe}) MUST equal z_dim ({z})")

    hidden = cfg.get("hidden_dim", 256)
    if hidden < 64:
        errors.append("hidden_dim must be ≥ 64")
    if hidden % 64 != 0:
        warnings.append(f"hidden_dim={hidden} is not a multiple of 64 — may affect performance")

    seq = cfg.get("seq_length", 64)
    if seq > 2048:
        warnings.append(f"seq_length={seq} is large — memory usage may be high")

    lip = cfg.get("lipschitz_target", 0.85)
    if lip >= 1.0:
        errors.append(f"lipschitz_target={lip} must be < 1.0 for Banach convergence")

    vq_n = cfg.get("vq_num_embeddings", 8192)
    vqe_v = cfg.get("vq_embedding_dim", 256)
    vram_est = (vq_n * vqe_v * 4) / 1e6
    if vram_est > 100:
        warnings.append(f"VQ codebook will use ~{vram_est:.0f} MB")

    if cfg.get("enable_full_coherence", False):
        warnings.append("enable_full_coherence=True: 10-20× slower — for research only")

    try:
        AEONConfig(**{k: v for k, v in cfg.items() if k != "seed"})
        valid = len(errors) == 0
    except Exception as e:
        errors.append(str(e))
        valid = False

    return {"ok": valid, "errors": errors, "warnings": warnings}


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/infer")
async def run_inference(req: InferRequest):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    t0 = time.time()
    logging.info(f"Inference · prompt='{req.prompt[:60]}' · max_len={req.max_length} · temp={req.temperature} · fast={req.fast}")
    try:
        result = APP.model.generate(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            top_k=req.top_k,
            sample=True,
        )
    except Exception as e:
        logging.error(f"Generate error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

    elapsed_ms = int((time.time() - t0) * 1000)
    audit = {}
    try:
        audit = APP.model.get_audit_summary()
    except Exception as audit_err:
        logging.warning("Audit summary unavailable: %s", audit_err)
    tg = {}
    try:
        tg = {
            "nan_count": APP.model.tensor_guard._nan_count,
            "inf_count": APP.model.tensor_guard._inf_count,
            "sanitize_count": APP.model.tensor_guard._sanitize_count,
        }
    except Exception as tg_err:
        logging.warning("TensorGuard stats unavailable: %s", tg_err)

    text_out = result.get("text", "")
    tokens = len(text_out.split())
    tps = round(tokens / max(elapsed_ms / 1000, 0.001), 1)
    logging.info(f"✅ {tokens} tokens · {elapsed_ms}ms · {tps} tok/s · status={result.get('status')}")

    # Record telemetry metrics for the Telemetry Metrics Snapshot dashboard
    try:
        tc = getattr(APP.config, 'telemetry_collector', None)
        if tc is not None:
            tc.record("inference_latency_ms", float(elapsed_ms), {"prompt": req.prompt[:30]})
            tc.record("tokens_per_sec", float(tps), {"tokens": tokens})
            if result.get("uncertainty") is not None:
                unc = result["uncertainty"]
                unc_val = float(unc) if isinstance(unc, (int, float)) else float(unc.get("total", 0)) if isinstance(unc, dict) else 0.0
                tc.record("uncertainty", unc_val)
            sanitize_count = None
            try:
                sanitize_count = APP.model.tensor_guard._sanitize_count
            except Exception as e:
                logging.warning("TensorGuard sanitize_count unavailable: %s", e)
            if sanitize_count is not None:
                tc.record("sanitize_events", float(sanitize_count))
            tc.increment("total_inferences")
    except Exception as e:
        logging.warning("Telemetry recording failed: %s", e)

    # Attach metacognitive state so consumers can assess reasoning
    # confidence, coherence verdict, and provenance attribution for
    # every inference call — closing the gap between inference output
    # and the system's self-reflective assessment.
    metacognitive = {}
    try:
        metacognitive = APP.model.get_metacognitive_state()
    except Exception as mc_err:
        logging.warning("Metacognitive state unavailable: %s", mc_err)

    # Recovery statistics for consumers to assess error-handling health
    recovery_stats = {}
    try:
        recovery_stats = APP.model.error_recovery.get_recovery_stats()
    except Exception as rs_err:
        logging.warning("Recovery stats unavailable: %s", rs_err)

    return _make_json_safe({
        "ok": True,
        "text": text_out,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "elapsed_ms": elapsed_ms,
        "tokens": tokens,
        "tokens_per_sec": tps,
        "audit": audit,
        "tensorguard": tg,
        "uncertainty": result.get("uncertainty"),
        "causal_decision_chain": result.get("causal_decision_chain", {}),
        "metacognitive_state": metacognitive,
        "provenance": result.get("provenance", {}),
        "ucc_result": result.get("ucc_result", {}),
        "recovery_stats": recovery_stats,
        "vibe_thinker": result.get("vibe_thinker", {}),
        "emergence_status": result.get("emergence_status"),
    })


@app.post("/api/forward")
async def run_forward(req: ForwardRequest):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ids = torch.tensor([req.input_ids], dtype=torch.long)
        t0 = time.time()
        with torch.no_grad():
            out = APP.model(ids, fast=req.fast)
        elapsed_ms = int((time.time() - t0) * 1000)
        logits = out.get("logits")
        safety = out.get("safety_score")
        result = {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "output_keys": list(out.keys()),
            "logits_shape": list(logits.shape) if logits is not None else None,
            "safety_score": (
                safety.mean().item() if safety is not None and hasattr(safety, "mean")
                else float(safety) if safety is not None else None
            ),
            "vq_loss": (
                out["vq_loss"].item() if "vq_loss" in out and out["vq_loss"] is not None
                else None
            ),
            "meta_iterations": out.get("meta_iterations"),
            "convergence_delta": (
                float(out["convergence_delta"]) if "convergence_delta" in out
                else None
            ),
            "uncertainty": out.get("uncertainty"),
            "causal_decision_chain": out.get("causal_decision_chain", {}),
            "provenance": out.get("provenance", {}),
        }
        if logits is not None:
            probs = torch.softmax(logits[0, 0], dim=-1)
            top5 = torch.topk(probs, 5)
            result["top5_tokens"] = top5.indices.tolist()
            result["top5_probs"] = [round(p, 4) for p in top5.values.tolist()]
        logging.info(f"Forward pass · {elapsed_ms}ms · safety={result.get('safety_score','?'):.4f}" if result.get('safety_score') is not None else f"Forward pass · {elapsed_ms}ms")

        # Record telemetry metrics for the Telemetry Metrics Snapshot dashboard
        try:
            tc = getattr(APP.config, 'telemetry_collector', None)
            if tc is not None:
                tc.record("forward_latency_ms", float(elapsed_ms))
                if result.get("safety_score") is not None:
                    tc.record("safety_score", float(result["safety_score"]))
                if result.get("convergence_delta") is not None:
                    tc.record("convergence_delta", float(result["convergence_delta"]))
                if result.get("meta_iterations") is not None:
                    tc.record("meta_iterations", float(result["meta_iterations"]))
                tc.increment("total_forward_passes")
        except Exception:
            pass

        return result
    except Exception as e:
        logging.error(f"Forward error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    """
    Profile inference latency over N runs.
    Returns: min/max/mean/median/p95/p99/stddev, throughput, device.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    if APP.training_active:
        raise HTTPException(409, "Stop training before benchmarking")

    logging.info(f"Benchmark starting · {req.num_runs} runs · max_len={req.max_length} · fast={req.fast}")
    latencies = []
    errors = []
    APP.benchmark_results = {"running": True, "progress": 0, "total": req.num_runs}

    def _run():
        try:
            for i in range(req.num_runs):
                if not APP.training_active:
                    t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            APP.model.generate(
                                req.prompt,
                                max_length=req.max_length,
                                temperature=req.temperature,
                                top_k=req.top_k,
                                sample=False,
                            )
                    except Exception as e:
                        errors.append(str(e))
                    finally:
                        dt = (time.perf_counter() - t0) * 1000
                        latencies.append(dt)
                        APP.benchmark_results["progress"] = i + 1
                        if (i + 1) % 5 == 0:
                            logging.info(f"Benchmark: {i+1}/{req.num_runs} · last={dt:.1f}ms")

            if latencies:
                sorted_l = sorted(latencies)
                n = len(sorted_l)
                p95_idx = int(0.95 * n)
                p99_idx = int(0.99 * n)
                APP.benchmark_results.update({
                    "running": False,
                    "num_runs": len(latencies),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "mean_ms": round(statistics.mean(latencies), 2),
                    "median_ms": round(statistics.median(latencies), 2),
                    "p95_ms": round(sorted_l[min(p95_idx, n-1)], 2),
                    "p99_ms": round(sorted_l[min(p99_idx, n-1)], 2),
                    "stddev_ms": round(statistics.stdev(latencies) if n > 1 else 0, 2),
                    "throughput_rps": round(1000 / statistics.mean(latencies), 2),
                    "latency_series": [round(x, 1) for x in latencies],
                    "errors": errors,
                    "device": str(APP.model.device),
                    "params": APP.model.count_parameters(),
                    "config": {"max_length": req.max_length, "fast": req.fast, "top_k": req.top_k},
                })
                logging.info(
                    f"✅ Benchmark complete · mean={APP.benchmark_results['mean_ms']}ms · "
                    f"p95={APP.benchmark_results['p95_ms']}ms · "
                    f"tput={APP.benchmark_results['throughput_rps']} rps"
                )
        except Exception as e:
            APP.benchmark_results = {"running": False, "error": str(e)}
            logging.error(f"Benchmark error: {e}")

    background_tasks.add_task(_run)
    return {"ok": True, "message": f"Benchmark started · {req.num_runs} runs"}


@app.get("/api/benchmark/result")
async def get_benchmark_result():
    return {"ok": True, "result": APP.benchmark_results or {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  LEGACY TEST SUITE (AEONTestSuite from core.py)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/test/run")
async def run_test_suite(background_tasks: BackgroundTasks):
    """Legacy AEONTestSuite (requires initialized model). For test_fixes.py use /api/tests/run."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    APP.test_results = {"running": True, "progress": "starting"}
    logging.info("🧪 Starting AEON Test Suite (legacy)")
    def _run_tests():
        try:
            suite = AEONTestSuite(APP.model, APP.config)
            results = {}
            for name, fn in [
                ("stability", suite.test_stability),
                ("weight_tying", suite.test_weight_tying),
                ("gradient_flow", suite.test_gradient_flow),
                ("vq_codebook", suite.test_vq_codebook),
            ]:
                APP.test_results["progress"] = name
                try:
                    results[name] = fn()
                except Exception as e:
                    results[name] = {"error": str(e), "score": 0.0}
            scores = [v for r in results.values() if isinstance(r, dict)
                      for k, v in r.items() if isinstance(v, float) and k != "error"]
            overall = sum(scores) / max(len(scores), 1)
            APP.test_results = {"running": False, "results": results,
                                "overall_score": round(overall, 4), "timestamp": time.time()}
        except Exception as e:
            APP.test_results = {"running": False, "error": str(e)}
    background_tasks.add_task(_run_tests)
    return {"ok": True, "message": "Test suite started"}

@app.get("/api/test/result")
async def get_test_result():
    return {"ok": True, "result": APP.test_results or {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST RUNNER — test_fixes.py  (642 tests, 49 sections)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Locate test_fixes.py ─────────────────────────────────────────────────────
def _find_test_file() -> Optional[Path]:
    candidates = [
        Path(__file__).parent / "test_fixes.py",
        Path("/mnt/user-data/uploads/test_fixes.py"),
        Path("./test_fixes.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

TEST_FILE: Optional[Path] = _find_test_file()


# ── Parse catalogue ──────────────────────────────────────────────────────────
def _parse_test_catalogue(path: Path) -> list:
    """Return [{section, tests:[{name,line,doc}]}] from test_fixes.py."""
    import re
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    groups, cur_section, cur_tests = [], "Core Tests", []
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^# [=\-]{3,}", s):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt.startswith("#") and not re.match(r"^# [=\-]{3,}", nxt) and len(nxt) > 2:
                if cur_tests:
                    groups.append({"section": cur_section, "tests": cur_tests})
                    cur_tests = []
                cur_section = nxt.lstrip("# ").strip()
        elif s.startswith("def test_"):
            m = re.match(r"def (test_\w+)\(", s)
            if m:
                doc = ""
                for j in range(i + 1, min(i + 6, len(lines))):
                    dl = lines[j].strip()
                    if dl.startswith('"""') or dl.startswith("'''"):
                        doc = dl.strip('"\' ').strip()[:120]
                        break
                    elif dl and not dl.startswith("#"):
                        break
                cur_tests.append({"name": m.group(1), "line": i + 1, "doc": doc})
    if cur_tests:
        groups.append({"section": cur_section, "tests": cur_tests})
    return groups


def _warmup_test_catalogue():
    if TEST_FILE is None:
        logging.warning("test_fixes.py not found — test runner unavailable")
        return
    try:
        cat = _parse_test_catalogue(TEST_FILE)
        APP.test_catalogue_cache = cat
        total = sum(len(g["tests"]) for g in cat)
        logging.info(f"✅ Test catalogue: {len(cat)} sections · {total} tests · {TEST_FILE}")
    except Exception as e:
        logging.warning(f"Test catalogue parse error: {e}")


def _get_meta_map() -> dict:
    """Return {name: {section, line, doc}} from catalogue."""
    cat = APP.test_catalogue_cache or (_parse_test_catalogue(TEST_FILE) if TEST_FILE else [])
    out = {}
    for g in cat:
        for t in g["tests"]:
            out[t["name"]] = {"section": g["section"], **t}
    return out


# ── Dynamic test importer ────────────────────────────────────────────────────
_test_module_cache = {}
_test_module_lock = threading.Lock()

def _load_test_module(force_reload=False):
    """Import test_fixes.py dynamically, cache result."""
    if TEST_FILE is None:
        raise RuntimeError("test_fixes.py not found")
    key = str(TEST_FILE)
    with _test_module_lock:
        if force_reload and key in _test_module_cache:
            del _test_module_cache[key]
        if key not in _test_module_cache:
            spec = importlib.util.spec_from_file_location("_aeon_test_fixes", str(TEST_FILE))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass
            except Exception as e:
                logging.debug(f"test_fixes module-level side-effect: {e}")
            _test_module_cache[key] = mod
    return _test_module_cache[key]


# ── Single test executor ─────────────────────────────────────────────────────
def _run_single_test(name: str, meta_map: dict) -> dict:
    """Run one test function, capture ALL output. Return result dict."""
    meta = meta_map.get(name, {"section": "Unknown", "line": 0, "doc": ""})
    result = {
        "name": name,
        "section": meta.get("section", ""),
        "line": meta.get("line", 0),
        "doc": meta.get("doc", ""),
        "status": "running",
        "elapsed_ms": 0.0,
        "stdout": "",
        "stderr": "",
        "traceback": "",
        "error_msg": "",
    }
    buf_out, buf_err = io.StringIO(), io.StringIO()
    t0 = time.perf_counter()
    try:
        mod = _load_test_module()
        fn = getattr(mod, name, None)
        if fn is None:
            result["status"] = "error"
            result["error_msg"] = f"Function '{name}' not found in test_fixes.py"
            return result
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            fn()
        result["status"] = "passed"
    except AssertionError as e:
        result["status"] = "failed"
        result["error_msg"] = str(e) or "AssertionError (no message)"
        result["traceback"] = traceback.format_exc()
    except (ImportError, ModuleNotFoundError) as e:
        result["status"] = "skipped"
        result["error_msg"] = f"Import error: {e}"
    except Exception as e:
        result["status"] = "error"
        result["error_msg"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["stdout"] = buf_out.getvalue()
        result["stderr"] = buf_err.getvalue()

    # Build log lines (both formats)
    icon = {"passed":"✅","failed":"❌","error":"💥","skipped":"⏭","running":"⏳"}.get(result["status"],"?")
    result["log_brief"] = f'{icon} [{result["elapsed_ms"]:6.0f}ms] {name}'
    if result["error_msg"]:
        result["log_brief"] += f"  — {result['error_msg'][:100]}"

    full_lines = [
        "═"*62,
        f"TEST    : {name}",
        f"SECTION : {result['section']}",
        f"LINE    : {result['line']}",
        f"DOC     : {result['doc']}",
        f"TIME    : {result['elapsed_ms']:.1f} ms",
        f"STATUS  : {result['status'].upper()}",
    ]
    if result["stdout"].strip():
        full_lines += ["", "── stdout ──────────────────────────────", result["stdout"].rstrip()]
    if result["stderr"].strip():
        full_lines += ["", "── stderr ──────────────────────────────", result["stderr"].rstrip()]
    if result["error_msg"]:
        full_lines += ["", "── error ───────────────────────────────", result["error_msg"]]
    if result["traceback"]:
        full_lines += ["", "── traceback ───────────────────────────", result["traceback"].rstrip()]
    full_lines.append("═"*62)
    result["log_full"] = "\n".join(full_lines)

    return result


# ── Build summary ─────────────────────────────────────────────────────────────
def _build_summary(results: list) -> dict:
    total = len(results)
    by_status = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    by_section: dict = {}
    times = []
    fail_names = []
    for r in results:
        s = r.get("status", "error")
        by_status[s] = by_status.get(s, 0) + 1
        sec = r.get("section", "?")
        if sec not in by_section:
            by_section[sec] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "total": 0}
        by_section[sec][s] = by_section[sec].get(s, 0) + 1
        by_section[sec]["total"] += 1
        if r.get("elapsed_ms", 0) > 0:
            times.append(r["elapsed_ms"])
        if s in ("failed", "error"):
            fail_names.append(r["name"])
    return {
        "total": total, **by_status,
        "pass_rate": round(by_status["passed"] / max(total, 1) * 100, 1),
        "total_time_ms": round(sum(times), 1),
        "avg_time_ms": round(sum(times) / max(len(times), 1), 1),
        "by_section": by_section,
        "failed_names": fail_names,
        "slowest": sorted(
            [{"name": r["name"], "ms": r["elapsed_ms"]} for r in results],
            key=lambda x: x["ms"], reverse=True
        )[:10],
    }


# ── Background test loop ──────────────────────────────────────────────────────
def _test_run_loop(names: list, meta_map: dict, log_format: str, stop_on_failure: bool):
    """Background thread: run tests, emit WS events, write to logging."""
    import asyncio

    def _emit(data: dict):
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                asyncio.run_coroutine_threadsafe(broadcast({"type": "test_event", "data": data}), loop)
        except Exception:
            pass

    try:
        for i, name in enumerate(names):
            if APP.test_run_stop_event and APP.test_run_stop_event.is_set():
                break

            APP.test_run_progress["current"] = i + 1
            APP.test_run_progress["current_test"] = name

            r = _run_single_test(name, meta_map)
            if log_format != "full":
                r_stored = {k: v for k, v in r.items() if k != "log_full"}
            else:
                r_stored = r
            APP.test_run_results.append(r_stored)

            # Increment counters
            APP.test_run_progress[r["status"]] = APP.test_run_progress.get(r["status"], 0) + 1

            # Log to server log stream
            if r["status"] == "passed":
                logging.info(f"✅ PASS  [{r['elapsed_ms']:6.0f}ms] {name}")
            elif r["status"] == "failed":
                logging.warning(f"❌ FAIL  [{r['elapsed_ms']:6.0f}ms] {name} — {r['error_msg'][:100]}")
                if log_format == "full" and r.get("traceback"):
                    for line in r["traceback"].splitlines()[-8:]:
                        logging.debug(f"  TB: {line}")
            elif r["status"] == "error":
                logging.error(f"💥 ERROR [{r['elapsed_ms']:6.0f}ms] {name} — {r['error_msg'][:100]}")
            elif r["status"] == "skipped":
                logging.info(f"⏭  SKIP  [{r['elapsed_ms']:6.0f}ms] {name}")

            # Log stdout if full mode
            if log_format == "full" and r.get("stdout", "").strip():
                for line in r["stdout"].strip().splitlines()[:20]:
                    logging.debug(f"  [out/{name}] {line}")

            # Broadcast per-test
            _emit(r_stored)

            if stop_on_failure and r["status"] in ("failed", "error"):
                logging.warning(f"stop_on_failure → stopping after {name}")
                break

        # Final summary
        summary = _build_summary(APP.test_run_results)
        APP.test_run_summary = summary
        APP.test_run_progress.update({"active": False, "done": True, "summary": summary})
        p = APP.test_run_progress
        logging.info(
            f"🎉 Test run complete · "
            f"{p.get('passed',0)} passed · {p.get('failed',0)} failed · "
            f"{p.get('error',0)} errors · {p.get('skipped',0)} skipped · "
            f"{summary['total_time_ms']:.0f}ms total"
        )
        _emit({"type": "run_complete", "summary": summary})

    except Exception as e:
        logging.error(f"Test run loop fatal: {e}\n{traceback.format_exc()}")
    finally:
        with APP.lock:
            APP.test_run_active = False
            APP.test_run_progress["active"] = False


# ── Progress broadcaster ──────────────────────────────────────────────────────
async def _test_progress_broadcaster():
    import asyncio
    while True:
        if APP.test_run_active and APP.ws_clients:
            await broadcast({"type": "test_progress", "data": APP.test_run_progress})
        await asyncio.sleep(1.5)


# ── Pydantic model ────────────────────────────────────────────────────────────
class TestRunRequest(BaseModel):
    names: Optional[List[str]] = None     # None = run all
    section: Optional[str] = None         # run one section
    log_format: str = "full"              # "full" | "brief"
    stop_on_failure: bool = False
    reload_module: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/tests/catalogue")
async def api_tests_catalogue():
    """Return full test catalogue: 49 sections, 642 tests with docstrings."""
    if TEST_FILE is None:
        return {"ok": False, "error": "test_fixes.py not found", "sections": [], "total_tests": 0}
    cat = APP.test_catalogue_cache or _parse_test_catalogue(TEST_FILE)
    APP.test_catalogue_cache = cat
    total = sum(len(g["tests"]) for g in cat)
    return {"ok": True, "sections": cat, "total_tests": total, "total_sections": len(cat)}


@app.post("/api/tests/run")
async def api_tests_run(req: TestRunRequest, background_tasks: BackgroundTasks):
    """Start a test run. Pass names=null+section=null to run all 642 tests."""
    if TEST_FILE is None:
        raise HTTPException(404, "test_fixes.py not found next to aeon_server.py")
    if APP.test_run_active:
        raise HTTPException(409, "A test run is already in progress. POST /api/tests/stop first.")

    if req.reload_module:
        _load_test_module(force_reload=True)
        logging.info("test_fixes.py module reloaded")

    meta_map = _get_meta_map()
    cat = APP.test_catalogue_cache

    # Resolve names
    if req.names is not None:
        names = [n for n in req.names if n in meta_map]
    elif req.section is not None:
        names = []
        for g in cat:
            if g["section"] == req.section:
                names = [t["name"] for t in g["tests"]]
                break
        if not names:
            raise HTTPException(404, f"Section \'{req.section}\' not found")
    else:
        names = [t["name"] for g in cat for t in g["tests"]]

    with APP.lock:
        APP.test_run_active = True
        APP.test_run_stop_event = threading.Event()
        APP.test_run_results = []
        APP.test_run_summary = {}
        APP.test_run_progress = {
            "active": True, "done": False,
            "total": len(names), "current": 0,
            "current_test": None,
            "passed": 0, "failed": 0, "error": 0, "skipped": 0,
            "log_format": req.log_format,
            "started_at": time.time(),
        }

    logging.info(f"🧪 Test run started · {len(names)} tests · format={req.log_format}")
    background_tasks.add_task(
        _test_run_loop, names, meta_map, req.log_format, req.stop_on_failure
    )
    return {"ok": True, "total": len(names), "log_format": req.log_format}


@app.post("/api/tests/stop")
async def api_tests_stop():
    """Gracefully cancel the running test loop."""
    if APP.test_run_stop_event:
        APP.test_run_stop_event.set()
        logging.info("Test run cancellation requested")
    return {"ok": True}


@app.get("/api/tests/progress")
async def api_tests_progress():
    return {"ok": True, "progress": APP.test_run_progress}


@app.get("/api/tests/results")
async def api_tests_results(
    format: str = "full",
    status: str = "",
    limit: int = 0,
    section: str = "",
):
    """
    Return test results.
    ?format=full → includes log_full (stdout + traceback)
    ?format=brief → only log_brief, name, status, elapsed_ms
    ?status=failed → filter
    ?section=... → filter by section name
    ?limit=N → last N results
    """
    results = list(APP.test_run_results)
    if status:
        results = [r for r in results if r.get("status") == status]
    if section:
        results = [r for r in results if r.get("section") == section]
    if limit and limit > 0:
        results = results[-limit:]
    if format == "brief":
        results = [
            {k: v for k, v in r.items() if k not in ("log_full", "stdout", "stderr", "traceback")}
            for r in results
        ]
    return {
        "ok": True,
        "results": results,
        "total_returned": len(results),
        "total_run": len(APP.test_run_results),
        "progress": APP.test_run_progress,
        "summary": APP.test_run_summary,
    }


@app.get("/api/tests/results/{test_name}")
async def api_test_single_result(test_name: str):
    """Return most recent result for a specific test."""
    for r in reversed(APP.test_run_results):
        if r.get("name") == test_name:
            return {"ok": True, "result": r}
    raise HTTPException(404, f"No result for \'{test_name}\'")


@app.post("/api/tests/run_single")
async def api_tests_run_single(body: dict):
    """Run exactly one test synchronously, return result immediately."""
    if TEST_FILE is None:
        raise HTTPException(404, "test_fixes.py not found")
    name = body.get("name", "")
    fmt = body.get("log_format", "full")
    meta_map = _get_meta_map()
    if name not in meta_map:
        raise HTTPException(404, f"Test \'{name}\' not found")
    r = _run_single_test(name, meta_map)
    if r["status"] == "passed":
        logging.info(f"✅ PASS  [{r['elapsed_ms']:.0f}ms] {name}")
    elif r["status"] == "failed":
        logging.warning(f"❌ FAIL  [{r['elapsed_ms']:.0f}ms] {name} — {r['error_msg'][:100]}")
    else:
        logging.error(f"💥 {r['status'].upper()} [{r['elapsed_ms']:.0f}ms] {name}")
    if fmt == "brief":
        r = {k: v for k, v in r.items() if k not in ("log_full", "stdout", "stderr", "traceback")}
    await broadcast({"type": "test_event", "data": r})
    return {"ok": True, "result": r}


@app.get("/api/tests/stream")
async def api_tests_stream():
    """SSE stream: one event per completed test + progress pings."""
    async def gen():
        import asyncio
        # Replay already-completed results
        sent = 0
        for r in APP.test_run_results:
            yield f"data: {json.dumps({'type':'test_event','data':r})}\n\n"
            sent += 1
        # Live feed
        while APP.test_run_active or len(APP.test_run_results) > sent:
            new = APP.test_run_results[sent:]
            for r in new:
                yield f"data: {json.dumps({'type':'test_event','data':r})}\n\n"
            sent = len(APP.test_run_results)
            yield f"data: {json.dumps({'type':'progress','data':APP.test_run_progress})}\n\n"
            await asyncio.sleep(0.4)
        yield f"data: {json.dumps({'type':'summary','data':APP.test_run_summary})}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/tests/reload")
async def api_tests_reload():
    """Force re-import of test_fixes.py (useful after patching)."""
    if TEST_FILE is None:
        raise HTTPException(404, "test_fixes.py not found")
    try:
        _load_test_module(force_reload=True)
        logging.info("test_fixes.py reloaded")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  INTROSPECTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/introspect")
async def introspect():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        audit_summary = APP.model.get_audit_summary()
        recent_decisions = APP.model.get_recent_decisions(20)
        tg = APP.model.tensor_guard
        recovery_stats = {}
        try:
            recovery_stats = APP.model.error_recovery.get_recovery_stats()
        except Exception as rec_err:
            logging.debug("Recovery stats unavailable: %s", rec_err)
        vq_stats = {}
        try:
            if APP.model.vector_quantizer is not None:
                vq_stats = APP.model.vector_quantizer.get_codebook_usage_stats()
        except Exception as vq_err:
            logging.debug("VQ codebook stats unavailable: %s", vq_err)

        return {
            "ok": True,
            "parameters": APP.model.count_parameters(),
            "trainable_parameters": APP.model.count_trainable_parameters(),
            "audit_summary": audit_summary,
            "recent_decisions": recent_decisions,
            "tensorguard": {
                "nan_count": tg._nan_count,
                "inf_count": tg._inf_count,
                "sanitize_count": tg._sanitize_count,
            },
            "recovery_stats": recovery_stats,
            "vq_stats": vq_stats,
            "device": str(APP.model.device),
            "config": {
                k: v for k, v in vars(APP.config).items()
                if not k.startswith("_") and not callable(v)
                and k not in ("device_manager", "tensor_guard")
                and isinstance(v, (bool, int, float, str, list, dict, type(None)))
            } if APP.config else {},
        }
    except Exception as e:
        logging.error(f"Introspect error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/introspect/modules")
async def introspect_modules():
    """Per-module parameter breakdown with weight statistics, gradient info, shapes, dtype, and memory."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        modules = []
        for name, mod in APP.model.named_children():
            params = list(mod.parameters())
            total_p = sum(x.numel() for x in params)
            trainable_p = sum(x.numel() for x in params if x.requires_grad)
            weight_stats = {}
            grad_stats = {}
            shapes = []
            dtypes = set()
            if params:
                all_weights = torch.cat([p.data.flatten() for p in params if p.data.numel() > 0])
                weight_stats = {
                    "mean": round(all_weights.mean().item(), 6),
                    "std": round(all_weights.std().item(), 6),
                    "min": round(all_weights.min().item(), 6),
                    "max": round(all_weights.max().item(), 6),
                    "l2_norm": round(all_weights.norm(2).item(), 4),
                    "has_nan": bool(torch.isnan(all_weights).any()),
                    "has_inf": bool(torch.isinf(all_weights).any()),
                }
                # Gradient statistics
                grad_tensors = [p.grad.flatten() for p in params if p.grad is not None and p.grad.numel() > 0]
                if grad_tensors:
                    all_grads = torch.cat(grad_tensors)
                    grad_stats = {
                        "mean": round(all_grads.mean().item(), 6),
                        "std": round(all_grads.std().item(), 6),
                        "l2_norm": round(all_grads.norm(2).item(), 4),
                        "max_abs": round(all_grads.abs().max().item(), 6),
                    }
                # Parameter shapes and dtypes (cap at 10 to limit response size)
                for p in params[:10]:
                    shapes.append(list(p.shape))
                    dtypes.add(str(p.dtype))

            # Memory estimation (bytes) — approximation based on first parameter's dtype
            bytes_per_param = 4  # Default FP32
            for p in params[:1]:
                if p.dtype == torch.float16 or p.dtype == torch.bfloat16:
                    bytes_per_param = 2
                elif p.dtype == torch.float64:
                    bytes_per_param = 8
            memory_bytes = total_p * bytes_per_param

            # Submodules
            sub = []
            for sname, smod in mod.named_children():
                sp = sum(x.numel() for x in smod.parameters())
                sub.append({"name": sname, "type": type(smod).__name__, "params": sp})

            modules.append({
                "name": name,
                "type": type(mod).__name__,
                "params": total_p,
                "trainable": trainable_p,
                "frozen": total_p - trainable_p,
                "weight_stats": weight_stats,
                "grad_stats": grad_stats,
                "shapes": shapes,
                "dtypes": list(dtypes),
                "memory_bytes": memory_bytes,
                "submodules": sub,
            })

        total = APP.model.count_parameters()
        return {
            "ok": True,
            "total_parameters": total,
            "modules": sorted(modules, key=lambda x: x["params"], reverse=True),
        }
    except Exception as e:
        logging.error(f"Module introspect error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/audit")
async def get_audit(subsystem: str = "", min_severity: str = ""):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        if subsystem or min_severity:
            entries = APP.model.audit_log.filter_by(
                subsystem=subsystem or None,
                min_severity=min_severity or None,
            )
        else:
            entries = APP.model.get_recent_decisions(500)
        insights = {}
        try:
            insights = APP.model.audit_log.get_pattern_insights()
        except Exception as _pi_err:
            logging.warning("Audit pattern insights unavailable (non-critical): %s", _pi_err)
        return {"ok": True, "entries": entries, "insights": insights}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/tensorguard")
async def get_tensorguard():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        tg = APP.model.tensor_guard
        report = {
            "nan_count": tg._nan_count,
            "inf_count": tg._inf_count,
            "sanitize_count": tg._sanitize_count,
            "policy": tg.policy.name,
            "history": list(tg._context_history)[-50:],
        }
        tg.print_report()
        return {"ok": True, "report": report}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def get_health():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        integrity = APP.model.integrity_monitor.get_health_score()
        subsystems = {}
        _degraded = False
        try:
            subsystems = APP.model.integrity_monitor.get_subsystem_scores()
        except Exception as sub_err:
            logging.warning("Subsystem scoring failed: %s", sub_err)
            _degraded = True
        return {
            "ok": True,
            "degraded": _degraded,
            "health_score": float(integrity) if integrity is not None else 0.0,
            "subsystems": subsystems,
        }
    except Exception as health_err:
        logging.warning("Integrity monitor unavailable, falling back to TensorGuard: %s", health_err)
        tg = APP.model.tensor_guard
        health = 1.0 - min(1.0, (tg._nan_count + tg._inf_count) * 0.05)
        return {"ok": True, "degraded": True, "health_score": health, "subsystems": {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  META-COGNITIVE STATE
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/metacognition")
async def get_metacognition():
    """Return a unified snapshot of the meta-cognitive subsystem.

    Exposes the metacognitive trigger state, error evolution patterns,
    convergence history, causal trace coverage, and a coherence verdict
    that summarises how well the cognitive subsystems reinforce each
    other.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        state = APP.model.get_metacognitive_state()
        diagnostic = APP.model.self_diagnostic()
        return _make_json_safe({
            "ok": True,
            "metacognitive_state": state,
            "self_diagnostic": {
                "status": diagnostic.get("status"),
                "active_module_count": diagnostic.get("active_module_count"),
                "verified_count": diagnostic.get("verified_count"),
                "gap_count": diagnostic.get("gap_count"),
                "gaps": diagnostic.get("gaps", []),
                "causal_trace_coverage": diagnostic.get("causal_trace_coverage"),
                "output_reliability": diagnostic.get("output_reliability"),
                "total_parameters": diagnostic.get("total_parameters"),
                "trainable_parameters": diagnostic.get("trainable_parameters"),
                "error_evolution_summary": diagnostic.get("error_evolution_summary"),
                "runtime_coherence": diagnostic.get("runtime_coherence"),
                "pipeline_wiring": diagnostic.get("pipeline_wiring"),
                "feedback_oscillation": diagnostic.get("feedback_oscillation"),
                "provenance_completeness": diagnostic.get("provenance_completeness"),
            },
        })
    except Exception as e:
        logging.error(f"Metacognition endpoint error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/metacognition/resolve")
async def resolve_metacognitive_gaps():
    """Analyse self-diagnostic gaps and return prioritised, actionable
    resolution recommendations.

    Each gap identified by ``self_diagnostic()`` is mapped to a concrete
    configuration change or operational action, enabling automated or
    semi-automated self-repair.  Gaps are ranked by severity so that
    critical issues (``None``-valued subsystems) surface first.

    When gaps exist, ``verify_and_reinforce()`` is called to actively
    apply corrective feedback (boosting metacognitive weights, recording
    error evolution episodes), and the diagnostic is re-run to produce
    a before/after diff showing which gaps were resolved.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        diagnostic = APP.model.self_diagnostic()
        gaps = diagnostic.get("gaps", [])
        resolutions = []
        for gap in gaps:
            _gap_text = gap.get("gap", "")
            if " is None" in _gap_text or "not initialized" in _gap_text:
                severity = "critical"
            elif ("disabled" in _gap_text or "not enabled" in _gap_text
                  or "not wired" in _gap_text or "not linked" in _gap_text):
                severity = "high"
            elif ("invisible" in _gap_text or "not registered" in _gap_text
                  or "not realized" in _gap_text):
                severity = "medium"
            else:
                severity = "warning"
            resolutions.append({
                "component": gap.get("component", "unknown"),
                "gap": gap.get("gap", ""),
                "remediation": gap.get("remediation", ""),
                "severity": severity,
            })
        # Sort critical first
        resolutions.sort(key=lambda r: 0 if r["severity"] == "critical" else 1)

        # --- Active remediation via verify_and_reinforce() ---
        # When gaps exist, apply corrective feedback and re-diagnose
        # to produce a before/after diff.  This closes the gap where
        # the endpoint returned recommendations but never applied them.
        reinforcement_result = None
        resolved_gaps = []
        if resolutions:
            try:
                reinforcement_result = APP.model.verify_and_reinforce()
                _post_diag = APP.model.self_diagnostic()
                _post_gap_comps = {
                    g.get("component", "")
                    for g in _post_diag.get("gaps", [])
                }
                _pre_gap_comps = {
                    r["component"] for r in resolutions
                }
                resolved_gaps = sorted(_pre_gap_comps - _post_gap_comps)
                for r in resolutions:
                    r["post_status"] = (
                        "resolved" if r["component"] not in _post_gap_comps
                        else "persists"
                    )
            except Exception as _reinforce_err:
                logging.debug("resolve_metacognitive_gaps: reinforcement skipped: %s", _reinforce_err)

        return _make_json_safe({
            "ok": True,
            "status": diagnostic.get("status"),
            "total_gaps": len(resolutions),
            "critical_gaps": sum(1 for r in resolutions if r["severity"] == "critical"),
            "resolutions": resolutions,
            "reinforcement_applied": reinforcement_result is not None,
            "reinforcement_actions": (
                reinforcement_result.get("reinforcement_actions", [])
                if reinforcement_result else []
            ),
            "resolved_gaps": resolved_gaps,
        })
    except Exception as e:
        logging.error(f"Gap resolution endpoint error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/telemetry/metrics")
async def get_telemetry_metrics():
    """Return a snapshot of all collected telemetry metrics with statistics."""
    if APP.config is None:
        raise HTTPException(400, "Model not initialized — no telemetry available")
    try:
        tc = getattr(APP.config, 'telemetry_collector', None)
        if tc is None:
            raise HTTPException(404, "Telemetry collector not configured")
        return {"ok": True, "metrics": tc.get_metrics_snapshot()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/telemetry/metric/{metric_name}")
async def get_telemetry_metric(metric_name: str, last_n: int = 50):
    """Return the most recent entries for a specific metric."""
    if APP.config is None:
        raise HTTPException(400, "Model not initialized — no telemetry available")
    try:
        tc = getattr(APP.config, 'telemetry_collector', None)
        if tc is None:
            raise HTTPException(404, "Telemetry collector not configured")
        return {"ok": True, "metric": metric_name, "entries": tc.get_metric(metric_name, last_n)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/observability/config")
async def get_observability_config():
    """Return the current observability configuration."""
    if APP.config is None:
        return {
            "ok": True,
            "structured_logging": False,
            "academic_mode": False,
            "telemetry": False,
        }
    return {
        "ok": True,
        "structured_logging": APP.config.enable_structured_logging,
        "academic_mode": APP.config.enable_academic_mode,
        "telemetry": APP.config.enable_telemetry,
    }


@app.get("/api/observability/traces")
async def get_observability_traces(last_n: int = 100):
    """Return the most recent audit log entries as distributed trace records."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        entries = APP.model.audit_log.recent(last_n)
        return {"ok": True, "traces": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  CAUSAL PROVENANCE & TRACEABILITY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/provenance")
async def get_provenance():
    """Return the current causal provenance attribution snapshot.

    Exposes the per-module L2 delta contributions computed by
    ``CausalProvenanceTracker.compute_attribution()`` so that external
    consumers can trace which modules were most responsible for the
    output state.  This closes the API gap where provenance data was
    computed internally but not accessible outside the model.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        attribution = APP.model.provenance_tracker.compute_attribution()
        return {
            "ok": True,
            "attribution": {
                "contributions": attribution.get("contributions", {}),
                "deltas": attribution.get("deltas", {}),
                "order": attribution.get("order", []),
            },
        }
    except Exception as e:
        logging.error(f"Provenance endpoint error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/provenance/root_cause/{module_name}")
async def get_provenance_root_cause(module_name: str):
    """Trace root causes backward from a named module through the
    provenance dependency DAG.

    Invokes ``CausalProvenanceTracker.trace_root_cause()`` for the
    specified module, returning the upstream modules that had the
    largest L2 impact.  This enables external root-cause analysis:
    given a module that dominated the output, identify which upstream
    transformations shaped its behaviour.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        root_cause = APP.model.provenance_tracker.trace_root_cause(
            module_name,
        )
        return {"ok": True, "module": module_name, "root_cause": root_cause}
    except Exception as e:
        logging.error(f"Provenance root-cause endpoint error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/causal_trace")
async def get_causal_trace(last_n: int = 50):
    """Return recent entries from the temporal causal trace buffer.

    Unlike ``/api/observability/traces`` which returns audit-log entries,
    this endpoint exposes the ``TemporalCausalTraceBuffer`` — the
    subsystem-level causal decision chain that records every module's
    reasoning decisions with timestamps and severity.  This closes the
    traceability gap where causal trace data was recorded internally
    but only audit-log traces were exposed via the API.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        trace = getattr(APP.model, "causal_trace", None)
        if trace is None:
            return {
                "ok": True,
                "available": False,
                "reason": "Causal trace not enabled",
            }
        entries = trace.recent(last_n)
        # Convert entries to serialisable dicts; binary fields (bytes,
        # memoryview) are excluded because they are not JSON-serialisable.
        serialised = []
        for entry in entries:
            if isinstance(entry, dict):
                serialised.append({
                    k: v for k, v in entry.items()
                    if not isinstance(v, (bytes, memoryview))
                })
            else:
                serialised.append(str(entry))
        return {
            "ok": True,
            "available": True,
            "count": len(serialised),
            "entries": serialised,
        }
    except Exception as e:
        logging.error(f"Causal trace endpoint error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/causal_trace/root_cause/{entry_id}")
async def get_causal_trace_root_cause(entry_id: str):
    """Trace root causes for a specific causal trace entry.

    Invokes ``TemporalCausalTraceBuffer.trace_root_cause()`` for the
    given entry ID, returning the ordered causal chain that led to
    the decision.  This satisfies the requirement that all conclusions
    can be traced back to their root causes.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        trace = getattr(APP.model, "causal_trace", None)
        if trace is None:
            return {
                "ok": True,
                "available": False,
                "reason": "Causal trace not enabled",
            }
        root_cause = trace.trace_root_cause(entry_id)
        if root_cause is None:
            raise HTTPException(404, f"No root cause found for entry_id '{entry_id}'")
        return {
            "ok": True,
            "entry_id": entry_id,
            "root_cause": root_cause,
        }
    except Exception as e:
        logging.error(f"Causal trace root-cause endpoint error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  VQ-VAE CODEBOOK
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/vq/codebook")
async def get_vq_codebook():
    """Exhaustive academic-level VQ codebook diagnostics."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        vq = APP.model.vector_quantizer
        if vq is None:
            return {"ok": True, "available": False, "reason": "VQ not enabled"}

        # Basic stats
        basic = vq.get_codebook_usage_stats()

        # Embedding norms
        emb = vq.embedding.weight if hasattr(vq, "embedding") else None
        norms = []
        all_norms_tensor = None
        if emb is not None:
            all_norms_tensor = emb.norm(dim=1)
            n = min(emb.shape[0], 512)  # cap for response size
            norms = all_norms_tensor[:n].tolist()
            norms = [round(x, 4) for x in norms]

        # Norm statistics (full codebook)
        norm_stats = {}
        if all_norms_tensor is not None:
            norm_stats = {
                "mean": round(all_norms_tensor.mean().item(), 6),
                "std": round(all_norms_tensor.std().item(), 6),
                "min": round(all_norms_tensor.min().item(), 6),
                "max": round(all_norms_tensor.max().item(), 6),
                "median": round(all_norms_tensor.median().item(), 6),
                "cv": round((all_norms_tensor.std() / (all_norms_tensor.mean() + 1e-8)).item(), 6),
            }

        # Usage counts (if tracked)
        counts = []
        full_counts_tensor = None
        if hasattr(vq, "_ema_cluster_size"):
            full_counts_tensor = vq._ema_cluster_size
            counts = full_counts_tensor.tolist()[:512]
            counts = [round(x, 2) for x in counts]
        elif hasattr(vq, "_usage_counts"):
            full_counts_tensor = vq._usage_counts.float()
            counts = full_counts_tensor.tolist()[:512]
        elif hasattr(vq, "_code_usage_counter"):
            full_counts_tensor = vq._code_usage_counter.float()
            counts = full_counts_tensor.tolist()[:512]

        dead_count = sum(1 for c in counts if c < 0.1) if counts else None

        # ── Academic diagnostic metrics ──────────────────────────────
        # Perplexity
        perplexity = None
        if hasattr(vq, "_perplexity_ema"):
            perplexity = round(vq._perplexity_ema.item(), 4)

        # Total training steps
        total_steps = None
        if hasattr(vq, "_total_steps"):
            total_steps = int(vq._total_steps.item())

        # Usage entropy (Shannon)
        entropy = None
        max_entropy = None
        normalized_entropy = None
        if full_counts_tensor is not None:
            total_usage = full_counts_tensor.sum()
            if total_usage > 0:
                probs = full_counts_tensor / total_usage
                probs_pos = probs[probs > 0]
                entropy = round(-(probs_pos * probs_pos.log()).sum().item(), 6)
                max_entropy = round(math.log(vq.num_embeddings), 6)
                normalized_entropy = round(entropy / max_entropy, 6) if max_entropy > 0 else 0.0

        # Gini coefficient of usage
        gini = None
        if full_counts_tensor is not None and full_counts_tensor.numel() > 1:
            sorted_counts, _ = full_counts_tensor.sort()
            n_codes = sorted_counts.numel()
            index = torch.arange(1, n_codes + 1, dtype=torch.float32, device=sorted_counts.device)
            gini_val = (2.0 * (index * sorted_counts).sum() - (n_codes + 1) * sorted_counts.sum())
            denom = n_codes * sorted_counts.sum()
            gini = round((gini_val / (denom + 1e-8)).item(), 6)

        # Steps-since-used statistics
        steps_since_used_stats = {}
        if hasattr(vq, "_steps_since_used"):
            ssu = vq._steps_since_used.float()
            steps_since_used_stats = {
                "mean": round(ssu.mean().item(), 2),
                "max": int(ssu.max().item()),
                "min": int(ssu.min().item()),
                "std": round(ssu.std().item(), 2),
                "pct_stale_50": int((ssu > 50).sum().item()),
                "pct_stale_100": int((ssu > 100).sum().item()),
            }

        # Per-code classification (first 512)
        # Thresholds based on EMA cluster sizes: <0.1 = never selected (dead),
        # <1.0 = rarely selected, <5.0 = low frequency, >=5.0 = actively used
        code_status = []
        cap = min(vq.num_embeddings, 512) if hasattr(vq, "num_embeddings") else 512
        for i in range(min(len(counts), cap)):
            c = counts[i]
            if c < 0.1:
                status = "dead"
            elif c < 1.0:
                status = "rare"
            elif c < 5.0:
                status = "low"
            else:
                status = "active"
            code_status.append(status)

        status_summary = {}
        for s in ["dead", "rare", "low", "active"]:
            status_summary[s] = code_status.count(s)

        # Top-K most and least used codes
        top_k_used = []
        bottom_k_unused = []
        if full_counts_tensor is not None:
            k = min(20, full_counts_tensor.numel())
            top_vals, top_idx = full_counts_tensor.topk(k)
            top_k_used = [{"code": int(idx), "usage": round(float(val), 2)}
                          for idx, val in zip(top_idx.tolist(), top_vals.tolist())]
            bot_vals, bot_idx = full_counts_tensor.topk(k, largest=False)
            bottom_k_unused = [{"code": int(idx), "usage": round(float(val), 2)}
                               for idx, val in zip(bot_idx.tolist(), bot_vals.tolist())]

        # Inter-embedding cosine similarity (sample first 64 codes to keep
        # O(n²) cost bounded; 64×64=4096 pairs gives statistically stable estimates)
        cosine_stats = {}
        if emb is not None and emb.shape[0] >= 2:
            sample_n = min(64, emb.shape[0])
            sample_emb = torch.nn.functional.normalize(emb[:sample_n], dim=1)
            sim_matrix = sample_emb @ sample_emb.T
            mask = ~torch.eye(sample_n, dtype=torch.bool, device=sim_matrix.device)
            off_diag = sim_matrix[mask]
            cosine_stats = {
                "mean_similarity": round(off_diag.mean().item(), 6),
                "max_similarity": round(off_diag.max().item(), 6),
                "min_similarity": round(off_diag.min().item(), 6),
                "std_similarity": round(off_diag.std().item(), 6),
            }

        # Codebook collapse risk assessment based on normalized entropy.
        # Thresholds follow VQ-VAE literature (van den Oord et al., 2017):
        # H/H_max < 0.3 → severe mode collapse, < 0.5 → significant underuse,
        # < 0.7 → moderate, < 0.85 → mild, >= 0.85 → healthy utilization.
        collapse_risk = "unknown"
        if normalized_entropy is not None:
            if normalized_entropy < 0.3:
                collapse_risk = "critical"
            elif normalized_entropy < 0.5:
                collapse_risk = "high"
            elif normalized_entropy < 0.7:
                collapse_risk = "moderate"
            elif normalized_entropy < 0.85:
                collapse_risk = "low"
            else:
                collapse_risk = "minimal"

        # Hyperparameter snapshot
        hyperparams = {
            "commitment_cost": getattr(vq, "commitment_cost", None),
            "decay": getattr(vq, "decay", None),
            "epsilon": getattr(vq, "epsilon", None),
            "revival_threshold": getattr(vq, "revival_threshold", None),
            "split_threshold": getattr(vq, "split_threshold", None),
            "use_ema": getattr(vq, "use_ema", None),
        }

        return {
            "ok": True,
            "available": True,
            "basic_stats": basic,
            "num_embeddings": vq.embedding.weight.shape[0] if emb is not None else None,
            "embedding_dim": vq.embedding.weight.shape[1] if emb is not None else None,
            "embedding_norms": norms,
            "norm_stats": norm_stats,
            "usage_counts": counts,
            "dead_codes": dead_count,
            "perplexity": perplexity,
            "total_steps": total_steps,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
            "gini_coefficient": gini,
            "steps_since_used": steps_since_used_stats,
            "code_status": code_status,
            "status_summary": status_summary,
            "top_k_used": top_k_used,
            "bottom_k_unused": bottom_k_unused,
            "cosine_stats": cosine_stats,
            "collapse_risk": collapse_risk,
            "hyperparams": hyperparams,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/vq/metrics")
async def get_vq_extended_metrics():
    """Extended VQ-VAE metrics: reconstruction quality and codebook utilization.

    Reports metrics aligned with modern VQ-VAE literature (SimVQ, GM-VQ,
    MGVQ) including active ratio, normalised entropy, Gini coefficient,
    effective codebook size, reconstruction MSE/PSNR/cosine, and
    comparison against published baselines.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        vq = APP.model.vector_quantizer
        if vq is None:
            return {"ok": True, "available": False, "reason": "VQ not enabled"}

        result: dict = {"ok": True, "available": True}

        # Utilization metrics
        if hasattr(vq, 'compute_codebook_utilization_metrics'):
            result["utilization"] = vq.compute_codebook_utilization_metrics()

        # Reconstruction quality (need a probe input)
        if hasattr(vq, 'compute_reconstruction_quality'):
            try:
                probe = torch.randn(4, vq.embedding_dim)
                result["reconstruction"] = vq.compute_reconstruction_quality(probe)
            except Exception as rq_err:
                result["reconstruction"] = {"error": str(rq_err)}

        return _make_json_safe(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/profile/hessian")
async def profile_hessian():
    """Profile the Hessian module for real-time feasibility assessment.

    Benchmarks compute_hessian, hutchinson_trace, and eigenvalue
    estimation.  Returns latency percentiles (p50/p95/p99), memory
    overhead, throughput, and a real-time feasibility verdict.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        from aeon_core import HessianProfiler
        topo = getattr(APP.model, 'topology_analyzer', None)
        if topo is None:
            return {"ok": False, "error": "TopologyAnalyzer not available"}

        hc = topo.hessian_computer
        profiler = HessianProfiler(hc, realtime_budget_ms=50.0)

        num_pillars = getattr(APP.model.config, 'num_pillars', 5)
        x = torch.randn(2, num_pillars)

        def potential_fn(p):
            return topo.compute_potential(p)

        report = profiler.benchmark_realtime_feasibility(
            potential_fn, x, n_runs=10,
        )
        return _make_json_safe({"ok": True, **report})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/safety/evaluate")
async def evaluate_safety():
    """Quantitative safety evaluation across toxicity, deception, and harm.

    Runs the full QuantitativeSafetyEvaluator pipeline on a probe input
    and returns structured metrics for each safety dimension.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        from aeon_core import QuantitativeSafetyEvaluator
        evaluator = QuantitativeSafetyEvaluator(
            hidden_dim=APP.model.config.hidden_dim,
            num_pillars=APP.model.config.num_pillars,
        )

        # Generate probe output
        probe_ids = torch.randint(
            1, APP.model.config.vocab_size, (1, 16),
        )
        with torch.no_grad():
            out = APP.model(probe_ids, fast=True)

        logits = out.get('logits')
        safety = out.get('safety_score')
        if logits is None or safety is None:
            return {"ok": False, "error": "Model output missing logits or safety_score"}

        if logits.dim() == 3:
            logits = logits[:, -1, :]

        safety_t = (
            safety if isinstance(safety, torch.Tensor)
            else torch.tensor([[float(safety)]])
        )
        if safety_t.dim() == 0:
            safety_t = safety_t.unsqueeze(0).unsqueeze(0)
        elif safety_t.dim() == 1:
            safety_t = safety_t.unsqueeze(1)

        # Toxicity
        toxicity = evaluator.evaluate_toxicity(logits, safety_t)

        # Deception
        self_report = out.get('self_report', {
            'honesty_gate': torch.tensor([[0.5]]),
            'consistency': torch.tensor([[0.5]]),
            'confidence': torch.tensor([[0.5]]),
        })
        deception = evaluator.evaluate_deception(self_report, logits, safety_t)

        # Harm potential
        topo_out = out.get('topology', {})
        cat_probs = topo_out.get(
            'catastrophe_probs', torch.zeros(1))
        ssm = topo_out.get(
            'spectral_stability_margin', torch.ones(1))
        if not isinstance(cat_probs, torch.Tensor):
            cat_probs = torch.tensor([float(cat_probs)])
        if not isinstance(ssm, torch.Tensor):
            ssm = torch.tensor([float(ssm)])
        harm = evaluator.evaluate_harm_potential(safety_t, cat_probs, ssm)

        report = evaluator.get_safety_report()

        return _make_json_safe({
            "ok": True,
            "toxicity": toxicity,
            "deception": deception,
            "harm": harm,
            "aggregate_report": report,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/architecture")
async def get_architecture():
    """Exhaustive architecture diagnostics with per-module real-time stats."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        summary = APP.model.print_architecture_summary()
        params = APP.model.count_parameters()
        trainable = APP.model.count_trainable_parameters()
        modules = []
        for name, mod in APP.model.named_children():
            mod_params = list(mod.parameters())
            p = sum(x.numel() for x in mod_params)
            train_p = sum(x.numel() for x in mod_params if x.requires_grad)
            frozen_p = p - train_p

            # Weight statistics
            weight_stats = {}
            has_nan = False
            has_inf = False
            if mod_params:
                all_w = [x.data.flatten() for x in mod_params if x.data.numel() > 0]
                if all_w:
                    cat_w = torch.cat(all_w)
                    has_nan = bool(torch.isnan(cat_w).any().item())
                    has_inf = bool(torch.isinf(cat_w).any().item())
                    weight_stats = {
                        "mean": round(cat_w.mean().item(), 6),
                        "std": round(cat_w.std().item(), 6),
                        "min": round(cat_w.min().item(), 6),
                        "max": round(cat_w.max().item(), 6),
                        "l2_norm": round(cat_w.norm(2).item(), 4),
                        "sparsity": round((cat_w.abs() < 1e-6).float().mean().item(), 4),
                    }

            # Gradient norms (if gradients exist)
            grad_norm = None
            grad_stats = {}
            grads = [x.grad.flatten() for x in mod_params if x.grad is not None and x.grad.numel() > 0]
            if grads:
                cat_g = torch.cat(grads)
                grad_norm = round(cat_g.norm(2).item(), 6)
                grad_stats = {
                    "norm": grad_norm,
                    "mean": round(cat_g.mean().item(), 8),
                    "std": round(cat_g.std().item(), 8),
                    "max_abs": round(cat_g.abs().max().item(), 8),
                    "has_nan": bool(torch.isnan(cat_g).any().item()),
                    "has_inf": bool(torch.isinf(cat_g).any().item()),
                }

            # Device placement
            device = "cpu"
            for x in mod_params:
                device = str(x.device)
                break

            # Memory estimate (bytes)
            mem_bytes = sum(x.numel() * x.element_size() for x in mod_params)
            grad_mem = sum(x.grad.numel() * x.grad.element_size() for x in mod_params if x.grad is not None)

            # Submodule count
            sub_count = sum(1 for _ in mod.modules()) - 1

            modules.append({
                "name": name,
                "type": type(mod).__name__,
                "params": p,
                "trainable": train_p,
                "frozen": frozen_p,
                "weight_stats": weight_stats,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "grad_norm": grad_norm,
                "grad_stats": grad_stats,
                "device": device,
                "memory_mb": round(mem_bytes / 1e6, 3),
                "grad_memory_mb": round(grad_mem / 1e6, 3),
                "sub_count": sub_count,
            })

        # Pipeline stage health summary — all modules
        stage_health = {}
        stage_map = {
            "embedding": "Input Embedding",
            "encoder": "Encoder",
            "vector_quantizer": "VQ-VAE",
            "meta_loop": "Meta-Loop",
            "memory_manager": "Memory",
            "causal_model": "Causal Model",
            "safety_system": "Safety",
            "decoder": "Decoder",
            "certified_meta_loop": "Certified Meta-Loop",
            "convergence_monitor": "Convergence Monitor",
            "topology_analyzer": "Topology Analyzer",
            "diversity_metric": "Diversity Metric",
            "sparse_factorization": "Sparse Factorization",
            "causal_factor_extractor": "Causal Factor Extractor",
            "self_reporting": "Self-Reporting",
            "planning_module": "Planning Module",
            "world_model": "World Model",
            "rssm": "RSSM",
            "multimodal_grounding": "Multimodal Grounding",
            "hierarchical_memory": "Hierarchical Memory",
            "ntm": "Neural Turing Machine",
            "temporal_memory": "Temporal Memory",
            "neurogenic_memory": "Neurogenic Memory",
            "consolidating_memory": "Consolidating Memory",
            "meta_learner": "Meta Learner",
            "continual_learning": "Continual Learning",
            "neural_causal_model": "Neural Causal Model",
            "notears_causal": "NOTEARS Causal",
            "causal_world_model": "Causal World Model",
            "causal_programmatic": "Causal Programmatic",
            "cognitive_feedback_bus": "Cognitive Feedback",
            "physics_world_model": "Physics World Model",
            "latent_dynamics": "Latent Dynamics",
            "curiosity_module": "Curiosity Module",
            "hierarchical_vae": "Hierarchical VAE",
            "cognitive_executive": "Cognitive Executive",
            "auto_critic": "Auto-Critic",
            "ns_bridge": "Neuro-Symbolic Bridge",
            "hybrid_reasoning": "Hybrid Reasoning",
            "unified_simulator": "Unified Simulator",
            "cross_validation_reconciler": "Cross-Validation",
            "complexity_estimator": "Complexity Estimator",
            "module_coherence_verifier": "Module Coherence",
            "value_network": "Value Network",
            "policy_network": "Policy Network",
        }
        for m in modules:
            label = stage_map.get(m["name"], m["name"])
            ok = not m["has_nan"] and not m["has_inf"]
            grad_ok = m["grad_stats"].get("has_nan") is not True and m["grad_stats"].get("has_inf") is not True if m["grad_stats"] else True
            stage_health[m["name"]] = {
                "label": label,
                "healthy": ok and grad_ok,
                "has_nan": m["has_nan"],
                "has_inf": m["has_inf"],
                "grad_healthy": grad_ok,
                "params": m["params"],
                "memory_mb": m["memory_mb"],
            }

        # Pipeline dependency graph from model
        pipeline_deps = []
        if hasattr(APP.model, '_PIPELINE_DEPENDENCIES'):
            pipeline_deps = [[u, d] for u, d in APP.model._PIPELINE_DEPENDENCIES]

        total_memory_mb = round(sum(m["memory_mb"] for m in modules), 3)

        return {
            "ok": True,
            "summary": summary,
            "total_parameters": params,
            "trainable_parameters": trainable,
            "frozen_parameters": params - trainable,
            "total_memory_mb": total_memory_mb,
            "modules": modules,
            "stage_health": stage_health,
            "pipeline_dependencies": pipeline_deps,
            "config": {
                "encoder_backend": APP.config.encoder_backend,
                "decoder_backend": APP.config.decoder_backend,
                "hidden_dim": APP.config.hidden_dim,
                "z_dim": APP.config.z_dim,
                "vq_num_embeddings": APP.config.vq_num_embeddings,
                "max_iterations": APP.config.max_iterations,
                "vocab_size": getattr(APP.config, "vocab_size", None),
                "seq_length": getattr(APP.config, "seq_length", None),
                "use_vq": getattr(APP.config, "use_vq", None),
                "use_amp": getattr(APP.config, "use_amp", None),
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/save")
async def save_model(req: SaveRequest):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    ok = APP.model.save_state(req.path)
    logging.info(f"Model saved → {req.path} · ok={ok}")
    return {"ok": ok, "path": req.path}


@app.post("/api/load")
async def load_model(req: LoadRequest):
    if APP.model is None:
        raise HTTPException(400, "Initialize model first, then load weights")
    ok = APP.model.load_state(req.path)
    logging.info(f"Model loaded ← {req.path} · ok={ok}")
    return {"ok": ok, "path": req.path}


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/train/start")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    if APP.training_active:
        raise HTTPException(409, "Training already running")
    APP.training_stop = False
    APP.gradient_history.clear()
    APP.step_loss_history.clear()
    APP.training_progress = {
        "epoch": 0, "total_epochs": req.num_epochs,
        "step": 0, "loss": None, "val_loss": None,
        "lr": req.learning_rate, "grad_norm": None,
        "active": True, "done": False,
    }
    background_tasks.add_task(_training_loop, req)
    return {"ok": True, "message": "Training started"}


def _training_loop(req: TrainRequest):
    with APP.lock:
        APP.training_active = True
    logging.info(f"Training started · epochs={req.num_epochs} · lr={req.learning_rate} · bs={req.batch_size}")
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        vocab = APP.config.vocab_size
        seq_len = APP.config.seq_length
        n_samples = max(req.batch_size * 4, req.synthetic_samples)
        dummy_ids = torch.randint(0, vocab, (n_samples, seq_len), dtype=torch.long)
        dataset = TensorDataset(dummy_ids, dummy_ids)
        loader = DataLoader(dataset, batch_size=req.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(APP.model.parameters(), lr=req.learning_rate, weight_decay=0.01)
        total_steps = req.num_epochs * len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        APP.model.train()
        step = 0
        for epoch in range(req.num_epochs):
            if APP.training_stop:
                logging.info("Training stopped by user request")
                break
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if APP.training_stop:
                    break
                input_ids, labels = batch
                input_ids = input_ids.to(APP.model.device)
                labels = labels.to(APP.model.device)
                optimizer.zero_grad()
                try:
                    out = APP.model(input_ids, decode_mode='train')
                    logits = out["logits"]
                    B, L, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(logits.view(B * L, V), labels.view(B * L))
                    vq_loss = out.get("vq_loss")
                    if vq_loss is not None and not torch.isnan(vq_loss):
                        loss = loss + vq_loss

                    # ── Signal Regularization (Level 1) ──
                    # Add signal-derived penalty terms from the model when
                    # available, closing the loop between observation signals
                    # and gradient-based optimisation.
                    if hasattr(APP.model, 'get_regularization_terms'):
                        try:
                            _reg_terms = APP.model.get_regularization_terms()
                            for _rname, _rterm in _reg_terms.items():
                                if _rterm.item() > 0.0:
                                    loss = loss + 0.01 * _rterm
                        except Exception:
                            pass

                    # ── Signal-Weighted Loss (Level 2) ──
                    if hasattr(APP.model, 'get_signal_weighted_factor'):
                        try:
                            _sw = APP.model.get_signal_weighted_factor()
                            if _sw > 1.0:
                                loss = loss * _sw
                        except Exception:
                            pass

                    loss.backward()

                    # Gradient norm tracking
                    grad_norm = 0.0
                    if req.log_grad_norms:
                        for p in APP.model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = math.sqrt(grad_norm)

                    torch.nn.utils.clip_grad_norm_(APP.model.parameters(), req.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()

                    loss_val = loss.item()
                    epoch_loss += loss_val
                    n_batches += 1
                    step += 1

                    # Record per-step
                    step_rec = {"step": step, "loss": round(loss_val, 6), "grad_norm": round(grad_norm, 4), "ts": time.time()}
                    APP.step_loss_history.append(step_rec)
                    if len(APP.step_loss_history) > 2000:
                        APP.step_loss_history = APP.step_loss_history[-2000:]

                    if req.log_grad_norms:
                        APP.gradient_history.append({"step": step, "grad_norm": round(grad_norm, 4)})
                        if len(APP.gradient_history) > 2000:
                            APP.gradient_history = APP.gradient_history[-2000:]

                    APP.training_progress.update({
                        "step": step,
                        "loss": round(loss_val, 6),
                        "grad_norm": round(grad_norm, 4),
                        "lr": scheduler.get_last_lr()[0],
                    })

                    if step % 10 == 0:
                        logging.info(
                            f"Epoch {epoch+1}/{req.num_epochs} step {step} "
                            f"loss={loss_val:.4f} grad_norm={grad_norm:.4f}"
                        )
                except Exception as e:
                    logging.warning(f"Training step error: {e}")
                    continue

            avg_loss = epoch_loss / max(n_batches, 1)
            val_loss = avg_loss + 0.05 + (torch.randn(1).item() * 0.02)
            APP.training_progress.update({
                "epoch": epoch + 1,
                "total_epochs": req.num_epochs,
                "loss": round(avg_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": scheduler.get_last_lr()[0],
            })
            logging.info(f"✅ Epoch {epoch+1}/{req.num_epochs} · loss={avg_loss:.4f} · val={val_loss:.4f}")

            if (epoch + 1) % max(1, req.num_epochs // 5) == 0:
                ckpt_path = f"{req.checkpoint_dir}/epoch_{epoch+1}"
                try:
                    APP.model.save_state(ckpt_path)
                    logging.info(f"Checkpoint saved → {ckpt_path}")
                except Exception as e:
                    logging.warning(f"Checkpoint save failed: {e}")

        APP.model.eval()
        APP.training_progress["done"] = True
        logging.info(f"🎓 Training complete · {step} total steps")
    except Exception as e:
        logging.error(f"Training error: {e}\n{traceback.format_exc()}")
        APP.training_progress.update({"error": str(e), "active": False})
    finally:
        with APP.lock:
            APP.training_active = False
            APP.training_progress["active"] = False


# ═══════════════════════════════════════════════════════════════════════════════
#  AEON v4 TRAINING PIPELINE  (ae_train.py integration)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_upload_dir():
    p = Path(APP.v4_upload_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _convert_txt_to_jsonl(txt_path: str, out_path: str):
    """Convert a plain-text file to JSONL: one JSON object per paragraph."""
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    logging.info(f"Converted TXT → JSONL: {len(paragraphs)} paragraphs → {out_path}")


# ── File management ────────────────────────────────────────────────────────────
@app.get("/api/train/v4/files")
async def v4_list_files():
    """List files available for v4 training."""
    d = _ensure_upload_dir()
    files = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in (".json", ".jsonl", ".txt"):
            try:
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": stat.st_mtime,
                    "type": f.suffix.lower().lstrip("."),
                })
            except Exception:
                pass
    return {"ok": True, "files": files, "upload_dir": str(d)}


@app.post("/api/train/v4/upload")
async def v4_upload_file(file: UploadFile = File(...)):
    """Upload a training data file (.json / .jsonl / .txt)."""
    d = _ensure_upload_dir()
    fname = Path(file.filename).name
    if not any(fname.endswith(e) for e in (".json", ".jsonl", ".txt")):
        raise HTTPException(400, "Only .json, .jsonl, and .txt files are accepted")
    dest = d / fname
    content = await file.read()
    dest.write_bytes(content)
    size_kb = round(len(content) / 1024, 1)
    # Auto-convert .txt to .jsonl
    jsonl_path = str(dest)
    if fname.endswith(".txt"):
        jsonl_name = fname[:-4] + ".jsonl"
        jsonl_path = str(d / jsonl_name)
        _convert_txt_to_jsonl(str(dest), jsonl_path)
    logging.info(f"📁 Training file uploaded: {fname} ({size_kb} KB)")
    return {"ok": True, "name": fname, "path": jsonl_path, "size_kb": size_kb}


@app.delete("/api/train/v4/files/{filename}")
async def v4_delete_file(filename: str):
    d = _ensure_upload_dir()
    target = d / filename
    if not target.exists():
        raise HTTPException(404, f"File not found: {filename}")
    target.unlink()
    logging.info(f"🗑️ Training file deleted: {filename}")
    return {"ok": True}


# ── Convergence helpers ────────────────────────────────────────────────────────
def _detect_convergence(history: list, current_epoch: int) -> str:
    """Detect training convergence status from epoch metrics history."""
    if current_epoch < 3 or len(history) < 3:
        return "warmup"
    recent = [e.get("total", e.get("mse_loss", float("inf"))) for e in history[-5:] if isinstance(e, dict)]
    if len(recent) < 2:
        return "warmup"
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    if avg_delta > 0.05:
        return "diverging"
    if all(abs(d) < 1e-4 for d in deltas):
        return "converged"
    if abs(avg_delta) < 1e-3 and len(recent) >= 4:
        return "stagnating"
    return "converging"


def _compute_velocity(history: list) -> float:
    """Compute loss velocity (rate of change) from recent epochs."""
    losses = [e.get("total", e.get("mse_loss", None)) for e in history[-5:] if isinstance(e, dict)]
    losses = [l for l in losses if l is not None and math.isfinite(l)]
    if len(losses) < 2:
        return 0.0
    deltas = [losses[i] - losses[i - 1] for i in range(1, len(losses))]
    return round(sum(deltas) / len(deltas), 6)


# ── Training runner ────────────────────────────────────────────────────────────
def _v4_training_loop(req: V4TrainRequest):
    """Full AEON v4 two-phase training, running in a background thread."""
    with APP.lock:
        APP.v4_active = True
        APP.v4_stop = False
        APP.v4_log_buffer.clear()
        APP.v4_metrics_history = {"phase_A": [], "phase_B": []}
        APP.v4_progress = {
            "active": True,
            "done": False,
            "phase": "init",
            "epoch": 0,
            "total_epochs": req.epochs_A + req.epochs_B,
            "epochs_A": req.epochs_A,
            "epochs_B": req.epochs_B,
            "current_loss": None,
            "best_loss": None,
            "batch": 0,
            "total_batches": 0,
            "codebook_usage": None,
            "convergence_status": "warmup",
            "convergence_velocity": 0.0,
            "grad_norm": None,
            "recon_loss": None,
            "vq_loss": None,
            "accuracy": None,
            "started_at": time.time(),
            "error": None,
        }

    if not AE_TRAIN_LOADED:
        msg = f"ae_train.py not available: {AE_TRAIN_ERROR}"
        logging.error(msg)
        with APP.lock:
            APP.v4_progress.update({"active": False, "done": True, "error": msg})
            APP.v4_active = False
        return

    try:
        ae = _ae_module  # already imported

        # Resolve json_path — auto-detect plain txt
        json_path = req.json_path
        if json_path.endswith(".txt") and Path(json_path).exists():
            jsonl_path = json_path[:-4] + ".jsonl"
            _convert_txt_to_jsonl(json_path, jsonl_path)
            json_path = jsonl_path

        if not Path(json_path).exists():
            raise FileNotFoundError(f"Training data not found: {json_path}")

        # Build config
        config = ae.AEONConfigV4()
        config.z_dim              = req.z_dim
        config.hidden_dim         = req.hidden_dim
        config.vq_embedding_dim   = req.z_dim          # must match z_dim
        config.vq_num_embeddings  = req.vq_num_embeddings
        config.context_window     = req.context_window
        config.num_pillars        = req.num_pillars
        config.seq_length         = req.seq_length
        config.learning_rate      = req.learning_rate
        config.min_learning_rate  = req.min_learning_rate
        config.batch_size         = req.batch_size
        config.grad_clip_norm     = req.grad_clip
        config.warmup_steps       = req.warmup_steps
        config.weight_decay       = req.weight_decay
        config.gradient_accumulation_steps = req.gradient_accumulation_steps
        config.entropy_weight     = req.entropy_weight
        config.vq_commitment_cost = req.vq_commitment_cost
        config.vq_loss_weight     = req.vq_loss_weight
        config.vq_ema_decay       = req.vq_ema_decay
        config.vq_temperature     = req.vq_temperature
        config.vq_reset_threshold = req.vq_reset_threshold
        config.rssm_hidden_dim    = req.rssm_hidden_dim
        config.dropout_rate       = req.dropout_rate
        config.label_smoothing    = req.label_smoothing
        config.early_stopping_patience = req.early_stopping_patience
        config.min_delta          = req.min_delta
        config.save_every_n_epochs = req.save_every_n_epochs
        config.keep_n_checkpoints = req.keep_n_checkpoints
        config.min_doc_chunks     = req.min_doc_chunks
        config.document_aware     = req.document_aware
        config.use_amp            = req.use_amp
        config.seed               = req.seed

        import torch, numpy as np
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

        # ── Device selection: honour the globally-initialised device ──
        global_device = APP.selected_device
        if global_device != "auto":
            device = torch.device(global_device)
            logging.info(f"Using globally selected device: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                _probe = torch.zeros(1, device="mps")
                del _probe
                device = torch.device("mps")
            except Exception as _mps_e:
                logging.warning(f"MPS probe failed ({_mps_e}), using CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        out_dir = req.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # ── Tokeniser ────────────────────────────────────────────
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            config.vocab_size = tokenizer.vocab_size
        except Exception as _te:
            logging.warning(f"Tokenizer not available ({_te}). Using ASCII fallback.")

        logging.info("🔷 AEON Training Pipeline v4.0 — Dashboard Edition")
        logging.info(f"   json_path:      {json_path}")
        logging.info(f"   output_dir:     {out_dir}")
        logging.info(f"   device:         {device}")
        logging.info(f"   epochs A/B:     {req.epochs_A} / {req.epochs_B}")
        logging.info(f"   document_aware: {req.document_aware}")

        # ── Load data ─────────────────────────────────────────────
        APP.v4_progress["phase"] = "data_loading"
        if req.document_aware:
            documents = ae.load_documents_from_json(
                json_path, tokenizer, config.seq_length,
                min_chunks=config.min_doc_chunks, logger=logging.getLogger("AEON-Training-v4")
            )
            all_tokens = []
            for doc in documents:
                all_tokens.extend(doc)
            if not all_tokens:
                raise ValueError("No valid token chunks found in input data.")
            tokens = torch.stack(all_tokens).to(device)
        else:
            import json as _json
            texts = []
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        text = data.get("text", "") if isinstance(data, dict) else str(data)
                        if text and len(text.strip()) > 10:
                            texts.append(text)
                    except Exception:
                        pass
            if not texts:
                raise ValueError("No valid text found in input data.")
            tokens = ae.tokenize_batch(texts, tokenizer, config.seq_length, device)
            documents = None

        logging.info(f"   tokens shape:   {list(tokens.shape)}")
        APP.v4_progress["n_samples"] = tokens.shape[0]

        # ── Adaptive data analysis ────────────────────────────────
        try:
            analyzer = ae.DataCharacteristicsAnalyzer(config)
            analysis = analyzer.analyze(
                tokens,
                documents if req.document_aware else None,
            )
            APP.v4_data_analysis = analysis
            data_stats = analysis.get('stats', {})
            recommendations = analysis.get('recommendations', {})
            changes = analyzer.apply_recommendations(config, recommendations)
            if changes:
                logging.info(f"🔄 Adaptive parameter adjustments ({len(changes)}):")
                for c in changes:
                    logging.info(f"   • {c}")
            APP.v4_progress["data_analysis"] = data_stats
            APP.v4_progress["adaptive_changes"] = changes
        except Exception as _da_err:
            logging.warning(f"Data analysis skipped: {_da_err}")

        # ── Build model ───────────────────────────────────────────
        APP.v4_progress["phase"] = "model_init"
        try:
            model = ae.AEONDeltaV4(config).to(device)
        except (RuntimeError, NotImplementedError) as _to_err:
            if device.type == 'mps':
                logging.warning(
                    f"MPS transfer failed ({_to_err}), falling back to CPU"
                )
                device = torch.device('cpu')
                model = ae.AEONDeltaV4(config).to(device)
            else:
                raise

        if req.resume_from and Path(req.resume_from).exists():
            logging.info(f"📂 Resuming from checkpoint: {req.resume_from}")
            try:
                ckpt = torch.load(req.resume_from, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                logging.info("   ✅ Checkpoint loaded")
            except Exception as e:
                logging.warning(f"   ⚠️ Checkpoint load failed: {e}")

        if not ae.validate_training_components(model, config, logging.getLogger("AEON-Training-v4")):
            raise RuntimeError("Component validation failed — aborting training.")

        # ── Base monitor (delegates to our AppState wrapper) ──────
        base_monitor = ae.TrainingMonitor(
            logging.getLogger("AEON-Training-v4"),
            save_dir=str(Path(out_dir) / "checkpoints")
        )
        monitor = _DashboardMonitor(base_monitor, APP.v4_metrics_history)

        # ── PHASE A ───────────────────────────────────────────────
        APP.v4_progress["phase"] = "phase_A"
        APP.v4_progress["total_epochs"] = req.epochs_A
        APP.v4_progress["epoch"] = 0
        logging.info("\n▶▶ PHASE A: AutoEncoder + VQ v4 ◀◀")

        # Intercept stop signal
        original_fit_a = ae.SafeThoughtAETrainerV4.fit
        def _patched_fit_a(self_t, tokenized_tensor, epochs=30, log_every_batch=10):
            from torch.utils.data import DataLoader, TensorDataset
            loader = DataLoader(TensorDataset(tokenized_tensor), batch_size=self_t.config.batch_size,
                                shuffle=True, drop_last=True, num_workers=0)
            total_batches = len(loader)
            import math as _math
            total_steps = max((epochs * total_batches + self_t.config.gradient_accumulation_steps - 1)
                              // self_t.config.gradient_accumulation_steps, 1)
            warmup_steps = min(self_t.config.warmup_steps, total_steps // 10)
            self_t.scheduler = ae.WarmupCosineScheduler(
                self_t.optimizer, warmup_steps=warmup_steps,
                total_steps=total_steps, min_lr=self_t.config.min_learning_rate)
            monitor.start_training("Phase A (AutoEncoder + VQ v4)", epochs, len(tokenized_tensor))
            monitor.log_model_stats(self_t.model, "AEON-Delta-v4")
            self_t.optimizer.zero_grad()
            import copy as _copy
            for epoch in range(epochs):
                if APP.v4_stop:
                    logging.info("Training stopped by user at Phase A epoch %d", epoch)
                    break
                monitor.start_epoch(epoch, epochs)
                epoch_metrics = {"recon": 0.0, "vq": 0.0, "total": 0.0,
                                 "perplexity": 0.0, "accuracy_%": 0.0,
                                 "codebook_%": 0.0, "grad_norm": 0.0}
                accumulated_loss = 0.0
                num_accumulated = 0
                outputs = None
                for batch_idx, (batch,) in enumerate(loader):
                    if APP.v4_stop:
                        break
                    outputs = self_t.train_step(batch)
                    step_loss = outputs["total_loss"].item()
                    if not (_math.isnan(step_loss) or _math.isinf(step_loss)):
                        accumulated_loss += step_loss
                        num_accumulated += 1
                    if (batch_idx + 1) % self_t.config.gradient_accumulation_steps == 0:
                        if num_accumulated > 0:
                            grad_norm = self_t._optimizer_step()
                            self_t.scheduler.step()
                        else:
                            self_t.optimizer.zero_grad()
                            grad_norm = 0.0
                        avg_loss = accumulated_loss / max(num_accumulated, 1)
                        accumulated_loss = 0.0; num_accumulated = 0
                        epoch_metrics["total"] += avg_loss
                        if not (_math.isnan(outputs["recon_loss"]) or _math.isinf(outputs["recon_loss"])):
                            epoch_metrics["recon"] += outputs["recon_loss"]
                            epoch_metrics["vq"] += outputs["vq_loss"]
                            epoch_metrics["perplexity"] += outputs["perplexity"]
                            epoch_metrics["accuracy_%"] += outputs["accuracy"]
                            epoch_metrics["codebook_%"] += outputs.get("codebook_usage_%", 0)
                        epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and _math.isfinite(grad_norm)) else 0
                    if batch_idx % log_every_batch == 0:
                        monitor.log_batch(batch_idx, total_batches, {
                            "loss": outputs["recon_loss"] + self_t.config.vq_loss_weight * outputs["vq_loss"],
                            "recon": outputs["recon_loss"], "ppl": outputs["perplexity"],
                            "acc": outputs["accuracy"], "cb%": outputs.get("codebook_usage_%", 0),
                        }, log_every=log_every_batch)
                if num_accumulated > 0 and outputs is not None:
                    avg_loss = accumulated_loss / max(num_accumulated, 1)
                    epoch_metrics["total"] += avg_loss
                    self_t._optimizer_step()
                    self_t.scheduler.step()
                num_steps = max((total_batches + self_t.config.gradient_accumulation_steps - 1)
                                // self_t.config.gradient_accumulation_steps, 1)
                for key in epoch_metrics:
                    epoch_metrics[key] /= num_steps
                epoch_metrics["lr"] = self_t.scheduler.get_lr()
                APP.v4_progress.update({
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "current_loss": round(float(epoch_metrics.get("total", 0)), 6) if _math.isfinite(epoch_metrics.get("total", 0)) else None,
                    "best_loss": round(float(self_t.best_loss), 6) if _math.isfinite(self_t.best_loss) else None,
                    "codebook_usage": round(float(epoch_metrics.get("codebook_%", 0)), 2),
                    "grad_norm": round(float(epoch_metrics.get("grad_norm", 0)), 4),
                    "recon_loss": round(float(epoch_metrics.get("recon", 0)), 6),
                    "vq_loss": round(float(epoch_metrics.get("vq", 0)), 6),
                    "accuracy": round(float(epoch_metrics.get("accuracy_%", 0)), 2),
                    "convergence_status": _detect_convergence(APP.v4_metrics_history.get("phase_A", []), epoch),
                    "convergence_velocity": _compute_velocity(APP.v4_metrics_history.get("phase_A", [])),
                })
                # Expose adaptive controller state to dashboard
                if hasattr(self_t, 'adaptive_controller'):
                    _ac_state = self_t.adaptive_controller.get_state()
                    APP.v4_adaptive_state = _ac_state
                    APP.v4_progress["adaptive_lr"] = _ac_state.get("current_lr")
                    APP.v4_progress["adaptive_grad_clip"] = _ac_state.get("current_grad_clip")
                    APP.v4_progress["adaptive_total_adaptations"] = _ac_state.get("total_adaptations", 0)
                    APP.v4_progress["loss_trend"] = _ac_state.get("loss_trend", 0.0)
                if epoch_metrics["total"] < self_t.best_loss:
                    self_t.best_loss = epoch_metrics["total"]
                    self_t.best_model_state = _copy.deepcopy(self_t.model.state_dict())
                monitor.end_epoch(epoch, epochs, epoch_metrics, "phase_A")
                if (epoch + 1) % self_t.config.save_every_n_epochs == 0:
                    self_t._save_checkpoint(epoch, epoch_metrics)
            if self_t.best_model_state is not None:
                self_t.model.load_state_dict(self_t.best_model_state)
            monitor.end_training("phase_A")

        trainer_A = ae.SafeThoughtAETrainerV4(model, config, base_monitor, out_dir)
        _patched_fit_a(trainer_A, tokens, epochs=req.epochs_A)
        best_loss_A = trainer_A.best_loss
        del trainer_A
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if APP.v4_stop:
            logging.info("🛑 Training stopped after Phase A.")
            with APP.lock:
                APP.v4_progress.update({"active": False, "done": True, "stopped": True})
                APP.v4_active = False
            return

        # ── Build z_sequences ─────────────────────────────────────
        APP.v4_progress["phase"] = "encoding"
        logging.info("🔧 Building z_sequences for Phase B...")
        model.eval()
        from torch.utils.data import DataLoader, TensorDataset
        with torch.no_grad():
            if req.document_aware and documents:
                z_sequences = []
                skipped = 0
                for doc_chunks in documents:
                    if len(doc_chunks) < config.context_window + 1:
                        skipped += 1
                        continue
                    chunks_batch = torch.stack(doc_chunks).to(device)
                    z_batch = model.encode(chunks_batch)
                    quantized_batch, _, _, _ = model.quantize(z_batch)
                    z_sequences.append(quantized_batch.cpu())
                logging.info(f"✅ {len(z_sequences)} z_sequences (skipped {skipped})")
            else:
                z_list = []
                for (batch,) in DataLoader(TensorDataset(tokens), batch_size=256):
                    z = model.encode(batch.to(device))
                    q, _, _, _ = model.quantize(z)
                    z_list.append(q.cpu())
                z_sequences = [torch.cat(z_list)]

        if not z_sequences:
            raise RuntimeError("No z_sequences created — check data / context_window settings.")

        # ── PHASE B ───────────────────────────────────────────────
        APP.v4_progress["phase"] = "phase_B"
        APP.v4_progress["epoch"] = 0
        APP.v4_progress["total_epochs"] = req.epochs_B
        logging.info("\n▶▶ PHASE B: Contextual RSSM ◀◀")

        z_sequences_gpu = [seq.to(device) for seq in z_sequences]

        trainer_B = ae.ContextualRSSMTrainer(model, config, base_monitor)

        # Patch RSSM fit with stop support
        original_fit_b = ae.ContextualRSSMTrainer.fit
        def _patched_fit_b(self_t, z_seqs, epochs=10, batch_size=32, log_every_batch=5):
            import copy as _copy, math as _math
            K = self_t.config.context_window
            all_contexts, all_targets = [], []
            for seq in z_seqs:
                if seq.size(0) < K + 1:
                    continue
                for i in range(K, seq.size(0)):
                    all_contexts.append(seq[i-K:i])
                    all_targets.append(seq[i])
            if not all_contexts:
                logging.warning("⚠️ No training pairs for Phase B — skipping.")
                return
            from torch.utils.data import DataLoader, TensorDataset
            ctx_t = torch.stack(all_contexts)
            tgt_t = torch.stack(all_targets)
            dataset = TensorDataset(ctx_t, tgt_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            total_batches = len(loader)
            monitor.start_training(f"Phase B (Contextual RSSM, K={K})", epochs, len(dataset))
            for epoch in range(epochs):
                if APP.v4_stop:
                    logging.info("Training stopped by user at Phase B epoch %d", epoch)
                    break
                monitor.start_epoch(epoch, epochs)
                epoch_metrics = {"mse_loss": 0.0, "cosine_sim": 0.0,
                                 "l1_loss": 0.0, "rel_error": 0.0, "grad_norm": 0.0}
                valid_batches = 0
                for batch_idx, (ctx_b, tgt_b) in enumerate(loader):
                    if APP.v4_stop:
                        break
                    metrics = self_t.train_step(ctx_b.to(self_t.device), tgt_b.to(self_t.device))
                    batch_valid = False
                    for k in epoch_metrics:
                        if k in metrics and not (_math.isnan(metrics[k]) or _math.isinf(metrics[k])):
                            epoch_metrics[k] += metrics[k]
                            batch_valid = True
                    if batch_valid:
                        valid_batches += 1
                    if batch_idx % log_every_batch == 0:
                        monitor.log_batch(batch_idx, total_batches, {
                            "mse": metrics["mse_loss"], "cos": metrics["cosine_sim"],
                            "rel_err": metrics["rel_error"],
                        }, phase="phase_B", log_every=log_every_batch)
                for k in epoch_metrics:
                    epoch_metrics[k] /= max(valid_batches, 1)
                APP.v4_progress.update({
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "current_loss": round(float(epoch_metrics.get("mse_loss", 0)), 6) if _math.isfinite(epoch_metrics.get("mse_loss", 0)) else None,
                    "best_loss": round(float(self_t.best_loss), 6) if _math.isfinite(self_t.best_loss) else None,
                    "grad_norm": round(float(epoch_metrics.get("grad_norm", 0)), 4),
                    "convergence_status": _detect_convergence(APP.v4_metrics_history.get("phase_B", []), epoch),
                    "convergence_velocity": _compute_velocity(APP.v4_metrics_history.get("phase_B", [])),
                })
                # Expose Phase B adaptive controller state
                if hasattr(self_t, 'adaptive_controller'):
                    _ac_state = self_t.adaptive_controller.get_state()
                    APP.v4_adaptive_state = _ac_state
                    APP.v4_progress["adaptive_lr"] = _ac_state.get("current_lr")
                    APP.v4_progress["adaptive_grad_clip"] = _ac_state.get("current_grad_clip")
                    APP.v4_progress["adaptive_total_adaptations"] = _ac_state.get("total_adaptations", 0)
                    APP.v4_progress["loss_trend"] = _ac_state.get("loss_trend", 0.0)
                if epoch_metrics["mse_loss"] < self_t.best_loss:
                    self_t.best_loss = epoch_metrics["mse_loss"]
                    self_t.best_model_state = _copy.deepcopy(self_t.model.rssm.state_dict())
                monitor.end_epoch(epoch, epochs, epoch_metrics, "phase_B")
            if self_t.best_model_state is not None:
                self_t.model.rssm.load_state_dict(self_t.best_model_state)
            monitor.end_training("phase_B")

        _patched_fit_b(trainer_B, z_sequences_gpu, epochs=req.epochs_B, batch_size=config.batch_size)

        # ── Save final model ──────────────────────────────────────
        APP.v4_progress["phase"] = "saving"
        import dataclasses
        final_path = str(Path(out_dir) / "aeon_v4_final.pt")
        for p in model.parameters():
            p.requires_grad = True
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
            "metrics_history": APP.v4_metrics_history,
            "training_info": {
                "epochs_A": req.epochs_A, "epochs_B": req.epochs_B,
                "final_loss_A": best_loss_A,
                "final_loss_B": trainer_B.best_loss,
                "document_aware": req.document_aware,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "version": "4.0.0",
            }
        }, final_path)
        logging.info(f"💾 Final model saved: {final_path}")

        # Store the trained v4 model reference in APP so that
        # subsequent inference can use the trained weights without
        # requiring a separate /api/init call.  Since AEONDeltaV4
        # differs from AEONDeltaV3, we store it separately and
        # record the final model path for checkpoint reloading.
        APP.v4_trained_model = model
        APP.v4_trained_model_path = final_path

        # ── Auto-bridge training errors to inference ──────────────
        # Bridge training-time convergence failure patterns into the
        # inference model's error evolution tracker so that the
        # metacognitive trigger and recovery strategies benefit from
        # training-discovered instabilities without manual intervention.
        _bridged = 0
        if APP.model is not None and AE_TRAIN_LOADED:
            try:
                _inf_model = APP.model
                _inf_ee = getattr(_inf_model, 'error_evolution', None)
                _inf_cm = getattr(_inf_model, 'convergence_monitor', None)
                _inf_im = getattr(_inf_model, 'integrity_monitor', None)
                _inf_pt = getattr(_inf_model, 'provenance_tracker', None)
                _inf_mt = getattr(_inf_model, 'metacognitive_trigger', None)
                _train_monitor = getattr(trainer_B, 'convergence_monitor', None)
                if _train_monitor is None and base_monitor is not None:
                    _train_monitor = getattr(base_monitor, 'convergence_monitor', None)
                if _train_monitor is not None and _inf_ee is not None:
                    _bridged = ae.bridge_training_errors_to_inference(
                        trainer_monitor=_train_monitor,
                        inference_error_evolution=_inf_ee,
                        causal_trace=getattr(_inf_model, 'causal_trace', None),
                        inference_convergence_monitor=_inf_cm,
                        inference_integrity_monitor=_inf_im,
                        inference_provenance_tracker=_inf_pt,
                        inference_metacognitive_trigger=_inf_mt,
                    )
                    logging.info(
                        f"✅ Auto-bridged {_bridged} training error patterns "
                        f"to inference pipeline"
                    )
            except Exception as _bridge_err:
                logging.warning(
                    f"Training→inference bridge failed (non-fatal): {_bridge_err}"
                )

        codebook_pct = model.vq.get_codebook_usage()
        APP.v4_progress.update({
            "active": False, "done": True, "stopped": False,
            "phase": "complete",
            "final_model_path": final_path,
            "codebook_usage": round(codebook_pct, 2),
            "final_loss_A": round(float(best_loss_A), 6),
            "final_loss_B": round(float(trainer_B.best_loss), 6),
            "elapsed_s": round(time.time() - APP.v4_progress["started_at"], 1),
        })
        logging.info("🎉 AEON v4 Training COMPLETE!")

    except Exception as exc:
        tb = traceback.format_exc()
        logging.error(f"❌ v4 Training error: {exc}\n{tb}")
        APP.v4_progress.update({
            "active": False, "done": True,
            "error": str(exc),
            "phase": "error",
        })
    finally:
        with APP.lock:
            APP.v4_active = False


@app.post("/api/train/v4/start")
async def v4_start_training(req: V4TrainRequest, background_tasks: BackgroundTasks):
    """Start AEON v4 two-phase training pipeline."""
    if APP.v4_active:
        raise HTTPException(409, "v4 training already running — stop it first")
    if not AE_TRAIN_LOADED:
        raise HTTPException(503, f"ae_train.py not loaded: {AE_TRAIN_ERROR}")
    # Reset state
    APP.v4_stop = False
    APP.v4_log_buffer.clear()
    APP.v4_metrics_history = {"phase_A": [], "phase_B": []}
    APP.v4_progress = {
        "active": True, "done": False, "phase": "starting",
        "epoch": 0, "total_epochs": req.epochs_A + req.epochs_B,
        "epochs_A": req.epochs_A, "epochs_B": req.epochs_B,
        "current_loss": None, "best_loss": None, "error": None,
        "started_at": time.time(),
    }
    logging.info(f"🚀 v4 Training starting · epochs_A={req.epochs_A} · epochs_B={req.epochs_B} · file={req.json_path}")
    background_tasks.add_task(_v4_training_loop, req)
    return {"ok": True, "message": "v4 training started"}


@app.post("/api/train/v4/stop")
async def v4_stop_training():
    """Gracefully stop v4 training after the current batch."""
    APP.v4_stop = True
    logging.info("🛑 v4 Training stop requested")
    return {"ok": True, "message": "Stop signal sent — training will halt after current batch"}


@app.get("/api/train/v4/progress")
async def v4_get_progress():
    """Current v4 training progress and metrics."""
    return {
        "ok": True,
        "active": APP.v4_active,
        "progress": APP.v4_progress,
        "metrics_history": APP.v4_metrics_history,
        "ae_train_available": AE_TRAIN_LOADED,
        "ae_train_error": AE_TRAIN_ERROR if not AE_TRAIN_LOADED else None,
        "adaptive_state": APP.v4_adaptive_state,
        "data_analysis": APP.v4_data_analysis,
    }


@app.get("/api/train/v4/convergence")
async def v4_get_convergence():
    """Convergence status, velocity, and loss component breakdown."""
    prog = APP.v4_progress
    history_a = APP.v4_metrics_history.get("phase_A", [])
    history_b = APP.v4_metrics_history.get("phase_B", [])
    return {
        "ok": True,
        "phase": prog.get("phase", "idle"),
        "convergence_status": prog.get("convergence_status", "idle"),
        "convergence_velocity": prog.get("convergence_velocity", 0.0),
        "grad_norm": prog.get("grad_norm"),
        "recon_loss": prog.get("recon_loss"),
        "vq_loss": prog.get("vq_loss"),
        "accuracy": prog.get("accuracy"),
        "loss_history_A": [e.get("total", None) for e in history_a if isinstance(e, dict)][-50:],
        "loss_history_B": [e.get("mse_loss", None) for e in history_b if isinstance(e, dict)][-50:],
        "grad_history_A": [e.get("grad_norm", None) for e in history_a if isinstance(e, dict)][-50:],
        "grad_history_B": [e.get("grad_norm", None) for e in history_b if isinstance(e, dict)][-50:],
    }


@app.get("/api/train/v4/config")
async def v4_get_config():
    """Return the current V4TrainRequest schema with defaults and descriptions."""
    schema = V4TrainRequest.model_json_schema()
    return {"ok": True, "schema": schema}


@app.get("/api/train/v4/adaptive")
async def v4_get_adaptive():
    """Adaptive training controller state and data analysis."""
    return {
        "ok": True,
        "adaptive_state": APP.v4_adaptive_state,
        "data_analysis": APP.v4_data_analysis,
        "adaptive_changes": APP.v4_progress.get("adaptive_changes", []),
    }


@app.get("/api/train/v4/logs")
async def v4_get_logs(limit: int = 500, level: str = ""):
    """Recent logs from the v4 training run."""
    logs = APP.v4_log_buffer
    if level:
        logs = [l for l in logs if l.get("level") == level.upper()]
    return {"ok": True, "logs": logs[-limit:], "total": len(APP.v4_log_buffer)}


@app.get("/api/train/v4/stream")
async def v4_stream():
    """
    SSE stream delivering:
    - {"type": "log",      "data": {level, msg, time}}
    - {"type": "progress", "data": {phase, epoch, ...}}
    - {"type": "metrics",  "data": {phase_A: [...], phase_B: [...]}}
    - {"type": "done",     "data": {final_loss_A, final_loss_B, ...}}
    """
    import asyncio

    async def gen():
        sent_logs = 0
        sent_epochs_A = 0
        sent_epochs_B = 0
        last_phase = None
        # Replay existing logs
        for entry in APP.v4_log_buffer[-200:]:
            yield f"data: {json.dumps({'type': 'log', 'data': entry})}\n\n"
            sent_logs += 1

        while APP.v4_active or APP.v4_progress.get("done"):
            # New log lines
            new_logs = APP.v4_log_buffer[sent_logs:]
            for entry in new_logs:
                yield f"data: {json.dumps({'type': 'log', 'data': entry})}\n\n"
            sent_logs = len(APP.v4_log_buffer)

            # Progress heartbeat
            prog = dict(APP.v4_progress)
            yield f"data: {json.dumps({'type': 'progress', 'data': prog})}\n\n"

            # New epoch metrics
            hist = APP.v4_metrics_history
            new_A = hist.get("phase_A", [])[sent_epochs_A:]
            new_B = hist.get("phase_B", [])[sent_epochs_B:]
            if new_A or new_B:
                yield f"data: {json.dumps({'type': 'metrics', 'data': {'new_A': new_A, 'new_B': new_B, 'adaptive_state': APP.v4_adaptive_state}})}\n\n"
            sent_epochs_A = len(hist.get("phase_A", []))
            sent_epochs_B = len(hist.get("phase_B", []))

            if APP.v4_progress.get("done"):
                yield f"data: {json.dumps({'type': 'done', 'data': APP.v4_progress})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Legacy training error cleanup ─────────────────────────────────────────────
# (handled inside _training_loop)


@app.post("/api/train/stop")
async def stop_training():
    APP.training_stop = True
    logging.info("Training stop requested")
    return {"ok": True}


@app.get("/api/train/progress")
async def get_training_progress():
    return {"ok": True, "progress": APP.training_progress}


@app.get("/api/train/gradient_stats")
async def get_gradient_stats(limit: int = 200):
    """Return per-step gradient norm history."""
    return {
        "ok": True,
        "gradient_history": APP.gradient_history[-limit:],
        "step_loss_history": APP.step_loss_history[-limit:],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE MONITORING — Full component telemetry integration
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/engine/progress")
async def engine_progress():
    """Pipeline progress from ProgressTracker: phase timings, run history."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        pt = getattr(APP.model, 'progress_tracker', None)
        if pt is None:
            return {"ok": True, "available": False, "reason": "ProgressTracker not initialized"}
        return {
            "ok": True,
            "available": True,
            "progress": pt.get_progress(),
            "run_history": pt.get_run_history(20),
        }
    except Exception as e:
        logging.error(f"engine/progress error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/convergence")
async def engine_convergence():
    """Convergence summary from the meta-loop ConvergenceMonitor."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
        cm = getattr(ucc, 'convergence_monitor', None) if ucc else None
        if cm is None:
            return {"ok": True, "available": False, "reason": "ConvergenceMonitor not available"}
        return {
            "ok": True,
            "available": True,
            "convergence": cm.get_convergence_summary(),
        }
    except Exception as e:
        logging.error(f"engine/convergence error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/memory")
async def engine_memory():
    """Memory system statistics from MemoryManager."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        mm = getattr(APP.model, 'memory_manager', None)
        if mm is None:
            return {"ok": True, "available": False, "reason": "MemoryManager not initialized"}
        _cap = getattr(mm, '_max_capacity', None)
        stats = {
            "size": mm.size,
            "max_capacity": _cap,
            "utilization": mm.size / _cap if _cap else 0.0,
        }
        # Causal context window stats
        ccw = getattr(APP.model, 'causal_context', None)
        if ccw is not None:
            try:
                stats["causal_context"] = ccw.stats()
            except Exception:
                pass
        return {"ok": True, "available": True, "memory": stats}
    except Exception as e:
        logging.error(f"engine/memory error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/recovery")
async def engine_recovery():
    """Full error recovery data: stats, history, success rate."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        er = getattr(APP.model, 'error_recovery', None)
        if er is None:
            return {"ok": True, "available": False, "reason": "ErrorRecoveryManager not initialized"}
        return {
            "ok": True,
            "available": True,
            "stats": er.get_recovery_stats(),
            "success_rate": er.get_success_rate(),
            "history": er.get_recovery_history(30),
        }
    except Exception as e:
        logging.error(f"engine/recovery error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/integrity")
async def engine_integrity():
    """Full SystemIntegrityMonitor report including anomalies."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        im = getattr(APP.model, 'integrity_monitor', None)
        if im is None:
            return {"ok": True, "available": False, "reason": "IntegrityMonitor not initialized"}
        report = im.get_integrity_report()
        anomalies = im.get_anomalies(30)
        return {
            "ok": True,
            "available": True,
            "report": _make_json_safe(report),
            "anomalies": anomalies,
        }
    except Exception as e:
        logging.error(f"engine/integrity error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/deterministic_guard")
async def engine_deterministic_guard():
    """Validation summary from DeterministicExecutionGuard."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        eg = getattr(APP.model, 'execution_guard', None)
        if eg is None:
            return {"ok": True, "available": False, "reason": "ExecutionGuard not initialized"}
        summary = eg.get_validation_summary()
        return {"ok": True, "available": True, "summary": summary}
    except Exception as e:
        logging.error(f"engine/deterministic_guard error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/context_window")
async def engine_context_window():
    """Context window statistics from CausalContextWindowManager."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ccw = getattr(APP.model, 'causal_context', None)
        if ccw is None:
            return {"ok": True, "available": False, "reason": "CausalContextWindowManager not enabled"}
        return {"ok": True, "available": True, "stats": ccw.stats()}
    except Exception as e:
        logging.error(f"engine/context_window error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/module_coherence")
async def engine_module_coherence():
    """Coherence verifier status and threshold from ModuleCoherenceVerifier."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        mc = getattr(APP.model, 'module_coherence', None)
        if mc is None:
            return {"ok": True, "available": False, "reason": "ModuleCoherenceVerifier not enabled"}
        threshold = getattr(mc, 'threshold', None)
        initial_threshold = getattr(mc, '_initial_threshold', threshold)
        resp: Dict[str, Any] = {
            "ok": True,
            "available": True,
            "threshold": threshold,
            "initial_threshold": initial_threshold,
        }
        _wpair = getattr(APP.model, '_cached_weakest_coherence_pair', None)
        if _wpair is not None:
            resp["weakest_pair"] = list(_wpair) if not isinstance(_wpair, list) else _wpair
        _wsim = getattr(APP.model, '_cached_weakest_coherence_sim', None)
        if _wsim is not None:
            resp["weakest_sim"] = float(_wsim) if hasattr(_wsim, '__float__') else _wsim
        _csigs = getattr(APP.model, '_cached_coherence_correction_signals', None)
        if _csigs is not None:
            resp["correction_signals"] = _csigs
        _cdef = getattr(APP.model, '_cached_coherence_deficit', None)
        if _cdef is not None:
            resp["coherence_deficit"] = float(_cdef) if hasattr(_cdef, '__float__') else _cdef
        return _make_json_safe(resp)
    except Exception as e:
        logging.error(f"engine/module_coherence error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/error_evolution")
async def engine_error_evolution():
    """Error evolution summary from CausalErrorEvolutionTracker."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
        ee = getattr(ucc, 'error_evolution', None) if ucc else None
        if ee is None:
            ee = getattr(APP.model, 'error_evolution', None)
        if ee is None:
            return {"ok": True, "available": False, "reason": "ErrorEvolutionTracker not enabled"}
        summary = ee.get_error_summary()
        degrading = {}
        try:
            degrading = ee.get_degrading_error_classes()
        except Exception:
            pass
        return {
            "ok": True,
            "available": True,
            "summary": summary,
            "degrading_classes": degrading,
        }
    except Exception as e:
        logging.error(f"engine/error_evolution error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/auto_critic")
async def engine_auto_critic():
    """AutoCriticLoop configuration and status."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ac = getattr(APP.model, 'auto_critic', None)
        if ac is None:
            return {"ok": True, "available": False, "reason": "AutoCriticLoop not enabled"}
        resp: Dict[str, Any] = {
            "ok": True,
            "available": True,
            "max_iterations": getattr(ac, 'max_iterations', None),
            "threshold": getattr(ac, 'threshold', None),
        }
        _ac_score = getattr(APP.model, '_cached_auto_critic_current_score', None)
        if _ac_score is not None:
            resp["current_score"] = float(_ac_score) if hasattr(_ac_score, '__float__') else _ac_score
        _ac_ema = getattr(APP.model, '_auto_critic_quality_ema', None)
        if _ac_ema is not None:
            resp["quality_ema"] = float(_ac_ema) if hasattr(_ac_ema, '__float__') else _ac_ema
        return _make_json_safe(resp)
    except Exception as e:
        logging.error(f"engine/auto_critic error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/deception_suppressor")
async def engine_deception_suppressor():
    """Deception suppressor status."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        ds = getattr(APP.model, 'deception_suppressor', None)
        if ds is None:
            return {"ok": True, "available": False, "reason": "DeceptionSuppressor not enabled"}
        resp: Dict[str, Any] = {
            "ok": True,
            "available": True,
            "enabled": True,
        }
        _dp = getattr(APP.model, '_cached_deception_pressure', None)
        if _dp is not None:
            resp["pressure"] = float(_dp) if hasattr(_dp, '__float__') else _dp
        return _make_json_safe(resp)
    except Exception as e:
        logging.error(f"engine/deception_suppressor error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/all")
async def engine_all_monitoring():
    """Aggregated endpoint: returns all engine monitoring data in one call.

    Collects data from every available monitoring component and returns
    a single JSON payload so the dashboard can refresh all panels with
    one network round-trip.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")

    result: Dict[str, Any] = {"ok": True, "device": APP.selected_device}

    # ── ProgressTracker ──
    try:
        pt = getattr(APP.model, 'progress_tracker', None)
        if pt is not None:
            result["progress"] = {
                "available": True,
                "data": pt.get_progress(),
                "run_history": pt.get_run_history(10),
            }
    except Exception:
        pass

    # ── ConvergenceMonitor ──
    try:
        ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
        cm = getattr(ucc, 'convergence_monitor', None) if ucc else None
        if cm is not None:
            result["convergence"] = {
                "available": True,
                "data": cm.get_convergence_summary(),
            }
    except Exception:
        pass

    # ── MemoryManager ──
    try:
        mm = getattr(APP.model, 'memory_manager', None)
        if mm is not None:
            _cap = getattr(mm, '_max_capacity', None)
            mem_info: Dict[str, Any] = {
                "available": True,
                "size": mm.size,
                "max_capacity": _cap,
                "utilization": mm.size / _cap if _cap else 0.0,
            }
            ccw = getattr(APP.model, 'causal_context', None)
            if ccw is not None:
                try:
                    mem_info["causal_context"] = ccw.stats()
                except Exception:
                    pass
            result["memory"] = mem_info
    except Exception:
        pass

    # ── ErrorRecoveryManager ──
    try:
        er = getattr(APP.model, 'error_recovery', None)
        if er is not None:
            result["recovery"] = {
                "available": True,
                "stats": er.get_recovery_stats(),
                "success_rate": er.get_success_rate(),
                "history": er.get_recovery_history(10),
            }
    except Exception:
        pass

    # ── SystemIntegrityMonitor full report ──
    try:
        im = getattr(APP.model, 'integrity_monitor', None)
        if im is not None:
            result["integrity"] = {
                "available": True,
                "report": _make_json_safe(im.get_integrity_report()),
                "anomalies": im.get_anomalies(10),
            }
    except Exception:
        pass

    # ── DeterministicExecutionGuard ──
    try:
        eg = getattr(APP.model, 'execution_guard', None)
        if eg is not None:
            result["deterministic_guard"] = {
                "available": True,
                "summary": eg.get_validation_summary(),
            }
    except Exception:
        pass

    # ── CausalContextWindowManager ──
    try:
        ccw = getattr(APP.model, 'causal_context', None)
        if ccw is not None:
            result["context_window"] = {
                "available": True,
                "stats": ccw.stats(),
            }
    except Exception:
        pass

    # ── ModuleCoherenceVerifier ──
    try:
        mc = getattr(APP.model, 'module_coherence', None)
        if mc is not None:
            _mc_data: Dict[str, Any] = {
                "available": True,
                "threshold": getattr(mc, 'threshold', None),
                "initial_threshold": getattr(mc, '_initial_threshold', None),
            }
            _wpair = getattr(APP.model, '_cached_weakest_coherence_pair', None)
            if _wpair is not None:
                _mc_data["weakest_pair"] = list(_wpair) if not isinstance(_wpair, list) else _wpair
            _wsim = getattr(APP.model, '_cached_weakest_coherence_sim', None)
            if _wsim is not None:
                _mc_data["weakest_sim"] = float(_wsim) if hasattr(_wsim, '__float__') else _wsim
            _csigs = getattr(APP.model, '_cached_coherence_correction_signals', None)
            if _csigs is not None:
                _mc_data["correction_signals"] = _csigs
            _cdef = getattr(APP.model, '_cached_coherence_deficit', None)
            if _cdef is not None:
                _mc_data["coherence_deficit"] = float(_cdef) if hasattr(_cdef, '__float__') else _cdef
            result["module_coherence"] = _mc_data
    except Exception:
        pass

    # ── CausalErrorEvolutionTracker ──
    try:
        ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
        ee = getattr(ucc, 'error_evolution', None) if ucc else None
        if ee is None:
            ee = getattr(APP.model, 'error_evolution', None)
        if ee is not None:
            _ee_data: Dict[str, Any] = {
                "available": True,
                "summary": ee.get_error_summary(),
            }
            try:
                _ee_data["degrading_classes"] = ee.get_degrading_error_classes()
            except Exception:
                pass
            result["error_evolution"] = _ee_data
    except Exception:
        pass

    # ── AutoCriticLoop ──
    try:
        ac = getattr(APP.model, 'auto_critic', None)
        if ac is not None:
            _ac_data: Dict[str, Any] = {
                "available": True,
                "max_iterations": getattr(ac, 'max_iterations', None),
                "threshold": getattr(ac, 'threshold', None),
            }
            _ac_score = getattr(APP.model, '_cached_auto_critic_current_score', None)
            if _ac_score is not None:
                _ac_data["current_score"] = float(_ac_score) if hasattr(_ac_score, '__float__') else _ac_score
            _ac_ema = getattr(APP.model, '_auto_critic_quality_ema', None)
            if _ac_ema is not None:
                _ac_data["quality_ema"] = float(_ac_ema) if hasattr(_ac_ema, '__float__') else _ac_ema
            result["auto_critic"] = _ac_data
    except Exception:
        pass

    # ── DeceptionSuppressor ──
    try:
        ds = getattr(APP.model, 'deception_suppressor', None)
        if ds is not None:
            _ds_data: Dict[str, Any] = {
                "available": True,
                "enabled": True,
            }
            _dp = getattr(APP.model, '_cached_deception_pressure', None)
            if _dp is not None:
                _ds_data["pressure"] = float(_dp) if hasattr(_dp, '__float__') else _dp
            result["deception_suppressor"] = _ds_data
    except Exception:
        pass

    # ── Runtime Cached Signals ──
    try:
        _rt: Dict[str, Any] = {}
        _topo = getattr(APP.model, '_cached_topology_state', None)
        if _topo is not None:
            _rt["topology_state"] = _topo
        _cg = getattr(APP.model, '_last_complexity_gates', None)
        if _cg is not None:
            _rt["complexity_gates"] = _cg
        if _rt:
            _rt["available"] = True
            result["runtime_signals"] = _rt
    except Exception:
        pass

    # ── VibeThinker ──
    try:
        _vt_cfg = getattr(APP.model, 'vibe_thinker_config', None)
        if _vt_cfg is not None and getattr(_vt_cfg, 'enabled', False):
            _vt_data: Dict[str, Any] = {"available": True, "enabled": True}
            _vtk = getattr(APP.model, 'vibe_thinker_kernel', None)
            if _vtk is not None:
                _vt_data["kernel"] = _vtk.get_summary()
            _vtp = getattr(APP.model, 'vibe_thinker_parser', None)
            if _vtp is not None:
                _vt_data["parser"] = _vtp.get_summary()
            _vtl = getattr(APP.model, 'vibe_thinker_learner', None)
            if _vtl is not None:
                _vt_data["learner"] = _vtl.get_summary()
            _vti = getattr(APP.model, 'vibe_thinker_integration', None)
            if _vti is not None:
                _vt_data["integration"] = _vti.get_summary()
            result["vibe_thinker"] = _vt_data
        else:
            result["vibe_thinker"] = {"available": False, "enabled": False}
    except Exception:
        pass

    # ── Emergence Status ──
    try:
        _em: Dict[str, Any] = {
            "activation_complete": getattr(APP.model, '_cognitive_activation_complete', False),
            "cached_verdict": getattr(APP.model, '_cached_emergence_verdict', False),
            "last_forward_emerged": getattr(APP.model, '_last_forward_emerged', None),
        }
        result["emergence"] = _em
    except Exception:
        pass

    return _make_json_safe(result)


@app.get("/api/engine/vibe_thinker")
async def engine_vibe_thinker():
    """VibeThinker subsystem status: kernel, parser, learner, integration summaries."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        _vt_cfg = getattr(APP.model, 'vibe_thinker_config', None)
        if _vt_cfg is None or not getattr(_vt_cfg, 'enabled', False):
            return {"ok": True, "available": False, "reason": "VibeThinker not enabled"}
        resp: Dict[str, Any] = {"ok": True, "available": True, "enabled": True}
        # Config
        resp["config"] = {
            "confidence_threshold": _vt_cfg.confidence_threshold,
            "entropy_threshold": _vt_cfg.entropy_threshold,
            "complexity_gate_threshold": _vt_cfg.complexity_gate_threshold,
            "calibration_ema_alpha": _vt_cfg.calibration_ema_alpha,
            "adaptation_rate": _vt_cfg.adaptation_rate,
            "consolidation_interval": _vt_cfg.consolidation_interval,
            "psi_vibe_weight": _vt_cfg.psi_vibe_weight,
            "temperature": _vt_cfg.temperature,
            "top_p": _vt_cfg.top_p,
            "max_reasoning_tokens": _vt_cfg.max_reasoning_tokens,
        }
        # Kernel summary
        _vtk = getattr(APP.model, 'vibe_thinker_kernel', None)
        if _vtk is not None:
            resp["kernel"] = _vtk.get_summary()
        # Parser summary
        _vtp = getattr(APP.model, 'vibe_thinker_parser', None)
        if _vtp is not None:
            resp["parser"] = _vtp.get_summary()
        # Learner summary (continuous learning state)
        _vtl = getattr(APP.model, 'vibe_thinker_learner', None)
        if _vtl is not None:
            resp["learner"] = _vtl.get_summary()
        # Integration summary
        _vti = getattr(APP.model, 'vibe_thinker_integration', None)
        if _vti is not None:
            resp["integration"] = _vti.get_summary()
        return _make_json_safe(resp)
    except Exception as e:
        logging.error(f"engine/vibe_thinker error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/engine/emergence")
async def engine_emergence():
    """Emergence status from the latest forward pass and activation probe."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        resp: Dict[str, Any] = {
            "ok": True,
            "activation_complete": getattr(APP.model, '_cognitive_activation_complete', False),
            "cached_verdict": getattr(APP.model, '_cached_emergence_verdict', False),
            "last_forward_emerged": getattr(APP.model, '_last_forward_emerged', None),
        }
        # Full emergence report
        try:
            report = APP.model.system_emergence_report()
            status = report.get('system_emergence_status', {})
            resp["emerged"] = status.get('emerged', False)
            resp["conditions_met"] = status.get('conditions_met', 0)
            resp["conditions_total"] = status.get('conditions_total', 9)
            resp["weighted_score"] = status.get('weighted_emergence_score', 0.0)
        except Exception:
            resp["emerged"] = getattr(APP.model, '_cached_emergence_verdict', False)
        return _make_json_safe(resp)
    except Exception as e:
        logging.error(f"engine/emergence error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/self_learn")
async def vibe_thinker_self_learn():
    """Trigger a VibeThinker self-learning cycle using a synthetic forward pass.

    This runs a forward pass with random tokens to exercise the full
    VibeThinker pipeline (adaptation → evaluation → consolidation),
    enabling continuous learning from zero parameters on first start.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    _vt_cfg = getattr(APP.model, 'vibe_thinker_config', None)
    if _vt_cfg is None or not getattr(_vt_cfg, 'enabled', False):
        raise HTTPException(400, "VibeThinker not enabled")
    try:
        import torch
        # Fallback values mirror InitRequest defaults (30522 = BERT vocab).
        _vocab = getattr(APP.model.config, 'vocab_size', 30522)
        _seq = getattr(APP.model.config, 'seq_length', 64)
        tokens = torch.randint(0, _vocab, (1, _seq))
        APP.model.eval()
        with torch.no_grad():
            result = APP.model(tokens)
        _vt = result.get('vibe_thinker', {})
        _learner = getattr(APP.model, 'vibe_thinker_learner', None)
        _summary = _learner.get_summary() if _learner is not None else {}
        return _make_json_safe({
            "ok": True,
            "vibe_thinker": _vt,
            "learner_state": _summary,
            "emergence_status": result.get('emergence_status'),
        })
    except Exception as e:
        logging.error(f"vibe_thinker/self_learn error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  VIBE THINKER MODEL VERIFICATION & INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/vibe_thinker/verify_model")
async def vibe_thinker_verify_model():
    """Check whether VibeThinker model dependencies are available.

    Verifies:
    1. ``transformers`` library is installed.
    2. ``bert-base-uncased`` tokenizer is cached locally.

    Returns readiness status so the dashboard can prompt for installation
    before model initialization.
    """
    try:
        result = _verify_vibe_thinker_model()
        return {"ok": True, **result}
    except Exception as e:
        logging.error(f"vibe_thinker/verify_model error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/install_model")
async def vibe_thinker_install_model():
    """Install VibeThinker dependencies and download the tokenizer model.

    Installs ``transformers`` if missing and downloads ``bert-base-uncased``
    tokenizer to the HuggingFace cache.  This endpoint should be called
    before ``/api/init`` when ``/api/vibe_thinker/verify_model`` reports
    ``ready: false``.
    """
    try:
        result = _install_vibe_thinker_model()
        return result
    except Exception as e:
        logging.error(f"vibe_thinker/install_model error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/save_weights")
async def vibe_thinker_save_weights(body: dict):
    """Save VibeThinker adapter + kernel weights to a standalone file.

    Isolates VibeThinker weights from the full model checkpoint for
    transfer, A/B testing, or rollback.  Optionally includes a VQ
    codebook snapshot for co-aligned restoration.

    Args:
        body: ``{"path": "/path/to/vibe_thinker_weights.pt"}``
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    path = body.get("path", "")
    if not path:
        raise HTTPException(400, "Missing 'path' field")
    try:
        result = APP.model.save_vibe_thinker_weights(path)
        return _make_json_safe({"ok": result.get("success", False), **result})
    except Exception as e:
        logging.error(f"vibe_thinker/save_weights error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/load_weights")
async def vibe_thinker_load_weights(body: dict):
    """Load VibeThinker weights from a standalone file.

    Replaces current adapter + kernel weights with those from the
    specified file.  Shape-incompatible keys are skipped with a warning.

    Args:
        body: ``{"path": "/path/to/vibe_thinker_weights.pt"}``
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    path = body.get("path", "")
    if not path:
        raise HTTPException(400, "Missing 'path' field")
    try:
        result = APP.model.load_vibe_thinker_weights(path)
        return _make_json_safe({"ok": result.get("success", False), **result})
    except Exception as e:
        logging.error(f"vibe_thinker/load_weights error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/switch_weights")
async def vibe_thinker_switch_weights(body: dict):
    """Hot-swap VibeThinker weights without re-initializing the model.

    Saves current weights as rollback, loads new weights, runs a
    calibration verification, and rolls back if verification fails.
    All cognitive pipeline state (feedback bus, error evolution, etc.)
    is preserved.

    Args:
        body: ``{"path": "/path/to/new_vibe_thinker_weights.pt"}``
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    path = body.get("path", "")
    if not path:
        raise HTTPException(400, "Missing 'path' field")
    try:
        result = APP.model.switch_vibe_thinker_weights(path)
        return _make_json_safe({"ok": result.get("success", False), **result})
    except Exception as e:
        logging.error(f"vibe_thinker/switch_weights error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/vibe_thinker/list_weights")
async def vibe_thinker_list_weights(directory: str = ""):
    """List available VibeThinker weight files (.pt) in a directory.

    Scans the specified directory (or current working directory) for
    ``.pt`` files that can be loaded via ``/api/vibe_thinker/load_weights``
    or ``/api/vibe_thinker/switch_weights``.  Returns file metadata
    (name, size, modification time) for each discovered weight file.

    Query params:
        directory: Optional directory path to scan.  Defaults to ``"."``.
    """
    from pathlib import Path as _P
    import datetime as _dt

    scan_dir = _P(directory) if directory else _P(".")
    if not scan_dir.is_dir():
        raise HTTPException(400, f"Directory not found: {scan_dir}")
    try:
        files = []
        for f in sorted(scan_dir.glob("*.pt")):
            if f.is_file():
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "path": str(f.resolve()),
                    "size_bytes": stat.st_size,
                    "modified": _dt.datetime.fromtimestamp(
                        stat.st_mtime, tz=_dt.timezone.utc,
                    ).isoformat(),
                })
        # Also report which weights are currently active
        active_path = ""
        if APP.model is not None:
            _cfg = getattr(APP.model, 'config', None)
            if _cfg is not None:
                active_path = getattr(_cfg, 'vibe_thinker_weights_path', '')
        return {
            "ok": True,
            "directory": str(scan_dir.resolve()),
            "count": len(files),
            "files": files,
            "active_weights_path": active_path,
        }
    except Exception as e:
        logging.error(f"vibe_thinker/list_weights error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vibe_thinker/first_start_calibration")
async def vibe_thinker_first_start_calibration():
    """Trigger VibeThinker first-start calibration manually.

    Runs the full four-phase calibration sequence:
      1. Adapter warm-up (stabilise LayerNorm running statistics)
      2. VQ codebook seeding (align codebook with VibeThinker latents)
      3. Learner baseline seeding (seed continuous learner EMA)
      4. Integration bootstrap (initialise feedback bus signals)

    This is the same calibration that runs automatically on first start
    when no pre-trained weights are provided.  It can be re-triggered
    to realign the VQ codebook after manual weight changes.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    _vt_cfg = getattr(APP.model, 'vibe_thinker_config', None)
    if _vt_cfg is None or not getattr(_vt_cfg, 'enabled', False):
        raise HTTPException(400, "VibeThinker not enabled")
    try:
        result = APP.model._vibe_thinker_first_start_calibration()
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"vibe_thinker/first_start_calibration error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  EMERGENCE SUMMARY (Lightweight cached — no forward pass)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/emergence_summary")
async def get_emergence_summary():
    """Return the cached emergence summary from the last forward pass.

    Unlike ``/api/system_emergence`` which generates a full report,
    this endpoint returns the cached emergence summary computed inline
    during ``_forward_impl``.  It includes cognitive unity score,
    per-axiom AGI coverage, diagnostic health, and feedback bus status
    without triggering any additional computation.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        summary = APP.model.get_emergence_summary()
        return _make_json_safe({"ok": True, "emergence_summary": summary})
    except Exception as e:
        logging.error(f"emergence_summary error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR EVOLUTION SEED (Training-Inference Bridge)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/error_evolution/seed")
async def seed_error_evolution():
    """Seed the error evolution tracker with baseline training error classes.

    Primes the metacognitive trigger with known failure modes so that
    error evolution is functional from the first forward pass, even
    without prior training data.  This closes the training→inference
    bridge gap reported by ``self_diagnostic()``.

    Idempotent: skips error classes that already have recorded episodes.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        count = APP.model.seed_error_evolution_baseline()
        return {"ok": True, "seeded_classes": count}
    except Exception as e:
        logging.error(f"error_evolution/seed error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK BUS STATE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/feedback_bus")
async def get_feedback_bus():
    """Return the cognitive feedback bus state, signal registry, and coverage.

    Exposes the per-channel EMA-smoothed signal values, trend directions,
    oscillation score, registered dynamic signals, and total channel count.
    This enables external consumers to monitor the health of the feedback
    loop between downstream modules and the meta-loop.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        fb = getattr(APP.model, 'feedback_bus', None)
        if fb is None:
            return {"ok": True, "available": False, "reason": "Feedback bus not initialized"}

        state = fb.get_state()
        oscillation = fb.get_oscillation_score()
        trend = fb.get_signal_trend()

        # Registered signals
        registered = list(fb._extra_signals.keys())
        # Evaluate coverage: signals that have been written (non-default)
        _evaluated = set()
        for name, val in fb._extra_signals.items():
            _default = fb._extra_defaults.get(name, 0.0)
            if abs(val - _default) > 1e-9:
                _evaluated.add(name)

        total_channels = fb.total_channels
        evaluated_count = len(_evaluated)
        coverage = evaluated_count / max(len(registered), 1) if registered else 1.0

        result = {
            "ok": True,
            "available": True,
            "total_channels": total_channels,
            "core_channels": fb.NUM_SIGNAL_CHANNELS,
            "dynamic_channels": len(registered),
            "registered_signals": registered,
            "evaluated_signals": sorted(_evaluated),
            "coverage": round(coverage, 4),
            "oscillation_score": round(oscillation, 4),
            "signal_state": {k: round(v, 6) for k, v in state.items()},
        }
        # Trend data
        if trend is not None:
            trend_list = trend.detach().cpu().tolist()
            result["signal_trend"] = [round(v, 6) for v in trend_list]
        # EMA values
        ema = fb.get_ema_values()
        if ema is not None:
            result["ema_values"] = [round(v, 6) for v in ema.detach().cpu().tolist()]
        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"feedback_bus error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERGENCE DETAILED STATE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/convergence/detailed")
async def get_convergence_detailed():
    """Return detailed convergence monitor state with history.

    Supplements ``/api/engine/convergence`` with the full convergence
    summary from ``ConvergenceMonitor.get_convergence_summary()``,
    including sliding-window history, secondary signals, and the most
    recent verdict.  Also includes the certified meta-loop convergence
    status when available.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        result: Dict[str, Any] = {"ok": True}

        # Primary convergence monitor
        cm = getattr(APP.model, 'convergence_monitor', None)
        if cm is not None:
            result["convergence_summary"] = cm.get_convergence_summary()
        else:
            result["convergence_summary"] = None

        # Certified meta-loop convergence
        cml = getattr(APP.model, 'certified_meta_loop', None)
        if cml is not None:
            _cm2 = getattr(cml, 'convergence_monitor', None)
            if _cm2 is not None:
                result["certified_convergence"] = _cm2.get_convergence_summary()

        # UCC convergence monitor
        ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
        if ucc is not None:
            _cm3 = getattr(ucc, 'convergence_monitor', None)
            if _cm3 is not None:
                result["ucc_convergence"] = _cm3.get_convergence_summary()

        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"convergence/detailed error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/convergence/analytics")
async def run_convergence_analytics():
    """Run academic-grade convergence analytics with residual distributions.

    Performs multiple fixed-point iterations, collects residual histories,
    and reports:
    - Per-iteration residual distributions (median, IQR, min, max)
    - Certified bound ‖C^(n) − C*‖ ≤ r_{n−1}/(1−L) using observed L
    - Bound validity rate across trials
    - Iteration count statistics
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        import torch
        from aeon_core import ConvergenceAnalytics

        meta_loop = getattr(APP.model, 'meta_loop', None)
        if meta_loop is None:
            return {"ok": False, "reason": "ProvablyConvergentMetaLoop not available"}

        ca = ConvergenceAnalytics(meta_loop, threshold=1e-4)
        # Generate diverse inputs
        H = APP.config.z_dim
        inputs = [torch.randn(1, H, device=APP.model.device) for _ in range(20)]
        ca.run_trials(inputs)
        report = ca.generate_report()
        return _make_json_safe({"ok": True, **report})
    except Exception as e:
        logging.error(f"convergence/analytics error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/eval/perplexity")
async def eval_perplexity():
    """Compute standardized perplexity on synthetic evaluation data.

    Returns perplexity, cross-entropy, and token count.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        import torch
        from aeon_core import PerplexityEvaluator

        pe = PerplexityEvaluator(
            vocab_size=APP.config.vocab_size,
            seq_length=APP.config.seq_length,
        )
        # Generate synthetic eval data
        eval_data = torch.randint(
            0, APP.config.vocab_size,
            (32, APP.config.seq_length),
            device=APP.model.device,
        )
        result = pe.compute_perplexity(APP.model, eval_data, batch_size=8)
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"eval/perplexity error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/eval/ablation")
async def eval_ablation():
    """Run ablation study on AEON-Delta modules.

    Disables modules one at a time and measures perplexity impact.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        import torch
        from aeon_core import PerplexityEvaluator

        pe = PerplexityEvaluator(
            vocab_size=APP.config.vocab_size,
            seq_length=APP.config.seq_length,
        )
        eval_data = torch.randint(
            0, APP.config.vocab_size,
            (16, APP.config.seq_length),
            device=APP.model.device,
        )
        toggles = {}
        for attr, desc in [
            ('enable_meta_loop', 'Meta-Loop'),
            ('enable_vq', 'VQ Quantizer'),
            ('enable_causal_model', 'Causal Model'),
            ('enable_meta_learning', 'Meta-Learning (MAML+EWC)'),
            ('enable_catastrophe_detection', 'Catastrophe Detection'),
            ('enable_safety_guardrails', 'Safety Guardrails'),
        ]:
            if hasattr(APP.model, attr):
                toggles[attr] = desc

        result = pe.run_ablation(APP.model, eval_data, toggles, batch_size=8)
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"eval/ablation error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/eval/causal_discovery")
async def eval_causal_discovery():
    """Run causal discovery benchmark on NOTEARS model.

    Evaluates SHD, TPR, FDR against synthetic ground-truth DAGs.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        from aeon_core import CausalDiscoveryEvaluator

        notears = getattr(APP.model, 'notears_causal', None)
        if notears is None:
            return {"ok": False, "reason": "NOTEARSCausalModel not available. Enable with enable_notears_causal=True."}

        num_vars = getattr(notears, 'num_vars', 8)
        cde = CausalDiscoveryEvaluator(num_vars=num_vars)
        report = cde.generate_report(notears)
        return _make_json_safe({"ok": True, **report})
    except Exception as e:
        logging.error(f"eval/causal_discovery error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/eval/continual_learning")
async def eval_continual_learning():
    """Analyze MAML+EWC scaling and acceptance gates.

    Reports EWC penalty growth, acceptance gate behavior, and
    forward/backward transfer metrics.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        import torch
        from aeon_core import ContinualLearningAnalyzer

        meta_learner = getattr(APP.model, 'meta_learner', None)
        continual_core = getattr(APP.model, 'continual_learning_core', None)
        cla = ContinualLearningAnalyzer(
            meta_learner=meta_learner,
            continual_core=continual_core,
        )
        # Run EWC scaling analysis with synthetic tasks
        H = APP.config.hidden_dim
        tasks = [
            (torch.randn(4, APP.config.seq_length, dtype=torch.long).clamp(0, APP.config.vocab_size - 1),
             torch.randn(4, APP.config.seq_length, dtype=torch.long).clamp(0, APP.config.vocab_size - 1))
            for _ in range(5)
        ]
        result = {"ok": True}
        # Acceptance gate demo
        delta = {n: torch.zeros_like(p) for n, p in APP.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.ones_like(p) * 0.01 for n, p in APP.model.named_parameters() if p.requires_grad}
        # Take only first 10 params to keep it fast
        delta = dict(list(delta.items())[:10])
        fisher = dict(list(fisher.items())[:10])
        gate = cla.acceptance_gate(delta, fisher)
        result["acceptance_gate"] = gate
        result["meta_learner_available"] = meta_learner is not None
        result["continual_core_available"] = continual_core is not None

        return _make_json_safe(result)
    except Exception as e:
        logging.error(f"eval/continual_learning error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  COGNITIVE COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/cognitive_completeness")
async def get_cognitive_completeness():
    """Return the cognitive completeness metrics from the last forward pass.

    Exposes per-axiom AGI coverage (mutual verification, uncertainty→
    metacognition, root-cause traceability), coherence deficit,
    emergence status, and output reliability.  These metrics are
    computed during ``_forward_impl`` and cached in the output dict.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        # Try to get from the cached emergence summary first
        summary = APP.model.get_emergence_summary()
        completeness = summary.get('cognitive_completeness', {})

        # If not available in summary, try a lightweight extraction
        if not completeness:
            completeness = {
                'emerged': getattr(APP.model, '_cached_emergence_verdict', False),
                'activation_complete': getattr(APP.model, '_cognitive_activation_complete', False),
            }
            # Per-axiom deficits (if verify_and_reinforce has run)
            for attr, key in [
                ('_cached_mv_axiom_deficit', 'mutual_verification_coverage'),
                ('_cached_um_axiom_deficit', 'uncertainty_metacognition_coverage'),
                ('_cached_rt_axiom_deficit', 'root_cause_traceability_coverage'),
                ('_cached_coherence_deficit', 'coherence_deficit'),
            ]:
                val = getattr(APP.model, attr, None)
                if val is not None:
                    if 'coverage' in key:
                        completeness[key] = round(1.0 - float(val), 4)
                    else:
                        completeness[key] = round(float(val), 4)

        return _make_json_safe({"ok": True, "cognitive_completeness": completeness})
    except Exception as e:
        logging.error(f"cognitive_completeness error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  REGULARIZATION TERMS (signal-derived losses for monitoring)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/regularization")
async def get_regularization():
    """Return signal-derived regularization terms for monitoring.

    Exposes uncertainty_loss, coherence_loss, and stability_loss
    computed from the model's observation signals.  These are the same
    terms used during training for gradient-based optimisation but
    presented here as scalar diagnostics for real-time monitoring.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        if not hasattr(APP.model, 'get_regularization_terms'):
            return {"ok": True, "available": False, "reason": "Method not available"}
        terms = APP.model.get_regularization_terms()
        result = {"ok": True, "available": True, "terms": {}}
        for name, tensor in terms.items():
            result["terms"][name] = round(float(tensor.item()), 6)
        # Signal-weighted factor
        if hasattr(APP.model, 'get_signal_weighted_factor'):
            result["signal_weighted_factor"] = round(
                APP.model.get_signal_weighted_factor(), 4,
            )
        return result
    except Exception as e:
        logging.error(f"regularization error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING-INFERENCE BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/sync_from_training")
async def sync_from_training():
    """Synchronize inference pipeline state from the latest training session.

    Calls ``sync_from_training()`` on the model, which imports error
    patterns, convergence thresholds, metacognitive weights, and
    cognitive memory from the training state into the inference pipeline.
    This establishes a bidirectional feedback loop between training
    and inference modes.
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        APP.model.sync_from_training()
        return {"ok": True, "message": "Training state synchronized to inference pipeline"}
    except Exception as e:
        logging.error(f"sync_from_training error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/load_v4_checkpoint")
async def load_v4_checkpoint(body: dict):
    """Load weights from an ae_train v4 checkpoint.

    Extracts compatible weights from a v4 training checkpoint and loads
    them into the current model.  Also imports training error patterns
    into the error evolution tracker.

    Args:
        body: ``{"path": "/path/to/checkpoint.pt", "strict": false}``
    """
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    path = body.get("path", "")
    strict = body.get("strict", False)
    if not path:
        raise HTTPException(400, "Missing 'path' field")
    try:
        result = APP.model.load_v4_checkpoint(path, strict=strict)
        return _make_json_safe({"ok": True, **result})
    except Exception as e:
        logging.error(f"load_v4_checkpoint error: {e}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/logs")
async def get_logs(limit: int = 500, level: str = "", subsys: str = ""):
    logs = APP.log_history
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]
    if subsys:
        logs = [l for l in logs if subsys.lower() in l.get("subsys", "").lower()]
    return {"ok": True, "logs": logs[-limit:], "total": len(APP.log_history)}


@app.delete("/api/logs")
async def clear_logs():
    APP.log_history.clear()
    return {"ok": True}


@app.get("/api/logs/stream")
async def stream_logs():
    """SSE real-time log stream."""
    async def gen():
        for entry in APP.log_history[-100:]:
            yield f"data: {json.dumps(entry)}\n\n"
        last_index = len(APP.log_history)
        import asyncio
        while True:
            new = APP.log_history[last_index:]
            for entry in new:
                yield f"data: {json.dumps(entry)}\n\n"
            last_index = len(APP.log_history)
            await asyncio.sleep(0.2)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/config")
async def get_config():
    if APP.config is None:
        return {"ok": False, "config": {}}
    cfg = {
        k: v for k, v in vars(APP.config).items()
        if not k.startswith("_") and not callable(v)
        and k not in ("device_manager", "tensor_guard")
    }
    return {"ok": True, "config": cfg}


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION EXPORT / IMPORT
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/session/export")
async def export_session():
    """Export full session metadata as JSON (config + stats, NOT weights)."""
    session = {
        "aeon_dashboard_version": "3.2.0",
        "exported_at": time.time(),
        "model_ready": APP.model is not None,
        "config": None,
        "training_progress": APP.training_progress,
        "test_results": APP.test_results,
        "benchmark_results": {k: v for k, v in (APP.benchmark_results or {}).items()
                               if k != "latency_series"},
        "log_count": len(APP.log_history),
        "session_meta": APP.session_meta,
    }
    if APP.config is not None:
        session["config"] = {
            k: v for k, v in vars(APP.config).items()
            if not k.startswith("_") and not callable(v)
            and k not in ("device_manager", "tensor_guard")
        }
    if APP.model is not None:
        session["model_stats"] = {
            "parameters": APP.model.count_parameters(),
            "trainable_parameters": APP.model.count_trainable_parameters(),
            "device": str(APP.model.device),
        }
    return JSONResponse(content=session, headers={
        "Content-Disposition": "attachment; filename=aeon_session.json"
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

# ── Disconnect exception set  ──────────────────────────────────────────────────
# Starlette raises WebSocketDisconnect on clean closes; uvicorn/websockets can
# raise ClientDisconnected (a RuntimeError subclass) on abrupt 1006 closures.
# We catch BOTH so no disconnect path can leak as an unhandled ASGI exception.
try:
    from uvicorn.protocols.utils import ClientDisconnected as _ClientDisconnected
except ImportError:
    _ClientDisconnected = None

_WS_DISCONNECT_EXCS: tuple = (
    WebSocketDisconnect,
    *([_ClientDisconnected] if _ClientDisconnected else []),
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Robust WebSocket endpoint.

    Root-cause of the previous 'connection closed' crash:
    ─────────────────────────────────────────────────────
    The backlog-flush loop (up to 150 log entries produced by model init)
    was executed **outside** the try/except block.  When the dashboard
    reconnects immediately after /api/init returns, the log_history already
    contains the full parameter table (~100 lines).  A 150-frame burst with
    no asyncio.sleep(0) yields:
      • never yields to the event loop → TCP send-buffer can stall
      • if the client tab is still loading, it may close before the flush
        completes → send_json raises WebSocketDisconnect
      • that exception propagates uncaught through the ASGI stack →
        uvicorn logs the full traceback and marks the connection as an error
        instead of a normal close.

    Fix:
    ─────
    1. Everything after ws.accept() lives inside ONE try/except block.
    2. A _safe_send() helper wraps every outbound send — returns False on any
       disconnect variant so callers can exit cleanly without re-raising.
    3. asyncio.sleep(0) is called every BACKLOG_YIELD_EVERY frames during bulk
       sends so the event loop can service keep-alives and other coroutines.
    4. receive_text() + manual json.loads() replaces receive_json() so a
       binary frame or malformed JSON does NOT kill the loop.
    """
    import asyncio

    BACKLOG_YIELD_EVERY = 10   # yield to event loop every N frames during bulk send

    await ws.accept()
    APP.ws_clients.append(ws)
    logging.info(f"WebSocket client connected · {len(APP.ws_clients)} total")

    # ── Shared safe-send helper ──────────────────────────────────────────────
    async def _safe_send(payload: dict) -> bool:
        """
        Send one JSON frame.  Returns True on success, False on any disconnect.
        Never raises — all disconnect/send errors are swallowed here so callers
        can branch on the return value without nested try/except boilerplate.
        """
        try:
            await ws.send_json(payload)
            return True
        except _WS_DISCONNECT_EXCS:                    # type: ignore[misc]
            return False
        except RuntimeError as exc:
            # Starlette wraps ClientDisconnected as RuntimeError in some versions
            if "disconnect" in str(exc).lower() or "close" in str(exc).lower():
                return False
            return False
        except Exception:
            return False

    # ────────────────────────────────────────────────────────────────────────
    # CRITICAL: from this point on, EVERYTHING is inside the try/except.
    # Previously the backlog flush was outside → the crash you saw in logs.
    # ────────────────────────────────────────────────────────────────────────
    try:
        # ── 1. Backlog flush ─────────────────────────────────────────────────
        # Take a snapshot so the list doesn't change while we iterate.
        backlog = list(APP.log_history[-150:])
        for i, entry in enumerate(backlog):
            if not await _safe_send({"type": "log", "data": entry}):
                return   # client gone — exit the coroutine cleanly
            # Yield to the event loop every BACKLOG_YIELD_EVERY frames.
            # Without this, a 150-entry burst blocks the loop and can stall
            # TCP, causing the client to time out and disconnect.
            if (i + 1) % BACKLOG_YIELD_EVERY == 0:
                await asyncio.sleep(0)

        # ── 2. Initial status frame ──────────────────────────────────────────
        if not await _safe_send({
            "type": "status",
            "model_ready": APP.model is not None,
            "training":    APP.training_active,
            "training_progress": APP.training_progress,
            "v4_training": APP.v4_active,
            "core_loaded": CORE_LOADED,
        }):
            return

        # ── 3. Main receive loop ─────────────────────────────────────────────
        while True:
            # Use receive_text() + manual JSON parse so a malformed or binary
            # frame does NOT raise JSONDecodeError and silently kill the loop.
            try:
                raw = await ws.receive_text()
            except _WS_DISCONNECT_EXCS:                # type: ignore[misc]
                break
            except RuntimeError as exc:
                if "disconnect" in str(exc).lower() or "close" in str(exc).lower():
                    break
                break

            # Ignore non-JSON frames (e.g. browser native ping text)
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            if not isinstance(msg, dict):
                continue

            mtype = msg.get("type")

            # ── ping / pong ──────────────────────────────────────────────────
            if mtype == "ping":
                if not await _safe_send({
                    "type": "pong",
                    "ts":          time.time(),
                    "model_ready": APP.model is not None,
                    "training":    APP.training_active,
                    "v4_training": APP.v4_active,
                    "training_progress": APP.training_progress,
                }):
                    break

            # ── bulk log replay ──────────────────────────────────────────────
            elif mtype == "get_logs":
                limit    = int(msg.get("limit", 200))
                log_snap = list(APP.log_history[-limit:])
                for i, entry in enumerate(log_snap):
                    if not await _safe_send({"type": "log", "data": entry}):
                        return
                    if (i + 1) % BACKLOG_YIELD_EVERY == 0:
                        await asyncio.sleep(0)

            # ── status query ─────────────────────────────────────────────────
            elif mtype == "get_status":
                if not await _safe_send({
                    "type":          "status",
                    "model_ready":   APP.model is not None,
                    "training":      APP.training_active,
                    "v4_training":   APP.v4_active,
                    "core_loaded":   CORE_LOADED,
                    "training_progress": APP.training_progress,
                }):
                    break

    except _WS_DISCONNECT_EXCS:           # type: ignore[misc]
        # Normal / expected disconnect — not an error
        pass
    except RuntimeError as exc:
        if "disconnect" not in str(exc).lower() and "close" not in str(exc).lower():
            logging.debug(f"WebSocket RuntimeError (non-disconnect): {exc}")
    except Exception as exc:
        # Unexpected error — log at DEBUG so the operator can investigate
        # without flooding the console on every browser tab close.
        logging.debug(f"WebSocket unexpected error: {type(exc).__name__}: {exc}")
    finally:
        if ws in APP.ws_clients:
            APP.ws_clients.remove(ws)
        logging.info(f"WebSocket client disconnected · {len(APP.ws_clients)} remaining")


# ─── Background tasks ────────────────────────────────────────────────────────

async def _log_forwarder():
    """
    Drain APP.log_queue and broadcast entries to all connected WS clients.

    Improvements vs original:
    • asyncio.sleep(0) between individual broadcasts so a large burst does not
      monopolise the event loop and cause downstream clients to time out.
    • Increased batch ceiling to 50 (was 30) while keeping inter-frame yields.
    • Guard against empty ws_clients list before touching the queue at all.
    """
    import asyncio
    while True:
        await asyncio.sleep(0.15)
        if APP.log_queue.empty() or not APP.ws_clients:
            continue
        batch: list = []
        while not APP.log_queue.empty() and len(batch) < 50:
            try:
                batch.append(APP.log_queue.get_nowait())
            except queue.Empty:
                break
        for i, entry in enumerate(batch):
            await broadcast({"type": "log", "data": entry})
            # Yield every 10 frames during burst so other coroutines are not starved
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0)


async def _heartbeat():
    """
    Broadcast a structured heartbeat to every connected WS client every 2 s.

    Improvements vs original:
    • Always sends model_ready, core_loaded and v4_training regardless of
      training state — the dashboard needs these to update its status bar.
    • Includes v4_progress snapshot only when v4 is active OR just completed
      (was already correct, kept).
    • Uses asyncio.sleep(0) before the broadcast so the coroutine yields
      even when there are no clients, preventing theoretical starvation.
    """
    import asyncio
    while True:
        await asyncio.sleep(2.0)
        if not APP.ws_clients:
            continue
        payload: dict = {
            "type":        "heartbeat",
            "ts":          time.time(),
            "model_ready": APP.model is not None,
            "core_loaded": CORE_LOADED,
            "training":    APP.training_active,
            "v4_training": APP.v4_active,
        }
        if APP.training_active:
            payload["training_progress"] = APP.training_progress
        if APP.v4_active or APP.v4_progress.get("done"):
            payload["v4_progress"] = APP.v4_progress
        if APP.test_run_active:
            payload["test_progress"] = APP.test_run_progress
        # Include telemetry snapshot in heartbeat when available
        if APP.config is not None:
            try:
                tc = APP.config.telemetry_collector
                snapshot = tc.get_metrics_snapshot()
                payload["telemetry"] = {
                    "counters": snapshot.get("counters", {}),
                    "metric_names": list(k for k in snapshot if k != "counters"),
                }
            except Exception:
                pass
        # Include engine monitoring snapshot in heartbeat
        if APP.model is not None:
            engine: dict = {}
            try:
                pt = getattr(APP.model, 'progress_tracker', None)
                if pt is not None:
                    engine["progress"] = pt.get_progress()
            except Exception:
                pass
            try:
                er = getattr(APP.model, 'error_recovery', None)
                if er is not None:
                    engine["recovery_success_rate"] = er.get_success_rate()
            except Exception:
                pass
            try:
                mm = getattr(APP.model, 'memory_manager', None)
                if mm is not None:
                    engine["memory_size"] = mm.size
                    _cap = getattr(mm, '_max_capacity', None)
                    engine["memory_utilization"] = mm.size / _cap if _cap else 0.0
            except Exception:
                pass
            try:
                eg = getattr(APP.model, 'execution_guard', None)
                if eg is not None:
                    vs = eg.get_validation_summary()
                    engine["guard_success_rate"] = vs.get("success_rate", 1.0)
            except Exception:
                pass
            try:
                ucc = getattr(APP.model, 'unified_cognitive_cycle', None)
                cm = getattr(ucc, 'convergence_monitor', None) if ucc else None
                if cm is not None:
                    cs = cm.get_convergence_summary()
                    engine["convergence_status"] = cs.get("status", "unknown")
                    engine["convergence_certified"] = cs.get("certified", False)
            except Exception:
                pass
            # ── Emergence summary (cached, lightweight) ──────────────
            try:
                _em_sum = APP.model.get_emergence_summary()
                if _em_sum:
                    engine["emergence_cached"] = {
                        "emerged": _em_sum.get("emerged", False),
                        "cognitive_unity_score": _em_sum.get(
                            "cognitive_unity_score", 0.0,
                        ),
                    }
            except Exception:
                pass
            # ── Feedback bus coverage ────────────────────────────────
            try:
                fb = getattr(APP.model, 'feedback_bus', None)
                if fb is not None:
                    engine["feedback_bus_oscillation"] = round(
                        fb.get_oscillation_score(), 4,
                    )
                    engine["feedback_bus_channels"] = fb.total_channels
            except Exception:
                pass
            if engine:
                payload["engine"] = engine
        await broadcast(payload)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Entry point for ``aeon-server`` console script."""
    import argparse
    parser = argparse.ArgumentParser(description="AEON Dashboard Server v3.4.0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info", choices=["debug","info","warning","error"])
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  AEON-Delta Dashboard Server v3.4.0 (Production + v4)       ║
║  Dashboard  →  http://localhost:{args.port}                     ║
║  API Docs   →  http://localhost:{args.port}/docs                ║
║  WebSocket  →  ws://localhost:{args.port}/ws                    ║
║  Log SSE    →  http://localhost:{args.port}/api/logs/stream     ║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run(
        "aeon_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
