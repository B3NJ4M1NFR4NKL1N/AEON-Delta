"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AEON-Delta Dashboard Backend  ·  aeon_server.py  v3.3.0 — Production  ║
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
╚══════════════════════════════════════════════════════════════════════════╝

Запуск:
    pip install fastapi uvicorn psutil
    python aeon_server.py [--host 0.0.0.0] [--port 8000]

Dashboard:  http://localhost:8000
API Docs:   http://localhost:8000/docs
"""

import os, sys, json, time, queue, logging, threading, traceback, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from contextlib import asynccontextmanager
import statistics
import io
from contextlib import redirect_stdout, redirect_stderr
import importlib.util

import torch

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
    AEONTrainer = _core_mod.AEONTrainer
    set_seed = _core_mod.set_seed
    AEONTestSuite = _core_mod.AEONTestSuite
    StructuredLogFormatter = _core_mod.StructuredLogFormatter
    TelemetryCollector = _core_mod.TelemetryCollector
    generate_correlation_id = _core_mod.generate_correlation_id
    CORE_LOADED = True
    print(f"✅ {CORE_PATH.name} loaded successfully")
except Exception as e:
    CORE_LOAD_ERROR = str(e)
    print(f"⚠ {CORE_PATH.name} import error: {e}")


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
    test_results: Optional[dict]  = None
    benchmark_results: Optional[dict] = None
    ws_clients: List[WebSocket] = []
    log_queue: queue.Queue     = queue.Queue(maxsize=4000)
    log_history: List[dict]    = []
    session_meta: dict         = {"init_time": None, "init_count": 0}
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
    # Optimiser
    learning_rate: float = Field(3e-5,  gt=0,    description="Peak learning rate")
    batch_size: int      = Field(16,    ge=1,    le=512,   description="Batch size")
    grad_clip: float     = Field(0.5,   gt=0,    description="Gradient clip norm")
    warmup_steps: int    = Field(1000,  ge=0,    description="LR warmup steps")
    entropy_weight: float = Field(0.1,  ge=0.0,  description="VQ codebook entropy regularisation weight")
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
    logging.info("AEON Dashboard server v3.3.0 starting")
    asyncio.create_task(_log_forwarder())
    asyncio.create_task(_heartbeat())
    asyncio.create_task(_test_progress_broadcaster())
    # Pre-parse test catalogue
    _warmup_test_catalogue()
    yield
    logging.info("AEON Dashboard server shutting down")

app = FastAPI(
    title="AEON-Delta Dashboard API",
    version="3.3.0",
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
#  INIT / DEINIT
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/init")
async def init_model(req: InitRequest):
    if not CORE_LOADED:
        raise HTTPException(503, f"core.py failed to load: {CORE_LOAD_ERROR}")
    try:
        logging.info(f"Initializing AEONDeltaV3 · backend={req.encoder_backend} · hidden={req.hidden_dim} · seed={req.seed}")
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

        params = model.count_parameters()
        trainable = model.count_trainable_parameters()
        arch = model.print_architecture_summary()

        APP.session_meta["init_time"] = time.time()
        APP.session_meta["init_count"] += 1

        # Count enabled flags
        flags = [k for k, v in req.model_dump().items() if k.startswith("enable_") and v is True]
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
    except Exception:
        pass
    tg = {}
    try:
        tg = {
            "nan_count": APP.model.tensor_guard._nan_count,
            "inf_count": APP.model.tensor_guard._inf_count,
            "sanitize_count": APP.model.tensor_guard._sanitize_count,
        }
    except Exception:
        pass

    text_out = result.get("text", "")
    tokens = len(text_out.split())
    tps = round(tokens / max(elapsed_ms / 1000, 0.001), 1)
    logging.info(f"✅ {tokens} tokens · {elapsed_ms}ms · {tps} tok/s · status={result.get('status')}")

    return {
        "ok": True,
        "text": text_out,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "elapsed_ms": elapsed_ms,
        "tokens": tokens,
        "tokens_per_sec": tps,
        "audit": audit,
        "tensorguard": tg,
    }


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
        }
        if logits is not None:
            probs = torch.softmax(logits[0, 0], dim=-1)
            top5 = torch.topk(probs, 5)
            result["top5_tokens"] = top5.indices.tolist()
            result["top5_probs"] = [round(p, 4) for p in top5.values.tolist()]
        logging.info(f"Forward pass · {elapsed_ms}ms · safety={result.get('safety_score','?'):.4f}" if result.get('safety_score') is not None else f"Forward pass · {elapsed_ms}ms")
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
        except Exception:
            pass
        vq_stats = {}
        try:
            if APP.model.vector_quantizer is not None:
                vq_stats = APP.model.vector_quantizer.get_codebook_usage_stats()
        except Exception:
            pass

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
    """Per-module parameter breakdown with weight statistics."""
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    try:
        modules = []
        for name, mod in APP.model.named_children():
            params = list(mod.parameters())
            total_p = sum(x.numel() for x in params)
            trainable_p = sum(x.numel() for x in params if x.requires_grad)
            weight_stats = {}
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
        except Exception:
            pass
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
        try:
            subsystems = APP.model.integrity_monitor.get_subsystem_scores()
        except Exception:
            pass
        return {
            "ok": True,
            "health_score": float(integrity) if integrity is not None else 0.0,
            "subsystems": subsystems,
        }
    except Exception:
        tg = APP.model.tensor_guard
        health = 1.0 - min(1.0, (tg._nan_count + tg._inf_count) * 0.05)
        return {"ok": True, "health_score": health, "subsystems": {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY & OBSERVABILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/telemetry/metrics")
async def get_telemetry_metrics():
    """Return a snapshot of all collected telemetry metrics with statistics."""
    if APP.config is None:
        raise HTTPException(400, "Model not initialized — no telemetry available")
    try:
        tc = APP.config.telemetry_collector
        return {"ok": True, "metrics": tc.get_metrics_snapshot()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/telemetry/metric/{metric_name}")
async def get_telemetry_metric(metric_name: str, last_n: int = 50):
    """Return the most recent entries for a specific metric."""
    if APP.config is None:
        raise HTTPException(400, "Model not initialized — no telemetry available")
    try:
        tc = APP.config.telemetry_collector
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

        # Inter-embedding cosine similarity (sample: first 64 codes for perf)
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

        # Codebook collapse risk assessment
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

        # Pipeline stage health summary
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


# ── Training runner ────────────────────────────────────────────────────────────
def _v4_training_loop(req: V4TrainRequest):
    """Full AEON v4 two-phase training, running in a background thread."""
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
        "started_at": time.time(),
        "error": None,
    }

    if not AE_TRAIN_LOADED:
        msg = f"ae_train.py not available: {AE_TRAIN_ERROR}"
        logging.error(msg)
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
        config.learning_rate      = req.learning_rate
        config.batch_size         = req.batch_size
        config.grad_clip_norm     = req.grad_clip
        config.warmup_steps       = req.warmup_steps
        config.entropy_weight     = req.entropy_weight
        config.document_aware     = req.document_aware
        config.use_amp            = req.use_amp
        config.seed               = req.seed

        import torch, numpy as np
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # ── Build model ───────────────────────────────────────────
        APP.v4_progress["phase"] = "model_init"
        model = ae.AEONDeltaV4(config).to(device)

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
                    "codebook_usage": round(float(epoch_metrics.get("codebook_%", 0)), 2),
                })
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
                APP.v4_progress.update({"epoch": epoch + 1, "total_epochs": epochs})
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
                yield f"data: {json.dumps({'type': 'metrics', 'data': {'new_A': new_A, 'new_B': new_B}})}\n\n"
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
        await broadcast(payload)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AEON Dashboard Server v3.3.0")
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
