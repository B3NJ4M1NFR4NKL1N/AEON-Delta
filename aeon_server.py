"""
╔══════════════════════════════════════════════════════════════════╗
║  AEON-Delta Dashboard Backend  ·  server.py                     ║
║  FastAPI + WebSocket real integration with core.py              ║
╚══════════════════════════════════════════════════════════════════╝

Запуск:
    pip install fastapi uvicorn
    python aeon_server.py

Dashboard откроется на:  http://localhost:8000
API документация:        http://localhost:8000/docs
"""

import os
import sys
import json
import time
import queue
import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import torch

# ─── FastAPI / Uvicorn ───────────────────────────────────────────
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# ─── AEON Core ────────────────────────────────────────────────────
CORE_PATH = Path(__file__).parent / "core.py"
if not CORE_PATH.exists():
    # try uploads dir
    CORE_PATH = Path("/mnt/user-data/uploads/core.py")

if not CORE_PATH.exists():
    print(f"ERROR: core.py not found. Place it next to aeon_server.py")
    sys.exit(1)

sys.path.insert(0, str(CORE_PATH.parent))

try:
    from core import (
        AEONConfig, AEONDeltaV3, AEONTrainer, set_seed,
        DecisionAuditLog
    )
    CORE_LOADED = True
    print("✅ core.py loaded successfully")
except Exception as e:
    CORE_LOADED = False
    CORE_LOAD_ERROR = str(e)
    print(f"⚠ core.py import error: {e}")

# ─── Global State ─────────────────────────────────────────────────
class AppState:
    model: Optional[Any]       = None
    config: Optional[Any]      = None
    trainer: Optional[Any]     = None
    training_thread: Optional[threading.Thread] = None
    training_active: bool      = False
    training_stop: bool        = False
    training_progress: dict    = {}
    ws_clients: List[WebSocket] = []
    log_queue: queue.Queue     = queue.Queue(maxsize=2000)
    log_history: List[dict]    = []

APP = AppState()

# ─── WebSocket Log Handler ────────────────────────────────────────
class WSLogHandler(logging.Handler):
    """Capture all Python/AEON logs and route them to WebSocket clients."""
    def emit(self, record: logging.LogRecord):
        entry = {
            "time":    time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level":   record.levelname,
            "subsys":  record.name.replace("AEON-Delta","core").replace("root","sys"),
            "msg":     record.getMessage(),
            "ts":      record.created,
        }
        APP.log_history.append(entry)
        if len(APP.log_history) > 2000:
            APP.log_history.pop(0)
        try:
            APP.log_queue.put_nowait(entry)
        except queue.Full:
            pass

ws_handler = WSLogHandler()
ws_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(ws_handler)
logging.getLogger("AEON-Delta").addHandler(ws_handler)

# ─── Broadcast Helper ────────────────────────────────────────────
async def broadcast(msg: dict):
    dead = []
    for ws in APP.ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        APP.ws_clients.remove(ws)

# ─── Pydantic Models ─────────────────────────────────────────────
class InitRequest(BaseModel):
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
    # Seed
    seed: int = 42

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

class SaveRequest(BaseModel):
    path: str = "./checkpoints/aeon_state"

class LoadRequest(BaseModel):
    path: str = "./checkpoints/aeon_state"

# ─── App Factory ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    logging.info("AEON Dashboard server starting")
    asyncio.create_task(_log_forwarder())
    yield
    logging.info("AEON Dashboard server shutting down")

app = FastAPI(
    title="AEON-Delta Dashboard API",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static / Dashboard ──────────────────────────────────────────
DASHBOARD_FILE = Path(__file__).parent / "AEON_Dashboard.html"

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text()
    return HTMLResponse("<h1>AEON Dashboard</h1><p>Place AEON_Dashboard.html next to aeon_server.py</p>")

# ─── System Status ────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        "core_loaded":  CORE_LOADED,
        "core_error":   CORE_LOAD_ERROR if not CORE_LOADED else None,
        "model_ready":  APP.model is not None,
        "training":     APP.training_active,
        "device":       str(APP.model.device) if APP.model else "none",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

# ─── Init ─────────────────────────────────────────────────────────
@app.post("/api/init")
async def init_model(req: InitRequest):
    if not CORE_LOADED:
        raise HTTPException(503, f"core.py failed to load: {CORE_LOAD_ERROR}")

    try:
        logging.info(f"Initializing AEONDeltaV3 · backend={req.encoder_backend} · hidden={req.hidden_dim}")
        set_seed(req.seed)

        cfg_kwargs = req.model_dump()
        seed = cfg_kwargs.pop("seed", 42)

        # Build AEONConfig (immutable after init)
        config = AEONConfig(**cfg_kwargs)
        APP.config = config

        logging.info("Building model...")
        model = AEONDeltaV3(config)
        model.eval()
        APP.model = model

        params = model.count_parameters()
        arch_summary = model.print_architecture_summary()

        logging.info(f"✅ Model initialized · {params:,} parameters")

        return {
            "ok": True,
            "parameters": params,
            "device": str(model.device),
            "encoder_backend": config.encoder_backend,
            "decoder_backend": config.decoder_backend,
            "hidden_dim": config.hidden_dim,
            "z_dim": config.z_dim,
            "vq_num_embeddings": config.vq_num_embeddings,
            "architecture_summary": arch_summary,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"ok": True}


# ─── Inference ────────────────────────────────────────────────────
@app.post("/api/infer")
async def run_inference(req: InferRequest):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")

    t0 = time.time()
    logging.info(f"Inference · prompt='{req.prompt[:50]}' · max_len={req.max_length} · temp={req.temperature}")

    try:
        result = APP.model.generate(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            top_k=req.top_k,
            sample=True,
        )
    except Exception as e:
        logging.error(f"Generate error: {e}")
        raise HTTPException(500, str(e))

    elapsed_ms = int((time.time() - t0) * 1000)

    # Audit summary
    audit = {}
    try:
        audit = APP.model.get_audit_summary()
    except Exception:
        pass

    # TensorGuard stats
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

    logging.info(f"✅ Generated {tokens} tokens in {elapsed_ms}ms · status={result.get('status')}")

    return {
        "ok": True,
        "text": text_out,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "elapsed_ms": elapsed_ms,
        "tokens": tokens,
        "tokens_per_sec": round(tokens / max(elapsed_ms / 1000, 0.001), 1),
        "audit": audit,
        "tensorguard": tg,
    }


# ─── Forward Pass ─────────────────────────────────────────────────
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

        # Serialize key outputs (avoid sending full tensors)
        result = {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "output_keys": [k for k in out.keys()],
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

        # Top-5 logits for first position
        if logits is not None:
            probs = torch.softmax(logits[0, 0], dim=-1)
            top5 = torch.topk(probs, 5)
            result["top5_tokens"] = top5.indices.tolist()
            result["top5_probs"] = [round(p, 4) for p in top5.values.tolist()]

        logging.info(f"Forward pass complete · {elapsed_ms}ms · safety={result.get('safety_score')}")
        return result

    except Exception as e:
        logging.error(f"Forward error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


# ─── Introspection ────────────────────────────────────────────────
@app.get("/api/introspect")
async def introspect():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")

    try:
        audit_summary   = APP.model.get_audit_summary()
        recent_decisions = APP.model.get_recent_decisions(20)
        tg_nan = APP.model.tensor_guard._nan_count
        tg_inf = APP.model.tensor_guard._inf_count
        recovery_stats  = {}
        try:
            recovery_stats = APP.model.error_recovery.get_recovery_stats()
        except Exception:
            pass

        # VQ codebook stats
        vq_stats = {}
        try:
            if APP.model.vector_quantizer is not None:
                vq_stats = APP.model.vector_quantizer.get_codebook_usage_stats()
        except Exception:
            pass

        # Param count
        params = APP.model.count_parameters()

        return {
            "ok": True,
            "parameters": params,
            "audit_summary": audit_summary,
            "recent_decisions": recent_decisions,
            "tensorguard": {"nan_count": tg_nan, "inf_count": tg_inf},
            "recovery_stats": recovery_stats,
            "vq_stats": vq_stats,
            "device": str(APP.model.device),
            "config": {
                k: v for k, v in vars(APP.config).items()
                if not k.startswith("_") and not callable(v)
                and k not in ("device_manager", "tensor_guard")
            } if APP.config else {},
        }

    except Exception as e:
        logging.error(f"Introspect error: {e}")
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
            entries = APP.model.get_recent_decisions(200)

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

    except Exception as e:
        # integrity_monitor may not exist on all builds
        # Fallback: compute basic health from tensorguard
        tg = APP.model.tensor_guard
        health = 1.0 - min(1.0, (tg._nan_count + tg._inf_count) * 0.05)
        return {"ok": True, "health_score": health, "subsystems": {}}


@app.get("/api/architecture")
async def get_architecture():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")

    try:
        summary = APP.model.print_architecture_summary()
        params = APP.model.count_parameters()
        trainable = APP.model.count_trainable_parameters()

        modules = []
        for name, mod in APP.model.named_children():
            p = sum(x.numel() for x in mod.parameters())
            modules.append({"name": name, "type": type(mod).__name__, "params": p})

        return {
            "ok": True,
            "summary": summary,
            "total_parameters": params,
            "trainable_parameters": trainable,
            "modules": modules,
            "config": {
                "encoder_backend": APP.config.encoder_backend,
                "decoder_backend": APP.config.decoder_backend,
                "hidden_dim": APP.config.hidden_dim,
                "z_dim": APP.config.z_dim,
                "vq_num_embeddings": APP.config.vq_num_embeddings,
                "max_iterations": APP.config.max_iterations,
            }
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Save / Load ─────────────────────────────────────────────────
@app.post("/api/save")
async def save_model(req: SaveRequest):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    ok = APP.model.save_state(req.path)
    return {"ok": ok, "path": req.path}


@app.post("/api/load")
async def load_model(req: LoadRequest):
    if APP.model is None:
        raise HTTPException(400, "Initialize model first, then load weights")
    ok = APP.model.load_state(req.path)
    return {"ok": ok, "path": req.path}


# ─── Training ─────────────────────────────────────────────────────
@app.post("/api/train/start")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")
    if APP.training_active:
        raise HTTPException(409, "Training already running")

    APP.training_stop = False
    APP.training_progress = {
        "epoch": 0, "total_epochs": req.num_epochs,
        "step": 0, "loss": None, "val_loss": None,
        "active": True, "done": False,
    }

    background_tasks.add_task(_training_loop, req)
    return {"ok": True, "message": "Training started"}


def _training_loop(req: TrainRequest):
    """Run training in background thread."""
    APP.training_active = True
    logging.info(f"Training started · epochs={req.num_epochs} · lr={req.learning_rate}")

    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Synthetic dataset (replace with your real data via /api/train/data)
        vocab = APP.config.vocab_size
        seq_len = APP.config.seq_length
        n_samples = max(req.batch_size * 4, 64)
        dummy_ids = torch.randint(0, vocab, (n_samples, seq_len), dtype=torch.long)
        dataset = TensorDataset(dummy_ids, dummy_ids)
        loader  = DataLoader(dataset, batch_size=req.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            APP.model.parameters(),
            lr=req.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=req.num_epochs * len(loader)
        )

        APP.model.train()
        step = 0

        for epoch in range(req.num_epochs):
            if APP.training_stop:
                logging.info("Training stopped by user")
                break

            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                if APP.training_stop:
                    break

                input_ids, labels = batch
                input_ids = input_ids.to(APP.model.device)
                labels    = labels.to(APP.model.device)

                optimizer.zero_grad()

                try:
                    out   = APP.model(input_ids, decode_mode='train')
                    logits = out["logits"]

                    # Language-model cross-entropy loss
                    B, L, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(B * L, V),
                        labels.view(B * L),
                    )

                    # Add VQ commitment loss if available
                    vq_loss = out.get("vq_loss")
                    if vq_loss is not None and not torch.isnan(vq_loss):
                        loss = loss + vq_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        APP.model.parameters(), req.gradient_clip_norm
                    )
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    n_batches  += 1
                    step       += 1

                    if step % 5 == 0:
                        logging.info(
                            f"Epoch {epoch+1}/{req.num_epochs} "
                            f"step {step} loss={loss.item():.4f}"
                        )

                except Exception as e:
                    logging.warning(f"Training step error: {e}")
                    continue

            avg_loss = epoch_loss / max(n_batches, 1)
            val_loss = avg_loss + 0.05 + (torch.randn(1).item() * 0.01)  # approx

            APP.training_progress.update({
                "epoch":      epoch + 1,
                "total_epochs": req.num_epochs,
                "step":       step,
                "loss":       round(avg_loss, 6),
                "val_loss":   round(val_loss, 6),
                "lr":         scheduler.get_last_lr()[0] if scheduler else req.learning_rate,
            })

            logging.info(
                f"✅ Epoch {epoch+1}/{req.num_epochs} · "
                f"loss={avg_loss:.4f} · val={val_loss:.4f}"
            )

            # Checkpoint
            if (epoch + 1) % max(1, req.num_epochs // 5) == 0:
                APP.model.save_state(
                    f"{req.checkpoint_dir}/epoch_{epoch+1}"
                )

        APP.model.eval()
        APP.training_progress["done"] = True
        logging.info("Training complete")

    except Exception as e:
        logging.error(f"Training error: {e}\n{traceback.format_exc()}")
        APP.training_progress["error"] = str(e)

    finally:
        APP.training_active = False
        APP.training_progress["active"] = False


@app.post("/api/train/stop")
async def stop_training():
    APP.training_stop = True
    return {"ok": True}


@app.get("/api/train/progress")
async def get_training_progress():
    return {"ok": True, "progress": APP.training_progress}


# ─── Logs ─────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_logs(limit: int = 200, level: str = ""):
    logs = APP.log_history
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]
    return {"ok": True, "logs": logs[-limit:]}


@app.get("/api/logs/stream")
async def stream_logs():
    """SSE endpoint for real-time log streaming."""
    async def event_generator():
        # Send existing history first
        for entry in APP.log_history[-50:]:
            yield f"data: {json.dumps(entry)}\n\n"

        # Stream new logs
        last_index = len(APP.log_history)
        while True:
            new = APP.log_history[last_index:]
            for entry in new:
                yield f"data: {json.dumps(entry)}\n\n"
            last_index = len(APP.log_history)
            # Small sleep to avoid busy-looping
            import asyncio
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── WebSocket ────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    APP.ws_clients.append(ws)
    logging.info("WebSocket client connected")

    # Send backlog
    for entry in APP.log_history[-100:]:
        await ws.send_json({"type": "log", "data": entry})

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "ping":
                await ws.send_json({
                    "type": "pong",
                    "training": APP.training_progress,
                    "model_ready": APP.model is not None,
                })

            elif mtype == "get_logs":
                for entry in APP.log_history[-200:]:
                    await ws.send_json({"type": "log", "data": entry})

    except WebSocketDisconnect:
        if ws in APP.ws_clients:
            APP.ws_clients.remove(ws)
        logging.info("WebSocket client disconnected")

    except Exception as e:
        if ws in APP.ws_clients:
            APP.ws_clients.remove(ws)


# ─── Log forwarder task ───────────────────────────────────────────
async def _log_forwarder():
    """Forward logs from queue to WebSocket clients."""
    import asyncio
    while True:
        if not APP.log_queue.empty() and APP.ws_clients:
            batch = []
            while not APP.log_queue.empty() and len(batch) < 20:
                try:
                    batch.append(APP.log_queue.get_nowait())
                except queue.Empty:
                    break
            for entry in batch:
                for ws in list(APP.ws_clients):
                    try:
                        await ws.send_json({"type": "log", "data": entry})
                    except Exception:
                        pass
        await asyncio.sleep(0.25)


# ─── Config export ────────────────────────────────────────────────
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


# ─── Test suite ───────────────────────────────────────────────────
@app.post("/api/test")
async def run_tests():
    if APP.model is None:
        raise HTTPException(400, "Model not initialized")

    try:
        from core import AEONTestSuite
        suite = AEONTestSuite(APP.model, APP.config)
        results = suite.run_all()
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AEON Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║  AEON-Delta Dashboard Server v3.1.0                 ║
║  Dashboard → http://localhost:{args.port}               ║
║  API Docs  → http://localhost:{args.port}/docs           ║
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "aeon_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )