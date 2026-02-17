"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AEON-Unified Server  ·  aeon_unified_server.py  v5.1.0 — Production        ║
║  FastAPI + WebSocket + SSE · Full V4 Training Pipeline + Dashboard          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIXES v5.1.0:                                                               ║
║  · Уникальный алиас + sys.modules инъекция ДО exec_module                  ║
║  · Устранён конфликт кеша sys.modules["ae_train"]                          ║
║  · Defensive _get() с диагностикой доступных атрибутов                     ║
║  · WebSocket backlog: try/except на каждое send_json                        ║
║  · Двойной лог устранён: propagate=False + очистка хендлеров ae_train      ║
║  · WSLogHandler восстанавливается после импорта                             ║
║  · /api/v4/diagnostics — подробный отчёт о состоянии импорта               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Запуск:
    pip install fastapi uvicorn psutil python-multipart
    python aeon_unified_server.py [--host 0.0.0.0] [--port 8000]
"""

import os, sys, json, time, queue, logging, threading, traceback
import math, shutil, tempfile, importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import asdict

import torch

try:
    from fastapi import (
        FastAPI, WebSocket, WebSocketDisconnect,
        HTTPException, BackgroundTasks, UploadFile, File
    )
    from fastapi.responses import (
        HTMLResponse, JSONResponse, StreamingResponse, FileResponse
    )
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("ERROR: pip install fastapi uvicorn pydantic python-multipart")
    sys.exit(1)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE  (defined first — before any logging setup)
# ══════════════════════════════════════════════════════════════════════════════
class _APP:
    log_queue:        queue.Queue     = queue.Queue(maxsize=10_000)
    log_history:      List[dict]      = []
    ws_clients:       List[WebSocket] = []
    model:            Any             = None
    model_config:     Any             = None
    training_active:  bool            = False
    training_stop:    bool            = False
    training_phase:   str             = "idle"
    training_progress: dict           = {}
    training_error:   str             = ""
    phase_a_metrics:  List[dict]      = []
    phase_b_metrics:  List[dict]      = []
    best_loss_a:      float           = float('inf')
    best_loss_b:      float           = float('inf')
    uploaded_files:   Dict[str,str]   = {}
    upload_dir:       str             = ""
    output_dir:       str             = "./aeon_training_output"
    last_checkpoint:  str             = ""

APP = _APP()
APP.upload_dir     = tempfile.mkdtemp(prefix="aeon_uploads_")
APP.phase_a_metrics = []
APP.phase_b_metrics = []
APP.uploaded_files  = {}


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING  (root-level, captures everything including ae_train noise)
# ══════════════════════════════════════════════════════════════════════════════
class _WSHandler(logging.Handler):
    def emit(self, record):
        try:
            entry = {
                "time":   time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level":  record.levelname,
                "subsys": record.name[:24],
                "msg":    self.format(record) if record.exc_info else record.getMessage(),
                "ts":     record.created,
            }
            APP.log_history.append(entry)
            if len(APP.log_history) > 10_000:
                APP.log_history = APP.log_history[-10_000:]
            APP.log_queue.put_nowait(entry)
        except Exception:
            pass


_ws_handler = _WSHandler()
_ws_handler.setLevel(logging.DEBUG)
_ws_handler.setFormatter(logging.Formatter('%(name)s | %(message)s'))

# Single console handler on root
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(
    logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
)

# Configure root: only our two handlers, no default handlers
_root = logging.getLogger()
_root.handlers.clear()
_root.setLevel(logging.DEBUG)
_root.addHandler(_ws_handler)
_root.addHandler(_console)

# Our server logger
logger = logging.getLogger("aeon_unified")
logger.propagate = True


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC IMPORT OF ae_train.py  (cache-safe, robust)
# ══════════════════════════════════════════════════════════════════════════════
def _find(name: str) -> Optional[Path]:
    for p in [
        Path(__file__).parent / name,
        Path(os.getcwd()) / name,
        Path("/mnt/user-data/uploads") / name,
    ]:
        if p.exists():
            return p
    return None


AE_TRAIN_PATH = _find("ae_train.py")

AE_TRAIN_LOADED      = False
AE_TRAIN_ERROR       = ""
AE_TRAIN_ATTRS_FOUND: List[str] = []

# Symbols to resolve from ae_train
AEONConfigV4                 = None
AEONDeltaV4                  = None
SafeThoughtAETrainerV4       = None
ContextualRSSMTrainer        = None
TrainingMonitor              = None
validate_training_components = None
tokenize_batch_fn            = None
load_documents_fn            = None

TRANSFORMERS_AVAILABLE = False
AMP_AVAILABLE          = False

# Unique internal name — never conflicts with any real module
_AE_MOD_NAME = "_aeon_train_v4_priv_"

if AE_TRAIN_PATH:
    try:
        # Flush any stale cache entries
        for _stale in [_AE_MOD_NAME, "ae_train"]:
            sys.modules.pop(_stale, None)

        spec     = importlib.util.spec_from_file_location(_AE_MOD_NAME, str(AE_TRAIN_PATH))
        ae_mod   = importlib.util.module_from_spec(spec)

        # Register BEFORE exec so internal imports resolve correctly
        sys.modules[_AE_MOD_NAME] = ae_mod

        # Temporarily silence "AEON-Training-v4" logger that ae_train configures
        # at module level — prevents double output during import
        _at_log = logging.getLogger("AEON-Training-v4")
        _at_log_prev_level = _at_log.level
        _at_log.setLevel(logging.CRITICAL)

        spec.loader.exec_module(ae_mod)

        # Restore + clean up: ae_train adds its own StreamHandler,
        # remove it to prevent double console output
        _at_log = logging.getLogger("AEON-Training-v4")
        for _h in list(_at_log.handlers):
            _at_log.removeHandler(_h)
        _at_log.addHandler(_ws_handler)   # WS only — console via root propagation
        _at_log.setLevel(logging.INFO)
        _at_log.propagate = True          # let root handle console

        # Collect exported names for diagnostics
        AE_TRAIN_ATTRS_FOUND = [k for k in dir(ae_mod) if not k.startswith('_')]

        def _g(primary, *fallbacks):
            """Get attribute with fallback names."""
            for name in (primary,) + fallbacks:
                v = getattr(ae_mod, name, None)
                if v is not None:
                    return v
            return None

        AEONConfigV4                 = _g('AEONConfigV4',  'AEONConfig')
        AEONDeltaV4                  = _g('AEONDeltaV4',   'AEONDelta')
        SafeThoughtAETrainerV4       = _g('SafeThoughtAETrainerV4', 'SafeThoughtAETrainer')
        ContextualRSSMTrainer        = _g('ContextualRSSMTrainer',  'RSSMTrainer')
        TrainingMonitor              = _g('TrainingMonitor')
        validate_training_components = _g('validate_training_components')
        tokenize_batch_fn            = _g('tokenize_batch')
        load_documents_fn            = _g('load_documents_from_json')
        TRANSFORMERS_AVAILABLE       = bool(getattr(ae_mod, 'TRANSFORMERS_AVAILABLE', False))
        AMP_AVAILABLE                = bool(getattr(ae_mod, 'AMP_AVAILABLE', False))

        if AEONConfigV4 is None:
            raise ImportError(
                f"AEONConfigV4 / AEONConfig not found in ae_train.py.\n"
                f"   File: {AE_TRAIN_PATH}\n"
                f"   Found: {AE_TRAIN_ATTRS_FOUND[:25]}"
            )

        AE_TRAIN_LOADED = True
        logger.info(f"✅ ae_train.py loaded: {AE_TRAIN_PATH}")
        logger.info(
            f"   Resolved: AEONConfigV4={AEONConfigV4.__name__}  "
            f"AEONDeltaV4={getattr(AEONDeltaV4,'__name__','N/A')}  "
            f"TRANSFORMERS={TRANSFORMERS_AVAILABLE}"
        )

    except Exception as _ie:
        AE_TRAIN_ERROR = str(_ie)
        sys.modules.pop(_AE_MOD_NAME, None)
        print(f"\n❌  ae_train.py import failed: {_ie}", file=sys.stderr)
        if AE_TRAIN_ATTRS_FOUND:
            print(f"   Available attrs: {AE_TRAIN_ATTRS_FOUND[:20]}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
else:
    AE_TRAIN_ERROR = (
        "ae_train.py not found. "
        "Place it in the same directory as aeon_unified_server.py"
    )
    print(f"❌  {AE_TRAIN_ERROR}", file=sys.stderr)


# ── Tokenizer ──────────────────────────────────────────────────────────────────
_tokenizer = None
if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info(f"✅ Tokenizer: bert-base-uncased  vocab={_tokenizer.vocab_size}")
    except Exception as _te:
        logger.warning(f"Tokenizer failed ({_te}) — char-level fallback active")


# ══════════════════════════════════════════════════════════════════════════════
#  WEB TRAINING MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if TrainingMonitor is not None:
    class WebTrainingMonitor(TrainingMonitor):
        def start_training(self, phase: str, total_epochs: int, total_samples: int):
            super().start_training(phase, total_epochs, total_samples)
            _p = "phase_A" if "A" in phase.upper() else "phase_B"
            APP.training_phase    = _p
            APP.training_progress = {
                "phase": _p, "epoch": 0,
                "total_epochs": total_epochs, "total_samples": total_samples,
                "metrics": {},
            }

        def end_epoch(self, epoch, total_epochs, epoch_metrics, phase="phase_A"):
            result = super().end_epoch(epoch, total_epochs, epoch_metrics, phase)
            safe = {
                k: (float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0)
                   if isinstance(v, (int, float)) else v
                for k, v in epoch_metrics.items()
                if isinstance(v, (int, float, str))
            }
            entry = {"epoch": epoch + 1, **safe}
            if phase == "phase_A":
                APP.phase_a_metrics.append(entry)
                APP.best_loss_a = min(APP.best_loss_a, float(self.best_loss))
            else:
                APP.phase_b_metrics.append(entry)
                APP.best_loss_b = min(APP.best_loss_b, float(self.best_loss))
            APP.training_progress = {
                "phase": phase, "epoch": epoch + 1,
                "total_epochs": total_epochs, "metrics": safe,
                "best_loss": float(self.best_loss),
            }
            return result

        def end_training(self, phase: str):
            super().end_training(phase)
            if "B" in phase.upper():
                APP.training_phase = "done"
else:
    WebTrainingMonitor = None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _txt_to_jsonl(src: str, dst: str) -> int:
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    segs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    if len(segs) < 3:
        segs = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    with open(dst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    logger.info(f"txt→jsonl: {Path(src).name} → {len(segs)} segments")
    return len(segs)


def _ensure_jsonl(fp: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ext = Path(fp).suffix.lower()
    if ext == ".jsonl":
        return fp
    dst = os.path.join(out_dir, "input_data.jsonl")
    if ext == ".json":
        try:
            with open(fp, "r", encoding="utf-8") as f:
                first = f.read(2).strip()
            if first.startswith("["):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with open(dst, "w", encoding="utf-8") as f:
                    for item in data:
                        t = item if isinstance(item, str) else item.get("text", str(item))
                        if len(t) > 10:
                            f.write(json.dumps({"text": t}) + "\n")
                logger.info(f"json array→jsonl: {len(data)} records")
                return dst
            return fp  # already line-delimited
        except json.JSONDecodeError:
            _txt_to_jsonl(fp, dst)
            return dst
    n = _txt_to_jsonl(fp, dst)
    if n == 0:
        raise ValueError(f"No text segments in: {fp}")
    return dst


def _char_tok(text: str, maxlen: int, vocab_size: int) -> torch.Tensor:
    ids = [min(ord(c), vocab_size - 1) for c in text[:maxlen]]
    ids += [0] * max(0, maxlen - len(ids))
    return torch.tensor(ids[:maxlen], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def _run_pipeline(
    file_path: str, cfg_ov: dict,
    epochs_a: int, epochs_b: int,
    doc_aware: bool, resume: Optional[str]
) -> None:
    import numpy as np

    out = APP.output_dir
    os.makedirs(out, exist_ok=True)

    try:
        APP.training_phase    = "starting"
        APP.training_progress = {"phase": "starting", "epoch": 0, "metrics": {}}
        APP.phase_a_metrics   = []
        APP.phase_b_metrics   = []
        APP.best_loss_a       = float('inf')
        APP.best_loss_b       = float('inf')

        if not AE_TRAIN_LOADED:
            raise RuntimeError(f"ae_train.py not loaded: {AE_TRAIN_ERROR}")

        logger.info("=" * 70)
        logger.info("  AEON UNIFIED TRAINING PIPELINE v5.1")
        logger.info("=" * 70)

        # Device
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Device: {dev}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Config
        config = AEONConfigV4()
        config.document_aware = doc_aware
        known = set(config.__dataclass_fields__.keys()) \
            if hasattr(config, '__dataclass_fields__') else set()
        for k, v in cfg_ov.items():
            if not known or k in known:
                try:
                    setattr(config, k, type(getattr(config, k))(v))
                except Exception:
                    pass
        if hasattr(config, 'vq_embedding_dim'):
            config.vq_embedding_dim = config.z_dim
        if _tokenizer:
            config.vocab_size = _tokenizer.vocab_size

        logger.info(f"📋 Config: z={config.z_dim} h={config.hidden_dim} "
                    f"seq={config.seq_length} vq={config.vq_num_embeddings} "
                    f"lr={config.learning_rate:.2e} bs={config.batch_size}")

        seed = getattr(config, 'seed', 42)
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Data
        APP.training_phase = "loading_data"
        logger.info(f"\n📥 Preparing: {Path(file_path).name}")
        jsonl = _ensure_jsonl(file_path, out)

        if doc_aware and load_documents_fn and _tokenizer:
            documents = load_documents_fn(
                jsonl, _tokenizer, config.seq_length,
                min_chunks=config.min_doc_chunks, logger=logger
            )
            all_tok = [c for doc in documents for c in doc]
            if not all_tok:
                raise ValueError("No token chunks — check min_doc_chunks and file")
            tokens = torch.stack(all_tok).to(dev)
            logger.info(f"   docs={len(documents)} chunks={len(all_tok)}")
        else:
            texts = []
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        t = d.get("text", "") if isinstance(d, dict) else str(d)
                        if len(t.strip()) > 10:
                            texts.append(t)
                    except Exception:
                        pass
            if not texts:
                raise ValueError("No valid texts — check file format")
            logger.info(f"   segments={len(texts)}")

            if tokenize_batch_fn and _tokenizer:
                tokens = tokenize_batch_fn(texts, _tokenizer, config.seq_length, dev)
            elif _tokenizer:
                all_ids = []
                for txt in texts:
                    enc = _tokenizer(txt, max_length=config.seq_length,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")
                    all_ids.append(enc["input_ids"].squeeze(0))
                tokens = torch.stack(all_ids).to(dev)
            else:
                logger.warning("⚠ Char-level tokenizer (no transformers)")
                tokens = torch.stack([
                    _char_tok(t, config.seq_length, config.vocab_size)
                    for t in texts
                ]).to(dev)
            documents = None

        if tokens.numel() == 0:
            raise ValueError("Token tensor empty — no usable training data")
        logger.info(f"   ✅ tokens={tokens.shape}")

        # Model
        APP.training_phase = "building_model"
        logger.info(f"\n🔨 Building AEONDeltaV4...")
        model = AEONDeltaV4(config).to(dev)
        APP.model        = model
        APP.model_config = config
        logger.info(f"   ✅ params={sum(p.numel() for p in model.parameters()):,}")

        # Resume
        if resume and os.path.exists(resume):
            logger.info(f"📂 Checkpoint: {resume}")
            try:
                try:
                    ck = torch.load(resume, map_location=dev, weights_only=True)
                except (RuntimeError, TypeError):
                    ck = torch.load(resume, map_location=dev, weights_only=False)
                if isinstance(ck, dict) and "model_state_dict" in ck:
                    model.load_state_dict(ck["model_state_dict"])
                    logger.info("   ✅ Checkpoint loaded")
            except Exception as ce:
                logger.error(f"   ❌ Checkpoint error: {ce}")

        # Validate
        APP.training_phase = "validating"
        if validate_training_components:
            ok = validate_training_components(model, config, logger)
            if not ok:
                raise RuntimeError("Component validation failed — see logs")

        if APP.training_stop:
            return

        # Monitor
        ckpt_dir = os.path.join(out, "checkpoints")
        monitor = (WebTrainingMonitor if WebTrainingMonitor else TrainingMonitor)(
            logger, save_dir=ckpt_dir
        )

        # ── Phase A ──────────────────────────────────────────────────────
        logger.info("\n" + "▶" * 50)
        logger.info("     PHASE A — AutoEncoder + VQ-VAE")
        logger.info("▶" * 50)
        APP.training_phase = "phase_A"

        trainer_a = SafeThoughtAETrainerV4(model, config, monitor, ckpt_dir)
        trainer_a.fit(tokens, epochs=epochs_a)
        best_a = trainer_a.best_loss
        APP.best_loss_a = float(best_a) if math.isfinite(best_a) else 0.0
        del trainer_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if APP.training_stop:
            logger.info("⏹ Stopped after Phase A")
            return

        # ── Build z_sequences ────────────────────────────────────────────
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("\n🔧 Building z_sequences...")
        APP.training_phase = "building_z"
        model.eval()

        with torch.no_grad():
            if doc_aware and documents:
                z_sequences, skipped = [], 0
                for di, doc_chunks in enumerate(documents):
                    if len(doc_chunks) < config.context_window + 1:
                        skipped += 1
                        continue
                    batch = torch.stack(doc_chunks).to(dev)
                    q, _, _, _ = model.quantize(model.encode(batch))
                    z_sequences.append(q.cpu())
                    if di > 0 and di % 100 == 0:
                        logger.info(f"   encoded {di}/{len(documents)} docs")
                logger.info(f"   ✅ z_sequences={len(z_sequences)} (skipped={skipped})")
            else:
                z_list = []
                for (b,) in DataLoader(TensorDataset(tokens.cpu()), batch_size=256):
                    q, _, _, _ = model.quantize(model.encode(b.to(dev)))
                    z_list.append(q.cpu())
                z_all = torch.cat(z_list)
                z_sequences = [z_all]
                logger.info(f"   ✅ z_all={z_all.shape}")

        if not z_sequences:
            raise ValueError("No z_sequences — increase context_window or corpus size")

        try:
            torch.save(z_sequences, os.path.join(out, "z_sequences.pt"))
        except OSError as se:
            logger.warning(f"z_sequences save skipped: {se}")

        if APP.training_stop:
            logger.info("⏹ Stopped before Phase B")
            return

        # ── Phase B ──────────────────────────────────────────────────────
        logger.info("\n" + "▶" * 50)
        logger.info("     PHASE B — Contextual RSSM")
        logger.info("▶" * 50)
        APP.training_phase = "phase_B"

        trainer_b = ContextualRSSMTrainer(model, config, monitor)
        trainer_b.fit([s.to(dev) for s in z_sequences], epochs=epochs_b)
        APP.best_loss_b = float(trainer_b.best_loss) if math.isfinite(trainer_b.best_loss) else 0.0

        # ── Save ─────────────────────────────────────────────────────────
        final = os.path.join(out, "aeon_v4_final.pt")
        for p in model.parameters():
            p.requires_grad = True
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config":           asdict(config),
                "phase_a_metrics":  APP.phase_a_metrics,
                "phase_b_metrics":  APP.phase_b_metrics,
                "training_info": {
                    "epochs_A": epochs_a, "epochs_B": epochs_b,
                    "best_loss_A": APP.best_loss_a,
                    "best_loss_B": APP.best_loss_b,
                    "doc_aware": doc_aware,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "5.1.0",
                },
            }, final)
            APP.last_checkpoint = final
            logger.info(f"💾 Saved: {final}")
        except OSError as se:
            logger.error(f"❌ Save failed: {se}")

        logger.info("\n" + "🎉" * 28)
        logger.info("     TRAINING COMPLETE")
        logger.info("🎉" * 28)
        logger.info(f"   Phase A best loss : {APP.best_loss_a:.6f}")
        logger.info(f"   Phase B best MSE  : {APP.best_loss_b:.6f}")
        try:
            logger.info(f"   Codebook usage    : {model.vq.get_codebook_usage():.2f}%")
        except Exception:
            pass
        logger.info(f"   Output            : {out}")

        APP.training_phase = "done"
        APP.training_progress = {
            "phase": "done",
            "best_loss_a": APP.best_loss_a,
            "best_loss_b": APP.best_loss_b,
        }

    except Exception as exc:
        APP.training_error = str(exc)
        APP.training_phase = "error"
        logger.error(f"❌ Pipeline error: {exc}")
        logger.debug(traceback.format_exc())
    finally:
        APP.training_active = False
        APP.training_stop   = False
        logger.info(f"Thread done. Phase: {APP.training_phase}")


# ══════════════════════════════════════════════════════════════════════════════
#  BROADCAST
# ══════════════════════════════════════════════════════════════════════════════
async def _bcast(msg: dict):
    dead = []
    for ws in list(APP.ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            APP.ws_clients.remove(ws)
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
class TrainStartRequest(BaseModel):
    file_name:         str   = ""
    epochs_a:          int   = Field(30, ge=1, le=500)
    epochs_b:          int   = Field(10, ge=1, le=200)
    document_aware:    bool  = True
    resume_from:       str   = ""
    z_dim:             int   = 256
    hidden_dim:        int   = 256
    vq_num_embeddings: int   = 2048
    batch_size:        int   = 16
    learning_rate:     float = 3e-5
    context_window:    int   = 3
    entropy_weight:    float = 0.1
    grad_clip_norm:    float = 0.5
    dropout_rate:      float = 0.1
    seq_length:        int   = 64
    warmup_steps:      int   = 1000
    label_smoothing:   float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  APP + LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio
    logger.info("AEON Unified Server v5.1.0 ready")
    asyncio.create_task(_log_fwd())
    asyncio.create_task(_heartbeat())
    yield
    shutil.rmtree(APP.upload_dir, ignore_errors=True)


app = FastAPI(
    title="AEON Unified Training API", version="5.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_DASH = Path(__file__).parent / "AEON_Unified_Dashboard.html"

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if _DASH.exists():
        return _DASH.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h1>AEON</h1><p>Place AEON_Unified_Dashboard.html here.</p>"
        "<p><a href='/docs'>API Docs</a> | "
        "<a href='/api/v4/diagnostics'>Diagnostics</a></p>"
    )


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    mp = None
    if APP.model:
        try:
            mp = sum(p.numel() for p in APP.model.parameters())
        except Exception:
            pass
    return {
        "ae_train_loaded": AE_TRAIN_LOADED,
        "ae_train_error":  AE_TRAIN_ERROR if not AE_TRAIN_LOADED else None,
        "ae_train_path":   str(AE_TRAIN_PATH) if AE_TRAIN_PATH else None,
        "tokenizer_ready": _tokenizer is not None,
        "model_ready":     APP.model is not None,
        "model_params":    mp,
        "training_active": APP.training_active,
        "training_phase":  APP.training_phase,
        "training_error":  APP.training_error,
        "torch_version":   torch.__version__,
        "cuda_available":  torch.cuda.is_available(),
        "uploaded_files":  list(APP.uploaded_files.keys()),
        "last_checkpoint": APP.last_checkpoint,
        "output_dir":      APP.output_dir,
    }


@app.get("/api/status/system")
async def sys_status():
    info: dict = {"ok": True, "torch_version": torch.__version__,
                  "cuda_available": torch.cuda.is_available()}
    if PSUTIL_AVAILABLE:
        try:
            vm = psutil.virtual_memory()
            info.update({
                "ram_total_gb":   round(vm.total / 1e9, 2),
                "ram_used_gb":    round(vm.used  / 1e9, 2),
                "ram_percent":    round(vm.percent, 1),
                "cpu_percent":    round(psutil.cpu_percent(interval=0.1), 1),
                "cpu_count":      psutil.cpu_count(),
                "process_ram_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
            })
        except Exception as e:
            info["psutil_error"] = str(e)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                p = torch.cuda.get_device_properties(i)
                a = torch.cuda.memory_allocated(i)
                info.setdefault("gpu_devices", []).append({
                    "index": i, "name": p.name,
                    "total_mb":     round(p.total_memory / 1e6, 0),
                    "allocated_mb": round(a               / 1e6, 1),
                    "reserved_mb":  round(torch.cuda.memory_reserved(i) / 1e6, 1),
                    "free_mb":      round((p.total_memory - a) / 1e6, 1),
                })
            except Exception:
                pass
    return info


@app.get("/api/v4/diagnostics")
async def diagnostics():
    return {
        "ae_train_path":        str(AE_TRAIN_PATH) if AE_TRAIN_PATH else None,
        "ae_train_loaded":      AE_TRAIN_LOADED,
        "ae_train_error":       AE_TRAIN_ERROR,
        "ae_train_attrs_found": AE_TRAIN_ATTRS_FOUND[:30],
        "internal_module":      _AE_MOD_NAME,
        "python":               sys.version,
        "torch":                torch.__version__,
        "cuda":                 torch.cuda.is_available(),
        "transformers":         TRANSFORMERS_AVAILABLE,
        "tokenizer":            _tokenizer is not None,
        "resolved": {
            "AEONConfigV4":          AEONConfigV4.__name__          if AEONConfigV4          else None,
            "AEONDeltaV4":           AEONDeltaV4.__name__           if AEONDeltaV4           else None,
            "SafeThoughtAETrainerV4": SafeThoughtAETrainerV4.__name__ if SafeThoughtAETrainerV4 else None,
            "ContextualRSSMTrainer":  ContextualRSSMTrainer.__name__  if ContextualRSSMTrainer  else None,
            "TrainingMonitor":        TrainingMonitor.__name__        if TrainingMonitor        else None,
        },
    }


# ── Files ─────────────────────────────────────────────────────────────────────
@app.post("/api/files/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".txt", ".json", ".jsonl"}:
        raise HTTPException(400, f"Unsupported: {ext}. Use .txt .json .jsonl")
    dest    = os.path.join(APP.upload_dir, file.filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    APP.uploaded_files[file.filename] = dest
    sz = round(len(content) / 1024, 1)
    logger.info(f"📁 Uploaded: {file.filename} ({sz} KB)")
    return {"ok": True, "filename": file.filename, "size_kb": sz}


@app.get("/api/files")
async def list_files():
    out = []
    for name, path in APP.uploaded_files.items():
        try:
            sz = round(os.path.getsize(path) / 1024, 1)
        except OSError:
            sz = 0.0
        out.append({"name": name, "size_kb": sz})
    return {"ok": True, "files": out}


@app.delete("/api/files/{filename}")
async def del_file(filename: str):
    if filename not in APP.uploaded_files:
        raise HTTPException(404, f"'{filename}' not found")
    try:
        os.remove(APP.uploaded_files.pop(filename))
    except OSError:
        pass
    return {"ok": True, "deleted": filename}


# ── Training ──────────────────────────────────────────────────────────────────
@app.post("/api/v4/train/start")
async def train_start(req: TrainStartRequest, bt: BackgroundTasks):
    if APP.training_active:
        raise HTTPException(409, "Already training — stop first")
    if not AE_TRAIN_LOADED:
        raise HTTPException(503, f"ae_train.py not loaded: {AE_TRAIN_ERROR}")

    fp = ""
    if req.file_name:
        if req.file_name not in APP.uploaded_files:
            raise HTTPException(404, f"'{req.file_name}' not uploaded")
        fp = APP.uploaded_files[req.file_name]
    elif len(APP.uploaded_files) == 1:
        fp = list(APP.uploaded_files.values())[0]
    elif not APP.uploaded_files:
        raise HTTPException(400, "No files — upload first via /api/files/upload")
    else:
        raise HTTPException(400, "Multiple files — specify file_name")

    cfg = {
        "z_dim": req.z_dim, "hidden_dim": req.hidden_dim,
        "vq_num_embeddings": req.vq_num_embeddings,
        "vq_embedding_dim": req.z_dim,
        "batch_size": req.batch_size, "learning_rate": req.learning_rate,
        "context_window": req.context_window, "entropy_weight": req.entropy_weight,
        "grad_clip_norm": req.grad_clip_norm, "dropout_rate": req.dropout_rate,
        "seq_length": req.seq_length, "warmup_steps": req.warmup_steps,
        "label_smoothing": req.label_smoothing,
    }
    APP.training_active = True
    APP.training_stop   = False
    APP.training_error  = ""
    APP.training_phase  = "starting"

    bt.add_task(_run_pipeline, fp, cfg, req.epochs_a, req.epochs_b,
                req.document_aware, req.resume_from.strip() or None)

    fname = Path(fp).name
    logger.info(f"🚀 Training | {fname} | A={req.epochs_a} B={req.epochs_b}")
    return {"ok": True, "file": fname, "epochs_a": req.epochs_a, "epochs_b": req.epochs_b}


@app.post("/api/v4/train/stop")
async def train_stop():
    if not APP.training_active:
        return {"ok": True, "message": "Nothing running"}
    APP.training_stop = True
    logger.info("⏹ Stop signal sent")
    return {"ok": True, "message": "Stop signal sent"}


@app.get("/api/v4/train/progress")
async def train_progress():
    return {
        "ok": True,
        "training_active": APP.training_active,
        "training_phase":  APP.training_phase,
        "training_error":  APP.training_error,
        "progress":        APP.training_progress,
        "best_loss_a": float(APP.best_loss_a) if math.isfinite(APP.best_loss_a) else None,
        "best_loss_b": float(APP.best_loss_b) if math.isfinite(APP.best_loss_b) else None,
    }


@app.get("/api/v4/train/metrics")
async def train_metrics():
    return {
        "ok": True,
        "phase_a": APP.phase_a_metrics,
        "phase_b": APP.phase_b_metrics,
        "best_loss_a": float(APP.best_loss_a) if math.isfinite(APP.best_loss_a) else None,
        "best_loss_b": float(APP.best_loss_b) if math.isfinite(APP.best_loss_b) else None,
        "training_phase": APP.training_phase,
    }


@app.get("/api/v4/model/info")
async def model_info():
    if APP.model is None:
        return {"ok": False, "message": "No model — start training first"}
    try:
        params = sum(p.numel() for p in APP.model.parameters())
        dev    = str(next(APP.model.parameters()).device)
        cfg    = asdict(APP.model_config) if APP.model_config else {}
        vq_pct = None
        try:
            vq_pct = float(APP.model.vq.get_codebook_usage())
        except Exception:
            pass
        return {"ok": True, "parameters": params, "device": dev,
                "config": cfg, "vq_usage_pct": vq_pct}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/v4/checkpoint/download")
async def checkpoint_dl():
    if not APP.last_checkpoint or not os.path.exists(APP.last_checkpoint):
        raise HTTPException(404, "No checkpoint yet")
    return FileResponse(
        APP.last_checkpoint, media_type="application/octet-stream",
        filename=Path(APP.last_checkpoint).name,
    )


# ── Logs ──────────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_logs(limit: int = 500, level: str = "", subsys: str = ""):
    logs = APP.log_history
    if level:
        logs = [l for l in logs if l.get("level") == level.upper()]
    if subsys:
        logs = [l for l in logs if subsys.lower() in l.get("subsys","").lower()]
    return {"ok": True, "logs": logs[-limit:], "total": len(APP.log_history)}


@app.delete("/api/logs")
async def clear_logs():
    APP.log_history.clear()
    return {"ok": True}


@app.get("/api/logs/stream")
async def log_stream():
    import asyncio
    async def _gen():
        for e in APP.log_history[-200:]:
            yield f"data: {json.dumps(e)}\n\n"
        last = len(APP.log_history)
        while True:
            for e in APP.log_history[last:]:
                yield f"data: {json.dumps(e)}\n\n"
            last = len(APP.log_history)
            await asyncio.sleep(0.15)
    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    APP.ws_clients.append(ws)
    logger.info(f"WS+ clients={len(APP.ws_clients)}")

    # Send backlog with individual guards
    for entry in APP.log_history[-200:]:
        try:
            await ws.send_json({"type": "log", "data": entry})
        except Exception:
            try:
                APP.ws_clients.remove(ws)
            except ValueError:
                pass
            return  # client gone — exit immediately

    # Initial state snapshot
    try:
        await ws.send_json({
            "type": "status",
            "training_active": APP.training_active,
            "training_phase":  APP.training_phase,
            "progress":        APP.training_progress,
        })
    except Exception:
        try:
            APP.ws_clients.remove(ws)
        except ValueError:
            pass
        return

    # Receive loop
    try:
        while True:
            msg = await ws.receive_json()
            t   = msg.get("type", "")
            if t == "ping":
                await ws.send_json({
                    "type": "pong", "ts": time.time(),
                    "training_active": APP.training_active,
                    "training_phase":  APP.training_phase,
                    "progress":        APP.training_progress,
                })
            elif t == "get_metrics":
                await ws.send_json({
                    "type": "metrics",
                    "phase_a": APP.phase_a_metrics,
                    "phase_b": APP.phase_b_metrics,
                })
            elif t == "get_logs":
                lim = int(msg.get("limit", 200))
                for entry in APP.log_history[-lim:]:
                    await ws.send_json({"type": "log", "data": entry})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS recv error: {e}")
    finally:
        try:
            APP.ws_clients.remove(ws)
        except ValueError:
            pass
        logger.info(f"WS- clients={len(APP.ws_clients)}")


# ── Background ────────────────────────────────────────────────────────────────
async def _log_fwd():
    import asyncio
    while True:
        if APP.ws_clients and not APP.log_queue.empty():
            batch = []
            try:
                while not APP.log_queue.empty() and len(batch) < 50:
                    batch.append(APP.log_queue.get_nowait())
            except queue.Empty:
                pass
            for e in batch:
                await _bcast({"type": "log", "data": e})
        await asyncio.sleep(0.1)


async def _heartbeat():
    import asyncio
    while True:
        if APP.ws_clients:
            p: dict = {
                "type": "heartbeat", "ts": time.time(),
                "training_active": APP.training_active,
                "training_phase":  APP.training_phase,
            }
            if APP.training_active:
                p["progress"]    = APP.training_progress
                p["phase_a_len"] = len(APP.phase_a_metrics)
                p["phase_b_len"] = len(APP.phase_b_metrics)
                p["best_loss_a"] = float(APP.best_loss_a) if math.isfinite(APP.best_loss_a) else None
                p["best_loss_b"] = float(APP.best_loss_b) if math.isfinite(APP.best_loss_b) else None
            await _bcast(p)
        await asyncio.sleep(2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AEON Unified Server v5.1.0")
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--port",       type=int, default=8000)
    p.add_argument("--reload",     action="store_true")
    p.add_argument("--log-level",  default="info",
                   choices=["debug","info","warning","error"])
    p.add_argument("--output-dir", default="./aeon_training_output")
    args = p.parse_args()

    APP.output_dir = args.output_dir
    os.makedirs(APP.output_dir, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  AEON Unified Training Server v5.1.0  (Production)              ║
║  Dashboard  →  http://localhost:{args.port}                         ║
║  API Docs   →  http://localhost:{args.port}/docs                    ║
║  WebSocket  →  ws://localhost:{args.port}/ws                        ║
║  Diag       →  http://localhost:{args.port}/api/v4/diagnostics      ║
╚══════════════════════════════════════════════════════════════════╝
ae_train   : {"✅ " + str(AE_TRAIN_PATH) if AE_TRAIN_LOADED else "❌ " + AE_TRAIN_ERROR}
tokenizer  : {"✅ bert-base-uncased" if _tokenizer else "⚠ char-level fallback"}
CUDA       : {"✅ " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"}
output dir : {args.output_dir}
""")

    uvicorn.run(
        "aeon_unified_server:app",
        host=args.host, port=args.port,
        reload=args.reload, log_level=args.log_level,
    )
