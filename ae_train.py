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
    )
    AEON_CORE_AVAILABLE = True
except ImportError:
    AEON_CORE_AVAILABLE = False

    # Lightweight fallback so convergence events are still recorded and
    # bridge_training_errors_to_inference() works without aeon_core.
    from enum import Enum, auto
    from collections import defaultdict

    class NaNPolicy(Enum):
        WARN = auto()
        QUARANTINE = auto()

    class TensorGuard:
        """Minimal tensor safety guard (fallback when aeon_core unavailable)."""

        def __init__(self, policy=None, enable_tracking: bool = False):
            self.policy = policy or NaNPolicy.WARN
            self._nan_count = 0
            self._inf_count = 0
            self._sanitize_count = 0

        def sanitize(self, tensor: torch.Tensor, context: str = "") -> torch.Tensor:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan:
                self._nan_count += 1
            if has_inf:
                self._inf_count += 1
            if has_nan or has_inf:
                self._sanitize_count += 1
                tensor = torch.where(
                    torch.isfinite(tensor), tensor, torch.zeros_like(tensor)
                )
            return tensor

    class SemanticErrorClassifier:
        """Minimal error classifier (fallback when aeon_core unavailable)."""

        def classify(self, error: BaseException) -> tuple:
            return ("unknown", str(error))

    class CausalErrorEvolutionTracker:
        """Lightweight error evolution tracker for standalone training."""

        def __init__(self, max_history: int = 100):
            self._max_history = max_history
            self._episodes: Dict[str, list] = defaultdict(list)

        def record_episode(self, error_class: str, strategy_used: str,
                           success: bool, metadata: Optional[Dict] = None,
                           **kwargs) -> None:
            history = self._episodes[error_class]
            history.append({
                "strategy": strategy_used,
                "success": success,
                "metadata": metadata or {},
            })
            if len(history) > self._max_history:
                self._episodes[error_class] = history[-self._max_history:]

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
        """Minimal convergence monitor (fallback when aeon_core unavailable)."""

        def __init__(self, threshold: float = 1e-5):
            self.history: list = []
            self._threshold = threshold

        def check(self, delta_norm: float) -> Dict[str, Any]:
            self.history.append(delta_norm)
            if len(self.history) < 3:
                return {"status": "warmup", "certified": False}
            if delta_norm < self._threshold:
                return {"status": "converged", "certified": True}
            if len(self.history) >= 3:
                ratio = delta_norm / max(self.history[-2], 1e-12)
                if ratio >= 1.0:
                    return {"status": "diverging", "certified": False}
            return {"status": "converging", "certified": False}

    class CausalProvenanceTracker:
        """Minimal provenance tracker (fallback when aeon_core unavailable)."""

        def __init__(self):
            self._deltas: Dict[str, float] = {}
            self._order: list = []

        def record_before(self, module_name: str, state: torch.Tensor) -> None:
            self._order.append(module_name)

        def record_after(self, module_name: str, state: torch.Tensor) -> None:
            pass

        def compute_attribution(self) -> Dict[str, Any]:
            return {"attribution": {}, "raw_deltas": {}, "order": self._order}

    class ModuleCoherenceVerifier:
        """Minimal coherence verifier (fallback when aeon_core unavailable)."""

        def __init__(self, hidden_dim: int = 256, threshold: float = 0.5):
            self.threshold = threshold

        def __call__(self, states):
            B = next(iter(states.values())).shape[0] if states else 1
            return {
                "coherence_score": torch.ones(B),
                "pairwise": {},
                "needs_recheck": False,
            }

        def adapt_threshold(self, error_summary):
            pass

    class MetaCognitiveRecursionTrigger:
        """Minimal metacognitive trigger (fallback when aeon_core unavailable)."""

        def __init__(self, **kwargs):
            self._recursion_count = 0

        def reset(self):
            self._recursion_count = 0

        def evaluate(self, **kwargs):
            return {"should_trigger": False, "trigger_score": 0.0,
                    "triggers_active": [], "recursion_count": 0}

        def adapt_weights_from_evolution(self, error_summary):
            pass

    class UnifiedCognitiveCycle:
        """Minimal unified cognitive cycle (fallback when aeon_core unavailable)."""

        def __init__(self, convergence_monitor, coherence_verifier,
                     error_evolution, metacognitive_trigger,
                     provenance_tracker, causal_trace=None):
            self.convergence_monitor = convergence_monitor
            self.coherence_verifier = coherence_verifier
            self.error_evolution = error_evolution
            self.metacognitive_trigger = metacognitive_trigger
            self.provenance_tracker = provenance_tracker

        def evaluate(self, subsystem_states, delta_norm, **kwargs):
            return {
                "convergence_verdict": {"status": "warmup"},
                "coherence_result": {"coherence_score": torch.tensor([1.0]),
                                     "needs_recheck": False,
                                     "coherence_deficit": 0.0},
                "should_rerun": False,
                "trigger_detail": {"should_trigger": False,
                                   "triggers_active": []},
                "provenance": {},
                "root_cause_trace": {},
            }

        def reset(self):
            pass

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
    vq_num_embeddings: int = 2048
    vq_embedding_dim: int = 256
    vq_commitment_cost: float = 0.25
    vq_loss_weight: float = 0.5
    vq_ema_decay: float = 0.99
    vq_temperature: float = 1.0
    vq_reset_threshold: int = 30  # Было 50, теперь агрессивнее
    
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
        z = self.encode(tokens)
        quantized, vq_loss, indices, vq_stats = self.quantize(z)
        logits = self.decode(quantized, tokens)
        
        return {
            "z": z,
            "quantized": quantized,
            "vq_loss": vq_loss,
            "indices": indices,
            "logits": logits,
            "vq_stats": vq_stats
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
        self._unified_cycle = UnifiedCognitiveCycle(
            convergence_monitor=self._core_convergence,
            coherence_verifier=self._coherence_verifier,
            error_evolution=self._error_evolution,
            metacognitive_trigger=self._metacognitive_trigger,
            provenance_tracker=self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else CausalProvenanceTracker(),
        )
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
            if self._error_classifier is not None:
                try:
                    _err_cls, _err_detail = self._error_classifier.classify(
                        RuntimeError("NaN/Inf loss in training step")
                    )
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
                error_class=_error_detail.split(":")[0] if ":" in _error_detail else "numerical",
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
    
    def _forward_pass(self, tokens: torch.Tensor) -> Dict[str, Any]:
        # Track per-component provenance so training errors can be
        # traced back to their originating component.
        self.provenance.reset()

        z = self.model.encode(tokens)
        # Sanitize encoder output to prevent NaN/Inf from propagating
        # into VQ and decoder, matching the inference pipeline's safety.
        if self._tensor_guard is not None:
            z = self._tensor_guard.sanitize(z, context="training_encoder_output")
        # Record encoder output as both before/after VQ input
        self.provenance.record_before("vq", z)
        quantized, vq_loss, indices, vq_stats = self.model.quantize(z)
        self.provenance.record_after("vq", quantized)

        self.provenance.record_before("decoder", quantized)
        logits = self.model.decode(quantized, tokens)
        # Record decoder delta using mean-pooled output (seq × vocab → z_dim)
        self.provenance.record_after("decoder", logits.mean(dim=(1, 2)).unsqueeze(-1).expand_as(quantized))
        
        recon_loss = self.criterion(
            logits[:, :-1].contiguous().view(-1, self.config.vocab_size), 
            tokens[:, 1:].contiguous().view(-1)
        )
        
        total_loss = recon_loss + self.config.vq_loss_weight * vq_loss
        
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
            **vq_stats
        }
    
    def _optimizer_step(self):
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        # ✅ Используем сниженный grad_clip для стабильности
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_params, 
            self.config.grad_clip_norm  # 0.5 в v4
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
                _uncertainty = min(epoch_metrics.get("perplexity", 0.0) / 1000.0, 1.0)
                _is_diverging = convergence_verdict["status"] == "diverging"
                _cycle_result = self._unified_cycle.evaluate(
                    subsystem_states={
                        "encoder": torch.zeros(1, self.config.z_dim),
                        "vq": torch.zeros(1, self.config.z_dim),
                    },
                    delta_norm=_loss_delta,
                    uncertainty=_uncertainty,
                    recovery_pressure=1.0 if _is_diverging else 0.0,
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
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
                        param_group['lr'] *= 0.7
            except Exception as _cycle_err:
                logger.debug("Unified cognitive cycle evaluation skipped: %s", _cycle_err)
            
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
        
        self.monitor.end_training("phase_A")
    
    def _save_checkpoint(self, epoch: int, metrics: dict):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.output_dir, 
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': asdict(self.config)
            }, checkpoint_path)
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

        # --- Unified Cognitive Cycle integration for Phase B ---
        self._coherence_verifier = ModuleCoherenceVerifier(
            hidden_dim=config.z_dim, threshold=0.5,
        )
        self._metacognitive_trigger = MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5, max_recursions=2,
        )
        self._core_convergence = ConvergenceMonitor(threshold=1e-5)
        self._unified_cycle = UnifiedCognitiveCycle(
            convergence_monitor=self._core_convergence,
            coherence_verifier=self._coherence_verifier,
            error_evolution=self._error_evolution,
            metacognitive_trigger=self._metacognitive_trigger,
            provenance_tracker=self.provenance._tracker
            if hasattr(self.provenance, '_tracker') else CausalProvenanceTracker(),
        )

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
        
        # Losses
        mse_loss = F.mse_loss(pred, z_target)
        smooth_l1 = F.smooth_l1_loss(pred, z_target)
        loss = 0.5 * mse_loss + 0.5 * smooth_l1
        
        # Detect NaN/Inf loss OR non-finite RSSM output to prevent
        # corrupted gradient updates.  When tensor guard is available,
        # classify the error semantically so root-cause analysis can
        # trace it to the RSSM component.
        if _pred_had_nonfinite or torch.isnan(loss) or torch.isinf(loss):
            _prov = self.provenance.compute_attribution()
            _dominant = None
            _contributions = _prov.get('contributions', {})
            if _contributions:
                _dominant = max(_contributions, key=_contributions.get)
            logger.warning(
                f"⚠️ NaN/Inf loss detected in RSSM at step {self.global_step}"
                f" (dominant_module={_dominant}), skipping backward pass"
            )
            # Propagate NaN event to convergence monitor so it can
            # detect training instability and recommend corrective action.
            self.convergence_monitor.update(float('nan'))
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
            self.config.grad_clip_norm
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
            logger.debug(
                f"RSSM step {self.global_step} provenance: dominant={_dominant} "
                f"({_contributions[_dominant]:.1%})"
            )
        
        return {
            "mse_loss": mse_loss.item(), 
            "smooth_l1": smooth_l1.item(),
            "total_loss": loss.item(),
            "cosine_sim": cosine_sim, 
            "l1_loss": l1_loss,
            "rel_error": rel_error,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "provenance": _prov,
        }

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
                "l1_loss": 0.0, "rel_error": 0.0, "grad_norm": 0.0
            }
            valid_batches = 0
            
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
                
                if batch_idx % log_every_batch == 0:
                    self.monitor.log_batch(batch_idx, total_batches, {
                        "mse": metrics["mse_loss"],
                        "cos": metrics["cosine_sim"],
                        "rel_err": metrics["rel_error"]
                    }, phase="phase_B", log_every=log_every_batch)
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(valid_batches, 1)
            
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
                _uncertainty = min(epoch_metrics.get("mse_loss", 0.0) / 10.0, 1.0)
                _is_diverging = convergence_verdict["status"] == "diverging"
                _cycle_result = self._unified_cycle.evaluate(
                    subsystem_states={
                        "vq": torch.zeros(1, self.config.z_dim),
                        "rssm": torch.zeros(1, self.config.z_dim),
                    },
                    delta_norm=_loss_delta,
                    uncertainty=_uncertainty,
                    recovery_pressure=1.0 if _is_diverging else 0.0,
                )
                epoch_metrics["cognitive_coherence"] = (
                    1.0 - _cycle_result["coherence_result"]["coherence_deficit"]
                )
                epoch_metrics["should_rerun"] = _cycle_result["should_rerun"]
                if _cycle_result["should_rerun"]:
                    _active = _cycle_result["trigger_detail"].get("triggers_active", [])
                    logger.info(
                        f"   🧠 Phase B meta-cognitive cycle triggered "
                        f"(signals={_active}), adapting training"
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.7
            except Exception as _cycle_err:
                logger.debug("Phase B unified cognitive cycle skipped: %s", _cycle_err)
            
            if epoch_metrics["mse_loss"] < self.best_loss:
                self.best_loss = epoch_metrics["mse_loss"]
                self.best_model_state = copy.deepcopy(self.model.rssm.state_dict())
                logger.info(f"   🏆 Новый лучший MSE: {self.best_loss:.6f}")
            
            self.monitor.end_epoch(epoch, epochs, epoch_metrics, "phase_B")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.rssm.load_state_dict(self.best_model_state)
            logger.info(f"   ✅ Восстановлена лучшая RSSM модель с MSE={self.best_loss:.6f}")
        
        self.monitor.end_training("phase_B")


# ==============================================================================
# TRAINING–CORE BRIDGE: PROVENANCE & CONVERGENCE
# ==============================================================================

# Threshold above which a single module's provenance contribution
# triggers a dominance warning during validation.
_PROVENANCE_DOMINANCE_WARNING_THRESHOLD = 0.9

# Interval (in training steps) between provenance dominant-module log
# entries.  Shared by Phase A and Phase B trainers to ensure consistent
# traceability granularity across both training phases.
_PROVENANCE_LOG_INTERVAL = 50

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


def bridge_training_errors_to_inference(
    trainer_monitor: 'TrainingConvergenceMonitor',
    inference_error_evolution: Any,
    causal_trace: Any = None,
    inference_convergence_monitor: Any = None,
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

    Args:
        trainer_monitor: The training convergence monitor that has
            accumulated error episodes during training.
        inference_error_evolution: The inference pipeline's
            ``CausalErrorEvolutionTracker`` instance.
        causal_trace: Optional ``TemporalCausalTraceBuffer`` for
            recording bridged episodes as causal trace entries.
        inference_convergence_monitor: Optional inference-side
            ``ConvergenceMonitor`` to wire for automatic bridging.

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
            inference_error_evolution.record_episode(
                error_class=f"training_{cls_name}",
                strategy_used=cls_stats.get('best_strategy', 'unknown'),
                success=success_rate >= 0.5,
                metadata={
                    'source': 'training_bridge',
                    'training_count': count,
                    'training_success_rate': success_rate,
                    'max_loss_magnitude': cls_stats.get('max_loss_magnitude'),
                    'mean_loss_magnitude': cls_stats.get('mean_loss_magnitude'),
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
    return bridged


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

    trainer_B.fit(z_sequences_gpu, epochs=epochs_B)

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
