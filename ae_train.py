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
from tqdm import tqdm
import logging
import os
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset, Dataset
import argparse
import copy
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


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
        
        if batch_idx % log_every == 0 or batch_idx == total_batches - 1:
            metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            progress = (batch_idx + 1) / total_batches * 100
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
        ) / self.temperature
        
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
        entropy_loss = self._compute_entropy_loss(indices)
        
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
        max_entropy = math.log(self.num_embeddings)
        
        # Loss = 1 - normalized_entropy (хотим максимизировать энтропию)
        entropy_loss = 1.0 - (entropy / max_entropy) if max_entropy > 0 else torch.tensor(0.0, device=indices.device)
        
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
                max_entropy = math.log(self.num_embeddings)
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
        
        # Residual connection к последнему z
        z_last = z_context[:, -1, :]
        z_pred = z_pred + self.residual_weight * z_last
        
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
        
        self.vq = VectorQuantizerHybridV4(
            config.vq_num_embeddings, 
            config.vq_embedding_dim,
            commitment_cost=config.vq_commitment_cost,
            decay=config.vq_ema_decay,
            temperature=config.vq_temperature,
            reset_threshold=config.vq_reset_threshold,
            entropy_weight=config.entropy_weight,
            code_reset_noise_scale=config.code_reset_noise_scale
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
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
        self.device = device
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
        
        self.use_amp = config.use_amp and AMP_AVAILABLE
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
        
    def train_step(self, tokens: torch.Tensor) -> Dict[str, float]:
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
        
        # Detect NaN/Inf loss to prevent corrupted gradient updates
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(
                f"⚠️ NaN/Inf loss detected at step {self.global_step}, skipping backward pass"
            )
            return outputs
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return outputs
    
    def _forward_pass(self, tokens: torch.Tensor) -> Dict[str, float]:
        z = self.model.encode(tokens)
        quantized, vq_loss, indices, vq_stats = self.model.quantize(z)
        logits = self.model.decode(quantized, tokens)
        
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
        
        return grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

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
        total_steps = epochs * total_batches // self.config.gradient_accumulation_steps
        
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
            
            for batch_idx, (batch,) in enumerate(loader):
                outputs = self.train_step(batch)
                step_loss = outputs['total_loss'].item()
                if not (math.isnan(step_loss) or math.isinf(step_loss)):
                    accumulated_loss += step_loss
                    num_accumulated += 1
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    grad_norm = self._optimizer_step()
                    self.scheduler.step()
                    
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
                    epoch_metrics["grad_norm"] += grad_norm if grad_norm else 0
                
                if batch_idx % log_every_batch == 0:
                    self.monitor.log_batch(batch_idx, total_batches, {
                        "loss": outputs['recon_loss'] + self.config.vq_loss_weight * outputs['vq_loss'],
                        "recon": outputs['recon_loss'],
                        "ppl": outputs['perplexity'],
                        "acc": outputs['accuracy'],
                        "cb%": outputs.get('codebook_usage_%', 0)
                    }, log_every=log_every_batch)
            
            if num_accumulated > 0:
                avg_loss = accumulated_loss / max(num_accumulated, 1)
                epoch_metrics["total"] += avg_loss
                grad_norm = self._optimizer_step()
                self.scheduler.step()
                epoch_metrics["grad_norm"] += grad_norm if grad_norm else 0
            
            num_steps = max(
                (total_batches + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps,
                1
            )
            for key in epoch_metrics:
                epoch_metrics[key] /= num_steps
            
            epoch_metrics["lr"] = self.scheduler.get_lr()
            
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
        
        # Предсказание
        pred = self.model.rssm(z_context)
        
        # Losses
        mse_loss = F.mse_loss(pred, z_target)
        smooth_l1 = F.smooth_l1_loss(pred, z_target)
        loss = 0.5 * mse_loss + 0.5 * smooth_l1
        
        # Detect NaN/Inf loss to prevent corrupted gradient updates
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"⚠️ NaN/Inf loss detected in RSSM at step {self.global_step}, skipping backward pass"
            )
            return {
                "mse_loss": float('nan'), "smooth_l1": float('nan'),
                "total_loss": float('nan'), "cosine_sim": 0.0,
                "l1_loss": float('nan'), "rel_error": float('nan'),
                "grad_norm": 0.0
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
            rel_error = (torch.norm(pred - z_target, dim=1) / (torch.norm(z_target, dim=1) + 1e-8)).mean().item()
        
        return {
            "mse_loss": mse_loss.item(), 
            "smooth_l1": smooth_l1.item(),
            "total_loss": loss.item(),
            "cosine_sim": cosine_sim, 
            "l1_loss": l1_loss,
            "rel_error": rel_error,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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
            
            for batch_idx, (ctx_batch, tgt_batch) in enumerate(loader):
                ctx_batch = ctx_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                metrics = self.train_step(ctx_batch, tgt_batch)
                
                for key in epoch_metrics:
                    if key in metrics and not (math.isnan(metrics[key]) or math.isinf(metrics[key])):
                        epoch_metrics[key] += metrics[key]
                
                if batch_idx % log_every_batch == 0:
                    self.monitor.log_batch(batch_idx, total_batches, {
                        "mse": metrics["mse_loss"],
                        "cos": metrics["cosine_sim"],
                        "rel_err": metrics["rel_error"]
                    }, phase="phase_B", log_every=log_every_batch)
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(total_batches, 1)
            
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
    test_batch = torch.randint(0, config.vocab_size, (2, config.seq_length), device=device)
    
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
            if name == "vq" and hasattr(component, 'embedding'):
                if component.embedding.weight.grad is not None:
                    logger.info(f"   ✅ {name}: градиенты через embedding")
                    continue
            issues.append(f"{name}: нет градиентов")
            logger.error(f"   ❌ {name}: нет градиентов")
    
    model.zero_grad()
    
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
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        
        tokens = tokenize_batch(texts, tokenizer, config.seq_length, device)
        documents = None
    
    logger.info(f"   Токенов для Phase A: {tokens.shape}")
    
    # Сохраняем токены
    try:
        torch.save(tokens.cpu(), os.path.join(output_dir, "tokens.pt"))
    except OSError as e:
        logger.error(f"❌ Failed to save tokens: {e}")

    # Создание модели v4
    model = AEONDeltaV4(config).to(device)
    
    # Загрузка checkpoint
    if resume_from and os.path.exists(resume_from):
        logger.info(f"📂 Загрузка checkpoint: {resume_from}")
        try:
            # Try safe loading first
            try:
                checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
            except Exception:
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

    # Save best loss before releasing Phase A resources
    best_loss_A = trainer_A.best_loss

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
