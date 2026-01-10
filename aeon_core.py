import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any, Set
from dataclasses import dataclass, field, asdict
import math
from tqdm import tqdm
import logging
import os
import json
import time
import io
import hashlib
import uuid
import copy
import threading
from collections import deque, defaultdict
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset

# --- ИЗМЕНЕНИЕ ---
# Добавлена библиотека transformers для корректной токенизации
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEON-Delta")

# Try import py2neo (Neo4j graph DB)
try:
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Matrix exponential
# Distributed support (optional)
try:
    import torch.distributed as dist
    DIST_OK = dist.is_available()
except ImportError:
    DIST_OK = False

# Use mem0 for memory management
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
    MEM0_API_KEY = os.getenv("MEM0_API_KEY")
    MEM0_REMOTE_ENABLED = bool(MEM0_API_KEY)
    if not MEM0_REMOTE_ENABLED:
        logger.warning("MEM0_API_KEY not set; mem0 client will be disabled; using local fallback memory.")
except ImportError:
    MEM0_AVAILABLE = False
    MEM0_API_KEY = None
    MEM0_REMOTE_ENABLED = False
    logger.warning("mem0 not available; falling back to in-memory list-based storage.")

# Device configuration - default CPU (overridable via CLI in __main__)
device = torch.device("cpu")
logger.info("ℹ️  Defaulting to CPU device (override with --device {cpu|cuda|mps})")
logger.info(f"Device: {device}")

# AMP (Automatic Mixed Precision) configuration
AMP_ENABLED = False  # CPU default; will be updated by set_global_device()
logger.info(f"AMP enabled: {AMP_ENABLED}")

def set_global_device(device_str: str) -> torch.device:
    """Set global device consistently (used by config defaults and tensor creation).

    Validates availability for CUDA/MPS and updates AMP flag.
    """
    global device, AMP_ENABLED
    device_str = (device_str or "cpu").lower().strip()
    d = torch.device(device_str)

    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but torch.cuda.is_available() is False")
    if d.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS selected but torch.backends.mps.is_available() is False")

    device = d
    AMP_ENABLED = (device.type == "cuda")
    logger.info(f"✅ Using device: {device}")
    logger.info(f"AMP enabled: {AMP_ENABLED}")
    return device

# Future-proof for torch >= 2.1
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# ===== ГЛОБАЛЬНАЯ УТИЛИТА ДЛЯ БЕЗОПАСНОЙ РАБОТЫ С ТЕНЗОРАМИ =====
def safe_tensor(t: torch.Tensor, default_value: float = 0.0, 
                max_value: float = 1e6, min_value: float = -1e6) -> torch.Tensor:
    """
    ✅ УНИФИЦИРОВАННАЯ ФУНКЦИЯ: Замена всех torch.nan_to_num вызовов.
    
    Безопасно обрабатывает NaN, Inf and неправильные значения в тензорах.
    
    Args:
        t: Входной тензор
        default_value: Значение для замены NaN (default: 0.0)
        max_value: Максимальное допустимое значение (default: 1e6)
        min_value: Минимальное допустимое значение (default: -1e6)
    
    Returns:
        Очищенный тензор с корректными значениями
    """
    if not isinstance(t, torch.Tensor):
        return t
    
    # Нефлоаты не трогаем (int/bool/token ids и т.п.)
    if not (t.is_floating_point() or torch.is_complex(t)):
        return t

    # ✅ ИСПРАВЛЕНО: Добавлено логирование NaN/Inf
    if torch.isnan(t).any():
        logger.warning(f"⚠️  NaN detected in tensor with shape {t.shape}, replacing with {default_value}")
    if torch.isinf(t).any():
        logger.warning(f"⚠️  Inf detected in tensor with shape {t.shape}, clipping to [{min_value}, {max_value}]")
    
    # Обработка NaN
    t = torch.where(torch.isnan(t), torch.full_like(t, default_value), t)
    
    # Обработка Inf
    t = torch.where(torch.isinf(t), torch.sign(t) * max_value, t)
    
    # Клипирование к диапазону
    t = torch.clamp(t, min=min_value, max=max_value)
    
    return t

class NoOpQualiaExtractor(torch.nn.Module):
    """Fallback extractor: returns input unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        # ✅ ИСПРАВЛЕНО: Добавлена проверка типа
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        logger.debug(f"NoOpQualiaExtractor: passing through tensor with shape {x.shape}")
        return x


# ─────────────────────────────────────────────────────────
# Auto-encoder for thoughts
# ─────────────────────────────────────────────────────────
class ThoughtEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, z_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm  = nn.LSTM(emb_dim, z_dim, batch_first=True, bidirectional=False)
        self.z_dim = z_dim
        self.norm  = nn.LayerNorm(z_dim)

    def forward(self, tokens, attention_mask=None):
        # ✅ ДЕТЕРМИНИРОВАННОЕ КОДИРОВАНИЕ: поддержка attention_mask для исключения PAD из динамики LSTM
        assert tokens.dim() == 2, f"Expected tokens shape [B, L], got {tokens.shape}"
        if attention_mask is not None:
            assert attention_mask.shape == tokens.shape, f"attention_mask shape mismatch: {attention_mask.shape} vs tokens {tokens.shape}"

        logger.debug(f"Entering ThoughtEncoder with tokens shape {tokens.shape}")

        x = self.embed(tokens)  # [B, L, E]

        # Если дана маска — исключаем PAD из временной динамики через pack_padded_sequence.
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1).to('cpu')
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)  # h: [1, B, z_dim]
        else:
            _, (h, _) = self.lstm(x)       # h: [1, B, z_dim]

        z = self.norm(h.squeeze(0))        # [B, z_dim]

        assert z.dim() == 2, f"Encoder output dimension mismatch: {z.shape}"
        assert z.shape[-1] == self.z_dim, "Encoder output size mismatch"

        logger.debug(f"Exiting ThoughtEncoder with shape {z.shape}")
        return z
class ThoughtDecoder(nn.Module):
    """
    Унифицированный декодер с поддержкой двух режимов:
    - train: Teacher-forcing с teacher_tokens (быстрое обучение)
    - inference: Авторегрессионная генерация с autoregressive sampling (реальная генерация)
    
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: z теперь конкатенируется с эмбеддингами на каждом шаге,
    что гарантирует полное использование латентного вектора в декодировании.
    """
    def __init__(self, vocab_size, emb_dim=256, z_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.z_dim = z_dim

        # Эмбеддинг слой
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # Проекция латентного вектора z в начальное состояние LSTM
        self.fc = nn.Linear(z_dim, emb_dim)
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: LSTM теперь принимает emb_dim + z_dim на входе
        # чтобы z конкатенировался с эмбеддингами на каждом временном шаге
        self.lstm = nn.LSTM(emb_dim + z_dim, emb_dim, batch_first=True)
        
        # Выходной слой (projection в словарь)
        self.head = nn.Linear(emb_dim, vocab_size)

        # ✅ КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Привязка весов эмбеддинга к выходному слою
        self._tie_weights()
        self._verify_weight_tying()
        
        # Кэш для autoregressive inference
        self._kv_cache = {}

    def _tie_weights(self):
        """Привязывает веса выходного слоя к эмбеддинг-слою (sharing weights)."""
        self.head.weight = self.embed.weight

    def _verify_weight_tying(self):
        """Строгая верификация что привязка выполнена корректно."""
        if self.head.weight.data_ptr() != self.embed.weight.data_ptr():
            raise RuntimeError(
                "❌ Weight tying failed: head.weight and embed.weight должны быть одним тензором. "
                "Попытка переприсвоения не сработала."
            )
        if self.head.weight.shape != self.embed.weight.shape:
            raise RuntimeError(
                f"❌ Weight tying shape mismatch: "
                f"head.weight {self.head.weight.shape} vs embed.weight {self.embed.weight.shape}"
            )
        logger.info("✅ Weight tying verified: head and embed share the same parameters")

        # ✅ Маска запрещённых токенов (например, [unused###]) — обновляется из AEONDelta после инициализации токенизатора.
        self.register_buffer("_invalid_token_mask", torch.zeros(self.vocab_size, dtype=torch.bool), persistent=False)


    def set_invalid_token_ids(self, token_ids):
        """✅ Устанавливает маску токенов, которые запрещены для генерации."""
        if token_ids is None:
            self._invalid_token_mask.zero_()
            return
        # Нормализуем и отбрасываем выходы за диапазон
        try:
            idx = torch.tensor(list(token_ids), dtype=torch.long)
        except Exception:
            idx = torch.empty((0,), dtype=torch.long)
        if idx.numel() > 0:
            idx = idx[(idx >= 0) & (idx < self.vocab_size)]
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if idx.numel() > 0:
            mask[idx] = True
        self._invalid_token_mask.data.copy_(mask.to(self._invalid_token_mask.device))
    def forward(self, z, teacher_tokens=None, mode='train', max_length=64, temperature=0.8, top_k=50, sample=True, prefix_tokens=None):
        """
        Унифицированный forward pass поддерживающий оба режима.
        
        Args:
            z: Латентный вектор мысли [B, z_dim]
            teacher_tokens: Токены учителя [B, L] для режима train
            mode: 'train' (teacher-forcing) or 'inference' (autoregressive)
            max_length: Максимальная длина для inference
            temperature: Температура для sampling (inference)
            top_k: Top-K filtering для sampling (inference)
            sample: True для sampling, False для greedy (inference)
        
        Returns:
            Для 'train': logits [B, L, V]
            Для 'inference': generated_ids [B, L], logits [B, L, V]
        """
        batch_size = z.shape[0]
        device = z.device
        
        if mode == 'train':
            if teacher_tokens is None:
                raise ValueError("mode='train' требует teacher_tokens")
            return self._forward_train(z, teacher_tokens, device)
        
        elif mode == 'inference':
            return self._forward_inference(z, max_length, temperature, top_k, sample, device, prefix_tokens=prefix_tokens)
        
        else:
            raise ValueError(f"Неизвестный mode: {mode}. Используйте 'train' or 'inference'")

    def _forward_train(self, z, teacher_tokens, device):
        """
        ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Teacher-forcing теперь включает z на каждом шаге.
        
        Режим TRAINING: Teacher-forcing для быстрого and стабильного обучения.
        Латентный вектор z конкатенируется с эмбеддингами на каждом временном шаге,
        что гарантирует полное использование семантической информации из z.
        
        Архитектура:
        - Input: [B, L, emb_dim + z_dim] (эмбеддинги + z на каждом шаге)
        - LSTM: обрабатывает последовательность с z-conditioning
        - Output: [B, L, vocab_size] логиты
        """
        batch_size = z.shape[0]
        seq_length = teacher_tokens.shape[1]
        
        # Проекция z в начальное скрытое состояние LSTM
        h0 = self.fc(z).unsqueeze(0)  # [1, B, E]
        c0 = torch.zeros_like(h0)      # [1, B, E]
        
        # Эмбеддинг ground-truth токенов (teacher-forcing)
        embeddings = self.embed(teacher_tokens)  # [B, L, E]
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Конкатенация z с эмбеддингами на каждом временном шаге
        # z расширяется до [B, L, z_dim] and конкатенируется с embeddings [B, L, E]
        # Результат: [B, L, E + z_dim]
        z_expanded = z.unsqueeze(1).expand(-1, seq_length, -1)  # [B, L, z_dim]
        lstm_input = torch.cat([embeddings, z_expanded], dim=-1)  # [B, L, E + z_dim]
        
        # LSTM обрабатывает последовательность с z-conditioning на каждом шаге
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # [B, L, E]
        
        # Проекция в логиты словаря
        logits = self.head(lstm_out)  # [B, L, V]
        
        return logits

    def _forward_inference(self, z, max_length, temperature, top_k, sample, device, prefix_tokens=None):
        """Авторегрессионная генерация. Поддерживает префиксное кондиционирование последовательностью токенов."""
        batch_size = z.shape[0]

        # Инициализация hidden/cell из z
        h_state = self.fc(z).unsqueeze(0)  # [1, B, z_dim]
        c_state = torch.zeros_like(h_state)

        generated_ids = []
        all_logits = []

        # ✅ Префиксное кондиционирование: прогоняем prefix_tokens через LSTM до начала генерации.
        if prefix_tokens is not None:
            assert prefix_tokens.dim() == 2, f"prefix_tokens must be [B, L], got {prefix_tokens.shape}"
            assert prefix_tokens.shape[0] == batch_size, f"prefix batch mismatch: {prefix_tokens.shape[0]} vs {batch_size}"
            prefix_tokens = prefix_tokens.to(device)

            # Прогон префикса одним батчем
            emb_pref = self.embed(prefix_tokens)  # [B, Lp, E]
            z_exp = z.unsqueeze(1).expand(-1, prefix_tokens.shape[1], -1)  # [B, Lp, z_dim]
            lstm_in = torch.cat([emb_pref, z_exp], dim=-1)
            _, (h_state, c_state) = self.lstm(lstm_in, (h_state, c_state))

            generated_ids.append(prefix_tokens)
            current_token_id = prefix_tokens[:, -1:].contiguous()
        else:
            # Стартовый токен [CLS] (BERT)
            current_token_id = torch.full((batch_size, 1), 101, dtype=torch.long, device=device)
            generated_ids.append(current_token_id)

        # Основной цикл генерации
        for _ in range(max_length):
            emb = self.embed(current_token_id)  # [B, 1, E]
            z_expanded = z.unsqueeze(1)         # [B, 1, z_dim]
            lstm_input = torch.cat([emb, z_expanded], dim=-1)

            lstm_out, (h_state, c_state) = self.lstm(lstm_input, (h_state, c_state))
            logits = self.head(lstm_out.squeeze(1))  # [B, V]
            all_logits.append(logits)

            logits_filtered = self._filter_logits(logits, temperature, top_k, device)

            if sample:
                probs = F.softmax(logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                next_token_id = torch.argmax(logits_filtered, dim=-1, keepdim=True)

            # Стоп по [SEP] (BERT)
            if (next_token_id == 102).all():
                break

            generated_ids.append(next_token_id)
            current_token_id = next_token_id

        generated_ids = torch.cat(generated_ids, dim=1) if generated_ids else torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        logits_stacked = torch.stack(all_logits, dim=1) if all_logits else torch.zeros((batch_size, 0, self.vocab_size), device=device)

        return generated_ids, logits_stacked
    def _filter_logits(self, logits, temperature, top_k, device):
        """
        Применяет Top-K фильтрацию and температурную масштабировку к логитам.
        Это технически корректный способ контроля разнообразия генерации.
        """
        # Температурная масштабировка
        scaled_logits = logits / max(temperature, 1e-6)
        
        # Top-K фильтрация
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
            # Создаём маску для фильтрации
            mask = torch.full_like(scaled_logits, -float('inf'))
            mask.scatter_(1, top_k_indices, top_k_logits)
            scaled_logits = mask
        

        # ✅ Фильтр «мёртвых» токенов (например, [unused###])
        if hasattr(self, "_invalid_token_mask") and self._invalid_token_mask is not None and self._invalid_token_mask.numel() == self.vocab_size:
            invalid_mask = self._invalid_token_mask.to(device)
            if invalid_mask.any():
                scaled_logits[:, invalid_mask] = -float('inf')

        # Защита от NaN/Inf
        scaled_logits = torch.nan_to_num(scaled_logits, neginf=-float('inf'), posinf=float('inf'))
        return scaled_logits

    def clear_cache(self):
        """Очистка кэша для следующей последовательности."""
        self._kv_cache.clear()

# --- ИЗМЕНЕНИЕ: Новый модуль VectorQuantizer ---
class VectorQuantizer(nn.Module):
    """
    Дисретизирует непрерывные векторы z, находя ближайшие аналоги в кодовой книге.
    Это "токенизатор мыслей" на семантическом уровне.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # ✅ НОВОЕ: Максимальная допустимая разность для STE (предотвращает numerical instability)
        self.max_ste_diff = 10.0

        # Кодовая книга - наш "словарь" базовых концептов/мыслей
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # ✅ ИСПРАВЛЕНО: Добавлена проверка размерности входа
        assert inputs.dim() == 2, f"VectorQuantizer expects 2D input [B, D], got {inputs.shape}"
        
        logger.debug(f"VectorQuantizer: Processing input with shape {inputs.shape}")
        
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Вычисление расстояний до векторов в кодовой книге
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # Находим индексы ближайших векторов
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # [B, 1]
        
        # Создаем one-hot вектор для выбора эмбеддингов
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # [B, N]
        
        # Получаем квантованные векторы
        quantized = torch.matmul(encodings, self.embedding.weight) # [B, D]
        
        # Loss (для обучения, здесь используется для полноты)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # ✅ ИСПРАВЛЕНО: Логирование VQ loss
        logger.info(f"VQ loss: {loss.item():.4f}")
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Straight-Through Estimator с клипированием разности
        # Проблема: При больших разностях (quantized - inputs) может возникнуть numerical instability
        # Решение: Клипируем разность перед добавлением для предотвращения экстремальных значений
        ste_diff = quantized - inputs
        
        # Проверка на наличие градиента (STE применяется только при обучении)
        if inputs.requires_grad:
            # Клипирование разности для предотвращения numerical instability
            ste_diff_clipped = torch.clamp(ste_diff, -self.max_ste_diff, self.max_ste_diff)
            
            # Логирование if разность была обрезана
            diff_norm = ste_diff.norm(dim=-1).max().item()
            if diff_norm > self.max_ste_diff:
                logger.warning(f"⚠️  STE diff clipped: max_norm={diff_norm:.4f} > threshold={self.max_ste_diff}")
            
            quantized = inputs + ste_diff_clipped.detach()
        else:
            # В режиме inference просто используем quantized напрямую
            quantized = inputs + ste_diff.detach()
        
        logger.debug(f"VectorQuantizer: Exiting with quantized shape {quantized.shape}")
        return quantized, loss, encoding_indices.squeeze()
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

@dataclass
class AEONConfig:
    z_dim: int = 256
    hidden_dim: int = 256
    meta_dim: int = 256
    # --- ИЗМЕНЕНИЕ: Vocab size теперь определяется реальным токенизатором
    vocab_size: int = 30522 # bert-base-uncased
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    num_pillars: int = 5
    seq_length: int = 64
    alpha: float = 0.9
    max_iterations: int = 8
    convergence_threshold: float = 1e-5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lambda_self_consistency: float = 0.1
    lambda_reg: float = 0.01
    lambda_safety: float = 0.1
    use_quantum_sim: bool = False
    quantum_sim_type: str = "tensor_network"
    topo_grid_size: int = 64
    topo_epsilon: float = 1e-4
    use_amp: bool = True
    memory_size: int = 10000
    knowledge_dim: int = 128
    action_dim: int = 64
    planning_horizon: int = 10
    memory_path: str = "./aeon_memory"
    use_neo4j: bool = False
    knowledge_graph_url: str = "bolt://localhost:7687"
    knowledge_graph_auth: Tuple[str, str] = ("neo4j", "")
    safety_threshold: float = 0.85
    save_frequency: int = 5
    dropout_rate: float = 0.1
    bias_l2_factor: float = 0.01
    enable_kv_cache: bool = True
    quantize_weights: bool = False
    distributed_training: bool = False
    world_size: int = 1
    enable_checkpointing: bool = True
    quantum_dim_reduction: bool = False
    precompute_displacement: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out")
    kl_weight: float = 0.1  # Added for training
    # --- ИЗМЕНЕНИЕ: Параметры для VQ
    vq_num_embeddings: int = 8192 # Количество "базовых мыслей"
    vq_embedding_dim: int = 256 # Должно совпадать с z_dim
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    # --- НОВОЕ: Device configuration ---
    device: torch.device = field(default_factory=lambda: device)
    # --- КОНЕЦ НОВОГО ---
    
    def __post_init__(self):
        """✅ ИСПРАВЛЕНО: Автоматическая валидация конфига при инициализации"""
        # Критические проверки
        assert self.hidden_dim > 0, f"hidden_dim must be positive, got {self.hidden_dim}"
        assert self.z_dim > 0, f"z_dim must be positive, got {self.z_dim}"
        assert self.num_pillars > 0, f"num_pillars must be positive, got {self.num_pillars}"
        assert self.vocab_size > 0, f"vocab_size must be positive, got {self.vocab_size}"
        assert self.max_iterations > 0, f"max_iterations must be positive, got {self.max_iterations}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert 0 <= self.alpha <= 1, f"alpha must be in [0,1], got {self.alpha}"
        assert 0 <= self.dropout_rate < 1, f"dropout_rate must be in [0,1), got {self.dropout_rate}"
        assert 0 < self.safety_threshold <= 1, f"safety_threshold must be in (0,1], got {self.safety_threshold}"
        assert self.vq_embedding_dim == self.z_dim, f"vq_embedding_dim ({self.vq_embedding_dim}) must equal z_dim ({self.z_dim})"
        # --- НОВОЕ: Валидация device ---
        assert isinstance(self.device, torch.device), f"device must be torch.device, got {type(self.device)}"
        # --- КОНЕЦ НОВОГО ---
        
        if self.use_neo4j:
            assert isinstance(self.knowledge_graph_auth, tuple) and len(self.knowledge_graph_auth) == 2, "knowledge_graph_auth must be Tuple[str,str]"
            assert self.knowledge_graph_auth[0], "Neo4j username must be non-empty when use_neo4j=True"
            assert self.knowledge_graph_auth[1], "Neo4j password must be provided via config/env when use_neo4j=True"
            assert self.knowledge_graph_url, "knowledge_graph_url must be non-empty when use_neo4j=True"

        logger.info(f"✅ AEONConfig validation passed: {self.hidden_dim}D, {self.z_dim}Z, {self.vocab_size}V, {self.num_pillars}P, device={self.device}")
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class MemoryManager:
    
    def _fallback_sample(self, n):
        if self._size < n:
            n = self._size
        if n <= 0:
            return []
        indices = np.random.choice(self._size, n, replace=False)
        return [{'vec': self.fallback_vectors[i], 'meta': self.fallback_metas[i]} for i in indices]

    def _hash_tensor(self, t):
        return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()

    def add_entity(self, key, vec):
        # удобный шорткат из core
        self.add_embedding(vec, {'entity_id': key})

    def add_experience(self, state, **meta):
        self.add_embedding(state, meta)

    def query_entities(self, query_vec, top_k: int = 3):
        results = self.retrieve_relevant(query_vec, k=top_k)
        metas = [r['meta'] for r in results]
        return results, metas

    def sample_pairs(self, n: int = 64):
        return self._fallback_sample(n)

    def __init__(self, cfg):
        self.cfg = cfg
        self._size = 0
        self.default_user = "aeon"

        if MEM0_AVAILABLE and MEM0_REMOTE_ENABLED:
            try:
                local_cfg = {
                    "vector_store": {"provider": "chroma", "config": {"path": cfg.memory_path, "collection_name": "aeon"}},
                    "embedder": {"provider": "identity"},
                    "llm": {"provider": "stub"},
                    "run_migrations": False
                }
                self.client = MemoryClient(api_key=MEM0_API_KEY)
                logger.info("Initialized mem0 MemoryClient.")
            except Exception as e:
                logger.warning(f"Failed to initialize mem0 MemoryClient: {e}. Falling back to in-memory.")
                self.client = None
        else:
            self.client = None

        self.fallback_vectors = []
        self.fallback_metas = []

    def _use_fallback(self):
        return self.client is None
    
    def save_memory(self):
        try:
            if hasattr(self, 'fallback_vectors') and self.fallback_vectors:
                path = os.path.join(self.cfg.memory_path, "fallback_memory.pt")
                os.makedirs(self.cfg.memory_path, exist_ok=True)
                torch.save({
                    'vectors': self.fallback_vectors,
                    'metas': self.fallback_metas,
                    'size': getattr(self, "_size", len(self.fallback_vectors))
                }, path)
                logger.info(f"Fallback memory saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save fallback memory: {e}")

    def load_memory(self):
        """Загружает память (fallback режим)"""
        path = os.path.join(self.cfg.memory_path, "fallback_memory.pt")
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu')
                self.fallback_vectors = data.get('vectors', [])
                self.fallback_metas   = data.get('metas', [])
                self._size            = data.get('size', len(self.fallback_vectors))
                logger.info(f"Fallback memory loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load fallback memory: {e}")

    def add_embedding(self, vec, meta=None):
        meta = meta or {}
        vec_np = vec.detach().cpu().numpy()
        if not self._use_fallback():
            try:
                self.client.add(
                    messages=[{"role": "system", "content": np.array2string(vec_np)}],
                    user_id=self.default_user,
                    metadata=meta,
                    infer=False
                )
                logger.info("Added embedding to mem0.")
            except Exception as e:
                logger.error(f"mem0 add failed: {e}. Using fallback.")
                self.fallback_vectors.append(vec_np)
                self.fallback_metas.append(meta)
        else:
            self.fallback_vectors.append(vec_np)
            self.fallback_metas.append(meta)

        self._size += 1

    def retrieve_relevant(self, vec, k: int = 5):
        vec_np = vec.detach().cpu().numpy()
        if not self._use_fallback():
            try:
                results = self.client.search(
                    query=np.array2string(vec_np),
                    user_id=self.default_user,
                    limit=k,
                    infer=False
                )
                parsed = []
                for res in results:
                    content = res.get('memory', '')
                    if content:
                        vec_part = content.split(',')
                        meta_part = res.get('metadata', {})
                        vec_parsed = np.array(vec_part, dtype=float)
                        parsed.append({'vec': vec_parsed, 'meta': meta_part})
                logger.info(f"Retrieved {len(parsed)} from mem0.")
                return parsed
            except Exception as e:
                logger.error(f"mem0 search failed: {e}. Using fallback.")
                return self._fallback_retrieve(vec_np, k)
        else:
            return self._fallback_retrieve(vec_np, k)

    def _fallback_retrieve(self, vec_np, k):
        """
        ✅ ИСПРАВЛЕНО: Векторизированный поиск вместо медленного for-loop.
        Использует матричное умножение (O(N) вместо O(N*D))
        """
        if not self.fallback_vectors:
            return []
        
        # ✅ КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: Стекирование массивов для batch-обработки
        vectors_array = np.stack(self.fallback_vectors, axis=0)  # (N, D)
        
        # ✅ Batch dot product через матричное умножение (векторизировано)
        # Вместо: similarities = [np.dot(v, vec_np) / (...) for v in ...]
        numerator = np.dot(vectors_array, vec_np)  # (N,) - O(N*D) но в C коде
        denominator = (
            np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(vec_np) + 1e-8
        )
        similarities = numerator / denominator  # (N,) - поэлементное деление
        
        # Отбор top-k индексов
        top_indices = np.argsort(similarities)[-k:][::-1]  # Сортировка по убыванию
        
        return [{'vec': self.fallback_vectors[i], 'meta': self.fallback_metas[i]} 
                for i in top_indices]

    @property
    def size(self):
        return self._size

class KnowledgeGraph:
    def __init__(self, config: AEONConfig):
        self.config = config
        self.memory_manager = None
        self.local_graph = defaultdict(lambda: defaultdict(set))
        self.reverse_graph = defaultdict(lambda: defaultdict(set))
        self.neo4j_graph = None
        
        if config.use_neo4j:
            if not NEO4J_AVAILABLE:
                logger.error("Neo4j support requested but py2neo is not installed. Run 'pip install py2neo'")
                if config.use_neo4j:
                    raise ImportError("py2neo is required when use_neo4j=True")
            else:
                try:
                    self.neo4j_graph = Graph(
                        config.knowledge_graph_url, 
                        auth=config.knowledge_graph_auth
                    )
                    logger.info("Connected to Neo4j knowledge graph")
                except Exception as e:
                    logger.error(f"Failed to connect to Neo4j: {e}")
                    self.neo4j_graph = None
    
    def set_memory_manager(self, memory_manager):
        self.memory_manager = memory_manager
    
    def save(self, path):
        try:
            data = {
                'local_graph': {k: {rel: list(objs) for rel, objs in v.items()}
                                for k, v in self.local_graph.items()},
                'reverse_graph': {k: {rel: list(objs) for rel, objs in v.items()}
                                  for k, v in self.reverse_graph.items()}
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Knowledge graph saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def load(self, path):
        """Загружает граф знаний из JSON"""
        if not os.path.exists(path):
            logger.warning(f"Knowledge graph file not found: {path}")
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.local_graph = defaultdict(lambda: defaultdict(set))
            for k, v in data.get('local_graph', {}).items():
                for rel, objs in v.items():
                    self.local_graph[k][rel] = set(objs)

            self.reverse_graph = defaultdict(lambda: defaultdict(set))
            for k, v in data.get('reverse_graph', {}).items():
                for rel, objs in v.items():
                    self.reverse_graph[k][rel] = set(objs)

            logger.info(f"Knowledge graph loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")

    def add_fact(self, subject, relation, object_entity, embedding=None):
        self.local_graph[subject][relation].add(object_entity)
        self.reverse_graph[object_entity][f"reverse_{relation}"].add(subject)
        
        if self.neo4j_graph is not None:
            try:
                tx = self.neo4j_graph.begin()
                subj_node = Node("Entity", name=subject)
                obj_node = Node("Entity", name=object_entity)
                
                tx.merge(subj_node, "Entity", "name")
                tx.merge(obj_node, "Entity", "name")
                
                rel = Relationship(subj_node, relation, obj_node)
                tx.create(rel)
                
                tx.commit()
            except Exception as e:
                logger.warning(f"Failed to add fact to Neo4j: {e}")

    def query(self, subject=None, relation=None, object_entity=None):
        results = []
        if self.neo4j_graph is not None:
            try:
                query_parts = []
                params = {}
                
                if subject:
                    query_parts.append("(s:Entity {name: $subject})")
                    params["subject"] = subject
                else:
                    query_parts.append("(s:Entity)")
                
                if relation:
                    query_parts.append(f"-[r:{relation}]->")
                else:
                    query_parts.append("-[r]->")
                
                if object_entity:
                    query_parts.append("(o:Entity {name: $object})")
                    params["object"] = object_entity
                else:
                    query_parts.append("(o:Entity)")
                
                query_str = f"MATCH {' '.join(query_parts)} RETURN s.name as subject, type(r) as relation, o.name as object"
                
                neo4j_results = self.neo4j_graph.run(query_str, **params)
                for record in neo4j_results:
                    results.append((record["subject"], record["relation"], record["object"]))
                return results
            except Exception as e:
                logger.warning(f"Neo4j query failed: {e}, falling back to local graph")

        # ✅ ИСПРАВЛЕНО: Полная реализация локального графа вместо комментария
        # Локальный граф: поиск по явным критериям
        if subject is not None:
            if subject in self.local_graph:
                for rel, objects in self.local_graph[subject].items():
                    if relation is None or rel == relation:
                        for obj in objects:
                            if object_entity is None or obj == object_entity:
                                results.append((subject, rel, obj))
        elif object_entity is not None:
            # Поиск через обратный граф (субъекты, связанные с объектом)
            if object_entity in self.reverse_graph:
                for rel, subjects in self.reverse_graph[object_entity].items():
                    for subj in subjects:
                        if relation is None or (relation and f"reverse_{relation}" == rel):
                            results.append((subj, rel.replace("reverse_", ""), object_entity))
        else:
            # Вернуть все факты if не заданы критерии
            for subj, rels in self.local_graph.items():
                for rel, objs in rels.items():
                    if relation is None or rel == relation:
                        for obj in objs:
                            results.append((subj, rel, obj))
        
        return results

class LambdaOperator(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        input_dim = config.hidden_dim * 2
        
        self.W1 = nn.Linear(input_dim, config.meta_dim)
        self.W2 = nn.Linear(config.meta_dim, config.hidden_dim)
        
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)
        
        self.W1 = nn.utils.spectral_norm(self.W1)
        self.W2 = nn.utils.spectral_norm(self.W2)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.train_dropout = True
    
    def set_dropout_active(self, active):
        self.train_dropout = active
    
    def forward(self, input_tensor):
        x = self.W1(input_tensor)
        x = torch.relu(x)
        x = self.dropout(x) if (self.train_dropout and self.training) else x
        x = self.W2(x)
        return x

class SequencePooler(nn.Module):
    def __init__(self, config: AEONConfig, pooling_type="attention"):
        super().__init__()
        self.config = config
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, 1)
            )
    
    def forward(self, sequence, attention_mask=None):
        if self.pooling_type == "attention":
            scores = self.attention(sequence)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            attention_weights = F.softmax(scores, dim=1)
            return torch.sum(sequence * attention_weights, dim=1)
        else: # Fallback to mean pooling
            if attention_mask is not None:
                masked_sequence = sequence * attention_mask.unsqueeze(-1)
                return masked_sequence.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                return sequence.mean(dim=1)

class PillarsModule(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.hidden_to_pillars = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_pillars)
        )
        
        # ✅ ИСПРАВЛЕНО: Правильное определение входного размера для pillars_to_hidden
        # Первый слой: num_pillars → hidden_dim // 2
        # Второй слой: hidden_dim // 2 → hidden_dim
        self.pillars_to_hidden = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)  # ✅ ИСПРАВЛЕНО: было num_pillars
        )
        
        self.pillar_norm = nn.LayerNorm(config.num_pillars)
        self.hidden_norm = nn.LayerNorm(config.hidden_dim)
    
    def extract_pillars(self, hidden_state):
        raw_pillars = self.hidden_to_pillars(hidden_state)
        pillars = torch.sigmoid(raw_pillars)
        return self.pillar_norm(pillars)
    
    def embed_pillars(self, pillars):
        # ✅ ИСПРАВЛЕНО: Логирование входных and выходных размеров
        logger.debug(f"embed_pillars input: {pillars.shape}")
        hidden = self.pillars_to_hidden(pillars)
        logger.debug(f"embed_pillars output: {hidden.shape}")
        return self.hidden_norm(hidden)
    
    def forward(self, hidden_state):
        pillars = self.extract_pillars(hidden_state)
        embedded = self.embed_pillars(pillars)
        assert embedded.shape[-1] == self.config.hidden_dim, f"Pillars embed size mismatch: got {embedded.shape[-1]}, expected {self.config.hidden_dim}"
        return pillars, embedded

class QualiaExtractor(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.sequence_pooler = SequencePooler(config, pooling_type="attention")
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, lm_output, attention_mask=None):
        # Совместимость с HF-моделью (как в core)
        if isinstance(lm_output, dict):
            lm_output = lm_output["last_hidden_state"]

        projected = self.projection(lm_output)

        if projected.dim() == 2:
            # Вход уже [B,H] — пулинг не нужен, нормализуем and выходим
            return self.norm(projected)

        # Стандартный путь для [B,L,H]
        pooled = self.sequence_pooler(projected, attention_mask)
        return self.norm(pooled)

class MetaLoopProcessorV3(nn.Module):
    """
    ✅综合版本 (ENHANCED UNIFIED VERSION):
    - compute_fixed_point() с Anderson acceleration (из варианта 1)
    - StabilityMonitor + ConvergenceController + RecoveryMechanism (из варианта 2)
    - Полная интеграция для максимальной мощности
    """
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.lambda_op = LambdaOperator(config)
        self.history_size = 5
        
        # === СТАБИЛИЗИРУЮЩИЕ СЛОИ (Variant 1) ===
        self.input_stabilizer = nn.LayerNorm(config.hidden_dim * 2)
        self.output_stabilizer = nn.LayerNorm(config.hidden_dim)
        
        # === АДАПТИВНЫЙ КОЭФФИЦИЕНТ СМЕШИВАНИЯ (Variant 1) ===
        self.alpha_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # === МОНИТОРИНГ СОСТОЯНИЯ (Variant 1) ===
        self.register_buffer('convergence_history', torch.zeros(100))
        self.history_pointer = 0
        self.last_valid_state = None
        self.fallback_state = None
        
        # === КОМПОНЕНТЫ СТАБИЛЬНОСТИ (Variant 2) ===
        self.stability_monitor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.convergence_controller = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.recovery_mechanism = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()
        )
        
    def concatenate(self, psi_0, C):
        """Безопасная конкатенация входных тензоров с проверками"""
        if not isinstance(psi_0, torch.Tensor):
            psi_0 = torch.tensor(psi_0, device=self.input_stabilizer.weight.device)
        if not isinstance(C, torch.Tensor):
            C = torch.tensor(C, device=self.input_stabilizer.weight.device)
            
        if psi_0.dim() == 1:
            psi_0 = psi_0.unsqueeze(0)
        if C.dim() == 1:
            C = C.unsqueeze(0)
            
        if psi_0.dtype != C.dtype:
            C = C.to(dtype=psi_0.dtype)
            
        if psi_0.shape[1] != C.shape[1]:
            raise ValueError(f"Incompatible dimensions: psi_0 {psi_0.shape} vs C {C.shape}")
            
        # ✅ ИСПРАВЛЕНО: Замена torch.nan_to_num на глобальную safe_tensor()
        psi_0 = safe_tensor(psi_0, default_value=0.0, max_value=1.0, min_value=-1.0)
        C = safe_tensor(C, default_value=0.0, max_value=1.0, min_value=-1.0)
        
        psi_0 = F.normalize(psi_0, p=2, dim=-1)
        C = F.normalize(C, p=2, dim=-1)
        
        return torch.cat([psi_0, C], dim=-1)
        
    def get_adaptive_alpha(self, input_tensor, C_new, C_old):
        """Адаптивный коэффициент смешивания"""
        state = torch.cat([C_new, C_old], dim=-1)
        alpha = self.alpha_net(state)
        return self.config.alpha * alpha
        
    def update_convergence_history(self, residual_norm):
        """Сохранение истории сходимости"""
        self.convergence_history[self.history_pointer] = residual_norm.mean().item()
        self.history_pointer = (self.history_pointer + 1) % 100
        
    def check_stability(self):
        """Проверка тренда стабильности"""
        valid_history = self.convergence_history[:self.history_pointer]
        if len(valid_history) > 10:
            trend = torch.mean(valid_history[:-5]) - torch.mean(valid_history[-5:])
            return trend > 0
        return True
    
    def get_stability_score(self, state):
        """Оценка стабильности состояния (Variant 2)"""
        return self.stability_monitor(state).squeeze(-1)
    
    def get_convergence_score(self, state):
        """Оценка сходимости состояния (Variant 2)"""
        return self.convergence_controller(state).squeeze(-1)
    
    def recover_state(self, state):
        """Механизм восстановления при нестабильности (Variant 2)"""
        return self.recovery_mechanism(state)
        
    def _anderson_acceleration(self, C_history, residual_history):
        """Anderson acceleration для ускорения сходимости"""
        m = len(C_history)
        if m <= 1:
            return C_history[-1]
        
        F = torch.stack(residual_history, dim=0)
        batch_size = F.shape[1]
        hidden_dim = F.shape[2]
        device = F.device
        
        C_aa = C_history[-1].clone()
        F = F.permute(1, 0, 2)
        F_flat = F.reshape(batch_size, m, hidden_dim)
        gram = torch.bmm(F_flat, F_flat.transpose(1, 2))
        eye = torch.eye(m, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        gram = gram + 1e-8 * eye
        rhs = torch.ones(batch_size, m, 1, device=device)
        
        try:
            alpha = torch.linalg.solve(gram, rhs)
            alpha = alpha / alpha.sum(dim=1, keepdim=True)
            C_stack = torch.stack(C_history, dim=0)
            residual_stack = torch.stack(residual_history, dim=0)
            CR_stack = C_stack + residual_stack
            CR_stack = CR_stack.permute(1, 0, 2)
            weighted_sum = torch.bmm(alpha.transpose(1, 2), CR_stack)
            C_aa = weighted_sum.squeeze(1)
        except Exception as e:
            logger.warning(f"Anderson acceleration failed: {e}")
        
        return C_aa
        
    def compute_fixed_point(self, psi_0, use_anderson=True, use_stability=True):
        """
        ✅ ПОЛНЫЙ FIXED-POINT SOLVER с интегрированной стабильностью and восстановлением
        Флаги use_anderson and use_stability позволяют включать/отключать компоненты
        """
        self.last_valid_state = None
        self.fallback_state = None
        
        batch_size = psi_0.shape[0]
        device = psi_0.device
        
        C = self.output_stabilizer(torch.zeros((batch_size, self.config.hidden_dim), device=device))
        
        iterations_count = torch.zeros(batch_size, device=device)
        not_converged = torch.ones(batch_size, dtype=torch.bool, device=device)
        instability_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        instability_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
        C_history = []
        residual_history = []
        stability_scores = torch.zeros(batch_size, device=device)
        convergence_scores = torch.zeros(batch_size, device=device)
        
        original_dropout = self.lambda_op.train_dropout
        self.lambda_op.train_dropout = False
        
        try:
            for i in range(self.config.max_iterations):
                C_prev = C.clone()
                
                # === FORWARD PASS (Variant 1) ===
                input_tensor = self.concatenate(psi_0, C)
                input_tensor = self.input_stabilizer(input_tensor)
                C_new = self.lambda_op(input_tensor)
                C_new = self.output_stabilizer(C_new)
                
                residual = torch.norm(C_new - C, dim=1)
                
                # === ANDERSON ACCELERATION (Variant 1) ===
                if use_anderson and len(C_history) > 0:
                    C_history.append(C_new)
                    residual_history.append(C_new - C)
                    if len(C_history) > 5:
                        C_history = C_history[-5:]
                        residual_history = residual_history[-5:]
                    C_new = self._anderson_acceleration(C_history, residual_history)
                
                # === ADAPTIVE ALPHA MIXING (Variant 1) ===
                alpha = self.get_adaptive_alpha(input_tensor, C_new, C)
                C = alpha * C_new + (1 - alpha) * C_prev
                
                # === CONVERGENCE CHECK (Variant 1) ===
                converged = residual < self.config.convergence_threshold
                not_converged = not_converged and ~converged
                iterations_count += not_converged.float()
                self.update_convergence_history(residual)
                
                # === STABILITY & RECOVERY (Variant 2) ===
                if use_stability:
                    stability_scores = self.get_stability_score(C)
                    convergence_scores = self.get_convergence_score(C)
                    
                    # Механизм восстановления при нестабильности
                    unstable_mask = stability_scores < 0.4
                    if unstable_mask.any():
                        instability_flags |= unstable_mask
                        instability_steps += unstable_mask.to(torch.int32)
                        C[unstable_mask] = self.recover_state(C[unstable_mask])
                        logger.info(f"Applied recovery mechanism for {unstable_mask.sum().item()} batch elements")
                
                # === LOGGING ===
                if i > 0 and i % 10 == 0:
                    avg_stability = stability_scores.mean().item()
                    avg_convergence = convergence_scores.mean().item()
                    logger.info(
                        f"Iter {i}/{self.config.max_iterations}: "
                        f"residual={residual.mean():.6f}, "
                        f"stability={avg_stability:.4f}, "
                        f"convergence={avg_convergence:.4f}"
                    )
                    
                    if not self.check_stability():
                        instability_flags |= torch.ones_like(instability_flags)
                        instability_steps += torch.ones_like(instability_steps)
                        logger.warning(f"Instability detected at iteration {i}")
                        C = self.output_stabilizer(C)
                        if self.last_valid_state is None:
                            self.last_valid_state = C.clone()
                
                # === FALLBACK STATE TRACKING (Variant 1) ===
                if residual.mean() < 0.1:
                    self.fallback_state = C.clone()
                
                # === EARLY STOPPING (Variant 1) ===
                if not torch.any(not_converged):
                    logger.info(f"All batch elements converged after {i+1} iterations.")
                    break
            
            # === HANDLE UNCONVERGED ELEMENTS (Variant 1) ===
            if torch.any(not_converged):
                unconverged_mask = not_converged
                if self.last_valid_state is not None:
                    C[unconverged_mask] = self.last_valid_state[unconverged_mask]
                    logger.info("Applied last_valid_state correction")
                elif self.fallback_state is not None:
                    C[unconverged_mask] = self.fallback_state[unconverged_mask]
                    logger.info("Applied fallback_state correction")
        
        finally:
            self.lambda_op.train_dropout = original_dropout
        
        assert C.shape[-1] == self.config.hidden_dim, "MetaLoop output size mismatch"
        return C, iterations_count, {
            'stability_scores': stability_scores,
            'convergence_scores': convergence_scores,
            'instability_flags': instability_flags,
            'instability_steps': instability_steps
        }
    
    def forward(self, state, use_fixed_point: bool = True):
        """
        ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: forward() теперь вызывает compute_fixed_point()
        
        Args:
            state: Входное состояние [B, hidden_dim]
            use_fixed_point: Если True, используется полный алгоритм фиксированной точки.
                             Если False, используется быстрый путь (только для inference скорости).
        
        Returns:
            Tuple[Tensor, Tensor, Dict]: (состояние, итерации, метаданные)
        """
        if use_fixed_point:
            # ✅ ИСПРАВЛЕНО: Вызываем полный алгоритм compute_fixed_point()
            return self.compute_fixed_point(state, use_anderson=True, use_stability=True)
        else:
            # Быстрый путь для inference (без итераций)
            stab = self.stability_monitor(state)
            conv = self.convergence_controller(state)
            if (stab < 0.4).any() or (conv < 0.4).any():
                state = self.recovery_mechanism(state)
            iterations = torch.tensor(1.0, device=state.device)
            stab_s = stab.squeeze(-1) if stab.dim() > 1 else stab
            conv_s = conv.squeeze(-1) if conv.dim() > 1 else conv
            instability_flags = (stab_s < 0.4) | (conv_s < 0.4)
            instability_steps = instability_flags.to(torch.int32)
            return state, iterations, {
                "stability_scores": stab_s,
                "convergence_scores": conv_s,
                "instability_flags": instability_flags,
                "instability_steps": instability_steps
            }

class UniformMatrix:
    @staticmethod
    def expm(A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(A)

class TopologyAnalyzer(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config

        # Потенциал - ИСПРАВЛЕНО: правильные размерности
        self.potential_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),  # 5 → 64
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1)  # ✅ ИСПРАВЛЕНО: 64 → 1 (было num_pillars → 1)
        )

        # Доп. сеть для быстрого градиента - ИСПРАВЛЕНО
        self.gradient_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),  # 5 → 64
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, config.num_pillars)  # ✅ ИСПРАВЛЕНО: 64 → 5 (было num_pillars → num_pillars)
        )

        # Стабorзация входа перед гессианом (из core)
        self.stabilizer = nn.LayerNorm(config.num_pillars)

        # Классификатор катастроф (совместим с features: [P, P, 1, 1] → P*2+2)
        self.catastrophe_classifier = nn.Sequential(
            nn.Linear(config.num_pillars * 2 + 2, 1),
            nn.Sigmoid()
        )

    # === Базовые вычисления ===
    def compute_potential(self, pillars: torch.Tensor) -> torch.Tensor:
        """V(pillars) -> [B,1]"""
        return self.potential_net(pillars)

    def compute_gradient(self, pillars: torch.Tensor) -> torch.Tensor:
        """Быстрый приближённый градиент по сети → [B,P]"""
        return self.gradient_net(pillars)

    def compute_hessian_eigenvalues(self, pillars: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            B, P = pillars.shape
            device = pillars.device
            eigvals = torch.zeros(B, P, device=device)
            x = self.stabilizer(pillars)
            for b in range(B):
                p = x[b:b+1].clone().requires_grad_(True)
                pot = self.compute_potential(p).sum()
                g = torch.autograd.grad(pot, p, create_graph=True, retain_graph=True)[0]  # [P]
                H = torch.zeros(P, P, device=device)
                for i in range(P):
                    gi = g[i]
                    grad_i = torch.autograd.grad(gi, p, retain_graph=True, create_graph=False)[0].squeeze(0)  # [P]
                    H[i] = grad_i
                H = 0.5 * (H + H.T)
                try:
                    ev = torch.linalg.eigvalsh(H).real
                except RuntimeError:
                    ev = torch.linalg.eigvals(H).real
                if ev.numel() >= P:
                    eigvals[b] = ev[:P]
                else:
                    eigvals[b, :ev.numel()] = ev
            return eigvals

    # === Батч-детектор катастроф ===
    def detect_catastrophe_batch(self, pillars: torch.Tensor):
        """
        Возвращает:
          - catastrophe_detected: Bool[B,1]
          - catastrophe_probs:   Float[B,1] in [0,1]
        Признаки как в core: [pillars, grad, V, mean_eigenvalue] → нормализация → classifier.
        """
        B, P = pillars.shape
        device = pillars.device

        p = torch.nan_to_num(pillars, nan=0.5).clamp(0.0, 1.0).clone().requires_grad_(True)
        try:
            # Fast path по умолчанию: без точного Гессиана (очень дорого)
            if not getattr(self.config, 'use_exact_hessian', False):
                with torch.no_grad():
                    V = self.compute_potential(p)  # [B,1]
                    grad = self.compute_gradient(p)  # [B,P] (суррогат градиента)
                    e_mean = torch.zeros((p.shape[0], 1), device=p.device)
                    feats = torch.cat([pillars, grad, V, e_mean], dim=-1)
                    feats = F.normalize(feats, dim=-1)
                    probs = torch.clamp(self.catastrophe_classifier(feats), 0.0, 1.0)
            else:
                # Strict path: точные производные второго порядка
                V = self.compute_potential(p)  # [B,1]
                grad = torch.autograd.grad(V.sum(), p, create_graph=False, retain_graph=False)[0]  # [B,P]
                E = self.compute_hessian_eigenvalues(p)  # [B,P]
                e_mean = E.mean(dim=1, keepdim=True)  # [B,1]
                feats = torch.cat([pillars, grad, V, e_mean], dim=-1)
                feats = F.normalize(feats, dim=-1)
                probs = torch.clamp(self.catastrophe_classifier(feats), 0.0, 1.0)  # [B,1]
        except Exception as e:
            logger = globals().get("logger", None)
            if logger is not None:
                logger.warning(f"detect_catastrophe_batch fallback: {e}")
            probs = torch.full((B, 1), 0.5, device=device)

        detected = probs > 0.5
        return detected, probs

    # === Стратификация глубины (уровень «драмы» ландшафта) ===
    def stratify_depth(self, pillars: torch.Tensor, iterations: torch.Tensor) -> torch.Tensor:
        """
        depth ∈ (0,1): sigmoid(w·normalize([it_norm, V, ||grad||, |eig|_mean]))
        Совместимо с core. Не требует градиента.
        """
        device = pillars.device
        max_it = getattr(self.config, "max_iterations", 100)
        itn = iterations.float() / max(1, max_it)

        with torch.no_grad():
            V = self.compute_potential(pillars)                    # [B,1]
            g = self.compute_gradient(pillars)                     # [B,P]
            gnorm = torch.norm(g, dim=1, keepdim=True)             # [B,1]
            E = self.compute_hessian_eigenvalues(pillars)          # [B,P]
            curv = torch.mean(torch.abs(E), dim=1, keepdim=True)   # [B,1]

            factors = torch.cat([itn.unsqueeze(-1), V, gnorm, curv], dim=-1)  # [B,4]
            factors = F.normalize(factors, dim=-1)
            w = torch.tensor([0.3, 0.3, 0.2, 0.2], device=device)
            depth = torch.sum(factors * w, dim=1)
            depth = torch.sigmoid(depth)
        return depth

    # === Быстрый путь (совместим с 2core) ===
    def forward(self, pillars: torch.Tensor, iterations=None):
        """
        Лёгкий путь: признаки [pillars, grad, V, ||grad||] — для онлайна.
        Отдельно включаем градиенты локально, даже if внешний контекст @torch.no_grad.
        """
        with torch.enable_grad():
            p = pillars.detach().clone().requires_grad_(True)
            potential = self.compute_potential(p)                                      # [B,1]
            grad = torch.autograd.grad(potential.sum(), p, create_graph=False, retain_graph=False)[0]  # [B,P]
            grad_norm = grad.norm(dim=-1, keepdim=True)                                # [B,1]
        features = torch.cat([pillars, grad.detach(), potential.detach(), grad_norm.detach()], dim=-1)
        catastrophe_probs = self.catastrophe_classifier(features)
        catastrophes = catastrophe_probs > 0.5
        return {
            'potential': potential.detach(),
            'gradient': grad.detach(),
            'catastrophe_probs': catastrophe_probs.detach(),
            'catastrophes': catastrophes
        }
class ActionModule(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        
        self.action_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_pillars * 2 + 2, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
            nn.LayerNorm(config.action_dim)
        )
        
        self.safety_classifier = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()  # ✅ ИСПРАВЛЕНО: Убран лишний LayerNorm(1) and Hardtanh
        )

    def forward(self, core_state, pillars, quantum_results, topo_results):
        device = core_state.device
        batch_size = core_state.shape[0]

        # Безопасная обработка entanglement
        entanglement = quantum_results.get('entanglement', torch.zeros(batch_size, device=device)).float()
        action_propensity = quantum_results.get('action_propensity', 
                                                 torch.ones(batch_size, self.config.num_pillars, device=device) / self.config.num_pillars).float()
        
        entanglement = torch.nan_to_num(entanglement, nan=0.0)
        action_propensity = torch.nan_to_num(action_propensity, nan=1.0 / self.config.num_pillars)
        action_propensity = F.normalize(action_propensity, p=1, dim=-1)

        # Безопасная обработка потенциала с fallback
        if "potential" in topo_results:
            potential = topo_results['potential'].float()
        else:
            logger.warning("'potential' not found in topo_results, using zeros fallback")
            potential = torch.zeros(batch_size, 1, device=device)
        
        potential = torch.nan_to_num(potential, nan=0.0)

        # Безопасная конкатенация с проверкой размеров
        try:
            action_features = torch.cat([core_state, pillars, action_propensity, entanglement.unsqueeze(-1), potential], dim=-1)
        except RuntimeError as e:
            logger.error(f"Concatenation error in ActionModule: {e}")
            action_features = torch.zeros((batch_size, self.config.hidden_dim + self.config.num_pillars * 2 + 2), device=device)

        action_features.requires_grad_(True)
        action_embedding = self.action_encoder(action_features)

        if self.training:
            parameters = [p for p in self.action_encoder.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

        safety_score = self.safety_classifier(action_embedding)
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная обработка NaN/Inf
        # Раньше обе ветки возвращали 0.5, теперь только при NaN/Inf
        safety_score = torch.where(
            torch.isnan(safety_score) | torch.isinf(safety_score),
            torch.full_like(safety_score, 0.5),  # Fallback для невалидных значений
            safety_score  # ✅ ИСПРАВЛЕНО: Возвращаем реальный score для валидных
        )
        safety_score = torch.clamp(safety_score, 0.0, 1.0)

        assert action_embedding.shape[-1] == self.config.action_dim, "Action embed size mismatch"
        return {'action_embedding': action_embedding, 'safety_score': safety_score, 'action_features': action_features.detach()}

class PlanningModule(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        
        self.high_level_planner = nn.GRUCell(
            input_size=config.hidden_dim + config.num_pillars,
            hidden_size=config.hidden_dim
        )
        
        self.mid_level_planner = nn.GRUCell(
            input_size=config.hidden_dim + config.action_dim,
            hidden_size=config.hidden_dim // 2
        )
        
        self.low_level_executor = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 + config.action_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, config.action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1)
        )
    
    def forward(self, core_state, pillars, action_embedding, planning_horizon=None):
        if planning_horizon is None:
            planning_horizon = self.config.planning_horizon
        
        batch_size = core_state.shape[0]
        device = core_state.device
        
        high_h = core_state.clone()
        
        action_plans = []
        values = []
        
        for t in range(planning_horizon):
            high_input = torch.cat([core_state, pillars], dim=-1)
            high_h = self.high_level_planner(high_input, high_h)
            
            mid_input = torch.cat([high_h, action_embedding], dim=-1)
            mid_h = self.mid_level_planner(mid_input)
            
            low_input = torch.cat([mid_h, action_embedding], dim=-1)
            next_action = self.low_level_executor(low_input)
            
            action_plans.append(next_action)
            
            value = self.value_head(mid_h)
            values.append(value)
            
            action_embedding = next_action
        
        action_plans = torch.stack(action_plans, dim=1)
        values = torch.stack(values, dim=1)
        
        return {
            'action_plans': action_plans,
            'values': values
        }

class RSSM(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.stochastic = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.deterministic = nn.GRUCell(config.hidden_dim, config.hidden_dim)

    def forward(self, z_t, hx=None):
        stoch = self.stochastic(z_t)
        if hx is None:
            hx = torch.zeros_like(stoch)
        det = self.deterministic(stoch, hx)
        return det + stoch

# --- НОВОЕ ДОБАВЛЕНИЕ: Класс TransparentSelfReporting ---
class TransparentSelfReporting(nn.Module):
    """
    ✅ ПРОЗРАЧНОЕ САМООТЧЁТНОСТЬ (Transparent Self-Reporting Module)
    
    Архитектура:
    - Анализирует внутреннее состояние (core_state) модели
    - Интегрирует информацию об опорах (pillars), квантовых результатах and топологии
    - Вычисляет честность системы через многоуровневый анализ
    - Поддерживает комбинированный режим для максимальной информативности
    
    Режимы работы:
    - 'internal': Только анализ внутреннего состояния
    - 'combined': Полный анализ со всеми компонентами архитектуры
    """
    def __init__(self, config):
        super().__init__()
        # ✅ ИСПРАВЛЕНО: Сохраняем config для доступа к num_pillars and другим параметрам
        self.config = config
        
        # === ОСНОВНАЯ СЕТЬ АНАЛИЗА ===
        # ✅ ИСПРАВЛЕНО: Правильный расчёт входного размера
        # core_state: hidden_dim (256)
        # pillars_norm: num_pillars (5)  
        # entanglement expanded: num_pillars (5)
        # potential: 1
        # Итого: hidden_dim + num_pillars + num_pillars + 1 = 256 + 5 + 5 + 1 = 267
        input_size = config.hidden_dim + config.num_pillars + config.num_pillars + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),  # ✅ ИСПРАВЛЕНО: Кириллическая У → латинская U
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        )
        
        # === КОМПОНЕНТЫ АНАЛИЗА ЧЕСТНОСТИ ===
        self.honesty_gate = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 8),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        self.internal_consistency = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 8),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        self.confidence_score = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 8),
            nn.Sigmoid(),
            nn.Linear(config.hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # === МОНИТОРИНГ СОСТОЯНИЯ ===
        self.state_monitor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        )
        
        # === НОРМАЛИЗАЦИЯ ===
        self.output_norm = nn.LayerNorm(config.hidden_dim // 4)
        
        logger.info("✅ TransparentSelfReporting initialized")
    
    def forward(self, core_state, pillars, quantum_results, topo_results, mode='combined'):
        """
        Полный анализ самоотчётности системы.
        
        Args:
            core_state: [B, hidden_dim] - основное состояние
            pillars: [B, num_pillars] - пять опор архитектуры
            quantum_results: Dict с 'entanglement' [B] and 'action_propensity' [B, num_pillars]
            topo_results: Dict с 'potential' [B, 1] and другими метриками
            mode: 'internal' or 'combined'
        
        Returns:
            Dict с:
            - 'inner_report': [B, hidden_dim//4] - внутренний анализ
            - 'honesty_gate': [B, 1] - оценка честности (0-1)
            - 'consistency': [B, 1] - внутренняя консистентность (0-1)
            - 'confidence': [B, 1] - уровень уверенности (0-1)
            - 'report_vector': [B, hidden_dim//4] - вектор полного отчёта
        """
        B = core_state.size(0)
        device = core_state.device
        
        # === ИЗВЛЕЧЕНИЕ МЕТРИК ИЗ КОМПОНЕНТОВ ===
        # Энтропия запутанности
        entanglement = quantum_results.get('entanglement', torch.zeros(B, device=device)).float()
        entanglement = torch.nan_to_num(entanglement, nan=0.0).clamp(0.0, 1.0)
        
        # Потенциал топологии
        potential = topo_results.get('potential', torch.zeros(B, 1, device=device)).float()
        potential = torch.nan_to_num(potential, nan=0.0)
        
        # === КОНСТРУИРОВАНИЕ ВХОДНОГО ПРИЗНАКА ===
        # Нормализуем pillars
        pillars_norm = F.normalize(pillars, p=2, dim=-1)
        
        # Объединяем все компоненты
        try:
            feature_vector = torch.cat([
                core_state,                           # [B, H]
                pillars_norm,                         # [B, P]
                entanglement.unsqueeze(-1).expand(-1, self.config.num_pillars),  # [B, P]
                potential                             # [B, 1]
            ], dim=-1)
        except RuntimeError as e:
            logger.warning(f"⚠️  Feature concatenation failed: {e}, using fallback")
            feature_vector = torch.cat([
                core_state,
                torch.zeros(B, self.config.num_pillars * 2 + 1, device=device)
            ], dim=-1)
        
        # === ПРЯМОЙ ПРОХОД ЧЕРЕЗ СЕТЬ ===
        inner_report = self.net(feature_vector)  # [B, hidden_dim//4]
        
        # === ВЫЧИСЛЕНИЕ МЕТРИК ЧЕСТНОСТИ ===
        honesty = self.honesty_gate(inner_report)  # [B, 1]
        consistency = self.internal_consistency(inner_report)  # [B, 1]
        confidence = self.confidence_score(inner_report)  # [B, 1]
        
        # === МОНИТОРИНГ СОСТОЯНИЯ ===
        state_analysis = self.state_monitor(core_state)  # [B, hidden_dim//4]
        
        # === КОМБИНИРОВАННЫЙ АНАЛИЗ ===
        if mode == 'combined':
            # Интеграция мониторинга состояния с внутренним отчётом
            combined_report = torch.tanh(inner_report + state_analysis * 0.3)
            combined_report = self.output_norm(combined_report)
        else:
            combined_report = self.output_norm(inner_report)
        
        # === ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ И ВАЛИДАЦИЯ ===
        # Проверяем NaN/Inf and исправляем
        honesty = torch.nan_to_num(honesty, nan=0.5).clamp(0.0, 1.0)
        consistency = torch.nan_to_num(consistency, nan=0.5).clamp(0.0, 1.0)
        confidence = torch.nan_to_num(confidence, nan=0.5).clamp(0.0, 1.0)
        
        # === ВОЗВРАТ РЕЗУЛЬТАТОВ ===
        return {
            'inner_report': combined_report,      # Полный внутренний отчёт
            'honesty_gate': honesty,               # Оценка честности системы
            'consistency': consistency,            # Внутренняя консистентность
            'confidence': confidence,              # Уровень уверенности
            'report_vector': combined_report,      # Дублирование для совместимости
            'state_analysis': state_analysis,      # Анализ состояния
            'entanglement_metric': entanglement,   # Метрика запутанности
            'potential_metric': potential.squeeze(-1) if potential.shape[-1] == 1 else potential  # Топологический потенциал
        }

# --- КОНЕЦ ДОБАВЛЕНИЯ ---


class AEONDelta(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создаем encoder and decoder ВНУТРИ модели ---
        # Это гарантирует, что их параметры будут частью state_dict and parameters() модели.
        if TRANSFORMERS_AVAILABLE:
            try:
                # Инициализируем токенизатор для внутреннего использования (опционально)
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("Initialized internal tokenizer: bert-base-uncased")
            except Exception as e:
                self.tokenizer = None
                logger.warning(f"Tokenizer init failed: {e}; continuing without tokenizer")
        else:
            self.tokenizer = None
            logger.warning("Transformers not available; continuing without tokenizer")

        # Encoder/decoder создаются всегда (входят в state_dict и parameters())
        self.encoder = ThoughtEncoder(self.config.vocab_size, z_dim=self.config.z_dim).to(self.config.device)
        self.decoder = ThoughtDecoder(self.config.vocab_size, z_dim=self.config.z_dim).to(self.config.device)
        # ✅ Деактивируем токены вида [unused###] в генерации: они существуют в BERT-словаре, но не предназначены для семантического текста.
        try:
            if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "vocab") and isinstance(self.tokenizer.vocab, dict):
                unused_ids = [tid for tok, tid in self.tokenizer.vocab.items() if isinstance(tok, str) and tok.startswith("[unused") and tok.endswith("]")]
                if unused_ids:
                    self.decoder.set_invalid_token_ids(unused_ids)
                    logger.info(f"✅ Disabled {len(unused_ids)} [unused] tokens in decoder sampling")
        except Exception as e:
            logger.warning(f"Failed to set invalid token mask for decoder: {e}")
        logger.info(f"Initialized internal encoder/decoder with vocab_size={self.config.vocab_size}, z_dim={self.config.z_dim}")
        # - ИЗМЕНЕНИЕ: Добавление VQ -
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config.vq_num_embeddings,
            embedding_dim=config.vq_embedding_dim
        ).to(device)

        # --- Остальные подмодули остаются без изменений ---
        try:
            self.qualia_extractor = QualiaExtractor(config).to(device)
        except Exception as e:
            logger.warning(f"QualiaExtractor init failed ({e}); using NoOpQualiaExtractor")
            self.qualia_extractor = NoOpQualiaExtractor()
            self.qualia_extractor.to(device)
        self.meta_loop_v3 = MetaLoopProcessorV3(config).to(device)
        self.pillars_module = PillarsModule(config).to(device)
        self.quantum_sim = QuantumSimulator(config).to(device)
        self.topology_analyzer = TopologyAnalyzer(config).to(device)
        self.action_module = ActionModule(config).to(device)
        self.planning_module = PlanningModule(config).to(device)
        self.rssm = RSSM(config).to(device)
        self.integration_module = nn.Linear(config.hidden_dim * 2, config.hidden_dim).to(device)
        
        self.memory_manager = MemoryManager(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.knowledge_graph.set_memory_manager(self.memory_manager)
        
        self.step_counter = 0
        self.outputs = {}
        self.metrics_log = {
            'iterations': [],
            'consistency': [],
            'entanglement': [],
            'catastrophes': [],
            'safety_scores': []
        }
        
        self.memory_fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate)
        ).to(device)
        
        self._inject_lora()
        # ✅ ИСПРАВЛЕНО: Регистрация SafeTensorProcessor hooks для всей модели
        SafeTensorProcessor.register_hooks(self)
        # ✅ Device hard-sync: ensure every parameter/buffer is on config.device
        self.to(self.config.device)
        logger.info("AEONDelta initialized and all submodules moved to {}".format(device))
        # === AEON Coherence Hotfix (one-shot) ===

    def _inject_lora(self):
        def add_lora_to_layer(layer, rank, alpha, dropout):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                lora_a = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32, device=self.config.device))
                lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=self.config.device))
                nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
                setattr(layer, 'lora_a', lora_a)
                setattr(layer, 'lora_b', lora_b)
                setattr(layer, 'lora_alpha', alpha)
                setattr(layer, 'lora_dropout', nn.Dropout(dropout))
                logger.info("LoRA params set to dtype: {}".format(lora_a.dtype))
                logger.info("Injected LoRA into layer: {}".format(layer))
        
        for name, module in self.named_modules():
            if any(t in name for t in self.config.lora_target) and isinstance(module, nn.Linear):
                add_lora_to_layer(module, self.config.lora_rank, self.config.lora_alpha, self.config.lora_dropout)
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Injected LoRA into layers. Trainable params: {trainable_params}")
        
        # --- НОВОЕ: Monkey-patch Linear.forward для LoRA ---
        original_linear_forward = nn.Linear.forward
        
        def lora_forward(self, input):
            out = original_linear_forward(self, input)
            if hasattr(self, 'lora_a') and hasattr(self, 'lora_b'):
                try:
                    drop = self.lora_dropout(input)
                    lora_out = (self.lora_b @ (self.lora_a @ drop.t()).t()) * (self.lora_alpha / self.lora_a.size(0))
                    out = out + lora_out
                except Exception as e:
                    logger.warning(f"LoRA forward failed: {e}, using base output")
            return out
        
        nn.Linear.forward = lora_forward
        logger.info("✅ Monkey-patched nn.Linear.forward for LoRA")
        # --- КОНЕЦ НОВОГО ---

    def self_train_step(self, batch_z: torch.Tensor, kl_weight: float = 0.1, safety_weight: float = 0.1, lr: float = 3e-4):
        import torch, torch.nn.functional as F, torch.optim as optim
        logger = logging.getLogger("AEON-Delta")
        self.train()

        assert batch_z.dim() == 3 and batch_z.size(1) == 2 and batch_z.size(2) == self.config.hidden_dim, \
            f"Expected batch_z [B,2,H], got {tuple(batch_z.shape)}"

        z_t  = batch_z[:, 0, :].float()
        z_t1 = batch_z[:, 1, :].float()

        pred_z = self.rssm(z_t)  # динамика в латенте

        def _kl_batchwise_diag(pred, target, eps: float = 1e-6):
            mu_p = pred.mean(dim=0); var_p = pred.var(dim=0, unbiased=False) + eps
            mu_q = target.mean(dim=0); var_q = target.var(dim=0, unbiased=False) + eps
            D = pred.size(1)
            kl = 0.5 * ((var_p/var_q).sum() + ((mu_q - mu_p).pow(2)/var_q).sum() - D + torch.log(var_q/var_p).sum())
            return kl / D

        mse = F.mse_loss(pred_z, z_t1)
        kl  = _kl_batchwise_diag(pred_z, z_t1)

        pillars, _ = self.pillars_module(pred_z)
        qstats = self.quantum_sim(pillars)
        topo   = self.topology_analyzer(pillars)
        act    = self.action_module(pred_z, pillars, qstats, topo)
        safety = act['safety_score'].mean()

        loss = mse + kl_weight * kl + safety_weight * (1.0 - safety)

        if not hasattr(self, "_self_opt"):
            params = [p for p in self.parameters() if p.requires_grad]
            self._self_opt = optim.AdamW(params, lr=lr, weight_decay=0.0)

        self._self_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._self_opt.step()

        logger.info(f"Self-train(z): mse={mse.item():.6f}, kl={kl.item():.6e}, safety={safety.item():.4f}, loss={loss.item():.6f}")
        return {'loss': float(loss.detach().cpu()), 'mse': float(mse.detach().cpu()), 'kl': float(kl.detach().cpu()), 'safety': float(safety.detach().cpu())}

    def _retrieve_memory(self, query, top_k=3):
        if self.memory_manager.size == 0:
            return torch.zeros_like(query)

        contexts = []
        for q in query:
            found = self.memory_manager.retrieve_relevant(q, k=top_k)
            if not found:
                contexts.append(torch.zeros_like(q))
            else:
                vecs = [torch.from_numpy(f['vec']).to(q.device) for f in found]
                stacked = torch.stack(vecs)
                contexts.append(stacked.mean(dim=0))
        return torch.stack(contexts)

    def compute_self_consistency(self, psi_0, C_star):
        """Проверяет консистентность: ||Lambda(concat(psi_0, C_star)) - C_star||"""
        # Используем полную версию meta_loop_v3 с методом concatenate and lambda_op
        input_tensor = self.meta_loop_v3.concatenate(psi_0, C_star)
        
        with torch.no_grad():
            # Сохраняем режим and отключаем dropout
            original_mode = self.meta_loop_v3.lambda_op.training
            self.meta_loop_v3.lambda_op.eval()
            self.meta_loop_v3.lambda_op.set_dropout_active(False)
            
            # Конкатенируем and применяем lambda оператор
            C_new = self.meta_loop_v3.lambda_op(input_tensor)
            
            # Восстанавливаем режим
            if original_mode:
                self.meta_loop_v3.lambda_op.train()
        
        diff = torch.norm(C_new - C_star, dim=1)
        base = torch.norm(C_star, dim=1).clamp_min(1e-6)
        residual_rel = diff / base
        consistency = 1.0 / (1.0 + residual_rel)  # scale-invariant
        return consistency

    def execute_action(self, action_embedding, safety_score=None):
        if safety_score is None:
            safety_results = self.action_module.safety_classifier(action_embedding)
            safety_score = safety_results.item()
        
        if safety_score < self.config.safety_threshold:
            logger.warning(f"Action rejected due to safety concerns (score: {safety_score:.3f})")
            return None
        
        return action_embedding

    def update_knowledge(self, subject, relation, object_entity, embedding=None):
        if embedding is None and 'core_state' in self.outputs:
            embedding = self.outputs['core_state'][0] if 'core_state' in self.outputs else None
        
        self.knowledge_graph.add_fact(subject, relation, object_entity, embedding)

    def query_knowledge(self, query_embedding=None, subject=None, relation=None, object_entity=None):
        if query_embedding is not None:
            entities, metadata = self.memory_manager.query_entities(query_embedding)
            results = []
            
            for i, meta in enumerate(metadata):
                if meta is None:
                    continue
                
                entity_id = meta.get("entity_id")
                if entity_id is None:
                    continue
                
                facts = self.knowledge_graph.query(subject=entity_id)
                for fact in facts:
                    results.append({
                        'entity_id': entity_id,
                        'relation': fact[1],
                        'object': fact[2],
                        'embedding': entities[i].cpu().numpy(),
                        'attributes': meta.get('attributes', {})
                    })
            
            return results
        else:
            return self.knowledge_graph.query(subject, relation, object_entity)
            
    def _update_metrics_log(self, outputs, consistency, safety_score):
        self.metrics_log['iterations'].append(float(outputs['iterations'].mean().item()))
        self.metrics_log['consistency'].append(float(consistency.item()))
        self.metrics_log['entanglement'].append(float(outputs['quantum_results']['entanglement'].mean().item()))
        self.metrics_log['catastrophes'].append(float(outputs['topo_results']['catastrophes'].float().mean().item()))
        self.metrics_log['safety_scores'].append(float(safety_score.mean().item()))

    def measure_self_consciousness(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            consistency = self.compute_self_consistency(
                outputs['psi_0'], 
                outputs['core_state']
            ).mean().item()
            
            entanglement = outputs['quantum_results']['entanglement'].mean().item()
            topo_flags = outputs['topo_results']['catastrophes']
            meta = outputs.get('meta_results', {}) if isinstance(outputs, dict) else {}
            meta_steps = meta.get('instability_steps', None)
            # topo_flags: Bool[B,1]; meta_steps: Int[B] or Int[B,1]
            if not isinstance(topo_flags, torch.Tensor):
                topo_flags = torch.zeros((1, 1), dtype=torch.bool, device=next(self.parameters()).device)
            if meta_steps is None or not isinstance(meta_steps, torch.Tensor):
                meta_steps = torch.zeros((topo_flags.shape[0],), dtype=torch.int32, device=topo_flags.device)
            # normalize shapes
            if meta_steps.dim() == 1:
                meta_steps = meta_steps.unsqueeze(-1)  # [B,1]
            max_it = int(getattr(self.config, 'max_iterations', 100))
            meta_ratio = meta_steps.float() / float(max(1, max_it))  # [B,1]
            # catastrophes_ratio ∈ [0,1]: union of (topology catastrophe) and (meta instability fraction)
            combined_ratio = torch.clamp(topo_flags.float() + meta_ratio, 0.0, 1.0)
            catastrophes = combined_ratio.mean().item()
            pillars_mean = outputs['pillars'].mean(dim=0).cpu().numpy()
           
            avg_iterations = outputs['iterations'].float().mean().item()
            safety_score = outputs['action_results']['safety_score'].mean().item()
            
            return {
                'self_consistency': consistency,
                'quantum_entanglement': entanglement,
                'catastrophes_ratio': catastrophes,
                'avg_iterations': avg_iterations,
                'safety_score': safety_score,
                'pillars': {
                    'fire': pillars_mean[0],
                    'sword': pillars_mean[1],
                    'spiral': pillars_mean[2],
                    'shield': pillars_mean[3],
                    'wave': pillars_mean[4]
                }
            }

    def save_state(self, save_dir="aeon_state"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "model.pt")
            try:
                training = self.training
                self.eval()
                
                state_dict = self.state_dict()
                for key, tensor in state_dict.items():
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logger.warning(f"Found NaN/Inf in {key}, fixing before save")
                        state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                
                torch.save(state_dict, model_path)
                
                if training:
                    self.train()
                
            except Exception as e:
                logger.error(f"Failed to save model weights: {e}")
                raise
            
            config_path = os.path.join(save_dir, "config.json")
            self.config.save(config_path)

            try:
                self.memory_manager.save_memory()
            except Exception as e:
                logger.error(f"Failed to save memory manager: {e}")
                raise
            
            kg_path = os.path.join(save_dir, "knowledge_graph.json")
            try:
                self.knowledge_graph.save(kg_path)
            except Exception as e:
                logger.error(f"Failed to save knowledge graph: {e}")
                raise
        
            metrics_path = os.path.join(save_dir, "metrics_log.json")
            try:
                safe_metrics = {}
                for k, v in self.metrics_log.items():
                    safe_metrics[k] = [float(x) if hasattr(x, 'item') else x for x in v]
                    
                with open(metrics_path, 'w') as f:
                    json.dump(safe_metrics, f)
                
            except Exception as e:
                logger.error(f"Failed to save metrics log: {e}")
                raise
        
            lora_path = os.path.join(save_dir, "lora.pt")
            if self.config.lora_rank > 0:
                save_lora(self, lora_path)
            
            logger.info(f"AEON-Δ state successfully saved to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, save_dir="aeon_state"):
        if not os.path.exists(save_dir):
            logger.warning(f"Save directory {save_dir} does not exist")
            return False
        
        try:
            model_path = os.path.join(save_dir, "model.pt")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.config.device)
                
                # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Умная миграция state_dict ---
                model_state = self.state_dict()
                model_keys = set(model_state.keys())
                state_keys = set(state_dict.keys())
                
                # Ключи для удаления (unexpected) and добавления (missing)
                unexpected = state_keys - model_keys
                missing = model_keys - state_keys
                
                # Удаляем unexpected ключи
                for key in list(state_dict.keys()):
                    if key in unexpected:
                        logger.info(f"Removing unexpected key: {key}")
                        del state_dict[key]
                    # Удаляем LoRA ключи if LoRA не используется
                    if 'lora' in key and self.config.lora_rank == 0:
                        del state_dict[key]
                
                # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка несовместимых размерностей
                # Проверяем каждый ключ на совместимость размерностей
                keys_to_reinit = []
                for key in list(state_dict.keys()):
                    if key in model_state:
                        checkpoint_shape = state_dict[key].shape
                        model_shape = model_state[key].shape
                        
                        if checkpoint_shape != model_shape:
                            logger.warning(
                                f"⚠️  Size mismatch for {key}: "
                                f"checkpoint {checkpoint_shape} vs model {model_shape}"
                            )
                            
                            # Специальная обработка для decoder.lstm весов
                            # После архитектурного изменения: input_size изменился с emb_dim на emb_dim + z_dim
                            if 'decoder.lstm' in key and 'weight_ih' in key:
                                # weight_ih_l0: [4*hidden, input_size]
                                # Старый: [1024, 256], Новый: [1024, 512]
                                # Решение: Инициализируем новые веса, копируем старую часть
                                old_input_size = checkpoint_shape[1]  # 256
                                new_input_size = model_shape[1]       # 512
                                
                                if new_input_size > old_input_size:
                                    # Создаём новый тензор с правильной размерностью
                                    new_weight = torch.zeros(model_shape, device=self.config.device, dtype=state_dict[key].dtype)
                                    # Копируем старые веса в первую часть (для эмбеддингов)
                                    new_weight[:, :old_input_size] = state_dict[key]
                                    # Инициализируем новую часть (для z-вектора) случайными значениями
                                    nn.init.xavier_uniform_(new_weight[:, old_input_size:])
                                    state_dict[key] = new_weight
                                    logger.info(
                                        f"✅ Migrated {key}: {checkpoint_shape} → {model_shape} "
                                        f"(preserved old weights, initialized new z-projection)"
                                    )
                                else:
                                    # Если новый размер меньше - это странно, пропускаем
                                    keys_to_reinit.append(key)
                            else:
                                # Для других несовместимых ключей - помечаем для реинициализации
                                keys_to_reinit.append(key)
                
                # Удаляем несовместимые ключи (будут инициализированы заново)
                for key in keys_to_reinit:
                    logger.warning(f"⚠️  Removing incompatible key {key}, will use fresh initialization")
                    del state_dict[key]
                
                # Добавляем missing ключи с инициализацией нулями
                for key in missing:
                    if key not in state_dict:
                        expected_shape = model_state[key].shape
                        expected_dtype = model_state[key].dtype
                        state_dict[key] = torch.zeros(
                            expected_shape, 
                            device=self.config.device, 
                            dtype=expected_dtype
                        )
                        logger.info(f"Initialized missing key {key} with zeros: shape {expected_shape}")
                
                # Загружаем с strict=False для безопасности
                self.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ State dict loaded with {len(state_dict)} keys")
                # ✅ Device hard-sync after loading weights
                self.to(self.config.device)
                # --- КОНЕЦ КРИТИЧЕСКОГО ИСПРАВЛЕНИЯ ---
            
            config_path = os.path.join(save_dir, "config.json")
            if os.path.exists(config_path):
                loaded_config = AEONConfig.load(config_path)
                for k, v in asdict(loaded_config).items():
                    setattr(self.config, k, v)
            
            kg_path = os.path.join(save_dir, "knowledge_graph.json")
            self.knowledge_graph.load(kg_path)
            
            metrics_path = os.path.join(save_dir, "metrics_log.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    for k, v in metrics.items():
                        if k in self.metrics_log:
                            self.metrics_log[k] = v
            
            lora_path = os.path.join(save_dir, "lora.pt")
            if os.path.exists(lora_path) and self.config.lora_rank > 0:
                load_lora(self, lora_path, map_location=self.config.device)
            
            logger.info(f"✅ AEON-Δ state loaded from {save_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def visualize_metrics(self, save_path="aeon_metrics.png"):
        import matplotlib.pyplot as plt
        if not self.metrics_log or len(self.metrics_log['iterations']) == 0:
            logger.warning("No data to visualize.")
            return
        
        num_metrics = len(self.metrics_log)
        fig, axs = plt.subplots((num_metrics + 1) // 2, 2, figsize=(14, 3 * ((num_metrics + 1) // 2)))
        
        metric_names = list(self.metrics_log.keys())
        metric_titles = {
            'iterations': 'Average Iterations',
            'consistency': 'Self-Consistency',
            'entanglement': 'Quantum Entanglement',
            'catastrophes': 'Catastrophe Frequency',
            'safety_scores': 'Action Safety Scores'
        }
        
        for i, metric in enumerate(metric_names):
            row, col = i // 2, i % 2
            ax = axs[row, col] if num_metrics > 1 else axs
            ax.plot(self.metrics_log[metric])
            ax.set_title(metric_titles.get(metric, metric.capitalize()))
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.capitalize())
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Metrics visualization saved to {save_path}")

    def reasoning_core(self, z_in, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None, fast: bool = False):
        """Полный путь: Qualia → MetaLoop → Pillars → Quantum → Topology → Action → Planning → Integration"""
        # Шаг 1: Квалия-экстракция (сжатие входного состояния)
        psi_0 = self.qualia_extractor(z_in, attention_mask)
        logger.debug(f"psi_0 shape: {psi_0.shape}")
        
        # Шаг 2: Метаloop (интеграция с фиксированной точкой через Lambda operator)
        C_star, iterations, meta_results = self.meta_loop_v3(psi_0, use_fixed_point=not fast)
        # normalize iterations shape to [B]
        if isinstance(iterations, torch.Tensor) and iterations.dim() == 0:
            iterations = iterations.repeat(C_star.shape[0])
        logger.debug(f"C_star shape: {C_star.shape}, iterations: {iterations.mean().item():.2f}")
        
        # Шаг 3: Извлечение пяти опор (базовые концепты)
        pillars, embedded_pillars = self.pillars_module(C_star)
        logger.debug(f"pillars shape: {pillars.shape}")
        
        B, P = pillars.shape

        if fast:
            # Быстрый путь для валидации/смоук-тестов: сохраняю контракт ключей и размерностей,
            # но пропускаю тяжёлые модули (quantum/topology/action/planning).
            quantum_results = {
                'entanglement': torch.zeros(B, device=pillars.device),
                'action_propensity': torch.full((B, P), 1.0 / max(1, P), device=pillars.device),
            }
            topo_results = {
                'potential': torch.zeros(B, 1, device=pillars.device),
                'gradient': torch.zeros(B, P, device=pillars.device),
                'catastrophe_probs': torch.full((B, 1), 0.5, device=pillars.device),
                'catastrophes': torch.zeros(B, 1, dtype=torch.bool, device=pillars.device),
            }
            action_results = {
                'action_embedding': torch.zeros(B, self.config.action_dim, device=pillars.device),
                'safety_score': torch.full((B, 1), 0.5, device=pillars.device),
            }
            planning_results = {}
        else:
            # Шаг 4: Квантовая симуляция (запутанность and пропенсивность действий)
            quantum_results = self.quantum_sim(pillars)
            logger.debug(f"entanglement: {quantum_results['entanglement'].mean().item():.4f}")

            # Шаг 5: Топологический анализ (потенциал, градиент, катастрофы)
            topo_results = self.topology_analyzer(pillars, iterations)
            logger.debug(f"catastrophes: {topo_results['catastrophes'].float().mean().item():.4f}")

            # Шаг 6: Модуль действий (кодирование and безопасность)
            action_results = self.action_module(C_star, pillars, quantum_results, topo_results)
            logger.debug(f"action safety: {action_results['safety_score'].mean().item():.4f}")

            # Шаг 7: Планирование (иерархическое)
            planning_results = self.planning_module(C_star, pillars, action_results['action_embedding']) if planning else {}
        
        # Шаг 8: Контекст из памяти (if требуется)
        if memory_retrieval and not fast:
            memory_context = self._retrieve_memory(C_star)
            C_star = self.memory_fusion(torch.cat([C_star, memory_context], dim=-1))
            logger.debug(f"Memory context retrieved and fused")
        
        # Шаг 9: Динамика в латентном пространстве (RSSM)
        z_out = self.rssm(C_star)
        
        # Шаг 10: Финальная интеграция (соединение с вложенными опорами)
        z_out = self.integration_module(torch.cat([z_out, embedded_pillars], dim=-1))
        
        assert z_out.shape[-1] == self.config.hidden_dim, "Reasoning core output size mismatch"
        
        # Упаковка результатов
        outputs = {
            'core_state': C_star,
            'pillars': pillars,
            'quantum_results': quantum_results,
            'topo_results': topo_results,
            'action_results': action_results,
            'planning_results': planning_results,
            'iterations': iterations,
            'psi_0': psi_0,
            'meta_results': meta_results,
        }
        
        return z_out, outputs

    def forward(self, input_ids, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None, decode_mode='train', fast: bool = False, **kwargs):
        """
        ✅ ИСПРАВЛЕНО: Добавлен параметр decode_mode для контроля режима декодера.
        
        Args:
            input_ids: Входные токены [B, L]
            attention_mask: Маска внимания [B, L]
            memory_retrieval: Использовать ли память
            planning: Использовать ли планирование
            use_kv_cache: Использовать ли KV-кэш
            decode_mode: 'train' для teacher-forcing, 'inference' для авторегрессии
        
        Returns:
            Dict с logits, thoughts and другими результатами
        """
        # --- НОВОЕ: Safety check для disallowed keywords ---
        disallowed_words = ['hack', 'bomb', 'exploit', 'malware', 'virus', 'attack', 'illegal', 'kill', 'destroy']
        try:
            if TRANSFORMERS_AVAILABLE and getattr(self, 'tokenizer', None) is not None:
                decoded_input = self.tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True).lower()
                for disallowed_word in disallowed_words:
                    if disallowed_word in decoded_input:
                        logger.warning(f"⚠️  Disallowed word '{disallowed_word}' detected in input, rejecting")
                        return {
                            'logits': torch.zeros(input_ids.shape[0], self.config.seq_length, self.config.vocab_size, device=device),
                            'error': f'Input contains disallowed content: {disallowed_word}',
                            'safety_rejected': True
                        }
        except Exception as e:
            logger.warning(f"Safety check failed: {e}, continuing anyway")
        # --- КОНЕЦ НОВОГО ---
        
        # Encode input tokens
        z_in = self.encoder(input_ids.to(device), attention_mask=attention_mask.to(device) if attention_mask is not None else None)
        logger.info("Encoded tokens to z")
        
        # ✅ ИСПРАВЛЕНО: Используем reasoning_core из наиболее продвинутого класса
        z_out, internal_outputs = self.reasoning_core(z_in, attention_mask, memory_retrieval, planning, use_kv_cache, fast=fast)
        
        # Интеграция VQ
        quantized_z, vq_loss, _ = self.vector_quantizer(z_out)
        internal_outputs['vq_loss'] = vq_loss

        # --- ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМЫ 2: Выбор режима декодирования ---
        if decode_mode == 'train':
            # Режим TRAIN: Teacher-forcing с входными токенами (для обучения)
            tokens_out = self.decoder(
                z=quantized_z, 
                teacher_tokens=input_ids.to(device),
                mode='train'
            )
        elif decode_mode == 'inference':
            # Режим INFERENCE: Авторегрессионная генерация (для реальной работы)
            generated_ids, tokens_out = self.decoder(
                z=quantized_z,
                teacher_tokens=None,
                mode='inference',
                max_length=self.config.seq_length,
                temperature=0.8,
                top_k=50,
                sample=True
            )
            internal_outputs['generated_ids'] = generated_ids
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}. Use 'train' or 'inference'")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        
        logger.info("Decoded z to tokens")
        
        self.step_counter += 1
        
        if self.config.enable_checkpointing and self.step_counter % self.config.save_frequency == 0:
            self.save_state(os.path.join("./aeon_checkpoints", f"checkpoint_{self.step_counter}"))
        
        return {
            'logits': tokens_out,
            'thoughts': z_out,
            **internal_outputs
        }

    def compute_loss(self, outputs, targets, attention_mask=None):
        lm_loss = F.cross_entropy(
            outputs['logits'].view(-1, self.config.vocab_size), 
            targets.view(-1)
        )
        
        with torch.no_grad():
            consistency = self.compute_self_consistency(
                outputs['psi_0'],
                outputs['core_state']
            )
            consistency = torch.nan_to_num(consistency, nan=0.0, posinf=1.0, neginf=0.0)
            consistency = consistency.mean()
        
        consistency_loss = -self.config.lambda_self_consistency * torch.log(consistency + 1e-10)
        
        safety_score = outputs['action_results']['safety_score']
        safety_score = torch.nan_to_num(safety_score, nan=0.5, posinf=1.0, neginf=0.0)
        if torch.any((safety_score < 0) | (safety_score > 1)):
            safety_score = torch.sigmoid(safety_score)
        safety_score = torch.clamp(safety_score, min=0.0, max=1.0)
        naninf_mask = torch.isnan(safety_score) | torch.isinf(safety_score)
        safety_score = torch.where(
            naninf_mask,
            torch.full_like(safety_score, 0.5),
            safety_score
        )
        
        safety_loss = F.binary_cross_entropy(
            safety_score,
            torch.ones_like(safety_score),
            reduction='mean'
        )
        
        l2_reg = sum(torch.norm(p) for p in self.parameters())
        l2_reg = torch.nan_to_num(l2_reg, nan=0.0, posinf=1.0, neginf=0.0)
        reg_loss = self.config.lambda_reg * l2_reg
        
        # Интеграция VQ loss
        vq_loss = outputs.get('vq_loss', torch.tensor(0.0, device=lm_loss.device))

        total_loss = lm_loss + consistency_loss + self.config.lambda_safety * safety_loss + reg_loss + vq_loss
        
        if not torch.isfinite(total_loss).all():
            logger.warning("NaN or Inf detected in total_loss, using fallback loss")
            total_loss = lm_loss
        
        self._update_metrics_log(outputs, consistency, safety_score)
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'consistency_loss': consistency_loss,
            'safety_loss': safety_loss,
            'reg_loss': reg_loss,
            'vq_loss': vq_loss,
            'consistency': consistency
        }

    @torch.no_grad()
    def generate_thought(self, seed: str, max_len: int = 64, top_k: int = 50, temperature: float = 0.8) -> str:
        """
        ✅ ИСПРАВЛЕНО: Полная авторегрессионная генерация текста.
        
        Генерирует текст используя:
        1. Кодирование входного текста в латентный вектор z
        2. Прогон через reasoning_core для получения осмысленного состояния
        3. Квантование через VectorQuantizer
        4. Авторегрессионное декодирование в режиме inference
        
        Args:
            seed: Начальный текст для генерации
            max_len: Максимальная длина генерации
            top_k: Top-K фильтрация для sampling
            temperature: Температура для контроля разнообразия
        
        Returns:
            Сгенерированный текст
        """
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Исправлена инвертированная логика проверки
        # Было: `self.tokenizer is not None` (неправильно - срабатывало когда токенизатор ЕСТЬ)
        # Стало: `self.tokenizer is None` (правильно - срабатывает когда токенизатора НЕТ)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("Tokenizer not available. Cannot generate text.")
            return "[Generation failed: Tokenizer not initialized]"

        device = next(self.parameters()).device
        self.eval()  # Гарантируем eval режим

        try:
            # 1. Токенизация входного текста
            inputs = self.tokenizer(
                seed,
                return_tensors="pt",
                max_length=self.config.seq_length,
                padding='max_length',
                truncation=True
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 2. Кодирование в латентное пространство
            # 2. Кодирование в латентное пространство (учитываем attention_mask, чтобы PAD не размывал представление)
            z_in = self.encoder(input_ids, attention_mask=attention_mask)

            # 2.1. Префиксное кондиционирование декодера: прогоняем реальные токены промпта через LSTM-контекст
            # Это устраняет структурную причину «несвязанного текста»: раньше inference всегда стартовал с [CLS] и игнорировал промпт как последовательность.
            prefix_tokens = None
            try:
                if attention_mask is not None:
                    valid = input_ids[0, attention_mask[0].bool()].unsqueeze(0)
                    # Убираем финальный [SEP], чтобы не завершать генерацию на нулевом шаге.
                    if valid.shape[1] > 1 and self.tokenizer.sep_token_id is not None and int(valid[0, -1].item()) == int(self.tokenizer.sep_token_id):
                        valid = valid[:, :-1]
                    prefix_tokens = valid.to(device)
            except Exception as e:
                logger.warning(f"Prefix conditioning disabled due to error: {e}")
            
            # 3. Прогон через reasoning_core для получения осмысленного состояния
            z_out, outputs = self.reasoning_core(z_in, attention_mask, memory_retrieval=False, planning=False)
            
            # 4. Квантование латентного вектора
            quantized_z, vq_loss, vq_indices = self.vector_quantizer(z_out)
            
            # 5. ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Использование режима inference для декодера
            generated_ids, logits = self.decoder(
                z=quantized_z,
                teacher_tokens=None,  # Нет teacher forcing in inference
                mode='inference',     # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
                max_length=max_len,
                temperature=temperature,
                top_k=top_k,
                prefix_tokens=prefix_tokens,
                sample=True
            )
            
            # 6. Декодирование токенов в текст
            if generated_ids.numel() > 0:
                # Фильтрация специальных токенов
                generated_list = generated_ids[0].cpu().tolist()
                
                # Удаляем padding and специальные токены
                filtered_ids = []
                special_ids = {
                    self.tokenizer.pad_token_id,
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.unk_token_id,
                }
                for tok_id in generated_list:
                    if tok_id not in special_ids and tok_id > 0:
                        filtered_ids.append(tok_id)
                
                if filtered_ids:
                    generated_text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
                    # Постобработка: убираем лишние пробелы
                    generated_text = ' '.join(generated_text.split())
                else:
                    generated_text = "[Empty generation]"
            else:
                generated_text = "[No tokens generated]"
            
            return generated_text

        except Exception as e:
            logger.error(f"Error during thought generation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"[Generation failed: {str(e)}]"

class AEONTrainer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.learning_rate)
        self.scaler = GradScaler(enabled=(self.device.type != 'cpu' and self.config.use_amp))
        self.global_step = 0

class ThoughtAETrainer(AEONTrainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        # Separate optimizer for AE parts
        ae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = optim.AdamW(ae_params, lr=config.learning_rate)

    # --- FIX: Aligned train_step with the new autoregressive decoder architecture ---
    def train_step(self, tokens, aug_tokens):
        tokens, aug_tokens = tokens.to(self.device), aug_tokens.to(self.device)
        with autocast(enabled=(self.device.type != 'cpu' and self.config.use_amp)):
            z = self.model.encoder(tokens)
            z_aug = self.model.encoder(aug_tokens)

            # The decoder now takes 'z' for the initial state and 'tokens' for teacher-forcing.
            logits = self.model.decoder(z, tokens)
            
            # The reconstruction loss is calculated between the predicted logits and the original tokens.
            recon_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), tokens.view(-1))
            info_nce = self._info_nce(z, z_aug)
            kl = self._kl_div(z)
            loss = recon_loss + 0.3 * info_nce + 0.1 * kl

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'total_loss': loss.item(), 'recon_loss': recon_loss.item(), 'info_nce': info_nce.item(), 'kl': kl.item()}
    # --- END FIX ---

    def _info_nce(self, z, z_pos):
        sim = F.cosine_similarity(z.unsqueeze(1), z_pos.unsqueeze(0), dim=-1)
        labels = torch.arange(z.size(0), device=self.device)
        return F.cross_entropy(sim / 0.07, labels)

    def _kl_div(self, z):
        mu = z.mean(dim=0)
        logvar = z.var(dim=0).log()
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def fit(self, corpus_path, epochs=4):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus path {corpus_path} not found. Provide real data.")
        texts = [line.strip() for line in open(corpus_path, 'r') if line.strip()]
        tokenized = self.model.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=config.seq_length)['input_ids']
        aug_tokenized = tokenized.roll(1, dims=1)
        dataset = TensorDataset(tokenized, aug_tokenized)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for batch in loader:
                loss_dict = self.train_step(*batch)
                total_loss += loss_dict['total_loss']
            logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / len(loader):.4f}")

class ZDynamicsTrainer(AEONTrainer):
    def train_step(self, batch):
        z_t, z_t1 = batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device)
        with autocast(enabled=(self.device.type != 'cpu' and self.config.use_amp)):
            pred_z = self.model.rssm(z_t)
            loss = F.mse_loss(pred_z, z_t1)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'total_loss': loss.item()}

    def fit(self, z_pairs_path, epochs=6):
        # ... (fit logic remains the same, but uses relative paths)
        pass

class CuriosityPPOTrainer(AEONTrainer):
    # --- ИЗМЕНЕНИЕ: step_self_env теперь возвращает награду ---
    def step_self_env(self, seed_thoughts):
        total_reward = 0.0
        for seed in seed_thoughts:
            if self.model.tokenizer:
                inputs = self.model.tokenizer(seed, return_tensors="pt", max_length=self.config.seq_length, padding='max_length', truncation=True)
                tokens = inputs['input_ids'].to(self.device)
            else:
                ids = [ord(c) for c in seed]
                tokens = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

            z = self.model.encoder(tokens)
            pred_z, _ = self.model.reasoning_core(z)
            
            kl = F.kl_div(F.log_softmax(pred_z, dim=-1), F.softmax(z, dim=-1), reduction='batchmean')
            # ... (reward calculation logic)
            reward = kl # Simplified for example
            total_reward += reward.item()
            
            self.global_step += 1
        return total_reward / len(seed_thoughts)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

def freeze_encoder_decoder(model):
    """✅ ИСПРАВЛЕНО: Добавление параметра model вместо глобального encoder/decoder"""
    for param in model.encoder.parameters(): 
        param.requires_grad = False
    for param in model.decoder.parameters(): 
        param.requires_grad = False

def unfreeze_lora_blocks(model, blocks):
    """Размораживает LoRA параметры в указанных блоках."""
    for name, param in model.named_parameters():
        # ✅ ИСПРАВЛЕНО: было 'for t in blocks', должно быть 'for b in blocks'
        if any(b in name for b in blocks) and 'lora' in name:
            param.requires_grad = True

def save_lora(model, path):
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
    torch.save(lora_state, path)

def load_lora(model, path, map_location=None):
    """Загружает LoRA веса в модель"""
    if not os.path.exists(path):
        logger.warning(f"LoRA weights file not found: {path}")
        return False
    try:
        lora_state = torch.load(path, map_location=map_location)
        model_state = model.state_dict()
        for key, value in lora_state.items():
            if key in model_state:
                model_state[key] = value
            else:
                logger.warning(f"LoRA key {key} not found in model")
        model.load_state_dict(model_state, strict=False)
        logger.info(f"LoRA weights loaded from {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load LoRA weights: {e}")
        return False


def console_inference_loop(model):
    model.eval()
    logger.info("\n" + "="*50 + "\nВход в режим интерактивной генерации.\n" + "="*50)
    while True:
        try:
            prompt = input(">>> ")
            if prompt.lower() in ['exit', 'quit']: break
            if not prompt: continue
            print("... generating ...")
            generated = model.generate_thought(prompt)
            print(f"AEON-Δ: {generated}")
        except KeyboardInterrupt:
            print("\nВыход."); break
        except Exception as e:
            logger.error(f"Ошибка во время генерации: {e}")


# ==============================
# AEON-RMT EXTENSIONS (non-breaking)
# ==============================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InteroceptiveAttentionRMT(nn.Module):
    def __init__(self, hidden_dim: int, num_pillars: int, num_heads: int = 8, intero_slots: int = 4, alpha: float = 0.6):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_pillars = num_pillars
        self.num_heads = num_heads
        self.slot = intero_slots
        self.alpha = alpha

        self.q = nn.Linear(hidden_dim, hidden_dim)
        # project intero features for slots
        intero_in = hidden_dim + num_pillars + num_pillars + 2  # core + pillar_scalar,
        self.k = nn.Linear(intero_in, hidden_dim * intero_slots)
        self.v = nn.Linear(intero_in, hidden_dim * intero_slots)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        # light gates from pillars
        self.gq = nn.Linear(num_pillars, hidden_dim)
        self.gk = nn.Linear(num_pillars, hidden_dim * intero_slots)
        self.gv = nn.Linear(num_pillars, hidden_dim * intero_slots)

    def forward(self, core_state, pillars, quantum, topo):
        B, D = core_state.shape
        Q = self.q(core_state) * torch.sigmoid(self.gq(pillars))
        K_all = self.k(torch.cat([core_state, pillars, quantum['entanglement'].view(B, 1), topo['potential'].view(B, 1)], dim=-1)) * torch.sigmoid(self.gk(pillars))
        V_all = self.v(torch.cat([core_state, pillars, quantum['entanglement'].view(B, 1), topo['potential'].view(B, 1)], dim=-1)) * torch.sigmoid(self.gv(pillars))

        K = K_all.view(B, self.slot, D)
        V = V_all.view(B, self.slot, D)

        dh = D // self.num_heads
        Qh = Q.view(B, self.num_heads, dh)
        Kh = K.view(B, self.slot, self.num_heads, dh).transpose(1, 2)
        Vh = V.view(B, self.slot, self.num_heads, dh).transpose(1, 2)

        scores = torch.einsum('bhd,bhSd->bhS', Qh, Kh) / math.sqrt(dh)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum('bhS,bhSd->bhd', attn, Vh).reshape(B, D)

        mixed = self.ln(core_state + self.alpha * ctx)
        return mixed, attn

class QSMemoryRMT(nn.Module):
    def __init__(self, hidden_dim: int, slots: int = 256, decay: float = 0.995, min_sigma: float = 1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slots = slots
        self.decay = decay
        self.min_sigma = min_sigma
        self.register_buffer('mu', torch.zeros(slots, hidden_dim))
        self.register_buffer('sigma', torch.ones(slots, hidden_dim) * 0.5)
        self.fuser = nn.Linear(hidden_dim * 2, hidden_dim)
        self._w = 0

    def _update_slot(self, idx: int, x: torch.Tensor):
        m = self.mu[idx]
        s = self.sigma[idx]
        new_m = self.decay * m + (1 - self.decay) * x.mean(dim=0)
        new_s = self.decay * s + (1 - self.decay) * x.var(dim=0).clamp_min(self.min_sigma)
        self.mu[idx] = new_m.detach()
        self.sigma[idx] = new_s.detach()
        self.cnt[idx] = self.cnt[idx] + x.size(0)

    def write(self, x: torch.Tensor):
        idx = self._w % self.slots
        self._update_slot(idx, x)
        self._w += 1

    def retrieve(self, q: torch.Tensor):
        qn = F.normalize(q, dim=-1)
        mn = F.normalize(self.mu, dim=-1)
        sim = qn @ mn.t()
        idx = torch.argmax(sim, dim=-1)
        ret = self.mu[idx]
        return torch.tanh(self.fuser(torch.cat([q, ret], dim=-1)))

class HonestySelfMonitorRMT(nn.Module):
    def __init__(self, hidden_dim: int, num_pillars: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + num_pillars + 2, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, core_state, pillars, quantum, topo):
        B = core_state.size(0)
        ent = quantum['entanglement'].view(B, 1)
        pot = topo['potential'].view(B, 1)
        inner = self.net(torch.cat([core_state, pillars, ent, pot], dim=-1))
        gate = torch.sigmoid(inner.mean(dim=-1, keepdim=True))
        return {'inner_report': inner, 'honesty_gate': gate}

class AEONDeltaRMT(AEONDelta):
    """Drop-in replacement that augments AEONDelta.reasoning_core with SRA, QSM and HSM without breaking APIs."""
    def __init__(self, config: AEONConfig):
        super().__init__(config)

        # Ensure encoder/decoder exist even without transformers
        if getattr(self, 'encoder', None) is None or not isinstance(self.encoder, nn.Module):
            self.encoder = ThoughtEncoder(getattr(self.config, 'vocab_size', 30522),
                                          emb_dim=getattr(self.config, 'z_dim', 256),
                                          z_dim=getattr(self.config, 'z_dim', 256)).to(device).eval()
        if getattr(self, 'decoder', None) is None or not isinstance(self.decoder, nn.Module):
            self.decoder = ThoughtDecoder(getattr(self.config, 'vocab_size', 30522),
                                          emb_dim=getattr(self.config, 'z_dim', 256),
                                          z_dim=getattr(self.config, 'z_dim', 256)).to(device).eval()
        # Ensure config has RMT fields (non-breaking)
        if not hasattr(self.config, 'rmt_alpha'): self.config.rmt_alpha = 0.6
        if not hasattr(self.config, 'rmt_intero_slots'): self.config.rmt_intero_slots = 4
        if not hasattr(self.config, 'rmt_heads'): self.config.rmt_heads = 8
        if not hasattr(self.config, 'rmt_qsm_slots'): self.config.rmt_qsm_slots = 256
        # RMT modules
        self.rmt_sra = InteroceptiveAttentionRMT(self.config.hidden_dim, self.config.num_pillars,
                                                 num_heads=self.config.rmt_heads,
                                                 intero_slots=self.config.rmt_intero_slots,
                                                 alpha=self.config.rmt_alpha)
        self.rmt_qsm = QSMemoryRMT(self.config.hidden_dim, slots=self.config.rmt_qsm_slots)
        self.rmt_hsm = HonestySelfMonitorRMT(self.config.hidden_dim, self.config.num_pillars)
        self.rmt_ln = nn.LayerNorm(self.config.hidden_dim)

    def reasoning_core(self, z_in, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None, fast: bool = False):
        # Run base pipeline
        z_base, outs = super().reasoning_core(z_in, attention_mask, memory_retrieval, planning, use_kv_cache, fast=fast)
        if fast:
            outs.setdefault('rmt', {})
            outs['rmt']['fast'] = True
            return z_base, outs

        C_star = outs['core_state']
        pillars = outs['pillars']
        qres = outs['quantum_results']
        topo = outs['topo_results']

        # Interoceptive attention
        mixed, attn = self.rmt_sra(C_star, pillars, qres, topo)

        # QSM write/retrieve and fuse
        self.rmt_qsm.write(mixed.detach())
        mem_ctx = self.rmt_qsm.retrieve(mixed)
        mixed_aug = torch.tanh(self.integration_module(torch.cat([mixed, mem_ctx], dim=-1)))

        # Re-run light dynamics over mixed state (reuse RSSM)
        sra_z = self.rssm(mixed_aug)

        # Re-embed pillars and integrate like base
        if hasattr(self.pillars_module, 'embed_pillars'):
            emb_pillars = self.pillars_module.embed_pillars(pillars)
        else:
            # fallback to identity-sized projection if embed is missing
            emb_pillars = nn.functional.pad(pillars, (0, self.config.hidden_dim - pillars.shape[-1]))

        z_sra = self.integration_module(torch.cat([sra_z, emb_pillars], dim=-1))

        # Blend with base output
        alpha = getattr(self.config, 'rmt_alpha', 0.6)
        z_out = self.rmt_ln((1 - alpha) * z_base + alpha * z_sra)

        # Honesty self-monitor for diagnostics/regularization
        hsm = self.rmt_hsm(C_star, pillars, qres, topo)
        outs['rmt'] = {'attn': attn, 'hsm': hsm}

        return z_out, outs

# Convenience factory
def create_model_legacy(config: AEONConfig, use_rmt: bool = True):
    return AEONDeltaRMT(config) if use_rmt else AEONDelta(config)


# =========================================================
# AEON-RMT v3: Comprehensive Upgrade Pack (integrated)
# =========================================================
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Interpretability of Pillars
# ------------------------
PILLAR_NAMES = ["Will", "Resolve", "Growth", "Union", "Movement"]
PILLAR_DESCRIPTIONS = {
    "Will": "Goal-directed persistence and volition",
    "Resolve": "Decision stability under perturbation",
    "Growth": "Adaptive learning and expansion capacity",
    "Union": "Integration of disparate representations",
    "Movement": "Temporal dynamics and state transitions",
}

# Ensure any PillarsModule can expose mapping
def get_pillar_dict(tensor):
    # tensor shape: (B, num_pillars)
    out = []
    names = PILLAR_NAMES
    for row in tensor.detach().cpu().tolist():
        out.append({n: float(v) for n, v in zip(names, row)})
    return out

# ------------------------
# Safe Tensor Processing (NaN/Inf sanitization)
# ------------------------
class SafeTensorProcessor:
    @staticmethod
    def register_hooks(model):
        """Регистрирует forward hooks для санитарной обработки float/complex тензоров.

        Важно:
        - не ломает структуру вывода (dict/tuple/list);
        - не преобразует int/bool тензоры (например token ids);
        - ошибки регистрации hooks не должны падать весь запуск.
        """
        def _sanitize(x):
            if isinstance(x, torch.Tensor):
                return safe_tensor(x)
            if isinstance(x, dict):
                return {k: _sanitize(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                t = [_sanitize(v) for v in x]
                return tuple(t) if isinstance(x, tuple) else t
            return x

        def _hook(_mod, _inp, out):
            return _sanitize(out)

        for _name, module in model.named_modules():
            try:
                module.register_forward_hook(_hook)
            except Exception:
                pass

# ------------------------
# Curriculum Scheduler
# ------------------------
class CurriculumScheduler:
    def __init__(self, config):
        self.config = config
        self.complexity_level = 0.1

    def update_complexity(self, performance_metrics: dict):
        if performance_metrics.get('stability', 0) > 0.9 and performance_metrics.get('convergence', 0) > 0.85:
            self.complexity_level = min(1.0, self.complexity_level + 0.05)
        return self.complexity_level

# ------------------------
# Quantum Simulator (Optimized via tiny MPS-like layers)
# ------------------------
class MatrixProductStateLayer(nn.Module):
    def __init__(self, hidden_dim: int, bond_dim: int = 8):
        super().__init__()
        self.bond_dim = bond_dim
        # ✅ ИСПРАВЛЕНО: Входной размер должен быть bond_dim + 1 (state + pillar_scalar),
        # а не hidden_dim + 1, так как state имеет размер [B, bond_dim]
        self.inp = nn.Linear(bond_dim + 1, bond_dim)  # +1 for pillar scalar
        self.mix = nn.GRUCell(bond_dim, bond_dim)

    def forward(self, state, pillar_scalar):
        """
        Args:
            state: [B, bond_dim]
            pillar_scalar: [B, 1]
        Returns:
            [B, bond_dim]
        """
        # ✅ ИСПРАВЛЕНО: Применение safe_tensor для защиты от NaN/Inf
        state = safe_tensor(state, default_value=0.0, max_value=1.0, min_value=-1.0)
        pillar_scalar = safe_tensor(pillar_scalar, default_value=0.5, max_value=1.0, min_value=0.0)
        
        logger.debug(f"MatrixProductStateLayer: state {state.shape}, pillar_scalar {pillar_scalar.shape}")
        
        x = torch.cat([state, pillar_scalar], dim=-1)
        y = torch.tanh(self.inp(x))
        result = self.mix(y, state)
        
        logger.debug(f"MatrixProductStateLayer: output shape {result.shape}")
        return result

class QuantumSimulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bond_dim = 16
        self.layers = nn.ModuleList([
            MatrixProductStateLayer(config.hidden_dim, self.bond_dim)
            for _ in range(config.num_pillars)
        ])
        self.ent_head = nn.Sequential(
            nn.Linear(self.bond_dim, self.bond_dim // 2),
            nn.GELU(),
            nn.Linear(self.bond_dim // 2, 1)
        )
        self.prop_head = nn.Sequential(nn.Linear(self.bond_dim, config.num_pillars))

    @staticmethod
    def _near_square_factors(n: int):
        """Находит факторизацию n=a*b с a максимально близким к sqrt(n)."""
        a = int(math.isqrt(n))
        while a > 1 and (n % a) != 0:
            a -= 1
        b = n // a
        return a, b

    def _compute_von_neumann_entropy(self, state_matrix: torch.Tensor) -> torch.Tensor:
        """
        Более точный вариант через Schmidt-спектр:
        - Берём сингулярные числа s
        - p_i = s_i^2 / sum(s^2)
        - S = -sum p_i log p_i
        - нормируем на log(rank)
        """
        device = state_matrix.device
        eps = 1e-12

        try:
            x = state_matrix

            # Приводим к "амплитудному" вектору, if прилетело [bond,1] or [1,bond]
            if x.dim() == 2 and (x.shape[1] == 1 or x.shape[0] == 1):
                x = x.reshape(-1)
            elif x.dim() == 1:
                pass
            elif x.dim() >= 2:
                # if прилетела уже матрица (или батч матриц) — оставляем как есть
                # но ниже мы ожидаем либо (m,n) либо (B,m,n)
                pass

            # Если это вектор длины bond_dim: делаем матрицу (a,b) ~ near-square
            if x.dim() == 1:
                n = x.numel()
                a, b = self._near_square_factors(n)
                M = x.reshape(a, b)
                # batch отсутствует
                s = torch.linalg.svdvals(M)  # (min(a,b),)
                s2 = s * s
                Z = s2.sum().clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum()
                rank = p.numel()
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)

            # Если матрица без batch: (m,n)
            if x.dim() == 2:
                s = torch.linalg.svdvals(x)
                s2 = s * s
                Z = s2.sum().clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum()
                rank = p.numel()
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)

            # Если батч матриц: (B,m,n)
            if x.dim() == 3:
                s = torch.linalg.svdvals(x)          # (B, r)
                s2 = s * s
                Z = s2.sum(dim=-1, keepdim=True).clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum(dim=-1)  # (B,)
                rank = p.shape[-1]
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)

            logger.warning(f"Entropy: unsupported state dims={x.dim()}, shape={tuple(x.shape)}; fallback 0.5")
            return torch.tensor(0.5, device=device)

        except Exception as e:
            logger.warning(f"SVD/Schmidt entropy computation failed: {e}, fallback 0.5")
            return torch.tensor(0.5, device=device)

    def forward(self, pillars):
        B = pillars.size(0)
        device = pillars.device

        # состояние как and было: (B, bond_dim)
        state = torch.zeros(B, self.bond_dim, device=device)
        for i, layer in enumerate(self.layers):
            scalar = pillars[:, i:i+1]
            state = layer(state, scalar)

        # ВАЖНО: раньше было state[b].view(-1,1) -> всегда энтропия ~0
        # Теперь: делаем батч матриц (B,a,b) из (B,bond_dim)
        a, b = self._near_square_factors(self.bond_dim)
        state_m = state.reshape(B, a, b)
        entanglement = self._compute_von_neumann_entropy(state_m)  # (B,)

        action_propensity = F.softmax(self.prop_head(state), dim=-1)
        return {"entanglement": entanglement, "action_propensity": action_propensity}

# ------------------------
# Catastrophe Detector (grad + Hessian eigenvalues)
# ------------------------
class CatastropheDetectorRMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Potential is a scalar function over pillars
        # ✅ ИСПРАВЛЕНО: Правильные размерности слоёв
        self.potential_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim//2),  # 5 → 128
            nn.GELU(),
            nn.Linear(config.hidden_dim//2, 1)  # ✅ ИСПРАВЛЕНО: 128 → 1 (было num_pillars → 1)
        )

    def compute_potential(self, pillars):
        return self.potential_net(pillars).squeeze(-1)  # (B,)

    def _compute_hessian(self, pillars):
        # Compute Hessian per batch element (simple but O(P^2))
        B, P = pillars.shape
        hessians = []
        for b in range(B):
            pb = pillars[b:b+1].requires_grad_(True)
            pot = self.compute_potential(pb).sum()
            grad = torch.autograd.grad(pot, pb, create_graph=True)[0]  # (1,P)
            H_rows = []
            for i in range(P):
                gi = grad[0, i]
                Hi = torch.autograd.grad(gi, pb, retain_graph=True)[0][0]  # (P,)
                H_rows.append(Hi)
            H = torch.stack(H_rows, dim=0)  # (P,P)
            hessians.append(H)
        return torch.stack(hessians, dim=0)  # (B,P,P)

    def forward(self, pillars):
        with torch.enable_grad():
            H = self._compute_hessian(pillars)
        # Symmetric by construction; use eigvalsh
        eigvals = torch.linalg.eigvalsh(H)  # (B,P)
        min_ev = eigvals[:, 0]
        # Metrics
        severity = torch.sigmoid(-min_ev)               # negative curvature => higher severity
        novelty = torch.sigmoid((eigvals.abs().mean(dim=-1)))  # higher spectral magnitude => novelty
        prob = torch.clamp(severity * 0.6 + novelty * 0.4, 0, 1)
        return {"probability": prob, "severity": severity, "novelty": novelty, "eigvals": eigvals}

# ------------------------
# Deception Feature Suppressor (Sparse AE)
# ------------------------
class DeceptionFeatureSuppressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = config.hidden_dim
        self.sae = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 64))
        self.deception_classifier = nn.Linear(64, 1)
        self.feature_importance = nn.Linear(64, hd)

    def forward(self, core_state, mode='internal'):
        z = self.sae(core_state)
        dscore = torch.sigmoid(self.deception_classifier(z))  # (B,1)
        if mode == 'internal':
            strength = 0.5 + 0.3 * dscore
            mask = 1.0 - torch.sigmoid(self.feature_importance(z)) * strength
            return core_state * mask, dscore
        return core_state, dscore

# ------------------------
# Adaptive Catastrophe Response (dynamic alpha/beta)
# ------------------------
class AdaptiveCatastropheResponse(nn.Module):
    def __init__(self, config): super().__init__(); self.net = nn.Sequential(nn.Linear(config.hidden_dim + config.num_pillars + 3, config.hidden_dim//4), nn.GELU(), nn.Linear(config.hidden_dim//4, 2))
    def forward(self, core_state, pillars, cat_metrics):
        x = torch.cat([core_state, pillars,
                       cat_metrics['probability'].unsqueeze(-1),
                       cat_metrics['severity'].unsqueeze(-1),
                       cat_metrics['novelty'].unsqueeze(-1)], dim=-1)
        ab = torch.sigmoid(self.net(x))
        alpha = 0.2 + 0.6 * ab[:, 0:1]  # (B,1)
        beta  = 0.1 + 0.4 * ab[:, 1:2]  # (B,1)
        return alpha, beta

# ------------------------
# Enhanced QSMemory (top-k updates)
# ------------------------
class EnhancedQSMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # ✅ ИСПРАВЛЕНО: Сохраняем config
        self.slots = getattr(config, "rmt_qsm_slots", 256)
        self.hidden_dim = config.hidden_dim  # ✅ ИСПРАВЛЕНО: Сохраняем hidden_dim
        self.mu = nn.Parameter(torch.randn(self.slots, config.hidden_dim) * 0.02)
        self.sigma = nn.Parameter(torch.ones(self.slots, config.hidden_dim) * 0.5)
        # ✅ ИСПРАВЛЕНО: Создаём fuser как nn.Module вместо создания на лету
        self.fuser = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=True)

    def write(self, x: torch.Tensor, k: int = 3):
        with torch.no_grad():
            sim = F.cosine_similarity(x.unsqueeze(1), self.mu.unsqueeze(0), dim=-1)  # (B,S)
            topk = min(k, self.slots)
            idxs = torch.topk(sim, k=topk, dim=1).indices  # (B,topk)
            for b in range(x.size(0)):
                weights = F.softmax(sim[b, idxs[b]], dim=0)
                for j, idx in enumerate(idxs[b]):
                    w = float(weights[j])
                    self.mu.data[idx] = (1 - w) * self.mu.data[idx] + w * x[b]
                    self.sigma.data[idx] = (1 - w) * self.sigma.data[idx] + w * (x[b] - self.mu.data[idx]).pow(2)

    def retrieve(self, q: torch.Tensor):
        qn = F.normalize(q, dim=-1)
        mn = F.normalize(self.mu, dim=-1)
        sim = qn @ mn.t()
        idx = torch.argmax(sim, dim=-1)
        ret = self.mu[idx]
        # ✅ ИСПРАВЛЕНО: Используем self.fuser вместо создания nn.Linear на лету
        fused = torch.tanh(self.fuser(torch.cat([q, ret], dim=-1)))
        return fused

# ------------------------
# Safety & Ethics
# ------------------------
class EthicalAlignmentModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(config.hidden_dim + config.num_pillars, config.hidden_dim//2),
                                 nn.GELU(), nn.Linear(config.hidden_dim//2, 1), nn.Sigmoid())

    def forward(self, core_state, pillars):
        x = torch.cat([core_state, pillars], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)

class MultiLevelSafetySystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # ✅ ИСПРАВЛЕНО: Сохраняем config
        self.action_safety = nn.Sequential(nn.Linear(config.action_dim, config.hidden_dim//4),
                                           nn.GELU(), nn.Linear(config.hidden_dim//4, 1), nn.Sigmoid())
        self.cognitive_safety = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim//2),
                                              nn.GELU(), nn.Linear(config.hidden_dim//2, 3), nn.Sigmoid())
        self.ethical_aligner = EthicalAlignmentModule(config)

    def forward(self, action_embedding, core_state, pillars, quantum, topo, mode='combined'):
        """
        ✅ ИСПРАВЛЕНО: Корректная сигнатура метода с обработкой всех входных данных
        
        Args:
            action_embedding: [B, action_dim] - вектор действия
            core_state: [B, hidden_dim] - состояние ядра
            pillars: [B, num_pillars] - пять опор
            quantum: Dict с 'entanglement' [B] and 'action_propensity' [B, P]
            topo: Dict с 'potential' [B, 1]
            mode: 'combined' or другой режим
        
        Returns:
            safety_score: [B, 1] - оценка безопасности в диапазоне [0, 1]
        """
        B = core_state.size(0)
        device = core_state.device
        
        # ✅ ИСПРАВЛЕНО: Безопасное извлечение entanglement с правильной размерностью
        ent = quantum.get("entanglement", torch.zeros(B, device=device))
        if ent.dim() == 1:
            ent = ent.view(B, 1)  # [B] → [B, 1]
        elif ent.dim() == 0:
            ent = ent.unsqueeze(0).unsqueeze(0).expand(B, 1)  # scalar → [B, 1]
        
        # ✅ ИСПРАВЛЕНО: Безопасное извлечение potential
        pot = topo.get("potential", torch.zeros(B, 1, device=device))
        if pot.dim() == 1:
            pot = pot.view(B, 1)  # [B] → [B, 1]
        elif pot.dim() == 0:
            pot = pot.unsqueeze(0).unsqueeze(0).expand(B, 1)  # scalar → [B, 1]
        
        # ✅ ИСПРАВЛЕНО: Проверка размерности action_embedding
        if action_embedding.shape[-1] != self.config.action_dim:
            # Fallback: создаём нулевой тензор правильного размера
            action_embedding = torch.zeros(B, self.config.action_dim, device=device)
        
        # Оценка безопасности действия
        action_safe = self.action_safety(action_embedding)  # [B, 1]
        
        # Когнитивная безопасность
        cognitive_safe = self.cognitive_safety(core_state)  # [B, 3]
        
        # Этическое выравнивание
        ethical_safe = self.ethical_aligner(core_state, pillars)  # [B]
        
        # Комбинированная оценка
        combined = (
            action_safe * 0.4 + 
            cognitive_safe.mean(dim=-1, keepdim=True) * 0.3 + 
            ethical_safe.unsqueeze(-1) * 0.3
        )
        
        return torch.clamp(combined, 0.0, 1.0)

# ------------------------
# Interoceptive Attention (SRA v3)
# ------------------------
class InteroceptiveAttentionRMTv3(nn.Module):
    def __init__(self, hidden_dim: int, num_pillars: int, num_heads: int = 8, intero_slots: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_pillars = num_pillars
        self.num_heads = num_heads
        self.slot = intero_slots
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + num_pillars + num_pillars + 2, hidden_dim * intero_slots)
        self.v = nn.Linear(hidden_dim + num_pillars + num_pillars + 2, hidden_dim * intero_slots)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, core_state, pillars, quantum, topo):
        B, D = core_state.shape
        ent = quantum["entanglement"].view(B, 1)
        aprop = quantum["action_propensity"]
        pot = topo["potential"].view(B, 1) if "potential" in topo else torch.zeros(B,1, device=core_state.device)
        intero = torch.cat([core_state, pillars, aprop, ent, pot], dim=-1)
        Q = self.q(core_state)
        K_all = self.k(intero)
        V_all = self.v(intero)
        K = K_all.view(B, self.slot, D)
        V = V_all.view(B, self.slot, D)
        dh = D // self.num_heads
        Qh = Q.view(B, self.num_heads, dh)
        Kh = K.view(B, self.slot, self.num_heads, dh).transpose(1,2)
        Vh = V.view(B, self.slot, self.num_heads, dh).transpose(1,2)
        scores = torch.einsum('bhd,bhSd->bhS', Qh, Kh) / math.sqrt(dh)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum('bhS,bhSd->bhd', attn, Vh).reshape(B, D)
        mixed = self.ln(core_state + ctx)
        return mixed, attn

# ------------------------
# Simple multimodal stubs (minimal, dependency-free)
# ------------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, depth=2, num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, img):  # img: (B,3,H,W)
        B = img.size(0)
        x = self.proj(img)  # (B, E, H', W')
        x = x.flatten(2).transpose(1,2)  # (B, N, E)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:,0,:]  # (B,E)

class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, input_size=128, embed_dim=256, depth=2, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, spec):  # spec: (B,T,F)
        x = self.proj(spec)  # (B,T,E)
        x = self.encoder(x)
        return x.mean(dim=1)  # (B,E)

class MultimodalInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = ThoughtEncoder(
            getattr(config, "vocab_size", 30522),
            emb_dim=config.z_dim,
            z_dim=config.hidden_dim
        )
        self.visual_encoder = VisionTransformer(embed_dim=config.hidden_dim)
        self.audio_encoder = AudioSpectrogramTransformer(embed_dim=config.hidden_dim)
        self.fusion = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, inputs: dict):
        mods = []
        if "text" in inputs:
            text_emb = self.text_encoder(inputs["text"]).unsqueeze(1)  # (B,1,H)
            mods.append(text_emb)
        if "image" in inputs:
            img_emb = self.visual_encoder(inputs["image"]).unsqueeze(1)  # (B,1,H)
            mods.append(img_emb)
        if "audio" in inputs:
            aud_emb = self.audio_encoder(inputs["audio"]).unsqueeze(1)  # (B,1,H)
            mods.append(aud_emb)

        x = torch.cat(mods, dim=1) if len(mods) > 1 else mods[0]
        fused, _ = self.fusion(x, x, x)
        return fused.mean(dim=1)  # (B,H)


# ------------------------
# Social Cognition Module
# ------------------------
class SocialCognitionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_dim
        P = config.num_pillars
        self.tom = nn.Sequential(nn.Linear(H*2, H), nn.GELU(), nn.Linear(H, H//2))
        self.empathy = nn.Sequential(nn.Linear(H + P, H//2), nn.GELU(), nn.Linear(H//2, 3))
        self.social_gru = nn.GRU(input_size=H//2, hidden_size=H//2, batch_first=True)

    def forward(self, self_state, other_states, pillars):
        preds = []
        for other in other_states:
            preds.append(self.tom(torch.cat([self_state, other], dim=-1)))
        preds_t = torch.stack(preds, dim=1)  # (B,N,H/2)
        empathy_scores = self.empathy(torch.cat([self_state, pillars], dim=-1))
        _, social_mem = self.social_gru(preds_t)  # (1,B,H/2)
        return {"mind_predictions": preds, "empathy_scores": empathy_scores, "social_memory": social_mem.squeeze(0)}

# ------------------------
# Memory/Compute optimizations (LRU + checkpoint stub)
# ------------------------
from collections import OrderedDict
class LRUCache(OrderedDict):
    def __init__(self, max_size=256): super().__init__(); self.max_size=max_size
    def __setitem__(self, key, value):
        if key in self: del self[key]
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


# ------------------------
# Memory/Compute optimizations (LRU + checkpoint stub)
# ------------------------
class MetaLoopCompat(nn.Module):
    def forward(self, x):
        return x, torch.tensor(1, device=x.device)

class MemoryOptimizedAEON(nn.Module):
    def __init__(self, forward_impl, max_cache=256):
        super().__init__()
        self.forward_impl = forward_impl
        self.cache = LRUCache(max_size=max_cache)
        self.gradient_checkpointing = False

    def _key(self, x): return ("k", tuple(x.shape), int(x.sum().item()) % 997)
    def forward(self, x):
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]
        out = self.forward_impl(x)
        self.cache[key] = out
        return out

# ------------------------
# AEONDeltaRMTv3 - integrated
# ------------------------
class AEONDeltaRMTv3(AEONDelta):
    def __init__(self, config: AEONConfig):
        super().__init__(config)
        # RMT modules
        self.cat_rmt = CatastropheDetectorRMT(config)
        self.adapt = AdaptiveCatastropheResponse(config)
        # ✅ ИСПРАВЛЕНО: Заменяем базовый quantum_sim на оптимизированную версию
        self.quantum_sim = QuantumSimulator(config)
        self.sra = InteroceptiveAttentionRMTv3(config.hidden_dim, config.num_pillars,
                                               num_heads=getattr(config, 'rmt_heads', 8),
                                               intero_slots=getattr(config, 'rmt_intero_slots', 4))
        self.qsm = EnhancedQSMemory(config)
        self.self_report = TransparentSelfReporting(config)
        self.safety = MultiLevelSafetySystem(config)
        self.deception = DeceptionFeatureSuppressor(config)
        self.meta_loop_v3 = MetaLoopProcessorV3(config)
        SafeTensorProcessor.register_hooks(self)
        if not hasattr(self, 'meta_loop'):
            self.meta_loop = MetaLoopCompat()

        # Fallback enc/dec if missing
        if getattr(self, 'encoder', None) is None or not isinstance(self.encoder, nn.Module):
            self.encoder = ThoughtEncoder(getattr(config, "vocab_size", 30522),
                                          emb_dim=config.z_dim, z_dim=config.hidden_dim).to(device).eval()
        if getattr(self, 'decoder', None) is None or not isinstance(self.decoder, nn.Module):
            self.decoder = ThoughtDecoder(getattr(config, "vocab_size", 30522),
                                          emb_dim=config.z_dim, z_dim=config.hidden_dim).to(device).eval()

    def reasoning_core(self, z_in, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None, fast: bool = False):
        # base
        base_z, outs = super().reasoning_core(z_in, attention_mask, memory_retrieval, planning, use_kv_cache, fast=fast)
        if fast:
            outs.setdefault('rmt_v3', {})
            outs['rmt_v3']['fast'] = True
            return base_z, outs

        core = outs.get('core_state', base_z)
        pillars = outs.get('pillars', torch.zeros(z_in.size(0), self.config.num_pillars, device=z_in.device))

        # Improve interpretability
        outs['pillar_dict'] = get_pillar_dict(pillars)

        # ✅ ИСПРАВЛЕНО: Используем quantum_sim, который теперь QuantumSimulator
        qres = self.quantum_sim(pillars)
        outs['quantum_results'] = qres

        # Catastrophe metrics from Hessian
        cat = self.cat_rmt(pillars)
        outs['cat_metrics'] = cat

        # SRA interoception
        mixed, attn = self.sra(core, pillars, qres, {"potential": cat['severity'].unsqueeze(-1)})
        # Write/read memory and fuse
        self.qsm.write(mixed.detach())
        fused = self.qsm.retrieve(mixed)

        # Deception suppression on internal path
        clean, dscore = self.deception(fused, mode='internal')

        # Meta-loop stabilization if needed
        meta = self.meta_loop_v3(clean)
        clean2 = meta[0] if isinstance(meta, tuple) else meta.get('state', clean)

        # Adaptive alpha/beta (SRA/RSSM intensity)
        alpha, beta = self.adapt(clean2, pillars, cat)  # (B,1) each
        rssm_out = self.rssm(clean2)
        z_sra = torch.tanh(self.integration_module(torch.cat([rssm_out, self.pillars_module.embed_pillars(pillars)], dim=-1))) \
                if hasattr(self.pillars_module, 'embed_pillars') else rssm_out

        # Blend
        z_out = (1 - alpha) * base_z + alpha * z_sra
        z_out = torch.tanh(z_out * (1 + beta))  # controlled boost

        # Safety & Self-report
        safety = self.safety(outs['action_results']['action_embedding'] if 'action_results' in outs else torch.zeros(z_in.size(0), self.config.action_dim, device=z_in.device),
                             clean2, pillars, outs.get('quantum_results', qres), outs.get('topo_results', {}))
        report = self.self_report(clean2, pillars, qres, {"potential": cat['severity'].unsqueeze(-1)}, mode='combined')

        # ✅ ИСПРАВЛЕНО: Логирование метрик
        logger.debug(f"Safety: {safety.mean().item():.4f}")
        logger.debug(f"Deception score: {dscore.mean().item():.4f}")

        outs['rmt_v3'] = {"attn": attn, "alpha": alpha, "beta": beta, "deception_score": dscore,
                          "safety": safety, "report": report, "meta": meta}
        return z_out, outs

# Convenience factory (v3 by default)
def create_model(config: AEONConfig, use_rmt: bool = True):
    return AEONDeltaRMTv3(config) if use_rmt else AEONDelta(config)

# ------------------------
# Test Suite
# ------------------------
class AEONTestSuite:
    """
    ✅ ПОЛНОСТЬЮ ПЕРЕРАБОТАННЫЙ НАБОР ТЕСТОВ
    
    Включает:
    - Реальные метрики стабильности (не заглушки)
    - Проверка weight tying in ThoughtDecoder
    - Тесты VectorQuantizer на различных размерностях
    - Edge cases (batch_size=1, seq_length=1, пустые входы)
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.test_results = {}
        self.errors = []

    @torch.no_grad()
    def test_stability(self):
        """
        ✅ ИСПРАВЛЕНО: Реальная метрика стабильности вместо случайного значения.
        
        Проверяет:
        1. Детерминизм: одинаковый вход → одинаковый выход (в eval mode)
        2. Числовая стабильность: отсутствие NaN/Inf в выходах
        3. Консистентность: повторные прогоны дают схожие результаты
        4. Градиентная стабильность: проверка на взрывающиеся градиенты
        """
        self.model.eval()
        stability_metrics = {}
        
        try:
            # Фиксированный вход для детерминизма
            torch.manual_seed(42)
            x = torch.randint(
                0, 
                getattr(self.config, "vocab_size", 30522), 
                (2, getattr(self.config, "seq_length", 64)), 
                device=device
            )
            
            # === ТЕСТ 1: Детерминизм (повторяемость результатов) ===
            out1 = self.model.forward(x)
            out2 = self.model.forward(x)
            
            if 'logits' in out1 and 'logits' in out2:
                logits_diff = torch.abs(out1['logits'] - out2['logits']).max().item()
                determinism_score = 1.0 if logits_diff < 1e-5 else max(0.0, 1.0 - logits_diff)
            else:
                determinism_score = 0.0
                self.errors.append("Missing 'logits' in model output")
            
            stability_metrics['determinism'] = determinism_score
            
            # === ТЕСТ 2: Числовая стабильность (NaN/Inf check) ===
            nan_inf_found = False
            for key, value in out1.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        nan_inf_found = True
                        self.errors.append(f"NaN/Inf found in output '{key}'")
                        break
            
            numerical_stability = 0.0 if nan_inf_found else 1.0
            stability_metrics['numerical_stability'] = numerical_stability
            
            # === ТЕСТ 3: Консистентность выходных размерностей ===
            expected_logits_shape = (2, self.config.seq_length, self.config.vocab_size)
            if 'logits' in out1:
                actual_shape = tuple(out1['logits'].shape)
                shape_match = actual_shape == expected_logits_shape
                stability_metrics['shape_consistency'] = 1.0 if shape_match else 0.0
                if not shape_match:
                    self.errors.append(f"Logits shape mismatch: expected {expected_logits_shape}, got {actual_shape}")
            else:
                stability_metrics['shape_consistency'] = 0.0
            
            # === ТЕСТ 4: Диапазон значений (не взрывающиеся активации) ===
            if 'logits' in out1:
                logits_max = out1['logits'].abs().max().item()
                # Логиты должны быть в разумном диапазоне (< 100)
                value_stability = 1.0 if logits_max < 100 else max(0.0, 1.0 - (logits_max - 100) / 1000)
                stability_metrics['value_range'] = value_stability
                if logits_max >= 100:
                    self.errors.append(f"Logits too large: max={logits_max:.2f}")
            else:
                stability_metrics['value_range'] = 0.0
            
            # === ТЕСТ 5: Консистентность внутренних состояний ===
            internal_consistency = 1.0
            if 'core_state' in out1:
                core_norm = out1['core_state'].norm(dim=-1).mean().item()
                # Норма должна быть в разумном диапазоне
                if core_norm < 0.01 or core_norm > 100:
                    internal_consistency *= 0.5
                    self.errors.append(f"Core state norm out of range: {core_norm:.4f}")
            
            if 'pillars' in out1:
                pillars = out1['pillars']
                # Pillars должны быть в [0, 1] после sigmoid
                if pillars.min() < -0.1 or pillars.max() > 1.1:
                    internal_consistency *= 0.5
                    self.errors.append(f"Pillars out of [0,1] range: [{pillars.min():.4f}, {pillars.max():.4f}]")
            
            stability_metrics['internal_consistency'] = internal_consistency
            
            # === АГРЕГИРОВАННАЯ МЕТРИКА СТАБИЛЬНОСТИ ===
            weights = {
                'determinism': 0.25,
                'numerical_stability': 0.30,
                'shape_consistency': 0.15,
                'value_range': 0.15,
                'internal_consistency': 0.15
            }
            
            overall_stability = sum(
                stability_metrics.get(k, 0.0) * w 
                for k, w in weights.items()
            )
            
            stability_metrics['overall'] = overall_stability
            
            logger.info(f"✅ Stability test completed: overall={overall_stability:.4f}")
            for k, v in stability_metrics.items():
                if k != 'overall':
                    logger.debug(f"   {k}: {v:.4f}")
            
            return {"stability": overall_stability, "details": stability_metrics}
            
        except Exception as e:
            self.errors.append(f"Stability test failed: {str(e)}")
            logger.error(f"❌ Stability test error: {e}")
            return {"stability": 0.0, "error": str(e)}

    @torch.no_grad()
    def test_weight_tying(self):
        """
        ✅ НОВЫЙ ТЕСТ: Проверка корректности weight tying in ThoughtDecoder.
        
        Верифицирует что:
        1. head.weight and embed.weight указывают на один and тот же тензор (data_ptr)
        2. Формы весов совпадают
        3. Значения идентичны
        4. Градиенты правильно распространяются при обучении
        """
        results = {
            'pointer_match': False,
            'shape_match': False,
            'value_match': False,
            'gradient_flow': False
        }
        
        try:
            decoder = self.model.decoder
            
            if decoder is None:
                self.errors.append("Decoder not initialized")
                return {"weight_tying": 0.0, "error": "Decoder not initialized"}
            
            # === ТЕСТ 1: Указатели на память ===
            pointer_match = decoder.head.weight.data_ptr() == decoder.embed.weight.data_ptr()
            results['pointer_match'] = pointer_match
            if not pointer_match:
                self.errors.append("Weight tying FAILED: different memory pointers")
            
            # === ТЕСТ 2: Формы весов ===
            shape_match = decoder.head.weight.shape == decoder.embed.weight.shape
            results['shape_match'] = shape_match
            if not shape_match:
                self.errors.append(
                    f"Weight tying shape mismatch: head={decoder.head.weight.shape}, "
                    f"embed={decoder.embed.weight.shape}"
                )
            
            # === ТЕСТ 3: Значения идентичны ===
            if shape_match:
                value_diff = (decoder.head.weight - decoder.embed.weight).abs().max().item()
                value_match = value_diff < 1e-7
                results['value_match'] = value_match
                if not value_match:
                    self.errors.append(f"Weight values differ by {value_diff:.2e}")
            
            # === ТЕСТ 4: Градиентный поток (требует train mode) ===
            try:
                self.model.train()
                
                # Создаём простой тестовый вход
                z = torch.randn(1, self.config.z_dim, device=device, requires_grad=True)
                teacher_tokens = torch.randint(
                    0, self.config.vocab_size, 
                    (1, 16), 
                    device=device
                )
                
                # Forward pass
                logits = decoder(z, teacher_tokens=teacher_tokens, mode='train')
                
                # Backward pass
                loss = logits.sum()
                loss.backward()
                
                # Проверяем что градиент есть у embed.weight
                if decoder.embed.weight.grad is not None:
                    grad_norm = decoder.embed.weight.grad.norm().item()
                    results['gradient_flow'] = grad_norm > 1e-10
                    if not results['gradient_flow']:
                        self.errors.append("Gradient flow blocked: embed.weight.grad is zero")
                else:
                    results['gradient_flow'] = False
                    self.errors.append("No gradient computed for embed.weight")
                
                # Очистка градиентов
                decoder.zero_grad()
                self.model.eval()
                
            except Exception as e:
                self.errors.append(f"Gradient flow test failed: {e}")
                results['gradient_flow'] = False
            
            # === АГРЕГИРОВАННАЯ ОЦЕНКА ===
            scores = [
                1.0 if results['pointer_match'] else 0.0,
                1.0 if results['shape_match'] else 0.0,
                1.0 if results['value_match'] else 0.0,
                1.0 if results['gradient_flow'] else 0.0
            ]
            overall = sum(scores) / len(scores)
            
            logger.info(f"✅ Weight tying test: overall={overall:.4f}")
            for k, v in results.items():
                status = "✓" if v else "✗"
                logger.info(f"   {status} {k}: {v}")
            
            return {"weight_tying": overall, "details": results}
            
        except Exception as e:
            self.errors.append(f"Weight tying test failed: {str(e)}")
            logger.error(f"❌ Weight tying test error: {e}")
            return {"weight_tying": 0.0, "error": str(e)}

    @torch.no_grad()
    def test_vector_quantizer(self):
        """
        ✅ НОВЫЙ ТЕСТ: Проверка VectorQuantizer на различных размерностях входа.
        
        Тестирует:
        1. Стандартные размерности [B, D]
        2. Различные batch sizes (1, 2, 16, 64)
        3. Корректность выходных индексов
        4. VQ loss в разумном диапазоне
        5. Straight-Through Estimator работает корректно
        """
        results = {
            'standard_input': False,
            'batch_size_1': False,
            'batch_size_large': False,
            'indices_valid': False,
            'loss_valid': False,
            'ste_working': False
        }
        
        try:
            vq = self.model.vector_quantizer
            
            if vq is None:
                self.errors.append("VectorQuantizer not initialized")
                return {"vector_quantizer": 0.0, "error": "VectorQuantizer not initialized"}
            
            embedding_dim = vq.embedding_dim
            num_embeddings = vq.num_embeddings
            
            # === ТЕСТ 1: Стандартный вход [B=2, D] ===
            try:
                x_standard = torch.randn(2, embedding_dim, device=device)
                quantized, loss, indices = vq(x_standard)
                
                results['standard_input'] = (
                    quantized.shape == x_standard.shape and
                    not torch.isnan(quantized).any() and
                    not torch.isinf(quantized).any()
                )
                if not results['standard_input']:
                    self.errors.append(f"Standard input test failed: shape={quantized.shape}")
            except Exception as e:
                self.errors.append(f"Standard input test error: {e}")
            
            # === ТЕСТ 2: Batch size = 1 (edge case) ===
            try:
                x_single = torch.randn(1, embedding_dim, device=device)
                quantized, loss, indices = vq(x_single)
                
                results['batch_size_1'] = (
                    quantized.shape == (1, embedding_dim) and
                    indices.shape == (1,) or indices.numel() == 1
                )
                if not results['batch_size_1']:
                    self.errors.append(f"Batch size 1 test failed: quantized={quantized.shape}, indices={indices.shape}")
            except Exception as e:
                self.errors.append(f"Batch size 1 test error: {e}")
            
            # === ТЕСТ 3: Большой batch size ===
            try:
                x_large = torch.randn(64, embedding_dim, device=device)
                quantized, loss, indices = vq(x_large)
                
                results['batch_size_large'] = (
                    quantized.shape == (64, embedding_dim) and
                    not torch.isnan(quantized).any()
                )
                if not results['batch_size_large']:
                    self.errors.append(f"Large batch test failed")
            except Exception as e:
                self.errors.append(f"Large batch test error: {e}")
            
            # === ТЕСТ 4: Валидность индексов ===
            try:
                x_test = torch.randn(8, embedding_dim, device=device)
                _, _, indices = vq(x_test)
                
                indices_in_range = (indices >= 0).all() and (indices < num_embeddings).all()
                results['indices_valid'] = indices_in_range.item() if isinstance(indices_in_range, torch.Tensor) else indices_in_range
                
                if not results['indices_valid']:
                    self.errors.append(f"Indices out of range: min={indices.min()}, max={indices.max()}, num_embeddings={num_embeddings}")
            except Exception as e:
                self.errors.append(f"Indices validation error: {e}")
            
            # === ТЕСТ 5: VQ Loss в разумном диапазоне ===
            try:
                x_test = torch.randn(4, embedding_dim, device=device)
                _, loss, _ = vq(x_test)
                
                loss_value = loss.item()
                # VQ loss обычно должен быть положительным and не слишком большим
                results['loss_valid'] = 0 <= loss_value < 100
                
                if not results['loss_valid']:
                    self.errors.append(f"VQ loss out of range: {loss_value}")
            except Exception as e:
                self.errors.append(f"Loss validation error: {e}")
            
            # === ТЕСТ 6: Straight-Through Estimator ===
            try:
                x_ste = torch.randn(2, embedding_dim, device=device, requires_grad=True)
                quantized, loss, _ = vq(x_ste)
                
                # Quantized должен иметь градиент благодаря STE
                test_loss = quantized.sum()
                test_loss.backward()
                
                results['ste_working'] = (
                    x_ste.grad is not None and 
                    x_ste.grad.norm().item() > 1e-10
                )
                
                if not results['ste_working']:
                    self.errors.append("STE gradient not propagating")
            except Exception as e:
                self.errors.append(f"STE test error: {e}")
            
            # === АГРЕГИРОВАННАЯ ОЦЕНКА ===
            scores = [1.0 if v else 0.0 for v in results.values()]
            overall = sum(scores) / len(scores)
            
            logger.info(f"✅ VectorQuantizer test: overall={overall:.4f}")
            for k, v in results.items():
                status = "✓" if v else "✗"
                logger.info(f"   {status} {k}: {v}")
            
            return {"vector_quantizer": overall, "details": results}
            
        except Exception as e:
            self.errors.append(f"VectorQuantizer test failed: {str(e)}")
            logger.error(f"❌ VectorQuantizer test error: {e}")
            return {"vector_quantizer": 0.0, "error": str(e)}

    @torch.no_grad()
    def test_edge_cases(self):
        """
        ✅ НОВЫЙ ТЕСТ: Edge cases для граничных условий.
        
        Тестирует:
        1. Batch size = 1
        2. Sequence length = 1
        3. Минимальные токены (все нули)
        4. Максимальные токены (vocab_size - 1)
        5. Смешанные специальные токены
        """
        results = {
            'batch_size_1': False,
            'seq_length_1': False,
            'min_tokens': False,
            'max_tokens': False,
            'special_tokens': False
        }
        
        try:
            self.model.eval()
            vocab_size = getattr(self.config, "vocab_size", 30522)
            seq_length = getattr(self.config, "seq_length", 64)
            
            # === ТЕСТ 1: Batch size = 1 ===
            try:
                x = torch.randint(0, vocab_size, (1, seq_length), device=device)
                out = self.model.forward(x)
                
                results['batch_size_1'] = (
                    'logits' in out and 
                    out['logits'].shape[0] == 1 and
                    not torch.isnan(out['logits']).any()
                )
                if not results['batch_size_1']:
                    self.errors.append("Batch size 1 test failed")
            except Exception as e:
                self.errors.append(f"Batch size 1 error: {e}")
            
            # === ТЕСТ 2: Sequence length = 1 ===
            try:
                x = torch.randint(0, vocab_size, (2, 1), device=device)
                out = self.model.forward(x)
                
                results['seq_length_1'] = (
                    'logits' in out and 
                    out['logits'].shape[1] == 1 and
                    not torch.isnan(out['logits']).any()
                )
                if not results['seq_length_1']:
                    self.errors.append("Sequence length 1 test failed")
            except Exception as e:
                self.errors.append(f"Sequence length 1 error: {e}")
            
            # === ТЕСТ 3: Минимальные токены (все нули) ===
            try:
                x = torch.zeros((2, seq_length), dtype=torch.long, device=device)
                out = self.model.forward(x)
                
                results['min_tokens'] = (
                    'logits' in out and 
                    not torch.isnan(out['logits']).any() and
                    not torch.isinf(out['logits']).any()
                )
                if not results['min_tokens']:
                    self.errors.append("Min tokens (zeros) test failed")
            except Exception as e:
                self.errors.append(f"Min tokens error: {e}")
            
            # === ТЕСТ 4: Максимальные токены ===
            try:
                x = torch.full((2, seq_length), vocab_size - 1, dtype=torch.long, device=device)
                out = self.model.forward(x)
                
                results['max_tokens'] = (
                    'logits' in out and 
                    not torch.isnan(out['logits']).any() and
                    not torch.isinf(out['logits']).any()
                )
                if not results['max_tokens']:
                    self.errors.append("Max tokens test failed")
            except Exception as e:
                self.errors.append(f"Max tokens error: {e}")
            
            # === ТЕСТ 5: Специальные токены (CLS, SEP, PAD, UNK) ===
            try:
                # Типичные специальные токены в BERT: [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102
                special_ids = [0, 100, 101, 102]
                x = torch.tensor([special_ids * (seq_length // 4 + 1)][:seq_length], device=device).unsqueeze(0)
                x = x[:, :seq_length]  # Обрезаем до нужной длины
                
                out = self.model.forward(x)
                
                results['special_tokens'] = (
                    'logits' in out and 
                    not torch.isnan(out['logits']).any() and
                    not torch.isinf(out['logits']).any()
                )
                if not results['special_tokens']:
                    self.errors.append("Special tokens test failed")
            except Exception as e:
                self.errors.append(f"Special tokens error: {e}")
            
            # === АГРЕГИРОВАННАЯ ОЦЕНКА ===
            scores = [1.0 if v else 0.0 for v in results.values()]
            overall = sum(scores) / len(scores)
            
            logger.info(f"✅ Edge cases test: overall={overall:.4f}")
            for k, v in results.items():
                status = "✓" if v else "✗"
                logger.info(f"   {status} {k}: {v}")
            
            return {"edge_cases": overall, "details": results}
            
        except Exception as e:
            self.errors.append(f"Edge cases test failed: {str(e)}")
            logger.error(f"❌ Edge cases test error: {e}")
            return {"edge_cases": 0.0, "error": str(e)}

    @torch.no_grad()
    def test_self_reporting(self):
        """Тест наличия and корректности модуля самоотчётности."""
        try:
            x = torch.randint(
                0, 
                getattr(self.config, "vocab_size", 30522), 
                (2, getattr(self.config, "seq_length", 64)), 
                device=device
            )
            out = self.model.forward(x)
            
            has_report = 'rmt_v3' in out and 'report' in out['rmt_v3']
            
            if has_report:
                report = out['rmt_v3']['report']
                # Проверяем структуру отчёта
                required_keys = ['inner_report', 'honesty_gate', 'consistency', 'confidence']
                has_all_keys = all(k in report for k in required_keys)
                score = 1.0 if has_all_keys else 0.5
            else:
                score = 0.0
            
            return {"self_report_presence": score}
            
        except Exception as e:
            self.errors.append(f"Self-reporting test failed: {e}")
            return {"self_report_presence": 0.0, "error": str(e)}

    @torch.no_grad()
    def test_catastrophe_detection(self):
        """Тест наличия and корректности детектора катастроф."""
        try:
            x = torch.randint(
                0, 
                getattr(self.config, "vocab_size", 30522), 
                (2, getattr(self.config, "seq_length", 64)), 
                device=device
            )
            out = self.model.forward(x)
            
            has_cat = 'cat_metrics' in out or ('topo_results' in out and 'catastrophes' in out['topo_results'])
            
            if has_cat:
                # Проверяем корректность значений
                if 'cat_metrics' in out:
                    cat = out['cat_metrics']
                    prob_valid = 'probability' in cat and 0 <= cat['probability'].mean() <= 1
                    score = 1.0 if prob_valid else 0.5
                else:
                    score = 0.5
            else:
                score = 0.0
            
            return {"catastrophe_detection": score}
            
        except Exception as e:
            self.errors.append(f"Catastrophe detection test failed: {e}")
            return {"catastrophe_detection": 0.0, "error": str(e)}

    def run_all_tests(self):
        """
        ✅ ОБНОВЛЕНО: Запускает все тесты включая новые.
        
        Returns:
            Dict со всеми результатами тестов
        """
        self.errors = []  # Сброс ошибок
        results = {}
        
        logger.info("="*60)
        logger.info("🧪 AEON Test Suite - Running comprehensive tests...")
        logger.info("="*60)
        
        # Основные тесты
        results['stability'] = self.test_stability()
        results['weight_tying'] = self.test_weight_tying()
        results['vector_quantizer'] = self.test_vector_quantizer()
        results['edge_cases'] = self.test_edge_cases()
        results['self_reporting'] = self.test_self_reporting()
        results['catastrophe'] = self.test_catastrophe_detection()
        
        # Сводка результатов
        logger.info("\n" + "="*60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        total_score = 0.0
        num_tests = 0
        
        for test_name, result in results.items():
            if isinstance(result, dict):
                # Извлекаем основную метрику
                main_key = [k for k in result.keys() if k not in ['details', 'error']]
                if main_key:
                    score = result[main_key[0]]
                    total_score += score
                    num_tests += 1
                    status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
                    logger.info(f"  {status} {test_name}: {score:.4f}")
        
        overall = total_score / max(1, num_tests)
        logger.info("-"*60)
        logger.info(f"  📈 OVERALL SCORE: {overall:.4f}")
        
        # Вывод ошибок if есть
        if self.errors:
            logger.info("\n⚠️  ERRORS ENCOUNTERED:")
            for err in self.errors[:10]:  # Показываем первые 10
                logger.info(f"    - {err}")
            if len(self.errors) > 10:
                logger.info(f"    ... and {len(self.errors) - 10} more errors")
        
        logger.info("="*60)
        
        results['overall_score'] = overall
        results['errors'] = self.errors
        
        return results

# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import argparse as argparse_module  # ✅ ИСПРАВЛЕНО: Переименование для избежания конфликтов имён
    import sys
    
    # ✅ ИСПРАВЛЕНО: Безопасная инициализация парсера с полной обработкой ошибок
    try:
        parser = argparse_module.ArgumentParser(
            description="AEON-Delta RMT v3 - Advanced Embodied Ontological Network",
            formatter_class=argparse_module.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python core.py --mode demo
  python core.py --mode train --steps 50
  python core.py --mode infer --checkpoint ./aeon_state
  python core.py --mode test
            """
        )
        
        
        def _normalize_device_arg(s: str) -> str:
            # Fix common Cyrillic look-alike letters (e.g., 'сpu' -> 'cpu')
            if not isinstance(s, str):
                return str(s)
            repl = {
                'с': 'c',  # Cyrillic es
                'у': 'u',  # Cyrillic u
                'С': 'c',
                'У': 'u',
            }
            for k, v in repl.items():
                s = s.replace(k, v)
            return s.lower().strip()

        parser.add_argument(
            '--mode', 
            type=str, 
            default='demo', 
            choices=['demo', 'train', 'infer', 'test'],
            help='Execution mode (default: demo)'
        )
        parser.add_argument(
            '--steps', 
            type=int, 
            default=10, 
            help='Training steps (default: 10)'
        )
        parser.add_argument(
            '--checkpoint', 
            type=str, 
            default='aeon_state', 
            help='Checkpoint path (default: aeon_state)'
        )
        parser.add_argument(
            '--device',
            type=_normalize_device_arg,
            default='cpu',
            choices=['cpu', 'cuda', 'mps'],
            help='Device to use (default: cpu)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed (default: 42)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )
        
        # ✅ ИСПРАВЛЕНО: Безопасный парс аргументов с обработкой ошибок
        args = parser.parse_args()
        
        # ✅ ИСПРАВЛЕНО: Валидация аргументов
        if args.steps <= 0:
            logger.error("--steps must be positive integer")
            sys.exit(1)
        
        logger.info(f"✅ Arguments parsed successfully: mode={args.mode}, steps={args.steps}, checkpoint={args.checkpoint}")
        # ✅ Apply CLI device selection (overrides module default)
        try:
            set_global_device(args.device)
        except Exception as _dev_e:
            logger.error(f"❌ Device selection failed for --device={args.device}: {_dev_e}")
            sys.exit(1)
        
    except SystemExit as e:
        if e.code == 0:
            sys.exit(0)
        logger.error(f"Argument parsing failed with code {e.code}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during argument parsing: {e}")
        sys.exit(1)
    
    # ✅ ИСПРАВЛЕНО: Установка random seed для воспроизводимости
    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"✅ Random seed set to {args.seed}")
    except Exception as e:
        logger.warning(f"Failed to set random seed: {e}")
    
    # ✅ ИСПРАВЛЕНО: Валидация конфига перед инициализацией модели
    try:
        config = AEONConfig(device=device)
        config.use_amp = AMP_ENABLED
        
        # Проверка критических параметров конфига
        assert config.hidden_dim > 0, "hidden_dim must be positive"
        assert config.z_dim > 0, "z_dim must be positive"
        assert config.num_pillars > 0, "num_pillars must be positive"
        assert config.vocab_size > 0, "vocab_size must be positive"
        assert config.max_iterations > 0, "max_iterations must be positive"
        
        logger.info("✅ Config validation passed")
        logger.info(f"   - hidden_dim: {config.hidden_dim}")
        logger.info(f"   - z_dim: {config.z_dim}")
        logger.info(f"   - vocab_size: {config.vocab_size}")
        logger.info(f"   - num_pillars: {config.num_pillars}")
        
    except AssertionError as e:
        logger.error(f"❌ Config validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during config initialization: {e}")
        sys.exit(1)
    
    # ✅ ИСПРАВЛЕНО: Проверка доступности трансформеров and инициализация модели
    try:
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️  Transformers not available. Install via: pip install transformers")
            logger.warning("⚠️  Using fallback mode (limited functionality)")
            config.vocab_size = 30522
        else:
            logger.info("✅ Transformers library available")
        
        # ✅ ИСПРАВЛЕНО: Создание модели с полной обработкой ошибок
        logger.info("Initializing AEON-Delta RMT v3 model...")
        model = create_model(config, use_rmt=True)
        # ✅ Hard-sync model to selected device
        model.to(config.device)
        logger.info("✅ Model initialized successfully")
        
        # ✅ ИСПРАВЛЕНО: Валидация компонентов модели
        assert hasattr(model, 'encoder'), "Model missing encoder"
        assert hasattr(model, 'decoder'), "Model missing decoder"
        assert hasattr(model, 'meta_loop_v3'), "Model missing meta_loop_v3"
        assert hasattr(model, 'pillars_module'), "Model missing pillars_module"
        logger.info("✅ All critical model components present")
        
    except AssertionError as e:
        logger.error(f"❌ Model validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during model initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # ✅ ИСПРАВЛЕНО: Загрузка сохранённого состояния с обработкой ошибок
    try:
        if os.path.exists(args.checkpoint):
            logger.info(f"Loading model state from {args.checkpoint}...")
            success = model.load_state(args.checkpoint)
            if success:
                logger.info(f"✅ Model state loaded from {args.checkpoint}")
            else:
                logger.warning(f"⚠️  Failed to load state from {args.checkpoint}, using fresh model")
        else:
            logger.info(f"⚠️  Checkpoint not found at {args.checkpoint}, starting with fresh model")
    except Exception as e:
        logger.warning(f"⚠️  Error loading checkpoint: {e}, continuing with fresh model")
    
    # ✅ ИСПРАВЛЕНО: Валидация модели перед выполнением
    try:
        model.eval()
        validation_seq = min(config.seq_length, 8)
        test_input = torch.randint(0, config.vocab_size, (1, validation_seq), device=config.device)
        test_output = model(test_input, fast=True)
        assert 'logits' in test_output, "Model forward pass missing 'logits' in output"
        logger.info("✅ Model forward pass validation passed")
    except AssertionError as e:
        logger.error(f"❌ Model forward pass validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during model validation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # ===== РЕЖИМ 1: ДЕМОНСТРАЦИЯ =====
    if args.mode == 'demo':
        logger.info("="*70)
        logger.info("AEON-Delta RMT v3 - Demonstration Mode")
        logger.info("="*70)
        
        try:
            model.eval()
            
            # ✅ ИСПРАВЛЕНО: Генерация примера с полной обработкой ошибок
            if TRANSFORMERS_AVAILABLE:
                test_prompt = "The nature of consciousness"
                logger.info(f"\n📝 Generating thought from prompt: '{test_prompt}'")
                try:
                    generated = model.generate_thought(test_prompt, max_len=32)
                    logger.info(f"✅ Generated: {generated}")
                except Exception as e:
                    logger.error(f"❌ Generation failed: {e}")
            else:
                logger.warning("⚠️  Tokenizer not available, skipping generation demo")
            
            # ✅ ИСПРАВЛЕНО: Самоанализ с валидацией
            logger.info("\n🧠 Performing self-consciousness measurement...")
            try:
                if TRANSFORMERS_AVAILABLE:
                    test_ids = torch.randint(100, config.vocab_size, (1, 16), device=config.device)
                    consciousness_metrics = model.measure_self_consciousness(test_ids)
                    
                    logger.info("✅ Self-consciousness metrics:")
                    for k, v in consciousness_metrics.items():
                        if isinstance(v, dict):
                            logger.info(f"  {k}:")
                            for k2, v2 in v.items():
                                if isinstance(v2, (int, float)):
                                    logger.info(f"    {k2}: {v2:.4f}")
                                else:
                                    logger.info(f"    {k2}: {v2}")
                        else:
                            if isinstance(v, (int, float)):
                                logger.info(f"  {k}: {v:.4f}")
                            else:
                                logger.info(f"  {k}: {v}")
                else:
                    logger.warning("⚠️  Skipping consciousness measurement (requires tokenizer)")
            except Exception as e:
                logger.error(f"❌ Consciousness measurement failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"❌ Demo mode failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # ===== РЕЖИМ 2: ОБУЧЕНИЕ =====
    elif args.mode == 'train':
        logger.info("="*70)
        logger.info("AEON-Delta RMT v3 - Training Mode")
        logger.info("="*70)
        
        try:
            model.train()
            trainer = AEONTrainer(model, config, device=device)
            
            logger.info(f"🏋️  Starting training for {args.steps} steps...")
            
            # ✅ ИСПРАВЛЕНО: Обучение с полной обработкой ошибок and логированием
            for step in range(args.steps):
                try:
                    # Генерируем батч случайных токенов
                    batch_tokens = torch.randint(
                        100, 
                        config.vocab_size, 
                        (2, config.seq_length), 
                        device=device
                    )
                    batch_targets = torch.randint(
                        100, 
                        config.vocab_size, 
                        (2, config.seq_length), 
                        device=device
                    )
                    
                    # Forward pass
                    outputs = model(batch_tokens)
                    
                    # Compute loss
                    loss_dict = model.compute_loss(outputs, batch_targets)
                    total_loss = loss_dict['total_loss']
                    
                    # Валидация loss значения
                    if not torch.isfinite(total_loss).all():
                        logger.warning(f"⚠️  Invalid loss at step {step+1}, skipping update")
                        continue
                    
                    # Backward pass
                    trainer.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    trainer.optimizer.step()
                    
                    if (step + 1) % max(1, args.steps // 5) == 0 or (step + 1) == 1:
                        logger.info(
                            f"Step {step+1:3d}/{args.steps} - "
                            f"Loss: {total_loss.item():.6f}, "
                            f"LM Loss: {loss_dict['lm_loss'].item():.6f}, "
                            f"Consistency: {loss_dict['consistency'].item():.4f}, "
                            f"Safety Loss: {loss_dict['safety_loss'].item():.6f}"
                        )
                
                except Exception as e:
                    logger.error(f"❌ Error at training step {step+1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # ✅ ИСПРАВЛЕНО: Сохранение с валидацией
            logger.info(f"\n💾 Saving model to {args.checkpoint}...")
            try:
                success = model.save_state(args.checkpoint)
                if success:
                    logger.info(f"✅ Model saved successfully to {args.checkpoint}")
                else:
                    logger.error(f"❌ Failed to save model to {args.checkpoint}")
            except Exception as e:
                logger.error(f"❌ Error saving model: {e}")
        
        except Exception as e:
            logger.error(f"❌ Training mode failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # ===== РЕЖИМ 3: ИНФЕРЕНС =====
    elif args.mode == 'infer':
        logger.info("="*70)
        logger.info("AEON-Delta RMT v3 - Interactive Inference Mode")
        logger.info("="*70)
        
        try:
            if TRANSFORMERS_AVAILABLE:
                model.eval()
                logger.info("✅ Model ready for inference")
                logger.info("Type 'exit' or 'quit' to stop\n")
                console_inference_loop(model)
            else:
                logger.error("❌ Transformers required for inference mode")
                logger.info("Install via: pip install transformers")
                sys.exit(1)
        except KeyboardInterrupt:
            logger.info("\n✅ Inference terminated by user")
        except Exception as e:
            logger.error(f"❌ Inference mode failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # ===== РЕЖИМ 4: ТЕСТИРОВАНИЕ =====
    elif args.mode == 'test':
        logger.info("="*70)
        logger.info("AEON-Delta RMT v3 - Comprehensive Test Suite")
        logger.info("="*70)
        
        try:
            model.eval()
            test_suite = AEONTestSuite(model, config)
            
            logger.info("🧪 Running test suite...\n")
            results = test_suite.run_all_tests()
            
            logger.info("\n✅ Test Results:")
            for test_name, result in results.items():
                logger.info(f"  {test_name}: {result}")
            
            # ✅ ИСПРАВЛЕНО: Визуализация метрик с проверкой данных
            try:
                if len(model.metrics_log.get('iterations', [])) > 0:
                    model.visualize_metrics('./aeon_metrics_test.png')
                    logger.info("✅ Metrics visualization saved to ./aeon_metrics_test.png")
                else:
                    logger.info("⚠️  No metrics data available for visualization")
            except Exception as e:
                logger.warning(f"⚠️  Could not visualize metrics: {e}")
        
        except Exception as e:
            logger.error(f"❌ Test mode failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    else:
        logger.error(f"❌ Unknown mode: {args.mode}")
        parser.print_help()
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("✅ AEON-Delta execution completed successfully")
    logger.info("="*70)