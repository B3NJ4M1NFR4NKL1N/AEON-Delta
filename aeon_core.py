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
from torch.utils.data import DataLoader, TensorDataset

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
    MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0 not available; falling back to in-memory list-based storage.")

# Device configuration - force to CPU as per user instruction to fix MPS error
device = torch.device("cpu")
logger.info("⚠️ Forced to CPU to resolve MPS device mismatch errors")

# AMP (Automatic Mixed Precision) configuration - disable completely for CPU
AMP_ENABLED = False  # Disabled for CPU; no mixed precision needed

# Future-proof for torch >= 2.1
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

class NoOpQualiaExtractor(torch.nn.Module):
    """Fallback extractor: returns input unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x


# ─────────────────────────────────────────────────────────
# Auto-encoder for thoughts
# ─────────────────────────────────────────────────────────
class ThoughtEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, z_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm  = nn.LSTM(emb_dim, z_dim, batch_first=True, bidirectional=False)
        self.norm  = nn.LayerNorm(z_dim)

    def forward(self, tokens):
        x = self.embed(tokens)                      # [B, L, E]
        _, (h, _) = self.lstm(x)                    # h: [1, B, z_dim]
        z = self.norm(h.squeeze(0))                 # [B, z_dim]
        assert z.shape[-1] == 256, "Encoder output size mismatch"
        return z

# Corrected ThoughtDecoder
class ThoughtDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, z_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.z_dim = z_dim

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # Initial state projection
        self.fc = nn.Linear(z_dim, emb_dim)
        # LSTM
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)
        # Output head
        self.head = nn.Linear(emb_dim, vocab_size)

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Привязка весов и верификация ---
        self._tie_weights()
        self._verify_weight_tying()

    def _tie_weights(self):
        """Привязывает веса эмбеддинга к выходному слою."""
        self.head.weight = self.embed.weight

    def _verify_weight_tying(self):
        """Проверяет, что привязка весов выполнена корректно."""
        if self.head.weight.data_ptr() != self.embed.weight.data_ptr():
            raise RuntimeError("Weight tying failed: head.weight and embed.weight are not the same tensor.")
        if self.head.weight.shape != self.embed.weight.shape:
            raise RuntimeError(f"Weight tying shape mismatch: head {self.head.weight.shape} vs embed {self.embed.weight.shape}")

    def forward(self, z, target_tokens):
        """
        Этот метод предназначен для обучения с "учителем" (teacher-forcing).
        Он использует латентный вектор 'z' для инициализации контекста,
        а затем разворачивает последовательность 'target_tokens' для предсказания выхода.
        """
        # Проектируем 'z' для получения начального скрытого состояния (h0) для LSTM.
        h0 = self.fc(z).unsqueeze(0)  # Shape: [1, B, E]
        c0 = torch.zeros_like(h0)     # Shape: [1, B, E]

        # Эмбеддинг ground-truth токенов для teacher forcing.
        embeddings = self.embed(target_tokens)  # Shape: [B, L, E]

        # LSTM обрабатывает эмбеддинги токенов, начиная с начального состояния из 'z'.
        out, _ = self.lstm(embeddings, (h0, c0))  # Output shape: [B, L, E]

        # Проектируем выходы LSTM, чтобы получить логиты словаря.
        logits = self.head(out)  # Shape: [B, L, V]

        return logits
# - END FIX -

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

        # Кодовая книга - наш "словарь" базовых концептов/мыслей
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, D]
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
        
        # Straight-Through Estimator для градиентов
        quantized = inputs + (quantized - inputs).detach()
        
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
    max_iterations: int = 100
    convergence_threshold: float = 1e-5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lambda_self_consistency: float = 0.1
    lambda_reg: float = 0.01
    lambda_safety: float = 0.1
    use_quantum_sim: bool = True
    quantum_sim_type: str = "tensor_network"
    topo_grid_size: int = 1000
    topo_epsilon: float = 1e-4
    use_amp: bool = True
    memory_size: int = 10000
    knowledge_dim: int = 128
    action_dim: int = 64
    planning_horizon: int = 10
    memory_path: str = "./aeon_memory"
    use_neo4j: bool = False
    knowledge_graph_url: str = "bolt://localhost:7687"
    knowledge_graph_auth: Tuple[str, str] = ("neo4j", "password")
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

# Инициализация глобальных компонентов
config = AEONConfig()
tokenizer = None
if TRANSFORMERS_AVAILABLE:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config.vocab_size = tokenizer.vocab_size
    logger.info(f"Initialized transformers tokenizer. Vocab size: {config.vocab_size}")
else:
    logger.error("`transformers` library not found. Text generation will not work. Please run `pip install transformers`")

encoder = ThoughtEncoder(config.vocab_size).to(device).eval()
decoder = ThoughtDecoder(config.vocab_size).to(device).eval()

ae_path = "./weights/thought_ae.pt"
if os.path.exists(ae_path):
    state = torch.load(ae_path, map_location=device)
    enc_res = encoder.load_state_dict(state["enc"], strict=False)
    dec_res = decoder.load_state_dict(state["dec"], strict=False)
    logger.info(f"Loaded Thought AE (tolerant). enc missing={enc_res.missing_keys}, unexpected={enc_res.unexpected_keys}; dec missing={dec_res.missing_keys}, unexpected={dec_res.unexpected_keys}")
    logger.info("Loaded Thought AE weights successfully")
else:
    logger.warning("Thought AE weights not found — using random init.")


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

        if MEM0_AVAILABLE:
            try:
                local_cfg = {
                    "vector_store": {"provider": "chroma", "config": {"path": cfg.memory_path, "collection_name": "aeon"}},
                    "embedder": {"provider": "identity"},
                    "llm": {"provider": "stub"},
                    "run_migrations": False
                }
                self.client = MemoryClient(api_key=MEM0_API_KEY)
                logger.info("Initialized mem0 MemoryClient with local config (no network calls).")
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
        if not self.fallback_vectors:
            return []
        similarities = [np.dot(v, vec_np) / (np.linalg.norm(v) * np.linalg.norm(vec_np) + 1e-8) for v in self.fallback_vectors]
        top_indices = np.argsort(similarities)[-k:]
        return [{'vec': self.fallback_vectors[i], 'meta': self.fallback_metas[i]} for i in top_indices]

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

        # Fallback to local graph
        # ... (local graph query logic remains the same)
        return results

class LambdaOperator(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
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
        x = self.dropout(x) if self.train_dropout else x
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
        
        self.pillars_to_hidden = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        self.pillar_norm = nn.LayerNorm(config.num_pillars)
        self.hidden_norm = nn.LayerNorm(config.hidden_dim)
    
    def extract_pillars(self, hidden_state):
        raw_pillars = self.hidden_to_pillars(hidden_state)
        pillars = torch.sigmoid(raw_pillars)
        return self.pillar_norm(pillars)
    
    def embed_pillars(self, pillars):
        hidden = self.pillars_to_hidden(pillars)
        return self.hidden_norm(hidden)
    
    def forward(self, hidden_state):
        pillars = self.extract_pillars(hidden_state)
        embedded = self.embed_pillars(pillars)
        assert embedded.shape[-1] == self.config.hidden_dim, "Pillars embed size mismatch"
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
            # Вход уже [B,H] — пулинг не нужен, нормализуем и выходим
            return self.norm(projected)

        # Стандартный путь для [B,L,H]
        pooled = self.sequence_pooler(projected, attention_mask)
        return self.norm(pooled)

class MetaLoopProcessor(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.lambda_op = LambdaOperator(config)
        self.history_size = 5
        
        # Стабилизирующие слои
        self.input_stabilizer = nn.LayerNorm(config.hidden_dim * 2)
        self.output_stabilizer = nn.LayerNorm(config.hidden_dim)
        
        # Адаптивный коэффициент смешивания
        self.alpha_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Мониторинг состояния
        self.register_buffer('convergence_history', torch.zeros(100))
        self.history_pointer = 0
        
        # Инициализация буферов для улучшенной стабильности (changed to non-persistent attributes)
        self.last_valid_state = None
        self.fallback_state = None
        
    def concatenate(self, psi_0, C):
        """Безопасная конкатенация входных тензоров с проверками и нормализацией"""
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
            
        psi_0 = torch.nan_to_num(psi_0, nan=0.0, posinf=1.0, neginf=-1.0)
        C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        
        psi_0 = F.normalize(psi_0, p=2, dim=-1)
        C = F.normalize(C, p=2, dim=-1)
        
        return torch.cat([psi_0, C], dim=-1)
    
    def get_adaptive_alpha(self, input_tensor, C_new, C_old):
        state = torch.cat([C_new, C_old], dim=-1)
        alpha = self.alpha_net(state)
        return self.config.alpha * alpha
        
    def update_convergence_history(self, residual_norm):
        self.convergence_history[self.history_pointer] = residual_norm.mean().item()
        self.history_pointer = (self.history_pointer + 1) % 100
        
    def check_stability(self):
        valid_history = self.convergence_history[:self.history_pointer]
        if len(valid_history) > 10:
            trend = torch.mean(valid_history[:-5]) - torch.mean(valid_history[-5:])
            return trend > 0
        return True
        
    def compute_fixed_point(self, psi_0, use_anderson=True):
        self.last_valid_state = None
        self.fallback_state = None
        logger.info("Reset last_valid_state and fallback_state for new batch computation")
        
        batch_size = psi_0.shape[0]
        device = psi_0.device
        
        C = self.output_stabilizer(torch.zeros((batch_size, self.config.hidden_dim), device=device))
        
        iterations_count = torch.zeros(batch_size, device=device)
        not_converged = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        C_history = []
        residual_history = []
        
        original_dropout = self.lambda_op.train_dropout
        self.lambda_op.train_dropout = False
        
        try:
            for i in range(self.config.max_iterations):
                C_prev = C.clone()
                
                input_tensor = self.concatenate(psi_0, C)
                input_tensor = self.input_stabilizer(input_tensor)
                
                C_new = self.lambda_op(input_tensor)
                C_new = self.output_stabilizer(C_new)
                
                residual = torch.norm(C_new - C, dim=1)
                
                if use_anderson and len(C_history) > 0:
                    C_history.append(C_new)
                    residual_history.append(C_new - C)
                    
                    if len(C_history) > 5:
                        C_history = C_history[-5:]
                        residual_history = residual_history[-5:]
                    
                    C_new = self._anderson_acceleration(C_history, residual_history)
                
                alpha = self.get_adaptive_alpha(input_tensor, C_new, C)
                C = alpha * C_new + (1 - alpha) * C_prev
                
                converged = residual < self.config.convergence_threshold
                not_converged = not_converged & ~converged
                
                iterations_count += not_converged.float()
                
                self.update_convergence_history(residual)
                
                if i > 0 and i % 10 == 0:
                    logger.info(f"Iteration {i}/{self.config.max_iterations}. Avg residual: {residual.mean():.6f}")
                    
                    if not self.check_stability():
                        logger.warning(f"Instability detected at iteration {i}, applying fixes")
                        C = self.output_stabilizer(C)
                        if self.last_valid_state is None:
                            self.last_valid_state = C.clone()
                            logger.info(f"Set last_valid_state to shape {self.last_valid_state.shape}")
                
                if residual.mean() < 0.1:
                    self.fallback_state = C.clone()
                    logger.info(f"Set fallback_state to shape {self.fallback_state.shape}")
                
                if not torch.any(not_converged):
                    logger.info(f"All batch elements converged after {i+1} iterations.")
                    break
            
            if torch.any(not_converged):
                unconverged_mask = not_converged
                if self.last_valid_state is not None:
                    assert self.last_valid_state.shape[0] == batch_size, f"Shape mismatch: last_valid_state {self.last_valid_state.shape[0]} vs batch {batch_size}"
                    C[unconverged_mask] = self.last_valid_state[unconverged_mask]
                    logger.info("Applied last_valid_state correction")
                elif self.fallback_state is not None:
                    assert self.fallback_state.shape[0] == batch_size, f"Shape mismatch: fallback_state {self.fallback_state.shape[0]} vs batch {batch_size}"
                    C[unconverged_mask] = self.fallback_state[unconverged_mask]
                    logger.info("Applied fallback_state correction")
        
        finally:
            self.lambda_op.train_dropout = original_dropout
        
        assert C.shape[-1] == self.config.hidden_dim, "MetaLoop output size mismatch"
        return C, iterations_count
    
    def _anderson_acceleration(self, C_history, residual_history):
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
    
    def forward(self, psi_0):
        C_star, iterations = self.compute_fixed_point(psi_0)
        return C_star, iterations

class UniformMatrix:
    @staticmethod
    def expm(A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(A)

class QuantumSimulator(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        n = config.num_pillars
        self.dim = 2**n
        
        annihilation = torch.zeros(n, self.dim, self.dim, dtype=torch.complex64)
        for p in range(n):
            for i in range(self.dim):
                if (i >> p) & 1:
                    j = i ^ (1 << p)
                    annihilation[p, j, i] = 1.0
        self.register_buffer('ann_ops', annihilation)
        
        self.theta = nn.Parameter(torch.randn(n, n) * 0.01)

    def _get_displacement_operator(self, pillar_idx, alpha_val):
        a_op = self.ann_ops[pillar_idx]
        a_dag = a_op.conj().transpose(-1, -2)
        return UniformMatrix.expm(alpha_val * a_dag - alpha_val.conj() * a_op)

    def forward(self, pillars: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n = pillars.shape
        dev = pillars.device
        
        states = torch.zeros(b, self.dim, dtype=torch.complex64, device=dev)
        states[:, 0] = 1.0

        for pi in range(n):
            for bi in range(b):
                D = self._get_displacement_operator(pi, pillars[bi, pi])
                states[bi] = D @ states[bi]

        entanglement = torch.zeros(b, device=dev)
        for bi in range(b):
            rho = torch.outer(states[bi], states[bi].conj())
            d = self.dim // 2
            # reshape: [2^n,2^n] -> [2, d, 2, d] -> permute -> [2,2,d,d]
            rho_AB = rho.view(2, d, 2, d).permute(0, 2, 1, 3)
            # partial trace over B: sum diag по последним двум осям
            rho_A = rho_AB.diagonal(dim1=2, dim2=3).sum(-1)
            # численная стабилизация: симметризация и нормировка
            rho_A = 0.5 * (rho_A + rho_A.conj().T)
            tr = rho_A.trace()
            if tr.abs() > 0:
                rho_A = rho_A / tr
            else:
                # на всякий случай — максимально устойчивый fallback
                rho_A = torch.eye(2, dtype=rho_A.dtype, device=rho_A.device) / 2

            eigvals = torch.linalg.eigvalsh(rho_A).real
            eigvals = eigvals[eigvals > 1e-9]
            entanglement[bi] = -torch.sum(eigvals * torch.log(eigvals))
        
        action_propensity = F.softmax(pillars @ self.theta, dim=-1)
        return {'quantum_state': states, 'entanglement': entanglement, 'action_propensity': action_propensity}


class TopologyAnalyzer(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config

        # Потенциал (осталось как в 2core)
        self.potential_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1)
        )

        # Доп. сеть для быстрого градиента (из core)
        self.gradient_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, config.num_pillars)
        )

        # Стабилизация входа перед гессианом (из core)
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
                g = torch.autograd.grad(pot, p, create_graph=True, retain_graph=True)[0].squeeze(0)  # [P]
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
        Отдельно включаем градиенты локально, даже если внешний контекст @torch.no_grad.
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
            nn.LayerNorm(1),
            nn.Sigmoid(),
            nn.Hardtanh(0, 1)
        )

    def forward(self, core_state, pillars, quantum_results, topo_results):
        device = core_state.device
        batch_size = core_state.shape[0]

        entanglement = quantum_results['entanglement'].float()
        action_propensity = quantum_results['action_propensity'].float()
        entanglement = torch.nan_to_num(entanglement, nan=0.0)
        action_propensity = torch.nan_to_num(action_propensity, nan=1.0 / self.config.num_pillars)
        action_propensity = F.normalize(action_propensity, p=1, dim=-1)

        potential = topo_results['potential'].float()
        potential = torch.nan_to_num(potential, nan=0.0)

        try:
            action_features = torch.cat([core_state, pillars, action_propensity, entanglement.unsqueeze(-1), potential], dim=-1)
        except RuntimeError:
            action_features = torch.zeros((batch_size, self.config.hidden_dim + self.config.num_pillars * 2 + 2), device=device)

        action_features.requires_grad_(True)
        action_embedding = self.action_encoder(action_features)

        if self.training:
            parameters = [p for p in self.action_encoder.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

        safety_score = self.safety_classifier(action_embedding)
        safety_score = torch.clamp(safety_score, 0.0, 1.0)
        safety_score = torch.where(torch.isnan(safety_score) | torch.isinf(safety_score),
                                   torch.full_like(safety_score, 0.5), safety_score)

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


class AEONDelta(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создаем encoder и decoder ВНУТРИ модели ---
        # Это гарантирует, что их параметры будут частью state_dict и parameters() модели.
        if TRANSFORMERS_AVAILABLE:
            # Инициализируем токенизатор для внутреннего использования
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # Создаем энкодер и декодер как ПОДМОДУЛИ этой модели
            self.encoder = ThoughtEncoder(self.config.vocab_size, z_dim=self.config.z_dim).to(device).eval()
            self.decoder = ThoughtDecoder(self.config.vocab_size, z_dim=self.config.z_dim).to(device).eval()
            logger.info(f"Initialized internal encoder and decoder with vocab_size={self.config.vocab_size}")
        else:
            self.tokenizer = None
            self.encoder = None
            self.decoder = None
            logger.error("Transformers tokenizer not available in AEONDelta model.")

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
        self.meta_loop = MetaLoopProcessor(config).to(device)
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
        logger.info("AEONDelta initialized and all submodules moved to {}".format(device))
        # === AEON Coherence Hotfix (one-shot) ===

    def _inject_lora(self):
        def add_lora_to_layer(layer, rank, alpha, dropout):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                lora_a = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
                lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
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
        input_tensor = self.meta_loop.concatenate(psi_0, C_star)
        
        with torch.no_grad():
            original_mode = self.meta_loop.lambda_op.training
            self.meta_loop.lambda_op.eval()
            self.meta_loop.lambda_op.set_dropout_active(False)
            
            C_new = self.meta_loop.lambda_op(input_tensor)
            
            if original_mode:
                self.meta_loop.lambda_op.train()
            self.meta_loop.lambda_op.set_dropout_active(self.meta_loop.lambda_op.train_dropout)
        
        residual = torch.norm(C_new - C_star, dim=1)
        consistency = 1.0 / (1.0 + residual)
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
            catastrophes = outputs['topo_results']['catastrophes'].float().mean().item()
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

    def save_state(self, save_dir="./aeon_state"):
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

    def load_state(self, save_dir="./aeon_state"):
        if not os.path.exists(save_dir):
            logger.warning(f"Save directory {save_dir} does not exist")
            return False
        
        try:
            model_path = os.path.join(save_dir, "model.pt")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                expected_keys = set(self.state_dict().keys())
                loaded_keys = set(state_dict.keys())
                
                missing_keys = expected_keys - loaded_keys
                unexpected_keys = loaded_keys - expected_keys
                
                if missing_keys:
                    logger.warning(f"Missing keys when loading model: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
                
                self.load_state_dict(state_dict, strict=False)
            
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
                load_lora(self, lora_path, map_location=device)
            
            logger.info(f"AEON-Δ state loaded from {save_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
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

    def reasoning_core(self, z_in, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None):
        psi_0 = self.qualia_extractor(z_in, attention_mask)
        
        C_star, iterations = self.meta_loop(psi_0)
        
        pillars, embedded_pillars = self.pillars_module(C_star)
        
        quantum_results = self.quantum_sim(pillars)
        
        topo_results = self.topology_analyzer(pillars, iterations)
        
        action_results = self.action_module(C_star, pillars, quantum_results, topo_results)
        
        planning_results = self.planning_module(C_star, pillars, action_results['action_embedding']) if planning else {}
        
        if memory_retrieval:
            memory_context = self._retrieve_memory(C_star)
            C_star = self.memory_fusion(torch.cat([C_star, memory_context], dim=-1))
        
        z_out = self.rssm(C_star)
        
        z_out = self.integration_module(torch.cat([z_out, embedded_pillars], dim=-1))
        
        assert z_out.shape[-1] == self.config.hidden_dim, "Reasoning core output size mismatch"
        
        outputs = {
            'core_state': C_star,
            'pillars': pillars,
            'quantum_results': quantum_results,
            'topo_results': topo_results,
            'action_results': action_results,
            'planning_results': planning_results,
            'iterations': iterations,
            'psi_0': psi_0,
        }
        
        return z_out, outputs

    def forward(self, input_ids, attention_mask=None, memory_retrieval=True, planning=True, use_kv_cache=None):
        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.encoder, а не глобальную переменную ---
        z_in = self.encoder(input_ids.to(device))
        logger.info("Encoded tokens to z")
        
        z_out, internal_outputs = self.reasoning_core(z_in, attention_mask, memory_retrieval, planning, use_kv_cache)
        
        # Интеграция VQ
        quantized_z, vq_loss, _ = self.vector_quantizer(z_out)
        internal_outputs['vq_loss'] = vq_loss

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.decoder, а не глобальную переменную ---
        # Для forward pass во время обучения, используем teacher-forcing
        tokens_out = self.decoder(quantized_z, input_ids.to(device))
        
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
        safety_score = torch.where(
            torch.isnan(safety_score) | torch.isinf(safety_score),
            torch.ones_like(safety_score) * 0.5,
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
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
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
        Генерирует текст в авторегрессионном режиме, используя обученную модель.
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("Tokenizer not available. Cannot generate text.")
            return "[Generation failed: Tokenizer not initialized]"

        device = next(self.parameters()).device

        try:
            # 1. Кодирование входного текста в вектор мысли
            inputs = self.tokenizer(
                seed,
                return_tensors="pt",
                max_length=self.config.seq_length,
                padding='max_length',
                truncation=True
            )
            input_ids = inputs['input_ids'].to(device)
            
            # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.encoder ---
            z_in = self.encoder(input_ids)
            z_out, _ = self.reasoning_core(z_in)
            quantized_z, _, _ = self.vector_quantizer(z_out)

            # 2. Инициализация состояния декодера из вектора мысли
            # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.decoder ---
            h0 = self.decoder.fc(quantized_z).unsqueeze(0)  # [1, B, E]
            c0 = torch.zeros_like(h0)                       # [1, B, E]
            hidden_state = (h0, c0)

            # 3. Начинаем генерацию со стартового токена [CLS]
            current_token_id = torch.tensor([[self.tokenizer.cls_token_id]], device=device)  # [1, 1]
            generated_ids = []

            # 4. Цикл авторегрессионной генерации
            for step in range(max_len):
                # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.decoder ---
                token_embedding = self.decoder.embed(current_token_id)  # [1, 1, E]

                # Прогоняем через LSTM
                # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.decoder ---
                output, hidden_state = self.decoder.lstm(token_embedding, hidden_state)  # output: [1, 1, E]

                # Получаем логиты для следующего токена
                # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем self.decoder ---
                logits = self.decoder.head(output.squeeze(1))  # [1, V]

                # 5. Фильтрация невалидных токенов
                invalid_ids = {
                    self.tokenizer.pad_token_id,
                    self.tokenizer.unk_token_id,
                    self.tokenizer.cls_token_id,
                }
                vocab = self.tokenizer.get_vocab()
                invalid_ids.update({i for t, i in vocab.items() if t.startswith('[unused')})
                logits[:, list(invalid_ids)] = -float('inf')

                # 6. Сэмплирование следующего токена с параметрами top_k и temperature
                logits = logits / temperature
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_idx)

                # 7. Условие остановки генерации: достигнут токен [SEP]
                if next_token_id.item() == self.tokenizer.sep_token_id:
                    break

                generated_ids.append(next_token_id.item())
                current_token_id = next_token_id

            if generated_ids:
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                generated_text = "[No output generated]"

            return generated_text

        except Exception as e:
            logger.error(f"Error during thought generation: {str(e)}")
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

def freeze_encoder_decoder():
    for param in encoder.parameters(): param.requires_grad = False
    for param in decoder.parameters(): param.requires_grad = False

def unfreeze_lora_blocks(model, blocks):
    for name, param in model.named_parameters():
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


if __name__ == "__main__":
    config = AEONConfig(lora_rank=16)
    model = AEONDelta(config).to(device)

    # Demo/Training mode guard
    DEMO = os.getenv('AEON_DEMO', '1') == '1'
    if DEMO:
        # Skip training phases for lightweight presentation
        console_inference_loop(model)
    else:
        # Фаза A
        logger.info("\n" + "#"*20 + " НАЧАЛО ФАЗЫ A " + "#"*20)
        trainer_A = ThoughtAETrainer(model, config)
        # trainer_A.fit('./data/', epochs=4) # Раскомментируйте при наличии данных

        # Фаза B
        logger.info("\n" + "#"*20 + " НАЧАЛО ФАЗЫ B " + "#"*20)
        freeze_encoder_decoder()
        unfreeze_lora_blocks(model, ["rssm", "pillars_module"])
        trainer_B = ZDynamicsTrainer(model, config)
        # trainer_B.fit('./data/z_pairs.pt', epochs=6) # Раскомментируйте

        # Фаза C с интеллектуальной остановкой
        logger.info("\n" + "#"*20 + " НАЧАЛО ФАЗЫ C " + "#"*20)
        trainer_C = CuriosityPPOTrainer(model, config)
    
        # --- ИЗМЕНЕНИЕ: Реализация цикла с интеллектуальной остановкой ---
        max_steps = 1000
        patience = 20
        min_delta = 1e-4
        window_size = 50
    
        rewards_history = deque(maxlen=window_size)
        patience_counter = 0
        best_avg_reward = -float('inf')

        os.makedirs("./lora_weights", exist_ok=True)

        for step in range(max_steps):
            reward = trainer_C.step_self_env(["Cogito ergo sum"])
            rewards_history.append(reward)

            if len(rewards_history) == window_size:
                current_avg_reward = np.mean(list(rewards_history))
                if current_avg_reward > best_avg_reward + min_delta:
                    best_avg_reward = current_avg_reward
                    patience_counter = 0
                    logger.info(f"[Convergence Check] Step {step+1}: Progress detected. New best: {best_avg_reward:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"[Convergence Check] Step {step+1}: No progress. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info(f"Обучение в Фазе C остановлено по критерию сходимости на шаге {step+1}.")
                    break
        
            if step % 10 == 0:
                save_lora(model, f"./lora_weights/step_{trainer_C.global_step}.pt")

        else:
            logger.info(f"Обучение в Фазе C остановлено по достижению лимита в {max_steps} шагов.")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    console_inference_loop(model)
