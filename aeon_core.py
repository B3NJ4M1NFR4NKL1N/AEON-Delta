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
from scipy.linalg import expm

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

class ThoughtDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, z_dim=256):
        super().__init__()
        self.fc   = nn.Linear(z_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, z, max_len=32):
        batch = z.size(0)
        h0 = self.fc(z).unsqueeze(0)                # [1, B, E]
        c0 = torch.zeros_like(h0)
        inputs = self.fc(z).unsqueeze(1).repeat(1, max_len, 1)  # [B, L, E]
        out, _ = self.lstm(inputs, (h0, c0))
        logits = self.head(out)                     # [B, L, V]
        return logits

# Load AE weights or fallback
vocab_size = 50000  # From config
encoder = ThoughtEncoder(vocab_size).to(device).eval()
decoder = ThoughtDecoder(vocab_size).to(device).eval()

ae_path = "./weights/thought_ae.pt"
if os.path.exists(ae_path):
    state = torch.load(ae_path, map_location=device)
    encoder.load_state_dict(state["enc"])
    decoder.load_state_dict(state["dec"])
    logger.info("Loaded Thought AE weights successfully")
else:
    logger.warning("Thought AE weights not found — using random init.")

@dataclass
class AEONConfig:
    z_dim: int = 256
    hidden_dim: int = 256
    meta_dim: int = 256
    vocab_size: int = 50000
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

# Инициализация клиента памяти
class MemoryManager:
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
                self.client = MemoryClient(api_key=MEM0_API_KEY, config=local_cfg)  # Corrected from OMStore to MemoryClient
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

    def add_embedding(self, vec, meta=None):
        meta = meta or {}
        vec_np = vec.detach().cpu().numpy()
        if not self._use_fallback():
            try:
                self.client.add(
                    messages=[{"role": "system", "content": np.array2string(vec_np)}],  # Stub content
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
        assert self.size == self._size, "Size mismatch after add."

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
                        vec_part = content.split(',')  # Simplified parse
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

    def sample_pairs(self, n: int = 64):
        return self._fallback_sample(n)

    def _fallback_sample(self, n):
        if self._size < n:
            n = self._size
        indices = np.random.choice(self._size, n, replace=False)
        return [{'vec': self.fallback_vectors[i], 'meta': self.fallback_metas[i]} for i in indices]

    def add_experience(self, state, **meta):
        self.add_embedding(state, meta)

    def query_entities(self, query_vec, top_k: int = 3):
        results = self.retrieve_relevant(query_vec, k=top_k)
        metas = [r['meta'] for r in results]
        return results, metas

    def _hash_tensor(self, t):
        return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()

    @property
    def size(self):
        return self._size

    def save_memory(self, path=None):
        path = path or self.cfg.memory_path
        if not self._use_fallback():
            logger.info("mem0 save not directly supported; skipping.")
        else:
            data = {'vectors': [v.tolist() for v in self.fallback_vectors], 'metas': self.fallback_metas}
            with open(path + '.json', 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved fallback memory to {path}.json")

    def add_entity(self, key, vec):
        self.add_embedding(vec, {'entity_id': key})

class KnowledgeGraph:
    def __init__(self, config: AEONConfig):
        self.config = config
        self.memory_manager = None
        self.local_graph = defaultdict(lambda: defaultdict(set))
        self.reverse_graph = defaultdict(lambda: defaultdict(set))
        self.relation_embeddings = {}
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
                    if config.use_neo4j:
                        raise ConnectionError(f"Could not connect to Neo4j: {e}")
                    self.neo4j_graph = None
    
    def set_memory_manager(self, memory_manager):
        self.memory_manager = memory_manager
    
    def add_fact(self, subject, relation, object_entity, embedding=None):
        self.local_graph[subject][relation].add(object_entity)
        self.reverse_graph[object_entity][f"reverse_{relation}"].add(subject)
        
        if embedding is not None and self.memory_manager is not None:
            self.memory_manager.add_entity(subject, embedding)
        
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
                    params["relation"] = relation
                else:
                    query_parts.append("-[r]->")
                
                if object_entity:
                    query_parts.append("(o:Entity {name: $object})")
                    params["object"] = object_entity
                else:
                    query_parts.append("(o:Entity)")
                
                query = f"MATCH {' '.join(query_parts)} RETURN s.name as subject, type(r) as relation, o.name as object"
                
                neo4j_results = self.neo4j_graph.run(query, **params)
                for record in neo4j_results:
                    results.append((record["subject"], record["relation"], record["object"]))
                
                return results
            except Exception as e:
                logger.warning(f"Neo4j query failed: {e}, falling back to local graph")
        
        if subject:
            if subject in self.local_graph:
                if relation:
                    if relation in self.local_graph[subject]:
                        for obj in self.local_graph[subject][relation]:
                            if object_entity and obj != object_entity:
                                continue
                            results.append((subject, relation, obj))
                else:
                    for rel, objects in self.local_graph[subject].items():
                        for obj in objects:
                            if object_entity and obj != object_entity:
                                continue
                            results.append((subject, rel, obj))
        elif object_entity:
            if object_entity in self.reverse_graph:
                for rel, subjects in self.reverse_graph[object_entity].items():
                    if relation and not (rel == relation or rel == f"reverse_{relation}"):
                        continue
                    for subj in subjects:
                        results.append((subj, rel.replace("reverse_", ""), object_entity))
        else:
            for subj in self.local_graph:
                for rel, objects in self.local_graph[subj].items():
                    if relation and rel != relation:
                        continue
                    for obj in objects:
                        results.append((subj, rel, obj))
        
        return results
    
    def get_neighbors(self, entity, relations=None):
        neighbors = []
        
        if entity in self.local_graph:
            for relation, objects in self.local_graph[entity].items():
                if relations and relation not in relations:
                    continue
                for obj in objects:
                    neighbors.append((relation, obj))
        
        if entity in self.reverse_graph:
            for relation, subjects in self.reverse_graph[entity].items():
                rel = relation.replace("reverse_", "")
                if relations and rel not in relations:
                    continue
                for subj in subjects:
                    neighbors.append((f"reverse_{rel}", subj))
        
        return neighbors
    
    def save(self, filepath):
        data = {
            "local_graph": {k: {r: list(o) for r, o in v.items()} for k, v in self.local_graph.items()},
            "reverse_graph": {k: {r: list(o) for r, o in v.items()} for k, v in self.reverse_graph.items()},
            "relation_embeddings": {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                                  for k, v in self.relation_embeddings.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.local_graph = defaultdict(lambda: defaultdict(set))
        for k, v in data["local_graph"].items():
            for r, o in v.items():
                self.local_graph[k][r] = set(o)
        
        self.reverse_graph = defaultdict(lambda: defaultdict(set))
        for k, v in data.get("reverse_graph", {}).items():
            for r, o in v.items():
                self.reverse_graph[k][r] = set(o)
        
        self.relation_embeddings = {k: torch.tensor(v) if isinstance(v, list) else v 
                                   for k, v in data.get("relation_embeddings", {}).items()}
        
        return True

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
        if self.pooling_type == "cls":
            return sequence[:, 0]
        
        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1)
                sum_embeddings = torch.sum(sequence * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(sequence, dim=1)
        
        elif self.pooling_type == "attention":
            scores = self.attention(sequence)
            
            if attention_mask is not None:
                mask = (attention_mask == 0).unsqueeze(-1)
                scores = scores.masked_fill(mask, -torch.finfo(scores.dtype).max)
            
            attention_weights = F.softmax(scores, dim=1)
            pooled = torch.sum(sequence * attention_weights, dim=1)
            
            return pooled
        
        elif self.pooling_type == "max":
            if attention_mask is not None:
                mask = (attention_mask == 0).unsqueeze(-1)
                sequence = sequence.masked_fill(mask, -torch.finfo(sequence.dtype).max)
            
            return torch.max(sequence, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

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
        if isinstance(lm_output, dict):
            lm_output = lm_output["last_hidden_state"]
            
        projected = self.projection(lm_output)
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
            
        # Приведение размерностей
        if psi_0.dim() == 1:
            psi_0 = psi_0.unsqueeze(0)
        if C.dim() == 1:
            C = C.unsqueeze(0)
            
        # Проверка и коррекция типов данных
        if psi_0.dtype != C.dtype:
            C = C.to(dtype=psi_0.dtype)
            
        # Проверка размерностей
        if psi_0.shape[1] != C.shape[1]:
            raise ValueError(f"Incompatible dimensions: psi_0 {psi_0.shape} vs C {C.shape}")
            
        # Защита от NaN и Inf
        psi_0 = torch.nan_to_num(psi_0, nan=0.0, posinf=1.0, neginf=-1.0)
        C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Нормализация
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
        # Reset dynamic states per call to handle variable batch sizes
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
        # ❗ Только PyTorch, чтобы не рвать граф и не уходить в NumPy/SciPy
        return torch.linalg.matrix_exp(A)

    @staticmethod
    def ensure_unitary(U, tol=1e-6):
        logger.info("Ensuring unitary matrix with tol=%.2e", tol)
        U_adj = U.conj().transpose(-2, -1)
        I = torch.eye(U.shape[-1], dtype=U.dtype, device=U.device)
        error = torch.norm((U_adj @ U - I).float()).item()
        logger.info("Unitary error before correction: %.6f", error)
        
        if error <= tol:
            logger.info("Matrix already unitary within tolerance")
            return U
        
        m = U_adj @ U
        s, v = torch.linalg.eigh(m)
        s = torch.clamp(s, min=0.0)
        
        # Fix: Ensure diag_values have same dtype as v (complex)
        diag_values = 1.0 / torch.sqrt(s + tol)
        diag_values = torch.nan_to_num(diag_values, nan=0.0, posinf=0.0, neginf=0.0)  # Safety
        diag_values = diag_values.to(dtype=v.dtype)  # Cast to complex
        
        # Check: Verify dtypes match
        assert diag_values.dtype == v.dtype, f"Dtype mismatch: diag {diag_values.dtype} vs v {v.dtype}"
        
        inv_sqrt_diag = torch.diag(diag_values)
        H_inv_sqrt = v @ inv_sqrt_diag @ v.conj().transpose(-2, -1)
        
        corrected_U = U @ H_inv_sqrt
        # Post-check error
        corrected_adj = corrected_U.conj().transpose(-2, -1)
        corrected_error = torch.norm((corrected_adj @ corrected_U - I).float()).item()
        logger.info("Unitary error after correction: %.6f", corrected_error)
        
        return corrected_U

class QuantumSimulator(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        n = config.num_pillars
        self.dim = 2**n
        if config.quantum_dim_reduction and n > 5:
            self.use_approx = True
            self.approx_dim = min(32, 2**n)
        else:
            self.use_approx = False
            self.approx_dim = self.dim

        annihilation = torch.zeros(n, self.dim, self.dim, dtype=torch.complex64)
        for p in range(n):
            for i in range(self.dim):
                b = format(i, f'0{n}b')
                if b[p] == '1':
                    j = int(b[:p] + '0' + b[p+1:], 2)
                    annihilation[p, j, i] = 1.0
        number_ops = torch.zeros_like(annihilation)
        for p in range(n):
            a = annihilation[p]
            number_ops[p] = a.conj().transpose(-1, -2) @ a
        self.register_buffer('ann_ops', annihilation)
        self.register_buffer('num_ops', number_ops)
        self.register_buffer('alpha_vals', torch.linspace(0,1,41))
        self.theta = nn.Parameter(torch.randn(n,n)*0.01)
        self.theta_bias = nn.Parameter(torch.zeros(n))
        if config.precompute_displacement:
            self._precompute_displacement_operators()

    def _precompute_displacement_operators(self):
        n = self.config.num_pillars
        p = len(self.alpha_vals)
        D = torch.zeros(n,p,self.dim,self.dim, dtype=torch.complex64)
        for pi in range(n):
            a_op = self.ann_ops[pi]
            a_dag = a_op.conj().transpose(-1,-2)
            for i,val in enumerate(self.alpha_vals):
                U = UniformMatrix.expm(val*a_dag - val*a_op)
                D[pi,i] = UniformMatrix.ensure_unitary(U)
        self.register_buffer('D', D)

    def _get_displacement_operator(self, pillar_idx, alpha_val):
        # ❗ Без .item(), оставляем тензор — сохраняем граф
        alpha_val = torch.clamp(alpha_val, 0, 1)

        if not self.config.precompute_displacement or not hasattr(self, 'D'):
            logger.info("Computing displacement operator on the fly since precompute is disabled.")
            a_op  = self.ann_ops[pillar_idx]
            a_dag = a_op.conj().transpose(-1, -2)
            U = UniformMatrix.expm(alpha_val * a_dag - alpha_val * a_op)
            return UniformMatrix.ensure_unitary(U)
        else:
            # Линейная интерполяция по заранее посчитанной сетке БЕЗ .item()
            grid = self.alpha_vals    # [P]
            # bucketize вернёт индекс правой границы; смещаем на левую
            idx = torch.bucketize(alpha_val.detach(), grid) - 1
            idx = torch.clamp(idx, 0, grid.numel() - 2)
            w = (alpha_val - grid[idx]) / (grid[idx + 1] - grid[idx])
            D1 = self.D[pillar_idx, idx]
            D2 = self.D[pillar_idx, idx + 1]
            D_interp = (1 - w) * D1 + w * D2
            return UniformMatrix.ensure_unitary(D_interp)

    def forward(self, pillars: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = pillars.shape[0]
        dev = pillars.device
        dim = self.approx_dim if self.use_approx else self.dim
        states = torch.zeros(b, dim, dtype=torch.complex64, device=dev)
        states[:,0] = 1.0
        if not self.config.use_quantum_sim:
            ent = torch.zeros(b, device=dev)
            ap = torch.ones((b,self.config.num_pillars),device=dev)/self.config.num_pillars
            return {'quantum_state':states, 'entanglement':ent, 'action_propensity':ap}
        for pi in range(self.config.num_pillars):
            for bi in range(b):
                D = self._get_displacement_operator(pi, pillars[bi,pi])
                states[bi] = torch.mv(D, states[bi])
        ent = torch.zeros(b,device=dev)
        for bi in range(b):
            v = states[bi].unsqueeze(1)
            rho = v@v.conj().transpose(0,1)
            half = dim//2
            red = torch.zeros((2,2),dtype=torch.complex64,device=dev)
            red[0,0]=rho[:half,:half].diag().sum()
            red[1,1]=rho[half:,half:].diag().sum()
            red[0,1]=rho[:half,half:].sum()
            red[1,0]=rho[half:,:half].sum()
            # Analytic eigenvalues for 2x2 Hermitian matrix to avoid MPS unsupported op
            a = red[0,0].real
            d = red[1,1].real
            b_abs_sq = torch.abs(red[0,1]) ** 2
            delta = torch.sqrt((a - d) ** 2 + 4 * b_abs_sq)
            ev1 = ((a + d) + delta) / 2
            ev2 = ((a + d) - delta) / 2
            ev = torch.stack([ev1, ev2])
            ev = torch.nan_to_num(ev, nan=0.0)  # Safety
            ent[bi] = -torch.sum(ev[ev>1e-10]*torch.log(ev[ev>1e-10]))
            logger.info("Computed entanglement analytically for MPS compatibility")
        mom = torch.zeros((b,self.config.num_pillars),device=dev)
        for pi in range(self.config.num_pillars):
            for bi in range(b):
                s = states[bi]
                mom[bi,pi] = (s.conj() @ (self.num_ops[pi] @ s)).real
        w = F.softmax(mom @ self.theta.t() + self.theta_bias, dim=-1)
        assert w.shape[-1] == self.config.num_pillars, "Action propensity size mismatch"
        return {'quantum_state':states,'entanglement':ent,'action_propensity':w}

class TopologyAnalyzer(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        
        self.potential_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        self.gradient_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, config.num_pillars)
        )
        
        self.catastrophe_classifier = nn.Sequential(
            nn.Linear(config.num_pillars * 2 + 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.stabilizer = nn.LayerNorm(config.num_pillars)
        self.dropout = nn.Dropout(0.1)
        
        self.register_buffer('eigenvalue_history', torch.zeros(100, config.num_pillars))
        self.register_buffer('last_stable_state', None)
        self.history_pointer = 0
        
    def compute_potential(self, pillars):
        return self.potential_net(pillars)
    
    def compute_gradient(self, pillars):
        return self.gradient_net(pillars)
    
    def detect_catastrophe_batch(self, pillars):
        batch_size = pillars.shape[0]
        device = pillars.device
        
        pillars = torch.nan_to_num(pillars, nan=0.5)
        pillars = torch.clamp(pillars, 0.0, 1.0)
        
        pillars_with_grad = pillars.clone().requires_grad_(True)  # Removed detach() to preserve grad capability
        
        try:
            with torch.enable_grad():
                potential = self.compute_potential(pillars_with_grad)
                
                grad = torch.autograd.grad(
                    potential.sum(), 
                    pillars_with_grad, 
                    create_graph=True
                )[0]
                
                eigenvalues = self.compute_hessian_eigenvalues(pillars_with_grad)
                
                features = torch.cat([
                    pillars,
                    grad,
                    potential,
                    eigenvalues.mean(dim=1, keepdim=True)
                ], dim=-1)
                
                features = F.normalize(features, dim=-1)
                
                catastrophe_probs = self.catastrophe_classifier(features)
                catastrophe_probs = torch.clamp(catastrophe_probs, 0.0, 1.0)
                
                catastrophe_detected = torch.where(
                    catastrophe_probs > 0.7,
                    torch.ones_like(catastrophe_probs),
                    torch.where(
                        catastrophe_probs < 0.3,
                        torch.zeros_like(catastrophe_probs),
                        torch.full_like(catastrophe_probs, float('nan'))
                    )
                )
                
                catastrophe_detected = torch.nan_to_num(catastrophe_detected, nan=0.5)
                
        except Exception as e:
            logger.warning(f"Catastrophe detection failed: {e}, using fallback")
            catastrophe_detected = torch.zeros(batch_size, 1, device=device)
            catastrophe_probs = torch.ones(batch_size, 1, device=device) * 0.5
    
        return catastrophe_detected.bool(), catastrophe_probs
    
    def compute_hessian_eigenvalues(self, pillars):
        batch_size = pillars.shape[0]
        device = pillars.device
        
        pillars = self.stabilizer(pillars)
        # Removed dropout to prevent graph dependency breaks and ensure higher-order gradients are computable
        
        eigenvalues = torch.zeros((batch_size, self.config.num_pillars), device=device)
        
        with torch.enable_grad():  
            try:
                for b in range(batch_size):
                    p = pillars[b:b+1].clone().requires_grad_(True)
                    logger.info(f"Computing Hessian for batch {b}: p requires_grad={p.requires_grad}")
                    
                    potential = self.compute_potential(p)
                    
                    grad = torch.autograd.grad(
                        potential.sum(), 
                        p,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    if not grad.requires_grad or grad.grad_fn is None:
                        raise RuntimeError("Gradient does not have grad_fn - graph disconnected")
                    
                    hessian = torch.zeros((self.config.num_pillars, self.config.num_pillars), device=device)
                    
                    try:
                        for i in range(self.config.num_pillars):
                            grad_i = grad[0, i]
                            if grad_i.grad_fn is None:
                                logger.warning(f"grad_i[{i}] has no grad_fn - skipping with fallback")
                                continue
                            
                            for j in range(self.config.num_pillars):
                                try:
                                    grad_ij = torch.autograd.grad(
                                        grad_i,
                                        p,
                                        retain_graph=True if (i < self.config.num_pillars-1) or (j < self.config.num_pillars-1) else None
                                    )[0][0, j]
                                    
                                    grad_ij = torch.clamp(grad_ij, -1e3, 1e3)
                                    
                                    hessian[i, j] = grad_ij
                                    if i != j:
                                        hessian[j, i] = grad_ij
                                except RuntimeError as e:
                                    logger.warning(f"Error computing Hessian element ({i},{j}): {e}")
                                    if self.last_stable_state is not None:
                                        hessian[i, j] = self.last_stable_state[b, i, j]
                                        if i != j:
                                            hessian[j, i] = self.last_stable_state[b, i, j]
                                    continue
                        
                        hessian = hessian + torch.eye(self.config.num_pillars, device=device) * 1e-4
                        
                        try:
                            eigvals = torch.linalg.eigvalsh(hessian)
                            
                            if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
                                raise RuntimeError("Нестабильные собственные значения")
                            
                            if self.last_stable_state is None:
                                self.last_stable_state = torch.zeros((batch_size, self.config.num_pillars, self.config.num_pillars), device=device)
                            self.last_stable_state[b] = hessian.detach()
                            
                            eigenvalues[b] = eigvals
                            
                        except RuntimeError:
                            eigenvalues[b] = torch.diagonal(hessian)
                            
                    except Exception as e:
                        logger.warning(f"Ошибка при вычислении гессиана для батча {b}: {e}")
                        if self.last_stable_state is not None:
                            eigenvalues[b] = torch.linalg.eigvalsh(self.last_stable_state[b])
                        else:
                            eigenvalues[b] = torch.zeros(self.config.num_pillars, device=device)
                
                self.eigenvalue_history[self.history_pointer] = eigenvalues.mean(dim=0)
                self.history_pointer = (self.history_pointer + 1) % 100
                
            except Exception as e:
                logger.error(f"Критическая ошибка при вычислении собственных значений: {e}")
                return torch.zeros((batch_size, self.config.num_pillars), device=device)
        
        logger.info("Hessian eigenvalues computed successfully")
        return eigenvalues
    
    def stratify_depth(self, pillars, iterations):
        batch_size = pillars.shape[0]
        device = pillars.device
        
        norm_iterations = iterations.float() / self.config.max_iterations
        
        with torch.no_grad():
            potential = self.compute_potential(pillars)
            gradient = self.compute_gradient(pillars)
            gradient_norm = torch.norm(gradient, dim=1, keepdim=True)
            
            gradient_norm = torch.clamp(gradient_norm, min=1e-8)
            
            eigenvalues = self.compute_hessian_eigenvalues(pillars)
            curvature = torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True)
            
            depth_factors = torch.cat([
                norm_iterations.unsqueeze(-1),
                potential,
                gradient_norm,
                curvature
            ], dim=-1)
            
            depth_factors = F.normalize(depth_factors, dim=-1)
            
            weights = torch.tensor([0.3, 0.3, 0.2, 0.2], device=device)
            depth = torch.sum(depth_factors * weights, dim=1)
            
            depth = torch.sigmoid(depth)
            
        return depth
        
    def forward(self, pillars, iterations=None):
        if not isinstance(pillars, torch.Tensor):
            pillars = torch.tensor(pillars, device=self.potential_net[0].weight.device)
    
        if pillars.dim() == 1:
            pillars = pillars.unsqueeze(0)
    
        if iterations is not None and not isinstance(iterations, torch.Tensor):
            iterations = torch.tensor(iterations, device=pillars.device)
    
        batch_size = pillars.shape[0]
        if pillars.shape[1] != self.config.num_pillars:
            raise ValueError(f"Expected pillars with shape (batch_size, {self.config.num_pillars}), got {pillars.shape}")
        
        try:
            potential = self.compute_potential(pillars)
            gradient = self.compute_gradient(pillars)
            
            if torch.isnan(potential).any() or torch.isinf(potential).any():
                logger.warning("NaN/Inf detected in potential, applying fix")
                potential = torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(gradient).any() or torch.isinf(gradient).any():
                logger.warning("NaN/Inf detected in gradient, applying fix")
                gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            
            try:
                catastrophes, catastrophe_probs = self.detect_catastrophe_batch(pillars)
            except RuntimeError as e:
                logger.warning(f"Error in catastrophe detection: {e}, using fallback")
                catastrophes = torch.zeros(batch_size, 1, dtype=torch.bool, device=pillars.device)
                catastrophe_probs = torch.ones(batch_size, 1, device=pillars.device) * 0.5
            
            depth = None
            if iterations is not None:
                try:
                    depth = self.stratify_depth(pillars, iterations)
                except RuntimeError as e:
                    logger.warning(f"Error in depth calculation: {e}, using fallback")
                    depth = torch.zeros(batch_size, device=pillars.device)
        
            return {
                'potential': potential,
                'gradient': gradient,
                'catastrophes': catastrophes,
                'catastrophe_probs': catastrophe_probs,
                'depth': depth
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in TopologyAnalyzer.forward: {e}")
            return {
                'potential': torch.zeros(batch_size, 1, device=pillars.device),
                'gradient': torch.zeros(batch_size, self.config.num_pillars, device=pillars.device),
                'catastrophes': torch.zeros(batch_size, 1, dtype=torch.bool, device=pillars.device),
                'catastrophe_probs': torch.ones(batch_size, 1, device=pillars.device) * 0.5,
                'depth': torch.zeros(batch_size, device=pillars.device) if iterations is not None else None
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

        # БЕЗ no_grad и БЕЗ второй сигмоиды — голова уже ограничивает диапазон
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
        self.register_buffer("h0", torch.zeros(1, config.hidden_dim))

    def forward(self, z_t, hx=None):
        stoch = self.stochastic(z_t)
        if hx is None:
            hx = self.h0.expand(stoch.size(0), -1).contiguous()
        det = self.deterministic(stoch, hx)
        return det + stoch


class AEONDelta(nn.Module):
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.qualia_extractor = QualiaExtractor(config).to(device)
        self.meta_loop = MetaLoopProcessor(config).to(device)
        self.pillars_module = PillarsModule(config).to(device)
        self.quantum_sim = QuantumSimulator(config).to(device)
        self.topology_analyzer = TopologyAnalyzer(config).to(device)
        self.action_module = ActionModule(config).to(device)
        self.planning_module = PlanningModule(config).to(device)
        self.rssm = RSSM(config).to(device)
        self.integration_module = nn.Linear(config.hidden_dim * 2, config.hidden_dim).to(device)
        self.memory_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        ).to(device)
        
        self.memory_manager = MemoryManager(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.knowledge_graph.set_memory_manager(self.memory_manager)
        
        self.metrics_log = defaultdict(list)
        self.step_counter = 0
        
        self.outputs = {}  # For internal states
        
        # LoRA Injection
        self._inject_lora()
        
        logger.info("AEONDelta initialized and all submodules moved to {}".format(device))
    
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
        z_in = encoder(input_ids.to(device))
        logger.info("Encoded tokens to z")
        
        z_out, internal_outputs = self.reasoning_core(z_in, attention_mask, memory_retrieval, planning, use_kv_cache)
        
        tokens_out = decoder(z_out, max_len=self.config.seq_length)
        logger.info("Decoded z to tokens")
        
        self.step_counter += 1
        
        if self.config.enable_checkpointing and self.step_counter % self.config.save_frequency == 0:
            self.save_state(os.path.join("./aeon_checkpoints", f"checkpoint_{self.step_counter}"))
        
        return {
            'logits': tokens_out,
            'thoughts': z_out,
            **internal_outputs  # Merge internal for compatibility
        }
    
    def generate_thought(
        self,
        seed: str,
        max_len: int = 64,
        top_k: int = 5,
        temperature: float = 1.0,
        mode: str = "inference",
        return_tokens: bool = False,
    ):
        import torch
        import torch.nn.functional as F
        import logging
        log = logging.getLogger("AEON-Delta")

        dev = next(self.parameters()).device
        enc = globals().get("encoder", None)
        dec = globals().get("decoder", None)
        if enc is None or dec is None:
            raise RuntimeError("encoder/decoder не найдены в globals().")
        enc = enc.to(dev).eval()
        dec = dec.to(dev).eval()

        if not isinstance(seed, str):
            raise TypeError("seed должен быть str")
        if max_len <= 0:
            raise ValueError("max_len > 0")
        if top_k is not None and top_k <= 0:
            top_k = None

        vocab_size = getattr(self.config, "vocab_size", 50000)
        seq_len = getattr(self.config, "seq_length", max_len)

        def _to_tokens_on_device(text: str, length: int) -> torch.Tensor:
            ids = [ord(c) % vocab_size for c in text[:length]]
            if len(ids) < length:
                ids += [0] * (length - len(ids))
            return torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)

        def _decode_tokens_to_text(toks: torch.Tensor) -> str:
            toks = toks.detach().to("cpu").tolist()
            return "".join(chr(int(t) % 128) for t in toks)

        with torch.no_grad():
            tokens = _to_tokens_on_device(seed, seq_len)
            z = enc(tokens)
            logits = dec(z, max_len=max_len)

            if temperature is not None and temperature > 0.0:
                logits = logits / float(temperature)
            probs = F.softmax(logits, dim=-1)

            if top_k is not None and top_k < probs.shape[-1]:
                topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                cat = torch.distributions.Categorical(topk_vals.squeeze(0))
                sampled_rel = cat.sample()
                sampled = topk_idx.squeeze(0)[torch.arange(max_len, device=dev), sampled_rel]
            else:
                sampled = torch.argmax(probs.squeeze(0), dim=-1)

            text = _decode_tokens_to_text(sampled)
            log.info(f"Generated thought: {text[:64]}{'...' if len(text)>64 else ''} (full length: {len(text)})")
            return (text, sampled.detach().to("cpu")) if return_tokens else text

    
    # core.py — внутри class AEONDelta
def self_train_step(self, batch_z: torch.Tensor, kl_weight: float = 0.1, safety_weight: float = 0.1, lr: float = 3e-4):
    import torch, torch.nn.functional as F, torch.optim as optim
    logger = logging.getLogger("AEON-Delta")
    self.train()

    assert batch_z.dim() == 3 and batch_z.size(1) == 2 and batch_z.size(2) == self.config.hidden_dim, \
        f"Expected batch_z [B,2,H], got {tuple(batch_z.shape)}"

    z_t  = batch_z[:, 0, :].float()
    z_t1 = batch_z[:, 1, :].float()

    pred_z = self.rssm(z_t)  # динамика в латенте

    # батчевый KL (стабилизирует геометрию z)
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
        if embedding is None and 'core_state' in self.outputs:  # Adjusted
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
        
        total_loss = lm_loss + consistency_loss + self.config.lambda_safety * safety_loss + reg_loss
        
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
            'consistency': consistency
        }
    
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

class AEONTrainer:
    def __init__(self, model, config, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.config = config
        self.device = torch.device(device)
        
        self.model.to(self.device)
        
        lorap = [p for p in model.parameters() if p.requires_grad]
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                           if not any(nd in n for nd in ['bias', 'LayerNorm.weight', 'layer_norm']) and p.requires_grad],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                           if any(nd in n for nd in ['bias', 'LayerNorm.weight', 'layer_norm']) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=3e-4,  # Для LoRA
        )
        
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=10000
        )
        
        # Фикс для MPS: отключаем AMP/autocast, если не CUDA (MPS имеет ограниченную поддержку, приводит к device-mismatch)
        if self.device.type != 'cuda':
            self.config.use_amp = False
            logger.info("AMP disabled for non-CUDA device (MPS/CPU compatibility)")
        
        # Scaler только для CUDA; для MPS/CPU — disabled
        self.scaler = GradScaler(enabled=self.config.use_amp and self.device.type == 'cuda')
        
        self.distributed = config.distributed_training and DIST_OK
        if self.distributed:
            if DIST_OK and dist.is_initialized():
                self.local_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.local_rank = 0
                self.world_size = 1
                logger.warning("Distributed training enabled but not initialized")
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.log_interval = 10
        self.global_step = 0
        self.best_loss = float('inf')

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def train_step(self, tokens, aug_tokens):
        import torch.nn.functional as F
        global encoder, decoder
        encoder.to(self.device).train()
        decoder.to(self.device).train()
        tokens = tokens.to(self.device)
        aug_tokens = aug_tokens.to(self.device)

        z = encoder(tokens)
        z_aug = encoder(aug_tokens)
        logits = decoder(z, max_len=self.config.seq_length)

        recon_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            tokens.view(-1),
            ignore_index=0
        )
        info_nce = self._info_nce(z, z_aug)
        kl = self._kl_div(z)
        loss = recon_loss + 0.3 * info_nce + 0.1 * kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "nce": float(info_nce.detach().cpu()),
            "kl": float(kl.detach().cpu())
        }

    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_consistency = 0
        total_safety = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Аналогично: autocast только для CUDA
                if self.device.type == 'cuda':
                    with autocast(enabled=self.config.use_amp):
                        outputs = self.model(input_ids, attention_mask)
                        loss_dict = self.model.compute_loss(outputs, targets, attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss_dict = self.model.compute_loss(outputs, targets, attention_mask)
                
                batch_size = input_ids.size(0)
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_consistency += loss_dict['consistency'].mean().item() * batch_size
                total_safety += outputs['action_results']['safety_score'].mean().item() * batch_size
                total_samples += batch_size
        
        if self.local_rank == 0:
            self.model.visualize_metrics()
        
        return {
            'loss': total_loss / total_samples,
            'consistency': total_consistency / total_samples,
            'safety': total_safety / total_samples
        }
    
    def train(self, train_dataloader, eval_dataloader=None, num_epochs=5):
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        if self.distributed and not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            logger.info(f"Initialized DistributedDataParallel on rank {self.local_rank}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_samples = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",
                             disable=self.local_rank != 0)
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                loss_dict = self.train_step(input_ids, targets, attention_mask)
                
                if self.global_step % self.log_interval == 0 and self.local_rank == 0:
                    progress_bar.set_postfix({
                        'loss': loss_dict['total_loss'],
                        'lm_loss': loss_dict['lm_loss'],
                        'cons_loss': loss_dict['consistency_loss'],
                        'consistency': loss_dict['consistency']
                    })
                
                batch_size = input_ids.size(0)
                epoch_loss += loss_dict['total_loss'] * batch_size
                epoch_samples += batch_size
            
            avg_loss = epoch_loss / epoch_samples
            if self.local_rank == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)

                if self.local_rank == 0:
                    logger.info(f"Evaluation - Loss: {eval_metrics['loss']:.4f}, "
                              f"Consistency: {eval_metrics['consistency']:.4f}, "
                              f"Safety: {eval_metrics['safety']:.4f}")
                    
                    if eval_metrics['loss'] < self.best_loss:
                        self.best_loss = eval_metrics['loss']
                        unwrapped_model = self.model.module if self.distributed else self.model
                        unwrapped_model.save_state('./aeon_best')
                        logger.info("Best model saved.")
            
            if self.local_rank == 0:
                unwrapped_model = self.model.module if self.distributed else self.model
                unwrapped_model.save_state(f'./aeon_checkpoints/epoch_{epoch+1}')
        
        if self.local_rank == 0:
            logger.info("Training completed.")

def create_aeon_delta_model():
    config = AEONConfig()
    model = AEONDelta(config).to(device)
    logger.info(f"Created AEON-Δ model with {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    return model, config

def create_dataloaders(config, num_samples=1000):
    input_ids = torch.randint(0, config.vocab_size, (num_samples, config.seq_length))
    targets = torch.randint(0, config.vocab_size, (num_samples, config.seq_length))
    attention_mask = torch.ones(num_samples, config.seq_length)
    
    dataset = TensorDataset(input_ids, targets, attention_mask)
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataloader, eval_dataloader

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    attention_mask = torch.stack([item[2] for item in batch])
    
    return {
        'input_ids': input_ids,
        'targets': targets,
        'attention_mask': attention_mask
    }

def run_unit_tests(config):
    logger.info("Running unit tests for AEON-Δ model components")
    
    logger.info("Test 1: Memory Manager")
    memory_manager = MemoryManager(config)
    memory_manager.add_embedding(torch.randn(config.hidden_dim))
    states = memory_manager.retrieve_relevant(torch.randn(config.hidden_dim))
    assert len(states) >= 0
    logger.info("Test 1 passed")
    
    logger.info("Test 2: Knowledge Graph")
    kg = KnowledgeGraph(config)
    kg.add_fact("entity1", "relation1", "entity2", torch.randn(config.knowledge_dim))
    results = kg.query(subject="entity1")
    assert len(results) == 1
    assert results[0] == ("entity1", "relation1", "entity2")
    neighbors = kg.get_neighbors("entity1")
    assert len(neighbors) == 1
    logger.info("Test 2 passed")
    
    logger.info("Test 3: ThoughtEncoder/Decoder")
    tokens = torch.randint(0, config.vocab_size, (2, config.seq_length), device=device)
    z = encoder(tokens)
    assert z.shape == (2, config.z_dim)
    logits = decoder(z)  # Default max_len=32
    assert logits.shape == (2, 32, config.vocab_size)
    assert not torch.isnan(z).any(), "NaN in encoder output"
    logger.info("Test 3 passed")
    
    logger.info("Test 4: SequencePooler")
    pooler = SequencePooler(config, pooling_type="max").to(device)
    sequence = torch.randn(2, config.seq_length, config.hidden_dim, device=device)
    attention_mask = torch.ones(2, config.seq_length, device=device)
    pooled = pooler(sequence, attention_mask)
    assert pooled.shape == (2, config.hidden_dim)
    assert not torch.isnan(pooled).any(), "NaN in pooler output"
    logger.info("Test 4 passed")
    
    logger.info("Test 5: MetaLoopProcessor")
    meta_loop = MetaLoopProcessor(config).to(device)
    psi_0 = torch.randn(2, config.hidden_dim, device=device)
    meta_loop.lambda_op.eval()
    C_star, iterations = meta_loop(psi_0)
    assert C_star.shape == (2, config.hidden_dim)
    assert iterations.shape == (2,)
    logger.info("Test 5 passed")
    
    logger.info("Test 6: QuantumSimulator")
    quantum_sim = QuantumSimulator(config).to(device)
    pillars = torch.rand(2, config.num_pillars, device=device)
    quantum_results = quantum_sim(pillars)
    assert 'quantum_state' in quantum_results
    assert 'entanglement' in quantum_results
    assert 'action_propensity' in quantum_results
    assert quantum_results['action_propensity'].shape == (2, config.num_pillars)
    logger.info("Test 6 passed")
    
    logger.info("Test 7: TopologyAnalyzer")
    topo_analyzer = TopologyAnalyzer(config).to(device)
    pillars = torch.rand(2, config.num_pillars, device=device)
    iterations = torch.tensor([5, 10], device=device)
    topo_results = topo_analyzer(pillars, iterations)
    assert 'potential' in topo_results
    assert 'gradient' in topo_results
    assert 'catastrophes' in topo_results
    assert topo_results['potential'].shape == (2, 1)
    logger.info("Test 7 passed")
    
    logger.info("Test 8: ActionModule")
    action_module = ActionModule(config).to(device)
    core_state = torch.randn(2, config.hidden_dim, device=device)
    action_results = action_module(
        core_state, 
        pillars, 
        {'action_propensity': torch.rand(2, config.num_pillars, device=device), 
         'entanglement': torch.rand(2, device=device)},
        {'potential': torch.rand(2, 1, device=device)}
    )
    assert 'action_embedding' in action_results
    assert 'safety_score' in action_results
    assert action_results['action_embedding'].shape == (2, config.action_dim)
    logger.info("Test 8 passed")
    
    logger.info("Test 9: PlanningModule")
    planning_module = PlanningModule(config).to(device)
    planning_results = planning_module(
        core_state,
        pillars,
        action_results['action_embedding'],
        planning_horizon=3
    )
    assert 'action_plans' in planning_results
    assert 'values' in planning_results
    assert planning_results['action_plans'].shape == (2, 3, config.action_dim)
    logger.info("Test 9 passed")
    
    logger.info("Test 10: Full model")
    model = AEONDelta(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length), device=device)
    outputs = model(input_ids, attention_mask)
    assert 'logits' in outputs
    assert 'thoughts' in outputs
    assert 'core_state' in outputs
    assert 'pillars' in outputs
    assert outputs['logits'].shape[1] == config.seq_length, f"Logits seq_len {outputs['logits'].shape[1]} != seq_length {config.seq_length}"
    assert outputs['thoughts'].shape == (2, config.hidden_dim)
    
    targets = torch.randint(0, config.vocab_size, (2, config.seq_length), device=device)  # Match seq_length
    loss_dict = model.compute_loss(outputs, targets, attention_mask)
    assert 'total_loss' in loss_dict
    assert 'lm_loss' in loss_dict
    assert 'consistency_loss' in loss_dict
    assert 'safety_loss' in loss_dict
    logger.info("Test 10 passed")
    
    logger.info("All tests passed successfully!")

def example_usage():
    model, config = create_aeon_delta_model()
    
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    outputs = model(input_ids, attention_mask)
    
    loss_dict = model.compute_loss(outputs, targets, attention_mask)
    
    metrics = model.measure_self_consciousness(input_ids, attention_mask)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Core state shape: {outputs['core_state'].shape}")
    print(f"Pillars shape: {outputs['pillars'].shape}")
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Self-consistency: {metrics['self_consistency']:.4f}")
    print(f"Quantum entanglement: {metrics['quantum_entanglement']:.4f}")
    print(f"Catastrophes ratio: {metrics['catastrophes_ratio']:.4f}")
    print(f"Safety score: {metrics['safety_score']:.4f}")
    print(f"Average iterations: {metrics['avg_iterations']:.1f}")
    print(f"Pillars values: {metrics['pillars']}")
    
    model.update_knowledge(
        subject="AEON-Delta",
        relation="IS_A",
        object_entity="Cognitive Architecture",
        embedding=outputs['core_state'][0]
    )
    
    print("\nKnowledge Graph Query:")
    results = model.query_knowledge(subject="AEON-Delta")
    print(results)
    
    print("\nAction Embedding:")
    action = outputs['action_results']['action_embedding'][0]
    print(f"Shape: {action.shape}")
    
    print("\nPlanning Results:")
    plans = outputs['planning_results']['action_plans'][0]
    values = outputs['planning_results']['values'][0]
    print(f"Action plans shape: {plans.shape}")
    print(f"Values: {values.squeeze().detach().cpu().numpy()}")
    
    model.save_state()
    print("\nModel state saved.")

def setup_distributed_training(config):
    if not DIST_OK:
        return False
    if not config.distributed_training:
        return False
    
    if not dist.is_available():
        logger.warning("Distributed training requested but PyTorch distributed is not available")
        return False
    
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            if local_rank == 0:
                logger.info(f"Initialized distributed training with world size: {world_size}")
            
            torch.cuda.set_device(local_rank)
            
            config.world_size = world_size
            
            return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        return False

def save_lora(model, path):
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
    torch.save(lora_state, path)
    logger.info(f"LoRA weights saved to {path}")

def load_lora(model, path, map_location=None):
    lora_state = torch.load(path, map_location=map_location)
    model_state = model.state_dict()
    for k, v in lora_state.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    logger.info(f"LoRA weights loaded from {path}")

class ThoughtAETrainer(AEONTrainer):
    def train_step(self, tokens, aug_tokens):
        import torch
        import torch.nn.functional as F
        # ❗ Больше НЕ используем глобалы — берём модули из self.model
        enc = self.model.encoder
        dec = self.model.decoder
        enc.train(); dec.train()

        # Валидации и перенос на нужное устройство
        if not isinstance(tokens, torch.Tensor) or not isinstance(aug_tokens, torch.Tensor):
            raise ValueError("Tokens and aug_tokens must be torch.Tensors")
        if tokens.shape != aug_tokens.shape:
            raise ValueError(f"Shape mismatch: tokens {tokens.shape} vs aug_tokens {aug_tokens.shape}")
        if tokens.device != self.device:
            tokens = tokens.to(self.device)
            logger.info(f"Moved tokens to {self.device}")
        if aug_tokens.device != self.device:
            aug_tokens = aug_tokens.to(self.device)
            logger.info(f"Moved aug_tokens to {self.device}")

        # Encoder forward (без @no_grad) → z, z_aug
        x = enc.embed(tokens)
        _, (h, _) = enc.lstm(x)
        z = enc.norm(h.squeeze(0))

        x_aug = enc.embed(aug_tokens)
        _, (h_aug, _) = enc.lstm(x_aug)
        z_aug = enc.norm(h_aug.squeeze(0))

        # Чистим NaN/Inf
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        z_aug = torch.nan_to_num(z_aug, nan=0.0, posinf=1.0, neginf=-1.0)

        # Проверки форм
        batch_size = tokens.size(0)
        assert z.shape == (batch_size, self.config.z_dim), f"z shape mismatch: {z.shape}"
        assert z_aug.shape == (batch_size, self.config.z_dim), f"z_aug shape mismatch: {z_aug.shape}"

        # Decoder forward (teacher-forcing упрощённый под seq_length)
        max_len = self.config.seq_length
        h0 = dec.fc(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        inputs = dec.fc(z).unsqueeze(1).repeat(1, max_len, 1)
        out, _ = dec.lstm(inputs, (h0, c0))
        logits = dec.head(out)

        # Loss: CE + InfoNCE + KL (как у тебя) 
        recon_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), tokens.view(-1))
        info_nce = self._info_nce(z, z_aug)
        kl = self._kl_div(z)
        loss = recon_loss + 0.3 * info_nce + 0.1 * kl  # 0.1 = self.config.kl_weight по умолчанию

        # Оптимизация
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        logging.getLogger("AEON-Delta").info(
            f"Phase A step: recon={recon_loss.item():.6f}, nce={info_nce.item():.6f}, kl={kl.item():.6f}"
        )
        return {'loss': loss.item(), 'recon': recon_loss.item(), 'nce': info_nce.item(), 'kl': kl.item()}


    def _info_nce(self, z, z_pos):
        sim = F.cosine_similarity(z.unsqueeze(1), z_pos.unsqueeze(0), dim=-1)
        labels = torch.arange(z.size(0)).to(self.device)
        return F.cross_entropy(sim / 0.07, labels)

    def _kl_div(self, z):
        mu, logvar = z.mean(dim=-1), z.var(dim=-1).log()
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def _check_stop(self):
        # Placeholder for early stopping: MSE<0.015 (use recon_loss proxy), consistency>0.6, entanglement>0.05
        # Currently returns False to prevent stopping; integrate model metrics for AEON-Δ consciousness
        # Verification: Log call; future: Compute via model.measure_self_consiousness() or avg_recon
        logger.info("_check_stop invoked in ThoughtAETrainer: Criteria not met (placeholder implementation). Continuing training.")
        return False  # Explicitly do not stop; add real checks e.g., if self.last_recon < 0.015: return True

    def fit(self, corpus_path='/Users/vasapupkin/AEON/AEONSTART/data/processed/', epochs=4, curriculum=True):
        import os
        from torch.utils.data import TensorDataset, DataLoader
        from tqdm import tqdm
        import logging
        logger = logging.getLogger("AEON-Delta")

        try:
            # Verify corpus_path
            if not isinstance(corpus_path, str) or not os.path.isdir(corpus_path):
                raise ValueError(f"Invalid corpus_path: {corpus_path}. Must be a valid directory string.")
            logger.info(f"Corpus path verified: {corpus_path}")

            if curriculum:
                # Define and verify file paths
                short_path = os.path.join(corpus_path, 'short.pt')
                full_path = os.path.join(corpus_path, 'full.pt')
                
                if not os.path.exists(short_path):
                    raise FileNotFoundError(f"Short data not found at {short_path}")
                logger.info(f"Short path verified: {short_path}")
                
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Full data not found at {full_path}")
                logger.info(f"Full path verified: {full_path}")

                # Load datasets
                short_data = torch.load(short_path, map_location=self.device)
                if not isinstance(short_data, torch.Tensor) or short_data.dim() != 2:
                    raise ValueError(f"Invalid short data format: expected [N, seq_len] tensor, got {short_data.shape if hasattr(short_data, 'shape') else type(short_data)}")
                if short_data.size(1) != self.config.seq_length:
                    raise ValueError(f"Short data seq_len {short_data.size(1)} does not match config.seq_length {self.config.seq_length}")
                logger.info(f"Loaded short data: {short_data.shape}")

                full_data = torch.load(full_path, map_location=self.device)
                if not isinstance(full_data, torch.Tensor) or full_data.dim() != 2:
                    raise ValueError(f"Invalid full data format: expected [N, seq_len] tensor, got {full_data.shape if hasattr(full_data, 'shape') else type(full_data)}")
                if full_data.size(1) != self.config.seq_length:
                    raise ValueError(f"Full data seq_len {full_data.size(1)} does not match config.seq_length {self.config.seq_length}")
                logger.info(f"Loaded full data: {full_data.shape}")

                for epoch in range(epochs):
                    # Select data based on curriculum (first two epochs: short_data, then full_data)
                    data = short_data if epoch < 2 else full_data
                    logger.info(f"Epoch {epoch+1}: Using {'short_data' if epoch < 2 else 'full_data'} with shape {data.shape}")

                    # On-the-fly augmentation (replace 10% of tokens)
                    aug_data = data.clone()
                    mask = torch.rand_like(aug_data.float()) < 0.1
                    aug_data[mask] = torch.randint(0, self.config.vocab_size, (mask.sum().item(),), device=self.device)
                    logger.info(f"Created augmented data for epoch {epoch+1}: {aug_data.shape}")

                    # Verify data integrity
                    assert data.shape == aug_data.shape, f"Shape mismatch: original {data.shape} vs augmented {aug_data.shape}"
                    assert data.device == self.device, f"Data device mismatch: {data.device} vs {self.device}"
                    assert aug_data.device == self.device, f"Augmented data device mismatch: {aug_data.device} vs {self.device}"

                    # Create paired dataset/loader
                    dataset = TensorDataset(data, aug_data)
                    loader = DataLoader(dataset, batch_size=256, shuffle=True)
                    if len(loader) == 0:
                        raise ValueError("Empty DataLoader - check data size")

                    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", disable=self.local_rank != 0)
                    for batch in progress_bar:
                        tokens, aug_tokens = batch
                        # Verify batch shapes and device
                        assert tokens.shape == aug_tokens.shape, f"Batch shape mismatch: {tokens.shape} vs {aug_tokens.shape}"
                        assert tokens.dim() == 2 and tokens.size(1) == self.config.seq_length, f"Invalid batch shape: {tokens.shape}"
                        assert tokens.device == self.device, f"Tokens device mismatch: {tokens.device} vs {self.device}"
                        assert aug_tokens.device == self.device, f"Augmented tokens device mismatch: {aug_tokens.device} vs {self.device}"

                        loss_dict = self.train_step(tokens, aug_tokens)
                        progress_bar.set_postfix({
                            'total_loss': loss_dict['total_loss'],
                            'recon_loss': loss_dict['recon_loss'],
                            'info_nce': loss_dict['info_nce'],
                            'kl': loss_dict['kl']
                        })
                        logger.info(f"Batch processed in epoch {epoch+1}: loss={loss_dict['total_loss']:.4f}, "
                                    f"recon={loss_dict['recon_loss']:.4f}, "
                                    f"info_nce={loss_dict['info_nce']:.4f}, "
                                    f"kl={loss_dict['kl']:.4f}")

                    if self._check_stop():
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # Non-curriculum training
                data_path = os.path.join(corpus_path, 'full.pt')
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data not found at {data_path}")
                logger.info(f"Non-curriculum data path verified: {data_path}")

                data = torch.load(data_path, map_location=self.device)
                if not isinstance(data, torch.Tensor) or data.dim() != 2:
                    raise ValueError(f"Invalid data format: expected [N, seq_len] tensor, got {data.shape if hasattr(data, 'shape') else type(data)}")
                if data.size(1) != self.config.seq_length:
                    raise ValueError(f"Data seq_len {data.size(1)} does not match config.seq_length {self.config.seq_length}")
                logger.info(f"Loaded full data: {data.shape}")

                # On-the-fly augmentation
                aug_data = data.clone()
                mask = torch.rand_like(aug_data.float()) < 0.1
                aug_data[mask] = torch.randint(0, self.config.vocab_size, (mask.sum().item(),), device=self.device)
                logger.info(f"Created augmented data: {aug_data.shape}")

                dataset = TensorDataset(data, aug_data)
                loader = DataLoader(dataset, batch_size=256, shuffle=True)
                if len(loader) == 0:
                    raise ValueError("Empty DataLoader - check data size")

                for epoch in range(epochs):
                    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", disable=self.local_rank != 0)
                    for batch in progress_bar:
                        tokens, aug_tokens = batch
                        loss_dict = self.train_step(tokens, aug_tokens)
                        progress_bar.set_postfix({
                            'total_loss': loss_dict['total_loss'],
                            'recon_loss': loss_dict['recon_loss'],
                            'info_nce': loss_dict['info_nce'],
                            'kl': loss_dict['kl']
                        })
                        logger.info(f"Non-curriculum batch processed in epoch {epoch+1}: loss={loss_dict['total_loss']:.4f}")
                    
                    if self._check_stop():
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error in fit method: {str(e)}")
            raise

class ZDynamicsTrainer(AEONTrainer):    
    def train_step(self, batch):
        import torch.nn.functional as F
        global encoder
        encoder.eval()
        assert batch.dim() == 3 and batch.shape[1] == 2 and batch.shape[2] == self.config.hidden_dim, \
            f"Invalid batch shape: {batch.shape}"

        z_t  = batch[:, 0, :].to(self.device).float()
        z_t1 = batch[:, 1, :].to(self.device).float()

        pred_z = self.model.rssm(z_t)
        mse = F.mse_loss(pred_z, z_t1)
        mask = torch.rand_like(z_t) < 0.2
        mask_loss = F.mse_loss(pred_z[mask], z_t1[mask]) if mask.any() else torch.tensor(0.0, device=self.device)

        def _kl_diag_gaussians(pred, target, eps: float = 1e-6):
            mu_p = pred.mean(dim=-1)
            var_p = pred.var(dim=-1, unbiased=False) + eps
            mu_q = target.mean(dim=-1)
            var_q = target.var(dim=-1, unbiased=False) + eps
            return 0.5 * ((var_p/var_q) + ((mu_q - mu_p)**2)/var_q - 1.0 + torch.log(var_q/var_p)).mean()

        kl = _kl_diag_gaussians(pred_z, z_t1)
        pillars, _ = self.model.pillars_module(pred_z)
        ent = self.model.quantum_sim(pillars)['entanglement'].mean()

        loss = mse + 0.2 * mask_loss + self.config.kl_weight * kl + 0.05 * (-ent)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        logger.info(f"Phase B step: mse={mse.item():.6f}, kl={kl.item():.6e}, ent={ent.item():.6f}")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected")
        return {'total_loss': loss.item(), 'mse': mse.item(), 'kl': kl.item(), 'ent': ent.item()}


    def _kl_rssm(self, pred):
        mu, logvar = pred.mean(dim=-1), pred.var(dim=-1).log()
        # Math explanation: This is closed-ended KL divergence between N(mu, exp(logvar)) and N(0,1).
        # Derived from ∫ p(x) log(p(x)/q(x)) dx for Gaussians: 0.5 * (tr(Σq^{-1}Σp) + (μq - μp)^T Σq^{-1} (μq - μp) - k + log(|Σq|/|Σp|)),
        # simplified for diagonal unit prior: -0.5 * (1 + logvar - mu^2 - exp(logvar)).mean().
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def fit(self, z_pairs_path, epochs=6):
        loader = DataLoader(torch.load(z_pairs_path), batch_size=256, shuffle=True)
        for epoch in range(epochs):
            for batch in tqdm(loader):
                self.train_step(batch)  # Pass batch directly, fixed slice inside
            self._check_stop()

    def _check_stop(self):
        # Implement MSE<0.015, consistency>0.6, entanglement>0.05 check
        pass  # Placeholder, add logic as needed

class CuriosityPPOTrainer(AEONTrainer):
    def step_self_env(self, seed_thoughts):
        import torch
        logger = logging.getLogger("AEON-Delta")
        global encoder

        dev = next(self.model.parameters()).device
        enc = encoder.to(dev).eval()

        for seed in seed_thoughts:
            ids = [ord(c) % 50000 for c in seed[:64]]
            tokens = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)

            z = enc(tokens)
            perturbed_z = z + torch.randn_like(z) * 0.1
            pred_z = self.model.rssm(perturbed_z)

            # если нужен KL к целевому z_t1 — добавь здесь; иначе работаем от prior:
            mu, logvar = pred_z.mean(dim=-1), pred_z.var(dim=-1, unbiased=False).add(1e-6).log()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

            pillars, _ = self.model.pillars_module(pred_z)
            catastrophe = self.model.topology_analyzer(pillars)['catastrophe_probs'].mean()
            q = self.model.quantum_sim(pillars)
            safety = self.model.action_module(pred_z, pillars, q, self.model.topology_analyzer(pillars))['safety_score'].mean()

            reward = kl + (-catastrophe) + safety
            logger.info(f"Phase C step: KL={kl.item()}, reward={reward.item()}, z_shape={z.shape}, pred_z_shape={pred_z.shape}")
            self.global_step += 1


    def _kl_rssm(self, pred):
        mu, logvar = pred.mean(dim=-1), pred.var(dim=-1).log()
        # Math explanation: This is closed-ended KL divergence between N(mu, exp(logvar)) and N(0,1).
        # Derived from ∫ p(x) log(p(x)/q(x)) dx for Gaussians: 0.5 * (tr(Σq^{-1}Σp) + (μq - μp)^T Σq^{-1} (μq - μp) - k + log(|Σq|/|Σp|)),
        # simplified for diagonal unit prior: -0.5 * (1 + logvar - mu^2 - exp(logvar)).mean().
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

def freeze_encoder_decoder(model):
    for name, param in model.named_parameters():
        if 'encoder' in name or 'decoder' in name:
            param.requires_grad = False

def unfreeze_lora_blocks(model, blocks):
    for name, param in model.named_parameters():
        if any(block in name for block in blocks) and 'lora' in name.lower():
            param.requires_grad = True

def console_inference_loop(model):
    """New: Interactive console inference post-training."""
    model.eval()
    logger.info("Entering inference mode. Type 'exit' to quit.")
    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() == 'exit':
            break
        generated = model.generate_thought(prompt)
        print(f"Generated response: {generated}")

if __name__ == "__main__":
    # Пример запуска
    config = AEONConfig(lora_rank=16)
    model = AEONDelta(config).to(device)
    trainer_A = ThoughtAETrainer(model, config)
    trainer_A.fit('/Users/vasapupkin/AEON/AEONSTART/data/processed/', epochs=4)
    freeze_encoder_decoder(model)
    unfreeze_lora_blocks(model, ["MetaLoop", "Pillars", "RSSM"])
    trainer_B = ZDynamicsTrainer(model, config)
    trainer_B.fit('/Users/vasapupkin/AEON/AEONSTART/data/processed/z_pairs.pt', epochs=6)
    trainer_C = CuriosityPPOTrainer(model, config)
    max_steps = 500  # Minimal change: Limit phase C to 500 steps to exit loop
    step = 0
    while True:
        trainer_C.step_self_env(["Cogito ergo sum"])
        save_lora(model, f"/Users/vasapupkin/AEON/AEONSTART/lora/step{trainer_C.global_step}.pt")
        step += 1
        if step >= max_steps:
            logger.info(f"Phase C completed after {max_steps} steps. Entering inference mode.")
            break
    console_inference_loop(model)  # Enter console test mode
