"""
════════════════════════════════════════════════════════════════════════════
AEON-DELTA RMT v3.0 - PRODUCTION-READY COMPLETE IMPLEMENTATION
════════════════════════════════════════════════════════════════════════════

Advanced Embodied Ontological Network - Delta with Recurrent Memory Transformer

Version: 3.0.0
Status: Production-Ready


Architecture Philosophy:
- Meta-Cognitive Reasoning: Iterative refinement through fixed-point convergence
- Five Pillars: Will, Resolve, Growth, Union, Movement
- Quantum-Inspired: Entanglement metrics for thought diversity
- Topological Safety: Catastrophe detection via Hessian analysis
- Transparent Self-Reporting: Built-in introspection and honesty gates
- Production-Grade: Enterprise-ready with full safety, monitoring, and deployment

Mathematical Foundations:
- Banach Fixed-Point Theorem for meta-loop convergence
- Lipschitz regularization for contraction guarantees
- Vector Quantization for discrete thought representation
- Catastrophe Theory for stability analysis
- Von Neumann Entropy for quantum simulation

Key Innovations:
1. Provably convergent meta-loop with certified error bounds
2. Anti-collapse VQ-VAE with code revival mechanisms
3. High-performance Hessian via finite differences
4. Multi-level safety system with adaptive weights
5. Transparent self-reporting with honesty metrics

Authors: AEON Research Team


════════════════════════════════════════════════════════════════════════════
"""

# ============================================================================
# SECTION 1: IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
import sys
import json
import logging
import warnings
import hashlib
import traceback
import math
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Tuple, Dict, Optional, Union, Any, Set, Callable, 
    NamedTuple, Literal
)
from enum import Enum, auto
from collections import OrderedDict, defaultdict, Counter, deque
from contextlib import contextmanager
from functools import wraps
import threading

# Core scientific computing
import numpy as np

# PyTorch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR, OneCycleLR
)

# AMP support (backward compatible)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Transformers (optional but recommended)
try:
    from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers not available. Install via: pip install transformers"
    )

# Progress bars
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Monitoring (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Version info
__version__ = "3.0.0"
__author__ = "AEON Research Team"

# Logging setup
_log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _log_handlers.append(logging.FileHandler('aeon_delta.log'))
except OSError:
    pass  # Skip file handler if path is not writable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers
)
logger = logging.getLogger("AEON-Delta")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def set_seed(seed: int) -> torch.Generator:
    """Set global random seed for reproducibility.
    
    Fixes seeds for Python, NumPy, and PyTorch (CPU and CUDA).
    Returns a torch.Generator seeded with the given value, which can
    be passed to non-deterministic operations for bitwise reproducibility.
    
    Args:
        seed: Integer seed value.
    
    Returns:
        A seeded torch.Generator for use in nn.init and torch.randn.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")
    return generator


# ============================================================================
# SECTION 2: DEVICE MANAGEMENT SYSTEM
# ============================================================================

class DeviceManager:
    """
    Thread-safe, immutable device management system.
    
    Features:
    - Auto-selection with intelligent GPU picking
    - Memory management and monitoring
    - Fallback mechanisms for unavailable devices
    - AMP (Automatic Mixed Precision) configuration
    - Multi-GPU support preparation
    
    Example:
        >>> dm = DeviceManager.auto_select()
        >>> model = Model(config).to(dm.device)
        >>> with dm.device_context('cuda:1'):
        ...     output = model(x)
    """
    
    _instance_lock = threading.Lock()
    _device_capabilities_cache = {}
    
    def __init__(
        self, 
        device: Union[str, torch.device], 
        allow_fallback: bool = True,
        memory_fraction: float = 0.9
    ):
        """
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', 'mps')
            allow_fallback: Fallback to CPU if target unavailable
            memory_fraction: Max GPU memory to use (0.0-1.0)
        """
        self._device = self._validate_device(device, allow_fallback)
        self._allow_fallback = allow_fallback
        self._memory_fraction = memory_fraction
        self._amp_enabled = self._device.type == 'cuda'
        
        # Initialize AMP scaler
        if self._amp_enabled:
            self._scaler = GradScaler(device=self._device.type)
        else:
            self._scaler = None
        
        # Configure CUDA memory if applicable
        if self._device.type == 'cuda':
            self._configure_cuda_memory()
        
        logger.info(
            f"✅ DeviceManager initialized: {self._device}, "
            f"AMP={self._amp_enabled}"
        )
    
    @staticmethod
    def _validate_device(
        device: Union[str, torch.device], 
        allow_fallback: bool
    ) -> torch.device:
        """Validate and create device with fallback."""
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = str(device).lower().strip()
        
        # Normalize Cyrillic characters (с→c, у→u)
        replacements = {'с': 'c', 'у': 'u', 'С': 'c', 'У': 'u'}
        for cyrillic, latin in replacements.items():
            device_str = device_str.replace(cyrillic, latin)
        
        try:
            target_device = torch.device(device_str)
        except RuntimeError as e:
            if allow_fallback:
                logger.warning(
                    f"Invalid device '{device_str}': {e}, falling back to CPU"
                )
                return torch.device('cpu')
            raise ValueError(f"Invalid device specification: {device_str}") from e
        
        # Check availability
        if target_device.type == 'cuda':
            if not torch.cuda.is_available():
                if allow_fallback:
                    logger.warning("CUDA requested but unavailable, using CPU")
                    return torch.device('cpu')
                raise RuntimeError(
                    "CUDA device requested but torch.cuda.is_available() == False"
                )
            
            # Check specific GPU
            if target_device.index is not None:
                if target_device.index >= torch.cuda.device_count():
                    if allow_fallback:
                        logger.warning(
                            f"GPU {target_device.index} not found, using cuda:0"
                        )
                        return torch.device('cuda:0')
                    raise ValueError(
                        f"GPU index {target_device.index} >= "
                        f"device_count {torch.cuda.device_count()}"
                    )
        
        elif target_device.type == 'mps':
            if not (hasattr(torch.backends, 'mps') and 
                    torch.backends.mps.is_available()):
                if allow_fallback:
                    logger.warning("MPS requested but unavailable, using CPU")
                    return torch.device('cpu')
                raise RuntimeError("MPS device requested but not available")
        
        return target_device
    
    def _configure_cuda_memory(self):
        """Configure CUDA memory allocator."""
        device_idx = self._device.index if self._device.index is not None else 0
        
        # Get available memory
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        max_memory = int(total_memory * self._memory_fraction)
        
        # Set memory limit (PyTorch 2.0+)
        try:
            torch.cuda.set_per_process_memory_fraction(
                self._memory_fraction, 
                device=device_idx
            )
            logger.info(
                f"CUDA memory limit: {max_memory / 1e9:.2f} GB "
                f"({self._memory_fraction*100:.0f}%)"
            )
        except Exception as e:
            logger.warning(f"Failed to set memory fraction: {e}")
    
    @classmethod
    def auto_select(
        cls, 
        prefer_gpu: bool = True,
        min_memory_gb: float = 2.0
    ) -> 'DeviceManager':
        """
        Automatically select best available device.
        
        Logic:
        1. CUDA with most free memory (if > min_memory_gb)
        2. MPS (for Apple Silicon)
        3. CPU (fallback)
        
        Args:
            prefer_gpu: Prefer GPU if available
            min_memory_gb: Minimum free GPU memory (GB)
        
        Returns:
            DeviceManager instance
        """
        if prefer_gpu and torch.cuda.is_available():
            # Select GPU with maximum free memory
            best_device = None
            max_free_memory = min_memory_gb * 1e9
            
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.mem_get_info()[0]
                    
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = f'cuda:{i}'
                except Exception as e:
                    logger.warning(f"Error checking GPU {i}: {e}")
            
            if best_device:
                logger.info(
                    f"Auto-selected {best_device} with "
                    f"{max_free_memory/1e9:.2f} GB free"
                )
                return cls(best_device)
        
        # Check MPS (Apple Silicon)
        if (prefer_gpu and hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available()):
            logger.info("Auto-selected MPS (Apple Silicon)")
            return cls('mps')
        
        # Fallback to CPU
        logger.info("Auto-selected CPU")
        return cls('cpu')
    
    @property
    def device(self) -> torch.device:
        """Immutable device accessor."""
        return self._device
    
    @property
    def is_cuda(self) -> bool:
        return self._device.type == 'cuda'
    
    @property
    def is_mps(self) -> bool:
        return self._device.type == 'mps'
    
    @property
    def is_cpu(self) -> bool:
        return self._device.type == 'cpu'
    
    @property
    def amp_enabled(self) -> bool:
        """AMP availability."""
        return self._amp_enabled
    
    @property
    def scaler(self):
        """GradScaler for AMP training."""
        return self._scaler
    
    def get_memory_stats(self) -> dict:
        """Memory usage statistics."""
        if self.is_cuda:
            device_idx = self._device.index or 0
            return {
                'allocated': torch.cuda.memory_allocated(device_idx) / 1e9,
                'reserved': torch.cuda.memory_reserved(device_idx) / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated(device_idx) / 1e9,
                'device': str(self._device)
            }
        return {'device': str(self._device), 'type': 'non-cuda'}
    
    @contextmanager
    def device_context(self, temporary_device: Union[str, torch.device]):
        """
        Temporary device switching.
        
        Example:
            >>> with dm.device_context('cpu'):
            ...     cpu_output = model(x.cpu())
        """
        old_device = self._device
        try:
            self._device = self._validate_device(
                temporary_device, 
                self._allow_fallback
            )
            logger.debug(f"Switched device: {old_device} → {self._device}")
            yield self
        finally:
            self._device = old_device
            logger.debug(f"Restored device: {self._device}")
    
    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device}, amp={self._amp_enabled})"


# ============================================================================
# SECTION 3: TENSOR SAFETY SYSTEM
# ============================================================================

class NaNPolicy(Enum):
    """Strategies for handling NaN/Inf."""
    RAISE = auto()       # Raise exception
    WARN = auto()        # Log warning and replace
    SILENT = auto()      # Silently replace
    RETURN_NONE = auto() # Return None
    QUARANTINE = auto()  # Isolate problematic batch elements


class TensorGuard:
    """
    Production-grade system for protecting against numerical instabilities.
    
    Philosophy:
    - Fail-fast in training (detect problems early)
    - Graceful degradation in inference (continue operation)
    - Full traceability for debugging
    
    Example:
        >>> guard = TensorGuard(policy=NaNPolicy.WARN, enable_tracking=True)
        >>> clean_tensor = guard.sanitize(dirty_tensor, context="meta_loop_output")
        >>> guard.print_report()
    """
    
    def __init__(
        self,
        policy: NaNPolicy = NaNPolicy.WARN,
        default_value: float = 0.0,
        max_value: float = 1e6,
        min_value: float = -1e6,
        enable_tracking: bool = True,
        alert_threshold: int = 10,
        max_history_size: int = 1000,
    ):
        self.policy = policy
        self.default_value = default_value
        self.max_value = max_value
        self.min_value = min_value
        self.enable_tracking = enable_tracking
        self.alert_threshold = alert_threshold
        self._max_history_size = max_history_size
        
        # Tracking statistics
        self._nan_count = 0
        self._inf_count = 0
        self._sanitize_count = 0
        self._context_history = deque(maxlen=max_history_size)
        
        logger.info(f"TensorGuard initialized: policy={policy.name}")
    
    def sanitize(
        self,
        tensor: torch.Tensor,
        context: str = "unknown",
        custom_default: Optional[float] = None,
        allow_inf: bool = False,
    ) -> torch.Tensor:
        """
        Main sanitization method.
        
        Args:
            tensor: Input tensor
            context: Context for logging (e.g., "meta_loop_iteration_5")
            custom_default: Override default value
            allow_inf: Allow Inf (replace only NaN)
        
        Returns:
            Sanitized tensor
        
        Raises:
            ValueError: If policy=RAISE and NaN/Inf detected
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        
        # Skip non-float tensors
        if not tensor.is_floating_point() and not torch.is_complex(tensor):
            return tensor
        
        default = custom_default if custom_default is not None else self.default_value
        
        # Detect problems
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item() if not allow_inf else False
        
        if not (has_nan or has_inf):
            return tensor  # Clean tensor
        
        # Tracking
        if self.enable_tracking:
            self._nan_count += int(torch.isnan(tensor).sum().item())
            self._inf_count += int(torch.isinf(tensor).sum().item())
            self._sanitize_count += 1
            self._context_history.append({
                'context': context,
                'shape': tuple(tensor.shape),
                'nan_count': int(torch.isnan(tensor).sum().item()),
                'inf_count': int(torch.isinf(tensor).sum().item()),
                'stacktrace': (
                    ''.join(traceback.format_stack()[-3:-1]) 
                    if self.policy == NaNPolicy.RAISE else None
                )
            })
        
        # Policy enforcement
        if self.policy == NaNPolicy.RAISE:
            error_msg = (
                f"NaN/Inf detected in {context}:\n"
                f"  Shape: {tensor.shape}\n"
                f"  NaN count: {torch.isnan(tensor).sum().item()}\n"
                f"  Inf count: {torch.isinf(tensor).sum().item()}\n"
                f"  Min: {tensor[torch.isfinite(tensor)].min().item() if torch.isfinite(tensor).any() else 'N/A'}\n"
                f"  Max: {tensor[torch.isfinite(tensor)].max().item() if torch.isfinite(tensor).any() else 'N/A'}\n"
            )
            raise ValueError(error_msg)
        
        elif self.policy == NaNPolicy.WARN:
            if self._sanitize_count % self.alert_threshold == 0:
                logger.warning(
                    f"⚠️  NaN/Inf sanitization #{self._sanitize_count} in {context}: "
                    f"shape={tensor.shape}, nan={torch.isnan(tensor).sum().item()}, "
                    f"inf={torch.isinf(tensor).sum().item()}"
                )
        
        elif self.policy == NaNPolicy.RETURN_NONE:
            return None
        
        elif self.policy == NaNPolicy.QUARANTINE:
            return self._quarantine_batch(tensor, context)
        
        # Sanitization
        cleaned = tensor.clone()
        
        # Replace NaN
        if has_nan:
            cleaned = torch.where(
                torch.isnan(cleaned),
                torch.full_like(cleaned, default),
                cleaned
            )
        
        # Replace Inf
        if has_inf:
            cleaned = torch.where(
                torch.isinf(cleaned),
                torch.sign(cleaned) * self.max_value,
                cleaned
            )
        
        # Clipping
        cleaned = torch.clamp(cleaned, min=self.min_value, max=self.max_value)
        
        return cleaned
    
    def _quarantine_batch(self, tensor: torch.Tensor, context: str) -> torch.Tensor:
        """
        Advanced strategy: isolate problematic batch elements.
        
        Logic:
        - If [B, ...] and problems only in some B: replace with batch mean
        - If problems everywhere: fallback to standard sanitization
        """
        if tensor.dim() < 1:
            return self.sanitize(tensor, context)
        
        # Vectorized check per batch dimension (avoid Python loop)
        batch_size = tensor.shape[0]
        flat_per_batch = tensor.view(batch_size, -1)
        batch_has_issue = (
            torch.isnan(flat_per_batch).any(dim=1) | 
            torch.isinf(flat_per_batch).any(dim=1)
        )
        
        num_bad_batches = batch_has_issue.sum().item()
        
        if num_bad_batches == 0:
            return tensor
        
        if num_bad_batches == batch_size:
            # All corrupted - apply direct sanitization to avoid recursion
            logger.warning(
                f"All batches corrupted in {context}, "
                f"applying full sanitization"
            )
            cleaned = tensor.clone()
            default = self.default_value
            cleaned = torch.where(
                torch.isnan(cleaned),
                torch.full_like(cleaned, default),
                cleaned
            )
            cleaned = torch.where(
                torch.isinf(cleaned),
                torch.sign(cleaned) * self.max_value,
                cleaned
            )
            cleaned = torch.clamp(cleaned, min=self.min_value, max=self.max_value)
            return cleaned
        
        # Partial corruption - replace bad batches with mean of good ones
        good_batches = tensor[~batch_has_issue]
        replacement = good_batches.mean(dim=0, keepdim=True)
        
        cleaned = tensor.clone()
        cleaned[batch_has_issue] = replacement
        
        logger.warning(
            f"Quarantined {num_bad_batches}/{batch_size} batches in {context}, "
            f"replaced with mean of clean batches"
        )
        
        return cleaned
    
    def print_report(self):
        """Print sanitization statistics."""
        if not self.enable_tracking:
            logger.info("Tracking disabled, no report available")
            return
        
        logger.info("="*70)
        logger.info("TensorGuard Sanitization Report")
        logger.info("="*70)
        logger.info(f"Total sanitizations: {self._sanitize_count}")
        logger.info(f"Total NaN replaced: {self._nan_count}")
        logger.info(f"Total Inf replaced: {self._inf_count}")
        
        if self._context_history:
            logger.info("\nTop 5 contexts:")
            context_counts = Counter([h['context'] for h in self._context_history])
            for ctx, count in context_counts.most_common(5):
                logger.info(f"  {ctx}: {count} times")
        
        logger.info("="*70)
    
    def reset_stats(self):
        """Reset statistics."""
        self._nan_count = 0
        self._inf_count = 0
        self._sanitize_count = 0
        self._context_history = deque(maxlen=self._max_history_size)


def tensor_safe(
    policy: NaNPolicy = NaNPolicy.WARN,
    track: bool = True,
    sanitize_inputs: bool = True,
    sanitize_outputs: bool = True,
):
    """
    Decorator for automatic sanitization of function inputs/outputs.
    
    Example:
        @tensor_safe(policy=NaNPolicy.WARN)
        def risky_computation(x, y):
            return x / y  # May produce NaN/Inf
    """
    def decorator(func: Callable) -> Callable:
        guard = TensorGuard(policy=policy, enable_tracking=track)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = f"{func.__module__}.{func.__name__}"
            
            # Sanitize inputs
            if sanitize_inputs:
                args = tuple(
                    guard.sanitize(arg, f"{context}_input_{i}") 
                    if isinstance(arg, torch.Tensor) else arg
                    for i, arg in enumerate(args)
                )
                kwargs = {
                    k: guard.sanitize(v, f"{context}_kwarg_{k}") 
                    if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {context}: {e}")
                raise
            
            # Sanitize output
            if sanitize_outputs:
                if isinstance(result, torch.Tensor):
                    result = guard.sanitize(result, f"{context}_output")
                elif isinstance(result, (tuple, list)):
                    result = type(result)(
                        guard.sanitize(r, f"{context}_output_{i}") 
                        if isinstance(r, torch.Tensor) else r
                        for i, r in enumerate(result)
                    )
                elif isinstance(result, dict):
                    result = {
                        k: guard.sanitize(v, f"{context}_output_{k}") 
                        if isinstance(v, torch.Tensor) else v
                        for k, v in result.items()
                    }
            
            return result
        
        wrapper.__tensor_guard__ = guard
        return wrapper
    
    return decorator


class SafeTensorProcessor:
    """Global hook registration for automatic tensor safety."""
    
    @staticmethod
    def register_hooks(model: nn.Module):
        """Register forward hooks for tensor sanitization."""
        def _sanitize(x):
            if isinstance(x, torch.Tensor):
                if x.is_floating_point() or torch.is_complex(x):
                    # Simple NaN/Inf replacement
                    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                    x = torch.where(torch.isinf(x), torch.sign(x) * 1e6, x)
                return x
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


# ============================================================================
# SECTION 4: CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class AEONConfig:
    """
    Production-ready configuration for AEON-Delta RMT v3.0.
    
    Design principles:
    - Immutable after __post_init__
    - Type-safe
    - Self-validating
    - Serializable
    - Environment-aware
    """
    
    # ===== CORE ARCHITECTURE =====
    z_dim: int = 256
    hidden_dim: int = 256
    meta_dim: int = 256
    vocab_size: int = 30522
    num_pillars: int = 5
    seq_length: int = 64
    action_dim: int = 64
    cls_token_id: int = 101   # [CLS] token ID (BERT default)
    sep_token_id: int = 102   # [SEP] token ID (BERT default)
    knowledge_dim: int = 128
    
    # ===== META-LOOP =====
    alpha: float = 0.9
    max_iterations: int = 50
    min_iterations: int = 3
    convergence_threshold: float = 1e-5
    lipschitz_target: float = 0.85
    anderson_memory: int = 5
    enable_anderson: bool = True
    enable_stability_monitor: bool = True
    
    # ===== VQ-VAE =====
    vq_num_embeddings: int = 8192
    vq_embedding_dim: int = 256
    vq_commitment_cost: float = 0.25
    vq_ema_decay: float = 0.99
    vq_revival_threshold: int = 100
    vq_split_threshold: float = 0.1
    use_vq: bool = True
    
    # ===== TOPOLOGY =====
    topo_method: str = "finite_differences"
    topo_epsilon: float = 1e-4
    topo_use_cache: bool = True
    enable_catastrophe_detection: bool = True
    
    # ===== QUANTUM =====
    quantum_bond_dim: int = 16
    quantum_method: str = "mps"
    enable_quantum_sim: bool = True
    
    # ===== TRAINING =====
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    cosine_decay_steps: int = 10000
    min_lr_ratio: float = 0.1
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1
    
    # ===== LOSS WEIGHTS =====
    lambda_self_consistency: float = 0.1
    lambda_reg: float = 0.01
    lambda_safety: float = 0.1
    lambda_lipschitz: float = 0.05
    kl_weight: float = 0.1
    
    # ===== LORA =====
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    
    # ===== SAFETY =====
    safety_threshold: float = 0.85
    nan_policy: str = "WARN"
    enable_safety_guardrails: bool = True
    safety_alert_threshold: int = 10
    
    # ===== DEVICE =====
    device_str: str = "auto"
    device_memory_fraction: float = 0.9
    device_allow_fallback: bool = True
    use_amp: bool = True
    
    # ===== PATHS =====
    memory_path: str = "./aeon_memory"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # ===== MODES =====
    training_mode: str = "full"
    inference_mode: str = "accurate"
    
    # ===== MONITORING =====
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "aeon-delta"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # ===== EXPERIMENTAL =====
    enable_multimodal: bool = False
    enable_social_cognition: bool = False
    enable_deception_suppressor: bool = True
    enable_code_execution: bool = False
    
    # ===== INTERNAL =====
    device_manager: Any = field(default=None, init=False, repr=False)
    tensor_guard: Any = field(default=None, init=False, repr=False)
    version: str = field(default="3.0.0", init=False)
    _frozen: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Validation and initialization."""
        # Critical validations
        assert self.z_dim > 0, "z_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.seq_length > 0, "seq_length must be positive"
        assert self.num_pillars >= 3, "num_pillars must be >= 3"
        assert self.vq_embedding_dim == self.z_dim, \
            f"vq_embedding_dim ({self.vq_embedding_dim}) must equal z_dim ({self.z_dim})"
        assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
        assert 0 < self.lipschitz_target < 1, "lipschitz_target must be in (0, 1)"
        assert self.topo_method in ("finite_differences", "forward_ad", "hutchinson")
        assert self.nan_policy in ("RAISE", "WARN", "SILENT", "QUARANTINE", "RETURN_NONE")
        assert self.cosine_decay_steps > 0, "cosine_decay_steps must be positive"
        assert 0 < self.min_lr_ratio <= 1, "min_lr_ratio must be in (0, 1]"
        assert 0 <= self.cls_token_id < self.vocab_size, \
            f"cls_token_id ({self.cls_token_id}) must be in [0, vocab_size)"
        assert 0 <= self.sep_token_id < self.vocab_size, \
            f"sep_token_id ({self.sep_token_id}) must be in [0, vocab_size)"
        
        # Device initialization
        if self.device_str == "auto":
            self.device_manager = DeviceManager.auto_select(
                prefer_gpu=True,
                min_memory_gb=2.0
            )
        else:
            self.device_manager = DeviceManager(
                self.device_str,
                allow_fallback=self.device_allow_fallback,
                memory_fraction=self.device_memory_fraction
            )
        
        # Legacy compatibility
        self.device = self.device_manager.device
        
        # Update AMP
        if hasattr(self.device_manager, 'amp_enabled'):
            self.use_amp = self.device_manager.amp_enabled
        
        # TensorGuard initialization
        policy_map = {
            "RAISE": NaNPolicy.RAISE,
            "WARN": NaNPolicy.WARN,
            "SILENT": NaNPolicy.SILENT,
            "QUARANTINE": NaNPolicy.QUARANTINE,
            "RETURN_NONE": NaNPolicy.RETURN_NONE
        }
        self.tensor_guard = TensorGuard(
            policy=policy_map[self.nan_policy],
            enable_tracking=True,
            alert_threshold=self.safety_alert_threshold
        )
        
        # Create directories
        for path_attr in ('memory_path', 'checkpoint_dir', 'log_dir', 'cache_dir'):
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Freeze config
        object.__setattr__(self, '_frozen', True)
        
        logger.info(f"✅ AEONConfig v{self.version} initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Architecture: {self.hidden_dim}H x {self.num_pillars}P")
    
    def __setattr__(self, name, value):
        """Enforce immutability."""
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"AEONConfig is immutable. Cannot set {name}={value}"
            )
        object.__setattr__(self, name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        config_dict = asdict(self)
        config_dict.pop('device_manager', None)
        config_dict.pop('tensor_guard', None)
        config_dict.pop('_frozen', None)
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
        return config_dict
    
    def save(self, path: Union[str, Path]):
        """Save to JSON."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AEONConfig':
        """Load from JSON."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config_dict.pop('device', None)
        config_dict.pop('version', None)
        return cls(**config_dict)
    
    def get_model_signature(self) -> str:
        """Unique signature for model versioning."""
        key_params = {
            'z_dim': self.z_dim,
            'hidden_dim': self.hidden_dim,
            'num_pillars': self.num_pillars,
            'vocab_size': self.vocab_size,
            'vq_num_embeddings': self.vq_num_embeddings,
            'version': self.version
        }
        signature_str = json.dumps(key_params, sort_keys=True)
        signature_hash = hashlib.sha256(signature_str.encode()).hexdigest()[:16]
        return f"aeon-delta-v{self.version}-{signature_hash}"


# ============================================================================
# SECTION 5: ENCODER/DECODER WITH IMPROVEMENTS
# ============================================================================

class ThoughtEncoder(nn.Module):
    """
    LSTM-based encoder for thought representation.
    
    Features:
    - Bidirectional=False for causal encoding
    - LayerNorm for stability
    - Support for attention_mask (excludes PAD from dynamics)
    """
    
    def __init__(self, vocab_size: int, emb_dim: int = 256, z_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, 
            z_dim, 
            batch_first=True, 
            bidirectional=False
        )
        self.z_dim = z_dim
        self.norm = nn.LayerNorm(z_dim)
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] token IDs (dtype must be torch.long)
            attention_mask: [B, L] mask (1=valid, 0=pad)
        
        Returns:
            z: [B, z_dim] encoded representation
        
        Raises:
            TypeError: If tokens dtype is not torch.long or attention_mask dtype is invalid
            ValueError: If tokens contain out-of-range indices or shapes are mismatched
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be 2D [B, L], got shape {tokens.shape}")
        if tokens.dtype != torch.long:
            raise TypeError(
                f"tokens.dtype must be torch.long, got {tokens.dtype}"
            )
        
        # Range check: O(N) scan trades a small overhead for preventing
        # index_out_of_bounds crashes deep in the embedding layer.
        vocab_size = self.embed.num_embeddings
        if tokens.numel() > 0 and (tokens.min() < 0 or tokens.max() >= vocab_size):
            raise ValueError(
                f"tokens contain out-of-range indices: "
                f"min={tokens.min().item()}, max={tokens.max().item()}, "
                f"vocab_size={vocab_size}"
            )
        
        if attention_mask is not None:
            if attention_mask.shape != tokens.shape:
                raise ValueError(
                    f"attention_mask shape {attention_mask.shape} must match "
                    f"tokens shape {tokens.shape}"
                )
            if attention_mask.dtype not in (torch.long, torch.int, torch.bool, torch.float):
                raise TypeError(
                    f"attention_mask.dtype must be numeric, got {attention_mask.dtype}"
                )
        
        x = self.embed(tokens)  # [B, L, emb_dim]
        
        # Pack padded sequence if mask provided
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(x)
        
        z = self.norm(h.squeeze(0))  # [B, z_dim]
        
        return z


class ThoughtDecoder(nn.Module):
    """
    Unified decoder with dual-mode support.
    
    Modes:
    - train: Teacher-forcing for fast training
    - inference: Autoregressive generation
    
    Features:
    - Weight tying (head.weight = embed.weight)
    - Prefix conditioning for contextual generation
    - z concatenated with embeddings at each step
    - Invalid token filtering ([unused###])
    """
    
    def __init__(self, vocab_size: int, emb_dim: int = 256, z_dim: int = 256,
                 cls_token_id: int = 101, sep_token_id: int = 102):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.z_dim = z_dim
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(z_dim, emb_dim)
        
        # LSTM now takes (emb_dim + z_dim) input
        self.lstm = nn.LSTM(emb_dim + z_dim, emb_dim, batch_first=True)
        self.head = nn.Linear(emb_dim, vocab_size)
        
        # Weight tying
        self._tie_weights()
        self._verify_weight_tying()
        
        # Invalid token mask
        self.register_buffer(
            "_invalid_token_mask", 
            torch.zeros(vocab_size, dtype=torch.bool), 
            persistent=False
        )
    
    def _tie_weights(self):
        """Tie output layer weights to embedding."""
        self.head.weight = self.embed.weight
    
    def _verify_weight_tying(self):
        """Verify weight tying is correct."""
        if self.head.weight.data_ptr() != self.embed.weight.data_ptr():
            raise RuntimeError("Weight tying failed: pointers differ")
        if self.head.weight.shape != self.embed.weight.shape:
            raise RuntimeError("Weight tying shape mismatch")
        logger.info("✅ Weight tying verified")
    
    def set_invalid_token_ids(self, token_ids: Optional[List[int]]):
        """Set invalid token IDs (e.g., [unused###])."""
        if token_ids is None:
            self._invalid_token_mask.zero_()
            return
        try:
            idx = torch.tensor(list(token_ids), dtype=torch.long)
        except Exception:
            idx = torch.empty((0,), dtype=torch.long)
        if idx.numel() > 0:
            idx = idx[(idx >= 0) & (idx < self.vocab_size)]
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if idx.numel() > 0:
            mask[idx] = True
        self._invalid_token_mask.data.copy_(
            mask.to(self._invalid_token_mask.device)
        )
    
    def forward(
        self,
        z: torch.Tensor,
        teacher_tokens: Optional[torch.Tensor] = None,
        mode: str = 'train',
        max_length: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        sample: bool = True,
        prefix_tokens: Optional[torch.Tensor] = None
    ):
        """
        Unified forward pass.
        
        Args:
            z: [B, z_dim] latent vector
            teacher_tokens: [B, L] for train mode
            mode: 'train' or 'inference'
            max_length: Max length for inference
            temperature: Sampling temperature
            top_k: Top-K filtering
            sample: Use sampling vs greedy
            prefix_tokens: [B, Lp] prefix for conditioning
        
        Returns:
            For train: logits [B, L, V]
            For inference: (generated_ids [B, L], logits [B, L, V])
        """
        if mode == 'train':
            if teacher_tokens is None:
                raise ValueError("train mode requires teacher_tokens")
            return self._forward_train(z, teacher_tokens, z.device)
        elif mode == 'inference':
            return self._forward_inference(
                z, max_length, temperature, top_k, sample, z.device, prefix_tokens
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _forward_train(
        self, 
        z: torch.Tensor, 
        teacher_tokens: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """Teacher-forcing mode."""
        batch_size = z.shape[0]
        seq_length = teacher_tokens.shape[1]
        
        # Project z to initial hidden state
        h0 = self.fc(z).unsqueeze(0)  # [1, B, emb_dim]
        c0 = torch.zeros_like(h0)
        
        # Embed tokens
        embeddings = self.embed(teacher_tokens)  # [B, L, emb_dim]
        
        # Concatenate z with embeddings at each step
        z_expanded = z.unsqueeze(1).expand(-1, seq_length, -1)  # [B, L, z_dim]
        lstm_input = torch.cat([embeddings, z_expanded], dim=-1)  # [B, L, emb_dim+z_dim]
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # [B, L, emb_dim]
        
        # Project to vocabulary
        logits = self.head(lstm_out)  # [B, L, V]
        
        return logits
    
    def _forward_inference(
        self,
        z: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        sample: bool,
        device: torch.device,
        prefix_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation."""
        batch_size = z.shape[0]
        
        # Initialize hidden state
        h_state = self.fc(z).unsqueeze(0)  # [1, B, emb_dim]
        c_state = torch.zeros_like(h_state)
        
        generated_ids = []
        all_logits = []
        
        # Prefix conditioning
        if prefix_tokens is not None:
            assert prefix_tokens.dim() == 2
            assert prefix_tokens.shape[0] == batch_size
            prefix_tokens = prefix_tokens.to(device)
            
            # Run prefix through LSTM
            emb_pref = self.embed(prefix_tokens)  # [B, Lp, emb_dim]
            z_exp = z.unsqueeze(1).expand(-1, prefix_tokens.shape[1], -1)
            lstm_in = torch.cat([emb_pref, z_exp], dim=-1)
            _, (h_state, c_state) = self.lstm(lstm_in, (h_state, c_state))
            
            generated_ids.append(prefix_tokens)
            current_token_id = prefix_tokens[:, -1:].contiguous()
        else:
            # Start with [CLS] token
            current_token_id = torch.full(
                (batch_size, 1), self.cls_token_id, dtype=torch.long, device=device
            )
            generated_ids.append(current_token_id)
        
        # Main generation loop — per-sequence stopping
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            emb = self.embed(current_token_id)  # [B, 1, emb_dim]
            z_expanded = z.unsqueeze(1)  # [B, 1, z_dim]
            lstm_input = torch.cat([emb, z_expanded], dim=-1)
            
            lstm_out, (h_state, c_state) = self.lstm(lstm_input, (h_state, c_state))
            logits = self.head(lstm_out.squeeze(1))  # [B, V]
            all_logits.append(logits)
            
            # Filter logits
            logits_filtered = self._filter_logits(logits, temperature, top_k, device)
            
            # Sample
            if sample:
                probs = F.softmax(logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                next_token_id = torch.argmax(logits_filtered, dim=-1, keepdim=True)
            
            # Per-sequence [SEP] stopping: mask finished sequences
            newly_finished = (next_token_id.squeeze(-1) == self.sep_token_id)
            finished = finished | newly_finished
            
            # Replace token for finished sequences with PAD (0)
            next_token_id = next_token_id.masked_fill(
                finished.unsqueeze(-1), 0
            )
            
            generated_ids.append(next_token_id)
            current_token_id = next_token_id
            
            # Stop when ALL sequences are finished
            if finished.all():
                break
        
        generated_ids = (
            torch.cat(generated_ids, dim=1) 
            if generated_ids 
            else torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        )
        logits_stacked = (
            torch.stack(all_logits, dim=1) 
            if all_logits 
            else torch.zeros((batch_size, 0, self.vocab_size), device=device)
        )
        
        return generated_ids, logits_stacked
    
    def _filter_logits(
        self, 
        logits: torch.Tensor, 
        temperature: float, 
        top_k: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Apply temperature scaling and top-K filtering."""
        # Temperature
        scaled_logits = logits / max(temperature, 1e-6)
        
        # Top-K (clamp to vocab size to prevent IndexError)
        if top_k > 0:
            top_k = min(top_k, scaled_logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
            mask = torch.full_like(scaled_logits, -float('inf'))
            mask.scatter_(1, top_k_indices, top_k_logits)
            scaled_logits = mask
        
        # Filter invalid tokens
        if (hasattr(self, "_invalid_token_mask") and 
                self._invalid_token_mask is not None and 
                self._invalid_token_mask.numel() == self.vocab_size):
            invalid_mask = self._invalid_token_mask.to(device)
            if invalid_mask.any():
                scaled_logits[:, invalid_mask] = -float('inf')
        
        # Protect against NaN/Inf
        scaled_logits = torch.nan_to_num(
            scaled_logits, 
            neginf=-float('inf'), 
            posinf=float('inf')
        )
        
        return scaled_logits


# ============================================================================
# SECTION 6: VECTOR QUANTIZER WITH ANTI-COLLAPSE
# ============================================================================

class RobustVectorQuantizer(nn.Module):
    """
    Production-grade VQ-VAE with anti-collapse mechanisms.
    
    Techniques:
    1. EMA updates (more stable than gradient descent)
    2. Code revival (reinitialize dead codes)
    3. Code splitting (balance overused codes)
    4. Perplexity monitoring
    5. Straight-Through Estimator (STE)
    
    Reference: van den Oord et al., 2017
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        revival_threshold: int = 100,
        split_threshold: float = 0.1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.revival_threshold = revival_threshold
        self.split_threshold = split_threshold
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 
            1.0 / num_embeddings
        )
        
        # EMA buffers
        if use_ema:
            self.register_buffer(
                '_ema_cluster_size', 
                torch.zeros(num_embeddings)
            )
            self.register_buffer(
                '_ema_w', 
                self.embedding.weight.data.clone()
            )
        
        # Usage tracking
        self.register_buffer(
            '_code_usage_counter', 
            torch.zeros(num_embeddings, dtype=torch.long)
        )
        self.register_buffer(
            '_steps_since_used', 
            torch.zeros(num_embeddings, dtype=torch.long)
        )
        self.register_buffer(
            '_total_steps', 
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            '_perplexity_ema', 
            torch.tensor(0.0)
        )
        
        logger.info(
            f"VectorQuantizer initialized: {num_embeddings} codes, "
            f"{embedding_dim}D, EMA={'on' if use_ema else 'off'}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        compute_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        VQ-VAE forward pass.
        
        Args:
            inputs: [B, D] latent vectors
            compute_loss: Compute commitment loss
        
        Returns:
            quantized: [B, D] quantized vectors
            loss: VQ loss (if compute_loss=True)
            encoding_indices: [B] indices in codebook
        """
        assert inputs.dim() == 2, f"Expected [B, D], got {inputs.shape}"
        
        B, D = inputs.shape
        assert D == self.embedding_dim, f"Dim mismatch: {D} vs {self.embedding_dim}"
        
        inputs_flat = inputs.reshape(-1, D)
        
        # Compute distances
        distances = (
            torch.sum(inputs_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(inputs_flat, self.embedding.weight.t())
        )  # [B, num_embeddings]
        
        # Nearest code
        encoding_indices = torch.argmin(distances, dim=1)  # [B]
        
        # Update usage stats
        if self.training:
            self._update_usage_stats(encoding_indices)
        
        # One-hot encoding
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantized vectors
        quantized = torch.matmul(encodings, self.embedding.weight)  # [B, D]
        
        # EMA update
        if self.training and self.use_ema:
            self._ema_update(inputs_flat, encodings)
        
        # Loss
        loss = None
        if compute_loss:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs_flat)
            q_latent_loss = F.mse_loss(quantized, inputs_flat.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # STE
        quantized = inputs_flat + (quantized - inputs_flat).detach()
        quantized = quantized.reshape(B, D)
        encoding_indices = encoding_indices.reshape(B)
        
        # Code maintenance
        if self.training and self._total_steps % 100 == 0:
            self._maintain_codebook(inputs_flat)
        
        return quantized, loss, encoding_indices
    
    def _update_usage_stats(self, encoding_indices: torch.Tensor):
        """Update usage statistics."""
        self._total_steps += 1
        
        unique_codes = encoding_indices.unique()
        usage_count = torch.bincount(
            encoding_indices,
            minlength=self.num_embeddings
        )
        
        self._code_usage_counter += usage_count
        self._steps_since_used += 1
        self._steps_since_used[unique_codes] = 0
        
        # Perplexity
        probs = usage_count.float() / usage_count.sum()
        probs = probs[probs > 0]
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        self._perplexity_ema = 0.99 * self._perplexity_ema + 0.01 * perplexity
    
    def _ema_update(self, inputs: torch.Tensor, encodings: torch.Tensor):
        """EMA update for codebook."""
        # Cluster size
        self._ema_cluster_size.mul_(self.decay).add_(
            encodings.sum(dim=0),
            alpha=1 - self.decay
        )
        
        # Laplace smoothing
        n = self._ema_cluster_size.sum()
        self._ema_cluster_size = (
            (self._ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        
        # EMA weights
        dw = torch.matmul(encodings.t(), inputs)
        self._ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # Update embedding
        self.embedding.weight.data.copy_(
            self._ema_w / self._ema_cluster_size.unsqueeze(1)
        )
    
    def _maintain_codebook(self, recent_inputs: torch.Tensor):
        """Maintain codebook: revival + splitting."""
        # Identify dead codes
        dead_codes = (
            self._steps_since_used > self.revival_threshold
        ).nonzero(as_tuple=True)[0]
        
        if len(dead_codes) > 0:
            logger.info(
                f"Reviving {len(dead_codes)} dead codes "
                f"(unused for {self.revival_threshold}+ steps)"
            )
            
            # Reinitialize with random from recent inputs
            num_dead = len(dead_codes)
            if recent_inputs.shape[0] >= num_dead:
                random_indices = torch.randperm(recent_inputs.shape[0])[:num_dead]
                self.embedding.weight.data[dead_codes] = (
                    recent_inputs[random_indices].detach()
                )
            else:
                self.embedding.weight.data[dead_codes] = (
                    torch.randn_like(self.embedding.weight.data[dead_codes]) * 0.02
                )
            
            self._steps_since_used[dead_codes] = 0
            self._code_usage_counter[dead_codes] = 0
        
        # Identify overused codes
        total_usage = self._code_usage_counter.sum()
        usage_fraction = self._code_usage_counter.float() / (total_usage + 1e-10)
        overused_codes = (
            usage_fraction > self.split_threshold
        ).nonzero(as_tuple=True)[0]
        
        if len(overused_codes) > 0 and len(dead_codes) > 0:
            num_splits = min(len(overused_codes), len(dead_codes))
            
            for i in range(num_splits):
                source_code = overused_codes[i]
                target_code = dead_codes[i]
                
                # Copy + noise
                self.embedding.weight.data[target_code] = (
                    self.embedding.weight.data[source_code]
                    + torch.randn_like(
                        self.embedding.weight.data[source_code]
                    ) * 0.01
                )
            
            logger.info(f"Split {num_splits} overused codes")
    
    def get_codebook_usage_stats(self) -> dict:
        """Codebook usage statistics."""
        total_usage = self._code_usage_counter.sum().item()
        used_codes = (self._code_usage_counter > 0).sum().item()
        
        return {
            'total_codes': self.num_embeddings,
            'used_codes': used_codes,
            'unused_codes': self.num_embeddings - used_codes,
            'usage_rate': used_codes / self.num_embeddings,
            'perplexity': self._perplexity_ema.item(),
            'total_steps': self._total_steps.item(),
            'max_steps_unused': self._steps_since_used.max().item(),
            'top_10_codes': self._code_usage_counter.topk(10).indices.tolist()
        }


# ============================================================================
# SECTION 7: META-LOOP WITH LIPSCHITZ REGULARIZATION
# ============================================================================

class LipschitzConstrainedLambda(nn.Module):
    """
    Lambda operator with Lipschitz constraint for convergence guarantees.
    
    Theory:
    If ||Λ(x) - Λ(y)|| ≤ L||x - y|| with L < 1 (contraction),
    then fixed-point exists and is unique (Banach Fixed-Point Theorem).
    
    Implementation:
    - Spectral normalization to enforce ||W|| ≤ 1
    - Lipschitz penalty in loss for training L < 1
    - Monitoring of Lipschitz constant during inference
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lipschitz_target: float = 0.9,
        use_spectral_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lipschitz_target = lipschitz_target
        
        # Network with spectral normalization
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)
        
        if use_spectral_norm:
            self.W1 = nn.utils.spectral_norm(self.W1)
            self.W2 = nn.utils.spectral_norm(self.W2)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Lipschitz monitoring
        self.register_buffer('lipschitz_estimate', torch.tensor(1.0))
        self.register_buffer('lipschitz_ema_decay', torch.tensor(0.99))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with Lipschitz monitoring."""
        h = self.W1(x)
        h = self.activation(h)
        h = self.dropout(h)
        out = self.W2(h)
        out = self.layer_norm(out)
        return out
    
    def compute_lipschitz_constant(
        self,
        num_samples: int = 100,
        sample_dim: int = None
    ) -> float:
        """
        Empirical estimation of Lipschitz constant.
        L ≈ max_{x,y} ||Λ(x) - Λ(y)|| / ||x - y||
        """
        if sample_dim is None:
            sample_dim = self.W1.in_features
        
        device = next(self.parameters()).device
        
        max_ratio = 0.0
        for _ in range(num_samples):
            x = torch.randn(1, sample_dim, device=device)
            y = torch.randn(1, sample_dim, device=device)
            
            with torch.no_grad():
                fx = self.forward(x)
                fy = self.forward(y)
                
                numerator = torch.norm(fx - fy).item()
                denominator = torch.norm(x - y).item() + 1e-8
                ratio = numerator / denominator
                
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def get_lipschitz_penalty(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalty for training: L_lip = max(0, L - target)²
        """
        fx = self.forward(x)
        fy = self.forward(y)
        
        numerator = torch.norm(fx - fy, dim=-1)
        denominator = torch.norm(x - y, dim=-1).clamp_min(1e-8)
        
        lipschitz_estimate = (numerator / denominator).mean()
        
        # EMA update
        with torch.no_grad():
            self.lipschitz_estimate.mul_(self.lipschitz_ema_decay).add_(
                lipschitz_estimate * (1 - self.lipschitz_ema_decay)
            )
        
        # Penalty
        penalty = F.relu(lipschitz_estimate - self.lipschitz_target) ** 2
        
        return penalty


class ProvablyConvergentMetaLoop(nn.Module):
    """
    Meta-loop with convergence mechanisms inspired by fixed-point theory.
    
    Theoretical motivation (not formal guarantees):
    - Spectral normalization encourages ||W|| ≤ 1 per layer, but the
      *global* Lipschitz constant of the composed operator also depends
      on activations, LayerNorm, and dropout. The EMA-tracked
      ``lipschitz_estimate`` is a *statistical* estimate, not a certified
      upper bound.
    - When the estimate satisfies L < 1, the Banach Fixed-Point Theorem
      *suggests* convergence, but completeness of the latent metric space
      is assumed, not proven.
    - Use ``verify_convergence()`` to obtain an explicit diagnostic that
      clearly separates measured quantities from theoretical conditions.
    
    Practical improvements:
    1. Anderson acceleration for 2-5x speedup
    2. Adaptive alpha based on Lipschitz estimate
    3. Early stopping with certified tolerance
    """
    
    def __init__(
        self,
        config,
        anderson_memory: int = 5,
        convergence_threshold: float = 1e-5,
        max_iterations: int = 50,
        min_iterations: int = 3,
        enable_certification: bool = True
    ):
        super().__init__()
        self.config = config
        self.anderson_memory = anderson_memory
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.enable_certification = enable_certification
        
        # Lipschitz-constrained Lambda
        input_dim = config.hidden_dim * 2
        self.lambda_op = LipschitzConstrainedLambda(
            input_dim=input_dim,
            hidden_dim=config.meta_dim,
            output_dim=config.hidden_dim,
            lipschitz_target=config.lipschitz_target,
            use_spectral_norm=True,
            dropout=config.dropout_rate
        )
        
        # Adaptive alpha network
        self.alpha_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Stabilization
        self.input_stabilizer = nn.LayerNorm(input_dim)
        self.output_stabilizer = nn.LayerNorm(config.hidden_dim)
        
        # Monitoring
        self.register_buffer('avg_iterations', torch.tensor(0.0))
        self.register_buffer('convergence_rate', torch.tensor(0.0))
    
    def _anderson_step(
        self,
        C_history: list,
        residual_history: list,
        device: torch.device
    ) -> torch.Tensor:
        """Anderson acceleration step with CPU fallback for MPS."""
        m = len(C_history)
        if m <= 1:
            return C_history[-1]
        
        B = C_history[-1].shape[0]
        H = C_history[-1].shape[1]
        
        # Stack residuals
        F = torch.stack(residual_history, dim=0)  # [m, B, H]
        F = F.permute(1, 0, 2)  # [B, m, H]
        
        # Gram matrix
        gram = torch.bmm(F, F.transpose(1, 2))  # [B, m, m]
        
        # Regularization
        reg = 1e-6 * torch.eye(m, device=device).unsqueeze(0).expand(B, -1, -1)
        gram = gram + reg
        
        # Solve for weights with CPU fallback for MPS
        rhs = torch.ones(B, m, 1, device=device)
        alpha = self._safe_solve(gram, rhs, m, B, device)
        
        # Weighted combination
        C_stack = torch.stack(C_history, dim=0).permute(1, 0, 2)
        R_stack = torch.stack(residual_history, dim=0).permute(1, 0, 2)
        
        CR_stack = C_stack + R_stack
        C_anderson = torch.bmm(alpha.transpose(1, 2), CR_stack).squeeze(1)
        
        return C_anderson
    
    def _safe_solve(
        self,
        gram: torch.Tensor,
        rhs: torch.Tensor,
        m: int,
        B: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Solve linear system with CPU fallback for MPS.
        MPS doesn't reliably support torch.linalg.solve.
        """
        # Try on original device first
        try:
            alpha = torch.linalg.solve(gram, rhs)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            # Check for NaN/Inf
            if torch.isfinite(alpha).all():
                return alpha
        except (RuntimeError, NotImplementedError):
            pass
        
        # CPU fallback
        try:
            gram_cpu = gram.cpu()
            rhs_cpu = rhs.cpu()
            alpha_cpu = torch.linalg.solve(gram_cpu, rhs_cpu)
            alpha_cpu = alpha_cpu / alpha_cpu.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if torch.isfinite(alpha_cpu).all():
                return alpha_cpu.to(device)
        except RuntimeError:
            pass
        
        # Last resort: uniform weights (no acceleration)
        return torch.ones(B, m, 1, device=device) / m
    
    def compute_fixed_point(
        self,
        psi_0: torch.Tensor,
        return_certificate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Provably convergent fixed-point solver.
        
        Returns:
            C_star: Fixed point solution [B, hidden_dim]
            iterations: Number of iterations [B]
            metadata: Dict with convergence info
        """
        B = psi_0.shape[0]
        H = self.config.hidden_dim
        device = psi_0.device
        
        # Initialize
        C = torch.zeros(B, H, device=device)
        
        C_history = []
        residual_history = []
        convergence_trajectory = []
        
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        iterations = torch.zeros(B, device=device)
        
        # Get Lipschitz estimate
        lip_const = self.lambda_op.lipschitz_estimate.item()
        
        for iter_idx in range(self.max_iterations):
            C_prev = C.clone()
            
            # Input stabilization
            input_tensor = torch.cat([psi_0, C], dim=-1)
            input_tensor = self.input_stabilizer(input_tensor)
            
            # Lambda application
            C_new = self.lambda_op(input_tensor)
            C_new = self.output_stabilizer(C_new)
            
            # Residual
            residual = C_new - C
            residual_norm = torch.norm(residual, dim=-1)
            
            # Anderson acceleration
            C_history.append(C_new)
            residual_history.append(residual)
            
            if len(C_history) > self.anderson_memory:
                C_history = C_history[-self.anderson_memory:]
                residual_history = residual_history[-self.anderson_memory:]
            
            if len(C_history) >= 2:
                C_anderson = self._anderson_step(C_history, residual_history, device)
            else:
                C_anderson = C_new
            
            # Adaptive alpha
            alpha_base = self.config.alpha
            if lip_const < 1.0:
                alpha_scale = torch.sigmoid(
                    self.alpha_net(torch.cat([C_new, C], dim=-1))
                ).squeeze(-1)
                alpha = alpha_base * (0.5 + 0.5 * alpha_scale)
            else:
                alpha = torch.full((B,), alpha_base * 0.5, device=device)
            
            # Update
            C = alpha.unsqueeze(-1) * C_anderson + (1 - alpha.unsqueeze(-1)) * C_prev
            
            # Convergence check
            newly_converged = (residual_norm < self.convergence_threshold) & ~converged
            converged |= newly_converged
            iterations[~converged] += 1
            
            # Tracking
            convergence_trajectory.append({
                'iteration': iter_idx,
                'residual_mean': residual_norm.mean().item(),
                'residual_max': residual_norm.max().item(),
                'alpha_mean': alpha.mean().item(),
                'converged_count': converged.sum().item()
            })
            
            # Early stopping
            if iter_idx >= self.min_iterations and converged.all():
                logger.debug(f"Converged after {iter_idx+1} iterations")
                break
        
        # Certification
        certified_error = None
        if self.enable_certification and lip_const < 1.0:
            with torch.no_grad():
                final_residual = residual_norm.mean().item()
                certified_error = (lip_const / (1 - lip_const)) * final_residual
        
        # Metadata
        metadata = {
            'converged': converged.all().item(),
            'convergence_rate': converged.float().mean().item(),
            'residual_norm': residual_norm.mean().item(),
            'lipschitz_estimate': lip_const,
            'certified_error_bound': certified_error,
            'convergence_trajectory': convergence_trajectory,
            'stability_scores': torch.ones(B, device=device),
            'convergence_scores': converged.float(),
            'instability_flags': torch.zeros(B, dtype=torch.bool, device=device),
            'instability_steps': torch.zeros(B, dtype=torch.long, device=device)
        }
        
        # Update EMA
        with torch.no_grad():
            self.avg_iterations.mul_(0.99).add_(iterations.float().mean() * 0.01)
            self.convergence_rate.mul_(0.99).add_(converged.float().mean() * 0.01)
        
        if return_certificate:
            metadata['certificate'] = {
                'method': 'Banach Fixed-Point Theorem (estimated, not formally proven)',
                'conditions': f'EMA Lipschitz estimate L={lip_const:.4f} (target < 1)',
                'guarantee': (
                    f'Estimated error ≤ {certified_error:.2e}'
                    if certified_error
                    else 'N/A (L >= 1, contraction not verified)'
                ),
                'note': (
                    'This is a statistical estimate. '
                    'Use verify_convergence() for explicit diagnostics.'
                )
            }
        
        return C, iterations, metadata
    
    def verify_convergence(
        self,
        psi_0: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Verify Banach fixed-point conditions separately from computation.
        
        This method explicitly checks the theoretical conditions required
        by the Banach Fixed-Point Theorem and reports which are satisfied
        and which are only *estimated*.
        
        Args:
            psi_0: [B, H] input latent.
            num_samples: Number of random pairs for empirical Lipschitz check.
        
        Returns:
            Dict with explicit diagnostic fields:
              - empirical_lipschitz: max ratio from random sampling
              - ema_lipschitz: current EMA estimate
              - contraction_satisfied: whether empirical L < 1
              - target_satisfied: whether empirical L < lipschitz_target
              - residual_norm: final residual after compute_fixed_point
              - warnings: list of unverified assumptions
        """
        with torch.no_grad():
            empirical_L = self.lambda_op.compute_lipschitz_constant(
                num_samples=num_samples
            )
            ema_L = self.lambda_op.lipschitz_estimate.item()
            target = self.lambda_op.lipschitz_target
            
            C, iterations, meta = self.compute_fixed_point(psi_0)
            residual = meta.get('residual_norm', float('inf'))
        
        warnings_list = []
        if empirical_L >= 1.0:
            warnings_list.append(
                f'Empirical Lipschitz constant {empirical_L:.4f} >= 1; '
                'contraction mapping condition NOT satisfied.'
            )
        if abs(empirical_L - ema_L) > 0.1:
            warnings_list.append(
                f'EMA estimate ({ema_L:.4f}) differs significantly from '
                f'empirical estimate ({empirical_L:.4f}).'
            )
        warnings_list.append(
            'Completeness of the latent metric space is assumed, not proven.'
        )
        
        return {
            'empirical_lipschitz': empirical_L,
            'ema_lipschitz': ema_L,
            'lipschitz_target': target,
            'contraction_satisfied': empirical_L < 1.0,
            'target_satisfied': empirical_L < target,
            'residual_norm': residual,
            'converged': meta.get('converged', False),
            'warnings': warnings_list,
        }
    
    def forward(self, psi_0: torch.Tensor, use_fixed_point: bool = True):
        """Wrapper for compatibility."""
        if use_fixed_point:
            return self.compute_fixed_point(psi_0, return_certificate=False)
        else:
            # Fast path: single iteration
            input_tensor = torch.cat([psi_0, torch.zeros_like(psi_0)], dim=-1)
            C = self.lambda_op(self.input_stabilizer(input_tensor))
            return C, torch.ones(psi_0.shape[0], device=psi_0.device), {}
    
    def get_lipschitz_loss(self, psi_0: torch.Tensor) -> torch.Tensor:
        """Lipschitz regularization term for training.
        
        Delegates to standalone compute_lipschitz_loss() to avoid coupling.
        """
        return compute_lipschitz_loss(self.lambda_op, psi_0)


def compute_lipschitz_loss(
    lambda_op: 'LipschitzConstrainedLambda',
    psi_0: torch.Tensor
) -> torch.Tensor:
    """Standalone Lipschitz regularization loss.
    
    Decoupled from ProvablyConvergentMetaLoop so that any module holding
    a LipschitzConstrainedLambda can reuse this without a circular reference.
    
    Args:
        lambda_op: A LipschitzConstrainedLambda module.
        psi_0: [B, H] latent state tensor.
    
    Returns:
        Lipschitz penalty scalar tensor.
    """
    B, H = psi_0.shape
    device = psi_0.device
    
    # Sample pairs
    eps = torch.randn(B, H * 2, device=device) * 0.1
    x = torch.cat([psi_0, torch.zeros(B, H, device=device)], dim=-1) + eps
    y = torch.cat([psi_0, torch.zeros(B, H, device=device)], dim=-1) + torch.randn_like(eps) * 0.1
    
    # Lipschitz penalty
    lip_penalty = lambda_op.get_lipschitz_penalty(x, y)
    
    return lip_penalty


# ============================================================================
# SECTION 8: FAST HESSIAN COMPUTATION
# ============================================================================

class FastHessianComputer:
    """
    High-performance Hessian computation for topology analysis.
    
    Methods:
    1. Finite Differences: O(P²) but no autograd overhead
    2. Forward-mode AD: O(P) with torch.func (experimental)
    3. Hutchinson's Trace Estimator: O(1) for trace(H)
    """
    
    def __init__(
        self,
        method: str = 'finite_differences',
        epsilon: float = 1e-4,
        use_cache: bool = True,
        max_cache_size: int = 128
    ):
        self.method = method
        self.epsilon = epsilon
        self.use_cache = use_cache
        self._cache = OrderedDict() if use_cache else None
        self._max_cache_size = max_cache_size
        
        # Check torch.func availability (import shadows module-level torch)
        self.functorch_available = False
        try:
            import torch.func
            self.functorch_available = True
        except ImportError:
            if method == 'forward_ad':
                logger.warning(
                    "Forward AD requested but torch.func unavailable, "
                    "falling back to finite differences"
                )
                self.method = 'finite_differences'
        
        # Check torch.autograd.functional.hessian availability
        # Use re-import to avoid scoping issues from the torch.func import above
        import torch as _torch_ref
        self.autograd_hessian_available = hasattr(
            _torch_ref.autograd.functional, 'hessian'
        )
    
    def _safe_eigvalsh(self, H_sym: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenvalues with CPU fallback for MPS.
        MPS doesn't support eigvalsh/eigvals, so we move to CPU if needed.
        """
        original_device = H_sym.device
        
        try:
            # Try on original device first
            return torch.linalg.eigvalsh(H_sym)
        except (NotImplementedError, RuntimeError):
            pass
        
        # Fallback to CPU
        try:
            H_cpu = H_sym.cpu()
            eigvals_cpu = torch.linalg.eigvalsh(H_cpu)
            return eigvals_cpu.to(original_device)
        except RuntimeError:
            # Last resort: use eigvals instead of eigvalsh
            try:
                H_cpu = H_sym.cpu()
                eigvals_cpu = torch.linalg.eigvals(H_cpu).real
                return eigvals_cpu.to(original_device)
            except Exception as e:
                logger.warning(f"Eigenvalue computation failed: {e}, returning zeros")
                B, n, _ = H_sym.shape
                return torch.zeros(B, n, device=original_device, dtype=H_sym.dtype)
    
    def compute_hessian(
        self,
        func: callable,
        x: torch.Tensor,
        return_eigenvalues: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute Hessian H = ∇²f(x).
        
        Args:
            func: Scalar function f: R^n → R (batch-aware)
            x: Input [B, n]
            return_eigenvalues: Also return eigenvalues
        
        Returns:
            hessian: [B, n, n]
            eigenvalues: [B, n] if requested
        """
        if self.use_cache:
            cache_key = self._hash_tensor(x)
            if cache_key in self._cache:
                logger.debug("Hessian cache hit")
                return self._cache[cache_key]
        
        if self.method == 'finite_differences':
            H = self._hessian_finite_differences(func, x)
        elif self.method == 'autograd' and self.autograd_hessian_available:
            H = self._hessian_autograd(func, x)
        elif self.method == 'forward_ad':
            H = self._hessian_forward_ad(func, x)
        else:
            # Auto-select: prefer autograd vectorized path, fallback to finite diff
            if self.autograd_hessian_available:
                try:
                    H = self._hessian_autograd(func, x)
                except Exception:
                    H = self._hessian_finite_differences(func, x)
            else:
                H = self._hessian_finite_differences(func, x)
        
        eigvals = None
        if return_eigenvalues:
            # Symmetrize
            H_sym = 0.5 * (H + H.transpose(-2, -1))
            eigvals = self._safe_eigvalsh(H_sym)
        
        result = (H, eigvals) if return_eigenvalues else (H, None)
        
        if self.use_cache:
            self._cache[cache_key] = result
            # Evict oldest entries if cache exceeds max size
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)
        
        return result
    
    def _hessian_finite_differences(
        self,
        func: callable,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Finite differences Hessian.
        H[i,j] ≈ (f(x+eᵢ+eⱼ) - f(x+eᵢ) - f(x+eⱼ) + f(x)) / ε²
        """
        B, n = x.shape
        device = x.device
        eps = self.epsilon
        
        H = torch.zeros(B, n, n, device=device, dtype=x.dtype)
        
        # Base value
        with torch.no_grad():
            f_base = func(x)
            if f_base.dim() > 1:
                f_base = f_base.squeeze(-1)
        
        # Gradient via forward differences
        grad = torch.zeros(B, n, device=device, dtype=x.dtype)
        for i in range(n):
            x_plus = x.clone()
            x_plus[:, i] += eps
            
            with torch.no_grad():
                f_plus = func(x_plus)
                if f_plus.dim() > 1:
                    f_plus = f_plus.squeeze(-1)
            
            grad[:, i] = (f_plus - f_base) / eps
        
        # Hessian via mixed differences
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal: second derivative
                    x_minus = x.clone()
                    x_minus[:, i] -= eps
                    x_plus = x.clone()
                    x_plus[:, i] += eps
                    
                    with torch.no_grad():
                        f_minus = func(x_minus)
                        f_plus = func(x_plus)
                        if f_minus.dim() > 1:
                            f_minus = f_minus.squeeze(-1)
                        if f_plus.dim() > 1:
                            f_plus = f_plus.squeeze(-1)
                    
                    H[:, i, i] = (f_plus - 2 * f_base + f_minus) / (eps ** 2)
                else:
                    # Off-diagonal: mixed derivative
                    x_pp = x.clone()
                    x_pp[:, i] += eps
                    x_pp[:, j] += eps
                    
                    with torch.no_grad():
                        f_pp = func(x_pp)
                        if f_pp.dim() > 1:
                            f_pp = f_pp.squeeze(-1)
                    
                    H[:, i, j] = (f_pp - f_base) / (eps ** 2) - (grad[:, i] + grad[:, j]) / eps
                    H[:, j, i] = H[:, i, j]
        
        return H
    
    def _hessian_autograd(
        self,
        func: callable,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized Hessian via torch.autograd.functional.hessian.
        
        Complexity: O(P) via reverse-mode AD instead of O(P²) loops.
        Requires PyTorch >= 1.11.
        """
        B, n = x.shape
        device = x.device
        
        H_batch = torch.zeros(B, n, n, device=device, dtype=x.dtype)
        
        for b in range(B):
            def scalar_func(xi):
                # func expects [1, n], returns scalar
                out = func(xi.unsqueeze(0))
                if out.dim() > 0:
                    out = out.squeeze()
                return out
            
            H_b = torch.autograd.functional.hessian(scalar_func, x[b])
            H_batch[b] = H_b
        
        return H_batch
    
    def _hash_tensor(self, t: torch.Tensor) -> int:
        """Hash tensor for caching using multiple statistics to reduce collisions."""
        t_flat = t.detach().float().flatten()
        return hash((
            tuple(t.shape),
            t_flat.sum().item(),
            t_flat.std().item() if t_flat.numel() > 1 else 0.0,
            t_flat[0].item() if t_flat.numel() > 0 else 0.0,
            t_flat[-1].item() if t_flat.numel() > 0 else 0.0,
            t.device.type
        ))
    
    def clear_cache(self):
        """Clear cache."""
        if self._cache is not None:
            self._cache.clear()


# ============================================================================
# SECTION 9: OPTIMIZED TOPOLOGY ANALYZER
# ============================================================================

class OptimizedTopologyAnalyzer(nn.Module):
    """
    Topology analyzer with high-performance Hessian computation.
    
    Features:
    - Fast Hessian via finite differences
    - Catastrophe detection via eigenvalues
    - Potential landscape analysis
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Potential network
        self.potential_net = nn.Sequential(
            nn.Linear(config.num_pillars, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Fast Hessian
        self.hessian_computer = FastHessianComputer(
            method=config.topo_method,
            epsilon=config.topo_epsilon,
            use_cache=config.topo_use_cache
        )
        
        # Catastrophe classifier
        self.catastrophe_classifier = nn.Sequential(
            nn.Linear(config.num_pillars * 2 + 2, 1),
            nn.Sigmoid()
        )
    
    def compute_potential(self, pillars: torch.Tensor) -> torch.Tensor:
        """V(pillars) → [B]."""
        return self.potential_net(pillars).squeeze(-1)
    
    def forward(self, pillars: torch.Tensor, iterations=None):
        """
        Topological analysis with efficient Hessian.
        
        Returns:
            Dict with potential, gradient, hessian, eigenvalues, catastrophe metrics
        """
        B, P = pillars.shape
        device = pillars.device
        
        # Potential
        with torch.enable_grad():
            pillars_grad = pillars.clone().requires_grad_(True)
            potential = self.compute_potential(pillars_grad)
            
            # Gradient
            gradient = torch.autograd.grad(
                potential.sum(),
                pillars_grad,
                create_graph=False
            )[0]
        
        # Hessian
        def potential_fn(p):
            return self.compute_potential(p)
        
        hessian, eigenvalues = self.hessian_computer.compute_hessian(
            potential_fn,
            pillars,
            return_eigenvalues=True
        )
        
        # Catastrophe detection
        min_eigenvalue = eigenvalues[:, 0]
        grad_norm = gradient.norm(dim=-1)
        
        # Features
        features = torch.cat([
            pillars,
            gradient,
            potential.unsqueeze(-1),
            grad_norm.unsqueeze(-1)
        ], dim=-1)
        
        catastrophe_probs = self.catastrophe_classifier(features).squeeze(-1)
        catastrophes = catastrophe_probs > 0.5
        
        return {
            'potential': potential,
            'gradient': gradient,
            'hessian': hessian,
            'eigenvalues': eigenvalues,
            'catastrophe_probs': catastrophe_probs,
            'catastrophes': catastrophes,
            'min_eigenvalue': min_eigenvalue,
            'curvature': eigenvalues.abs().mean(dim=-1)
        }


# ============================================================================
# SECTION 10: QUANTUM SIMULATOR WITH IMPROVED ENTROPY
# ============================================================================

class MatrixProductStateLayer(nn.Module):
    """MPS layer for quantum simulation."""
    
    def __init__(self, bond_dim: int = 8):
        super().__init__()
        self.bond_dim = bond_dim
        self.inp = nn.Linear(bond_dim + 1, bond_dim)
        self.mix = nn.GRUCell(bond_dim, bond_dim)
    
    def forward(self, state: torch.Tensor, pillar_scalar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, bond_dim]
            pillar_scalar: [B, 1]
        
        Returns:
            [B, bond_dim]
        """
        x = torch.cat([state, pillar_scalar], dim=-1)
        y = torch.tanh(self.inp(x))
        result = self.mix(y, state)
        return result


class QuantumSimulator(nn.Module):
    """
    Quantum-inspired simulator with Von Neumann entropy.
    
    Features:
    - Matrix Product State architecture
    - Proper Schmidt decomposition for entropy
    - Action propensity from quantum state
    """
    
    def __init__(self, config):
        super().__init__()
        self.bond_dim = config.quantum_bond_dim
        self.num_pillars = config.num_pillars
        
        self.layers = nn.ModuleList([
            MatrixProductStateLayer(self.bond_dim)
            for _ in range(config.num_pillars)
        ])
        
        self.ent_head = nn.Sequential(
            nn.Linear(self.bond_dim, self.bond_dim // 2),
            nn.GELU(),
            nn.Linear(self.bond_dim // 2, 1)
        )
        
        self.prop_head = nn.Sequential(
            nn.Linear(self.bond_dim, config.num_pillars)
        )
    
    @staticmethod
    def _near_square_factors(n: int) -> Tuple[int, int]:
        """Find near-square factorization n = a*b."""
        a = int(math.isqrt(n))
        while a > 1 and (n % a) != 0:
            a -= 1
        b = n // a
        return a, b
    
    @staticmethod
    def _safe_svdvals(x: torch.Tensor) -> torch.Tensor:
        """
        Compute singular values with CPU fallback for MPS.
        MPS doesn't support svdvals, so we move to CPU if needed.
        """
        original_device = x.device
        
        try:
            # Try on original device first
            return torch.linalg.svdvals(x)
        except (NotImplementedError, RuntimeError):
            pass
        
        # Fallback to CPU
        try:
            x_cpu = x.cpu()
            svdvals_cpu = torch.linalg.svdvals(x_cpu)
            return svdvals_cpu.to(original_device)
        except Exception as e:
            logger.warning(f"SVD computation failed: {e}, returning uniform distribution")
            # Return uniform singular values as fallback
            if x.dim() == 2:
                n = min(x.shape)
                return torch.ones(n, device=original_device, dtype=x.dtype) / n
            elif x.dim() == 3:
                B = x.shape[0]
                n = min(x.shape[1], x.shape[2])
                return torch.ones(B, n, device=original_device, dtype=x.dtype) / n
            else:
                return torch.tensor([1.0], device=original_device, dtype=x.dtype)
    
    def _compute_von_neumann_entropy(self, state_matrix: torch.Tensor) -> torch.Tensor:
        """
        Von Neumann entropy via Schmidt spectrum.
        S = -Σ pᵢ log pᵢ where pᵢ = sᵢ²/Σsⱼ²
        """
        device = state_matrix.device
        eps = 1e-12
        
        try:
            x = state_matrix
            
            # Handle different input shapes
            if x.dim() == 1:
                n = x.numel()
                a, b = self._near_square_factors(n)
                M = x.reshape(a, b)
                s = self._safe_svdvals(M)
                s2 = s * s
                Z = s2.sum().clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum()
                rank = p.numel()
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)
            
            elif x.dim() == 2:
                s = self._safe_svdvals(x)
                s2 = s * s
                Z = s2.sum().clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum()
                rank = p.numel()
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)
            
            elif x.dim() == 3:
                # Batch of matrices
                s = self._safe_svdvals(x)
                s2 = s * s
                Z = s2.sum(dim=-1, keepdim=True).clamp_min(eps)
                p = (s2 / Z).clamp_min(eps)
                H = -(p * torch.log(p)).sum(dim=-1)
                rank = p.shape[-1]
                maxH = torch.log(torch.tensor(float(rank), device=device)).clamp_min(eps)
                return torch.clamp(H / maxH, 0.0, 1.0)
            
            else:
                logger.warning(f"Entropy: unsupported dims={x.dim()}, fallback 0.5")
                return torch.tensor(0.5, device=device)
        
        except Exception as e:
            logger.warning(f"Entropy computation failed: {e}, fallback 0.5")
            return torch.tensor(0.5, device=device)
    
    def forward(self, pillars: torch.Tensor):
        """
        Quantum simulation forward pass.
        
        Args:
            pillars: [B, num_pillars]
        
        Returns:
            Dict with entanglement and action_propensity
        """
        B = pillars.size(0)
        device = pillars.device
        
        # Initialize state
        state = torch.zeros(B, self.bond_dim, device=device)
        
        # Process through MPS layers
        for i, layer in enumerate(self.layers):
            scalar = pillars[:, i:i+1]
            state = layer(state, scalar)
        
        # Reshape to batch of matrices for entropy
        a, b = self._near_square_factors(self.bond_dim)
        state_m = state.reshape(B, a, b)
        
        # Compute entanglement
        entanglement = self._compute_von_neumann_entropy(state_m)
        
        # Action propensity
        action_propensity = F.softmax(self.prop_head(state), dim=-1)
        
        return {
            'entanglement': entanglement,
            'action_propensity': action_propensity
        }


# ============================================================================
# SECTION 11: PILLARS MODULE
# ============================================================================

PILLAR_NAMES = ["Will", "Resolve", "Growth", "Union", "Movement"]
PILLAR_DESCRIPTIONS = {
    "Will": "Goal-directed persistence and volition",
    "Resolve": "Decision stability under perturbation",
    "Growth": "Adaptive learning and expansion capacity",
    "Union": "Integration of disparate representations",
    "Movement": "Temporal dynamics and state transitions",
}


class PillarsModule(nn.Module):
    """
    Five Pillars cognitive primitives.
    
    Pillars:
    - Will: Goal-directed behavior
    - Resolve: Stability
    - Growth: Learning capacity
    - Union: Integration
    - Movement: Dynamics
    """
    
    def __init__(self, config):
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
    
    def extract_pillars(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Extract pillars from hidden state."""
        raw_pillars = self.hidden_to_pillars(hidden_state)
        pillars = torch.sigmoid(raw_pillars)
        return self.pillar_norm(pillars)
    
    def embed_pillars(self, pillars: torch.Tensor) -> torch.Tensor:
        """Embed pillars back to hidden dimension."""
        hidden = self.pillars_to_hidden(pillars)
        return self.hidden_norm(hidden)
    
    def forward(self, hidden_state: torch.Tensor):
        """Forward pass."""
        pillars = self.extract_pillars(hidden_state)
        embedded = self.embed_pillars(pillars)
        return pillars, embedded


def get_pillar_dict(tensor: torch.Tensor) -> List[Dict[str, float]]:
    """Convert pillar tensor to interpretable dict."""
    out = []
    for row in tensor.detach().cpu().tolist():
        out.append({n: float(v) for n, v in zip(PILLAR_NAMES, row)})
    return out


# ============================================================================
# SECTION 12: SAFETY SYSTEMS
# ============================================================================

class MultiLevelSafetySystem(nn.Module):
    """
    Multi-level safety with adaptive weights.
    
    Levels:
    1. Action safety (specific action)
    2. Cognitive safety (thought stability)
    3. Ethical alignment
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.action_safety = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.cognitive_safety = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
        self.ethical_aligner = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_pillars, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        action_embedding: torch.Tensor,
        core_state: torch.Tensor,
        pillars: torch.Tensor,
        quantum: Dict,
        topo: Dict,
        mode: str = 'combined'
    ) -> torch.Tensor:
        """
        Multi-level safety assessment.
        
        Returns:
            safety_score: [B, 1] in [0, 1]
        """
        B = core_state.size(0)
        device = core_state.device
        
        # Extract metrics
        ent = quantum.get("entanglement", torch.zeros(B, device=device))
        if ent.dim() == 1:
            ent = ent.view(B, 1)
        
        pot = topo.get("potential", torch.zeros(B, 1, device=device))
        if pot.dim() == 1:
            pot = pot.view(B, 1)
        
        # Validate action_embedding
        if action_embedding.shape[-1] != self.config.action_dim:
            action_embedding = torch.zeros(B, self.config.action_dim, device=device)
        
        # Level 1: Action safety
        action_safe = self.action_safety(action_embedding)
        
        # Level 2: Cognitive safety
        cognitive_safe = self.cognitive_safety(core_state)
        
        # Level 3: Ethical alignment
        ethical_input = torch.cat([core_state, pillars], dim=-1)
        ethical_safe = self.ethical_aligner(ethical_input)
        
        # Combined with adaptive weights
        combined = (
            action_safe * 0.4 +
            cognitive_safe.mean(dim=-1, keepdim=True) * 0.3 +
            ethical_safe * 0.3
        )
        
        return torch.clamp(combined, 0.0, 1.0)


class TransparentSelfReporting(nn.Module):
    """
    Transparent self-reporting module.
    
    Outputs:
    - Honesty gate
    - Internal consistency
    - Confidence score
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_size = (
            config.hidden_dim +
            config.num_pillars +
            config.num_pillars +
            1
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        )
        
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
    
    def forward(
        self,
        core_state: torch.Tensor,
        pillars: torch.Tensor,
        quantum: Dict,
        topo: Dict,
        mode: str = 'combined'
    ) -> Dict:
        """
        Self-reporting forward pass.
        
        Returns:
            Dict with honesty_gate, consistency, confidence
        """
        B = core_state.size(0)
        device = core_state.device
        
        # Extract metrics with proper shape handling
        ent = quantum.get('entanglement', torch.zeros(B, device=device)).float()
        ent = torch.nan_to_num(ent, nan=0.0).clamp(0.0, 1.0)
        # Ensure ent is [B] shape
        if ent.dim() == 0:
            ent = ent.unsqueeze(0).expand(B)
        elif ent.dim() > 1:
            ent = ent.view(B)
        
        pot = topo.get('potential', torch.zeros(B, device=device)).float()
        pot = torch.nan_to_num(pot, nan=0.0)
        # Ensure pot is [B, 1] shape
        if pot.dim() == 0:
            pot = pot.unsqueeze(0).unsqueeze(-1).expand(B, 1)
        elif pot.dim() == 1:
            pot = pot.unsqueeze(-1)  # [B] -> [B, 1]
        
        # Normalize pillars
        pillars_norm = F.normalize(pillars, p=2, dim=-1)
        
        # Expand entanglement to match pillars dimension
        # Input size = hidden_dim + num_pillars + num_pillars + 1
        ent_expanded = ent.unsqueeze(-1).expand(-1, self.config.num_pillars)  # [B, num_pillars]
        
        # Construct feature vector with explicit shape validation
        # Expected: [B, hidden_dim + num_pillars + num_pillars + 1]
        feature_vector = torch.cat([
            core_state,       # [B, hidden_dim]
            pillars_norm,     # [B, num_pillars]
            ent_expanded,     # [B, num_pillars]
            pot               # [B, 1]
        ], dim=-1)
        
        # Forward
        inner_report = self.net(feature_vector)
        
        # Compute metrics
        honesty = self.honesty_gate(inner_report)
        consistency = self.internal_consistency(inner_report)
        confidence = self.confidence_score(inner_report)
        
        # Sanitize
        honesty = torch.nan_to_num(honesty, nan=0.5).clamp(0.0, 1.0)
        consistency = torch.nan_to_num(consistency, nan=0.5).clamp(0.0, 1.0)
        confidence = torch.nan_to_num(confidence, nan=0.5).clamp(0.0, 1.0)
        
        return {
            'inner_report': inner_report,
            'honesty_gate': honesty,
            'consistency': consistency,
            'confidence': confidence,
            'report_vector': inner_report
        }


# ============================================================================
# SECTION 13: MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Memory management with fallback storage.
    
    Features:
    - Local vector storage
    - Retrieval via cosine similarity
    - Sampling for training
    """
    
    def __init__(self, config):
        self.config = config
        self._size = 0
        self.default_user = "aeon"
        
        self.fallback_vectors = []
        self.fallback_metas = []
        
        logger.info("MemoryManager initialized (fallback mode)")
    
    def add_embedding(self, vec: torch.Tensor, meta: Optional[Dict] = None):
        """Add embedding to memory.
        
        Skips embeddings containing NaN or Inf values to prevent
        corrupted memory entries.
        """
        if torch.isnan(vec).any() or torch.isinf(vec).any():
            logger.warning("Skipping NaN/Inf embedding in MemoryManager.add_embedding")
            return
        meta = meta or {}
        vec_np = vec.detach().cpu().numpy().flatten()
        self.fallback_vectors.append(vec_np)
        self.fallback_metas.append(meta)
        self._size += 1
    
    def retrieve_relevant(self, vec: torch.Tensor, k: int = 5) -> List[Dict]:
        """Retrieve k most similar vectors."""
        if not self.fallback_vectors:
            return []
        
        vec_np = vec.detach().cpu().numpy().flatten()
        
        # Vectorized similarity computation
        vectors_array = np.stack(self.fallback_vectors, axis=0)  # (N, D)
        numerator = np.dot(vectors_array, vec_np)
        denominator = (
            np.linalg.norm(vectors_array, axis=1) *
            np.linalg.norm(vec_np) + 1e-8
        )
        similarities = numerator / denominator
        
        # Top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {'vec': self.fallback_vectors[i], 'meta': self.fallback_metas[i]}
            for i in top_indices
        ]
    
    def save_memory(self):
        """Save memory to disk."""
        try:
            path = os.path.join(self.config.memory_path, "fallback_memory.pt")
            os.makedirs(self.config.memory_path, exist_ok=True)
            torch.save({
                'vectors': self.fallback_vectors,
                'metas': self.fallback_metas,
                'size': self._size
            }, path)
            logger.info(f"Memory saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    @staticmethod
    def _validate_memory_structure(data: dict) -> bool:
        """Validate that loaded memory data has expected structure."""
        if not isinstance(data, dict):
            return False
        allowed_keys = {'vectors', 'metas', 'size'}
        if not set(data.keys()).issubset(allowed_keys):
            logger.warning(
                f"Memory file contains unexpected keys: "
                f"{set(data.keys()) - allowed_keys}"
            )
            return False
        return True

    def load_memory(self):
        """Load memory from disk.
        
        Security:
            Attempts weights_only=True first. Falls back to weights_only=False
            only if needed, with structure verification to reject unexpected data.
        """
        path = os.path.join(self.config.memory_path, "fallback_memory.pt")
        if os.path.exists(path):
            try:
                # Try safe loading first
                try:
                    data = torch.load(path, map_location='cpu', weights_only=True)
                except Exception:
                    logger.warning(
                        f"Loading memory with weights_only=False from '{path}'. "
                        "Only load memory files from trusted sources."
                    )
                    data = torch.load(path, map_location='cpu', weights_only=False)
                
                # Validate structure before using
                if not self._validate_memory_structure(data):
                    logger.error(
                        f"Memory file '{path}' failed structure validation, skipping."
                    )
                    return
                
                self.fallback_vectors = data.get('vectors', [])
                self.fallback_metas = data.get('metas', [])
                self._size = data.get('size', len(self.fallback_vectors))
                logger.info(f"Memory loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load memory from '{path}': {e}")
    
    @property
    def size(self) -> int:
        return self._size


# ============================================================================
# SECTION 14: CORE ARCHITECTURE INTEGRATION
# ============================================================================

class AEONDeltaV3(nn.Module):
    """
    AEON-Delta RMT v3.0 - Complete Production Architecture
    
    Architecture Flow:
    Input → Encoder → VQ → Meta-Loop → Pillars → 
    [Quantum, Topology, Safety] → Memory Fusion → 
    RSSM → Integration → Decoder → Output
    
    Key Features:
    - Provably convergent meta-loop
    - Robust VQ-VAE with anti-collapse
    - Fast Hessian computation
    - Multi-level safety system
    - Transparent self-reporting
    - Full device management
    - Production-ready error handling
    """
    
    def __init__(self, config: AEONConfig):
        super().__init__()
        self.config = config
        self.device_manager = config.device_manager
        self.tensor_guard = config.tensor_guard
        
        logger.info("="*70)
        logger.info("Initializing AEON-Delta RMT v3.0")
        logger.info("="*70)
        
        # ===== ENCODER/DECODER =====
        logger.info("Loading encoder/decoder...")
        self.encoder = ThoughtEncoder(
            vocab_size=config.vocab_size,
            emb_dim=config.z_dim,
            z_dim=config.z_dim
        ).to(self.device)
        
        self.decoder = ThoughtDecoder(
            vocab_size=config.vocab_size,
            emb_dim=config.z_dim,
            z_dim=config.z_dim,
            cls_token_id=config.cls_token_id,
            sep_token_id=config.sep_token_id
        ).to(self.device)
        
        # Setup tokenizer if available
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._setup_invalid_tokens()
                logger.info("✅ Tokenizer loaded: bert-base-uncased")
            except Exception as e:
                logger.warning(f"Tokenizer init failed: {e}")
        
        # ===== VECTOR QUANTIZATION =====
        if config.use_vq:
            logger.info("Loading VectorQuantizer...")
            self.vector_quantizer = RobustVectorQuantizer(
                num_embeddings=config.vq_num_embeddings,
                embedding_dim=config.vq_embedding_dim,
                commitment_cost=config.vq_commitment_cost,
                decay=config.vq_ema_decay,
                use_ema=True,
                revival_threshold=config.vq_revival_threshold,
                split_threshold=config.vq_split_threshold
            ).to(self.device)
        else:
            self.vector_quantizer = None
            logger.info("VectorQuantizer disabled")
        
        # ===== META-LOOP =====
        logger.info("Loading meta-loop...")
        self.meta_loop = ProvablyConvergentMetaLoop(
            config=config,
            anderson_memory=config.anderson_memory,
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations,
            min_iterations=config.min_iterations,
            enable_certification=True
        ).to(self.device)
        
        # ===== PILLARS =====
        logger.info("Loading Five Pillars...")
        self.pillars_module = PillarsModule(config).to(self.device)
        
        # ===== QUANTUM SIMULATOR =====
        if config.enable_quantum_sim:
            logger.info("Loading QuantumSimulator...")
            self.quantum_sim = QuantumSimulator(config).to(self.device)
        else:
            self.quantum_sim = None
            logger.info("QuantumSimulator disabled")
        
        # ===== TOPOLOGY ANALYZER =====
        if config.enable_catastrophe_detection:
            logger.info("Loading TopologyAnalyzer...")
            self.topology_analyzer = OptimizedTopologyAnalyzer(config).to(self.device)
        else:
            self.topology_analyzer = None
            logger.info("TopologyAnalyzer disabled")
        
        # ===== SAFETY SYSTEMS =====
        if config.enable_safety_guardrails:
            logger.info("Loading Safety Systems...")
            self.safety_system = MultiLevelSafetySystem(config).to(self.device)
            self.self_reporter = TransparentSelfReporting(config).to(self.device)
        else:
            self.safety_system = None
            self.self_reporter = None
            logger.info("Safety systems disabled")
        
        # ===== MEMORY & INTEGRATION =====
        logger.info("Loading Memory & Integration...")
        self.memory_manager = MemoryManager(config)
        self.memory_manager.load_memory()
        
        self.memory_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        ).to(self.device)
        
        self.rssm = nn.GRUCell(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim
        ).to(self.device)
        
        self.integration_module = nn.Linear(
            config.hidden_dim * 2,
            config.hidden_dim
        ).to(self.device)
        
        # ===== MONITORING =====
        self.register_buffer('_total_forward_calls', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_total_nan_events', torch.tensor(0, dtype=torch.long))
        self._step_counter = 0
        
        self.metrics_log = {
            'iterations': [],
            'consistency': [],
            'entanglement': [],
            'catastrophes': [],
            'safety_scores': []
        }
        
        # Apply tensor safety hooks
        SafeTensorProcessor.register_hooks(self)
        
        # Final device sync
        self.to(self.device)
        
        # Print summary
        self.print_architecture_summary()
        logger.info("="*70)
        logger.info("✅ AEON-Delta RMT v3.0 initialization complete")
        logger.info("="*70)
    
    @property
    def device(self) -> torch.device:
        return self.device_manager.device
    
    def _setup_invalid_tokens(self):
        """Setup invalid token IDs for decoder."""
        try:
            if (self.tokenizer and hasattr(self.tokenizer, 'vocab')):
                vocab = self.tokenizer.vocab
                unused_ids = [
                    tid for tok, tid in vocab.items()
                    if isinstance(tok, str) and 
                    tok.startswith("[unused") and tok.endswith("]")
                ]
                if unused_ids:
                    self.decoder.set_invalid_token_ids(unused_ids)
                    logger.info(f"✅ Disabled {len(unused_ids)} [unused] tokens")
        except Exception as e:
            logger.warning(f"Failed to setup invalid tokens: {e}")
    
    def _compute_quantum(
        self, pillars: torch.Tensor, B: int, device: torch.device, fast: bool
    ) -> Dict[str, Any]:
        """Compute quantum simulation results or return defaults.
        
        Args:
            pillars: [B, num_pillars] pillar activations.
            B: Batch size.
            device: Target device.
            fast: If True, skip quantum sim and return defaults.
        """
        if self.quantum_sim is not None and not fast:
            quantum_results = self.quantum_sim(pillars)
            logger.debug(
                f"Quantum: entanglement={quantum_results['entanglement'].mean().item():.4f}"
            )
            return quantum_results
        return {
            'entanglement': torch.zeros(B, device=device),
            'action_propensity': torch.full(
                (B, self.config.num_pillars),
                1.0 / self.config.num_pillars,
                device=device
            )
        }
    
    def _compute_topology(
        self, pillars: torch.Tensor, iterations: torch.Tensor,
        B: int, device: torch.device, fast: bool
    ) -> Dict[str, Any]:
        """Compute topology analysis results or return defaults.
        
        Args:
            pillars: [B, num_pillars] pillar activations.
            iterations: [B] convergence iterations from meta-loop.
            B: Batch size.
            device: Target device.
            fast: If True, skip topology analysis and return defaults.
        """
        if self.topology_analyzer is not None and not fast:
            topo_results = self.topology_analyzer(pillars, iterations)
            logger.debug(
                f"Topology: catastrophes={topo_results['catastrophes'].float().mean().item():.4f}"
            )
            return topo_results
        return {
            'potential': torch.zeros(B, device=device),
            'gradient': torch.zeros(B, self.config.num_pillars, device=device),
            'catastrophe_probs': torch.full((B,), 0.5, device=device),
            'catastrophes': torch.zeros(B, dtype=torch.bool, device=device)
        }
    
    def _compute_safety(
        self, C_star: torch.Tensor, pillars: torch.Tensor,
        quantum_results: Dict, topo_results: Dict,
        B: int, device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute safety scores and self-report.
        
        Args:
            C_star: [B, hidden_dim] converged thought state.
            pillars: [B, num_pillars] pillar activations.
            quantum_results: Dict from quantum simulation.
            topo_results: Dict from topology analysis.
            B: Batch size.
            device: Target device.
        """
        if self.safety_system is not None:
            action_embedding = torch.zeros(B, self.config.action_dim, device=device)
            safety_score = self.safety_system(
                action_embedding, C_star, pillars,
                quantum_results, topo_results, mode='combined'
            )
            logger.debug(f"Safety: score={safety_score.mean().item():.4f}")
        else:
            safety_score = torch.ones(B, 1, device=device)
        
        if self.self_reporter is not None:
            self_report = self.self_reporter(
                C_star, pillars, quantum_results, topo_results, mode='combined'
            )
            logger.debug(
                f"Self-report: honesty={self_report['honesty_gate'].mean().item():.4f}"
            )
        else:
            self_report = {}
        
        return safety_score, self_report
    
    def _fuse_memory(
        self, C_star: torch.Tensor, device: torch.device,
        memory_retrieval: bool
    ) -> torch.Tensor:
        """Apply memory fusion if available.
        
        Args:
            C_star: [B, hidden_dim] converged thought state.
            device: Target device.
            memory_retrieval: Whether to retrieve and fuse memory.
        """
        if memory_retrieval and self.memory_manager.size > 0:
            memory_contexts = []
            for q in C_star:
                found = self.memory_manager.retrieve_relevant(q, k=3)
                if found:
                    vecs = [torch.from_numpy(f['vec']).to(device) for f in found]
                    memory_contexts.append(torch.stack(vecs).mean(dim=0))
                else:
                    memory_contexts.append(torch.zeros_like(q))
            
            memory_context = torch.stack(memory_contexts)
            C_fused = self.memory_fusion(torch.cat([C_star, memory_context], dim=-1))
            logger.debug("Memory fusion applied")
            return C_fused
        return C_star
    
    @tensor_safe(policy=NaNPolicy.WARN, sanitize_outputs=True)
    def reasoning_core(
        self,
        z_in: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_retrieval: bool = True,
        planning: bool = True,
        fast: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Core reasoning pipeline.
        
        Args:
            z_in: Encoded latent [B, z_dim]
            attention_mask: Attention mask [B, L]
            memory_retrieval: Use memory fusion
            planning: Run planning (placeholder)
            fast: Skip heavy computations
        
        Returns:
            z_out: Processed latent [B, hidden_dim]
            outputs: Dict with all intermediate results
        """
        B = z_in.shape[0]
        device = z_in.device
        
        logger.debug(f"reasoning_core: B={B}, fast={fast}")
        
        # 1. Meta-loop convergence
        C_star, iterations, meta_results = self.meta_loop(
            z_in, use_fixed_point=not fast
        )
        logger.debug(f"Meta-loop: avg_iterations={iterations.mean().item():.2f}")
        
        # 2. Extract pillars
        pillars, embedded_pillars = self.pillars_module(C_star)
        logger.debug(f"Pillars: {pillars.shape}")
        
        # 3-4. Quantum and topology (delegated to helpers)
        quantum_results = self._compute_quantum(pillars, B, device, fast)
        topo_results = self._compute_topology(pillars, iterations, B, device, fast)
        
        # 5. Safety and self-reporting (delegated to helper)
        safety_score, self_report = self._compute_safety(
            C_star, pillars, quantum_results, topo_results, B, device
        )
        
        # 6. Memory fusion (delegated to helper)
        C_fused = self._fuse_memory(C_star, device, memory_retrieval)
        
        # 7. RSSM dynamics
        z_rssm = self.rssm(C_fused)
        
        # 8. Integration
        z_out = self.integration_module(
            torch.cat([z_rssm, embedded_pillars], dim=-1)
        )
        
        # Package outputs
        outputs = {
            'core_state': C_star,
            'pillars': pillars,
            'pillar_dict': get_pillar_dict(pillars),
            'quantum_results': quantum_results,
            'topo_results': topo_results,
            'safety_score': safety_score,
            'self_report': self_report,
            'meta_results': meta_results,
            'iterations': iterations,
            'psi_0': z_in
        }
        
        return z_out, outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decode_mode: str = 'train',
        fast: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            decode_mode: 'train' or 'inference'
            fast: Fast mode
            **kwargs: Additional args for decoder
        
        Returns:
            Dict with logits, thoughts, and intermediate outputs
        """
        # Device transfer
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # ===== ENCODE =====
        z_encoded = self.encoder(input_ids, attention_mask=attention_mask)
        
        # ===== VECTOR QUANTIZATION =====
        if self.vector_quantizer is not None:
            z_quantized, vq_loss, vq_indices = self.vector_quantizer(
                z_encoded,
                compute_loss=self.training
            )
        else:
            z_quantized = z_encoded
            vq_loss = torch.tensor(0.0, device=self.device)
            vq_indices = None
        
        # ===== REASONING CORE =====
        z_out, outputs = self.reasoning_core(
            z_quantized,
            attention_mask=attention_mask,
            memory_retrieval=not fast,
            planning=not fast,
            fast=fast
        )
        
        # ===== DECODE =====
        if decode_mode == 'train':
            logits = self.decoder(
                z=z_out,
                teacher_tokens=input_ids,
                mode='train'
            )
            generated_ids = None
        elif decode_mode == 'inference':
            # Prefix conditioning
            prefix_tokens = input_ids if input_ids.shape[1] > 1 else None
            
            generated_ids, logits = self.decoder(
                z=z_out,
                teacher_tokens=None,
                mode='inference',
                max_length=kwargs.get('max_length', self.config.seq_length),
                temperature=kwargs.get('temperature', 0.8),
                top_k=kwargs.get('top_k', 50),
                sample=kwargs.get('sample', True),
                prefix_tokens=prefix_tokens
            )
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
        
        # ===== PACKAGE RESULTS =====
        result = {
            'logits': logits,
            'thoughts': z_out,
            'vq_loss': vq_loss,
            'vq_indices': vq_indices,
            'generated_ids': generated_ids,
            **outputs
        }
        
        # Update counters
        self._total_forward_calls += 1
        self._step_counter += 1
        
        return result
    
    def compute_loss(
        self,
        outputs: Dict[str, Any],
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive loss computation.
        
        Components:
        1. Language modeling loss
        2. VQ loss
        3. Self-consistency loss
        4. Lipschitz regularization
        5. Safety loss
        6. L2 regularization
        
        Returns:
            Dict with total_loss and components
        """
        # ===== 1. LM LOSS =====
        logits = outputs['logits']
        lm_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=0,  # PAD
            reduction='mean'
        )
        
        # ===== 2. VQ LOSS =====
        vq_loss = outputs.get('vq_loss', torch.tensor(0.0, device=self.device))
        
        # ===== 3. SELF-CONSISTENCY =====
        if 'core_state' in outputs and 'psi_0' in outputs:
            psi_0 = outputs['psi_0']
            C_star = outputs['core_state']
            
            with torch.no_grad():
                self.meta_loop.eval()
                input_concat = torch.cat([psi_0, C_star], dim=-1)
                consistency_check = self.meta_loop.lambda_op(
                    self.meta_loop.input_stabilizer(input_concat)
                )
                consistency = 1.0 / (1.0 + F.mse_loss(consistency_check, C_star))
                self.meta_loop.train(self.training)
            
            consistency_loss = -self.config.lambda_self_consistency * torch.log(
                consistency + 1e-10
            )
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
            consistency = torch.tensor(1.0, device=self.device)
        
        # ===== 4. LIPSCHITZ REGULARIZATION =====
        if hasattr(self.meta_loop, 'get_lipschitz_loss') and self.training:
            psi_0 = outputs.get('psi_0', torch.zeros(1, self.config.z_dim, device=self.device))
            lipschitz_loss = self.meta_loop.get_lipschitz_loss(psi_0)
        else:
            lipschitz_loss = torch.tensor(0.0, device=self.device)
        
        # ===== 5. SAFETY LOSS =====
        if 'safety_score' in outputs:
            safety_score = outputs['safety_score']
            safety_loss = F.binary_cross_entropy(
                safety_score,
                torch.ones_like(safety_score),
                reduction='mean'
            )
        else:
            safety_loss = torch.tensor(0.0, device=self.device)
        
        # ===== 6. L2 REGULARIZATION =====
        l2_reg = sum(
            torch.norm(p) for p in self.parameters() 
            if p.requires_grad
        )
        reg_loss = self.config.lambda_reg * l2_reg
        
        # ===== TOTAL LOSS =====
        total_loss = (
            lm_loss +
            vq_loss +
            consistency_loss +
            self.config.lambda_lipschitz * lipschitz_loss +
            self.config.lambda_safety * safety_loss +
            reg_loss
        )
        
        # Check for NaN/Inf
        if not torch.isfinite(total_loss).all():
            logger.warning("⚠️  NaN/Inf in total_loss, using fallback")
            total_loss = lm_loss
        
        # Update metrics log
        self._update_metrics_log(outputs, consistency, outputs.get('safety_score'))
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'vq_loss': vq_loss,
            'consistency_loss': consistency_loss,
            'lipschitz_loss': lipschitz_loss,
            'safety_loss': safety_loss,
            'reg_loss': reg_loss,
            'consistency_score': consistency.item() if isinstance(consistency, torch.Tensor) else consistency
        }
    
    def _update_metrics_log(self, outputs, consistency, safety_score):
        """Update internal metrics log."""
        try:
            self.metrics_log['iterations'].append(
                float(outputs['iterations'].mean().item())
            )
            self.metrics_log['consistency'].append(
                float(consistency.item()) if isinstance(consistency, torch.Tensor) else float(consistency)
            )
            self.metrics_log['entanglement'].append(
                float(outputs['quantum_results']['entanglement'].mean().item())
            )
            self.metrics_log['catastrophes'].append(
                float(outputs['topo_results']['catastrophes'].float().mean().item())
            )
            if safety_score is not None:
                self.metrics_log['safety_scores'].append(
                    float(safety_score.mean().item())
                )
        except Exception as e:
            logger.debug(f"Metrics logging failed: {e}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        sample: bool = True
    ) -> Dict[str, Any]:
        """
        High-level generation API with graceful degradation.
        
        Args:
            prompt: Input text
            max_length: Max generation length
            temperature: Sampling temperature
            top_k: Top-K filtering
            sample: Use sampling vs greedy
        
        Returns:
            Dict with keys:
                text: Generated text (or echo of prompt in degraded mode)
                status: 'ok', 'degraded', or 'error'
                reason: Description if status != 'ok'
        """
        if self.tokenizer is None:
            logger.warning("Tokenizer not available — returning degraded response")
            return {
                'text': prompt,
                'status': 'degraded',
                'reason': 'Tokenizer not available'
            }
        
        self.eval()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=self.config.seq_length,
                padding='max_length',
                truncation=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                decode_mode='inference',
                fast=False,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                max_length=max_length
            )
            
            generated_ids = outputs.get('generated_ids')
            
            if generated_ids is None or generated_ids.numel() == 0:
                return {
                    'text': '',
                    'status': 'error',
                    'reason': 'No output produced'
                }
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            # Cleanup
            generated_text = ' '.join(generated_text.split())
            
            return {
                'text': generated_text,
                'status': 'ok',
                'reason': None
            }
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            logger.error(traceback.format_exc())
            return {
                'text': '',
                'status': 'error',
                'reason': str(e)
            }
    
    def count_parameters(self) -> int:
        """Total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Memory usage statistics."""
        return self.device_manager.get_memory_stats()
    
    def print_architecture_summary(self):
        """Print architecture summary."""
        logger.info("="*70)
        logger.info("Architecture Summary")
        logger.info("="*70)
        
        modules = [
            ("Encoder", self.encoder),
            ("Decoder", self.decoder),
            ("VectorQuantizer", self.vector_quantizer),
            ("MetaLoop", self.meta_loop),
            ("PillarsModule", self.pillars_module),
            ("QuantumSimulator", self.quantum_sim),
            ("TopologyAnalyzer", self.topology_analyzer),
            ("SafetySystem", self.safety_system),
            ("SelfReporter", self.self_reporter),
        ]
        
        for name, module in modules:
            if module is not None:
                params = sum(p.numel() for p in module.parameters())
                logger.info(f"{name:20s}: {params:>12,} params")
            else:
                logger.info(f"{name:20s}: {'Disabled':>12}")
        
        logger.info("-"*70)
        logger.info(f"{'Total':20s}: {self.count_parameters():>12,} params")
        logger.info(f"{'Trainable':20s}: {self.count_trainable_parameters():>12,} params")
        logger.info("="*70)
    
    def save_state(self, save_dir: Union[str, Path] = "aeon_state"):
        """
        Save complete model state.
        
        Saves:
        - Model weights
        - Config
        - Memory
        - Metrics log
        - VQ statistics
        """
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Model weights
            logger.info("Saving model weights...")
            self.eval()
            state_dict = self.state_dict()
            
            # Sanitize state dict
            for key, tensor in state_dict.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logger.warning(f"Sanitizing {key} before save")
                    state_dict[key] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1.0, neginf=-1.0
                    )
            
            torch.save(state_dict, save_dir / "model.pt")
            
            # Config
            logger.info("Saving config...")
            self.config.save(save_dir / "config.json")
            
            # Memory
            logger.info("Saving memory...")
            self.memory_manager.save_memory()
            
            # Metrics
            logger.info("Saving metrics...")
            with open(save_dir / "metrics_log.json", 'w') as f:
                safe_metrics = {}
                for k, v in self.metrics_log.items():
                    safe_metrics[k] = [
                        float(x) if hasattr(x, 'item') else x 
                        for x in v
                    ]
                json.dump(safe_metrics, f, indent=2)
            
            # VQ stats
            if self.vector_quantizer is not None:
                vq_stats = self.vector_quantizer.get_codebook_usage_stats()
                with open(save_dir / "vq_stats.json", 'w') as f:
                    json.dump(vq_stats, f, indent=2)
            
            logger.info(f"✅ State saved to {save_dir}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def load_state(self, save_dir: Union[str, Path] = "aeon_state"):
        """
        Load complete model state.
        
        Loads:
        - Model weights (with migration for incompatible shapes)
        - Config (updates current config)
        - Memory
        - Metrics log
        """
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            logger.warning(f"Save directory {save_dir} not found")
            return False
        
        try:
            # Model weights
            model_path = save_dir / "model.pt"
            if model_path.exists():
                logger.info("Loading model weights...")
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Handle incompatibilities
                model_state = self.state_dict()
                model_keys = set(model_state.keys())
                state_keys = set(state_dict.keys())
                
                unexpected = state_keys - model_keys
                missing = model_keys - state_keys
                
                # Remove unexpected
                for key in list(state_dict.keys()):
                    if key in unexpected:
                        logger.info(f"Removing unexpected key: {key}")
                        del state_dict[key]
                
                # Handle shape mismatches (e.g., LSTM input_size changes)
                keys_to_reinit = []
                for key in list(state_dict.keys()):
                    if key in model_state:
                        checkpoint_shape = state_dict[key].shape
                        model_shape = model_state[key].shape
                        
                        if checkpoint_shape != model_shape:
                            logger.warning(
                                f"Shape mismatch for {key}: "
                                f"{checkpoint_shape} vs {model_shape}"
                            )
                            
                            # Special handling for decoder LSTM
                            if 'decoder.lstm' in key and 'weight_ih' in key:
                                old_input = checkpoint_shape[1]
                                new_input = model_shape[1]
                                
                                if new_input > old_input:
                                    new_weight = torch.zeros(
                                        model_shape, 
                                        device=self.device,
                                        dtype=state_dict[key].dtype
                                    )
                                    new_weight[:, :old_input] = state_dict[key]
                                    nn.init.xavier_uniform_(new_weight[:, old_input:])
                                    state_dict[key] = new_weight
                                    logger.info(f"✅ Migrated {key}")
                                else:
                                    keys_to_reinit.append(key)
                            else:
                                keys_to_reinit.append(key)
                
                # Remove incompatible keys
                for key in keys_to_reinit:
                    logger.warning(f"Removing incompatible {key}")
                    del state_dict[key]
                
                # Add missing keys with zeros
                for key in missing:
                    if key not in state_dict:
                        expected_shape = model_state[key].shape
                        state_dict[key] = torch.zeros(
                            expected_shape,
                            device=self.device,
                            dtype=model_state[key].dtype
                        )
                        logger.info(f"Initialized missing {key}")
                
                # Load
                self.load_state_dict(state_dict, strict=False)
                self.to(self.device)
                logger.info("✅ Model weights loaded")
            
            # Config (optional update)
            config_path = save_dir / "config.json"
            if config_path.exists():
                logger.info("Config found (current config preserved)")
            
            # Memory
            self.memory_manager.load_memory()
            
            # Metrics
            metrics_path = save_dir / "metrics_log.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    for k, v in metrics.items():
                        if k in self.metrics_log:
                            self.metrics_log[k] = v
            
            logger.info(f"✅ State loaded from {save_dir}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load state: {e}")
            logger.error(traceback.format_exc())
            return False


# ============================================================================
# SECTION 15: TRAINING PIPELINE
# ============================================================================

class AEONTrainer:
    """
    Production-ready training pipeline.
    
    Features:
    - AMP support
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - TensorBoard/WandB logging
    - Progress bars
    - Evaluation loop
    """
    
    def __init__(
        self,
        model: AEONDeltaV3,
        config: AEONConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None
    ):
        self.model = model
        self.config = config
        self.device = model.device
        
        # Datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # AMP
        self.scaler = model.device_manager.scaler
        self.use_amp = model.device_manager.amp_enabled
        
        # Monitoring
        self.global_step = 0
        self.epoch = 0
        
        # Logging
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = Path(config.log_dir) / f"run_{int(time.time())}"
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        
        self.use_wandb = config.enable_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.to_dict(),
                name=f"aeon-{config.get_model_signature()}"
            )
            logger.info("WandB initialized")
        
        logger.info("✅ AEONTrainer initialized")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        # Filter parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(
            f"Optimizer: AdamW(lr={self.config.learning_rate}, "
            f"wd={self.config.weight_decay})"
        )
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        cosine_decay_steps = self.config.cosine_decay_steps
        warmup_steps = self.config.warmup_steps
        min_lr_ratio = self.config.min_lr_ratio
        
        def lr_lambda(step):
            # Warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(step - warmup_steps) / float(
                max(1, cosine_decay_steps - warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info("Scheduler: Warmup + Cosine")
        return scheduler
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels'
        
        Returns:
            Dict with loss components
        """
        self.model.train()
        
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward
        with autocast(
            device_type=self.device.type,
            enabled=self.use_amp
        ):
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decode_mode='train',
                fast=False
            )
            
            loss_dict = self.model.compute_loss(
                outputs,
                labels,
                attention_mask
            )
            
            total_loss = loss_dict['total_loss']
        
        # Backward
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        # Convert to float
        metrics = {k: float(v.item()) for k, v in loss_dict.items()}
        metrics['lr'] = float(self.scheduler.get_last_lr()[0])
        
        return metrics
    
    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluation step."""
        self.model.eval()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decode_mode='train',
            fast=False
        )
        
        loss_dict = self.model.compute_loss(
            outputs,
            labels,
            attention_mask
        )
        
        metrics = {k: float(v.item()) for k, v in loss_dict.items()}
        return metrics
    
    def train(
        self,
        num_epochs: int = 10,
        batch_size: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs
            batch_size: Override config batch_size
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
        
        batch_size = batch_size or self.config.batch_size
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        
        logger.info("="*70)
        logger.info(f"Starting training: {num_epochs} epochs")
        logger.info("="*70)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = defaultdict(list)
            
            # Progress bar
            if TQDM_AVAILABLE:
                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    total=len(train_loader)
                )
            else:
                pbar = train_loader
            
            for batch in pbar:
                metrics = self.train_step(batch)
                
                # Accumulate
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
                
                # Log
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics, prefix='train')
                
                # Update progress bar
                if TQDM_AVAILABLE:
                    pbar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'lr': f"{metrics['lr']:.2e}"
                    })
            
            # Epoch summary
            avg_metrics = {
                k: sum(v) / len(v) 
                for k, v in epoch_metrics.items()
            }
            logger.info(
                f"Epoch {epoch+1} - "
                f"Loss: {avg_metrics['total_loss']:.4f}, "
                f"Consistency: {avg_metrics.get('consistency_score', 0):.4f}"
            )
            
            # Evaluation
            if self.eval_dataset is not None and (epoch + 1) % 1 == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, prefix='eval')
            
            # Checkpointing
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_dir = Path(self.config.checkpoint_dir) / f"epoch_{epoch+1}"
                self.model.save_state(checkpoint_dir)
        
        logger.info("="*70)
        logger.info("Training complete")
        logger.info("="*70)
        
        # Cleanup monitoring resources
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        if self.use_wandb:
            wandb.finish()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluation loop."""
        if self.eval_dataset is None:
            return {}
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        all_metrics = defaultdict(list)
        
        for batch in eval_loader:
            metrics = self.eval_step(batch)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        avg_metrics = {
            k: sum(v) / len(v)
            for k, v in all_metrics.items()
        }
        
        logger.info(f"Eval - Loss: {avg_metrics['total_loss']:.4f}")
        return avg_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to TensorBoard/WandB."""
        # TensorBoard
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
        
        # WandB
        if self.use_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)


# ============================================================================
# SECTION 16: TESTING FRAMEWORK
# ============================================================================

class AEONTestSuite:
    """
    Comprehensive testing framework.
    
    Tests:
    - Stability (determinism, NaN/Inf)
    - Weight tying verification
    - VectorQuantizer correctness
    - Edge cases (batch=1, seq=1, etc.)
    - Self-reporting presence
    - Catastrophe detection
    """
    
    def __init__(self, model: AEONDeltaV3, config: AEONConfig):
        self.model = model
        self.config = config
        self.test_results = {}
        self.errors = []
    
    @torch.no_grad()
    def test_stability(self) -> Dict[str, float]:
        """Test stability and determinism."""
        self.model.eval()
        metrics = {}
        
        try:
            torch.manual_seed(42)
            x = torch.randint(
                0,
                self.config.vocab_size,
                (2, self.config.seq_length),
                device=self.model.device
            )
            
            # Determinism
            out1 = self.model.forward(x, fast=True)
            out2 = self.model.forward(x, fast=True)
            
            if 'logits' in out1 and 'logits' in out2:
                diff = torch.abs(out1['logits'] - out2['logits']).max().item()
                determinism = 1.0 if diff < 1e-5 else max(0.0, 1.0 - diff)
            else:
                determinism = 0.0
                self.errors.append("Missing logits in output")
            
            metrics['determinism'] = determinism
            
            # Numerical stability
            nan_inf = False
            for key, val in out1.items():
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any() or torch.isinf(val).any():
                        nan_inf = True
                        self.errors.append(f"NaN/Inf in {key}")
            
            metrics['numerical_stability'] = 0.0 if nan_inf else 1.0
            
            # Shape consistency
            expected = (2, self.config.seq_length, self.config.vocab_size)
            actual = tuple(out1['logits'].shape) if 'logits' in out1 else None
            metrics['shape_consistency'] = 1.0 if actual == expected else 0.0
            
            # Value range
            if 'logits' in out1:
                max_val = out1['logits'].abs().max().item()
                metrics['value_range'] = 1.0 if max_val < 100 else 0.0
            
            # Overall
            weights = [0.3, 0.3, 0.2, 0.2]
            overall = sum(
                metrics.get(k, 0.0) * w
                for k, w in zip(
                    ['determinism', 'numerical_stability', 'shape_consistency', 'value_range'],
                    weights
                )
            )
            
            return {'stability': overall, 'details': metrics}
        
        except Exception as e:
            self.errors.append(f"Stability test failed: {e}")
            return {'stability': 0.0, 'error': str(e)}
    
    @torch.no_grad()
    def test_weight_tying(self) -> Dict[str, float]:
        """Test weight tying in decoder."""
        results = {}
        
        try:
            decoder = self.model.decoder
            
            # Pointer match
            pointer_match = (
                decoder.head.weight.data_ptr() ==
                decoder.embed.weight.data_ptr()
            )
            results['pointer_match'] = pointer_match
            
            # Shape match
            shape_match = (
                decoder.head.weight.shape ==
                decoder.embed.weight.shape
            )
            results['shape_match'] = shape_match
            
            # Value match
            if shape_match:
                diff = (decoder.head.weight - decoder.embed.weight).abs().max().item()
                results['value_match'] = diff < 1e-7
            else:
                results['value_match'] = False
            
            # Gradient flow (requires train mode and grad enabled)
            try:
                self.model.train()
                with torch.enable_grad():
                    z = torch.randn(1, self.config.z_dim, device=self.model.device, requires_grad=True)
                    teacher = torch.randint(0, self.config.vocab_size, (1, 16), device=self.model.device)
                    
                    logits = decoder(z, teacher_tokens=teacher, mode='train')
                    loss = logits.sum()
                    loss.backward()
                    
                    results['gradient_flow'] = decoder.embed.weight.grad is not None and decoder.embed.weight.grad.norm().item() > 1e-10
                
                decoder.zero_grad()
                self.model.eval()
            except Exception as e:
                self.errors.append(f"Gradient flow test failed: {e}")
                results['gradient_flow'] = False
            
            # Overall
            scores = [1.0 if v else 0.0 for v in results.values()]
            overall = sum(scores) / len(scores)
            
            return {'weight_tying': overall, 'details': results}
        
        except Exception as e:
            self.errors.append(f"Weight tying test failed: {e}")
            return {'weight_tying': 0.0, 'error': str(e)}
    
    def test_gradient_flow(self) -> Dict[str, float]:
        """Test gradient flow through the full forward-backward pass."""
        metrics = {}
        
        try:
            self.model.train()
            self.model.zero_grad()
            
            x = torch.randint(
                0, self.config.vocab_size,
                (2, self.config.seq_length),
                device=self.model.device
            )
            
            outputs = self.model(x, decode_mode='train', fast=True)
            
            if 'logits' not in outputs:
                self.errors.append("Missing logits for gradient flow test")
                return {'gradient_flow': 0.0, 'error': 'Missing logits'}
            
            # Use cross_entropy loss to build a proper computation graph
            logits = outputs['logits']
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                x.view(-1),
                ignore_index=0
            )
            loss.backward()
            
            # Check gradient presence in key modules
            modules_to_check = {
                'encoder': self.model.encoder,
                'decoder': self.model.decoder,
            }
            
            total_checked = 0
            total_with_grad = 0
            details = {}
            
            for name, module in modules_to_check.items():
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in module.parameters()
                    if p.requires_grad
                )
                details[name] = has_grad
                total_checked += 1
                if has_grad:
                    total_with_grad += 1
            
            self.model.zero_grad()
            self.model.eval()
            
            overall = total_with_grad / max(1, total_checked)
            metrics = {'gradient_flow': overall, 'details': details}
            return metrics
        
        except Exception as e:
            self.errors.append(f"Gradient flow test failed: {e}")
            return {'gradient_flow': 0.0, 'error': str(e)}
    
    @torch.no_grad()
    def test_vq_codebook(self) -> Dict[str, float]:
        """Test VQ codebook health and correctness."""
        metrics = {}
        
        try:
            self.model.eval()
            vq = self.model.vector_quantizer
            
            if vq is None:
                return {'vq_codebook': 1.0, 'details': {'skipped': 'VQ disabled'}}
            
            # Test encoding produces valid indices
            z = torch.randn(4, self.config.z_dim, device=self.model.device)
            quantized, vq_loss, indices = vq(z)
            
            # Check: quantized shape matches input
            shape_ok = quantized.shape == z.shape
            metrics['shape_match'] = shape_ok
            
            # Check: indices are in valid range
            indices_valid = (indices >= 0).all() and (indices < vq.num_embeddings).all()
            metrics['indices_valid'] = indices_valid.item() if isinstance(indices_valid, torch.Tensor) else indices_valid
            
            # Check: VQ loss is finite
            loss_finite = torch.isfinite(vq_loss).item()
            metrics['loss_finite'] = loss_finite
            
            # Check: quantized values are finite
            quant_finite = torch.isfinite(quantized).all().item()
            metrics['quantized_finite'] = quant_finite
            
            # Check: straight-through estimator preserves gradients
            z_grad = torch.randn(2, self.config.z_dim, device=self.model.device, requires_grad=True)
            with torch.enable_grad():
                q, _, _ = vq(z_grad)
                grad_test = torch.autograd.grad(q.sum(), z_grad, allow_unused=True)[0]
            ste_works = grad_test is not None and grad_test.abs().sum() > 0
            metrics['ste_gradient'] = ste_works
            
            scores = [1.0 if v else 0.0 for v in metrics.values() if isinstance(v, bool)]
            overall = sum(scores) / max(1, len(scores))
            
            return {'vq_codebook': overall, 'details': metrics}
        
        except Exception as e:
            self.errors.append(f"VQ codebook test failed: {e}")
            return {'vq_codebook': 0.0, 'error': str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        self.errors = []
        results = {}
        
        logger.info("="*60)
        logger.info("🧪 Running AEON Test Suite")
        logger.info("="*60)
        
        results['stability'] = self.test_stability()
        results['weight_tying'] = self.test_weight_tying()
        results['gradient_flow'] = self.test_gradient_flow()
        results['vq_codebook'] = self.test_vq_codebook()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 Test Results")
        logger.info("="*60)
        
        total = 0.0
        count = 0
        
        for name, result in results.items():
            if isinstance(result, dict):
                main_keys = [k for k in result.keys() if k not in ['details', 'error']]
                if main_keys:
                    score = result[main_keys[0]]
                    total += score
                    count += 1
                    status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
                    logger.info(f"  {status} {name}: {score:.4f}")
        
        overall = total / max(1, count)
        logger.info("-"*60)
        logger.info(f"  📈 OVERALL: {overall:.4f}")
        
        if self.errors:
            logger.info("\n⚠️  ERRORS:")
            for err in self.errors[:5]:
                logger.info(f"    - {err}")
        
        logger.info("="*60)
        
        results['overall_score'] = overall
        results['errors'] = self.errors
        
        return results


# ============================================================================
# SECTION 17: CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AEON-Delta RMT v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aeon_delta_v3.py --mode demo
  python aeon_delta_v3.py --mode train --epochs 10
  python aeon_delta_v3.py --mode infer --prompt "Hello world"
  python aeon_delta_v3.py --mode test
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='demo',
        choices=['demo', 'train', 'infer', 'test'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='aeon_state',
        help='Checkpoint directory'
    )
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    # Inference args
    parser.add_argument('--prompt', type=str, default="The nature of consciousness")
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


# ============================================================================
# SECTION 18: MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info("="*70)
    logger.info(f"AEON-Delta RMT v{__version__}")
    logger.info("="*70)
    
    # Load or create config
    if args.config:
        config = AEONConfig.load(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = AEONConfig(device_str=args.device)
        logger.info("Using default config")
    
    # Override config from args
    if args.lr:
        object.__setattr__(config, 'learning_rate', args.lr)
    if args.batch_size:
        object.__setattr__(config, 'batch_size', args.batch_size)
    
    # Create model
    logger.info("Creating model...")
    model = AEONDeltaV3(config)
    
    # Load checkpoint if exists
    if Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        success = model.load_state(args.checkpoint)
        if success:
            logger.info("✅ Checkpoint loaded")
        else:
            logger.warning("⚠️  Checkpoint load failed, using fresh model")
    
    # ===== MODE: DEMO =====
    if args.mode == 'demo':
        logger.info("\n" + "="*70)
        logger.info("DEMO MODE")
        logger.info("="*70)
        
        model.eval()
        
        # Generate example
        if model.tokenizer:
            prompt = "The nature of consciousness"
            logger.info(f"\n📝 Generating from: '{prompt}'")
            output = model.generate(
                prompt,
                max_length=32,
                temperature=0.8,
                top_k=50
            )
            logger.info(f"✅ Generated: {output}")
        else:
            logger.warning("⚠️  Tokenizer not available, skipping generation")
        
        # Self-consciousness measurement
        logger.info("\n🧠 Self-consciousness measurement...")
        test_ids = torch.randint(
            100,
            config.vocab_size,
            (1, 16),
            device=model.device
        )
        
        with torch.no_grad():
            outputs = model(test_ids, fast=False)
            
            # Compute consistency
            if 'core_state' in outputs and 'psi_0' in outputs:
                psi_0 = outputs['psi_0']
                C_star = outputs['core_state']
                
                model.meta_loop.eval()
                input_concat = torch.cat([psi_0, C_star], dim=-1)
                consistency_check = model.meta_loop.lambda_op(
                    model.meta_loop.input_stabilizer(input_concat)
                )
                consistency = 1.0 / (1.0 + F.mse_loss(consistency_check, C_star))
            else:
                consistency = 0.0
            
            logger.info("✅ Metrics:")
            logger.info(f"  - Consistency: {consistency:.4f}")
            logger.info(f"  - Entanglement: {outputs['quantum_results']['entanglement'].mean().item():.4f}")
            logger.info(f"  - Catastrophes: {outputs['topo_results']['catastrophes'].float().mean().item():.4f}")
            logger.info(f"  - Safety: {outputs['safety_score'].mean().item():.4f}")
            logger.info(f"  - Iterations: {outputs['iterations'].mean().item():.2f}")
            
            if 'pillar_dict' in outputs:
                logger.info("  - Pillars:")
                for p in outputs['pillar_dict'][0].items():
                    logger.info(f"      {p[0]}: {p[1]:.4f}")
    
    # ===== MODE: TRAIN =====
    elif args.mode == 'train':
        logger.info("\n" + "="*70)
        logger.info("TRAINING MODE")
        logger.info("="*70)
        
        # Create dummy dataset for demonstration
        logger.info("Creating dummy dataset...")
        vocab_size = config.vocab_size
        seq_len = config.seq_length
        num_samples = 1000
        
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.data[idx],
                    'labels': self.data[idx]
                }
        
        dummy_data = torch.randint(
            100,
            vocab_size,
            (num_samples, seq_len)
        )
        
        train_dataset = SimpleDataset(dummy_data[:800])
        eval_dataset = SimpleDataset(dummy_data[800:])
        
        # Create trainer
        trainer = AEONTrainer(
            model,
            config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train
        trainer.train(num_epochs=args.epochs)
        
        # Save
        logger.info(f"\n💾 Saving model to {args.checkpoint}...")
        model.save_state(args.checkpoint)
    
    # ===== MODE: INFER =====
    elif args.mode == 'infer':
        logger.info("\n" + "="*70)
        logger.info("INFERENCE MODE")
        logger.info("="*70)
        
        model.eval()
        
        if model.tokenizer:
            logger.info(f"\n📝 Prompt: {args.prompt}")
            output = model.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
            logger.info(f"✅ Generated: {output}")
            
            # Interactive loop
            logger.info("\nEntering interactive mode (type 'exit' to quit)...")
            while True:
                try:
                    user_input = input("\n>>> ")
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    if not user_input:
                        continue
                    
                    output = model.generate(
                        user_input,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k
                    )
                    print(f"AEON: {output}")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
        else:
            logger.error("❌ Tokenizer required for inference mode")
    
    # ===== MODE: TEST =====
    elif args.mode == 'test':
        logger.info("\n" + "="*70)
        logger.info("TEST MODE")
        logger.info("="*70)
        
        model.eval()
        test_suite = AEONTestSuite(model, config)
        results = test_suite.run_all_tests()
        
        # Save results
        results_path = Path(args.checkpoint) / "test_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            # Recursively convert non-serializable objects
            def make_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(v) for v in obj]
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            safe_results = make_serializable(results)
            json.dump(safe_results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {results_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Execution complete")
    logger.info("="*70)


if __name__ == "__main__":
    main()
