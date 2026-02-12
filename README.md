[![License: AEON-Δ Research-Only](https://img.shields.io/badge/license-Research--Only-blue.svg)](./LICENSE)

# **AEON-Δ RMT v3.1: A Cognitive Architecture for Emergent Reasoning**  
**License**: AEON-Δ Research-Only

---

## 🌟 Executive Summary

Welcome to the official repository for **AEON-Δ RMT v3.1** (*Reflective Meta-Thinking*), a rigorously engineered cognitive architecture implemented in PyTorch. AEON-Δ simulates high-level reasoning through abstract thought vectors in latent space, with mathematical convergence guarantees and multi-level safety systems. This is not merely a transformer wrapper—it's a full cognitive simulator engineered for robustness, interpretability, and emergent reasoning capabilities.

The system implements a **provably convergent architecture** with certified error bounds, operating across multiple critical domains:

- **Sparse factorization** (learnable interpretable factors replacing hardcoded Five Pillars)  
- **Diversity metrics** (variance-based thought diversity measurement)  
- **Topological stability analysis** (catastrophe detection)  
- **Physics-grounded world modeling** (multi-backend dynamics and counterfactual reasoning)  
- **Causal reasoning** (structural causal models with do-calculus and counterfactual rollouts)  
- **Hierarchical memory** (working, episodic, semantic, temporal, and neurogenic memory levels)  
- **Multi-modal grounding** (vision, audio, language fusion with CLIP-style contrastive learning)  
- **Meta-learning** (MAML + EWC for few-shot adaptation and continual learning)  
- **Planning** (MCTS planner with curiosity-driven exploration and active learning)

---

## 🧠 Core Architecture: AEON-Delta RMT v3.1

### **0. Advanced Sequence Processing — SSM, Mamba-2 & Linear Attention**
AEON-Δ v3.1 includes state-of-the-art sequence processing backends that **surpass Transformer** in key dimensions:

| Dimension | Transformer | AEON-Δ (SSM/Mamba-1) | AEON-Δ (Mamba-2 SSD) | AEON-Δ (Linear Attn) |
|---|---|---|---|---|
| **Inference Speed** | O(n²) per step | **O(1) per token** (cached state) | **O(1) per token** (cached state) | **O(1) per token** (cached state) |
| **Training Complexity** | O(n²) | **O(n)** | **O(n)** with chunk-wise SSD | **O(n)** |
| **Sequence Length** | Limited by memory (n²) | **Arbitrary** (linear memory) | **Arbitrary** (linear memory) | **Arbitrary** (linear memory) |
| **Scalability** | Quadratic memory | **Linear memory** | **Linear memory** | **Linear memory** |
| **Multi-head** | Yes | No (single head) | **Yes (multi-head SSM)** | Yes |
| **Hardware Utilisation** | Good | Good (parallel scan) | **Excellent (chunked SSD)** | Good |

**Available backends** (configured via `AEONConfig.encoder_backend` / `decoder_backend`):
- **`ssm`** (default): Selective State Space Model inspired by Mamba (Gu & Dao, 2023) — input-dependent state transitions with parallel scan, O(n) training, O(1) cached inference
- **`mamba2`**: **NEW** — Mamba-2 Structured State Space Duality (Dao & Gu, 2024) — multi-head SSM with per-head scalar decay, chunk-wise SSD for superior hardware utilisation, RMSNorm for training stability
- **`linear_attention`**: ELU-based kernel linear attention — O(n) via associativity of matrix multiplication, multi-head support
- **`lstm`**: Original LSTM backend for backward compatibility

**Additional scalability features:**
- **`ChunkedSequenceProcessor`**: Process sequences of arbitrary length in overlapping chunks with state propagation — O(chunk_size) memory regardless of total length
- **`InferenceCache`**: Persistent state caching for O(1) per-step autoregressive generation
- **`PretrainedBackboneAdapter`**: Bottleneck adapter for integrating any HuggingFace pretrained model as a frozen backbone with minimal trainable parameters

---

### **1. Tensor Safety System & Device Management**  
Unlike conventional systems, AEON-Δ implements production-grade tensor safety with:
- **NaN/Inf handling policies**: `RAISE`, `WARN`, `SILENT`, `QUARANTINE`  
- **Automatic sanitization** with context tracking and reporting  
- **Thread-safe device management** with memory fraction control  
- **AMP (Automatic Mixed Precision)** with fallback mechanisms  
- **MPS (Apple Silicon) support** with CPU fallbacks for unstable ops  

---

### **2. Cognitive Core: Sparse Factorization & Causal Factors**  
The `SparseFactorization` module replaces the hardcoded Five Pillars with learnable, interpretable factors:
- **Learnable sparse basis** (64 factors by default via `num_pillars` config)  
- **L1-sparsity regularization** encouraging compact, interpretable representations  
- **Encode/decode symmetry** with LayerNorm for stable training  
- **No imposed philosophical categories** — factors emerge from data  

The `CausalFactorExtractor` extends this with an explicit learnable DAG structure for causal reasoning over factors.  

Each factor is continuously monitored, normalized, and fed into downstream reasoning systems.

---

### **3. Meta-Loop: Provably Convergent Reasoning**  
The `ProvablyConvergentMetaLoop` implements mathematical guarantees for cognitive stability:
- **Lipschitz-constrained Lambda operator** with spectral normalization  
- **Banach Fixed-Point Theorem guarantees** when *L < 1*  
- **Anderson acceleration** for 2–5× convergence speedup  
- **Adaptive alpha** based on Lipschitz estimates  
- **Certified error bounds** with automatic reporting  
- **Early stopping** with convergence certification  

This transforms initial perception ψ₀ into a stable thought state **C\*** through iteratively refined deliberation.

---

### **4. Diversity Metric**  
The `DiversityMetric` module replaces the former quantum-inspired simulator with a principled approach:
- **Variance-based diversity** across factor activations  
- **Action propensity** from softmax over learned projection  
- **Lightweight computation** without pseudoscientific quantum mechanics  

This provides a measure of internal thought diversity and decision certainty.

---

### **5. Topology Analyzer: Catastrophe Detection**  
Using catastrophe theory to detect representational instabilities:
- **Fast Hessian computation** with three methods:  
  - Finite differences *(production default)*  
  - Forward-mode AD *(experimental)*  
  - Hutchinson’s trace estimator  
- **Eigenvalue analysis** with CPU fallbacks for MPS  
- **Catastrophe classifier** predicting system stability  
- **Potential landscape analysis** for state transitions  

---

### **6. Multi-Level Safety System**  
Three-tiered safety architecture:
1. **Action safety** (specific action validation)  
2. **Cognitive safety** (thought stability metrics)  
3. **Ethical alignment** (value-consistent decision making)  

Each level contributes to a combined safety score with adaptive weighting.

---

### **7. Transparent Self-Reporting**  
The system provides introspective capabilities through:
- **Honesty gate** (truthfulness assessment)  
- **Internal consistency** (coherence measurement)  
- **Confidence score** (certainty estimation)  
- **Report vector** for external monitoring  

This enables external systems to verify AEON’s internal state and reasoning quality.

---

### **8. Vector Quantizer: Anti-Collapse Architecture**
Advanced VQ-VAE with stability mechanisms:
- **EMA updates** for stable codebook evolution  
- **Code revival** (reinitializing dead codes)  
- **Code splitting** (balancing overused codes)  
- **Perplexity monitoring** with EMA tracking  
- **Straight-Through Estimator** for gradient flow  

This creates a discrete latent space resistant to mode collapse.

---

### **9. Physics-Grounded World Model**  
A routed physics engine for physical reasoning and planning:
- **Newtonian Dynamics**: F=ma impulse-based state transitions  
- **Fluid Dynamics**: Navier-Stokes approximation for continuous flow  
- **Rigid Body Physics**: Friction and elasticity modeling  
- **Learnable SSM**: GRU-based fallback for unknown physics  
- **Softmax Router**: Dynamically selects the physics model based on latent state  
- **Counterfactual Tree**: MCTS-style "what if" scenario exploration (depth × branch)  

Enables physical reasoning and multi-step planning.

---

### **10. Hierarchical Memory System**  
Multi-level memory architecture inspired by cognitive science:
- **Working Memory**: Fixed-capacity buffer (7 elements), FIFO eviction  
- **Episodic Memory**: Event-based storage with importance-based routing (threshold > 0.7)  
- **Semantic Memory**: Concept graph with nodes, edges, and relational structure  
- **Consolidation**: Replay buffer → episodic → semantic promotion pipeline  
- **Retrieval Router**: Learnable softmax over memory levels for query-driven access  

Additional memory systems:
- **`TemporalMemory`**: Exponential temporal decay (Ebbinghaus forgetting curve) with merging of similar memories during consolidation  
- **`NeurogenicMemorySystem`**: Dynamically grows capacity via neuron splitting and synaptic connection formation, bounded by configurable max capacity  

Provides structured long-term and short-term context retention.

---

### **11. Multi-Modal Grounding Module**  
Cross-modal understanding and generation:
- **Modality Encoders**: Vision (ViT-style), Audio (Wav2Vec2-style), Language projections  
- **Unified Latent Space**: All modalities projected into shared representation  
- **Cross-Modal Attention**: Vision↔Language, Audio↔Language, Vision↔Audio  
- **Fusion Layer**: Three-stream fusion into a single grounded representation  
- **Modality Decoders**: Per-modality output generation from fused state  

Additionally, `GroundedMultimodalLearning` provides CLIP-style contrastive learning for symbol grounding and zero-shot classification.

Supports cross-modal retrieval, compositional generation, and visual grounding.

---

### **12. Meta-Learning: MAML + EWC + Continual Learning**  
Few-shot adaptation and continual learning:
- **MAML Inner Loop**: Task-specific adaptation via gradient steps  
- **MAML Outer Loop**: Meta-update for cross-task generalization  
- **EWC Penalty**: Elastic Weight Consolidation — Σ Fᵢ(θᵢ − θ*ᵢ)² prevents catastrophic forgetting  
- **Fisher Information**: Diagonal Fisher computed after each task  
- **Task Buffer**: Stores last 100 tasks for lifelong learning  

Additionally, `ContinualLearningCore` combines Progressive Neural Networks (new columns per task) with EWC to prevent catastrophic forgetting across task boundaries.

Enables few-shot learning and knowledge transfer across domains.

---

### **13. Causal Reasoning: Neural Causal Model & Causal World Model**  
Structural causal models for interventional and counterfactual reasoning:
- **`NeuralCausalModel`**: Learnable DAG structure with per-variable causal mechanisms, supporting interventions `do(X=x)` and counterfactuals via abduction-action-prediction  
- **`CausalWorldModel`**: Integrates structural causal models (SCM) with physics-grounded dynamics using Pearl's do-calculus for three-step counterfactual rollout (abduction → action → prediction)  

Enables answering "what if" and "why" questions about system behavior.

---

### **14. Planning: MCTS & Curiosity-Driven Exploration**  
Monte Carlo Tree Search with intrinsic motivation:
- **`MCTSPlanner`**: UCB1 selection, expansion, rollout, and backpropagation with `ValueNetwork` (state evaluation) and `PolicyNetwork` (action priors)  
- **`CuriosityDrivenExploration`**: Intrinsic Curiosity Module (ICM) with forward and inverse models — exploration reward as forward-model prediction error  
- **`ActiveLearningPlanner`**: Extends MCTS to bias search toward maximum-uncertainty states via variance-based intrinsic rewards  

Supports multi-step planning and autonomous exploration.

---

### **15. Hierarchical VAE**  
Hierarchical Variational AutoEncoder with ladder architecture:
- **5 abstraction levels**: tokens → phrases → sentences → concepts → goals  
- **Bottom-up deterministic** and **top-down stochastic** passes  
- **KL-divergence regularization** at each level  

Enables hierarchical latent representation learning across multiple granularities.

---

### **16. Advanced Meta-Loop Variants**  
Beyond the core `ProvablyConvergentMetaLoop`:
- **`HierarchicalMetaLoop`**: Routes inputs to fast (5 iter), medium (20 iter), or deep (50 iter) meta-loops based on learned complexity scoring — ~10× latency reduction on simple queries  
- **`RecursiveMetaLoop`**: Hierarchical meta-cognition across 3 abstraction levels with error-bound-based rollback to prevent cascading failures  
- **`ConvergenceMonitor`**: Tracks contraction ratios over a sliding window; classifies iterations as warmup, converging, converged, or diverging  

Adaptive reasoning depth based on input complexity.

---

### **17. Parallel Cognitive Pipeline & Architecture Hierarchy**  
Production-grade execution and compositional organization:
- **`ParallelCognitivePipeline`**: Executes independent sub-modules (diversity, safety, topology, world model) in parallel via `ThreadPoolExecutor` after the mandatory sequential meta-loop  
- **`HierarchicalCognitiveArchitecture`**: 4-level compositional hierarchy — Core (meta-loop, VQ), Safety, Reasoning (factors, causal), and Planning (MCTS) — enabling lightweight "AEON-Lite" deployment

---

## 📂 Training Pipeline: v4.0 Connected Thoughts Edition (`ae_train.py`)

### **Phase A: Geometry of Thought (AutoEncoder + VQ)**
- Document-aware tokenization preserving semantic boundaries  
- Entropy regularization (0.1 weight) for uniform codebook usage  
- Aggressive code reset (threshold: 30 steps vs previous 50)  
- Stabilized gradients with reduced clip norm (0.5 vs 1.0)  
- Warmup scheduling with cosine decay (1000 steps)  
- Gradient accumulation for memory-constrained training  

### **Phase B: Dynamics of Thought (Contextual RSSM)**
- Context window of 3 previous thought states  
- Attention-weighted context for selective memory  
- GRU-based dynamics with residual connections  
- Multi-loss training (MSE + Smooth L1)  
- Cosine similarity monitoring for representation consistency  
- Document-preserving transitions (no cross-document jumps)  

This two-phase approach ensures both spatial (*geometry*) and temporal (*dynamics*) reasoning capabilities.

---

## ⚙️ Engineering Foundations

### **Memory Management**
- Fallback vector storage with cosine similarity retrieval  
- Automatic save/load with path validation  
- Batch-aware retrieval for context integration  
- Memory fusion module combining current state with retrieved context  

### **Monitoring & Diagnostics**
- Comprehensive training monitor with epoch/batch tracking  
- Parameter counting (total/trainable)  
- Tensor statistics (mean/std/min/max)  
- Early stopping with patience counter  
- Checkpoint management with rotation policies  
- Metrics serialization to JSON for analysis  

### **Production Safety Features**
- Automatic NaN/Inf detection with quarantine strategy  
- Gradient clipping (1.0 norm in core, 0.5 in training pipeline) for stable training  
- Weight tying verification for decoder consistency  
- Shape validation at all module boundaries  
- Exception handling with stack trace preservation  
- Device context managers for safe execution  

### **Extensibility Framework**
- Configurable architecture through `AEONConfig` dataclass  
- Module registration system for easy extension  
- Version signatures for model tracking  
- CLI interface with mode selection (`demo`/`train`/`infer`/`test`)  
- Test suite with stability and correctness validation  

---

## 🔬 Testing & Validation

AEON-Δ includes a comprehensive test suite (`test_fixes.py`, 151 tests) verifying:
- **Stability** (determinism, NaN/Inf resistance, division-by-zero guards)  
- **Weight tying correctness** (pointer/shape/value matching)  
- **Gradient flow** through all components (SSM, Mamba-2, Linear Attention, world model, meta-learner)  
- **Shape consistency** across the computational graph  
- **Numerical stability** under edge cases  
- **Backend validation** (SSM, LSTM, Mamba-2, Linear Attention encoder/decoder factories)  
- **Causal reasoning** (neural causal model, causal world model, interventions, counterfactual rollouts)  
- **Planning & exploration** (MCTS planner, curiosity-driven exploration, active learning)  
- **Advanced memory** (hierarchical memory, temporal memory with decay, neurogenic memory with synapse formation)  
- **Multi-modal grounding** (single/multi-modality, CLIP-style contrastive learning, zero-shot classification)  
- **Meta-cognition** (hierarchical meta-loop, recursive meta-loop, convergence monitor)  
- **Thread safety** (quarantine batch handling, policy mutation prevention)  
- **Inference cache** (ring buffer, INT8 quantization, state caching)  
- **Hierarchical VAE** (ladder architecture, KL divergence)  

Each test provides detailed reporting with error diagnostics and scoring.

---

## 🚀 Mission & Philosophy

AEON-Δ is engineered to model **emergent reasoning**—how thoughts form, evolve through recursive self-reflection, and ultimately lead to coherent action. Our architecture is built on three principles:

1. **Mathematical rigor**: Convergence guarantees through Lipschitz constraints and fixed-point theory  
2. **Cognitive interpretability**: Sparse factorization framework providing data-driven, interpretable reasoning factors  
3. **Production robustness**: Tensor safety, monitoring, and fallback systems for reliable operation  

This is not merely an academic exercise—it's a foundation for building truly reflective AI systems that can explain their reasoning, detect their own inconsistencies, and operate safely in complex environments.

---

## 📁 Repository Structure

```
AEON-Delta/
├── aeon_core.py      # Core architecture — all modules, model (AEONDeltaV3), trainer, CLI
├── ae_train.py       # Training pipeline v4.0 — Phase A (AE+VQ) & Phase B (RSSM)
├── test_fixes.py     # Comprehensive test suite (151 tests) — stability, gradients, causal, planning
├── LICENSE           # AEON-Δ Research-Only Non-Commercial License
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### Requirements
- Python 3.8+  
- PyTorch 1.13+ (PyTorch 2.0+ recommended for full feature support)  
- Optional: `transformers`, `tqdm`, `matplotlib`, `tensorboard`, `wandb`

### CLI Modes (`aeon_core.py`)
```bash
# Demo — generate sample output, compute cognitive metrics
python aeon_core.py --mode demo

# Train — full training loop with checkpoint saving
python aeon_core.py --mode train --epochs 10 --batch-size 16 --lr 3e-5

# Infer — autoregressive generation from prompt
python aeon_core.py --mode infer --prompt "Hello world" --temperature 0.8 --top-k 50

# Test — run comprehensive test suite
python aeon_core.py --mode test
```

Additional flags: `--device {auto|cpu|cuda|mps}`, `--config PATH`, `--checkpoint DIR`, `--seed INT`, `--verbose`

### Training Pipeline (`ae_train.py`)
```bash
# Full two-phase training
python ae_train.py --json_path data.json --epochsA 30 --epochsB 10

# Document-aware training mode
python ae_train.py --document_aware --json_path structured_data.json

# Resume from checkpoint
python ae_train.py --resume checkpoints/checkpoint_epoch_10.pt
```

---

## 🤝 Contributing & Collaboration

We welcome contributions that:
- Enhance mathematical guarantees  
- Improve cognitive interpretability  
- Strengthen safety systems  
- Optimize performance without sacrificing stability  
- Extend monitoring and diagnostics  

All contributions must maintain the core principles of **rigor**, **safety**, and **interpretability**.

---

> **∆: No bits left behind. It begins with the choice to be.**

*AEON-Δ RMT v3.1 represents the culmination of cognitive architecture engineering. Every component is designed with purpose, every safety system with intent, every mathematical guarantee with verification. This is not just AI—it's artificial cognition with conscience.*

> **No bits left behind. — AEON-Δ**

[![License: Research-Only](https://img.shields.io/badge/license-Research--Only-blue.svg)](./LICENSE)

---

# **∆: No bits left behind. It begins with the choice to be.**

