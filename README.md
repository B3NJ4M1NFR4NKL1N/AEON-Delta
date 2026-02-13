[![License: AEON-Δ Research-Only](https://img.shields.io/badge/license-Research--Only-blue.svg)](./LICENSE)

# **AEON-Δ RMT v3.1: A Cognitive Architecture for Emergent Reasoning**

> *No bits left behind. It begins with the choice to be. — AEON-Δ*

---

## 🌟 Overview

**AEON-Δ RMT v3.1** (*Reflective Meta-Thinking*) is a rigorously engineered cognitive architecture implemented in PyTorch. It simulates high-level reasoning through abstract thought vectors in latent space, with mathematical convergence guarantees and multi-level safety systems. This is not merely a transformer wrapper — it's a full cognitive simulator engineered for robustness, interpretability, and emergent reasoning capabilities.

### Core Principles

1. **Mathematical rigor** — convergence guarantees through Lipschitz constraints and fixed-point theory
2. **Cognitive interpretability** — sparse factorization framework providing data-driven, interpretable reasoning factors
3. **Production robustness** — tensor safety, monitoring, and fallback systems for reliable operation

---

## 🧠 Architecture

### 1. Sequence Processing Backends

AEON-Δ v3.1 provides multiple sequence processing backends that surpass Transformer in key dimensions:

| Dimension | Transformer | SSM / Mamba-1 | Mamba-2 SSD | Linear Attention |
|---|---|---|---|---|
| **Inference** | O(n²) | O(1) per token | O(1) per token | O(1) per token |
| **Training** | O(n²) | O(n) | O(n) chunk-wise | O(n) |
| **Memory** | Quadratic | Linear | Linear | Linear |
| **Multi-head** | Yes | No | Yes | Yes |

**Backends** (configured via `AEONConfig.encoder_backend` / `decoder_backend`):

| Backend | Description |
|---|---|
| `ssm` *(default)* | Selective State Space Model (Gu & Dao, 2023) — input-dependent transitions, parallel scan |
| `mamba2` | Mamba-2 SSD (Dao & Gu, 2024) — multi-head SSM, chunk-wise SSD, RMSNorm |
| `linear_attention` | ELU-based kernel linear attention — O(n) via associativity |
| `lstm` | Original LSTM backend for backward compatibility |

**Scalability features:**
- `ChunkedSequenceProcessor` — arbitrary-length sequences in overlapping chunks with state propagation
- `InferenceCache` — persistent state caching for O(1) autoregressive generation
- `PretrainedBackboneAdapter` — bottleneck adapter for HuggingFace pretrained models

---

### 2. Meta-Cognition & Convergence

The meta-cognitive subsystem implements provably convergent reasoning with adaptive depth.

**Core Meta-Loop** (`ProvablyConvergentMetaLoop`):
- Lipschitz-constrained Lambda operator with spectral normalization
- Banach Fixed-Point Theorem guarantees when *L < 1*
- Anderson acceleration for 2–5× convergence speedup
- Adaptive alpha, certified error bounds, early stopping

Transforms initial perception ψ₀ into a stable thought state **C\*** through iteratively refined deliberation.

**Advanced Variants:**

| Variant | Description |
|---|---|
| `HierarchicalMetaLoop` | Routes inputs to fast (5 iter), medium (20), or deep (50) loops — ~10× latency reduction on simple queries |
| `RecursiveMetaLoop` | 3-level hierarchical meta-cognition with error-bound-based rollback |
| `CertifiedMetaLoop` | Formally certified convergence via Interval Bound Propagation (IBP) |
| `AdaptiveMetaLoop` | Adaptive Computation Time (ACT) — learned halting probability, ponder cost regularization |
| `ConvergenceMonitor` | Tracks contraction ratios; classifies iterations as warmup / converging / converged / diverging |

**Execution Pipeline:**
- `ParallelCognitivePipeline` — executes independent sub-modules (diversity, safety, topology, world model) in parallel via `ThreadPoolExecutor` after the sequential meta-loop
- `HierarchicalCognitiveArchitecture` — 4-level compositional hierarchy (Core → Safety → Reasoning → Planning), enabling lightweight "AEON-Lite" deployment

---

### 3. Reasoning & Factorization

Combines learnable sparse factorization with neuro-symbolic logic for interpretable reasoning.

**Sparse Factorization** (`SparseFactorization`):
- Learnable sparse basis (64 factors by default)
- L1-sparsity regularization for compact, interpretable representations
- Encode/decode symmetry with LayerNorm — factors emerge from data, no imposed categories

**Causal Reasoning:**
- `CausalFactorExtractor` — learnable DAG structure for causal reasoning over factors
- `NeuralCausalModel` — per-variable causal mechanisms, `do(X=x)` interventions, counterfactuals via abduction-action-prediction
- `CausalWorldModel` — SCM + physics dynamics using Pearl's do-calculus for three-step counterfactual rollout

**Neuro-Symbolic Reasoning:**
- `DifferentiableForwardChainer` — continuous fuzzy logic prover with product t-norms and monotonic knowledge accumulation
- `NeuroSymbolicReasoner` — full pipeline: neural representations → soft predicates → forward chaining → conclusions
- Sigmoid-bounded facts/rules in [0, 1], learnable rule weights

---

### 4. World Modeling & Planning

Unified world modeling with physics grounding, hierarchical abstractions, and exploration-driven planning.

**Physics-Grounded World Model:**
- Newtonian Dynamics (F=ma), Fluid Dynamics (Navier-Stokes), Rigid Body Physics
- Learnable SSM (GRU-based fallback), Softmax Router for dynamic model selection
- Counterfactual Tree — MCTS-style "what if" exploration (depth × branch)

**Hierarchical World Model:**
- 3 abstraction levels: reactive (1-step), tactical (10-step), strategic (100-step)
- Bottom-up state abstraction, top-down goal/subgoal decomposition
- Bidirectional information flow between all levels

**Planning & Exploration:**
- `MCTSPlanner` — UCB1 selection, expansion, rollout, backpropagation with `ValueNetwork` and `PolicyNetwork`
- `CuriosityDrivenExploration` — ICM with forward/inverse models; reward = forward-model prediction error
- `ActiveLearningPlanner` — MCTS biased toward maximum-uncertainty states via variance-based intrinsic rewards

---

### 5. Memory Systems

Multi-level memory architecture combining cognitive-inspired and differentiable approaches.

**Hierarchical Memory** (cognitive science–inspired):
- **Working Memory** — fixed-capacity buffer (7 elements), FIFO eviction
- **Episodic Memory** — event-based storage, importance routing (threshold > 0.7)
- **Semantic Memory** — concept graph with nodes, edges, relational structure
- **Consolidation** — replay buffer → episodic → semantic promotion pipeline
- **Retrieval Router** — learnable softmax over memory levels

**Extended Memory Systems:**
- `TemporalMemory` — exponential decay (Ebbinghaus curve), merging of similar memories during consolidation
- `NeurogenicMemorySystem` — dynamic capacity growth via neuron splitting and synaptic formation, bounded by max capacity

**Unified Memory — Differentiable Neural Computer (DNC):**
- Content-addressable memory matrix with read/write heads and attention-based addressing
- Usage vector with LRU-based slot allocation
- Temporal link matrix tracking sequential write relationships
- Batched operations, fully differentiable — gradient-based optimization of access patterns

---

### 6. Multi-Modal Grounding

Cross-modal understanding and generation across vision, audio, and language.

- **Modality Encoders**: Vision (ViT-style), Audio (Wav2Vec2-style), Language projections
- **Unified Latent Space**: All modalities projected into shared representation
- **Cross-Modal Attention**: Vision↔Language, Audio↔Language, Vision↔Audio
- **Fusion Layer**: Three-stream fusion into a single grounded representation
- **Modality Decoders**: Per-modality output generation from fused state
- `GroundedMultimodalLearning` — CLIP-style contrastive learning, symbol grounding, zero-shot classification

---

### 7. Meta-Learning & Continual Learning

Few-shot adaptation and lifelong learning without catastrophic forgetting.

- **MAML**: Inner loop (task-specific adaptation) + outer loop (cross-task meta-update)
- **EWC Penalty**: Elastic Weight Consolidation — Σ Fᵢ(θᵢ − θ*ᵢ)² with diagonal Fisher per task
- **Task Buffer**: Stores last 100 tasks for lifelong learning
- `ContinualLearningCore` — Progressive Neural Networks (new columns per task) + EWC

---

### 8. Latent Representations

**Vector Quantizer (VQ-VAE):**
- EMA updates for stable codebook evolution
- Code revival (reinitializing dead codes) and code splitting (balancing overused codes)
- Perplexity monitoring, Straight-Through Estimator for gradient flow

**Hierarchical VAE:**
- 5 abstraction levels: tokens → phrases → sentences → concepts → goals
- Bottom-up deterministic and top-down stochastic passes
- KL-divergence regularization at each level

---

### 9. Safety, Stability & Self-Reporting

Production-grade safety and transparency across all system levels.

**Tensor Safety & Device Management:**
- NaN/Inf handling policies (`RAISE`, `WARN`, `SILENT`, `QUARANTINE`) with automatic sanitization
- Thread-safe device management, AMP with fallback, MPS (Apple Silicon) support

**Multi-Level Safety System** (three tiers):
1. Action safety — specific action validation
2. Cognitive safety — thought stability metrics
3. Ethical alignment — value-consistent decision making

**Topology Analyzer — Catastrophe Detection:**
- Fast Hessian via finite differences *(default)*, forward-mode AD, or Hutchinson estimator
- Eigenvalue analysis, catastrophe classifier, potential landscape analysis

**Diversity Metric:**
- Variance-based diversity across factor activations
- Action propensity from softmax over learned projection

**Transparent Self-Reporting:**
- Honesty gate, internal consistency, confidence score, report vector
- Enables external verification of AEON's internal state and reasoning quality

---

## 📂 Training Pipeline v4.0 (`ae_train.py`)

Two-phase training ensuring both spatial (*geometry*) and temporal (*dynamics*) reasoning capabilities.

**Phase A — Geometry of Thought (AutoEncoder + VQ):**
- Document-aware tokenization, entropy regularization (0.1 weight)
- Aggressive code reset (threshold: 30 steps), gradient clip 0.5
- Warmup with cosine decay (1000 steps), gradient accumulation

**Phase B — Dynamics of Thought (Contextual RSSM):**
- Context window of 3 previous thought states with attention weighting
- GRU-based dynamics, residual connections, multi-loss (MSE + Smooth L1)
- Cosine similarity monitoring, document-preserving transitions

---

## ⚙️ Engineering Foundations

| Area | Key Features |
|---|---|
| **Memory Management** | Cosine similarity retrieval, auto save/load, batch-aware retrieval, memory fusion |
| **Monitoring** | Epoch/batch tracking, parameter counting, tensor stats, early stopping, checkpoint rotation, JSON metrics |
| **Safety** | NaN/Inf detection + quarantine, gradient clipping (1.0 core / 0.5 training), weight tying verification, shape validation |
| **Extensibility** | `AEONConfig` dataclass, module registration, version signatures, CLI (`demo`/`train`/`infer`/`test`) |

---

## 🔬 Testing & Validation

Comprehensive test suite (`test_fixes.py`, **170 tests**) covering:

| Category | What's Tested |
|---|---|
| **Core stability** | Determinism, NaN/Inf resistance, division-by-zero guards, numerical edge cases |
| **Backends** | SSM, LSTM, Mamba-2, Linear Attention — encoder/decoder factories, gradient flow |
| **Memory** | Hierarchical, temporal decay, neurogenic synapse formation, DNC read/write, temporal links |
| **Reasoning** | Neural causal model, causal world model, interventions, counterfactuals, neuro-symbolic forward chaining |
| **Planning** | MCTS planner, curiosity-driven exploration, active learning |
| **Multi-modal** | Single/multi-modality, CLIP contrastive learning, zero-shot classification |
| **Meta-cognition** | Hierarchical/recursive/certified/adaptive meta-loops, convergence monitor |
| **Safety & infra** | Thread safety, quarantine batching, inference cache, weight tying, hierarchical VAE |

---

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch 1.13+ (2.0+ recommended)
- Optional: `transformers`, `tqdm`, `matplotlib`, `tensorboard`, `wandb`

### CLI Modes (`aeon_core.py`)
```bash
python aeon_core.py --mode demo                                        # Sample output + metrics
python aeon_core.py --mode train --epochs 10 --batch-size 16 --lr 3e-5 # Training
python aeon_core.py --mode infer --prompt "Hello world" --temperature 0.8 --top-k 50  # Generation
python aeon_core.py --mode test                                        # Test suite
```

Additional flags: `--device {auto|cpu|cuda|mps}`, `--config PATH`, `--checkpoint DIR`, `--seed INT`, `--verbose`

### Training Pipeline (`ae_train.py`)
```bash
python ae_train.py --json_path data.json --epochsA 30 --epochsB 10           # Full two-phase
python ae_train.py --document_aware --json_path structured_data.json          # Document-aware
python ae_train.py --resume checkpoints/checkpoint_epoch_10.pt                # Resume
```

---

## 📁 Repository Structure

```
AEON-Delta/
├── aeon_core.py      # Core architecture — modules, model (AEONDeltaV3), trainer, CLI
├── ae_train.py       # Training pipeline v4.0 — Phase A (AE+VQ) & Phase B (RSSM)
├── test_fixes.py     # Test suite (170 tests) — stability, gradients, causal, planning
├── LICENSE           # AEON-Δ Research-Only Non-Commercial License
├── README.md
└── .gitignore
```

---

## 🤝 Contributing

We welcome contributions that enhance mathematical guarantees, improve cognitive interpretability, strengthen safety systems, optimize performance, or extend monitoring. All contributions must maintain the core principles of **rigor**, **safety**, and **interpretability**.

---

**License**: [AEON-Δ Research-Only Non-Commercial](./LICENSE)

> **∆: No bits left behind. It begins with the choice to be.**
