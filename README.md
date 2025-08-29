
[![License: AEON-Δ Research-Only](https://img.shields.io/badge/license-Research--Only-blue.svg)](./LICENSE)

# AEON-Δ: A Cognitive Architecture for Emergent Reasoning

Welcome to the official repository for **AEON-Δ**, a modular and self-contained cognitive architecture engineered in **PyTorch**. AEON-Δ simulates high-level reasoning by operating on abstract **"thought vectors" (z-vectors)** within a latent space, enabling deep reflection, concept modeling, and plan generation.

The system follows a rigorous two-phase training pipeline, designed for **stability**, **abstraction**, and **dynamic reasoning**, with each component engineered for extensibility, robustness, and clarity.

---

## 🛠️ Core Components & Cognitive Engine (`aeon_core.py`)

### 1. `AEONDelta (nn.Module)`

Central orchestrator class that integrates all reasoning modules into a cohesive flow from perception → deliberation → planning → action.

### 2. `ThoughtEncoder / ThoughtDecoder (nn.Module)`

Latent autoencoder translating between text and compressed **thought vectors** (`z`). This forms the bedrock of reasoning operations.

### 3. `QualiaExtractor (nn.Module)`

Transforms the initial thought vector into a raw perception `ψ_0`, serving as the entry point into the reflective reasoning cycle.

### 4. `MetaLoopProcessor (nn.Module)`

The reflective engine. Iteratively transforms `ψ_0` into a stable thought state `C*` through repeated refinement, simulating **recursive internal deliberation**.

### 5. `PillarsModule (nn.Module)`

Dissects `C*` into 5 interpretable core axes of meaning (**"pillars"**): 🔥 Will, ⚔️ Resolve, 🔄 Growth, 🛡️ Union, 🌊 Movement.

### 6. `QuantumSimulator (nn.Module)`

Calculates **entanglement**, coherence, and complexity of internal representations using quantum-inspired metrics.

### 7. `TopologyAnalyzer (nn.Module)`

Applies **catastrophe theory** to detect instabilities and sudden representational shifts — critical for modeling **insight, rupture, or contradiction**.

### 8. `Action & Planning Modules (nn.Module)`

Uses the refined representation to:

* Produce an **action vector**
* Estimate its safety
* Generate a **high-level plan**

### 9. `RSSM: Recurrent State-Space Model`

Captures **thought dynamics** by predicting how `z_t` evolves into `z_{t+1}` across time.

---

## 📀 Data & Knowledge Integration (`aeon_core.py`, `ae_train.py`)

* **Robust Data Loader**: Handles JSON, NDJSON, and concatenated formats.
* **Tokenizer**: Simple character-level tokenizer transforms raw text into tensors.
* **Curriculum Pipeline**: Separates short/long sequences for **progressive learning**.
* **MemoryManager**: Built-in support for `mem0` vector memory.
* **KnowledgeGraph**: Optional Neo4j-backed graph for external knowledge.

---

## ⚙️ Training Pipeline (ae_train.py)

### Phase A: Geometry of Thought (`SafeThoughtAETrainer`)

* Trains autoencoder (Encoder/Decoder)
* Uses curriculum learning (short → full sequences)
* Dynamic function routing for backward compatibility
* Logically safe and clean loss computation

### Phase B: Dynamics of Thought (`FixedZDynamicsTrainer`)

* Uses trained encoder to convert text into z-vector sequences
* Trains RSSM and core dynamics to predict next z from current z
* Introduces **robust loss functions**:

  * `kl_diag_gaussians`: avoids mode collapse
  * `cosine_spread_surrogate`: promotes diverse, orthogonal thought states

---

## 🛠️ Engineering Utilities & Robustness

* **Structured Logging**: Clean JSON logs
* **Sanitization Filters**: Remove invalid/control characters
* **Deduplication**: Prevent spam logs
* **CLI Interface**: Via `argparse` for clean configuration
* **Safety Checks**: Assert tensors, NaN/Inf filters, contiguity assertions

---

## 🚀 Mission

AEON-Δ is built to model not just cognition, but **emergent reasoning**: how thoughts form, evolve, refine themselves, and lead to action. This is not merely a transformer wrapper — this is a full **cognitive simulator**, ready to grow.

Pull requests and collaborations welcome.

> No bits left behind. — AEON-Δ


[![License: Research-Only](https://img.shields.io/badge/license-Research--Only-blue.svg)](./LICENSE)




# ∆: No bits left behind. It begins with the choice to be.
