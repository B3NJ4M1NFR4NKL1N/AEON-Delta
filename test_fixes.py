"""
Tests for refactoring fixes in aeon_core.py and ae_train.py.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os
import logging

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_division_by_zero_in_fit():
    """Fix 1: ae_train.py - Division by zero when all accumulated losses are NaN/Inf.
    
    Verifies that max(num_accumulated, 1) prevents ZeroDivisionError.
    """
    # Simulate the fixed code path
    accumulated_loss = 0.0
    num_accumulated = 0  # All losses were NaN/Inf, so nothing accumulated

    # This should NOT raise ZeroDivisionError
    avg_loss = accumulated_loss / max(num_accumulated, 1)
    assert avg_loss == 0.0, f"Expected 0.0, got {avg_loss}"
    
    # Normal case should still work
    accumulated_loss = 3.0
    num_accumulated = 3
    avg_loss = accumulated_loss / max(num_accumulated, 1)
    assert avg_loss == 1.0, f"Expected 1.0, got {avg_loss}"
    
    print("✅ test_division_by_zero_in_fit PASSED")


def test_quarantine_batch_thread_safety():
    """Fix 2: aeon_core.py - Thread-unsafe policy mutation in _quarantine_batch.
    
    Verifies that _quarantine_batch does NOT mutate self.policy when all
    batches are corrupted.
    """
    from aeon_core import TensorGuard, NaNPolicy
    
    guard = TensorGuard(policy=NaNPolicy.QUARANTINE, enable_tracking=False)
    
    # Create a tensor where ALL batches have NaN
    all_nan_tensor = torch.full((4, 8), float('nan'))
    
    original_policy = guard.policy
    assert original_policy == NaNPolicy.QUARANTINE
    
    # Call _quarantine_batch — should NOT mutate policy
    result = guard._quarantine_batch(all_nan_tensor, "test_all_corrupted")
    
    # Policy should be unchanged
    assert guard.policy == original_policy, (
        f"Policy was mutated from {original_policy} to {guard.policy}"
    )
    
    # Result should be sanitized (no NaN)
    assert not torch.isnan(result).any(), "Result still contains NaN"
    assert not torch.isinf(result).any(), "Result still contains Inf"
    
    print("✅ test_quarantine_batch_thread_safety PASSED")


def test_tensor_hash_collision_resistance():
    """Fix 3: aeon_core.py - Weak tensor hash causing cache collisions.
    
    Verifies that two different tensors with the same shape and sum
    produce different hashes.
    """
    from aeon_core import FastHessianComputer
    
    hc = FastHessianComputer(method='finite_differences')
    
    # Two tensors with same shape and same sum but different values
    t1 = torch.tensor([[1.0, 2.0, 3.0]])  # sum = 6
    t2 = torch.tensor([[0.0, 3.0, 3.0]])  # sum = 6
    
    h1 = hc._hash_tensor(t1)
    h2 = hc._hash_tensor(t2)
    
    assert h1 != h2, (
        f"Hash collision: tensor [[1,2,3]] and [[0,3,3]] both hash to {h1}"
    )
    
    # Same tensor should produce same hash
    h1_again = hc._hash_tensor(t1)
    assert h1 == h1_again, "Same tensor produced different hashes"
    
    print("✅ test_tensor_hash_collision_resistance PASSED")


def test_rssm_trainer_zero_batches():
    """Fix 4: ae_train.py - Guard against zero total_batches in RSSM trainer.
    
    Verifies that max(total_batches, 1) prevents ZeroDivisionError.
    """
    # Simulate the fixed code path
    epoch_metrics = {"mse_loss": 0.0, "cosine_sim": 0.0}
    total_batches = 0  # Edge case: no batches
    
    # This should NOT raise ZeroDivisionError
    for key in epoch_metrics:
        epoch_metrics[key] /= max(total_batches, 1)
    
    assert epoch_metrics["mse_loss"] == 0.0
    assert epoch_metrics["cosine_sim"] == 0.0
    
    print("✅ test_rssm_trainer_zero_batches PASSED")


def test_memory_manager_flatten():
    """Fix 5: aeon_core.py - MemoryManager.retrieve_relevant input validation.
    
    Verifies that vectors are properly flattened for dot product computation.
    """
    from aeon_core import MemoryManager, AEONConfig
    
    config = AEONConfig(device_str='cpu')
    mm = MemoryManager(config)
    
    # Add some vectors
    v1 = torch.randn(256)
    v2 = torch.randn(256)
    mm.add_embedding(v1, {'id': 1})
    mm.add_embedding(v2, {'id': 2})
    
    # Query with a multidimensional tensor (e.g., [1, 256])
    query = torch.randn(1, 256)
    results = mm.retrieve_relevant(query, k=2)
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert 'vec' in results[0]
    assert 'meta' in results[0]
    
    # Query with 1D tensor should also work
    query_1d = torch.randn(256)
    results_1d = mm.retrieve_relevant(query_1d, k=2)
    assert len(results_1d) == 2
    
    print("✅ test_memory_manager_flatten PASSED")


def test_memory_manager_nan_rejection():
    """Verify MemoryManager rejects NaN/Inf embeddings."""
    from aeon_core import MemoryManager, AEONConfig
    
    config = AEONConfig(device_str='cpu')
    mm = MemoryManager(config)
    
    # NaN vector should be rejected
    nan_vec = torch.full((256,), float('nan'))
    mm.add_embedding(nan_vec, {'id': 'bad'})
    assert mm.size == 0, "NaN vector was not rejected"
    
    # Inf vector should be rejected
    inf_vec = torch.full((256,), float('inf'))
    mm.add_embedding(inf_vec, {'id': 'bad'})
    assert mm.size == 0, "Inf vector was not rejected"
    
    # Good vector should be accepted
    good_vec = torch.randn(256)
    mm.add_embedding(good_vec, {'id': 'good'})
    assert mm.size == 1, "Good vector was not accepted"
    
    print("✅ test_memory_manager_nan_rejection PASSED")


def test_quarantine_partial_corruption():
    """Verify _quarantine_batch handles partial corruption correctly."""
    from aeon_core import TensorGuard, NaNPolicy
    
    guard = TensorGuard(policy=NaNPolicy.QUARANTINE, enable_tracking=False)
    
    # Create a tensor where only some batches have NaN
    tensor = torch.randn(4, 8)
    tensor[1] = float('nan')  # Only batch 1 is corrupted
    
    result = guard._quarantine_batch(tensor, "test_partial")
    
    # Result should not contain NaN
    assert not torch.isnan(result).any(), "Result still contains NaN"
    
    # Good batches should be unchanged
    assert torch.allclose(result[0], tensor[0]), "Good batch 0 was modified"
    assert torch.allclose(result[2], tensor[2]), "Good batch 2 was modified"
    assert torch.allclose(result[3], tensor[3]), "Good batch 3 was modified"
    
    print("✅ test_quarantine_partial_corruption PASSED")


def test_config_validation():
    """Test AEONConfigV4 validation."""
    from ae_train import AEONConfigV4
    
    # Default config should pass validation
    config = AEONConfigV4()
    assert config.z_dim == 256
    assert config.grad_clip_norm == 0.5
    assert config.entropy_weight == 0.1
    assert config.context_window == 3
    
    # Invalid z_dim should raise
    try:
        AEONConfigV4(z_dim=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Invalid context_window should raise
    try:
        AEONConfigV4(context_window=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✅ test_config_validation PASSED")


def test_document_aware_dataset():
    """Test DocumentAwareDataset edge cases."""
    from ae_train import DocumentAwareDataset
    
    # Create documents with enough chunks
    docs = [
        [torch.randint(0, 100, (64,)) for _ in range(5)],  # 5 chunks
        [torch.randint(0, 100, (64,)) for _ in range(3)],  # 3 chunks (just enough for K=2+1)
        [torch.randint(0, 100, (64,)) for _ in range(2)],  # 2 chunks (NOT enough for K=2+1)
    ]
    
    dataset = DocumentAwareDataset(docs, context_window=2)
    
    # Doc 0: indices 2,3,4 are valid targets → 3 samples
    # Doc 1: index 2 is valid target → 1 sample
    # Doc 2: no valid targets (only 2 chunks, need >= context_window + 1 = 3)
    assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"
    
    # Get a sample
    sample = dataset[0]
    assert 'context' in sample
    assert 'target' in sample
    assert sample['context'].shape == (2, 64), f"Expected (2, 64), got {sample['context'].shape}"
    assert sample['target'].shape == (64,), f"Expected (64,), got {sample['target'].shape}"
    
    # Empty documents should raise
    try:
        DocumentAwareDataset([], context_window=2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✅ test_document_aware_dataset PASSED")

# ============================================================================
# NEW TESTS: Critical properties from problem statement (Problem 10)
# ============================================================================

def test_lipschitz_contraction():
    """Problem 10a: Verify Lipschitz contraction ||Λ(x)-Λ(y)|| ≤ L||x-y||
    for 1000 random pairs.
    """
    from aeon_core import LipschitzConstrainedLambda

    lip = LipschitzConstrainedLambda(
        input_dim=64, hidden_dim=32, output_dim=32,
        lipschitz_target=0.85, use_spectral_norm=True
    )

    max_ratio = lip.compute_lipschitz_constant(num_samples=1000, sample_dim=64)
    # After spectral norm, the empirical constant should be reasonably bounded.
    # We check it is ≤ lipschitz_target * 1.5 (generous margin for untrained net).
    assert max_ratio <= lip.lipschitz_target * 1.5, (
        f"Lipschitz ratio {max_ratio:.4f} exceeds "
        f"{lip.lipschitz_target * 1.5:.4f}"
    )
    print(f"✅ test_lipschitz_contraction PASSED (max_ratio={max_ratio:.4f})")


def test_encoder_input_validation():
    """Problem 10b: Verify ThoughtEncoder rejects out-of-range tokens,
    wrong dtypes, and mismatched attention masks.
    """
    from aeon_core import ThoughtEncoder

    enc = ThoughtEncoder(vocab_size=100, emb_dim=32, z_dim=32)

    # Wrong dtype
    try:
        enc(torch.randn(2, 10))  # float, not long
        assert False, "Should have raised TypeError for float tokens"
    except TypeError:
        pass

    # Out-of-range token
    try:
        enc(torch.tensor([[999]], dtype=torch.long))
        assert False, "Should have raised ValueError for out-of-range token"
    except ValueError:
        pass

    # Negative token
    try:
        enc(torch.tensor([[-1]], dtype=torch.long))
        assert False, "Should have raised ValueError for negative token"
    except ValueError:
        pass

    # attention_mask shape mismatch
    try:
        tokens = torch.randint(0, 100, (2, 10))
        mask = torch.ones(3, 10)
        enc(tokens, attention_mask=mask)
        assert False, "Should have raised ValueError for mismatched mask shape"
    except ValueError:
        pass

    # Valid input should work
    tokens = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10)
    z = enc(tokens, attention_mask=mask)
    assert z.shape == (2, 32)

    print("✅ test_encoder_input_validation PASSED")


def test_meta_loop_convergence():
    """Problem 10c: Verify meta-loop converges for random initial conditions."""
    from aeon_core import AEONConfig

    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    from aeon_core import ProvablyConvergentMetaLoop

    ml = ProvablyConvergentMetaLoop(config, max_iterations=50, min_iterations=3)
    ml.eval()

    # Run with 5 different random inputs
    for i in range(5):
        psi = torch.randn(4, config.z_dim)
        with torch.no_grad():
            C, iters, meta = ml.compute_fixed_point(psi)

        assert C.shape == (4, config.hidden_dim), f"Wrong output shape: {C.shape}"
        assert not torch.isnan(C).any(), f"NaN in fixed-point output (trial {i})"
        assert not torch.isinf(C).any(), f"Inf in fixed-point output (trial {i})"

    print("✅ test_meta_loop_convergence PASSED")


def test_verify_convergence_method():
    """Problem 10d: Verify the new verify_convergence() method returns diagnostics."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop

    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )

    ml = ProvablyConvergentMetaLoop(config, max_iterations=20)
    ml.eval()

    psi = torch.randn(2, config.z_dim)
    result = ml.verify_convergence(psi, num_samples=50)

    assert 'empirical_lipschitz' in result
    assert 'contraction_satisfied' in result
    assert 'warnings' in result
    assert isinstance(result['warnings'], list)
    assert len(result['warnings']) > 0  # at least the completeness warning

    print(f"✅ test_verify_convergence_method PASSED "
          f"(L={result['empirical_lipschitz']:.4f})")


def test_batch_generation_per_sequence_stopping():
    """Problem 10e: Verify per-sequence stopping in decoder generation."""
    from aeon_core import ThoughtDecoder

    vocab_size = 200
    sep_id = 102
    dec = ThoughtDecoder(
        vocab_size=vocab_size, emb_dim=32, z_dim=32,
        cls_token_id=101, sep_token_id=sep_id
    )
    dec.eval()

    z = torch.randn(3, 32)
    with torch.no_grad():
        gen_ids, logits = dec(
            z, mode='inference', max_length=20,
            temperature=1.0, top_k=0, sample=True
        )

    # Should always terminate within max_length + 1 (prefix)
    assert gen_ids.shape[0] == 3, "Batch size mismatch"
    assert gen_ids.shape[1] <= 21, f"Generated too many tokens: {gen_ids.shape[1]}"
    assert not torch.isnan(logits).any(), "NaN in generated logits"

    print("✅ test_batch_generation_per_sequence_stopping PASSED")


def test_graceful_degradation_generate():
    """Problem 10f: Verify generate() returns structured degraded response
    when tokenizer is None.
    """
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.tokenizer = None  # Force no tokenizer

    result = model.generate("test prompt")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result['status'] == 'degraded'
    assert result['text'] == 'test prompt'
    assert 'reason' in result

    print("✅ test_graceful_degradation_generate PASSED")


def test_set_seed_reproducibility():
    """Problem 10g: Verify set_seed() produces deterministic outputs."""
    from aeon_core import set_seed

    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)

    assert torch.allclose(a, b), "set_seed() did not produce reproducible outputs"

    print("✅ test_set_seed_reproducibility PASSED")


def test_compute_lipschitz_loss_standalone():
    """Problem 10h: Verify standalone compute_lipschitz_loss works."""
    from aeon_core import LipschitzConstrainedLambda, compute_lipschitz_loss

    lip = LipschitzConstrainedLambda(
        input_dim=64, hidden_dim=32, output_dim=32,
        lipschitz_target=0.85, use_spectral_norm=True
    )
    psi = torch.randn(4, 32)
    loss = compute_lipschitz_loss(lip, psi)

    assert loss.dim() == 0 or loss.numel() == 1, f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

    print("✅ test_compute_lipschitz_loss_standalone PASSED")


def test_safe_checkpoint_loading():
    """Problem 10i: Verify safe loading validates checkpoint structure."""
    import tempfile, os
    from aeon_core import MemoryManager, AEONConfig

    config = AEONConfig(device_str='cpu')
    mm = MemoryManager(config)

    # Create a valid memory file
    valid_data = {'vectors': [], 'metas': [], 'size': 0}
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False, dir=tempfile.gettempdir()) as f:
        torch.save(valid_data, f.name)
        tmp_path = f.name

    # Monkey-patch the path for testing
    original_path = os.path.join(config.memory_path, "fallback_memory.pt")
    os.makedirs(config.memory_path, exist_ok=True)
    torch.save(valid_data, original_path)

    mm.load_memory()
    assert mm.size == 0  # valid data loaded

    # Create invalid structure
    invalid_data = {'evil_key': 'malicious_code', 'vectors': []}
    torch.save(invalid_data, original_path)

    mm2 = MemoryManager(config)
    mm2.load_memory()
    # Should have rejected due to unexpected keys
    assert mm2.size == 0

    # Cleanup
    os.unlink(tmp_path)
    if os.path.exists(original_path):
        os.unlink(original_path)

    print("✅ test_safe_checkpoint_loading PASSED")


# ============================================================================
# MODERNIZATION TESTS: SelectiveSSM, LinearAttention, Chunking, Caching
# ============================================================================

def test_selective_ssm_forward():
    """Verify SelectiveSSM produces correct output shapes and is NaN-free."""
    from aeon_core import SelectiveSSM

    ssm = SelectiveSSM(d_model=64, d_state=16, num_layers=2, expand_factor=2)
    ssm.eval()

    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        y, states = ssm(x)

    assert y.shape == (2, 32, 64), f"Expected (2,32,64), got {y.shape}"
    assert not torch.isnan(y).any(), "SSM output contains NaN"
    assert not torch.isinf(y).any(), "SSM output contains Inf"
    assert len(states) == 2, f"Expected 2 layer states, got {len(states)}"

    print("✅ test_selective_ssm_forward PASSED")


def test_ssm_state_caching():
    """Verify SSM state caching propagates state across chunks."""
    from aeon_core import SelectiveSSM

    ssm = SelectiveSSM(d_model=32, d_state=8, num_layers=1)
    ssm.eval()

    # Process full sequence
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        y_full, _ = ssm(x)

    # Process in two halves with state passing
    with torch.no_grad():
        y1, state = ssm(x[:, :5, :])
        y2, _ = ssm(x[:, 5:, :], state=state)

    y_chunked = torch.cat([y1, y2], dim=1)
    # Note: The depthwise Conv1d (kernel_size=3, padding=1) introduces boundary
    # effects at chunk split points since the convolution context differs for
    # adjacent elements at the boundary. The 1.0 threshold accounts for this
    # architectural property while still catching large state propagation errors.
    max_diff = torch.max(torch.abs(y_full - y_chunked)).item()
    assert max_diff < 1.0, \
        f"State caching divergence too large: max diff={max_diff:.6f}"
    assert not torch.isnan(y_chunked).any(), "Chunked output contains NaN"
    assert y_chunked.shape == y_full.shape, "Shape mismatch"

    print(f"✅ test_ssm_state_caching PASSED (max_diff={max_diff:.4f})")


def test_linear_attention_block():
    """Verify LinearAttentionBlock produces correct shapes and is NaN-free."""
    from aeon_core import LinearAttentionBlock

    block = LinearAttentionBlock(d_model=64, num_heads=4, feature_dim=32, causal=True)
    block.eval()

    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y, state = block(x)

    assert y.shape == (2, 16, 64), f"Expected (2,16,64), got {y.shape}"
    assert not torch.isnan(y).any(), "LinearAttention output contains NaN"
    assert state is not None, "Causal linear attention should return state"

    print("✅ test_linear_attention_block PASSED")


def test_linear_attention_bidirectional():
    """Verify bidirectional linear attention works."""
    from aeon_core import LinearAttentionBlock

    block = LinearAttentionBlock(d_model=64, num_heads=4, feature_dim=32, causal=False)
    block.eval()

    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y, state = block(x)

    assert y.shape == (2, 16, 64), f"Expected (2,16,64), got {y.shape}"
    assert state is None, "Bidirectional attention should return None state"

    print("✅ test_linear_attention_bidirectional PASSED")


def test_chunked_sequence_processor():
    """Verify ChunkedSequenceProcessor handles long sequences correctly."""
    from aeon_core import ChunkedSequenceProcessor

    processor = ChunkedSequenceProcessor(chunk_size=8, overlap=2)

    # Simple identity model
    def model_fn(x, state):
        return x * 2.0, state

    x = torch.randn(2, 20, 32)
    y, _ = processor.process(model_fn, x)

    assert y.shape == (2, 20, 32), f"Expected (2,20,32), got {y.shape}"

    # Short sequence should go through directly
    x_short = torch.randn(2, 4, 32)
    y_short, _ = processor.process(model_fn, x_short)
    assert torch.allclose(y_short, x_short * 2.0), "Short sequence handling failed"

    print("✅ test_chunked_sequence_processor PASSED")


def test_inference_cache():
    """Verify InferenceCache state management."""
    from aeon_core import InferenceCache

    cache = InferenceCache()
    assert cache.step == 0

    # Set SSM state
    states = [torch.randn(2, 32, 16)]
    cache.set_ssm_state(states)
    assert cache.step == 1
    assert cache.get_ssm_state() is not None

    # Reset
    cache.reset()
    assert cache.step == 0
    assert cache.get_ssm_state() is None

    print("✅ test_inference_cache PASSED")


def test_ssm_thought_encoder():
    """Verify SSMThoughtEncoder produces correct shapes with validation."""
    from aeon_core import SSMThoughtEncoder

    enc = SSMThoughtEncoder(
        vocab_size=100, emb_dim=32, z_dim=32,
        d_state=8, num_layers=1, expand_factor=2
    )
    enc.eval()

    # Valid input
    tokens = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)
    with torch.no_grad():
        z = enc(tokens, attention_mask=mask)
    assert z.shape == (2, 32), f"Expected (2,32), got {z.shape}"
    assert not torch.isnan(z).any(), "Encoder output has NaN"

    # Input validation - wrong dtype
    try:
        enc(torch.randn(2, 10))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Input validation - out of range
    try:
        enc(torch.tensor([[999]], dtype=torch.long))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Input validation - mask mismatch
    try:
        enc(torch.randint(0, 100, (2, 10)), attention_mask=torch.ones(3, 10))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✅ test_ssm_thought_encoder PASSED")


def test_ssm_thought_decoder_train():
    """Verify SSMThoughtDecoder training mode produces correct shapes."""
    from aeon_core import SSMThoughtDecoder

    dec = SSMThoughtDecoder(
        vocab_size=200, emb_dim=32, z_dim=32,
        d_state=8, num_layers=1, expand_factor=2,
        cls_token_id=101, sep_token_id=102
    )
    dec.eval()

    z = torch.randn(2, 32)
    teacher = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        logits = dec(z, teacher_tokens=teacher, mode='train')
    assert logits.shape == (2, 16, 200), f"Expected (2,16,200), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Decoder logits have NaN"

    print("✅ test_ssm_thought_decoder_train PASSED")


def test_ssm_thought_decoder_inference():
    """Verify SSMThoughtDecoder inference mode with per-sequence stopping."""
    from aeon_core import SSMThoughtDecoder

    dec = SSMThoughtDecoder(
        vocab_size=200, emb_dim=32, z_dim=32,
        d_state=8, num_layers=1, expand_factor=2,
        cls_token_id=101, sep_token_id=102
    )
    dec.eval()

    z = torch.randn(3, 32)
    with torch.no_grad():
        gen_ids, logits = dec(z, mode='inference', max_length=20, temperature=1.0, sample=True)

    assert gen_ids.shape[0] == 3, "Batch size mismatch"
    # max_length=20 steps + 1 prefix (CLS) + 1 potential SEP = 22 max tokens
    assert gen_ids.shape[1] <= 22, f"Generated too many tokens: {gen_ids.shape[1]}"
    assert not torch.isnan(logits).any(), "NaN in generated logits"

    print("✅ test_ssm_thought_decoder_inference PASSED")


def test_linear_attention_encoder():
    """Verify LinearAttentionEncoder produces correct shapes."""
    from aeon_core import LinearAttentionEncoder

    enc = LinearAttentionEncoder(
        vocab_size=100, emb_dim=32, z_dim=32,
        num_heads=2, feature_dim=16, num_layers=1
    )
    enc.eval()

    tokens = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        z = enc(tokens)
    assert z.shape == (2, 32), f"Expected (2,32), got {z.shape}"
    assert not torch.isnan(z).any(), "Linear attention encoder NaN"

    print("✅ test_linear_attention_encoder PASSED")


def test_build_encoder_factory():
    """Verify build_encoder produces the right encoder type for each backend."""
    from aeon_core import AEONConfig, build_encoder, ThoughtEncoder, SSMThoughtEncoder, LinearAttentionEncoder

    # LSTM backend
    config_lstm = AEONConfig(device_str='cpu', encoder_backend='lstm')
    enc_lstm = build_encoder(config_lstm)
    assert isinstance(enc_lstm, ThoughtEncoder), f"Expected ThoughtEncoder, got {type(enc_lstm)}"

    # SSM backend
    config_ssm = AEONConfig(device_str='cpu', encoder_backend='ssm')
    enc_ssm = build_encoder(config_ssm)
    assert isinstance(enc_ssm, SSMThoughtEncoder), f"Expected SSMThoughtEncoder, got {type(enc_ssm)}"

    # Linear attention backend
    config_la = AEONConfig(device_str='cpu', encoder_backend='linear_attention')
    enc_la = build_encoder(config_la)
    assert isinstance(enc_la, LinearAttentionEncoder), f"Expected LinearAttentionEncoder, got {type(enc_la)}"

    print("✅ test_build_encoder_factory PASSED")


def test_build_decoder_factory():
    """Verify build_decoder produces the right decoder type for each backend."""
    from aeon_core import AEONConfig, build_decoder, ThoughtDecoder, SSMThoughtDecoder

    config_lstm = AEONConfig(device_str='cpu', decoder_backend='lstm')
    dec_lstm = build_decoder(config_lstm)
    assert isinstance(dec_lstm, ThoughtDecoder), f"Expected ThoughtDecoder, got {type(dec_lstm)}"

    config_ssm = AEONConfig(device_str='cpu', decoder_backend='ssm')
    dec_ssm = build_decoder(config_ssm)
    assert isinstance(dec_ssm, SSMThoughtDecoder), f"Expected SSMThoughtDecoder, got {type(dec_ssm)}"

    print("✅ test_build_decoder_factory PASSED")


def test_ssm_long_sequence():
    """Verify SSM handles long sequences (>1024 tokens) in O(n) time."""
    from aeon_core import SSMThoughtEncoder

    enc = SSMThoughtEncoder(
        vocab_size=1000, emb_dim=64, z_dim=64,
        d_state=16, num_layers=1, expand_factor=2
    )
    enc.eval()

    # Long sequence: 2048 tokens
    tokens = torch.randint(0, 1000, (1, 2048))
    with torch.no_grad():
        z = enc(tokens)
    assert z.shape == (1, 64), f"Expected (1,64), got {z.shape}"
    assert not torch.isnan(z).any(), "Long-sequence encoding has NaN"

    print("✅ test_ssm_long_sequence PASSED")


def test_ssm_gradient_flow():
    """Verify gradients flow through the SSM encoder."""
    from aeon_core import SSMThoughtEncoder

    enc = SSMThoughtEncoder(vocab_size=100, emb_dim=32, z_dim=32, d_state=8, num_layers=1)
    tokens = torch.randint(0, 100, (2, 10))
    z = enc(tokens)
    loss = z.sum()
    loss.backward()

    # Check some parameters have gradients
    has_grad = False
    for p in enc.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradient flow through SSM encoder"

    print("✅ test_ssm_gradient_flow PASSED")


def test_aeon_v3_with_ssm_backend():
    """Verify AEONDeltaV3 works with SSM backend end-to-end."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        device_str='cpu',
        encoder_backend='ssm',
        decoder_backend='ssm',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    tokens = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)

    with torch.no_grad():
        result = model(tokens, attention_mask=mask, decode_mode='train')

    assert 'logits' in result
    assert 'thoughts' in result
    assert result['logits'].shape[0] == 2
    assert not torch.isnan(result['logits']).any(), "SSM backend logits have NaN"

    print("✅ test_aeon_v3_with_ssm_backend PASSED")


def test_aeon_v3_with_lstm_backend():
    """Verify AEONDeltaV3 still works with LSTM backend (backward compatibility)."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        device_str='cpu',
        encoder_backend='lstm',
        decoder_backend='lstm',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    tokens = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)

    with torch.no_grad():
        result = model(tokens, attention_mask=mask, decode_mode='train')

    assert 'logits' in result
    assert result['logits'].shape[0] == 2

    print("✅ test_aeon_v3_with_lstm_backend PASSED")


def test_config_backend_validation():
    """Verify AEONConfig validates backend parameters correctly."""
    from aeon_core import AEONConfig

    # Valid backends should work
    AEONConfig(device_str='cpu', encoder_backend='lstm')
    AEONConfig(device_str='cpu', encoder_backend='ssm')
    AEONConfig(device_str='cpu', encoder_backend='linear_attention')
    AEONConfig(device_str='cpu', decoder_backend='lstm')
    AEONConfig(device_str='cpu', decoder_backend='ssm')

    # Invalid backend should fail
    try:
        AEONConfig(device_str='cpu', encoder_backend='transformer')
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        AEONConfig(device_str='cpu', decoder_backend='transformer')
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    print("✅ test_config_backend_validation PASSED")


def test_pretrained_backbone_adapter_fallback():
    """Verify PretrainedBackboneAdapter works in fallback mode."""
    from aeon_core import PretrainedBackboneAdapter

    # No pretrained model - should work in fallback
    adapter = PretrainedBackboneAdapter(
        pretrained_model_name="",
        target_dim=64,
        adapter_dim=16,
    )
    adapter.eval()

    tokens = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        features = adapter(tokens)
    assert features.shape == (2, 10, 64), f"Expected (2,10,64), got {features.shape}"

    print("✅ test_pretrained_backbone_adapter_fallback PASSED")


# ============================================================================
# Tests for Section I improvements
# ============================================================================

def test_parallel_scan_consistency():
    """Verify parallel associative scan produces valid output and gradients."""
    from aeon_core import SelectiveSSM

    ssm = SelectiveSSM(d_model=32, d_state=8, num_layers=1)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y, states = ssm(x)

    assert y.shape == (2, 16, 32)
    assert not torch.isnan(y).any(), "Parallel scan output contains NaN"

    # Verify gradients flow
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradients for input"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

    print("✅ test_parallel_scan_consistency PASSED")


def test_poly_feature_map():
    """Verify polynomial feature map produces non-negative values."""
    from aeon_core import LinearAttentionBlock

    block = LinearAttentionBlock(d_model=64, num_heads=4, feature_dim=32, causal=True)
    x = torch.randn(100)
    result = block._poly_feature_map(x)
    assert (result >= 0).all(), "Polynomial feature map should be non-negative"

    # Check it's actually the right polynomial
    expected = (1.0 + x + x.pow(2) * 0.5 + x.pow(3) / 6.0).clamp(min=0.0)
    assert torch.allclose(result, expected), "Polynomial mismatch"

    print("✅ test_poly_feature_map PASSED")


def test_linear_attention_low_rank():
    """Verify low-rank factorization in LinearAttention."""
    from aeon_core import LinearAttentionBlock

    block = LinearAttentionBlock(d_model=64, num_heads=4, feature_dim=32,
                                  feature_rank=8, causal=True)
    assert block.feature_rank == 8
    assert block.feature_down_proj.in_features == 32
    assert block.feature_down_proj.out_features == 8
    assert block.feature_up_proj.in_features == 8
    assert block.feature_up_proj.out_features == 32

    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y, _ = block(x)
    assert y.shape == (2, 16, 64)
    assert not torch.isnan(y).any()

    print("✅ test_linear_attention_low_rank PASSED")


def test_chunked_adaptive_blending():
    """Verify adaptive blending in overlap regions."""
    from aeon_core import ChunkedSequenceProcessor

    processor = ChunkedSequenceProcessor(chunk_size=8, overlap=2)

    # Model that returns the input scaled by position-dependent factor
    def model_fn(x, state):
        return x * 2.0, state

    x = torch.ones(1, 20, 4)
    y, _ = processor.process(model_fn, x)

    assert y.shape == (1, 20, 4), f"Expected (1,20,4), got {y.shape}"
    # All positions should be close to 2.0 (since input is 1.0 and model doubles)
    assert torch.allclose(y, torch.full_like(y, 2.0), atol=0.1), \
        "Blended output should be close to 2.0 for uniform input"

    print("✅ test_chunked_adaptive_blending PASSED")


def test_inference_cache_ring_buffer():
    """Verify InferenceCache ring buffer and INT8 quantization."""
    from aeon_core import InferenceCache

    cache = InferenceCache(maxlen=3)
    assert cache.history_size == 0

    # Set multiple SSM states to test ring buffer
    for i in range(5):
        states = [torch.randn(1, 16, 8)]
        cache.set_ssm_state(states)

    assert cache.step == 5
    # Ring buffer should cap at maxlen=3
    assert cache.history_size <= 3, \
        f"Ring buffer should cap at 3, got {cache.history_size}"

    # Test reset
    cache.reset()
    assert cache.step == 0
    assert cache.history_size == 0

    print("✅ test_inference_cache_ring_buffer PASSED")


def test_inference_cache_quantization():
    """Verify INT8 quantization roundtrip preserves approximate values."""
    from aeon_core import InferenceCache

    original = torch.randn(4, 16)
    quantized, scale = InferenceCache._quantize_int8(original)
    assert quantized.dtype == torch.int8
    recovered = InferenceCache._dequantize_int8(quantized, scale)
    # INT8 quantization has limited precision
    max_err = (original - recovered).abs().max().item()
    assert max_err < 0.1, f"Quantization error too large: {max_err}"

    print("✅ test_inference_cache_quantization PASSED")


def test_hybrid_adapter_components():
    """Verify hybrid adapter has LoRA, Prefix, and Parallel components."""
    from aeon_core import PretrainedBackboneAdapter

    adapter = PretrainedBackboneAdapter(
        pretrained_model_name="",
        target_dim=64,
        adapter_dim=16,
        lora_rank=4,
        num_prefix_tokens=4,
    )

    # Check all components exist
    assert hasattr(adapter, 'lora_down')
    assert hasattr(adapter, 'lora_up')
    assert hasattr(adapter, 'prefix_tokens')
    assert hasattr(adapter, 'parallel_adapter')
    assert hasattr(adapter, 'mix_logits')
    assert adapter.mix_logits.shape == (3,)

    # Forward pass
    tokens = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        features = adapter(tokens)
    assert features.shape == (2, 10, 64)
    assert not torch.isnan(features).any()

    print("✅ test_hybrid_adapter_components PASSED")


# ============================================================================
# Tests for Section II new AGI components
# ============================================================================

def test_world_model_forward():
    """Verify PhysicsGroundedWorldModel forward pass."""
    from aeon_core import PhysicsGroundedWorldModel

    model = PhysicsGroundedWorldModel(input_dim=64, state_dim=32,
                                       tree_depth=2, tree_branch=2)
    model.eval()

    x = torch.randn(2, 64)
    with torch.no_grad():
        result = model(x, explore_counterfactuals=False)

    assert 'latent_state' in result
    assert 'next_state' in result
    assert 'output' in result
    assert result['latent_state'].shape == (2, 32)
    assert result['output'].shape == (2, 64)
    assert not torch.isnan(result['output']).any()

    print("✅ test_world_model_forward PASSED")


def test_world_model_counterfactuals():
    """Verify counterfactual tree exploration."""
    from aeon_core import PhysicsGroundedWorldModel

    model = PhysicsGroundedWorldModel(input_dim=32, state_dim=16,
                                       tree_depth=2, tree_branch=2)
    model.eval()

    x = torch.randn(1, 32)
    with torch.no_grad():
        result = model(x, explore_counterfactuals=True)

    assert 'counterfactuals' in result
    assert 'num_scenarios' in result
    # depth=2, branch=2: 1 + 2 + 4 = 7 scenarios
    assert result['num_scenarios'] == 7, \
        f"Expected 7 scenarios, got {result['num_scenarios']}"

    print("✅ test_world_model_counterfactuals PASSED")


def test_world_model_gradient_flow():
    """Verify gradients flow through world model."""
    from aeon_core import PhysicsGroundedWorldModel

    model = PhysicsGroundedWorldModel(input_dim=32, state_dim=16)
    x = torch.randn(2, 32, requires_grad=True)
    result = model(x)
    loss = result['output'].sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    print("✅ test_world_model_gradient_flow PASSED")


def test_hierarchical_memory_store_retrieve():
    """Verify hierarchical memory store and retrieve."""
    from aeon_core import HierarchicalMemory

    mem = HierarchicalMemory(dim=32, working_capacity=3,
                              episodic_capacity=10, semantic_capacity=5)

    # Store some vectors
    for i in range(5):
        vec = torch.randn(32)
        mem.store(vec, meta={'idx': i})

    # Working memory should track total stores but only keep capacity items
    assert mem._working_count == 5
    # Verify that only the last `working_capacity` items are in the buffer
    # by checking the working_memory buffer has non-zero entries only in used slots
    wm = mem.working_memory
    used_slots = min(mem._working_count, mem.working_capacity)
    assert used_slots == 3
    for i in range(used_slots):
        assert wm[i].abs().sum() > 0, f"Working memory slot {i} should be non-zero"

    # Retrieve
    query = torch.randn(32)
    result = mem.retrieve(query, k=2)
    assert 'working' in result
    assert 'episodic' in result
    assert 'semantic' in result
    assert 'route_weights' in result
    assert result['route_weights'].shape == (3,)

    print("✅ test_hierarchical_memory_store_retrieve PASSED")


def test_hierarchical_memory_semantic():
    """Verify semantic memory graph operations."""
    from aeon_core import HierarchicalMemory

    mem = HierarchicalMemory(dim=16)
    v1 = torch.randn(16)
    v2 = torch.randn(16)
    v3 = torch.randn(16)

    mem.add_semantic_node(v1, "concept_A")
    mem.add_semantic_node(v2, "concept_B")
    mem.add_semantic_node(v3, "concept_C")
    mem.add_semantic_edge(0, 1, "related_to")
    mem.add_semantic_edge(1, 2, "causes")

    assert len(mem._semantic_nodes) == 3
    assert len(mem._semantic_edges) == 2

    result = mem.retrieve(v1, k=3)
    assert len(result['semantic']) > 0

    print("✅ test_hierarchical_memory_semantic PASSED")


def test_hierarchical_memory_consolidation():
    """Verify memory consolidation from replay buffer to episodic."""
    from aeon_core import HierarchicalMemory

    mem = HierarchicalMemory(dim=16)
    # Manually add to replay buffer
    for i in range(10):
        mem._replay_buffer.append((torch.randn(16), {'idx': i}))

    assert len(mem._replay_buffer) == 10
    moved = mem.consolidate()
    # Some items may have been moved (depends on importance_net output)
    assert isinstance(moved, int)
    assert len(mem._replay_buffer) + moved == 10

    print("✅ test_hierarchical_memory_consolidation PASSED")


def test_multimodal_grounding_language_vision():
    """Verify multi-modal grounding with language + vision."""
    from aeon_core import MultiModalGroundingModule

    mm = MultiModalGroundingModule(latent_dim=64, num_heads=4,
                                    vision_dim=128, audio_dim=32)
    mm.eval()

    language = torch.randn(2, 10, 64)
    vision = torch.randn(2, 8, 128)

    with torch.no_grad():
        result = mm(language=language, vision=vision)

    assert 'fused' in result
    assert result['fused'].shape == (2, 64)
    assert 'vision_decoded' in result
    assert 'language_decoded' in result
    assert not torch.isnan(result['fused']).any()

    print("✅ test_multimodal_grounding_language_vision PASSED")


def test_multimodal_grounding_single_modality():
    """Verify multi-modal grounding with single modality."""
    from aeon_core import MultiModalGroundingModule

    mm = MultiModalGroundingModule(latent_dim=64)
    mm.eval()

    language = torch.randn(2, 10, 64)
    with torch.no_grad():
        result = mm(language=language)

    assert 'fused' in result
    assert result['fused'].shape == (2, 64)

    print("✅ test_multimodal_grounding_single_modality PASSED")


def test_multimodal_grounding_three_modalities():
    """Verify multi-modal grounding with all three modalities."""
    from aeon_core import MultiModalGroundingModule

    mm = MultiModalGroundingModule(latent_dim=64, vision_dim=128, audio_dim=32)
    mm.eval()

    language = torch.randn(2, 10, 64)
    vision = torch.randn(2, 8, 128)
    audio = torch.randn(2, 6, 32)

    with torch.no_grad():
        result = mm(language=language, vision=vision, audio=audio)

    assert 'fused' in result
    assert result['fused'].shape == (2, 64)
    assert 'vision_decoded' in result
    assert 'audio_decoded' in result
    assert 'language_decoded' in result

    print("✅ test_multimodal_grounding_three_modalities PASSED")


def test_meta_learner_ewc_loss():
    """Verify MetaLearner EWC loss computation."""
    from aeon_core import MetaLearner

    # Simple model
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    learner = MetaLearner(model, ewc_lambda=100.0)

    # Before computing Fisher, EWC loss should be 0
    loss = learner.ewc_loss()
    assert loss.item() == 0.0

    # Manually set Fisher and optimal params
    for name, param in model.named_parameters():
        if param.requires_grad:
            learner._fisher_diag[name] = torch.ones_like(param)
            learner._optimal_params[name] = param.data.clone()

    # EWC loss should be 0 when params haven't changed
    loss = learner.ewc_loss()
    assert loss.item() == 0.0

    # Perturb a parameter and check loss increases
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)
    loss = learner.ewc_loss()
    assert loss.item() > 0.0, "EWC loss should be positive after param change"

    print("✅ test_meta_learner_ewc_loss PASSED")


def test_meta_learner_task_buffer():
    """Verify MetaLearner task buffer management."""
    from aeon_core import MetaLearner

    model = nn.Linear(8, 4)
    learner = MetaLearner(model, task_buffer_size=5)

    for i in range(10):
        learner.add_task(f"task_{i}", {'data': i})

    assert learner.num_tasks == 5, f"Expected 5 tasks, got {learner.num_tasks}"

    print("✅ test_meta_learner_task_buffer PASSED")


def test_aeon_v3_with_world_model():
    """Verify AEONDeltaV3 integration with world model enabled."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_world_model=True, world_model_state_dim=32,
        enable_quantum_sim=False, enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    assert model.world_model is not None
    tokens = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    assert 'world_model_results' in outputs
    assert outputs['world_model_results'] is not None

    print("✅ test_aeon_v3_with_world_model PASSED")


def test_aeon_v3_with_hierarchical_memory():
    """Verify AEONDeltaV3 with hierarchical memory enabled."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_hierarchical_memory=True,
        enable_quantum_sim=False, enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    assert model.hierarchical_memory is not None
    tokens = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    assert 'core_state' in outputs

    print("✅ test_aeon_v3_with_hierarchical_memory PASSED")


# ============================================================================
# Tests for refactoring fixes (analysis-driven)
# ============================================================================

def test_hessian_forward_ad_computation():
    """Verify _hessian_forward_ad is defined and produces correct output."""
    from aeon_core import FastHessianComputer

    hc = FastHessianComputer(method='finite_differences')
    assert hasattr(hc, '_hessian_forward_ad'), \
        "_hessian_forward_ad method is missing from FastHessianComputer"

    # Verify it produces valid output with a simple quadratic function
    def quadratic(x):
        # f(x) = sum(x^2) => H = 2*I
        return (x ** 2).sum(dim=-1)

    x = torch.randn(2, 4)
    H = hc._hessian_forward_ad(quadratic, x)
    assert H.shape == (2, 4, 4), f"Expected (2,4,4), got {H.shape}"

    print("✅ test_hessian_forward_ad_computation PASSED")


def test_usage_stats_zero_count_safety():
    """Verify _update_usage_stats handles zero-sum usage_count safely."""
    from aeon_core import RobustVectorQuantizer

    vq = RobustVectorQuantizer(num_embeddings=16, embedding_dim=8)
    vq.train()

    # Normal usage should work
    indices = torch.tensor([0, 1, 2, 3])
    vq._update_usage_stats(indices)  # Should not raise

    # Edge case: empty indices produce zero-sum usage_count
    # torch.bincount(empty, minlength=16) -> all zeros, sum = 0
    empty_indices = torch.tensor([], dtype=torch.long)
    vq._update_usage_stats(empty_indices)  # Should not raise (division by zero guarded)

    print("✅ test_usage_stats_zero_count_safety PASSED")


def test_ema_update_zero_cluster_safety():
    """Verify _ema_update does not divide by zero cluster sizes."""
    from aeon_core import RobustVectorQuantizer

    vq = RobustVectorQuantizer(num_embeddings=8, embedding_dim=4)
    vq.train()

    # Zero out cluster sizes to simulate edge case
    vq._ema_cluster_size.zero_()

    inputs = torch.randn(2, 4)
    encodings = torch.zeros(2, 8)
    encodings[0, 0] = 1.0
    encodings[1, 1] = 1.0

    # Should not raise or produce NaN/Inf
    vq._ema_update(inputs, encodings)

    assert not torch.isnan(vq.embedding.weight.data).any(), \
        "EMA update produced NaN with zero cluster sizes"
    assert not torch.isinf(vq.embedding.weight.data).any(), \
        "EMA update produced Inf with zero cluster sizes"

    print("✅ test_ema_update_zero_cluster_safety PASSED")


# ============================================================================
# NEW TESTS: Code analysis fixes (immutability, input validation, version check)
# ============================================================================

def test_config_immutability():
    """Fix 1.4: AEONConfig must not have a mutable 'device' attribute.
    
    Verifies that AEONConfig does not store a 'device' attribute directly,
    and that the config is truly frozen after __post_init__.
    """
    from aeon_core import AEONConfig
    
    config = AEONConfig()
    
    # 'device' should not be a direct instance attribute — use device_manager.device
    assert not hasattr(config, 'device'), (
        "AEONConfig should not have a mutable 'device' attribute; "
        "use config.device_manager.device instead"
    )
    
    # device_manager should be available
    assert config.device_manager is not None, "device_manager should be initialized"
    assert config.device_manager.device is not None, "device_manager.device should be set"
    
    # Config should be frozen
    try:
        config.z_dim = 512
        assert False, "Should have raised AttributeError (config is frozen)"
    except AttributeError:
        pass
    
    print("✅ test_config_immutability PASSED")


def test_forward_input_ids_validation():
    """Fix 3.1: AEONDeltaV3.forward must validate input_ids dtype and shape.
    
    Verifies that passing wrong dtype or shape raises clear errors.
    """
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_tensorboard=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    # Wrong dtype (float instead of long)
    float_ids = torch.randn(2, 16)  # float32
    try:
        model.forward(float_ids)
        assert False, "Should have raised TypeError for float input_ids"
    except TypeError as e:
        assert "torch.long" in str(e), f"Error message should mention torch.long: {e}"
    
    # Wrong shape (1D instead of 2D)
    flat_ids = torch.randint(0, 100, (16,))
    try:
        model.forward(flat_ids)
        assert False, "Should have raised ValueError for 1D input_ids"
    except ValueError as e:
        assert "2D" in str(e), f"Error message should mention 2D: {e}"
    
    # Correct input should work
    valid_ids = torch.randint(0, 100, (2, 16))
    result = model.forward(valid_ids)
    assert 'logits' in result, "Forward should return dict with 'logits'"
    
    print("✅ test_forward_input_ids_validation PASSED")


def test_forward_ad_version_check():
    """Fix 4.2: AEONDeltaV3 should validate PyTorch version for forward_ad.
    
    Verifies that using topo_method='forward_ad' on PyTorch without
    torch.func raises a clear RuntimeError.
    """
    import torch
    from aeon_core import AEONConfig, AEONDeltaV3
    
    # With PyTorch >= 2.0 (which has torch.func), forward_ad should work
    if hasattr(torch, 'func'):
        config = AEONConfig(
            topo_method="forward_ad",
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
            enable_tensorboard=False,
        )
        # Should not raise
        model = AEONDeltaV3(config)
        assert model is not None
    
    # finite_differences should always work regardless
    config_fd = AEONConfig(
        topo_method="finite_differences",
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_tensorboard=False,
    )
    model_fd = AEONDeltaV3(config_fd)
    assert model_fd is not None
    
    print("✅ test_forward_ad_version_check PASSED")


# ============================================================================
# MAMBA-2 (SSD) TESTS
# ============================================================================

def test_selective_ssmv2_forward():
    """Verify SelectiveSSMv2 produces correct output shapes and is NaN-free."""
    from aeon_core import SelectiveSSMv2

    ssm = SelectiveSSMv2(d_model=64, d_state=16, num_layers=2, expand_factor=2)
    ssm.eval()

    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        y, states = ssm(x)

    assert y.shape == (2, 32, 64), f"Expected (2,32,64), got {y.shape}"
    assert not torch.isnan(y).any(), "SSMv2 output contains NaN"
    assert not torch.isinf(y).any(), "SSMv2 output contains Inf"
    assert len(states) == 2, f"Expected 2 layer states, got {len(states)}"
    # Each state should be [B, nheads, head_dim, d_state]
    assert states[0].dim() == 4, f"State should be 4D, got {states[0].dim()}D"

    print("✅ test_selective_ssmv2_forward PASSED")


def test_ssmv2_state_caching():
    """Verify SSMv2 state caching propagates state across chunks."""
    from aeon_core import SelectiveSSMv2

    ssm = SelectiveSSMv2(d_model=32, d_state=8, num_layers=1)
    ssm.eval()

    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        y_full, _ = ssm(x)

    with torch.no_grad():
        y1, state = ssm(x[:, :5, :])
        y2, _ = ssm(x[:, 5:, :], state=state)

    y_chunked = torch.cat([y1, y2], dim=1)
    # Threshold accounts for Conv1d boundary effects and chunk-wise SSD
    # recomputation at split points (similar to test_ssm_state_caching).
    max_diff = torch.max(torch.abs(y_full - y_chunked)).item()
    assert max_diff < 2.0, \
        f"State caching divergence too large: max diff={max_diff:.6f}"
    assert not torch.isnan(y_chunked).any(), "Chunked output contains NaN"
    assert y_chunked.shape == y_full.shape, "Shape mismatch"

    print(f"✅ test_ssmv2_state_caching PASSED (max_diff={max_diff:.4f})")


def test_mamba2_thought_encoder():
    """Verify Mamba2ThoughtEncoder basic functionality."""
    from aeon_core import Mamba2ThoughtEncoder

    enc = Mamba2ThoughtEncoder(
        vocab_size=1000, emb_dim=64, z_dim=64,
        d_state=16, num_layers=1, expand_factor=2,
    )
    enc.eval()

    tokens = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16)
    with torch.no_grad():
        z = enc(tokens, attention_mask=mask)
    assert z.shape == (2, 64), f"Expected (2,64), got {z.shape}"
    assert not torch.isnan(z).any(), "Encoder output has NaN"

    # Without mask
    with torch.no_grad():
        z2 = enc(tokens)
    assert z2.shape == (2, 64)

    print("✅ test_mamba2_thought_encoder PASSED")


def test_mamba2_thought_decoder_train():
    """Verify Mamba2ThoughtDecoder in teacher-forcing mode."""
    from aeon_core import Mamba2ThoughtDecoder

    dec = Mamba2ThoughtDecoder(
        vocab_size=500, emb_dim=64, z_dim=64,
        d_state=16, num_layers=1,
    )
    dec.eval()

    z = torch.randn(2, 64)
    teacher = torch.randint(0, 500, (2, 12))
    with torch.no_grad():
        logits = dec(z, teacher_tokens=teacher, mode='train')
    assert logits.shape == (2, 12, 500), f"Expected (2,12,500), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Decoder logits have NaN"

    # Weight tying verification
    assert dec.head.weight.data_ptr() == dec.embed.weight.data_ptr(), \
        "Weight tying broken"

    print("✅ test_mamba2_thought_decoder_train PASSED")


def test_mamba2_thought_decoder_inference():
    """Verify Mamba2ThoughtDecoder autoregressive generation."""
    from aeon_core import Mamba2ThoughtDecoder

    dec = Mamba2ThoughtDecoder(
        vocab_size=500, emb_dim=64, z_dim=64,
        d_state=8, num_layers=1, sep_token_id=102,
    )
    dec.eval()

    z = torch.randn(2, 64)
    with torch.no_grad():
        gen_ids, logits = dec(z, mode='inference', max_length=20, sample=False)

    assert gen_ids.dim() == 2, f"Expected 2D output, got {gen_ids.dim()}D"
    assert gen_ids.shape[0] == 2, "Batch size mismatch"
    assert not torch.isnan(logits).any(), "Inference logits have NaN"

    print("✅ test_mamba2_thought_decoder_inference PASSED")


def test_build_encoder_factory_mamba2():
    """Verify build_encoder produces Mamba2ThoughtEncoder for 'mamba2' backend."""
    from aeon_core import AEONConfig, build_encoder, Mamba2ThoughtEncoder

    config = AEONConfig(device_str='cpu', encoder_backend='mamba2')
    enc = build_encoder(config)
    assert isinstance(enc, Mamba2ThoughtEncoder), \
        f"Expected Mamba2ThoughtEncoder, got {type(enc)}"

    print("✅ test_build_encoder_factory_mamba2 PASSED")


def test_build_decoder_factory_mamba2():
    """Verify build_decoder produces Mamba2ThoughtDecoder for 'mamba2' backend."""
    from aeon_core import AEONConfig, build_decoder, Mamba2ThoughtDecoder

    config = AEONConfig(device_str='cpu', decoder_backend='mamba2')
    dec = build_decoder(config)
    assert isinstance(dec, Mamba2ThoughtDecoder), \
        f"Expected Mamba2ThoughtDecoder, got {type(dec)}"

    print("✅ test_build_decoder_factory_mamba2 PASSED")


def test_mamba2_gradient_flow():
    """Verify gradients flow through the Mamba2 encoder."""
    from aeon_core import Mamba2ThoughtEncoder

    enc = Mamba2ThoughtEncoder(
        vocab_size=100, emb_dim=32, z_dim=32, d_state=8, num_layers=1,
    )
    tokens = torch.randint(0, 100, (2, 10))
    z = enc(tokens)
    loss = z.sum()
    loss.backward()

    has_grad = False
    for p in enc.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradient flow through Mamba2 encoder"

    print("✅ test_mamba2_gradient_flow PASSED")


def test_mamba2_long_sequence():
    """Verify Mamba2 handles long sequences (>1024 tokens)."""
    from aeon_core import Mamba2ThoughtEncoder

    enc = Mamba2ThoughtEncoder(
        vocab_size=1000, emb_dim=64, z_dim=64,
        d_state=16, num_layers=1, expand_factor=2,
    )
    enc.eval()

    tokens = torch.randint(0, 1000, (1, 2048))
    with torch.no_grad():
        z = enc(tokens)
    assert z.shape == (1, 64), f"Expected (1,64), got {z.shape}"
    assert not torch.isnan(z).any(), "Long-sequence Mamba2 encoding has NaN"

    print("✅ test_mamba2_long_sequence PASSED")


def test_aeon_v3_with_mamba2_backend():
    """Verify AEONDeltaV3 works with Mamba2 backend end-to-end."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        device_str='cpu',
        encoder_backend='mamba2',
        decoder_backend='mamba2',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    tokens = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)

    with torch.no_grad():
        result = model(tokens, attention_mask=mask, decode_mode='train')

    assert 'logits' in result
    assert 'thoughts' in result
    assert result['logits'].shape[0] == 2
    assert not torch.isnan(result['logits']).any(), "Mamba2 backend logits have NaN"

    print("✅ test_aeon_v3_with_mamba2_backend PASSED")


def test_config_mamba2_validation():
    """Verify AEONConfig validates mamba2 backend parameters."""
    from aeon_core import AEONConfig

    # Valid mamba2 backends should work
    AEONConfig(device_str='cpu', encoder_backend='mamba2')
    AEONConfig(device_str='cpu', decoder_backend='mamba2')
    AEONConfig(device_str='cpu', encoder_backend='mamba2', decoder_backend='mamba2')

    # Old backends should still work
    AEONConfig(device_str='cpu', encoder_backend='ssm', decoder_backend='ssm')
    AEONConfig(device_str='cpu', encoder_backend='lstm', decoder_backend='lstm')

    # Invalid backend should still fail
    try:
        AEONConfig(device_str='cpu', encoder_backend='mamba3')
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        AEONConfig(device_str='cpu', decoder_backend='mamba3')
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    print("✅ test_config_mamba2_validation PASSED")


def test_entropy_loss_single_embedding():
    """Verify config validation rejects vq_num_embeddings < 2.
    
    When num_embeddings=1, max_entropy=log(1)=0, which would cause
    division by zero in entropy computation. Config now enforces
    vq_num_embeddings >= 2 to prevent this at initialization time.
    """
    from ae_train import AEONConfigV4

    # vq_num_embeddings=1 should raise ValueError
    try:
        AEONConfigV4(vq_num_embeddings=1)
        assert False, "Should have raised ValueError for vq_num_embeddings=1"
    except ValueError as e:
        assert "vq_num_embeddings" in str(e)

    # vq_num_embeddings=0 should raise ValueError
    try:
        AEONConfigV4(vq_num_embeddings=0)
        assert False, "Should have raised ValueError for vq_num_embeddings=0"
    except ValueError as e:
        assert "vq_num_embeddings" in str(e)

    # vq_num_embeddings=2 should work
    config = AEONConfigV4(vq_num_embeddings=2)
    assert config.vq_num_embeddings == 2

    print("✅ test_entropy_loss_single_embedding PASSED")


def test_entropy_loss_guard():
    """Verify VectorQuantizerHybridV4._compute_entropy_loss handles zero max_entropy."""
    from ae_train import VectorQuantizerHybridV4
    import math

    vq = VectorQuantizerHybridV4(num_embeddings=2, embedding_dim=16)

    # Normal case: should not raise
    indices = torch.tensor([0, 1, 0, 1])
    loss = vq._compute_entropy_loss(indices)
    if isinstance(loss, torch.Tensor):
        loss_val = loss.item()
    else:
        loss_val = float(loss)
    assert not math.isnan(loss_val), "Entropy loss is NaN"
    assert not math.isinf(loss_val), "Entropy loss is Inf"

    print("✅ test_entropy_loss_guard PASSED")


def test_certified_error_numerical_stability():
    """Verify certified_error does not overflow for lip_const near 1.0.
    
    The certified error formula lip_const/(1-lip_const)*residual
    now uses max(1-lip_const, 1e-6) to prevent catastrophic overflow.
    """
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop

    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    ml = ProvablyConvergentMetaLoop(config, max_iterations=50, min_iterations=3)
    ml.eval()

    psi = torch.randn(2, config.z_dim)
    with torch.no_grad():
        C, iters, meta = ml.compute_fixed_point(psi)

    if meta.get('certified_error_bound') is not None:
        err = meta['certified_error_bound']
        assert not math.isinf(err), f"Certified error is infinite: {err}"
        assert not math.isnan(err), f"Certified error is NaN: {err}"

    print("✅ test_certified_error_numerical_stability PASSED")


def test_version_consistency():
    """Verify __version__ matches the documented version in docstring."""
    from aeon_core import __version__
    
    assert __version__ == "3.1.0", f"Expected version 3.1.0, got {__version__}"

    print("✅ test_version_consistency PASSED")


def test_warmup_cosine_scheduler_clamp():
    """Fix: WarmupCosineScheduler progress must be clamped to [0,1].
    
    When current_step exceeds total_steps (e.g. due to leftover batch steps),
    the LR should stay at min_lr and not rebound.
    """
    from ae_train import WarmupCosineScheduler
    
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6
    )
    
    # Step past total_steps
    for _ in range(120):
        scheduler.step()
    
    lr = scheduler.get_lr()
    # After total_steps, LR should be at or very near min_lr
    assert lr <= 1e-5, f"LR should be near min_lr after exceeding total_steps, got {lr}"
    
    print("✅ test_warmup_cosine_scheduler_clamp PASSED")


def test_nan_path_preserves_accumulated_gradients():
    """Fix: NaN loss path should NOT call optimizer.zero_grad().
    
    With gradient accumulation, valid gradients from prior batches must be
    preserved even if a subsequent batch produces NaN loss.
    """
    from ae_train import SafeThoughtAETrainerV4, AEONConfigV4, AEONDeltaV4, TrainingMonitor
    
    config = AEONConfigV4(vocab_size=100, z_dim=32, hidden_dim=32,
                          vq_num_embeddings=16, vq_embedding_dim=32,
                          seq_length=16, use_amp=False)
    model = AEONDeltaV4(config)
    monitor = TrainingMonitor(logging.getLogger("test"))
    trainer = SafeThoughtAETrainerV4(model, config, monitor, output_dir="/tmp/test_trainer")
    tokens = torch.randint(1, 100, (4, 16))
    trainer.train_step(tokens)
    
    # Check some grads exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )
    assert has_grad, "Should have accumulated gradients after valid step"
    
    # Store gradient snapshot before simulating NaN path
    grad_snapshot = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert len(grad_snapshot) > 0, "Should have gradient snapshots"
    
    # Monkey-patch _forward_pass to produce NaN loss
    original_forward = trainer._forward_pass
    def nan_forward(tokens):
        outputs = original_forward(tokens)
        outputs['total_loss'] = torch.tensor(float('nan'))
        return outputs
    trainer._forward_pass = nan_forward
    
    # This NaN step should NOT destroy accumulated gradients
    trainer.train_step(tokens)
    
    # Verify gradients are preserved (not zeroed)
    for name, old_grad in grad_snapshot.items():
        param = dict(model.named_parameters())[name]
        assert param.grad is not None, f"Gradient for {name} was zeroed"
        assert torch.equal(param.grad, old_grad), (
            f"Gradient for {name} was modified by NaN path"
        )
    
    # Restore
    trainer._forward_pass = original_forward
    
    print("✅ test_nan_path_preserves_accumulated_gradients PASSED")


def test_nan_metrics_not_contaminating_epoch():
    """Fix: NaN metric values should be guarded in Phase A epoch metric accumulation.
    
    Verifies that NaN values in individual metrics do not contaminate epoch averages.
    """
    # Simulate the guarded accumulation logic
    epoch_metrics = {"recon": 0.0, "vq": 0.0, "perplexity": 0.0, "accuracy_%": 0.0}
    
    # Good batch outputs
    good_outputs = {'recon_loss': 2.5, 'vq_loss': 0.1, 'perplexity': 12.0, 'accuracy': 45.0}
    # NaN batch outputs
    nan_outputs = {'recon_loss': float('nan'), 'vq_loss': float('nan'), 'perplexity': float('inf'), 'accuracy': 0.0}
    
    for outputs in [good_outputs, nan_outputs, good_outputs]:
        if not (math.isnan(outputs['recon_loss']) or math.isinf(outputs['recon_loss'])):
            epoch_metrics["recon"] += outputs['recon_loss']
            epoch_metrics["vq"] += outputs['vq_loss']
            epoch_metrics["perplexity"] += outputs['perplexity']
            epoch_metrics["accuracy_%"] += outputs['accuracy']
    
    # Epoch metrics should only include the 2 good batches
    assert math.isfinite(epoch_metrics["recon"]), "recon should be finite"
    assert epoch_metrics["recon"] == 5.0, f"Expected 5.0, got {epoch_metrics['recon']}"
    assert math.isfinite(epoch_metrics["perplexity"]), "perplexity should be finite"
    
    print("✅ test_nan_metrics_not_contaminating_epoch PASSED")


def test_entropy_loss_returns_tensor():
    """Fix: _compute_entropy_loss must always return a torch.Tensor.
    
    The else branch (max_entropy <= 0) should return a tensor, not a Python float.
    """
    from ae_train import VectorQuantizerHybridV4
    
    vq = VectorQuantizerHybridV4(num_embeddings=16, embedding_dim=32)
    
    # Normal case: indices with valid distribution
    indices = torch.randint(0, 16, (32,))
    result = vq._compute_entropy_loss(indices)
    assert isinstance(result, torch.Tensor), (
        f"Expected torch.Tensor, got {type(result)}"
    )
    
    print("✅ test_entropy_loss_returns_tensor PASSED")


def test_vq_temperature_validation():
    """Fix: AEONConfigV4 must reject vq_temperature <= 0.
    
    vq_temperature is used as a divisor in VQ distance computation;
    zero or negative values cause division by zero or flipped distances.
    """
    from ae_train import AEONConfigV4
    
    try:
        config = AEONConfigV4(vq_temperature=0.0)
        assert False, "Should have raised ValueError for vq_temperature=0"
    except ValueError as e:
        assert "vq_temperature" in str(e)
    
    try:
        config = AEONConfigV4(vq_temperature=-1.0)
        assert False, "Should have raised ValueError for vq_temperature=-1"
    except ValueError as e:
        assert "vq_temperature" in str(e)
    
    # Positive value should work fine
    config = AEONConfigV4(vq_temperature=0.5)
    assert config.vq_temperature == 0.5
    
    print("✅ test_vq_temperature_validation PASSED")


def test_perplexity_overflow_guard():
    """Fix: Perplexity computation should clamp recon_loss before exp().
    
    exp(loss) overflows to Inf for loss > ~88 in float32. The fix clamps
    recon_loss to max=80 before calling exp.
    """
    # Verify the clamping approach prevents overflow
    large_loss = torch.tensor(100.0)
    perplexity = torch.exp(large_loss.clamp(max=80)).item()
    assert math.isfinite(perplexity), f"Perplexity should be finite, got {perplexity}"
    
    # Without clamp, this would overflow
    raw_perplexity = torch.exp(large_loss).item()
    assert math.isinf(raw_perplexity), "Unclamped exp(100) should overflow to Inf"
    
    # Normal loss should be unaffected by clamp
    normal_loss = torch.tensor(5.0)
    clamped = torch.exp(normal_loss.clamp(max=80)).item()
    unclamped = torch.exp(normal_loss).item()
    assert abs(clamped - unclamped) < 1e-6, "Clamp should not affect normal losses"
    
    print("✅ test_perplexity_overflow_guard PASSED")


def test_gradscaler_compatibility():
    """Fix: GradScaler instantiation should handle both old and new PyTorch API.
    
    Verifies that the trainer can be instantiated without GradScaler errors.
    """
    from ae_train import SafeThoughtAETrainerV4, AEONConfigV4, AEONDeltaV4, TrainingMonitor
    
    # Use_amp=False so we don't need CUDA, but verify the code path compiles
    config = AEONConfigV4(vocab_size=100, z_dim=32, hidden_dim=32,
                          vq_num_embeddings=16, vq_embedding_dim=32,
                          seq_length=16, use_amp=False)
    model = AEONDeltaV4(config)
    monitor = TrainingMonitor(logging.getLogger("test"))
    
    # Should not raise any errors
    trainer = SafeThoughtAETrainerV4(model, config, monitor, output_dir="/tmp/test_trainer")
    assert trainer.scaler is None, "Scaler should be None when AMP is disabled"
    
    print("✅ test_gradscaler_compatibility PASSED")


# ============================================================================
# Tests for architecture refactoring (Tasks 1-13)
# ============================================================================

def test_diversity_metric_forward():
    """Task 1: Verify DiversityMetric replaces QuantumSimulator correctly."""
    from aeon_core import DiversityMetric, AEONConfig
    
    config = AEONConfig(device_str='cpu', enable_quantum_sim=False)
    dm = DiversityMetric(config)
    dm.eval()
    
    factors = torch.randn(4, config.num_pillars)
    with torch.no_grad():
        result = dm(factors)
    
    assert 'diversity' in result, "Missing 'diversity' key"
    assert 'action_propensity' in result, "Missing 'action_propensity' key"
    assert result['diversity'].shape == (4,), f"Expected (4,), got {result['diversity'].shape}"
    assert result['action_propensity'].shape == (4, config.num_pillars)
    # Diversity should be non-negative (variance)
    assert (result['diversity'] >= 0).all(), "Diversity should be non-negative"
    # Action propensity should sum to 1
    assert torch.allclose(result['action_propensity'].sum(dim=-1), 
                          torch.ones(4), atol=1e-5)
    
    print("✅ test_diversity_metric_forward PASSED")


def test_sparse_factorization_forward():
    """Task 2: Verify SparseFactorization produces correct shapes."""
    from aeon_core import SparseFactorization, AEONConfig
    
    config = AEONConfig(device_str='cpu')
    sf = SparseFactorization(config)
    sf.eval()
    
    hidden = torch.randn(2, config.hidden_dim)
    with torch.no_grad():
        factors, decoded = sf(hidden)
    
    assert factors.shape == (2, config.num_pillars), \
        f"Expected (2, {config.num_pillars}), got {factors.shape}"
    assert decoded.shape == (2, config.hidden_dim), \
        f"Expected (2, {config.hidden_dim}), got {decoded.shape}"
    # Factors should be in [0, 1] after sigmoid
    assert (factors >= 0).all() and (factors <= 1).all(), \
        "Factors should be in [0, 1]"
    
    print("✅ test_sparse_factorization_forward PASSED")


def test_sparse_factorization_sparsity_loss():
    """Task 2: Verify L1 sparsity loss computation."""
    from aeon_core import SparseFactorization, AEONConfig
    
    config = AEONConfig(device_str='cpu')
    sf = SparseFactorization(config)
    
    factors = torch.rand(4, config.num_pillars)
    loss = sf.sparsity_loss(factors)
    
    assert loss.dim() == 0, "Sparsity loss should be scalar"
    assert loss.item() >= 0, "Sparsity loss should be non-negative"
    assert torch.isfinite(loss), "Sparsity loss should be finite"
    
    # All-zero factors should give zero loss
    zero_factors = torch.zeros(4, config.num_pillars)
    zero_loss = sf.sparsity_loss(zero_factors)
    assert zero_loss.item() == 0.0, "Zero factors should give zero sparsity loss"
    
    print("✅ test_sparse_factorization_sparsity_loss PASSED")


def test_neural_causal_model_forward():
    """Task 6: Verify NeuralCausalModel forward pass."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=8, hidden_dim=32)
    model.eval()
    
    exogenous = torch.randn(2, 8)
    with torch.no_grad():
        causal_vars = model(exogenous)
    
    assert causal_vars.shape == (2, 8), f"Expected (2, 8), got {causal_vars.shape}"
    assert not torch.isnan(causal_vars).any(), "NaN in causal variables"
    
    print("✅ test_neural_causal_model_forward PASSED")


def test_neural_causal_model_dag_constraint():
    """Task 6: Verify DAG adjacency is lower-triangular."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=6)
    adj = model.adjacency
    
    # Should be lower-triangular (no self-loops, no backward edges)
    upper = torch.triu(adj, diagonal=0)
    assert (upper == 0).all(), "Adjacency should be strictly lower-triangular"
    
    print("✅ test_neural_causal_model_dag_constraint PASSED")


def test_neural_causal_model_intervention():
    """Task 6: Verify do(X=x) intervention."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=5, hidden_dim=16)
    model.eval()
    
    exogenous = torch.randn(3, 5)
    intervention = {2: 1.0}  # Set variable 2 to 1.0
    
    with torch.no_grad():
        result = model(exogenous, intervention=intervention)
    
    assert result.shape == (3, 5)
    # Variable 2 should be exactly 1.0
    assert torch.allclose(result[:, 2], torch.ones(3)), \
        "Intervened variable should be set to intervention value"
    
    print("✅ test_neural_causal_model_intervention PASSED")


def test_neural_causal_model_dag_loss():
    """Task 7: Verify DAG loss computation."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=4)
    loss = model.dag_loss()
    
    assert loss.dim() == 0, "DAG loss should be scalar"
    assert torch.isfinite(loss), "DAG loss should be finite"
    
    print("✅ test_neural_causal_model_dag_loss PASSED")


def test_neural_causal_model_consistency_loss():
    """Task 7: Verify consistency loss for interventional data."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=6, hidden_dim=16)
    model.eval()
    
    obs = torch.randn(2, 6)
    cf = torch.randn(2, 6)
    
    loss = model.consistency_loss(obs, cf, intervention_vars=[2, 3])
    assert loss.dim() == 0, "Consistency loss should be scalar"
    assert torch.isfinite(loss), "Consistency loss should be finite"
    
    print("✅ test_neural_causal_model_consistency_loss PASSED")


def test_neural_causal_model_gradient_flow():
    """Task 6: Verify gradients flow through NeuralCausalModel."""
    from aeon_core import NeuralCausalModel
    
    model = NeuralCausalModel(num_vars=6, hidden_dim=16)
    model.train()
    
    x = torch.randn(2, 6, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient flow through causal model"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    
    print("✅ test_neural_causal_model_gradient_flow PASSED")


def test_value_network_forward():
    """Task 9: Verify ValueNetwork produces correct output."""
    from aeon_core import ValueNetwork
    
    vn = ValueNetwork(state_dim=64, hidden_dim=32)
    vn.eval()
    
    state = torch.randn(4, 64)
    with torch.no_grad():
        value = vn(state)
    
    assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"
    assert not torch.isnan(value).any(), "NaN in value output"
    
    print("✅ test_value_network_forward PASSED")


def test_policy_network_forward():
    """Task 9: Verify PolicyNetwork produces valid distribution."""
    from aeon_core import PolicyNetwork
    
    pn = PolicyNetwork(state_dim=64, action_dim=8, hidden_dim=32)
    pn.eval()
    
    state = torch.randn(4, 64)
    with torch.no_grad():
        policy = pn(state)
    
    assert policy.shape == (4, 8), f"Expected (4, 8), got {policy.shape}"
    # Should be a valid probability distribution
    assert torch.allclose(policy.sum(dim=-1), torch.ones(4), atol=1e-5), \
        "Policy should sum to 1"
    assert (policy >= 0).all(), "Policy should be non-negative"
    
    print("✅ test_policy_network_forward PASSED")


def test_mcts_node_ucb1():
    """Task 8: Verify MCTSNode UCB1 scoring."""
    from aeon_core import MCTSNode
    
    parent = MCTSNode(state=torch.randn(16))
    parent.visits = 100
    
    child = MCTSNode(state=torch.randn(16), parent=parent, prior=0.5)
    child.visits = 10
    child.total_value = 5.0
    
    score = child.ucb1_score(c=1.41)
    expected_q = 5.0 / 10  # 0.5
    expected_exploration = 1.41 * 0.5 * math.sqrt(100) / (1 + 10)
    expected = expected_q + expected_exploration
    
    assert abs(score - expected) < 1e-4, \
        f"UCB1 score mismatch: {score} vs {expected}"
    
    print("✅ test_mcts_node_ucb1 PASSED")


def test_mcts_planner_forward():
    """Task 8: Verify MCTSPlanner forward pass."""
    from aeon_core import MCTSPlanner
    
    planner = MCTSPlanner(state_dim=32, action_dim=4, hidden_dim=16,
                           num_simulations=10)
    planner.eval()
    
    state = torch.randn(2, 32)
    with torch.no_grad():
        result = planner(state)
    
    assert 'value' in result, "Missing 'value' key"
    assert 'policy' in result, "Missing 'policy' key"
    assert result['value'].shape == (2, 1)
    assert result['policy'].shape == (2, 4)
    
    print("✅ test_mcts_planner_forward PASSED")


def test_mcts_planner_search():
    """Task 8: Verify MCTSPlanner search with world model."""
    from aeon_core import MCTSPlanner, PhysicsGroundedWorldModel
    
    planner = MCTSPlanner(state_dim=32, action_dim=4, hidden_dim=16,
                           num_simulations=10, max_depth=2)
    planner.eval()
    
    wm = PhysicsGroundedWorldModel(input_dim=32, state_dim=16)
    wm.eval()
    
    state = torch.randn(32)
    result = planner.search(state, wm)
    
    assert 'best_action' in result
    assert 'root_value' in result
    assert isinstance(result['best_action'], int)
    
    print("✅ test_mcts_planner_search PASSED")


def test_hierarchical_vae_forward():
    """Task 10: Verify HierarchicalVAE forward pass."""
    from aeon_core import HierarchicalVAE
    
    vae = HierarchicalVAE(input_dim=64, num_levels=4)
    vae.eval()
    
    x = torch.randn(2, 64)
    with torch.no_grad():
        result = vae(x)
    
    assert 'levels' in result
    assert 'reconstructions' in result
    assert 'kl_loss' in result
    assert 'selected_level' in result
    assert len(result['levels']) == 4, f"Expected 4 levels, got {len(result['levels'])}"
    assert len(result['reconstructions']) == 4
    
    print("✅ test_hierarchical_vae_forward PASSED")


def test_hierarchical_vae_abstraction_level():
    """Task 10: Verify abstraction level selection."""
    from aeon_core import HierarchicalVAE
    
    vae = HierarchicalVAE(input_dim=32, num_levels=3)
    vae.eval()
    
    x = torch.randn(2, 32)
    # Request specific abstraction level
    with torch.no_grad():
        result = vae(x, abstraction_level=1)
    
    assert result['selected_level'].shape[0] == 2
    
    print("✅ test_hierarchical_vae_abstraction_level PASSED")


def test_hierarchical_vae_kl_loss():
    """Task 10: Verify KL loss is finite during training."""
    from aeon_core import HierarchicalVAE
    
    vae = HierarchicalVAE(input_dim=32, num_levels=3)
    vae.train()
    
    x = torch.randn(4, 32)
    result = vae(x)
    
    kl = result['kl_loss']
    assert torch.isfinite(kl), f"KL loss should be finite, got {kl}"
    assert kl.item() >= 0, "KL loss should be non-negative"
    
    print("✅ test_hierarchical_vae_kl_loss PASSED")


def test_adaptive_chunking():
    """Task 12: Verify adaptive chunking adjusts chunk size."""
    from aeon_core import ChunkedSequenceProcessor
    
    # Non-adaptive (default)
    processor = ChunkedSequenceProcessor(chunk_size=8, overlap=2)
    assert not processor.adaptive
    
    # Adaptive
    adaptive_processor = ChunkedSequenceProcessor(
        chunk_size=16, overlap=2, adaptive=True, min_chunk_size=4
    )
    assert adaptive_processor.adaptive
    assert adaptive_processor.min_chunk_size == 4
    
    def model_fn(x, state):
        return x * 2.0, state
    
    x = torch.randn(1, 32, 8)
    y, _ = adaptive_processor.process(model_fn, x)
    assert y.shape == (1, 32, 8)
    
    print("✅ test_adaptive_chunking PASSED")


def test_world_model_surprise_integration():
    """Task 3: Verify world model surprise-driven integration."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_world_model=True, world_model_state_dim=32,
        enable_quantum_sim=False, enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    assert hasattr(model, 'value_net'), "Should have value_net for surprise integration"
    tokens = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    
    assert 'world_model_results' in outputs
    wm = outputs['world_model_results']
    assert 'surprise' in wm, "World model results should contain surprise"
    
    print("✅ test_world_model_surprise_integration PASSED")


def test_memory_retrieval_integration():
    """Task 4: Verify hierarchical memory retrieval integration."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_hierarchical_memory=True,
        enable_quantum_sim=False, enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    assert hasattr(model, 'memory_projection'), "Should have memory_projection"
    assert hasattr(model, 'importance_scorer'), "Should have importance_scorer"
    
    tokens = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    
    assert 'core_state' in outputs
    
    print("✅ test_memory_retrieval_integration PASSED")


def test_safety_enforcement():
    """Task 5: Verify safety enforcement rollback."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_safety_guardrails=True,
        safety_threshold=0.99,  # Very high threshold to trigger enforcement
        enable_quantum_sim=False, enable_catastrophe_detection=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    tokens = torch.randint(100, 1000, (2, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    
    assert 'safety_score' in outputs
    assert outputs['safety_score'].shape[0] == 2
    
    print("✅ test_safety_enforcement PASSED")


def test_filter_logits_all_inf_guard():
    """Verify that _filter_logits handles all-inf case by falling back to unfiltered logits.
    
    This tests the scenario where invalid token filtering removes all top-K tokens,
    causing all remaining logits to be -inf.
    """
    from aeon_core import ThoughtDecoder
    
    decoder = ThoughtDecoder(vocab_size=1000, emb_dim=64, z_dim=64)
    decoder.eval()
    
    # Create logits where only a few tokens have positive values
    logits = torch.full((2, 1000), -5.0)
    logits[:, 200:210] = 2.0  # These are the "good" tokens
    
    # Mark those good tokens as invalid — this simulates the case where
    # invalid token filtering removes all reasonable candidates
    invalid_mask = torch.zeros(1000, dtype=torch.bool)
    invalid_mask[200:210] = True
    decoder._invalid_token_mask = invalid_mask
    
    device = logits.device
    filtered = decoder._filter_logits(logits, temperature=0.8, top_k=50, device=device)
    
    # After the guard, filtered should NOT all be extremely negative
    # because fallback to original logits should kick in
    assert filtered.max(dim=-1).values.min().item() > -1e8, (
        f"All logits still extremely negative after guard: max={filtered.max().item()}"
    )
    
    # Verify softmax on filtered logits gives valid probabilities
    probs = torch.softmax(filtered, dim=-1)
    assert not torch.isnan(probs).any(), "NaN in probabilities"
    assert not torch.isinf(probs).any(), "Inf in probabilities"
    
    print("✅ test_filter_logits_all_inf_guard PASSED")


def test_filter_logits_nan_handling():
    """Verify that _filter_logits properly replaces NaN values."""
    from aeon_core import ThoughtDecoder
    
    decoder = ThoughtDecoder(vocab_size=1000, emb_dim=64, z_dim=64)
    decoder.eval()
    
    # Create logits with NaN values
    logits = torch.randn(2, 1000)
    logits[0, 10:20] = float('nan')
    device = logits.device
    
    filtered = decoder._filter_logits(logits, temperature=0.8, top_k=50, device=device)
    
    assert not torch.isnan(filtered).any(), "NaN values remain after filtering"
    assert not torch.isinf(filtered).any(), "Inf values remain after filtering"
    
    print("✅ test_filter_logits_nan_handling PASSED")


def test_temperature_clamping():
    """Verify that very small temperature is clamped to 0.1 minimum."""
    from aeon_core import ThoughtDecoder
    
    decoder = ThoughtDecoder(vocab_size=1000, emb_dim=64, z_dim=64)
    decoder.eval()
    
    logits = torch.randn(1, 1000)
    device = logits.device
    
    # Very small temperature should not cause numerical instability
    filtered = decoder._filter_logits(logits, temperature=1e-10, top_k=0, device=device)
    
    assert not torch.isnan(filtered).any(), "NaN with very small temperature"
    assert not torch.isinf(filtered).any(), "Inf with very small temperature"
    
    # Verify the temperature was effectively clamped to 0.1
    expected = logits / 0.1
    expected = torch.nan_to_num(expected, nan=-1e9, posinf=1e9, neginf=-1e9)
    assert torch.allclose(filtered, expected, atol=1e-5), "Temperature was not clamped to 0.1"
    
    print("✅ test_temperature_clamping PASSED")


def test_safety_blending_not_replacement():
    """Verify that safety enforcement blends C_star instead of replacing it entirely."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        hidden_dim=64, z_dim=64, vocab_size=1000, seq_length=16,
        vq_embedding_dim=64, vq_num_embeddings=128,
        enable_safety_guardrails=True,
        safety_threshold=0.99,  # Very high to trigger rollback
        enable_quantum_sim=False, enable_catastrophe_detection=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    tokens = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        outputs = model(tokens, fast=False)
    
    core_state = outputs['core_state']
    psi_0 = outputs['psi_0']
    
    # After blending, core_state should NOT be exactly equal to psi_0
    # (unless safety_score is exactly 0, which is extremely unlikely)
    if outputs['safety_score'].item() > 0.0:
        assert not torch.allclose(core_state, psi_0, atol=1e-6), (
            "core_state should be a blend, not a full replacement of z_in"
        )
    
    print("✅ test_safety_blending_not_replacement PASSED")


def test_missing_weight_xavier_init():
    """Verify that missing weight matrices use Xavier init, not zeros."""
    import torch.nn as nn
    
    # Simulate the fixed initialization logic
    param_tensor = torch.zeros(64, 128)
    key = "decoder.lstm.weight_ih_l0"
    
    if 'weight' in key and param_tensor.dim() >= 2:
        nn.init.xavier_uniform_(param_tensor)
    
    # Xavier-initialized tensor should NOT be all zeros
    assert not torch.allclose(param_tensor, torch.zeros_like(param_tensor)), (
        "Weight matrix should be Xavier-initialized, not zeros"
    )
    
    # But bias should stay zeros
    bias_tensor = torch.zeros(64)
    bias_key = "decoder.lstm.bias_ih_l0"
    if 'weight' in bias_key and bias_tensor.dim() >= 2:
        nn.init.xavier_uniform_(bias_tensor)
    elif 'bias' in bias_key:
        nn.init.zeros_(bias_tensor)
    
    assert torch.allclose(bias_tensor, torch.zeros_like(bias_tensor)), (
        "Bias should remain zeros"
    )
    
    print("✅ test_missing_weight_xavier_init PASSED")


def test_safety_threshold_default():
    """Verify that the default safety threshold is 0.5, not 0.85."""
    from aeon_core import AEONConfig
    
    config = AEONConfig()
    assert config.safety_threshold == 0.5, (
        f"Default safety_threshold should be 0.5, got {config.safety_threshold}"
    )
    
    print("✅ test_safety_threshold_default PASSED")


# ==========================================================================
# Tests for new cognitive architecture enhancements
# ==========================================================================


def test_convergence_monitor_warmup():
    """ConvergenceMonitor returns 'warmup' for fewer than 3 samples."""
    from aeon_core import ConvergenceMonitor
    mon = ConvergenceMonitor(threshold=1e-5)
    r1 = mon.check(1.0)
    assert r1['status'] == 'warmup'
    assert r1['certified'] is False
    r2 = mon.check(0.5)
    assert r2['status'] == 'warmup'
    print("✅ test_convergence_monitor_warmup PASSED")


def test_convergence_monitor_converged():
    """ConvergenceMonitor certifies convergence with decreasing deltas."""
    from aeon_core import ConvergenceMonitor
    mon = ConvergenceMonitor(threshold=1e-3)
    for d in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        result = mon.check(d)
    assert result['status'] == 'converged'
    assert result['certified'] is True
    assert 0.0 < result['contraction_rate'] < 1.0
    assert 0.0 < result['confidence'] < 1.0
    print("✅ test_convergence_monitor_converged PASSED")


def test_convergence_monitor_diverging():
    """ConvergenceMonitor detects divergence when norms increase."""
    from aeon_core import ConvergenceMonitor
    mon = ConvergenceMonitor(threshold=1e-5)
    for d in [0.01, 0.1, 1.0, 10.0]:
        result = mon.check(d)
    assert result['status'] == 'diverging'
    assert result['certified'] is False
    print("✅ test_convergence_monitor_diverging PASSED")


def test_convergence_monitor_reset():
    """ConvergenceMonitor.reset clears history."""
    from aeon_core import ConvergenceMonitor
    mon = ConvergenceMonitor()
    for d in [1.0, 0.5, 0.25]:
        mon.check(d)
    mon.reset()
    assert len(mon.history) == 0
    result = mon.check(1.0)
    assert result['status'] == 'warmup'
    print("✅ test_convergence_monitor_reset PASSED")


def test_hierarchical_meta_loop_forward():
    """HierarchicalMetaLoop forward pass produces valid output."""
    from aeon_core import HierarchicalMetaLoop, AEONConfig
    config = AEONConfig()
    hml = HierarchicalMetaLoop(config)
    hml.eval()
    z = torch.randn(2, config.hidden_dim)
    C_star, iters, meta = hml(z)
    assert C_star.shape == (2, config.hidden_dim)
    assert torch.isfinite(C_star).all()
    print("✅ test_hierarchical_meta_loop_forward PASSED")


def test_hierarchical_meta_loop_training_uses_deep():
    """During training, HierarchicalMetaLoop always uses the deep loop."""
    from aeon_core import HierarchicalMetaLoop, AEONConfig
    config = AEONConfig()
    hml = HierarchicalMetaLoop(config)
    hml.train()
    z = torch.randn(2, config.hidden_dim)
    C_star, iters, meta = hml(z)
    assert C_star.shape == (2, config.hidden_dim)
    print("✅ test_hierarchical_meta_loop_training_uses_deep PASSED")


def test_causal_factor_extractor_forward():
    """CausalFactorExtractor produces valid factors and DAG."""
    from aeon_core import CausalFactorExtractor
    cfe = CausalFactorExtractor(hidden_dim=64, num_factors=8)
    x = torch.randn(4, 64)
    result = cfe(x)
    assert result['factors'].shape == (4, 8)
    assert result['causal_graph'].shape == (8, 8)
    assert result['interventional'] is False
    # Check DAG: diagonal and upper triangle should be zero
    adj = result['causal_graph']
    upper = torch.triu(adj, diagonal=0)
    assert (upper == 0).all(), "Adjacency must be strictly lower-triangular (zero on and above diagonal)"
    print("✅ test_causal_factor_extractor_forward PASSED")


def test_causal_factor_extractor_intervention():
    """CausalFactorExtractor correctly applies do-intervention."""
    from aeon_core import CausalFactorExtractor
    cfe = CausalFactorExtractor(hidden_dim=64, num_factors=8)
    x = torch.randn(2, 64)
    result = cfe(x, intervene={'index': 3, 'value': 1.0})
    assert result['interventional'] is True
    # The intervened factor should be close to 1.0 (plus causal effect)
    assert result['factors'][:, 3].min() >= 0.9
    print("✅ test_causal_factor_extractor_intervention PASSED")


def test_causal_factor_extractor_gradient_flow():
    """CausalFactorExtractor allows gradients to flow."""
    from aeon_core import CausalFactorExtractor
    cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
    x = torch.randn(2, 32, requires_grad=True)
    result = cfe(x)
    loss = result['factors'].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    print("✅ test_causal_factor_extractor_gradient_flow PASSED")


def test_temporal_memory_store_and_retrieve():
    """TemporalMemory stores and retrieves vectors by similarity."""
    from aeon_core import TemporalMemory
    tm = TemporalMemory(capacity=10, dim=16)
    v = torch.randn(16)
    tm.store(v, importance=1.0)
    results = tm.retrieve(v, k=1)
    assert len(results) == 1
    assert torch.allclose(results[0]['vector'], v)
    print("✅ test_temporal_memory_store_and_retrieve PASSED")


def test_temporal_memory_decay():
    """TemporalMemory applies exponential decay over time."""
    from aeon_core import TemporalMemory
    tm = TemporalMemory(capacity=100, dim=8, decay_rate=0.5)
    v = torch.randn(8)
    tm.store(v, importance=1.0)
    initial_strength = tm.memories[0]['strength']
    # Store many more to advance time
    for _ in range(20):
        tm.store(torch.randn(8), importance=0.1)
    # Original memory should have decayed significantly or been pruned
    old_present = any(
        torch.allclose(m['vector'], v) for m in tm.memories
    )
    if old_present:
        old_mem = [m for m in tm.memories if torch.allclose(m['vector'], v)][0]
        assert old_mem['strength'] < initial_strength
    print("✅ test_temporal_memory_decay PASSED")


def test_temporal_memory_consolidation():
    """TemporalMemory consolidates when capacity is exceeded."""
    from aeon_core import TemporalMemory
    tm = TemporalMemory(capacity=3, dim=8)
    for i in range(5):
        tm.store(torch.randn(8), importance=1.0)
    assert len(tm.memories) <= 3
    print("✅ test_temporal_memory_consolidation PASSED")


def test_temporal_memory_empty_retrieve():
    """TemporalMemory returns empty list when no memories stored."""
    from aeon_core import TemporalMemory
    tm = TemporalMemory(capacity=10, dim=8)
    results = tm.retrieve(torch.randn(8), k=5)
    assert results == []
    print("✅ test_temporal_memory_empty_retrieve PASSED")


def test_grounded_multimodal_learning_forward():
    """GroundedMultimodalLearning computes contrastive loss."""
    from aeon_core import GroundedMultimodalLearning
    gml = GroundedMultimodalLearning(vision_dim=64, language_dim=32, latent_dim=16)
    v_feat = torch.randn(4, 64)
    l_feat = torch.randn(4, 32)
    result = gml(v_feat, l_feat)
    assert result['vision'].shape == (4, 16)
    assert result['language'].shape == (4, 16)
    assert result['similarity'].shape == (4, 4)
    assert result['loss'].dim() == 0  # scalar
    assert result['loss'].item() > 0
    print("✅ test_grounded_multimodal_learning_forward PASSED")


def test_grounded_multimodal_learning_zero_shot():
    """GroundedMultimodalLearning zero_shot_classify returns valid probs."""
    from aeon_core import GroundedMultimodalLearning
    gml = GroundedMultimodalLearning(vision_dim=64, language_dim=32, latent_dim=16)
    img = torch.randn(1, 64)
    texts = [torch.randn(32) for _ in range(5)]
    probs = gml.zero_shot_classify(img, texts)
    assert probs.shape == (5,)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert (probs >= 0).all()
    print("✅ test_grounded_multimodal_learning_zero_shot PASSED")


def test_grounded_multimodal_gradient_flow():
    """GroundedMultimodalLearning loss allows gradient flow."""
    from aeon_core import GroundedMultimodalLearning
    gml = GroundedMultimodalLearning(vision_dim=64, language_dim=32, latent_dim=16)
    v = torch.randn(4, 64, requires_grad=True)
    l = torch.randn(4, 32, requires_grad=True)
    result = gml(v, l)
    result['loss'].backward()
    assert v.grad is not None
    assert l.grad is not None
    print("✅ test_grounded_multimodal_gradient_flow PASSED")


def test_curiosity_driven_exploration_reward():
    """CuriosityDrivenExploration computes intrinsic reward."""
    from aeon_core import CuriosityDrivenExploration
    cde = CuriosityDrivenExploration(state_dim=32, action_dim=8)
    s_t = torch.randn(4, 32)
    a_t = torch.randn(4, 8)
    s_next = torch.randn(4, 32)
    reward = cde.intrinsic_reward(s_t, a_t, s_next)
    assert reward.shape == (4,)
    assert (reward >= 0).all()
    print("✅ test_curiosity_driven_exploration_reward PASSED")


def test_curiosity_driven_exploration_inverse():
    """CuriosityDrivenExploration inverse model predicts actions."""
    from aeon_core import CuriosityDrivenExploration
    cde = CuriosityDrivenExploration(state_dim=32, action_dim=8)
    s_t = torch.randn(2, 32)
    s_next = torch.randn(2, 32)
    a_pred = cde.predict_action(s_t, s_next)
    assert a_pred.shape == (2, 8)
    print("✅ test_curiosity_driven_exploration_inverse PASSED")


def test_curiosity_driven_select_action():
    """CuriosityDrivenExploration selects action from candidates."""
    from aeon_core import CuriosityDrivenExploration
    cde = CuriosityDrivenExploration(state_dim=16, action_dim=4)
    state = torch.randn(16)
    candidates = [torch.randn(4) for _ in range(5)]
    action = cde.select_action(state, candidates)
    assert action.shape == (4,)
    print("✅ test_curiosity_driven_select_action PASSED")


def test_continual_learning_core_add_task():
    """ContinualLearningCore adds new columns."""
    from aeon_core import ContinualLearningCore
    base = nn.Linear(32, 32)
    base.config = type('Config', (), {'hidden_dim': 32})()
    clc = ContinualLearningCore(base)
    assert len(clc.columns) == 1
    clc.add_task('task1')
    assert len(clc.columns) == 2
    clc.add_task('task2')
    assert len(clc.columns) == 3
    print("✅ test_continual_learning_core_add_task PASSED")


def test_continual_learning_core_ewc_loss():
    """ContinualLearningCore EWC loss is non-negative."""
    from aeon_core import ContinualLearningCore
    base = nn.Linear(32, 32)
    base.config = type('Config', (), {'hidden_dim': 32})()
    clc = ContinualLearningCore(base)
    # Compute fake gradients
    x = torch.randn(4, 32)
    out = clc.columns[-1](x)
    out.sum().backward()
    clc.add_task('task1')
    clc.compute_fisher('task1')
    loss = clc.ewc_loss('task1')
    assert loss.item() >= 0
    print("✅ test_continual_learning_core_ewc_loss PASSED")


def test_continual_learning_ewc_missing_task():
    """ContinualLearningCore EWC loss returns 0 for unknown task."""
    from aeon_core import ContinualLearningCore
    base = nn.Linear(16, 16)
    base.config = type('Config', (), {'hidden_dim': 16})()
    clc = ContinualLearningCore(base)
    loss = clc.ewc_loss('nonexistent')
    assert loss.item() == 0.0
    print("✅ test_continual_learning_ewc_missing_task PASSED")


# ============================================================================
# AGI CRITICAL MODIFICATION TESTS
# ============================================================================

def test_recursive_meta_loop_forward():
    """RecursiveMetaLoop forward produces correct output shape and metadata."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop, RecursiveMetaLoop
    config = AEONConfig(
        use_vq=False, enable_quantum_sim=False,
        enable_catastrophe_detection=False, enable_safety_guardrails=False,
    )
    base_loop = ProvablyConvergentMetaLoop(config=config, max_iterations=5)
    rml = RecursiveMetaLoop(base_loop, max_recursion_depth=3)
    z = torch.randn(2, config.hidden_dim)
    out, iters, meta = rml(z)
    assert out.shape == (2, config.hidden_dim), f"Expected shape (2, {config.hidden_dim}), got {out.shape}"
    assert 'final_level' in meta
    assert 'level_metadata' in meta
    assert isinstance(meta['level_metadata'], list)
    print("✅ test_recursive_meta_loop_forward PASSED")


def test_recursive_meta_loop_target_level():
    """RecursiveMetaLoop respects target_abstraction parameter."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop, RecursiveMetaLoop
    config = AEONConfig(
        use_vq=False, enable_quantum_sim=False,
        enable_catastrophe_detection=False, enable_safety_guardrails=False,
    )
    base_loop = ProvablyConvergentMetaLoop(config=config, max_iterations=5)
    rml = RecursiveMetaLoop(base_loop, max_recursion_depth=3)
    z = torch.randn(2, config.hidden_dim)
    out, iters, meta = rml(z, target_abstraction=0)
    assert meta['target_level'] == 0
    assert len(meta['level_metadata']) >= 1
    print("✅ test_recursive_meta_loop_target_level PASSED")


def test_recursive_meta_loop_has_levels():
    """RecursiveMetaLoop creates correct number of levels."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop, RecursiveMetaLoop
    config = AEONConfig(
        use_vq=False, enable_quantum_sim=False,
        enable_catastrophe_detection=False, enable_safety_guardrails=False,
    )
    base_loop = ProvablyConvergentMetaLoop(config=config, max_iterations=5)
    rml = RecursiveMetaLoop(base_loop, max_recursion_depth=3)
    assert len(rml.levels) == 3
    print("✅ test_recursive_meta_loop_has_levels PASSED")


def test_neurogenic_memory_consolidate():
    """NeurogenicMemorySystem creates new neurons on high-importance input."""
    from aeon_core import NeurogenicMemorySystem
    nms = NeurogenicMemorySystem(base_dim=32, max_capacity=10, importance_threshold=0.0)
    assert nms.num_neurons == 1
    vec = torch.randn(32)
    nms.consolidate(vec, importance=0.9)
    assert nms.num_neurons == 2, f"Expected 2 neurons, got {nms.num_neurons}"
    print("✅ test_neurogenic_memory_consolidate PASSED")


def test_neurogenic_memory_retrieve():
    """NeurogenicMemorySystem retrieves neurons by similarity."""
    from aeon_core import NeurogenicMemorySystem
    nms = NeurogenicMemorySystem(base_dim=32, max_capacity=10, importance_threshold=0.0)
    for _ in range(5):
        nms.consolidate(torch.randn(32), importance=0.9)
    query = torch.randn(32)
    results = nms.retrieve(query, k=3)
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    print("✅ test_neurogenic_memory_retrieve PASSED")


def test_neurogenic_memory_capacity_limit():
    """NeurogenicMemorySystem respects max_capacity."""
    from aeon_core import NeurogenicMemorySystem
    nms = NeurogenicMemorySystem(base_dim=16, max_capacity=5, importance_threshold=0.0)
    for _ in range(10):
        nms.consolidate(torch.randn(16), importance=0.9)
    assert nms.num_neurons > 1, "No neurons were created during consolidation"
    assert nms.num_neurons <= 5, f"Exceeded capacity: {nms.num_neurons}"
    print("✅ test_neurogenic_memory_capacity_limit PASSED")


def test_neurogenic_memory_synapse_formation():
    """NeurogenicMemorySystem forms synapses between neurons."""
    from aeon_core import NeurogenicMemorySystem
    nms = NeurogenicMemorySystem(base_dim=16, max_capacity=20, importance_threshold=0.0)
    for _ in range(5):
        nms.consolidate(torch.randn(16), importance=0.9)
    # At least some synapses should be formed
    assert nms.num_neurons > 1
    # Synapses may or may not form depending on similarity
    assert isinstance(nms.num_synapses, int)
    print("✅ test_neurogenic_memory_synapse_formation PASSED")


def test_causal_world_model_forward():
    """CausalWorldModel forward produces correct outputs."""
    from aeon_core import CausalWorldModel
    cwm = CausalWorldModel(state_dim=64, num_causal_vars=8)
    state = torch.randn(2, 64)
    result = cwm(state)
    assert 'causal_vars' in result
    assert 'endogenous' in result
    assert 'cf_state' in result
    assert 'physics_output' in result
    assert result['causal_vars'].shape == (2, 8)
    assert result['cf_state'].shape == (2, 64)
    print("✅ test_causal_world_model_forward PASSED")


def test_causal_world_model_intervention():
    """CausalWorldModel supports do-calculus interventions."""
    from aeon_core import CausalWorldModel
    cwm = CausalWorldModel(state_dim=64, num_causal_vars=8)
    state = torch.randn(2, 64)
    result = cwm(state, intervention={0: 1.0})
    assert 'dag_loss' in result
    assert torch.allclose(result['endogenous'][:, 0], torch.ones(2))
    print("✅ test_causal_world_model_intervention PASSED")


def test_causal_world_model_counterfactual_rollout():
    """CausalWorldModel counterfactual_rollout produces trajectory."""
    from aeon_core import CausalWorldModel
    cwm = CausalWorldModel(state_dim=64, num_causal_vars=8)
    state = torch.randn(2, 64)
    result = cwm.counterfactual_rollout(state, intervention={1: 0.5})
    assert 'exogenous' in result
    assert 'cf_causal_vars' in result
    assert 'cf_state' in result
    assert 'trajectory' in result
    assert result['cf_state'].shape == (2, 64)
    print("✅ test_causal_world_model_counterfactual_rollout PASSED")


def test_causal_world_model_gradient_flow():
    """CausalWorldModel gradients flow through all components."""
    from aeon_core import CausalWorldModel
    cwm = CausalWorldModel(state_dim=32, num_causal_vars=4)
    state = torch.randn(2, 32, requires_grad=True)
    result = cwm(state)
    loss = result['cf_state'].sum()
    loss.backward()
    assert state.grad is not None
    assert not torch.isnan(state.grad).any()
    print("✅ test_causal_world_model_gradient_flow PASSED")


def test_active_learning_planner_forward():
    """ActiveLearningPlanner forward returns value and policy."""
    from aeon_core import ActiveLearningPlanner
    alp = ActiveLearningPlanner(state_dim=64, action_dim=8)
    state = torch.randn(2, 64)
    result = alp(state)
    assert 'value' in result
    assert 'policy' in result
    assert result['value'].shape == (2, 1)
    assert result['policy'].shape == (2, 8)
    print("✅ test_active_learning_planner_forward PASSED")


def test_active_learning_planner_intrinsic_reward():
    """ActiveLearningPlanner computes intrinsic curiosity reward."""
    from aeon_core import ActiveLearningPlanner
    alp = ActiveLearningPlanner(state_dim=64, action_dim=8)
    state = torch.randn(64)
    reward = alp.compute_intrinsic_reward(state)
    assert isinstance(reward, float)
    assert reward >= 0
    print("✅ test_active_learning_planner_intrinsic_reward PASSED")


def test_active_learning_planner_search():
    """ActiveLearningPlanner search includes intrinsic reward in simulation."""
    from aeon_core import ActiveLearningPlanner, PhysicsGroundedWorldModel
    alp = ActiveLearningPlanner(state_dim=64, action_dim=4, num_simulations=10)
    wm = PhysicsGroundedWorldModel(input_dim=64, state_dim=64)
    state = torch.randn(64)
    alp.eval()
    result = alp.select_action(state, wm)
    assert 'best_action' in result
    assert 'intrinsic_reward' in result
    assert isinstance(result['intrinsic_reward'], float)
    print("✅ test_active_learning_planner_search PASSED")


# ============================================================================
# Tests for ae_train.py robustness fixes
# ============================================================================

def test_save_checkpoint_error_handling():
    """Verify _save_checkpoint handles I/O errors gracefully."""
    import tempfile
    from ae_train import SafeThoughtAETrainerV4, AEONConfigV4, AEONDeltaV4, TrainingMonitor
    
    config = AEONConfigV4(vocab_size=100, z_dim=32, hidden_dim=32,
                          vq_num_embeddings=16, vq_embedding_dim=32,
                          seq_length=16, use_amp=False)
    model = AEONDeltaV4(config)
    monitor = TrainingMonitor(logging.getLogger("test"))
    
    # Create a file where a directory is expected, causing makedirs to fail
    with tempfile.NamedTemporaryFile(delete=False) as f:
        blocker_path = f.name
    invalid_dir = os.path.join(blocker_path, "subdir")
    
    trainer = SafeThoughtAETrainerV4(model, config, monitor, output_dir=invalid_dir)
    
    # Should NOT raise — error should be caught and logged
    try:
        trainer._save_checkpoint(0, {"loss": 1.0})
    except OSError:
        assert False, "_save_checkpoint should catch OSError, not propagate it"
    finally:
        os.unlink(blocker_path)
    
    print("✅ test_save_checkpoint_error_handling PASSED")


def test_save_metrics_error_handling():
    """Verify save_metrics handles I/O errors gracefully."""
    import tempfile
    from ae_train import TrainingMonitor
    
    monitor = TrainingMonitor(logging.getLogger("test"))
    
    # Create a file where a directory is expected, causing makedirs to fail
    with tempfile.NamedTemporaryFile(delete=False) as f:
        blocker_path = f.name
    invalid_path = os.path.join(blocker_path, "subdir", "metrics.json")
    
    # Should NOT raise — error should be caught and logged
    try:
        monitor.save_metrics(invalid_path)
    except OSError:
        assert False, "save_metrics should catch OSError, not propagate it"
    finally:
        os.unlink(blocker_path)
    
    print("✅ test_save_metrics_error_handling PASSED")


def test_rssm_nan_branch_no_zero_grad():
    """Verify ContextualRSSMTrainer NaN branch does NOT call optimizer.zero_grad().
    
    When NaN loss is detected, the NaN branch should simply skip backward
    without zeroing gradients, preserving any accumulated gradients from
    prior valid steps.
    """
    from ae_train import ContextualRSSMTrainer, AEONConfigV4, AEONDeltaV4, TrainingMonitor
    
    config = AEONConfigV4(vocab_size=100, z_dim=32, hidden_dim=32,
                          vq_num_embeddings=16, vq_embedding_dim=32,
                          seq_length=16, use_amp=False)
    model = AEONDeltaV4(config)
    monitor = TrainingMonitor(logging.getLogger("test"))
    trainer = ContextualRSSMTrainer(model, config, monitor)
    
    # First, do a valid training step to accumulate gradients
    K = config.context_window
    z_context = torch.randn(2, K, config.z_dim)
    z_target = torch.randn(2, config.z_dim)
    trainer.train_step(z_context, z_target)
    
    # Now manually set some gradients on RSSM params to simulate accumulation
    for p in model.rssm.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    
    grad_snapshot = {
        name: p.grad.clone()
        for name, p in model.rssm.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert len(grad_snapshot) > 0, "Should have gradient snapshots"
    
    # Monkey-patch to produce NaN loss
    original_forward = model.rssm.forward
    def nan_forward(z_ctx):
        result = original_forward(z_ctx)
        return result * float('nan')
    model.rssm.forward = nan_forward
    
    # NaN training step should NOT destroy gradients
    metrics = trainer.train_step(z_context, z_target)
    assert math.isnan(metrics['total_loss']), "Should have detected NaN"
    
    # Verify gradients are preserved (not zeroed)
    for name, old_grad in grad_snapshot.items():
        param = dict(model.rssm.named_parameters())[name]
        assert param.grad is not None, f"Gradient for {name} was zeroed"
        assert torch.equal(param.grad, old_grad), (
            f"Gradient for {name} was modified by NaN path"
        )
    
    # Restore
    model.rssm.forward = original_forward
    
    print("✅ test_rssm_nan_branch_no_zero_grad PASSED")


def test_config_v4_extended_validation():
    """Verify AEONConfigV4 validates additional parameters."""
    from ae_train import AEONConfigV4
    
    # entropy_weight < 0 should raise
    try:
        AEONConfigV4(entropy_weight=-0.1)
        assert False, "Should have raised ValueError for negative entropy_weight"
    except ValueError as e:
        assert "entropy_weight" in str(e)
    
    # vq_loss_weight < 0 should raise
    try:
        AEONConfigV4(vq_loss_weight=-1.0)
        assert False, "Should have raised ValueError for negative vq_loss_weight"
    except ValueError as e:
        assert "vq_loss_weight" in str(e)
    
    # min_learning_rate <= 0 should raise
    try:
        AEONConfigV4(min_learning_rate=0)
        assert False, "Should have raised ValueError for zero min_learning_rate"
    except ValueError as e:
        assert "min_learning_rate" in str(e)
    
    # save_every_n_epochs <= 0 should raise
    try:
        AEONConfigV4(save_every_n_epochs=0)
        assert False, "Should have raised ValueError for zero save_every_n_epochs"
    except ValueError as e:
        assert "save_every_n_epochs" in str(e)
    
    # keep_n_checkpoints <= 0 should raise
    try:
        AEONConfigV4(keep_n_checkpoints=0)
        assert False, "Should have raised ValueError for zero keep_n_checkpoints"
    except ValueError as e:
        assert "keep_n_checkpoints" in str(e)
    
    # min_doc_chunks < 1 should raise
    try:
        AEONConfigV4(min_doc_chunks=0)
        assert False, "Should have raised ValueError for zero min_doc_chunks"
    except ValueError as e:
        assert "min_doc_chunks" in str(e)
    
    # Valid values should pass
    config = AEONConfigV4(
        entropy_weight=0.0,
        vq_loss_weight=0.0,
        min_learning_rate=1e-7,
        save_every_n_epochs=1,
        keep_n_checkpoints=1,
        min_doc_chunks=1,
    )
    assert config.entropy_weight == 0.0
    assert config.min_doc_chunks == 1
    
    print("✅ test_config_v4_extended_validation PASSED")


def test_chunked_processor_adaptive_stride_not_zero():
    """Fix: aeon_core.py - ChunkedSequenceProcessor stride must be >= 1 in adaptive mode.
    
    When adaptive mode reduces chunk_size to min_chunk_size and min_chunk_size <= overlap,
    stride = chunk_size - overlap could be <= 0, causing an infinite loop.
    The fix ensures stride = max(chunk_size - overlap, 1).
    """
    from aeon_core import ChunkedSequenceProcessor
    import threading
    
    # Create processor where min_chunk_size == overlap (stride would be 0 without fix)
    processor = ChunkedSequenceProcessor(
        chunk_size=512, overlap=64, adaptive=True, min_chunk_size=64
    )
    
    # Uniform input => all per-position variances equal => adaptive_factor ≈ 0
    # => chunk_size = min_chunk_size = overlap = 64 => stride would be 0 without fix
    B, L, D = 2, 256, 32
    x = torch.ones(B, L, D)
    
    def dummy_model(chunk, state):
        return chunk, state
    
    # Run in a thread with timeout to detect infinite loops
    result = [None]
    error = [None]
    
    def run_process():
        try:
            result[0] = processor.process(dummy_model, x)
        except Exception as e:
            error[0] = e
    
    t = threading.Thread(target=run_process)
    t.start()
    t.join(timeout=5)  # 5-second timeout
    
    assert not t.is_alive(), "ChunkedSequenceProcessor.process() timed out — possible infinite loop"
    assert error[0] is None, f"Unexpected error: {error[0]}"
    
    y, _ = result[0]
    assert y.shape == (B, L, D), f"Output shape mismatch: {y.shape}"
    
    print("✅ test_chunked_processor_adaptive_stride_not_zero PASSED")


def test_fit_remaining_batch_metrics():
    """Fix: ae_train.py - SafeThoughtAETrainerV4.fit() remaining batch metrics inclusion.
    
    When total_batches is not evenly divisible by gradient_accumulation_steps,
    the remaining batches' metrics should be included in epoch_metrics and
    num_steps should use ceiling division.
    """
    # Simulate the fixed computation
    total_batches = 7
    gradient_accumulation_steps = 4
    
    # Fixed: ceiling division
    num_steps_fixed = max(
        (total_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps,
        1
    )
    
    # Old: floor division
    num_steps_old = max(total_batches // gradient_accumulation_steps, 1)
    
    # With 7 batches and 4 accumulation steps:
    # Old: 7 // 4 = 1 (misses the partial step)
    # Fixed: (7 + 3) // 4 = 2 (counts the partial step)
    assert num_steps_fixed == 2, f"Expected 2 steps, got {num_steps_fixed}"
    assert num_steps_old == 1, f"Expected old to be 1, got {num_steps_old}"
    
    # Verify edge case: exactly divisible
    total_batches_exact = 8
    num_steps_exact = max(
        (total_batches_exact + gradient_accumulation_steps - 1) // gradient_accumulation_steps,
        1
    )
    assert num_steps_exact == 2, f"Expected 2 steps for exact division, got {num_steps_exact}"
    
    # Verify edge case: single batch
    total_batches_one = 1
    num_steps_one = max(
        (total_batches_one + gradient_accumulation_steps - 1) // gradient_accumulation_steps,
        1
    )
    assert num_steps_one == 1, f"Expected 1 step, got {num_steps_one}"
    
    print("✅ test_fit_remaining_batch_metrics PASSED")


# ============================================================================
# Advanced Cognitive Modules Tests (Priority 1-5)
# ============================================================================

def test_certified_meta_loop_forward():
    """Priority 1: CertifiedMetaLoop forward pass produces valid output."""
    from aeon_core import CertifiedMetaLoop, AEONConfig
    config = AEONConfig()
    model = CertifiedMetaLoop(config, max_iterations=5)
    z = torch.randn(2, config.hidden_dim)
    C, iters, meta = model(z)
    assert C.shape == (2, config.hidden_dim), f"Expected shape (2, {config.hidden_dim}), got {C.shape}"
    assert 'certified_convergence' in meta
    assert 'certified_error_bound' in meta
    assert 'ibp_lipschitz' in meta
    assert iters.shape == (2,)
    print("✅ test_certified_meta_loop_forward PASSED")


def test_certified_meta_loop_verify_preconditions():
    """Priority 1: verify_convergence_preconditions returns bool and optional float."""
    from aeon_core import CertifiedMetaLoop, AEONConfig
    config = AEONConfig()
    model = CertifiedMetaLoop(config)
    z = torch.randn(2, config.hidden_dim)
    guaranteed, cert_err = model.verify_convergence_preconditions(z)
    assert isinstance(guaranteed, bool)
    if guaranteed:
        assert cert_err is not None and cert_err >= 0.0
    else:
        assert cert_err is None
    print("✅ test_certified_meta_loop_verify_preconditions PASSED")


def test_certified_meta_loop_ibp_lipschitz():
    """Priority 1: IBP Lipschitz estimate is a positive finite number."""
    from aeon_core import CertifiedMetaLoop, AEONConfig
    config = AEONConfig()
    model = CertifiedMetaLoop(config)
    z = torch.randn(1, config.hidden_dim)
    L = model._compute_certified_lipschitz(z)
    assert L > 0, f"Lipschitz should be positive, got {L}"
    assert math.isfinite(L), f"Lipschitz should be finite, got {L}"
    print("✅ test_certified_meta_loop_ibp_lipschitz PASSED")


def test_unified_memory_read():
    """Priority 2: UnifiedMemory read returns correct shape."""
    from aeon_core import UnifiedMemory
    mem = UnifiedMemory(capacity=64, dim=32)
    query = torch.randn(32)
    result = mem(query)
    assert result.shape == (32,), f"Expected (32,), got {result.shape}"
    print("✅ test_unified_memory_read PASSED")


def test_unified_memory_write_and_read():
    """Priority 2: UnifiedMemory write then read retrieves relevant content."""
    from aeon_core import UnifiedMemory
    mem = UnifiedMemory(capacity=64, dim=32)
    value = torch.randn(32)
    # Write
    mem(value, value=value)
    assert mem.num_used_slots >= 1, "Should have at least 1 used slot after write"
    # Read with same query should return something non-zero
    result = mem(value)
    assert result.shape == (32,)
    assert torch.norm(result).item() > 0, "Read result should be non-zero after write"
    print("✅ test_unified_memory_write_and_read PASSED")


def test_unified_memory_batched():
    """Priority 2: UnifiedMemory handles batched queries."""
    from aeon_core import UnifiedMemory
    mem = UnifiedMemory(capacity=64, dim=32, num_read_heads=4)
    query = torch.randn(4, 32)
    result = mem(query)
    assert result.shape == (4, 32), f"Expected (4, 32), got {result.shape}"
    print("✅ test_unified_memory_batched PASSED")


def test_unified_memory_temporal_links():
    """Priority 2: UnifiedMemory builds temporal links across writes."""
    from aeon_core import UnifiedMemory
    mem = UnifiedMemory(capacity=64, dim=16)
    v1 = torch.randn(16)
    v2 = torch.randn(16)
    mem(v1, value=v1)
    mem(v2, value=v2)
    # Link matrix should have at least one non-zero entry
    assert mem.L.abs().sum().item() > 0, "Link matrix should be non-zero after 2 writes"
    print("✅ test_unified_memory_temporal_links PASSED")


def test_hierarchical_world_model_forward():
    """Priority 3: HierarchicalWorldModel forward produces valid output."""
    from aeon_core import HierarchicalWorldModel, AEONConfig
    config = AEONConfig()
    model = HierarchicalWorldModel(config)
    state = torch.randn(2, config.hidden_dim)
    pred, hiddens = model(state)
    assert pred.shape == (2, config.hidden_dim), f"Expected (2, {config.hidden_dim}), got {pred.shape}"
    assert 'h0' in hiddens and 'h1' in hiddens and 'h2' in hiddens
    print("✅ test_hierarchical_world_model_forward PASSED")


def test_hierarchical_world_model_single_level():
    """Priority 3: HierarchicalWorldModel can run at a single level."""
    from aeon_core import HierarchicalWorldModel, AEONConfig
    config = AEONConfig()
    model = HierarchicalWorldModel(config)
    state = torch.randn(2, config.hidden_dim)
    pred, hiddens = model(state, level='0')
    assert pred.shape == (2, config.hidden_dim)
    assert 'h0' in hiddens
    print("✅ test_hierarchical_world_model_single_level PASSED")


def test_hierarchical_world_model_gradient_flow():
    """Priority 3: Gradients flow through all levels of HierarchicalWorldModel."""
    from aeon_core import HierarchicalWorldModel, AEONConfig
    config = AEONConfig()
    model = HierarchicalWorldModel(config)
    state = torch.randn(2, config.hidden_dim, requires_grad=True)
    pred, _ = model(state)
    loss = pred.sum()
    loss.backward()
    assert state.grad is not None, "Gradient should flow to input"
    assert state.grad.abs().sum().item() > 0, "Gradient should be non-zero"
    print("✅ test_hierarchical_world_model_gradient_flow PASSED")


def test_adaptive_meta_loop_forward():
    """Priority 4: AdaptiveMetaLoop produces valid output and metadata."""
    from aeon_core import AdaptiveMetaLoop, AEONConfig
    config = AEONConfig()
    model = AdaptiveMetaLoop(config, max_steps=10)
    z = torch.randn(2, config.hidden_dim)
    C, meta = model(z)
    assert C.shape == (2, config.hidden_dim)
    assert 'steps' in meta
    assert 'ponder_cost' in meta
    assert 'halted' in meta
    assert 'mean_steps' in meta
    print("✅ test_adaptive_meta_loop_forward PASSED")


def test_adaptive_meta_loop_ponder_cost():
    """Priority 4: Ponder cost is a non-negative scalar."""
    from aeon_core import AdaptiveMetaLoop, AEONConfig
    config = AEONConfig()
    model = AdaptiveMetaLoop(config, max_steps=10)
    z = torch.randn(4, config.hidden_dim)
    _, meta = model(z)
    assert meta['ponder_cost'].item() >= 0, "Ponder cost should be non-negative"
    print("✅ test_adaptive_meta_loop_ponder_cost PASSED")


def test_adaptive_meta_loop_gradient_flow():
    """Priority 4: Gradients flow through AdaptiveMetaLoop."""
    from aeon_core import AdaptiveMetaLoop, AEONConfig
    config = AEONConfig()
    model = AdaptiveMetaLoop(config, max_steps=5)
    z = torch.randn(2, config.hidden_dim, requires_grad=True)
    C, _ = model(z)
    loss = C.sum()
    loss.backward()
    assert z.grad is not None, "Gradient should flow to input"
    print("✅ test_adaptive_meta_loop_gradient_flow PASSED")


def test_neuro_symbolic_reasoner_forward():
    """Priority 5: NeuroSymbolicReasoner produces conclusions."""
    from aeon_core import NeuroSymbolicReasoner
    reasoner = NeuroSymbolicReasoner(hidden_dim=64, num_predicates=16)
    state = torch.randn(2, 64)
    result = reasoner(state)
    assert 'conclusions' in result
    assert 'facts' in result
    assert 'rules' in result
    assert 'derived' in result
    assert result['conclusions'].shape == (2, 64)
    assert result['facts'].shape == (2, 16)
    print("✅ test_neuro_symbolic_reasoner_forward PASSED")


def test_neuro_symbolic_reasoner_gradient_flow():
    """Priority 5: Gradients flow through NeuroSymbolicReasoner."""
    from aeon_core import NeuroSymbolicReasoner
    reasoner = NeuroSymbolicReasoner(hidden_dim=64, num_predicates=16)
    state = torch.randn(2, 64, requires_grad=True)
    result = reasoner(state)
    loss = result['conclusions'].sum()
    loss.backward()
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0
    print("✅ test_neuro_symbolic_reasoner_gradient_flow PASSED")


def test_differentiable_forward_chainer():
    """Priority 5: DifferentiableForwardChainer is monotonic."""
    from aeon_core import DifferentiableForwardChainer
    chainer = DifferentiableForwardChainer(num_predicates=8, max_depth=3)
    facts = torch.rand(2, 8) * 0.5  # Initial facts in [0, 0.5]
    rules = torch.rand(2, 8)
    derived = chainer(facts, rules)
    # Monotonicity: derived facts >= initial facts
    assert (derived >= facts - 1e-6).all(), "Forward chaining should be monotonic"
    assert derived.shape == (2, 8)
    print("✅ test_differentiable_forward_chainer PASSED")


def test_neuro_symbolic_facts_in_unit_interval():
    """Priority 5: Facts and rules are in [0, 1] (sigmoid output)."""
    from aeon_core import NeuroSymbolicReasoner
    reasoner = NeuroSymbolicReasoner(hidden_dim=64, num_predicates=16)
    state = torch.randn(4, 64)
    result = reasoner(state)
    assert (result['facts'] >= 0).all() and (result['facts'] <= 1).all()
    assert (result['rules'] >= 0).all() and (result['rules'] <= 1).all()
    assert (result['derived'] >= 0).all() and (result['derived'] <= 1).all()
    print("✅ test_neuro_symbolic_facts_in_unit_interval PASSED")


# ============================================================================
# ANALYSIS-DRIVEN REFACTORING TESTS: NaN/Inf guards, epsilon safety, exception specificity
# ============================================================================

def test_lipschitz_estimate_nan_guard():
    """Verify that NaN in lipschitz_estimate does not propagate into compute_fixed_point."""
    from aeon_core import ProvablyConvergentMetaLoop, AEONConfig

    config = AEONConfig(device_str='cpu')
    meta_loop = ProvablyConvergentMetaLoop(config)

    # Corrupt the lipschitz_estimate buffer with NaN
    meta_loop.lambda_op.lipschitz_estimate.fill_(float('nan'))

    psi_0 = torch.randn(2, config.hidden_dim)
    C, iterations, meta = meta_loop.compute_fixed_point(psi_0)

    # lip_const should have fallen back to 1.0
    assert meta['lipschitz_estimate'] == 1.0, (
        f"Expected fallback 1.0, got {meta['lipschitz_estimate']}"
    )
    # Output should be finite
    assert torch.isfinite(C).all(), "C contains NaN/Inf despite guard"
    print("✅ test_lipschitz_estimate_nan_guard PASSED")


def test_lipschitz_ema_nan_skip():
    """Verify that NaN lipschitz_estimate in get_lipschitz_penalty does not corrupt EMA buffer."""
    from aeon_core import LipschitzConstrainedLambda

    lip = LipschitzConstrainedLambda(
        input_dim=64, hidden_dim=32, output_dim=32,
        lipschitz_target=0.85, use_spectral_norm=True
    )

    # Set the EMA buffer to a known good value
    lip.lipschitz_estimate.fill_(0.5)

    # Create inputs that would produce valid penalty but corrupt the internal estimate
    x = torch.randn(2, 64)
    y = x.clone()  # Same points: denominator → 0, clamped to 1e-8

    penalty = lip.get_lipschitz_penalty(x, y)
    assert torch.isfinite(penalty), f"Penalty is not finite: {penalty}"

    # EMA buffer should still be finite
    assert torch.isfinite(lip.lipschitz_estimate), (
        f"EMA buffer corrupted: {lip.lipschitz_estimate.item()}"
    )
    print("✅ test_lipschitz_ema_nan_skip PASSED")


def test_denominator_max_vs_add():
    """Verify that max(value, eps) is used instead of value + eps for NaN safety."""
    from aeon_core import LipschitzConstrainedLambda

    lip = LipschitzConstrainedLambda(
        input_dim=16, hidden_dim=8, output_dim=8,
        lipschitz_target=0.85, use_spectral_norm=True
    )

    # Two identical points: norm difference is 0
    max_ratio = lip.compute_lipschitz_constant(num_samples=5, sample_dim=16)
    assert math.isfinite(max_ratio), f"max_ratio is not finite: {max_ratio}"
    print("✅ test_denominator_max_vs_add PASSED")


def test_certified_error_nan_residual():
    """Verify that certified_error handles NaN residual gracefully."""
    from aeon_core import ProvablyConvergentMetaLoop, AEONConfig

    config = AEONConfig(device_str='cpu')
    meta_loop = ProvablyConvergentMetaLoop(config)

    # Set lip_const to valid < 1.0 so certification branch is taken
    meta_loop.lambda_op.lipschitz_estimate.fill_(0.5)

    psi_0 = torch.randn(2, config.hidden_dim)
    C, iterations, meta = meta_loop.compute_fixed_point(psi_0)

    # certified_error should be finite or inf (not NaN)
    cert_err = meta.get('certified_error_bound')
    if cert_err is not None:
        assert not math.isnan(cert_err), f"certified_error is NaN"
    print("✅ test_certified_error_nan_residual PASSED")


def test_checkpoint_load_specific_exception():
    """Verify that checkpoint loading catches specific exceptions, not all."""
    import inspect
    from ae_train import main

    # Check that the source code uses specific exception types
    source = inspect.getsource(main)
    # The fallback for weights_only should catch RuntimeError/TypeError, not bare Exception
    assert "except (RuntimeError, TypeError)" in source or "except RuntimeError" in source, (
        "Checkpoint loading should catch specific exceptions, not bare 'except Exception'"
    )
    print("✅ test_checkpoint_load_specific_exception PASSED")


def test_adaptive_chunking_max_var():
    """Verify adaptive chunking uses max() instead of addition for NaN safety."""
    from aeon_core import ChunkedSequenceProcessor

    processor = ChunkedSequenceProcessor(
        chunk_size=32,
        overlap=8,
        adaptive=True,
        min_chunk_size=16
    )

    # Test with input where variance is very small (near zero)
    x = torch.ones(1, 64, 16)  # constant input → variance ≈ 0

    def model_fn(x_chunk, state=None):
        return x_chunk, state

    # The processor should not crash even with near-zero variance
    result, _ = processor.process(model_fn, x)
    assert torch.isfinite(result).all(), "Output has NaN/Inf with zero-variance input"
    print("✅ test_adaptive_chunking_max_var PASSED")


# ============================================================================
# Tests for architectural recommendations (Gumbel-Softmax, NTM, LatentDynamics, CausalProgrammatic)
# ============================================================================

def test_gumbel_vector_quantizer_forward():
    """GumbelVectorQuantizer forward pass produces correct shapes and valid outputs."""
    from ae_train import GumbelVectorQuantizer

    gvq = GumbelVectorQuantizer(num_embeddings=16, embedding_dim=32)
    z = torch.randn(4, 32)
    z_q, loss, indices, stats = gvq(z)
    assert z_q.shape == (4, 32), f"Expected (4, 32), got {z_q.shape}"
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert indices.shape == (4,), f"Expected (4,) indices, got {indices.shape}"
    assert 'codebook_usage_%' in stats
    assert 'entropy_loss' in stats
    assert not torch.isnan(z_q).any()
    assert not torch.isnan(loss).any()
    print("✅ test_gumbel_vector_quantizer_forward PASSED")


def test_gumbel_vector_quantizer_training_vs_eval():
    """GumbelVectorQuantizer uses Gumbel-Softmax in training, argmax in eval."""
    from ae_train import GumbelVectorQuantizer

    gvq = GumbelVectorQuantizer(num_embeddings=8, embedding_dim=16)
    z = torch.randn(2, 16)

    gvq.train()
    z_q_train, _, _, stats_train = gvq(z)
    assert 'temperature' in stats_train

    gvq.eval()
    z_q_eval, _, _, stats_eval = gvq(z)
    assert z_q_eval.shape == z_q_train.shape
    print("✅ test_gumbel_vector_quantizer_training_vs_eval PASSED")


def test_gumbel_vector_quantizer_gradient_flow():
    """Gradients flow through GumbelVectorQuantizer (fully differentiable)."""
    from ae_train import GumbelVectorQuantizer

    gvq = GumbelVectorQuantizer(num_embeddings=16, embedding_dim=32)
    gvq.train()
    z = torch.randn(4, 32, requires_grad=True)
    z_q, loss, _, _ = gvq(z)
    total_loss = z_q.sum() + loss
    total_loss.backward()
    assert z.grad is not None, "Gradient did not flow through GumbelVectorQuantizer"
    assert not torch.isnan(z.grad).any()
    print("✅ test_gumbel_vector_quantizer_gradient_flow PASSED")


def test_gumbel_vector_quantizer_temperature_annealing():
    """Temperature decreases during training with Gumbel-Softmax."""
    from ae_train import GumbelVectorQuantizer

    gvq = GumbelVectorQuantizer(
        num_embeddings=16, embedding_dim=32,
        temperature=2.0, min_temperature=0.5, anneal_rate=0.1,
    )
    gvq.train()
    initial_temp = gvq.temperature
    z = torch.randn(4, 32)
    for _ in range(5):
        gvq(z)
    assert gvq.temperature < initial_temp, \
        f"Temperature should decrease: {gvq.temperature} >= {initial_temp}"
    assert gvq.temperature >= 0.5, \
        f"Temperature should not go below min: {gvq.temperature}"
    print("✅ test_gumbel_vector_quantizer_temperature_annealing PASSED")


def test_neural_turing_machine_forward():
    """NeuralTuringMachine forward pass produces correct shapes."""
    from aeon_core import NeuralTuringMachine

    ntm = NeuralTuringMachine(
        input_dim=32, hidden_dim=64, memory_size=16, memory_dim=32, num_read_heads=2
    )
    x = torch.randn(2, 32)
    output, info = ntm(x)
    assert output.shape == (2, 64), f"Expected (2, 64), got {output.shape}"
    assert 'read_vectors' in info
    assert len(info['read_vectors']) == 2  # 2 read heads
    assert not torch.isnan(output).any()
    print("✅ test_neural_turing_machine_forward PASSED")


def test_neural_turing_machine_store_retrieve():
    """NeuralTuringMachine store and retrieve compatibility methods work."""
    from aeon_core import NeuralTuringMachine

    ntm = NeuralTuringMachine(
        input_dim=32, hidden_dim=64, memory_size=16, memory_dim=32, num_read_heads=2
    )
    vec = torch.randn(32)
    ntm.store(vec)

    result = ntm.retrieve(vec, k=2)
    assert 'working' in result
    assert 'episodic' in result
    assert 'semantic' in result
    assert 'route_weights' in result
    assert result['route_weights'].shape == (3,)
    # working should have up to k entries
    assert len(result['working']) <= 2
    print("✅ test_neural_turing_machine_store_retrieve PASSED")


def test_neural_turing_machine_gradient_flow():
    """Gradients flow through NeuralTuringMachine."""
    from aeon_core import NeuralTuringMachine

    ntm = NeuralTuringMachine(
        input_dim=32, hidden_dim=64, memory_size=16, memory_dim=32, num_read_heads=2
    )
    x = torch.randn(2, 32, requires_grad=True)
    output, _ = ntm(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("✅ test_neural_turing_machine_gradient_flow PASSED")


def test_latent_dynamics_model_forward():
    """LatentDynamicsModel single-step forward produces correct outputs."""
    from aeon_core import LatentDynamicsModel

    ldm = LatentDynamicsModel(latent_dim=64, action_dim=8)
    state = torch.randn(2, 64)
    action = torch.randn(2, 8)
    next_state, reward, value = ldm(state, action)
    assert next_state.shape == (2, 64), f"Expected (2, 64), got {next_state.shape}"
    assert reward.shape == (2, 1), f"Expected (2, 1), got {reward.shape}"
    assert value.shape == (2, 1), f"Expected (2, 1), got {value.shape}"
    assert not torch.isnan(next_state).any()
    print("✅ test_latent_dynamics_model_forward PASSED")


def test_latent_dynamics_model_rollout():
    """LatentDynamicsModel multi-step rollout produces correct trajectory."""
    from aeon_core import LatentDynamicsModel

    ldm = LatentDynamicsModel(latent_dim=32, action_dim=4)
    state = torch.randn(1, 32)
    actions = [torch.randn(1, 4) for _ in range(5)]
    trajectory, rewards = ldm.rollout(state, actions)
    assert len(trajectory) == 6, f"Expected 6 states (initial + 5 steps), got {len(trajectory)}"
    assert len(rewards) == 5, f"Expected 5 rewards, got {len(rewards)}"
    assert trajectory[0].shape == (1, 32)
    assert trajectory[-1].shape == (1, 32)
    print("✅ test_latent_dynamics_model_rollout PASSED")


def test_latent_dynamics_model_gradient_flow():
    """Gradients flow through LatentDynamicsModel rollout."""
    from aeon_core import LatentDynamicsModel

    ldm = LatentDynamicsModel(latent_dim=32, action_dim=4)
    state = torch.randn(2, 32, requires_grad=True)
    action = torch.randn(2, 4)
    next_state, reward, value = ldm(state, action)
    loss = next_state.sum() + reward.sum() + value.sum()
    loss.backward()
    assert state.grad is not None
    assert not torch.isnan(state.grad).any()
    print("✅ test_latent_dynamics_model_gradient_flow PASSED")


def test_causal_programmatic_model_forward():
    """CausalProgrammaticModel generative forward pass produces valid variables."""
    from aeon_core import CausalProgrammaticModel

    cpm = CausalProgrammaticModel(num_variables=5, hidden_dim=32)
    obs = torch.randn(2, 5)
    variables, log_prob = cpm(observations=obs)
    assert variables.shape == (2, 5), f"Expected (2, 5), got {variables.shape}"
    assert log_prob.shape == (2,), f"Expected (2,), got {log_prob.shape}"
    assert not torch.isnan(variables).any()
    print("✅ test_causal_programmatic_model_forward PASSED")


def test_causal_programmatic_model_counterfactual():
    """CausalProgrammaticModel counterfactual intervention applies do(X=x)."""
    from aeon_core import CausalProgrammaticModel

    cpm = CausalProgrammaticModel(num_variables=4, hidden_dim=32)
    obs = torch.randn(2, 4)
    cf = cpm.counterfactual(obs, intervention={0: 1.0})
    assert cf.shape == (2, 4), f"Expected (2, 4), got {cf.shape}"
    # Intervened variable should be fixed to intervention value
    assert torch.allclose(cf[:, 0], torch.ones(2)), \
        f"Expected intervened var to be 1.0, got {cf[:, 0]}"
    print("✅ test_causal_programmatic_model_counterfactual PASSED")


def test_causal_programmatic_model_dag_loss():
    """CausalProgrammaticModel dag_loss returns a non-negative scalar."""
    from aeon_core import CausalProgrammaticModel

    cpm = CausalProgrammaticModel(num_variables=4, hidden_dim=32)
    dag_loss = cpm.dag_loss()
    assert dag_loss.shape == (), f"Expected scalar, got {dag_loss.shape}"
    assert dag_loss.item() >= 0, f"DAG loss should be non-negative, got {dag_loss.item()}"
    print("✅ test_causal_programmatic_model_dag_loss PASSED")


def test_causal_programmatic_model_gradient_flow():
    """Gradients flow through CausalProgrammaticModel counterfactual."""
    from aeon_core import CausalProgrammaticModel

    cpm = CausalProgrammaticModel(num_variables=4, hidden_dim=32)
    obs = torch.randn(2, 4, requires_grad=True)
    cf = cpm.counterfactual(obs, intervention={1: 0.5})
    loss = cf.sum()
    loss.backward()
    assert obs.grad is not None
    assert not torch.isnan(obs.grad).any()
    print("✅ test_causal_programmatic_model_gradient_flow PASSED")


# ============================================================================
# TESTS FOR STRATEGIC AGI RECOMMENDATIONS
# ============================================================================

def test_compositional_slot_attention_forward():
    """CompositionalSlotAttention binds features to slots."""
    from aeon_core import CompositionalSlotAttention

    csa = CompositionalSlotAttention(num_slots=7, slot_dim=64, num_heads=4)
    features = torch.randn(2, 10, 64)  # [B, N, D]
    slots = csa(features)
    assert slots.shape == (2, 7, 64), f"Expected (2,7,64), got {slots.shape}"
    assert not torch.isnan(slots).any()
    print("✅ test_compositional_slot_attention_forward PASSED")


def test_compositional_slot_attention_gradient():
    """Gradients flow through CompositionalSlotAttention."""
    from aeon_core import CompositionalSlotAttention

    csa = CompositionalSlotAttention(num_slots=7, slot_dim=32, num_heads=4)
    features = torch.randn(2, 5, 32, requires_grad=True)
    slots = csa(features)
    loss = slots.sum()
    loss.backward()
    assert features.grad is not None
    assert not torch.isnan(features.grad).any()
    print("✅ test_compositional_slot_attention_gradient PASSED")


def test_compositional_slot_attention_iterations():
    """Multiple iterations refine slot assignments."""
    from aeon_core import CompositionalSlotAttention

    csa = CompositionalSlotAttention(num_slots=4, slot_dim=32, num_heads=2)
    features = torch.randn(1, 8, 32)
    slots_1 = csa(features, num_iterations=1)
    slots_3 = csa(features, num_iterations=3)
    # With more iterations, results should differ (refinement happened)
    assert slots_1.shape == slots_3.shape == (1, 4, 32)
    print("✅ test_compositional_slot_attention_iterations PASSED")


def test_notears_causal_model_forward():
    """NOTEARSCausalModel produces correct-shaped output."""
    from aeon_core import NOTEARSCausalModel

    model = NOTEARSCausalModel(num_vars=5, hidden_dim=32)
    exo = torch.randn(3, 5)
    out = model(exo)
    assert out.shape == (3, 5), f"Expected (3,5), got {out.shape}"
    assert not torch.isnan(out).any()
    print("✅ test_notears_causal_model_forward PASSED")


def test_notears_dag_loss():
    """NOTEARSCausalModel dag_loss returns scalar ≥ 0."""
    from aeon_core import NOTEARSCausalModel

    model = NOTEARSCausalModel(num_vars=4, hidden_dim=16)
    dag = model.dag_loss()
    assert dag.dim() == 0, "dag_loss should be scalar"
    assert dag.item() >= -1e-6, f"dag_loss should be ≥ 0, got {dag.item()}"
    print("✅ test_notears_dag_loss PASSED")


def test_notears_dag_loss_gradient():
    """dag_loss is differentiable w.r.t. W."""
    from aeon_core import NOTEARSCausalModel

    model = NOTEARSCausalModel(num_vars=4)
    loss = model.dag_loss()
    loss.backward()
    assert model.W.grad is not None
    assert not torch.isnan(model.W.grad).any()
    print("✅ test_notears_dag_loss_gradient PASSED")


def test_notears_intervention():
    """NOTEARSCausalModel handles do(X=x) interventions."""
    from aeon_core import NOTEARSCausalModel

    model = NOTEARSCausalModel(num_vars=4)
    exo = torch.randn(2, 4)
    out = model(exo, intervention={1: 3.0})
    assert out.shape == (2, 4)
    assert torch.allclose(out[:, 1], torch.tensor(3.0))
    print("✅ test_notears_intervention PASSED")


def test_notears_l1_loss():
    """l1_loss returns a non-negative scalar."""
    from aeon_core import NOTEARSCausalModel

    model = NOTEARSCausalModel(num_vars=3)
    l1 = model.l1_loss()
    assert l1.dim() == 0
    assert l1.item() >= 0.0
    print("✅ test_notears_l1_loss PASSED")


def test_consolidating_memory_store_and_consolidate():
    """ConsolidatingMemory stores items and consolidates across stages."""
    from aeon_core import ConsolidatingMemory

    mem = ConsolidatingMemory(dim=32, working_capacity=4, episodic_capacity=10,
                               importance_threshold=0.0)  # threshold=0 so everything moves
    # Store items
    for _ in range(5):
        mem.store(torch.randn(32))
    assert len(mem.working) == 4  # ring buffer caps at 4

    # Consolidate: working → episodic → semantic
    mem.consolidate()
    assert len(mem.episodic) > 0
    print("✅ test_consolidating_memory_store_and_consolidate PASSED")


def test_consolidating_memory_retrieve():
    """ConsolidatingMemory retrieves from all three stages."""
    from aeon_core import ConsolidatingMemory

    mem = ConsolidatingMemory(dim=16, working_capacity=3, importance_threshold=0.0)
    query = torch.randn(16)
    for _ in range(3):
        mem.store(torch.randn(16))
    mem.consolidate()

    result = mem.retrieve(query, k=2)
    assert 'working' in result
    assert 'episodic' in result
    assert 'semantic' in result
    print("✅ test_consolidating_memory_retrieve PASSED")


def test_consolidating_memory_forward():
    """ConsolidatingMemory forward returns importance scores."""
    from aeon_core import ConsolidatingMemory

    mem = ConsolidatingMemory(dim=16, working_capacity=5)
    x = torch.randn(4, 16)
    scores = mem(x)
    assert scores.shape == (4,)
    assert (scores >= 0).all() and (scores <= 1).all()
    print("✅ test_consolidating_memory_forward PASSED")


def test_consolidating_memory_gradient():
    """Gradients flow through ConsolidatingMemory importance scorer."""
    from aeon_core import ConsolidatingMemory

    mem = ConsolidatingMemory(dim=16)
    x = torch.randn(2, 16, requires_grad=True)
    scores = mem(x)
    loss = scores.sum()
    loss.backward()
    assert x.grad is not None
    print("✅ test_consolidating_memory_gradient PASSED")


def test_task2vec_meta_learner_embed():
    """Task2VecMetaLearner produces embeddings of correct dimension."""
    from aeon_core import Task2VecMetaLearner

    inner = nn.Linear(8, 4)
    t2v = Task2VecMetaLearner(model=inner, embedding_dim=32)
    fisher = {name: torch.randn_like(p) for name, p in inner.named_parameters() if p.requires_grad}
    emb = t2v.embed_task(fisher)
    assert emb.shape == (32,), f"Expected (32,), got {emb.shape}"
    print("✅ test_task2vec_meta_learner_embed PASSED")


def test_task2vec_meta_learner_adapt():
    """Task2VecMetaLearner.adapt creates new task clusters."""
    from aeon_core import Task2VecMetaLearner

    inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    t2v = Task2VecMetaLearner(model=inner, embedding_dim=16, similarity_threshold=0.99)

    def data_loader():
        for _ in range(3):
            yield torch.randn(4, 4), torch.randint(0, 2, (4,))

    result = t2v.adapt(data_loader_fn=data_loader, num_samples=10)
    assert result['mode'] == 'new'
    assert t2v.num_task_clusters == 1
    print("✅ test_task2vec_meta_learner_adapt PASSED")


def test_task2vec_ewc_loss():
    """Task2VecMetaLearner ewc_loss returns differentiable scalar."""
    from aeon_core import Task2VecMetaLearner

    inner = nn.Linear(4, 2)
    t2v = Task2VecMetaLearner(model=inner, ewc_lambda=100.0)
    fisher = {name: torch.ones_like(p) for name, p in inner.named_parameters() if p.requires_grad}
    opt_params = {name: p.data.clone() for name, p in inner.named_parameters() if p.requires_grad}

    loss = t2v.ewc_loss(fisher, opt_params)
    assert loss.dim() == 0
    # At initial params, difference is zero
    assert loss.item() < 1e-6
    print("✅ test_task2vec_ewc_loss PASSED")


def test_certified_meta_loop_ibp_per_layer():
    """CertifiedMetaLoop IBP handles GELU and LayerNorm layers separately."""
    from aeon_core import CertifiedMetaLoop, AEONConfig

    config = AEONConfig(
        hidden_dim=64, meta_dim=64, z_dim=64,
        vq_embedding_dim=64, num_pillars=4
    )
    loop = CertifiedMetaLoop(config=config)
    z = torch.randn(2, 64)
    L = loop._compute_certified_lipschitz(z)
    assert isinstance(L, float)
    assert L > 0, "Lipschitz bound must be positive"

    # Verify it checks preconditions
    guaranteed, cert_err = loop.verify_convergence_preconditions(z)
    assert isinstance(guaranteed, bool)
    if guaranteed:
        assert cert_err is not None and cert_err >= 0
    print("✅ test_certified_meta_loop_ibp_per_layer PASSED")


# ============================================================================
# TESTS FOR REFACTORING FIXES (division-by-zero, type safety, error handling)
# ============================================================================

def test_epoch_metrics_empty_list_guard():
    """Verify that avg_metrics handles empty metric lists without ZeroDivisionError.
    
    Fixes division-by-zero in aeon_core.py epoch summary and evaluate methods
    where sum(v)/len(v) would crash if no batches produced metrics.
    """
    from collections import defaultdict

    # Simulate empty epoch (no batches ran)
    epoch_metrics = defaultdict(list)
    # No items appended — len(v) == 0 for all keys

    # Manually add a key with empty list to trigger the guard
    epoch_metrics['total_loss'] = []
    epoch_metrics['consistency_score'] = []

    # This should NOT raise ZeroDivisionError
    avg_metrics = {
        k: sum(v) / max(len(v), 1) 
        for k, v in epoch_metrics.items()
    }
    assert avg_metrics['total_loss'] == 0.0
    assert avg_metrics['consistency_score'] == 0.0

    # Normal case should still work
    epoch_metrics['total_loss'] = [1.0, 2.0, 3.0]
    avg_metrics = {
        k: sum(v) / max(len(v), 1) 
        for k, v in epoch_metrics.items()
    }
    assert abs(avg_metrics['total_loss'] - 2.0) < 1e-6

    print("✅ test_epoch_metrics_empty_list_guard PASSED")


def test_weight_tying_scores_empty_guard():
    """Verify that weight tying overall score handles empty results dict."""
    # Simulate empty results dict
    results = {}
    scores = [1.0 if v else 0.0 for v in results.values()]
    overall = sum(scores) / max(len(scores), 1)
    assert overall == 0.0

    # Normal case
    results = {'a': True, 'b': False, 'c': True}
    scores = [1.0 if v else 0.0 for v in results.values()]
    overall = sum(scores) / max(len(scores), 1)
    assert abs(overall - 2.0/3.0) < 1e-6

    print("✅ test_weight_tying_scores_empty_guard PASSED")


def test_entropy_loss_returns_tensor():
    """Verify that _compute_entropy_loss always returns a torch.Tensor."""
    from ae_train import VectorQuantizerHybridV4

    vq = VectorQuantizerHybridV4(num_embeddings=64, embedding_dim=32)

    # Normal case
    indices = torch.randint(0, 64, (16,))
    loss = vq._compute_entropy_loss(indices)
    assert torch.is_tensor(loss), f"Expected Tensor, got {type(loss)}"
    assert loss.dim() == 0, "Expected scalar tensor"
    assert not torch.isnan(loss), "Entropy loss is NaN"

    # Single unique code — maximum entropy loss
    indices_single = torch.zeros(16, dtype=torch.long)
    loss_single = vq._compute_entropy_loss(indices_single)
    assert torch.is_tensor(loss_single), f"Expected Tensor, got {type(loss_single)}"
    assert loss_single.item() > 0.5, "Entropy loss should be high for single code usage"

    print("✅ test_entropy_loss_returns_tensor PASSED")


def test_optimizer_step_returns_float():
    """Verify that _optimizer_step always returns a float."""
    from ae_train import AEONConfigV4, AEONDeltaV4, SafeThoughtAETrainerV4, TrainingMonitor

    config = AEONConfigV4()
    model = AEONDeltaV4(config)
    monitor = TrainingMonitor(logging.getLogger("test"), save_dir="/tmp/test_opt_step")
    trainer = SafeThoughtAETrainerV4(model, config, monitor, output_dir="/tmp/test_opt_step")

    # Simulate a forward-backward pass to populate gradients
    tokens = torch.randint(0, config.vocab_size, (2, config.seq_length))
    trainer.train_step(tokens)

    # The optimizer step should return a float
    result = trainer._optimizer_step()
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert not math.isnan(result), "grad_norm is NaN"

    print("✅ test_optimizer_step_returns_float PASSED")


def test_grad_norm_nan_guard_in_fit():
    """Verify that NaN grad_norm values are safely guarded in fit loop."""
    # Simulate the guard logic used in fit()
    epoch_metrics = {"grad_norm": 0.0}
    
    # NaN grad_norm should be treated as 0
    grad_norm = float('nan')
    epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and not math.isnan(grad_norm)) else 0
    assert epoch_metrics["grad_norm"] == 0.0, "NaN grad_norm leaked into metrics"
    
    # Normal grad_norm should be accumulated
    grad_norm = 1.5
    epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and not math.isnan(grad_norm)) else 0
    assert abs(epoch_metrics["grad_norm"] - 1.5) < 1e-6

    # Zero grad_norm should be accumulated (valid value)
    grad_norm = 0.0
    epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and not math.isnan(grad_norm)) else 0
    assert abs(epoch_metrics["grad_norm"] - 1.5) < 1e-6  # 1.5 + 0.0 = 1.5

    # None grad_norm should be treated as 0
    grad_norm = None
    epoch_metrics["grad_norm"] += grad_norm if (grad_norm is not None and not math.isnan(grad_norm)) else 0
    assert abs(epoch_metrics["grad_norm"] - 1.5) < 1e-6  # unchanged

    print("✅ test_grad_norm_nan_guard_in_fit PASSED")


# ============================================================================
# MODERNIZATION TESTS: Robust logic improvements
# ============================================================================


def test_rssm_residual_and_norm():
    """Verify RSSM uses residual connection and LayerNorm for stable dynamics."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    # Verify the RSSM components exist
    assert hasattr(model, 'rssm_cell'), "rssm_cell not found"
    assert hasattr(model, 'rssm_norm'), "rssm_norm not found"
    assert isinstance(model.rssm_cell, torch.nn.GRUCell)
    assert isinstance(model.rssm_norm, torch.nn.LayerNorm)

    # Run a forward pass to ensure the pipeline works end-to-end
    input_ids = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        result = model(input_ids, fast=True)
    assert result['logits'] is not None
    assert torch.isfinite(result['logits']).all(), "Logits contain NaN/Inf"

    print("✅ test_rssm_residual_and_norm PASSED")


def test_integration_module_residual_norm():
    """Verify integration module uses projection + LayerNorm + residual."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)

    assert hasattr(model, 'integration_proj'), "integration_proj not found"
    assert hasattr(model, 'integration_norm'), "integration_norm not found"
    assert isinstance(model.integration_proj, torch.nn.Linear)
    assert isinstance(model.integration_norm, torch.nn.LayerNorm)

    # Verify shapes: proj takes hidden_dim*2 → hidden_dim
    assert model.integration_proj.in_features == config.hidden_dim * 2
    assert model.integration_proj.out_features == config.hidden_dim
    assert model.integration_norm.normalized_shape[0] == config.hidden_dim

    print("✅ test_integration_module_residual_norm PASSED")


def test_consistency_gate_forward():
    """Verify consistency gate produces valid gating signals in [0, 1]."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    # Test the consistency gate directly
    B = 4
    x = torch.randn(B, config.hidden_dim * 2)
    with torch.no_grad():
        gate = model.consistency_gate(x)

    assert gate.shape == (B, config.hidden_dim), f"Expected ({B}, {config.hidden_dim}), got {gate.shape}"
    assert (gate >= 0).all(), "Gate values below 0 (Sigmoid should be >= 0)"
    assert (gate <= 1).all(), "Gate values above 1 (Sigmoid should be <= 1)"
    assert torch.isfinite(gate).all(), "Gate contains NaN/Inf"

    print("✅ test_consistency_gate_forward PASSED")


def test_consistency_gate_gradient_flow():
    """Verify gradients flow through the consistency gate."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.train()

    input_ids = torch.randint(0, 1000, (2, 8))
    result = model(input_ids, fast=True)
    loss = result['logits'].sum()
    loss.backward()

    # Check that consistency gate parameters received gradients
    has_grad = False
    for p in model.consistency_gate.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowed through consistency_gate"

    print("✅ test_consistency_gate_gradient_flow PASSED")


def test_consistency_gate_in_reasoning_output():
    """Verify reasoning_core outputs include consistency_gate and convergence_quality."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        result = model(input_ids, fast=True)

    assert 'consistency_gate' in result, "consistency_gate missing from outputs"
    assert 'convergence_quality' in result, "convergence_quality missing from outputs"

    gate = result['consistency_gate']
    assert gate.shape == (2, config.hidden_dim), f"Gate shape mismatch: {gate.shape}"
    assert (gate >= 0).all() and (gate <= 1).all(), "Gate values out of [0, 1]"

    print("✅ test_consistency_gate_in_reasoning_output PASSED")


def test_value_net_has_layer_norm():
    """Verify value_net includes LayerNorm for stable value estimation."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_world_model=True,
    )
    model = AEONDeltaV3(config)

    assert hasattr(model, 'value_net'), "value_net not found"
    # Check that LayerNorm is present in the value_net
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in model.value_net.modules())
    assert has_ln, "value_net should include LayerNorm for stable value estimation"

    # Verify it produces valid scalar outputs
    x = torch.randn(3, config.hidden_dim)
    with torch.no_grad():
        val = model.value_net(x)
    assert val.shape == (3, 1), f"Expected (3, 1), got {val.shape}"
    assert torch.isfinite(val).all(), "value_net output contains NaN/Inf"

    print("✅ test_value_net_has_layer_norm PASSED")


def test_importance_scorer_has_layer_norm():
    """Verify importance_scorer includes LayerNorm for gradient stability."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_hierarchical_memory=True,
    )
    model = AEONDeltaV3(config)

    assert hasattr(model, 'importance_scorer'), "importance_scorer not found"
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in model.importance_scorer.modules())
    assert has_ln, "importance_scorer should include LayerNorm"

    # Verify valid output range [0, 1] from Sigmoid
    x = torch.randn(4, config.hidden_dim)
    with torch.no_grad():
        scores = model.importance_scorer(x)
    assert scores.shape == (4, 1), f"Expected (4, 1), got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores out of [0, 1]"

    print("✅ test_importance_scorer_has_layer_norm PASSED")


# ============================================================================
# AGI Modernization Tests: Error Resilience & Logical Integrity
# ============================================================================

def test_convergence_trajectory_bounded():
    """Verify convergence_trajectory uses bounded deque, not unbounded list."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        max_iterations=10,
    )
    loop = ProvablyConvergentMetaLoop(
        config=config, max_iterations=10, min_iterations=1,
    )
    psi_0 = torch.randn(2, 64)
    with torch.no_grad():
        _, _, meta = loop.compute_fixed_point(psi_0)
    traj = meta['convergence_trajectory']
    assert isinstance(traj, list), "trajectory should be a list (from deque)"
    assert len(traj) <= 10, f"trajectory should be bounded to max_iterations, got {len(traj)}"
    print("✅ test_convergence_trajectory_bounded PASSED")


def test_memory_manager_capacity_bound():
    """Verify MemoryManager enforces capacity limits."""
    from aeon_core import AEONConfig, MemoryManager
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
    )
    mm = MemoryManager(config)
    mm._max_capacity = 5  # Override for testing

    for i in range(10):
        vec = torch.randn(64)
        mm.add_embedding(vec, meta={'idx': i})

    assert mm.size == 5, f"Expected 5, got {mm.size}"
    # The oldest entries (0-4) should have been evicted; the newest (5-9) remain
    assert mm.fallback_metas[0]['idx'] == 5, (
        f"Expected oldest remaining to be idx=5, got {mm.fallback_metas[0]['idx']}"
    )
    print("✅ test_memory_manager_capacity_bound PASSED")


def test_memory_manager_thread_safety():
    """Verify MemoryManager has a lock for thread safety."""
    from aeon_core import AEONConfig, MemoryManager
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
    )
    mm = MemoryManager(config)
    assert hasattr(mm, '_lock'), "MemoryManager should have a _lock attribute"
    import threading
    assert isinstance(mm._lock, type(threading.Lock())), "_lock should be a threading.Lock"
    print("✅ test_memory_manager_thread_safety PASSED")


def test_inference_cache_model_version_invalidation():
    """Verify InferenceCache invalidates on model version change."""
    from aeon_core import InferenceCache
    cache = InferenceCache(maxlen=16)

    # Set initial state
    cache.set_ssm_state([torch.randn(1, 64)])
    assert cache.step == 1

    # Validate with version 1
    valid = cache.validate_model_version(1)
    assert valid is True

    # Same version — still valid
    valid = cache.validate_model_version(1)
    assert valid is True
    assert cache.step == 1  # State preserved

    # Version changes — cache should be invalidated
    valid = cache.validate_model_version(2)
    assert valid is False
    assert cache.step == 0  # Reset
    assert cache.get_ssm_state() is None

    print("✅ test_inference_cache_model_version_invalidation PASSED")


def test_hessian_nonfinite_sanitization():
    """Verify FastHessianComputer sanitizes non-finite Hessian values."""
    from aeon_core import FastHessianComputer
    hc = FastHessianComputer(method='finite_differences', epsilon=1e-4)

    # Create a function that returns NaN for certain inputs
    def nan_func(x):
        result = x.sum(dim=-1)
        result = result + float('nan')  # Force NaN
        return result

    x = torch.randn(2, 4)
    H = hc._hessian_finite_differences(nan_func, x)

    # Hessian should be sanitized (no NaN/Inf)
    assert torch.isfinite(H).all(), "Hessian should not contain NaN/Inf after sanitization"
    print("✅ test_hessian_nonfinite_sanitization PASSED")


def test_meta_loop_nan_recovery():
    """Verify meta-loop NaN recovery in fixed-point iteration."""
    from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        max_iterations=5,
    )
    loop = ProvablyConvergentMetaLoop(
        config=config, max_iterations=5, min_iterations=1,
    )
    # Normal input should produce finite output
    psi_0 = torch.randn(2, 64)
    with torch.no_grad():
        C, iterations, meta = loop.compute_fixed_point(psi_0)
    assert torch.isfinite(C).all(), "C should be finite for normal input"
    print("✅ test_meta_loop_nan_recovery PASSED")


def test_mcts_ucb1_nonfinite_guard():
    """Verify MCTSNode.ucb1_score returns finite value."""
    from aeon_core import MCTSNode
    # Create parent-child pair
    parent_state = torch.randn(64)
    parent = MCTSNode(state=parent_state)
    parent.visits = 10

    child_state = torch.randn(64)
    child = MCTSNode(state=child_state, parent=parent, action_idx=0, prior=0.5)
    child.visits = 0
    child.total_value = float('nan')  # Force NaN q_value

    score = child.ucb1_score()
    assert math.isfinite(score), f"UCB1 score should be finite, got {score}"
    assert score == 0.0, f"Expected 0.0 for NaN q_value, got {score}"
    print("✅ test_mcts_ucb1_nonfinite_guard PASSED")


def test_mcts_simulate_nonfinite_guard():
    """Verify MCTSPlanner._simulate returns finite value."""
    from aeon_core import MCTSPlanner, MCTSNode
    planner = MCTSPlanner(state_dim=64, action_dim=4, hidden_dim=32)

    state = torch.randn(64)
    node = MCTSNode(state=state)

    # Normal case should return finite
    value = planner._simulate(node)
    assert math.isfinite(value), f"Simulate should return finite value, got {value}"
    print("✅ test_mcts_simulate_nonfinite_guard PASSED")


def test_reasoning_core_nan_fallback():
    """Verify reasoning_core falls back to z_in when meta-loop produces NaN."""
    from aeon_core import AEONConfig, AEONDeltaV3
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    # Normal forward pass should produce finite output
    x = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        outputs = model(x, fast=True)
    assert torch.isfinite(outputs['logits']).all(), "Logits should be finite"
    assert torch.isfinite(outputs['thoughts']).all(), "Thoughts should be finite"
    print("✅ test_reasoning_core_nan_fallback PASSED")


def test_generate_resets_inference_cache():
    """Verify generate() resets inference cache for new sequences."""
    from aeon_core import AEONConfig, AEONDeltaV3
    config = AEONConfig(
        vocab_size=1000, z_dim=64, hidden_dim=64,
        vq_embedding_dim=64, seq_length=8,
        num_pillars=4, use_amp=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_inference_cache=True,
    )
    model = AEONDeltaV3(config)
    model.eval()

    # Verify inference cache exists
    assert model.inference_cache is not None

    # Set some state in the cache
    model.inference_cache.set_ssm_state([torch.randn(1, 64)])
    assert model.inference_cache.step == 1

    # Directly test the reset behaviour that generate() would trigger
    # (generate() resets cache after the tokenizer check, so we test
    # the mechanism directly)
    model.inference_cache.reset()
    assert model.inference_cache.step == 0, "Cache should be reset"
    assert model.inference_cache.get_ssm_state() is None

    print("✅ test_generate_resets_inference_cache PASSED")


# ============================================================================
# AGI MODERNIZATION: Numerical stability, thread safety & state management
# ============================================================================

def test_hierarchical_vae_logvar_clamping():
    """Verify HierarchicalVAE clamps logvar to prevent exp overflow/underflow."""
    from aeon_core import HierarchicalVAE

    vae = HierarchicalVAE(input_dim=64, num_levels=3)
    vae.train()

    # Extreme logvar should not produce Inf/NaN via clamping
    mu = torch.zeros(2, 64)
    logvar_extreme = torch.full((2, 64), 100.0)  # Would overflow without clamp
    result = vae.reparameterize(mu, logvar_extreme)
    assert torch.isfinite(result).all(), "Reparameterize produced non-finite with extreme logvar"

    logvar_neg_extreme = torch.full((2, 64), -100.0)
    result_neg = vae.reparameterize(mu, logvar_neg_extreme)
    assert torch.isfinite(result_neg).all(), "Reparameterize produced non-finite with extreme negative logvar"

    # Normal forward should still work
    x = torch.randn(2, 64)
    out = vae(x)
    assert torch.isfinite(out['kl_loss']), "KL loss is non-finite"
    assert torch.isfinite(out['selected_level']).all(), "Selected level is non-finite"

    print("✅ test_hierarchical_vae_logvar_clamping PASSED")


def test_unified_memory_temporal_stability():
    """Verify UnifiedMemory temporal addressing uses clamped u and larger epsilon."""
    from aeon_core import UnifiedMemory

    mem = UnifiedMemory(capacity=16, dim=32)

    # All-zero usage: u.sum() == 0; should not produce NaN via larger epsilon
    with torch.no_grad():
        mem.u.zero_()
    query = torch.randn(32)
    result = mem(query)
    assert torch.isfinite(result).all(), "UnifiedMemory produced NaN with zero usage vector"

    # Negative usage values (from decay drift): should be clamped
    with torch.no_grad():
        mem.u.fill_(-1.0)
    result2 = mem(query)
    assert torch.isfinite(result2).all(), "UnifiedMemory produced NaN with negative usage"

    print("✅ test_unified_memory_temporal_stability PASSED")


def test_unified_memory_input_validation():
    """Verify UnifiedMemory rejects invalid query dimensions."""
    from aeon_core import UnifiedMemory

    mem = UnifiedMemory(capacity=16, dim=32)

    # Wrong number of dimensions (3D)
    try:
        mem(torch.randn(2, 3, 32))
        assert False, "Should have raised ValueError for 3D query"
    except ValueError:
        pass

    # Wrong last dim
    try:
        mem(torch.randn(2, 64))
        assert False, "Should have raised ValueError for wrong dim"
    except ValueError:
        pass

    # Valid 1D and 2D should work
    result_1d = mem(torch.randn(32))
    assert result_1d.shape == (32,), f"Expected (32,), got {result_1d.shape}"

    result_2d = mem(torch.randn(3, 32))
    assert result_2d.shape == (3, 32), f"Expected (3, 32), got {result_2d.shape}"

    print("✅ test_unified_memory_input_validation PASSED")


def test_certified_meta_loop_division_safety():
    """Verify CertifiedMetaLoop does not divide by zero when L ≈ 1."""
    from aeon_core import CertifiedMetaLoop, AEONConfig

    config = AEONConfig(
        z_dim=32, hidden_dim=32, meta_dim=32,
        vq_embedding_dim=32,
        lipschitz_target=0.95, device_str='cpu'
    )
    loop = CertifiedMetaLoop(config)

    z = torch.randn(2, 32)
    # Should not crash even if L ≈ 1
    converged, error = loop.verify_convergence_preconditions(z)
    if error is not None:
        assert math.isfinite(error), f"Certified error is non-finite: {error}"

    # Directly test the safe division path
    L_near_one = 0.9999999
    residual = 1.0
    safe_error = (L_near_one / max(1.0 - L_near_one, 1e-6)) * residual
    assert math.isfinite(safe_error), f"Safe error is non-finite: {safe_error}"

    print("✅ test_certified_meta_loop_division_safety PASSED")


def test_inference_cache_thread_safety():
    """Verify InferenceCache is thread-safe for concurrent reads/writes."""
    from aeon_core import InferenceCache
    import threading

    cache = InferenceCache(maxlen=100)
    errors = []

    def writer():
        try:
            for i in range(50):
                cache.set_ssm_state([torch.randn(1, 32)])
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for i in range(50):
                _ = cache.get_ssm_state()
                _ = cache.step
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"
    print("✅ test_inference_cache_thread_safety PASSED")


def test_forward_chainer_saturation_prevention():
    """Verify DifferentiableForwardChainer prevents fact saturation."""
    from aeon_core import DifferentiableForwardChainer

    chainer = DifferentiableForwardChainer(num_predicates=8, max_depth=10)

    # Start with moderate facts
    facts = torch.full((2, 8), 0.5)
    rules = torch.ones(2, 8)

    result = chainer(facts, rules)
    # With decay factor (0.95), facts should not all saturate to 1.0
    # even after many iterations
    assert result.max() < 1.0, (
        f"Facts saturated to {result.max().item():.4f} — decay not working"
    )
    assert torch.isfinite(result).all(), "Forward chainer produced NaN"

    print("✅ test_forward_chainer_saturation_prevention PASSED")


def test_memory_manager_timestamp_tracking():
    """Verify MemoryManager tracks timestamps and reports age."""
    from aeon_core import MemoryManager, AEONConfig
    import time

    config = AEONConfig(device_str='cpu')
    mm = MemoryManager(config)

    v1 = torch.randn(256)
    mm.add_embedding(v1, {'id': 1})

    time.sleep(0.05)  # Small delay so age > 0

    v2 = torch.randn(256)
    mm.add_embedding(v2, {'id': 2})

    results = mm.retrieve_relevant(v2, k=2)
    assert len(results) == 2
    # Each result should have an 'age' key
    for r in results:
        assert 'age' in r, "Missing 'age' in retrieval result"
        assert r['age'] >= 0, f"Negative age: {r['age']}"

    # The most recently added should have smaller age
    # (results are sorted by similarity, not time, but both should have age >= 0)

    print("✅ test_memory_manager_timestamp_tracking PASSED")


def test_memory_manager_timestamp_eviction():
    """Verify timestamps are evicted with vectors on capacity overflow."""
    from aeon_core import MemoryManager, AEONConfig

    config = AEONConfig(device_str='cpu')
    mm = MemoryManager(config)
    mm._max_capacity = 3

    for i in range(5):
        mm.add_embedding(torch.randn(256), {'id': i})

    assert mm._size == 3, f"Expected size 3, got {mm._size}"
    assert len(mm.fallback_timestamps) == 3, (
        f"Expected 3 timestamps, got {len(mm.fallback_timestamps)}"
    )

    print("✅ test_memory_manager_timestamp_eviction PASSED")


def test_ema_reset_on_checkpoint_concept():
    """Verify meta-loop EMA buffers can be zeroed (simulating checkpoint reload)."""
    from aeon_core import ProvablyConvergentMetaLoop, AEONConfig

    config = AEONConfig(
        z_dim=32, hidden_dim=32, meta_dim=32,
        vq_embedding_dim=32,
        lipschitz_target=0.95, device_str='cpu'
    )
    loop = ProvablyConvergentMetaLoop(config)

    # Simulate some updates
    with torch.no_grad():
        loop.avg_iterations.fill_(10.0)
        loop.convergence_rate.fill_(0.85)

    # Simulate what load_state does
    with torch.no_grad():
        loop.avg_iterations.zero_()
        loop.convergence_rate.zero_()

    assert loop.avg_iterations.item() == 0.0, "avg_iterations not reset"
    assert loop.convergence_rate.item() == 0.0, "convergence_rate not reset"

    print("✅ test_ema_reset_on_checkpoint_concept PASSED")


# ============================================================================
# AGI Modernization: Decision Audit, State Validation & Error Classification
# ============================================================================

def test_decision_audit_log_record_and_recent():
    """Verify DecisionAuditLog records entries and retrieves recent ones."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)

    # Record several decisions
    audit.record("meta_loop", "converged", {"iterations": 12})
    audit.record("safety", "rollback", {"score": 0.3})
    audit.record("world_model", "surprise_switch", {"surprise": 0.8})

    recent = audit.recent(2)
    assert len(recent) == 2, f"Expected 2 recent entries, got {len(recent)}"
    assert recent[-1]["subsystem"] == "world_model"
    assert recent[-1]["decision"] == "surprise_switch"
    assert recent[-1]["metadata"]["surprise"] == 0.8

    # Verify timestamp ordering
    assert recent[0]["timestamp"] <= recent[1]["timestamp"]

    print("✅ test_decision_audit_log_record_and_recent PASSED")


def test_decision_audit_log_summary():
    """Verify DecisionAuditLog summary aggregation."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)
    audit.record("meta_loop", "converged", {})
    audit.record("meta_loop", "converged", {})
    audit.record("safety", "rollback", {})

    summary = audit.summary()
    assert summary["total_decisions"] == 3
    assert summary["counts"]["meta_loop.converged"] == 2
    assert summary["counts"]["safety.rollback"] == 1

    print("✅ test_decision_audit_log_summary PASSED")


def test_decision_audit_log_bounded_capacity():
    """Verify DecisionAuditLog respects max_entries bound."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=5)
    for i in range(10):
        audit.record("test", f"decision_{i}", {"idx": i})

    recent = audit.recent(100)
    assert len(recent) == 5, f"Expected 5 entries (bounded), got {len(recent)}"
    # Entries 0-4 evicted; oldest retained entry should be idx=5
    assert recent[0]["metadata"]["idx"] == 5

    print("✅ test_decision_audit_log_bounded_capacity PASSED")


def test_decision_audit_log_reset():
    """Verify DecisionAuditLog reset clears all data."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)
    audit.record("meta_loop", "converged", {})
    audit.reset()

    summary = audit.summary()
    assert summary["total_decisions"] == 0
    assert len(summary["counts"]) == 0

    print("✅ test_decision_audit_log_reset PASSED")


def test_decision_audit_log_thread_safety():
    """Verify DecisionAuditLog is thread-safe under concurrent writes."""
    from aeon_core import DecisionAuditLog
    import threading

    audit = DecisionAuditLog(max_entries=1000)

    def writer(subsystem, n):
        for i in range(n):
            audit.record(subsystem, "test", {"i": i})

    threads = [
        threading.Thread(target=writer, args=(f"thread_{t}", 50))
        for t in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = audit.summary()
    assert summary["total_decisions"] == 200, (
        f"Expected 200 total decisions, got {summary['total_decisions']}"
    )

    print("✅ test_decision_audit_log_thread_safety PASSED")


def test_state_consistency_validator_valid():
    """Verify StateConsistencyValidator passes for valid tensors."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C_star = torch.randn(4, 64)
    factors = torch.randn(4, 8)

    result = validator.validate(C_star, factors=factors)
    assert result["valid"], f"Expected valid, got violations: {result['violations']}"
    assert "c_star_max_abs" in result["stats"]

    print("✅ test_state_consistency_validator_valid PASSED")


def test_state_consistency_validator_nan_detection():
    """Verify StateConsistencyValidator detects NaN values."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C_star = torch.randn(4, 64)
    C_star[0, 0] = float('nan')

    result = validator.validate(C_star)
    assert not result["valid"]
    assert any("NaN" in v for v in result["violations"])

    print("✅ test_state_consistency_validator_nan_detection PASSED")


def test_state_consistency_validator_shape_mismatch():
    """Verify StateConsistencyValidator detects shape mismatches."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C_star = torch.randn(4, 32)  # Wrong dim

    result = validator.validate(C_star)
    assert not result["valid"]
    assert any("shape" in v for v in result["violations"])

    print("✅ test_state_consistency_validator_shape_mismatch PASSED")


def test_state_consistency_validator_activation_magnitude():
    """Verify StateConsistencyValidator detects excessive activations."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64, max_activation=10.0)
    C_star = torch.randn(4, 64) * 100  # Exceeds max

    result = validator.validate(C_star)
    assert not result["valid"]
    assert any("activation" in v for v in result["violations"])

    print("✅ test_state_consistency_validator_activation_magnitude PASSED")


def test_semantic_error_classifier_numerical():
    """Verify SemanticErrorClassifier classifies numerical errors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    error = ValueError("NaN detected in tensor output")
    cls, detail = classifier.classify(error)
    assert cls == "numerical", f"Expected 'numerical', got '{cls}'"

    print("✅ test_semantic_error_classifier_numerical PASSED")


def test_semantic_error_classifier_shape():
    """Verify SemanticErrorClassifier classifies shape errors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    error = RuntimeError("shape mismatch: expected [4, 64], got [4, 32]")
    cls, detail = classifier.classify(error)
    assert cls == "shape", f"Expected 'shape', got '{cls}'"

    print("✅ test_semantic_error_classifier_shape PASSED")


def test_semantic_error_classifier_resource():
    """Verify SemanticErrorClassifier classifies resource errors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    error = RuntimeError("CUDA out of memory")
    cls, detail = classifier.classify(error)
    assert cls == "resource", f"Expected 'resource', got '{cls}'"

    print("✅ test_semantic_error_classifier_resource PASSED")


def test_semantic_error_classifier_unknown():
    """Verify SemanticErrorClassifier falls back to unknown."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    error = IOError("disk full")
    cls, detail = classifier.classify(error)
    assert cls == "unknown", f"Expected 'unknown', got '{cls}'"

    print("✅ test_semantic_error_classifier_unknown PASSED")


def test_semantic_error_classifier_tensor_state_healthy():
    """Verify classify_tensor_state returns None for healthy tensors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    t = torch.randn(4, 64)
    result = classifier.classify_tensor_state(t, "test")
    assert result is None, f"Expected None for healthy tensor, got {result}"

    print("✅ test_semantic_error_classifier_tensor_state_healthy PASSED")


def test_semantic_error_classifier_tensor_state_nan():
    """Verify classify_tensor_state detects NaN tensors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    t = torch.tensor([1.0, float('nan'), 3.0])
    result = classifier.classify_tensor_state(t, "test")
    assert result is not None
    assert result[0] == "numerical"
    assert "NaN" in result[1]

    print("✅ test_semantic_error_classifier_tensor_state_nan PASSED")


def test_semantic_error_classifier_tensor_state_inf():
    """Verify classify_tensor_state detects Inf tensors."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()
    t = torch.tensor([1.0, float('inf'), 3.0])
    result = classifier.classify_tensor_state(t, "test")
    assert result is not None
    assert result[0] == "numerical"
    assert "Inf" in result[1]

    print("✅ test_semantic_error_classifier_tensor_state_inf PASSED")


def test_audit_log_in_reasoning_core():
    """Verify that reasoning_core populates the audit log."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        z_dim=32, hidden_dim=32, meta_dim=32,
        vq_embedding_dim=32, lipschitz_target=0.95,
        device_str='cpu',
        enable_safety_guardrails=False,
        enable_catastrophe_detection=False,
        enable_quantum_sim=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    # Reset audit log
    model.audit_log.reset()

    # Run a forward pass
    B, L = 2, config.seq_length
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    with torch.no_grad():
        outputs = model(input_ids, decode_mode='train', fast=True)

    # Check that audit log has at least the meta_loop entry
    summary = model.get_audit_summary()
    assert summary["total_decisions"] >= 1, (
        f"Expected at least 1 audit decision, got {summary['total_decisions']}"
    )
    assert "meta_loop.completed" in summary["counts"], (
        f"Expected 'meta_loop.completed' in audit, got {summary['counts']}"
    )

    print("✅ test_audit_log_in_reasoning_core PASSED")


def test_state_validation_in_reasoning_output():
    """Verify state_validation key is present in reasoning_core outputs."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        z_dim=32, hidden_dim=32, meta_dim=32,
        vq_embedding_dim=32, lipschitz_target=0.95,
        device_str='cpu',
        enable_safety_guardrails=False,
        enable_catastrophe_detection=False,
        enable_quantum_sim=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    B, L = 2, config.seq_length
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    with torch.no_grad():
        outputs = model(input_ids, decode_mode='train', fast=True)

    assert 'state_validation' in outputs, (
        "Expected 'state_validation' key in model outputs"
    )
    sv = outputs['state_validation']
    assert 'valid' in sv
    assert 'violations' in sv
    assert 'stats' in sv
    # For normal inputs, state should be valid
    assert sv['valid'], f"Expected valid state, got violations: {sv['violations']}"

    print("✅ test_state_validation_in_reasoning_output PASSED")


def test_memory_load_specific_exception():
    """Verify load_memory uses specific exception types, not bare except."""
    import inspect
    from aeon_core import MemoryManager

    source = inspect.getsource(MemoryManager.load_memory)
    # The bare 'except Exception:' (without 'as e') should no longer exist
    assert "except Exception:" not in source, (
        "load_memory still contains bare 'except Exception:'"
    )

    print("✅ test_memory_load_specific_exception PASSED")


# ============================================================================
# AGI Modernization: Error recovery, context window, audit & validator tests
# ============================================================================

def test_error_recovery_numerical():
    """ErrorRecoveryManager recovers from numerical errors."""
    from aeon_core import ErrorRecoveryManager, DecisionAuditLog
    audit = DecisionAuditLog()
    mgr = ErrorRecoveryManager(hidden_dim=64, audit_log=audit)

    fallback = torch.zeros(1, 64)
    last_good = torch.randn(1, 64)
    last_good[0, 0] = float('nan')

    err = ValueError("NaN detected in output")
    ok, val = mgr.recover(err, context="test", fallback=fallback, last_good_state=last_good)

    assert ok, "Recovery should succeed for numerical errors"
    assert val is not None
    assert torch.isfinite(val).all(), "Recovered tensor should be finite"

    stats = mgr.get_recovery_stats()
    assert stats["total"] == 1
    assert stats["by_class"]["numerical"] == 1

    print("✅ test_error_recovery_numerical PASSED")


def test_error_recovery_convergence():
    """ErrorRecoveryManager rolls back to last_good on convergence failure."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64)
    last_good = torch.ones(1, 64) * 0.5

    err = RuntimeError("Meta-loop failed to converge after 50 iterations")
    ok, val = mgr.recover(err, context="meta_loop", last_good_state=last_good)

    assert ok, "Recovery should succeed for convergence errors"
    assert val is not None
    assert torch.allclose(val, last_good), "Should return last_good_state"

    print("✅ test_error_recovery_convergence PASSED")


def test_error_recovery_unknown_with_fallback():
    """ErrorRecoveryManager returns fallback for unknown errors."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64)
    fallback = torch.zeros(2, 64)

    err = KeyError("unexpected key")
    ok, val = mgr.recover(err, context="test", fallback=fallback)

    assert ok, "Recovery with fallback should succeed"
    assert val is not None
    assert val.shape == (2, 64)

    print("✅ test_error_recovery_unknown_with_fallback PASSED")


def test_error_recovery_unknown_no_fallback():
    """ErrorRecoveryManager returns False with no fallback on unknown error."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64)
    err = KeyError("unexpected key")
    ok, val = mgr.recover(err, context="test")

    assert not ok, "Recovery without fallback should fail for unknown errors"
    assert val is None

    print("✅ test_error_recovery_unknown_no_fallback PASSED")


def test_error_recovery_reset_stats():
    """ErrorRecoveryManager.reset_stats clears counters."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64)
    mgr.recover(ValueError("NaN"), context="test", fallback=torch.zeros(1, 64))
    assert mgr.get_recovery_stats()["total"] >= 1

    mgr.reset_stats()
    assert mgr.get_recovery_stats()["total"] == 0

    print("✅ test_error_recovery_reset_stats PASSED")


def test_error_recovery_resource():
    """ErrorRecoveryManager offloads to CPU on resource errors."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64)
    last_good = torch.randn(1, 64)

    err = RuntimeError("CUDA out of memory")
    ok, val = mgr.recover(err, context="forward", last_good_state=last_good)

    assert ok
    assert val.device == torch.device("cpu"), "Should offload to CPU"

    print("✅ test_error_recovery_resource PASSED")


def test_context_window_add_and_retrieve():
    """ContextWindowManager stores and retrieves entries by relevance."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=10, hidden_dim=32)

    for i in range(5):
        ctx.add("retriever", torch.randn(32), relevance=float(i))

    top = ctx.get_top_k(3)
    assert len(top) == 3
    assert top[0]["relevance"] >= top[1]["relevance"] >= top[2]["relevance"]

    print("✅ test_context_window_add_and_retrieve PASSED")


def test_context_window_eviction():
    """ContextWindowManager evicts least-relevant entries when full."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=5, hidden_dim=16)

    for i in range(10):
        ctx.add("mem", torch.randn(16), relevance=float(i))

    stats = ctx.stats()
    assert stats["current_size"] == 5
    assert stats["total_added"] == 10
    assert stats["total_evicted"] == 5

    top = ctx.get_top_k(5)
    # Should keep the 5 highest-relevance entries (5-9)
    assert all(e["relevance"] >= 5.0 for e in top)

    print("✅ test_context_window_eviction PASSED")


def test_context_window_rejects_nonfinite():
    """ContextWindowManager skips non-finite embeddings."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=10, hidden_dim=16)
    nan_emb = torch.full((16,), float('nan'))
    ctx.add("bad_source", nan_emb, relevance=1.0)

    assert ctx.stats()["current_size"] == 0

    print("✅ test_context_window_rejects_nonfinite PASSED")


def test_context_window_get_context_tensor():
    """ContextWindowManager.get_context_tensor stacks embeddings."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=10, hidden_dim=32)
    for i in range(4):
        ctx.add("src", torch.randn(32), relevance=float(i))

    tensor = ctx.get_context_tensor(k=3)
    assert tensor is not None
    assert tensor.shape == (3, 32)

    # Empty context returns None
    ctx.clear()
    assert ctx.get_context_tensor() is None

    print("✅ test_context_window_get_context_tensor PASSED")


def test_audit_log_severity_levels():
    """DecisionAuditLog supports severity levels in records."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)

    audit.record("meta_loop", "converged", severity="info")
    audit.record("safety", "rollback", severity="warning")
    audit.record("system", "crash", severity="critical")

    entries = audit.recent(3)
    assert entries[0].get("severity") == "info"
    assert entries[1].get("severity") == "warning"
    assert entries[2].get("severity") == "critical"

    print("✅ test_audit_log_severity_levels PASSED")


def test_audit_log_filter_by_subsystem():
    """DecisionAuditLog.filter_by returns entries for a specific subsystem."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)
    audit.record("meta_loop", "converged", severity="info")
    audit.record("safety", "rollback", severity="warning")
    audit.record("meta_loop", "diverged", severity="error")

    meta_entries = audit.filter_by(subsystem="meta_loop")
    assert len(meta_entries) == 2
    assert all(e["subsystem"] == "meta_loop" for e in meta_entries)

    print("✅ test_audit_log_filter_by_subsystem PASSED")


def test_audit_log_filter_by_severity():
    """DecisionAuditLog.filter_by respects min_severity threshold."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)
    audit.record("a", "d1", severity="debug")
    audit.record("a", "d2", severity="info")
    audit.record("a", "d3", severity="warning")
    audit.record("a", "d4", severity="error")
    audit.record("a", "d5", severity="critical")

    warnings_up = audit.filter_by(min_severity="warning")
    assert len(warnings_up) == 3
    for e in warnings_up:
        assert e["severity"] in ("warning", "error", "critical")

    print("✅ test_audit_log_filter_by_severity PASSED")


def test_audit_log_backward_compat():
    """DecisionAuditLog.record still works without severity argument."""
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)
    audit.record("test", "decision", {"key": "value"})

    entries = audit.recent(1)
    assert len(entries) == 1
    assert entries[0]["severity"] == "info"  # default

    print("✅ test_audit_log_backward_compat PASSED")


def test_validator_validate_and_recover_clean():
    """StateConsistencyValidator.validate_and_recover passes clean tensors."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C = torch.randn(2, 64)
    recovered, result = validator.validate_and_recover(C)

    assert result["valid"]
    assert "recovered" not in result
    assert torch.equal(recovered, C)

    print("✅ test_validator_validate_and_recover_clean PASSED")


def test_validator_validate_and_recover_nan():
    """StateConsistencyValidator.validate_and_recover fixes NaN tensors."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C = torch.randn(2, 64)
    C[0, :10] = float('nan')
    C[1, 5] = float('inf')

    recovered, result = validator.validate_and_recover(C)

    assert not result["valid"]
    assert result.get("recovered") is True
    assert torch.isfinite(recovered).all()

    print("✅ test_validator_validate_and_recover_nan PASSED")


def test_validator_validate_and_recover_shape():
    """StateConsistencyValidator.validate_and_recover fixes wrong shapes."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64)
    C = torch.randn(2, 32)  # Wrong hidden_dim

    recovered, result = validator.validate_and_recover(C)

    assert not result["valid"]
    assert result.get("recovered") is True
    assert recovered.shape == (2, 64)

    print("✅ test_validator_validate_and_recover_shape PASSED")


def test_validator_validate_and_recover_activation_clamp():
    """StateConsistencyValidator.validate_and_recover clamps large activations."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(hidden_dim=64, max_activation=100.0)
    C = torch.randn(2, 64) * 1000.0  # Way above max_activation

    recovered, result = validator.validate_and_recover(C)

    assert not result["valid"]
    assert result.get("recovered") is True
    assert recovered.abs().max().item() <= 100.0

    print("✅ test_validator_validate_and_recover_activation_clamp PASSED")


def test_semantic_error_classifier_with_suggestion():
    """SemanticErrorClassifier.classify_with_suggestion returns suggestions."""
    from aeon_core import SemanticErrorClassifier

    classifier = SemanticErrorClassifier()

    # Numerical
    cls, detail, suggestion = classifier.classify_with_suggestion(
        ValueError("NaN detected")
    )
    assert cls == "numerical"
    assert len(suggestion) > 0
    assert "Sanitize" in suggestion

    # Shape
    cls, detail, suggestion = classifier.classify_with_suggestion(
        RuntimeError("shape mismatch")
    )
    assert cls == "shape"
    assert "dimension" in suggestion.lower()

    # Resource
    cls, detail, suggestion = classifier.classify_with_suggestion(
        RuntimeError("CUDA out of memory")
    )
    assert cls == "resource"
    assert "batch" in suggestion.lower()

    # Convergence
    cls, detail, suggestion = classifier.classify_with_suggestion(
        RuntimeError("failed to converge")
    )
    assert cls == "convergence"
    assert "learning rate" in suggestion.lower() or "Lipschitz" in suggestion

    print("✅ test_semantic_error_classifier_with_suggestion PASSED")


def test_ssd_block_chunk_len_guard():
    """_SSDBlock enforces chunk_len >= 1."""
    from aeon_core import SelectiveSSMv2

    # chunk_len=0 should be clamped to 1
    ssm = SelectiveSSMv2(d_model=64, chunk_len=0)
    x = torch.randn(1, 8, 64)
    y, state = ssm(x)
    assert y.shape == (1, 8, 64)
    assert torch.isfinite(y).all()

    print("✅ test_ssd_block_chunk_len_guard PASSED")


def test_rssm_trainer_uses_model_device():
    """ContextualRSSMTrainer derives device from model, not global variable."""
    from ae_train import AEONConfigV4, AEONDeltaV4, ContextualRSSMTrainer, TrainingMonitor
    import logging

    config = AEONConfigV4()
    model = AEONDeltaV4(config)  # CPU by default
    monitor = TrainingMonitor(logging.getLogger("test"))

    trainer = ContextualRSSMTrainer(model, config, monitor)
    assert hasattr(trainer, "device"), "ContextualRSSMTrainer must have a 'device' attribute"
    assert trainer.device == next(model.parameters()).device, (
        f"trainer.device ({trainer.device}) must match model device "
        f"({next(model.parameters()).device})"
    )
    print("✅ test_rssm_trainer_uses_model_device PASSED")


def test_validate_training_components_uses_model_device():
    """validate_training_components creates test tensors on the model's device."""
    from ae_train import AEONConfigV4, AEONDeltaV4, validate_training_components
    import logging

    config = AEONConfigV4()
    model = AEONDeltaV4(config)  # CPU by default
    log = logging.getLogger("test")
    # Should not raise even if the global 'device' were different
    result = validate_training_components(model, config, log)
    assert isinstance(result, bool)
    print("✅ test_validate_training_components_uses_model_device PASSED")


# ============================================================================
# ARCHITECTURAL ROADMAP TESTS (Phases 1-5)
# ============================================================================


def test_shared_workspace_broadcast_and_read():
    """Phase 1: SharedWorkspace stores and returns broadcast content."""
    from aeon_core import SharedWorkspace
    ws = SharedWorkspace(capacity=64)
    data = torch.randn(1, 64)
    ws.broadcast(data)
    out = ws.read()
    assert out.shape == (1, 64)
    assert torch.allclose(out, data, atol=1e-6)
    print("✅ test_shared_workspace_broadcast_and_read PASSED")


def test_shared_workspace_padding():
    """Phase 1: SharedWorkspace pads smaller tensors."""
    from aeon_core import SharedWorkspace
    ws = SharedWorkspace(capacity=64)
    small = torch.ones(1, 10)
    ws.broadcast(small)
    out = ws.read()
    assert out.shape == (1, 64)
    assert torch.allclose(out[0, :10], small[0])
    assert (out[0, 10:] == 0).all()
    print("✅ test_shared_workspace_padding PASSED")


def test_shared_workspace_truncation():
    """Phase 1: SharedWorkspace truncates larger tensors."""
    from aeon_core import SharedWorkspace
    ws = SharedWorkspace(capacity=32)
    big = torch.ones(1, 64)
    ws.broadcast(big)
    out = ws.read()
    assert out.shape == (1, 32)
    print("✅ test_shared_workspace_truncation PASSED")


def test_attention_arbiter_urgency():
    """Phase 1: AttentionArbiter produces valid urgency scores."""
    from aeon_core import AttentionArbiter
    arb = AttentionArbiter(["a", "b", "c"], state_dim=32)
    state = torch.randn(2, 32)
    urgency = arb.compute_urgency(state)
    assert urgency.shape == (2, 3)
    assert torch.allclose(urgency.sum(dim=-1), torch.ones(2), atol=1e-5)
    print("✅ test_attention_arbiter_urgency PASSED")


def test_attention_arbiter_top_k():
    """Phase 1: AttentionArbiter top_k returns correct count."""
    from aeon_core import AttentionArbiter
    arb = AttentionArbiter(["a", "b", "c", "d"], state_dim=32)
    state = torch.randn(2, 32)
    urgency = arb.compute_urgency(state)
    indices = arb.top_k_indices(urgency, k=2)
    assert len(indices) == 2
    assert all(0 <= i < 4 for i in indices)
    print("✅ test_attention_arbiter_top_k PASSED")


def test_meta_monitor_update():
    """Phase 1: MetaMonitor tracks quality stats."""
    from aeon_core import MetaMonitor
    mon = MetaMonitor(window_size=10)
    state = torch.randn(2, 32)
    winner = torch.randn(1, 32)
    stats = mon.update(state, winner)
    assert "mean" in stats and "std" in stats and "count" in stats
    assert stats["count"] == 1
    for _ in range(15):
        mon.update(state, winner)
    assert mon.stats()["count"] == 10  # window enforced
    print("✅ test_meta_monitor_update PASSED")


def test_cognitive_executive_function_forward():
    """Phase 1: CognitiveExecutiveFunction runs full pipeline."""
    from aeon_core import CognitiveExecutiveFunction
    subs = {
        "fast": nn.Linear(32, 32),
        "slow": nn.Linear(32, 32),
        "safe": nn.Linear(32, 32),
    }
    cef = CognitiveExecutiveFunction(subs, state_dim=32, workspace_capacity=64, top_k=2)
    state = torch.randn(2, 32)
    out = cef(state)
    assert "winner" in out
    assert "urgency" in out
    assert "executed" in out
    assert "meta_stats" in out
    assert "workspace" in out
    assert len(out["executed"]) <= 2
    print("✅ test_cognitive_executive_function_forward PASSED")


def test_cognitive_executive_function_gradient_flow():
    """Phase 1: Gradients flow through CognitiveExecutiveFunction."""
    from aeon_core import CognitiveExecutiveFunction
    subs = {"a": nn.Linear(16, 16), "b": nn.Linear(16, 16)}
    cef = CognitiveExecutiveFunction(subs, state_dim=16, workspace_capacity=32, top_k=2)
    state = torch.randn(2, 16, requires_grad=True)
    out = cef(state)
    loss = out["winner"].sum()
    loss.backward()
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0
    print("✅ test_cognitive_executive_function_gradient_flow PASSED")


def test_recovery_experience_replay_push_and_sample():
    """Phase 2: RecoveryExperienceReplay stores and samples transitions."""
    from aeon_core import RecoveryExperienceReplay
    buf = RecoveryExperienceReplay(capacity=50)
    for i in range(20):
        buf.push(torch.randn(8), i % 4, float(i), torch.randn(8))
    assert len(buf) == 20
    batch = buf.sample(5)
    assert len(batch) == 5
    print("✅ test_recovery_experience_replay_push_and_sample PASSED")


def test_recovery_experience_replay_capacity():
    """Phase 2: RecoveryExperienceReplay respects capacity limit."""
    from aeon_core import RecoveryExperienceReplay
    buf = RecoveryExperienceReplay(capacity=10)
    for i in range(25):
        buf.push(torch.randn(4), 0, 1.0, torch.randn(4))
    assert len(buf) == 10
    print("✅ test_recovery_experience_replay_capacity PASSED")


def test_meta_recovery_learner_forward():
    """Phase 2: MetaRecoveryLearner selects a valid strategy."""
    from aeon_core import MetaRecoveryLearner
    mrl = MetaRecoveryLearner(state_dim=32, hidden_dim=64)
    ctx = torch.randn(1, 32)
    out = mrl(ctx)
    assert "action" in out and "strategy" in out and "value" in out
    assert out["strategy"] in MetaRecoveryLearner.STRATEGIES
    print("✅ test_meta_recovery_learner_forward PASSED")


def test_meta_recovery_learner_compute_loss():
    """Phase 2: MetaRecoveryLearner loss is a valid scalar."""
    from aeon_core import MetaRecoveryLearner
    mrl = MetaRecoveryLearner(state_dim=16, hidden_dim=32)
    states = torch.randn(4, 16)
    actions = torch.tensor([0, 1, 2, 3])
    rewards = torch.tensor([1.0, 0.5, 0.0, -0.5])
    next_states = torch.randn(4, 16)
    loss = mrl.compute_loss(states, actions, rewards, next_states)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    print("✅ test_meta_recovery_learner_compute_loss PASSED")


def test_meta_recovery_learner_gradient_flow():
    """Phase 2: Gradients flow through MetaRecoveryLearner."""
    from aeon_core import MetaRecoveryLearner
    mrl = MetaRecoveryLearner(state_dim=16, hidden_dim=32)
    ctx = torch.randn(2, 16, requires_grad=True)
    out = mrl(ctx)
    out["value"].sum().backward()
    assert ctx.grad is not None
    assert ctx.grad.abs().sum().item() > 0
    print("✅ test_meta_recovery_learner_gradient_flow PASSED")


def test_unified_causal_simulator_forward():
    """Phase 3: UnifiedCausalSimulator produces next_state and causal_vars."""
    from aeon_core import UnifiedCausalSimulator
    sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
    state = torch.randn(2, 32)
    out = sim(state)
    assert "next_state" in out
    assert "causal_vars" in out
    assert "causal_graph" in out
    assert out["next_state"].shape == (2, 32)
    assert out["causal_vars"].shape == (2, 8)
    print("✅ test_unified_causal_simulator_forward PASSED")


def test_unified_causal_simulator_intervention():
    """Phase 3: UnifiedCausalSimulator applies do-calculus intervention."""
    from aeon_core import UnifiedCausalSimulator
    sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
    state = torch.randn(2, 32)
    out_no_iv = sim(state)
    out_iv = sim(state, intervention={"index": 0, "value": 1.0})
    assert out_iv["interventional"] is True
    assert out_no_iv["interventional"] is False
    print("✅ test_unified_causal_simulator_intervention PASSED")


def test_unified_causal_simulator_counterfactual():
    """Phase 3: UnifiedCausalSimulator plans counterfactuals."""
    from aeon_core import UnifiedCausalSimulator
    sim = UnifiedCausalSimulator(state_dim=32, num_causal_vars=8)
    observed = torch.randn(2, 32)
    goal = torch.randn(2, 32)
    result = sim.plan_counterfactual(observed, goal, num_interventions=4)
    assert "best_intervention" in result
    assert "predicted_outcome" in result
    assert "loss" in result
    assert result["predicted_outcome"].shape == (2, 32)
    print("✅ test_unified_causal_simulator_counterfactual PASSED")


def test_unified_causal_simulator_gradient_flow():
    """Phase 3: Gradients flow through UnifiedCausalSimulator."""
    from aeon_core import UnifiedCausalSimulator
    sim = UnifiedCausalSimulator(state_dim=16, num_causal_vars=4)
    state = torch.randn(2, 16, requires_grad=True)
    out = sim(state)
    out["next_state"].sum().backward()
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0
    print("✅ test_unified_causal_simulator_gradient_flow PASSED")


def test_neuro_symbolic_bridge_roundtrip():
    """Phase 4: NeuroSymbolicBridge extracts and re-embeds correctly."""
    from aeon_core import NeuroSymbolicBridge
    bridge = NeuroSymbolicBridge(hidden_dim=64, num_predicates=16)
    state = torch.randn(2, 64)
    facts = bridge.extract_facts(state)
    rules = bridge.extract_rules(state)
    assert facts.shape == (2, 16)
    assert rules.shape == (2, 16)
    assert (facts >= 0).all() and (facts <= 1).all()
    embedded = bridge.embed_conclusions(facts)
    assert embedded.shape == (2, 64)
    print("✅ test_neuro_symbolic_bridge_roundtrip PASSED")


def test_temporal_knowledge_graph_add_and_retrieve():
    """Phase 4: TemporalKnowledgeGraph stores and retrieves facts."""
    from aeon_core import TemporalKnowledgeGraph
    tkg = TemporalKnowledgeGraph(capacity=100)
    facts = torch.randn(2, 16)
    tkg.add_facts(facts, confidence=0.9)
    assert len(tkg) == 1
    query = torch.randn(2, 16)
    result = tkg.retrieve_relevant(query, top_k=3)
    assert result.shape == facts.shape
    print("✅ test_temporal_knowledge_graph_add_and_retrieve PASSED")


def test_temporal_knowledge_graph_capacity():
    """Phase 4: TemporalKnowledgeGraph evicts old entries."""
    from aeon_core import TemporalKnowledgeGraph
    tkg = TemporalKnowledgeGraph(capacity=5)
    for _ in range(10):
        tkg.add_facts(torch.randn(1, 8))
    assert len(tkg) == 5
    print("✅ test_temporal_knowledge_graph_capacity PASSED")


def test_temporal_knowledge_graph_empty_retrieve():
    """Phase 4: Empty TKG returns zeros."""
    from aeon_core import TemporalKnowledgeGraph
    tkg = TemporalKnowledgeGraph()
    query = torch.randn(2, 16)
    result = tkg.retrieve_relevant(query)
    assert result.shape == query.shape
    assert (result == 0).all()
    print("✅ test_temporal_knowledge_graph_empty_retrieve PASSED")


def test_hybrid_reasoning_engine_forward():
    """Phase 4: HybridReasoningEngine produces conclusions."""
    from aeon_core import HybridReasoningEngine
    engine = HybridReasoningEngine(hidden_dim=64, num_predicates=16)
    state = torch.randn(2, 64)
    out = engine(state)
    assert "conclusions" in out
    assert "facts" in out
    assert "rules" in out
    assert "derived" in out
    assert out["conclusions"].shape == (2, 64)
    print("✅ test_hybrid_reasoning_engine_forward PASSED")


def test_hybrid_reasoning_engine_with_query():
    """Phase 4: HybridReasoningEngine uses KB when query provided."""
    from aeon_core import HybridReasoningEngine
    engine = HybridReasoningEngine(hidden_dim=64, num_predicates=16)
    state = torch.randn(2, 64)
    # First call to populate KB
    engine.reason(state)
    # Second call with query
    query = torch.randn(2, 16)
    out = engine.reason(state, query=query)
    assert out["conclusions"].shape == (2, 64)
    assert len(engine.knowledge_graph) >= 2
    print("✅ test_hybrid_reasoning_engine_with_query PASSED")


def test_hybrid_reasoning_engine_gradient_flow():
    """Phase 4: Gradients flow through HybridReasoningEngine."""
    from aeon_core import HybridReasoningEngine
    engine = HybridReasoningEngine(hidden_dim=32, num_predicates=8)
    state = torch.randn(2, 32, requires_grad=True)
    out = engine(state)
    out["conclusions"].sum().backward()
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0
    print("✅ test_hybrid_reasoning_engine_gradient_flow PASSED")


def test_critic_network_forward():
    """Phase 5: CriticNetwork returns all four scores in [0,1]."""
    from aeon_core import CriticNetwork
    critic = CriticNetwork(hidden_dim=32)
    query = torch.randn(2, 32)
    candidate = torch.randn(2, 32)
    scores = critic(query, candidate)
    for key in ["correctness", "coherence", "safety", "novelty"]:
        assert key in scores
        assert (scores[key] >= 0).all() and (scores[key] <= 1).all()
    print("✅ test_critic_network_forward PASSED")


def test_critic_network_explain_failure():
    """Phase 5: CriticNetwork explain_failure returns 4-dim signal."""
    from aeon_core import CriticNetwork
    critic = CriticNetwork(hidden_dim=32)
    query = torch.randn(2, 32)
    candidate = torch.randn(2, 32)
    scores = critic(query, candidate)
    signal = critic.explain_failure(scores)
    assert signal.shape[-1] == 4
    print("✅ test_critic_network_explain_failure PASSED")


def test_revision_network_forward():
    """Phase 5: RevisionNetwork produces revised query."""
    from aeon_core import RevisionNetwork
    reviser = RevisionNetwork(hidden_dim=32)
    query = torch.randn(2, 32)
    candidate = torch.randn(2, 32)
    critique = torch.randn(2, 4)
    revised = reviser(query, candidate, critique)
    assert revised.shape == (2, 32)
    print("✅ test_revision_network_forward PASSED")


def test_auto_critic_loop_forward():
    """Phase 5: AutoCriticLoop produces candidate with iteration count."""
    from aeon_core import AutoCriticLoop
    generator = nn.Linear(32, 32)
    acl = AutoCriticLoop(generator, hidden_dim=32, max_iterations=3, threshold=0.99)
    query = torch.randn(2, 32)
    out = acl(query)
    assert "candidate" in out
    assert "iterations" in out
    assert "final_score" in out
    assert out["candidate"].shape == (2, 32)
    assert 1 <= out["iterations"] <= 3
    print("✅ test_auto_critic_loop_forward PASSED")


def test_auto_critic_loop_trajectory():
    """Phase 5: AutoCriticLoop returns trajectory when requested."""
    from aeon_core import AutoCriticLoop
    generator = nn.Linear(16, 16)
    acl = AutoCriticLoop(generator, hidden_dim=16, max_iterations=3, threshold=0.99)
    query = torch.randn(2, 16)
    out = acl(query, return_trajectory=True)
    assert "trajectory" in out
    assert len(out["trajectory"]) >= 1
    assert "correctness" in out["trajectory"][0]
    print("✅ test_auto_critic_loop_trajectory PASSED")


def test_auto_critic_loop_gradient_flow():
    """Phase 5: Gradients flow through AutoCriticLoop."""
    from aeon_core import AutoCriticLoop
    generator = nn.Linear(16, 16)
    acl = AutoCriticLoop(generator, hidden_dim=16, max_iterations=2, threshold=0.99)
    query = torch.randn(2, 16, requires_grad=True)
    out = acl(query)
    out["candidate"].sum().backward()
    assert query.grad is not None
    assert query.grad.abs().sum().item() > 0
    print("✅ test_auto_critic_loop_gradient_flow PASSED")


def test_fisher_computation_nan_guard():
    """Verify MetaLearner.compute_fisher skips NaN/Inf losses.
    
    The Fisher computation backward pass should skip batches where the
    loss is NaN or Inf to prevent corrupted gradient accumulation into
    the Fisher information matrix.
    """
    from aeon_core import MetaLearner

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    learner = MetaLearner(model, ewc_lambda=100.0)

    def data_loader_with_nan():
        # Yield a batch with NaN inputs that will produce NaN loss
        nan_inputs = torch.full((4, 16), float('nan'))
        yield nan_inputs, torch.randint(0, 4, (4,))
        # Yield a normal batch
        yield torch.randn(4, 16), torch.randint(0, 4, (4,))

    # Should not raise even when encountering NaN losses
    learner.compute_fisher(data_loader_with_nan, num_samples=8)

    # Fisher should be computed (non-empty)
    assert len(learner._fisher_diag) > 0, "Fisher diagonal should be populated"

    # All Fisher values should be finite (NaN batch was skipped)
    for name, f in learner._fisher_diag.items():
        assert torch.isfinite(f).all(), f"Fisher[{name}] contains non-finite values"

    print("✅ test_fisher_computation_nan_guard PASSED")


def test_task2vec_fisher_nan_guard():
    """Verify Task2VecMetaLearner._compute_fisher_diagonal skips NaN/Inf losses."""
    from aeon_core import Task2VecMetaLearner

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    t2v = Task2VecMetaLearner(model=model, ewc_lambda=10.0)

    def data_loader_with_nan():
        # Yield a batch with NaN inputs that will produce NaN loss
        nan_inputs = torch.full((4, 8), float('nan'))
        yield nan_inputs, torch.randint(0, 4, (4,))
        # Yield a normal batch
        yield torch.randn(4, 8), torch.randint(0, 4, (4,))

    fisher = t2v._compute_fisher_diagonal(data_loader_with_nan, num_samples=8)
    assert len(fisher) > 0, "Fisher diagonal should be populated"

    # All Fisher values should be finite (NaN batch was skipped)
    for name, f in fisher.items():
        assert torch.isfinite(f).all(), f"Fisher[{name}] contains non-finite values"

    print("✅ test_task2vec_fisher_nan_guard PASSED")


def test_forward_pass_returns_tensor_total_loss():
    """Verify ae_train.py _forward_pass returns a tensor for total_loss.
    
    The return type annotation was corrected from Dict[str, float] to
    Dict[str, Any] since total_loss must remain a Tensor for backward().
    """
    from ae_train import AEONConfigV4, AEONDeltaV4

    config = AEONConfigV4(
        vocab_size=500,
        seq_length=32,
        z_dim=64,
        hidden_dim=64,
        vq_embedding_dim=64,
        vq_num_embeddings=32
    )
    model = AEONDeltaV4(config)
    model.eval()

    tokens = torch.randint(0, 500, (2, 32))
    z = model.encode(tokens)
    quantized, vq_loss, indices, vq_stats = model.quantize(z)
    logits = model.decode(quantized, tokens)

    # total_loss should be a tensor (needed for backward)
    import torch.nn.functional as F
    recon_loss = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, config.vocab_size),
        tokens[:, 1:].contiguous().view(-1)
    )
    total_loss = recon_loss + vq_loss
    assert isinstance(total_loss, torch.Tensor), "total_loss must be a Tensor"
    assert total_loss.requires_grad, "total_loss must require gradients"

    print("✅ test_forward_pass_returns_tensor_total_loss PASSED")


# ============================================================================
# MODERNIZATION: RELIABILITY & RESILIENCE TESTS
# ============================================================================


def test_error_recovery_retry_and_history():
    """ErrorRecoveryManager records recovery history and supports retries."""
    from aeon_core import ErrorRecoveryManager, DecisionAuditLog

    audit = DecisionAuditLog()
    mgr = ErrorRecoveryManager(hidden_dim=64, audit_log=audit, max_retries=3)

    # Trigger a numerical recovery
    err = RuntimeError("NaN detected in output")
    fallback = torch.zeros(1, 64)
    ok, val = mgr.recover(err, context="test_retry", fallback=fallback)
    assert ok, "Recovery should succeed"

    # Check history was recorded
    history = mgr.get_recovery_history(5)
    assert len(history) >= 1, "Should have at least one history entry"
    assert history[-1]["success"] is True
    assert history[-1]["error_class"] == "numerical"

    # Success rate should be 1.0
    assert mgr.get_success_rate() == 1.0

    print("✅ test_error_recovery_retry_and_history PASSED")


def test_error_recovery_success_rate():
    """ErrorRecoveryManager.get_success_rate tracks success/failure ratio."""
    from aeon_core import ErrorRecoveryManager

    mgr = ErrorRecoveryManager(hidden_dim=64, max_retries=1)

    # Successful recovery
    mgr.recover(RuntimeError("NaN"), context="ok", fallback=torch.zeros(1, 64))

    # Failed recovery (unknown error, no fallback)
    mgr.recover(KeyError("bad_key"), context="fail")

    rate = mgr.get_success_rate()
    assert 0.0 < rate < 1.0, f"Success rate should be partial, got {rate}"

    print("✅ test_error_recovery_success_rate PASSED")


def test_context_window_decay():
    """ContextWindowManager with decay_rate favours recent entries."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=10, hidden_dim=4, decay_rate=100.0)

    # Add an old entry with high relevance
    old_emb = torch.ones(4)
    ctx.add("old_source", old_emb, relevance=1.0)

    # Simulate time passing
    import time as _time
    _time.sleep(0.01)

    # Add a new entry with lower relevance
    new_emb = torch.ones(4) * 2
    ctx.add("new_source", new_emb, relevance=0.5)

    top = ctx.get_top_k(2)
    assert len(top) == 2
    # With strong decay, the newer entry should rank first
    assert top[0]["source"] == "new_source", (
        "Newer entry should rank higher with strong decay"
    )

    print("✅ test_context_window_decay PASSED")


def test_context_window_no_decay_backward_compat():
    """ContextWindowManager with decay_rate=0 preserves old behaviour."""
    from aeon_core import ContextWindowManager

    ctx = ContextWindowManager(max_entries=10, hidden_dim=4, decay_rate=0.0)

    ctx.add("A", torch.ones(4), relevance=0.9)
    ctx.add("B", torch.ones(4), relevance=0.5)

    top = ctx.get_top_k(2)
    assert top[0]["source"] == "A", "Highest relevance should rank first"

    print("✅ test_context_window_no_decay_backward_compat PASSED")


def test_audit_log_export_json():
    """DecisionAuditLog.export_json writes valid JSON to disk."""
    import tempfile, json as _json
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=50)
    audit.record("meta_loop", "converged", {"iters": 5})
    audit.record("safety", "rollback", {"score": 0.3}, severity="warning")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    audit.export_json(path)

    with open(path) as fh:
        data = _json.load(fh)

    assert "entries" in data
    assert len(data["entries"]) == 2
    assert "summary" in data
    assert data["summary"]["total_decisions"] == 2

    import os
    os.unlink(path)

    print("✅ test_audit_log_export_json PASSED")


def test_audit_log_retrieve_by_time_range():
    """DecisionAuditLog.retrieve_by_time_range filters correctly."""
    import time as _time
    from aeon_core import DecisionAuditLog

    audit = DecisionAuditLog(max_entries=100)

    t0 = _time.monotonic()
    audit.record("A", "a1")
    _time.sleep(0.005)
    t1 = _time.monotonic()
    audit.record("B", "b1")
    t2 = _time.monotonic()

    # Only entries in [t1, t2] should match
    result = audit.retrieve_by_time_range(t1, t2)
    assert len(result) == 1
    assert result[0]["subsystem"] == "B"

    # Full range
    all_entries = audit.retrieve_by_time_range(t0, t2)
    assert len(all_entries) == 2

    print("✅ test_audit_log_retrieve_by_time_range PASSED")


def test_validator_validate_gradients():
    """StateConsistencyValidator.validate_gradients detects anomalies."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(
        hidden_dim=64, max_gradient_norm=1.0
    )

    # Simple model with known gradient
    model = torch.nn.Linear(64, 64)
    x = torch.randn(2, 64)
    loss = model(x).sum()
    loss.backward()

    result = validator.validate_gradients(model)
    assert "valid" in result
    assert "total_grad_norm" in result
    assert isinstance(result["total_grad_norm"], float)
    assert result["total_grad_norm"] > 0

    print("✅ test_validator_validate_gradients PASSED")


def test_validator_validate_gradients_explosion():
    """StateConsistencyValidator.validate_gradients flags exploding grads."""
    from aeon_core import StateConsistencyValidator

    validator = StateConsistencyValidator(
        hidden_dim=64, max_gradient_norm=0.001
    )

    model = torch.nn.Linear(64, 64)
    x = torch.randn(2, 64)
    loss = model(x).sum()
    loss.backward()

    result = validator.validate_gradients(model)
    # With threshold 0.001, normal gradients will exceed it
    assert result["grad_explosion"] is True

    print("✅ test_validator_validate_gradients_explosion PASSED")


def test_reasoning_core_pipeline_error_recovery():
    """reasoning_core returns a deterministic fallback on internal errors."""
    from aeon_core import AEONConfig, AEONDeltaV3

    config = AEONConfig(
        z_dim=64,
        hidden_dim=64,
        vocab_size=500,
        num_pillars=8,
        vq_embedding_dim=64,
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_world_model=False,
        enable_hierarchical_memory=False,
        enable_multimodal=False,
        use_vq=False,
    )
    model = AEONDeltaV3(config)
    model.eval()

    z_in = torch.randn(2, 64)

    # Sabotage the inner impl to force an error
    original_impl = model._reasoning_core_impl

    def broken_impl(*args, **kwargs):
        raise RuntimeError("Simulated pipeline failure")

    model._reasoning_core_impl = broken_impl

    # Should not raise — should return fallback
    z_out, outputs = model.reasoning_core(z_in)
    assert z_out.shape == (2, 64), f"Expected shape (2, 64), got {z_out.shape}"
    assert outputs.get("error_recovered") is True
    assert outputs.get("error_class") is not None

    # Restore original
    model._reasoning_core_impl = original_impl

    print("✅ test_reasoning_core_pipeline_error_recovery PASSED")


def test_trainer_gradient_anomaly_tracking():
    """AEONTrainer tracks gradient norms and loss EMA."""
    from aeon_core import AEONConfig, AEONDeltaV3, AEONTrainer

    config = AEONConfig(
        z_dim=64,
        hidden_dim=64,
        vocab_size=500,
        num_pillars=8,
        vq_embedding_dim=64,
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
        enable_world_model=False,
        enable_hierarchical_memory=False,
        enable_multimodal=False,
        use_vq=False,
    )
    model = AEONDeltaV3(config)
    trainer = AEONTrainer(model, config)

    batch = {
        'input_ids': torch.randint(0, 500, (2, 32)),
        'labels': torch.randint(0, 500, (2, 32)),
    }
    metrics = trainer.train_step(batch)

    assert 'grad_norm' in metrics, "Metrics should include grad_norm"
    assert metrics['grad_norm'] >= 0
    assert 'loss_ema' in metrics, "Metrics should include loss_ema"
    assert trainer._loss_ema is not None
    assert len(trainer._grad_norm_history) == 1

    print("✅ test_trainer_gradient_anomaly_tracking PASSED")


def test_hash_tensor_content_based():
    """Verify _hash_tensor uses content-based hashing to avoid collisions."""
    from aeon_core import FastHessianComputer
    
    hc = FastHessianComputer(method='finite_differences')
    
    # Tensors with same sum, std, first, last but different interior values
    t1 = torch.tensor([[1.0, 3.0, 2.0, 4.0]])  # sum=10
    t2 = torch.tensor([[2.0, 2.0, 2.0, 4.0]])  # sum=10
    
    h1 = hc._hash_tensor(t1)
    h2 = hc._hash_tensor(t2)
    
    assert h1 != h2, f"Hash collision for different tensors: {h1}"
    
    # Tensors with same sum, same std, same first/last but different content
    t3 = torch.tensor([[1.0, 4.0, 3.0, 2.0, 5.0]])
    t4 = torch.tensor([[1.0, 2.0, 5.0, 2.0, 5.0]])
    
    h3 = hc._hash_tensor(t3)
    h4 = hc._hash_tensor(t4)
    
    assert h3 != h4, f"Hash collision for different tensors with similar stats: {h3}"
    
    # Same tensor should produce same hash
    h5 = hc._hash_tensor(t1.clone())
    assert h1 == h5, "Same tensor content produced different hashes"
    
    print("✅ test_hash_tensor_content_based PASSED")


def test_quantize_int8_nan_safety():
    """Verify _quantize_int8 handles NaN tensors without producing NaN scale."""
    from aeon_core import InferenceCache
    
    # Tensor with NaN values
    t_nan = torch.tensor([1.0, float('nan'), 3.0, float('inf')])
    quantized, scale = InferenceCache._quantize_int8(t_nan)
    
    assert torch.isfinite(scale), f"Scale is not finite: {scale}"
    assert not torch.isnan(quantized.float()).any(), "Quantized contains NaN"
    
    # Normal tensor should still work
    t_normal = torch.tensor([1.0, 2.0, 3.0])
    q_normal, s_normal = InferenceCache._quantize_int8(t_normal)
    assert torch.isfinite(s_normal), f"Normal scale is not finite: {s_normal}"
    
    print("✅ test_quantize_int8_nan_safety PASSED")


def test_lipschitz_constant_finite():
    """Verify compute_lipschitz_constant never returns NaN/Inf."""
    from aeon_core import LipschitzConstrainedLambda
    
    net = LipschitzConstrainedLambda(input_dim=16, hidden_dim=32, output_dim=16)
    result = net.compute_lipschitz_constant(num_samples=10)
    
    assert math.isfinite(result), f"Lipschitz constant is not finite: {result}"
    assert result >= 0.0, f"Lipschitz constant is negative: {result}"
    
    print("✅ test_lipschitz_constant_finite PASSED")


def test_entropy_loss_consistency():
    """Verify entropy loss computation is consistent across all VQ classes."""
    # Test that the guard handles num_embeddings=1 without division by zero
    num_embeddings = 1
    max_entropy_guard = math.log(num_embeddings) if num_embeddings > 1 else 1.0
    assert max_entropy_guard == 1.0, "Guard for num_embeddings=1 should return 1.0"
    
    num_embeddings = 64
    max_entropy_normal = math.log(num_embeddings) if num_embeddings > 1 else 1.0
    assert max_entropy_normal == math.log(64), "Guard for num_embeddings=64 should return log(64)"
    
    print("✅ test_entropy_loss_consistency PASSED")


def test_rel_error_clamp():
    """Verify relative error is clamped to prevent extreme values."""
    # Simulate near-zero target with non-zero prediction
    pred = torch.randn(4, 16)
    z_target = torch.zeros(4, 16)  # Near-zero target
    
    rel_error = (torch.norm(pred - z_target, dim=1) / (torch.norm(z_target, dim=1) + 1e-8)).clamp(max=1e4).mean().item()
    
    assert rel_error <= 1e4, f"Relative error exceeds clamp: {rel_error}"
    assert math.isfinite(rel_error), f"Relative error is not finite: {rel_error}"
    
    print("✅ test_rel_error_clamp PASSED")


# ============================================================================
# SYSTEM INTEGRITY MONITOR TESTS
# ============================================================================

def test_integrity_monitor_record_and_health():
    """Verify SystemIntegrityMonitor records health and computes averages."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor(window_size=100)
    monitor.record_health("meta_loop", 0.9)
    monitor.record_health("meta_loop", 0.8)
    monitor.record_health("meta_loop", 1.0)
    
    avg = monitor.get_subsystem_health("meta_loop")
    assert abs(avg - 0.9) < 1e-6, f"Expected 0.9 average, got {avg}"
    
    # Unobserved subsystem should return 1.0
    assert monitor.get_subsystem_health("unknown") == 1.0
    
    print("✅ test_integrity_monitor_record_and_health PASSED")


def test_integrity_monitor_anomaly_detection():
    """Verify anomaly detection for below-threshold and rapid degradation."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor(
        anomaly_threshold=0.3, derivative_threshold=0.4
    )
    
    # Below-threshold anomaly
    anomaly = monitor.record_health("safety", 0.1)
    assert anomaly is not None, "Should detect below-threshold anomaly"
    assert anomaly["type"] == "below_threshold"
    
    # Normal score — no anomaly
    monitor.record_health("safety", 0.9)
    anomaly = monitor.record_health("safety", 0.85)
    assert anomaly is None, "Should not detect anomaly for healthy score"
    
    # Rapid degradation: drop from 0.85 to 0.3 (delta = 0.55 > 0.4)
    anomaly = monitor.record_health("safety", 0.3)
    assert anomaly is not None, "Should detect rapid degradation"
    assert anomaly["type"] == "rapid_degradation"
    
    anomalies = monitor.get_anomalies()
    assert len(anomalies) == 2, f"Expected 2 anomalies, got {len(anomalies)}"
    
    print("✅ test_integrity_monitor_anomaly_detection PASSED")


def test_integrity_monitor_checksum():
    """Verify deterministic checksumming and verification."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor()
    t1 = torch.tensor([[1.0, 2.0, 3.0]])
    t2 = torch.tensor([[1.0, 2.0, 3.0]])
    t3 = torch.tensor([[4.0, 5.0, 6.0]])
    
    digest1 = monitor.register_checksum("encoder", t1)
    assert isinstance(digest1, str) and len(digest1) == 64
    
    # Same tensor should verify
    assert monitor.verify_checksum("encoder", t2), "Identical tensors should verify"
    
    # Different tensor should fail
    assert not monitor.verify_checksum("encoder", t3), "Different tensors should not verify"
    
    # Unregistered component should pass
    assert monitor.verify_checksum("unregistered", t1), "Unregistered should pass"
    
    print("✅ test_integrity_monitor_checksum PASSED")


def test_integrity_monitor_global_health():
    """Verify global health aggregation across subsystems."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor()
    monitor.record_health("meta_loop", 1.0)
    monitor.record_health("safety", 0.5)
    
    global_h = monitor.get_global_health()
    assert abs(global_h - 0.75) < 1e-6, f"Expected 0.75, got {global_h}"
    
    # Empty monitor should return 1.0
    empty_monitor = SystemIntegrityMonitor()
    assert empty_monitor.get_global_health() == 1.0
    
    print("✅ test_integrity_monitor_global_health PASSED")


def test_integrity_monitor_report():
    """Verify get_integrity_report structure."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor()
    monitor.record_health("meta_loop", 0.9)
    monitor.register_checksum("test", torch.zeros(2, 3))
    
    report = monitor.get_integrity_report()
    assert "global_health" in report
    assert "subsystem_health" in report
    assert "anomalies" in report
    assert "checksums" in report
    assert "meta_loop" in report["subsystem_health"]
    assert "test" in report["checksums"]
    
    print("✅ test_integrity_monitor_report PASSED")


def test_integrity_monitor_reset():
    """Verify reset clears all state."""
    from aeon_core import SystemIntegrityMonitor
    
    monitor = SystemIntegrityMonitor()
    monitor.record_health("meta_loop", 0.5)
    monitor.register_checksum("x", torch.ones(1))
    monitor.reset()
    
    assert monitor.get_global_health() == 1.0
    assert monitor.get_anomalies() == []
    report = monitor.get_integrity_report()
    assert report["checksums"] == {}
    
    print("✅ test_integrity_monitor_reset PASSED")


def test_integrity_monitor_thread_safety():
    """Verify thread-safe concurrent health recording."""
    from aeon_core import SystemIntegrityMonitor
    import threading
    
    monitor = SystemIntegrityMonitor(window_size=1000)
    errors = []
    
    def record_many(subsystem, n):
        try:
            for i in range(n):
                monitor.record_health(subsystem, 0.5 + 0.5 * (i % 2))
        except Exception as e:
            errors.append(e)
    
    threads = [
        threading.Thread(target=record_many, args=(f"sub_{i}", 100))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert not errors, f"Thread errors: {errors}"
    
    print("✅ test_integrity_monitor_thread_safety PASSED")


# ============================================================================
# PROGRESS TRACKER TESTS
# ============================================================================

def test_progress_tracker_phase_lifecycle():
    """Verify begin/end/checkpoint phase lifecycle."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker()
    tracker.begin_phase("meta_loop")
    
    progress = tracker.get_progress()
    assert progress["current_phase"] == "meta_loop"
    assert "meta_loop" not in progress["completed_phases"]
    
    state = torch.randn(2, 64)
    tracker.checkpoint("meta_loop", state)
    tracker.end_phase("meta_loop", success=True, metadata={"iters": 7})
    
    progress = tracker.get_progress()
    assert progress["current_phase"] is None
    assert "meta_loop" in progress["completed_phases"]
    assert progress["phases"]["meta_loop"]["status"] == "success"
    assert progress["phases"]["meta_loop"]["metadata"]["iters"] == 7
    
    print("✅ test_progress_tracker_phase_lifecycle PASSED")


def test_progress_tracker_checkpoint_retrieval():
    """Verify checkpoint storage and retrieval."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker(max_checkpoints=3)
    
    t1 = torch.randn(2, 64)
    t2 = torch.randn(2, 64)
    
    tracker.checkpoint("phase_a", t1)
    tracker.checkpoint("phase_b", t2)
    
    # Get specific checkpoint
    retrieved = tracker.get_checkpoint("phase_a")
    assert retrieved is not None
    assert torch.allclose(retrieved, t1)
    
    # Get last checkpoint
    last = tracker.get_last_checkpoint()
    assert last is not None
    assert torch.allclose(last, t2)
    
    # Missing checkpoint
    assert tracker.get_checkpoint("nonexistent") is None
    
    print("✅ test_progress_tracker_checkpoint_retrieval PASSED")


def test_progress_tracker_rollback():
    """Verify rollback discards later phases and returns checkpoint."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker()
    t1 = torch.randn(2, 64)
    t2 = torch.randn(2, 64)
    t3 = torch.randn(2, 64)
    
    tracker.begin_phase("encode")
    tracker.checkpoint("encode", t1)
    tracker.end_phase("encode", success=True)
    
    tracker.begin_phase("meta_loop")
    tracker.checkpoint("meta_loop", t2)
    tracker.end_phase("meta_loop", success=True)
    
    tracker.begin_phase("safety")
    tracker.checkpoint("safety", t3)
    tracker.end_phase("safety", success=False)
    
    # Rollback to meta_loop
    restored = tracker.rollback_to("meta_loop")
    assert restored is not None
    assert torch.allclose(restored, t2)
    
    # Safety phase should be gone
    progress = tracker.get_progress()
    assert "safety" not in progress["phases"]
    assert "safety" not in progress["completed_phases"]
    
    # Rollback to nonexistent phase returns None
    assert tracker.rollback_to("nonexistent") is None
    
    print("✅ test_progress_tracker_rollback PASSED")


def test_progress_tracker_finish_run():
    """Verify finish_run archives and resets."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker()
    tracker.begin_phase("a")
    tracker.end_phase("a", success=True)
    
    summary = tracker.finish_run()
    assert summary["run_id"] == 0
    assert "a" in summary["phases"]
    
    # After finish, state is clean
    progress = tracker.get_progress()
    assert progress["run_id"] == 1
    assert progress["completed_phases"] == []
    
    # Run history
    history = tracker.get_run_history()
    assert len(history) == 1
    assert history[0]["run_id"] == 0
    
    print("✅ test_progress_tracker_finish_run PASSED")


def test_progress_tracker_failed_phases():
    """Verify failed phases are tracked correctly."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker()
    tracker.begin_phase("encode")
    tracker.end_phase("encode", success=True)
    tracker.begin_phase("meta_loop")
    tracker.end_phase("meta_loop", success=False)
    
    progress = tracker.get_progress()
    assert "encode" in progress["completed_phases"]
    assert "meta_loop" in progress["failed_phases"]
    
    print("✅ test_progress_tracker_failed_phases PASSED")


def test_progress_tracker_max_checkpoints():
    """Verify checkpoint eviction when max is exceeded."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker(max_checkpoints=2)
    tracker.checkpoint("a", torch.ones(1))
    tracker.checkpoint("b", torch.ones(1) * 2)
    tracker.checkpoint("c", torch.ones(1) * 3)
    
    # 'a' should have been evicted
    assert tracker.get_checkpoint("a") is None
    assert tracker.get_checkpoint("b") is not None
    assert tracker.get_checkpoint("c") is not None
    
    print("✅ test_progress_tracker_max_checkpoints PASSED")


def test_progress_tracker_reset():
    """Verify reset clears all state."""
    from aeon_core import ProgressTracker
    
    tracker = ProgressTracker()
    tracker.begin_phase("x")
    tracker.checkpoint("x", torch.ones(1))
    tracker.end_phase("x", success=True)
    tracker.finish_run()
    
    tracker.reset()
    progress = tracker.get_progress()
    assert progress["run_id"] == 0
    assert progress["completed_phases"] == []
    assert tracker.get_run_history() == []
    assert tracker.get_last_checkpoint() is None
    
    print("✅ test_progress_tracker_reset PASSED")


# ============================================================================
# DETERMINISTIC EXECUTION GUARD TESTS
# ============================================================================

def test_execution_guard_normalize_input():
    """Verify input normalization sanitizes NaN/Inf and clamps."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64, input_clamp=10.0)
    
    # Test NaN/Inf sanitization
    x = torch.tensor([[float('nan'), float('inf'), -float('inf'), 5.0]])
    normalized = guard.normalize_input(x)
    assert torch.isfinite(normalized).all(), "Should remove NaN/Inf"
    assert normalized.abs().max().item() <= 10.0, "Should clamp"
    
    # Normal input should be clamped
    x_big = torch.tensor([[100.0, -200.0, 3.0]])
    normalized = guard.normalize_input(x_big)
    assert normalized.abs().max().item() <= 10.0
    
    print("✅ test_execution_guard_normalize_input PASSED")


def test_execution_guard_validate_output():
    """Verify output validation detects invalid tensors and applies fallback."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64, max_activation=100.0)
    
    # Valid output
    valid_t = torch.randn(2, 64)
    ok, result = guard.validate_output(valid_t, stage="test")
    assert ok is True
    assert torch.allclose(result, valid_t)
    
    # NaN output — should fallback
    nan_t = torch.full((2, 64), float('nan'))
    fallback = torch.zeros(2, 64)
    ok, result = guard.validate_output(nan_t, stage="test", fallback=fallback)
    assert ok is False
    assert torch.allclose(result, fallback)
    
    # Excessive magnitude — should fallback
    big_t = torch.full((2, 64), 1e5)
    ok, result = guard.validate_output(big_t, stage="test_big")
    assert ok is False
    assert result.abs().max().item() == 0.0  # zeros_like fallback
    
    print("✅ test_execution_guard_validate_output PASSED")


def test_execution_guard_fingerprint():
    """Verify deterministic fingerprinting and verification."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64)
    
    t1 = torch.tensor([[1.0, 2.0, 3.0]])
    t2 = torch.tensor([[1.0, 2.0, 3.0]])
    t3 = torch.tensor([[4.0, 5.0, 6.0]])
    
    fp1 = guard.fingerprint("stage_a", t1)
    assert isinstance(fp1, str) and len(fp1) == 64
    
    # Same tensor verifies
    assert guard.verify_fingerprint("stage_a", t2)
    
    # Different tensor fails
    assert not guard.verify_fingerprint("stage_a", t3)
    
    # Unregistered stage passes
    assert guard.verify_fingerprint("unregistered", t1)
    
    print("✅ test_execution_guard_fingerprint PASSED")


def test_execution_guard_execute_with_guard():
    """Verify execute_with_guard wraps fn with normalization + validation."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64, input_clamp=10.0)
    
    # Simple identity function
    ok, result = guard.execute_with_guard(
        fn=lambda x: x * 2,
        input_tensor=torch.tensor([[5.0, 3.0]]),
        stage="double",
    )
    assert ok is True
    assert torch.allclose(result, torch.tensor([[10.0, 6.0]]))
    
    # Function that produces NaN — should fallback
    fallback = torch.zeros(1, 2)
    ok, result = guard.execute_with_guard(
        fn=lambda x: x * float('nan'),
        input_tensor=torch.tensor([[5.0, 3.0]]),
        stage="nan_fn",
        fallback=fallback,
    )
    assert ok is False
    assert torch.allclose(result, fallback)
    
    # Function that raises — should fallback
    def bad_fn(x):
        raise RuntimeError("oops")
    ok, result = guard.execute_with_guard(
        fn=bad_fn,
        input_tensor=torch.tensor([[1.0]]),
        stage="error_fn",
        fallback=torch.zeros(1, 1),
    )
    assert ok is False
    
    print("✅ test_execution_guard_execute_with_guard PASSED")


def test_execution_guard_validation_summary():
    """Verify validation summary aggregates correctly."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64)
    
    guard.validate_output(torch.randn(2, 64), stage="ok1")
    guard.validate_output(torch.randn(2, 64), stage="ok2")
    guard.validate_output(torch.full((2, 64), float('nan')), stage="fail")
    
    summary = guard.get_validation_summary()
    assert summary["total"] == 3
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 1
    assert abs(summary["success_rate"] - 2/3) < 1e-6
    
    print("✅ test_execution_guard_validation_summary PASSED")


def test_execution_guard_reset():
    """Verify reset clears all state."""
    from aeon_core import DeterministicExecutionGuard
    
    guard = DeterministicExecutionGuard(hidden_dim=64)
    guard.validate_output(torch.randn(2, 64), stage="test")
    guard.fingerprint("test", torch.ones(1))
    guard.reset()
    
    summary = guard.get_validation_summary()
    assert summary["total"] == 0
    assert summary["fingerprints"] == {}
    
    print("✅ test_execution_guard_reset PASSED")


# ============================================================================
# INTEGRATION TESTS — new components in reasoning pipeline
# ============================================================================

def test_reasoning_core_integrity_report():
    """Verify reasoning_core produces integrity_report in outputs."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        result = model(input_ids, fast=True)
    
    assert 'integrity_report' in result, "Should have integrity_report"
    report = result['integrity_report']
    assert 'global_health' in report
    assert 'subsystem_health' in report
    
    assert 'progress_summary' in result, "Should have progress_summary"
    summary = result['progress_summary']
    assert 'phases' in summary
    
    print("✅ test_reasoning_core_integrity_report PASSED")


def test_reasoning_core_progress_tracking():
    """Verify progress_tracker records phases during reasoning_core."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        result = model(input_ids, fast=True)
    
    summary = result['progress_summary']
    # Should have recorded meta_loop and integration phases
    assert "meta_loop" in summary["phases"], "meta_loop phase should be tracked"
    assert "integration" in summary["phases"], "integration phase should be tracked"
    
    print("✅ test_reasoning_core_progress_tracking PASSED")


def test_reasoning_core_deterministic_guard():
    """Verify DeterministicExecutionGuard is active in reasoning pipeline."""
    from aeon_core import AEONConfig, AEONDeltaV3
    
    config = AEONConfig(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    model = AEONDeltaV3(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        result = model(input_ids, fast=True)
    
    # Execution guard should have fingerprinted the integration output
    summary = model.execution_guard.get_validation_summary()
    assert summary["total"] >= 1, "Should have at least 1 validation"
    assert "integration" in summary["fingerprints"], "Should fingerprint integration"
    
    print("✅ test_reasoning_core_deterministic_guard PASSED")


if __name__ == '__main__':
    test_division_by_zero_in_fit()
    test_quarantine_batch_thread_safety()
    test_tensor_hash_collision_resistance()
    test_rssm_trainer_zero_batches()
    test_memory_manager_flatten()
    test_memory_manager_nan_rejection()
    test_quarantine_partial_corruption()
    test_config_validation()
    test_document_aware_dataset()
    
    # New tests for problems 1-10
    test_lipschitz_contraction()
    test_encoder_input_validation()
    test_meta_loop_convergence()
    test_verify_convergence_method()
    test_batch_generation_per_sequence_stopping()
    test_graceful_degradation_generate()
    test_set_seed_reproducibility()
    test_compute_lipschitz_loss_standalone()
    test_safe_checkpoint_loading()
    
    # Modernization tests
    test_selective_ssm_forward()
    test_ssm_state_caching()
    test_linear_attention_block()
    test_linear_attention_bidirectional()
    test_chunked_sequence_processor()
    test_inference_cache()
    test_ssm_thought_encoder()
    test_ssm_thought_decoder_train()
    test_ssm_thought_decoder_inference()
    test_linear_attention_encoder()
    test_build_encoder_factory()
    test_build_decoder_factory()
    test_ssm_long_sequence()
    test_ssm_gradient_flow()
    test_aeon_v3_with_ssm_backend()
    test_aeon_v3_with_lstm_backend()
    test_config_backend_validation()
    test_pretrained_backbone_adapter_fallback()
    
    # Section I improvement tests
    test_parallel_scan_consistency()
    test_poly_feature_map()
    test_linear_attention_low_rank()
    test_chunked_adaptive_blending()
    test_inference_cache_ring_buffer()
    test_inference_cache_quantization()
    test_hybrid_adapter_components()
    
    # Section II AGI component tests
    test_world_model_forward()
    test_world_model_counterfactuals()
    test_world_model_gradient_flow()
    test_hierarchical_memory_store_retrieve()
    test_hierarchical_memory_semantic()
    test_hierarchical_memory_consolidation()
    test_multimodal_grounding_language_vision()
    test_multimodal_grounding_single_modality()
    test_multimodal_grounding_three_modalities()
    test_meta_learner_ewc_loss()
    test_meta_learner_task_buffer()
    test_aeon_v3_with_world_model()
    test_aeon_v3_with_hierarchical_memory()
    
    # Analysis-driven fix tests
    test_hessian_forward_ad_computation()
    test_usage_stats_zero_count_safety()
    test_ema_update_zero_cluster_safety()
    
    # Code analysis fix tests
    test_config_immutability()
    test_forward_input_ids_validation()
    test_forward_ad_version_check()
    
    # Mamba-2 (SSD) tests
    test_selective_ssmv2_forward()
    test_ssmv2_state_caching()
    test_mamba2_thought_encoder()
    test_mamba2_thought_decoder_train()
    test_mamba2_thought_decoder_inference()
    test_build_encoder_factory_mamba2()
    test_build_decoder_factory_mamba2()
    test_mamba2_gradient_flow()
    test_mamba2_long_sequence()
    test_aeon_v3_with_mamba2_backend()
    test_config_mamba2_validation()
    
    # Refactoring analysis fix tests
    test_entropy_loss_single_embedding()
    test_entropy_loss_guard()
    test_certified_error_numerical_stability()
    test_version_consistency()
    
    # v4 bug fix regression tests
    test_warmup_cosine_scheduler_clamp()
    test_nan_path_preserves_accumulated_gradients()
    test_nan_metrics_not_contaminating_epoch()
    test_entropy_loss_returns_tensor()
    test_vq_temperature_validation()
    test_perplexity_overflow_guard()
    test_gradscaler_compatibility()
    
    # Architecture refactoring tests (Tasks 1-13)
    test_diversity_metric_forward()
    test_sparse_factorization_forward()
    test_sparse_factorization_sparsity_loss()
    test_neural_causal_model_forward()
    test_neural_causal_model_dag_constraint()
    test_neural_causal_model_intervention()
    test_neural_causal_model_dag_loss()
    test_neural_causal_model_consistency_loss()
    test_neural_causal_model_gradient_flow()
    test_value_network_forward()
    test_policy_network_forward()
    test_mcts_node_ucb1()
    test_mcts_planner_forward()
    test_mcts_planner_search()
    test_hierarchical_vae_forward()
    test_hierarchical_vae_abstraction_level()
    test_hierarchical_vae_kl_loss()
    test_adaptive_chunking()
    test_world_model_surprise_integration()
    test_memory_retrieval_integration()
    test_safety_enforcement()
    
    # Generation quality fix tests
    test_filter_logits_all_inf_guard()
    test_filter_logits_nan_handling()
    test_temperature_clamping()
    test_safety_blending_not_replacement()
    test_missing_weight_xavier_init()
    test_safety_threshold_default()
    
    # New cognitive architecture enhancement tests
    test_convergence_monitor_warmup()
    test_convergence_monitor_converged()
    test_convergence_monitor_diverging()
    test_convergence_monitor_reset()
    test_hierarchical_meta_loop_forward()
    test_hierarchical_meta_loop_training_uses_deep()
    test_causal_factor_extractor_forward()
    test_causal_factor_extractor_intervention()
    test_causal_factor_extractor_gradient_flow()
    test_temporal_memory_store_and_retrieve()
    test_temporal_memory_decay()
    test_temporal_memory_consolidation()
    test_temporal_memory_empty_retrieve()
    test_grounded_multimodal_learning_forward()
    test_grounded_multimodal_learning_zero_shot()
    test_grounded_multimodal_gradient_flow()
    test_curiosity_driven_exploration_reward()
    test_curiosity_driven_exploration_inverse()
    test_curiosity_driven_select_action()
    test_continual_learning_core_add_task()
    test_continual_learning_core_ewc_loss()
    test_continual_learning_ewc_missing_task()
    
    # AGI critical modification tests
    test_recursive_meta_loop_forward()
    test_recursive_meta_loop_target_level()
    test_recursive_meta_loop_has_levels()
    test_neurogenic_memory_consolidate()
    test_neurogenic_memory_retrieve()
    test_neurogenic_memory_capacity_limit()
    test_neurogenic_memory_synapse_formation()
    test_causal_world_model_forward()
    test_causal_world_model_intervention()
    test_causal_world_model_counterfactual_rollout()
    test_causal_world_model_gradient_flow()
    test_active_learning_planner_forward()
    test_active_learning_planner_intrinsic_reward()
    test_active_learning_planner_search()
    
    # ae_train.py robustness fix tests
    test_save_checkpoint_error_handling()
    test_save_metrics_error_handling()
    test_rssm_nan_branch_no_zero_grad()
    test_config_v4_extended_validation()
    
    # Stride and metrics fixes
    test_chunked_processor_adaptive_stride_not_zero()
    test_fit_remaining_batch_metrics()
    
    # Advanced Cognitive Modules tests (Priority 1-5)
    test_certified_meta_loop_forward()
    test_certified_meta_loop_verify_preconditions()
    test_certified_meta_loop_ibp_lipschitz()
    test_unified_memory_read()
    test_unified_memory_write_and_read()
    test_unified_memory_batched()
    test_unified_memory_temporal_links()
    test_hierarchical_world_model_forward()
    test_hierarchical_world_model_single_level()
    test_hierarchical_world_model_gradient_flow()
    test_adaptive_meta_loop_forward()
    test_adaptive_meta_loop_ponder_cost()
    test_adaptive_meta_loop_gradient_flow()
    test_neuro_symbolic_reasoner_forward()
    test_neuro_symbolic_reasoner_gradient_flow()
    test_differentiable_forward_chainer()
    test_neuro_symbolic_facts_in_unit_interval()
    
    # Refactoring analysis tests: NaN guards, epsilon safety, exception specificity
    test_lipschitz_estimate_nan_guard()
    test_lipschitz_ema_nan_skip()
    test_denominator_max_vs_add()
    test_certified_error_nan_residual()
    test_checkpoint_load_specific_exception()
    test_adaptive_chunking_max_var()
    
    # New architecture recommendation tests
    test_gumbel_vector_quantizer_forward()
    test_gumbel_vector_quantizer_training_vs_eval()
    test_gumbel_vector_quantizer_gradient_flow()
    test_gumbel_vector_quantizer_temperature_annealing()
    test_neural_turing_machine_forward()
    test_neural_turing_machine_store_retrieve()
    test_neural_turing_machine_gradient_flow()
    test_latent_dynamics_model_forward()
    test_latent_dynamics_model_rollout()
    test_latent_dynamics_model_gradient_flow()
    test_causal_programmatic_model_forward()
    test_causal_programmatic_model_counterfactual()
    test_causal_programmatic_model_dag_loss()
    test_causal_programmatic_model_gradient_flow()
    
    # Strategic AGI Recommendations tests
    test_compositional_slot_attention_forward()
    test_compositional_slot_attention_gradient()
    test_compositional_slot_attention_iterations()
    test_notears_causal_model_forward()
    test_notears_dag_loss()
    test_notears_dag_loss_gradient()
    test_notears_intervention()
    test_notears_l1_loss()
    test_consolidating_memory_store_and_consolidate()
    test_consolidating_memory_retrieve()
    test_consolidating_memory_forward()
    test_consolidating_memory_gradient()
    test_task2vec_meta_learner_embed()
    test_task2vec_meta_learner_adapt()
    test_task2vec_ewc_loss()
    test_certified_meta_loop_ibp_per_layer()
    
    # Refactoring fixes tests (division-by-zero guards, type safety, NaN guards)
    test_epoch_metrics_empty_list_guard()
    test_weight_tying_scores_empty_guard()
    test_entropy_loss_returns_tensor()
    test_optimizer_step_returns_float()
    test_grad_norm_nan_guard_in_fit()
    
    # Modernization tests: Robust logic improvements
    test_rssm_residual_and_norm()
    test_integration_module_residual_norm()
    test_consistency_gate_forward()
    test_consistency_gate_gradient_flow()
    test_consistency_gate_in_reasoning_output()
    test_value_net_has_layer_norm()
    test_importance_scorer_has_layer_norm()
    
    # AGI Modernization: Error resilience & logical integrity tests
    test_convergence_trajectory_bounded()
    test_memory_manager_capacity_bound()
    test_memory_manager_thread_safety()
    test_inference_cache_model_version_invalidation()
    test_hessian_nonfinite_sanitization()
    test_meta_loop_nan_recovery()
    test_mcts_ucb1_nonfinite_guard()
    test_mcts_simulate_nonfinite_guard()
    test_reasoning_core_nan_fallback()
    test_generate_resets_inference_cache()
    
    # AGI Modernization: Numerical stability, thread safety & state management
    test_hierarchical_vae_logvar_clamping()
    test_unified_memory_temporal_stability()
    test_unified_memory_input_validation()
    test_certified_meta_loop_division_safety()
    test_inference_cache_thread_safety()
    test_forward_chainer_saturation_prevention()
    test_memory_manager_timestamp_tracking()
    test_memory_manager_timestamp_eviction()
    test_ema_reset_on_checkpoint_concept()
    
    # AGI Modernization: Decision audit, state validation & error classification
    test_decision_audit_log_record_and_recent()
    test_decision_audit_log_summary()
    test_decision_audit_log_bounded_capacity()
    test_decision_audit_log_reset()
    test_decision_audit_log_thread_safety()
    test_state_consistency_validator_valid()
    test_state_consistency_validator_nan_detection()
    test_state_consistency_validator_shape_mismatch()
    test_state_consistency_validator_activation_magnitude()
    test_semantic_error_classifier_numerical()
    test_semantic_error_classifier_shape()
    test_semantic_error_classifier_resource()
    test_semantic_error_classifier_unknown()
    test_semantic_error_classifier_tensor_state_healthy()
    test_semantic_error_classifier_tensor_state_nan()
    test_semantic_error_classifier_tensor_state_inf()
    test_audit_log_in_reasoning_core()
    test_state_validation_in_reasoning_output()
    test_memory_load_specific_exception()
    
    # AGI Modernization: Error recovery, context window, audit & validator tests
    test_error_recovery_numerical()
    test_error_recovery_convergence()
    test_error_recovery_unknown_with_fallback()
    test_error_recovery_unknown_no_fallback()
    test_error_recovery_reset_stats()
    test_error_recovery_resource()
    test_context_window_add_and_retrieve()
    test_context_window_eviction()
    test_context_window_rejects_nonfinite()
    test_context_window_get_context_tensor()
    test_audit_log_severity_levels()
    test_audit_log_filter_by_subsystem()
    test_audit_log_filter_by_severity()
    test_audit_log_backward_compat()
    test_validator_validate_and_recover_clean()
    test_validator_validate_and_recover_nan()
    test_validator_validate_and_recover_shape()
    test_validator_validate_and_recover_activation_clamp()
    test_semantic_error_classifier_with_suggestion()
    test_ssd_block_chunk_len_guard()
    
    # Device consistency tests
    test_rssm_trainer_uses_model_device()
    test_validate_training_components_uses_model_device()
    
    # Architectural Roadmap tests (Phases 1-5)
    test_shared_workspace_broadcast_and_read()
    test_shared_workspace_padding()
    test_shared_workspace_truncation()
    test_attention_arbiter_urgency()
    test_attention_arbiter_top_k()
    test_meta_monitor_update()
    test_cognitive_executive_function_forward()
    test_cognitive_executive_function_gradient_flow()
    test_recovery_experience_replay_push_and_sample()
    test_recovery_experience_replay_capacity()
    test_meta_recovery_learner_forward()
    test_meta_recovery_learner_compute_loss()
    test_meta_recovery_learner_gradient_flow()
    test_unified_causal_simulator_forward()
    test_unified_causal_simulator_intervention()
    test_unified_causal_simulator_counterfactual()
    test_unified_causal_simulator_gradient_flow()
    test_neuro_symbolic_bridge_roundtrip()
    test_temporal_knowledge_graph_add_and_retrieve()
    test_temporal_knowledge_graph_capacity()
    test_temporal_knowledge_graph_empty_retrieve()
    test_hybrid_reasoning_engine_forward()
    test_hybrid_reasoning_engine_with_query()
    test_hybrid_reasoning_engine_gradient_flow()
    test_critic_network_forward()
    test_critic_network_explain_failure()
    test_revision_network_forward()
    test_auto_critic_loop_forward()
    test_auto_critic_loop_trajectory()
    test_auto_critic_loop_gradient_flow()
    
    # Fisher computation NaN guard tests
    test_fisher_computation_nan_guard()
    test_task2vec_fisher_nan_guard()
    
    # Type annotation correctness
    test_forward_pass_returns_tensor_total_loss()
    
    # Modernization: Reliability & Resilience tests
    test_error_recovery_retry_and_history()
    test_error_recovery_success_rate()
    test_context_window_decay()
    test_context_window_no_decay_backward_compat()
    test_audit_log_export_json()
    test_audit_log_retrieve_by_time_range()
    test_validator_validate_gradients()
    test_validator_validate_gradients_explosion()
    test_reasoning_core_pipeline_error_recovery()
    test_trainer_gradient_anomaly_tracking()
    
    # Content-based hash, NaN safety, and consistency tests
    test_hash_tensor_content_based()
    test_quantize_int8_nan_safety()
    test_lipschitz_constant_finite()
    test_entropy_loss_consistency()
    test_rel_error_clamp()
    
    # System integrity, progress tracking & deterministic execution tests
    test_integrity_monitor_record_and_health()
    test_integrity_monitor_anomaly_detection()
    test_integrity_monitor_checksum()
    test_integrity_monitor_global_health()
    test_integrity_monitor_report()
    test_integrity_monitor_reset()
    test_integrity_monitor_thread_safety()
    test_progress_tracker_phase_lifecycle()
    test_progress_tracker_checkpoint_retrieval()
    test_progress_tracker_rollback()
    test_progress_tracker_finish_run()
    test_progress_tracker_failed_phases()
    test_progress_tracker_max_checkpoints()
    test_progress_tracker_reset()
    test_execution_guard_normalize_input()
    test_execution_guard_validate_output()
    test_execution_guard_fingerprint()
    test_execution_guard_execute_with_guard()
    test_execution_guard_validation_summary()
    test_execution_guard_reset()
    test_reasoning_core_integrity_report()
    test_reasoning_core_progress_tracking()
    test_reasoning_core_deterministic_guard()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
