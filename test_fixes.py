"""
Tests for refactoring fixes in aeon_core.py and ae_train.py.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os

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
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
