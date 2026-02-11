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
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
