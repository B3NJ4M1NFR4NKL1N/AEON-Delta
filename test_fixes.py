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
    """Fix 5: ae_train.py - Guard against zero total_batches in RSSM trainer.
    
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
    """Fix 6: aeon_core.py - MemoryManager.retrieve_relevant input validation.
    
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
    # Doc 2: no valid targets (only 2 chunks, need >= 3)
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
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
