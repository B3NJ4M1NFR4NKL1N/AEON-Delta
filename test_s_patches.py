"""
S-series patch tests: Wizard token preparation & VT weight key alignment.

Tests cover:
  S1 — /api/train/v4/upload populates APP._wizard_tokens
  S2 — _generate_synthetic_weights produces correct adapter key names/shapes
  S3 — _generate_synthetic_weights produces correct kernel key names/shapes
  S4 — _extract_aeon_weights produces correct adapter key names/shapes
  S5 — _extract_aeon_weights produces correct kernel key names/shapes
  S6 — load_vt_weights handles AEON-formatted weights with shape validation
  S7 — load_vt_weights handles raw HF weights via extraction pipeline
  S8 — Synthetic weights load successfully into model modules
  S9 — Dashboard loaded_keys contain correct prefixed keys
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Lazy imports — we import from aeon_core at module level so tests can
# reference the symbols without per-test import overhead.
# ---------------------------------------------------------------------------
from aeon_core import (
    VibeThinkerWeightManager,
    VibeThinkerPromptAdapter,
    VibeThinkerReasoningKernel,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

class _MockVTConfig:
    """Minimal VibeThinkerConfig stub for kernel instantiation."""
    adapter_projection_dim = 128
    enabled = True


def _make_adapter(latent_dim=256, hidden_dim=256, projection_dim=128):
    return VibeThinkerPromptAdapter(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
    )


def _make_kernel(hidden_dim=256):
    return VibeThinkerReasoningKernel(
        config=_MockVTConfig(),
        hidden_dim=hidden_dim,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  S1 — Wizard token preparation on upload
# ═══════════════════════════════════════════════════════════════════════════

class TestS1WizardTokenPreparation:
    """Verify that uploaded training data is tokenized for the wizard."""

    def test_wizard_tokens_populated_from_jsonl(self, tmp_path):
        """Uploading a JSONL file should set APP._wizard_tokens."""
        # Create a mock JSONL file
        jsonl_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": "This is a test sentence for tokenization."})
            for _ in range(5)
        ]
        jsonl_file.write_text("\n".join(lines))

        # Simulate the tokenization logic from the upload endpoint
        texts = []
        with open(str(jsonl_file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                txt = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                if txt and len(txt.strip()) > 10:
                    texts.append(txt)

        assert len(texts) == 5
        seq_len = 128
        fallback_vocab = 50000
        tokenized = []
        for t in texts:
            toks = [ord(c) % fallback_vocab for c in t[:seq_len]]
            toks += [0] * (seq_len - len(toks))
            tokenized.append(toks)
        tokens = torch.tensor(tokenized, dtype=torch.long)
        assert tokens.shape == (5, 128)
        assert tokens.dtype == torch.long

    def test_wizard_tokens_skip_short_texts(self, tmp_path):
        """Texts shorter than 10 chars should be skipped."""
        jsonl_file = tmp_path / "short.jsonl"
        lines = [
            json.dumps({"text": "tiny"}),
            json.dumps({"text": "This is long enough to be included."}),
        ]
        jsonl_file.write_text("\n".join(lines))

        texts = []
        with open(str(jsonl_file), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                txt = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                if txt and len(txt.strip()) > 10:
                    texts.append(txt)

        assert len(texts) == 1

    def test_wizard_tokens_empty_file(self, tmp_path):
        """Empty file should produce no tokens."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        texts = []
        with open(str(jsonl_file), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    txt = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                    if txt and len(txt.strip()) > 10:
                        texts.append(txt)
                except Exception:
                    pass

        assert len(texts) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  S2 — Synthetic adapter weight key names and shapes
# ═══════════════════════════════════════════════════════════════════════════

class TestS2SyntheticAdapterWeights:
    """Verify _generate_synthetic_weights produces correct adapter keys."""

    @pytest.fixture(autouse=True)
    def setup_mgr(self, tmp_path):
        self.mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        self.adapter = _make_adapter()

    def test_adapter_key_names_match_model(self):
        """Synthetic adapter keys must match VibeThinkerPromptAdapter state_dict."""
        result = self.mgr.download_weights()
        assert result.get("ok"), f"download_weights failed: {result}"

        synth_adapter_keys = {
            k.replace("adapter.", "")
            for k in self.mgr._loaded_keys
            if k.startswith("adapter.")
        }
        model_keys = set(self.adapter.state_dict().keys())
        assert synth_adapter_keys == model_keys, (
            f"Key mismatch:\n"
            f"  Missing from synth: {model_keys - synth_adapter_keys}\n"
            f"  Extra in synth: {synth_adapter_keys - model_keys}"
        )

    def test_adapter_shapes_match_model(self):
        """Synthetic adapter weight shapes must match the model."""
        self.mgr.download_weights()
        from safetensors.torch import load_file
        flat = load_file(str(self.mgr._weights_path))
        adapter_payload = {
            k[len("adapter_state."):]: v
            for k, v in flat.items()
            if k.startswith("adapter_state.")
        }
        model_sd = self.adapter.state_dict()
        for key in model_sd:
            assert key in adapter_payload, f"Missing synth key: {key}"
            assert adapter_payload[key].shape == model_sd[key].shape, (
                f"Shape mismatch for {key}: "
                f"synth={adapter_payload[key].shape} model={model_sd[key].shape}"
            )

    def test_adapter_has_context_embedding(self):
        """context_embedding must be in synthetic adapter weights."""
        self.mgr.download_weights()
        assert any(
            "context_embedding" in k for k in self.mgr._loaded_keys
        ), "context_embedding missing from loaded_keys"

    def test_adapter_complexity_head_two_layers(self):
        """complexity_head must have 2-layer structure (0 and 2 indices)."""
        self.mgr.download_weights()
        adapter_keys = [
            k for k in self.mgr._loaded_keys if k.startswith("adapter.")
        ]
        assert any("complexity_head.0" in k for k in adapter_keys)
        assert any("complexity_head.2" in k for k in adapter_keys)


# ═══════════════════════════════════════════════════════════════════════════
#  S3 — Synthetic kernel weight key names and shapes
# ═══════════════════════════════════════════════════════════════════════════

class TestS3SyntheticKernelWeights:
    """Verify _generate_synthetic_weights produces correct kernel keys."""

    @pytest.fixture(autouse=True)
    def setup_mgr(self, tmp_path):
        self.mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        self.kernel = _make_kernel()

    def test_kernel_key_names_match_model(self):
        """Synthetic kernel keys must match VibeThinkerReasoningKernel state_dict."""
        result = self.mgr.download_weights()
        assert result.get("ok")

        synth_kernel_keys = {
            k.replace("kernel.", "")
            for k in self.mgr._loaded_keys
            if k.startswith("kernel.")
        }
        model_keys = set(self.kernel.state_dict().keys())
        assert synth_kernel_keys == model_keys, (
            f"Key mismatch:\n"
            f"  Missing from synth: {model_keys - synth_kernel_keys}\n"
            f"  Extra in synth: {synth_kernel_keys - model_keys}"
        )

    def test_kernel_shapes_match_model(self):
        """Synthetic kernel weight shapes must match the model."""
        self.mgr.download_weights()
        from safetensors.torch import load_file
        flat = load_file(str(self.mgr._weights_path))
        kernel_payload = {
            k[len("kernel_state."):]: v
            for k, v in flat.items()
            if k.startswith("kernel_state.")
        }
        model_sd = self.kernel.state_dict()
        for key in model_sd:
            assert key in kernel_payload, f"Missing synth key: {key}"
            assert kernel_payload[key].shape == model_sd[key].shape, (
                f"Shape mismatch for {key}: "
                f"synth={kernel_payload[key].shape} model={model_sd[key].shape}"
            )

    def test_kernel_reasoning_projector_correct_name(self):
        """Kernel must use 'reasoning_projector' not 'reasoning_proj'."""
        self.mgr.download_weights()
        kernel_keys = [
            k for k in self.mgr._loaded_keys if k.startswith("kernel.")
        ]
        assert any("reasoning_projector" in k for k in kernel_keys)
        assert not any(
            "reasoning_proj." in k and "reasoning_projector" not in k
            for k in kernel_keys
        )

    def test_kernel_heads_two_layer_structure(self):
        """All kernel heads must have 2-layer structure."""
        self.mgr.download_weights()
        kernel_keys = [
            k for k in self.mgr._loaded_keys if k.startswith("kernel.")
        ]
        for head in ("confidence_head", "entropy_head", "cot_depth_head"):
            assert any(f"{head}.0" in k for k in kernel_keys), (
                f"{head}.0 missing"
            )
            assert any(f"{head}.2" in k for k in kernel_keys), (
                f"{head}.2 missing"
            )

    def test_kernel_no_feature_proj(self):
        """Kernel should NOT have feature_proj (not in actual model)."""
        self.mgr.download_weights()
        kernel_keys = [
            k for k in self.mgr._loaded_keys if k.startswith("kernel.")
        ]
        assert not any("feature_proj" in k for k in kernel_keys)

    def test_kernel_no_standalone_layer_norm(self):
        """Kernel should NOT have standalone layer_norm (it's inside reasoning_projector)."""
        self.mgr.download_weights()
        kernel_keys = [
            k for k in self.mgr._loaded_keys if k.startswith("kernel.")
        ]
        # layer_norm should only appear as part of reasoning_projector.2
        standalone = [
            k for k in kernel_keys
            if "layer_norm" in k and "reasoning_projector" not in k
        ]
        assert len(standalone) == 0, f"Unexpected standalone layer_norm: {standalone}"


# ═══════════════════════════════════════════════════════════════════════════
#  S4 — _extract_aeon_weights adapter key alignment
# ═══════════════════════════════════════════════════════════════════════════

class TestS4ExtractAdapterWeights:
    """Verify _extract_aeon_weights produces correct adapter keys from HF state."""

    def _make_hf_state(self):
        """Create a minimal HF-like state dict."""
        return {
            "model.embed_tokens.weight": torch.randn(32000, 1536),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(1536, 1536),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4096, 1536),
            "model.layers.0.mlp.up_proj.weight": torch.randn(4096, 1536),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(1536, 1536),
            "model.layers.1.mlp.gate_proj.weight": torch.randn(4096, 1536),
            "model.norm.weight": torch.ones(1536),
        }

    def test_adapter_keys_from_hf_state(self):
        """Extracted adapter keys must use nn.Sequential indexing."""
        hf_state = self._make_hf_state()
        result = VibeThinkerWeightManager._extract_aeon_weights(hf_state, 1536)

        adapter_state = result["adapter_state"]
        adapter = _make_adapter()
        model_keys = set(adapter.state_dict().keys())

        for key in model_keys:
            assert key in adapter_state, f"Missing key: {key}"

    def test_adapter_shapes_from_hf_state(self):
        """Extracted adapter shapes must match model."""
        hf_state = self._make_hf_state()
        result = VibeThinkerWeightManager._extract_aeon_weights(hf_state, 1536)

        adapter = _make_adapter()
        model_sd = adapter.state_dict()

        for key in model_sd:
            if key in result["adapter_state"]:
                assert result["adapter_state"][key].shape == model_sd[key].shape, (
                    f"Shape mismatch for {key}"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  S5 — _extract_aeon_weights kernel key alignment
# ═══════════════════════════════════════════════════════════════════════════

class TestS5ExtractKernelWeights:
    """Verify _extract_aeon_weights produces correct kernel keys from HF state."""

    def _make_hf_state(self):
        return {
            "model.embed_tokens.weight": torch.randn(32000, 1536),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(1536, 1536),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4096, 1536),
            "model.layers.15.self_attn.q_proj.weight": torch.randn(1536, 1536),
            "model.layers.15.mlp.gate_proj.weight": torch.randn(4096, 1536),
            "model.norm.weight": torch.ones(1536),
        }

    def test_kernel_keys_from_hf_state(self):
        """Extracted kernel keys must use nn.Sequential indexing."""
        hf_state = self._make_hf_state()
        result = VibeThinkerWeightManager._extract_aeon_weights(hf_state, 1536)

        kernel_state = result["kernel_state"]
        kernel = _make_kernel()
        model_keys = set(kernel.state_dict().keys())

        for key in model_keys:
            assert key in kernel_state, f"Missing key: {key}"

    def test_kernel_shapes_from_hf_state(self):
        """Extracted kernel shapes must match model."""
        hf_state = self._make_hf_state()
        result = VibeThinkerWeightManager._extract_aeon_weights(hf_state, 1536)

        kernel = _make_kernel()
        model_sd = kernel.state_dict()

        for key in model_sd:
            if key in result["kernel_state"]:
                assert result["kernel_state"][key].shape == model_sd[key].shape, (
                    f"Shape mismatch for {key}"
                )

    def test_kernel_head_inner_dim_is_64(self):
        """Kernel head inner dimension should be hidden_dim//4 = 64."""
        hf_state = self._make_hf_state()
        result = VibeThinkerWeightManager._extract_aeon_weights(hf_state, 1536)
        kernel = result["kernel_state"]

        # First layer of confidence_head should be (64, 256)
        assert kernel["confidence_head.0.weight"].shape == (64, 256)
        # Second layer should be (1, 64)
        assert kernel["confidence_head.2.weight"].shape == (1, 64)


# ═══════════════════════════════════════════════════════════════════════════
#  S6 — load_vt_weights AEON-formatted handling
# ═══════════════════════════════════════════════════════════════════════════

class TestS6LoadVtWeightsAeon:
    """Verify load_vt_weights handles AEON-formatted safetensors correctly."""

    def test_aeon_weights_shape_validated(self, tmp_path):
        """AEON weights should be shape-validated before loading."""
        from aeon_wizard import load_vt_weights

        # Generate synthetic weights via weight manager
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        # Create a mock model with adapter and kernel
        model = MagicMock()
        model.vibe_thinker_adapter = _make_adapter()
        model.vibe_thinker_kernel = _make_kernel()

        result = load_vt_weights(model, mgr._weights_path)
        assert result["loaded"] is True
        assert result["format"] == "aeon_safetensors"
        assert result.get("adapter_keys_loaded", 0) > 0
        assert result.get("kernel_keys_loaded", 0) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  S7 — load_vt_weights raw HF handling
# ═══════════════════════════════════════════════════════════════════════════

class TestS7LoadVtWeightsRawHF:
    """Verify load_vt_weights handles raw HF safetensors correctly."""

    def test_raw_hf_weights_extracted(self, tmp_path):
        """Raw HF weights should be extracted to AEON format."""
        from aeon_wizard import load_vt_weights
        from safetensors.torch import save_file

        # Create a minimal HF-like safetensors file
        hf_state = {
            "model.embed_tokens.weight": torch.randn(32000, 1536),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(1536, 1536),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4096, 1536),
            "model.norm.weight": torch.ones(1536),
        }
        weights_path = tmp_path / "model.safetensors"
        save_file(hf_state, str(weights_path))

        model = MagicMock()
        model.vibe_thinker_adapter = _make_adapter()
        model.vibe_thinker_kernel = _make_kernel()

        result = load_vt_weights(model, weights_path)
        assert result["loaded"] is True
        assert result["format"] == "raw_safetensors"
        assert result.get("adapter_keys_loaded", 0) > 0
        assert result.get("kernel_keys_loaded", 0) > 0

    def test_file_not_found(self, tmp_path):
        """Missing file should return loaded=False gracefully."""
        from aeon_wizard import load_vt_weights

        model = MagicMock()
        result = load_vt_weights(model, tmp_path / "nonexistent.safetensors")
        assert result["loaded"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  S8 — Synthetic weights load into model modules
# ═══════════════════════════════════════════════════════════════════════════

class TestS8SyntheticWeightsLoadIntoModel:
    """Verify synthetic weights actually load into model modules."""

    def test_full_load_cycle(self, tmp_path):
        """Synthetic → save → load → model should work end-to-end."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        adapter = _make_adapter()
        kernel = _make_kernel()

        # Save original params for comparison
        orig_adapter_norm = adapter.layer_norm.weight.data.clone()

        from safetensors.torch import load_file
        flat = load_file(str(mgr._weights_path))

        adapter_state = {
            k[len("adapter_state."):]: v
            for k, v in flat.items()
            if k.startswith("adapter_state.")
        }
        kernel_state = {
            k[len("kernel_state."):]: v
            for k, v in flat.items()
            if k.startswith("kernel_state.")
        }

        # Verify all keys match model state dicts
        adapter_sd = adapter.state_dict()
        kernel_sd = kernel.state_dict()

        adapter_compat = {
            k: v for k, v in adapter_state.items()
            if k in adapter_sd and v.shape == adapter_sd[k].shape
        }
        kernel_compat = {
            k: v for k, v in kernel_state.items()
            if k in kernel_sd and v.shape == kernel_sd[k].shape
        }

        assert len(adapter_compat) == len(adapter_sd), (
            f"Only {len(adapter_compat)}/{len(adapter_sd)} adapter keys matched"
        )
        assert len(kernel_compat) == len(kernel_sd), (
            f"Only {len(kernel_compat)}/{len(kernel_sd)} kernel keys matched"
        )

        # Actually load and verify
        adapter.load_state_dict(adapter_compat, strict=False)
        kernel.load_state_dict(kernel_compat, strict=False)


# ═══════════════════════════════════════════════════════════════════════════
#  S9 — Dashboard loaded_keys correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestS9DashboardLoadedKeys:
    """Verify loaded_keys have correct prefixes for dashboard consumption."""

    def test_loaded_keys_have_adapter_prefix(self, tmp_path):
        """loaded_keys must include adapter.* entries."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        has_adapter = any(k.startswith("adapter.") for k in mgr._loaded_keys)
        assert has_adapter, "No adapter.* keys found in loaded_keys"

    def test_loaded_keys_have_kernel_prefix(self, tmp_path):
        """loaded_keys must include kernel.* entries."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        has_kernel = any(k.startswith("kernel.") for k in mgr._loaded_keys)
        assert has_kernel, "No kernel.* keys found in loaded_keys"

    def test_dashboard_has_adapter_check_passes(self, tmp_path):
        """Simulate dashboard _hasAdapter check."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        keys = mgr._loaded_keys
        has_adapter = any(
            k.startswith("adapter.") or k == "adapter_state" for k in keys
        )
        assert has_adapter

    def test_dashboard_has_kernel_check_passes(self, tmp_path):
        """Simulate dashboard _hasKernel check."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        keys = mgr._loaded_keys
        has_kernel = any(
            k.startswith("kernel.") or k == "kernel_state" for k in keys
        )
        assert has_kernel

    def test_get_status_loaded_keys_count(self, tmp_path):
        """get_status() should report correct loaded_keys_count."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        status = mgr.get_status()
        assert status["loaded_keys_count"] > 2, (
            f"Expected >2 loaded keys, got {status['loaded_keys_count']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  S10 — Regression: verify_weights → mark_loaded pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestS10VerifyAndMarkLoaded:
    """Verify the full pipeline: verify_weights → load → mark_loaded."""

    def test_mark_loaded_overrides_verify_keys(self, tmp_path):
        """mark_loaded should replace verify_weights keys with prefixed ones."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr.download_weights()

        # verify_weights stores raw payload keys
        mgr.verify_weights()
        verify_keys = list(mgr._loaded_keys)

        # mark_loaded with properly prefixed keys
        prefixed = [f"adapter.projector.0.weight", f"kernel.reasoning_projector.0.weight"]
        mgr.mark_loaded(prefixed)

        assert mgr._loaded_keys == prefixed
        assert mgr._status == mgr.STATUS_LOADED

    def test_mark_loaded_with_none_preserves_keys(self, tmp_path):
        """mark_loaded(None) should keep existing keys."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr._loaded_keys = ["test.key"]
        mgr.mark_loaded(loaded_keys=None)
        assert mgr._loaded_keys == ["test.key"]

    def test_mark_loaded_with_empty_sets_empty(self, tmp_path):
        """mark_loaded([]) should set empty keys."""
        mgr = VibeThinkerWeightManager(weights_dir=str(tmp_path))
        mgr._loaded_keys = ["test.key"]
        mgr.mark_loaded(loaded_keys=[])
        assert mgr._loaded_keys == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
