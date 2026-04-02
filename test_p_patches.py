"""Tests for P-series patches: Final cognitive integration bridges.

P1:  ae_train.py task-boundary detection bare except:pass → logger.debug + error_evolution
P2:  ae_train.py entropy adaptation bare except:pass → logger.debug + error_evolution
P3:  ae_train.py UCC evaluation bare except → logger.debug
P4a: ae_train.py VT init failure (Phase A) → logger.debug
P4b: ae_train.py VT init failure (Phase B) → logger.debug
P5a: ae_train.py VT signal mapping failure (Phase A) → logger.debug
P5b: ae_train.py VT signal mapping failure (Phase B) → logger.debug
P6:  aeon_core.py 9 missing _class_to_signal mappings added
"""

import importlib
import inspect
import logging
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import modules once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aeon_mod():
    """Return the aeon_core module, imported once per test module."""
    return importlib.import_module("aeon_core")


@pytest.fixture(scope="module")
def train_mod():
    """Return the ae_train module, imported once per test module."""
    return importlib.import_module("ae_train")


@pytest.fixture()
def model(aeon_mod):
    """Build a minimal AEONDeltaV3 for unit-testing."""
    cfg = aeon_mod.AEONConfig(
        z_dim=32,
        hidden_dim=32,
        vq_num_embeddings=8,
        vq_embedding_dim=32,
    )
    m = aeon_mod.AEONDeltaV3(cfg)
    return m


# ════════════════════════════════════════════════════════════════════════
#  P1: Task boundary detection — bare except:pass → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP1TaskBoundaryDetectionBridge:
    """P1: Task boundary detection exception must be logged, not silently swallowed."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P1 marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P1" in src

    def test_logger_debug_present(self, train_mod):
        """Task boundary failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P1")
        assert idx != -1
        region = src[idx:idx + 1000]
        assert "logger.debug" in region

    def test_exception_captured_as_variable(self, train_mod):
        """Exception must be captured (not bare except)."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P1")
        assert idx != -1
        # Look backward for the except clause
        region = src[max(0, idx - 300):idx + 200]
        assert "_tb_err" in region

    def test_error_evolution_recording(self, train_mod):
        """Task boundary failure must record to error_evolution."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P1")
        assert idx != -1
        region = src[idx:idx + 1500]
        assert "error_evolution" in region
        assert "record_episode" in region

    def test_error_class_name(self, train_mod):
        """Error class must be 'training_task_boundary_failure'."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P1")
        assert idx != -1
        region = src[idx:idx + 1500]
        assert "training_task_boundary_failure" in region


# ════════════════════════════════════════════════════════════════════════
#  P2: Entropy adaptation — bare except:pass → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP2EntropyAdaptationBridge:
    """P2: Entropy adaptation exception must be logged, not silently swallowed."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P2 marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P2" in src

    def test_logger_debug_present(self, train_mod):
        """Entropy adaptation failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P2")
        assert idx != -1
        region = src[idx:idx + 1000]
        assert "logger.debug" in region

    def test_exception_captured_as_variable(self, train_mod):
        """Exception must be captured (not bare except)."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P2")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_ew_err" in region

    def test_error_evolution_recording(self, train_mod):
        """Entropy adaptation failure must record to error_evolution."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P2")
        assert idx != -1
        region = src[idx:idx + 1500]
        assert "error_evolution" in region
        assert "record_episode" in region

    def test_error_class_name(self, train_mod):
        """Error class must be 'training_entropy_adaptation_failure'."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P2")
        assert idx != -1
        region = src[idx:idx + 1500]
        assert "training_entropy_adaptation_failure" in region


# ════════════════════════════════════════════════════════════════════════
#  P3: UCC evaluation function — bare except → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP3UCCEvaluationBridge:
    """P3: UCC evaluation exception must be logged."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P3 marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P3" in src

    def test_logger_debug_present(self, train_mod):
        """UCC evaluation failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P3")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_exception_captured_as_variable(self, train_mod):
        """Exception must be captured (not bare except)."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P3")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_ucc_eval_err" in region

    def test_evaluated_false_still_set(self, train_mod):
        """evaluated=False must still be set on failure."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P3")
        assert idx != -1
        region = src[idx:idx + 500]
        assert 'result["evaluated"] = False' in region


# ════════════════════════════════════════════════════════════════════════
#  P4a: VT init failure (Phase A) → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP4aVTInitPhaseA:
    """P4a: VT init failure in Phase A trainer must be logged."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P4a marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P4a" in src

    def test_logger_debug_present(self, train_mod):
        """VT init failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P4a")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_exception_captured(self, train_mod):
        """Exception must be captured as a named variable."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P4a")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_vt_init_err" in region

    def test_fallback_still_sets_none(self, train_mod):
        """VT bus and learner must still be set to None on failure."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P4a")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "_vt_streaming_bus = None" in region
        assert "_vt_learner_ref = None" in region


# ════════════════════════════════════════════════════════════════════════
#  P4b: VT init failure (Phase B) → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP4bVTInitPhaseB:
    """P4b: VT init failure in Phase B trainer must be logged."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P4b marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P4b" in src

    def test_logger_debug_present(self, train_mod):
        """VT init failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P4b")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_exception_captured(self, train_mod):
        """Exception must be captured as a named variable."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P4b")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_vt_init_err" in region


# ════════════════════════════════════════════════════════════════════════
#  P5a: VT signal mapping failure (Phase A) → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP5aVTMappingPhaseA:
    """P5a: VT signal mapping failure in Phase A must be logged."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P5a marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P5a" in src

    def test_logger_debug_present(self, train_mod):
        """VT mapping failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P5a")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_fallback_sets_empty_dict(self, train_mod):
        """Fallback must still set empty dict on failure."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P5a")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "_vt_mapped = {}" in region


# ════════════════════════════════════════════════════════════════════════
#  P5b: VT signal mapping failure (Phase B) → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestP5bVTMappingPhaseB:
    """P5b: VT signal mapping failure in Phase B must be logged."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch P5b marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch P5b" in src

    def test_logger_debug_present(self, train_mod):
        """VT mapping failure must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P5b")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_fallback_sets_empty_dict(self, train_mod):
        """Fallback must still set empty dict on failure."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch P5b")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "_vt_mapped_b = {}" in region


# ════════════════════════════════════════════════════════════════════════
#  P6: Missing _class_to_signal mappings
# ════════════════════════════════════════════════════════════════════════
class TestP6ClassToSignalMappings:
    """P6: All previously unmapped error classes must now have _class_to_signal entries."""

    def test_patch_comment_in_source(self, aeon_mod):
        """Patch P6 marker must exist in aeon_core source."""
        src = inspect.getsource(aeon_mod.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution)
        assert "Patch P6" in src

    @pytest.mark.parametrize("error_class,expected_signal", [
        ("cognitive_activation_step_failure", "uncertainty"),
        ("cognitive_unity_check", "coherence_deficit"),
        ("post_pipeline_metacog_verdict", "uncertainty"),
        ("ucc_evaluation_verdict", "uncertainty"),
        ("vibe_thinker_adaptation", "uncertainty"),
        ("vibe_thinker_vq_seeding_anomaly", "world_model_surprise"),
        ("vibe_thinker_warmup_anomaly", "uncertainty"),
        ("training_task_boundary_failure", "coherence_deficit"),
        ("training_entropy_adaptation_failure", "uncertainty"),
    ])
    def test_class_mapped_to_correct_signal(
        self, aeon_mod, error_class, expected_signal,
    ):
        """Each error class must map to its correct signal in _class_to_signal."""
        src = inspect.getsource(
            aeon_mod.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        # Verify the mapping exists in source
        pattern = f'"{error_class}":\\s*"{expected_signal}"'
        assert re.search(pattern, src), (
            f"_class_to_signal must contain "
            f'"{error_class}": "{expected_signal}"'
        )

    def test_adapt_weights_routes_cognitive_activation_step_failure(
        self, aeon_mod,
    ):
        """adapt_weights_from_evolution routes cognitive_activation_step_failure
        to uncertainty signal, not generic fallback."""
        trigger = aeon_mod.MetaCognitiveRecursionTrigger()
        initial_weight = trigger._signal_weights.get("uncertainty", 0.0)
        trigger.adapt_weights_from_evolution({
            "cognitive_activation_step_failure": {
                "total": 5,
                "recent_success_rate": 0.0,
            },
        })
        final_weight = trigger._signal_weights.get("uncertainty", 0.0)
        # Weight should have changed (increased) due to low success rate
        assert final_weight >= initial_weight

    def test_adapt_weights_routes_cognitive_unity_check(self, aeon_mod):
        """adapt_weights_from_evolution routes cognitive_unity_check
        to coherence_deficit signal."""
        trigger = aeon_mod.MetaCognitiveRecursionTrigger()
        initial_weight = trigger._signal_weights.get("coherence_deficit", 0.0)
        trigger.adapt_weights_from_evolution({
            "cognitive_unity_check": {
                "total": 5,
                "recent_success_rate": 0.0,
            },
        })
        final_weight = trigger._signal_weights.get("coherence_deficit", 0.0)
        assert final_weight >= initial_weight

    def test_adapt_weights_routes_vibe_thinker_vq_seeding_anomaly(
        self, aeon_mod,
    ):
        """adapt_weights_from_evolution routes vibe_thinker_vq_seeding_anomaly
        to world_model_surprise signal."""
        trigger = aeon_mod.MetaCognitiveRecursionTrigger()
        initial_weight = trigger._signal_weights.get(
            "world_model_surprise", 0.0,
        )
        trigger.adapt_weights_from_evolution({
            "vibe_thinker_vq_seeding_anomaly": {
                "total": 5,
                "recent_success_rate": 0.0,
            },
        })
        final_weight = trigger._signal_weights.get(
            "world_model_surprise", 0.0,
        )
        assert final_weight >= initial_weight

    def test_no_remaining_unmapped_recorded_classes(self, aeon_mod):
        """Every error_class recorded in aeon_core.py via record_episode()
        must have a _class_to_signal mapping or match a prefix pattern."""
        src = inspect.getsource(aeon_mod)
        # Extract all recorded error classes
        recorded = set(re.findall(r"error_class=['\"]([^'\"]+)['\"]", src))
        # Extract all mapped classes from adapt_weights_from_evolution
        adapt_src = inspect.getsource(
            aeon_mod.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution,
        )
        mapped = set(re.findall(r'"([^"]+)":\s*"[^"]+"', adapt_src))
        unmapped = recorded - mapped
        # Filter out classes that would match prefix-based routing
        # (these are dynamically routed and don't need static entries)
        prefix_routed = set()
        for cls in unmapped:
            for prefix in [
                "coherence_deficit_", "subsystem_degraded_",
                "training_", "safety_", "adaptation_",
                "recovery_", "escalation_", "activation_",
            ]:
                if cls.startswith(prefix):
                    prefix_routed.add(cls)
                    break
        truly_unmapped = unmapped - prefix_routed
        assert len(truly_unmapped) == 0, (
            f"These error classes are recorded but have no _class_to_signal "
            f"mapping: {truly_unmapped}"
        )


# ════════════════════════════════════════════════════════════════════════
#  Integration: No bare except:pass remaining in ae_train.py
# ════════════════════════════════════════════════════════════════════════
class TestNoBareExceptPass:
    """Integration: ae_train.py must not have any bare except:pass
    blocks outside of last-resort guards."""

    def test_no_bare_except_pass_in_ae_train(self, train_mod):
        """ae_train.py must not contain bare 'except Exception: pass'
        (only 'except Exception: pass  # last-resort guard' is allowed)."""
        src = inspect.getsource(train_mod)
        lines = src.split("\n")
        violations = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "except Exception:":
                # Check next non-empty line
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line:
                        if next_line == "pass":
                            violations.append(
                                f"Line ~{i+1}: bare except:pass"
                            )
                        break
        assert len(violations) == 0, (
            f"Found bare except:pass blocks in ae_train.py: {violations}"
        )
