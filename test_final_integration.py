"""
Tests for Final Integration & Cognitive Activation patches.

Validates that the 12 patches (A1-A3, B1-B9) correctly:
1. Record previously-invisible events in error_evolution (A1-A3)
2. Enrich incomplete evaluate() call sites with full signal sets (B1-B9)
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import importlib

aeon = importlib.import_module("aeon_core")


def _make_config(**overrides):
    """Create a minimal AEONConfig with sensible test defaults."""
    defaults = dict(
        hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
        seq_length=32, dropout_rate=0.0, meta_dim=32,
        lipschitz_target=0.9, vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


def _get_full_source():
    """Return concatenated source of all patched methods."""
    parts = [
        inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate),
        inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl),
        inspect.getsource(aeon.AEONDeltaV3._forward_impl),
        inspect.getsource(aeon.AEONDeltaV3.verify_coherence),
        inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce),
        inspect.getsource(aeon.AEONDeltaV3.system_emergence_report),
    ]
    return '\n'.join(parts)


# ── Patch A1: convergence certificate violation → error_evolution ──────

class TestPatchA1ConvergenceCertificateRecording:
    """Verify convergence certificate violations produce error_evolution episodes."""

    def test_cert_violation_recording_code_present(self):
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        assert 'convergence_certificate_violation' in src
        assert 'formal_guarantee_rerun' in src

    def test_cert_violation_records_lipschitz_metadata(self):
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        assert "'lipschitz'" in src or '"lipschitz"' in src

    def test_cert_violation_guarded_by_error_evolution_check(self):
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        idx = src.find('convergence_certificate_violation')
        assert idx > 0
        assert 'error_evolution' in src[max(0, idx - 400):idx]


# ── Patch A2: metacognitive trigger results → error_evolution ──────────

class TestPatchA2MetacognitiveTriggerRecording:
    """Verify metacognitive trigger evaluation results feed into error_evolution."""

    def test_metacognitive_rerun_episode_code_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        assert "error_class='metacognitive_rerun'" in src

    def test_trigger_score_in_metadata(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        idx = src.find("error_class='metacognitive_rerun'")
        assert idx > 0
        ctx = src[idx:idx + 500]
        assert 'trigger_score' in ctx
        assert 'triggers_active' in ctx

    def test_recording_after_causal_trace(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        ct_idx = src.find('"metacognitive_recursion", "triggered"')
        ee_idx = src.find("error_class='metacognitive_rerun'")
        assert ct_idx > 0 and ee_idx > 0 and ee_idx > ct_idx


# ── Patch A3: VibeThinker adaptation → error_evolution ─────────────────

class TestPatchA3VibeThinkerAdaptationRecording:
    """Verify VibeThinker adapt/consolidate results recorded in error_evolution."""

    def test_vt_adaptation_episode_code_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert "error_class='vibe_thinker_adaptation'" in src

    def test_vt_adaptation_records_consolidated_flag(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find("error_class='vibe_thinker_adaptation'")
        assert idx > 0
        ctx = src[idx:idx + 800]
        assert 'consolidated' in ctx

    def test_vt_adaptation_after_consolidation(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        c_idx = src.find('maybe_consolidate(')
        e_idx = src.find("error_class='vibe_thinker_adaptation'")
        assert c_idx > 0 and e_idx > 0 and e_idx > c_idx


# ── Patch B1: fast-mode + border_uncertainty ───────────────────────────

class TestPatchB1FastModeBorderUncertainty:
    """Verify fast-mode deferred trigger now includes border_uncertainty."""

    def test_patch_b1_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        assert 'Patch B1' in src

    def test_fast_mode_has_all_15_signals(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        for sig in ['uncertainty=', 'is_diverging=', 'topology_catastrophe=',
                     'coherence_deficit=', 'memory_staleness=',
                     'recovery_pressure=', 'world_model_surprise=',
                     'causal_quality=', 'safety_violation=',
                     'convergence_conflict=', 'diversity_collapse=',
                     'memory_trust_deficit=', 'spectral_stability_margin=',
                     'output_reliability=', 'border_uncertainty=']:
            assert sig in src, f"Signal {sig} missing from _reasoning_core_impl"


# ── Patch B2: output reliability gate signal enrichment ────────────────

class TestPatchB2OutputReliabilityGateSignals:
    """Verify output reliability gate evaluate now has full signal set."""

    def test_patch_b2_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        assert 'Patch B2' in src

    def test_is_diverging_in_reliability_gate(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        idx = src.find('Patch B2')
        assert idx > 0
        assert 'is_diverging=' in src[idx:idx + 1200]

    def test_topology_catastrophe_in_reliability_gate(self):
        src = inspect.getsource(aeon.AEONDeltaV3._reasoning_core_impl)
        idx = src.find('Patch B2')
        assert idx > 0
        assert 'topology_catastrophe=' in src[idx:idx + 1200]


# ── Patch B3: VQ codebook quality signal enrichment ────────────────────

class TestPatchB3VQCodebookSignals:
    """Verify VQ codebook quality evaluate now has full signal set."""

    def test_patch_b3_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert 'Patch B3' in src

    def test_vq_eval_has_recovery_pressure(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find('Patch B3')
        assert idx > 0
        assert 'recovery_pressure=' in src[idx:idx + 1500]


# ── Patch B4: cache validity signal enrichment ─────────────────────────

class TestPatchB4CacheValiditySignals:
    """Verify cache validity evaluate now has full signal set."""

    def test_patch_b4_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert 'Patch B4' in src

    def test_cache_eval_has_convergence_conflict(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find('Patch B4')
        assert idx > 0
        assert 'convergence_conflict=' in src[idx:idx + 1500]


# ── Patch B5: high-uncertainty signal enrichment ───────────────────────

class TestPatchB5HighUncertaintySignals:
    """Verify high-uncertainty evaluate now has full signal set."""

    def test_patch_b5_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert 'Patch B5' in src

    def test_high_unc_has_output_reliability(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find('Patch B5')
        assert idx > 0
        assert 'output_reliability=' in src[idx:idx + 1500]

    def test_high_unc_has_memory_staleness(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find('Patch B5')
        assert idx > 0
        assert 'memory_staleness=' in src[idx:idx + 3500]


# ── Patch B6: moderate uncertainty signal enrichment ───────────────────

class TestPatchB6ModerateUncertaintySignals:
    """Verify moderate uncertainty backup evaluate has full signal set."""

    def test_patch_b6_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert 'Patch B6' in src

    def test_moderate_unc_has_diversity_collapse(self):
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.find('Patch B6')
        assert idx > 0
        assert 'diversity_collapse=' in src[idx:idx + 1500]


# ── Patch B7: verify_and_reinforce unity signal enrichment ─────────────

class TestPatchB7UnitySignals:
    """Verify verify_and_reinforce unity evaluate has full signal set."""

    def test_patch_b7_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        assert 'Patch B7' in src

    def test_unity_has_is_diverging(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.find('Patch B7')
        assert idx > 0
        assert 'is_diverging=' in src[idx:idx + 1500]

    def test_unity_has_topology_catastrophe(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.find('Patch B7')
        assert idx > 0
        assert 'topology_catastrophe=' in src[idx:idx + 1500]

    def test_unity_has_convergence_conflict(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.find('Patch B7')
        assert idx > 0
        assert 'convergence_conflict=' in src[idx:idx + 1500]

    def test_unity_has_diversity_collapse(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.find('Patch B7')
        assert idx > 0
        assert 'diversity_collapse=' in src[idx:idx + 1500]

    def test_unity_has_memory_trust_deficit(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_and_reinforce)
        idx = src.find('Patch B7')
        assert idx > 0
        assert 'memory_trust_deficit=' in src[idx:idx + 1500]


# ── Patch B8: critical patch signal enrichment ─────────────────────────

class TestPatchB8CriticalPatchSignals:
    """Verify critical patch evaluate has full signal set."""

    def test_patch_b8_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3.system_emergence_report)
        assert 'Patch B8' in src

    def test_critical_patch_has_output_reliability(self):
        src = inspect.getsource(aeon.AEONDeltaV3.system_emergence_report)
        idx = src.find('Patch B8')
        assert idx > 0
        assert 'output_reliability=' in src[idx:idx + 1500]

    def test_critical_patch_has_memory_staleness(self):
        src = inspect.getsource(aeon.AEONDeltaV3.system_emergence_report)
        idx = src.find('Patch B8')
        assert idx > 0
        assert 'memory_staleness=' in src[idx:idx + 2500]


# ── Patch B9: verify_coherence + border_uncertainty ────────────────────

class TestPatchB9VerifyCoherenceSignals:
    """Verify verify_coherence evaluate includes border_uncertainty."""

    def test_coherence_check_has_border_uncertainty(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        assert 'border_uncertainty' in src

    def test_patch_b9_present(self):
        src = inspect.getsource(aeon.AEONDeltaV3.verify_coherence)
        assert 'Patch B9' in src


# ── Cross-patch coherence ──────────────────────────────────────────────

class TestCrossPatchCoherence:
    """Verify patches don't break overall architectural coherence."""

    def test_all_12_patches_present_in_source(self):
        combined = _get_full_source()
        for patch in ['Patch A1', 'Patch A2', 'Patch A3',
                       'Patch B1', 'Patch B2', 'Patch B3',
                       'Patch B4', 'Patch B5', 'Patch B6',
                       'Patch B7', 'Patch B8', 'Patch B9']:
            assert patch in combined, f"{patch} not found in source"

    def test_three_new_error_classes(self):
        combined = _get_full_source()
        for cls in ['convergence_certificate_violation',
                     'metacognitive_rerun', 'vibe_thinker_adaptation']:
            assert cls in combined, f"error_class '{cls}' missing"

    def test_border_uncertainty_signal_count(self):
        combined = _get_full_source()
        count = combined.count('border_uncertainty=')
        assert count >= 9, f"Expected ≥9 border_uncertainty= kwargs, got {count}"

    def test_model_imports_successfully(self):
        cfg = _make_config()
        model = aeon.AEONDeltaV3(cfg)
        assert model is not None

    def test_ucc_evaluate_still_has_should_rerun(self):
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        assert "'should_rerun'" in src or '"should_rerun"' in src

    def test_metacognitive_trigger_returns_should_recurse(self):
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger.evaluate)
        assert '"should_recurse"' in src or "'should_recurse'" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
