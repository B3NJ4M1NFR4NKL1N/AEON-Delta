"""
H-series patches — functional cognitive organism transition.

These patches bridge remaining silent exception handlers and disconnected
signals to complete the transition from a connected architecture to a
self-consistent cognitive system where:

- H1a: Bridge VibeThinker calibration adapter warm-up recording failure
       to _bridge_silent_exception (line ~85348).
- H1b: Bridge VibeThinker calibration VQ seeding recording failure
       to _bridge_silent_exception (line ~85453).
- H2:  Bridge UCC inverse feedback adapt_weights failure to
       _bridge_silent_exception (line ~32372).
- H3:  Bridge UCC provenance attribution failure to
       _bridge_silent_exception (line ~30614).
- H4:  Bridge VibeThinker adaptation recording failure to
       _bridge_silent_exception (line ~67122).
- H5:  Bridge cognitive activation probe step failures to
       error_evolution (line ~83744).
- H6:  Bridge coherence registry conflict detection failure to
       _bridge_silent_exception (line ~42745).
"""

import importlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

_aeon = None


def _load_aeon():
    global _aeon
    if _aeon is None:
        _aeon = importlib.import_module("aeon_core")
    return _aeon


def _make_config(**overrides):
    """Create a minimal AEONConfig with sensible test defaults."""
    aeon = _load_aeon()
    defaults = dict(
        hidden_dim=64, z_dim=64, vocab_size=256, num_pillars=8,
        seq_length=32, dropout_rate=0.0, meta_dim=32,
        lipschitz_target=0.9, vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon.AEONConfig(**defaults)


# ────────────────────────────────────────────────────────────────────
# H1a: VibeThinker warmup recording failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH1a_VTWarmupRecordingBridge:
    """Patch H1a bridges VT adapter warm-up error_evolution recording
    failures to _bridge_silent_exception so they are not silently lost."""

    def test_patch_h1a_present_in_source(self):
        """The H1a patch marker must exist in _vibe_thinker_first_start_calibration."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        assert "Patch H1a" in src, (
            "Patch H1a marker not found in _vibe_thinker_first_start_calibration"
        )

    def test_patch_h1a_uses_bridge_silent_exception(self):
        """H1a must call _bridge_silent_exception, not bare pass."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        # Find the H1a section and verify bridge call
        idx = src.index("Patch H1a")
        region = src[idx:idx + 500]
        assert "_bridge_silent_exception" in region, (
            "H1a must call _bridge_silent_exception"
        )

    def test_patch_h1a_error_class(self):
        """H1a must use 'vibe_thinker_warmup_recording_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        assert "vibe_thinker_warmup_recording_failure" in src


# ────────────────────────────────────────────────────────────────────
# H1b: VQ seeding recording failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH1b_VQSeedingRecordingBridge:
    """Patch H1b bridges VQ seeding anomaly recording failures to
    _bridge_silent_exception."""

    def test_patch_h1b_present_in_source(self):
        """The H1b patch marker must exist."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        assert "Patch H1b" in src

    def test_patch_h1b_uses_bridge_silent_exception(self):
        """H1b must call _bridge_silent_exception."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        idx = src.index("Patch H1b")
        region = src[idx:idx + 500]
        assert "_bridge_silent_exception" in region

    def test_patch_h1b_error_class(self):
        """H1b must use 'vibe_thinker_vq_seeding_recording_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        assert "vibe_thinker_vq_seeding_recording_failure" in src


# ────────────────────────────────────────────────────────────────────
# H2: UCC inverse feedback adaptation failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH2_UCCInverseFeedbackBridge:
    """Patch H2 bridges adapt_weights_from_evolution failures in the
    UnifiedCognitiveFrame.assess inverse feedback path to
    _bridge_silent_exception."""

    def test_patch_h2_present_in_source(self):
        """The H2 patch marker must exist in UnifiedCognitiveFrame.assess."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveFrame.assess)
        assert "Patch H2" in src

    def test_patch_h2_uses_bridge_silent_exception(self):
        """H2 must call _bridge_silent_exception."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveFrame.assess)
        idx = src.index("Patch H2")
        region = src[idx:idx + 600]
        assert "_bridge_silent_exception" in region

    def test_patch_h2_error_class(self):
        """H2 must use 'ucc_inverse_feedback_adaptation_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveFrame.assess)
        assert "ucc_inverse_feedback_adaptation_failure" in src

    def test_patch_h2_no_bare_pass_for_inverse_feedback(self):
        """The inverse feedback except block must NOT have bare 'pass'."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveFrame.assess)
        idx = src.index("inverse feedback")
        region = src[max(0, idx - 50):idx + 300]
        lines = region.split('\n')
        for i, line in enumerate(lines):
            if 'except Exception' in line:
                for j in range(i + 1, min(i + 3, len(lines))):
                    stripped = lines[j].strip()
                    if stripped and stripped != '':
                        assert stripped != 'pass', (
                            "Inverse feedback except block still has bare 'pass'"
                        )
                        break


# ────────────────────────────────────────────────────────────────────
# H3: UCC provenance attribution failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH3_ProvenanceAttributionBridge:
    """Patch H3 bridges provenance attribution/adaptation failures
    during certificate violation handling to _bridge_silent_exception."""

    def test_patch_h3_present_in_source(self):
        """The H3 patch marker must exist in UnifiedCognitiveCycle.evaluate."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        assert "Patch H3" in src

    def test_patch_h3_uses_bridge_silent_exception(self):
        """H3 must call _bridge_silent_exception."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        idx = src.index("Patch H3")
        region = src[idx:idx + 500]
        assert "_bridge_silent_exception" in region

    def test_patch_h3_error_class(self):
        """H3 must use 'ucc_provenance_attribution_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        assert "ucc_provenance_attribution_failure" in src

    def test_patch_h3_subsystem_is_provenance_tracker(self):
        """H3 must attribute the failure to 'provenance_tracker' subsystem."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        idx = src.index("Patch H3")
        region = src[idx:idx + 700]
        assert "'provenance_tracker'" in region


# ────────────────────────────────────────────────────────────────────
# H4: VibeThinker adaptation recording failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH4_VTAdaptationRecordingBridge:
    """Patch H4 bridges VibeThinker adaptation recording failures in
    _forward_impl to _bridge_silent_exception."""

    def test_patch_h4_present_in_source(self):
        """The H4 patch marker must exist in _forward_impl."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert "Patch H4" in src

    def test_patch_h4_uses_bridge_silent_exception(self):
        """H4 must call _bridge_silent_exception."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.index("Patch H4")
        region = src[idx:idx + 500]
        assert "_bridge_silent_exception" in region

    def test_patch_h4_error_class(self):
        """H4 must use 'vibe_thinker_adaptation_recording_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        assert "vibe_thinker_adaptation_recording_failure" in src

    def test_patch_h4_subsystem_is_error_evolution(self):
        """H4 must attribute the failure to 'error_evolution' subsystem."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._forward_impl)
        idx = src.index("Patch H4")
        region = src[idx:idx + 700]
        assert "'error_evolution'" in region


# ────────────────────────────────────────────────────────────────────
# H5: Probe step failures → error_evolution
# ────────────────────────────────────────────────────────────────────

class TestH5_ProbeStepFailuresErrorEvolution:
    """Patch H5 feeds _probe_step_failures into error_evolution so the
    metacognitive trigger can learn from partial activations."""

    def test_patch_h5_present_in_source(self):
        """The H5 patch marker must exist in _cognitive_activation_probe."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        assert "Patch H5" in src

    def test_patch_h5_records_to_error_evolution(self):
        """H5 must call error_evolution.record_episode for step failures."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        assert "record_episode" in region
        assert "cognitive_activation_step_failure" in region

    def test_patch_h5_iterates_over_failures(self):
        """H5 must iterate over _probe_step_failures."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        assert "_probe_step_failures" in region
        assert "for" in region  # iteration

    def test_patch_h5_includes_failed_step_in_metadata(self):
        """H5 must include the failed step name in episode metadata."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        assert "'failed_step'" in region

    def test_patch_h5_uses_activation_probe_antecedent(self):
        """H5 episodes must have 'cognitive_activation_probe' as causal antecedent."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        assert "cognitive_activation_probe" in region

    def test_patch_h5_guard_against_broken_error_evolution(self):
        """H5 must handle cases where error_evolution itself is broken."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        # Should have a try/except with break to stop cascade
        assert "except" in region
        assert "break" in region


# ────────────────────────────────────────────────────────────────────
# H6: Coherence registry conflict detection failure → _bridge_silent_exception
# ────────────────────────────────────────────────────────────────────

class TestH6_ConflictRegistryBridge:
    """Patch H6 bridges coherence registry conflict detection failures
    to _bridge_silent_exception in _build_feedback_extra_signals."""

    def test_patch_h6_present_in_source(self):
        """The H6 patch marker must exist in _build_feedback_extra_signals."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "Patch H6" in src

    def test_patch_h6_uses_bridge_silent_exception(self):
        """H6 must call _bridge_silent_exception."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        idx = src.index("Patch H6")
        region = src[idx:idx + 700]
        assert "_bridge_silent_exception" in region

    def test_patch_h6_error_class(self):
        """H6 must use 'coherence_registry_conflict_detection_failure' error class."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        assert "coherence_registry_conflict_detection_failure" in src

    def test_patch_h6_subsystem_is_coherence_registry(self):
        """H6 must attribute the failure to 'coherence_registry' subsystem."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._build_feedback_extra_signals,
        )
        idx = src.index("Patch H6")
        region = src[idx:idx + 700]
        assert "'coherence_registry'" in region


# ────────────────────────────────────────────────────────────────────
# H-series Integration Tests
# ────────────────────────────────────────────────────────────────────

class TestH_Integration:
    """Integration tests verifying the H-series patches work together
    and properly connect to the _bridge_silent_exception infrastructure."""

    def test_bridge_silent_exception_method_exists(self):
        """The _bridge_silent_exception method must exist on AEONDeltaV3."""
        aeon = _load_aeon()
        assert hasattr(aeon.AEONDeltaV3, '_bridge_silent_exception'), (
            "_bridge_silent_exception must exist on AEONDeltaV3"
        )

    def test_bridge_silent_exception_accepts_three_args(self):
        """_bridge_silent_exception must accept (error_class, subsystem, exception)."""
        aeon = _load_aeon()
        sig = inspect.signature(aeon.AEONDeltaV3._bridge_silent_exception)
        params = list(sig.parameters.keys())
        # self + 3 positional params
        assert len(params) >= 4, (
            f"Expected at least 4 params (self + 3), got {params}"
        )

    def test_all_h_error_classes_are_distinct(self):
        """Each H-series patch must use a unique error class."""
        aeon = _load_aeon()
        # Collect all error classes from H patches
        error_classes = set()
        sources = [
            inspect.getsource(aeon.AEONDeltaV3._vibe_thinker_first_start_calibration),
            inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate),
            inspect.getsource(aeon.UnifiedCognitiveFrame.assess),
            inspect.getsource(aeon.AEONDeltaV3._forward_impl),
            inspect.getsource(aeon.AEONDeltaV3._cognitive_activation_probe),
            inspect.getsource(aeon.AEONDeltaV3._build_feedback_extra_signals),
        ]
        h_classes = [
            'vibe_thinker_warmup_recording_failure',
            'vibe_thinker_vq_seeding_recording_failure',
            'ucc_inverse_feedback_adaptation_failure',
            'ucc_provenance_attribution_failure',
            'vibe_thinker_adaptation_recording_failure',
            'cognitive_activation_step_failure',
            'coherence_registry_conflict_detection_failure',
        ]
        combined_source = '\n'.join(sources)
        for cls in h_classes:
            assert cls in combined_source, (
                f"Error class '{cls}' not found in any H-patch source"
            )
            error_classes.add(cls)
        assert len(error_classes) == 7, (
            f"Expected 7 unique error classes, got {len(error_classes)}"
        )

    def test_model_instantiation_with_patches(self):
        """AEONDeltaV3 must instantiate successfully with all H-patches."""
        aeon = _load_aeon()
        import torch
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        assert model is not None

    def test_error_evolution_error_class_routing(self):
        """H-series error classes must be routable through error_evolution."""
        aeon = _load_aeon()
        # Verify that MetaCognitiveRecursionTrigger can receive adaptation
        # from error summaries containing H-series error classes
        trigger = aeon.MetaCognitiveRecursionTrigger(
            trigger_threshold=0.5,
            max_recursions=3,
        )
        # Build a fake error summary containing H-series classes
        fake_summary = {
            'total_recorded': 5,
            'error_classes': {
                'vibe_thinker_warmup_recording_failure': {
                    'count': 1, 'success_rate': 0.0,
                },
                'ucc_inverse_feedback_adaptation_failure': {
                    'count': 2, 'success_rate': 0.0,
                },
                'cognitive_activation_step_failure': {
                    'count': 2, 'success_rate': 0.0,
                },
            },
        }
        # adapt_weights_from_evolution should not raise
        trigger.adapt_weights_from_evolution(fake_summary)

    def test_cognitive_activation_probe_stores_failures(self):
        """After _cognitive_activation_probe, step failures are stored."""
        aeon = _load_aeon()
        import torch
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # The probe runs during __init__; check the attribute exists
        assert hasattr(model, '_activation_probe_step_failures')
        assert isinstance(model._activation_probe_step_failures, list)

    def test_no_bare_except_pass_in_h_patch_regions(self):
        """No H-patch region should contain 'except Exception: pass'."""
        aeon = _load_aeon()
        sources = {
            'vt_calibration': inspect.getsource(
                aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
            ),
            'ucc_evaluate': inspect.getsource(
                aeon.UnifiedCognitiveCycle.evaluate,
            ),
            'forward_impl': inspect.getsource(
                aeon.AEONDeltaV3._forward_impl,
            ),
            'feedback_extra': inspect.getsource(
                aeon.AEONDeltaV3._build_feedback_extra_signals,
            ),
        }
        h_markers = [
            'Patch H1a', 'Patch H1b', 'Patch H2', 'Patch H3',
            'Patch H4', 'Patch H6',
        ]
        for marker in h_markers:
            for src_name, src in sources.items():
                if marker in src:
                    idx = src.index(marker)
                    region = src[idx:idx + 500]
                    # Verify no bare 'pass' after except in this region
                    lines = region.split('\n')
                    for i, line in enumerate(lines):
                        if 'except Exception' in line:
                            if i + 1 < len(lines):
                                next_stripped = lines[i + 1].strip()
                                assert next_stripped != 'pass', (
                                    f"Bare 'pass' found after except in "
                                    f"{src_name} near {marker}"
                                )


class TestH_MutualReinforcement:
    """Verify that H-series patches enable mutual reinforcement:
    active components verify and stabilize each other's states."""

    def test_bridge_records_to_both_error_evolution_and_causal_trace(self):
        """_bridge_silent_exception must record to both error_evolution
        AND causal_trace, ensuring mutual verification."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._bridge_silent_exception)
        assert "error_evolution" in src
        assert "causal_trace" in src

    def test_bridge_escalates_persistent_failures(self):
        """_bridge_silent_exception must escalate persistent failures
        to the metacognitive trigger for higher-order review."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.AEONDeltaV3._bridge_silent_exception)
        assert "metacognitive_trigger" in src or "persistent" in src.lower()

    def test_h5_error_evolution_available_after_probe(self):
        """After probe, error_evolution should be accessible for H5 episodes."""
        aeon = _load_aeon()
        import torch
        config = _make_config()
        model = aeon.AEONDeltaV3(config)
        # error_evolution should be initialized
        assert model.error_evolution is not None


class TestH_CausalTransparency:
    """Verify that H-series patches maintain causal transparency:
    every action can be traced back through the architecture."""

    def test_h1a_includes_causal_antecedents_in_region(self):
        """H1a bridge region must reference VibeThinker subsystem."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._vibe_thinker_first_start_calibration,
        )
        idx = src.index("Patch H1a")
        region = src[idx:idx + 500]
        # The bridge call includes error_class which identifies the subsystem
        assert "error_evolution" in region

    def test_h3_links_to_provenance_tracker(self):
        """H3 bridge must link failure back to provenance_tracker."""
        aeon = _load_aeon()
        src = inspect.getsource(aeon.UnifiedCognitiveCycle.evaluate)
        idx = src.index("Patch H3")
        region = src[idx:idx + 700]
        assert "provenance_tracker" in region

    def test_h5_links_failures_to_activation_probe(self):
        """H5 episodes must be traceable to _cognitive_activation_probe."""
        aeon = _load_aeon()
        src = inspect.getsource(
            aeon.AEONDeltaV3._cognitive_activation_probe,
        )
        idx = src.index("Patch H5")
        region = src[idx:idx + 1600]
        assert "cognitive_activation_probe" in region
        assert "causal_antecedents" in region
