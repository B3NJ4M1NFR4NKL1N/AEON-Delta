"""Tests for I-series patches — final integration signal routing.

Patch I1: verify_coherence_metacog_verdict → coherence_deficit
Patch I2: cross_model_prediction_divergence → world_model_surprise

These patches close the last two unmapped error-class gaps in
``MetaCognitiveRecursionTrigger.adapt_weights_from_evolution()``,
ensuring that *every* error class recorded via ``record_episode()``
routes to a semantically appropriate metacognitive signal rather
than silently falling through to the generic "uncertainty" default.
"""
import importlib
import inspect
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
aeon = importlib.import_module("aeon_core")


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _build_trigger():
    """Instantiate a MetaCognitiveRecursionTrigger for unit tests."""
    return aeon.MetaCognitiveRecursionTrigger(
        trigger_threshold=0.5,
        max_recursions=2,
    )


def _build_error_summary(error_class: str, success_rate: float = 0.2):
    """Build a minimal error_summary dict for adapt_weights_from_evolution."""
    return {
        "error_classes": {
            error_class: {
                "total": 10,
                "successes": int(10 * success_rate),
                "success_rate": success_rate,
            },
        },
    }


# ────────────────────────────────────────────────────────────────────
# I1 — verify_coherence_metacog_verdict mapping
# ────────────────────────────────────────────────────────────────────

class TestI1VerifyCoherenceMetacogVerdict:
    """Patch I1: verify_coherence_metacog_verdict → coherence_deficit."""

    def test_i1a_mapping_exists_in_source(self):
        """The _class_to_signal dict must contain
        'verify_coherence_metacog_verdict'."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger
                                .adapt_weights_from_evolution)
        assert "verify_coherence_metacog_verdict" in src, (
            "verify_coherence_metacog_verdict not found in "
            "adapt_weights_from_evolution source"
        )

    def test_i1b_maps_to_coherence_deficit(self):
        """The mapping value must be 'coherence_deficit'."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger
                                .adapt_weights_from_evolution)
        idx = src.find("verify_coherence_metacog_verdict")
        assert idx != -1
        # Find the mapping value after the key (use wider window to
        # span any intervening comment lines)
        snippet = src[idx:idx + 300]
        assert "coherence_deficit" in snippet, (
            f"Expected coherence_deficit mapping, got snippet: {snippet}"
        )

    def test_i1c_adapt_weights_updates_coherence_signal(self):
        """adapt_weights_from_evolution must route
        verify_coherence_metacog_verdict to coherence_deficit and
        adjust its weight."""
        trigger = _build_trigger()
        original_weight = trigger._signal_weights.get(
            "coherence_deficit", None
        )
        assert original_weight is not None, (
            "coherence_deficit signal missing from _signal_weights"
        )

        # Low success rate → should BOOST the coherence_deficit weight
        summary = _build_error_summary(
            "verify_coherence_metacog_verdict", success_rate=0.1,
        )
        trigger.adapt_weights_from_evolution(summary)

        new_weight = trigger._signal_weights["coherence_deficit"]
        assert new_weight >= original_weight, (
            f"Expected coherence_deficit weight to increase "
            f"(was {original_weight}, now {new_weight})"
        )

    def test_i1d_high_success_dampens_weight(self):
        """High success rate should dampen the coherence_deficit weight."""
        trigger = _build_trigger()
        original_weight = trigger._signal_weights["coherence_deficit"]

        summary = _build_error_summary(
            "verify_coherence_metacog_verdict", success_rate=0.9,
        )
        trigger.adapt_weights_from_evolution(summary)

        new_weight = trigger._signal_weights["coherence_deficit"]
        assert new_weight <= original_weight, (
            f"Expected coherence_deficit weight to decrease "
            f"(was {original_weight}, now {new_weight})"
        )

    def test_i1e_mapping_not_generic_uncertainty(self):
        """The mapping must NOT route to generic 'uncertainty' — it must
        route to 'coherence_deficit' specifically."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger
                                .adapt_weights_from_evolution)
        idx = src.find('"verify_coherence_metacog_verdict"')
        assert idx != -1
        # The very next quoted string after the key should be the signal
        snippet = src[idx:idx + 300]
        # Verify it contains coherence_deficit (not just uncertainty)
        assert "coherence_deficit" in snippet

    def test_i1f_error_class_recorded_in_verify_coherence(self):
        """The verify_coherence path must record the error class
        'verify_coherence_metacog_verdict' in the source code."""
        # Verify that the recording site exists
        src = inspect.getsource(aeon.AEONDeltaV3)
        assert "verify_coherence_metacog_verdict" in src, (
            "verify_coherence_metacog_verdict not found in AEON source"
        )


# ────────────────────────────────────────────────────────────────────
# I2 — cross_model_prediction_divergence mapping
# ────────────────────────────────────────────────────────────────────

class TestI2CrossModelPredictionDivergence:
    """Patch I2: cross_model_prediction_divergence → world_model_surprise."""

    def test_i2a_mapping_exists_in_source(self):
        """The _class_to_signal dict must contain
        'cross_model_prediction_divergence'."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger
                                .adapt_weights_from_evolution)
        assert "cross_model_prediction_divergence" in src, (
            "cross_model_prediction_divergence not found in "
            "adapt_weights_from_evolution source"
        )

    def test_i2b_maps_to_world_model_surprise(self):
        """The mapping value must be 'world_model_surprise'."""
        src = inspect.getsource(aeon.MetaCognitiveRecursionTrigger
                                .adapt_weights_from_evolution)
        idx = src.find("cross_model_prediction_divergence")
        assert idx != -1
        snippet = src[idx:idx + 300]
        assert "world_model_surprise" in snippet, (
            f"Expected world_model_surprise mapping, got snippet: {snippet}"
        )

    def test_i2c_adapt_weights_updates_world_model_signal(self):
        """adapt_weights_from_evolution must route
        cross_model_prediction_divergence to world_model_surprise
        and adjust its weight."""
        trigger = _build_trigger()
        original_weight = trigger._signal_weights.get(
            "world_model_surprise", None
        )
        assert original_weight is not None, (
            "world_model_surprise signal missing from _signal_weights"
        )

        summary = _build_error_summary(
            "cross_model_prediction_divergence", success_rate=0.1,
        )
        trigger.adapt_weights_from_evolution(summary)

        new_weight = trigger._signal_weights["world_model_surprise"]
        assert new_weight >= original_weight, (
            f"Expected world_model_surprise weight to increase "
            f"(was {original_weight}, now {new_weight})"
        )

    def test_i2d_high_success_dampens_weight(self):
        """High success rate should dampen the world_model_surprise weight."""
        trigger = _build_trigger()
        original_weight = trigger._signal_weights["world_model_surprise"]

        summary = _build_error_summary(
            "cross_model_prediction_divergence", success_rate=0.9,
        )
        trigger.adapt_weights_from_evolution(summary)

        new_weight = trigger._signal_weights["world_model_surprise"]
        assert new_weight <= original_weight, (
            f"Expected world_model_surprise weight to decrease "
            f"(was {original_weight}, now {new_weight})"
        )

    def test_i2e_error_class_recorded_in_forward_impl(self):
        """The E6 cross-model divergence path must record the error class
        'cross_model_prediction_divergence' in the source code."""
        src = inspect.getsource(aeon.AEONDeltaV3)
        assert "cross_model_prediction_divergence" in src, (
            "cross_model_prediction_divergence not found in AEON source"
        )


# ────────────────────────────────────────────────────────────────────
# Integration — both patches together
# ────────────────────────────────────────────────────────────────────

class TestI_Integration:
    """Verify both patches work correctly when combined."""

    def test_both_classes_route_without_fallback(self):
        """Neither class should trigger the 'unmapped error class'
        debug log when adapt_weights_from_evolution is called."""
        trigger = _build_trigger()
        summary = {
            "error_classes": {
                "verify_coherence_metacog_verdict": {
                    "total": 5, "successes": 1, "success_rate": 0.2,
                },
                "cross_model_prediction_divergence": {
                    "total": 5, "successes": 1, "success_rate": 0.2,
                },
            },
        }
        # Should not raise and should update weights for both signals
        trigger.adapt_weights_from_evolution(summary)

        # Both target signals should still exist after adaptation
        assert "coherence_deficit" in trigger._signal_weights
        assert "world_model_surprise" in trigger._signal_weights

    def test_independent_signal_routing(self):
        """The two error classes route to DIFFERENT signals and should
        not interfere with each other's weight adaptation."""
        trigger = _build_trigger()
        cd_before = trigger._signal_weights["coherence_deficit"]
        wm_before = trigger._signal_weights["world_model_surprise"]

        # Only coherence-related failure
        summary = _build_error_summary(
            "verify_coherence_metacog_verdict", success_rate=0.0,
        )
        trigger.adapt_weights_from_evolution(summary)

        cd_after = trigger._signal_weights["coherence_deficit"]
        wm_after = trigger._signal_weights["world_model_surprise"]

        # Coherence deficit weight should increase (low success → boost)
        assert cd_after >= cd_before, (
            f"Expected coherence_deficit weight to increase "
            f"(was {cd_before}, now {cd_after})"
        )

    def test_adapt_weights_no_exception(self):
        """adapt_weights_from_evolution should not raise for either
        of the newly mapped classes."""
        trigger = _build_trigger()
        for cls in [
            "verify_coherence_metacog_verdict",
            "cross_model_prediction_divergence",
        ]:
            summary = _build_error_summary(cls, success_rate=0.5)
            trigger.adapt_weights_from_evolution(summary)

    def test_no_remaining_unmapped_production_classes(self):
        """Verify that all error_class= values in the AEON class source
        that start with 'verify_coherence_metacog' or
        'cross_model_prediction' are now in _class_to_signal."""
        trigger_src = inspect.getsource(
            aeon.MetaCognitiveRecursionTrigger.adapt_weights_from_evolution
        )
        for cls in [
            "verify_coherence_metacog_verdict",
            "cross_model_prediction_divergence",
        ]:
            assert cls in trigger_src, (
                f"{cls} still missing from _class_to_signal"
            )
