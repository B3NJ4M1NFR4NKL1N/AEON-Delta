"""Tests for PATCH-RMT34-A and PATCH-RMT34-B.

PATCH-RMT34-A: ``CognitiveFeedbackBus._build_projection`` now also
rebuilds the wired ``FeedbackSignalAttention`` so the CA-8 / S-5
selective-attention branch survives dynamic signal extension instead of
silently failing with a shape-mismatch and falling back to plain
projection.

PATCH-RMT34-B: ``MetaCognitiveRecursionTrigger._class_to_signal`` now
maps ``unrouted_feedback_signal`` → ``coherence_deficit`` so routing
regressions surfaced by PATCH-RESID-D actually boost the correct MCT
trigger weight bucket instead of being flattened to the generic
``uncertainty`` fallback.

Together they close two intra-pass discontinuities in the cognitive
flow that were both reproducing on every forward pass.
"""

from __future__ import annotations

import io
import logging
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")

from aeon_core import (  # noqa: E402  (import after warning filter)
    AEONConfig,
    AEONDeltaV3,
    CognitiveFeedbackBus,
    FeedbackSignalAttention,
    MetaCognitiveRecursionTrigger,
)


# ────────────────────────────────────────────────────────────────────
# PATCH-RMT34-A: FeedbackSignalAttention follows bus growth
# ────────────────────────────────────────────────────────────────────


class TestPatchRMT34A_AttentionFollowsBusGrowth:
    def test_register_signal_rebuilds_signal_attention(self):
        """register_signal must resize the wired FeedbackSignalAttention."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        attn = FeedbackSignalAttention(
            num_signals=bus.NUM_SIGNAL_CHANNELS, hidden_dim=32,
        )
        bus._signal_attention = attn

        assert bus._signal_attention.num_signals == bus.NUM_SIGNAL_CHANNELS

        bus.register_signal("custom_signal_a", 0.0)
        bus.register_signal("custom_signal_b", 0.0)

        expected = bus.NUM_SIGNAL_CHANNELS + 2
        assert bus._signal_attention.num_signals == expected
        # And the inner Linear actually has the new in_features.
        assert bus._signal_attention.attention.in_features == expected

    def test_attention_path_does_not_silently_fail(self):
        """After signal extension, bus.forward must use the attention
        branch (no shape-mismatch exception, no fallback)."""
        bus = CognitiveFeedbackBus(hidden_dim=32)
        bus._signal_attention = FeedbackSignalAttention(
            num_signals=bus.NUM_SIGNAL_CHANNELS, hidden_dim=32,
        )
        for i in range(20):
            bus.register_signal(f"extra_{i}", 0.0)

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        try:
            out = bus(batch_size=1, device=torch.device("cpu"))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)

        assert out.shape == (1, 32)
        assert "S-5: FeedbackSignalAttention failed" not in buf.getvalue()

    def test_attention_branch_actually_contributes_to_output(self):
        """With distinct attention weights, the bus output should differ
        from the pure-projection fallback (proves attention path is live).
        """
        bus = CognitiveFeedbackBus(hidden_dim=32)
        bus._signal_attention = FeedbackSignalAttention(
            num_signals=bus.NUM_SIGNAL_CHANNELS, hidden_dim=32,
        )
        # Add several extras to exercise the resize path too.
        for i in range(5):
            bus.register_signal(f"extra_{i}", 0.0)
        # Force attention to produce a distinguishable signature.
        with torch.no_grad():
            bus._signal_attention.signal_projection.weight.fill_(1.0)
            bus._signal_attention.signal_projection.bias.fill_(0.5)

        with_attn = bus(batch_size=1, device=torch.device("cpu"))

        # Now detach attention and compare.
        bus._signal_attention = None
        without_attn = bus(batch_size=1, device=torch.device("cpu"))

        assert not torch.allclose(with_attn, without_attn)

    def test_full_model_forward_no_shape_mismatch_logs(self):
        """End-to-end: a single forward pass must produce zero
        ``S-5: FeedbackSignalAttention failed`` log lines."""
        torch.manual_seed(0)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        try:
            model = AEONDeltaV3(AEONConfig())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)

        log_text = buf.getvalue()
        assert "S-5: FeedbackSignalAttention failed" not in log_text, (
            "FeedbackSignalAttention is still silently failing — "
            "PATCH-RMT34-A did not fully bridge the bus → attention path."
        )
        # Attention module width must equal the bus width.
        bus = model.feedback_bus
        expected = bus.NUM_SIGNAL_CHANNELS + len(bus._extra_signals)
        assert bus._signal_attention.num_signals == expected


# ────────────────────────────────────────────────────────────────────
# PATCH-RMT34-B: unrouted_feedback_signal → coherence_deficit
# ────────────────────────────────────────────────────────────────────


class TestPatchRMT34B_UnroutedFeedbackSignalMapping:
    def test_class_to_signal_has_explicit_entry(self):
        """The MCT must know about the routing-regression class itself."""
        mct = MetaCognitiveRecursionTrigger()
        # Drive a tiny adaptation call and assert that the
        # ``unrouted_feedback_signal`` class no longer falls into the
        # generic ``uncertainty`` fallback (which logs a debug warning).
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        try:
            mct.adapt_weights_from_evolution({
                "error_classes": {
                    "unrouted_feedback_signal": {"success_rate": 0.0},
                },
            })
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)
        log_text = buf.getvalue()
        assert (
            "unmapped error class 'unrouted_feedback_signal'"
            not in log_text
        ), (
            "unrouted_feedback_signal is still falling through the "
            "generic uncertainty fallback — PATCH-RMT34-B did not bridge "
            "the routing-regression episode back into MCT."
        )

    def test_routes_to_coherence_deficit_not_uncertainty(self):
        """Routing-regressions are wiring/coherence failures.

        With a pure ``unrouted_feedback_signal`` failure (success_rate=0)
        the coherence_deficit weight should *increase* relative to the
        baseline.
        """
        mct = MetaCognitiveRecursionTrigger()
        base = dict(mct._signal_weights)
        mct.adapt_weights_from_evolution({
            "error_classes": {
                "unrouted_feedback_signal": {"success_rate": 0.0},
            },
        })
        post = dict(mct._signal_weights)

        assert "coherence_deficit" in post
        delta_coherence = post["coherence_deficit"] - base["coherence_deficit"]
        assert delta_coherence > 0.0, (
            "PATCH-RMT34-B failed: unrouted_feedback_signal did not "
            "boost coherence_deficit weight."
        )
        # Sanity: an unmapped class auto-registers against uncertainty
        # (proves the contrast — unrouted_feedback_signal must NOT do
        # that).
        mct2 = MetaCognitiveRecursionTrigger()
        base_u = mct2._signal_weights["uncertainty"]
        mct2.adapt_weights_from_evolution({
            "error_classes": {
                "some_genuinely_unmapped_class_xyz_for_rmt34": {
                    "success_rate": 0.0,
                },
            },
        })
        assert mct2._signal_weights["uncertainty"] != base_u

    def test_full_model_forward_no_unmapped_log_lines(self):
        """End-to-end: a model init + (implicit) forward passes must
        produce zero ``unmapped error class 'unrouted_feedback_signal'``
        log lines."""
        torch.manual_seed(0)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        try:
            AEONDeltaV3(AEONConfig())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)

        log_text = buf.getvalue()
        assert (
            "unmapped error class 'unrouted_feedback_signal'"
            not in log_text
        ), (
            "Routing-regression episodes are still being flattened to "
            "the generic uncertainty fallback — PATCH-RMT34-B is not "
            "actually wired into the MCT class mapping."
        )


# ────────────────────────────────────────────────────────────────────
# Integration: both patches active simultaneously
# ────────────────────────────────────────────────────────────────────


class TestRMT34Integration:
    def test_full_init_quiet_on_both_failure_modes(self):
        """A clean init must not log either failure mode."""
        torch.manual_seed(0)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        try:
            AEONDeltaV3(AEONConfig())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)
        log_text = buf.getvalue()
        assert "S-5: FeedbackSignalAttention failed" not in log_text
        assert (
            "unmapped error class 'unrouted_feedback_signal'"
            not in log_text
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
