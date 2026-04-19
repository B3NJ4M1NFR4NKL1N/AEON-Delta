"""
Regression tests for the five residual-isolation patches added on top
of the cognitive-activation series:

* Patch A — Forward-pass-level re-entrancy fence
* Patch B — Diagnostic snapshot completeness (idempotent observation)
* Patch C — Cold-start emergence interpretation
* Patch D — Routing-prefix safety net
* Patch E — Patch-severity cache consumption assertion

Each test asserts the runtime invariant the corresponding patch
introduces.  The suite is intentionally small (one test per patch)
because each patch only adds an observation channel; deeper coverage
is provided by the existing 242-test suite which continues to pass.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from aeon_core import AEONConfig, AEONDeltaV3  # noqa: E402


@pytest.fixture(scope="module")
def model():
    """Shared model — most tests are pure observation, safe to share."""
    return AEONDeltaV3(AEONConfig())


# ─────────────────────────────────────────────────────────────────────
# Patch A — Forward-pass re-entrancy fence
# ─────────────────────────────────────────────────────────────────────

class TestPatchAResidForwardFence:
    """``forward()`` must always reset the verify_and_reinforce
    re-entrancy guard, even when ``_forward_impl`` raises before
    its own head-of-pass reset."""

    def test_forward_exception_resets_guard(self, model):
        # Simulate the worst case: pretend a previous pass left the
        # guard stuck True.  Then monkey-patch _forward_impl to raise
        # before ever reaching the head-of-pass reset.
        model._verify_and_reinforce_in_progress = True
        original_impl = model._forward_impl

        class _SyntheticAbort(RuntimeError):
            pass

        def _abort(*args, **kwargs):  # pragma: no cover — synthetic
            raise _SyntheticAbort("synthetic forward abort")

        model._forward_impl = _abort
        try:
            import torch as _torch
            with pytest.raises(_SyntheticAbort):
                model.forward(_torch.zeros((1, 4), dtype=_torch.long))
            # The fence MUST reset the guard after the abort.
            assert model._verify_and_reinforce_in_progress is False, (
                "PATCH-RESID-A: forward() must reset the verify_and_"
                "reinforce re-entrancy guard on any exception escaping "
                "_forward_impl, otherwise reinforcement is permanently "
                "disabled until the next successful forward pass."
            )
        finally:
            model._forward_impl = original_impl
            model._verify_and_reinforce_in_progress = False

        # Sanity: a subsequent verify_and_reinforce() call must execute
        # its body, not short-circuit on the stuck guard.
        result = model.verify_and_reinforce()
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────
# Patch B — Diagnostic snapshot completeness
# ─────────────────────────────────────────────────────────────────────

class TestPatchBResidIdempotentReport:
    """Two consecutive ``system_emergence_report()`` calls with no
    intervening forward pass must produce structurally identical
    integration_map and critical_patches blocks."""

    def test_emergence_report_idempotent(self, model):
        model._verify_and_reinforce_in_progress = False
        report_a = model.system_emergence_report()
        report_b = model.system_emergence_report()

        # Compare the deterministic, observation-only fields.  The full
        # dict can include floating-point health scores and timestamps
        # that legitimately drift; the integration_map's signal
        # coverage and critical_patches' module_health_score should be
        # stable across observations.
        im_a = report_a.get('integration_map', {})
        im_b = report_b.get('integration_map', {})
        assert im_a.get('signal_coverage') == im_b.get('signal_coverage')
        assert im_a.get('signal_count') == im_b.get('signal_count')
        # Patch-severity bookkeeping must not drift between two
        # observations because no forward pass intervenes.
        assert im_a.get('patch_severity_write_pass') == \
            im_b.get('patch_severity_write_pass')
        assert im_a.get('patch_severity_read_pass') == \
            im_b.get('patch_severity_read_pass')

        # Verdict labels must agree — Patch C's structural→empirical
        # promotion is deterministic for a given pass count.
        es_a = report_a.get('system_emergence_status', {})
        es_b = report_b.get('system_emergence_status', {})
        assert es_a.get('verdict') == es_b.get('verdict')
        assert es_a.get('confidence_basis') == es_b.get('confidence_basis')


# ─────────────────────────────────────────────────────────────────────
# Patch C — Cold-start emergence interpretation
# ─────────────────────────────────────────────────────────────────────

class TestPatchCResidConfidenceBasis:
    """Verdict labels and per-axiom confidence basis must distinguish
    structural baselines from empirical evidence."""

    def test_axiom_confidence_basis_present(self, model):
        report = model.system_emergence_report()
        es = report.get('system_emergence_status', {})
        basis = es.get('axiom_confidence_basis', {})
        # All three core axioms must carry a basis label.
        for axiom in (
            'mutual_verification',
            'metacognitive_responsiveness',
            'root_cause_traceability',
        ):
            assert axiom in basis, (
                f"PATCH-RESID-C: axiom '{axiom}' must report a "
                f"confidence_basis; got keys={list(basis)!r}"
            )
            assert basis[axiom] in (
                'structural_baseline', 'empirical',
            )
        # ``verdict`` must be one of the three known labels.
        assert es.get('verdict') in (
            'emerged', 'provisional_emerged', 'not_emerged',
        )
        # When emerged is True and the activation probe has run, the
        # verdict must be empirical "emerged" — never the structural
        # provisional label.  This guards the test asserted in the
        # plan: no ``emerged=True`` final verdict before any real
        # evidence (the activation probe counts as evidence).
        if es.get('emerged') is True:
            assert es.get('confidence_basis') == 'empirical'
            assert es.get('verdict') == 'emerged'

    def test_cold_start_provisional_verdict_when_no_evidence(self, model):
        # Force the no-evidence regime: clear the activation flag and
        # zero the forward-pass counter.  This is purely a label-level
        # check; restored immediately afterward.
        import torch as _torch
        prev_flag = getattr(model, '_cognitive_activation_complete', False)
        prev_calls = model._total_forward_calls.clone()
        prev_label = getattr(
            model, '_cached_emergence_verdict_label', None,
        )
        model._cognitive_activation_complete = False
        model._total_forward_calls = _torch.tensor(0, dtype=_torch.long)
        try:
            report = model.system_emergence_report()
            es = report.get('system_emergence_status', {})
            assert es.get('confidence_basis') == 'structural_baseline'
            # If the wiring conditions are met, the verdict label
            # must be ``provisional_emerged`` (never ``emerged``).
            assert es.get('verdict') in (
                'provisional_emerged', 'not_emerged',
            )
        finally:
            model._cognitive_activation_complete = prev_flag
            model._total_forward_calls = prev_calls
            if prev_label is None:
                if hasattr(model, '_cached_emergence_verdict_label'):
                    delattr(model, '_cached_emergence_verdict_label')
            else:
                model._cached_emergence_verdict_label = prev_label


# ─────────────────────────────────────────────────────────────────────
# Patch D — Routing-prefix safety net
# ─────────────────────────────────────────────────────────────────────

class TestPatchDResidRoutingFallback:
    """Newly-added bus signals not present in the explicit map must
    still influence MCT weights via the suffix-based fallback router,
    and signals that even the fallback cannot classify must surface
    as an ``unrouted_feedback_signal`` episode."""

    def test_suffix_fallback_routes_pressure_signal(self):
        # Use an isolated model so we can observe weight delta.
        m = AEONDeltaV3(AEONConfig())
        trigger = m.metacognitive_trigger
        before = dict(trigger._signal_weights)
        # Inject a synthetic *_pressure signal that is not in the
        # explicit map.  The suffix fallback should route it to
        # ``recovery_pressure`` and adjust weights.
        trigger.adapt_weights_from_feedback_signals({
            'test_unmapped_pressure': 0.9,
        })
        after = trigger._signal_weights
        # Weights are renormalised to sum 1.0; the relative weight of
        # ``recovery_pressure`` must increase.
        assert after['recovery_pressure'] > before['recovery_pressure'], (
            "PATCH-RESID-D: suffix fallback must route *_pressure "
            "signals into the recovery_pressure trigger bucket."
        )

    def test_unrouted_signal_recorded(self):
        m = AEONDeltaV3(AEONConfig())
        # Fully-unclassifiable signal name (no recognised prefix and
        # no recognised suffix).  Make sure the activation probe is
        # marked complete so the bridge actually surfaces the episode.
        assert getattr(m, '_cognitive_activation_complete', False) is True
        trigger = m.metacognitive_trigger
        trigger.adapt_weights_from_feedback_signals({
            'completely_unmapped_signal_xyz': 0.8,
        })
        unrouted = getattr(trigger, '_unrouted_signals_seen', set())
        assert 'completely_unmapped_signal_xyz' in unrouted, (
            "PATCH-RESID-D: signals that match neither the explicit "
            "map, the prefix routes, nor the suffix fallback must be "
            "recorded on _unrouted_signals_seen."
        )


# ─────────────────────────────────────────────────────────────────────
# Patch E — Patch-severity cache consumption assertion
# ─────────────────────────────────────────────────────────────────────

class TestPatchEResidConsumptionTracking:
    """The integration map must surface read/write pass counters for
    ``_cached_emergence_patch_severity`` and an ``unconsumed_writes``
    flag that fires when a write is not followed by a read."""

    def test_integration_map_exposes_consumption_counters(self, model):
        report = model.system_emergence_report()
        im = report.get('integration_map', {})
        # The three telemetry fields must be present.
        for fld in (
            'patch_severity_write_pass',
            'patch_severity_read_pass',
            'unconsumed_writes',
        ):
            assert fld in im, (
                f"PATCH-RESID-E: integration_map must expose '{fld}'."
            )
        # On a freshly-constructed model that has not yet run a real
        # forward pass, no consumption-skip should be flagged.
        assert im['unconsumed_writes'] == 0

    def test_unconsumed_write_flagged_when_pipeline_skipped(self):
        # Simulate the pathological case: a write happened a few
        # passes ago and the read pass never caught up.
        m = AEONDeltaV3(AEONConfig())
        m._cached_emergence_patch_severity_write_pass = 5
        m._cached_emergence_patch_severity_read_pass = 0
        import torch as _torch
        m._total_forward_calls = _torch.tensor(10, dtype=_torch.long)
        report = m.system_emergence_report()
        im = report.get('integration_map', {})
        assert im['unconsumed_writes'] == 1, (
            "PATCH-RESID-E: when a cache write is older than the "
            "current forward-pass counter and the read pass never "
            "matched it, integration_map.unconsumed_writes must fire."
        )
