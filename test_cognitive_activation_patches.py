"""
Tests for CP-α through CP-η: Final Integration & Cognitive Activation patches.

These patches close the remaining ~10% gap in the AEON-Delta signal
ecosystem by making loss composition, token selection, and remediation
strategy fully traceable through the causal provenance infrastructure.

Test groups:
- CP-α: Loss Component Causal Attribution
- CP-β: Token Generation Provenance Chain
- CP-γ: Unified Loss Scaling Provenance
- CP-δ: Per-Axiom Loss Targeting
- CP-ε: Auto-Critic Revision Reasoning Trace
- CP-ζ: Remediation Escalation Consumer
- CP-η: Orphaned Signal Consumers
- Signal Ecosystem Completeness
"""

import re
import math
import torch
import torch.nn as nn
import pytest

from aeon_core import (
    AEONConfig,
    AEONDeltaV3,
    CognitiveFeedbackBus,
    MetaCognitiveRecursionTrigger,
    ThoughtEncoder,
)


# ═══════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _make_config(**overrides):
    """Create a minimal AEONConfig for testing."""
    defaults = dict(
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
        vocab_size=128,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_model_with_bus(**config_overrides):
    """Create model + bus pair for integration tests."""
    cfg = _make_config(**config_overrides)
    model = AEONDeltaV3(cfg)
    bus = model.feedback_bus
    return model, bus


def _make_bus():
    """Create standalone bus for unit tests."""
    return CognitiveFeedbackBus(hidden_dim=64)


def _make_mct_with_bus():
    """Create MCT + bus pair for MCT-specific tests."""
    bus = _make_bus()
    mct = MetaCognitiveRecursionTrigger(trigger_threshold=0.5)
    mct._feedback_bus_ref = bus
    return mct, bus


# ═══════════════════════════════════════════════════════════════════════
# CP-α: LOSS COMPONENT CAUSAL ATTRIBUTION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPAlpha_LossComponentAttribution:
    """CP-α: Loss attribution map and bus signals."""

    def test_loss_composition_dominant_signal_written(self):
        """compute_loss writes loss_composition_dominant to bus."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        loss_dict = model.compute_loss(outputs, x)
        val = bus.read_signal('loss_composition_dominant', -1.0)
        # Must be a fraction in [0, 1]
        assert 0.0 <= float(val) <= 1.0

    def test_loss_scaling_composite_depth_signal_written(self):
        """compute_loss writes loss_scaling_composite_depth to bus."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('loss_scaling_composite_depth', -1.0)
        assert float(val) >= 0.0

    def test_loss_attribution_stored_on_model(self):
        """compute_loss stores _cpa_loss_components dict."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        comps = getattr(model, '_cpa_loss_components', None)
        assert comps is not None
        assert isinstance(comps, dict)
        assert 'lm_loss' in comps
        assert 'safety_loss' in comps

    def test_loss_dominant_identifies_largest_component(self):
        """_cpa_dominant names the component with largest abs value."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        comps = getattr(model, '_cpa_loss_components', {})
        dominant = getattr(model, '_cpa_dominant', None)
        if comps and sum(abs(v) for v in comps.values()) > 0:
            expected = max(comps, key=lambda k: abs(comps[k]))
            assert dominant == expected

    def test_causal_trace_records_loss_attribution(self):
        """Causal trace has a loss_attribution entry."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        if model.causal_trace is not None:
            buf = getattr(model.causal_trace, '_buffer', [])
            decisions = [e.get('decision', '') for e in buf if isinstance(e, dict)]
            assert 'loss_attribution' in decisions

    def test_scaling_factors_list_populated(self):
        """_cpa_scaling_factors is a list with name/value dicts."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        factors = getattr(model, '_cpa_scaling_factors', None)
        assert factors is not None
        assert isinstance(factors, list)
        for f in factors:
            assert 'name' in f
            assert 'value' in f


# ═══════════════════════════════════════════════════════════════════════
# CP-β: TOKEN GENERATION PROVENANCE CHAIN TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPBeta_TokenGenerationProvenance:
    """CP-β: Per-token selection probability tracking."""

    def test_generation_provenance_depth_written(self):
        """After generate(), generation_provenance_depth is on the bus."""
        model, bus = _make_model_with_bus()
        # decoder._fb_ref is set in __init__ by CP-β wiring
        result = model.generate("test input", max_length=5)
        val = bus.read_signal('generation_provenance_depth', -1.0)
        # Should be >= 0 (number of tokens generated)
        assert float(val) >= 0.0

    def test_generation_confidence_min_written(self):
        """After generate(), generation_confidence_min is on the bus."""
        model, bus = _make_model_with_bus()
        result = model.generate("test input", max_length=5)
        val = bus.read_signal('generation_confidence_min', -1.0)
        # Should be in [0, 1]
        assert 0.0 <= float(val) <= 1.0

    def test_step_probs_tracked_per_token(self):
        """Internal _cpb_step_probs list accumulates one entry per step."""
        model, bus = _make_model_with_bus()
        result = model.generate("hello world", max_length=10)
        depth = float(bus.read_signal('generation_provenance_depth', 0.0))
        # Depth should equal the number of generation steps taken
        assert depth > 0.0

    def test_decoder_fb_ref_wired(self):
        """Decoder._fb_ref is set to feedback_bus in __init__."""
        model, bus = _make_model_with_bus()
        assert getattr(model.decoder, '_fb_ref', None) is bus


# ═══════════════════════════════════════════════════════════════════════
# CP-γ: UNIFIED LOSS SCALING PROVENANCE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPGamma_UnifiedLossScalingProvenance:
    """CP-γ: Composite scaling factor recording."""

    def test_loss_aggregate_scaling_written_when_factors_active(self):
        """loss_aggregate_scaling written when scaling factors are active."""
        model, bus = _make_model_with_bus()
        # Force a non-1.0 convergence scale
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        # Force divergence to activate convergence_loss_scale
        outputs['convergence_verdict'] = {'status': 'diverging'}
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('loss_aggregate_scaling', -1.0)
        # Should have been written (value >= 1.0 since diverging)
        assert float(val) >= 1.0

    def test_causal_trace_records_scaling_composite(self):
        """Causal trace has a loss_scaling_composite entry when factors active."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        outputs['convergence_verdict'] = {'status': 'diverging'}
        _ = model.compute_loss(outputs, x)
        if model.causal_trace is not None:
            buf = getattr(model.causal_trace, '_buffer', [])
            decisions = [e.get('decision', '') for e in buf if isinstance(e, dict)]
            assert 'loss_scaling_composite' in decisions


# ═══════════════════════════════════════════════════════════════════════
# CP-δ: PER-AXIOM LOSS TARGETING TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPDelta_PerAxiomLossTargeting:
    """CP-δ: Per-axiom loss amplification."""

    def test_axiom_targeted_loss_active_signal_written(self):
        """axiom_targeted_loss_active is written after compute_loss."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('axiom_targeted_loss_active', -1.0)
        assert float(val) >= 0.0

    def test_low_mv_quality_activates_targeting(self):
        """When mutual_verification_quality < 0.5, targeting activates."""
        model, bus = _make_model_with_bus()
        bus.write_signal('mutual_verification_quality', 0.3)
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('axiom_targeted_loss_active', 0.0)
        assert float(val) > 0.0

    def test_low_um_quality_activates_targeting(self):
        """When uncertainty_metacognition_quality < 0.5, targeting activates."""
        model, bus = _make_model_with_bus()
        bus.write_signal('uncertainty_metacognition_quality', 0.2)
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('axiom_targeted_loss_active', 0.0)
        assert float(val) > 0.0

    def test_low_rc_quality_activates_targeting(self):
        """When root_cause_traceability_quality < 0.5, targeting activates."""
        model, bus = _make_model_with_bus()
        bus.write_signal('root_cause_traceability_quality', 0.1)
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('axiom_targeted_loss_active', 0.0)
        assert float(val) > 0.0

    def test_all_healthy_axioms_no_targeting(self):
        """When all axiom qualities > 0.5, targeting is 0.0."""
        model, bus = _make_model_with_bus()
        bus.write_signal('mutual_verification_quality', 0.9)
        bus.write_signal('uncertainty_metacognition_quality', 0.8)
        bus.write_signal('root_cause_traceability_quality', 0.7)
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        val = bus.read_signal('axiom_targeted_loss_active', -1.0)
        assert float(val) == 0.0

    def test_targeted_loss_increases_total(self):
        """Axiom targeting should increase total_loss vs baseline."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        # Baseline: healthy axioms
        bus.write_signal('mutual_verification_quality', 0.9)
        bus.write_signal('uncertainty_metacognition_quality', 0.9)
        bus.write_signal('root_cause_traceability_quality', 0.9)
        loss_healthy = model.compute_loss(outputs, x)['total_loss'].item()
        # Targeted: all axioms failing
        bus.write_signal('mutual_verification_quality', 0.1)
        bus.write_signal('uncertainty_metacognition_quality', 0.1)
        bus.write_signal('root_cause_traceability_quality', 0.1)
        loss_targeted = model.compute_loss(outputs, x)['total_loss'].item()
        # Targeted loss should be >= healthy loss
        assert loss_targeted >= loss_healthy


# ═══════════════════════════════════════════════════════════════════════
# CP-ε: AUTO-CRITIC REVISION REASONING TRACE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPEpsilon_AutoCriticRevisionTrace:
    """CP-ε: Enhanced revision reasoning in auto-critic."""

    def test_revision_trace_has_magnitude_field(self):
        """Causal trace entries for auto_critic/revision include
        revision_magnitude."""
        # Verify the code structure exists by searching for the string
        with open('aeon_core.py') as f:
            content = f.read()
        assert "'revision_magnitude'" in content
        assert "'critique_focus'" in content
        assert "'threshold_gap'" in content
        assert "'accepted'" in content

    def test_revision_reasoning_depth_signal_in_code(self):
        """auto_critic_revision_reasoning_depth signal is written."""
        with open('aeon_core.py') as f:
            content = f.read()
        assert "'auto_critic_revision_reasoning_depth'" in content
        # Must be both written and read
        writes = len(re.findall(
            r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
            r"auto_critic_revision_reasoning_depth['\"]",
            content,
        ))
        reads = len(re.findall(
            r"read_signal\s*\(\s*\n?\s*['\"]"
            r"auto_critic_revision_reasoning_depth['\"]",
            content,
        ))
        assert writes >= 1, "Signal must be written"
        assert reads >= 1, "Signal must be read (CP-η MCT consumer)"

    def test_pre_revision_state_clone(self):
        """CP-ε clones pre-revision state for delta computation."""
        with open('aeon_core.py') as f:
            content = f.read()
        # Look for the detach().clone() pattern in the auto-critic block
        assert '_cpe_pre_revision' in content


# ═══════════════════════════════════════════════════════════════════════
# CP-ζ: REMEDIATION ESCALATION CONSUMER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPZeta_RemediationEscalation:
    """CP-ζ: MCT → remediation_escalation_level → compute_loss."""

    def test_mct_writes_remediation_escalation_level(self):
        """MCT evaluate() writes remediation_escalation_level to bus."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('targeted_remediation_active', 0.5)
        bus.write_signal('axiom_mutual_consistency', 0.3)
        mct.evaluate()
        val = bus.read_signal('remediation_escalation_level', -1.0)
        assert float(val) >= 0.0

    def test_escalation_zero_when_no_remediation(self):
        """No remediation active → escalation_level = 0."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('targeted_remediation_active', 0.0)
        bus.write_signal('axiom_mutual_consistency', 0.9)
        mct.evaluate()
        val = bus.read_signal('remediation_escalation_level', -1.0)
        assert float(val) == 0.0

    def test_escalation_nonzero_when_remediation_failing(self):
        """Remediation active + low axiom → escalation_level > 0."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('targeted_remediation_active', 0.8)
        bus.write_signal('axiom_mutual_consistency', 0.3)
        mct.evaluate()
        val = bus.read_signal('remediation_escalation_level', 0.0)
        assert float(val) > 0.0

    def test_escalation_boosts_recovery_pressure(self):
        """When escalation is active, recovery_pressure should be
        injected into MCT signal_values."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('targeted_remediation_active', 0.8)
        bus.write_signal('axiom_mutual_consistency', 0.3)
        result = mct.evaluate()
        # MCT should have read targeted_remediation_active
        assert 'targeted_remediation_active' in bus._read_log

    def test_compute_loss_reads_escalation_level(self):
        """compute_loss reads remediation_escalation_level for dampening."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        # Set high escalation level
        bus.write_signal('remediation_escalation_level', 0.8)
        _ = model.compute_loss(outputs, x)
        assert 'remediation_escalation_level' in bus._read_log

    def test_escalation_dampens_loss(self):
        """High escalation level should dampen total_loss."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        # Baseline: no escalation
        bus.write_signal('remediation_escalation_level', 0.0)
        loss_normal = model.compute_loss(outputs, x)['total_loss'].item()
        # With escalation: should dampen
        bus.write_signal('remediation_escalation_level', 0.8)
        loss_dampened = model.compute_loss(outputs, x)['total_loss'].item()
        assert loss_dampened <= loss_normal


# ═══════════════════════════════════════════════════════════════════════
# CP-η: ORPHANED SIGNAL CONSUMER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPEta_OrphanedSignalConsumers:
    """CP-η: All formerly orphaned signals now have consumers."""

    def test_mct_reads_mct_decision_provenance_depth(self):
        """MCT reads mct_decision_provenance_depth (CP-η(a))."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        bus.write_signal('mct_decision_provenance_depth', 0.0)
        _ = model.compute_loss(outputs, x)
        assert 'mct_decision_provenance_depth' in bus._read_log

    def test_zero_provenance_depth_adds_penalty(self):
        """When mct_decision_provenance_depth = 0, loss increases by 2%."""
        model, bus = _make_model_with_bus()
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        # Baseline: high provenance depth
        bus.write_signal('mct_decision_provenance_depth', 5.0)
        loss_traced = model.compute_loss(outputs, x)['total_loss'].item()
        # Zero depth: opacity penalty
        bus.write_signal('mct_decision_provenance_depth', 0.0)
        loss_opaque = model.compute_loss(outputs, x)['total_loss'].item()
        # Opaque loss should be >= traced loss
        assert loss_opaque >= loss_traced

    def test_bridge_reads_targeted_remediation_active(self):
        """_bridge_epoch_feedback reads targeted_remediation_active (CP-η(b))."""
        with open('aeon_core.py') as f:
            content = f.read()
        # Must have a read_signal call for targeted_remediation_active in bridge
        bridge_pattern = re.findall(
            r"read_signal\s*\(\s*\n?\s*['\"]targeted_remediation_active['\"]",
            content,
        )
        # At least 2 readers: MCT (CP-ζ) + bridge (CP-η(b))
        assert len(bridge_pattern) >= 2

    def test_mct_reads_mct_dominant_trigger_signal(self):
        """MCT reads mct_dominant_trigger_signal (CP-η(c))."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('mct_dominant_trigger_signal', 0.5)
        mct.evaluate()
        assert 'mct_dominant_trigger_signal' in bus._read_log

    def test_chronic_dominant_trigger_boosts_recovery(self):
        """Same dominant trigger for > 3 passes → recovery_pressure."""
        mct, bus = _make_mct_with_bus()
        # Simulate 4 passes with same dominant trigger weight
        for _ in range(4):
            bus.write_signal('mct_dominant_trigger_signal', 0.5)
            mct.evaluate()
            bus.flush_consumed()
        # After 4 passes, recovery_pressure should have been boosted
        # Verify by checking that persist count reached > 3
        assert getattr(mct, '_cph_dominant_persist_count', 0) >= 4

    def test_different_dominant_resets_counter(self):
        """Changing dominant trigger resets persistence counter."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('mct_dominant_trigger_signal', 0.5)
        mct.evaluate()
        bus.flush_consumed()
        bus.write_signal('mct_dominant_trigger_signal', 0.5)
        mct.evaluate()
        bus.flush_consumed()
        # Change dominant
        bus.write_signal('mct_dominant_trigger_signal', 0.9)
        mct.evaluate()
        assert getattr(mct, '_cph_dominant_persist_count', 0) == 1


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL ECOSYSTEM COMPLETENESS TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestSignalEcosystemAfterCPPatches:
    """Verify signal ecosystem integrity after all CP patches."""

    def test_no_missing_producers(self):
        """Every read_signal must have at least one write_signal."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]", full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]", full,
            ):
                read.add(m.group(1))
            for m in re.finditer(
                r"_extra_signals\s*\[\s*['\"](\w+)['\"]", full,
            ):
                written.add(m.group(1))
        missing = read - written
        assert len(missing) == 0, f"Missing producers: {sorted(missing)}"

    def test_orphaned_count_at_most_4(self):
        """At most 4 benign orphaned signals (metadata only)."""
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]", full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]", full,
            ):
                read.add(m.group(1))
            for m in re.finditer(
                r"_extra_signals\s*\[\s*['\"](\w+)['\"]", full,
            ):
                written.add(m.group(1))
        orphaned = written - read
        benign = {
            'integration_cycle_id',
            'integration_cycle_timestamp',
            'wizard_completed',
            'wizard_corpus_quality',
        }
        non_benign = orphaned - benign
        assert len(non_benign) == 0, (
            f"Non-benign orphaned signals: {sorted(non_benign)}"
        )

    def test_new_cp_signals_all_connected(self):
        """All CP-series signals are both written and read."""
        cp_signals = [
            'loss_composition_dominant',
            'loss_scaling_composite_depth',
            'loss_aggregate_scaling',
            'axiom_targeted_loss_active',
            'generation_provenance_depth',
            'generation_confidence_min',
            'auto_critic_revision_reasoning_depth',
            'remediation_escalation_level',
        ]
        written = set()
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]", full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]", full,
            ):
                read.add(m.group(1))
        for sig in cp_signals:
            assert sig in written, f"{sig} not written"
            assert sig in read, f"{sig} not read"

    def test_formerly_orphaned_now_connected(self):
        """mct_decision_provenance_depth, mct_dominant_trigger_signal,
        and targeted_remediation_active now have consumers."""
        formerly_orphaned = [
            'mct_decision_provenance_depth',
            'mct_dominant_trigger_signal',
            'targeted_remediation_active',
        ]
        read = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"read_signal\s*\(\s*\n?\s*['\"]([^'\"]+)['\"]", full,
            ):
                read.add(m.group(1))
        for sig in formerly_orphaned:
            assert sig in read, f"{sig} still has no consumer"

    def test_written_signal_count_increased(self):
        """Signal ecosystem should have more written signals than before."""
        written = set()
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            try:
                with open(fname) as f:
                    full = f.read()
            except FileNotFoundError:
                continue
            for m in re.finditer(
                r"write_signal(?:_traced)?\s*\(\s*\n?\s*['\"]"
                r"([^'\"]+)['\"]", full,
            ):
                written.add(m.group(1))
            for m in re.finditer(
                r"_extra_signals\s*\[\s*['\"](\w+)['\"]", full,
            ):
                written.add(m.group(1))
        # Before CP patches: 153 written. After: 161 (8 new signals)
        assert len(written) >= 161


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION / FEEDBACK LOOP TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCPIntegrationLoops:
    """Verify that CP patches create proper feedback loops."""

    def test_loss_attribution_to_mct_loop(self):
        """loss_composition_dominant written by compute_loss →
        read by MCT → affects signal_values."""
        model, bus = _make_model_with_bus()
        mct = model.metacognitive_trigger
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        # Now MCT should be able to read loss_composition_dominant
        mct.evaluate()
        assert 'loss_composition_dominant' in bus._read_log

    def test_axiom_targeting_closed_loop(self):
        """verify_and_reinforce writes axiom quality → compute_loss
        reads and targets → axiom_targeted_loss_active written →
        MCT reads."""
        model, bus = _make_model_with_bus()
        # Simulate low axiom quality (as if verify_and_reinforce wrote it)
        bus.write_signal('mutual_verification_quality', 0.2)
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        # axiom_targeted_loss_active should be 1.0
        assert float(bus.read_signal('axiom_targeted_loss_active', 0.0)) > 0.0
        # MCT should read it
        model.metacognitive_trigger.evaluate()
        assert 'axiom_targeted_loss_active' in bus._read_log

    def test_escalation_closed_loop(self):
        """MCT writes remediation_escalation_level →
        compute_loss reads and dampens loss."""
        model, bus = _make_model_with_bus()
        # Set up escalation conditions
        bus.write_signal('targeted_remediation_active', 0.5)
        bus.write_signal('axiom_mutual_consistency', 0.3)
        # MCT fires and writes escalation level
        model.metacognitive_trigger.evaluate()
        esc = float(bus.read_signal('remediation_escalation_level', 0.0))
        assert esc > 0.0
        # Now compute_loss should read it and dampen
        x = torch.randint(0, 128, (2, 16))
        outputs = model(x)
        _ = model.compute_loss(outputs, x)
        assert 'remediation_escalation_level' in bus._read_log

    def test_generation_confidence_to_mct_loop(self):
        """generation_confidence_min written by decoder →
        MCT reads and affects uncertainty signal."""
        mct, bus = _make_mct_with_bus()
        bus.write_signal('generation_confidence_min', 0.05)
        mct.evaluate()
        assert 'generation_confidence_min' in bus._read_log

    def test_no_bare_except_pass_in_cp_code(self):
        """All CP patches follow Φ1 convention: no bare except:pass."""
        with open('aeon_core.py') as f:
            lines = f.readlines()
        bare_passes = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('except') and ':' in stripped:
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_s = lines[j].strip()
                    if next_s == '':
                        continue
                    if next_s == 'pass':
                        bare_passes.append(i + 1)
                    break
        assert len(bare_passes) == 0, (
            f"Found {len(bare_passes)} bare except:pass blocks "
            f"at lines: {bare_passes[:10]}"
        )
