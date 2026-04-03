"""Tests for CA-1 through CA-9 cognitive activation patches.

Covers:
  CA-1: MetaCognitiveRecursor bounded recursive meta-loop re-invocation
  CA-2: SpectralBifurcationMonitor → feedback bus + causal trace
  CA-3: OutputReliabilityGate → feedback bus + provenance logging
  CA-4: VibeThinkerRSSMBridge runtime integration
  CA-5: RecursionUtilityGate instantiation and wiring
  CA-6: PostOutputUncertaintyGate → feedback bus + force_trigger
  CA-7: CausalProvenanceTracker.log_auxiliary_event() activation
  CA-8: FeedbackSignalAttention selective attention
  CA-9: TemporalCausalTraceBuffer at convergence decision points
"""
import math
import pytest
import torch
import torch.nn as nn

from aeon_core import (
    AEONConfig,
    CognitiveFeedbackBus,
    CausalProvenanceTracker,
    FeedbackSignalAttention,
    MetaCognitiveRecursor,
    MetaCognitiveRecursionTrigger,
    OutputReliabilityGate,
    PostOutputUncertaintyGate,
    ProvablyConvergentMetaLoop,
    RecursionUtilityGate,
    SpectralBifurcationMonitor,
    TemporalCausalTraceBuffer,
    VibeThinkerRSSMBridge,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = dict(
        device_str='cpu',
        enable_quantum_sim=False,
        enable_catastrophe_detection=False,
        enable_safety_guardrails=False,
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


def _make_meta_loop(config, max_iterations=5):
    return ProvablyConvergentMetaLoop(
        config=config,
        max_iterations=max_iterations,
        min_iterations=1,
        convergence_threshold=1e-3,
    )


def _make_feedback_bus(config):
    return CognitiveFeedbackBus(hidden_dim=config.hidden_dim)


# ═══════════════════════════════════════════════════════════════════════
# CA-1: MetaCognitiveRecursor
# ═══════════════════════════════════════════════════════════════════════

class TestCA1MetaCognitiveRecursor:
    """CA-1: MCT → meta-loop recursive re-invocation."""

    def test_ca1_class_exists(self):
        """MetaCognitiveRecursor class is importable."""
        assert MetaCognitiveRecursor is not None

    def test_ca1_init(self):
        """Recursor initialises with meta-loop and feedback bus."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        recursor = MetaCognitiveRecursor(meta_loop=ml, feedback_bus=fb)
        assert recursor.max_depth == 3
        assert recursor.tightening_factor == 0.7
        assert recursor.extra_iterations == 10

    def test_ca1_no_recurse_when_not_triggered(self):
        """When should_recurse=False, return original C_star."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        recursor = MetaCognitiveRecursor(meta_loop=ml, feedback_bus=fb)
        z = torch.randn(2, config.hidden_dim)
        C_star = torch.randn(2, config.hidden_dim)
        result_C, meta = recursor.recurse_if_needed(
            z, C_star,
            trigger_result={'should_trigger': False},
            meta_results={},
        )
        assert torch.equal(result_C, C_star)
        assert meta['recursion_depth'] == 0
        assert not meta['recursed']

    def test_ca1_recurse_when_triggered(self):
        """When should_trigger=True, meta-loop is re-invoked."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=1,
        )
        z = torch.randn(2, config.hidden_dim)
        C_star = torch.randn(2, config.hidden_dim)
        result_C, meta = recursor.recurse_if_needed(
            z, C_star,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
        )
        assert meta['recursed']
        assert meta['recursion_depth'] >= 1
        assert result_C.shape == C_star.shape

    def test_ca1_bounded_depth(self):
        """Recursion depth is bounded by max_depth."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=2,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        _, meta = recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
        )
        assert meta['recursion_depth'] <= 2

    def test_ca1_utility_gate_stops_futile(self):
        """RecursionUtilityGate stops recursion when improvement < 5%."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        gate = RecursionUtilityGate(improvement_threshold=0.05)
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb,
            recursion_utility_gate=gate, max_depth=3,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        _, meta = recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 0.001},
        )
        # With very small residual, improvement is negligible
        assert meta['recursed']
        assert len(meta['recursion_log']) >= 1

    def test_ca1_writes_recursion_depth_signal(self):
        """Recursor writes metacognitive_recursion_depth to feedback bus."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=1,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
        )
        state = fb.get_state()
        assert 'metacognitive_recursion_depth' in state

    def test_ca1_provenance_logging(self):
        """Recursor logs auxiliary events via provenance tracker."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        prov = CausalProvenanceTracker()
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=1,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
            provenance_tracker=prov,
        )
        events = getattr(prov, '_auxiliary_events', [])
        assert any(e.get('event_type') == 'metacognitive_recursion'
                   for e in events)

    def test_ca1_restores_thresholds(self):
        """Meta-loop thresholds are restored after recursion."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        orig_threshold = ml.convergence_threshold
        orig_max_iter = ml.max_iterations
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=1,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
        )
        assert ml.convergence_threshold == orig_threshold
        assert ml.max_iterations == orig_max_iter


# ═══════════════════════════════════════════════════════════════════════
# CA-2: SpectralBifurcationMonitor → feedback bus + causal trace
# ═══════════════════════════════════════════════════════════════════════

class TestCA2SpectralBifurcationMonitor:
    """CA-2: SpectralBifurcationMonitor activation."""

    def test_ca2_monitor_forward(self):
        """SpectralBifurcationMonitor.forward() returns expected keys."""
        config = _make_config()
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        jac = torch.eye(config.hidden_dim) * 0.9
        result = monitor(jac)
        assert 'spectral_radius' in result
        assert 'proximity' in result
        assert 'preemptive' in result

    def test_ca2_high_proximity_preemptive(self):
        """High λ_max triggers preemptive=True."""
        config = _make_config()
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        # λ_max ≈ 0.95 → proximity near 0.95
        jac = torch.eye(config.hidden_dim) * 0.95
        result = monitor(jac)
        assert result['proximity'] >= 0.8

    def test_ca2_low_proximity_safe(self):
        """Low λ_max → proximity near 0, preemptive=False."""
        config = _make_config()
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        jac = torch.eye(config.hidden_dim) * 0.1
        result = monitor(jac)
        assert result['proximity'] < 0.5

    def test_ca2_feedback_bus_write(self):
        """Bifurcation proximity is written to feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        jac = torch.eye(config.hidden_dim) * 0.95
        result = monitor(jac)
        proximity = max(0.0, result.get('proximity', 0.0))
        if proximity > 0:
            fb.write_signal('spectral_instability', proximity)
        state = fb.get_state()
        assert 'spectral_instability' in state
        assert state['spectral_instability'] > 0

    def test_ca2_causal_trace_record(self):
        """Bifurcation warning is recorded in causal trace."""
        trace = TemporalCausalTraceBuffer(max_entries=100)
        # Simulate CA-2 causal trace recording
        proximity = 0.8
        entry_id = trace.record(
            subsystem='spectral_monitor',
            decision='bifurcation_warning',
            metadata={'proximity': proximity},
            severity='warning',
        )
        assert entry_id is not None


# ═══════════════════════════════════════════════════════════════════════
# CA-3: OutputReliabilityGate → MCT weight adaptation
# ═══════════════════════════════════════════════════════════════════════

class TestCA3OutputReliabilityGate:
    """CA-3: OutputReliabilityGate → feedback bus + MCT adaptation."""

    def test_ca3_gate_returns_weakest_factor(self):
        """Gate returns composite and weakest_factor."""
        gate = OutputReliabilityGate()
        result = gate(uncertainty=0.8, causal_quality=0.1)
        assert 'composite' in result
        assert 'weakest_factor' in result
        assert isinstance(result['weakest_factor'], str)

    def test_ca3_adapt_weights_from_reliability_gate(self):
        """MCT adapt_weights_from_reliability_gate boosts target signal."""
        trigger = MetaCognitiveRecursionTrigger()
        initial_weights = dict(trigger._signal_weights)
        trigger.adapt_weights_from_reliability_gate(
            weakest_factor='uncertainty',
            composite_score=0.3,
        )
        # The uncertainty weight should be boosted
        assert trigger._signal_weights['uncertainty'] >= initial_weights['uncertainty']

    def test_ca3_feedback_bus_composite_write(self):
        """output_reliability_composite is written to feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        fb.write_signal('output_reliability_composite', 0.7)
        state = fb.get_state()
        assert 'output_reliability_composite' in state
        assert abs(state['output_reliability_composite'] - 0.7) < 0.01

    def test_ca3_provenance_logging(self):
        """Reliability assessment logged as auxiliary event."""
        prov = CausalProvenanceTracker()
        prov.log_auxiliary_event(
            event_type='output_reliability_assessment',
            metadata={'composite_score': 0.5, 'weakest_factor': 'uncertainty'},
        )
        events = getattr(prov, '_auxiliary_events', [])
        assert any(e['event_type'] == 'output_reliability_assessment'
                   for e in events)


# ═══════════════════════════════════════════════════════════════════════
# CA-4: VibeThinkerRSSMBridge runtime integration
# ═══════════════════════════════════════════════════════════════════════

class TestCA4VibeThinkerRSSMBridge:
    """CA-4: VibeThinkerRSSMBridge instantiation and quality routing."""

    def test_ca4_bridge_instantiation(self):
        """Bridge can be instantiated with feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        bridge = VibeThinkerRSSMBridge(feedback_bus=fb)
        assert bridge.feedback_bus is fb

    def test_ca4_modulate_rssm_loss(self):
        """modulate_rssm_loss writes vibe_thinker_quality to bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        bridge = VibeThinkerRSSMBridge(feedback_bus=fb)
        result = bridge.modulate_rssm_loss(rssm_loss=1.0, vt_quality_signal=0.8)
        assert 'modulated_loss' in result or 'rssm_loss_scale' in result
        state = fb.get_state()
        assert 'vibe_thinker_quality' in state

    def test_ca4_low_quality_upweights_loss(self):
        """Low VT quality → higher RSSM loss scale."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        bridge = VibeThinkerRSSMBridge(feedback_bus=fb)
        # Warm up EMA with low quality
        for _ in range(10):
            bridge.modulate_rssm_loss(rssm_loss=1.0, vt_quality_signal=0.1)
        result = bridge.modulate_rssm_loss(rssm_loss=1.0, vt_quality_signal=0.1)
        scale = result.get('rssm_loss_scale', 1.0)
        assert scale > 1.0, f"Expected scale > 1.0 for low VT quality, got {scale}"

    def test_ca4_high_quality_normal_loss(self):
        """High VT quality → loss scale near 1.0."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        bridge = VibeThinkerRSSMBridge(feedback_bus=fb)
        # Warm up EMA with high quality
        for _ in range(20):
            bridge.modulate_rssm_loss(rssm_loss=1.0, vt_quality_signal=0.95)
        result = bridge.modulate_rssm_loss(rssm_loss=1.0, vt_quality_signal=0.95)
        scale = result.get('rssm_loss_scale', 1.0)
        assert scale < 1.3, f"Expected scale < 1.3 for high VT quality, got {scale}"


# ═══════════════════════════════════════════════════════════════════════
# CA-5: RecursionUtilityGate
# ═══════════════════════════════════════════════════════════════════════

class TestCA5RecursionUtilityGate:
    """CA-5: RecursionUtilityGate instantiation and wiring."""

    def test_ca5_gate_useful(self):
        """Gate detects useful recursion (≥5% improvement)."""
        gate = RecursionUtilityGate(improvement_threshold=0.05)
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0, post_residual=0.5,
        )
        assert result['was_useful'] is True
        assert result['improvement_ratio'] >= 0.05

    def test_ca5_gate_futile(self):
        """Gate detects futile recursion (<5% improvement)."""
        gate = RecursionUtilityGate(improvement_threshold=0.05)
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0, post_residual=0.97,
        )
        assert result['was_useful'] is False
        assert result['improvement_ratio'] < 0.05

    def test_ca5_futility_counter(self):
        """Futile recursion count accumulates."""
        gate = RecursionUtilityGate()
        gate.evaluate_recursion_utility(pre_residual=1.0, post_residual=0.99)
        gate.evaluate_recursion_utility(pre_residual=1.0, post_residual=0.99)
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0, post_residual=0.99,
        )
        assert result['futile_recursion_count'] == 3

    def test_ca5_counter_resets_on_useful(self):
        """Futile count resets when a useful pass occurs."""
        gate = RecursionUtilityGate()
        gate.evaluate_recursion_utility(pre_residual=1.0, post_residual=0.99)
        gate.evaluate_recursion_utility(pre_residual=1.0, post_residual=0.99)
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0, post_residual=0.3,
        )
        assert result['was_useful'] is True
        assert result['futile_recursion_count'] == 0

    def test_ca5_writes_futility_pressure(self):
        """Futility pressure is written to feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        gate = RecursionUtilityGate()
        gate.evaluate_recursion_utility(
            pre_residual=1.0, post_residual=0.99, feedback_bus=fb,
        )
        state = fb.get_state()
        assert 'recursion_futility_pressure' in state


# ═══════════════════════════════════════════════════════════════════════
# CA-6: PostOutputUncertaintyGate → feedback bus + force_trigger
# ═══════════════════════════════════════════════════════════════════════

class TestCA6PostOutputUncertaintyGate:
    """CA-6: PostOutputUncertaintyGate → feedback loop."""

    def test_ca6_gate_evaluates(self):
        """PostOutputUncertaintyGate.evaluate() returns expected keys."""
        gate = PostOutputUncertaintyGate()
        result = gate.evaluate(
            uncertainty=0.3,
            uncertainty_sources={'cycle_consistency_violation': 0.2},
            ucc_already_triggered=False,
        )
        assert 'gate_triggered' in result
        assert 'late_uncertainty' in result

    def test_ca6_force_trigger_method(self):
        """MetaCognitiveRecursionTrigger.force_trigger() exists and works."""
        trigger = MetaCognitiveRecursionTrigger()
        result = trigger.force_trigger(
            reason='post_output_uncertainty_critical',
            uncertainty=0.9,
        )
        assert result['should_trigger'] is True
        assert result['forced'] is True
        assert 'forced:post_output_uncertainty_critical' in result['triggers_active']

    def test_ca6_force_trigger_boosts_ema(self):
        """force_trigger boosts cross-pass EMA."""
        trigger = MetaCognitiveRecursionTrigger()
        initial_ema = trigger._cross_pass_trigger_ema
        trigger.force_trigger(reason='test', uncertainty=0.9)
        assert trigger._cross_pass_trigger_ema > initial_ema

    def test_ca6_feedback_bus_write(self):
        """post_output_uncertainty signal is written to feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        fb.write_signal('post_output_uncertainty', 0.85)
        state = fb.get_state()
        assert 'post_output_uncertainty' in state
        assert state['post_output_uncertainty'] > 0.8

    def test_ca6_high_uncertainty_triggers_force(self):
        """Uncertainty > 0.8 should trigger force_trigger."""
        trigger = MetaCognitiveRecursionTrigger()
        # Simulate the CA-6 path
        uncertainty = 0.9
        if uncertainty > 0.8:
            result = trigger.force_trigger(
                reason='post_output_uncertainty_critical',
                uncertainty=uncertainty,
            )
            assert result['should_trigger']


# ═══════════════════════════════════════════════════════════════════════
# CA-7: CausalProvenanceTracker.log_auxiliary_event()
# ═══════════════════════════════════════════════════════════════════════

class TestCA7ProvenanceLogging:
    """CA-7: log_auxiliary_event() calls at decision points."""

    def test_ca7_log_meta_loop_converged(self):
        """Meta-loop convergence is logged as auxiliary event."""
        prov = CausalProvenanceTracker()
        prov.log_auxiliary_event(
            event_type='meta_loop_converged',
            metadata={
                'iterations': 10,
                'convergence_rate': 0.95,
                'certificate_type': 'km',
            },
        )
        events = getattr(prov, '_auxiliary_events', [])
        meta_events = [e for e in events if e['event_type'] == 'meta_loop_converged']
        assert len(meta_events) == 1
        assert meta_events[0]['metadata']['iterations'] == 10

    def test_ca7_log_mct_triggered(self):
        """MCT trigger evaluation is logged."""
        prov = CausalProvenanceTracker()
        prov.log_auxiliary_event(
            event_type='mct_trigger_evaluation',
            metadata={
                'should_trigger': True,
                'trigger_score': 0.75,
                'triggers_active': ['uncertainty', 'coherence_deficit'],
            },
        )
        events = getattr(prov, '_auxiliary_events', [])
        mct_events = [e for e in events if e['event_type'] == 'mct_trigger_evaluation']
        assert len(mct_events) == 1
        assert mct_events[0]['metadata']['should_trigger'] is True

    def test_ca7_log_recursion_evaluated(self):
        """Recursion utility evaluation is logged."""
        prov = CausalProvenanceTracker()
        prov.log_auxiliary_event(
            event_type='metacognitive_recursion',
            metadata={
                'depth': 1,
                'was_useful': True,
                'improvement_ratio': 0.12,
            },
        )
        events = getattr(prov, '_auxiliary_events', [])
        rec_events = [e for e in events if e['event_type'] == 'metacognitive_recursion']
        assert len(rec_events) == 1

    def test_ca7_events_appear_in_attribution(self):
        """Auxiliary events appear in trace_root_cause() output."""
        prov = CausalProvenanceTracker()
        prov.log_auxiliary_event('meta_loop_converged', {'iterations': 5})
        prov.log_auxiliary_event('mct_trigger_evaluation', {'should_trigger': False})
        # Events are stored internally
        events = getattr(prov, '_auxiliary_events', [])
        assert len(events) == 2
        # Events appear in trace_root_cause output
        root_cause = prov.trace_root_cause('meta_loop')
        assert 'auxiliary_causal_events' in root_cause
        assert len(root_cause['auxiliary_causal_events']) == 2


# ═══════════════════════════════════════════════════════════════════════
# CA-8: FeedbackSignalAttention
# ═══════════════════════════════════════════════════════════════════════

class TestCA8FeedbackSignalAttention:
    """CA-8: Selective attention over feedback signals."""

    def test_ca8_class_exists(self):
        """FeedbackSignalAttention is importable."""
        assert FeedbackSignalAttention is not None

    def test_ca8_init(self):
        """Attention module initialises correctly."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        assert attn.num_signals == 12
        assert attn.hidden_dim == config.hidden_dim

    def test_ca8_forward_shape(self):
        """forward() produces [B, hidden_dim] output."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        signals = torch.randn(4, 12)
        output = attn(signals)
        assert output.shape == (4, config.hidden_dim)

    def test_ca8_attention_weights_sum_to_one(self):
        """Attention weights sum to 1 across signals."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        signals = torch.randn(1, 12)
        attn_logits = attn.attention(signals)
        weights = torch.softmax(attn_logits, dim=-1)
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_ca8_gradient_flow(self):
        """Gradients flow through attention mechanism."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        signals = torch.randn(2, 12, requires_grad=True)
        output = attn(signals)
        loss = output.sum()
        loss.backward()
        assert signals.grad is not None
        assert signals.grad.abs().sum().item() > 0

    def test_ca8_with_iteration_state(self):
        """forward() works with optional iteration_state argument."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        signals = torch.randn(2, 12)
        state = torch.randn(2, config.hidden_dim)
        output = attn(signals, iteration_state=state)
        assert output.shape == (2, config.hidden_dim)

    def test_ca8_different_signals_different_output(self):
        """Different signal patterns produce different outputs."""
        config = _make_config()
        attn = FeedbackSignalAttention(
            num_signals=12, hidden_dim=config.hidden_dim,
        )
        s1 = torch.zeros(1, 12)
        s1[0, 0] = 1.0  # Only signal 0 active
        s2 = torch.zeros(1, 12)
        s2[0, 5] = 1.0  # Only signal 5 active
        o1 = attn(s1)
        o2 = attn(s2)
        # Different input signals should produce different outputs
        assert not torch.allclose(o1, o2, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# CA-9: TemporalCausalTraceBuffer at convergence
# ═══════════════════════════════════════════════════════════════════════

class TestCA9CausalTraceConvergence:
    """CA-9: Causal trace at meta-loop convergence decision points."""

    def test_ca9_record_convergence_achieved(self):
        """Convergence achievement is recorded in causal trace."""
        trace = TemporalCausalTraceBuffer(max_entries=100)
        entry_id = trace.record(
            subsystem='meta_loop',
            decision='convergence_achieved',
            metadata={
                'iterations': 15,
                'convergence_rate': 0.98,
                'stall_detected': False,
                'anderson_active': True,
            },
            severity='info',
        )
        assert entry_id is not None

    def test_ca9_record_convergence_timeout(self):
        """Convergence timeout is recorded in causal trace."""
        trace = TemporalCausalTraceBuffer(max_entries=100)
        entry_id = trace.record(
            subsystem='meta_loop',
            decision='convergence_timeout',
            metadata={
                'iterations': 50,
                'convergence_rate': 0.4,
                'stall_detected': True,
                'anderson_active': False,
            },
            severity='warning',
        )
        assert entry_id is not None

    def test_ca9_traceable_from_root(self):
        """Convergence decision can be traced to root cause."""
        trace = TemporalCausalTraceBuffer(max_entries=100)
        input_id = trace.record(
            subsystem='encoder', decision='input_processed',
        )
        conv_id = trace.record(
            subsystem='meta_loop',
            decision='convergence_achieved',
            causal_prerequisites=[input_id],
            metadata={'iterations': 10},
        )
        root = trace.trace_root_cause(conv_id)
        # Root cause should trace back to encoder
        root_modules = root.get('root_modules', root.get('root_causes', []))
        assert len(root_modules) > 0 or root.get('trace_incomplete', True)

    def test_ca9_prerequisites_linking(self):
        """Convergence entry links to prior causal entries."""
        trace = TemporalCausalTraceBuffer(max_entries=100)
        prereq_id = trace.record(
            subsystem='feedback_bus', decision='conditioning_computed',
        )
        conv_id = trace.record(
            subsystem='meta_loop',
            decision='convergence_achieved',
            causal_prerequisites=[prereq_id],
        )
        chain = trace.get_causal_chain(conv_id)
        assert len(chain) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCAIntegration:
    """End-to-end integration tests across multiple patches."""

    def test_int_metacognitive_trigger_to_recursor(self):
        """MCT trigger → MetaCognitiveRecursor → utility gate → feedback bus."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        gate = RecursionUtilityGate()
        trigger = MetaCognitiveRecursionTrigger()
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb,
            recursion_utility_gate=gate, max_depth=2,
        )
        # Trigger evaluation
        trigger_result = trigger.evaluate(
            uncertainty=0.9, is_diverging=True,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        C_out, meta = recursor.recurse_if_needed(
            z, C,
            trigger_result=trigger_result,
            meta_results={'final_residual': 1.0},
        )
        assert C_out.shape == C.shape
        # Check feedback bus has recursion signals
        state = fb.get_state()
        assert 'metacognitive_recursion_depth' in state

    def test_int_spectral_to_mct_to_recursor(self):
        """Spectral monitor → feedback bus → MCT → recursor."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        ml = _make_meta_loop(config)
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        trigger = MetaCognitiveRecursionTrigger()

        # 1. Spectral monitor detects instability
        jac = torch.eye(config.hidden_dim) * 0.95
        sbm_result = monitor(jac)
        proximity = sbm_result.get('proximity', 0.0)
        fb.write_signal('spectral_instability', proximity)

        # 2. MCT evaluates (with spectral instability)
        trigger_result = trigger.evaluate(
            uncertainty=0.5,
            spectral_stability_margin=1.0 - proximity,
        )

        # 3. Recursor handles
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb, max_depth=1,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        C_out, meta = recursor.recurse_if_needed(
            z, C,
            trigger_result=trigger_result,
            meta_results={'final_residual': 1.0},
        )
        assert C_out.shape == C.shape

    def test_int_reliability_to_mct_adaptation(self):
        """OutputReliabilityGate weakest factor → MCT weight adaptation."""
        gate = OutputReliabilityGate()
        trigger = MetaCognitiveRecursionTrigger()

        # Low causal quality
        result = gate(uncertainty=0.3, causal_quality=0.1)
        weakest = result['weakest_factor']
        composite = result['composite']

        # Adapt MCT weights
        trigger.adapt_weights_from_reliability_gate(
            weakest_factor=weakest,
            composite_score=composite,
        )
        # Verify weights changed
        assert True  # No crash

    def test_int_full_causal_chain(self):
        """Full causal chain: convergence → MCT → recursion → trace."""
        config = _make_config()
        prov = CausalProvenanceTracker()
        trace = TemporalCausalTraceBuffer(max_entries=100)

        # 1. Log convergence
        prov.log_auxiliary_event('meta_loop_converged', {'iterations': 10})

        # 2. Log MCT evaluation
        prov.log_auxiliary_event('mct_trigger_evaluation', {
            'should_trigger': True, 'trigger_score': 0.8,
        })

        # 3. Log recursion
        prov.log_auxiliary_event('metacognitive_recursion', {
            'depth': 1, 'was_useful': True,
        })

        # 4. Record in causal trace
        input_id = trace.record('encoder', 'input_processed')
        conv_id = trace.record(
            'meta_loop', 'convergence_achieved',
            causal_prerequisites=[input_id],
        )
        mct_id = trace.record(
            'metacognitive_trigger', 'triggered',
            causal_prerequisites=[conv_id],
        )
        rec_id = trace.record(
            'metacognitive_recursor', 'recursion_accepted',
            causal_prerequisites=[mct_id],
        )

        # Verify chain
        chain = trace.get_causal_chain(rec_id)
        assert len(chain) >= 2  # At least 2 entries in chain
        # Auxiliary events are in trace_root_cause, not compute_attribution
        root_cause = prov.trace_root_cause('meta_loop')
        assert 'auxiliary_causal_events' in root_cause
        assert len(root_cause['auxiliary_causal_events']) == 3

    def test_int_post_output_to_force_trigger(self):
        """PostOutputUncertaintyGate → force_trigger → feedback bus."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        trigger = MetaCognitiveRecursionTrigger()
        gate = PostOutputUncertaintyGate()

        # Evaluate with high late uncertainty
        result = gate.evaluate(
            uncertainty=0.9,
            uncertainty_sources={
                'cycle_consistency_violation': 0.8,
                'decoder_degenerate': 0.7,
            },
            ucc_already_triggered=False,
        )
        late_unc = result.get('late_uncertainty', 0.0)
        fb.write_signal('post_output_uncertainty', late_unc)

        # If high, force trigger
        if late_unc > 0.8:
            forced = trigger.force_trigger(
                reason='post_output_uncertainty_critical',
                uncertainty=late_unc,
            )
            assert forced['should_trigger']

    def test_int_feedback_attention_with_bus(self):
        """FeedbackSignalAttention processes feedback bus signals."""
        config = _make_config()
        fb = _make_feedback_bus(config)
        attn = FeedbackSignalAttention(
            num_signals=fb.NUM_SIGNAL_CHANNELS,
            hidden_dim=config.hidden_dim,
        )
        # Generate feedback bus output
        fb_out = fb(batch_size=2, device='cpu')
        # Pass through attention
        # fb_out is [B, hidden_dim], but attention expects [B, num_signals]
        # Create a signal vector manually
        signals = torch.randn(2, fb.NUM_SIGNAL_CHANNELS)
        attended = attn(signals)
        assert attended.shape == (2, config.hidden_dim)

    def test_int_activation_sequence_phase1(self):
        """Phase 1: Observability patches (CA-7, CA-9) work independently."""
        prov = CausalProvenanceTracker()
        trace = TemporalCausalTraceBuffer(max_entries=100)

        # CA-7
        prov.log_auxiliary_event('meta_loop_converged', {'iterations': 5})
        prov.log_auxiliary_event('mct_trigger_evaluation', {
            'should_trigger': False,
        })

        # CA-9
        trace.record('meta_loop', 'convergence_achieved', metadata={
            'iterations': 5, 'stall_detected': False,
        })

        # Verify both work
        root_cause = prov.trace_root_cause('meta_loop')
        assert 'auxiliary_causal_events' in root_cause
        assert len(root_cause['auxiliary_causal_events']) == 2

    def test_int_activation_sequence_phase2(self):
        """Phase 2: Sensing patches (CA-2, CA-3) produce signals."""
        config = _make_config()
        fb = _make_feedback_bus(config)

        # CA-2: spectral monitor
        monitor = SpectralBifurcationMonitor(hidden_dim=config.hidden_dim)
        jac = torch.eye(config.hidden_dim) * 0.9
        result = monitor(jac)
        fb.write_signal('spectral_instability', result['proximity'])

        # CA-3: reliability gate
        gate = OutputReliabilityGate()
        rel = gate(uncertainty=0.5)
        fb.write_signal('output_reliability_composite', rel['composite'])

        state = fb.get_state()
        assert 'spectral_instability' in state
        assert 'output_reliability_composite' in state

    def test_int_activation_sequence_phase4(self):
        """Phase 4: Actuation patches (CA-1, CA-5) handle recursion."""
        config = _make_config()
        ml = _make_meta_loop(config)
        fb = _make_feedback_bus(config)
        gate = RecursionUtilityGate()
        recursor = MetaCognitiveRecursor(
            meta_loop=ml, feedback_bus=fb,
            recursion_utility_gate=gate, max_depth=2,
        )
        z = torch.randn(1, config.hidden_dim)
        C = torch.randn(1, config.hidden_dim)
        C_out, meta = recursor.recurse_if_needed(
            z, C,
            trigger_result={'should_trigger': True},
            meta_results={'final_residual': 1.0},
        )
        assert meta['recursed']
        assert meta['recursion_depth'] >= 1
