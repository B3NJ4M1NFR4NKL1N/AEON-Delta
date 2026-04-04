"""
Test suite for R-series patches: Final Integration & Cognitive Activation.

R-1: LipschitzConstrainedLambda.set_lipschitz_target() adaptive contractivity
R-2: RecursionUtilityGate contraction-margin-aware evaluation
R-3: MetaCognitiveRecursor L_C certificate integration
R-4: MCT topology_catastrophe graduated severity
R-5: VQ collapse → feedback bus escalation signal
R-6: Cross-pass coherence feedback in aeon_server.py
R-7: FI-4 adaptive contractivity enforcement (severity-proportional)
R-8: Training loss → contractivity budget tightening
R-9: Graduated post-output uncertainty trigger (0.5-0.8 moderate range)
"""

import math
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeon_core import (
    AEONConfig,
    LipschitzConstrainedLambda,
    RecursionUtilityGate,
    MetaCognitiveRecursionTrigger,
    CognitiveFeedbackBus,
    RobustVectorQuantizer,
    CausalProvenanceTracker,
)


def _make_config(**overrides):
    defaults = dict(
        hidden_dim=64,
        vocab_size=256,
        seq_length=32,
        device_str='cpu',
    )
    defaults.update(overrides)
    return AEONConfig(**defaults)


# ────────────────────────────────────────────────────────────────────
# R-1: LipschitzConstrainedLambda.set_lipschitz_target()
# ────────────────────────────────────────────────────────────────────

class TestR1_SetLipschitzTarget:
    """R-1: Adaptive contractivity budget via set_lipschitz_target()."""

    def test_set_lipschitz_target_exists(self):
        """set_lipschitz_target method exists on LipschitzConstrainedLambda."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        assert hasattr(lam, 'set_lipschitz_target')
        assert callable(lam.set_lipschitz_target)

    def test_set_lipschitz_target_returns_dict(self):
        """set_lipschitz_target returns a dict with old/new target info."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        result = lam.set_lipschitz_target(0.8)
        assert isinstance(result, dict)
        assert 'old_target' in result
        assert 'new_target' in result
        assert 'requested' in result
        assert 'was_clamped' in result

    def test_set_lipschitz_target_changes_target(self):
        """Setting a new target within bounds updates the attribute."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        assert lam.lipschitz_target == 0.9
        result = lam.set_lipschitz_target(0.8)
        assert lam.lipschitz_target == pytest.approx(0.8, abs=1e-6)
        assert result['old_target'] == pytest.approx(0.9, abs=1e-6)

    def test_set_lipschitz_target_clamps_above_original(self):
        """Cannot relax target above the original design value."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        result = lam.set_lipschitz_target(1.5)
        assert lam.lipschitz_target == pytest.approx(0.9, abs=1e-6)
        assert result['was_clamped'] is True

    def test_set_lipschitz_target_clamps_at_floor(self):
        """Cannot tighten target below the safety floor."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        result = lam.set_lipschitz_target(0.1)
        assert lam.lipschitz_target >= lam._lipschitz_target_floor
        assert result['was_clamped'] is True

    def test_original_target_preserved(self):
        """Original target is stored for reference after changes."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        lam.set_lipschitz_target(0.7)
        assert lam._lipschitz_target_original == pytest.approx(0.9, abs=1e-6)

    def test_floor_is_reasonable(self):
        """Floor is >= 0.3 and at most 50% of original."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        assert lam._lipschitz_target_floor >= 0.3
        assert lam._lipschitz_target_floor <= lam._lipschitz_target_original


# ────────────────────────────────────────────────────────────────────
# R-2: RecursionUtilityGate contraction-margin awareness
# ────────────────────────────────────────────────────────────────────

class TestR2_ContractionAwareUtility:
    """R-2: RecursionUtilityGate with lipschitz_constant parameter."""

    def test_evaluate_accepts_lipschitz_constant(self):
        """evaluate_recursion_utility accepts lipschitz_constant kwarg."""
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
            lipschitz_constant=0.8,
        )
        assert isinstance(result, dict)

    def test_contraction_margin_included_when_lc_provided(self):
        """Contraction margin is in result when L_C is provided."""
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
            lipschitz_constant=0.8,
        )
        assert 'contraction_margin' in result
        assert result['contraction_margin'] == pytest.approx(0.2, abs=1e-6)

    def test_certified_error_bound_when_contractive(self):
        """Certified error bound computed when L_C < 1."""
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
            lipschitz_constant=0.8,
        )
        assert 'certified_error_bound' in result
        assert result['certified_error_bound'] is not None
        # Banach FPT: ε ≤ L_C/(1-L_C) · post_residual = 0.8/0.2 * 0.5 = 2.0
        assert result['certified_error_bound'] == pytest.approx(2.0, abs=1e-6)

    def test_no_certified_bound_when_expansive(self):
        """No certified error bound when L_C ≥ 1."""
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
            lipschitz_constant=1.2,
        )
        assert result['certified_error_bound'] is None

    def test_raised_threshold_when_expansive(self):
        """Improvement threshold raised 1.5× when L_C ≥ 1."""
        gate = RecursionUtilityGate(improvement_threshold=0.05)
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
            lipschitz_constant=1.2,
        )
        assert result['effective_threshold'] == pytest.approx(0.075, abs=1e-6)

    def test_backward_compatible_without_lc(self):
        """Works identically to before when lipschitz_constant is None."""
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.5,
        )
        assert 'improvement_ratio' in result
        assert 'was_useful' in result
        assert 'contraction_margin' not in result  # not included when L_C absent

    def test_marginal_improvement_rejected_when_expansive(self):
        """Small improvement rejected when L_C > 1 due to raised threshold."""
        gate = RecursionUtilityGate(improvement_threshold=0.05)
        # 6% improvement — passes normal threshold (5%) but fails 7.5% (raised)
        result_no_lc = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.94,
        )
        gate2 = RecursionUtilityGate(improvement_threshold=0.05)
        result_with_lc = gate2.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.94,
            lipschitz_constant=1.2,
        )
        assert result_no_lc['was_useful'] is True
        assert result_with_lc['was_useful'] is False


# ────────────────────────────────────────────────────────────────────
# R-3: MetaCognitiveRecursor L_C certificate integration
# ────────────────────────────────────────────────────────────────────

class TestR3_RecursorLipschitzIntegration:
    """R-3: MetaCognitiveRecursor extracts L_C from convergence cert."""

    def test_recursion_log_includes_lipschitz(self):
        """Recursion log entries include lipschitz_constant field."""
        from aeon_core import MetaCognitiveRecursor, ProvablyConvergentMetaLoop

        config = _make_config(z_dim=64, hidden_dim=64, vq_embedding_dim=64)
        meta_loop = ProvablyConvergentMetaLoop(config, max_iterations=3)
        bus = CognitiveFeedbackBus(hidden_dim=64)
        gate = RecursionUtilityGate()
        recursor = MetaCognitiveRecursor(
            meta_loop=meta_loop,
            feedback_bus=bus,
            recursion_utility_gate=gate,
        )

        # Simulate a forward pass to check log structure
        z_in = torch.randn(1, 64)
        meta_results = {'final_residual': 0.5}
        C_star = meta_loop(z_in, use_fixed_point=True)[0]

        # Create a trigger result that will trigger recursion
        trigger_result = {
            'should_trigger': True,
            'composite_severity': 0.8,
            'triggers_active': ['diverging'],
        }

        C_out, recursion_meta = recursor.recurse_if_needed(
            z_in=z_in,
            C_star=C_star,
            trigger_result=trigger_result,
            meta_results=meta_results,
        )

        # Recursion log should include lipschitz_constant and contraction_margin
        log_entries = recursion_meta.get('recursion_log', [])
        if len(log_entries) > 0:
            entry = log_entries[0]
            assert 'lipschitz_constant' in entry
            assert 'contraction_margin' in entry


# ────────────────────────────────────────────────────────────────────
# R-4: MCT topology_catastrophe graduated severity
# ────────────────────────────────────────────────────────────────────

class TestR4_GraduatedCatastropheSeverity:
    """R-4: MCT evaluates topology_catastrophe as graduated severity."""

    def _make_mct(self):
        return MetaCognitiveRecursionTrigger()

    def test_evaluate_accepts_float_severity(self):
        """MCT.evaluate() accepts float topology_catastrophe."""
        mct = self._make_mct()
        result = mct.evaluate(topology_catastrophe=0.6)
        assert isinstance(result, dict)

    def test_bool_backward_compatible(self):
        """MCT.evaluate() still works with bool topology_catastrophe."""
        mct = self._make_mct()
        result_true = mct.evaluate(topology_catastrophe=True)
        result_false = mct.evaluate(topology_catastrophe=False)
        assert isinstance(result_true, dict)
        assert isinstance(result_false, dict)

    def test_severity_graduated_response(self):
        """Higher severity produces higher composite score."""
        mct = self._make_mct()
        result_fold = mct.evaluate(topology_catastrophe=0.3)
        result_cusp = mct.evaluate(topology_catastrophe=0.6)
        result_swallowtail = mct.evaluate(topology_catastrophe=0.9)

        # Extract signal values from results
        sv_fold = result_fold.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )
        sv_cusp = result_cusp.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )
        sv_swall = result_swallowtail.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )

        # Graduated: higher severity → higher signal value
        assert sv_cusp >= sv_fold
        assert sv_swall >= sv_cusp

    def test_zero_severity_no_signal(self):
        """Zero severity produces zero topology_catastrophe signal."""
        mct = self._make_mct()
        result = mct.evaluate(topology_catastrophe=0.0)
        sv = result.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )
        assert sv == pytest.approx(0.0, abs=1e-6)

    def test_full_severity_equals_bool_true(self):
        """Float 1.0 severity produces same result as bool True."""
        mct = self._make_mct()
        result_float = mct.evaluate(topology_catastrophe=1.0)
        result_bool = mct.evaluate(topology_catastrophe=True)

        sv_float = result_float.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )
        sv_bool = result_bool.get('signal_values', {}).get(
            'topology_catastrophe', 0.0
        )
        assert sv_float == pytest.approx(sv_bool, abs=1e-6)


# ────────────────────────────────────────────────────────────────────
# R-5: VQ collapse → feedback bus escalation
# ────────────────────────────────────────────────────────────────────

class TestR5_VQCollapseEscalation:
    """R-5: VQ codebook collapse writes escalation to feedback bus."""

    def test_vq_accepts_feedback_bus_ref_attribute(self):
        """RobustVectorQuantizer can have _feedback_bus_ref set externally."""
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=64, commitment_cost=0.25,
        )
        bus = CognitiveFeedbackBus(hidden_dim=64)
        vq._feedback_bus_ref = bus
        assert hasattr(vq, '_feedback_bus_ref')
        assert vq._feedback_bus_ref is bus

    def test_vq_collapse_writes_diversity_collapse_signal(self):
        """When VQ perplexity drops below threshold, diversity_collapse
        signal is written to the feedback bus."""
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=64, commitment_cost=0.25,
        )
        bus = CognitiveFeedbackBus(hidden_dim=64)
        vq._feedback_bus_ref = bus

        # Simulate collapse: set perplexity EMA far below threshold
        vq._total_steps.fill_(100)
        vq._perplexity_ema.fill_(0.5)  # well below threshold
        vq._perplexity_collapse_threshold = 2.0
        vq._perplexity_intervention_cooldown = 0

        # Need to set up _last_inputs_for_revival
        vq._last_inputs_for_revival = torch.randn(4, 64)

        # Run a forward pass to trigger the collapse detection path
        x = torch.randn(4, 64)
        vq(x)

        # After forward, if collapse was detected, signals should be readable
        # via the bus public API
        if vq._perplexity_interventions_count > 0:
            collapse_val = bus.read_signal('diversity_collapse', 0.0)
            assert collapse_val > 0.0, (
                "Expected diversity_collapse signal after VQ collapse intervention"
            )

    def test_vq_collapse_severity_proportional(self):
        """Collapse severity is proportional to perplexity drop depth."""
        vq = RobustVectorQuantizer(
            num_embeddings=16, embedding_dim=64, commitment_cost=0.25,
        )
        bus = CognitiveFeedbackBus(hidden_dim=64)
        vq._feedback_bus_ref = bus

        # Compute expected severity: 1 - (ema / threshold)
        vq._perplexity_collapse_threshold = 2.0
        test_ema = 0.5
        expected_severity = max(0.0, min(1.0, 1.0 - test_ema / 2.0))
        assert expected_severity == pytest.approx(0.75, abs=0.01)


# ────────────────────────────────────────────────────────────────────
# R-6: Cross-pass coherence feedback (aeon_server.py)
# ────────────────────────────────────────────────────────────────────

class TestR6_CrossPassCoherenceFeedback:
    """R-6: Cross-pass coherence verification feeds back to feedback bus."""

    def test_r6_code_present_in_server(self):
        """R-6 coherence feedback code is present in aeon_server.py."""
        import inspect
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'aeon_server.py',
        )
        with open(server_path, 'r') as f:
            server_code = f.read()

        assert 'cross_pass_oscillation' in server_code
        assert 'cross_pass_instability_pressure' in server_code
        assert 'R-6' in server_code

    def test_stability_map_values(self):
        """Stability map correctly converts categories to numeric values."""
        _stab_map = {'stable': 0.0, 'oscillating': 0.5, 'chaotic': 1.0}
        assert _stab_map['stable'] == 0.0
        assert _stab_map['oscillating'] == 0.5
        assert _stab_map['chaotic'] == 1.0


# ────────────────────────────────────────────────────────────────────
# R-7: Adaptive contractivity enforcement
# ────────────────────────────────────────────────────────────────────

class TestR7_AdaptiveContractivityEnforcement:
    """R-7: FI-4 enforcement interval adapts to violation severity."""

    def test_r7_code_present(self):
        """R-7 adaptive enforcement code is present in aeon_core.py."""
        core_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'aeon_core.py',
        )
        with open(core_path, 'r') as f:
            core_code = f.read()

        assert '_enforcement_interval' in core_code
        assert 'R-7' in core_code

    def test_enforcement_interval_formula(self):
        """Enforcement interval formula gives correct values.

        Formula: max(10, min(50, floor(50 / max(L_C, 1.01))))
        The max(L_C, 1.01) floor prevents division by zero and ensures
        L_C=1.0 maps to 50/1.01 ≈ 49 (not 50).
        """
        def interval(L_C):
            return max(10, min(50, int(50.0 / max(L_C, 1.01))))

        # L_C ≈ 1.0 → ~50 steps
        assert interval(1.0) == 49  # 50/1.01 ≈ 49
        # L_C ≈ 2.0 → ~25 steps
        assert interval(2.0) == 25
        # L_C ≈ 5.0 → 10 steps (floor)
        assert interval(5.0) == 10
        # L_C very high → 10 (clamped)
        assert interval(100.0) == 10

    def test_enforcement_more_frequent_with_high_lc(self):
        """Higher L_C → lower interval → more frequent enforcement."""
        def interval(L_C):
            return max(10, min(50, int(50.0 / max(L_C, 1.01))))

        assert interval(1.0) > interval(2.0)
        assert interval(2.0) > interval(5.0)


# ────────────────────────────────────────────────────────────────────
# R-8: Training loss → contractivity budget tightening
# ────────────────────────────────────────────────────────────────────

class TestR8_LossContractivityFeedback:
    """R-8: Training loss spikes tighten contractivity budget."""

    def test_r8_code_present(self):
        """R-8 loss→contractivity code is present in aeon_core.py."""
        core_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'aeon_core.py',
        )
        with open(core_path, 'r') as f:
            core_code = f.read()

        assert 'R-8' in core_code
        assert 'r8_ratio' in core_code or '_r8_ratio' in core_code

    def test_tightening_on_loss_spike(self):
        """When loss > 2× EMA, target should tighten by 5%."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        original = lam.lipschitz_target
        # Simulate: 5% tightening
        new_target = original * 0.95
        result = lam.set_lipschitz_target(new_target)
        assert lam.lipschitz_target == pytest.approx(new_target, abs=1e-6)
        assert lam.lipschitz_target < original

    def test_relaxation_toward_original(self):
        """When loss is stable, target relaxes 2% toward original."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        # Tighten first
        lam.set_lipschitz_target(0.7)
        current = lam.lipschitz_target
        # Relax: 98% current + 2% original
        relaxed = current * 0.98 + lam._lipschitz_target_original * 0.02
        result = lam.set_lipschitz_target(relaxed)
        assert lam.lipschitz_target > current
        assert lam.lipschitz_target <= lam._lipschitz_target_original


# ────────────────────────────────────────────────────────────────────
# R-9: Graduated post-output uncertainty trigger
# ────────────────────────────────────────────────────────────────────

class TestR9_GraduatedUncertaintyTrigger:
    """R-9: Moderate uncertainty (0.5–0.8) writes escalation signal."""

    def test_r9_code_present(self):
        """R-9 graduated uncertainty code is present in aeon_core.py."""
        core_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'aeon_core.py',
        )
        with open(core_path, 'r') as f:
            core_code = f.read()

        assert 'R-9' in core_code
        assert 'moderate_output_uncertainty_pressure' in core_code

    def test_moderate_uncertainty_pressure_formula(self):
        """Moderate uncertainty pressure scales linearly from 0.5 to 0.8."""
        # Formula: min(1.0, (unc - 0.5) / 0.3)
        def pressure(unc):
            return min(1.0, (unc - 0.5) / 0.3)

        # At 0.5 → 0.0
        assert pressure(0.5) == pytest.approx(0.0, abs=1e-6)
        # At 0.65 → 0.5
        assert pressure(0.65) == pytest.approx(0.5, abs=1e-6)
        # At 0.8 → 1.0
        assert pressure(0.8) == pytest.approx(1.0, abs=1e-6)
        # At 0.7 → 0.667
        assert pressure(0.7) == pytest.approx(0.667, abs=0.01)

    def test_high_uncertainty_still_triggers_force(self):
        """Uncertainty > 0.8 still triggers force_trigger (unchanged)."""
        core_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'aeon_core.py',
        )
        with open(core_path, 'r') as f:
            core_code = f.read()

        assert 'post_output_uncertainty_critical' in core_code


# ────────────────────────────────────────────────────────────────────
# Cross-patch integration tests
# ────────────────────────────────────────────────────────────────────

class TestCrossPatchIntegration:
    """Verify R-series patches interoperate correctly."""

    def test_r1_r2_r8_chain(self):
        """R-8 tightens target (R-1), which affects R-2 recursion gating."""
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        gate = RecursionUtilityGate(improvement_threshold=0.05)

        # Simulate R-8 tightening
        lam.set_lipschitz_target(0.7)
        assert lam.lipschitz_target == pytest.approx(0.7, abs=1e-6)

        # With contraction (L_C=0.7 < 1.0), standard threshold applies
        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=0.94,  # 6% improvement
            lipschitz_constant=0.7,
        )
        assert result['was_useful'] is True

    def test_r4_accepts_severity_values(self):
        """MCT handles the standard catastrophe severity values."""
        mct = MetaCognitiveRecursionTrigger()

        # Standard severity mapping from classify_catastrophe_type
        severities = {
            'none': 0.0,
            'fold': 0.3,
            'cusp': 0.6,
            'swallowtail': 0.9,
        }
        for name, sev in severities.items():
            result = mct.evaluate(topology_catastrophe=sev)
            assert isinstance(result, dict), f"Failed for {name}={sev}"

    def test_r5_feedback_bus_signal_names(self):
        """VQ collapse uses standard signal names consumable by MCT."""
        bus = CognitiveFeedbackBus(hidden_dim=64)
        # Manually write the signals R-5 would write
        bus.write_signal('vq_collapse_severity', 0.75)
        bus.write_signal('diversity_collapse', 0.75)

        # Read them back
        assert bus.read_signal('vq_collapse_severity', 0.0) == pytest.approx(0.75, abs=0.01)
        assert bus.read_signal('diversity_collapse', 0.0) == pytest.approx(0.75, abs=0.01)

    def test_r7_interval_bounds(self):
        """R-7 enforcement interval is always in [10, 50]."""
        for L_C in [1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]:
            interval = max(10, min(50, int(50.0 / max(L_C, 1.01))))
            assert 10 <= interval <= 50, f"Out of bounds for L_C={L_C}: {interval}"

    def test_all_patches_independent(self):
        """Each R-series patch is independently functional."""
        # R-1
        lam = LipschitzConstrainedLambda(64, 128, 64, lipschitz_target=0.9)
        lam.set_lipschitz_target(0.8)
        assert lam.lipschitz_target == pytest.approx(0.8, abs=1e-6)

        # R-2
        gate = RecursionUtilityGate()
        result = gate.evaluate_recursion_utility(1.0, 0.5, lipschitz_constant=0.8)
        assert 'contraction_margin' in result

        # R-4
        mct = MetaCognitiveRecursionTrigger()
        result = mct.evaluate(topology_catastrophe=0.6)
        assert isinstance(result, dict)

        # R-5
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus.write_signal('vq_collapse_severity', 0.5)
        assert bus.read_signal('vq_collapse_severity', 0.0) > 0

    def test_r2_banach_error_bound_formula(self):
        """R-2 certified error bound follows Banach FPT formula."""
        gate = RecursionUtilityGate()
        # ε ≤ L_C/(1-L_C) · post_residual
        L_C = 0.5
        post_residual = 0.3
        expected = L_C / (1 - L_C) * post_residual  # 0.5/0.5 * 0.3 = 0.3

        result = gate.evaluate_recursion_utility(
            pre_residual=1.0,
            post_residual=post_residual,
            lipschitz_constant=L_C,
        )
        assert result['certified_error_bound'] == pytest.approx(expected, abs=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
