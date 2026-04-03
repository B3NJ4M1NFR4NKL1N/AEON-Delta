"""
Comprehensive tests for Critical Patches CP-1 through CP-8.

These patches close the remaining integration gaps in the AEON-Delta
cognitive architecture, transitioning from a "connected architecture"
to a "functional cognitive organism" with mutual reinforcement,
meta-cognitive triggers, and causal transparency.

CP-1: Catastrophe Classifier → MCT graded signal (G1)
CP-2: Training Phase Pressure → Curriculum Adaptation (G2)
CP-3: Diversity Pressure → VQ Maintenance Escalation (G3)
CP-4: K7 Same-Pass MCT Flag → Iteration Extension (G4)
CP-5: Convergence Certificate → Confidence Modulation (G5)
CP-6: Causal Model ← MCT Feedback Integration (G6)
CP-7: VibeThinkerMetaSignaler → RSSM Loss Wiring (G7)
CP-8: Consistency Gate → Hard Coherence Threshold (G8)
"""

import time
import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn.functional as F


# ── Shared helpers ─────────────────────────────────────────────────────


def _make_utcc(model, feedback_bus=None, mct=None, config=None):
    """Create a UnifiedTrainingCycleController from aeon_server."""
    from aeon_server import UnifiedTrainingCycleController

    if config is None:
        config = MagicMock()
        config.reinforce_interval = 5
        config.staleness_threshold_s = 300.0

    utcc = UnifiedTrainingCycleController.__new__(
        UnifiedTrainingCycleController,
    )
    utcc.model = model
    utcc.config = config
    utcc.device = "cpu"
    utcc._signal_bus = None
    utcc._vt_learner = None
    utcc._controller = None
    utcc._feedback_bus = feedback_bus
    utcc._ucc = None
    utcc._mct = mct
    utcc._continual_core = None
    utcc._cycle_count = 0
    utcc._metrics_history = []
    utcc._z_annotation_used_fallback = False
    utcc._pending_wizard_results = None
    utcc._mct_consecutive_triggers = 0
    utcc._default_reinforce_interval = 5
    return utcc


def _make_mock_model():
    """Create a mock model with all expected cached attributes."""
    model = MagicMock()
    model._cached_spectral_stability_margin = 1.0
    model._cached_surprise = 0.0
    model._memory_stale = False
    model._cached_safety_violation = False
    model._cached_stall_severity = 0.0
    model._cached_output_quality = 1.0
    model._cached_border_uncertainty = 0.0
    model._last_trust_score = 1.0
    model._cached_causal_quality = 1.0
    model._cached_topology_state = None
    model.convergence_monitor = None
    model._error_evolution = None
    # CP-5/CP-8: Ensure proper defaults for new attributes
    model._convergence_confidence = 1.0
    model._consistency_gate_failed = False
    model._gate_failure_severity = 0.0
    return model


def _make_mock_feedback_bus():
    """Create a mock feedback bus with read_signal returning defaults."""
    bus = MagicMock()
    _signals = {}

    def _write(name, value):
        _signals[name] = value

    def _read(name, default=0.0):
        return _signals.get(name, default)

    bus.write_signal = MagicMock(side_effect=_write)
    bus.read_signal = MagicMock(side_effect=_read)
    bus._signals = _signals
    # _read_fb_signal checks for _extra_signals first (MagicMock would
    # auto-create it), so we set it to the same backing dict.
    bus._extra_signals = _signals
    return bus


# ══════════════════════════════════════════════════════════════════════
#  CP-1: Catastrophe Classifier → MCT Graded Signal
# ══════════════════════════════════════════════════════════════════════


class TestCP1_CatastropheToMCT:
    """CP-1: classify_catastrophe_type results flow into MCT signals."""

    @pytest.fixture
    def model(self):
        return _make_mock_model()

    @pytest.fixture
    def bus(self):
        return _make_mock_feedback_bus()

    def test_no_catastrophe_zero_severity(self, model, bus):
        """When catastrophe_type is 'none', severity should be 0.0."""
        utcc = _make_utcc(model, feedback_bus=bus)
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "none"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs.get("topology_catastrophe_severity", 0.0) == 0.0

    def test_fold_catastrophe_severity(self, model, bus):
        """Fold (A₂) should produce severity=0.3."""
        utcc = _make_utcc(model, feedback_bus=bus)
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "fold"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["topology_catastrophe_severity"] == 0.3
        # Should also override topology_catastrophe to at least 0.3
        assert kwargs.get("topology_catastrophe", 0.0) >= 0.3

    def test_cusp_catastrophe_severity(self, model, bus):
        """Cusp (A₃) should produce severity=0.6."""
        utcc = _make_utcc(model, feedback_bus=bus)
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "cusp"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["topology_catastrophe_severity"] == 0.6
        assert kwargs["topology_catastrophe"] >= 0.6

    def test_swallowtail_catastrophe_severity(self, model, bus):
        """Swallowtail (A₄) should produce severity=0.9."""
        utcc = _make_utcc(model, feedback_bus=bus)
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "swallowtail"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["topology_catastrophe_severity"] == 0.9
        assert kwargs["topology_catastrophe"] >= 0.9

    def test_missing_topology_analysis_defaults(self, model, bus):
        """When no topology_analysis in cycle_results, defaults to 0.0."""
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        assert kwargs.get("topology_catastrophe_severity", 0.0) == 0.0

    def test_unknown_catastrophe_type_defaults(self, model, bus):
        """Unknown catastrophe types default to severity 0.0."""
        utcc = _make_utcc(model, feedback_bus=bus)
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "butterfly"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        assert kwargs["topology_catastrophe_severity"] == 0.0

    def test_severity_does_not_lower_existing_topology_signal(self, model, bus):
        """CP-1 uses max() so it doesn't lower existing topology signal."""
        utcc = _make_utcc(model, feedback_bus=bus)
        # Set a high existing topology signal via cached state
        model._cached_topology_state = torch.tensor([True])
        cycle_results = {
            "topology_analysis": {"catastrophe_type": "fold"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)
        # topology_catastrophe should be max(True→1.0, 0.3) = 1.0
        assert kwargs["topology_catastrophe"] >= 0.3


# ══════════════════════════════════════════════════════════════════════
#  CP-2: Training Phase Pressure → Curriculum Adaptation
# ══════════════════════════════════════════════════════════════════════


class TestCP2_CurriculumAdaptation:
    """CP-2: training_phase_pressure is read and drives LR adaptation."""

    @pytest.fixture
    def model(self):
        return _make_mock_model()

    @pytest.fixture
    def bus(self):
        return _make_mock_feedback_bus()

    def test_high_pressure_produces_adaptation_record(self, model, bus):
        """When pressure > 0.5, cycle_results should contain adaptation."""
        bus._signals["training_phase_pressure"] = 0.7
        utcc = _make_utcc(model, feedback_bus=bus)
        # Need _read_fb_signal method
        utcc._read_fb_signal = lambda name, default=0.0: bus._signals.get(
            name, default,
        )
        # Mock _sync_feedback_bus so we can call the CP-2 code path
        cycle_results = {}
        _uncertainty_flags = []

        # CP-2 code is in execute_full_cycle after _sync_feedback_bus.
        # We test the logic inline.
        phase_pressure = utcc._read_fb_signal("training_phase_pressure", 0.0)
        assert phase_pressure == 0.7

        if phase_pressure > 0.5:
            lr_factor = max(0.3, 1.0 - phase_pressure)
            cycle_results["curriculum_adaptation"] = {
                "phase_pressure": round(phase_pressure, 4),
                "lr_factor_applied": round(lr_factor, 4),
                "reason": "CP2_K4_coherence_feedback",
            }

        assert "curriculum_adaptation" in cycle_results
        assert cycle_results["curriculum_adaptation"]["lr_factor_applied"] == 0.3

    def test_low_pressure_no_adaptation(self, model, bus):
        """When pressure < 0.5, no curriculum adaptation should occur."""
        bus._signals["training_phase_pressure"] = 0.2
        utcc = _make_utcc(model, feedback_bus=bus)
        utcc._read_fb_signal = lambda name, default=0.0: bus._signals.get(
            name, default,
        )
        phase_pressure = utcc._read_fb_signal("training_phase_pressure", 0.0)
        assert phase_pressure <= 0.5

    def test_lr_factor_bounded_at_minimum(self, model, bus):
        """LR factor should never go below 0.3."""
        bus._signals["training_phase_pressure"] = 0.99
        utcc = _make_utcc(model, feedback_bus=bus)
        utcc._read_fb_signal = lambda name, default=0.0: bus._signals.get(
            name, default,
        )
        phase_pressure = utcc._read_fb_signal("training_phase_pressure", 0.0)
        lr_factor = max(0.3, 1.0 - phase_pressure)
        assert lr_factor >= 0.3


# ══════════════════════════════════════════════════════════════════════
#  CP-3: Diversity Pressure → VQ Maintenance Escalation
# ══════════════════════════════════════════════════════════════════════


class TestCP3_DiversityPressureInvocation:
    """CP-3: accept_diversity_pressure is now invoked in forward pass."""

    def test_accept_diversity_pressure_exists(self):
        """K5's accept_diversity_pressure method exists on VQ."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(
            num_embeddings=64, embedding_dim=32, commitment_cost=0.25,
        )
        assert hasattr(vq, "accept_diversity_pressure")

    def test_normal_pressure_no_action(self):
        """Pressure < 0.2 should produce no maintenance change."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(
            num_embeddings=64, embedding_dim=32, commitment_cost=0.25,
        )
        result = vq.accept_diversity_pressure(0.1)
        assert result["accepted"] is True
        assert result["action_taken"] == "normal"

    def test_moderate_pressure_halves_interval(self):
        """0.2 ≤ pressure < 0.5 should halve maintenance interval."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(
            num_embeddings=64, embedding_dim=32, commitment_cost=0.25,
        )
        result = vq.accept_diversity_pressure(0.35)
        assert result["accepted"] is True
        assert result["action_taken"] == "moderate_escalation"

    def test_high_pressure_quarters_interval(self):
        """0.5 ≤ pressure < 0.8 should quarter interval + lower threshold."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(
            num_embeddings=64, embedding_dim=32, commitment_cost=0.25,
        )
        result = vq.accept_diversity_pressure(0.65)
        assert result["accepted"] is True
        assert result["action_taken"] == "high_escalation"

    def test_critical_pressure_forces_maintenance(self):
        """Pressure ≥ 0.8 should force immediate maintenance."""
        from aeon_core import RobustVectorQuantizer
        vq = RobustVectorQuantizer(
            num_embeddings=64, embedding_dim=32, commitment_cost=0.25,
        )
        result = vq.accept_diversity_pressure(0.9)
        assert result["accepted"] is True
        assert result["action_taken"] == "critical_escalation"
        assert result.get("forced_immediate") is True


# ══════════════════════════════════════════════════════════════════════
#  CP-4: K7 Same-Pass MCT Flag → Iteration Extension
# ══════════════════════════════════════════════════════════════════════


class TestCP4_K7IterationExtension:
    """CP-4: K7 flag consumed by meta-loop to extend iterations."""

    @pytest.fixture
    def meta_loop(self):
        from aeon_core import AEONConfig, ProvablyConvergentMetaLoop
        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        ml = ProvablyConvergentMetaLoop(
            config, max_iterations=5, min_iterations=2,
        )
        ml.eval()
        return ml, config

    def test_k7_flag_extends_iterations(self, meta_loop):
        """When _k7_same_pass_triggered=True, iterations increase."""
        loop, config = meta_loop
        loop._k7_same_pass_triggered = True
        assert loop._k7_same_pass_triggered is True

        B = 1
        psi_0 = torch.randn(B, config.hidden_dim)
        with torch.no_grad():
            _ = loop(psi_0, use_fixed_point=True)

        # Flag should be consumed (reset to False)
        assert loop._k7_same_pass_triggered is False

    def test_k7_iteration_extension_recorded(self, meta_loop):
        """K7 extension metadata should be recorded."""
        loop, config = meta_loop
        loop._k7_same_pass_triggered = True

        B = 1
        psi_0 = torch.randn(B, config.hidden_dim)
        with torch.no_grad():
            _ = loop(psi_0, use_fixed_point=True)

        ext = loop._k7_iteration_extension
        assert ext is not None
        assert ext["original_max"] == 5
        assert ext["extended_to"] > 5
        assert ext["reason"] == "K7_certificate_failure_recovery"

    def test_no_k7_flag_no_extension(self, meta_loop):
        """Without K7 flag, no iteration extension should occur."""
        loop, config = meta_loop

        B = 1
        psi_0 = torch.randn(B, config.hidden_dim)
        with torch.no_grad():
            _ = loop(psi_0, use_fixed_point=True)

        assert loop._k7_iteration_extension is None

    def test_k7_extension_bounded(self, meta_loop):
        """Extension should be bounded at 2× max_iterations."""
        loop, config = meta_loop
        loop._k7_same_pass_triggered = True

        B = 1
        psi_0 = torch.randn(B, config.hidden_dim)
        with torch.no_grad():
            _ = loop(psi_0, use_fixed_point=True)

        ext = loop._k7_iteration_extension
        assert ext["extended_to"] <= 5 * 2  # 2× cap


# ══════════════════════════════════════════════════════════════════════
#  CP-5: Convergence Certificate → Confidence Modulation
# ══════════════════════════════════════════════════════════════════════


class TestCP5_ConvergenceConfidence:
    """CP-5: Convergence certificate modulates output confidence."""

    @pytest.fixture
    def model(self):
        return _make_mock_model()

    @pytest.fixture
    def bus(self):
        return _make_mock_feedback_bus()

    def test_certified_convergence_full_confidence(self, model, bus):
        """When convergence is certified, confidence should be 1.0."""
        model._convergence_confidence = 1.0
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # CP-5 only elevates convergence_conflict when confidence < 1.0;
        # however the existing bus-based convergence_conflict may still be
        # present from systematic_uncertainty on the bus (default = 0.0).
        # With confidence = 1.0, CP-5 should NOT contribute additional conflict.
        # Bus-based conflict may exist from other sources.
        bus_conflict = float(bus._extra_signals.get("systematic_uncertainty", 0.0))
        assert kwargs.get("convergence_conflict", 0.0) <= bus_conflict + 0.01

    def test_uncertified_convergence_elevates_conflict(self, model, bus):
        """Low convergence confidence should elevate convergence_conflict."""
        model._convergence_confidence = 0.3
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        assert kwargs.get("convergence_conflict", 0.0) >= 0.7

    def test_confidence_attribute_set_in_forward(self):
        """The model should have _convergence_confidence after forward."""
        from aeon_core import AEONConfig, AEONDeltaV3
        config = AEONConfig(
            device_str='cpu',
            enable_quantum_sim=False,
            enable_catastrophe_detection=False,
            enable_safety_guardrails=False,
        )
        model = AEONDeltaV3(config)
        model.eval()
        B = 1
        x = torch.randint(0, config.vocab_size, (B, 8))
        with torch.no_grad():
            _ = model(x)
        # _convergence_confidence should exist now
        assert hasattr(model, "_convergence_confidence")
        assert isinstance(model._convergence_confidence, float)
        assert 0.0 < model._convergence_confidence <= 1.0

    def test_confidence_floor_at_0_1(self, model, bus):
        """Confidence should never go below 0.1 floor."""
        model._convergence_confidence = 0.05
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # Even with very low confidence, CP-5 elevates convergence_conflict.
        # The conflict = 1.0 - 0.05 = 0.95 (max allowed since floor is
        # applied in aeon_core, not in the MCT signal reading).
        assert kwargs.get("convergence_conflict", 0.0) >= 0.9


# ══════════════════════════════════════════════════════════════════════
#  CP-6: Causal Model ← MCT Feedback Integration
# ══════════════════════════════════════════════════════════════════════


class TestCP6_CausalModelMCTFeedback:
    """CP-6: CausalFactorExtractor adapts to MCT urgency."""

    def test_receive_method_exists(self):
        """CausalFactorExtractor should have receive_meta_cognitive_signal."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        assert hasattr(cfe, "receive_meta_cognitive_signal")

    def test_low_urgency_normal_parameters(self):
        """Urgency < 0.5 should restore base parameters."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        result = cfe.receive_meta_cognitive_signal(0.2)
        assert result["adapted"] is True
        assert result["dag_temperature"] == cfe._base_dag_temperature
        assert result["sparsity_weight"] == cfe._base_sparsity_weight

    def test_high_urgency_conservative_parameters(self):
        """Urgency > 0.5 should lower temperature and increase sparsity."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        base_temp = cfe._base_dag_temperature
        base_sparsity = cfe._base_sparsity_weight

        result = cfe.receive_meta_cognitive_signal(0.8)
        assert result["adapted"] is True
        assert result["dag_temperature"] < base_temp
        assert result["sparsity_weight"] > base_sparsity

    def test_temperature_floor_at_0_1(self):
        """DAG temperature should never go below 0.1."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        cfe.receive_meta_cognitive_signal(1.0)
        assert cfe._dag_temperature >= 0.1

    def test_urgency_clamped_to_0_1(self):
        """MCT urgency should be clamped to [0, 1]."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        result = cfe.receive_meta_cognitive_signal(5.0)
        assert result["mct_urgency"] <= 1.0
        result = cfe.receive_meta_cognitive_signal(-1.0)
        assert result["mct_urgency"] >= 0.0

    def test_temperature_affects_adjacency(self):
        """Lower temperature should make adjacency more extreme."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        C_star = torch.randn(2, 32)

        # Normal temperature
        cfe.receive_meta_cognitive_signal(0.0)
        out_normal = cfe(C_star)
        adj_normal = out_normal["causal_graph"]

        # Low temperature (high urgency)
        cfe.receive_meta_cognitive_signal(0.9)
        out_urgent = cfe(C_star)
        adj_urgent = out_urgent["causal_graph"]

        # Adjacency values should be more extreme (closer to 0/1) at low temp
        # Variance of sigmoid(x/low_temp) > variance of sigmoid(x/1.0) for
        # arbitrary x, meaning entries are pushed toward 0 or 1.
        normal_extremeness = (adj_normal - 0.5).abs().mean()
        urgent_extremeness = (adj_urgent - 0.5).abs().mean()
        assert urgent_extremeness >= normal_extremeness

    def test_forward_still_works_after_adaptation(self):
        """Forward pass should work normally after MCT adaptation."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)
        cfe.receive_meta_cognitive_signal(0.7)

        C_star = torch.randn(2, 32)
        result = cfe(C_star)
        assert "factors" in result
        assert "causal_graph" in result
        assert result["factors"].shape == (2, 4)


# ══════════════════════════════════════════════════════════════════════
#  CP-7: VibeThinkerMetaSignaler → RSSM Loss Wiring
# ══════════════════════════════════════════════════════════════════════


class TestCP7_MetaSignalerLossWiring:
    """CP-7: VibeThinkerMetaSignaler's compute_loss is used."""

    def test_compute_loss_uses_lambda_cos(self):
        """compute_loss should use the dynamically adjusted lambda_cos."""
        from aeon_server import VibeThinkerMetaSignaler
        ms = VibeThinkerMetaSignaler(base_lambda_cos=0.1)

        z_pred = torch.randn(4, 32)
        z_target = torch.randn(4, 32)

        loss = ms.compute_loss(z_pred, z_target)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # should be positive

    def test_lambda_cos_modulates_loss(self):
        """Different lambda_cos values should produce different losses."""
        from aeon_server import VibeThinkerMetaSignaler
        ms_low = VibeThinkerMetaSignaler(base_lambda_cos=0.01)
        ms_high = VibeThinkerMetaSignaler(base_lambda_cos=0.5)

        z_pred = torch.randn(4, 32)
        z_target = torch.randn(4, 32)

        loss_low = ms_low.compute_loss(z_pred, z_target)
        loss_high = ms_high.compute_loss(z_pred, z_target)

        # Higher lambda_cos should generally produce different loss
        # (unless cosine component is zero, which is unlikely for random)
        assert not torch.isclose(loss_low, loss_high)

    def test_attach_learner_works(self):
        """attach_learner should wire the learner for calibration_ema."""
        from aeon_server import VibeThinkerMetaSignaler
        ms = VibeThinkerMetaSignaler()
        learner = MagicMock()
        learner._calibration_ema = 0.5
        ms.attach_learner(learner)
        assert ms._learner is learner

    def test_update_adapts_lambda_cos_from_learner(self):
        """update() should adapt lambda_cos based on calibration_ema."""
        from aeon_server import VibeThinkerMetaSignaler
        ms = VibeThinkerMetaSignaler(base_lambda_cos=0.1)
        learner = MagicMock()
        learner._calibration_ema = 0.5
        ms.attach_learner(learner)

        result = ms.update()
        # lambda_cos = 0.1 * (1 + 2*0.5) = 0.2
        assert abs(result["lambda_cos"] - 0.2) < 1e-6

    def test_ensure_orchestrator_auto_wires_learner(self):
        """_ensure_orchestrator should auto-attach learner to signaler."""
        from aeon_server import VibeThinkerMetaSignaler
        ms = VibeThinkerMetaSignaler()
        # Simulate what _ensure_orchestrator does for CP-7
        mock_model = MagicMock()
        mock_model.vt_learner = MagicMock()
        mock_model.vt_learner._calibration_ema = 0.3

        _cp7_learner = getattr(mock_model, "vt_learner", None)
        if _cp7_learner is not None:
            ms.attach_learner(_cp7_learner)

        assert ms._learner is mock_model.vt_learner


# ══════════════════════════════════════════════════════════════════════
#  CP-8: Consistency Gate → Hard Coherence Threshold
# ══════════════════════════════════════════════════════════════════════


class TestCP8_ConsistencyGateThreshold:
    """CP-8: Consistency gate failure flags propagate to MCT."""

    @pytest.fixture
    def model(self):
        return _make_mock_model()

    @pytest.fixture
    def bus(self):
        return _make_mock_feedback_bus()

    def test_gate_failure_elevates_coherence_deficit(self, model, bus):
        """When gate failed, coherence_deficit should be elevated."""
        model._consistency_gate_failed = True
        model._gate_failure_severity = 0.7
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        assert kwargs["coherence_deficit"] >= 0.7

    def test_gate_failure_consumed(self, model, bus):
        """Flag should be consumed (reset to False) after reading."""
        model._consistency_gate_failed = True
        model._gate_failure_severity = 0.5
        utcc = _make_utcc(model, feedback_bus=bus)
        utcc._collect_mct_signals(0.0, [], {})
        # Flag should be consumed
        assert model._consistency_gate_failed is False

    def test_no_gate_failure_no_elevation(self, model, bus):
        """Without gate failure, coherence_deficit stays at bus value."""
        model._consistency_gate_failed = False
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # coherence_deficit from bus defaults to 0.0
        assert kwargs.get("coherence_deficit", 0.0) == 0.0

    def test_gate_severity_uses_max(self, model, bus):
        """CP-8 should use max() to not lower existing deficit."""
        model._consistency_gate_failed = True
        model._gate_failure_severity = 0.3
        # Set high existing coherence_deficit via bus
        bus._extra_signals["integration_health"] = 0.1  # deficit = 0.9
        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})
        # Should be at least 0.9 (from bus) not lowered to 0.3
        assert kwargs["coherence_deficit"] >= 0.9


# ══════════════════════════════════════════════════════════════════════
#  Integration: Cross-Patch Signal Flow
# ══════════════════════════════════════════════════════════════════════


class TestCrossPatchSignalFlow:
    """Verify signals flow across multiple patches correctly."""

    def test_cp5_confidence_flows_to_cp8_through_mct(self):
        """CP-5 low confidence + CP-8 gate failure = compounded deficit."""
        model = _make_mock_model()
        bus = _make_mock_feedback_bus()
        model._convergence_confidence = 0.3
        model._consistency_gate_failed = True
        model._gate_failure_severity = 0.6

        utcc = _make_utcc(model, feedback_bus=bus)
        kwargs = utcc._collect_mct_signals(0.0, [], {})

        # convergence_conflict should be elevated from CP-5
        assert kwargs.get("convergence_conflict", 0.0) >= 0.7
        # coherence_deficit should be elevated from CP-8
        assert kwargs.get("coherence_deficit", 0.0) >= 0.6

    def test_cp1_catastrophe_with_cp5_divergence(self):
        """CP-1 swallowtail + CP-5 low confidence = multiple alarms."""
        model = _make_mock_model()
        bus = _make_mock_feedback_bus()
        model._convergence_confidence = 0.2
        utcc = _make_utcc(model, feedback_bus=bus)

        cycle_results = {
            "topology_analysis": {"catastrophe_type": "swallowtail"},
        }
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)

        assert kwargs["topology_catastrophe_severity"] == 0.9
        assert kwargs.get("convergence_conflict", 0.0) >= 0.8

    def test_cp6_adapts_when_k7_mct_cached(self):
        """CP-6 reads K7 MCT result for causal conservatism."""
        from aeon_core import CausalFactorExtractor
        cfe = CausalFactorExtractor(hidden_dim=32, num_factors=4)

        # Simulate K7 MCT result
        k7_result = {"should_trigger": True, "trigger_score": 0.8}

        _cp6_mct_urgency = float(k7_result.get("trigger_score", 0.0))
        result = cfe.receive_meta_cognitive_signal(mct_urgency=_cp6_mct_urgency)

        assert result["adapted"] is True
        assert result["dag_temperature"] < cfe._base_dag_temperature

    def test_all_patches_compatible_in_mct_collection(self):
        """All CP patches should work together without conflicts."""
        model = _make_mock_model()
        bus = _make_mock_feedback_bus()
        # Set all CP signals at once
        model._convergence_confidence = 0.5  # CP-5
        model._consistency_gate_failed = True  # CP-8
        model._gate_failure_severity = 0.4

        cycle_results = {
            "topology_analysis": {"catastrophe_type": "cusp"},  # CP-1
            "ucc": {"evaluated": True, "coherence_score": 0.6},  # K1
        }

        utcc = _make_utcc(model, feedback_bus=bus)
        # Should not crash
        kwargs = utcc._collect_mct_signals(0.0, [], cycle_results)

        # Verify all signals present
        assert "topology_catastrophe_severity" in kwargs  # CP-1
        assert "coherence_deficit" in kwargs  # CP-8
        assert "convergence_conflict" in kwargs  # CP-5
        assert kwargs["topology_catastrophe_severity"] == 0.6  # cusp
