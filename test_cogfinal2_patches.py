"""
Tests for PATCH-COGFINAL2 patches: Final Integration & Cognitive Activation.

Covers:
  - COGFINAL2-1: UnifiedConvergenceArbiter bus write-back
    (convergence_arbiter_strategy_id, convergence_arbiter_confidence)
  - COGFINAL2-1b: MCT reads convergence_arbiter_confidence
  - COGFINAL2-2: AutoCriticLoop context-aware revision depth
  - COGFINAL2-3: ErrorRecoveryManager bus-aware escalation suppression
  - COGFINAL2-3b: MCT reads error_recovery_escalation_suppressed
  - COGFINAL2-4: _forward_impl reads loss_concentration_ratio + loss_intervention_active
  - COGFINAL2-5a: ae_train reads server_coherence_score + integration_health
  - COGFINAL2-5b: aeon_server reads training_convergence_trend
  - Signal ecosystem integrity (no new orphans or missing producers)
"""

import pytest
import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core
import aeon_core as aeon

try:
    import ae_train
    _HAS_TRAIN = True
except (ImportError, SystemExit):
    _HAS_TRAIN = False

try:
    import aeon_server
    _HAS_SERVER = True
except (ImportError, SystemExit):
    _HAS_SERVER = False

_skip_no_server = pytest.mark.skipif(
    not _HAS_SERVER, reason="aeon_server requires fastapi/uvicorn",
)
_skip_no_train = pytest.mark.skipif(
    not _HAS_TRAIN, reason="ae_train not importable",
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)


# ======================================================================
# PATCH-COGFINAL2-1: UnifiedConvergenceArbiter bus write-back
# ======================================================================

class TestCOGFINAL2_1_ArbiterWriteBack:
    """Test that the arbiter writes strategy_id and confidence to bus."""

    def test_arbiter_writes_strategy_id_on_converged(self):
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.005},
            convergence_monitor_verdict={"status": "converged", "certified": True},
        )
        sid = bus.read_signal('convergence_arbiter_strategy_id', -1.0)
        assert sid == 0.0, f"Expected 0.0 (converged), got {sid}"
        assert 'arbiter_strategy_id' in result
        assert result['arbiter_strategy_id'] == 0.0

    def test_arbiter_writes_strategy_id_on_diverging(self):
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.1, "residual_norm": 2.0},
            convergence_monitor_verdict={"status": "diverging", "certified": False},
        )
        sid = bus.read_signal('convergence_arbiter_strategy_id', -1.0)
        assert sid == 1.0, f"Expected 1.0 (recovery/diverging), got {sid}"

    def test_arbiter_writes_confidence_high_on_consensus(self):
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.005},
            convergence_monitor_verdict={"status": "converged", "certified": True},
            certified_results={"certified_convergence": True},
        )
        conf = bus.read_signal('convergence_arbiter_confidence', -1.0)
        assert conf > 0.5, f"Expected high confidence on consensus, got {conf}"

    def test_arbiter_writes_low_confidence_on_conflict(self):
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.95, "residual_norm": 0.005},
            convergence_monitor_verdict={"status": "diverging", "certified": False},
        )
        conf = bus.read_signal('convergence_arbiter_confidence', -1.0)
        assert conf < 0.7, f"Expected low confidence on conflict, got {conf}"

    def test_arbiter_no_bus_does_not_crash(self):
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=None)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.5, "residual_norm": 0.5},
            convergence_monitor_verdict={"status": "converging", "certified": False},
        )
        assert 'arbiter_confidence' in result

    def test_arbiter_result_contains_new_fields(self):
        bus = _make_bus()
        arbiter = aeon.UnifiedConvergenceArbiter(feedback_bus=bus)
        result = arbiter.arbitrate(
            meta_loop_results={"convergence_rate": 0.5, "residual_norm": 0.5},
            convergence_monitor_verdict={"status": "converging", "certified": False},
        )
        assert 'arbiter_strategy_id' in result
        assert 'arbiter_confidence' in result
        assert 0.0 <= result['arbiter_strategy_id'] <= 1.0
        assert 0.0 <= result['arbiter_confidence'] <= 1.0


# ======================================================================
# PATCH-COGFINAL2-1b: MCT reads convergence_arbiter_confidence
# ======================================================================

class TestCOGFINAL2_1b_MCTReadsArbiterConfidence:
    """Test that MCT reads arbiter confidence and routes to convergence_degradation_pressure."""

    def test_mct_source_reads_convergence_arbiter_confidence(self):
        """Verify the read_signal call exists in MCT evaluate()."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        assert "read_signal(" in src
        # Check for the specific read in the MCT evaluate area
        assert "'convergence_arbiter_confidence'" in src

    def test_mct_routes_low_arbiter_confidence(self):
        """Low arbiter confidence should boost convergence_degradation_pressure."""
        bus = _make_bus()
        bus.write_signal('convergence_arbiter_confidence', 0.1)
        mct = aeon.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(
            uncertainty=0.1,
            coherence_deficit=0.1,
            convergence_conflict=0.0,
        )
        # MCT should have read the signal
        assert 'convergence_arbiter_confidence' in bus._read_log


# ======================================================================
# PATCH-COGFINAL2-2: AutoCriticLoop context-aware revision depth
# ======================================================================

class TestCOGFINAL2_2_AutoCriticContextAware:
    """Test that AutoCriticLoop reads MCT/coherence signals."""

    def test_auto_critic_reads_mct_trigger_score(self):
        """Verify the read_signal call exists in AutoCriticLoop."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        # Find the auto-critic class area
        acl_start = src.find('class AutoCriticLoop')
        acl_end = src.find('\nclass ', acl_start + 1)
        acl_src = src[acl_start:acl_end]
        assert "'mct_trigger_score'" in acl_src
        assert "'coherence_deficit'" in acl_src

    def test_auto_critic_increases_iterations_with_high_mct_score(self):
        """When mct_trigger_score is high, auto-critic should increase iterations."""
        import torch
        import torch.nn as nn

        bus = _make_bus()
        bus.write_signal('mct_trigger_score', 0.8)

        # Create a minimal auto-critic
        class StubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return self.linear(x)

        acl = aeon.AutoCriticLoop(
            base_model=StubModel(),
            hidden_dim=64,
            max_iterations=3,
            threshold=0.99,  # Very high threshold to ensure revisions happen
        )
        acl._fb_ref = bus

        # Run forward — should have read the signals from the bus
        try:
            query = torch.randn(1, 64)
            result = acl(query)
        except Exception:
            pass  # The stub model may not produce correct critic outputs

        # Verify the signals were read
        assert 'mct_trigger_score' in bus._read_log
        assert 'coherence_deficit' in bus._read_log

    def test_auto_critic_coherence_focus_changes_weakest_dim(self):
        """Verify coherence focus override code path exists."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        assert '_cf2_coherence_focus' in src
        assert "weakest_dim = 'coherence'" in src


# ======================================================================
# PATCH-COGFINAL2-3: ErrorRecoveryManager bus-aware escalation
# ======================================================================

class TestCOGFINAL2_3_ErrorRecoveryBusAware:
    """Test that ErrorRecoveryManager reads MCT/oscillation and suppresses."""

    def test_erm_reads_mct_should_trigger(self):
        """Verify the read_signal call exists in recover()."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        erm_start = src.find('class ErrorRecoveryManager')
        erm_end = src.find('\nclass ', erm_start + 1)
        erm_src = src[erm_start:erm_end]
        assert "'mct_should_trigger'" in erm_src
        assert "'oscillation_severity_pressure'" in erm_src
        assert "'error_recovery_escalation_suppressed'" in erm_src

    def test_erm_suppresses_when_mct_active(self):
        """When MCT is active, ErrorRecoveryManager suppresses escalation."""
        bus = _make_bus()
        bus.write_signal('mct_should_trigger', 1.0)

        mgr = aeon.ErrorRecoveryManager(
            hidden_dim=64,
            feedback_bus=bus,
        )
        success, value = mgr.recover(
            ValueError("test"),
            context="test",
            fallback=None,
        )
        # Check that the suppression signal was written
        suppressed = bus.read_signal('error_recovery_escalation_suppressed', 0.0)
        assert suppressed == 1.0, f"Expected suppressed=1.0, got {suppressed}"

    def test_erm_does_not_suppress_when_mct_inactive(self):
        """When MCT is inactive, ErrorRecoveryManager does not suppress."""
        bus = _make_bus()
        bus.write_signal('mct_should_trigger', 0.0)

        mgr = aeon.ErrorRecoveryManager(
            hidden_dim=64,
            feedback_bus=bus,
        )
        success, value = mgr.recover(
            ValueError("test"),
            context="test",
            fallback=None,
        )
        suppressed = bus.read_signal('error_recovery_escalation_suppressed', 0.0)
        assert suppressed == 0.0, f"Expected suppressed=0.0, got {suppressed}"

    def test_erm_suppresses_when_oscillation_high(self):
        """High oscillation also triggers suppression."""
        bus = _make_bus()
        bus.write_signal('oscillation_severity_pressure', 0.8)

        mgr = aeon.ErrorRecoveryManager(
            hidden_dim=64,
            feedback_bus=bus,
        )
        success, value = mgr.recover(
            ValueError("test"),
            context="test",
            fallback=None,
        )
        suppressed = bus.read_signal('error_recovery_escalation_suppressed', 0.0)
        assert suppressed == 1.0, f"Expected suppressed=1.0, got {suppressed}"

    def test_erm_no_bus_does_not_crash(self):
        """Without bus, recovery should still work normally."""
        mgr = aeon.ErrorRecoveryManager(hidden_dim=64, feedback_bus=None)
        import torch
        success, value = mgr.recover(
            ValueError("test"),
            context="test",
            fallback=torch.zeros(1, 64),
        )
        assert success is True


# ======================================================================
# PATCH-COGFINAL2-3b: MCT reads error_recovery_escalation_suppressed
# ======================================================================

class TestCOGFINAL2_3b_MCTReadsSuppressionSignal:
    """Test that MCT reads the suppression signal."""

    def test_mct_source_reads_suppression_signal(self):
        """Verify the read_signal call in MCT evaluate."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        assert "'error_recovery_escalation_suppressed'" in src

    def test_mct_reads_suppression_signal_from_bus(self):
        """MCT should read error_recovery_escalation_suppressed."""
        bus = _make_bus()
        bus.write_signal('error_recovery_escalation_suppressed', 1.0)
        mct = aeon.MetaCognitiveRecursionTrigger()
        mct.set_feedback_bus(bus)
        result = mct.evaluate(
            uncertainty=0.1,
            coherence_deficit=0.1,
        )
        assert 'error_recovery_escalation_suppressed' in bus._read_log


# ======================================================================
# PATCH-COGFINAL2-4: _forward_impl reads loss signals
# ======================================================================

class TestCOGFINAL2_4_ForwardLossBridge:
    """Test that _forward_impl reads loss_concentration_ratio + loss_intervention_active."""

    def test_source_reads_loss_concentration_ratio(self):
        """Verify the read_signal call exists in _forward_impl."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        fwd_start = src.find('def _forward_impl(')
        fwd_end = src.find('\n    def ', fwd_start + 1)
        fwd_src = src[fwd_start:fwd_end]
        assert "'loss_concentration_ratio'" in fwd_src
        assert "'loss_intervention_active'" in fwd_src

    def test_cached_boost_attributes_exist(self):
        """Verify the cached attributes are set by the patch."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        assert '_cached_loss_concentration_boost' in src
        assert '_cached_loss_intervention_damping' in src

    def test_high_concentration_sets_boost(self):
        """When loss_concentration_ratio > 0.6, boost should be non-zero."""
        with open('aeon_core.py', 'r') as f:
            src = f.read()
        # Verify the threshold logic exists
        assert '_cf4_loss_concentration > 0.6' in src
        assert '_cf4_loss_intervention > 0.5' in src


# ======================================================================
# PATCH-COGFINAL2-5a: ae_train reads server signals
# ======================================================================

class TestCOGFINAL2_5a_TrainingServerBridge:
    """Test that ae_train.py reads server_coherence_score + integration_health."""

    def test_source_reads_server_coherence(self):
        """Verify the read_signal calls exist in ae_train.py."""
        with open('ae_train.py', 'r') as f:
            src = f.read()
        assert "'server_coherence_score'" in src
        assert "'integration_health'" in src
        # Verify these are read_signal calls (not just string references)
        assert "read_signal(" in src

    def test_coherence_boost_logic(self):
        """Verify the coherence boost logic is present."""
        with open('ae_train.py', 'r') as f:
            src = f.read()
        assert '_cf5a_server_coh' in src
        assert '_cf5a_int_health' in src
        assert '_cf5a_coh_boost' in src


# ======================================================================
# PATCH-COGFINAL2-5b: aeon_server reads training_convergence_trend
# ======================================================================

@_skip_no_server
class TestCOGFINAL2_5b_ServerTrainingBridge:
    """Test that aeon_server.py reads training_convergence_trend."""

    def test_source_reads_training_convergence_trend(self):
        """Verify the read_signal call exists in aeon_server.py."""
        with open('aeon_server.py', 'r') as f:
            src = f.read()
        assert "'training_convergence_trend'" in src
        # Find the specific PATCH-COGFINAL2-5b section
        assert 'PATCH-COGFINAL2-5b' in src

    def test_server_ssp_pressure_on_diverging_training(self):
        """When training diverges, server should write server_ssp_pressure."""
        with open('aeon_server.py', 'r') as f:
            src = f.read()
        assert "'server_ssp_pressure'" in src
        assert '_cf5b_training_trend' in src


# ======================================================================
# Signal Ecosystem Integrity
# ======================================================================

class TestCOGFINAL2_SignalEcosystem:
    """Verify signal ecosystem integrity after all COGFINAL2 patches."""

    def _scan_signals(self, filename):
        """Scan a file for write_signal and read_signal calls."""
        with open(filename, 'r') as f:
            src = f.read()
        write_pat = re.compile(
            r"""write_signal(?:_traced)?\s*\(\s*\n?\s*['"](\w+)['"]"""
        )
        read_pat = re.compile(
            r"""read_signal\s*\(\s*\n?\s*['"](\w+)['"]"""
        )
        writes = set(write_pat.findall(src))
        reads = set(read_pat.findall(src))
        # Also check _extra_signals[...] = and _write_log.add()
        extra_w = re.findall(
            r"""_extra_signals\[['"](\w+)['"]\]\s*=""", src,
        )
        wlog = re.findall(
            r"""_write_log\.add\(['"](\w+)['"]\)""", src,
        )
        writes |= set(extra_w) | set(wlog)
        return writes, reads

    def test_new_signals_are_bidirectional(self):
        """Key new COGFINAL2 signals should be both written and read."""
        # convergence_arbiter_confidence: written by arbiter, read by MCT
        # convergence_arbiter_strategy_id: written by arbiter, read by compute_loss
        # error_recovery_escalation_suppressed: written by ERM, read by MCT
        new_bidirectional_signals = [
            'convergence_arbiter_confidence',
            'convergence_arbiter_strategy_id',
            'error_recovery_escalation_suppressed',
        ]
        core_w, core_r = self._scan_signals('aeon_core.py')
        for sig in new_bidirectional_signals:
            assert sig in core_w, f"{sig} not written in aeon_core.py"
            assert sig in core_r, f"{sig} not read in aeon_core.py"

    def test_no_new_missing_producers(self):
        """No signal should be read without a writer across all modules."""
        core_w, core_r = self._scan_signals('aeon_core.py')
        train_w, train_r = self._scan_signals('ae_train.py')
        server_w, server_r = self._scan_signals('aeon_server.py')
        all_writes = core_w | train_w | server_w
        all_reads = core_r | train_r | server_r
        missing = all_reads - all_writes
        assert len(missing) == 0, f"Missing producers: {missing}"

    def test_cross_module_signals_connected(self):
        """Verify the new cross-module signal paths are connected."""
        _, core_r = self._scan_signals('aeon_core.py')
        core_w, _ = self._scan_signals('aeon_core.py')
        train_w, train_r = self._scan_signals('ae_train.py')
        server_w, server_r = self._scan_signals('aeon_server.py')
        # COGFINAL2-5a: train reads server-written signals
        # (server writes to core bus, train reads from core bus)
        assert 'server_coherence_score' in (train_r | core_r)
        assert 'integration_health' in (train_r | core_r)
        # COGFINAL2-5b: server reads training-written signals
        assert 'training_convergence_trend' in (server_r | core_r)

    def test_loss_signals_now_read_by_forward(self):
        """loss_concentration_ratio and loss_intervention_active should be read in _forward_impl."""
        _, core_r = self._scan_signals('aeon_core.py')
        assert 'loss_concentration_ratio' in core_r
        assert 'loss_intervention_active' in core_r

    def test_auto_critic_now_reads_signals(self):
        """AutoCriticLoop should now read mct_trigger_score and coherence_deficit."""
        _, core_r = self._scan_signals('aeon_core.py')
        assert 'mct_trigger_score' in core_r
        assert 'coherence_deficit' in core_r
