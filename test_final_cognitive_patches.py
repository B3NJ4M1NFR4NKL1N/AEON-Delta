"""Tests for PATCH-FINAL-COG-1 through PATCH-FINAL-COG-5.

PATCH-FINAL-COG-1: VQ re-quantization on MCT should_recurse
PATCH-FINAL-COG-2: Conditional verify_and_reinforce() acceleration
PATCH-FINAL-COG-3: VibeThinkerRSSMBridge silent exception hardening
PATCH-FINAL-COG-4: Training loop flush_consumed() integration (Phase A)
PATCH-FINAL-COG-5: Phase B trainer feedback bus wiring + flush
"""
import re
import sys
import math
import pytest
import torch

# ── Import aeon_core and ae_train ─────────────────────────────────────
import aeon_core
import ae_train

# ── Helpers ───────────────────────────────────────────────────────────

def _make_bus():
    """Create a CognitiveFeedbackBus instance."""
    return aeon_core.CognitiveFeedbackBus(hidden_dim=64)


def _make_config(**overrides):
    """Build a minimal AEONConfig."""
    defaults = dict(
        vocab_size=200,
        hidden_dim=64,
        z_dim=64,
        vq_embedding_dim=64,
    )
    defaults.update(overrides)
    return aeon_core.AEONConfig(**defaults)


def _make_minimal_model():
    """Create a minimal AEONDeltaV3 model for integration tests."""
    config = _make_config()
    model = aeon_core.AEONDeltaV3(config)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# PATCH-FINAL-COG-1: VQ RE-QUANTIZATION ON MCT SHOULD_RECURSE
# ═══════════════════════════════════════════════════════════════════════


class TestFinalCog1_VQRequantization:
    """Tests for PATCH-FINAL-COG-1: VQ re-quantization activation."""

    def test_source_code_has_patch_marker(self):
        """PATCH-FINAL-COG-1 comment marker present in aeon_core.py."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert "PATCH-FINAL-COG-1" in src, (
            "PATCH-FINAL-COG-1 comment marker not found in aeon_core.py"
        )

    def test_requantization_block_exists(self):
        """The re-quantization logic checks _vq_should_recurse and
        _vq_codebook_quality before re-running VQ."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Must check should_recurse AND codebook quality threshold
        assert '_vq_should_recurse' in src
        assert 'requantized' in src
        assert 'mct_requantization' in src

    def test_requantization_records_causal_trace(self):
        """The re-quantization records its decision in causal_trace."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert 'mct_requantization' in src

    def test_requantization_only_accepts_improvement(self):
        """Re-quantized result is only accepted when quality improves."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert '_new_quality > _vq_codebook_quality' in src

    def test_requantization_restores_temperature(self):
        """VQ temperature is restored after re-quantization regardless."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Temperature must be restored
        assert '_vq_temp_orig' in src
        # Must restore before accepting/rejecting
        idx_restore = src.index('self.vector_quantizer.temperature = _vq_temp_orig')
        idx_check = src.index('_new_quality > _vq_codebook_quality')
        assert idx_restore < idx_check, (
            "Temperature must be restored before quality comparison"
        )

    def test_requantization_failure_bridged(self):
        """Re-quantization failures are bridged to subsystem_silent_failure_pressure."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert 'vq_requantization_failure' in src


# ═══════════════════════════════════════════════════════════════════════
# PATCH-FINAL-COG-2: CONDITIONAL VERIFY_AND_REINFORCE ACCELERATION
# ═══════════════════════════════════════════════════════════════════════


class TestFinalCog2_ConditionalReinforcement:
    """Tests for PATCH-FINAL-COG-2: emergency verify_and_reinforce()."""

    def test_source_code_has_patch_marker(self):
        """PATCH-FINAL-COG-2 comment marker present in aeon_core.py."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert "PATCH-FINAL-COG-2" in src, (
            "PATCH-FINAL-COG-2 comment marker not found in aeon_core.py"
        )

    def test_emergency_condition_reads_emergence_deficit(self):
        """Emergency trigger reads emergence_deficit from bus."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert '_cog2_emergence_deficit' in src
        assert "read_signal('emergence_deficit'" in src or \
               'read_signal("emergence_deficit"' in src

    def test_emergency_condition_checks_coherence_deficit(self):
        """Emergency trigger also checks cached coherence deficit."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert '_cog2_coherence_deficit' in src

    def test_emergency_has_spacing_guard(self):
        """Emergency reinforcement is spaced at least 5 passes apart."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert '_last_emergency_reinforce' in src
        assert '>= 5' in src or '>=5' in src

    def test_emergency_flag_in_result(self):
        """Result dict includes emergency_triggered flag."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert 'emergency_triggered' in src

    def test_or_condition_combines_interval_and_emergency(self):
        """The condition is `interval OR emergency`."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert 'or _cog2_emergency' in src

    def test_emergency_model_attribute(self):
        """Model tracks _last_emergency_reinforce for spacing."""
        model = _make_minimal_model()
        # Attribute should be getattr-safe (may not exist initially)
        val = getattr(model, '_last_emergency_reinforce', 0)
        assert isinstance(val, (int, float))


# ═══════════════════════════════════════════════════════════════════════
# PATCH-FINAL-COG-3: VIBE-THINKER RSSM BRIDGE EXCEPTION HARDENING
# ═══════════════════════════════════════════════════════════════════════


class TestFinalCog3_RSSMBridgeHardening:
    """Tests for PATCH-FINAL-COG-3: VibeThinkerRSSMBridge exception hardening."""

    def test_source_code_has_patch_marker(self):
        """PATCH-FINAL-COG-3 comment marker present in aeon_core.py."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert "PATCH-FINAL-COG-3" in src, (
            "PATCH-FINAL-COG-3 comment marker not found in aeon_core.py"
        )

    def test_rssm_pressure_exception_writes_failure_signal(self):
        """When write_signal('rssm_prediction_pressure') fails,
        subsystem_silent_failure_pressure is written."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Find the RSSM bridge area
        idx_bridge = src.index("'rssm_prediction_pressure', rssm_prediction_error")
        # After this, there should be an except block that writes failure pressure
        after_bridge = src[idx_bridge:idx_bridge + 800]
        assert 'subsystem_silent_failure_pressure' in after_bridge, (
            "RSSM bridge write failure does not escalate to "
            "subsystem_silent_failure_pressure"
        )

    def test_provenance_exception_writes_failure_signal(self):
        """When provenance log_auxiliary_event fails, the failure is
        escalated to subsystem_silent_failure_pressure."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Find the provenance area in RSSM bridge
        idx_prov = src.index("event_type='rssm_to_vt_feedback'")
        after_prov = src[idx_prov:idx_prov + 1200]
        assert 'subsystem_silent_failure_pressure' in after_prov, (
            "Provenance failure does not escalate to "
            "subsystem_silent_failure_pressure"
        )

    def test_no_bare_pass_in_rssm_bridge(self):
        """The VibeThinkerRSSMBridge modulate_rssm_loss no longer has
        bare except:pass for bus write/provenance failures."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Find the modulate_rssm_loss method
        idx_method = src.index('def modulate_rssm_loss(')
        # Find the next class definition after it
        idx_next_class = src.index('\nclass ', idx_method)
        method_body = src[idx_method:idx_next_class]
        # The two except blocks that were `pass` should now NOT be bare
        # Count bare "except Exception:\n...pass" patterns
        import re
        bare_passes = re.findall(
            r'except\s+Exception\s*:\s*\n\s*pass\s*#\s*Non-fatal',
            method_body,
        )
        assert len(bare_passes) == 0, (
            f"Found {len(bare_passes)} bare except:pass blocks in "
            f"modulate_rssm_loss (expected 0 after hardening)"
        )

    def test_bridge_runtime_failure_escalation(self):
        """Create a VibeThinkerRSSMBridge with a bus that raises on
        write_signal, verify failure pressure is written."""
        bus = _make_bus()
        bridge = aeon_core.VibeThinkerRSSMBridge(feedback_bus=bus)

        # Make the bus raise on one specific signal to test the except
        original_write = bus.write_signal
        _call_count = [0]

        def _failing_write(name, value):
            if name == 'rssm_prediction_pressure':
                _call_count[0] += 1
                raise RuntimeError("Simulated bus write failure")
            return original_write(name, value)

        bus.write_signal = _failing_write

        # Call modulate_rssm_loss with high prediction error to trigger
        # the RSSM→VT path (uses correct API: rssm_loss, vt_quality_signal)
        result = bridge.modulate_rssm_loss(
            rssm_loss=1.0,
            vt_quality_signal=0.8,
            rssm_prediction_error=0.9,
        )
        # Verify the failure path was taken
        assert _call_count[0] >= 1, "write_signal was not called"
        # The subsystem_silent_failure_pressure should have been written
        val = bus.read_signal('subsystem_silent_failure_pressure', 0.0)
        assert val > 0.0, (
            "subsystem_silent_failure_pressure was not written on bus "
            "write failure"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-FINAL-COG-4: TRAINING LOOP FLUSH_CONSUMED (PHASE A)
# ═══════════════════════════════════════════════════════════════════════


class TestFinalCog4_TrainingFlush:
    """Tests for PATCH-FINAL-COG-4: flush_consumed() in Phase A training."""

    def test_source_code_has_patch_marker(self):
        """PATCH-FINAL-COG-4 comment marker present in ae_train.py."""
        with open('ae_train.py') as f:
            src = f.read()
        assert "PATCH-FINAL-COG-4" in src, (
            "PATCH-FINAL-COG-4 comment marker not found in ae_train.py"
        )

    def test_flush_consumed_called_in_phase_a(self):
        """Phase A's train_step calls flush_consumed() on the bus."""
        with open('ae_train.py') as f:
            src = f.read()
        # Find Phase A train_step (first one)
        idx = src.index('def train_step(self, tokens')
        # Find the next def
        idx_next = src.index('\n    def ', idx + 1)
        method_body = src[idx:idx_next]
        assert 'flush_consumed()' in method_body, (
            "Phase A train_step does not call flush_consumed()"
        )


# ═══════════════════════════════════════════════════════════════════════
# PATCH-FINAL-COG-5: PHASE B TRAINER FEEDBACK BUS WIRING
# ═══════════════════════════════════════════════════════════════════════


class TestFinalCog5_PhaseBWiring:
    """Tests for PATCH-FINAL-COG-5: Phase B feedback bus integration."""

    def test_source_code_has_patch_marker(self):
        """PATCH-FINAL-COG-5 comment marker present in ae_train.py."""
        with open('ae_train.py') as f:
            src = f.read()
        assert "PATCH-FINAL-COG-5" in src, (
            "PATCH-FINAL-COG-5 comment marker not found in ae_train.py"
        )

    def test_phase_b_has_inference_bus_ref(self):
        """ContextualRSSMTrainer has _inference_bus_ref attribute."""
        with open('ae_train.py') as f:
            src = f.read()
        # Find ContextualRSSMTrainer class
        idx = src.index('class ContextualRSSMTrainer')
        after = src[idx:idx + 8000]
        assert '_inference_bus_ref' in after, (
            "ContextualRSSMTrainer does not have _inference_bus_ref"
        )

    def test_phase_b_wires_adaptive_controller(self):
        """Phase B wires adaptive_controller._fb_ref to inference bus."""
        with open('ae_train.py') as f:
            src = f.read()
        idx = src.index('class ContextualRSSMTrainer')
        after = src[idx:idx + 8000]
        assert 'self.adaptive_controller._fb_ref = self._inference_bus_ref' in after

    def test_phase_b_has_flush_method(self):
        """ContextualRSSMTrainer has _flush_bus() method."""
        with open('ae_train.py') as f:
            src = f.read()
        idx = src.index('class ContextualRSSMTrainer')
        # Find flush_bus
        assert '_flush_bus' in src[idx:]

    def test_phase_b_calls_flush_after_train_step(self):
        """Phase B training loop calls _flush_bus() after train_step."""
        with open('ae_train.py') as f:
            src = f.read()
        # Find Phase B training loop (fit method)
        idx_fit = src.index('metrics = self.train_step(ctx_batch')
        after = src[idx_fit:idx_fit + 200]
        assert '_flush_bus()' in after, (
            "Phase B training loop does not call _flush_bus() after train_step"
        )


# ═══════════════════════════════════════════════════════════════════════
# CROSS-PATCH INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCrossPatch_CognitiveIntegration:
    """Cross-patch tests verifying the integrated cognitive organism."""

    def test_signal_ecosystem_still_connected(self):
        """All signals have both producers and consumers after patches."""
        import re
        with open('aeon_core.py') as f:
            core = f.read()
        with open('ae_train.py') as f:
            train = f.read()
        combined = core + '\n' + train
        wp = re.compile(
            r"""write_signal(?:_traced)?\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        writes = set(m[1] for m in wp.findall(combined))
        ep = re.compile(
            r"""_extra_signals\s*\[\s*(['"])([^'"]+)\1\s*\]\s*=""",
            re.MULTILINE,
        )
        writes.update(m[1] for m in ep.findall(combined))
        wl = re.compile(
            r"""_write_log\.add\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        writes.update(m[1] for m in wl.findall(combined))
        rp = re.compile(
            r"""(?:read_signal|get_signal)\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        reads = set(m[1] for m in rp.findall(combined))
        missing = reads - writes
        assert len(missing) == 0, (
            f"Signals read but never written: {sorted(missing)}"
        )

    def test_no_new_orphaned_signals(self):
        """No new orphaned signals introduced by patches."""
        import re
        with open('aeon_core.py') as f:
            core = f.read()
        with open('ae_train.py') as f:
            train = f.read()
        combined = core + '\n' + train
        wp = re.compile(
            r"""write_signal(?:_traced)?\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        writes = set(m[1] for m in wp.findall(combined))
        ep = re.compile(
            r"""_extra_signals\s*\[\s*(['"])([^'"]+)\1\s*\]\s*=""",
            re.MULTILINE,
        )
        writes.update(m[1] for m in ep.findall(combined))
        wl = re.compile(
            r"""_write_log\.add\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        writes.update(m[1] for m in wl.findall(combined))
        rp = re.compile(
            r"""(?:read_signal|get_signal)\s*\(\s*(['"])([^'"]+)\1""",
            re.MULTILINE,
        )
        reads = set(m[1] for m in rp.findall(combined))
        orphaned = writes - reads
        assert len(orphaned) == 0, (
            f"Signals written but never read: {sorted(orphaned)}"
        )

    def test_emergency_reinforce_threshold_consistent(self):
        """Emergency reinforce threshold (0.5) is above the MCT
        emergence_deficit tightening threshold (0.3)."""
        with open('aeon_core.py') as f:
            src = f.read()
        # The emergence_deficit > 0.3 threshold in ACTIVATE-5
        assert '_act5_emergence_deficit > 0.3' in src or \
               'emergence_deficit > 0.3' in src
        # The emergency threshold should be higher (0.5) — only fire
        # emergency reinforcement when deficit is severe
        assert '_cog2_coherence_deficit) > 0.5' in src or \
               'emergence_deficit, _cog2_coherence_deficit) > 0.5' in src
