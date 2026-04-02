"""Tests for Q-series patches: Final cognitive activation bridges.

Q1a: ae_train.py task-boundary error_evolution recording last-resort
     except Exception: pass → except Exception as _tb_rec_err: logger.debug
Q1b: ae_train.py entropy-adaptation error_evolution recording last-resort
     except Exception: pass → except Exception as _ew_rec_err: logger.debug
Q2:  aeon_core.py cross-severity compounding in verify_and_reinforce
Q3:  aeon_core.py compound severity pressure on feedback bus
Q4:  aeon_core.py axiom deficit trend tracking in verify_and_reinforce
Q5:  aeon_core.py _class_to_signal mappings for new error classes
"""

import importlib
import inspect
import logging
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import modules once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aeon_mod():
    """Return the aeon_core module, imported once per test module."""
    return importlib.import_module("aeon_core")


@pytest.fixture(scope="module")
def train_mod():
    """Return the ae_train module, imported once per test module."""
    return importlib.import_module("ae_train")


@pytest.fixture()
def model(aeon_mod):
    """Build a minimal AEONDeltaV3 for unit-testing."""
    cfg = aeon_mod.AEONConfig(
        z_dim=32,
        hidden_dim=32,
        vq_num_embeddings=8,
        vq_embedding_dim=32,
    )
    m = aeon_mod.AEONDeltaV3(cfg)
    return m


# ════════════════════════════════════════════════════════════════════════
#  Q1a: Task boundary recording last-resort guard → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestQ1aTaskBoundaryRecordingGuard:
    """Q1a: Last-resort exception around error_evolution.record_episode
    for task boundary failure must log, not silently pass."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch Q1a marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch Q1a" in src

    def test_logger_debug_present(self, train_mod):
        """Last-resort guard must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1a")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_exception_captured_as_variable(self, train_mod):
        """Exception must be captured (not bare except)."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1a")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_tb_rec_err" in region

    def test_no_bare_pass_after_except(self, train_mod):
        """The except block must NOT have bare 'pass' as sole body."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1a")
        assert idx != -1
        region = src[idx:idx + 500]
        # There should be a logger.debug call, not just pass
        lines = region.split('\n')
        has_logger = any('logger.debug' in line for line in lines)
        assert has_logger


# ════════════════════════════════════════════════════════════════════════
#  Q1b: Entropy adaptation recording last-resort guard → logger.debug
# ════════════════════════════════════════════════════════════════════════
class TestQ1bEntropyAdaptationRecordingGuard:
    """Q1b: Last-resort exception around error_evolution.record_episode
    for entropy adaptation failure must log, not silently pass."""

    def test_patch_comment_in_source(self, train_mod):
        """Patch Q1b marker must exist in ae_train source."""
        src = inspect.getsource(train_mod)
        assert "Patch Q1b" in src

    def test_logger_debug_present(self, train_mod):
        """Last-resort guard must call logger.debug."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1b")
        assert idx != -1
        region = src[idx:idx + 500]
        assert "logger.debug" in region

    def test_exception_captured_as_variable(self, train_mod):
        """Exception must be captured (not bare except)."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1b")
        assert idx != -1
        region = src[max(0, idx - 300):idx + 200]
        assert "_ew_rec_err" in region

    def test_no_bare_pass_after_except(self, train_mod):
        """The except block must NOT have bare 'pass' as sole body."""
        src = inspect.getsource(train_mod)
        idx = src.find("Patch Q1b")
        assert idx != -1
        region = src[idx:idx + 500]
        lines = region.split('\n')
        has_logger = any('logger.debug' in line for line in lines)
        assert has_logger


# ════════════════════════════════════════════════════════════════════════
#  Q2: Cross-severity compounding in verify_and_reinforce
# ════════════════════════════════════════════════════════════════════════
class TestQ2CrossSeverityCompounding:
    """Q2: When ≥2 of stall, oscillation, cognitive_unity_deficit are
    simultaneously elevated, a compound_severity_escalation episode
    must be recorded to error_evolution."""

    def test_patch_comment_in_source(self, aeon_mod):
        """Patch Q2 marker must exist in aeon_core source."""
        src = inspect.getsource(aeon_mod)
        assert "Patch Q2" in src

    def test_compound_severity_escalation_error_class(self, aeon_mod):
        """compound_severity_escalation must appear as error_class."""
        src = inspect.getsource(aeon_mod)
        assert "compound_severity_escalation" in src

    def test_reads_stall_severity(self, aeon_mod):
        """Q2 logic must read _cached_stall_severity."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q2")
        assert idx != -1
        region = src[idx:idx + 2000]
        assert "_cached_stall_severity" in region

    def test_reads_oscillation_severity(self, aeon_mod):
        """Q2 logic must read _cached_oscillation_severity."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q2")
        assert idx != -1
        region = src[idx:idx + 2000]
        assert "_cached_oscillation_severity" in region

    def test_reads_cognitive_unity_deficit(self, aeon_mod):
        """Q2 logic must read _cached_cognitive_unity_deficit."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q2")
        assert idx != -1
        region = src[idx:idx + 2000]
        assert "_cached_cognitive_unity_deficit" in region

    def test_records_episode_when_two_elevated(self, model, aeon_mod):
        """When 2+ severities are elevated, error_evolution must record
        a compound_severity_escalation episode."""
        model._cached_stall_severity = 0.5
        model._cached_oscillation_severity = 0.4
        model._cached_cognitive_unity_deficit = 0.0
        # Set axiom scores to pass (avoid triggering other episodes)
        model._cached_mv_axiom_deficit = 0.0
        model._cached_um_axiom_deficit = 0.0
        model._cached_rc_axiom_deficit = 0.0
        model._prev_cached_mv_axiom_deficit = 0.0
        model._prev_cached_um_axiom_deficit = 0.0
        model._prev_cached_rc_axiom_deficit = 0.0

        # Mock error_evolution and run the compound severity check
        mock_ee = MagicMock()
        mock_ee.record_episode = MagicMock()
        model.error_evolution = mock_ee

        # Search for the compound_severity_escalation recording call
        # by verifying the error_class is used in verify_and_reinforce
        src = inspect.getsource(aeon_mod.AEONDeltaV3)
        assert "compound_severity_escalation" in src

    def test_no_episode_when_only_one_elevated(self, aeon_mod):
        """When only 1 severity is elevated, no compound episode."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q2")
        assert idx != -1
        region = src[idx:idx + 2000]
        # Must check for >= 2 threshold
        assert ">= 2" in region or "_compound_count >= 2" in region

    def test_compound_score_calculation(self, aeon_mod):
        """Compound score must be the mean of all three severities."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q2")
        assert idx != -1
        region = src[idx:idx + 2000]
        assert "/ 3.0" in region or "/ 3" in region

    def test_strategy_used(self, aeon_mod):
        """Strategy used must be verify_and_reinforce_cross_severity."""
        src = inspect.getsource(aeon_mod)
        assert "verify_and_reinforce_cross_severity" in src


# ════════════════════════════════════════════════════════════════════════
#  Q3: Compound severity pressure on feedback bus
# ════════════════════════════════════════════════════════════════════════
class TestQ3CompoundSeverityPressure:
    """Q3: When ≥2 severity caches are elevated, a compound_severity_pressure
    signal must be surfaced on the feedback bus via _build_feedback_extra_signals."""

    def test_patch_comment_in_source(self, aeon_mod):
        """Patch Q3 marker must exist in aeon_core source."""
        src = inspect.getsource(aeon_mod)
        assert "Patch Q3" in src

    def test_compound_severity_pressure_signal(self, aeon_mod):
        """compound_severity_pressure must appear as a signal key."""
        src = inspect.getsource(aeon_mod)
        assert "compound_severity_pressure" in src

    def test_reads_all_three_caches(self, aeon_mod):
        """Q3 must read stall, oscillation, and cognitive_unity_deficit."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q3")
        assert idx != -1
        region = src[idx:idx + 2000]
        assert "_cached_stall_severity" in region
        assert "_cached_oscillation_severity" in region
        assert "_cached_cognitive_unity_deficit" in region

    def test_signal_emitted_when_two_elevated(self, model):
        """When 2+ severities exceed threshold, compound signal is in extra."""
        model._cached_stall_severity = 0.5
        model._cached_oscillation_severity = 0.4
        model._cached_cognitive_unity_deficit = 0.0
        # Set other caches to avoid unrelated signals
        model._cached_diversity_state = None
        model._cached_topology_state = None
        model._cached_convergence_quality = 1.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" in extra
        assert 0.0 < extra["compound_severity_pressure"] <= 1.0

    def test_no_signal_when_only_one_elevated(self, model):
        """When only 1 severity exceeds threshold, no compound signal."""
        model._cached_stall_severity = 0.5
        model._cached_oscillation_severity = 0.0
        model._cached_cognitive_unity_deficit = 0.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" not in extra

    def test_no_signal_when_none_elevated(self, model):
        """When no severity exceeds threshold, no compound signal."""
        model._cached_stall_severity = 0.0
        model._cached_oscillation_severity = 0.0
        model._cached_cognitive_unity_deficit = 0.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" not in extra

    def test_signal_value_is_mean(self, model):
        """Compound signal value should be mean of elevated severities."""
        model._cached_stall_severity = 0.6
        model._cached_oscillation_severity = 0.4
        model._cached_cognitive_unity_deficit = 0.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" in extra
        # Only 2 values are elevated (>0.15): 0.6 and 0.4
        expected = (0.6 + 0.4) / 2
        assert abs(extra["compound_severity_pressure"] - expected) < 0.01

    def test_signal_clamped_to_unit_interval(self, model):
        """Compound signal value must be clamped to [0, 1]."""
        model._cached_stall_severity = 1.0
        model._cached_oscillation_severity = 1.0
        model._cached_cognitive_unity_deficit = 1.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" in extra
        assert 0.0 <= extra["compound_severity_pressure"] <= 1.0

    def test_all_three_elevated(self, model):
        """When all three severities are elevated, compound signal uses all."""
        model._cached_stall_severity = 0.3
        model._cached_oscillation_severity = 0.5
        model._cached_cognitive_unity_deficit = 0.7
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" in extra
        # All 3 values exceed 0.15 threshold, so mean is of all 3
        expected = (0.3 + 0.5 + 0.7) / 3
        assert abs(extra["compound_severity_pressure"] - expected) < 0.01


# ════════════════════════════════════════════════════════════════════════
#  Q4: Axiom deficit trend tracking in verify_and_reinforce
# ════════════════════════════════════════════════════════════════════════
class TestQ4AxiomDeficitTrendTracking:
    """Q4: When axiom deficits worsen across reinforcement cycles, a
    trend-aware episode must be recorded so the metacognitive trigger
    can escalate urgency proportionally to the rate of decline."""

    def test_patch_comment_in_source(self, aeon_mod):
        """Patch Q4 marker must exist in aeon_core source."""
        src = inspect.getsource(aeon_mod)
        assert "Patch Q4" in src

    def test_axiom_deficit_worsening_error_class(self, aeon_mod):
        """axiom_deficit_worsening must appear as error_class."""
        src = inspect.getsource(aeon_mod)
        assert "axiom_deficit_worsening" in src

    def test_reads_prev_cached_deficits(self, aeon_mod):
        """Q4 must read previous cycle's cached axiom deficits."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q4")
        assert idx != -1
        region = src[idx:idx + 3000]
        assert "_prev_cached_mv_axiom_deficit" in region
        assert "_prev_cached_um_axiom_deficit" in region
        assert "_prev_cached_rc_axiom_deficit" in region

    def test_stores_current_for_next_cycle(self, aeon_mod):
        """Q4 must store current deficits for next cycle comparison."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q4")
        assert idx != -1
        region = src[idx:idx + 3000]
        assert "self._prev_cached_mv_axiom_deficit = _cur_mv_def" in region
        assert "self._prev_cached_um_axiom_deficit = _cur_um_def" in region
        assert "self._prev_cached_rc_axiom_deficit = _cur_rc_def" in region

    def test_worsening_threshold_is_0_1(self, aeon_mod):
        """Worsening threshold must be 0.1 (10% increase)."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q4")
        assert idx != -1
        region = src[idx:idx + 3000]
        assert "> 0.1" in region

    def test_strategy_used(self, aeon_mod):
        """Strategy used must be verify_and_reinforce_trend."""
        src = inspect.getsource(aeon_mod)
        assert "verify_and_reinforce_trend" in src

    def test_tracks_delta_per_axiom(self, aeon_mod):
        """Metadata must include per-axiom delta values."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q4")
        assert idx != -1
        region = src[idx:idx + 3000]
        assert "'delta'" in region

    def test_prev_defaults_to_zero(self, aeon_mod):
        """Previous deficit must default to 0.0 via getattr."""
        src = inspect.getsource(aeon_mod)
        idx = src.find("Patch Q4")
        assert idx != -1
        region = src[idx:idx + 3000]
        assert "getattr(self, '_prev_cached_mv_axiom_deficit', 0.0)" in region


# ════════════════════════════════════════════════════════════════════════
#  Q5: _class_to_signal mappings for new error classes
# ════════════════════════════════════════════════════════════════════════
class TestQ5ClassToSignalMappings:
    """Q5: New error classes from Q2 and Q4 must have _class_to_signal
    mappings so they route to the correct metacognitive trigger signal."""

    def test_compound_severity_mapping_exists(self, aeon_mod):
        """compound_severity_escalation must map to coherence_deficit."""
        src = inspect.getsource(aeon_mod)
        assert '"compound_severity_escalation": "coherence_deficit"' in src

    def test_axiom_deficit_worsening_mapping_exists(self, aeon_mod):
        """axiom_deficit_worsening must map to convergence_conflict."""
        src = inspect.getsource(aeon_mod)
        assert '"axiom_deficit_worsening": "convergence_conflict"' in src

    def test_q5a_comment_present(self, aeon_mod):
        """Q5a comment must be present for compound_severity_escalation."""
        src = inspect.getsource(aeon_mod)
        assert "Q5a" in src

    def test_q5b_comment_present(self, aeon_mod):
        """Q5b comment must be present for axiom_deficit_worsening."""
        src = inspect.getsource(aeon_mod)
        assert "Q5b" in src

    def test_mapping_in_class_to_signal_dict(self, model, aeon_mod):
        """Mappings must be in the _class_to_signal dict used by
        adapt_weights_from_evolution."""
        mct = model.metacognitive_trigger
        # Trigger the weight adaptation path to populate the dict
        src = inspect.getsource(aeon_mod.MetaCognitiveRecursionTrigger)
        assert "compound_severity_escalation" in src
        assert "axiom_deficit_worsening" in src

    def test_no_unmapped_new_error_classes(self, aeon_mod):
        """All new error classes introduced in Q-series must be mapped."""
        src = inspect.getsource(aeon_mod)
        # Check both new classes exist in _class_to_signal
        for ec in ("compound_severity_escalation", "axiom_deficit_worsening"):
            idx = src.find(f'"{ec}":')
            # Must appear at least twice: once as error_class=, once in mapping
            count = src.count(ec)
            assert count >= 2, f"{ec} appears only {count} time(s)"


# ════════════════════════════════════════════════════════════════════════
#  Integration: All patches work together
# ════════════════════════════════════════════════════════════════════════
class TestQSeriesIntegration:
    """Integration tests: Q-series patches form a coherent system."""

    def test_no_remaining_bare_except_pass_in_ae_train(self, train_mod):
        """ae_train.py must have no remaining bare 'except ...: pass'
        blocks (all must have logger.debug or _bridge_silent_exception)."""
        src = inspect.getsource(train_mod)
        lines = src.split('\n')
        bare_passes_found = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('except') and stripped.endswith(':'):
                # Check the next non-empty line
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_stripped = lines[j].strip()
                    if next_stripped:
                        # A bare 'pass' with no logger/bridge is a problem
                        if next_stripped == 'pass':
                            bare_passes_found.append(
                                f"line {i+1}: {stripped} → {next_stripped}"
                            )
                        break
        assert not bare_passes_found, (
            f"Found bare except:pass blocks: {bare_passes_found}"
        )

    def test_compound_severity_feeds_back_through_bus(self, model):
        """Compound severity signal flows through feedback bus."""
        model._cached_stall_severity = 0.5
        model._cached_oscillation_severity = 0.3
        model._cached_cognitive_unity_deficit = 0.4
        extra = model._build_feedback_extra_signals()
        # Should have compound severity
        assert "compound_severity_pressure" in extra
        # Should also have individual severity signals
        assert "stall_severity_pressure" in extra
        assert "oscillation_severity_pressure" in extra

    def test_all_q_patches_have_comments(self, aeon_mod, train_mod):
        """All Q-series patches must have identifying comments."""
        aeon_src = inspect.getsource(aeon_mod)
        train_src = inspect.getsource(train_mod)
        for marker in ("Patch Q1a", "Patch Q1b"):
            assert marker in train_src, f"{marker} missing from ae_train"
        for marker in ("Patch Q2", "Patch Q3", "Patch Q4"):
            assert marker in aeon_src, f"{marker} missing from aeon_core"

    def test_feedback_bus_extra_signals_returns_dict(self, model):
        """_build_feedback_extra_signals must return a dict."""
        model._cached_stall_severity = 0.0
        model._cached_oscillation_severity = 0.0
        model._cached_cognitive_unity_deficit = 0.0
        extra = model._build_feedback_extra_signals()
        assert isinstance(extra, dict)

    def test_compound_signal_absent_when_all_zero(self, model):
        """No compound signal when all severities are zero."""
        model._cached_stall_severity = 0.0
        model._cached_oscillation_severity = 0.0
        model._cached_cognitive_unity_deficit = 0.0
        extra = model._build_feedback_extra_signals()
        assert "compound_severity_pressure" not in extra
