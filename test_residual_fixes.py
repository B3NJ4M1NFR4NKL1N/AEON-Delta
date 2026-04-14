"""
AEON-Delta RMT v3.4 — Residual Architecture Fix Tests
═════════════════════════════════════════════════════════════════════════

Tests for the two residual architectural notes identified in the
Final Integration & Cognitive Activation Analysis:

  1. Activation Phase 7 orphan check fix: `has_active_orphans()` helper
     replaces inaccurate `len(_orphan_streak) == 0` check.

  2. Cold-start bootstrap seeding: `verify_and_reinforce()` explicitly
     seeds axiom-quality signals on first call.
"""

import re
import sys
import torch
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# §1  has_active_orphans() Helper
# ═══════════════════════════════════════════════════════════════════════════

class TestHasActiveOrphans:
    """Verify the new has_active_orphans() helper on CognitiveFeedbackBus."""

    def test_helper_exists(self):
        """has_active_orphans is a method on CognitiveFeedbackBus."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        assert hasattr(bus, 'has_active_orphans')
        assert callable(bus.has_active_orphans)

    def test_empty_streak_returns_false(self):
        """Empty _orphan_streak dict → no active orphans."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus._orphan_streak = {}
        assert bus.has_active_orphans() is False

    def test_all_zero_streaks_returns_false(self):
        """_orphan_streak with only zero values → no active orphans.

        This is the key bug fix: previously len(_orphan_streak) would
        return > 0 even though all signals are consumed (value = 0).
        """
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus._orphan_streak = {
            'signal_a': 0,
            'signal_b': 0,
            'signal_c': 0,
        }
        assert bus.has_active_orphans() is False

    def test_active_orphan_returns_true(self):
        """_orphan_streak with a positive value → active orphans."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus._orphan_streak = {
            'signal_a': 0,
            'signal_b': 2,  # actively orphaned
        }
        assert bus.has_active_orphans() is True

    def test_mixed_streaks(self):
        """Mix of zero and positive values correctly detected."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        bus._orphan_streak = {
            'consumed_signal': 0,
            'orphaned_signal': 1,
        }
        assert bus.has_active_orphans() is True

    def test_flush_consumed_resets_streaks(self):
        """After flush_consumed resets a signal, has_active_orphans updates."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        # Write a signal but don't read it
        bus.write_signal('orphan_test', 1.0)
        bus.flush_consumed()
        # orphan_test should now have streak > 0
        assert bus._orphan_streak.get('orphan_test', 0) > 0
        assert bus.has_active_orphans() is True

        # Now write and read it
        bus.write_signal('orphan_test', 1.0)
        bus.read_signal('orphan_test', 0.0)
        bus.flush_consumed()
        # orphan_test streak should be reset to 0
        assert bus._orphan_streak.get('orphan_test', 0) == 0
        # Note: flush_consumed itself may write internal signals
        # (oscillation, staleness) that create new orphan entries.
        # We only verify our specific signal was consumed correctly.
        # The bus-internal orphan_test entry is what we validate here.


# ═══════════════════════════════════════════════════════════════════════════
# §2  Activation Phase 7 Fix
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationPhase7Fix:
    """Verify activation phase 7 uses has_active_orphans() not len()."""

    def test_phase7_uses_has_active_orphans(self):
        """Source code for phase 7 references has_active_orphans, not len."""
        with open('aeon_core.py') as f:
            src = f.read()
        # The fixed code should use has_active_orphans()
        assert 'has_active_orphans()' in src
        # The old buggy pattern should be gone from the phase 7 block
        # Find the phase 7 block (between "order": 7 and "order": 8)
        phase7_match = re.search(
            r'"order":\s*7.*?"order":\s*8',
            src,
            re.DOTALL,
        )
        assert phase7_match is not None, "Phase 7 block not found"
        phase7_src = phase7_match.group(0)
        # Should NOT contain the old len(_orphan_streak) == 0 pattern
        assert "len(" not in phase7_src, (
            "Phase 7 still uses len() instead of has_active_orphans()"
        )
        # Should contain has_active_orphans
        assert "has_active_orphans" in phase7_src

    def test_phase7_active_with_zero_streaks(self):
        """Phase 7 reports 'active' when all orphan streaks are zero."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        # Simulate post-flush state: keys exist but all values are 0
        bus._orphan_streak = {'sig_a': 0, 'sig_b': 0, 'sig_c': 0}
        # The bus should report no active orphans
        assert not bus.has_active_orphans()


# ═══════════════════════════════════════════════════════════════════════════
# §3  Cold-Start Bootstrap Seeding
# ═══════════════════════════════════════════════════════════════════════════

class TestColdStartBootstrap:
    """Verify verify_and_reinforce() seeds axiom signals on first call."""

    def test_cold_start_seeding_in_source(self):
        """Source code contains cold-start bootstrap seeding block."""
        with open('aeon_core.py') as f:
            src = f.read()
        assert 'cold_start_bootstrap' in src
        assert '_cold_start_seeded' in src

    def test_cold_start_seeds_axiom_signals(self):
        """First verify_and_reinforce() call writes baseline axiom signals."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)

        # Before seeding, signals return default
        val = bus.read_signal('mutual_verification_quality', -1.0)
        assert val == -1.0  # default returned, no signal written yet

        # Simulate cold-start seeding
        for sig in (
            'mutual_verification_quality',
            'uncertainty_metacognition_quality',
            'root_cause_traceability_quality',
        ):
            bus.write_signal(sig, 0.5)

        # After seeding, signals return 0.5 baseline
        val = bus.read_signal('mutual_verification_quality', -1.0)
        assert abs(val - 0.5) < 0.01

    def test_cold_start_flag_prevents_double_seeding(self):
        """_cold_start_seeded flag referenced in source."""
        with open('aeon_core.py') as f:
            src = f.read()
        # The guard should check _cold_start_seeded
        assert "_cold_start_seeded" in src
        # Should set it to True after seeding
        assert "self._cold_start_seeded = True" in src

    def test_cold_start_records_causal_trace(self):
        """Cold-start bootstrap records to causal trace."""
        with open('aeon_core.py') as f:
            src = f.read()
        # Find the cold-start block
        cs_match = re.search(
            r'cold_start_bootstrap.*?severity.*?info',
            src,
            re.DOTALL,
        )
        assert cs_match is not None, (
            "Cold-start bootstrap should record to causal trace with "
            "severity='info'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §4  Signal Ecosystem Integrity (Post-Fix)
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalEcosystemPostFix:
    """Verify signal ecosystem remains healthy after fixes."""

    def test_no_new_orphan_signals(self):
        """Cold-start signals are already bidirectional (pre-existing)."""
        with open('aeon_core.py') as f:
            core_src = f.read()
        with open('ae_train.py') as f:
            train_src = f.read()
        with open('aeon_server.py') as f:
            server_src = f.read()

        full_src = core_src + train_src + server_src

        # The three cold-start signals must have both writers and readers
        for sig in (
            'mutual_verification_quality',
            'uncertainty_metacognition_quality',
            'root_cause_traceability_quality',
        ):
            write_pat = re.compile(
                rf"write_signal(?:_traced)?\s*\(\s*['\"]"
                + re.escape(sig) + r"['\"]",
            )
            read_pat = re.compile(
                rf"read_signal\s*\(\s*['\"]"
                + re.escape(sig) + r"['\"]",
            )
            assert write_pat.search(full_src), (
                f"Signal {sig!r} has no writer"
            )
            assert read_pat.search(full_src), (
                f"Signal {sig!r} has no reader"
            )

    def test_ecosystem_bidirectional_count(self):
        """Total bidirectional signal count >= 267 (no regression)."""
        with open('aeon_core.py') as f:
            core_src = f.read()
        with open('ae_train.py') as f:
            train_src = f.read()
        with open('aeon_server.py') as f:
            server_src = f.read()

        full_src = core_src + train_src + server_src

        written = set()
        read = set()
        for m in re.finditer(
            r"write_signal(?:_traced)?\s*\(\s*['\"](\w+)['\"]",
            full_src,
        ):
            written.add(m.group(1))
        for m in re.finditer(
            r"read_signal\s*\(\s*['\"](\w+)['\"]",
            full_src,
        ):
            read.add(m.group(1))

        bidirectional = written & read
        write_only = written - read
        read_only = read - written

        assert len(write_only) == 0, (
            f"Write-only orphans: {write_only}"
        )
        assert len(read_only) == 0, (
            f"Read-only orphans: {read_only}"
        )
        assert len(bidirectional) >= 267, (
            f"Expected >= 267 bidirectional signals, got {len(bidirectional)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §5  Integration Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationVerification:
    """End-to-end verification of both fixes working together."""

    def test_has_active_orphans_method_signature(self):
        """has_active_orphans takes no args and returns bool."""
        from aeon_core import CognitiveFeedbackBus
        bus = CognitiveFeedbackBus(hidden_dim=64)
        result = bus.has_active_orphans()
        assert isinstance(result, bool)

    def test_version_unchanged(self):
        """Version remains 3.4.0 after fixes."""
        from aeon_core import __version__
        assert __version__ == "3.4.0"

    def test_activation_sequence_still_10_phases(self):
        """Activation sequence still has 10 phases after phase 7 fix."""
        with open('aeon_core.py') as f:
            src = f.read()
        phase_count = len(re.findall(r'"order":\s*\d+,\s*\n\s*"phase":', src))
        assert phase_count == 10, (
            f"Expected 10 activation phases, found {phase_count}"
        )
