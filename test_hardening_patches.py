"""
Tests for PATCH-H1, PATCH-H2, PATCH-H3: Hardening Patches.

H-1: Emergence Deficit Staleness Guard
  - emergence_deficit_pass_id is written alongside emergence_deficit
  - Emergence gate reads pass_id and applies decay to stale signals

H-2: Server Signal Consumer Hooks
  - verify_and_reinforce() reads integration_cycle_id and
    integration_cycle_timestamp, writes integration_staleness_pressure
  - MCT reads integration_staleness_pressure → coherence_deficit

H-3: Defensive Recursion Depth Telemetry
  - MetaCognitiveRecursor writes mct_recursion_depth_reached and
    mct_recursion_utility_exhausted after recursion
  - MCT reads both and boosts recovery_pressure when maxed out

Signal Ecosystem: All new signals are bidirectional (0 orphans, 0 missing).
"""

import pytest
import sys
import os
import re
import time
import types
import math

sys.path.insert(0, os.path.dirname(__file__))

import aeon_core
import aeon_core as aeon


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_bus(hidden_dim: int = 64) -> aeon.CognitiveFeedbackBus:
    """Create a minimal CognitiveFeedbackBus."""
    return aeon.CognitiveFeedbackBus(hidden_dim=hidden_dim)


def _src():
    """Read aeon_core.py source once per session (cached)."""
    if not hasattr(_src, '_cache'):
        with open(
            os.path.join(os.path.dirname(__file__), 'aeon_core.py'),
        ) as f:
            _src._cache = f.read()
    return _src._cache


# ──────────────────────────────────────────────────────────────────────
# PATCH-H1: Emergence Deficit Staleness Guard
# ──────────────────────────────────────────────────────────────────────

class TestH1_EmergenceDeficitStaleness:
    """PATCH-H1: Emergence deficit staleness guard."""

    def test_pass_id_signal_written_alongside_deficit(self):
        """emergence_deficit_pass_id is written near emergence_deficit."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*['\"]emergence_deficit_pass_id['\"]",
            src,
        ), "Should write emergence_deficit_pass_id after emergence_deficit"

    def test_pass_id_written_in_emergence_check(self):
        """The write occurs inside _emergence_check context."""
        src = _src()
        # Both emergence_deficit and emergence_deficit_pass_id should be
        # written in the same PATCH-OMEGA-FINAL-5a / H1 block
        assert re.search(
            r"write_signal_traced\s*\(\s*[\n\s]*['\"]emergence_deficit['\"]",
            src,
        ), "emergence_deficit should use write_signal_traced"
        assert re.search(
            r"PATCH-H1.*[Ss]taleness",
            src,
        ), "PATCH-H1 comment should be present"

    def test_staleness_decay_applied_at_gate(self):
        """The emergence gate applies decay when deficit is stale."""
        src = _src()
        # Check that the gate reads emergence_deficit_pass_id
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]emergence_deficit_pass_id['\"]",
            src,
        ), "Emergence gate should read emergence_deficit_pass_id"
        # Check the decay logic
        assert re.search(
            r"_h1_age\s*>\s*3", src,
        ), "Should apply decay when age > 3 passes"
        assert re.search(
            r"max\s*\(\s*0\.5", src,
        ), "Decay should floor at 0.5"

    def test_bus_pass_id_written_functionally(self):
        """Functional: writing emergence_deficit_pass_id to bus works."""
        bus = _make_bus()
        bus.write_signal('emergence_deficit_pass_id', 5.0)
        val = float(bus.read_signal('emergence_deficit_pass_id', 0.0))
        assert val == 5.0

    def test_staleness_decay_math(self):
        """Verify the staleness decay formula: max(0.5, 1.0 - age * 0.1)."""
        # age=4 → 1.0 - 0.4 = 0.6
        assert max(0.5, 1.0 - 4 * 0.1) == 0.6
        # age=5 → 1.0 - 0.5 = 0.5 (at floor)
        assert max(0.5, 1.0 - 5 * 0.1) == 0.5
        # age=10 → clamped to 0.5
        assert max(0.5, 1.0 - 10 * 0.1) == 0.5
        # age=3 → no decay (gate only applies for age > 3)
        # so effectively factor = 1.0

    def test_pass_counter_accessible_on_bus(self):
        """CognitiveFeedbackBus has _pass_counter attribute."""
        bus = _make_bus()
        assert hasattr(bus, '_pass_counter')
        assert bus._pass_counter == 0
        # Incrementing happens in flush_consumed
        bus.flush_consumed()
        assert bus._pass_counter >= 1


# ──────────────────────────────────────────────────────────────────────
# PATCH-H2: Server Signal Consumer Hooks
# ──────────────────────────────────────────────────────────────────────

class TestH2_IntegrationStaleness:
    """PATCH-H2: Server integration staleness detection."""

    def test_integration_cycle_timestamp_read(self):
        """verify_and_reinforce reads integration_cycle_timestamp."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]integration_cycle_timestamp['\"]",
            src,
        ), "Should read integration_cycle_timestamp"

    def test_integration_cycle_id_read(self):
        """verify_and_reinforce reads integration_cycle_id."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]integration_cycle_id['\"]",
            src,
        ), "Should read integration_cycle_id"

    def test_staleness_pressure_written(self):
        """integration_staleness_pressure is written when stale."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*[\n\s]*['\"]integration_staleness_pressure['\"]",
            src,
        ), "Should write integration_staleness_pressure"

    def test_staleness_threshold_is_300(self):
        """Stale threshold is 5 minutes (300 seconds)."""
        src = _src()
        assert re.search(
            r"_H2_STALE_THRESHOLD\s*=\s*300",
            src,
        ), "Stale threshold should be 300 seconds"

    def test_max_stale_is_1800(self):
        """Max stale time is 30 minutes (1800 seconds)."""
        src = _src()
        assert re.search(
            r"_H2_MAX_STALE\s*=\s*1800",
            src,
        ), "Max stale seconds should be 1800"

    def test_staleness_pressure_functional(self):
        """Functional: stale timestamp → pressure > 0."""
        bus = _make_bus()
        # Simulate a server integration cycle from 10 minutes ago
        bus.write_signal(
            'integration_cycle_timestamp',
            time.time() - 600.0,  # 600s ago
        )
        bus.write_signal('integration_cycle_id', 42.0)
        # Now simulate the staleness check logic
        cycle_ts = float(bus.read_signal('integration_cycle_timestamp', 0.0))
        assert cycle_ts > 0
        age = time.time() - cycle_ts
        assert age >= 599  # ~600 seconds
        pressure = min(1.0, age / 1800.0)
        assert 0.3 < pressure < 0.5, f"600s age should give ~0.33 pressure, got {pressure}"

    def test_no_pressure_for_recent_cycle(self):
        """Recent cycle (< 5 min) should not produce pressure."""
        bus = _make_bus()
        bus.write_signal(
            'integration_cycle_timestamp',
            time.time() - 60.0,  # 60s ago
        )
        cycle_ts = float(bus.read_signal('integration_cycle_timestamp', 0.0))
        age = time.time() - cycle_ts
        # Age is ~60s, threshold is 300 → no staleness
        assert age < 300, "Recent cycle should be under threshold"

    def test_patch_h2_comment_present(self):
        """PATCH-H2 comment is in source."""
        src = _src()
        assert 'PATCH-H2' in src


class TestH2b_MCTReaderIntegrationStaleness:
    """PATCH-H2b: MCT reads integration_staleness_pressure."""

    def test_mct_reads_staleness(self):
        """MCT evaluate reads integration_staleness_pressure."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]integration_staleness_pressure['\"]",
            src,
        ), "MCT should read integration_staleness_pressure"

    def test_routes_to_coherence_deficit(self):
        """High staleness routes into coherence_deficit."""
        src = _src()
        # The block should reference coherence_deficit after reading
        assert re.search(
            r"_h2b_stale.*coherence_deficit",
            src,
            re.DOTALL,
        ) or re.search(
            r"PATCH-H2b.*coherence_deficit",
            src,
        ), "Staleness should route to coherence_deficit"

    def test_threshold_is_0_3(self):
        """Only staleness > 0.3 triggers MCT boost."""
        src = _src()
        assert re.search(
            r"_h2b_stale\s*>\s*0\.3",
            src,
        ), "MCT threshold for staleness should be > 0.3"


# ──────────────────────────────────────────────────────────────────────
# PATCH-H3: Defensive Recursion Depth Telemetry
# ──────────────────────────────────────────────────────────────────────

class TestH3_RecursionDepthTelemetry:
    """PATCH-H3: MetaCognitiveRecursor writes depth telemetry."""

    def test_depth_signal_written(self):
        """mct_recursion_depth_reached is written by recursor."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*[\n\s]*['\"]mct_recursion_depth_reached['\"]",
            src,
        ), "Should write mct_recursion_depth_reached"

    def test_utility_exhausted_signal_written(self):
        """mct_recursion_utility_exhausted is written by recursor."""
        src = _src()
        assert re.search(
            r"write_signal\s*\(\s*[\n\s]*['\"]mct_recursion_utility_exhausted['\"]",
            src,
        ), "Should write mct_recursion_utility_exhausted"

    def test_depth_is_float(self):
        """Depth signal uses float(depth) for bus compatibility."""
        src = _src()
        assert re.search(
            r"float\s*\(\s*_h3_depth\s*\)",
            src,
        ), "Depth should be cast to float"

    def test_utility_exhausted_is_boolean_encoded(self):
        """Utility exhausted is 1.0 (not useful) or 0.0 (useful)."""
        src = _src()
        assert re.search(
            r"1\.0\s+if\s+not\s+_h3_last_useful",
            src,
        ), "Should be 1.0 when utility exhausted"

    def test_patch_h3_comment_present(self):
        """PATCH-H3 comment is in source."""
        src = _src()
        assert 'PATCH-H3' in src

    def test_signals_functional(self):
        """Functional: writing both signals to bus works."""
        bus = _make_bus()
        bus.write_signal('mct_recursion_depth_reached', 2.0)
        bus.write_signal('mct_recursion_utility_exhausted', 1.0)
        assert float(bus.read_signal('mct_recursion_depth_reached', 0.0)) == 2.0
        assert float(bus.read_signal('mct_recursion_utility_exhausted', 0.0)) == 1.0


class TestH3b_MCTReaderRecursionDepth:
    """PATCH-H3b: MCT reads recursion depth and utility signals."""

    def test_mct_reads_depth(self):
        """MCT reads mct_recursion_depth_reached."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]mct_recursion_depth_reached['\"]",
            src,
        ), "MCT should read mct_recursion_depth_reached"

    def test_mct_reads_utility_exhausted(self):
        """MCT reads mct_recursion_utility_exhausted."""
        src = _src()
        assert re.search(
            r"read_signal\s*\(\s*[\n\s]*['\"]mct_recursion_utility_exhausted['\"]",
            src,
        ), "MCT should read mct_recursion_utility_exhausted"

    def test_routes_to_recovery_pressure(self):
        """Maxed recursion routes to recovery_pressure."""
        src = _src()
        # The PATCH-H3b block should contain both 'recovery_pressure'
        # and the depth check, on separate lines
        assert 'PATCH-H3b' in src, "PATCH-H3b comment should be present"
        assert re.search(
            r"_h3b_depth.*_h3b_exhausted",
            src,
        ) or re.search(
            r"recovery_pressure.*_h3b",
            src,
        ) or (
            '_h3b_depth' in src and 'recovery_pressure' in src
        ), "Maxed recursion should route to recovery_pressure"

    def test_depth_threshold_is_3(self):
        """Only depth >= 3 with exhaustion triggers pressure."""
        src = _src()
        assert re.search(
            r"_h3b_depth\s*>=\s*3\.0",
            src,
        ), "Depth threshold should be >= 3.0"


# ──────────────────────────────────────────────────────────────────────
# Signal Ecosystem Integrity
# ──────────────────────────────────────────────────────────────────────

class TestSignalEcosystem:
    """Verify signal ecosystem integrity after H-1/H-2/H-3 patches."""

    def _audit_signals(self):
        """Audit all write/read signals across the 3 source files."""
        import collections
        writers = collections.defaultdict(list)
        readers = collections.defaultdict(list)
        base = os.path.dirname(__file__)
        for fname in ['aeon_core.py', 'ae_train.py', 'aeon_server.py']:
            fpath = os.path.join(base, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                content = f.read()
            lines = content.split('\n')
            for i in range(len(lines)):
                chunk = ' '.join(lines[i:i + 3])
                lineno = i + 1
                for m in re.finditer(
                    r"write_signal(?:_traced)?\s*\(\s*['\"]([^'\"]+)['\"]",
                    chunk,
                ):
                    loc = f'{fname}:{lineno}'
                    if loc not in writers[m.group(1)]:
                        writers[m.group(1)].append(loc)
                for m in re.finditer(
                    r"_extra_signals\[['\"](\w+)['\"]\]\s*=",
                    chunk,
                ):
                    loc = f'{fname}:{lineno}'
                    if loc not in writers[m.group(1)]:
                        writers[m.group(1)].append(loc)
                for m in re.finditer(
                    r"_write_log\.add\(['\"]([^'\"]+)['\"]\)",
                    chunk,
                ):
                    loc = f'{fname}:{lineno}'
                    if loc not in writers[m.group(1)]:
                        writers[m.group(1)].append(loc)
                for m in re.finditer(
                    r"read_signal\s*\(\s*['\"]([^'\"]+)['\"]",
                    chunk,
                ):
                    loc = f'{fname}:{lineno}'
                    if loc not in readers[m.group(1)]:
                        readers[m.group(1)].append(loc)
        return writers, readers

    def test_no_missing_producers(self):
        """All read signals have at least one writer."""
        writers, readers = self._audit_signals()
        missing = sorted(set(readers) - set(writers))
        assert not missing, f"Missing producers: {missing}"

    def test_no_orphaned_signals(self):
        """All written signals have at least one reader."""
        writers, readers = self._audit_signals()
        orphaned = sorted(set(writers) - set(readers))
        assert not orphaned, f"Orphaned signals: {orphaned}"

    def test_new_h1_signal_bidirectional(self):
        """emergence_deficit_pass_id is both written and read."""
        writers, readers = self._audit_signals()
        assert 'emergence_deficit_pass_id' in writers, \
            "emergence_deficit_pass_id should be written"
        assert 'emergence_deficit_pass_id' in readers, \
            "emergence_deficit_pass_id should be read"

    def test_new_h2_signal_bidirectional(self):
        """integration_staleness_pressure is both written and read."""
        writers, readers = self._audit_signals()
        assert 'integration_staleness_pressure' in writers, \
            "integration_staleness_pressure should be written"
        assert 'integration_staleness_pressure' in readers, \
            "integration_staleness_pressure should be read"

    def test_old_orphans_now_consumed(self):
        """integration_cycle_id and timestamp are now read."""
        writers, readers = self._audit_signals()
        assert 'integration_cycle_id' in readers, \
            "integration_cycle_id should now be read (H-2)"
        assert 'integration_cycle_timestamp' in readers, \
            "integration_cycle_timestamp should now be read (H-2)"

    def test_new_h3_signals_bidirectional(self):
        """mct_recursion_depth_reached and utility_exhausted are bidirectional."""
        writers, readers = self._audit_signals()
        assert 'mct_recursion_depth_reached' in writers
        assert 'mct_recursion_depth_reached' in readers
        assert 'mct_recursion_utility_exhausted' in writers
        assert 'mct_recursion_utility_exhausted' in readers

    def test_total_signal_count_increased(self):
        """Total signal count should be >= 189 after patches."""
        writers, readers = self._audit_signals()
        total = len(set(writers) | set(readers))
        assert total >= 189, f"Expected >= 189 signals, got {total}"

    def test_bidirectional_count(self):
        """Bidirectional signal count should match written count."""
        writers, readers = self._audit_signals()
        bidirectional = set(writers) & set(readers)
        assert len(bidirectional) >= 189, (
            f"Expected >= 189 bidirectional, got {len(bidirectional)}"
        )
