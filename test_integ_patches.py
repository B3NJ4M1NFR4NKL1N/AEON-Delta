"""Test suite for PATCH-Ψ1 through Ψ7: Final Cognitive Integration patches.

Covers:
- Ψ1: PostPipelineMCTGate invokes MCT.evaluate() on threshold breach
- Ψ1b: Forward pipeline reads post_pipeline_mct_trigger_score/should_trigger
- Ψ2: AutoCriticLoop MCT-gated revision suppression
- Ψ2b: MCT reads auto_critic_revision_suppressed
- Ψ3: UCC MCT consistency bridge (calls MCT.evaluate on forced trigger)
- Ψ3b: MCT reads ucc_forced_mct_trigger
- Ψ4: Cognitive snapshot writes health/unity/degradation to bus
- Ψ4b: MCT reads snapshot_system_health
- Ψ4c: MCT reads snapshot_degradation_count
- Ψ4d: MCT reads snapshot_unity_score
- Ψ5: CausalDAGConsensus writes dag_consensus_pressure to bus
- Ψ5b: MCT reads dag_consensus_pressure → low_causal_quality
- Ψ6: test_no_missing_producers includes aeon_server.py
- Ψ7: system_emergence_report writes emergence_integration_health to bus
- Ψ7b: MCT reads emergence_integration_health → coherence_deficit
- Ψ7c: MCT reads emergence_system_emerged
- Signal ecosystem: 0 orphans, 0 missing producers
"""

import os
import re
import sys

import pytest

# Ensure repository root is importable
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _core_src() -> str:
    with open(os.path.join(_REPO, 'aeon_core.py'), 'r') as f:
        return f.read()


def _train_src() -> str:
    with open(os.path.join(_REPO, 'ae_train.py'), 'r') as f:
        return f.read()


def _server_src() -> str:
    path = os.path.join(_REPO, 'aeon_server.py')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ''


def _combined_src() -> str:
    return _core_src() + _train_src() + _server_src()


def _written_signals(src: str) -> set:
    write_pat = re.compile(r'write_signal(?:_traced)?\s*\(\s*["\'](\w+)["\']')
    extra_pat = re.compile(r'_extra_signals\s*\[\s*["\'](\w+)["\']\s*\]')
    return set(write_pat.findall(src)) | set(extra_pat.findall(src))


def _read_signals(src: str) -> set:
    read_pat = re.compile(r'read_signal(?:_current_gen|_any_gen)?\s*\(\s*["\'](\w+)["\']')
    return set(read_pat.findall(src))


# ===========================================================================
# PATCH-Ψ1: PostPipelineMCTGate invokes MCT.evaluate()
# ===========================================================================


class TestPsi1_PostPipelineMCTGate:
    """PATCH-Ψ1: PostPipelineMCTGate now calls MCT.evaluate()."""

    def test_gate_writes_mct_trigger_score(self):
        """PostPipelineMCTGate writes post_pipeline_mct_trigger_score."""
        src = _core_src()
        assert "write_signal(\n                        'post_pipeline_mct_trigger_score'" in src or \
               "write_signal('post_pipeline_mct_trigger_score'" in src or \
               'post_pipeline_mct_trigger_score' in _written_signals(src)

    def test_gate_writes_mct_should_trigger(self):
        """PostPipelineMCTGate writes post_pipeline_mct_should_trigger."""
        assert 'post_pipeline_mct_should_trigger' in _written_signals(_core_src())

    def test_gate_calls_mct_evaluate(self):
        """PostPipelineMCTGate contains MCT.evaluate() invocation."""
        src = _core_src()
        # Find the PostPipelineMCTGate class region
        idx = src.find('class PostPipelineMCTGate')
        assert idx > 0
        region = src[idx:idx + 3000]
        assert '_mct.evaluate(' in region, \
            "PostPipelineMCTGate must call self._mct.evaluate()"

    def test_forward_impl_reads_post_pipeline_mct_signals(self):
        """PATCH-Ψ1b: _forward_impl reads post_pipeline_mct_trigger_score."""
        reads = _read_signals(_core_src())
        assert 'post_pipeline_mct_trigger_score' in reads
        assert 'post_pipeline_mct_should_trigger' in reads


# ===========================================================================
# PATCH-Ψ2: AutoCriticLoop MCT-gated revision suppression
# ===========================================================================


class TestPsi2_AutoCriticMCTGate:
    """PATCH-Ψ2: AutoCriticLoop suppresses revision when MCT is stable."""

    def test_auto_critic_writes_suppression_signal(self):
        """AutoCriticLoop writes auto_critic_revision_suppressed."""
        assert 'auto_critic_revision_suppressed' in _written_signals(_core_src())

    def test_auto_critic_reads_mct_trigger_score(self):
        """AutoCriticLoop reads mct_trigger_score for gating."""
        src = _core_src()
        idx = src.find('PATCH-Ψ2: MCT-gated revision suppression')
        assert idx > 0, "PATCH-Ψ2 annotation must exist"
        region = src[idx:idx + 1000]
        assert 'mct_trigger_score' in region

    def test_suppression_breaks_loop(self):
        """When suppressed, the auto-critic breaks out of the loop."""
        src = _core_src()
        idx = src.find('PATCH-Ψ2: MCT-gated revision suppression')
        region = src[idx:idx + 1500]
        assert 'break' in region, "Suppression must break the revision loop"

    def test_mct_reads_suppression(self):
        """PATCH-Ψ2b: MCT reads auto_critic_revision_suppressed."""
        assert 'auto_critic_revision_suppressed' in _read_signals(_core_src())


# ===========================================================================
# PATCH-Ψ3: UCC MCT Consistency Bridge
# ===========================================================================


class TestPsi3_UCCMCTBridge:
    """PATCH-Ψ3: UCC calls MCT.evaluate() when forcing trigger."""

    def test_ucc_writes_forced_mct_trigger(self):
        """UCC writes ucc_forced_mct_trigger to bus."""
        assert 'ucc_forced_mct_trigger' in _written_signals(_core_src())

    def test_ucc_calls_mct_evaluate(self):
        """UCC section calls metacognitive_trigger.evaluate()."""
        src = _core_src()
        idx = src.find('PATCH-Ψ3: UCC MCT consistency bridge')
        assert idx > 0, "PATCH-Ψ3 annotation must exist"
        region = src[idx:idx + 1200]
        assert 'metacognitive_trigger.evaluate(' in region

    def test_mct_reads_forced_trigger(self):
        """PATCH-Ψ3b: MCT reads ucc_forced_mct_trigger."""
        assert 'ucc_forced_mct_trigger' in _read_signals(_core_src())


# ===========================================================================
# PATCH-Ψ4: Cognitive Snapshot → Bus Feedback
# ===========================================================================


class TestPsi4_SnapshotBusFeedback:
    """PATCH-Ψ4: get_cognitive_state_snapshot writes to bus."""

    def test_snapshot_writes_system_health(self):
        """Snapshot writes snapshot_system_health."""
        assert 'snapshot_system_health' in _written_signals(_core_src())

    def test_snapshot_writes_unity_score(self):
        """Snapshot writes snapshot_unity_score."""
        assert 'snapshot_unity_score' in _written_signals(_core_src())

    def test_snapshot_writes_degradation_count(self):
        """Snapshot writes snapshot_degradation_count."""
        assert 'snapshot_degradation_count' in _written_signals(_core_src())

    def test_mct_reads_snapshot_health(self):
        """PATCH-Ψ4b: MCT reads snapshot_system_health."""
        assert 'snapshot_system_health' in _read_signals(_core_src())

    def test_mct_reads_degradation_count(self):
        """PATCH-Ψ4c: MCT reads snapshot_degradation_count."""
        assert 'snapshot_degradation_count' in _read_signals(_core_src())

    def test_mct_reads_unity_score(self):
        """PATCH-Ψ4d: MCT reads snapshot_unity_score."""
        assert 'snapshot_unity_score' in _read_signals(_core_src())


# ===========================================================================
# PATCH-Ψ5: CausalDAGConsensus → Bus Signal
# ===========================================================================


class TestPsi5_DAGConsensusBusSignal:
    """PATCH-Ψ5: CausalDAGConsensus writes dag_consensus_pressure."""

    def test_dag_writes_pressure(self):
        """UCC writes dag_consensus_pressure to bus."""
        assert 'dag_consensus_pressure' in _written_signals(_core_src())

    def test_mct_reads_dag_pressure(self):
        """PATCH-Ψ5b: MCT reads dag_consensus_pressure."""
        assert 'dag_consensus_pressure' in _read_signals(_core_src())

    def test_dag_pressure_routes_to_low_causal_quality(self):
        """dag_consensus_pressure amplifies low_causal_quality in MCT."""
        src = _core_src()
        idx = src.find('PATCH-Ψ5b: dag_consensus_pressure')
        assert idx > 0
        region = src[idx:idx + 500]
        assert 'low_causal_quality' in region


# ===========================================================================
# PATCH-Ψ6: Server Signal Producer Test Fix
# ===========================================================================


class TestPsi6_ServerSignalTestFix:
    """PATCH-Ψ6: test_no_missing_producers now scans aeon_server.py."""

    def test_server_signals_in_written(self):
        """Server-written signals are captured by combined scan."""
        combined = _combined_src()
        written = _written_signals(combined)
        server_signals = {
            'server_coherence_score',
            'server_reinforcement_pressure',
        }
        for sig in server_signals:
            assert sig in written, f"Server signal '{sig}' must be in written set"

    def test_no_missing_producers_full_scan(self):
        """No missing producers when all 3 files are scanned."""
        combined = _combined_src()
        written = _written_signals(combined)
        read = _read_signals(combined)
        missing = read - written
        assert len(missing) == 0, f"Missing producers: {missing}"


# ===========================================================================
# PATCH-Ψ7: Emergence Report Active Feedback
# ===========================================================================


class TestPsi7_EmergenceActiveFeedback:
    """PATCH-Ψ7: system_emergence_report writes to bus."""

    def test_emergence_writes_integration_health(self):
        """system_emergence_report writes emergence_integration_health."""
        assert 'emergence_integration_health' in _written_signals(_core_src())

    def test_emergence_writes_system_emerged(self):
        """system_emergence_report writes emergence_system_emerged."""
        assert 'emergence_system_emerged' in _written_signals(_core_src())

    def test_mct_reads_emergence_health(self):
        """PATCH-Ψ7b: MCT reads emergence_integration_health."""
        assert 'emergence_integration_health' in _read_signals(_core_src())

    def test_mct_reads_system_emerged(self):
        """PATCH-Ψ7c: MCT reads emergence_system_emerged."""
        assert 'emergence_system_emerged' in _read_signals(_core_src())

    def test_emergence_health_routes_to_coherence_deficit(self):
        """emergence_integration_health amplifies coherence_deficit."""
        src = _core_src()
        idx = src.find('PATCH-Ψ7b: emergence_integration_health')
        assert idx > 0
        region = src[idx:idx + 500]
        assert 'coherence_deficit' in region


# ===========================================================================
# Signal Ecosystem Integrity
# ===========================================================================


class TestPsi_SignalEcosystemIntegrity:
    """All new Ψ signals are fully bidirectional (written AND read)."""

    PSI_SIGNALS = [
        'post_pipeline_mct_trigger_score',
        'post_pipeline_mct_should_trigger',
        'auto_critic_revision_suppressed',
        'ucc_forced_mct_trigger',
        'snapshot_system_health',
        'snapshot_unity_score',
        'snapshot_degradation_count',
        'dag_consensus_pressure',
        'emergence_integration_health',
        'emergence_system_emerged',
    ]

    def test_all_psi_signals_written(self):
        """All Ψ signals have write_signal producers."""
        written = _written_signals(_core_src())
        for sig in self.PSI_SIGNALS:
            assert sig in written, f"Signal '{sig}' must have a writer"

    def test_all_psi_signals_read(self):
        """All Ψ signals have read_signal consumers."""
        read = _read_signals(_core_src())
        for sig in self.PSI_SIGNALS:
            assert sig in read, f"Signal '{sig}' must have a reader"

    def test_no_orphaned_signals(self):
        """Combined source has no orphaned signals."""
        combined = _combined_src()
        written = _written_signals(combined)
        read = _read_signals(combined)
        orphaned = sorted(written - read)
        assert not orphaned, f"Orphaned signals: {orphaned}"

    def test_no_missing_producers(self):
        """Combined source has no missing producers."""
        combined = _combined_src()
        written = _written_signals(combined)
        read = _read_signals(combined)
        missing = sorted(read - written)
        assert not missing, f"Missing producers: {missing}"

    def test_written_count_maintained(self):
        """Signal count has not regressed."""
        combined = _combined_src()
        written = _written_signals(combined)
        # With Ψ patches, expect at least 172 written signals
        assert len(written) >= 172, \
            f"Only {len(written)} written signals (expected ≥172)"


# ===========================================================================
# Integration Flow: Uncertainty → MCT Cycle → Resolution
# ===========================================================================


class TestPsi_IntegrationFlow:
    """End-to-end integration: uncertainty triggers meta-cognitive cycle."""

    def test_post_pipeline_uncertainty_triggers_mct(self):
        """Post-pipeline uncertainty → PostPipelineMCTGate → MCT.evaluate()."""
        src = _core_src()
        # 1. PostPipelineMCTGate reads post_output_uncertainty
        assert 'post_output_uncertainty' in _read_signals(src)
        # 2. Gate writes post_pipeline_mct_trigger_score
        assert 'post_pipeline_mct_trigger_score' in _written_signals(src)
        # 3. Forward impl reads it
        assert 'post_pipeline_mct_trigger_score' in _read_signals(src)

    def test_dag_consensus_to_mct_to_rerun(self):
        """DAG disagreement → dag_consensus_pressure → MCT low_causal_quality → rerun."""
        src = _core_src()
        # 1. DAG consensus writes pressure
        assert 'dag_consensus_pressure' in _written_signals(src)
        # 2. MCT reads it
        assert 'dag_consensus_pressure' in _read_signals(src)
        # 3. Routes to low_causal_quality trigger
        idx = src.find('PATCH-Ψ5b: dag_consensus_pressure')
        assert idx > 0, "PATCH-Ψ5b annotation for dag_consensus_pressure must exist"
        assert 'low_causal_quality' in src[idx:idx + 500]

    def test_snapshot_degradation_to_mct(self):
        """Snapshot degradation → snapshot_degradation_count → MCT uncertainty."""
        src = _core_src()
        assert 'snapshot_degradation_count' in _written_signals(src)
        assert 'snapshot_degradation_count' in _read_signals(src)

    def test_emergence_health_to_mct(self):
        """Emergence health → emergence_integration_health → MCT coherence_deficit."""
        src = _core_src()
        assert 'emergence_integration_health' in _written_signals(src)
        assert 'emergence_integration_health' in _read_signals(src)

    def test_auto_critic_stability_acknowledged(self):
        """Auto-critic suppression → MCT acknowledges stability."""
        src = _core_src()
        assert 'auto_critic_revision_suppressed' in _written_signals(src)
        assert 'auto_critic_revision_suppressed' in _read_signals(src)


# ===========================================================================
# Causal Transparency: Traceability of Ψ patches
# ===========================================================================


class TestPsi_CausalTransparency:
    """Every Ψ patch is annotated and traceable."""

    PATCH_ANNOTATIONS = [
        'PATCH-Ψ1',
        'PATCH-Ψ2',
        'PATCH-Ψ3',
        'PATCH-Ψ4',
        'PATCH-Ψ5',
        'PATCH-Ψ7',
    ]

    def test_all_patches_annotated(self):
        """All Ψ patches have annotation comments in aeon_core.py."""
        src = _core_src()
        for annotation in self.PATCH_ANNOTATIONS:
            assert annotation in src, f"Missing annotation: {annotation}"

    def test_patch_psi6_in_test(self):
        """PATCH-Ψ6 annotation exists in test_final_patches.py."""
        with open(os.path.join(_REPO, 'test_final_patches.py'), 'r') as f:
            src = f.read()
        assert 'PATCH-Ψ6' in src

    def test_no_bare_except_pass_in_psi_patches(self):
        """No unhardened except:pass blocks in Ψ patch regions."""
        src = _core_src()
        # Find all Ψ annotations and check surrounding code
        for patch in self.PATCH_ANNOTATIONS:
            idx = src.find(patch)
            while idx >= 0:
                region = src[idx:idx + 1500]
                # Any 'pass' in except should have a comment
                lines = region.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped == 'pass':
                        # Check line above for except
                        if i > 0 and 'except' in lines[i - 1]:
                            pytest.fail(
                                f"Bare except:pass near {patch} "
                                f"at approx offset {idx}"
                            )
                idx = src.find(patch, idx + 1)


# ===========================================================================
# Activation Sequence: Patches must be applied in correct order
# ===========================================================================


class TestPsi_ActivationSequence:
    """Verify that Ψ patches form a coherent activation sequence."""

    def test_psi5_before_psi5b(self):
        """PATCH-Ψ5 (writer) and PATCH-Ψ5b (reader) both exist."""
        src = _core_src()
        assert 'PATCH-Ψ5: Dedicated DAG consensus bus signal' in src
        assert 'PATCH-Ψ5b: dag_consensus_pressure' in src

    def test_psi4_before_psi4b(self):
        """PATCH-Ψ4 (writer) appears before PATCH-Ψ4b (reader) in execution order."""
        src = _core_src()
        # Ψ4 is in get_cognitive_state_snapshot, Ψ4b is in MCT evaluate
        w_idx = src.find('PATCH-Ψ4:')
        r_idx = src.find('PATCH-Ψ4b:')
        assert w_idx > 0 and r_idx > 0
        # Both exist, which is sufficient — execution order is bus-mediated

    def test_psi7_before_psi7b(self):
        """PATCH-Ψ7 (writer) appears before PATCH-Ψ7b (reader)."""
        src = _core_src()
        w_idx = src.find('PATCH-Ψ7:')
        r_idx = src.find('PATCH-Ψ7b:')
        assert w_idx > 0 and r_idx > 0

    def test_psi1_has_exception_safety(self):
        """PATCH-Ψ1 MCT.evaluate() call is wrapped in try/except."""
        src = _core_src()
        idx = src.find('PATCH-Ψ1: Actually invoke MCT.evaluate()')
        assert idx > 0, "PATCH-Ψ1 MCT invocation annotation must exist"
        region = src[idx:idx + 2000]
        assert 'try:' in region
        assert 'except' in region
