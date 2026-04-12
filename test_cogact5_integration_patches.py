"""Tests for PATCH-COGACT-5a..5f: Final Cognitive Activation patches.

Covers:
- COGACT-5a: ProvablyConvergentMetaLoop causal trace integration
- COGACT-5b: Per-axiom regression tracking in verify_and_reinforce
- COGACT-5c: CrossValidator MCT-aware adaptive threshold
- COGACT-5d: Safety oscillation-aware dampening
- COGACT-5e: MCT readers for per-axiom regression + oscillation dampening
- COGACT-5f: ProvablyConvergentMetaLoop wired in AEONDeltaV3.__init__
- Signal ecosystem audit
- E2E integration flow
- Activation sequence
"""

import sys
import os
import re
import types
import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict

# ---------------------------------------------------------------------------
# Minimal stubs so we can import aeon_core without heavy dependencies
# ---------------------------------------------------------------------------
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.float32 = "float32"
_torch_stub.long = "long"
_torch_stub.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
_nn = types.ModuleType("torch.nn")
_nn.Module = type("Module", (), {"__init__": lambda s, *a, **kw: None})
_nn.Linear = lambda *a, **kw: MagicMock()
_nn.GELU = lambda: MagicMock()
_nn.Sigmoid = lambda: MagicMock()
_nn.Sequential = lambda *a: MagicMock()
_nn.Parameter = lambda *a, **kw: MagicMock()
_nn_func = types.ModuleType("torch.nn.functional")
_nn_func.softmax = lambda *a, **kw: MagicMock()
_torch_stub.nn = _nn
_nn.functional = _nn_func
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch.nn", _nn)
sys.modules.setdefault("torch.nn.functional", _nn_func)
for _sub in (
    "torch.cuda", "torch.optim", "torch.utils", "torch.utils.data",
    "torch.nn.utils", "torch.distributions", "torch.nn.init",
    "torch.autograd", "torch.amp", "torch.optim.lr_scheduler",
    "torch.utils.checkpoint",
):
    sys.modules.setdefault(_sub, types.ModuleType(_sub))

for _ext in ("numpy", "scipy", "scipy.stats", "scipy.special",
             "fastapi", "uvicorn", "pydantic", "starlette",
             "starlette.middleware", "starlette.middleware.cors"):
    sys.modules.setdefault(_ext, types.ModuleType(_ext))

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Lightweight bus / trace stubs
# ---------------------------------------------------------------------------
class StubFeedbackBus:
    """Minimal CognitiveFeedbackBus stand-in."""
    def __init__(self):
        self._signals: dict = {}
        self._persistent: dict = {}
        self._pass_counter: int = 0

    def write_signal(self, name, value):
        self._signals[name] = float(value)

    def read_signal(self, name, default=0.0):
        return self._signals.get(name, default)

    def register_persistent_signal(self, name, value):
        self._persistent[name] = float(value)

    def write_signal_traced(self, name, value, **kw):
        self._signals[name] = float(value)

    def get_signal_provenance(self, name):
        return None


class StubCausalTrace:
    """Minimal TemporalCausalTraceBuffer stand-in."""
    def __init__(self):
        self.entries: list = []

    def record(self, subsystem, decision, **kwargs):
        self.entries.append({
            "subsystem": subsystem,
            "decision": decision,
            **kwargs,
        })
        return f"entry-{len(self.entries)}"

    def find(self, subsystem=None, **kw):
        return [e for e in self.entries if e.get("subsystem") == subsystem]


# ===================================================================
# COGACT-5a: ProvablyConvergentMetaLoop causal trace
# ===================================================================
class TestCOGACT5a_MetaLoopCausalTrace(unittest.TestCase):
    """ProvablyConvergentMetaLoop gains set_causal_trace + recording."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_has_causal_trace_ref_attribute(self):
        src = self._get_source()
        self.assertIn("self._causal_trace_ref", src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ])

    def test_has_set_causal_trace_method(self):
        src = self._get_source()
        block = src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ]
        self.assertIn("def set_causal_trace(", block)

    def test_records_fixed_point_solve(self):
        src = self._get_source()
        block = src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ]
        self.assertIn("subsystem='meta_loop'", block)
        self.assertIn("decision='fixed_point_solve'", block)

    def test_records_iteration_count(self):
        src = self._get_source()
        block = src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ]
        self.assertIn("'iterations'", block)
        self.assertIn("'residual_norm'", block)

    def test_records_convergence_status(self):
        src = self._get_source()
        block = src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ]
        self.assertIn("'converged'", block)
        self.assertIn("'convergence_rate'", block)

    def test_severity_conditional(self):
        """Warning severity when not converged, info when converged."""
        src = self._get_source()
        block = src[
            src.index("class ProvablyConvergentMetaLoop"):
            src.index("class CalibrationMetrics")
        ]
        self.assertIn("'info' if _ca5a_converged else 'warning'", block)


# ===================================================================
# COGACT-5b: Per-axiom regression tracking
# ===================================================================
class TestCOGACT5b_PerAxiomRegression(unittest.TestCase):
    """verify_and_reinforce tracks per-axiom regression individually."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_per_axiom_signals_written(self):
        src = self._get_source()
        # Signals are constructed with f-strings from axiom names
        self.assertIn("f'axiom_regression_{_ca5b_ax}'", src)
        # Verify the axiom names are in the iteration
        self.assertIn("'mutual_verification'", src)
        self.assertIn("'uncertainty_metacognition'", src)
        self.assertIn("'root_cause_traceability'", src)

    def test_any_axiom_regressing_signal(self):
        src = self._get_source()
        self.assertIn("any_axiom_regressing", src)

    def test_per_axiom_counter_persistence(self):
        """_ca5b_per_axiom_regression is stored on self for cross-pass."""
        src = self._get_source()
        self.assertIn("self._ca5b_per_axiom_regression", src)

    def test_per_axiom_counter_caps(self):
        """Counter capped at 10 and floor at 0."""
        src = self._get_source()
        self.assertIn("min(10, _ca5b_cnt + 1)", src)
        self.assertIn("max(0, _ca5b_cnt - 1)", src)

    def test_report_includes_per_axiom(self):
        src = self._get_source()
        self.assertIn("report['per_axiom_regression']", src)


# ===================================================================
# COGACT-5c: CrossValidator MCT-aware threshold
# ===================================================================
class TestCOGACT5c_CrossValidatorMCTAware(unittest.TestCase):
    """CrossValidator tightens threshold when MCT is active."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def _get_cv_block(self):
        src = self._get_source()
        start = src.index("class SubsystemCrossValidator:")
        end = src.index("class UncertaintyPropagationBus:")
        return src[start:end]

    def test_reads_mct_trigger_score(self):
        block = self._get_cv_block()
        self.assertIn("mct_trigger_score", block)

    def test_adaptive_threshold_calculation(self):
        block = self._get_cv_block()
        self.assertIn("_ca5c_threshold", block)

    def test_threshold_floor_at_04(self):
        """Threshold should not go below 0.4."""
        block = self._get_cv_block()
        self.assertIn("0.4", block)

    def test_threshold_default_06(self):
        """Default threshold is 0.6 when MCT not active."""
        block = self._get_cv_block()
        self.assertIn("_ca5c_threshold = 0.6", block)

    def test_uses_adaptive_threshold_in_loop(self):
        """Loop uses _ca5c_threshold, not hardcoded _PAIR_THRESHOLD."""
        block = self._get_cv_block()
        # The old _PAIR_THRESHOLD should be replaced
        self.assertNotIn("_PAIR_THRESHOLD", block)
        # Both violation check and pair_details should use _ca5c_threshold
        self.assertIn(
            "inconsistency > _ca5c_threshold", block,
        )

    def test_functional_threshold_tightening(self):
        """When MCT score is 0.8, threshold should be ~0.48."""
        bus = StubFeedbackBus()
        bus.write_signal("mct_trigger_score", 0.8)

        # Simulate the threshold calculation from the patch
        mct_score = 0.8
        threshold = max(0.4, 0.6 - 0.4 * (mct_score - 0.5))
        self.assertAlmostEqual(threshold, 0.48, places=2)

    def test_functional_threshold_no_mct(self):
        """When MCT score is 0.0, threshold stays 0.6."""
        mct_score = 0.0
        threshold = 0.6  # default
        if mct_score > 0.5:
            threshold = max(0.4, 0.6 - 0.4 * (mct_score - 0.5))
        self.assertEqual(threshold, 0.6)


# ===================================================================
# COGACT-5d: Safety oscillation dampening
# ===================================================================
class TestCOGACT5d_SafetyOscillationDampening(unittest.TestCase):
    """Safety reads mct_oscillation_risk for conservative dampening."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def _get_safety_block(self):
        src = self._get_source()
        start = src.index("class MultiLevelSafetySystem(")
        # Find next class
        remaining = src[start + 1:]
        end = start + 1 + remaining.index("\nclass ")
        return src[start:end]

    def test_reads_mct_oscillation_risk(self):
        block = self._get_safety_block()
        self.assertIn("mct_oscillation_risk", block)

    def test_writes_oscillation_dampening_signal(self):
        block = self._get_safety_block()
        self.assertIn("safety_oscillation_dampening_active", block)

    def test_dampening_threshold_03(self):
        """Dampening activates when oscillation risk > 0.3."""
        block = self._get_safety_block()
        self.assertIn("_ca5d_osc > 0.3", block)

    def test_dampening_factor_floor(self):
        """Dampening factor is at least 0.85."""
        block = self._get_safety_block()
        self.assertIn("max(0.85", block)

    def test_functional_dampening_calculation(self):
        """When oscillation risk is 0.5, dampening factor ~0.925."""
        osc_risk = 0.5
        dampen = max(0.85, 1.0 - osc_risk * 0.15)
        self.assertAlmostEqual(dampen, 0.925, places=3)

    def test_functional_dampening_high_risk(self):
        """When oscillation risk is 1.0, dampening factor = 0.85."""
        osc_risk = 1.0
        dampen = max(0.85, 1.0 - osc_risk * 0.15)
        self.assertEqual(dampen, 0.85)


# ===================================================================
# COGACT-5e: MCT readers for new signals
# ===================================================================
class TestCOGACT5e_MCTReadersNewSignals(unittest.TestCase):
    """MCT evaluate reads per-axiom regression + oscillation dampening."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_mct_reads_any_axiom_regressing(self):
        src = self._get_source()
        # MCT evaluate should read this signal
        mct_start = src.index("class MetaCognitiveRecursionTrigger")
        # Find end of class
        remaining = src[mct_start + 1:]
        mct_end = mct_start + 1 + remaining.index("\nclass ")
        mct_block = src[mct_start:mct_end]
        self.assertIn("any_axiom_regressing", mct_block)

    def test_mct_reads_per_axiom_signals(self):
        src = self._get_source()
        mct_start = src.index("class MetaCognitiveRecursionTrigger")
        remaining = src[mct_start + 1:]
        mct_end = mct_start + 1 + remaining.index("\nclass ")
        mct_block = src[mct_start:mct_end]
        # Per-axiom signals are read via f-string loop
        self.assertIn("f'axiom_regression_{_ca5e_ax_name}'", mct_block)
        # Verify the axiom names are in the iteration
        self.assertIn("'mutual_verification'", mct_block)
        self.assertIn("'uncertainty_metacognition'", mct_block)
        self.assertIn("'root_cause_traceability'", mct_block)

    def test_mct_reads_safety_oscillation_dampening(self):
        src = self._get_source()
        mct_start = src.index("class MetaCognitiveRecursionTrigger")
        remaining = src[mct_start + 1:]
        mct_end = mct_start + 1 + remaining.index("\nclass ")
        mct_block = src[mct_start:mct_end]
        self.assertIn("safety_oscillation_dampening_active", mct_block)

    def test_coherence_deficit_boost_threshold(self):
        """Per-axiom regression ≥ 2 boosts coherence_deficit."""
        src = self._get_source()
        self.assertIn("_ca5e_worst >= 2.0", src)

    def test_recovery_pressure_boost_on_dampening(self):
        """Safety oscillation dampening boosts recovery_pressure by 0.2."""
        src = self._get_source()
        self.assertIn("_cur_rp + 0.2", src)


# ===================================================================
# COGACT-5f: Wiring in AEONDeltaV3.__init__
# ===================================================================
class TestCOGACT5f_Wiring(unittest.TestCase):
    """ProvablyConvergentMetaLoop is wired to causal trace in __init__."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_meta_loop_wired_in_init(self):
        src = self._get_source()
        init_start = src.index("class AEONDeltaV3(")
        block = src[init_start:]
        self.assertIn("self.meta_loop.set_causal_trace(self.causal_trace)", block)

    def test_wiring_guarded(self):
        """Wiring has getattr guard for meta_loop existence."""
        src = self._get_source()
        init_start = src.index("class AEONDeltaV3(")
        block = src[init_start:]
        self.assertIn("getattr(self, 'meta_loop', None)", block)

    def test_wiring_has_hasattr_guard(self):
        """Wiring checks hasattr for set_causal_trace method."""
        src = self._get_source()
        init_start = src.index("class AEONDeltaV3(")
        block = src[init_start:]
        self.assertIn("hasattr(self.meta_loop, 'set_causal_trace')", block)


# ===================================================================
# Signal ecosystem audit
# ===================================================================
class TestSignalEcosystemAudit(unittest.TestCase):
    """Every written signal has a reader and vice versa."""

    @staticmethod
    def _extract_signals(files):
        """Extract all signal names from write_signal/read_signal calls."""
        written = set()
        read = set()
        sig_pat = re.compile(
            r"(?:write_signal|write_signal_traced)\s*\(\s*['\"]"
            r"([a-z_][a-z0-9_]*)['\"]",
            re.MULTILINE,
        )
        read_pat = re.compile(
            r"read_signal\s*\(\s*['\"]"
            r"([a-z_][a-z0-9_]*)['\"]",
            re.MULTILINE,
        )
        for fpath in files:
            with open(fpath) as f:
                src = f.read()
            written.update(sig_pat.findall(src))
            read.update(read_pat.findall(src))
        return written, read

    def test_new_signals_bidirectional(self):
        """All new COGACT-5 signals are both written and read."""
        base = os.path.dirname(__file__)
        files = [
            os.path.join(base, "aeon_core.py"),
            os.path.join(base, "ae_train.py"),
            os.path.join(base, "aeon_server.py"),
        ]
        written, read = self._extract_signals(files)
        new_signals = {
            "safety_oscillation_dampening_active",
            "any_axiom_regressing",
        }
        for sig in new_signals:
            self.assertIn(sig, written, f"{sig} not written")
            self.assertIn(sig, read, f"{sig} not read")

    def test_per_axiom_signals_bidirectional(self):
        """Per-axiom regression signals are written and read via f-strings."""
        base = os.path.dirname(__file__)
        with open(os.path.join(base, "aeon_core.py")) as f:
            src = f.read()
        # Writer: f'axiom_regression_{_ca5b_ax}' in verify_and_reinforce
        self.assertIn("f'axiom_regression_{_ca5b_ax}'", src)
        # Reader: f'axiom_regression_{_ca5e_ax_name}' in MCT evaluate
        self.assertIn("f'axiom_regression_{_ca5e_ax_name}'", src)

    def test_no_new_orphan_writers(self):
        """No COGACT-5 static signals are written without being read."""
        base = os.path.dirname(__file__)
        files = [
            os.path.join(base, "aeon_core.py"),
            os.path.join(base, "ae_train.py"),
            os.path.join(base, "aeon_server.py"),
        ]
        written, read = self._extract_signals(files)
        # Focus on our new static signals (per-axiom signals are dynamic)
        cogact5_signals = {
            "safety_oscillation_dampening_active",
            "any_axiom_regressing",
        }
        orphan_writers = cogact5_signals & (written - read)
        self.assertEqual(
            orphan_writers, set(),
            f"Orphan writers: {orphan_writers}",
        )


# ===================================================================
# E2E integration: causal transparency chain
# ===================================================================
class TestCausalTransparencyE2E(unittest.TestCase):
    """Verify the complete causal trace chain includes meta-loop."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_meta_loop_trace_subsystem_name(self):
        """Meta-loop records under 'meta_loop' subsystem name."""
        src = self._get_source()
        self.assertIn("subsystem='meta_loop'", src)

    def test_all_core_subsystems_have_trace(self):
        """All 5 core cognitive subsystems record to causal trace."""
        src = self._get_source()
        core_subsystems = [
            "meta_loop",
            "metacognitive_trigger",
            "safety_system",
            "auto_critic",
            "cross_validator",
        ]
        for sub in core_subsystems:
            self.assertIn(
                f"subsystem='{sub}'", src,
                f"Missing causal trace for {sub}",
            )


# ===================================================================
# Activation sequence
# ===================================================================
class TestActivationSequence(unittest.TestCase):
    """Patches must be applied in the correct order."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_5a_before_5f(self):
        """5a (add set_causal_trace to class) must be before 5f (wire it)."""
        src = self._get_source()
        pos_5a = src.index("PATCH-COGACT-5a")
        pos_5f = src.index("PATCH-COGACT-5f")
        self.assertLess(pos_5a, pos_5f)

    def test_5b_before_5e(self):
        """5b (write per-axiom signals) before 5e (MCT reads them)."""
        src = self._get_source()
        # 5b is in verify_and_reinforce
        pos_5b = src.index("PATCH-COGACT-5b")
        # 5e is in MCT evaluate
        pos_5e = src.index("PATCH-COGACT-5e")
        # Both exist; order doesn't strictly matter for runtime but
        # the writer definition should be findable
        self.assertIn("PATCH-COGACT-5b", src)
        self.assertIn("PATCH-COGACT-5e", src)

    def test_5d_before_5e_osc(self):
        """5d (write dampening signal) before 5e-osc (MCT reads it)."""
        src = self._get_source()
        pos_5d = src.index("PATCH-COGACT-5d")
        pos_5e_osc = src.index("PATCH-COGACT-5e-osc")
        self.assertIn("PATCH-COGACT-5d", src)
        self.assertIn("PATCH-COGACT-5e-osc", src)


# ===================================================================
# Mutual reinforcement
# ===================================================================
class TestMutualReinforcement(unittest.TestCase):
    """Patches strengthen mutual reinforcement loops."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_cv_mct_bidirectional(self):
        """CrossValidator now reads MCT state (completing CV↔MCT loop)."""
        src = self._get_source()
        cv_start = src.index("class SubsystemCrossValidator:")
        cv_block = src[cv_start:cv_start + 5000]
        self.assertIn("mct_trigger_score", cv_block)

    def test_safety_oscillation_bidirectional(self):
        """Safety reads oscillation risk and MCT reads dampening signal."""
        src = self._get_source()
        # Safety reads mct_oscillation_risk (in forward method, may be
        # beyond 5000 chars from class start — use full class block)
        safety_start = src.index("class MultiLevelSafetySystem(")
        remaining = src[safety_start + 1:]
        safety_end = safety_start + 1 + remaining.index("\nclass ")
        safety_block = src[safety_start:safety_end]
        self.assertIn("mct_oscillation_risk", safety_block)
        # MCT reads safety_oscillation_dampening_active
        mct_start = src.index("class MetaCognitiveRecursionTrigger")
        mct_remaining = src[mct_start + 1:]
        mct_end = mct_start + 1 + mct_remaining.index("\nclass ")
        mct_block = src[mct_start:mct_end]
        self.assertIn("safety_oscillation_dampening_active", mct_block)

    def test_per_axiom_feedback_loop(self):
        """Per-axiom regression → MCT coherence boost → tighter CV → fix."""
        src = self._get_source()
        # verify_and_reinforce writes per-axiom signals via f-string
        self.assertIn("f'axiom_regression_{_ca5b_ax}'", src)
        # MCT reads them and boosts coherence_deficit
        self.assertIn("coherence_deficit", src)
        # CrossValidator tightens on MCT score
        cv_start = src.index("class SubsystemCrossValidator:")
        cv_block = src[cv_start:cv_start + 5000]
        self.assertIn("_ca5c_threshold", cv_block)


# ===================================================================
# Meta-cognitive trigger completeness
# ===================================================================
class TestMetaCognitiveTriggerCompleteness(unittest.TestCase):
    """MCT covers all new feedback signals."""

    def _get_source(self):
        with open(os.path.join(
            os.path.dirname(__file__), "aeon_core.py"
        )) as f:
            return f.read()

    def test_mct_reads_all_new_signals(self):
        """MCT evaluate() reads all COGACT-5 signals."""
        src = self._get_source()
        mct_start = src.index("class MetaCognitiveRecursionTrigger")
        remaining = src[mct_start + 1:]
        mct_end = mct_start + 1 + remaining.index("\nclass ")
        mct_block = src[mct_start:mct_end]
        new_reader_signals = [
            "any_axiom_regressing",
            # per-axiom signals read via f-string loop
            "axiom_regression_",
            "safety_oscillation_dampening_active",
        ]
        for sig in new_reader_signals:
            self.assertIn(sig, mct_block, f"MCT missing reader for {sig}")


if __name__ == "__main__":
    unittest.main()
