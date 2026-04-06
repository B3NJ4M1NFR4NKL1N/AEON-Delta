#!/usr/bin/env python3
"""Merge 33 test files into section blocks for appending to test_aeon_unified.py.

Reads all test_*.py files (except test_aeon_unified.py), extracts test classes
and helper functions, deduplicates helpers, and writes merged sections to
merged_sections.py in the same directory.
"""

import ast
import os
import re
import sys
import textwrap
from collections import OrderedDict

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "merged_sections.py")

SECTION_START = 34  # sections 01-33 already exist in test_aeon_unified.py

# Ordered mapping: filename → (section description, series prefix for dedup)
FILE_SECTIONS = OrderedDict([
    ("test_academic_refinements.py",
     ("ACADEMIC REFINEMENTS — KM, Banach, IQC, Catastrophe", "academic")),
    ("test_act_patches.py",
     ("ACT-SERIES — Oscillation, Spectral, Certificate, Criticality", "act")),
    ("test_ca_patches.py",
     ("CA-SERIES — Meta-Cognitive Recursion, Spectral Bifurcation", "ca")),
    ("test_cact_patches.py",
     ("CACT-SERIES — Orphaned Signals, Bidirectional Bridges", "cact")),
    ("test_cogact_patches.py",
     ("COGACT-SERIES — Unified Convergence, Lipschitz, SSM", "cogact")),
    ("test_cogfinal_patches.py",
     ("COGFINAL-SERIES — Anderson Safeguard, Lyapunov, Memory", "cogfinal")),
    ("test_cognitive_activation_final.py",
     ("COGNITIVE ACTIVATION FINAL — LayerNorm, Gates, IQC", "cogactfinal")),
    ("test_cognitive_analysis_fixes.py",
     ("COGNITIVE ANALYSIS — Contraction, NOTEARS, Von Neumann", "coganalysis")),
    ("test_cp_integration_patches.py",
     ("CP-INTEGRATION — Causal Trace, Training Bus", "cpint")),
    ("test_cp_patches.py",
     ("CP-SERIES — Catastrophe MCT, Curriculum, Diversity", "cp")),
    ("test_d_series_patches.py",
     ("D-SERIES PATCHES — Error Evolution, Recursion, Oscillation", "dseries")),
    ("test_deep_cognitive_analysis.py",
     ("DEEP COGNITIVE ANALYSIS — T-IQC, LayerNorm, KM", "deepcog")),
    ("test_emerge_patches.py",
     ("EMERGE-SERIES — Memory Bus, Social/Sandbox, World Model", "emerge")),
    ("test_emrg_patches.py",
     ("EMRG-SERIES — Error Recording, Convergence Spectral", "emrg")),
    ("test_fca_patches.py",
     ("FCA-SERIES — Convergence Guarantees, KM Bounds, IQC", "fca")),
    ("test_fci_patches.py",
     ("FCI-SERIES — Lyapunov Iteration, Gamma, Joint Lipschitz", "fci")),
    ("test_fia_patches.py",
     ("FIA-SERIES — Post-Output Uncertainty, Verdict, Recovery", "fia")),
    ("test_final_cognitive_activation.py",
     ("FINAL COGNITIVE ACTIVATION — Bus, MCT, Causal", "finalcogact")),
    ("test_final_integration.py",
     ("FINAL INTEGRATION — Spectral Gates, Anderson", "finalint")),
    ("test_final_integration_cognitive_activation.py",
     ("FINAL INTEGRATION ACTIVATION — Sandwich, Spectral", "finalintact")),
    ("test_final_integration_patches.py",
     ("FINAL INTEGRATION PATCHES — Bus Init, Cross-Pass", "finalintpatch")),
    ("test_final_patches.py",
     ("FINAL PATCHES — Error Recovery, Orphaned, MCT Loss", "finalpatch")),
    ("test_gap_patches.py",
     ("GAP-SERIES — Oscillation MCT, Anderson, Decoder", "gap")),
    ("test_integration_patches.py",
     ("INTEGRATION PATCHES — Spectral, Feedback Gate", "intpatch")),
    ("test_k_series_patches.py",
     ("K-SERIES — UCC Coherence, SSP, Reexecution", "kseries")),
    ("test_p_series_patches.py",
     ("P-SERIES — Wizard Bridge, Causal Trace, Provenance", "pseries")),
    ("test_r_series_patches.py",
     ("R-SERIES — Lipschitz, Contraction, Recursion", "rseries")),
    ("test_rigor_patches.py",
     ("RIGOR-SERIES — SSM Diagonal, Banach, LayerNorm, KM", "rigor")),
    ("test_s_series_patches.py",
     ("S-SERIES — Silent Exceptions, Missing Signals, Orphans", "sseries")),
    ("test_sigma_integration_patches.py",
     ("SIGMA-INTEGRATION — MCT Planner, Convergence, OOM", "sigmaint")),
    ("test_sigma_patches.py",
     ("SIGMA-SERIES — Seven Sigma Patches (Σ1–Σ7)", "sigma")),
    ("test_syn_patches.py",
     ("SYN-SERIES — Stall Severity, Axiom Bus, Error Recovery", "syn")),
    ("test_theoretical_rigor_fixes.py",
     ("THEORETICAL RIGOR — IQC, Catastrophe, DAG, KM", "theorigor")),
])

# ── Source extraction helpers ──────────────────────────────────────────────


def _read_source_lines(filepath):
    """Read file and return list of lines (with newlines)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.readlines()


def _get_node_source(lines, node, next_node=None):
    """Extract source code for an AST node using line numbers.

    Uses the start of the next sibling node (or EOF) to determine the end,
    then strips trailing blank lines.
    """
    start = node.lineno - 1  # 0-indexed
    if next_node is not None:
        end = next_node.lineno - 1
    else:
        end = len(lines)
    source_lines = lines[start:end]
    # Strip trailing blank lines
    while source_lines and source_lines[-1].strip() == "":
        source_lines.pop()
    return "".join(source_lines)


def _get_decorator_start(lines, node):
    """Find the first decorator line for a node, if any."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno - 1
    return node.lineno - 1


def _get_node_source_with_decorators(lines, node, next_node=None):
    """Extract source including decorators."""
    start = _get_decorator_start(lines, node)
    if next_node is not None:
        end = _get_decorator_start(lines, next_node)
    else:
        end = len(lines)
    source_lines = lines[start:end]
    while source_lines and source_lines[-1].strip() == "":
        source_lines.pop()
    return "".join(source_lines)


def _replace_aeon_core_refs(source):
    """Replace aeon_core.ClassName with ClassName for direct imports."""
    return re.sub(r'\baeon_core\.(\w+)', r'\1', source)


def _is_helper_function(node):
    """Check if a FunctionDef is a module-level helper (starts with _)."""
    return isinstance(node, ast.FunctionDef) and node.name.startswith("_")


def _is_fixture_function(node):
    """Check if a FunctionDef has @pytest.fixture decorator."""
    if not isinstance(node, ast.FunctionDef):
        return False
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "fixture":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "fixture":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "fixture":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "fixture":
                return True
    return False


def _is_test_class(node):
    """Check if node is a test class."""
    return isinstance(node, ast.ClassDef) and node.name.startswith("Test")


def _rename_with_prefix(name, prefix):
    """Generate a prefixed name for deduplication."""
    if name.startswith("_"):
        return f"_{prefix}_{name.lstrip('_')}"
    return f"{prefix}_{name}"


def _is_module_level_assign(node):
    """Check if node is a module-level assignment (constants, dicts, etc.)."""
    if not isinstance(node, ast.Assign):
        return False
    for target in node.targets:
        if isinstance(target, ast.Name):
            return True
    return False


def _get_assign_names(node):
    """Get the names from an assignment node."""
    names = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _count_test_methods(node):
    """Count test methods in a class."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
            count += 1
    return count


# ── Main merge logic ──────────────────────────────────────────────────────


def extract_file(filepath, series_prefix, seen_helpers, is_emerge):
    """Extract helpers, constants, fixtures, and test classes from a file.

    Returns (helpers_code, classes_code, stats_dict).
    """
    lines = _read_source_lines(filepath)
    source = "".join(lines)
    tree = ast.parse(source)

    top_level = list(ast.iter_child_nodes(tree))

    helpers_parts = []
    classes_parts = []
    num_tests = 0
    num_classes = 0
    num_helpers = 0

    for i, node in enumerate(top_level):
        next_node = top_level[i + 1] if i + 1 < len(top_level) else None

        # Skip imports, docstrings, sys.path manipulation, if __name__ blocks
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(
            node.value, ast.Constant
        ):
            continue
        if isinstance(node, ast.If):
            # Skip `if __name__ == "__main__"` blocks
            continue

        # Module-level assignments (constants like STRICT_TOL, _MCT_DEFAULTS)
        if _is_module_level_assign(node):
            assign_names = _get_assign_names(node)
            code = _get_node_source(lines, node, next_node)
            if is_emerge:
                code = _replace_aeon_core_refs(code)

            # Check for duplicates
            new_names = []
            for name in assign_names:
                if name in seen_helpers:
                    new_name = _rename_with_prefix(name, series_prefix)
                    code = re.sub(rf'\b{re.escape(name)}\b', new_name, code, count=1)
                    new_names.append(new_name)
                else:
                    seen_helpers.add(name)
                    new_names.append(name)

            helpers_parts.append(code)
            num_helpers += 1
            continue

        # Helper functions (starting with _) or fixture functions
        if isinstance(node, ast.FunctionDef):
            is_helper = _is_helper_function(node)
            is_fixture = _is_fixture_function(node)

            if is_helper or is_fixture:
                code = _get_node_source_with_decorators(lines, node, next_node)
                if is_emerge:
                    code = _replace_aeon_core_refs(code)

                func_name = node.name
                if func_name in seen_helpers:
                    new_name = _rename_with_prefix(func_name, series_prefix)
                    # Replace definition
                    code = re.sub(
                        rf'\bdef {re.escape(func_name)}\b',
                        f'def {new_name}',
                        code,
                        count=1,
                    )
                    seen_helpers.add(new_name)
                else:
                    seen_helpers.add(func_name)

                helpers_parts.append(code)
                num_helpers += 1
                continue

        # Test classes
        if _is_test_class(node):
            code = _get_node_source_with_decorators(lines, node, next_node)
            if is_emerge:
                code = _replace_aeon_core_refs(code)
            classes_parts.append(code)
            num_classes += 1
            num_tests += _count_test_methods(node)
            continue

    stats = {
        "tests": num_tests,
        "classes": num_classes,
        "helpers": num_helpers,
    }
    return helpers_parts, classes_parts, stats


def make_section_header(section_num, description, filename, num_tests, num_classes):
    """Generate a section banner."""
    bar = "═" * 75
    return (
        f"\n\n# {bar}\n"
        f"#  SECTION {section_num:02d}: {description}\n"
        f"#  Source: {filename} | Tests: {num_tests} | Classes: {num_classes}\n"
        f"# {bar}\n"
    )


def main():
    os.chdir(SCRIPT_DIR)

    # Verify all expected files exist
    missing = [f for f in FILE_SECTIONS if not os.path.isfile(f)]
    if missing:
        print(f"ERROR: Missing files: {missing}", file=sys.stderr)
        sys.exit(1)

    seen_helpers = set()
    all_sections = []
    total_tests = 0
    total_classes = 0
    total_helpers = 0
    section_num = SECTION_START

    for filename, (description, series_prefix) in FILE_SECTIONS.items():
        filepath = os.path.join(SCRIPT_DIR, filename)
        is_emerge = filename == "test_emerge_patches.py"

        helpers_parts, classes_parts, stats = extract_file(
            filepath, series_prefix, seen_helpers, is_emerge
        )

        header = make_section_header(
            section_num, description, filename,
            stats["tests"], stats["classes"],
        )

        section_code = header
        if helpers_parts:
            section_code += "\n" + "\n\n\n".join(helpers_parts) + "\n"
        if classes_parts:
            section_code += "\n\n" + "\n\n\n".join(classes_parts) + "\n"

        all_sections.append(section_code)

        print(
            f"  Section {section_num:02d}: {filename:<50s} "
            f"classes={stats['classes']:>2d}  tests={stats['tests']:>3d}  "
            f"helpers={stats['helpers']:>2d}"
        )

        total_tests += stats["tests"]
        total_classes += stats["classes"]
        total_helpers += stats["helpers"]
        section_num += 1

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("".join(all_sections))
        f.write("\n")

    print()
    print("=" * 60)
    print(f"  MERGE COMPLETE")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"  Sections: {len(all_sections)}")
    print(f"  Total classes: {total_classes}")
    print(f"  Total tests:   {total_tests}")
    print(f"  Total helpers: {total_helpers}")
    print(f"  Deduped helper names tracked: {len(seen_helpers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
