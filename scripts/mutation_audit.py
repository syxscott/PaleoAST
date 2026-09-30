"""Targeted mutation audit for the scientific core.

Coverage says code RAN. It does not say a test would FAIL if the code were
wrong. This script answers the second question for the error classes that
have actually occurred in this repository -- almost every mutation below
reinstates a real bug that shipped and was later found.

A mutation that SURVIVES means the test suite does not bite on that class of
error, which is a stronger statement about research-readiness than any
coverage percentage.

Run:  .venv/Scripts/python.exe scripts/mutation_audit.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (label, file, original snippet, mutated snippet, pytest target)
# `original` must appear EXACTLY ONCE in the file, and the mutation must be
# semantically different from the original (an equivalent rewrite would make
# the mutation score a lie).
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "PERMANOVA total SS divisor back to /n",
        "stats/permanova.py",
        "SS_T = np.sum(D_sq[np.triu_indices(n, k=1)]) / (n - 1)",
        "SS_T = np.sum(D_sq[np.triu_indices(n, k=1)]) / n",
        "tests/stats/test_permanova_ss.py",
    ),
    (
        "PERMANOVA within-group divisor back to /n_g",
        "stats/permanova.py",
        "ss_within += (1.0 / (n_g - 1)) * grp_sum",
        "ss_within += (1.0 / n_g) * grp_sum",
        "tests/stats/test_permanova_ss.py",
    ),
    (
        "Procrustes reflection branch inverted",
        "morphometrics/shape_stats.py",
        "if no_reflect and np.linalg.det(Vt.T @ U.T) < 0:",
        "if (not no_reflect) and np.linalg.det(Vt.T @ U.T) < 0:",
        "tests/golden/test_procrustes_ground_truth.py",
    ),
    (
        "Fitch down-pass ignores the last child (off-by-one)",
        "phylogenetics/fitch.py",
        "                for child_set in child_sets[1:]:",
        "                for child_set in child_sets[1:-1]:",
        "tests/phylogenetics/test_fitch.py",
    ),
    (
        "Fitch union taken from the running intersection",
        "phylogenetics/fitch.py",
        "                for child_set in child_sets[1:]:\n                    intersection = intersection & child_set",
        "                for child_set in child_sets[1:]:\n                    if not (intersection & child_set):\n                        intersection = intersection | child_set\n                    else:\n                        intersection = intersection & child_set",
        "tests/phylogenetics/test_fitch.py",
    ),
    (
        "Isotope 0.27 calibration offset dropped",
        "stratigraphy/isotope_analysis.py",
        "EL_BEMIS_VSMOW_TO_VPDB_OFFSET = 0.27",
        "EL_BEMIS_VSMOW_TO_VPDB_OFFSET = 0.0",
        "tests/stratigraphy/test_isotope_scale_conversion.py",
    ),
    (
        "Tukey q primary path multiplies by sqrt(2) again",
        "stats/univariate.py",
        "se = np.sqrt(ms_within * (1.0 / len(group_data[i]) + 1.0 / len(group_data[j])) / 2.0)",
        "se = np.sqrt(ms_within * (1.0 / len(group_data[i]) + 1.0 / len(group_data[j]))) / 2.0",
        "tests/stats/test_tukey_q.py",
    ),
    (
        "Tukey scipy-fallback path multiplies by sqrt(2) again",
        "stats/univariate.py",
        "se = np.sqrt(ms_within * (1.0 / ni + 1.0 / nj) / 2.0)",
        "se = np.sqrt(ms_within * (1.0 / ni + 1.0 / nj)) / 2.0",
        "tests/stats/test_tukey_q.py",
    ),
    (
        "Bray-Curtis returns 0 for two empty samples",
        "stats/distance_metrics.py",
        "D = np.where(denominator == 0, np.nan, numerator / denominator)",
        "D = np.where(denominator == 0, 0.0, numerator / denominator)",
        "tests/stats/test_bray_curtis_nan.py",
    ),
    (
        "PCoA correction added to ALL squared distances (no-op)",
        "stats/pcoa.py",
        "D_sq = D_sq + c_shift * (ones - np.eye(n))",
        "D_sq = D_sq + c_shift * ones",
        "tests/golden/test_pcoa_correction_math.py",
    ),
    (
        "FBD relative import restored (beyond top-level package)",
        "macroevolution/fbd.py",
        "from phylogenetics.tree import PhyloTree",
        "from ..phylogenetics.tree import PhyloTree",
        "tests/macroevolution/test_fbd.py",
    ),
    (
        "TPS evaluates 3-D with the 2-D kernel",
        "morphometrics/tps.py",
        "            else:\n                U = distances",
        "            else:\n                U = distances**2 * np.log(distances)",
        "tests/morphometrics/test_defect2_tps_3d_consistency.py",
    ),
    (
        "DAT parser stops recognising ? and * as missing",
        "parsers/sentinels.py",
        '        "?",  # NEXUS missing / tpsDig missing landmark',
        '        "  ",  # deliberately broken for the mutation audit',
        "tests/parsers/test_missing_value_sentinels.py",
    ),
    (
        "DataMatrix imputes all-NaN columns as 0 again",
        "models/data_matrix.py",
        "            # instead of a RuntimeWarning + the silently-zeroed fallback.\n            all_nan_cols = np.all(nan_mask, axis=0)\n            if np.any(all_nan_cols):",
        "            # instead of a RuntimeWarning + the silently-zeroed fallback.\n            all_nan_cols = np.all(nan_mask, axis=0)\n            if False:",
        "tests/models/test_data_matrix_types_and_imputation.py",
    ),
    (
        "Strict consensus silently unions mismatched taxon sets again",
        "phylogenetics/strict_consensus.py",
        "if len(trees) == 1:\n            return self._clone_tree(trees[0])\n\n        all_taxa = self._require_shared_taxa(trees)",
        "if len(trees) == 1:\n            return self._clone_tree(trees[0])\n\n        all_taxa = set(trees[0].leaf_names)\n        for _t in trees[1:]:\n            all_taxa |= set(_t.leaf_names)",
        "tests/phylogenetics/test_strict_consensus_ground_truth.py",
    ),
    (
        "TBR ancestor check inverted again (walk up from r1 looking for n1)",
        "phylogenetics/heuristic_search.py",
        "        walk = n1\n        while walk is not r1:",
        "        walk = r1\n        while walk is not n1:",
        "tests/phylogenetics/test_regression_pinned.py",
    ),
    (
        "DistanceMatrix.from_array accepts asymmetric input again",
        "phylogenetics/distance_methods.py",
        "        asymmetry = float(np.abs(matrix - matrix.T).max()) if n > 1 else 0.0",
        "        asymmetry = 0.0",
        "tests/phylogenetics/test_regression_pinned.py",
    ),
    (
        "LDA canonical roots revert to bounded variance ratios",
        "stats/lda.py",
        "        L = np.linalg.cholesky(S_W)",
        "        L = np.linalg.cholesky(S_W + 1.0 * np.eye(n_vars))",
        "tests/stats/test_lda_ground_truth.py",
    ),
    (
        "RASC swap loop pinned position 0 again",
        "stratigraphy/biostratigraphy.py",
        "                for i in range(n_events - 1):",
        "                for i in range(1, n_events - 1):",
        "tests/stratigraphy/test_biostratigraphy_ground_truth.py",
    ),
    (
        "Mesh volume computed for an open mesh again",
        "morpho3d/mesh.py",
        "        if boundary_edges and require_closed:",
        "        if False:",
        "tests/phylogenetics/test_regression_pinned.py",
    ),
    (
        "Fisher log-series fabricates x=0.5 when no root exists",
        "ecology/advanced.py",
        "        except ValueError as exc:\n            # No fabrication. A fitted parameter that contradicts the data is\n            # worse than an error, because a downstream goodness-of-fit number\n            # would then be computed against a model that never held.\n            raise ValueError(",
        "        except ValueError as exc:\n            x = 0.5\n            _unused = (\n            # No fabrication. A fitted parameter that contradicts the data is\n            # worse than an error, because a downstream goodness-of-fit number\n            # would then be computed against a model that never held.\n            )",
        "tests/phylogenetics/test_regression_pinned.py",
    ),
    (
        "DFA minimisation skips refinement when every state is accepting",
        "state_machine/automaton.py",
        "        ordered_alphabet = sorted(alphabet)",
        "        ordered_alphabet = sorted(alphabet)\n        if len(partitions) > 1:",
        "tests/test_regex_engine_ground_truth.py",
    ),
    (
        "Character-class ranges collapse to their endpoint characters again",
        "state_machine/automaton.py",
        "        members = _expand_character_class(chars[1:] if negated else chars)",
        "        members = set(chars[1:] if negated else chars)",
        "tests/test_regex_engine_ground_truth.py",
    ),
    (
        "The dot becomes a literal character again",
        "state_machine/automaton.py",
        "            self._advance()\n            return RegexNode(RegexNodeType.ANY)",
        "            self._advance()\n            return RegexNode(RegexNodeType.CHAR, value=char)",
        "tests/test_regex_engine_ground_truth.py",
    ),
    (
        "LaTeX preamble loads a package twice again",
        "reporting/latex_preamble.py",
        "        if any(self._package_key(line) == key for line in self._packages):",
        "        if False:",
        "tests/test_reporting_layer.py",
    ),
    (
        "Task scheduler swallows the caller's positional arguments",
        "hpc/task_scheduler.py",
        "        /,\n        *args,\n        task_id: str | None = None,",
        "        task_id: str | None = None,",
        "tests/test_hpc_layer.py",
    ),
    (
        "Task scheduler accepts work after shutdown again",
        "hpc/task_scheduler.py",
        "        if self._shutdown_event.is_set():",
        "        if False:",
        "tests/test_hpc_layer.py",
    ),
    (
        "TBR returns the starting topology re-rooted (a self-loop)",
        "phylogenetics/heuristic_search.py",
        "            if start_splits is not None and key == start_splits:",
        "            if False:",
        "tests/phylogenetics/test_heuristic_search_ground_truth.py",
    ),
    (
        "ARMA manual fallback leaves MA coefficients at zero again",
        "stratigraphy/arma.py",
        "ma_params = self._fit_ma_hannan_rissanen(y_c=y_centered if p > 0 else (y - mu), p=p, q=q)",
        "ma_params = np.zeros(q)",
        # test_arma_ground_truth.py alone did not catch this; the coverage lives
        # in test_arma_hannan_rissanen.py, which forces the statsmodels path to
        # fail and then asserts a real coefficient comes back.
        "tests/stratigraphy/test_arma_hannan_rissanen.py",
    ),
    (
        "ARMA order search skips p=0 and q=0 again",
        "stratigraphy/arma.py",
        "        for p in range(0, max_p + 1):\n            for q in range(0, max_q + 1):",
        "        for p in range(1, max_p + 1):\n            for q in range(1, max_q + 1):",
        "tests/stratigraphy/test_arma_ground_truth.py",
    ),
    (
        "ARMA AIC/BIC becomes -inf on a degenerate fit again",
        "stratigraphy/arma.py",
        "            if not np.isfinite(sigma2) or sigma2 <= 0.0:",
        "            if False:",
        "tests/stratigraphy/test_arma_ground_truth.py",
    ),
    (
        "Convex hull reports inf instead of refusing a degenerate cloud again",
        "stats/geometry.py",
        '            raise ComputationError(f"Convex hull computation failed: {e}")',
        '            return float("inf")',
        "tests/stats/test_geometry_ground_truth.py",
    ),
    (
        "ProcessPool.map puts None through for failed items again",
        "hpc/process_pool.py",
        "        if (failed_items or failed_chunks) and raise_on_error:",
        "        if False:",
        "tests/test_process_pool.py",
    ),
    (
        "ProcessPool.get_result stops waiting again",
        "hpc/process_pool.py",
        '        deadline = None if timeout is None else __import__("time").time() + timeout',
        '        deadline = __import__("time").time() - 1.0 if timeout else None',
        "tests/test_process_pool.py",
    ),
    (
        "Lexer stops populating the group-to-rule map (every token UNKNOWN)",
        "state_machine/tokenizer.py",
        "            self._group_rules[group_name] = rule",
        "            self._group_rules[group_name] = None",
        "tests/test_tokenizer_ground_truth.py",
    ),
    (
        "create_basic_lexer goes back to case-sensitive keyword matching",
        "state_machine/tokenizer.py",
        "    lexer = LexerTokenizer(rules, case_sensitive=False)",
        "    lexer = LexerTokenizer(rules, case_sensitive=True)",
        "tests/test_tokenizer_ground_truth.py",
    ),
    (
        "Rarefaction goes back to the Stirling/log-gamma approximation",
        "ecology/rarefaction.py",
        "        return float(math.comb(n, k))",
        "        return np.exp(_log_factorial(n) - _log_factorial(k) - _log_factorial(n - k))",
        "tests/ecology/test_rarefaction_ground_truth.py",
    ),
]


def run(target: str) -> tuple[int, str]:
    """Run one pytest target; return (exit code, last line of output).

    ``encoding``/``errors`` are explicit because the suite prints Chinese, and a
    subprocess on a machine whose console encoding is not UTF-8 (a Windows
    runner with a CJK locale, for instance) emits bytes that ``text=True``
    would otherwise try to decode as UTF-8. That raised UnicodeDecodeError
    inside the reader thread, which left ``proc.stdout`` as None and turned a
    perfectly good mutation into
    ``AttributeError: 'NoneType' object has no attribute 'strip'`` -- the audit
    died instead of reporting. Decoding leniently costs nothing here: the
    summary line only needs to say how many tests passed.
    """
    proc = subprocess.run(
        [PY, "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    output = proc.stdout or ""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return proc.returncode, (lines[-1] if lines else "")


def main() -> int:
    print("=" * 80)
    print("TARGETED MUTATION AUDIT - reinstating bugs that were really shipped")
    print("=" * 80)

    killed: list[str] = []
    survived: list[tuple[str, str, str]] = []
    broken: list[tuple[str, str]] = []

    for label, rel, original, mutated, target in MUTATIONS:
        path = ROOT / rel
        if not path.exists():
            broken.append((label, "file missing: " + rel))
            continue
        source = path.read_text(encoding="utf-8")
        count = source.count(original)
        if count != 1:
            broken.append((label, f"anchor found {count}x, need exactly 1"))
            continue

        path.write_text(source.replace(original, mutated), encoding="utf-8")
        try:
            code, summary = run(target)
        except subprocess.TimeoutExpired:
            code, summary = 1, "TIMEOUT"
        finally:
            path.write_text(source, encoding="utf-8")  # always restore

        if code == 0:
            survived.append((label, target, summary))
            print("  [SURVIVED  tests do not bite ] " + label)
        else:
            killed.append(label)
            print("  [killed                        ] " + label)

    runnable = len(killed) + len(survived)
    print()
    print("-" * 80)
    print(f"killed    : {len(killed)}/{runnable}")
    print(f"survived  : {len(survived)}")
    print(f"unrunnable: {len(broken)}")
    print(f"MUTATION SCORE: {100 * len(killed) / max(1, runnable):.1f}%")
    print("-" * 80)

    if survived:
        print()
        print("SURVIVORS (the suite does not detect these):")
        for label, target, summary in survived:
            print("  * " + label)
            print("      target: " + target)
            print("      " + summary[:80])
    if broken:
        print()
        print("UNRUNNABLE (anchor needs re-targeting):")
        for label, why in broken:
            print("  * " + label + ": " + why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
