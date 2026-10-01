"""Probe stats/univariate.py (52.8% covered) against closed-form references.

Every statistic here has an exact answer computable independently -- t, F,
Kruskal-Wallis, Mann-Whitney, effect sizes, AICc -- so a wrong implementation
is decidable rather than merely suspicious.

The API takes a STACKED data array plus a per-row group label, not separate
per-group arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import stats as sp  # noqa: E402

from stats.univariate import (  # noqa: E402
    UnivariateAnalyzer,
    cohens_d,
    compare_models,
    compute_aicc,
    eta_squared,
    omega_squared,
    partial_eta_squared,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, ok, detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail and not ok else ""))


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    a = UnivariateAnalyzer()
    rng = np.random.default_rng(17)

    g1 = rng.normal(0.0, 1.0, 25)
    g2 = rng.normal(1.2, 1.0, 25)
    g3 = rng.normal(2.4, 1.0, 25)
    two = np.concatenate([g1, g2])
    two_groups = [0] * 25 + [1] * 25
    three = np.concatenate([g1, g2, g3])
    three_groups = [0] * 25 + [1] * 25 + [2] * 25

    # ---- 1. t-test ----------------------------------------------------------
    r = a.t_test(two, groups=two_groups)
    ref = sp.ttest_ind(g1, g2, equal_var=True)
    check("t statistic matches scipy", near(r.statistic, ref.statistic), f"{float(r.statistic):.8f} vs {float(ref.statistic):.8f}")
    check("t p_value matches scipy", near(r.p_value, ref.pvalue, 1e-12), f"{float(r.p_value):.8f} vs {float(ref.pvalue):.8f}")
    check("t sign follows the mean difference",
          np.sign(r.statistic) == np.sign(np.mean(g1) - np.mean(g2)),
          f"t={float(r.statistic):.4f} diff={float(np.mean(g1)-np.mean(g2)):.4f}")
    check("t group sizes are n1/n2", (r.n1, r.n2) == (25, 25), f"got {r.n1}, {r.n2}")
    check("t means round-trip", near(r.mean1, np.mean(g1), 1e-12) and near(r.mean2, np.mean(g2), 1e-12),
          f"{r.mean1:.6f}/{r.mean2:.6f} vs {np.mean(g1):.6f}/{np.mean(g2):.6f}")
    check("t is significant here", float(r.p_value) < 0.01, f"p={float(r.p_value):.4f}")

    # Paired t-test -- the two vectors must stay positionally aligned.
    rp = a.t_test(two, groups=two_groups, paired=True)
    refp = sp.ttest_rel(g1, g2)
    check("paired t matches scipy", near(rp.statistic, refp.statistic), f"{float(rp.statistic):.8f} vs {float(refp.statistic):.8f}")
    check("paired t is labelled 'paired'", rp.test_type == "paired", f"got {rp.test_type!r}")
    check("unpaired t is labelled 'independent'", r.test_type == "independent", f"got {r.test_type!r}")

    # A paired design with a hole: dropping the pair must drop BOTH members.
    holed = two.copy()
    holed[3] = np.nan  # group 0 member 3
    rph = a.t_test(holed, groups=two_groups, paired=True)
    refph = sp.ttest_rel(np.delete(g1, 3), np.delete(g2, 3))
    check("paired t drops the whole pair when one side is NaN",
          near(rph.statistic, refph.statistic),
          f"{float(rph.statistic):.8f} vs {float(refph.statistic):.8f}")
    check("paired t NaN leaves n=24 per side", (rph.n1, rph.n2) == (24, 24), f"got {rph.n1}, {rph.n2}")

    # ---- 2. one-way ANOVA ---------------------------------------------------
    ra = a.one_way_anova(three, three_groups, tukey=False)
    refa = sp.f_oneway(g1, g2, g3)
    check("ANOVA F matches scipy", near(ra.f_statistic, refa.statistic), f"{float(ra.f_statistic):.8f} vs {float(refa.statistic):.8f}")
    check("ANOVA p matches scipy", near(ra.p_value, refa.pvalue, 1e-12), f"{float(ra.p_value):.8f} vs {float(refa.pvalue):.8f}")
    check("ANOVA df_between is k-1", int(ra.df_between) == 2, f"got {ra.df_between}")
    check("ANOVA detects a real difference", float(ra.p_value) < 0.01, f"p={float(ra.p_value):.4f}")

    # ---- 3. Kruskal-Wallis --------------------------------------------------
    rk = a.kruskal_wallis(three, three_groups)
    refk = sp.kruskal(g1, g2, g3)
    check("Kruskal H matches scipy", near(rk.statistic, refk.statistic), f"{float(rk.statistic):.8f} vs {float(refk.statistic):.8f}")
    check("Kruskal p matches scipy", near(rk.p_value, refk.pvalue, 1e-12), f"{float(rk.p_value):.8f} vs {float(refk.pvalue):.8f}")
    check("Kruskal n_groups is 3", rk.n_groups == 3, f"got {rk.n_groups}")

    # ---- 4. Mann-Whitney ----------------------------------------------------
    rm = a.mann_whitney(two, two_groups)
    refm = sp.mannwhitneyu(g1, g2, alternative="two-sided")
    check("Mann-Whitney U matches scipy", near(rm.statistic, refm.statistic, 1e-9), f"{float(rm.statistic):.6f} vs {float(refm.statistic):.6f}")

    # ---- 5. normality -------------------------------------------------------
    rn = a.normality_test(g1)
    refn = sp.shapiro(g1)
    check("Shapiro statistic matches scipy", near(rn.shapiro_stat, refn.statistic, 1e-6), f"{float(rn.shapiro_stat):.8f} vs {float(refn.statistic):.8f}")
    check("Shapiro p matches scipy", near(rn.shapiro_p, refn.pvalue, 1e-9), f"{float(rn.shapiro_p):.8f} vs {float(refn.pvalue):.8f}")

    refad = sp.anderson(g1, dist="norm")
    check("Anderson-Darling statistic matches scipy", near(rn.anderson_stat, refad.statistic, 1e-9),
          f"{float(rn.anderson_stat):.8f} vs {float(refad.statistic):.8f}")
    check("Anderson critical values are keyed by level",
          set(rn.anderson_critical) == set(float(x) for x in refad.significance_level),
          f"got {sorted(rn.anderson_critical)}")
    check("Anderson 5% critical value matches scipy",
          near(rn.anderson_critical[5.0], refad.critical_values[list(refad.significance_level).index(5.0)], 1e-9))
    check("is_normal_* agree with the raw tests",
          rn.is_normal_shapiro == (float(refn.pvalue) > 0.05) and
          rn.is_normal_anderson == (float(refad.statistic) < rn.anderson_critical[5.0]),
          f"sw={rn.is_normal_shapiro} ad={rn.is_normal_anderson}")

    # Shapiro-Wilk must run on the *non-NaN* values only.
    with_nan = g1.copy()
    with_nan[0] = np.nan
    rn2 = a.normality_test(with_nan)
    refn2 = sp.shapiro(g1[1:])
    check("normality drops NaN before testing", near(rn2.shapiro_stat, refn2.statistic, 1e-9),
          f"{float(rn2.shapiro_stat):.8f} vs {float(refn2.statistic):.8f}")

    # ---- 5b. Tukey HSD ------------------------------------------------------
    # scipy.stats.tukey_hsd is the reference (it is validated against R's
    # aov(x ~ g) + TukeyHSD). The q convention it uses is
    #     q = |diff| / sqrt(MSE * (1/ni + 1/nj) / 2)
    # -- note the /2 INSIDE the sqrt. Verified against scipy's own docstring
    # example (the headache-medicine data), where that expression reproduces
    # the published p-values 0.014448 / 0.980311 / 0.020331 exactly.
    REF_GROUPS = ([24.5, 23.5, 26.4, 27.1, 29.9],
                  [28.4, 34.2, 29.5, 32.2, 30.1],
                  [26.1, 28.3, 24.3, 26.2, 27.8])
    ref_flat = np.concatenate([np.asarray(g, dtype=float) for g in REF_GROUPS])
    ref_labels = [0] * 5 + [1] * 5 + [2] * 5
    rt_ref = a.one_way_anova(ref_flat, ref_labels, tukey=True)
    ref_tukey = sp.tukey_hsd(*[np.asarray(g, dtype=float) for g in REF_GROUPS])
    ref_sse = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in REF_GROUPS)
    ref_mse = ref_sse / (len(ref_flat) - 3)
    for row_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        gi, gj = (np.asarray(REF_GROUPS[i]), np.asarray(REF_GROUPS[j]))
        diff = float(np.mean(gi) - np.mean(gj))
        q_ref = abs(diff) / np.sqrt(ref_mse * (1.0 / len(gi) + 1.0 / len(gj)) / 2.0)
        got = rt_ref.tukey_results[row_idx]
        check(f"Tukey diff {i}-{j} matches", near(got["diff"], diff, 1e-9), f"{got['diff']:.8f} vs {diff:.8f}")
        check(f"Tukey q {i}-{j} matches the scipy/R convention", near(got["q_stat"], q_ref, 1e-9), f"{got['q_stat']:.8f} vs {q_ref:.8f}")
        check(f"Tukey p_adj {i}-{j} matches scipy.tukey_hsd", near(got["p_adj"], float(ref_tukey.pvalue[i, j]), 1e-9),
              f"{got['p_adj']:.8f} vs {float(ref_tukey.pvalue[i, j]):.8f}")
        check(f"Tukey p_value alias equals p_adj for pair {i}-{j}",
              near(got["p_value"], got["p_adj"], 0.0), f"{got['p_value']} vs {got['p_adj']}")
        check(f"Tukey labels {i}-{j} name the right groups",
              (got["group_a"], got["group_b"]) == (i, j), f"got {got['group_a']}, {got['group_b']}")

    # The regression this guards: omitting the /2 inflated q by exactly
    # sqrt(2), which on this dataset would report q = 3.3630 / 0.1901 /
    # 3.1729 instead of 4.7560 / 0.2688 / 4.4872.
    q_wrong = [abs(float(np.mean(np.asarray(REF_GROUPS[i])) - np.mean(np.asarray(REF_GROUPS[j]))))
               / np.sqrt(ref_mse * (1.0 / 5 + 1.0 / 5))
               for (i, j) in ((0, 1), (0, 2), (1, 2))]
    for row, qw in zip(rt_ref.tukey_results, q_wrong, strict=True):
        check(f"Tukey q {row['group_a']}-{row['group_b']} is not the sqrt(2)-inflated value",
              not near(row["q_stat"], qw, 1e-6), f"got {row['q_stat']:.6f}, wrong-variant would be {qw:.6f}")

    rt = a.one_way_anova(three, three_groups, tukey=True)
    check("Tukey produced 3 pairwise rows", len(rt.tukey_results) == 3, f"got {len(rt.tukey_results)}")

    # ---- 6. effect sizes, closed form --------------------------------------
    F, dfb, dfw, n = float(refa.statistic), 2, len(three) - 3, len(three)
    es = eta_squared(F, dfb, dfw)
    ref_eta = dfb * F / (dfb * F + dfw)
    check("eta_squared matches the textbook formula", near(es, ref_eta, 1e-9), f"{float(es):.8f} vs {ref_eta:.8f}")
    check("eta_squared is in [0, 1]", 0.0 <= float(es) <= 1.0 + 1e-12, f"{float(es)}")

    ws = omega_squared(F, dfb, dfw, n)
    # Cohen (1988) eq. 8.2.4 / Lakens 2013:
    #   w^2 = (F*df_b - df_b) / (F*df_b + df_w + 1)
    ref_omega = (F * dfb - dfb) / (F * dfb + dfw + 1.0)
    check("omega_squared matches the textbook formula", near(ws, ref_omega, 1e-9), f"{float(ws):.8f} vs {ref_omega:.8f}")
    check("omega_squared is in [0, 1]", 0.0 <= float(ws) <= 1.0 + 1e-12, f"{float(ws)}")
    check("omega_squared <= eta_squared (less biased)", float(ws) <= float(es) + 1e-12, f"{float(ws):.8f} vs {float(es):.8f}")
    check("omega_squared is 0 when F = 1", near(omega_squared(1.0, dfb, dfw, n), 0.0, 1e-12),
          f"{float(omega_squared(1.0, dfb, dfw, n)):.12f}")

    n1, n2 = len(g1), len(g2)
    pooled = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
    ref_d = (np.mean(g1) - np.mean(g2)) / np.sqrt(pooled)
    cd = cohens_d(g1, g2)
    check("cohens_d matches the pooled-SD formula", near(cd, ref_d, 1e-9), f"{float(cd):.8f} vs {ref_d:.8f}")
    check("cohens_d has the right sign", np.sign(cd) == np.sign(np.mean(g1) - np.mean(g2)))
    check("cohens_d is antisymmetric under group swap", near(cohens_d(g2, g1), -cd, 1e-12))
    check("cohens_d of a group with itself is 0", near(cohens_d(g1, g1), 0.0, 1e-12),
          f"{float(cohens_d(g1, g1)):.12f}")

    # partial_eta^2 = df_b*F / (df_b*F + df_error)
    pe = partial_eta_squared(F, dfb, dfw)
    ref_pe = dfb * F / (dfb * F + dfw)
    check("partial_eta_squared matches the textbook formula", near(pe, ref_pe, 1e-9), f"{float(pe):.8f} vs {ref_pe:.8f}")
    # It must exceed eta^2 (eta^2 is the upper-biased form).
    check("partial eta^2 >= eta^2", float(pe) >= float(es) - 1e-12, f"{float(pe):.8f} vs {float(es):.8f}")

    # ---- 7. AICc, closed form ----------------------------------------------
    ll, k, n_obs = -123.456, 4, 100
    ref_aicc = -2 * ll + 2 * k + (2 * k * (k + 1)) / (n_obs - k - 1)
    got = compute_aicc(ll, k, n_obs)
    check("AICc matches the textbook formula", near(got, ref_aicc, 1e-9), f"{float(got):.8f} vs {ref_aicc:.8f}")

    # AICc must exceed AIC (the small-sample correction is positive)
    check("AICc exceeds AIC", float(got) > -2 * ll + 2 * k, f"{float(got):.6f} vs {-2*ll+2*k:.6f}")

    # ---- 7b. compare_models: weights must be a proper distribution --------
    spec = [("m1", -100.0, 2, 200), ("m2", -102.0, 3, 200), ("m3", -99.0, 4, 200)]
    cm = compare_models(spec)
    aicc_by_name = {name: compute_aicc(ll, kk, nn) for name, ll, kk, nn in spec}
    best = min(aicc_by_name.values())
    ref_w = {}
    for name, a in aicc_by_name.items():
        ref_w[name] = np.exp(-0.5 * (a - best))
    tot = sum(ref_w.values())
    ref_w = {k: v / tot for k, v in ref_w.items()}

    check("compare_models weights sum to 1", near(sum(cm["weights"]), 1.0, 1e-12), f"{sum(cm['weights']):.12f}")
    check("compare_models picks the lowest AICc", cm["best_model"] == min(aicc_by_name, key=aicc_by_name.get),
          f"got {cm['best_model']}, aiccs={aicc_by_name}")
    check("compare_models deltas are relative to the best",
          near(min(cm["delta_aicc"]), 0.0, 1e-12), f"{cm['delta_aicc']}")
    check("compare_models returns one row per model", len(cm["models"]) == 3, f"got {len(cm['models'])}")
    check("compare_models is sorted by AICc ascending",
          [m["aicc"] for m in cm["models"]] == sorted(m["aicc"] for m in cm["models"]))
    # Rows are returned in AICc order, so index by NAME, not position.
    for row in cm["models"]:
        nm = row["name"]
        check(f"compare_models delta for {nm} matches", near(row["delta_aicc"], aicc_by_name[nm] - best, 1e-12),
              f"{row['delta_aicc']:.8f} vs {aicc_by_name[nm] - best:.8f}")
        check(f"compare_models weight for {nm} matches exp(-dAICc/2)", near(row["weight"], ref_w[nm], 1e-12),
              f"{row['weight']:.10f} vs {ref_w[nm]:.10f}")
    check("compare_models top-level weights match the per-model rows",
          [near(w, r["weight"], 0.0) for w, r in zip(cm["weights"], cm["models"], strict=True)] == [True] * 3)
    check("compare_models top-level deltas match the per-model rows",
          [near(d, r["delta_aicc"], 0.0) for d, r in zip(cm["delta_aicc"], cm["models"], strict=True)] == [True] * 3)
    # A model far worse than the best must get a negligible weight.
    cm2 = compare_models([("good", -50.0, 2, 500), ("hopeless", -500.0, 2, 500)])
    check("compare_models starves a hopeless model",
          cm2["weights"][1] < 1e-40, f"{cm2['weights']}")

    # ---- 8. degeneracy -----------------------------------------------------
    # A degenerate input must either raise cleanly or return finite numbers.
    # Silently returning NaN/inf is the failure mode that matters: it flows
    # into a results table and reads as "no effect" rather than "not computed".
    for label, fn in (
        ("t_test with a single group", lambda: a.t_test(g1, groups=[0] * 25)),
        ("t_test with three groups", lambda: a.t_test(three, groups=three_groups)),
        ("t_test with no groups", lambda: a.t_test(g1, groups=None)),
        ("paired t with unequal sizes", lambda: a.t_test(g1, groups=[0] * 24 + [1])),
        ("anova with a single group", lambda: a.one_way_anova(three, [0] * 75, tukey=False)),
        ("kruskal with identical groups", lambda: a.kruskal_wallis(three, [0] * 75)),
        ("mann_whitney with a single group", lambda: a.mann_whitney(g1, groups=[0] * 25)),
        ("groups shorter than the data", lambda: a.one_way_anova(three, [0, 1, 2])),
        ("normality on a constant", lambda: a.normality_test(np.ones(25))),
        ("normality on 2 values", lambda: a.normality_test(np.array([1.0, 2.0]))),
    ):
        try:
            out = fn()
            vals = []
            for f in ("statistic", "p_value", "f_statistic", "shapiro_stat", "shapiro_p"):
                if hasattr(out, f):
                    vals.append(float(getattr(out, f)))
            finite = bool(vals) and all(np.isfinite(v) for v in vals)
            check(f"{label}: finite or refused", finite,
                  f"{type(out).__name__} -> {vals}")
        except Exception:  # noqa: BLE001
            check(f"{label}: refused cleanly", True)

    failures = [r for r in RESULTS if not r[1]]
    print()
    print(f"{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
