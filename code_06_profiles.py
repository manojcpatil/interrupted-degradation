"""
code_06_profiles.py  ->  Table 11
================================
Expected value and variance of the number of E1 occurrences (k=3, r=1) for the
three CNC operating profiles over horizons n = 50, 100, 200, 500.

Three independent routes to the same numbers are cross-checked, in place of
asserting the agreement:

  * moments of the full exact PMF,
  * the O(n s^2) vector moment recursion of Proposition 2 (any k, r, family),
  * for k=2, r=1 only, the scalar recursion of Corollary 3.

The claim that "mean and variance scale linearly with n" is turned into a
measurement: a straight line is fitted to mu_n over the reported horizons and
the largest residual is printed, together with the variance-to-mean ratio,
which is what actually distinguishes the three profiles.

Outputs: Tables/tab_profiles.tex, Output/profiles.csv
"""

from __future__ import annotations

import numpy as np

import mcem_core as C
from common import banner, latex_table, save_environment, write_csv, write_table

K, R = 3, 1
PROFILES = [("Profile 1", 0.05), ("Profile 2", 0.15), ("Profile 3", 0.30)]
HORIZONS = (50, 100, 200, 500)


def main():
    save_environment()

    banner("A. Cross-check of the three moment routines")
    worst = 0.0
    for p2 in (0.1, 0.3, 0.5, 0.7):
        p = C.symmetric_p(p2)
        for n in (10, 50, 200):
            a = C.pmf_moments(C.pmf(n, "E1", 2, 1, p))
            b = C.moments(n, "E1", 2, 1, p)
            c = C.moments_scalar_k2r1(n, p)
            worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]),
                        abs(a[0] - c[0]), abs(a[1] - c[1]))
    print("  max discrepancy between PMF moments, Proposition 2 and "
          "Corollary 3: %.3e" % worst)
    for fam in C.FAMILIES:
        w = 0.0
        for p2 in (0.2, 0.6):
            for n in (20, 60):
                a = C.pmf_moments(C.pmf(n, fam, 3, 2, C.symmetric_p(p2)))
                b = C.moments(n, fam, 3, 2, C.symmetric_p(p2))
                w = max(w, abs(a[0] - b[0]), abs(a[1] - b[1]))
        print("  %s: Proposition 2 against the exact PMF, max diff %.3e"
              % (fam, w))

    banner("B. Table 9: profile sensitivity (E1, k=%d, r=%d)" % (K, R))
    rows, fits = [], []
    print("  %-10s %6s %10s %10s %10s" % ("profile", "n", "mean", "variance",
                                          "var/mean"))
    for name, p2 in PROFILES:
        p = C.symmetric_p(p2)
        mus = []
        for n in HORIZONS:
            mu, var = C.moments(n, "E1", K, R, p)
            mu_chk, var_chk = C.pmf_moments(C.pmf(n, "E1", K, R, p))
            assert abs(mu - mu_chk) < 1e-9 and abs(var - var_chk) < 1e-9
            mus.append(mu)
            rows.append([name, p2, n, mu, var, var / mu])
            print("  %-10s %6d %10.4f %10.4f %10.4f" % (name, n, mu, var,
                                                        var / mu))
        slope, intercept = np.polyfit(HORIZONS, mus, 1)
        resid = float(np.max(np.abs(np.asarray(mus)
                                    - (slope * np.asarray(HORIZONS) + intercept))))
        fits.append((name, slope, intercept, resid))
        print("       fit mu_n = %.6f n %+.6f, max residual %.2e"
              % (slope, intercept, resid))
    write_csv("profiles.csv",
              ["profile", "p2", "n", "mean", "variance", "var_over_mean"], rows)

    tex = []
    for name, p2 in PROFILES:
        block = [r_ for r_ in rows if r_[0] == name]
        tex.append(["\\multirow{3}{*}{\\shortstack[l]{\\textbf{%s}\\\\($p_2=%.2f$)}}"
                    % (name, p2), "Mean $\\mu_n$"]
                   + ["%.4f" % r_[3] for r_ in block])
        tex.append(["", "Variance $\\sigma_n^2$"]
                   + ["%.4f" % r_[4] for r_ in block])
        tex.append(["", "$\\sigma_n^2/\\mu_n$"]
                   + ["%.4f" % r_[5] for r_ in block])
        tex.append(None)
    tex = tex[:-1]

    write_table("tab_profiles.tex", latex_table(
        caption=("Expected value, variance and variance-to-mean ratio of the "
                 "number of $\\mathcal{E}_1$ occurrences ($k=%d$, $r=%d$) for "
                 "the three operating profiles." % (K, R)),
        label="tab:profiles",
        colspec="@{}ll cccc@{}",
        header=[["\\multirow{2}{*}{\\textbf{System profile}} & "
                 "\\multirow{2}{*}{\\textbf{Metric}} & "
                 "\\multicolumn{4}{c}{\\textbf{Horizon} $n$}"],
                ["\\cmidrule(l){3-6}\n & & "
                 + " & ".join("\\textbf{%d}" % n for n in HORIZONS)]],
        rows=tex, small=False,
        notes=("A straight line fitted to $\\mu_n$ over these horizons leaves a "
               "maximum residual of %s respectively, so the growth is linear to "
               "numerical precision. The variance-to-mean ratio falls from "
               "$%.4f$ to $%.4f$ as degradation intensifies, reflecting the "
               "negative dependence induced by consuming a completed pattern."
               % (", ".join("$%.0e$" % f[3] for f in fits),
                  rows[0][5], rows[-1][5]))))
    return rows


if __name__ == "__main__":
    main()
