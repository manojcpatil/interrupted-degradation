"""
code_07_distributions.py  ->  Tables 12, 13, 14, 15
===================================================
Exact distributions of N^{(i)}_{30,5,2}, i = 1..4, across degradation
intensities p2 in {0.1, 0.3, 0.5, 0.7, 0.9}.

Note: the E2, E3 and E4 columns are sensitive to the embedding, so they are
computed from the chains of Theorems 1 and 2 and cross-checked; and the
unimodality of each mass function is checked rather than asserted.

Every PMF is verified to sum to one and to vanish outside its support, and the
number of sign changes in the first difference is reported so that the shape
claims in the text can be checked against the numbers.

Outputs: Tables/tab_dist_E*.tex, Output/dist_E*.csv, Output/shape_summary.csv
"""

from __future__ import annotations

import numpy as np

import mcem_core as C
from common import banner, latex_table, save_environment, write_csv, write_table

N, K, R = 30, 5, 2
P2S = (0.1, 0.3, 0.5, 0.7, 0.9)
NAME = {"E1": ("strict non-overlapping", "\\mathcal{E}_1"),
        "E2": ("gap-threshold non-overlapping", "\\mathcal{E}_2"),
        "E3": ("strict overlapping", "\\mathcal{E}_3"),
        "E4": ("gap-threshold overlapping", "\\mathcal{E}_4")}


def modes(f, tol=1e-15):
    """Number of sign changes in the first difference over the support."""
    nz = np.nonzero(f > 1e-12)[0]
    if nz.size < 3:
        return 0
    g = f[nz[0]:nz[-1] + 1]
    signs = [np.sign(d) for d in np.diff(g) if abs(d) > tol]
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def main():
    save_environment()
    shape = []
    for fam in C.FAMILIES:
        banner("Table for %s (n=%d, k=%d, r=%d)" % (fam, N, K, R))
        cols, mus, vs = {}, {}, {}
        for p2 in P2S:
            f = C.pmf(N, fam, K, R, C.symmetric_p(p2))
            assert abs(f.sum() - 1.0) < 1e-9, (fam, p2, f.sum())
            assert len(f) - 1 <= C.max_count(fam, K, R, N)
            cols[p2] = f
            mus[p2], vs[p2] = C.pmf_moments(f)
            nsc = modes(f)
            shape.append([fam, p2, mus[p2], vs[p2], vs[p2] / mus[p2] if mus[p2] else np.nan,
                          nsc, "unimodal" if nsc <= 1 else "multimodal"])
        xmax = max(int(np.max(np.nonzero(c > 1e-12)[0])) for c in cols.values())

        rows = [[x] + [float(cols[p2][x]) if x < len(cols[p2]) else 0.0
                       for p2 in P2S] for x in range(xmax + 1)]
        rows += [["mean"] + [mus[p2] for p2 in P2S],
                 ["variance"] + [vs[p2] for p2 in P2S]]
        write_csv("dist_%s.csv" % fam,
                  ["x"] + ["p2=%.1f" % p for p in P2S], rows)

        print("  %5s %12s %12s %8s %s" % ("p2", "mean", "variance", "sign ch.",
                                          "shape"))
        for p2 in P2S:
            s = [q for q in shape if q[0] == fam and q[1] == p2][0]
            print("  %5.1f %12.6f %12.6f %8d %s" % (p2, s[2], s[3], s[5], s[6]))

        tex = [[str(x)] + [C.prob_fmt(cols[p2][x] if x < len(cols[p2]) else 0.0)
                           for p2 in P2S] for x in range(xmax + 1)]
        tex += [None,
                ["\\textbf{Mean}"] + ["\\textbf{%.6f}" % mus[p2] for p2 in P2S],
                ["\\textbf{Variance}"] + ["\\textbf{%.6f}" % vs[p2] for p2 in P2S]]
        multi = [p2 for p2 in P2S
                 if [q for q in shape if q[0] == fam and q[1] == p2][0][5] > 1]
        note = "$^{*}$ probability below $10^{-6}$."
        if multi:
            note += (" The distribution is multimodal at $p_2 \\in \\{%s\\}$."
                     % ", ".join("%.1f" % m for m in multi))
        else:
            note += " The distribution is unimodal at every $p_2$ shown."
        write_table("tab_dist_%s.tex" % fam, latex_table(
            caption=("Exact distribution of the %s pattern $%s$ "
                     "($n=%d$, $k=%d$, $r=%d$, $p_0=p_1=(1-p_2)/2$)."
                     % (NAME[fam][0], NAME[fam][1], N, K, R)),
            label="tab:dist_%s" % fam,
            colspec="@{}l ccccc@{}",
            header=[["\\multirow{2}{*}{$x$} & "
                     "\\multicolumn{5}{c}{\\textbf{Degradation intensity} $p_2$}"],
                    ["\\cmidrule(lr){2-6}\n"
                     + " & ".join([""] + ["$%.1f$" % p for p in P2S])]],
            rows=tex, notes=note))

    write_csv("shape_summary.csv",
              ["family", "p2", "mean", "variance", "var_over_mean",
               "sign_changes", "shape"], shape)

    banner("Shape summary")
    bad = [s for s in shape if s[5] > 1]
    if bad:
        print("  multimodal cases: "
              + ", ".join("%s at p2=%.1f (%d sign changes)" % (s[0], s[1], s[5])
                          for s in bad))
    else:
        print("  all distributions unimodal")
    return shape


if __name__ == "__main__":
    main()
