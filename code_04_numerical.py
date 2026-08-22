"""
code_04_numerical.py  ->  Tables 5-6 and Figures 1-2
====================================================
What the exact distributions look like, and what changes when the tolerance r
or the degradation intensity p2 changes.

Everything here comes from the recursion of Proposition 2, with the matrix form
used once as a check.  Moments come from differentiating the conditional
generating function at z = 1, which costs one pass and never needs the mass
function.

Outputs: Tables/tab_moments.tex, Tables/tab_shape.tex,
         Figures/fig1_pmf.png, Figures/fig2_tolerance.png,
         Output/moments.csv, Output/shape.csv
"""

from __future__ import annotations

import numpy as np

import embedding as E
from common import (banner, figure_path, latex_table, save_environment,
                    write_csv, write_table)

N, K, R = 50, 3, 2
P2S = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
LABEL = {"E1": r"$\mathcal{E}_1$ strict, non-overlapping",
         "E2": r"$\mathcal{E}_2$ threshold, non-overlapping",
         "E3": r"$\mathcal{E}_3$ strict, overlapping",
         "E4": r"$\mathcal{E}_4$ threshold, overlapping"}


def moments_table(n=N, k=K, r=R, p2s=P2S):
    banner("Table 5: mean and variance at n=%d, k=%d, r=%d" % (n, k, r))
    print("  %5s %s" % ("p2", "  ".join("%-18s" % f for f in E.FAMILIES)))
    rows, tex = [], []
    for p2 in p2s:
        p = E.symmetric_p(p2)
        cells, rec = [], []
        for family in E.FAMILIES:
            mu, var = E.moments(n, family, k, r, p)
            cells.append("%8.4f %8.4f" % (mu, var))
            rec += [mu, var]
        print("  %5.2f %s" % (p2, "  ".join(cells)))
        rows.append([p2] + rec)
        tex.append(["%.2f" % p2] + ["%.4f" % v for v in rec])
    write_csv("moments.csv",
              ["p2"] + [f + s for f in E.FAMILIES
                        for s in ("_mean", "_var")], rows)

    write_table("tab_moments.tex", latex_table(
        caption=("Mean and variance of $N^{(i)}_{n,k,r}$ at $n=%d$, $k=%d$, "
                 "$r=%d$, with $p_0=p_1=(1-p_2)/2$. Both follow from "
                 "differentiating the conditional generating function at "
                 "$z=1$, so neither requires the mass function."
                 % (n, k, r)),
        label="tab:moments",
        colspec="c cc cc cc cc",
        header=[["\\multirow{2}{*}{$p_2$} & "
                 "\\multicolumn{2}{c}{$\\mathcal{E}_1$} & "
                 "\\multicolumn{2}{c}{$\\mathcal{E}_2$} & "
                 "\\multicolumn{2}{c}{$\\mathcal{E}_3$} & "
                 "\\multicolumn{2}{c}{$\\mathcal{E}_4$}"],
                ["\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
                 "\\cmidrule(lr){8-9}\n"
                 " & Mean & Var. & Mean & Var. & Mean & Var. & Mean & Var."]],
        rows=tex, star=True, resize=True,
        notes=("The two strict families rise with $p_2$ throughout. The "
               "threshold families do not: they need a gap of at least $r$ "
               "warning states somewhere in the window, and warning states "
               "become scarce as $p_2$ grows, so their means turn over.")))
    return rows


def shape_table(n=N, k=K, r=R, p2=0.5):
    """Dispersion and the atom at zero, which is where the families differ."""
    banner("Table 6: shape at p2=%.2f" % p2)
    p = E.symmetric_p(p2)
    rows, tex = [], []
    print("  %-4s %10s %10s %10s %10s %10s"
          % ("fam", "mean", "var", "var/mean", "P(N=0)", "mode"))
    for family in E.FAMILIES:
        f = E.pmf(n, family, k, r, p)
        mu, var = E.pmf_moments(f)
        mode = int(np.argmax(f))
        ratio = var / mu if mu > 0 else float("nan")
        print("  %-4s %10.4f %10.4f %10.4f %10.6f %10d"
              % (family, mu, var, ratio, f[0], mode))
        rows.append([family, n, k, r, p2, mu, var, ratio, float(f[0]), mode])
        tex.append(["$\\mathcal{E}_%s$" % family[1], "%.4f" % mu,
                    "%.4f" % var, "%.3f" % ratio, "%.6f" % f[0], "%d" % mode])
    write_csv("shape.csv",
              ["family", "n", "k", "r", "p2", "mean", "variance",
               "var_over_mean", "P0", "mode"], rows)

    write_table("tab_shape.tex", latex_table(
        caption=("Shape of the four distributions at $n=%d$, $k=%d$, $r=%d$, "
                 "$p_2=%.2f$." % (n, k, r, p2)),
        label="tab:shape",
        colspec="l rrr rr",
        header=[["Family", "Mean", "Variance", "Var/Mean", "$\\Pr(N=0)$",
                 "Mode"]],
        rows=tex,
        notes=("A variance-to-mean ratio far from one rules out a Poisson "
               "approximation. The overlapping families are the most "
               "dispersed, because one long stretch of critical states "
               "contributes many occurrences at once rather than one.")))
    return rows


def plot_pmf(n=N, k=K, r=R, p2s=(0.3, 0.5, 0.7)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    for ax, family in zip(axes.ravel(), E.FAMILIES):
        for p2 in p2s:
            f = E.pmf(n, family, k, r, E.symmetric_p(p2))
            keep = np.nonzero(f > 1e-6)[0]
            hi = int(keep.max()) if keep.size else 0
            ax.plot(np.arange(hi + 1), f[:hi + 1], marker="o", ms=3, lw=1.3,
                    label=r"$p_2=%.1f$" % p2)
        ax.set_title(LABEL[family], fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\Pr(N=x)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(figure_path("fig1_pmf.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig1_pmf.png")


def plot_tolerance(n=N, k=K, rs=(0, 1, 2, 3), p2s=P2S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for family, ax in (("E1", axes[0]), ("E2", axes[1])):
        for r in rs:
            ys = [E.moments(n, family, k, r, E.symmetric_p(p2))[0]
                  for p2 in p2s]
            ax.plot(p2s, ys, marker="o", ms=3.6, lw=1.4, label="$r=%d$" % r)
        ax.set_xlabel("$p_2$")
        ax.set_ylabel(r"$\mathbb{E}[N]$")
        ax.set_title(LABEL[family], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(figure_path("fig2_tolerance.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig2_tolerance.png")


def main():
    save_environment()
    moments_table()
    shape_table()
    banner("Figures")
    plot_pmf()
    plot_tolerance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
