"""
code_03_verify.py  ->  Table 4
==============================
Three checks on the embedding, each with a comparator that does not use the
transition matrices.

  1. Every sequence in {N,W,C}^n is counted twice, once by replaying the chain
     and once by a scanner written from the definition, and the two counts are
     compared path by path.
  2. All 3^n paths are enumerated and weighted by their probabilities, giving
     the mass function without any embedding.
  3. A million simulated trajectories are counted by that same scanner.

The point of all three is the same.  An error in a transition matrix produces a
perfectly well-behaved distribution of the wrong random variable, and no check
that uses the matrices can detect it.

The Monte Carlo run is split into batches with seeds derived from one master
seed, and the whole table is summarised by a single pooled chi-square rather
than by a collection of per-cell scores.

Outputs: Tables/tab_verify.tex, Output/verify.csv, Output/gof.csv
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

import embedding as E
from common import (banner, latex_table, save_environment, sci_latex,
                    write_csv, write_table)

AUDIT_GRID = ((2, 1), (3, 1), (3, 2), (4, 2))
AUDIT_NS = (9, 10)
BF_N, BF_P2 = 10, 0.40
MC_N, MC_K, MC_R = 30, 5, 2
MC_P2S = (0.3, 0.7)
BATCHES = 20
MIN_EXPECTED = 5.0


def path_audit(grid=AUDIT_GRID, ns=AUDIT_NS):
    """Chain replay against the definition-based scanner, on every path."""
    rows = []
    for n in ns:
        seqs = list(itertools.product((0, 1, 2), repeat=n))
        for k, r in grid:
            for family in E.FAMILIES:
                bad = sum(1 for s in seqs
                          if E.count_occurrences(s, family, k, r)
                          != E.count_by_definition(s, family, k, r))
                rows.append([family, k, r, n, 3 ** n, bad])
        print("  n=%2d: %d families x %d settings, %d paths each"
              % (n, len(E.FAMILIES), len(grid), 3 ** n))
    bad = sum(r[5] for r in rows)
    print("  disagreeing paths over the whole grid: %d" % bad)
    return rows, bad


def bruteforce(n=BF_N, p2=BF_P2, grid=((2, 1), (3, 2))):
    """Mass function by weighted enumeration, with no embedding involved."""
    rows, worst = [], 0.0
    p = E.symmetric_p(p2)
    for k, r in grid:
        for family in E.FAMILIES:
            a = E.pmf(n, family, k, r, p)
            b = E.pmf_bruteforce(n, family, k, r, p)
            m = min(len(a), len(b))
            d = float(np.max(np.abs(a[:m] - b[:m])))
            worst = max(worst, d)
            rows.append([family, k, r, n, p2, d])
    print("  largest difference against brute force: %.2e" % worst)
    return rows, worst


def batch_seeds(master, batches=BATCHES):
    """Independent batch seeds.

    `master` may be a single integer or a sequence of integers. Passing the
    test index alongside the master seed gives every (family, profile) pair its
    own independent stream while keeping the whole design reproducible from one
    recorded number. That independence is what lets the per-test chi-square
    statistics be pooled: summing them, and summing their degrees of freedom,
    is only valid for independent components.
    """
    ss = np.random.SeedSequence(master)
    return [int(s.generate_state(1)[0]) for s in ss.spawn(batches)]


def chi_square(hist, f, total, min_expected=MIN_EXPECTED):
    L = max(len(hist), len(f))
    obs, exp = np.zeros(L), np.zeros(L)
    obs[:len(hist)] = hist
    exp[:len(f)] = np.asarray(f) * total
    keep = exp >= min_expected
    o, e = list(obs[keep]), list(exp[keep])
    o_t, e_t = float(obs[~keep].sum()), float(exp[~keep].sum())
    if e_t > 0:
        o.append(o_t)
        e.append(e_t)
    o, e = np.asarray(o), np.asarray(e)
    return float(np.sum((o - e) ** 2 / e)), len(o) - 1


def chi_square_pvalue(x2, df):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x2, df))
    except Exception:
        from math import erfc, sqrt
        return float(0.5 * erfc((x2 - df) / sqrt(4.0 * df)))


def monte_carlo(trials=1_000_000, seed=20260820):
    """Simulated paths counted by the scanner, never by the chain."""
    per = trials // BATCHES
    rows, x2_tot, df_tot = [], 0.0, 0
    print("  %-4s %4s %11s %11s %9s %8s"
          % ("fam", "p2", "E[N] exact", "E[N] sim", "max |z|", "X2/df"))
    for fi, family in enumerate(E.FAMILIES):
        for pi, p2 in enumerate(MC_P2S):
            p = E.symmetric_p(p2)
            f = E.pmf(MC_N, family, MC_K, MC_R, p)
            mu, _ = E.pmf_moments(f)
            hist = None
            means = []
            # A distinct stream per (family, profile): see batch_seeds.
            for sd in batch_seeds([seed, fi, pi]):
                sim = E.simulate_vectorised(MC_N, family, MC_K, MC_R, p,
                                            per, sd)
                h = np.bincount(sim).astype(np.int64)
                if hist is None:
                    hist = h
                else:
                    if len(h) > len(hist):
                        hist = np.concatenate(
                            [hist, np.zeros(len(h) - len(hist), dtype=np.int64)])
                    hist[:len(h)] += h
                means.append(float(sim.mean()))
            total = per * BATCHES
            emp = hist / total
            zs = []
            for x in range(3):
                se = np.sqrt(max(f[x] * (1 - f[x]), 1e-300) / total)
                zs.append(abs(float(emp[x]) - f[x]) / se)
            x2, df = chi_square(hist, f, total)
            x2_tot += x2
            df_tot += df
            print("  %-4s %4.1f %11.6f %11.6f %9.2f %8.2f"
                  % (family, p2, mu, float(np.mean(means)), max(zs), x2 / df))
            rows.append([family, p2, mu, float(np.mean(means)),
                         f[0], float(emp[0]), f[1], float(emp[1]),
                         max(zs), x2, df])
    return rows, x2_tot, df_tot, per


def main(trials=1_000_000, seed=20260820):
    save_environment()

    banner("A. Every path in {N,W,C}^n, chain against the definition")
    audit, bad = path_audit()

    banner("B. Mass function against weighted enumeration of all 3^%d paths"
           % BF_N)
    bf, bf_worst = bruteforce()

    banner("C. Monte Carlo, counted by the scanner")
    mc, x2_tot, df_tot, per = monte_carlo(trials, seed)
    pval = chi_square_pvalue(x2_tot, df_tot)
    print("\n  pooled goodness of fit: X2 = %.1f on %d df, p = %.3f"
          % (x2_tot, df_tot, pval))

    write_csv("verify.csv",
              ["family", "p2", "mu_exact", "mu_sim", "P0_exact", "P0_sim",
               "P1_exact", "P1_sim", "max_abs_z", "chi_square", "df"], mc)
    write_csv("gof.csv", ["quantity", "value"],
              [["paths_compared", sum(r[4] for r in audit)],
               ["disagreeing_paths", bad],
               ["max_diff_bruteforce", bf_worst],
               ["chi_square", x2_tot], ["df", df_tot], ["p_value", pval]])

    tex = []
    for r_ in mc:
        tex.append(["$\\mathcal{E}_%s$" % r_[0][1], "%.1f" % r_[1],
                    "%.6f" % r_[2], "%.6f" % r_[3],
                    "%.6f" % r_[4], "%.6f" % r_[5],
                    "%.6f" % r_[6], "%.6f" % r_[7], "%.2f" % r_[8]])

    write_table("tab_verify.tex", latex_table(
        caption=("Exact distributions against simulation at $n=%d$, $k=%d$, "
                 "$r=%d$, with $p_0=p_1=(1-p_2)/2$. The run is $%s$ paths in "
                 "$%d$ batches of $%s$. Every family and profile draws its own "
                 "independent stream, all spawned from a master seed of $%d$. "
                 "Simulated paths are counted by the scanner of "
                 "Algorithm~\\ref{alg:scan}, which never touches "
                 "$\\boldsymbol{A}_t$ or $\\boldsymbol{B}_t$."
                 % (MC_N, MC_K, MC_R, format(per * BATCHES, ","), BATCHES,
                    format(per, ","), seed)),
        label="tab:verify",
        colspec="l c cc cc cc c",
        header=[["\\multirow{2}{*}{Family} & \\multirow{2}{*}{$p_2$} & "
                 "\\multicolumn{2}{c}{$\\mathbb{E}[N]$} & "
                 "\\multicolumn{2}{c}{$\\Pr(N=0)$} & "
                 "\\multicolumn{2}{c}{$\\Pr(N=1)$} & "
                 "\\multirow{2}{*}{$\\max|z|$}"],
                ["\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
                 " & & Exact & Sim. & Exact & Sim. & Exact & Sim. &"]],
        rows=tex, star=True, resize=True,
        notes=("Two further checks sit behind this table. Every sequence in "
               "$\\{\\N,\\W,\\Cc\\}^{n}$ for $n=9$ and $n=10$, over four "
               "$(k,r)$ settings and all four families, was counted once by "
               "replaying the chain and once by the scanner: $%s$ paths in "
               "total, with $%d$ disagreements. Enumerating all $3^{%d}$ paths "
               "and weighting them by their probabilities reproduces the mass "
               "function to %s. Pooling every cell of the table above into one "
               "chi-square, with cells of expected count below $%.0f$ merged "
               "into the tail, gives $X^2=%.1f$ on $%d$ degrees of freedom "
               "($p=%.2f$)."
               % (format(sum(r[4] for r in audit), ","), bad, BF_N,
                  sci_latex(bf_worst), MIN_EXPECTED, x2_tot, df_tot, pval))))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()
    raise SystemExit(main(trials=a.trials, seed=a.seed))
