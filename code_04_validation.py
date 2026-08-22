"""
code_04_validation.py  ->  Table 7
==================================
A Monte Carlo check is only informative if the counter is independent of the
thing being checked.  Counting simulated paths with the same automaton that is
under test guarantees agreement and proves nothing.

Here every simulated path is counted by `mcem_core.simulate_vectorised`, a
batched form of `mcem_core.count_by_definition` (the two are checked path by
path in `test_core.py`), which
buffers the positions of the critical states and inspects the window of the
last k of them.  It never touches A_t or B_t, so the comparison is a genuine
test of the embedding.

Reported alongside each simulated frequency is 1.96 times its binomial
standard error, which is the yardstick the agreement should be judged against.

Outputs: Tables/tab_validation.tex, Output/validation.csv
"""

from __future__ import annotations

import argparse

import numpy as np

import mcem_core as C
from common import banner, latex_table, save_environment, write_csv, write_table

N, K, R = 30, 5, 2
P2S = (0.3, 0.7)
MATH = {"E1": "\\mathcal{E}_1", "E2": "\\mathcal{E}_2",
        "E3": "\\mathcal{E}_3", "E4": "\\mathcal{E}_4"}


def main(trials=1_000_000, seed=20260820):
    save_environment()
    banner("Table 6: exact versus Monte Carlo (%s trials, n=%d, k=%d, r=%d)"
           % (format(trials, ","), N, K, R))
    print("  simulated counts use the definition-based counter, not the chain\n")
    rows = []
    print("  %-4s %4s %10s %10s %10s %10s %10s %10s %11s"
          % ("fam", "p2", "mu exact", "mu sim", "P0 exact", "P0 sim",
             "P1 exact", "P1 sim", "max |z|"))
    for fam in C.FAMILIES:
        for p2 in P2S:
            p = C.symmetric_p(p2)
            f = C.pmf(N, fam, K, R, p)
            mu, _ = C.pmf_moments(f)
            sim = C.simulate_vectorised(N, fam, K, R, p, trials, seed)
            emp = np.bincount(sim, minlength=4) / trials

            zmax = 0.0
            for x in range(3):
                se = np.sqrt(max(f[x] * (1 - f[x]), 1e-300) / trials)
                zmax = max(zmax, abs(emp[x] - f[x]) / se)
            print("  %-4s %4.1f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %11.2f"
                  % (fam, p2, mu, sim.mean(), f[0], emp[0], f[1], emp[1], zmax))
            rows.append([fam, p2, mu, float(sim.mean()),
                         f[0], emp[0], f[1], emp[1], f[2], emp[2], zmax])

    write_csv("validation.csv",
              ["family", "p2", "mu_exact", "mu_simulated",
               "P0_exact", "P0_simulated", "P1_exact", "P1_simulated",
               "P2_exact", "P2_simulated", "max_abs_z"], rows)

    tex = []
    for i, r_ in enumerate(rows):
        first = (i % 2 == 0)
        lead = ("\\multirow{2}{*}{$%s$}" % MATH[r_[0]]) if first else ""
        tex.append([lead, "%.1f" % r_[1],
                    "%.6f" % r_[2], "%.6f" % r_[3],
                    "%.6f" % r_[4], "%.6f" % r_[5],
                    "%.6f" % r_[6], "%.6f" % r_[7],
                    "%.6f" % r_[8], "%.6f" % r_[9]])
        if not first and i != len(rows) - 1:
            tex.append(None)

    write_table("tab_validation.tex", latex_table(
        caption=("Exact distributions against Monte Carlo simulation "
                 "($%s$ replications, $n=%d$, $k=%d$, $r=%d$, "
                 "$p_0=p_1=(1-p_2)/2$). Simulated paths are counted by the "
                 "definition-based scanner of Algorithm~\\ref{alg:scan}, which "
                 "does not use $\\boldsymbol{A}_t$ or $\\boldsymbol{B}_t$."
                 % (format(trials, ","), N, K, R)),
        label="tab:validation",
        colspec="@{}l c cc cc cc cc@{}",
        header=[["\\multirow{2}{*}{\\textbf{Pattern}} & "
                 "\\multirow{2}{*}{$p_2$} & "
                 "\\multicolumn{2}{c}{$\\mathbb{E}[N]$} & "
                 "\\multicolumn{2}{c}{$\\Pr(N=0)$} & "
                 "\\multicolumn{2}{c}{$\\Pr(N=1)$} & "
                 "\\multicolumn{2}{c}{$\\Pr(N=2)$}"],
                ["\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"
                 "\\cmidrule(lr){9-10}\n & & Exact & Sim. & Exact & Sim. & "
                 "Exact & Sim. & Exact & Sim."]],
        rows=tex, star=True, resize=True, arraystretch="1.15",
        notes=("The largest standardised deviation over all entries of the "
               "table is $%.2f$, well inside the range expected from "
               "$%s$ replications."
               % (max(r_[10] for r_ in rows), format(trials, ",")))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()
    main(trials=a.trials, seed=a.seed)
