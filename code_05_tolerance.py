"""
code_05_tolerance.py  ->  Tables 9 and 10
========================================
A tempting comparison is the multi-state framework against a "binary
collapsed" monitoring scheme, read as evidence that binary aggregation
underestimates alarm activity.  Section A below shows why that reading is
unavailable: the binary column is numerically identical to the multi-state
pattern E1 with the same k and r = 0, because collapsing {0,1} to non-critical
and counting runs of k critical states is exactly the r = 0 member of the same
family.  Nothing is lost by the aggregation; the two columns differ because two
different patterns are being counted, and the more permissive one must count at
least as often on every path.  The comparison is therefore reported for what it
is - a study of the interruption tolerance r - and no claim about information
loss is made.

Section C then compares the schemes the way monitoring rules have to be
compared: at a common horizon, with the false-alarm probability under a
healthy profile and the detection probability under a degraded profile, plus
the expected time to the first alarm.  On that footing the tolerant scheme is
not uniformly preferable, which is the honest finding.

Outputs: Tables/tab_tolerance.tex, Tables/tab_operating.tex,
         Output/tolerance.csv, Output/operating_characteristics.csv
"""

from __future__ import annotations

import numpy as np

import mcem_core as C
from common import banner, latex_table, save_environment, write_csv, write_table

N, K = 50, 2
P2S = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
HEALTHY, DEGRADED = 0.05, 0.25


def survival_to_first_alarm(nmax, family, k, r, p):
    """P(T > t) for t = 0..nmax, T the trial index of the first occurrence."""
    A, _B = C.build_matrices(family, k, r, p)
    v = C.initial_vector(family, k, r)
    surv = np.ones(nmax + 1)
    for t in range(1, nmax + 1):
        v = v @ A
        surv[t] = v.sum()
    return surv


def main():
    save_environment()

    banner("A. Identity: the 'binary collapsed' scheme is E1 with r = 0")
    worst = 0.0
    for p2 in P2S:
        three = C.pmf(N, "E1", K, 0, C.symmetric_p(p2))
        binary = C.pmf(N, "E1", K, 0, (1.0 - p2, 0.0, p2))
        worst = max(worst, float(np.max(np.abs(three - binary))))
    print("  max |PMF(3-state, r=0) - PMF(collapsed Bernoulli, r=0)| = %.3e"
          % worst)
    print("  the two are the same distribution; the binary column of the first")
    print("  submission was the r = 0 member of the proposed family.\n")

    banner("B. Table 7: interruption tolerance r = 1 against r = 0 (n=%d, k=%d)"
           % (N, K))
    rows = []
    print("  %5s %10s %10s %12s %12s %12s %12s"
          % ("p2", "E[N] r=1", "E[N] r=0", "P0 r=1", "P0 r=0",
             "P(alarm) r=1", "r=0"))
    for p2 in P2S:
        p = C.symmetric_p(p2)
        a = C.pmf(N, "E1", K, 1, p)
        b = C.pmf(N, "E1", K, 0, p)
        ma, _ = C.pmf_moments(a)
        mb, _ = C.pmf_moments(b)
        print("  %5.2f %10.4f %10.4f %12.3e %12.3e %12.6f %12.6f"
              % (p2, ma, mb, a[0], b[0], 1 - a[0], 1 - b[0]))
        rows.append([p2, ma, mb, a[0], b[0], 1 - a[0], 1 - b[0]])
    write_csv("tolerance.csv",
              ["p2", "E_r1", "E_r0", "P0_r1", "P0_r0",
               "P_alarm_r1", "P_alarm_r0"], rows)

    write_table("tab_tolerance.tex", latex_table(
        caption=("Effect of the interruption tolerance $r$ on "
                 "$\\mathcal{E}_1$ ($n=%d$, $k=%d$, $p_0=p_1=(1-p_2)/2$). "
                 "The $r=0$ columns coincide with non-overlapping runs of $k$ "
                 "critical states in the binary-collapsed sequence."
                 % (N, K)),
        label="tab:tolerance",
        colspec="c cc cc cc",
        header=[["$p_2$",
                 "$\\mathbb{E}[N]_{r=1}$", "$\\mathbb{E}[N]_{r=0}$",
                 "$\\Pr(N=0)_{r=1}$", "$\\Pr(N=0)_{r=0}$",
                 "$\\Pr(\\text{alarm})_{r=1}$", "$\\Pr(\\text{alarm})_{r=0}$"]],
        rows=[["%.2f" % r_[0], "%.4f" % r_[1], "%.4f" % r_[2],
               "%.4g" % r_[3], "%.4g" % r_[4],
               "%.6f" % r_[5], "%.6f" % r_[6]] for r_ in rows],
        notes=("Tolerating a single warning state between two critical states "
               "can only increase the count on any given path, so the ordering "
               "of the two columns is a property of the definitions and not an "
               "empirical finding. Table~\\ref{tab:operating} compares the two "
               "schemes at matched operating characteristics.")))

    banner("C. Table 8: alarm operating characteristics")
    print("  healthy p2 = %.2f, degraded p2 = %.2f, horizon n = %d"
          % (HEALTHY, DEGRADED, N))
    print("  %-14s %10s %10s %10s %12s"
          % ("scheme", "P(FA)", "P(detect)", "Youden J", "E[min(T,n)]"))
    oc = []
    for k in (2, 3):
        for r in (0, 1, 2):
            pfa = 1 - C.pmf(N, "E1", k, r, C.symmetric_p(HEALTHY))[0]
            pdet = 1 - C.pmf(N, "E1", k, r, C.symmetric_p(DEGRADED))[0]
            surv = survival_to_first_alarm(N, "E1", k, r, C.symmetric_p(DEGRADED))
            et = float(np.sum(surv[:-1]))
            lab = "k=%d, r=%d" % (k, r)
            print("  %-14s %10.5f %10.5f %10.5f %12.2f"
                  % (lab + (" (binary)" if r == 0 else ""), pfa, pdet,
                     pdet - pfa, et))
            oc.append([k, r, pfa, pdet, pdet - pfa, et])
    write_csv("operating_characteristics.csv",
              ["k", "r", "P_false_alarm", "P_detect", "youden_J",
               "E_min_T_n"], oc)

    best = {}
    for k in (2, 3):
        cand = [o for o in oc if o[0] == k]
        best[k] = max(cand, key=lambda o: o[4])
    write_table("tab_operating.tex", latex_table(
        caption=("Alarm operating characteristics of $\\mathcal{E}_1$ over a "
                 "horizon of $n=%d$ trials: false-alarm probability under a "
                 "healthy profile ($p_2=%.2f$), detection probability under a "
                 "degraded profile ($p_2=%.2f$), Youden's $J$, and the expected "
                 "time to the first alarm under degradation."
                 % (N, HEALTHY, DEGRADED)),
        label="tab:operating",
        colspec="cc rrr r",
        header=[["$k$", "$r$", "$\\Pr(\\text{false alarm})$",
                 "$\\Pr(\\text{detect})$", "$J$",
                 "$\\mathbb{E}[\\min(T_1,n)]$"]],
        rows=[["%d" % o[0], "%d" % o[1], "%.5f" % o[2], "%.5f" % o[3],
               "%.5f" % o[4], "%.2f" % o[5]] for o in oc],
        notes=("Increasing $r$ raises detection and false alarms together. At "
               "$k=2$ the net effect on $J$ is slightly negative "
               "($%.5f$ at $r=0$ against $%.5f$ at $r=1$), whereas at $k=3$ "
               "tolerance is clearly beneficial ($%.5f$ at $r=0$ against "
               "$%.5f$ at $r=2$). Interruption tolerance is therefore useful "
               "for longer patterns rather than universally."
               % (oc[0][4], oc[1][4], oc[3][4], oc[5][4]))))
    return rows, oc


if __name__ == "__main__":
    main()
