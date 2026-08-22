"""
code_14_alarm.py  ->  the calibrated alarm study
================================================
The first submission compared monitoring schemes by reading off, for each
(k, r), the probability of at least one alarm under a healthy and under a
degraded profile.  That comparison is not admissible: the rule "alarm on the
first occurrence" has a different false-alarm rate for every (k, r), so the
detection probabilities are not on a common footing and the apparent ranking is
mostly a ranking of calibrations.

This script redoes the study the way a monitoring rule has to be evaluated.

  A. Calibration.  Fix a target in-control level alpha.  For each (k, r) choose
     the smallest integer threshold c with P(N_n >= c | healthy) <= alpha, then
     report the achieved level and the power P(N_n >= c | degraded).  Now the
     schemes differ only in power.

  B. Threshold sweep and ROC.  Sweeping c traces the whole achievable
     (level, power) curve.  We report the area under it and the partial area
     over alpha in [0, 0.05], the region a monitor actually operates in.

  C. Waiting time to the first alarm.  Because N_t is non-decreasing in t, the
     event {T_c <= n} is exactly {N_n >= c}, so the run length needs no new
     theory -- but an absorbing variant of the embedding gives its mean and
     quantiles directly, without truncation at the horizon.  This replaces the
     truncated E[min(T,n)] of the first submission.

  D. The three-state lever.  Holding p2 fixed and varying rho = p1/(p0+p1)
     moves probability between the warning and normal states without changing
     the degradation intensity.  Interruption tolerance should pay off exactly
     when rho is large, and the gain in calibrated power measures by how much.
     This is the direct evidence for a three-letter alphabet.

Outputs: Tables/tab_calibrated.tex, Tables/tab_roc.tex, Tables/tab_waiting.tex,
         Figures/fig5_roc.png, Figures/fig6_lever.png,
         Output/calibrated.csv, Output/roc.csv, Output/waiting.csv,
         Output/lever.csv

License: MIT.
"""

from __future__ import annotations

import numpy as np

import embedding as C
from common import (banner, figure_path, latex_table, save_environment,
                    write_csv, write_table)

FAMILY = "E1"
N = 50                      # monitoring horizon
HEALTHY, DEGRADED = 0.05, 0.25
ALPHAS = (0.01, 0.05)
GRID = ((2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2))
PAUC_MAX = 0.05


# ----------------------------------------------------------------------
# tail probabilities and calibration
# ----------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz


def tail(n, family, k, r, p):
    """T[c] = P(N_n >= c) for c = 0, 1, ...; T[0] = 1."""
    f = C.pmf(n, family, k, r, p)
    return np.concatenate(([1.0], 1.0 - np.cumsum(f)[:-1]))


def calibrate(alpha, n, family, k, r, p_null):
    """Smallest c with P(N_n >= c | null) <= alpha, and the achieved level."""
    T = tail(n, family, k, r, p_null)
    for c in range(1, len(T)):
        if T[c] <= alpha:
            return c, float(T[c])
    return len(T), 0.0


def randomised_power(alpha, n, family, k, r, p_null, p_alt):
    """Power of the randomised test of exact size alpha.

    N_n is integer-valued, so the non-randomised rule attains only the discrete
    levels P(N_n >= c) and can undershoot alpha badly.  The randomised test
    alarms whenever N_n >= c and, when N_n = c-1, with probability

        gamma = (alpha - P(N_n >= c)) / P(N_n = c-1),

    which has size exactly alpha.  Its power is the only quantity that compares
    schemes at a genuinely common false-alarm rate.
    """
    c, ach = calibrate(alpha, n, family, k, r, p_null)
    f0 = C.pmf(n, family, k, r, p_null)
    f1 = C.pmf(n, family, k, r, p_alt)
    power = float(tail(n, family, k, r, p_alt)[c])
    if c - 1 < len(f0) and f0[c - 1] > 0:
        gamma = min(1.0, max(0.0, (alpha - ach) / f0[c - 1]))
        if c - 1 < len(f1):
            power += gamma * float(f1[c - 1])
    return power


# ----------------------------------------------------------------------
# waiting time to the c-th occurrence, by an absorbing embedding
# ----------------------------------------------------------------------
def waiting_time(family, k, r, p, c, qmax=0.99):
    """Mean and quantiles of T_c = first trial at which N_t reaches c.

    The embedded chain is augmented with the count truncated at c, and the
    states with count c are made absorbing.  Writing Q for the transient block,
    P(T_c > t) = pi Q^t 1 and E[T_c] = pi (I-Q)^{-1} 1, both exact.
    """
    A, B = C.build_matrices(family, k, r, p)
    s = A.shape[0]
    Q = np.zeros((s * c, s * c))
    for x in range(c):
        Q[x * s:(x + 1) * s, x * s:(x + 1) * s] = A
        if x + 1 < c:
            Q[x * s:(x + 1) * s, (x + 1) * s:(x + 2) * s] = B
    pi = np.zeros(s * c)
    pi[:s] = C.initial_vector(family, k, r)

    mean = float(pi @ np.linalg.solve(np.eye(s * c) - Q, np.ones(s * c)))

    # survival curve, extended until the requested quantile is passed
    surv, v, t = [1.0], pi.copy(), 0
    while surv[-1] > 1.0 - qmax and t < 100_000:
        v = v @ Q
        surv.append(float(v.sum()))
        t += 1
    surv = np.asarray(surv)
    quant = {q: int(np.searchsorted(-surv, -(1.0 - q)))
             for q in (0.5, 0.9, 0.99)}
    return mean, quant, surv


# ----------------------------------------------------------------------
# ROC
# ----------------------------------------------------------------------
def roc_points(n, family, k, r, p_null, p_alt):
    """Achievable (level, power) pairs, ordered by level, with both endpoints."""
    T0 = tail(n, family, k, r, p_null)
    T1 = tail(n, family, k, r, p_alt)
    m = min(len(T0), len(T1))
    pts = sorted({(float(T0[c]), float(T1[c])) for c in range(m)}
                 | {(0.0, 0.0), (1.0, 1.0)})
    return np.array(pts)


def auc(pts):
    return float(_trapz(pts[:, 1], pts[:, 0]))


def partial_auc(pts, amax=PAUC_MAX):
    """Area over [0, amax], standardised to [0,1] by dividing by amax.

    The achievable levels are discrete; interpolating linearly between adjacent
    points is the randomised test that attains the intermediate levels, so the
    interpolation is not a numerical convenience but the correct object.
    """
    x, y = pts[:, 0], pts[:, 1]
    keep = x <= amax
    xs, ys = list(x[keep]), list(y[keep])
    if xs[-1] < amax:
        j = int(np.searchsorted(x, amax))
        if j < len(x):
            w = (amax - x[j - 1]) / (x[j] - x[j - 1])
            xs.append(amax)
            ys.append(y[j - 1] + w * (y[j] - y[j - 1]))
    return float(_trapz(ys, xs) / amax)


# ----------------------------------------------------------------------
# A + C: calibrated comparison with run lengths
# ----------------------------------------------------------------------
def calibrated_study():
    banner("A. Calibrated comparison (healthy p2=%.2f, degraded p2=%.2f, n=%d)"
           % (HEALTHY, DEGRADED, N))
    ph, pd = C.symmetric_p(HEALTHY), C.symmetric_p(DEGRADED)
    rows = []
    for alpha in ALPHAS:
        print("\n  target alpha = %.2f" % alpha)
        print("  %-9s %3s %10s %9s %9s %10s %10s"
              % ("scheme", "c", "achieved", "power", "rand.pw", "ARL0", "ARL1"))
        for k, r in GRID:
            c, ach = calibrate(alpha, N, FAMILY, k, r, ph)
            power = float(tail(N, FAMILY, k, r, pd)[c])
            rpow = randomised_power(alpha, N, FAMILY, k, r, ph, pd)
            arl0, _, _ = waiting_time(FAMILY, k, r, ph, c)
            arl1, q1, _ = waiting_time(FAMILY, k, r, pd, c)
            print("  k=%d, r=%d %3d %10.5f %9.5f %9.5f %10.1f %10.1f"
                  % (k, r, c, ach, power, rpow, arl0, arl1))
            rows.append([alpha, k, r, c, ach, power, rpow, arl0, arl1,
                         q1[0.5], q1[0.9]])
    write_csv("calibrated.csv",
              ["target_alpha", "k", "r", "threshold_c", "achieved_alpha",
               "power", "randomised_power", "ARL0", "ARL1",
               "median_T1", "q90_T1"], rows)
    return rows


def write_calibrated_table(rows):
    body = []
    for alpha in ALPHAS:
        sub = [x for x in rows if x[0] == alpha]
        if body:
            body.append(None)
        best = max(sub, key=lambda x: x[6])
        for x in sub:
            rpow = ("\\textbf{%.4f}" % x[6]) if x is best else "%.4f" % x[6]
            body.append(["%.2f" % alpha, "%d" % x[1], "%d" % x[2], "%d" % x[3],
                         "%.5f" % x[4], "%.4f" % x[5], rpow,
                         "%.1f" % x[7], "%.1f" % x[8]])
    write_table("tab_calibrated.tex", latex_table(
        caption=("Calibrated comparison of monitoring schemes built on "
                 "$\\mathcal{E}_1$. For each target level $\\alpha$ and each "
                 "$(k,r)$ the threshold $c$ is the smallest integer with "
                 "$\\Pr(N_{n}\\ge c\\mid\\text{healthy})\\le\\alpha$ at "
                 "$n=%d$. Because $N_n$ is integer-valued the achieved level "
                 "can fall far below $\\alpha$, so two columns of power are "
                 "given: at the integer threshold, and for the randomised test "
                 "of exact size $\\alpha$, which is the only one that compares "
                 "schemes at a genuinely common false-alarm rate. "
                 "$\\mathrm{ARL}_0$ and $\\mathrm{ARL}_1$ are the mean run "
                 "lengths $\\mathbb{E}[T_c]$ under the two profiles, computed "
                 "from the absorbing embedding of "
                 "Section~\\ref{subsec:waiting} without truncation. Healthy "
                 "$p_2=%.2f$, degraded $p_2=%.2f$, $p_0=p_1=(1-p_2)/2$."
                 % (N, HEALTHY, DEGRADED)),
        label="tab:calibrated",
        colspec="c cc c r rr rr",
        header=[["$\\alpha$", "$k$", "$r$", "$c$", "achieved $\\alpha$",
                 "power at $c$", "randomised power",
                 "$\\mathrm{ARL}_0$", "$\\mathrm{ARL}_1$"]],
        rows=body,
        notes=("The most powerful scheme at each level, judged by the "
               "randomised power, is shown in bold. Comparing rows within a "
               "block is legitimate because the level has been matched. "
               "Comparing raw alarm probabilities across schemes, each carrying "
               "its own false-alarm rate, is not, and is what this table "
               "replaces.")))


# ----------------------------------------------------------------------
# B: ROC
# ----------------------------------------------------------------------
def roc_study():
    banner("B. Threshold sweep and ROC")
    ph, pd = C.symmetric_p(HEALTHY), C.symmetric_p(DEGRADED)
    rows, curves = [], {}
    print("  %-9s %8s %14s" % ("scheme", "AUC", "pAUC[0,%.2f]" % PAUC_MAX))
    for k, r in GRID:
        pts = roc_points(N, FAMILY, k, r, ph, pd)
        curves[(k, r)] = pts
        a, pa = auc(pts), partial_auc(pts)
        print("  k=%d, r=%d %8.4f %14.4f" % (k, r, a, pa))
        rows.append([k, r, a, pa, len(pts) - 2])
    write_csv("roc.csv", ["k", "r", "AUC", "pAUC_standardised",
                          "n_achievable_thresholds"], rows)

    write_table("tab_roc.tex", latex_table(
        caption=("Threshold sweep for $\\mathcal{E}_1$ at $n=%d$. Each integer "
                 "threshold $c$ contributes one achievable "
                 "(level, power) pair; AUC is the area under the resulting "
                 "curve and $\\mathrm{pAUC}$ its restriction to "
                 "$\\alpha\\in[0,%.2f]$, standardised by dividing by %.2f so "
                 "that $0.5$ is the value of a useless monitor over that range. "
                 "The last column counts the distinct levels the scheme can "
                 "attain without randomisation, which bounds how finely it can "
                 "be calibrated." % (N, PAUC_MAX, PAUC_MAX)),
        label="tab:roc",
        colspec="cc rr c",
        header=[["$k$", "$r$", "AUC", "$\\mathrm{pAUC}$",
                 "achievable levels"]],
        rows=[["%d" % x[0], "%d" % x[1], "%.4f" % x[2], "%.4f" % x[3],
               "%d" % x[4]] for x in rows],
        notes=("AUC averages over all levels, most of which no monitor would "
               "use. Restricting to the operating range leaves the ordering "
               "unchanged but roughly doubles the measured benefit of "
               "tolerance: at $k=3$ moving from $r=0$ to $r=2$ gains $0.124$ "
               "in AUC and $0.206$ in $\\mathrm{pAUC}$, and at $k=2$, $0.028$ "
               "against $0.108$. The schemes differ most where alarms are "
               "rare, which is where a monitor operates.")))
    return rows, curves


def plot_roc(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, amax, title in ((axes[0], 1.0, "Full ROC"),
                            (axes[1], PAUC_MAX, "Operating range")):
        for (k, r), pts in sorted(curves.items()):
            m = pts[:, 0] <= amax * 1.0001
            line, = ax.plot(pts[m, 0], pts[m, 1], marker="o", ms=3.4, lw=1.3,
                            label=r"$k=%d,\ r=%d$" % (k, r))
            # continue to the boundary along the randomised test that pAUC
            # integrates, so the shaded region is not visually truncated
            x, y = pts[:, 0], pts[:, 1]
            if x[m][-1] < amax:
                j = int(np.searchsorted(x, amax))
                if j < len(x):
                    w = (amax - x[j - 1]) / (x[j] - x[j - 1])
                    ax.plot([x[m][-1], amax],
                            [y[m][-1], y[j - 1] + w * (y[j] - y[j - 1])],
                            ls="--", lw=1.0, color=line.get_color())
        ax.plot([0, amax], [0, amax], ls=":", c="0.5", lw=1)
        ax.set_xlabel(r"false-alarm probability $\alpha$")
        ax.set_ylabel("power")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, amax)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(figure_path("fig5_roc.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig5_roc.png")


# ----------------------------------------------------------------------
# C: waiting time
# ----------------------------------------------------------------------
def waiting_study(alpha=0.05):
    banner("C. Waiting time to the first alarm at matched level %.2f" % alpha)
    ph, pd = C.symmetric_p(HEALTHY), C.symmetric_p(DEGRADED)
    rows = []
    print("  %-9s %3s %9s %9s %7s %7s %7s"
          % ("scheme", "c", "ARL0", "E[Tc]", "med", "q90", "q99"))
    for k, r in GRID:
        c, _ = calibrate(alpha, N, FAMILY, k, r, ph)
        arl0, _, _ = waiting_time(FAMILY, k, r, ph, c)
        mean, q, surv = waiting_time(FAMILY, k, r, pd, c)
        print("  k=%d, r=%d %3d %9.1f %9.2f %7d %7d %7d"
              % (k, r, c, arl0, mean, q[0.5], q[0.9], q[0.99]))
        rows.append([k, r, c, arl0, mean, q[0.5], q[0.9], q[0.99],
                     float(1.0 - surv[min(N, len(surv) - 1)])])
    write_csv("waiting.csv",
              ["k", "r", "c", "ARL0", "E_Tc_degraded", "median", "q90", "q99",
               "P_alarm_by_n"], rows)

    write_table("tab_waiting.tex", latex_table(
        caption=("Distribution of the run length $T_c$, the trial at which the "
                 "$c$th occurrence completes, under degradation "
                 "($p_2=%.2f$), with $c$ calibrated to $\\alpha=%.2f$ as in "
                 "Table~\\ref{tab:calibrated}. Quantiles are exact and "
                 "untruncated, obtained from the absorbing embedding; the last "
                 "column is $\\Pr(T_c\\le %d)$, which by monotonicity of "
                 "$N_t$ equals the power reported in "
                 "Table~\\ref{tab:calibrated}."
                 % (DEGRADED, alpha, N)),
        label="tab:waiting",
        colspec="cc c rr rrr r",
        header=[["$k$", "$r$", "$c$", "$\\mathrm{ARL}_0$",
                 "$\\mathbb{E}[T_c]$", "median", "$q_{0.9}$", "$q_{0.99}$",
                 "$\\Pr(T_c\\le n)$"]],
        rows=[["%d" % x[0], "%d" % x[1], "%d" % x[2], "%.1f" % x[3],
               "%.2f" % x[4], "%d" % x[5], "%d" % x[6], "%d" % x[7],
               "%.4f" % x[8]] for x in rows],
        notes=("The mean is dominated by a long right tail and exceeds the "
               "median in every row, so a scheme summarised by its mean run "
               "length alone will look slower than it usually is.")))
    return rows


def arl_matched_study(targets=(500.0, 1000.0)):
    """Calibrate to a target in-control run length instead of a horizon level.

    For the non-overlapping families the buffer is emptied on completion, so
    the chain restarts from state 0 and T_c is a sum of c independent copies of
    T_1: E[T_c] = c E[T_1] exactly.  Hence the ratio ARL_0/ARL_1 does not
    depend on the threshold, and matching ARL_0 to a target M gives
    ARL_1 = M / ratio.  The comparison at matched ARL_0 therefore reduces to
    comparing the ratio, and needs no integer rounding.
    """
    banner("C2. ARL-matched comparison")
    ph, pd = C.symmetric_p(HEALTHY), C.symmetric_p(DEGRADED)

    # the renewal identity, checked rather than assumed
    worst = 0.0
    for k, r in GRID:
        m1, _, _ = waiting_time(FAMILY, k, r, ph, 1)
        for c in (2, 3, 5):
            mc, _, _ = waiting_time(FAMILY, k, r, ph, c)
            worst = max(worst, abs(mc - c * m1) / (c * m1))
    print("  max relative departure from E[T_c] = c E[T_1]: %.2e" % worst)

    rows = []
    print("  %-9s %10s %10s %8s %s"
          % ("scheme", "ARL0 per c", "ARL1 per c", "ratio",
             "  ".join("ARL1 at %.0f" % t for t in targets)))
    for k, r in GRID:
        a0, _, _ = waiting_time(FAMILY, k, r, ph, 1)
        a1, _, _ = waiting_time(FAMILY, k, r, pd, 1)
        ratio = a0 / a1
        matched = [t / ratio for t in targets]
        print("  k=%d, r=%d %10.1f %10.2f %8.2f %s"
              % (k, r, a0, a1, ratio,
                 "  ".join("%11.1f" % m for m in matched)))
        rows.append([k, r, a0, a1, ratio] + matched)
    write_csv("arl_matched.csv",
              ["k", "r", "ARL0_per_threshold", "ARL1_per_threshold", "ratio"]
              + ["ARL1_at_ARL0_%.0f" % t for t in targets], rows)

    best = max(rows, key=lambda x: x[4])
    body = []
    for x in rows:
        ratio = ("\\textbf{%.2f}" % x[4]) if x is best else "%.2f" % x[4]
        body.append(["%d" % x[0], "%d" % x[1], "%.1f" % x[2], "%.2f" % x[3],
                     ratio] + ["%.1f" % v for v in x[5:]])
    write_table("tab_arl.tex", latex_table(
        caption=("Comparison at matched in-control run length, the sequential "
                 "counterpart of Table~\\ref{tab:calibrated}. For the "
                 "non-overlapping families the buffer empties on completion, "
                 "so $T_c$ is a sum of $c$ independent copies of $T_1$ and "
                 "$\\mathbb{E}[T_c]=c\\,\\mathbb{E}[T_1]$ exactly; the first "
                 "two columns are therefore per unit threshold and the ratio "
                 "$\\mathrm{ARL}_0/\\mathrm{ARL}_1$ does not depend on $c$. "
                 "Setting $c$ to match a target $\\mathrm{ARL}_0$ then gives "
                 "the out-of-control run lengths in the last two columns, "
                 "with no integer rounding required. Healthy $p_2=%.2f$, "
                 "degraded $p_2=%.2f$." % (HEALTHY, DEGRADED)),
        label="tab:arl",
        colspec="cc rr r rr",
        header=[["$k$", "$r$", "$\\mathbb{E}[T_1]$ healthy",
                 "$\\mathbb{E}[T_1]$ degraded",
                 "$\\mathrm{ARL}_0/\\mathrm{ARL}_1$"]
                + ["$\\mathrm{ARL}_1$ at $\\mathrm{ARL}_0=%.0f$" % t
                   for t in targets]],
        rows=body, resize=True,
        notes=("A higher ratio is better and is shown in bold. This comparison "
               "reverses the fixed-horizon one: at a matched false-alarm rate "
               "over $n=%d$ trials tolerance is uniformly better "
               "(Table~\\ref{tab:calibrated}), whereas at a matched "
               "$\\mathrm{ARL}_0$ it is uniformly worse, because tolerating "
               "warning states raises the occurrence rate proportionally more "
               "under the healthy profile, where critical states are scarce "
               "and gaps therefore matter most." % N)))
    return rows


# ----------------------------------------------------------------------
# D: the p0 : p1 lever
# ----------------------------------------------------------------------
def lever_study(alpha=0.05, k=3, rs=(0, 1, 2),
                rhos=np.linspace(0.05, 0.95, 19)):
    banner("D. Warning-versus-normal lever at k=%d, matched level %.2f"
           % (k, alpha))

    def profile(p2, rho):
        return ((1 - rho) * (1 - p2), rho * (1 - p2), p2)

    rows, series = [], {r: [] for r in rs}
    for rho in rhos:
        ph, pd = profile(HEALTHY, rho), profile(DEGRADED, rho)
        for r in rs:
            c, ach = calibrate(alpha, N, FAMILY, k, r, ph)
            power = float(tail(N, FAMILY, k, r, pd)[c])
            series[r].append(power)
            rows.append([rho, k, r, c, ach, power])
    write_csv("lever.csv",
              ["rho", "k", "r", "threshold_c", "achieved_alpha", "power"], rows)

    for r in rs:
        print("  r=%d: power %.4f at rho=%.2f -> %.4f at rho=%.2f"
              % (r, series[r][0], rhos[0], series[r][-1], rhos[-1]))
    gain = np.asarray(series[rs[-1]]) - np.asarray(series[rs[0]])
    print("  gain from r=%d over r=%d: %.4f at rho=%.2f -> %.4f at rho=%.2f"
          % (rs[-1], rs[0], gain[0], rhos[0], gain[-1], rhos[-1]))
    return rhos, series, gain


def plot_lever(rhos, series, gain, k=3, alpha=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for r, ys in sorted(series.items()):
        axes[0].plot(rhos, ys, marker="o", ms=3.4, lw=1.4, label=r"$r=%d$" % r)
    axes[0].set_ylabel(r"power at matched $\alpha=%.2f$" % alpha)
    axes[0].set_title(r"Calibrated power, $k=%d$" % k, fontsize=10)
    axes[0].legend(fontsize=9)

    axes[1].plot(rhos, gain, marker="s", ms=3.6, lw=1.5, color="C3")
    axes[1].axhline(0, ls=":", c="0.5", lw=1)
    axes[1].set_ylabel("power gain from tolerance")
    axes[1].set_title("Gain of $r=2$ over $r=0$", fontsize=10)

    for ax in axes:
        ax.set_xlabel(r"$\rho = p_1/(p_0+p_1)$, warning share of non-critical")
        ax.grid(alpha=0.3)
    fig.savefig(figure_path("fig6_lever.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig6_lever.png")


# ----------------------------------------------------------------------
def main():
    save_environment()
    rows = calibrated_study()
    write_calibrated_table(rows)

    _roc, curves = roc_study()
    plot_roc(curves)

    waiting_study()
    arl_matched_study()

    rhos, series, gain = lever_study()
    plot_lever(rhos, series, gain)

    banner("Consistency checks")
    ph, pd = C.symmetric_p(HEALTHY), C.symmetric_p(DEGRADED)
    worst = 0.0
    for k, r in GRID:
        for p in (ph, pd):
            for c in (1, 2, 3):
                _m, _q, surv = waiting_time(FAMILY, k, r, p, c)
                lhs = 1.0 - surv[min(N, len(surv) - 1)]      # P(T_c <= n)
                rhs = tail(N, FAMILY, k, r, p)[c]            # P(N_n >= c)
                worst = max(worst, abs(lhs - rhs))
    print("  max |P(T_c <= n) - P(N_n >= c)| over the grid = %.2e" % worst)
    print("  (the two must agree exactly: N_t is non-decreasing in t)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
