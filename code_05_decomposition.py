"""
code_12_decomposition.py  ->  Table 9
=====================================
The i.i.d. block decomposition as an independent computational channel.

Lemma 1 says that when the trials are i.i.d. the normal states cut the
trajectory into maximal normal-free blocks; conditionally on the block lengths
the symbols inside a block are i.i.d. on {W,C} with P(C) = p2/(p1+p2), and the
count is the sum of the within-block counts.  Corollary 1 turns that into a
renewal equation for the generating function.

This script computes the mass function twice:

  (i)  by the embedding of Section 3 (embedding.pmf);
  (ii) by the decomposition -- within-block laws obtained by brute-force
       enumeration of every string in {W,C}^L, counted by a rule written
       directly from Definition 1, then compounded over the block structure by
       the renewal recursion.

Channel (ii) never constructs A or B and never calls the manuscript's scanner,
so agreement is evidence about the embedding rather than about one
implementation of it.  For E1 and E3 the within-block rule is the binary
interrupted-run statistic of Dafnis et al. (2021); for E2 and E4 it is the
gap-threshold rule of Definition 1, which has no published binary antecedent.

Brute force over the block interiors caps the horizon: the enumeration is
2^L strings for every block length L <= n, so n is kept at 16 or below.

Writes Output/decomposition.csv and Tables/tab_decomposition.tex.

License: MIT.
"""

from __future__ import annotations

import itertools

import numpy as np

import embedding as C
from common import banner, latex_table, sci_latex, write_csv, write_table

FAMILIES = ("E1", "E2", "E3", "E4")

# (k, r, n) settings; n <= NMAX because the within-block laws are brute forced.
SETTINGS = ((2, 1, 12), (3, 1, 14), (3, 2, 16), (4, 2, 16))

# (p0, p1, p2) settings.  p0 > 0 throughout: at p0 = 0 there is a single block
# and the decomposition says nothing that Table 8 does not already say.
PROFILES = ((0.50, 0.30, 0.20), (0.20, 0.30, 0.50), (0.34, 0.33, 0.33))

NMAX = max(n for _, _, n in SETTINGS)


def block_count(bits, family, k, r):
    """Occurrences of `family` inside one normal-free block.

    `bits` is the block interior: 1 = critical, 0 = warning.  There is no
    normal state and therefore no reset.  Transcribed from Definition 1 and the
    scan of Algorithm 1; it shares no code with the embedding.
    """
    strict = family in ("E1", "E3")
    overlapping = family in ("E3", "E4")
    count = 0
    buf = []                                   # positions of held critical states
    for pos, b in enumerate(bits):
        if b != 1:
            continue
        if strict and buf and (pos - buf[-1] - 1) > r:
            buf = []                           # gap too long: history dies
        buf.append(pos)
        if len(buf) >= k:
            window = buf[-k:]
            gaps = [window[j + 1] - window[j] - 1 for j in range(k - 1)]
            qualifies = max(gaps) <= r if strict else max(gaps) >= r
            if qualifies:
                count += 1
                if not overlapping:
                    buf = []                   # consume the window
    return count


def block_histograms(k, r, nmax=NMAX):
    """H[family][L] = integer array over (successes j, count x) of string tallies.

    The count of a binary string does not depend on q2, so the enumeration is
    done once and reweighted for every profile.  Entry [j, x] is the number of
    strings in {W,C}^L with j critical states and count x.
    """
    H = {f: [] for f in FAMILIES}
    for L in range(nmax + 1):
        tallies = {f: {} for f in FAMILIES}
        for bits in itertools.product((0, 1), repeat=L):
            j = sum(bits)
            for f in FAMILIES:
                key = (j, block_count(bits, f, k, r))
                tallies[f][key] = tallies[f].get(key, 0) + 1
        for f in FAMILIES:
            xmax = max((x for _, x in tallies[f]), default=0)
            arr = np.zeros((L + 1, xmax + 1))
            for (j, x), m in tallies[f].items():
                arr[j, x] = m
            H[f].append(arr)
    return H


def block_pmf(hist_L, q2):
    """Within-block mass function psi_L from the tally array at success prob q2."""
    L = hist_L.shape[0] - 1
    j = np.arange(L + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(hist_L.any(axis=1),
                     q2 ** j * (1.0 - q2) ** (L - j), 0.0)
    return hist_L.T @ w


def decomposition_pmf(n, hists, p):
    """Mass function of N_n by the renewal recursion of Corollary 1.

    f_0 = delta_0 and, for n >= 1,
        f_n = p0 f_{n-1}
              + sum_{L=1}^{n-1} (1-p0)^L p0 (psi_L * f_{n-L-1})
              + (1-p0)^n psi_n,
    conditioning on the length L of the initial normal-free block.
    """
    p0, p1, p2 = p
    q2 = p2 / (p1 + p2)
    psi = [block_pmf(h, q2) for h in hists[:n + 1]]

    f = [np.array([1.0])]                      # f_0
    for m in range(1, n + 1):
        acc = p0 * f[m - 1]
        for L in range(1, m):
            term = np.convolve(psi[L], f[m - L - 1]) * ((1 - p0) ** L * p0)
            acc = _add(acc, term)
        acc = _add(acc, psi[m] * (1 - p0) ** m)
        f.append(acc)
    return f[n]


def _add(a, b):
    """Sum two mass vectors of possibly different length."""
    if len(a) < len(b):
        a, b = b, a
    out = a.copy()
    out[:len(b)] += b
    return out


def compare(settings=SETTINGS, profiles=PROFILES):
    rows = []
    for k, r, n in settings:
        banner("k=%d r=%d n=%d: enumerating block interiors up to length %d"
               % (k, r, n, n))
        H = block_histograms(k, r, nmax=n)
        for p in profiles:
            for family in FAMILIES:
                f_embed = C.pmf(n, family, k, r, p)
                f_decomp = decomposition_pmf(n, H[family], p)
                size = max(len(f_embed), len(f_decomp))
                a = np.zeros(size); a[:len(f_embed)] = f_embed
                b = np.zeros(size); b[:len(f_decomp)] = f_decomp
                err = float(np.max(np.abs(a - b)))
                tv = float(0.5 * np.sum(np.abs(a - b)))
                rows.append((family, k, r, n, p[0], p[1], p[2], err, tv))
                print("  %s k=%d r=%d n=%2d p=(%.2f,%.2f,%.2f): "
                      "max |diff| = %.2e, total variation = %.2e"
                      % (family, k, r, n, p[0], p[1], p[2], err, tv))
    return rows


def write_summary_table(rows):
    """One row per (family, k, r, n): worst case over the probability profiles."""
    by_key = {}
    for family, k, r, n, p0, p1, p2, err, tv in rows:
        key = (family, k, r, n)
        acc = by_key.setdefault(key, {"err": 0.0, "tv": 0.0})
        acc["err"] = max(acc["err"], err)
        acc["tv"] = max(acc["tv"], tv)

    out = []
    for (family, k, r, n), acc in sorted(by_key.items(),
                                         key=lambda kv: (kv[0][1], kv[0][2],
                                                         kv[0][0])):
        out.append(["$\\mathcal{E}_%s$" % family[1], k, r, n,
                    sci_latex(acc["err"]), sci_latex(acc["tv"])])

    write_table("tab_decomposition.tex", latex_table(
        caption=("Embedding against the i.i.d.\\ block decomposition. The mass "
                 "function of $N^{(i)}_{n,k,r}$ is computed twice: once by the "
                 "embedded chain of Section~\\ref{sec:embedding}, and once by "
                 "Lemma~\\ref{lem:decomposition} and the renewal recursion "
                 "\\eqref{eq:renewal}, whose within-block laws come from "
                 "brute-force enumeration of every string in "
                 "$\\{\\W,\\Cc\\}^{L}$ counted by a rule transcribed from "
                 "Definition~\\ref{def:patterns}. The second channel constructs "
                 "no transition matrices. Each row reports the worst case over "
                 "the three profiles $(p_0,p_1,p_2)\\in\\{(0.50,0.30,0.20),"
                 "(0.20,0.30,0.50),(0.34,0.33,0.33)\\}$."),
        label="tab:decomposition",
        colspec="l ccc cc",
        header=["Family", "$k$", "$r$", "$n$",
                "Max PMF difference", "Total variation"],
        rows=out))
    return out


def main():
    rows = compare()
    write_csv("decomposition.csv",
              ("family", "k", "r", "n", "p0", "p1", "p2",
               "max_abs_pmf_diff", "total_variation"), rows)
    write_summary_table(rows)

    worst = max(row[7] for row in rows)
    banner("Summary")
    print("  %d comparisons; largest absolute PMF difference %.2e"
          % (len(rows), worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
