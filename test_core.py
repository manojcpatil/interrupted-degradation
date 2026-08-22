"""
test_core.py
============
Self-contained correctness tests for `mcem_core`.  Run before regenerating any
table:

    python test_core.py

Every check is an assertion; the script exits non-zero on the first failure.

  T1  the embedded chain reproduces Definition 1 on every path in {N,W,C}^n
  T2  A_t + B_t is row-stochastic and non-negative, and s matches the closed
      forms s1 and s2
  T3  the exact PMF equals brute-force enumeration of all 3^n weighted paths
  T4  the PMF sums to one and vanishes outside the stated support
  T5  the sparse and dense implementations agree
  T6  the general moment recursion agrees with the moments of the PMF
  T7  Corollaries 2 and 3 (k=2, r=1) agree with the matrix embedding
  T8  the auto-generated scalar recursion agrees with the matrix embedding
  T9  the vectorised simulator counts exactly as the scalar scanner
  T10 degenerate parameter values behave as documented
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

import mcem_core as C

GRID = [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 2)]
PASSED = []


def check(name, condition, detail=""):
    if not condition:
        print("FAIL  %s  %s" % (name, detail))
        sys.exit(1)
    PASSED.append(name)
    print("ok    %-52s %s" % (name, detail))


def t1_paths():
    for k, r in GRID:
        n = 9 if k <= 3 else 10
        for fam in C.FAMILIES:
            bad = sum(1 for q in itertools.product((0, 1, 2), repeat=n)
                      if C.count_occurrences(q, fam, k, r)
                      != C.count_by_definition(q, fam, k, r))
            check("T1 chain == Definition 1  %s k=%d r=%d" % (fam, k, r),
                  bad == 0, "%d paths" % 3 ** n)


def t2_structure():
    p = C.symmetric_p(0.37)
    for fam in C.FAMILIES:
        for k in (2, 3, 4, 5):
            for r in (0, 1, 2, 3):
                A, B = C.build_matrices(fam, k, r, p)
                defect = C.row_stochastic_defect(fam, k, r, p)
                ok = (defect < 1e-14 and A.min() >= 0 and B.min() >= 0
                      and A.shape[0] == C.state_dimension(fam, k, r))
                if not ok:
                    check("T2 %s k=%d r=%d" % (fam, k, r), False,
                          "defect=%.2e" % defect)
    check("T2 row-stochastic, non-negative, dimension formula", True,
          "4 families x k in 2..5 x r in 0..3")


def t3_bruteforce():
    for k, r in [(2, 1), (3, 2)]:
        for fam in C.FAMILIES:
            for p2 in (0.25, 0.55):
                p = C.symmetric_p(p2)
                a = C.pmf(10, fam, k, r, p)
                b = C.pmf_bruteforce(10, fam, k, r, p)
                m = max(len(a), len(b))
                x = np.zeros(m)
                x[:len(a)] = a
                y = np.zeros(m)
                y[:len(b)] = b
                d = float(np.max(np.abs(x - y)))
                check("T3 PMF == brute force  %s k=%d r=%d p2=%.2f"
                      % (fam, k, r, p2), d < 1e-12, "max diff %.2e" % d)


def t4_support():
    for fam in C.FAMILIES:
        for k, r in [(2, 1), (5, 2)]:
            for n in (10, 30):
                f = C.pmf(n, fam, k, r, C.symmetric_p(0.6))
                check("T4 support and total mass  %s k=%d r=%d n=%d"
                      % (fam, k, r, n),
                      abs(f.sum() - 1) < 1e-12
                      and len(f) - 1 <= C.max_count(fam, k, r, n),
                      "sum-1 = %.1e" % (f.sum() - 1))


def t5_sparse():
    worst = 0.0
    for fam in C.FAMILIES:
        for k, r in [(2, 1), (3, 2), (5, 2)]:
            a = C.pmf(60, fam, k, r, C.symmetric_p(0.45))
            b = C.pmf_sparse(60, fam, k, r, C.symmetric_p(0.45))
            worst = max(worst, float(np.max(np.abs(a - b))))
    check("T5 sparse == dense", worst < 1e-12, "max diff %.2e" % worst)


def t6_moments():
    worst = 0.0
    for fam in C.FAMILIES:
        for k, r in [(2, 1), (3, 2), (5, 2)]:
            for p2 in (0.2, 0.5, 0.8):
                for n in (15, 45):
                    p = C.symmetric_p(p2)
                    a = C.pmf_moments(C.pmf(n, fam, k, r, p))
                    b = C.moments(n, fam, k, r, p)
                    worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    check("T6 Proposition 2 == moments of the PMF", worst < 1e-9,
          "max diff %.2e" % worst)


def t7_corollaries():
    w1 = w2 = 0.0
    for p2 in (0.1, 0.3, 0.5, 0.7, 0.9):
        p = C.symmetric_p(p2)
        for n in (2, 5, 20, 80, 200):
            a = C.pmf(n, "E1", 2, 1, p)
            b = C.pmf_scalar_k2r1(n, p)
            w1 = max(w1, float(np.max(np.abs(a[:len(b)] - b))))
            m1 = C.pmf_moments(a)
            m2 = C.moments_scalar_k2r1(n, p)
            w2 = max(w2, abs(m1[0] - m2[0]), abs(m1[1] - m2[1]))
    check("T7 Corollary 2 (scalar PMF)", w1 < 1e-12, "max diff %.2e" % w1)
    check("T7 Corollary 3 (moments)", w2 < 1e-8, "max diff %.2e" % w2)


def t8_auto_recursion():
    worst = 0.0
    for fam, k, r in [("E1", 2, 1), ("E2", 2, 1), ("E3", 2, 1), ("E4", 2, 1),
                      ("E1", 3, 1), ("E2", 3, 1)]:
        for p2 in (0.3, 0.6):
            p = C.symmetric_p(p2)
            rec = C.scalar_recursion(fam, k, r, p)
            for n in (10, 40):
                a = C.pmf(n, fam, k, r, p)
                b = C.pmf_from_scalar_recursion(n, fam, k, r, p, rec)
                L = min(len(a), len(b))
                worst = max(worst, float(np.max(np.abs(a[:L] - b[:L]))))
    check("T8 auto-generated scalar recursion == embedding", worst < 1e-11,
          "max diff %.2e" % worst)


def t9_simulator():
    rng = np.random.default_rng(11)
    for k, r in [(2, 1), (3, 2), (5, 2)]:
        for p2 in (0.3, 0.7):
            p = C.symmetric_p(p2)
            Z = rng.choice(3, size=(2000, 30), p=p)
            for fam in C.FAMILIES:
                ref = np.array([C.count_by_definition(row.tolist(), fam, k, r)
                                for row in Z])
                got = _replay_vectorised(Z, fam, k, r)
                check("T9 vectorised == scalar scanner  %s k=%d r=%d p2=%.1f"
                      % (fam, k, r, p2), np.array_equal(got, ref),
                      "%d paths" % len(Z))


def _replay_vectorised(Z, fam, k, r):
    """Run the batched counting rule of `simulate_vectorised` on given paths."""
    b, n = Z.shape
    strict = C.is_strict(fam)
    consume = fam in ("E1", "E2")
    gaps = np.zeros((b, max(k - 1, 1)), dtype=np.int32)
    m = np.zeros(b, dtype=np.int32)
    d = np.zeros(b, dtype=np.int32)
    cnt = np.zeros(b, dtype=np.int32)
    for t in range(n):
        z = Z[:, t]
        isN, isW, isC = z == 0, z == 1, z == 2
        m[isN] = 0
        d[isN] = 0
        gaps[isN] = 0
        d[isW & (m > 0)] += 1
        if not isC.any():
            continue
        c = np.nonzero(isC)[0]
        if strict:
            kill = c[(m[c] > 0) & (d[c] > r)]
            m[kill] = 0
            gaps[kill] = 0
        push = c[m[c] > 0]
        if push.size and k > 1:
            gaps[push, :-1] = gaps[push, 1:]
            gaps[push, -1] = d[c[m[c] > 0]]
        m[c] += 1
        d[c] = 0
        full = c[m[c] == k]
        if full.size:
            mx = gaps[full].max(axis=1)
            hit = (mx <= r) if strict else (mx >= r)
            won = full[hit]
            cnt[won] += 1
            if consume:
                m[won] = 0
                gaps[won] = 0
                m[full[~hit]] = k - 1
            else:
                m[full] = k - 1
    return cnt


def t10_degenerate():
    p = C.symmetric_p(0.4)
    a = C.pmf(40, "E1", 3, 0, p)
    b = C.pmf(40, "E1", 3, 0, (1 - 0.4, 0.0, 0.4))
    check("T10 r=0 equals binary-collapsed runs of k",
          float(np.max(np.abs(a - b))) < 1e-13, "identical distributions")
    seq = C.read_path("CCCCCC")
    check("T10 E2 with r=0 counts every block of k criticals",
          C.count_occurrences(seq, "E2", 3, 0) == 2, "CCCCCC -> 2")
    check("T10 E3 caps the gap after a completion",
          C.count_occurrences(C.read_path("CCCWWWC"), "E3", 3, 2) == 1,
          "CCCWWWC -> 1")
    check("T10 E2 slides the window on a failed attempt",
          C.count_occurrences(C.read_path("CCCWWC"), "E2", 3, 2) == 1,
          "CCCWWC -> 1")


def t11_binary_reduction():
    """At p0 = 0 the strict families must be the binary interrupted-run count,
    and at p0 > 0 they must not be: the reset is what makes the third letter
    more than a relabelling."""
    from code_11_reduction import binary_interrupted_runs

    for k, r in ((2, 1), (3, 2), (4, 2)):
        for family, overlapping in (("E1", False), ("E3", True)):
            bad = 0
            for bits in itertools.product((0, 1), repeat=11):
                seq = [C.CRITICAL if b else C.WARNING for b in bits]
                if (C.count_occurrences(seq, family, k, r)
                        != binary_interrupted_runs(bits, k, r, overlapping)):
                    bad += 1
            check("T11 p0=0 reduces to the binary rule %s k=%d r=%d"
                  % (family, k, r), bad == 0, "%d paths agree" % 2 ** 11)

    # the reduction is a boundary case, not an identity: with p0 > 0 the
    # normal state resets history and the two counts must part company
    seq = C.read_path("CNC")
    check("T11 the reduction is strict: N resets, W does not",
          C.count_occurrences(seq, "E1", 2, 1) == 0
          and C.count_occurrences(C.read_path("CWC"), "E1", 2, 1) == 1,
          "CNC -> 0 but CWC -> 1")


def main():
    for fn in (t1_paths, t2_structure, t3_bruteforce, t4_support, t5_sparse,
               t6_moments, t7_corollaries, t8_auto_recursion, t9_simulator,
               t10_degenerate, t11_binary_reduction):
        fn()
    print("\n%d checks passed." % len(PASSED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
