"""
mcem_core.py
============
Core library for

    "Exact Distributions of Interrupted Degradation Patterns in Multi-State
     Systems: A Markov Chain Embedding Approach with Scalar Recursive
     Computation"  (revision of 20 August 2026)

Everything the revised manuscript reports is computed from this one module.

Degradation alphabet
--------------------
    0 = Normal   (N)  hard reset
    1 = Warning  (W)  interruption / gap
    2 = Critical (C)  counted event

Probabilities are always a triple p = (p0, p1, p2) summing to one.

Pattern families (Definition 1 and Algorithm 1 of the revised manuscript)
------------------------------------------------------------------------
An *occurrence* is a window of k critical states lying inside a block that
contains no Normal state, whose k-1 inter-critical warning gaps satisfy

    E1, E3 :  every gap <= r         ("strict")
    E2, E4 :  at least one gap >= r  ("gap-threshold")

Counting is by earliest completion, scanning left to right:

    E1, E2 :  non-overlapping - the window is consumed on completion
    E3, E4 :  overlapping     - the window slides forward by one critical state

Embedded state spaces
---------------------
    E1, E3 :  Omega_strict = {0} u {(i,d) : 1<=i<=k-1, 0<=d<=r}
              s1 = 1 + (k-1)(r+1)
    E2, E4 :  Omega_thr    = {0} u {(i,d,a) : 1<=i<=k-1, 0<=d<=r, 0<=a<=i-1}
              s2 = 1 + (r+1) k (k-1) / 2

where i counts buffered critical states, d is the current warning gap capped
at r, and a is the age (measured back from the newest gap) of the most recent
gap that reached the threshold, with a = 0 meaning "no such gap in the window".

Completion targets:  E1, E2 -> state 0 ;  E3 -> (k-1, 0) ;  E4 -> (k-1, 0, a').
No terminal state is required by any family; the overlapping convention is
carried by the target of the count-incrementing transition.

`step` is the single source of truth: both the transition matrices and the
path counter are built from it, so they cannot drift apart.  It is checked
against the independent scanner `count_by_definition` in `code_09_audit.py`
and `test_core.py`.

License: MIT.
"""

from __future__ import annotations

import itertools

import numpy as np

FAMILIES = ("E1", "E2", "E3", "E4")
NORMAL, WARNING, CRITICAL = 0, 1, 2
STRICT = ("E1", "E3")
OVERLAPPING = ("E3", "E4")


# ======================================================================
# 1. State space
# ======================================================================
def is_strict(family):
    return family in STRICT


def is_overlapping(family):
    return family in OVERLAPPING


def _check(family, k, r):
    if family not in FAMILIES:
        raise ValueError("family must be one of %s, got %r" % (FAMILIES, family))
    if k < 2:
        raise ValueError("k must be at least 2")
    if r < 0:
        raise ValueError("r must be non-negative")


def state_space(family, k, r):
    """Ordered embedded states of one counting block, plus a label -> index map."""
    _check(family, k, r)
    states = [0]
    if is_strict(family):
        states += [(i, d) for i in range(1, k) for d in range(r + 1)]
    else:
        states += [(i, d, a)
                   for i in range(1, k) for d in range(r + 1) for a in range(i)]
    return states, {s: j for j, s in enumerate(states)}


def state_dimension(family, k, r):
    """s = |Omega|.  Equals the closed forms s1 and s2 documented above."""
    if is_strict(family):
        return 1 + (k - 1) * (r + 1)
    return 1 + (r + 1) * k * (k - 1) // 2


def state_label(st):
    """Human-readable label used in the printed matrices."""
    if st == 0:
        return "0"
    if len(st) == 2:
        return "(%d,%d)" % st
    return "(%d,%d,%d)" % st


# ======================================================================
# 2. Transition function -- the single source of truth
# ======================================================================
def step(state, z, family, k, r):
    """One trial.  Returns (next_state, number_of_completions in {0,1}).

    This function *is* Algorithm 1 of the manuscript.  Both the transition
    matrices and the path counter are derived from it, so they cannot drift
    apart.
    """
    strict = is_strict(family)
    overlap = is_overlapping(family)

    if z == NORMAL:
        return 0, 0

    if state == 0:
        if z == CRITICAL:
            return ((1, 0) if strict else (1, 0, 0)), 0
        return 0, 0

    i, d = state[0], state[1]
    a = None if strict else state[2]

    if z == WARNING:
        if d < r:
            return ((i, d + 1) if strict else (i, d + 1, a)), 0
        # d == r
        if strict:
            return 0, 0                       # gap limit exceeded: hard reset
        return (i, r, a), 0                   # threshold saturation loop

    # ---- z == CRITICAL -------------------------------------------------
    if strict:
        if i <= k - 2:
            return (i + 1, 0), 0
        return ((k - 1, 0) if overlap else 0), 1

    # threshold families: update the age of the newest saturated gap
    a_new = 1 if d >= r else (a + 1 if a > 0 else 0)
    if i <= k - 2:
        return (i + 1, 0, a_new), 0
    if a_new == 0:
        return (k - 1, 0, 0), 0               # window slides, still no count
    if overlap:
        return (k - 1, 0, a_new if a_new <= k - 2 else 0), 1
    return 0, 1


def count_occurrences(sequence, family, k, r):
    """Number of occurrences of the pattern on a realised path."""
    _check(family, k, r)
    st, total = 0, 0
    for z in sequence:
        st, c = step(st, z, family, k, r)
        total += c
    return total


# ======================================================================
# 3. Transition matrices
# ======================================================================
def build_matrices(family, k, r, p, dtype=float):
    """(A, B): within-block and count-incrementing s x s matrices.

    `p` may be floats or sympy symbols (pass dtype=object for the latter).
    """
    states, idx = state_space(family, k, r)
    s = len(states)
    A = np.zeros((s, s), dtype=dtype)
    B = np.zeros((s, s), dtype=dtype)
    for st in states:
        for z in (NORMAL, WARNING, CRITICAL):
            nxt, c = step(st, z, family, k, r)
            target = B if c else A
            target[idx[st], idx[nxt]] += p[z]
    return A, B


def initial_vector(family, k, r, dtype=float):
    v = np.zeros(state_dimension(family, k, r), dtype=dtype)
    v[0] = 1
    return v


def row_stochastic_defect(family, k, r, p):
    A, B = build_matrices(family, k, r, p)
    return float(np.max(np.abs((A + B).sum(axis=1) - 1.0)))


def sparsity(family, k, r, p=(1 / 3, 1 / 3, 1 / 3)):
    A, B = build_matrices(family, k, r, p)
    return int((A != 0).sum() + (B != 0).sum())


# ======================================================================
# 4. Exact distribution
# ======================================================================
def max_count(family, k, r, n):
    """Sharp upper bound on the support of N_{n,k,r}."""
    if n < k:
        return 0
    if family in ("E1", "E2"):
        return n // k
    return n - k + 1


def pmf(n, family, k, r, p):
    """Exact PMF via f_t(x) = f_{t-1}(x) A + f_{t-1}(x-1) B."""
    A, B = build_matrices(family, k, r, p)
    xmax = max_count(family, k, r, n)
    f = np.zeros((xmax + 1, A.shape[0]))
    f[0] = initial_vector(family, k, r)
    for _t in range(n):
        nxt = f @ A
        nxt[1:] += f[:-1] @ B
        f = nxt
    return f.sum(axis=1)


def pmf_sparse(n, family, k, r, p):
    """Same recursion using scipy.sparse; O(s) rather than O(s^2) per trial."""
    import scipy.sparse as sps
    A, B = build_matrices(family, k, r, p)
    At, Bt = sps.csr_matrix(A.T), sps.csr_matrix(B.T)
    xmax = max_count(family, k, r, n)
    f = np.zeros((xmax + 1, A.shape[0]))
    f[0] = initial_vector(family, k, r)
    for _t in range(n):
        nxt = (At @ f.T).T
        nxt[1:] += (Bt @ f[:-1].T).T
        f = nxt
    return f.sum(axis=1)


def pmf_moments(f):
    x = np.arange(len(f))
    m = float(np.dot(x, f))
    return m, float(np.dot(x * x, f)) - m * m


# ======================================================================
# 5. Exact moments without the PMF  (Proposition 2 of the revision)
# ======================================================================
def moments(n, family, k, r, p):
    """Exact (mean, variance) in O(n s^2), for ANY (k, r, family).

        S_t = S_{t-1}(A+B)
        M_t = M_{t-1}(A+B) + S_{t-1} B
        V_t = V_{t-1}(A+B) + 2 M_{t-1} B
        mu_n = sum(M_n),  nu_n = sum(V_n),  Var = nu_n + mu_n - mu_n^2
    """
    A, B = build_matrices(family, k, r, p)
    C = A + B
    S = initial_vector(family, k, r)
    M = np.zeros_like(S)
    V = np.zeros_like(S)
    for _t in range(n):
        V = V @ C + 2.0 * (M @ B)
        M = M @ C + S @ B
        S = S @ C
    mu = float(M.sum())
    return mu, float(V.sum()) + mu - mu * mu


def moment_trajectory(n, family, k, r, p):
    """mu[0..n] and var[0..n] from a single pass of the same recursion."""
    A, B = build_matrices(family, k, r, p)
    C = A + B
    S = initial_vector(family, k, r)
    M = np.zeros_like(S)
    V = np.zeros_like(S)
    mu = np.zeros(n + 1)
    var = np.zeros(n + 1)
    for t in range(1, n + 1):
        V = V @ C + 2.0 * (M @ B)
        M = M @ C + S @ B
        S = S @ C
        mu[t] = M.sum()
        var[t] = V.sum() + mu[t] - mu[t] ** 2
    return mu, var


# ======================================================================
# 6. Scalar recursions
# ======================================================================
def pmf_scalar_k2r1(n, p):
    """Corollary 2 (E1, k=2, r=1), pure scalar arithmetic, three-term memory.

        f_n(x) = (p0+p1) f_{n-1}(x) + p0 p2 f_{n-2}(x)
                 + p1 p2 (p0+p1) f_{n-3}(x)
                 + p2^2 f_{n-2}(x-1) + p1 p2^2 f_{n-3}(x-1)
    """
    p0, p1, p2 = p
    xmax = n // 2
    f0 = np.zeros(xmax + 1)
    f0[0] = 1.0
    f1 = f0.copy()
    f2 = np.zeros(xmax + 1)
    f2[0] = 1.0 - p2 * p2
    if xmax >= 1:
        f2[1] = p2 * p2
    if n == 0:
        return f0
    if n == 1:
        return f1
    if n == 2:
        return f2
    c1, c2, c3 = p0 + p1, p0 * p2, p1 * p2 * (p0 + p1)
    b2, b3 = p2 * p2, p1 * p2 * p2
    for _t in range(3, n + 1):
        row = c1 * f2 + c2 * f1 + c3 * f0
        row[1:] += b2 * f1[:-1] + b3 * f0[:-1]
        f0, f1, f2 = f1, f2, row
    return f2


def moments_scalar_k2r1(n, p):
    """Corollary 3 (E1, k=2, r=1): recursive mean and second factorial moment."""
    p0, p1, p2 = p
    mu = [0.0, 0.0, p2 * p2]
    nu = [0.0, 0.0, 0.0]
    if n <= 2:
        m, v = mu[n], nu[n]
        return m, v + m - m * m
    for t in range(3, n + 1):
        mu.append((p0 + p1) * mu[t - 1] + p2 * (p0 + p2) * mu[t - 2]
                  + p1 * p2 * mu[t - 3] + p2 * p2 * (1.0 + p1))
        nu.append((p0 + p1) * nu[t - 1] + p2 * (p0 + p2) * nu[t - 2]
                  + p1 * p2 * nu[t - 3]
                  + 2.0 * p2 * p2 * mu[t - 2] + 2.0 * p1 * p2 * p2 * mu[t - 3])
    m, v = mu[n], nu[n]
    return m, v + m - m * m


def scalar_recursion(family, k, r, p=None, symbolic=False):
    """Proposition 3: the linear recurrence obeyed by phi_n(z), any (k, r).

    Phi(z,w) = pi_0 [I - w(A+zB)]^(-1) 1' is rational in w with denominator
    det[I - w(A+zB)] = 1 - sum_j c_j(z) w^j of degree m <= s, hence

        phi_n(z) = sum_{j=1..m} c_j(z) phi_{n-j}(z).

    Returns a dict with `order` (m), `s`, and `den`:
      symbolic=True  -> den is a list of sympy expressions in p0, p1, p2, z
      symbolic=False -> p must be given; den[j-1] is a list of
                        (power_of_z, numeric coefficient) pairs
    """
    import sympy as sp

    z, w = sp.symbols("z w")
    if symbolic:
        pv = sp.symbols("p0 p1 p2", nonnegative=True)
    else:
        if p is None:
            raise ValueError("numeric mode needs p")
        pv = [sp.Rational(str(v)).limit_denominator(10 ** 9) for v in p]
    A, B = build_matrices(family, k, r, pv, dtype=object)
    s = A.shape[0]
    Am = sp.Matrix(s, s, lambda i, j: A[i, j])
    Bm = sp.Matrix(s, s, lambda i, j: B[i, j])
    det = sp.Poly(sp.expand((sp.eye(s) - w * (Am + z * Bm)).det(method="berkowitz")), w)
    coeffs = det.all_coeffs()[::-1]
    den = []
    for j in range(1, len(coeffs)):
        e = sp.expand(-coeffs[j])
        if symbolic:
            den.append(sp.factor(e))
        elif e == 0:
            den.append([])
        else:
            pz = sp.Poly(e, z)
            den.append([(int(m[0]), float(c))
                        for m, c in zip(pz.monoms(), pz.coeffs())])
    while den and (den[-1] == [] or den[-1] == 0):
        den.pop()
    out = {"den": den, "order": len(den), "s": s}
    if symbolic:
        out["symbols"] = tuple(pv) + (z,)
    return out


def pmf_from_scalar_recursion(n, family, k, r, p, rec=None):
    """Evaluate the PMF through the scalar recursion of `scalar_recursion`."""
    if rec is None:
        rec = scalar_recursion(family, k, r, p)
    den, m = rec["den"], rec["order"]
    xmax = max_count(family, k, r, n)

    A, B = build_matrices(family, k, r, p)
    f = np.zeros((xmax + 1, A.shape[0]))
    f[0] = initial_vector(family, k, r)
    hist = [f.sum(axis=1).copy()]
    for _t in range(1, min(m, n + 1)):
        nxt = f @ A
        nxt[1:] += f[:-1] @ B
        f = nxt
        hist.append(f.sum(axis=1).copy())
    if n < m:
        return hist[n]
    for _t in range(m, n + 1):
        acc = np.zeros(xmax + 1)
        for j in range(1, m + 1):
            prev = hist[-j]
            for power, c in den[j - 1]:
                if power == 0:
                    acc += c * prev
                else:
                    acc[power:] += c * prev[:-power]
        hist.append(acc)
    return hist[n]


# ======================================================================
# 7. Independent verification tools
# ======================================================================
def count_by_definition(sequence, family, k, r):
    """Count occurrences straight from Definition 1, with NO reference to the
    embedded chain: buffer the positions of the critical states, look at the
    window formed by the last k of them, and test its gaps.

    This is the routine used by the Monte Carlo study and the audit, so the
    validation is not circular.
    """
    _check(family, k, r)
    strict = is_strict(family)
    consume = family in ("E1", "E2")
    pos, total = [], 0
    for t, z in enumerate(sequence):
        if z == NORMAL:
            pos = []
        elif z == CRITICAL:
            if strict and pos and (t - pos[-1] - 1) > r:
                pos = []                       # an oversized gap kills every window
            pos.append(t)
            if len(pos) >= k:
                win = pos[-k:]
                gaps = [win[j + 1] - win[j] - 1 for j in range(k - 1)]
                hit = (max(gaps) <= r) if strict else (max(gaps) >= r)
                if hit:
                    total += 1
                    if consume:
                        pos = []
    return total


def pmf_bruteforce(n, family, k, r, p, counter=count_by_definition):
    """Exact PMF by enumerating all 3**n paths.  Ground truth for n <= ~12."""
    xmax = max_count(family, k, r, n)
    acc = np.zeros(xmax + 2)
    for seq in itertools.product((0, 1, 2), repeat=n):
        pr = 1.0
        for z in seq:
            pr *= p[z]
        acc[counter(seq, family, k, r)] += pr
    return acc


def simulate(n, family, k, r, p, trials, seed=0, counter=count_by_definition):
    """Monte Carlo counts using the independent definition-based counter."""
    rng = np.random.default_rng(seed)
    paths = rng.choice(3, size=(trials, n), p=p)
    return np.fromiter((counter(row.tolist(), family, k, r) for row in paths),
                       dtype=np.int32, count=trials)


def simulate_vectorised(n, family, k, r, p, trials, seed=0, batch=200_000):
    """Same counting rule as `count_by_definition`, evaluated across trials at
    once so that 10**6 replications are affordable.

    The per-trial memory is the buffer of gap lengths between the critical
    states seen so far, exactly as in the scalar scanner; the embedded chain is
    not used.  `test_core.py` checks the two counters agree path by path.
    """
    _check(family, k, r)
    strict = is_strict(family)
    consume = family in ("E1", "E2")
    rng = np.random.default_rng(seed)
    out = np.empty(trials, dtype=np.int32)

    done = 0
    while done < trials:
        b = min(batch, trials - done)
        Z = rng.choice(3, size=(b, n), p=p)
        gaps = np.zeros((b, max(k - 1, 1)), dtype=np.int32)   # oldest .. newest
        m = np.zeros(b, dtype=np.int32)                       # buffered criticals
        d = np.zeros(b, dtype=np.int32)                       # current gap
        cnt = np.zeros(b, dtype=np.int32)

        for t in range(n):
            z = Z[:, t]
            isN, isW, isC = z == NORMAL, z == WARNING, z == CRITICAL

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
                if k == 1:
                    hit = np.ones(full.size, dtype=bool)
                else:
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
        out[done:done + b] = cnt
        done += b
    return out


# ======================================================================
# 8. Helpers
# ======================================================================
def symmetric_p(p2):
    """p0 = p1 = (1-p2)/2, the parameterisation used throughout the paper."""
    return ((1.0 - p2) / 2.0, (1.0 - p2) / 2.0, float(p2))


def show_path(seq):
    return "".join("NWC"[z] for z in seq)


def read_path(text):
    """'NCWCCC' -> [0,2,1,2,2,2]"""
    return [{"N": 0, "W": 1, "C": 2}[ch] for ch in text.upper() if ch in "NWC"]


def prob_fmt(x, digits=6, star_below=1e-6):
    return "*" if x < star_below else ("%." + str(digits) + "f") % x
