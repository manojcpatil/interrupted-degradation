"""
code_13_minimality.py  ->  Table 10
===================================
An independent check on Proposition 1: are the embedded dimensions minimal?

The embedding of Section 3 is a deterministic machine that reads the trajectory
one symbol at a time and emits, at each step, the number of occurrences
completed by that symbol (0 or 1).  That is a Mealy machine over the alphabet
{N,W,C} with output in {0,1}, and Algorithm 1 is its specification.  Any other
machine computing the same counting function step by step must have at least as
many states as its minimal form, so Moore's partition-refinement algorithm
answers the question directly.

For each (family, k, r) the script

  1. computes the states reachable from the empty buffer,
  2. minimises the Mealy machine by partition refinement,
  3. compares the result with the closed forms s1 = s2 = 1 + (k-1)(r+1) and
     s4 = 1 + (k-1) + r k (k-1)/2 of Theorems 1 and 2.

Proposition 1 proves that the two agree for every k and r; this script confirms
it numerically on a grid, and would expose an error in the construction or in
the proof's bookkeeping.

Nothing here uses the transition matrices; it uses embedding.step, which is
Algorithm 1 itself.

Writes Output/minimality.csv and Tables/tab_minimality.tex.

License: MIT.
"""

from __future__ import annotations

import embedding as C
from common import banner, latex_table, write_csv, write_table

FAMILIES = ("E1", "E2", "E3", "E4")
ALPHABET = (C.NORMAL, C.WARNING, C.CRITICAL)

GRID = tuple((k, r) for k in (2, 3, 4, 5) for r in (0, 1, 2, 3))


def reachable(family, k, r):
    """States reachable from the empty buffer, in discovery order."""
    seen = {0}
    order = [0]
    stack = [0]
    while stack:
        s = stack.pop()
        for z in ALPHABET:
            t, _ = C.step(s, z, family, k, r)
            if t not in seen:
                seen.add(t)
                order.append(t)
                stack.append(t)
    return order


def minimise(family, k, r):
    """Moore partition refinement for the Mealy machine (step, output).

    Returns (n_reachable, n_minimal).
    """
    states = reachable(family, k, r)
    trans = {s: {z: C.step(s, z, family, k, r) for z in ALPHABET} for s in states}

    # Initial partition: states agreeing on the output of every letter.
    cls = {}
    for s in states:
        key = tuple(trans[s][z][1] for z in ALPHABET)
        cls[s] = key
    cls = _renumber(cls, states)

    while True:
        refined = {}
        for s in states:
            refined[s] = (cls[s],) + tuple(cls[trans[s][z][0]] for z in ALPHABET)
        refined = _renumber(refined, states)
        if _n_classes(refined) == _n_classes(cls):
            break
        cls = refined

    return len(states), _n_classes(cls)


def _renumber(labels, states):
    """Map arbitrary hashable class labels onto 0,1,2,... in first-seen order."""
    index = {}
    out = {}
    for s in states:
        key = labels[s]
        if key not in index:
            index[key] = len(index)
        out[s] = index[key]
    return out


def _n_classes(labels):
    return len(set(labels.values()))


def survey(grid=GRID, families=FAMILIES):
    rows = []
    for family in families:
        for k, r in grid:
            if k < 2:
                continue
            s_closed = C.state_dimension(family, k, r)
            n_reach, n_min = minimise(family, k, r)
            rows.append((family, k, r, s_closed, n_reach, n_min,
                         "yes" if n_min == n_reach == s_closed else "no"))
            flag = "" if n_min == n_reach == s_closed else "   <-- not minimal"
            print("  %s k=%d r=%d: closed form %3d, reachable %3d, minimal %3d%s"
                  % (family, k, r, s_closed, n_reach, n_min, flag))
    return rows


def write_summary_table(rows, grid=((2, 1), (3, 1), (3, 2), (4, 2), (5, 3))):
    keep = {(k, r) for k, r in grid}
    out = []
    for family, k, r, s_closed, n_reach, n_min, ok in rows:
        if (k, r) in keep:
            out.append(["$\\mathcal{E}_%s$" % family[1], k, r,
                        s_closed, n_reach, n_min])

    out.sort(key=lambda row: (row[1], row[2], row[0]))

    write_table("tab_minimality.tex", latex_table(
        caption=("Minimality of the embedded dimension. The embedding is a "
                 "deterministic machine reading $\\{\\N,\\W,\\Cc\\}$ and "
                 "emitting the number of occurrences completed at each step; "
                 "its minimal form is computed by partition refinement. "
                 "``Closed form'' is the dimension given by "
                 "Theorem~\\ref{thm:strict} or \\ref{thm:threshold}, "
                 "``reachable'' the number of states reachable from the empty "
                 "buffer, and ``minimal'' the number of states after "
                 "refinement. Proposition~\\ref{prop:minimal} proves the three "
                 "columns equal for every $k$ and $r$; the table confirms it "
                 "numerically and would expose an error in either the "
                 "construction or the proof."),
        label="tab:minimality",
        colspec="l cc ccc",
        header=["Family", "$k$", "$r$", "Closed form", "Reachable", "Minimal"],
        rows=out))
    return out


def main():
    banner("Minimal Mealy form of the embedding versus the closed-form dimension")
    rows = survey()
    write_csv("minimality.csv",
              ("family", "k", "r", "closed_form", "reachable", "minimal",
               "minimal_equals_closed_form"), rows)
    write_summary_table(rows)

    bad = [row for row in rows if row[6] == "no"]
    banner("Summary")
    print("  %d settings examined" % len(rows))
    if bad:
        print("  %d settings where the closed form exceeds the minimal form:"
              % len(bad))
        for family, k, r, s_closed, n_reach, n_min, _ in bad:
            print("    %s k=%d r=%d: closed %d, reachable %d, minimal %d"
                  % (family, k, r, s_closed, n_reach, n_min))
    else:
        print("  the closed form is reachable and minimal in every setting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
