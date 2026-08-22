# Python Codes

Everything the revised manuscript reports is computed here. Output goes to
`../Tables`, `../Figures` and `../Output`.

```bash
pip install -r requirements.txt
python test_core.py     # 101 correctness checks; run this first
python run_all.py       # regenerate every table and figure
python run_all.py --quick   # fast smoke test
```

## Libraries

| File | Role |
|---|---|
| `mcem_core.py` | The whole method. State spaces, the transition function, matrices, exact PMF (dense and sparse), exact moments, scalar recursions, an independent definition-based counter, brute-force enumeration and a vectorised simulator. |
| `legacy_matrices.py` | The transition matrices of the **first** submission. Used only by the audit, to quantify what changed. Nothing in the revised manuscript is computed from it. |
| `common.py` | Output paths, LaTeX table and matrix writers, environment stamp. |
| `test_core.py` | The correctness gate. Run after any change to `mcem_core.py`. |

### The design rule

`mcem_core.step(state, z, family, k, r)` is the single source of truth. It
returns the next embedded state and whether a count occurred, and it *is*
Algorithm 1 of the manuscript. Both the transition matrices
(`build_matrices`) and the path counter (`count_occurrences`) are built from
it, so they cannot drift apart.

The independent check is `count_by_definition`, which is written the naive
way — buffer the positions of the critical states, look at the window formed by
the last $k$ of them, test its gap profile — and never touches `A` or `B`. Test
`T1` compares the two on every sequence in $\{N,W,C\}^n$ for seven $(k,r)$
settings. That comparison is what caught the two errors this revision fixes.

## Scripts

| Script | Produces |
|---|---|
| `code_01_matrices.py` | Table 3; the Appendix A matrices; structural checks over a $(k,r)$ grid |
| `code_02_benchmark.py` | Table 5, Figure 1 — dense, sparse, scalar and auto-generated recursion, medians with quartiles |
| `code_03_scaling.py` | Table 6, Figure 2 — cost as a function of the embedded dimension $s$ |
| `code_04_validation.py` | Table 7 — $10^6$ Monte Carlo replications counted independently of the embedding |
| `code_05_tolerance.py` | Tables 8 and 9 — interruption tolerance and alarm operating characteristics |
| `code_06_profiles.py` | Table 10 — profile sensitivity, with three moment routines cross-checked |
| `code_07_distributions.py` | Tables 11–14 — exact distributions, with unimodality computed not assumed |
| `code_08_recursions.py` | Table 4, Appendix B — scalar recursions for arbitrary $(k,r,\text{family})$ |
| `code_09_audit.py` | Table 15 — the verification appendix, including the comparison with the first submission |
| `code_10_figures.py` | Figures 3 and 4, plus the computed stochastic-ordering verdict |

Each script takes command-line arguments for the parameters it varies; run any
of them with `--help`.

## Key API

```python
import mcem_core as C

p = C.symmetric_p(0.3)              # (p0, p1, p2) with p0 = p1 = (1-p2)/2

f  = C.pmf(30, "E2", 5, 2, p)       # exact PMF, dense embedding
f2 = C.pmf_sparse(30, "E2", 5, 2, p)
mu, var = C.moments(30, "E2", 5, 2, p)      # Proposition 2, no PMF formed

C.state_dimension("E2", 5, 2)               # 31
C.build_matrices("E2", 5, 2, p)             # (A, B)

seq = C.read_path("CCCWWC")
C.count_occurrences(seq, "E2", 3, 2)        # via the embedded chain
C.count_by_definition(seq, "E2", 3, 2)      # via the definition; must agree

rec = C.scalar_recursion("E2", 3, 1, p)     # numeric coefficients c_j(z)
C.pmf_from_scalar_recursion(40, "E2", 3, 1, p, rec)

C.simulate_vectorised(30, "E4", 5, 2, p, 1_000_000, seed=1)
```

Symbolic matrices for the appendix:

```python
import sympy as sp
A, B = C.build_matrices("E2", 3, 2, sp.symbols("p_0 p_1 p_2"), dtype=object)
```

## Requirements

`numpy`, `scipy`, `matplotlib`, `sympy`; Python 3.9 or later.

## Licence

MIT.
