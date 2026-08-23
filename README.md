# Interrupted Patterns on an Ordinal Alphabet

Replication code for *Interrupted Patterns on an Ordinal Alphabet: Minimal
Markov Chain Embeddings and Exact Distributions*.

Four interrupted pattern families are defined on the alphabet
`{normal, warning, critical}`, in which `k` critical events must occur with
limited interruption between them. The normal state resets the accumulated
pattern history; the warning state interrupts it without resetting. Their exact
distributions are obtained by finite Markov chain embedding, the embedded
dimensions are shown to be minimal, and the joint generating function of the
count and the horizon yields linear recursions that compute the mass function
and the moments without forming a matrix.

Every number, table and figure in the paper is produced by this code.

## Reproducing the paper

```bash
pip install -r requirements.txt
```

```bash
python run_all.py
```

For a fast check that everything executes — smaller simulation, shorter
symbolic pass — use:

```bash
python run_all.py --quick
```

The `--quick` pass is a smoke test. It does **not** reproduce the published
numbers; the full run is required for those.

## Layout

The scripts sit at the top level of this repository and write three directories
beside themselves:

```
run_all.py            runs everything, in order
requirements.txt      pinned dependencies
embedding.py          the core module
common.py             paths, LaTeX writers, environment stamp
code_*.py             one script per group of results
Tables/               generated LaTeX fragments, read by the manuscript
Figures/              generated PNG figures
Output/               CSV data, logs, environment.json
```

Nothing in `Tables/`, `Figures/` or `Output/` should be edited by hand; all
three are regenerated from scratch on every run.

Alongside the manuscript the same scripts live in a `Python Codes/` folder and
write to its siblings instead. `common.py` detects which layout it is in, so no
script needs editing either way.

## What each script does

Run in this order by `run_all.py`:

| Script | Produces |
| --- | --- |
| `embedding.py` | core module: state spaces, transition matrices, mass function, generating function, scalar recursions, simulator |
| `code_01_matrices.py` | embedded dimensions and the appendix matrices; checks row sums and state counts |
| `code_06_minimality.py` | minimality of the embedded dimensions by partition refinement, against the closed forms |
| `code_02_recursions.py` | scalar recursions; extracts them from `det[I - w(A+zB)]` and checks against the matrix form |
| `code_03_verify.py` | exhaustive path audit, brute-force exact probabilities, Monte Carlo |
| `code_05_decomposition.py` | embedding against the i.i.d. block decomposition |
| `code_07_alarm.py` | calibrated monitoring study: ROC, run length, lever |
| `code_04_numerical.py` | moments, distribution shape, mass function figures |

## How the verification is arranged

The claim the code exists to support is that each embedded chain counts the
random variable the paper defines, and not some neighbouring one. An error in a
transition matrix does not announce itself — it produces a well-behaved
distribution of the wrong thing — so every check uses a comparator that does not
touch the matrices:

- **Exhaustive enumeration.** Every sequence in `{N,W,C}^n` for `n = 9, 10` and
  four `(k, r)` settings, counted by an implementation written directly from the
  definition. 1,259,712 paths.
- **Brute force.** All `3^10` paths weighted by probability, giving the mass
  function with no chain involved.
- **Simulation.** Paths counted by the scanner rather than by the embedded
  chain, which is what makes the comparison informative.
- **Block decomposition.** The count assembled from within-block laws through a
  renewal recursion, forming neither `A` nor `B`.
- **The binary boundary.** Setting `p0 = 0` reproduces the binary
  interrupted-run statistic of Dafnis, Makri and Koutras (2021) — a comparator
  from outside the paper entirely.

`embedding.step` is the only place the pattern is described. The transition
matrices and the path counter are both built from it, so they cannot drift
apart, and `count_by_definition` is a second, independent implementation
transcribed from the definition for the audits to compare against.

## Requirements

`numpy`, `sympy`, `scipy`, `matplotlib`. Versions are pinned in
`requirements.txt` to those used for the published results; the full environment
stamp is written to `Output/environment.json` on every run.

No result depends on execution timing, so none of it is sensitive to the machine
used.

## Citation

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
