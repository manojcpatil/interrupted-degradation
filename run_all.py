"""
run_all.py
==========
Regenerate every table and figure of the revised manuscript.

    python run_all.py            full run (Monte Carlo 10^6, benchmark to n=5000)
    python run_all.py --quick    fast pass for a smoke test

Tables land in ../Tables, figures in ../Figures, data and logs in ../Output.
The manuscript reads them with \\input{Tables/...} and \\graphicspath{{Figures/}},
so recompiling after a run picks the new numbers up automatically.
"""

from __future__ import annotations

import argparse
import io
import sys
import traceback
from contextlib import redirect_stdout

from common import banner, write_output

STEPS = [
    ("code_01_matrices", "Table 3 and the appendix matrices"),
    ("code_09_audit", "exhaustive checks behind Section 5"),
    ("code_11_reduction", "Table 8: reduction to the binary case at p0 = 0"),
    ("code_08_recursions", "Table 4 and Appendix B: scalar recursions"),
    ("code_02_benchmark", "Table 5 and Figure 1: benchmark"),
    ("code_03_scaling", "Table 6 and Figure 2: cost versus s"),
    ("code_04_validation", "Table 7: Monte Carlo validation"),
    ("code_05_tolerance", "Tables 9 and 10: interruption tolerance"),
    ("code_06_profiles", "Table 11: profile sensitivity"),
    ("code_07_distributions", "Tables 12-15: exact distributions"),
    ("code_10_figures", "Figures 3 and 4"),
]


def main(quick=False):
    log = io.StringIO()
    failures = []
    for name, title in STEPS:
        banner("%s - %s" % (name, title))
        buf = io.StringIO()
        try:
            mod = __import__(name)
            with redirect_stdout(buf):
                if name == "code_09_audit":
                    mod.main(quick=quick)
                elif name == "code_04_validation":
                    mod.main(trials=50_000 if quick else 1_000_000)
                elif name == "code_02_benchmark":
                    mod.main(ns=(100, 500, 1000) if quick else mod.DEFAULT_NS,
                             repeats=3 if quick else 5)
                elif name == "code_03_scaling":
                    mod.main(n=800 if quick else 2000)
                elif name == "code_08_recursions":
                    mod.main(nmax=40 if quick else 80)
                elif name == "code_10_figures":
                    mod.sensitivity_grid(30, 5, 2, [0.3, 0.5, 0.7, 0.9])
                    mod.cdf_comparison(30, 5, 2, 0.6)
                    mod.dominance(30, 5, 2, 0.6)
                else:
                    mod.main()
        except Exception:
            failures.append(name)
            buf.write("\nFAILED\n" + traceback.format_exc())
        text = buf.getvalue()
        print(text)
        log.write("\n" + "=" * 78 + "\n%s - %s\n" % (name, title)
                  + "=" * 78 + "\n" + text)

    write_output("run_all_log.txt", log.getvalue())
    banner("Summary")
    if failures:
        print("  failed: %s" % ", ".join(failures))
        return 1
    print("  all %d steps completed" % len(STEPS))
    print("  tables -> ../Tables, figures -> ../Figures, data -> ../Output")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    sys.exit(main(quick=ap.parse_args().quick))
