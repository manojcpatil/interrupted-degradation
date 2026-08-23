"""
run_all.py
==========
Regenerate every table and figure in the manuscript.

    python run_all.py            full run (10^6 simulated paths)
    python run_all.py --quick    fast pass for a smoke test

Tables land in Tables/, figures in Figures/, data and logs in Output/, resolved
by common.ROOT: beside the manuscript when these scripts sit in "Python Codes/",
and beside the scripts themselves in the flat repository layout. The manuscript
reads them with \\input{Tables/...} and \\graphicspath{{Figures/}}, so recompiling
after a run picks up the new numbers.
"""

from __future__ import annotations

import argparse
import io
import sys
import traceback
from contextlib import redirect_stdout

from common import ROOT, banner, write_output

STEPS = [
    ("code_01_matrices", "embedded dimensions and the appendix matrices"),
    ("code_06_minimality", "minimality of the embedded dimensions"),
    ("code_02_recursions", "scalar recursions and Appendix B"),
    ("code_03_verify", "exhaustive, brute-force and Monte Carlo checks"),
    ("code_05_decomposition", "embedding against the i.i.d. block decomposition"),
    ("code_07_alarm", "calibrated monitoring study: ROC, run length, lever"),
    ("code_04_numerical", "moments, shape and the mass function figures"),
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
                if name == "code_03_verify":
                    mod.main(trials=50_000 if quick else 1_000_000)
                elif name == "code_02_recursions":
                    mod.main(nmax=40 if quick else 80)
                elif name == "code_06_minimality":
                    mod.main()
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
    print("  output written under %s" % ROOT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    sys.exit(main(quick=ap.parse_args().quick))
