#!/usr/bin/env python3
"""Compare a finished case with rows YOU typed into data/benchmark_naca1368_template.csv.

    python3 scripts/compare_benchmark.py runs/base

The CSV is empty on purpose: nothing is ever filled in or guessed here.
Columns: beta1_deg, incidence_deg, turning_deg, loss_coefficient, source
A row is used if its beta1 and incidence match the case within the tolerances (default 0.3 deg).
Confirm first: the report's Reynolds number, the exact blade designation, the measuring plane,
and the loss definition.

Why this script is so careful: the one benchmark row I have was read off a plot, so the script prints the error and the
list of error sources and nothing else. It never fills in or guesses a missing value, and with an empty file it tells
me to write "checked against Carter's rule and mass conservation only" instead of "validated".
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--benchmark", default=str(ROOT / "data/benchmark_naca1368_template.csv"))
    ap.add_argument("--tol", type=float, default=0.3, help="angle matching tolerance [deg]")
    a = ap.parse_args()

    mpath = Path(a.case) / "metrics.json"
    if not mpath.exists():
        print(f"{mpath} not found: run scripts/run_case.py first")
        return 1
    m = json.loads(mpath.read_text())
    meta = json.loads((Path(a.case) / "case_meta.json").read_text())
    if not m.get("converged"):
        print("WARNING: the case is not converged; the comparison is not meaningful.")

    rows = []
    with open(a.benchmark, newline="") as fh:
        for r in csv.DictReader(l for l in fh if l.strip() and not l.lstrip().startswith("#")):
            try:
                rows.append({k: (float(v) if k != "source" else v) for k, v in r.items() if v not in (None, "")})
            except ValueError:
                print("skipping unreadable row:", r)
    if not rows:
        print("The benchmark file has no data rows. Nothing to compare.")
        print("Do NOT write 'validated' anywhere. Say: 'checked against Carter's rule and mass conservation only'.")
        c = m["carter"]
        print(f"  (available checks: turning {m['turning_deg']:.2f} deg vs Carter {c['turning_deg']:.2f} deg, "
              f"mass flow ratio {m['mass_flow_ratio']:.5f})")
        return 0

    b1 = m["beta1_deg"]
    inc = b1 - meta["blade"]["inlet_metal_angle_deg"]
    print(f"case: beta1 = {b1:.2f} deg, incidence = {inc:+.2f} deg, Re = {meta['config']['flow']['reynolds']:.3g}")
    used = [r for r in rows if abs(r.get("beta1_deg", 1e9) - b1) <= a.tol and abs(r.get("incidence_deg", 1e9) - inc) <= a.tol]
    if not used:
        print(f"No benchmark row within {a.tol} deg of this operating point. Rows present:")
        for r in rows:
            print("  ", r)
        return 1
    print(f"\n{'quantity':16s} {'CFD':>10s} {'benchmark':>10s} {'error %':>9s}  source")
    for r in used:
        for key, cfd in (("turning_deg", m["turning_deg"]), ("loss_coefficient", m["loss_coefficient"])):
            if key in r:
                print(f"{key:16s} {cfd:10.4f} {r[key]:10.4f} {100 * (cfd - r[key]) / r[key]:9.2f}  {r.get('source', '')}")
    print("\nExpected error sources: fully turbulent model (no transition), 2D, Re / blade designation assumptions,")
    print("flux-averaged (not mixed-out) loss, measuring-plane position, tunnel effects in the experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
