#!/usr/bin/env python3
"""Mesh, solve and post-process one case.

    python3 scripts/run_case.py configs/base.json --case runs/base
    python3 scripts/run_case.py configs/base.json --case runs/base --mesh-only
    python3 scripts/run_case.py configs/base.json --case runs/fine --set mesh.h_ref=0.00566

Outputs in the case folder: log.* files, metrics.json, blade_outline.csv, case_meta.json.

Why a single entry point: I run the baseline with this exact script, and the mesh study and the sweep call the same
code, so there is no "special" baseline. Exit code 2 means "ran, but the residual criterion was not reached", so a
shell loop cannot silently use an unconverged result.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.case import load_config
from cascade.runner import apply_overrides, run_one


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--case", required=True, help="case directory (recreated; must be empty or created by this code)")
    ap.add_argument("--mesh-only", action="store_true", help="stop after checkMesh")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config value, e.g. --set mesh.h_ref=0.0113 (repeatable)")
    ap.add_argument("--max-iter", type=int, default=None, help="shortcut for --set solver.max_iter=N")
    ap.add_argument("--legacy-cyclic", action="store_true",
                    help="createPatchDict without transform/separationVector (fallback if v2606 rejects them)")
    ap.add_argument("--no-yplus", action="store_true", help="skip the y+ evaluation")
    a = ap.parse_args()

    cfg = load_config(a.config)
    sets = list(a.set)
    if a.max_iter:
        sets += [f"solver.max_iter={a.max_iter}", f"solver.write_interval={a.max_iter}"]
    cfg = apply_overrides(cfg, sets)
    m = run_one(cfg, a.case, mesh_only=a.mesh_only, legacy_cyclic=a.legacy_cyclic, yplus=not a.no_yplus)
    if a.mesh_only:
        print(json.dumps(m, indent=2))
        return 0
    keys = ["converged", "iterations", "n_cells", "loss_coefficient", "turning_deg", "beta1_deg", "beta2_deg",
            "static_pressure_rise_coeff", "diffusion_factor", "mass_flow_ratio"]
    print("\n==== summary (" + str(Path(a.case) / "metrics.json") + ") ====")
    for k in keys:
        v = m[k]
        print(f"  {k:30s} {v:.5f}" if isinstance(v, float) else f"  {k:30s} {v}")
    c = m["carter"]
    print(f"  {'Carter turning (sanity only)':30s} {c['turning_deg']:.3f} deg  (beta2 {c['beta2_deg']:.3f})")
    if m.get("yplus_blade"):
        print(f"  {'y+ on blade':30s} {m['yplus_blade']}")
    if not m["converged"]:
        print("\nWARNING: residual criterion NOT reached; check log.simpleFoam before using these numbers.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
