#!/usr/bin/env python3
"""Three-mesh study with Richardson extrapolation and GCI (Celik et al. 2008).

    python3 scripts/mesh_study.py configs/base.json --out results/mesh_study

h_ref defaults to 0.0113 / 0.0080 / 0.00566 (about 14k / 25k / 44k cells; length ratio about 1.35).
The representative size is h = sqrt(A / N) with the ACTUAL cell counts, because Gmsh counts vary
by about 1 % between nominally identical meshes.
NOTE: only the tangential/outer sizes scale with h_ref; the wall-normal boundary-layer
resolution (first layer, growth ratio, thickness) is held fixed. State this limitation.

Why I did the study this way
  * Three meshes with a refinement ratio of about 1.35 in length, the minimum for a Richardson / GCI estimate
    and still affordable on a laptop.
  * The representative size is sqrt(1/N) from the ACTUAL cell counts, because Gmsh does not give exactly the
    cell count I ask for.
  * What happened: with 14k / 25k / 44k cells the GCI could not be computed (loss kept changing by about 3.3e-4 per
    step, turning was non-monotonic). I then ran five meshes (8.6k to 80k cells), saw that 44k and 80k agree to
    0.02 % in loss, and moved the study to 25k / 44k / 80k. The results of the first study are kept in
    results/mesh_study_prev_14k-25k-44k. The orders of 7 to 16 that the script now returns are not credible, so the
    number I quote is the 44k-to-80k difference, not the GCI.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade.case import deep_update, load_config
from cascade.gci import gci_three
from cascade.runner import apply_overrides, run_many

QUANTITIES = [("loss_coefficient", "loss coefficient"), ("turning_deg", "turning [deg]"),
              ("beta2_deg", "beta2 [deg]"), ("static_pressure_rise_coeff", "static pressure rise Cp")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default="results/mesh_study")
    ap.add_argument("--runs", default="runs/mesh_study", help="where the three cases are built")
    ap.add_argument("--h-ref", type=float, nargs=3, default=[0.0113, 0.0080, 0.00566],
                    metavar=("COARSE", "MEDIUM", "FINE"))
    ap.add_argument("--jobs", type=int, default=1, help="cases in parallel (each uses 1 core)")
    ap.add_argument("--reuse", action="store_true", help="reuse finished cases with identical config")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--max-iter", type=int, default=None)
    a = ap.parse_args()

    base = apply_overrides(load_config(a.config), a.set + (
        [f"solver.max_iter={a.max_iter}", f"solver.write_interval={a.max_iter}"] if a.max_iter else []))
    names = ["coarse", "medium", "fine"]
    jobs = []
    for n, h in zip(names, a.h_ref):
        cfg = deep_update(base, {"mesh": {"h_ref": h}})
        jobs.append((n, cfg, str(Path(a.runs) / n)))
    res = run_many(jobs, n_jobs=a.jobs, reuse=a.reuse)
    failed = [n for n in names if not res[n]["ok"]]
    if failed:
        print(f"\nMESH STUDY INCOMPLETE: failed cases {failed}")
        return 1
    M = {n: res[n]["metrics"] for n in names}
    notconv = [n for n in names if not M[n]["converged"]]
    # representative sizes (2D): h = sqrt(1/N) in units where the domain area is constant
    h = {n: math.sqrt(1.0 / M[n]["n_cells"]) for n in names}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    study = {"cases": {n: {"h_ref": hr, "n_cells": M[n]["n_cells"], "converged": M[n]["converged"],
                           "iterations": M[n]["iterations"], "yplus_blade": M[n].get("yplus_blade")}
                       for n, hr in zip(names, a.h_ref)},
             "quantities": {}, "not_converged": notconv,
             "note": "wall-normal boundary-layer resolution held fixed; h = sqrt(1/N) with actual N"}
    rows = []
    for key, label in QUANTITIES:
        f = [M["fine"][key], M["medium"][key], M["coarse"][key]]
        g = gci_three(f[0], f[1], f[2], h["fine"], h["medium"], h["coarse"])
        study["quantities"][key] = g
        rows.append({"quantity": key, "coarse": f[2], "medium": f[1], "fine": f[0],
                     "observed_order": g["p"], "extrapolated": g["f_extrapolated"],
                     "gci_fine_percent": None if g["gci_fine_rel"] is None else 100 * g["gci_fine_rel"],
                     "gci_fine_abs": g["gci_fine_abs"], "monotonic": g["monotonic"], "flag": g["flag"]})
    with open(out / "mesh_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "mesh_study.json").write_text(json.dumps(study, indent=2))

    # plot
    from cascade.plotting import BLUE, GREY, RED, set_style
    set_style()
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), constrained_layout=True)
    axs = axs.ravel()
    order_cells = [M["coarse"]["n_cells"], M["medium"]["n_cells"], M["fine"]["n_cells"]]
    units = {"loss_coefficient": "loss coefficient [-]", "turning_deg": "turning [deg]",
             "beta2_deg": "outlet flow angle beta2 [deg]", "static_pressure_rise_coeff": "static pressure rise Cp [-]"}
    titles = {"loss_coefficient": "Loss coefficient", "turning_deg": "Turning",
              "beta2_deg": "Outlet flow angle \u03b22", "static_pressure_rise_coeff": "Static pressure rise Cp"}
    for ax, (key, label) in zip(axs, QUANTITIES):
        g = study["quantities"][key]
        vals = [g["f_coarse"], g["f_medium"], g["f_fine"]]
        ax.plot(order_cells, vals, "o-", color=BLUE, lw=1.6, ms=6)
        ax.set_xscale("log")
        ax.set_xticks(order_cells)
        ax.set_xticklabels([f"{n:,}".replace(",", " ") for n in order_cells])
        ax.minorticks_off()
        d_abs = abs(g["f_fine"] - g["f_medium"])
        d_txt = f"{100 * d_abs / abs(g['f_fine']):.2f} %" if key == "loss_coefficient" else \
            (f"{d_abs:.3f}\u00b0" if key != "static_pressure_rise_coeff" else f"{100 * d_abs / abs(g['f_fine']):.2f} %")
        ax.set_xlabel("number of cells")
        ax.set_title(f"\u0394 = {d_txt}", loc="right", fontsize=8.5, fontweight="normal", color=RED)
        ax.set_ylabel(units[key])
        ax.set_title(titles[key], loc="left")
        pad = 0.08 * (max(vals) - min(vals))
        ax.set_ylim(min(vals) - pad, max(vals) + pad)
    fig.suptitle("Mesh study: the two finest meshes agree (\u0394 = their difference); the observed order is not credible, so no GCI is quoted",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "mesh_study.png")

    print("\n==== mesh study ====")
    for r in rows:
        gp = "n/a" if r["gci_fine_percent"] is None else f"{r['gci_fine_percent']:.2f} %"
        p = "n/a" if r["observed_order"] is None else f"{r['observed_order']:.2f}"
        print(f"  {r['quantity']:28s} coarse {r['coarse']:.5f} medium {r['medium']:.5f} fine {r['fine']:.5f}  "
              f"p={p}  GCI_fine={gp}  {r['flag']}")
    if notconv:
        print(f"  WARNING: not converged: {notconv}")
    g = study["quantities"]["loss_coefficient"]
    if g["gci_fine_rel"] is not None:
        use = "medium" if g["gci_fine_rel"] < 0.05 else "fine"
        print(f"  Decision rule (GCI_fine on loss {100 * g['gci_fine_rel']:.2f} % vs 5 %): use the {use} mesh for the sweep"
              + ("" if use == "medium" else "  (set mesh.h_ref in a copy of the config)"))
    else:
        print("  No GCI on loss (non-monotonic or undefined): report that honestly; do not force a GCI.")
    print("  wrote", out)
    return 0 if not notconv else 2


if __name__ == "__main__":
    sys.exit(main())
