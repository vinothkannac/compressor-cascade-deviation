#!/usr/bin/env python3
"""Baseline + six one-at-a-time geometry deviations.

    python3 scripts/deviation_sweep.py configs/base.json --out results/deviation_sweep

LE bluntness -20 / +20 % (le_scale), max thickness -5 / +5 % (thickness_scale),
stagger -1 / +1 deg (stagger_delta_deg).  An effect smaller than the numerical uncertainty
(GCI from results/mesh_study/mesh_study.json, and a mesh-noise floor of 1 % of the baseline value)
is reported as NOT RESOLVED, not as a result.

Why these choices
  * Sizes of the deviations: leading-edge bluntness +-20 %, maximum thickness +-5 %, stagger +-1 deg. They are small,
    the kind of scatter I would expect between manufactured parts, but big enough that I hoped to see them.
  * One change at a time, each against the same baseline, so an effect can be attributed.
  * A result counts as RESOLVED only if it is larger than the numerical uncertainty. For loss and turning I use a
    1 % floor on the baseline value (--noise-rel, my estimate of the mesh noise); the GCI is taken if it is bigger.
    Anything smaller is reported as "not resolved". (My own first mesh pair differed by about 2 % in loss.)
  * The sanity warnings at the end (blunter LE should not lower the loss, a thicker blade should not lower it,
    stagger should move beta2 by about the same angle) are there so that a result that contradicts basic
    expectations sends me looking for a bug first and writing it up second.
  * Run on the 44k mesh (--set mesh.h_ref=0.00566), not the 25k mesh, because 25k is about 1.9 % off in loss.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade.case import deep_update, load_config
from cascade.runner import apply_overrides, run_many

VARIANTS = [
    ("base", {}, "baseline"),
    ("le_minus20", {"le_scale": 0.8}, "LE bluntness -20 %"),
    ("le_plus20", {"le_scale": 1.2}, "LE bluntness +20 %"),
    ("thick_minus5", {"thickness_scale": 0.95}, "max thickness -5 %"),
    ("thick_plus5", {"thickness_scale": 1.05}, "max thickness +5 %"),
    ("stagger_minus1", {"stagger_delta_deg": -1.0}, "stagger -1 deg"),
    ("stagger_plus1", {"stagger_delta_deg": 1.0}, "stagger +1 deg"),
]


def uncertainty(study_json: Path):
    """GCI_fine absolute values for loss and turning, or None."""
    if not study_json.exists():
        return None
    q = json.loads(study_json.read_text())["quantities"]
    return {k: q[k]["gci_fine_abs"] for k in ("loss_coefficient", "turning_deg") if q.get(k)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default="results/deviation_sweep")
    ap.add_argument("--runs", default="runs/deviation_sweep")
    ap.add_argument("--mesh-study", default="results/mesh_study/mesh_study.json")
    ap.add_argument("--noise-rel", type=float, default=0.01, help="mesh-noise floor, fraction of the baseline value")
    ap.add_argument("--jobs", type=int, default=1, help="cases in parallel")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override for ALL cases, e.g. --set mesh.h_ref=0.00566")
    ap.add_argument("--max-iter", type=int, default=None)
    a = ap.parse_args()

    base = apply_overrides(load_config(a.config), a.set + (
        [f"solver.max_iter={a.max_iter}", f"solver.write_interval={a.max_iter}"] if a.max_iter else []))
    jobs = [(n, deep_update(base, {"deviation": dev}), str(Path(a.runs) / n)) for n, dev, _ in VARIANTS]
    res = run_many(jobs, n_jobs=a.jobs, reuse=a.reuse)
    failed = [n for n, _, _ in VARIANTS if not res[n]["ok"]]
    if failed:
        print(f"\nSWEEP INCOMPLETE: failed cases {failed}")
        return 1
    M = {n: res[n]["metrics"] for n, _, _ in VARIANTS}
    b = M["base"]
    unc = uncertainty(Path(a.mesh_study))
    if unc is None:
        print(f"\nNOTE: {a.mesh_study} not found: only the {100 * a.noise_rel:.0f} % noise floor is used "
              "to judge resolution (run scripts/mesh_study.py first).")
    floor_loss = a.noise_rel * abs(b["loss_coefficient"])
    floor_turn = a.noise_rel * abs(b["turning_deg"])
    u_loss = max(floor_loss, (unc or {}).get("loss_coefficient") or 0.0)
    u_turn = max(floor_turn, (unc or {}).get("turning_deg") or 0.0)

    rows = []
    for n, _, label in VARIANTS:
        m = M[n]
        d_loss = m["loss_coefficient"] - b["loss_coefficient"]
        d_turn = m["turning_deg"] - b["turning_deg"]
        rows.append({
            "case": n, "description": label, "converged": bool(m["converged"]), "n_cells": int(m["n_cells"]),
            "loss": m["loss_coefficient"], "loss_change_percent": 100 * d_loss / b["loss_coefficient"],
            "turning_deg": m["turning_deg"], "turning_change_deg": d_turn,
            "beta2_deg": m["beta2_deg"], "beta2_change_deg": m["beta2_deg"] - b["beta2_deg"],
            "diffusion_factor": m["diffusion_factor"], "df_change": m["diffusion_factor"] - b["diffusion_factor"],
            "loss_resolved": bool(abs(d_loss) > u_loss) if n != "base" else "",
            "turning_resolved": bool(abs(d_turn) > u_turn) if n != "base" else "",
        })
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "deviation_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "deviation_sweep.json").write_text(json.dumps(
        {"rows": rows, "uncertainty": {"loss_abs": u_loss, "turning_deg": u_turn,
                                       "mesh_study_used": unc is not None, "noise_rel": a.noise_rel}}, indent=2))

    # plot: sorted by |loss effect|
    from cascade.plotting import BLUE, GREY, RED, set_style
    from matplotlib.patches import Patch
    set_style()
    dev = [r for r in rows if r["case"] != "base"]
    order = sorted(dev, key=lambda r: abs(r["loss_change_percent"]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, key, ukey, thr, xl, fmt in (
        (axs[0], "loss_change_percent", "loss_resolved", 100 * u_loss / b["loss_coefficient"],
         "change in loss coefficient [%]", "{:+.2f} %"),
        (axs[1], "turning_change_deg", "turning_resolved", u_turn, "change in turning [deg]", "{:+.2f}\u00b0"),
    ):
        vals = [r[key] for r in order]
        cols = [BLUE if r[ukey] else GREY for r in order]
        ypos = range(len(order))
        ax.axvspan(-thr, thr, color=RED, alpha=0.10, lw=0, label="numerical uncertainty")
        ax.barh(list(ypos), vals, color=cols, height=0.62, zorder=3)
        span = max(abs(v) for v in vals) * 1.35
        ax.set_xlim(-span, span)
        for y, v in zip(ypos, vals):
            lab = fmt.format(v).replace("-0.00", "0.00").replace("+0.00", "0.00")
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, lab, va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, zorder=4)
        ax.axvline(0, color="black", lw=0.8, zorder=4)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([r["description"] for r in order])
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=BLUE, label="resolved"), Patch(color=GREY, label="not resolved"),
                        Patch(color=RED, alpha=0.25, label="numerical uncertainty (1 % floor)")],
               loc="outside lower center", ncol=3)
    fig.suptitle("Effect of the six geometry deviations on loss and turning (44k-cell mesh)", fontsize=11,
                 fontweight="bold")
    fig.savefig(out / "deviation_sweep.png")

    print("\n==== deviation sweep ====")
    print(f"  baseline: loss {b['loss_coefficient']:.5f}, turning {b['turning_deg']:.3f} deg, beta2 {b['beta2_deg']:.3f}, DF {b['diffusion_factor']:.4f}")
    print(f"  uncertainty used: loss +-{u_loss:.5f} ({100 * u_loss / b['loss_coefficient']:.2f} %), turning +-{u_turn:.3f} deg")
    for r in dev:
        print(f"  {r['description']:22s} loss {r['loss']:.5f} ({r['loss_change_percent']:+.2f} %, "
              f"{'resolved' if r['loss_resolved'] else 'NOT RESOLVED'})  turning {r['turning_deg']:.3f} "
              f"({r['turning_change_deg']:+.3f}, {'resolved' if r['turning_resolved'] else 'NOT RESOLVED'})  "
              f"beta2 {r['beta2_change_deg']:+.3f}  conv={r['converged']}")
    # physical sanity checks (warnings only; a contradiction means: look for a bug before writing it up)
    by = {r["case"]: r for r in rows}
    warn = []
    if by["le_plus20"]["loss_change_percent"] < by["le_minus20"]["loss_change_percent"]:
        warn.append("blunter LE does not give higher loss than sharper LE")
    if by["thick_plus5"]["loss_change_percent"] < by["thick_minus5"]["loss_change_percent"]:
        warn.append("thicker blade does not give higher loss than thinner blade")
    for n, sgn in (("stagger_plus1", 1), ("stagger_minus1", -1)):
        d = by[n]["beta2_change_deg"]
        if d * sgn < 0.3 or d * sgn > 1.7:
            warn.append(f"{n}: beta2 shift {d:+.2f} deg is far from the expected ~{sgn:+d} deg")
    for w in warn:
        print("  SANITY WARNING:", w)
    nc = [r["case"] for r in rows if not r["converged"]]
    if nc:
        print("  WARNING: not converged:", nc)
    print("  wrote", out)
    return 0 if not nc else 2


if __name__ == "__main__":
    sys.exit(main())
