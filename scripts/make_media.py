#!/usr/bin/env python3
"""Make the pictures and the animation used in the README from FINISHED cases (no new simulation).

    python3 scripts/make_media.py --sweep runs/deviation_sweep_fine --out results/media

Writes
  mesh.png              the 44k-cell mesh with zooms on the leading and trailing edge
  residuals.png         residual history of the baseline run
  variants.gif          local total-pressure loss field of the baseline and the six deviations, one frame each

Why this script exists: the solver is steady, so there is no time evolution to film. The only honest "movie" I can
make is a flip-book through the seven finished cases, which is what shows the effect of a deviation best.
The GIF is committed to the repo (runs/ is git-ignored), the cases themselves are not.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.collections import PolyCollection

from cascade import foam
from cascade.plotting import BLUE, INK, RED, draw_field, set_style
from cascade.postproc import FieldData, plane_averages, residual_history

# same order and labels as scripts/deviation_sweep.py
FRAMES = [("base", "baseline"), ("thick_minus5", "max thickness -5 %"), ("thick_plus5", "max thickness +5 %"),
          ("stagger_minus1", "stagger -1 deg"), ("stagger_plus1", "stagger +1 deg"),
          ("le_minus20", "LE bluntness -20 %"), ("le_plus20", "LE bluntness +20 %")]


def read_msh_front(msh: Path):
    """2D cells of the Gmsh v2.2 file: the elements of the physical group 'front' (z = 0 face of the extrusion)."""
    lines = msh.read_text().splitlines()
    names = {}
    i = lines.index("$PhysicalNames") + 2
    while lines[i] != "$EndPhysicalNames":
        _, tag, name = lines[i].split(maxsplit=2)
        names[name.strip('"')] = int(tag)
        i += 1
    i = lines.index("$Nodes") + 2
    nodes = {}
    while lines[i] != "$EndNodes":
        a = lines[i].split()
        nodes[int(a[0])] = (float(a[1]), float(a[2]))
        i += 1
    i = lines.index("$Elements") + 2
    polys = []
    while lines[i] != "$EndElements":
        a = lines[i].split()
        etype, ntags = int(a[1]), int(a[2])
        if etype in (2, 3) and int(a[3]) == names["front"]:
            polys.append([nodes[int(n)] for n in a[3 + ntags:]])
        i += 1
    return polys


def mesh_figure(case: Path, out: Path):
    polys = read_msh_front(case / "cascade.msh")
    meta = json.loads((case / "case_meta.json").read_text())
    outline = np.loadtxt(case / "blade_outline.csv", delimiter=",", skiprows=1)
    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    views = [(fig.add_subplot(gs[0, :]), None, "whole passage"),
             (fig.add_subplot(gs[1, 0]), (-0.03, 0.10, -0.05, 0.07), "leading edge"),
             (fig.add_subplot(gs[1, 1]), (0.78, 0.93, 0.40, 0.55), "trailing edge (blunt, with boundary-layer fan)")]
    for ax, box, title in views:
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="0.25", linewidths=0.25 if box is None else 0.5))
        ax.fill(outline[:, 0], outline[:, 1], color="0.75", zorder=3)
        ax.plot(outline[:, 0], outline[:, 1], color="C3", lw=0.8, zorder=4)
        ax.set_aspect("equal")
        if box is None:
            d = meta["domain"]
            ax.set_xlim(d["x_in"], d["x_out"])
            ax.set_ylim(d["y_bot"], d["y_top"])
        else:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2] + 0.0, box[3] + 0.0)
        ax.set_title(title, fontsize=10)
    # leading edge sits at the origin; the trailing edge at about (0.87, 0.5)
    fig.suptitle(f"Mesh of one blade passage: {meta['mesh']['n_cells']} cells, h_ref = {meta['mesh']['h_ref']}, "
                 f"first layer {meta['mesh'].get('wall_first_layer', 1e-4)} c", fontsize=11)
    fig.savefig(out)
    plt.close(fig)


def residual_figure(case: Path, out: Path):
    hist = residual_history(foam.log_text(case, "simpleFoam"))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, v in hist.items():
        ax.semilogy(np.arange(1, len(v) + 1), v, label={"Ux": "Ux", "Uy": "Uy", "p": "p", "k": "k", "omega": "\u03c9"}[name], lw=1.4)
    ax.set_xlabel("SIMPLE iteration")
    ax.set_ylabel("initial residual")
    ax.set_title("Baseline residual history (44k cells)", loc="left")
    ax.legend(ncol=5, loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def variants_gif(sweep: Path, out: Path, fps: float = 0.8):
    # one fixed normalisation (the baseline inlet plane) so the colours of all frames are comparable
    data = {}
    for n, label in FRAMES:
        data[n] = (FieldData(sweep / n), json.loads((sweep / n / "metrics.json").read_text()), label)
    fd0 = data["base"][0]
    a1 = plane_averages(fd0.sample_plane(fd0.dom["x_plane_in"]))
    den = a1["p0_flux"] - a1["p_area"]
    b = data["base"][1]
    levels = np.linspace(0, 0.6, 41)

    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 0.6), cmap="magma_r")
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.01, ticks=np.linspace(0, 0.6, 7))
    cb.set_label("local total-pressure loss  (p01 - p0) / (p01 - p1)")
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=110):
        for n, _ in FRAMES:
            fd, m, label = data[n]
            ax.clear()
            p0 = fd.p + 0.5 * (fd.U[:, 0] ** 2 + fd.U[:, 1] ** 2)
            outline = np.loadtxt(sweep / n / "blade_outline.csv", delimiter=",", skiprows=1)
            draw_field(ax, fd, (a1["p0_flux"] - p0) / den, outline, levels, "magma_r", 2, xlim=(-0.4, 2.4))
            ax.set_xlabel("x / c")
            ax.set_ylabel("y / c")
            dl = 100 * (m["loss_coefficient"] - b["loss_coefficient"]) / b["loss_coefficient"]
            dt = m["turning_deg"] - b["turning_deg"]
            ax.set_title(f"{label}", loc="left")
            ax.set_title(f"loss {m['loss_coefficient']:.4f} ({dl:+.2f} %)   turning {m['turning_deg']:.2f}\u00b0 ({dt:+.2f}\u00b0)"
                         if n != "base" else f"loss {m['loss_coefficient']:.4f}   turning {m['turning_deg']:.2f}\u00b0",
                         loc="right", fontsize=9.5, fontweight="normal")
            writer.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="runs/deviation_sweep_fine", help="folder with the seven finished sweep cases")
    ap.add_argument("--out", default="results/media")
    a = ap.parse_args()
    set_style()
    sweep, out = Path(a.sweep), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh_figure(sweep / "base", out / "mesh.png")
    print("wrote", out / "mesh.png")
    residual_figure(sweep / "base", out / "residuals.png")
    print("wrote", out / "residuals.png")
    variants_gif(sweep, out / "variants.gif")
    print("wrote", out / "variants.gif")


if __name__ == "__main__":
    main()
