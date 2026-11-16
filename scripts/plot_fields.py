#!/usr/bin/env python3
"""Contour plots of |U|/V1 and the local total-pressure loss coefficient for a finished case.

    python3 scripts/plot_fields.py runs/base --out results/fields_base.png

Why these two fields: |U|/V1 shows whether the flow is smooth, attached and leaves with a thin wake; the local
total-pressure loss shows where the loss is made (blade boundary layers and wake). It is my visual check that a run
is physically sensible before I look at any number.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascade.plotting import draw_field, set_style
from cascade.postproc import FieldData, plane_averages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--out", default=None, help="output png (default: <case>/fields.png)")
    ap.add_argument("--periods", type=int, default=2, help="number of passages shown (1-3)")
    a = ap.parse_args()

    set_style()
    fd = FieldData(a.case)
    dom, meta = fd.dom, fd.meta
    V1 = meta["config"]["flow"]["V1"]
    a1 = plane_averages(fd.sample_plane(dom["x_plane_in"]))
    den = a1["p0_flux"] - a1["p_area"]

    mag = np.hypot(fd.U[:, 0], fd.U[:, 1]) / V1
    p0 = fd.p + 0.5 * (fd.U[:, 0] ** 2 + fd.U[:, 1] ** 2)
    loss = (a1["p0_flux"] - p0) / den
    outline = np.loadtxt(Path(a.case) / "blade_outline.csv", delimiter=",", skiprows=1)
    n_pass = max(1, min(3, a.periods))

    fig, axs = plt.subplots(2, 1, figsize=(9.5, 2.2 * n_pass * 2 + 0.6), constrained_layout=True, sharex=True)
    panels = ((axs[0], mag, "Velocity magnitude  |U| / V1", np.linspace(0, 1.6, 41), "viridis", "|U| / V1"),
              (axs[1], loss, "Local total-pressure loss", np.linspace(0, 0.6, 41), "magma_r",
               "(p01 - p0) / (p01 - p1)"))
    for ax, val, title, levels, cmap, label in panels:
        cs = draw_field(ax, fd, val, outline, levels, cmap, n_pass)
        ax.axvline(dom["x_plane_in"], color="white", ls=(0, (3, 3)), lw=0.9)
        ax.axvline(dom["x_plane_out"], color="white", ls=(0, (3, 3)), lw=0.9)
        ax.set_title(title, loc="left")
        ax.set_ylabel("y / c")
        cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.01, ticks=np.linspace(levels[0], levels[-1], 5))
        cb.set_label(label)
    axs[1].set_xlabel("x / c   (dashed: inlet and outlet measuring planes)")
    fig.suptitle(f"Baseline flow: {meta['mesh']['n_cells']:,} cells, converged in {int(float(fd.time))} iterations",
                 fontsize=12, fontweight="bold")
    out = Path(a.out) if a.out else Path(a.case) / "fields.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
