"""Shared plot style and the field drawing used by every figure of this project.

Why one module: I wanted all figures (README, mesh study, sweep, GIF) to look like one set: same fonts, same colours,
same colour scales. The colours are colour-blind safe (viridis for speed, a sequential magma for loss, one blue and one grey
for "resolved" and "not resolved").
"""
from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np
from matplotlib.path import Path as MPath

BLUE = "#1f5fa8"      # resolved effect
GREY = "#a6a6a6"      # not resolved
RED = "#c0392b"       # uncertainty band / marks
INK = "#222222"


def set_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
    })


def periodic_triangulation(xy: np.ndarray, outline: np.ndarray, pitch: float, n_pass: int = 2):
    """Triangulation of n_pass stacked passages.

    The cell centres stop half a cell short of the periodic boundary, so drawing each passage alone leaves a white seam.
    I triangulate the points of one extra passage above and below as well, so the seam is filled with real data, and I
    mask the triangles inside the blades and the long ones that bridge a blade.
    Returns (triangulation, index of the source cell of every point).
    """
    ks = range(-1, n_pass + 1)
    pts = np.vstack([xy + [0.0, k * pitch] for k in ks])
    src = np.concatenate([np.arange(len(xy)) for _ in ks])
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    cen = pts[tri.triangles].mean(axis=1)
    mask = np.max(np.ptp(pts[tri.triangles], axis=1), axis=1) > 0.2
    for k in ks:
        mask |= MPath(outline + [0.0, k * pitch]).contains_points(cen)
    tri.set_mask(mask)
    return tri, src


def draw_field(ax, fd, values, outline, levels, cmap, n_pass=2, xlim=None):
    """Filled contours of a cell-centred field over n_pass periodic passages, blades in dark grey."""
    s = fd.pitch
    tri, src = periodic_triangulation(fd.mesh.xy, outline, s, n_pass)
    cs = ax.tricontourf(tri, np.asarray(values)[src], levels=levels, cmap=cmap, extend="both")
    for k in range(-1, n_pass + 1):
        ax.fill(outline[:, 0], outline[:, 1] + k * s, facecolor="#3a3a3a", edgecolor="black", linewidth=0.5, zorder=3)
    d = fd.dom
    ax.set_aspect("equal")
    ax.set_xlim(*(xlim or (d["x_in"], d["x_out"])))
    ax.set_ylim(d["y_bot"], d["y_bot"] + n_pass * s)
    ax.grid(False)
    return cs
