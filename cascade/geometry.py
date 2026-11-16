"""Blade geometry: NACA 65-(12)10 type section with manufacturing-like deviations.

Conventions:
  * x is axial, y is tangential (pitch direction), z is the 2D normal.
  * Angles are measured from the axial direction, positive towards +y.
  * Chord line starts at the leading edge (LE) at the origin and points along
    (cos(stagger), sin(stagger)).  Camber bulges to the left of the chord
    direction, so the blade turns the flow towards the axial direction.

Deviations (one at a time in the study):
  le_scale            scales the leading-edge radius (bluntness) by this factor.
                      Implemented as a local thickness factor sqrt(le_scale) that
                      fades out with exp(-x/le_blend); nose radius ~ t^2, so the
                      radius scales with le_scale.  Max thickness is not touched.
  thickness_scale     scales the whole thickness distribution (max thickness).
  stagger_delta_deg   rotates the blade about its mid-chord point.

The thickness table data/naca65_010_thickness.csv was checked against the NACA tables (see README).

Why I built the blade this way
  * NACA 65-(12)10: this is the blade family of the NACA low-speed cascade tests (RM L51G31 / Report 1368), so there is
    at least one published point I can compare my CFD with. A made-up blade would have nothing to be compared with.
  * Thickness and mean line are kept as two separate functions (65-010 thickness on an a = 1.0 mean line), exactly how
    the NACA series is defined, so thickness_scale really changes only the thickness and nothing else.
  * The three deviations are the ones I can imagine from manufacturing or assembly: a blunter or sharper leading edge,
    a thicker or thinner blade, and a blade set at a slightly wrong angle (stagger). I change one at a time so each
    effect can be read on its own.
  * The stagger deviation rotates the blade about mid-chord, not about the leading edge, so the blade stays inside the
    same passage and the domain (made from the NOMINAL blade, see nominal_frame) is identical for all seven cases.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE = ROOT / "data" / "naca65_010_thickness.csv"


@dataclass
class BladeParams:
    chord: float = 1.0
    stagger_deg: float = 30.0
    cl0: float = 1.2
    camber_per_cl0_deg: float = 25.3      # camber angle estimate, ASSUMPTION
    te_half_thickness: float = 0.0015     # half thickness added at the TE (blunt TE, meshing)
    le_blend: float = 0.04                # x/c length scale of the LE-bluntness blend
    x_end: float = 0.9995                 # last sampled chordwise station (avoid x=1 singularity)
    n_points: int = 140                   # points per surface
    le_scale: float = 1.0
    thickness_scale: float = 1.0
    stagger_delta_deg: float = 0.0
    table: str = str(DEFAULT_TABLE)

    @classmethod
    def from_config(cls, cfg: dict) -> "BladeParams":
        b = dict(cfg.get("blade", {}))
        d = dict(cfg.get("deviation", {}))
        keys = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in b.items() if k in keys}
        kw.update({k: v for k, v in d.items() if k in keys})
        table = kw.get("table")
        if table is not None and not Path(table).is_absolute():
            kw["table"] = str(ROOT / table)
        return cls(**kw)

    @property
    def camber_deg(self) -> float:
        return self.camber_per_cl0_deg * self.cl0

    @property
    def stagger_eff_deg(self) -> float:
        return self.stagger_deg + self.stagger_delta_deg

    @property
    def inlet_metal_angle_deg(self) -> float:
        return self.stagger_eff_deg + 0.5 * self.camber_deg

    @property
    def outlet_metal_angle_deg(self) -> float:
        return self.stagger_eff_deg - 0.5 * self.camber_deg


def load_thickness_table(path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    header, data = rows[0], rows[1:]
    if [h.strip() for h in header] != ["x_percent", "y_percent"]:
        raise ValueError(f"{path}: expected header x_percent,y_percent, got {header}")
    for r in data:
        xs.append(float(r[0]) / 100.0)
        ys.append(float(r[1]) / 100.0)
    x = np.array(xs)
    y = np.array(ys)
    if x[0] != 0.0 or abs(x[-1] - 1.0) > 1e-12 or np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: x must increase strictly from 0 to 100 percent")
    y[0] = 0.0
    return x, y


_THICKNESS_CACHE: dict = {}


def base_half_thickness(x, table=DEFAULT_TABLE):
    """Half thickness y_t(x)/c of the 65-010 form. Interpolated in sqrt(x) so the
    nose keeps its sqrt(x) behaviour; PCHIP keeps it monotone (no overshoot).
    My reason: a plain spline through the table wiggles near the nose, and the nose is exactly where I want to
    change the bluntness, so a wiggle there would pollute the LE result."""
    key = str(table)
    if key not in _THICKNESS_CACHE:
        tx, ty = load_thickness_table(table)
        _THICKNESS_CACHE[key] = PchipInterpolator(np.sqrt(tx), ty)
    return _THICKNESS_CACHE[key](np.sqrt(np.clip(x, 0.0, 1.0)))


def mean_line(x, cl0):
    """NACA a = 1.0 mean line, y_c and dy_c/dx (slope is log-singular at both ends)."""
    x = np.asarray(x, dtype=float)
    k = cl0 / (4.0 * math.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(x > 0, x * np.log(x), 0.0)
        b = np.where(x < 1, (1 - x) * np.log(1 - x), 0.0)
        yc = -k * (a + b)
        slope = k * (np.log(1 - x) - np.log(x))
    return yc, slope


def half_thickness(x, p: BladeParams):
    x = np.asarray(x, dtype=float)
    # the 65-010 table ends in a sharp trailing edge; I add a small linear thickness (0.0015 c at the TE) so the
    # trailing edge is a finite face (0.003 c) that Gmsh can put a boundary layer on, for meshing robustness.
    yt = base_half_thickness(x, p.table) + p.te_half_thickness * x
    # LE bluntness: nose radius is proportional to (local thickness)^2, so scaling the local thickness by sqrt(le_scale)
    # scales the radius by le_scale. The factor fades out with exp(-x/0.04) so the maximum thickness stays untouched
    # (otherwise the "LE" case would secretly also be a thickness case).
    if p.le_scale != 1.0:
        yt = yt * (1.0 + (math.sqrt(p.le_scale) - 1.0) * np.exp(-x / p.le_blend))
    return yt * p.thickness_scale


def chord_stations(n: int, x_end: float) -> np.ndarray:
    """Cosine spacing from 0 to x_end (dense at LE, fairly dense at TE)."""
    u = np.linspace(0.0, 1.0, n)
    return x_end * 0.5 * (1.0 - np.cos(math.pi * u))


def surfaces(p: BladeParams):
    """Return (upper, lower) arrays (n,2) in chord coordinates (xi, eta), LE -> TE.
    'upper' is the suction-like side (the camber-bulge side)."""
    x = chord_stations(p.n_points, p.x_end)
    yc, slope = mean_line(x, p.cl0)
    th = np.arctan(slope)
    yt = half_thickness(x, p)
    upper = np.column_stack((x - yt * np.sin(th), yc + yt * np.cos(th)))
    lower = np.column_stack((x + yt * np.sin(th), yc - yt * np.cos(th)))
    return upper, lower


def rotate(pts: np.ndarray, angle_deg: float, about=(0.0, 0.0)) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    r = np.array([[c, -s], [s, c]])
    o = np.asarray(about)
    return (pts - o) @ r.T + o


def blade_outline(p: BladeParams) -> dict:
    """Blade in the cascade frame (LE of the NOMINAL blade at the origin).

    Returns dict with 'upper', 'lower' (LE->TE, (n,2)), 'closed' polygon
    (counter-clockwise start at TE-lower ... see below) and key points.
    """
    up, lo = surfaces(p)
    cs = p.chord
    up, lo = up * cs, lo * cs
    # nominal stagger first, then the deviation as a rotation about mid-chord
    up = rotate(up, p.stagger_deg)
    lo = rotate(lo, p.stagger_deg)
    mid = np.array([0.5 * cs * math.cos(math.radians(p.stagger_deg)),
                    0.5 * cs * math.sin(math.radians(p.stagger_deg))])
    if p.stagger_delta_deg != 0.0:
        up = rotate(up, p.stagger_delta_deg, mid)
        lo = rotate(lo, p.stagger_delta_deg, mid)
    # closed polygon: LE point -> upper surface to TE -> TE face -> lower surface back to LE
    poly = np.vstack((up, lo[::-1][:-1]))
    return {"upper": up, "lower": lo, "polygon": poly, "mid_chord": mid}


def nominal_frame(p: BladeParams) -> dict:
    """Quantities of the NOMINAL blade (no deviation) that fix the domain."""
    a = math.radians(p.stagger_deg)
    ax = p.chord * math.cos(a)
    le = np.array([0.0, 0.0])
    te = np.array([p.chord * math.cos(a), p.chord * math.sin(a)])
    return {"axial_chord": ax, "le": le, "te": te, "mid": 0.5 * (le + te)}


def max_thickness(p: BladeParams) -> float:
    x = np.linspace(0, 1, 2001)
    return float(2 * np.max(half_thickness(x, p)))


def polygon_self_intersects(poly: np.ndarray) -> bool:
    """O(n^2) check of a closed polygon for self-intersection (n ~ 300, fine)."""
    n = len(poly)
    a = poly
    b = np.roll(poly, -1, axis=0)

    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            d1 = ccw(a[i], b[i], a[j])
            d2 = ccw(a[i], b[i], b[j])
            d3 = ccw(a[j], b[j], a[i])
            d4 = ccw(a[j], b[j], b[i])
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
