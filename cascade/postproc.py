"""Post-processing in Python (no OpenFOAM function objects).

Two measuring planes (x = const, 1201 points across one pitch), flux-weighted by axial
velocity.  Losses are FLUX-AVERAGED, not mixed-out: this is a stated limitation.

Quantities (kinematic pressure, rho = 1):
  loss      (p01 - p02) / (p01 - p1)        p01, p02: flux-averaged total pressure
                                            p1: area-averaged static pressure, inlet plane
  turning   beta1 - beta2                   flux-averaged flow angles from axial [deg]
  Cp        (p2 - p1) / (p01 - p1)          static pressure rise, area-averaged p
  DF        1 - V2/V1 + (Vt1 - Vt2)/(2 sigma V1)     Lieblein diffusion factor
  mass      out/in mass flow ratio (sanity check, ~1)
  Carter    deviation / turning estimate from Carter's rule (sanity check, NOT a benchmark)

Why I measure it this way
  * Loss is a total-pressure drop between an inlet and an outlet plane, normalised with the inlet dynamic head
    (p01 - p1): the usual cascade loss coefficient, and it is directly comparable with the NACA data.
  * The planes are half an axial chord upstream of the LE and half an axial chord downstream of the TE. That is close
    enough to the blade to be meaningful and far enough that the wake is formed. The NACA rake sat about one chord
    downstream, so I list the difference as an error source.
  * Flux-averaging (weighted with axial velocity) is simple and conserves the quantity that crosses the plane. It is
    NOT mixed-out averaging, which would give a slightly higher loss; I state that as a limitation.
  * All sampling is done in Python on the cell data instead of OpenFOAM function objects. On v1912 the function
    objects crashed with a sha1 stream error in my first tests; reading ASCII fields myself also keeps the numbers independent of the solver.
  * Carter's rule and the mass-flow ratio are cheap checks that catch a gross mistake (wrong angle, leaking
    boundary). They are not a validation.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from . import foam
from .foamio import Mesh2D, latest_time, read_field
from .case import load_meta

N_SAMPLE = 1201


class FieldData:
    """Cell-centred U and p of the latest time plus mesh centres, with periodic sampling."""

    def __init__(self, case, time: str | None = None):
        self.case = Path(case)
        self.meta = load_meta(case)
        self.mesh = Mesh2D(case)
        self.time = time or latest_time(case)
        if self.time == "0":
            raise RuntimeError(f"{case}: no solution written (only time 0 present)")
        n = self.mesh.n_cells
        self.U = read_field(case, self.time, "U", "vector", n)["internal"]
        self.p = read_field(case, self.time, "p", "scalar", n)["internal"]
        if len(self.U) != n or len(self.p) != n:
            raise RuntimeError("field size does not match the mesh")
        if not (np.all(np.isfinite(self.U)) and np.all(np.isfinite(self.p))):
            raise RuntimeError("non-finite values in U or p")
        self.dom = self.meta["domain"]
        self.pitch = self.dom["pitch"]

    def sampler(self, x_plane: float, margin: float = 0.25):
        xy = self.mesh.xy
        sel = np.abs(xy[:, 0] - x_plane) < margin
        pts, U, p = xy[sel], self.U[sel], self.p[sel]
        yb, yt, s = self.dom["y_bot"], self.dom["y_top"], self.pitch
        lo = pts[:, 1] < yb + margin
        hi = pts[:, 1] > yt - margin
        pts = np.vstack((pts, pts[hi] - [0, s], pts[lo] + [0, s]))
        U = np.vstack((U, U[hi], U[lo]))
        p = np.concatenate((p, p[hi], p[lo]))
        vals = np.column_stack((U[:, 0], U[:, 1], p))
        return LinearNDInterpolator(pts, vals), NearestNDInterpolator(pts, vals)

    def sample_plane(self, x_plane: float, n: int = N_SAMPLE) -> dict:
        lin, near = self.sampler(x_plane)
        y = np.linspace(self.dom["y_bot"], self.dom["y_top"], n)
        P = np.column_stack((np.full(n, x_plane), y))
        v = lin(P)
        bad = ~np.isfinite(v).all(axis=1)
        if bad.any():
            v[bad] = near(P[bad])
        return {"y": y, "Ux": v[:, 0], "Uy": v[:, 1], "p": v[:, 2]}


def _trapz(f, y):
    return float(np.trapezoid(f, y)) if hasattr(np, "trapezoid") else float(np.trapz(f, y))


def plane_averages(s: dict) -> dict:
    y, Ux, Uy, p = s["y"], s["Ux"], s["Uy"], s["p"]
    L = y[-1] - y[0]
    mdot = _trapz(Ux, y)
    if mdot <= 0:
        raise RuntimeError("non-positive mass flux through measuring plane (reversed flow?)")
    p0 = p + 0.5 * (Ux ** 2 + Uy ** 2)
    avg = lambda f: _trapz(f * Ux, y) / mdot
    Vx, Vt = avg(Ux), avg(Uy)
    return {
        "mdot": mdot, "p0_flux": avg(p0), "p_area": _trapz(p, y) / L, "p_flux": avg(p),
        "Vx": Vx, "Vt": Vt, "V": math.hypot(Vx, Vt),
        "beta_deg": math.degrees(math.atan2(Vt, Vx)),
        "min_Ux": float(Ux.min()),
    }


def carter_estimate(blade_meta: dict, solidity: float, beta1_flow_deg: float, a_over_c: float = 0.5) -> dict:
    """Carter's rule: delta = m * phi / sqrt(sigma), m = 0.23 (2a/c)^2 + 0.1 * beta2'/50,
    beta2' = outlet metal angle [deg]. Sanity check only (assumes a = 0.5c max camber position
    and the camber angle phi = camber_per_cl0 * cl0)."""
    phi = blade_meta["camber_deg"]
    b2m = blade_meta["outlet_metal_angle_deg"]
    m = 0.23 * (2 * a_over_c) ** 2 + 0.1 * b2m / 50.0
    delta = m * phi / math.sqrt(solidity)
    beta2 = b2m + delta
    return {"m": m, "deviation_deg": delta, "beta2_deg": beta2, "turning_deg": beta1_flow_deg - beta2}


def compute_metrics(fd: FieldData) -> dict:
    dom, meta = fd.dom, fd.meta
    s1 = fd.sample_plane(dom["x_plane_in"])
    s2 = fd.sample_plane(dom["x_plane_out"])
    a1, a2 = plane_averages(s1), plane_averages(s2)
    sigma = meta["config"]["blade"]["chord"] / fd.pitch
    den = a1["p0_flux"] - a1["p_area"]
    loss = (a1["p0_flux"] - a2["p0_flux"]) / den
    turning = a1["beta_deg"] - a2["beta_deg"]
    cp = (a2["p_area"] - a1["p_area"]) / den
    df = 1.0 - a2["V"] / a1["V"] + (a1["Vt"] - a2["Vt"]) / (2.0 * sigma * a1["V"])
    carter = carter_estimate(meta["blade"], sigma, a1["beta_deg"])
    return {
        "loss_coefficient": loss,
        "turning_deg": turning,
        "beta1_deg": a1["beta_deg"],
        "beta2_deg": a2["beta_deg"],
        "static_pressure_rise_coeff": cp,
        "diffusion_factor": df,
        "mass_flow_ratio": a2["mdot"] / a1["mdot"],
        "carter": carter,
        "min_axial_velocity_outlet_plane": a2["min_Ux"],
        "plane_inlet": a1, "plane_outlet": a2,
        "time": fd.time,
    }


# ----------------------------------------------------------------------- convergence
def parse_solver_log(text: str) -> dict:
    out = {"converged": False, "iterations": None, "final_residuals": {}}
    m = re.search(r"SIMPLE solution converged in\s+(\d+)\s+iterations", text)
    times = re.findall(r"^Time = (\d+(?:\.\d+)?)\s*$", text, re.M)
    if m:
        out["converged"] = True
        out["iterations"] = int(m.group(1))
    elif times:
        out["iterations"] = int(float(times[-1]))
    if times:
        last = text.rsplit("\nTime = ", 1)[-1]
        for fld in ("Ux", "Uy", "p", "k", "omega"):
            mm = re.search(r"Solving for %s, Initial residual = ([\d.eE+-]+)" % fld, last)
            if mm:
                out["final_residuals"][fld] = float(mm.group(1))
    out["diverged"] = bool(re.search(r"nan|FOAM FATAL|Floating point exception", text, re.I))
    return out


def residual_history(text: str) -> dict:
    """First 'Initial residual' of each field per iteration."""
    hist = {f: [] for f in ("Ux", "Uy", "p", "k", "omega")}
    for blk in text.split("\nTime = ")[1:]:
        for f in hist:
            mm = re.search(r"Solving for %s, Initial residual = ([\d.eE+-]+)" % f, blk)
            hist[f].append(float(mm.group(1)) if mm else float("nan"))
    return hist


# ----------------------------------------------------------------------- y+
def compute_yplus(case) -> dict | None:
    """y+ on the blade patch via 'simpleFoam -postProcess -func yPlus'.
    NOTE: the standalone 'postProcess -func yPlus' returns 0 on v2606 here (it does not load the
    turbulence model), the solver in -postProcess mode gives the real values."""
    case = Path(case)
    try:
        foam.run(["simpleFoam", "-postProcess", "-func", "yPlus", "-latestTime"], case, "yPlus", check=True)
        t = latest_time(case)
        fld = read_field(case, t, "yPlus", "scalar", None)
        v = fld["boundary"].get("blade")
        if v is None or len(v) == 0:
            return None
        return {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()), "n_faces": int(len(v))}
    except Exception as e:  # post-processing nicety, never fatal
        return {"error": str(e).splitlines()[0]}


def write_metrics(case, extra: dict | None = None, with_yplus: bool = True) -> dict:
    case = Path(case)
    fd = FieldData(case)
    m = compute_metrics(fd)
    conv = parse_solver_log(foam.log_text(case, "simpleFoam"))
    meta = fd.meta
    out = {
        "case": meta["name"],
        "converged": conv["converged"],
        "iterations": conv["iterations"],
        "final_residuals": conv["final_residuals"],
        "n_cells": meta["mesh"]["n_cells"],
        "h_ref": meta["mesh"]["h_ref"],
        "deviation": meta["config"].get("deviation", {}),
        **{k: v for k, v in m.items() if k not in ("plane_inlet", "plane_outlet")},
        "planes": {"inlet": m["plane_inlet"], "outlet": m["plane_outlet"]},
        "notes": "flux-averaged (not mixed-out) loss; kinematic pressure; fully turbulent k-omega SST",
    }
    if with_yplus:
        out["yplus_blade"] = compute_yplus(case)
    if extra:
        out.update(extra)
    (case / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
