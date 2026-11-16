"""Gmsh mesh of one periodic blade passage, extruded by one cell in z (2D for OpenFOAM).

Physical groups written to the .msh (v2.2, ASCII):
  inlet, outlet, blade, bottom_raw, top_raw, front, back, fluid (volume)

bottom_raw / top_raw are plain patches here; cascade/case.py turns them into
matched cyclic patches with createPatch (gmshToFoam does not order periodic faces).

Why I meshed it this way
  * One blade passage with periodic top/bottom instead of a single blade in a big box: the real thing is a row of
    equal blades, and a periodic passage gets the blade-to-blade interaction for the price of one blade.
  * Gmsh (open source, scriptable from Python) so every geometry variant is re-meshed by the same recipe.
  * Hybrid mesh: quads in a boundary layer around the blade, triangles elsewhere. The boundary layer is there to
    resolve the wall for the low-Re wall treatment (first layer 1e-4 c gives y+ of about 0.5 on average, max 1.7;
    I checked it, it is below my target of 2). Triangles keep the cell count low far from the blade.
  * The mesh is refined in a band along the expected wake line (25 deg from axial) because the loss I measure is
    made of the wake that has to reach the outlet plane; a coarse wake would smear it and change the loss.
  * Extruded by ONE cell in z with 'empty' front/back: OpenFOAM's way of running 2D.
  * Only h_ref is changed in the mesh study; every other size is a multiple of h_ref. The wall-normal layers are
    NOT scaled, which I state as a limitation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import BladeParams, blade_outline, nominal_frame


@dataclass
class MeshParams:
    h_ref: float = 0.008              # reference cell size (the one parameter the mesh study changes)
    wall_first_layer: float = 1.0e-4  # first layer height on the blade [chord]
    bl_ratio: float = 1.2             # boundary-layer growth ratio
    bl_thickness: float = 0.02        # total boundary-layer thickness [chord]: about 20 layers at ratio 1.2, roughly the turbulent BL of this blade
    h_far_factor: float = 8.0         # far-field size = factor * h_ref
    h_blade_factor: float = 0.75      # surface tangential size = factor * h_ref
    h_edge_factor: float = 0.12       # LE / TE size = factor * h_ref: the nose radius and the 0.003 c TE face need small cells
    h_wake_factor: float = 0.9        # wake size = factor * h_ref
    blade_dist_max: float = 0.5       # distance over which size grows from blade to far field
    wake_angle_deg: float = 25.0      # wake direction from axial: I set it to about the outlet flow angle (beta2 is 23.6 deg)
    wake_halfwidth: float = 0.03     # full refinement inside this distance from the wake line
    wake_blend: float = 0.20          # distance over which wake refinement fades out
    z_thickness: float = 0.05         # extrusion thickness (single cell): arbitrary for a 2D case, only has to be well-shaped
    algorithm: int = 6                # 2D algorithm (6 = frontal-Delaunay)
    n_fan: int = 3                    # boundary-layer fan elements at the two TE corners
    bl_on_te_face: bool = True        # also grow the boundary layer on the blunt TE face

    @classmethod
    def from_config(cls, cfg: dict) -> "MeshParams":
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in cfg.get("mesh", {}).items() if k in keys})


@dataclass
class CascadeDomain:
    pitch: float
    x_in: float
    x_out: float
    y_bot: float
    y_top: float
    x_plane_in: float
    x_plane_out: float
    axial_chord: float
    le: tuple
    te: tuple

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def make_domain(blade: BladeParams, cfg: dict) -> CascadeDomain:
    """Domain from the NOMINAL blade, so every deviation variant uses the same box."""
    c = cfg.get("cascade", {})
    nom = nominal_frame(blade)
    ax = nom["axial_chord"]
    pitch = float(c.get("pitch", 1.0)) * blade.chord
    n_in = float(c.get("inlet_axial_chords", 1.0))
    n_out = float(c.get("outlet_axial_chords", 2.0))
    n_pl = float(c.get("plane_axial_chords", 0.5))
    le, te = nom["le"], nom["te"]
    ymid = float(nom["mid"][1])
    return CascadeDomain(
        pitch=pitch,
        x_in=float(le[0] - n_in * ax),
        x_out=float(te[0] + n_out * ax),
        y_bot=ymid - 0.5 * pitch,
        y_top=ymid + 0.5 * pitch,
        x_plane_in=float(le[0] - n_pl * ax),
        x_plane_out=float(te[0] + n_pl * ax),
        axial_chord=ax,
        le=(float(le[0]), float(le[1])),
        te=(float(te[0]), float(te[1])),
    )


def _wake_distance_expr(dom: CascadeDomain, te_xy, angle_deg: float) -> list[str]:
    """MathEval expressions (one per periodic image) for the distance to the wake line.
    Upstream of the TE point the distance is to the TE point itself."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    x0, y0 = te_xy
    exprs = []
    for k in (-1, 0, 1):
        dx = f"(x-({x0:.8f}))"
        dy = f"(y-({y0 + k * dom.pitch:.8f}))"
        along = f"({dx}*{ca:.10f}+{dy}*{sa:.10f})"
        perp = f"(-{dx}*{sa:.10f}+{dy}*{ca:.10f})"
        # neg = min(along,0) = 0.5*(along - |along|)
        neg = f"(0.5*({along}-Fabs({along})))"
        exprs.append(f"Sqrt({perp}*{perp}+{neg}*{neg})")
    return exprs


def _build_once(blade: BladeParams, mp: MeshParams, cfg: dict, msh_path, verbose: bool = False) -> dict:
    """Generate the mesh and write it to msh_path (Gmsh format 2.2). Returns meta info."""
    import gmsh

    dom = make_domain(blade, cfg)
    out = blade_outline(blade)
    up, lo = out["upper"], out["lower"]

    msh_path = Path(msh_path)
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.clear()
    gmsh.option.setNumber("General.NumThreads", 1)         # reproducibility, fewer surprises
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
    gmsh.model.add("cascade")
    g = gmsh.model.geo

    h_far = mp.h_far_factor * mp.h_ref
    h_blade = mp.h_blade_factor * mp.h_ref
    h_edge = mp.h_edge_factor * mp.h_ref
    h_wake = mp.h_wake_factor * mp.h_ref

    # ---- outer boundary --------------------------------------------------------------
    p1 = g.addPoint(dom.x_in, dom.y_bot, 0, h_far)
    p2 = g.addPoint(dom.x_out, dom.y_bot, 0, h_far)
    p3 = g.addPoint(dom.x_out, dom.y_top, 0, h_far)
    p4 = g.addPoint(dom.x_in, dom.y_top, 0, h_far)
    l_bot = g.addLine(p1, p2)
    l_out = g.addLine(p2, p3)
    l_top = g.addLine(p4, p3)         # same direction as l_bot (periodic master/slave)
    l_in = g.addLine(p1, p4)

    # ---- blade -----------------------------------------------------------------------
    pts_up = [g.addPoint(float(x), float(y), 0, h_blade) for x, y in up]
    pts_lo = [g.addPoint(float(x), float(y), 0, h_blade) for x, y in lo[1:]]   # lo[0] == up[0] == LE
    le_pt = pts_up[0]
    te_up, te_lo = pts_up[-1], pts_lo[-1]
    c_up = g.addSpline(pts_up)                    # LE -> TE (upper)
    c_te = g.addLine(te_up, te_lo)                # TE face
    c_lo = g.addSpline([le_pt] + pts_lo)          # LE -> TE (lower)
    blade_curves = [c_up, c_te, c_lo]

    outer = g.addCurveLoop([l_bot, l_out, -l_top, -l_in])
    # blade loop, orientation opposite to outer is handled by gmsh (hole): LE->TE upper, TE face, TE->LE lower
    inner = g.addCurveLoop([c_up, c_te, -c_lo])
    surf = g.addPlaneSurface([outer, inner])
    g.synchronize()

    # periodic: top is the slave of bottom, translation by +pitch in y
    gmsh.model.mesh.setPeriodic(1, [l_top], [l_bot],
                                [1, 0, 0, 0, 0, 1, 0, dom.pitch, 0, 0, 1, 0, 0, 0, 0, 1])

    # ---- size fields -----------------------------------------------------------------
    F = gmsh.model.mesh.field
    fid = {}

    def add(kind, **opts):
        i = F.add(kind)
        for k, v in opts.items():
            if isinstance(v, (list, tuple)):
                F.setNumbers(i, k, list(v))
            elif isinstance(v, str):
                F.setString(i, k, v)
            else:
                F.setNumber(i, k, v)
        return i

    d_blade = add("Distance", CurvesList=blade_curves, Sampling=400)
    t_blade = add("Threshold", InField=d_blade, SizeMin=h_blade, SizeMax=h_far,
                  DistMin=0.01, DistMax=mp.blade_dist_max)
    d_edge = add("Distance", PointsList=[le_pt, te_up, te_lo], Sampling=1)
    t_edge = add("Threshold", InField=d_edge, SizeMin=h_edge, SizeMax=h_far,
                 DistMin=0.004, DistMax=0.12)

    te_mid = (0.5 * (up[-1][0] + lo[-1][0]), 0.5 * (up[-1][1] + lo[-1][1]))
    wake_fields = []
    for expr in _wake_distance_expr(dom, te_mid, mp.wake_angle_deg):
        m = add("MathEval", F=expr)
        wake_fields.append(add("Threshold", InField=m, SizeMin=h_wake, SizeMax=h_far,
                               DistMin=mp.wake_halfwidth, DistMax=mp.wake_halfwidth + mp.wake_blend))
    all_fields = [t_blade, t_edge] + wake_fields
    bg = add("Min", FieldsList=all_fields)

    # boundary layer (quads) on the blade
    bl = F.add("BoundaryLayer")
    F.setNumbers(bl, "CurvesList", blade_curves if mp.bl_on_te_face else [c_up, c_lo])
    F.setNumber(bl, "hwall_n", mp.wall_first_layer)
    F.setNumber(bl, "ratio", mp.bl_ratio)
    F.setNumber(bl, "thickness", mp.bl_thickness)
    F.setNumber(bl, "Quads", 1)
    F.setNumber(bl, "IntersectMetrics", 0)
    if mp.bl_on_te_face and mp.n_fan > 0:
        F.setNumbers(bl, "FanPointsList", [te_up, te_lo])
        F.setNumbers(bl, "FanPointsSizesList", [mp.n_fan, mp.n_fan])
    F.setAsBoundaryLayer(bl)
    F.setAsBackgroundMesh(bg)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", mp.algorithm)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # ---- one-cell extrusion (prisms from triangles, hexes from quads) ----------------
    ext = g.extrude([(2, surf)], 0, 0, mp.z_thickness, [1], recombine=True)
    g.synchronize()
    vol = [t for d, t in ext if d == 3][0]

    # classify boundary surfaces of the volume by bounding box
    eps = 1e-7
    groups = {k: [] for k in ("inlet", "outlet", "blade", "bottom_raw", "top_raw", "front", "back")}
    for d, t in gmsh.model.getBoundary([(3, vol)], oriented=False, combined=False):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, t)
        if abs(z1 - z0) < eps:
            groups["front" if z0 < 0.5 * mp.z_thickness else "back"].append(t)
        elif abs(x1 - x0) < eps and abs(x0 - dom.x_in) < 1e-6:
            groups["inlet"].append(t)
        elif abs(x1 - x0) < eps and abs(x0 - dom.x_out) < 1e-6:
            groups["outlet"].append(t)
        elif abs(y1 - y0) < eps and abs(y0 - dom.y_bot) < 1e-6:
            groups["bottom_raw"].append(t)
        elif abs(y1 - y0) < eps and abs(y0 - dom.y_top) < 1e-6:
            groups["top_raw"].append(t)
        else:
            groups["blade"].append(t)
    n_blade_expected = 3
    for name, tags in groups.items():
        if not tags:
            raise RuntimeError(f"mesh.py: boundary group '{name}' is empty (surface classification failed)")
    if len(groups["blade"]) != n_blade_expected:
        raise RuntimeError(f"mesh.py: expected {n_blade_expected} blade surfaces, found {len(groups['blade'])}")
    for i, (name, tags) in enumerate(groups.items(), start=1):
        gmsh.model.addPhysicalGroup(2, tags, i, name)
    gmsh.model.addPhysicalGroup(3, [vol], 100, "fluid")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))

    n_tri = n_quad = 0
    for et in gmsh.model.mesh.getElementTypes(2):
        n = len(gmsh.model.mesh.getElementsByType(et)[0])
        if et == 2:
            n_tri += n
        elif et == 3:
            n_quad += n
    n_prism = n_hex = 0
    for et in gmsh.model.mesh.getElementTypes(3):
        n = len(gmsh.model.mesh.getElementsByType(et)[0])
        if et == 6:
            n_prism += n
        elif et == 5:
            n_hex += n
    nodes = gmsh.model.mesh.getNodes()[0]
    meta = {
        "n_cells_expected": n_prism + n_hex,
        "n_prism": n_prism, "n_hex": n_hex,
        "n_nodes_3d": int(len(nodes)),
        "msh": str(msh_path),
        "h_ref": mp.h_ref,
        "wall_first_layer": mp.wall_first_layer,
        "domain": dom.to_dict(),
    }
    gmsh.clear()
    return meta


# I did not want a study that dies because Gmsh fails on one of seven variants, so a failed mesh is retried with a
# slightly thinner boundary layer first, and a different 2D algorithm only as a last resort. Every attempt
# number is stored in the case meta data, and the README says if a fallback was needed.
# Gmsh occasionally fails with "Edge not recovered" (self-intersecting boundary-layer front) for
# some combinations of sizes.  The fallbacks below nudge only the boundary-layer thickness
# and fan size, then the 2D algorithm; the attempt used is recorded in the meta data.
_FALLBACKS = [
    {},
    {"bl_thickness_factor": 0.9},
    {"bl_thickness_factor": 0.8, "n_fan": 0, "algorithm": 5},
]


def build_mesh(blade: BladeParams, mp: MeshParams, cfg: dict, msh_path, verbose: bool = False) -> dict:
    """Generate the mesh (with fallbacks, see above). Raises RuntimeError if all attempts fail."""
    from dataclasses import replace

    errors = []
    for i, fb in enumerate(_FALLBACKS):
        kw = dict(fb)
        f = kw.pop("bl_thickness_factor", 1.0)
        mp_i = replace(mp, bl_thickness=mp.bl_thickness * f, **kw)
        try:
            meta = _build_once(blade, mp_i, cfg, msh_path, verbose=verbose)
        except Exception as e:  # gmsh raises plain Exception
            errors.append(f"attempt {i} {fb}: {e}")
            try:
                import gmsh
                if gmsh.isInitialized():
                    gmsh.clear()
            except Exception:
                pass
            continue
        meta["mesh_attempt"] = i
        meta["mesh_fallback"] = fb
        if i > 0:
            print(f"[mesh] WARNING: Gmsh needed fallback #{i} {fb} (previous: {errors[-1]})")
        return meta
    raise RuntimeError("Gmsh failed for all fallback settings:\n  " + "\n  ".join(errors))
