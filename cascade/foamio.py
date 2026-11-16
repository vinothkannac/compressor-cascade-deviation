"""Minimal ASCII reader for OpenFOAM polyMesh and fields (no OpenFOAM tools needed).

Supports: polyMesh/{points,faces,owner,neighbour,boundary}, internalField and
boundaryField values of volScalarField / volVectorField in ASCII format
(uniform or nonuniform List).  Binary and compressed files are rejected with a clear error.

Why: I wanted the post-processing to run without ParaView or OpenFOAM sampling tools, so results can be recomputed
from the case folder with Python alone. That is why writeFormat stays ascii in controlDict.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _read_text(path: Path) -> str:
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    if not path.exists():
        raise FileNotFoundError(path)
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    txt = raw.decode("utf-8", errors="replace")
    m = re.search(r"format\s+(\w+)\s*;", txt[:2000])
    if m and m.group(1) != "ascii":
        raise ValueError(f"{path}: format '{m.group(1)}' not supported, keep writeFormat ascii in controlDict")
    return txt


def _strip_header(txt: str) -> str:
    """Remove comments and the FoamFile header."""
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    txt = re.sub(r"FoamFile\s*\{.*?\}", "", txt, count=1, flags=re.S)
    return txt


def _list_body(txt: str):
    """Return (count, body) of the first 'N ( ... )' list in txt."""
    m = re.search(r"(\d+)\s*\(", txt)
    if not m:
        raise ValueError("no list found")
    n = int(m.group(1))
    start = m.end()
    end = txt.rfind(")")
    return n, txt[start:end]


def read_points(case) -> np.ndarray:
    txt = _strip_header(_read_text(Path(case) / "constant/polyMesh/points"))
    n, body = _list_body(txt)
    v = np.array(re.findall(_FLOAT, body.replace("(", " ").replace(")", " ")), dtype=float)
    pts = v.reshape(-1, 3)
    if len(pts) != n:
        raise ValueError(f"points: expected {n}, got {len(pts)}")
    return pts


def read_label_list(path) -> np.ndarray:
    txt = _strip_header(_read_text(path))
    # owner/neighbour have a note line with nPoints/nCells in the header comment: removed above
    n, body = _list_body(txt)
    a = np.array(body.split(), dtype=np.int64)
    if len(a) != n:
        raise ValueError(f"{path}: expected {n} labels, got {len(a)}")
    return a


def read_faces(case) -> list[np.ndarray]:
    txt = _strip_header(_read_text(Path(case) / "constant/polyMesh/faces"))
    n, body = _list_body(txt)
    faces = [np.array(m.split(), dtype=np.int64) for m in re.findall(r"\d+\s*\(([^)]*)\)", body)]
    if len(faces) != n:
        raise ValueError(f"faces: expected {n}, got {len(faces)}")
    return faces


def read_boundary(case) -> dict:
    """Return {patch: {'type':..., 'nFaces':int, 'startFace':int, ...}} in file order."""
    txt = _strip_header(_read_text(Path(case) / "constant/polyMesh/boundary"))
    body = txt[txt.index("(") + 1: txt.rindex(")")]
    out = {}
    for m in re.finditer(r"(\S+)\s*\{([^{}]*)\}", body):
        d = {}
        for k, v in re.findall(r"(\w+)\s+([^;]+);", m.group(2)):
            d[k] = v.strip()
        d["nFaces"] = int(d.get("nFaces", 0))
        d["startFace"] = int(d.get("startFace", 0))
        out[m.group(1)] = d
    return out


class Mesh2D:
    """Cell centres (x, y) and bookkeeping of a one-cell-thick mesh."""

    def __init__(self, case):
        self.case = Path(case)
        self.points = read_points(case)
        self.faces = read_faces(case)
        self.owner = read_label_list(self.case / "constant/polyMesh/owner")
        self.neighbour = read_label_list(self.case / "constant/polyMesh/neighbour")
        self.boundary = read_boundary(case)
        self.n_cells = int(self.owner.max()) + 1
        self._compute_centres()

    def _compute_centres(self):
        """2D cell centres from the polygon centroid of each cell's face on the 'front'
        patch (one-cell-thick extrusion: exactly one such face per cell)."""
        if "front" not in self.boundary:
            raise ValueError("Mesh2D expects an extruded 2D mesh with a 'front' patch")
        P = self.points
        ids = self.patch_faces("front")
        cells = self.owner[ids]
        if len(ids) != self.n_cells or len(np.unique(cells)) != self.n_cells:
            raise ValueError("Mesh2D: the 'front' patch does not hold exactly one face per cell")
        cen = np.zeros((self.n_cells, 2))
        area = np.zeros(self.n_cells)
        for fid, c in zip(ids, cells):
            q = P[self.faces[fid]][:, :2]
            x, y = q[:, 0], q[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            a = 0.5 * cr.sum()
            if abs(a) < 1e-300:
                cen[c] = q.mean(axis=0)
            else:
                cen[c] = (np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * a))
            area[c] = abs(a)
        self.xy = cen
        self.cell_area = area

    def patch_faces(self, name):
        b = self.boundary[name]
        s, n = b["startFace"], b["nFaces"]
        return np.arange(s, s + n)


def latest_time(case) -> str:
    """Name of the latest numeric time directory (excluding 0 unless it is the only one)."""
    case = Path(case)
    times = []
    for d in case.iterdir():
        if d.is_dir():
            try:
                times.append((float(d.name), d.name))
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"no time directories in {case}")
    times.sort()
    return times[-1][1]


def _parse_field_value(spec: str, kind: str, n_expected: int | None):
    """Parse 'uniform X' or 'nonuniform List<T> N ( ... )' -> array."""
    spec = spec.strip()
    ncomp = 3 if kind == "vector" else 1
    if spec.startswith("uniform"):
        vals = np.array(re.findall(_FLOAT, spec[len("uniform"):]), dtype=float)
        if len(vals) != ncomp:
            raise ValueError(f"cannot parse uniform {kind}: {spec[:60]}")
        n = n_expected if n_expected is not None else 1
        return np.tile(vals, (n, 1)) if ncomp == 3 else np.full(n, vals[0])
    if spec.startswith("nonuniform"):
        if re.match(r"nonuniform\s+0\s*\(\s*\)", spec) or re.match(r"nonuniform\s+List<\w+>\s+0\s*\(\s*\)", spec):
            return np.zeros((0, 3)) if ncomp == 3 else np.zeros(0)
        m = re.match(r"nonuniform\s+(?:List<\w+>\s+)?(\d+)\s*\(", spec)
        if not m:
            raise ValueError(f"cannot parse nonuniform list: {spec[:80]}")
        n = int(m.group(1))
        body = spec[m.end(): spec.rindex(")")]
        v = np.array(re.findall(_FLOAT, body), dtype=float)
        if len(v) != n * ncomp:
            raise ValueError(f"field list: expected {n * ncomp} numbers, got {len(v)}")
        return v.reshape(-1, 3) if ncomp == 3 else v
    raise ValueError(f"unsupported field entry: {spec[:60]}")


def _grab_entry(txt: str, start: int) -> tuple[str, int]:
    """Entry text from 'start' up to the terminating ';' at bracket depth 0."""
    depth = 0
    i = start
    while i < len(txt):
        ch = txt[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";" and depth == 0:
            return txt[start:i], i
        i += 1
    raise ValueError("unterminated entry")


def read_field(case, time: str, name: str, kind: str, n_cells: int | None = None) -> dict:
    """Return {'internal': array, 'boundary': {patch: array or None}}."""
    txt = _strip_header(_read_text(Path(case) / time / name))
    m = re.search(r"internalField\s+", txt)
    if not m:
        raise ValueError(f"{name}: no internalField")
    spec, end = _grab_entry(txt, m.end())
    internal = _parse_field_value(spec, kind, n_cells)
    bnd = {}
    mb = re.search(r"boundaryField\s*\{", txt)
    if mb:
        i = mb.end()
        depth = 1
        body_start = i
        while i < len(txt) and depth:
            depth += {"{": 1, "}": -1}.get(txt[i], 0)
            i += 1
        body = txt[body_start: i - 1]
        # iterate over patch blocks (top-level)
        j = 0
        while j < len(body):
            mm = re.compile(r'\s*("[^"]+"|\S+)\s*\{').match(body, j)
            if not mm:
                j += 1
                continue
            pname = mm.group(1).strip('"')
            k = mm.end()
            d = 1
            while k < len(body) and d:
                d += {"{": 1, "}": -1}.get(body[k], 0)
                k += 1
            block = body[mm.end(): k - 1]
            mv = re.search(r"\bvalue\s+", block)
            val = None
            if mv:
                try:
                    spec_b, _ = _grab_entry(block, mv.end())
                    val = _parse_field_value(spec_b, kind, None)
                except ValueError:
                    val = None
            bnd[pname] = val
            j = k
    return {"internal": internal, "boundary": bnd}
