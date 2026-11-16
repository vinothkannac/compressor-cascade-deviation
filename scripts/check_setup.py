#!/usr/bin/env python3
"""Environment check + test mesh + checkMesh + simpleFoam -dry-run + a short real solver run.

    python3 scripts/check_setup.py [--keep]

Ends with 'ALL CHECKS PASSED' only if every check passed.

Why: the whole chain (Python packages, OpenFOAM v2606, Gmsh, mesh, solver, post-processing) has to work before any
study number means something. This test runs the complete chain on a small mesh for 20 iterations, so a broken
installation or a version change shows up in seconds and not after an hour of runs.
"""
import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import gmsh, matplotlib, numpy, scipy  # noqa: F401,E401
except ImportError as _e:
    print(f"  [FAIL] python packages\n         {_e}")
    print("\n1 CHECK(S) FAILED: python packages")
    sys.exit(1)

results = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                msg = fn(*a, **k)
                results.append((name, True, msg or ""))
                print(f"  [ OK ] {name}  {msg or ''}  ({time.time() - t:.1f} s)")
                return True
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  [FAIL] {name}\n         {str(e).strip().splitlines()[0] if str(e).strip() else repr(e)}")
                tail = "\n".join(str(e).strip().splitlines()[1:12])
                if tail:
                    print("         " + tail.replace("\n", "\n         "))
                return False
        return run
    return deco


@check("python >= 3.9")
def c_python():
    if sys.version_info < (3, 9):
        raise RuntimeError(f"found {sys.version}")
    return sys.version.split()[0]


@check("python packages")
def c_packages():
    out = []
    for m in ("numpy", "scipy", "matplotlib", "gmsh"):
        try:
            mod = importlib.import_module(m)
        except ImportError as e:
            raise RuntimeError(f"{m} missing ({e}). pip install -r requirements.txt "
                               "(Linux: sudo apt install libglu1-mesa for gmsh)")
        out.append(f"{m} {getattr(mod, '__version__', '?')}")
    return ", ".join(out)


@check("OpenFOAM environment")
def c_foam():
    from cascade import foam
    v = foam.foam_version()
    if not v:
        raise RuntimeError("OpenFOAM not found. source /usr/lib/openfoam/openfoam2606/etc/bashrc (or set OPENFOAM_BASHRC)")
    if "v2606" not in v and "2606" not in v:
        raise RuntimeError(f"found OpenFOAM version '{v}', the project targets ESI v2606 (openfoam.com)")
    for app in ("gmshToFoam", "createPatch", "checkMesh", "simpleFoam"):
        p = foam.run([app, "-help"], ROOT / "runs/_check_help", f"help_{app}", check=False, add_case=False)
        if p.returncode != 0:
            raise RuntimeError(f"{app} is not usable (exit {p.returncode})")
    shutil.rmtree(ROOT / "runs/_check_help", ignore_errors=True)
    return f"ESI {v}, env {'loaded' if foam.env_loaded() else 'auto-sourced from ' + str(foam.find_bashrc())}"


@check("project files")
def c_files():
    need = ["configs/base.json", "case_template/system/controlDict", "case_template/system/fvSchemes",
            "case_template/system/fvSolution", "case_template/system/createPatchDict",
            "case_template/constant/transportProperties", "case_template/constant/turbulenceProperties",
            "data/naca65_010_thickness.csv", "data/benchmark_naca1368_template.csv"]
    for n in ("U", "p", "k", "omega", "nut"):
        need.append(f"case_template/0.orig/{n}")
    miss = [n for n in need if not (ROOT / n).exists()]
    if miss:
        raise RuntimeError("missing: " + ", ".join(miss))
    txt = (ROOT / "case_template/system/controlDict").read_text()
    if "writeFormat     ascii;" not in txt:
        raise RuntimeError("case_template/system/controlDict must keep 'writeFormat ascii;'")
    return "ok"


@check("data files")
def c_data():
    from cascade.geometry import load_thickness_table
    x, y = load_thickness_table(ROOT / "data/naca65_010_thickness.csv")
    nb = [l for l in (ROOT / "data/benchmark_naca1368_template.csv").read_text().splitlines()[1:] if l.strip()]
    return (f"thickness table {len(x)} rows, max {2 * y.max():.4f} c (verified vs PDAS recomputation and NACA RM L51G31 Table I); "
            f"benchmark file has {len(nb)} data row(s) (read off the report plot, see its source column)")


@check("blade geometry")
def c_geometry():
    from cascade.case import load_config
    from cascade.geometry import BladeParams, blade_outline, max_thickness, polygon_self_intersects
    cfg = load_config(ROOT / "configs/base.json")
    for dev in ({}, {"le_scale": 0.8}, {"le_scale": 1.2}, {"thickness_scale": 0.95}, {"thickness_scale": 1.05},
                {"stagger_delta_deg": -1.0}, {"stagger_delta_deg": 1.0}):
        c = json.loads(json.dumps(cfg))
        c["deviation"].update(dev)
        p = BladeParams.from_config(c)
        poly = blade_outline(p)["polygon"]
        if polygon_self_intersects(poly):
            raise RuntimeError(f"blade outline self-intersects for deviation {dev}")
    p = BladeParams.from_config(cfg)
    t = max_thickness(p)
    if not 0.095 < t < 0.107:
        raise RuntimeError(f"max thickness {t:.4f} c is not about 0.10 c")
    return f"max thickness {t:.4f} c, camber {p.camber_deg:.2f} deg, metal angles {p.inlet_metal_angle_deg:.2f}/{p.outlet_metal_angle_deg:.2f} deg"


@check("GCI self-test")
def c_gci():
    from cascade.gci import selftest
    selftest()
    return "recovers order and extrapolated value on synthetic data"


@check("test case (mesh, checkMesh, dry-run, 20 iterations, post-processing)")
def c_case(keep=False):
    from cascade import foam
    from cascade.case import load_config
    from cascade.postproc import FieldData, compute_metrics
    from cascade.runner import apply_overrides
    from cascade.case import build_case
    cfg = apply_overrides(load_config(ROOT / "configs/base.json"),
                          ["mesh.h_ref=0.0113", "solver.max_iter=20", "solver.write_interval=20"])
    case = ROOT / "runs/_check"
    meta = build_case(cfg, case, name="_check")
    mm = meta["mesh"]
    foam.run(["simpleFoam", "-dry-run"], case, "dryRun")
    foam.run(["simpleFoam"], case, "simpleFoam")
    fd = FieldData(case)
    m = compute_metrics(fd)
    for k in ("loss_coefficient", "turning_deg", "mass_flow_ratio"):
        if not (m[k] == m[k]) or abs(m[k]) > 1e3:
            raise RuntimeError(f"post-processing produced a bad {k}: {m[k]}")
    if abs(m["beta1_deg"] - cfg["flow"]["beta1_deg"]) > 1.0:
        raise RuntimeError(f"inlet flow angle {m['beta1_deg']:.2f} deg differs from the set {cfg['flow']['beta1_deg']} deg")
    if not keep:
        shutil.rmtree(case, ignore_errors=True)
    return (f"{mm['n_cells']} cells, non-ortho max {mm.get('max_non_ortho'):.1f}, "
            f"mesh attempt {mm.get('mesh_attempt')}; 20-iteration solution read back (beta1 {m['beta1_deg']:.2f} deg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep runs/_check")
    a = ap.parse_args()
    print("Checking the setup in", ROOT)
    c_python()
    ok_pkg = c_packages()
    if ok_pkg:
        c_foam()
        c_files()
        c_data()
        c_gci()
        if c_geometry() and all(r[1] for r in results):
            c_case(a.keep)
    bad = [r for r in results if not r[1]]
    print()
    if bad:
        print(f"{len(bad)} CHECK(S) FAILED:", ", ".join(r[0] for r in bad))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
