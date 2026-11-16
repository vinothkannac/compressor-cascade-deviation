"""Run one case end to end (build -> simpleFoam -> metrics) and run several in parallel.

Why: each case is a single core and takes about 15 s to 6 min, so I run independent cases side by side (one process
per case) rather than parallelising the solver. A case counts as converged only if simpleFoam stopped on the residual
criterion, not because it reached the iteration cap; that flag is stored in metrics.json and checked by every script.
"""
from __future__ import annotations

import json
import time
import multiprocessing as mp
from pathlib import Path

from . import foam
from .case import build_case, deep_update, load_meta
from .postproc import write_metrics


def _strip(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def run_one(cfg: dict, case_dir, mesh_only: bool = False, legacy_cyclic: bool = False,
            yplus: bool = True, reuse: bool = False, name: str | None = None, quiet: bool = False) -> dict:
    """Return the metrics dict (or the checkMesh dict if mesh_only).
    reuse=True: if the case already holds metrics for an identical config, return them."""
    case = Path(case_dir)
    if reuse and (case / "metrics.json").exists() and (case / "case_meta.json").exists():
        try:
            if load_meta(case)["config"] == json.loads(json.dumps(_strip(cfg))):
                m = json.loads((case / "metrics.json").read_text())
                if not quiet:
                    print(f"[{case.name}] reusing finished case")
                return m
        except Exception:
            pass
    t0 = time.time()
    meta = build_case(cfg, case, legacy_cyclic=legacy_cyclic, name=name)
    if not quiet:
        mm = meta["mesh"]
        print(f"[{case.name}] mesh OK: {mm['n_cells']} cells, max non-orth {mm.get('max_non_ortho')}, "
              f"max skew {mm.get('max_skewness')}  ({time.time() - t0:.0f} s)")
    if mesh_only:
        return meta["mesh"]
    t1 = time.time()
    foam.run(["simpleFoam"], case, "simpleFoam", check=True)
    m = write_metrics(case, extra={"wall_time_solver_s": round(time.time() - t1, 1)}, with_yplus=yplus)
    if not quiet:
        print(f"[{case.name}] simpleFoam done: converged={m['converged']} in {m['iterations']} it "
              f"({m['wall_time_solver_s']} s), loss={m['loss_coefficient']:.5f}, turning={m['turning_deg']:.3f} deg")
    return m


def _worker(args):
    cfg, case, kw = args
    try:
        return {"ok": True, "metrics": run_one(cfg, case, **kw)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def run_many(jobs: list[tuple[str, dict, str]], n_jobs: int = 1, **kw) -> dict:
    """jobs: [(label, cfg, case_dir)]. Returns {label: {'ok': bool, 'metrics' or 'error'}}.
    One failing case does not stop the others. n_jobs > 1 runs cases side by side in separate processes."""
    args = [(cfg, case, {**kw, "name": label}) for label, cfg, case in jobs]
    if n_jobs <= 1:
        outputs = [_worker(a) for a in args]
    else:
        # "spawn" gives every worker a fresh interpreter, which is the safe choice with Gmsh
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            outputs = pool.map(_worker, args)
    results = {}
    for (label, _, _), out in zip(jobs, outputs):
        results[label] = out
        if not out["ok"]:
            print(f"[{label}] FAILED: {out['error']}")
    return results


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """--set section.key=value (value parsed as JSON, else string)."""
    out = deep_update(cfg, {})
    for s in sets or []:
        if "=" not in s or "." not in s.split("=")[0]:
            raise ValueError(f"--set expects section.key=value, got '{s}'")
        k, v = s.split("=", 1)
        sec, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if sec not in out:
            raise ValueError(f"--set: unknown section '{sec}'")
        out[sec][key] = v
    return out
