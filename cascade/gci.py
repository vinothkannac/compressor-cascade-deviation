"""Grid convergence index after Celik et al. (2008), J. Fluids Eng. 130(7):078001.

Three solutions f1 (fine), f2 (medium), f3 (coarse) with representative cell sizes h1 < h2 < h3.
In 2D, h = sqrt(A / N).  Returns the observed order p, the extrapolated value and GCI_fine.
A non-monotonic change is reported as such and NO GCI is forced.

Why I wrote it like this: I wanted the standard Celik et al. procedure and I wanted it to refuse when the data do not
support it. A tool that always prints a number would have hidden what my first mesh study showed (changes that do not
shrink with refinement). The function is tested on synthetic data with a known order and a known exact answer
(scripts/check_setup.py).
"""
from __future__ import annotations

import math


def gci_three(f1: float, f2: float, f3: float, h1: float, h2: float, h3: float, fs: float = 1.25) -> dict:
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"f_fine": f1, "f_medium": f2, "f_coarse": f3, "r21": r21, "r32": r32,
           "eps21": e21, "eps32": e32, "monotonic": False, "p": None, "f_extrapolated": None,
           "gci_fine_rel": None, "gci_fine_abs": None, "gci_medium_rel": None, "flag": ""}
    if r21 <= 1 or r32 <= 1:
        out["flag"] = "cell sizes must increase from fine to coarse"
        return out
    if e21 == 0 or e32 == 0:
        out["flag"] = "no change between two meshes: order not defined"
        return out
    ratio = e32 / e21
    if ratio < 0:
        out["flag"] = "oscillatory (non-monotonic) convergence: no GCI computed"
        return out
    s = 1.0
    p = abs(math.log(ratio)) / math.log(r21)
    for _ in range(500):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError):
            out["flag"] = "order iteration failed"
            return out
        p_new = abs(math.log(ratio) + q) / math.log(r21)
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new
    out["monotonic"] = True
    if ratio < 1:
        out["flag"] = "changes grow with refinement (diverging): no order, no GCI"
        return out
    out["p"] = p
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1) if f1 != 0 else float("nan")
    ea32 = abs((f2 - f3) / f2) if f2 != 0 else float("nan")
    out["f_extrapolated"] = f_ext
    out["gci_fine_rel"] = fs * ea21 / (r21 ** p - 1)
    out["gci_fine_abs"] = out["gci_fine_rel"] * abs(f1)
    out["gci_medium_rel"] = fs * ea32 / (r32 ** p - 1)
    if p > 4.0 and not out["flag"]:
        out["flag"] = "observed order > 4: treat the GCI with caution"
    if p < 0.5 and not out["flag"]:
        out["flag"] = "observed order < 0.5: far from the asymptotic range"
    return out


def selftest() -> None:
    """Synthetic data f = f0 + C h^2: must return p = 2 and f0."""
    f0, C = 0.0213, 0.8
    h = [0.01, 0.01 * 1.35, 0.01 * 1.35 ** 2]
    f = [f0 + C * x ** 2 for x in h]
    r = gci_three(f[0], f[1], f[2], *h)
    assert r["monotonic"] and abs(r["p"] - 2.0) < 1e-6, r
    assert abs(r["f_extrapolated"] - f0) < 1e-9, r
    # unequal refinement ratios
    h = [0.01, 0.014, 0.022]
    f = [f0 + C * x ** 1.5 for x in h]
    r = gci_three(f[0], f[1], f[2], *h)
    assert abs(r["p"] - 1.5) < 1e-6 and abs(r["f_extrapolated"] - f0) < 1e-9, r
    # oscillatory
    r = gci_three(1.0, 1.1, 1.05, 0.01, 0.014, 0.02)
    assert not r["monotonic"] and r["gci_fine_rel"] is None
    # diverging (changes grow towards the fine mesh)
    r = gci_three(1.0, 1.1, 1.12, 0.01, 0.014, 0.02)
    assert r["gci_fine_rel"] is None and r["p"] is None and "diverging" in r["flag"]
