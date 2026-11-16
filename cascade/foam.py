"""Run OpenFOAM utilities with -case and write log.<name> files.

If the OpenFOAM environment is not loaded in the calling shell, the bashrc is located
automatically (env OPENFOAM_BASHRC, or the usual install paths) and sourced for each call.

Why: I did not want the scripts to depend on me remembering to source the OpenFOAM environment first. Every utility
writes a log.<name> file in the case, and when something fails I read that log before changing code.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

BASHRC_CANDIDATES = [
    "/usr/lib/openfoam/openfoam2606/etc/bashrc",
    str(Path.home() / "OpenFOAM/OpenFOAM-v2606/etc/bashrc"),
    "/opt/openfoam2606/etc/bashrc",
    "/usr/lib/openfoam/openfoam/etc/bashrc",
]


class FoamError(RuntimeError):
    pass


def find_bashrc() -> str | None:
    env = os.environ.get("OPENFOAM_BASHRC")
    if env and Path(env).is_file():
        return env
    for c in BASHRC_CANDIDATES:
        if Path(c).is_file():
            return c
    return None


def env_loaded() -> bool:
    return bool(os.environ.get("WM_PROJECT_DIR")) and shutil.which("simpleFoam") is not None


def _wrap(cmd: list[str]) -> list[str]:
    if env_loaded():
        return cmd
    rc = find_bashrc()
    if rc is None:
        raise FoamError("OpenFOAM environment not loaded and no bashrc found. "
                        "Run: source /usr/lib/openfoam/openfoam2606/etc/bashrc  (or set OPENFOAM_BASHRC)")
    line = f"source {shlex.quote(rc)} >/dev/null 2>&1; exec " + " ".join(shlex.quote(c) for c in cmd)
    return ["bash", "-c", line]


def foam_version() -> str:
    v = os.environ.get("WM_PROJECT_VERSION")
    if v:
        return v
    rc = find_bashrc()
    if rc is None:
        return ""
    out = subprocess.run(["bash", "-c", f"source {shlex.quote(rc)} >/dev/null 2>&1; echo $WM_PROJECT_VERSION"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def run(cmd: list[str] | str, case, log_name: str | None = None, check: bool = True,
        timeout: float | None = None, add_case: bool = True) -> subprocess.CompletedProcess:
    """Run an OpenFOAM utility. '-case <case>' is appended (add_case=False to skip).
    Output goes to <case>/log.<log_name> (default: name of the utility)."""
    cmd = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    case = Path(case).resolve()
    case.mkdir(parents=True, exist_ok=True)
    name = log_name or Path(cmd[0]).name
    full = cmd + (["-case", str(case)] if add_case else [])
    log = case / f"log.{name}"
    with open(log, "w") as fh:
        fh.write("# " + " ".join(full) + "\n")
        fh.flush()
        try:
            p = subprocess.run(_wrap(full), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                               cwd=case, text=True)
        except subprocess.TimeoutExpired:
            raise FoamError(f"{name}: timeout after {timeout}s (see {log})")
    if check and p.returncode != 0:
        tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-25:])
        raise FoamError(f"{name} failed with exit code {p.returncode} (log: {log})\n{tail}")
    return p


def log_text(case, name: str) -> str:
    p = Path(case) / f"log.{name}"
    return p.read_text(errors="replace") if p.exists() else ""
