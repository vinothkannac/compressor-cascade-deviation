# Source this file:  source env.sh
# Loads OpenFOAM ESI v2606 (if not loaded) and activates the project virtual environment.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$WM_PROJECT_DIR" ]; then
    for rc in "${OPENFOAM_BASHRC:-}" /usr/lib/openfoam/openfoam2606/etc/bashrc "$HOME/OpenFOAM/OpenFOAM-v2606/etc/bashrc"; do
        if [ -n "$rc" ] && [ -f "$rc" ]; then source "$rc"; break; fi
    done
fi
[ -z "$WM_PROJECT_DIR" ] && echo "env.sh: OpenFOAM not found (set OPENFOAM_BASHRC to its etc/bashrc)"
if [ -f "$_here/.venv/bin/activate" ]; then source "$_here/.venv/bin/activate"; else
    echo "env.sh: no .venv yet. Create it:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; fi
unset _here rc
