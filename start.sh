#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import numpy, pandas, scipy, sklearn, joblib' 2>/dev/null; then
    echo 'A modellfuttatáshoz szükséges csomagok telepítése…'
    dependency_file=requirements.txt
    if [[ -f requirements-lock.txt ]]; then dependency_file=requirements-lock.txt; fi
    .venv/bin/python -m pip install --no-cache-dir -r "$dependency_file"
fi
if ! .venv/bin/python -c 'from server import public_models; assert all(m["ready"] for m in public_models())' 2>/dev/null; then
    echo 'A négy modell tanítása és mentése…'
    .venv/bin/python train_models.py
fi
exec .venv/bin/python server.py
