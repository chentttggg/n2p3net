#!/usr/bin/env sh
# Resolve the declared platform branch; never manually replace accelerator wheels.
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv 0.12.9, then rerun this script." >&2
    exit 1
fi
uv lock --check
uv sync --locked --python "${N2P3_PYTHON:-3.12}" --extra signals --extra baselines --extra dev
uv pip check
.venv/bin/python -c 'import torch; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()})'
