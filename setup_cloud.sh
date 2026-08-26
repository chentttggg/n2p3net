#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
if [[ -z "${BASE_PYTHON:-}" ]]; then
    if [[ -x "$VENV/bin/python" ]]; then
        BASE_PYTHON="$VENV/bin/python"
    else
        BASE_PYTHON="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"
    fi
fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

die() {
    echo "[setup:error] $*" >&2
    exit 2
}

command -v "$BASE_PYTHON" >/dev/null 2>&1 || die "找不到基础 Python: $BASE_PYTHON"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[setup] creating virtualenv: $VENV"
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV" || die "无法创建 virtualenv；请确认 python3-venv 已安装"
fi

PYTHON_BIN="$VENV/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[setup] installing CUDA Torch from $TORCH_INDEX_URL"
    "$PYTHON_BIN" -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
else
    echo "[setup] keeping existing CUDA-enabled Torch"
fi

"$PYTHON_BIN" -m pip install --upgrade -r "$ROOT/requirements-minimal.txt"

PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import sys
import torch

from models.n2p3net import N2P3Net
from train.recipe import NEURAL_RIDE_V11_STRICT_PAST_RESEARCH

kwargs = NEURAL_RIDE_V11_STRICT_PAST_RESEARCH.model_kwargs(
    n_channels=3,
    channel_names=("Fz", "Cz", "Pz"),
    tmin_ms=-200.0,
    tmax_ms=1200.0,
    sfreq=256.0,
    n_time=358,
    baseline_mode="trial",
    tau0_ms=(220.0, 300.0, 460.0),
    tau0_bounds=((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)),
    sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 150.0)),
)
print(f"[setup] python={sys.executable}")
print(f"[setup] torch={torch.__version__}")
print(f"[setup] cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu={torch.cuda.get_device_name(0)}")
    print(f"[setup] capability={torch.cuda.get_device_capability(0)}")
    print(f"[setup] bf16={torch.cuda.is_bf16_supported()}")
print(f"[setup] strict-past parameters={N2P3Net(**kwargs).num_parameters():,}")
PY

echo "[setup] complete; future code-only updates use: bash patch.sh <patch.tar.gz>"
