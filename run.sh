#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        PYTHON_BIN="$VENV_DIR/bin/python"
    elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
        PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
    else
        PYTHON_BIN="$VENV_DIR/bin/python"
    fi
fi
CACHE_DIR="${CACHE_DIR:-$ROOT/experiments/cache}"
CACHE_FILE="$CACHE_DIR/gtn_events_v2_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz"

usage() {
    cat <<'EOF'
用法:
  bash run.sh setup                  首次创建云端环境（只需一次）
  bash run.sh patch FILE             快速应用代码 patch，不重装依赖
  bash run.sh check                  检查 Python、CUDA、cache 和参数量
  bash run.sh benchmark              跑 1 fold 速度/显存冒烟
  bash run.sh train [参数...]        strict-past 训练（默认 4 fold）
  bash run.sh monitor RUN PID        记录训练进程/RSS/GPU 资源曲线
  bash run.sh dashboard              启动图形化 dashboard HTTP 服务（含确认式终止按钮）
  bash run.sh help                   显示本帮助

常用环境变量:
  PYTHON_BIN    覆盖项目 Python；默认严格使用 .venv/bin/python
  PYTHON_BOOTSTRAP 仅 setup 创建 .venv 时使用的基础 Python
  BATCH_SIZE    默认 2048；显存紧张时降低 micro-batch
  ACCUM_STEPS   默认 1；配合较小 BATCH_SIZE 保持有效 batch
  MAX_FOLDS     默认 4
  FOLD_JOBS     默认 1；大显存 Linux 卡可设为 4
  FOLD_CPU_THREADS 默认 2；每个并行 fold worker 的 CPU/BLAS 线程数
  CACHE_DIR     GTN cache 目录，默认 experiments/cache
EOF
}

python_cmd() {
    PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

die() {
    echo "[error] $*" >&2
    exit 2
}

check_python() {
    if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        die "找不到项目 venv Python: $PYTHON_BIN；先运行 bash run.sh setup"
    fi
    python_cmd -c 'import sys; print(f"[python] {sys.executable} {sys.version.split()[0]}")'
}

check_cuda() {
    python_cmd - <<'PY'
import torch

print(f"[torch] {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("[cuda] unavailable; 请安装 CUDA-enabled Torch")
props = torch.cuda.get_device_properties(0)
print(f"[cuda] {props.name}, capability {props.major}.{props.minor}, VRAM {props.total_memory / 2**30:.1f} GiB")
print(f"[amp] bf16={torch.cuda.is_bf16_supported()}")
PY
}

check_cache() {
    if [[ -f "$CACHE_FILE" ]]; then
        echo "[cache] $CACHE_FILE ($(du -h "$CACHE_FILE" | awk '{print $1}'))"
    elif [[ -f "$CACHE_DIR/gtn_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz" ]]; then
        die "发现旧版 GTN cache；请使用 gtn_events_v2_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz"
    else
        die "缺少 GTN cache: $CACHE_FILE"
    fi
}

check_model() {
    python_cmd - <<'PY'
from models.n2p3net import N2P3Net
from train.recipe import NEURAL_RIDE_V11_STRICT_PAST_RESEARCH

kwargs = NEURAL_RIDE_V11_STRICT_PAST_RESEARCH.model_kwargs(
    n_channels=3, channel_names=("Fz", "Cz", "Pz"),
    tmin_ms=-200.0, tmax_ms=1200.0, sfreq=256.0, n_time=358,
    baseline_mode="trial", tau0_ms=(220.0, 300.0, 460.0),
    tau0_bounds=((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)),
    sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 150.0)),
)
print(f"[model] strict-past trainable parameters: {N2P3Net(**kwargs).num_parameters():,}")
PY
}

setup() {
    local bootstrap="${BASE_PYTHON:-${PYTHON_BOOTSTRAP:-}}"
    if [[ -z "$bootstrap" ]]; then
        if [[ -x "$VENV_DIR/bin/python" ]]; then
            bootstrap="$VENV_DIR/bin/python"
        elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
            bootstrap="$VENV_DIR/Scripts/python.exe"
        else
            bootstrap="python3"
        fi
    fi
    BASE_PYTHON="$bootstrap" VENV="$VENV_DIR" bash "$ROOT/setup_cloud.sh"
}

check() {
    check_python
    check_cuda
    check_cache
    check_model
    echo "[check] ready"
}

patch_code() {
    bash "$ROOT/patch.sh" "$@"
}

contains_option() {
    local option="$1"
    shift
    local arg
    for arg in "$@"; do
        [[ "$arg" == "$option" || "$arg" == "$option="* ]] && return 0
    done
    return 1
}

train() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        python_cmd experiments/run_n2p3net_gtn.py "$@"
        return
    fi
    check
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
    export FOLD_CPU_THREADS="${FOLD_CPU_THREADS:-2}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    local -a args=(
        --device cuda
        --batch-size "${BATCH_SIZE:-2048}"
        --accum-steps "${ACCUM_STEPS:-1}"
        --fold-jobs "${FOLD_JOBS:-1}"
        --lambda-innovation "${LAMBDA_INNOVATION:-1.0}"
    )
    if ! contains_option --max-folds "$@"; then
        args+=(--max-folds "${MAX_FOLDS:-4}")
    fi
    echo "[train] batch=${BATCH_SIZE:-2048}, accum_steps=${ACCUM_STEPS:-1}, effective_batch=$(( ${BATCH_SIZE:-2048} * ${ACCUM_STEPS:-1} )), fold_jobs=${FOLD_JOBS:-1}, cpu_threads=${FOLD_CPU_THREADS}, AMP=bf16"
    export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYTHON_BIN" experiments/run_n2p3net_gtn.py "${args[@]}" "$@"
}

benchmark() {
    train --benchmark --subjects "${SUBJECTS:-5}" --max-folds 1 --epochs "${EPOCHS:-16}" "$@"
}

monitor() {
    local run_name="${1:-}"
    local pid="${2:-}"
    [[ -n "$run_name" && -n "$pid" ]] || die "用法: bash run.sh monitor RUN_NAME PID"
    shift 2
    python_cmd experiments/watch_resources.py --run "$run_name" --pid "$pid" "$@"
}

dashboard() {
    local port="${DASHBOARD_PORT:-8812}"
    local bind="${DASHBOARD_BIND:-127.0.0.1}"
    echo "[dashboard] http://${bind}:${port}/dashboard.html"
    python_cmd experiments/dashboard_server.py --port "$port" --bind "$bind" --directory "$ROOT/experiments"
}

command_name="${1:-help}"
shift || true
case "$command_name" in
    setup) setup "$@" ;;
    patch) patch_code "$@" ;;
    check) check "$@" ;;
    benchmark) benchmark "$@" ;;
    train) train "$@" ;;
    monitor) monitor "$@" ;;
    dashboard) dashboard "$@" ;;
    help|-h|--help) usage ;;
    *) usage; die "未知命令: $command_name" ;;
esac
