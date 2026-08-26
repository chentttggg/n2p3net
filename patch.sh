#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

die() {
    echo "[patch:error] $*" >&2
    exit 2
}

[[ -n "$PATCH_FILE" ]] || die "用法: bash patch.sh /path/to/n2p3net-patch.tar.gz"
[[ -f "$PATCH_FILE" ]] || die "patch 文件不存在: $PATCH_FILE"
command -v tar >/dev/null 2>&1 || die "找不到 tar"
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    die "找不到已配置 Python: $PYTHON_BIN；先运行 bash run.sh setup"
fi

TMP_DIR="$(mktemp -d)"
BACKUP_DIR="$ROOT/.patch-backups/$(date -u +%Y%m%dT%H%M%SZ)"
trap 'rm -rf "$TMP_DIR"' EXIT

mapfile -t ENTRIES < <(tar -tzf "$PATCH_FILE")
[[ "${#ENTRIES[@]}" -gt 0 ]] || die "patch 是空归档"
for entry in "${ENTRIES[@]}"; do
    case "$entry" in
        /*|../*|*/../*|*'/..'|.git/*|.venv/*|experiments/cache/*|experiments/runs/*|tmp/*)
            die "patch 包含非法或受保护路径: $entry"
            ;;
    esac
done

HAS_ROOT=0
for entry in "${ENTRIES[@]}"; do
    [[ "$entry" == n2p3-net/* ]] && HAS_ROOT=1 && break
done
if [[ "$HAS_ROOT" -eq 1 ]]; then
    tar -xzf "$PATCH_FILE" -C "$TMP_DIR" --strip-components=1 --no-same-owner
    PATCH_ROOT="$TMP_DIR"
else
    tar -xzf "$PATCH_FILE" -C "$TMP_DIR" --no-same-owner
    PATCH_ROOT="$TMP_DIR"
fi

mapfile -d '' -t FILES < <(find "$PATCH_ROOT" -type f -print0)
[[ "${#FILES[@]}" -gt 0 ]] || die "patch 没有文件"

mkdir -p "$BACKUP_DIR"
for source in "${FILES[@]}"; do
    relative="${source#"$PATCH_ROOT"/}"
    case "$relative" in
        /*|../*|*/../*|*'/..'|experiments/cache/*|experiments/runs/*|tmp/*)
            die "patch 包含非法或受保护路径: $relative"
            ;;
    esac
    target="$ROOT/$relative"
    if [[ -e "$target" ]]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
        cp -a "$target" "$BACKUP_DIR/$relative"
    fi
done

"$PYTHON_BIN" -m compileall -q "$PATCH_ROOT/src" "$PATCH_ROOT/experiments" || \
    die "patch 语法检查失败；尚未写入当前环境"

for source in "${FILES[@]}"; do
    relative="${source#"$PATCH_ROOT"/}"
    target="$ROOT/$relative"
    mkdir -p "$(dirname "$target")"
    temporary="$target.patch-new"
    rm -f "$temporary"
    cp -a "$source" "$temporary"
    mv -f "$temporary" "$target"
done

PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
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
print(f"[patch] strict-past parameters={N2P3Net(**kwargs).num_parameters():,}")
PY

printf 'patch=%s\nsha256=%s\nbackup=%s\n' \
    "$(basename "$PATCH_FILE")" \
    "$(sha256sum "$PATCH_FILE" | awk '{print $1}')" \
    "$BACKUP_DIR" > "$ROOT/.n2p3net_patch_state"
echo "[patch] applied: $(basename "$PATCH_FILE")"
echo "[patch] backup: $BACKUP_DIR"
