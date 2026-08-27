#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

die() {
    echo "[patch:error] $*" >&2
    exit 2
}

validate_tree() {
    local tree="$1"
    PYTHONPATH="$tree/src:$tree:$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - <<'PY'
from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.n2p3net import N2P3Net
from train.factory import BINARY_MODEL_NAMES

contract = DEFAULT_P300_DATA_CONTRACT
assert contract.sample_rate_hz == 128.0
assert contract.n_times == 128
model = N2P3Net(
    8,
    n_times=contract.n_times,
    sfreq=contract.sample_rate_hz,
    tmin_s=contract.tmin_ms / 1000.0,
)
record = model.architecture_record()
assert record["st_temporal_receptive_span_ms"] == 500.0
assert record["mst_receptive_span_ms"] == [125.0, 500.0]
required = {"n2p3net_lmbc", "n2p3net_global_average", "ms_eegnet", "eegnet"}
assert required <= set(BINARY_MODEL_NAMES)
print(
    "[patch] contract=128Hz/128 samples "
    "receptive_spans=500/125/500ms ablations=ready"
)
PY
}

[[ -n "$PATCH_FILE" ]] || die "usage: bash patch.sh /path/to/n2p3net-patch.tar.gz"
[[ -f "$PATCH_FILE" ]] || die "patch file not found: $PATCH_FILE"
command -v tar >/dev/null 2>&1 || die "tar is unavailable"
[[ -x "$PYTHON_BIN" ]] || die "project Python is unavailable: $PYTHON_BIN"

TMP_DIR="$(mktemp -d)"
BACKUP_DIR="$ROOT/.patch-backups/$(date -u +%Y%m%dT%H%M%SZ)"
trap 'rm -rf "$TMP_DIR"' EXIT

mapfile -t ENTRIES < <(tar -tzf "$PATCH_FILE")
[[ "${#ENTRIES[@]}" -gt 0 ]] || die "patch archive is empty"
for entry in "${ENTRIES[@]}"; do
    case "$entry" in
        /*|../*|*/../*|*'/..'|.git/*|.venv/*|experiments/cache/*|experiments/runs/*|tmp/*)
            die "patch contains protected path: $entry"
            ;;
    esac
done

if printf '%s\n' "${ENTRIES[@]}" | grep -q '^n2p3-net/'; then
    tar -xzf "$PATCH_FILE" -C "$TMP_DIR" --strip-components=1 --no-same-owner
else
    tar -xzf "$PATCH_FILE" -C "$TMP_DIR" --no-same-owner
fi

mapfile -d '' -t FILES < <(find "$TMP_DIR" -type f -print0)
[[ "${#FILES[@]}" -gt 0 ]] || die "patch archive contains no files"
"$PYTHON_BIN" -m compileall -q "$TMP_DIR/src" "$TMP_DIR/experiments" || \
    die "patch syntax validation failed"
validate_tree "$TMP_DIR"

mkdir -p "$BACKUP_DIR"
for source in "${FILES[@]}"; do
    relative="${source#"$TMP_DIR"/}"
    target="$ROOT/$relative"
    if [[ -e "$target" ]]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
        cp -a "$target" "$BACKUP_DIR/$relative"
    fi
done

for source in "${FILES[@]}"; do
    relative="${source#"$TMP_DIR"/}"
    target="$ROOT/$relative"
    mkdir -p "$(dirname "$target")"
    cp -a "$source" "$target.patch-new"
    mv -f "$target.patch-new" "$target"
done

validate_tree "$ROOT"
printf 'patch=%s\nsha256=%s\nbackup=%s\n' \
    "$(basename "$PATCH_FILE")" \
    "$(sha256sum "$PATCH_FILE" | awk '{print $1}')" \
    "$BACKUP_DIR" > "$ROOT/.n2p3net_patch_state"
echo "[patch] applied: $(basename "$PATCH_FILE")"
echo "[patch] backup: $BACKUP_DIR"
