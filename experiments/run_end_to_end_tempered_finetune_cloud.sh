#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/n2p3-net}
V4_ROOT=${V4_ROOT:-/root/autodl-tmp/n2p3-kernel-ablation-20260831}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/experiments/runs/end_to_end_tempered_20260831}
MAX_JOBS=${MAX_JOBS:-8}

cd "$PROJECT_ROOT"
PYTHON=.venv/bin/python
RUNNER=experiments/run_end_to_end_tempered_finetune.py
MANIFEST=experiments/end_to_end_tempered_finetune_manifest_20260831.json
RUNTIME_LOCK=experiments/end_to_end_tempered_runtime_20260831.sha256
CACHE=$V4_ROOT/artifacts/cache/gtn_causal_ss_lf0.1_t1200.npz
CHECKPOINTS=$V4_ROOT/artifacts/checkpoints
BLOCKS=$V4_ROOT/artifacts/manifests

mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs"
sha256sum --check --strict "$RUNTIME_LOCK"
cp "$MANIFEST" "$RUNTIME_LOCK" "$RUN_ROOT/"

run_one() {
  local kernel=$1
  local seed=$2
  local block=$3
  local result="$RUN_ROOT/results/k${kernel}_seed${seed}_blk${block}.json"
  local checkpoint="$RUN_ROOT/checkpoints/k${kernel}_seed${seed}_blk${block}.pt"
  if [[ -s "$result" && -s "$checkpoint" ]]; then
    echo "REUSE_COMPLETE k${kernel}_seed${seed}_blk${block}"
    return 0
  fi
  "$PYTHON" "$RUNNER" run \
    --dataset-cache "$CACHE" \
    --base-checkpoint "$CHECKPOINTS/k${kernel}_seed${seed}_blk${block}.pt" \
    --target-subjects-file "$BLOCKS/block_${block}.json" \
    --manifest "$MANIFEST" \
    --kernel "$kernel" \
    --seed "$seed" \
    --block "$block" \
    --device cuda \
    --output-checkpoint "$checkpoint" \
    --output "$result" \
    >"$RUN_ROOT/logs/k${kernel}_seed${seed}_blk${block}.log" 2>&1
}

pids=()
names=()
failed=0
for kernel in 35 65; do
  for seed in 20260828 20260829 20260830; do
    for block in 0 1 2 3; do
      run_one "$kernel" "$seed" "$block" &
      pids+=("$!")
      names+=("k${kernel}_seed${seed}_blk${block}")
      if (( ${#pids[@]} >= MAX_JOBS )); then
        for index in "${!pids[@]}"; do
          if ! wait "${pids[$index]}"; then
            echo "RUN_FAILED ${names[$index]}" >&2
            tail -n 30 "$RUN_ROOT/logs/${names[$index]}.log" >&2 || true
            failed=1
          else
            tail -n 1 "$RUN_ROOT/logs/${names[$index]}.log"
          fi
        done
        pids=()
        names=()
      fi
    done
  done
done
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "RUN_FAILED ${names[$index]}" >&2
    tail -n 30 "$RUN_ROOT/logs/${names[$index]}.log" >&2 || true
    failed=1
  else
    tail -n 1 "$RUN_ROOT/logs/${names[$index]}.log"
  fi
done
if (( failed != 0 )); then
  exit 1
fi

"$PYTHON" "$RUNNER" analyze \
  --result-dir "$RUN_ROOT/results" \
  --manifest "$MANIFEST" \
  --output "$RUN_ROOT/analysis.json" \
  >"$RUN_ROOT/logs/analysis.log" 2>&1
tail -n 1 "$RUN_ROOT/logs/analysis.log"
