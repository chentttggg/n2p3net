#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/n2p3-bi-ad4f2cd}
RUN_ROOT=${RUN_ROOT:-$ROOT/artifacts/multidomain_joint_20260901}
MAX_JOBS=${MAX_JOBS:-8}
cd "$ROOT"

used_bytes=$(df -B1 --output=used /root/autodl-tmp | tail -n 1 | tr -d ' ')
size_bytes=$(df -B1 --output=size /root/autodl-tmp | tail -n 1 | tr -d ' ')
limit_bytes=$((48 * 1024 * 1024 * 1024))
echo "DATA_DISK used=$used_bytes size=$size_bytes limit=$limit_bytes"
if (( used_bytes > limit_bytes )); then
  echo "DATA_DISK_LIMIT_EXCEEDED" >&2
  exit 2
fi

PYTHON=.venv/bin/python
TARGET_CACHE=artifacts/multidomain/bi2014a_candidate_causal_v2_car5.npz
BI_CACHE=artifacts/multidomain/bi2014a_candidate_causal_v2_car5.npz
JOINT_CACHE=artifacts/multidomain/bi_bnci_joint_causal_v2_car5.npz
BLOCK_DIR=experiments/manifests/bi2014a_cross_decision_car5
MANIFEST=experiments/multidomain_joint_manifest_20260901.json
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/results" "$RUN_ROOT/logs"
git rev-parse HEAD >"$RUN_ROOT/source_commit.txt"
sha256sum "$TARGET_CACHE" "$JOINT_CACHE" "$MANIFEST" "$BLOCK_DIR"/*.json \
  >"$RUN_ROOT/input_sha256.txt"
cp "$MANIFEST" "$BLOCK_DIR"/*.json "$RUN_ROOT/"

pids=()
names=()
failed=0
flush_jobs() {
  local index
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
}
launch() {
  local name=$1
  shift
  "$@" >"$RUN_ROOT/logs/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  if (( ${#pids[@]} >= MAX_JOBS )); then flush_jobs; fi
}

arms=("bi_only_car5:$BI_CACHE" "bi_bnci_joint_car5:$JOINT_CACHE")
for arm_spec in "${arms[@]}"; do
  IFS=: read -r arm source_cache <<<"$arm_spec"
  for seed in 20260901 20260902 20260903; do
    for block in 0 1 2 3; do
      name="pretrain_${arm}_seed${seed}_blk${block}"
      checkpoint="$RUN_ROOT/checkpoints/${arm}_seed${seed}_blk${block}.pt"
      if [[ -s "$checkpoint" ]]; then echo "REUSE_COMPLETE $name"; continue; fi
      holdout=$($PYTHON -c "import json; print(','.join(json.load(open('$BLOCK_DIR/block_${block}.json'))))")
      launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_pretrain_supervised.py \
        --source-cache "$source_cache" \
        --holdout-subjects "$holdout" \
        --cohort p300_causal \
        --pooling-mode full_unfold \
        --temporal-kernel-size 35 \
        --epochs 100 \
        --batch-size 512 \
        --seed "$seed" \
        --qc-ptp-uv 0 \
        --checkpoint "$checkpoint" \
        --device cuda
    done
  done
done
flush_jobs
if (( failed != 0 )); then exit 1; fi

for arm_spec in "${arms[@]}"; do
  IFS=: read -r arm _ <<<"$arm_spec"
  for seed in 20260901 20260902 20260903; do
    for block in 0 1 2 3; do
      name="${arm}_seed${seed}_blk${block}"
      result="$RUN_ROOT/results/${name}.json"
      if [[ -s "$result" ]]; then echo "REUSE_COMPLETE $name"; continue; fi
      launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_bi2014a_candidate.py \
        --dataset-cache "$TARGET_CACHE" \
        --checkpoint "$RUN_ROOT/checkpoints/${arm}_seed${seed}_blk${block}.pt" \
        --target-subjects-file "$BLOCK_DIR/block_${block}.json" \
        --calibration-selections 5 \
        --test-reps 2 \
        --head zero_shot \
        --normalization source \
        --batch-size 256 \
        --seed "$seed" \
        --device cuda \
        --output "$result"
    done
  done
done
flush_jobs
if (( failed != 0 )); then exit 1; fi

$PYTHON experiments/analyze_bi2014a_cross_decision.py \
  --result-dir "$RUN_ROOT/results" \
  --block-dir "$BLOCK_DIR" \
  --manifest "$MANIFEST" \
  --output "$RUN_ROOT/analysis.json" \
  >"$RUN_ROOT/logs/analysis.log" 2>&1
tail -n 1 "$RUN_ROOT/logs/analysis.log"
