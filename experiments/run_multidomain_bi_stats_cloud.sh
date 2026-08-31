#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/n2p3-bi-ad4f2cd}
PARENT_RUN_ROOT=${PARENT_RUN_ROOT:-$ROOT/artifacts/multidomain_weighted_20260901}
RUN_ROOT=${RUN_ROOT:-$ROOT/artifacts/multidomain_bi_stats_20260901}
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
JOINT_CACHE=artifacts/multidomain/bi_bnci_joint_causal_v2_car5.npz
BLOCK_DIR=experiments/manifests/bi2014a_cross_decision_car5
MANIFEST=experiments/multidomain_bi_stats_amendment_20260901.json
ARM=bi_bnci_joint_bi_stats_car5
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/results" "$RUN_ROOT/logs"
git rev-parse HEAD >"$RUN_ROOT/source_commit.txt"

if [[ ! -s "$PARENT_RUN_ROOT/analysis.json" ]]; then
  echo "PARENT_ANALYSIS_MISSING $PARENT_RUN_ROOT/analysis.json" >&2
  exit 3
fi
for arm in bi_only_car5 bi_bnci_joint_car5; do
  parent_count=$(find "$PARENT_RUN_ROOT/results" -maxdepth 1 -type f -name "${arm}_*.json" | wc -l)
  if (( parent_count != 12 )); then
    echo "PARENT_RESULT_COUNT arm=$arm expected=12 actual=$parent_count" >&2
    exit 3
  fi
  cp "$PARENT_RUN_ROOT/results"/${arm}_*.json "$RUN_ROOT/results/"
done
sha256sum "$TARGET_CACHE" "$JOINT_CACHE" "$MANIFEST" "$PARENT_RUN_ROOT/analysis.json" \
  "$PARENT_RUN_ROOT/results"/bi_only_car5_*.json \
  "$PARENT_RUN_ROOT/results"/bi_bnci_joint_car5_*.json \
  "$BLOCK_DIR"/*.json >"$RUN_ROOT/input_sha256.txt"
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

for seed in 20260901 20260902 20260903; do
  for block in 0 1 2 3; do
    name="pretrain_${ARM}_seed${seed}_blk${block}"
    checkpoint="$RUN_ROOT/checkpoints/${ARM}_seed${seed}_blk${block}.pt"
    if [[ -s "$checkpoint" ]]; then echo "REUSE_COMPLETE $name"; continue; fi
    holdout=$($PYTHON -c "import json; print(','.join(json.load(open('$BLOCK_DIR/block_${block}.json'))))")
    launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_pretrain_supervised.py \
      --source-cache "$JOINT_CACHE" \
      --holdout-subjects "$holdout" \
      --cohort p300_causal \
      --pooling-mode full_unfold \
      --temporal-kernel-size 35 \
      --epochs 100 \
      --batch-size 512 \
      --seed "$seed" \
      --input-stats-subject-prefix 'BI::' \
      --qc-ptp-uv 0 \
      --checkpoint "$checkpoint" \
      --device cuda
  done
done
flush_jobs
if (( failed != 0 )); then exit 1; fi

for seed in 20260901 20260902 20260903; do
  for block in 0 1 2 3; do
    name="${ARM}_seed${seed}_blk${block}"
    result="$RUN_ROOT/results/${name}.json"
    if [[ -s "$result" ]]; then echo "REUSE_COMPLETE $name"; continue; fi
    launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_bi2014a_candidate.py \
      --dataset-cache "$TARGET_CACHE" \
      --checkpoint "$RUN_ROOT/checkpoints/${ARM}_seed${seed}_blk${block}.pt" \
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
flush_jobs
if (( failed != 0 )); then exit 1; fi

$PYTHON experiments/analyze_bi2014a_cross_decision.py \
  --result-dir "$RUN_ROOT/results" \
  --block-dir "$BLOCK_DIR" \
  --manifest "$MANIFEST" \
  --output "$RUN_ROOT/analysis.json" \
  >"$RUN_ROOT/logs/analysis.log" 2>&1
tail -n 1 "$RUN_ROOT/logs/analysis.log"

used_bytes_after=$(df -B1 --output=used /root/autodl-tmp | tail -n 1 | tr -d ' ')
echo "DATA_DISK_AFTER used=$used_bytes_after limit=$limit_bytes"
if (( used_bytes_after > limit_bytes )); then
  echo "DATA_DISK_LIMIT_EXCEEDED_AFTER_RUN" >&2
  exit 2
fi
