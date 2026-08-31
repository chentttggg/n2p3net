#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/n2p3-bi-ad4f2cd}
RUN_ROOT=${RUN_ROOT:-$ROOT/artifacts/bi2014a_cross_decision_20260901}
MAX_JOBS=${MAX_JOBS:-8}
cd "$ROOT"

PYTHON=.venv/bin/python
CACHE=artifacts/bi2014a_candidate_causal_v2.npz
BLOCK_DIR=experiments/manifests/bi2014a_cross_decision_v2
MANIFEST=experiments/bi2014a_cross_decision_manifest_20260901.json
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/results" "$RUN_ROOT/logs"
git rev-parse HEAD >"$RUN_ROOT/source_commit.txt"
sha256sum "$CACHE" "$MANIFEST" "$BLOCK_DIR"/*.json >"$RUN_ROOT/input_sha256.txt"
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
  if (( ${#pids[@]} >= MAX_JOBS )); then
    flush_jobs
  fi
}

for seed in 20260901 20260902 20260903; do
  for block in 0 1 2 3; do
    name="pretrain_seed${seed}_blk${block}"
    checkpoint="$RUN_ROOT/checkpoints/source_k35_seed${seed}_blk${block}.pt"
    if [[ -s "$checkpoint" ]]; then
      echo "REUSE_COMPLETE $name"
      continue
    fi
    holdout=$($PYTHON -c "import json; print(','.join(json.load(open('$BLOCK_DIR/block_${block}.json'))))")
    launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_pretrain_supervised.py \
      --source-cache "$CACHE" \
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
flush_jobs
if (( failed != 0 )); then exit 1; fi

arms=(
  "zero_shot_source:zero_shot:source"
  "classifier_fine_source:classifier_fine:source"
  "classifier_fine_target:classifier_fine:target_prefix"
  "classifier_fine_shrinkage:classifier_fine:shrinkage"
  "full_fine_source:full_fine:source"
  "full_fine_target:full_fine:target_prefix"
  "full_fine_shrinkage:full_fine:shrinkage"
  "linear_source:linear:source"
  "linear_target:linear:target_prefix"
  "linear_shrinkage:linear:shrinkage"
  "mlp16_source:mlp16:source"
  "mlp16_target:mlp16:target_prefix"
  "mlp16_shrinkage:mlp16:shrinkage"
)

for arm_spec in "${arms[@]}"; do
  IFS=: read -r arm head normalization <<<"$arm_spec"
  for seed in 20260901 20260902 20260903; do
    for block in 0 1 2 3; do
      name="${arm}_seed${seed}_blk${block}"
      result="$RUN_ROOT/results/${name}.json"
      if [[ -s "$result" ]]; then
        echo "REUSE_COMPLETE $name"
        continue
      fi
      launch "$name" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_bi2014a_candidate.py \
        --dataset-cache "$CACHE" \
        --checkpoint "$RUN_ROOT/checkpoints/source_k35_seed${seed}_blk${block}.pt" \
        --target-subjects-file "$BLOCK_DIR/block_${block}.json" \
        --calibration-selections 5 \
        --test-reps 2 \
        --head "$head" \
        --normalization "$normalization" \
        --epoch-selection fixed_budget \
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
