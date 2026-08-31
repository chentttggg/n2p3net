"""Paired subject analysis for no-finetune and end-to-end tempered evidence arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_end_to_end_tempered_finetune import (  # noqa: E402
    RESULT_SCHEMA,
    read_json_mapping,
    read_subjects,
    validate_manifest,
)
from experiments.run_tempered_evidence_ablation import (  # noqa: E402
    load_ledger_evidence,
    score_subject,
)

ANALYSIS_SCHEMA = "n2p3_end_to_end_tempered_paired_analysis/1"
NO_FINETUNE_ARMS = {
    "no_finetune_mean": 0.0,
    "no_finetune_sqrt_count": 0.5,
    "no_finetune_sum": 1.0,
}
FINE_TUNE_ARMS = ("fine_tuned_fixed_mean", "fine_tuned_learned_tempered")


def _paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> list[float]:
    values = np.empty(iterations, dtype=np.float64)
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        indices = rng.integers(0, len(differences), size=(take, len(differences)))
        values[offset : offset + take] = differences[indices].mean(axis=1)
        offset += take
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _sign_flip_p(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    exceed = 0
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        signs = rng.integers(0, 2, size=(take, len(differences)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed))
        offset += take
    return float((exceed + 1) / (iterations + 1))


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _load_fine_tuned_correctness(
    result_dir: Path,
    *,
    manifest_sha256: str,
    kernels: tuple[int, ...],
    seeds: tuple[int, ...],
    blocks: tuple[int, ...],
    subjects_by_block: dict[int, tuple[str, ...]],
) -> tuple[dict[tuple[int, int, str, str], float], dict[str, Any]]:
    correctness: dict[tuple[int, int, str, str], float] = {}
    training_contracts: dict[str, Any] = {}
    for kernel in kernels:
        for seed in seeds:
            seen: set[str] = set()
            for block in blocks:
                path = result_dir / f"k{kernel}_seed{seed}_blk{block}.json"
                result = read_json_mapping(path, label="end-to-end result")
                if result.get("schema") != RESULT_SCHEMA:
                    raise ValueError(f"{path} has the wrong result schema.")
                if result.get("manifest_sha256") != manifest_sha256:
                    raise ValueError(f"{path} does not match the frozen manifest.")
                if (result.get("kernel"), result.get("seed"), result.get("block")) != (
                    kernel,
                    seed,
                    block,
                ):
                    raise ValueError(f"{path} run identity mismatch.")
                expected_subjects = set(subjects_by_block[block])
                records = result.get("subjects")
                if not isinstance(records, list):
                    raise ValueError(f"{path} lacks subject records.")
                actual_subjects = {str(record["subject"]) for record in records}
                if actual_subjects != expected_subjects:
                    raise ValueError(f"{path} target-subject denominator mismatch.")
                if seen & actual_subjects:
                    raise ValueError("one seed repeats a subject across target blocks.")
                seen |= actual_subjects
                for record in records:
                    subject = str(record["subject"])
                    correctness[(kernel, seed, subject, "fine_tuned_fixed_mean")] = float(
                        bool(record["fixed_mean"]["hit"])
                    )
                    correctness[
                        (kernel, seed, subject, "fine_tuned_learned_tempered")
                    ] = float(bool(record["learned_tempered"]["hit"]))
                training_contracts[f"k{kernel}_seed{seed}_blk{block}"] = result[
                    "training_contract"
                ]
    return correctness, training_contracts


def _arm_metrics(
    correctness: dict[tuple[int, int, str, str], float],
    *,
    kernels: tuple[int, ...],
    seeds: tuple[int, ...],
    subjects: tuple[str, ...],
    arms: tuple[str, ...],
) -> tuple[dict[str, Any], dict[tuple[int, str, str], float]]:
    metrics: dict[str, Any] = {}
    subject_seed_means: dict[tuple[int, str, str], float] = {}
    for kernel in kernels:
        kernel_metrics: dict[str, Any] = {}
        for arm in arms:
            per_seed = {
                str(seed): float(
                    np.mean([correctness[(kernel, seed, subject, arm)] for subject in subjects])
                )
                for seed in seeds
            }
            values = []
            for subject in subjects:
                value = float(
                    np.mean([correctness[(kernel, seed, subject, arm)] for seed in seeds])
                )
                subject_seed_means[(kernel, arm, subject)] = value
                values.append(value)
            kernel_metrics[arm] = {
                "operational_hit_seed_mean": float(np.mean(values)),
                "subject_seed_mean_sd": float(np.std(values, ddof=1)),
                "per_seed": per_seed,
            }
        metrics[str(kernel)] = kernel_metrics
    return metrics, subject_seed_means


def _contrast(
    subject_seed_means: dict[tuple[int, str, str], float],
    correctness: dict[tuple[int, int, str, str], float],
    *,
    left: tuple[int, str],
    right: tuple[int, str],
    subjects: tuple[str, ...],
    seeds: tuple[int, ...],
    iterations: int,
    analysis_seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            subject_seed_means[(left[0], left[1], subject)]
            - subject_seed_means[(right[0], right[1], subject)]
            for subject in subjects
        ],
        dtype=np.float64,
    )
    return {
        "left": {"kernel": left[0], "arm": left[1]},
        "right": {"kernel": right[0], "arm": right[1]},
        "operational_delta": float(differences.mean()),
        "paired_subject_bootstrap_ci95": _paired_bootstrap_ci(
            differences,
            iterations=iterations,
            rng=np.random.default_rng(analysis_seed),
        ),
        "paired_sign_flip_p": _sign_flip_p(
            differences,
            iterations=iterations,
            rng=np.random.default_rng(analysis_seed + 1),
        ),
        "positive_subjects": int(np.sum(differences > 0.0)),
        "negative_subjects": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "delta_by_seed": {
            str(seed): float(
                np.mean(
                    [
                        correctness[(left[0], seed, subject, left[1])]
                        - correctness[(right[0], seed, subject, right[1])]
                        for subject in subjects
                    ]
                )
            )
            for seed in seeds
        },
    }


def analyze(args: argparse.Namespace) -> None:
    manifest, manifest_sha = validate_manifest(args.manifest)
    kernels = tuple(int(value) for value in manifest["kernels"])
    seeds = tuple(int(value) for value in manifest["seeds"])
    blocks = tuple(int(value) for value in manifest["blocks"])
    subjects_by_block = {
        block: read_subjects(Path(args.block_manifest_dir) / f"block_{block}.json")
        for block in blocks
    }
    subjects = tuple(sorted(subject for values in subjects_by_block.values() for subject in values))
    if len(subjects) != int(manifest["requested_subjects"]) or len(set(subjects)) != len(
        subjects
    ):
        raise ValueError("block manifests do not form the requested disjoint cohort.")

    evidence, candidates, subject_blocks = load_ledger_evidence(
        args.ledger_dir,
        kernels=kernels,
        seeds=seeds,
        blocks=blocks,
        subjects_by_block=subjects_by_block,
    )
    if set(subject_blocks) != set(subjects):
        raise ValueError("ledger subjects differ from the frozen block manifests.")
    correctness: dict[tuple[int, int, str, str], float] = {}
    for kernel in kernels:
        for seed in seeds:
            for subject in subjects:
                record = evidence[(kernel, seed, subject)]
                for arm, count_power in NO_FINETUNE_ARMS.items():
                    prediction, _ = score_subject(
                        record, candidates, count_power=count_power
                    )
                    correctness[(kernel, seed, subject, arm)] = float(
                        prediction is not None and prediction == record["truth"]
                    )
    fine_correctness, training_contracts = _load_fine_tuned_correctness(
        Path(args.result_dir),
        manifest_sha256=manifest_sha,
        kernels=kernels,
        seeds=seeds,
        blocks=blocks,
        subjects_by_block=subjects_by_block,
    )
    correctness.update(fine_correctness)
    arms = (*NO_FINETUNE_ARMS, *FINE_TUNE_ARMS)
    metrics, subject_seed_means = _arm_metrics(
        correctness,
        kernels=kernels,
        seeds=seeds,
        subjects=subjects,
        arms=arms,
    )

    comparisons = {
        "fine_learned_K35_minus_K65": ((35, FINE_TUNE_ARMS[1]), (65, FINE_TUNE_ARMS[1])),
        "no_finetune_mean_K35_minus_K65": ((35, "no_finetune_mean"), (65, "no_finetune_mean")),
        "K35_fine_learned_minus_no_finetune_mean": (
            (35, FINE_TUNE_ARMS[1]),
            (35, "no_finetune_mean"),
        ),
        "K65_fine_learned_minus_no_finetune_mean": (
            (65, FINE_TUNE_ARMS[1]),
            (65, "no_finetune_mean"),
        ),
        "K35_fine_fixed_minus_no_finetune_mean": (
            (35, FINE_TUNE_ARMS[0]),
            (35, "no_finetune_mean"),
        ),
        "K65_fine_fixed_minus_no_finetune_mean": (
            (65, FINE_TUNE_ARMS[0]),
            (65, "no_finetune_mean"),
        ),
        "K35_fine_learned_minus_fine_fixed": (
            (35, FINE_TUNE_ARMS[1]),
            (35, FINE_TUNE_ARMS[0]),
        ),
        "K65_fine_learned_minus_fine_fixed": (
            (65, FINE_TUNE_ARMS[1]),
            (65, FINE_TUNE_ARMS[0]),
        ),
    }
    contrasts = {
        name: _contrast(
            subject_seed_means,
            correctness,
            left=left,
            right=right,
            subjects=subjects,
            seeds=seeds,
            iterations=args.iterations,
            analysis_seed=args.analysis_seed + index * 10,
        )
        for index, (name, (left, right)) in enumerate(comparisons.items())
    }
    adjusted = _holm_adjust(
        {name: float(record["paired_sign_flip_p"]) for name, record in contrasts.items()}
    )
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value

    source_rows = sorted(
        {
            int(contract["source_rows_after_qc"])
            for contract in training_contracts.values()
        }
    )
    eligible_groups = sorted(
        {
            int(contract["source_listwise_eligible_groups"])
            for contract in training_contracts.values()
        }
    )
    ineligible_groups = sorted(
        {
            int(contract["source_listwise_ineligible_groups"])
            for contract in training_contracts.values()
        }
    )
    result = {
        "schema": ANALYSIS_SCHEMA,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "result_dir": str(Path(args.result_dir).resolve()),
        "ledger_dir": str(Path(args.ledger_dir).resolve()),
        "requested_subjects": len(subjects),
        "seeds": list(seeds),
        "kernels": list(kernels),
        "arms": list(arms),
        "metrics": metrics,
        "contrasts": contrasts,
        "training_coverage": {
            "source_rows_after_qc_by_block": source_rows,
            "source_listwise_eligible_groups_by_block": eligible_groups,
            "source_listwise_ineligible_groups_by_block": ineligible_groups,
            "every_legal_source_row_once_per_epoch": True,
        },
        "statistical_unit": "subject after averaging three seed correctness indicators",
        "method": {
            "paired_subject_bootstrap_iterations": args.iterations,
            "paired_sign_flip_iterations": args.iterations,
            "analysis_seed": args.analysis_seed,
            "multiplicity": "Holm across the eight reported paired contrasts",
            "normality_assumption": "not used; paired differences are discrete three-seed means with many ties",
        },
        "evidence_scope": (
            "conditional GTN development-cohort inference; not independent confirmation and not adult personalization evidence"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics, "contrasts": contrasts}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--block-manifest-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--analysis-seed", type=int, default=20260901)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
