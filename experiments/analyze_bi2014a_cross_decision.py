"""Aggregate the frozen BI2014a cross-decision model/normalization comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

RESULT_SCHEMA = "n2p3_bi2014a_cross_decision_result/1"
ANALYSIS_SCHEMA = "n2p3_bi2014a_cross_decision_analysis/1"


def _read_json(path: str | Path) -> dict[str, Any]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} must contain a JSON mapping.")
    return decoded


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_ci(
    differences: np.ndarray, *, iterations: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        indices = rng.integers(0, len(differences), size=(take, len(differences)))
        values[offset : offset + take] = differences[indices].mean(axis=1)
        offset += take
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _sign_flip_p(
    differences: np.ndarray, *, iterations: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    exceed = 0
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        signs = rng.integers(0, 2, size=(take, len(differences)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed))
        offset += take
    return float((exceed + 1) / (iterations + 1))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def analyze(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    if manifest.get("schema") != "n2p3_bi2014a_cross_decision_experiment/1":
        raise ValueError("unsupported BI experiment manifest schema.")
    source_training = manifest["source_training"]
    target_protocol = manifest["target_protocol"]
    seeds = tuple(int(value) for value in source_training["seeds"])
    repetitions = int(target_protocol["test_repetitions"])
    arms = tuple(str(record["name"]) for record in manifest["arms"])
    arm_contract = {str(record["name"]): dict(record) for record in manifest["arms"]}
    block_dir = Path(args.block_dir)
    blocks = {
        block: tuple(json.loads((block_dir / f"block_{block}.json").read_text(encoding="utf-8")))
        for block in range(4)
    }
    all_subjects = tuple(sorted(subject for values in blocks.values() for subject in values))
    if len(all_subjects) != 64 or len(set(all_subjects)) != 64:
        raise ValueError("BI block files must be disjoint and cover exactly 64 subjects.")

    subject_values: dict[tuple[str, int, str], float] = {}
    per_run: list[dict[str, Any]] = []
    result_dir = Path(args.result_dir)
    for arm in arms:
        expected = arm_contract[arm]
        for seed in seeds:
            seen: set[str] = set()
            for block, expected_subjects in blocks.items():
                path = result_dir / f"{arm}_seed{seed}_blk{block}.json"
                result = _read_json(path)
                if result.get("schema") != RESULT_SCHEMA:
                    raise ValueError(f"{path} has an unsupported result schema.")
                if result.get("head") != expected["head"] or result.get(
                    "normalization"
                ) != expected["normalization"]:
                    raise ValueError(f"{path} arm contract mismatch.")
                if int(result.get("seed", -1)) != seed:
                    raise ValueError(f"{path} seed mismatch.")
                requested = tuple(str(value) for value in result["requested_subjects"])
                if set(requested) != set(expected_subjects):
                    raise ValueError(f"{path} requested subjects do not equal block {block}.")
                if set(str(value) for value in result["checkpoint_holdout_subjects"]) != set(
                    expected_subjects
                ):
                    raise ValueError(f"{path} checkpoint holdout does not equal block {block}.")
                if result["target_cache_sha256"] != manifest["dataset_cache_sha256"]:
                    raise ValueError(f"{path} target cache identity mismatch.")
                expected_source_cache = expected.get("source_cache_sha256")
                if (
                    expected_source_cache is not None
                    and result.get("checkpoint_source_cache_sha256") != expected_source_cache
                ):
                    raise ValueError(f"{path} checkpoint source cache identity mismatch.")
                ledger = result["subject_decision_ledger"]
                ledger_subjects = {str(record["subject"]) for record in ledger}
                if ledger_subjects != set(expected_subjects) or seen & ledger_subjects:
                    raise ValueError(f"{path} subject ledger is incomplete or duplicated.")
                seen |= ledger_subjects
                for record in ledger:
                    subject = str(record["subject"])
                    subject_values[(arm, seed, subject)] = float(
                        record["operational_hit_by_repetition"][str(repetitions)]
                    )
                accounting = result["decision_accounting"]
                correct = int(accounting["correct_by_repetition"][str(repetitions)])
                per_run.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "block": block,
                        "requested_subjects": len(expected_subjects),
                        "usable_subjects": int(result["n_subjects"]),
                        "requested_decisions": int(accounting["requested"]),
                        "eligible_decisions": int(accounting["eligible"]),
                        "failed_decisions": int(accounting["failed"]),
                        "correct_decisions": correct,
                        "binary_auc_mean": float(result["binary_auc_mean"]),
                    }
                )
            if seen != set(all_subjects):
                raise ValueError(f"{arm} seed {seed} does not cover the 64-subject denominator.")

    subject_seed_means: dict[tuple[str, str], float] = {}
    metrics: dict[str, Any] = {}
    for arm in arms:
        seed_metrics = {}
        for seed in seeds:
            rows = [row for row in per_run if row["arm"] == arm and row["seed"] == seed]
            requested = sum(row["requested_decisions"] for row in rows)
            eligible = sum(row["eligible_decisions"] for row in rows)
            correct = sum(row["correct_decisions"] for row in rows)
            subject_hits = np.asarray(
                [subject_values[(arm, seed, subject)] for subject in all_subjects], dtype=float
            )
            seed_metrics[str(seed)] = {
                "operational_subject_macro_hit": float(subject_hits.mean()),
                "operational_decision_hit": float(correct / max(requested, 1)),
                "eligible_conditional_decision_hit": float(correct / max(eligible, 1)),
                "requested_decisions": requested,
                "eligible_decisions": eligible,
                "failed_decisions": requested - eligible,
                "usable_subjects": sum(row["usable_subjects"] for row in rows),
                "binary_auc_block_mean": float(np.mean([row["binary_auc_mean"] for row in rows])),
            }
        values = []
        for subject in all_subjects:
            value = float(
                np.mean([subject_values[(arm, seed, subject)] for seed in seeds])
            )
            subject_seed_means[(arm, subject)] = value
            values.append(value)
        metrics[arm] = {
            "operational_subject_macro_hit_seed_mean": float(np.mean(values)),
            "subject_seed_mean_sd": float(np.std(values, ddof=1)),
            "per_seed": seed_metrics,
        }

    contrasts: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    iterations = int(manifest["bootstrap_iterations"])
    analysis_seed = int(manifest["analysis_seed"])
    for index, pair in enumerate(manifest["planned_contrasts"]):
        left, right = (str(value) for value in pair)
        name = f"{left}-minus-{right}"
        differences = np.asarray(
            [
                subject_seed_means[(left, subject)]
                - subject_seed_means[(right, subject)]
                for subject in all_subjects
            ],
            dtype=float,
        )
        p_value = _sign_flip_p(
            differences, iterations=iterations, seed=analysis_seed + index * 10
        )
        p_values[name] = p_value
        contrasts[name] = {
            "left": left,
            "right": right,
            "operational_subject_macro_delta": float(differences.mean()),
            "paired_subject_bootstrap_ci95": _bootstrap_ci(
                differences,
                iterations=iterations,
                seed=analysis_seed + index * 10 + 1,
            ),
            "paired_sign_flip_p": p_value,
            "positive_subjects": int(np.sum(differences > 0.0)),
            "negative_subjects": int(np.sum(differences < 0.0)),
            "ties": int(np.sum(differences == 0.0)),
            "delta_by_seed": {
                str(seed): float(
                    np.mean(
                        [
                            subject_values[(left, seed, subject)]
                            - subject_values[(right, seed, subject)]
                            for subject in all_subjects
                        ]
                    )
                )
                for seed in seeds
            },
        }
    for name, value in _holm(p_values).items():
        contrasts[name]["holm_adjusted_p"] = value

    result = {
        "schema": ANALYSIS_SCHEMA,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "result_dir": str(result_dir.resolve()),
        "requested_subjects": len(all_subjects),
        "test_repetitions": repetitions,
        "seeds": list(seeds),
        "metrics": metrics,
        "planned_contrasts": contrasts,
        "runs": per_run,
        "statistical_unit": "subject after averaging the three matched source/adaptation seeds",
        "multiplicity": "Holm across the manifest-declared contrasts",
        "evidence_scope": manifest["evidence_scope"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics, "contrasts": contrasts}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--block-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
