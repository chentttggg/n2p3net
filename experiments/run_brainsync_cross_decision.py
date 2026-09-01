"""Evaluate known BrainSync decisions against later target-changing 9-choice decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import assert_causal_p300_input_contract  # noqa: E402
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.cohort import read_subject_manifest, resolve_subject_scope  # noqa: E402
from transfer.evaluation import candidate_evidence_endpoints  # noqa: E402
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import calibration_decision_split  # noqa: E402

RESULT_SCHEMA = "n2p3_brainsync_cross_decision_result/1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-selections", type=int, default=5)
    parser.add_argument("--test-reps", type=int, default=5)
    parser.add_argument("--max-test-selections", type=int, default=None)
    parser.add_argument(
        "--head",
        choices=("zero_shot", "classifier_fine", "linear", "mlp16", "full_fine"),
        default="zero_shot",
    )
    parser.add_argument(
        "--normalization",
        choices=("source", "target_prefix", "shrinkage"),
        default="source",
    )
    parser.add_argument("--target-stat-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-subjects-file", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.target_subjects_file is not None and args.max_subjects is not None:
        parser.error("--target-subjects-file cannot be combined with --max-subjects")
    if args.head == "zero_shot" and args.normalization != "source":
        parser.error("zero_shot requires source normalization")

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    assert_causal_p300_input_contract(dataset.preprocessing)
    target_cache_sha = str(read_epoch_cache_attestation(args.dataset_cache)["sha256"])
    split = calibration_decision_split(
        dataset,
        calibration_selections=args.calibration_selections,
        test_repetitions=args.test_reps,
        max_test_selections=args.max_test_selections,
        candidate_vocabulary=range(1, 10),
    )
    manifest_subjects = (
        read_subject_manifest(args.target_subjects_file)
        if args.target_subjects_file is not None
        else None
    )
    try:
        requested_order, usable_order = resolve_subject_scope(
            np.asarray(dataset.subject_ids).astype(str),
            split.usable_subjects,
            requested_subjects=manifest_subjects,
            max_subjects=args.max_subjects,
        )
    except ValueError as error:
        parser.error(str(error))
    if not usable_order:
        raise ValueError("No usable BrainSync subjects remain in the requested cohort.")

    payload = load_checkpoint_payload(args.checkpoint)
    checkpoint_stats = checkpoint_input_stats(payload, dataset.n_channels, required=True)
    assert checkpoint_stats is not None
    if args.head in {"zero_shot", "classifier_fine", "full_fine"} and not (
        checkpoint_classifier_is_trained(payload)
    ):
        parser.error(f"--head {args.head} requires a supervised source classifier")
    epochs = (
        0
        if args.head == "zero_shot"
        else args.epochs
        if args.epochs is not None
        else 5
        if args.head in {"classifier_fine", "full_fine"}
        else 30
    )
    lr = (
        0.0
        if args.head == "zero_shot"
        else args.lr
        if args.lr is not None
        else 1e-4
        if args.head in {"classifier_fine", "full_fine"}
        else 1e-3
    )
    trial_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
    )

    records = []
    for subject in usable_order:
        subject_rows = np.flatnonzero(np.asarray(dataset.subject_ids).astype(str) == subject)
        calibration_rows = subject_rows[split.calibration_mask[subject_rows]]
        test_rows = subject_rows[split.test_mask[subject_rows]]
        if not len(calibration_rows) or not len(test_rows):
            continue
        trunk, _ = load_n2p3_trunk_checkpoint(
            payload,
            dataset,
            target_subject=subject,
            target_cache_sha256=target_cache_sha,
        )
        suffix_X = np.asarray(dataset.X[test_rows], dtype=np.float32)
        suffix_mask = trial_mask[test_rows]
        if args.head == "zero_shot":
            raw_logits = predict_n2p3_checkpoint(
                trunk,
                suffix_X,
                input_stats=checkpoint_stats,
                device=device,
                trial_channel_mask=suffix_mask,
            )
            llr, calibration = checkpoint_scores_to_llr(payload, raw_logits)
            trainable_parameters = 0
        else:
            adapter = SubjectAdapter(
                trunk,
                config=SubjectAdapterConfig(
                    head_kind=args.head,
                    epochs=int(epochs),
                    batch_size=args.batch_size,
                    lr=float(lr),
                    seed=args.seed,
                    val_group_fraction=None,
                    refit_full_prefix=False,
                    freeze_batchnorm_stats=True,
                    input_statistics=args.normalization,
                    target_stat_weight=args.target_stat_weight,
                    input_mean=checkpoint_stats[0],
                    input_std=checkpoint_stats[1],
                ),
                device=device,
            )
            adapter.fit(
                np.asarray(dataset.X[calibration_rows], dtype=np.float32),
                np.asarray(dataset.y[calibration_rows], dtype=np.int64),
                trial_channel_mask=trial_mask[calibration_rows],
            )
            raw_logits = adapter.predict_logit(suffix_X, trial_channel_mask=suffix_mask)
            llr, calibration = checkpoint_scores_to_llr(
                {
                    "training_pos_weight": adapter.training_pos_weight_,
                    "training_prior": adapter.training_prior_,
                },
                raw_logits,
            )
            calibration["source"] = "target_fixed_budget_weighted_ce_analytic"
            trainable_parameters = adapter.parameter_count()

        groups = split.group_ids[test_rows]
        truth = {group: split.truth_by_group[group] for group in np.unique(groups)}
        endpoints = candidate_evidence_endpoints(
            llr,
            split.candidate_codes[test_rows],
            groups,
            truth,
            split.test_repetition_indices[test_rows],
            aggregation="mean",
            max_repetitions=args.test_reps,
            candidate_vocabulary=split.candidate_vocab,
        )
        auc = float(roc_auc_score(dataset.y[test_rows], raw_logits))
        requested = split.requested_test_groups_by_subject[subject]
        eligible = split.test_groups_by_subject[subject]
        failed = split.failed_test_groups_by_subject[subject]
        correct = {
            str(repetition): int(endpoints["correct_by_repetition"].get(repetition, 0))
            for repetition in range(1, args.test_reps + 1)
        }
        records.append(
            {
                "subject": subject,
                "calibration_groups": split.calibration_groups_by_subject[subject],
                "requested_test_groups": requested,
                "test_groups": eligible,
                "failed_test_groups": failed,
                "binary_auc": auc,
                "trainable_parameters": trainable_parameters,
                "calibration": calibration,
                "correct_by_repetition": correct,
                "operational_hit_by_repetition": {
                    str(repetition): correct[str(repetition)] / max(len(requested), 1)
                    for repetition in range(1, args.test_reps + 1)
                },
                "suffix_predictions": {
                    "epoch_rows": test_rows.tolist(),
                    "raw_logits": raw_logits.tolist(),
                    "llr_scores": llr.tolist(),
                    "labels": np.asarray(dataset.y[test_rows], dtype=int).tolist(),
                    "candidate_codes": split.candidate_codes[test_rows].tolist(),
                    "group_ids": groups.tolist(),
                    "repetition_indices": split.test_repetition_indices[test_rows].tolist(),
                },
            }
        )

    by_subject = {record["subject"]: record for record in records}
    ledger = []
    for subject in requested_order:
        requested = split.requested_test_groups_by_subject.get(subject, ())
        eligible = split.test_groups_by_subject.get(subject, ())
        record = by_subject.get(subject)
        correct = {
            str(repetition): (
                int(record["correct_by_repetition"][str(repetition)])
                if record is not None
                else 0
            )
            for repetition in range(1, args.test_reps + 1)
        }
        ledger.append(
            {
                "subject": subject,
                "requested": len(requested),
                "eligible": len(eligible),
                "failed": len(requested) - len(eligible),
                "correct_by_repetition": correct,
                "operational_hit_by_repetition": {
                    str(repetition): correct[str(repetition)] / max(len(requested), 1)
                    for repetition in range(1, args.test_reps + 1)
                },
                "subject_exclusion_reason": split.excluded_subjects.get(subject),
                "failed_test_groups": split.failed_test_groups_by_subject.get(subject, {}),
            }
        )
    requested_decisions = sum(record["requested"] for record in ledger)
    eligible_decisions = sum(record["eligible"] for record in ledger)
    correct = {
        str(repetition): sum(
            record["correct_by_repetition"][str(repetition)] for record in ledger
        )
        for repetition in range(1, args.test_reps + 1)
    }
    summary = {
        "schema": RESULT_SCHEMA,
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "target_cache_sha256": target_cache_sha,
        "input_preprocessing": asdict(dataset.preprocessing),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "head": args.head,
        "normalization": args.normalization,
        "epochs": epochs,
        "lr": lr,
        "seed": args.seed,
        "calibration_selections": args.calibration_selections,
        "test_repetitions": args.test_reps,
        "requested_subjects": list(requested_order),
        "usable_subjects": len(records),
        "excluded_subjects": split.excluded_subjects,
        "decision_accounting": {
            "requested": requested_decisions,
            "eligible": eligible_decisions,
            "failed": requested_decisions - eligible_decisions,
            "correct_by_repetition": correct,
        },
        "operational_subject_hit_by_repetition": {
            str(repetition): float(
                np.mean(
                    [record["operational_hit_by_repetition"][str(repetition)] for record in ledger]
                )
            )
            for repetition in range(1, args.test_reps + 1)
        },
        "operational_decision_hit_by_repetition": {
            str(repetition): correct[str(repetition)] / max(requested_decisions, 1)
            for repetition in range(1, args.test_reps + 1)
        },
        "subject_decision_ledger": ledger,
        "records": records,
        "evidence_scope": "adult BrainSync target-changing cross-decision evaluation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
