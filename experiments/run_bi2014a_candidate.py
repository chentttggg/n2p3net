"""Evaluate known early BI2014a decisions -> later unknown character decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.calibration import (  # noqa: E402
    calibration_data_from_model,
    fit_weighted_logit_temperature,
)
from data.artifact import FoldLocalArtifactPolicy, apply_fold_local_artifact_policy  # noqa: E402
from data.contract import assert_causal_p300_input_contract  # noqa: E402
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from data.qc_features import compute_epoch_qc_features  # noqa: E402
from models.n2p3net import N2P3Net  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.bi_decision import hit_at_repetition_6x6  # noqa: E402
from transfer.bi_within_subject import bi2014a_calibration_decision_split  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import chronological_time_validation_split  # noqa: E402


def _load_trunk(
    path: str | Path | dict[str, object],
    dataset,
    *,
    target_subject: str,
    target_cache_sha256: str | None = None,
) -> N2P3Net:
    trunk, _ = load_n2p3_trunk_checkpoint(
        path,
        dataset,
        target_subject=target_subject,
        target_cache_sha256=target_cache_sha256,
    )
    return trunk


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibration-selections", type=int, default=5)
    parser.add_argument("--test-reps", type=int, default=2)
    parser.add_argument("--max-test-selections", type=int, default=None)
    parser.add_argument(
        "--head",
        choices=("auto", "zero_shot", "classifier_fine", "linear", "mlp16", "full_fine"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--epoch-selection",
        choices=("fixed_budget", "target_time_split"),
        default="fixed_budget",
    )
    parser.add_argument(
        "--adapt-batchnorm",
        action="store_true",
        help="Experimental full-fine arm: update target BatchNorm running statistics.",
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "source", "target_prefix", "shrinkage"),
        default="auto",
    )
    parser.add_argument("--target-stat-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fold-local-qc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    target_cache_sha256 = str(read_epoch_cache_attestation(args.dataset_cache)["sha256"])
    assert_causal_p300_input_contract(dataset.preprocessing)
    split = bi2014a_calibration_decision_split(
        dataset,
        calibration_selections=args.calibration_selections,
        test_repetitions=args.test_reps,
        max_test_selections=args.max_test_selections,
    )
    subjects = np.asarray(split.usable_subjects)
    requested_subjects = set(np.unique(np.asarray(dataset.subject_ids).astype(str)).tolist())
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        requested_subjects = set(str(value) for value in subjects)

    checkpoint_payload = load_checkpoint_payload(args.checkpoint) if args.checkpoint else None
    checkpoint_stats = (
        checkpoint_input_stats(checkpoint_payload, dataset.n_channels, required=True)
        if checkpoint_payload is not None
        else None
    )
    head_mode = args.head
    if head_mode == "auto":
        head_mode = "zero_shot" if checkpoint_payload is not None else "linear"
    if head_mode in {"zero_shot", "classifier_fine"} and checkpoint_payload is None:
        parser.error(f"--head {head_mode} requires --checkpoint")
    if head_mode in {"zero_shot", "classifier_fine"} and not checkpoint_classifier_is_trained(
        checkpoint_payload
    ):
        parser.error(f"--head {head_mode} requires a supervised classifier checkpoint")
    if head_mode == "zero_shot" and args.fold_local_qc:
        parser.error("zero_shot cannot fit target calibration QC")
    normalization = args.normalization
    if normalization == "auto":
        normalization = "source" if checkpoint_payload is not None else "target_prefix"
    if checkpoint_payload is None and normalization in {"source", "shrinkage"}:
        parser.error(f"--normalization {normalization} requires --checkpoint")
    if head_mode == "zero_shot" and normalization != "source":
        parser.error("zero_shot uses source checkpoint statistics")
    epochs = (
        0
        if head_mode == "zero_shot"
        else int(args.epochs)
        if args.epochs is not None
        else 5
        if head_mode in {"classifier_fine", "full_fine"}
        else 30
    )
    lr = (
        0.0
        if head_mode == "zero_shot"
        else float(args.lr)
        if args.lr is not None
        else 1e-4
        if head_mode in {"classifier_fine", "full_fine"}
        else 1e-3
    )
    trial_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(
            np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2]
        ).copy()
    )
    qc_features = dataset.qc_features
    if qc_features is None and args.fold_local_qc:
        qc_features = compute_epoch_qc_features(
            dataset.X,
            channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
            trial_channel_mask=trial_mask,
        )
    evidence = np.asarray(dataset.event_timeline.evidence_indices, dtype=np.int64)
    available_events = np.flatnonzero(evidence >= 0)
    onset_by_epoch = np.empty(dataset.n_epochs, dtype=float)
    available_by_epoch = np.empty(dataset.n_epochs, dtype=float)
    onset_by_epoch[evidence[available_events]] = np.asarray(
        dataset.event_timeline.onset_times_s, dtype=float
    )[available_events]
    available_by_epoch[evidence[available_events]] = np.asarray(
        dataset.event_timeline.evidence_available_times_s, dtype=float
    )[available_events]

    records = []
    for subject in subjects:
        subject_rows = np.flatnonzero(dataset.subject_ids == subject)
        pre_rows = subject_rows[split.calibration_mask[subject_rows]]
        post_rows = subject_rows[split.test_mask[subject_rows]]
        if not len(pre_rows) or not len(post_rows):
            continue
        local_rows = np.concatenate((pre_rows, post_rows))
        local_train = np.zeros(len(local_rows), dtype=bool)
        local_train[: len(pre_rows)] = True
        local_test = ~local_train
        local_meta = dataset.metadata.iloc[local_rows]
        local_groups = local_meta["selection_id"].astype(str).to_numpy()
        local_X = np.asarray(dataset.X[local_rows], dtype=np.float32)
        local_y = np.asarray(dataset.y[local_rows], dtype=np.int64)
        local_mask = trial_mask[local_rows]
        local_qc = None if qc_features is None else qc_features.subset(local_rows)
        artifact_audit = {"enabled": False, "scope": "known_calibration_decisions"}
        if args.fold_local_qc:
            local_X, local_mask, effective_train, artifact_audit = apply_fold_local_artifact_policy(
                FoldLocalArtifactPolicy(),
                local_X,
                local_groups,
                local_train,
                local_test,
                local_mask,
                local_qc,
                materialize_masked_data=True,
            )
            effective_train &= local_train
            before_counts = np.bincount(local_y[local_train], minlength=2)
            after_counts = np.bincount(local_y[effective_train], minlength=2)
            artifact_audit["train_label_counts_before"] = before_counts.astype(int).tolist()
            artifact_audit["train_label_counts_after"] = after_counts.astype(int).tolist()
            artifact_audit["train_label_retention"] = (
                after_counts / np.maximum(before_counts, 1)
            ).tolist()
        else:
            effective_train = local_train

        trunk = (
            _load_trunk(
                checkpoint_payload,
                dataset,
                target_subject=str(subject),
                target_cache_sha256=target_cache_sha256,
            )
            if args.checkpoint
            else N2P3Net(
                dataset.n_channels,
                n_times=dataset.n_times,
                sfreq=dataset.preprocessing.sfreq,
                tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                pooling_mode="ms_flatten",
            )
        )
        suffix_X = local_X[local_test]
        suffix_mask = local_mask[local_test]
        y_post = local_y[local_test]
        if head_mode == "zero_shot":
            assert checkpoint_payload is not None and checkpoint_stats is not None
            suffix_logits = predict_n2p3_checkpoint(
                trunk,
                suffix_X,
                input_stats=checkpoint_stats,
                device=device,
                trial_channel_mask=suffix_mask,
            )
            suffix_llr, calibration_record = checkpoint_scores_to_llr(
                checkpoint_payload, suffix_logits
            )
            total_parameters = trunk.parameter_count()
            trainable_parameters = 0
            refit_record = {"enabled": False, "mode": "zero_shot"}
        else:
            adapter = SubjectAdapter(
                trunk,
                config=SubjectAdapterConfig(
                    head_kind=head_mode,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=lr,
                    seed=args.seed,
                    val_group_fraction=(
                        None if args.epoch_selection == "fixed_budget" else 0.1
                    ),
                    refit_full_prefix=args.epoch_selection == "target_time_split",
                    freeze_batchnorm_stats=not args.adapt_batchnorm,
                    input_statistics=normalization,
                    target_stat_weight=args.target_stat_weight,
                    input_mean=None if checkpoint_stats is None else checkpoint_stats[0],
                    input_std=None if checkpoint_stats is None else checkpoint_stats[1],
                ),
                device=device,
            )
            calibration_rows = local_rows[effective_train]
            inner = None
            if args.epoch_selection == "target_time_split":
                inner = chronological_time_validation_split(
                    onset_by_epoch[calibration_rows],
                    available_by_epoch[calibration_rows],
                    local_y[effective_train],
                    epoch_start_offset_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
                )
                adapter.fit(
                    local_X[effective_train],
                    local_y[effective_train],
                    training_mask=inner.train_mask,
                    validation_mask=inner.validation_mask,
                    trial_channel_mask=local_mask[effective_train],
                )
            else:
                adapter.fit(
                    local_X[effective_train],
                    local_y[effective_train],
                    trial_channel_mask=local_mask[effective_train],
                )
            suffix_logits = adapter.predict_logit(
                suffix_X, trial_channel_mask=suffix_mask
            )
            if inner is not None:
                calibration_logits, calibration_y, calibration_source = (
                    calibration_data_from_model(
                        adapter, local_X[effective_train], local_y[effective_train]
                    )
                )
                calibration = fit_weighted_logit_temperature(
                    calibration_logits,
                    calibration_y,
                    pos_weight=float(adapter.training_pos_weight_),
                    train_prior=float(adapter.training_prior_),
                    source=calibration_source,
                )
                suffix_llr = calibration.to_llr(suffix_logits)
                calibration_record = {
                    "mode": "weighted_ce_llr_pre_refit_selection_scale",
                    "source": calibration.source,
                    "n_samples": calibration.n_samples,
                    "temperature": calibration.temperature,
                    "offset": calibration.offset,
                    "order_preserving": True,
                    "probability_calibration_valid_for_refit_model": False,
                }
            else:
                suffix_llr, calibration_record = checkpoint_scores_to_llr(
                    {
                        "training_pos_weight": adapter.training_pos_weight_,
                        "training_prior": adapter.training_prior_,
                    },
                    suffix_logits,
                )
                calibration_record["source"] = "target_fixed_budget_weighted_ce_analytic"
            total_parameters = adapter.total_parameter_count()
            trainable_parameters = adapter.parameter_count()
            refit_record = {
                "enabled": bool(adapter.last_history.get("refit_full_prefix")),
                "selected_epochs": adapter.last_history.get("best_epoch"),
                "refit_epochs": adapter.last_history.get("refit_epochs"),
                "inner_embargo_trials": int(
                    np.count_nonzero(inner.embargo_mask)
                    if inner is not None and inner.embargo_mask is not None
                    else 0
                ),
                "epoch_selection": args.epoch_selection,
            }
        auc = None
        if len(np.unique(y_post)) == 2:
            auc = float(roc_auc_score(y_post, suffix_logits))
        meta = dataset.metadata.iloc[post_rows]
        hits = hit_at_repetition_6x6(
            suffix_llr,
            meta["flash_code"].to_numpy(dtype=np.int64),
            meta["target_row"].to_numpy(dtype=np.int64),
            meta["target_col"].to_numpy(dtype=np.int64),
            meta["selection_id"].astype(str).to_numpy(),
            split.test_repetition_indices[post_rows],
            max_repetitions=args.test_reps,
        )
        requested_test = split.requested_test_selections_by_subject[str(subject)]
        eligible_test = split.test_selections_by_subject[str(subject)]
        failed_test = split.failed_test_selections_by_subject[str(subject)]
        correct_by_repetition = {
            str(repetition): int(round(float(hits[repetition]) * len(eligible_test)))
            for repetition in range(1, args.test_reps + 1)
        }
        operational_hit_by_repetition = {
            str(repetition): (
                correct_by_repetition[str(repetition)] / len(requested_test)
                if requested_test
                else 0.0
            )
            for repetition in range(1, args.test_reps + 1)
        }
        records.append(
            {
                "subject": str(subject),
                "n_prefix": int(len(pre_rows)),
                "n_prefix_after_qc": int(effective_train.sum()),
                "n_suffix": int(len(post_rows)),
                "binary_auc": auc,
                "hit_by_repetition": hits,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "head": head_mode,
                "calibration": calibration_record,
                "prefix_refit": refit_record,
                "artifact_quality": artifact_audit,
                "calibration_selections": split.calibration_selections_by_subject[str(subject)],
                "requested_test_selections": requested_test,
                "test_selections": eligible_test,
                "failed_test_selections": failed_test,
                "decision_accounting": {
                    "requested": len(requested_test),
                    "eligible": len(eligible_test),
                    "failed": len(failed_test),
                    "correct_by_repetition": correct_by_repetition,
                    "operational_hit_by_repetition": operational_hit_by_repetition,
                },
                "suffix_predictions": {
                    "epoch_rows": post_rows.astype(int).tolist(),
                    "raw_logits": suffix_logits.tolist(),
                    "llr_scores": suffix_llr.tolist(),
                    "labels": y_post.astype(int).tolist(),
                    "flash_codes": meta["flash_code"].to_numpy(dtype=np.int64).tolist(),
                    "selection_ids": meta["selection_id"].astype(str).tolist(),
                    "repetition_indices": (
                        split.test_repetition_indices[post_rows].astype(int).tolist()
                    ),
                    "onset_times_s": onset_by_epoch[post_rows].tolist(),
                    "evidence_available_times_s": available_by_epoch[post_rows].tolist(),
                },
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    hit_curves = np.asarray(
        [[rec["hit_by_repetition"].get(r, np.nan) for r in range(1, args.test_reps + 1)] for rec in records]
    )
    records_by_subject = {str(record["subject"]): record for record in records}
    subject_decision_ledger = []
    for subject in sorted(requested_subjects):
        requested = split.requested_test_selections_by_subject.get(subject, ())
        eligible = split.test_selections_by_subject.get(subject, ())
        failed = split.failed_test_selections_by_subject.get(subject, {})
        record = records_by_subject.get(subject)
        correct = {
            str(repetition): (
                int(record["decision_accounting"]["correct_by_repetition"][str(repetition)])
                if record is not None
                else 0
            )
            for repetition in range(1, args.test_reps + 1)
        }
        subject_decision_ledger.append(
            {
                "subject": subject,
                "requested": len(requested),
                "eligible": len(eligible),
                "failed": len(requested) - len(eligible),
                "correct_by_repetition": correct,
                "operational_hit_by_repetition": {
                    str(repetition): (
                        correct[str(repetition)] / len(requested) if requested else 0.0
                    )
                    for repetition in range(1, args.test_reps + 1)
                },
                "failed_test_selections": failed,
                "subject_exclusion_reason": split.excluded_subjects.get(subject),
            }
        )
    requested_decisions = sum(item["requested"] for item in subject_decision_ledger)
    eligible_decisions = sum(item["eligible"] for item in subject_decision_ledger)
    failed_decisions = sum(item["failed"] for item in subject_decision_ledger)
    correct_decisions = {
        str(repetition): sum(
            item["correct_by_repetition"][str(repetition)]
            for item in subject_decision_ledger
        )
        for repetition in range(1, args.test_reps + 1)
    }
    summary = {
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint else None,
        "target_cache_sha256": target_cache_sha256,
        "calibration_selections": args.calibration_selections,
        "test_reps": args.test_reps,
        "head": head_mode,
        "epochs": epochs,
        "lr": lr,
        "epoch_selection": args.epoch_selection,
        "adapt_batchnorm": bool(args.adapt_batchnorm),
        "normalization": normalization,
        "target_stat_weight": args.target_stat_weight,
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "estimand": "known_early_decisions_to_later_unknown_decisions",
        "excluded_subjects": split.excluded_subjects,
        "excluded_selections": split.excluded_selections,
        "requested_subjects": sorted(requested_subjects),
        "subject_coverage": len(records) / max(len(requested_subjects), 1),
        "n_subjects": len(records),
        "decision_accounting": {
            "requested": requested_decisions,
            "eligible": eligible_decisions,
            "failed": failed_decisions,
            "correct_by_repetition": correct_decisions,
        },
        "subject_decision_ledger": subject_decision_ledger,
        "binary_auc_mean": float(np.nanmean([rec["binary_auc"] for rec in records if rec["binary_auc"] is not None])),
        "hit_mean_by_repetition": {
            str(r): float(np.nanmean(hit_curves[:, r - 1])) for r in range(1, args.test_reps + 1)
        },
        "operational_subject_hit_by_repetition": {
            str(r): float(
                np.mean(
                    [
                        item["operational_hit_by_repetition"][str(r)]
                        for item in subject_decision_ledger
                    ]
                )
            )
            for r in range(1, args.test_reps + 1)
        },
        "operational_decision_hit_by_repetition": {
            str(r): correct_decisions[str(r)] / max(requested_decisions, 1)
            for r in range(1, args.test_reps + 1)
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
