"""Evaluate chronological single-subject prefix -> suffix digit decoding.

The cache must be causal (``filter_phase=forward``) and must expose a complete
candidate/repetition chain. Training uses earlier repetitions of every digit;
evaluation uses only later repetitions of the same subject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
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
from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    apply_fold_local_artifact_policy,
)
from data.contract import (  # noqa: E402
    PAPER_GTN_CAUSAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation  # noqa: E402
from data.identity import IdentityExclusionPolicy  # noqa: E402
from data.qc_features import compute_epoch_qc_features  # noqa: E402
from models.decision import (  # noqa: E402
    COUNT_AGGREGATIONS,
    DEFAULT_EVIDENCE_AGGREGATION,
    DEFAULT_EVIDENCE_COUNT_POWER,
)
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    POOLING_MODES,
    N2P3Net,
)
from research.evaluation import (  # noqa: E402
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    source_snapshot_sha256_from_archive_manifest,
)
from train.device import get_device  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    checkpoint_training_contract,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.evaluation import candidate_evidence_endpoints  # noqa: E402
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import (  # noqa: E402
    causal_prefix_suffix_split,
    chronological_time_validation_split,
)


def _load_trunk(
    path: str | Path | dict[str, object],
    dataset,
    *,
    target_subject: str,
    identity_exclusion_policy: IdentityExclusionPolicy,
) -> N2P3Net:
    trunk, _ = load_n2p3_trunk_checkpoint(
        path,
        dataset,
        target_subject=target_subject,
        identity_exclusion_policy=identity_exclusion_policy,
    )
    return trunk


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_fixed_budget_calibration_payload(
    *, pos_weight: float, train_prior: float
) -> dict[str, object]:
    return {
        "source_calibration": {
            "pos_weight": pos_weight,
            "train_prior": train_prior,
            "temperature": 1.0,
            "source": "target_fixed_budget_weighted_ce_analytic",
        }
    }


def _parse_test_repetitions(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    try:
        repetitions = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("test repetitions must be a positive integer or 'all'") from error
    if repetitions < 1:
        raise argparse.ArgumentTypeError("test repetitions must be a positive integer or 'all'")
    return repetitions


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument(
        "--identity-exclusion-policy",
        required=True,
        choices=("source", "source_or_global", "global"),
    )
    parser.add_argument(
        "--prefix-reps",
        type=int,
        default=None,
        help=(
            "Available evidence per candidate before the suffix. Default: 0 for zero_shot "
            "(session-start estimate), 5 for oracle adaptation proxy. Pass 5 explicitly for "
            "a matched-time zero-target-fit comparison."
        ),
    )
    parser.add_argument(
        "--test-reps",
        type=_parse_test_repetitions,
        default=None,
        help=(
            "Positive balanced suffix budget per candidate, or 'all' to retain every "
            "post-boundary observation and report raw-all plus balanced-all endpoints "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--head",
        choices=("auto", "zero_shot", "classifier_fine", "linear", "mlp16", "full_fine"),
        default="auto",
        help=(
            "auto keeps a supervised checkpoint classifier unchanged (zero_shot), "
            "or trains a linear head when no checkpoint is supplied."
        ),
    )
    parser.add_argument(
        "--pooling-mode",
        default=DEFAULT_N2P3_POOLING_MODE,
        choices=sorted(POOLING_MODES - {"latency_marginal_contrast"}),
        help="Scratch-trunk readout hypothesis; ignored when --checkpoint is given.",
    )
    parser.add_argument(
        "--temporal-kernel-size",
        type=int,
        default=DEFAULT_N2P3_ARCHITECTURE.temporal_kernel_size,
        help="ST temporal kernel width for the scratch trunk (odd, >=3).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--epoch-selection",
        choices=("fixed_budget", "target_time_split"),
        default="fixed_budget",
        help=(
            "fixed_budget trains on every prefix row using a preregistered epoch count; "
            "target_time_split selects epochs on a real-time held-out tail then refits all prefix."
        ),
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
        "--aggregation",
        choices=sorted(COUNT_AGGREGATIONS | {"trim0.2"}),
        default=DEFAULT_EVIDENCE_AGGREGATION,
        help=(
            "Suffix candidate evidence aggregation. Mean is the all-evidence default; "
            "use explicit sum for frozen historical reproduction."
        ),
    )
    parser.add_argument(
        "--evidence-count-power",
        type=float,
        default=DEFAULT_EVIDENCE_COUNT_POWER,
        help=(
            "Effective-count exponent for tempered_evidence: 0=mean, "
            "0.5=sqrt-count (default), 1=sum."
        ),
    )
    parser.add_argument(
        "--fold-local-qc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental target-prefix QC arm. It is off by default because source-QC "
            "zero-shot is the current accuracy leader and target QC changes that estimand."
        ),
    )
    parser.add_argument(
        "--allow-oracle-same-selection-adaptation",
        action="store_true",
        help=(
            "Acknowledge that GTN prefix labels reveal the same thought digit tested in "
            "the suffix. Non-zero-shot modes are an oracle personalization proxy, not "
            "an unknown-number calibration estimate."
        ),
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--subject-offset",
        type=int,
        default=0,
        help=(
            "Skip this many leading usable groups before --max-subjects. Combined "
            "with a leave-block-out checkpoint this lets one checkpoint serve "
            "exactly one contiguous block of targets."
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=("default", "gtn_paper"),
        default="default",
        help=(
            "Causal contract family to assert: 'default' is the current 0.1 Hz / "
            "1200 ms contract; 'gtn_paper' is the 0.5 Hz / 1200 ms SOTA anchor."
        ),
    )
    parser.add_argument(
        "--tmax-ms",
        type=float,
        default=None,
        help="Explicit epoch-end recipe override for matched factorials.",
    )
    parser.add_argument(
        "--target-subjects-file",
        type=Path,
        default=None,
        help="JSON list or newline-delimited target subjects assigned to this checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.evidence_count_power) or not 0.0 <= args.evidence_count_power <= 1.0:
        parser.error("--evidence-count-power must be finite and in [0, 1]")
    if args.target_subjects_file is not None and (
        args.subject_offset or args.max_subjects is not None
    ):
        parser.error(
            "--target-subjects-file already defines the exact target cohort; "
            "do not combine it with --subject-offset/--max-subjects"
        )
    source_snapshot_manifest = args.source_snapshot_manifest.resolve()
    source_snapshot_sha256 = source_snapshot_sha256_from_archive_manifest(
        source_snapshot_manifest
    )

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    target_cache_sha256 = str(loaded_epoch_cache_attestation(dataset)["sha256"])
    causal_contract = {
        "default": SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        "gtn_paper": PAPER_GTN_CAUSAL_DATA_CONTRACT,
    }[args.cohort]
    if args.tmax_ms is not None:
        causal_contract = replace(causal_contract, tmax_ms=float(args.tmax_ms))
    assert_p300_input_contract(dataset.preprocessing, causal_contract)
    checkpoint_payload = None
    ckpt_input_stats = None
    if args.checkpoint:
        checkpoint_payload = load_checkpoint_payload(args.checkpoint)
        if (
            checkpoint_training_contract(checkpoint_payload).source_snapshot_sha256
            != source_snapshot_sha256
        ):
            raise ValueError(
                "checkpoint TrainingRunContract source snapshot disagrees with the "
                "verified physical source freeze."
            )
        ckpt_input_stats = checkpoint_input_stats(checkpoint_payload, dataset.n_channels)
    head_mode = args.head
    if head_mode == "auto":
        head_mode = "zero_shot" if checkpoint_payload is not None else "linear"
    if head_mode == "zero_shot" and checkpoint_payload is None:
        parser.error("--head zero_shot requires --checkpoint")
    if head_mode == "classifier_fine" and checkpoint_payload is None:
        parser.error("--head classifier_fine requires --checkpoint")
    if head_mode in {"zero_shot", "classifier_fine"} and checkpoint_payload is not None:
        if not checkpoint_classifier_is_trained(checkpoint_payload):
            parser.error(
                f"--head {head_mode} requires a supervised checkpoint with a trained classifier"
            )
    if head_mode == "zero_shot" and args.fold_local_qc:
        parser.error("zero_shot cannot fit target-prefix QC; use --no-fold-local-qc")
    normalization = args.normalization
    if normalization == "auto":
        normalization = "source" if checkpoint_payload is not None else "target_prefix"
    if checkpoint_payload is None and normalization in {"source", "shrinkage"}:
        parser.error(f"--normalization {normalization} requires --checkpoint")
    if head_mode == "zero_shot" and normalization != "source":
        parser.error("zero_shot uses source checkpoint statistics; target normalization is adaptation")
    if head_mode != "zero_shot" and not args.allow_oracle_same_selection_adaptation:
        parser.error(
            "GTN supervised prefix adaptation uses the same hidden digit as the suffix. "
            "Pass --allow-oracle-same-selection-adaptation only for an explicitly labelled proxy."
        )
    estimand = (
        "target_excluded_session_start_zero_calibration"
        if head_mode == "zero_shot" and args.prefix_reps in {None, 0}
        else "target_excluded_zero_target_fit_late_suffix"
        if head_mode == "zero_shot"
        else "oracle_label_same_selection_personalization_proxy"
    )
    prefix_reps = (
        int(args.prefix_reps)
        if args.prefix_reps is not None
        else (0 if head_mode == "zero_shot" else 5)
    )
    if head_mode != "zero_shot" and prefix_reps < 3:
        parser.error("supervised adaptation requires at least three prefix evidence rows per candidate")
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
    split = causal_prefix_suffix_split(
        dataset,
        prefix_repetitions=prefix_reps,
        test_repetitions=args.test_reps,
        contract=causal_contract,
    )

    if dataset.trial_channel_mask is None:
        dataset_trial_mask = np.broadcast_to(
            np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2]
        ).copy()
    else:
        dataset_trial_mask = np.asarray(dataset.trial_channel_mask, dtype=bool)
    dataset_qc_features = dataset.qc_features
    if dataset_qc_features is None and args.fold_local_qc:
        # QC features are label-free.  Computing them at the runner boundary is
        # safe for an old cache, but the provenance records that the cache was
        # not a v4 attested feature store.
        dataset_qc_features = compute_epoch_qc_features(
            dataset.X,
            channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
            trial_channel_mask=dataset_trial_mask,
        )

    records = []
    groups = split.usable_groups
    requested_groups = set(split.usable_groups) | set(split.excluded_groups)
    target_subject_filter = None
    if args.target_subjects_file is not None:
        raw_subjects = args.target_subjects_file.read_text(encoding="utf-8").strip()
        try:
            decoded = json.loads(raw_subjects)
        except json.JSONDecodeError:
            decoded = [line.strip() for line in raw_subjects.splitlines() if line.strip()]
        if not isinstance(decoded, list) or not all(isinstance(value, str) for value in decoded):
            parser.error("--target-subjects-file must contain a JSON list or one subject per line")
        target_subject_filter = set(decoded)
        scheduled_groups = np.asarray(dataset.event_timeline.group_ids).astype(str)
        scheduled_subjects = np.asarray(dataset.event_timeline.subject_ids).astype(str)
        unknown_targets = target_subject_filter - set(scheduled_subjects.tolist())
        if unknown_targets:
            parser.error(
                "--target-subjects-file contains subjects absent from the target cache: "
                f"{sorted(unknown_targets)}"
            )
        requested_groups = {
            group
            for group in requested_groups
            if len(
                np.unique(scheduled_subjects[scheduled_groups == group])
            )
            == 1
            and str(np.unique(scheduled_subjects[scheduled_groups == group])[0])
            in target_subject_filter
        }
        filtered_groups = []
        for group in groups:
            rows = np.flatnonzero(split.group_ids == group)
            subjects = np.unique(np.asarray(dataset.subject_ids).astype(str)[rows])
            if len(subjects) == 1 and str(subjects[0]) in target_subject_filter:
                filtered_groups.append(group)
        groups = tuple(filtered_groups)
    if args.subject_offset:
        groups = groups[args.subject_offset :]
    if args.max_subjects is not None:
        groups = groups[: args.max_subjects]
    if args.target_subjects_file is None and (args.subject_offset or args.max_subjects is not None):
        requested_groups = set(groups)
    if not groups:
        raise ValueError("No usable target groups remain after subject filtering.")
    for group in groups:
        group_rows = np.flatnonzero(split.group_ids == group)
        target_subjects = np.unique(np.asarray(dataset.subject_ids).astype(str)[group_rows])
        if len(target_subjects) != 1:
            raise ValueError(f"selection group {group!r} must map to exactly one target subject.")
        target_subject = str(target_subjects[0])
        pre_rows = group_rows[split.prefix_mask[group_rows]]
        post_rows = group_rows[split.suffix_mask[group_rows]]
        local_rows = np.concatenate((pre_rows, post_rows))
        n_pre = len(pre_rows)
        n_post = len(post_rows)
        local_train = np.zeros(len(local_rows), dtype=bool)
        local_train[:n_pre] = True
        local_test = ~local_train
        local_repetitions = split.repetition_indices[local_rows]
        local_onsets = split.onset_times_s[local_rows]
        local_available_at = split.evidence_available_times_s[local_rows]
        local_groups = np.asarray(
            [f"{group}:rep{int(rep)}" for rep in local_repetitions], dtype=str
        )
        local_X = np.asarray(dataset.X[local_rows], dtype=np.float32)
        local_y = np.asarray(dataset.y[local_rows], dtype=np.int64)
        local_mask = dataset_trial_mask[local_rows]
        local_qc = None if dataset_qc_features is None else dataset_qc_features.subset(local_rows)
        artifact_audit = {"enabled": False, "scope": "target_prefix"}
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
            # The policy only drops training epochs.  Keep the suffix rows in
            # their original order and pass the frozen channel mask downstream.
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
        if head_mode != "zero_shot" and not bool(effective_train.any()):
            raise ValueError(f"selection group {group!r} has no prefix epochs after QC.")
        X_pre = local_X[effective_train]
        y_pre = local_y[effective_train]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            trunk = _load_trunk(
                checkpoint_payload,
                dataset,
                target_subject=target_subject,
                identity_exclusion_policy=args.identity_exclusion_policy,
            ) if args.checkpoint else N2P3Net(
                dataset.n_channels,
                n_times=dataset.n_times,
                sfreq=dataset.preprocessing.sfreq,
                tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                pooling_mode=args.pooling_mode,
                temporal_kernel_size=args.temporal_kernel_size,
            )
        suffix_X = local_X[local_test]
        suffix_mask = local_mask[local_test]
        y_post = local_y[local_test]
        if head_mode == "zero_shot":
            assert checkpoint_payload is not None and ckpt_input_stats is not None
            suffix_logits = predict_n2p3_checkpoint(
                trunk,
                suffix_X,
                input_stats=ckpt_input_stats,
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
                    input_mean=None if ckpt_input_stats is None else ckpt_input_stats[0],
                    input_std=None if ckpt_input_stats is None else ckpt_input_stats[1],
                ),
                device=device,
            )
            # Select epoch count on earlier->later prefix data, then refit from
            # the same initialization on every retained prefix trial.
            inner_split = None
            if args.epoch_selection == "target_time_split":
                inner_split = chronological_time_validation_split(
                    local_onsets[effective_train],
                    local_available_at[effective_train],
                    y_pre,
                    epoch_start_offset_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
                )
                adapter.fit(
                    X_pre,
                    y_pre,
                    training_mask=inner_split.train_mask,
                    validation_mask=inner_split.validation_mask,
                    trial_channel_mask=local_mask[effective_train],
                )
            else:
                adapter.fit(
                    X_pre,
                    y_pre,
                    trial_channel_mask=local_mask[effective_train],
                )
            suffix_logits = adapter.predict_logit(
                suffix_X, trial_channel_mask=suffix_mask
            )
            if inner_split is not None:
                calibration_logits, calibration_y, calibration_source = (
                    calibration_data_from_model(adapter, X_pre, y_pre)
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
                    "pos_weight": calibration.pos_weight,
                    "train_prior": calibration.train_prior,
                    "order_preserving": True,
                    "probability_calibration_valid_for_refit_model": False,
                }
            else:
                suffix_llr, calibration_record = checkpoint_scores_to_llr(
                    _target_fixed_budget_calibration_payload(
                        pos_weight=adapter.training_pos_weight_,
                        train_prior=adapter.training_prior_,
                    ),
                    suffix_logits,
                )
            total_parameters = adapter.total_parameter_count()
            trainable_parameters = adapter.parameter_count()
            refit_record = {
                "enabled": bool(adapter.last_history.get("refit_full_prefix")),
                "selected_epochs": adapter.last_history.get("best_epoch"),
                "refit_epochs": adapter.last_history.get("refit_epochs"),
                "inner_embargo_trials": int(
                    np.count_nonzero(inner_split.embargo_mask)
                    if inner_split is not None and inner_split.embargo_mask is not None
                    else 0
                ),
                "epoch_selection": args.epoch_selection,
            }
        auc = None
        if len(np.unique(y_post)) == 2:
            auc = float(roc_auc_score(y_post, suffix_logits))
        endpoints = candidate_evidence_endpoints(
            suffix_llr,
            split.candidate_codes[post_rows],
            split.group_ids[post_rows],
            split.truth_by_group,
            split.suffix_repetition_indices[post_rows],
            aggregation=args.aggregation,
            max_repetitions=args.test_reps,
            candidate_vocabulary=split.candidate_vocab,
            evidence_count_power=args.evidence_count_power,
        )
        hits = endpoints["hit_by_repetition"]
        records.append(
            {
                "group": group,
                "target_subject": target_subject,
                "n_prefix": int(n_pre),
                "n_prefix_after_qc": int(effective_train.sum()),
                "n_suffix": int(n_post),
                "binary_auc": auc,
                "hit_by_repetition": hits,
                "hit_eligible_by_repetition": endpoints[
                    "eligible_by_repetition"
                ],
                "hit_correct_by_repetition": endpoints[
                    "correct_by_repetition"
                ],
                "raw_hit_at_all": endpoints["raw_all_hit_rate"],
                "hit_at_all_balanced": endpoints["balanced_all_hit_rate"],
                "all_endpoint_scope": (
                    "all_post_boundary"
                    if args.test_reps is None
                    else "fixed_budget_suffix"
                ),
                "raw_prediction_at_all": endpoints[
                    "raw_predictions_by_group"
                ].get(group),
                "balanced_prediction_at_all": endpoints[
                    "balanced_predictions_by_group"
                ].get(group),
                "suffix_candidate_counts": endpoints[
                    "candidate_counts_by_group"
                ].get(group),
                "balanced_all_repetitions": endpoints[
                    "balanced_repetitions_by_group"
                ].get(group),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "head": head_mode,
                "calibration": calibration_record,
                "prefix_refit": refit_record,
                "artifact_quality": artifact_audit,
                "selected_scheduled_repetitions": (
                    split.selected_scheduled_repetitions.get(group)
                ),
                "evidence_cost": split.evidence_cost_by_group.get(group),
                "evidence_cost_by_repetition": (
                    split.evidence_cost_by_repetition.get(group)
                ),
                "suffix_predictions": {
                    "epoch_rows": post_rows.astype(int).tolist(),
                    "raw_logits": suffix_logits.tolist(),
                    "llr_scores": suffix_llr.tolist(),
                    "labels": y_post.astype(int).tolist(),
                    "candidate_codes": split.candidate_codes[post_rows].astype(int).tolist(),
                    "scheduled_repetition_indices": (
                        split.repetition_indices[post_rows].astype(int).tolist()
                    ),
                    "valid_repetition_indices": (
                        split.suffix_repetition_indices[post_rows].astype(int).tolist()
                    ),
                    "onset_times_s": split.onset_times_s[post_rows].tolist(),
                    "evidence_available_times_s": (
                        split.evidence_available_times_s[post_rows].tolist()
                    ),
                },
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    curve_limit = max(
        int(record["balanced_all_repetitions"]) for record in records
    )
    operational_successes = {
        r: sum(
            int(record["hit_correct_by_repetition"].get(r, 0))
            for record in records
        )
        for r in range(1, curve_limit + 1)
    }
    eligible_by_repetition = {
        r: sum(
            int(record["hit_eligible_by_repetition"].get(r, 0))
            for record in records
        )
        for r in range(1, curve_limit + 1)
    }
    raw_all_successes = sum(int(record["raw_hit_at_all"]) for record in records)
    balanced_all_successes = sum(
        int(record["hit_at_all_balanced"]) for record in records
    )
    evidence_by_repetition = {}
    for repetition in range(1, curve_limit + 1):
        costs = [
            record["evidence_cost_by_repetition"].get(str(repetition))
            for record in records
            if str(repetition) in record["evidence_cost_by_repetition"]
        ]
        evidence_by_repetition[str(repetition)] = {
            "eligible_groups": eligible_by_repetition[repetition],
            "requested_groups": len(requested_groups),
            "coverage": eligible_by_repetition[repetition]
            / max(len(requested_groups), 1),
            "available_trials_mean": float(
                np.mean([cost["available_trials"] for cost in costs])
            ),
            "scheduled_events_through_evidence_mean": float(
                np.mean(
                    [cost["scheduled_events_through_evidence"] for cost in costs]
                )
            ),
            "elapsed_seconds_after_prefix_mean": float(
                np.mean([cost["elapsed_seconds_after_prefix"] for cost in costs])
            ),
        }
    requested_truths = [
        int(split.truth_by_group[group])
        for group in requested_groups
        if group in split.truth_by_group
    ]
    truth_counts = np.bincount(requested_truths, minlength=len(split.candidate_vocab))
    scheduled_groups = np.asarray(dataset.event_timeline.group_ids).astype(str)
    scheduled_subjects = np.asarray(dataset.event_timeline.subject_ids).astype(str)
    requested_subjects = tuple(
        sorted(
            {
                str(subject)
                for group in requested_groups
                for subject in np.unique(scheduled_subjects[scheduled_groups == group])
            }
        )
    )
    checkpoint_sha256 = _sha256_file(args.checkpoint) if args.checkpoint else None
    model_origin = (
        checkpoint_model_origin(
            checkpoint_payload,
            checkpoint_sha256=checkpoint_sha256,
        )
        if checkpoint_payload is not None and checkpoint_sha256 is not None
        else scratch_model_origin(
            {
                "model": "N2P3Net",
                "pooling_mode": args.pooling_mode,
                "temporal_kernel_size": args.temporal_kernel_size,
                "head": head_mode,
                "seed": args.seed,
                "n_channels": dataset.n_channels,
                "n_times": dataset.n_times,
            }
        )
    )
    evaluation_contract = build_evaluation_run_contract(
        arm_name=args.arm_name,
        model_origin=model_origin,
        dataset=dataset,
        target_cache_sha256=target_cache_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        requested_subjects=requested_subjects,
        identity_policy=args.identity_exclusion_policy,
        target_protocol={
            "estimand": estimand,
            "prefix_repetitions": prefix_reps,
            "test_repetitions": "all" if args.test_reps is None else args.test_reps,
            "cohort": args.cohort,
            "epoch_selection": args.epoch_selection,
        },
        adaptation={
            "head": head_mode,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": args.batch_size,
            "adapt_batchnorm": bool(args.adapt_batchnorm),
            "normalization": normalization,
            "target_stat_weight": args.target_stat_weight,
            "fold_local_qc": bool(args.fold_local_qc),
            "identity_exclusion_policy": args.identity_exclusion_policy,
        },
        decision={
            "task": "nine_choice_digit",
            "aggregation": args.aggregation,
            "evidence_count_power": args.evidence_count_power,
            "tie_policy": "abstain",
        },
        evidence_scope={
            "stage": "development",
            "dataset": dataset.name,
            "estimand": estimand,
            "same_selection_oracle_proxy": head_mode != "zero_shot",
            "product_confirmation": False,
        },
    )
    summary = {
        "schema": "n2p3_within_subject_transfer_result/2",
        "run_status": "completed",
        "arm_name": args.arm_name,
        "evaluation_contract": evaluation_contract.record(),
        "evaluation_contract_digest": evaluation_contract.digest(),
        "requested_participant_keys": list(
            evaluation_contract.requested_participant_keys
        ),
        "source_snapshot_manifest": str(source_snapshot_manifest),
        "source_snapshot_sha256": source_snapshot_sha256,
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "target_cache_sha256": target_cache_sha256,
        "prefix_reps": prefix_reps,
        "test_reps": "all" if args.test_reps is None else args.test_reps,
        "test_repetition_mode": "all_post_boundary" if args.test_reps is None else "fixed",
        "all_endpoint_scope": (
            "all_post_boundary" if args.test_reps is None else "fixed_budget_suffix"
        ),
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
        "estimand": estimand,
        "uses_target_prefix_labels": head_mode != "zero_shot",
        "n_groups": len(records),
        "n_subjects": len({rec["target_subject"] for rec in records}),
        "requested_groups": sorted(requested_groups),
        "excluded_groups": {
            group: reason
            for group, reason in split.excluded_groups.items()
            if group in requested_groups
        },
        "eligible_coverage": float(
            len(records) / max(len(requested_groups), 1)
        ),
        "aggregation": args.aggregation,
        "evidence_count_power": args.evidence_count_power,
        "fold_local_qc": bool(args.fold_local_qc),
        "checkpoint_architecture": (
            checkpoint_payload.get("architecture") if checkpoint_payload is not None else None
        ),
        "target_subjects_file": (
            str(args.target_subjects_file.resolve())
            if args.target_subjects_file is not None
            else None
        ),
        "binary_auc_mean": float(np.nanmean([rec["binary_auc"] for rec in records if rec["binary_auc"] is not None])),
        "raw_hit_at_all": raw_all_successes / max(len(records), 1),
        "raw_hit_at_all_operational": raw_all_successes
        / max(len(requested_groups), 1),
        "hit_at_all_balanced": balanced_all_successes / max(len(records), 1),
        "hit_at_all_balanced_operational": balanced_all_successes
        / max(len(requested_groups), 1),
        "raw_hit_at_all_wilson95_operational": _wilson_interval(
            raw_all_successes, len(requested_groups)
        ),
        "hit_at_all_balanced_wilson95_operational": _wilson_interval(
            balanced_all_successes, len(requested_groups)
        ),
        "hit_mean_by_repetition": {
            str(r): operational_successes[r] / eligible_by_repetition[r]
            for r in range(1, curve_limit + 1)
            if eligible_by_repetition[r] > 0
        },
        "hit_coverage_by_repetition": {
            str(r): eligible_by_repetition[r] / max(len(requested_groups), 1)
            for r in range(1, curve_limit + 1)
        },
        "operational_hit_mean_by_repetition": {
            str(r): operational_successes[r] / max(len(requested_groups), 1)
            for r in range(1, curve_limit + 1)
        },
        "operational_hit_wilson95_by_repetition": {
            str(r): _wilson_interval(operational_successes[r], len(requested_groups))
            for r in range(1, curve_limit + 1)
        },
        "evidence_by_repetition": evidence_by_repetition,
        "target_distribution": {
            "counts_by_candidate_code": truth_counts.astype(int).tolist(),
            "uniform_baseline": 1.0 / len(split.candidate_vocab),
            "empirical_majority_baseline": float(truth_counts.max() / max(truth_counts.sum(), 1)),
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
