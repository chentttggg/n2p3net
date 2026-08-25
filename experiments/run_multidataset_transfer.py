"""Train Neural-RIDE v11 on heterogeneous EpochDataset montages.

Example:
    python experiments/run_multidataset_transfer.py \
      --dataset primary=cache/primary_epoch_dataset_v2.npz \
      --dataset auxiliary=cache/auxiliary_epoch_dataset_v2.npz \
      --main-domain primary --output-dir experiments/runs/v11-transfer
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.calibration import fit_logit_calibration  # noqa: E402
from baselines.evaluate import _fold_threadpool_limits  # noqa: E402
from baselines.multidataset_acceptance import (  # noqa: E402
    adapt_unlabeled_adapter,
    binary_acceptance_metrics,
    dataclass_record,
    random_channel_masks,
    repetition_acceptance_metrics,
    unlabeled_calibration_split,
)
from data.epochs import EpochDataset, load_epoch_dataset  # noqa: E402
from models.multidataset import (  # noqa: E402
    MontageBranchSpec,
    MultiMontageN2P3Net,
    save_multimontage_checkpoint,
)
from train.contracts import TrialContext  # noqa: E402
from train.device import get_device  # noqa: E402
from train.multidataset import (  # noqa: E402
    MultiDatasetSchedule,
    MultiDatasetTrainer,
)
from train.preloaded import PreloadedDataLoader  # noqa: E402
from train.recipe import (  # noqa: E402
    BINARY_ODDBALL_TASK,
    NEURAL_RIDE_V11_TRANSFER,
)


def _parse_datasets(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path_text = value.partition("=")
        name = name.strip()
        if not separator or not name or not path_text.strip():
            raise ValueError("Each --dataset must use NAME=PATH syntax.")
        if name in output:
            raise ValueError(f"Duplicate dataset name {name!r}.")
        output[name] = Path(path_text).expanduser().resolve()
    if len(output) < 2:
        raise ValueError("Multi-dataset transfer requires at least two --dataset entries.")
    return output


def _validate_shared_time_axis(datasets: dict[str, EpochDataset]) -> None:
    first_name = next(iter(datasets))
    first = datasets[first_name].preprocessing
    fields = ("sfreq", "tmin_ms", "tmax_ms", "n_times", "baseline_mode")
    for name, dataset in datasets.items():
        mismatched = [
            field
            for field in fields
            if getattr(dataset.preprocessing, field) != getattr(first, field)
        ]
        if mismatched:
            raise ValueError(
                f"Dataset {name!r} has a different physical time axis from "
                f"{first_name!r}: {mismatched}. Rebuild the cache; do not pad epochs."
            )


def _subject_split(
    dataset: EpochDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(dataset.subject_ids)
    if len(subjects) < 2:
        raise ValueError(
            f"Dataset {dataset.name!r} needs at least two subjects for disjoint validation."
        )
    generator = np.random.default_rng(seed)
    subjects = subjects[generator.permutation(len(subjects))]
    n_validation = min(
        len(subjects) - 1,
        max(1, int(round(len(subjects) * validation_fraction))),
    )
    validation_subjects = subjects[:n_validation]
    validation = np.isin(dataset.subject_ids, validation_subjects)
    return ~validation, validation


def _observed_dataset(dataset: EpochDataset) -> EpochDataset:
    """Remove channels that are permanently absent in this fixed-layout cache."""

    present = np.asarray(dataset.channel_mask, dtype=bool)
    if bool(present.all()):
        return dataset
    output = EpochDataset(
        name=dataset.name,
        X=dataset.X[:, present, :],
        y=dataset.y,
        subject_ids=dataset.subject_ids,
        channel_names=tuple(
            name for name, keep in zip(dataset.channel_names, present, strict=True) if keep
        ),
        channel_positions_m=dataset.channel_positions_m[present],
        channel_mask=np.ones(int(present.sum()), dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline,
        metadata=dataset.metadata,
        provenance={**dataset.provenance, "permanently_absent_channels_removed": True},
        trial_channel_mask=(
            dataset.trial_channel_mask[:, present]
            if dataset.trial_channel_mask is not None
            else None
        ),
    )
    output.validate(require_labels=True)
    return output


def _pos_weight(labels: np.ndarray) -> float:
    positives = int((labels > 0).sum())
    negatives = int((labels <= 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Main-domain training split requires both target classes.")
    return negatives / positives


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return get_device()
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but is unavailable.")
    return device


def _restricted_subject_split(
    dataset: EpochDataset,
    eligible: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(dataset.subject_ids[eligible])
    if len(subjects) < 2:
        raise ValueError(
            f"Dataset {dataset.name!r} needs two eligible subjects for inner validation."
        )
    generator = np.random.default_rng(seed)
    subjects = subjects[generator.permutation(len(subjects))]
    n_validation = min(
        len(subjects) - 1,
        max(1, int(round(len(subjects) * validation_fraction))),
    )
    validation = eligible & np.isin(dataset.subject_ids, subjects[:n_validation])
    return eligible & ~validation, validation


def _model_for_datasets(datasets: dict[str, EpochDataset]) -> MultiMontageN2P3Net:
    profile = next(iter(datasets.values())).preprocessing
    prototype = next(iter(datasets.values()))
    model_kwargs = NEURAL_RIDE_V11_TRANSFER.model_kwargs(
        n_channels=prototype.n_channels,
        channel_names=prototype.channel_names,
        channel_positions_m=tuple(map(tuple, prototype.channel_positions_m.tolist())),
        tmin_ms=profile.tmin_ms,
        tmax_ms=profile.tmax_ms,
        sfreq=profile.sfreq,
        n_time=profile.n_times,
        baseline_mode=profile.baseline_mode,
    )
    for branch_key in ("n_channels", "channel_names", "channel_positions_m"):
        model_kwargs.pop(branch_key)
    montages = {
        name: MontageBranchSpec(
            channel_names=dataset.channel_names,
            channel_positions_m=tuple(map(tuple, dataset.channel_positions_m.tolist())),
            coordinate_registration=_coordinate_registration_contract(dataset),
        )
        for name, dataset in datasets.items()
    }
    return MultiMontageN2P3Net(
        montages,
        canonical_channel_names=NEURAL_RIDE_V11_TRANSFER.canonical_channel_names,
        model_kwargs=model_kwargs,
    )


def _coordinate_registration_contract(dataset: EpochDataset) -> dict[str, object]:
    registration = dataset.provenance.get("coordinate_registration")
    if isinstance(registration, dict):
        return registration
    montage = str(dataset.provenance.get("montage", ""))
    if montage in {"standard_1005", "standard_1020"}:
        return {
            "source": "average_head_template",
            "template": montage,
            "method": "legacy_cache_template_coordinates",
            "coordinate_frame": "head",
            "units": "m",
            "verified": False,
        }
    return {
        "source": "legacy_unverified_coordinates",
        "method": "cache_did_not_record_registration",
        "coordinate_frame": "unknown",
        "units": "m",
        "verified": False,
    }


def _trial_masks(dataset: EpochDataset, rows: np.ndarray) -> torch.Tensor | None:
    if dataset.trial_channel_mask is None:
        return None
    return torch.from_numpy(np.asarray(dataset.trial_channel_mask[rows], dtype=bool))


def _train_fold(
    datasets: dict[str, EpochDataset],
    *,
    active_domains: tuple[str, ...],
    main_domain: str,
    train_masks: dict[str, np.ndarray],
    validation_masks: dict[str, np.ndarray],
    args,
) -> tuple[MultiMontageN2P3Net, MultiDatasetTrainer, dict[str, object], object]:
    if main_domain not in active_domains:
        raise ValueError("Fold main_domain must be an active source domain.")
    domain_names = tuple(datasets)
    main_train = train_masks[main_domain]
    overrides = {
        "main_domain": domain_names.index(main_domain),
        "pos_weight": _pos_weight(datasets[main_domain].y[main_train]),
        "early_stop_patience": args.early_stop_patience,
    }
    if len(active_domains) == 1:
        overrides.update(lambda_adv=0.0, lambda_private=0.0)
    trainer_config = NEURAL_RIDE_V11_TRANSFER.trainer_config(
        BINARY_ODDBALL_TASK,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        overrides=overrides,
    )
    model = _model_for_datasets(datasets)
    train_loaders = {}
    val_loaders = {}
    reconstruction_contexts = {}
    channel_masks = {}
    for name in active_domains:
        dataset = datasets[name]
        train_rows = train_masks[name]
        validation_rows = validation_masks[name]
        epochs = torch.from_numpy(np.asarray(dataset.X, dtype=np.float32))
        labels = torch.from_numpy(np.asarray(dataset.y, dtype=np.float32))
        train_loaders[name] = PreloadedDataLoader(
            epochs[train_rows],
            labels[train_rows],
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            channel_mask=_trial_masks(dataset, train_rows),
        )
        val_loaders[name] = PreloadedDataLoader(
            epochs[validation_rows],
            labels[validation_rows],
            batch_size=args.batch_size,
            shuffle=False,
            channel_mask=_trial_masks(dataset, validation_rows),
        )
        reconstruction_contexts[name] = TrialContext(
            X=epochs[train_rows],
            y=labels[train_rows],
            channel_mask=_trial_masks(dataset, train_rows),
        )
        channel_masks[name] = torch.from_numpy(dataset.channel_mask)
    trainer = MultiDatasetTrainer(
        model,
        trainer_config,
        channel_masks=channel_masks,
        active_domains=active_domains,
        schedule=MultiDatasetSchedule(
            sampling=args.sampling,
            steps_per_epoch=args.steps_per_epoch,
        ),
        device=_resolve_device(args.device),
    )
    history = trainer.fit(
        train_loaders,
        val_loaders,
        reconstruction_contexts=reconstruction_contexts,
    )
    return model, trainer, history, trainer_config


@torch.inference_mode()
def _predict_logits(
    model: MultiMontageN2P3Net,
    dataset: EpochDataset,
    domain: str,
    rows: np.ndarray,
    *,
    batch_size: int,
    channel_mask_override: np.ndarray | None = None,
) -> np.ndarray:
    indices = np.flatnonzero(rows)
    if len(indices) == 0:
        raise ValueError("Prediction split is empty.")
    branch = model.branch(domain)
    device = next(branch.parameters()).device
    domain_id = model.domain_index[domain]
    base_masks = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
    )
    masks = (
        base_masks
        if channel_mask_override is None
        else np.asarray(channel_mask_override, dtype=bool)
    )
    if masks.shape != dataset.X.shape[:2] or np.any(masks & ~base_masks):
        raise ValueError("Prediction masks must align and cannot re-enable missing channels.")
    was_training = branch.training
    branch.eval()
    outputs: list[torch.Tensor] = []
    try:
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            X = torch.from_numpy(np.asarray(dataset.X[batch_indices], dtype=np.float32)).to(device)
            mask = torch.from_numpy(masks[batch_indices]).to(device)
            output = branch(
                X * mask[:, :, None].to(dtype=X.dtype),
                channel_mask=mask,
                domain_id=torch.full((len(X),), domain_id, device=device, dtype=torch.long),
                return_attention=False,
            )
            if output.heads is None:
                raise RuntimeError("Acceptance prediction requires final logits.")
            outputs.append(output.heads.logit_target.detach().cpu().reshape(-1))
    finally:
        branch.train(was_training)
    return torch.cat(outputs).numpy()


def _source_calibration(
    model: MultiMontageN2P3Net,
    datasets: dict[str, EpochDataset],
    source_domain: str,
    validation_mask: np.ndarray,
    *,
    batch_size: int,
):
    logits = _predict_logits(
        model,
        datasets[source_domain],
        source_domain,
        validation_mask,
        batch_size=batch_size,
    )
    labels = datasets[source_domain].y[validation_mask]
    calibration = fit_logit_calibration(
        logits,
        labels,
        source=f"{source_domain}_inner_subject_validation",
    )
    return calibration, float(labels.mean())


def _mean_records(records: list[dict[str, object]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("balanced_accuracy", "roc_auc", "nll", "ece", "brier"):
        values = np.asarray([record[key] for record in records], dtype=float)
        finite = values[np.isfinite(values)]
        output[key] = float(finite.mean()) if len(finite) else float("nan")
    return output


def _run_loso_acceptance(datasets: dict[str, EpochDataset], args) -> dict[str, object]:
    dataset = datasets[args.main_domain]
    subjects = np.unique(dataset.subject_ids)
    if len(subjects) < 3:
        raise ValueError("LOSO acceptance requires at least three main-domain subjects.")
    if args.acceptance_max_loso_folds is not None:
        subjects = subjects[: args.acceptance_max_loso_folds]
    folds: list[dict[str, object]] = []
    active_domains = tuple(datasets)
    for fold_index, subject in enumerate(subjects):
        test = dataset.subject_ids == subject
        train_masks = {}
        validation_masks = {}
        for index, (name, current) in enumerate(datasets.items()):
            if name == args.main_domain:
                eligible = ~test
                train, validation = _restricted_subject_split(
                    current,
                    eligible,
                    validation_fraction=args.validation_fraction,
                    seed=args.seed + fold_index,
                )
            else:
                train, validation = _subject_split(
                    current,
                    validation_fraction=args.validation_fraction,
                    seed=args.seed + fold_index + index,
                )
            train_masks[name] = train
            validation_masks[name] = validation
        with _fold_threadpool_limits():
            model, _, history, _ = _train_fold(
                datasets,
                active_domains=active_domains,
                main_domain=args.main_domain,
                train_masks=train_masks,
                validation_masks=validation_masks,
                args=args,
            )
        calibration, prior = _source_calibration(
            model,
            datasets,
            args.main_domain,
            validation_masks[args.main_domain],
            batch_size=args.batch_size,
        )
        logits = _predict_logits(
            model,
            dataset,
            args.main_domain,
            test,
            batch_size=args.batch_size,
        )
        metrics = binary_acceptance_metrics(
            logits,
            dataset.y[test],
            calibration,
            calibration_prior=prior,
        )
        repetition = repetition_acceptance_metrics(
            logits,
            dataset.y[test],
            dataset.metadata.loc[test].reset_index(drop=True),
            dataset.subject_ids[test],
            calibration,
            repetition_duration_s=args.repetition_duration_s,
        )
        folds.append(
            {
                "held_out_subject": str(subject),
                "metrics": dataclass_record(metrics),
                "repetition": repetition,
                "best_epoch": history["best_epoch"],
                "calibration_source": calibration.source,
            }
        )
    return {
        "protocol": "LOSO",
        "dataset": args.main_domain,
        "n_folds": len(folds),
        "complete_cohort": len(folds) == len(np.unique(dataset.subject_ids)),
        "mean_metrics": _mean_records([fold["metrics"] for fold in folds]),
        "folds": folds,
    }


def _run_subjectwise_unlabeled_calibration(
    model: MultiMontageN2P3Net,
    target: EpochDataset,
    domain: str,
    *,
    duration_s: float,
    base_masks: np.ndarray,
    calibration,
    prior: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, object]:
    """Adapt and score each target subject from an identical zero-shot state."""

    calibration_rows, evaluation_rows = unlabeled_calibration_split(
        target.subject_ids,
        target.metadata,
        duration_s=duration_s,
        sfreq=target.preprocessing.sfreq,
    )
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    subject_records: dict[str, object] = {}
    pooled_logits: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    subjects = target.subject_ids.astype(str)
    try:
        for subject_index, subject in enumerate(np.unique(subjects)):
            model.load_state_dict(initial_state)
            subject_rows = subjects == subject
            subject_calibration = subject_rows & calibration_rows
            subject_evaluation = subject_rows & evaluation_rows
            if not subject_calibration.any() or not subject_evaluation.any():
                raise ValueError(
                    f"Subject {subject!r} has no calibration or evaluation trials after the "
                    f"{duration_s:g} s split."
                )
            report = adapt_unlabeled_adapter(
                model.branch(domain),
                torch.from_numpy(np.asarray(target.X[subject_calibration], dtype=np.float32)),
                torch.from_numpy(np.asarray(base_masks[subject_calibration], dtype=bool)),
                domain_id=model.domain_index[domain],
                expected_target_prior=prior,
                source_calibration=calibration,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                seed=seed + subject_index,
            )
            logits = _predict_logits(
                model,
                target,
                domain,
                subject_evaluation,
                batch_size=batch_size,
            )
            labels = np.asarray(target.y[subject_evaluation], dtype=np.int64)
            metrics = binary_acceptance_metrics(
                logits,
                labels,
                calibration,
                calibration_prior=prior,
            )
            pooled_logits.append(logits)
            pooled_labels.append(labels)
            subject_records[str(subject)] = {
                "adapter_report": dataclass_record(report),
                "metrics": dataclass_record(metrics),
                "n_calibration_trials": int(subject_calibration.sum()),
                "n_evaluation_trials": int(subject_evaluation.sum()),
                "initialization": "shared_zero_shot_checkpoint",
            }
    finally:
        model.load_state_dict(initial_state)

    pooled_metrics = binary_acceptance_metrics(
        np.concatenate(pooled_logits),
        np.concatenate(pooled_labels),
        calibration,
        calibration_prior=prior,
    )
    metric_records = [record["metrics"] for record in subject_records.values()]
    return {
        "adaptation_unit": "subject",
        "state_shared_across_subjects": False,
        "subject_metrics": subject_records,
        "mean_subject_metrics": _mean_records(metric_records),
        "metrics": dataclass_record(pooled_metrics),
        "n_calibration_trials": int(calibration_rows.sum()),
        "n_evaluation_trials": int(evaluation_rows.sum()),
    }


def _run_lodo_acceptance(datasets: dict[str, EpochDataset], args) -> dict[str, object]:
    results: dict[str, object] = {}
    fractions = tuple(float(value) for value in args.channel_drop_fractions.split(","))
    for held_out_index, held_out in enumerate(datasets):
        sources = tuple(name for name in datasets if name != held_out)
        source_main = args.main_domain if args.main_domain in sources else sources[0]
        train_masks = {}
        validation_masks = {}
        for index, name in enumerate(sources):
            train_masks[name], validation_masks[name] = _subject_split(
                datasets[name],
                validation_fraction=args.validation_fraction,
                seed=args.seed + held_out_index + index,
            )
        model, _, history, _ = _train_fold(
            datasets,
            active_domains=sources,
            main_domain=source_main,
            train_masks=train_masks,
            validation_masks=validation_masks,
            args=args,
        )
        calibration, prior = _source_calibration(
            model,
            datasets,
            source_main,
            validation_masks[source_main],
            batch_size=args.batch_size,
        )
        target = datasets[held_out]
        all_rows = np.ones(target.n_epochs, dtype=bool)
        zero_shot_logits = _predict_logits(
            model,
            target,
            held_out,
            all_rows,
            batch_size=args.batch_size,
        )
        zero_shot = binary_acceptance_metrics(
            zero_shot_logits,
            target.y,
            calibration,
            calibration_prior=prior,
        )
        base_masks = (
            np.asarray(target.trial_channel_mask, dtype=bool)
            if target.trial_channel_mask is not None
            else np.broadcast_to(target.channel_mask, target.X.shape[:2])
        )
        masking: dict[str, object] = {}
        for fraction in fractions:
            repeat_metrics = []
            for mask in random_channel_masks(
                base_masks,
                drop_fraction=fraction,
                repeats=args.channel_mask_repeats,
                seed=args.seed + held_out_index * 1000,
            ):
                logits = _predict_logits(
                    model,
                    target,
                    held_out,
                    all_rows,
                    batch_size=args.batch_size,
                    channel_mask_override=mask,
                )
                repeat_metrics.append(
                    dataclass_record(
                        binary_acceptance_metrics(
                            logits,
                            target.y,
                            calibration,
                            calibration_prior=prior,
                        )
                    )
                )
            masking[str(fraction)] = {
                "mean_metrics": _mean_records(repeat_metrics),
                "repeats": repeat_metrics,
            }
        calibration_results: dict[str, object] = {}
        for duration in (30.0, 60.0):
            calibration_results[f"{int(duration)}s"] = _run_subjectwise_unlabeled_calibration(
                model,
                target,
                held_out,
                duration_s=duration,
                base_masks=base_masks,
                calibration=calibration,
                prior=prior,
                steps=args.unlabeled_calibration_steps,
                batch_size=args.batch_size,
                lr=args.unlabeled_calibration_lr,
                seed=args.seed + held_out_index,
            )
        results[held_out] = {
            "source_domains": list(sources),
            "source_selection_domain": source_main,
            "target_labels_visible_during_training": False,
            "target_labels_visible_during_adapter_calibration": False,
            "zero_shot": dataclass_record(zero_shot),
            "zero_shot_repetition": repetition_acceptance_metrics(
                zero_shot_logits,
                target.y,
                target.metadata,
                target.subject_ids,
                calibration,
                repetition_duration_s=args.repetition_duration_s,
            ),
            "unlabeled_calibration": calibration_results,
            "random_channel_masking": masking,
            "best_epoch": history["best_epoch"],
            "calibration_source": calibration.source,
        }
    return {"protocol": "leave-one-dataset-out", "datasets": results}


def _run_acceptance(datasets: dict[str, EpochDataset], args) -> dict[str, object]:
    loso = _run_loso_acceptance(datasets, args)
    lodo = _run_lodo_acceptance(datasets, args)
    return {
        "executed": True,
        "label_boundary": {
            "held_out_labels": "final_scoring_only",
            "unlabeled_adapter_signature_accepts_labels": False,
        },
        "loso": loso,
        "leave_one_dataset_out": lodo,
        "complete": bool(loso["complete_cohort"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeat NAME=PATH for each prepared EpochDataset cache.",
    )
    parser.add_argument("--main-domain", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--sampling", choices=("balanced", "proportional"), default="balanced")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "xpu", "cpu"), default="auto")
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="Execute LOSO, leave-one-dataset-out, zero-shot, unlabeled calibration and masking.",
    )
    parser.add_argument("--acceptance-max-loso-folds", type=int, default=None)
    parser.add_argument("--unlabeled-calibration-steps", type=int, default=50)
    parser.add_argument("--unlabeled-calibration-lr", type=float, default=1e-4)
    parser.add_argument("--channel-drop-fractions", default="0,0.25,0.5,0.75")
    parser.add_argument("--channel-mask-repeats", type=int, default=5)
    parser.add_argument("--repetition-duration-s", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0,1).")
    if args.acceptance_max_loso_folds is not None and args.acceptance_max_loso_folds < 1:
        raise ValueError("--acceptance-max-loso-folds must be positive.")

    paths = _parse_datasets(args.dataset)
    if args.main_domain not in paths:
        raise ValueError("--main-domain must match one of the --dataset names.")
    datasets = {
        name: _observed_dataset(load_epoch_dataset(path, require_labels=True))
        for name, path in paths.items()
    }
    _validate_shared_time_axis(datasets)
    splits = {
        name: _subject_split(
            dataset,
            validation_fraction=args.validation_fraction,
            seed=args.seed + index,
        )
        for index, (name, dataset) in enumerate(datasets.items())
    }

    model, trainer, history, trainer_config = _train_fold(
        datasets,
        active_domains=tuple(datasets),
        main_domain=args.main_domain,
        train_masks={name: split[0] for name, split in splits.items()},
        validation_masks={name: split[1] for name, split in splits.items()},
        args=args,
    )
    acceptance = _run_acceptance(datasets, args) if args.acceptance else None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    save_multimontage_checkpoint(checkpoint_path, model)
    record = {
        "recipe": NEURAL_RIDE_V11_TRANSFER.name,
        "scope": (
            "development_acceptance_complete"
            if acceptance is not None and acceptance["complete"]
            else (
                "development_acceptance_partial"
                if acceptance is not None
                else "development_transfer_training_only"
            )
        ),
        "main_domain": args.main_domain,
        "dataset_paths": {name: str(path) for name, path in paths.items()},
        "datasets": {name: dataset.record() for name, dataset in datasets.items()},
        "trainer_config": asdict(trainer_config),
        "schedule": asdict(trainer.schedule),
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
            "device": str(trainer.device),
        },
        "history": history,
        "checkpoint": str(checkpoint_path),
        "acceptance": acceptance,
    }
    record_path = output_dir / "record.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[checkpoint] {checkpoint_path}", flush=True)
    print(f"[record] {record_path}", flush=True)


if __name__ == "__main__":
    main()
