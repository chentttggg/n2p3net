"""Leakage-controlled acceptance metrics for arbitrary-layout transfer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines.calibration import LogitCalibration
from baselines.repetition_metrics import wolpaw_bits_per_selection
from data.events import selection_group_id


@dataclass(frozen=True)
class BinaryAcceptanceMetrics:
    balanced_accuracy: float
    roc_auc: float
    nll: float
    ece: float
    brier: float
    n_trials: int
    positive_rate: float


@dataclass(frozen=True)
class UnlabeledCalibrationReport:
    n_samples: int
    steps: int
    final_loss: float
    consistency_loss: float
    prior_loss: float
    entropy_loss: float
    label_access: bool
    trainable_parameters: tuple[str, ...]


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if probabilities.ndim != 1 or labels.ndim != 1:
        raise ValueError("probabilities/labels must be one-dimensional.")
    probabilities = probabilities.astype(float, copy=False)
    labels = labels.astype(float, copy=False)
    if len(probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("probabilities/labels must be non-empty and aligned.")
    if (
        n_bins < 2
        or not np.isfinite(probabilities).all()
        or not np.isfinite(labels).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any((labels < 0.0) | (labels > 1.0))
    ):
        raise ValueError("ECE requires finite probabilities and at least two bins.")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(len(labels))
    value = 0.0
    for index in range(int(n_bins)):
        upper = (
            probabilities <= edges[index + 1]
            if index == n_bins - 1
            else probabilities < edges[index + 1]
        )
        selected = (probabilities >= edges[index]) & upper
        if selected.any():
            value += (
                selected.sum()
                / total
                * abs(probabilities[selected].mean() - labels[selected].mean())
            )
    return float(value)


def binary_acceptance_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    calibration: LogitCalibration,
    *,
    calibration_prior: float,
) -> BinaryAcceptanceMetrics:
    """Score held-out labels using a source-only calibration map."""

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("held-out logits/labels must be one-dimensional.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("held-out labels must have an integer dtype.")
    logits = logits.astype(float, copy=False)
    if len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("held-out logits/labels must be non-empty and aligned.")
    if not np.isfinite(logits).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("held-out logits must be finite and labels binary.")
    if not 0.0 < calibration_prior < 1.0:
        raise ValueError("calibration_prior must lie in (0,1).")
    prior_log_odds = math.log(calibration_prior / (1.0 - calibration_prior))
    posterior_log_odds = calibration.to_llr(logits) + prior_log_odds
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(posterior_log_odds, -40.0, 40.0)))
    n_classes = len(np.unique(labels))
    return BinaryAcceptanceMetrics(
        balanced_accuracy=(
            float(
                balanced_accuracy_score(
                    labels,
                    (logits >= calibration.threshold).astype(np.int64),
                )
            )
            if n_classes == 2
            else float("nan")
        ),
        roc_auc=(float(roc_auc_score(labels, logits)) if n_classes == 2 else float("nan")),
        nll=float(np.mean(np.logaddexp(0.0, posterior_log_odds) - labels * posterior_log_odds)),
        ece=expected_calibration_error(probabilities, labels),
        brier=float(np.mean((probabilities - labels) ** 2)),
        n_trials=int(len(labels)),
        positive_rate=float(labels.mean()),
    )


def acquisition_time_seconds(metadata: pd.DataFrame, *, sfreq: float) -> np.ndarray:
    """Resolve actual event acquisition times; never infer them from epoch count."""

    for column in ("acquisition_time_s", "onset", "time_s", "timestamp"):
        if column in metadata:
            values = metadata[column].to_numpy(dtype=float)
            break
    else:
        if "sample" not in metadata:
            raise ValueError(
                "30/60 s calibration requires acquisition_time_s/onset/time_s/timestamp "
                "or sample metadata. Epoch duration is not acquisition duration."
            )
        if sfreq <= 0.0:
            raise ValueError("sfreq must be positive when acquisition time uses sample indices.")
        values = metadata["sample"].to_numpy(dtype=float) / float(sfreq)
    if not np.isfinite(values).all():
        raise ValueError("Acquisition timestamps contain NaN/inf.")
    return values


def unlabeled_calibration_split(
    subject_ids: np.ndarray,
    metadata: pd.DataFrame,
    *,
    duration_s: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Take a chronological prefix per subject without consulting labels."""

    subjects = np.asarray(subject_ids).astype(str)
    if len(subjects) != len(metadata) or duration_s <= 0.0:
        raise ValueError("subject metadata must align and duration_s must be positive.")
    times = acquisition_time_seconds(metadata, sfreq=sfreq)
    calibration = np.zeros(len(subjects), dtype=bool)
    recording_columns = [column for column in ("session", "run", "recording") if column in metadata]
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        elapsed = np.empty(len(rows), dtype=float)
        if recording_columns:
            keys = list(
                metadata.iloc[rows][recording_columns]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        else:
            keys = [("recording",)] * len(rows)
        ordered_keys = list(dict.fromkeys(keys))
        offset = 0.0
        for key in ordered_keys:
            local = np.flatnonzero(np.asarray([value == key for value in keys]))
            local_times = times[rows[local]]
            relative = local_times - local_times.min()
            elapsed[local] = offset + relative
            unique_times = np.unique(local_times)
            positive_steps = np.diff(unique_times)
            positive_steps = positive_steps[positive_steps > 0.0]
            step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
            offset += float(relative.max()) + step
        calibration[rows[elapsed < float(duration_s)]] = True
    evaluation = ~calibration
    invalid_subjects = [
        subject
        for subject in np.unique(subjects)
        if not calibration[subjects == subject].any() or not evaluation[subjects == subject].any()
    ]
    if invalid_subjects:
        raise ValueError(
            f"The {duration_s:g} s prefix leaves no calibration or evaluation trials for "
            f"subjects {invalid_subjects}."
        )
    return calibration, evaluation


def random_channel_masks(
    base_mask: np.ndarray,
    *,
    drop_fraction: float,
    repeats: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic per-trial masks while retaining one real sensor."""

    base = np.asarray(base_mask)
    if base.dtype != np.dtype(bool):
        raise ValueError("base_mask must have boolean dtype.")
    if base.ndim != 2 or not base.any(axis=1).all():
        raise ValueError("base_mask must be (N,C) with one observed channel per trial.")
    if not 0.0 <= drop_fraction < 1.0 or repeats < 1:
        raise ValueError("drop_fraction must be in [0,1) and repeats must be positive.")
    outputs: list[np.ndarray] = []
    for repeat in range(int(repeats)):
        generator = np.random.default_rng(int(seed) + repeat)
        mask = base.copy()
        for row in range(len(mask)):
            observed = np.flatnonzero(mask[row])
            n_drop = min(len(observed) - 1, int(round(len(observed) * drop_fraction)))
            if n_drop > 0:
                mask[row, generator.choice(observed, size=n_drop, replace=False)] = False
        outputs.append(mask)
    return outputs


def _adapter_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    if model.dataset_adapter is not None:
        selected.extend(
            (f"dataset_adapter.{name}", parameter)
            for name, parameter in model.dataset_adapter.named_parameters()
        )
    for block_index, block in enumerate(model.encoder.blocks):
        for name in ("dom_scale", "dom_shift"):
            parameter = getattr(block, name, None)
            if parameter is not None:
                selected.append((f"encoder.blocks.{block_index}.{name}", parameter))
    return selected


def adapt_unlabeled_adapter(
    model,
    X: torch.Tensor,
    channel_mask: torch.Tensor | None,
    *,
    domain_id: int,
    expected_target_prior: float,
    source_calibration: LogitCalibration,
    steps: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    mask_fraction: float = 0.25,
    seed: int = 0,
) -> UnlabeledCalibrationReport:
    """Adapter/FiLM-only test-time calibration with no label argument.

    A full-view teacher anchors the representation while a randomly masked
    student learns montage robustness. A weak Bernoulli-prior constraint avoids
    entropy collapse under the known oddball class prior.
    """

    if X.dim() != 3 or len(X) == 0:
        raise ValueError("Unlabeled calibration X must be non-empty (N,C,T).")
    if not X.is_floating_point():
        raise ValueError("Unlabeled calibration X must have a floating dtype.")
    if not 0.0 < expected_target_prior < 1.0:
        raise ValueError("expected_target_prior must lie in (0,1).")
    if steps < 1 or batch_size < 1 or lr <= 0.0 or not 0.0 < mask_fraction < 1.0:
        raise ValueError("Invalid unlabeled adaptation optimization settings.")
    device = next(model.parameters()).device
    if channel_mask is not None and channel_mask.dtype != torch.bool:
        raise ValueError("Unlabeled channel_mask must have boolean dtype.")
    base_mask = (
        torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        if channel_mask is None
        else channel_mask.to(device=X.device)
    )
    if base_mask.shape == (X.shape[1],):
        base_mask = base_mask[None].expand(X.shape[0], -1)
    if base_mask.shape != X.shape[:2] or not bool(base_mask.any(dim=1).all()):
        raise ValueError("Unlabeled channel_mask must be (C,) or valid (N,C).")

    selected = _adapter_parameters(model)
    if not selected:
        raise ValueError("Unlabeled calibration requires a low-rank adapter or domain FiLM.")
    requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    anchors = {name: parameter.detach().clone() for name, parameter in selected}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=float(lr))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    was_training = model.training
    model.eval()
    last = (float("nan"),) * 4
    try:
        for _ in range(int(steps)):
            indices = torch.randint(
                len(X),
                (min(int(batch_size), len(X)),),
                generator=generator,
            )
            batch = X[indices].to(device)
            observed = base_mask[indices].to(device)
            with torch.no_grad():
                teacher = model(
                    batch,
                    channel_mask=observed,
                    domain_id=torch.full(
                        (len(batch),), int(domain_id), device=device, dtype=torch.long
                    ),
                    return_attention=False,
                )
            dropout_draw = torch.rand(observed.shape, generator=generator).to(device)
            student_mask = observed & (dropout_draw >= float(mask_fraction))
            empty = ~student_mask.any(dim=1)
            if bool(empty.any()):
                fallback = dropout_draw.masked_fill(~observed, -1.0).argmax(dim=1)
                student_mask[empty, fallback[empty]] = True
            student = model(
                batch * student_mask[:, :, None].to(dtype=batch.dtype),
                channel_mask=student_mask,
                domain_id=torch.full(
                    (len(batch),), int(domain_id), device=device, dtype=torch.long
                ),
                return_attention=False,
            )
            if teacher.heads is None or student.heads is None:
                raise RuntimeError("Unlabeled calibration requires task logits.")
            consistency = F.smooth_l1_loss(
                student.shared_features,
                teacher.shared_features.detach(),
            ) + 0.1 * F.mse_loss(
                student.heads.logit_target,
                teacher.heads.logit_target.detach(),
            )
            raw_logits = student.heads.logit_target.float()
            prior_log_odds = math.log(expected_target_prior / (1.0 - expected_target_prior))
            posterior_log_odds = (
                float(source_calibration.llr_slope) * raw_logits
                + float(source_calibration.llr_intercept)
                + prior_log_odds
            )
            probability = posterior_log_odds.sigmoid().clamp(1e-5, 1.0 - 1e-5)
            marginal = probability.mean()
            prior = probability.new_tensor(float(expected_target_prior))
            prior_loss = (
                marginal * (marginal / prior).log()
                + (1.0 - marginal) * ((1.0 - marginal) / (1.0 - prior)).log()
            )
            entropy = -(
                probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()
            ).mean()
            anchor = sum(
                (parameter - anchors[name]).square().mean() for name, parameter in selected
            )
            loss = consistency + 0.1 * prior_loss + 0.01 * entropy + 1e-3 * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = tuple(
                float(value.detach().cpu()) for value in (loss, consistency, prior_loss, entropy)
            )
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
        model.train(was_training)
    return UnlabeledCalibrationReport(
        n_samples=int(len(X)),
        steps=int(steps),
        final_loss=last[0],
        consistency_loss=last[1],
        prior_loss=last[2],
        entropy_loss=last[3],
        label_access=False,
        trainable_parameters=tuple(name for name, _ in selected),
    )


def repetition_acceptance_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    subject_ids: np.ndarray,
    calibration: LogitCalibration,
    *,
    evidence_ks: tuple[int, ...] = (3, 5, 10, 15),
    repetition_duration_s: float | None = None,
    target_error_rate: float = 0.05,
) -> dict[str, object] | None:
    """Report fixed-K LLR decisions when candidate metadata are available."""

    if repetition_duration_s is not None:
        raise ValueError(
            "ITR cannot be derived as K * repetition_duration_s for ragged/rejected events; "
            "use a complete online-causal event timeline and time@T metrics."
        )

    digit_column = next(
        (column for column in ("stimulus_digit", "digit") if column in metadata),
        None,
    )
    if digit_column is None:
        return None
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    subjects = np.asarray(subject_ids)
    if logits.ndim != 1 or labels.ndim != 1 or subjects.ndim != 1:
        raise ValueError("Repetition arrays must be one-dimensional.")
    if not (
        np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.dtype(bool)
    ) or not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError("Repetition labels must be integer binary values.")
    logits = logits.astype(float, copy=False)
    subjects = subjects.astype(str)
    if not (len(logits) == len(labels) == len(metadata) == len(subjects)):
        raise ValueError("Repetition arrays and metadata must align.")
    digits = metadata[digit_column].to_numpy()
    if not np.issubdtype(digits.dtype, np.integer):
        raise ValueError("Repetition stimulus digits must have an integer dtype.")
    if any(
        isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
        for k in evidence_ks
    ):
        raise ValueError("evidence_ks must contain positive integers.")
    datasets = (
        metadata["dataset"].astype(str).to_numpy()
        if "dataset" in metadata
        else np.repeat("unknown_dataset", len(metadata))
    )
    sessions = (
        metadata["session"].astype(str).to_numpy()
        if "session" in metadata
        else np.repeat("", len(metadata))
    )
    runs = (
        metadata["run"].astype(str).to_numpy()
        if "run" in metadata
        else np.repeat("", len(metadata))
    )
    selections = (
        metadata["selection_id"].astype(str).to_numpy() if "selection_id" in metadata else subjects
    )
    groups = np.asarray(
        [
            selection_group_id(dataset, subject, session, run, selection)
            for dataset, subject, session, run, selection in zip(
                datasets, subjects, sessions, runs, selections, strict=True
            )
        ]
    )
    llr = calibration.to_llr(logits)
    points: dict[str, dict[str, float | int]] = {}
    repetitions_to_target: int | None = None
    for k in evidence_ks:
        hits: list[bool] = []
        nlls: list[float] = []
        confidences: list[float] = []
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            vocab = np.unique(digits[rows])
            true_digits = np.unique(digits[rows][labels[rows] > 0])
            if len(vocab) < 2 or len(true_digits) != 1:
                continue
            selected: list[int] = []
            complete = True
            for digit in vocab:
                digit_rows = rows[digits[rows] == digit]
                if len(digit_rows) < int(k):
                    complete = False
                    break
                selected.extend(digit_rows[: int(k)].tolist())
            if not complete:
                continue
            scores = np.asarray([llr[selected][digits[selected] == digit].sum() for digit in vocab])
            scores = scores - scores.max()
            probability = np.exp(scores) / np.exp(scores).sum()
            target_index = int(np.flatnonzero(vocab == true_digits[0])[0])
            predicted = int(np.argmax(scores))
            hits.append(predicted == target_index)
            nlls.append(float(-np.log(np.clip(probability[target_index], 1e-12, 1.0))))
            confidences.append(float(probability[predicted]))
        coverage = len(hits) / max(len(np.unique(groups)), 1)
        accuracy = float(np.mean(hits)) if hits else float("nan")
        bits = (
            wolpaw_bits_per_selection(accuracy, len(np.unique(digits)))
            if math.isfinite(accuracy)
            else float("nan")
        )
        itr = float("nan")
        selection_ece = (
            expected_calibration_error(np.asarray(confidences), np.asarray(hits, dtype=float))
            if hits
            else float("nan")
        )
        points[str(k)] = {
            "accuracy": accuracy,
            "nll": float(np.mean(nlls)) if nlls else float("nan"),
            "ece": selection_ece,
            "coverage": float(coverage),
            "n_covered": len(hits),
            "bits_per_selection": bits,
            "itr_bits_per_minute": itr,
        }
        if (
            repetitions_to_target is None
            and math.isfinite(accuracy)
            and 1.0 - accuracy <= target_error_rate
            and coverage >= 0.90
        ):
            repetitions_to_target = int(k)
    return {
        "points": points,
        "target_error_rate": float(target_error_rate),
        "repetitions_to_target_error": repetitions_to_target,
        "repetition_duration_s": repetition_duration_s,
    }


def dataclass_record(value) -> dict[str, object]:
    return asdict(value)
