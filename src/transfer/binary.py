"""Chronological binary calibration without candidate or selection assumptions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from baselines.features import downsample_epochs
from data.calibration import LIMITED_CALIBRATION_FRACTION as LIMITED_CALIBRATION_FRACTION
from data.calibration import STANDARD_CALIBRATION_TRIALS as STANDARD_CALIBRATION_TRIALS
from data.calibration import TEST_SIZE_POLICIES as TEST_SIZE_POLICIES
from data.calibration import BinaryTimeSplit as BinaryTimeSplit
from data.calibration import TestBudget as TestBudget
from data.calibration import binary_recording_split as binary_recording_split
from data.calibration import binary_time_split as binary_time_split
from data.calibration import calibration_budget_requests as calibration_budget_requests
from data.calibration import dataset_calibration_splits as dataset_calibration_splits
from data.calibration import resolve_calibration_splits as resolve_calibration_splits
from data.calibration import validate_calibration_budgets as validate_calibration_budgets
from data.digests import compact_array_digest
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from research.comparison import assert_matched_cells

CALIBRATED_TRANSFER_ARMS = frozenset({"classifier_fine", "full_fine", "feature_shrinkage_lda"})
SCRATCH_CONTROL_ARMS = frozenset({"target_only", "target_only_tuned"})


def validate_matched_transfer_controls(arms: tuple[str, ...], budgets: tuple[int, ...]) -> None:
    if any(k > 0 for k in budgets) and CALIBRATED_TRANSFER_ARMS.intersection(arms):
        if not SCRATCH_CONTROL_ARMS.intersection(arms):
            raise ValueError("calibrated transfer requires a same-K scratch control: target_only or target_only_tuned")


def matched_split_digests(calibration_rows: np.ndarray, test_rows: np.ndarray) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest()
        for name, rows in (("calibration_rows_sha256", calibration_rows), ("test_rows_sha256", test_rows))
    }


def matched_input_digests(
    X: np.ndarray,
    y: np.ndarray,
    test_rows: np.ndarray,
    trial_channel_mask: np.ndarray | None,
    preprocessing,
) -> dict[str, str]:
    """Bind a scored test cohort to the bytes and physical preprocessing contract.

    Row indices alone are insufficient: a participant's queue can change while
    retaining the same participant and K labels.  This digest therefore covers
    the actual scored samples, labels, observed-channel mask and preprocessing
    declaration.  The label digest is kept separate so a changed target queue
    cannot be hidden inside an unchanged feature cache identity.
    """
    values = np.asarray(X)
    labels = np.asarray(y)
    raw_rows = np.asarray(test_rows)
    if raw_rows.ndim != 1 or (len(raw_rows) and raw_rows.dtype.kind not in "iu"):
        raise ValueError("test_rows must contain integer indices")
    rows = raw_rows.astype(np.int64)
    if values.ndim != 3 or labels.ndim != 1 or values.shape[0] != labels.shape[0]:
        raise ValueError("scored X/y must be aligned (N,C,T) and (N,)")
    if rows.ndim != 1 or rows.size == 0 or np.any(rows < 0) or np.any(rows >= len(labels)):
        raise ValueError("test_rows must be a nonempty in-range vector")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("test_rows cannot repeat physical evidence")
    if not np.isfinite(values[rows]).all() or not np.isin(labels[rows], (0, 1)).all():
        raise ValueError("scored values must be finite and labels must be binary")
    if trial_channel_mask is None:
        mask_digest = None
    else:
        mask = np.asarray(trial_channel_mask)
        if mask.shape != values.shape[:2] or mask.dtype != bool:
            raise ValueError("trial_channel_mask must be boolean and aligned with dataset rows/channels")
        mask_digest = compact_array_digest(mask[rows].astype(bool, copy=False))
    if hasattr(preprocessing, "record"):
        declaration = preprocessing.record()
    elif isinstance(preprocessing, dict):
        declaration = preprocessing
    else:
        raise TypeError("preprocessing must provide a record() or be a mapping")
    label_digest = compact_array_digest(labels[rows])
    semantic = {
        "schema": "n2p3_scored_input/1",
        "rows": compact_array_digest(rows),
        "X": compact_array_digest(values[rows]),
        "labels": label_digest,
        "trial_channel_mask": mask_digest,
        "preprocessing": declaration,
    }
    input_digest = hashlib.sha256(json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return {"input_sha256": input_digest, "test_labels_sha256": label_digest}


def paired_transfer_effects(cells: list[dict], *, budget_field: str) -> list[dict]:
    """Report actual-K contrasts, including failed controls, with paired row checks."""
    lookup = {}
    for cell in cells:
        key = (cell["participant_key"], cell[budget_field], cell["arm"])
        if key in lookup:
            raise ValueError("duplicate participant/K/arm outcome")
        lookup[key] = cell
    pairs = []
    for (participant, budget, arm), cell in lookup.items():
        if budget == 0 or arm not in CALIBRATED_TRANSFER_ARMS:
            continue
        controls = [lookup[participant, budget, name] for name in sorted(SCRATCH_CONTROL_ARMS)
                    if (participant, budget, name) in lookup]
        if not controls:
            raise ValueError("missing same-K scratch outcome for calibrated transfer")
        for control in controls:
            assert_matched_cells((cell, control))
            complete = cell["status"] == control["status"] == "completed"
            if complete and not all(isinstance(value, (int, float)) and np.isfinite(value)
                                    for value in (cell.get("auc"), control.get("auc"))):
                raise ValueError("completed paired outcomes require finite AUC")
            pairs.append({
                "participant_key": participant, "actual_calibration_trials": budget,
                "transfer_arm": arm, "control_arm": control["arm"],
                "transfer_status": cell["status"], "control_status": control["status"],
                "status": "completed" if complete else "incomplete_pair",
                "delta_auc": float(cell["auc"] - control["auc"]) if complete else None,
                "calibration_rows_sha256": cell["calibration_rows_sha256"],
                "test_rows_sha256": cell["test_rows_sha256"],
            })
    return pairs


def calibration_arms(arms: tuple[str, ...], budget: int, budgets: tuple[int, ...]) -> tuple[str, ...]:
    if budget == 0:
        return tuple(arm for arm in arms if arm == "zero_shot")
    if 0 not in budgets:
        return arms
    return tuple(arm for arm in arms if (arm == "zero_shot") == (budget == 0))


def plain_trunk_configuration(trunk: N2P3Net) -> N2P3ArchitectureConfig:
    """Return the validated constructor configuration of a plain source core."""
    if type(trunk) is not N2P3Net or trunk.splitbn_config is not None:
        raise ValueError("the matched scratch baseline requires a plain N2P3Net core.")
    attributes = {"dropout": "dropout_probability", "feature_residual": "feature_residual_config"}
    return N2P3ArchitectureConfig(
        **{
            field.name: getattr(trunk, attributes.get(field.name, field.name))
            for field in fields(N2P3ArchitectureConfig)
        }
    )


def scratch_like_trunk(trunk: N2P3Net, *, seed: int) -> N2P3Net:
    """Construct a fresh compact baseline with no pretrained tensors or stats."""
    configuration = plain_trunk_configuration(trunk)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fresh = N2P3Net(
            trunk.n_channels,
            n_times=trunk.n_times,
            sfreq=trunk.sfreq,
            tmin_s=trunk.tmin_s,
            pooling_mode=trunk.pooling_mode,
            **configuration.model_kwargs(),
        )
    if fresh.architecture_record() != trunk.architecture_record():
        raise ValueError("scratch and source architecture records differ.")
    return fresh


def target_shrinkage_scores(
    train: np.ndarray, labels: np.ndarray, test: np.ndarray, *, sfreq: float
) -> np.ndarray:
    """Fit a regularized target-only ERP baseline without test-derived state."""
    train_features = target_shrinkage_features(train, sfreq=sfreq)
    test_features = target_shrinkage_features(test, sfreq=sfreq)
    return shrinkage_feature_scores(train_features, labels, test_features)


def target_shrinkage_features(X: np.ndarray, *, sfreq: float) -> np.ndarray:
    """The fixed ERP feature map shared by research and saved predictors."""
    return downsample_epochs(X, sfreq, target_hz=20.0).reshape(len(X), -1)


def fit_shrinkage_feature_model(train: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Fit the shared scaler and shrinkage LDA on declared training features."""
    model = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    model.fit(train, labels)
    return model


def shrinkage_feature_scores(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Fit an inductive shrinkage head on any declared, fixed feature map."""
    model = fit_shrinkage_feature_model(train, labels)
    return np.asarray(model.decision_function(test), dtype=np.float64)


def copy_shrinkage_lda_to_binary_head(model: Pipeline, head: torch.nn.Linear) -> None:
    """Fold a fitted scaler/LDA into logits whose difference is its decision score.

    The caller owns the training-only fit scope. Further gradient updates make
    this an LDA-initialized linear head, not a newly solved LDA classifier.
    """
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError("expected the shared scaler/shrinkage-LDA pipeline")
    scaler, lda = model.steps[0][1], model.steps[1][1]
    if not isinstance(scaler, StandardScaler) or not isinstance(lda, LinearDiscriminantAnalysis):
        raise ValueError("expected StandardScaler followed by LinearDiscriminantAnalysis")
    check_is_fitted(scaler)
    check_is_fitted(lda)
    if not isinstance(head, torch.nn.Linear) or head.out_features != 2 or head.bias is None:
        raise ValueError("LDA initialization requires a two-output affine linear head")
    if not np.array_equal(lda.classes_, [0, 1]):
        raise ValueError("LDA classes must be non-target=0 and target=1")
    if lda.coef_.shape != (1, head.in_features):
        raise ValueError("LDA feature count and classifier input dimension disagree")
    # a^T ((h - mean) / scale) + b = w^T h + intercept.
    mean = scaler.mean_ if scaler.with_mean else np.zeros(head.in_features)
    scale = scaler.scale_ if scaler.with_std else np.ones(head.in_features)
    weight = lda.coef_[0] / scale
    intercept = lda.intercept_[0] - np.dot(weight, mean)
    weights = torch.as_tensor(
        np.stack((-weight / 2, weight / 2)), dtype=head.weight.dtype, device=head.weight.device
    )
    biases = torch.as_tensor(
        [-intercept / 2, intercept / 2], dtype=head.bias.dtype, device=head.bias.device
    )
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(biases).all()):
        raise ValueError("LDA initialization contains nonfinite classifier parameters")
    with torch.no_grad():
        head.weight.copy_(weights)
        head.bias.copy_(biases)
