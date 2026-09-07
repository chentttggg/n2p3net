"""Chronological binary calibration without candidate or selection assumptions."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from baselines.features import downsample_epochs
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net


@dataclass(frozen=True)
class BinaryTimeSplit:
    calibration_rows: np.ndarray
    test_rows: np.ndarray
    purged_rows: np.ndarray


def binary_time_split(
    onset_times: np.ndarray,
    available_times: np.ndarray,
    *,
    calibration_trials: int,
    maximum_calibration_trials: int,
    tmin_s: float,
) -> BinaryTimeSplit:
    """Keep one fixed suffix across calibration budgets in a single recording.

    The purge separates raw epoch support. A causal filter may retain past
    state, as in deployment; this is not a claim of independent IIR support.
    """
    onset = np.asarray(onset_times, dtype=np.float64)
    available = np.asarray(available_times, dtype=np.float64)
    if onset.ndim != 1 or available.shape != onset.shape:
        raise ValueError("onset and availability must be aligned vectors.")
    if not np.isfinite(onset).all() or not np.isfinite(available).all():
        raise ValueError("event times must be finite.")
    if not np.isfinite(tmin_s) or np.any(np.diff(onset) <= 0):
        raise ValueError("one recording with strictly increasing onset times is required.")
    if np.any(available < onset):
        raise ValueError("evidence cannot be available before stimulus onset.")
    if not 0 < calibration_trials <= maximum_calibration_trials < len(onset):
        raise ValueError("calibration budgets must retain a later test suffix.")
    rows = np.arange(len(onset), dtype=np.int64)
    boundary = float(available[:maximum_calibration_trials].max())
    later = rows >= maximum_calibration_trials
    test = later & (onset + tmin_s >= boundary)
    if not bool(test.any()):
        raise ValueError("no test evidence remains after the temporal purge.")
    return BinaryTimeSplit(rows[:calibration_trials], rows[test], rows[later & ~test])


def binary_recording_split(
    onset_times: np.ndarray,
    available_times: np.ndarray,
    recording_ids: np.ndarray,
    *,
    recording_order: tuple[str, ...],
    test_recordings: tuple[str, ...],
    calibration_trials: int,
    maximum_calibration_trials: int,
) -> BinaryTimeSplit:
    """Use explicitly ordered earlier recordings and a fixed later-recording test.

    Local clocks may reset between recordings. Ordering is supplied from the
    acquisition protocol, never inferred from anonymized timestamps or strings.
    """
    onset = np.asarray(onset_times, dtype=np.float64)
    available = np.asarray(available_times, dtype=np.float64)
    recordings = np.asarray(recording_ids).astype(str)
    if onset.ndim != 1 or available.shape != onset.shape or recordings.shape != onset.shape:
        raise ValueError("recording ids and event times must be aligned vectors")
    if not np.isfinite(onset).all() or not np.isfinite(available).all() or np.any(available < onset):
        raise ValueError("recording times must be finite and evidence available after onset")
    if len(recording_order) != len(set(recording_order)) or set(recording_order) != set(recordings):
        raise ValueError("recording_order must identify every recording exactly once")
    if not test_recordings or tuple(recording_order[-len(test_recordings):]) != test_recordings:
        raise ValueError("test recordings must be a nonempty suffix of the declared order")
    blocks = []
    for recording in recording_order:
        rows = np.flatnonzero(recordings == recording)
        if np.any(np.diff(onset[rows]) <= 0):
            raise ValueError("onsets must increase within each recording")
        blocks.append(rows)
    first_test = len(blocks) - len(test_recordings)
    if first_test == 0:
        raise ValueError("at least one earlier calibration recording is required")
    calibration_pool = np.concatenate(blocks[:first_test])
    if not 0 < calibration_trials <= maximum_calibration_trials <= len(calibration_pool):
        raise ValueError("calibration budget exceeds the declared earlier recordings")
    return BinaryTimeSplit(
        calibration_pool[:calibration_trials], np.concatenate(blocks[first_test:]),
        np.empty(0, dtype=np.int64),
    )


def plain_trunk_configuration(trunk: N2P3Net) -> N2P3ArchitectureConfig:
    """Return the validated constructor configuration of a plain source core."""
    if type(trunk) is not N2P3Net or trunk.splitbn_config is not None:
        raise ValueError("the matched scratch baseline requires a plain N2P3Net core.")
    return N2P3ArchitectureConfig(
        **{
            field.name: getattr(
                trunk, "dropout_probability" if field.name == "dropout" else field.name
            )
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
