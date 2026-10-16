"""Chronological binary calibration without candidate or selection assumptions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from baselines.features import downsample_epochs
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net

STANDARD_CALIBRATION_TRIALS = (0, 200, 800, 2000)
LIMITED_CALIBRATION_FRACTION = 0.8
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
            for field in ("calibration_rows_sha256", "test_rows_sha256"):
                if not cell.get(field) or cell[field] != control.get(field):
                    raise ValueError("transfer and scratch must use the same calibration and test rows")
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


def validate_calibration_budgets(budgets: tuple[int, ...]) -> None:
    if (not budgets or any(type(value) is not int or value < 0 for value in budgets)
            or tuple(sorted(set(budgets))) != budgets):
        raise ValueError("calibration budgets must be unique increasing nonnegative integers")


def calibration_arms(arms: tuple[str, ...], budget: int, budgets: tuple[int, ...]) -> tuple[str, ...]:
    if budget == 0:
        return tuple(arm for arm in arms if arm == "zero_shot")
    if 0 not in budgets:
        return arms
    return tuple(arm for arm in arms if (arm == "zero_shot") == (budget == 0))


def resolve_calibration_splits(
    budgets: tuple[int, ...],
    builder: Callable[[int, int], BinaryTimeSplit],
    *,
    minimum_test_trials: int,
    available_trials: int,
    separate_test_recordings: bool = False,
) -> tuple[dict[int, BinaryTimeSplit], dict[int, str]]:
    """Cap scarce data at an 80% chronological calibration allocation.

    For an original recording-disjoint Train/Test protocol, available_trials
    counts only the earlier calibration pool. Its later Test pool stays fixed.
    Returned adjustment keys are requested budgets, never mislabeled fit sizes.
    """
    validate_calibration_budgets(budgets)
    if minimum_test_trials < 1:
        raise ValueError("minimum_test_trials must be positive")
    if type(available_trials) is not int or available_trials < 1:
        raise ValueError("available_trials must be a positive integer")
    # Validate chronology independently so fallback cannot hide malformed data.
    baseline = builder(0, 0)
    if not len(baseline.test_rows):
        raise ValueError("no independent test evidence")
    maximum = max(budgets)
    adjustments = {}
    try:
        if not separate_test_recordings and maximum > int(available_trials * LIMITED_CALIBRATION_FRACTION):
            raise ValueError("requested calibration would leave less than about 20% for testing")
        split = builder(maximum, maximum)
        if len(split.test_rows) < minimum_test_trials:
            raise ValueError("too few later test rows after reserving calibration and purge")
    except ValueError as error:
        reason = str(error)
        maximum = min(maximum, int(available_trials * LIMITED_CALIBRATION_FRACTION))
        # If epoch support is unusually long, reduce the cap further to retain
        # test evidence. Never move test rows into calibration after scoring.
        low, high = 0, maximum
        while low < high:
            candidate = (low + high + 1) // 2
            try:
                candidate_split = builder(candidate, candidate)
                feasible = len(candidate_split.test_rows) > 0
            except ValueError:
                feasible = False
            if feasible:
                low = candidate
            else:
                high = candidate - 1
        maximum = low
        adjustments = {
            budget: f"{reason}; limited-data 80% policy: requested={budget}, effective={maximum}"
            for budget in budgets if budget > maximum
        }
    effective_budgets = sorted({budget for budget in budgets if budget <= maximum} | {maximum})
    splits = {budget: builder(budget, maximum) for budget in effective_budgets}
    common = splits[maximum].test_rows
    for split in splits.values():
        np.testing.assert_array_equal(split.test_rows, common)
    return splits, adjustments


def dataset_calibration_splits(dataset, subject: str, protocol: dict, budgets: tuple[int, ...], *, unavailable=None):
    """Resolve authoritative epoch chronology once for public and source selectors."""
    rows = np.flatnonzero(np.asarray(dataset.subject_ids).astype(str) == subject)
    timeline = dataset.event_timeline
    events = np.flatnonzero(timeline.evidence_indices >= 0)
    row_events = np.empty(dataset.n_epochs, dtype=np.int64)
    if not np.array_equal(np.sort(timeline.evidence_indices[events]), np.arange(dataset.n_epochs)):
        raise ValueError("one physical event per epoch is required")
    row_events[timeline.evidence_indices[events]] = events
    ev = row_events[rows]
    onset, available = timeline.onset_times_s[ev], timeline.evidence_available_times_s[ev]
    session, run = timeline.session_ids[ev].astype(str), timeline.run_ids[ev].astype(str)
    if protocol["kind"] == "single_recording_prefix":
        available_trials = len(rows)
        if len(set(zip(session, run, strict=True))) != 1:
            raise ValueError("single-recording protocol cannot concatenate reset clocks")

        def builder(k, maximum):
            return binary_time_split(
                onset, available, calibration_trials=k, maximum_calibration_trials=maximum,
                tmin_s=dataset.preprocessing.tmin_ms / 1000,
            )
    elif protocol["kind"] == "last_session_holdout":
        declared_sessions, declared_runs = protocol["session_order"], protocol["run_order"]
        if not set(session) <= set(declared_sessions) or not set(run) <= set(declared_runs):
            raise ValueError("recording ids are absent from the declared original protocol order")
        pairs = [(s, r) for s in declared_sessions for r in declared_runs
                 if bool(((session == s) & (run == r)).any())]
        if len({s for s, _ in pairs}) < 2 or pairs[-1][0] != protocol["test_session"]:
            raise ValueError("required later test session or earlier calibration is missing")

        def key(pair):
            return json.dumps(pair, separators=(",", ":"))

        order = tuple(key(pair) for pair in pairs)
        tests = tuple(key(pair) for pair in pairs if pair[0] == protocol["test_session"])
        identifiers = np.asarray([key(pair) for pair in zip(session, run, strict=True)])
        available_trials = int((session != protocol["test_session"]).sum())

        def builder(k, maximum):
            return binary_recording_split(
                onset, available, identifiers, recording_order=order, test_recordings=tests,
                calibration_trials=k, maximum_calibration_trials=maximum,
            )
    else:
        raise ValueError("unsupported explicit target protocol")
    splits, missing = resolve_calibration_splits(
        budgets, builder, minimum_test_trials=protocol.get("minimum_test_trials", 200),
        available_trials=available_trials,
        separate_test_recordings=protocol["kind"] == "last_session_holdout",
    )
    if unavailable is not None:
        unavailable.update(missing)
    return rows, splits


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
    if not 0 <= calibration_trials <= maximum_calibration_trials < len(onset):
        raise ValueError("calibration budgets must retain a later test suffix.")
    rows = np.arange(len(onset), dtype=np.int64)
    if maximum_calibration_trials == 0:
        return BinaryTimeSplit(rows[:0], rows, rows[:0])
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
    if not 0 <= calibration_trials <= maximum_calibration_trials <= len(calibration_pool):
        raise ValueError("calibration budget exceeds the declared earlier recordings")
    return BinaryTimeSplit(
        calibration_pool[:calibration_trials], np.concatenate(blocks[first_test:]),
        np.empty(0, dtype=np.int64),
    )


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
