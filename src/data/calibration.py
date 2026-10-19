"""Pure chronological calibration scopes and requested/effective test budgets.

Both data preparation and transfer learners consume this shared data contract.
The historical transfer.binary names remain aliases of these same objects.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import numpy as np

STANDARD_CALIBRATION_TRIALS = (0, 200, 800, 2000)

LIMITED_CALIBRATION_FRACTION = 0.8

TEST_SIZE_POLICIES = frozenset({"limited_data_80_percent", "strict_minimum"})

@dataclass(frozen=True)
class TestBudget:
    """Resolved test-size policy, including an explicit limited-data exception."""

    policy: str
    requested_minimum_test_trials: int
    planned_test_trials: int
    actual_test_trials: int
    separate_test_recordings: bool

    def __post_init__(self) -> None:
        if self.policy not in TEST_SIZE_POLICIES:
            raise ValueError("unknown test-size policy")
        for field in ("requested_minimum_test_trials", "planned_test_trials", "actual_test_trials"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.actual_test_trials > self.planned_test_trials:
            raise ValueError("actual test rows exceed the resolved test scope")
        if type(self.separate_test_recordings) is not bool:
            raise ValueError("separate_test_recordings must be boolean")
        if self.policy == "strict_minimum" and not self.minimum_satisfied:
            raise ValueError("strict test minimum is not satisfied")

    @property
    def minimum_satisfied(self) -> bool:
        return self.actual_test_trials >= self.requested_minimum_test_trials

    def with_actual_count(self, count: int) -> TestBudget:
        if type(count) is not int:
            raise ValueError("actual test count must be an integer")
        return replace(self, actual_test_trials=count)

    def record(self) -> dict:
        return {
            "schema": "n2p3_test_budget/1", "policy": self.policy,
            "requested_minimum_test_trials": self.requested_minimum_test_trials,
            "planned_test_trials": self.planned_test_trials,
            "actual_test_trials": self.actual_test_trials,
            "separate_test_recordings": self.separate_test_recordings,
            "minimum_satisfied": self.minimum_satisfied,
            "relaxation_reason": (None if self.minimum_satisfied else
                "explicit_limited_data_policy_preserves_nonempty_original_test_scope"),
            "removed_after_scope_resolution": self.planned_test_trials - self.actual_test_trials,
        }

    @classmethod
    def from_record(cls, record: dict, *, actual_test_trials: int | None = None) -> TestBudget:
        if not isinstance(record, dict):
            raise ValueError("resolved test budget is required")
        try:
            value = cls(**{field.name: record[field.name] for field in fields(cls)})
        except (KeyError, TypeError) as error:
            raise ValueError("resolved test budget fields are invalid") from error
        if value.record() != record:
            raise ValueError("resolved test budget record is inconsistent")
        if actual_test_trials is not None and value.actual_test_trials != actual_test_trials:
            raise ValueError("resolved test budget differs from actual retained test rows")
        return value

def validate_calibration_budgets(budgets: tuple[int, ...]) -> None:
    if (not budgets or any(type(value) is not int or value < 0 for value in budgets)
            or tuple(sorted(set(budgets))) != budgets):
        raise ValueError("calibration budgets must be unique increasing nonnegative integers")

def calibration_budget_requests(requested_budgets, effective_budgets, adjustments):
    """Bind requested labels to the already resolved chronological fit scopes."""
    requested, effective = tuple(requested_budgets), tuple(effective_budgets)
    validate_calibration_budgets(requested)
    validate_calibration_budgets(effective)
    maximum = max(effective)
    expected = {budget for budget in requested if budget > maximum}
    if set(adjustments) != expected or any(not isinstance(reason, str) or not reason for reason in adjustments.values()):
        raise ValueError("calibration request adjustments differ from the resolved budget cap")
    mapping = {budget: {"requested_K": budget, "actual_K": min(budget, maximum),
        "adjustment": adjustments.get(budget)} for budget in requested}
    if {row["actual_K"] for row in mapping.values()} != set(effective):
        raise ValueError("calibration request mapping differs from the prepared fit scopes")
    return mapping

def resolve_calibration_splits(
    budgets: tuple[int, ...],
    builder: Callable[[int, int], BinaryTimeSplit],
    *,
    minimum_test_trials: int,
    available_trials: int,
    separate_test_recordings: bool = False,
    test_size_policy: str = "limited_data_80_percent",
) -> tuple[dict[int, BinaryTimeSplit], dict[int, str]]:
    """Cap scarce data at an 80% chronological calibration allocation.

    For an original recording-disjoint Train/Test protocol, available_trials
    counts only the earlier calibration pool. Its later Test pool stays fixed.
    Returned adjustment keys are requested budgets, never mislabeled fit sizes.
    """
    validate_calibration_budgets(budgets)
    if type(minimum_test_trials) is not int or minimum_test_trials < 1:
        raise ValueError("minimum_test_trials must be positive")
    if test_size_policy not in TEST_SIZE_POLICIES:
        raise ValueError("unknown test-size policy")
    if type(available_trials) is not int or available_trials < 1:
        raise ValueError("available_trials must be a positive integer")
    # Validate chronology independently so fallback cannot hide malformed data.
    baseline = builder(0, 0)
    if not len(baseline.test_rows):
        raise ValueError("no independent test evidence")
    strict = test_size_policy == "strict_minimum"
    if strict and len(baseline.test_rows) < minimum_test_trials:
        raise ValueError("even zero calibration cannot satisfy the strict test minimum")
    maximum = max(budgets)
    adjustments = {}
    try:
        if (not strict and not separate_test_recordings
                and maximum > int(available_trials * LIMITED_CALIBRATION_FRACTION)):
            raise ValueError("requested calibration would leave less than about 20% for testing")
        split = builder(maximum, maximum)
        if len(split.test_rows) < (minimum_test_trials if strict else 1):
            raise ValueError("too few later test rows after reserving calibration and purge")
    except ValueError as error:
        reason = str(error)
        maximum = min(maximum, available_trials if strict else
                      int(available_trials * LIMITED_CALIBRATION_FRACTION))
        # If epoch support is unusually long, reduce the cap further to retain
        # test evidence. Never move test rows into calibration after scoring.
        low, high = 0, maximum
        while low < high:
            candidate = (low + high + 1) // 2
            try:
                candidate_split = builder(candidate, candidate)
                feasible = len(candidate_split.test_rows) >= (minimum_test_trials if strict else 1)
            except ValueError:
                feasible = False
            if feasible:
                low = candidate
            else:
                high = candidate - 1
        maximum = low
        adjustments = {
            budget: f"{reason}; {test_size_policy}: requested={budget}, effective={maximum}"
            for budget in budgets if budget > maximum
        }
    effective_budgets = sorted({budget for budget in budgets if budget <= maximum} | {maximum})
    splits = {budget: builder(budget, maximum) for budget in effective_budgets}
    common = splits[maximum].test_rows
    budget = TestBudget(test_size_policy, minimum_test_trials, len(common), len(common),
                        separate_test_recordings)
    for split in splits.values():
        np.testing.assert_array_equal(split.test_rows, common)
    return {key: replace(split, test_budget=budget) for key, split in splits.items()}, adjustments

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
        test_size_policy=protocol.get("test_size_policy", "limited_data_80_percent"),
    )
    if unavailable is not None:
        unavailable.update(missing)
    return rows, splits

@dataclass(frozen=True)
class BinaryTimeSplit:
    calibration_rows: np.ndarray
    test_rows: np.ndarray
    purged_rows: np.ndarray
    test_budget: TestBudget | None = None

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
