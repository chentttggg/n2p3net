"""BrainSync session semantics layered on the generic BIDS-EEG raw contract."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from data.bids_eeg import (
    BidsEEGInputContract,
    ValidatedBidsEEGRecording,
    validate_bids_eeg_recording,
)

BRAIN_SYNC_SESSION_SCHEMA = "brainsync-gtn-session/3"
BRAIN_SYNC_CHANNEL_COUNT = 8
BRAIN_SYNC_RAW_STAGE = "bids_raw"
BRAIN_SYNC_RAW_TIME_BASE = "continuous_recording"
BRAIN_SYNC_PREPROCESSING_STATUS = "pending"
BRAIN_SYNC_INPUT_STATUSES = frozenset({"completed", "aborted"})
DEFAULT_ADULT_MIN_AGE_YEARS = 18.0
BRAIN_SYNC_BIDS_CONTRACT = BidsEEGInputContract(
    minimum_bids_version=(1, 11, 0),
    task_name="gtn",
    stimulus_trial_type="stimulus",
    rest_trial_type="rest",
    candidate_column="digit",
    candidate_vocabulary=tuple(str(value) for value in range(1, 10)),
    coordinate_units="m",
)


class DecisionTargetPolicy(StrEnum):
    """Target-sequence estimands supported by chronological decision splitting."""

    OBSERVED_SEQUENCE = "observed_sequence"
    FORCED_SWITCH = "forced_switch"
    UNSEEN_CALIBRATION_CODES = "unseen_calibration_codes"

    @property
    def evidence_claim(self) -> str:
        return {
            self.OBSERVED_SEQUENCE: "later_session_observed_target_sequence",
            self.FORCED_SWITCH: "later_session_target_switch",
            self.UNSEEN_CALIBRATION_CODES: "later_session_target_unseen_in_calibration",
        }[self]


class PopulationScopePolicy(StrEnum):
    """Controls whether age metadata may support an explicit adult claim."""

    DESCRIPTIVE = "descriptive"
    ADULT_ONLY = "adult_only"


@dataclass(frozen=True)
class ValidatedBrainSyncSession:
    """A completed v3 session whose BIDS raw boundary passed validation."""

    root: Path
    session_path: Path
    manifest: dict[str, Any]
    session_id: str
    subject_id: str
    thought_digit: int
    started_utc: str
    started_timestamp_s: float
    ended_utc: str
    ended_timestamp_s: float
    target_confirmed_utc: str
    target_confirmed_timestamp_s: float
    age_years: float | None
    sex: Any
    bids: ValidatedBidsEEGRecording


@dataclass(frozen=True)
class PopulationScope:
    policy: str
    label: str
    adult_min_age_years: float
    age_source: str
    subject_count: int
    session_count: int
    sessions_with_age: int
    age_min_years: float | None
    age_max_years: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrainSyncEvidenceScope:
    dataset: str
    session_schema: str
    decision_unit: str
    target_policy: str
    decision_claim: str
    test_target_role: str
    population: PopulationScope

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["population"] = self.population.to_dict()
        return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"BrainSync session is missing {path.name}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"BrainSync session file {path} is not valid readable JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BrainSync session file {path} must contain a JSON object.")
    return payload


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BrainSync {field_name} must be a non-empty string.")
    return value.strip()


def _digit(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 9:
        raise ValueError(f"BrainSync {field_name} must be an integer in the 1-9 vocabulary.")
    return value


def _finite_nonnegative(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"BrainSync {field_name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BrainSync {field_name} must be a finite non-negative number.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"BrainSync {field_name} must be a finite non-negative number.")
    return result


def _parse_timestamp(value: Any, field_name: str) -> tuple[str, float]:
    text = _nonempty_string(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"BrainSync {field_name} must be an ISO timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"BrainSync {field_name} must include a timezone.")
    return text, float(parsed.timestamp())


def _optional_age(value: Any) -> float | None:
    if value is None or value == "":
        return None
    age = _finite_nonnegative(value, "experiment.age")
    if age == 0.0:
        raise ValueError("BrainSync experiment.age must be positive when declared.")
    return age


def _validate_manifest_rest_intervals(
    timeline: dict[str, Any], bids: ValidatedBidsEEGRecording
) -> None:
    declared = timeline.get("rest_segments")
    if not isinstance(declared, list):
        raise ValueError("BrainSync recording.timeline.rest_segments must be an array.")
    expected = tuple(
        (interval.onset_seconds, interval.end_seconds) for interval in bids.rest_intervals
    )
    observed: list[tuple[float, float]] = []
    for index, item in enumerate(declared):
        if not isinstance(item, dict):
            raise ValueError("BrainSync timeline rest segment must be an object.")
        start = _finite_nonnegative(item.get("start_seconds"), f"rest_segments[{index}].start_seconds")
        end = _finite_nonnegative(item.get("end_seconds"), f"rest_segments[{index}].end_seconds")
        duration = _finite_nonnegative(
            item.get("duration_seconds"), f"rest_segments[{index}].duration_seconds"
        )
        if end <= start or not math.isclose(end - start, duration, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("BrainSync timeline rest segment bounds and duration conflict.")
        observed.append((start, end))
    if len(observed) != len(expected) or any(
        not math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(left[1], right[1], rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(observed, expected, strict=True)
    ):
        raise ValueError("BrainSync timeline rest segments conflict with BIDS events.tsv.")


def validate_brainsync_bids_session(session_dir: str | Path) -> ValidatedBrainSyncSession:
    """Validate session and BIDS metadata before an EEG binary reader is invoked."""

    root = Path(session_dir).expanduser().resolve()
    session_path = root / "session.json"
    manifest = _load_json_object(session_path)
    if manifest.get("schema") != BRAIN_SYNC_SESSION_SCHEMA:
        raise ValueError(
            f"Unsupported BrainSync session schema: {manifest.get('schema')!r}; "
            f"expected {BRAIN_SYNC_SESSION_SCHEMA!r}."
        )
    status = manifest.get("status")
    if status not in BRAIN_SYNC_INPUT_STATUSES:
        raise ValueError(f"BrainSync session status is not a model input: {status!r}.")

    target_label = manifest.get("target_label")
    if not isinstance(target_label, dict) or target_label.get("status") != "confirmed":
        raise ValueError("BrainSync session target_label must be post-experiment confirmed.")
    if target_label.get("source") != "post_experiment_confirmation":
        raise ValueError("BrainSync target_label source must be post_experiment_confirmation.")
    thought_digit = _digit(target_label.get("thought_digit"), "target_label.thought_digit")

    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        raise ValueError("BrainSync recording must be an object.")
    if recording.get("stage") != BRAIN_SYNC_RAW_STAGE:
        raise ValueError("BrainSync recording stage must be bids_raw.")
    if recording.get("preprocessing_status") != BRAIN_SYNC_PREPROCESSING_STATUS:
        raise ValueError("BrainSync BIDS raw preprocessing_status must be pending.")
    timeline = recording.get("timeline")
    if not isinstance(timeline, dict) or timeline.get("status") != "finalized":
        raise ValueError("BrainSync BIDS raw timeline must be finalized.")
    if timeline.get("time_base") != BRAIN_SYNC_RAW_TIME_BASE:
        raise ValueError("BrainSync BIDS raw timeline must use continuous_recording time.")
    if timeline.get("rest_segments_retained") is not True:
        raise ValueError("BrainSync BIDS raw must retain rest segments.")
    bids_files = recording.get("bids")
    if not isinstance(bids_files, dict):
        raise ValueError("BrainSync recording.bids must be a file manifest.")

    quality = manifest.get("quality")
    continuity = quality.get("eeg_continuity") if isinstance(quality, dict) else None
    if not isinstance(continuity, dict) or continuity.get("passed") is not True:
        raise ValueError("BrainSync EEG continuity quality must pass before preprocessing.")

    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("BrainSync session.experiment must be an object.")
    subject_id = _nonempty_string(experiment.get("subject_id"), "experiment.subject_id")
    if _digit(experiment.get("thought_digit"), "experiment.thought_digit") != thought_digit:
        raise ValueError("BrainSync experiment and confirmed target labels conflict.")
    if experiment.get("target_label_status") != "confirmed_post_experiment":
        raise ValueError("BrainSync experiment.target_label_status must be confirmed_post_experiment.")

    session_id = _nonempty_string(manifest.get("session_id"), "session_id")
    started_utc, started_timestamp_s = _parse_timestamp(manifest.get("started_utc"), "started_utc")
    target_confirmed_utc, target_confirmed_timestamp_s = _parse_timestamp(
        target_label.get("confirmed_utc"), "target_label.confirmed_utc"
    )
    ended_utc, ended_timestamp_s = _parse_timestamp(manifest.get("ended_utc"), "ended_utc")
    if not started_timestamp_s <= target_confirmed_timestamp_s <= ended_timestamp_s:
        raise ValueError("BrainSync timestamps must satisfy start <= confirmation <= end.")

    bids = validate_bids_eeg_recording(root, bids_files, contract=BRAIN_SYNC_BIDS_CONTRACT)
    recording_path = recording.get("path")
    if not isinstance(recording_path, str) or (root / recording_path).resolve() != bids.raw_path:
        raise ValueError("BrainSync recording.path must identify the BIDS EEG recording.")
    if len(bids.channel_names) != BRAIN_SYNC_CHANNEL_COUNT:
        raise ValueError(f"BrainSync BIDS raw must declare {BRAIN_SYNC_CHANNEL_COUNT} EEG channels.")
    recording_rate = _finite_nonnegative(
        recording.get("source_sample_rate_hz"), "recording.source_sample_rate_hz"
    )
    timeline_rate = _finite_nonnegative(
        timeline.get("source_sample_rate_hz"), "recording.timeline.source_sample_rate_hz"
    )
    timeline_duration = _finite_nonnegative(
        timeline.get("output_duration_seconds"), "recording.timeline.output_duration_seconds"
    )
    if not (
        math.isclose(recording_rate, bids.sample_rate_hz, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(timeline_rate, bids.sample_rate_hz, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("BrainSync manifest and BIDS sampling frequencies conflict.")
    if not math.isclose(timeline_duration, bids.duration_seconds, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("BrainSync timeline and BIDS recording durations conflict.")
    _validate_manifest_rest_intervals(timeline, bids)
    if bids.stimuli[-1].onset_seconds > target_confirmed_timestamp_s - started_timestamp_s:
        raise ValueError("BrainSync target confirmation precedes a BIDS stimulus onset.")
    if bids.duration_seconds > ended_timestamp_s - started_timestamp_s:
        raise ValueError("BrainSync BIDS recording duration exceeds the session lifetime.")

    return ValidatedBrainSyncSession(
        root=root,
        session_path=session_path,
        manifest=manifest,
        session_id=session_id,
        subject_id=subject_id,
        thought_digit=thought_digit,
        started_utc=started_utc,
        started_timestamp_s=started_timestamp_s,
        ended_utc=ended_utc,
        ended_timestamp_s=ended_timestamp_s,
        target_confirmed_utc=target_confirmed_utc,
        target_confirmed_timestamp_s=target_confirmed_timestamp_s,
        age_years=_optional_age(experiment.get("age")),
        sex=experiment.get("sex"),
        bids=bids,
    )


def derive_population_scope(
    subject_ids: Sequence[object],
    age_years: Sequence[object],
    *,
    policy: PopulationScopePolicy | str,
    adult_min_age_years: float = DEFAULT_ADULT_MIN_AGE_YEARS,
) -> PopulationScope:
    """Derive a conservative population label from session-level age evidence."""

    try:
        resolved_policy = PopulationScopePolicy(policy)
    except ValueError as exc:
        raise ValueError(f"Unsupported population scope policy: {policy!r}.") from exc
    if len(subject_ids) != len(age_years) or not subject_ids:
        raise ValueError("population scope requires aligned, non-empty subject and age rows.")
    threshold = float(adult_min_age_years)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("adult_min_age_years must be finite and positive.")
    subjects = {_nonempty_string(value, "population.subject_id") for value in subject_ids}
    known_ages: list[float] = []
    missing = 0
    for value in age_years:
        if value is None or (isinstance(value, str) and not value.strip()):
            missing += 1
            continue
        try:
            numeric_age = float(value)
        except (TypeError, ValueError):
            numeric_age = math.inf
        if math.isnan(numeric_age):
            missing += 1
            continue
        known_ages.append(_finite_nonnegative(value, "population.age_years"))
    if resolved_policy is PopulationScopePolicy.ADULT_ONLY:
        if missing:
            raise ValueError("adult_only evidence requires age for every BrainSync session.")
        if any(age < threshold for age in known_ages):
            raise ValueError("adult_only evidence contains an age below adult_min_age_years.")
        label = "adult"
    else:
        label = "age_descriptive"
    return PopulationScope(
        policy=resolved_policy.value,
        label=label,
        adult_min_age_years=threshold,
        age_source="session.experiment.age",
        subject_count=len(subjects),
        session_count=len(subject_ids),
        sessions_with_age=len(known_ages),
        age_min_years=min(known_ages) if known_ages else None,
        age_max_years=max(known_ages) if known_ages else None,
    )


def derive_brainsync_evidence_scope(
    population: PopulationScope,
    *,
    target_policy: DecisionTargetPolicy | str,
) -> BrainSyncEvidenceScope:
    try:
        resolved_target_policy = DecisionTargetPolicy(target_policy)
    except ValueError as exc:
        raise ValueError(f"Unsupported decision target policy: {target_policy!r}.") from exc
    return BrainSyncEvidenceScope(
        dataset="BrainSync-GTN",
        session_schema=BRAIN_SYNC_SESSION_SCHEMA,
        decision_unit="session",
        target_policy=resolved_target_policy.value,
        decision_claim=resolved_target_policy.evidence_claim,
        test_target_role="eligibility_and_scoring_only",
        population=population,
    )


def session_population_rows(
    sessions: Iterable[ValidatedBrainSyncSession],
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    values = tuple(sessions)
    return (
        tuple(session.subject_id for session in values),
        tuple(session.age_years for session in values),
    )
