"""Strict, dependency-light contracts for BrainSync GTN session v2 artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

BRAIN_SYNC_SESSION_SCHEMA = "brainsync-gtn-session/2"
BRAIN_SYNC_MONTAGE_SCHEMA = "brainsync-channel-montage/2"
BRAIN_SYNC_CHANNEL_COUNT = 8
BRAIN_SYNC_ANALYSIS_TIME_BASE = "rest_removed_recording"
BRAIN_SYNC_ANALYSIS_READY_STATUSES = frozenset({"completed", "aborted"})
BRAIN_SYNC_MARKER_EVENT = "recording_marker"
BRAIN_SYNC_ONSET_KIND = "onset"
DEFAULT_ADULT_MIN_AGE_YEARS = 18.0

_RECORDING_MARKER_KINDS = frozenset({"onset", "offset", "rest_start", "rest_end"})
_FORBIDDEN_V2_MARKER_FIELDS = frozenset(
    {
        "decision_id",
        "is_target",
        "repetition_index",
        "selection_id",
        "target_digit",
    }
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
class BrainSyncOnsetMarker:
    trial_id: str
    block_id: int
    trial_index: int
    digit: int
    eeg_time_seconds: float
    line_number: int


@dataclass(frozen=True)
class ValidatedBrainSyncSession:
    """A v2 session that passed every pre-raw-read analysis-ready gate."""

    root: Path
    session_path: Path
    recording_path: Path
    events_path: Path
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
    source_sample_rate_hz: float
    output_duration_seconds: float
    age_years: float | None
    sex: Any
    onset_markers: tuple[BrainSyncOnsetMarker, ...]


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
    except json.JSONDecodeError as exc:
        raise ValueError(f"BrainSync session file {path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BrainSync session file {path} must contain a JSON object.")
    return payload


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BrainSync {field_name} must be a non-empty string.")
    return value.strip()


def _integer(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"BrainSync {field_name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"BrainSync {field_name} must be at least {minimum}.")
    return value


def _digit(value: Any, field_name: str) -> int:
    digit = _integer(value, field_name)
    if not 1 <= digit <= 9:
        raise ValueError(f"BrainSync {field_name} must be in the 1-9 vocabulary.")
    return digit


def _finite_nonnegative(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"BrainSync {field_name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"BrainSync {field_name} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"BrainSync {field_name} must be a finite non-negative number.")
    return result


def _finite_positive(value: Any, field_name: str) -> float:
    result = _finite_nonnegative(value, field_name)
    if result <= 0.0:
        raise ValueError(f"BrainSync {field_name} must be positive.")
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


def _contained_file(root: Path, value: Any, field_name: str) -> Path:
    text = _nonempty_string(value, field_name)
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"BrainSync {field_name} must be relative to the session directory.")
    if ".." in relative.parts:
        raise ValueError(f"BrainSync {field_name} cannot contain parent traversal.")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"BrainSync {field_name} cannot escape the session directory.") from exc
    if not candidate.is_file():
        raise ValueError(f"BrainSync {field_name} does not exist: {text}")
    return candidate


def _read_v2_onset_markers(path: Path) -> tuple[BrainSyncOnsetMarker, ...]:
    markers: list[BrainSyncOnsetMarker] = []
    trial_ids: set[str] = set()
    scheduled_ids: set[tuple[int, int]] = set()
    previous_recording_time = -math.inf
    previous_schedule: tuple[int, int] | None = None
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot open BrainSync event file {path}.") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid BrainSync event JSON at line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"BrainSync event line {line_number} must be an object.")
            if record.get("event") != BRAIN_SYNC_MARKER_EVENT:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(
                    f"BrainSync recording marker at line {line_number} needs an object payload."
                )
            kind = payload.get("kind")
            if kind not in _RECORDING_MARKER_KINDS:
                raise ValueError(
                    f"BrainSync recording marker at line {line_number} has unsupported kind {kind!r}."
                )
            leaked = sorted(_FORBIDDEN_V2_MARKER_FIELDS.intersection(payload))
            if leaked:
                raise ValueError(
                    "BrainSync v2 recording markers cannot contain derived decision/label fields: "
                    + ", ".join(leaked)
                    + "."
                )
            if payload.get("eeg_time_base") != BRAIN_SYNC_ANALYSIS_TIME_BASE:
                raise ValueError(
                    f"BrainSync recording marker at line {line_number} must use "
                    f"eeg_time_base={BRAIN_SYNC_ANALYSIS_TIME_BASE!r}."
                )
            marker_time = _finite_nonnegative(
                payload.get("eeg_time_seconds"), "recording_marker.eeg_time_seconds"
            )
            if marker_time < previous_recording_time:
                raise ValueError("BrainSync recording markers must be chronological in EEG time.")
            previous_recording_time = marker_time
            if kind != BRAIN_SYNC_ONSET_KIND:
                continue

            trial_id = _nonempty_string(payload.get("trial_id"), "recording_marker.trial_id")
            block_id = _integer(
                payload.get("block_id"), "recording_marker.block_id", minimum=1
            )
            trial_index = _integer(
                payload.get("trial_index"), "recording_marker.trial_index", minimum=1
            )
            digit = _digit(payload.get("digit"), "recording_marker.digit")
            schedule_id = (block_id, trial_index)
            if trial_id in trial_ids:
                raise ValueError(f"Duplicate BrainSync onset trial_id: {trial_id!r}.")
            if schedule_id in scheduled_ids:
                raise ValueError(
                    "Duplicate BrainSync onset (block_id, trial_index): "
                    f"{schedule_id!r}."
                )
            if previous_schedule is not None and schedule_id <= previous_schedule:
                raise ValueError(
                    "BrainSync onset block_id/trial_index values must follow schedule order."
                )
            if markers and marker_time <= markers[-1].eeg_time_seconds:
                raise ValueError("BrainSync onset EEG times must be strictly increasing.")
            trial_ids.add(trial_id)
            scheduled_ids.add(schedule_id)
            previous_schedule = schedule_id
            markers.append(
                BrainSyncOnsetMarker(
                    trial_id=trial_id,
                    block_id=block_id,
                    trial_index=trial_index,
                    digit=digit,
                    eeg_time_seconds=marker_time,
                    line_number=line_number,
                )
            )
    if not markers:
        raise ValueError(f"BrainSync event file {path} contains no onset recording markers.")
    return tuple(markers)


def validate_analysis_ready_brainsync_session(
    session_dir: str | Path,
) -> ValidatedBrainSyncSession:
    """Validate all v2 manifest and marker gates before an EEG reader is invoked."""

    root = Path(session_dir).expanduser().resolve()
    session_path = root / "session.json"
    manifest = _load_json_object(session_path)
    schema = manifest.get("schema")
    if schema != BRAIN_SYNC_SESSION_SCHEMA:
        raise ValueError(
            f"Unsupported BrainSync session schema: {schema!r}; expected "
            f"{BRAIN_SYNC_SESSION_SCHEMA!r}."
        )
    status = manifest.get("status")
    if status not in BRAIN_SYNC_ANALYSIS_READY_STATUSES:
        raise ValueError(f"BrainSync session status is not analysis-ready: {status!r}.")

    target_label = manifest.get("target_label")
    if not isinstance(target_label, dict) or target_label.get("status") != "confirmed":
        raise ValueError("BrainSync session target_label must be post-experiment confirmed.")
    if target_label.get("source") != "post_experiment_confirmation":
        raise ValueError(
            "BrainSync confirmed target_label must declare post_experiment_confirmation."
        )
    thought_digit = _digit(target_label.get("thought_digit"), "target_label.thought_digit")

    recording = manifest.get("recording")
    if not isinstance(recording, dict) or recording.get("analysis_ready") is not True:
        raise ValueError("BrainSync recording has not passed the analysis-ready gate.")
    timeline = recording.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("BrainSync recording.timeline must be an object.")
    if timeline.get("status") != "finalized":
        raise ValueError("BrainSync recording timeline must be finalized.")
    if timeline.get("time_base") != BRAIN_SYNC_ANALYSIS_TIME_BASE:
        raise ValueError(
            "BrainSync recording timeline must use the rest-removed EEG time base."
        )
    recording_rate = _finite_positive(
        recording.get("source_sample_rate_hz"), "recording.source_sample_rate_hz"
    )
    timeline_rate = _finite_positive(
        timeline.get("source_sample_rate_hz"),
        "recording.timeline.source_sample_rate_hz",
    )
    if not math.isclose(recording_rate, timeline_rate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("BrainSync recording and finalized timeline sample rates conflict.")
    output_duration_seconds = _finite_positive(
        timeline.get("output_duration_seconds"),
        "recording.timeline.output_duration_seconds",
    )

    quality = manifest.get("quality")
    continuity = quality.get("eeg_continuity") if isinstance(quality, dict) else None
    if not isinstance(continuity, dict) or continuity.get("passed") is not True:
        raise ValueError("BrainSync EEG continuity quality must pass before analysis.")

    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("BrainSync session.experiment must be an object.")
    subject_id = _nonempty_string(experiment.get("subject_id"), "experiment.subject_id")
    experiment_target = _digit(experiment.get("thought_digit"), "experiment.thought_digit")
    if experiment_target != thought_digit:
        raise ValueError("BrainSync experiment and confirmed target labels conflict.")
    if experiment.get("target_label_status") != "confirmed_post_experiment":
        raise ValueError(
            "BrainSync experiment.target_label_status must be "
            "'confirmed_post_experiment'."
        )

    session_id = _nonempty_string(manifest.get("session_id"), "session_id")
    started_utc, started_timestamp_s = _parse_timestamp(
        manifest.get("started_utc"), "started_utc"
    )
    target_confirmed_utc, target_confirmed_timestamp_s = _parse_timestamp(
        target_label.get("confirmed_utc"), "target_label.confirmed_utc"
    )
    ended_utc, ended_timestamp_s = _parse_timestamp(
        manifest.get("ended_utc"), "ended_utc"
    )
    if not started_timestamp_s <= target_confirmed_timestamp_s <= ended_timestamp_s:
        raise ValueError(
            "BrainSync timestamps must satisfy started_utc <= target confirmation <= ended_utc."
        )
    recording_path = _contained_file(root, recording.get("path"), "recording.path")
    timeline_output_path = _contained_file(
        root,
        timeline.get("output_path"),
        "recording.timeline.output_path",
    )
    if recording_path != timeline_output_path:
        raise ValueError(
            "BrainSync recording.path must identify the finalized timeline.output_path."
        )
    events_path = _contained_file(
        root, "events/events.jsonl", "events/events.jsonl"
    )
    onset_markers = _read_v2_onset_markers(events_path)
    last_onset_seconds = onset_markers[-1].eeg_time_seconds
    if last_onset_seconds > output_duration_seconds:
        raise ValueError(
            "BrainSync onset marker exceeds the finalized output duration."
        )
    if last_onset_seconds > target_confirmed_timestamp_s - started_timestamp_s:
        raise ValueError(
            "BrainSync target confirmation precedes a recorded stimulus onset."
        )
    if output_duration_seconds > ended_timestamp_s - started_timestamp_s:
        raise ValueError(
            "BrainSync finalized recording duration exceeds the session lifetime."
        )
    return ValidatedBrainSyncSession(
        root=root,
        session_path=session_path,
        recording_path=recording_path,
        events_path=events_path,
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
        source_sample_rate_hz=timeline_rate,
        output_duration_seconds=output_duration_seconds,
        age_years=_optional_age(experiment.get("age")),
        sex=experiment.get("sex"),
        onset_markers=onset_markers,
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
        age = _finite_nonnegative(value, "population.age_years")
        known_ages.append(age)
    if resolved_policy is PopulationScopePolicy.ADULT_ONLY:
        if missing:
            raise ValueError("adult_only evidence requires age for every BrainSync session.")
        below = [age for age in known_ages if age < threshold]
        if below:
            raise ValueError(
                "adult_only evidence contains an age below adult_min_age_years."
            )
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
