"""Immutable scheduled-event ledger used by cache and evaluation protocols."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

EVENT_TIMELINE_SCHEMA = "n2p3net_scheduled_events/2"
LEGACY_EVENT_TIMELINE_SCHEMAS = frozenset({"n2p3net_scheduled_events/1"})
AVAILABLE = "available"
UNAVAILABLE_EVENT_STATUSES = frozenset(
    {"artifact_rejected", "boundary_dropped", "missing", "acquisition_rejected"}
)
EVENT_STATUSES = frozenset({AVAILABLE, *UNAVAILABLE_EVENT_STATUSES})


@dataclass(frozen=True)
class EncodedCandidateSelection:
    """Dataset-neutral integer view of a string candidate-selection ledger."""

    vocabulary: tuple[str, ...]
    candidate_codes: np.ndarray
    target_codes: np.ndarray
    group_ids: np.ndarray
    repetition_indices: np.ndarray
    truth_by_group: dict[str, int]

    def __post_init__(self) -> None:
        for name in ("candidate_codes", "target_codes", "repetition_indices"):
            value = np.array(getattr(self, name), dtype=np.int64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        groups = np.array(self.group_ids, dtype=str, copy=True)
        groups.setflags(write=False)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "truth_by_group", dict(self.truth_by_group))


def selection_group_id(
    dataset: object,
    subject: object,
    session: object = "",
    run: object = "",
    selection: object = "",
) -> str:
    """Return an unambiguous dataset/subject/recording/selection key."""

    values = [
        "" if value is None else str(value) for value in (dataset, subject, session, run, selection)
    ]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def candidate_repetition_indices(
    candidate_ids: np.ndarray, group_ids: np.ndarray
) -> np.ndarray:
    """Count each candidate's chronological occurrences within each selection group."""

    candidates = np.asarray(candidate_ids).astype(str)
    groups = np.asarray(group_ids).astype(str)
    if candidates.ndim != 1 or groups.shape != candidates.shape:
        raise ValueError("candidate_ids and group_ids must be aligned one-dimensional arrays.")
    if np.any(np.char.strip(candidates) == "") or np.any(np.char.strip(groups) == ""):
        raise ValueError("candidate_ids and group_ids cannot contain empty identifiers.")
    output = np.empty(len(candidates), dtype=np.int64)
    counts: dict[tuple[str, str], int] = {}
    for index, (group, candidate) in enumerate(zip(groups, candidates, strict=True)):
        key = (group, candidate)
        output[index] = counts.get(key, 0)
        counts[key] = int(output[index]) + 1
    return output


@dataclass(frozen=True)
class ScheduledEventTimeline:
    """One row per scheduled stimulus, including rows without model evidence.

    ``evidence_index`` maps an available event to the model-ready epoch tensor and is
    ``-1`` for every unavailable event.  It must be a bijection onto ``range(n_epochs)``.
    """

    event_ids: np.ndarray
    group_ids: np.ndarray
    subject_ids: np.ndarray
    stimulus_ids: np.ndarray
    onset_samples: np.ndarray
    onset_times_s: np.ndarray
    evidence_available_times_s: np.ndarray
    evidence_indices: np.ndarray
    statuses: np.ndarray
    status_details: np.ndarray
    dataset_ids: np.ndarray
    session_ids: np.ndarray
    run_ids: np.ndarray
    selection_ids: np.ndarray
    complete: bool
    online_causal: bool
    timing_source: str
    candidate_ids: np.ndarray | None = None
    target_candidate_ids: np.ndarray | None = None
    repetition_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_events = len(np.asarray(self.event_ids))
        optional_defaults = {
            "candidate_ids": np.repeat("", n_events),
            "target_candidate_ids": np.repeat("", n_events),
            "repetition_indices": np.full(n_events, -1, dtype=np.int64),
        }
        for name, default in optional_defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        raw_arrays = {
            name: np.asarray(getattr(self, name))
            for name in (
                "event_ids",
                "group_ids",
                "subject_ids",
                "stimulus_ids",
                "onset_samples",
                "onset_times_s",
                "evidence_available_times_s",
                "evidence_indices",
                "statuses",
                "status_details",
                "dataset_ids",
                "session_ids",
                "run_ids",
                "selection_ids",
                "candidate_ids",
                "target_candidate_ids",
                "repetition_indices",
            )
        }
        if any(value.ndim != 1 for value in raw_arrays.values()):
            raise ValueError("Every scheduled-event field must be one-dimensional.")
        for name in ("stimulus_ids", "onset_samples", "evidence_indices", "repetition_indices"):
            dtype = raw_arrays[name].dtype
            if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                raise ValueError(f"{name} must have an integer dtype; coercion is forbidden.")
        for name in ("onset_times_s", "evidence_available_times_s"):
            if not np.issubdtype(raw_arrays[name].dtype, np.number):
                raise ValueError(f"{name} must have a numeric dtype.")
        if not isinstance(self.complete, (bool, np.bool_)) or not isinstance(
            self.online_causal, (bool, np.bool_)
        ):
            raise ValueError("complete and online_causal must be strict booleans.")
        if not isinstance(self.timing_source, str):
            raise ValueError("timing_source must be a string.")
        dtypes = {
            "event_ids": str,
            "group_ids": str,
            "subject_ids": str,
            "stimulus_ids": np.int64,
            "onset_samples": np.int64,
            "onset_times_s": np.float64,
            "evidence_available_times_s": np.float64,
            "evidence_indices": np.int64,
            "statuses": str,
            "status_details": str,
            "dataset_ids": str,
            "session_ids": str,
            "run_ids": str,
            "selection_ids": str,
            "candidate_ids": str,
            "target_candidate_ids": str,
            "repetition_indices": np.int64,
        }
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in dtypes.items():
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays[name] = value
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "online_causal", bool(self.online_causal))
        object.__setattr__(self, "timing_source", str(self.timing_source))

        lengths = {name: len(value) for name, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Scheduled-event fields must have equal length, got {lengths}.")
        if not self.timing_source.strip():
            raise ValueError("timing_source must identify how acquisition times were obtained.")

    @property
    def n_events(self) -> int:
        return int(len(self.event_ids))

    @property
    def n_available(self) -> int:
        return int(np.count_nonzero(np.asarray(self.statuses).astype(str) == AVAILABLE))

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(np.unique(np.asarray(self.group_ids).astype(str)).tolist()))

    @property
    def has_candidate_ids(self) -> bool:
        candidates = np.asarray(self.candidate_ids).astype(str)
        return bool(len(candidates) and np.all(np.char.strip(candidates) != ""))

    @property
    def has_candidate_sets(self) -> bool:
        targets = np.asarray(self.target_candidate_ids).astype(str)
        return bool(
            self.has_candidate_ids
            and len(targets)
            and np.all(np.char.strip(targets) != "")
        )

    @property
    def has_repetition_structure(self) -> bool:
        repetitions = np.asarray(self.repetition_indices, dtype=np.int64)
        return bool(self.has_candidate_sets and len(repetitions) and np.all(repetitions >= 0))

    @property
    def supports_full_candidate_chain(self) -> bool:
        """Whether the timeline contains complete, ordered candidate evidence."""

        if not (self.complete and self.has_repetition_structure):
            return False
        candidates = np.asarray(self.candidate_ids).astype(str)
        groups = np.asarray(self.group_ids).astype(str)
        vocabularies = {
            tuple(sorted(np.unique(candidates[groups == group]).tolist()))
            for group in np.unique(groups)
        }
        return len(vocabularies) == 1 and bool(next(iter(vocabularies), ()))

    def encoded_candidate_selection(
        self, *, available_only: bool = True, require_full_chain: bool = False
    ) -> EncodedCandidateSelection:
        """Return stable integer codes without constraining candidate labels to digits."""

        self.validate()
        if not self.has_candidate_sets:
            raise ValueError("Candidate selection needs complete candidate and target metadata.")
        if require_full_chain and not self.supports_full_candidate_chain:
            raise ValueError(
                "The scheduled-event ledger does not satisfy the complete candidate-chain contract."
            )
        candidates = np.asarray(self.candidate_ids).astype(str)
        targets = np.asarray(self.target_candidate_ids).astype(str)
        groups = np.asarray(self.group_ids).astype(str)
        repetitions = np.asarray(self.repetition_indices, dtype=np.int64)
        vocabulary = tuple(sorted(np.unique(candidates).tolist()))
        code_by_candidate = {candidate: index for index, candidate in enumerate(vocabulary)}
        candidate_codes = np.asarray([code_by_candidate[value] for value in candidates])
        target_codes = np.asarray([code_by_candidate[value] for value in targets])
        truth_by_group = {
            str(group): int(target_codes[np.flatnonzero(groups == group)[0]])
            for group in np.unique(groups)
        }
        if available_only:
            evidence = np.asarray(self.evidence_indices, dtype=np.int64)
            available = evidence >= 0
            order = np.argsort(evidence[available], kind="stable")
            rows = np.flatnonzero(available)[order]
        else:
            rows = np.arange(self.n_events)
        return EncodedCandidateSelection(
            vocabulary=vocabulary,
            candidate_codes=candidate_codes[rows],
            target_codes=target_codes[rows],
            group_ids=groups[rows],
            repetition_indices=repetitions[rows],
            truth_by_group=truth_by_group,
        )

    def validate(self, *, n_epochs: int | None = None) -> ScheduledEventTimeline:
        event_ids = np.asarray(self.event_ids).astype(str)
        groups = np.asarray(self.group_ids).astype(str)
        subjects = np.asarray(self.subject_ids).astype(str)
        statuses = np.asarray(self.statuses).astype(str)
        details = np.asarray(self.status_details).astype(str)
        stimulus = np.asarray(self.stimulus_ids)
        samples = np.asarray(self.onset_samples)
        onset = np.asarray(self.onset_times_s, dtype=float)
        available_at = np.asarray(self.evidence_available_times_s, dtype=float)
        evidence = np.asarray(self.evidence_indices)
        candidates = np.asarray(self.candidate_ids).astype(str)
        targets = np.asarray(self.target_candidate_ids).astype(str)
        repetitions = np.asarray(self.repetition_indices)

        if self.n_events == 0:
            raise ValueError("A scheduled-event timeline cannot be empty.")
        all_fields = (
            event_ids,
            groups,
            subjects,
            stimulus,
            samples,
            onset,
            available_at,
            evidence,
            statuses,
            details,
            self.dataset_ids,
            self.session_ids,
            self.run_ids,
            self.selection_ids,
            candidates,
            targets,
            repetitions,
        )
        if any(np.asarray(value).ndim != 1 for value in all_fields):
            raise ValueError("Every scheduled-event field must be one-dimensional.")
        if len(set(event_ids.tolist())) != len(event_ids) or np.any(event_ids == ""):
            raise ValueError("event_ids must be globally unique and non-empty.")
        if np.any(groups == "") or np.any(subjects == ""):
            raise ValueError("Every scheduled event needs non-empty group and subject ids.")
        unknown = set(statuses.tolist()) - EVENT_STATUSES
        if unknown:
            raise ValueError(f"Unknown scheduled-event statuses: {sorted(unknown)}.")
        if not np.issubdtype(stimulus.dtype, np.integer):
            raise ValueError("stimulus_ids must be integers.")
        if not np.issubdtype(samples.dtype, np.integer) or np.any(samples < 0):
            raise ValueError("onset_samples must be non-negative integers.")
        if not np.isfinite(onset).all():
            raise ValueError("onset_times_s must be finite.")
        if not np.issubdtype(evidence.dtype, np.integer):
            raise ValueError("evidence_indices must be integers.")

        candidate_present = np.char.strip(candidates) != ""
        target_present = np.char.strip(targets) != ""
        if candidate_present.any() and not candidate_present.all():
            raise ValueError("candidate_ids must be either absent or present for every event.")
        if target_present.any() and not target_present.all():
            raise ValueError(
                "target_candidate_ids must be either absent or present for every event."
            )
        if target_present.any() and not candidate_present.all():
            raise ValueError("target_candidate_ids cannot be declared without candidate_ids.")
        repetition_present = repetitions >= 0
        if repetition_present.any() and not repetition_present.all():
            raise ValueError(
                "repetition_indices must be either -1 for every event or declared everywhere."
            )
        if np.any(repetitions < -1):
            raise ValueError("repetition_indices must be non-negative or uniformly -1.")
        if repetition_present.any() and not candidate_present.all():
            raise ValueError("repetition_indices cannot be declared without candidate_ids.")

        available = statuses == AVAILABLE
        if np.any(evidence[available] < 0) or np.any(evidence[~available] != -1):
            raise ValueError("Only available events may have non-negative evidence_indices.")
        if np.any(~np.isfinite(available_at[available])) or np.any(
            np.isfinite(available_at[~available])
        ):
            raise ValueError(
                "Evidence availability time must be finite exactly for available events."
            )
        if np.any(available_at[available] < onset[available]):
            raise ValueError("Evidence cannot become available before its scheduled onset.")
        mapped = evidence[available].astype(np.int64)
        expected_epochs = len(mapped) if n_epochs is None else int(n_epochs)
        if sorted(mapped.tolist()) != list(range(expected_epochs)):
            raise ValueError(
                "Available evidence_indices must be a bijection onto the epoch tensor rows."
            )
        if len(mapped) != expected_epochs:
            raise ValueError("Scheduled-event availability does not match the epoch count.")
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            if np.any(np.diff(onset[rows]) < 0.0):
                raise ValueError(f"Events for group {group!r} are not chronological.")
            if len(np.unique(subjects[rows])) != 1:
                raise ValueError(f"Selection group {group!r} spans multiple subjects.")
            for field_name in ("dataset_ids", "session_ids", "run_ids", "selection_ids"):
                values = np.asarray(getattr(self, field_name)).astype(str)[rows]
                if len(np.unique(values)) != 1:
                    raise ValueError(f"Selection group {group!r} spans multiple {field_name}.")
            if target_present.all():
                group_targets = np.unique(targets[rows])
                if len(group_targets) != 1:
                    raise ValueError(
                        f"Selection group {group!r} has mixed target_candidate_ids."
                    )
                group_candidates = np.unique(candidates[rows])
                if len(group_candidates) < 2:
                    raise ValueError(
                        f"Selection group {group!r} needs at least two candidate_ids."
                    )
                if group_targets[0] not in set(group_candidates.tolist()):
                    raise ValueError(
                        f"Selection group {group!r} target is absent from its candidates."
                    )
            if repetition_present.all():
                for candidate in np.unique(candidates[rows]):
                    candidate_rows = rows[candidates[rows] == candidate]
                    candidate_repetitions = repetitions[candidate_rows]
                    if np.any(np.diff(candidate_repetitions) <= 0):
                        raise ValueError(
                            f"Selection group {group!r} candidate {candidate!r} has "
                            "non-increasing repetition_indices."
                        )
                    if self.complete and not np.array_equal(
                        candidate_repetitions,
                        np.arange(len(candidate_repetitions), dtype=np.int64),
                    ):
                        raise ValueError(
                            f"Complete selection group {group!r} candidate {candidate!r} "
                            "must use contiguous repetition_indices starting at zero."
                        )
        return self

    def subset_groups(
        self, groups: set[str] | tuple[str, ...] | list[str]
    ) -> ScheduledEventTimeline:
        selected = {str(group) for group in groups}
        mask = np.isin(np.asarray(self.group_ids).astype(str), tuple(selected))
        if not bool(mask.any()):
            raise ValueError("Scheduled-event subset contains no events.")
        old_evidence = np.asarray(self.evidence_indices, dtype=np.int64)[mask]
        available_old = old_evidence[old_evidence >= 0]
        remap = {int(old): new for new, old in enumerate(sorted(available_old.tolist()))}
        new_evidence = np.asarray(
            [remap.get(int(index), -1) for index in old_evidence], dtype=np.int64
        )
        return self._slice(mask, evidence_indices=new_evidence).validate()

    def with_evidence_offset(self, offset: int) -> ScheduledEventTimeline:
        evidence = np.asarray(self.evidence_indices, dtype=np.int64).copy()
        evidence[evidence >= 0] += int(offset)
        return self._slice(np.ones(self.n_events, dtype=bool), evidence_indices=evidence)

    def _slice(
        self, mask: np.ndarray, *, evidence_indices: np.ndarray | None = None
    ) -> ScheduledEventTimeline:
        mask = np.asarray(mask, dtype=bool)
        return ScheduledEventTimeline(
            event_ids=np.asarray(self.event_ids)[mask].copy(),
            group_ids=np.asarray(self.group_ids)[mask].copy(),
            subject_ids=np.asarray(self.subject_ids)[mask].copy(),
            stimulus_ids=np.asarray(self.stimulus_ids)[mask].copy(),
            onset_samples=np.asarray(self.onset_samples)[mask].copy(),
            onset_times_s=np.asarray(self.onset_times_s)[mask].copy(),
            evidence_available_times_s=np.asarray(self.evidence_available_times_s)[mask].copy(),
            evidence_indices=(
                np.asarray(self.evidence_indices)[mask].copy()
                if evidence_indices is None
                else evidence_indices
            ),
            statuses=np.asarray(self.statuses)[mask].copy(),
            status_details=np.asarray(self.status_details)[mask].copy(),
            dataset_ids=np.asarray(self.dataset_ids)[mask].copy(),
            session_ids=np.asarray(self.session_ids)[mask].copy(),
            run_ids=np.asarray(self.run_ids)[mask].copy(),
            selection_ids=np.asarray(self.selection_ids)[mask].copy(),
            candidate_ids=np.asarray(self.candidate_ids)[mask].copy(),
            target_candidate_ids=np.asarray(self.target_candidate_ids)[mask].copy(),
            repetition_indices=np.asarray(self.repetition_indices)[mask].copy(),
            complete=bool(self.complete),
            online_causal=bool(self.online_causal),
            timing_source=self.timing_source,
        )

    def fingerprint(self, *, truth: dict[str, Any] | None = None) -> str:
        self.validate()
        rows = [
            [
                str(self.event_ids[index]),
                str(self.group_ids[index]),
                str(self.subject_ids[index]),
                int(self.stimulus_ids[index]),
                int(self.onset_samples[index]),
                float(self.onset_times_s[index]),
                float(self.evidence_available_times_s[index])
                if np.isfinite(self.evidence_available_times_s[index])
                else None,
                int(self.evidence_indices[index]),
                str(self.statuses[index]),
                str(self.status_details[index]),
                str(self.dataset_ids[index]),
                str(self.session_ids[index]),
                str(self.run_ids[index]),
                str(self.selection_ids[index]),
                str(self.candidate_ids[index]),
                str(self.target_candidate_ids[index]),
                int(self.repetition_indices[index]),
            ]
            for index in range(self.n_events)
        ]
        payload = {
            "schema": EVENT_TIMELINE_SCHEMA,
            "complete": bool(self.complete),
            "online_causal": bool(self.online_causal),
            "timing_source": self.timing_source,
            "events": rows,
            "truth": sorted((str(key), value) for key, value in (truth or {}).items()),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def observed_only_timeline(
    *,
    dataset_id: str,
    subject_ids: np.ndarray,
    stimulus_ids: np.ndarray,
    onset_times_s: np.ndarray | None = None,
    evidence_available_times_s: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    online_causal: bool = False,
    timing_source: str = "observed_epoch_order_only",
    selection_ids: np.ndarray | None = None,
    session_ids: np.ndarray | None = None,
    run_ids: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
    target_candidate_ids: np.ndarray | None = None,
    repetition_indices: np.ndarray | None = None,
) -> ScheduledEventTimeline:
    """Build an explicitly incomplete ledger when a source lacks its schedule."""

    raw_subjects = np.asarray(subject_ids)
    raw_stimulus = np.asarray(stimulus_ids)
    if raw_subjects.ndim != 1:
        raise ValueError("subject_ids must be one-dimensional.")
    if raw_stimulus.shape != raw_subjects.shape or not np.issubdtype(
        raw_stimulus.dtype, np.integer
    ):
        raise ValueError("stimulus_ids must be a one-dimensional integer array aligned with subjects.")
    subjects = raw_subjects.astype(str)
    n_events = len(subjects)
    groups = subjects if group_ids is None else np.asarray(group_ids).astype(str)
    if groups.shape != (n_events,):
        raise ValueError("group_ids must contain one value per event.")
    onset = (
        np.arange(n_events, dtype=float)
        if onset_times_s is None
        else np.asarray(onset_times_s, dtype=float)
    )
    if onset.shape != (n_events,):
        raise ValueError("onset_times_s must contain one value per event.")
    available_at = (
        onset.copy()
        if evidence_available_times_s is None
        else np.asarray(evidence_available_times_s, dtype=float)
    )
    if available_at.shape != (n_events,) or not np.isfinite(available_at).all():
        raise ValueError("evidence_available_times_s must be finite and align with events.")
    if np.any(available_at < onset):
        raise ValueError("evidence cannot be available before event onset.")
    dataset_ids = np.repeat(str(dataset_id), n_events)
    def _strings_or_default(values: np.ndarray | None, default: np.ndarray) -> np.ndarray:
        output = default if values is None else np.asarray(values).astype(str)
        if output.shape != (n_events,):
            raise ValueError("Observed timeline metadata fields must align with events.")
        return output

    selections = _strings_or_default(selection_ids, groups.copy())
    sessions = _strings_or_default(session_ids, np.repeat("", n_events))
    runs = _strings_or_default(run_ids, np.repeat("", n_events))
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"{dataset_id}:{index}" for index in range(n_events)]),
        group_ids=groups,
        subject_ids=subjects,
        stimulus_ids=raw_stimulus.astype(np.int64, copy=False),
        onset_samples=np.arange(n_events, dtype=np.int64),
        onset_times_s=onset,
        evidence_available_times_s=available_at,
        evidence_indices=np.arange(n_events, dtype=np.int64),
        statuses=np.repeat(AVAILABLE, n_events),
        status_details=np.repeat("source_exposes_observed_epochs_only", n_events),
        dataset_ids=dataset_ids,
        session_ids=sessions,
        run_ids=runs,
        selection_ids=selections,
        complete=False,
        online_causal=bool(online_causal),
        timing_source=timing_source,
        candidate_ids=candidate_ids,
        target_candidate_ids=target_candidate_ids,
        repetition_indices=repetition_indices,
    )
    return timeline.validate(n_epochs=n_events)


def concatenate_event_timelines(
    timelines: list[ScheduledEventTimeline] | tuple[ScheduledEventTimeline, ...],
) -> ScheduledEventTimeline:
    if not timelines:
        raise ValueError("At least one scheduled-event timeline is required.")
    offset = 0
    shifted: list[ScheduledEventTimeline] = []
    for timeline in timelines:
        timeline.validate()
        shifted.append(timeline.with_evidence_offset(offset))
        offset += timeline.n_available
    fields = (
        "event_ids",
        "group_ids",
        "subject_ids",
        "stimulus_ids",
        "onset_samples",
        "onset_times_s",
        "evidence_available_times_s",
        "evidence_indices",
        "statuses",
        "status_details",
        "dataset_ids",
        "session_ids",
        "run_ids",
        "selection_ids",
        "candidate_ids",
        "target_candidate_ids",
        "repetition_indices",
    )
    kwargs = {
        field: np.concatenate([np.asarray(getattr(item, field)) for item in shifted])
        for field in fields
    }
    output = ScheduledEventTimeline(
        **kwargs,
        complete=all(item.complete for item in shifted),
        online_causal=all(item.online_causal for item in shifted),
        timing_source=";".join(dict.fromkeys(item.timing_source for item in shifted)),
    )
    return output.validate(n_epochs=offset)
