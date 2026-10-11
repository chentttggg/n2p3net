"""Project-wide participant exclusions, applied to acquisition and source identity."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import unquote, urlsplit

from data.identity import DatasetIdentityTable

PARTICIPANT_EXCLUSION_POLICY = "brain_invaders_author_exclusions_20260907"
PARTICIPANT_EXCLUSION_SOURCE = "https://hal.science/hal-02171575/document#page=4"
BI2014A_EXCLUDED_SUBJECTS = frozenset({62, 63, 64})
_BI2014A_SOURCE_NAMES = frozenset({"BI2014a", "BrainInvaders2014a"})
BI2015A_EXCLUDED_SUBJECTS = frozenset({1, 27, 44, 45, 46, 47, 48, 49, 50})
BI2015A_EXCLUSION_SOURCE = "https://hal.science/hal-02172347/document#page=5"
_BI2015A_SOURCE_NAMES = frozenset({"BI2015a", "BrainInvaders2015a"})


def source_participant_excluded(dataset_id: str, subject_id: object) -> bool:
    source = str(dataset_id)
    if source in _BI2014A_SOURCE_NAMES or source.startswith("BI2014a-candidate:"):
        excluded = BI2014A_EXCLUDED_SUBJECTS
    elif source in _BI2015A_SOURCE_NAMES or source.startswith("BI2015a-candidate:"):
        excluded = BI2015A_EXCLUDED_SUBJECTS
    else:
        return False
    subject = str(subject_id).removeprefix("subject_")
    return subject.isdecimal() and int(subject) in excluded


def excluded_identity_subjects(table: DatasetIdentityTable) -> tuple[str, ...]:
    excluded = []
    for record in table.records:
        for origin in record.origin_subject_keys:
            parsed = urlsplit(origin)
            parts = parsed.path.split("/")
            if parsed.scheme == "n2p3-origin" and len(parts) == 3 and parts[1] == "participant":
                if source_participant_excluded(unquote(parsed.netloc), unquote(parts[2])):
                    excluded.append(record.local_subject_id)
                    break
    return tuple(excluded)


def assert_origin_keys_allowed(keys: Sequence[str]) -> None:
    for origin in keys:
        parsed = urlsplit(origin)
        parts = parsed.path.split("/")
        if parsed.scheme == "n2p3-origin" and len(parts) == 3 and parts[1] == "participant":
            if source_participant_excluded(unquote(parsed.netloc), unquote(parts[2])):
                raise ValueError(f"{PARTICIPANT_EXCLUSION_POLICY}: excluded source identity {origin}.")


def assert_identity_participants_allowed(table: DatasetIdentityTable) -> None:
    excluded = excluded_identity_subjects(table)
    if excluded:
        raise ValueError(
            f"{PARTICIPANT_EXCLUSION_POLICY}: excluded BI2014a/BI2015a participants {excluded}; "
            "rebuild the cache or retrain the checkpoint using the author-admitted cohort."
        )


def select_allowed_subjects(
    dataset_id: str, subjects: Sequence[int], *, explicit: bool
) -> list[int]:
    excluded = [subject for subject in subjects if source_participant_excluded(dataset_id, subject)]
    if excluded and explicit:
        raise ValueError(f"{PARTICIPANT_EXCLUSION_POLICY}: explicitly requested excluded subjects {excluded}.")
    return [subject for subject in subjects if subject not in excluded]
