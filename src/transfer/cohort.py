"""Explicit target-subject cohort manifests for block-scoped evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def read_subject_manifest(path: str | Path) -> tuple[str, ...]:
    """Read a JSON string list or a newline-delimited subject manifest."""

    manifest_path = Path(path)
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("subject manifest is empty.")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(decoded, list) or not decoded or not all(
        isinstance(value, str) and value.strip() for value in decoded
    ):
        raise ValueError("subject manifest must be a non-empty string list.")
    subjects = tuple(value.strip() for value in decoded)
    if len(set(subjects)) != len(subjects):
        raise ValueError("subject manifest contains duplicate subjects.")
    return subjects


def resolve_subject_scope(
    available_subjects: Sequence[str],
    usable_subjects: Sequence[str],
    *,
    requested_subjects: Sequence[str] | None = None,
    max_subjects: int | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return requested denominator and its usable ordered subset."""

    available = tuple(sorted({str(value) for value in available_subjects}))
    usable = {str(value) for value in usable_subjects}
    if requested_subjects is not None and max_subjects is not None:
        raise ValueError("an explicit subject manifest cannot be combined with max_subjects.")
    requested = (
        tuple(str(value) for value in requested_subjects)
        if requested_subjects is not None
        else available
    )
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"requested subjects are absent from the dataset: {sorted(unknown)}")
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive or None.")
        requested = requested[:max_subjects]
    selected_usable = tuple(subject for subject in requested if subject in usable)
    return requested, selected_usable
