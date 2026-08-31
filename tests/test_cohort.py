from __future__ import annotations

import pytest

from transfer.cohort import read_subject_manifest, resolve_subject_scope


def test_subject_manifest_accepts_json_and_lines(tmp_path) -> None:
    json_path = tmp_path / "subjects.json"
    text_path = tmp_path / "subjects.txt"
    json_path.write_text('["s2", "s1"]', encoding="utf-8")
    text_path.write_text("s2\ns1\n", encoding="utf-8")

    assert read_subject_manifest(json_path) == ("s2", "s1")
    assert read_subject_manifest(text_path) == ("s2", "s1")


def test_subject_scope_preserves_requested_denominator_and_filters_usable() -> None:
    requested, usable = resolve_subject_scope(
        ["s1", "s2", "s3"],
        ["s1", "s3"],
        requested_subjects=["s3", "s2"],
    )

    assert requested == ("s3", "s2")
    assert usable == ("s3",)


def test_subject_scope_rejects_unknown_duplicates_and_conflicting_limit(tmp_path) -> None:
    path = tmp_path / "subjects.json"
    path.write_text('["s1", "s1"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_subject_manifest(path)
    with pytest.raises(ValueError, match="absent"):
        resolve_subject_scope(["s1"], ["s1"], requested_subjects=["s2"])
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_subject_scope(
            ["s1"],
            ["s1"],
            requested_subjects=["s1"],
            max_subjects=1,
        )
