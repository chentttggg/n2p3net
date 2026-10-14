"""Saved graph scope uses each original record's local clock and global labels."""

from copy import deepcopy

import pytest

from research.prefix_selection_replay import _audit_record_before_held, _scope_before_held


def record_scope():
    held = {"record_id": "second:held", "raw_scope": [350, 850],
        "view": {"source_sha256": "b" * 64, "source_relative_path": "second.mat"}}
    scopes = {"first": {"source_sha256": "a" * 64, "source_relative_path": "first.mat",
                        "global_label_indices": [0, 12], "read_sample_support": [0, 900]},
              "second": {"source_sha256": "b" * 64, "source_relative_path": "second.mat",
                         "global_label_indices": [12, 14], "read_sample_support": [0, 350]}}
    scope = {"actual_K": 14, "input": {"budget": 14, "target_contract": {
        "authority_kind": "record_aware_calibration_prefix/1", "raw_scope": None, "record_scopes": scopes},
        "complete_groups": []}}
    return scope, held


def test_graph_before_held_reads_only_the_matching_original_record_clock():
    scope, held = record_scope()
    _scope_before_held(scope, held)


@pytest.mark.parametrize("violation", ["sample_overlap", "ambiguous_record", "held_in_groups"])
def test_graph_rejects_actual_matching_record_overlap_or_repeated_authority(violation):
    scope, held = record_scope()
    if violation == "sample_overlap":
        scope["input"]["target_contract"]["record_scopes"]["second"]["read_sample_support"][1] = 351
    elif violation == "ambiguous_record":
        scope["input"]["target_contract"]["record_scopes"]["duplicate"] = deepcopy(
            scope["input"]["target_contract"]["record_scopes"]["second"])
    else:
        scope["input"]["complete_groups"] = [held]
    with pytest.raises(ValueError, match="held|record"):
        _scope_before_held(scope, held)


def original_and_earlier():
    scope, held = record_scope()
    for name, record in scope["input"]["target_contract"]["record_scopes"].items():
        record.update(record_id=name, source_dataset_id="BI2014a", source_subject_id="01",
                      native_clock="actual_record_native_sample")
    scope["input"]["native_records"] = [
        {"record_id": "first", "event_onsets": list(range(12)), "fit_input_digest": "c" * 64},
        {"record_id": "second", "event_onsets": [100, 300], "fit_input_digest": "d" * 64}]
    original = deepcopy(scope["input"])
    original["budget"] = 16
    original["target_contract"]["record_scopes"]["second"].update(global_label_indices=[12, 16], read_sample_support=[0, 900])
    original["native_records"][1]["event_onsets"] = [100, 300, 360, 500]
    # The held first flash is at 500 but its preroll starts at 350. The 360
    # onset is excluded from the earlier C, giving exactly global K=12+2.
    held["view"].update(raw_scope=[350, 850], onsets=[150])
    return original, scope, held["view"]


def test_structural_record_audit_uses_preroll_and_preserves_previous_acquisition():
    original, scope, held = original_and_earlier()
    _audit_record_before_held(original, scope, held)


@pytest.mark.parametrize("violation", ["first_flash_instead_of_preroll", "changed_previous_record", "retained_unknown_event"])
def test_structural_record_audit_rejects_wrong_global_dose_or_original_operands(violation):
    original, scope, held = original_and_earlier()
    if violation == "first_flash_instead_of_preroll":
        scope["actual_K"] = 15
    elif violation == "changed_previous_record":
        scope["input"]["native_records"][0]["fit_input_digest"] = "e" * 64
    else:
        scope["input"]["native_records"][1]["event_onsets"] = [100, 360]
    with pytest.raises(ValueError, match="earlier|original|held"):
        _audit_record_before_held(original, scope, held)
