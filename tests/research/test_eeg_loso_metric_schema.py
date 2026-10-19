from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import baselines.evaluate as evaluation
from data.identity import DatasetIdentityTable
from data.qc_features import EpochQCFeatures
from experiments import run_eeg_loso as runner
from tests.support.datasets import epoch_dataset


def _metric_v2() -> dict[str, object]:
    return {
        "metric_schema": evaluation.LOSO_METRICS_SCHEMA_V2,
        "primary_decision_hit_rate": 0.2,
        "primary_decision_hit_rate_decision_micro": 0.2,
        "primary_participant_macro_hit_rate": 1 / 3,
        "primary_hit_scope": {
            "kind": "complete_repetition_terminal",
            "denominator": "requested_pre_qc_decision_groups",
        },
        "hit_at_repetition_macro": {"1": 0.2},
        "hit_at_repetition_decision_micro": {"1": 0.2},
        "hit_at_repetition_participant_macro": {"1": 1 / 3},
        "participant_macro_scope": {
            "reduction": "participant_macro_equal_weight",
            "participant_count": 3,
            "participant_identity": "explicit_authority_key",
            "denominator": "requested_pre_qc_decision_groups",
            "unavailable_policy": "incorrect",
        },
    }


def test_participant_macro_keeps_unequal_decision_counts_equal() -> None:
    assert evaluation._participant_macro(
        {"A": [True], "B": [False, False, False]}, ("A", "B")
    ) == pytest.approx(0.5)


def test_participant_macro_counts_an_unusable_participant_level_as_a_miss() -> None:
    assert evaluation._participant_macro(
        {"A": [True], "B": []}, ("A", "B")
    ) == pytest.approx(0.5)


def test_legacy_json_record_preserves_values_with_unknown_macro_and_denominator() -> None:
    original = {"primary_decision_hit_rate": 0.2, "hit_at_repetition_macro": {"1": 0.2}}
    raw = json.dumps(original)
    parsed = evaluation.validate_loso_metric_record(json.loads(raw))
    assert parsed["metric_schema"] == evaluation.LOSO_METRICS_SCHEMA_V1
    assert parsed["primary_decision_hit_rate"] == 0.2
    assert parsed["primary_participant_macro_hit_rate"] is None
    assert parsed["hit_at_repetition_macro"] == {"1": 0.2}
    assert parsed["hit_at_repetition_participant_macro"] is None
    assert parsed["primary_decision_hit_rate_decision_micro"] is None
    assert json.dumps(original) == raw


def test_v2_requires_explicit_macro_scope_and_preserves_micro_aliases() -> None:
    parsed = evaluation.validate_loso_metric_record(json.loads(json.dumps(_metric_v2())))
    assert parsed["metric_schema"] == evaluation.LOSO_METRICS_SCHEMA_V2
    assert parsed["primary_decision_hit_rate"] == pytest.approx(0.2)
    assert parsed["primary_participant_macro_hit_rate"] == pytest.approx(1 / 3)
    broken = _metric_v2()
    broken.pop("participant_macro_scope")
    with pytest.raises(ValueError, match="lacks fields"):
        evaluation.validate_loso_metric_record(broken)


def test_mixed_legacy_and_participant_macro_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="mixes version-2"):
        evaluation.validate_loso_metric_record({
            "primary_decision_hit_rate": 0.2,
            "primary_participant_macro_hit_rate": 0.5,
        })
    broken = _metric_v2()
    broken["primary_decision_hit_rate_decision_micro"] = 0.3
    with pytest.raises(ValueError, match="aliases disagree"):
        evaluation.validate_loso_metric_record(broken)


def _fixed_summary(monkeypatch, *, with_identity=True, participant_ids=None):
    # A owns a training-only group and one correct test decision. B has three
    # wrong decisions across sessions/domains. C loses its entire group to QC.
    group_ids = np.repeat(["train_without_truth", "a", "b1", "b2", "b3", "c"], 2)
    fit_groups = np.repeat(["train-session", "a-session", "b-session1", "b-session2", "b-session3", "c-session"], 2)
    domains = np.repeat(["train", "a", "b1", "b2", "b3", "c"], 2)
    if participant_ids is None:
        participant_ids = np.repeat(["source:A", "source:A", "source:B", "source:B", "source:B", "source:C"], 2)
    codes = np.tile([0, 1], 6)
    logits = np.tile([0.0, 2.0], 6)
    logits[2:4] = [2.0, 0.0]
    test = np.ones(len(codes), dtype=bool)
    test[:2] = False
    effective = test.copy()
    effective[-2:] = False
    binary = evaluation.BinaryFoldResult(1.0, 1.0, int(effective.sum()), 0.0, "fixed", 0.0)
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda *args, **kwargs: (
        [(0, binary, logits[effective], effective, None)], "fixed_predictions", 1, "none", 1, 1, 1,
    ))
    return evaluation.evaluate_candidate_selection(
        object(), np.zeros((len(codes), 1, 1)), codes, codes, group_ids,
        {"a": 0, "b1": 0, "b2": 0, "b3": 0, "c": 0}, [(~test, test)], [0, 1],
        fit_group_ids=fit_groups, participant_ids=participant_ids if with_identity else None,
        domain_ids=domains, repetition_indices=np.zeros(len(codes), dtype=np.int64),
    )


def test_candidate_summary_keeps_participants_equal_across_sessions_domains_and_qc(monkeypatch):
    summary = _fixed_summary(monkeypatch)
    assert summary.primary_hit_rate == pytest.approx(1 / 5)
    assert summary.primary_participant_macro_hit_rate == pytest.approx(1 / 3)
    assert summary.hit_at_repetition_participant_macro == {"1": pytest.approx(1 / 3)}
    assert summary.hit_at_repetition_macro == {"1": pytest.approx(1 / 5)}
    assert summary.participant_macro_scope["participant_count"] == 3
    assert summary.per_fold[0].n_decisions == 5
    assert summary.subject_records[-1] == (None, 0, "c")
    parsed = evaluation.validate_loso_metric_record(json.loads(json.dumps(runner._summary_metric_fields(summary))))
    assert parsed["primary_decision_hit_rate"] == pytest.approx(1 / 5)
    assert parsed["primary_participant_macro_hit_rate"] == pytest.approx(1 / 3)


def test_missing_authority_never_infers_participants_from_fit_or_domain_groups(monkeypatch):
    summary = _fixed_summary(monkeypatch, with_identity=False)
    assert summary.primary_hit_rate == pytest.approx(1 / 5)
    assert summary.primary_participant_macro_hit_rate is None
    assert summary.hit_at_repetition_participant_macro is None
    assert summary.participant_macro_scope is None
    with pytest.raises(ValueError, match="scope"):
        runner._summary_metric_fields(summary)


def test_terminal_macro_and_repetition_curve_use_each_requested_physical_horizon(monkeypatch):
    groups = np.asarray(["train"] * 2 + ["a"] * 4 + ["b1"] * 2 + ["b2"] * 2 + ["b3"] * 2)
    participants = np.asarray(["training"] * 2 + ["A"] * 4 + ["B"] * 6)
    codes = np.tile([0, 1], 6)
    repetitions = np.asarray([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    test = np.arange(12) >= 2
    # A is right at R1, wrong at its physical R2 terminal; B is right at R1
    # terminals but cannot supply R2. All four requests remain at every level.
    llr = np.asarray([3.0, 0.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    binary = evaluation.BinaryFoldResult(1.0, 1.0, 10, 0.0, "fixed", 0.0)
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda *args, **kwargs: (
        [(0, binary, llr, test, None)], "fixed_predictions", 1, "none", 1, 1, 1,
    ))
    summary = evaluation.evaluate_candidate_selection(
        object(), np.zeros((12, 1, 1)), codes, codes, groups,
        {"a": 0, "b1": 0, "b2": 0, "b3": 0}, [(~test, test)], [0, 1],
        participant_ids=participants, repetition_indices=repetitions,
    )
    assert summary.primary_hit_rate == pytest.approx(3 / 4)
    assert summary.primary_participant_macro_hit_rate == pytest.approx(1 / 2)
    assert summary.hit_at_repetition_participant_macro == {"1": 1.0, "2": 0.0}
    assert summary.per_fold[0].hit_total_by_repetition == {"1": 4, "2": 4}


def test_same_local_label_in_different_libraries_uses_the_declared_authority(monkeypatch):
    left = DatasetIdentityTable.from_source_rows(["01"], ["left-study"])
    right = DatasetIdentityTable.from_source_rows(["01"], ["right-study"])
    left_key, right_key = left.record_for("01").authority_key(), right.record_for("01").authority_key()
    assert left_key != right_key
    summary = _fixed_summary(monkeypatch, participant_ids=np.repeat([
        left_key, left_key, right_key, right_key, right_key, right_key,
    ], 2))
    assert summary.primary_participant_macro_hit_rate == pytest.approx(0.5)
    assert summary.participant_macro_scope["participant_count"] == 2


def test_cli_authority_mapping_preserves_selected_row_order():
    table = DatasetIdentityTable.from_source_rows(["A", "B", "C"], ["study"] * 3)
    dataset = SimpleNamespace(identity_table=table, subject_ids=np.asarray(["A", "B", "C"]))
    actual = runner._participant_authority_ids(dataset, np.asarray(["B", "A", "B"]))
    assert actual.tolist() == [table.record_for(local).authority_key() for local in ("B", "A", "B")]


@pytest.mark.parametrize("field,value", [
    ("hit_at_repetition_decision_micro", {"1": 0.3}),
    ("hit_at_repetition_participant_macro", {"2": 0.5}),
    ("primary_participant_macro_hit_rate", None),
])
def test_new_record_rejects_mixed_reductions_or_missing_values(field, value):
    record = _metric_v2()
    record[field] = value
    with pytest.raises(ValueError):
        evaluation.validate_loso_metric_record(record)


@pytest.mark.parametrize("field,value", [
    ("denominator", "successful_decisions"),
    ("unavailable_policy", "omit"),
    ("participant_count", 0),
    ("participant_identity", "fit_group_ids"),
])
def test_new_record_rejects_inconsistent_participant_scope(field, value):
    record = _metric_v2()
    record["participant_macro_scope"][field] = value
    with pytest.raises(ValueError):
        evaluation.validate_loso_metric_record(record)


def test_binary_metric_record_has_no_candidate_claim():
    fields = runner._summary_metric_fields(evaluation.BinarySummary(1.0, 0.0, 1.0))
    parsed = evaluation.validate_loso_metric_record(json.loads(json.dumps(fields)))
    assert parsed["participant_macro_scope"] == {
        "reduction": "not_applicable_binary", "participant_count": 0,
        "denominator": "not_applicable",
    }
    assert parsed["primary_participant_macro_hit_rate"] is None
    fields["primary_participant_macro_hit_rate"] = 0.5
    with pytest.raises(ValueError, match="no candidate metrics"):
        evaluation.validate_loso_metric_record(fields)


def test_actual_cli_writes_versioned_metrics_with_real_identity_mapping(tmp_path, monkeypatch):
    # Cache/model boundaries are synthetic; CLI, authority mapping, evaluator,
    # metric writer, final JSON and done marker exercise the real call chain.
    dataset = epoch_dataset()
    dataset.qc_features = EpochQCFeatures(
        np.ones((6, 3)), np.ones((6, 3)), np.ones(6), np.ones((6, 3), dtype=bool),
    )
    monkeypatch.setattr(runner, "load_epoch_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(runner, "loaded_epoch_cache_attestation", lambda data: {"sha256": "synthetic"})
    monkeypatch.setattr(runner, "assert_p300_input_contract", lambda *args: None)
    monkeypatch.setattr(runner, "assert_p300_source_provenance", lambda *args: None)
    monkeypatch.setattr(runner, "resolve_fold_local_artifact_models", lambda *args, **kwargs: (
        {}, {"fit_seconds": 0.0},
    ))
    monkeypatch.setattr(runner, "build_binary_model", lambda *args, **kwargs: SimpleNamespace(
        configure_epoch_progress=lambda output: None,
    ))
    monkeypatch.setattr(runner, "describe_binary_model", lambda *args: {"name": "synthetic"})
    binary = evaluation.BinaryFoldResult(1.0, 1.0, 2, 0.0, "fixed", 0.0)
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda model, X, y, fit_groups, folds, **kwargs: (
        [(index, binary, np.asarray([0.0, 2.0]), test, None)
         for index, (_, test) in enumerate(folds)], "fixed_predictions", 1, "none", 1, 1, 1,
    ))
    monkeypatch.setattr(sys, "argv", ["run_eeg_loso", "--dataset-cache", "synthetic.npz",
        "--models", "window_lr", "--run-dir", str(tmp_path), "--run-name", "software_probe",
        "--device", "cpu", "--subjects", "2", "--max-folds", "1"])
    runner.main()
    output = tmp_path / "software_probe/window_lr"
    record = json.loads((output / "record.json").read_text(encoding="utf-8"))
    parsed = evaluation.validate_loso_metric_record(record)
    assert parsed["metric_schema"] == evaluation.LOSO_METRICS_SCHEMA_V2
    assert parsed["primary_participant_macro_hit_rate"] == 1.0
    assert parsed["participant_macro_scope"]["participant_count"] == 1
    assert record["dataset"]["n_subjects"] == 3
    assert json.loads((output / "progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])["type"] == "done"
    assert not (output / ".record.json.tmp").exists()
