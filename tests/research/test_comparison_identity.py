from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from research.comparison import ContrastPlan, evaluation_input_identity
from tests.support.datasets import binary_dataset


def identity_dataset():
    dataset = binary_dataset()
    dataset.provenance["source_preparation"] = {
        "consumed_inputs": [{"role": "raw", "sha256": "a" * 64}]}
    return dataset


def test_declared_baseline_contrast_preserves_physical_evidence_but_not_input():
    dataset = identity_dataset()
    new = copy.deepcopy(dataset)
    new.preprocessing = replace(new.preprocessing, baseline_mode="none")
    new.X += 1e-6
    left = evaluation_input_identity(dataset, np.arange(20), np.arange(20, 40))
    right = evaluation_input_identity(new, np.arange(20), np.arange(20, 40))
    assert left["fit"]["evidence_sha256"] == right["fit"]["evidence_sha256"]
    assert left["digest"] != right["digest"]
    ContrastPlan(("baseline",)).validate_pair(left, right)
    with pytest.raises(ValueError, match="materialized"):
        ContrastPlan().validate_pair(left, right)
    with pytest.raises(ValueError, match="undeclared"):
        ContrastPlan(("filter",)).validate_pair(left, right)


def test_declared_operator_cannot_hide_changed_truth_or_undeclared_raw_support():
    dataset = identity_dataset()
    new = copy.deepcopy(dataset)
    new.y[0] = 1-new.y[0]
    new.preprocessing = replace(new.preprocessing, baseline_mode="none")
    with pytest.raises(ValueError, match="physical"):
        ContrastPlan(("baseline",)).validate_pair(
            evaluation_input_identity(dataset, [0, 1], [2, 3]),
            evaluation_input_identity(new, [0, 1], [2, 3]))


def test_calibration_mutation_invalidates_materialization_without_test_change():
    dataset = identity_dataset()
    before = evaluation_input_identity(dataset, [0, 1], [2, 3])
    dataset.X[0] *= 3
    after = evaluation_input_identity(dataset, [0, 1], [2, 3])
    assert before["score"] == after["score"]
    assert before["digest"] != after["digest"]
    with pytest.raises(ValueError, match="without a declared changed operator"):
        ContrastPlan(("filter",)).validate_pair(before, after)


def test_sidecar_only_evidence_cannot_certify_a_raw_operator_contrast():
    dataset = identity_dataset()
    dataset.provenance["source_preparation"]["consumed_inputs"][0]["role"] = "events"
    new = copy.deepcopy(dataset)
    new.preprocessing = replace(new.preprocessing, baseline_mode="none")
    left, right = (evaluation_input_identity(data, [0, 1], [2, 3]) for data in (dataset, new))
    assert left["raw_bytes_bound"] is False
    with pytest.raises(ValueError, match="raw byte"):
        ContrastPlan(("baseline",)).validate_pair(left, right)


def test_one_verified_raw_parent_does_not_certify_an_unbound_parent():
    from data.epochs import concatenate_epoch_datasets, materialize_dataset_lineage
    from data.lineage import DataLineage

    known, unknown = binary_dataset(), binary_dataset()
    known.lineage = DataLineage.derive([materialize_dataset_lineage(known)],
        operation="verified_source_record_preparation",
        parameters={"consumed_inputs": [{"role": "raw", "sha256": "a" * 64}]})
    unknown.name = "unbound_second_parent"
    unknown.provenance = {**unknown.provenance, "source": "unknown_second_reader"}
    unknown.event_timeline = replace(unknown.event_timeline,
        event_ids=np.asarray([f"unknown-{index}" for index in range(unknown.n_epochs)]),
        group_ids=np.repeat("second_group", unknown.n_epochs), run_ids=np.repeat("second_run", unknown.n_epochs))
    combined = concatenate_epoch_datasets((known, unknown), name="mixed",
        provenance={"source_reference": known.provenance["source_reference"]})
    identity = evaluation_input_identity(combined, [40, 41], [42, 43])
    assert identity["raw_bytes_bound"] is False


def test_baseline_permission_does_not_also_authorize_implementation_changes():
    from research.contracts import semantic_sha256

    dataset = identity_dataset()
    new = copy.deepcopy(dataset)
    new.preprocessing = replace(new.preprocessing, baseline_mode="none")
    left, right = (evaluation_input_identity(data, [0, 1], [2, 3]) for data in (dataset, new))
    for identity, version in ((left, "a"), (right, "b")):
        identity["operator"]["execution"] = {"implementation": {"source_sha256": version * 64}}
        identity["digest"] = semantic_sha256({key: value for key, value in identity.items() if key != "digest"})
    with pytest.raises(ValueError, match="operator_implementation"):
        ContrastPlan(("baseline",)).validate_pair(left, right)
    ContrastPlan(("baseline", "operator_implementation")).validate_pair(left, right)


def test_decision_supervision_context_cannot_be_changed_under_same_input_values():
    dataset = identity_dataset()
    original = evaluation_input_identity(dataset, [0, 1], [2, 3])
    dataset.provenance["candidate_vocabulary"] = ["left", "right"]
    changed = evaluation_input_identity(dataset, [0, 1], [2, 3])
    assert original["fit"] == changed["fit"]
    assert original["digest"] != changed["digest"]
    with pytest.raises(ValueError, match="decision semantics"):
        ContrastPlan(("baseline",)).validate_pair(original, changed)


@pytest.mark.parametrize("missing", ["decision_semantics", "fit_context", "requested_schedule_sha256", "metadata_sha256"])
def test_resigned_partial_identity_cannot_certify_two_matched_cells(missing):
    from research.contracts import semantic_sha256

    identity = evaluation_input_identity(identity_dataset(), [0, 1], [2, 3])
    container = (identity["decision_semantics"] if missing == "requested_schedule_sha256" else
                 identity["fit_context"] if missing == "metadata_sha256" else identity)
    del container[missing]
    identity["digest"] = semantic_sha256({key: value for key, value in identity.items() if key != "digest"})
    with pytest.raises(ValueError, match="incomplete.*identity"):
        ContrastPlan().validate_pair(identity, copy.deepcopy(identity))


def test_old_partial_schema_is_not_silently_upgraded_for_current_statistics():
    from research.contracts import semantic_sha256

    identity = evaluation_input_identity(identity_dataset(), [0, 1], [2, 3])
    identity["schema"] = "n2p3_evaluation_input_identity/1"
    identity["digest"] = semantic_sha256({key: value for key, value in identity.items() if key != "digest"})
    with pytest.raises(ValueError, match="current input identity"):
        ContrastPlan().validate_pair(identity, copy.deepcopy(identity))


@pytest.mark.parametrize("rows", [[1.2], [True], [1, 1], [-1], [100]])
def test_physical_rows_never_truncate_or_repeat(rows):
    with pytest.raises(ValueError, match="rows"):
        evaluation_input_identity(identity_dataset(), rows, [3])


def test_statistics_accepts_declared_operator_change_and_rejects_unplanned_change():
    from stats.transfer import analyze_transfer_contrasts
    from transfer.binary import matched_input_digests, matched_split_digests

    old = identity_dataset()
    new = copy.deepcopy(old)
    new.preprocessing = replace(new.preprocessing, baseline_mode="none")
    new.X += 1e-6
    fit, score = np.arange(20), np.arange(20, 40)
    rows = []
    for arm, data in (("old", old), ("new", new)):
        observed = np.broadcast_to(data.channel_mask, data.X.shape[:2])
        rows.append(dict(study="toy", participant_key="p1", K=20, source_seed=1, target_seed=1,
                         status="completed", arm=arm, auc=.6,
                         **matched_split_digests(fit, score),
                         **matched_input_digests(data.X, data.y, score, observed, data.preprocessing),
                         comparison_identity=evaluation_input_identity(data, fit, score)))
    options = dict(expected_studies=["toy"], source_seeds=[1], target_seeds=[1],
                   standard_budgets=[20], iterations=1000)
    with pytest.raises(ValueError, match="materialized"):
        analyze_transfer_contrasts(rows, {"baseline": {"new": 1., "old": -1.}}, **options)
    result = analyze_transfer_contrasts(rows, {"baseline": {"new": 1., "old": -1.}},
        contrast_plans={"baseline": ContrastPlan(("baseline",)).record()}, **options)
    assert result["groups"][0]["contrasts"]["baseline"]["exact_K"]["20"]["mean_difference"] == 0
