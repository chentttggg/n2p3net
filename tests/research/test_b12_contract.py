"""Independent final B12 manifest counterexamples and its actual caller."""

from __future__ import annotations

import copy
import json

import pytest

from data.source_records import prepare_source_record_prefixes
from experiments.run_erp_comparator import _run_comparison_serial
from research.b12_contract import validate_b12_execution_manifest
from research.contracts import semantic_sha256
from tests.support.raw_prefix import write_real_record


def manifest_fixture(*, public=False):
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["p"], "budgets": [200, 800],
            "arms": [{"name": "nn", "family": "full_fine"}], "seed_pairs": [[11, 21]],
            "source_seed_catalog": [{"outer_target_study": "target", "source_seed": 11,
                "sha256": "a" * 64, "procedure": {"kind": "selection_refit"}}],
            "evidence_scope": "public_development" if public else "software_correctness"}
    producer = {"kind": "selection_refit", "selection": "per_source_seed",
                "source_seed": 11, "source_studies": ["source"]}
    source = {"outer_target_study": "target", "source_seed": 11, "checkpoint_sha256": "a" * 64,
              "procedure": {"kind": "selection_refit"}, "producer": producer,
              "producer_digest": semantic_sha256(producer), "source_studies": ["source"]}
    cell = {"subject": "p", "arm": "nn", "K": 160, "actual_K": 160, "requested_K": 200,
            "requested_Ks": [200, 800], "source_seed": 11, "target_seed": 21, "status": "completed"}
    report = {"schema": "n2p3_erp_comparator_result/2", "seed_pairs": [[11, 21]], "cells": [cell],
              "requested_participants": [{"subject": "p", "arm": "nn", "requested_K": budget,
                  "actual_K": 160, "source_seed": 11, "target_seed": 21, "cell_index": 0} for budget in (200, 800)],
              "requested_budget_coverage": [{"requested_K": budget, "requested_cells": 1, "completed_cells": 1,
                  "protocol_unavailable_cells": 0, "failed_cells": 0} for budget in (200, 800)],
              "participant_authority": {"p": {"subject": "p", "study": "target", "participant_key": "target/p",
                                                "identity_record": {"fixture": True}}},
              "input_manifests": {"prefix.json": "b" * 64}, "source_seed_catalog": [source]}
    return plan, report


def test_requested_k_aliases_keep_two_denominators_for_one_actual_cell():
    plan, report = manifest_fixture()
    binding = validate_b12_execution_manifest(report, plan)
    assert binding["status"] == "bound"
    assert binding["requested_Ks"] == [200, 800]
    assert set(binding["failure_accounting"]) == {"200", "800"}
    assert len(report["cells"]) == 1


@pytest.mark.parametrize("change", ["missing_request", "duplicate_request", "wrong_actual_K", "dropped_failure", "roster", "extra_cell"])
def test_rehashed_final_manifest_cannot_drop_or_change_requested_authority(change):
    plan, report = manifest_fixture()
    if change == "missing_request":
        report["requested_participants"].pop()
    elif change == "duplicate_request":
        report["requested_participants"][1] = copy.deepcopy(report["requested_participants"][0])
    elif change == "wrong_actual_K":
        report["requested_participants"][1]["actual_K"] = 200
    elif change == "dropped_failure":
        report["requested_budget_coverage"][0]["completed_cells"] = 0
    elif change == "roster":
        report["cells"][0]["arm"] = "another"
    else:
        report["cells"].append(copy.deepcopy(report["cells"][0]))
    with pytest.raises(ValueError, match="B12"):
        validate_b12_execution_manifest(report, plan)


def test_public_admission_requires_original_source_producer_and_target_manifest():
    plan, report = manifest_fixture(public=True)
    original = copy.deepcopy(report)
    assert validate_b12_execution_manifest(report, plan)["source_producers"][0]["producer_digest"] == report["source_seed_catalog"][0]["producer_digest"]
    report["source_seed_catalog"][0]["producer"]["source_seed"] = 12
    with pytest.raises(ValueError, match="public source producer"):
        validate_b12_execution_manifest(report, plan)
    report = original
    report.pop("input_manifests")
    with pytest.raises(ValueError, match="target producer manifest"):
        validate_b12_execution_manifest(report, plan)


def test_software_fixture_does_not_invent_public_producer_identity():
    plan, report = manifest_fixture()
    for field in ("producer", "producer_digest"):
        report["source_seed_catalog"][0].pop(field)
    binding = validate_b12_execution_manifest(report, plan)
    assert binding["source_producers"][0]["producer_digest"] is None


@pytest.mark.parametrize("mutation", ["false_coverage", "failed_cell", "duplicate_source", "checkpoint",
    "source_seed", "kind", "target_study", "study_mirror"])
def test_bound_report_requires_consistent_terminal_and_source_facts(mutation):
    plan, report = manifest_fixture(public=True)
    source = report["source_seed_catalog"][0]
    if mutation == "false_coverage":
        report["requested_budget_coverage"][0].update(completed_cells=0, failed_cells=1)
    elif mutation == "failed_cell":
        report["cells"][0].update(status="failed", error="actual numerical failure")
    elif mutation == "duplicate_source":
        report["source_seed_catalog"].append(copy.deepcopy(source))
    elif mutation == "checkpoint":
        source["checkpoint_sha256"] = "d" * 64
    elif mutation == "source_seed":
        source["producer"]["source_seed"] += 1
    elif mutation == "kind":
        source["producer"].update(kind="fixed_recipe", selection="not_applicable")
    elif mutation == "target_study":
        source["producer"]["source_studies"].append("target")
        source["source_studies"] = source["producer"]["source_studies"]
    else:
        source["source_studies"] = ["different_source"]
    source["producer_digest"] = semantic_sha256(source["producer"])
    with pytest.raises(ValueError, match="B12"):
        validate_b12_execution_manifest(report, plan)


def test_unresolved_peer_cannot_remove_a_completed_peers_source():
    plan, report = manifest_fixture(public=True)
    plan["subjects"].append("unresolved")
    failed = copy.deepcopy(report["cells"][0])
    failed.update(subject="unresolved", status="failed", error="worker exited before preparation")
    report["cells"].append(failed)
    report["requested_participants"].extend({**request, "subject": "unresolved", "cell_index": 1}
        for request in list(report["requested_participants"]))
    for coverage in report["requested_budget_coverage"]:
        coverage.update(requested_cells=2, failed_cells=1)
    assert validate_b12_execution_manifest(report, plan)["status"] == "unresolved_target_authority"
    report["source_seed_catalog"] = []
    with pytest.raises(ValueError, match="required source producer"):
        validate_b12_execution_manifest(report, plan)


def test_failed_known_target_retains_missing_source_as_unresolved():
    plan, report = manifest_fixture(public=True)
    report["cells"][0].update(status="failed", error="source dependency unavailable")
    for coverage in report["requested_budget_coverage"]:
        coverage.update(completed_cells=0, failed_cells=1)
    report["source_seed_catalog"] = []
    binding = validate_b12_execution_manifest(report, plan)
    assert binding["status"] == "unresolved_source_producers"
    assert binding["unresolved_source_producers"] == [{"outer_target_study": "target", "source_seed": 11}]


def test_report_binding_digest_excludes_previous_binding():
    plan, report = manifest_fixture()
    first = validate_b12_execution_manifest(report, plan)
    report["b12_execution_binding"] = first
    assert validate_b12_execution_manifest(report, plan) == first


def test_failed_actor_preserves_denominators_and_unresolved_authority():
    plan, report = manifest_fixture()
    plan["arms"][0]["family"] = "waveform_lda"
    plan.pop("source_seed_catalog")
    report["participant_authority"] = {}
    report["source_seed_catalog"] = None
    report["cells"][0].update(status="failed", error="source dependency blocked before input admission")
    for row in report["requested_budget_coverage"]:
        row.update(completed_cells=0, failed_cells=1)
    binding = validate_b12_execution_manifest(report, plan)
    assert binding["status"] == "unresolved_target_authority"
    assert binding["unresolved_subjects"] == ["p"]
    assert all(row["failed_cells"] == 1 for row in binding["failure_accounting"].values())


def test_actual_paired_runner_consumes_final_manifest_with_software_scope(tmp_path):
    item, spec, _ = write_real_record(tmp_path)
    prefix = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prepared",
        preprocessing=spec, budgets=(0, 8, 200, 800), minimum_test_trials=2)
    collection = tmp_path / "prefix.json"
    collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": prefix["preprocessing"], "preparation_method": prefix["preparation_method"],
        "records": [prefix]}), encoding="utf-8")
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [0, 8, 200, 800],
        "arms": [{"name": "waveform", "family": "waveform_lda", "prefix_collection": str(collection)}],
        "allowed_input_axes": [], "evidence_scope": "software_correctness", "seed_pairs": [[11, 21]],
        "source_seed_catalog": [], "contrasts": {}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    report = _run_comparison_serial(path, tmp_path / "result")
    binding = report["b12_execution_binding"]
    assert binding["status"] == "bound" and binding["source_producers"] == []
    assert binding["requested_Ks"] == [0, 8, 200, 800]
    assert binding["failure_accounting"]["0"]["protocol_unavailable_cells"] == 1
    assert binding["failure_accounting"]["200"]["completed_cells"] == 1
    assert binding["failure_accounting"]["800"]["completed_cells"] == 1
