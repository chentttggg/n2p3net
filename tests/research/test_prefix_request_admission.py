"""JSON-exported complete programme requests retain every scientific field."""
import json
from dataclasses import replace

import pytest

import experiments.run_candidate_ensemble_study as runner
from research.calibration_selection import CalibrationProgramCandidate
from research.contracts import semantic_sha256
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_views import observed_spatial_projection, prepare_bnci_independent_prefixes


@pytest.fixture(scope="module")
def request_case(tmp_path_factory):
    root, _, path, preprocessing, experts, original = independent_prefix_case(
        tmp_path_factory.mktemp("serialized-prefix-requests"))
    channels = ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
    native = original.calibration[200].target_contract["channel_names"]
    matrix, _ = observed_spatial_projection(native, channels, reference_channel="Cz")
    spatial = {"mode": "observed_measured_reference", "slot_channels": list(channels),
        "reference_channel": "Cz", "matrix": matrix.tolist(), "native_sample_rate_hz": 256.,
        "observed_only_for_raw_covariance": True}
    raw = replace(experts[0], expert_id="raw_cz6", spatial=spatial)
    ensemble = EnsembleProgram.equal_weights("raw_cz6", (raw.expert_id,))
    candidate = CalibrationProgramCandidate("fixed", (raw,), ensemble)
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path, source_sha256=original.ledger["source_sha256"],
            budgets=(200,), profiles=(), authorization_preprocessing=preprocessing,
            response_seconds=308 / 256, pre_roll_seconds=.2)
    plan = {"evaluation_design": "independent_prefix", "seed_pairs": [[11, 21]], "budgets": [200],
        "experts": [raw.record()], "source_checkpoints": {}, "source_seed_catalog": [],
        "evidence_scope": "software_correctness", "authorization_preprocessing": preprocessing.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2,
        "prefix_selection": {"candidates": [candidate.record()], "baseline_name": "fixed",
                             "minimum_selection_rows": 3, "output_name": "selected"}}
    requests = runner._prefix_selection_requests(plan, views.ledger)
    serialized = json.loads(json.dumps(list(requests.values())))
    assert serialized[0] != next(iter(requests.values()))
    assert semantic_sha256(serialized) == semantic_sha256(list(requests.values()))
    return plan, views.ledger, serialized, {"path": str(path), "sha256": original.ledger["source_sha256"]}


@pytest.mark.parametrize("receipt_kind", ["started", "failed", "saved"])
def test_actual_serialized_request_reaches_next_ingress_without_tuple_list_rejection(
        request_case, tmp_path, monkeypatch, receipt_kind):
    plan, scopes, requests, record = request_case
    request = requests[0]
    directory = tmp_path / "participant"
    receipts = directory / "selection_fit_receipts"
    receipts.mkdir(parents=True)
    runner._sealed_record(receipts / (request["package"] + ".started.json"), request)
    if receipt_kind == "failed":
        runner._sealed_record(receipts / (request["package"] + ".failed.json"),
                              {"request": request, "error_type": "ExpectedSubjectError", "detail": "fixture"})
    ledger = {"record": record, "selection_fit_requests": requests}
    if receipt_kind == "saved":
        from research import prefix_selection_replay

        def graph_ingress(*args, **kwargs):
            raise ValueError("reached independent graph ingress")

        monkeypatch.setattr(prefix_selection_replay, "_SelectionReplayAdmission", graph_ingress)
        runner._sealed_record(receipts / (request["package"] + ".saved.json"),
            {"request": request, "files": [], "graph_sha256": "a" * 64})
        with cpu_thread_budget(1), pytest.raises(ValueError, match="reached independent graph ingress"):
            runner._admit_prefix_selection_packages(directory, tmp_path, plan, scopes, ledger)
    else:
        admitted, summaries, started, failed = runner._admit_prefix_selection_packages(
            directory, tmp_path, plan, scopes, ledger)
        assert not admitted and not summaries and started == {request["package"]}
        assert len(failed) == int(receipt_kind == "failed")


@pytest.mark.parametrize("mutation", ["matrix", "channel_order", "reference", "sample_rate", "source_digest",
    "seed", "scope", "candidate", "extra_field", "duplicate"])
def test_complete_canonical_request_rejects_real_changes_and_duplicate_packages(request_case, tmp_path, mutation):
    plan, scopes, requests, record = request_case
    changed = json.loads(json.dumps(requests))
    request = changed[0]
    spatial = request["candidates"][0]["expert_programs"][0]["spatial"]
    if mutation == "matrix":
        spatial["matrix"][0][0] += .01
    elif mutation == "channel_order":
        spatial["slot_channels"].reverse()
    elif mutation == "reference":
        spatial["reference_channel"] = "Fz"
    elif mutation == "sample_rate":
        spatial["native_sample_rate_hz"] = 128.
    elif mutation == "source_digest":
        request["source_checkpoint_sha256"]["raw_cz6"] = "a" * 64
    elif mutation == "seed":
        request["source_seed"] += 1
    elif mutation == "scope":
        request["target_contract"]["raw_scope"][1] += 1
    elif mutation == "candidate":
        request["candidates"] = []
    elif mutation == "extra_field":
        request["undeclared_fit_authority"] = "must not be projected away"
    else:
        changed.append(json.loads(json.dumps(request)))
    with pytest.raises(ValueError, match="complete programme requests"):
        runner._admit_prefix_selection_packages(tmp_path, tmp_path, plan, scopes,
            {"record": record, "selection_fit_requests": changed})
