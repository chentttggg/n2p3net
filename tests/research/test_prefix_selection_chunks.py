"""Prepared graph seals and independent node work retain the original replay gate."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from research import prefix_selection_replay as owner
from research.contracts import semantic_sha256
from research.frozen_metadata import plain_metadata
from research.selection_graph_store import SelectionGraphStore
from tests.support.prefix_replay import create_prefix_replay_graph
from train.runtime import cpu_thread_budget


@pytest.fixture(scope="module", name="actual_graph_case")
def make_actual_graph_case(tmp_path_factory):
    return create_prefix_replay_graph(tmp_path_factory.mktemp("admitted-prefix"))


@pytest.fixture(scope="module")
def completed_chunks(actual_graph_case):
    directory, views, future, graph_sha256, _ = actual_graph_case
    prefix = views.calibration[800]
    with cpu_thread_budget(1):
        admitted = owner._SelectionReplayAdmission(directory, expected_graph_sha256=graph_sha256)
        plan = admitted.prepare_replay(prefix=prefix, prefix_compiler=views.prefix_compiler,
            future_views=(future,))
        results = owner.replay_selection_chunk(plan.node_requests, directory=directory,
            parent_prefix=prefix, prefix_compiler=views.prefix_compiler, future_views=(future,))
    return plan, results


def test_actual_graph_serial_and_reversed_chunks_keep_the_same_numerical_oracle(
    actual_graph_case, monkeypatch,
):
    directory, views, future, graph_sha256, _ = actual_graph_case
    prefix = views.calibration[800]
    admitted = owner._SelectionReplayAdmission(directory, expected_graph_sha256=graph_sha256)
    with cpu_thread_budget(1):
        serial = admitted.replay(prefix=prefix, prefix_compiler=views.prefix_compiler,
            future_views=(future,))
        plan = admitted.prepare_replay(prefix=prefix, prefix_compiler=views.prefix_compiler,
            future_views=(future,))

        def no_graph_or_fit(*args, **kwargs):
            raise AssertionError("chunk attempted to reopen the large graph or fit a model")

        import transfer.ensemble_fit as fitter

        monkeypatch.setattr(owner, "_load_graph", no_graph_or_fit)
        monkeypatch.setattr(fitter, "_fit_expert", no_graph_or_fit)
        results = []
        for start in range(0, len(plan.node_requests), 3):
            results.extend(owner.replay_selection_chunk(plan.node_requests[start:start + 3],
                directory=directory, parent_prefix=prefix, prefix_compiler=views.prefix_compiler,
                future_views=(future,)))
        chunked = plan.finish_replay(reversed(results))
    assert chunked == serial
    assert chunked["zero_new_fits"] and chunked["exact_score_checks"] > 0
    assert all(row["record"]["native_threads"] == 1 for row in results)
    assert all(row["record"]["replay_device"] == "cpu" for row in results)
    assert max(len(json.dumps(request)) for request in plan.node_requests) < 10_000


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_scope", "false_checks", "false_scores"])
def test_reduction_rejects_incomplete_or_resigned_numerical_results(completed_chunks, mutation):
    plan, original = completed_chunks
    results = deepcopy(original)
    if mutation == "missing":
        results.pop()
    elif mutation == "duplicate":
        results.append(deepcopy(results[0]))
    else:
        record = results[0]["record"]
        if mutation == "extra":
            record["request"]["request"]["node_index"] = len(results)
        elif mutation == "wrong_scope":
            record["request"]["request"]["actual_K"] += 1
        elif mutation == "false_checks":
            record["replay"]["exact_score_checks"] += 1
        else:
            record["replay"]["scores"][0]["decision_id"] = "another-decision"
        record["request"]["request_sha256"] = semantic_sha256(record["request"]["request"])
        results[0]["record_sha256"] = semantic_sha256(record)
    with cpu_thread_budget(1), pytest.raises(ValueError, match="missing|repeats|request|score"):
        plan.finish_replay(results)


@pytest.mark.parametrize("mutation", ["scope", "source", "held", "held_id", "digest"])
def test_chunk_rejects_changed_request_before_numerical_replay(actual_graph_case, completed_chunks, mutation):
    directory, views, future, _, _ = actual_graph_case
    plan, _ = completed_chunks
    request = deepcopy(next(row for row in plan.node_requests if not row["request"]["final"]))
    if mutation == "scope":
        request["request"]["actual_K"] += 1
    elif mutation == "source":
        request["request"]["source_checkpoint_sha256"] = {"raw": "f" * 64}
    elif mutation == "held":
        request["request"]["views"][0]["input_sha256"] = "f" * 64
    elif mutation == "held_id":
        request["request"]["views"][0]["record_id"] = "another-held-record"
    else:
        request["request_sha256"] = "f" * 64
    if mutation != "digest":
        request["request_sha256"] = semantic_sha256(request["request"])
    with pytest.raises(ValueError, match="scope|source|observations|digest|held record"):
        owner.replay_selection_chunk((request,), directory=directory, parent_prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=(future,))


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "mixed_graph"])
def test_chunk_requires_a_unique_original_graph(completed_chunks, actual_graph_case, mutation):
    plan, _ = completed_chunks
    directory, views, future, _, _ = actual_graph_case
    requests = list(plan.node_requests[:2])
    if mutation == "empty":
        requests.clear()
    elif mutation == "duplicate":
        requests.append(requests[0])
    else:
        requests[1]["request"]["graph_sha256"] = "f" * 64
        requests[1]["request_sha256"] = semantic_sha256(requests[1]["request"])
    with pytest.raises(ValueError, match="least one|repeats|mixes"):
        owner.replay_selection_chunk(requests, directory=directory, parent_prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=(future,))


def test_chunk_rejects_resigned_original_record_against_external_node_seal(completed_chunks, actual_graph_case):
    plan, _ = completed_chunks
    directory, views, future, _, _ = actual_graph_case
    request = plan.node_requests[0]
    path = directory / request["request"]["package"] / "prefix_inner_replay.json"
    original = path.read_bytes()
    changed = json.loads(original)
    changed["record"]["actual_K"] += 1
    changed["record_sha256"] = semantic_sha256(changed["record"])
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="externally sealed producer"):
            owner.replay_selection_chunk((request,), directory=directory, parent_prefix=views.calibration[800],
                prefix_compiler=views.prefix_compiler, future_views=(future,))
    finally:
        path.write_bytes(original)


def test_final_reduction_keeps_fresh_loaded_model_relationship(completed_chunks, actual_graph_case, monkeypatch):
    plan, results = completed_chunks
    directory, views, future, graph_sha256, _ = actual_graph_case
    graph = plain_metadata(SelectionGraphStore(directory / "prefix_selection.json").graph)
    graph["selection"]["final_producer_sha256"] = "f" * 64
    admitted = owner._SelectionReplayAdmission.from_prepared_graph(directory, graph,
        expected_graph_sha256=semantic_sha256(graph))
    altered = admitted.prepare_replay(prefix=views.calibration[800], prefix_compiler=views.prefix_compiler,
        future_views=(future,))
    rebound = []
    for original, request in zip(results, altered.node_requests, strict=True):
        envelope = deepcopy(original)
        envelope["record"]["request"] = request
        envelope["record_sha256"] = semantic_sha256(envelope["record"])
        rebound.append(envelope)
    loads = []
    original_package = owner._PrefixPackageAdmission

    def track_final(*args, **kwargs):
        loads.append(Path(args[0]))
        return original_package(*args, **kwargs)

    monkeypatch.setattr(owner, "_PrefixPackageAdmission", track_final)
    with cpu_thread_budget(1), pytest.raises(ValueError, match="final numerical programme"):
        altered.finish_replay(rebound)
    assert loads == [directory / "models" / graph["selection"]["final_refit_key"]]
    assert plan.node_requests[0]["request"]["graph_sha256"] == graph_sha256


def test_exported_request_mutation_cannot_change_private_reduction_plan(completed_chunks):
    plan, _ = completed_chunks
    original = plan.node_requests
    changed = plan.node_requests
    changed[0]["request"]["actual_K"] += 1
    assert plan.node_requests == original


def test_chunk_applies_native_one_to_the_real_numerical_gate(completed_chunks, actual_graph_case, monkeypatch):
    from threadpoolctl import threadpool_info

    plan, _ = completed_chunks
    directory, views, future, _, _ = actual_graph_case
    request = next(row for row in plan.node_requests if row["request"]["final"])
    calls = []
    original = owner._PrefixPackageAdmission.replay

    def track_native_threads(package, *args, **kwargs):
        calls.append(torch.get_num_threads())
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        assert kwargs["atol"] == kwargs["rtol"] == 0.0
        return original(package, *args, **kwargs)

    monkeypatch.setattr(owner._PrefixPackageAdmission, "replay", track_native_threads)
    with cpu_thread_budget(3):
        owner.replay_selection_chunk((request,), directory=directory, parent_prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=(future,))
        assert torch.get_num_threads() == 3
    assert calls == [1]


def test_reduction_summary_cannot_mutate_private_graph(completed_chunks):
    plan, results = completed_chunks
    with cpu_thread_budget(1):
        before = plan.finish_replay(results)
        changed = plan.finish_replay(results)
        changed["selected_candidate"]["name"] = "caller-mutation"
        changed["scores"][0]["decision_id"] = "caller-mutation"
        assert plan.finish_replay(results) == before


def test_prepared_plan_owns_original_graph_across_async_caller_mutation(actual_graph_case, completed_chunks):
    directory, views, future, graph_sha256, _ = actual_graph_case
    _, results = completed_chunks
    admitted = owner._SelectionReplayAdmission(directory, expected_graph_sha256=graph_sha256)
    plan = admitted.prepare_replay(prefix=views.calibration[800], prefix_compiler=views.prefix_compiler,
        future_views=(future,))
    requests = plan.node_requests
    with cpu_thread_budget(1):
        before = plan.finish_replay(results)
        assert plan._graph is admitted.graph
        with pytest.raises(TypeError, match="frozen"):
            admitted.graph["selection"]["baseline"] = "caller-mutation"
        with pytest.raises(TypeError, match="frozen"):
            admitted.graph["selection"]["outer_metrics"]["requested_participant_count"] += 1
        with pytest.raises(AttributeError):
            admitted.graph["selection"]["candidates"].clear()
        with pytest.raises(AttributeError):
            admitted.graph["nodes"].reverse()
        assert plan.node_requests == requests
        assert plan.finish_replay(results) == before


def test_prepared_graph_uses_external_exact_seal_without_reopening_original(tmp_path, monkeypatch):
    graph = {"schema": owner.SCHEMA, "selection": {"source": "original", "candidates": []}, "nodes": []}
    digest = semantic_sha256(graph)

    def no_original_read(path):
        raise AssertionError("prepared graph reopened the original large JSON")

    monkeypatch.setattr(Path, "read_bytes", no_original_read)
    admitted = owner._SelectionReplayAdmission.from_prepared_graph(tmp_path, graph,
        expected_graph_sha256=digest)
    assert semantic_sha256(admitted.graph) == semantic_sha256(graph) and admitted.graph_sha256 == digest
    changed = {**graph, "selection": {"source": "another", "candidates": []}}
    with pytest.raises(ValueError, match="graph"):
        owner._SelectionReplayAdmission.from_prepared_graph(tmp_path, changed,
            expected_graph_sha256=digest)


def test_prepared_graph_requires_the_external_seal(tmp_path):
    graph = {"schema": owner.SCHEMA, "selection": {}, "nodes": []}
    with pytest.raises(ValueError, match="external"):
        owner._SelectionReplayAdmission.from_prepared_graph(tmp_path, graph,
            expected_graph_sha256=None)


@pytest.mark.parametrize("source_kind", ["dict", "canonical_bytes"])
def test_prepared_graph_owns_a_private_snapshot_after_caller_mutation(tmp_path, source_kind):
    graph = {"schema": owner.SCHEMA, "selection": {"source": "original", "candidates": []},
        "nodes": [{"scope": {"actual_K": 0}}]}
    digest = semantic_sha256(graph)
    source = (graph if source_kind == "dict" else
        json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    admitted = owner._SelectionReplayAdmission.from_prepared_graph(tmp_path, source,
        expected_graph_sha256=digest)
    graph["selection"]["source"] = "changed"
    graph["nodes"][0]["scope"]["actual_K"] = 200
    assert admitted.graph["selection"]["source"] == "original"
    assert admitted.graph["nodes"][0]["scope"]["actual_K"] == 0
    assert semantic_sha256(admitted.graph) == digest
