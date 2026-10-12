"""One ingress reuses admitted saved state; new callers still re-admit bytes."""
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.calibration_selection import CalibrationProgramCandidate, fit_nested_prefix_selection
from research.contracts import semantic_sha256
from research.prefix_selection_replay import load_prefix_selection, replay_prefix_selection
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_views import prepare_bnci_independent_prefixes


@pytest.fixture(scope="module")
def graph_case(tmp_path_factory):
    root, _, path, spec, experts, old_views = independent_prefix_case(tmp_path_factory.mktemp("admitted-prefix"))
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path, source_sha256=old_views.ledger["source_sha256"],
            budgets=(0, 800), profiles=(experts[1].profile,), authorization_preprocessing=spec)
        raw = replace(experts[0], n_basis=1)
        roster = EnsembleProgram.equal_weights("replay-admission", (raw.expert_id, experts[1].expert_id))
        candidates = (CalibrationProgramCandidate("fixed", (raw, experts[1]), roster),
            CalibrationProgramCandidate("grouped", (raw, experts[1]), roster, "grouped_temperature"))
        future = next(view for view in views.decisions if view.samples is not None)
        selected = fit_nested_prefix_selection(views.calibration[800], candidates, baseline_name="fixed",
            prefix_compiler=views.prefix_compiler, minimum_selection_rows=1,
            context={"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]})
        expected = selected.score(future).record()
        directory = selected.save(root / "selection")
    return directory, views, future, selected.graph_sha256, expected


@pytest.fixture(autouse=True)
def no_new_fit(monkeypatch):
    import transfer.ensemble_fit as fitter

    def forbidden(*args, **kwargs):
        raise AssertionError("admitted package replay attempted a numerical fit")

    monkeypatch.setattr(fitter, "_fit_expert", forbidden)


def test_graph_replay_reads_each_saved_package_once_and_keeps_actual_score_checks(graph_case, monkeypatch):
    directory, views, future, graph_digest, expected = graph_case
    graph = json.loads((directory / "prefix_selection.json").read_bytes())["record"]
    reads = Counter()
    original = Path.read_bytes

    def read_once(path):
        if path.is_relative_to(directory):
            reads[path] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    with cpu_thread_budget(1):
        result = replay_prefix_selection(directory, prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=(future,), expected_graph_sha256=graph_digest)
    assert reads[directory / "prefix_selection.json"] == 1
    expected_checks = 0
    inner_count = 0
    for node in graph["nodes"]:
        package = directory / node["package"]
        record = json.loads(original(package / "prefix_inner_replay.json"))["record"]
        assert reads[package / "prefix_inner_replay.json"] == 1
        assert reads[package / "temperature_programs.json"] == 1
        if record["package_hashes"]["ensemble_json_sha256"] is not None:
            assert reads[package / "ensemble.json"] == 1
        expected_checks += sum(len(row["oof_rows"]) for row in record["inner_producers"])
        expected_checks += len(record["future_score_rows"])
        for producer in record["inner_producers"]:
            if producer["status"] == "completed":
                inner_count += 1
                assert reads[package / producer["package"] / "ensemble.json"] == 1
    assert inner_count > 0
    assert result["exact_score_checks"] == expected_checks
    assert result["producer_count"] == len(graph["nodes"])
    strategy = graph["selection"]["selected_candidate"]["strategy"]
    assert result["scores"][0]["strategies"][strategy] == expected
    assert result["zero_new_fits"]


def test_new_ingress_rejects_changed_model_even_with_same_path_size_and_mtime(graph_case):
    directory, _, _, graph_digest, _ = graph_case
    with cpu_thread_budget(1):
        load_prefix_selection(directory, expected_graph_sha256=graph_digest)
    model_path = next(path for path in directory.rglob("*.pt"))
    contents, stat = model_path.read_bytes(), model_path.stat()
    changed = bytearray(contents)
    changed[-1] ^= 1
    model_path.write_bytes(changed)
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    try:
        with cpu_thread_budget(1), pytest.raises(ValueError, match="model bytes changed"):
            load_prefix_selection(directory, expected_graph_sha256=graph_digest)
    finally:
        model_path.write_bytes(contents)
        os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_new_ingress_rejects_self_resigned_graph_against_external_seal(graph_case):
    directory, _, _, graph_digest, _ = graph_case
    path = directory / "prefix_selection.json"
    before = path.read_bytes()
    payload = json.loads(before)
    payload["record"]["selection"]["minimum_selection_rows"] += 1
    payload["record_sha256"] = semantic_sha256(payload["record"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="externally sealed original selection"):
            load_prefix_selection(directory, expected_graph_sha256=graph_digest)
    finally:
        path.write_bytes(before)


def test_returned_predictor_state_cannot_poison_another_ingress(graph_case):
    directory, _, _, graph_digest, _ = graph_case
    with cpu_thread_budget(1):
        first = load_prefix_selection(directory, expected_graph_sha256=graph_digest)
        second = load_prefix_selection(directory, expected_graph_sha256=graph_digest)
    first_trial = next(expert.predictor for expert in first.final._experts if expert.program.family == "trial")
    second_trial = next(expert.predictor for expert in second.final._experts if expert.program.family == "trial")
    assert first_trial is not second_trial
    name = next(iter(first_trial._state))
    original = second_trial._state[name].clone()
    first_trial._state[name].add_(1)
    assert torch.equal(second_trial._state[name], original)
    with cpu_thread_budget(1):
        third = load_prefix_selection(directory, expected_graph_sha256=graph_digest)
    third_trial = next(expert.predictor for expert in third.final._experts if expert.program.family == "trial")
    assert torch.equal(third_trial._state[name], original)
