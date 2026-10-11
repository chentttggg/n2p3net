"""Actual same-run whole-prefix selection graph, with zero-fit replay."""

from __future__ import annotations

import json
from pathlib import Path

from models.candidate_evidence import CandidateEvidence
from research.calibration_selection import (
    CalibrationProgramCandidate,
    FittedPrefixSelection,
    _metrics,
    _prefix_common_training_groups,
    select_calibration_candidate,
)
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, atomic_write_json
from research.prefix_inner_replay import (
    _PrefixPackageAdmission,
    _recursive_close,
    persist_prefix_inner_replay,
    prefix_input_identity,
)

SCHEMA = "n2p3_whole_prefix_selection_replay/1"


def _failure(value):
    return ExpectedSubjectError(value["error_code"], stage=value["stage"], detail=value["detail"])


def _load_graph(directory, expected_selection_sha256=None, expected_graph_sha256=None):
    directory = Path(directory)
    path = directory / "prefix_selection.json"
    if path.is_symlink():
        raise ValueError("nested prefix graph path is a symlink")
    payload = json.loads(path.read_bytes())
    graph = payload["record"]
    digest = semantic_sha256(graph)
    if (graph.get("schema") != SCHEMA or digest != payload["record_sha256"]
            or expected_graph_sha256 is not None and digest != expected_graph_sha256
            or expected_selection_sha256 is not None
            and semantic_sha256(graph["selection"]) != expected_selection_sha256):
        raise ValueError("nested prefix graph differs from externally sealed original selection")
    return graph


def _group_map(input_identity):
    return {row["record_id"]: row for row in input_identity["complete_groups"]}


def _scope_before_held(scope, held):
    input_identity = scope["input"]
    if (scope["actual_K"] != input_identity["budget"]
            or input_identity["target_contract"]["raw_scope"][1] > held["raw_scope"][0]
            or held["record_id"] in _group_map(input_identity)):
        raise ValueError("nested prefix fit includes its held observations or selection labels")


def _audit_graph(graph, node_records):
    selection = graph["selection"]
    candidates = tuple(CalibrationProgramCandidate.from_record(row) for row in selection["candidates"])
    parent = selection["parent_scope"]["input"]
    groups = _group_map(parent)
    if (selection["outer_scores_used_for_final_selection"] is not False
            or len(selection["outer_rows"]) != len(groups)
            or [row["held_record_id"] for row in selection["outer_rows"]] != list(groups)
            or semantic_sha256(selection["outer_requested"]) != semantic_sha256([row["group"] for row in groups.values()])):
        raise ValueError("nested prefix graph changed complete original outer denominator")
    if len({node["fit_key"] for node in graph["nodes"]}) != len(graph["nodes"]):
        raise ValueError("nested prefix graph repeats an actual fit key")
    declared = {candidate.name: candidate for candidate in candidates}
    for node in graph["nodes"]:
        record = node_records[node["fit_key"]]
        candidate = CalibrationProgramCandidate.from_record(node["candidate"])
        if (candidate.name not in declared or semantic_sha256(candidate.record()) != semantic_sha256(declared[candidate.name].record())
                or semantic_sha256(record["prefix_input"]) != semantic_sha256(node["scope"]["input"])
                or semantic_sha256(record["expert_programs"]) != semantic_sha256(candidate.record()["expert_programs"])
                or semantic_sha256(record["ensemble_program"]) != candidate.ensemble_program.digest()
                or set(record["programs"]) != {candidate.strategy}):
            raise ValueError("nested prefix actual node differs from declared programme/scope")
        for use in node["uses"]:
            for key in ("outer_held_record_id", "selection_held_record_id"):
                if use.get(key) is not None:
                    if use[key] not in groups:
                        raise ValueError("nested prefix use names an unauthorized held group")
                    _scope_before_held(node["scope"], groups[use[key]])

    required_groups = _prefix_common_training_groups(candidates)
    for outer in selection["outer_rows"]:
        _scope_before_held(outer["refit_scope"], groups[outer["held_record_id"]])
        chosen = CalibrationProgramCandidate.from_record(outer["selected_candidate"])
        node = next(row for row in graph["nodes"] if row["fit_key"] == outer["refit_key"])
        if (semantic_sha256(outer["selection"]["training_scope"]) != semantic_sha256(outer["refit_scope"])
                or chosen.name != outer["selection"]["decision"]["selected_candidate"]
                or semantic_sha256(chosen.record()) != semantic_sha256(node["candidate"])):
            raise ValueError("outer prefix meta-selection/refit read different scopes or programmes")
        _audit_selection(outer["selection"], declared, node_records, required_groups, selection, graph)
        _audit_score_row(outer, node_records, graph)
    _audit_selection(selection["final_selection"], declared, node_records, required_groups, selection, graph)
    final_node = next(row for row in graph["nodes"] if row["fit_key"] == selection["final_refit_key"])
    if (semantic_sha256(selection["parent_scope"]) != semantic_sha256(selection["final_refit_scope"])
            or semantic_sha256(selection["final_selection"]["training_scope"]) != semantic_sha256(selection["parent_scope"])
            or semantic_sha256(final_node["candidate"]) != semantic_sha256(selection["selected_candidate"])
            or semantic_sha256(node_records[selection["final_refit_key"]]["prefix_input"]) != semantic_sha256(parent)):
        raise ValueError("nested prefix selected final refit lost actual full-C dose")


def _audit_score_row(row, node_records, graph):
    key = row.get("fit_key", row.get("refit_key"))
    node = next(node for node in graph["nodes"] if node["fit_key"] == key)
    record = node_records[key]
    captured = [value for value in record["future_score_rows"]
                if value["input"]["view"]["record_id"] == row["held_record_id"]]
    if len(captured) != 1:
        raise ValueError("nested prefix held score lacks its actual captured producer")
    from transfer.ensemble import EnsembleProgram, EnsembleScorer

    programme = EnsembleProgram.from_record(record["ensemble_program"])
    evidence = EnsembleScorer(programme).score({name: CandidateEvidence.from_record(value)
        for name, value in captured[0]["expert_evidence"].items()})
    _recursive_close(evidence.record(), row["evidence"])
    if (semantic_sha256(row.get("training_scope", row.get("refit_scope"))["input"]) !=
            semantic_sha256(node["scope"]["input"])):
        raise ValueError("nested prefix held producer and actual training scope differ")


def _audit_selection(record, declared, node_records, required_groups, selection, graph):
    groups = _group_map(record["training_scope"]["input"])
    # Every physical group is represented either as predeclared structural
    # warm-up or a complete common candidate evaluation row. No numeric failure
    # can be reclassified as warm-up to shrink the requested selection set.
    scopes = {row["held_record_id"]: row["training_scope"] for row in record["structural_scope_rows"]}
    if list(scopes) != list(groups):
        raise ValueError("nested prefix structural scopes lost original groups")
    for name, scope in scopes.items():
        _scope_before_held(scope, groups[name])
        source_onsets = record["training_scope"]["input"]["raw"]["event_onsets"]
        held = groups[name]["view"]
        first = held["onsets"][0] + held["raw_scope"][0]
        expected_K = sum(onset < first for onset in source_onsets)
        derived = scope["input"]
        if (scope["actual_K"] != expected_K or expected_K > 0
                and list(derived["raw"]["event_onsets"]) != list(source_onsets[:expected_K])):
            raise ValueError("nested structural scope is not the actual prefix before its held group")
    expected = [name for name, scope in scopes.items()
                if len(scope["input"]["complete_groups"]) >= required_groups]
    if (record["minimum_training_groups"] != required_groups
            or record["minimum_selection_rows"] != selection["minimum_selection_rows"]
            or list(record["requested_held_record_ids"]) != expected
            or [row["held_record_id"] for row in record["structural_warmup_rows"]] != [name for name in groups if name not in expected]):
        raise ValueError("nested prefix common structural selection schedule changed")
    if len(expected) < selection["minimum_selection_rows"]:
        if (record["candidate_selection_rows"] or record["decision"]["candidate_metrics"]
                or record["decision"]["selected_candidate"] != selection["baseline"]):
            raise ValueError("insufficient prefix selection must retain its declared baseline")
        return
    if set(record["candidate_selection_rows"]) != set(declared):
        raise ValueError("nested prefix selection lost a declared candidate")
    for name, rows in record["candidate_selection_rows"].items():
        if [row["held_record_id"] for row in rows] != expected:
            raise ValueError("nested prefix candidate lost requested held probabilities/failures")
        for row in rows:
            _scope_before_held(row["training_scope"], groups[row["held_record_id"]])
            node = node_records[row["fit_key"]]
            if (semantic_sha256(node["prefix_input"]) != semantic_sha256(row["training_scope"]["input"])
                    or semantic_sha256(row["training_scope"]) != semantic_sha256(scopes[row["held_record_id"]])
                    or semantic_sha256(node["ensemble_program"]) != declared[name].ensemble_program.digest()
                    or semantic_sha256(node["expert_programs"]) != semantic_sha256(declared[name].record()["expert_programs"])
                    or set(node["programs"]) != {declared[name].strategy}):
                raise ValueError("nested prefix candidate held score changed its real producer")
            _audit_score_row(row, node_records, graph)
    metrics = record["decision"]["candidate_metrics"]
    if set(metrics) != set(declared):
        raise ValueError("nested prefix policy lost original candidate metrics")


def persist_prefix_selection(selection, directory):
    fits = selection._fits
    if fits is None or selection.record["final_refit_key"] not in fits.nodes:
        raise ValueError("nested prefix graph requires the original same-run capture")
    final_node = fits.nodes[selection.record["final_refit_key"]]
    strategy = final_node["candidate"].strategy
    if final_node["programs"][strategy] is not selection.final:
        raise ValueError("nested prefix final object differs from same-run refit")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    nodes = []
    for key, node in sorted(fits.nodes.items()):
        path = directory / "models" / key
        persist_prefix_inner_replay(node["prefix"], node["programs"], path,
            recorder=node["recorder"], source_checkpoints=node["sources"], context=node["context"], device=fits.device)
        nodes.append({"fit_key": key, "package": f"models/{key}",
            "candidate": node["candidate"].record(),
            "scope": {"actual_K": node["prefix"].budget, "input": prefix_input_identity(node["prefix"])},
            "uses": fits.uses[key], "prefix_receipt_sha256": node["recorder"].prefix_receipt_sha256})
    graph = {"schema": SCHEMA, "selection": selection.record, "nodes": nodes,
        "same_run_producers": True, "zero_fit_replay": True}
    atomic_write_json(directory / "prefix_selection.json", {"record": graph, "record_sha256": semantic_sha256(graph)})
    selection.graph_sha256 = semantic_sha256(graph)
    return directory


def _node_path(directory, node):
    key = node["fit_key"]
    if (len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
            or node["package"] != f"models/{key}"):
        raise ValueError("nested prefix producer package path differs from actual fit key")
    path = Path(directory) / "models" / key
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("nested prefix producer path is a symlink")
    return path


class _SelectionReplayAdmission:
    """One private graph snapshot, independently opened by each consumer."""

    def __init__(self, directory, *, expected_selection_sha256=None, expected_graph_sha256=None):
        self.directory = Path(directory)
        self.graph = _load_graph(self.directory, expected_selection_sha256, expected_graph_sha256)

    def load(self):
        return _load_admitted_selection(self)

    def replay(self, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
        return _replay_admitted_selection(self, prefix=prefix, prefix_compiler=prefix_compiler,
                                         future_views=future_views, source_checkpoints=source_checkpoints)


def _loaded_final_selection(graph, final_record, final_programs):
    record = graph["selection"]
    selected = CalibrationProgramCandidate.from_record(record["selected_candidate"])
    entry = final_record["programs"][selected.strategy]
    final = (final_programs[selected.strategy] if entry["status"] == "completed"
             else _failure(entry["failure"]))
    return FittedPrefixSelection(final, record)


def _load_admitted_selection(admission):
    graph, directory = admission.graph, admission.directory
    records, final_programs = {}, None
    for node in graph["nodes"]:
        path = _node_path(directory, node)
        package = _PrefixPackageAdmission(path, expected_record_sha256=node["prefix_receipt_sha256"])
        package.replay()
        records[node["fit_key"]] = package.record
        if node["fit_key"] == graph["selection"]["final_refit_key"]:
            final_programs = package.programs
    _audit_graph(graph, records)
    return _loaded_final_selection(graph, records[graph["selection"]["final_refit_key"]], final_programs)


def _replay_admitted_selection(admission, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
    graph, directory = admission.graph, admission.directory
    prefix_compiler.assert_original_prefix(prefix)
    if semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(graph["selection"]["parent_scope"]["input"]):
        raise ValueError("nested prefix replay original parent observations/labels changed")
    future_views = tuple(future_views)
    parent_groups = {group.view.record_id: group for group in prefix.complete_groups}
    records, results, final_programs = {}, {}, None
    for node in graph["nodes"]:
        path = _node_path(directory, node)
        current = prefix_compiler.compile(node["scope"]["actual_K"])
        package = _PrefixPackageAdmission(path, expected_record_sha256=node["prefix_receipt_sha256"])
        record = package.record
        views = future_views if node["fit_key"] == graph["selection"]["final_refit_key"] else tuple(
            parent_groups[name].view for name in record["fit_context"]["requested_decision_ids"])
        candidate = CalibrationProgramCandidate.from_record(node["candidate"])
        sources = {} if source_checkpoints is None else {name: source_checkpoints[name]
            for name in candidate.ensemble_program.expert_ids if name in source_checkpoints}
        results[node["fit_key"]] = package.replay(prefix=current, future_views=views, source_checkpoints=sources)
        records[node["fit_key"]] = record
        if node["fit_key"] == graph["selection"]["final_refit_key"]:
            final_programs = package.programs
    _audit_graph(graph, records)
    selection = graph["selection"]
    # Recompute the actual policy on the same saved h probabilities/truths.
    # Truth is C-only and is never supplied to a scorer or numerical fitter.
    traces = [outer["selection"] for outer in selection["outer_rows"]] + [selection["final_selection"]]
    for trace in traces:
        for row in trace["structural_scope_rows"]:
            actual = prefix_compiler.before_group(prefix, parent_groups[row["held_record_id"]])
            if semantic_sha256(prefix_input_identity(actual)) != semantic_sha256(row["training_scope"]["input"]):
                raise ValueError("nested prefix structural scope differs from original owner compilation")
        if not trace["candidate_selection_rows"]:
            continue
        held = tuple(parent_groups[name] for name in trace["requested_held_record_ids"])
        metrics = {name: _metrics(held, [CandidateEvidence.from_record(row["evidence"]) for row in rows])
            for name, rows in trace["candidate_selection_rows"].items()}
        name, decision = select_calibration_candidate(metrics, baseline_name=selection["baseline"],
            tie_order=tuple(candidate["name"] for candidate in selection["candidates"]))
        if name != trace["decision"]["selected_candidate"]:
            raise ValueError("nested prefix actual selection decision does not replay")
        _recursive_close(decision, trace["decision"])
    # All nodes were independently admitted and numerically replayed above.
    # The final loaded-object relationship uses those same saved states; it
    # does not reopen the graph or reload every final/inner predictor.
    _loaded_final_selection(graph, records[selection["final_refit_key"]], final_programs)
    return {"status": "replayed", "actual_K": prefix.budget, "producer_count": len(graph["nodes"]),
        "exact_score_checks": sum(row["exact_score_checks"] for row in results.values()),
        "scores": results[selection["final_refit_key"]]["scores"], "zero_new_fits": True,
        "outer_requested": len(selection["outer_requested"]), "selected_candidate": selection["selected_candidate"]}


def load_prefix_selection(directory, *, expected_selection_sha256=None, expected_graph_sha256=None):
    admission = _SelectionReplayAdmission(directory, expected_selection_sha256=expected_selection_sha256,
                                          expected_graph_sha256=expected_graph_sha256)
    return admission.load()


def replay_prefix_selection(directory, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None,
                            expected_selection_sha256=None, expected_graph_sha256=None):
    """Rebuild original scopes and independently replay every saved producer."""
    admission = _SelectionReplayAdmission(directory, expected_selection_sha256=expected_selection_sha256,
                                          expected_graph_sha256=expected_graph_sha256)
    return admission.replay(prefix=prefix, prefix_compiler=prefix_compiler,
                            future_views=future_views, source_checkpoints=source_checkpoints)
