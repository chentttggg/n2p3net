"""Actual same-run whole-prefix selection graph, with zero-fit replay."""

from __future__ import annotations

import json
from pathlib import Path

from models.candidate_evidence import CandidateEvidence
from research.calibration_selection import (
    CalibrationProgramCandidate,
    FittedPrefixSelection,
    _check_candidate_temperature_subset,
    _metrics,
    _prefix_common_training_groups,
    select_calibration_candidate,
)
from research.contracts import canonical_json_bytes, semantic_sha256
from research.execution import ExpectedSubjectError
from research.frozen_metadata import plain_metadata
from research.prefix_inner_replay import (
    _manifest,
    _normalise_sources,
    _PrefixPackageAdmission,
    _PrefixScopeOwner,
    _recursive_close,
    _SourceSeedAdmission,
    future_input_identity,
    persist_prefix_inner_replay,
    prefix_input_identity,
)
from research.selection_graph_store import SelectionGraphStore, save_selection_graph
from train.runtime import cpu_thread_budget
from transfer.noise_model import freeze_metadata

SCHEMA = "n2p3_whole_prefix_selection_replay/1"
NODE_REQUEST_SCHEMA = "n2p3_selection_replay_node/1"
NODE_RESULT_SCHEMA = "n2p3_selection_replay_node_result/1"


def _failure(value):
    return ExpectedSubjectError(value["error_code"], stage=value["stage"], detail=value["detail"])


def _load_graph(directory, expected_selection_sha256=None, expected_graph_sha256=None):
    store = SelectionGraphStore(Path(directory) / "prefix_selection.json",
        expected_graph_sha256=expected_graph_sha256, expected_selection_sha256=expected_selection_sha256)
    return store.graph, store.graph_sha256


def _admit_graph_record(graph, record_sha256, *, expected_selection_sha256=None, expected_graph_sha256=None):
    graph = freeze_metadata(graph)
    digest = semantic_sha256(graph)
    if (graph.get("schema") != SCHEMA or digest != record_sha256
            or expected_graph_sha256 is not None and digest != expected_graph_sha256
            or expected_selection_sha256 is not None
            and semantic_sha256(graph["selection"]) != expected_selection_sha256):
        raise ValueError("nested prefix graph differs from externally sealed original selection")
    return graph, digest


def _group_map(input_identity):
    return {row["record_id"]: row for row in input_identity["complete_groups"]}


def _scope_before_held(scope, held):
    input_identity = scope["input"]
    target = input_identity["target_contract"]
    if target.get("authority_kind") == "record_aware_calibration_prefix/1":
        matches = [record for record in target["record_scopes"].values()
                   if record["source_sha256"] == held["view"]["source_sha256"]
                   and record["source_relative_path"] == held["view"]["source_relative_path"]]
        if (len(matches) > 1 or matches and matches[0]["read_sample_support"][1] > held["raw_scope"][0]
                or scope["actual_K"] != input_identity["budget"]
                or held["record_id"] in _group_map(input_identity)):
            raise ValueError("nested prefix record fit includes its held observations or selection labels")
        return
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
        programme = record["programs"][candidate.strategy]
        if programme["status"] == "completed":
            _check_candidate_temperature_subset(candidate, programme["temperature_calibration"],
                beta_calibration=programme.get('beta_calibration'))
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
    evidence = EnsembleScorer(programme, record["programs"][captured[0]["strategy"]].get("pool_calibration")).score({name: CandidateEvidence.from_record(value)
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
        original = record["training_scope"]["input"]
        held = groups[name]["view"]
        derived = scope["input"]
        if original["target_contract"].get("authority_kind") == "record_aware_calibration_prefix/1":
            _audit_record_before_held(original, scope, held)
        else:
            source_onsets = original["raw"]["event_onsets"]
            first = held["onsets"][0] + held["raw_scope"][0]
            expected_K = sum(onset < first for onset in source_onsets)
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


def _audit_record_before_held(original, scope, held):
    """Independently join the saved earlier K to original record-local operands."""
    original_scopes = original["target_contract"]["record_scopes"]
    matches = [(key, record) for key, record in original_scopes.items()
               if record["source_sha256"] == held["source_sha256"]
               and record["source_relative_path"] == held["source_relative_path"]]
    if len(matches) != 1:
        raise ValueError("nested held group lacks one original physical record authority")
    key, authority = matches[0]
    native = next(row for row in original["native_records"] if row["record_id"] == key)
    onsets = native["event_onsets"]
    count = sum(onset < held["raw_scope"][0] for onset in onsets)
    base = authority["global_label_indices"][0]
    expected_K = base + count
    actual = scope["input"]
    derived_scopes = actual["target_contract"]["record_scopes"]
    expected_records = {name: record for name, record in original_scopes.items()
                        if record["global_label_indices"][1] <= base}
    if count:
        expected_records[key] = authority
    if scope["actual_K"] != expected_K or set(derived_scopes) != set(expected_records):
        raise ValueError("nested record structural scope differs from actual earlier global K")
    original_native = {row["record_id"]: row for row in original["native_records"]}
    derived_native = {row["record_id"]: row for row in actual["native_records"]}
    if set(derived_native) != set(expected_records):
        raise ValueError("nested earlier scope changed its original native record roster")
    for name, record in expected_records.items():
        derived = derived_scopes[name]
        for field in ("source_sha256", "source_relative_path", "source_dataset_id", "source_subject_id",
                      "native_clock", "record_id"):
            if derived[field] != record[field]:
                raise ValueError("nested earlier prefix changed an original record identity")
        if name == key:
            if (tuple(derived["global_label_indices"]) != (base, expected_K)
                    or list(derived_native[name]["event_onsets"]) != list(onsets[:count])
                    or derived["read_sample_support"][1] > held["raw_scope"][0]):
                raise ValueError("nested earlier record scope retains held labels or observations")
        elif (semantic_sha256(derived) != semantic_sha256(record)
                or semantic_sha256(derived_native[name]) != semantic_sha256(original_native[name])):
            raise ValueError("nested earlier prefix changed an already authorized original record")


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
    nodes, seeds = [], _SourceSeedAdmission()
    for key, node in sorted(fits.nodes.items()):
        path = directory / "models" / key
        persist_prefix_inner_replay(node["prefix"], node["programs"], path,
            recorder=node["recorder"], source_checkpoints=node["sources"], context=node["context"], device=fits.device,
            _source_seed_admission=seeds)
        nodes.append({"fit_key": key, "package": f"models/{key}",
            "candidate": node["candidate"].record(),
            "scope": fits.scope_owner.scope(node["prefix"]),
            "uses": fits.uses[key], "prefix_receipt_sha256": node["recorder"].prefix_receipt_sha256})
    graph = {"schema": SCHEMA, "selection": selection.record, "nodes": nodes,
        "same_run_producers": True, "zero_fit_replay": True}
    from research.source_metadata_store import collect_source_dependencies

    dependencies = collect_source_dependencies(directory,
        {package.parent for name in ("ensemble.json", "prefix_inner_replay.json") for package in directory.rglob(name)},
        include_joint_prefixes=True)
    if dependencies is not None:
        graph["source_metadata"] = dependencies
    graph = freeze_metadata(graph)
    selection.graph_sha256 = semantic_sha256(graph)
    save_selection_graph(directory / "prefix_selection.json", graph)
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

    def __init__(self, directory, *, expected_selection_sha256=None, expected_graph_sha256=None,
                 metadata_store=None, _metadata_admission=None, _joint_source_preparation=None):
        self.directory = Path(directory)
        self.graph, self.graph_sha256 = _load_graph(self.directory, expected_selection_sha256, expected_graph_sha256)
        self._source_seed_admission = _SourceSeedAdmission()
        self._admit_source_closure(metadata_store=metadata_store, admission=_metadata_admission)
        self._admit_joint_source(_joint_source_preparation)

    def _admit_joint_source(self, preparation=None):
        if preparation is None and any(spec["family"] == "joint_flat"
                for candidate in self.graph["selection"]["candidates"] for spec in candidate["expert_programs"]):
            from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

            preparation = FrozenJointFlatSourcePreparation()
        self._joint_source_preparation = preparation

    def _admit_source_closure(self, *, metadata_store=None, admission=None):
        from research.source_metadata_store import (
            _SourceMetadataAdmission,
            admit_source_dependencies,
            collect_source_dependencies,
        )

        descriptor = plain_metadata(self.graph.get("source_metadata"))
        actual = collect_source_dependencies(self.directory,
            {package.parent for name in ("ensemble.json", "prefix_inner_replay.json")
             for package in self.directory.rglob(name)}, include_joint_prefixes=True)
        if semantic_sha256(actual) != semantic_sha256(descriptor):
            raise ValueError("selection graph source closure differs from its actual numerical packages")
        self._metadata_admission = (admission or _SourceMetadataAdmission(store=metadata_store)
            if descriptor is None else admit_source_dependencies(descriptor, self.directory,
                store=metadata_store, admission=admission))

    @classmethod
    def from_prepared_graph(cls, directory, graph, *, expected_graph_sha256, expected_selection_sha256=None,
                            _metadata_admission=None, _joint_source_preparation=None):
        """Open a private graph snapshot from the original canonical seal."""
        if expected_graph_sha256 is None:
            raise ValueError("prepared graph requires its external original seal")
        if isinstance(graph, SelectionGraphStore):
            store = graph
            if (store.graph_sha256 != expected_graph_sha256 or expected_selection_sha256 is not None
                    and store.selection_sha256 != expected_selection_sha256):
                raise ValueError("prepared graph differs from its external original seal")
        elif isinstance(graph, bytes):
            store = SelectionGraphStore.from_bytes(graph, directory=directory,
                expected_graph_sha256=expected_graph_sha256, expected_selection_sha256=expected_selection_sha256)
        else:
            store = SelectionGraphStore.from_record(graph, directory=directory,
                expected_graph_sha256=expected_graph_sha256, expected_selection_sha256=expected_selection_sha256)
        admission = cls.__new__(cls)
        admission.directory, admission.graph, admission.graph_sha256 = Path(directory), store.graph, store.graph_sha256
        admission._source_seed_admission = _SourceSeedAdmission()
        admission._admit_source_closure(admission=_metadata_admission)
        admission._admit_joint_source(_joint_source_preparation)
        return admission

    def load(self):
        return _load_admitted_selection(self)

    def replay(self, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
        return _replay_admitted_selection(self, prefix=prefix, prefix_compiler=prefix_compiler,
                                         future_views=future_views, source_checkpoints=source_checkpoints)

    def prepare_replay(self, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
        """Prepare the graph once; export only sealed, ordered node requests."""
        return SelectionReplayPlan(self, prefix=prefix, prefix_compiler=prefix_compiler,
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
        package = _PrefixPackageAdmission(path, expected_record_sha256=node["prefix_receipt_sha256"],
                                           _metadata_admission=admission._metadata_admission,
                                           _joint_source_preparation=admission._joint_source_preparation,
                                           _source_seed_admission=admission._source_seed_admission)
        if package.record.get("schema") != "n2p3_joint_flat_prefix_inner_replay/1":
            package.replay()
        records[node["fit_key"]] = package.record
        if node["fit_key"] == graph["selection"]["final_refit_key"]:
            final_programs = package.programs
    _audit_graph(graph, records)
    return _loaded_final_selection(graph, records[graph["selection"]["final_refit_key"]], final_programs)


def _selection_node_manifest(directory, *, expected_record_sha256):
    """Read an externally sealed reducer record without reopening numeric state."""
    directory = Path(directory)
    path = directory / "prefix_inner_replay.json"
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("selection node manifest path is a symlink")
    payload = json.loads(path.read_bytes())
    if payload.get("record", {}).get("schema") != "n2p3_joint_flat_prefix_inner_replay/1":
        return _manifest(directory, expected_record_sha256=expected_record_sha256)
    record = payload["record"]
    if semantic_sha256(record) != payload["record_sha256"] or payload["record_sha256"] != expected_record_sha256:
        raise ValueError("joint selection node differs from its original externally sealed receipt")
    return record


def _original_parent(prefix, prefix_compiler, scope_owner=None):
    prefix_compiler.assert_original_prefix(prefix)
    identity = prefix_input_identity(prefix) if scope_owner is None else scope_owner.scope(prefix)["input"]
    return semantic_sha256(identity)


def _seal_node_request(record):
    return {"request": record, "request_sha256": semantic_sha256(record)}


def _read_node_request(envelope):
    request = envelope["request"]
    if (request.get("schema") != NODE_REQUEST_SCHEMA
            or envelope.get("request_sha256") != semantic_sha256(request)):
        raise ValueError("selection node request digest changed")
    _node_path(".", request)
    return request


def _requested_views(node, *, final, future_views, parent_groups):
    if final:
        return future_views
    names = dict.fromkeys(use.get("selection_held_record_id", use.get("outer_held_record_id"))
        for use in node["uses"])
    if not names or None in names or set(names) - set(parent_groups):
        raise ValueError("selection node use lacks its original parent held record")
    return tuple(parent_groups[name].view for name in names)


def _view_bindings(views):
    return [{"record_id": view.record_id, "input_sha256": semantic_sha256(future_input_identity(view))}
        for view in views]


def _saved_replay_result(record, request):
    """Reconstruct the original receipt rows, without admitting a checked flag."""
    from transfer.ensemble import EnsembleProgram, EnsembleScorer

    captured = {(row["strategy"], row["input"]["view"]["record_id"]): row
        for row in record["future_score_rows"]}
    expected = {(strategy, row["record_id"]) for strategy in record["programs"] for row in request["views"]}
    if len(captured) != len(record["future_score_rows"]) or set(captured) != expected:
        raise ValueError("selection node result lost the original requested score roster")
    program = EnsembleProgram.from_record(record["ensemble_program"])
    scores = []
    for view in request["views"]:
        row = {"decision_id": None, "strategies": {}}
        for strategy in record["programs"]:
            saved = captured[strategy, view["record_id"]]
            if semantic_sha256(saved["input"]) != view["input_sha256"]:
                raise ValueError("selection node result requested observations changed")
            row["decision_id"] = saved["input"]["view"]["decision_id"]
            scorer = EnsembleScorer(program, record["programs"][strategy].get("pool_calibration"))
            row["strategies"][strategy] = scorer.score({name: CandidateEvidence.from_record(value)
                for name, value in saved["expert_evidence"].items()}).record()
        scores.append(row)
    return {"status": "replayed", "actual_K": record["actual_K"], "requested_Ks": record["requested_Ks"],
        "producer_count": len(record["inner_producers"]),
        "exact_score_checks": sum(len(row["oof_rows"]) for row in record["inner_producers"])
            + len(record["future_score_rows"]),
        "preserved_oof_score_failures": [{"fit_key": producer["fit_key"], **failure}
            for producer in record["inner_producers"] for failure in producer.get("oof_score_failures", ())],
        "scores": scores, "truth_free": True}


def _replay_selection_node(envelope, *, directory, parent_prefix, prefix_compiler,
                           future_views, source_checkpoints, source_sha256, parent_sha256,
                           _metadata_admission=None, _joint_source_preparation=None,
                           _scope_owner=None, _source_seed_admission=None):
    request = _read_node_request(envelope)
    if request["parent_input_sha256"] != parent_sha256:
        raise ValueError("selection node request changed its original parent scope")
    current = prefix_compiler.compile(request["actual_K"])
    identity = prefix_input_identity(current) if _scope_owner is None else _scope_owner.scope(current)["input"]
    if (semantic_sha256(identity) != request["prefix_input_sha256"]
            or semantic_sha256({"actual_K": current.budget, "input": identity}) != request["scope_sha256"]):
        raise ValueError("selection node request changed its actual compiled scope")
    parent_groups = {group.view.record_id: group for group in parent_prefix.complete_groups}
    if not request["final"] and any(row["record_id"] not in parent_groups for row in request["views"]):
        raise ValueError("selection node request names an unauthorized original parent held record")
    views = future_views if request["final"] else tuple(parent_groups[row["record_id"]].view
        for row in request["views"])
    if _view_bindings(views) != request["views"]:
        raise ValueError("selection node request changed its original requested observations")
    candidate = CalibrationProgramCandidate.from_record(request["candidate"])
    sources = {name: source_checkpoints[name] for name in candidate.ensemble_program.expert_ids
        if name in source_checkpoints}
    if {name: source_sha256[name] for name in sources} != request["source_checkpoint_sha256"]:
        raise ValueError("selection node request changed its original source checkpoints")
    path = _node_path(directory, request)
    package = _PrefixPackageAdmission(path, expected_record_sha256=request["prefix_receipt_sha256"],
                                     _metadata_admission=_metadata_admission,
                                     _joint_source_preparation=_joint_source_preparation,
                                     _source_seed_admission=_source_seed_admission, _scope_owner=_scope_owner)
    record = package.record
    if (semantic_sha256(record["prefix_input"]) != request["prefix_input_sha256"]
            or semantic_sha256(record["expert_programs"]) != semantic_sha256(candidate.record()["expert_programs"])
            or semantic_sha256(record["ensemble_program"]) != candidate.ensemble_program.digest()
            or set(record["programs"]) != {candidate.strategy}
            or record["fit_context"].get("requested_decision_ids") != [view.record_id for view in views]):
        raise ValueError("selection node request differs from its original producer declaration")
    programme = record["programs"][candidate.strategy]
    if programme["status"] == "completed":
        _check_candidate_temperature_subset(candidate, programme["temperature_calibration"],
            beta_calibration=programme.get('beta_calibration'))
    # Captured device remains a fit identity. Loaded predictors replay on CPU;
    # passing device='cpu' would incorrectly rebind a historical CUDA fit.
    result = package.replay(prefix=current, future_views=views, prefix_compiler=prefix_compiler, source_checkpoints=sources,
        atol=0.0, rtol=0.0)
    result_record = {"schema": NODE_RESULT_SCHEMA, "request": envelope,
        "prefix_record_ref": f"{request['package']}/prefix_inner_replay.json",
        "prefix_record_sha256": semantic_sha256(record), "replay": result,
        "captured_fit_device": record["device"], "replay_device": "cpu", "native_threads": 1,
        "atol": 0.0, "rtol": 0.0}
    return {"record": result_record, "record_sha256": semantic_sha256(result_record)}, package


def replay_selection_chunk(requests, *, directory, parent_prefix, prefix_compiler,
                           future_views=(), source_checkpoints=None, _joint_source_preparation=None):
    """Independently replay sealed nodes at native1/exact0, without a graph load."""
    requests = tuple(requests)
    if not requests:
        raise ValueError("selection replay chunk requires at least one original node")
    records = [_read_node_request(request) for request in requests]
    keys = [(row["node_index"], row["fit_key"]) for row in records]
    if (len(set(keys)) != len(keys) or len({row["fit_key"] for row in records}) != len(records)
            or len({(row["graph_sha256"], row["selection_sha256"], row["parent_input_sha256"])
                for row in records}) != 1):
        raise ValueError("selection replay chunk repeats a node or mixes original graphs")
    with cpu_thread_budget(1):
        from research.source_metadata_store import _SourceMetadataAdmission

        metadata_admission = _SourceMetadataAdmission()
        scope_owner, seed_admission = _PrefixScopeOwner(prefix_compiler), _SourceSeedAdmission()
        joint_source_preparation = _joint_source_preparation
        if joint_source_preparation is None and any(
                spec["family"] == "joint_flat" for row in records for spec in row["candidate"]["expert_programs"]):
            from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

            joint_source_preparation = FrozenJointFlatSourcePreparation()
        parent_sha256 = _original_parent(parent_prefix, prefix_compiler, scope_owner)
        future_views, sources = tuple(future_views), _normalise_sources(source_checkpoints)
        source_sha256 = {name: source.digest for name, source in sources.items()}
        return [_replay_selection_node(request, directory=directory, parent_prefix=parent_prefix,
            prefix_compiler=prefix_compiler, future_views=future_views, source_checkpoints=sources,
            source_sha256=source_sha256, parent_sha256=parent_sha256,
            _metadata_admission=metadata_admission, _joint_source_preparation=joint_source_preparation,
            _scope_owner=scope_owner, _source_seed_admission=seed_admission)[0]
            for request in requests]


class SelectionReplayPlan:
    """One private original graph and complete ordered numerical reduction."""

    def __init__(self, admission, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
        self._graph, self._directory = freeze_metadata(admission.graph), admission.directory
        self._metadata_admission = admission._metadata_admission
        self._joint_source_preparation = admission._joint_source_preparation
        self._scope_owner = _PrefixScopeOwner(prefix_compiler)
        self._source_seed_admission = admission._source_seed_admission
        self._prefix, self._compiler = prefix, prefix_compiler
        self._future_views = tuple(future_views)
        self._sources = _normalise_sources(source_checkpoints)
        self._source_sha256 = {name: source.digest for name, source in self._sources.items()}
        parent_sha256 = _original_parent(prefix, prefix_compiler, self._scope_owner)
        selection = self._graph["selection"]
        if parent_sha256 != semantic_sha256(selection["parent_scope"]["input"]):
            raise ValueError("nested prefix replay original parent observations/labels changed")
        if self._source_sha256 != selection["source_checkpoint_sha256"]:
            raise ValueError("nested prefix replay original source checkpoint roster changed")
        parent_groups = {group.view.record_id: group for group in prefix.complete_groups}
        selection_sha256 = semantic_sha256(selection)
        requests = []
        for index, node in enumerate(self._graph["nodes"]):
            _node_path(self._directory, node)
            final = node["fit_key"] == selection["final_refit_key"]
            candidate = CalibrationProgramCandidate.from_record(node["candidate"])
            views = _requested_views(node, final=final, future_views=self._future_views,
                parent_groups=parent_groups)
            request = {"schema": NODE_REQUEST_SCHEMA, "graph_sha256": admission.graph_sha256,
                "selection_sha256": selection_sha256, "node_index": index, "fit_key": node["fit_key"],
                "package": node["package"], "prefix_receipt_sha256": node["prefix_receipt_sha256"],
                "candidate": node["candidate"], "scope_sha256": semantic_sha256(node["scope"]),
                "prefix_input_sha256": semantic_sha256(node["scope"]["input"]),
                "actual_K": node["scope"]["actual_K"], "parent_input_sha256": parent_sha256,
                "final": final, "views": _view_bindings(views),
                "source_checkpoint_sha256": {name: self._source_sha256[name]
                    for name in candidate.ensemble_program.expert_ids if name in self._sources}}
            requests.append(canonical_json_bytes(_seal_node_request(request)))
        if (not requests or len({node["fit_key"] for node in self._graph["nodes"]}) != len(requests)
                or sum(node["fit_key"] == selection["final_refit_key"] for node in self._graph["nodes"]) != 1):
            raise ValueError("selection replay plan needs its complete unique original node roster")
        self._requests = tuple(requests)
        self._serial_records, self._serial_final_programs = {}, None

    @property
    def node_requests(self):
        """Small JSON values cannot mutate this plan's private request seals."""
        return tuple(json.loads(content) for content in self._requests)

    def finish_replay(self, verified_results):
        """Reduce complete independently executed nodes in original graph order."""
        requests = self.node_requests
        expected = {request["request_sha256"]: request for request in requests}
        outcomes = {}
        for envelope in verified_results:
            result = envelope["record"]
            if (result.get("schema") != NODE_RESULT_SCHEMA
                    or envelope.get("record_sha256") != semantic_sha256(result)):
                raise ValueError("selection node numerical result digest changed")
            supplied = result["request"]
            digest = supplied["request_sha256"]
            if digest not in expected or canonical_json_bytes(supplied) != canonical_json_bytes(expected[digest]):
                raise ValueError("selection node result differs from its original ordered request")
            if digest in outcomes:
                raise ValueError("selection replay reduction repeats a requested node")
            request = expected[digest]["request"]
            if (result["prefix_record_ref"] != f"{request['package']}/prefix_inner_replay.json"
                    or result["prefix_record_sha256"] != request["prefix_receipt_sha256"]
                    or result["replay_device"] != "cpu" or result["native_threads"] != 1
                    or result["atol"] != 0.0 or result["rtol"] != 0.0):
                raise ValueError("selection node result changed its producer or exact replay programme")
            outcomes[digest] = result
        if set(outcomes) != set(expected):
            raise ValueError("selection replay reduction is missing original requested nodes")
        records, results = {}, {}
        for envelope in requests:
            request = envelope["request"]
            result = outcomes[envelope["request_sha256"]]
            key = request["fit_key"]
            record = self._serial_records.get(key)
            if record is None:
                record = _selection_node_manifest(_node_path(self._directory, request),
                    expected_record_sha256=request["prefix_receipt_sha256"])
            if result["captured_fit_device"] != record["device"]:
                raise ValueError("selection node result changed the captured fit device")
            _recursive_close(result["replay"], _saved_replay_result(record, request))
            records[key], results[key] = record, result["replay"]
        graph, selection = self._graph, self._graph["selection"]
        _audit_graph(graph, records)
        _audit_replayed_policy(graph, self._prefix, self._compiler, scope_owner=self._scope_owner)
        final_key = selection["final_refit_key"]
        final_programs = self._serial_final_programs
        if final_programs is None:
            node = next(node for node in graph["nodes"] if node["fit_key"] == final_key)
            final_package = _PrefixPackageAdmission(_node_path(self._directory, node),
                expected_record_sha256=node["prefix_receipt_sha256"],
                _metadata_admission=self._metadata_admission,
                _joint_source_preparation=self._joint_source_preparation,
                _source_seed_admission=self._source_seed_admission)
            final_programs = final_package.programs
        _loaded_final_selection(graph, records[final_key], final_programs)
        summary = {"status": "replayed", "actual_K": self._prefix.budget, "producer_count": len(graph["nodes"]),
            "exact_score_checks": sum(row["exact_score_checks"] for row in results.values()),
            "scores": results[final_key]["scores"], "zero_new_fits": True,
            "outer_requested": len(selection["outer_requested"]), "selected_candidate": selection["selected_candidate"]}
        return json.loads(canonical_json_bytes(summary))


def _audit_replayed_policy(graph, prefix, prefix_compiler, *, scope_owner=None):
    parent_groups = {group.view.record_id: group for group in prefix.complete_groups}
    selection = graph["selection"]
    # Recompute the actual policy on the same saved h probabilities/truths.
    # Truth is C-only and is never supplied to a scorer or numerical fitter.
    traces = [outer["selection"] for outer in selection["outer_rows"]] + [selection["final_selection"]]
    for trace in traces:
        for row in trace["structural_scope_rows"]:
            actual = prefix_compiler.before_group(prefix, parent_groups[row["held_record_id"]])
            identity = prefix_input_identity(actual) if scope_owner is None else scope_owner.scope(actual)["input"]
            if semantic_sha256(identity) != semantic_sha256(row["training_scope"]["input"]):
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


def _replay_admitted_selection(admission, *, prefix, prefix_compiler, future_views=(), source_checkpoints=None):
    plan = admission.prepare_replay(prefix=prefix, prefix_compiler=prefix_compiler,
        future_views=future_views, source_checkpoints=source_checkpoints)
    results = []
    with cpu_thread_budget(1):
        for envelope in plan.node_requests:
            result, package = _replay_selection_node(envelope, directory=plan._directory,
                parent_prefix=prefix, prefix_compiler=prefix_compiler, future_views=plan._future_views,
                source_checkpoints=plan._sources, source_sha256=plan._source_sha256,
                parent_sha256=envelope["request"]["parent_input_sha256"],
                _metadata_admission=admission._metadata_admission,
                _joint_source_preparation=admission._joint_source_preparation,
                _scope_owner=plan._scope_owner, _source_seed_admission=admission._source_seed_admission)
            results.append(result)
            plan._serial_records[envelope["request"]["fit_key"]] = package.record
            if envelope["request"]["final"]:
                plan._serial_final_programs = package.programs
        return plan.finish_replay(results)


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
