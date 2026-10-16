"""Cloud adapters for original-compiler preparation and exact prefix admission.

The executor owns processes and temporary phase peaks. Its coordinator owns the
single input holder until every reader retires. Adapters transport only sealed
input descriptors, requests, task facts and artifact receipts, never models or
a private live graph plan.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

from research.contracts import canonical_json_bytes, semantic_sha256

SCHEMA = "n2p3_prefix_replay_execution/2"
PREPARATION_SCHEMA = "n2p3_prefix_replay_preparation/2"
RESULT_SCHEMA = "n2p3_prefix_replay_execution_result/2"
ADMISSION_SCHEMA = "n2p3_prefix_replay_execution_admission/2"
GRAPH_PHASE_MEMORY_BYTES = 8 * 2**30
_ROOT = Path(__file__).resolve().parents[2]
_EXECUTED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CONSUMER_PATHS = (
    "src/research/prefix_replay_execution.py", "src/research/prefix_selection_replay.py",
    "src/research/prefix_inner_replay.py", "src/research/prefix_replay_inputs.py",
    "src/transfer/ensemble_views.py", "src/research/selection_graph_store.py",
    "src/research/frozen_metadata.py", "src/research/execution.py",
    "src/research/calibration_selection.py", "src/research/contracts.py",
    "src/research/nested_selection_replay.py", "src/research/joint_flat_replay.py",
    "src/transfer/ensemble_fit.py", "src/transfer/joint_flat_model.py", "src/transfer/joint_flat_programme.py",
)
_SOURCE_RESPONSE_CONSUMER_PATHS = (
    "src/research/source_response_replay.py", "src/research/source_response_states.py",
    "src/transfer/source_response_programme.py", "src/research/finite_response_law.py",
    "src/transfer/native_response_dictionary.py", "src/research/channel_noise_mechanism.py",
    "src/research/response_history_identification.py", "src/research/history_authority.py",
    "src/research/latent_response_reference.py", "src/research/multichannel_c_latent.py",
    "src/research/joint_response_scale.py", "src/research/joint_source_catalog.py",
    "src/transfer/noise_model.py", "src/transfer/continuous_decision.py",
    "src/transfer/response_dictionary.py", "src/transfer/ensemble.py",
    "src/models/candidate_evidence.py", "src/research/candidate_metrics.py",
    "experiments/run_candidate_ensemble_study.py",
    "src/transfer/dictionary_artifact.py", "src/transfer/temporal_noise.py",
    "src/research/shared_stationary_factor.py", "src/research/stationary_noise_artifact.py",
    "src/research/joint_flat_authority.py",
)


def artifact_receipt(path):
    path = Path(path).resolve()
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
            total += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": total}


def _artifact_bytes(receipt):
    path = Path(receipt["path"])
    payload = path.read_bytes()
    if (path.is_symlink() or len(payload) != receipt["bytes"]
            or hashlib.sha256(payload).hexdigest() != receipt["sha256"]):
        raise ValueError("prefix replay artifact differs from its externally returned bytes")
    return payload


def _sealed(path, record):
    from research.execution import atomic_write_json

    atomic_write_json(path, {"record": record, "record_sha256": semantic_sha256(record)})
    return artifact_receipt(path)


def _read_sealed(receipt):
    payload = json.loads(_artifact_bytes(receipt))
    record = payload["record"]
    if payload["record_sha256"] != semantic_sha256(record):
        raise ValueError("prefix replay sealed artifact semantic digest changed")
    return record


def _consumer_paths(request):
    for row in (*request.get("ordinary_packages", ()), *request.get("selection_packages", ())):
        original = row["original_request"]
        groups = (candidate["expert_programs"] for candidate in original["candidates"]) if "candidates" in original else (
            original["expert_programs"],)
        if any(spec["family"] == "source_response" for group in groups for spec in group):
            return (*_CONSUMER_PATHS, *_SOURCE_RESPONSE_CONSUMER_PATHS)
    return _CONSUMER_PATHS


def _consumer_identity(request):
    actual = {name: artifact_receipt(_ROOT / name)["sha256"] for name in _consumer_paths(request)}
    if (any(request["consumer_source"].get(name) != digest for name, digest in actual.items())
            or actual["src/research/prefix_replay_execution.py"] != _EXECUTED_SHA256):
        raise ValueError("prefix replay adapter differs from the externally bound executed consumer")
    return actual


def _package(directory, name):
    path = Path(directory) / name
    if (not isinstance(name, str) or Path(name).name != name or name in {".", ".."}
            or path.is_symlink() or not path.resolve().is_relative_to(Path(directory).resolve())):
        raise ValueError("prefix replay package escapes its original participant")
    return path


def _roster(request, kind):
    rows = request[kind + "_packages"]
    names = [row["package"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("prefix replay repeats an original saved package")
    for name in names:
        _package(request["participant_directory"], name)
    return {row["package"]: row for row in rows}


def _saved_byte_contract(request, row, *, graph=False):
    saved = _read_sealed(row["saved_seal"])
    if (semantic_sha256(saved["request"]) != semantic_sha256(row["original_request"])
            or semantic_sha256(saved["files"]) != semantic_sha256(row["files"])
            or graph and saved["graph_sha256"] != row["graph_sha256"]):
        raise ValueError("prefix replay changed the original externally saved request/file seal")
    paths = [receipt["path"] for receipt in row["files"]]
    if len(set(paths)) != len(paths):
        raise ValueError("prefix replay original byte seal repeats a file")
    base = Path(request["participant_directory"]).resolve()
    for relative in paths:
        path = base / relative
        if (Path(relative).is_absolute() or Path(relative).parts[0] != row["package"]
                or not path.resolve().is_relative_to(base)):
            raise ValueError("prefix replay original byte seal escapes its actual package")
    return {receipt["path"]: receipt for receipt in row["files"]}


def _verify_package_files(request, row, *, already_read=()):
    base = Path(request["participant_directory"]).resolve()
    for receipt in row["files"]:
        if receipt["path"] in already_read:
            continue
        actual = artifact_receipt(base / receipt["path"])
        if actual["sha256"] != receipt["sha256"] or actual["bytes"] != receipt["bytes"]:
            raise ValueError("prefix replay original saved package byte seal changed")


def _validate_request(request):
    if request["schema"] != SCHEMA:
        raise ValueError("unsupported prefix replay execution request")
    identity = _consumer_identity(request)
    ledger = _read_sealed(request["producer_ledger"])
    if (Path(request["producer_ledger"]["path"]).resolve()
            != Path(request["participant_directory"]).resolve() / "request_ledger.json"
            or ledger["execution_source"] != request["producer_source"]
            or ledger["record"]["sha256"] != request["mat"]["sha256"]
            or ledger["scopes_sha256"] != request["scopes"]["sha256"]
            or ledger["participant_key"] != request["input_request"]["participant_key"]
            or ledger["native_threads"] != replay_native_threads(request, ordinary=True)):
        raise ValueError("prefix replay native programme differs from its original source-bound ledger")
    old = request["producer_module"]
    inputs = request["input_request"]
    if (request["producer_source"].get("src/transfer/ensemble_views.py") != old["sha256"]
            or inputs["producer_source_sha256"] != old["sha256"]
            or inputs.get("dataset_source") is None and inputs["original_source_sha256"] != request["mat"]["sha256"]
            or inputs["preparation_task_id"] != request["preparation_task_id"]):
        raise ValueError("prefix replay original module/MAT/preparation authority changed")
    for name, artifact in request.get("producer_input_modules", {}).items():
        if artifact["sha256"] != request["producer_source"].get("src/"+name.replace(".", "/")+".py"):
            raise ValueError("prefix replay original dataset decoder differs from the actual producer source")
    _roster(request, "selection")
    _roster(request, "ordinary")
    return identity


def replay_native_threads(request, *, ordinary):
    """Ordinary scores retain the producer context; selection owns native1."""
    native = request["ordinary_native_threads"]
    if type(native) is not int or native < 1:
        raise ValueError("ordinary prefix replay requires the original positive native thread count")
    return native if ordinary else 1


def _require_native_resources(task, request, *, ordinary):
    native = replay_native_threads(request, ordinary=ordinary)
    if (task.resources.phase != "evaluation" or task.resources.device != "cpu"
            or task.resources.native_threads != native
            or task.resources.interop_threads != 1 or task.resources.cpu_threads != native + 1):
        raise ValueError("prefix replay task resources differ from its original native programme")


def validate_replay_reader_resources(task):
    """Scientific reader roles own numeric policy; the input pool owns leases."""
    adapters = {__name__ + ":" + name: ordinary for name, ordinary in (
        ("ordinary_prefix_replay_adapter", True), ("ordinary_reducer_adapter", True),
        ("selection_chunk_adapter", False), ("selection_reducer_adapter", False))}
    if task.adapter in adapters:
        _require_native_resources(task, task.arguments["request"], ordinary=adapters[task.adapter])
    elif task.adapter == "experiments.run_candidate_ensemble_study:candidate_participant_admission_adapter":
        request = task.arguments["replay_request"]
        original = task.arguments["request"]["producer_task"]["resources"]
        if original["native_threads"] != replay_native_threads(request, ordinary=True):
            raise ValueError("prefix replay parent differs from its original producer native programme")
        _require_native_resources(task, request, ordinary=True)
    else:
        raise ValueError("shared prefix replay reader has no declared scientific adapter role")


def _worker_native_threads(request, *, ordinary):
    import torch

    native = replay_native_threads(request, ordinary=ordinary)
    if any(os.environ.get(name) != str(native)
           for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("prefix replay worker environment differs from its original native programme")
    if torch.get_num_threads() != native or torch.get_num_interop_threads() != 1:
        raise ValueError("prefix replay worker numeric pools differ from its declared native/inter-op programme")
    return native


def _request_source_programs(row):
    """Resolve each source role from its original declared expert programme."""
    from transfer.ensemble_fit import CalibrationExpertProgram

    request = row["original_request"]
    groups = (candidate["expert_programs"] for candidate in request["candidates"]) if "candidates" in request else (
        request["expert_programs"],)
    programs = {}
    for group in groups:
        for record in group:
            spec = CalibrationExpertProgram.from_record(record)
            previous = programs.get(spec.expert_id)
            if previous is not None and (previous.family, previous.source_checkpoint_role) != (
                    spec.family, spec.source_checkpoint_role):
                raise ValueError("prefix replay source has conflicting original expert roles")
            programs[spec.expert_id] = spec
    if not set(row["source_checkpoints"]) <= set(programs):
        raise ValueError("prefix replay source lacks its original declared expert role")
    return programs


def _joint_sources_owner(row):
    if not row["source_checkpoints"]:
        return None
    programs = _request_source_programs(row)
    if any(programs[name].family in {"joint_flat", 'source_response'} for name in row["source_checkpoints"]):
        from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

        return FrozenJointFlatSourcePreparation()
    return None


def _sources(row, *, joint_source_preparation=None):
    from transfer.ensemble_fit import FrozenSourceCheckpoint

    if not row["source_checkpoints"]:
        return {}
    programs = _request_source_programs(row)
    sources = {}
    for name, receipt in row["source_checkpoints"].items():
        content = _artifact_bytes(receipt)
        if programs[name].family in {"joint_flat", 'source_response'}:
            from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

            if joint_source_preparation is None:
                joint_source_preparation = FrozenJointFlatSourcePreparation()
            if not isinstance(joint_source_preparation, FrozenJointFlatSourcePreparation):
                raise TypeError("joint replay source requires its actual private preparation owner")
            sources[name] = joint_source_preparation.prepare(content, expected_sha256=receipt["sha256"])
        else:
            sources[name] = FrozenSourceCheckpoint(content)
    return sources


def _writer(output, estimate):
    from research.source_execution import current_input_admission

    admission = current_input_admission()
    return admission.artifact_directory(output, estimate) if admission is not None else nullcontext()


def _require_retired(task, outcome, *, adapter):
    from research.cloud_execution import CloudResult, CloudTask

    if not isinstance(task, CloudTask) or not isinstance(outcome, CloudResult):
        raise TypeError("prefix replay admission requires actual CloudExecutor task and result objects")
    worker = outcome.worker or {}
    remaining = worker.get("process_tree_retirement", {}).get("active_owned_processes_after_exit")
    if (task.adapter != __name__ + ":" + adapter or outcome.task_id != task.task_id
            or outcome.status != "completed" or outcome.failure is not None
            or worker.get("alive") is not False or type(worker.get("exit_code")) is not int
            or worker["exit_code"] != 0 or type(remaining) is not int or remaining != 0):
        raise ValueError("prefix replay requires successful actual exit and complete owned-tree retirement")


def _graph_budgets(graph):
    selection = graph["selection"]
    budgets = {selection["parent_scope"]["actual_K"], selection["final_refit_scope"]["actual_K"]}
    budgets.update(node["scope"]["actual_K"] for node in graph["nodes"])
    for trace in [outer["selection"] for outer in selection["outer_rows"]] + [selection["final_selection"]]:
        budgets.add(trace["training_scope"]["actual_K"])
        budgets.update(row["training_scope"]["actual_K"] for row in trace["structural_scope_rows"])
    return budgets


def _prepare_history_capabilities(views, budgets, records):
    """Reissue original requests from actual source operands before export."""
    from research.prefix_selection_replay import _selection_node_manifest
    from transfer.ensemble_views import calibration_prefix_identity

    joint_records = []
    for directory, expected in records:
        record = _selection_node_manifest(directory, expected_record_sha256=expected)
        if record["schema"] in {"n2p3_joint_flat_prefix_inner_replay/1", 'n2p3_source_response_prefix_replay/1'}:
            joint_records.append((directory, record))
    if not joint_records:
        return (), ()
    prefixes, original_views = {}, {}
    compiler = views.prefix_compiler
    for budget in budgets:
        prefix = compiler.compile(budget)
        prefixes[semantic_sha256(calibration_prefix_identity(prefix))] = prefix
        for group in prefix.complete_groups:
            original_views[semantic_sha256(group.view.record())] = group.view
    original_views.update((semantic_sha256(view.record()), view) for view in views.decisions)
    capabilities, refusals = {}, {}
    for directory, record in joint_records:
        for row in (*record.get("OOF", ()), *record["future"]):
            for permission in row["issued_history_permissions"].values():
                digest, binding = permission["sha256"], permission["binding"]
                if semantic_sha256(binding) != digest:
                    raise ValueError("original history permission differs from its captured complete binding")
                if digest in capabilities:
                    continue
                prefix = prefixes.get(semantic_sha256(binding["prefix_identity"]))
                view = original_views.get(semantic_sha256(binding["current_view"]))
                if prefix is None or view is None:
                    raise ValueError("original history request lacks its actual prefix or physical view")
                cap = compiler.record_read_scope(view, prefix=prefix,
                    conditioning_stop=binding["conditioning_sample_support"][1],
                    available_stop=binding["score_sample_support"][1], program=binding["program"])
                if cap.digest != digest or semantic_sha256(cap.binding) != semantic_sha256(binding):
                    raise ValueError("original compiler reissued a different C/P/T history permission")
                capabilities[digest] = cap
            for entry in row.get('captures', {}).values():
                if 'readset_transition' in entry:
                    from research.joint_flat_replay import _load_capture
                    from research.joint_readset_transition import reissue_joint_readset_record

                    transition = entry['readset_transition']
                    cap = capabilities[transition['current_capability_sha256']]
                    reissue_joint_readset_record(compiler, cap,
                        _load_capture(Path(directory)/entry['capture']['path']), transition, directory=directory)
            for permission in row.get("issued_history_refusals", {}).values():
                from research.history_authority import HistoryUnavailableError
                from transfer.ensemble_views import _record_history_refusal_key

                digest, binding = permission["sha256"], permission["binding"]
                if semantic_sha256(binding) != digest:
                    raise ValueError("original history refusal differs from its captured complete binding")
                if digest in refusals:
                    continue
                request = binding["request"]
                prefix = prefixes.get(semantic_sha256(request["prefix_identity"]))
                view = original_views.get(semantic_sha256(request["current_view"]))
                if prefix is None or view is None:
                    raise ValueError("original history refusal lacks its actual prefix or physical view")
                try:
                    compiler.record_read_scope(view, prefix=prefix, conditioning_stop=request["conditioning_stop"],
                        available_stop=request["available_stop"], program=request["program"])
                except HistoryUnavailableError as error:
                    if semantic_sha256(error.record()) != semantic_sha256(binding["error"]):
                        raise ValueError("original compiler returned a different history refusal") from error
                else:
                    raise ValueError("original compiler now permits a previously refused history request")
                refusal = compiler._record_history_refusals.get(_record_history_refusal_key(binding))
                if refusal is None or refusal.digest != digest or semantic_sha256(refusal.binding) != semantic_sha256(binding):
                    raise ValueError("original compiler reissued a different exact-request refusal witness")
                refusals[digest] = refusal
    return tuple(capabilities.values()), tuple(refusals.values())


def _prepared_graph_owner(job, original, participant_directory):
    """Reopen one sealed compact store, independently of original graph bytes."""
    from research.selection_graph_store import SelectionGraphStore

    if job["logical_graph_sha256"] != original["graph_sha256"]:
        raise ValueError("prepared selection store changed the original logical graph seal")
    imported = json.loads(_artifact_bytes(job["graph_import"]))
    graph_name = original["package"] + "/prefix_selection.json"
    source = next(row for row in original["files"] if row["path"] == graph_name)
    if (imported["schema"] != "n2p3_legacy_selection_graph_import/1"
            or Path(imported["original"]["path"]).resolve() != Path(participant_directory).resolve() / graph_name
            or imported["original"]["sha256"] != source["sha256"]
            or imported["original"]["bytes"] != source["bytes"]
            or imported["consumer_store"] != job["graph"]
            or imported["logical_graph_sha256"] != original["graph_sha256"]
            or imported["selection_sha256"] != job["selection_sha256"]):
        raise ValueError("prepared selection graph lost its original byte/logical binding")
    graph = SelectionGraphStore.from_bytes(_artifact_bytes(job["graph"]),
        directory=Path(job["graph"]["path"]).resolve().parent,
        expected_graph_sha256=original["graph_sha256"], expected_selection_sha256=job["selection_sha256"])
    if semantic_sha256(graph.scope_receipts) != semantic_sha256(job["graph_scope_files"]):
        raise ValueError("prepared selection graph changed its complete scope dependency roster")
    return graph


def _selection_metadata(selection):
    return {key: selection[key] for key in ("candidates", "baseline", "minimum_selection_rows",
        "source_checkpoint_sha256", "selected_candidate", "final_refit_key") } | {
        "parent_scope": {"actual_K": selection["parent_scope"]["actual_K"],
            "input": {"prefix_record_sha256": selection["parent_scope"]["input"]["prefix_record_sha256"]}}}


def prefix_replay_prepare_adapter(arguments):
    """Run the actual frozen compiler in a fresh owned child, once per MAT."""
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    _validate_request(request)
    _worker_native_threads(request, ordinary=False)
    _artifact_bytes(request["producer_module"])
    if artifact_receipt(request["mat"]["path"]) != request["mat"]:
        raise ValueError("prefix replay original MAT bytes changed")
    _artifact_bytes(request["scopes"])
    with _writer(output, arguments["output_estimate_bytes"]):
        output.mkdir(parents=True, exist_ok=False)
        control = output / "arguments.json"
        control.write_bytes(canonical_json_bytes(arguments))
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_ROOT / "src"), str(_ROOT)]),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
            "--frozen-preparation", str(control)], env=environment, check=True)
        manifest = _read_sealed(artifact_receipt(output / "preparation.json"))
        return {"capsule": manifest["capsule"], "preparation": artifact_receipt(output / "preparation.json")}


def _frozen_preparation(arguments):
    """No transfer class may be imported before installing the real old module."""
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    identity = _validate_request(request)
    old = request["producer_module"]
    verified_code = _artifact_bytes(old)
    if "transfer.ensemble_views" in sys.modules:
        raise RuntimeError("frozen preparation must start before consumer transfer classes are imported")
    modules = request.get("producer_input_modules", {})
    ordered = sorted(modules, key=lambda name: (0 if name == "data.candidate_task" else
                                              1 if name == "data.candidate_input" else 2, name))
    for name in ordered:
        artifact = modules[name]
        content = _artifact_bytes(artifact)
        if name in sys.modules or not name.startswith("data."):
            raise ValueError("original candidate decoder must be installed before consumer data classes")
        import_spec = importlib.util.spec_from_file_location(name, artifact["path"])
        original = importlib.util.module_from_spec(import_spec)
        sys.modules[name] = original
        exec(compile(content, artifact["path"], "exec"), original.__dict__)
    spec = importlib.util.spec_from_file_location("transfer.ensemble_views", old["path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(verified_code, old["path"], "exec"), module.__dict__)
    from data.epochs import parse_preprocessing_spec
    from research.prefix_replay_inputs import export_prepared_views
    from research.prefix_selection_replay import _node_path, _SelectionReplayAdmission
    from research.selection_graph_store import prepare_selection_graph_store
    from train.runtime import configure_spawned_worker_threads, cpu_thread_budget

    inputs = request["input_request"]
    profiles = tuple(module.SelectionEpochProfile.from_record(row) for row in inputs["profiles"])
    configure_spawned_worker_threads(1, interop_threads=1)
    with cpu_thread_budget(1):
        declaration = inputs.get("dataset_source", {"path": request["mat"]["path"], "sha256": inputs["original_source_sha256"]})
        views = module.prepare_dataset_independent_prefixes(declaration,
            budgets=inputs["requested_Ks"], profiles=profiles,
            authorization_preprocessing=parse_preprocessing_spec(inputs["authorization_preprocessing"]),
            response_seconds=inputs["observation_response_seconds"], pre_roll_seconds=inputs["pre_roll_seconds"])
        scopes = json.loads(_artifact_bytes(request["scopes"]))
        if semantic_sha256(views.ledger) != semantic_sha256(scopes):
            raise ValueError("frozen original MAT compilation differs from the sealed actor scopes")
        jobs, budgets, history_records = [], set(inputs["actual_Ks"]), []
        for row in request["selection_packages"]:
            directory = _package(request["participant_directory"], row["package"])
            files = _saved_byte_contract(request, row, graph=True)
            graph_name = row["package"] + "/prefix_selection.json"
            original = files[graph_name]
            path = output / ("graph_" + row["graph_sha256"] + ".json")
            store, imported = prepare_selection_graph_store({**original,
                "path": str(Path(request["participant_directory"]).resolve() / graph_name)}, path,
                expected_graph_sha256=row["graph_sha256"])
            try:
                admitted = _SelectionReplayAdmission.from_prepared_graph(directory, store,
                    expected_graph_sha256=row["graph_sha256"])
                _verify_package_files(request, row, already_read=(graph_name,))
                graph = admitted.graph
                budgets.update(_graph_budgets(graph))
                history_records.extend((_node_path(directory, node), node["prefix_receipt_sha256"])
                    for node in graph["nodes"])
                plan = admitted.prepare_replay(prefix=views.prefix_compiler.compile(row["original_request"]["K"]),
                    prefix_compiler=views.prefix_compiler, future_views=views.decisions,
                    source_checkpoints=_sources(row, joint_source_preparation=admitted._joint_source_preparation))
                jobs.append({"package": row["package"], "graph": imported["consumer_store"],
                    "logical_graph_sha256": store.graph_sha256, "graph_scope_files": list(store.scope_receipts),
                    "graph_import": artifact_receipt(path.with_suffix(".import.json")),
                    "selection_sha256": store.selection_sha256, "node_requests": plan.node_requests,
                    "original_request": row["original_request"]})
                del plan, admitted, graph
            finally:
                store.close()
        pending, complete = list(budgets), set()
        while pending:
            budget = pending.pop()
            if budget in complete:
                continue
            prefix = views.prefix_compiler.compile(budget)
            complete.add(budget)
            for group in prefix.complete_groups:
                earlier = views.prefix_compiler.before_group(prefix, group)
                if earlier.budget not in complete:
                    pending.append(earlier.budget)
        prepared_inputs = {**inputs, "actual_Ks": sorted(complete)}
        history_records.extend((_package(request["participant_directory"], row["package"]), row["prefix_record_sha256"])
            for row in request["ordinary_packages"])
        histories, refusals = _prepare_history_capabilities(views, complete, history_records)
        capsule = export_prepared_views(views, profiles, output / "inputs", prepared_inputs,
            record_history_scopes=histories, record_history_refusals=refusals)
        manifest = {"schema": PREPARATION_SCHEMA, "request": request,
            "request_sha256": semantic_sha256(request), "input_request": prepared_inputs,
            "input_request_sha256": semantic_sha256(prepared_inputs), "capsule": capsule,
            "consumer_identity": identity, "compiled_input_source_sha256": old["sha256"],
            "selection_jobs": jobs, "ordinary_jobs": request["ordinary_packages"],
            "scoring_truth": artifact_receipt(output / "inputs" / "scoring_truth.json")}
        _sealed(output / "preparation.json", manifest)


def read_prefix_preparation(task, outcome, expected_request):
    """Read small actual returned facts; root alone adopts and owns the holder."""
    _require_retired(task, outcome, adapter="prefix_replay_prepare_adapter")
    _require_native_resources(task, expected_request, ordinary=False)
    _validate_request(expected_request)
    if (semantic_sha256(task.arguments["request"]) != semantic_sha256(expected_request)
            or set(outcome.value) != {"capsule", "preparation"}
            or Path(outcome.value["preparation"]["path"]).resolve() != Path(task.output).resolve() / "preparation.json"):
        raise ValueError("prefix preparation return differs from its actual original request/output")
    manifest = _read_sealed(outcome.value["preparation"])
    if (manifest["schema"] != PREPARATION_SCHEMA or manifest["request_sha256"] != semantic_sha256(expected_request)
            or semantic_sha256(manifest["request"]) != semantic_sha256(expected_request)
            or manifest["capsule"] != outcome.value["capsule"]
            or manifest["consumer_identity"] != _consumer_identity(expected_request)):
        raise ValueError("prefix preparation manifest changed its external source or request")
    derived = manifest["input_request"]
    original = expected_request["input_request"]
    if (semantic_sha256({key: value for key, value in derived.items() if key != "actual_Ks"})
            != semantic_sha256({key: value for key, value in original.items() if key != "actual_Ks"})
            or not set(original["actual_Ks"]).issubset(derived["actual_Ks"])
            or manifest["input_request_sha256"] != semantic_sha256(derived)):
        raise ValueError("prefix preparation changed its original input programme")
    jobs = {row["package"]: row for row in manifest["selection_jobs"]}
    expected = _roster(expected_request, "selection")
    if len(jobs) != len(manifest["selection_jobs"]) or set(jobs) != set(expected):
        raise ValueError("prefix preparation lost the complete original selection package roster")
    for name, row in jobs.items():
        if (row["logical_graph_sha256"] != expected[name]["graph_sha256"]
                or Path(row["graph"]["path"]).resolve() != Path(task.output).resolve() / ("graph_" + row["logical_graph_sha256"] + ".json")
                or semantic_sha256(row["original_request"]) != semantic_sha256(expected[name]["original_request"])
                or not row["node_requests"]
                or any(node["request"]["graph_sha256"] != expected[name]["graph_sha256"]
                    or node["request"]["actual_K"] not in derived["actual_Ks"] for node in row["node_requests"])):
            raise ValueError("prefix preparation changed original graph/node/actual-K authority")
    if semantic_sha256(manifest["ordinary_jobs"]) != semantic_sha256(expected_request["ordinary_packages"]):
        raise ValueError("prefix preparation lost original ordinary packages")
    return manifest


def _borrow(arguments):
    from research.prefix_replay_inputs import open_prepared_inputs

    return open_prepared_inputs(arguments["shared_inputs"],
        expected_request_sha=arguments["input_request_sha256"], expected_capsule_sha=arguments["capsule_sha256"])


def _require_input_binding(arguments, receipt):
    if (receipt["request_sha256"] != arguments["input_request_sha256"]
            or receipt["capsule_sha256"] != arguments["capsule_sha256"]
            or receipt["producer_source_sha256"] != arguments["request"]["producer_module"]["sha256"]):
        raise ValueError("prefix replay input provenance differs from its unique original preparation")


def _result(arguments, record):
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    with _writer(output, arguments["output_estimate_bytes"]):
        output.mkdir(parents=True, exist_ok=False)
        return _sealed(output / "replay.json", {"schema": RESULT_SCHEMA,
            "request_sha256": semantic_sha256(request), "consumer_identity": _consumer_identity(request),
            "interop_threads": 1,
            **record})


def selection_chunk_adapter(arguments):
    """One native1 CPU chunk consumes borrowed original inputs and small requests."""
    from research.prefix_selection_replay import replay_selection_chunk

    request = arguments["request"]
    _validate_request(request)
    _worker_native_threads(request, ordinary=False)
    row = _roster(request, "selection")[arguments["package"]]
    with _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        source_preparation = _joint_sources_owner(row)
        results = replay_selection_chunk(arguments["node_requests"],
            directory=_package(request["participant_directory"], row["package"]),
            parent_prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
            source_checkpoints=_sources(row, joint_source_preparation=source_preparation),
            _joint_source_preparation=source_preparation)
        return _result(arguments, {"kind": "selection_chunk", "package": row["package"],
            "native_threads": 1,
            "node_requests_sha256": semantic_sha256(arguments["node_requests"]),
            "input_consumer": borrowed.receipt, "node_results": results})


def ordinary_prefix_replay_adapter(arguments):
    """The original ordinary package gate, with real prefix/OOF/future replay."""
    from research.prefix_inner_replay import _PrefixPackageAdmission
    from train.runtime import cpu_thread_budget

    request = arguments["request"]
    _validate_request(request)
    native = _worker_native_threads(request, ordinary=True)
    row = _roster(request, "ordinary")[arguments["package"]]
    with cpu_thread_budget(native), _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        _saved_byte_contract(request, row)
        _verify_package_files(request, row)
        path = _package(request["participant_directory"], row["package"])
        source_preparation = _joint_sources_owner(row)
        package = _PrefixPackageAdmission(path, expected_record_sha256=row["prefix_record_sha256"],
            _joint_source_preparation=source_preparation)
        replay = package.replay(prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            prefix_compiler=borrowed.views.prefix_compiler,
            future_views=borrowed.views.decisions,
            source_checkpoints=_sources(row, joint_source_preparation=source_preparation),
            context=row["context"], atol=0.0, rtol=0.0)
        return _result(arguments, {"kind": "ordinary", "package": row["package"],
            "prefix_record_sha256": row["prefix_record_sha256"], "replay": replay,
            "captured_fit_device": package.record["device"], "replay_device": "cpu",
            "native_threads": native, "atol": 0.0, "rtol": 0.0, "input_consumer": borrowed.receipt})


def _admit_worker(task, outcome, request, *, adapter, kind, package):
    _require_retired(task, outcome, adapter=adapter)
    _require_native_resources(task, request, ordinary=kind == "ordinary")
    if (semantic_sha256(task.arguments["request"]) != semantic_sha256(request)
            or task.arguments["package"] != package
            or Path(outcome.value["path"]).resolve() != Path(task.output).resolve() / "replay.json"):
        raise ValueError("prefix replay worker return rewrites its task or request")
    record = _read_sealed(outcome.value)
    if (record["schema"] != RESULT_SCHEMA or record["kind"] != kind or record["package"] != package
            or record["request_sha256"] != semantic_sha256(request)
            or record["consumer_identity"] != _consumer_identity(request)
            or record["interop_threads"] != 1
            or record["native_threads"] != replay_native_threads(request, ordinary=kind == "ordinary")):
        raise ValueError("prefix replay worker artifact differs from its actual bound consumer/request")
    _require_input_binding(task.arguments, record["input_consumer"])
    return record


def _admission(arguments, record):
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    with _writer(output, arguments["output_estimate_bytes"]):
        output.mkdir(parents=True, exist_ok=False)
        return _sealed(output / "admission.json", {"schema": ADMISSION_SCHEMA,
            "request_sha256": semantic_sha256(request), "consumer_identity": _consumer_identity(request),
            "interop_threads": 1,
            "producer_source_digest": semantic_sha256(request["producer_source"]),
            "consumer_source_digest": semantic_sha256(request["consumer_source"]),
            "scopes": request["scopes"], **record})


def selection_reducer_adapter(arguments):
    """One temporary graph decode, original ordered policy and fresh final state."""
    from research.prefix_selection_replay import _SelectionReplayAdmission
    from train.runtime import cpu_thread_budget

    request, job = arguments["request"], arguments["job"]
    _validate_request(request)
    _worker_native_threads(request, ordinary=False)
    row = _roster(request, "selection")[job["package"]]
    if job["logical_graph_sha256"] != row["graph_sha256"]:
        raise ValueError("prefix selection reducer lost its original external graph seal")
    results, seen = [], set()
    for task, outcome in arguments["node_outcomes"]:
        if task.task_id in seen:
            raise ValueError("selection reducer repeats an actual worker task")
        seen.add(task.task_id)
        record = _admit_worker(task, outcome, request, adapter="selection_chunk_adapter",
            kind="selection_chunk", package=row["package"])
        _require_input_binding(arguments, record["input_consumer"])
        if record["node_requests_sha256"] != semantic_sha256(task.arguments["node_requests"]):
            raise ValueError("selection reducer worker omitted its original node request chunk")
        results.extend(record["node_results"])
    with cpu_thread_budget(1), _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        store = _prepared_graph_owner(job, row, request["participant_directory"])
        admitted = _SelectionReplayAdmission.from_prepared_graph(
            _package(request["participant_directory"], row["package"]), store,
            expected_graph_sha256=row["graph_sha256"], expected_selection_sha256=job["selection_sha256"])
        plan = admitted.prepare_replay(prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
            source_checkpoints=_sources(row, joint_source_preparation=admitted._joint_source_preparation))
        if semantic_sha256(plan.node_requests) != semantic_sha256(job["node_requests"]):
            raise ValueError("selection reducer prepared nodes differ from original graph/input authority")
        summary = plan.finish_replay(results)
        selected = admitted.graph["selection"]
        original_summary = None if row["replayed_summary"] is None else _read_sealed(row["replayed_summary"])
        if original_summary is not None and (original_summary["graph_sha256"] != row["graph_sha256"]
                or original_summary["selection_sha256"] != job["selection_sha256"]
                or original_summary["producer_count"] != summary["producer_count"]
                or original_summary["exact_score_checks"] != summary["exact_score_checks"]
                or semantic_sha256(original_summary["selected_candidate"]) != semantic_sha256(selected["selected_candidate"])
                or semantic_sha256(original_summary["final_refit_scope"]) != semantic_sha256(selected["final_refit_scope"])):
            raise ValueError("prefix selection reducer differs from the original sealed replay summary")
        return _admission(arguments, {"kind": "selection", "package": row["package"],
            "native_threads": 1,
            "graph_sha256": row["graph_sha256"], "selection_sha256": job["selection_sha256"],
            "selection": _selection_metadata(selected), "summary": summary,
            "byte_seal_files": row["files"], "original_summary": original_summary,
            "future": {result["decision_id"]: result["strategies"][selected["selected_candidate"]["strategy"]]
                for result in summary["scores"]}, "input_consumer": borrowed.receipt})


def ordinary_reducer_adapter(arguments):
    """Read the original manifest once and bind exact actual worker output."""
    from research.prefix_inner_replay import _recursive_close, future_input_identity
    from research.prefix_selection_replay import _saved_replay_result, _selection_node_manifest
    from train.runtime import cpu_thread_budget

    request = arguments["request"]
    _validate_request(request)
    native = _worker_native_threads(request, ordinary=True)
    row = _roster(request, "ordinary")[arguments["package"]]
    task, outcome = arguments["ordinary_outcome"]
    record = _admit_worker(task, outcome, request, adapter="ordinary_prefix_replay_adapter",
        kind="ordinary", package=row["package"])
    _require_input_binding(arguments, record["input_consumer"])
    if (record["prefix_record_sha256"] != row["prefix_record_sha256"]
            or record["replay_device"] != "cpu" or record["native_threads"] != native
            or record["atol"] != 0.0 or record["rtol"] != 0.0):
        raise ValueError("ordinary prefix reducer changed its original exact CPU replay programme")
    with cpu_thread_budget(native), _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        original = _selection_node_manifest(_package(request["participant_directory"], row["package"]),
            expected_record_sha256=row["prefix_record_sha256"])
        if (record["captured_fit_device"] != original["device"]
                or semantic_sha256(original["fit_context"]) != semantic_sha256(row["context"])):
            raise ValueError("ordinary prefix replay rewrites captured fit device or scope")
        views = [{"record_id": view.record_id, "input_sha256": semantic_sha256(future_input_identity(view))}
            for view in borrowed.views.decisions]
        _recursive_close(record["replay"], _saved_replay_result(original, {"views": views}))
        return _admission(arguments, {"kind": "ordinary", "package": row["package"],
            "native_threads": native,
            "record_sha256": row["prefix_record_sha256"], "future_rows": original["future_score_rows"],
            "byte_seal_files": row["files"], "summary": record["replay"], "input_consumer": borrowed.receipt})


class CompletedParticipantReplay:
    """Private results of complete actual reducer admissions, never caller flags."""

    def __init__(self):
        raise TypeError("completed replay is constructed only from actual reducer outcomes")

    @classmethod
    def from_reduced(cls, request, reducer_outcomes):
        _validate_request(request)
        expected = {(kind, name) for kind in ("selection", "ordinary") for name in _roster(request, kind)}
        results, seen = {}, set()
        for task, outcome in reducer_outcomes:
            kind = "selection" if task.adapter.endswith(":selection_reducer_adapter") else "ordinary"
            _require_retired(task, outcome, adapter=kind + "_reducer_adapter")
            _require_native_resources(task, request, ordinary=kind == "ordinary")
            package = task.arguments["job"]["package"] if kind == "selection" else task.arguments["package"]
            key = kind, package
            if key not in expected or key in results or task.task_id in seen:
                raise ValueError("completed prefix replay lost its unique complete saved-package roster")
            if (semantic_sha256(task.arguments["request"]) != semantic_sha256(request)
                    or Path(outcome.value["path"]).resolve() != Path(task.output).resolve() / "admission.json"):
                raise ValueError("completed prefix replay changes its actual reducer request/output")
            record = _read_sealed(outcome.value)
            if (record["schema"] != ADMISSION_SCHEMA or record["kind"] != kind or record["package"] != package
                    or record["request_sha256"] != semantic_sha256(request)
                    or record["consumer_identity"] != _consumer_identity(request)
                    or record["interop_threads"] != 1
                    or record["native_threads"] != replay_native_threads(request, ordinary=kind == "ordinary")
                    or record["producer_source_digest"] != semantic_sha256(request["producer_source"])
                    or record["consumer_source_digest"] != semantic_sha256(request["consumer_source"])
                    or record["scopes"] != request["scopes"]
                    or semantic_sha256(record["byte_seal_files"]) != semantic_sha256(_roster(request, kind)[package]["files"])):
                raise ValueError("completed replay artifact changed original scopes/source/task authority")
            _require_input_binding(task.arguments, record["input_consumer"])
            results[key] = canonical_json_bytes(record)
            seen.add(task.task_id)
        if set(results) != expected:
            raise ValueError("completed prefix replay is missing original saved-package admissions")
        instance = cls.__new__(cls)
        instance._request_sha256, instance._results = semantic_sha256(request), results
        return instance

    def assert_request(self, request):
        if semantic_sha256(request) != self._request_sha256:
            raise ValueError("completed replay belongs to another original participant request")

    def selection(self, package):
        return json.loads(self._results["selection", package])

    def ordinary(self, package):
        return json.loads(self._results["ordinary", package])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--frozen-preparation":
        raise SystemExit("expected --frozen-preparation CONTROL.json")
    _frozen_preparation(json.loads(Path(sys.argv[2]).read_bytes()))
