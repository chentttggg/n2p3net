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

SCHEMA = "n2p3_prefix_replay_execution/1"
PREPARATION_SCHEMA = "n2p3_prefix_replay_preparation/1"
RESULT_SCHEMA = "n2p3_prefix_replay_execution_result/1"
ADMISSION_SCHEMA = "n2p3_prefix_replay_execution_admission/1"
GRAPH_PHASE_MEMORY_BYTES = 8 * 2**30
_ROOT = Path(__file__).resolve().parents[2]
_EXECUTED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CONSUMER_PATHS = (
    "src/research/prefix_replay_execution.py", "src/research/prefix_selection_replay.py",
    "src/research/prefix_inner_replay.py", "src/research/prefix_replay_inputs.py",
    "src/transfer/ensemble_views.py",
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


def _consumer_identity(request):
    actual = {name: artifact_receipt(_ROOT / name)["sha256"] for name in _CONSUMER_PATHS}
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
    old = request["producer_module"]
    inputs = request["input_request"]
    if (request["producer_source"].get("src/transfer/ensemble_views.py") != old["sha256"]
            or inputs["producer_source_sha256"] != old["sha256"]
            or inputs["original_source_sha256"] != request["mat"]["sha256"]
            or inputs["preparation_task_id"] != request["preparation_task_id"]):
        raise ValueError("prefix replay original module/MAT/preparation authority changed")
    _roster(request, "selection")
    _roster(request, "ordinary")
    return identity


def _sources(row):
    from transfer.ensemble_fit import FrozenSourceCheckpoint

    return {name: FrozenSourceCheckpoint(_artifact_bytes(receipt))
        for name, receipt in row["source_checkpoints"].items()}


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


def _selection_metadata(selection):
    return {key: selection[key] for key in ("candidates", "baseline", "minimum_selection_rows",
        "source_checkpoint_sha256", "selected_candidate", "final_refit_key") } | {
        "parent_scope": {"actual_K": selection["parent_scope"]["actual_K"],
            "input": {"prefix_record_sha256": selection["parent_scope"]["input"]["prefix_record_sha256"]}}}


def prefix_replay_prepare_adapter(arguments):
    """Run the actual frozen compiler in a fresh owned child, once per MAT."""
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    _validate_request(request)
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
    spec = importlib.util.spec_from_file_location("transfer.ensemble_views", old["path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(verified_code, old["path"], "exec"), module.__dict__)
    from data.epochs import parse_preprocessing_spec
    from research.prefix_replay_inputs import export_prepared_views
    from research.prefix_selection_replay import _SelectionReplayAdmission
    from train.runtime import cpu_thread_budget

    inputs = request["input_request"]
    profiles = tuple(module.SelectionEpochProfile.from_record(row) for row in inputs["profiles"])
    with cpu_thread_budget(1):
        views = module.prepare_bnci_independent_prefixes(request["mat"]["path"],
            source_sha256=inputs["original_source_sha256"], budgets=inputs["requested_Ks"], profiles=profiles,
            authorization_preprocessing=parse_preprocessing_spec(inputs["authorization_preprocessing"]),
            response_seconds=inputs["observation_response_seconds"], pre_roll_seconds=inputs["pre_roll_seconds"])
        scopes = json.loads(_artifact_bytes(request["scopes"]))
        if semantic_sha256(views.ledger) != semantic_sha256(scopes):
            raise ValueError("frozen original MAT compilation differs from the sealed actor scopes")
        jobs, budgets = [], set(inputs["actual_Ks"])
        for row in request["selection_packages"]:
            directory = _package(request["participant_directory"], row["package"])
            files = _saved_byte_contract(request, row, graph=True)
            graph_name = row["package"] + "/prefix_selection.json"
            original = files[graph_name]
            payload = json.loads(_artifact_bytes({**original,
                "path": str(Path(request["participant_directory"]).resolve() / graph_name)}))
            if payload["record_sha256"] != row["graph_sha256"]:
                raise ValueError("prefix replay original graph semantic seal changed")
            canonical = canonical_json_bytes(payload["record"])
            del payload
            admitted = _SelectionReplayAdmission.from_prepared_graph(directory, canonical,
                expected_graph_sha256=row["graph_sha256"])
            _verify_package_files(request, row, already_read=(graph_name,))
            graph = admitted.graph
            budgets.update(_graph_budgets(graph))
            plan = admitted.prepare_replay(prefix=views.prefix_compiler.compile(row["original_request"]["K"]),
                prefix_compiler=views.prefix_compiler, future_views=views.decisions, source_checkpoints=_sources(row))
            if hashlib.sha256(canonical).hexdigest() != row["graph_sha256"]:
                raise ValueError("prepared canonical graph differs from the original external graph seal")
            path = output / ("canonical_" + row["graph_sha256"] + ".json")
            with path.open("xb") as stream:
                stream.write(canonical)
            jobs.append({"package": row["package"], "graph": artifact_receipt(path),
                "selection_sha256": semantic_sha256(graph["selection"]), "node_requests": plan.node_requests,
                "original_request": row["original_request"]})
            del plan, admitted, graph, canonical
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
        capsule = export_prepared_views(views, profiles, output / "inputs", prepared_inputs)
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
        if (row["graph"]["sha256"] != expected[name]["graph_sha256"]
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
            **record})


def selection_chunk_adapter(arguments):
    """One native1 CPU chunk consumes borrowed original inputs and small requests."""
    from research.prefix_selection_replay import replay_selection_chunk

    request = arguments["request"]
    _validate_request(request)
    row = _roster(request, "selection")[arguments["package"]]
    with _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        results = replay_selection_chunk(arguments["node_requests"],
            directory=_package(request["participant_directory"], row["package"]),
            parent_prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
            source_checkpoints=_sources(row))
        return _result(arguments, {"kind": "selection_chunk", "package": row["package"],
            "node_requests_sha256": semantic_sha256(arguments["node_requests"]),
            "input_consumer": borrowed.receipt, "node_results": results})


def ordinary_prefix_replay_adapter(arguments):
    """The original ordinary package gate, with real prefix/OOF/future replay."""
    from research.prefix_inner_replay import _PrefixPackageAdmission
    from train.runtime import cpu_thread_budget

    request = arguments["request"]
    _validate_request(request)
    row = _roster(request, "ordinary")[arguments["package"]]
    with cpu_thread_budget(1), _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        _saved_byte_contract(request, row)
        _verify_package_files(request, row)
        path = _package(request["participant_directory"], row["package"])
        package = _PrefixPackageAdmission(path, expected_record_sha256=row["prefix_record_sha256"])
        replay = package.replay(prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            future_views=borrowed.views.decisions, source_checkpoints=_sources(row),
            context=row["context"], atol=0.0, rtol=0.0)
        return _result(arguments, {"kind": "ordinary", "package": row["package"],
            "prefix_record_sha256": row["prefix_record_sha256"], "replay": replay,
            "captured_fit_device": package.record["device"], "replay_device": "cpu",
            "native_threads": 1, "atol": 0.0, "rtol": 0.0, "input_consumer": borrowed.receipt})


def _admit_worker(task, outcome, request, *, adapter, kind, package):
    _require_retired(task, outcome, adapter=adapter)
    if (semantic_sha256(task.arguments["request"]) != semantic_sha256(request)
            or task.arguments["package"] != package
            or Path(outcome.value["path"]).resolve() != Path(task.output).resolve() / "replay.json"):
        raise ValueError("prefix replay worker return rewrites its task or request")
    record = _read_sealed(outcome.value)
    if (record["schema"] != RESULT_SCHEMA or record["kind"] != kind or record["package"] != package
            or record["request_sha256"] != semantic_sha256(request)
            or record["consumer_identity"] != _consumer_identity(request)):
        raise ValueError("prefix replay worker artifact differs from its actual bound consumer/request")
    _require_input_binding(task.arguments, record["input_consumer"])
    return record


def _admission(arguments, record):
    request, output = arguments["request"], Path(arguments["output"]).resolve()
    with _writer(output, arguments["output_estimate_bytes"]):
        output.mkdir(parents=True, exist_ok=False)
        return _sealed(output / "admission.json", {"schema": ADMISSION_SCHEMA,
            "request_sha256": semantic_sha256(request), "consumer_identity": _consumer_identity(request),
            "producer_source_digest": semantic_sha256(request["producer_source"]),
            "consumer_source_digest": semantic_sha256(request["consumer_source"]),
            "scopes": request["scopes"], **record})


def selection_reducer_adapter(arguments):
    """One temporary graph decode, original ordered policy and fresh final state."""
    from research.prefix_selection_replay import _SelectionReplayAdmission
    from train.runtime import cpu_thread_budget

    request, job = arguments["request"], arguments["job"]
    _validate_request(request)
    row = _roster(request, "selection")[job["package"]]
    if job["graph"]["sha256"] != row["graph_sha256"]:
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
        admitted = _SelectionReplayAdmission.from_prepared_graph(
            _package(request["participant_directory"], row["package"]), _artifact_bytes(job["graph"]),
            expected_graph_sha256=row["graph_sha256"], expected_selection_sha256=job["selection_sha256"])
        plan = admitted.prepare_replay(prefix=borrowed.views.prefix_compiler.compile(row["original_request"]["K"]),
            prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
            source_checkpoints=_sources(row))
        if semantic_sha256(plan.node_requests) != semantic_sha256(job["node_requests"]):
            raise ValueError("selection reducer prepared nodes differ from original graph/input authority")
        summary = plan.finish_replay(results)
        selected = admitted.graph["selection"]
        original_summary = _read_sealed(row["replayed_summary"])
        if (original_summary["graph_sha256"] != row["graph_sha256"]
                or original_summary["selection_sha256"] != job["selection_sha256"]
                or original_summary["producer_count"] != summary["producer_count"]
                or original_summary["exact_score_checks"] != summary["exact_score_checks"]
                or semantic_sha256(original_summary["selected_candidate"]) != semantic_sha256(selected["selected_candidate"])
                or semantic_sha256(original_summary["final_refit_scope"]) != semantic_sha256(selected["final_refit_scope"])):
            raise ValueError("prefix selection reducer differs from the original sealed replay summary")
        return _admission(arguments, {"kind": "selection", "package": row["package"],
            "graph_sha256": row["graph_sha256"], "selection_sha256": job["selection_sha256"],
            "selection": _selection_metadata(selected), "summary": summary,
            "byte_seal_files": row["files"], "original_summary": original_summary,
            "future": {result["decision_id"]: result["strategies"][selected["selected_candidate"]["strategy"]]
                for result in summary["scores"]}, "input_consumer": borrowed.receipt})


def ordinary_reducer_adapter(arguments):
    """Read the original manifest once and bind exact actual worker output."""
    from research.prefix_inner_replay import _manifest, _recursive_close, future_input_identity
    from research.prefix_selection_replay import _saved_replay_result
    from train.runtime import cpu_thread_budget

    request = arguments["request"]
    _validate_request(request)
    row = _roster(request, "ordinary")[arguments["package"]]
    task, outcome = arguments["ordinary_outcome"]
    record = _admit_worker(task, outcome, request, adapter="ordinary_prefix_replay_adapter",
        kind="ordinary", package=row["package"])
    _require_input_binding(arguments, record["input_consumer"])
    if (record["prefix_record_sha256"] != row["prefix_record_sha256"]
            or record["replay_device"] != "cpu" or record["native_threads"] != 1
            or record["atol"] != 0.0 or record["rtol"] != 0.0):
        raise ValueError("ordinary prefix reducer changed its original exact CPU replay programme")
    with cpu_thread_budget(1), _borrow(arguments) as borrowed:
        _require_input_binding(arguments, borrowed.receipt)
        original = _manifest(_package(request["participant_directory"], row["package"]),
            expected_record_sha256=row["prefix_record_sha256"])
        if (record["captured_fit_device"] != original["device"]
                or semantic_sha256(original["fit_context"]) != semantic_sha256(row["context"])):
            raise ValueError("ordinary prefix replay rewrites captured fit device or scope")
        views = [{"record_id": view.record_id, "input_sha256": semantic_sha256(future_input_identity(view))}
            for view in borrowed.views.decisions]
        _recursive_close(record["replay"], _saved_replay_result(original, {"views": views}))
        return _admission(arguments, {"kind": "ordinary", "package": row["package"],
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
