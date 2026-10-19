"""Run the declared thirteen original people and thirteen source-only folds.

The caller pins the recipe and effective source manifest. Resource allocation is
explicit and separate. This runner publishes actual owner outputs for external
review; it does not manufacture the independent receipt needed for live use.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock


def artifact(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def read(reference):
    content = Path(reference["path"]).read_bytes()
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError("the caller-pinned source dependency changed: "+reference["path"])
    return json.loads(content)


def pinned(path, expected):
    reference = artifact(path)
    if reference["sha256"] != expected:
        raise ValueError("the caller's external artifact pin differs: "+str(path))
    return reference


def verify_effective_inputs(manifest):
    """Verify actual frozen-root bytes before any cloud owner is imported."""
    source_root = Path(manifest["source_root"]).resolve()
    for row in manifest["files"]:
        actual = artifact(source_root / row["relative"])
        if actual != {name: row[name] for name in ("path", "bytes", "sha256")}:
            raise ValueError("an effective source file changed: " + row["module"])
    inputs = {row["relative"]: row for row in manifest["execution_inputs"]}
    required = {"configs/cloud_training_resources.json", "pyproject.toml", "uv.lock"}
    if set(inputs) != required or len(inputs) != len(manifest["execution_inputs"]):
        raise ValueError("the complete runtime policy and environment pins are required")
    for relative, row in inputs.items():
        if artifact(source_root / relative) != {name: row[name] for name in ("path", "bytes", "sha256")}:
            raise ValueError("a frozen execution input changed: " + relative)
    return source_root, inputs["configs/cloud_training_resources.json"]


def verify_cloud_policy_owner(config_path, source_root, expected):
    """The shared owner must consume the policy file just verified here."""
    if (Path(config_path).resolve() != source_root / "configs/cloud_training_resources.json"
            or artifact(config_path) != {name: expected[name] for name in ("path", "bytes", "sha256")}):
        raise ValueError("the actual cloud owner policy is outside the pinned frozen source root")


def verify_loaded_sources(manifest):
    expected = {row["module"]: row for row in manifest["files"]}
    for name, module in tuple(sys.modules.items()):
        if name in expected and getattr(module, "__file__", None):
            if artifact(module.__file__)["sha256"] != expected[name]["sha256"]:
                raise ValueError("a runtime module differs from its effective source pin: " + name)


def resolve_population_resources(resources, *, cloud_owner_resources):
    """Consume the existing cloud programme lease; never create a second ledger."""
    from research.population_source_cloud import require_programme_phase
    from research.resource_scope import require_managed_phase

    managed = require_managed_phase(resources, phase="source", device="cpu")
    require_programme_phase(resources, cloud_owner_resources, managed)
    return cloud_owner_resources


@contextmanager
def programme_member_scope(admission, native_threads, memory_bytes, task_id, *, resources):
    """Members consume the held aggregate lease until the owner retires its tree."""
    from research.resource_scope import require_managed_phase

    managed = require_managed_phase(resources, phase="source", device="cpu")
    phase = managed.task.resources
    if (admission is not managed.admission or native_threads > phase.native_threads
            or memory_bytes > phase.memory_bytes or not task_id):
        raise ValueError("a source/fold member exceeds its actual aggregate cloud phase")
    yield


def main(argv=None, *, cloud_owner_resources=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--resources-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true", help="verify code/recipe without original EEG reads or fitting")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective_argv)
    recipe_ref = pinned(args.recipe, args.recipe_sha256)
    manifest_ref = pinned(args.source_manifest, args.source_manifest_sha256)
    recipe, manifest = read(recipe_ref), read(manifest_ref)
    if (recipe["schema"] != "full_original_population_source_recipe/1"
            or manifest["schema"] != "public_population_effective_source_manifest/3"
            or manifest["recipe"] != recipe_ref or manifest["launcher"] != artifact(__file__)):
        raise ValueError("the named source recipe/launcher manifest is required")
    source_root, cloud_policy_pin = verify_effective_inputs(manifest)
    matrix = read(recipe["source_matrix"])
    read(matrix["source_freeze"])
    if (len(matrix["person_programmes"]) != 13 or len(matrix["fold_programmes"]) != 13
            or matrix["pending_new_gaussian_score_cells"] != 4914
            or recipe["programme_seed"] != matrix["programme_seed"]):
        raise ValueError("the complete thirteen-person source programme differs")
    if args.inspect:
        native_threads = 1
    else:
        if any(v is None for v in (args.resources, args.resources_sha256, args.output)):
            parser.error("execution requires --resources, --resources-sha256 and a new --output")
        resources_ref = pinned(args.resources, args.resources_sha256)
        resources = read(resources_ref)
        if (resources.get("schema"), resources.get("resource_scope"), resources.get("device")) not in {
                ("public_population_source_resources/3", "cloud", "cpu"),
                ("public_population_source_local_resources/1", "local_development", "cpu")}:
            raise ValueError("source computation requires its explicit owned local or cloud CPU declaration")
        source_workers, fold_workers, native_threads, cpu_budget = (
            resources[name] for name in ("source_workers", "fold_workers", "native_threads_per_worker", "cpu_threads_total"))
        if (any(type(v) is not int or v < 1 for v in (source_workers, fold_workers, native_threads, cpu_budget))
                or max(source_workers, fold_workers) > 13
                or max(source_workers, fold_workers)*native_threads + 1 > cpu_budget):
            raise ValueError("source/fold native concurrency plus the controller exceeds the total CPU budget")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(native_threads)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source_root / "src"), str(source_root)]
    if not args.inspect:
        from research.source_execution import CLOUD_RESOURCE_CONFIG

        verify_cloud_policy_owner(CLOUD_RESOURCE_CONFIG, source_root, cloud_policy_pin)
    if not args.inspect and cloud_owner_resources is None:
        from research.population_source_cloud import run_population_source_cloud

        verify_loaded_sources(manifest)
        return run_population_source_cloud(argv=effective_argv, resources=resources, output=args.output,
            identity={"recipe": recipe_ref, "source_manifest": manifest_ref, "resource_plan": resources_ref})
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits

    from data.epochs import PreprocessingSpec
    from research import population_response_forecast as population
    from research import public_population_source as owner
    from research.execution import atomic_write_json
    from research.resource_scope import require_managed_phase
    from research.source_likelihood_export import SharedLikelihoodStore
    from transfer.native_response_dictionary import NativeResponseDictionary
    from transfer.response_dictionary import ResponseDictionary
    from transfer.spatial_programme import SpatialSignalProgram

    verify_loaded_sources(manifest)
    dictionary = ResponseDictionary.from_record(read(recipe["dictionary"]))
    native_dictionary = NativeResponseDictionary.from_record(read(recipe["native_dictionary"]))
    signal = SpatialSignalProgram.from_record(recipe["signal_programme"])
    scale_record = read(recipe["scale_policy"])
    scale = population.LogScaleProgram(tuple(scale_record["prior_mean"]), tuple(scale_record["prior_sd"]),
        tuple(scale_record["initial"]), scale_record["maximum_iterations"], scale_record["gradient_tolerance"],
        recipe["scale_policy"]["sha256"])
    if not np.array_equal(dictionary.matrix, native_dictionary.matrix):
        raise ValueError("source fitting and native continuation do not use the same frozen matrix")
    if args.inspect:
        print(json.dumps({"status": "source_recipe_runtime_and_code_pins_verified", "recipe": recipe_ref,
            "manifest": manifest_ref, "people": 13, "folds": 13, "new_score_cells": 4914,
            "source_owner": artifact(owner.__file__), "population_core": artifact(population.__file__),
            "dictionary_digest": dictionary.digest(), "native_dictionary_digest": native_dictionary.digest(),
            "signal_programme": signal.record(), "scale_programme": scale.record(),
            "actual_EEG_reads": 0, "actual_source_fits": 0, "actual_population_hyperfits": 0}, indent=2))
        return
    owner_resources = resolve_population_resources(resources, cloud_owner_resources=cloud_owner_resources)
    managed = require_managed_phase(resources, phase="source", device="cpu")
    admission = managed.admission
    available = owner_resources["available_host_memory_bytes"]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    store = SharedLikelihoodStore(output/"people")
    lock = Lock()
    checkpoint = {"schema": "complete_population_source_progress/1", "status": "running", "pid": os.getpid(),
        "process_create_time": psutil.Process().create_time(), "recipe": recipe_ref, "source_manifest": manifest_ref,
        "resource_plan": resources_ref, "source_requested": 13, "fold_requested": 13,
        "requested_records": 234, "common_records": 213, "common_labels": 101388,
        "persons": {row["source_person_id"]: {"status": "pending"} for row in matrix["person_programmes"]},
        "folds": {row["fold_id"]: {"status": "pending"} for row in matrix["fold_programmes"]},
        "new_score_cells_pending": 4914, "resource_owner": "source_family_CloudExecutor_owned_aggregate_CPU_programme",
        "resource_scope": resources["resource_scope"], "requested_resource_plan": resources, "resolved_resources": owner_resources, "actual_GPU_workers": 0, "ready_GPU_tasks": 0,
        "independent_acceptance_issued_here": False, "future_records_attached": False,
        "memory_available_at_launch": available}

    def event(stage, key, status, **fields):
        row = {"utc": datetime.now(UTC).isoformat(), "stage": stage, "key": key, "status": status, **fields}
        with lock:
            if stage in ("persons", "folds"):
                checkpoint[stage][key] = row
            atomic_write_json(output/"checkpoint.json", checkpoint)
            with (output/"events.jsonl").open("a", encoding="utf8", newline="\n") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    def fit_person(row):
        with programme_member_scope(admission, native_threads, resources["source_worker_reservation_bytes"],
                "person:"+row["source_person_id"], resources=resources):
            key = row["source_person_id"]
            started = time.perf_counter()
            event("persons", key, "running")
            person = owner.open_frozen_source_person(matrix["source_freeze"], key, profiles=(),
                authorization_preprocessing=PreprocessingSpec(**recipe["authorization_preprocessing"]),
                response_seconds=recipe["response_seconds"], pre_roll_seconds=recipe["pre_roll_seconds"])
            compact, publication = owner.fit_and_export_source_person(person, signal_programme=signal,
                dictionary=dictionary, noise_options=recipe["noise_options"],
                output_directory=output/("source-"+key.rsplit("/", 1)[-1]), shared_store=store)
            event("persons", key, "complete", elapsed_seconds=time.perf_counter()-started,
                publication=publication, shared_likelihood=compact.record())
            return compact, publication

    def run_stage(rows, workers, operation):
        complete, errors = [], []
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(operation, row): row for row in rows}
            for future in as_completed(futures):
                row = futures.pop(future)
                try:
                    complete.append(future.result())
                except Exception as error:
                    key = row.get("source_person_id", row.get("fold_id"))
                    stage = "persons" if "source_person_id" in row else "folds"
                    failure_path = output / "failures" / (stage + "-" + hashlib.sha256(key.encode()).hexdigest() + ".traceback.txt")
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_path.open("x", encoding="utf8", newline="\n") as stream:
                        stream.write("".join(traceback.format_exception(error)))
                    event(stage, key, "failed", exception_type=type(error).__name__, reason=str(error),
                        original_traceback=artifact(failure_path))
                    errors.append((key, type(error).__name__, str(error)))
                    traceback.clear_frames(error.__traceback__)
                finally:
                    del future
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise RuntimeError("incomplete source programme; actual completed artifacts preserved: "+repr(errors))
        return complete

    try:
        with threadpool_limits(limits=native_threads):
            try:
                pools = threadpool_info()
                if any(pool["num_threads"] > native_threads for pool in pools):
                    raise ValueError("effective BLAS/OpenMP concurrency exceeds the declared native worker budget")
                event("programme", "source", "launched", effective_pools=pools,
                    source_workers=source_workers, fold_workers=fold_workers, native_threads_per_worker=native_threads)
                person_results = run_stage(matrix["person_programmes"], source_workers, fit_person)
                references = tuple(row[0] for row in person_results)
                publications = {row[0].participant_key: row[1] for row in person_results}
                gc.collect()

                def fit_fold(row):
                    with programme_member_scope(admission, native_threads, resources["fold_worker_reservation_bytes"],
                            "fold:"+row["fold_id"], resources=resources):
                        started = time.perf_counter()
                        event("folds", row["fold_id"], "running")
                        fold = population.fold_from_freeze_candidate(matrix["source_freeze"]["path"],
                            expected_sha256=matrix["source_freeze"]["sha256"],
                            target_participant_key=row["target_participant_key"])
                        directory = output/row["fold_id"]
                        reference, templates, policy, noise, within = owner.prepare_fold_response_reference(
                            store, fold, references, person_publications=[publications[k] for k in fold.source_participant_keys],
                            dictionary=native_dictionary, conditioning_samples=recipe["noise_options"]["conditioning_samples"],
                            scale_program=scale, output_directory=directory/"reference")
                        forecast, fold_ref = owner.fit_shared_population_fold(store, fold, references, reference=reference,
                            templates=templates, scale_program=scale, frozen_policy_reference=policy)
                        saved = owner.save_shared_forecast(forecast, directory/"forecast", store=store, fold_reference=fold_ref)
                        result = {"forecast": saved, "within_reference": within, "noise_reference": noise,
                            "reference_policy": policy}
                        event("folds", row["fold_id"], "complete", elapsed_seconds=time.perf_counter()-started, **result)
                        return result

                folds = run_stage(matrix["fold_programmes"], fold_workers, fit_fold)
                execution = owner.publish_source_population_execution(recipe["source_matrix"],
                    person_publications=[publications[row["source_person_id"]] for row in matrix["person_programmes"]],
                    fold_publications=folds, recipe_references=(recipe_ref, manifest_ref, resources_ref),
                    output_path=output/"SOURCE_EXECUTION.json")
            finally:
                checkpoint["resource_observation"] = admission.record()
        checkpoint["status"] = "complete_pending_independent_acceptance"
        checkpoint["execution"] = execution
        event("programme", "source", checkpoint["status"], execution=execution,
            actual_person_programmes=len(publications), actual_fold_hyperfits=len(folds))
        return execution
    except BaseException as error:
        checkpoint["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        for stage in ("persons", "folds"):
            for row in checkpoint[stage].values():
                if row["status"] in {"pending", "running"}:
                    row.update(status="failed", exception_type=type(error).__name__, reason=str(error))
        event("programme", "source", checkpoint["status"], exception_type=type(error).__name__, reason=str(error))
        raise


if __name__ == "__main__":
    main()
