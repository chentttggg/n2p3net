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
import multiprocessing
import os
import sys
import time
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


def resolve_population_resources(resources):
    """Use the shared quota and memory owner without granting CUDA work."""
    from research.source_execution import resolve_source_resources
    from train.runtime import available_cpu_threads

    scope = resources.get("resource_scope")
    if scope not in {"local_development", "software_correctness"}:
        raise ValueError("the CPU-only source programme requires its explicit local/software scope")
    budget = resources["cpu_threads_total"]
    native = resources["native_threads_per_worker"]
    concurrency = max(resources["source_workers"], resources["fold_workers"])
    if budget > available_cpu_threads() or concurrency*native+1 > budget:
        raise ValueError("the effective CPU quota cannot meet the declared workers plus controller")
    if scope == "local_development" and budget >= .8*(os.cpu_count() or 1):
        raise ValueError("the local source programme must retain the below-80-percent whole-machine CPU preference")
    for name in ("source_worker_reservation_bytes", "fold_worker_reservation_bytes"):
        if type(resources[name]) is not int or resources[name] < 1:
            raise ValueError("source/fold memory claims require positive bytes")
    headroom = resources["memory_headroom_fraction"]
    if isinstance(headroom, bool) or not isinstance(headroom, (float, int)) or not .05 <= headroom < 1:
        raise ValueError("the source CPU owner requires at least five percent host headroom")
    maximum_memory = max(resources["source_worker_reservation_bytes"], resources["fold_worker_reservation_bytes"])
    resolved = resolve_source_resources({"resource_scope": scope, "cpu_budget": budget,
        "source_workers": concurrency, "target_workers": concurrency,
        "source_worker_memory_gib": maximum_memory/2**30,
        "target_worker_memory_gib": maximum_memory/2**30,
        "shared_staging_memory_gib": 0, "host_memory_fraction": 1-headroom}, 13, "cpu")
    if (resolved["cpu_budget"] != budget or resolved["source_workers"] != concurrency
            or resolved["target_workers"] != concurrency):
        raise ValueError("shared admission cannot meet the declared source/fold capacity")
    return resolved


@contextmanager
def phase_lease(admission, native_threads, memory_bytes, task_id):
    """Both scientific stages are source learning in the same owner process."""
    token = None
    while token is None:
        token = admission.reserve_phase("source", cpu_threads=native_threads,
            memory_bytes=memory_bytes, gpu_workers=0, task_id=task_id)
        if token is None:
            time.sleep(.05)
    try:
        yield
    finally:
        admission.release_phase(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--resources-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true", help="verify code/recipe without original EEG reads or fitting")
    args = parser.parse_args()
    recipe_ref = pinned(args.recipe, args.recipe_sha256)
    manifest_ref = pinned(args.source_manifest, args.source_manifest_sha256)
    recipe, manifest = read(recipe_ref), read(manifest_ref)
    if (recipe["schema"] != "full_original_population_source_recipe/1"
            or manifest["schema"] != "public_population_effective_source_manifest/2"
            or manifest["recipe"] != recipe_ref or manifest["launcher"] != artifact(__file__)):
        raise ValueError("the named source recipe/launcher manifest is required")
    for row in manifest["files"]:
        actual = artifact(row["path"])
        if (actual["bytes"], actual["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("an effective source file changed: "+row["module"])
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
        if resources["schema"] != "public_population_source_resources/2" or resources.get("device") != "cpu":
            raise ValueError("the explicit CPU source programme requires a post-audit /2 resource declaration")
        source_workers, fold_workers, native_threads, cpu_budget = (
            resources[name] for name in ("source_workers", "fold_workers", "native_threads_per_worker", "cpu_threads_total"))
        if (any(type(v) is not int or v < 1 for v in (source_workers, fold_workers, native_threads, cpu_budget))
                or max(source_workers, fold_workers) > 13
                or max(source_workers, fold_workers)*native_threads + 1 > cpu_budget):
            raise ValueError("source/fold native concurrency plus the controller exceeds the total CPU budget")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(native_threads)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(manifest["source_root"])/"src"))
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits

    from data.epochs import PreprocessingSpec
    from research import population_response_forecast as population
    from research import public_population_source as owner
    from research.execution import atomic_write_json
    from research.source_execution import HostMemoryAdmission
    from research.source_likelihood_export import SharedLikelihoodStore
    from transfer.native_response_dictionary import NativeResponseDictionary
    from transfer.response_dictionary import ResponseDictionary
    from transfer.spatial_programme import SpatialSignalProgram

    expected = {row["module"]: row for row in manifest["files"]}
    for name, module in tuple(sys.modules.items()):
        if name in expected and getattr(module, "__file__", None):
            if artifact(module.__file__)["sha256"] != expected[name]["sha256"]:
                raise ValueError("a runtime module differs from its effective source pin: "+name)
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
    owner_resources = resolve_population_resources(resources)
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
        "new_score_cells_pending": 4914, "resource_owner": "HostMemoryAdmission_same_process_CPU_phases",
        "requested_resource_plan": resources, "resolved_resources": owner_resources, "actual_GPU_workers": 0,
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
        with phase_lease(admission, native_threads, resources["source_worker_reservation_bytes"],
                "person:"+row["source_person_id"]):
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
                try:
                    complete.append(future.result())
                except Exception as error:
                    row = futures[future]
                    key = row.get("source_person_id", row.get("fold_id"))
                    stage = "persons" if "source_person_id" in row else "folds"
                    event(stage, key, "failed", exception_type=type(error).__name__, reason=str(error))
                    errors.append((key, type(error).__name__, str(error)))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise RuntimeError("incomplete source programme; actual completed artifacts preserved: "+repr(errors))
        return complete

    try:
        with multiprocessing.get_context("spawn").Manager() as manager, threadpool_limits(limits=native_threads):
            admission = HostMemoryAdmission(manager, owner_resources)
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
                    with phase_lease(admission, native_threads, resources["fold_worker_reservation_bytes"],
                            "fold:"+row["fold_id"]):
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
