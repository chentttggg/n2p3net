"""Run explicit public-source pools through the shared optimized trajectory interface."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from data.domain import (  # noqa: E402
    SOURCE_DOMAIN_AXIS_SCHEMA,
    SOURCE_DOMAIN_COLUMN,
    ensure_epoch_dataset_namespace,
)
from data.epochs import (  # noqa: E402
    concatenate_epoch_datasets,
    load_epoch_dataset,
    parse_preprocessing_spec,
    select_epoch_channels,
    select_scheduled_events,
)
from data.source_admission import PreparedSourceInventory, bind_source_inputs  # noqa: E402
from data.source_records import SOURCE_EOG_PREPROCESSING, file_sha256  # noqa: E402
from experiments.run_source_training_chain import fit_or_reuse_source_trajectory  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_execution import SharedTrajectoryRegistry  # noqa: E402
from research.source_matrix import (  # noqa: E402
    freeze_matched_dose,
    matched_dose_rows,
    matched_dose_tasks,
    select_source_matrix_tasks,
    source_matrix_tasks,
)
from research.source_selection import source_identity_axes  # noqa: E402
from train.runtime import available_cpu_threads, cpu_thread_budget  # noqa: E402


@lru_cache(maxsize=1)
def source_cache(path, digest, preprocessing_json=None):
    if file_sha256(path) != digest:
        raise ValueError("prepared source cache digest changed")
    data = load_epoch_dataset(path, validation="attested")
    spec = (SOURCE_EOG_PREPROCESSING if preprocessing_json is None else
            parse_preprocessing_spec(json.loads(preprocessing_json)))
    if (data.provenance.get("source_reference") != "measured_channel:CZ" or data.preprocessing != spec):
        raise ValueError("source matrix requires its declared measured-Cz preprocessing")
    return ensure_epoch_dataset_namespace(data, data.provenance["dataset_class"])


def source_provenance(task):
    return {"source_reference": "measured_channel:CZ", "source_pool_studies": task["studies"],
            "source_domain_axis": {"schema": SOURCE_DOMAIN_AXIS_SCHEMA,
                                   "column": SOURCE_DOMAIN_COLUMN, "domains": task["studies"]}}


def prepare_dose_views(tasks, prepared, dose, *, preprocessing_json=None):
    """Decode one study once, retain only each seed's exact authorized dose rows."""
    views = {task["id"]: [] for task in tasks}
    for study in tasks[0]["studies"]:
        data = source_cache(prepared[study]["cache"], prepared[study]["cache_sha256"], preprocessing_json)
        domains, people, _ = source_identity_axes(data)
        for task in tasks:
            selected = [p for p in dose["selections"][task["id"]] if p[0] == study]
            scope = {**dose, "selections": {task["id"]: selected}}
            rows = matched_dose_rows(data, task, scope, people, domains)
            keep_rows = np.zeros(data.n_epochs, bool)
            keep_rows[rows] = True
            evidence = data.event_timeline.evidence_indices
            keep_events = np.zeros(len(evidence), bool)
            available = evidence >= 0
            keep_events[available] = keep_rows[evidence[available]]
            views[task["id"]].append(select_scheduled_events(data, keep_events,
                reason="source-only matched participant/class/trial dose; exact seeded rows"))
        del data, domains, people
        source_cache.cache_clear()
        gc.collect()
    return {task["id"]: concatenate_epoch_datasets(views[task["id"]], name="explicit_source_pool",
             provenance={**source_provenance(task), "source_dose": dose["folds"][task["target"]]}) for task in tasks}


def execute_group(arguments):
    tasks, prepared, output, snapshot, config, dose, threads, preprocessing_json, device = arguments
    with cpu_thread_budget(threads):
        if tasks[0]["dose"] == "matched":
            dose_views = prepare_dose_views(tasks, prepared, dose, preprocessing_json=preprocessing_json)
        else:
            datasets = [source_cache(prepared[s]["cache"], prepared[s]["cache_sha256"], preprocessing_json)
                        for s in tasks[0]["studies"]]
            data = concatenate_epoch_datasets(datasets, name="explicit_source_pool",
                                             provenance=source_provenance(tasks[0]))
            del datasets
            source_cache.cache_clear()
            gc.collect()
            if tasks[0]["channels"] is not None:
                data = select_epoch_channels(data, tasks[0]["channels"])
        cache_digest = semantic_sha256({s: prepared[s]["cache_sha256"] for s in tasks[0]["studies"]})
        result = []
        registry = SharedTrajectoryRegistry(Path(output) / "trajectory_registry")
        for task in tasks:
            destination = Path(output) / task["id"]
            if task["dose"] == "matched":
                data = dose_views[task["id"]]
            rows = np.arange(data.n_epochs)
            started = time.perf_counter()
            fitted = fit_or_reuse_source_trajectory(registry=registry, dataset=data, rows=rows,
                config=DeepConfig(**config, seed=task["seed"]), destination=destination,
                task_id=task["id"], source_cache_sha256=cache_digest,
                source_snapshot_sha256=snapshot["archive_sha256"],
                excluded_source_datasets=task["excluded"], device=device, evidence_scope="development")
            record = {"task": task, "status": "completed", "seconds": time.perf_counter() - started,
                      "trajectory": fitted, "source_snapshot": snapshot,
                      "input_caches": {s: prepared[s]["cache_sha256"] for s in task["studies"]}}
            atomic_write_json(destination / "result.json", record)
            print(json.dumps({"source_task": task["id"], "status": "completed", "seconds": record["seconds"]}), flush=True)
            result.append(record)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--compile-mode", choices=("none", "reduce-overhead"), default="reduce-overhead")
    parser.add_argument("--task", action="append", help="Explicit probe selection; omission executes the full matrix")
    parser.add_argument("--matrix-plan", type=Path, help="Explicit source pool plan; omission preserves all 153 historical tasks")
    parser.add_argument("--preprocessing-config", type=Path, help="Complete preprocessing profile shared by all source caches")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and publish the plan without fitting")
    parser.add_argument("--wait-for-preparation-seconds", type=float, default=0,
                        help="Bounded input wait; positive values allow preparation and training to overlap")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads_per_worker, args.updates) < 1:
        parser.error("workers, threads and updates must be positive")
    if not np.isfinite(args.wait_for_preparation_seconds) or args.wait_for_preparation_seconds < 0:
        parser.error("preparation wait must be finite and nonnegative")
    snapshot = verified_executed_snapshot(args.snapshot, root=ROOT)
    config = dict(fixed_updates=args.updates, batch_size=512, lr=.001, weight_decay=.0001,
        sampling="uniform_rows", input_statistics="observed_rows", val_group_frac=None,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
        preload_memory_fraction=.10, batch_memory_fraction=.25)
    matrix_plan = None if args.matrix_plan is None else json.loads(args.matrix_plan.read_text(encoding="utf-8-sig"))
    tasks = select_source_matrix_tasks(selected_ids=args.task, plan=matrix_plan)
    dose_tasks = matched_dose_tasks(tasks, source_matrix_tasks(matrix_plan))
    spec = (SOURCE_EOG_PREPROCESSING if args.preprocessing_config is None else
            parse_preprocessing_spec(json.loads(args.preprocessing_config.read_text(encoding="utf-8-sig"))))
    preprocessing_json = json.dumps(spec.record(), sort_keys=True)
    required_studies = {s for task in (*tasks, *dose_tasks) for s in task["studies"]}
    inventory = PreparedSourceInventory(args.prepared, preprocessing=spec)
    if args.updates != 4000 and not args.task:
        raise ValueError("nonstandard update budget is permitted only for explicitly selected probes")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(schema="n2p3_explicit_source_matrix/2", tasks=list(tasks), config=config,
                source_snapshot=snapshot, workers=max(1, min(args.workers, available_cpu_threads() // args.threads_per_worker)),
                threads_per_worker=args.threads_per_worker, device=args.device,
                preprocessing=spec.record(), dose_comparison_tasks=list(dose_tasks))
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("source matrix resume requires the identical execution plan")
    atomic_write_json(plan_path, plan)
    initial = inventory.inspect(required_studies)
    atomic_write_json(args.output / "input_status.json", initial)
    if initial["failed"] or (initial["pending"] and (args.dry_run or not args.wait_for_preparation_seconds)):
        raise ValueError(f"source inputs are not ready: failed={initial['failed']}, pending={initial['pending']}")
    binding_path = args.output / "input_bindings.json"
    bindings = json.loads(binding_path.read_text(encoding="utf-8")) if binding_path.exists() else {}
    bindings = bind_source_inputs(bindings, initial["ready"])
    atomic_write_json(binding_path, bindings)
    if args.dry_run:
        print(json.dumps({"status": "planned", "tasks": len(tasks), "fitting_performed": False}), flush=True)
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads_per_worker)
    groups = {}
    for task in tasks:
        # Exact prior results are reused through the trajectory registry; never overwritten under a new signature.
        key = semantic_sha256({k: task[k] for k in ("studies", "channels", "excluded", "dose")})
        groups.setdefault(key, []).append(task)
    pending, active, results = list(groups.values()), {}, []
    dose = None
    input_deadline = time.monotonic() + args.wait_for_preparation_seconds
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        while pending or active:
            state = inventory.inspect(required_studies)
            prepared = state["ready"]
            atomic_write_json(args.output / "input_status.json", state)
            bindings = bind_source_inputs(bindings, prepared)
            atomic_write_json(binding_path, bindings)
            input_errors = dict(state["failed"])
            if time.monotonic() >= input_deadline:
                input_errors.update({s: f"preparation wait expired: {reason}" for s, reason in state["pending"].items()})
            dose_studies = {s for task in dose_tasks for s in task["studies"]}
            for group in list(pending):
                dependencies = set(group[0]["studies"]) | (dose_studies if group[0]["dose"] == "matched" else set())
                if dependencies & set(input_errors):
                    reason = {s: input_errors[s] for s in sorted(dependencies & set(input_errors))}
                    results.extend({"id": t["id"], "status": "failed", "input_errors": reason} for t in group)
                    pending.remove(group)
            if dose is None and dose_tasks and dose_studies <= set(prepared):
                counts = {}
                for study in sorted(dose_studies):
                    data = source_cache(prepared[study]["cache"], prepared[study]["cache_sha256"], preprocessing_json)
                    _, people, _ = source_identity_axes(data)
                    counts[study] = {p: np.bincount(data.y[people == p], minlength=2).tolist() for p in sorted(set(people))}
                dose = freeze_matched_dose(counts, tasks=dose_tasks)
                atomic_write_json(args.output / "matched_dose.json", dose)
                source_cache.cache_clear()
            for group in list(pending):
                if len(active) >= plan["workers"]:
                    break
                if not set(group[0]["studies"]) <= set(prepared):
                    continue
                # Run full-pool trajectories as soon as their inputs arrive; dose jobs wait for common P/D.
                ready = [t for t in group if t["dose"] == "full" or dose is not None]
                if not ready:
                    continue
                remaining = [t for t in group if t not in ready]
                pending.remove(group)
                if remaining:
                    pending.append(remaining)
                future = pool.submit(execute_group, (ready, prepared, str(args.output), snapshot, config,
                                                     dose, args.threads_per_worker, preprocessing_json, args.device))
                active[future] = ready
            if active:
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    group = active.pop(future)
                    try:
                        records = future.result()
                        results.extend({"id": r["task"]["id"], "status": r["status"]} for r in records)
                    except Exception as error:
                        results.extend({"id": t["id"], "status": "failed", "error": repr(error)} for t in group)
                        print(json.dumps({"group_failure": [t["id"] for t in group], "error": repr(error)}), flush=True)
            elif pending:
                time.sleep(10)
            atomic_write_json(args.output / "progress.json", dict(status="running", results=results,
                active=[[t["id"] for t in group] for group in active.values()],
                pending=[t["id"] for group in pending for t in group], prepared_studies=sorted(prepared)))
    atomic_write_json(args.output / "progress.json", dict(status="completed" if all(r["status"] == "completed" for r in results)
        else "failed", results=results, active=[], pending=[]))
    if any(r["status"] != "completed" for r in results):
        raise RuntimeError("source matrix has unresolved failures; see progress.json")


if __name__ == "__main__":
    main()
