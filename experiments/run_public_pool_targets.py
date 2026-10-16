"""Run the agreed public target matrix using the optimized target task executor."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from data.study import dataset_study_membership  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_matrix import MATRIX_SEEDS, source_matrix_tasks  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.binary import dataset_calibration_splits  # noqa: E402
from transfer.calibration_candidates import waveform_lda_recipe  # noqa: E402
from transfer.checkpoint import load_checkpoint_payload  # noqa: E402
from transfer.recipes import TargetRecipe  # noqa: E402
from transfer.result_reuse import admit_completed_result  # noqa: E402
from transfer.target_experiments import execute_target_tasks  # noqa: E402


def protocol_for(study, declarations=None):
    if declarations is not None and study in declarations:
        return declarations[study]
    if study in {"BI2014a", "BNCI2014-008"}:
        return {"kind": "single_recording_prefix"}
    if study in {"BI2015a", "Huebner2017"}:
        return {"kind": "last_session_holdout", "session_order": ["0", "1", "2"],
                "run_order": ["0"] if study == "BI2015a" else [str(i) for i in range(9)], "test_session": "2"}
    raise ValueError("public targets are restricted to the four predefined original studies")


@lru_cache(maxsize=4)
def target_data(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("public target cache changed")
    return load_epoch_dataset(path, validation="attested")


@lru_cache(maxsize=4)
def source_payload(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("public source checkpoint changed")
    return load_checkpoint_payload(path)


def build_manifest(data, task, subjects, *, policy=None):
    scopes, tasks, adjustments = {}, [], {}
    policy = policy or {"control_pool": "B0", "lda_seed": MATRIX_SEEDS[0], "protocols": {}}
    protocol = protocol_for(task["target"], policy["protocols"])
    for subject in subjects:
        adjusted = {}
        rows, splits = dataset_calibration_splits(data, subject, protocol, (0, 200, 800, 2000), unavailable=adjusted)
        participant = data.identity_table.record_for(subject).authority_key()
        adjustments[subject] = {str(k): value for k, value in adjusted.items()}
        for budget, split in splits.items():
            key = f"{subject}:{budget}"
            train, test = rows[split.calibration_rows], rows[split.test_rows]
            scopes[key] = dict(subject=subject, participant_key=participant, K=budget,
                calibration_rows=train.tolist(), test_rows=test.tolist(), purged_rows=rows[split.purged_rows].tolist())
            common = dict(scope_id=key, participant_key=participant, subject=subject, K=budget)
            if budget == 0:
                tasks.append({**common, "arm": "zero_shot", "recipe": None})
                continue
            recipes = [TargetRecipe(name="full_fine", batch_size=512, max_updates=200, seed=task["seed"])]
            if task["pool"] == policy["control_pool"] and task["dose"] == "full":
                recipes.append(TargetRecipe(name="scratch", initialization="scratch", batch_size=512,
                                             max_updates=200, seed=task["seed"]))
                if task["seed"] == policy["lda_seed"]:
                    recipes.append(waveform_lda_recipe(batch_size=512, seed=policy["lda_seed"]))
            tasks.extend({**common, "arm": recipe.name, "recipe": recipe.record()} for recipe in recipes)
    return dict(schema="n2p3_target_task_matrix/1", stage="public_pool_fixed_updates", source_task=task,
                target_seed=task["seed"], scopes=scopes, tasks=tasks, budget_adjustments=adjustments,
                shared_controls=f"same-study {policy['control_pool']}/full scratch for each paired seed; LDA from first seed")


def target_context(job, data, payload):
    """The same complete execution declaration is used when fitting and resuming."""
    return dict(target_study=job["task"]["target"], target_protocol=protocol_for(job["task"]["target"], job.get("policy", {}).get("protocols")),
        target_cache_sha256=job["cache_sha256"], checkpoint_sha256=job["checkpoint_sha256"],
        source_seed=job["task"]["seed"], evidence_scope="development", evaluation_snapshot=job["snapshot"],
        source_study_membership=payload["source_study_membership"], target_study_membership=dataset_study_membership(data),
        environment=job["environment"], plan=job.get("policy", {}))


def execute_job(job):
    with cpu_thread_budget(1):
        data = target_data(job["cache"], job["cache_sha256"])
        payload = source_payload(job["checkpoint"], job["checkpoint_sha256"])
        manifest = build_manifest(data, job["task"], job["subjects"], policy=job.get("policy"))
        context = target_context(job, data, payload)
        result = execute_target_tasks(data, payload, manifest, output=Path(job["output"]), context=context, device="cuda")
        return dict(id=job["id"], status="completed", completed_cells=result["completed_cells"], failed_cells=result["failed_cells"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--dose-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--matrix-plan", type=Path, help="The same source pool plan used by the source runner")
    parser.add_argument("--target-protocols", type=Path, help="Explicit recording-order protocols for additional target studies")
    parser.add_argument("--wait-for-input-seconds", type=float, default=0,
                        help="Bounded wait for running source/preparation jobs; omission reports missing inputs immediately")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if not np.isfinite(args.wait_for_input_seconds) or args.wait_for_input_seconds < 0:
        parser.error("input wait must be finite and nonnegative")
    snapshot = verified_executed_snapshot(args.snapshot, root=ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    environment = {"torch": torch.__version__, "numpy": np.__version__, "device": "cuda", "threads_per_worker": 1}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    matrix_plan = None if args.matrix_plan is None else json.loads(args.matrix_plan.read_text(encoding="utf-8-sig"))
    declared_tasks = source_matrix_tasks() if matrix_plan is None else source_matrix_tasks(matrix_plan)
    source_tasks = [t for t in declared_tasks if t["family"] == "Bpublic"]
    if not source_tasks:
        raise ValueError("public target evaluation requires declared public source tasks")
    protocols = {} if args.target_protocols is None else json.loads(args.target_protocols.read_text(encoding="utf-8-sig"))
    for study in {t["target"] for t in source_tasks}:
        protocol_for(study, protocols)
    policy = {"control_pool": source_tasks[0]["pool"], "lda_seed": source_tasks[0]["seed"], "protocols": protocols}
    plan_path = args.output / "plan.json"
    declaration = {"source_tasks": source_tasks, "policy": policy, "snapshot": snapshot}
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != declaration:
        raise ValueError("public target resume requires the identical source-pool and control plan")
    atomic_write_json(plan_path, declaration)
    deadline = time.monotonic() + args.wait_for_input_seconds
    requested, active, results, registered = [], {}, [], set()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while True:
            for task in source_tasks:
                if task["id"] in registered:
                    continue
                source_root = args.dose_sources if task["dose"] == "matched" else args.sources
                source_path = source_root / task["id"] / "result.json"
                prep_path = args.prepared / task["target"] / "preparation.json"
                if not source_path.exists() or not prep_path.exists():
                    if time.monotonic() >= deadline:
                        registered.add(task["id"])
                        results.append(dict(id=task["id"], status="failed", error="source/preparation input wait expired"))
                    continue
                source = json.loads(source_path.read_text())
                prepared = json.loads(prep_path.read_text())
                if source["status"] != "completed" or prepared["status"] != "completed":
                    if "failed" in (source["status"], prepared["status"]) or time.monotonic() >= deadline:
                        registered.add(task["id"])
                        results.append(dict(id=task["id"], status="failed", error="source/preparation did not complete"))
                    continue
                if source["task"] != task:
                    raise ValueError("source result differs from the declared pool task")
                checkpoint = source["trajectory"]["checkpoints"]["4000"]
                data = target_data(prepared["cache"], prepared["cache_sha256"])
                subjects = sorted(set(data.subject_ids.astype(str)))
                for offset in range(0, len(subjects), 4):
                    job_id = f"{task['id']}_chunk{offset // 4:03d}"
                    requested.append(dict(id=job_id, task=task, subjects=subjects[offset:offset + 4],
                        cache=prepared["cache"], cache_sha256=prepared["cache_sha256"],
                        checkpoint=checkpoint, checkpoint_sha256=file_sha256(checkpoint), snapshot=snapshot,
                        output=str(args.output / job_id), environment=environment,
                        **({"policy": policy} if matrix_plan is not None or protocols else {})))
                registered.add(task["id"])
            while requested and len(active) < args.workers:
                job = requested.pop(0)
                path = Path(job["output"]) / "record.json"
                if path.exists():
                    data = target_data(job["cache"], job["cache_sha256"])
                    payload = source_payload(job["checkpoint"], job["checkpoint_sha256"])
                    previous = admit_completed_result(path, kind="public_target", expected={
                        **target_context(job, data, payload),
                        "task_manifest": build_manifest(data, job["task"], job["subjects"], policy=job.get("policy")),
                    })
                    results.append(dict(id=job["id"], status="completed", completed_cells=previous["completed_cells"],
                                        failed_cells=previous["failed_cells"], reused=True))
                else:
                    active[pool.submit(execute_job, job)] = job
            if active:
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = dict(id=job["id"], status="failed", error=repr(error))
                    results.append(record)
                    print(json.dumps(record), flush=True)
            elif len(registered) < len(source_tasks):
                time.sleep(10)
            atomic_write_json(args.output / "progress.json", dict(status="running", results=results,
                registered_sources=sorted(registered), active=[j["id"] for j in active.values()], queued=len(requested)))
            if len(registered) == len(source_tasks) and not requested and not active:
                break
    failed_jobs = sum(r["status"] != "completed" for r in results)
    failed_cells = sum(r.get("failed_cells", 0) for r in results)
    atomic_write_json(args.output / "progress.json", dict(status="failed" if failed_jobs or failed_cells else "completed",
        results=results, failed_jobs=failed_jobs, failed_cells=failed_cells))
    if failed_jobs or failed_cells:
        raise RuntimeError("public target matrix has unresolved failures; see progress.json")


if __name__ == "__main__":
    main()
