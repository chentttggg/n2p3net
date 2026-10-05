"""Run the declared P1/P2 matrix with shared source trajectories and target fits."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for search in (ROOT, ROOT / "src"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from experiments.run_source_training_chain import (  # noqa: E402
    _evaluate_outer_group,
    execute_source_chain,
)
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402


def target_job_groups(source_record, plan, output, *, device, program="P1_P2"):
    """Keep each cache/subject shard resident across the declared source states."""
    if program not in {"P1_P2", "unified"}:
        raise ValueError("unsupported domain transfer program")
    workers = source_record["resource_policy"]["target_workers"]
    task_outer = {str(Path(checkpoint).resolve()): task["key"].split("/")[0]
                  for task in source_record["tasks"] for checkpoint in task.get("checkpoints", {}).values()}
    outers = {outer["name"]: outer for outer in plan["outer_folds"]}
    groups, declarations = {}, []
    # P1 precedes P2 so identical CE/full, scratch and LDA fits can be reused.
    for stage in (("unified",) if program == "unified" else ("P1", "P2")):
        for entry in source_record["outer_evaluation_jobs"]:
            source_plan = Path(entry["plan"])
            payload = json.loads(source_plan.read_text(encoding="utf-8"))
            if stage == "P1" and payload["source_arm"] != "P2_source_ce":
                continue
            outer_name = task_outer[str(Path(payload["checkpoint"]).resolve())]
            all_subjects = entry["argv"][entry["argv"].index("--subjects") + 1].split(",")
            for shard, old_output in enumerate(entry["shards"]):
                # The source runner declares subjects in the same strided shard order.
                subjects = all_subjects[shard::workers]
                if not subjects:
                    continue
                destination = output / "targets" / stage / source_plan.stem / f"shard{shard}"
                job = {"plan": str(source_plan), "subjects": list(subjects), "output": str(destination),
                       "cpu_threads": source_record["resource_policy"].get("native_threads_per_worker", 1),
                       "device": device, "target_cache": payload["target_cache"], "target_stage": stage}
                groups.setdefault((payload["target_cache"], shard), []).append(job)
                declarations.append({**job, "outer": outer_name,
                    "family_exclusion": bool(outers[outer_name].get("family_exclusion", False)),
                    "source_arm": payload["source_arm"], "source_seed": payload["source_seed"],
                    "target_seed": payload["target_seed"], "study": payload["target_study"],
                    **({"source_factors": payload["source_factors"]} if program == "unified" else {}),
                    "source_runner_reserved_output": old_output})
    return list(groups.values()), declarations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--program", choices=("P1_P2", "unified"), default="P1_P2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.program == "P1_P2" and "ce" not in plan.get("source_objectives", {"ce": {}}):
        raise ValueError("the combined matrix requires its CE control to be named ce")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    started = time.time()
    progress = {"schema": "n2p3_domain_transfer_matrix/1", "status": "running_sources", "program": args.program,
                "started_unix": started, "plan": plan, "completed_groups": [], "failed_groups": []}
    atomic_write_json(output / "matrix.json", progress)
    with cpu_thread_budget(1):
        source = execute_source_chain(plan, output=output / "sources", dry_run=args.dry_run,
            device=args.device, stages=("unified_source",) if args.program == "unified" else ("P2_source",), evaluate_outer=False)
    if args.dry_run:
        progress.update(status="planned", fitting_performed=False, source_tasks=source["task_count"],
                        target_stages=["unified"] if args.program == "unified" else ["P1", "P2"])
        atomic_write_json(output / "matrix.json", progress)
        return
    if source.get("source_failure_count") or source.get("preflight_failures"):
        progress.update(status="source_failure", source_record=str(output / "sources/source_chain_record.json"))
        atomic_write_json(output / "matrix.json", progress)
        raise RuntimeError("source matrix contains failures; preserve completed trajectories before resuming")
    groups, declarations = target_job_groups(source, plan, output, device=args.device, program=args.program)
    atomic_write_json(output / "target_jobs.json", {"jobs": declarations})
    progress.update(status="running_targets", target_groups=len(groups), target_jobs=len(declarations),
                    resource_policy=source["resource_policy"])
    atomic_write_json(output / "matrix.json", progress)
    workers = source["resource_policy"]["target_workers"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_evaluate_outer_group, group): index for index, group in enumerate(groups)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                progress["completed_groups"].append({"index": index, "results": future.result()})
            except Exception as error:
                progress["failed_groups"].append({"index": index, "type": type(error).__name__, "error": str(error)})
            progress["elapsed_seconds"] = time.time() - started
            atomic_write_json(output / "matrix.json", progress)
    failures = sum(row["failed_cells"] for group in progress["completed_groups"] for row in group["results"])
    progress.update(status="failed" if failures or progress["failed_groups"] else "completed",
                    failed_cells=failures, elapsed_seconds=time.time() - started)
    atomic_write_json(output / "matrix.json", progress)
    if progress["status"] != "completed":
        raise RuntimeError("target matrix contains failures; completed results remain available")


if __name__ == "__main__":
    main()
