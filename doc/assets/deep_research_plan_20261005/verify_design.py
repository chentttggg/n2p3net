"""Independent plan checks and counterexamples for evaluation and scheduling claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def verify_plan(plan):
    tasks = {row["id"]: row for row in plan["tasks"]}
    assert len(tasks) == len(plan["tasks"])
    declared_ids = {row["id"] for row in plan["wishlist"]}
    lines = (HERE / "wishlist_source.zh.md").read_text(encoding="utf-8").splitlines()
    source_ids = set()
    group = number = 0
    for line_number, line in enumerate(lines, 1):
        if line.startswith("这份愿望清单已经保存"):
            break
        if re.match(r"^\*\*[一二三四五六七八九十]+、", line):
            group += 1
            number = 0
        if group and (line.startswith("- ") or re.match(r"^\d+\. ", line)):
            number += 1
            identifier = f"W{group:02d}.{number:02d}"
            source_ids.add(identifier)
            row = next(row for row in plan["wishlist"] if row["id"] == identifier)
            assert row["source_line"] == line_number
            assert row["text"] == re.sub(r"^(?:- |\d+\. )", "", line)
    assert declared_ids == source_ids and len(source_ids) == 123
    assert all(row["tasks"] and set(row["tasks"]) <= tasks.keys() for row in plan["wishlist"])
    for name, expected in plan["source_text_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    done = set()
    layers = []
    while len(done) < len(tasks):
        ready = sorted(name for name, task in tasks.items() if name not in done
                       and set(task["depends_on"]) <= done)
        assert ready, "unknown dependencies or a dependency cycle"
        layers.append(ready)
        done.update(ready)
    for row in tasks.values():
        assert row["priority"] in {"P0", "P1", "P2"}
        assert row["research_question"] and row["exit_evidence"] and row["falsification"]
        assert row["implementation_status"] == row["execution_status"] == "not_started"
        assert all((ROOT / name).is_file() for name in row["owner_files"])
    core_dependencies = set(tasks["B12"]["depends_on"])
    assert "B04" in core_dependencies and "B14" in core_dependencies
    assert not core_dependencies & {"B05", "B06", "B07", "B08", "B09", "B10", "B11"}
    floor = plan["resource_contract"]["cloud"]["cloud_resource_floor"]
    assert floor["cpu_threads"] >= 25 and floor["gpu_workers"] >= 4
    assert sum(plan["resource_contract"]["cpu_partition_at_4_workers"]) == 25
    return {"wishlist_items": len(source_ids), "tasks": len(tasks), "dependency_layers": layers,
            "ready_implementation_tasks": sorted(name for name, task in tasks.items() if not task["depends_on"]),
            "all_future_execution_not_started": True}


def evaluation_counterexamples():
    # A small easy participant must count as one participant, not one trial.
    requested = np.array([2, 8], dtype=float)
    correct = np.array([2, 0], dtype=float)
    macro = float(np.mean(correct / requested))
    pooled = float(correct.sum() / requested.sum())
    assert macro == 0.5 and pooled == 0.2
    # Dropping unavailable decisions improves the apparent score without any new hit.
    operational = 6 / 10
    valid_only = 6 / 6
    assert operational == 0.6 and valid_only == 1.0
    # Better NLL alone cannot justify a worse decision classifier.
    base = np.array([[0.51, 0.49], [0.51, 0.49], [0.999, 0.001]])
    alternative = np.array([[0.49, 0.51], [0.49, 0.51], [0.01, 0.99]])
    truth = np.array([0, 0, 1])
    base_hit = float(np.mean(base.argmax(1) == truth))
    alt_hit = float(np.mean(alternative.argmax(1) == truth))
    base_nll = float(-np.log(base[np.arange(3), truth]).mean())
    alt_nll = float(-np.log(alternative[np.arange(3), truth]).mean())
    assert alt_nll < base_nll and alt_hit < base_hit
    # Identical seed copies do not create new independent participants.
    effects = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.25, -0.15, 0.1])
    repeated = np.repeat(effects, 3)
    cluster_variance = np.var(effects, ddof=0) / len(effects)
    pseudoreplicated_variance = np.var(repeated, ddof=0) / len(repeated)
    assert np.isclose(cluster_variance / pseudoreplicated_variance, 3)
    # Incompleteness can lower valid-only NLL just by losing difficult decisions.
    nll_all = np.array([0.1, 0.2, 7.0])
    assert nll_all[:2].mean() < nll_all.mean()
    return {"participant_macro_hit": macro, "incorrect_pooled_hit": pooled,
            "requested_hit": operational, "incorrect_valid_only_hit": valid_only,
            "nll_better_hit_worse": {"base_hit": base_hit, "candidate_hit": alt_hit,
                                     "base_nll": base_nll, "candidate_nll": alt_nll},
            "three_seed_pseudoreplication_variance_ratio": float(cluster_variance / pseudoreplicated_variance),
            "missing_nll_selection_bias_reproduced": True}


def scheduling_counterexamples():
    # The repeated-input batch reduces physical loading, not scientific fit count.
    jobs = [(subject, seed, arm) for subject in range(8) for seed in range(3)
            for arm in range(6)]
    naive_loads = len(jobs)
    grouped_loads = len({(subject, seed) for subject, seed, _ in jobs})
    assert naive_loads == 144 and grouped_loads == 24
    # Four processes with six native threads plus the coordinator consume 25.
    assert 4 * 6 + 1 == 25
    # A real DAG tail with one ready job has one active worker; capacity is still four.
    ready = 1
    capacity = 4
    active = min(ready, capacity)
    assert active == 1 and capacity == 4
    return {"logical_cells": naive_loads, "batched_input_loads": grouped_loads,
            "illustrative_load_count_ratio": naive_loads / grouped_loads,
            "measured_speedup_claimed": False, "dependency_tail": {"ready": ready, "capacity": capacity, "active": active}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = HERE / "research_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    record = {"evidence_scope": "software_and_mathematical_counterexamples",
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "coverage_and_graph": verify_plan(plan),
              "evaluation": evaluation_counterexamples(), "scheduling": scheduling_counterexamples(),
              "status": "passed", "public_efficacy_matrix_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": record["status"], "wishlist": 123, "tasks": 15}))


if __name__ == "__main__":
    main()
