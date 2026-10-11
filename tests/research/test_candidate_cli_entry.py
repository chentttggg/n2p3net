"""Real direct-script CLI admission and exit status from an unrelated cwd."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from tests import PROJECT_ROOT
from tests.support.bnci import bnci_mat_fields
from tests.support.candidate_programs import study_plan
from transfer.ensemble import EnsembleProgram


@pytest.fixture(scope="module")
def two_participant_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("candidate-direct-cli-inputs")
    plan = json.loads(study_plan(directory).read_bytes())
    # One existing AR0 classical expert makes real fits and saved replay cheap;
    # the direct CLI still uses the actual shared CloudExecutor and spawn.
    plan["experts"] = plan["experts"][:1]
    plan["mixtures"] = [EnsembleProgram.equal_weights("raw", ("raw",)).record()]
    plan["baseline_mixture"] = "raw"
    path = directory / "A02.mat"
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(40).normal(0, 4, fields["X"].shape).astype(np.float32)
    savemat(path, {"data": fields})
    plan["records"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plan.update(workers=2, cpu_threads_per_worker=1, cpu_budget=3, source_workers=2, target_workers=2,
                source_worker_memory_gib=.25, target_worker_memory_gib=.25)
    return plan


@pytest.mark.parametrize("worker_ingress_failure", [False, True], ids=["completed", "failed-source-contract"])
def test_real_direct_cli_spawns_actors_without_pythonpath_and_reports_result_status(
        tmp_path, two_participant_plan, worker_ingress_failure):
    plan = json.loads(json.dumps(two_participant_plan))
    if worker_ingress_failure:
        for index, row in enumerate(plan["records"]):
            row["sha256"] = f"{index:064x}"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "results"
    outside = tmp_path / "unrelated-working-directory"
    outside.mkdir()
    assert not outside.is_relative_to(PROJECT_ROOT)
    environment = {name: value for name, value in os.environ.items() if name.upper() != "PYTHONPATH"}
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    assert "PYTHONPATH" not in environment
    script = PROJECT_ROOT / "experiments/run_candidate_ensemble_study.py"
    completed = subprocess.run([sys.executable, "-B", str(script), "--plan", str(plan_path),
        "--output", str(output)], cwd=outside, env=environment, capture_output=True, text=True)

    expected_status = "failed" if worker_ingress_failure else "completed"
    assert completed.returncode == int(worker_ingress_failure), (completed.stdout, completed.stderr)
    summary = json.loads(completed.stdout)
    assert summary == {"status": expected_status, "output": str(output.resolve())}
    result_path = output / "result.json"
    assert result_path.is_file() and not (output / "report.json").exists()
    report = json.loads(result_path.read_bytes())
    assert report["schema"] == "n2p3_candidate_ensemble_report/1"
    assert report["status"] == expected_status and report["evidence_scope"] == "software_correctness"
    assert json.loads((output / "progress.json").read_bytes())["status"] == expected_status

    diagnostics = sorted((output / "worker_diagnostics").rglob("*.events.jsonl"))
    assert len(diagnostics) == 2
    executions = []
    for path in diagnostics:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        execution = next(row for row in events if row["phase"] == "adapter_execution")
        assert execution["role"]["adapter"] == "experiments.run_candidate_ensemble_study:candidate_participant_adapter"
        assert execution["role"]["resources"]["native_threads"] == 1
        assert events[-1]["phase"] == "cache_group_cleanup_returned"
        assert events[-1]["status"] == expected_status
        executions.append(execution)
    assert len({row["pid"] for row in executions}) == 2
    assert {row["identity"]["cloud_task_id"] for row in executions} == {"candidate:000", "candidate:001"}
    assert "ModuleNotFoundError" not in completed.stderr
    if worker_ingress_failure:
        assert report["participants"] == [] and report["unresolved_record_indexes"] == [0, 1]
        assert len(report["execution_failures"]) == len(report["requested_pipeline_cells"]) == 2
        assert all(row["failure"]["stage"] == "adapter_execution" for row in report["execution_failures"])
        assert all(row["failure"]["type"] == "ValueError" for row in report["execution_failures"])
        assert not list(output.rglob("*.started.json"))
    else:
        assert {row["record_index"] for row in report["participants"]} == {0, 1}
        assert sum(len(row["requested"]) for row in report["participants"]) == 4
        assert all(row["status"] == "completed" for row in report["participants"])
        assert all(cell["saved_replay_exact"] for row in report["participants"] for cell in row["predictions"])
        actual_source = json.loads((output / "executed_source.json").read_bytes())
        assert actual_source["experiments/run_candidate_ensemble_study.py"] == hashlib.sha256(script.read_bytes()).hexdigest()
