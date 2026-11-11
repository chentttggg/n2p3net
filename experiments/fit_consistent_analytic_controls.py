"""Fit the deterministic LDA/xDAWN controls for available matched target inputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=["waveform_lda", "xdawn_erp"], default=["waveform_lda", "xdawn_erp"])
    parser.add_argument("--pipelines", nargs="+", choices=["basic", "QC"], default=["basic", "QC"])
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = {}
    for path in args.prepared:
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, cell in record["cells"].items():
            if cell["status"] == "prepared" and (name not in cells or cell["policy"]["enabled"]):
                cells[name] = cell
    ledger = {"schema": "n2p3_consistent_analytic_control_fits/1", "status": "running",
        "seed": 20261031, "source_fits": 0,
        "scope": "fixed full-C target-only controls; source NN and scratch fits remain separate",
        "fits": {}}
    for name, cell in cells.items():
        run = output / name
        run.mkdir()
        pipeline = name.split("_K")[0]
        if pipeline not in args.pipelines:
            continue
        rows = [{"session_id": r["session_id"], "subject_id": "P001", "target": r["target"],
            "paths": {pipeline: {"path": r["path"], "sha256": r["sha256"]}}} for r in cell["rows"]]
        plan = {"schema": "n2p3_consistent_analytic_cell/1", "calibration_order": cell["calibration_sessions"],
            "test_sessions": cell["test_sessions"], "rows": rows, "epochs": 90, "batch_size": 512,
            "learning_rate": .001, "preprocessing_policy": cell["policy"]}
        atomic_write_json(run / "plan.json", plan)
        for family in args.families:
            key = name + "_" + family
            job = {"id": key, "run_root": str(run), "output": str(run / family), "pipeline": pipeline,
                "family": family, "seed": 20261031, "dose": len(cell["calibration_sessions"]),
                "checkpoint": None, "device": "cpu", "fixed_analytic_fit": True}
            job_path = run / (family + "_job.json")
            atomic_write_json(job_path, job)
            with (run / (family + ".log")).open("w", encoding="utf-8") as stream:
                result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "experiments/run_brainsync_real_matrix.py"),
                    "--job-file", str(job_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            path = run / family / "result.json"
            fitted = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            ledger["fits"][key] = {"status": fitted.get("status", "process_failure"), "result": str(path),
                "returncode": result.returncode, "failure": fitted.get("failure")}
            atomic_write_json(output / "checkpoint.json", ledger)
            print(key, ledger["fits"][key]["status"], flush=True)
    ledger["status"] = "completed" if all(f["status"] == "completed" for f in ledger["fits"].values()) else "has_failures"
    atomic_write_json(output / "checkpoint.json", ledger)


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "2"
    main()
