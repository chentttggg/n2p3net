"""Run every evaluation arm of one partition in a single result process.

The batch process keeps the attested target cache and checkpoint payload
resident between arms and delegates each arm to ``run_candidate_cross_decision.main``.
The batch ledger is the audited task output; the individual result files keep
their existing schemas and remain the manifest inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments import run_candidate_cross_decision as result_runner  # noqa: E402
from experiments.build_candidate_cross_decision_manifest import RESULT_SCHEMA  # noqa: E402
from experiments.result_batch_contract import (  # noqa: E402
    RESULT_BATCH_JOBS_SCHEMA,
    RESULT_BATCH_LEDGER_SCHEMA,
)
from research.execution import atomic_write_json  # noqa: E402

_COMMON_FLAGS = (
    "--dataset-cache",
    "--checkpoint",
    "--source-snapshot-manifest",
    "--identity-exclusion-policy",
    "--calibration-selections",
    "--test-reps",
    "--min-evidence-level",
    "--seed",
    "--target-subjects-file",
    "--device",
    "--cpu-threads",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    try:
        return str(argv[argv.index(flag) + 1])
    except IndexError as error:
        raise ValueError(f"job argv contains incomplete flag {flag}.") from error


def _output_from_argv(argv: Sequence[str]) -> Path:
    value = _flag_value(argv, "--output")
    if value is None:
        raise ValueError("job argv lacks --output.")
    return Path(value).resolve()


def _validate_jobs_record(record: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], ...]:
    if record.get("schema") != RESULT_BATCH_JOBS_SCHEMA:
        raise ValueError(f"{path} has an unsupported result-batch jobs schema.")
    raw_jobs = record.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("result-batch jobs record must declare at least one job.")
    if record.get("job_count") != len(raw_jobs):
        raise ValueError("result-batch job_count disagrees with jobs length.")
    jobs: list[dict[str, Any]] = []
    outputs: set[Path] = set()
    common: dict[str, str] = {}
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"result-batch job {index} must be an object.")
        arm_name = raw_job.get("arm_name")
        if not isinstance(arm_name, str) or not arm_name.strip():
            raise ValueError(f"result-batch job {index} lacks arm_name.")
        raw_argv = raw_job.get("argv")
        if not isinstance(raw_argv, list) or not raw_argv or not all(
            isinstance(value, str) and "\x00" not in value for value in raw_argv
        ):
            raise ValueError(f"result-batch job {arm_name!r} has an invalid argv list.")
        output = _output_from_argv(raw_argv)
        if output in outputs:
            raise ValueError(f"result-batch job {arm_name!r} repeats an output path.")
        outputs.add(output)
        for flag in _COMMON_FLAGS:
            value = _flag_value(raw_argv, flag)
            if value is None:
                raise ValueError(
                    f"result-batch job {arm_name!r} lacks required shared flag {flag}."
                )
            previous = common.setdefault(flag, value)
            if value != previous:
                raise ValueError(
                    f"result-batch job {arm_name!r} disagrees on shared flag {flag}."
                )
        evaluation_manifest = _flag_value(raw_argv, "--evaluation-snapshot-manifest") or (
            common["--source-snapshot-manifest"]
        )
        previous_evaluation = common.setdefault("--evaluation-snapshot-manifest", evaluation_manifest)
        if evaluation_manifest != previous_evaluation:
            raise ValueError(
                f"result-batch job {arm_name!r} disagrees on shared evaluation snapshot."
            )
        jobs.append({"arm_name": arm_name, "argv": tuple(raw_argv), "output": output})
    return tuple(jobs)


def _validated_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _ledger_job(job: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(job["output"])
    if not output.is_file():
        raise ValueError(f"result arm output is missing after execution: {output}.")
    record = _read_json_mapping(output, f"result arm output {output}")
    if record.get("schema") != RESULT_SCHEMA or record.get("run_status") not in {
        "completed",
        "completed_with_selection_failures",
    }:
        raise ValueError(f"result arm output is not a completed current schema: {output}.")
    evaluation_digest = _validated_digest(
        record.get("evaluation_contract_digest"), "result evaluation_contract_digest"
    )
    decision_digest = _validated_digest(
        record.get("decision_plan_digest"), "result decision_plan_digest"
    )
    return {
        "arm_name": job["arm_name"],
        "output": str(output),
        "sha256": _sha256_file(output),
        "byte_size": output.stat().st_size,
        "evaluation_contract_digest": evaluation_digest,
        "decision_plan_digest": decision_digest,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs_path = args.jobs_file.resolve()
    output_path = args.output.resolve()
    jobs = _validate_jobs_record(_read_json_mapping(jobs_path, "result-batch jobs"), jobs_path)

    ledger_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        print(
            f"[batch:{index + 1}/{len(jobs)}] {job['arm_name']} -> {job['output']}",
            flush=True,
        )
        result_runner.main(job["argv"])
        ledger_jobs.append(_ledger_job(job))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ledger = {
        "schema": RESULT_BATCH_LEDGER_SCHEMA,
        "completed": True,
        "jobs_file": str(jobs_path),
        "jobs_file_sha256": _sha256_file(jobs_path),
        "job_count": len(ledger_jobs),
        "jobs": ledger_jobs,
    }
    atomic_write_json(output_path, ledger)
    print(f"[summary] {output_path}", flush=True)


if __name__ == "__main__":
    main()
