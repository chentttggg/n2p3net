"""Merge matched basic/QC study views through the shared bounded source owner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.epochs import _read_epoch_cache_record  # noqa: E402
from data.source_records import file_sha256, verify_prepared_cache  # noqa: E402
from experiments.prepare_consistent_source_cohorts import STUDIES  # noqa: E402
from experiments.prepare_multidomain_source import main as merge  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_execution import SourceInputPhases  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--memory-gib", type=float, default=14.)
    args = parser.parse_args(argv)
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    expected = {p + ":" + s for p in ("basic", "QC") for s in STUDIES}
    if set(queue["completed"]) != expected:
        raise ValueError("both complete twelve-study source views are required before pooling")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = (json.loads(args.binding.read_text(encoding="utf-8")) if args.binding.exists() else
        {"schema": "n2p3_matched_source_pool_binding/1", "status": "merging", "pools": {}})
    for pipeline in ("basic", "QC"):
        parents = {study: queue["completed"][pipeline + ":" + study] for study in STUDIES}
        identity = {study: {"cache_sha256": row["cache_sha256"],
            "admission_sha256": file_sha256(row["preparation"]), "requested_records": row["records"],
            "no_evidence_records": row.get("no_evidence_records", 0)} for study, row in parents.items()}
        if pipeline in binding["pools"]:
            previous = binding["pools"][pipeline]
            if previous["parent_inputs"] != identity:
                raise ValueError("source pool parent identities changed after publication")
            verify_prepared_cache(previous["cache"], cache_sha256=previous["cache_sha256"],
                cache_record_sha256=previous["cache_record_sha256"])
            continue
        path = output / (pipeline + "_all12.npz")
        phase = SourceInputPhases(output / (pipeline + "_merge_phases.json"))
        cmd = ["--output", str(path), "--target-channels", "Fz,P3,Pz,P4,PO8,Oz",
            "--event-contract", "binary_evidence", "--reference-mode", "established" if pipeline == "QC" else "measured_channel",
            "--missing-channel-policy", "mask", "--source-workers", "1", "--cpu-threads", "1",
            "--source-memory-budget-bytes", str(int(args.memory_gib * 2**30)),
            "--verified-snapshot-directory", str(output / "snapshots"), "--uncompressed"]
        if pipeline == "basic":
            cmd.extend(["--reference-channel", "Cz"])
        for study in STUDIES:
            row = parents[study]
            source_metadata, _ = _read_epoch_cache_record(row["cache"])
            policy = source_metadata["provenance"].get("artifact_preprocessing_policy", {})
            if policy.get("enabled") != (pipeline == "QC"):
                raise ValueError("a source study belongs to the other preprocessing program")
            cmd.extend(["--source", study + "=" + row["cache"]])
        production = merge(cmd, retain_production=True, phases=phase)
        attestation = production.verify()
        metadata, _ = _read_epoch_cache_record(path)
        binding["pools"][pipeline] = {"cache": str(path), "cache_sha256": attestation["sha256"],
            "cache_record_sha256": file_sha256(path.with_suffix(".record.json")),
            "shape": metadata["shape"], "source_studies": list(STUDIES), "parent_inputs": identity,
            "preprocessing_policy": metadata["provenance"]["artifact_preprocessing_policy"]}
        atomic_write_json(args.binding, binding)
        print(pipeline, "source pool ready", metadata["shape"], flush=True)
    binding["status"] = "completed"
    atomic_write_json(args.binding, binding)


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    main()
