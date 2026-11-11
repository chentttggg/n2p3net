"""Verify actual auxiliary channels in the matched public source intake."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.eog_preprocessing import EOGPreprocessingPolicy  # noqa: E402
from data.epochs import load_preprocessing_spec  # noqa: E402
from data.source_parsers import validate_source_manifest  # noqa: E402
from data.source_records import (  # noqa: E402
    _verified_source_record,
    file_sha256,
    prepare_source_record,
)
from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-sentinels", nargs="*", default=[])
    parser.add_argument("--full-qc", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "availability.json"
    if record_path.exists():
        raise FileExistsError("audit requires a fresh output; preserve previous evidence")
    record = {"schema": "n2p3_matched_EOG_intake_audit/1", "status": "running",
        "scope": "one manifest-declared original per study; representative availability, not whole-cohort admission",
        "EOG_role": "preprocessing_only; never synthesize from EEG", "studies": {}}
    spec = load_preprocessing_spec(ROOT / "configs/brainsync_bidirectional_preprocessing.json")
    for manifest_path in sorted(args.manifests.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "n2p3_source_records/1":
            continue
        items = validate_source_manifest(manifest)
        item = items[0]
        study = item["study"]
        row = {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha256(manifest_path),
            "manifest_record_count": len(items), "representative_record": item}
        try:
            with _verified_source_record(item, root=args.raw_root.resolve(), output=output, retain_eog=True) as loaded:
                raw, events, _, inputs, _ = loaded
                row.update(status="read_verified_original", consumed_inputs=inputs,
                    channels=[{"name": n, "type": kind} for n, kind in
                              zip(raw.ch_names, raw.get_channel_types(), strict=True)],
                    actual_parser_EOG=[n for n, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
                                       if kind == "eog"],
                    sample_rate_hz=float(raw.info["sfreq"]), native_samples=int(raw.n_times),
                    original_events=len(events))
            if study in args.prepare_sentinels:
                row["preparation_sentinels"] = {}
                for name, enabled in (("basic", False), ("QC", True)):
                    policy = (EOGPreprocessingPolicy.corrected(continuous_qc=args.full_qc) if enabled else
                        EOGPreprocessingPolicy("basic_eog_ols_v1", False, "record_unavailable"))
                    row["preparation_sentinels"][name] = prepare_source_record(item,
                        root=args.raw_root.resolve(), output=output / "sentinels" / study / name,
                        preprocessing=spec, eog_policy=policy)
        except Exception as error:
            row.update(status="unavailable_or_failed", failure={"type": type(error).__name__, "message": str(error)})
        record["studies"][study] = row
        atomic_write_json(record_path, record)
        print(study, row["status"], row.get("actual_parser_EOG"), flush=True)
    record["status"] = "completed_with_unavailable_inputs" if any(
        row["status"] != "read_verified_original" for row in record["studies"].values()) else "completed"
    atomic_write_json(record_path, record)


if __name__ == "__main__":
    main()
