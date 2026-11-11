"""Cold-replay native startup cuts from original EDFs and verify live ingress."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import edfio  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from data.brainsync import load_brainsync_session  # noqa: E402
from data.recording_review import RecordingExcludedError  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    verified_executed_snapshot(args.source_manifest, root=ROOT)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["recordings"]:
        source = edfio.read_edf(record["original_raw"])
        stop = record["stop_sample"]
        assert file_sha256(record["original_raw"]) == record["raw_sha256"]
        assert file_sha256(record["raw_cut"]) == record["raw_cut_sha256"]
        source.update_data_record_duration(1. / record["sample_rate_hz"])
        source.slice_between_seconds(stop / record["sample_rate_hz"],
                                     record["original_sample_count"] / record["sample_rate_hz"])
        stream = io.BytesIO()
        source.write(stream)
        assert hashlib.sha256(stream.getvalue()).hexdigest() == record["raw_cut_sha256"]
        raw = mne.io.read_raw_edf(record["original_raw"], preload=False, verbose="ERROR")
        cropped = mne.io.read_raw_edf(record["raw_cut"], preload=False, verbose="ERROR")
        try:
            np.testing.assert_array_equal(raw.get_data(start=stop), cropped.get_data())
        finally:
            raw.close()
            cropped.close()
        row = {"session_id": record["session_id"], "EDF_replay_sha256_equal": True,
               "MNE_voltage_suffix_bitwise_equal": True}
        try:
            data = load_brainsync_session(record["original_root"])
            n = len(record["dropped_event_indices"])
            assert np.all(data.event_timeline.evidence_indices[:n] == -1)
            spans = data.provenance["preprocessing_execution"]["source_context"]["filter_segments"]
            assert all(start >= stop for intervals in spans.values() for start, end in intervals)
            row.update(status="admitted", requested_events=data.event_timeline.n_events, epochs=data.n_epochs)
        except RecordingExcludedError:
            assert record["recording_review"] is not None
            row["status"] = "whole_record_exclusion_preserved"
        rows.append(row)
        print("REPLAYED", record["session_id"], row["status"], flush=True)
    atomic_write_json(args.output, {"schema": "brainsync_startup_cut_cold_replay/1", "status": "passed",
        "batch_manifest_sha256": file_sha256(args.manifest),
        "executed_source_manifest": str(args.source_manifest.resolve()),
        "executed_source_manifest_sha256": file_sha256(args.source_manifest),
        "records": rows, "new_fits": 0, "raw_files_modified": 0})


if __name__ == "__main__":
    main()
