"""Validate the finalized acquisition ledger before interpreting stored samples."""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def validate_recording_clock(root: Path, recording: dict, *, sample_rate_hz: float,
                             duration_seconds: float) -> dict:
    clock = recording.get("clock")
    if clock is None:
        return {"alignment_level": "legacy_host_only", "absolute_offset_bound_seconds": None}
    if not isinstance(clock, dict) or clock.get("schema") != "brainsync-sample-clock/1":
        raise ValueError("unsupported recording clock schema")
    paths = {}
    for key in ("metadata_path", "journal_path"):
        value = clock.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("recording clock requires contained metadata and journal paths")
        path = (root / value).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("recording clock path is missing or outside the session")
        paths[key] = path
    metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("recording clock sidecar must be an object")
    # Sidecar location hints can name the original workstation. Only the
    # contained manifest paths above are opened after a session is relocated.
    if ({key: value for key, value in clock.items() if key not in paths}
            != {key: value for key, value in metadata.items() if key not in paths}):
        raise ValueError("recording clock manifest and finalized sidecar disagree")
    if (metadata.get("finalized") is not True or metadata.get("integrity_failed") is not False
            or metadata.get("sample_rate_hz") != sample_rate_hz
            or metadata.get("alignment_level") != "delivery_anchored"
            or metadata.get("absolute_offset_bound_seconds") is not None):
        raise ValueError("recording clock integrity, rate or alignment declaration is invalid")
    stored = metadata.get("stored_samples")
    acquired = metadata.get("acquired_samples")
    serialized = metadata.get("edf_samples")
    padding = metadata.get("padding_samples")
    if (any(type(value) is not int for value in (
            stored, acquired, serialized, padding, metadata.get("voltage_samples")))
            or stored <= 0 or acquired != stored or metadata["voltage_samples"] != stored
            or serialized < acquired or padding != serialized - acquired):
        raise ValueError("recording clock acquired and serialized sample counts disagree")
    try:
        duration = Decimal(str(duration_seconds))
        rate = Decimal(str(sample_rate_hz))
    except InvalidOperation as error:
        raise ValueError("recording clock duration and rate must be finite numbers") from error
    if (not duration.is_finite() or not rate.is_finite() or duration <= 0 or rate <= 0
            or duration * rate != serialized):
        raise ValueError("recording clock and voltage duration disagree")
    contents = paths["journal_path"].read_bytes()
    if hashlib.sha256(contents).hexdigest() != metadata.get("journal_sha256"):
        raise ValueError("recording clock journal checksum differs")
    rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
    received_cursor = stored_cursor = 0
    for index, row in enumerate(rows):
        received, accepted = row.get("n_received"), row.get("n_stored")
        if (row.get("batch_id") != index or row.get("received_start") != received_cursor
                or type(received) is not int or type(accepted) is not int
                or received < 0 or accepted not in (0, received)
                or (accepted and row.get("stored_start") != stored_cursor)
                or (accepted and row.get("sequence_valid") is not True)):
            raise ValueError("recording clock batch offsets or accepted sequence are invalid")
        for name in ("seq_num", "delta_time_raw"):
            if len(row.get(name, [])) != received:
                raise ValueError("recording clock sample counters do not align with received rows")
        received_cursor += received
        stored_cursor += accepted
    if (stored_cursor != stored or received_cursor != metadata.get("received_samples")
            or len(rows) != metadata.get("batch_count")):
        raise ValueError("recording clock batch totals disagree with the finalized recording")
    cursor = 0
    segments = metadata.get("segments", [])
    for index, segment in enumerate(segments):
        if (segment.get("segment_id") != index or segment.get("stored_start") != cursor
                or type(segment.get("stored_stop")) is not int or segment["stored_stop"] < cursor):
            raise ValueError("recording clock acquisition segments are not contiguous in stored coordinates")
        cursor = segment["stored_stop"]
    if cursor != stored:
        raise ValueError("recording clock segments do not cover stored samples")
    for row in rows:
        if row["n_stored"]:
            segment = next((item for item in segments if item["segment_id"] == row.get("segment_id")), None)
            if segment is None or not (segment["stored_start"] <= row["stored_start"]
                    and row["stored_start"] + row["n_stored"] <= segment["stored_stop"]):
                raise ValueError("stored batch crosses an acquisition segment")
    return metadata
