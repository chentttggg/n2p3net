"""Audit sealed S2 metadata and EDF/BDF integrity without fitting or scoring a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mne
import numpy as np

PRODUCT_CHANNELS = ("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_signal_headers(path: Path) -> list[dict]:
    """Read the standard EDF/BDF signal header fields, without decoding EEG."""
    fields = (
        ("label", 16),
        ("transducer", 80),
        ("unit", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefilter", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    )
    with path.open("rb") as stream:
        fixed = stream.read(256)
        count = int(fixed[252:256])
        result = [{} for _ in range(count)]
        for name, width in fields:
            for entry in result:
                entry[name] = stream.read(width).decode("ascii").strip()
    return result


def parse_original_identity(relative: str) -> dict:
    path = Path(relative)
    subject, session, role, condition, filename = path.parts
    if not subject.startswith("S2_") or session != "SE001":
        raise ValueError(f"Unexpected S2 acquisition identity: {relative}")
    if role not in {"Train", "Test"} or condition not in {"House", "Tool"}:
        raise ValueError(f"Unexpected role/condition: {relative}")
    prefix = f"{subject}_{session}_{condition}_{role}"
    if not filename.startswith(prefix) or not filename.endswith(".edf"):
        raise ValueError(f"Inconsistent filename identity: {relative}")
    record = int(filename[len(prefix) : -4])
    return {
        "subject": int(subject[3:]),
        "session": session,
        "role": role,
        "condition": condition,
        "original_record_number": record,
    }


def validate_events(
    begin: np.ndarray,
    types: np.ndarray,
    rows: list[dict],
    sidecar_rate: float,
    retained_samples: int,
) -> dict:
    flags = np.rint(begin).astype(int)
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("StimulusBegin is not binary")
    edges = np.flatnonzero(np.diff(flags, prepend=0) > 0)
    samples = np.array([int(row["sample"]) for row in rows])
    if not np.array_equal(edges, samples):
        raise ValueError("Original stimulus edges differ from BIDS sample indices")
    original_labels = np.rint(types[edges]).astype(int)
    if not np.isin(original_labels, [0, 1]).all():
        raise ValueError("StimulusType is not binary at event edges")
    labels = np.array([{"Target": 1, "NonTarget": 0}[row["trial_type"]] for row in rows])
    values = np.array([int(row["value"]) for row in rows])
    if not np.array_equal(labels, original_labels) or not np.array_equal(values, labels + 1):
        raise ValueError("Original stimulus labels differ from BIDS labels/values")
    onset_samples = np.array([float(row["onset"]) * sidecar_rate for row in rows])
    error = float(np.max(np.abs(onset_samples - samples)))
    if error > 1e-6 or len(samples) == 0 or samples[-1] >= retained_samples:
        raise ValueError("Invalid derivative event clock or retained event extent")
    return {
        "event_count": len(samples),
        "target_count": int(labels.sum()),
        "event_samples_and_labels_exact": True,
        "onset_max_error_samples": error,
        "last_event_to_end_samples": int(retained_samples - samples[-1]),
    }


def compare_waveforms(original: np.ndarray, converted: np.ndarray, quantum_v: np.ndarray) -> dict:
    if (
        original.shape != converted.shape
        or not np.isfinite(original).all()
        or not np.isfinite(converted).all()
    ):
        raise ValueError("Signal shape or finite-value mismatch")
    errors = np.max(np.abs(original - converted), axis=1)
    # Nearest-integer export has at most half a BDF quantum of error.
    bounds = quantum_v * 0.5001 + np.finfo(float).eps
    if np.any(errors > bounds):
        raise ValueError(f"Signal exceeds BDF quantization bound: {errors.max()} V")
    return {
        "max_absolute_difference_v": float(errors.max()),
        "max_error_in_bdf_quanta": float(np.max(errors / quantum_v)),
        "all_channels_within_half_quantum": True,
    }


def audit_file(root: Path, source: dict, run: int) -> dict:
    identity = parse_original_identity(source["file"])
    if str(identity["subject"]) != str(source["subject"]):
        raise ValueError("Provenance subject contradicts original path")
    edf = root / "sourcedata" / source["file"]
    if edf.stat().st_size != source["bytes"] or digest(edf) != source["sha256"]:
        raise ValueError(f"Original file integrity mismatch: {edf}")
    subject = identity["subject"]
    stem = f"sub-{subject}_ses-0_task-p300_run-{run}"
    base = root / f"sub-{subject}" / "ses-0" / "eeg"
    bdf = base / f"{stem}_eeg.bdf"
    events = base / f"{stem}_events.tsv"
    sidecar = base / f"{stem}_eeg.json"
    channels = base / f"{stem}_channels.tsv"
    metadata = json.loads(sidecar.read_text())
    with events.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    with channels.open(newline="", encoding="utf-8-sig") as stream:
        channel_rows = list(csv.DictReader(stream, delimiter="\t"))
    eeg_headers = [h for h in read_signal_headers(edf) if h["label"].startswith("EEG_")]
    bdf_headers = [h for h in read_signal_headers(bdf) if h["label"] != "BDF Annotations"]
    if len(eeg_headers) != 32 or any(h["unit"] != "uV" for h in eeg_headers + bdf_headers):
        raise ValueError("Unexpected EEG count or physical unit")
    raw = mne.io.read_raw_edf(edf, include="EEG_.*|StimulusBegin|StimulusType", verbose="ERROR")
    converted = mne.io.read_raw_bdf(bdf, verbose="ERROR")
    mapped = [h["label"].removeprefix("EEG_").replace("FP", "Fp") for h in eeg_headers]
    if mapped != converted.ch_names or not set(PRODUCT_CHANNELS).issubset(mapped):
        raise ValueError("Original-to-derivative EEG channel mismatch")
    if [r["name"] for r in channel_rows] != mapped or any(r["type"] != "EEG" for r in channel_rows):
        raise ValueError("channels.tsv does not describe measured EEG")
    n = converted.n_times
    expected_n = (raw.n_times // int(converted.info["sfreq"])) * int(converted.info["sfreq"])
    if n != expected_n or converted.info["sfreq"] != 256.0:
        raise ValueError("Unexpected derivative rate or whole-second crop")
    quantum = np.array(
        [
            (float(h["physical_max"]) - float(h["physical_min"]))
            / (float(h["digital_max"]) - float(h["digital_min"]))
            * 1e-6
            for h in bdf_headers
        ]
    )
    waveform = compare_waveforms(
        raw.get_data(picks=[h["label"] for h in eeg_headers], stop=n), converted.get_data(), quantum
    )
    state = raw.get_data(picks=["StimulusBegin", "StimulusType"])
    event_check = validate_events(state[0], state[1], rows, metadata["SamplingFrequency"], n)
    if event_check["last_event_to_end_samples"] < np.ceil(1.2 * converted.info["sfreq"]):
        raise ValueError("Tail crop truncates a 1.2-second event epoch")
    raw.close()
    converted.close()
    return {
        **identity,
        "bids_run": run,
        "mapping_status": "signal_and_event_verified",
        "original_path": source["file"],
        "original_sha256": source["sha256"],
        "bdf_path": bdf.relative_to(root).as_posix(),
        "bdf_sha256": digest(bdf),
        "events_sha256": digest(events),
        "eeg_json_sha256": digest(sidecar),
        "channels_sha256": digest(channels),
        "measured_eeg_channels": mapped,
        "product_channel_observed_mask": [True] * len(PRODUCT_CHANNELS),
        "source_sample_rate": float(raw.info["sfreq"]),
        "bdf_sample_rate": 256.0,
        "source_samples": int(raw.n_times),
        "bdf_samples": int(n),
        "tail_samples_removed": int(raw.n_times - n),
        "original_prefilters": sorted({h["prefilter"] for h in eeg_headers}),
        "original_transducers": sorted({h["transducer"] for h in eeg_headers}),
        "derivative_reference": metadata["EEGReference"],
        "derivative_ground": metadata["EEGGround"],
        **waveform,
        **event_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Audit output must be outside the read-only raw dataset")
    provenance_path = root / "sourcedata" / "sourcedata_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    upstream = {}
    for line in args.upstream_manifest.read_text().splitlines():
        checksum, relative = line.split(maxsplit=1)
        prefix = "bigP3BCI-data/StudyS2/"
        if relative.startswith(prefix) and relative.endswith(".edf"):
            upstream[relative.removeprefix(prefix)] = checksum
    if {item["file"]: item["sha256"] for item in provenance["files"]} != upstream:
        raise ValueError("Local provenance does not match complete official S2 manifest")
    groups: dict[int, list] = {}
    for item in provenance["files"]:
        groups.setdefault(parse_original_identity(item["file"])["subject"], []).append(item)
    if len(groups) != 24 or any(len(group) != 12 for group in groups.values()):
        raise ValueError("Expected 24 participants with 12 original files each")
    tasks = [
        (item, run)
        for _, group in sorted(groups.items())
        for run, item in enumerate(sorted(group, key=lambda x: x["file"]))
    ]
    if len(list(root.rglob("*.bdf"))) != len(tasks):
        raise ValueError("BDF inventory is not one-to-one with original ledger")
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for record in pool.map(lambda t: audit_file(root, *t), tasks):
            records.append(record)
            if len(records) % 24 == 0:
                print(f"verified {len(records)}/{len(tasks)}", flush=True)
    role_counts = Counter((r["subject"], r["role"], r["condition"]) for r in records)
    if len(role_counts) != 96 or set(role_counts.values()) != {3}:
        raise ValueError("Unexpected per-participant role/condition inventory")
    report = {
        "schema": "mainsah-s2-source-derivative-audit/1",
        "status": "passed",
        "scope": "metadata_signal_event_integrity_only_no_model_fit_or_score",
        "provenance_sha256": digest(provenance_path),
        "script_sha256": digest(Path(__file__)),
        "upstream_manifest_sha256": digest(args.upstream_manifest),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "participant_count": len(groups),
        "record_count": len(records),
        "total_events": sum(r["event_count"] for r in records),
        "max_absolute_difference_v": max(r["max_absolute_difference_v"] for r in records),
        "max_error_in_bdf_quanta": max(r["max_error_in_bdf_quanta"] for r in records),
        "reference_authority": "Use separately reviewed acquisition reference evidence; never infer from n/a",
        "chronology": "Train before Test within condition; cross-condition order not recoverable from anonymized timestamps",
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
