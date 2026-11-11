"""Build and replay frozen, content-bound QC for the reviewed BrainSync batch."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import mne
import numpy as np
from analyze_brainsync_artifact_remedies import epoch_values, filter_values
from threadpoolctl import threadpool_limits

from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    IndependentRecordingScope,
    load_brainsync_inference_session,
)
from data.brainsync_contract import validate_brainsync_bids_recording
from data.continuous_qc import (
    ContinuousQCSpec,
    fit_continuous_qc,
    load_continuous_qc,
    save_continuous_qc,
)
from data.epochs import load_epoch_dataset, materialize_dataset_identity, save_epoch_dataset
from data.recording_review import RecordingExcludedError
from data.recovery import fit_epoch_state_recovery
from data.reviewed_intervals import ReviewedInterval, ReviewedRecording
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from research.recording_quality import create_recording_quality_report, file_sha256

REPO = Path(__file__).resolve().parents[1]
REFERENCE_ID = "P001_20261028_224755_01"
EXCLUDED_ID = "P001_20261030_144501_01"
BAD_FZ_IDS = {"P001_20261028_221030_01", "P001_20261028_222559_01"}
FIT_INTERVALS_S = ((120., 200.), (570., 625.))
VALIDATION_INTERVALS_S = ((240., 300.), (650., 700.))


def read_native(root):
    validated = validate_brainsync_bids_recording(root)
    try:
        raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose=False)
        raw.set_channel_types(dict(zip(raw.ch_names, [t.lower() for t in validated.bids.channel_types], strict=True)))
        boundaries = list(validated.bids.acquisition_boundaries)
        if boundaries:
            raw.set_annotations(raw.annotations + mne.Annotations(boundaries, [0.] * len(boundaries),
                ["EDGE boundary"] * len(boundaries), orig_time=raw.annotations.orig_time))
        return raw
    finally:
        validated.bids.close()


def reviewed_plan(entry, numerical, numerical_directory):
    root = Path(entry["input_root"])
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    raw_names = [name for name, kind in zip(manifest["channels"], manifest["channel_types"], strict=True) if kind == "eeg"]
    canonical = {name.upper(): name for name in raw_names}
    intervals = []
    rate = float(manifest["recording"]["source_sample_rate_hz"])
    samples = int(manifest["recording"]["clock"]["edf_samples"])
    for row in numerical["candidate_intervals"]:
        # Reviewed batch curation, not a generic 500uV production threshold.
        a = max(row["segment_start_sample"], row["start_sample"] - round(.2 * rate))
        b = min(row["segment_stop_sample"], row["stop_sample"] + round(.2 * rate))
        intervals.append(ReviewedInterval(a, b, (canonical[row["channel_name"]],), "reviewed_transient"))
    with np.load(numerical_directory / f"{entry['directory_id']}_arrays.npz") as archive:
        ptp, onsets = archive["epoch_ptp_uv"], np.rint(archive["onsets_s"] * rate).astype(int)
        names = archive["channel_names"].tolist()
    for trial, channel in np.argwhere(ptp > 500):
        a = max(0, int(onsets[trial]) - round(.2 * rate))
        b = min(samples, int(onsets[trial]) + round(1.2 * rate) + 1)
        intervals.append(ReviewedInterval(a, b, (canonical[names[channel]],), "reviewed_gross_spike_epoch"))
    if entry["directory_id"] in BAD_FZ_IDS:
        intervals.append(ReviewedInterval(0, samples, (canonical["FZ"],), "reviewed_persistent_unstable_channel"))
    # Keep per-channel review reasons. Intersections are merged by the filter-plan owner.
    return ReviewedRecording(entry["report"]["input_sha256"]["raw"], samples, entry["directory_id"],
        "2026-10-30 signal-quality implementation: reviewed candidate transients, gross spike epochs and persistent Fz; no target/accuracy threshold selection",
        tuple(intervals))


def validate_reference_boundaries(raw, recovery):
    fs = float(raw.info["sfreq"])
    channels = list(recovery.channel_names)
    values = raw.get_data(picks=channels)
    spec = BRAIN_SYNC_PREPROCESSING
    trials, failures, maximum = 0, 0, 0.
    for begin_s, end_s in VALIDATION_INTERVALS_S:
        a, b = round(begin_s * fs), round(end_s * fs)
        span = b - a
        reference = filter_values(values[:, a:b], fs, [(0, span)], spec)
        for left_s, right_s in ((5., 35.), (15., 45.), (5., 45.)):
            if right_s > span / fs - 5:
                continue
            left, right = round(left_s * fs), round(right_s * fs)
            local = filter_values(values[:, a+left:a+right], fs, [(0, right-left)], spec)
            onsets = np.arange(left + round(1 * fs), right - round(1.5 * fs), round(1.5 * fs))
            if not len(onsets):
                continue
            expected = epoch_values(reference, fs, onsets, spec)
            actual = epoch_values(local, fs, onsets-left, spec)
            starts = onsets - left + round(spec.tmin_ms / 1000 * fs)
            bound = recovery.epoch_error_bounds(right-left, starts)
            usable = bound <= recovery.tolerance_v
            error = np.max(np.abs(actual - expected), axis=-1)
            trials += int(usable.sum())
            failures += int(((error > recovery.tolerance_v + 1e-12) & usable).sum())
            if usable.any():
                maximum = max(maximum, float(error[usable].max()))
    return {"reference_intervals_s": VALIDATION_INTERVALS_S, "admitted_channel_epochs": trials,
            "admitted_errors_exceeding_tolerance": failures, "max_admitted_error_v": maximum,
            "condition": "Compared actual shorter/longer finite contexts in held reference intervals; no clean-brain ground truth."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--numerical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch_path, numerical_path = args.batch_manifest.resolve(), args.numerical_manifest.resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    numerical = json.loads(numerical_path.read_text(encoding="utf-8"))
    entries = [r for r in batch["sessions"] if r["group"].startswith("Brain") and r.get("extra")]
    usable_entries = [r for r in entries if r["directory_id"] != EXCLUDED_ID]
    reference_entry = next(r for r in entries if r["directory_id"] == REFERENCE_ID)
    reference_root = Path(reference_entry["input_root"])
    sources = [Path(__file__), *(REPO / "src/data").glob("*.py"), REPO / "src/data/recording_reviews.json",
               REPO / "src/research/recording_quality.py", REPO / "src/transfer/quality.py"]
    source_sha = {p.relative_to(REPO).as_posix(): file_sha256(p) for p in sources}
    execution = {"schema": "brainsync_reviewed_quality_execution/1", "status": "running",
                 "started_utc": datetime.now(UTC).isoformat(), "role": "retrospective_single_person_development",
                 "parent_sha256": {str(p): file_sha256(p) for p in (batch_path, numerical_path)},
                 "source_sha256": source_sha, "qc_classifier_fit": False,
                 "frozen_boundary_tolerance_v": 1e-6, "modal_box_inflation": 2.,
                 "boundary_fit_intervals_s": FIT_INTERVALS_S, "reference_session": REFERENCE_ID,
                 "whole_record_excluded": EXCLUDED_ID, "minimum_native_channels": 5,
                 "automatic_amplitude_rejection": False,
                 "reviewed_only": True,
                 "remaining_scope": "Known reviewed EDF bytes only. New recordings require a new review/QC validation; this is not a generic future detector."}
    atomic_write_json(output / "execution.json", execution)
    with threadpool_limits(limits=2):
        dref = load_brainsync_inference_session(reference_root)
        participant = materialize_dataset_identity(dref).record_for("P001").authority_key("source_or_global")
        validated = validate_brainsync_bids_recording(reference_root)
        ended = validated.ended_timestamp_s
        validated.bids.close()
        raw = read_native(reference_root)
        try:
            fs = float(raw.info["sfreq"])
            names = tuple(raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, eog=False))
            mask = np.zeros(raw.n_times, dtype=bool)
            for a, b in FIT_INTERVALS_S:
                mask[round(a * fs):round(b * fs)] = True
            provenance = {"participant_key": participant, "source_reference": "left earlobe",
                "calibration_session_ids": [REFERENCE_ID], "calibration_end_timestamp_s": ended,
                "reference_raw_sha256": reference_entry["report"]["input_sha256"]["raw"],
                "reference_intervals_s": FIT_INTERVALS_S,
                "role": "reviewed_stable_development_reference_not_clean_brain_certification"}
            print("FIT_BOUNDARY_REFERENCE", REFERENCE_ID, FIT_INTERVALS_S, flush=True)
            recovery = fit_epoch_state_recovery(raw, preprocessing=BRAIN_SYNC_PREPROCESSING, channels=names,
                fit_sample_mask=mask, tolerance_v=1e-6, inflation=2., warmup_samples=round(5 * fs),
                probe_step_samples=round(5 * fs), fit_provenance=provenance)
            validation = validate_reference_boundaries(raw, recovery)
            atomic_write_json(output / "boundary_validation.json", validation)
            if validation["admitted_channel_epochs"] == 0 or validation["admitted_errors_exceeding_tolerance"]:
                execution.update(status="failed_boundary_validation", boundary_validation=validation)
                atomic_write_json(output / "execution.json", execution)
                raise ValueError("frozen boundary model failed independent finite-context validation")
            qc = fit_continuous_qc(raw, spec=ContinuousQCSpec(12., 1e-7, 25, channel_local=True,
                minimum_channel_fraction=5/7, flatline_min_duration_s=5., range_action="diagnostic",
                boundary_action="artifact_edges"), fit_sample_mask=mask, fit_provenance=provenance, epoch_recovery=recovery)
        finally:
            raw.close()
        declarations = tuple(reviewed_plan(entry, next(r for r in numerical["sessions"] if r["session"] == entry["directory_id"]),
                                          numerical_path.parent) for entry in usable_entries)
        qc = replace(qc, reviewed_recordings=declarations, reviewed_only=True)
        profile_path = output / "continuous_reviewed_native7.json"
        save_continuous_qc(profile_path, qc)
        restored = load_continuous_qc(profile_path)
        assert semantic_sha256(restored.record()) == semantic_sha256(qc.record())
        scope = IndependentRecordingScope((REFERENCE_ID,), tuple(e["directory_id"] for e in usable_entries if e["directory_id"] != REFERENCE_ID))
        execution.update(frozen_profile_sha256=file_sha256(profile_path), scope=scope.record(), boundary_validation=validation,
                         reviewed_recordings=[r.record() for r in declarations], sessions=[])
        atomic_write_json(output / "execution.json", execution)
        for entry in entries:
            session = entry["directory_id"]
            root = Path(entry["input_root"])
            folder = output / session
            state = {}

            def loader(path, state=state):
                state["dataset"] = load_brainsync_inference_session(path, continuous_qc_model=restored, recording_scope=scope)
                return state["dataset"]

            print("REPLAY", session, flush=True)
            report = create_recording_quality_report(root, folder, loader=loader,
                request={"continuous_qc_model": str(profile_path), "recording_scope": scope.record(), "role": execution["role"]})
            row = {"session": session, "report_status": report["status"], "counts": report.get("counts"),
                   "report": str(folder / "report.json")}
            if "dataset" in state:
                dataset = state["dataset"]
                path = save_epoch_dataset(folder / "epochs.npz", dataset)
                reloaded = load_epoch_dataset(path)
                assert np.array_equal(dataset.X, reloaded.X)
                assert np.array_equal(dataset.trial_channel_mask, reloaded.trial_channel_mask)
                assert len(dataset.event_timeline.evidence_indices) == 450
                if session in BAD_FZ_IDS:
                    assert not dataset.trial_channel_mask[:, dataset.channel_names.index("FZ")].any()
                assert np.all(dataset.trial_channel_mask.sum(axis=1) >= 5)
                row.update(cache=str(path), retained_epochs=dataset.n_epochs,
                    complete_seven_channel_epochs=int(dataset.trial_channel_mask.all(axis=1).sum()),
                    observed_epochs_by_channel=dict(zip(dataset.channel_names, dataset.trial_channel_mask.sum(axis=0).tolist(), strict=True)),
                    cache_roundtrip_exact=True)
            else:
                row["retained_epochs"] = 0
            if session == EXCLUDED_ID:
                assert report["status"] == "excluded" and report["counts"]["scheduled_trials"] == 450
                try:
                    validate_brainsync_bids_recording(root)
                except RecordingExcludedError:
                    row["model_ingress_rejection_verified"] = True
                else:
                    raise AssertionError("excluded record passed model ingress")
            assert all(file_sha256(root / p) == digest for p, digest in entry["input_sha256"].items())
            execution["sessions"].append(row)
            atomic_write_json(output / "execution.json", execution)
            print("RESULT", session, row["report_status"], row["retained_epochs"], flush=True)
        assert all(file_sha256(REPO / p) == digest for p, digest in source_sha.items())
        execution.update(status="completed", finished_utc=datetime.now(UTC).isoformat(),
                         input_files_unchanged=True, model_digest=semantic_sha256(qc.record()),
                         requested_flashes_including_excluded=3600,
                         retained_epochs_total=sum(r["retained_epochs"] for r in execution["sessions"]))
        atomic_write_json(output / "execution.json", execution)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
