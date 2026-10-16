"""Fit frozen raw QC from explicitly enumerated clean calibration intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.continuous_qc import (  # noqa: E402
    ContinuousQCSpec,
    FilterStateRecovery,
    fit_continuous_qc,
    save_continuous_qc,
)
from data.dataset import read_raw  # noqa: E402
from data.epochs import PreprocessingSpec, materialize_dataset_identity  # noqa: E402
from data.recovery import fit_epoch_state_recovery  # noqa: E402


def fit_brainsync_continuous_manifest(path: str | Path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "brainsync_continuous_qc_calibration/1":
        raise ValueError("explicit continuous QC calibration manifest is required")
    preprocessing = (PreprocessingSpec(**manifest["preprocessing"])
                     if "preprocessing" in manifest else BRAIN_SYNC_PREPROCESSING)
    preprocessing.validate()
    specification = dict(manifest["spec"])
    if preprocessing.filter_phase == "zero":
        specification.setdefault("range_action", "diagnostic")
        specification.setdefault("boundary_action", "artifact_edges")
        if manifest.get("epoch_recovery") is None:
            raise ValueError("zero-phase QC fitting requires explicit epoch_recovery calibration parameters")
    spec = ContinuousQCSpec(**specification)
    raws, masks, sessions, participants, references, records = [], [], [], set(), set(), []
    for item in manifest["sessions"]:
        root = (path.parent / item["session_dir"]).resolve()
        session = validate_brainsync_bids_session(root)
        dataset = load_brainsync_session(root, preprocessing=preprocessing)
        participants.add(materialize_dataset_identity(dataset).record_for(session.subject_id).authority_key("source_or_global"))
        references.add(session.bids.source_reference)
        raw = read_raw(session.bids.raw_path, preload=True)
        raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
        raw.info["bads"] = [name for name, status in zip(raw.ch_names, session.bids.channel_statuses, strict=True) if status == "bad"]
        mask = np.zeros(raw.n_times, dtype=bool)
        for start, stop in item["clean_intervals_samples"]:
            if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= raw.n_times:
                raise ValueError("clean calibration intervals must be half-open native sample bounds")
            mask[start:stop] = True
        if not mask.any() or not item.get("clean_interval_evidence"):
            raise ValueError("clean interval selection requires samples and evidence")
        clock = session.manifest["recording"].get("clock")
        if clock is not None:
            acquired = clock["acquired_samples"]
            mask[acquired:] = False
            if acquired < raw.n_times:
                raw.annotations.append(acquired / raw.info["sfreq"],
                    (raw.n_times-acquired) / raw.info["sfreq"], "BAD_acq_skip_edf_padding")
        if session.bids.acquisition_boundaries:
            raw.annotations.append(session.bids.acquisition_boundaries,
                [0.] * len(session.bids.acquisition_boundaries), ["EDGE boundary"] * len(session.bids.acquisition_boundaries))
        for interval in session.bids.excluded_intervals:
            start = int(np.floor(interval.onset_seconds * raw.info["sfreq"]))
            stop = int(np.ceil((interval.onset_seconds + interval.duration_seconds) * raw.info["sfreq"]))
            mask[max(0, start):min(raw.n_times, stop)] = False
        raws.append(raw)
        masks.append(mask)
        sessions.append(session)
        records.append({"session_id": session.session_id,
                        "raw_sha256": hashlib.sha256(session.bids.raw_path.read_bytes()).hexdigest(),
                        "clean_intervals_samples": item["clean_intervals_samples"],
                        "clean_interval_evidence": item["clean_interval_evidence"]})
    if (len(participants) != 1 or len(references) != 1
            or len({session.session_id for session in sessions}) != len(sessions)):
        raise ValueError("continuous QC requires unique sessions from one participant and reference")
    recovery = manifest.get("recovery")
    provenance = {"participant_key": next(iter(participants)),
                        "source_reference": next(iter(references)), "sources": records,
                        "calibration_session_ids": [session.session_id for session in sessions],
                        "calibration_end_timestamp_s": max(session.ended_timestamp_s for session in sessions),
                        "role": "explicit_clean_calibration_intervals",
                        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    raw = mne.concatenate_raws(raws, verbose=False)
    mask = np.concatenate(masks)
    epoch_recovery = None
    if manifest.get("epoch_recovery") is not None:
        channels = tuple(raw.ch_names[index] for index in mne.pick_types(raw.info,
            eeg=True, eog=spec.include_eog, exclude="bads"))
        epoch_recovery = fit_epoch_state_recovery(raw, preprocessing=preprocessing,
            channels=channels, fit_sample_mask=mask, fit_provenance=provenance,
            **manifest["epoch_recovery"])
    return fit_continuous_qc(raw, spec=spec, fit_sample_mask=mask,
        recovery=None if recovery is None else FilterStateRecovery(**recovery),
        epoch_recovery=epoch_recovery, fit_provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = fit_brainsync_continuous_manifest(args.calibration_manifest)
    print(save_continuous_qc(args.output, model))


if __name__ == "__main__":
    main()
