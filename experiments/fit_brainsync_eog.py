"""Fit EOG regression on explicitly supplied known calibration sessions only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.brainsync import load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.continuous_qc import (  # noqa: E402
    FittedContinuousQC,
    load_continuous_qc,
    validate_continuous_calibration_budget,
)
from data.dataset import read_raw  # noqa: E402
from data.epochs import PreprocessingSpec, materialize_dataset_identity  # noqa: E402
from data.ocular import (  # noqa: E402
    EOG_REFERENCE_POLICIES,
    SHARED_MEASURED_REFERENCE,
    FittedEOGRegression,
    fit_eog_regression,
    save_eog_regression,
)
from research.contracts import semantic_sha256  # noqa: E402


def fit_brainsync_eog_sessions(
    session_dirs: list[Path] | tuple[Path, ...], *, reference_channel: str | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    eog_reference_policy: str = SHARED_MEASURED_REFERENCE,
    fit_intervals_by_session: dict[str, list[list[int]]] | None = None,
    correction_strength: float = 1.0,
    preprocessing: PreprocessingSpec | None = None,
) -> FittedEOGRegression:
    """Fit the reusable ocular model from explicitly selected labelled sessions."""
    roots = tuple(Path(path).resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("unique calibration session paths are required")
    raws, sessions, participants, references, records, masks, derivations = [], [], set(), set(), [], [], []
    for root in roots:
        session = validate_brainsync_bids_session(root)
        dataset = load_brainsync_session(root, **({"preprocessing": preprocessing} if preprocessing is not None else {}))
        participants.add(
            materialize_dataset_identity(dataset)
            .record_for(session.subject_id)
            .authority_key("source_or_global")
        )
        references.add(session.bids.source_reference)
        raw = read_raw(session.bids.raw_path, preload=True)
        raw.set_channel_types(
            dict(
                zip(
                    raw.ch_names, (kind.lower() for kind in session.bids.channel_types), strict=True
                )
            )
        )
        raw.info["bads"] = [
            name
            for name, status in zip(raw.ch_names, session.bids.channel_statuses, strict=True)
            if status == "bad"
        ]
        mask = np.ones(raw.n_times, dtype=bool)
        authorized_intervals = None
        if fit_intervals_by_session is not None:
            authorized_intervals = fit_intervals_by_session.get(session.session_id)
            if not authorized_intervals:
                raise ValueError("ocular fit requires explicit intervals for each calibration session")
            mask[:] = False
            for start, stop in authorized_intervals:
                if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= raw.n_times:
                    raise ValueError("ocular fit intervals must be half-open native sample bounds")
                mask[start:stop] = True
        clock = session.manifest["recording"].get("clock")
        if clock is not None:
            acquired = clock["acquired_samples"]
            mask[acquired:] = False
            if acquired < raw.n_times:
                raw.annotations.append(acquired / raw.info["sfreq"],
                    (raw.n_times-acquired) / raw.info["sfreq"], "BAD_acq_skip_edf_padding")
        for interval in session.bids.excluded_intervals:
            start = max(0, int(np.floor(interval.onset_seconds * raw.info["sfreq"])))
            end = min(raw.n_times, int(np.ceil((interval.onset_seconds + interval.duration_seconds) * raw.info["sfreq"])))
            mask[start:end] = False
        if session.bids.acquisition_boundaries:
            raw.annotations.append(session.bids.acquisition_boundaries,
                [0.] * len(session.bids.acquisition_boundaries), ["EDGE boundary"] * len(session.bids.acquisition_boundaries))
        if continuous_qc_model is not None:
            mask &= continuous_qc_model.detect(raw).valid_sample_mask
        masks.append(mask)
        derivations.append(dict(session.bids.eog_derivation))
        raws.append(raw)
        sessions.append(session)
        records.append(
            {
                "session_id": session.session_id,
                "raw_sha256": hashlib.sha256(session.bids.raw_path.read_bytes()).hexdigest(),
                "session_sha256": hashlib.sha256(session.session_path.read_bytes()).hexdigest(),
                **({"authorized_fit_intervals_samples": authorized_intervals}
                   if authorized_intervals is not None else {}),
            }
        )
    if len(participants) != 1 or len(references) != 1 or any(value != derivations[0] for value in derivations):
        raise ValueError(
            "ocular calibration sessions must share one participant authority and reference."
        )
    continuous_processing = {"status": "not_applied"} if continuous_qc_model is None else {
        "status": "applied", "model": continuous_qc_model.record(),
        "model_digest": semantic_sha256(continuous_qc_model.record())}
    validate_continuous_calibration_budget(continuous_processing,
        {next(iter(participants)): tuple(session.session_id for session in sessions)})
    if continuous_qc_model is not None and continuous_qc_model.fit_provenance.get("source_reference") != next(iter(references)):
        raise ValueError("continuous QC reference differs from ocular calibration raw")
    if fit_intervals_by_session is not None and set(fit_intervals_by_session) != {s.session_id for s in sessions}:
        raise ValueError("ocular fit interval sessions differ from the supplied calibration recordings")
    model = fit_eog_regression(
        mne.concatenate_raws(raws, verbose=False),
        participant_key=next(iter(participants)),
        source_reference=next(iter(references)),
        reference_channel=reference_channel,
        eog_reference_policy=eog_reference_policy,
        fit_sample_mask=np.concatenate(masks),
        eog_derivation=derivations[0],
        fit_provenance={
            "role": "known_calibration_sessions_only",
            "sources": records,
            "calibration_session_ids": [session.session_id for session in sessions],
            "calibration_end_timestamp_s": max(session.ended_timestamp_s for session in sessions),
            "signal_scope": "explicit_valid_calibration_samples_including_retained_rest",
            "continuous_qc_digest": None if continuous_qc_model is None else semantic_sha256(continuous_qc_model.record()),
            "continuous_artifact_processing": continuous_processing,
        },
    )
    return replace(model, correction_strength=correction_strength)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-session-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-channel", default=None)
    parser.add_argument("--eog-reference-policy", choices=EOG_REFERENCE_POLICIES,
                        default=SHARED_MEASURED_REFERENCE)
    parser.add_argument("--continuous-qc-model", type=Path)
    parser.add_argument("--correction-strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a new output path is required")
    model = fit_brainsync_eog_sessions(
        args.calibration_session_dir, reference_channel=args.reference_channel,
        eog_reference_policy=args.eog_reference_policy,
        correction_strength=args.correction_strength,
        continuous_qc_model=None if args.continuous_qc_model is None else load_continuous_qc(args.continuous_qc_model),
    )
    print(f"[ocular] {save_eog_regression(args.output, model)}", flush=True)


if __name__ == "__main__":
    main()
