"""Fit EOG regression on explicitly supplied known calibration sessions only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.brainsync import load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.dataset import read_raw  # noqa: E402
from data.epochs import materialize_dataset_identity  # noqa: E402
from data.ocular import FittedEOGRegression, fit_eog_regression, save_eog_regression  # noqa: E402


def fit_brainsync_eog_sessions(
    session_dirs: list[Path] | tuple[Path, ...], *, reference_channel: str | None = None
) -> FittedEOGRegression:
    """Fit the reusable ocular model from explicitly selected labelled sessions."""
    roots = tuple(Path(path).resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("unique calibration session paths are required")
    raws, sessions, participants, references, records = [], [], set(), set(), []
    for root in roots:
        session = validate_brainsync_bids_session(root)
        dataset = load_brainsync_session(root)
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
        raws.append(raw)
        sessions.append(session)
        records.append(
            {
                "session_id": session.session_id,
                "raw_sha256": hashlib.sha256(session.bids.raw_path.read_bytes()).hexdigest(),
                "session_sha256": hashlib.sha256(session.session_path.read_bytes()).hexdigest(),
            }
        )
    if len(participants) != 1 or len(references) != 1:
        raise ValueError(
            "ocular calibration sessions must share one participant authority and reference."
        )
    return fit_eog_regression(
        mne.concatenate_raws(raws, verbose=False),
        participant_key=next(iter(participants)),
        source_reference=next(iter(references)),
        reference_channel=reference_channel,
        fit_provenance={
            "role": "known_calibration_sessions_only",
            "sources": records,
            "calibration_session_ids": [session.session_id for session in sessions],
            "calibration_end_timestamp_s": max(session.ended_timestamp_s for session in sessions),
            "signal_scope": "native_raw_including_calibration_rest_no_test_data",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-session-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-channel", default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a new output path is required")
    model = fit_brainsync_eog_sessions(
        args.calibration_session_dir, reference_channel=args.reference_channel
    )
    print(f"[ocular] {save_eog_regression(args.output, model)}", flush=True)


if __name__ == "__main__":
    main()
