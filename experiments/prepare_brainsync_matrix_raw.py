"""Export actual original-axis block scopes for unfiltered continuous experts."""

# ruff: noqa: E402
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import mne
import numpy as np

from data.brainsync_contract import validate_brainsync_bids_session
from data.brainsync_startup import startup_cut_for
from data.continuous_qc import load_continuous_qc
from research.execution import atomic_write_json

RUN = ROOT / "experiments/runs/brainsync_matrix_20261031"
PROFILE = (
    ROOT
    / "experiments/runs/brainsync_quality_20261030/implementation/accepted/continuous_reviewed_native7.json"
)
NAMES = ("P3", "PZ", "P4", "PO8", "OZ")


def main():
    plan = json.loads((RUN / "plan.json").read_bytes())
    profile = load_continuous_qc(PROFILE).record()
    reviewed = {r["raw_sha256"]: r for r in profile["reviewed_recordings"]}
    for entry in plan["rows"]:
        if entry["subject_id"] != "P001":
            continue
        validated = validate_brainsync_bids_session(entry["original_root"])
        try:
            startup_cut = startup_cut_for(validated.bids)
            path = validated.bids.raw_path
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            with mne.io.read_raw_edf(path, preload=True, verbose=False) as raw:
                fs = float(raw.info["sfreq"])
                names = [n.upper() for n in raw.ch_names]
                indices = [names.index(n) for n in NAMES]
                cz = names.index("CZ")
                values = raw.get_data(picks=indices) - raw.get_data(picks=[cz])
                clock = validated.manifest["recording"].get("clock")
                if clock is not None:
                    segments = clock["segments"]
                    segment_authority = "original_saved_sample_clock"
                else:
                    # Legacy records expose their actual BIDS acquisition
                    # boundaries, without claiming a saved modern clock.
                    edges = (
                        [0]
                        + [round(t * fs) for t in validated.bids.acquisition_boundaries]
                        + [raw.n_times]
                    )
                    segments = [
                        {"stored_start": a, "stored_stop": b}
                        for a, b in zip(edges[:-1], edges[1:], strict=True)
                    ]
                    segment_authority = "legacy_BIDS_acquisition_boundaries_no_modern_clock_claim"
                rows, arrays = [], {}
                for block in range(1, 11):
                    events = [s for s in validated.bids.stimuli if s.block_id == block]
                    assert len(events) == 45
                    left = events[0].sample - round(0.2 * fs)
                    right = events[-1].sample + round(1.2 * fs) + 1
                    legal = (
                        left >= 0
                        and (startup_cut is None or left >= startup_cut["stop_sample"])
                        and right <= raw.n_times
                        and any(
                            s["stored_start"] <= left and right <= s["stored_stop"]
                            for s in segments
                        )
                    )
                    conflicts = []
                    for interval in reviewed.get(content, {}).get("intervals", []):
                        if (
                            set(n.upper() for n in interval["channels"]) & (set(NAMES) | {"CZ"})
                            and interval["start_sample"] < right
                            and interval["stop_sample"] > left
                        ):
                            conflicts.append(interval)
                    onsets = np.array([s.sample - left for s in events], dtype=np.int64)
                    if legal:
                        arrays[f"X_{block}"] = np.asarray(values[:, left:right].T, dtype=np.float64)
                        arrays[f"onsets_{block}"] = onsets
                        arrays[f"labels_{block}"] = np.array(
                            [int(s.candidate_id) == entry["target"] for s in events]
                        )
                        arrays[f"membership_{block}"] = np.eye(9, dtype=bool)[
                            [int(s.candidate_id) - 1 for s in events]
                        ]
                    rows.append(
                        {
                            "block": block,
                            "raw_scope": [left, right],
                            "legal": legal,
                            "reviewed_clean": legal and not conflicts,
                            "conflicts": conflicts,
                            "event_ids": [s.trial_id for s in events],
                            "source_edf_sha256": content,
                            "raw_startup_cut": startup_cut,
                        }
                    )
                output = RUN / "inputs" / entry["session_id"] / "raw_blocks.npz"
                if output.exists() or output.with_suffix(".json").exists():
                    raise FileExistsError("raw block exports require a new run; existing scientific results are immutable")
                np.savez(output, **arrays)
                receipt = {
                    "source_sha256": content,
                    "sample_rate_hz": fs,
                    "channels": list(NAMES),
                    "reference": "measured_channel:CZ",
                    "signal_unit": "V",
                    "filter": "none",
                    "sampling": "native original axis; no resampling or concatenation",
                    "segment_authority": segment_authority,
                    "raw_startup_cut": startup_cut,
                    "rows": rows,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
                atomic_write_json(output.with_suffix(".json"), receipt)
                print(
                    entry["session_id"],
                    "raw legal",
                    sum(r["legal"] for r in rows),
                    "reviewed clean",
                    sum(r["reviewed_clean"] for r in rows),
                    flush=True,
                )
        finally:
            validated.bids.close()


if __name__ == "__main__":
    main()
