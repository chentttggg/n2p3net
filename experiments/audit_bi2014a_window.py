"""Measure BI2014a event overlap from raw ZIPs without fitting a model.

Counts are observed onsets in the half-open response window, not ERP energy
fractions. CSV columns are zero-based: 17 is flash code, 18 is raw label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.bi2014a_candidate import BI2014A_SFREQ  # noqa: E402
from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE  # noqa: E402
from data.exclusions import select_allowed_subjects  # noqa: E402


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
            "p90": float(np.quantile(values, .9))}


def following_onsets(samples: np.ndarray, *, sfreq: float, tmax_ms: float) -> np.ndarray:
    """Count each actual subsequent onset in (onset, onset + tmax)."""
    samples = np.asarray(samples)
    if (samples.ndim != 1 or not np.issubdtype(samples.dtype, np.integer)
            or np.any(samples < 0) or np.any(np.diff(samples) <= 0)):
        raise ValueError("onsets must be increasing nonnegative integer samples")
    if not np.isfinite([sfreq, tmax_ms]).all() or sfreq <= 0 or tmax_ms <= 0:
        raise ValueError("sfreq and tmax_ms must be positive finite values")
    return (np.searchsorted(samples, samples + tmax_ms * sfreq / 1000, side="left")
            - np.arange(len(samples)) - 1)


def audit_zip(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(path) as archive:
        member = path.stem + ".csv"
        with archive.open(member) as stream:
            table = pd.read_csv(stream, header=None, usecols=[17, 18]).to_numpy()
    samples = np.flatnonzero((table[:, 0] >= 20) & (table[:, 0] <= 85))
    codes = table[samples, 0]
    labels = table[samples, 1]
    audit = BI2014A_FLASH_SCHEDULE.audit_raw_labels(
        codes, labels, flash_samples=samples, stage="raw_zip_window_audit")
    width = 2 * BI2014A_FLASH_SCHEDULE.grid_size
    complete = len(samples) // width * width
    if complete == 0:
        raise ValueError(f"{path}: no complete repetition")
    # Validate every complete repetition before interpreting its boundary.
    for block in codes[:complete].reshape(-1, width):
        events = [BI2014A_FLASH_SCHEDULE.decode(code) for code in block]
        for axis in ("row", "column"):
            selected = [event for event in events if event.axis == axis]
            if (sorted(event.candidate_index for event in selected)
                    != list(range(BI2014A_FLASH_SCHEDULE.grid_size))
                    or sum(event.is_target for event in selected) != 1):
                raise ValueError(f"{path}: invalid complete repetition")
    deltas = np.diff(samples[:complete]) * 1000 / BI2014A_SFREQ
    within = np.arange(len(deltas)) % width != width - 1
    prefix = samples[samples < 400000]
    prefix_deltas = np.diff(prefix) * 1000 / BI2014A_SFREQ
    prefix_within = np.arange(len(prefix_deltas)) % width != width - 1
    return {
        "subject": path.stem, "zip_sha256": digest, "csv_member": member,
        "raw_samples": len(table), "sfreq": BI2014A_SFREQ,
        "flash_column_zero_based": 17, "label_column_zero_based": 18,
        "n_flashes": len(samples), "n_complete_repetitions": complete // width,
        "incomplete_tail_flashes": len(samples) - complete,
        "label_mismatches": audit.n_mismatches,
        "within_repetition_ms": distribution(deltas[within]),
        "between_repetition_ms": distribution(deltas[~within]),
        "following_onsets": {
            str(end): distribution(following_onsets(samples, sfreq=BI2014A_SFREQ, tmax_ms=end))
            for end in (600, 800, 1200)
        },
        "first_400000_samples": {
            "n_flashes": len(prefix),
            "within_repetition_ms": distribution(prefix_deltas[prefix_within]),
            "between_repetition_ms": distribution(prefix_deltas[~prefix_within]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "mne_data" /
                        "MNE-braininvaders2014a-data" / "zenodo" / "3266223")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 62)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    subjects = select_allowed_subjects("BI2014a", args.subjects, explicit=True)
    if len(set(subjects)) != len(subjects):
        parser.error("subjects must be unique")
    paths = [args.root / f"subject_{subject:02d}.zip" for subject in subjects]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(audit_zip, paths))
    sources = [Path(__file__), ROOT / "src/data/bi2014a_schedule.py",
               ROOT / "src/data/bi2014a_candidate.py", ROOT / "src/data/exclusions.py"]
    result = {
        "schema": "bi2014a_raw_window_audit/1", "model_fitting": False,
        "interpretation": "onset overlap only; not ERP energy, noise fraction, or causal harm",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "n_subjects": len(records), "subjects": records,
        "participant_distribution_of_within_rep_medians_ms": distribution(np.array([
            record["within_repetition_ms"]["median"] for record in records])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_subjects": len(records),
                      "within_rep_medians": result["participant_distribution_of_within_rep_medians_ms"]}))


if __name__ == "__main__":
    main()
