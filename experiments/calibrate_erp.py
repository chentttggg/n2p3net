"""Calibrate ERP component priors from any universal EpochDataset cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from models.erp_calibration import calibrate_erp_fold  # noqa: E402
from models.time_axis import EpochTimeAxis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration-scope",
        choices=("independent_development", "diagnostic_only"),
        default="diagnostic_only",
    )
    args = parser.parse_args()

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True)
    if args.subjects is not None:
        subjects = np.unique(dataset.subject_ids)[: args.subjects]
        keep = np.isin(dataset.subject_ids, subjects)
        X, y, subject_ids = dataset.X[keep], dataset.y[keep], dataset.subject_ids[keep]
    else:
        X, y, subject_ids = dataset.X, dataset.y, dataset.subject_ids
    profile = dataset.preprocessing
    calibration = calibrate_erp_fold(
        X,
        y,
        subject_ids,
        time_axis=EpochTimeAxis(
            profile.tmin_ms,
            profile.tmax_ms,
            profile.sfreq,
            profile.n_times,
        ),
        channel_names=dataset.channel_names,
    )
    calibration["calibration_scope"] = args.calibration_scope
    calibration["dataset"] = dataset.record()
    calibration["evidence_units"] = "volts"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[erp-calibration] {dataset.name}: subjects={calibration['n_subjects']} "
        f"tau0_ms={[round(value, 1) for value in calibration['tau0_ms']]} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
