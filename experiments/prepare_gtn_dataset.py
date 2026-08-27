"""Prepare the GTN NIX corpus as a current 9-choice EpochDataset cache."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.epochs import save_epoch_dataset  # noqa: E402
from data.gtn_dataset import GTN_LMBC_PREPROCESSING, build_gtn_epoch_dataset  # noqa: E402


def _derive_n_times(sfreq: float, tmin_ms: float, tmax_ms: float) -> int:
    return math.floor((tmax_ms - tmin_ms) * sfreq / 1000.0 + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing Experiment_*_P3_Numbers.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sfreq", type=float, default=GTN_LMBC_PREPROCESSING.sfreq)
    parser.add_argument("--l-freq", type=float, default=GTN_LMBC_PREPROCESSING.l_freq)
    parser.add_argument("--tmin-ms", type=float, default=GTN_LMBC_PREPROCESSING.tmin_ms)
    parser.add_argument("--tmax-ms", type=float, default=GTN_LMBC_PREPROCESSING.tmax_ms)
    parser.add_argument("--n-times", type=int, default=None)
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Record invalid or duplicate public GTN sources instead of failing the cohort build.",
    )
    args = parser.parse_args()
    n_times = args.n_times or _derive_n_times(args.sfreq, args.tmin_ms, args.tmax_ms)
    preprocessing = replace(
        GTN_LMBC_PREPROCESSING,
        sfreq=args.sfreq,
        l_freq=args.l_freq,
        tmin_ms=args.tmin_ms,
        tmax_ms=args.tmax_ms,
        n_times=n_times,
    )
    dataset = build_gtn_epoch_dataset(
        args.root,
        preprocessing=preprocessing,
        allow_skipped=args.allow_skipped,
    )
    output = save_epoch_dataset(args.output, dataset)
    record_path = output.with_suffix(".record.json")
    print(
        f"[prepared] {output} X={dataset.X.shape} subjects={len(set(dataset.subject_ids))} "
        f"candidate_chain={dataset.event_timeline.supports_full_candidate_chain}",
        flush=True,
    )
    print(f"[record] {record_path}", flush=True)


if __name__ == "__main__":
    main()
