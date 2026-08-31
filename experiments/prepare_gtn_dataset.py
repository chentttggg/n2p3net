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

from data.contract import CAUSAL_IIR_INITIAL_STATE  # noqa: E402
from data.epochs import save_epoch_dataset  # noqa: E402
from data.gtn_dataset import (  # noqa: E402
    GTN_COHORT_CONTRACTS,
    build_gtn_epoch_dataset,
    gtn_preprocessing_for_cohort,
)


def _derive_n_times(sfreq: float, tmin_ms: float, tmax_ms: float) -> int:
    return math.floor((tmax_ms - tmin_ms) * sfreq / 1000.0 + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing Experiment_*_P3_Numbers.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cohort",
        choices=tuple(GTN_COHORT_CONTRACTS),
        default="default",
        help=(
            "Contract family to bake into the cache: default/gtn are 0.1 Hz "
            "zero/forward; gtn_paper_offline/gtn_paper are 0.5 Hz zero/forward. "
            "Chronological paths require a forward profile."
        ),
    )
    parser.add_argument("--sfreq", type=float, default=None)
    parser.add_argument("--l-freq", type=float, default=None)
    parser.add_argument("--tmin-ms", type=float, default=None)
    parser.add_argument("--tmax-ms", type=float, default=None)
    parser.add_argument("--n-times", type=int, default=None)
    parser.add_argument(
        "--filter-phase",
        choices=("zero", "forward"),
        default=None,
        help=(
            "forward is required for chronological single-subject prefix/suffix "
            "folds; defaults to the selected --cohort contract."
        ),
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Record invalid or duplicate public GTN sources instead of failing the cohort build.",
    )
    args = parser.parse_args()
    base = gtn_preprocessing_for_cohort(args.cohort)
    sfreq = base.sfreq if args.sfreq is None else args.sfreq
    l_freq = base.l_freq if args.l_freq is None else args.l_freq
    tmin_ms = base.tmin_ms if args.tmin_ms is None else args.tmin_ms
    tmax_ms = base.tmax_ms if args.tmax_ms is None else args.tmax_ms
    filter_phase = base.filter_phase if args.filter_phase is None else args.filter_phase
    causal_iir_initial_state = (
        CAUSAL_IIR_INITIAL_STATE if filter_phase == "forward" else "not_applicable"
    )
    n_times = args.n_times or _derive_n_times(sfreq, tmin_ms, tmax_ms)
    preprocessing = replace(
        base,
        sfreq=sfreq,
        l_freq=l_freq,
        tmin_ms=tmin_ms,
        tmax_ms=tmax_ms,
        n_times=n_times,
        filter_phase=filter_phase,
        causal_iir_initial_state=causal_iir_initial_state,
    )
    preprocessing.validate()
    dataset = build_gtn_epoch_dataset(
        args.root,
        preprocessing=preprocessing,
        allow_skipped=args.allow_skipped,
    )
    output = save_epoch_dataset(args.output, dataset)
    record_path = output.with_suffix(".record.json")
    print(
        f"[prepared] {output} cohort={args.cohort} contract={preprocessing.name} "
        f"sfreq={preprocessing.sfreq:g} l_freq={preprocessing.l_freq} "
        f"tmax_ms={preprocessing.tmax_ms:g} n_times={preprocessing.n_times} "
        f"phase={preprocessing.filter_phase} X={dataset.X.shape} "
        f"subjects={len(set(dataset.subject_ids))} "
        f"candidate_chain={dataset.event_timeline.supports_full_candidate_chain}",
        flush=True,
    )
    print(f"[record] {record_path}", flush=True)


if __name__ == "__main__":
    main()
