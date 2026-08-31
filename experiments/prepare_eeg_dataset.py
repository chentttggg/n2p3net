"""Prepare MOABB or raw-file EEG into the universal EpochDataset cache."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.brainsync import load_brainsync_sessions  # noqa: E402
from data.contract import CAUSAL_IIR_INITIAL_STATE  # noqa: E402
from data.epochs import P300_PERFORMANCE_PREPROCESSING, save_epoch_dataset  # noqa: E402
from data.manifest import build_manifest_dataset, load_manifest  # noqa: E402
from data.moabb import prepare_moabb_p300  # noqa: E402


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    output: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise argparse.ArgumentTypeError(f"Invalid subject range {token!r}.")
            output.extend(range(lo, hi + 1))
        else:
            output.append(int(token))
    return output


def _parse_channels(value: str) -> tuple[str, ...] | None:
    if value.strip().lower() == "native":
        return None
    channels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not channels:
        raise argparse.ArgumentTypeError("channels must be 'native' or a comma-separated list.")
    return channels


def _parse_float_pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers: START,END")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reference window values must be numbers") from exc


def _parse_optional_float(value: str) -> float | None:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number or 'none'") from exc


def _derive_exclusive_n_times(*, sfreq: float, tmin_ms: float, tmax_ms: float) -> int:
    """Derive the cache width for the preprocessing contract's exclusive right edge."""

    return int(np.floor((tmax_ms - tmin_ms) * sfreq / 1000.0 + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)

    moabb_parser = subparsers.add_parser("moabb", help="Prepare any installed MOABB P300 dataset")
    moabb_parser.add_argument("--dataset-class", required=True)
    moabb_parser.add_argument("--subjects", default=None, help="Comma/range list, e.g. 1-8,12")
    moabb_parser.add_argument("--channels", default="native")
    moabb_parser.add_argument("--montage", default="standard_1005")
    moabb_parser.add_argument("--target-label", default="Target")
    moabb_parser.add_argument("--sfreq", type=float, default=None)
    moabb_parser.add_argument(
        "--l-freq", type=_parse_optional_float, default=argparse.SUPPRESS
    )
    moabb_parser.add_argument(
        "--h-freq", type=_parse_optional_float, default=argparse.SUPPRESS
    )
    moabb_parser.add_argument("--tmin-ms", type=float, default=None)
    moabb_parser.add_argument("--tmax-ms", type=float, default=None)
    moabb_parser.add_argument("--n-times", type=int, default=None)
    moabb_parser.add_argument(
        "--baseline-mode",
        choices=("mean_only", "none", "trial_reference"),
        default=None,
    )
    moabb_parser.add_argument(
        "--filter-phase",
        choices=("zero", "forward"),
        default=None,
        help="Use forward with steady-state initialization for causal transfer caches.",
    )
    moabb_parser.add_argument(
        "--trial-reference-window-ms",
        type=_parse_float_pair,
        default=None,
        metavar="START,END",
    )
    moabb_parser.add_argument(
        "--trial-reference-center",
        choices=("mean", "median"),
        default=None,
    )
    moabb_parser.add_argument(
        "--trial-reference-scale",
        choices=("none",),
        default=None,
    )
    moabb_parser.add_argument("--output", required=True)
    moabb_parser.add_argument("--uncompressed", action="store_true")

    manifest_parser = subparsers.add_parser(
        "manifest", help="Prepare arbitrary MNE-supported raw recordings"
    )
    manifest_parser.add_argument("--manifest", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--uncompressed", action="store_true")

    brainsync_parser = subparsers.add_parser(
        "brainsync", help="Prepare one or more BrainSync GTN sessions into EpochDataset"
    )
    brainsync_parser.add_argument(
        "--session-dir",
        action="append",
        required=True,
        help="Repeat for each session/decision to include in the target cache.",
    )
    brainsync_parser.add_argument("--output", required=True)
    brainsync_parser.add_argument("--uncompressed", action="store_true")

    args = parser.parse_args()
    if args.source == "moabb":
        preprocessing_overrides = {
            key: value
            for key, value in {
                "tmin_ms": args.tmin_ms,
                "tmax_ms": args.tmax_ms,
                "n_times": args.n_times,
                "baseline_mode": args.baseline_mode,
                "trial_reference_window_ms": args.trial_reference_window_ms,
                "trial_reference_center": args.trial_reference_center,
                "trial_reference_scale": args.trial_reference_scale,
                "sfreq": args.sfreq,
                "filter_phase": args.filter_phase,
            }.items()
            if value is not None
        }
        for argument, field in (
            ("l_freq", "l_freq"),
            ("h_freq", "h_freq"),
        ):
            if hasattr(args, argument):
                preprocessing_overrides[field] = getattr(args, argument)
        if args.n_times is None and any(
            key in preprocessing_overrides for key in ("sfreq", "tmin_ms", "tmax_ms")
        ):
            sfreq = float(
                preprocessing_overrides.get("sfreq", P300_PERFORMANCE_PREPROCESSING.sfreq)
            )
            tmin_ms = float(
                preprocessing_overrides.get("tmin_ms", P300_PERFORMANCE_PREPROCESSING.tmin_ms)
            )
            tmax_ms = float(
                preprocessing_overrides.get("tmax_ms", P300_PERFORMANCE_PREPROCESSING.tmax_ms)
            )
            preprocessing_overrides["n_times"] = _derive_exclusive_n_times(
                sfreq=sfreq,
                tmin_ms=tmin_ms,
                tmax_ms=tmax_ms,
            )
        preprocessing = replace(P300_PERFORMANCE_PREPROCESSING, **preprocessing_overrides)
        if preprocessing.filter_phase == "forward":
            preprocessing = replace(
                preprocessing,
                causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
            )
        elif preprocessing.filter_phase == "zero":
            preprocessing = replace(preprocessing, causal_iir_initial_state="not_applicable")
        preprocessing.validate()
        dataset = prepare_moabb_p300(
            args.dataset_class,
            subjects=_parse_int_list(args.subjects),
            channels=_parse_channels(args.channels),
            montage=args.montage,
            preprocessing=preprocessing,
            target_label=args.target_label,
        )
    elif args.source == "manifest":
        dataset = build_manifest_dataset(load_manifest(args.manifest))
    else:
        dataset = load_brainsync_sessions(args.session_dir)

    output = save_epoch_dataset(args.output, dataset, compressed=not args.uncompressed)
    record_path = output.with_suffix(".record.json")
    print(
        f"[prepared] {output} X={dataset.X.shape} subjects={len(set(dataset.subject_ids))} "
        f"channels={list(dataset.channel_names)}",
        flush=True,
    )
    print(f"[record] {record_path}", flush=True)


if __name__ == "__main__":
    main()
