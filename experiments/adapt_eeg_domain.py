"""Select a declared domain montage without silently re-referencing it."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.domain import (  # noqa: E402
    adapt_common_channel_average_reference,
    adapt_observed_montage,
    ensure_native_reference_selection,
    namespace_epoch_dataset,
    project_binary_evidence_source_view,
)
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    materialize_dataset_lineage,
    save_epoch_dataset,
)
from data.lineage import DataLineage  # noqa: E402

DIRECT_PARENT_CACHE_SCHEMA = "n2p3_loaded_epoch_cache_parent/1"


def _bind_loaded_parent_cache(dataset, source_path: Path):
    verified_load = loaded_epoch_cache_attestation(dataset)
    lineage_record = {
        "schema": DIRECT_PARENT_CACHE_SCHEMA,
        "role": "direct_parent_cache",
        "verified_load": verified_load,
    }
    provenance_record = {
        **lineage_record,
        "path": str(source_path),
    }
    return replace(
        dataset,
        provenance={
            **dataset.provenance,
            "direct_parent_cache": provenance_record,
        },
        lineage=DataLineage.derive(
            [materialize_dataset_lineage(dataset)],
            operation="bind_loaded_epoch_cache",
            parameters=lineage_record,
        ),
        verified_cache_attestation=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument(
        "--target-channels", required=True, help="Comma-separated canonical EEG channels."
    )
    parser.add_argument(
        "--reference-mode",
        choices=("native_preserve", "common_average", "measured_channel"),
        default="native_preserve",
        help=(
            "Preserve the source reference by default; common_average is an explicit "
            "legacy/control transform."
        ),
    )
    parser.add_argument(
        "--min-common-channels",
        type=int,
        default=3,
        help="Minimum selected channels allowed for common_average mode.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--subject-namespace", default=None)
    parser.add_argument("--missing-channel-policy", choices=("require", "mask"), default="require")
    parser.add_argument("--reference-channel", default=None)
    parser.add_argument(
        "--event-contract",
        choices=("preserve", "binary_evidence"),
        default="preserve",
        help="Use binary_evidence for source ERP learning without candidate-task assumptions.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    if args.missing_channel_policy == "mask" and args.reference_mode == "common_average":
        parser.error(
            "masked montage requires native_preserve; variable observed-channel CAR is not a common reference"
        )
    if (args.reference_mode == "measured_channel") != (args.reference_channel is not None):
        parser.error("measured_channel mode requires exactly one explicit --reference-channel")

    channels = tuple(value.strip() for value in args.target_channels.split(",") if value.strip())
    if args.min_common_channels < 2:
        parser.error("--min-common-channels must be at least two")
    if args.reference_mode == "common_average" and len(channels) < args.min_common_channels:
        parser.error(
            "common_average mode requires at least "
            f"{args.min_common_channels} selected channels; use native_preserve for the "
            "active route."
        )
    source_path = Path(args.dataset_cache).resolve(strict=True)
    dataset = load_epoch_dataset(source_path, validation="attested")
    dataset = _bind_loaded_parent_cache(dataset, source_path)
    if args.reference_mode == "common_average":
        adapted = adapt_common_channel_average_reference(dataset, channels, name=args.name)
    elif args.missing_channel_policy == "mask" or args.reference_mode == "measured_channel":
        adapted = adapt_observed_montage(
            dataset, channels, name=args.name, reference_channel=args.reference_channel
        )
        if args.missing_channel_policy == "require" and not bool(adapted.trial_channel_mask.all()):
            raise ValueError(
                "the requested montage contains unobserved channels; declare mask policy explicitly"
            )
    else:
        adapted = ensure_native_reference_selection(dataset, channels, name=args.name)
    if args.subject_namespace is not None:
        adapted = namespace_epoch_dataset(adapted, args.subject_namespace, name=args.name)
    if args.event_contract == "binary_evidence":
        adapted = project_binary_evidence_source_view(adapted)
    output = save_epoch_dataset(args.output, adapted, compressed=not args.uncompressed)
    print(f"[adapted] {output} reference_mode={args.reference_mode}", flush=True)


if __name__ == "__main__":
    main()
