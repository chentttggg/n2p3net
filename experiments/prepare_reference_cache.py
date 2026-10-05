"""Derive one reference-normalized cache from an attested epoch cache.

This is the single-cache complement of ``prepare_multidomain_source.py``.  The
experiment workflow builds derived caches locally and uploads only caches to
the cloud; this script keeps that derivation explicit and attested instead of
asking the runner to transform tensors at launch time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.domain import (
    ensure_common_channel_average_reference,
    ensure_native_reference_selection,
)
from data.epochs import load_epoch_dataset, save_epoch_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--target-channels", required=True)
    parser.add_argument(
        "--reference-mode",
        choices=("native_preserve", "common_average"),
        required=True,
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    channels = tuple(value.strip() for value in args.target_channels.split(",") if value.strip())
    output = args.output.resolve(strict=False)
    collisions = [output, output.with_suffix(".record.json")]
    if any(path.exists() for path in collisions):
        raise FileExistsError(
            "reference cache derivation requires a new output path; existing artifacts: "
            + ", ".join(str(path) for path in collisions)
        )

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    if args.reference_mode == "native_preserve":
        derived = ensure_native_reference_selection(dataset, channels, name=args.name)
    else:
        derived = ensure_common_channel_average_reference(dataset, channels, name=args.name)
    saved = save_epoch_dataset(output, derived, compressed=not args.uncompressed)
    print(
        f"[reference-cache] {saved} X={derived.X.shape} "
        f"reference={derived.provenance.get('source_reference')} "
        f"channels={list(derived.channel_names)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
