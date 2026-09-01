"""Apply an explicit common-channel average-reference domain contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.domain import (  # noqa: E402
    adapt_common_channel_average_reference,
    namespace_epoch_dataset,
)
from data.epochs import load_epoch_dataset, save_epoch_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--target-channels", required=True, help="Comma-separated canonical EEG channels.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--subject-namespace", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    channels = tuple(value.strip() for value in args.target_channels.split(",") if value.strip())
    dataset = load_epoch_dataset(args.dataset_cache, validation="attested")
    adapted = adapt_common_channel_average_reference(
        dataset,
        channels,
        name=args.name,
    )
    if args.subject_namespace is not None:
        adapted = namespace_epoch_dataset(adapted, args.subject_namespace, name=args.name)
    output = save_epoch_dataset(args.output, adapted, compressed=not args.uncompressed)
    print(f"[adapted] {output}", flush=True)


if __name__ == "__main__":
    main()
