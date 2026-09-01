"""Apply an explicit common-channel average-reference domain contract."""

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
    namespace_epoch_dataset,
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
    parser.add_argument("--target-channels", required=True, help="Comma-separated canonical EEG channels.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--subject-namespace", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    channels = tuple(value.strip() for value in args.target_channels.split(",") if value.strip())
    source_path = Path(args.dataset_cache).resolve(strict=True)
    dataset = load_epoch_dataset(source_path, validation="attested")
    dataset = _bind_loaded_parent_cache(dataset, source_path)
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
