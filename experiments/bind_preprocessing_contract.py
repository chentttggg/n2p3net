"""Bind an attested cache to a canonical named preprocessing contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import SOURCE_COHORT_DATA_CONTRACTS  # noqa: E402
from data.epochs import (  # noqa: E402
    bind_named_preprocessing_contract,
    load_epoch_dataset,
    save_epoch_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument(
        "--contract",
        choices=tuple(SOURCE_COHORT_DATA_CONTRACTS),
        required=True,
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.dataset_cache).resolve(strict=True)
    output_path = Path(args.output).resolve(strict=False)
    if source_path == output_path:
        raise ValueError("contract binding output must not overwrite its attested source cache.")
    output_artifacts = (
        output_path,
        output_path.with_suffix(".record.json"),
        output_path.with_suffix(output_path.suffix + ".tmp.npz"),
    )
    collisions = [path for path in output_artifacts if path.exists()]
    if collisions:
        raise FileExistsError(
            "contract binding requires a new output path; existing artifacts: "
            + ", ".join(str(path) for path in collisions)
        )

    dataset = load_epoch_dataset(source_path, validation="attested")
    rebound = bind_named_preprocessing_contract(
        dataset,
        SOURCE_COHORT_DATA_CONTRACTS[args.contract],
        name=args.name,
    )
    output = save_epoch_dataset(
        output_path,
        rebound,
        compressed=not args.uncompressed,
    )
    print(f"[contract-bound] {output}", flush=True)


if __name__ == "__main__":
    main()
