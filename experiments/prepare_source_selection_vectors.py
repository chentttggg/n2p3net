"""Export attested per-row label/group vectors for source-row sampling.

``prepare_source_selection.py`` intentionally consumes feature matrices plus
optional label/group vectors.  This helper derives those vectors from an
attested ``EpochDataset`` so the byte-level provenance (cache SHA-256 and
vector digests) is explicit and auditable instead of relying on ad-hoc local
``np.save`` calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.epochs import (  # noqa: E402
    epoch_row_identity_digest,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
)
from train.source_sampling import array_digest  # noqa: E402


def _atomic_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-groups", type=Path, default=None)
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--groups-key", default="groups")
    parser.add_argument(
        "--group-column",
        default="subject",
        help="Metadata column used for source groups; default subject.",
    )
    args = parser.parse_args(argv)

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    if dataset.y is None:
        raise ValueError("source cache must carry labels.")
    if args.group_column not in dataset.metadata:
        raise ValueError(
            f"--group-column {args.group_column!r} is absent from dataset metadata."
        )
    for output in (args.output_labels, args.output_groups):
        if output is not None:
            resolved = output.resolve(strict=False)
            if resolved.suffix.lower() != ".npz":
                parser.error(f"{output} must end with .npz")
            if resolved.exists():
                raise FileExistsError(f"vector output already exists: {resolved}")

    labels = np.asarray(dataset.y, dtype=np.int64)
    groups = (
        None
        if args.output_groups is None
        else np.asarray(
            dataset.metadata[args.group_column].astype(str).to_list(), dtype=str
        )
    )
    _atomic_save(
        args.output_labels,
        **{args.labels_key: labels},
    )
    if groups is not None:
        if bool((groups == "").any()):
            raise ValueError("source groups must be non-empty strings.")
        _atomic_save(
            args.output_groups,
            **{args.groups_key: groups},
        )
    print(
        json.dumps(
            {
                "dataset_cache": str(args.dataset_cache.resolve()),
                "cache_sha256": str(loaded_epoch_cache_attestation(dataset)["sha256"]),
                "row_identity_digest": epoch_row_identity_digest(dataset),
                "labels": {
                    "path": str(args.output_labels.resolve()),
                    "key": args.labels_key,
                    "digest": array_digest(labels),
                    "n_rows": int(len(labels)),
                },
                "groups": (
                    None
                    if groups is None
                    else {
                        "path": str(args.output_groups.resolve()),
                        "key": args.groups_key,
                        "digest": array_digest(groups),
                        "n_rows": int(len(groups)),
                    }
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
