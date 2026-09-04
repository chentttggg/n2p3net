"""Derive a shape-preserving Euclidean-Alignment epoch cache from an attested cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.domain import source_domain_ids  # noqa: E402
from data.epochs import (  # noqa: E402
    DataLineage,
    EpochDataset,
    epoch_row_identity_digest,
    load_epoch_dataset,
    materialize_dataset_identity,
    materialize_dataset_lineage,
    save_epoch_dataset,
)
from data.euclidean_alignment import (  # noqa: E402
    EuclideanAlignmentConfig,
    fit_euclidean_alignment,
    fit_per_domain_euclidean_alignment,
    transform_euclidean_alignment,
)
from data.qc_features import compute_epoch_qc_features  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fit-scope",
        choices=("source_rows", "target_calibration", "unlabelled_buffer", "per_domain"),
        required=True,
    )
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--max-condition-number", type=float, default=1e12)
    parser.add_argument("--name", default=None)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve(strict=False)
    collisions = [output, output.with_suffix(".record.json")]
    if any(path.exists() for path in collisions):
        raise FileExistsError("EA cache derivation requires a new output path.")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    if args.fit_scope == "per_domain":
        if dataset.provenance.get("source_domain_axis") is None:
            parser.error("--fit-scope per_domain requires a source_domain_axis provenance.")
        domains = source_domain_ids(dataset)
    else:
        domains = None

    config = EuclideanAlignmentConfig(
        shrinkage=args.shrinkage,
        regularization=args.regularization,
        max_condition_number=args.max_condition_number,
    )
    if domains is not None:
        states = fit_per_domain_euclidean_alignment(dataset.X, domains, config=config)
        transformed = np.empty_like(dataset.X, dtype=np.float32)
        for domain, state in states.items():
            mask = domains == domain
            transformed[mask] = transform_euclidean_alignment(dataset.X[mask], state)
        alignment_record = {
            "schema": "n2p3_euclidean_alignment_fit/1",
            "scope": "per_domain",
            "states": {domain: state.record() for domain, state in states.items()},
            "row_identity_digest": epoch_row_identity_digest(dataset),
        }
    else:
        state = fit_euclidean_alignment(dataset.X, fit_scope=args.fit_scope, config=config)
        transformed = transform_euclidean_alignment(dataset.X, state)
        alignment_record = {
            "schema": "n2p3_euclidean_alignment_fit/1",
            "scope": args.fit_scope,
            "state": state.record(),
            "row_identity_digest": epoch_row_identity_digest(dataset),
        }

    if dataset.trial_channel_mask is not None and not bool(
        np.asarray(dataset.trial_channel_mask).all()
    ):
        raise ValueError("EA cache derivation requires complete observed channel masks.")

    adapted = EpochDataset(
        name=args.name or f"{dataset.name}-ea-{args.fit_scope}",
        X=np.ascontiguousarray(transformed, dtype=np.float32),
        y=np.asarray(dataset.y, dtype=np.int64),
        subject_ids=np.asarray(dataset.subject_ids).astype(str),
        channel_names=tuple(dataset.channel_names),
        channel_positions_m=np.asarray(dataset.channel_positions_m, dtype=np.float32),
        channel_mask=np.ones(dataset.n_channels, dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline,
        metadata=dataset.metadata.copy(),
        provenance={
            **dataset.provenance,
            "source": "explicit_euclidean_alignment",
            "parent_dataset_name": dataset.name,
            "data_alignment": alignment_record,
        },
        trial_channel_mask=None,
        qc_features=compute_epoch_qc_features(
            transformed,
            channel_mask=np.ones(dataset.n_channels, dtype=bool),
        ),
        identity_table=materialize_dataset_identity(dataset),
        lineage=DataLineage.derive(
            [materialize_dataset_lineage(dataset)],
            operation="euclidean_alignment",
            parameters={"fit_scope": args.fit_scope, "shrinkage": args.shrinkage},
        ),
    )
    adapted.validate(require_labels=True)
    saved = save_epoch_dataset(output, adapted, compressed=not args.uncompressed)
    print(
        json.dumps(
            {
                "output": str(saved),
                "fit_scope": args.fit_scope,
                "shape": list(transformed.shape),
                "row_identity_digest": alignment_record["row_identity_digest"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
