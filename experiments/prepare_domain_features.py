"""Build an explicit feature matrix from an attested EpochDataset for GALIA/PDS.

The default lift is identity flattening ``(N,C,T) -> (N,C*T)``, recorded
explicitly.  Optional spatial/temporal basis files may be supplied later;
missing channels are never padded implicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation


def _load_basis(path: Path | None, *, name: str) -> np.ndarray | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    basis = np.load(resolved, allow_pickle=False)
    if name not in basis.files:
        raise ValueError(f"{resolved} lacks {name!r}.")
    values = np.asarray(basis[name], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-lift", type=Path, default=None)
    parser.add_argument("--spatial-lift-key", default="spatial_lift")
    parser.add_argument("--temporal-basis", type=Path, default=None)
    parser.add_argument("--temporal-basis-key", default="temporal_basis")
    parser.add_argument("--feature-key", default="features")
    args = parser.parse_args()

    output = args.output.resolve(strict=False)
    if output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    if output.exists():
        raise FileExistsError(f"feature output already exists: {output}")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    spatial = _load_basis(args.spatial_lift, name=args.spatial_lift_key)
    temporal = _load_basis(args.temporal_basis, name=args.temporal_basis_key)
    if spatial is not None and spatial.shape[1] != dataset.n_channels:
        raise ValueError(
            f"spatial_lift columns {spatial.shape[1]} do not match channels {dataset.n_channels}."
        )
    if temporal is not None and temporal.shape[0] != dataset.n_times:
        raise ValueError(
            f"temporal_basis rows {temporal.shape[0]} do not match times {dataset.n_times}."
        )

    values = np.asarray(dataset.X, dtype=np.float64)
    if spatial is not None:
        values = np.einsum("pc,nct->npt", spatial, values, optimize=True)
    if temporal is not None:
        values = np.einsum("npt,tq->npq", values, temporal, optimize=True)
    features = np.ascontiguousarray(values.reshape(values.shape[0], -1), dtype=np.float32)
    spatial_kind = "identity" if spatial is None else "explicit_file"
    temporal_kind = "identity" if temporal is None else "explicit_file"
    record = {
        "schema": "n2p3_domain_feature_matrix/1",
        "source_cache": str(args.dataset_cache.resolve(strict=True)),
        "source_cache_sha256": str(loaded_epoch_cache_attestation(dataset)["sha256"]),
        "shape": list(features.shape),
        "spatial_lift": {
            "kind": spatial_kind,
            "shape": None if spatial is None else list(spatial.shape),
            "path": None if args.spatial_lift is None else str(args.spatial_lift.resolve(strict=True)),
            "key": args.spatial_lift_key,
        },
        "temporal_basis": {
            "kind": temporal_kind,
            "shape": None if temporal is None else list(temporal.shape),
            "path": (
                None
                if args.temporal_basis is None
                else str(args.temporal_basis.resolve(strict=True))
            ),
            "key": args.temporal_basis_key,
        },
        "channel_names": list(dataset.channel_names),
        "n_times": int(dataset.n_times),
        "feature_dim": int(features.shape[1]),
        "labels_included": False,
        "domain_axis_included": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        feature_key=np.asarray(args.feature_key),
        schema=np.asarray(record["schema"]),
    )
    record_path = output.with_suffix(".record.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[domain-features] {output} X={dataset.X.shape} features={features.shape} "
        f"spatial={spatial_kind} temporal={temporal_kind}",
        flush=True,
    )


if __name__ == "__main__":
    main()
