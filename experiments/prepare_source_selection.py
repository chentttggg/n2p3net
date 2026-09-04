"""Build an auditable PDS source-row selection from two domain feature matrices.

Inputs are the ``*.npz`` matrices written by ``prepare_domain_features.py``.
The output JSON carries the full :class:`PDSSelection` record plus the source
and target feature-matrix records, so downstream training can bind the
selection to the exact cache/feature artifacts that produced it.  The default
sampler is ``target_similarity_diverse``; ``--sampler dense_pds`` selects the
faithful Flores et al. (2024) dart-throwing sampler instead.
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

from data.epochs import TARGET_CALIBRATION_SCOPE_SCHEMA  # noqa: E402
from train.source_sampling import (  # noqa: E402
    VALID_SELECTION_SCOPES,
    array_digest,
    select_dense_pds_rows,
    select_pds_rows,
)

FEATURE_MATRIX_SCHEMA = "n2p3_domain_feature_matrix/1"


def _load_feature_matrix(path: Path, *, feature_key: str) -> tuple[np.ndarray, dict[str, object]]:
    resolved = path.resolve(strict=True)
    archive = np.load(resolved, allow_pickle=False)
    if "features" not in archive.files:
        raise ValueError(f"{resolved} lacks a 'features' matrix.")
    if "feature_key" in archive.files:
        stored_key = str(np.asarray(archive["feature_key"]).reshape(-1)[0])
        if stored_key != feature_key:
            raise ValueError(
                f"{resolved} feature_key {stored_key!r} disagrees with --feature-key {feature_key!r}."
            )
    if "schema" in archive.files:
        stored_schema = str(np.asarray(archive["schema"]).reshape(-1)[0])
        if stored_schema != FEATURE_MATRIX_SCHEMA:
            raise ValueError(f"{resolved} has unsupported feature-matrix schema {stored_schema!r}.")
    features = np.asarray(archive["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError(f"{resolved} features must have shape (N,P), got {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError(f"{resolved} features contain NaN or infinity.")
    record_path = resolved.with_suffix(".record.json")
    if not record_path.exists():
        raise ValueError(f"{resolved} lacks the adjacent feature record {record_path.name}.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{record_path} does not contain a JSON object.")
    if record.get("schema") != FEATURE_MATRIX_SCHEMA:
        raise ValueError(f"{record_path} has unsupported feature-matrix schema.")
    if record.get("source_cache_sha256") is None:
        raise ValueError(f"{record_path} lacks source_cache_sha256.")
    if record.get("feature_dim") != int(features.shape[1]):
        raise ValueError(f"{record_path} feature_dim disagrees with the matrix shape.")
    return features, record


def _load_vector(
    path: Path | None,
    *,
    key: str,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    archive = np.load(resolved, allow_pickle=False)
    if key not in archive.files:
        raise ValueError(f"{resolved} lacks vector {key!r} for {name}.")
    values = np.asarray(archive[key])
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(f"{name} must be a vector aligned with source rows ({expected_length}).")
    return values


def _parse_pds_mixture(
    text: str,
    unique_labels: list[object],
) -> tuple[float, ...]:
    """Parse ``--pds-mixture`` as JSON or comma-separated sorted-label weights."""
    stripped = text.strip()
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        expected_keys = {str(label) for label in unique_labels}
        if set(parsed) != expected_keys:
            raise ValueError(
                "JSON object keys must be exactly the sorted unique source labels "
                f"{expected_keys!r}."
            )
        raw_weights: list[object] = [parsed[str(label)] for label in unique_labels]
    elif isinstance(parsed, list):
        raw_weights = parsed
    elif parsed is None:
        raw_weights = [part.strip() for part in stripped.split(",")]
    else:
        raise ValueError("expected a JSON list/object or comma-separated weights.")
    if not raw_weights or any(
        not isinstance(part, str) and not np.isscalar(part) for part in raw_weights
    ):
        raise ValueError("expected non-empty numeric mixture weights.")
    try:
        weights = tuple(float(part) for part in raw_weights)
    except (TypeError, ValueError) as error:
        raise ValueError("expected numeric mixture weights.") from error
    if len(weights) != len(unique_labels):
        raise ValueError(
            f"expected {len(unique_labels)} weights aligned to sorted unique labels "
            f"{[str(label) for label in unique_labels]!r}."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("mixture weights must be finite and positive.")
    if not np.isclose(sum(weights), 1.0, rtol=1e-9, atol=1e-9):
        raise ValueError("mixture weights must sum to one.")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-select", type=int, required=True)
    parser.add_argument(
        "--selection-scope",
        choices=sorted(VALID_SELECTION_SCOPES),
        default="target_calibration",
        help="Declared target buffer used by the sampler; target test scope is forbidden.",
    )
    parser.add_argument(
        "--sampler",
        choices=["target_similarity_diverse", "dense_pds"],
        default="target_similarity_diverse",
        help=(
            "Source-row sampler.  target_similarity_diverse is the default TSD "
            "greedy sampler; dense_pds is the faithful Flores et al. dart-throwing "
            "sampler and ignores all TSD tuning flags."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument(
        "--pds-radius",
        type=float,
        default=None,
        help="Dense PDS hard-disk rejection radius; required for --sampler dense_pds.",
    )
    parser.add_argument(
        "--pds-mixture",
        default=None,
        help=(
            "Dense PDS mingling-mixture.  Comma-separated weights are aligned with "
            "sorted unique source labels, e.g. 0.5,0.5; JSON list/object is also "
            "accepted.  Default is the empirical label frequency."
        ),
    )
    parser.add_argument(
        "--pds-max-attempts",
        type=int,
        default=10000,
        help="Maximum dart-throwing attempts before dense_pds falls back to max-min rows.",
    )
    parser.add_argument(
        "--feature-key",
        default="features",
        help="Feature matrix key inside the prepared *.npz inputs.",
    )
    parser.add_argument("--source-labels", type=Path, default=None)
    parser.add_argument("--source-labels-key", default="labels")
    parser.add_argument("--source-groups", type=Path, default=None)
    parser.add_argument("--source-groups-key", default="groups")
    parser.add_argument("--min-per-label", type=int, default=0)
    parser.add_argument("--min-per-group", type=int, default=0)
    args = parser.parse_args()

    output = args.output.resolve(strict=False)
    if output.suffix.lower() != ".json":
        parser.error("--output must end with .json")
    if output.exists():
        raise FileExistsError(f"selection output already exists: {output}")

    source_features, source_record = _load_feature_matrix(
        args.source_features, feature_key=args.feature_key
    )
    target_features, target_record = _load_feature_matrix(
        args.target_features, feature_key=args.feature_key
    )

    target_scope = target_record.get("dataset_scope")
    if args.selection_scope == "target_calibration" and target_scope:
        if target_scope.get("schema") != TARGET_CALIBRATION_SCOPE_SCHEMA:
            raise ValueError(
                "selection_scope='target_calibration' requires target features derived "
                "from a target-calibration-scoped cache "
                f"({TARGET_CALIBRATION_SCOPE_SCHEMA}); target feature record declares "
                f"{target_scope.get('schema')!r}."
            )

    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "source and target feature dimensions must match: "
            f"{source_features.shape[1]} vs {target_features.shape[1]}."
        )

    labels = _load_vector(
        args.source_labels,
        key=args.source_labels_key,
        expected_length=len(source_features),
        name="source labels",
    )
    groups = _load_vector(
        args.source_groups,
        key=args.source_groups_key,
        expected_length=len(source_features),
        name="source groups",
    )
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        if bool((groups == "").any()):
            raise ValueError("source groups must be non-empty strings.")

    dense_pds = args.sampler == "dense_pds"
    if dense_pds:
        if args.pds_radius is None:
            parser.error("--sampler dense_pds requires --pds-radius")
        if args.pds_max_attempts < 1:
            parser.error("--pds-max-attempts must be a positive integer")
        if labels is None:
            parser.error("--sampler dense_pds requires --source-labels")
        if args.min_per_label != 0:
            parser.error("--min-per-label is TSD-only and cannot be used with dense_pds")
        if args.min_per_group != 0:
            parser.error("--min-per-group is TSD-only and cannot be used with dense_pds")
        if args.target_temperature != 1.0:
            parser.error("--target-temperature is TSD-only and cannot be set for dense_pds")
        if args.diversity_weight != 1.0:
            parser.error("--diversity-weight is TSD-only and cannot be set for dense_pds")
        if args.bandwidth is not None:
            parser.error("--bandwidth is TSD-only and cannot be set for dense_pds")
        mixture = None
        if args.pds_mixture is not None:
            unique_labels = sorted(np.unique(labels).tolist())
            try:
                mixture = _parse_pds_mixture(args.pds_mixture, unique_labels)
            except ValueError as error:
                parser.error(f"invalid --pds-mixture: {error}")
        selection = select_dense_pds_rows(
            source_features,
            target_features,
            n_select=args.n_select,
            selection_scope=args.selection_scope,
            seed=args.seed,
            labels=labels,
            pds_radius=args.pds_radius,
            pds_mingling_mixture=mixture,
            pds_max_attempts=args.pds_max_attempts,
        )
    else:
        selection = select_pds_rows(
            source_features,
            target_features,
            n_select=args.n_select,
            selection_scope=args.selection_scope,
            seed=args.seed,
            target_temperature=args.target_temperature,
            diversity_weight=args.diversity_weight,
            bandwidth=args.bandwidth,
            labels=labels,
            groups=groups,
            min_per_label=args.min_per_label,
            min_per_group=args.min_per_group,
        )

    record = selection.record()
    record.update(
        {
            "n_select": int(args.n_select),
            "feature_key": args.feature_key,
            "source_features_path": str(args.source_features.resolve(strict=True)),
            "target_features_path": str(args.target_features.resolve(strict=True)),
            "source_cache_sha256": str(source_record["source_cache_sha256"]),
            "target_cache_sha256": str(target_record["source_cache_sha256"]),
            "source_feature_record": source_record,
            "target_feature_record": target_record,
            "source_labels": (
                None
                if labels is None
                else {
                    "path": str(args.source_labels.resolve(strict=True)),
                    "key": args.source_labels_key,
                    "digest": array_digest(labels),
                }
            ),
            "source_groups": (
                None
                if groups is None
                else {
                    "path": str(args.source_groups.resolve(strict=True)),
                    "key": args.source_groups_key,
                    "digest": array_digest(groups),
                }
            ),
            "min_per_label": int(args.min_per_label),
            "min_per_group": int(args.min_per_group),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "selection_record": str(output),
                "sampler_kind": selection.sampler_kind,
                "n_select": int(len(selection.selected_indices)),
                "source_count": int(selection.source_count),
                "target_count": int(selection.target_count),
                "selection_scope": selection.selection_scope,
                "source_digest": selection.source_digest,
                "target_digest": selection.target_digest,
                "selected_digest": selection.selected_digest,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
