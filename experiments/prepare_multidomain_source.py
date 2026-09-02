"""Build one supervised source cache from explicitly adapted source domains."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.domain import (  # noqa: E402
    SOURCE_DOMAIN_AXIS_SCHEMA,
    SOURCE_DOMAIN_COLUMN,
    annotate_source_domain,
    ensure_common_channel_average_reference,
    ensure_epoch_dataset_namespace,
    project_binary_evidence_source_view,
)
from data.epochs import (  # noqa: E402
    concatenate_epoch_datasets,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    save_epoch_dataset,
)


def _source_spec(value: str) -> tuple[str, Path]:
    namespace, separator, path = value.partition("=")
    if not separator or not namespace.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--source must use NAMESPACE=cache.npz")
    return namespace.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source_spec, required=True)
    parser.add_argument("--target-channels", required=True)
    parser.add_argument(
        "--event-contract",
        choices=("preserve", "binary_evidence"),
        default="preserve",
        help="Explicit common event view; binary_evidence drops candidate-only source fields.",
    )
    parser.add_argument("--name", default="MultiSource-P300")
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    if len(args.source) < 2:
        parser.error("multi-domain source preparation requires at least two --source values")
    namespaces = [namespace for namespace, _ in args.source]
    if len(set(namespaces)) != len(namespaces):
        parser.error("source namespaces must be unique")
    channels = tuple(value.strip() for value in args.target_channels.split(",") if value.strip())
    output_path = Path(args.output).resolve(strict=False)
    output_artifacts = (
        output_path,
        output_path.with_suffix(".record.json"),
        output_path.with_suffix(output_path.suffix + ".tmp.npz"),
    )
    collisions = [path for path in output_artifacts if path.exists()]
    if collisions:
        raise FileExistsError(
            "multi-domain source preparation requires a new output path; existing artifacts: "
            + ", ".join(str(path) for path in collisions)
        )

    adapted = []
    source_records = []
    for namespace, path in args.source:
        dataset = load_epoch_dataset(path, require_labels=True, validation="attested")
        cache_attestation = loaded_epoch_cache_attestation(dataset)
        namespace_preexisting = dataset.provenance.get("subject_namespace") is not None
        aligned = ensure_common_channel_average_reference(
            dataset,
            channels,
        )
        aligned = ensure_epoch_dataset_namespace(aligned, namespace)
        aligned = annotate_source_domain(aligned, namespace)
        candidate_metadata_projected = False
        if args.event_contract == "binary_evidence":
            source_aligned = aligned
            aligned = project_binary_evidence_source_view(aligned)
            candidate_metadata_projected = aligned is not source_aligned
        adapted.append(aligned)
        source_records.append(
            {
                "namespace": namespace,
                "namespace_preexisting": namespace_preexisting,
                "candidate_metadata_projected": candidate_metadata_projected,
                "cache": str(path.resolve()),
                "cache_sha256": str(cache_attestation["sha256"]),
                "verified_cache_attestation": cache_attestation,
                "dataset_name": dataset.name,
                "n_subjects": len(set(dataset.subject_ids)),
                "n_epochs": dataset.n_epochs,
                "parent_source_reference": dataset.provenance.get("source_reference"),
                "identity_table_digest": (
                    dataset.identity_table.digest()
                    if dataset.identity_table is not None
                    else None
                ),
            }
        )
    reference = adapted[0].provenance["source_reference"]
    domain_adapter = adapted[0].provenance.get("domain_adapter")
    if not isinstance(domain_adapter, Mapping) or any(
        dataset.provenance.get("domain_adapter") != domain_adapter for dataset in adapted[1:]
    ):
        raise ValueError("multi-domain sources do not share one verified domain_adapter contract.")
    merged = concatenate_epoch_datasets(
        adapted,
        name=args.name,
        provenance={
            "source": "explicit_multidomain_common_channel_car",
            "source_reference": reference,
            "domain_adapter": dict(domain_adapter),
            "target_channels": list(adapted[0].channel_names),
            "event_contract": args.event_contract,
            "source_domain_axis": {
                "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
                "column": SOURCE_DOMAIN_COLUMN,
                "domains": namespaces,
            },
            "sources": source_records,
        },
    )
    merged.provenance["identity_table_digest"] = (
        merged.identity_table.digest() if merged.identity_table is not None else None
    )
    output = save_epoch_dataset(output_path, merged, compressed=not args.uncompressed)
    print(
        f"[multidomain] {output} X={merged.X.shape} "
        f"subjects={len(set(merged.subject_ids))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
