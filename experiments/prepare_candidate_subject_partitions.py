"""Freeze balanced target-subject partitions for candidate transfer."""

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

from data.candidate_task import candidate_task_contract_from_provenance  # noqa: E402
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402

SCHEMA = "n2p3_candidate_subject_partition_manifest/1"


def balanced_subject_partitions(
    subject_ids: np.ndarray, *, n_partitions: int
) -> tuple[tuple[str, ...], ...]:
    """Balance unlabeled epoch workload while keeping equal subject counts."""

    subjects, counts = np.unique(np.asarray(subject_ids).astype(str), return_counts=True)
    if n_partitions < 2 or len(subjects) < n_partitions or len(subjects) % n_partitions:
        raise ValueError("subjects must divide evenly across at least two partitions.")
    capacity = len(subjects) // n_partitions
    partitions: list[list[str]] = [[] for _ in range(n_partitions)]
    loads = np.zeros(n_partitions, dtype=np.int64)
    order = sorted(
        zip(subjects.tolist(), counts.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    for subject, count in order:
        candidates = [
            index for index, partition in enumerate(partitions) if len(partition) < capacity
        ]
        partition = min(candidates, key=lambda index: (int(loads[index]), index))
        partitions[partition].append(subject)
        loads[partition] += int(count)
    return tuple(tuple(sorted(partition)) for partition in partitions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partitions", type=int, default=4)
    args = parser.parse_args()

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    cache = read_epoch_cache_attestation(args.dataset_cache)
    candidate_task = candidate_task_contract_from_provenance(dataset.provenance)
    partitions = balanced_subject_partitions(dataset.subject_ids, n_partitions=args.partitions)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    for index, subjects in enumerate(partitions):
        path = output / f"partition_{index}.json"
        path.write_text(json.dumps(list(subjects), indent=2), encoding="utf-8")
        records.append(
            {
                "partition": index,
                "subjects": list(subjects),
                "n_subjects": len(subjects),
                "n_epochs": int(np.isin(subject_ids, subjects).sum()),
                "file": path.name,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "dataset_cache_sha256": cache["sha256"],
        "dataset_cache_byte_size": cache["byte_size"],
        "dataset_id": candidate_task.dataset_id,
        "task_id": candidate_task.task_id,
        "candidate_task_contract": candidate_task.record(),
        "candidate_task_contract_digest": semantic_sha256(candidate_task.record()),
        "assignment": (
            "sort subjects by descending unlabeled epoch count then id; assign to the "
            "lowest-load non-full partition; equal subject capacity"
        ),
        "n_subjects": len(set(subject_ids.tolist())),
        "n_epochs": len(subject_ids),
        "partitions": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
