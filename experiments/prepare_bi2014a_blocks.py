"""Freeze balanced target-subject blocks for BI2014a cross-decision transfer."""

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

from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402

SCHEMA = "n2p3_bi2014a_cross_decision_blocks/1"


def balanced_subject_blocks(
    subject_ids: np.ndarray, *, n_blocks: int
) -> tuple[tuple[str, ...], ...]:
    """Balance unlabeled epoch workload while keeping equal subject counts."""

    subjects, counts = np.unique(np.asarray(subject_ids).astype(str), return_counts=True)
    if n_blocks < 2 or len(subjects) < n_blocks or len(subjects) % n_blocks:
        raise ValueError("subjects must divide evenly across at least two blocks.")
    capacity = len(subjects) // n_blocks
    blocks: list[list[str]] = [[] for _ in range(n_blocks)]
    loads = np.zeros(n_blocks, dtype=np.int64)
    order = sorted(
        zip(subjects.tolist(), counts.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    for subject, count in order:
        candidates = [index for index, block in enumerate(blocks) if len(block) < capacity]
        block = min(candidates, key=lambda index: (int(loads[index]), index))
        blocks[block].append(subject)
        loads[block] += int(count)
    return tuple(tuple(sorted(block)) for block in blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    cache = read_epoch_cache_attestation(args.dataset_cache)
    blocks = balanced_subject_blocks(dataset.subject_ids, n_blocks=args.blocks)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    for index, subjects in enumerate(blocks):
        path = output / f"block_{index}.json"
        path.write_text(json.dumps(list(subjects), indent=2), encoding="utf-8")
        records.append(
            {
                "block": index,
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
        "assignment": (
            "sort subjects by descending unlabeled epoch count then id; assign to the "
            "lowest-load non-full block; equal subject capacity"
        ),
        "n_subjects": len(set(subject_ids.tolist())),
        "n_epochs": len(subject_ids),
        "blocks": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
