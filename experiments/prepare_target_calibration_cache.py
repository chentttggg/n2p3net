"""Derive a target-calibration-only EpochDataset cache from an attested cache.

The split reuses :func:`transfer.candidate_within_subject.candidate_calibration_decision_split`
so the calibration rows here are exactly the rows the candidate runner uses as
the known-prefix calibration stream.  Source-selection target features derived
from this cache therefore cannot leak held-out target-test decisions.
"""

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

from data.epochs import (  # noqa: E402
    TARGET_CALIBRATION_SCOPE_SCHEMA,
    DataLineage,
    EpochDataset,
    load_epoch_dataset,
    materialize_dataset_identity,
    materialize_dataset_lineage,
    save_epoch_dataset,
)
from data.qc_features import compute_epoch_qc_features  # noqa: E402
from transfer.candidate_within_subject import (  # noqa: E402
    candidate_calibration_decision_split,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-selections", type=int, required=True)
    parser.add_argument("--test-repetitions", type=int, required=True)
    parser.add_argument("--selection-repetition-floor", type=int, default=None)
    parser.add_argument("--max-test-selections", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args(argv)

    if args.calibration_selections < 1:
        parser.error("--calibration-selections must be at least one")
    if args.test_repetitions < 1:
        parser.error("--test-repetitions must be at least one")
    if args.selection_repetition_floor is not None and args.selection_repetition_floor < 1:
        parser.error("--selection-repetition-floor must be at least one")
    if args.max_test_selections is not None and args.max_test_selections < 1:
        parser.error("--max-test-selections must be at least one")

    output = args.output.resolve(strict=False)
    collisions = [output, output.with_suffix(".record.json")]
    if any(path.exists() for path in collisions):
        raise FileExistsError("target-calibration cache derivation requires a new output path.")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    split = candidate_calibration_decision_split(
        dataset,
        calibration_selections=args.calibration_selections,
        test_repetitions=args.test_repetitions,
        selection_repetition_floor=args.selection_repetition_floor,
        max_test_selections=args.max_test_selections,
    )
    mask = np.asarray(split.calibration_mask, dtype=bool)
    if not bool(mask.any()):
        raise ValueError("calibration split selected no rows; cannot derive a target buffer.")
    rows = np.flatnonzero(mask)
    subjects = np.asarray(dataset.subject_ids)[rows].astype(str)
    parent_identity = materialize_dataset_identity(dataset)
    parent_lineage = materialize_dataset_lineage(dataset)

    calibration_scope = {
        "schema": TARGET_CALIBRATION_SCOPE_SCHEMA,
        "calibration_selections": int(args.calibration_selections),
        "test_repetitions": int(args.test_repetitions),
        "selection_repetition_floor": int(
            args.test_repetitions
            if args.selection_repetition_floor is None
            else args.selection_repetition_floor
        ),
        "max_test_selections": args.max_test_selections,
        "usable_subjects": sorted(np.unique(subjects).tolist()),
        "excluded_subjects": dict(sorted(split.excluded_subjects.items())),
    }

    adapted = EpochDataset(
        name=args.name or f"{dataset.name}-target-calibration-{args.calibration_selections}",
        X=np.ascontiguousarray(np.asarray(dataset.X)[rows], dtype=np.float32),
        y=np.asarray(dataset.y, dtype=np.int64)[rows],
        subject_ids=subjects,
        channel_names=tuple(dataset.channel_names),
        channel_positions_m=np.asarray(dataset.channel_positions_m, dtype=np.float32),
        channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline.subset_rows(mask),
        metadata=dataset.metadata.iloc[rows].copy(),
        provenance={
            **dataset.provenance,
            "source": "explicit_target_calibration_subset",
            "parent_dataset_name": dataset.name,
            "target_calibration_scope": calibration_scope,
        },
        trial_channel_mask=(
            None
            if dataset.trial_channel_mask is None
            else np.asarray(dataset.trial_channel_mask, dtype=bool)[rows]
        ),
        qc_features=compute_epoch_qc_features(
            np.asarray(dataset.X)[rows],
            channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
            trial_channel_mask=(
                None
                if dataset.trial_channel_mask is None
                else np.asarray(dataset.trial_channel_mask, dtype=bool)[rows]
            ),
        ),
        identity_table=parent_identity.subset(np.unique(subjects).tolist()),
        lineage=DataLineage.derive(
            [parent_lineage],
            operation="target_calibration_subset",
            parameters={
                "calibration_selections": args.calibration_selections,
                "test_repetitions": args.test_repetitions,
                "selection_repetition_floor": calibration_scope["selection_repetition_floor"],
                "max_test_selections": args.max_test_selections,
            },
        ),
    )
    adapted.validate(require_labels=True)
    saved = save_epoch_dataset(output, adapted, compressed=not args.uncompressed)
    print(
        json.dumps(
            {
                "output": str(saved),
                "n_rows": int(len(rows)),
                "n_subjects": int(len(np.unique(subjects))),
                "parent_dataset_name": dataset.name,
                "calibration_scope_schema": TARGET_CALIBRATION_SCOPE_SCHEMA,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
