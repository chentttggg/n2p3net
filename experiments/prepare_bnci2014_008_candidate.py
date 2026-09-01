"""Prepare strict BNCI2014-008 row/column candidate caches.

The default run requires all eight official Level-5 MAT files.  ``--subjects``
may select a smaller development subset, but the root inventory is still
validated before any cache is produced.
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

from data.bnci2014_008_candidate import (  # noqa: E402
    BNCI2014_008_CANDIDATE_TASK_CONTRACT,
    BNCI2014_008_DATASET_ID,
    BNCI2014_008_REPETITIONS_PER_SELECTION,
    BNCI2014_008_SOURCE_GROUND,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    BNCI2014_008_SUBJECT_IDS,
    bnci2014_008_mat_source_contract,
    build_bnci2014_008_subject_dataset,
    discover_bnci2014_008_files,
    load_bnci2014_008_candidate_record,
)
from data.candidate_task import validate_candidate_membership_metadata  # noqa: E402
from data.contract import (  # noqa: E402
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_causal_p300_input_contract,
)
from data.epochs import (  # noqa: E402
    concatenate_epoch_datasets,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.raw_artifacts import verify_raw_artifact_manifest  # noqa: E402


def _parse_subjects(value: str) -> tuple[str, ...]:
    if not value.strip():
        return BNCI2014_008_SUBJECT_IDS
    selected: set[int] = set()
    for token in value.split(","):
        item = token.strip().upper().removeprefix("A")
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start = int(start_text.removeprefix("A"))
            stop = int(stop_text.removeprefix("A"))
            if start > stop:
                raise ValueError(f"Invalid descending subject range {token!r}.")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))
    invalid = sorted(selected - set(range(1, 9)))
    if invalid or not selected:
        raise ValueError(f"BNCI2014-008 subject numbers must be 1..8; got {invalid}.")
    return tuple(f"A{index:02d}" for index in sorted(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-artifact-manifest", type=Path, required=True)
    parser.add_argument("--raw-artifact-root", type=Path, required=True)
    parser.add_argument("--raw-artifact-snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma/range subset such as 1-3,8; default is all eight subjects.",
    )
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_artifact_attestation = verify_raw_artifact_manifest(
        args.raw_artifact_manifest,
        args.raw_artifact_root,
        snapshot_root=args.raw_artifact_snapshot_root,
        cache_workspace_root=args.output.parent,
        expected_dataset_class=BNCI2014_008_DATASET_ID,
    )
    inventory = discover_bnci2014_008_files(raw_artifact_attestation)
    selected_ids = set(_parse_subjects(args.subjects))
    paths = tuple(path for path in inventory if Path(path).stem in selected_ids)
    preprocessing = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    preprocessing.validate()
    assert_causal_p300_input_contract(preprocessing)

    datasets = []
    audits: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    for path in paths:
        record = load_bnci2014_008_candidate_record(
            path,
            raw_artifact_attestation=raw_artifact_attestation,
        )
        dataset = build_bnci2014_008_subject_dataset(
            record,
            preprocessing=preprocessing,
        )
        datasets.append(dataset)
        pairs = np.unique(np.column_stack((record.target_row, record.target_col)), axis=0)
        audit = {
            "subject": record.subject_id,
            "source_sha256": record.source_sha256,
            "n_samples": int(record.eeg_uv.shape[0]),
            "n_flash_onsets": int(len(record.flash_sample)),
            "n_selections": int(len(np.unique(record.selection_id))),
            "n_repetitions_per_selection": BNCI2014_008_REPETITIONS_PER_SELECTION,
            "n_target_pairs_observed": int(len(pairs)),
            "n_epochs": dataset.n_epochs,
            "complete_event_timeline": dataset.event_timeline.complete,
            "online_causal": dataset.event_timeline.online_causal,
        }
        audits.append(audit)
        snapshot = raw_artifact_attestation.snapshot_for(record.source_relative_path)
        source_files.append(
            {
                "subject": record.subject_id,
                "source_relative_path": record.source_relative_path,
                "sha256": record.source_sha256,
                "size_bytes": snapshot.size_bytes,
                "snapshot_relative_path": snapshot.snapshot_relative_path,
                "snapshot_role": snapshot.role,
            }
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)

    combined = concatenate_epoch_datasets(
        datasets,
        name="BNCI2014-008-candidate",
        provenance={
            "source": "bnci_horizon_008_2014_mat_level5",
            "dataset_id": BNCI2014_008_DATASET_ID,
            "raw_artifact_manifest_sha256": raw_artifact_attestation.manifest_sha256,
            "raw_artifact_manifest_role": "explicit_verified_input_manifest",
            "raw_artifact_official_source": dict(raw_artifact_attestation.official_source),
            "raw_artifact_official_record": dict(raw_artifact_attestation.official_record),
            "source_files": source_files,
            "source_reference": BNCI2014_008_SOURCE_REFERENCE,
            "source_ground": BNCI2014_008_SOURCE_GROUND,
            "source_sample_rate_hz": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
            "source_signal_unit": "uV",
            "signal_unit": "V",
            "mat_source_contract": bnci2014_008_mat_source_contract(),
            "candidate_task_contract": BNCI2014_008_CANDIDATE_TASK_CONTRACT.record(),
            "evidence_boundary": ("public_processed_ALS_development_data;not_product_confirmation"),
        },
    )
    validate_candidate_membership_metadata(
        combined.metadata,
        BNCI2014_008_CANDIDATE_TASK_CONTRACT,
        labels=combined.y,
    )
    assert_causal_p300_input_contract(combined.preprocessing)
    output = save_epoch_dataset(
        args.output,
        combined,
        compressed=not args.uncompressed,
    )
    audit_path = output.with_suffix(".candidate_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "dataset_id": BNCI2014_008_DATASET_ID,
                "candidate_task_contract": BNCI2014_008_CANDIDATE_TASK_CONTRACT.record(),
                "mat_source_contract": bnci2014_008_mat_source_contract(),
                "raw_artifact_manifest_sha256": raw_artifact_attestation.manifest_sha256,
                "source_files": source_files,
                "subjects": audits,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[prepared] {output}", flush=True)
    print(f"[audit] {audit_path}", flush=True)


if __name__ == "__main__":
    main()
