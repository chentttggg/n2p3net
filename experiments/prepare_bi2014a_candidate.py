"""Prepare BI2014a as a causal 6x6 speller candidate cache.

This is not a MOABB adapter: it reads ``subject_XX.csv`` directly because the
generic MOABB path discards the row/column flash codes required for
within-subject character decisions.
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

from data.bi2014a_candidate import (  # noqa: E402
    BI2014A_CHANNELS,
    build_bi2014a_subject_dataset,
    recover_bi2014a_candidates,
)
from data.contract import (  # noqa: E402
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_causal_p300_input_contract,
)
from data.epochs import (  # noqa: E402
    PreprocessingSpec,
    concatenate_epoch_datasets,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)

DEFAULT_BI_ROOT = ROOT / "mne_data" / "MNE-braininvaders2014a-data" / "zenodo" / "3266223"


def _preprocessing() -> PreprocessingSpec:
    return preprocessing_spec_from_contract(
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_BI_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--subjects",
        default="",
        help="comma/range list, e.g. 1-8,12; default is all 64",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    subject_dirs = sorted(
    (p for p in root.glob("subject_*") if p.is_dir()),
    key=lambda p: int(p.name.rsplit("_", 1)[1]),
)
    if args.subjects:
        selected: set[int] = set()
        for token in args.subjects.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = (int(part) for part in token.split("-", 1))
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))
        subject_dirs = [p for p in subject_dirs if int(p.name.rsplit("_", 1)[1]) in selected]
    if args.max_subjects is not None:
        subject_dirs = subject_dirs[: args.max_subjects]
    if not subject_dirs:
        raise ValueError("No BI2014a subject directories matched.")

    preprocessing = _preprocessing()
    preprocessing.validate()
    assert_causal_p300_input_contract(preprocessing)
    datasets = []
    audit = []
    for subject_dir in subject_dirs:
        recovered = recover_bi2014a_candidates(subject_dir)
        expected_targets = 2 * recovered.n_repetitions
        if not (len(recovered.flash_sample) == 12 * recovered.n_repetitions and np.count_nonzero(recovered.target_label == 2) == expected_targets):
            raise ValueError(f"{subject_dir.name} failed its complete-repetition check.")
        datasets.append(build_bi2014a_subject_dataset(subject_dir, preprocessing=preprocessing))
        audit.append(
            {
                "subject": subject_dir.name,
                "n_flashes": int(len(recovered.flash_sample)),
                "n_targets": int(np.count_nonzero(recovered.target_label == 2)),
                "n_nontargets": int(np.count_nonzero(recovered.target_label == 1)),
                "n_repetitions": recovered.n_repetitions,
                "n_selections": int(len(np.unique(recovered.selection_id))),
                "n_explicit_selection_boundaries": recovered.n_explicit_boundaries,
                "selection_boundary_rule": (
                    "target_pair_change_or_raw_session_100_or_level_restart_104"
                ),
                "n_epochs": datasets[-1].n_epochs,
                "dropped_tail_flashes": recovered.dropped_tail_flashes,
            }
        )
        print(json.dumps(audit[-1], ensure_ascii=False), flush=True)

    dataset = concatenate_epoch_datasets(
        datasets,
        name="BI2014a-candidate",
        provenance={
            "source": "bi2014a_raw_csv",
            "source_root": str(root.resolve()),
            "subject_dirs": [str(p.resolve()) for p in subject_dirs],
            "channels": list(BI2014A_CHANNELS),
            "candidate_grid": "6x6",
            "source_reference": "right earlobe",
            "source_sample_rate_hz": 512.0,
        },
    )
    assert_causal_p300_input_contract(dataset.preprocessing)
    output = save_epoch_dataset(args.output, dataset, compressed=not args.uncompressed)
    audit_path = output.with_suffix(".candidate_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepared] {output}", flush=True)
    print(f"[audit] {audit_path}", flush=True)


if __name__ == "__main__":
    main()
