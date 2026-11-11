"""Prepare independently recorded, byte-bound inputs for the BrainSync matrix."""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    IndependentRecordingScope,
    load_brainsync_session,
)
from data.continuous_qc import load_continuous_qc
from data.domain import adapt_observed_montage
from data.epochs import load_epoch_dataset, save_epoch_dataset, select_scheduled_events
from research.execution import atomic_write_json

RUN = ROOT / "experiments/runs/brainsync_matrix_20261031"
QUALITY = ROOT / "experiments/runs/brainsync_quality_20261030"
ANCHOR = "P001_20261028_224755_01"
EXCLUDED = "P001_20261030_144501_01"
HELD = ("P001_20261030_142755_01", "P001_20261030_152515_01")
CHANNELS = ("FZ", "P3", "PZ", "P4", "PO8", "OZ")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    entries = [
        e
        for e in json.loads((QUALITY / "manifest.json").read_bytes())["sessions"]
        if e["status"] == "completed" and e["directory_id"] != EXCLUDED
    ]
    ids = tuple(e["directory_id"] for e in entries)
    scope = IndependentRecordingScope(tuple(s for s in ids if s not in HELD), HELD)
    qc = load_continuous_qc(QUALITY / "implementation/accepted/continuous_reviewed_native7.json")
    rows = []
    for e in entries:
        session = e["directory_id"]
        original = Path(e["input_root"])
        manifest = json.loads((original / "session.json").read_bytes())
        entry = {
            "session_id": session,
            "original_root": str(original),
            "subject_id": manifest["experiment"]["subject_id"],
            "target": manifest["experiment"]["thought_digit"],
            "started_utc": manifest["started_utc"],
            "paths": {},
        }
        for pipeline in ("basic", "reviewed"):
            if pipeline == "reviewed" and "202609" in session:
                entry["paths"][pipeline] = entry["paths"]["basic"]
                continue
            output = RUN / "inputs" / session / f"{pipeline}.npz"
            if output.is_file():
                data = load_epoch_dataset(output, require_labels=True, validation="attested")
                entry["paths"][pipeline] = {
                    "path": output.relative_to(RUN).as_posix(),
                    "sha256": sha(output),
                    "n_epochs": data.n_epochs,
                    "scheduled": data.event_timeline.n_events,
                }
                print(session, pipeline, "verified existing", data.n_epochs, flush=True)
                continue
            data = load_brainsync_session(
                original,
                preprocessing=BRAIN_SYNC_PREPROCESSING,
                continuous_qc_model=qc if pipeline == "reviewed" else None,
                recording_scope=scope if pipeline == "reviewed" else None,
            )
            # Measured-reference projection requires an observed Cz. Preserve
            # every scheduled event when a reference-unavailable epoch is lost.
            if data.trial_channel_mask is not None:
                index = list(data.channel_names).index("CZ")
                usable = data.trial_channel_mask[:, index]
                if not usable.all():
                    evidence = data.event_timeline.evidence_indices
                    keep = np.zeros(len(evidence), dtype=bool)
                    observed = evidence >= 0
                    keep[observed] = usable[evidence[observed]]
                    data = select_scheduled_events(
                        data,
                        keep,
                        reason="measured_Cz_reference_unavailable",
                        preserve_schedule=True,
                    )
            data = adapt_observed_montage(data, CHANNELS, reference_channel="CZ")
            output.parent.mkdir(parents=True, exist_ok=True)
            save_epoch_dataset(output, data, compressed=False, exclusive=True)
            entry["paths"][pipeline] = {
                "path": output.relative_to(RUN).as_posix(),
                "sha256": sha(output),
                "n_epochs": data.n_epochs,
                "scheduled": data.event_timeline.n_events,
            }
            print(session, pipeline, data.n_epochs, "/", data.event_timeline.n_events, flush=True)
        rows.append(entry)
    pool = [
        e["session_id"] for e in rows if e["subject_id"] == "P001" and e["session_id"] not in HELD
    ]
    order = [ANCHOR] + [s for s in pool if s != ANCHOR]
    assert len(order) == 10 and not set(order) & set(HELD)
    plan = {
        "schema": "n2p3_brainsync_real_matrix/1",
        "status": "prepared_before_fitting",
        "same_person_authority": "User confirmed same person; original P003 kept separate, zero-shot only.",
        "calibration_order": order,
        "calibration_session_counts": [1, 2, 3, 5, 8, 10],
        "test_sessions": list(HELD),
        "zero_supervised_calibration_sessions": ids,
        "target_seeds": [20261031, 20261101, 20261102],
        "source_seed": 20260918,
        "randomness_scope": "target_only_fixed_pretrained_source",
        "epochs": 90,
        "batch_size": 512,
        "learning_rate": 0.001,
        "fixed_temperature": 1.0,
        "pool_weights": "equal_arithmetic",
        "model_selection": "none; fixed recipes and no held-score-driven selection",
        "primary_metric": "strict_complete_45_flash_nine_choice_hit_over_all_20_requested_blocks",
        "secondary_metric": "partial_evidence_nine_choice_diagnostic_and_binary_AUC",
        "uncertainty": "one person; 2 correlated recording clusters, descriptive development",
        "data_role": "previously inspected retrospective development, not pristine confirmation",
        "excluded_session": EXCLUDED,
        "qc_reference_mandatory_in_every_nonzero_C": ANCHOR,
        "QC_fit_supervised_labels": False,
        "QC_reference_counts_toward_C": True,
        "old_reviewed_QC_support": "frozen old backend does not implement schema /5; retain incompatibility",
        "panels": ["old_basic", "current_basic", "current_reviewed"],
        "current_trial_experts": ["nn_source", "nn_scratch", "waveform_lda", "xdawn_erp"],
        "current_raw_experts": ["continuous_ar0", "continuous_ar8"],
        "rows": rows,
    }
    atomic_write_json(RUN / "plan.json", plan)


if __name__ == "__main__":
    main()
