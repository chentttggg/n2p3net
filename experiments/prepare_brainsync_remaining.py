"""Apply the frozen C-only preprocessing to every recording outside C."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.brainsync import (  # noqa: E402
    BrainSyncNoUsableEvidenceError,
    IndependentRecordingScope,
    load_brainsync_session,
)
from data.continuous_qc import load_continuous_qc  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.epochs import load_preprocessing_spec, save_epoch_dataset  # noqa: E402
from data.ocular import load_eog_regression  # noqa: E402
from data.source_records import SIX, file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((ROOT / "experiments/runs/brainsync_matrix_20261031/plan.json").read_text())
    quality = json.loads((run / "confirmed_EOG_inputs/quality_manifest.json").read_text())
    roots = {row["directory_id"]: Path(row["input_root"]) for row in quality["sessions"]}
    truth = {row["session_id"]: row["target"] for row in previous["rows"]}
    parent = run / "target_nominal_v1/prepared.json"
    prepared = json.loads(parent.read_text())
    result = copy.deepcopy(prepared)
    result.update(status="preparing", parent_manifest=str(parent), parent_manifest_sha256=file_sha256(parent),
        new_preprocessing_fits=0, evaluation_scope="all P001 recordings outside current C; fixed future test separately labelled")
    spec = load_preprocessing_spec(ROOT / "configs/brainsync_bidirectional_preprocessing.json")
    for name, cell in result["cells"].items():
        folder = output / name
        folder.mkdir()
        remaining = [s for s in previous["calibration_order"] if s not in cell["calibration_sessions"]]
        scope = IndependentRecordingScope(tuple(cell["calibration_sessions"]), tuple(cell["test_sessions"] + remaining))
        cell["remaining_test_sessions"] = remaining
        cell["all_evaluation_sessions"] = cell["test_sessions"] + remaining
        qc = ocular = None
        if name.startswith("QC_"):
            fitted = run / "target_group_QC_v1" / name
            qc = load_continuous_qc(fitted / "continuous.json")
            ocular = load_eog_regression(fitted / "ocular.json")
        for session in remaining:
            row = {"session_id": session, "target": truth[session], "role": "remaining_independent_recordings_retrospective",
                "status": "prepared"}
            try:
                data = load_brainsync_session(roots[session], preprocessing=spec,
                    channels=SIX if ocular is not None else None, ocular_model=ocular,
                    continuous_qc_model=qc, recording_scope=scope if ocular is not None else None)
                data = adapt_observed_montage(data, SIX, reference_channel="Cz" if ocular is None else None,
                    preserve_reference_label=ocular is not None)
                data.provenance["artifact_preprocessing_policy"] = cell["policy"]
                path = save_epoch_dataset(folder / (session + ".npz"), data, compressed=False, exclusive=True)
                row.update(path=str(path), sha256=file_sha256(path), n_epochs=data.n_epochs, scheduled=data.event_timeline.n_events)
            except BrainSyncNoUsableEvidenceError as error:
                ledger = folder / (session + "_unscored.json")
                atomic_write_json(ledger, {"session_id": error.session_id, "ledger": error.ledger,
                    "scheduled_trials": error.scheduled_trials, "reason": "no_usable_EEG_after_frozen_QC"})
                row.update(status="no_usable_evidence", scheduled=error.scheduled_trials, n_epochs=0,
                    reason="no_usable_EEG_after_frozen_QC", ledger=str(ledger))
            cell["rows"].append(row)
        print(name, "all remaining recordings processed", flush=True)
        atomic_write_json(output / "prepared.json", result)
    result["status"] = "prepared"
    atomic_write_json(output / "prepared.json", result)


if __name__ == "__main__":
    main()
