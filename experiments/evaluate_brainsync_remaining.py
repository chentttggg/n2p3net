"""Zero-fit replay of saved controls on every requested recording outside C."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from experiments.analyze_brainsync_consistent import decision_records, render  # noqa: E402
from experiments.run_brainsync_real_matrix import prediction_dataset  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.target_recognition import (  # noqa: E402
    compare_common_scored_decisions,
    summarize_target_decisions,
)
from transfer.calibrated import load_calibrated_predictor  # noqa: E402


def unavailable_records(*, session_id, target, repetitions, reason, scheduled):
    if scheduled != 450 or repetitions not in (1, 5):
        raise ValueError("unscored requests require the verified complete R1/R5 BrainSync schedule")
    records = []
    for start in range(0, scheduled, 9 * repetitions):
        block, cycle = start // 45 + 1, start % 45 // 9 + 1
        identifier = f"{session_id}:R{repetitions}:block:{block}" + (f":cycle:{cycle}" if repetitions == 1 else "")
        records.append({"decision_id": identifier, "session_id": session_id, "block": block,
            "cycle": cycle if repetitions == 1 else None, "repetitions": repetitions,
            "prediction": None, "target": str(target), "status": "unscored", "reason": reason,
            "counts": [0]*9, "posterior": None})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads(args.prepared.read_text(encoding="utf-8"))
    if preparation["status"] != "prepared" or preparation["new_preprocessing_fits"] != 0:
        raise ValueError("remaining evaluation requires completed frozen preprocessing")
    result = json.loads((run / "analysis_local_20261101/results.json").read_text())
    result.update(all_remaining_cells={}, all_remaining_pairing={}, new_model_fits=0,
        all_remaining_preparation_sha256=file_sha256(args.prepared), additional_prediction_evidence=[])
    decisions = {}
    for name, cell in preparation["cells"].items():
        for family in ("xdawn_erp", "waveform_lda"):
            fit_folder = run / "analytic_controls_nominal_v1" / name / family
            fit = json.loads((fit_folder / "result.json").read_text())
            if file_sha256(fit_folder / "model.pt") != fit["model_sha256"]:
                raise ValueError("saved control model changed after the original fit")
            predictor = load_calibrated_predictor(fit_folder / "model.pt", device="cpu")
            folder = output / "predictions" / name / family
            folder.mkdir(parents=True)
            grouped = {label: defaultdict(list) for label in ("all_remaining", "fixed_future_test", "remaining_retrospective")}
            for row in cell["rows"]:
                if row["role"] == "C":
                    continue
                region = "fixed_future_test" if row["role"] == "fixed_future_test" else "remaining_retrospective"
                logits = dataset = None
                if row.get("status", "prepared") == "prepared":
                    if file_sha256(row["path"]) != row["sha256"]:
                        raise ValueError("prepared remaining recording changed")
                    dataset = load_epoch_dataset(row["path"], validation="attested")
                    logits = predictor.predict_logit(prediction_dataset(dataset))
                    path = folder / (row["session_id"] + "_predictions.npz")
                    np.savez(path, logits=logits, event_ids=dataset.event_timeline.event_ids,
                        evidence_indices=dataset.event_timeline.evidence_indices)
                    result["additional_prediction_evidence"].append({"cell": name, "family": family,
                        "session_id": row["session_id"], "role": row["role"], "input_sha256": row["sha256"],
                        "model_sha256": fit["model_sha256"], "prediction_sha256": file_sha256(path)})
                for repetitions in (1, 5):
                    if dataset is None:
                        records = unavailable_records(session_id=row["session_id"], target=row["target"],
                            repetitions=repetitions, reason=row["reason"], scheduled=row["scheduled"])
                    else:
                        records, _ = decision_records(dataset, logits, session_id=row["session_id"],
                            revealed_target=row["target"], repetitions=repetitions)
                    grouped[region][f"R{repetitions}"].extend(records)
                    grouped["all_remaining"][f"R{repetitions}"].extend(records)
            key = name + "_" + family
            decisions[key] = {region: dict(levels) for region, levels in grouped.items()}
            result["all_remaining_cells"][key] = {region: {level: summarize_target_decisions(records)
                for level, records in levels.items()} for region, levels in grouped.items()}
            # The expanded test roster must preserve the original fixed-test outputs exactly.
            if result["all_remaining_cells"][key]["fixed_future_test"] != result["cells"][key]:
                raise RuntimeError("expanding the retrospective test roster changed the original fixed-test result")
            print(name, family, "all outside-C recordings scored without fitting", flush=True)
    for dose in (1, 3, 5, 10):
        for family in ("xdawn_erp", "waveform_lda"):
            left, right = (decisions[f"{p}_K{dose}_{family}"]["all_remaining"] for p in ("basic", "QC"))
            result["all_remaining_pairing"][f"K{dose}_{family}"] = {
                level: compare_common_scored_decisions(left[level], right[level]) for level in ("R1", "R5")}
    atomic_write_json(output / "results.json", result)
    atomic_write_json(output / "decisions.json", decisions)
    render(output / "index.html", result)


if __name__ == "__main__":
    main()
