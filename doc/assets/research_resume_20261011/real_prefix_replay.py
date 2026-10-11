"""One native A01 input probe of the saved prefix/caller contract.

This is a public-data software check.  It does not fit a source checkpoint,
select a recipe from future truth, or admit a full public efficacy matrix.
Run it with the repository venv and an explicit PYTHONPATH to frozen source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from experiments.run_candidate_ensemble_study import run_study
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.ensemble_views import SelectionEpochProfile
from transfer.recipes import erp_target_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mat = args.mat.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mat_digest = hashlib.sha256(mat.read_bytes()).hexdigest()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    profile = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS,
        reference=BNCI2014_008_SOURCE_REFERENCE)
    experts = (
        CalibrationExpertProgram("raw", "continuous_ar", noise_order=0, n_basis=2),
        CalibrationExpertProgram("toeplitz", "trial", erp_target_recipe("toeplitz_lda", bin_width=16), profile),
    )
    mixtures = (EnsembleProgram.equal_weights("trial", ("toeplitz",)),
                EnsembleProgram.equal_weights("strict", ("raw", "toeplitz")))
    plan = {
        "schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[0, 20260918]],
        "records": [{"path": str(mat), "sha256": mat_digest}], "budgets": [200, 800],
        "experts": [expert.record() for expert in experts],
        "mixtures": [mixture.record() for mixture in mixtures], "baseline_mixture": "trial",
        "authorization_preprocessing": spec.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "source_checkpoints": {}, "source_seed_catalog": [],
        "calibration_strategies": ["fixed", "grouped_temperature"],
        "temperature_expert_ids": ["toeplitz"],
    }
    plan_path = output / "input-plan.json"
    atomic_write_json(plan_path, plan)
    report = run_study(plan_path, output / "run", execution="inline")
    assert report["status"] == "completed", report["status"]
    assert hashlib.sha256(mat.read_bytes()).hexdigest() == mat_digest
    package_rows = []
    participant = report["participants"][0]
    directory = output / "run" / "participant_000"
    for package in sorted(directory.glob("source0_target20260918_K*")):
        payload = json.loads((package / "prefix_inner_replay.json").read_bytes())
        record = payload["record"]
        predictions = [row for row in participant["predictions"] if row["K"] == record["actual_K"]]
        assert predictions and all(row["prefix_replay_receipt_sha256"] == payload["record_sha256"]
                                   for row in predictions)
        package_rows.append({
            "actual_K": record["actual_K"], "requested_Ks": record["requested_Ks"],
            "raw_labels_used": record["prefix_input"]["label_use"]["raw"]["used_labels"],
            "trial_labels_used": {key: row["used_labels"] for key, row in
                                  record["prefix_input"]["label_use"]["trial"].items()},
            "strategies": {name: row["status"] for name, row in record["programs"].items()},
            "inner_producers": len(record["inner_producers"]),
            "captured_future_rows": len(record["future_score_rows"]),
            "prediction_rows": len(predictions),
            "probability_available_rows": sum(row["evidence"]["log_probability"] is not None for row in predictions),
            "prefix_receipt_sha256": payload["record_sha256"],
            "prefix_receipt_file_sha256": hashlib.sha256((package / "prefix_inner_replay.json").read_bytes()).hexdigest(),
        })
    assert len(package_rows) == 2
    binding = {
        "schema": "n2p3_native_prefix_replay_check/1", "status": "completed",
        "scope": "one public MAT software prefix/caller check; no efficacy acceptance",
        "mat_sha256": mat_digest, "source_fit_calls": 0,
        "seed_pair": plan["seed_pairs"][0], "plan_sha256": semantic_sha256(plan),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executed_source_file_sha256": hashlib.sha256((output / "run" / "executed_source.json").read_bytes()).hexdigest(),
        "requested_decisions": len(participant["requested"]),
        "prediction_rows": len(participant["predictions"]), "packages": package_rows,
        "numerical_replay_and_parent_admission": "required by the actual run_study caller",
        "future_truth_role": "external reporting only; fixed declared programmes",
    }
    atomic_write_json(output / "binding.json", binding)
    print(json.dumps(binding, ensure_ascii=False))


if __name__ == "__main__":
    main()
