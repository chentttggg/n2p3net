"""Actual spawned MAT fits must return artifact records across the process pool."""
import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from experiments.run_candidate_ensemble_study import run_study
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram


def test_two_spawned_participants_return_saved_records_without_pickling_frozen_state(tmp_path):
    records = []
    for index in range(2):
        path = tmp_path / f"A{index + 1:02d}.mat"
        fields = bnci_mat_fields()
        fields["X"] = np.random.default_rng(48 + index).normal(0, 4, fields["X"].shape).astype(np.float32)
        savemat(path, {"data": fields})
        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    expert = CalibrationExpertProgram("ar0", "continuous_ar", noise_order=0, n_basis=2,
                                      conditioning_samples=32)
    mixture = EnsembleProgram.equal_weights("ar0", ("ar0",))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
            "records": records, "authorized_labels": 800, "group_count": 2,
            "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2,
            "experts": [expert.record()], "mixtures": [mixture.record()], "baseline_mixture": "ar0",
            "seeds": [23], "device": "cpu", "workers": 2, "cpu_threads_per_worker": 1,
            "calibration_strategy": "fixed", "source_checkpoints": {}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "results"
    report = run_study(plan_path, output)
    assert report["status"] == "completed"
    assert len(report["participants"]) == 2
    persisted = json.loads((output / "result.json").read_bytes())
    assert report["participants"] == persisted["participants"]
    for index in range(2):
        participant = json.loads((output / f"participant_{index:03d}" / "result.json").read_bytes())
        assert len(participant["predictions"]) == 2
        assert all(cell["saved_replay_exact"] for cell in participant["predictions"])
        assert participant in report["participants"]


def test_spawned_contract_failure_stops_pending_participants(tmp_path):
    # The first two bad physical identities exercise actual worker ingress.
    # Later records must never be dispatched after that contract failure.
    records = []
    for index in range(4):
        path = tmp_path / f"A{index + 1:02d}.mat"
        if index < 2:
            savemat(path, {"data": bnci_mat_fields()})
        records.append({"path": str(path), "sha256": f"{index:064x}"})
    expert = CalibrationExpertProgram("ar0", "continuous_ar", noise_order=0, n_basis=2, conditioning_samples=32)
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
        "records": records, "authorized_labels": 800, "group_count": 2,
        "observation_response_seconds": 308/256, "pre_roll_seconds": .2,
        "experts": [expert.record()], "mixtures": [EnsembleProgram.equal_weights("ar0", ("ar0",)).record()],
        "baseline_mixture": "ar0", "seeds": [23], "device": "cpu", "workers": 2,
        "cpu_threads_per_worker": 1, "calibration_strategy": "fixed", "source_checkpoints": {}}
    path = tmp_path / "failed_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "failed_matrix"
    with pytest.raises((ValueError, RuntimeError)):
        run_study(path, output)
    assert json.loads((output / "progress.json").read_bytes())["status"] == "failed"
    assert not (output / "participant_002").exists() and not (output / "participant_003").exists()
    assert not (output / "result.json").exists()


def test_parent_interrupt_stops_workers_and_persists_failed_status(tmp_path, monkeypatch):
    import experiments.run_candidate_ensemble_study as runner
    from tests.research.test_candidate_ensemble_study import study_plan

    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    plan["workers"] = 2
    path.write_text(json.dumps(plan), encoding="utf-8")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "wait", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.run_study(path, output)
    assert json.loads((output / "progress.json").read_bytes())["status"] == "failed"
    assert not (output / "result.json").exists()
