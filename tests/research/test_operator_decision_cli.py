import json

import numpy as np
import pytest

from experiments.run_operator_decision_study import main


def test_cli_calibration_model_and_truth_free_prediction(tmp_path):
    rng = np.random.default_rng(21)
    onsets = np.arange(20, 121, 20)
    labels = np.arange(len(onsets)) % 2
    basis = np.array([[0.], [1.], [2.], [1.], [0.]])
    np.save(tmp_path / "basis.npy", basis)
    records = []
    for i in range(3):
        samples = rng.normal(0, 0.2, (150, 1))
        for onset, target in zip(onsets, labels, strict=True):
            samples[onset:onset + 5] += (1 + target * 3) * basis
        np.savez(tmp_path / f"cal{i}.npz", samples=samples, onsets=onsets, target_mask=labels)
        records.append({"path": f"cal{i}.npz", "record_id": f"cal{i}"})
    plan = {"sfreq": 20., "channels": ["Cz"], "candidate_ids": ["a", "b"],
            "operator_identity": "declared-raw", "fit_scope": "target_calibration",
            "basis_path": "basis.npy", "records": records}
    (tmp_path / "fit.json").write_text(json.dumps(plan), encoding="utf-8")
    assert main(["fit", "--plan", str(tmp_path / "fit.json"), "--output", str(tmp_path / "fit")]) == 0
    assert json.loads((tmp_path / "fit/fit.json").read_text())["reload_state_equal"]
    samples = rng.normal(0, 0.2, (150, 1))
    for onset, target in zip(onsets, labels, strict=True):
        samples[onset:onset + 5] += (1 + target * 3) * basis
    np.savez(tmp_path / "test.npz", samples=samples, onsets=onsets,
             membership=np.eye(2, dtype=bool)[labels])
    metadata = {"sfreq": 20., "channels": ["Cz"], "record_id": "independent-test",
                "candidate_ids": ["a", "b"],
                "operator_identity": "declared-raw"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    args = ["predict", "--model", str(tmp_path / "fit/model.npz"), "--input", str(tmp_path / "test.npz"),
            "--metadata", str(tmp_path / "metadata.json"), "--output", str(tmp_path / "prediction")]
    assert main(args) == 0
    prediction = json.loads((tmp_path / "prediction/prediction.json").read_text())
    assert prediction["winner"] == "b"
    assert prediction["n_unique_samples"] == 150
    np.savez(tmp_path / "test.npz", samples=samples, onsets=onsets,
             membership=np.eye(2, dtype=bool)[labels], truth="b")
    args[-1] = str(tmp_path / "rejected")
    with pytest.raises(ValueError, match="truth stays in scorer"):
        main(args)


def test_cli_joint_noise_fit_and_projection(tmp_path):
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(30, 2))
    state = raw[:, :1] * 2
    np.savez(tmp_path / "draws.npz", raw_residual_draws=raw, state_draws=state)
    plan = {"input": "draws.npz", "fit_scope": "source_training", "scope_id": "source-A",
            "draw_ids": [str(i) for i in range(30)], "shrinkage": 0}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert main(["noise-fit", "--plan", str(tmp_path / "plan.json"), "--output", str(tmp_path / "fit")]) == 0
    np.savez(tmp_path / "operators.npz", A=np.array([[1., 0.]]), B=np.array([[-0.5]]))
    assert main(["noise-project", "--model", str(tmp_path / "fit/noise.npz"),
                 "--input", str(tmp_path / "operators.npz"), "--output", str(tmp_path / "project")]) == 0
    with np.load(tmp_path / "project/propagated.npz") as archive:
        assert abs(archive["covariance"][0, 0]) < 1e-14
    # Complete R1 slice has a nondegenerate output and fits only class means.
    np.savez(tmp_path / "operators.npz", A=np.eye(2), B=np.zeros((2, 1)))
    labels = np.arange(30) % 2
    features = rng.normal(size=(30, 2)) + labels[:, None] * 2
    np.savez(tmp_path / "calibration.npz", features=features, labels=labels)
    lda_plan = {"input": "calibration.npz", "noise_model": "fit/noise.npz",
                "operators": "operators.npz", "scope_id": "calibration",
                "input_identity": "raw-feature-v1", "calibration_ids": [str(i) for i in range(30)]}
    (tmp_path / "lda_plan.json").write_text(json.dumps(lda_plan), encoding="utf-8")
    assert main(["operator-lda-fit", "--plan", str(tmp_path / "lda_plan.json"),
                 "--output", str(tmp_path / "lda")]) == 0
    np.savez(tmp_path / "test_features.npz", features=np.array([[5., 5.], [-5., -5.]]))
    (tmp_path / "lda_metadata.json").write_text(json.dumps({"input_identity": "raw-feature-v1"}), encoding="utf-8")
    assert main(["operator-lda-predict", "--model", str(tmp_path / "lda/model.npz"),
                 "--input", str(tmp_path / "test_features.npz"), "--metadata", str(tmp_path / "lda_metadata.json"),
                 "--output", str(tmp_path / "lda_prediction")]) == 0
    logits = json.loads((tmp_path / "lda_prediction/prediction.json").read_text())["logits"]
    assert logits[0] > 0 > logits[1]


def test_cli_hashes_exact_consumed_input_and_plan_despite_path_replacement(tmp_path, monkeypatch):
    import hashlib

    from transfer.noise_model import JointNoiseModel, array_digest
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(10, 2))
    state = np.empty((10, 0))
    path, plan_path = tmp_path / "data.npz", tmp_path / "plan.json"
    np.savez(path, raw_residual_draws=raw, state_draws=state)
    plan = {"input": str(path), "scope_id": "cal", "fit_scope": "target_calibration",
            "draw_ids": list(map(str, range(10)))}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    expected_input = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_plan = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    original = JointNoiseModel.fit.__func__
    def replacing(cls, *args, **kwargs):
        fitted = original(cls, *args, **kwargs)
        np.savez(path, raw_residual_draws=raw * 9, state_draws=state)
        plan_path.write_text(json.dumps({**plan, "scope_id": "different"}), encoding="utf-8")
        return fitted
    monkeypatch.setattr(JointNoiseModel, "fit", classmethod(replacing))
    main(["noise-fit", "--plan", str(plan_path), "--output", str(tmp_path / "run")])
    report = json.loads((tmp_path / "run/fit.json").read_text())
    model = JointNoiseModel.load(tmp_path / "run/noise.npz")
    assert report["input_file_sha256"] == expected_input
    assert report["plan_sha256"] == expected_plan
    assert model.provenance["input_sha256"] == array_digest(raw, state)


def test_cli_prediction_keeps_consumed_model_input_and_metadata_identities(tmp_path, monkeypatch):
    import hashlib

    from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
    rng = np.random.default_rng(8)
    onsets = np.array([5, 15, 25])
    labels = np.array([0, 1, 0])
    model = ContinuousDecisionModel.fit([
        CalibrationRecord("cal", rng.normal(size=(40, 1)), onsets, labels)],
        sfreq=10., channels=("Cz",), candidate_ids=("a", "b"), operator_identity="raw",
        basis=np.ones((3, 1)))
    model_path, input_path, metadata_path = (tmp_path / name for name in ("model.npz", "input.npz", "metadata.json"))
    model.save(model_path)
    np.savez(input_path, samples=rng.normal(size=(40, 1)), onsets=onsets, membership=np.eye(2)[labels])
    metadata_path.write_text(json.dumps(dict(sfreq=10., channels=["Cz"], candidate_ids=["a", "b"],
        operator_identity="raw", record_id="heldout")), encoding="utf-8")
    original_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (
        ("model_sha256", model_path), ("input_file_sha256", input_path), ("metadata_sha256", metadata_path))}
    original_score = ContinuousDecisionModel.score
    def replace_sources(self, *args, **kwargs):
        result = original_score(self, *args, **kwargs)
        model_path.write_bytes(b"new unrelated model")
        input_path.write_bytes(b"new unrelated input")
        metadata_path.write_text("{}", encoding="utf-8")
        return result
    monkeypatch.setattr(ContinuousDecisionModel, "score", replace_sources)
    main(["predict", "--model", str(model_path), "--input", str(input_path),
          "--metadata", str(metadata_path), "--output", str(tmp_path / "prediction")])
    reported = json.loads((tmp_path / "prediction/prediction.json").read_text())
    assert all(reported[key] == expected for key, expected in original_hashes.items())


def test_cli_failed_fit_is_recorded_without_a_successful_model(tmp_path):
    data = np.ones((2, 1))  # Too few covariance draws.
    np.savez(tmp_path / "bad.npz", raw_residual_draws=data, state_draws=np.empty((2, 0)))
    plan = {"input": "bad.npz", "fit_scope": "target_calibration", "scope_id": "C", "draw_ids": ["a", "b"]}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="three paired"):
        main(["noise-fit", "--plan", str(tmp_path / "plan.json"), "--output", str(output)])
    status = json.loads((output / "execution.json").read_text())
    assert status["status"] == "failed" and status["error"]["type"] == "ValueError"
    assert not (output / "noise.npz").exists()
