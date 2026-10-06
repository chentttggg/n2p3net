import json

import numpy as np

from experiments.run_operator_decision_study import main
from transfer.continuous_decision import ContinuousDecisionModel


def test_real_continuous_cli_consumes_owned_physical_dictionary_and_replays(tmp_path):
    rng = np.random.default_rng(5)
    records = []
    for index in range(3):
        path = f"C{index}.npz"
        np.savez(tmp_path/path, samples=rng.normal(size=(172, 2)), onsets=np.array([40, 80, 120]), target_mask=np.array([0, 1, index % 2]))
        records.append({"path": path, "record_id": f"C{index}"})
    plan = {"sfreq": 64., "channels": ["a", "b"], "candidate_ids": ["one", "two"],
            "operator_identity": "native-fixture", "fit_scope": "target_calibration", "records": records,
            "dictionary": {"family": "cosine", "response_seconds": .5, "n_basis": 5, "endpoint": "hard_zero"},
            "noise_order": 1, "conditioning_samples": 32, "noise_estimator": "burg"}
    (tmp_path/"plan.json").write_text(json.dumps(plan), encoding="utf-8")
    main(["fit", "--plan", str(tmp_path/"plan.json"), "--output", str(tmp_path/"fit")])
    model = ContinuousDecisionModel.load(tmp_path/"fit/model.npz")
    assert model.dictionary.matrix.shape == (32, 3)
    assert model.dictionary_geometry["capacity_changed_by_hard_endpoint"] is True
    np.savez(tmp_path/"held.npz", samples=rng.normal(size=(172, 2)), onsets=np.array([40, 80, 120]), membership=np.array([[1, 0], [0, 1], [1, 0]]))
    metadata = {"sfreq": 64., "channels": ["a", "b"], "candidate_ids": ["one", "two"],
                "operator_identity": "native-fixture", "record_id": "held"}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    main(["predict", "--model", str(tmp_path/"fit/model.npz"), "--input", str(tmp_path/"held.npz"),
          "--metadata", str(tmp_path/"metadata.json"), "--output", str(tmp_path/"prediction")])
    report = json.loads((tmp_path/"prediction/prediction.json").read_text())
    assert report["conditioning_sample_support"] == [0, 32]
    assert report["score_sample_support"] == [32, 172]
