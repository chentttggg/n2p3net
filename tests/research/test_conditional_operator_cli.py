import json

import numpy as np
import pytest

from experiments.run_operator_decision_study import main


def setup(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.arange(80) % 2
    raw = rng.normal(size=(80, 2)) + labels[:, None]*.1
    features = .7*raw[:, :1]+.4*labels[:, None]+rng.normal(size=(80, 1))*.5
    means_raw = np.stack([raw[labels == q].mean(axis=0) for q in (0, 1)])
    means_state = np.stack([features[labels == q].mean(axis=0) for q in (0, 1)])
    np.savez(tmp_path/"draws.npz", raw_residual_draws=raw-means_raw[labels], state_draws=features-means_state[labels])
    noise_plan = {"input": "draws.npz", "fit_scope": "target_calibration", "scope_id": "C",
                  "draw_ids": [f"C-{i}" for i in range(80)], "shrinkage": 0.}
    (tmp_path/"noise_plan.json").write_text(json.dumps(noise_plan), encoding="utf-8")
    main(["noise-fit", "--plan", str(tmp_path/"noise_plan.json"), "--output", str(tmp_path/"noise")])
    np.savez(tmp_path/"cal.npz", features=features, labels=labels, conditioning_features=raw)
    np.savez(tmp_path/"operators.npz", A=np.zeros((1, 2)), B=np.eye(1))
    support = {"conditioning_sample_support": [0, 2], "response_sample_support": [2, 3], "available_samples": 3}
    plan = {"input": "cal.npz", "noise_model": "noise/noise.npz", "operators": "operators.npz",
            "scope_id": "C", "calibration_ids": noise_plan["draw_ids"], "input_identity": "observed-continuation",
            "observation_support": support, "ridge_relative": 0.}
    (tmp_path/"fit_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return raw, features, support


def test_real_operator_cli_uses_conditional_owner_saves_replays_and_reports_actual_support(tmp_path):
    raw, features, support = setup(tmp_path)
    main(["operator-lda-fit", "--plan", str(tmp_path/"fit_plan.json"), "--output", str(tmp_path/"fit")])
    fit = json.loads((tmp_path/"fit/fit.json").read_text())
    assert fit["fit"]["conditioning_program"] == "conditional_tail_with_fixed_class_prior"
    assert fit["scored_sample_support"] == [2, 3]
    assert fit["conditioning_sample_support"] == [0, 2]
    assert fit["reload_score_equal"] is True
    np.savez(tmp_path/"query.npz", features=features[:5], observed_raw=raw[:5])
    metadata = {"input_identity": "observed-continuation", "observation_support": support}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    main(["operator-lda-predict", "--input", str(tmp_path/"query.npz"), "--model", str(tmp_path/"fit/model.npz"),
          "--metadata", str(tmp_path/"metadata.json"), "--output", str(tmp_path/"predict")])
    prediction = json.loads((tmp_path/"predict/prediction.json").read_text())
    assert prediction["raw_likelihood_multiplied"] is False
    assert len(prediction["logits"]) == 5
    assert prediction["scored_sample_support"] == [2, 3]


def test_conditional_cli_rejects_held_labels_or_oracle_states_and_wrong_physical_support(tmp_path):
    raw, features, support = setup(tmp_path)
    main(["operator-lda-fit", "--plan", str(tmp_path/"fit_plan.json"), "--output", str(tmp_path/"fit")])
    np.savez(tmp_path/"query.npz", features=features[:5], observed_raw=raw[:5], state_draws=np.ones((5, 1)))
    metadata = {"input_identity": "observed-continuation", "observation_support": support}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    args = ["operator-lda-predict", "--input", str(tmp_path/"query.npz"), "--model", str(tmp_path/"fit/model.npz"),
            "--metadata", str(tmp_path/"metadata.json"), "--output", str(tmp_path/"oracle_rejected")]
    with pytest.raises(ValueError, match="only|oracle"):
        main(args)
    assert json.loads((tmp_path/"oracle_rejected/execution.json").read_text())["status"] == "failed"
    np.savez(tmp_path/"query.npz", features=features[:5], observed_raw=raw[:5])
    metadata["observation_support"]["response_sample_support"] = [1, 3]
    (tmp_path/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    args[-1] = str(tmp_path/"wrong_support")
    with pytest.raises(ValueError, match="support"):
        main(args)


def test_noise_condition_real_cli_returns_full_candidate_state_means_and_shared_schur(tmp_path):
    raw, _, _ = setup(tmp_path)
    np.savez(tmp_path/"condition.npz", raw_values=raw[:2], raw_means=np.zeros((2, 2)), state_means=np.array([[0.], [1.]]))
    main(["noise-condition", "--input", str(tmp_path/"condition.npz"), "--model", str(tmp_path/"noise/noise.npz"),
          "--output", str(tmp_path/"conditional")])
    with np.load(tmp_path/"conditional/conditioned.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["mean"][:, 1]-archive["mean"][:, 0], 1.)
        assert archive["covariance"].shape == (1, 1)


def test_real_noise_condition_cli_rejects_complex_and_leaves_failed_receipt(tmp_path):
    raw, _, _ = setup(tmp_path)
    np.savez(tmp_path/"complex.npz", raw_values=raw[:2].astype(complex)+1j,
             raw_means=np.zeros((2, 2)), state_means=np.array([[0.], [1.]]))
    with pytest.raises(ValueError, match="finite real"):
        main(["noise-condition", "--input", str(tmp_path/"complex.npz"), "--model", str(tmp_path/"noise/noise.npz"),
              "--output", str(tmp_path/"rejected")])
    assert json.loads((tmp_path/"rejected/execution.json").read_text())["status"] == "failed"
    assert not (tmp_path/"rejected/conditioned.npz").exists()
