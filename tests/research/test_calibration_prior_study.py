"""Attributable prior panels traverse the actual MAT fit/save/score runner."""

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.run_candidate_ensemble_study import _calibration_prior_panel, run_study
from tests.research.test_candidate_ensemble_study import study_plan
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.kernel_prior import KernelPriorSpec


def prior_panel_programs():
    priors = [("ridge1", KernelPriorSpec.isotropic_ridge(1.))]
    for g in (1., 9., 99.):
        priors.extend([(f"g{g:g}", KernelPriorSpec.calibration_g(g)),
                       (f"trace{g:g}", KernelPriorSpec.calibration_trace_isotropic(g))])
    return tuple(CalibrationExpertProgram(f"ar{order}_{estimator}_{name}", "continuous_ar",
        noise_order=order, conditioning_samples=32, noise_estimator=estimator, kernel_prior=prior)
        for order, estimator in ((0, "burg"), (8, "burg"), (8, "conditional_profile"), (32, "conditional_profile"))
        for name, prior in priors)


def test_prior_panel_declares_shape_and_strength_contrasts_on_same_c_information():
    panel = _calibration_prior_panel(prior_panel_programs())
    assert panel["noise_programs"] == [[0, "burg"], [8, "burg"], [8, "conditional_profile"], [32, "conditional_profile"]]
    assert panel["variance_multipliers"] == [1., 9., 99.]
    assert panel["reference_information"] == "all_authorized_C_AR0_per_record_diffuse_affine"
    assert panel["energy_matching"] == "tr(G H_g)=k*g=tr(G H_isotropic); lambda_C=tr(G)/(k*g)"
    assert len(panel["shape_contrasts"]) == 12
    assert all(row["g"] in (1., 9., 99.) for row in panel["shape_contrasts"])


@pytest.mark.parametrize("change", ["missing", "duplicate_cell", "prefix", "basis", "response", "ridge", "g", "noise"])
def test_prior_attribution_rejects_incomplete_or_confounded_panels(change):
    experts = list(prior_panel_programs())
    if change == "missing":
        experts.pop()
    elif change == "duplicate_cell":
        experts[-1] = replace(experts[-2], expert_id="duplicate")
    elif change == "prefix":
        experts[0] = replace(experts[0], conditioning_samples=33)
    elif change == "basis":
        experts[0] = replace(experts[0], n_basis=9)
    elif change == "response":
        experts[0] = replace(experts[0], response_seconds=1.)
    elif change == "ridge":
        experts[0] = replace(experts[0], kernel_prior=KernelPriorSpec.isotropic_ridge(2.), ridge=None)
    elif change == "g":
        experts[1] = replace(experts[1], kernel_prior=KernelPriorSpec.calibration_g(2.))
    else:
        experts[-1] = replace(experts[-1], noise_order=16)
    with pytest.raises(ValueError, match="prior panel"):
        _calibration_prior_panel(experts)


def test_mat_g_and_trace_control_have_actual_equal_reference_energy_and_exact_replay(tmp_path):
    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    experts = [CalibrationExpertProgram(name, "continuous_ar", noise_order=0, n_basis=2,
        conditioning_samples=32, kernel_prior=prior) for name, prior in
        (("g9", KernelPriorSpec.calibration_g(9.)),
         ("trace9", KernelPriorSpec.calibration_trace_isotropic(9.)))]
    plan.update(experts=[row.record() for row in experts],
        mixtures=[EnsembleProgram.equal_weights(row.expert_id, (row.expert_id,)).record() for row in experts],
        baseline_mixture="trace9")
    path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "paired_prior"
    result = run_study(path, output)
    assert result["status"] == "completed"
    assert result["contrasts"]["g9"]["paired_probability_count"] == 2
    from transfer.continuous_decision import ContinuousDecisionModel

    participant = result["participants"][0]
    assert len(participant["fits"]) == len(participant["artifact_receipts"]) == 4
    assert not participant["failed_fit_receipts"] and not participant["inflight_fits"]
    assert all(row["fit_origin"]["physical_successful_fits"] == 1 for row in participant["fits"])
    actor_name = f"participant_{participant['record_index']:03d}"
    directory = output / "participant_groups" / actor_name / actor_name
    models = {(receipt["request"]["fold"], receipt["request"]["expert_programs"][0]["expert_id"]):
        ContinuousDecisionModel.load(directory / next(file["path"] for file in receipt["files"]
            if file["path"].endswith(".npz"))) for receipt in participant["artifact_receipts"]}
    assert set(models) == {(fold, expert.expert_id) for fold in range(2) for expert in experts}
    for fold in range(2):
        priors = [models[fold, expert.expert_id].kernel_prior for expert in experts]
        assert priors[0].provenance["records"] == priors[1].provenance["records"]
        np.testing.assert_array_equal(priors[0].reference_root, priors[1].reference_root)
        G = priors[0].reference_root.T @ priors[0].reference_root
        assert G.shape == (4, 4)  # Two basis columns for both common and target responses.
        np.testing.assert_allclose([np.trace(G @ prior.row_covariance) for prior in priors], [36., 36.], rtol=2e-12)
    for row in result["participants"][0]["predictions"]:
        assert row["saved_replay_exact"] and row["training_labels"] > 0
        assert row["continuous_score_support"][row["mixture"]]["n_scored_samples"] > 0
