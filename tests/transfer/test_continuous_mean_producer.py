"""A valid cancelling ridge fit must publish one consistent QR posterior."""
from dataclasses import replace

import numpy as np
import pytest

from tests.transfer.test_ensemble_fit import selections
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


@pytest.mark.parametrize("noise_order", [0, None])
def test_stationary_fit_owns_the_same_mean_factor_and_saved_response(tmp_path, noise_order):
    groups = selections()
    copied = replace(groups[0], view=replace(groups[0].view,
        record_id="independent-copied-waveform", decision_id="copied-selection",
        samples=groups[0].view.samples * 1.01, raw_scope=(3400, 3951), available_sample=3951))
    actual = (groups[3], copied)
    model = ContinuousDecisionModel.fit([
        CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
        for group in actual], sfreq=256., channels=("Fz", "Cz", "Pz"),
        candidate_ids=("a", "b"), operator_identity="native_unfiltered_volts/1", noise_order=noise_order,
        dictionary=ResponseDictionary.compile(ResponseDictionarySpec("gaussian", 1.2, 2), sample_rate_hz=256.))
    state = model.posterior_state
    assert model.provenance["kernel_prior_fit_statistics"]["method"] == "all_C_actual_noise_per_record_diffuse_QR_response"
    path = tmp_path / "owned-posterior.npz"
    model.save(path)
    loaded = ContinuousDecisionModel.load(path)
    np.testing.assert_array_equal(loaded.kernel_mean, model.kernel_mean)
    for key in ("precision_root", "response", "orthogonal"):
        np.testing.assert_array_equal(loaded.posterior_state[key], state[key])
    values = np.random.default_rng(404).normal(0., 1e-6, (551, 3))
    settings = {"sfreq": 256., "channels": model.channels, "candidate_ids": model.candidate_ids,
                "operator_identity": model.operator_identity, "record_id": "new-software-decision"}
    before = model.score(values, groups[0].view.onsets, groups[0].view.decision.membership, **settings)
    after = loaded.score(values, groups[0].view.onsets, groups[0].view.decision.membership, **settings)
    np.testing.assert_array_equal(before.log_evidence, after.log_evidence)
    # Persistence remains tied to the real producer: a changed mean cannot be
    # credited by weakening the componentwise posterior relation.
    wrong = model.kernel_mean.copy()
    wrong[3, 0] += 1e-8
    with pytest.raises(ValueError, match="posterior mean"):
        replace(model, kernel_mean=wrong)
