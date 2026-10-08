import json

import numpy as np
import pytest

from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
)
from transfer.kernel_prior import KernelPriorSpec
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def inputs():
    rng = np.random.default_rng(616)
    records = [CalibrationRecord(f"C{i}", rng.normal(size=(172, 2)), np.array([40, 80, 120]), np.array([0, 1, i % 2]))
               for i in range(3)]
    return records


def fit(dictionary, prior=None, *, records=None):
    return ContinuousDecisionModel.fit(inputs() if records is None else records, sfreq=64., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-fixture", dictionary=dictionary,
        noise_order=1, conditioning_samples=32, kernel_prior=prior)


@pytest.mark.parametrize("prior", [None, KernelPriorSpec.calibration_g(4.), KernelPriorSpec.calibration_trace_isotropic(4.)])
def test_hard_dictionary_prior_pullback_and_true_capacity_match_independent_dense(prior, tmp_path):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    model = fit(dictionary, prior)
    geometry = model.dictionary_geometry
    ref = np.asarray(geometry["reference_information_root"])
    actual = np.asarray(geometry["fitted_information_root"])
    precision = np.eye(6) if prior is None else model.kernel_prior.precision_root.T @ model.kernel_prior.precision_root
    reference_h = np.linalg.solve(ref.T @ ref+precision, np.eye(6))
    np.testing.assert_allclose(geometry["df_ref"], np.trace(ref.T @ ref @ reference_h), atol=2e-13)
    np.testing.assert_allclose(geometry["df_fit"], np.trace(actual.T @ actual @ model.kernel_row_covariance), atol=2e-13)
    embedding = np.kron(np.eye(2), dictionary.constraint_map)
    ambient_precision = np.asarray(geometry["ambient_prior_precision"])
    np.testing.assert_allclose(embedding.T @ ambient_precision @ embedding, precision, atol=1e-12)
    assert geometry["capacity_changed_by_hard_endpoint"] is True
    assert geometry["ambient_dimensions"] == 10 and geometry["kernel_dimensions"] == 6
    path = tmp_path/"model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert "dictionary_fitted_information_root" not in archive.files
        assert "prior_fit_information_root" in archive.files
        if prior is not None:
            assert "dictionary_reference_information_root" not in archive.files
            assert "dictionary_original_reference_root" not in archive.files
    restored = ContinuousDecisionModel.load(path)
    samples = np.random.default_rng(2).normal(size=(172, 2))
    onsets, membership = np.array([40, 80, 120]), np.array([[1, 0], [0, 1], [1, 0]])
    common = dict(sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"), operator_identity="native-fixture", record_id="held")
    np.testing.assert_array_equal(model.score(samples, onsets, membership, **common).probabilities,
                                  restored.score(samples, onsets, membership, **common).probabilities)


def test_owned_dictionary_conflict_and_numeric_payload_mismatch_rejected(tmp_path):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    with pytest.raises(ValueError, match="exclusive|conflict"):
        ContinuousDecisionModel.fit(inputs(), sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
            operator_identity="native-fixture", dictionary=dictionary, basis=dictionary.matrix)
    model = fit(dictionary)
    path = tmp_path/"model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(values["metadata"]))
    assert metadata["schema"] == "n2p3_continuous_dictionary_model/2"
    values["dictionary_constraint_map"] = values["dictionary_constraint_map"].copy()
    values["dictionary_constraint_map"][0, 0] += .1
    np.savez_compressed(tmp_path/"wrong.npz", **values)
    with pytest.raises(ValueError, match="state|dictionary|constraint"):
        ContinuousDecisionModel.load(tmp_path/"wrong.npz")
