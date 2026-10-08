"""Exercise frozen numerical ingress and new priors through actual model APIs."""
import json

import numpy as np
import pytest

import transfer.response_dictionary as dictionary_owner
from tests.transfer.test_continuous_dictionary import fit, inputs
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
from transfer.kernel_prior import KernelPriorRankError, KernelPriorSpec
from transfer.noise_model import metadata_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def _rewrite_diagnostic(source, destination, value):
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata"]))
    metadata["dictionary_geometry"]["df_fit"] = value
    metadata["artifact_sha256"] = metadata_digest({key: item for key, item in metadata.items()
                                                 if key != "artifact_sha256"})
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(destination, **payload)


def test_model_retains_producer_diagnostic_roundoff_and_exact_prediction(tmp_path, monkeypatch):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    model = fit(dictionary)
    source, saved = tmp_path/"source.npz", tmp_path/"producer_roundoff.npz"
    model.save(source)
    diagnostic = float(np.nextafter(model.dictionary_geometry["df_fit"], np.inf))
    _rewrite_diagnostic(source, saved, diagnostic)

    def forbidden(*args, **kwargs):
        raise AssertionError("model load recompiled its actual dictionary")

    monkeypatch.setattr(ResponseDictionary, "compile", forbidden)
    monkeypatch.setattr(dictionary_owner, "compile_state", forbidden)
    restored = ContinuousDecisionModel.load(saved)
    assert restored.dictionary_geometry["df_fit"] == diagnostic
    np.testing.assert_array_equal(restored.basis, model.basis)
    group = inputs()[0]
    held_samples = np.random.default_rng(617).normal(size=group.samples.shape)
    member = np.array([[1, 0], [0, 1], [1, 0]])
    common = dict(sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
                  operator_identity="native-fixture", record_id="independent_held")
    np.testing.assert_array_equal(restored.score(held_samples, group.onsets, member, **common).probabilities,
                                  model.score(held_samples, group.onsets, member, **common).probabilities)


def test_resigned_capacity_claim_cannot_change_actual_fitted_dimension(tmp_path):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    model = fit(dictionary)
    source, altered = tmp_path/"source.npz", tmp_path/"altered.npz"
    model.save(source)
    _rewrite_diagnostic(source, altered, model.dictionary_geometry["df_fit"]+1.)
    with pytest.raises(ValueError, match="capacity diagnostic"):
        ContinuousDecisionModel.load(altered)


def test_completed_prior_fits_censored_rank_deficient_c_and_restores_full_held(tmp_path):
    rng = np.random.default_rng(616)
    records = [CalibrationRecord(f"short{i}", rng.normal(size=(64, 2)),
                                 np.array([40, 62]), np.array([0, 1]), right_censored=True)
               for i in range(3)]
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 8), sample_rate_hz=64.)
    arguments = dict(sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
                     operator_identity="native-fixture", dictionary=dictionary, noise_order=0,
                     conditioning_samples=32)
    with pytest.raises(KernelPriorRankError):
        ContinuousDecisionModel.fit(records, kernel_prior=KernelPriorSpec.calibration_g(4.), **arguments)
    model = ContinuousDecisionModel.fit(records,
        kernel_prior=KernelPriorSpec.calibration_completed_g(4., 1.), **arguments)
    assert model.kernel_prior.provenance["rank_ledger"]["rank"] < 16
    assert model.dictionary_geometry["prior_pullback"] == "restricted_precision_extension_zero_on_constraint_complement"
    filename = tmp_path/"completed.npz"
    model.save(filename)
    restored = ContinuousDecisionModel.load(filename)
    assert restored.kernel_prior.spec == model.kernel_prior.spec
    held = inputs()[0]
    membership = np.array([[1, 0], [0, 1], [1, 0]])
    common = dict(sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
                  operator_identity="native-fixture", record_id="new_held")
    before = model.score(held.samples, held.onsets, membership, **common)
    assert np.isfinite(before.log_evidence).all()
    np.testing.assert_array_equal(restored.score(held.samples, held.onsets, membership, **common).probabilities,
                                  before.probabilities)
