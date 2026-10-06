import json

import numpy as np
import pytest

from tests.transfer.test_continuous_dictionary import inputs
from transfer.continuous_decision import (
    _DICTIONARY_GEOMETRY_KEYS,
    ContinuousDecisionModel,
    _dictionary_geometry_from_roots,
)
from transfer.kernel_prior import KernelPriorSpec
from transfer.noise_model import array_digest, metadata_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


@pytest.mark.parametrize("order,prefix,prior", [(None, 0, None), (1, 32, None), (1, 32, KernelPriorSpec.calibration_g(4.))])
def test_recomputed_capacity_self_summary_cannot_change_information_with_fixed_actual_posterior(tmp_path, order, prefix, prior):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    model = ContinuousDecisionModel.fit(inputs(), dictionary=dictionary, sfreq=64., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-fixture", noise_order=order,
        conditioning_samples=prefix, kernel_prior=prior)
    path = tmp_path/"model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    key = "dictionary_fitted_information_root" if "dictionary_fitted_information_root" in arrays else "prior_fit_information_root"
    arrays[key] *= 2
    roots = {**model.dictionary_geometry, "fitted_information_root": arrays[key]}
    prior_root = np.eye(6) if prior is None else model.kernel_prior.precision_root
    geometry = _dictionary_geometry_from_roots(dictionary, roots, prior_root, roots["n_rows"], roots["n_records"])
    metadata = json.loads(str(arrays["metadata"]))
    metadata["dictionary_geometry"] = {k: v for k, v in geometry.items() if k not in _DICTIONARY_GEOMETRY_KEYS}
    if key == "prior_fit_information_root":
        metadata["provenance"]["kernel_prior_fit_statistics"]["state_sha256"] = array_digest(arrays[key], arrays["prior_fit_score"])
    metadata["state_sha256"] = array_digest(*(arrays[k] for k in model._state_arrays()))
    metadata["artifact_sha256"] = metadata_digest({k: v for k, v in metadata.items() if k != "artifact_sha256"})
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    altered = tmp_path/"altered.npz"
    np.savez_compressed(altered, **arrays)
    with pytest.raises(ValueError, match="posterior|actual|capacity"):
        ContinuousDecisionModel.load(altered)
