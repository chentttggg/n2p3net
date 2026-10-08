import numpy as np
import pytest

from transfer.continuous_decision import smooth_erp_basis
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def test_portable_gaussian_has_the_same_physical_family_and_actual_default_grid():
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", 1.2, 8), sample_rate_hz=256.)
    # The new declared scalar compiler has its own exact numerical identity;
    # the legacy NumPy program remains an independent physical-family check.
    np.testing.assert_allclose(dictionary.matrix, smooth_erp_basis(256., 1.2, 8),
                               rtol=0, atol=4*np.finfo(float).eps)
    assert dictionary.metadata()["compiler"]["semantic_id"] == "n2p3_physical_dictionary_decimal96_binary64/1"
    assert dictionary.matrix[0, 0] == 1. and dictionary.matrix[-1, -1] == 1.
    np.testing.assert_array_equal(dictionary.times, np.arange(307)/256.)
    assert dictionary.actual_seconds == 307/256.
    assert dictionary.times[-1] == 306/256.
    assert dictionary.spec.response_seconds == 1.2
    with pytest.raises(ValueError):
        dictionary.matrix.setflags(write=True)


def test_hard_endpoint_restricts_the_same_ambient_cosine_and_owns_a_real_nullspace_map():
    free = ResponseDictionary.compile(ResponseDictionarySpec("cosine", 1.2, 8), sample_rate_hz=256.)
    hard = ResponseDictionary.compile(ResponseDictionarySpec("cosine", 1.2, 8, "hard_zero"), sample_rate_hz=256.)
    np.testing.assert_array_equal(free.ambient_matrix, hard.ambient_matrix)
    np.testing.assert_allclose(hard.matrix, free.matrix @ hard.constraint_map, atol=2e-15)
    np.testing.assert_allclose(hard.endpoint_rows @ hard.constraint_map, 0., atol=2e-15)
    np.testing.assert_allclose(hard.constraint_map.T @ hard.constraint_map, np.eye(6), atol=2e-15)
    assert hard.matrix.shape == (307, 6)
    assert hard.record()["endpoint_evaluation"] == "u=0_and_actual_L_over_sample_rate"
    # The right functional endpoint is outside the half-open observed grid.
    assert np.linalg.norm(hard.matrix[-1]) > 0


def test_dictionary_reconstructs_numeric_source_and_constraints_not_just_a_checksum():
    original = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.)
    replay = ResponseDictionary.from_record(original.record())
    np.testing.assert_array_equal(original.matrix, replay.matrix)
    payload = original.record()
    payload["constraint_map"][0][0] += .1
    with pytest.raises(ValueError, match="numerical|constraint|record"):
        ResponseDictionary.from_record(payload)


@pytest.mark.parametrize("spec", [ResponseDictionarySpec("cosine", .5, 2, "hard_zero"),
                                  ResponseDictionarySpec("gaussian", .5, 3, "hard_zero")])
def test_unsupported_or_empty_endpoint_space_is_rejected_without_hidden_column_drop(spec):
    with pytest.raises(ValueError, match="endpoint|empty|hard"):
        ResponseDictionary.compile(spec, sample_rate_hz=64.)
