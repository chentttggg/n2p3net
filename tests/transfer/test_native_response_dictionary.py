"""Source-native byte anchors, exact off-grid sites and package admission."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from research.joint_response_scale import NumericalResolutionError
from research.latent_response_reference import ResponseState, analytic_basis, selection_design
from transfer.continuous_decision import smooth_erp_basis
from transfer.dictionary_artifact import DictionaryArtifactError
from transfer.native_response_dictionary import NativeResponseDictionary, NativeResponseEvaluation
from transfer.noise_model import array_digest


def _dictionary(matrix=None, *, rate=8., terminal=None):
    matrix = np.array([[1.], [2.], [4.]]) if matrix is None else matrix
    terminal = np.zeros(matrix.shape[1]) if terminal is None else terminal
    return NativeResponseDictionary.from_source_matrix(matrix, sample_rate_hz=rate,
        source_basis_sha256=array_digest(matrix), terminal_values=terminal)


def test_original_legacy_basis_is_a_byte_anchor_and_not_a_recompiled_family(tmp_path):
    source = smooth_erp_basis(256., 1.2, 8)
    dictionary = _dictionary(source, rate=256.)
    assert dictionary.matrix.tobytes() == source.tobytes()
    assert dictionary.evaluate(np.arange(307), coordinate="sample").matrix.tobytes() == source.tobytes()
    assert dictionary.spec.family == "source_native_anchored_linear"
    dictionary.save(tmp_path/"dictionary.json")
    restored = NativeResponseDictionary.load(tmp_path/"dictionary.json")
    assert restored.record() == dictionary.record()
    assert restored.matrix.tobytes() == source.tobytes()
    with pytest.raises(ValueError):
        restored.matrix.setflags(write=True)
    with pytest.raises(NumericalResolutionError, match="unsupported"):
        analytic_basis(restored, [.01], derivative=1)


def test_exact_rational_sites_support_and_explicit_terminal_are_distinct():
    dictionary = _dictionary(terminal=np.array([2.]))
    inner = Fraction(3)-Fraction(1, 2**70)
    assert float(inner) == 3.
    values = dictionary.evaluate([Fraction(1, 3), inner, Fraction(3), Fraction(4)],
        coordinate="sample", support="half_open")
    assert values.matrix[0, 0] == float(Fraction(4, 3))
    assert values.support_mask.tolist() == [True, True, False, False]
    assert values.matrix[1, 0] == 2. and values.matrix[2, 0] == 0.
    closed = dictionary.evaluate([Fraction(3), Fraction(4)], coordinate="sample", support="analytic")
    np.testing.assert_array_equal(closed.matrix[:, 0], [2., 0.])
    assert closed.support_mask.tolist() == [False, False]
    exact = Fraction(4, 3)
    assert abs(Fraction.from_float(values.matrix[0, 0])-exact) <= Fraction.from_float(values.error[0, 0])


def test_seconds_are_their_actual_binary64_sites_without_snapping_to_native_rows():
    dictionary = _dictionary(np.array([[0.], [1.], [1e16]]), rate=10.)
    exact = dictionary.evaluate([Fraction(1, 10)], coordinate="seconds")
    rounded = dictionary.evaluate([.1], coordinate="seconds")
    assert exact.matrix[0, 0] == 1.
    assert abs(rounded.matrix[0, 0]-1.) > .5
    assert exact.metadata()["sample_coordinates_exact_ratio"] != rounded.metadata()["sample_coordinates_exact_ratio"]


def test_native_sites_keep_signed_zero_and_fractional_zero_directions():
    matrix = np.array([[-0., 1.], [0., 2.], [0., 4.]])
    dictionary = _dictionary(matrix)
    values = dictionary.evaluate([0, 1, 2], coordinate="sample")
    assert values.matrix.tobytes() == matrix.tobytes()
    assert np.count_nonzero(values.error) == 0
    fractional = dictionary.evaluate([Fraction(1, 4)], coordinate="sample")
    assert fractional.matrix[0, 0] == fractional.error[0, 0] == 0


def test_actual_selection_design_consumes_new_role_and_retains_prefix_censor():
    dictionary = _dictionary()
    state = ResponseState(1, 1, 1/16, 1.5, 1)
    design, ledger = selection_design(dictionary, np.array([3]), np.array([1]),
        origin_sample=0, observed_samples=8, state=state)
    expected = np.array([float(Fraction(4, 3)), 2., float(Fraction(10, 3)), float(Fraction(8, 3))])
    np.testing.assert_array_equal(design[:4], 0.)
    np.testing.assert_array_equal(design[4:, 0], expected)
    np.testing.assert_array_equal(design[:, 0], design[:, 1])
    assert ledger["events"][0]["response_sample_centers_half_open"] == [4, 8]
    assert ledger["events"][0]["right_censored"] is False
    shorter, short_ledger = selection_design(dictionary, np.array([3]), np.array([1]),
        origin_sample=0, observed_samples=7, state=state)
    assert shorter.tobytes() == design[:7].tobytes()
    assert short_ledger["events"][0]["right_censored"] is True


def test_evaluation_save_load_binds_original_rows_terminal_and_exact_query(tmp_path):
    dictionary = _dictionary()
    record = dictionary.record()
    record["implementation_sha256"] = "a"*64
    original = NativeResponseDictionary.from_record(record)
    evaluation = original.evaluate([Fraction(1, 3), Fraction(7, 3)], coordinate="sample")
    saved = evaluation.record()
    saved["consumer_implementation_sha256"] = "b"*64
    old = NativeResponseEvaluation.from_record(saved, dictionary=original)
    old.save(tmp_path/"evaluation.json")
    restored = NativeResponseEvaluation.load(tmp_path/"evaluation.json", dictionary=original)
    assert restored.record() == saved
    assert restored.matrix.tobytes() == evaluation.matrix.tobytes()
    changed = deepcopy(saved)
    changed["matrix"][0][0] = np.nextafter(changed["matrix"][0][0], np.inf)
    with pytest.raises(DictionaryArtifactError, match="relation_differs"):
        NativeResponseEvaluation.from_record(changed, dictionary=original)
    with pytest.raises(DictionaryArtifactError, match="relation_differs"):
        NativeResponseEvaluation.from_record(saved, dictionary=_dictionary(terminal=np.ones(1)))
    changed_parent = deepcopy(record)
    changed_parent["terminal_values"][0] = 1.
    with pytest.raises(DictionaryArtifactError, match="relation_differs"):
        NativeResponseDictionary.from_record(changed_parent)


def test_source_native_binding_rejects_a_different_basis():
    with pytest.raises(DictionaryArtifactError, match="binding_differs"):
        NativeResponseDictionary.from_source_matrix(np.ones((3, 1)), sample_rate_hz=8.,
            source_basis_sha256=array_digest(np.zeros((3, 1))), terminal_values=np.zeros(1))
