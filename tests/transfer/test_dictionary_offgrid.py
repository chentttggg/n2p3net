"""The saved native program, exact physical sites and portable query artifacts."""
from __future__ import annotations

import copy
import json
from decimal import ROUND_DOWN, localcontext
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from scipy.interpolate import BSpline

from transfer.dictionary_artifact import DictionaryArtifactError
from transfer.noise_model import array_digest
from transfer.response_dictionary import (
    ResponseDictionary,
    ResponseDictionaryEvaluation,
    ResponseDictionarySpec,
)

LEGAL = [(family, endpoint) for family in ("gaussian", "cosine", "bspline", "raised_cosine", "multiscale_bspline")
         for endpoint in ("free", "hard_zero") if (family, endpoint) != ("gaussian", "hard_zero")]


def _dictionary(family, endpoint="free", *, rate=12.1, duration=1.):
    return ResponseDictionary.compile(ResponseDictionarySpec(family, duration,
        9 if family == "multiscale_bspline" else 5, endpoint), sample_rate_hz=rate)


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_all_legal_native_sites_keep_original_program_bits_and_endpoint_side(family, endpoint):
    dictionary = _dictionary(family, endpoint)
    n = len(dictionary.matrix)
    native = dictionary.evaluate(np.arange(n), coordinate="sample")
    assert native.ambient_matrix.tobytes() == dictionary.ambient_matrix.tobytes()
    assert native.matrix.tobytes() == dictionary.matrix.tobytes()
    rate = Fraction.from_float(dictionary.sample_rate_hz)
    exact_seconds = dictionary.evaluate([Fraction(i)/rate for i in range(n)])
    assert exact_seconds.matrix.tobytes() == dictionary.matrix.tobytes()
    endpoint_eval = dictionary.evaluate([0, n], coordinate="sample")
    assert endpoint_eval.ambient_matrix.tobytes() == dictionary.endpoint_rows.tobytes()
    cropped = dictionary.evaluate([-Fraction(1, 10), 0, n-Fraction(1, 10**30), n],
                                  coordinate="sample", support="half_open")
    assert cropped.support_mask.tolist() == [False, True, True, False]
    assert np.array_equal(cropped.matrix[[0, 3]], np.zeros((2, dictionary.matrix.shape[1])))
    assert dictionary.metadata()["schema"] in {"n2p3_response_dictionary/2", "n2p3_response_dictionary/3",
                                               "n2p3_response_dictionary/4"}


def test_gaussian_hard_value_stays_original_typed_unsupported():
    with pytest.raises(DictionaryArtifactError, match="unsupported_or_empty_hard_endpoint"):
        _dictionary("gaussian", "hard_zero")


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_offgrid_scalar_values_match_independent_function_and_saved_chart(family, endpoint):
    dictionary = _dictionary(family, endpoint, rate=16.)
    length, width = len(dictionary.matrix), dictionary.spec.n_basis
    sites = [Fraction(1, 7), Fraction(length, 2), Fraction(7*length, 8), Fraction(length)]
    evaluated = dictionary.evaluate(sites, coordinate="sample")
    with mp.workdps(110):
        oracle = []
        for site in sites:
            q = mp.mpf(site.numerator)/site.denominator
            row = []
            for column in range(width):
                if family == "cosine":
                    value = mp.cos(mp.pi*q*column/length)
                elif family == "gaussian":
                    center = mp.mpf(column)*(length-1)/(width-1)
                    value = mp.exp(-mp.mpf('.5')*((q-center)/(mp.mpf(length)/(width-1)))**2)
                elif family == "raised_cosine":
                    phase = q*(width-1)/length-column
                    value = 0 if abs(phase) >= 1 else (1+mp.cos(mp.pi*phase))/2
                else:
                    columns = 6 if family == "multiscale_bspline" and column < 6 else 9 if family == "multiscale_bspline" else width
                    index = column if family != "multiscale_bspline" or column < 6 else (2, 4, 6)[column-6]
                    knots = np.r_[np.zeros(4), np.arange(1, columns-3)/(columns-3), np.ones(4)]
                    value = BSpline(knots, np.eye(columns), 3)(float(site/length))[index]
                row.append(float(value))
            oracle.append(row)
    np.testing.assert_allclose(evaluated.ambient_matrix, oracle, atol=5e-16, rtol=0)
    # Explicit exact dot on the admitted binary64 functions and same saved N.
    expected = []
    for row in evaluated.ambient_matrix:
        expected.append([float(sum((Fraction.from_float(float(a))*Fraction.from_float(float(b))
             for a, b in zip(row, column, strict=True)), Fraction())) for column in dictionary.constraint_map.T])
    assert evaluated.matrix.tobytes() == np.asarray(expected).tobytes()


def test_seconds_binary64_and_exact_native_site_are_distinct_without_snapping():
    dictionary = _dictionary("cosine", rate=12.1)
    site = 1
    rounded_seconds = dictionary.times[site]
    real_seconds = Fraction(site)/Fraction.from_float(dictionary.sample_rate_hz)
    query = dictionary.evaluate([rounded_seconds, real_seconds])
    record = query.record()
    assert record["query_exact_ratio"][0] != record["query_exact_ratio"][1]
    assert record["sample_coordinates_exact_ratio"][1] == [1, 1]
    assert record["sample_coordinates_exact_ratio"][0] != [1, 1]
    assert query.matrix[1].tobytes() == dictionary.matrix[site].tobytes()


def test_multiscale_offgrid_same_span_requires_actual_physical_prior_pullback():
    multi = _dictionary("multiscale_bspline", rate=16.)
    fine = ResponseDictionary.compile(ResponseDictionarySpec("bspline", 1., 9), sample_rate_hz=16.)
    refinement = np.asarray([[float(Fraction(*pair)) for pair in row]
                            for row in multi.metadata()["generation"]["coarse_in_fine_exact_ratio"]])
    chart = np.column_stack((refinement, np.eye(9)[:, (2, 4, 6)]))
    sites = [Fraction(16*i, 31) for i in range(32)]
    old = fine.evaluate(sites, coordinate="sample").matrix
    new = multi.evaluate(sites, coordinate="sample").matrix
    np.testing.assert_allclose(new, old@chart, atol=2e-16, rtol=0)
    covariance = np.diag(np.arange(1., 10.))
    inverse = np.linalg.inv(chart)
    corrected = new@(inverse@covariance@inverse.T)@new.T
    np.testing.assert_allclose(corrected, old@covariance@old.T, atol=3e-14, rtol=0)
    assert np.max(abs(new@covariance@new.T-old@covariance@old.T)) > 1.
    assert multi.metadata()["generation"]["span"].startswith("independent_nine_column_chart")


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_save_load_keeps_native_producer_query_and_outputs_without_platform_math(family, endpoint, tmp_path, monkeypatch):
    original = _dictionary(family, endpoint)
    parent = original.record()
    parent["implementation_sha256"], parent["compiler"]["source_sha256"] = "a"*64, "b"*64
    dictionary = ResponseDictionary.from_record(parent)
    sites = [[Fraction(1, 7), Fraction(5, 3)], [0, len(dictionary.matrix)]]
    evaluation = dictionary.evaluate(sites, coordinate="sample", support="half_open")
    record = json.loads(json.dumps(evaluation.record()))
    path = evaluation.save(tmp_path/"evaluation.json")
    for name in ("cos", "exp", "interp"):
        monkeypatch.setattr(np, name, lambda *a, **kw: pytest.fail("platform math used"))
    with localcontext() as context:
        context.prec, context.rounding = 6, ROUND_DOWN
        restored = ResponseDictionaryEvaluation.load(path, dictionary=dictionary)
    assert restored.record() == record
    assert restored.matrix.tobytes() == evaluation.matrix.tobytes()
    assert restored.metadata()["dictionary"]["implementation_sha256"] == "a"*64
    assert restored.metadata()["dictionary"]["compiler"]["source_sha256"] == "b"*64
    assert dictionary.record() == parent
    for key in ("matrix", "ambient_matrix", "support_mask"):
        with pytest.raises(ValueError):
            getattr(restored, key).setflags(write=True)


@pytest.mark.parametrize("attack", ["value", "query", "support", "program", "parent", "mask"])
def test_rehashed_evaluation_mutation_cannot_substitute_function_query_or_parent(attack):
    dictionary = _dictionary("cosine", "hard_zero")
    record = dictionary.evaluate([Fraction(1, 7), Fraction(-1, 7)], coordinate="sample").record()
    forged = copy.deepcopy(record)
    if attack == "value":
        forged["matrix"][0][0] = np.nextafter(forged["matrix"][0][0], np.inf)
    elif attack == "query":
        forged["query_exact_ratio"][0] = [2, 7]
    elif attack == "support":
        forged["support_program"] = "half_open"
    elif attack == "program":
        forged["program"]["scalar_expression"] = "interpolate_from_native_array"
    elif attack == "parent":
        forged["dictionary"]["spec"]["family"] = "gaussian"
    else:
        forged["support_mask"][1] = True
    forged["state_sha256"] = array_digest(np.asarray(forged["ambient_matrix"]), np.asarray(forged["matrix"]),
                                        np.asarray(forged["support_mask"], dtype=bool))
    with pytest.raises(DictionaryArtifactError):
        ResponseDictionaryEvaluation.from_record(forged, dictionary=dictionary)


def test_in_memory_replacement_and_different_parent_are_rejected():
    dictionary = _dictionary("bspline")
    evaluated = dictionary.evaluate(Fraction(1, 7), coordinate="sample")
    assert evaluated.matrix.shape == (5,) and evaluated.support_mask.shape == ()
    with pytest.raises(DictionaryArtifactError):
        ResponseDictionaryEvaluation.from_record(evaluated.record(), dictionary=_dictionary("bspline", rate=14.1))
    evaluated.matrix = evaluated.matrix+1.
    with pytest.raises(DictionaryArtifactError, match="immutable_evaluation"):
        evaluated.record()


@pytest.mark.parametrize("query", [True, float("nan"), float("inf"), complex(1), "1", []])
def test_unrepresentable_query_types_cannot_silently_be_cast_to_real(query):
    with pytest.raises(DictionaryArtifactError):
        _dictionary("cosine").evaluate(query)
