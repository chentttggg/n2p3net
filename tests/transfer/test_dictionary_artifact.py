"""Portable compile/restore and forged-package probes through real ingress."""
import hashlib
import json
from decimal import ROUND_DOWN, localcontext

import numpy as np
import pytest

import transfer.response_dictionary as dictionary_owner
from research.contracts import semantic_sha256
from tests.transfer.test_ensemble_fit import selections
from transfer.dictionary_artifact import ARRAY_KEYS, DictionaryArtifactError, _binary64
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.noise_model import array_digest, metadata_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


@pytest.mark.parametrize("numerator,denominator,expected", [
    (1, 1, "0x1.0000000000000p+0"),
    (2**53+1, 2**53, "0x1.0000000000000p+0"),
    (2**53+3, 2**53, "0x1.0000000000002p+0"),
    (1, 2**1075, "0x0.0p+0"),
    (-1, 2**1075, "-0x0.0p+0"),
    (3, 2**1075, "0x0.0000000000002p-1022"),
    (2**52-1, 2**1074, "0x0.fffffffffffffp-1022"),
    (2**53-1, 2**1075, "0x1.0000000000000p-1022"),
    (-3, 2, "-0x1.8000000000000p+0"),
])
def test_integer_binary64_rounding_handles_ties_subnormal_and_signed_underflow(numerator, denominator, expected):
    assert _binary64(numerator, denominator).hex() == expected


@pytest.mark.parametrize("family,endpoint", [("gaussian", "free"), ("cosine", "free"), ("cosine", "hard_zero")])
def test_restore_retains_saved_state_and_producer_without_calling_compile(family, endpoint, monkeypatch):
    original = ResponseDictionary.compile(ResponseDictionarySpec(family, .5, 5, endpoint), sample_rate_hz=64.)
    record = original.record()
    # Current module bytes may differ. Retain identities, never certify their
    # authenticity from these claims alone; the executor/package owns that gate.
    record["implementation_sha256"] = "a"*64
    record["compiler"]["source_sha256"] = "b"*64

    def forbidden(*args, **kwargs):
        raise AssertionError("restore recompiled or called platform transcendental math")

    monkeypatch.setattr(ResponseDictionary, "compile", forbidden)
    monkeypatch.setattr(dictionary_owner, "compile_state", forbidden)
    for name in ("exp", "cos", "sqrt", "linspace"):
        monkeypatch.setattr(np, name, forbidden)
    restored = ResponseDictionary.from_record(json.loads(json.dumps(record)))
    assert restored.record() == record
    for key in ARRAY_KEYS:
        assert getattr(restored, key).tobytes() == getattr(original, key).tobytes()
        with pytest.raises(ValueError):
            getattr(restored, key).setflags(write=True)


def test_compile_is_independent_of_callers_decimal_context_and_has_exact_physical_anchors():
    spec = ResponseDictionarySpec("cosine", .5, 5, "hard_zero")
    standard = ResponseDictionary.compile(spec, sample_rate_hz=64.)
    with localcontext() as context:
        context.prec, context.rounding = 7, ROUND_DOWN
        context.Emax, context.Emin = 9, -9
        other = ResponseDictionary.compile(spec, sample_rate_hz=64.)
    assert other.record() == standard.record()
    np.testing.assert_array_equal(standard.times, np.arange(32)/64.)
    np.testing.assert_array_equal(standard.endpoint_rows, [[1., 1., 1., 1., 1.], [1., -1., 1., -1., 1.]])
    assert standard.ambient_matrix[16, 1] == 0.
    assert standard.ambient_matrix[16, 2] == -1.
    assert standard.ambient_matrix[16, 4] == 1.


@pytest.mark.parametrize("key,index", [("times", (3,)), ("ambient_matrix", (4, 1)),
    ("matrix", (4, 1)), ("endpoint_rows", (1, 2)), ("constraint_map", (0, 1))])
def test_coordinated_one_ulp_payload_rehash_does_not_establish_physical_relation(key, index):
    record = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), sample_rate_hz=64.).record()
    arrays = {name: np.asarray(record[name]) for name in ARRAY_KEYS}
    arrays[key][index] = np.nextafter(arrays[key][index], np.inf)
    record.update({name: value.tolist() for name, value in arrays.items()})
    record["state_sha256"] = array_digest(*(arrays[name] for name in ARRAY_KEYS))
    with pytest.raises(DictionaryArtifactError) as caught:
        ResponseDictionary.from_record(record)
    assert caught.value.reason == "physical_expression_relation_mismatch"
    assert caught.value.field == key
    assert caught.value.index == index


def test_legacy_record_is_not_silently_recognized_or_relabelled():
    record = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 3), sample_rate_hz=64.).record()
    record["schema"] = "n2p3_response_dictionary/1"
    with pytest.raises(DictionaryArtifactError) as caught:
        ResponseDictionary.from_record(record)
    assert caught.value.record()["reason"] == "unsupported_dictionary_schema"


def test_scalar_spec_substitution_cannot_relabel_existing_physical_basis():
    record = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 3), sample_rate_hz=64.).record()
    record["spec"]["family"] = "cosine"
    with pytest.raises(DictionaryArtifactError):
        ResponseDictionary.from_record(record)


@pytest.mark.parametrize("dtype", [np.float32, np.complex128, bool])
def test_numerical_artifact_rejects_lossy_or_nonreal_payload_before_conversion(dtype):
    record = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 1), sample_rate_hz=64.).record()
    record["ambient_matrix"] = np.asarray(record["ambient_matrix"], dtype=dtype)
    with pytest.raises(DictionaryArtifactError) as caught:
        ResponseDictionary.from_record(record)
    assert caught.value.reason == "invalid_numerical_payload"
    assert caught.value.field == "ambient_matrix"


def test_real_ensemble_package_restores_admitted_dictionary_without_compile_and_rejects_rehashed_basis(tmp_path, monkeypatch):
    groups = selections()
    spec = CalibrationExpertProgram("portable", "continuous_ar", noise_order=1, conditioning_samples=32,
        dictionary=ResponseDictionarySpec("gaussian", .5, 3))
    mixture = EnsembleProgram.equal_weights("portable", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture)
    before = fitted.score(groups[3].view)
    model = fitted._experts[0].predictor
    state_keys = tuple(model._state_arrays())
    package = fitted.save(tmp_path/"ensemble")

    def forbidden(*args, **kwargs):
        raise AssertionError("package restore recompiled its frozen dictionary")

    monkeypatch.setattr(ResponseDictionary, "compile", forbidden)
    monkeypatch.setattr(dictionary_owner, "compile_state", forbidden)
    restored = FittedEnsemble.load(package)
    assert restored.score(groups[3].view).record() == before.record()
    assert restored._experts[0].predictor.dictionary.record() == model.dictionary.record()

    filename = package/"expert_000.npz"
    with np.load(filename, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(values["metadata"]))
    assert metadata["schema"] == "n2p3_continuous_dictionary_model/2"
    # An attacker updates *both* equal free-basis arrays and every transport
    # digest. Only physical-expression admission, not a mirror hash, can reject.
    for key in ("basis", "dictionary_ambient_matrix"):
        values[key][4, 1] = np.nextafter(values[key][4, 1], np.inf)
    dictionary_arrays = {key: values["basis"] if key == "matrix" else values["dictionary_"+key]
                         for key in ARRAY_KEYS}
    metadata["dictionary"]["state_sha256"] = array_digest(*(dictionary_arrays[key] for key in ARRAY_KEYS))
    metadata["state_sha256"] = array_digest(*(values[key] for key in state_keys))
    metadata["artifact_sha256"] = metadata_digest({key: value for key, value in metadata.items() if key != "artifact_sha256"})
    values["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(filename, **values)
    manifest_path = package/"ensemble.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["record"]["experts"][0]["model_file_sha256"] = hashlib.sha256(filename.read_bytes()).hexdigest()
    manifest["record_sha256"] = semantic_sha256(manifest["record"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DictionaryArtifactError) as caught:
        FittedEnsemble.load(package)
    assert caught.value.reason == "physical_expression_relation_mismatch"
    assert caught.value.field == "ambient_matrix"
