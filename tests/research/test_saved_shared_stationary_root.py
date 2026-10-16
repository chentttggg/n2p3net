"""Unchanged /3 saved-root admission through a separate small-factor witness."""

import json
from fractions import Fraction as Q

import numpy as np
import pytest

from research import channel_noise_mechanism as cm
from research import shared_stationary_factor as factor
from research.execution import UnidentifiableFitError
from research.stationary_noise_artifact import stationary_root
from transfer.noise_model import array_digest, metadata_digest
from transfer.temporal_noise import TemporalNoise


def _multiply(a, b):
    return [[sum((x*y for x, y in zip(row, column, strict=True)), Q())
             for column in zip(*b, strict=True)] for row in a]


def _transpose(a):
    return list(map(list, zip(*a, strict=True)))


def _inverse(a):
    size = len(a)
    rows = [list(row)+[Q(i == j) for j in range(size)] for i, row in enumerate(a)]
    for i in range(size):
        divisor = rows[i][i]
        rows[i] = [value/divisor for value in rows[i]]
        for j in range(size):
            if i != j:
                multiplier = rows[j][i]
                rows[j] = [v-multiplier*w for v, w in zip(rows[j], rows[i], strict=True)]
    return [row[size:] for row in rows]


@pytest.mark.parametrize("scale", [1., 1e-320, 1e200])
def test_original_nonproduct_root_pullback_has_an_independent_exact_bound(scale):
    reflections = [-.625]
    sigma = np.array([[1., -.25], [-.25, .75]])*scale
    root = stationary_root([reflections]*2, sigma, 3)
    # A true non-Kronecker perturbation, at the level of the original /3 root.
    root[-1, 0] += abs(root[-1, -1])*2.**-38
    before = root.tobytes()
    certificate = factor.admit_saved_shared_stationary_root(reflections, sigma, 3, root)
    phi = Q(5, 8)
    covariance = [[phi**abs(i//2-j//2)/(1-phi*phi)*Q.from_float(float(sigma[i % 2, j % 2]))
                   for j in range(6)] for i in range(6)]
    inverse = _inverse([[Q.from_float(float(v)) for v in row] for row in root])
    pullback = _multiply(_multiply(inverse, covariance), _transpose(inverse))
    squared = sum(((value-Q(i == j))**2 for i, row in enumerate(pullback)
                   for j, value in enumerate(row)), Q())
    bound = Q.from_float(certificate["saved_root_law_pullback_F_bound"])
    assert squared <= bound**2 and bound < Q.from_float(1e-8*np.sqrt(6))
    assert root.tobytes() == before
    assert certificate["saved_root_sha256"] == array_digest(root)
    assert certificate["saved_root_sha256"] != certificate["witness_root_sha256"]


def test_absolute_small_root_damage_in_a_weak_direction_is_rejected():
    sigma = np.diag([1., 2.**-80])
    root = stationary_root([[], []], sigma, 2)
    root[1, 1] *= 1.001
    assert abs(root[1, 1]-2.**-40) < 1e-15
    with pytest.raises(UnidentifiableFitError, match="original law policy"):
        factor.admit_saved_shared_stationary_root([], sigma, 2, root)


def test_distance_bound_covers_exact_subtraction_and_squared_underflow():
    a = np.array([[2.**-600, np.nextafter(0., 1.)], [-2.**-600, 1.]])
    b = np.array([[0., 0.], [2.**-600, np.nextafter(1., 0.)]])
    squared = sum(((Q.from_float(float(x))-Q.from_float(float(y)))**2
                   for x, y in zip(a.flat, b.flat, strict=True)), Q())
    assert 0 < squared <= factor._difference_norm_upper(a, b)**2
    smallest = np.array([[np.nextafter(0., 1.)]])
    assert Q.from_float(float(smallest[0, 0]))**2 <= factor._difference_norm_upper(smallest, np.zeros((1, 1)))**2


def _noise(family="shared"):
    temporal = TemporalNoise.from_reflections([-.625], innovation_variance=7., reference_id="original-shape")
    geometry = cm.NoiseGeometry(("a", "b"), 1000., "V", "nose", "native", 2)
    return cm.FittedChannelNoise((temporal, temporal), np.array([[1., -.25], [-.25, .75]])*1e-12,
        geometry, 3, family, {"fit_scope": "analytic_reference", "reference_id": "saved-original", "records": []})


def test_old_codec_load_save_preserves_original_root_and_record_bytes(tmp_path, monkeypatch):
    original = _noise()
    path = tmp_path/"old.json"
    original.save(path)
    original_bytes = path.read_bytes()
    expected = json.loads(original_bytes)
    samples = np.arange(20).reshape(10, 2)*1e-7
    white = original.whiten(samples)
    small_shapes = []
    original_cholesky = factor._cholesky

    def trace_small(law):
        small_shapes.append(len(law))
        return original_cholesky(law)

    def forbidden(*args, **kwargs):
        raise AssertionError("a saved /3 root was compiled or replaced by a /4 producer")

    monkeypatch.setattr(factor, "_cholesky", trace_small)
    monkeypatch.setattr(cm, "stationary_root", forbidden)
    monkeypatch.setattr(cm, "compile_shared_stationary_root", forbidden)
    restored = cm.FittedChannelNoise.load(path)
    assert small_shapes == [3, 2]
    assert restored.shared_factor_state is None
    assert metadata_digest(restored.record()) == expected["state_sha256"]
    assert json.loads(json.dumps(restored.record())) == expected["state"]
    assert restored.saved_root_admission["saved_root_sha256"] == array_digest(original.compiled_prefix_root)
    assert restored.saved_root_admission["input_law_sha256"] == expected["state"]["prefix_compiler"]["input_law_sha256"]
    np.testing.assert_array_equal(restored.whiten(samples), white)
    restored.save(tmp_path/"resaved.json")
    assert (tmp_path/"resaved.json").read_bytes() == original_bytes == path.read_bytes()
    with pytest.raises(TypeError, match="frozen"):
        restored.saved_root_admission["program"] = "changed"


def test_unresolved_witness_uses_canonical_admission_of_the_same_saved_root(tmp_path, monkeypatch):
    original = _noise()
    path = tmp_path/"old.json"
    original.save(path)
    calls = []

    def unresolved(*args, **kwargs):
        raise UnidentifiableFitError("small numerical witness unresolved")

    def canonical(*args, **kwargs):
        calls.append(array_digest(kwargs["supplied_root"]))
        return stationary_root(*args, **kwargs)

    monkeypatch.setattr(cm, "admit_saved_shared_stationary_root", unresolved)
    monkeypatch.setattr(cm, "stationary_root", canonical)
    restored = cm.FittedChannelNoise.load(path)
    assert calls == [array_digest(original.compiled_prefix_root)]
    assert restored.record() == original.record()
    assert restored.saved_root_admission["program"] == "canonical_pairwise_saved_root_admission/1"


def test_format_error_is_not_caught_as_a_numerical_witness_failure(tmp_path, monkeypatch):
    original = _noise()
    path = tmp_path/"old.json"
    original.save(path)

    def malformed(*args, **kwargs):
        raise ValueError("witness format or original identity mismatch")

    def forbidden(*args, **kwargs):
        raise AssertionError("format error entered a numerical fallback")

    monkeypatch.setattr(cm, "admit_saved_shared_stationary_root", malformed)
    monkeypatch.setattr(cm, "stationary_root", forbidden)
    with pytest.raises(ValueError, match="format or original identity"):
        cm.FittedChannelNoise.load(path)


def test_resigned_invalid_saved_root_still_fails_canonical_law_after_witness_failure(tmp_path):
    original = _noise()
    payload = json.loads(json.dumps(original.record()))
    changed = np.asarray(payload["compiled_prefix_root"])
    changed[-1, -1] *= 1.001
    payload["compiled_prefix_root"] = changed.tolist()
    payload["prefix_compiler"]["compiled_root_sha256"] = array_digest(changed)
    path = tmp_path/"resigned.json"
    path.write_text(json.dumps({"state": payload, "state_sha256": metadata_digest(payload)}), encoding="utf8")
    with pytest.raises(ValueError, match="declared stationary noise model"):
        cm.FittedChannelNoise.load(path)


def test_explicit_factor_codec_does_not_enter_the_old_root_witness(tmp_path, monkeypatch):
    original = _noise()
    factor_noise = cm.FittedChannelNoise.from_shared_stationary_factors(original.temporal,
        original.innovation_covariance, geometry=original.geometry, conditioning_samples=3,
        provenance=original.provenance)
    path = tmp_path/"factor.json"
    factor_noise.save(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("factor codec entered an original /3 saved-root witness")

    monkeypatch.setattr(cm, "admit_saved_shared_stationary_root", forbidden)
    restored = cm.FittedChannelNoise.load(path)
    assert restored.record() == factor_noise.record()
    assert restored.saved_root_admission is None


def test_private_saved_root_retains_its_canonical_admission(tmp_path, monkeypatch):
    original = _noise("private")
    path = tmp_path/"private.json"
    original.save(path)
    calls = []

    def canonical(*args, **kwargs):
        calls.append(array_digest(kwargs["supplied_root"]))
        return stationary_root(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("private saved root entered shared numerical witness")

    monkeypatch.setattr(cm, "admit_saved_shared_stationary_root", forbidden)
    monkeypatch.setattr(cm, "stationary_root", canonical)
    restored = cm.FittedChannelNoise.load(path)
    assert calls == [array_digest(original.compiled_prefix_root)]
    assert restored.record() == original.record()
    assert restored.saved_root_admission is None
