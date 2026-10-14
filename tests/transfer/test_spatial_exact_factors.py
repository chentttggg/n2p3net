"""Actual physical-map/D/Z products retain their exact structural kernel."""
from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from transfer.noise_model import array_digest, metadata_digest
from transfer.spatial_signal_prior import (
    ERPChannelEstimate,
    ERPPhysicalEffect,
    SpatialLayout,
    SpatialLikelihoodDependence,
    SpatialObservation,
    SpatialPriorIdentificationError,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
    _exact_structural_null,
    _factor_structural_rowspace,
    _SpatialCompression,
    _SpatialStructuralFactors,
)
from transfer.temporal_noise import TemporalNoise


def generic_source(mapping=None, *, nuisance_alias=False):
    mapping = np.array([[1., 3., 5., -9.]]) if mapping is None else np.asarray(mapping)
    rng = np.random.default_rng(7740)
    layout = SpatialLayout(("Fz", "Cz", "Pz", "Oz"), "source-earlobe", "1" * 64, ("EEG",) * 4)
    basis = np.array([[1., 0.], [.7, .4], [.2, 1.], [0., .6]])
    weights = np.array([0., .25, .5, .25])
    temporal = TemporalNoise.from_parameters([], innovation_variance=1., reference_id="actual-source-noise")
    binding = {"dictionary_sha256": "b" * 64, "actual_basis_sha256": array_digest(basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": 128.,
        "operator_identity": "synthetic_native_raw", "coefficient_groups": ["common", "target_increment"]}
    prior = SpatialSignalPrior(layout, np.full((4, 4), 12.), np.eye(4), .2 * np.eye(4), binding,
        {"role": "predeclared", "authority_sha256": "c" * 64})
    observation = SpatialObservation(layout, mapping, tuple(f"actual-row-{i}" for i in range(len(mapping))),
                                     "explicit_actual_map")
    effect = ERPPhysicalEffect.from_prior(prior, basis=basis, waveform_weights=weights,
        grid_seconds=np.arange(4) / 128., observation=observation)
    native_effect = replace(effect, observation=SpatialObservation.native(layout))
    noise_factor = np.array([[1., 0., 0., 0.], [.2, .8, 0., 0.], [-.1, .15, 1.2, 0.], [.3, -.1, .2, .7]])
    noise = noise_factor @ noise_factor.T
    beta = rng.normal(size=(4, 4)) * .2
    blocks = []
    for index in range(2):
        d = rng.normal(size=(18, 4))
        z = np.column_stack((np.ones(18), np.linspace(-1., 1., 18)))
        if nuisance_alias:
            d[:, 3] = z[:, 0]
        samples = d @ beta + rng.normal(size=(18, 4)) @ noise_factor.T + z @ rng.normal(size=(2, 4))
        authority = {"record_id": f"source:C{index}", "fit_scope": "source_training",
            "source_sha256": "2" * 64, "dictionary_sha256": "b" * 64,
            "temporal_program_sha256": temporal.record()["artifact_sha256"],
            "raw_sample_support": [30 * index, 30 * index + 18],
            "original_view": {"participant_key": "actual-source-person"}}
        blocks.append(SpatialResponseBlock(observation.project(samples), d, observation, authority, z))
    posterior = SpatialSignalPosterior.fit(prior, blocks, native_noise_covariance=noise,
                                           noise_binding={"authority_sha256": "d" * 64})
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    return posterior, effect, native_effect, dependence, temporal, blocks


def direct(inputs, effect=None):
    posterior, original, _, dependence, temporal, _ = inputs
    return posterior.estimate_effect(original if effect is None else effect, dependence, temporal_noise=temporal)


def destination(inputs, mapping):
    posterior, effect, *_ = inputs
    layout = replace(effect.layout, acquisition_reference="independent-target-reference", authority_sha256="3" * 64)
    prior = replace(posterior.prior, layout=layout)
    return ERPPhysicalEffect.from_prior(prior, basis=effect.basis, waveform_weights=effect.waveform_weights,
        grid_seconds=effect.grid_seconds, observation=SpatialObservation(layout, mapping,
            tuple(f"destination-row-{i}" for i in range(len(mapping))), "explicit_reference_free_map"))


def transported(inputs, effect):
    posterior, _, native, dependence, temporal, _ = inputs
    return ERPChannelEstimate.from_transported_posterior(posterior, source_effect=native,
        destination_effect=effect, dependence=dependence, temporal_noise=temporal)


def test_original_factor_kernel_and_generic_blue_match_physical_oracle(tmp_path):
    inputs = generic_source()
    posterior, effect, *_ = inputs
    # One observed covector on four EEG: three lost spatial directions for
    # each of the four response coordinates, despite rounded-kron remnants.
    assert len(posterior.structural_null_basis) == 12
    assert posterior.structural_binding["arithmetic"] == "exact_binary64_joint_temporal_rowspace_record_local_quotient_beta_fraction_rref"
    rounded_operators = []
    for block in inputs[5]:
        mapping = block.observation._structural_matrix()
        rounded_operators.append(np.column_stack((np.kron(mapping, block.design),
                                                  np.kron(np.eye(len(mapping)), block.nuisance))))
    rounded_basis, _ = _exact_structural_null(rounded_operators, posterior.prior.mean.size)
    assert len(rounded_basis) == 0
    null = np.zeros(16)
    null[:4], null[4:8] = 3., -1.
    assert any(sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                    for a, b in zip(row, null, strict=True)), Fraction(0))
               for row in rounded_operators[0][:, :16])
    measured = direct(inputs)
    # Independent oracle from the original observed records, not saved QR.
    designs, samples = [], []
    for block in inputs[5]:
        z = block.nuisance
        designs.append(block.design - z @ np.linalg.lstsq(z, block.design, rcond=None)[0])
        samples.append(block.samples - z @ np.linalg.lstsq(z, block.samples, rcond=None)[0])
    design, y = np.vstack(designs), np.vstack(samples)
    inverse = np.linalg.pinv(design)
    expected_factor = effect.response_functional @ inverse
    variance = float((effect.spatial_map @ posterior.native_noise_covariance @ effect.spatial_map.T)[0, 0])
    np.testing.assert_allclose(measured.mean, expected_factor @ y, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(measured.covariance, np.array([[variance * (expected_factor @ expected_factor)]]),
                               rtol=1e-11, atol=1e-12)
    actual = transported(inputs, destination(inputs, np.array([[1., 3., 5., -9.]])))
    np.testing.assert_allclose(actual.mean, measured.mean, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual.covariance, measured.covariance, rtol=1e-12, atol=1e-13)
    path = tmp_path / "physical-source.json"
    posterior.save(path)
    restored = SpatialSignalPosterior.load(path, expected_artifact_sha256=posterior.record()["artifact_sha256"])
    assert restored.structural_null_basis == posterior.structural_null_basis
    np.testing.assert_array_equal(restored.mean, posterior.mean)
    np.testing.assert_array_equal(restored.precision_root, posterior.precision_root)
    restored_dependence = SpatialLikelihoodDependence.independent(restored, authority_sha256="e" * 64)
    replay = restored.estimate_effect(effect, restored_dependence, temporal_noise=inputs[4])
    np.testing.assert_array_equal(replay.mean, measured.mean)
    np.testing.assert_array_equal(replay.covariance_factor, measured.covariance_factor)


@pytest.mark.parametrize("power", [44, 48])
def test_tiny_genuine_unobserved_contrast_is_exactly_rejected(power):
    inputs = generic_source()
    delta = 2. ** -power
    mapping = np.array([[1. + delta, 3. - delta, 5., -9.]])
    missing = replace(inputs[1], observation=SpatialObservation(inputs[1].layout, mapping,
                      ("tiny-unobserved-contrast",), "explicit_reference_free_map"))
    assert sum(missing.structural_spatial_map[0], Fraction(0)) == 0
    defect = sum((value * coefficient for value, coefficient in zip(
        missing.structural_spatial_map[0], (3, -1, 0, 0), strict=True)), Fraction(0))
    assert defect == Fraction(1, 2 ** (power - 2))
    with pytest.raises(SpatialPriorIdentificationError, match="physical_effect_not_identified_by_structural_C"):
        direct(inputs, missing)
    with pytest.raises(SpatialPriorIdentificationError, match="physical_effect_not_identified_by_structural_C"):
        transported(inputs, destination(inputs, mapping))


@pytest.mark.parametrize("gain", [.5, -2.])
def test_exact_output_gains_keep_the_same_source_information(gain):
    original = generic_source()
    changed = generic_source(np.array([[1., 3., 5., -9.]]) * gain)
    common = destination(original, np.array([[1., 3., 5., -9.]]))
    control, measured = transported(original, common), transported(changed, common)
    assert len(changed[0].structural_null_basis) == 12
    np.testing.assert_allclose(measured.mean, control.mean, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(measured.covariance, control.covariance, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(measured.noise_sd, control.noise_sd)


def test_exact_nonorthogonal_output_chart_preserves_physical_kernel():
    mapping = np.array([[1., 3., 5., -9.], [-2., 4., -6., 4.]])
    chart = np.array([[1., .25], [.5, 1.]])
    original, changed = generic_source(mapping), generic_source(chart @ mapping)
    common = destination(original, mapping)
    control, measured = transported(original, common), transported(changed, common)
    assert len(original[0].structural_null_basis) == len(changed[0].structural_null_basis) == 8
    np.testing.assert_allclose(measured.mean, control.mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(measured.covariance, control.covariance, rtol=1e-11, atol=1e-12)


def test_record_local_nuisance_remains_part_of_exact_identifiability():
    inputs = generic_source(nuisance_alias=True)
    assert len(inputs[0].structural_null_basis) == 13
    with pytest.raises(SpatialPriorIdentificationError, match="physical_effect_not_identified_by_structural_C"):
        direct(inputs)


@pytest.mark.parametrize("field", ["design", "nuisance"])
def test_resigned_original_factor_cannot_use_another_compressed_likelihood(field):
    posterior, *_ = generic_source()
    saved = copy.deepcopy(posterior.record())
    factors = saved["structural_factors"]
    factors[0][field][0][0] += .25
    observation = SpatialObservation.from_record(saved["fit_records"][0]["observation"])
    changed = _SpatialStructuralFactors(observation, factors[0]["design"], factors[0]["nuisance"],
        _SpatialCompression.from_record(factors[0]["source_compression"]))
    factors[0] = changed.record()
    saved["fit_records"][0]["compression"]["physical_structural_factors_sha256"] = factors[0]["artifact_sha256"]
    saved["structural_binding"]["factors_sha256"] = metadata_digest(factors)
    # Keep the original producer's likelihood/QR state. Recompute the exact
    # factor certificate so the failure reaches the missing source-law join.
    all_factors = tuple(_SpatialStructuralFactors.from_record(factor,
        observation=SpatialObservation.from_record(record["observation"]))
        for factor, record in zip(factors, saved["fit_records"], strict=True))
    basis, resolution = _factor_structural_rowspace(all_factors, posterior.prior.mean.size)
    saved["structural_rowspace"] = [[[value.numerator, value.denominator] for value in vector] for vector in basis]
    saved["structural_binding"]["resolution"] = resolution
    saved["fit_sha256"] = metadata_digest(saved["fit_records"])
    saved["artifact_sha256"] = metadata_digest({key: value for key, value in saved.items() if key != "artifact_sha256"})
    with pytest.raises(ValueError, match="QR root differs from actual C likelihood information"):
        SpatialSignalPosterior.from_record(saved)


def test_resigned_noise_map_cannot_change_the_actual_source_information_law():
    posterior, *_ = generic_source()
    saved = copy.deepcopy(posterior.record())
    factor = saved["structural_factors"][0]
    compression = factor["source_compression"]
    compression["spatial_map"][0][0] += .01
    compression["artifact_sha256"] = metadata_digest({key: value for key, value in compression.items()
                                                      if key != "artifact_sha256"})
    factor["artifact_sha256"] = metadata_digest({key: value for key, value in factor.items() if key != "artifact_sha256"})
    saved["fit_records"][0]["compression"]["physical_structural_factors_sha256"] = factor["artifact_sha256"]
    saved["structural_binding"]["factors_sha256"] = metadata_digest(saved["structural_factors"])
    saved["fit_sha256"] = metadata_digest(saved["fit_records"])
    saved["artifact_sha256"] = metadata_digest({key: value for key, value in saved.items() if key != "artifact_sha256"})
    with pytest.raises(ValueError, match="source noise map differs from its original observed noise factor"):
        SpatialSignalPosterior.from_record(saved)


@pytest.mark.parametrize("scale", [1e-200, 1e155])
def test_source_compression_certifies_tiny_and_huge_original_column_units(scale):
    layout = SpatialLayout(("Fz",), "actual-reference", "1" * 64, ("EEG",))
    observation = SpatialObservation.native(layout)
    q = np.array([[1.], [0.]])
    root = np.array([[scale]])
    compression = _SpatialCompression(np.ones((1, 1)), np.ones((1, 1)), q, root)
    legal = _SpatialStructuralFactors(observation, np.array([[scale], [0.]]), compression=compression)
    compression.validate(legal, root, np.ones((1, 1)))
    changed = replace(legal, design=2 * legal.design)
    with pytest.raises(ValueError, match="QR root differs from actual C likelihood information"):
        compression.validate(changed, root, np.ones((1, 1)))


def test_source_compression_rejects_finite_q_with_unrepresentable_gram():
    with pytest.raises(ValueError, match="QR factor is not orthonormal"):
        _SpatialCompression._orthogonal(np.array([[1e200], [0.]]))
