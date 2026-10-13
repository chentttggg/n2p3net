"""Source-owned BLUE in a common reference-free effect chart, without raw solver mocks."""
from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from research.channel_noise_mechanism import FittedChannelNoise, FrozenERPResponse, NoiseGeometry
from tests.transfer.test_spatial_source_learning import learn, make_source
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
)
from transfer.temporal_noise import TemporalNoise


def common_measurement(inputs, destination):
    posterior, effect, dependence, temporal, _ = inputs
    return ERPChannelEstimate.from_transported_posterior(posterior, source_effect=effect,
        destination_effect=destination, dependence=dependence, temporal_noise=temporal)


def test_common_reference_blue_and_full_noise_sd_follow_actual_source_likelihood():
    inputs = make_source(0, channels=("Pz", "Fz", "Cz"), acquisition_reference="left earlobe", ar=.3)
    posterior, source, _, temporal, _ = inputs
    destination = make_source(10, scope="target_calibration", observation="cz")[1]
    measured = common_measurement(inputs, destination)
    channel_map = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    mapping = destination.spatial_map @ channel_map
    functional = np.kron(mapping, source.response_functional[None, :])
    x, y = np.vstack(posterior.information_roots), np.concatenate(posterior.responses)
    inverse = np.linalg.pinv(x)
    np.testing.assert_allclose(measured.mean, functional @ inverse @ y, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(measured.covariance, functional @ inverse @ inverse.T @ functional.T,
                               rtol=1e-12, atol=1e-13)
    inverse_white = np.linalg.inv(temporal.whiten(np.eye(len(source.waveform_weights))))
    variance = np.sum((source.waveform_weights @ inverse_white) ** 2)
    expected_sd = np.sqrt(variance * np.diag(mapping @ posterior.native_noise_covariance @ mapping.T))
    np.testing.assert_allclose(measured.noise_sd, expected_sd, rtol=1e-14)
    native = posterior.estimate_effect(source, inputs[2], temporal_noise=temporal)
    assert not np.allclose(measured.mean / measured.noise_sd, mapping @ (native.mean / native.noise_sd))
    assert measured.effect.record() == destination.record()
    assert measured.receipt["participant_key"] == "person-0"
    assert measured.receipt["resolution"]["prior_information_used"] is False
    assert measured.receipt["posterior_mean_used_as_measurement"] is False
    assert measured.record()["schema"] == "n2p3_ERP_effect_measurement/2"


def test_transport_identifies_common_contrast_without_unobserved_native_completion():
    inputs = make_source(0, observation="cz", rank_case="only_effect")
    posterior, _, dependence, temporal, _ = inputs
    # The source temporal definition can be native even though its native
    # common-reference component is not observed by the actual C likelihood.
    native = ERPPhysicalEffect.from_prior(posterior.prior, basis=inputs[1].basis,
        waveform_weights=inputs[1].waveform_weights, grid_seconds=inputs[1].grid_seconds,
        observation=SpatialObservation.native(posterior.prior.layout))
    with pytest.raises(SpatialPriorIdentificationError, match="structural_C"):
        posterior.estimate_effect(native, dependence, temporal_noise=temporal)
    destination = make_source(10, scope="target_calibration", observation="cz")[1]
    measured = common_measurement((posterior, native, dependence, temporal, inputs[4]), destination)
    direct = posterior.estimate_effect(inputs[1], dependence, temporal_noise=temporal)
    np.testing.assert_allclose(measured.mean, direct.mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(measured.covariance, direct.covariance, rtol=1e-13, atol=1e-13)
    with pytest.raises(SpatialPriorIdentificationError, match="structural_C"):
        common_measurement(make_source(1, rank_case="true_null"), destination)


def test_unused_destination_electrode_does_not_become_a_zero_filled_source_observation():
    inputs = make_source(0)
    target, destination, *_ = make_source(10, scope="target_calibration", observation="cz")
    layout = replace(destination.layout, channels=(*destination.layout.channels, "Oz"), channel_types=("EEG",) * 4)
    reference = replace(target.prior, layout=layout, mean=np.zeros((4, 4)), signal_covariance=np.eye(4))
    mapping = np.column_stack((destination.spatial_map, np.zeros(2)))
    effect = ERPPhysicalEffect.from_prior(reference, basis=destination.basis,
        waveform_weights=destination.waveform_weights, grid_seconds=destination.grid_seconds,
        observation=SpatialObservation(layout, mapping, destination.observation.output_names,
                                       destination.observation.output_reference))
    actual, control = common_measurement(inputs, effect), common_measurement(inputs, destination)
    np.testing.assert_array_equal(actual.mean, control.mean)
    np.testing.assert_array_equal(actual.covariance_factor, control.covariance_factor)
    local_layout = actual.receipt["physical_transport"]["local_effect"]["layout"]
    assert tuple(local_layout["channels"]) == inputs[0].prior.layout.channels
    assert "Oz" not in local_layout["channels"]


def test_reference_string_and_common_channel_intersection_do_not_supply_missing_information():
    inputs = make_source(0)
    native_destination = make_source(10, scope="target_calibration")[1]
    with pytest.raises(SpatialPriorIdentificationError, match="reference_component_not_observed"):
        common_measurement(inputs, native_destination)
    target = make_source(10, scope="target_calibration", observation="cz")[1]
    missing = make_source(1, channels=("Fz", "Oz", "Pz"))
    with pytest.raises(SpatialPriorIdentificationError, match="required_electrode_not_measured"):
        common_measurement(missing, target)
    matrix = np.array([[1e-30, -1., 1.]])
    tiny = ERPPhysicalEffect.from_prior(native_destination_prior := make_source(11)[0].prior,
        basis=target.basis, waveform_weights=target.waveform_weights, grid_seconds=target.grid_seconds,
        observation=SpatialObservation(native_destination_prior.layout, matrix, ("tiny-native",), "explicit"))
    with pytest.raises(SpatialPriorIdentificationError, match="reference_component_not_observed"):
        common_measurement(inputs, tiny)
    changed_grid = replace(target, grid_seconds=target.grid_seconds + 1.)
    with pytest.raises(ValueError, match="temporal effect"):
        common_measurement(inputs, changed_grid)


def test_transported_measurement_save_load_recomputes_transport_and_source_blue(tmp_path):
    inputs = make_source(0, acquisition_reference="left earlobe", channels=("Pz", "Cz", "Fz"))
    target = make_source(10, scope="target_calibration", observation="cz")[1]
    measured = common_measurement(inputs, target)
    posterior_path = tmp_path / "actual-source.json"
    inputs[0].save(posterior_path)
    posterior = SpatialSignalPosterior.load(posterior_path,
        expected_artifact_sha256=inputs[0].record()["artifact_sha256"])
    path = tmp_path / "measurement.json"
    measured.save(path)
    restored = ERPChannelEstimate.load(path, posterior=posterior, temporal_noise=inputs[3],
        expected_artifact_sha256=measured.record()["artifact_sha256"])
    np.testing.assert_array_equal(restored.mean, measured.mean)
    np.testing.assert_array_equal(restored.covariance_factor, measured.covariance_factor)
    np.testing.assert_array_equal(restored.noise_sd, measured.noise_sd)
    saved = json.loads(json.dumps(measured.record()))
    saved["receipt"]["physical_transport"]["channel_map"][0][0] = 2.
    transport = saved["receipt"]["physical_transport"]
    transport["artifact_sha256"] = metadata_digest({k: v for k, v in transport.items() if k != "artifact_sha256"})
    saved["artifact_sha256"] = metadata_digest({k: v for k, v in saved.items() if k != "artifact_sha256"})
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="transport"):
        ERPChannelEstimate.load(path, posterior=posterior, temporal_noise=inputs[3],
            expected_artifact_sha256=saved["artifact_sha256"])


def test_cross_reference_source_forecast_target_prior_and_frozen_response_replay(tmp_path):
    target_inputs = make_source(10, scope="target_calibration", observation="cz")
    target, effect, _, _, blocks = target_inputs
    sources = [make_source(0), make_source(1, channels=("Pz", "Fz", "Cz"),
        acquisition_reference="left earlobe"), make_source(2, channels=("Cz", "Pz", "Fz"),
        acquisition_reference="nose")]
    measurements = [common_measurement(source, effect) for source in sources]
    program = learn(measurements)
    path = tmp_path / "source-group.json"
    program.save(path)
    restored_program = type(program).load(path, expected_artifact_sha256=program.record()["artifact_sha256"])
    prior = restored_program.spatial_prior(target.prior, effect=effect, completion="separable",
                                          target_participant_key="person-10")
    np.testing.assert_allclose(effect.functional @ prior.mean.ravel(order="F"), program.mean,
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(effect.functional @ np.kron(prior.signal_covariance, prior.temporal_covariance)
        @ effect.functional.T, program.forecast_covariance, rtol=1e-12, atol=1e-13)
    fitted = SpatialSignalPosterior.fit(prior, blocks, native_noise_covariance=target.native_noise_covariance,
                                        noise_binding=dict(target.noise_binding))
    path = tmp_path / "target.json"
    fitted.save(path)
    restored = SpatialSignalPosterior.load(path, expected_artifact_sha256=fitted.record()["artifact_sha256"])
    geometry = NoiseGeometry(effect.layout.channels, 128., "V", "right earlobe", "synthetic_native_raw", 3)
    ledger = [{"record_id": b.authorization["record_id"], "participant_key": "person-10",
        "recording_id": "target-recording", "source_sha256": b.authorization["source_sha256"],
        "read_support": list(b.authorization["raw_sample_support"]), "fit_scope": "target_calibration"} for b in blocks]
    response = FrozenERPResponse.from_spatial_posterior(restored, geometry=geometry,
        source_model_sha256="3" * 64, response_id="cross-reference-target", training_records=ledger)
    row_order = np.arange(12).reshape(4, 3, order="F").ravel()
    np.testing.assert_allclose(response.coefficient_factor @ response.coefficient_factor.T,
        fitted.covariance[np.ix_(row_order, row_order)], rtol=1e-12, atol=1e-13)
    score_noise = FittedChannelNoise.from_parameters(np.empty((3, 0)), np.eye(3), geometry=geometry,
        conditioning_samples=4, reference_id="independent_score_noise", family="shared")
    rng = np.random.default_rng(71)
    score = score_noise.score(rng.normal(size=(12, 3)), rng.normal(size=(1, 12, 4)), response,
        candidate_ids=("candidate",), geometry=geometry, record_id="new-held-record",
        recording_id="target-recording", sample_start=200)
    assert np.isfinite(score["log_density"]).all()
    assert score["parameters_updated"] is False and score["truth_used"] is False


def car_source_and_destination(count, source_kind, gain_mode):
    channels = ("Fz", "Cz", "Pz", "P3", "P4", "Oz", "PO8")[:count]
    layout = SpatialLayout(channels, "left earlobe", "1" * 64, ("EEG",) * count)
    basis = np.array([[1., 0.], [.7, .4], [.2, 1.], [0., .6]])
    weights = np.array([0., .25, .5, .25])
    binding = {"dictionary_sha256": "b" * 64, "actual_basis_sha256": array_digest(basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": 128.,
        "operator_identity": "synthetic_native_raw", "coefficient_groups": ["common", "target_increment"]}
    prior = SpatialSignalPrior(layout, np.zeros((4, count)), np.eye(4), .2 * np.eye(count), binding,
        {"role": "predeclared", "authority_sha256": "c" * 64})
    temporal = TemporalNoise.from_parameters([], innovation_variance=1., reference_id="CAR-source-noise")
    source_observation = (SpatialObservation.native(layout) if source_kind == "native"
                          else SpatialObservation.average_reference(layout))
    source_effect = ERPPhysicalEffect.from_prior(prior, basis=basis, waveform_weights=weights,
        grid_seconds=np.arange(4) / 128., observation=SpatialObservation.native(layout))
    rng = np.random.default_rng(20113 + count)
    design = rng.integers(-2, 3, size=(14, 4)).astype(float)
    noise = np.eye(count) + .1 * np.ones((count, count))
    samples = design @ rng.normal(size=(4, count)) + rng.multivariate_normal(np.zeros(count), noise, size=14)
    authorization = {"record_id": f"source-CAR-{count}", "fit_scope": "source_training",
        "source_sha256": "2" * 64, "dictionary_sha256": "b" * 64,
        "temporal_program_sha256": temporal.record()["artifact_sha256"], "raw_sample_support": [0, 14],
        "original_view": {"participant_key": f"CAR-person-{count}"}}
    block = SpatialResponseBlock(source_observation.project(samples), design, source_observation, authorization)
    posterior = SpatialSignalPosterior.fit(prior, (block,), native_noise_covariance=noise,
                                          noise_binding={"authority_sha256": "d" * 64})
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    destination_layout = replace(layout, channels=channels[::-1], acquisition_reference="right earlobe",
                                 authority_sha256="3" * 64)
    destination_prior = replace(prior, layout=destination_layout)
    gains = (np.ones(count) if gain_mode == "unit" else np.array([1., *([1e-20] * (count - 1))])
             if gain_mode == "extreme" else np.array([(-1.) ** i * (i + 1) / 3 for i in range(count)]))
    destination_observation = SpatialObservation.average_reference(destination_layout).with_output_gains(gains)
    destination = ERPPhysicalEffect.from_prior(destination_prior, basis=basis, waveform_weights=weights,
        grid_seconds=np.arange(4) / 128., observation=destination_observation)
    return (posterior, source_effect, dependence, temporal, (block,)), destination


@pytest.mark.parametrize("count", [3, 6, 7])
@pytest.mark.parametrize("source_kind", ["native", "car"])
@pytest.mark.parametrize("gain_mode", ["unit", "mixed", "extreme"])
def test_declared_car_generation_and_actual_source_blue_transport(count, source_kind, gain_mode, tmp_path):
    inputs, destination = car_source_and_destination(count, source_kind, gain_mode)
    assert all(sum(row, Fraction(0)) == 0 for row in destination.structural_spatial_map)
    measured = common_measurement(inputs, destination)
    mapping = destination.spatial_map[:, ::-1]
    functional = np.kron(mapping, inputs[1].response_functional[None, :])
    x, y = np.vstack(inputs[0].information_roots), np.concatenate(inputs[0].responses)
    oracle_factor = functional @ np.linalg.pinv(x)
    # Compare in measurement units; a tiny output gain must retain its direction.
    root = np.linalg.cholesky(oracle_factor @ oracle_factor.T)
    np.testing.assert_allclose(np.linalg.solve(root, measured.mean - oracle_factor @ y), 0., atol=2e-12)
    actual_covariance = np.linalg.solve(root, measured.covariance)
    np.testing.assert_allclose(np.linalg.solve(root, actual_covariance.T).T, np.eye(count - 1), atol=2e-12)
    expected_noise = np.sqrt(np.sum(destination.waveform_weights ** 2)
                             * np.diag(mapping @ inputs[0].native_noise_covariance @ mapping.T))
    np.testing.assert_allclose(measured.noise_sd, expected_noise, rtol=2e-14, atol=0.)
    source_path = tmp_path / "source.json"
    inputs[0].save(source_path)
    source = SpatialSignalPosterior.load(source_path, expected_artifact_sha256=inputs[0].record()["artifact_sha256"])
    path = tmp_path / "effect.json"
    measured.save(path)
    restored = ERPChannelEstimate.load(path, posterior=source, temporal_noise=inputs[3],
        expected_artifact_sha256=measured.record()["artifact_sha256"])
    np.testing.assert_array_equal(restored.mean, measured.mean)
    np.testing.assert_array_equal(restored.covariance_factor, measured.covariance_factor)
    np.testing.assert_array_equal(restored.noise_sd, measured.noise_sd)
    if source_kind == "car":
        assert source.structural_binding["observation_coordinates"] == ("declared_CAR_exact_EEG_differences",)


def test_car_generation_authority_does_not_authorize_arbitrary_tiny_native_map():
    inputs, destination = car_source_and_destination(3, "native", "unit")
    assert any(sum((Fraction.from_float(float(value)) for value in row), Fraction(0))
               for row in destination.spatial_map)
    common_measurement(inputs, destination)
    arbitrary = replace(destination, observation=SpatialObservation(destination.layout, destination.spatial_map,
        ("support-0", "support-1"), destination.observation.output_reference))
    with pytest.raises(SpatialPriorIdentificationError, match="reference_component_not_observed"):
        common_measurement(inputs, arbitrary)


@pytest.mark.parametrize("mutation", ["kind", "complete_observation"])
def test_resigned_source_observation_kind_cannot_grant_physical_transport_capability(mutation):
    inputs = make_source(0)
    _, source_effect, _, temporal, blocks = inputs
    observation = SpatialObservation(source_effect.layout, np.array([[1., 3., -4.]]),
                                     ("generic-contrast",), "generic_reference_free")
    generic_blocks = tuple(replace(block, samples=block.samples @ observation.matrix.T,
                                   observation=observation) for block in blocks)
    posterior = SpatialSignalPosterior.fit(inputs[0].prior, generic_blocks,
        native_noise_covariance=inputs[0].native_noise_covariance, noise_binding=dict(inputs[0].noise_binding))
    destination = make_source(10, scope="target_calibration", observation="cz")[1]
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    with pytest.raises(SpatialPriorIdentificationError, match="requires_declared_observation"):
        common_measurement((posterior, source_effect, dependence, temporal, generic_blocks), destination)
    saved = json.loads(json.dumps(posterior.record()))
    for record in saved["fit_records"]:
        if mutation == "kind":
            declaration = record["observation"]["declaration"]
            declaration.update(kind="native_EEG", native_channels=list(source_effect.layout.channels), row_gains=[1.])
        else:
            record["observation"] = SpatialObservation.native(source_effect.layout).record()
    saved["fit_sha256"] = metadata_digest(saved["fit_records"])
    saved["artifact_sha256"] = metadata_digest({key: value for key, value in saved.items() if key != "artifact_sha256"})
    restored = SpatialSignalPosterior.from_record(saved)
    dependence = SpatialLikelihoodDependence.independent(restored, authority_sha256="e" * 64)
    with pytest.raises(ValueError, match="(?:actual dimensions|structural operator dimensions)"):
        common_measurement((restored, source_effect, dependence, temporal, generic_blocks), destination)
