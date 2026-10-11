"""Complete source-effects/group-prior chain from owned synthetic C artifacts."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from research.channel_noise_mechanism import FrozenERPResponse, NoiseGeometry
from transfer.noise_model import array_digest, metadata_digest
from transfer.spatial_signal_prior import (
    ERPChannelEstimate,
    ERPPhysicalEffect,
    SourceGroupPolicy,
    SourceGroupPrior,
    SpatialLayout,
    SpatialLikelihoodDependence,
    SpatialObservation,
    SpatialPriorIdentificationError,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)
from transfer.temporal_noise import TemporalNoise


def make_source(person=0, *, rank_case="full", scope="source_training", observation="native", ar=0., unit_scale=1.,
                zero_sample=False):
    rng = np.random.default_rng(20261011 + person)
    layout = SpatialLayout(("Fz", "Cz", "Pz"), "right earlobe", f"{person + 1:064x}", ("EEG",) * 3,
                           "V" if unit_scale == 1 else "uV")
    basis = np.array([[1., 0.], [.7, .4], [.2, 1.], [0., .6]])
    if zero_sample:
        basis[1] = 0
    weights = np.array([0., .25, .5, .25])
    noise = np.array([[1.5, .2, -.1], [.2, .8, .07], [-.1, .07, .6]]) * (1 + person / 10) * unit_scale ** 2
    temporal = TemporalNoise.from_parameters([] if ar == 0 else [ar],
        innovation_variance=1., reference_id=f"source-noise-{person}")
    binding = {"dictionary_sha256": "b" * 64, "actual_basis_sha256": array_digest(basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": 128.,
        "operator_identity": "synthetic_native_raw", "coefficient_groups": ["common", "target_increment"]}
    prior = SpatialSignalPrior(layout, np.zeros((4, 3)), np.eye(4), .2 * unit_scale ** 2 * np.eye(3), binding,
        {"role": "predeclared", "authority_sha256": "c" * 64})
    mapping = (SpatialObservation.native(layout) if observation == "native"
               else SpatialObservation.measured_reference(layout, "Cz"))
    effect = ERPPhysicalEffect.from_prior(prior, basis=basis, waveform_weights=weights,
        grid_seconds=np.arange(4) / 128., observation=mapping)
    beta = rng.normal(size=(4, 3)) * .15
    beta[2:] += np.array([.2, .5, .8]) * (1 + person / 8)
    beta *= unit_scale
    blocks = []
    for index in range(2):
        d = rng.normal(size=(18, 4))
        if rank_case == "local_deficient":
            d[:, [1, 3] if index == 0 else [0, 2]] = 0
        elif rank_case == "only_effect":
            d[:, :2] = 0
        elif rank_case == "true_null":
            d[:, 3] = 0
        elif rank_case == "weak":
            d[:, 3] *= 1e-16
        temporal_factor = np.linalg.solve(temporal.whiten(np.eye(18)), np.eye(18))
        y = d @ beta + temporal_factor @ rng.multivariate_normal(np.zeros(3), noise, size=18)
        auth = {"record_id": f"person-{person}:C{index}", "fit_scope": scope,
            "source_sha256": f"{person + 100:064x}", "dictionary_sha256": "b" * 64,
            "temporal_program_sha256": temporal.record()["artifact_sha256"],
            "raw_sample_support": [30 * index, 30 * index + 18],
            "original_view": {"participant_key": f"person-{person}"}}
        blocks.append(SpatialResponseBlock(mapping.project(temporal.whiten(y)), temporal.whiten(d), mapping, auth,
            temporal.whiten(np.column_stack((np.ones(18), np.linspace(-1, 1, 18))))))
    posterior = SpatialSignalPosterior.fit(prior, blocks, native_noise_covariance=noise,
                                          noise_binding={"authority_sha256": "d" * 64})
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    return posterior, effect, dependence, temporal, blocks


def estimate(inputs):
    posterior, effect, dependence, temporal, _ = inputs
    return posterior.estimate_effect(effect, dependence, temporal_noise=temporal)


def policy(width=3):
    vectors = (np.array([0., 1., 1.]), np.array([0., 1., -1.])) if width == 3 else (np.array([1., 1.]),)
    names = ("independent", "posterior", "central_posterior_contrast") if width == 3 else ("independent", "posterior")
    return SourceGroupPolicy(names,
        np.array([np.eye(width), *(np.outer(v, v) / (v @ v) for v in vectors)]),
        np.eye(width), np.full(len(names), .03), np.full(len(names), 100.),
        20., np.zeros(width), .3 * np.eye(width), 100., "f" * 64)


def learn(estimates, chosen_policy=None):
    authority = {"fit_scope": "source_training", "independent_unit": "participant",
        "participant_keys": [e.receipt["participant_key"] for e in estimates], "authority_sha256": "1" * 64}
    return SpatialSignalPrior.learn(mechanism="source_group", estimates=estimates,
        policy=policy(len(estimates[0].mean)) if chosen_policy is None else chosen_policy,
        source_authority=authority)


@pytest.mark.parametrize("rank_case", ["full", "local_deficient", "only_effect"])
def test_joint_estimable_effect_from_owned_likelihood_matches_dense_oracle(rank_case):
    inputs = make_source(rank_case=rank_case)
    posterior, effect, _, _, _ = inputs
    measured = estimate(inputs)
    x, y = np.vstack(posterior.information_roots), np.concatenate(posterior.responses)
    # Independent least-squares oracle used only in a test. The implementation
    # checks every F in the identified joint space and never manufactures nulls.
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    q, singular, vt = np.linalg.svd(x, full_matrices=False)
    rank = np.sum(singular > 1e-10)
    factor = effect.functional @ (vt[:rank].T / singular[:rank])
    np.testing.assert_allclose(measured.mean, effect.functional @ beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(measured.covariance, factor @ factor.T, rtol=1e-10, atol=1e-12)
    assert not measured.receipt["resolution"]["prior_information_used"]
    if rank_case == "only_effect":
        assert measured.receipt["resolution"]["joint_rank"] < posterior.mean.size
    if rank_case == "local_deficient":
        assert all(np.linalg.matrix_rank(r) < posterior.mean.size for r in posterior.information_roots)
        assert measured.receipt["resolution"]["joint_rank"] == posterior.mean.size
    assert np.linalg.norm(measured.mean - effect.functional @ posterior.mean.ravel(order="F")) > 1e-3


@pytest.mark.parametrize("rank_case", ["true_null", "weak"])
def test_true_null_effect_is_unsupported_and_no_raw_array_entry(rank_case):
    inputs = make_source(rank_case=rank_case)
    with pytest.raises(SpatialPriorIdentificationError, match="not_identified_by_joint_C"):
        estimate(inputs)
    with pytest.raises(TypeError):
        ERPChannelEstimate(np.ones(3), np.eye(3))
    with pytest.raises(ValueError):
        SpatialSignalPrior.learn(mechanism="dprime", estimates=(), policy=policy(), source_authority={})


def test_full_dependency_factor_is_bound_to_actual_compressed_frame():
    posterior, effect, independent, temporal, blocks = make_source()
    rows = len(independent.factor)
    covariance = np.eye(rows)
    split = rows // 2
    covariance[:split, split:] = .6 * np.eye(split)
    covariance[split:, :split] = .6 * np.eye(split)
    dependence = SpatialLikelihoodDependence.from_factor(posterior, np.linalg.cholesky(covariance),
        authority={"authority_sha256": "2" * 64, "law": "external_full_compressed_errors"})
    result = posterior.estimate_effect(effect, dependence, temporal_noise=temporal)
    x, y = np.vstack(posterior.information_roots), np.concatenate(posterior.responses)
    precision_x = np.linalg.solve(covariance, x)
    information = x.T @ precision_x
    expected_covariance = effect.functional @ np.linalg.solve(information, effect.functional.T)
    expected_mean = effect.functional @ np.linalg.solve(information, precision_x.T @ y)
    np.testing.assert_allclose(result.mean, expected_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.covariance, expected_covariance, rtol=1e-10, atol=1e-12)
    swapped = SpatialSignalPosterior.fit(posterior.prior, tuple(reversed(blocks)),
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    with pytest.raises(ValueError, match="rows/frame"):
        swapped.estimate_effect(effect, dependence, temporal_noise=temporal)


def test_source_policy_scope_requested_cohort_and_whole_program_bounds():
    estimates = [estimate(make_source(i)) for i in range(3)]
    with pytest.raises(ValueError, match="unique"):
        learn((estimates[0], estimates[0]))
    with pytest.raises(ValueError, match="scope"):
        learn([estimates[0], estimate(make_source(99, scope="target_calibration"))])
    with pytest.raises(SpatialPriorIdentificationError, match="whole_source_program"):
        learn([estimates[0], estimate(make_source(90))],
              replace(policy(), component_ceiling=np.array([.03, .03, .03])))
    with pytest.raises(SpatialPriorIdentificationError, match="mean_scale"):
        learn(estimates, replace(policy(), maximum_mean_metric_energy=1e-12))
    wrong = {"fit_scope": "source_training", "independent_unit": "participant",
        "participant_keys": ["person-0", "person-1", "person-2", "missing-person"], "authority_sha256": "1" * 64}
    with pytest.raises(ValueError, match="complete actual"):
        SourceGroupPrior.fit(estimates, policy=policy(), source_authority=wrong)


def test_complete_source_forecast_mean_covariance_save_load_and_target_chain(tmp_path):
    sources = [make_source(i) for i in range(4)]
    estimates = [estimate(s) for s in sources]
    program = learn(estimates)
    assert program.fit_binding["component_source_weight"] == pytest.approx(3 / 23)
    assert len({e.receipt["native_noise_sha256"] for e in estimates}) == 4
    path = tmp_path / "source-program.json"
    program.save(path)
    replay = SourceGroupPrior.load(path, expected_artifact_sha256=program.record()["artifact_sha256"])
    np.testing.assert_array_equal(replay.mean, program.mean)
    np.testing.assert_array_equal(replay.mean_precision_root, program.mean_precision_root)
    target, effect, _, temporal, blocks = make_source(10, scope="target_calibration")
    control = replay.spatial_prior(target.prior, effect=effect, completion="mean_only", target_participant_key="person-10")
    prior = replay.spatial_prior(target.prior, effect=effect, completion="separable", target_participant_key="person-10")
    np.testing.assert_array_equal(control.signal_covariance, target.prior.signal_covariance)
    np.testing.assert_allclose(effect.functional @ prior.mean.ravel(order="F"), program.mean, rtol=1e-12)
    np.testing.assert_allclose(effect.functional @ np.kron(prior.signal_covariance, prior.temporal_covariance)
        @ effect.functional.T, program.forecast_covariance, rtol=1e-12, atol=1e-13)
    with pytest.raises(ValueError, match="independent declared"):
        replay.spatial_prior(target.prior, effect=effect, completion="separable", target_participant_key="person-0")
    fitted = SpatialSignalPosterior.fit(prior, blocks, native_noise_covariance=target.native_noise_covariance,
                                      noise_binding=dict(target.noise_binding))
    path = tmp_path / "target.json"
    fitted.save(path)
    restored = SpatialSignalPosterior.load(path, expected_artifact_sha256=fitted.record()["artifact_sha256"])
    assert restored.prior.source["source_learning"]["target_participant_key"] == "person-10"
    geometry = NoiseGeometry(effect.layout.channels, 128., "V", "right earlobe", "synthetic_native_raw", 3)
    ledger = [{"record_id": b.authorization["record_id"], "participant_key": "person-10",
        "recording_id": "target-recording", "source_sha256": b.authorization["source_sha256"],
        "read_support": list(b.authorization["raw_sample_support"]), "fit_scope": "target_calibration"} for b in blocks]
    response = FrozenERPResponse.from_spatial_posterior(restored, geometry=geometry,
        source_model_sha256="3" * 64, response_id="learned_target", training_records=ledger)
    order = [channel * 4 + row for row in range(4) for channel in range(3)]
    np.testing.assert_allclose(response.coefficient_factor @ response.coefficient_factor.T,
                              fitted.covariance[np.ix_(order, order)], rtol=1e-12, atol=1e-13)
    assert temporal.order == 0
    with pytest.raises(ValueError, match="independent declared target C"):
        wrong_blocks = [replace(b, authorization={**b.authorization,
            "original_view": {"participant_key": "person-0"}}) for b in blocks]
        SpatialSignalPosterior.fit(prior, wrong_blocks, native_noise_covariance=target.native_noise_covariance,
                                   noise_binding=dict(target.noise_binding))


def test_effect_definition_mismatch_and_measurement_roundtrip(tmp_path):
    inputs = make_source()
    posterior, effect, dependence, temporal, _ = inputs
    measured = estimate(inputs)
    path = tmp_path / "effect.json"
    measured.save(path)
    loaded = ERPChannelEstimate.load(path, posterior=posterior, temporal_noise=temporal,
                                    expected_artifact_sha256=measured.record()["artifact_sha256"])
    np.testing.assert_array_equal(loaded.mean, measured.mean)
    program = learn([measured, estimate(make_source(1)), estimate(make_source(2))])
    prior = posterior.prior
    doubled = replace(prior, design_binding={**prior.design_binding, "actual_basis_sha256": array_digest(2 * effect.basis)})
    changed = ERPPhysicalEffect.from_prior(doubled, basis=2 * effect.basis,
        waveform_weights=effect.waveform_weights, grid_seconds=effect.grid_seconds,
        observation=effect.observation)
    with pytest.raises(ValueError, match="definition differs"):
        program.spatial_prior(doubled, effect=changed, completion="mean_only", target_participant_key="new-target")
    with pytest.raises(ValueError, match="actual prior/design"):
        effect.validate_prior(doubled)


def test_joint_observed_reference_effect_and_explicit_proper_native_completion():
    sources = [make_source(i, rank_case="only_effect", observation="cz") for i in range(4)]
    estimates = [estimate(s) for s in sources]
    program = learn(estimates)
    assert len(program.mean) == 2
    target, effect, _, _, _ = make_source(11, scope="target_calibration", observation="cz")
    prior = program.spatial_prior(target.prior, effect=effect, completion="separable", target_participant_key="person-11")
    assert np.linalg.eigvalsh(prior.signal_covariance).min() > 0
    np.testing.assert_allclose(effect.functional @ np.kron(prior.signal_covariance, prior.temporal_covariance)
        @ effect.functional.T, program.forecast_covariance, rtol=1e-12, atol=1e-13)
    assert prior.source["source_learning"]["unseen_spatial_directions"] == "proper_reference_conditional_completion"


def test_saved_source_relationship_mutation_is_rejected_even_when_resigned(tmp_path):
    program = learn([estimate(make_source(i)) for i in range(3)])
    saved = program.record()
    for field in ("components", "mean", "between_covariance"):
        changed = copy.deepcopy(saved)
        if field == "between_covariance":
            changed[field][0][0] *= 2
        else:
            changed[field][0] += .5
        changed["artifact_sha256"] = metadata_digest({k: v for k, v in changed.items() if k != "artifact_sha256"})
        path = tmp_path / (field + ".json")
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            SourceGroupPrior.load(path, expected_artifact_sha256=changed["artifact_sha256"])


def test_effect_and_population_voltage_units_and_distinct_temporal_noise_laws():
    raw = [make_source(i, ar=.2 * (i % 2)) for i in range(3)]
    scaled = [make_source(i, ar=.2 * (i % 2), unit_scale=1e6) for i in range(3)]
    original_estimates, scaled_estimates = [estimate(s) for s in raw], [estimate(s) for s in scaled]
    for original, changed in zip(original_estimates, scaled_estimates, strict=True):
        np.testing.assert_allclose(changed.mean / 1e6, original.mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(changed.covariance / 1e12, original.covariance, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(changed.mean / changed.noise_sd, original.mean / original.noise_sd, rtol=1e-10)
    p = policy()
    changed_policy = replace(p, metric_covariance=p.metric_covariance * 1e12,
        component_anchor=p.component_anchor * 1e12, component_ceiling=p.component_ceiling * 1e12,
        mean_anchor=p.mean_anchor * 1e6, mean_anchor_covariance=p.mean_anchor_covariance * 1e12)
    program, changed_program = learn(original_estimates), learn(scaled_estimates, changed_policy)
    np.testing.assert_allclose(changed_program.mean / 1e6, program.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(changed_program.forecast_covariance / 1e12,
                              program.forecast_covariance, rtol=1e-10, atol=1e-12)


def test_explicit_gain_and_missing_functional_measurement_matches_physical_push():
    posterior, effect, dependence, temporal, _ = make_source()
    original = posterior.estimate_effect(effect, dependence, temporal_noise=temporal)
    observations = (
        SpatialObservation.native(effect.layout).with_output_gains([1e-3, -2., 1e3]),
        SpatialObservation(effect.layout, np.eye(3)[[0, 2]], ("Fz", "Pz"), "right earlobe"),
        SpatialObservation.average_reference(effect.layout),
    )
    for observation in observations:
        changed_effect = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
            waveform_weights=effect.waveform_weights, grid_seconds=effect.grid_seconds, observation=observation)
        changed = posterior.estimate_effect(changed_effect, dependence, temporal_noise=temporal)
        a = observation.support_basis.T @ observation.matrix
        np.testing.assert_allclose(changed.mean, a @ original.mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(changed.covariance, a @ original.covariance @ a.T, rtol=1e-10, atol=1e-12)
        assert len(changed.mean) == observation.rank_ledger["rank"]


def test_renamed_source_participant_cannot_duplicate_actual_c_support():
    posterior, effect, dependence, temporal, blocks = make_source()
    duplicate_blocks = [replace(b, authorization={**b.authorization,
        "record_id": b.authorization["record_id"] + ":renamed",
        "original_view": {"participant_key": "renamed-person"}}) for b in blocks]
    duplicate = SpatialSignalPosterior.fit(posterior.prior, duplicate_blocks,
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    duplicate_dependence = SpatialLikelihoodDependence.independent(duplicate, authority_sha256="e" * 64)
    original = posterior.estimate_effect(effect, dependence, temporal_noise=temporal)
    relabelled = duplicate.estimate_effect(effect, duplicate_dependence, temporal_noise=temporal)
    with pytest.raises(ValueError, match="duplicate actual physical C"):
        learn((original, relabelled))


def test_zero_functional_is_identified_and_roundtrips_as_exact_zero(tmp_path):
    posterior, effect, dependence, temporal, _ = make_source(zero_sample=True)
    weights = np.array([0., 1., 0., 0.])
    zero_effect = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis, waveform_weights=weights,
        grid_seconds=effect.grid_seconds, observation=effect.observation)
    measured = posterior.estimate_effect(zero_effect, dependence, temporal_noise=temporal)
    assert np.array_equal(measured.mean, np.zeros(3))
    assert np.array_equal(measured.covariance, np.zeros((3, 3)))
    path = tmp_path / "zero.json"
    measured.save(path)
    loaded = ERPChannelEstimate.load(path, posterior=posterior, temporal_noise=temporal,
                                    expected_artifact_sha256=measured.record()["artifact_sha256"])
    np.testing.assert_array_equal(loaded.mean, measured.mean)
