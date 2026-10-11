"""Complete source-effects/group-prior chain from owned synthetic C artifacts."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from research.channel_noise_mechanism import FittedChannelNoise, FrozenERPResponse, NoiseGeometry
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
    _exact_structural_null,
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
    with pytest.raises(SpatialPriorIdentificationError, match="(?:not_identified_by_joint_C|structural_C)"):
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
    # The real B05 frozen-response consumer also scores this learned state.
    # The score-time noise law is independent of the source or target fit law.
    rng = np.random.default_rng(41)
    y, d = rng.normal(size=(12, 3)), rng.normal(size=(12, 4))
    score_noise = FittedChannelNoise.from_parameters(np.empty((3, 0)), np.eye(3),
        geometry=geometry, conditioning_samples=4, reference_id="independent_score_noise", family="shared")
    scored = score_noise.score(y, d[None, ...], response, candidate_ids=("candidate",), geometry=geometry,
        record_id="new-held-record", recording_id="target-recording", sample_start=200)
    covariance = fitted.covariance[np.ix_(order, order)]
    x = np.kron(d, np.eye(3))
    marginal = np.eye(y.size) + x @ covariance @ x.T
    centered = y.ravel() - x @ fitted.mean.ravel()
    z = np.kron(np.column_stack((np.ones(len(y)), (2 * np.arange(len(y)) - 3) / 3)), np.eye(3))

    def flat_log_density(count):
        rows = 3 * count
        k, values, nuisance = marginal[:rows, :rows], centered[:rows], z[:rows]
        precision = np.linalg.inv(k)
        nuisance_information = nuisance.T @ precision @ nuisance
        residual_precision = precision - precision @ nuisance @ np.linalg.solve(
            nuisance_information, nuisance.T @ precision)
        return -.5 * ((rows - 6) * np.log(2 * np.pi) + np.linalg.slogdet(k)[1]
            + np.linalg.slogdet(nuisance_information)[1] + values @ residual_precision @ values)

    expected_density = flat_log_density(len(y)) - flat_log_density(4)
    np.testing.assert_allclose(scored["log_density"], [expected_density], rtol=2e-12, atol=1e-12)
    assert scored["truth_used"] is False and scored["parameters_updated"] is False
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


@pytest.mark.parametrize("null_weight", [1e-10, 1e-30])
def test_small_real_null_component_cannot_be_hidden_by_an_observed_functional(null_weight):
    posterior, effect, dependence, temporal, _ = make_source(rank_case="true_null")
    # The first basis row measures the identified target coefficient, while
    # the last row contributes ONLY the exactly unobserved target coefficient.
    weights = np.array([1., 0., 0., null_weight])
    effect = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
        waveform_weights=weights, grid_seconds=effect.grid_seconds, observation=effect.observation)
    assert np.all(np.vstack(posterior.information_roots)[:, [3, 7, 11]] == 0)
    assert np.all(effect.functional[:, [3, 7, 11]].diagonal() != 0)
    with pytest.raises(SpatialPriorIdentificationError, match="(?:not_identified_by_joint_C|structural_C)"):
        posterior.estimate_effect(effect, dependence, temporal_noise=temporal)


def test_learned_target_ownership_is_checked_by_saved_posterior_admission():
    program = learn([estimate(make_source(i)) for i in range(3)])
    target, effect, _, _, blocks = make_source(10, scope="target_calibration")
    prior = program.spatial_prior(target.prior, effect=effect, completion="separable",
        target_participant_key="person-10")
    fitted = SpatialSignalPosterior.fit(prior, blocks,
        native_noise_covariance=target.native_noise_covariance, noise_binding=dict(target.noise_binding))
    saved = fitted.record()
    for record in saved["fit_records"]:
        record["authorization"]["original_view"]["participant_key"] = "person-0"
    saved["fit_sha256"] = metadata_digest(saved["fit_records"])
    saved["artifact_sha256"] = metadata_digest({k: v for k, v in saved.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="independent declared target C"):
        SpatialSignalPosterior.from_record(saved)


def test_effect_measurement_rejects_resigned_dprime_relationship_mutation(tmp_path):
    posterior, effect, dependence, temporal, _ = make_source()
    measured = posterior.estimate_effect(effect, dependence, temporal_noise=temporal)
    saved = measured.record()
    saved["dprime_mean"][0] += .1
    saved["artifact_sha256"] = metadata_digest({k: v for k, v in saved.items() if k != "artifact_sha256"})
    path = tmp_path / "inconsistent-dprime.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="d-prime"):
        ERPChannelEstimate.load(path, posterior=posterior, temporal_noise=temporal,
            expected_artifact_sha256=saved["artifact_sha256"])


@pytest.mark.parametrize("field,value", [("fit_scope", "target_calibration"), ("independent_unit", "record")])
def test_population_load_rechecks_source_authority_relationship(tmp_path, field, value):
    program = learn([estimate(make_source(i)) for i in range(3)])
    saved = json.loads(json.dumps(program.record()))
    saved["fit_binding"]["source_authority"][field] = value
    saved["artifact_sha256"] = metadata_digest({k: v for k, v in saved.items() if k != "artifact_sha256"})
    path = tmp_path / "inconsistent-authority.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="source.*scope|source.*authority"):
        SourceGroupPrior.load(path, expected_artifact_sha256=saved["artifact_sha256"])


@pytest.mark.parametrize("rank_case", ["full", "only_effect", "local_deficient"])
def test_effect_blue_does_not_inherit_a_nonorthogonal_prior_chart(rank_case):
    posterior, effect, _, temporal, blocks = make_source(rank_case=rank_case)
    original = estimate((posterior, effect, SpatialLikelihoodDependence.independent(
        posterior, authority_sha256="e" * 64), temporal, blocks))
    temporal_root = np.array([[1., 0., 0., 0.], [.4, 2., 0., 0.],
        [-.2, .3, .7, 0.], [.5, -.1, .2, 1.3]])
    spatial_root = np.array([[1., 0., 0.], [.6, 2., 0.], [-.3, .4, .9]])
    prior = replace(posterior.prior, mean=np.full((4, 3), 30.),
        temporal_covariance=temporal_root @ temporal_root.T,
        signal_covariance=spatial_root @ spatial_root.T)
    changed = SpatialSignalPosterior.fit(prior, blocks,
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    measured = changed.estimate_effect(effect,
        SpatialLikelihoodDependence.independent(changed, authority_sha256="e" * 64), temporal_noise=temporal)
    np.testing.assert_allclose(measured.mean, original.mean, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(measured.covariance, original.covariance, rtol=2e-12, atol=2e-13)


def test_cz_measurement_does_not_authorize_unobserved_native_functionals():
    posterior, effect, dependence, temporal, _ = make_source(observation="cz")
    native = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
        waveform_weights=effect.waveform_weights, grid_seconds=effect.grid_seconds,
        observation=SpatialObservation.native(effect.layout))
    with pytest.raises(SpatialPriorIdentificationError, match="(?:not_identified_by_joint_C|structural_C)"):
        posterior.estimate_effect(native, dependence, temporal_noise=temporal)


def test_population_mean_uncertainty_and_new_person_covariance_match_dense_oracle():
    estimates = [estimate(make_source(i)) for i in range(4)]
    original = policy()
    scalar_policy = replace(original, template_names=("independent",), templates=np.eye(3)[None, ...],
        component_anchor=original.component_anchor[:1], component_ceiling=original.component_ceiling[:1])
    program = learn(estimates, scalar_policy)
    means = np.array([e.mean for e in estimates])
    errors = np.array([e.covariance for e in estimates])
    moment = np.cov(means, rowvar=False, ddof=1) - errors.mean(axis=0)
    raw_variance = max(np.trace(moment) / 3, 0.)
    weight = 3 / 23
    between = np.eye(3) * ((1 - weight) * .03 + weight * raw_variance)
    information = np.linalg.inv(scalar_policy.mean_anchor_covariance)
    rhs = information @ scalar_policy.mean_anchor
    for mean, error in zip(means, errors, strict=True):
        precision = np.linalg.inv(between + error)
        information += precision
        rhs += precision @ mean
    mean_covariance = np.linalg.inv(information)
    np.testing.assert_allclose(program.between_covariance, between, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(program.mean, mean_covariance @ rhs, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(program.mean_covariance, mean_covariance, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(program.forecast_covariance, between + mean_covariance, rtol=1e-12, atol=1e-14)
    assert np.linalg.norm(program.forecast_covariance - between) > .01


def test_effect_binding_cannot_reorder_common_and_increment_groups():
    posterior, effect, *_ = make_source()
    changed = replace(posterior.prior, design_binding={**posterior.prior.design_binding,
        "coefficient_groups": ["target_increment", "common"]})
    with pytest.raises(ValueError, match="coefficient groups"):
        effect.validate_prior(changed)


def test_dependent_nonzero_columns_separate_a_real_null_from_roundoff():
    posterior, effect, _, temporal, blocks = make_source()
    changed_blocks = []
    for block in blocks:
        d = block.design.copy()
        d[:, 3] = d[:, 2]
        changed_blocks.append(replace(block, design=d))
    posterior = SpatialSignalPosterior.fit(posterior.prior, changed_blocks,
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    dependency = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    for delta in (0., 1e-10):
        weights = np.array([1., 0., 0., (1. + delta) / .6])
        functional = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
            waveform_weights=weights, grid_seconds=effect.grid_seconds, observation=effect.observation)
        if delta:
            with pytest.raises(SpatialPriorIdentificationError, match="(?:not_identified_by_joint_C|structural_C)"):
                posterior.estimate_effect(functional, dependency, temporal_noise=temporal)
        else:
            measured = posterior.estimate_effect(functional, dependency, temporal_noise=temporal)
            x, y = np.vstack(posterior.information_roots), np.concatenate(posterior.responses)
            np.testing.assert_allclose(measured.mean, functional.functional @ np.linalg.lstsq(x, y, rcond=None)[0],
                rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("delta", [1e-14, 1e-12])
def test_exact_structural_null_rejects_tiny_effect_without_prior_fallback(delta):
    posterior, effect, _, temporal, blocks = make_source()
    changed_blocks = [replace(block, design=np.column_stack((block.design[:, :3], block.design[:, 2])))
                      for block in blocks]
    posterior = SpatialSignalPosterior.fit(posterior.prior, changed_blocks,
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    weights = np.array([1., 0., 0., (1. + delta) / .6])
    functional = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
        waveform_weights=weights, grid_seconds=effect.grid_seconds, observation=effect.observation)
    with pytest.raises(SpatialPriorIdentificationError, match="structural_C"):
        posterior.estimate_effect(functional, dependence, temporal_noise=temporal)


def test_exact_structural_null_budget_fails_closed():
    with pytest.raises(SpatialPriorIdentificationError, match="structural_identifiability_unresolved"):
        _exact_structural_null((np.zeros((1000, 300)),), 10)


def test_saved_structural_operator_and_null_certificate_cannot_be_resigned(tmp_path):
    posterior, _, _, _, _ = make_source()
    saved = posterior.record()
    saved["structural_null_basis"] = [[[1, 1]]]
    saved["artifact_sha256"] = metadata_digest({key: item for key, item in saved.items()
                                                 if key != "artifact_sha256"})
    path = tmp_path / "bad-null.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="structural C null basis"):
        SpatialSignalPosterior.from_record(saved)
    saved = posterior.record()
    saved["structural_operators"][0][0][0] += 1.
    saved["artifact_sha256"] = metadata_digest({key: item for key, item in saved.items()
                                                 if key != "artifact_sha256"})
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="structural C identifiability binding"):
        SpatialSignalPosterior.from_record(saved)


def test_fixed_cohort_outlier_is_not_capped_or_silently_dropped():
    inputs = [make_source(i) for i in range(3)]
    originals = [estimate(s) for s in inputs]
    posterior, effect, _, temporal, blocks = inputs[-1]
    # Same participants, actual C/design/noise, and declared policy throughout.
    # Add a physical effect to ONE participant, rather than changing n/weights.
    changed_blocks = [replace(block, samples=block.samples + 1e4 * block.design[:, 2, None]
        * np.ones((1, 3))) for block in blocks]
    changed_posterior = SpatialSignalPosterior.fit(posterior.prior, changed_blocks,
        native_noise_covariance=posterior.native_noise_covariance, noise_binding=dict(posterior.noise_binding))
    outlier = changed_posterior.estimate_effect(effect,
        SpatialLikelihoodDependence.independent(changed_posterior, authority_sha256="e" * 64), temporal_noise=temporal)
    chosen_policy = policy()
    accepted = learn(originals, chosen_policy)
    assert accepted.fit_binding["source_participant_keys"] == ("person-0", "person-1", "person-2")
    with pytest.raises(SpatialPriorIdentificationError, match="whole_source_program_exceeds_component_scale") as rejected:
        learn([*originals[:-1], outlier], chosen_policy)
    assert rejected.value.record()["participant_keys"] == ["person-0", "person-1", "person-2"]


def test_exact_zero_effect_cannot_learn_a_nonzero_proper_population_forecast():
    measurements = []
    for person in range(3):
        posterior, effect, dependence, temporal, _ = make_source(person, zero_sample=True)
        zero = ERPPhysicalEffect.from_prior(posterior.prior, basis=effect.basis,
            waveform_weights=np.array([0., 1., 0., 0.]), grid_seconds=effect.grid_seconds,
            observation=effect.observation)
        measurements.append(posterior.estimate_effect(zero, dependence, temporal_noise=temporal))
    with pytest.raises(SpatialPriorIdentificationError, match="zero_effect"):
        learn(measurements)


@pytest.mark.parametrize("field,value", [
    ("component_source_weight", 0.), ("component_fit_residual", 12345.),
    ("variance_noise_dependency_uncertainty_integrated", True), ("forecast", "between_only"),
])
def test_source_group_load_rejects_resigned_derived_semantics(field, value, tmp_path):
    program = learn([estimate(make_source(i)) for i in range(3)])
    saved = json.loads(json.dumps(program.record()))
    saved["fit_binding"][field] = value
    saved["artifact_sha256"] = metadata_digest({key: item for key, item in saved.items()
                                                 if key != "artifact_sha256"})
    path = tmp_path / (field + ".json")
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic|component|NNLS|forecast"):
        SourceGroupPrior.load(path, expected_artifact_sha256=saved["artifact_sha256"])


@pytest.mark.parametrize("field,value", [
    ("completion", "mean_only"), ("variance_uncertainty_integrated", True),
    ("unseen_spatial_directions", "zero_filled"), ("source_forecast_uncertainty_propagated", False),
    ("effect_definition_sha256", "not-a-sha"),
])
def test_target_source_learning_load_rejects_resigned_completion_semantics(field, value):
    program = learn([estimate(make_source(i)) for i in range(3)])
    target, effect, _, _, _ = make_source(10, scope="target_calibration")
    learned = program.spatial_prior(target.prior, effect=effect, completion="separable",
        target_participant_key="person-10")
    saved = json.loads(json.dumps(learned.record()))
    saved["source"]["source_learning"][field] = value
    saved["artifact_sha256"] = metadata_digest({key: item for key, item in saved.items()
                                                 if key != "artifact_sha256"})
    with pytest.raises(ValueError, match="semantic|completion|source-learned"):
        SpatialSignalPrior.from_record(saved)
