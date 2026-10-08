"""Independent dense Gaussian and adversarial controls for the B05 seam."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import (
    FittedChannelNoise,
    FrozenERPResponse,
    NoiseGeometry,
    ResidualRecord,
    known_covariance_discriminant_efficiency,
)
from transfer.noise_model import metadata_digest
from transfer.temporal_noise import TemporalNoise


def _geometry(unit="V", **changes):
    state = {"channels": ("Cz", "Pz"), "sfreq": 128., "unit": unit,
             "reference": "native_declared", "operator_identity": "raw_identity",
             "observation_rank": 2}
    return NoiseGeometry(**(state | changes))


def _response(geometry, *, uncertainty=True, scale=1.):
    mean = np.array([[.6, -.1], [.3, .5]]) * scale
    factor = np.diag([.12, .2, .15, .1]) * scale if uncertainty else np.empty((4, 0))
    return FrozenERPResponse(mean, factor, geometry, "a" * 64, "frozen_ERP",
                             {"fit_scope": "analytic_reference"})


def _private(geometry=None, *, scale=1., prefix=8):
    return FittedChannelNoise.from_parameters([[.7], [-.35]],
        np.array([[1., .3], [.3, 1.4]]) * scale ** 2,
        geometry=geometry or _geometry(), conditioning_samples=prefix, reference_id="known_noise")


def _ar1_covariance(coefficients, innovations, n):
    """Closed-form diagonal VAR1 covariance; independent of production compiler."""
    phi = np.asarray(coefficients)
    stationary = innovations / (1 - phi[:, None] * phi[None, :])
    blocks = np.empty((n * len(phi), n * len(phi)))
    for t in range(n):
        for s in range(n):
            covariance = np.diag(phi ** abs(t - s)) @ stationary
            blocks[t * len(phi):(t + 1) * len(phi), s * len(phi):(s + 1) * len(phi)] = (
                covariance if t >= s else covariance.T)
    return blocks


def _dense_log_density(y, design, response, covariance, prefix):
    """Integrate proper kernel covariance, then condition diffuse drift on prefix."""
    n, c = y.shape
    mean = (design @ response.mean).ravel()
    factor = np.einsum("nk,kcr->ncr", design,
        response.coefficient_factor.reshape(response.mean.shape[0], c, -1)).reshape(n * c, -1)
    total = covariance + factor @ factor.T
    drift = np.column_stack((np.ones(n), (2 * np.arange(n) - (prefix - 1)) / (prefix - 1)))
    nuisance = np.kron(drift, np.eye(c))
    cut = prefix * c
    pp, tp, tt = total[:cut, :cut], total[cut:, :cut], total[cut:, cut:]
    zp, zt = nuisance[:cut], nuisance[cut:]
    residual = y.ravel() - mean
    solved_z, solved_y = np.linalg.solve(pp, zp), np.linalg.solve(pp, residual[:cut])
    drift_covariance = np.linalg.inv(zp.T @ solved_z)
    drift_mean = drift_covariance @ zp.T @ solved_y
    conditional_nuisance = zt - tp @ solved_z
    conditional_mean = tp @ solved_y + conditional_nuisance @ drift_mean
    conditional_covariance = (tt - tp @ np.linalg.solve(pp, tp.T)
                              + conditional_nuisance @ drift_covariance @ conditional_nuisance.T)
    root = np.linalg.cholesky((conditional_covariance + conditional_covariance.T) / 2)
    standardized = solve_triangular(root, residual[cut:] - conditional_mean, lower=True)
    return -.5 * (standardized @ standardized + 2 * np.log(np.diag(root)).sum()
                  + (n - prefix) * c * np.log(2 * np.pi))


@pytest.mark.parametrize("uncertainty", [False, True])
@pytest.mark.parametrize("family", ["shared", "private"])
def test_conditional_prediction_matches_independent_dense_oracle(uncertainty, family):
    rng = np.random.default_rng(20260918)
    geometry = _geometry()
    coefficients = [[.7], [.7]] if family == "shared" else [[.7], [-.35]]
    covariance = np.array([[1., .3], [.3, 1.4]])
    noise = FittedChannelNoise.from_parameters(coefficients, covariance, geometry=geometry,
        conditioning_samples=8, family=family, reference_id="dense_oracle")
    response = _response(geometry, uncertainty=uncertainty)
    designs = rng.normal(size=(3, 30, 2))
    samples = rng.normal(size=(30, 2))
    score = noise.score(samples, designs, response, candidate_ids=("a", "b", "c"),
        geometry=geometry, record_id="held", recording_id="held_raw", sample_start=0)
    dense_covariance = _ar1_covariance(np.asarray(coefficients)[:, 0], covariance, 30)
    expected = [_dense_log_density(samples, d, response, dense_covariance, 8) for d in designs]
    np.testing.assert_allclose(score["log_density"], expected, rtol=2e-11, atol=2e-10)
    assert score["truth_used"] is False and score["parameters_updated"] is False
    assert score["conditioning_support"] == [0, 8]
    assert score["score_support"] == [8, 30]


def test_private_correlated_innovations_are_not_separable():
    noise = _private()
    covariance = _ar1_covariance([.7, -.35], noise.innovation_covariance, 10)
    block0, block1 = covariance[:2, :2], covariance[2:4, :2]
    best_scalar = np.sum(block0 * block1) / np.sum(block0 ** 2)
    assert np.linalg.norm(block1 - best_scalar * block0) > .5 * np.linalg.norm(block1)
    np.testing.assert_allclose(noise.compiled_prefix_root @ noise.compiled_prefix_root.T,
                              covariance[:16, :16], rtol=2e-12, atol=2e-12)
    assert noise.record()["prediction_family"] == "general_vec_Gaussian_not_private_MN_row_posterior"


def test_mapping_must_apply_to_observations_and_all_design_terms():
    noise = _private()
    rng = np.random.default_rng(17)
    y, d, b = rng.normal(size=(55, 2)), rng.normal(size=(55, 3)), rng.normal(size=(3, 2))
    np.testing.assert_allclose(noise.whiten(y - d @ b), noise.whiten(y) - noise.whiten(d @ b),
                              atol=3e-14)
    assert np.linalg.norm(noise.whiten(y) - d @ b - noise.whiten(y - d @ b)) > 1
    changed = y.copy()
    changed[7, 0] += 2
    assert noise.whiten(changed)[8, 0] != noise.whiten(y)[8, 0]


def test_correct_invertible_map_preserves_information_but_wrong_noise_weighting_hurts():
    delta, covariance = np.ones(2), np.diag([1., 100.])
    exact = known_covariance_discriminant_efficiency(delta, covariance, covariance)
    wrong = known_covariance_discriminant_efficiency(delta, covariance, np.diag([100., 1.]))
    transform = np.array([[2., 1.], [0., .4]])
    mapped = known_covariance_discriminant_efficiency(transform @ delta,
        transform @ covariance @ transform.T, transform @ covariance @ transform.T)
    assert exact["efficiency"] == pytest.approx(1.)
    assert wrong["efficiency"] < .02
    assert mapped["oracle_mahalanobis_squared"] == pytest.approx(exact["oracle_mahalanobis_squared"])


def test_actual_noise_estimator_can_weight_the_wrong_direction_under_residual_mismatch():
    # These are deliberately misspecified training residuals. The known held
    # covariance is different; this is not evidence of an iid estimation bias.
    rng = np.random.default_rng(20260918)
    residual = rng.multivariate_normal(np.zeros(2), np.diag([100., 1.]), size=2048)
    fitted = FittedChannelNoise.fit([_record(residual, "misspecified_residual")], family="private", order=0)
    efficiency = known_covariance_discriminant_efficiency(np.ones(2), np.diag([1., 100.]),
                                                          fitted.innovation_covariance)
    assert efficiency["efficiency"] < .03
    diagnostic = fitted.diagnose([_record(residual, "misspecified_residual")], max_lag=8)
    assert np.max(np.abs(np.asarray(diagnostic["standardized_innovation"]["acf_record_equal"])[1:])) < .1
    assert fitted.provenance["estimator_is_joint_MLE"] is False


def test_diffuse_nuisance_can_erase_erp_contrast_while_noise_map_is_invertible():
    geometry = _geometry()
    noise = _private(geometry)
    response = FrozenERPResponse(np.ones((1, 2)), np.empty((2, 0)), geometry,
                                 "a" * 64, "constant_ERP", {"fit_scope": "analytic_reference"})
    designs = np.stack((np.zeros((40, 1)), np.ones((40, 1))))
    y = np.random.default_rng(8).normal(size=(40, 2))
    score = noise.score(y, designs, response, candidate_ids=("a", "b"), geometry=geometry,
                       record_id="held", recording_id="held_raw", sample_start=0)
    np.testing.assert_allclose(score["probabilities"], [.5, .5], atol=2e-12)
    assert score["signal_geometry"][0]["fitted_noise_mean_contrast_squared"] > 0
    assert score["signal_geometry"][0]["diffuse_retention_fraction"] < 1e-25


def test_voltage_unit_changes_preserve_scores_if_all_physical_states_change_together():
    rng = np.random.default_rng(11)
    designs, samples = rng.normal(size=(2, 44, 2)), rng.normal(size=(44, 2))
    scores = []
    for unit, scale in (("V", 1.), ("uV", 1e6)):
        geometry = _geometry(unit)
        scores.append(_private(geometry, scale=scale).score(samples * scale, designs,
            _response(geometry, scale=scale), candidate_ids=("a", "b"), geometry=geometry,
            record_id="held", recording_id="held_raw", sample_start=9))
    np.testing.assert_allclose(scores[0]["log_probability"], scores[1]["log_probability"], atol=2e-10)
    np.testing.assert_allclose(np.asarray(scores[0]["log_density"]) - np.asarray(scores[1]["log_density"]),
                              (44 - 8) * 2 * np.log(1e6), atol=2e-9)
    with pytest.raises(ValueError, match="held units"):
        _private().score(samples, designs, _response(_geometry()), candidate_ids=("a", "b"),
                         geometry=_geometry("uV"), record_id="held", recording_id="held_raw", sample_start=0)
    with pytest.raises(ValueError, match="full-rank"):
        _geometry(observation_rank=1, reference="CAR_two_channels")


def _record(values, record_id, *, geometry=None, sample_start=0, recording_id=None, role="target_calibration"):
    geometry = geometry or _geometry()
    return ResidualRecord.from_arrays(samples=values, design=np.zeros((len(values), 1)),
        kernel_mean=np.zeros((1, 2)), nuisance=np.zeros_like(values), record_id=record_id,
        participant_key="explicit_origin/A01", recording_id=recording_id or record_id,
        sample_start=sample_start, geometry=geometry, role=role,
        source_model_sha256="a" * 64, source_input_sha256="b" * 64)


def test_actual_segmented_fit_distinguishes_private_spectra_without_joining_records(tmp_path):
    rng = np.random.default_rng(20260918)
    covariance = np.array([[1., .3], [.3, 1.4]])
    phi = np.array([.8, -.4])
    stationary = covariance / (1 - phi[:, None] * phi[None, :])
    records = []
    for index in range(3):
        innovations = rng.multivariate_normal(np.zeros(2), covariance, size=800)
        y = np.empty_like(innovations)
        y[0] = rng.multivariate_normal(np.zeros(2), stationary)
        for t in range(1, len(y)):
            y[t] = phi * y[t - 1] + innovations[t]
        records.append(_record(y, f"train_{index}"))
    private = FittedChannelNoise.fit(records, family="private", order=1)
    shared = FittedChannelNoise.fit(records, family="shared", order=1)
    np.testing.assert_allclose([t.coefficients[0] for t in private.temporal], phi, atol=.07)
    assert abs(shared.temporal[0].coefficients[0] - phi[0]) > .2
    assert private.record()["parameter_budget"]["AR_shape"] == 2
    assert shared.record()["parameter_budget"]["AR_shape"] == 1
    assert private.provenance["covariance_training_rows"] == 3 * (800 - 32)
    diagnostic = private.diagnose(records)
    assert diagnostic["standardized_innovation"]["record_lengths"] == [768] * 3
    assert diagnostic["selection_allowed"] is False
    path = tmp_path / "noise.json"
    private.save(path)
    loaded = FittedChannelNoise.load(path)
    assert loaded.record() == private.record()
    np.testing.assert_array_equal(loaded.whiten(records[0].residual), private.whiten(records[0].residual))
    with pytest.raises(ValueError):
        private.innovation_covariance.setflags(write=True)
    with pytest.raises(TypeError):
        private.provenance["fit_scope"] = "held_diagnostic"
    payload = json.loads(path.read_bytes())
    payload["state"]["innovation_covariance"][0][0] += .1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        FittedChannelNoise.load(path)


def test_fit_and_held_support_gates_preserve_reference_and_authority():
    y = np.random.default_rng(4).normal(size=(150, 2))
    record = _record(y, "train", sample_start=10, recording_id="raw_session")
    with pytest.raises(ValueError, match="held diagnostics"):
        FittedChannelNoise.fit([_record(y, "held", role="held_diagnostic")], family="private", order=1)
    with pytest.raises(ValueError, match="overlap"):
        FittedChannelNoise.fit([record, _record(y, "train2", sample_start=80,
                              recording_id="raw_session")], family="shared", order=1)
    noise = FittedChannelNoise.fit([record], family="private", order=1)
    with pytest.raises(ValueError, match="overlap"):
        noise.score(y, np.zeros((2, len(y), 2)), _response(_geometry()),
            candidate_ids=("a", "b"), geometry=_geometry(), record_id="different_alias",
            recording_id="raw_session", sample_start=20)
    with pytest.raises(ValueError, match="geometry differs"):
        FittedChannelNoise.fit([record, _record(y, "other", geometry=_geometry(reference="Cz"))],
                               family="private", order=1)
    with pytest.raises(ValueError, match="finite real"):
        _record(y.astype(complex) + 1j, "complex")


def test_frozen_array_consumer_preserves_original_producer_and_validates_compiled_state():
    record = _record(np.random.default_rng(8).normal(size=(200, 2)), "actual_fit")
    temporal = TemporalNoise.fit([record.residual], order=1, record_ids=[record.record_id],
                                 fit_scope="target_calibration", startup_mode="stationary_innovations")
    covariance = np.array([[1., .3], [.3, 1.4]])
    original = FittedChannelNoise.from_temporal_noise(temporal, covariance, geometry=_geometry(),
        conditioning_samples=32, source_model_sha256="c" * 64, training_records=[record])
    kwargs = {"geometry": _geometry(), "conditioning_samples": 32, "source_model_sha256": "c" * 64,
              "actual_temporal_record": temporal.record(), "training_records": [record]}
    coefficients = np.asarray([temporal.coefficients] * 2)
    reflections = np.asarray([temporal.reflection_coefficients] * 2)
    consumer = FittedChannelNoise.from_frozen_shared_arrays(coefficients, reflections, covariance,
        original.compiled_prefix_root, **kwargs)
    assert consumer.provenance["actual_temporal_record"] == original.provenance["actual_temporal_record"]
    np.testing.assert_array_equal(consumer.whiten(record.residual), original.whiten(record.residual))
    wrong_root = original.compiled_prefix_root * 1.2
    with pytest.raises(ValueError, match="stationary noise model"):
        FittedChannelNoise.from_frozen_shared_arrays(coefficients, reflections, covariance, wrong_root, **kwargs)
    with pytest.raises(ValueError, match="actual frozen producer"):
        FittedChannelNoise.from_frozen_shared_arrays(coefficients + .01, reflections, covariance,
                                                     original.compiled_prefix_root, **kwargs)


def test_frozen_erp_training_support_is_checked_even_with_known_noise_oracle():
    geometry = _geometry()
    response = FrozenERPResponse(np.ones((1, 2)), np.empty((2, 0)), geometry, "a" * 64,
        "actual_fitted_ERP", {"fit_scope": "target_calibration", "records": [{"record_id": "old_id",
        "participant_key": "explicit_origin/A01", "recording_id": "raw_session", "read_support": [0, 100]}]})
    with pytest.raises(ValueError, match="frozen ERP producer"):
        _private().score(np.ones((40, 2)), np.ones((2, 40, 1)), response,
            candidate_ids=("a", "b"), geometry=geometry, record_id="new_alias",
            recording_id="raw_session", sample_start=50)


def test_compiled_prefix_relation_protects_weak_covariance_directions():
    noise = FittedChannelNoise.from_parameters(np.zeros((2, 1)), np.diag([1., 1e-16]),
        geometry=_geometry(), conditioning_samples=8, reference_id="weak_direction", family="shared")
    root = noise.compiled_prefix_root.copy()
    root[1::2] *= 1.2
    # A raw Frobenius covariance check would ignore this relative change.
    assert np.linalg.norm(root @ root.T - noise.compiled_prefix_root @ noise.compiled_prefix_root.T) < 1e-12
    with pytest.raises(ValueError, match="stationary noise model"):
        FittedChannelNoise(noise.temporal, noise.innovation_covariance, noise.geometry,
                           8, "shared", dict(noise.provenance), root)


def test_matrix_normal_response_conversion_uses_physical_row_major_covariance():
    row, spatial = np.array([[.5, .1], [.1, .3]]), np.array([[1., .2], [.2, 2.]])
    response = FrozenERPResponse.from_matrix_normal(np.zeros((2, 2)), row, spatial,
        geometry=_geometry(), source_model_sha256="a" * 64, response_id="actual_MN",
        provenance={"fit_scope": "analytic_reference"})
    np.testing.assert_allclose(response.coefficient_factor @ response.coefficient_factor.T,
                              np.kron(row, spatial), rtol=1e-14, atol=1e-14)


def test_bound_cli_scores_real_file_operands_and_refuses_changed_source(tmp_path):
    script = Path(__file__).resolve().parents[2] / "experiments/audit_channel_noise_mechanism.py"
    specification = importlib.util.spec_from_file_location("b05_audit_cli", script)
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    rng = np.random.default_rng(20260918)
    response = _response(_geometry(), uncertainty=False)
    source = tmp_path / "original_model.json"
    source.write_text('{"actual_fixture_source":"analytic_known_mean"}', encoding="utf-8")
    state = tmp_path / "state.npz"
    np.savez(state, mean=response.mean, factor=response.coefficient_factor)
    training = tmp_path / "train.npz"
    np.savez(training, samples=rng.normal(size=(180, 2)), design=np.zeros((180, 2)), nuisance=np.zeros((180, 2)))
    held = tmp_path / "held.npz"
    designs = rng.normal(size=(2, 100, 2))
    np.savez(held, samples=rng.normal(size=(100, 2)) + designs[0] @ response.mean,
             candidate_designs=designs)
    def bound(path, **keys):
        return {"path": path.name, "sha256": cli._sha(path), **keys}
    manifest = {"schema": "n2p3_channel_noise_audit_input/1",
        "evidence_scope": "public_calibration_development", "geometry": _geometry().record(),
        "conditioning_samples": 32, "fit_scope": "target_calibration", "max_diagnostic_lag": 12,
        "responses": [{"response_id": "anchor", "fit_scope": "analytic_reference",
            "source_model": bound(source), "exporter_identity": {"implementation_sha256": "a" * 64},
            "arrays": bound(state, mean_key="mean", coefficient_factor_key="factor")}],
        "training": [{"record_id": "train", "participant_key": "A01", "recording_id": "raw_train",
            "sample_start": 0, "arrays": bound(training, samples_key="samples", design_key="design", nuisance_key="nuisance")}],
        "noise_arms": [{"arm_id": "shared1", "family": "shared", "order": 1, "residual_response_id": "anchor"},
                       {"arm_id": "private1", "family": "private", "order": 1, "residual_response_id": "anchor"}],
        "held": [{"record_id": "held", "recording_id": "raw_held", "sample_start": 0,
                  "participant_key": "A01", "candidate_ids": ["a", "b"], "target_candidate": "a",
                  "arrays": bound(held, samples_key="samples", candidate_designs_key="candidate_designs")}]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = cli.audit(path, tmp_path / "audit")
    assert report["status"] == "completed" and len(report["cells"]) == 2
    assert all(cell["saved_noise_score_replay_exact"] for cell in report["cells"])
    assert report["no_held_labels_in_fit_or_score"] is True
    assert report["metrics"]["private1::anchor"]["participants"]["A01"]["requested_decisions"] == 1
    source.write_text("changed producer bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="byte identity"):
        cli.audit(path, tmp_path / "changed_audit")
    assert not (tmp_path / "changed_audit").exists()
    assert metadata_digest(manifest) != metadata_digest(report)


def _diagnostic_edge_case_manifest(tmp_path, *, constant_training):
    script = Path(__file__).resolve().parents[2] / "experiments/audit_channel_noise_mechanism.py"
    specification = importlib.util.spec_from_file_location("b05_diagnostic_cli", script)
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    rng = np.random.default_rng(20260918)
    response = _response(_geometry(), uncertainty=False)
    source, state = tmp_path / "source.json", tmp_path / "state.npz"
    source.write_text('{"scope":"synthetic_known_response_software_counterexample"}', encoding="utf-8")
    np.savez(state, mean=response.mean, factor=response.coefficient_factor)
    def bound(path, **keys):
        return {"path": path.name, "sha256": cli._sha(path), **keys}
    training = []
    for index in range(2):
        samples = (np.tile(np.eye(2)[index], (120, 1)) if constant_training
                   else rng.normal(size=(120, 2)))
        path = tmp_path / f"train{index}.npz"
        np.savez(path, samples=samples, design=np.zeros((120, 2)), nuisance=np.zeros_like(samples))
        training.append({"record_id": f"train{index}", "participant_key": "synthetic/A01",
            "recording_id": f"raw_train{index}", "sample_start": 0,
            "arrays": bound(path, samples_key="samples", design_key="design", nuisance_key="nuisance")})
    designs = rng.normal(size=(2, 100, 2))
    held = tmp_path / "held.npz"
    samples = designs[0] @ response.mean
    np.savez(held, samples=samples, designs=designs, truth_design=designs[0], nuisance=np.zeros_like(samples))
    descriptor = bound(held, samples_key="samples", candidate_designs_key="designs")
    if not constant_training:
        descriptor.update(design_key="truth_design", nuisance_key="nuisance")
    manifest = {"schema": "n2p3_channel_noise_audit_input/1",
        "evidence_scope": "public_calibration_development", "geometry": _geometry().record(),
        "conditioning_samples": 32, "fit_scope": "target_calibration", "max_diagnostic_lag": 12,
        "responses": [{"response_id": "anchor", "fit_scope": "analytic_reference", "source_model": bound(source),
            "exporter_identity": {"scope": "software_counterexample"},
            "arrays": bound(state, mean_key="mean", coefficient_factor_key="factor")}],
        "training": training,
        "noise_arms": [{"arm_id": "private0", "family": "private", "order": 0,
                        "residual_response_id": "anchor"}],
        "held": [{"record_id": "held", "recording_id": "raw_held", "sample_start": 0,
                  "participant_key": "synthetic/A01", "candidate_ids": ["a", "b"], "target_candidate": "a",
                  "arrays": descriptor}]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return cli, path


def test_optional_truth_residual_spectrum_does_not_turn_a_valid_score_into_failure(tmp_path):
    cli, path = _diagnostic_edge_case_manifest(tmp_path, constant_training=False)
    report = cli.audit(path, tmp_path / "audit")
    cell = report["cells"][0]
    assert cell["status"] == "scored" and cell["score"]["winner"] == "a"
    assert cell["truth_residual_diagnostic"]["status"] == "unavailable"
    assert "zero-energy" in cell["truth_residual_diagnostic"]["detail"]
    assert report["metrics"]["private0::anchor"]["participants"]["synthetic/A01"]["probability_coverage"] == 1


def test_undefined_centered_training_spectrum_does_not_turn_an_identified_noise_fit_into_failure(tmp_path):
    cli, path = _diagnostic_edge_case_manifest(tmp_path, constant_training=True)
    report = cli.audit(path, tmp_path / "audit")
    arm = report["noise_arms"][0]
    assert arm["status"] == "fitted_or_frozen"
    assert arm["training_diagnostic"]["status"] == "unavailable"
    assert report["cells"][0]["status"] == "scored"
    assert report["metrics"]["private0::anchor"]["participants"]["synthetic/A01"]["fit_or_score_failures"] == 0


def test_short_residual_tail_is_diagnostic_unavailable_without_weakening_invalid_declaration_gate(tmp_path):
    cli, path = _diagnostic_edge_case_manifest(tmp_path, constant_training=False)
    payload = json.loads(path.read_bytes())
    payload["max_diagnostic_lag"] = 100
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = cli.audit(path, tmp_path / "audit")
    assert report["cells"][0]["status"] == "scored"
    assert report["noise_arms"][0]["training_diagnostic"]["status"] == "unavailable"
    assert report["cells"][0]["truth_residual_diagnostic"]["status"] == "unavailable"
    payload["max_diagnostic_lag"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        cli.audit(path, tmp_path / "invalid_audit")
    assert not (tmp_path / "invalid_audit").exists()


def test_noise_restore_consumes_frozen_compiled_temporal_state_without_startup_recompile(tmp_path, monkeypatch):
    import transfer.temporal_noise as temporal_module
    noise = _private()
    path = tmp_path / "frozen_noise.json"
    noise.save(path)
    original = noise.record()
    assert original["schema"] == "n2p3_channel_noise_mechanism/2"
    assert all(record["schema"] == "n2p3_temporal_noise/5"
               for record in original["compiled_temporal_records"])
    def forbidden_compile(*args, **kwargs):
        raise AssertionError("a frozen artifact must not regenerate startup state")
    monkeypatch.setattr(temporal_module, "_innovation_factor", forbidden_compile)
    monkeypatch.setattr(temporal_module, "_startup", forbidden_compile)
    loaded = FittedChannelNoise.load(path)
    assert loaded.record() == original
    np.testing.assert_array_equal(loaded.whiten(np.ones((40, 2))), noise.whiten(np.ones((40, 2))))
