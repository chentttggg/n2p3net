"""The actual g program, from initialization through frozen held prediction."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag, solve_discrete_lyapunov, toeplitz

from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    _conditional_prediction,
)
from transfer.continuous_gradient import ConditionalGaussianObjective
from transfer.kernel_prior import CompiledKernelPrior, KernelPriorSpec
from transfer.noise_model import metadata_digest
from transfer.temporal_noise import TemporalNoise


def design(onsets, labels, n, basis):
    result = np.zeros((n, 2*basis.shape[1]))
    for onset, label in zip(onsets, labels, strict=True):
        count = min(len(basis), n-onset)
        result[onset:onset+count, :basis.shape[1]] += basis[:count]
        result[onset:onset+count, basis.shape[1]:] += label*basis[:count]
    return result


def inputs():
    rng = np.random.default_rng(781)
    t = np.arange(12)
    basis = np.column_stack((np.exp(-((t-3)/3)**2), np.exp(-((t-9)/3)**2)))
    mean = np.array([[.3, -.1], [.1, .2], [.8, .3], [.2, -.3]])
    records = []
    for i, n in enumerate((180, 213)):
        onsets = np.arange(5+i, n-len(basis), 23, dtype=np.int64)
        labels = (np.arange(len(onsets))%3 == 0).astype(np.int64)
        noise = rng.normal(scale=.35, size=(n, 2))
        noise[1:] += .2*noise[:-1].copy()
        x = design(onsets, labels, n, basis)@mean+noise
        x += np.column_stack((np.ones(n), np.linspace(-1, 1, n)))@np.array([[1., -.4], [.3, .1]])
        records.append(CalibrationRecord(f"C{i}", x, onsets, labels))
    return records, basis


def fit(records, basis, *, order=4, estimator="burg", prior=None):
    return ContinuousDecisionModel.fit(records, sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
        operator_identity="synthetic-C/1", basis=basis, noise_order=order, conditioning_samples=32,
        noise_estimator=estimator, kernel_prior=KernelPriorSpec.calibration_g(9.) if prior is None else prior)


def held(model):
    n = 103
    onsets = np.arange(4, n-len(model.basis), 17, dtype=np.int64)
    membership = np.eye(2, dtype=np.int64)[np.arange(len(onsets))%2]
    samples = design(onsets, membership[:, 0], n, model.basis)@model.kernel_mean
    samples += np.random.default_rng(9).normal(scale=.3, size=samples.shape)
    return samples, onsets, membership


def score(model, values, identity="held"):
    return model.score(*values, sfreq=model.sfreq, channels=model.channels,
        candidate_ids=model.candidate_ids, operator_identity=model.operator_identity, record_id=identity)


def stationary_covariance(coefficients, n):
    p = len(coefficients)
    transition = np.zeros((p, p))
    transition[0] = coefficients
    transition[1:, :-1] = np.eye(p-1)
    driving = np.zeros((p, p))
    driving[0, 0] = 1
    covariance = solve_discrete_lyapunov(transition, driving)
    acf, power = [], np.eye(p)
    for _ in range(n):
        acf.append((power@covariance)[0, 0])
        power = power@transition
    return toeplitz(acf)


def diffuse_scatter(y, d, z, k, h):
    variance = k+d@h@d.T
    inverse_z = np.linalg.solve(variance, z)
    drift = np.linalg.solve(z.T@inverse_z, z.T@np.linalg.solve(variance, y))
    residual = y-z@drift
    return residual.T@np.linalg.solve(variance, residual)


def test_public_fit_matches_independent_dense_shared_kernel_and_full_prefix_sigma():
    records, basis = inputs()
    model = fit(records, basis)
    ds = [design(r.onsets, r.target_mask, len(r.samples), basis) for r in records]
    zs = [np.column_stack((np.ones(len(d)), np.linspace(-1, 1, len(d)))) for d in ds]
    g0 = sum((d-z@np.linalg.solve(z.T@z, z.T@d)).T@(d-z@np.linalg.solve(z.T@z, z.T@d))
             for d, z in zip(ds, zs, strict=True))
    h0 = 9*np.linalg.inv(g0)
    ks = [stationary_covariance(model.noise.coefficients, len(r.samples)) for r in records]
    information, score_vector = np.zeros((4, 4)), np.zeros((4, 2))
    for record, d, z, k in zip(records, ds, zs, ks, strict=True):
        inverse_z, inverse_d = np.linalg.solve(k, z), np.linalg.solve(k, d)
        residual_precision_d = inverse_d-inverse_z@np.linalg.solve(z.T@inverse_z, z.T@inverse_d)
        information += d.T@residual_precision_d
        score_vector += residual_precision_d.T@record.samples
    expected_h = np.linalg.inv(np.linalg.inv(h0)+information)
    np.testing.assert_allclose(model.kernel_row_covariance, expected_h, atol=3e-13)
    np.testing.assert_allclose(model.kernel_mean, expected_h@score_vector, atol=3e-13)
    y, d, z, k = np.vstack([r.samples for r in records]), np.vstack(ds), block_diag(*zs), block_diag(*ks)
    prefix = np.r_[np.arange(32), np.arange(32)+len(records[0].samples)]
    q_full = diffuse_scatter(y, d, z, k, h0)
    q_prefix = diffuse_scatter(y[prefix], d[prefix], z[prefix], k[np.ix_(prefix, prefix)], h0)
    rows = sum(len(r.samples)-32 for r in records)
    np.testing.assert_allclose(model.innovation_covariance, (q_full-q_prefix)/rows, atol=2e-12)


def test_general_basis_change_preserves_initial_residual_burg_and_final_frozen_program(monkeypatch):
    records, basis = inputs()
    actual_residuals = []
    original_fit = TemporalNoise.fit.__func__

    def capture(cls, residuals, **kwargs):
        actual_residuals.append([np.array(r, copy=True) for r in residuals])
        return original_fit(cls, residuals, **kwargs)

    monkeypatch.setattr(TemporalNoise, "fit", classmethod(capture))
    original = fit(records, basis)
    transform = np.array([[2.1, .7], [-.3, .4]])
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = matrix[2:, 2:] = transform
    changed = fit(records, basis@transform)
    for a, b in zip(*actual_residuals, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-12)
    np.testing.assert_allclose(original.noise.reflection_coefficients, changed.noise.reflection_coefficients, atol=2e-12)
    np.testing.assert_allclose(changed.kernel_mean, np.linalg.solve(matrix, original.kernel_mean), atol=2e-12)
    np.testing.assert_allclose(changed.innovation_covariance, original.innovation_covariance, atol=2e-12)
    values = held(original)
    np.testing.assert_allclose(score(original, values).log_evidence, score(changed, values).log_evidence, atol=2e-10)


def test_reference_root_is_all_c_and_does_not_follow_fitted_noise_order():
    records, basis = inputs()
    zero, four = fit(records, basis, order=0), fit(records, basis, order=4)
    np.testing.assert_array_equal(zero.kernel_prior.precision_root, four.kernel_prior.precision_root)
    assert zero.kernel_prior.rank_ledger["observed_rows"] == sum(len(r.samples) for r in records)
    assert zero.noise.provenance["objective_rows"] == sum(len(r.samples)-32 for r in records)
    assert "ridge" not in zero.provenance


def test_voltage_scaling_preserves_prior_root_and_scales_column_covariance():
    records, basis = inputs()
    original = fit(records, basis)
    scaled = fit([replace(r, samples=1e6*r.samples) for r in records], basis)
    np.testing.assert_array_equal(scaled.kernel_prior.precision_root, original.kernel_prior.precision_root)
    np.testing.assert_allclose(scaled.kernel_mean, 1e6*original.kernel_mean, rtol=2e-11)
    np.testing.assert_allclose(scaled.innovation_covariance, 1e12*original.innovation_covariance, rtol=2e-11)


def test_exact_gradient_consumes_the_same_frozen_root_and_matches_independent_central_difference():
    records, basis = inputs()
    model = fit(records, basis, order=1)
    samples = [r.samples for r in records]
    designs = [design(r.onsets, r.target_mask, len(r.samples), basis) for r in records]
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=32, kernel_prior=model.kernel_prior)
    point = np.array([.31])
    _, gradient = producer.value_gradient(point)

    def numpy_value(eta):
        noise = TemporalNoise.from_reflections(np.tanh(eta), innovation_variance=1., reference_id="g-gradient-oracle")
        scatter, determinant, rows, _, _ = _conditional_prediction(samples, designs, noise, 32,
            np.zeros_like(model.kernel_mean), None, prior_precision_root=model.kernel_prior.precision_root)
        root = np.linalg.cholesky(scatter/rows)
        return np.log(np.diag(root)).sum()+.5*2*determinant/rows

    finite = (numpy_value(point+1e-5)-numpy_value(point-1e-5))/2e-5
    np.testing.assert_allclose(gradient, [finite], atol=3e-8)


def test_save_load_freezes_actual_root_and_held_does_not_compile_any_prior(tmp_path, monkeypatch):
    records, basis = inputs()
    model = fit(records, basis, order=1, estimator="conditional_profile")
    path = tmp_path/"g.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        import json
        assert json.loads(str(archive["metadata"]))["schema"] == "n2p3_continuous_decision_model/5"
    restored = ContinuousDecisionModel.load(path)
    before = restored.digest()
    values = held(model)
    import transfer.continuous_decision as module
    monkeypatch.setattr(module, "compile_kernel_prior", lambda *a, **k: pytest.fail("held rebuilt prior"))
    np.testing.assert_array_equal(score(model, values).log_evidence, score(restored, values).log_evidence)
    assert restored.digest() == before
    assert restored.kernel_prior.digest() == model.kernel_prior.digest()


def test_relabeling_g_and_its_root_cannot_keep_the_old_actual_posterior():
    records, basis = inputs()
    model = fit(records, basis)
    payload = model.kernel_prior.record()
    payload["spec"]["g"] = 8.
    payload["precision_root"] = (model.kernel_prior.reference_root/np.sqrt(8.)).tolist()
    from transfer.noise_model import array_digest
    payload["state_sha256"] = array_digest(np.array(payload["precision_root"]), np.array(payload["reference_root"]),
                                           np.array(payload["design_root"]))
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    fake = CompiledKernelPrior.from_record(payload)
    provenance = {**model.provenance, "kernel_prior_spec": fake.spec.record(), "kernel_prior_digest": fake.digest()}
    with pytest.raises(ValueError, match="posterior"):
        replace(model, kernel_prior=fake, provenance=provenance)


def test_g_numeric_payload_cannot_be_hidden_by_downgrading_metadata_schema(tmp_path):
    import json

    records, basis = inputs()
    path = tmp_path/"actual_g.npz"
    fit(records, basis).save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema"] = "n2p3_continuous_decision_model/4"
    metadata.pop("kernel_prior")
    for field in ("kernel_prior_spec", "kernel_prior_digest", "kernel_prior_fit_statistics"):
        metadata["provenance"].pop(field)
    metadata["provenance"]["ridge"] = 1.
    metadata["artifact_sha256"] = metadata_digest({k: v for k, v in metadata.items() if k != "artifact_sha256"})
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    bad = tmp_path/"relabelled_ridge.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="numeric payload"):
        ContinuousDecisionModel.load(bad)


def test_g_and_ridge_ambiguity_is_rejected_at_fit_ingress():
    records, basis = inputs()
    with pytest.raises(ValueError, match="mutually"):
        ContinuousDecisionModel.fit(records, sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
            operator_identity="synthetic-C/1", basis=basis, noise_order=0, conditioning_samples=32,
            ridge=1., kernel_prior=KernelPriorSpec.calibration_g(9.))


def test_gradient_rejects_full_c_prior_when_given_different_fold_designs():
    records, basis = inputs()
    model = fit(records, basis)
    with pytest.raises(ValueError, match="designs"):
        ConditionalGaussianObjective([records[0].samples],
            [design(records[0].onsets, records[0].target_mask, len(records[0].samples), basis)],
            conditioning_samples=32, kernel_prior=model.kernel_prior)


@pytest.mark.parametrize("changed", ["record_id", "n_samples", "read_sample_support", "design_sha256"])
def test_reference_ledger_cannot_diverge_from_actual_c_binding_after_resigning(changed):
    records, basis = inputs()
    model = fit(records, basis)
    payload = model.kernel_prior.record()
    row = payload["provenance"]["records"][0]
    if changed == "record_id":
        row[changed] = "unrelated_C_group"
    elif changed == "n_samples":
        row[changed] += 1
        row["read_sample_support"][1] += 1
    elif changed == "read_sample_support":
        row[changed][0] = 1
    else:
        row[changed] = "f"*64
    payload["provenance"]["reference_design_sha256"] = metadata_digest(payload["provenance"]["records"])
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="support|actual authorized C"):
        CompiledKernelPrior.from_record(payload)


@pytest.mark.parametrize("changed", ["basis_sha256", "input_sha256"])
def test_prior_binding_cannot_change_basis_or_actual_input_after_resigning(changed):
    records, basis = inputs()
    model = fit(records, basis)
    payload = model.kernel_prior.record()
    if changed == "basis_sha256":
        payload["provenance"]["binding"][changed] = "f"*64
    else:
        payload["provenance"]["binding"]["calibration_inputs"][0][changed] = "f"*64
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    alternate = CompiledKernelPrior.from_record(payload)
    with pytest.raises(ValueError, match="binding"):
        replace(model, kernel_prior=alternate,
                provenance={**model.provenance, "kernel_prior_digest": alternate.digest()})
