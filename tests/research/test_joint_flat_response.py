"""Independent raw-covariance oracles and original-source joint-flat consumers."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest
from scipy.io import savemat
from scipy.linalg import block_diag
from scipy.stats import multivariate_t

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    FittedChannelNoise,
    NoiseGeometry,
)
from research.execution import UnidentifiableFitError
from research.joint_flat_artifact import load_joint_flat_state, save_joint_flat_state
from research.joint_flat_authority import (
    build_c_plugin_parameters,
    build_source_parameters,
    compile_original_c,
    compile_original_continuation,
    compile_original_no_label,
    sample_original_c,
    source_parameters_from_saved_model,
)
from research.joint_flat_response import (
    FlatCompiledDecision,
    compile_flat_decision,
    compile_owned_flat_decision,
    flat_tail_components,
    predict_flat_tail,
    prepare_flat_model,
    prepare_flat_prediction,
)
from research.joint_response_sampling import (
    JointResponseSamples,
    conditional_scale_parameters,
    prepare_model,
    run_gibbs,
)
from research.joint_response_scale import (
    DecisionBlock,
    DiscreteGammaRule,
    FixedHyperparameters,
    GlobalGaussianPrior,
    compile_decision,
    conditional_gaussian_components,
    joint_evidence,
)
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble_views import CalibrationPrefixCompiler, _bnci_prediction_view


def raw_potential(y, d, z, covariance, prior, logjac=0.):
    """Integrate beta first in original y coordinates, then original flat da."""
    shape = covariance+d@prior.root@prior.root.T@d.T
    residual = y-d@prior.mean
    inverse_z = np.linalg.solve(shape, z)
    information = z.T@inverse_z
    fitted = np.linalg.solve(information, z.T@np.linalg.solve(shape, residual))
    remaining = residual-z@fitted
    q = remaining@np.linalg.solve(shape, remaining)
    return -.5*((len(y)-z.shape[1])*np.log(2*np.pi)
        +np.linalg.slogdet(shape)[1]+np.linalg.slogdet(information)[1]+q)+logjac


def fixture(nu=4.):
    rng = np.random.default_rng(71)
    prior = GlobalGaussianPrior([.15, -.2], [[.8, 0], [.3, .6]], "external")
    hyper = FixedHyperparameters(nu, 1.2, "externally_fixed")
    y, d = rng.normal(size=14), rng.normal(size=(14, 2))
    time = np.arange(7)/16
    common = np.column_stack((np.ones(7), time))
    z = np.einsum("nj,cd->ncjd", common, np.eye(2)).reshape(14, 4)
    phi = np.array([.4, -.15])
    sigma = np.array([[1.3, .3], [.3, .7]])
    stationary = sigma/(1-phi[:, None]*phi[None, :])
    covariance = np.empty((14, 14))
    for i in range(7):
        for j in range(7):
            block = np.diag(phi**abs(i-j))@stationary
            covariance[2*i:2*i+2, 2*j:2*j+2] = block if i >= j else block.T
    block = DecisionBlock("record", y, d, np.linalg.cholesky(covariance), "target_calibration")
    return prior, hyper, block, z, covariance


def test_raw_fourteen_rank_four_keeps_effective_ten_and_one_beta_prior():
    prior, hyper, block, z, covariance = fixture()
    compiled = compile_flat_decision(prior, hyper, block, z, conditioning_rows=6, qr_chunk_rows=3)
    model = prepare_flat_model(prior, (compiled,), hyper)
    shapes, _ = conditional_scale_parameters(model, [0., 0.])
    assert (compiled.observed_rows, compiled.nuisance_rank, compiled.effective_rows) == (14, 4, 10)
    np.testing.assert_array_equal(shapes, [7.])
    scales = np.array([[.25], [1.], [3.2]])
    got = conditional_gaussian_components(prior, [compiled], hyper, scales)[0]
    expected = [raw_potential(block.samples, block.design, z,
        hyper.relative_noise_shape*covariance/lam, prior) for lam in scales[:, 0]]
    np.testing.assert_allclose(got, expected, atol=3e-13)
    with pytest.raises(TypeError, match="flat potential"):
        joint_evidence(prior, [compiled], hyper, scale_rule=DiscreteGammaRule([1.], [1.], 4., 0))


def test_same_da_change_of_chart_retains_absolute_potential_and_tail():
    prior, hyper, block, z, covariance = fixture()
    transform = np.diag([.1, 7., 2., .3])
    logjac = np.linalg.slogdet(transform)[1]
    original = compile_flat_decision(prior, hyper, block, z, conditioning_rows=6)
    changed = compile_flat_decision(prior, hyper, block, z@transform,
        conditioning_rows=6, measure_logjacobian=logjac)
    scales = np.array([[.7]])
    left = conditional_gaussian_components(prior, [original], hyper, scales)[0]
    right = conditional_gaussian_components(prior, [changed], hyper, scales)[0]
    np.testing.assert_allclose(left, right, atol=1e-13)
    bare = compile_flat_decision(prior, hyper, block, z@transform, conditioning_rows=6)
    assert conditional_gaussian_components(prior, [bare], hyper, scales)[0][0] == pytest.approx(left[0]-logjac)
    no_c = prepare_flat_model(prior, (), hyper)
    charts = np.array([[.5, -.3], [0., 1.]])
    original_logs = flat_tail_components(prepare_flat_prediction(no_c, original), charts)
    changed_logs = flat_tail_components(prepare_flat_prediction(no_c, changed), charts)
    np.testing.assert_allclose(original_logs, changed_logs, atol=1e-13)
    assert abs(raw_potential(block.samples, block.design, z, covariance, prior)) > 0


def raw_conditional_t(block, z, covariance, hyper, prior, chart, cut):
    """Raw block-conditioning; independent of any stored whitening program."""
    beta = prior.mean+prior.root@chart
    kp, ktp = covariance[:cut, :cut], covariance[cut:, :cut]
    zp, zt = z[:cut], z[cut:]
    residual_p = block.samples[:cut]-block.design[:cut]@beta
    drift_cov = np.linalg.inv(zp.T@np.linalg.solve(kp, zp))
    drift_mean = drift_cov@zp.T@np.linalg.solve(kp, residual_p)
    bridge = ktp@np.linalg.inv(kp)
    image = zt-bridge@zp
    mean = block.design[cut:]@beta+bridge@residual_p+image@drift_mean
    shape = covariance[cut:, cut:]-bridge@ktp.T+image@drift_cov@image.T
    remaining = residual_p-zp@drift_mean
    qp = remaining@np.linalg.solve(hyper.relative_noise_shape*kp, remaining)
    dp = cut-z.shape[1]
    nu = hyper.degrees_of_freedom
    if np.isposinf(nu):
        from scipy.stats import multivariate_normal

        return multivariate_normal.logpdf(block.samples[cut:], mean=mean,
                                         cov=hyper.relative_noise_shape*shape)
    shape *= hyper.relative_noise_shape*(nu+qp)/(nu+dp)
    return multivariate_t.logpdf(block.samples[cut:], loc=mean, shape=shape, df=nu+dp)


@pytest.mark.parametrize("cut", [4, 6])
@pytest.mark.parametrize("nu", [4., np.inf])
def test_raw_tail_includes_prefix_drift_posterior_saturated_and_regular(cut, nu):
    prior, hyper, block, z, covariance = fixture(nu)
    full = compile_flat_decision(prior, hyper, block, z, conditioning_rows=cut)
    assert full.prefix.effective_rows == cut-4
    no_c = prepare_flat_model(prior, (), hyper)
    charts = np.array([[.4, -.2], [0., 0.], [1., -2.]])
    _, got = flat_tail_components(prepare_flat_prediction(no_c, full), charts)
    want = [raw_conditional_t(block, z, covariance, hyper, prior, chart, cut) for chart in charts]
    np.testing.assert_allclose(got, want, atol=2e-12)
    # Removing the learned drift covariance changes a real raw-tail density.
    wrong = multivariate_t.logpdf(block.samples[cut:], loc=block.design[cut:]@prior.mean,
        shape=hyper.relative_noise_shape*covariance[cut:, cut:], df=nu)
    if np.isfinite(nu):
        assert abs(wrong-got[1]) > .01


def test_unidentified_prefix_cannot_switch_to_a_different_flat_measure():
    prior, hyper, block, z, _ = fixture()
    with pytest.raises(UnidentifiableFitError, match="not identified"):
        compile_flat_decision(prior, hyper, block, z, conditioning_rows=2)


def test_no_flat_delegates_existing_proper_compiler_and_gibbs_exactly():
    prior, hyper, block, _, _ = fixture()
    got = compile_flat_decision(prior, hyper, block, np.empty((14, 0)), conditioning_rows=4)
    want = compile_decision(prior, hyper, block, conditioning_rows=4)
    assert not isinstance(got, FlatCompiledDecision)
    np.testing.assert_array_equal(got.augmented_root, want.augmented_root)
    args = dict(burnin=2, draws=8, seeds=(5, 8), initial_charts=np.array([[0., 0.], [1., 1.]]))
    left = run_gibbs(prepare_model(prior, [got], hyper), **args)
    right = run_gibbs(prepare_model(prior, [want], hyper), **args)
    np.testing.assert_array_equal(left.chart, right.chart)
    np.testing.assert_array_equal(left.scales, right.scales)


def test_two_records_keep_record_local_flat_and_independent_declared_scales():
    prior, hyper, block, z, covariance = fixture()
    second = DecisionBlock("second", block.samples*.3, block.design*.7, block.noise_root, block.role)
    left = compile_flat_decision(prior, hyper, block, z)
    right = compile_flat_decision(prior, hyper, second, z)
    got = conditional_gaussian_components(prior, [left, right], hyper, [[.6, 1.7]])[0][0]
    want = raw_potential(np.r_[block.samples, second.samples], np.vstack((block.design, second.design)),
        block_diag(z, z), block_diag(hyper.relative_noise_shape*covariance/.6,
                                    hyper.relative_noise_shape*covariance/1.7), prior)
    assert got == pytest.approx(want, abs=5e-13)


@pytest.mark.parametrize("nu", [1e-8, 4., 1e200])
def test_extreme_finite_gamma_identity_and_large_flat_shift_weak_beta(nu):
    prior, hyper, block, z, _ = fixture(nu)
    prior = GlobalGaussianPrior(prior.mean, prior.root*1e-4, "external")
    shift = z@np.array([2e3, -3e3, 1e3, -4e3])
    moved = DecisionBlock(block.decision_id, block.samples+shift, block.design, block.noise_root, block.role)
    left = compile_flat_decision(prior, hyper, block, z, conditioning_rows=6)
    right = compile_flat_decision(prior, hyper, moved, z, conditioning_rows=6)
    model = prepare_flat_model(prior, [left], hyper)
    shifted_model = prepare_flat_model(prior, [right], hyper)
    shapes, rates = conditional_scale_parameters(model, [0., 0.])
    shifted_shapes, shifted_rates = conditional_scale_parameters(shifted_model, [0., 0.])
    np.testing.assert_allclose(shapes, shifted_shapes)
    np.testing.assert_allclose(rates, shifted_rates, rtol=1e-11)
    samples = run_gibbs(model, burnin=2, draws=8, seeds=(91, 192),
                        initial_charts=np.array([[0., 0.], [2., -2.]]))
    assert not samples.gaussian_limit and hyper.degrees_of_freedom == nu
    assert samples.scales.shape == (2, 8, 1) and np.all(samples.scales > 0)


def test_owned_original_whitening_once_and_same_record_c_future_prefix_join(monkeypatch):
    prior, hyper, block, z, covariance = fixture()
    geometry = NoiseGeometry(("Cz", "Pz"), 16., "V", "native", "software", 2)
    noise = FittedChannelNoise.from_parameters([[.4], [-.15]], [[1.3, .3], [.3, .7]],
        geometry=geometry, conditioning_samples=2, reference_id="independent-raw-AR")
    y, d, nuisance = block.samples.reshape(7, 2), block.design.reshape(7, 2, 2), z.reshape(7, 2, 4)
    calls, original = [], FittedChannelNoise.whiten

    def counted(owner, value):
        calls.append(value.shape)
        return original(owner, value)

    monkeypatch.setattr(FittedChannelNoise, "whiten", counted)
    c = compile_owned_flat_decision(prior, hyper, noise, decision_id="same-record", role="target_calibration",
        native_samples=3, raw_batches=((0, y[:3], d[:3], nuisance[:3]),))
    full = compile_owned_flat_decision(prior, hyper, noise, decision_id="same-record", role="read_only_inference",
        native_samples=7, raw_batches=((0, y[:2], d[:2], nuisance[:2]), (2, y[2:], d[2:], nuisance[2:])),
        conditioning_samples=4, earlier_conditioning_samples=(3,))
    model = prepare_flat_model(prior, [c], hyper)
    prediction = prepare_flat_prediction(model, full)
    assert not prediction.prefix_already_in_model and prediction.conditioning_reference.observed_rows == 6
    assert len(model.blocks) == 1 and calls == [(3, 2, 7), (7, 2, 7)]
    charts = np.array([[.2, -.1]])
    _, got = flat_tail_components(prediction, charts)
    assert got[0] == pytest.approx(raw_conditional_t(block, z, covariance, hyper, prior, charts[0], 8), abs=2e-12)


def test_new_record_prefix_ratio_mcse_matches_independent_weight_derivative():
    from scipy.special import logsumexp

    prior, hyper, block, z, _ = fixture(np.inf)
    c = compile_flat_decision(prior, hyper, block, z)
    model = prepare_flat_model(prior, [c], hyper)
    future = DecisionBlock("new-independent-reference", block.samples*.7, block.design*.3,
                           block.noise_root, "read_only_inference")
    full = compile_flat_decision(prior, hyper, future, z, conditioning_rows=6)
    prediction = prepare_flat_prediction(model, full)
    chart = np.array([[[v, v*v/3] for v in [-2, -.5, 1, .3, -.2, .7, -1, 2]],
                      [[v, v*v/4] for v in [1.2, -.3, -1.7, .2, .6, 2.1, -.7, -1.1]]])
    samples = JointResponseSamples(model, chart, np.ones((2, 8, 1)), 0, 8, (2, 9), 0.)
    result = predict_flat_tail(samples, prediction, batch_size=2, max_rhat=100.,
                               min_bulk_ess=0., min_tail_ess=0.)
    prefix, tail = flat_tail_components(prediction, chart)

    def weighted_log_ratio(logweights):
        return logsumexp(prefix+tail+logweights)-logsumexp(prefix+logweights)

    derivatives = []
    delta = 1e-5
    for start in range(0, 16, 2):
        up, down = np.zeros(16), np.zeros(16)
        up[start:start+2], down[start:start+2] = np.log1p(delta), np.log1p(-delta)
        # Reweight a real draw batch; common normalization cancels in the
        # ratio. This finite difference is independent of the MCSE formula.
        derivatives.append(8*(weighted_log_ratio(up)-weighted_log_ratio(down))/(2*delta))
    expected = np.std(derivatives, ddof=1)/np.sqrt(len(derivatives))
    assert result["log_density_batch_mcse"] == pytest.approx(expected, abs=2e-8)


def compiler_fixture(subject="A01.mat", seed=904, *, recovery_source_sha=None, recovery_timestamp=None):
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(seed).normal(0, 5, fields["X"].shape).astype(np.float32)
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True)
    payload = stream.getvalue()
    if recovery_source_sha is not None:
        import time

        # scipy MAT v5 embeds an asctime header. Recover the ORIGINAL complete
        # bytes against its saved SHA; no input/member receipt is re-signed.
        for delta in range(-120, 121):
            header = ("MATLAB 5.0 MAT-file Platform: nt, Created on: "
                +time.asctime(time.localtime(recovery_timestamp+delta))).encode("ascii")
            candidate = header.ljust(116, b"\0")+payload[116:]
            if hashlib.sha256(candidate).hexdigest() == recovery_source_sha:
                payload = candidate
                break
        else:
            raise AssertionError("original synthetic MAT bytes could not be recovered against original SHA")
    compiler = CalibrationPrefixCompiler(payload, source_sha256=hashlib.sha256(payload).hexdigest(),
        source_relative_path=subject, budgets=(0, 120), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        response_seconds=.03125, pre_roll_seconds=.03125)
    return compiler


def test_actual_c_issuer_same_record_future_and_model_only_replay(tmp_path, monkeypatch):
    noise_fits = []
    original_noise_fit = FittedChannelNoise.fit

    def counted_noise_fit(records, **kwargs):
        noise_fits.append({"scope": records[0].role, "records": [record.record_id for record in records]})
        return original_noise_fit(records, **kwargs)

    monkeypatch.setattr(FittedChannelNoise, "fit", counted_noise_fit)
    compiler = compiler_fixture()
    prefix = compiler.compile(119)
    basis = np.ones((8, 1))
    geometry = {"channels": list(prefix.target_contract["channel_names"]), "sfreq": 256., "unit": "V",
        "reference": prefix.target_contract["reference"], "operator_identity": "native_unfiltered_volts/1",
        "observation_rank": 8}
    plan = json.dumps({"schema": "n2p3_joint_flat_external_prior/1",
        "freeze_role": "predeclared_external_numeric_parameters", "geometry": geometry,
        "basis": basis.tolist(), "mean": np.zeros(16).tolist(),
        "covariance_root": (np.eye(16)*1e-6).tolist()}).encode()
    parameters = build_c_plugin_parameters(compiler, prefix, external_prior_plan=plan,
        degrees_of_freedom=4., relative_noise_shape=1., noise_family="shared", noise_order=0,
        conditioning_samples=4)
    authorized = compile_original_c(parameters, compiler, prefix)
    samples = sample_original_c(authorized, burnin=2, draws=16, seeds=(51, 81),
        initial_charts=np.array([np.zeros(16), np.ones(16)]))
    # Follow actual first occurrence order; numeric selection_index is zero
    # based and is not a human selection number.
    selections = tuple(dict.fromkeys(compiler._record.selection_id))
    rows = np.flatnonzero(compiler._record.selection_id == selections[1])
    start, stop = int(compiler._record.flash_sample[rows[0]])-8, int(compiler._record.flash_sample[rows[-1]])+8
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    full, receipt = compile_original_continuation(authorized, compiler, prefix, view,
        candidate_id=view.decision.candidate_vocabulary[0],
        conditioning_stop=int(compiler._record.flash_sample[rows[0]])+2)
    prediction = prepare_flat_prediction(samples.model, full)
    policy = dict(batch_size=4, max_rhat=100., min_bulk_ess=0., min_tail_ess=0.)
    before = predict_flat_tail(samples, prediction, **policy)
    assert receipt["truth_consumed"] is False and receipt["reset_at_selection_boundary"] is False
    assert receipt["known_C_selection_continuations"][0]["authorized_C_labels"] == 119
    assert receipt["known_C_selection_continuations"][0]["new_labels_read"] == 0
    assert before["same_record_Gamma_variable"] and samples.scales.shape[-1] == 1
    identity = save_joint_flat_state(tmp_path/"package", samples, predictions=(prediction,),
                                    authority_binding={"C": authorized.binding, "future": receipt},
                                    parameters=parameters)

    def no_fit(*args, **kwargs):
        raise AssertionError("model-only replay called an original fitter or raw whitening")

    monkeypatch.setattr(FittedChannelNoise, "fit", no_fit)
    monkeypatch.setattr(FittedChannelNoise, "whiten", no_fit)
    recovered, predictions, binding, restored_parameters = load_joint_flat_state(tmp_path/"package", artifact_sha256=identity)
    after = predict_flat_tail(recovered, predictions[0], **policy)
    np.testing.assert_array_equal(before["component_tail_logs"], after["component_tail_logs"])
    np.testing.assert_array_equal(before["prefix_weights"], after["prefix_weights"])
    assert before["log_density"] == after["log_density"]
    assert binding["C"]["actual_Gamma_variables"] == 1
    assert len(noise_fits) == 1 and noise_fits[0]["scope"] == "target_calibration"
    (tmp_path/"software_fits.json").write_text(json.dumps({"public_fits": 0,
        "software_C_noise_fits": noise_fits, "retained_beta_draws": int(samples.chart.shape[0]*samples.chart.shape[1]),
        "complete_C_Gamma_axes": samples.scales.shape[2], "model_only_replay_fits": 0}), encoding="utf8")
    assert restored_parameters.digest == parameters.digest
    np.testing.assert_array_equal(restored_parameters.noise.compiled_prefix_root,
                                  parameters.noise.compiled_prefix_root)
    shutil.copytree(tmp_path/"package", tmp_path/"changed-package")
    changed = bytearray((tmp_path/"changed-package/state.npz").read_bytes())
    changed[-1] ^= 1
    (tmp_path/"changed-package/state.npz").write_bytes(changed)
    with pytest.raises(ValueError, match="identity"):
        load_joint_flat_state(tmp_path/"changed-package", artifact_sha256=identity)


def test_actual_source_builder_excludes_target_and_k0_does_not_read_target_labels(tmp_path, monkeypatch):
    from transfer.continuous_decision import ContinuousDecisionModel

    fits = []
    original_source_fit, original_noise_fit = ContinuousDecisionModel.fit, FittedChannelNoise.fit

    def counted_source_fit(*args, **kwargs):
        fits.append({"owner": "original_response", "scope": kwargs["fit_scope"]})
        return original_source_fit(*args, **kwargs)

    def counted_noise_fit(records, **kwargs):
        fits.append({"owner": "original_noise", "scope": records[0].role})
        return original_noise_fit(records, **kwargs)

    monkeypatch.setattr(ContinuousDecisionModel, "fit", counted_source_fit)
    monkeypatch.setattr(FittedChannelNoise, "fit", counted_noise_fit)
    source, target = compiler_fixture("A01.mat", 840), compiler_fixture("A02.mat", 841)
    source_prefix, zero = source.compile(120), target.compile(0)
    parameters, fitted_source = build_source_parameters(((source, source_prefix),),
        target_participant_key=zero.target_contract["participant_key"], degrees_of_freedom=np.inf,
        relative_noise_shape=1., basis=np.ones((8, 1)), ridge=1.,
        noise_order=0, conditioning_samples=4, source_model_path=tmp_path/"source.npz")
    parameters.noise.save(tmp_path/"source-noise.json")
    np.savez(tmp_path/"source-parameters.npz", prior_mean=parameters.prior.mean,
        prior_root=parameters.prior.root, basis=parameters.basis)
    (tmp_path/"source-parameter-binding.json").write_text(json.dumps(parameters.binding), encoding="utf8")
    assert fitted_source.provenance["fit_scope"] == "source_training"
    assert parameters.prior.program == "source_only" and parameters.noise.provenance["fit_scope"] == "frozen_production"
    first_selection = next(iter(dict.fromkeys(target._record.selection_id)))
    rows = np.flatnonzero(target._record.selection_id == first_selection)
    start, stop = int(target._record.flash_sample[rows[0]])-8, int(target._record.flash_sample[rows[-1]])+8
    view = _bnci_prediction_view(target._record, rows, start, stop, available_sample=stop)
    # Mutating unconsumed target truth cannot alter the numerical read program.
    original_labels = target._record.raw_is_target
    object.__setattr__(target._record, "raw_is_target", ~original_labels)
    try:
        full, receipt = compile_original_no_label(parameters, target, zero, view,
            candidate_id=view.decision.candidate_vocabulary[0], conditioning_stop=32)
    finally:
        object.__setattr__(target._record, "raw_is_target", original_labels)
    assert receipt["target_labels_read"] == receipt["target_parameter_fits"] == 0
    model = prepare_flat_model(parameters.prior, (), parameters.hyperparameters)
    logs = flat_tail_components(prepare_flat_prediction(model, full), np.zeros((1, 16)))[1]
    assert np.isfinite(logs).all()
    with pytest.raises(ValueError, match="excluded target"):
        build_source_parameters(((source, source_prefix),),
            target_participant_key=source_prefix.target_contract["participant_key"], degrees_of_freedom=4.,
            relative_noise_shape=1., basis=np.ones((8, 1)), ridge=1.,
            noise_order=0, conditioning_samples=4, source_model_path=tmp_path/"rejected.npz")
    assert fits == [{"owner": "original_response", "scope": "source_training"}]
    (tmp_path/"software_fits.json").write_text(json.dumps({"public_fits": 0,
        "actual_original_source_software_fits": fits, "target_fits": 0, "target_labels_read": 0}), encoding="utf8")


def test_saved_actual_source_no_label_prefix_tail_without_refit(tmp_path, monkeypatch):
    """Consume the original failed run's source owner; fresh target-only reads."""
    import os
    from pathlib import Path

    from transfer.continuous_decision import ContinuousDecisionModel

    if "N2P3_JOINT_FLAT_SOURCE_RECOVERY" not in os.environ:
        pytest.skip("this recovery probe requires the independently saved original source artifact")
    artifact = Path(os.environ["N2P3_JOINT_FLAT_SOURCE_RECOVERY"])
    saved = ContinuousDecisionModel.load(artifact/"source.npz")
    expected_source_sha = saved.provenance["records"][0]["record_id"].rsplit(":", 1)[1]
    source = compiler_fixture("A01.mat", 840, recovery_source_sha=expected_source_sha,
                              recovery_timestamp=(artifact/"source.npz").stat().st_mtime)

    def no_fit(*args, **kwargs):
        raise AssertionError("source recovery or no-label prediction performed a fit")

    monkeypatch.setattr(ContinuousDecisionModel, "fit", no_fit)
    monkeypatch.setattr(FittedChannelNoise, "fit", no_fit)
    target = compiler_fixture("A02.mat", 841)
    zero = target.compile(0)
    original_model_bytes = (artifact/"source.npz").read_bytes()
    (tmp_path/"recovered_original_source.mat").write_bytes(source._history_source_payload)
    parameters, _ = source_parameters_from_saved_model(((source, source.compile(120)),),
        target_participant_key=zero.target_contract["participant_key"], degrees_of_freedom=np.inf,
        relative_noise_shape=1., source_model_path=artifact/"source.npz")
    prior = parameters.prior
    first_selection = next(iter(dict.fromkeys(target._record.selection_id)))
    rows = np.flatnonzero(target._record.selection_id == first_selection)
    start, stop = int(target._record.flash_sample[rows[0]])-8, int(target._record.flash_sample[rows[-1]])+8
    view = _bnci_prediction_view(target._record, rows, start, stop, available_sample=stop)
    full, receipt = compile_original_no_label(parameters, target, zero, view,
        candidate_id=view.decision.candidate_vocabulary[0], conditioning_stop=32)
    model = prepare_flat_model(prior, (), parameters.hyperparameters)
    samples = run_gibbs(model, burnin=0, draws=8, seeds=(32, 45), initial_charts=np.zeros((2, 16)))
    prediction = prepare_flat_prediction(model, full)
    identity = save_joint_flat_state(tmp_path/"source_only_package", samples, predictions=(prediction,),
        authority_binding=receipt, parameters=parameters)
    (tmp_path/"package_identity.json").write_text(json.dumps({"artifact_sha256": identity}), encoding="utf8")
    assert original_model_bytes == (artifact/"source.npz").read_bytes()
    restored_samples, restored_predictions, restored_binding, restored_parameters = load_joint_flat_state(
        tmp_path/"source_only_package", artifact_sha256=identity)
    assert restored_binding["target_parameter_fits"] == 0 and restored_parameters.digest == parameters.digest
    status = {"diagnostic_artifact": "complete", "prediction_status": "not_evaluated"}
    try:
        logs = flat_tail_components(prediction, np.zeros((1, 16)))[1]
        assert np.isfinite(logs).all() and receipt["target_labels_read"] == 0
        np.testing.assert_array_equal(flat_tail_components(prediction, samples.chart)[1],
            flat_tail_components(restored_predictions[0], restored_samples.chart)[1])
        status.update(prediction_status="complete", raw_tail_log_density=logs.tolist())
    except DiffuseProjectionResolutionError as error:
        status.update(prediction_status="numerical_resolution_rejected", reason=str(error),
                      raw_tail_log_density=None)
    (tmp_path/"prediction_status.json").write_text(json.dumps(status, indent=2), encoding="utf8")
    parameters.noise.save(tmp_path/"original_source_noise.json")
    (tmp_path/"source_parameter_binding.json").write_text(json.dumps(parameters.binding), encoding="utf8")
    (tmp_path/"software_fits.json").write_text(json.dumps({"public_fits": 0,
        "source_fits": 0, "target_fits": 0, "target_labels_read": 0,
        "source_model_file_sha256": hashlib.sha256((artifact/"source.npz").read_bytes()).hexdigest(),
        "prediction_status": status["prediction_status"]}), encoding="utf8")
    if os.environ.get("N2P3_JOINT_FLAT_REQUIRE_PREDICTION_COMPLETE") == "1":
        assert status["prediction_status"] == "complete", status
