"""Direct covariance and original-issuer source response implementation checks."""
import copy
import gc
import hashlib
import io
import json
import weakref
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from scipy.io import savemat
from scipy.special import logsumexp
from threadpoolctl import threadpool_info, threadpool_limits

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.channel_noise_mechanism import (
    FittedChannelNoise,
    NoiseGeometry,
    StructuredGaussianMarginalProgram,
)
from research.finite_response_law import FiniteResponseLaw
from research.history_authority import HistoryUnavailableError
from research.joint_flat_authority import build_source_parameters
from research.latent_response_reference import ResponseState
from research.source_response_states import (
    SourceInformedFiniteResponseProgram,
    SourceInformedResponseModel,
    SourceInformedResponseParameters,
    _assert_history_scope,
    _physical_nuisance,
    fit_source_informed_response,
)
from tests.support.bnci import bnci_mat_fields
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.ensemble_views import CalibrationPrefixCompiler, _bnci_prediction_view
from transfer.joint_flat_programme import FrozenJointFlatSource, FrozenJointFlatSourcePreparation
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest


@pytest.fixture(scope="module", autouse=True)
def native_one_thread():
    with threadpool_limits(1):
        yield


def dense(y, mean, image, z, covariance):
    shape = covariance+image@image.T
    residual = y-mean
    inverse_z = np.linalg.solve(shape, z)
    information = z.T@inverse_z
    remaining = residual-z@np.linalg.solve(information, z.T@np.linalg.solve(shape, residual))
    return -.5*((len(y)-z.shape[1])*np.log(2*np.pi)+np.linalg.slogdet(shape)[1]
        +np.linalg.slogdet(information)[1]+remaining@np.linalg.solve(shape, remaining))


def compact_compiler(subject, seed):
    fields = bnci_mat_fields()
    fields["y"].fill(0)
    fields["y_stim"].fill(0)
    starts = 16+8000*np.arange(fields["trial"].size)
    fields["trial"] = (starts+1).reshape(1, -1).astype(float)
    rng = np.random.default_rng(seed)
    fields["X"][:int(starts[-1])+8000] = rng.normal(0, 5, (int(starts[-1])+8000, 8))
    for selection, start in enumerate(starts):
        row, col = selection % 6, (selection+2) % 6
        for repetition in range(10):
            for position, candidate in enumerate(np.roll(np.arange(12), repetition)):
                onset = int(start+(repetition*12+position)*64)
                fields["y_stim"][onset:onset+32, 0] = candidate+1
                fields["y"][onset:onset+32, 0] = 2. if candidate in (row, 6+col) else 1.
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True)
    content = stream.getvalue()
    return CalibrationPrefixCompiler(content, source_sha256=hashlib.sha256(content).hexdigest(),
        source_relative_path=subject, budgets=(0, 120), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        response_seconds=4/256, pre_roll_seconds=4/256)


def test_central_noise_scale_full_jacobian_direct_factor_and_design_error():
    rng = np.random.default_rng(416)
    geometry = NoiseGeometry(("Cz", "Pz"), 16., "V", "native", "native_unfiltered_volts/1", 2)
    sigma = np.array([[1.2, .2], [.2, .8]])
    noise = FittedChannelNoise.from_parameters(np.empty((2, 0)), sigma, geometry=geometry,
        conditioning_samples=3, family="shared", reference_id="scale-oracle")
    n, c = 12, 2
    y, design, mean, factor = rng.normal(size=(n, c)), rng.normal(size=(n, 2)), rng.normal(size=(2, c)), rng.normal(size=(4, 3))
    z = _physical_nuisance(n, c, 16.)
    eta = .375
    primitive = StructuredGaussianMarginalProgram(noise, y, mean, factor, z, intercept_chart=True, noise_scale=eta)
    got = primitive.posterior_state(design)
    image = np.einsum("nk,kcr->ncr", design, factor.reshape(2, c, 3)).reshape(n*c, 3)
    expected = dense(y.ravel(), (design@mean).ravel(), image, z.reshape(n*c, -1), eta*np.kron(np.eye(n), sigma))
    assert got.log_marginal == pytest.approx(expected, abs=2e-12)
    pair = primitive.evaluate(design)
    assert pair.joint_log_marginal == pytest.approx(expected, abs=2e-12)
    prefix = dense(y[:3].ravel(), (design[:3]@mean).ravel(), image[:6], z[:3].reshape(6, -1), eta*np.kron(np.eye(3), sigma))
    assert pair.conditional_log_density == pytest.approx(expected-prefix, abs=3e-12)
    error = np.full(design.shape, 1e-15)
    with_error = primitive.posterior_state(design, design_source_error=error)
    assert with_error.log_marginal == got.log_marginal
    assert with_error.resolution["C"]["absolute_log_density_error_estimate"] > got.resolution["C"]["absolute_log_density_error_estimate"]
    legacy = StructuredGaussianMarginalProgram(noise, y, mean, factor, z, intercept_chart=True)
    explicit = StructuredGaussianMarginalProgram(noise, y, mean, factor, z, intercept_chart=True, noise_scale=1.)
    np.testing.assert_array_equal(legacy.posterior_state(design).precision_root, explicit.posterior_state(design).precision_root)


def test_new_design_source_error_positive_long_dot_includes_its_own_reduction_roundoff(monkeypatch):
    geometry = NoiseGeometry(("Cz",), 16., "V", "native", "native_unfiltered_volts/1", 1)
    noise = FittedChannelNoise.from_parameters(np.empty((1, 0)), [[1.]], geometry=geometry,
        conditioning_samples=2, reference_id="positive-source-error")
    n, k = 5, 1024
    primitive = StructuredGaussianMarginalProgram(noise, np.zeros((n, 1)), np.ones((k, 1)), np.ones((k, 1)),
        _physical_nuisance(n, 1, 16.))
    design_error = np.full((n, k), 1e-17)
    design_error[:, 0] = 1.
    observed = []
    original = primitive._white_error

    def capture(raw, white, source_error, geometry):
        observed.append(np.asarray(source_error).copy())
        return original(raw, white, source_error, geometry)

    monkeypatch.setattr(primitive, "_white_error", capture)
    primitive._posterior_operands(np.zeros((n, k)), design_source_error=design_error)
    exact = Fraction(1)+1023*Fraction.from_float(1e-17)
    assert Fraction.from_float(float((design_error@np.ones(k))[0])) < exact
    assert Fraction.from_float(float(observed[0][0, 0])) >= exact
    assert Fraction.from_float(float(observed[1][0, 0, 0])) >= exact


@pytest.fixture(scope="module")
def source_preparation():
    return FrozenJointFlatSourcePreparation()


@pytest.fixture(scope="module")
def actual_source(tmp_path_factory, source_preparation):
    directory = tmp_path_factory.mktemp("source_response_original")
    source, target = compact_compiler("A01.mat", 211), compact_compiler("A02.mat", 212)
    source_prefix, zero = source.compile(120), target.compile(0)
    # Exactly one small synthetic original source fit; no public EEG is read.
    path = directory/"original-source.npz"
    parameters, _ = build_source_parameters(((source, source_prefix),),
        target_participant_key=zero.target_contract["participant_key"], degrees_of_freedom=4.,
        relative_noise_shape=.7, basis=np.array([[0.], [1.], [.5], [0.]]), ridge=1., noise_order=0,
        conditioning_samples=64, source_model_path=path)
    frozen = FrozenJointFlatSource.from_parameters(parameters, original_model_content=path.read_bytes())
    dictionary = NativeResponseDictionary.from_source_matrix(parameters.basis,
        sample_rate_hz=parameters.noise.geometry.sfreq, source_basis_sha256=array_digest(parameters.basis), terminal_values=[0.])
    exported = SourceInformedResponseParameters.from_source(frozen, target_authority=zero,
        continuation=dictionary, continuation_role="new_declared_continuation_of_source_native_arrays")
    (directory/"source-input.mat").write_bytes(source._history_source_payload)
    (directory/"target-input.mat").write_bytes(target._history_source_payload)
    exported.save(directory/"response-parameters.json")
    (directory/"execution-receipt.json").write_text(json.dumps({"scope": "software_synthetic_original_BNCI_issuer",
        "source_parameter_fits": 1, "target_parameter_fits": 0, "Gamma_sampler_calls": 0,
        "source_original_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_parameters_digest": exported.digest(), "native_thread_pools": threadpool_info()}, sort_keys=True), encoding="utf8")
    return source, target, exported


def programme(arm, law=None):
    return SourceInformedFiniteResponseProgram(law or FiniteResponseLaw.from_states((
        ResponseState(1., 1., 0., 1., .6), ResponseState(1., 1., 1/1024, 1.25, .4))), arm,
        "whole_observed_current_q", 100, 10000, 512*1024*1024, -15)


def current_view(target, selection):
    rows = np.flatnonzero(target._record.selection_index == selection)
    start = int(target._record.flash_sample[rows[0]])-4
    stop = int(target._record.flash_sample[rows[-1]])+4
    return _bnci_prediction_view(target._record, rows, start, stop, available_sample=stop)


@pytest.mark.parametrize("arm", ["nominal", "mean_shape", "finite_shape", "gaussian_moment"])
def test_original_source_c0_whole_model_codec_and_truth_free_replay(actual_source, source_preparation, tmp_path, monkeypatch, arm):
    _, target, parameters = actual_source
    zero = target.compile(0)
    model = fit_source_informed_response(zero, compiler=target, parameters=parameters, program=programme(arm))
    assert model.posteriors == (None,)
    assert model.binding["C_state"] == "source_prior_only_common_flat_measure"
    view = current_view(target, 0)
    cap = model.history_scope(target, view, prefix=zero, conditioning_stop=1)
    evidence, binding = model.score_history(view, cap)
    assert logsumexp(evidence.log_candidate_probability) == pytest.approx(0., abs=1e-12)
    assert not evidence.resolution["P_identification_required"]
    assert binding["evaluated_positive_states"] == (72 if arm == "finite_shape" else 36)
    path = model.save(tmp_path/f"{arm}.json")
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    original_labels = target._record.raw_is_target
    object.__setattr__(target._record, "raw_is_target", ~original_labels)
    try:
        monkeypatch.setattr(FittedChannelNoise, "fit", lambda *a, **k: pytest.fail("replay noise refit"))
        loaded = SourceInformedResponseModel.load(path, expected_sha256=expected_sha, source_preparation=source_preparation)
        replay = loaded.score_history(view, loaded.history_scope(target, view, prefix=zero, conditioning_stop=1))[0]
    finally:
        object.__setattr__(target._record, "raw_is_target", original_labels)
    np.testing.assert_array_equal(replay.log_candidate_probability, evidence.log_candidate_probability)
    with pytest.raises(ValueError, match="external digest"):
        SourceInformedResponseModel.load(path, expected_sha256="0"*64)
    conditional = fit_source_informed_response(zero, compiler=target, parameters=parameters,
        program=replace(programme(arm), scoring_law="fixed_current_prior_conditional_tail"))
    with pytest.raises((ValueError, ArithmeticError)):
        conditional.score_history(view, conditional.history_scope(target, view, prefix=zero, conditioning_stop=1))


def test_source_participant_exclusion(actual_source):
    source, _, parameters = actual_source
    with pytest.raises(ValueError, match="target participant"):
        SourceInformedResponseParameters.from_source(parameters.source, target_authority=source.compile(0),
            continuation=parameters.dictionary, continuation_role="new_declared_continuation_of_source_native_arrays")


@pytest.mark.parametrize("arm", ["nominal", "mean_shape", "finite_shape", "gaussian_moment"])
def test_actual_partial_c_crossing_selection_future_new_local_and_zero_fit_replay(actual_source, source_preparation, tmp_path, monkeypatch, arm):
    _, target, parameters = actual_source
    prefix = target.compile(119)
    model = fit_source_informed_response(prefix, compiler=target, parameters=parameters, program=programme(arm))
    assert model.calibration_samples == len(prefix.raw.samples)
    assert len(model.components) == (2 if arm == "finite_shape" else 1)
    view = current_view(target, 1)
    conditioning = int(target._record.flash_sample[np.flatnonzero(target._record.selection_index == 1)[0]])-15
    cap = model.history_scope(target, view, prefix=prefix, conditioning_stop=conditioning)
    evidence, binding = model.score_history(view, cap)
    assert logsumexp(evidence.log_candidate_probability) == pytest.approx(0., abs=1e-12)
    assert not binding["future_truth_consumed"] and binding["C_refits"] == 0
    assert len(cap.selections[0].onsets) == 120 and len(model.binding["C_selections"][0]["onsets"]) == 119
    assert binding["evaluated_positive_states"] == (144 if arm == "finite_shape" else 36)
    path = model.save(tmp_path/f"partial-{arm}.json")
    loaded = SourceInformedResponseModel.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                             source_preparation=source_preparation)
    original_state = loaded.record()["posteriors"]
    monkeypatch.setattr(ContinuousDecisionModel, "fit", lambda *a, **k: pytest.fail("replay source parameter fit"))
    monkeypatch.setattr(FittedChannelNoise, "fit", lambda *a, **k: pytest.fail("replay noise parameter fit"))
    replay_cap = loaded.history_scope(target, view, prefix=prefix, conditioning_stop=conditioning)
    replay, replay_binding = loaded.score_history(view, replay_cap)
    assert loaded.operational_receipt()["C_QR_reconstructions"] == len(model.components)
    assert loaded.record()["posteriors"] == original_state
    np.testing.assert_array_equal(replay.log_candidate_probability, evidence.log_candidate_probability)
    assert loaded._verify_actual_c(replay_cap)[1] == 0
    assert replay_binding == binding
    if arm == "nominal":
        assert loaded.score_history(view, replay_cap)[1] == replay_binding
    if arm == "gaussian_moment":
        assert model.posteriors[0].proper_dimensions == 50
        assert len(cap.selections) == 2  # Future selection adds its independent local image.
    if arm == "nominal":
        tampered = json.loads(path.read_bytes())
        tampered["posteriors"][0]["precision_root"][0][0] *= 1.01
        content = json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
        forged = SourceInformedResponseModel.from_bytes(content, expected_sha256=hashlib.sha256(content).hexdigest(),
                                                      source_preparation=source_preparation)
        forged_cap = forged.history_scope(target, view, prefix=prefix, conditioning_stop=conditioning)
        with pytest.raises(ValueError, match="actual C QR reconstruction"):
            forged.score_history(view, forged_cap)


def test_actual_history_registry_rejects_copied_q_conditioning_and_scope_mutations(actual_source):
    _, target, parameters = actual_source
    prefix = target.compile(0)
    model = fit_source_informed_response(prefix, compiler=target, parameters=parameters, program=programme("nominal"))
    view = current_view(target, 0)
    cap = model.history_scope(target, view, prefix=prefix, conditioning_stop=1)
    with pytest.raises(ValueError, match="registered"):
        model.score_history(view, copy.copy(cap))
    for name, mutation in (("allowed", {cap.selections[0].selection_id: (0,)}),
                           ("conditioning_samples", 2), ("selections", cap.selections[::-1]+cap.selections)):
        original = getattr(cap, name)
        object.__setattr__(cap, name, mutation)
        try:
            with pytest.raises(ValueError):
                model.score_history(view, cap)
        finally:
            object.__setattr__(cap, name, original)
    with pytest.raises(ValueError, match="conditioning policy"):
        model.history_scope(target, view, prefix=prefix, conditioning_stop=2)



def test_source_only_conditional_p1_has_core_typed_refusal_and_whole_retains_full_path(actual_source, monkeypatch):
    _, target, parameters = actual_source
    prefix, view = target.compile(0), current_view(target, 0)
    model = fit_source_informed_response(prefix, compiler=target, parameters=parameters,
        program=replace(programme("nominal"), scoring_law="fixed_current_prior_conditional_tail"))
    cap = model.history_scope(target, view, prefix=prefix, conditioning_stop=1)
    with pytest.raises(HistoryUnavailableError) as captured:
        model.score_history(view, cap)
    assert captured.value.failure_kind == "numerical"
    assert captured.value.reason == "source_only_conditional_prefix_cannot_identify_physical_flat"
    assert captured.value.evidence["prefix_scalar_rows"] < captured.value.evidence["physical_flat_columns"]
    whole = fit_source_informed_response(prefix, compiler=target, parameters=parameters, program=programme("nominal"))

    class FullPathReachedError(RuntimeError):
        pass

    def reached(*args, **kwargs):
        raise FullPathReachedError

    monkeypatch.setattr("research.source_response_states._primitive", reached)
    with pytest.raises(FullPathReachedError):
        whole.score_history(view, whole.history_scope(target, view, prefix=prefix, conditioning_stop=1))


def test_history_wrapper_gc_releases_private_registry_and_live_copy_stays_rejected(actual_source):
    _, target, parameters = actual_source
    prefix, view = target.compile(0), current_view(target, 0)
    model = fit_source_informed_response(prefix, compiler=target, parameters=parameters, program=programme("nominal"))
    cap = model.history_scope(target, view, prefix=prefix, conditioning_stop=1)
    _assert_history_scope(model, view, cap)
    identifier, reference = id(cap), weakref.ref(cap)
    assert target._source_response_history_scopes[identifier][0]() is cap
    with pytest.raises(ValueError, match="registered"):
        model.score_history(view, copy.copy(cap))
    del cap
    gc.collect()
    assert reference() is None and identifier not in target._source_response_history_scopes


def test_scoped_source_ingress_reuses_math_decode_but_rechecks_bytes_and_target(actual_source, source_preparation, tmp_path, monkeypatch):
    _, _, parameters = actual_source
    source = source_preparation.prepare(parameters.source.content, expected_sha256=parameters.source.digest)
    monkeypatch.setattr(FrozenJointFlatSource, "from_bytes", lambda *a, **k: pytest.fail("duplicate mathematical source decode"))
    record = parameters.record()
    assert SourceInformedResponseParameters.from_record(record, source_preparation=source_preparation).source is source
    path = parameters.save(tmp_path/"parameters.json")
    loaded = SourceInformedResponseParameters.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                 source_preparation=source_preparation)
    assert loaded.source is source and loaded.digest() == parameters.digest()
    altered = json.loads(json.dumps(record))
    altered["source_base64"] = altered["source_base64"][:-4]+"AAAA"
    with pytest.raises(ValueError, match="declared bytes"):
        SourceInformedResponseParameters.from_record(altered, source_preparation=source_preparation)
    altered = json.loads(json.dumps(record))
    altered["binding"]["target_contract"]["participant_key"] = source.parameters.binding["source_prefixes"][0]["target_contract"]["participant_key"]
    with pytest.raises(ValueError, match="target participant"):
        SourceInformedResponseParameters.from_record(altered, source_preparation=source_preparation)


def test_actual_c2_preserves_ambiguous_q_six_times_two_shapes_and_counts_failed_reconstruction(actual_source, source_preparation, tmp_path):
    _, target, parameters = actual_source
    prefix = target.compile(2)
    model = fit_source_informed_response(prefix, compiler=target, parameters=parameters, program=programme("finite_shape"))
    assert model.binding["C_selections"][0]["allowed_target_indices"] == tuple(range(6))
    assert len(model.components) == 12
    assert tuple((c["target_indices"][0], c["response_state_indices"][0]) for c in model.components) == tuple(
        (q, h) for q in range(6) for h in range(2))
    qlogs = [logsumexp([c["log_posterior_weight"] for c in model.components if c["target_indices"][0] == q]) for q in range(6)]
    np.testing.assert_allclose(qlogs, -np.log(6), rtol=0, atol=3e-15)
    with pytest.raises(HistoryUnavailableError, match="12>11"):
        fit_source_informed_response(prefix, compiler=target, parameters=parameters,
            program=replace(programme("finite_shape"), maximum_calibration_components=11))
    path = model.save(tmp_path/"ambiguous-C2.json")
    loaded = SourceInformedResponseModel.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                             source_preparation=source_preparation)
    assert loaded.record() == model.record()
    forged = replace(loaded, posteriors=(replace(loaded.posteriors[0], precision_root=1.01*loaded.posteriors[0].precision_root),
                                        *loaded.posteriors[1:]))
    view = current_view(target, 1)
    conditioning = int(view.onsets[0])+view.raw_scope[0]+model.program.conditioning_offset_samples
    cap = forged.history_scope(target, view, prefix=prefix, conditioning_stop=conditioning)
    with pytest.raises(ValueError, match="actual C QR reconstruction"):
        forged.score_history(view, cap)
    assert forged.operational_receipt()["C_QR_reconstruction_attempts"] == 1
    assert forged.operational_receipt()["C_QR_reconstructions"] == 1
