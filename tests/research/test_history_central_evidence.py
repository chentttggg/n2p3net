"""Central certificates must survive finite mixture and original-byte replay."""
from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.history_artifact import predict_history, replay_history_capsule, save_history_capsule
from research.history_authority import FrozenHistoryLaw
from research.response_history_identification import (
    HistoryMixtureResolutionError,
    PublicResponseHistory,
    PublicSelection,
    ResponsePosterior,
    StructuredResponseObservation,
    conditional_history_evidence,
)
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_prefix
from transfer.ensemble_views import (
    CalibrationPrefixCompiler,
    _bnci_prediction_view,
    calibration_prefix_identity,
)
from transfer.response_dictionary import ResponseDictionarySpec


def test_each_component_can_pass_while_complete_mixture_requires_numerical_unavailable():
    posterior = ResponsePosterior(np.array([[1.], [.5]]), np.zeros((2, 1)), np.eye(2), np.eye(1), 10.)
    current = PublicSelection("current", ("q0", "q1"), np.array([3, 5]), np.eye(2))
    history = PublicResponseHistory((current,), "current", np.array([.5, .5]))
    observation = StructuredResponseObservation(np.zeros((8, 1)), 0, np.ones((8, 1)), 2, False)
    budget = 6*np.sqrt(np.finfo(float).eps)

    def marginal(design):
        del design
        # Both components pass their own 6-row budget. The stable mixture
        # propagates two prefix errors plus one conditional error and rejects.
        certificate = {"absolute_log_density_error_estimate": .4*budget,
                       "absolute_log_density_error_budget": budget}
        return SimpleNamespace(joint_log_marginal=0., prefix_log_marginal=0., conditional_log_density=0.,
            resolution={"full": certificate, "prefix": certificate,
                        "conditional": {**certificate, "actual_tail_scalar_rows": 6},
                        "source_sha256": "a"*64})

    with pytest.raises(HistoryMixtureResolutionError, match="mixture error estimate"):
        conditional_history_evidence(posterior, history, observation, marginal_evaluator=marginal)


def test_large_common_conditional_density_keeps_actual_candidate_evidence_normalized():
    posterior = ResponsePosterior(np.ones((2, 1)), np.zeros((2, 2)), np.eye(2), np.eye(2), 10.)
    current = PublicSelection("current", ("q0", "q1", "q2"), np.array([4, 8, 12]), np.eye(3))
    history = PublicResponseHistory((current,), "current", np.array([.1, .3, .6]))
    observation = StructuredResponseObservation(np.zeros((32, 2)), 0, np.ones((32, 1)), 3, False)
    budget = 58*np.sqrt(np.finfo(float).eps)

    def marginal(design):
        q = int(np.flatnonzero(design[:, 1])[0]//4)-1
        density = 5e6-(q+1.)
        certificate = {"absolute_log_density_error_estimate": 0., "absolute_log_density_error_budget": budget}
        return SimpleNamespace(joint_log_marginal=density, prefix_log_marginal=0., conditional_log_density=density,
            resolution={"full": certificate, "prefix": certificate,
                        "conditional": {**certificate, "actual_tail_scalar_rows": 58},
                        "source_sha256": "a"*64})

    actual = conditional_history_evidence(posterior, history, observation, marginal_evaluator=marginal)
    identity = CandidateExpertIdentity("history", "b"*64, "c"*64, None, "d"*64, "history_fixed_prior", False)
    evidence = CandidateEvidence(current.candidate_ids, actual.log_candidate_probability, "participant-software",
        "current", "e"*64, (identity,), 1, (1, 1, 1), (1, 1, 1))
    assert evidence.probability_available
    np.testing.assert_allclose(evidence.probability.sum(), 1., rtol=0, atol=2e-15)
    assert actual.mixture_resolution["normalized_log_probability_error_estimate"] <= budget


@pytest.fixture(scope="module")
def saved_central_parent(tmp_path_factory):
    """One software C fit; normal and two drift decisions share its exact bytes."""
    directory = tmp_path_factory.mktemp("central-history-parent")/"parent"
    fields = bnci_mat_fields()
    raw = np.random.default_rng(20261012).normal(0, 5, fields["X"].shape)
    # The entire source is declared before C fit. Held drift is never read or
    # used by the fitter. Source-byte replay will enforce this same recording.
    for selection, volts in ((3, 1e3), (4, 1e6)):
        start = int(fields["trial"][0, selection])-1-16
        stop = int(fields["trial"][0, selection])-1+120*64
        x = np.linspace(-1., 1., stop-start)
        raw[start:stop] += volts*1e6*(np.arange(1, 9)[None, :]+x[:, None]*np.arange(8, 0, -1)[None, :])
    fields["X"] = raw
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    payload = stream.getvalue()
    compiler = CalibrationPrefixCompiler(payload, source_sha256=hashlib.sha256(payload).hexdigest(),
        source_relative_path="A01.mat", budgets=(24,), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        response_seconds=.1, pre_roll_seconds=.0625)
    prefix = compiler.compile(24)
    spec = CalibrationExpertProgram("native", "continuous_ar", noise_order=0, conditioning_samples=16,
        dictionary=ResponseDictionarySpec("gaussian", .1, 1))
    parent = fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights("central-history-software", ("native",)))
    parent.save(directory)
    parent = FittedEnsemble.load(directory)
    model = parent._experts[0].predictor
    plan = {"schema": "n2p3_frozen_history_protocol_law/1", "prior_kind": "independent_uniform_selection_targets",
        "freeze_role": "predeclared_protocol", "model_digest": model.digest(),
        "dictionary_digest": model.dictionary.digest(), "calibration_prefix_digest": calibration_prefix_identity(prefix),
        "response_states": [{"common_amplitude": 1., "target_amplitude": 1., "latency_seconds": 0.,
                             "width_multiplier": 1., "probability": 1.}]}
    law = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=model, prefix=prefix)
    return compiler, prefix, parent, directory, law


@pytest.mark.parametrize("selection,status", [(2, "complete"), (3, "not_available"), (4, "not_available")])
def test_actual_parent_central_certificate_and_typed_rejection_survive_original_byte_capsule(
        saved_central_parent, tmp_path, monkeypatch, selection, status):
    from research.channel_noise_mechanism import _IMPLEMENTATION_SHA256
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.temporal_noise import TemporalNoise

    compiler, prefix, parent, directory, law = saved_central_parent
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == selection)
    start = int(compiler._record.flash_sample[rows[0]])-16
    stop = int(compiler._record.flash_sample[rows[-1]])+20
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    original_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("central history capsule rebuilt an original response/noise producer")

    monkeypatch.setattr(ContinuousDecisionModel, "fit", forbidden)
    monkeypatch.setattr(TemporalNoise, "fit", forbidden)
    identity = save_history_capsule(tmp_path/"capsule", parent_directory=directory, expert_id="native",
        compiler=compiler, prefix=prefix, view=view, law=law)
    saved = json.loads((tmp_path/"capsule/capsule.json").read_bytes())["result"]
    replay = replay_history_capsule(tmp_path/"capsule", **identity)
    assert saved["status"] == replay["status"] == status
    assert len(replay["requested"]["candidate_vocabulary"]) == 36
    if status == "complete":
        numerical = replay["prediction"]
        assert numerical["central_consumer_source_sha256"] == (_IMPLEMENTATION_SHA256,)
        assert numerical["consumer_source_sha256"] != model.provenance["implementation_sha256"]
        assert len(numerical["state_numerical_resolution"]) == 36
        for state in numerical["state_numerical_resolution"]:
            cert = state["resolution"]
            assert {"full", "prefix", "conditional", "source_sha256"} <= cert.keys()
            assert cert["source_sha256"] == _IMPLEMENTATION_SHA256
            assert cert["full"]["flat_dimensions"] == cert["prefix"]["flat_dimensions"] == 16
        mixture = numerical["mixture_numerical_resolution"]
        assert len(mixture["candidates"]) == 36 and not mixture["all_input_certificate"]
        assert mixture["normalized_log_probability_error_estimate"] <= mixture["absolute_log_density_error_budget"]
    else:
        assert replay["probability"] is None
        assert replay["failure"]["kind"] == "history_numerical_unavailable"
        assert replay["failure"]["central_consumer_source_sha256"] == _IMPLEMENTATION_SHA256
        assert replay["capability_digest"] and not replay["failure"]["all_input_certificate"]
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original_bytes


def test_nonnominal_whole_selection_state_cannot_reuse_a_partially_fitted_c_posterior(saved_central_parent):
    compiler, prefix, parent, _, law = saved_central_parent
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == 2)
    start, stop = int(compiler._record.flash_sample[rows[0]])-16, int(compiler._record.flash_sample[rows[-1]])+20
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    plan = json.loads(law.plan_bytes)
    # C has the first 24 events of selection0. An 80s shift makes that whole
    # selection's response visible in selection2's actual observed block.
    # Binary C labels determine q; they do not update theta_C and beta jointly.
    plan["response_states"][0]["latency_seconds"] = 80.
    shifted = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=model, prefix=prefix)
    result = predict_history(parent, "native", compiler, prefix, view, shifted)
    assert result["status"] == "not_available" and result["probability"] is None
    assert len(result["requested"]["candidate_vocabulary"]) == 36
    assert result["failure"]["reason"] == "C_history_latent_state_not_conditioned_by_original_frozen_fit"
    evidence = result["failure"]["evidence"]
    assert evidence["new_C_latent_producer_required"]
    assert len(evidence["original_authority"]["full_actual_event_ledger"]) == 360
    assert evidence["C_fit_selection_id"] in evidence["original_authority"]["C_fit_selection_ids"]
    assert sum(compiler._record.selection_id[:prefix.budget] == evidence["C_fit_selection_id"]) == 24


def test_explicit_joint_c_nominal_clamp_keeps_variable_future_states_legal(saved_central_parent):
    compiler, prefix, parent, _, law = saved_central_parent
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == 2)
    start, stop = int(compiler._record.flash_sample[rows[0]])-16, int(compiler._record.flash_sample[rows[-1]])+20
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    names = [str(name) for name in dict.fromkeys(compiler._record.selection_id[compiler._record.flash_sample < stop])]
    plan = json.loads(law.plan_bytes)
    nominal = {**plan["response_states"][0], "probability": .5}
    shifted = {**nominal, "latency_seconds": 80.}
    mass = np.ones((36, 36, 36, 2, 2, 2))
    mass[:, :, :, 1, :, :] = 0  # C selection0 is nominal; later theta axes vary.
    mass /= mass.sum()
    plan.update(prior_kind="explicit_joint_selection_targets_and_response_states", selection_ids=names,
        candidate_vocabularies=[list(view.decision.candidate_vocabulary)]*3,
        response_states=[nominal, shifted], joint_mass=mass.tolist())
    explicit = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=model, prefix=prefix)
    capability = compiler.history_scope(view, prefix=prefix, model=model, law=explicit)
    assert capability.binding["C_selection_state_admission"][0]["positive_state_indices"] == (0,)
    assert capability.binding["C_selection_state_admission"][0]["program"] == (
        "all_C_conditioned_positive_q_state_designs_exactly_zero")
    assert names[0] in capability.binding["exact_zero_design_axis_marginalization"]["selection_ids"]
    assert capability.conditioned_joint_mass.sum(axis=(0, 1, 2, 4, 5))[1] == 0
    np.testing.assert_allclose(capability.conditioned_joint_mass.sum(axis=(0, 1, 2, 3, 4)), [.5, .5])
    result = predict_history(parent, "native", compiler, prefix, view, explicit)
    assert result["status"] == "complete"
    assert result["prediction"]["complete_finite_state_shape"] == (36, 2)


def test_tiny_positive_c_state_cannot_underflow_out_of_the_authority_support(saved_central_parent):
    compiler, _, _, _, base_law = saved_central_parent
    prefix = compiler.compile(2)  # One target row and one non-target: q remains 6-way.
    spec = CalibrationExpertProgram("native", "continuous_ar", noise_order=0, conditioning_samples=16,
        dictionary=ResponseDictionarySpec("gaussian", .1, 1))
    parent = fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights("tiny-state-software", ("native",)))
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == 2)
    start, stop = int(compiler._record.flash_sample[rows[0]])-16, int(compiler._record.flash_sample[rows[-1]])+20
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    plan = json.loads(base_law.plan_bytes)
    tiny = float(np.nextafter(0., 1.))*2
    assert tiny > 0 and tiny/6 == 0
    nominal = {**plan["response_states"][0], "probability": 1.}
    shifted = {**nominal, "latency_seconds": 80., "probability": tiny}
    plan.update(model_digest=model.digest(), dictionary_digest=model.dictionary.digest(),
        calibration_prefix_digest=calibration_prefix_identity(prefix), response_states=[nominal, shifted])
    law = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=model, prefix=prefix)
    result = predict_history(parent, "native", compiler, prefix, view, law)
    assert result["status"] == "not_available"
    assert result["failure"]["reason"] == "C_history_latent_state_not_conditioned_by_original_frozen_fit"
    assert result["failure"]["evidence"]["positive_state_indices"] == (0, 1)
