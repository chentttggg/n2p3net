"""Original MAT decoder -> actual saved C parent -> native history -> byte replay."""
import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.history_artifact import (
    history_candidate_evidence,
    predict_history,
    replay_history_capsule,
    save_history_capsule,
)
from research.history_authority import FrozenHistoryLaw
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_prefix
from transfer.ensemble_views import (
    CalibrationPrefixCompiler,
    _bnci_prediction_view,
    calibration_prefix_identity,
)
from transfer.response_dictionary import ResponseDictionarySpec


@pytest.fixture(scope="module")
def actual_source_parent(tmp_path_factory):
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(65121).normal(0, 5, fields["X"].shape).astype(np.float32)
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    payload = stream.getvalue()
    compiler = CalibrationPrefixCompiler(payload, source_sha256=hashlib.sha256(payload).hexdigest(),
        source_relative_path="A01.mat", budgets=(24,), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        response_seconds=.1, pre_roll_seconds=.0625)
    prefix = compiler.compile(24)
    spec = CalibrationExpertProgram("native", "continuous_ar", noise_order=0, conditioning_samples=16,
        dictionary=ResponseDictionarySpec("gaussian", .1, 2))
    fitted = fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights("history-software", ("native",)))
    directory = tmp_path_factory.mktemp("history-native-source-parent")/"parent"
    fitted.save(directory)
    parent = FittedEnsemble.load(directory)
    assert parent.fit_available
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == 2)
    start, stop = int(compiler._record.flash_sample[rows[0]])-16, int(compiler._record.flash_sample[rows[-1]])+20
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    plan = {"schema": "n2p3_frozen_history_protocol_law/1", "prior_kind": "independent_uniform_selection_targets",
        "freeze_role": "predeclared_protocol", "model_digest": model.digest(),
        "dictionary_digest": model.dictionary.digest(), "calibration_prefix_digest": calibration_prefix_identity(prefix),
        "response_states": [{"common_amplitude": 1., "target_amplitude": 1., "latency_seconds": 0.,
                             "width_multiplier": 1., "probability": 1.}]}
    law = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=model, prefix=prefix)
    return compiler, prefix, view, parent, directory, law


def test_original_source_full_event_ledger_censor_and_exact_zero_axis_mass(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    model = parent._experts[0].predictor
    capability = compiler.history_scope(view, prefix=prefix, model=model, law=law)
    assert len(capability.binding["full_actual_event_ledger"]) == 360
    assert len(capability.selections) == 3
    assert capability.binding["current_response_right_censored"] is True
    assert capability.binding["read_support"] == view.raw_scope
    assert capability.binding["conditioning_support"] == (view.raw_scope[0], view.raw_scope[0]+16)
    assert len(capability.binding["exact_zero_design_axis_marginalization"]["selection_ids"]) == 2
    assert all(len(s.candidate_ids) == 36 for s in capability.selections)
    assert capability.binding["truth_consumed"] is False


def test_actual_saved_parent_native_history_capsule_replays_without_fit_or_dense_noise(actual_source_parent, tmp_path, monkeypatch):
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.temporal_noise import TemporalNoise

    compiler, prefix, view, parent, directory, law = actual_source_parent
    before = [(path.name, path.read_bytes()) for path in directory.iterdir()]

    def forbidden(*args, **kwargs):
        raise AssertionError("history replay refitted an original producer")

    monkeypatch.setattr(ContinuousDecisionModel, "fit", forbidden)
    monkeypatch.setattr(TemporalNoise, "fit", forbidden)
    identity = save_history_capsule(tmp_path/"capsule", parent_directory=directory, expert_id="native",
        compiler=compiler, prefix=prefix, view=view, law=law)
    actual = replay_history_capsule(tmp_path/"capsule", **identity)
    assert actual["status"] == "complete"
    assert len(actual["probability"]) == 36
    assert actual["fits"] == actual["response_fits"] == 0
    assert actual["prediction"]["complete_finite_state_shape"] == (36, 1)
    assert before == [(path.name, path.read_bytes()) for path in directory.iterdir()]
    for name, payload in before:
        assert (tmp_path/"capsule/parent"/name).read_bytes() == payload


@pytest.mark.parametrize("mutation", ["onsets", "samples", "participant", "cutoff", "source"])
def test_current_coherently_changed_history_view_is_rejected_before_score(actual_source_parent, mutation):
    compiler, prefix, view, parent, _, law = actual_source_parent
    changes = {"onsets": {"onsets": view.onsets+1},
        "samples": {"samples": view.samples+1e-8},
        "participant": {"participant_key": view.participant_key+"-other"},
        "cutoff": {"available_sample": view.available_sample+1},
        "source": {"source_sha256": "f"*64}}
    changed = replace(view, **changes[mutation])
    with pytest.raises(ValueError):
        predict_history(parent, "native", compiler, prefix, changed, law)


def test_source_decoder_event_deletion_cannot_resign_complete_history(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    raw = compiler._record
    before = raw.flash_sample
    try:
        object.__setattr__(raw, "flash_sample", before+1)
        with pytest.raises(ValueError, match="decoder operands"):
            compiler.history_scope(view, prefix=prefix, model=parent._experts[0].predictor, law=law)
    finally:
        object.__setattr__(raw, "flash_sample", before)


def test_typed_resource_unavailable_preserves_requested_current_candidates(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    actual = predict_history(parent, "native", compiler, prefix, view, law, maximum_evaluated_states=1)
    assert actual["status"] == "not_available" and actual["probability"] is None
    assert len(actual["requested"]["candidate_vocabulary"]) == 36
    assert actual["failure"]["kind"] == "authorized_history_unavailable"
    evidence = history_candidate_evidence(parent, "native", compiler, prefix, view, law, maximum_evaluated_states=1)
    assert len(evidence.candidate_vocabulary) == 36 and not evidence.probability_available


def test_nonzero_latency_width_law_owns_union_support_and_observed_censor(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    plan = json.loads(law.plan_bytes)
    plan["response_states"][0].update(latency_seconds=.01, width_multiplier=1.2)
    changed = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=parent._experts[0].predictor,
        prefix=prefix)
    capability = compiler.history_scope(view, prefix=prefix, model=parent._experts[0].predictor, law=changed)
    last = capability.binding["full_actual_event_ledger"][-1]
    onset = last["actual_onset_sample"]
    assert last["union_response_support"][1] > onset+len(parent._experts[0].predictor.basis)
    assert last["right_censored"] and capability.binding["read_support"] == view.raw_scope
    outcome = predict_history(parent, "native", compiler, prefix, view, changed)
    assert outcome["status"] == "complete" and len(outcome["probability"]) == 36


def test_explicit_correlated_joint_mass_zero_axes_are_summed_not_assumed_independent(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    names = [str(name) for name in dict.fromkeys(compiler._record.selection_id[
        compiler._record.flash_sample < view.available_sample])]
    plan = json.loads(law.plan_bytes)
    plan.update(prior_kind="explicit_joint_selection_targets_and_response_states",
        selection_ids=names, candidate_vocabularies=[list(view.decision.candidate_vocabulary)]*3)
    mass = np.zeros((36, 36, 36, 1, 1, 1))
    # Preserve correlations between earlier C/context and current: every q
    # remains positive, but its prior changes when C constrains the first axis.
    q = np.arange(36)+1.
    mass[:, :, :, 0, 0, 0] = 1+q[None, None, :]*np.arange(36)[:, None, None]/36
    mass /= mass.sum()
    plan["joint_mass"] = mass.tolist()
    correlated = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=parent._experts[0].predictor,
        prefix=prefix)
    actual = predict_history(parent, "native", compiler, prefix, view, correlated)
    assert actual["status"] == "complete"
    assert actual["prediction"]["complete_finite_state_shape"] == (36, 1)
    current_prior = np.asarray(actual["prediction"]["current_prior"])
    assert not np.allclose(current_prior, np.ones(36)/36)


def test_native_actual_c_label_input_digest_rejection_is_independent_of_wrapper(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    model = parent._experts[0].predictor
    provenance = json.loads(json.dumps(model.provenance))
    provenance["records"][0]["input_sha256"] = "f"*64
    changed = replace(model, provenance=provenance)
    plan = json.loads(law.plan_bytes)
    plan["model_digest"] = changed.digest()
    forged = FrozenHistoryLaw.from_protocol_plan(json.dumps(plan).encode(), model=changed, prefix=prefix)
    with pytest.raises(ValueError, match="native fitter"):
        compiler.history_scope(view, prefix=prefix, model=changed, law=forged)


def test_capsule_coherent_history_deletion_is_rejected_by_original_byte_rebuild(actual_source_parent, tmp_path):
    compiler, prefix, view, _, directory, law = actual_source_parent
    identity = save_history_capsule(tmp_path/"capsule", parent_directory=directory, expert_id="native",
        compiler=compiler, prefix=prefix, view=view, law=law)
    path = tmp_path/"capsule/capsule.json"
    record = json.loads(path.read_bytes())
    record["result"]["capability"]["full_actual_event_ledger"].pop(0)
    # Re-sign the consumer shell; original raw/parent/plan external hashes
    # remain fixed and replay must regenerate the missing past event.
    payload = json.dumps(record).encode()
    path.write_bytes(payload)
    identity["capsule_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="history replay differs"):
        replay_history_capsule(tmp_path/"capsule", **identity)


def test_actual_partial_next_event_prefix_keeps_all_zero_target_columns_and_no_future_clock(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    next_rows = np.flatnonzero(compiler._record.selection_index == 3)
    stop = int(compiler._record.flash_sample[next_rows[0]])+1
    # A separate declared observed interval; its current physical schedule is
    # unchanged. The source owner authorizes every read and event through b.
    extended = _bnci_prediction_view(compiler._record, np.flatnonzero(compiler._record.selection_index == 2),
        view.raw_scope[0], stop, available_sample=stop)
    capability = compiler.history_scope(extended, prefix=prefix, model=parent._experts[0].predictor, law=law)
    next_prefix = capability.selections[-1]
    assert len(next_prefix.onsets) == 1 and len(next_prefix.candidate_ids) == 36
    assert np.count_nonzero(next_prefix.membership.sum(axis=0) == 0) == 30
    assert np.all(next_prefix.onsets < stop)
    assert not np.any(next_prefix.onsets == compiler._record.flash_sample[next_rows[1]])


def test_current_only_positive_parity_with_original_frozen_native_scorer(actual_source_parent):
    compiler, prefix, view, parent, _, law = actual_source_parent
    model = parent._experts[0].predictor
    rows = np.flatnonzero(compiler._record.selection_index == 2)
    stop = int(compiler._record.flash_sample[rows[-1]])+len(model.basis)
    complete = _bnci_prediction_view(compiler._record, rows, view.raw_scope[0], stop, available_sample=stop)
    result = predict_history(parent, "native", compiler, prefix, complete, law)
    original = model.score(complete.samples, complete.onsets, complete.decision.membership,
        sfreq=model.sfreq, channels=model.channels, operator_identity=model.operator_identity,
        candidate_ids=model.candidate_ids, record_id=complete.record_id)
    assert result["status"] == "complete"
    np.testing.assert_allclose(result["prediction"]["log_evidence"], original.log_evidence, rtol=0, atol=2e-7)
    np.testing.assert_allclose(result["prediction"]["log_probability"], original.log_probability, rtol=0, atol=2e-7)
