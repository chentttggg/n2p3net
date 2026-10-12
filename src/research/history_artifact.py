"""Exact original bytes -> original scope/C/parent load -> history replay capsule.

Producer packages are copied byte-for-byte. The consumer never calls model.save
or creates a second fitted producer. Loading requires external expected source,
parent, protocol plan and capsule hashes; coherent edits cannot self-admit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data.epochs import parse_preprocessing_spec
from research.continuous_history_adapter import score_authorized_history
from research.contracts import semantic_sha256
from research.history_authority import FrozenHistoryLaw, HistoryUnavailableError
from research.joint_response_scale import NumericalResolutionError
from research.response_history_identification import HistoryNuisanceResolutionError
from transfer.ensemble_fit import FittedEnsemble
from transfer.ensemble_views import (
    CalibrationPrefixCompiler,
    PredictionDecisionView,
    SelectionEpochProfile,
)


def _raw_sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def _owned_model(parent, expert_id):
    matches = [expert for expert in parent._experts if expert.program.expert_id == expert_id]
    if len(matches) != 1 or matches[0].program.family != "continuous_ar" or matches[0].predictor is None:
        raise HistoryUnavailableError("original_parent_has_no_frozen_continuous_history_expert")
    return matches[0].predictor


def predict_history(parent, expert_id, compiler, prefix, view, law, *, maximum_evaluated_states=100000):
    """Actual opt-in caller; no existing current-only programme is rerouted."""
    model = _owned_model(parent, expert_id)
    if parent.participant_key != view.participant_key:
        raise ValueError("history parent cannot score a different participant")
    if semantic_sha256(parent._target_contract) != semantic_sha256(prefix.target_contract):
        raise ValueError("history parent target authorization differs from original C")
    if semantic_sha256(parent.fit_receipts) != semantic_sha256(
            model.provenance["ensemble_fit_binding"]["training_receipts"]):
        raise ValueError("actual parent/model fit authority differs")
    from research.prefix_inner_replay import check_native_fit_inputs, prefix_native_fit_inputs

    spec = next(expert.program for expert in parent._experts if expert.program.expert_id == expert_id)
    # The shared checker owns actual native labels/dose; inspect only this
    # original continuous expert, not unrelated trial experts or their inputs.
    class SelectedExpert:
        _experts = tuple(expert for expert in parent._experts if expert.program.expert_id == expert_id)

    check_native_fit_inputs(prefix_native_fit_inputs(prefix, (spec,)), SelectedExpert(), actual_K=prefix.budget)
    requested = {"participant_key": view.participant_key, "decision_id": view.decision_id,
                 "candidate_vocabulary": list(view.decision.candidate_vocabulary),
                 "physical_support_digest": view.physical_support_digest}
    try:
        capability = compiler.history_scope(view, prefix=prefix, model=model, law=law)
        evidence, binding = score_authorized_history(model, capability,
            maximum_evaluated_states=maximum_evaluated_states)
    except HistoryUnavailableError as error:
        return {"schema": "n2p3_history_prediction_outcome/1", "requested": requested,
                "status": error.failure_kind, "failure": error.record(), "probability": None,
                "model_digest": model.digest(), "fits": 0, "response_fits": 0}
    except (HistoryNuisanceResolutionError, NumericalResolutionError, np.linalg.LinAlgError) as error:
        return {"schema": "n2p3_history_prediction_outcome/1", "requested": requested,
                "status": "not_available", "failure": {"kind": "history_numerical_unavailable",
                    "reason": str(error)}, "probability": None, "model_digest": model.digest(),
                "fits": 0, "response_fits": 0}
    return {"schema": "n2p3_history_prediction_outcome/1", "requested": requested,
            "status": "complete", "probability": evidence.probabilities.tolist(),
            "capability": capability.binding, "capability_digest": capability.digest,
            "prediction": binding, "model_digest": model.digest(), "fits": 0, "response_fits": 0}


def history_candidate_evidence(parent, expert_id, compiler, prefix, view, law, *, maximum_evaluated_states=100000):
    """Owned CandidateEvidence for the same requested current decision roster."""
    from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity

    outcome = predict_history(parent, expert_id, compiler, prefix, view, law,
        maximum_evaluated_states=maximum_evaluated_states)
    expert = next(expert for expert in parent._experts if expert.program.expert_id == expert_id)
    score_kind = "finite_history_conditional_tail_fixed_current_prior/1"
    identity = CandidateExpertIdentity(expert_id, semantic_sha256(outcome), expert.model_digest,
        expert.source_digest, expert.fit_digest, score_kind, False)
    counts = tuple(int(value) for value in view.decision.membership.sum(axis=0))
    observed = tuple(int(value) for value in view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0))
    arguments = {"candidate_vocabulary": view.decision.candidate_vocabulary,
        "participant_key": view.participant_key, "decision_id": view.decision_id,
        "physical_support_digest": view.physical_support_digest, "expert_identities": (identity,),
        "evidence_level": view.decision.repetitions,
        "expected_event_counts": counts, "observed_event_counts": observed}
    if outcome["status"] != "complete":
        return CandidateEvidence(log_probability=None, failure_kind=outcome["status"],
            failure_reason=json.dumps(outcome["failure"], sort_keys=True), **arguments)
    return CandidateEvidence(log_probability=np.asarray(outcome["prediction"]["log_probability"]), **arguments)


def save_history_capsule(directory, *, parent_directory, expert_id, compiler, prefix, view, law,
                         maximum_evaluated_states=100000):
    """Seal a prediction against the exact existing saved parent and source."""
    directory, parent_directory = Path(directory), Path(parent_directory)
    parent = FittedEnsemble.load(parent_directory)
    model = _owned_model(parent, expert_id)
    if model.digest() != law.record["model_digest"]:
        raise ValueError("history plan does not bind the actual saved parent producer")
    result = predict_history(parent, expert_id, compiler, prefix, view, law,
        maximum_evaluated_states=maximum_evaluated_states)
    directory.mkdir(parents=True, exist_ok=False)
    parent_payload = (parent_directory/"ensemble.json").read_bytes()
    parent_record = json.loads(parent_payload)["record"]
    copied = {}
    (directory/"parent").mkdir()
    for name in ["ensemble.json", *(row["model"] for row in parent_record["experts"] if row["model"] is not None)]:
        source = parent_directory/name
        if source.resolve().parent != parent_directory.resolve():
            raise ValueError("original parent member escapes its saved package")
        payload = source.read_bytes()
        (directory/"parent"/name).write_bytes(payload)
        copied[name] = _raw_sha(payload)
    source_bytes = compiler._history_source_payload
    (directory/"original_source.mat").write_bytes(source_bytes)
    (directory/"protocol_plan.json").write_bytes(law.plan_bytes)
    payload = {"schema": "n2p3_history_replay_capsule/1", "source_sha256": _raw_sha(source_bytes),
        "parent_manifest_sha256": _raw_sha(parent_payload), "parent_members": copied,
        "protocol_plan_sha256": _raw_sha(law.plan_bytes), "expert_id": expert_id,
        "original_ingress": compiler._history_ingress, "calibration_budget": prefix.budget,
        "current_view": view.record(), "maximum_evaluated_states": maximum_evaluated_states,
        "result": result, "old_producer_resigned": False}
    capsule_bytes = _json_bytes(payload)
    (directory/"capsule.json").write_bytes(capsule_bytes)
    return {"capsule_sha256": _raw_sha(capsule_bytes), "source_sha256": _raw_sha(source_bytes),
            "parent_manifest_sha256": _raw_sha(parent_payload), "protocol_plan_sha256": _raw_sha(law.plan_bytes)}


def replay_history_capsule(directory, *, capsule_sha256, source_sha256,
                           parent_manifest_sha256, protocol_plan_sha256):
    """Read verified original bytes once, rebuild source permission, recompute."""
    directory = Path(directory)
    capsule_bytes = (directory/"capsule.json").read_bytes()
    if _raw_sha(capsule_bytes) != capsule_sha256:
        raise ValueError("history capsule differs from external frozen consumer admission")
    record = json.loads(capsule_bytes)
    original = (directory/"original_source.mat").read_bytes()
    plan = (directory/"protocol_plan.json").read_bytes()
    parent_payload = (directory/"parent/ensemble.json").read_bytes()
    if (record["schema"] != "n2p3_history_replay_capsule/1"
            or _raw_sha(original) != source_sha256 or source_sha256 != record["source_sha256"]
            or _raw_sha(parent_payload) != parent_manifest_sha256
            or parent_manifest_sha256 != record["parent_manifest_sha256"]
            or _raw_sha(plan) != protocol_plan_sha256
            or protocol_plan_sha256 != record["protocol_plan_sha256"]):
        raise ValueError("history source/parent/protocol bytes differ from external original authority")
    for name, expected in record["parent_members"].items():
        path = directory/"parent"/name
        if path.resolve().parent != (directory/"parent").resolve() or _raw_sha(path.read_bytes()) != expected:
            raise ValueError("original parent member changed")
    arguments = dict(record["original_ingress"])
    arguments["budgets"] = tuple(arguments["budgets"])
    arguments["profiles"] = tuple(SelectionEpochProfile.from_record(p) for p in arguments["profiles"])
    arguments["authorization_preprocessing"] = parse_preprocessing_spec(arguments["authorization_preprocessing"])
    compiler = CalibrationPrefixCompiler(original, **arguments)
    prefix = compiler.compile(record["calibration_budget"])
    parent = FittedEnsemble.load(directory/"parent")
    model = _owned_model(parent, record["expert_id"])
    law = FrozenHistoryLaw.from_protocol_plan(plan, model=model, prefix=prefix)
    physical = record["current_view"]
    start, stop = physical["raw_scope"]
    samples = (None if physical.get("unavailable_reason") else
               np.asarray(compiler._record.eeg_uv[start:stop], dtype=np.float64)*1e-6)
    view = PredictionDecisionView.from_record(physical, samples=samples)
    actual = predict_history(parent, record["expert_id"], compiler, prefix, view, law,
        maximum_evaluated_states=record["maximum_evaluated_states"])
    if semantic_sha256(actual) != semantic_sha256(record["result"]):
        raise ValueError("history replay differs from actual original-source permission or numerical result")
    return actual
