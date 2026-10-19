"""Fit, persist and replay trial experts on one frozen future decision roster."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from data.candidate_task import MembershipDecision
from data.digests import array_digest, file_sha256
from data.epochs import EpochDataset
from data.study import dataset_study_membership
from models.candidate_evidence import (
    MEMBERSHIP_SCORE_METHOD,
    CandidateEvidence,
    CandidateExpertIdentity,
    membership_decision_scores,
)
from research.cloud_execution import worker_compute_phase
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, atomic_write_json
from research.source_execution import current_input_admission
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.calibration_state import FrozenTrialPredictor
from transfer.checkpoint import ScopedSourceCheckpointPreparation
from transfer.ensemble_fit import FrozenSourceCheckpoint
from transfer.recipes import TargetRecipe


@dataclass(frozen=True)
class TrialCandidateDecision:
    participant_key: str
    decision_id: str
    decision: MembershipDecision
    physical_support_digest: str
    evidence_level: int
    unavailable_reason: str | None = None
    expected_event_counts: tuple[int, ...] | None = None
    secondary_evidence_level: int | None = None

    @classmethod
    def from_arguments(cls, row):
        return cls(**{**row, "decision": MembershipDecision.from_record(row["decision"])})

    def authority_record(self):
        return {"participant_key": self.participant_key, "decision_id": self.decision_id,
            "evidence_level": self.evidence_level, "physical_support_digest": self.physical_support_digest,
            "membership_digest": semantic_sha256(self.decision.record()),
            "unavailable_reason": self.unavailable_reason, "expected_event_counts": self.expected_event_counts,
            "secondary_evidence_level": self.secondary_evidence_level}


def _evidence(request, logits, identity):
    decision = request.decision.prefix(request.evidence_level)
    if request.unavailable_reason is not None:
        observed = decision.evidence_indices >= 0
        return CandidateEvidence(
            candidate_vocabulary=decision.candidate_vocabulary, log_probability=None,
            participant_key=request.participant_key, decision_id=request.decision_id,
            physical_support_digest=request.physical_support_digest, expert_identities=(identity,),
            evidence_level=request.evidence_level,
            expected_event_counts=(tuple(request.expected_event_counts) if request.expected_event_counts is not None
                                   else tuple(decision.membership.sum(axis=0, dtype=np.int64))),
            observed_event_counts=tuple(decision.membership[observed].sum(axis=0, dtype=np.int64)),
            failure_kind="not_available", failure_reason=request.unavailable_reason,
        )
    observed = decision.evidence_indices >= 0
    values = np.zeros(len(decision.event_ids), dtype=np.float64)
    values[observed] = logits[decision.evidence_indices[observed]]
    scores = membership_decision_scores(values, decision, temperature=1.)
    return CandidateEvidence.from_membership_scores(
        scores, decision, participant_key=request.participant_key, decision_id=request.decision_id,
        physical_support_digest=request.physical_support_digest, expert_identities=(identity,),
    )


def _secondary_diagnostics(requests, logits, identity):
    rows = []
    for request in requests:
        if request.secondary_evidence_level is not None:
            level = request.secondary_evidence_level
            prefix = request.decision.prefix(level)
            expected = tuple(int(value) for value in request.decision.prefix(1).membership.sum(axis=0) * level)
            diagnostic = replace(request, evidence_level=level, unavailable_reason="terminal_partial_raw_episode",
                expected_event_counts=expected, secondary_evidence_level=None,
                physical_support_digest=semantic_sha256({"primary_support": request.physical_support_digest,
                    "prefix": prefix.record()}))
            rows.append({"endpoint": "terminal_partial_censor", "primary_evidence_level": request.evidence_level,
                "evidence": _evidence(diagnostic, logits, identity).record()})
    return rows


def execute_trial_candidate_participant(
    *, calibration: EpochDataset, future: EpochDataset,
    requests: Sequence[TrialCandidateDecision], source: FrozenSourceCheckpoint,
    recipes: Sequence[TargetRecipe], local_subject: str, output: str | Path,
    source_snapshot_sha256: str, execution_source_sha256: str,
    device: str = "cpu", excluded_source_datasets: Sequence[str] = (),
    source_seed: int | None = None,
) -> dict:
    """Keep target fitting, shared future input and independent loaded replay together.

    The dataset adapter compiles raw scope, requested keys and terminal censoring.
    This owner consumes those inputs and the existing source/fitter/model owners.
    """
    if future.y is not None:
        raise ValueError("future trial input must be label-free")
    requests, recipes = tuple(requests), tuple(recipes)
    if not requests:
        raise ValueError("trial participant requires its original requested decisions")
    if len({(item.participant_key, item.decision_id) for item in requests}) != len(requests):
        raise ValueError("future trial requests contain duplicate physical decisions")
    if len({recipe.name for recipe in recipes}) != len(recipes):
        raise ValueError("trial recipes require distinct declared arms")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    admission = current_input_admission()
    mask = (future.trial_channel_mask if future.trial_channel_mask is not None
            else np.broadcast_to(future.channel_mask, future.X.shape[:2]))
    calibration_mask = (calibration.trial_channel_mask if calibration.trial_channel_mask is not None
                        else np.broadcast_to(calibration.channel_mask, calibration.X.shape[:2]))
    input_digest = semantic_sha256({"samples": array_digest(future.X), "observed": array_digest(mask),
                                   "timeline": future.event_timeline.fingerprint()})
    owner = ScopedSourceCheckpointPreparation()
    arms = [{"name": "source_zero_shot", "recipe": None},
            *[{"name": recipe.name, "recipe": recipe} for recipe in recipes]]
    result = {"schema": "n2p3_trial_candidate_participant/1", "status": "running",
              "local_subject": local_subject, "source_checkpoint_sha256": source.digest,
              "source_snapshot_sha256": source_snapshot_sha256,
              "execution_source_sha256": execution_source_sha256,
              "source_seed": source_seed,
              "future_input_digest": input_digest,
              "calibration_input_digest": array_digest(calibration.X),
              "calibration_labels_digest": array_digest(calibration.y),
              "calibration_observed_digest": array_digest(calibration_mask),
              "calibration_timeline_digest": calibration.event_timeline.fingerprint(),
              "calibration_provenance_digest": semantic_sha256(calibration.provenance),
              "calibration_metadata_digest": semantic_sha256(calibration.metadata.to_dict(orient="records")),
              "recipes": [recipe.record() for recipe in recipes],
              "requested": [item.authority_record() for item in requests], "arms": []}
    request = {key: value for key, value in result.items() if key not in {"arms", "status"}}
    request_digest = semantic_sha256(request)
    request_path = destination / "request.json"
    if request_path.exists():
        import json

        if semantic_sha256(json.loads(request_path.read_bytes())) != request_digest:
            raise ValueError("partial trial participant differs from the actual input/recipe/source request")
    else:
        atomic_write_json(request_path, request, input_admission=admission)
    for arm in arms:
        recipe = arm["recipe"]
        binding_data = future if recipe is None else calibration
        prepared = owner.prepare(source.content, binding_data, local_subject)
        if prepared.payload["training_contract"]["source_snapshot_sha256"] != source_snapshot_sha256:
            raise ValueError("declared source snapshot differs from the admitted checkpoint producer")
        if source_seed is not None and prepared.payload["training_contract"]["seed"] != source_seed:
            raise ValueError("declared source seed differs from the admitted checkpoint producer")
        prepared.assert_library_exclusion(target_membership=dataset_study_membership(binding_data),
                                          excluded_source_datasets=excluded_source_datasets)
        fit_record = {"arm": arm["name"], "recipe": None if recipe is None else recipe.record(),
                      "architecture_control_checkpoint_sha256": source.digest,
                      "initialization_checkpoint_sha256": (None if recipe is not None and recipe.initialization == "scratch"
                                                           else source.digest),
                      "source_snapshot_sha256": source_snapshot_sha256,
                      "execution_source_sha256": execution_source_sha256,
                      "calibration_labels_used": 0 if recipe is None else calibration.n_epochs,
                      "calibration_identity_digest": binding_data.identity_table.digest(),
                      "calibration_input_digest": None if recipe is None else array_digest(calibration.X),
                      "calibration_labels_digest": None if recipe is None else array_digest(calibration.y),
                      "calibration_observed_digest": None if recipe is None else array_digest(calibration_mask),
                      "calibration_timeline_digest": None if recipe is None else calibration.event_timeline.fingerprint(),
                      "calibration_provenance_digest": None if recipe is None else semantic_sha256(calibration.provenance),
                      "calibration_metadata_digest": None if recipe is None else semantic_sha256(calibration.metadata.to_dict(orient="records")),
                      "future_input_digest": input_digest}
        path = destination / f"{arm['name']}.pt"
        receipt_path = destination / f"{arm['name']}.receipt.json"
        if receipt_path.exists():
            import json

            entry = json.loads(receipt_path.read_bytes())
            if entry["request_digest"] != request_digest or entry["arm"] != arm["name"]:
                raise ValueError("saved trial arm differs from its original participant request")
            if any(entry["fit"][key] != value for key, value in fit_record.items()):
                raise ValueError("saved trial fit differs from the actual source/recipe/input authority")
            if entry["status"] == "completed":
                if file_sha256(path) != entry["model_artifact"]["sha256"]:
                    raise ValueError("saved trial arm model bytes changed")
                loaded = FrozenTrialPredictor.load(path)
                if loaded.digest() != entry["model_digest"]:
                    raise ValueError("saved trial arm numeric model changed")
                binding = {"request_digest": request_digest,
                           "fit": {key: value for key, value in entry["fit"].items() if key != "model_digest"}}
                if semantic_sha256(loaded.record["ensemble_fit_binding"]) != semantic_sha256(binding):
                    raise ValueError("saved trial model differs from its actual fit binding")
                if recipe is not None:
                    contract = loaded.record["class_calibration"]["fit_contract"]
                    actual_fit = {key: value for key, value in contract.items() if key != "frozen_state_digest"}
                    if (semantic_sha256(actual_fit) != semantic_sha256(entry["fit"]["fit_history"]["probability_fit"])
                            or semantic_sha256(actual_fit["recipe"]) != semantic_sha256(recipe.record())):
                        raise ValueError("saved trial model actual probability fit differs from its declared target recipe")
                with worker_compute_phase(device):
                    replay = loaded.predict_logit(future.X, mask, device=device)
                identity = CandidateExpertIdentity(**entry["predictions"][0]["expert_identities"][0])
                expected_source = source.digest if recipe is None or recipe.initialization == "source" else None
                if identity != CandidateExpertIdentity(arm["name"], input_digest, loaded.digest(),
                        expected_source, semantic_sha256(entry["fit"]), MEMBERSHIP_SCORE_METHOD, False):
                    raise ValueError("saved trial expert differs from its actual fitted numeric authority")
                if [_evidence(item, replay, identity).record() for item in requests] != entry["predictions"]:
                    raise ValueError("saved trial arm predictions differ from independent loaded replay")
                if _secondary_diagnostics(requests, replay, identity) != entry["secondary_diagnostics"]:
                    raise ValueError("saved trial secondary prefix differs from independent loaded replay")
            elif entry["status"] != "fit_failure":
                raise ValueError("saved trial arm lacks a completed or expected-failure terminal")
            result["arms"].append(entry)
            continue
        try:
            if recipe is None:
                predictor = FrozenTrialPredictor.from_source(prepared, local_subject)
            else:
                with worker_compute_phase(device):
                    fitted = fit_calibration_candidate(
                        calibration, recipe, np.ones(calibration.n_epochs, dtype=bool),
                        source_trunk=prepared.trunk_for_target(local_subject),
                        source_input_stats=prepared.input_statistics, device=device,
                    )
                predictor = FrozenTrialPredictor.from_fitted(fitted)
                fit_record["fit_history"] = fitted.fit_history
            predictor = predictor.with_fit_binding({"request_digest": request_digest, "fit": fit_record})
            predictor.save(path)
            loaded = FrozenTrialPredictor.load(path)
            model_digest = predictor.digest()
            if loaded.digest() != model_digest:
                raise ValueError("loaded trial model differs from the actual fitted numerical state")
            with worker_compute_phase(device):
                logits = predictor.predict_logit(future.X, mask, device=device)
                replay = loaded.predict_logit(future.X, mask, device=device)
            np.testing.assert_array_equal(logits, replay)
            fit_record["model_digest"] = model_digest
            fit_digest = semantic_sha256(fit_record)
            source_model_digest = source.digest if recipe is None or recipe.initialization == "source" else None
            identity = CandidateExpertIdentity(arm["name"], input_digest, model_digest,
                                               source_model_digest, fit_digest, MEMBERSHIP_SCORE_METHOD, False)
            predictions = [_evidence(item, logits, identity) for item in requests]
            replay_predictions = [_evidence(item, replay, identity) for item in requests]
            if [item.record() for item in predictions] != [item.record() for item in replay_predictions]:
                raise ValueError("loaded trial replay differs on the complete candidate probability roster")
            entry = {"arm": arm["name"], "status": "completed", "fit": fit_record,
                     "request_digest": request_digest,
                     "model_artifact": {"path": path.name, "sha256": file_sha256(path)},
                     "model_digest": model_digest, "saved_replay_exact": True,
                     "predictions": [item.record() for item in predictions],
                     "secondary_diagnostics": _secondary_diagnostics(requests, logits, identity)}
        except ExpectedSubjectError as error:
            entry = {"arm": arm["name"], "status": "fit_failure", "fit": fit_record,
                     "request_digest": request_digest,
                     "failure": error.record(subject=local_subject), "predictions": []}
        atomic_write_json(receipt_path, entry, input_admission=admission)
        result["arms"].append(entry)
    result.update(status="completed", source_preparation=owner.record())
    atomic_write_json(destination / "result.json", result, input_admission=admission)
    return result
