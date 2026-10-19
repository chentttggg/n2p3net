"""Export actual shared-owner C/held views for the paired frozen-response audit.

This builder does not fit a response, noise, drift or temperature. Native public
inputs are rebuilt by the existing BNCI owner; fitted modern models are supplied
by their original producer. C residual nuisance is explicitly zero in this first
program, rather than inferred from unavailable historical drift state.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from experiments.audit_channel_noise_mechanism import (  # noqa: E402
    PAIRED_PROGRAM,
    WINNER_PROGRAM,
    BoundInput,
    _executing_source,
    _sha,
    paired_input_digests,
    public_calibration_authority,
)
from research.channel_noise_mechanism import FrozenERPResponse, NoiseGeometry  # noqa: E402
from research.contracts import (  # noqa: E402
    DecisionPlanContract,
    DecisionPlanEntry,
    ParticipantSelectionFailure,
)
from research.execution import atomic_write_json  # noqa: E402
from transfer.continuous_decision import ContinuousDecisionModel, _event_design  # noqa: E402
from transfer.ensemble_views import (  # noqa: E402
    CalibrationSelection,
    PredictionDecisionView,
    prepare_bnci_calibration_groups,
)
from transfer.noise_model import array_digest  # noqa: E402


def build_paired_noise_input(participants, *, truth, output, evidence_scope,
                             seed=20260918, source_records=(), selection_failures=()):
    """Lower admitted actual views; external truth goes only into DecisionPlan.

    Each participant supplies a modern model path, complete actual fit groups
    and held PredictionDecisionViews. All requested participants are in truth,
    including explicitly declared selection failures with no resolved decisions.
    """
    if evidence_scope not in {"software_correctness", "public_calibration_development"}:
        raise ValueError("paired builder requires explicit software/public calibration scope")
    if type(seed) is not int or seed < 0:
        raise ValueError("paired program seed must be a declared nonnegative integer")
    if evidence_scope == "public_calibration_development" and not source_records:
        raise ValueError("public builder requires original MAT source declarations")
    failures = tuple(selection_failures)
    if any(not isinstance(row, ParticipantSelectionFailure) for row in failures):
        raise TypeError("selection failure ledger must use the shared owner")
    before = _executing_source()
    output = Path(output).resolve()
    original_inputs = BoundInput(output / "input.json")
    authority = (public_calibration_authority(source_records, original_inputs)
                 if evidence_scope == "public_calibration_development" else None)
    if authority is not None:
        source_people = {group.view.participant_key for group in authority.values()}
        if set(truth) != source_people:
            raise ValueError("public requested cohort differs from actual original source participants")
    output.mkdir(parents=True, exist_ok=False)
    rows, views, entries, people = [], [], [], set()
    vocabulary = level = None
    bound_models = {}

    def arrays(name, **values):
        path = output / name
        np.savez_compressed(path, **values)
        return {"path": name, "sha256": _sha(path), **{f"{key}_key": key for key in values}}

    for index, participant in enumerate(participants):
        training, held = tuple(participant["training"]), tuple(participant["held"])
        if (not training or not held or any(not isinstance(g, CalibrationSelection) for g in training)
                or any(not isinstance(v, PredictionDecisionView) for v in held)):
            raise ValueError("paired builder requires actual labeled C and label-free held owner types")
        person = training[0].view.participant_key
        if person in people or any(v.participant_key != person for v in [g.view for g in training] + list(held)):
            raise ValueError("one fixed response requires one unique participant")
        people.add(person)
        if authority is not None:
            for group in training:
                actual = authority.get(group.view.record_id)
                if (actual is None or actual.view.physical_support_digest != group.view.physical_support_digest
                        or actual.view.raw_input_digest != group.view.raw_input_digest
                        or group.target_candidate != actual.target_candidate
                        or not np.array_equal(group.labels, actual.labels)):
                    raise ValueError("paired builder C labels/target/view differ from actual original source owner")
            for view in held:
                actual = authority.get(view.record_id)
                if (actual is None or actual.view.physical_support_digest != view.physical_support_digest
                        or actual.view.raw_input_digest != view.raw_input_digest
                        or truth[person][view.decision_id] != actual.target_candidate):
                    raise ValueError("paired builder held view/truth differs from actual original source owner")
        model_path = Path(participant["model"]).resolve()
        content = model_path.read_bytes()
        model_sha = hashlib.sha256(content).hexdigest()
        bound_models[model_path] = model_sha
        model = ContinuousDecisionModel.load(io.BytesIO(content))
        view = training[0].view
        geometry = NoiseGeometry(view.channel_names, view.sample_rate_hz, view.signal_unit,
                                 view.reference, model.operator_identity, len(view.channel_names))
        if model.conditioning_samples != 32 or model.noise is None or model.noise.order != 8:
            raise ValueError("paired anchor must be the original order8/prefix32 response")
        records = [{"record_id": g.view.record_id, "participant_key": person,
            "recording_id": g.view.source_sha256, "read_support": list(g.view.raw_scope),
            "fit_scope": "target_calibration", "source_sha256": g.view.source_sha256,
            "input_sha256": array_digest(g.view.samples, g.view.onsets, g.labels),
            "n_samples": len(g.view.samples), "original_view": g.view.record()} for g in training]
        FrozenERPResponse.from_continuous_model(model, geometry=geometry, source_model_sha256=model_sha,
            response_id="fixed_response", training_records=records)
        training_rows, held_rows = [], []
        for group_index, group in enumerate(training):
            view = group.view
            descriptor = arrays(f"C_{index:03d}_{group_index:03d}.npz", samples=view.samples,
                onsets=view.onsets, labels=group.labels,
                design=_event_design(view.onsets, group.labels, len(view.samples), model.basis),
                nuisance=np.zeros_like(view.samples))
            training_rows.append({"view": view.record(), "arrays": descriptor, "target_candidate": group.target_candidate})
        for held_index, view in enumerate(held):
            if vocabulary is None:
                vocabulary, level = view.decision.candidate_vocabulary, view.decision.repetitions
            if view.decision.candidate_vocabulary != vocabulary or view.decision.repetitions != level:
                raise ValueError("paired builder cannot mix physical tasks or repetitions")
            values = {"samples": np.empty((0, len(view.channel_names))) if view.samples is None else view.samples}
            if view.samples is not None:
                values["candidate_designs"] = np.asarray([_event_design(view.onsets,
                    view.decision.membership[:, candidate], len(view.samples), model.basis)
                    for candidate in range(len(vocabulary))])
            held_rows.append({"view": view.record(), "arrays": arrays(f"held_{index:03d}_{held_index:03d}.npz", **values)})
            entries.append(DecisionPlanEntry(person, view.decision_id, truth[person][view.decision_id]))
        views.extend([group.view for group in training] + list(held))
        rows.append({"geometry": geometry.record(), "response": {"response_id": "fixed_response",
            "source_model": {"path": str(model_path), "sha256": model_sha, "format": "modern_continuous_model_npz"}},
            "training": training_rows, "held": held_rows})
    if not entries:
        raise ValueError("paired builder requires at least one resolved actual decision")
    actual_truth_keys = {(entry.participant_key, entry.decision_id) for entry in entries}
    declared_truth_keys = {(person, decision) for person, decisions in truth.items() for decision in decisions}
    if actual_truth_keys != declared_truth_keys or people | {row.participant_key for row in failures} != set(truth):
        raise ValueError("external requested truth/participant roster differs from actual held views")
    cache, identity = paired_input_digests(views)
    plan = DecisionPlanContract(cache, identity, tuple(truth), tuple(entries), failures)
    if people & {row.participant_key for row in failures} or people | {row.participant_key for row in failures} != set(truth):
        raise ValueError("resolved and selection-failed participants must partition the complete requested cohort")
    payload = {"schema": "n2p3_channel_noise_audit_input/2", "evidence_scope": evidence_scope,
        "program": {"id": PAIRED_PROGRAM, "winner_program": WINNER_PROGRAM, "seed": seed,
            "conditioning_samples": 32, "noise_order": 8, "residual_nuisance": "explicit_zero_C_nuisance/1",
            "response_rule": "same_saved_physical_M_R_original_Sigma_for_both_noise_arms",
            "C_residual_interpretation": "no_C_side_nuisance_subtraction_mixed_residual_may_include_drift_or_ERP_misspecification",
            "noise_estimator_scope": "same_C_residuals_Burg_then_common_tail_LedoitWolf_not_joint_MLE"},
        "candidate_vocabulary": list(vocabulary), "evidence_level": level, "decision_plan": plan.record(),
        "source_records": list(source_records), "participants": rows,
        "noise_arms": [{"arm_id": "shared8", "family": "shared", "order": 8},
                       {"arm_id": "private8", "family": "private", "order": 8}],
        "builder_source": before}
    if _executing_source() != before or any(_sha(path) != digest for path, digest in bound_models.items()):
        raise RuntimeError("paired builder executing source/model changed during export")
    original_inputs.verify_unchanged()
    manifest = output / "input.json"
    atomic_write_json(manifest, payload)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True,
                        help="JSON records with original MAT source descriptor, existing modern model and owner parameters")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.source_spec.read_bytes())
    participants, truth, sources = [], {}, []
    for row in spec["records"]:
        if not row.get("model"):
            raise ValueError("native preparation has no fitted modern response model; complete source review first")
        source = {**row["source"], "path": str((args.source_spec.parent / row["source"]["path"]).resolve())}
        groups = prepare_bnci_calibration_groups(source["path"], source_sha256=source["sha256"], **row["owner"])
        if len(groups.groups) != 6:
            raise ValueError("native paired source specification requires six complete groups")
        person = groups.groups[0].view.participant_key
        if person in truth:
            raise ValueError("native source specification repeats a participant")
        participants.append({"model": (args.source_spec.parent / row["model"]).resolve(),
            "training": groups.groups[:5], "held": [groups.groups[5].view]})
        truth[person] = {groups.groups[5].view.decision_id: groups.groups[5].target_candidate}
        sources.append({"source": source, "owner": row["owner"]})
    path = build_paired_noise_input(participants, truth=truth, output=args.output,
        evidence_scope="public_calibration_development", seed=spec["seed"], source_records=sources)
    print(json.dumps({"status": "exported", "input_manifest": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
