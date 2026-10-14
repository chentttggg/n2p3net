"""Same-run persistence and replay for ``CalibrationPrefix`` producers.

This contract is deliberately separate from strict nested selection.  A prefix
fit owns one chronological ``K`` (including K0), its right-censored raw/trial
authority, optional complete C groups used by grouped temperature OOF, and a
requested-K alias ledger.  The recorder is attached to the original fit call;
loading this receipt never invokes a numerical fitter.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, atomic_write_json
from research.nested_selection_replay import InnerReplayRecorder, _group_identity
from transfer.ensemble import EnsembleProgram, EnsembleScorer
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    FrozenSourceCheckpoint,
    _calibration_expert_fit_digest,
)
from transfer.ensemble_views import CalibrationPrefix, PredictionDecisionView
from transfer.noise_model import array_digest

SCHEMA = "n2p3_prefix_inner_replay/2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_sources(source_checkpoints):
    if source_checkpoints is None:
        return {}
    return {name: (value if isinstance(value, FrozenSourceCheckpoint)
                   else FrozenSourceCheckpoint.from_path(value))
            for name, value in source_checkpoints.items()}


def _source_roles(expert_programs, sources):
    roles = {}
    for spec in expert_programs:
        source = sources.get(spec.expert_id)
        role = None
        if source is not None:
            role = ("pretrained_weights" if spec.family == "trial"
                    and spec.recipe is not None and spec.recipe.initialization == "source"
                    else "architecture_only")
        roles[spec.expert_id] = {"digest": None if source is None else source.digest, "role": role}
    return roles


def _array_identity(value, *extra):
    if value is None:
        return None
    return array_digest(value, *extra)


def actual_expert_fit_inputs(spec, data, raw_records, target, *, source_only=False):
    """Expected native fitter inputs from the actual authorized arrays.

    These are distinct from the consumer wrapper's programme/receipt digest.
    Spatial raw operands and trial byte order use their original owners.
    """
    from transfer.calibration_candidates import _observed, waveform_lda_training_selection
    from transfer.ensemble_views import project_raw_observations

    def training_sha(samples, labels, observed):
        digest = hashlib.sha256()
        for value in (samples, labels, observed):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    if source_only:
        return {"origin": "source_only_or_unavailable", "actual_K": 0}
    if spec.family == "continuous_ar":
        records = []
        for raw in raw_records:
            values, _, _, _ = project_raw_observations(raw.samples,
                target["channel_names"], target["reference"], spec.spatial,
                sample_rate_hz=target["sample_rate_hz"])
            records.append({"record_id": raw.record_id, "n_samples": len(values),
                "n_events": len(raw.onsets), "n_targets": int(raw.target_mask.sum()),
                "input_sha256": array_digest(values, raw.onsets, raw.target_mask),
                "raw_input_sha256": array_digest(values, raw.onsets, raw.target_mask),
                "original_input_sha256": array_digest(raw.samples, raw.onsets, raw.target_mask),
                "right_censored": raw.right_censored, "read_sample_support": [0, len(values)]})
        return {"origin": "target_continuous", "records": records}
    if data is None:
        return {"origin": "unavailable_trial_rows"}
    observed = _observed(data)
    channels, training, _ = waveform_lda_training_selection(data, np.ones(data.n_epochs, dtype=bool))
    return {"origin": "target_trial",
        "original_numerical_input_digest": array_digest(data.X, data.y, data.trial_channel_mask),
        "full_training_data_sha256": training_sha(data.X, data.y, observed),
        "full_class_counts": np.bincount(data.y, minlength=2).tolist(),
        "waveform_training_data_sha256": training_sha(data.X[training][:, channels],
            data.y[training], observed[training][:, channels]),
        "waveform_class_counts": np.bincount(data.y[training], minlength=2).tolist(),
        "waveform_channels": channels.tolist()}


def prefix_native_fit_inputs(prefix, expert_programs):
    return {spec.expert_id: actual_expert_fit_inputs(spec,
        prefix.trial_datasets.get(spec.profile.digest) if spec.family == "trial" else None,
        [] if prefix.raw is None else [prefix.raw], prefix.target_contract, source_only=prefix.budget == 0)
        for spec in expert_programs}


def check_native_fit_inputs(expected_inputs, model, *, actual_K=None):
    """Reject a real different-dose/label producer with a reassigned wrapper."""
    for expert in model._experts:
        if expert.predictor is None:
            continue
        expected = expected_inputs[expert.program.expert_id]
        if expert.program.family == "continuous_ar":
            actual = expert.predictor.provenance["records"]
            if expert.program.spatial is None and any(row["input_sha256"] != row["original_input_sha256"]
                                                      for row in expected["records"]):
                raise ValueError("native raw input capture is not the original unprojected observations")
            if len(actual) != len(expected["records"]) or any(
                any(semantic_sha256(saved.get(key)) != semantic_sha256(value)
                    for key, value in wanted.items() if key != "original_input_sha256")
                for saved, wanted in zip(actual, expected["records"], strict=True)):
                raise ValueError("prefix native continuous fitter input/dose differs from authorized observations")
        elif actual_K != 0:
            predictor = expert.predictor
            calibration = predictor.record["class_calibration"]
            fit = calibration["fit_contract"]
            generative = predictor.record["kind"] != "n2p3"
            stem = "waveform" if generative else "full"
            if (fit["training_data_sha256"] != expected[f"{stem}_training_data_sha256"]
                    or tuple(fit["class_counts"]) != tuple(expected[f"{stem}_class_counts"])
                    or semantic_sha256(fit["recipe"]) != semantic_sha256(expert.program.recipe.record())):
                raise ValueError("prefix native trial fitter samples/labels/recipe differ from authorized observations")
            if generative and tuple(predictor._state["waveform_channels"].numpy()) != tuple(expected["waveform_channels"]):
                raise ValueError("prefix native trial fitter observed channel scope changed")
        elif expert.predictor.record.get("class_calibration", {}).get("origin") != "source_mixture":
            raise ValueError("K0 native predictor is not the actual source-only numerical programme")


def prefix_input_identity(prefix: CalibrationPrefix) -> dict[str, Any]:
    """Return numerical and label identities for one frozen prefix.

    ``CalibrationPrefix.record`` is intentionally compact and omits numerical
    arrays.  This identity adds independent sample/label digests so a stale
    shell digest cannot authorize replay after raw, trial, or group mutation.
    """
    if not isinstance(prefix, CalibrationPrefix):
        raise TypeError("prefix replay requires an authorized CalibrationPrefix")
    raw = prefix.raw
    raw_identity = None
    if raw is not None:
        raw_identity = {
            "record_id": raw.record_id,
            "raw_scope": list(prefix.target_contract["raw_scope"]),
            "right_censored": bool(raw.right_censored),
            "available_sample": prefix.target_contract.get("available_sample", prefix.target_contract["raw_scope"][1]),
            "samples_sha256": _array_identity(raw.samples, raw.onsets),
            "fit_input_digest": array_digest(raw.samples, raw.onsets, raw.target_mask),
            "onsets_sha256": _array_identity(raw.onsets),
            "event_onsets": raw.onsets.tolist(),
            "labels_sha256": _array_identity(raw.target_mask),
            "label_use": prefix.label_use["raw"],
        }
    trial_identity = {}
    for digest, data in prefix.trial_datasets.items():
        trial_identity[digest] = {
            "digest": digest,
            "numerical_input_digest": _array_identity(data.X, data.y, data.trial_channel_mask),
            "used_label_indices": prefix.label_use["trial"][digest].get("used_label_indices", []),
            "label_use": prefix.label_use["trial"][digest],
        }
    groups = [_group_identity(group) for group in prefix.complete_groups]
    return {
        "budget": prefix.budget,
        "target_contract": prefix.target_contract,
        "label_use": prefix.label_use,
        "raw": raw_identity,
        "trial_datasets": trial_identity,
        "complete_groups": groups,
        "prefix_record_sha256": semantic_sha256(prefix.record()),
    }


def future_input_identity(view):
    if not isinstance(view, PredictionDecisionView):
        raise TypeError("prefix future replay requires a truth-free PredictionDecisionView")
    actual = view.record()
    # The complete owner record (including membership, onsets and timing) is
    # bound by one semantic digest.  Future score rows need no second expanded
    # copy of its membership matrix for every K/strategy/seed.
    fields = ("record_id", "participant_key", "decision_id", "source_dataset_id", "source_subject_id",
              "source_sha256", "raw_scope", "available_sample", "physical_support_digest", "raw_input_digest")
    return {"view": {key: actual[key] for key in fields}, "view_record_sha256": semantic_sha256(actual),
            "samples_sha256": None if view.samples is None else array_digest(view.samples, view.onsets)}


def _final_identity(model):
    from research.calibration_selection import _final_identity as identity

    return identity(model)


def _fit_record(value, *, participant_key):
    if isinstance(value, ExpectedSubjectError):
        return {"status": "unavailable", "failure": value.record(subject=participant_key)}
    if not isinstance(value, FittedEnsemble):
        raise TypeError("prefix programme values must be FittedEnsemble or ExpectedSubjectError")
    return {
        "status": "completed",
        "strategy": value.probability_strategy,
        "temperature_calibration": value.temperature_record,
        "model_identity": _final_identity(value),
        "oof_probability_sha256": None if value.oof_probability_record is None else
            semantic_sha256(value.oof_probability_record),
        **({"pool_calibration": value.scorer.pool_calibration} if value.scorer.pool_calibration is not None else {}),
        **({"pool_OOF_probability_sha256": semantic_sha256(value.pool_oof_record)} if value.pool_oof_record is not None else {}),
    }


def _check_aliases(prefix, context):
    aliases = context.get("requested_Ks", context.get("requested_ks", [prefix.budget]))
    aliases = [int(value) for value in aliases]
    if not aliases or len(set(aliases)) != len(aliases) or any(value < 0 for value in aliases):
        raise ValueError("prefix replay requested_K aliases must be distinct nonnegative integers")
    mapping = context.get("requested_to_actual_K")
    if mapping is not None:
        actual = {str(key): int(value) for key, value in mapping.items()}
        if any(str(alias) not in actual or actual[str(alias)] != prefix.budget for alias in aliases):
            raise ValueError("requested K alias does not resolve to actual prefix K")
    return aliases, None if mapping is None else {str(k): int(v) for k, v in mapping.items()}


def _check_seeds(specs, sources, context):
    from transfer.checkpoint import load_checkpoint_payload

    target_seed, source_seed = context.get("target_seed"), context.get("source_seed")
    if target_seed is not None and any(spec.recipe is not None and spec.recipe.seed != target_seed for spec in specs):
        raise ValueError("prefix target seed differs from the actual recipe seed")
    actual = {}
    for name, source in sources.items():
        payload = load_checkpoint_payload(io.BytesIO(source.content))
        seed = payload.get("training_contract", {}).get("seed")
        if source_seed is not None and seed != source_seed:
            raise ValueError("prefix source seed differs from the actual checkpoint seed")
        actual[name] = {"digest": source.digest, "seed": seed}
    return actual


def _validate_future(view, record):
    identity = future_input_identity(view)
    target = record["prefix_input"]["target_contract"]
    if (view.participant_key != target["participant_key"] or view.source_sha256 != target["source_sha256"]
            or tuple(view.decision.candidate_vocabulary) != tuple(target["candidate_vocabulary"])
            or (view.samples is not None and view.raw_scope[0] < target["raw_scope"][1])):
        raise ValueError("prefix future view source identity or raw support differs from authorized test scope")
    test_scope = record["fit_context"].get("test_scope")
    if test_scope is None:
        raise ValueError("prefix future replay requires the actual common independent test scope")
    if view.samples is not None and (view.raw_scope[0] < test_scope["start_sample"]
            or view.raw_scope[1] > test_scope["stop_sample"]):
        raise ValueError("prefix future view exceeds the declared independent test scope")
    return identity


def _prefix_record(identity):
    return {"budget": identity["budget"], "target_contract": identity["target_contract"],
            "label_use": identity["label_use"],
            "temperature_group_records": [row["group"] for row in identity["complete_groups"]]}


def _check_final_authority(record, loaded):
    """Bind receipts back to the numerical package's original fit contracts."""
    identity = record["prefix_input"]
    authority = _prefix_record(identity)
    if (semantic_sha256(authority) != record["prefix_record_sha256"]
            or semantic_sha256(authority) != identity["prefix_record_sha256"]
            or record["actual_K"] != identity["budget"]
            or record["participant_key"] != identity["target_contract"]["participant_key"]):
        raise ValueError("prefix authority receipt changed")
    specs = tuple(CalibrationExpertProgram.from_record(row) for row in record["expert_programs"])
    program = EnsembleProgram.from_record(record["ensemble_program"])
    if tuple(spec.expert_id for spec in specs) != program.expert_ids:
        raise ValueError("prefix declared expert programme/roster changed")
    expected_roles = {}
    for spec in specs:
        digest = record["source_checkpoint_digests"].get(spec.expert_id)
        expected_roles[spec.expert_id] = {"digest": digest, "role": None if digest is None else
            "pretrained_weights" if spec.recipe.initialization == "source" else "architecture_only"}
    if expected_roles != record["source_checkpoint_roles"]:
        raise ValueError("prefix source checkpoint role/digest changed")
    target_seed = record["fit_context"].get("target_seed")
    if target_seed is not None and any(spec.recipe is not None and spec.recipe.seed != target_seed for spec in specs):
        raise ValueError("prefix target seed differs from actual programme recipe")
    expected_receipts = [] if identity["raw"] is None else [{**identity["target_contract"],
        "record_id": identity["raw"]["record_id"], "raw_input_digest": identity["raw"]["fit_input_digest"],
        "label_use": identity["label_use"]}]
    for digest, trial in identity["trial_datasets"].items():
        if trial["numerical_input_digest"] != identity["label_use"]["trial"][digest]["numerical_input_digest"]:
            raise ValueError("prefix actual trial samples/labels differ from fit authority")
    for spec in specs:
        native = record["native_fit_inputs"][spec.expert_id]
        if spec.family == "continuous_ar" and identity["raw"] is not None:
            rows = native["records"]
            if (len(rows) != 1 or rows[0]["record_id"] != identity["raw"]["record_id"]
                    or rows[0]["n_events"] != record["actual_K"]
                    or rows[0]["n_samples"] != identity["target_contract"]["raw_scope"][1]
                    or rows[0]["original_input_sha256"] != identity["raw"]["fit_input_digest"]):
                raise ValueError("native continuous input capture differs from original prefix authority")
        elif spec.family == "trial" and spec.profile.digest in identity["trial_datasets"]:
            if native["original_numerical_input_digest"] != identity["trial_datasets"][spec.profile.digest]["numerical_input_digest"]:
                raise ValueError("native trial input capture differs from original prefix authority")
    for model in loaded.values():
        if (model.scorer.program != program
                or semantic_sha256([expert.program.record() for expert in model._experts]) !=
                    semantic_sha256([spec.record() for spec in specs])
                or semantic_sha256(model._target_contract) != semantic_sha256(identity["target_contract"])
                or semantic_sha256(model.fit_receipts) != semantic_sha256(expected_receipts)):
            raise ValueError("prefix declaration or raw numerical input differs from actual package fit authority")
        for expert in model._experts:
            digest = record["source_checkpoint_digests"].get(expert.program.expert_id)
            weights_from_source = (expert.program.family == "trial" and expert.program.recipe.initialization == "source")
            if expert.source_digest != (digest if weights_from_source and expert.predictor is not None else None):
                raise ValueError("prefix numerical producer source weights digest changed")
            if expert.predictor is not None:
                expected = _calibration_expert_fit_digest(expert.program, authority, digest)
            else:
                kind = "fit_failure" if expert.unavailable_kind == "fit_failure" else "unavailable"
                expected = semantic_sha256({"program": expert.program.record(), "prefix": authority,
                                           kind: expert.unavailable_reason})
            if expert.fit_digest != expected:
                raise ValueError("prefix label/source authority differs from numerical producer fit binding")
        check_native_fit_inputs(record["native_fit_inputs"], model, actual_K=record["actual_K"])
    _check_oof_temperature_learning(loaded)
    return authority


def _check_oof_temperature_learning(loaded):
    """Bind saved temperatures to the original captured learner, once per OOF.

    This small meta-objective replay never fits an EEG model, consumes future
    scores, modifies frozen state, or silently changes the declared estimator.
    """
    from transfer.ensemble_fit import temperature_from_oof

    seen = set()
    for model in loaded.values():
        oof = model.oof_probability_record
        if oof is None or oof["schema"] != "n2p3_complete_group_probability_oof/1" or semantic_sha256(oof) in seen:
            continue
        seen.add(semantic_sha256(oof))
        groups = {group["view"]["record_id"]: group for group in oof["calibration_groups"]}
        calibration = oof["temperature_calibration"]
        selected = tuple(oof["captured_expert_ids"])
        if (tuple(calibration["temperature_expert_ids"]) != selected
                or tuple(calibration["temperature_bounds"]) != (.05, 20.)
                or calibration["score_transform"] != "posterior_power_temperature/1"
                or len(oof["rows"]) != len(groups)
                or len(groups) < 3):
            raise ValueError("prefix temperature learner declaration differs from original OOF programme")
        for name in selected:
            values, targets = [], []
            for row in oof["rows"]:
                evidence = row["expert_evidence"][name]
                values.append(evidence["log_probability"])
                targets.append(tuple(evidence["candidate_vocabulary"]).index(groups[row["held_record_id"]]["target_candidate"]))
            expected = temperature_from_oof(np.asarray(values, dtype=np.float64), np.asarray(targets, dtype=int))
            if expected != calibration["temperatures"][name]:
                raise ValueError("prefix saved temperature is not the original captured OOF learning result")
        for name in calibration["fixed_control_expert_ids"]:
            if calibration["temperatures"][name] != 1.:
                raise ValueError("prefix fixed temperature control changed its frozen unit value")


def _inner_path(directory, producer):
    raw = directory / producer["package"]
    expected = directory / "inner_models" / producer["fit_key"]
    if (raw.is_symlink() or any(parent.is_symlink() for parent in raw.parents if parent != directory.parent)
            or raw.resolve() != expected.resolve() or not raw.resolve().is_relative_to(directory.resolve())
            or len(producer["fit_key"]) != 64 or any(char not in "0123456789abcdef" for char in producer["fit_key"])):
        raise ValueError("prefix inner producer package path escapes its owned fit key")
    return raw


def unavailable_prefix_expert_scores(prefix_record, spec_records, strategy, failure, view):
    """Keep the complete original expert roster for an unavailable strategy."""
    counts = tuple(int(count) for count in view.decision.membership.sum(axis=0))
    observed = tuple(int(count) for count in view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0))
    error = ExpectedSubjectError(failure["error_code"], stage=failure["stage"], detail=failure["detail"])
    return {spec["expert_id"]: CandidateEvidence(view.decision.candidate_vocabulary, None,
        view.participant_key, view.decision_id, view.physical_support_digest,
        (CandidateExpertIdentity(spec["expert_id"], view.raw_input_digest, None, None,
            semantic_sha256({"prefix": prefix_record, "program": spec, "strategy": strategy}),
            f"temperature:{error.code.value}", False),), view.decision.repetitions, counts, observed,
        failure_reason=str(error), failure_kind="not_available" if error.code.value == "insufficient_calibration_groups"
            else "fit_failure") for spec in spec_records}


def _load_inner_roster(directory, record, loaded, authority, *, calibration_groups=None):
    producers = record["inner_producers"]
    keys = [row["fit_key"] for row in producers]
    if len(set(keys)) != len(keys):
        raise ValueError("prefix inner producer roster repeats a fit key")
    groups = {row["view"]["record_id"]: row for row in authority["temperature_group_records"]}
    actual_groups = {row["record_id"]: row for row in record["prefix_input"]["complete_groups"]}
    models = {}
    epoch_inputs = {}
    captured_rows = []
    for producer in producers:
        if producer["status"] == "unavailable":
            if (producer["package"] is not None or not producer["failure"] or not producer["uses"]
                    or producer["oof_rows"] or producer["model_identity"] is not None
                    or set(producer["training_record_ids"]) - set(groups)
                    or any(use["training_record_ids"] != producer["training_record_ids"]
                        or (use.get("pool_held_record_id") or use.get("temperature_held_record_id")) in producer["training_record_ids"] for use in producer["uses"])):
                raise ValueError("prefix unavailable inner producer has an invalid scope or failure receipt")
            continue
        if producer["status"] != "completed":
            raise ValueError("prefix inner producer requires explicit completed/unavailable status")
        path = _inner_path(directory, producer)
        if not path.is_dir():
            raise ValueError("prefix inner producer package changed")
        ensemble_content = (path / "ensemble.json").read_bytes()
        if hashlib.sha256(ensemble_content).hexdigest() != producer["package_sha256"]:
            raise ValueError("prefix inner producer package changed")
        model = FittedEnsemble._load_content(path, ensemble_content)
        models[producer["fit_key"]] = model
        if semantic_sha256(_final_identity(model)) != semantic_sha256(producer["model_identity"]):
            raise ValueError("prefix inner producer numerical identity changed")
        check_native_fit_inputs(producer["native_fit_inputs"], model)
        for expert in model._experts:
            if calibration_groups is not None and expert.program.family == "continuous_ar":
                from transfer.continuous_decision import CalibrationRecord

                training = [calibration_groups[name] for name in producer["training_record_ids"]]
                raw_records = [CalibrationRecord(group.view.record_id, group.view.samples,
                                                group.view.onsets, group.labels) for group in training]
                fresh = actual_expert_fit_inputs(expert.program, None, raw_records, training[0].view.record())
                if semantic_sha256(fresh) != semantic_sha256(producer["native_fit_inputs"][expert.program.expert_id]):
                    raise ValueError("prefix inner native raw operands differ from original C group owner")
            if expert.program.family == "trial" and calibration_groups is not None:
                from transfer.ensemble_views import build_calibration_epoch_dataset

                key = (tuple(producer["training_record_ids"]), expert.program.profile.digest)
                if key not in epoch_inputs:
                    data = build_calibration_epoch_dataset(
                        [calibration_groups[name] for name in producer["training_record_ids"]], expert.program.profile)
                    epoch_inputs[key] = data
                fresh = actual_expert_fit_inputs(expert.program, epoch_inputs[key], [],
                                                calibration_groups[producer["training_record_ids"][0]].view.record())
                if semantic_sha256(fresh) != semantic_sha256(producer["native_fit_inputs"][expert.program.expert_id]):
                    raise ValueError("prefix inner trial native input differs from original C group transformation")
            if expert.program.family != "continuous_ar":
                continue
            captured_native = producer["native_fit_inputs"][expert.program.expert_id]["records"]
            if len(captured_native) != len(producer["training_record_ids"]):
                raise ValueError("prefix inner native fitter lost original training records")
            for name, native in zip(producer["training_record_ids"], captured_native, strict=True):
                original = actual_groups[name]
                if (native["record_id"] != name
                        or native["original_input_sha256"] != original["native_raw_fit_input_digest"]
                        or native["n_events"] != original["group"]["authorized_label_count"]
                        or native["n_samples"] != original["raw_scope"][1] - original["raw_scope"][0]):
                    raise ValueError("prefix inner actual native inputs differ from authorized C group")
        training_ids = [row["record_id"] for row in model.fit_receipts]
        if (training_ids != producer["training_record_ids"] or set(training_ids) - set(groups)
                or semantic_sha256(model.fit_receipts) != semantic_sha256([groups[name]["view"] for name in training_ids])):
            raise ValueError("prefix inner producer actual training scope changed")
        failures = producer.get("oof_score_failures", [])
        if not producer["uses"] or not (producer["oof_rows"] or failures):
            raise ValueError("prefix inner producer is missing actual OOF uses or score rows")
        for use in producer["uses"]:
            held_id = use.get("pool_held_record_id") or use.get("temperature_held_record_id")
            if (held_id in training_ids
                    or held_id not in groups
                    or use["training_record_ids"] != training_ids
                    or semantic_sha256(use["program"]) != model.scorer.program.digest()
                    or semantic_sha256(use["expert_programs"]) != semantic_sha256(
                        [expert.program.record() for expert in model._experts])):
                raise ValueError("prefix OOF use differs from actual held/training programme")
        for failure in failures:
            if (failure["held_record_id"] in training_ids or failure["held_record_id"] not in groups
                    or failure["training_record_ids"] != training_ids or not failure["failure"]
                    or not any((use.get("pool_held_record_id") or use.get("temperature_held_record_id")) == failure["held_record_id"] for use in producer["uses"])):
                raise ValueError("prefix actual OOF raised-score failure scope changed")
        for row in producer["oof_rows"]:
            held_id = row["held_record_id"]
            if (held_id in training_ids or held_id not in groups or row["training_record_ids"] != training_ids
                    or not any((use.get("pool_held_record_id") or use.get("temperature_held_record_id")) == held_id for use in producer["uses"])):
                raise ValueError("prefix OOF held row or training authority changed")
            evidence = row["evidence"]
            if set(evidence) != set(model.scorer.program.expert_ids):
                raise ValueError("prefix OOF evidence lost an actual expert")
            for expert in model._experts:
                fit_digest = _calibration_expert_fit_digest(expert.program,
                    [groups[name] for name in training_ids], record["source_checkpoint_digests"].get(expert.program.expert_id))
                identities = evidence[expert.program.expert_id]["expert_identities"]
                if expert.fit_digest != fit_digest or len(identities) != 1 or (
                        identities[0]["fit_digest"] != expert.fit_digest or identities[0]["model_digest"] != expert.model_digest):
                    raise ValueError("prefix inner OOF score differs from actual numerical fit binding")
            captured_rows.append((producer["fit_key"], row, model))
    actual_oof = next((model.pool_oof_record or model.oof_probability_record for model in loaded.values()
                       if model.pool_oof_record is not None or model.oof_probability_record is not None), None)
    if actual_oof is not None:
        if (actual_oof["fit_authority_sha256"] != semantic_sha256(authority)
                or semantic_sha256(actual_oof["calibration_groups"]) != semantic_sha256(authority["temperature_group_records"])
                or actual_oof["actual_K"] != record["actual_K"]
                or len(actual_oof["rows"]) != len(groups)):
            raise ValueError("prefix final OOF differs from actual prefix/complete C authority")
        if len(captured_rows) != len(actual_oof["rows"]):
            raise ValueError("prefix required OOF inner producer/score roster is incomplete")
        seen = set()
        for actual in actual_oof["rows"]:
            matches = [(key, row, model) for key, row, model in captured_rows
                if row["held_record_id"] == actual["held_record_id"]
                and row["training_record_ids"] == list(actual["training_record_ids"])]
            if len(matches) != 1:
                raise ValueError("prefix required OOF held/training producer is missing or repeated")
            key, row, model = matches[0]
            if (semantic_sha256(row["evidence"]) != semantic_sha256(actual["expert_evidence"])
                    or semantic_sha256(model.fit_receipts) != actual["training_receipts_sha256"]
                    or model.scorer.program.digest() != semantic_sha256(actual_oof["OOF_program"])
                    or semantic_sha256([expert.program.record() for expert in model._experts]) !=
                        semantic_sha256(actual_oof["OOF_expert_programs"])):
                raise ValueError("prefix captured OOF producer differs from final numerical package OOF binding")
            seen.add(key)
        if seen != set(keys):
            raise ValueError("prefix inner producer roster includes an unrequested producer")
    return captured_rows


def _manifest(directory: Path, *, expected_record_sha256=None):
    path = directory / "prefix_inner_replay.json"
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("prefix replay receipt path is a symlink")
    payload = json.loads(path.read_bytes())
    record = payload.get("record")
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("prefix replay manifest has an unsupported schema; historical /1 requires its original frozen owner")
    digest = semantic_sha256(record)
    if payload.get("record_sha256") != digest:
        raise ValueError("prefix replay manifest digest changed")
    if expected_record_sha256 is not None and digest != expected_record_sha256:
        raise ValueError("prefix replay receipt differs from the externally sealed producer receipt")
    if record.get("truth_free_future_replay") is not True or record.get("selection_performed") is not False:
        raise ValueError("prefix replay is truth-free and performs no candidate selection")
    return record


def _recursive_close(left, right, *, atol=0.0, rtol=0.0, path="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"prefix replay score fields differ at {path}")
        for key in left:
            _recursive_close(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"prefix replay score length differs at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _recursive_close(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")
        return
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                raise ValueError(f"prefix replay score differs at {path}")
        elif not np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=True):
            raise ValueError(f"prefix replay score differs at {path}")
        return
    if left != right:
        raise ValueError(f"prefix replay score differs at {path}")


def persist_prefix_inner_replay(prefix, programs, directory, *, recorder=None,
                                expert_programs=None, program=None, source_checkpoints=None,
                                context=None, device="cpu"):
    """Persist a captured prefix fit and all grouped-temperature inner models."""
    if not isinstance(prefix, CalibrationPrefix):
        raise TypeError("prefix replay requires an authorized CalibrationPrefix")
    if recorder is None or not isinstance(recorder, InnerReplayRecorder):
        raise ValueError("prefix persistence requires an opt-in recorder from the original fit run")
    declaration = getattr(recorder, "prefix_fit_declaration", None)
    if not declaration or declaration.get("prefix") is not prefix:
        raise ValueError("prefix persistence requires the same prefix fit run")
    if expert_programs is None:
        expert_programs = declaration["expert_programs"]
    if program is None:
        program = declaration["program"]
    sources = _normalise_sources(source_checkpoints if source_checkpoints is not None
                                  else declaration.get("source_checkpoints"))
    captured_context = declaration.get("context", {})
    if context is not None and semantic_sha256(context) != semantic_sha256(captured_context):
        raise ValueError("prefix persistence context differs from the original fit run")
    context = dict(captured_context)
    if (set(programs) != set(recorder.prefix_programs)
            or any(value is not recorder.prefix_programs[name] for name, value in programs.items())
            or semantic_sha256([spec.record() for spec in expert_programs]) != semantic_sha256(
                [spec.record() for spec in declaration["expert_programs"]])
            or program != declaration["program"]
            or {name: source.digest for name, source in sources.items()} != {
                name: source.digest for name, source in declaration["source_checkpoints"].items()}):
        raise ValueError("prefix persistence must use the actual captured programs/source declaration")
    if semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(declaration["input_identity"]):
        raise ValueError("prefix inputs changed after the captured original fit")
    aliases, alias_map = _check_aliases(prefix, context)
    actual_source_seeds = _check_seeds(expert_programs, sources, context)
    directory = Path(directory)
    package_manifest = directory / "temperature_programs.json"
    if not package_manifest.exists():
        FittedEnsemble.save_programs(directory, recorder.prefix_programs, participant_key=prefix.target_contract["participant_key"])
    else:
        # Existing caller package must be the actual package returned by this
        # run; load validates model bytes and the strategy roster.
        saved = FittedEnsemble.load_programs(directory)
        expected = {name: value for name, value in recorder.prefix_programs.items() if isinstance(value, FittedEnsemble)}
        if set(saved) != set(expected) or any(semantic_sha256(_final_identity(value)) != semantic_sha256(
                _final_identity(expected[name])) for name, value in saved.items()):
            raise ValueError("prefix existing numerical package differs from actual captured programs")
        ensemble = json.loads((directory / "ensemble.json").read_bytes())["record"] if saved else {}
        allowed = {"ensemble.json", "temperature_programs.json"} | {
            entry["model"] for entry in ensemble.get("experts", ()) if entry["model"] is not None}
        if {path.name for path in directory.iterdir()} - allowed:
            raise ValueError("prefix existing package has unrelated artifacts")
    (directory / "inner_models").mkdir(exist_ok=True)
    inner_records = []
    for key, model in sorted(recorder.producers.items()):
        if isinstance(model, ExpectedSubjectError):
            uses = recorder.uses.get(key, [])
            inner_records.append({"fit_key": key, "status": "unavailable", "package": None,
                "package_sha256": None, "model_identity": None, "uses": uses,
                "training_record_ids": uses[0]["training_record_ids"], "oof_rows": [],
                "failure": model.record(subject=prefix.target_contract["participant_key"])})
            continue
        path = directory / "inner_models" / key
        if path.exists():
            raise ValueError("duplicate prefix inner producer package")
        model.save(path)
        uses = recorder.uses.get(key, [])
        inner_records.append({
            "fit_key": key,
            "status": "completed",
            "failure": None,
            "package": str(path.relative_to(directory)).replace("\\", "/"),
            "package_sha256": _sha256(path / "ensemble.json"),
            "model_identity": _final_identity(model),
            "training_record_ids": [row["record_id"] for row in model.fit_receipts],
            "uses": uses,
            "oof_rows": recorder.oof_rows.get(key, []),
            "oof_score_failures": recorder.oof_score_failures.get(key, []),
            "temperature_record": model.temperature_record,
            "native_fit_inputs": recorder.native_fit_inputs[id(model)],
        })
    package_hashes = {
        "ensemble_json_sha256": None if not (directory / "ensemble.json").exists() else _sha256(directory / "ensemble.json"),
        "temperature_programs_json_sha256": _sha256(package_manifest),
    }
    declarations = {name: _fit_record(value, participant_key=prefix.target_contract["participant_key"])
                    for name, value in recorder.prefix_programs.items()}
    record = {
        "schema": SCHEMA,
        "evidence_scope": "calibration_development",
        "participant_key": prefix.target_contract["participant_key"],
        "actual_K": prefix.budget,
        "requested_Ks": aliases,
        "requested_to_actual_K": alias_map,
        "source_seed": context.get("source_seed"),
        "target_seed": context.get("target_seed"),
        "device": str(device),
        "prefix_input": prefix_input_identity(prefix),
        "native_fit_inputs": declaration["native_fit_inputs"],
        "prefix_record_sha256": semantic_sha256(prefix.record()),
        "expert_programs": [spec.record() for spec in expert_programs],
        "ensemble_program": program.record(),
        "source_checkpoint_roles": _source_roles(expert_programs, sources),
        "source_checkpoint_digests": {name: source.digest for name, source in sources.items()},
        "actual_source_seeds": actual_source_seeds,
        "programs": declarations,
        "inner_producers": inner_records,
        "package_hashes": package_hashes,
        "requested_denominator": {
            "requested_Ks": aliases, "actual_K": prefix.budget,
            "source_seed": context.get("source_seed"), "target_seed": context.get("target_seed"),
        },
        "fit_context": context,
        "future_score_rows": recorder.prefix_future_scores,
        "selection_performed": False,
        "truth_free_future_replay": True,
        "public_scope": "software_bounded_calibration_development",
    }
    atomic_write_json(directory / "prefix_inner_replay.json", {
        "record": record, "record_sha256": semantic_sha256(record),
    })
    recorder.prefix_receipt_sha256 = semantic_sha256(record)
    return directory


class _PrefixPackageAdmission:
    """Private numerical state belonging to one actual synchronous ingress.

    The owner never crosses a caller invocation. Its record and predictors are
    not supplied by a caller, and a fresh ingress always rereads saved bytes.
    """

    def __init__(self, directory, *, expected_record_sha256=None):
        self.directory = Path(directory)
        self.record = _manifest(self.directory, expected_record_sha256=expected_record_sha256)
        if ((self.directory / "ensemble.json").is_symlink()
                or (self.directory / "temperature_programs.json").is_symlink()):
            raise ValueError("prefix numerical package manifest path is a symlink")
        if (self.record["package_hashes"]["ensemble_json_sha256"] is not None
                and not (self.directory / "ensemble.json").exists()):
            raise ValueError("prefix ensemble package differs from captured numerical package")
        temperature, self.programs = FittedEnsemble._admit_programs(
            self.directory, package_hashes=self.record["package_hashes"])
        self._admit_declarations(temperature["programs"])

    def _admit_declarations(self, actual_programs):
        record, loaded = self.record, self.programs
        _check_program_declarations(record, loaded, actual_programs)
        self.authority = _check_final_authority(record, loaded)
        alias_owner = type("PrefixBudget", (), {"budget": record["actual_K"]})()
        aliases, alias_map = _check_aliases(alias_owner, record["fit_context"])
        if aliases != record["requested_Ks"] or alias_map != record["requested_to_actual_K"]:
            raise ValueError("prefix requested K aliases differ from the captured caller fit context")

    def replay(self, *, prefix=None, future_views=None, source_checkpoints=None,
               context=None, device=None, atol=0.0, rtol=0.0):
        return _replay_admitted_prefix(self, prefix=prefix, future_views=future_views,
            source_checkpoints=source_checkpoints, context=context, device=device, atol=atol, rtol=rtol)


def _check_program_declarations(record, loaded, actual_programs):
    if set(actual_programs) != set(record["programs"]):
        raise ValueError("prefix strategy/failure roster changed")
    for name, value in record["programs"].items():
        if value["status"] == "unavailable":
            if actual_programs[name] != value:
                raise ValueError("prefix unavailable program failure changed")
        elif (value["status"] != "completed" or name not in loaded
                or value["strategy"] != name
                or actual_programs[name]["status"] != "completed"
                or semantic_sha256(actual_programs[name]["temperature_calibration"]) !=
                    semantic_sha256(value["temperature_calibration"])
                or semantic_sha256(actual_programs[name].get("pool_calibration")) !=
                    semantic_sha256(value.get("pool_calibration"))
                or semantic_sha256(_final_identity(loaded[name])) != semantic_sha256(value["model_identity"])):
            raise ValueError("prefix completed program numerical or temperature identity changed")
    if (record["source_seed"] != record["fit_context"].get("source_seed")
            or record["target_seed"] != record["fit_context"].get("target_seed")
            or record["requested_Ks"] != record["requested_denominator"]["requested_Ks"]
            or record["actual_K"] != record["requested_denominator"]["actual_K"]
            or record["source_seed"] != record["requested_denominator"]["source_seed"]
            or record["target_seed"] != record["requested_denominator"]["target_seed"]):
        raise ValueError("prefix requested denominator or seed pair differs from fit context")


def _replay_admitted_prefix(admission, *, prefix=None, future_views=None,
                            source_checkpoints=None, context=None, device=None, atol=0.0, rtol=0.0):
    directory, record, loaded = admission.directory, admission.record, admission.programs
    authority = admission.authority
    future_views = None if future_views is None else tuple(future_views)
    if context is not None and semantic_sha256(context) != semantic_sha256(record["fit_context"]):
        raise ValueError("prefix source/target seed pair or requested fit context changed")
    if device is not None and str(device) != record["device"]:
        raise ValueError("prefix replay device differs from captured fit")
    if prefix is not None:
        actual = prefix_input_identity(prefix)
        if semantic_sha256(actual) != semantic_sha256(record["prefix_input"]):
            raise ValueError("prefix replay raw/sample/label/scope identity changed")
        if prefix.budget != record["actual_K"]:
            raise ValueError("prefix replay actual K changed")
        sources = _normalise_sources(source_checkpoints)
        specs = tuple(CalibrationExpertProgram.from_record(value) for value in record["expert_programs"])
        if semantic_sha256(prefix_native_fit_inputs(prefix, specs)) != semantic_sha256(record["native_fit_inputs"]):
            raise ValueError("prefix native fitter operands differ from freshly compiled original input owner")
        if _source_roles(specs, sources) != record["source_checkpoint_roles"]:
            raise ValueError("prefix replay source checkpoint role or digest changed")
        if _check_seeds(specs, sources, record["fit_context"]) != record["actual_source_seeds"]:
            raise ValueError("prefix replay actual source checkpoint seed changed")
        groups = {group.view.record_id: group for group in prefix.complete_groups}
    else:
        groups = {}
    exact_checks = 0
    for _, row, model in _load_inner_roster(directory, record, loaded, authority,
                                          calibration_groups=groups if prefix is not None else None):
        if prefix is not None:
            held = groups[row["held_record_id"]]
            score = {name: item.record() for name, item in model.score_experts(held.view).items()}
            _recursive_close(score, row["evidence"], atol=atol, rtol=rtol)
            exact_checks += 1
    captured = {(row["strategy"], row["input"]["view"]["record_id"]): row
                for row in record["future_score_rows"]}
    if len(captured) != len(record["future_score_rows"]):
        raise ValueError("prefix future score roster repeats a requested row")
    requested_ids = record["fit_context"].get("requested_decision_ids")
    if requested_ids is not None and (len(set(requested_ids)) != len(requested_ids)
            or set(captured) != {(name, record_id) for name in record["programs"] for record_id in requested_ids}):
        raise ValueError("prefix future score roster is missing a requested strategy or decision")
    scores = []
    if future_views is not None and requested_ids is not None and [view.record_id for view in future_views] != requested_ids:
        raise ValueError("prefix future queue differs from requested denominator")
    # This group is loaded independently from the original captured fit group.
    # Share only its genuine all-C expert tuple within this current-view call;
    # C-inner single-fixed checks above retain their original scoring path.
    replay_programs = {name: loaded[name] if name in loaded else ExpectedSubjectError(
        programme["failure"]["error_code"], stage=programme["failure"]["stage"], detail=programme["failure"]["detail"])
        for name, programme in record["programs"].items()}
    for view in future_views or ():
        if isinstance(view, dict) and ({"truth", "target_candidate", "labels"} & set(view)):
            raise ValueError("prefix future replay cannot accept truth or target labels")
        if any(hasattr(view, name) for name in ("truth", "target_candidate", "labels")):
            raise ValueError("prefix future replay cannot accept truth or target labels")
        identity = _validate_future(view, record)
        row = {"decision_id": getattr(view, "decision_id", None), "strategies": {}}
        shared = FittedEnsemble.score_programs(replay_programs, view)
        for name, programme in record["programs"].items():
            evidence = (shared[name] if not isinstance(shared[name], ExpectedSubjectError) else
                unavailable_prefix_expert_scores(authority, record["expert_programs"], name, programme["failure"], view))
            expected = captured.get((name, view.record_id))
            if expected is not None:
                if semantic_sha256(identity) != semantic_sha256(expected["input"]):
                    raise ValueError("prefix future sample/input identity changed")
                _recursive_close({expert: item.record() for expert, item in evidence.items()},
                                 expected["expert_evidence"], atol=atol, rtol=rtol)
                exact_checks += 1
            row["strategies"][name] = EnsembleScorer(EnsembleProgram.from_record(record["ensemble_program"]),
                programme.get("pool_calibration")).score(evidence).record()
        scores.append(row)
    return {"status": "replayed" if (prefix is not None or future_views) else "validated_package",
            "actual_K": record["actual_K"], "requested_Ks": record["requested_Ks"],
            "producer_count": len(record["inner_producers"]), "exact_score_checks": exact_checks,
            "preserved_oof_score_failures": [{"fit_key": producer["fit_key"], **failure}
                for producer in record["inner_producers"] for failure in producer.get("oof_score_failures", ())],
            "scores": scores, "truth_free": True}


def replay_prefix_inner_replay(directory, *, prefix=None, future_views=None,
                               source_checkpoints=None, context=None, expected_record_sha256=None,
                               device=None, atol=0.0, rtol=0.0):
    """Admit saved state once and replay every actual requested observation."""
    admission = _PrefixPackageAdmission(directory, expected_record_sha256=expected_record_sha256)
    return admission.replay(prefix=prefix, future_views=future_views, source_checkpoints=source_checkpoints,
                            context=context, device=device, atol=atol, rtol=rtol)


__all__ = [
    "SCHEMA", "prefix_input_identity",
    "persist_prefix_inner_replay", "replay_prefix_inner_replay",
]
