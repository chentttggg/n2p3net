"""Run frozen GPU recovery comparisons on declared held-out recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    parse_candidate_bad_channel_fractions,
    parse_candidate_quantiles,
)
from data.epochs import load_epoch_dataset  # noqa: E402
from data.personal_quality import PersonalEpochQualityPolicy  # noqa: E402
from models.candidate_evidence import candidate_evidence_scores  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.binary import plain_trunk_configuration  # noqa: E402
from transfer.calibrated import (  # noqa: E402
    CalibratedPredictor,
    _calibration_data_digest,
    _input_contract,
    _participant,
    _session_scope,
    load_calibrated_predictor,
    load_calibration_source_checkpoint,
    save_calibrated_predictor,
)
from transfer.calibration_candidates import (  # noqa: E402
    _state_digest,
    fit_calibration_candidate,
    score_calibration_candidate,
)
from transfer.checkpoint import checkpoint_training_contract  # noqa: E402
from transfer.quality import load_epoch_quality, validate_frozen_epoch_quality  # noqa: E402
from transfer.recipes import TargetRecipe, target_batch_budget  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_evaluation_scope(calibration, evaluation, *, mode="prospective_recording"):
    if mode not in {"prospective_recording", "independent_recording_holdout"}:
        raise ValueError("unknown recording evaluation mode")
    training = _session_scope(calibration, calibration=True)
    testing = _session_scope(evaluation, calibration=False)
    training_ids = {row["session_id"] for row in training}
    if training_ids & {row["session_id"] for row in testing}:
        raise ValueError("calibration and evaluation sessions must be disjoint")
    if set(calibration.event_timeline.event_ids) & set(evaluation.event_timeline.event_ids):
        raise ValueError("calibration and evaluation events must be disjoint")
    if _participant(calibration) != _participant(evaluation):
        raise ValueError("personal recovery evaluation must preserve participant authority")
    if _input_contract(calibration) != _input_contract(evaluation):
        raise ValueError("calibration and evaluation input contracts differ")
    latest = max(row["calibration_available_timestamp_s"] for row in training)
    later = all(row["session_start_timestamp_s"] > latest for row in testing)
    if mode == "prospective_recording" and not later:
        raise ValueError("evaluation precedes completed calibration")
    return {"mode": mode, "disjoint_sessions": True, "disjoint_events": True,
            "evaluation_after_calibration_ready": later,
            "calibration_sessions": sorted(training_ids),
            "evaluation_sessions": [row["session_id"] for row in testing]}


def quality_policy_from_plan(record):
    if record is None:
        return None
    fields = dict(record)
    method = fields.pop("method", "fold_local_ptp_cv")
    if method == "personal_epoch_signal_evidence":
        policy = PersonalEpochQualityPolicy(**fields)
        policy.validate()
        return policy
    if method != "fold_local_ptp_cv":
        raise ValueError("unsupported recovery epoch quality policy method")
    if "candidate_quantiles" in fields:
        fields["candidate_quantiles"] = parse_candidate_quantiles(fields["candidate_quantiles"])
    if "candidate_bad_channel_fractions" in fields:
        fields["candidate_bad_channel_fractions"] = parse_candidate_bad_channel_fractions(fields["candidate_bad_channel_fractions"])
    policy = FoldLocalArtifactPolicy(**fields)
    policy.validate()
    return policy


def observed(dataset):
    result = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
    if dataset.trial_channel_mask is not None:
        result &= dataset.trial_channel_mask
    return result


def dense_prediction(dataset, logits):
    timeline = dataset.event_timeline
    values = np.full(timeline.n_events, np.nan)
    indices = np.asarray(timeline.evidence_indices)
    available = indices >= 0
    values[available] = logits[indices[available]]
    return values


def binary_metrics(labels, logits, mask=None):
    usable = np.isfinite(logits)
    if mask is not None:
        usable &= mask
    y, score = labels[usable], logits[usable]
    both = set(np.unique(y)) == {0, 1}
    return {"rows": int(usable.sum()), "positives": int(y.sum()),
            "auc": float(roc_auc_score(y, score)) if both else None,
            "average_precision": float(average_precision_score(y, score)) if both else None,
            "bacc_zero_logit": float(balanced_accuracy_score(y, score >= 0)) if both else None}


def complete_cycle_diagnostic(dataset, logits, *, minimum_cycles):
    """Availability-only balanced cycle selection; not the fixed R5 endpoint."""
    values = dense_prediction(dataset, logits)
    timeline = dataset.event_timeline
    codes = np.asarray(timeline.candidate_ids, dtype=int)
    targets = np.asarray(timeline.target_candidate_ids, dtype=int)
    if len(values) % 45:
        raise ValueError("diagnostic requires complete 45-event scheduled blocks")
    vocabulary = tuple(range(1, 10))
    records = []
    for start in range(0, len(values), 45):
        selected, cycles = [], []
        for cycle in range(5):
            rows = np.arange(start + cycle*9, start + (cycle+1)*9)
            if sorted(codes[rows].tolist()) != list(vocabulary):
                raise ValueError("cycle diagnostic requires declared balanced permutation cycles")
            if np.isfinite(values[rows]).all():
                selected.extend(rows.tolist())
                cycles.append(cycle)
        evidence = candidate_evidence_scores(values[selected], codes[selected], vocabulary,
            aggregation="mean", missing_candidate_policy="abstain")
        prediction = evidence.predicted if len(cycles) >= minimum_cycles else None
        truth = np.unique(targets[start:start+45])
        if len(truth) != 1 or truth[0] not in vocabulary:
            raise ValueError("cycle diagnostic requires one confirmed block target")
        records.append({"block_index": start // 45, "complete_cycle_indices": cycles,
            "usable_cycles": len(cycles), "used_scheduled_event_indices": selected,
            "prediction": prediction, "hit": prediction == int(truth[0]),
            "reason": "insufficient_complete_cycles" if len(cycles) < minimum_cycles else
                      "tie" if prediction is None else None})
    return {"scope": "development_diagnostic_not_fixed_R5_or_product_policy",
            "minimum_cycles": minimum_cycles, "requested_blocks": len(records),
            "predicted_blocks": sum(row["prediction"] is not None for row in records),
            "hit_including_abstentions": sum(row["hit"] for row in records)/len(records),
            "blocks": records, "cycle_selection_uses_labels_or_scores": False}


def candidate_artifact(dataset, fitted, payload, gate, *, device, plan_sha256):
    subject, participant = _participant(dataset)
    sessions = _session_scope(dataset, calibration=True)
    continuous = dataset.provenance.get("continuous_artifact_processing", {})
    continuous_model = continuous.get("model") if continuous.get("status") == "applied" else None
    reference = dataset.provenance.get("domain_adapter", {}).get("reference", {})
    anchor = (continuous_model["spec"].get("reference_channel") if continuous_model else
              reference.get("channel") if reference.get("mode") == "measured_channel" else None)
    ocular = dataset.provenance.get("ocular_artifact_processing", {})
    ocular_model = ocular.get("model") if ocular.get("status") == "applied" else None
    if ocular_model is not None:
        anchor = ocular_model.get("reference_channel")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(), "participant_key": participant,
        "subject_id": subject, "calibration_sessions": sessions,
        "calibration_available_timestamp_s": max(row["calibration_available_timestamp_s"] for row in sessions),
        "calibration_trials": dataset.n_epochs, "calibration_positives": int(dataset.y.sum()),
        "calibration_data_sha256": _calibration_data_digest(dataset),
        "input_contract": _input_contract(dataset), "candidate_vocabulary": list(range(1, 10)),
        "input_request": {"reference_channel": anchor, "ocular_model": ocular_model,
            "continuous_qc_model": continuous_model, "channel_names": list(dataset.channel_names)},
        "evidence_scope": "fixed_recipe_recovery_development_not_product_selection_or_confirmation",
        "model_kind": "source_initialized_n2p3" if fitted.recipe.initialization == "source" else
                      "scratch_n2p3" if fitted.adapter is not None else "target_only_shrinkage_lda",
        "source_pretraining_used": fitted.recipe.initialization == "source",
        "recipe": fitted.recipe.record(), "seed": fitted.recipe.seed,
        "selection": {"mode": "predeclared_fixed_recipe_no_selection", "plan_sha256": plan_sha256},
        "actual_fitted_state_sha256": _state_digest(fitted.state),
        "epoch_quality": fitted.fit_history.get("epoch_quality"), "fit_history": fitted.fit_history,
    }
    if fitted.adapter is not None:
        trunk = fitted.adapter.trunk
        metadata.update(architecture=trunk.architecture_record(),
            architecture_configuration=asdict(plain_trunk_configuration(trunk)), pooling_mode=trunk.pooling_mode)
    if fitted.recipe.initialization == "source":
        metadata.update(source_checkpoint_sha256=gate["checkpoint_sha256"],
                        source_training_contract=checkpoint_training_contract(payload).record())
    return CalibratedPredictor(metadata, fitted.state, device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "n2p3_brainsync_recovery_experiment/1" or plan["device"] != "cuda":
        raise ValueError("frozen recovery experiment plan with CUDA required")
    evaluation_mode = plan.get("evaluation_mode", "prospective_recording")
    if evaluation_mode not in {"prospective_recording", "independent_recording_holdout"}:
        raise ValueError("unknown recording evaluation mode")
    policies = {arm["name"]: quality_policy_from_plan(arm["epoch_quality_policy"]) for arm in plan["pipelines"]}
    frozen_quality = {arm["name"]: load_epoch_quality(arm["epoch_quality_model"])
        if arm.get("epoch_quality_model") else None for arm in plan["pipelines"]}
    if any(policies[name] is not None and fitted is not None for name, fitted in frozen_quality.items()):
        raise ValueError("choose a frozen epoch quality model or a fitted quality policy")
    verified_executed_snapshot(args.snapshot, root=ROOT)
    required = ({plan["checkpoint"]} if plan["checkpoint"] is not None else set()) | {path for arm in plan["pipelines"]
        for path in [arm["calibration_cache"], *arm["evaluation_caches"]]}
    required |= {arm["epoch_quality_model"] for arm in plan["pipelines"] if arm.get("epoch_quality_model")}
    if not required <= set(plan["input_sha256"]):
        raise ValueError("every executed cache and checkpoint must have a frozen hash")
    for path, expected in plan["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU experiment cannot fall back to CPU")
    if plan.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with cpu_thread_budget(1):
        datasets = {path: load_epoch_dataset(path, require_labels=True, validation="attested")
                    for arm in plan["pipelines"]
                    for path in [arm["calibration_cache"], *arm["evaluation_caches"]]}
        evaluation_scopes = []
        for arm in plan["pipelines"]:
            calibration = datasets[arm["calibration_cache"]]
            training_scope = _session_scope(calibration, calibration=True)
            expected_sessions = arm.get("calibration_sessions", [plan["calibration_session"]])
            if [row["session_id"] for row in training_scope] != expected_sessions:
                raise ValueError("calibration sessions differ from the declared pipeline budget")
            if frozen_quality[arm["name"]] is not None:
                validate_frozen_epoch_quality(calibration, frozen_quality[arm["name"]],
                    np.ones(calibration.n_epochs, dtype=bool))
            for path in arm["evaluation_caches"]:
                evaluation = datasets[path]
                scope = _session_scope(evaluation, calibration=False)
                if len(scope) != 1 or scope[0]["session_id"] not in plan["evaluation_sessions"]:
                    raise ValueError("evaluation session is outside the frozen plan")
                evaluation_scopes.append(validate_evaluation_scope(
                    calibration, evaluation, mode=evaluation_mode))
        preflight = {"plan_sha256": sha(args.plan), "source_manifest_sha256": sha(args.snapshot),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "worker_count": 1, "native_threads": 1, "cpu_fallback": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cache_rows": {path: dataset.n_epochs for path, dataset in datasets.items()},
            "data_source": "already_observed_P001_development_sessions", "plan": plan,
            "evaluation_scopes": evaluation_scopes}
        atomic_write_json(args.output / "preflight.json", preflight)
        if args.dry_run:
            print(json.dumps({"status": "preflight_passed", "fits": len(plan["recipes"])*len(plan["pipelines"])}))
            return
        completed, evaluations, prepared_sources = [], [], {}
        for arm in plan["pipelines"]:
            calibration = datasets[arm["calibration_cache"]]
            if arm["calibration_cache"] not in prepared_sources:
                prepared_sources[arm["calibration_cache"]] = load_calibration_source_checkpoint(
                    calibration, plan["checkpoint"])
            source, payload, stats, gate = prepared_sources[arm["calibration_cache"]]
            requires_source = any(item["initialization"] == "source" for item in plan["recipes"])
            if ((requires_source or plan["checkpoint"] is not None) and gate["status"] != "compatible"):
                raise ValueError(gate)
            quality_policy = policies[arm["name"]]
            for declaration in plan["recipes"]:
                recipe = TargetRecipe(**declaration)
                key = arm["name"] + "__" + recipe.name
                directory = args.output / key
                directory.mkdir()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                fitted = fit_calibration_candidate(calibration, recipe,
                    np.ones(calibration.n_epochs, dtype=bool), source_trunk=source,
                    source_input_stats=stats, device="cuda", epoch_quality_policy=quality_policy,
                    epoch_quality_model=frozen_quality[arm["name"]])
                torch.cuda.synchronize()
                duration = time.perf_counter() - started
                predictor = candidate_artifact(calibration, fitted, payload, gate,
                    device="cuda", plan_sha256=preflight["plan_sha256"])
                predictor.metadata["evaluation_mode"] = evaluation_mode
                artifact = save_calibrated_predictor(directory / "model.pt", predictor)
                restored = load_calibrated_predictor(artifact, device="cuda")
                n_fit = int(fitted.fit_history["fit_scope"]["training_rows"])
                record = {"id": key, "pipeline": arm["name"], "recipe": recipe.record(),
                    "fit_seconds": duration, "fit_history": fitted.fit_history, "actual_n_fit": n_fit,
                    "expected_budget": None if recipe.initialization == "waveform_lda" else target_batch_budget(
                        n_fit, recipe.batch_size, epochs=recipe.epochs, max_updates=recipe.max_updates),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "artifact": str(artifact),
                    "artifact_sha256": sha(artifact), "state_sha256": _state_digest(fitted.state)}
                if fitted.adapter is not None:
                    for field in ("updates", "draws"):
                        if fitted.fit_history["fit_budget"][field] != record["expected_budget"][field]:
                            raise ValueError(f"actual {field} differs from the frozen training budget")
                atomic_write_json(directory / "fit.json", record)
                completed.append(record)
                for path in arm["evaluation_caches"]:
                    dataset = datasets[path]
                    if evaluation_mode == "prospective_recording":
                        restored.validate_inference_scope(dataset)
                    else:
                        validate_evaluation_scope(calibration, dataset, mode=evaluation_mode)
                    session = _session_scope(dataset, calibration=False)[0]["session_id"]
                    logits = restored.predict_logit(dataset)
                    original = predictor.predict_logit(dataset)
                    np.testing.assert_array_equal(logits, original)
                    candidate_logits = fitted.predict_logit(dataset.X, observed(dataset))
                    np.testing.assert_allclose(logits, candidate_logits, rtol=1e-4, atol=1e-4, equal_nan=True)
                    values = dense_prediction(dataset, logits)
                    timeline = dataset.event_timeline
                    codes = np.asarray(timeline.candidate_ids, dtype=int)
                    labels = (codes == np.asarray(timeline.target_candidate_ids, dtype=int)).astype(int)
                    np.savez_compressed(directory / f"{session}.npz", logits=values, y=labels,
                        event_ids=np.asarray(timeline.event_ids, dtype=str), candidate_ids=codes)
                    result = {"id": key, "pipeline": arm["name"], "initialization": recipe.initialization,
                        "seed": recipe.seed, "session_id": session, "scheduled_events": len(values),
                        "metrics": binary_metrics(labels, values), "reload_bitwise_equal": True,
                        "candidate_artifact_agree": True,
                        "strict_R5": score_calibration_candidate(dataset, logits,
                            np.ones(dataset.n_epochs, dtype=bool), protocol="brainsync_operational_r5"),
                        "balanced_cycle_diagnostic": complete_cycle_diagnostic(dataset, logits,
                            minimum_cycles=plan["diagnostic_minimum_complete_cycles"])}
                    atomic_write_json(directory / f"{session}.json", result)
                    evaluations.append(result)
                if record["state_sha256"] != _state_digest(fitted.state):
                    raise ValueError("read-only prediction changed fitted model state")
                atomic_write_json(args.output / "progress.json", {"completed_fits": len(completed), "last": key})
                print(json.dumps({"id": key, "n_fit": n_fit, "seconds": round(duration, 3)}), flush=True)
        common = []
        for session in plan["evaluation_sessions"]:
            for comparison, selected in (("neural_methods", [r for r in evaluations if r["session_id"] == session
                                                             and r["initialization"] != "waveform_lda"]),
                                         ("all_methods", [r for r in evaluations if r["session_id"] == session])):
                arrays = [np.load(args.output / row["id"] / f"{session}.npz", allow_pickle=False) for row in selected]
                for item in arrays[1:]:
                    np.testing.assert_array_equal(item["event_ids"], arrays[0]["event_ids"])
                    np.testing.assert_array_equal(item["y"], arrays[0]["y"])
                mask = np.logical_and.reduce([np.isfinite(item["logits"]) for item in arrays])
                common.append({"session_id": session, "comparison": comparison,
                    "event_ids": arrays[0]["event_ids"][mask].tolist(), "rows": int(mask.sum()),
                    "results": [{"id": row["id"], **binary_metrics(item["y"], item["logits"], mask)}
                                for row, item in zip(selected, arrays, strict=True)]})
                for item in arrays:
                    item.close()
        atomic_write_json(args.output / "completed.json", {"fits": completed, "evaluations": evaluations,
            "common_evaluation_support": common, "source_manifest_sha256": preflight["source_manifest_sha256"],
            "plan_sha256": preflight["plan_sha256"], "model_promoted": False,
            "source_checkpoint_loads": 0 if plan["checkpoint"] is None else len(prepared_sources),
            "evaluation_scopes": evaluation_scopes,
            "different_training_support_is_not_pure_preprocessing_causal_effect": True})
        verified_executed_snapshot(args.snapshot, root=ROOT)
        print(json.dumps({"status": "completed", "fits": len(completed), "evaluations": len(evaluations)}))


if __name__ == "__main__":
    main()
