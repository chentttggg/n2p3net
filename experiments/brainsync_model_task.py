"""Calibrate a saved participant model or infer a completed unlabelled BrainSync block."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from data.brainsync import (  # noqa: E402
    BRAIN_SYNC_PREPROCESSING,
    BrainSyncNoUsableEvidenceError,
    apply_brainsync_reference,
    load_brainsync_sessions,
    resolve_brainsync_reference,
)
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.candidate_task import MembershipDecision  # noqa: E402
from data.continuous_qc import load_continuous_qc  # noqa: E402
from data.epochs import load_preprocessing_spec  # noqa: E402
from data.ocular import (  # noqa: E402
    EOG_REFERENCE_POLICIES,
    SHARED_MEASURED_REFERENCE,
    load_eog_regression,
)
from data.personal_quality import PersonalEpochQualityPolicy  # noqa: E402
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions  # noqa: E402
from models.candidate_evidence import (  # noqa: E402
    candidate_evidence_scores,
    membership_decision_scores,
)
from research.execution import atomic_write_json  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.calibrated import (  # noqa: E402
    SELECTOR_MODES,
    fit_calibrated_predictor,
    load_calibrated_predictor,
    preflight_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.quality import load_epoch_quality, save_epoch_quality  # noqa: E402
from transfer.recipes import TargetRecipe  # noqa: E402
from transfer.workstation_profiles import load_old_basic_profile, validate_executed_profile_source  # noqa: E402

TASK_RESULT_SCHEMA = "n2p3_brainsync_model_task/2"


def calibration_preprocessing(path):
    if path is None:
        return BRAIN_SYNC_PREPROCESSING
    return load_preprocessing_spec(path)


def _progress(value: int, message: str) -> None:
    print(f"PROGRESS {value}", flush=True)
    print(message, flush=True)


def model_task_device(requested: str) -> str:
    device = get_device() if requested == "auto" else torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable in the backend environment")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device index is unavailable")
    return str(device)


def _model_summary(predictor) -> dict[str, object]:
    predictor.validate_personal_workflow_metadata()
    metadata = predictor.metadata
    ocular = metadata["input_request"].get("ocular_model")
    quality = metadata.get("epoch_quality")
    fit_scope = metadata["selection"].get("fit_scope", metadata.get("fit_history", {}).get("fit_scope", {}))
    return {
        "model_kind": metadata["model_kind"], "method": metadata["model_kind"],
        "participant_key": metadata["participant_key"], "subject_id": metadata["subject_id"],
        "calibration_trials": metadata["calibration_trials"],
        "actual_training_trials": fit_scope.get("training_rows", metadata["calibration_trials"]),
        "target_trials": metadata["calibration_positives"],
        "calibration_sessions": metadata["calibration_sessions"],
        "calibration_available_timestamp_s": metadata["calibration_available_timestamp_s"],
        "selected_recipe": metadata["selection"].get("selected"),
        "internal_validation": metadata["selection"],
        "source_pretraining_used": metadata["source_pretraining_used"],
        "calibration_profile": metadata.get("calibration_profile"),
        "ocular_correction_applied": ocular is not None,
        "epoch_qc_applied": quality is not None,
        "epoch_qc_method": None if quality is None else quality["model"]["method"],
        "epoch_qc_schema": None if quality is None else quality["model"].get("schema"),
        "epoch_qc_model_digest": None if quality is None else quality["model_digest"],
        "epoch_qc_fit_sessions": [] if quality is None else quality["model"]["fit_provenance"]["session_ids"],
        "epoch_qc_lifecycle": None if quality is None else quality.get("lifecycle", "fit_inside_calibration"),
        "ocular_fit_sessions": [] if ocular is None else ocular["fit_provenance"]["calibration_session_ids"],
        "ocular_fit_policy": None if ocular is None else (
            'authorized_calibration_sessions' if set(ocular['fit_provenance']['calibration_session_ids']) ==
            {row['session_id'] for row in metadata['calibration_sessions']} else 'frozen_subset_of_authorized_calibration_sessions'),
        "reference": metadata["input_contract"]["source_reference"],
        "acquisition_references": metadata["input_request"].get("acquisition_references", []),
        "input_contract": metadata["input_contract"],
        "simulation_status": metadata["input_contract"]["simulation_status"],
    }


def _export_epoch_quality(predictor, output_dir: Path) -> str | None:
    if predictor.epoch_quality is None:
        return None
    return str(save_epoch_quality(output_dir / "epoch_quality.json", predictor.epoch_quality).resolve())


def inference_trial_counts(*, scheduled_trials: int, input_epochs: int, usable_predictions: int) -> dict[str, int]:
    if not 0 <= usable_predictions <= input_epochs <= scheduled_trials:
        raise ValueError("inference admission counts do not reconcile with the scheduled ledger")
    return {"scheduled_trials": scheduled_trials, "input_epoch_trials": input_epochs,
            "evidence_trials": usable_predictions,
            "preprocessing_unavailable_trials": scheduled_trials - input_epochs,
            "model_unavailable_trials": input_epochs - usable_predictions,
            "unavailable_trials": scheduled_trials - usable_predictions}


def inference_record(predictor, dataset, *, repetitions: int | None = None, diagnostic=False) -> dict[str, object]:
    """Use all nine candidates and the scheduled ledger; target labels are never read."""
    predictor.validate_inference_scope(dataset)
    logits = predictor.predict_logit(dataset)
    timeline = dataset.event_timeline
    finite_times = np.asarray(timeline.evidence_available_times_s, dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    available_at = None if not len(finite_times) else float(finite_times.max())
    vocabulary = tuple(range(1, 10))
    try:
        codes = np.asarray(timeline.candidate_ids, dtype=int)
    except ValueError as error:
        raise ValueError("inference requires explicit digit candidate identities") from error
    indices = np.asarray(timeline.evidence_indices, dtype=int)
    available_events = indices >= 0
    available_events[available_events] &= np.isfinite(logits[indices[available_events]])
    repeat = np.asarray(timeline.repetition_indices, dtype=int)
    if not len(indices) or np.any(repeat < 0) or not np.isin(codes, vocabulary).all():
        raise ValueError("inference candidate schedule is invalid")
    requested = int(repeat.max()) + 1 if repetitions is None else repetitions
    if requested < 1:
        raise ValueError("requested repetitions must be positive")
    rows = []
    decision_score = predictor.metadata.get("decision_score")
    for level in range(1, requested + 1):
        if decision_score is not None:
            scheduled = repeat < level
            decision = MembershipDecision.from_timeline(
                timeline, scheduled, repetitions=level,
                candidate_vocabulary=tuple(str(value) for value in vocabulary),
            )
            aligned = np.zeros(int(scheduled.sum()), dtype=float)
            scoped_indices = indices[scheduled]
            observed = available_events[scheduled]
            aligned[observed] = logits[scoped_indices[observed]]
            evidence = membership_decision_scores(aligned, decision, available=observed,
                temperature=decision_score["temperature"])
            prediction = None if evidence.predicted is None else int(evidence.predicted)
            rows.append({"repetition": level, "prediction": prediction,
                "evidence_available_time_s": available_at,
                "interpretation": "retrospective_after_completed_recording",
                "status": "predicted" if prediction is not None else "abstained",
                "scores": dict(zip(evidence.candidate_vocabulary, evidence.scores.tolist(), strict=True)),
                "counts": dict(zip(evidence.candidate_vocabulary, evidence.counts.tolist(), strict=True)),
                "reason": evidence.reason, "probability_calibrated": False,
                "normalized_scores": evidence.posterior.tolist()})
            continue
        selected = (repeat < level) & available_events
        evidence = candidate_evidence_scores(
            logits[indices[selected]], codes[selected], vocabulary,
            aggregation="mean",
        )
        complete = all(evidence.counts[digit] == level for digit in vocabulary)
        prediction = evidence.predicted if complete else None
        rows.append({
            "repetition": level, "prediction": prediction,
            "evidence_available_time_s": available_at,
            "interpretation": "retrospective_after_completed_recording",
            "status": "predicted" if prediction is not None else "abstained",
            "scores": {str(digit): float(score) if np.isfinite(score) else None
                       for digit, score in evidence.scores.items()},
            "counts": {str(digit): count for digit, count in evidence.counts.items()},
            "reason": "incomplete_candidate_coverage" if not complete else "tie" if prediction is None else None,
        })
    final = rows[-1]
    protocol_compliant = (len(indices) == 45 and requested == 5 and np.array_equal(np.unique(repeat), np.arange(5))
        and all(np.count_nonzero((repeat == level) & (codes == digit)) == 1 for level in range(5) for digit in vocabulary))
    if not protocol_compliant and not diagnostic:
        final = {**final, 'prediction': None, 'status': 'abstained', 'reason': 'requires_one_complete_45_flash_R5_recording'}
    session_ids = np.unique(np.asarray(timeline.session_ids).astype(str))
    if len(session_ids) != 1:
        raise ValueError("one completed inference recording is required")
    return {
        "session_id": str(session_ids[0]), "requested_repetitions": requested,
        "prediction": final["prediction"], "status": final["status"],
        "candidate_scores": final["scores"], "by_repetition": rows, "reason": final["reason"],
        **inference_trial_counts(scheduled_trials=len(indices), input_epochs=dataset.n_epochs,
                                 usable_predictions=int(np.isfinite(logits).sum())),
        "target_labels_used": False, "score_kind": (
            "mean_binary_evidence_not_probability" if decision_score is None else decision_score["method"]),
        "mode": "completed_recording_inference",
        'protocol_compliant': bool(protocol_compliant),
        'evaluation_role': 'explicit_repetition_diagnostic' if diagnostic else 'product_R5',
        'required_product_protocol': {'scheduled_trials': 45, 'repetitions': 5, 'candidate_count': 9},
        "filter_phase": dataset.preprocessing.filter_phase,
        "online_causal": timeline.online_causal,
        "evidence_available_time_s": available_at,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("calibrate", "infer", "inspect"))
    parser.add_argument("--request-id", help="opaque caller receipt identity; never used for model selection")
    parser.add_argument("--session-dir", type=Path, action="append", default=[])
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--calibration-profile", choices=("old_basic_nn_source",),
                        help="frozen twelve-study basic source and full-calibration program")
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-channel")
    parser.add_argument("--reference-mode", choices=("auto", "native"), default="auto",
                        help="native rejects QC/EOG models that implicitly rereference the EEG")
    parser.add_argument("--ocular-model", type=Path)
    parser.add_argument("--ocular-correction", action="store_true")
    parser.add_argument("--eog-reference-policy", choices=EOG_REFERENCE_POLICIES,
                        default=SHARED_MEASURED_REFERENCE)
    parser.add_argument("--continuous-qc-model", type=Path)
    parser.add_argument("--preprocessing-config", type=Path,
                        help="complete PreprocessingSpec JSON for a declared calibration control")
    epoch_quality_group = parser.add_mutually_exclusive_group()
    epoch_quality_group.add_argument("--epoch-qc", action="store_true",
                                     help="fit personal signal-evidence QC inside calibration")
    epoch_quality_group.add_argument("--epoch-qc-model", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument('--diagnostic-repetitions', action='store_true', help='explicit retrospective repetition diagnostic; not product hit@all')
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:INDEX], xpu[:INDEX], or auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--selector-mode", choices=SELECTOR_MODES, default="legacy_product")
    parser.add_argument("--calibration-recipe", type=Path,
                        help="explicit TargetRecipe JSON for unified or fixed_erp calibration")
    parser.add_argument("--training-budget", type=Path,
                        help="one UnifiedTrainingBudget JSON with fixed personal training epochs")
    parser.add_argument("--selection-protocol", choices=("binary_auc", "brainsync_operational_r5"),
                        help="recipe scoring for other selectors; unified uses a fixed full-calibration budget")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-rule", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.calibration_profile and (
            args.task != 'calibrate' or args.selector_mode != 'legacy_product'
            or args.calibration_recipe or args.training_budget or args.preprocessing_config
            or args.seed is not None or args.batch_size is not None or args.batch_rule
            or args.selection_protocol is not None or args.reference_channel
            or args.reference_mode != 'auto' or args.ocular_correction or args.ocular_model
            or args.epoch_qc or args.epoch_qc_model or args.continuous_qc_model
            or args.eog_reference_policy != SHARED_MEASURED_REFERENCE):
        parser.error('old basic profile owns the calibration recipe, Cz reference and processing; overrides are not allowed')
    if args.task != 'calibrate' and (args.seed is not None or args.selection_protocol is not None):
        parser.error('seed and selection protocol belong to calibration')
    if args.seed is None:
        args.seed = 20260906
    if args.task != 'infer' and args.repetitions is not None:
        parser.error('repetitions belong to inference')
    if args.selector_mode == "unified":
        if args.selection_protocol is not None:
            parser.error("unified uses a fixed full-calibration budget; --selection-protocol belongs to other selectors")
        args.selection_protocol = "fixed_full_calibration"
    elif args.selector_mode == 'legacy_product':
        if args.selection_protocol not in {None, 'binary_auc'}:
            parser.error('legacy_product uses binary_auc; another selection protocol would be unused')
        args.selection_protocol = 'binary_auc'
    elif args.selection_protocol is None:
        args.selection_protocol = "brainsync_operational_r5"
    training_budget, transfer_policy = None, None
    if args.training_budget is not None:
        if args.selector_mode != "unified" or args.task != "calibrate":
            parser.error("training-budget belongs to unified calibration")
    if args.calibration_recipe is not None:
        if args.selector_mode not in {'unified', 'fixed_erp'} or args.task != 'calibrate':
            parser.error('calibration-recipe belongs to unified or fixed_erp calibration')
    elif args.selector_mode in {'unified', 'fixed_erp'} and args.task == 'calibrate':
        parser.error('unified and fixed_erp calibration require --calibration-recipe')
    if args.model_artifact.resolve() in {(args.output_dir/name).resolve() for name in ('summary.json', 'epoch_quality.json')}:
        parser.error('model artifact must differ from task summary and exported epoch quality')
    if args.task == 'inspect' and (args.session_dir or args.source_checkpoint or args.ocular_model or args.ocular_correction
            or args.reference_channel or args.reference_mode != 'auto' or args.continuous_qc_model or args.epoch_qc
            or args.epoch_qc_model or args.batch_rule or args.repetitions is not None):
        parser.error('inspect reads the frozen model only; session and preprocessing overrides are unused')
    if args.task != 'calibrate' and (args.batch_size is not None or args.selector_mode != 'legacy_product'):
        parser.error('calibration selector and batch settings do not apply to a saved model workflow')
    if args.diagnostic_repetitions and args.task != 'infer':
        parser.error('diagnostic repetitions belong to inference')
    if args.task == 'infer' and args.repetitions not in {None, 5} and not args.diagnostic_repetitions:
        parser.error('non-R5 repetitions require --diagnostic-repetitions')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory; existing task records are immutable")
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.selector_mode == "legacy_product" and (args.batch_size not in {None, 64} or args.batch_rule is not None):
        parser.error("legacy product CLI replay uses the fixed product batch size 64")
    if args.dry_run and args.task != "calibrate":
        parser.error("dry-run is only supported for calibration task planning")
    torch.set_num_threads(args.cpu_threads)
    summary: dict[str, object] = {
        "schema": TASK_RESULT_SCHEMA, "task": args.task,
        "artifact_path": str(args.model_artifact.resolve()),
        "model_artifact": str(args.model_artifact.resolve()),
        "request_id": args.request_id,
        "session_dirs": [str(path.resolve()) for path in args.session_dir],
    }
    try:
        profile = None
        if args.calibration_profile:
            profile = load_old_basic_profile(source_checkpoint=args.source_checkpoint)
            args.source_checkpoint = profile['source_checkpoint']
            args.preprocessing_config = profile['preprocessing']
            args.training_budget = profile['training_budget']
            args.selector_mode = 'unified'
            args.selection_protocol = 'fixed_full_calibration'
            args.reference_channel = 'Cz'
            transfer_policy = profile['recipe']
            args.batch_size, args.seed = transfer_policy.batch_size, transfer_policy.seed
            summary['calibration_profile'] = profile['record']
        for name in ('training_budget', 'calibration_recipe'):
            path = getattr(args, name)
            if path is not None:
                content = path.read_bytes()
                summary[name+'_input'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest()}
                value = json.loads(content)
                if name == 'training_budget':
                    training_budget = value
                else:
                    transfer_policy = TargetRecipe(**value)
                    if args.batch_size is None:
                        args.batch_size = transfer_policy.batch_size
        requested_device = args.device
        args.device = model_task_device(requested_device)
        summary["execution"] = {"requested_device": requested_device, "tensor_device": args.device,
                                "torch_threads": torch.get_num_threads()}
        if args.repetitions is not None and args.repetitions < 1:
            raise ValueError("requested repetitions must be positive")
        if args.preprocessing_config is not None and args.task != "calibrate":
            raise ValueError("preprocessing overrides are only allowed when calibrating a new model")
        if args.eog_reference_policy != SHARED_MEASURED_REFERENCE and (
                args.task != "calibrate" or not args.ocular_correction):
            raise ValueError("an EOG reference policy requires fitting automatic ocular correction")
        _progress(5, "validate_recordings")
        roots = tuple(root.resolve() for root in args.session_dir)
        if args.task in {"inspect", "infer"}:
            artifact_bytes = args.model_artifact.read_bytes()
            summary["artifact_sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
            predictor = load_calibrated_predictor(io.BytesIO(artifact_bytes), device=args.device)
            predictor.validate_personal_workflow_metadata()
            summary.update(subject_id=predictor.metadata["subject_id"],
                           calibration_profile=predictor.metadata.get('calibration_profile'),
                           simulation_status=predictor.metadata["input_contract"]["simulation_status"])
        if args.task == "inspect":
            summary.update(status="completed", **_model_summary(predictor))
            summary["epoch_qc_model_path"] = _export_epoch_quality(predictor, args.output_dir)
            atomic_write_json(summary_path, summary)
            _progress(100, f"result {summary_path.resolve()}")
            return
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("session directories must be nonempty and unique")
        if args.task == "calibrate":
            preprocessing = calibration_preprocessing(args.preprocessing_config)
            batch_rule = json.loads(args.batch_rule.read_text(encoding="utf-8")) if args.batch_rule else None
            if args.model_artifact.exists():
                raise FileExistsError("model artifact exists; choose a new version")
            if args.ocular_correction and args.ocular_model is not None:
                raise ValueError("choose automatic ocular correction or an existing ocular model")
            with ExitStack() as metadata_leases:
                validated = []
                for root in roots:
                    session = validate_brainsync_bids_session(root)
                    metadata_leases.callback(session.bids.close)
                    validated.append((session, root))
                validated.sort(key=lambda item: item[0].started_timestamp_s)
            continuous = load_continuous_qc(args.continuous_qc_model) if args.continuous_qc_model else None
            resolve_brainsync_reference(args.reference_channel, None, continuous, mode=args.reference_mode)
            epoch_quality = load_epoch_quality(args.epoch_qc_model) if args.epoch_qc_model else None
            epoch_policy = PersonalEpochQualityPolicy() if args.epoch_qc else None
            if (not args.dry_run and args.selector_mode == "legacy_product"
                    and args.source_checkpoint is not None and len(validated) < 2):
                raise ValueError("source recipe selection requires at least two labelled calibration sessions")
            if args.dry_run:
                ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
                reference = resolve_brainsync_reference(args.reference_channel, ocular, continuous, mode=args.reference_mode)
                dataset = apply_brainsync_reference(
                    load_brainsync_sessions(roots, preprocessing=preprocessing, ocular_model=ocular,
                                           continuous_qc_model=continuous), reference, ocular,
                )
                summary.update(preflight_calibrated_predictor(
                    dataset, source_checkpoint=args.source_checkpoint, selector_mode=args.selector_mode,
                    selection_protocol=args.selection_protocol, batch_size=args.batch_size, seed=args.seed,
                    planned_ocular=args.ocular_correction,
                    batch_rule=batch_rule,
                    epoch_quality_policy=epoch_policy,
                    epoch_quality_model=epoch_quality,
                    declared_preprocessing=preprocessing if args.preprocessing_config is not None else None,
                    transfer_policy=transfer_policy,
                    unified_training_budget=training_budget,
                ))
                summary["planned_epoch_qc"] = args.epoch_qc or epoch_quality is not None
                if profile is not None:
                    validate_executed_profile_source(profile['record'], summary['source_checkpoint_gate'].get('checkpoint_sha256'))
                summary["planned_ocular"]["eog_reference_policy"] = args.eog_reference_policy
                atomic_write_json(summary_path, summary)
                _progress(100, f"dry run {summary_path.resolve()}")
                return
            ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
            planned_reference = resolve_brainsync_reference(args.reference_channel, ocular, continuous, mode=args.reference_mode)
            if args.ocular_correction:
                fitting = validated
                if (args.selector_mode != "unified" and len(validated) > 1
                        and (args.source_checkpoint is not None or args.selector_mode != "legacy_product")):
                    fitting = validated[:-max(1, int(np.ceil(0.2 * len(validated))))]
                _progress(10, "fit_calibration_eog_regression")
                ocular = fit_brainsync_eog_sessions(
                    tuple(root for _, root in fitting), reference_channel=planned_reference,
                    continuous_qc_model=continuous,
                    eog_reference_policy=args.eog_reference_policy,
                    preprocessing=preprocessing,
                )
            reference = resolve_brainsync_reference(args.reference_channel, ocular, continuous, mode=args.reference_mode)
            dataset = apply_brainsync_reference(
                load_brainsync_sessions(roots, preprocessing=preprocessing, ocular_model=ocular,
                                       continuous_qc_model=continuous), reference, ocular
            )
            request = {
                "reference_channel": reference,
                "reference_mode": args.reference_mode,
                "acquisition_references": [
                    {"session_id": item.session_id, "label": item.bids.source_reference,
                     "details": dict(item.bids.source_reference_details)}
                    for item, _ in validated
                ],
                "ocular_model": None if ocular is None else ocular.record(),
                "continuous_qc_model": None if continuous is None else continuous.record(),
            }
            predictor = fit_calibrated_predictor(
                dataset, source_checkpoint=args.source_checkpoint, device=args.device,
                input_request=request, progress=_progress,
                selector_mode=args.selector_mode, selection_protocol=args.selection_protocol,
                transfer_policy=transfer_policy,
                unified_training_budget=training_budget,
                batch_size=args.batch_size, seed=args.seed,
                batch_rule=batch_rule,
                epoch_quality_policy=epoch_policy,
                epoch_quality_model=epoch_quality,
                declared_preprocessing=preprocessing if args.preprocessing_config is not None else None,
            )
            if profile is not None:
                validate_executed_profile_source(profile['record'], predictor.metadata.get('source_checkpoint_sha256'))
                predictor.metadata['calibration_profile'] = profile['record']
            save_calibrated_predictor(args.model_artifact, predictor)
            artifact_bytes = args.model_artifact.read_bytes()
            restored = load_calibrated_predictor(io.BytesIO(artifact_bytes), device=args.device)
            if restored.metadata != predictor.metadata:
                raise ValueError('saved personal model metadata changed during numeric reload')
            summary['artifact_reload_verified'] = True
            summary.update(status="completed", **_model_summary(predictor))
            summary["epoch_qc_model_path"] = _export_epoch_quality(predictor, args.output_dir)
        else:
            if len(roots) != 1:
                raise ValueError("inference takes exactly one completed recording")
            if (args.source_checkpoint or args.ocular_model or args.ocular_correction or args.reference_channel or args.reference_mode != "auto"
                    or args.batch_rule or args.continuous_qc_model or args.epoch_qc or args.epoch_qc_model):
                raise ValueError("inference preprocessing is frozen in the saved model artifact")
            dataset = predictor.load_inference_session(roots[0])
            _progress(65, "predict_unlabelled_recording")
            summary.update(inference_record(predictor, dataset, repetitions=args.repetitions if args.diagnostic_repetitions else 5,
                diagnostic=args.diagnostic_repetitions))
            summary.update(model_kind=predictor.metadata["model_kind"],
                           participant_key=predictor.metadata["participant_key"],
                           simulation_status=predictor.metadata["input_contract"]["simulation_status"])
        if args.task == "calibrate":
            summary["artifact_sha256"] = hashlib.sha256(args.model_artifact.read_bytes()).hexdigest()
        atomic_write_json(summary_path, summary)
        _progress(100, f"result {summary_path.resolve()}")
    except BrainSyncNoUsableEvidenceError as error:
        if args.task != "infer":
            summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                           continuous_quality_ledger=error.ledger)
            atomic_write_json(summary_path, summary)
            raise SystemExit(1) from error
        requested = (args.repetitions or max(1, (error.scheduled_trials+8)//9)) if args.diagnostic_repetitions else 5
        rows = [{"repetition": level, "prediction": None, "status": "abstained",
                 "scores": {str(digit): None for digit in range(1, 10)},
                 "counts": {str(digit): 0 for digit in range(1, 10)},
                 "reason": "continuous_qc_no_usable_epoch"} for level in range(1, requested + 1)]
        summary.update(status="abstained", prediction=None, session_id=error.session_id,
            participant_key=error.participant_key, requested_repetitions=requested,
            **inference_trial_counts(scheduled_trials=error.scheduled_trials, input_epochs=0, usable_predictions=0),
            reason="continuous_qc_no_usable_epoch", candidate_scores=rows[-1]["scores"], by_repetition=rows,
            target_labels_used=False, mode="completed_recording_inference",
            continuous_quality_ledger=error.ledger)
        summary.update(protocol_compliant=False if error.scheduled_trials != 45 else None,
            protocol_validation='unavailable_after_full_continuous_QC_rejection',
            evaluation_role='explicit_repetition_diagnostic' if args.diagnostic_repetitions else 'product_R5',
            required_product_protocol={'scheduled_trials': 45, 'repetitions': 5, 'candidate_count': 9})
        atomic_write_json(summary_path, summary)
        _progress(100, f"abstained {summary_path.resolve()}")
    except Exception as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        atomic_write_json(summary_path, summary)
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
