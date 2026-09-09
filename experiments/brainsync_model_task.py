"""Calibrate a saved participant model or infer a completed unlabelled BrainSync block."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    load_brainsync_inference_session,
    load_brainsync_sessions,
)
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.channel import canonical_channel_name  # noqa: E402
from data.continuous_qc import continuous_qc_from_record, load_continuous_qc  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.epochs import PreprocessingSpec, load_preprocessing_spec  # noqa: E402
from data.ocular import (  # noqa: E402
    EOG_REFERENCE_POLICIES,
    SHARED_MEASURED_REFERENCE,
    eog_regression_from_record,
    load_eog_regression,
)
from data.personal_quality import PersonalEpochQualityPolicy  # noqa: E402
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions  # noqa: E402
from models.candidate_evidence import candidate_evidence_scores  # noqa: E402
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
        "ocular_correction_applied": ocular is not None,
        "epoch_qc_applied": quality is not None,
        "epoch_qc_method": None if quality is None else quality["model"]["method"],
        "epoch_qc_schema": None if quality is None else quality["model"].get("schema"),
        "epoch_qc_model_digest": None if quality is None else quality["model_digest"],
        "epoch_qc_fit_sessions": [] if quality is None else quality["model"]["fit_provenance"]["session_ids"],
        "epoch_qc_lifecycle": None if quality is None else quality.get("lifecycle", "fit_inside_calibration"),
        "ocular_fit_sessions": [] if ocular is None else ocular["fit_provenance"]["calibration_session_ids"],
        "ocular_fit_policy": None if ocular is None else (
            "frozen_inner_training_sessions" if metadata["source_pretraining_used"]
            else "authorized_calibration_sessions"
        ),
        "reference": metadata["input_contract"]["source_reference"],
        "acquisition_references": metadata["input_request"].get("acquisition_references", []),
        "input_contract": metadata["input_contract"],
        "simulation_status": metadata["input_contract"]["simulation_status"],
    }


def _export_epoch_quality(predictor, output_dir: Path) -> str | None:
    if predictor.epoch_quality is None:
        return None
    return str(save_epoch_quality(output_dir / "epoch_quality.json", predictor.epoch_quality).resolve())


def _apply_reference(dataset, reference_channel, ocular_model, *, channels=None):
    if reference_channel is None:
        return dataset
    anchor = canonical_channel_name(reference_channel)
    if ocular_model is not None:
        if ocular_model.reference_channel != anchor:
            raise ValueError("requested reference differs from the fitted EOG model reference")
        return dataset
    qc = dataset.provenance.get("continuous_artifact_processing", {}).get("model", {})
    qc_reference = qc.get("spec", {}).get("reference_channel")
    if qc_reference is not None:
        if anchor != qc_reference or dataset.provenance["source_reference"] != "measured_channel:" + anchor:
            raise ValueError("requested reference differs from the frozen continuous QC reference")
        if channels is not None and tuple(channels) != tuple(dataset.channel_names):
            raise ValueError("QC-owned final montage differs from the saved input request")
        return dataset
    selected = channels or tuple(name for name in dataset.channel_names if canonical_channel_name(name) != anchor)
    return adapt_observed_montage(dataset, selected, reference_channel=anchor)


def _resolve_reference_request(reference_channel, ocular_model, continuous_model, *, mode="auto"):
    anchors = [canonical_channel_name(value) for value in (
        reference_channel,
        None if ocular_model is None else ocular_model.reference_channel,
        None if continuous_model is None else continuous_model.spec.reference_channel,
    ) if value is not None]
    if len(set(anchors)) > 1:
        raise ValueError("Requested, ocular and continuous-QC references must agree")
    if mode == "native" and anchors:
        raise ValueError("Native acquisition reference cannot use a measured-reference QC/EOG model or channel")
    return anchors[0] if anchors else None


def inference_trial_counts(*, scheduled_trials: int, input_epochs: int, usable_predictions: int) -> dict[str, int]:
    if not 0 <= usable_predictions <= input_epochs <= scheduled_trials:
        raise ValueError("inference admission counts do not reconcile with the scheduled ledger")
    return {"scheduled_trials": scheduled_trials, "input_epoch_trials": input_epochs,
            "evidence_trials": usable_predictions,
            "preprocessing_unavailable_trials": scheduled_trials - input_epochs,
            "model_unavailable_trials": input_epochs - usable_predictions,
            "unavailable_trials": scheduled_trials - usable_predictions}


def inference_record(predictor, dataset, *, repetitions: int | None = None) -> dict[str, object]:
    """Use all nine candidates and the scheduled ledger; target labels are never read."""
    predictor.validate_inference_scope(dataset)
    logits = predictor.predict_logit(dataset)
    timeline = dataset.event_timeline
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
    for level in range(1, requested + 1):
        selected = (repeat < level) & available_events
        evidence = candidate_evidence_scores(
            logits[indices[selected]], codes[selected], vocabulary,
            aggregation="mean", missing_candidate_policy="abstain",
        )
        complete = all(evidence.counts[digit] == level for digit in vocabulary)
        prediction = evidence.predicted if complete else None
        rows.append({
            "repetition": level, "prediction": prediction,
            "status": "predicted" if prediction is not None else "abstained",
            "scores": {str(digit): float(score) if np.isfinite(score) else None
                       for digit, score in evidence.scores.items()},
            "counts": {str(digit): count for digit, count in evidence.counts.items()},
            "reason": "incomplete_candidate_coverage" if not complete else "tie" if prediction is None else None,
        })
    final = rows[-1]
    session_ids = np.unique(np.asarray(timeline.session_ids).astype(str))
    if len(session_ids) != 1:
        raise ValueError("one completed inference recording is required")
    return {
        "session_id": str(session_ids[0]), "requested_repetitions": requested,
        "prediction": final["prediction"], "status": final["status"],
        "candidate_scores": final["scores"], "by_repetition": rows, "reason": final["reason"],
        **inference_trial_counts(scheduled_trials=len(indices), input_epochs=dataset.n_epochs,
                                 usable_predictions=int(np.isfinite(logits).sum())),
        "target_labels_used": False, "score_kind": "mean_binary_evidence_not_probability",
        "mode": "completed_recording_inference",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("calibrate", "infer", "inspect"))
    parser.add_argument("--session-dir", type=Path, action="append", default=[])
    parser.add_argument("--source-checkpoint", type=Path)
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
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:INDEX], xpu[:INDEX], or auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--selector-mode", choices=SELECTOR_MODES, default="legacy_product")
    parser.add_argument("--selection-protocol", choices=("binary_auc", "brainsync_operational_r5"),
                        default="brainsync_operational_r5")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-rule", type=Path)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
    }
    try:
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
        if args.task == "inspect":
            predictor = load_calibrated_predictor(args.model_artifact, device=args.device)
            summary.update(status="completed", **_model_summary(predictor))
            summary["epoch_qc_model_path"] = _export_epoch_quality(predictor, args.output_dir)
            summary["artifact_sha256"] = hashlib.sha256(args.model_artifact.read_bytes()).hexdigest()
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
            validated = sorted(
                ((validate_brainsync_bids_session(root), root) for root in roots),
                key=lambda item: item[0].started_timestamp_s,
            )
            continuous = load_continuous_qc(args.continuous_qc_model) if args.continuous_qc_model else None
            _resolve_reference_request(args.reference_channel, None, continuous, mode=args.reference_mode)
            epoch_quality = load_epoch_quality(args.epoch_qc_model) if args.epoch_qc_model else None
            epoch_policy = PersonalEpochQualityPolicy() if args.epoch_qc else None
            if (not args.dry_run and args.selector_mode == "legacy_product"
                    and args.source_checkpoint is not None and len(validated) < 2):
                raise ValueError("source recipe selection requires at least two labelled calibration sessions")
            if args.dry_run:
                ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
                reference = _resolve_reference_request(args.reference_channel, ocular, continuous, mode=args.reference_mode)
                dataset = _apply_reference(
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
                ))
                summary["planned_epoch_qc"] = args.epoch_qc or epoch_quality is not None
                summary["planned_ocular"]["eog_reference_policy"] = args.eog_reference_policy
                atomic_write_json(summary_path, summary)
                _progress(100, f"dry run {summary_path.resolve()}")
                return
            ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
            planned_reference = _resolve_reference_request(args.reference_channel, ocular, continuous, mode=args.reference_mode)
            if args.ocular_correction:
                fitting = validated
                if len(validated) > 1 and (args.source_checkpoint is not None or args.selector_mode != "legacy_product"):
                    fitting = validated[:-max(1, int(np.ceil(0.2 * len(validated))))]
                _progress(10, "fit_calibration_eog_regression")
                ocular = fit_brainsync_eog_sessions(
                    tuple(root for _, root in fitting), reference_channel=planned_reference,
                    continuous_qc_model=continuous,
                    eog_reference_policy=args.eog_reference_policy,
                )
            reference = _resolve_reference_request(args.reference_channel, ocular, continuous, mode=args.reference_mode)
            dataset = _apply_reference(
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
                batch_size=args.batch_size, seed=args.seed,
                batch_rule=batch_rule,
                epoch_quality_policy=epoch_policy,
                epoch_quality_model=epoch_quality,
                declared_preprocessing=preprocessing if args.preprocessing_config is not None else None,
            )
            save_calibrated_predictor(args.model_artifact, predictor)
            summary.update(status="completed", **_model_summary(predictor))
            summary["epoch_qc_model_path"] = _export_epoch_quality(predictor, args.output_dir)
        else:
            if len(roots) != 1:
                raise ValueError("inference takes exactly one completed recording")
            if (args.source_checkpoint or args.ocular_model or args.ocular_correction or args.reference_channel or args.reference_mode != "auto"
                    or args.batch_rule or args.continuous_qc_model or args.epoch_qc or args.epoch_qc_model):
                raise ValueError("inference preprocessing is frozen in the saved model artifact")
            predictor = load_calibrated_predictor(args.model_artifact, device=args.device)
            request = predictor.metadata["input_request"]
            preprocessing = PreprocessingSpec(**predictor.metadata["input_contract"]["preprocessing"])
            preprocessing.validate()
            ocular = (
                eog_regression_from_record(request["ocular_model"])
                if request.get("ocular_model") is not None else None
            )
            continuous = continuous_qc_from_record(request["continuous_qc_model"]) if request.get("continuous_qc_model") else None
            _resolve_reference_request(request.get("reference_channel"), ocular, continuous,
                                       mode=request.get("reference_mode", "auto"))
            dataset = _apply_reference(
                load_brainsync_inference_session(
                    roots[0], preprocessing=preprocessing, ocular_model=ocular,
                    continuous_qc_model=continuous,
                    channels=request.get("channel_names")
                    if ocular is not None or request.get("reference_channel") is None else None,
                ),
                request.get("reference_channel"), ocular,
                channels=request.get("channel_names"),
            )
            _progress(65, "predict_unlabelled_recording")
            summary.update(inference_record(predictor, dataset, repetitions=args.repetitions))
            summary.update(model_kind=predictor.metadata["model_kind"],
                           participant_key=predictor.metadata["participant_key"],
                           simulation_status=predictor.metadata["input_contract"]["simulation_status"])
        summary["artifact_sha256"] = hashlib.sha256(args.model_artifact.read_bytes()).hexdigest()
        atomic_write_json(summary_path, summary)
        _progress(100, f"result {summary_path.resolve()}")
    except BrainSyncNoUsableEvidenceError as error:
        if args.task != "infer":
            summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                           continuous_quality_ledger=error.ledger)
            atomic_write_json(summary_path, summary)
            raise SystemExit(1) from error
        if (error.participant_key != predictor.metadata["participant_key"]
                or error.started_timestamp_s <= predictor.metadata["calibration_available_timestamp_s"]
                or error.session_id in {row["session_id"] for row in predictor.metadata["calibration_sessions"]}):
            summary.update(status="failed", error={"type": "ValueError", "message": "unavailable recording violates independent participant/time scope"})
            atomic_write_json(summary_path, summary)
            raise SystemExit(1) from error
        requested = args.repetitions or error.scheduled_trials // 9
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
        atomic_write_json(summary_path, summary)
        _progress(100, f"abstained {summary_path.resolve()}")
    except Exception as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        atomic_write_json(summary_path, summary)
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
