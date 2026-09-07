"""Calibrate a saved participant model or infer a completed unlabelled BrainSync block."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from data.brainsync import (  # noqa: E402
    load_brainsync_inference_session,
    load_brainsync_sessions,
)
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.channel import canonical_channel_name  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.ocular import eog_regression_from_record, load_eog_regression  # noqa: E402
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions  # noqa: E402
from models.candidate_evidence import candidate_evidence_scores  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.calibrated import (  # noqa: E402
    fit_calibrated_predictor,
    load_calibrated_predictor,
    save_calibrated_predictor,
)

TASK_RESULT_SCHEMA = "n2p3_brainsync_model_task/1"


def _progress(value: int, message: str) -> None:
    print(f"PROGRESS {value}", flush=True)
    print(message, flush=True)


def _model_summary(predictor) -> dict[str, object]:
    metadata = predictor.metadata
    ocular = metadata["input_request"].get("ocular_model")
    return {
        "model_kind": metadata["model_kind"], "method": metadata["model_kind"],
        "participant_key": metadata["participant_key"], "subject_id": metadata["subject_id"],
        "calibration_trials": metadata["calibration_trials"],
        "target_trials": metadata["calibration_positives"],
        "calibration_sessions": metadata["calibration_sessions"],
        "calibration_available_timestamp_s": metadata["calibration_available_timestamp_s"],
        "selected_recipe": metadata["selection"].get("selected"),
        "internal_validation": metadata["selection"],
        "source_pretraining_used": metadata["source_pretraining_used"],
        "ocular_correction_applied": ocular is not None,
        "ocular_fit_sessions": [] if ocular is None else ocular["fit_provenance"]["calibration_session_ids"],
        "ocular_fit_policy": None if ocular is None else (
            "frozen_inner_training_sessions" if metadata["source_pretraining_used"]
            else "authorized_calibration_sessions"
        ),
        "reference": metadata["input_contract"]["source_reference"],
        "input_contract": metadata["input_contract"],
        "simulation_status": metadata["input_contract"]["simulation_status"],
    }


def _apply_reference(dataset, reference_channel, ocular_model, *, channels=None):
    if reference_channel is None:
        return dataset
    anchor = canonical_channel_name(reference_channel)
    if ocular_model is not None:
        if ocular_model.reference_channel != anchor:
            raise ValueError("requested reference differs from the fitted EOG model reference")
        return dataset
    selected = channels or tuple(name for name in dataset.channel_names if canonical_channel_name(name) != anchor)
    return adapt_observed_montage(dataset, selected, reference_channel=anchor)


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
    repeat = np.asarray(timeline.repetition_indices, dtype=int)
    if not len(indices) or np.any(repeat < 0) or not np.isin(codes, vocabulary).all():
        raise ValueError("inference candidate schedule is invalid")
    requested = int(repeat.max()) + 1 if repetitions is None else repetitions
    if requested < 1:
        raise ValueError("requested repetitions must be positive")
    rows = []
    for level in range(1, requested + 1):
        selected = (repeat < level) & (indices >= 0)
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
        "evidence_trials": len(logits), "scheduled_trials": len(indices),
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
    parser.add_argument("--ocular-model", type=Path)
    parser.add_argument("--ocular-correction", action="store_true")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory; existing task records are immutable")
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    summary: dict[str, object] = {
        "schema": TASK_RESULT_SCHEMA, "task": args.task,
        "artifact_path": str(args.model_artifact.resolve()),
        "model_artifact": str(args.model_artifact.resolve()),
    }
    try:
        _progress(5, "validate_recordings")
        roots = tuple(root.resolve() for root in args.session_dir)
        if args.task == "inspect":
            predictor = load_calibrated_predictor(args.model_artifact, device=args.device)
            summary.update(status="completed", **_model_summary(predictor))
            summary["artifact_sha256"] = hashlib.sha256(args.model_artifact.read_bytes()).hexdigest()
            atomic_write_json(summary_path, summary)
            _progress(100, f"result {summary_path.resolve()}")
            return
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("session directories must be nonempty and unique")
        if args.task == "calibrate":
            if args.model_artifact.exists():
                raise FileExistsError("model artifact exists; choose a new version")
            if args.ocular_correction and args.ocular_model is not None:
                raise ValueError("choose automatic ocular correction or an existing ocular model")
            validated = sorted(
                ((validate_brainsync_bids_session(root), root) for root in roots),
                key=lambda item: item[0].started_timestamp_s,
            )
            if args.source_checkpoint is not None and len(validated) < 2:
                raise ValueError("source recipe selection requires at least two labelled calibration sessions")
            ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
            if args.ocular_correction:
                fitting = validated
                if args.source_checkpoint is not None:
                    fitting = validated[:-max(1, int(np.ceil(0.2 * len(validated))))]
                _progress(10, "fit_calibration_eog_regression")
                ocular = fit_brainsync_eog_sessions(
                    tuple(root for _, root in fitting), reference_channel=args.reference_channel
                )
            reference = (ocular.reference_channel if ocular is not None else None)
            if args.reference_channel is not None:
                reference = canonical_channel_name(args.reference_channel)
            dataset = _apply_reference(
                load_brainsync_sessions(roots, ocular_model=ocular), reference, ocular
            )
            request = {
                "reference_channel": reference,
                "ocular_model": None if ocular is None else ocular.record(),
            }
            predictor = fit_calibrated_predictor(
                dataset, source_checkpoint=args.source_checkpoint, device=args.device,
                input_request=request, progress=_progress,
            )
            save_calibrated_predictor(args.model_artifact, predictor)
            summary.update(status="completed", **_model_summary(predictor))
        else:
            if len(roots) != 1:
                raise ValueError("inference takes exactly one completed recording")
            if args.source_checkpoint or args.ocular_model or args.ocular_correction or args.reference_channel:
                raise ValueError("inference preprocessing is frozen in the saved model artifact")
            predictor = load_calibrated_predictor(args.model_artifact, device=args.device)
            request = predictor.metadata["input_request"]
            ocular = (
                eog_regression_from_record(request["ocular_model"])
                if request.get("ocular_model") is not None else None
            )
            dataset = _apply_reference(
                load_brainsync_inference_session(
                    roots[0], ocular_model=ocular,
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
    except Exception as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        atomic_write_json(summary_path, summary)
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
