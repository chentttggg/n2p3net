"""Matched whole-recording filter/QC study through saved personal models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from data.brainsync import (  # noqa: E402
    BRAIN_SYNC_PREPROCESSING,
    IndependentRecordingScope,
    load_brainsync_session,
    load_brainsync_sessions,
)
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.continuous_qc import save_continuous_qc  # noqa: E402
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT  # noqa: E402
from data.epochs import preprocessing_spec_from_contract, save_epoch_dataset  # noqa: E402
from data.personal_quality import PersonalEpochQualityPolicy  # noqa: E402
from experiments.fit_brainsync_continuous_qc import fit_brainsync_continuous_manifest  # noqa: E402
from experiments.run_brainsync_recovery import validate_evaluation_scope  # noqa: E402
from experiments.run_filter_phase_study import candidate_metrics  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.calibrated import (  # noqa: E402
    fit_calibrated_predictor,
    load_calibrated_predictor,
    save_calibrated_predictor,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    calibration = [Path(p) for p in plan["calibration_sessions"]]
    evaluation = [Path(p) for p in plan["evaluation_sessions"]]
    sessions = [validate_brainsync_bids_session(p) for p in calibration+evaluation]
    scope = IndependentRecordingScope(tuple(s.session_id for s in sessions[:len(calibration)]),
                                      tuple(s.session_id for s in sessions[len(calibration):]))
    if len({s.subject_id for s in sessions}) != 1:
        raise ValueError("matched personal study requires one declared participant")
    raw_hashes = {str(s.bids.raw_path):sha(s.bids.raw_path) for s in sessions}
    qc_base = json.loads(Path(plan["continuous_calibration_manifest"]).read_text(encoding="utf-8"))
    if not {Path(s["session_dir"]).resolve() for s in qc_base["sessions"]} <= {p.resolve() for p in calibration}:
        raise ValueError("continuous quality fit exceeds the study calibration sessions")
    profiles = {
        "forward_range_reject": (preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT), "reject", "all_edges"),
        "zero_range_reject": (BRAIN_SYNC_PREPROCESSING, "reject", "artifact_edges"),
        "zero_range_diagnostic": (BRAIN_SYNC_PREPROCESSING, "diagnostic", "artifact_edges"),
    }
    references = {"native7": None, "cz6": "Cz"}
    execution = dict(plan=plan, plan_sha256=sha(args.plan), raw_hashes=raw_hashes,
        expected_models=len(profiles)*len(references), expected_predictions=len(profiles)*len(references)*len(evaluation),
        profiles={k:dict(preprocessing=v[0].record(), range_action=v[1], boundary_action=v[2]) for k,v in profiles.items()},
        fit_scope="explicit_clean_calibration_only", evaluation_mode="independent_recording_holdout",
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), *sorted((ROOT/"src").rglob("*.py"))]})
    atomic_write_json(output/"execution.json", execution)
    records = []
    fitted = {}
    with threadpool_limits(limits=1):
        for reference, anchor in references.items():
            for name, (spec, range_action, boundary_action) in profiles.items():
                started = time.perf_counter()
                folder = output/f"{reference}_{name}"
                folder.mkdir()
                manifest = json.loads(json.dumps(qc_base))
                manifest["preprocessing"] = spec.record()
                manifest["spec"].update(reference_channel=anchor, flatline_min_duration_s=5.,
                                        range_action=range_action, boundary_action=boundary_action)
                path = folder/"continuous_manifest.json"
                atomic_write_json(path, manifest)
                # Reference projection and action switches do not refit native
                # thresholds/state envelopes. Share the actual native fit per phase.
                if spec.filter_phase not in fitted:
                    fitted[spec.filter_phase] = fit_brainsync_continuous_manifest(path)
                base = fitted[spec.filter_phase]
                quality = replace(base, spec=replace(base.spec, reference_channel=anchor,
                                                     range_action=range_action, boundary_action=boundary_action))
                save_continuous_qc(folder/"continuous_qc.json", quality)
                train = load_brainsync_sessions(calibration, preprocessing=spec,
                    continuous_qc_model=quality, recording_scope=scope)
                save_epoch_dataset(folder/"calibration.npz", train)
                model = fit_calibrated_predictor(train, declared_preprocessing=spec,
                    epoch_quality_policy=PersonalEpochQualityPolicy())
                model.metadata["evaluation_mode"] = execution["evaluation_mode"]
                model.metadata["executed_source"] = execution["source_files"]
                artifact = save_calibrated_predictor(folder/"personal_model.pt", model)
                restored = load_calibrated_predictor(artifact)
                row = dict(condition=folder.name, reference=reference, profile=name,
                    preprocessing=spec.record(), range_action=range_action, boundary_action=boundary_action,
                    calibration_scheduled=train.event_timeline.n_events, calibration_epochs=train.n_epochs,
                    calibration_complete_channel_epochs=int(train.trial_channel_mask.all(axis=1).sum()),
                    model_path=str(artifact), model_sha256=sha(artifact), evaluations=[])
                for root, session in zip(evaluation, sessions[len(calibration):], strict=True):
                    data = load_brainsync_session(root, preprocessing=spec, continuous_qc_model=quality,
                                                  recording_scope=scope)
                    holdout = validate_evaluation_scope(train, data, mode=execution["evaluation_mode"])
                    cache = save_epoch_dataset(folder/f"{session.session_id}.npz", data)
                    scores = model.predict_logit(data)
                    replay = restored.predict_logit(data)
                    np.testing.assert_array_equal(np.isfinite(scores), np.isfinite(replay))
                    np.testing.assert_allclose(scores, replay, rtol=0, atol=1e-10, equal_nan=True)
                    usable = np.isfinite(scores)
                    metric = float(roc_auc_score(data.y[usable], scores[usable])) if len(np.unique(data.y[usable]))==2 else None
                    prediction_path = folder/f"{session.session_id}_predictions.npz"
                    np.savez_compressed(prediction_path, scores=scores, labels=data.y,
                        event_ids=data.event_timeline.event_ids, evidence_indices=data.event_timeline.evidence_indices)
                    row["evaluations"].append(dict(session=session.session_id, scope=holdout,
                        scheduled=data.event_timeline.n_events, input_epochs=data.n_epochs,
                        complete_channel_epochs=int(data.trial_channel_mask.all(axis=1).sum()),
                        predicted_epochs=int(usable.sum()), auc=metric, r5=candidate_metrics(data, scores),
                        cache=str(cache), predictions=str(prediction_path), predictions_sha256=sha(prediction_path)))
                row["seconds"] = time.perf_counter()-started
                records.append(row)
                atomic_write_json(folder/"result.json", row)
                print(json.dumps({k:v for k,v in row.items() if k not in {"evaluations","preprocessing"}}),flush=True)
    assert all(sha(path)==digest for path,digest in raw_hashes.items())
    atomic_write_json(output/"results.json", dict(execution=execution, conditions=records,
        completed_models=len(records), verified_predictions=sum(len(r["evaluations"]) for r in records),
        raw_hashes_unchanged=True))


if __name__ == "__main__":
    main()
