"""Frozen-backend and current-expert comparison on separate real recordings.

The same driver runs in isolated interpreters against each backend. It never
patches a restored backend or changes a checkpoint's provenance. The current
CloudExecutor owns all worker and child processes, resources and retirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(os.environ.get("N2P3_MATRIX_BACKEND_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_bytes())


def evaluation_sessions(plan, job):
    if job["family"] == "zero_shot":
        return plan["zero_supervised_calibration_sessions"]
    calibration = set(plan["calibration_order"][: job["dose"]])
    remainder = [
        r["session_id"]
        for r in plan["rows"]
        if r["subject_id"] == "P001"
        and r["session_id"] not in calibration
        and r["session_id"] not in plan["test_sessions"]
    ]
    return list(plan["test_sessions"]) + remainder


def evaluation_failure(entry, error):
    return {
        "status": "incompatible_input",
        "failure": {"type": type(error).__name__, "message": str(error)},
        "session_id": entry["session_id"],
        "requested_trials": 450,
        "available_trial_scores": 0,
        "conditional_binary_auc": None,
        "blocks": [
            {
                "session_id": entry["session_id"],
                "block": b,
                "target": str(entry["target"]),
                "requested_flashes": 45,
                "eligible": False,
                "prediction": None,
                "counts": [0] * 9,
                "scores": None,
                "posterior": None,
                "reason": "incompatible_input",
            }
            for b in range(1, 11)
        ],
    }


def datasets(plan, run, pipeline, sessions):
    from data.epochs import load_epoch_dataset

    rows = {r["session_id"]: r for r in plan["rows"]}
    result = []
    for name in sessions:
        spec = rows[name]["paths"][pipeline]
        path = run / spec["path"]
        if sha(path) != spec["sha256"]:
            raise ValueError("input cache identity changed")
        result.append(load_epoch_dataset(path, require_labels=True, validation="attested"))
    return result


def merge(parts):
    from data.epochs import concatenate_epoch_datasets

    parts = sorted(parts, key=lambda d: float(d.metadata["session_start_timestamp_s"].min()))
    owner = next(
        (
            d
            for d in parts
            if d.provenance.get("continuous_artifact_processing", {}).get("status") == "applied"
        ),
        parts[0],
    )
    provenance = {
        **owner.provenance,
        "session_ids": [str(d.metadata["session"].iloc[0]) for d in parts],
        "source": "brainsync_sessions",
        "n_sessions": len(parts),
        "session_dirs": [d.provenance["session_dir"] for d in parts],
        "continuous_quality_ledgers": [
            d.provenance.get("continuous_quality_ledger") for d in parts
        ],
        "acquisition_references": [
            {
                "session_id": str(d.metadata["session"].iloc[0]),
                "label": d.provenance["acquisition_source_reference"],
                "details": d.provenance["acquisition_reference_details"],
            }
            for d in parts
        ],
    }
    return concatenate_epoch_datasets(
        parts, name="BrainSync-matrix-authorized-C", provenance=provenance
    )


def block_scores(dataset, logits, entry):
    import numpy as np

    from data.candidate_task import MembershipDecision
    from models.candidate_evidence import membership_decision_scores

    timeline = dataset.event_timeline
    blocks = []
    for block in range(1, 11):
        # Physical per-block IDs are present on both retained rows and the
        # scheduled timeline. Never reconstruct a requested block from survivors.
        indices = np.flatnonzero(
            np.asarray(timeline.group_ids).astype(str) == f"{entry['session_id']}:block:{block}"
        )
        if not len(indices):
            # BrainSync's timeline groups identify the whole recording. Its
            # original trial order still contains all 450 scheduled events.
            indices = np.arange((block - 1) * 45, block * 45)
        assert len(indices) == 45 and timeline.n_events == 450
        codes = np.asarray(timeline.candidate_ids)[indices].astype(str)
        occurrence = {str(q): 0 for q in range(1, 10)}
        repeats = []
        for code in codes:
            repeats.append(occurrence[code])
            occurrence[code] += 1
        assert set(occurrence.values()) == {5}
        decision = MembershipDecision.direct(
            codes,
            tuple(map(str, range(1, 10))),
            event_ids=np.asarray(timeline.event_ids)[indices].astype(str),
            repetition_indices=np.array(repeats),
            repetitions=5,
            evidence_indices=np.asarray(timeline.evidence_indices)[indices],
        )
        physical = decision.evidence_indices >= 0
        usable = np.zeros(45, dtype=bool)
        local_logits = np.zeros(45, dtype=np.float64)
        positions = decision.evidence_indices[physical]
        usable[physical] = np.isfinite(np.asarray(logits)[positions])
        local_logits[usable] = np.asarray(logits)[decision.evidence_indices[usable]]
        scores = membership_decision_scores(local_logits, decision, available=usable)
        record = scores.record()
        # The test target enters only this evaluation result, after scoring.
        record.update(
            session_id=entry["session_id"],
            block=block,
            target=str(entry["target"]),
            event_ids=list(decision.event_ids),
            requested_flashes=45,
            input_cache_sha256=entry["paths"][dataset.provenance.get("matrix_pipeline", "basic")][
                "sha256"
            ],
        )
        blocks.append(record)
    finite = np.isfinite(logits)
    from sklearn.metrics import roc_auc_score

    auc = (
        float(roc_auc_score(dataset.y[finite], np.asarray(logits)[finite]))
        if len(np.unique(dataset.y[finite])) == 2
        else None
    )
    return {
        "blocks": blocks,
        "available_trial_scores": int(finite.sum()),
        "requested_trials": 450,
        "conditional_binary_auc": auc,
        "conditional_auc_trials": int(finite.sum()),
    }


def prediction_dataset(dataset):
    # Fitters never receive a held recording. Predictors receive no target
    # labels, including the confirmed-target columns and timeline targets.
    import numpy as np

    from data.epochs import EpochDataset

    metadata = dataset.metadata.drop(
        columns=[
            c
            for c in dataset.metadata.columns
            if c
            in {
                "is_target",
                "target_candidate",
                "target_candidate_id",
                "target_digit",
                "thought_digit",
            }
        ]
    ).copy()
    timeline = replace(
        dataset.event_timeline, target_candidate_ids=np.repeat("", dataset.event_timeline.n_events)
    )
    result = replace(
        dataset,
        y=None,
        metadata=metadata,
        event_timeline=timeline,
        provenance={
            k: v
            for k, v in dataset.provenance.items()
            if k not in {"target_digits", "thought_digit"}
        },
        verified_cache_attestation=None,
    )
    assert isinstance(result, EpochDataset) and result.y is None
    return result


def fit_trial(job, plan, run, output):
    import numpy as np
    import torch

    from transfer.calibrated import (
        fit_calibrated_predictor,
        fit_fixed_waveform_lda,
        load_calibrated_predictor,
        save_calibrated_predictor,
    )
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.recipes import TargetRecipe

    selected = plan["calibration_order"][: job["dose"]]
    assert not set(selected) & set(plan["test_sessions"])
    calibration = merge(datasets(plan, run, job["pipeline"], selected))
    family, seed = job["family"], job["seed"]
    source = job["checkpoint"] if family in {"nn_source", "nn_scratch"} else None
    if family in {"nn_source", "nn_scratch"}:
        recipe = TargetRecipe(
            family,
            initialization="source" if family == "nn_source" else "scratch",
            input_statistics="target_prefix",
            bn_buffers="calibration",
            bn_affine="trainable",
            weights="non_bn",
            dropout="off",
            batch_size=plan["batch_size"],
            lr=plan["learning_rate"],
            epochs=plan["epochs"],
            seed=seed,
            candidate_loss_weight=0.0,
        )
        options = {
            "selector_mode": "unified",
            "transfer_policy": recipe,
            "unified_training_budget": read(ROOT / "configs/unified_training_budget.json"),
        }
    elif family == "waveform_lda":
        from transfer.target_experiments import operator_fixed_target_batch_rule

        if job["pipeline"] == "reviewed" or job.get("fixed_analytic_fit", False):
            # This public route directly fits the fixed analytic LDA on all C.
            # Its historical batch64 field has no analytic-LDA update effect.
            # Inner recording selection would place the numerical QC reference
            # outside an inner training fold and must remain rejected.
            recipe = waveform_lda_recipe(batch_size=64, seed=seed)
            options = {"selector_mode": "legacy_product"}
        else:
            recipe = waveform_lda_recipe(batch_size=plan["batch_size"], seed=seed)
            options = {
                "selector_mode": "no_transfer",
                "recipes": (recipe,),
                "batch_rule": operator_fixed_target_batch_rule(
                    batch_size=plan["batch_size"],
                    rationale="Predeclared fixed analytic control; no target-score batch selection",
                ),
            }
    else:
        recipe = TargetRecipe(
            family,
            initialization=family,
            weights="none",
            bn_buffers="source",
            bn_affine="frozen",
            dropout="off",
            batch_size=plan["batch_size"],
            epochs=1,
            seed=seed,
        )
        options = {"selector_mode": "fixed_erp", "transfer_policy": recipe}
    if family == "waveform_lda" and job.get("fixed_analytic_fit", False):
        predictor = fit_fixed_waveform_lda(calibration, device=job["device"], seed=seed,
            declared_preprocessing=calibration.preprocessing)
    else:
        predictor = fit_calibrated_predictor(
            calibration,
            source_checkpoint=source,
            device=job["device"],
            seed=seed,
            batch_size=recipe.batch_size,
            declared_preprocessing=calibration.preprocessing,
            **options,
        )
    if predictor.metadata["recipe"] != recipe.record():
        raise ValueError("declared expert was replaced by fallback")
    save_calibrated_predictor(output / "model.pt", predictor)
    loaded = load_calibrated_predictor(output / "model.pt", device=job["device"])
    evaluations = []
    rows = {r["session_id"]: r for r in plan["rows"]}
    names = evaluation_sessions(plan, job)
    for dataset, name in zip(datasets(plan, run, job["pipeline"], names), names, strict=True):
        inference = prediction_dataset(dataset)
        try:
            before, after = predictor.predict_logit(inference), loaded.predict_logit(inference)
        except ValueError as error:
            # Earlier remaining recordings can have a different frozen QC
            # contract. Preserve those requested failures instead of refitting
            # preprocessing or changing the accepted model's input contract.
            result = evaluation_failure(rows[name], error)
        else:
            if not np.array_equal(before, after, equal_nan=True):
                raise ValueError("cold-loaded model changed held logits")
            dataset.provenance["matrix_pipeline"] = job["pipeline"]
            result = block_scores(dataset, after, rows[name])
            np.savez(
                output / f"{name}_predictions.npz",
                logits=after,
                event_ids=np.asarray(dataset.event_timeline.event_ids),
                evidence_indices=np.asarray(dataset.event_timeline.evidence_indices),
            )
        result["evaluation_role"] = (
            "fixed_future_test"
            if name in plan["test_sessions"]
            else "remaining_independent_recordings_retrospective"
        )
        evaluations.append(result)
    return {
        "status": "completed",
        "recipe": recipe.record(),
        "calibration_sessions": selected,
        "requested_calibration_trials": 450 * len(selected),
        "retained_calibration_epochs": calibration.n_epochs,
        "actual_fit_scope": predictor.metadata.get(
            "fit_history", predictor.metadata["selection"]
        ).get("fit_scope"),
        "selection": predictor.metadata["selection"],
        "model_sha256": sha(output / "model.pt"),
        "reload_logits_bitwise_equal": True,
        "evaluations": evaluations,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if job["device"] == "cuda"
        else 0,
    }


def zero_shot(job, plan, run, output):
    from transfer.checkpoint import (
        checkpoint_input_stats,
        checkpoint_scores_to_llr,
        load_n2p3_trunk_checkpoint,
        predict_n2p3_checkpoint,
    )

    rows = {r["session_id"]: r for r in plan["rows"]}
    results = []
    for dataset, name in zip(
        datasets(plan, run, job["pipeline"], plan["zero_supervised_calibration_sessions"]),
        plan["zero_supervised_calibration_sessions"],
        strict=True,
    ):
        subject = str(dataset.subject_ids[0])
        trunk, payload = load_n2p3_trunk_checkpoint(
            job["checkpoint"], dataset, target_subject=subject
        )
        logits = predict_n2p3_checkpoint(
            trunk,
            dataset.X,
            trial_channel_mask=dataset.trial_channel_mask,
            input_stats=checkpoint_input_stats(payload, dataset.n_channels),
            device=job["device"],
        )
        logits = checkpoint_scores_to_llr(payload, logits)[0]
        dataset.provenance["matrix_pipeline"] = job["pipeline"]
        results.append(block_scores(dataset, logits, rows[name]))
    return {
        "status": "completed",
        "target_fits": 0,
        "checkpoint_sha256": sha(job["checkpoint"]),
        "evaluations": results,
        "calibration_supervised_labels": 0,
    }


def fit_raw(job, plan, run, output):
    import numpy as np

    from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
    from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec

    records, bindings = [], []
    metadata = None
    selected = plan["calibration_order"][: job["dose"]]
    for name in selected:
        base = run / "inputs" / name / "raw_blocks"
        metadata = read(base.with_suffix(".json"))
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
            for row in metadata["rows"]:
                if not row["legal"] or job["pipeline"] == "reviewed" and not row["reviewed_clean"]:
                    continue
                block = row["block"]
                record_id = f"{name}:block:{block}"
                records.append(
                    CalibrationRecord(
                        record_id,
                        archive[f"X_{block}"],
                        archive[f"onsets_{block}"],
                        archive[f"labels_{block}"],
                    )
                )
                bindings.append(
                    {
                        "session": name,
                        "block": block,
                        "parent_sha256": metadata["sha256"],
                        "raw_scope": row["raw_scope"],
                        "source_sha256": metadata["source_sha256"],
                    }
                )
    if not records:
        raise ValueError("no complete clean raw calibration blocks")
    rate, channels = metadata["sample_rate_hz"], tuple(metadata["channels"])
    model = ContinuousDecisionModel.fit(
        records,
        sfreq=rate,
        channels=channels,
        candidate_ids=tuple(map(str, range(1, 10))),
        operator_identity="native_unfiltered_volts/1",
        dictionary=ResponseDictionary.compile(
            ResponseDictionarySpec("gaussian", 1.2, 8), sample_rate_hz=rate
        ),
        ridge=1.0,
        noise_order=job["noise_order"],
    )
    model.save(output / "model.npz")
    loaded = ContinuousDecisionModel.load(output / "model.npz")
    results = []
    entries = {r["session_id"]: r for r in plan["rows"]}
    for name in evaluation_sessions(plan, job):
        base = run / "inputs" / name / "raw_blocks"
        metadata = read(base.with_suffix(".json"))
        blocks = []
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
            for row in metadata["rows"]:
                b = row["block"]
                record = {
                    "session_id": name,
                    "block": b,
                    "target": str(entries[name]["target"]),
                    "requested_flashes": 45,
                    "event_ids": row["event_ids"],
                }
                if not row["legal"] or job["pipeline"] == "reviewed" and not row["reviewed_clean"]:
                    record.update(
                        eligible=False,
                        prediction=None,
                        reason="raw_scope_not_clean",
                        counts=[0] * 9,
                        scores=None,
                        posterior=None,
                    )
                else:
                    args = dict(
                        operator_identity="native_unfiltered_volts/1",
                        sfreq=rate,
                        channels=channels,
                        candidate_ids=tuple(map(str, range(1, 10))),
                        record_id=f"{name}:block:{b}",
                    )
                    first = model.score(
                        archive[f"X_{b}"],
                        archive[f"onsets_{b}"],
                        archive[f"membership_{b}"],
                        **args,
                    )
                    second = loaded.score(
                        archive[f"X_{b}"],
                        archive[f"onsets_{b}"],
                        archive[f"membership_{b}"],
                        **args,
                    )
                    if not np.array_equal(first.log_evidence, second.log_evidence):
                        raise ValueError("raw-model replay changed held evidence")
                    record.update(
                        eligible=second.winner is not None,
                        prediction=second.winner,
                        reason=second.reason,
                        counts=[5] * 9,
                        scores=second.log_evidence.tolist(),
                        posterior=second.probabilities.tolist(),
                    )
                blocks.append(record)
        results.append(
            {
                "blocks": blocks,
                "evaluation_role": (
                    "fixed_future_test"
                    if name in plan["test_sessions"]
                    else "remaining_independent_recordings_retrospective"
                ),
            }
        )
    return {
        "status": "completed",
        "calibration_sessions": selected,
        "raw_fit_blocks": bindings,
        "requested_calibration_trials": 450 * len(selected),
        "actual_calibration_flashes": 45 * len(records),
        "model_sha256": sha(output / "model.npz"),
        "reload_logits_bitwise_equal": True,
        "evaluations": results,
    }


def execute_job(path):
    job = read(path)
    run, output = Path(job["run_root"]), Path(job["output"])
    plan = read(run / "plan.json")
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if job["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        function = (
            zero_shot
            if job["family"] == "zero_shot"
            else fit_raw
            if job["family"].startswith("continuous")
            else fit_trial
        )
        result = function(job, plan, run, output)
    except Exception as error:
        result = {
            "status": "failed",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "evaluations": [],
        }
    result.update(
        job=job,
        started_unix=start,
        finished_unix=time.time(),
        backend_root=str(ROOT),
        driver_sha256=sha(__file__),
        plan_sha256=sha(run / "plan.json"),
    )
    write(output / "result.json", result)
    print(
        json.dumps(
            {
                "job": job["id"],
                "status": result["status"],
                "seconds": result["finished_unix"] - start,
                "failure": result.get("failure", {}).get("message"),
            }
        ),
        flush=True,
    )
    return result


def cloud_worker(arguments):
    job = read(arguments["job_file"])
    env = dict(
        os.environ,
        N2P3_MATRIX_BACKEND_ROOT=job["backend_root"],
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    log = Path(job["output"]).with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf8") as stream:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--job-file", arguments["job_file"]],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = Path(job["output"]) / "result.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"worker interpreter exit {completed.returncode}; inspect {log}")
    return {
        "result_file": str(result_path),
        "sha256": sha(result_path),
        "scientific_status": read(result_path)["status"],
    }


def launch(run):
    from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
    from research.source_execution import HostMemoryAdmission, resolve_source_resources

    run = Path(run).resolve()
    jobs = read(run / "jobs.json")
    plan = {
        "cpu_budget": 25,
        "gpu_workers": 4,
        "source_workers": 4,
        "target_workers": 4,
        "source_worker_memory_gib": 4,
        "target_worker_memory_gib": 4,
        "gpu_task_memory_gib": 4,
        "host_memory_fraction": 0.9,
        "target_native_threads_per_worker": 4,
    }
    resources = resolve_source_resources(plan, 4, "cuda")
    write(run / "resolved_resources.json", resources)
    tasks = []
    for job in jobs:
        phase = PhaseResources(
            "calibration" if job["device"] == "cuda" else "evaluation",
            5,
            4 * 2**30,
            4,
            job["device"],
            interop_threads=1,
        )
        tasks.append(
            CloudTask(
                job["id"],
                "experiments.run_brainsync_real_matrix:cloud_worker",
                {"job_file": str(run / "jobs" / f"{job['id']}.json")},
                job["output"],
                phase,
            )
        )

    def observed(record):
        write(run / "live_observation.json", record)

    def finished(outcome):
        write(run / "job_terminal" / f"{outcome.task_id}.json", asdict(outcome))
        print(
            json.dumps(
                {
                    "terminal": outcome.task_id,
                    "execution_status": outcome.status,
                    "science": (outcome.value or {}).get("scientific_status"),
                }
            ),
            flush=True,
        )
        return ()

    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        executor = CloudExecutor(
            admission,
            max_workers=4,
            diagnostic_directory=run / "cloud_diagnostics",
            identity={"plan_sha256": sha(run / "plan.json"), "jobs_sha256": sha(run / "jobs.json")},
        )
        result = executor.run(tasks, on_finished=finished, on_observation=observed)
    write(
        run / "cloud_run.json",
        {
            "status": result.status,
            "results": {k: asdict(v) for k, v in result.results.items()},
            "diagnostics": result.diagnostics,
            "observation_failures": result.observation_failures,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--launch", type=Path)
    args = parser.parse_args()
    if args.job_file:
        execute_job(args.job_file)
    elif args.launch:
        launch(args.launch)
    else:
        parser.error("choose --job-file or --launch")


if __name__ == "__main__":
    main()
