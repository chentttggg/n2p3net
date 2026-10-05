"""Execute paired A/B/C personal fits with frozen preprocessing and saved-model replay."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.channel import canonical_channel_name  # noqa: E402
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    select_epoch_channels,
    select_scheduled_events,
)
from data.source_records import file_sha256  # noqa: E402
from experiments.run_brainsync_recovery import (  # noqa: E402
    binary_metrics,
    candidate_artifact,
    dense_prediction,
    observed,
    validate_evaluation_scope,
)
from research.contracts import semantic_sha256  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_matrix import MATRIX_SEEDS  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.calibrated import load_calibrated_predictor, save_calibrated_predictor  # noqa: E402
from transfer.calibration_candidates import (  # noqa: E402
    fit_calibration_candidate,
    score_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.checkpoint import PreparedTransferCheckpoint, predict_n2p3_checkpoint  # noqa: E402
from transfer.recipes import TargetRecipe  # noqa: E402
from transfer.result_reuse import (  # noqa: E402
    ResultIntegrityError,
    admit_completed_result,
    seal_result_record,
)


def personal_tasks(plan):
    tasks = []
    for budget, record in plan["budgets"].items():
        for key, variant in record["variants"].items():
            for seed in MATRIX_SEEDS:
                for initialization in ("source", "scratch"):
                    tasks.append(dict(id=f"C_K{budget}_{key}_{initialization}_s{seed}", group="C",
                        budget=int(budget), variant=variant, initialization=initialization, seed=seed,
                        source_task=f"Bproduct_B0_s{seed}", channels=None,
                        aliases=["Bproduct_B0" if initialization == "source" else "Bproduct_scratch"]
                        if key == "alpha0_pipeline" else []))
            tasks.append(dict(id=f"C_K{budget}_{key}_lda", group="C", budget=int(budget), variant=variant,
                initialization="waveform_lda", seed=MATRIX_SEEDS[0], source_task=f"Bproduct_B0_s{MATRIX_SEEDS[0]}",
                channels=None, aliases=["Bproduct_LDA"] if key == "alpha0_pipeline" else []))
        baseline = record["variants"]["alpha0_pipeline"]
        for seed in MATRIX_SEEDS:
            tasks.append(dict(id=f"Bproduct_B7_K{budget}_s{seed}", group="Bproduct", budget=int(budget), variant=baseline,
                initialization="source", seed=seed, source_task=f"Bproduct_B7_s{seed}", channels=None, aliases=[]))
            for initialization in ("source", "scratch"):
                tasks.append(dict(id=f"A_K{budget}_{initialization}_s{seed}", group="A", budget=int(budget), variant=baseline,
                    initialization=initialization, seed=seed, source_task=f"A_GTN_s{seed}", channels=["Fz", "Pz"], aliases=[]))
        tasks.append(dict(id=f"A_K{budget}_lda", group="A", budget=int(budget), variant=baseline,
            initialization="waveform_lda", seed=MATRIX_SEEDS[0], source_task=f"A_GTN_s{MATRIX_SEEDS[0]}",
            channels=["Fz", "Pz"], aliases=[]))
    for seed in MATRIX_SEEDS:
        for group, source, channels in (("A", "A_GTN", ["Fz", "Pz"]),
                                         ("B0", "Bproduct_B0", None), ("B7", "Bproduct_B7", None)):
            tasks.append(dict(id=f"{group}_K0_s{seed}", group=group, budget=0, initialization="zero_shot", seed=seed,
                source_task=f"{source}_s{seed}", channels=channels, variant={"test": plan["zero_test"]}, aliases=[]))
    for task in tasks:
        task["evaluation_sessions"] = list(plan["test_sessions"])
    return tasks


@lru_cache(maxsize=6)
def dataset(path, digest, channels):
    if file_sha256(path) != digest:
        raise ValueError("personal cache changed from the frozen preparation plan")
    data = load_epoch_dataset(path, validation="attested")
    if channels is None:
        return data
    indices = [list(map(canonical_channel_name, data.channel_names)).index(canonical_channel_name(c)) for c in channels]
    usable = observed(data)[:, indices].any(axis=1)
    if not usable.all():
        evidence = data.event_timeline.evidence_indices
        keep = np.zeros(len(evidence), bool)
        available = evidence >= 0
        keep[available] = usable[evidence[available]]
        data = select_scheduled_events(data, keep, preserve_schedule=True,
            reason="chosen_model_montage_has_no_observed_EEG")
    return select_epoch_channels(data, channels)


def personal_result_contract(job):
    """Bind one treatment independently of the subset queued for this invocation."""
    task = job["task"]
    recipe = None if task["budget"] == 0 else (
        waveform_lda_recipe(batch_size=512, seed=task["seed"])
        if task["initialization"] == "waveform_lda" else TargetRecipe(name=task["initialization"],
            initialization=task["initialization"], batch_size=512, max_updates=200, seed=task["seed"]))
    return dict(schema="n2p3_personal_matrix_result/1", task=task,
        source_snapshot=job["snapshot"], checkpoint_sha256=job["checkpoint_sha256"],
        environment=job["environment"], recipe=None if recipe is None else recipe.record(),
        fit_performed=recipe is not None)


def execute_fit(job):
    with cpu_thread_budget(1):
        task = job["task"]
        variant = task["variant"]
        channels = None if task["channels"] is None else tuple(task["channels"])
        test = dataset(variant["test"]["path"], variant["test"]["sha256"], channels)
        train = None if task["budget"] == 0 else dataset(variant["train"]["path"], variant["train"]["sha256"], channels)
        output = Path(job["output"])
        output.mkdir(parents=True, exist_ok=False)
        if file_sha256(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("source checkpoint changed")
        prepared = PreparedTransferCheckpoint(job["checkpoint"], test if train is None else train)
        subject = str(test.subject_ids[0])
        source = prepared.trunk_for_target(subject)
        record = {**personal_result_contract(job), "plan_sha256": job["plan_sha256"],
                  "source_training_contract_digest": prepared.training_contract_digest}
        artifacts = {}
        started = time.perf_counter()
        if train is None:
            logits = predict_n2p3_checkpoint(source, test.X, input_stats=prepared.input_statistics,
                device=torch.device("cuda"), trial_channel_mask=observed(test))
            record.update(fit_performed=False, target_updates=0)
        else:
            validate_evaluation_scope(train, test)
            recipe = TargetRecipe(**record["recipe"])
            fitted = fit_calibration_candidate(train, recipe, np.ones(train.n_epochs, bool),
                source_trunk=source, source_input_stats=prepared.input_statistics, device="cuda")
            predictor = candidate_artifact(train, fitted, prepared.payload,
                {"status": "compatible", "checkpoint_sha256": job["checkpoint_sha256"]},
                device="cuda", plan_sha256=job["plan_sha256"])
            predictor.metadata["experiment_source_snapshot"] = job["snapshot"]
            predictor.metadata["calibration_raw_authorization"] = train.provenance.get("calibration_raw_authorization")
            predictor.metadata["matrix_task"] = task
            model_path = save_calibrated_predictor(output / "model.pt", predictor)
            artifacts["model"] = model_path
            restored = load_calibrated_predictor(model_path, device="cuda")
            restored.validate_inference_scope(test)
            logits = restored.predict_logit(test)
            np.testing.assert_array_equal(logits, predictor.predict_logit(test))
            np.testing.assert_allclose(logits, fitted.predict_logit(test.X, observed(test)), rtol=1e-4, atol=1e-4, equal_nan=True)
            record.update(fit_performed=True, recipe=recipe.record(), fit_history=fitted.fit_history,
                model_sha256=file_sha256(model_path), model_reload_equal=True, actual_training_rows=train.n_epochs)
            if task["initialization"] != "waveform_lda" and fitted.fit_history["fit_budget"]["updates"] != 200:
                raise ValueError("personal target optimizer updates differ from frozen budget")
        torch.cuda.synchronize()
        record["seconds"] = time.perf_counter() - started
        scoring_ids = variant.get("scoring_event_ids")
        dense = dense_prediction(test, logits)
        if scoring_ids is not None:
            dense[~np.isin(test.event_timeline.event_ids, scoring_ids)] = np.nan
        predictions = []
        for session in sorted(set(test.event_timeline.session_ids.astype(str))):
            data = select_scheduled_events(test, test.event_timeline.session_ids == session,
                                            reason="per-session locked development evaluation")
            event_mask = test.event_timeline.session_ids == session
            values = dense[event_mask]
            timeline = data.event_timeline
            rows = np.flatnonzero(timeline.evidence_indices >= 0)
            row_logits = np.full(data.n_epochs, np.nan)
            row_logits[timeline.evidence_indices[rows]] = values[rows]
            labels = (timeline.candidate_ids == timeline.target_candidate_ids).astype(int)
            path = output / f"{session}.npz"
            np.savez_compressed(path, scores=values, y=labels, event_ids=timeline.event_ids,
                raw_scores=dense_prediction(test, logits)[event_mask], candidate_ids=timeline.candidate_ids,
                repetition_indices=timeline.repetition_indices, group_ids=timeline.group_ids)
            artifacts[f"prediction:{session}"] = path
            predictions.append(dict(session=session, requested_events=len(values),
                metrics=binary_metrics(labels, values), prediction_sha256=file_sha256(path),
                strict_R5=score_calibration_candidate(data, row_logits, np.ones(data.n_epochs, bool),
                                                      protocol="brainsync_operational_r5")))
        record.update(status="completed", predictions=predictions)
        record = seal_result_record(record, files=artifacts)
        atomic_write_json(output / "result.json", record)
        return dict(id=task["id"], status="completed", seconds=record["seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--task", action="append", help="Explicit unfinished task IDs for an audited resume")
    args = parser.parse_args()
    snapshot = verified_executed_snapshot(args.snapshot, root=ROOT)
    preparation = json.loads(args.prepared.read_text())
    if preparation["status"] != "completed":
        raise ValueError("personal preprocessing is incomplete")
    tasks = personal_tasks(preparation)
    if args.task:
        if set(args.task) - {t["id"] for t in tasks}:
            raise ValueError("unknown personal matrix task")
        tasks = [t for t in tasks if t["id"] in args.task]
    plan = {"tasks": tasks, "snapshot": snapshot, "preparation_sha256": file_sha256(args.prepared)}
    plan_digest = semantic_sha256(plan)
    environment = {"torch": torch.__version__, "numpy": np.__version__, "device": "cuda", "threads_per_worker": 1}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / f"plan.{plan_digest}.json"
    if plan_path.exists():
        if semantic_sha256(json.loads(plan_path.read_text())) != plan_digest:
            raise ResultIntegrityError("result integrity: stored invocation plan changed")
    else:
        atomic_write_json(plan_path, plan)
    if not (args.output / "plan.json").exists():
        atomic_write_json(args.output / "plan.json", plan)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    pending, active, results = list(tasks), {}, []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while pending or active:
            for task in list(pending):
                if len(active) >= args.workers:
                    break
                source_path = args.sources / task["source_task"] / "result.json"
                if not source_path.exists():
                    continue
                source_record = json.loads(source_path.read_text())
                checkpoint = source_record["trajectory"]["checkpoints"]["4000"]
                job = dict(task=task, checkpoint=checkpoint, checkpoint_sha256=file_sha256(checkpoint),
                    snapshot=snapshot, plan_sha256=plan_digest, environment=environment,
                    output=str(args.output / task["id"]))
                if Path(job["output"]).exists():
                    admit_completed_result(Path(job["output"]) / "result.json", kind="personal",
                                           expected=personal_result_contract(job))
                    results.append(dict(id=task["id"], status="completed", reused=True))
                else:
                    active[pool.submit(execute_fit, job)] = task
                pending.remove(task)
            if active:
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = dict(id=task["id"], status="failed", error=repr(error))
                    results.append(result)
                    print(json.dumps(result), flush=True)
            else:
                time.sleep(10)
            atomic_write_json(args.output / "progress.json", dict(status="running", results=results,
                active=[t["id"] for t in active.values()], pending=[t["id"] for t in pending],
                plan_sha256=plan_digest, plan_path=str(plan_path)))
    atomic_write_json(args.output / "progress.json", dict(status="completed", results=results,
        failed_fits=sum(r["status"] != "completed" for r in results),
        plan_sha256=plan_digest, plan_path=str(plan_path)))


if __name__ == "__main__":
    main()
