"""Fit predeclared BrainSync pilot arms, freeze selection, then evaluate later sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget, resolve_cpu_worker_plan  # noqa: E402
from transfer.calibrated import (  # noqa: E402
    fit_calibrated_predictor,
    load_calibrated_predictor,
    load_calibration_source_checkpoint,
    save_calibrated_predictor,
)
from transfer.calibration_candidates import score_calibration_candidate  # noqa: E402
from transfer.recipes import TargetRecipe  # noqa: E402
from transfer.target_experiments import resolve_target_batch_rule  # noqa: E402


@lru_cache(maxsize=3)
def _dataset(path: str):
    return load_epoch_dataset(path, require_labels=True, validation="attested")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sessions(dataset):
    return set(dataset.metadata["session"].astype(str))


def validate_pilot_scopes(calibration, test, full):
    """Require complete separate recordings and label availability before testing."""
    if _sessions(calibration) & _sessions(test):
        raise ValueError("calibration and final test sessions overlap")
    if _sessions(calibration) | _sessions(test) != _sessions(full):
        raise ValueError("full refit must contain exactly calibration and test sessions")
    if len(_sessions(calibration)) < 2:
        raise ValueError("pilot selection needs separate inner calibration sessions")
    if float(calibration.metadata["calibration_available_timestamp_s"].max()) >= float(
        test.metadata["session_start_timestamp_s"].min()
    ):
        raise ValueError("calibration labels must be available before the later test starts")
    people = [set(dataset.subject_ids.astype(str)) for dataset in (calibration, test, full)]
    if len(people[0]) != 1 or any(value != people[0] for value in people[1:]):
        raise ValueError("pilot scopes require one identical participant")
    for dataset in (calibration, test):
        for session in _sessions(dataset):
            left = dataset.metadata["session"].astype(str).to_numpy() == session
            right = full.metadata["session"].astype(str).to_numpy() == session
            if (not np.array_equal(dataset.X[left], full.X[right])
                    or not np.array_equal(dataset.y[left], full.y[right])
                    or dataset.metadata.loc[left, "trial_id"].tolist() != full.metadata.loc[right, "trial_id"].tolist()):
                raise ValueError("full refit session rows differ from the evaluation inputs")


def _fit_job(job):
    plan, arm, phase, output = job
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with cpu_thread_budget(1):
        dataset = _dataset(plan["caches"]["full" if phase == "deployment" else "calibration"])
        recipe = TargetRecipe(**arm["recipe"])
        checkpoint = arm.get("checkpoint")
        if plan["device"] == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictor = fit_calibrated_predictor(
            dataset, source_checkpoint=checkpoint, recipes=(recipe,), device=plan["device"],
            seed=recipe.seed, batch_size=recipe.batch_size, batch_rule=plan["batch_rule"],
            selector_mode="five_candidates", selection_protocol="brainsync_operational_r5",
        )
        if predictor.metadata.get("recipe") != recipe.record():
            raise ValueError("fixed pilot arm unexpectedly fell back or changed its recipe")
        if plan["device"] == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        directory = Path(output) / phase
        directory.mkdir(parents=True, exist_ok=True)
        artifact = save_calibrated_predictor(directory / f"{arm['name']}.pt", predictor)
        loaded = load_calibrated_predictor(artifact, device=plan["device"])
        before, after = predictor.predict_logit(dataset), loaded.predict_logit(dataset)
        if not np.array_equal(before, after):
            raise ValueError("saved predictor changed logits on reload")
        selection = predictor.metadata["selection"]
        selected = selection["selected"]
        if selected is None:
            raise ValueError("pilot arm lacks legal internal validation")
        result = {
            "name": arm["name"], "source": arm["source"], "phase": phase,
            "recipe": recipe.record(), "status": "completed", "artifact": str(artifact),
            "artifact_sha256": _sha(artifact), "fit_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if plan["device"] == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if plan["device"] == "cuda" else None,
            "inner_primary_score": selected["primary_score"], "inner_auc": selected["validation_auc"],
            "selection": selection, "source_pretraining_used": predictor.metadata["source_pretraining_used"],
            "calibration_trials": dataset.n_epochs, "calibration_positives": int(dataset.y.sum()),
            "calibration_sessions": predictor.metadata["calibration_sessions"],
            "reload_logits_bitwise_equal": True,
        }
        atomic_write_json(directory / f"{arm['name']}.json", result)
        return result


def pilot_rank(record):
    preference = {"waveform_lda": 0, "scratch": 1, "source": 2}
    return (-record["inner_primary_score"], -record["inner_auc"],
            preference[record["recipe"]["initialization"]], record["recipe"]["epochs"], record["name"])


def _run_jobs(plan, arms, phase, output, workers):
    cpu_plan = resolve_cpu_worker_plan(len(arms), requested_workers=workers, total_threads=workers)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cpu_plan.effective_workers, mp_context=mp.get_context("spawn")) as executor:
        results = list(executor.map(_fit_job, [(plan, arm, phase, output) for arm in arms]))
    return results, {"wall_seconds": time.perf_counter() - started, "cpu_plan": cpu_plan.record()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "n2p3_brainsync_pilot/1":
        raise ValueError("versioned pilot plan required")
    verified_executed_snapshot(args.snapshot, root=ROOT)
    for path, expected in plan["input_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    batch, rule = resolve_target_batch_rule({"batch_rule": plan["batch_rule"]})
    calibration, test, full = (_dataset(plan["caches"][scope]) for scope in ("calibration", "test", "full"))
    validate_pilot_scopes(calibration, test, full)
    if len({arm["name"] for arm in plan["arms"]}) != len(plan["arms"]):
        raise ValueError("arm names must be unique")
    gates = {}
    for arm in plan["arms"]:
        recipe = TargetRecipe(**arm["recipe"])
        if recipe.batch_size != batch:
            raise ValueError("arm batch does not match the predeclared cap")
        checkpoint = arm.get("checkpoint")
        if checkpoint is not None and checkpoint not in gates:
            gates[checkpoint] = load_calibration_source_checkpoint(calibration, checkpoint)[3]
            if gates[checkpoint]["status"] != "compatible":
                raise ValueError(gates[checkpoint])
        if recipe.initialization == "source":
            controls = [item for item in plan["arms"] if item["source"] == arm["source"]
                        and item["recipe"]["initialization"] == "scratch"]
            if not any(all(item["recipe"][key] == arm["recipe"][key]
                           for key in ("batch_size", "epochs", "lr", "weight_decay", "seed")) for item in controls):
                raise ValueError("every transfer arm needs its matched target-only budget")
    args.output.mkdir(parents=True, exist_ok=False)
    preflight = {
        "status": "preflight_passed", "plan_sha256": _sha(args.plan), "batch_rule": rule,
        "source_gates": gates, "scope_trials": {key: _dataset(value).n_epochs for key, value in plan["caches"].items()},
        "scope_sessions": {key: sorted(_sessions(_dataset(value))) for key, value in plan["caches"].items()},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if plan["device"] == "cuda" else "cpu",
        "source_snapshot_manifest_sha256": _sha(args.snapshot), "arms": plan["arms"],
        "scope": "one_participant_retrospective_device_pilot_not_adult_confirmation",
    }
    atomic_write_json(args.output / "preflight.json", preflight)
    print(json.dumps({"status": preflight["status"], "arms": len(plan["arms"]), "trials": preflight["scope_trials"]}), flush=True)
    if args.dry_run:
        return
    probes = [arm for arm in plan["arms"] if arm["recipe"]["initialization"] == "source"
              and arm["recipe"]["epochs"] == 60][:2]
    timings = []
    for workers in (1, 2):
        _, timing = _run_jobs(plan, probes, f"probe_w{workers}", args.output, workers)
        timings.append({"workers": workers, **timing})
    workers = min(timings, key=lambda item: item["wall_seconds"])["workers"]
    atomic_write_json(args.output / "performance.json", {"comparisons": timings, "chosen_workers": workers,
                      "batch_size": batch, "batch_selected_for_accuracy": False,
                      "measurement": "end_to_end_spawn_fit_reload_two_calibration_jobs", "test_used": False})
    print(f"workers={workers}, calibration fitting", flush=True)
    records, training_timing = _run_jobs(plan, plan["arms"], "calibration", args.output, workers)
    ranked = sorted(records, key=pilot_rank)
    winner = ranked[0]["name"]
    deployment_names = {winner}
    for source in {arm["source"] for arm in plan["arms"] if arm["recipe"]["initialization"] == "source"}:
        deployment_names.add(next(row["name"] for row in ranked if row["source"] == source
                                  and row["recipe"]["initialization"] == "source"))
    deployment_names.add(next(row["name"] for row in ranked if row["recipe"]["initialization"] == "waveform_lda"))
    frozen_selection = {"winner": winner, "ranking": [row["name"] for row in ranked],
                        "deployment_names": sorted(deployment_names), "test_labels_used": False,
                        "rule": "inner_operational_r5_then_auc_then_baseline_then_lower_budget_then_name"}
    atomic_write_json(args.output / "selection_before_test.json", frozen_selection)
    evaluation = []
    with cpu_thread_budget(1):
        for row in records:
            predictor = load_calibrated_predictor(row["artifact"], device=plan["device"])
            predictor.validate_inference_scope(test)
            logits = predictor.predict_logit(test)
            np.savez_compressed(args.output / f"test_{row['name']}.npz", logits=logits, y=test.y,
                                session=test.metadata["session"].astype(str).to_numpy(dtype=str),
                                trial_id=test.metadata["trial_id"].astype(str).to_numpy(dtype=str))
            metrics = score_calibration_candidate(test, logits, np.ones(test.n_epochs, dtype=bool),
                                                  protocol="brainsync_operational_r5")
            evaluation.append({"name": row["name"], "source": row["source"], "recipe": row["recipe"],
                               "selected_before_test": row["name"] == winner, "metrics": metrics,
                               "r5_coverage": sum(block["complete"] for block in metrics["blocks"]) / len(metrics["blocks"])})
    atomic_write_json(args.output / "heldout_evaluation.json", {"selection": frozen_selection,
                      "independent_participants": 1, "independent_test_sessions": len(_sessions(test)),
                      "test_trials": test.n_epochs, "calibration_trials": calibration.n_epochs,
                      "blocks_are_repeated_observations_of_same_target": True, "arms": evaluation})
    deployments, refit_timing = _run_jobs(plan, [arm for arm in plan["arms"] if arm["name"] in deployment_names],
                                         "deployment", args.output, workers)
    atomic_write_json(args.output / "completed.json", {
        "status": "completed", "selection": frozen_selection, "calibration": records,
        "heldout_evaluation": evaluation, "deployment": deployments,
        "training_timing": training_timing, "deployment_timing": refit_timing,
        "deployment_models_independently_tested": False,
        "deployment_scope": "all_valid_sessions_including_prior_holdout_new_independent_recording_required",
    })
    print(json.dumps({"status": "completed", "winner": winner, "deployments": len(deployments)}), flush=True)


if __name__ == "__main__":
    main()
