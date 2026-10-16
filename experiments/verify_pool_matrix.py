"""Verify sealed public matrix artifacts and replay every saved target model."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--dose-sources", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def initialize(runtime):
    sys.path[:0] = [str(runtime), str(Path(runtime) / "src")]
    import torch
    torch.set_num_threads(1)


@lru_cache(maxsize=4)
def data_cache(path, digest):
    from data.epochs import load_epoch_dataset
    from data.source_records import file_sha256
    if file_sha256(path) != digest:
        raise ValueError("target cache digest differs from executed evidence")
    return load_epoch_dataset(path, validation="attested")


@lru_cache(maxsize=4)
def prepared_source(checkpoint, digest, cache, cache_digest):
    from data.source_records import file_sha256
    from transfer.checkpoint import PreparedTransferCheckpoint
    if file_sha256(checkpoint) != digest:
        raise ValueError("source checkpoint digest differs from executed evidence")
    return PreparedTransferCheckpoint(checkpoint, data_cache(cache, cache_digest))


def verify_job(job):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    from transfer.binary import matched_split_digests, target_shrinkage_features
    from transfer.checkpoint import predict_n2p3_checkpoint
    from transfer.recipes import TargetRecipe, target_batch_budget
    from transfer.result_reuse import read_result_artifacts, read_result_record
    from transfer.target_experiments import _tensor_state_digest

    path = Path(job["record"])
    report = read_result_record(path)
    data = data_cache(job["cache"], report["target_cache_sha256"])
    prepared = prepared_source(job["checkpoint"], report["checkpoint_sha256"], job["cache"], report["target_cache_sha256"])
    observed = data.trial_channel_mask
    if observed is None:
        observed = np.broadcast_to(data.channel_mask, data.X.shape[:2])
    errors, maximum_error, fits = [], 0., 0
    for index, cell in enumerate(report["cells"]):
        sealed = read_result_record(path.parent / f"{index:05d}.json")
        if sealed != cell:
            raise ValueError("cell record differs from its sealed matrix report")
        if cell["status"] != "completed":
            errors.append({"index": index, "failure": cell.get("failure")})
            continue
        predictions, model = read_result_artifacts(cell, path.parent)
        train, test = predictions["train_rows"], predictions["test_rows"]
        scope = report["task_manifest"]["scopes"][cell["scope_id"]]
        np.testing.assert_array_equal(train, scope["calibration_rows"])
        np.testing.assert_array_equal(test, scope["test_rows"])
        np.testing.assert_array_equal(predictions["y"], data.y[test])
        for key, value in matched_split_digests(train, test).items():
            if cell[key] != value:
                raise ValueError("calibration/test row digest changed")
        if len(train) != cell["K"] or np.intersect1d(train, test).size:
            raise ValueError("actual K or temporal row disjointness differs from the declared scope")
        prepared.assert_target_excluded(cell["subject"])
        if model is None:
            if cell["arm"] != "zero_shot" or len(train):
                raise ValueError("a fitted target treatment has no saved model")
            trunk = prepared.trunk_for_target(cell["subject"])
            scores = predict_n2p3_checkpoint(trunk, data.X[test], input_stats=prepared.input_statistics,
                                            device=torch.device("cuda"), trial_channel_mask=observed[test])
        else:
            fits += 1
            state = model["state"]
            if _tensor_state_digest(state) != cell["fitted_state_sha256"] or model["recipe"] != cell["recipe"]:
                raise ValueError("saved target model state or recipe changed")
            recipe = model["recipe"]
            if recipe["initialization"] == "waveform_lda":
                channels = state["waveform_channels"].numpy()
                if not observed[test][:, channels].all():
                    raise ValueError("public LDA prediction contains an unobserved selected EEG channel")
                features = target_shrinkage_features(data.X[test][:, channels], sfreq=data.preprocessing.sfreq)
                scores = ((features - state["feature_mean"].numpy()) / state["feature_scale"].numpy()) @ state["coefficient"].numpy() + state["intercept"].item()
            else:
                expected = target_batch_budget(len(train), recipe["batch_size"], epochs=recipe["epochs"], max_updates=recipe["max_updates"])
                for field in ("updates", "draws"):
                    if cell["fit_history"]["fit_budget"][field] != expected[field]:
                        raise ValueError("target optimizer updates/draws differ from the frozen recipe")
                if expected["updates"] != 200 or expected["effective_batch_size"] > 512:
                    raise ValueError("target budget violates the requested matrix")
                trunk = prepared.trunk_for_target(cell["subject"])
                if recipe["weights"] == "adapter":
                    from models.adapters import FeatureDomainTransfer

                    trunk = FeatureDomainTransfer(trunk, config=TargetRecipe(**recipe).target_adapter_config)
                    trunk.set_target_adapter_enabled(True)
                if model.get("architecture") is not None and json.dumps(
                    trunk.architecture_record(), sort_keys=True
                ) != json.dumps(model["architecture"], sort_keys=True):
                    raise ValueError("saved target architecture does not reconstruct exactly")
                trunk.load_state_dict({key.removeprefix("trunk."): value for key, value in state.items()
                                       if key.startswith("trunk.")}, strict=True)
                scores = predict_n2p3_checkpoint(trunk, data.X[test],
                    input_stats=(state["input_mean"].numpy(), state["input_std"].numpy()),
                    device=torch.device("cuda"), trial_channel_mask=observed[test])
        np.testing.assert_allclose(scores, predictions["scores"], rtol=1e-4, atol=1e-4)
        maximum_error = max(maximum_error, float(np.max(np.abs(scores - predictions["scores"]))))
        y, original = predictions["y"], predictions["scores"]
        for field, actual in (("auc", roc_auc_score(y, original)), ("ap", average_precision_score(y, original)),
                              ("bacc_at_raw_score_zero", balanced_accuracy_score(y, original > 0))):
            if not np.isclose(cell[field], actual, atol=1e-12, rtol=0):
                raise ValueError("saved metrics do not reproduce from their predictions")
    return dict(record=str(path), record_sha256=report["result_integrity"]["record_sha256"],
        status="passed" if not errors else "recorded_failures", cells=len(report["cells"]),
        fitted_models=fits, maximum_prediction_difference=maximum_error, failures=errors)


def main():
    args = arguments()
    initialize(args.runtime_checkout)
    from research.evaluation import verified_executed_snapshot
    from research.execution import atomic_write_json

    snapshot = verified_executed_snapshot(args.snapshot, root=args.runtime_checkout)
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "verification.json"
    record = json.loads(progress.read_text()) if progress.exists() else dict(results={})
    record.update(runtime_snapshot=snapshot, verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.runtime_checkout),)) as pool:
        while True:
            jobs = []
            for path in sorted(args.results.glob("*/record.json")):
                try:
                    result = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                if result.get("status") != "completed":
                    continue
                previous = record["results"].get(str(path))
                digest = result["result_integrity"]["record_sha256"]
                if previous and previous.get("record_sha256") == digest and previous["status"] == "passed":
                    continue
                task = result["task_manifest"]["source_task"]
                source_root = args.dose_sources if task["dose"] == "matched" else args.sources
                source = json.loads((source_root / task["id"] / "result.json").read_text())
                prepared = json.loads((args.prepared / task["target"] / "preparation.json").read_text())
                jobs.append(dict(record=str(path), cache=prepared["cache"],
                                 checkpoint=source["trajectory"]["checkpoints"]["4000"]))
            for result in pool.map(verify_job, jobs):
                record["results"][result["record"]] = result
                record["status"] = "running"
                atomic_write_json(progress, record)
                print(json.dumps({k: v for k, v in result.items() if k != "failures"}), flush=True)
            state = json.loads((args.results / "progress.json").read_text())
            if state.get("status") == "completed" or not args.watch:
                record.update(status="passed" if all(r["status"] == "passed" for r in record["results"].values()) else "failed",
                    verified_cells=sum(r["cells"] for r in record["results"].values()),
                    verified_fitted_models=sum(r["fitted_models"] for r in record["results"].values()))
                atomic_write_json(progress, record)
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
