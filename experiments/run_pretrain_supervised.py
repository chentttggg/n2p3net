"""Supervised source pretraining for the N2P3 trunk (Gate 3-S1).

Trains N2P3NetBaseline (the exact LOSO-floor training path) on every source
subject except the holdout block. The saved checkpoint carries the full model
state dict plus the auditable ``training_subject_keys`` ledger consumed by
``experiments/run_within_subject_transfer.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.deep import DeepConfig  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from data.contract import (  # noqa: E402
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset  # noqa: E402
from train.device import get_device  # noqa: E402
from train.runtime import GpuPerformanceScheduler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--holdout-subjects", default="", help="comma separated; never pretrain on these")
    parser.add_argument(
        "--cohort",
        choices=("default", "gtn"),
        default="default",
        help=(
            "Causal contract family asserted for forward-phase source caches: "
            "'gtn' enforces the revised 0.1 Hz / 1200 ms child-cohort contract."
        ),
    )
    parser.add_argument(
        "--pooling-mode",
        default="ms_flatten",
        choices=sorted({"ms_flatten", "full_unfold", "mlp_full_unfold", "quadratic_full_unfold"}),
    )
    parser.add_argument(
        "--temporal-kernel-size",
        type=int,
        default=65,
        help="ST temporal kernel width for the trunk (odd, >=3).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.source_cache, require_labels=True, validation="attested")
    if dataset.preprocessing.filter_phase == "forward":
        expected = (
            GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT
            if args.cohort == "gtn"
            else SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
        )
        assert_p300_input_contract(dataset.preprocessing, expected)
    holdout = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    subjects = np.asarray(dataset.subject_ids).astype(str)
    all_subjects = set(subjects.tolist())
    unknown = holdout - all_subjects
    if unknown:
        raise ValueError(f"holdout subjects absent from cache: {sorted(unknown)}")
    source_rows = ~np.isin(subjects, list(holdout))
    if int(source_rows.sum()) < 1000:
        raise ValueError("too few source rows remain after holdout exclusion.")

    runtime = GpuPerformanceScheduler(device, precision="fp32")
    config = DeepConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    baseline = N2P3NetBaseline(
        dataset.n_channels,
        dataset.n_times,
        dataset.preprocessing.sfreq,
        config=config,
        device=device,
        runtime=runtime,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode=args.pooling_mode,
    )

    X = np.ascontiguousarray(dataset.X[source_rows])
    y = np.asarray(dataset.y)[source_rows]
    src_subjects = subjects[source_rows]
    started = time.perf_counter()
    baseline.fit(X, y, group_ids=src_subjects)
    fit_sec = time.perf_counter() - started
    model = baseline.model_

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = getattr(baseline, "last_history", {}) or {}
    payload = {
        "trunk_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "pooling_mode": args.pooling_mode,
            "temporal_kernel_size": args.temporal_kernel_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "training": "N2P3NetBaseline supervised (LOSO-identical path)",
        },
        "source_cache": str(Path(args.source_cache).resolve()),
        "holdout_subjects": sorted(holdout),
        "source_dataset_name": dataset.name,
        "source_subjects": sorted(all_subjects),
        "training_subjects": sorted(all_subjects - holdout),
        "training_subject_keys": [
            f"{dataset.name}\0{subject}" for subject in sorted(all_subjects - holdout)
        ],
        "n_source_epochs_used": int(len(X)),
        "fit_seconds": fit_sec,
        "best_epoch": history.get("best_epoch"),
        "final_task_val_auc": history.get("final_task_val_auc"),
        "runtime": {"device": str(device)},
    }
    torch.save(payload, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "fit_seconds": fit_sec,
                "n_source_epochs_used": payload["n_source_epochs_used"],
                "training_subjects": len(payload["training_subjects"]),
                "best_epoch": history.get("best_epoch"),
                "final_task_val_auc": history.get("final_task_val_auc"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
