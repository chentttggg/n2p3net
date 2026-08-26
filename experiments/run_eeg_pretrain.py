"""Supervised binary-P300 backbone pretraining from a universal EpochDataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from baselines.calibration import fit_logit_calibration  # noqa: E402
from baselines.deep import DeepBaseline, DeepConfig  # noqa: E402
from data.epochs import load_epoch_dataset, select_epoch_channels  # noqa: E402
from train.device import get_device  # noqa: E402


def _auto_pos_weight(y: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Pretraining requires both target classes.")
    return negatives / positives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--model", required=True, choices=("eegnet", "inception", "conformer"))
    parser.add_argument("--channels", default=None, help="Optional exact comma-separated subset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "xpu", "cpu"))
    parser.add_argument("--out-dir", default="experiments/checkpoints")
    args = parser.parse_args()
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val-frac must be in (0,1).")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True)
    if args.channels:
        dataset = select_epoch_channels(
            dataset,
            tuple(channel.strip() for channel in args.channels.split(",") if channel.strip()),
        )
    unique_subjects = np.unique(dataset.subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("Subject-disjoint pretraining validation requires at least two subjects.")
    train_subjects, validation_subjects = train_test_split(
        unique_subjects,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
    )
    train_mask = np.isin(dataset.subject_ids, train_subjects)
    validation_mask = np.isin(dataset.subject_ids, validation_subjects)
    pos_weight = args.pos_weight if args.pos_weight > 0 else _auto_pos_weight(dataset.y[train_mask])

    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but is unavailable.")
    classifier = DeepBaseline(
        args.model,
        n_chans=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        channel_mask=dataset.channel_mask,
        config=DeepConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=pos_weight,
            seed=args.seed,
            standardize_input=True,
        ),
        device=device,
    )
    trial_channel_mask = dataset.trial_channel_mask
    classifier.fit(
        dataset.X[train_mask],
        dataset.y[train_mask],
        subject_ids=dataset.subject_ids[train_mask],
        trial_channel_mask=(
            trial_channel_mask[train_mask] if trial_channel_mask is not None else None
        ),
    )
    logits = classifier.predict_logit(
        dataset.X[validation_mask],
        trial_channel_mask=(
            trial_channel_mask[validation_mask] if trial_channel_mask is not None else None
        ),
    )
    truth = dataset.y[validation_mask]
    if len(np.unique(truth)) != 2:
        raise ValueError("Subject-disjoint pretraining validation must contain both classes.")
    if classifier.calibration_logits_ is None or classifier.calibration_labels_ is None:
        raise RuntimeError("Deep pretraining did not produce subject-disjoint calibration scores.")
    calibration = fit_logit_calibration(
        classifier.calibration_logits_,
        classifier.calibration_labels_,
        source=str(classifier.calibration_source_),
    )
    auc = float(roc_auc_score(truth, logits))
    bacc = float(
        balanced_accuracy_score(truth, (logits >= calibration.threshold).astype(np.int64))
    )
    centered = logits - logits.mean()
    transductive_bacc = float(
        balanced_accuracy_score(truth, (centered > 0).astype(np.int64))
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset.name).strip("_") or "dataset"
    stem = f"{dataset_slug}_{args.model}_{dataset.n_channels}ch"
    checkpoint = classifier.save_checkpoint(output_dir / f"{stem}.pt")
    report = {
        "dataset": dataset.record(),
        "model": args.model,
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pos_weight": pos_weight,
        "seed": args.seed,
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
            "device": str(classifier.device),
        },
        "validation_auc": auc,
        "validation_bacc": bacc,
        "validation_threshold": calibration.threshold,
        "validation_threshold_source": calibration.source,
        "transductive_validation_bacc": transductive_bacc,
        "checkpoint": str(checkpoint),
        "scope": "auxiliary_domain_only_not_main_task_acceptance",
    }
    report_path = output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pretrain] AUC={auc:.4f} bacc={bacc:.4f} checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
