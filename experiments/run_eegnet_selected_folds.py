"""Run the EEGNet baseline on the seven folds used by the depth comparison.

This runner deliberately keeps the full 242-subject cohort in every training
fold and holds out only the selected test subject.  That matches the partial
LOSO construction used for the dep3/dep4 comparison while avoiding the
ambiguous ``--subjects N`` prefix selection in the general baseline runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.deep import DeepBaseline  # noqa: E402
from baselines.evaluate import evaluate  # noqa: E402
from experiments.run_gtn_baseline import (  # noqa: E402
    GTN_DEFAULT_DEEP_EPOCHS,
    _load_gtn_cache,
    save_subject_scores,
)

TARGET_SUBJECTS = (
    "P3Numbers_20141023_f_11_003",
    "P3Numbers_20141023_f_11_002",
    "P3Numbers_20141023_f_12_001",
    "P3Numbers_20141023_f_12_002",
    "P3Numbers_20141023_f_13_001",
    "P3Numbers_20141023_f_14_001",
    "P3Numbers_20141023_f_14_002",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=True) + "\n")
        stream.flush()


def _selected_folds(subject_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    available = set(np.unique(subject_ids).astype(str).tolist())
    missing = [subject for subject in TARGET_SUBJECTS if subject not in available]
    if missing:
        raise RuntimeError(f"Selected EEGNet folds are missing from the cache: {missing}")
    return [
        (subject_ids != subject, subject_ids == subject)
        for subject in TARGET_SUBJECTS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEGNet on the fixed seven comparison folds")
    parser.add_argument(
        "--cache-path",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "cache"
            / "gtn_events_v2_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "tmp" / "eegnet-seven-analysis" / "eegnet_seven_20260826"),
    )
    parser.add_argument("--epochs", type=int, default=GTN_DEFAULT_DEEP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    cache_path = Path(args.cache_path).resolve()
    run_dir = Path(args.run_dir).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    X, y, digits, subject_ids, true_digits, skipped, event_timeline = _load_gtn_cache(cache_path)
    subject_ids = np.asarray(subject_ids).astype(str)
    folds = _selected_folds(subject_ids)
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    metadata = {
        "schema": "n2p3net_eegnet_selected_folds/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "model": "EEGNet",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(device),
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "X_shape": list(X.shape),
        "n_subjects": int(len(np.unique(subject_ids))),
        "target_subjects": list(TARGET_SUBJECTS),
        "skipped": list(skipped),
        "fold_protocol": "partial_loso",
    }
    _write_json(run_dir / "metadata.json", metadata)
    print(
        f"[data] X={X.shape}, subjects={len(np.unique(subject_ids))}, "
        f"selected_folds={len(folds)}, cache={cache_path}",
        flush=True,
    )
    print(f"[device] {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)

    model = DeepBaseline(
        "eegnet",
        n_chans=int(X.shape[1]),
        n_times=int(X.shape[2]),
        sfreq=256.0,
        device=device,
    )
    model.cfg.epochs = int(args.epochs)
    model.cfg.batch_size = int(args.batch_size)
    model.cfg.seed = int(args.seed)

    def on_fold_end(fold_index, fold_result, _records):
        subject = TARGET_SUBJECTS[int(fold_index)]
        payload = {
            "event": "fold_end",
            "fold": int(fold_index + 1),
            "subject": subject,
            "n_test_trials": int(fold_result.n_test_trials),
            "auc": float(fold_result.auc),
            "balanced_acc": float(fold_result.balanced_acc),
            "hit_rate": float(fold_result.hit_rate),
            "epochs_ran": fold_result.epochs_ran,
            "best_epoch": fold_result.training_history.get("best_epoch"),
            "fit_sec": fold_result.fit_sec,
            "train_losses": fold_result.train_losses,
            "val_losses": fold_result.val_losses,
        }
        _write_json(run_dir / f"fold_{fold_index + 1}.json", payload)
        _append_jsonl(progress_path, payload)
        print(
            f"[fold {fold_index + 1}/7] {subject}: trials={fold_result.n_test_trials} "
            f"AUC={fold_result.auc:.4f} BACC={fold_result.balanced_acc:.4f} "
            f"epochs={fold_result.epochs_ran}",
            flush=True,
        )

    summary = evaluate(
        model,
        X,
        y,
        digits,
        subject_ids,
        true_digits,
        folds,
        primary_decision_metric="exact_llr@3",
        fixed_error_rate=0.05,
        primary_min_coverage=0.90,
        efficiency_min_coverage=0.90,
        flash_budgets=(9, 27, 45, 90, 135),
        event_timeline=event_timeline,
        evaluation_units=TARGET_SUBJECTS,
        fold_protocol="partial_loso",
        dataset_sha256=cache_sha256,
        n_jobs=1,
        parallel_backend="thread",
        on_fold_end=on_fold_end,
    )

    scores_path = save_subject_scores(
        summary,
        "eegnet",
        run_dir / "scores" / "eegnet.json",
        seed=args.seed,
        evaluation_mode="development",
    )
    final_payload = {
        **metadata,
        "finished_utc": datetime.now(UTC).isoformat(),
        "results": asdict(summary),
        "scores_path": str(scores_path),
    }
    _write_json(run_dir / "record.json", final_payload)
    print(
        f"[result] EEGNet: AUC={summary.auc_mean:.4f} "
        f"BACC={summary.balanced_acc_mean:.4f} "
        f"hit_rate={summary.hit_rate_mean:.4f}; record={run_dir / 'record.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
