"""Two-fold synthetic P3b latency identifiability audit for the current PCW.

The audit trains two group-disjoint synthetic folds, then evaluates paired
held-out trials at a fixed base latency and known temporal shifts. It reports
absolute tau, effective dtau=tau-tau0, and paired delta-dtau recovery without
using held-out latency to select checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, train_test_split

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.n2p3net import N2P3Net  # noqa: E402
from train.device import get_device  # noqa: E402
from train.preloaded import PreloadedDataLoader  # noqa: E402
from train.recipe import NEURAL_RIDE_V11  # noqa: E402
from train.trainer import Trainer, TrainerConfig  # noqa: E402

SFREQ = 256.0
TMIN_MS = -200.0
TMAX_MS = 1200.0
N_TIME = 358
CHANNEL_NAMES = ("Fz", "Cz", "Pz")
P3B_INDEX = 2


@dataclass(frozen=True)
class SyntheticTrainingData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    true_p3b_latency_ms: np.ndarray


def _time_ms() -> np.ndarray:
    return TMIN_MS + np.arange(N_TIME, dtype=np.float32) * (1000.0 / SFREQ)


def _colored_noise(rng: np.random.Generator, scale: float) -> np.ndarray:
    white = rng.standard_normal((len(CHANNEL_NAMES), N_TIME)).astype(np.float32)
    colored = np.empty_like(white)
    colored[:, 0] = white[:, 0]
    innovation_scale = math.sqrt(1.0 - 0.85**2)
    for index in range(1, N_TIME):
        colored[:, index] = 0.85 * colored[:, index - 1] + innovation_scale * white[:, index]
    return colored * np.float32(scale)


def _gaussian(time_ms: np.ndarray, center_ms: float, width_ms: float) -> np.ndarray:
    return np.exp(-0.5 * np.square((time_ms - center_ms) / width_ms)).astype(np.float32)


def _synthetic_trial(
    noise: np.ndarray,
    *,
    target: bool,
    p3b_latency_ms: float,
    subject_gain: float,
) -> np.ndarray:
    trial = np.asarray(noise, dtype=np.float32).copy()
    time_ms = _time_ms()
    n2 = _gaussian(time_ms, 220.0, 32.0)
    p3a = _gaussian(time_ms, 300.0, 45.0)
    trial += subject_gain * np.asarray((-0.45, -0.75, -1.15), dtype=np.float32)[:, None] * n2
    trial += subject_gain * np.asarray((0.25, 0.45, 0.65), dtype=np.float32)[:, None] * p3a
    if target:
        p3b = _gaussian(time_ms, p3b_latency_ms, 58.0)
        trial += subject_gain * np.asarray((1.6, 2.8, 4.8), dtype=np.float32)[:, None] * p3b
    return trial


def make_synthetic_training_data(
    *,
    n_subjects: int,
    n_target_per_subject: int,
    n_nontarget_per_subject: int,
    base_latency_ms: float,
    train_jitter_ms: float,
    noise_std: float,
    seed: int,
) -> SyntheticTrainingData:
    if n_subjects < 4:
        raise ValueError("The two-fold audit requires at least four synthetic subjects.")
    if min(n_target_per_subject, n_nontarget_per_subject) < 1:
        raise ValueError("Each synthetic subject requires both target classes.")
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    latency: list[float] = []
    for subject in range(n_subjects):
        subject_gain = float(rng.uniform(0.85, 1.15))
        subject_noise = float(noise_std * rng.uniform(0.9, 1.1))
        for _ in range(n_target_per_subject):
            true_latency = float(
                base_latency_ms + rng.uniform(-train_jitter_ms, train_jitter_ms)
            )
            X.append(
                _synthetic_trial(
                    _colored_noise(rng, subject_noise),
                    target=True,
                    p3b_latency_ms=true_latency,
                    subject_gain=subject_gain,
                )
            )
            y.append(1)
            groups.append(subject)
            latency.append(true_latency)
        for _ in range(n_nontarget_per_subject):
            X.append(
                _synthetic_trial(
                    _colored_noise(rng, subject_noise),
                    target=False,
                    p3b_latency_ms=base_latency_ms,
                    subject_gain=subject_gain,
                )
            )
            y.append(0)
            groups.append(subject)
            latency.append(float("nan"))
    order = rng.permutation(len(X))
    return SyntheticTrainingData(
        X=np.stack(X, dtype=np.float32)[order],
        y=np.asarray(y, dtype=np.int64)[order],
        groups=np.asarray(groups, dtype=np.int64)[order],
        true_p3b_latency_ms=np.asarray(latency, dtype=np.float64)[order],
    )


def make_paired_latency_probe(
    *,
    subject_ids: Sequence[int],
    n_trials_per_subject: int,
    base_latency_ms: float,
    shifts_ms: Sequence[float],
    noise_std: float,
    seed: int,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    if 0.0 not in shifts_ms:
        raise ValueError("Paired latency probes must include the zero-shift control.")
    rng = np.random.default_rng(seed)
    probes = {float(shift): [] for shift in shifts_ms}
    pair_ids: list[str] = []
    for subject in subject_ids:
        subject_gain = float(rng.uniform(0.85, 1.15))
        subject_noise = float(noise_std * rng.uniform(0.9, 1.1))
        for trial_index in range(n_trials_per_subject):
            noise = _colored_noise(rng, subject_noise)
            for shift in shifts_ms:
                probes[float(shift)].append(
                    _synthetic_trial(
                        noise,
                        target=True,
                        p3b_latency_ms=base_latency_ms + float(shift),
                        subject_gain=subject_gain,
                    )
                )
            pair_ids.append(f"{subject}:{trial_index}")
    return (
        {shift: np.stack(values, dtype=np.float32) for shift, values in probes.items()},
        np.asarray(pair_ids, dtype=object),
    )


def recovery_metrics(predicted: np.ndarray, expected: np.ndarray | float) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    truth = np.broadcast_to(np.asarray(expected, dtype=np.float64), prediction.shape)
    error = prediction - truth
    return {
        "bias_ms": float(error.mean()),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
        "mae_ms": float(np.mean(np.abs(error))),
        "predicted_mean_ms": float(prediction.mean()),
        "predicted_std_ms": float(prediction.std()),
        "n": int(prediction.size),
    }


def _model_kwargs() -> dict[str, object]:
    return NEURAL_RIDE_V11.model_kwargs(
        n_channels=3,
        channel_names=CHANNEL_NAMES,
        tmin_ms=TMIN_MS,
        tmax_ms=TMAX_MS,
        sfreq=SFREQ,
        n_time=N_TIME,
        baseline_mode="trial",
        tau0_ms=(220.0, 300.0, 460.0),
        tau0_bounds=((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)),
        sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 150.0)),
        overrides={
            "component_decoder": False,
            "use_innovation_likelihood": False,
            "use_repetition_evidence": False,
        },
    )


def _predict_p3b_tau(model: N2P3Net, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            output = model(tensor[start : start + 256], return_likelihood=False)
            rows.append(output.tau[:, P3B_INDEX].float().cpu())
    return torch.cat(rows).numpy().astype(np.float64)


def run_fold(
    *,
    fold_index: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    data: SyntheticTrainingData,
    probes: dict[float, np.ndarray],
    base_latency_ms: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    train_groups = sorted(set(data.groups[train_indices].tolist()))
    test_groups = sorted(set(data.groups[test_indices].tolist()))
    if set(train_groups) & set(test_groups):
        raise RuntimeError("Synthetic train/test groups must be disjoint.")
    fit_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.15,
        random_state=seed + fold_index,
        stratify=data.y[train_indices],
    )
    model = N2P3Net(**_model_kwargs())
    config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        lambda2=0.3,
        lambda3=0.0,
        lambda_pcw=0.3,
        lambda_jit=0.0,
        jit_prob=0.0,
        pos_weight=8.0,
        early_stop_patience=6,
        augment=False,
        seed=seed + fold_index,
    )
    trainer = Trainer(model, config, device=device)

    def loader(indices: np.ndarray, *, shuffle: bool) -> PreloadedDataLoader:
        return PreloadedDataLoader(
            torch.from_numpy(data.X[indices]),
            torch.from_numpy(data.y[indices]).float().unsqueeze(1),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed + fold_index,
            device=device,
        )

    started = time.perf_counter()
    history = trainer.fit(loader(fit_indices, shuffle=True), loader(val_indices, shuffle=False))
    fit_seconds = time.perf_counter() - started
    tau0 = float(model.component_window.tau0_bounded.detach().cpu()[P3B_INDEX])
    predictions = {
        shift: _predict_p3b_tau(model, values, device) for shift, values in probes.items()
    }
    base_prediction = predictions[0.0]
    conditions: dict[str, dict[str, object]] = {}
    for shift, tau_prediction in predictions.items():
        effective_dtau = tau_prediction - tau0
        paired_delta = tau_prediction - base_prediction
        conditions[f"{shift:+g}"] = {
            "shift_ms": float(shift),
            "true_tau_ms": float(base_latency_ms + shift),
            "tau": recovery_metrics(tau_prediction, base_latency_ms + shift),
            "dtau": recovery_metrics(effective_dtau, shift),
            "paired_delta_dtau": recovery_metrics(paired_delta, shift),
        }
    return {
        "fold": fold_index,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "n_fit": int(len(fit_indices)),
        "n_val": int(len(val_indices)),
        "n_test_source_rows": int(len(test_indices)),
        "fit_seconds": fit_seconds,
        "epochs_ran": len(history["train_losses"]),
        "best_task_epoch_zero_based": history.get("best_task_epoch"),
        "tau0_p3b_ms": tau0,
        "tau0": recovery_metrics(np.asarray([tau0]), base_latency_ms),
        "conditions": conditions,
    }


def _pool_fold_results(
    folds: Sequence[dict[str, object]],
    shifts_ms: Sequence[float],
    *,
    base_latency_ms: float,
) -> dict:
    tau0 = np.asarray([fold["tau0_p3b_ms"] for fold in folds], dtype=np.float64)
    pooled: dict[str, object] = {
        "tau0_p3b_ms": recovery_metrics(tau0, base_latency_ms)
    }
    conditions: dict[str, dict[str, float]] = {}
    for shift in shifts_ms:
        key = f"{float(shift):+g}"
        row: dict[str, float] = {"shift_ms": float(shift)}
        for metric_name in ("tau", "dtau", "paired_delta_dtau"):
            fold_metrics = [fold["conditions"][key][metric_name] for fold in folds]
            count = sum(metric["n"] for metric in fold_metrics)
            bias = sum(metric["bias_ms"] * metric["n"] for metric in fold_metrics) / count
            mse = sum(metric["rmse_ms"] ** 2 * metric["n"] for metric in fold_metrics) / count
            row[f"{metric_name}_bias_ms"] = float(bias)
            row[f"{metric_name}_rmse_ms"] = float(math.sqrt(mse))
        conditions[key] = row
    pooled["conditions"] = conditions
    return pooled


def _parse_shifts(raw: str) -> tuple[float, ...]:
    shifts = tuple(float(value) for value in raw.split(",") if value.strip())
    shifts = tuple(dict.fromkeys((0.0, *shifts)))
    if set(shifts) != {0.0, -20.0, 20.0, 40.0}:
        raise ValueError("This locked audit requires shifts -20, 0, +20, and +40 ms.")
    return shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--subjects", type=int, default=4)
    parser.add_argument("--target-per-subject", type=int, default=64)
    parser.add_argument("--nontarget-per-subject", type=int, default=512)
    parser.add_argument("--probe-trials-per-subject", type=int, default=64)
    parser.add_argument("--base-latency-ms", type=float, default=460.0)
    parser.add_argument("--train-jitter-ms", type=float, default=40.0)
    parser.add_argument("--shifts-ms", default="-20,20,40")
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/latency_identifiability_2fold/record.json"),
    )
    args = parser.parse_args()
    if args.folds != 2:
        parser.error("The time-bounded identifiability protocol is locked to exactly two folds.")
    shifts = _parse_shifts(args.shifts_ms)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(args.seed)
    data = make_synthetic_training_data(
        n_subjects=args.subjects,
        n_target_per_subject=args.target_per_subject,
        n_nontarget_per_subject=args.nontarget_per_subject,
        base_latency_ms=args.base_latency_ms,
        train_jitter_ms=args.train_jitter_ms,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    splitter = GroupKFold(n_splits=args.folds)
    folds: list[dict[str, object]] = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(data.X, data.y, groups=data.groups)
    ):
        test_groups = np.unique(data.groups[test_indices])
        probes, pair_ids = make_paired_latency_probe(
            subject_ids=test_groups.tolist(),
            n_trials_per_subject=args.probe_trials_per_subject,
            base_latency_ms=args.base_latency_ms,
            shifts_ms=shifts,
            noise_std=args.noise_std,
            seed=args.seed + 10_000 + fold_index,
        )
        print(
            f"[latency fold {fold_index + 1}/2] train_groups="
            f"{sorted(set(data.groups[train_indices].tolist()))} "
            f"test_groups={test_groups.tolist()} pairs={len(pair_ids)}",
            flush=True,
        )
        folds.append(
            run_fold(
                fold_index=fold_index,
                train_indices=train_indices,
                test_indices=test_indices,
                data=data,
                probes=probes,
                base_latency_ms=args.base_latency_ms,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
    record = {
        "schema": "n2p3net_synthetic_latency_identifiability/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "folds": 2,
            "group_disjoint": True,
            "paired_test_noise": True,
            "checkpoint_selection_uses_test_latency": False,
            "base_latency_ms": args.base_latency_ms,
            "shifts_ms": list(shifts),
            "tau_semantics": "effective_dtau = predicted_tau - learned_tau0",
            "current_training_has_explicit_latency_supervision": False,
        },
        "args": {**vars(args), "output": str(args.output)},
        "model_kwargs": _model_kwargs(),
        "data": {
            "shape": list(data.X.shape),
            "target_rate": float(data.y.mean()),
            "groups": sorted(set(data.groups.tolist())),
        },
        "folds": folds,
        "pooled": _pool_fold_results(
            folds,
            shifts,
            base_latency_ms=args.base_latency_ms,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(record["pooled"], ensure_ascii=False, indent=2), flush=True)
    print(f"[record] {args.output}", flush=True)


if __name__ == "__main__":
    main()
