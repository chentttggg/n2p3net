"""Pretrain the compact MS-EEGNet trunk with masked EEG reconstruction.

The decoder is discardable and is never saved in the deployed checkpoint.
Source data may use either the offline zero-phase contract (cross-subject
sources) or the causal contract; target fine-tuning still requires a causal
cache produced by ``run_within_subject_transfer.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import (  # noqa: E402
    DEFAULT_P300_DATA_CONTRACT,
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    PAPER_GTN_CAUSAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from train.device import get_device  # noqa: E402
from train.runtime import GpuPerformanceScheduler, MatrixBatchSource  # noqa: E402
from transfer.losses import ReconstructionLossConfig  # noqa: E402
from transfer.masking import MaskingConfig  # noqa: E402
from transfer.pretraining import PretrainingConfig, PretrainingTask  # noqa: E402


def _parse_bands(value: str) -> tuple[tuple[float, float], ...]:
    bands: list[tuple[float, float]] = []
    for part in value.split(","):
        start, end = part.split("-", 1)
        bands.append((float(start), float(end)))
    return tuple(bands)


def _source_training_rows(
    subject_ids: np.ndarray,
    holdout: set[str],
) -> tuple[np.ndarray, set[str]]:
    subjects = np.asarray(subject_ids).astype(str)
    all_subjects = set(subjects.tolist())
    unknown_holdout = holdout - all_subjects
    if unknown_holdout:
        raise ValueError(
            "holdout subjects are absent from the source cache: "
            f"{sorted(unknown_holdout)}"
        )
    return ~np.isin(subjects, list(holdout)), all_subjects


def _subject_probe_validation_mask(
    subject_ids: np.ndarray,
    *,
    seed: int,
    fraction: float = 0.2,
) -> np.ndarray:
    """Hold out a deterministic within-subject trial subset for probe validation."""

    subjects = np.asarray(subject_ids).astype(str)
    validation = np.zeros(len(subjects), dtype=bool)
    rng = np.random.default_rng(seed)
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        if len(rows) < 2:
            raise ValueError("subject probe validation requires at least two rows per subject.")
        count = min(len(rows) - 1, max(1, int(round(fraction * len(rows)))))
        validation[rng.choice(rows, size=count, replace=False)] = True
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout-subjects", default="", help="comma separated; never pretrain on these")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--min-block", type=int, default=12)
    parser.add_argument("--max-block", type=int, default=32)
    parser.add_argument("--bands", default="1-4,4-8,8-13,13-30")
    parser.add_argument("--waveform-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0)
    parser.add_argument("--band-weight-estimation-samples", type=int, default=4096)
    parser.add_argument(
        "--subject-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train a stop-gradient subject-identity probe and record leakage accuracy.",
    )
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument(
        "--pooling-mode",
        choices=sorted(POOLING_MODES - {"latency_marginal_contrast"}),
        default=DEFAULT_N2P3_POOLING_MODE,
        help="Readout geometry retained in the transfer checkpoint.",
    )
    parser.add_argument(
        "--temporal-kernel-size",
        type=int,
        default=DEFAULT_N2P3_ARCHITECTURE.temporal_kernel_size,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cohort",
        choices=("default", "p300_causal", "gtn", "gtn_paper"),
        default="default",
        help=(
            "Causal contract family asserted for forward-phase source caches: "
            "'gtn' enforces the revised 0.1 Hz / 1200 ms child-cohort contract; "
            "'default' enforces 2 Hz / 800 ms."
        ),
    )
    parser.add_argument(
        "--tmax-ms",
        type=float,
        default=None,
        help="Explicit epoch-end recipe override for matched factorials.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.source_cache, require_labels=False, validation="attested")
    source_cache_sha256 = str(read_epoch_cache_attestation(args.source_cache)["sha256"])
    expected_contract = {
        "default": DEFAULT_P300_DATA_CONTRACT,
        "p300_causal": SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        "gtn": GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
        "gtn_paper": PAPER_GTN_CAUSAL_DATA_CONTRACT,
    }[args.cohort]
    if args.tmax_ms is not None:
        expected_contract = replace(expected_contract, tmax_ms=float(args.tmax_ms))
    assert_p300_input_contract(dataset.preprocessing, expected_contract)
    holdout = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    source_rows, all_subjects = _source_training_rows(dataset.subject_ids, holdout)
    if not source_rows.any():
        raise ValueError("holdout subjects removed every source epoch.")

    X = np.asarray(dataset.X[source_rows], dtype=np.float32)
    source_subjects = np.asarray(dataset.subject_ids).astype(str)[source_rows]
    probe_subjects = tuple(sorted(np.unique(source_subjects).tolist()))
    probe_index = {subject: index for index, subject in enumerate(probe_subjects)}
    probe_labels = np.asarray([probe_index[subject] for subject in source_subjects], dtype=np.int64)
    probe_validation = (
        _subject_probe_validation_mask(source_subjects, seed=args.seed + 31_337)
        if args.subject_probe
        else np.zeros(len(source_subjects), dtype=bool)
    )
    # One integer carries both the class and fixed probe split through MatrixBatchSource.
    encoded_probe_labels = probe_labels * 2 + probe_validation.astype(np.int64)
    if args.standardize:
        mean = X.reshape(X.shape[0], X.shape[1], -1).mean(axis=(0, 2), keepdims=True)
        std = X.reshape(X.shape[0], X.shape[1], -1).std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        X = ((X - mean) / std).astype(np.float32)

    trunk = N2P3Net(
        dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode=args.pooling_mode,
        **N2P3ArchitectureConfig(
            temporal_kernel_size=args.temporal_kernel_size
        ).model_kwargs(),
    ).to(device)
    config = PretrainingConfig(
        mask=MaskingConfig(
            mask_fraction=args.mask_fraction,
            min_block_samples=args.min_block,
            max_block_samples=args.max_block,
        ),
        loss=ReconstructionLossConfig(
            waveform_weight=args.waveform_weight,
            spectral_weight=args.spectral_weight,
            bands_hz=_parse_bands(args.bands),
        ),
        seed=args.seed,
        band_weight_estimation_samples=args.band_weight_estimation_samples,
        subject_probe_subjects=(len(probe_subjects) if args.subject_probe else 0),
    )
    task = PretrainingTask(trunk, config).to(device)
    reconstruction_parameters = [*task.trunk.parameters(), *task.decoder.parameters()]
    optimizer = torch.optim.AdamW(
        reconstruction_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    probe_optimizer = (
        torch.optim.AdamW(task.probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if task.probe is not None
        else None
    )
    rng = np.random.default_rng(args.seed)
    tensor = torch.from_numpy(np.ascontiguousarray(X))
    runtime = GpuPerformanceScheduler(device, precision="fp32")
    history: list[dict[str, float]] = []

    with runtime.lease():
        preload = runtime.can_preload(tensor.numel() * tensor.element_size())
        source = MatrixBatchSource(
            tensor,
            torch.from_numpy(encoded_probe_labels) if task.probe is not None else None,
            runtime,
            preload=preload,
        )
        probe_validation_source = (
            MatrixBatchSource(
                tensor[torch.from_numpy(np.flatnonzero(probe_validation))],
                torch.from_numpy(encoded_probe_labels[probe_validation]),
                runtime,
                preload=preload,
            )
            if task.probe is not None
            else None
        )
        weight_count = min(config.band_weight_estimation_samples, len(tensor))
        weight_rng = np.random.default_rng(args.seed + 97_531)
        weight_rows = torch.from_numpy(
            np.ascontiguousarray(
                weight_rng.choice(len(tensor), size=weight_count, replace=False),
                dtype=np.int64,
            )
        )
        if source.preloaded:
            assert source.device_X is not None
            weight_input = source.device_X.index_select(0, weight_rows.to(runtime.device))
        else:
            weight_input = runtime.to_device(source.cpu_X.index_select(0, weight_rows))
        task.update_band_weights(weight_input)
        del weight_input, weight_rows
        indices_device = runtime.device if source.preloaded else torch.device("cpu")
        for epoch in range(1, args.epochs + 1):
            permutation = rng.permutation(len(tensor))
            indices = torch.as_tensor(permutation, device=indices_device)
            epoch_loss = torch.zeros((), dtype=torch.float32, device=device)
            epoch_wave = torch.zeros((), dtype=torch.float32, device=device)
            epoch_spec = torch.zeros((), dtype=torch.float32, device=device)
            epoch_probe = torch.zeros((), dtype=torch.float32, device=device)
            epoch_probe_correct = torch.zeros((), dtype=torch.int64, device=device)
            epoch_probe_rows = 0
            for batch, (x, subject_batch) in enumerate(
                source.batches(args.batch_size, indices=indices)
            ):
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed * 1_000_003 + epoch * 10_001 + batch
                )
                components = task.loss_components(
                    x,
                    generator=generator,
                )
                loss = components["total"]
                optimizer.zero_grad(set_to_none=True)
                if probe_optimizer is not None:
                    probe_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if probe_optimizer is not None:
                    assert subject_batch is not None
                    probe_train = subject_batch.remainder(2) == 0
                    probe_components = None
                    if bool(probe_train.any()):
                        probe_targets = torch.div(
                            subject_batch[probe_train], 2, rounding_mode="floor"
                        )
                        probe_components = task.subject_probe_components(
                            x[probe_train], probe_targets
                        )
                        probe_components["loss"].backward()
                torch.nn.utils.clip_grad_norm_(reconstruction_parameters, 1.0)
                optimizer.step()
                if probe_optimizer is not None and probe_components is not None:
                    probe_optimizer.step()
                epoch_loss += loss.detach().float() * len(x)
                epoch_wave += components["waveform"].detach().float() * len(x)
                epoch_spec += components["spectral"].detach().float() * len(x)
                if probe_optimizer is not None:
                    epoch_probe += (
                        probe_components["loss"].detach().float() * int(probe_train.sum())
                    )
                    epoch_probe_correct += probe_components["correct"].detach()
                    epoch_probe_rows += int(probe_train.sum())
            record = {
                "epoch": epoch,
                "total": float(epoch_loss) / len(tensor),
                "waveform": float(epoch_wave) / len(tensor),
                "spectral": float(epoch_spec) / len(tensor),
            }
            if probe_optimizer is not None:
                task.trunk.eval()
                task.probe.eval()
                validation_loss = torch.zeros((), dtype=torch.float32, device=device)
                validation_correct = torch.zeros((), dtype=torch.int64, device=device)
                validation_rows = int(probe_validation.sum())
                assert probe_validation_source is not None
                with torch.inference_mode():
                    for validation_x, validation_subjects in probe_validation_source.batches(
                        args.batch_size
                    ):
                        assert validation_subjects is not None
                        validation_targets = torch.div(
                            validation_subjects, 2, rounding_mode="floor"
                        )
                        probe_validation_components = task.subject_probe_components(
                            validation_x, validation_targets
                        )
                        validation_loss += (
                            probe_validation_components["loss"].float()
                            * len(validation_x)
                        )
                        validation_correct += probe_validation_components["correct"]
                task.trunk.train()
                task.probe.train()
                record.update(
                    {
                        "subject_probe_train_loss": float(epoch_probe) / epoch_probe_rows,
                        "subject_probe_train_accuracy": (
                            float(epoch_probe_correct) / epoch_probe_rows
                        ),
                        "subject_probe_validation_loss": (
                            float(validation_loss) / validation_rows
                        ),
                        "subject_probe_validation_accuracy": (
                            float(validation_correct) / validation_rows
                        ),
                    }
                )
            history.append(record)
            print(json.dumps(history[-1]), flush=True)

    probe_enabled = probe_optimizer is not None
    probe_audit = {
        "enabled": probe_enabled,
        "stop_gradient": True,
        "n_subjects": len(probe_subjects) if probe_enabled else 0,
        "chance_accuracy": 1.0 / len(probe_subjects) if probe_enabled else None,
        "validation_fraction": 0.2 if probe_enabled else None,
        "training_rows": int((~probe_validation).sum()) if probe_enabled else 0,
        "validation_rows": int(probe_validation.sum()) if probe_enabled else 0,
        "final_train_loss": history[-1].get("subject_probe_train_loss"),
        "final_train_accuracy": history[-1].get("subject_probe_train_accuracy"),
        "final_validation_loss": history[-1].get("subject_probe_validation_loss"),
        "final_validation_accuracy": history[-1].get(
            "subject_probe_validation_accuracy"
        ),
    }
    task.discard_decoder()
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trunk_state_dict": trunk.state_dict(),
        "config": asdict(config),
        "source_cache": str(Path(args.source_cache).resolve()),
        "holdout_subjects": sorted(holdout),
        "source_dataset_name": dataset.name,
        "n_channels": int(dataset.n_channels),
        "n_times": int(dataset.n_times),
        "input_sample_rate_hz": float(dataset.preprocessing.sfreq),
        "input_tmin_s": float(dataset.preprocessing.tmin_ms) / 1000.0,
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_channel_names": list(dataset.channel_names),
        "input_source_reference": dataset.provenance.get("source_reference"),
        "source_cache_sha256": source_cache_sha256,
        "architecture": trunk.architecture_record(),
        "classifier_trained": False,
        "subject_probe_audit": probe_audit,
        "model_config": {
            "pooling_mode": args.pooling_mode,
            "temporal_kernel_size": args.temporal_kernel_size,
        },
        "source_subjects": sorted(all_subjects),
        "training_subjects": sorted(all_subjects - holdout),
        "training_subject_keys": [
            f"{dataset.name}\0{subject}" for subject in sorted(all_subjects - holdout)
        ],
        "training_cache_subject_keys": [
            f"{source_cache_sha256}\0{subject}"
            for subject in sorted(all_subjects - holdout)
        ],
        "n_source_epochs": int(source_rows.sum()),
        "standardized": args.standardize,
        "input_mean": (
            np.asarray(mean, dtype=np.float32).squeeze().tolist()
            if args.standardize
            else np.zeros(dataset.n_channels, dtype=np.float32).tolist()
        ),
        "input_std": (
            np.asarray(std, dtype=np.float32).squeeze().tolist()
            if args.standardize
            else np.ones(dataset.n_channels, dtype=np.float32).tolist()
        ),
        "runtime": {
            "device": str(device),
            "preloaded": source.preloaded,
            "memory": runtime.memory_record(),
        },
    }
    torch.save(payload, checkpoint)
    print(f"[pretrained] {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
