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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.epochs import load_epoch_dataset  # noqa: E402
from models.n2p3net import N2P3Net  # noqa: E402
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
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.source_cache, require_labels=False, validation="attested")
    if dataset.preprocessing.filter_phase == "forward":
        from data.contract import assert_causal_p300_input_contract

        assert_causal_p300_input_contract(dataset.preprocessing)
    holdout = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    source_rows = ~np.isin(dataset.subject_ids.astype(str), list(holdout))
    if not source_rows.any():
        raise ValueError("holdout subjects removed every source epoch.")

    X = np.asarray(dataset.X[source_rows], dtype=np.float32)
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
        pooling_mode="ms_flatten",
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
    )
    task = PretrainingTask(trunk, config).to(device)
    optimizer = torch.optim.AdamW(
        list(task.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    tensor = torch.from_numpy(np.ascontiguousarray(X))
    runtime = GpuPerformanceScheduler(device, precision="fp32")
    history: list[dict[str, float]] = []

    with runtime.lease():
        preload = runtime.can_preload(tensor.numel() * tensor.element_size())
        source = MatrixBatchSource(tensor, None, runtime, preload=preload)
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
            for batch, (x, _) in enumerate(source.batches(args.batch_size, indices=indices)):
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed * 1_000_003 + epoch * 10_001 + batch
                )
                components = task.loss_components(x, generator=generator)
                loss = components["total"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.detach().float() * len(x)
                epoch_wave += components["waveform"].detach().float() * len(x)
                epoch_spec += components["spectral"].detach().float() * len(x)
            history.append(
                {
                    "epoch": epoch,
                    "total": float(epoch_loss) / len(tensor),
                    "waveform": float(epoch_wave) / len(tensor),
                    "spectral": float(epoch_spec) / len(tensor),
                }
            )
            print(json.dumps(history[-1]), flush=True)

    task.discard_decoder()
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trunk_state_dict": trunk.state_dict(),
        "config": asdict(config),
        "source_cache": str(Path(args.source_cache).resolve()),
        "holdout_subjects": sorted(holdout),
        "n_source_epochs": int(source_rows.sum()),
        "standardized": args.standardize,
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
