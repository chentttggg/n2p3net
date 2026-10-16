"""Measure the actual compact source/target paths on prepared synthetic arrays."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from data.candidate_task import MembershipDecision  # noqa: E402
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net  # noqa: E402
from models.parameter_groups import SourceParameterPrior  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(device, *, repeats, epochs, source_updates, compile_mode):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 6, 179)).astype(np.float32)
    mask = np.ones((200, 6), dtype=bool)
    mask[::2, 0] = False
    X[~mask] = 0
    y = np.arange(200) % 2
    codes = tuple(str(i) for i in range(9))
    group = MembershipDecision.direct(codes * 5, codes, repetitions=5,
        repetition_indices=np.repeat(np.arange(5), 9), evidence_indices=np.arange(45),
        event_ids=tuple(f"event-{i}" for i in range(45)))
    architecture = N2P3ArchitectureConfig.unified()
    torch.manual_seed(71)
    original = N2P3Net(6, n_times=179, **architecture.model_kwargs()).to(device)
    prior = SourceParameterPrior(original)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.add_(.01)

    def scalar_oracle():
        return torch.stack([torch.stack([
            (p.float() - r.float()).square().mean() / (2 * scale) for p, r, scale in members
        ]).mean() for members in prior.groups.values()]).sum()

    actual, expected = prior.loss(), scalar_oracle()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    left = torch.autograd.grad(actual, tuple(original.parameters()))
    right = torch.autograd.grad(expected, tuple(original.parameters()))
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)

    metrics = {}
    for name, fn in (("packed_prior", prior.loss), ("tensorwise_prior_oracle", scalar_oracle)):
        samples = []
        for _ in range(repeats + 1):
            synchronize(device)
            start = time.perf_counter()
            for _ in range(100):
                original.zero_grad(set_to_none=True)
                fn().backward()
            synchronize(device)
            samples.append((time.perf_counter() - start) / 100)
        metrics[name] = {"cold_seconds": samples[0], "warm_seconds": samples[1:],
                         "median_warm_seconds": statistics.median(samples[1:])}

    # Compare both implementations inside the same full fitting path as well.
    packed_loss = SourceParameterPrior.loss
    for name, joint, tensorwise in (("Q0", False, False), ("Q1", True, False),
                                    ("Q1_tensorwise_oracle", True, True)):
        samples = []
        labels = y.copy()
        labels[:45] = group.membership[:, 0]
        config = SubjectAdapterConfig(head_kind="full_fine", trainable_weights="non_bn",
            batchnorm_affine="trainable", batchnorm_buffers="calibration", input_statistics="target_prefix",
            dropout_mode="off", epochs=epochs, batch_size=512, val_group_fraction=None,
            refit_full_prefix=False, source_prior_weight=.01 if joint else 0.,
            candidate_loss_weight=1. if joint else 0., seed=71)
        try:
            if tensorwise:
                SourceParameterPrior.loss = lambda self: torch.stack([torch.stack([
                    (p.float() - r.float()).square().mean() / (2 * scale) for p, r, scale in members
                ]).mean() for members in self.groups.values()]).sum()
            for repetition in range(repeats + 1):
                synchronize(device)
                start = time.perf_counter()
                adapter = SubjectAdapter(copy.deepcopy(original).cpu(), device=device, config=config)
                adapter.fit(X, labels, trial_channel_mask=mask, decision_groups=[group], decision_targets=[0])
                scores = adapter.predict_logit(X, trial_channel_mask=mask)
                synchronize(device)
                samples.append(time.perf_counter() - start)
                if not np.isfinite(scores).all():
                    raise ValueError("nonfinite benchmark predictions")
                if repetition == repeats:
                    history = adapter.last_history
                del adapter
        finally:
            SourceParameterPrior.loss = packed_loss
        metrics[name] = {"cold_seconds": samples[0], "warm_seconds": samples[1:],
            "median_warm_seconds": statistics.median(samples[1:]), "fit_budget": history["fit_budget"]}
    for name, objective in (("A0", None), ("A1", {"supcon_weight": .05, "mmd_weight": .05,
                                               "feature_location": "post_residual"})):
        samples, runtime = [], {}
        for _ in range(repeats + 1):
            synchronize(device)
            start = time.perf_counter()
            source = N2P3NetBaseline(6, 179, 128., device=device, architecture=architecture,
                config=DeepConfig(batch_size=512, fixed_updates=source_updates, sampling="uniform_rows",
                    compile_mode=compile_mode, source_objective=objective, seed=71))
            source.fit(X, y, trial_channel_mask=mask,
                source_domain_ids=np.asarray([f"domain-{i % 3}" for i in range(200)]),
                risk_unit_ids=np.asarray([f"person-{i % 12}" for i in range(200)]))
            synchronize(device)
            samples.append(time.perf_counter() - start)
            runtime = source.last_runtime
            del source
        metrics[name] = {"cold_seconds": samples[0], "warm_seconds": samples[1:],
                        "median_warm_seconds": statistics.median(samples[1:]), "runtime": runtime}
    return {"schema": "n2p3_unified_training_benchmark/1", "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": torch.__version__, "batch_cap": 512, "samples": 200, "target_epochs": epochs,
            "source_updates": source_updates, "repeats": repeats, "compile_mode": compile_mode,
            "prior_value_and_gradient_verified": True, "measurements": metrics,
            "scope": "Synthetic prepared arrays; includes fitting, input statistics, BN and target prediction. "
                     "Excludes raw preparation, cloud transfers and multi-worker throughput; no efficacy claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source-updates", type=int, default=30)
    parser.add_argument("--compile-mode", choices=("none", "default", "reduce-overhead"), default="none")
    args = parser.parse_args()
    if min(args.repeats, args.epochs, args.source_updates) < 1:
        parser.error("all work counts must be positive")
    if args.output.exists():
        parser.error("choose a new benchmark output")
    with cpu_thread_budget(1):
        record = benchmark(torch.device(args.device), repeats=args.repeats, epochs=args.epochs,
                           source_updates=args.source_updates,
                           compile_mode=None if args.compile_mode == "none" else args.compile_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seconds": {name: item["median_warm_seconds"]
                    for name, item in record["measurements"].items()}}))


if __name__ == "__main__":
    main()
