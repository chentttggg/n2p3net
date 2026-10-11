"""Independent architecture probes; no production edits or real-data training.

Use the repository virtual environment. JSON defaults to this script's directory;
--output selects another artifact path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.func import functional_call  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from data.contract import DEFAULT_P300_DATA_CONTRACT as CONTRACT  # noqa: E402
from models.adapters import (  # noqa: E402
    FeatureDomainTransfer,
    FeatureResidualAdapter,
    FeatureTransferConfig,
)
from models.n2p3net import N2P3Net  # noqa: E402
from transfer.recipes import p1_target_recipes  # noqa: E402
from transfer.subject_adapter import SubjectAdapter  # noqa: E402


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inspect_parameters():
    core = N2P3Net(
        6, n_times=CONTRACT.n_times, sfreq=CONTRACT.sample_rate_hz, tmin_s=CONTRACT.tmin_ms / 1000
    )
    wrapper = FeatureDomainTransfer(copy.deepcopy(core), config=FeatureTransferConfig())
    p1 = []
    for recipe in p1_target_recipes():
        if recipe.initialization == "waveform_lda":
            continue
        trunk = copy.deepcopy(wrapper if recipe.weights == "adapter" else core)
        adapter = SubjectAdapter(trunk, config=recipe.adapter_config())
        adapter._set_trainable(trunk_trainable=recipe.weights == "non_bn")
        p1.append(
            {
                "name": recipe.name,
                "total": parameter_count(trunk),
                "trainable": adapter.parameter_count(),
            }
        )
    result = {
        "counts": {
            "core6": parameter_count(core),
            "core7": parameter_count(
                N2P3Net(
                    7,
                    n_times=CONTRACT.n_times,
                    sfreq=CONTRACT.sample_rate_hz,
                    tmin_s=CONTRACT.tmin_ms / 1000,
                )
            ),
            "core_groups": {
                k: sum(p.numel() for p in v) for k, v in core.adaptation_parameter_groups().items()
            },
            "feature_shape": list(
                core.eval().forward_features(torch.randn(2, 6, CONTRACT.n_times)).shape
            ),
            "wrapper_total": parameter_count(wrapper),
            "wrapper_groups": {
                k: sum(p.numel() for p in v)
                for k, v in wrapper.adaptation_parameter_groups().items()
            },
            "minimal_residual_total": parameter_count(core)
            + parameter_count(wrapper.target_adapter),
            "film_pre_elu_total": parameter_count(core)
            + 2 * (core.spatial_features + len(core.mst_branches) * core.mst_features_per_scale),
            "p1_trainable": p1,
        }
    }
    return core, result


def check_film():
    result = {}
    # Last-layer FiLM exactly folds into a free linear readout.
    z = torch.randn(9, 4, 44, dtype=torch.float64)
    gamma = torch.randn(4, dtype=torch.float64)
    beta = torch.randn(4, dtype=torch.float64)
    w = torch.randn(2, 176, dtype=torch.float64)
    bias = torch.randn(2, dtype=torch.float64)
    a = F.linear((gamma[None, :, None] * z + beta[None, :, None]).flatten(1), w, bias)
    g = gamma.repeat_interleave(44)
    b = beta.repeat_interleave(44)
    c = F.linear(z.flatten(1), w * g[None], bias + w @ b)
    result["last_film_folding_max_abs"] = float((a - c).abs().max())
    # FiLM before training BN: shift disappears, positive scale mostly cancels.
    h = torch.randn(12, 4, 44, dtype=torch.float64)
    scale = torch.tensor([0.3, 2.0, 4.0, 1.5])[None, :, None]
    shift = torch.tensor([7.0, -3.0, 2.0, 8.0])[None, :, None]
    result["film_before_bn_max_abs"] = {
        str(eps): float(
            (
                F.batch_norm(h, None, None, training=True, eps=eps)
                - F.batch_norm(scale * h + shift, None, None, training=True, eps=eps)
            )
            .abs()
            .max()
        )
        for eps in [1e-12, 1e-5]
    }
    return result


def check_residual():
    result = {}
    # FeatureResidualAdapter identity and gradient startup.
    for dead in [False, True]:
        residual = FeatureResidualAdapter(
            4, bottleneck_channels=4, kernel_size=9, max_residual=0.25
        )
        if dead:
            nn.init.zeros_(residual.up.weight)
            nn.init.zeros_(residual.up.bias)
        x = torch.randn(7, 4, 44)
        coeff = torch.randn_like(x)
        out = residual(x)
        (out * coeff).sum().backward()
        result["residual_double_zero" if dead else "residual_default"] = {
            "identity_error": float((out - x).abs().max().detach()),
            "grad_norms": {name: float(p.grad.norm()) for name, p in residual.named_parameters()},
        }
    # Scalar bound on gate does not bound residual output amplitude.
    residual = FeatureResidualAdapter(4, bottleneck_channels=4, kernel_size=9, max_residual=0.25)
    with torch.no_grad():
        residual.gate.fill_(1.0)
        residual.up.bias.fill_(100.0)
    x = torch.ones(2, 4, 44)
    ratio = (residual(x) - x).norm() / x.norm()
    result["bounded_gate_residual_norm_ratio"] = float(ratio.detach())
    return result


def check_mask_and_nonlinearity():
    result = {}
    # Affine offsets revive a missing physical electrode unless mask is applied afterwards.
    mask = torch.tensor([1.0, 0.0])
    raw = torch.zeros(1, 2, 4)
    raw[:, 0] = 1.0
    offset = torch.tensor([0.2, 0.7])[None, :, None]
    encoded = raw + offset
    result["missing_channel_affine"] = {
        "without_post_affine_mask": encoded.sum(1).tolist(),
        "with_post_affine_mask": (encoded * mask[None, :, None]).sum(1).tolist(),
    }
    # Before a nonlinearity, shift need not be representable by the downstream affine head.
    x = torch.linspace(-3, 3, 25, dtype=torch.float64)
    base = F.elu(x)
    mod = F.elu(x + 1.0)
    design = torch.stack((base, torch.ones_like(base)), 1)
    fitted = design @ torch.linalg.lstsq(design, mod).solution
    result["pre_elu_shift_best_head_max_error"] = float((fitted - mod).abs().max())
    return result


def check_exact_meta(core: N2P3Net):
    result = {}
    # One synthetic exact-MAML step through the actual model with frozen BN buffers.
    m = copy.deepcopy(core).eval()
    params = dict(m.named_parameters())
    buffers = {k: v.clone() for k, v in m.named_buffers()}
    sx = torch.randn(8, 6, CONTRACT.n_times)
    qx = torch.randn(8, 6, CONTRACT.n_times)
    labels = torch.arange(8) % 2
    loss = F.cross_entropy(functional_call(m, (params, buffers), (sx,)), labels)
    grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
    fast = {k: p - 0.01 * g for (k, p), g in zip(params.items(), grads, strict=True)}
    query = F.cross_entropy(functional_call(m, (fast, buffers), (qx,)), labels)
    meta = torch.autograd.grad(query, tuple(params.values()))
    result["actual_encoder_exact_maml_one_step"] = {
        "query_finite": bool(torch.isfinite(query)),
        "all_meta_gradients_finite": all(bool(torch.isfinite(g).all()) for g in meta),
        "nonzero_groups": sum(float(g.norm()) > 0 for g in meta),
        "buffers_unchanged": all(torch.equal(buffers[k], v) for k, v in m.named_buffers()),
    }
    # Max-norm active, two exact inner updates: the functional projection must not mutate parameters.
    m = copy.deepcopy(core).eval()
    with torch.no_grad():
        m.st_spatial.weight.mul_(100.0)
    base_params = dict(m.named_parameters())
    base_before = {k: v.detach().clone() for k, v in base_params.items()}
    fast = base_params
    buffers = {k: v.clone() for k, v in m.named_buffers()}
    buffer_before = {k: v.clone() for k, v in buffers.items()}
    raw_norms = m.st_spatial.weight.flatten(1).norm(dim=1)
    for _ in range(2):
        support_loss = F.cross_entropy(functional_call(m, (fast, buffers), (sx,)), labels)
        inner_grad = torch.autograd.grad(support_loss, tuple(fast.values()), create_graph=True)
        fast = {k: p - 0.01 * g for (k, p), g in zip(fast.items(), inner_grad, strict=True)}
    query = F.cross_entropy(functional_call(m, (fast, buffers), (qx,)), labels)
    meta = torch.autograd.grad(query, tuple(base_params.values()))
    result["active_maxnorm_two_step_exact_meta"] = {
        "raw_spatial_norm_min": float(raw_norms.min().detach()),
        "effective_spatial_norm_max": float(
            m.st_spatial.effective_weight().flatten(1).norm(dim=1).max().detach()
        ),
        "all_meta_gradients_finite": all(bool(torch.isfinite(g).all()) for g in meta),
        "base_parameters_unchanged": all(
            torch.equal(base_before[k], v) for k, v in base_params.items()
        ),
        "model_parameters_unchanged": all(
            torch.equal(base_before[k], v) for k, v in m.named_parameters()
        ),
        "buffers_unchanged": all(torch.equal(buffer_before[k], v) for k, v in buffers.items()),
    }
    return result


def run_probe() -> dict:
    """Run the existing deterministic count, identity, gradient, and meta probes."""
    torch.set_num_threads(1)
    torch.manual_seed(123)
    core, result = inspect_parameters()
    result.update(check_film())
    result.update(check_residual())
    result.update(check_mask_and_nonlinearity())
    result.update(check_exact_meta(core))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args(argv)
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result_path": str(args.output.resolve()), "results": result}))


if __name__ == "__main__":
    main()
