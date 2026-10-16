"""AS-MMD logit-level alignment from Chen & Delorme 2025.

Implements equations (1)-(9) of

    W. Chen and A. Delorme, "Adaptive Split-MMD Training for Small-Sample
    Cross-Dataset P300 EEG Classification", arXiv:2510.21969 (2025).

The module is self-contained: target weighting, the warm-up-scaled MMD
coefficient, and the unbiased Gretton RBF-MMD^2 estimator are all defined
here, without importing other transfer helpers.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _finite_float(value: object, name: str) -> float:
    """Convert *value* to a finite Python float or raise ``ValueError``."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _warmup_alpha(epoch: float, warmup_epochs: float) -> float:
    """Equation (1): alpha_e = min(1, e / W)."""
    return min(1.0, epoch / warmup_epochs)


def _validated_epoch(epoch: object) -> float:
    epoch_value = _finite_float(epoch, "epoch")
    if epoch_value < 0.0:
        raise ValueError("epoch must be non-negative.")
    return epoch_value


def _validated_target_weight_bounds(minimum: object, maximum: object) -> tuple[float, float]:
    minimum_value = _finite_float(minimum, "target_weight_min")
    maximum_value = _finite_float(maximum, "target_weight_max")
    if not 0.0 < minimum_value <= maximum_value:
        raise ValueError("target weight bounds must satisfy 0 < minimum <= maximum.")
    return minimum_value, maximum_value


def _validated_bandwidth(bandwidth: object) -> float:
    result = _finite_float(bandwidth, "bandwidth")
    if result <= 0.0:
        raise ValueError("bandwidth must be positive.")
    return result


def target_weight(
    epoch: float,
    source_count: float,
    target_count: float,
    warmup_epochs: float,
    minimum: float = 1.0,
    maximum: float = 6.0,
) -> float:
    """Return the AS-MMD target CE weight of equations (1)-(4).

    ``alpha_e = min(1, epoch / warmup_epochs)``, ``w_t0`` is the square
    root of ``source_count / target_count``, and ``w_t = 1 + alpha_e *
    (clip(w_t0, minimum, maximum) - 1)``.
    """

    epoch_value = _validated_epoch(epoch)
    source_value = _positive_float(source_count, "source_count")
    target_value = _positive_float(target_count, "target_count")
    warmup_value = _positive_float(warmup_epochs, "warmup_epochs")
    minimum_value, maximum_value = _validated_target_weight_bounds(minimum, maximum)

    alpha_e = _warmup_alpha(epoch_value, warmup_value)
    w_t0 = math.sqrt(source_value / target_value)
    w_tilde = min(max(w_t0, minimum_value), maximum_value)
    return float(1.0 + alpha_e * (w_tilde - 1.0))


def mmd_schedule(epoch: float, warmup_epochs: float, base_weight: float) -> float:
    """Return the AS-MMD MMD coefficient of equation (9).

    ``lambda_mmd(e) = alpha_e * base_weight``.
    """

    epoch_value = _validated_epoch(epoch)
    warmup_value = _positive_float(warmup_epochs, "warmup_epochs")
    base_value = _nonnegative_float(base_weight, "base_weight")

    return float(_warmup_alpha(epoch_value, warmup_value) * base_value)


def _prepare_logits(
    logits: torch.Tensor,
    *,
    name: str,
    min_samples: int,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not logits.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor.")
    if logits.dtype not in {torch.float32, torch.float64}:
        raise ValueError(f"{name} must be float32 or float64, got {logits.dtype}.")
    if logits.ndim not in {1, 2}:
        raise ValueError(f"{name} must have shape (N,) or (N, C), got {tuple(logits.shape)}.")
    if logits.shape[0] < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} samples.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    return logits


def _median_bandwidth(source: torch.Tensor, target: torch.Tensor) -> float:
    """Median heuristic over the detached combined logits.

    Pairwise Euclidean distances are computed with :func:`torch.cdist` and
    the median is taken over the distinct upper-triangular pairs (i < j) of
    the combined sample, i.e. Gretton's aggregate-sample median heuristic.
    If every distance is zero (or the median is zero), fall back to 1.0.
    """

    combined = torch.cat((source.detach(), target.detach()), dim=0)
    pairwise = torch.cdist(combined, combined, p=2.0)
    count = combined.shape[0]
    rows, columns = torch.triu_indices(count, count, offset=1, device=combined.device)
    distances = pairwise[rows, columns]
    median = float(torch.median(distances).detach().item())
    if median == 0.0 or not math.isfinite(median):
        return 1.0
    return median


def _rbf_kernel(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Equation (6): k(u, v) = exp(-||u - v||^2 / (2 sigma^2))."""
    squared_distances = torch.cdist(left, right, p=2.0).square()
    return torch.exp(-squared_distances / (2.0 * bandwidth * bandwidth))


def rbf_mmd_logits_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    bandwidth: float | None = None,
    min_samples: int = 2,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return the unbiased logit-level RBF-MMD^2 of equation (7).

    The bandwidth is frozen (detached) before the kernel is evaluated, so
    the median heuristic does not contribute a gradient path. Diagnostics
    are Python scalars detached from the autograd graph.
    """

    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("min_samples must be an integer of at least 2 for unbiased MMD.")

    source = _prepare_logits(source_logits, name="source_logits", min_samples=min_samples)
    target = _prepare_logits(target_logits, name="target_logits", min_samples=min_samples)

    if source.device != target.device:
        raise ValueError("source_logits and target_logits must be on the same device.")
    if source.dtype != target.dtype:
        raise ValueError("source_logits and target_logits must have the same dtype.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_logits and target_logits must have the same feature size.")

    if bandwidth is None:
        sigma = _median_bandwidth(source, target)
    else:
        sigma = _validated_bandwidth(bandwidth)

    source_kernel = _rbf_kernel(source, source, sigma)
    target_kernel = _rbf_kernel(target, target, sigma)
    cross_kernel = _rbf_kernel(source, target, sigma)

    source_count = source.shape[0]
    target_count = target.shape[0]
    source_term = (source_kernel.sum() - source_kernel.diagonal().sum()) / (
        source_count * (source_count - 1)
    )
    target_term = (target_kernel.sum() - target_kernel.diagonal().sum()) / (
        target_count * (target_count - 1)
    )
    # Equation (7): the cross term is (2 / (n * m)) * sum_{i,j} k(s_i, t_j).
    cross_term = (2.0 / (source_count * target_count)) * cross_kernel.sum()

    mmd_squared = source_term + target_term - cross_term
    detached = mmd_squared.detach()
    return mmd_squared, {
        "bandwidth": float(sigma),
        "source_count": float(source_count),
        "target_count": float(target_count),
        "mmd_squared": float(detached),
        "negative_mmd": bool(detached.item() < 0.0),
    }


def _validate_binary_labels(
    labels: torch.Tensor,
    *,
    name: str,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        try:
            labels = torch.as_tensor(labels)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise TypeError(f"{name} must be a torch.Tensor of 0/1 labels.") from exc
    if labels.is_complex():
        raise ValueError(f"{name} must be encoded as 0/1.")
    if labels.ndim != 1 or labels.shape[0] != count:
        raise ValueError(f"{name} must be a 1D tensor aligned with the {count} samples.")
    valid = (labels == 0) | (labels == 1)
    if not bool(valid.all()):
        raise ValueError(f"{name} must be encoded as 0/1.")
    return labels.to(device=device, dtype=torch.long)


def _prepare_joint_logits(logits: torch.Tensor, *, name: str) -> torch.Tensor:
    prepared = _prepare_logits(logits, name=name, min_samples=2)
    if prepared.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2) for binary cross-entropy.")
    return prepared


def asmmd_joint_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    epoch: float,
    warmup_epochs: float,
    base_weight: float,
    source_count: float,
    target_count: float,
    target_weight_min: float = 1.0,
    target_weight_max: float = 6.0,
    pos_weight: float = 1.0,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return the AS-MMD total loss of equations (1)-(9) plus diagnostics.

    The supervised terms use binary cross-entropy with class weights
    ``[1, pos_weight]``; the target CE is multiplied by ``target_weight``
    and the RBF-MMD^2 term by ``mmd_schedule``.
    """

    source = _prepare_joint_logits(source_logits, name="source_logits")
    target = _prepare_joint_logits(target_logits, name="target_logits")
    if source.device != target.device:
        raise ValueError("source_logits and target_logits must be on the same device.")
    if source.dtype != target.dtype:
        raise ValueError("source_logits and target_logits must have the same dtype.")

    source_labels_ce = _validate_binary_labels(
        source_labels,
        name="source_labels",
        count=source.shape[0],
        device=source.device,
    )
    target_labels_ce = _validate_binary_labels(
        target_labels,
        name="target_labels",
        count=target.shape[0],
        device=target.device,
    )

    epoch_value = _validated_epoch(epoch)
    warmup_value = _positive_float(warmup_epochs, "warmup_epochs")
    base_value = _nonnegative_float(base_weight, "base_weight")
    source_count_value = _positive_float(source_count, "source_count")
    target_count_value = _positive_float(target_count, "target_count")
    _validated_target_weight_bounds(target_weight_min, target_weight_max)
    pos_weight_value = _nonnegative_float(pos_weight, "pos_weight")

    target_w = target_weight(
        epoch_value,
        source_count_value,
        target_count_value,
        warmup_value,
        minimum=target_weight_min,
        maximum=target_weight_max,
    )
    lambda_mmd = mmd_schedule(epoch_value, warmup_value, base_value)

    source_ce_weight = torch.tensor(
        [1.0, pos_weight_value], dtype=source.dtype, device=source.device
    )
    target_ce_weight = torch.tensor(
        [1.0, pos_weight_value], dtype=target.dtype, device=target.device
    )
    source_ce = F.cross_entropy(source_logits, source_labels_ce, weight=source_ce_weight)
    target_ce = F.cross_entropy(target_logits, target_labels_ce, weight=target_ce_weight)

    mmd_squared, mmd_diagnostics = rbf_mmd_logits_loss(
        source_logits,
        target_logits,
        bandwidth=bandwidth,
        min_samples=2,
    )
    total = source_ce + target_w * target_ce + lambda_mmd * mmd_squared

    return total, {
        "source_ce": float(source_ce.detach()),
        "target_ce": float(target_ce.detach()),
        "target_w": target_w,
        "target_weight": target_w,
        "lambda_mmd": lambda_mmd,
        "mmd_squared": mmd_diagnostics["mmd_squared"],
        "negative_mmd": mmd_diagnostics["negative_mmd"],
        "bandwidth": mmd_diagnostics["bandwidth"],
        "epoch": epoch_value,
        "source_count": source_count_value,
        "target_count": target_count_value,
        "batch_source_count": mmd_diagnostics["source_count"],
        "batch_target_count": mmd_diagnostics["target_count"],
    }


__all__ = [
    "asmmd_joint_loss",
    "mmd_schedule",
    "rbf_mmd_logits_loss",
    "target_weight",
]
