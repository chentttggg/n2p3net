"""Class-conditional feature alignment for multi-source P300 pretraining.

The 2026-09-02 alignment audit requires that any distribution constraint be
conditional rather than marginal: aligning only ``P(Z)`` across domains can
mix target and non-target trials and destroy the P300 evidence. During
multi-source supervised pretraining every source row carries a real trial
label, so the class-conditional means

    mu_{d,c} = E[H' | domain=d, y=c]

are estimable without pseudo-labels. The loss is the mean squared distance
between per-class mean feature maps over all unordered domain pairs,

    L_align = mean_{(d1,d2), c} || mu_{d1,c} - mu_{d2,c} ||^2 / (C_f * T_f),

computed on the adapted features ``H'`` (post-adapter, pre-pooling) so the
constraint targets exactly the representation the classifier consumes. This
is the first-order (linear-kernel conditional MMD) term; second-order CORAL
on the structured map is left as a preregistered ablation.

Cells with fewer than ``min_cell_count`` rows are skipped rather than
contributing noisy means; the diagnostic record exposes how many terms were
active so a silently-degenerate alignment weight is detectable in ledgers.
"""

from __future__ import annotations

import torch

DEFAULT_ALIGNMENT_MIN_CELL_COUNT = 8
DEFAULT_MMD_MIN_CELL_COUNT = 8


def _flatten_samples(features: torch.Tensor, *, name: str) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError(f"{name} must have shape (N, ...), got {tuple(features.shape)}.")
    if features.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    flattened = features.float().reshape(features.shape[0], -1)
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    return flattened


def _median_bandwidth(source: torch.Tensor, target: torch.Tensor) -> float:
    combined = torch.cat((source.detach(), target.detach()), dim=0)
    distances = torch.pdist(combined, p=2)
    positive = distances[distances > 0.0]
    if positive.numel() == 0:
        return 1.0
    median = float(torch.median(positive).item())
    return median if median > 0.0 else 1.0


def rbf_mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return an unbiased RBF-MMD estimate and frozen-bandwidth diagnostics.

    The bandwidth is computed from detached samples with the median heuristic
    when omitted, so the heuristic itself does not create a gradient path.
    ``source`` and ``target`` may be structured feature maps; axis zero is the
    sample axis.
    """

    source_flat = _flatten_samples(source, name="source")
    target_flat = _flatten_samples(target, name="target")
    if source_flat.device != target_flat.device:
        raise ValueError("source and target must be on the same device.")
    if bandwidth is None:
        bandwidth = _median_bandwidth(source_flat, target_flat)
    if not isinstance(bandwidth, (int, float)) or not torch.isfinite(
        torch.tensor(float(bandwidth))
    ) or float(bandwidth) <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")
    sigma = float(bandwidth)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(left, right, p=2).square()
        return torch.exp(-distances / (2.0 * sigma * sigma))

    source_kernel = kernel(source_flat, source_flat)
    target_kernel = kernel(target_flat, target_flat)
    cross_kernel = kernel(source_flat, target_flat)
    n_source = source_flat.shape[0]
    n_target = target_flat.shape[0]
    source_term = (source_kernel.sum() - torch.diagonal(source_kernel).sum()) / (
        n_source * (n_source - 1)
    )
    target_term = (target_kernel.sum() - torch.diagonal(target_kernel).sum()) / (
        n_target * (n_target - 1)
    )
    loss = source_term + target_term - 2.0 * cross_kernel.mean()
    return loss, {
        "bandwidth": sigma,
        "source_count": float(n_source),
        "target_count": float(n_target),
        "mmd_squared": float(loss.detach()),
    }


def conditional_rbf_mmd_loss(
    source: torch.Tensor,
    source_labels: torch.Tensor,
    target: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    min_cell_count: int = DEFAULT_MMD_MIN_CELL_COUNT,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Align source and target feature distributions separately for labels 0/1."""

    if min_cell_count < 2:
        raise ValueError("min_cell_count must be at least two for unbiased MMD.")
    source_flat = _flatten_samples(source, name="source")
    target_flat = _flatten_samples(target, name="target")
    if source_labels.ndim != 1 or source_labels.shape[0] != source_flat.shape[0]:
        raise ValueError("source_labels must align with source samples.")
    if target_labels.ndim != 1 or target_labels.shape[0] != target_flat.shape[0]:
        raise ValueError("target_labels must align with target samples.")
    source_labels = source_labels.to(device=source_flat.device, dtype=torch.long)
    target_labels = target_labels.to(device=target_flat.device, dtype=torch.long)
    if bool(((source_labels < 0) | (source_labels > 1)).any()) or bool(
        ((target_labels < 0) | (target_labels > 1)).any()
    ):
        raise ValueError("conditional MMD labels must be encoded as 0/1.")

    losses: list[torch.Tensor] = []
    active_labels: list[int] = []
    bandwidths: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        source_rows = source_labels == label
        target_rows = target_labels == label
        source_count = int(source_rows.sum())
        target_count = int(target_rows.sum())
        counts[str(label)] = {"source": source_count, "target": target_count}
        if source_count < min_cell_count or target_count < min_cell_count:
            continue
        loss, diagnostics = rbf_mmd_loss(
            source_flat[source_rows],
            target_flat[target_rows],
            bandwidth=bandwidth,
        )
        losses.append(loss)
        active_labels.append(label)
        bandwidths[str(label)] = diagnostics["bandwidth"]
    if losses:
        output = torch.stack(losses).mean()
    else:
        output = source_flat.new_zeros(())
    return output, {
        "active_labels": tuple(active_labels),
        "active_terms": float(len(losses)),
        "counts": counts,
        "bandwidths": bandwidths,
        "mmd_squared": float(output.detach()),
        "alignment_inactive": not bool(losses),
    }


def conditional_domain_rbf_mmd_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    domain_indices: torch.Tensor,
    domain_count: int,
    *,
    min_cell_count: int = DEFAULT_MMD_MIN_CELL_COUNT,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply conditional RBF-MMD to every eligible source-domain pair."""

    if domain_count < 2:
        raise ValueError("conditional domain MMD requires at least two domains.")
    features_flat = _flatten_samples(features, name="features")
    if labels.ndim != 1 or labels.shape[0] != features_flat.shape[0]:
        raise ValueError("labels must align with features.")
    if domain_indices.ndim != 1 or domain_indices.shape[0] != features_flat.shape[0]:
        raise ValueError("domain_indices must align with features.")
    labels = labels.to(device=features_flat.device, dtype=torch.long)
    domains = domain_indices.to(device=features_flat.device, dtype=torch.long)
    if bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError("conditional domain MMD labels must be encoded as 0/1.")
    if bool(((domains < 0) | (domains >= domain_count)).any()):
        raise ValueError("domain_indices must lie in [0, domain_count).")

    losses: list[torch.Tensor] = []
    active_pairs: list[tuple[int, int, int]] = []
    bandwidths: dict[str, float] = {}
    for first in range(domain_count):
        for second in range(first + 1, domain_count):
            for label in (0, 1):
                left = features_flat[(domains == first) & (labels == label)]
                right = features_flat[(domains == second) & (labels == label)]
                if len(left) < min_cell_count or len(right) < min_cell_count:
                    continue
                loss, diagnostics = rbf_mmd_loss(left, right, bandwidth=bandwidth)
                losses.append(loss)
                active_pairs.append((first, second, label))
                bandwidths[f"{first}:{second}:{label}"] = diagnostics["bandwidth"]
    output = torch.stack(losses).mean() if losses else features_flat.new_zeros(())
    return output, {
        "active_pairs": tuple(active_pairs),
        "active_terms": float(len(losses)),
        "bandwidths": bandwidths,
        "mmd_squared": float(output.detach()),
        "alignment_inactive": not bool(losses),
    }


def adaptive_target_weight(
    source_count: int,
    target_count: int,
    epoch: int,
    *,
    warmup_epochs: int,
    minimum: float = 1.0,
    maximum: float = 6.0,
) -> float:
    """Return the AS-MMD target weight without consulting held-out truth."""

    if source_count < 1 or target_count < 1:
        raise ValueError("source_count and target_count must be positive.")
    if epoch < 0 or warmup_epochs < 1:
        raise ValueError("epoch must be non-negative and warmup_epochs must be positive.")
    if not (0.0 < minimum <= maximum) or not torch.isfinite(
        torch.tensor([minimum, maximum], dtype=torch.float64)
    ).all():
        raise ValueError("minimum and maximum must be finite with 0 < minimum <= maximum.")
    base = min(max((source_count / target_count) ** 0.5, minimum), maximum)
    alpha = min(1.0, epoch / warmup_epochs)
    return float(1.0 + alpha * (base - 1.0))


def conditional_mean_alignment_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    domain_indices: torch.Tensor,
    domain_count: int,
    *,
    min_cell_count: int = DEFAULT_ALIGNMENT_MIN_CELL_COUNT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the class-conditional mean-alignment loss and its diagnostics.

    Parameters
    ----------
    features : torch.Tensor
        Adapted trunk features ``(B, C_f, T_f)``.
    labels : torch.Tensor
        Binary trial labels ``(B,)``.
    domain_indices : torch.Tensor
        Domain index per row ``(B,)``, values in ``[0, domain_count)``.
    domain_count : int
        Number of source domains participating in this batch.
    min_cell_count : int
        Minimum rows a (domain, class) cell needs to enter the loss.

    Returns
    -------
    loss, diagnostics
        ``loss`` is a scalar tensor (zero when no cell pair is active).
        Diagnostics report active-term counts and the detached mean/max
        squared distance for ledger recording.
    """

    if features.ndim != 3:
        raise ValueError(f"Expected features (B,C,T), got {tuple(features.shape)}.")
    batch = features.shape[0]
    if labels.ndim != 1 or labels.shape[0] != batch:
        raise ValueError("labels must align with the feature batch.")
    if domain_indices.ndim != 1 or domain_indices.shape[0] != batch:
        raise ValueError("domain_indices must align with the feature batch.")
    if domain_count < 1:
        raise ValueError("domain_count must be positive.")
    if min_cell_count < 1:
        raise ValueError("min_cell_count must be positive.")
    if batch:
        if bool((domain_indices < 0).any()) or bool((domain_indices >= domain_count).any()):
            raise ValueError("domain_indices must lie in [0, domain_count).")

    # float32 statistics regardless of autocast precision: the mean maps are
    # averaged over hundreds of rows and bf16 accumulation would silently
    # bias the pairwise distances.
    stats = features.float()
    labels = labels.to(torch.long)
    domains = domain_indices.to(torch.long)
    element_count = features.shape[1] * features.shape[2]

    if batch == 0 or domain_count < 2:
        zero = stats.new_zeros(())
        return zero, {"active_terms": 0.0, "mean_squared_distance": 0.0, "max_squared_distance": 0.0}

    cell_means: dict[tuple[int, int], torch.Tensor] = {}
    cell_counts: dict[tuple[int, int], int] = {}
    for domain in range(domain_count):
        for label in (0, 1):
            rows = (domains == domain) & (labels == label)
            count = int(rows.sum())
            if count >= min_cell_count:
                cell_means[(domain, label)] = stats[rows].mean(dim=0)
                cell_counts[(domain, label)] = count

    squared_distances: list[torch.Tensor] = []
    domains_present = sorted({domain for domain, _ in cell_means})
    for first_index, first in enumerate(domains_present):
        for second in domains_present[first_index + 1 :]:
            for label in (0, 1):
                left = cell_means.get((first, label))
                right = cell_means.get((second, label))
                if left is None or right is None:
                    continue
                difference = left - right
                squared_distances.append(
                    (difference * difference).sum() / element_count
                )

    if not squared_distances:
        zero = stats.new_zeros(())
        return zero, {"active_terms": 0.0, "mean_squared_distance": 0.0, "max_squared_distance": 0.0}

    stacked = torch.stack(squared_distances)
    loss = stacked.mean()
    diagnostics = {
        "active_terms": float(len(squared_distances)),
        "mean_squared_distance": float(stacked.detach().mean()),
        "max_squared_distance": float(stacked.detach().max()),
    }
    return loss, diagnostics
