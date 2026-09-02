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
