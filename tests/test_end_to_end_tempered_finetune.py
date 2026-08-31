from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.run_end_to_end_tempered_finetune import (
    EndToEndTemperedDecision,
    GroupSpec,
    dense_evidence,
    fit_end_to_end,
    iter_group_batches,
    pack_groups,
)


def _groups(n_groups: int = 3) -> tuple[GroupSpec, ...]:
    groups = []
    row = 0
    for group_index in range(n_groups):
        rows = np.arange(row, row + 18, dtype=np.int64)
        candidates = np.repeat(np.arange(9, dtype=np.int64), 2)
        groups.append(
            GroupSpec(
                group_id=f"g{group_index}",
                subject_id=f"s{group_index}",
                target_index=group_index % 9,
                epoch_rows=rows,
                candidate_indices=candidates,
                occurrence_slots=np.tile(np.arange(2, dtype=np.int64), 9),
                occurrence_fraction=np.tile(np.asarray([0.0, 1.0], dtype=np.float32), 9),
            )
        )
        row += 18
    return tuple(groups)


def test_packed_batches_visit_every_trial_once() -> None:
    groups = _groups()
    batches = list(
        iter_group_batches(
            groups,
            groups_per_batch=2,
            rng=np.random.default_rng(17),
        )
    )

    visited = np.concatenate([batch.local_rows for batch in batches])
    assert sorted(visited.tolist()) == list(range(54))
    assert sum(len(batch.targets) for batch in batches) == 3


def test_listwise_gradient_reaches_trial_evidence_and_decision_parameters() -> None:
    packed = pack_groups(_groups(2))
    evidence = torch.linspace(-1.0, 1.0, len(packed.local_rows), requires_grad=True)
    values, mask, occurrence = dense_evidence(
        evidence,
        packed,
        device=torch.device("cpu"),
    )
    decision = EndToEndTemperedDecision(
        abs_center=0.0,
        abs_scale=1.0,
        score_scale=1.0,
    )

    scores = decision(values, mask, occurrence)
    loss = F.cross_entropy(scores, torch.as_tensor(packed.targets))
    loss.backward()

    assert evidence.grad is not None
    assert torch.count_nonzero(evidence.grad).item() > 0
    assert all(parameter.grad is not None for parameter in decision.parameters())


def test_end_to_end_fit_updates_backbone_and_consumes_all_rows() -> None:
    torch.manual_seed(4)
    groups = _groups(2)
    X = torch.randn(36, 4)
    y = torch.zeros(36, dtype=torch.long)
    for group in groups:
        y[group.epoch_rows[group.candidate_indices == group.target_index]] = 1
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    decision = EndToEndTemperedDecision(
        abs_center=0.0,
        abs_scale=1.0,
        score_scale=1.0,
    )
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    history, runtime = fit_end_to_end(
        model,
        decision,
        X,
        y,
        groups,
        device=torch.device("cpu"),
        epochs=2,
        groups_per_batch=2,
        seed=9,
        backbone_learning_rate=1e-2,
        decision_learning_rate=1e-2,
        weight_decay=0.0,
        pos_weight=8.0,
        decision_loss_weight=1.0,
        compile_mode=None,
        fused_adam=False,
    )

    assert [row["trials"] for row in history] == [36, 36]
    assert runtime["optimizer_steps"] == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, model.parameters(), strict=True)
    )
