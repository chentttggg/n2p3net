"""Historical mean-scale row/column candidate-ranking loss for target adaptation.

Trial-level cross-entropy does not see the evaluation object: the evaluator
aggregates evidence per row and per column, then takes the intersection of the
two argmax winners.  This module adds exactly that surrogate on top of
``DecisionLevelLoss``, so the frozen vocabulary, incomplete-coverage state and
count-neutral mean semantics stay in one shared implementation.

The row and column axes are treated as two independent candidate sets:

    L = L_row + L_col

    L_row = -log softmax_r( mean_{i:c_i=r} e_i / tau )_{r=r*}
    L_col = -log softmax_c( mean_{i:c_i=n_rows+c} e_i / tau )_{c=c*}

This is the direct differentiable surrogate of "row argmax x column argmax"
used by the operational evaluator.  It is not ``hit@R`` itself and remains an
additive term next to trial CE.

The unified sum method uses MembershipDecision.row_column and
transfer.decision_loss.membership_decision_loss. This wrapper retains the saved
mean temperature interpretation; its DecisionLevelLoss arithmetic delegates to
the common membership kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from transfer.decision_loss import DecisionLevelLoss


@dataclass(frozen=True)
class RowColumnCandidateLossDiagnostics:
    decision_count: int
    complete_count: int
    skipped_count: int
    row_nll: float | None
    column_nll: float | None

    def record(self) -> dict[str, object]:
        return {
            "decision_count": self.decision_count,
            "complete_count": self.complete_count,
            "skipped_count": self.skipped_count,
            "row_nll": self.row_nll,
            "column_nll": self.column_nll,
        }


def row_column_candidate_loss(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    selection_ids: torch.Tensor | Sequence[object],
    target_rows: torch.Tensor,
    target_columns: torch.Tensor,
    *,
    num_rows: int,
    num_columns: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, RowColumnCandidateLossDiagnostics]:
    """Return the mean complete-decision row/column ranking loss and its ledger.

    Parameters are aligned row tensors on the same device.  A decision is used
    only when every row and column candidate is observed at least once inside
    that decision; incomplete decisions are counted in ``skipped_count`` and do
    not contribute a silent zero gradient.
    """

    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("logits must be a one-dimensional floating tensor.")
    batch = logits.shape[0]
    for name, value in (
        ("candidate_ids", candidate_ids),
        ("target_rows", target_rows),
        ("target_columns", target_columns),
    ):
        if value.ndim != 1 or value.shape[0] != batch:
            raise ValueError(f"{name} must be a one-dimensional tensor aligned with logits.")
        if value.device != logits.device:
            raise ValueError(f"{name} must share the logits device.")
    if not torch.is_tensor(selection_ids):
        raw_selection_ids = list(selection_ids)
        if len(raw_selection_ids) != batch:
            raise ValueError("selection_ids must be aligned with logits.")
        selection_values = [str(value) for value in raw_selection_ids]
    else:
        if selection_ids.ndim != 1 or selection_ids.shape[0] != batch:
            raise ValueError("selection_ids must be a one-dimensional tensor aligned with logits.")
        if selection_ids.dtype == torch.bool or selection_ids.dtype.is_floating_point:
            raise ValueError("selection_ids must contain string-encoded integer values.")
        selection_values = [str(value) for value in selection_ids.tolist()]
    if candidate_ids.dtype != torch.long:
        raise ValueError("candidate_ids must have dtype torch.long.")
    if target_rows.dtype != torch.long or target_columns.dtype != torch.long:
        raise ValueError("target_rows and target_columns must have dtype torch.long.")
    if num_rows < 2 or num_columns < 2:
        raise ValueError("num_rows and num_columns must both be at least two.")
    if batch == 0:
        raise ValueError("row_column_candidate_loss requires a non-empty batch.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinity.")
    if bool(
        ((candidate_ids < 0) | (candidate_ids >= num_rows + num_columns))
        .any()
        .item()
    ):
        raise ValueError("candidate_ids contain values outside the row/column vocabulary.")

    order: list[str] = []
    seen: set[str] = set()
    for value in selection_values:
        if value not in seen:
            seen.add(value)
            order.append(value)

    row_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    column_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    row_nll = torch.zeros((), dtype=torch.float32, device=logits.device)
    column_nll = torch.zeros((), dtype=torch.float32, device=logits.device)
    complete_count = 0
    skipped_count = 0

    for selection in order:
        row_indices = [index for index, value in enumerate(selection_values) if value == selection]
        rows = torch.as_tensor(row_indices, dtype=torch.long, device=logits.device)
        if rows.numel() == 0:
            continue
        target_row_values = torch.unique(target_rows[rows])
        target_column_values = torch.unique(target_columns[rows])
        if target_row_values.numel() != 1 or target_column_values.numel() != 1:
            skipped_count += 1
            continue
        target_row = int(target_row_values[0].item())
        target_column = int(target_column_values[0].item())
        evidence = logits[rows]
        membership = candidate_ids[rows]
        row_rows = torch.nonzero(membership < num_rows, as_tuple=False).flatten()
        column_rows = torch.nonzero(membership >= num_rows, as_tuple=False).flatten()
        if row_rows.numel() == 0 or column_rows.numel() == 0:
            skipped_count += 1
            continue
        row_result = DecisionLevelLoss(
            candidate_vocabulary=tuple(str(index) for index in range(num_rows)),
            temperature=temperature,
        ).forward(
            evidence[row_rows],
            membership[row_rows],
            target_row,
            decision_id=f"row:{selection}",
        )
        column_result = DecisionLevelLoss(
            candidate_vocabulary=tuple(str(index) for index in range(num_columns)),
            temperature=temperature,
        ).forward(
            evidence[column_rows],
            membership[column_rows] - num_rows,
            target_column,
            decision_id=f"column:{selection}",
        )
        if row_result.loss is None or column_result.loss is None:
            skipped_count += 1
            continue
        row_loss = row_loss + row_result.loss
        column_loss = column_loss + column_result.loss
        row_nll = row_nll + row_result.loss.detach().float()
        column_nll = column_nll + column_result.loss.detach().float()
        complete_count += 1

    decision_count = len(order)
    if complete_count == 0:
        zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
        return zero, RowColumnCandidateLossDiagnostics(
            decision_count=decision_count,
            complete_count=0,
            skipped_count=skipped_count,
            row_nll=None,
            column_nll=None,
        )
    divisor = float(complete_count)
    diagnostics = RowColumnCandidateLossDiagnostics(
        decision_count=decision_count,
        complete_count=complete_count,
        skipped_count=skipped_count,
        row_nll=float((row_nll / divisor).item()),
        column_nll=float((column_nll / divisor).item()),
    )
    return (row_loss + column_loss) / divisor, diagnostics
