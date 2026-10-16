from __future__ import annotations

import pytest
import torch

from transfer.candidate_set_loss import row_column_candidate_loss


def _complete_decisions() -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    # Two 2x2 decisions. Each decision contains exactly one row0, row1, col0,
    # col1 flash (candidate ids 0,1 for rows and 2,3 for columns).
    candidate_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    selection_ids = ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"]
    target_rows = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    target_columns = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    return candidate_ids, selection_ids, target_rows, target_columns


@pytest.mark.parametrize(
    "selection_ids,values,expected_decisions",
    [
        (["d1"] * 4 + ["d2"] * 4, [0.5, -0.2, 0.1, -0.4, -0.1, 0.3, -0.5, 0.2], 2),
        (torch.tensor([7, 7, 7, 7]), [0.0] * 4, 1),
    ],
    ids=["string-decisions", "integer-tensor-decision"],
)
def test_row_column_loss_matches_decision_level_surrogate_and_has_gradient(
    selection_ids, values, expected_decisions
) -> None:
    candidate_ids, _, target_rows, target_columns = _complete_decisions()
    count = len(values)
    logits = torch.tensor(
        values,
        dtype=torch.float64,
        requires_grad=True,
    )

    loss, diagnostics = row_column_candidate_loss(
        logits,
        candidate_ids[:count],
        selection_ids,
        target_rows[:count],
        target_columns[:count],
        num_rows=2,
        num_columns=2,
    )

    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    assert diagnostics.decision_count == expected_decisions
    assert diagnostics.complete_count == expected_decisions
    assert diagnostics.skipped_count == 0
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float((logits.grad != 0).any())


def test_row_column_loss_skips_incomplete_decisions_without_fake_zero_gradient() -> None:
    candidate_ids = torch.tensor([0, 1, 2])
    selection_ids = ["incomplete", "incomplete", "incomplete"]
    target_rows = torch.tensor([0, 0, 0])
    target_columns = torch.tensor([1, 1, 1])
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    loss, diagnostics = row_column_candidate_loss(
        logits,
        candidate_ids,
        selection_ids,
        target_rows,
        target_columns,
        num_rows=2,
        num_columns=2,
    )

    assert float(loss) == 0.0
    assert diagnostics.decision_count == 1
    assert diagnostics.complete_count == 0
    assert diagnostics.skipped_count == 1
    assert diagnostics.row_nll is None


def test_row_column_loss_rejects_out_of_vocabulary_candidate() -> None:
    candidate_ids = torch.tensor([0, 1, 4, 3])
    selection_ids = ["bad", "bad", "bad", "bad"]
    target_rows = torch.tensor([0, 0, 0, 0])
    target_columns = torch.tensor([1, 1, 1, 1])
    logits = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside the row/column vocabulary"):
        row_column_candidate_loss(
            logits,
            candidate_ids,
            selection_ids,
            target_rows,
            target_columns,
            num_rows=2,
            num_columns=2,
        )
