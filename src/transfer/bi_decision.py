"""Decision evaluation for the BI2014a 6x6 row/column speller.

Unlike the 9-choice digit task, a BI flash is partial evidence: it votes for
one of six rows or one of six columns. The character is the intersection of
the winning row and winning column.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _row_col(flash_code: int) -> tuple[int, int]:
    code = int(flash_code)
    if 20 <= code <= 25:
        return code - 20, -1
    if 40 <= code <= 45:
        return -1, code - 40
    if 60 <= code <= 65:
        return code - 60, -1
    if 80 <= code <= 85:
        return -1, code - 80
    raise ValueError(f"flash_code {flash_code} is outside the BI2014a schedule.")


def hit_at_repetition_6x6(
    logits: Sequence[float],
    flash_codes: Sequence[int],
    target_rows: Sequence[int],
    target_cols: Sequence[int],
    selection_ids: Sequence,
    repetition_indices: Sequence[int],
    *,
    max_repetitions: int | None = None,
) -> dict[int, float]:
    """Return character hit rate at each suffix repetition prefix 1..R.

    A hit requires both the winning row and the winning column to match the
    target pair. Every row/column is flashed once per repetition, so one
    repetition already yields six row votes and six column votes.
    """

    logits = np.asarray(logits, dtype=float)
    flash_codes = np.asarray(flash_codes, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    target_cols = np.asarray(target_cols, dtype=np.int64)
    selection_ids = np.asarray(selection_ids).astype(str)
    repetition_indices = np.asarray(repetition_indices, dtype=np.int64)
    if not (
        len(logits)
        == len(flash_codes)
        == len(target_rows)
        == len(target_cols)
        == len(selection_ids)
        == len(repetition_indices)
    ):
        raise ValueError("decision arrays must be aligned.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN/inf.")
    row_codes = np.empty(len(flash_codes), dtype=np.int64)
    col_codes = np.empty(len(flash_codes), dtype=np.int64)
    for index, code in enumerate(flash_codes):
        row_codes[index], col_codes[index] = _row_col(code)
    selections = np.unique(selection_ids)
    if max_repetitions is None:
        max_repetitions = int(repetition_indices.max()) + 1
    if max_repetitions < 1:
        raise ValueError("max_repetitions must be positive.")

    hits: dict[int, float] = {}
    for r in range(1, max_repetitions + 1):
        correct = 0
        total = 0
        for selection in selections:
            sel = (selection_ids == selection) & (repetition_indices < r)
            if not sel.any():
                continue
            row_score = np.zeros(6, dtype=float)
            col_score = np.zeros(6, dtype=float)
            row_sel = sel & (row_codes >= 0)
            col_sel = sel & (col_codes >= 0)
            np.add.at(row_score, row_codes[row_sel], logits[row_sel])
            np.add.at(col_score, col_codes[col_sel], logits[col_sel])
            row_winners = np.flatnonzero(
                np.isclose(row_score, float(row_score.max()), rtol=1e-12, atol=1e-12)
            )
            col_winners = np.flatnonzero(
                np.isclose(col_score, float(col_score.max()), rtol=1e-12, atol=1e-12)
            )
            predicted_row = int(row_winners[0]) if len(row_winners) == 1 else None
            predicted_col = int(col_winners[0]) if len(col_winners) == 1 else None
            truth_rows = np.unique(target_rows[sel])
            truth_cols = np.unique(target_cols[sel])
            if len(truth_rows) == 1 and len(truth_cols) == 1:
                correct += int(
                    predicted_row is not None
                    and predicted_col is not None
                    and predicted_row == truth_rows[0]
                    and predicted_col == truth_cols[0]
                )
                total += 1
        hits[r] = float(correct / total) if total else float("nan")
    return hits
