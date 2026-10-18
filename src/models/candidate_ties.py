"""Common-offset-independent candidate tie resolution in fixed score units."""
from __future__ import annotations

import numpy as np

CANDIDATE_SCORE_TIE_ATOL = 1e-12


def candidate_top_tie_mask(scores, *, axis=-1):
    """Top candidates within the fixed absolute contrast floor.

    Callers own score validity and complete candidate coverage. In log-score
    units the contrast is log odds. The score origin and distant losers never
    change this floor; arithmetic that changes represented contrasts is still
    evaluated on the actual resulting floating-point scores.
    """
    values = np.asarray(scores)
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.isclose(values, maximum, rtol=0., atol=CANDIDATE_SCORE_TIE_ATOL)
