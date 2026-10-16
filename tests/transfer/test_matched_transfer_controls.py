from __future__ import annotations

import numpy as np
import pytest

from transfer.binary import (
    matched_split_digests,
    paired_transfer_effects,
    validate_matched_transfer_controls,
)


def test_positive_transfer_cannot_omit_no_transfer_neural_control():
    with pytest.raises(ValueError, match="scratch"):
        validate_matched_transfer_controls(("zero_shot", "full_fine", "shrinkage_lda"), (0, 200, 800, 2000))


def test_zero_calibration_and_pure_baselines_need_no_unfittable_scratch():
    validate_matched_transfer_controls(("zero_shot",), (0,))
    validate_matched_transfer_controls(("shrinkage_lda",), (200,))
    validate_matched_transfer_controls(("zero_shot", "full_fine", "target_only"), (0, 200))
    validate_matched_transfer_controls(("full_fine", "target_only_tuned"), (200,))


def cells():
    common = {"participant_key": "n2p3-origin://Study/participant/P1", "K": 120,
              **matched_split_digests(np.arange(120), np.arange(130, 150))}
    return [{**common, "arm": "full_fine", "status": "completed", "auc": .7},
            {**common, "arm": "target_only", "status": "completed", "auc": .6}]


def test_fallback_budget_is_compared_at_actual_k():
    pair, = paired_transfer_effects(cells(), budget_field="K")
    assert pair["actual_calibration_trials"] == 120
    assert pair["delta_auc"] == pytest.approx(.1)
    assert pair["status"] == "completed"


def test_different_calibration_or_test_rows_cannot_claim_a_matched_contrast():
    rows = cells()
    rows[1].update(matched_split_digests(np.arange(120), np.arange(131, 151)))
    with pytest.raises(ValueError, match="same calibration and test"):
        paired_transfer_effects(rows, budget_field="K")


def test_missing_or_wrong_k_control_is_rejected():
    rows = cells()
    rows[1]["K"] = 200
    with pytest.raises(ValueError, match="same-K scratch"):
        paired_transfer_effects(rows, budget_field="K")


def test_failed_scratch_keeps_pair_without_manufactured_delta():
    rows = cells()
    rows[1].update(status="fit_failure", auc=None)
    pair, = paired_transfer_effects(rows, budget_field="K")
    assert pair["status"] == "incomplete_pair" and pair["delta_auc"] is None
    assert pair["control_status"] == "fit_failure"
