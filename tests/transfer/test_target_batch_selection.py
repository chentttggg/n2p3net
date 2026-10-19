from __future__ import annotations

import pytest

from transfer.target_experiments import freeze_target_batch


def _rows():
    rows = []
    for batch in (32, 128, 512):
        for study, participant, budgets in (("A", "p1", (200, 800)), ("A", "p2", (200,)), ("B", "p3", (200,))):
            for budget in budgets:
                for arm in ("source", "scratch"):
                    rows.append(dict(study=study, participant_key=participant, K=budget,
                                     batch_size=batch, budget_mode="epochs", arm=arm,
                                     status="completed", score=0.7, elapsed_seconds=10.0,
                                     calibration_rows_sha256=f"{participant}-{budget}",
                                     validation_rows_sha256=f"{participant}-{budget}-v"))
    return rows


def test_batch_selection_weights_budgets_then_people_then_studies():
    rows = _rows()
    for row in rows:
        if row["batch_size"] == 128:
            row["score"] = 1.0 if row["participant_key"] == "p1" else 0.6
    result = freeze_target_batch(rows)
    assert result["batch_size"] == 32
    # For 128: A=(1+.6)/2=.8, B=.6, mean=.7. More budgets do not weight p1 twice.
    assert result["candidates"][1]["score"] == pytest.approx(0.7)
    assert result["selection_evidence"] is True


def test_failed_batch_keeps_common_cohort_and_cannot_win_on_success_subset():
    rows = _rows()
    for row in rows:
        if row["batch_size"] == 512:
            row["score"] = 0.99
    rows[-1].update(status="fit_failed", score=None)
    result = freeze_target_batch(rows)
    assert result["batch_size"] == 32
    assert result["candidates"][-1]["score"] is None
    assert result["candidates"][-1]["failed_cells"] == 1


def test_batch_tie_uses_measured_total_time_before_size():
    rows = _rows()
    for row in rows:
        if row["batch_size"] == 128:
            row["elapsed_seconds"] = 1.0
    assert freeze_target_batch(rows)["batch_size"] == 128


def test_batch_selection_rejects_changed_validation_or_duplicate_pair():
    rows = _rows()
    rows[-1]["validation_rows_sha256"] = "changed"
    with pytest.raises(ValueError, match="same.*rows"):
        freeze_target_batch(rows)
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        freeze_target_batch(rows + [rows[0]])


def test_no_legal_inner_evidence_preserves_historical_batch_without_claim():
    result = freeze_target_batch([])
    assert result["batch_size"] == 32
    assert result["selection_evidence"] is False


def test_explicit_user_fixed_batch_is_not_b0_selection_and_cannot_be_overridden():
    from transfer.target_experiments import (
        resolve_target_batch_rule,
        user_fixed_target_batch_rule,
    )

    rule = user_fixed_target_batch_rule(batch_size=512)
    batch, accepted = resolve_target_batch_rule({"batch_rule": rule})
    assert batch == 512 and accepted == rule
    assert accepted["selection_evidence"] is False
    assert accepted["provenance"] == "user_explicit"
    assert "batch_cohort_digest" not in accepted
    for dry_run in (False, True):
        with pytest.raises(ValueError, match="frozen.*rule"):
            resolve_target_batch_rule({"batch_rule": rule}, requested_batch=128, dry_run=dry_run)
    for changes in ({"provenance": "inferred"}, {"selection_evidence": True}):
        with pytest.raises(ValueError, match="user-fixed"):
            resolve_target_batch_rule({"batch_rule": {**rule, **changes}})
    with pytest.raises(ValueError, match="fallback"):
        resolve_target_batch_rule({"batch_rule": {"schema": "n2p3_target_batch_rule/1",
            "batch_size": 512, "selection_evidence": False}})
