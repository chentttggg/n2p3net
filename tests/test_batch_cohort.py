from __future__ import annotations

import pytest

from transfer.batch_cohort import declare_batch_cohort, verify_batch_cohort_results


def _preflight(study, person, legal=True):
    return {"target_study": study, "stage": "B0", "dry_run_fits_data": False,
            "scopes": {person: {"participant_key": person, "K": 200,
                "inner_split": {"legal": legal, "reason": None if legal else "too few rows"},
                "inner_training_rows": [0, 1], "inner_validation_rows": [2, 3]}},
            "protocol_exclusions": []}


def test_declaring_batch_cohort_rejects_a_missing_predeclared_person():
    with pytest.raises(ValueError, match="complete.*participant"):
        declare_batch_cohort([_preflight("A", "p1")], {"A": ["p1", "p2"]})


def test_batch_freeze_rejects_partial_results_and_retains_illegal_inner_requests():
    cohort = declare_batch_cohort([_preflight("A", "p1"), _preflight("B", "p2"),
                                  _preflight("B", "p3", legal=False)], {"A": ["p1"], "B": ["p2", "p3"]})
    case = cohort["cases"][0]
    rows = [{**case, "budget_mode": "epochs", "batch_size": batch, "arm": arm,
             "status": "completed", "score": .6}
            for batch in (32, 128, 512) for arm in ("source", "scratch")]
    with pytest.raises(ValueError, match="complete.*cohort"):
        verify_batch_cohort_results(rows, cohort)
    rows.extend({**cohort["cases"][1], "budget_mode": "epochs", "batch_size": batch,
                 "arm": arm, "status": "fit_failed", "score": None}
                for batch in (32, 128, 512) for arm in ("source", "scratch"))
    verify_batch_cohort_results(rows, cohort)
    assert cohort["requested_participants"] == 3
    assert cohort["eligible_actual_K_cases"] == 2
    assert len(cohort["cases"]) == 3
