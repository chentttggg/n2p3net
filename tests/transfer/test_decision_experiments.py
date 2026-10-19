from __future__ import annotations

import copy

import pytest

from transfer.decision_experiments import expand_decision_diagnostics
from transfer.recipes import unified_target_recipes


def test_diagnostic_matrix_is_matched_and_counts_physical_groups_not_updates():
    recipes = unified_target_recipes(membership_available=True, source_prior_weight=.01)
    tasks = [dict(scope_id="p:200", arm=recipe.name, recipe=recipe.record(), K=200,
                  training_rows=200, validation_rows=0, membership_available=True) for recipe in recipes]
    manifest = dict(stage="unified", tasks=tasks,
                    scopes={"p:200": {"membership": {"available": True,
                             "groups": [{"decision_id": "one", "target_candidate": "3"}]}}})
    before = copy.deepcopy(manifest)
    result = expand_decision_diagnostics(manifest, {"schema": "n2p3_prior_decision_diagnostic/1",
        "decision_weight": 1.,
        "full_joint_interpretation": "exploratory_not_low_G_sufficiency"})
    assert manifest == before
    assert result["task_count"] == result["fit_calls"] == 9
    rows = {row["arm"]: row for row in result["tasks"]}
    assert rows["diag_decision"]["recipe"].get("source_prior_weight", 0) == 0
    assert rows["diag_prior"]["recipe"]["candidate_loss_weight"] == 0
    assert rows["diag_prior_decision"]["recipe"]["source_prior_weight"] == .01
    assert rows["diag_head_decision"]["recipe"]["weights"] == "head"
    assert rows["diag_head_decision"]["recipe"]["bn_affine"] == "frozen"
    for name, row in rows.items():
        if name.startswith("diag_"):
            assert row["decision_information"]["physical_decisions"] == 1
            assert row["decision_information"]["independent_decisions_created_by_repeated_updates"] == 0


def test_diagnostic_design_records_selected_zero_prior_without_inventing_effect():
    recipes = unified_target_recipes(membership_available=True, source_prior_weight=0)
    manifest = dict(stage="unified", scopes={"s": {"membership": {"available": True}}},
                    tasks=[dict(scope_id="s", arm=recipe.name, recipe=recipe.record(), training_rows=200) for recipe in recipes])
    result = expand_decision_diagnostics(manifest, {"schema": "n2p3_prior_decision_diagnostic/1",
        "decision_weight": 1., "full_joint_interpretation": "exploratory_not_low_G_sufficiency"})
    assert result["tasks"][0]["decision_information"]["prior_contrast_degenerate"]
    with pytest.raises(ValueError, match="differs"):
        expand_decision_diagnostics(manifest, {"schema": "n2p3_prior_decision_diagnostic/1",
            "decision_weight": .5, "full_joint_interpretation": "exploratory_not_low_G_sufficiency"})
