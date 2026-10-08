"""Cheap source programme rejection precedes any heavy cache preparation."""
from __future__ import annotations

import pytest

from experiments.run_source_training_chain import _execute_source_groups
from research.source_selection import resolve_source_program


@pytest.mark.parametrize("scope", ["public_source_development", "public_calibration_development", "unknown"])
def test_invalid_source_evidence_scope_cannot_create_or_merge_caches(tmp_path, scope):
    output = tmp_path / "must_remain_absent"
    with pytest.raises(ValueError, match="only development or software_correctness"):
        _execute_source_groups({"evidence_scope": scope}, output=output, device="cpu",
                               stages=("P2_source",), evaluate_outer=False, groups=[({}, "six")])
    assert not output.exists()
    assert not list(tmp_path.rglob("*.npz"))


def test_source_programme_rejects_software_budget_under_development_at_same_owner():
    with pytest.raises(ValueError, match="correctness budgets"):
        resolve_source_program({"evidence_scope": "development", "correctness_source_updates": 2}, ("P2_source",))


@pytest.mark.parametrize("scope", ["development", "software_correctness"])
def test_legal_source_scopes_resolve_before_data_phase(scope):
    assert resolve_source_program({"evidence_scope": scope}, ("P2_source",)) == {"name": "declared_legacy_stages"}
