"""Maintain live contract checks whose old test entry was lost in regrouping."""

from __future__ import annotations

import json

import pytest

from experiments.analyze_candidate_cross_decision import analyze_manifest


def test_analysis_rejects_retired_bi_schema_before_artifact_access(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema": "n2p3_bi2014a_cross_decision_experiment/1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="BI-specific schemas are unsupported"):
        analyze_manifest(path)
