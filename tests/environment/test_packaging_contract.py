from __future__ import annotations

import json

from data import source_records
from research.source_matrix import source_matrix_tasks
from tests import PROJECT_ROOT
from train.early_stopping import UnifiedTrainingBudget


def test_source_preparation_method_uses_installed_library_bytes(monkeypatch):
    original = source_records.file_sha256
    hashed = []

    def record_hash(path):
        hashed.append(path)
        assert "experiments" not in path.parts
        return original(path)

    monkeypatch.setattr(source_records, "file_sha256", record_hash)
    method = source_records.source_preparation_method()
    assert method["schema"] == "n2p3_source_record_method/3"
    assert method["runner_module"] == "data.source_records"
    assert method["runner_sha256"] == original(source_records.Path(source_records.__file__))
    assert hashed


def test_declared_training_budget_example_matches_default_policy():
    configuration = json.loads((PROJECT_ROOT / "configs/unified_training_budget.json").read_text())
    assert UnifiedTrainingBudget(**configuration).record() == UnifiedTrainingBudget().record()


def test_source_pool_expansion_example_retains_unique_complete_holdouts():
    configuration = json.loads((PROJECT_ROOT / "configs/source_pool_expansion.example.json").read_text())
    tasks = source_matrix_tasks(configuration)
    assert len(tasks) == 48
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert all(not set(task["studies"]) & set(task["excluded"]) for task in tasks)
