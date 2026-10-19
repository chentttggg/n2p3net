"""Stub checkpoint work validates E2 pool/data lifetime without model fitting."""
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_source_training_chain as actual_runner


@pytest.fixture
def runner(monkeypatch):
    owner = actual_runner
    monkeypatch.setattr(owner, "_E2_DATASET", None)
    counts = {"pools": 0, "initializers": 0, "closed": 0, "calls": [], "lifecycle": []}

    class StubPool:
        def __init__(self, *, max_workers, initializer, initargs, **kwargs):
            self.reference = initargs[0]
            assert self.reference["metadata"].X is None
            assert all("path" in row for row in self.reference["arrays"].values())
            counts["pools"] += 1
            counts["child_count"] = max_workers
            counts["reference"] = self.reference
            for _ in range(max_workers):
                counts["initializers"] += 1
                # Attach the real mmap owner; avoid changing the pytest process's native pools.
                dataset = owner.copy(self.reference["metadata"])
                for name, row in self.reference["arrays"].items():
                    setattr(dataset, name, np.load(row["path"], mmap_mode="r", allow_pickle=False))
                owner._E2_DATASET = dataset
        def __enter__(self):
            return self
        def submit(self, function, arguments):
            future = Future()
            try:
                future.set_result(function(arguments))
            except Exception as error:
                future.set_exception(error)
            return future
        def __exit__(self, *error):
            counts["closed"] += 1
            counts["lifecycle"].append("child_pool_closed")
            for value in vars(owner._E2_DATASET).values():
                if isinstance(value, np.memmap):
                    value._mmap.close()
            owner._E2_DATASET = None

    def score(path, dataset, cohort, *, recipe, **options):
        rows = []
        for cell in cohort:
            assert not isinstance(dataset.X, np.memmap) or not dataset.X.flags.writeable
            counts["calls"].append((recipe.seed, cell["participant_key"], tuple(cell["test_rows"])))
            if recipe.seed == -1 and cell["participant_key"] == "p2":
                raise RuntimeError("synthetic unexpected worker failure")
            rows.append({**cell, "seed": recipe.seed, "status": "fit_failure" if cell["participant_key"] == "p1" else "completed",
                "observed": dataset.X[cell["test_rows"]].tolist()})
        return rows

    monkeypatch.setattr(owner, "ProcessPoolExecutor", StubPool)
    monkeypatch.setattr(owner, "_score_checkpoint", score)
    return owner, counts


def operands():
    dataset = SimpleNamespace(X=np.arange(48, dtype=np.float32).reshape(8, 2, 3),
        y=np.arange(8), subject_ids=np.array(["a"]*8))
    cohort = [{"participant_key": f"p{i}", "actual_K": i, "test_rows": [7-i, i]}
              for i in range(3)]
    return dataset, cohort


def test_seeds_and_axes_share_one_pool_and_one_read_only_eeg_view(runner, tmp_path):
    owner, counts = runner
    dataset, cohort = operands()
    with owner._E2ScoringSession(dataset, workers=3) as session:
        first = owner._score_e2_candidates(dataset, tmp_path, cohort, (1, 2), workers=3,
            progress_path=tmp_path/"participant.json", recipe=SimpleNamespace(seed=11), device="cpu", session=session)
        second = owner._score_e2_candidates(dataset, tmp_path, list(reversed(cohort)), (1, 2), workers=3,
            progress_path=tmp_path/"library.json", recipe=SimpleNamespace(seed=23), device="cpu", session=session)
        assert counts["pools"] == 1
        assert counts["initializers"] == counts["child_count"] == 2
        assert session.workers == 3
        assert len(first) == len(second) == 6
        assert {row["seed"] for row in first} == {11}
        assert {row["seed"] for row in second} == {23}
        assert sum(row["status"] == "fit_failure" for row in first) == 2
        for row in first+second:
            assert row["observed"] == dataset.X[row["test_rows"]].tolist()
        shared_paths = [Path(row["path"]) for row in counts["reference"]["arrays"].values()]
        assert all(path.exists() for path in shared_paths)
    assert counts["closed"] == 1
    assert all(not path.exists() for path in shared_paths)
    assert np.array_equal(dataset.X, np.arange(48, dtype=np.float32).reshape(8, 2, 3))


def test_one_ready_person_can_score_independent_checkpoint_updates_in_parallel(runner, tmp_path):
    owner, counts = runner
    dataset, cohort = operands()
    with owner._E2ScoringSession(dataset, workers=4) as session:
        results = session.score(tmp_path, cohort[:1], (1, 2, 3), progress_path=tmp_path/"one-person.json",
            recipe=SimpleNamespace(seed=5), device="cpu")
        assert session.workers == 3
        assert counts["child_count"] == 2
        assert {row["updates"] for row in results} == {1, 2, 3}


def test_fixed_resource_grant_does_not_rebuild_the_pool_for_each_seed_or_axis(runner, tmp_path):
    from contextlib import contextmanager

    owner, counts = runner
    dataset, cohort = operands()
    class Admission:
        native_threads = 1
        entries = 0
        @contextmanager
        def target_workers(self, requested):
            assert requested == 4
            self.entries += 1
            yield 2
    admission = Admission()
    with owner._E2ScoringSession(dataset, workers=4, memory_admission=admission) as session:
        for seed in (11, 23):
            rows = session.score(tmp_path, cohort, (1, 2), progress_path=tmp_path/f"cap-{seed}.json",
                recipe=SimpleNamespace(seed=seed), device="cpu")
            assert len(rows) == 6
        assert session.workers == 2
        assert session.starts == counts["pools"] == admission.entries == 1
        assert session.capacity_limiter == "resource_admission"
    assert counts["closed"] == 1


def test_successful_jobs_reuse_allocator_until_session_exit_and_failures_cleanup_immediately(runner, tmp_path, monkeypatch):
    from research import source_execution

    owner, _ = runner
    dataset, cohort = operands()
    released = []
    monkeypatch.setattr(source_execution, "release_source_worker", released.append)
    with owner._E2ScoringSession(dataset, workers=1) as session:
        for seed in (11, 23):
            session.score(tmp_path, cohort[:1], (1, 2), progress_path=tmp_path/f"success-{seed}.json",
                recipe=SimpleNamespace(seed=seed), device="cpu")
        assert released == []
        session.score(tmp_path, cohort[1:2], (1,), progress_path=tmp_path/"declared-failure.json",
            recipe=SimpleNamespace(seed=11), device="cpu")
        assert released == ["cpu"]
    assert released == ["cpu", "cpu"]


def test_pool_exit_releases_parent_allocator_before_returning_session_resource_lease(runner, tmp_path, monkeypatch):
    from contextlib import contextmanager

    from research import source_execution

    owner, counts = runner
    dataset, cohort = operands()
    events = counts["lifecycle"]
    class Admission:
        native_threads = 1
        @contextmanager
        def target_workers(self, requested):
            events.append("lease_granted")
            try:
                yield 2
            finally:
                events.append("lease_returned")
    monkeypatch.setattr(source_execution, "release_source_worker", lambda device: events.append("parent_allocator_released"))
    with owner._E2ScoringSession(dataset, workers=2, memory_admission=Admission()) as session:
        session.score(tmp_path, cohort[:1], (1, 2), progress_path=tmp_path/"lease-order.json",
            recipe=SimpleNamespace(seed=5), device="cpu")
        assert events == ["lease_granted"]
    assert events == ["lease_granted", "child_pool_closed", "parent_allocator_released", "lease_returned"]


def test_unexpected_worker_failure_keeps_requested_and_completed_denominators(runner, tmp_path):
    owner, counts = runner
    dataset, cohort = operands()
    with owner._E2ScoringSession(dataset, workers=3) as session:
        with pytest.raises(RuntimeError, match="requested jobs and completed cells"):
            session.score(tmp_path, cohort, (7,), progress_path=tmp_path/"failure.json",
                recipe=SimpleNamespace(seed=-1), device="cpu")
    progress = json.loads((tmp_path/"failure.json").read_bytes())
    assert progress["requested_cells"] == 3
    assert progress["completed_cells"] == 2
    assert progress["failed_cells"] == 1  # The declared per-cell failure remains a returned outcome.
    assert progress["scoring_session"]["failed_jobs"][0]["people"] == ["p2"]
    assert counts["closed"] == 1


def test_actual_selection_call_reuses_pool_for_target_seeds_and_both_axes_before_refit(runner, tmp_path, monkeypatch):
    from research import source_selection

    owner, counts = runner
    dataset, cohort = operands()
    dataset.n_epochs = len(dataset.X)
    for cell in cohort:
        cell["domain"] = "software-domain"
    folds = [{"axis": axis, "fold_id": f"f{i}", "validation_rows": [i]}
             for i, axis in enumerate(("participant", "library"))]
    monkeypatch.setattr(source_selection, "source_inner_folds", lambda *args, **kwargs: folds)
    monkeypatch.setattr(source_selection, "select_source_updates", lambda *args, **kwargs:
        {"U_ce": 1, "U_cal": 1, "U_zero": 1, "requested_cells": kwargs["requested_cells"]})
    monkeypatch.setattr(source_selection, "source_fit_ledger", lambda *args, **kwargs: {"software_ledger": True})
    monkeypatch.setattr(owner, "_pseudo_target_cohort", lambda *args, **kwargs: (cohort, []))
    refits = []
    def fit(**kwargs):
        assert counts["closed"] == 1
        refits.append(kwargs)
        return {"checkpoints": {1: str(tmp_path/"fitted.u1.pt")}, "fit_ledger": {"software_ledger": True},
            "source_training_reuse": {}, "physical_optimizer_updates": 1, "physical_trial_draws": 512}
    monkeypatch.setattr(owner, "fit_or_reuse_source_trajectory", fit)
    outer = {"name": "software-outer", "excluded_source_datasets": ["held"], "source_protocols": []}
    report = {"schema": "n2p3_source_training_chain_run/1", "tasks": [{"key": f"software-outer/common_three/seed7/E2_{row['axis']}_{row['fold_id']}",
        "checkpoints": {"1": "source.u1.pt"}, "status": "completed", "config": {"seed": 7}}
        for row in folds], "selection": [], "preflight_failures": [], "outer_evaluation_jobs": []}
    owner._run_e2_selection(dataset, outer, 7, (11, 23), tmp_path, "a"*64, "b"*64,
        {"evidence_scope": "software_correctness", "correctness_candidate_updates": [1],
         "correctness_target_epochs": 1, "target_workers": 3, "compile_mode": None},
        report, False, "cpu", {})
    assert counts["pools"] == 1 and counts["initializers"] == 2
    assert len(refits) == 1  # Same original source seed/update is reused across selectors.
    assert len(report["selection"]) == len(report["outer_evaluation_jobs"]) == 4
    assert {(row["target_seed"], row["axis"]) for row in report["selection"]} == {
        (11, "participant"), (11, "library"), (23, "participant"), (23, "library")}
    assert all(len(row["outcomes"]) == len(row["requested_cells"]) == 3 for row in report["selection"])
