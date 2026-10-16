from research.source_matrix import (
    CORE_STUDIES,
    SOURCE_POOLS,
    freeze_matched_dose,
    source_matrix_tasks,
)


def test_complete_matrix_and_study_holdout():
    tasks = source_matrix_tasks()
    assert len(tasks) == 153
    assert len({t["id"] for t in tasks}) == 153
    for task in tasks:
        assert not set(task["studies"]) & set(task["excluded"])
        if task["family"] == "Bpublic":
            assert task["target"] in CORE_STUDIES
            assert set(task["studies"]) == set(SOURCE_POOLS[task["pool"]]) - {task["target"]}


def test_dose_preserves_pool_and_matches_people_classes_trials():
    studies = SOURCE_POOLS["B7"]
    counts = {s: {f"{s}/{i}": [100 + i, 20 + i] for i in range(5)} for s in studies}
    dose = freeze_matched_dose(counts)
    for fold in dose["folds"].values():
        assert fold["participants"] == 15
        assert fold["class_counts"] == [15 * n for n in fold["per_participant_class_counts"]]
    lookup = {t["id"]: t for t in source_matrix_tasks()}
    for name, selection in dose["selections"].items():
        assert len(selection) == len(set(selection)) == 15
        assert {s for s, _ in selection} == set(lookup[name]["studies"])
    assert dose == freeze_matched_dose(counts)


def test_dose_budget_does_not_use_heldout_study_counts():
    studies = SOURCE_POOLS["B7"]
    counts = {s: {f"{s}/{i}": [100 + i, 20 + i] for i in range(5)} for s in studies}
    original = freeze_matched_dose(counts)
    counts["BI2014a"] = {"different_person": [1, 1]}
    changed = freeze_matched_dose(counts)
    assert original["folds"]["BI2014a"] == changed["folds"]["BI2014a"]
    for task in source_matrix_tasks():
        if task.get("target") == "BI2014a" and task["dose"] == "matched":
            assert original["selections"][task["id"]] == changed["selections"][task["id"]]


def test_new_pool_target_controls_follow_declared_pool_and_seed(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from experiments import run_public_pool_targets as runner
    from tests.support.datasets import binary_dataset

    data = binary_dataset()
    subject = str(data.subject_ids[0])
    split = SimpleNamespace(calibration_rows=np.arange(20), test_rows=np.arange(20, 40), purged_rows=np.array([], int))
    monkeypatch.setattr(runner, "dataset_calibration_splits", lambda *a, **k: (np.arange(data.n_epochs), {20: split}))
    policy = {"control_pool": "new_reference", "lda_seed": 91, "protocols": {"new-target": {"kind": "single_recording_prefix"}}}
    task = dict(target="new-target", seed=91, pool="new_reference", dose="full")
    first = runner.build_manifest(data, task, [subject], policy=policy)
    assert {t["recipe"]["initialization"] for t in first["tasks"]} == {"source", "scratch", "waveform_lda"}
    other_seed = runner.build_manifest(data, {**task, "seed": 92}, [subject], policy=policy)
    assert {t["recipe"]["initialization"] for t in other_seed["tasks"]} == {"source", "scratch"}
    other_pool = runner.build_manifest(data, {**task, "pool": "extra"}, [subject], policy=policy)
    assert {t["recipe"]["initialization"] for t in other_pool["tasks"]} == {"source"}


def test_missing_public_inputs_fail_with_all_requested_tasks_recorded(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    from experiments import run_public_pool_targets as runner

    tasks = [dict(id=f"missing-{i}", family="Bpublic", target="BI2014a", pool="B0", dose="full", seed=i) for i in (7, 9)]
    monkeypatch.setattr(runner, "source_matrix_tasks", lambda: tasks)
    monkeypatch.setattr(runner, "verified_executed_snapshot", lambda *a, **k: {"archive_sha256": "0" * 64})
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda *, max_workers, **_: ThreadPoolExecutor(max_workers=max_workers))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: pytest.fail("missing inputs must not wait without an explicit deadline"))
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="unresolved failures"):
        runner.main(["--prepared", str(tmp_path), "--sources", str(tmp_path), "--dose-sources", str(tmp_path),
                     "--output", str(output), "--snapshot", "unused", "--workers", "1"])
    record = json.loads((output / "progress.json").read_text())
    assert record["status"] == "failed" and record["failed_jobs"] == 2
    assert {r["id"] for r in record["results"]} == {t["id"] for t in tasks}


def test_compact_dose_views_keep_exact_old_rows_and_order(monkeypatch):
    from dataclasses import replace
    from functools import lru_cache

    import numpy as np

    from data.domain import annotate_source_domain, ensure_epoch_dataset_namespace
    from data.epochs import concatenate_epoch_datasets
    from data.identity import DatasetIdentityTable
    from experiments import run_source_pool_matrix as runner
    from research.source_matrix import matched_dose_rows
    from research.source_selection import source_identity_axes
    from tests.support.datasets import binary_dataset

    sources = {}
    for study in ("left", "right"):
        data = binary_dataset(24)
        data.identity_table = DatasetIdentityTable.from_source_rows(data.subject_ids.tolist(), [study] * data.n_epochs)
        data.event_timeline = replace(data.event_timeline, dataset_ids=np.repeat(study, data.event_timeline.n_events))
        data.provenance.update(dataset_class=study, source_reference="measured_channel:CZ")
        data.lineage = None
        sources[study] = annotate_source_domain(ensure_epoch_dataset_namespace(data, study), study)
    @lru_cache(maxsize=1)
    def cached(path, digest, preprocessing_json=None):
        return sources[path]
    monkeypatch.setattr(runner, "source_cache", cached)
    tasks = [dict(id=f"task{seed}", studies=["left", "right"], target="held", seed=seed) for seed in (7, 9)]
    people = [(study, source_identity_axes(data)[1][0]) for study, data in sources.items()]
    dose = {"folds": {"held": {"per_participant_class_counts": [3, 2]}},
            "selections": {task["id"]: people for task in tasks}}
    prepared = {study: {"cache": study, "cache_sha256": "mock"} for study in sources}
    compact = runner.prepare_dose_views(tasks, prepared, dose)
    whole = concatenate_epoch_datasets(list(sources.values()), name="whole", provenance=runner.source_provenance(tasks[0]))
    domains, keys, _ = source_identity_axes(whole)
    for task in tasks:
        rows = matched_dose_rows(whole, task, dose, keys, domains)
        actual = compact[task["id"]]
        np.testing.assert_array_equal(actual.X, whole.X[rows])
        np.testing.assert_array_equal(actual.y, whole.y[rows])
        np.testing.assert_array_equal(actual.subject_ids, whole.subject_ids[rows])
        np.testing.assert_array_equal(actual.event_timeline.event_ids, whole.event_timeline.event_ids[rows])
