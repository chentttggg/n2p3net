from __future__ import annotations

import numpy as np
import pytest

from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from tests.support.io import write_json
from transfer.binary import binary_time_split


def test_e1_dry_run_preserves_actual_budgets_and_makes_no_fit(monkeypatch):
    from transfer.subject_adapter import SubjectAdapter
    from transfer.target_experiments import build_target_tasks

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run fitted a model")

    monkeypatch.setattr(SubjectAdapter, "fit", forbidden)
    dataset = binary_dataset()
    splits = {k: binary_time_split(np.arange(40) * 2.0, np.arange(40) * 2.0 + .8,
                                  calibration_trials=k, maximum_calibration_trials=20, tmin_s=-.2)
              for k in (0, 20)}
    plan = build_target_tasks(dataset, {"p1": (np.arange(40), splits)}, stage="E1", batch_size=128, seed=9)
    assert plan["task_count"] == 24
    assert plan["fit_calls"] == 23
    assert len(plan["scopes"]) == 2
    assert set(task["K"] for task in plan["tasks"]) == {0, 20}
    assert plan["dry_run_fits_data"] is False
    assert plan["scopes"]["p1:20"]["test_rows"] == list(range(20, 40))


def test_b0_dry_run_deduplicates_effective_batches_and_keeps_requested_aliases():
    from transfer.target_experiments import build_target_tasks

    dataset = binary_dataset()
    split = binary_time_split(np.arange(40) * 2.0, np.arange(40) * 2.0 + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    plan = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})}, stage="B0", seed=9)
    assert plan["task_count"] == 4  # two origins, two distinct stopping rules, all caps are full batch
    for task in plan["tasks"]:
        assert task["requested_batch_aliases"] == [32, 128, 512]
        assert task["training_rows"] == 16
        assert task["validation_rows"] == 4
    assert plan["final_test_used_for_selection"] is False


def test_compact_dry_run_has_six_positive_arms_and_one_k0_without_inner_selection(monkeypatch):
    from transfer.subject_adapter import SubjectAdapter
    from transfer.target_experiments import build_target_tasks

    def forbidden(*args, **kwargs):
        raise AssertionError("compact dry-run fitted a model")

    monkeypatch.setattr(SubjectAdapter, "fit", forbidden)
    dataset = binary_dataset()
    splits = {k: binary_time_split(np.arange(40) * 2.0, np.arange(40) * 2.0 + .8,
                                  calibration_trials=k, maximum_calibration_trials=20, tmin_s=-.2)
              for k in (0, 20)}
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), splits)},
                                  stage="compact", batch_size=512)
    assert manifest["task_count"] == 7
    assert manifest["fit_calls"] == 6
    assert {task["arm"] for task in manifest["tasks"]} == {
        "zero_shot", "compact_waveform_lda", "compact_scratch", "compact_full_fine",
        "compact_source_statistics", "compact_frozen_bn_buffers", "compact_head_only",
    }
    assert all(task["validation_rows"] == 0 for task in manifest["tasks"])
    assert manifest["scopes"]["p1:20"]["test_rows"] == list(range(20, 40))
    for task in manifest["tasks"]:
        if task.get("recipe") and task["recipe"]["weights"] != "none":
            assert task["budget"]["updates"] == 60
            assert task["budget"]["draws"] == 1200
            assert task["budget"]["effective_batch_size"] == 20


def test_early_stopping_dry_run_separates_selection_cost_and_full_refit():
    from transfer.target_experiments import build_target_tasks

    dataset = binary_dataset()
    splits = {k: binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=k, maximum_calibration_trials=20, tmin_s=-.2) for k in (0, 20)}
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), splits)}, stage="ES", batch_size=512)
    assert manifest["task_count"] == 11 and manifest["fit_calls"] == 16
    task = next(t for t in manifest["tasks"] if t["arm"] == "es_full_patience")
    assert task["selection_budget"]["optimizer_updates_upper_bound"] == 120
    assert task["selection_budget"]["trial_draws_upper_bound"] == 2160
    assert task["epoch_selection"] == {"mode": "patience", "patience": 6, "min_delta": 1e-6}
    assert task["validation_rows"] == 4
    assert {t["treatment"] for t in manifest["tasks"] if "epoch_selection" in t} == {"full", "head", "scratch"}


def test_fixed_epoch_extension_declares_only_requested_fits_and_preserves_other_conditions():
    from transfer.target_experiments import build_target_tasks

    dataset = binary_dataset()
    splits = {k: binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=k, maximum_calibration_trials=20, tmin_s=-.2) for k in (0, 20)}
    common = dict(stage="ES", batch_size=512, seed=12)
    original = build_target_tasks(dataset, {"p1": (np.arange(40), splits)}, **common)
    extended = build_target_tasks(dataset, {"p1": (np.arange(40), splits)}, **common,
        epoch_selection={"epochs": 90, "treatments": ["full", "scratch"],
                         "modes": ["fixed"], "include_baselines": False})
    assert extended["task_count"] == extended["fit_calls"] == 2
    assert extended["scopes"] == original["scopes"]
    assert {t["arm"] for t in extended["tasks"]} == {"es_full_fixed", "es_scratch_fixed"}
    for task in extended["tasks"]:
        prior = next(t for t in original["tasks"] if t["arm"] == task["arm"])
        assert task["recipe"] == {**prior["recipe"], "epochs": 90}
        assert task["budget"]["updates"] == 90 and task["budget"]["draws"] == 1800
        assert task["validation_rows"] == task["selection_budget"]["inner_epochs_upper_bound"] == 0


@pytest.mark.parametrize("parameters", [{"epochs": True}, {"epochs": 90.5}, {"epochs": 0},
    {"treatments": []}, {"treatments": ["full", "full"]}, {"treatments": ["unknown"]},
    {"modes": "fixed"}, {"modes": ["test_auc"]}, {"include_baselines": 0}])
def test_epoch_matrix_rejects_ambiguous_or_invalid_declarations(parameters):
    from transfer.target_experiments import build_target_tasks

    with pytest.raises(ValueError):
        build_target_tasks(binary_dataset(), {}, stage="ES", epoch_selection=parameters)


def test_early_stopping_public_execution_preserves_full_prefix_and_persisted_state(tmp_path):
    import hashlib

    import torch

    from research.contracts import semantic_sha256
    from transfer.target_experiments import (
        apply_correctness_budget,
        build_target_tasks,
        execute_target_tasks,
    )

    dataset = binary_dataset()
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=False)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})}, stage="ES", batch_size=512,
        epoch_selection={"patience": 1, "min_delta": 10.})
    apply_correctness_budget(manifest, epochs=3)
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64}, target_protocol={"kind": "single_recording_prefix"},
        evidence_scope="software_correctness")
    report = execute_target_tasks(dataset, payload, manifest, output=tmp_path / "run", context=context, device="cpu")
    assert report["completed_cells"] == 10 and report["failed_cells"] == 0
    assert report["fit_execution"] == {"requested_fits": 16, "fit_calls": 13, "cache_hits": 3}
    for index, cell in enumerate(report["cells"]):
        assert semantic_sha256(cell["evaluation_contract"]) == cell["evaluation_contract_digest"]
        if "epoch_selection" not in cell:
            continue
        result = cell["epoch_selection_result"]
        assert result["refit_history"]["fit_scope"]["training_rows"] == 20
        assert cell["evaluation_contract"]["adaptation"]["epoch_selection"] == cell["epoch_selection"]
        if cell["epoch_selection"]["mode"] != "fixed":
            assert result["selected_epochs"] == 1
            assert result["inner_epochs_run"] == (2 if cell["epoch_selection"]["mode"] == "patience" else 3)
        saved = torch.load(tmp_path / f"run/{index:05d}.pt", weights_only=False)
        assert saved["recipe"]["epochs"] == result["selected_epochs"]
        assert cell["requested_recipe"]["epochs"] == 3
        assert cell["recipe"] == saved["recipe"]
        assert saved["epoch_selection"]["policy"] == cell["epoch_selection"]


def test_public_execution_binds_contract_and_test_changes_cannot_change_fit(tmp_path):
    import copy
    import hashlib

    import torch

    from transfer.target_experiments import build_target_tasks, execute_target_tasks

    dataset = binary_dataset()
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=False)
    split = binary_time_split(np.arange(40) * 2.0, np.arange(40) * 2.0 + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})}, stage="E1", seed=9)
    manifest["tasks"] = [task for task in manifest["tasks"] if task["arm"] == "e1_waveform_lda"]
    manifest.update(task_count=1, fit_calls=1)
    context = dict(target_study="synthetic_p300", source_seed=1, checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
                   target_protocol={"kind": "single_recording_prefix"})
    first = execute_target_tasks(dataset, payload, manifest, output=tmp_path / "first", context=context, device="cpu")
    changed = copy.deepcopy(dataset)
    changed.X[20:] *= 100
    changed.y[20:] = 1 - changed.y[20:]
    second = execute_target_tasks(changed, payload, manifest, output=tmp_path / "second", context=context, device="cpu")
    left, right = first["cells"][0], second["cells"][0]
    assert left["status"] == right["status"] == "completed"
    assert left["fitted_state_sha256"] == right["fitted_state_sha256"]
    assert left["evaluation_contract"]["model_origin"]["kind"] == "scratch"
    assert left["participant_key"] == "n2p3-origin://synthetic_p300/participant/p1"


def test_b0_public_execution_records_validation_curves_without_changing_fit(tmp_path):
    import copy
    import hashlib

    import torch

    from transfer.target_experiments import (
        apply_correctness_budget,
        build_target_tasks,
        execute_target_tasks,
    )

    dataset = binary_dataset()
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=True)
    split = binary_time_split(np.arange(40) * 2.0, np.arange(40) * 2.0 + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})}, stage="B0", seed=9)
    apply_correctness_budget(manifest, epochs=2)
    context = dict(target_study="synthetic_p300", source_seed=1,
                   checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
                   target_protocol={"kind": "single_recording_prefix"}, evidence_scope="software_correctness")
    first = execute_target_tasks(dataset, payload, manifest, output=tmp_path / "first", context=context, device="cpu")
    changed = copy.deepcopy(dataset)
    changed.X[16:20] *= 100
    changed.y[16:20] = 1 - changed.y[16:20]
    second = execute_target_tasks(changed, payload, manifest, output=tmp_path / "second", context=context, device="cpu")
    for left, right in zip(first["cells"], second["cells"], strict=True):
        assert left["status"] == right["status"] == "completed"
        assert left["fitted_state_sha256"] == right["fitted_state_sha256"]
        for cell in (left, right):
            history = cell["fit_history"]
            assert history["validation_behavior"] == "diagnostics_only"
            assert history["best_epoch"] is None
            assert len(history["validation_curve"]) == 2
            assert history["fit_budget"]["updates"] == 2
            assert history["fit_budget"]["draws"] == 32
            assert all(row["auc"] is not None for row in history["validation_curve"])


def test_software_probe_cannot_freeze_scientific_batch_rule(tmp_path, capsys):

    import pytest

    from experiments.run_cross_dataset_transfer import main
    from transfer.batch_cohort import declare_batch_cohort

    record = tmp_path / "probe.json"
    write_json(record, {"status": "completed", "task_manifest": {"stage": "B0"},
                                  "evidence_scope": "software_correctness"})
    cohort = declare_batch_cohort([{"target_study": "A", "stage": "B0", "dry_run_fits_data": False,
        "scopes": {"p": {"participant_key": "p", "K": 0}}, "protocol_exclusions": []}], {"A": ["p"]})
    cohort_path = tmp_path / "cohort.json"
    write_json(cohort_path, cohort)
    with pytest.raises(SystemExit):
        main(["--freeze-batch-from", str(record), "--batch-cohort", str(cohort_path),
              "--output", str(tmp_path / "rule.json")])
    assert "software correctness probes" in capsys.readouterr().err
    assert not (tmp_path / "rule.json").exists()
