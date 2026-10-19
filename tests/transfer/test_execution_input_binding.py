from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.binary import binary_time_split
from transfer.result_reuse import ResultIntegrityError
from transfer.target_experiments import build_target_tasks, execute_target_tasks


@pytest.mark.parametrize("changed_axis", ["test_values", "fit_values", "snapshot", "evidence_scope"])
def test_imported_prediction_requires_original_fit_score_and_execution_identity(tmp_path, changed_axis):
    dataset = binary_dataset()
    checkpoint = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(checkpoint, weights_only=False)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})},
                                  stage="compact", seed=9, batch_size=512)
    manifest["tasks"] = [row for row in manifest["tasks"] if row["arm"] == "compact_waveform_lda"]
    manifest.update(task_count=1, fit_calls=1)
    context = dict(target_study="synthetic_p300", source_seed=1,
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
                   target_protocol={"kind": "single_recording_prefix"},
                   evidence_scope="software_correctness", plan={})
    first = execute_target_tasks(dataset, payload, copy.deepcopy(manifest), output=tmp_path / "first",
                                 context=context, device="cpu")
    assert first["completed_cells"] == 1
    changed = copy.deepcopy(dataset)
    new_context = copy.deepcopy(context)
    new_context["plan"]["reuse_target_records"] = [str(tmp_path / "first/record.json")]
    if changed_axis == "test_values":
        changed.X[20:] *= -1
    elif changed_axis == "fit_values":
        changed.X[:20] *= 7
    elif changed_axis == "snapshot":
        new_context["evaluation_snapshot"]["archive_sha256"] = "d" * 64
    else:
        new_context["evidence_scope"] = "development"
    with pytest.raises((ValueError, ResultIntegrityError), match="identity|execution|snapshot|scope|input"):
        execute_target_tasks(dataset, payload, copy.deepcopy(manifest), output=tmp_path / "imported",
                             context=new_context, device="cpu", scope_datasets={"p1:20": changed})


def test_reuse_tracks_actual_independent_inner_arrays_and_reuses_unchanged_fit(tmp_path):
    from data.epochs import select_scheduled_events
    from train.early_stopping import GlobalEarlyStoppingPolicy
    from transfer.epoch_selection import IndependentEpochSelectionData
    from transfer.result_reuse import TargetResultReuse
    from transfer.target_experiments import TargetScopeDatasets

    dataset = binary_dataset()
    checkpoint = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(checkpoint, weights_only=False)
    split = binary_time_split(np.arange(40)*2., np.arange(40)*2.+.8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    policy = GlobalEarlyStoppingPolicy(patience=1, min_delta=100., target_max_epochs=2)
    manifest = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})}, stage="P1",
                                  batch_size=512, global_early_stopping=policy.record())
    manifest["tasks"] = [row for row in manifest["tasks"] if row["arm"] == "p1_full_fine"]
    manifest["task_count"] = 1
    scope = manifest["tasks"][0]["scope_id"]
    inner = select_scheduled_events(dataset, np.arange(40) < 20, reason="authorized inner regression fixture")
    training = np.arange(20) < 16
    views = TargetScopeDatasets()
    views[scope] = dataset
    views.selection[scope] = IndependentEpochSelectionData(inner, training, ~training,
                                      copy.deepcopy(manifest["scopes"][scope]["inner_split"]))
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64}, target_protocol={"kind": "single_recording_prefix"},
        evidence_scope="software_correctness")
    reuse = TargetResultReuse()

    def run(name):
        return execute_target_tasks(dataset, payload, copy.deepcopy(manifest), output=tmp_path/name,
            context=context, device="cpu", result_reuse=reuse, scope_datasets=views)["cells"][0]

    first = run("first")
    inner.X[~training] *= 50
    changed = run("changed")
    assert changed["execution_identity"] != first["execution_identity"]
    assert changed["fit_execution"]["fit_calls"] > 0
    repeated = run("repeated")
    assert repeated["execution_identity"] == changed["execution_identity"]
    assert repeated["fit_execution"]["fit_calls"] == 0
    assert repeated["fit_execution"]["cache_hits"] > 0
