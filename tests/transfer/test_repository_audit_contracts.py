"""Counterexamples for the 20261019 transfer and scientific identity audit."""
from __future__ import annotations

import copy
import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync_contract import BRAIN_SYNC_SESSION_SCHEMA
from data.identity import DatasetIdentityTable
from data.study import dataset_study_membership
from research.contracts import semantic_sha256
from tests.support.checkpoints import checkpoint_identity_fields, small_trunk, write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.calibration_candidates import (
    _fit_rows_digest,
    calibration_candidate_recipes,
    select_calibration_candidate,
)
from transfer.checkpoint import (
    CHECKPOINT_SCHEMA,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
)
from transfer.quality import PersonalEpochRawQualityError, validate_epoch_quality_fit_scope
from transfer.result_reuse import _tensor_state_digest


def test_transfer_only_never_retries_target_only_after_all_source_fits_fail(monkeypatch):
    import transfer.calibration_candidates as candidates

    attempted = []
    def fail(_dataset, recipe, *_args, **_kwargs):
        attempted.append(recipe.initialization)
        raise ValueError("independent optimizer failure")
    monkeypatch.setattr(candidates, "fit_calibration_candidate", fail)
    dataset = binary_dataset()
    with pytest.raises(ValueError, match="transfer_only source calibration failed"):
        select_calibration_candidate(dataset, calibration_candidate_recipes("transfer_only"),
            calibration_mask=np.ones(dataset.n_epochs, bool), selector_mode="transfer_only")
    assert attempted and set(attempted) == {"source"}


@pytest.mark.parametrize("status", ["not_supplied", "incompatible"])
def test_transfer_only_source_gate_rejects_absence_or_incompatibility(status):
    from transfer.calibrated import _calibration_source_failure

    assert _calibration_source_failure({"status": status, "message": "channel mismatch"}, "transfer_only")


def test_public_pool_rejects_another_person_from_the_whole_target_study():
    from experiments.run_public_pool_targets import target_context

    data = binary_dataset()
    membership = dataset_study_membership(data)
    stream = next(iter(membership["stream_to_study"]))
    training = DatasetIdentityTable.from_source_rows(["another_person"], [stream])
    payload = checkpoint_identity_fields(training, data)
    payload["source_study_membership"] = membership
    payload["training_contract"]["objective"]["source_study_membership"] = membership
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    job = {"task": {"target": "BI2014a", "excluded": ["BI2014a"], "seed": 9},
           "cache_sha256": "a" * 64, "checkpoint_sha256": "b" * 64, "snapshot": {}, "environment": {}}
    with pytest.raises(ValueError, match="excluded target libraries"):
        target_context(job, data, payload)


def test_same_shape_weight_substitution_is_rejected_by_path_and_mapping(tmp_path):
    dataset = binary_dataset()
    path = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="f" * 64)
    payload = load_checkpoint_payload(path)
    payload.update(schema=CHECKPOINT_SCHEMA, trunk_state_dict_sha256=_tensor_state_digest(payload["trunk_state_dict"]))
    load_n2p3_trunk_checkpoint(payload, dataset, target_subject="p1")
    name = next(name for name, value in payload["trunk_state_dict"].items() if value.is_floating_point())
    payload["trunk_state_dict"][name].reshape(-1)[0] += 0.25
    content = io.BytesIO()
    torch.save(payload, content)
    with pytest.raises(ValueError, match="weight bytes"):
        load_checkpoint_payload(io.BytesIO(content.getvalue()))
    with pytest.raises(ValueError, match="weight bytes"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="p1")


def test_new_checkpoint_cannot_remove_the_mandatory_weight_binding(tmp_path):
    dataset = binary_dataset()
    path = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="f" * 64)
    payload = load_checkpoint_payload(path)
    payload["schema"] = CHECKPOINT_SCHEMA
    payload.pop("trunk_state_dict_sha256", None)
    content = io.BytesIO()
    torch.save(payload, content)
    with pytest.raises(ValueError, match="weight bytes"):
        load_checkpoint_payload(io.BytesIO(content.getvalue()))


def test_selector_plan_counts_candidate_fits_and_refit_and_correctness_budget():
    from transfer.binary import binary_time_split
    from transfer.target_experiments import apply_correctness_budget, build_target_tasks

    data = binary_dataset()
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(data, {"p1": (np.arange(40), {20: split})}, stage="E3", batch_size=32)
    expected = sum(len(task["recipes"]) + 1 for task in manifest["tasks"])
    assert manifest["fit_calls"] == expected

    original = next(task for task in manifest["tasks"] if task["selector_mode"] == "five_candidates")
    assert original["selection_budget"]["fit_calls_upper_bound"] == 6
    prior_updates = original["selection_budget"]["optimizer_updates_upper_bound"]
    apply_correctness_budget(manifest, epochs=1)
    assert original["selection_budget"]["optimizer_updates_upper_bound"] < prior_updates
    assert manifest["fit_calls"] == expected


def test_declared_recording_clocks_require_the_labels_to_be_ready():
    from transfer.calibration_split import calibration_inner_split

    arguments = (np.tile([0, 1], 4), np.tile(np.arange(4) * 2., 2),
                 np.tile(np.arange(4) * 2. + .8, 2), np.repeat(["a", "b"], 4))
    missing = calibration_inner_split(*arguments, tmin_s=-.2,
        recording_start_times={"a": 0., "b": 100.})
    assert not missing.legal and "label-availability" in missing.reason
    complete = calibration_inner_split(*arguments, tmin_s=-.2,
        recording_start_times={"a": 0., "b": 100.}, label_available_times={"a": 10., "b": 110.})
    assert complete.legal


def test_brainsync_fold_local_qc_requires_its_raw_quality_stage():
    data = binary_dataset()
    data = replace(data, provenance={**data.provenance, "session_schema": BRAIN_SYNC_SESSION_SCHEMA})
    with pytest.raises(PersonalEpochRawQualityError, match="channel-local continuous QC /4"):
        validate_epoch_quality_fit_scope(data, np.ones(data.n_epochs, bool), policy=FoldLocalArtifactPolicy())


def test_brainsync_fold_local_qc_accepts_an_actual_fitted_and_executed_raw_stage(tmp_path):
    from data.brainsync import load_brainsync_sessions
    from tests.support.brainsync import write_bids_session
    from tests.support.quality import fit_raw_quality
    from transfer.quality import fit_epoch_quality

    root = write_bids_session(tmp_path / "cal", session_id="cal", target=2,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    data = load_brainsync_sessions([root], continuous_qc_model=fit_raw_quality(root))
    mask = np.ones(data.n_epochs, bool)
    frozen = fit_epoch_quality(data, mask, FoldLocalArtifactPolicy(min_clean_epochs=2))
    validate_epoch_quality_fit_scope(data, mask, model=frozen)
    assert frozen.fit_n_epochs == data.n_epochs


@pytest.mark.parametrize("field", ["block_id", "selection_repetitions", "calibration_available_timestamp_s"])
def test_fit_cache_tracks_grouping_and_availability_metadata(field):
    data = binary_dataset()
    metadata = data.metadata.copy()
    metadata[field] = np.arange(data.n_epochs)
    data = replace(data, metadata=metadata)
    changed = metadata.copy()
    changed.loc[0, field] += 1
    mask = np.ones(data.n_epochs, bool)
    assert _fit_rows_digest(data, mask) != _fit_rows_digest(replace(data, metadata=changed), mask)
    # Information outside a declared fit remains irrelevant to that fit.
    mask[0] = False
    assert _fit_rows_digest(data, mask) == _fit_rows_digest(replace(data, metadata=changed), mask)


def test_declared_optimizer_batch_does_not_depend_on_live_memory(monkeypatch):
    import train.runtime as runtime_module
    from train.runtime import GpuPerformanceScheduler

    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    runtime.device = torch.device("cuda")
    model = torch.nn.Linear(8, 2)
    results = []
    for free in (1024, 64 * 1024 ** 3):
        monkeypatch.setattr(runtime_module, "_memory_info", lambda _device, free=free: (free, free))
        results.append(runtime.choose_batch_size(64, (3, 100), model=model))
    assert results == [64, 64]


def test_oom_aborts_the_declared_optimizer_program_without_a_smaller_batch(monkeypatch):
    from baselines.deep import DeepBaseline, DeepConfig

    config = DeepConfig(epochs=1, batch_size=8)
    config.optimizer_batch_policy = "fixed"
    model = DeepBaseline("eegnet", n_chans=3, n_times=100, sfreq=100.,
        config=config, device=torch.device("cpu"))
    attempted = []
    def fail(*_args, requested_batch_size, **_kwargs):
        attempted.append(requested_batch_size)
        model._active_batch_size = requested_batch_size
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(model, "_fit_attempt", fail)
    data = binary_dataset()
    with pytest.raises(RuntimeError, match="declared optimizer batch"):
        model.fit(data.X, data.y)
    assert attempted == [8]


def test_explicit_adaptive_profile_retries_and_records_the_actual_update_rows(monkeypatch):
    import hashlib

    from baselines.deep import DeepBaseline, DeepConfig

    config = DeepConfig(epochs=2, batch_size=8, seed=19, val_group_frac=None)
    config.optimizer_batch_policy = "adaptive"
    model = DeepBaseline("eegnet", n_chans=3, n_times=100, sfreq=100.,
        config=config, device=torch.device("cpu"))
    original = model._fit_attempt
    attempts = []
    def first_oom(*args, requested_batch_size, **kwargs):
        attempts.append(requested_batch_size)
        if len(attempts) == 1:
            model._active_batch_size = requested_batch_size
            raise RuntimeError("CUDA out of memory")
        return original(*args, requested_batch_size=requested_batch_size, **kwargs)
    monkeypatch.setattr(model, "_fit_attempt", first_oom)
    data = binary_dataset(count=16)
    model.fit(data.X, data.y)
    assert attempts == [8, 4] and model.last_runtime["oom_retries"] == 1
    history = model.last_history
    assert history["optimizer_batch_policy"] == "adaptive"
    assert history["batch_sizes"] == [4] * 8
    assert history["optimizer_updates"] == 8 and history["trial_draws"] == 32
    generator = torch.Generator(device="cpu").manual_seed(19)
    digest = hashlib.sha256()
    for _ in range(2):
        digest.update(torch.randperm(16, generator=generator).numpy().tobytes())
    assert history["batch_sequence_sha256"] == digest.hexdigest()


def test_asmmd_actual_fit_is_independent_of_ambient_torch_rng():
    from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

    rng = np.random.default_rng(19)
    inputs = rng.normal(size=(16, 3, 128)).astype(np.float32)
    labels = np.tile([0, 1], 8)
    source_logits = rng.normal(size=(24, 2)).astype(np.float32)
    source_labels = np.tile([0, 1], 12)
    core = small_trunk()
    fits = []
    for ambient in (3, 987):
        adapter = SubjectAdapter(copy.deepcopy(core), config=SubjectAdapterConfig(
            epochs=2, batch_size=8, seed=17, val_group_fraction=None,
            refit_full_prefix=False, as_mmd_lambda0=.3))
        torch.manual_seed(ambient)
        torch.rand(37)
        adapter.fit(inputs, labels, source_logits=source_logits, source_labels=source_labels)
        fits.append(adapter)
    assert fits[0].last_history["as_mmd_losses"] == fits[1].last_history["as_mmd_losses"]
    assert fits[0].fitted_state_digest() == fits[1].fitted_state_digest()


def test_pretraining_band_weights_survive_numeric_state_roundtrip():
    from transfer.pretraining import PretrainingConfig, PretrainingTask

    task = PretrainingTask(small_trunk(), PretrainingConfig())
    task.update_band_weights(torch.randn(16, 3, 128))
    restored = PretrainingTask(small_trunk(), PretrainingConfig())
    restored.load_state_dict(task.state_dict())
    assert torch.equal(restored.band_weights, task.band_weights)


def test_quality_report_refuses_to_invoke_a_fitting_callback(tmp_path):
    from research.recording_quality import create_recording_quality_report
    from tests.support.brainsync import write_bids_session

    called = []
    recording = write_bids_session(tmp_path / "recording", session_id="quality", target=2,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    report = create_recording_quality_report(recording, tmp_path / "report",
        epoch_quality=lambda: called.append(True))
    assert not called
    assert "fitting callback" in report["error"]["message"]


def test_negative_weight_decay_is_rejected_before_fitting():
    from transfer.recipes import TargetRecipe

    with pytest.raises(ValueError, match="weight_decay"):
        TargetRecipe("invalid", weight_decay=-.1)


def test_requested_people_cannot_duplicate_the_research_denominator():
    from transfer.cohort import resolve_subject_scope

    with pytest.raises(ValueError, match="duplicate participants"):
        resolve_subject_scope(["p1", "p2"], ["p1", "p2"], requested_subjects=["p1", "p1"])


def test_protocol_exclusion_keeps_the_preflight_study_owner():
    from transfer.batch_cohort import declare_batch_cohort

    manifest = {"stage": "B0", "dry_run_fits_data": False, "scopes": {},
                "protocol_exclusions": [{"participant_key": "p1", "study": "foreign"}]}
    cohort = declare_batch_cohort([{"target_study": "actual", "task_manifest": manifest}], {"actual": ["p1"]})
    assert cohort["protocol_exclusions"][0]["study"] == "actual"


def test_numeric_personal_state_loads_without_promoting_it_to_a_workflow(tmp_path):
    from baselines.features import downsampled_time_points
    from transfer.calibrated import (
        CalibratedPredictor,
        _input_contract,
        load_calibrated_predictor,
        save_calibrated_predictor,
    )

    data = binary_dataset()
    size = data.n_channels * downsampled_time_points(data.n_times, data.preprocessing.sfreq)
    predictor = CalibratedPredictor({"input_contract": _input_contract(data),
        "model_kind": "target_only_shrinkage_lda", "participant_key": data.identity_table.authority_keys()[0],
        "selection": {}}, {"coefficient": torch.zeros(size), "feature_mean": torch.zeros(size),
        "feature_scale": torch.ones(size), "intercept": torch.zeros(1)})
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "numeric.pt", predictor))
    restored.validate_input(data)
    with pytest.raises(ValueError, match="personal workflow metadata is missing"):
        restored.load_inference_session(tmp_path / "absent_session")
