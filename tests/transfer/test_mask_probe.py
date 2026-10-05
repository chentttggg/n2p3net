"""Fixed-model mask interventions use observed data and production contracts."""

import copy
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.domain import adapt_observed_montage
from data.epochs import PreprocessingSpec, save_epoch_dataset
from data.identity import DatasetIdentityTable
from research.contracts import TrainingRunContract
from tests import PROJECT_ROOT
from tests.support.checkpoints import checkpoint_identity_fields, dataset_trunk
from tests.support.datasets import reference_dataset
from tests.support.io import read_json
from transfer.mask_probe import run_checkpoint_mask_probe


def _bind_study_domain(payload, study):
    contract = TrainingRunContract(**payload["training_contract"])
    contract = replace(
        contract,
        objective={**contract.objective, "source_risk_contract": {"domain_mass": {study: 1.0}}},
    )
    payload.update(training_contract=contract.record(), training_contract_digest=contract.digest())


def _fixture():
    raw = reference_dataset(
        np.random.default_rng(81).normal(0, 1e-6, (8, 7, 128)),
        reference="native-fixture",
        channels=("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz"),
    )
    raw = replace(
        raw,
        preprocessing=PreprocessingSpec(
            name="probe-fixture",
            sfreq=128,
            n_times=128,
            tmin_ms=-200,
            tmax_ms=800,
            filter_phase="forward",
            causal_iir_initial_state="steady_state_first_sample",
        ),
        identity_table=DatasetIdentityTable.from_source_rows(["s"], ["BNCI2014-008"]),
        event_timeline=replace(raw.event_timeline, dataset_ids=np.repeat("BNCI2014-008", 8)),
    )
    data = adapt_observed_montage(
        raw, ("Fz", "P3", "Pz", "P4", "PO8", "Oz"), reference_channel="Cz"
    )
    source = DatasetIdentityTable.from_source_rows(["other"], ["independent-source"])
    payload = checkpoint_identity_fields(source, data)
    _bind_study_domain(payload, "independent-source")
    trunk = dataset_trunk(data)
    payload["trunk_state_dict"] = trunk.state_dict()
    payload["trunk_state_dict"]["st_temporal_norm.running_mean"].fill_(3)
    payload["trunk_state_dict"]["st_temporal_norm.running_var"].fill_(4)
    payload["trunk_state_dict"]["st_temporal_norm.weight"].fill_(2)
    payload["trunk_state_dict"]["st_temporal_norm.bias"].fill_(1)
    return data, payload


def _probe(data, payload, **kwargs):
    return run_checkpoint_mask_probe(
        data,
        payload,
        target_subject="s",
        checkpoint_sha256="1" * 64,
        target_cache_sha256="2" * 64,
        evaluation_snapshot_sha256="3" * 64,
        **kwargs,
    )


def test_probe_deletes_observed_channels_and_records_bn_offsets_without_state_changes():
    data, payload = _fixture()
    before = copy.deepcopy(payload)
    result = _probe(data, payload, batch_size=3)
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert len(result["arms"]) == 5
    full, removed = result["arms"][:2]
    assert full["name"] == "full"
    assert removed["removed_channels"] == ["FZ"]
    assert removed["activations"]["bn_pre"]["channel_rms"][0] == 0
    assert removed["activations"]["bn_post"]["channel_mean"][0] == pytest.approx(
        1 - 6 / np.sqrt(4 + 1e-5), abs=1e-6
    )
    assert len(removed["logits"]) == data.n_epochs
    assert removed["evaluation_contract"]["requested_participant_keys"] == [
        "n2p3-origin://BNCI2014-008/participant/s"
    ]
    assert np.array_equal(data.X, _fixture()[0].X)
    for key, value in payload["trunk_state_dict"].items():
        assert value.equal(before["trunk_state_dict"][key])


def test_probe_rejects_claimed_six_slots_with_missing_real_observations():
    data, payload = _fixture()
    mask = data.trial_channel_mask.copy()
    mask[0, 0] = False
    values = data.X.copy()
    values[0, 0] = 0
    data = replace(data, X=values, trial_channel_mask=mask, qc_features=None)
    with pytest.raises(ValueError, match="all six.*observed"):
        _probe(data, payload, dry_run=True)


def test_probe_rejects_other_person_from_target_library_in_source():
    data, payload = _fixture()
    contaminated = DatasetIdentityTable.from_source_rows(["other"], ["BNCI2014-008"])
    payload.update(checkpoint_identity_fields(contaminated, data))
    _bind_study_domain(payload, "BNCI2014-008")
    with pytest.raises(ValueError, match="entire BNCI"):
        _probe(data, payload, dry_run=True)


def test_probe_dry_run_from_outside_repository_has_no_predictions_or_fit(tmp_path):
    data, payload = _fixture()
    cache = save_epoch_dataset(tmp_path / "cache.npz", data)
    checkpoint = tmp_path / "source.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / "plan.json"
    command = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments/probe_checkpoint_masks.py"),
            "--cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "5/5 admitted" in command.stdout
    result = read_json(output)
    assert result["optimizer_updates"] == 0
    assert result["requested_tasks"] == 5
    assert result["execution_snapshot_status"] == "required_before_execution"
    assert all("logits" not in arm for arm in result["records"][0]["arms"])
