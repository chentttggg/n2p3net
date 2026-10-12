"""Synthetic SubjectAdapter installation identity and full-prefix refit."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from models.splitbn import validate_splitbn_target_install_receipt
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def test_subject_adapter_records_initial_and_refit_mask_scopes_and_replays_saved_state():
    rng = np.random.default_rng(20261012)
    core = N2P3Net(3, n_times=64, dropout=0, observed_channel_norm=True,
        splitbn_domain_count=2, splitbn_target_buffer_source="target_calibration")
    adapter = SubjectAdapter(core, config=SubjectAdapterConfig(
        head_kind="full_fine", epochs=1, batch_size=8, val_group_fraction=None,
        input_statistics="source", input_mean=np.zeros(3), input_std=np.ones(3),
        as_mmd_lambda0=.1, as_mmd_warmup_epochs=1, refit_full_prefix=True))
    x = rng.normal(size=(24, 3, 64)).astype(np.float32)
    y = np.tile([0, 1], 12)
    mask = np.ones((24, 3), dtype=bool)
    mask[:8, 1] = False
    x[~mask] = np.nan
    validation = np.arange(24) >= 16
    adapter.fit(x, y, trial_channel_mask=mask, validation_mask=validation,
        prefix_mask=np.ones(24, dtype=bool), source_X=rng.normal(size=(16, 3, 64)).astype(np.float32),
        source_labels=np.tile([0, 1], 8), source_domain_indices=np.repeat([0, 1], 8).astype(np.int64))
    receipts = adapter.last_history["splitbn_install_receipts"]
    initial, refit = receipts["initial"], receipts["refit"]
    assert initial is not None and refit is not None
    assert initial["st_temporal_norm"]["scope"]["phase"] == "initial_fit"
    assert initial["st_temporal_norm"]["sample_count"] == 16
    assert refit["st_temporal_norm"]["scope"]["phase"] == "refit_full_prefix"
    assert refit["st_temporal_norm"]["sample_count"] == 24
    assert initial["st_temporal_norm"]["channel_mask_sha256"] != refit["st_temporal_norm"]["channel_mask_sha256"]
    assert initial["st_temporal_norm"]["scope_sha256"] != refit["st_temporal_norm"]["scope_sha256"]
    validate_splitbn_target_install_receipt(core, refit)

    # Software save/load only: no public checkpoint or model admission.
    stream = io.BytesIO()
    torch.save({"state": core.state_dict(), "history": json.loads(json.dumps(adapter.last_history))}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    loaded = N2P3Net(3, n_times=64, dropout=0, observed_channel_norm=True,
        splitbn_domain_count=2, splitbn_target_buffer_source="target_calibration")
    loaded.load_state_dict(saved["state"])
    validate_splitbn_target_install_receipt(loaded, saved["history"]["splitbn_install_receipts"]["refit"])
    forged = json.loads(json.dumps(refit))
    forged["st_temporal_norm"]["scope"]["row_count"] = 25
    with pytest.raises(ValueError, match="receipt digest"):
        validate_splitbn_target_install_receipt(loaded, forged)
