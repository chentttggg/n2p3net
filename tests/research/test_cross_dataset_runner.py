from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.run_cross_dataset_transfer import frozen_features, planned_splits
from models.n2p3net import N2P3Net
from transfer.checkpoint import predict_n2p3_checkpoint_with_features


def test_batched_frozen_features_preserve_scores_features_and_source_state():
    torch.manual_seed(7)
    model = N2P3Net(n_channels=3, n_times=179, sfreq=128)
    X = np.random.default_rng(7).normal(size=(7, 3, 179)).astype(np.float32)
    mask = np.ones((7, 3), dtype=bool)
    mask[:, 1] = False
    X[:, 1] = 0
    stats = (np.zeros((1, 3, 1), np.float32), np.ones((1, 3, 1), np.float32))
    device = torch.device("cpu")
    expected = predict_n2p3_checkpoint_with_features(model, X, input_stats=stats, device=device, trial_channel_mask=mask)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    actual = frozen_features(model, X, mask, stats, device, batch_size=2)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_whole_study_multirun_protocol_rejects_undeclared_order():
    from types import SimpleNamespace
    timeline = SimpleNamespace(
        evidence_indices=np.arange(600), onset_times_s=np.tile(np.arange(300), 2),
        evidence_available_times_s=np.tile(np.arange(300), 2)+0.8,
        session_ids=np.repeat(["0", "2"], 300), run_ids=np.repeat("run-10", 600),
    )
    dataset = SimpleNamespace(subject_ids=np.repeat("P1", 600), n_epochs=600, event_timeline=timeline)
    protocol = {"kind": "last_session_holdout", "session_order": ["0", "1", "2"],
                "run_order": ["run-2"], "test_session": "2"}
    with pytest.raises(ValueError, match="original protocol order"):
        planned_splits(dataset, "P1", protocol, (48, 96, 192))


@pytest.fixture
def admitted_prefix_input(tmp_path, monkeypatch):
    import json

    from data.epochs import load_epoch_dataset
    from data.source_records import (
        file_sha256,
        load_target_prefixes,
        prepare_source_record,
        prepare_source_record_prefixes,
    )
    from data.study import dataset_study_membership
    from tests.support.signal import make_raw

    raw = make_raw(sfreq=256, n_seconds=30, ch_names=("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz"))
    labels = np.tile([0, 1], 12)
    events = np.column_stack(((np.arange(24) + 2) * 256, np.zeros(24, int), labels + 1))
    monkeypatch.setattr("data.source_records._read_record", lambda *args: (raw.copy(), events.copy(), labels.copy()))
    raw_path = tmp_path / "source.bin"
    raw_path.write_bytes(b"scoped target admission fixture")
    item = dict(study="BNCI2014-008", subject="4", path=raw_path.name, sha256=file_sha256(raw_path),
                sample_rate_hz=256, reference="nose", session="0", run="0")
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prefixes",
                                         budgets=(0, 200), minimum_test_trials=2)
    declaration = {"schema": "n2p3_target_raw_prefix_collection/1", "preprocessing": pair["preprocessing"],
                   "preparation_method": pair["preparation_method"], "records": [pair]}
    path = tmp_path / "prefixes.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    prefixes = load_target_prefixes(path)
    source_record = prepare_source_record({**item, "study": "BI2014a", "subject": "9"},
                                          root=tmp_path, output=tmp_path / "source")
    source = load_epoch_dataset(source_record["cache"], validation="attested")
    return prefixes["4"]["test"], prefixes, dict(requested=["4"], budgets=(0, 200),
        plan={"target_study": item["study"], "target_protocol": {"kind": "single_recording_prefix"}},
        training_identity=source.identity_table, source_membership=dataset_study_membership(source),
        prefix_manifest=declaration)


def test_unified_admission_retains_requested_effective_and_actual_budget_counts(admitted_prefix_input):
    from experiments.run_cross_dataset_transfer import admit_unified_prefixes

    fixed, prefixes, options = admitted_prefix_input
    admitted = admit_unified_prefixes(fixed, prefixes, **options)
    assert admitted["prefix_allocations"]["4"]["requested_budgets"] == [0, 200]
    assert admitted["prefix_allocations"]["4"]["effective_budgets"] == [0, 19]
    assert admitted["prefix_allocations"]["4"]["retained_calibration_trials"] == {"0": 0, "19": 19}
    assert admitted["budget_adjustments"][0]["requested_K"] == 200
    assert admitted["budget_adjustments"][0]["effective_K"] == 19


@pytest.mark.parametrize("case", ["missing_request", "missing_adjustment", "stale_loaded_manifest"])
def test_unified_admission_rejects_incomplete_or_unbound_allocation(admitted_prefix_input, case):
    from copy import deepcopy

    from experiments.run_cross_dataset_transfer import admit_unified_prefixes

    fixed, original, options = admitted_prefix_input
    prefixes = {"4": dict(original["4"])}
    options = dict(options)
    if case == "missing_request":
        options["budgets"] = (0, 16, 200)
    elif case == "missing_adjustment":
        evidence = deepcopy(prefixes["4"]["manifest"])
        evidence["adjustments"] = {}
        prefixes["4"]["manifest"] = evidence
        options["prefix_manifest"] = {**options["prefix_manifest"], "records": [evidence]}
    else:
        prefixes["4"]["manifest"] = {**prefixes["4"]["manifest"], "unexpected_revision": True}
    with pytest.raises(ValueError, match="budgets|adjustments|frozen manifest"):
        admit_unified_prefixes(fixed, prefixes, **options)


@pytest.mark.parametrize("role", ["test", "calibration", "inner_training"])
def test_unified_admission_checks_identity_of_every_actual_view(admitted_prefix_input, role):
    from dataclasses import replace

    from data.identity import DatasetIdentityTable
    from experiments.run_cross_dataset_transfer import admit_unified_prefixes

    fixed, original, options = admitted_prefix_input
    entry = {**original["4"], "calibration": dict(original["4"]["calibration"]),
             "early_stopping_inner": {k: dict(v) for k, v in original["4"]["early_stopping_inner"].items()}}
    identity = DatasetIdentityTable.from_source_rows(["4"], ["BI2014a"])
    if role == "test":
        entry["test"] = replace(entry["test"], identity_table=identity)
    elif role == "calibration":
        entry["calibration"][19] = replace(entry["calibration"][19], identity_table=identity)
    else:
        inner = entry["early_stopping_inner"][19]
        assert inner["status"] == "ready"
        inner["training"] = replace(inner["training"], identity_table=identity)
    with pytest.raises(ValueError, match="participant authority"):
        admit_unified_prefixes(fixed, {"4": entry}, **options)


def test_unified_admission_rejects_another_test_recording_with_same_person(admitted_prefix_input):
    from dataclasses import replace

    from experiments.run_cross_dataset_transfer import admit_unified_prefixes

    fixed, prefixes, options = admitted_prefix_input
    other = replace(fixed, X=fixed.X + np.float32(1e-6), verified_cache_attestation=None)
    with pytest.raises(ValueError, match="frozen target cache evidence"):
        admit_unified_prefixes(other, prefixes, **options)
