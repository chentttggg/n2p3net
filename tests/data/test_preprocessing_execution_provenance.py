"""Execution receipts survive caches and derived views without inventing evidence."""

from copy import deepcopy
from dataclasses import replace

import mne
import numpy as np
import pytest

from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.dataset import build_subject
from data.epochs import (
    bind_named_preprocessing_contract,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    preprocessing_execution_identity,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
    select_epoch_channels,
    select_scheduled_events,
)
from data.manifest import _subject_epoch_dataset
from research.contracts import semantic_sha256

SPEC = replace(preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
               name="synthetic-execution-provenance")


def _recording(recording_id, *, rate=250., first_samp=0, duration=8.):
    values = np.random.default_rng(31).normal(0, 2e-6, (3, round(rate * duration)))
    raw = mne.io.RawArray(values, mne.create_info(["Fz", "Cz", "Pz"], rate, "eeg"),
                          first_samp=first_samp, verbose=False)
    onsets = first_samp + np.rint(np.array([2., 4., 6.]) * rate).astype(np.int64)
    events = np.column_stack((onsets, np.zeros(3, dtype=np.int64), np.array([1, 2, 1])))
    subject = build_subject(raw, events, np.array([0, 1, 0]), preprocessing=SPEC,
        subject_id=recording_id, metadata={"reference": "synthetic-reference", "session": recording_id,
            "run": "run", "dataset_id": "execution-synthetic"})
    return _subject_epoch_dataset(subject, SPEC, dataset_name="execution-synthetic")


def _merge(*datasets):
    return concatenate_epoch_datasets(datasets, name="execution-synthetic",
                                      provenance={"source_reference": "synthetic-reference"})


@pytest.mark.parametrize("validation", ["full", "attested"])
def test_real_builder_execution_receipt_round_trips_epoch_cache(tmp_path, validation):
    dataset = _recording("raw-a", first_samp=125)
    receipt = deepcopy(dataset.provenance["preprocessing_execution"])
    assert "preprocessing_execution" not in dataset.metadata
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "cache.npz", dataset),
                                  validation=validation, require_labels=True)
    assert restored.provenance["preprocessing_execution"] == receipt
    assert preprocessing_execution_identity(restored) == preprocessing_execution_identity(dataset)
    np.testing.assert_array_equal(restored.X, dataset.X)
    np.testing.assert_array_equal(restored.event_timeline.onset_samples, [625, 1125, 1625])


def test_concat_preserves_distinct_record_contexts_without_requiring_identical_source_grids(tmp_path):
    first = _recording("raw-a", rate=250., first_samp=125, duration=8.)
    second = _recording("raw-b", rate=500., first_samp=500, duration=10.)
    merged = _merge(first, second)
    bundle = merged.provenance["preprocessing_execution"]
    assert bundle["unverified_inputs"] == 0
    assert len(bundle["recordings"]) == 2
    contexts = {(r["source_context"]["sample_rate_hz"], r["source_context"]["first_samp"],
                 r["source_context"]["n_times"]) for r in bundle["recordings"]}
    assert contexts == {(250., 125, 2000), (500., 500, 5000)}
    assert preprocessing_execution_identity(merged) == preprocessing_execution_identity(first)
    assert preprocessing_execution_identity(first) == preprocessing_execution_identity(second)
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "merged.npz", merged))
    assert restored.provenance["preprocessing_execution"] == bundle


def test_mixing_historical_inputs_preserves_unknown_count_through_nested_concat():
    fresh = _recording("fresh")
    old_a, old_b = _recording("old-a"), _recording("old-b")
    old_a.provenance.pop("preprocessing_execution")
    old_b.provenance.pop("preprocessing_execution")
    mixed = _merge(fresh, old_a)
    assert mixed.provenance["preprocessing_execution"]["unverified_inputs"] == 1
    assert preprocessing_execution_identity(mixed) is None
    nested = _merge(mixed, old_b)
    bundle = nested.provenance["preprocessing_execution"]
    assert bundle["unverified_inputs"] == 2
    assert bundle["recordings"] == [fresh.provenance["preprocessing_execution"]]
    assert preprocessing_execution_identity(nested) is None
    # Caller metadata is not evidence that either historical child executed it.
    historical = concatenate_epoch_datasets([old_a, old_b], name="execution-synthetic",
        provenance={"source_reference": "synthetic-reference",
                    "preprocessing_execution": fresh.provenance["preprocessing_execution"]})
    assert "preprocessing_execution" not in historical.provenance
    assert preprocessing_execution_identity(historical) is None


def test_row_and_channel_selection_keep_parent_receipt_and_explicit_lineage(tmp_path):
    dataset = _recording("parent")
    receipt = deepcopy(dataset.provenance["preprocessing_execution"])
    selected = select_scheduled_events(dataset, np.array([True, False, True]), reason="synthetic held-out scope")
    selected = select_epoch_channels(selected, ["Pz", "Fz"])
    assert selected.X.shape == (2, 2, SPEC.n_times)
    assert selected.provenance["preprocessing_execution"] == receipt
    assert receipt["output"]["shape"] == [3, 3, SPEC.n_times]
    assert selected.lineage is not None
    assert preprocessing_execution_identity(selected) == preprocessing_execution_identity(dataset)
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "subset.npz", selected))
    assert restored.provenance["preprocessing_execution"] == receipt
    np.testing.assert_array_equal(restored.X, dataset.X[[0, 2]][:, [2, 0]])


@pytest.mark.parametrize("mutation", ["dataset_frequency", "receipt_frequency", "receipt_grid", "digest"])
def test_cache_admission_rejects_tampered_physical_execution(mutation, tmp_path):
    dataset = _recording("tamper")
    receipt = dataset.provenance["preprocessing_execution"]
    if mutation == "dataset_frequency":
        dataset.preprocessing = replace(dataset.preprocessing, l_freq=.5)
    elif mutation == "receipt_frequency":
        receipt["declaration"]["l_freq"] = .5
    elif mutation == "receipt_grid":
        receipt["epoch_grid"]["source_n_times"] += 1
    else:
        receipt["receipt_sha256"] = "0" * 64
    if mutation in {"receipt_frequency", "receipt_grid"}:
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = semantic_sha256(receipt)
    with pytest.raises(ValueError, match="execution"):
        save_epoch_dataset(tmp_path / "invalid.npz", dataset)


def test_historical_name_binding_keeps_executed_name_and_valid_receipt(tmp_path):
    source = _recording("named")
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "source.npz", source), validation="attested")
    receipt = deepcopy(restored.provenance["preprocessing_execution"])
    bound = bind_named_preprocessing_contract(restored, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    assert bound.preprocessing.name == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name
    assert bound.provenance["preprocessing_execution"] == receipt
    assert receipt["declaration"]["name"] == SPEC.name
    np.testing.assert_array_equal(bound.X, source.X)
    assert preprocessing_execution_identity(bound) == preprocessing_execution_identity(source)
