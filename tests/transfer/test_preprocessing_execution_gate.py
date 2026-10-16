"""Frozen models bind execution evidence while historical models stay readable."""

from copy import deepcopy
from dataclasses import replace

import mne
import numpy as np
import pytest
import torch

from data.epochs import preprocessing_execution_binding
from data.preprocess import execute_preprocessing
from research.contracts import TrainingRunContract, semantic_sha256
from tests.support.checkpoints import candidate_checkpoint_payload
from tests.support.datasets import binary_dataset
from transfer.calibrated import (
    CalibratedPredictor,
    _input_contract,
    load_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.checkpoint import checkpoint_training_contract, load_n2p3_trunk_checkpoint


def _executed_dataset(*, source_rate=250., first_samp=0, duration=20.):
    dataset = binary_dataset(8)
    values = np.random.default_rng(92).normal(0, 2e-6, (dataset.n_channels, round(source_rate * duration)))
    raw = mne.io.RawArray(values, mne.create_info(list(dataset.channel_names), source_rate, "eeg"),
                          first_samp=first_samp, verbose=False)
    onsets = first_samp + np.rint((1 + 2 * np.arange(dataset.n_epochs)) * source_rate).astype(np.int64)
    events = np.column_stack((onsets, np.zeros(dataset.n_epochs, dtype=np.int64), dataset.y + 1))
    result = execute_preprocessing(raw, events, dataset.preprocessing)
    dataset.X = result.data
    dataset.event_timeline = replace(dataset.event_timeline, onset_samples=result.event_samples,
        onset_times_s=result.event_times_s, evidence_available_times_s=result.evidence_available_times_s,
        timing_source=result.timing_source)
    dataset.provenance["preprocessing_execution"] = result.execution_receipt
    dataset.validate()
    return dataset


def _historical(dataset):
    result = deepcopy(dataset)
    result.provenance.pop("preprocessing_execution", None)
    result.validate()
    return result


def _checkpoint(dataset, *, evidence):
    payload = candidate_checkpoint_payload(dataset, source_snapshot_sha256="f" * 64)
    if evidence:
        receipt = deepcopy(dataset.provenance["preprocessing_execution"])
        contract = checkpoint_training_contract(payload)
        contract = replace(contract, preprocessing={**dict(contract.preprocessing),
            **preprocessing_execution_binding(receipt, dataset.preprocessing)})
        payload.update(input_preprocessing_execution=receipt, training_contract=contract.record(),
                       training_contract_digest=contract.digest())
    return payload


def _personal(dataset, *, evidence):
    metadata = {"input_contract": _input_contract(dataset), "model_kind": "target_only_shrinkage_lda",
        "participant_key": dataset.identity_table.record_for("p1").authority_key("source_or_global")}
    if evidence:
        receipt = deepcopy(dataset.provenance["preprocessing_execution"])
        metadata.update(preprocessing_execution=receipt,
            preprocessing_execution_binding=preprocessing_execution_binding(receipt, dataset.preprocessing))
    # A finite fixed estimator state exercises load/validation, without fitting
    # on test inputs or invoking a training path.
    size = dataset.n_channels * dataset.n_times
    state = {"coefficient": torch.zeros(size), "feature_mean": torch.zeros(size),
             "feature_scale": torch.ones(size), "intercept": torch.zeros(1)}
    return CalibratedPredictor(metadata, state)


def test_source_checkpoint_requires_execution_identity_but_allows_new_record_context():
    source = _executed_dataset()
    later = _executed_dataset(source_rate=500., first_samp=500, duration=22.)
    payload = _checkpoint(source, evidence=True)
    load_n2p3_trunk_checkpoint(payload, later, target_subject="p1")
    assert (source.provenance["preprocessing_execution"]["receipt_sha256"]
            != later.provenance["preprocessing_execution"]["receipt_sha256"])
    with pytest.raises(ValueError, match="execution evidence"):
        load_n2p3_trunk_checkpoint(payload, _historical(later), target_subject="p1")


def test_historical_checkpoint_accepts_declared_old_and_new_input_without_inventing_binding():
    dataset = _executed_dataset()
    payload = _checkpoint(dataset, evidence=False)
    original = deepcopy(payload["training_contract"])
    for candidate in (dataset, _historical(dataset)):
        load_n2p3_trunk_checkpoint(payload, candidate, target_subject="p1")
    assert payload["training_contract"] == original
    assert "input_preprocessing_execution" not in payload


@pytest.mark.parametrize("tamper", ["drop_receipt", "substitute_receipt", "physical_declaration"])
def test_checkpoint_rejects_resealed_execution_provenance_that_does_not_match_training_binding(tamper):
    source = _executed_dataset()
    payload = _checkpoint(source, evidence=True)
    if tamper == "drop_receipt":
        payload.pop("input_preprocessing_execution")
    elif tamper == "substitute_receipt":
        payload["input_preprocessing_execution"] = _executed_dataset(duration=22.).provenance["preprocessing_execution"]
    else:
        receipt = payload["input_preprocessing_execution"]
        receipt["declaration"]["h_freq"] = 20.
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = semantic_sha256(receipt)
    with pytest.raises(ValueError, match="execution|input signature"):
        load_n2p3_trunk_checkpoint(payload, source, target_subject="p1")


def test_new_personal_model_keeps_binding_on_save_load_and_admits_new_source_context(tmp_path):
    source = _executed_dataset()
    later = _executed_dataset(source_rate=500., first_samp=500, duration=22.)
    predictor = _personal(source, evidence=True)
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "personal.pt", predictor))
    assert restored.metadata == predictor.metadata
    restored.validate_input(later)
    with pytest.raises(ValueError, match="execution evidence"):
        restored.validate_input(_historical(later))


def test_historical_personal_model_reloads_without_promoting_execution_evidence(tmp_path):
    dataset = _executed_dataset()
    predictor = _personal(dataset, evidence=False)
    path = save_calibrated_predictor(tmp_path / "historical.pt", predictor)
    payload = torch.load(path, weights_only=True)
    payload["schema"] = "n2p3_calibrated_predictor/3"
    torch.save(payload, path)
    restored = load_calibrated_predictor(path)
    for candidate in (dataset, _historical(dataset)):
        restored.validate_input(candidate)
    assert "preprocessing_execution_binding" not in restored.metadata
    assert "preprocessing_execution" not in restored.metadata


def test_personal_model_rejects_receipt_substitution_even_when_outer_metadata_is_resealed(tmp_path):
    source = _executed_dataset()
    predictor = _personal(source, evidence=True)
    path = save_calibrated_predictor(tmp_path / "changed.pt", predictor)
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["preprocessing_execution"] = _executed_dataset(duration=22.).provenance["preprocessing_execution"]
    payload["metadata_digest"] = semantic_sha256(payload["metadata"])
    torch.save(payload, path)
    with pytest.raises(ValueError, match="execution binding"):
        load_calibrated_predictor(path)


def test_incomplete_source_evidence_remains_noncertifying_in_checkpoint_contract():
    source = _executed_dataset()
    receipt = source.provenance["preprocessing_execution"]
    source.provenance["preprocessing_execution"] = {
        "schema": "n2p3_preprocessing_execution_bundle/1", "recordings": [receipt], "unverified_inputs": 1}
    payload = _checkpoint(source, evidence=True)
    contract = TrainingRunContract(**payload["training_contract"])
    assert contract.preprocessing["execution_identity"] is None
    assert contract.preprocessing["execution_provenance_sha256"]
    load_n2p3_trunk_checkpoint(payload, _historical(source), target_subject="p1")
