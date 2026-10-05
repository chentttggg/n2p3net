"""Recovery cost reports preserve frozen weights and distinguish input pollution."""

import copy
import json

import numpy as np
from scipy.signal import butter

from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.recovery import EpochStateRecovery
from research.contracts import semantic_sha256

BANDPASS_PREPROCESSING = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)

def _model(envelope=None):
    return EpochStateRecovery.from_envelope(
        butter(4, [.1, 30], btype="bandpass", fs=250, output="sos"), 250,
        BANDPASS_PREPROCESSING, ("Cz", "Pz"),
        np.full((2, 8), 1e-4) if envelope is None else envelope,
        tolerance_v=1e-6, fit_provenance={"scope": "known_synthetic_state_envelope"},
    )


def test_certificate_binds_frozen_model_without_changing_serialized_state():
    model = _model()
    before = copy.deepcopy(model.record())
    certificate = model.recovery_certificate()
    assert certificate["schema"] == "n2p3_epoch_recovery_certificate/1"
    assert certificate["recovery_model_digest"] == semantic_sha256(before)
    assert model.record() == before
    assert semantic_sha256(model.record()) == semantic_sha256(before)
    assert model.recovery_certificate() == certificate
    assert json.loads(json.dumps(certificate, allow_nan=False)) == certificate


def test_certificate_checks_the_discrete_release_boundary_in_final_epoch_space():
    model = _model()
    certificate = model.recovery_certificate()
    for channel, row in enumerate(certificate["channels"]):
        wait = model.recovery_samples()[channel]
        assert wait > 0 and row["release_samples"] == wait
        assert row["release_seconds"] == wait / 250
        assert row["state_error_bound_at_release_v"] == model.epoch_error_bound(wait)[channel]
        assert row["state_error_bound_at_previous_sample_v"] == model.epoch_error_bound(wait - 1)[channel]
        assert row["previous_sample"] == wait - 1
        assert row["release_meets_tolerance"] and row["previous_sample_exceeds_tolerance"]
        assert model.epoch_error_bound(wait + 1000)[channel] <= row["state_error_bound_at_release_v"]


def test_zero_state_uncertainty_has_no_invented_previous_sample():
    certificate = _model(np.zeros((2, 8))).recovery_certificate()
    for row in certificate["channels"]:
        assert row["release_samples"] == 0 and row["release_seconds"] == 0
        assert row["state_error_bound_at_release_v"] == 0
        assert row["previous_sample"] is None
        assert row["state_error_bound_at_previous_sample_v"] is None
        assert row["previous_sample_exceeds_tolerance"] is None
        assert row["release_meets_tolerance"]
    assert certificate["continuing_input_contamination"]["covered"] is False


def test_certificate_distinguishes_epoch_start_stimulus_and_complete_sample_support():
    model = _model()
    certificate = model.recovery_certificate()
    assert certificate["state_error_bound"]["age_endpoint"] == "first_source_sample_of_epoch_not_stimulus_onset"
    support = certificate["epoch_source_support"]
    assert support["source_samples_including_both_endpoints"] == 351
    assert support["first_to_last_sample_seconds"] == 1.4
    assert support["stimulus_onset_offset_from_epoch_start_samples"] == 50
    assert "quality_confirmation" in support["availability"]
    for row in certificate["channels"]:
        assert row["earliest_stimulus_onset_samples_after_reset"] == row["release_samples"] + 50
        assert row["earliest_complete_epoch_samples_after_reset"] == row["release_samples"] + 350


def test_projected_reference_certificate_uses_both_native_uncertainty_envelopes():
    native = _model()
    projected = native.project(np.array([[-1., 1.]]), ("Pz",))
    certificate = projected.recovery_certificate()
    assert certificate["recovery_model_digest"] == semantic_sha256(projected.record())
    assert certificate["recovery_model_digest"] != semantic_sha256(native.record())
    assert certificate["channels"][0]["channel"] == "Pz"
    assert certificate["channels"][0]["release_samples"] > max(native.recovery_samples())
    assert "final_reference_and_frozen_eog_dependencies_are_projected_before_using_this_certificate" in certificate["conditions"]
