"""Counterexamples for native-reference, episode authority and future secrecy."""
import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from data import bnci_trial_preparation as preparation
from data.bnci2014_008_candidate import read_bnci2014_008_candidate_record
from data.epochs import (
    preprocessing_spec_from_contract,
    save_epoch_dataset,
    select_scheduled_events,
)
from data.preprocess import RawSampleScope
from tests.support.bnci import bnci_mat_fields


@pytest.fixture(scope="module")
def record():
    fields = bnci_mat_fields()
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    contents = stream.getvalue()
    parsed = read_bnci2014_008_candidate_record(io.BytesIO(contents), source_relative_path="A01.mat",
                                              source_sha256=hashlib.sha256(contents).hexdigest())
    return replace(parsed, eeg_uv=np.random.default_rng(20261012).normal(0, 3, parsed.eeg_uv.shape))


@pytest.fixture(scope="module")
def profile():
    return preprocessing_spec_from_contract(preparation.BNCI_TRIAL_INPUT_PROFILE)


@pytest.fixture(scope="module")
def prepared(record, profile):
    return preparation.build_bnci_trial_prepared_participant(record, preprocessing=profile)


def test_native256_source_profile128_c600_and_all_r10_requested_keys(prepared):
    c, future, ledger = prepared.calibration, prepared.future, prepared.ledger
    assert c.X.shape == (600, 6, 179)
    assert future.X.shape == (3600, 6, 179)
    assert ledger["actual_calibration_class_counts"] == {"0": 500, "1": 100}
    assert ledger["source_sample_rate_hz"] == 256
    assert c.preprocessing.sfreq == 128 and c.preprocessing.baseline_mode == "none"
    assert c.channel_names == preparation.BNCI_TRIAL_CHANNELS
    assert c.channel_mask.tolist() == [True] * 6
    assert c.provenance["source_reference"] == "measured_channel:CZ"
    assert len(prepared.future_decisions) == 30
    assert [row["primary_horizon"] for row in ledger["future_decisions"]] == [10] * 30
    for decision in prepared.future_decisions:
        assert decision.repetitions == 10 and len(decision.event_ids) == 120
        assert decision.schedule_complete and np.all(decision.evidence_indices >= 0)
        assert set(decision.stimulus_indices.tolist()) == set(range(12))
    assert ledger["calibration_last_raw_sample_exclusive"] <= ledger["common_future_first_raw_sample"]
    scopes = ledger["calibration_raw_scopes"] + ledger["future_raw_scopes"]
    assert all(a["stop_sample"] <= b["start_sample"] for a, b in zip(scopes[:-1], scopes[1:], strict=True))
    assert len(c.provenance["preprocessing_execution"]["recordings"]) == 5
    assert len(future.provenance["preprocessing_execution"]["recordings"]) == 30


def test_projection_keeps_weak_native_difference_until_float64_preprocessing(record, profile, monkeypatch):
    original = preparation.execute_preprocessing
    seen = []
    # A weak measured difference would disappear in a premature float32 cast.
    values = record.eeg_uv * 1e-3 + 1e8
    weak = replace(record, eeg_uv=values)
    expected = (values[:, [0, 4, 2, 5, 7, 3]] - values[:, 1, None]).T * 1e-6

    def capture(raw, events, spec, **kwargs):
        actual = raw.get_data()
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-14)
        assert np.count_nonzero(actual) > len(actual.T)
        seen.append((raw.info["sfreq"], spec.sfreq, kwargs["raw_scope"].name))
        return original(raw, events, spec, **kwargs)

    monkeypatch.setattr(preparation, "execute_preprocessing", capture)
    result = preparation.build_bnci_trial_prepared_participant(weak, preprocessing=profile)
    assert len(seen) == 35 and all(row[:2] == (256, 128) for row in seen)
    assert result.calibration.X.dtype == np.float32


def test_future_signal_perturbation_cannot_change_authorized_calibration(record, profile, prepared):
    values = record.eeg_uv.copy()
    first = prepared.ledger["common_future_first_raw_sample"]
    values[first:] += np.random.default_rng(314).normal(0, 1e4, values[first:].shape)
    perturbed = preparation.build_bnci_trial_prepared_participant(replace(record, eeg_uv=values), preprocessing=profile)
    np.testing.assert_array_equal(perturbed.calibration.X, prepared.calibration.X)
    np.testing.assert_array_equal(perturbed.calibration.y, prepared.calibration.y)
    assert perturbed.calibration.provenance["preprocessing_execution"] == prepared.calibration.provenance["preprocessing_execution"]
    assert not np.array_equal(perturbed.future.X, prepared.future.X)


def test_whole_future_episode_purge_keeps_original_raw_truth_and_minus_one_mapping(record, profile, monkeypatch):
    original_scopes = preparation.bnci_trial_episode_scopes(record, profile)
    scopes = list(original_scopes)
    first = scopes[5]
    scopes[5] = RawSampleScope(first.name, first.start_sample, first.start_sample + 50, first.event_indices)
    monkeypatch.setattr(preparation, "bnci_trial_episode_scopes", lambda *_: tuple(scopes))
    result = preparation.build_bnci_trial_prepared_participant(record, preprocessing=profile)
    decision = result.future_decisions[0]
    assert len(decision.event_ids) == 120 and np.all(decision.evidence_indices == -1)
    assert len(result.future_decisions) == 30 and result.future.event_timeline.n_events == 3600
    assert result.ledger["parent_scoring_truth"][first.name] == int(record.target_row[600] * 6 + record.target_col[600])
    assert first.name not in set(result.future.metadata.selection_id)
    later = result.future_decisions[1]
    np.testing.assert_array_equal(later.evidence_indices, np.arange(120))


def test_future_labels_and_target_fields_are_absent_even_when_raw_truth_changes(record, profile, prepared):
    rows = record.selection_index >= 5
    changed_row, changed_col = record.target_row.copy(), record.target_col.copy()
    changed_row[rows] = (changed_row[rows] + 1) % 6
    changed_col[rows] = (changed_col[rows] + 2) % 6
    alternative = preparation.build_bnci_trial_prepared_participant(
        replace(record, target_row=changed_row, target_col=changed_col), preprocessing=profile)
    assert alternative.future.y is None
    assert not preparation._FUTURE_TARGET_FIELDS & set(alternative.future.metadata)
    assert np.all(alternative.future.event_timeline.target_candidate_ids == "")
    assert set(alternative.future.event_timeline.stimulus_ids) == set(range(12))
    np.testing.assert_array_equal(alternative.future.X, prepared.future.X)
    assert [row.record() for row in alternative.future_decisions] == [row.record() for row in prepared.future_decisions]
    assert alternative.ledger["parent_scoring_truth"] != prepared.ledger["parent_scoring_truth"]


def _seal(tmp_path, prepared):
    ledger = dict(prepared.ledger)
    for role in ("calibration", "future"):
        path = save_epoch_dataset(tmp_path / f"{role}.npz", getattr(prepared, role), compressed=False)
        ledger[f"{role}_cache"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "preparation.json").write_text(json.dumps(ledger))
    return ledger


def test_sealed_pair_roundtrip_and_cache_tamper(tmp_path, prepared):
    _seal(tmp_path, prepared)
    loaded = preparation.load_bnci_trial_prepared_participant(tmp_path)
    np.testing.assert_array_equal(loaded.calibration.X, prepared.calibration.X)
    assert loaded.ledger["parent_scoring_truth"] == prepared.ledger["parent_scoring_truth"]
    assert loaded.future.y is None
    (tmp_path / "future.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sealed preparation"):
        preparation.load_bnci_trial_prepared_participant(tmp_path)


def test_sealed_descriptor_must_match_actual_evidence_axis(tmp_path, prepared):
    ledger = json.loads(json.dumps(_seal(tmp_path, prepared)))
    ledger["future_decisions"][0]["membership_decision"]["evidence_indices"][0] = 2000
    (tmp_path / "preparation.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="physical decision ledger"):
        preparation.load_bnci_trial_prepared_participant(tmp_path)


@pytest.mark.parametrize(("field", "replacement"), [
    ("episode_id", "different-original-selection"),
    ("scheduled_flashes", 119),
    ("retained_flashes", 119),
    ("complete_raw_repetitions", 8),
    ("primary_horizon", 8),
    ("terminal_partial_flashes", 2),
    ("terminal_censor", True),
    ("primary_horizon", 10.0),
    ("terminal_censor", 0),
])
def test_sealed_outer_descriptor_must_match_actual_schedule(tmp_path, prepared, field, replacement):
    ledger = json.loads(json.dumps(_seal(tmp_path, prepared)))
    ledger["future_decisions"][0][field] = replacement
    (tmp_path / "preparation.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="physical decision ledger"):
        preparation.load_bnci_trial_prepared_participant(tmp_path)


def test_descriptor_compiler_keeps_raw_r10_when_all_episode_eeg_is_missing(prepared):
    future = prepared.future
    scoped = select_scheduled_events(future,
        future.event_timeline.selection_ids != future.event_timeline.selection_ids[0],
        reason="entire_future_episode_eeg_unavailable_counterexample", preserve_schedule=True)
    descriptors = preparation._decision_descriptors(scoped, preparation._decisions(scoped))
    assert descriptors[0]["scheduled_flashes"] == 120
    assert descriptors[0]["retained_flashes"] == 0
    assert descriptors[0]["complete_raw_repetitions"] == descriptors[0]["primary_horizon"] == 10
    assert descriptors[0]["terminal_censor"] is False


def test_descriptor_compiler_cannot_invent_timeline_completeness(prepared):
    future = replace(prepared.future, event_timeline=replace(prepared.future.event_timeline, complete=False))
    with pytest.raises(ValueError, match="actual future schedule"):
        preparation._decision_descriptors(future, preparation._decisions(future))


def test_source_none_baseline_is_not_replaced_by_product_mean_only(record, profile):
    with pytest.raises(ValueError, match="baseline"):
        preparation.build_bnci_trial_prepared_participant(record, preprocessing=replace(profile, baseline_mode="mean_only"))
