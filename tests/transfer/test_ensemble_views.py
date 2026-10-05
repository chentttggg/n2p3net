from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from data.candidate_task import MembershipDecision
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT, BRAIN_SYNC_EEG_CHANNELS
from data.domain import adapt_observed_montage
from data.epochs import load_epoch_dataset, preprocessing_spec_from_contract, save_epoch_dataset
from data.identity import origin_subject_key
from data.preprocess import filter_bidirectional_segment, transform_epoch_samples
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble_views import (
    CalibrationSelection,
    PredictionDecisionView,
    SelectionEpochProfile,
    build_calibration_epoch_dataset,
    prepare_bnci_calibration_groups,
)


def decision_view():
    decision = MembershipDecision.direct(
        ("a", "b", "b", "a"), ("a", "b"),
        event_ids=("flash0", "flash1", "flash2", "flash3"),
        repetition_indices=np.array([0, 0, 1, 1]), repetitions=2,
    )
    samples = np.random.default_rng(31).normal(0, 1e-6, (550, 3))
    return PredictionDecisionView(
        record_id="record01", participant_key=origin_subject_key("study", "person01"),
        decision_id="selection01", samples=samples, onsets=np.array([51, 115, 179, 243]),
        decision=decision, source_sha256="a" * 64, raw_scope=(1000, 1550),
        sample_rate_hz=256.0, channel_names=("Fz", "Cz", "Pz"),
        channel_types=("EEG", "EEG", "EEG"), reference="right earlobe",
        source_relative_path="record.mat", source_dataset_id="study", source_subject_id="person01",
        available_sample=1550,
    )


def test_prediction_has_public_membership_and_immutable_samples_without_truth():
    view = decision_view()
    original_digest = view.physical_support_digest
    calibration = CalibrationSelection(view, np.array([True, False, False, True]), "a")
    assert calibration.view is view
    assert calibration.target_candidate == "a"
    assert not any(hasattr(view, field) for field in ("labels", "target_candidate", "truth"))
    for values in (view.samples, view.onsets, view.decision.membership, calibration.labels):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    swapped = CalibrationSelection(view, np.array([False, True, True, False]), "b")
    assert swapped.view.physical_support_digest == original_digest
    with pytest.raises(ValueError, match="membership"):
        CalibrationSelection(view, np.array([False, False, False, True]), "a")
    with pytest.raises(ValueError, match="actual raw scope"):
        replace(view, raw_scope=(1000, 1551), available_sample=1551)
    with pytest.raises(ValueError, match="EEG"):
        replace(view, channel_types=("EEG", "EOG", "EEG"))


def test_selection_epochs_match_shared_transform_and_reject_unobserved_endpoint():
    view = decision_view()
    contract = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                       name="actual_closed_grid", tmin_ms=-51 / 256 * 1000,
                       tmax_ms=306 / 256 * 1000, baseline_mode="none")
    spec = preprocessing_spec_from_contract(contract)
    profile = SelectionEpochProfile.from_preprocessing(
        spec, source_sample_rate_hz=256, channel_names=view.channel_names,
        reference=view.reference,
    )
    X, mask, receipt = profile.transform(view)
    # Independent slicing oracle: actual source samples -51 through +306.
    executed = filter_bidirectional_segment(
        np.asarray(profile.operator.sos), view.samples.T,
        padding_limit=profile.operator.padding_limit,
    )
    epochs = np.stack([executed.data[:, onset - 51:onset + 307] for onset in view.onsets])
    expected = transform_epoch_samples(epochs, 256, spec).astype(np.float32)
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(mask, np.ones((4, 3), dtype=bool))
    assert X.shape == (4, 3, 178)
    assert receipt["epoch_geometry"]["source_samples"] == 358
    assert receipt["epoch_geometry"]["fft_resampled_samples"] == 179
    assert receipt["physical_support_digest"] == view.physical_support_digest
    np.testing.assert_array_equal(receipt["filter"]["left_state"], executed.left_state)
    np.testing.assert_array_equal(receipt["filter"]["right_state"], executed.right_state)
    assert receipt["available_sample"] == view.available_sample
    source_profile = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        source_sample_rate_hz=256, channel_names=view.channel_names, reference=view.reference,
    )
    with pytest.raises(ValueError, match="observed raw support"):
        source_profile.transform(view)


@pytest.fixture(scope="module")
def bnci_payload():
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(34).normal(0, 5, fields["X"].shape).astype(np.float32)
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def test_bnci_complete_groups_share_original_bytes_and_account_unused_authorized_labels(bnci_payload, tmp_path):
    payload, digest = bnci_payload
    path = tmp_path / "A01.mat"
    path.write_bytes(payload)
    prepared = prepare_bnci_calibration_groups(path, source_sha256=digest)
    assert len(prepared.groups) == 6
    assert prepared.ledger["used_label_count"] == 720
    assert prepared.ledger["unused_authorized_labels"] == 80
    assert prepared.ledger["unused_authorized_label_indices"] == tuple(range(720, 800))
    for index, group in enumerate(prepared.groups):
        view = group.view
        assert view.participant_key == origin_subject_key("BNCI2014-008", "A01")
        assert view.source_sha256 == digest
        assert view.decision.schedule_complete
        np.testing.assert_array_equal(view.decision.membership.sum(axis=0), np.full(36, 20))
        assert view.available_sample == view.raw_scope[1]
        assert len(group.labels) == 120
        if index:
            assert prepared.groups[index - 1].view.raw_scope[1] < view.raw_scope[0]
    with pytest.raises(ValueError, match="original raw bytes"):
        prepare_bnci_calibration_groups(path, source_sha256="0" * 64)
    with pytest.raises(ValueError, match="complete independent groups"):
        prepare_bnci_calibration_groups(path, source_sha256=digest, authorized_labels=200)
    smaller = prepare_bnci_calibration_groups(path, source_sha256=digest, authorized_labels=200, group_count=1)
    assert smaller.ledger["used_label_count"] == 120
    assert smaller.ledger["unused_authorized_labels"] == 80
    with pytest.raises(ValueError, match="next unknown flash"):
        prepare_bnci_calibration_groups(path, source_sha256=digest, response_seconds=20)


def test_calibration_dataset_preserves_real_identity_lineage_membership_and_179_samples(bnci_payload, tmp_path):
    payload, digest = bnci_payload
    path = tmp_path / "A01.mat"
    path.write_bytes(payload)
    prepared = prepare_bnci_calibration_groups(path, source_sha256=digest, response_seconds=308 / 256)
    first = prepared.groups[0].view
    profile = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        source_sample_rate_hz=256, channel_names=first.channel_names, reference=first.reference,
    )
    dataset = build_calibration_epoch_dataset(prepared.groups[:2], profile)
    dataset.validate(require_labels=True)
    assert dataset.X.shape == (240, 8, 179)
    assert dataset.identity_table.authority_keys() == (first.participant_key,)
    assert dataset.lineage.entities[0].operation == "source_ingress"
    assert dataset.lineage.entities[-1].operation == "selection_epoch_transform"
    assert dataset.event_timeline.event_ids[0] == "BNCI2014-008:A01:0"
    assert dataset.event_timeline.event_ids[-1] == "BNCI2014-008:A01:239"
    assert dataset.provenance["source_file_sha256"] == digest
    assert dataset.provenance["selection_epoch_execution"][0]["physical_support_digest"] == first.physical_support_digest
    assert "preprocessing_execution" not in dataset.provenance  # No invented mne_raw receipt.
    for group_index in range(2):
        group = prepared.groups[group_index]
        rows = dataset.event_timeline.group_ids == group.view.decision_id
        expected, expected_mask, _ = profile.transform(group.view)
        np.testing.assert_array_equal(dataset.X[rows], expected)
        np.testing.assert_array_equal(dataset.trial_channel_mask[rows], expected_mask)
        np.testing.assert_array_equal(dataset.y[rows], group.labels)
        np.testing.assert_array_equal(dataset.event_timeline.onset_samples[rows],
                                      group.view.onsets + group.view.raw_scope[0])
        np.testing.assert_array_equal(dataset.event_timeline.evidence_available_times_s[rows],
                                      np.full(120, group.view.available_sample / 256))
    saved = save_epoch_dataset(tmp_path / "actual_selection_epochs.npz", dataset)
    reloaded = load_epoch_dataset(saved, require_labels=True)
    np.testing.assert_array_equal(dataset.X, reloaded.X)
    np.testing.assert_array_equal(dataset.y, reloaded.y)
    assert dataset.identity_table.digest() == reloaded.identity_table.digest()
    assert dataset.lineage.entity_digest == reloaded.lineage.entity_digest


def test_profile_rejects_declared_edge_policy_that_its_finite_operator_cannot_execute():
    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   zero_phase_edge_policy="legacy")
    with pytest.raises(ValueError, match="edge policy"):
        SelectionEpochProfile.from_preprocessing(
            spec, source_sample_rate_hz=256, channel_names=("Fz", "Cz", "Pz"), reference="right earlobe",
        )


def test_view_profile_roundtrip_and_truth_mutation_preserve_exact_prediction_values():
    view = decision_view()
    contract = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                       name="actual_closed_grid", tmin_ms=-51 / 256 * 1000,
                       tmax_ms=306 / 256 * 1000)
    profile = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(contract), source_sample_rate_hz=256,
        channel_names=view.channel_names, reference=view.reference,
    )
    restored_view = PredictionDecisionView.from_record(json.loads(json.dumps(view.record())), samples=view.samples)
    restored_profile = SelectionEpochProfile.from_record(json.loads(json.dumps(profile.record())))
    original, mask, receipt = profile.transform(view)
    replay, replay_mask, replay_receipt = restored_profile.transform(restored_view)
    np.testing.assert_array_equal(original, replay)
    np.testing.assert_array_equal(mask, replay_mask)
    assert receipt == replay_receipt
    authorized = CalibrationSelection(view, np.array([True, False, False, True]), "a")
    mutated = CalibrationSelection(view, np.array([False, True, True, False]), "b")
    np.testing.assert_array_equal(profile.transform(authorized.view)[0], profile.transform(mutated.view)[0])
    assert authorized.record()["labels_sha256"] != mutated.record()["labels_sha256"]
    with pytest.raises(TypeError, match="unlabeled"):
        profile.transform(authorized)
    with pytest.raises(ValueError, match="actual physical or numerical"):
        PredictionDecisionView.from_record(view.record(), samples=view.samples * 2)


def test_numeric_contrast_changes_real_input_without_rewriting_physical_support():
    view = decision_view()
    contract = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                       name="actual_closed_grid", tmin_ms=-51 / 256 * 1000,
                       tmax_ms=306 / 256 * 1000, baseline_mode="none")
    none = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(contract), source_sample_rate_hz=256,
        channel_names=view.channel_names, reference=view.reference,
    )
    mean_only = replace(none, preprocessing=replace(none.preprocessing, baseline_mode="mean_only"))
    raw_before = view.samples.copy()
    X_none, _, receipt_none = none.transform(view)
    X_mean, _, receipt_mean = mean_only.transform(view)
    assert not np.array_equal(X_none, X_mean)
    assert receipt_none["physical_support_digest"] == receipt_mean["physical_support_digest"]
    assert receipt_none["numerical_input_digest"] != receipt_mean["numerical_input_digest"]
    np.testing.assert_array_equal(view.samples, raw_before)
    later = replace(view, available_sample=view.available_sample + 1)
    assert later.physical_support_digest != view.physical_support_digest


def test_candidate_axis_permutation_never_uses_authorized_truth_as_numeric_input():
    view = decision_view()
    swapped = replace(view, decision=replace(
        view.decision, candidate_vocabulary=("b", "a"), membership=view.decision.membership[:, ::-1],
    ))
    left = CalibrationSelection(view, np.array([True, False, False, True]), "a")
    right = CalibrationSelection(swapped, left.labels, "a")
    contract = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                       name="actual_closed_grid", tmin_ms=-51 / 256 * 1000,
                       tmax_ms=306 / 256 * 1000)
    profile = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(contract), source_sample_rate_hz=256,
        channel_names=view.channel_names, reference=view.reference,
    )
    np.testing.assert_array_equal(profile.transform(left.view)[0], profile.transform(right.view)[0])
    # The frozen ordered vocabulary belongs to the physical receipt. Reindexing
    # cannot reuse a cached evidence vector under the old ordered vocabulary.
    assert view.physical_support_digest != swapped.physical_support_digest


def test_explicit_cz_projection_matches_existing_montage_owner_without_native_relabeling(bnci_payload, tmp_path):
    payload, digest = bnci_payload
    path = tmp_path / "A01.mat"
    path.write_bytes(payload)
    groups = prepare_bnci_calibration_groups(path, source_sha256=digest, response_seconds=308 / 256).groups[:1]
    first = groups[0].view
    native = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        source_sample_rate_hz=256, channel_names=first.channel_names, reference=first.reference,
    )
    six = tuple(name for name in BRAIN_SYNC_EEG_CHANNELS if name != "Cz")
    cz_profile = replace(native, channel_names=six, reference_channel="Cz")
    existing = adapt_observed_montage(build_calibration_epoch_dataset(groups, native), six, reference_channel="Cz")
    actual, mask, receipt = cz_profile.transform(first)
    np.testing.assert_array_equal(actual, existing.X)
    np.testing.assert_array_equal(mask, existing.trial_channel_mask)
    assert first.reference == cz_profile.reference == "right earlobe"
    assert receipt["output"]["source_reference"] == "right earlobe"
    assert receipt["output"]["reference"] == existing.provenance["source_reference"] == "measured_channel:CZ"
    assert receipt["physical_support_digest"] == first.physical_support_digest
    with pytest.raises(ValueError, match="absent from actual observations"):
        replace(cz_profile, channel_names=(*six, "FCz")).transform(first)
    with pytest.raises(ValueError, match="actual native rate or acquisition reference"):
        replace(native, reference="left earlobe").transform(first)


def test_previous_unknown_response_and_inadequate_label_prefix_are_rejected(bnci_payload, tmp_path):
    payload, digest = bnci_payload
    path = tmp_path / "A01.mat"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="complete independent groups"):
        prepare_bnci_calibration_groups(path, source_sha256=digest, authorized_labels=119, group_count=1)
    fields = bnci_mat_fields()
    # Move the entire authentic-shape acquisition schedule later, leaving real
    # pre-roll observations for the first group. A nine-second preroll then
    # touches the previous selection's declared ERP support in group two.
    for field_name in ("y", "y_stim"):
        values = fields[field_name]
        fields[field_name] = np.concatenate((np.zeros_like(values[:4000]), values[:-4000]))
    fields["trial"] = fields["trial"] + 4000
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    shifted = stream.getvalue()
    path.write_bytes(shifted)
    with pytest.raises(ValueError, match="previous unknown flash response"):
        prepare_bnci_calibration_groups(path, source_sha256=hashlib.sha256(shifted).hexdigest(),
                                         pre_roll_seconds=9)


def test_constructor_rejects_fabricated_179_metadata_and_filter_override():
    contract = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                       name="actual_closed_grid", tmin_ms=-51 / 256 * 1000, tmax_ms=306 / 256 * 1000)
    spec = preprocessing_spec_from_contract(contract)
    profile = SelectionEpochProfile.from_preprocessing(
        spec, source_sample_rate_hz=256, channel_names=("Fz", "Cz", "Pz"), reference="right earlobe",
    )
    with pytest.raises(ValueError, match="178 samples"):
        replace(profile, preprocessing=replace(spec, n_times=179))
    with pytest.raises(ValueError, match="filter differs"):
        replace(profile, operator=replace(profile.operator, padding_limit=profile.operator.padding_limit + 1))
    with pytest.raises(ValueError, match="availability"):
        replace(decision_view(), available_sample=1549)
