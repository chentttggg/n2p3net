"""Raw scopes isolate finite filtering before labels, QC fitting and epoching."""

from copy import deepcopy
from dataclasses import replace

import mne
import numpy as np
import pytest

from data.dataset import build_subject
from data.epochs import PreprocessingSpec, validate_preprocessing_execution
from data.preprocess import execute_preprocessing


def _record():
    raw = mne.io.RawArray(
        np.random.default_rng(47).normal(0, 4e-6, (3, 6500)),
        mne.create_info(["Fz", "Cz", "Pz"], 250, "eeg"), first_samp=750, verbose=False,
    )
    labels = np.arange(20) % 2
    events = np.column_stack((750 + (np.arange(20) + 2) * 250, np.zeros(20, int), labels + 1))
    spec = PreprocessingSpec(filter_phase="zero", zero_phase_edge_policy="mne_reflect_limited",
                             notch_freqs_hz=(50.,))
    return raw, events, labels, spec


def test_scoped_preprocessing_ignores_outside_waveforms_and_unauthorized_tail_labels():
    from data.preprocess import prefix_raw_scopes

    raw, events, labels, spec = _record()
    calibration, _ = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    first = build_subject(raw, events, labels, preprocessing=spec, raw_scope=calibration)
    changed = raw.copy()
    changed._data[:, calibration.stop_sample - raw.first_samp:] = np.nan
    changed_labels = labels.copy()
    changed_labels[5:] = 999
    second = build_subject(changed, events, changed_labels, preprocessing=spec, raw_scope=calibration)
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.labels, labels[:5])
    np.testing.assert_array_equal(second.labels, labels[:5])
    assert first.event_timeline.n_events == 5  # A later onset inside the tail context is not labelled.
    assert first.event_timeline.event_ids.tolist() == second.event_timeline.event_ids.tolist()
    receipt = first.metadata["preprocessing_execution"]
    assert receipt["source_context"]["first_samp"] == calibration.start_sample
    assert receipt["source_context"]["n_times"] == calibration.stop_sample - calibration.start_sample
    validate_preprocessing_execution(receipt, spec)


def test_fixed_heldout_raw_scope_keeps_absolute_events_and_original_ids_across_k():
    from data.preprocess import prefix_raw_scopes

    raw, events, labels, spec = _record()
    _, heldout_a = prefix_raw_scopes(raw, events, spec, calibration_events=2, maximum_calibration_events=8)
    _, heldout_b = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    assert heldout_a == heldout_b
    result = execute_preprocessing(raw, events, spec, raw_scope=heldout_a)
    assert result.scheduled_event_indices.tolist() == list(range(8, 20))
    np.testing.assert_array_equal(result.event_samples, events[8:, 0])
    assert result.event_statuses[0] == "boundary_dropped"
    assert result.event_evidence_indices[0] == -1
    assert (result.source_epoch_start_samples[result.event_evidence_indices >= 0] >= heldout_a.start_sample).all()
    assert (result.source_epoch_end_samples[result.event_evidence_indices >= 0] < heldout_a.stop_sample).all()
    subject = build_subject(raw, events, labels, preprocessing=spec, raw_scope=heldout_a)
    assert subject.event_timeline.event_ids[0].endswith(":8")
    assert subject.event_timeline.event_ids[-1].endswith(":19")


def test_labels_outside_authorized_candidate_scope_may_remain_unknown():
    from data.preprocess import prefix_raw_scopes

    raw, events, labels, spec = _record()
    scope, _ = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    targets = np.where(np.arange(len(events)) < 5, "2", "")
    subject = build_subject(raw, events, labels, preprocessing=spec, raw_scope=scope,
        candidate_ids=events[:, 2].astype(str), target_candidate_ids=targets,
        repetition_indices=np.arange(len(events)) // 2)
    assert subject.event_timeline.target_candidate_ids.tolist() == ["2"] * 5
    np.testing.assert_array_equal(subject.labels, labels[:5])


def test_inner_scope_rebuild_and_qc_fit_ignore_the_rest_of_authorized_calibration():
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
    from data.preprocess import crop_raw_scope, prefix_raw_scopes

    raw, events, _, spec = _record()
    outer, _ = prefix_raw_scopes(raw, events, spec, calibration_events=10, maximum_calibration_events=12)
    inner, validation = prefix_raw_scopes(raw, events, spec, calibration_events=6, parent_scope=outer)
    assert inner.stop_sample == validation.start_sample
    assert validation.stop_sample == outer.stop_sample
    original = execute_preprocessing(raw, events, spec, raw_scope=inner)
    final = execute_preprocessing(raw, events, spec, raw_scope=outer)
    changed = raw.copy()
    changed._data[:, inner.stop_sample - raw.first_samp:] += 1e-3 * np.sin(np.arange(6500 - inner.stop_sample + raw.first_samp))
    updated = execute_preprocessing(changed, events, spec, raw_scope=inner)
    np.testing.assert_array_equal(original.data, updated.data)
    assert not np.array_equal(final.data, execute_preprocessing(changed, events, spec, raw_scope=outer).data)
    cropped, changed_crop = crop_raw_scope(raw, inner), crop_raw_scope(changed, inner)
    options = dict(spec=ContinuousQCSpec(12., 1e-7, 10),
                   fit_sample_mask=np.ones(cropped.n_times, bool), fit_provenance={"scope": inner.record()})
    assert fit_continuous_qc(cropped, **options).record() == fit_continuous_qc(changed_crop, **options).record()


def test_closed_source_endpoint_and_short_scope_are_respected():
    from data.preprocess import RawSampleScope

    raw, events, _, spec = _record()
    onset = int(events[0, 0])
    scope = RawSampleScope("short", onset - 50, onset + 301, (0,))
    result = execute_preprocessing(raw, events, spec, raw_scope=scope)
    assert result.data.shape == (1, 3, 179) and np.isfinite(result.data).all()
    assert result.source_epoch_start_samples.tolist() == [onset - 50]
    assert result.source_epoch_end_samples.tolist() == [onset + 300]
    with pytest.warns(RuntimeWarning, match="All epochs were dropped|epochs were dropped|empty"):
        unsupported = execute_preprocessing(raw, events, spec, raw_scope=replace(scope, stop_sample=onset + 300))
    assert unsupported.n_epochs == 0 and unsupported.event_statuses.tolist() == ["boundary_dropped"]


@pytest.mark.parametrize("source_rate", [128., 127.999, 512.])
def test_raw_scope_crop_keeps_exact_samples_at_other_source_rates(source_rate):
    from data.preprocess import crop_raw_scope, prefix_raw_scopes

    raw = mne.io.RawArray(np.random.default_rng(9).normal(0, 3e-6, (3, int(26 * source_rate))),
        mne.create_info(["Fz", "Cz", "Pz"], source_rate, "eeg"), first_samp=701, verbose=False)
    samples = 701 + np.rint((np.arange(20) + 2) * source_rate).astype(int)
    events = np.column_stack((samples, np.zeros(20, int), np.ones(20, int)))
    spec = _record()[-1]
    scope, _ = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    cropped = crop_raw_scope(raw, scope)
    assert cropped.first_samp == scope.start_sample
    assert cropped.n_times == scope.stop_sample - scope.start_sample
    actual = execute_preprocessing(raw, events, spec, raw_scope=scope)
    expected = execute_preprocessing(cropped, events[:5], spec)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("field,value", [("parent_n_times", 1), ("parent_event_count", 2),
                                       ("operator_order", "filter_then_crop")])
def test_raw_scope_receipt_rejects_false_parent_support_or_execution_order(field, value):
    from data.preprocess import prefix_raw_scopes
    from research.contracts import semantic_sha256

    raw, events, _, spec = _record()
    scope, _ = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    receipt = deepcopy(execute_preprocessing(raw, events, spec, raw_scope=scope).execution_receipt)
    receipt["source_context"]["raw_scope"][field] = value
    receipt["receipt_sha256"] = semantic_sha256({key: item for key, item in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="raw scope"):
        validate_preprocessing_execution(receipt, spec)


def test_k0_never_constructs_target_fit_scope():
    from data.preprocess import prefix_raw_scopes

    raw, events, _, spec = _record()
    calibration, heldout = prefix_raw_scopes(raw, events, spec, calibration_events=0, maximum_calibration_events=8)
    assert calibration is None and heldout.event_indices == tuple(range(8, 20))


def test_scope_normalizes_numpy_integer_indices_without_coercing_nonintegers():
    import json

    from data.preprocess import RawSampleScope

    scope = RawSampleScope("native_indices", np.int64(750), np.int64(3000), tuple(np.arange(5)))
    assert RawSampleScope.from_record(json.loads(json.dumps(scope.record()))) == scope


def test_source_preparation_binds_profile_scope_and_only_checks_authorized_samples(tmp_path, monkeypatch):
    from data.epochs import load_epoch_dataset
    from data.preprocess import prefix_raw_scopes
    from data.source_records import file_sha256, prepare_source_record

    raw, events, labels, spec = _record()
    scope, heldout = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    source = tmp_path / "synthetic.mat"
    source.write_bytes(b"synthetic parser input")
    item = dict(study="BNCI2014-008", subject="synthetic", path=source.name,
                sha256=file_sha256(source), sample_rate_hz=250., reference="right earlobe", session="s", run="r")
    monkeypatch.setattr("data.source_records._read_record", lambda *args: (raw.copy(), events, labels))
    first = prepare_source_record(item, root=tmp_path, output=tmp_path / "cache", preprocessing=spec, raw_scope=scope)
    dataset = load_epoch_dataset(first["cache"], validation="attested")
    assert dataset.preprocessing == spec and first["scheduled"] == 5
    assert first["preparation"]["raw_scope"] == scope.record()
    assert dataset.provenance["source_reference"] == "measured_channel:CZ"
    test = prepare_source_record(item, root=tmp_path, output=tmp_path / "test", preprocessing=spec, raw_scope=heldout)
    assert test["cache"] != first["cache"]
    raw._data[:, scope.stop_sample - raw.first_samp:] = np.nan
    labels[5:] = 999
    repeated = prepare_source_record({**item, "preprocessing": spec.record(), "raw_scope": scope.record()},
                                     root=tmp_path, output=tmp_path / "repeated")
    np.testing.assert_array_equal(load_epoch_dataset(repeated["cache"]).X, dataset.X)
    np.testing.assert_array_equal(load_epoch_dataset(repeated["cache"]).y, dataset.y)


def test_scoped_source_preparation_still_rejects_missing_measured_cz(tmp_path, monkeypatch):
    from data.preprocess import prefix_raw_scopes
    from data.source_records import file_sha256, prepare_source_record

    raw, events, labels, spec = _record()
    scope, _ = prefix_raw_scopes(raw, events, spec, calibration_events=5, maximum_calibration_events=8)
    raw.drop_channels(["Cz"])
    source = tmp_path / "synthetic.mat"
    source.write_bytes(b"synthetic parser input")
    item = dict(study="BNCI2014-008", subject="synthetic", path=source.name,
                sha256=file_sha256(source), sample_rate_hz=250., reference="right earlobe", session="s", run="r")
    monkeypatch.setattr("data.source_records._read_record", lambda *args: (raw.copy(), events, labels))
    monkeypatch.setattr("data.source_records.build_subject", lambda *args, **kwargs: pytest.fail("missing Cz must fail before filtering"))
    with pytest.raises(ValueError, match="reference"):
        prepare_source_record(item, root=tmp_path, output=tmp_path / "cache", preprocessing=spec, raw_scope=scope)


@pytest.mark.parametrize("changes", [{"event_indices": (1, 1)}, {"event_indices": (2, 1)},
                                     {"start_sample": True}, {"stop_sample": 700}, {"name": ""}])
def test_scope_rejects_ambiguous_or_invalid_declarations(changes):
    from data.preprocess import RawSampleScope

    fields = dict(name="train", start_sample=750, stop_sample=3000, event_indices=(0, 1))
    with pytest.raises(ValueError):
        RawSampleScope(**{**fields, **changes})
