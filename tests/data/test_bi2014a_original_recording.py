"""Original MAT/CSV share a physical schedule while preserving numeric origin."""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat


def original_samples(*, repetitions=4, tail=0):
    onsets = 512 + np.arange(repetitions * 12 + tail) * 96
    samples = np.zeros((int(onsets[-1]) + 700, 19))
    samples[:, 0] = np.arange(len(samples)) / 512
    samples[:, 1:17] = np.random.default_rng(7).normal(1, 3, (len(samples), 16))
    codes = np.tile([60, 82, 21, 22, 23, 24, 25, 40, 41, 43, 44, 45], repetitions)
    if tail:
        codes = np.concatenate((codes, [21, 22][:tail]))
    samples[onsets, 17] = codes
    samples[onsets, 18] = np.where(codes >= 60, 2, 1)
    samples[200, 17] = 100
    samples[onsets[24] - 20, 17] = 104
    samples[onsets[12] - 20, 17] = 106
    return samples


def source(tmp_path, samples, *, kind="mat", subject="01"):
    path = tmp_path / f"subject_{subject}.zip"
    stream = io.BytesIO()
    if kind == "mat":
        savemat(stream, {"samples": samples})
    else:
        stream.write(pd.DataFrame(samples).to_csv(index=False, header=False, float_format="%.5g").encode())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"subject_{subject}.{kind}", stream.getvalue())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_zip_mat_requires_no_csv_and_keeps_private_field_authority(tmp_path):
    from data.bi2014a_recording import read_bi2014a_original_record

    samples = original_samples(tail=1)
    path, digest = source(tmp_path, samples)
    record = read_bi2014a_original_record(path, source_sha256=digest, source_format="original_mat_zip")
    assert record.subject_id == "01" and record.source_sha256 == digest
    assert record.source_receipt["member"]["name"] == "subject_01.mat"
    assert record.source_receipt["original_format"] == "original_mat_zip"
    assert record.source_receipt["clock"]["recorded_time_exact_on_sample_grid"] is True
    assert np.array_equal(record.eeg_uv, samples[:, 1:17])
    assert record.schedule.n_repetitions == 4 and record.schedule.dropped_tail_flashes == 1
    assert record.schedule.raw_label_audit.n_mismatches == 0
    assert record.schedule.repetition_index.reshape(4, 12)[:, 0].tolist() == [0, 1, 0, 1]
    assert not list(tmp_path.glob("*.csv")) and not list(tmp_path.glob("*.mat"))
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("subject_01.mat")
    assert record.source_receipt["member"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_csv_rounding_preserves_schedule_but_has_its_own_numeric_source(tmp_path):
    from data.bi2014a_recording import read_bi2014a_original_record

    samples = original_samples()
    mat_dir, csv_dir = tmp_path / "mat", tmp_path / "csv"
    mat_dir.mkdir()
    csv_dir.mkdir()
    mat, mat_sha = source(mat_dir, samples)
    csv, csv_sha = source(csv_dir, samples, kind="csv")
    a = read_bi2014a_original_record(mat, source_sha256=mat_sha, source_format="original_mat_zip")
    b = read_bi2014a_original_record(csv, source_sha256=csv_sha, source_format="original_csv_zip")
    assert np.array_equal(a.schedule.flash_sample, b.schedule.flash_sample)
    assert np.array_equal(a.schedule.flash_code, b.schedule.flash_code)
    assert np.array_equal(a.schedule.selection_id, b.schedule.selection_id)
    assert np.array_equal(a.schedule.target_label, b.schedule.target_label)
    assert not np.array_equal(a.eeg_uv, b.eeg_uv)
    assert a.source_receipt["numeric_fields_sha256"] != b.source_receipt["numeric_fields_sha256"]
    assert b.source_receipt["clock"]["recorded_time_exact_on_sample_grid"] is False


@pytest.mark.parametrize("change,match", [("clock", "clock"), ("label", "label"), ("subject", "excluded subjects"), ("sha", "bytes")])
def test_original_decoder_rejects_changed_physical_authority(tmp_path, change, match):
    from data.bi2014a_recording import read_bi2014a_original_record

    samples = original_samples()
    if change == "clock":
        samples[10, 0] += .01
    if change == "label":
        samples[np.flatnonzero(samples[:, 17] == 60)[0], 18] = 1
    path, digest = source(tmp_path, samples, subject="62" if change == "subject" else "01")
    if change == "sha":
        digest = "0" * 64
    with pytest.raises(ValueError, match=match):
        read_bi2014a_original_record(path, source_sha256=digest, source_format="original_mat_zip")


def test_adapter_only_exports_public_memberships_and_preserves_unknown_terminal_request(tmp_path):
    from data.bi2014a_recording import bi2014a_dataset_adapter

    samples = original_samples(tail=1)
    onsets = np.flatnonzero((samples[:, 17] >= 20) & (samples[:, 17] <= 85))
    samples[onsets[-1] - 20, 17] = 104
    path, digest = source(tmp_path, samples)
    adapter = bi2014a_dataset_adapter([{"path": str(path), "sha256": digest, "subject": "01"}],
                                       source_format="original_mat_zip")
    assert adapter.dataset_id == "BI2014a" and adapter.recordings[0].source_subject_id == "01"
    native = adapter.recordings[0]
    assert native.native_sample_count == len(samples) and native.sample_rate_hz == 512
    assert len(native.channel_names) == 16 and native.reference == "right earlobe"
    assert len(native.selections) == 3 and native.selections[-1].parent_target_candidate is None
    assert not native.selections[-1].decision.schedule_complete
    assert sum(len(row.event_indices) for row in native.selections) == len(onsets)
    for public in adapter.record()["recordings"][0]["selections"]:
        assert set(public) == {"selection_id", "event_indices", "decision"}
        assert "parent_target_candidate" not in public
        assert all(0 <= value < 12 for value in public["decision"]["stimulus_indices"])
        assert not {"flash_code", "target_row", "target_col", "raw_label", "selection_boundary_reason"} & public.keys()
    np.testing.assert_array_equal(native.read_native(10, 20), samples[10:20, 1:17] * 1e-6)
    rebuilt = bi2014a_dataset_adapter(adapter.source_artifacts, source_format=adapter.source_format)
    assert rebuilt.digest == adapter.digest


def test_private_target_change_cannot_supply_a_public_selection_boundary(tmp_path):
    from data.bi2014a_recording import bi2014a_dataset_adapter

    samples = original_samples()
    onsets = np.flatnonzero((samples[:, 17] >= 20) & (samples[:, 17] <= 85))
    # A valid second repetition with another private target, inside the same
    # original public episode: its targets cannot authorize a new boundary.
    third = np.asarray([61, 82, 20, 22, 23, 24, 25, 40, 41, 43, 44, 45])
    samples[onsets[12:24], 17] = third
    samples[onsets[12:24], 18] = np.where(third >= 60, 2, 1)
    path, digest = source(tmp_path, samples)
    with pytest.raises(ValueError, match="physical episode contains conflicting"):
        bi2014a_dataset_adapter([{"path": str(path), "sha256": digest, "subject": "01"}],
                               source_format="original_mat_zip")
