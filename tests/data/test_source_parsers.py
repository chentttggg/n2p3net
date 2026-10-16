"""Continuous-format counterexamples at the same ingress used by preparation."""
import numpy as np
import pytest
from scipy.io import savemat

from data.epochs import load_epoch_dataset
from data.source_parsers import read_source_record, validate_source_manifest
from data.source_records import file_sha256, prepare_source_record, verify_source_record_inputs


def _brainform(tmp_path):
    path = tmp_path / "EXP_P02" / "stamp__ERPAutotrainingPipeline_raw_data0.csv"
    path.parent.mkdir()
    data = np.zeros((6000, 14))
    data[:, :8] = np.random.default_rng(9).normal(size=(6000, 8))
    for i in range(30):
        sample = 1500 + 80 * i
        candidate = i % 10 + 1
        target = int(candidate == 1 and i < 20)
        data[sample, 8:] = [1, candidate, target, 1-target, i // 10 + 1 if i < 20 else 1, int(candidate == 10)]
    np.savetxt(path, data, delimiter=",", header="samplingrate: 250", comments="#")
    item = dict(study="BrainForm", subject="P02", session="cb", run="stamp", path=str(path.relative_to(tmp_path)),
                sha256=file_sha256(path), sample_rate_hz=250, reference="common_mastoid",
                excluded_channels=["Fz"], channel_exclusion_reason="synthetic record-specific bad channel")
    return path, data, item


def test_brainform_uses_only_calibration_truth_and_preserves_samples_and_masks(tmp_path):
    path, data, item = _brainform(tmp_path)
    raw, events, labels = read_source_record(item, path)
    assert len(labels) == 20 and labels.sum() == 2
    assert raw.first_samp == 1245 and events[0, 0] == 1500
    np.testing.assert_allclose(raw.get_data(), data[1245:, :8].T * 1e-6)
    raw.close()
    result = prepare_source_record(item, root=tmp_path, output=tmp_path / "prepared")
    cache = load_epoch_dataset(result["cache"], validation="attested")
    assert cache.n_epochs == 20 and cache.event_timeline.n_events == 20
    assert not cache.channel_mask[0] and not cache.channel_mask[1] and not cache.channel_mask[3]
    assert np.count_nonzero(cache.X[:, [0, 1, 3]]) == 0
    with pytest.raises(ValueError, match="participant/run"):
        read_source_record({**item, "subject": "P99"}, path)
    data[1500, 10:12] = [0, 1]
    np.savetxt(path, data, delimiter=",", header="samplingrate: 250", comments="#")
    with pytest.raises(ValueError, match="truth/sequence"):
        read_source_record(item, path)


def test_cattan_keeps_continuous_gaps_and_only_explicitly_repairs_one_bad_tail(tmp_path):
    path = tmp_path / "subject_01_VR.mat"
    data = np.zeros((6000, 22))
    data[:, 1:17] = np.random.default_rng(4).normal(size=(6000, 16))
    data[[1000, 4000], 18] = 1
    data[[2000, 5000], 19] = 1
    data[-1, 5] = np.nan
    savemat(path, {"data": data})
    item = dict(study="Cattan2019_VR", subject="1", session="VR", run="continuous",
                valid_samples_exclusive_end=5999)
    raw, events, labels = read_source_record(item, path)
    assert raw.n_times == 5999 and raw.first_samp == 0
    np.testing.assert_array_equal(events[:, 0], [1000, 2000, 4000, 5000])
    np.testing.assert_array_equal(labels, [1, 0, 1, 0])
    raw.close()
    for boundary in (5998, 6000):
        with pytest.raises(ValueError, match="repair|nonfinite"):
            read_source_record({**item, "valid_samples_exclusive_end": boundary}, path)


def test_brainvision_payload_and_markers_are_both_verified(tmp_path):
    header = tmp_path / "sample.vhdr"
    header.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=sample.eeg\nMarkerFile=sample.vmrk\n", encoding="utf-8")
    (tmp_path / "sample.eeg").write_bytes(b"signal")
    markers = tmp_path / "sample.vmrk"
    markers.write_bytes(b"events")
    item = dict(study="Chailloux2020", path=header.name, sha256=file_sha256(header))
    original = verify_source_record_inputs(item, root=tmp_path)
    assert {r["role"] for r in original} == {"raw", "signal", "markers"}
    item["sidecars"] = {r["role"]: {"path": r["path"], "sha256": r["sha256"]} for r in original if r["role"] != "raw"}
    markers.write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="markers sidecar changed"):
        verify_source_record_inputs(item, root=tmp_path)


def test_download_status_cannot_admit_an_unsupported_parser():
    with pytest.raises(ValueError, match="download completion is not parser admission"):
        validate_source_manifest({"schema": "n2p3_source_records/1", "study": "Lee2019", "status": "complete"})
