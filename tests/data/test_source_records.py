"""Explicit source ingress retains the shared executed preprocessing contract."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from data.epochs import load_epoch_dataset
from data.source_records import SOURCE_EOG_PREPROCESSING, file_sha256, prepare_source_record
from tests.support.signal import make_raw


def _prepare_raw(tmp_path, monkeypatch, raw, *, output="prepared"):
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified synthetic source")
    labels = np.tile([0, 1], 4)
    events = np.column_stack(((np.arange(8) + 2) * 256, np.zeros(8, int), labels + 1))
    monkeypatch.setattr("data.source_records._read_record", lambda _item, _path: (raw.copy(), events, labels))
    item = dict(study="BNCI2014-008", subject="synthetic-person", path=source.name, sha256=file_sha256(source),
                sample_rate_hz=256, reference="nose", session="s1", run="r1")
    return prepare_source_record(item, root=tmp_path, output=tmp_path / output)


def test_explicit_source_record_retains_receipt_after_measured_reference_projection(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified synthetic source")
    raw = make_raw(sfreq=256, n_seconds=12, ch_names=("Fz", "Cz", "Pz"))
    labels = np.tile([0, 1], 4)
    events = np.column_stack(((np.arange(8) + 2) * 256, np.zeros(8, int), labels + 1))
    monkeypatch.setattr("data.source_records._read_record", lambda _item, _path: (raw.copy(), events, labels))
    item = dict(study="BNCI2014-008", subject="synthetic-person", path=source.name, sha256=file_sha256(source),
                sample_rate_hz=256, reference="nose", session="s1", run="r1")
    result = prepare_source_record(item, root=tmp_path, output=tmp_path / "prepared")
    data = load_epoch_dataset(result["cache"], validation="attested")
    assert data.preprocessing == SOURCE_EOG_PREPROCESSING
    assert data.provenance["source_reference"] == "measured_channel:CZ"
    receipt = data.provenance["preprocessing_execution"]
    assert receipt["declaration"] == SOURCE_EOG_PREPROCESSING.record()
    assert receipt["output"]["channel_names"] == ["FZ", "CZ", "PZ"]
    assert receipt["output"]["scheduled_events"] == len(events)
    assert data.n_channels == 6 and data.channel_mask.sum() == 2


def test_bad_nonreference_is_removed_before_numeric_processing_and_keeps_healthy_values(tmp_path, monkeypatch):
    raw = make_raw(sfreq=256, n_seconds=12, ch_names=("Fz", "Cz", "Pz"))
    original = load_epoch_dataset(_prepare_raw(tmp_path, monkeypatch, raw, output="clean")["cache"])
    raw.info["bads"] = ["Pz"]
    raw._data[2] = np.nan
    prepared = _prepare_raw(tmp_path, monkeypatch, raw, output="bad")
    dataset = load_epoch_dataset(prepared["cache"], validation="attested")
    assert dataset.channel_mask.tolist() == [True, False, False, False, False, False]
    assert not dataset.trial_channel_mask[:, 2].any()
    assert np.count_nonzero(dataset.X[:, 2]) == 0
    np.testing.assert_array_equal(dataset.X[:, 0], original.X[:, 0])
    admission = dataset.provenance["source_channel_admission"]
    assert admission["original_bads"] == ["Pz"]
    assert admission["removed_bad_channels"] == ["Pz"]
    assert any(entity.parameters.get("channel_admission") == admission for entity in dataset.lineage.entities)


@pytest.mark.parametrize("case", ["bad_reference", "missing_reference", "wrong_reference_role", "alias_collision", "wrong_target_role", "only_reference"])
def test_invalid_measured_reference_or_channel_identity_fails_before_preprocessing(tmp_path, monkeypatch, case):
    raw = make_raw(sfreq=256, n_seconds=12, ch_names=("Fz", "Cz", "Pz"))
    if case == "bad_reference":
        raw.info["bads"] = ["Cz"]
    elif case == "missing_reference":
        raw.drop_channels(["Cz"])
    elif case == "wrong_reference_role":
        raw.set_channel_types({"Cz": "eog"})
    elif case == "alias_collision":
        raw.rename_channels({"Pz": "EEG Cz-REF"})
    elif case == "wrong_target_role":
        raw.set_channel_types({"Pz": "eog"})
    else:
        raw.info["bads"] = ["Fz", "Pz"]
    monkeypatch.setattr("data.source_records.build_subject", lambda *a, **k: pytest.fail("numeric pipeline must not run"))
    with pytest.raises(ValueError):
        _prepare_raw(tmp_path, monkeypatch, raw)


def test_healthy_nan_is_rejected_and_unique_aliases_preserve_numbers(tmp_path, monkeypatch):
    raw = make_raw(sfreq=256, n_seconds=12, ch_names=("Fz", "Cz", "Pz"))
    expected = load_epoch_dataset(_prepare_raw(tmp_path, monkeypatch, raw, output="original")["cache"])
    raw.rename_channels({"Fz": "EEG Fz-REF", "Cz": "EEG Cz-REF", "Pz": "EEG Pz-REF"})
    raw.reorder_channels(raw.ch_names[::-1])
    actual = load_epoch_dataset(_prepare_raw(tmp_path, monkeypatch, raw, output="alias")["cache"])
    np.testing.assert_array_equal(actual.X, expected.X)
    raw._data[0, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _prepare_raw(tmp_path, monkeypatch, raw, output="invalid")


@pytest.fixture
def huebner_record(tmp_path, monkeypatch):
    import data.source_records as records
    from experiments import prepare_source_records as runner

    source = tmp_path / "person_eeg.bdf"
    source.write_bytes(b"verified synthetic BDF bytes")
    channels = tmp_path / "person_channels.tsv"
    channels.write_text("name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\nPz\tEEG\tbad\n", encoding="utf-8")
    events = tmp_path / "person_events.tsv"
    events.write_text("sample\tonset\ttrial_type\n" + "".join(
        f"{(i+2)*256}\t{i+2}\t{'Target' if i % 2 else 'NonTarget'}\n" for i in range(8)), encoding="utf-8")
    raw = make_raw(sfreq=256, n_seconds=12, ch_names=("Fz", "Cz", "Pz"))
    raw._data[2] = np.nan
    monkeypatch.setattr("data.source_parsers.mne.io.read_raw_bdf", lambda *a, **k: raw.copy())
    # Bind one implementation snapshot while other independent agents may edit data modules.
    method = records.source_preparation_method()
    monkeypatch.setattr(records, "source_preparation_method", lambda: method)
    monkeypatch.setattr(runner, "source_preparation_method", lambda: method)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    item = dict(study="Huebner2017", subject="1", path=source.name, sha256=file_sha256(source),
                sample_rate_hz=256, reference="common", session="s1", run="r1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": item["study"],
        "remote_root": str(tmp_path), "records": [item]}), encoding="utf-8")
    return runner, source, channels, events, item, manifest


def test_huebner_actual_sidecars_bind_lineage_and_completed_resume(huebner_record, tmp_path, monkeypatch):
    from data.source_admission import PreparedSourceInventory, bind_source_inputs
    runner, source, channels, events, _, manifest = huebner_record
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    runner.main(argv)
    ledger_path = output / "preparation.json"
    ledger = json.loads(ledger_path.read_text())
    inputs = ledger["records"][0]["preparation"]["consumed_inputs"]
    assert {row["role"]: row["sha256"] for row in inputs} == {
        "raw": file_sha256(source), "channels": file_sha256(channels), "events": file_sha256(events)}
    dataset = load_epoch_dataset(ledger["cache"], validation="attested")
    assert dataset.channel_mask.sum() == 1 and not dataset.trial_channel_mask[:, 2].any()
    assert np.count_nonzero(dataset.X[:, 2]) == 0
    assert any(entity.parameters.get("consumed_inputs") == inputs for entity in dataset.lineage.entities)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("completed resume must not launch preparation"))
    monkeypatch.setattr("data.epochs._epoch_cache_archive_source", lambda *a, **k: pytest.fail("completed admission must not copy or decompress epoch matrices"))
    runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())
    inventory = PreparedSourceInventory(tmp_path, preprocessing=SOURCE_EOG_PREPROCESSING)
    state = inventory.inspect(["Huebner2017", "missing-study"])
    assert set(state["ready"]) == {"Huebner2017"}
    assert state["pending"] == {"missing-study": "missing preparation ledger"}
    assert not state["failed"]
    bound = bind_source_inputs({}, state["ready"])
    assert bind_source_inputs(bound, state["ready"]) == bound
    with pytest.raises(ValueError, match="input changed"):
        bind_source_inputs(bound, {"Huebner2017": {**ledger, "cache_sha256": "0" * 64}})
    assert inventory.inspect(["Huebner2017"])["ready"]["Huebner2017"] == ledger
    ledger["study"] = "different-study"
    for row in ledger["records"]:
        row["record"]["study"] = "different-study"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    assert "different-study" in inventory.inspect(["different-study"])["failed"]


def test_pool_cli_checks_preparation_and_resume_without_starting_fits(huebner_record, tmp_path, monkeypatch):
    from experiments import run_source_pool_matrix as pool_runner

    preparer, _, _, _, _, manifest = huebner_record
    preparer.main(["--manifest", str(manifest), "--output", str(tmp_path / "prepared" / "Huebner2017"), "--workers", "1"])
    plan = tmp_path / "pool.json"
    plan.write_text(json.dumps({"schema": "n2p3_source_pool_plan/1", "source_pools": {}, "seeds": [7],
        "standalone": {"Huebner": {"studies": ["Huebner2017"], "channels": ["Fz"]}}}), encoding="utf-8")
    monkeypatch.setattr(pool_runner, "verified_executed_snapshot", lambda *a, **k: {"archive_sha256": "0" * 64})
    monkeypatch.setattr(pool_runner, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("dry-run must not start workers"))
    args = ["--prepared", str(tmp_path / "prepared"), "--output", str(tmp_path / "matrix"),
            "--snapshot", str(tmp_path / "snapshot.json"), "--matrix-plan", str(plan), "--dry-run", "--device", "cpu"]
    pool_runner.main(args)
    pool_runner.main(args)
    binding = json.loads((tmp_path / "matrix/input_bindings.json").read_text())
    assert set(binding) == {"Huebner2017"}
    record_path = tmp_path / "prepared/Huebner2017/preparation.json"
    record = json.loads(record_path.read_text())
    record["status"] = "failed"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not ready"):
        pool_runner.main(args)


@pytest.mark.parametrize("artifact", ["shard", "merged"])
@pytest.mark.parametrize("damage", ["missing", "identity_changed"])
def test_completed_resume_requires_original_cache_attestation(huebner_record, tmp_path, monkeypatch, artifact, damage):
    from pathlib import Path

    runner, _, _, _, _, manifest = huebner_record
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    runner.main(argv)
    ledger = json.loads((output / "preparation.json").read_text())
    cache = Path(ledger["records"][0]["cache"] if artifact == "shard" else ledger["cache"])
    record_path = cache.with_suffix(".record.json")
    if damage == "missing":
        record_path.unlink()
    else:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["identity"]["records"][0]["local_subject_id"] = "different-person"
        record_path.write_text(json.dumps(record), encoding="utf-8")
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("damaged cache admission must not start workers"))
    with pytest.raises(ValueError, match="attestation|record"):
        runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("changed", ["raw", "channels", "events", "method", "legacy"])
def test_completed_resume_revalidates_actual_inputs_and_method_before_cache_reuse(huebner_record, tmp_path, monkeypatch, changed):
    runner, source, channels, events, _, manifest = huebner_record
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    runner.main(argv)
    if changed in {"raw", "channels", "events"}:
        path = {"raw": source, "channels": channels, "events": events}[changed]
        path.write_bytes(path.read_bytes() + b"changed")
    elif changed == "method":
        method = {**runner.source_preparation_method(), "schema": "changed_implementation"}
        monkeypatch.setattr(runner, "source_preparation_method", lambda: method)
    else:
        path = output / "preparation.json"
        ledger = json.loads(path.read_text())
        ledger.pop("preparation_method")
        ledger["schema"] = "n2p3_source_preparation/1"
        path.write_text(json.dumps(ledger), encoding="utf-8")
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("invalid reuse must fail before workers"))
    with pytest.raises(ValueError, match="changed|unrecorded"):
        runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())


def test_declared_sidecar_digest_and_bad_status_are_checked(huebner_record, tmp_path):
    _, _, channels, _, item, _ = huebner_record
    item = {**item, "sidecars": {"channels": {"path": channels.name, "sha256": "a" * 64}}}
    with pytest.raises(ValueError, match="sidecar changed"):
        prepare_source_record(item, root=tmp_path, output=tmp_path / "wrong_digest")
    item.pop("sidecars")
    channels.write_text(channels.read_text().replace("bad", "unknown"), encoding="utf-8")
    with pytest.raises(ValueError, match="good/bad"):
        prepare_source_record(item, root=tmp_path, output=tmp_path / "wrong_status")


def test_gtn_parser_metadata_candidates_and_primary_file_are_bound(tmp_path):
    from data.source_records import verify_source_record_inputs

    source = tmp_path / "record.nix"
    source.write_bytes(b"nix fingerprint")
    (tmp_path / "Data").mkdir()
    metadata = tmp_path / "Data" / "participant.txt"
    metadata.write_text("thought digit 1", encoding="utf-8")
    item = {"study": "GTN", "path": source.name, "sha256": file_sha256(source)}
    first = verify_source_record_inputs(item, root=tmp_path)
    assert first[1]["sha256"] == file_sha256(metadata)
    metadata.write_text("thought digit 2", encoding="utf-8")
    assert verify_source_record_inputs(item, root=tmp_path) != first
    (tmp_path / "other.nix").write_bytes(b"ambiguous second NIX")
    with pytest.raises(ValueError, match="only parsed NIX"):
        verify_source_record_inputs(item, root=tmp_path)


def test_archive_member_verification_binds_consumed_bytes(tmp_path):
    import hashlib
    from zipfile import ZipFile

    from data.source_records import verify_source_record_inputs

    archive = tmp_path / "source.zip"
    data = b"the consumed source member"
    with ZipFile(archive, "w") as stream:
        stream.writestr("subject/record.mat", data)
    item = {"study": "EPFLP300", "path": archive.name, "member": "subject/record.mat",
            "sha256": hashlib.sha256(data).hexdigest()}
    assert verify_source_record_inputs(item, root=tmp_path)[0]["sha256"] == item["sha256"]
    with ZipFile(archive, "w") as stream:
        stream.writestr("subject/record.mat", b"changed input")
    with pytest.raises(ValueError, match="changed"):
        verify_source_record_inputs(item, root=tmp_path)
