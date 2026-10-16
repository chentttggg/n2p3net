"""Real BDF ingress proves per-K raw operators and profile-bound CLI resume."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import load_epoch_dataset, preprocessing_spec_from_contract
from data.source_records import file_sha256, load_target_prefixes, prepare_source_record_prefixes


def _write_real_record(tmp_path):
    import edfio

    source = tmp_path / "person_eeg.bdf"
    samples = np.random.default_rng(508).normal(0, 4, (3, 26 * 250))
    edfio.Bdf([edfio.BdfSignal(values, sampling_frequency=250, label=name,
        physical_dimension="uV", physical_range=(-100, 100))
        for name, values in zip(("Fz", "Cz", "Pz"), samples, strict=True)]).write(source)
    (tmp_path / "person_channels.tsv").write_text(
        "name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\nPz\tEEG\tgood\n", encoding="utf-8")
    (tmp_path / "person_events.tsv").write_text("sample\tonset\ttrial_type\n" + "".join(
        f"{(i+2)*250}\t{i+2}\t{'Target' if i % 2 else 'NonTarget'}\n" for i in range(20)), encoding="utf-8")
    item = dict(study="Huebner2017", subject="1", path=source.name, sha256=file_sha256(source),
                sample_rate_hz=250., reference="native common", session="s1", run="r1")
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": item["study"],
        "remote_root": str(tmp_path), "records": [item]}), encoding="utf-8")
    return item, spec, manifest


@pytest.fixture
def real_record(tmp_path, monkeypatch):
    import data.source_records as records
    from experiments import prepare_source_records as runner

    item, spec, manifest = _write_real_record(tmp_path)
    # Other agents can edit unrelated data files during this bounded numeric probe.
    method = records.source_preparation_method(spec)
    def frozen_method(profile=None):
        return {**deepcopy(method), "preprocessing": (records.SOURCE_EOG_PREPROCESSING if profile is None else profile).record()}
    monkeypatch.setattr(records, "source_preparation_method", frozen_method)
    monkeypatch.setattr(runner, "source_preparation_method", frozen_method)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    return item, spec, manifest, runner


@pytest.fixture(scope="module")
def prepared_prefixes(tmp_path_factory):
    """Prepare immutable filter/cache evidence once for independent admission attacks."""
    root = tmp_path_factory.mktemp("prefix-admission")
    item, spec, _ = _write_real_record(root)
    return prepare_source_record_prefixes(item, root=root, output=root / "shards",
        preprocessing=spec, budgets=(0, 4, 8), minimum_test_trials=2)


def _collection(tmp_path, pair):
    path = tmp_path / "target_prefixes.json"
    path.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": pair["preprocessing"], "preparation_method": pair["preparation_method"],
        "records": [pair]}), encoding="utf-8")
    return path


def test_real_raw_is_read_once_and_each_k_executes_its_own_filter(real_record, tmp_path, monkeypatch):
    import data.source_records as records

    item, spec, _, _ = real_record
    original = records._read_record
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(records, "_read_record", counted)
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0, 4, 8), minimum_test_trials=2)
    assert len(calls) == 1
    loaded = load_target_prefixes(_collection(tmp_path, pair))["1"]
    assert pair["effective_budgets"] == [0, 4, 8]
    assert loaded["calibration"][0] is None
    c4, c8, test = loaded["calibration"][4], loaded["calibration"][8], loaded["test"]
    assert (c4.n_epochs, c8.n_epochs, test.n_epochs) == (4, 8, 11)
    assert not np.array_equal(c4.X, c8.X[:4])  # Full-C filtering then slicing cannot stand in for K4.
    assert set(c4.event_timeline.session_ids) == set(test.event_timeline.session_ids) == {"s1"}
    assert test.event_timeline.event_ids[0].endswith(":8")
    assert pair["test"]["purged_event_ids"] == [str(test.event_timeline.event_ids[0])]
    for k in (4, 8):
        view = pair["calibration"][str(k)]
        assert view["authorized_label_count"] == view["retained_label_count"] == k
        assert view["preparation"]["raw_scope"]["stop_sample"] <= pair["test"]["preparation"]["raw_scope"]["start_sample"]


def test_limited_data_fallback_records_actual_allocation(real_record, tmp_path):
    item, spec, _, _ = real_record
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0, 4, 200), minimum_test_trials=2)
    assert pair["effective_budgets"] == [0, 4, 16]
    assert pair["maximum_calibration_events"] == 16
    assert set(pair["adjustments"]) == {"200"}
    assert "200" not in pair["calibration"]
    assert load_target_prefixes(_collection(tmp_path, pair))["1"]["test"].n_epochs == 3


def test_target_subject_filter_skips_other_cache_payloads_and_rejects_missing_subject(real_record, tmp_path):
    item, spec, _, _ = real_record
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0,), minimum_test_trials=2)
    path = _collection(tmp_path, pair)
    collection = json.loads(path.read_text())
    other = deepcopy(pair)
    other["record"]["subject"] = "2"
    other["test"]["cache"] = "unavailable-other-subject.npz"
    collection["records"].append(other)
    path.write_text(json.dumps(collection), encoding="utf-8")
    selected = load_target_prefixes(path, subjects=("1",))
    assert set(selected) == {"1"}
    assert selected["1"]["calibration"] == {0: None}
    assert selected["1"]["test"].n_epochs == 20
    with pytest.raises(ValueError, match="absent"):
        load_target_prefixes(path, subjects=("3",))
    with pytest.raises(ValueError, match="original|missing"):
        load_target_prefixes(path)


@pytest.mark.parametrize("location,change", [
    pytest.param(("calibration", "4", "preparation", "raw_scope", "stop_sample"), lambda value: value + 1, id="scope"),
    pytest.param(("calibration", "4", "retained_label_count"), lambda value: value + 1, id="count"),
    pytest.param(("preprocessing", "notch_freqs_hz"), lambda value: [], id="profile"),
    pytest.param(("effective_budgets",), lambda value: [0, 4, 200], id="budget"),
])
def test_prefix_loader_rejects_relabelled_evidence(prepared_prefixes, tmp_path, location, change):
    pair = deepcopy(prepared_prefixes)
    target = pair
    for field in location[:-1]:
        target = target[field]
    target[location[-1]] = change(target[location[-1]])
    with pytest.raises(ValueError):
        load_target_prefixes(_collection(tmp_path, pair))


def test_prefix_loader_requires_original_cache_record(prepared_prefixes, tmp_path):
    import shutil

    pair = deepcopy(prepared_prefixes)
    damaged = tmp_path / "without-record.npz"
    shutil.copyfile(pair["test"]["cache"], damaged)
    pair["test"]["cache"] = str(damaged)
    with pytest.raises(ValueError, match="attestation"):
        load_target_prefixes(_collection(tmp_path, pair))


def test_cli_profile_scope_execution_and_completed_resume(real_record, tmp_path, monkeypatch):
    item, spec, manifest, runner = real_record
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(spec.record()), encoding="utf-8")
    declaration = json.loads(manifest.read_text())
    declaration["preprocessing"] = spec.record()
    declaration["records"][0]["raw_scope"] = dict(schema="n2p3_raw_sample_scope/1", name="authorized",
        start_sample=0, stop_sample=2051, event_indices=list(range(6)))
    manifest.write_text(json.dumps(declaration), encoding="utf-8")
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1", "--preprocessing-config", str(config)]
    runner.main(argv)
    ledger = json.loads((output / "preparation.json").read_text())
    assert ledger["preparation_method"]["preprocessing"] == spec.record()
    cache = load_epoch_dataset(ledger["cache"], validation="attested")
    assert cache.preprocessing == spec and cache.n_epochs == 6
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("resume cannot launch workers"))
    runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())
    config.write_text(json.dumps({**spec.record(), "notch_freqs_hz": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="same preprocessing"):
        runner.main(argv)


def test_cli_prefix_manifest_is_consumable_and_resume_does_not_parse_raw(real_record, tmp_path, monkeypatch):
    _, spec, manifest, runner = real_record
    declaration = json.loads(manifest.read_text())
    declaration["preprocessing"] = spec.record()
    manifest.write_text(json.dumps(declaration), encoding="utf-8")
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1",
            "--target-prefix-budgets", "0", "4", "8", "--minimum-test-trials", "2"]
    monkeypatch.setattr("data.source_records._prepare_scope_inner_views",
                        lambda **kwargs: pytest.fail("fixed target preparation must not filter unused inner views"))
    runner.main(argv)
    prepared = load_target_prefixes(output / "target_prefixes.json")["1"]
    assert set(prepared["calibration"]) == {0, 4, 8}
    assert prepared["early_stopping_inner"] == {}
    assert not (output / "epochs.npz").exists()
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr("data.source_records._read_record", lambda *a: pytest.fail("completed resume must not parse raw"))
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("completed resume must not launch workers"))
    runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())
    with pytest.raises(ValueError, match="prefix allocation changed"):
        runner.main([*argv, "--early-stopping-inner"])


def _multiple_records(item, tmp_path):
    import shutil

    first = {**item, "session": "earlier", "run": "first"}
    later = []
    for name, session, run in [("middle", "earlier", "second"), ("last", "later", "first")]:
        for suffix in ("eeg.bdf", "channels.tsv", "events.tsv"):
            shutil.copyfile(tmp_path / ("person_" + suffix), tmp_path / (name + "_" + suffix))
        later.append({**item, "path": name + "_eeg.bdf", "session": session, "run": run})
    protocol = dict(kind="last_session_holdout", session_order=["earlier", "later"],
                    run_order=["first", "second"], test_session="later", minimum_test_trials=2)
    return [later[1], later[0], first], protocol  # File list order does not define chronology.


def test_multiple_record_prefixes_use_explicit_order_and_partial_last_record(real_record, tmp_path):
    from data.source_records import prepare_source_subject_prefixes, target_raw_view_identity
    from transfer.epoch_selection import IndependentEpochSelectionData

    item, spec, _, _ = real_record
    items, protocol = _multiple_records(item, tmp_path)
    pair = prepare_source_subject_prefixes(items, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0, 10, 25, 40), protocol=protocol)
    loaded = load_target_prefixes(_collection(tmp_path, pair))["1"]
    assert pair["effective_budgets"] == [0, 10, 25, 40]
    assert loaded["calibration"][0] is None
    assert loaded["test"].n_epochs == 20
    c10, c25, c40 = (loaded["calibration"][k] for k in (10, 25, 40))
    assert (c10.n_epochs, c25.n_epochs, c40.n_epochs) == (10, 25, 40)
    assert c25.event_timeline.run_ids.tolist() == ["first"] * 20 + ["second"] * 5
    assert set(c25.event_timeline.session_ids) == {"earlier"}
    assert set(loaded["test"].event_timeline.session_ids) == {"later"}
    first, partial = pair["calibration"]["25"]
    assert first["preparation"]["raw_scope"]["stop_sample"] == 6500
    assert partial["preparation"]["raw_scope"]["stop_sample"] == 1801
    assert partial["authorized_label_count"] == 5
    assert not np.array_equal(c25.X[20:], c40.X[20:25])
    inner = loaded["early_stopping_inner"][25]
    assert inner["status"] == "ready"
    assert inner["training"].n_epochs == 20 and inner["validation"].n_epochs == 5
    assert set(inner["training"].event_timeline.run_ids) == {"first"}
    assert set(inner["validation"].event_timeline.run_ids) == {"second"}
    composite = target_raw_view_identity(c25)
    assert composite["kind"] == "ordered_attested_cache_composition"
    assert len(composite["components"]) == 2
    assert composite["sha256"] != target_raw_view_identity(c40)["sha256"]
    assert len(c25.provenance["target_raw_record_scopes"]) == 2
    parent_inventory = c25.provenance["target_raw_record_scopes"]
    assert [row["raw_identity"]["run"] for row in parent_inventory] == ["first", "second"]
    selection = IndependentEpochSelectionData.from_views(inner["training"], inner["validation"], c25)
    assert selection.training_mask.sum() == 20 and selection.validation_mask.sum() == 5
    pair["calibration"]["25"].reverse()
    with pytest.raises(ValueError, match="recording order"):
        load_target_prefixes(_collection(tmp_path, pair))


def test_single_record_inner_views_rebuild_inside_c_and_reject_scope_tampering(real_record, tmp_path):
    from transfer.epoch_selection import IndependentEpochSelectionData

    item, spec, _, _ = real_record
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0, 16), minimum_test_trials=2)
    path = _collection(tmp_path, pair)
    loaded = load_target_prefixes(path)["1"]
    inner = loaded["early_stopping_inner"][16]
    assert inner["status"] == "ready"
    train, valid, whole = inner["training"], inner["validation"], loaded["calibration"][16]
    assert (train.n_epochs, valid.n_epochs, whole.n_epochs) == (12, 3, 16)
    assert not np.array_equal(train.X, whole.X[:12])
    t_scope, v_scope, c_scope = [dataset.provenance["target_raw_record_scopes"][0]
                               for dataset in (train, valid, whole)]
    assert t_scope["raw_identity"] == v_scope["raw_identity"] == c_scope["raw_identity"]
    assert t_scope["raw_scope"]["stop_sample"] == v_scope["raw_scope"]["start_sample"]
    assert v_scope["raw_scope"]["stop_sample"] == c_scope["raw_scope"]["stop_sample"]
    selection = IndependentEpochSelectionData.from_views(train, valid, whole)
    assert selection.training_mask.sum() == 12 and selection.validation_mask.sum() == 3
    pair["early_stopping_inner"]["16"]["validation"]["preparation"]["raw_scope"]["start_sample"] -= 1
    with pytest.raises(ValueError, match="declared original input"):
        load_target_prefixes(_collection(tmp_path, pair))


def test_multiple_record_budget_fallback_and_unconsumed_early_raw(real_record, tmp_path, monkeypatch):
    import data.source_records as records

    item, spec, _, _ = real_record
    items, protocol = _multiple_records(item, tmp_path)
    pair = records.prepare_source_subject_prefixes(items, root=tmp_path, output=tmp_path / "fallback",
        preprocessing=spec, budgets=(0, 200), protocol=protocol)
    assert pair["effective_budgets"] == [0, 32]
    assert load_target_prefixes(_collection(tmp_path, pair))["1"]["calibration"][32].n_epochs == 32
    clean = records.prepare_source_subject_prefixes(items, root=tmp_path, output=tmp_path / "clean",
        preprocessing=spec, budgets=(0, 10), protocol=protocol)
    expected = load_target_prefixes(_collection(tmp_path, clean))["1"]["calibration"][10]
    original = records._read_record
    def contaminated(item, path):
        raw, events, labels = original(item, path)
        if item["run"] == "second":
            raw._data[:] = np.nan
            labels[:] = 999
        return raw, events, labels
    monkeypatch.setattr(records, "_read_record", contaminated)
    changed = records.prepare_source_subject_prefixes(items, root=tmp_path, output=tmp_path / "changed",
        preprocessing=spec, budgets=(0, 10), protocol=protocol)
    actual = load_target_prefixes(_collection(tmp_path, changed))["1"]["calibration"][10]
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.y, expected.y)


def test_partial_last_record_can_have_one_class_and_reports_unavailable_inner(real_record, tmp_path):
    from data.source_records import prepare_source_subject_prefixes

    item, spec, _, _ = real_record
    items, protocol = _multiple_records(item, tmp_path)
    pair = prepare_source_subject_prefixes(items, root=tmp_path, output=tmp_path / "shards",
        preprocessing=spec, budgets=(0, 21), protocol=protocol)
    loaded = load_target_prefixes(_collection(tmp_path, pair))["1"]
    assert loaded["calibration"][21].n_epochs == 21
    assert set(loaded["calibration"][21].y) == {0, 1}
    assert loaded["early_stopping_inner"][21]["status"] == "unavailable"
    assert "both binary classes" in loaded["early_stopping_inner"][21]["reason"]


def test_cli_multiple_record_prefixes_and_resume(real_record, tmp_path, monkeypatch):
    item, spec, manifest, runner = real_record
    items, protocol = _multiple_records(item, tmp_path)
    declaration = json.loads(manifest.read_text())
    declaration.update(records=items, target_protocol=protocol, preprocessing=spec.record())
    manifest.write_text(json.dumps(declaration), encoding="utf-8")
    output = tmp_path / "run"
    argv = ["--manifest", str(manifest), "--output", str(output), "--workers", "1",
            "--target-prefix-budgets", "0", "10", "25", "40"]
    runner.main(argv)
    loaded = load_target_prefixes(output / "target_prefixes.json")["1"]
    assert loaded["calibration"][25].n_epochs == 25
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr("data.source_records._read_record", lambda *a: pytest.fail("completed resume must not parse raw"))
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("completed resume must not launch workers"))
    runner.main(argv)
    assert all(path.read_bytes() == value for path, value in before.items())
