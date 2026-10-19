"""Behavioral audit controls against the current research data owners."""
import json
from pathlib import Path

import mne
import numpy as np
import pytest

from data.channel import resolve_channel_layout
from data.epochs import PreprocessingSpec
from data.events import concatenate_event_timelines, observed_only_timeline
from data.identity import (
    DatasetIdentityTable,
    ParticipantIdentityRecord,
    assert_target_identity_excluded,
    origin_subject_key,
)
from data.manifest import load_manifest


def test_manifest_rejects_string_or_canonical_duplicate_channels_before_reader(tmp_path):
    payload = {"schema": "n2p3net_raw_manifest/2", "name": "probe",
        "preprocessing": PreprocessingSpec().record(),
        "records": [{"path": "not-opened.fif", "subject_id": "p", "reference": "nose"}]}
    path = tmp_path / "input.json"
    for invalid in ("Fz", ["Fz", "EEG FZ-ref"], []):
        path.write_text(json.dumps({**payload, "channels": invalid}), encoding="utf8")
        with pytest.raises(ValueError, match="channels"):
            load_manifest(path)


def test_observed_event_identity_separates_sessions_but_rejects_same_source_twice():
    def timeline(session):
        return observed_only_timeline(dataset_id="same-study", subject_ids=np.array(["p", "p"]),
            stimulus_ids=np.array([1, 2]), session_ids=np.array([session, session]),
            run_ids=np.array(["run", "run"]), onset_times_s=np.array([1., 2.]))
    first, second = timeline("s1"), timeline("s2")
    joined = concatenate_event_timelines((first, second))
    assert len(set(joined.event_ids)) == 4
    assert np.array_equal(first.event_ids, timeline("s1").event_ids)
    with pytest.raises(ValueError, match="unique"):
        concatenate_event_timelines((first, first))


def test_partial_global_evidence_is_unknown_and_cannot_prove_exclusion():
    source = ParticipantIdentityRecord("source", (origin_subject_key("study-a", "p"),), (), "source_verified")
    target = ParticipantIdentityRecord("target", (origin_subject_key("study-b", "p"),),
        ("registry:person",), "global_verified")
    with pytest.raises(ValueError, match="global.*complete|global.*every"):
        assert_target_identity_excluded(DatasetIdentityTable((source,)), target)
    assert_target_identity_excluded(DatasetIdentityTable((source,)), target, policy="source")
    with pytest.raises(ValueError, match="global.*complete|global.*every"):
        assert_target_identity_excluded(DatasetIdentityTable((target,)), source)
    source_target = ParticipantIdentityRecord("new", (origin_subject_key("study-b", "q"),), (), "source_verified")
    assert_target_identity_excluded(DatasetIdentityTable((source,)), source_target)
    other_global = ParticipantIdentityRecord("other", (origin_subject_key("study-c", "q"),),
        ("registry:other-person",), "global_verified")
    assert_target_identity_excluded(DatasetIdentityTable((other_global,)), target)


def test_mixed_coordinates_keep_per_sensor_registration_and_fiducial_transform():
    shift = np.array([.4, -.2, .3])
    original = np.array([-.03, .02, .08]) + shift
    fiducials = {"lpa": np.array([-.07, 0., 0.])+shift,
        "rpa": np.array([.07, 0., 0.])+shift, "nasion": np.array([0., .1, 0.])+shift}
    montage = mne.channels.make_dig_montage(ch_pos={"X2": [.03, -.02, .08]}, coord_frame="head")
    layout = resolve_channel_layout(("X1", "X2"), positions_m={"X1": original},
        coordinate_frame="digitizer", fiducials_m=fiducials, montage=montage)
    record = layout.registration.record()
    assert record["transform_to_head"] is None
    components = record["component_registrations"]
    assert [row["channels"] for row in components] == [["X1"], ["X2"]]
    explicit = components[0]["registration"]
    assert explicit["fiducials_used"] == ["lpa", "rpa", "nasion"]
    mapped = np.asarray(explicit["transform_to_head"]) @ np.r_[original, 1.]
    np.testing.assert_allclose(mapped[:3], layout.positions_m[0], atol=1e-7)


def test_explicit_coordinate_alias_collision_is_rejected():
    with pytest.raises(ValueError, match="duplicate|unique"):
        resolve_channel_layout(("Fz",), positions_m={"Fz": [0., .02, .08], "EEG FZ-ref": [0., .03, .08]}, montage=None)


def test_attested_default_decodes_same_open_file_without_disk_snapshot(tmp_path, monkeypatch):
    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = owner.save_epoch_dataset(tmp_path / "cache.npz", epoch_dataset())
    def forbidden(*args, **kwargs):
        raise AssertionError("default attested loading copied a full snapshot or inspected snapshot disk space")
    monkeypatch.setattr(owner.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(owner.shutil, "disk_usage", forbidden)
    actual = owner.load_epoch_dataset(path, validation="attested")
    assert np.array_equal(actual.X, epoch_dataset().X)


def test_attested_in_place_write_during_decode_fails_closed(tmp_path, monkeypatch):
    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = owner.save_epoch_dataset(tmp_path / "cache.npz", epoch_dataset(), compressed=False)
    original = owner.np.load
    def mutate_then_decode(stream, *args, **kwargs):
        with path.open("r+b") as output:
            output.seek(-1, 2)
            output.write(b"\x00")
        return original(stream, *args, **kwargs)
    monkeypatch.setattr(owner.np, "load", mutate_then_decode)
    with pytest.raises((ValueError, OSError), match="changed|Permission|denied"):
        owner.load_epoch_dataset(path, validation="attested")


def test_exclusive_writer_race_publishes_exactly_one_complete_pair(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = tmp_path / "cache.npz"
    barrier = Barrier(2)
    def publish(_):
        barrier.wait()
        try:
            owner.save_epoch_dataset(path, epoch_dataset(), exclusive=True)
            return "published"
        except FileExistsError:
            return "refused"
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(publish, range(2))) == ["published", "refused"]
    owner.load_epoch_dataset(path, validation="attested")


def test_cache_pair_publication_failure_restores_previous_payload_and_record(tmp_path, monkeypatch):
    from pathlib import Path

    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = owner.save_epoch_dataset(tmp_path / "cache.npz", epoch_dataset())
    record_path = path.with_suffix(".record.json")
    before = path.read_bytes(), record_path.read_bytes()
    original = Path.replace
    failed = False
    def fail_record_once(self, target):
        nonlocal failed
        if Path(target) == record_path and self.suffix == ".json" and not failed:
            failed = True
            raise OSError("injected record publication failure")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", fail_record_once)
    with pytest.raises(OSError, match="injected"):
        owner.save_epoch_dataset(path, epoch_dataset())
    assert failed and before == (path.read_bytes(), record_path.read_bytes())
    owner.load_epoch_dataset(path, validation="attested")


def test_second_staging_allocation_failure_reclaims_first_temp_and_preserves_pair(tmp_path, monkeypatch):
    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = owner.save_epoch_dataset(tmp_path / "cache.npz", epoch_dataset())
    record_path = path.with_suffix(".record.json")
    before = path.read_bytes(), record_path.read_bytes()
    allocate = owner._temporary_path
    def fail_second(member, suffix):
        if member == record_path:
            raise OSError("injected second staging allocation failure")
        return allocate(member, suffix)
    monkeypatch.setattr(owner, "_temporary_path", fail_second)
    with pytest.raises(OSError, match="second staging"):
        owner.save_epoch_dataset(path, epoch_dataset())
    assert (path.read_bytes(), record_path.read_bytes()) == before
    assert not list(tmp_path.glob(".cache.npz-*.npz"))


@pytest.mark.parametrize("rate,expected_origin", [(128., -.203125), (250., -.2)])
def test_actual_grid_origin_is_recorded_without_retiming_existing_operator(rate, expected_origin):
    from data.preprocess import execute_preprocessing
    profile = PreprocessingSpec(sfreq=128., l_freq=None, h_freq=None, baseline_mode="none")
    raw = mne.io.RawArray(np.zeros((1, int(rate*5))), mne.create_info(["Cz"], rate, "eeg"), verbose=False)
    result = execute_preprocessing(raw, np.array([[int(rate*2), 0, 1]]), profile, channels=("Cz",))
    grid = result.execution_receipt["actual_epoch_grid"]
    assert grid["source_grid_origin_s"] == expected_origin
    assert grid["model_nominal_spacing_s"] == 1/128
    assert grid["model_nominal_last_sample_s"] == expected_origin+178/128
    assert grid["model_nominal_end_exclusive_s"] == expected_origin+179/128
    assert grid["baseline_index_origin_s"] == -.2
    import copy

    from data.epochs import _semantic_sha256, validate_preprocessing_execution
    changed = copy.deepcopy(result.execution_receipt)
    changed["actual_epoch_grid"]["source_grid_origin_s"] += .001
    changed["receipt_sha256"] = _semantic_sha256({key: value for key, value in changed.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="actual.*grid origin"):
        validate_preprocessing_execution(changed, profile)


def test_bids_current_ingress_content_binding_detects_same_metadata_different_samples(tmp_path):
    from data.brainsync_contract import validate_brainsync_bids_session
    from tests.support.brainsync import write_bids_session
    root = write_bids_session(tmp_path / "session", session_id="audit", target=1,
        started_utc="2026-01-01T00:00:00+00:00")
    validated = validate_brainsync_bids_session(root)
    original = validated.bids.content_record()
    assert original["origin"] == "current_ingress_not_acquisition_time_attestation"
    recording = validated.bids.raw_path
    import os
    if os.name == "nt":
        with pytest.raises(PermissionError):
            recording.open("r+b")
        with pytest.raises(PermissionError):
            recording.rename(recording.with_suffix(".moved"))
    validated.bids.close()
    with pytest.raises(ValueError, match="closed"):
        validated.bids.assert_content_unchanged()
    with recording.open("r+b") as stream:
        stream.seek(-1, 2)
        original_byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original_byte[0] ^ 1]))
    fresh = validate_brainsync_bids_session(root)
    assert fresh.bids.content_record()["files"] != original["files"]
    fresh.bids.close()
    session_path = root / "session.json"
    manifest = json.loads(session_path.read_bytes())
    manifest["recording"]["bids_content"] = {"schema": "bids_acquisition_content/1", "files": original["files"]}
    session_path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="acquisition-time"):
        validate_brainsync_bids_session(root)


def test_brainsync_actual_reader_holds_lease_and_releases_on_success_and_failure(tmp_path, monkeypatch):
    import os

    from data import brainsync as owner
    from tests.support.brainsync import write_bids_session
    root = write_bids_session(tmp_path / "session", session_id="actual-reader", target=1,
        started_utc="2026-01-01T00:00:00+00:00")
    manifest = json.loads((root / "session.json").read_bytes())
    recording = root / manifest["recording"]["path"]
    original_read = owner.read_raw
    observations = []
    def check_lease(path, **kwargs):
        if os.name == "nt":
            with pytest.raises(PermissionError):
                recording.open("r+b")
        observations.append("actual_raw_reader")
        return original_read(path, **kwargs)
    monkeypatch.setattr(owner, "read_raw", check_lease)
    dataset = owner.load_brainsync_session(root)
    assert observations == ["actual_raw_reader"]
    assert dataset.provenance["bids_content"]["files"]["recording"]["sha256"]
    with recording.open("r+b"):
        pass
    retained = []
    def reject_before_builder(validated):
        retained.append(validated)
        raise ValueError("injected recording admission callback rejection")
    with pytest.raises(ValueError, match="callback rejection"):
        owner.load_brainsync_inference_session(root, recording_validator=reject_before_builder)
    with recording.open("r+b"):
        pass

    def fail_read(*args, **kwargs):
        check_lease(*args, **kwargs).close()
        raise OSError("injected actual reader failure")
    monkeypatch.setattr(owner, "read_raw", fail_read)
    with pytest.raises(OSError, match="injected") as retained_failure:
        owner.load_brainsync_session(root)
    assert retained_failure.value.__traceback__ is not None
    with recording.open("r+b"):
        pass


def test_batch_metadata_rejection_releases_all_leases_with_traceback_retained(tmp_path):
    from data import brainsync as owner
    from tests.support.brainsync import write_bids_session
    roots = [write_bids_session(tmp_path / name, session_id="duplicate-public-id", target=1,
        started_utc="2026-01-01T00:00:00+00:00") for name in ("a", "b")]
    with pytest.raises(ValueError, match="globally unique") as failure:
        owner.load_brainsync_sessions(roots)
    assert failure.value.__traceback__ is not None
    for root in roots:
        manifest = json.loads((root / "session.json").read_bytes())
        with (root / manifest["recording"]["path"]).open("r+b"):
            pass


def test_record_failure_preserves_a_competing_publication_and_original_backup(tmp_path, monkeypatch):
    import data.epochs as owner
    from tests.support.datasets import epoch_dataset
    path = owner.save_epoch_dataset(tmp_path / "cache.npz", epoch_dataset())
    record_path = path.with_suffix(".record.json")
    before = path.read_bytes()
    replace = Path.replace
    competing = b"independent writer bytes must not be removed"
    def conflict(self, target):
        if Path(target) == record_path and self.suffix == ".json":
            other = tmp_path / "foreign-cache.npz"
            other.write_bytes(competing)
            replace(other, path)
            raise OSError("injected record failure after competing publication")
        return replace(self, target)
    monkeypatch.setattr(Path, "replace", conflict)
    with pytest.raises(OSError, match="injected"):
        owner.save_epoch_dataset(path, epoch_dataset())
    assert path.read_bytes() == competing
    assert any(p.read_bytes() == before for p in tmp_path.glob(".cache.npz-*.rollback"))


def test_live_raw_issuer_holds_through_lazy_read_and_closes_without_revoking_numeric_cap(tmp_path, monkeypatch):
    import os

    from data import brainsync_raw_authority as owner
    from tests.data.test_brainsync_raw_authority import SESSION_ID, recording
    root = recording.__wrapped__(tmp_path)
    manifest = json.loads((root / "session.json").read_bytes())
    raw_path = root / manifest["recording"]["path"]
    issuer = owner.open_brainsync_recording(root, scope=owner.BrainSyncRecordingScope((), (SESSION_ID,)))
    original_read = issuer._raw.get_data
    observed = []
    def lazy_read(*args, **kwargs):
        if os.name == "nt":
            with pytest.raises(PermissionError):
                raw_path.open("r+b")
        observed.append("actual_lazy_read")
        return original_read(*args, **kwargs)
    monkeypatch.setattr(issuer._raw, "get_data", lazy_read)
    cap = issuer.issue_raw(0, prefix_stop=250, full_stop=1500, available_stop=1500)
    assert observed == ["actual_lazy_read"]
    issuer.close()
    with pytest.raises(ValueError, match="closed"):
        issuer.issue_raw(0, prefix_stop=250, full_stop=1500, available_stop=1500)
    assert owner.require_raw_capability(cap) is cap
    with raw_path.open("r+b"):
        pass


def test_offline_moabb_runs_actual_raw_preprocessing_and_records_operator(monkeypatch, tmp_path):
    from data import moabb as owner
    from tests.data.test_moabb_adapter import (
        _allow_narrow_moabb_test_double,
        _configure_attested_mne_root,
        _fake_dataset,
        _moabb_profile,
        _verified_artifact_attestation,
    )
    _allow_narrow_moabb_test_double(monkeypatch)
    attestation = _verified_artifact_attestation(tmp_path, "FakeP300")
    _configure_attested_mne_root(monkeypatch, attestation)
    dataset = _fake_dataset((1,), loader_path=Path(attestation.expected_mne_data_root)/"source.bin")
    rng = np.random.default_rng(19)
    raw = mne.io.RawArray(rng.normal(0, 1e-6, (2, 800)), mne.create_info(["Fz", "Cz"], 100., "eeg"), verbose=False)
    raw.set_montage("standard_1005")
    raw.set_annotations(mne.Annotations([2., 3., 4., 5.], [0.]*4, ["NonTarget", "Target"]*2))
    dataset.event_id = {"NonTarget": 1, "Target": 2}
    dataset.get_data = lambda subjects: {1: {"session-a": {"run-1": raw}}}
    monkeypatch.setattr(owner, "resolve_moabb_dataset", lambda name: dataset)
    def forbidden(*args, **kwargs):
        raise AssertionError("offline source entered unobserved MOABB P300 epoch/filter pipeline")
    monkeypatch.setattr("moabb.paradigms.P300", forbidden)
    profile = _moabb_profile(zero_phase_edge_policy="mne_reflect_limited", tmin_ms=-200., tmax_ms=800., n_times=100)
    result = owner.prepare_moabb_p300("FakeP300", raw_artifact_attestation=attestation, preprocessing=profile)
    bundle = result.provenance["preprocessing_execution"]
    assert bundle["unverified_inputs"] == 0
    assert len(bundle["recordings"]) == 1
    receipt = bundle["recordings"][0]
    assert receipt["filter"]["backend"] == "mne_reflect_limited"
    assert receipt["filter"]["sos"]
    assert receipt["declaration"] == profile.record()
    assert result.provenance["source"] == "moabb_raw_offline_p300"
    assert result.provenance["raw_preprocessing_program"]["offline_highlevel_epoch_program_used"] is False
    import hashlib
    assert result.provenance["raw_preprocessing_program"]["adapter_source_sha256"] == hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()
    assert result.event_timeline.online_causal is False
    assert result.event_timeline.candidate_ids.tolist() == [""]*4
    assert result.event_timeline.supports_full_candidate_chain is False
