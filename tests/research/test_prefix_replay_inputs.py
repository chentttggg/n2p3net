"""Shared-input immutability, original-producer authority, and prefix semantics."""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.cloud_execution import CloudResult
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import (
    PrefixReplayInputHolder,
    export_prepared_views,
    immutable_input_array,
    open_prepared_inputs,
    read_scoring_truth,
)
from tests.support.independent_prefix import independent_prefix_case
from transfer.ensemble_views import calibration_prefix_identity


class Lease:
    def __init__(self):
        self.released = False

    def release_unused(self):
        self.released = True


class Admission:
    def __init__(self, available=True):
        self.available, self.lease, self.claims = available, Lease(), []

    def try_resident_input(self, nbytes):
        self.claims.append(nbytes)
        return self.lease if self.available else None


def _outcome(descriptor, request):
    return CloudResult(request["preparation_task_id"], "completed", value=descriptor, worker={
        "alive": False, "exit_code": 0,
        "process_tree_retirement": {"active_owned_processes_after_exit": 0}})


def _request(views, spec, profiles):
    import transfer.ensemble_views as owner

    prefix = views.calibration[max(views.calibration)]
    return {"schema": "n2p3_prefix_replay_input_request/1", "preparation_task_id": "prepare-original-inputs",
        "original_source_sha256": prefix.target_contract["source_sha256"],
        "producer_source_sha256": hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        "participant_key": prefix.target_contract["participant_key"],
        "source_relative_path": prefix.target_contract["source_relative_path"],
        "requested_Ks": views.ledger["requested_Ks"], "actual_Ks": sorted(views.calibration),
        "profiles": [profile.record() for profile in profiles], "authorization_preprocessing": spec.record(),
        "observation_response_seconds": 1.2, "pre_roll_seconds": .2}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-prefix-inputs")
    _, _, _, spec, experts, views = independent_prefix_case(root)
    profiles = (experts[1].profile,)
    request = _request(views, spec, profiles)
    descriptor = export_prepared_views(views, profiles, root / "prepared", request)
    return views, profiles, request, descriptor


def _holder(prepared, admission=None):
    _, _, request, descriptor = prepared
    return PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
        prep_outcome=_outcome(descriptor, request), admission=admission or Admission())


def _open(holder, prepared):
    _, _, request, descriptor = prepared
    return open_prepared_inputs(holder.descriptor, expected_request_sha=semantic_sha256(request),
                                expected_capsule_sha=descriptor["sha256"])


def test_terminal_bytes_share_storage_and_headers_are_independent():
    original = np.arange(24, dtype=np.float64).reshape(4, 6)
    frozen = immutable_input_array(original, original.dtype)
    first = immutable_input_array(frozen, frozen.dtype)
    second = immutable_input_array(frozen, frozen.dtype)
    assert np.shares_memory(first, second) and first is not second
    first.shape = (24,)
    first.dtype = np.uint8
    assert frozen.shape == second.shape == (4, 6) and second.dtype == np.float64
    original[:] = -1
    np.testing.assert_array_equal(second, np.arange(24).reshape(4, 6))
    with pytest.raises(ValueError):
        second.setflags(write=True)
    assert immutable_input_array(np.empty((0, 2)), np.float64).shape == (0, 2)


def test_readonly_flags_memoryview_and_ordinary_readonly_mmap_are_copied(tmp_path):
    original = np.arange(8, dtype=np.float64)
    original.setflags(write=False)
    frozen = immutable_input_array(memoryview(original).toreadonly(), original.dtype)
    original.setflags(write=True)
    original[:] = 99
    np.testing.assert_array_equal(frozen, np.arange(8))
    path = tmp_path / "mutable.bin"
    path.write_bytes(np.arange(8, dtype=np.float64).tobytes())
    times = path.stat()
    with path.open("rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        value = np.frombuffer(mapping, dtype=np.float64)
        frozen = immutable_input_array(value, value.dtype)
        with path.open("r+b") as writer:
            writer.write(np.full(8, -1., dtype=np.float64).tobytes())
        os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns))
        np.testing.assert_array_equal(frozen, np.arange(8))
        del value
        mapping.close()


def test_restore_keeps_all_original_authority_and_no_future_truth(prepared):
    original, profiles, request, descriptor = prepared
    admission = Admission()
    holder = _holder(prepared, admission)
    with _open(holder, prepared) as borrowed:
        views = borrowed.views
        assert views.scoring_truth == {}
        assert semantic_sha256(views.ledger) == semantic_sha256(original.ledger)
        assert [row.record() for row in views.decisions] == [row.record() for row in original.decisions]
        for budget, prefix in original.calibration.items():
            actual = views.prefix_compiler.compile(budget)
            assert calibration_prefix_identity(actual) == calibration_prefix_identity(prefix)
            for profile in profiles:
                assert semantic_sha256(actual.trial_target(profile).provenance) == semantic_sha256(
                    prefix.trial_target(profile).provenance)
            for group in prefix.complete_groups:
                assert calibration_prefix_identity(views.prefix_compiler.before_group(actual, group)) == (
                    calibration_prefix_identity(original.prefix_compiler.before_group(prefix, group)))
        future_samples = [row.samples for row in views.decisions if row.samples is not None]
        assert future_samples
        with pytest.raises(ValueError):
            future_samples[0].setflags(write=True)
        with pytest.raises(ValueError, match="not compiled"):
            views.prefix_compiler.compile(17)
        truth_path = Path(descriptor["path"]).parent / "scoring_truth.json"
        payload = truth_path.read_bytes()
        truth = read_scoring_truth({"path": str(truth_path), "sha256": hashlib.sha256(payload).hexdigest(),
                                   "bytes": len(payload)}, capsule_sha=descriptor["sha256"],
                                  request_sha=semantic_sha256(request), future_views=views.decisions)
        assert truth == original.scoring_truth
        assert borrowed.receipt["future_truth_present"] is False
    holder.close()
    capsule = json.loads(Path(descriptor["path"]).read_bytes())
    assert admission.lease.released
    assert holder._backing is None and holder._capsule is None
    assert admission.claims == [capsule["blob"]["bytes"] + descriptor["bytes"]]


@pytest.mark.parametrize("mutation", ["exit", "tree", "alive", "failed", "task", "value"])
def test_actual_preparation_exit_and_tree_must_match_before_adoption(prepared, mutation):
    _, _, request, descriptor = prepared
    outcome = _outcome(descriptor, request)
    if mutation == "exit":
        outcome.worker["exit_code"] = 1
    elif mutation == "tree":
        outcome.worker["process_tree_retirement"]["active_owned_processes_after_exit"] = 1
    elif mutation == "alive":
        outcome.worker["alive"] = True
    elif mutation == "failed":
        outcome.status = "failed"
    elif mutation == "task":
        outcome.task_id = "another-producer"
    else:
        outcome.value = {**descriptor, "sha256": "f" * 64}
    admission = Admission()
    with pytest.raises(ValueError):
        PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
                                             prep_outcome=outcome, admission=admission)
    assert not admission.claims


def test_no_resident_capacity_defers_without_reading_numeric_blob(prepared, monkeypatch):
    admission = Admission(False)
    numeric_path = Path(prepared[-1]["path"]).parent / "arrays.bin"
    real_open = Path.open

    def forbid(path, *args, **kwargs):
        if path == numeric_path:
            raise AssertionError("unadmitted numeric allocation/read")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid)
    assert _holder(prepared, admission) is None
    assert admission.claims and not admission.lease.released


def test_actual_artifact_bundle_selects_capsule_without_replacing_outcome(prepared, monkeypatch):
    import research.prefix_replay_inputs as owner

    _, _, request, descriptor = prepared
    outcome = _outcome(descriptor, request)
    observed = []
    validate = owner._validate_preparation

    def observe(descriptor, request_sha, actual_outcome, result_key):
        observed.append(actual_outcome)
        return validate(descriptor, request_sha, actual_outcome, result_key)

    monkeypatch.setattr(owner, "_validate_preparation", observe)
    bundle = {"capsule": descriptor, "preparation": {"path": "owned/preparation.json", "sha256": "a" * 64, "bytes": 1}}
    outcome.value = bundle
    holder = PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
        prep_outcome=outcome, admission=Admission(), result_key="capsule")
    assert holder is not None and observed[0] is outcome and outcome.value is bundle
    holder.close()
    for key in (None, "absent", "preparation"):
        with pytest.raises(ValueError):
            PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
                prep_outcome=outcome, admission=Admission(), result_key=key)
    outcome.worker["exit_code"] = 1
    with pytest.raises(ValueError):
        PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
            prep_outcome=outcome, admission=Admission(), result_key="capsule")


def test_blob_tampering_even_same_size_and_restored_mtime_is_rejected(prepared):
    path = Path(prepared[-1]["path"]).parent / "arrays.bin"
    times = path.stat()
    with path.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0] ^ 1]))
    os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns))
    admission = Admission()
    try:
        with pytest.raises(ValueError, match="numeric bytes differ"):
            _holder(prepared, admission)
        assert admission.lease.released
    finally:
        with path.open("r+b") as stream:
            stream.write(original)
        os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns))


def test_producer_request_cannot_relabel_current_compilation_as_old(prepared, tmp_path):
    views, profiles, request, _ = prepared
    with pytest.raises(ValueError, match="frozen implementation"):
        export_prepared_views(views, profiles, tmp_path / "false-old",
                              {**request, "producer_source_sha256": "f" * 64})
    assert not (tmp_path / "false-old").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux kernel memfd seals")
def test_linux_seals_share_numeric_storage_and_check_parent_fd(prepared):
    import fcntl

    holder = _holder(prepared)
    descriptor = holder.descriptor
    assert isinstance(descriptor["capsule"], dict) and descriptor["capsule"]["backing"] == "sealed_memfd"
    assert len(json.dumps(descriptor)) < 2048
    fd = os.open(f"/proc/{descriptor['parent_pid']}/fd/{descriptor['fd']}", os.O_RDWR)
    try:
        with pytest.raises(PermissionError):
            os.pwrite(fd, b"x", 0)
        with pytest.raises(PermissionError):
            os.ftruncate(fd, descriptor["bytes"] - 1)
        assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) == descriptor["seals"]
    finally:
        os.close(fd)
    with _open(holder, prepared) as first, _open(holder, prepared) as second:
        prefix = first.views.calibration[200]
        original = prefix.raw.samples
        same = immutable_input_array(original, original.dtype)
        assert np.shares_memory(original, same)
        same.shape = (same.size,)
        assert prefix.raw.samples.shape == second.views.calibration[200].raw.samples.shape
    bad = {**descriptor, "parent_birth": descriptor["parent_birth"] + 1}
    with pytest.raises(ValueError, match="parent identity"):
        open_prepared_inputs(bad, expected_request_sha=descriptor["request_sha256"],
                             expected_capsule_sha=descriptor["capsule_sha256"])
    holder.close()


def test_real_frozen_old_producer_capability_survives_new_consumer(tmp_path):
    """Execute actual previous source bytes in a separate process, not SHA surgery."""
    repository = Path(__file__).resolve().parents[2]
    original_owner = "c432d3d32a1a876c99455c82ad1d4d2e53f4f641"
    prior = subprocess.run(["git", "show", original_owner + ":src/transfer/ensemble_views.py"], cwd=repository,
                           capture_output=True, check=True).stdout
    source = tmp_path / "original_ensemble_views.py"
    source.write_bytes(prior)
    script = '''
import importlib.util, json, pathlib, sys
root, producer = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("transfer.ensemble_views", producer)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from tests.support.independent_prefix import independent_prefix_case
from research.prefix_replay_inputs import export_prepared_views
from research.prefix_inner_replay import prefix_input_identity, prefix_native_fit_inputs
from tests.research.test_prefix_replay_inputs import _request
case = root / "case"
case.mkdir()
_, _, path, preprocessing, experts, views = independent_prefix_case(case)
profiles = (experts[1].profile,)
views = module.prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
    budgets=(0, 400), profiles=profiles, authorization_preprocessing=preprocessing)
request = _request(views, preprocessing, profiles)
descriptor = export_prepared_views(views, profiles, root / "old_prepared", request)
(root / "old_output.json").write_text(json.dumps({"request": request, "descriptor": descriptor,
    "experts": [expert.record() for expert in experts],
    "identities": {str(k): prefix_input_identity(prefix) for k, prefix in views.prefix_compiler._prefixes.items()},
    "native": {str(k): prefix_native_fit_inputs(prefix, experts) for k, prefix in views.prefix_compiler._prefixes.items()}}))
'''
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repository / "src"), str(repository)]),
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path), str(source)],
                   cwd=repository, env=environment, check=True, capture_output=True)
    result = json.loads((tmp_path / "old_output.json").read_bytes())
    request, descriptor = result["request"], result["descriptor"]
    holder = PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(request),
        prep_outcome=_outcome(descriptor, request), admission=Admission())
    with open_prepared_inputs(holder.descriptor, expected_request_sha=semantic_sha256(request),
                              expected_capsule_sha=descriptor["sha256"]) as borrowed:
        assert borrowed.receipt["producer_source_sha256"] == hashlib.sha256(prior).hexdigest()
        assert borrowed.receipt["consumer_selection_source_sha256"] != hashlib.sha256(prior).hexdigest()
        profile = __import__("transfer.ensemble_views", fromlist=["SelectionEpochProfile"]).SelectionEpochProfile.from_record(
            request["profiles"][0])
        for prefix in borrowed.views.calibration.values():
            capability = prefix.trial_target(profile).provenance["selection_epoch_capability"]
            assert capability["producer_source_sha256"] == hashlib.sha256(prior).hexdigest()
        from research.prefix_inner_replay import prefix_input_identity, prefix_native_fit_inputs
        from transfer.ensemble_fit import CalibrationExpertProgram

        experts = tuple(CalibrationExpertProgram.from_record(row) for row in result["experts"])
        assert set(borrowed.receipt["prepared_actual_Ks"]) > set(request["actual_Ks"])
        for key, identity in result["identities"].items():
            prefix = borrowed.views.prefix_compiler.compile(int(key))
            assert semantic_sha256(prefix_input_identity(prefix)) == semantic_sha256(identity)
            assert semantic_sha256(prefix_native_fit_inputs(prefix, experts)) == semantic_sha256(result["native"][key])
    holder.close()
