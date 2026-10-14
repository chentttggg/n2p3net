"""Shared-input immutability, original-producer authority, and prefix semantics."""
from __future__ import annotations

import errno
import hashlib
import json
import mmap
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from research.cloud_execution import CloudResult
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import (
    PrefixReplayInputHolder,
    export_prepared_views,
    immutable_input_array,
    immutable_input_array_geometry,
    open_prepared_inputs,
    read_scoring_truth,
)
from tests.support.prefix_replay_inputs import prepared_prefix_inputs
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


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return prepared_prefix_inputs(tmp_path_factory.mktemp("shared-prefix-inputs"))


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
def test_linux_missing_python_memfd_bindings_preserve_holder_reader_contract(prepared, tmp_path):
    """Real dual-fd holder and reader under the observed missing-binding runtime."""
    _, _, request, descriptor = prepared
    contract = tmp_path / "missing_python_memfd_bindings.json"
    contract.write_text(json.dumps({"request": request, "descriptor": descriptor}), encoding="utf-8")
    script = '''
import errno, fcntl, json, mmap, os, pathlib, sys
for module, names in ((os, ("memfd_create", "MFD_CLOEXEC", "MFD_ALLOW_SEALING")),
        (fcntl, ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL"))):
    for name in names:
        if hasattr(module, name):
            delattr(module, name)
import numpy as np
from research.cloud_execution import CloudResult
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import (
    PrefixReplayInputHolder, immutable_input_array, immutable_input_backing, open_prepared_inputs)

class Lease:
    released = 0
    def release_unused(self):
        self.released += 1
class Admission:
    def __init__(self):
        self.lease = Lease()
        self.claims = []
    def try_resident_input(self, nbytes):
        self.claims.append(nbytes)
        return self.lease

contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
request, prepared = contract["request"], contract["descriptor"]
outcome = CloudResult(request["preparation_task_id"], "completed", value=prepared,
    worker={"alive": False, "exit_code": 0,
            "process_tree_retirement": {"active_owned_processes_after_exit": 0}})
admission = Admission()
os.close(0)
holder = PrefixReplayInputHolder.from_prepared(prepared,
    request_sha=semantic_sha256(request), prep_outcome=outcome, admission=admission)
assert holder is not None and admission.lease.released == 0
shared = holder.descriptor
assert shared["backing"] == shared["capsule"]["backing"] == "sealed_memfd"
owned_fds = [shared["fd"], shared["capsule"]["fd"]]
assert owned_fds[0] == 0
for backing in [shared, shared["capsule"]]:
    fd = backing["fd"]
    assert not os.get_inheritable(fd)
    assert fcntl.fcntl(fd, 1034) == backing["seals"] == 0x0f
    for operation in [lambda: os.pwrite(fd, b"x", 0),
            lambda: os.ftruncate(fd, backing["bytes"] - 1),
            lambda: os.ftruncate(fd, backing["bytes"] + 1),
            lambda: fcntl.fcntl(fd, 1033, 0x10)]:
        try:
            operation()
        except OSError as exc:
            assert exc.errno == errno.EPERM
        else:
            raise AssertionError("kernel-sealed backing accepted mutation")
    reader_fd = os.open(f"/proc/{backing['parent_pid']}/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
    try:
        assert os.fstat(reader_fd).st_ino == backing["inode"]
        assert not os.get_inheritable(reader_fd)
        with mmap.mmap(reader_fd, 0, access=mmap.ACCESS_READ) as mapping:
            assert mapping.size() == backing["bytes"]
            try:
                mapping[0] = 0
            except TypeError:
                pass
            else:
                raise AssertionError("reader mapping accepted write")
    finally:
        os.close(reader_fd)
with open_prepared_inputs(shared, expected_request_sha=semantic_sha256(request),
        expected_capsule_sha=prepared["sha256"]) as borrowed:
    original = borrowed.views.calibration[200].raw.samples
    assert isinstance(immutable_input_backing(original), mmap.mmap)
    assert np.shares_memory(original, immutable_input_array(original, original.dtype))
    try:
        original.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("numeric reader became writable")
assert not hasattr(os, "memfd_create") and not hasattr(fcntl, "F_GET_SEALS")
del original
holder.close()
holder.close()
assert admission.lease.released == 1
for fd in owned_fds:
    try:
        os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        raise AssertionError("retired holder leaked a file descriptor")
print(json.dumps({"dual_fd_seals": 15, "readonly_numeric_and_metadata": True,
    "no_numeric_copy": True, "close_on_exec": True, "released_once": True,
    "all_holder_fds_closed": True, "python_bindings_remain_absent": True}))
'''
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", script, str(contract)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["dual_fd_seals"] == 15 and receipt["all_holder_fds_closed"]



@pytest.mark.skipif(sys.platform != "linux", reason="Linux kernel memfd seals")
def test_linux_kernel_memfd_creation_error_preserves_errno_and_releases_input_lease(prepared, monkeypatch):
    import research.prefix_replay_inputs as owner

    api = owner._linux_memfd_api()
    create, numeric_fds = api.create, []

    def fail_capsule_creation(name):
        if name == "n2p3-prefix-capsule":
            return create("n2p3-prefix-" + "x" * 400)
        fd = create(name)
        numeric_fds.append(fd)
        return fd

    monkeypatch.setattr(api, "create", fail_capsule_creation)
    admission = Admission()
    with pytest.raises(OSError) as failure:
        _holder(prepared, admission)
    assert failure.value.errno == errno.EINVAL
    assert admission.lease.released and len(numeric_fds) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(numeric_fds[0])
    assert closed.value.errno == errno.EBADF



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
        assert fcntl.fcntl(fd, 1034) == descriptor["seals"]
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
from tests.support.prefix_replay_inputs import prefix_input_request as _request
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


def test_owned_identity_is_hashed_once_and_foreign_inputs_still_use_full_identity(prepared, monkeypatch):
    import transfer.ensemble_views as owner

    holder = _holder(prepared)
    original_identity, calls = owner.calibration_prefix_identity, []

    def measured(prefix):
        calls.append(prefix)
        return original_identity(prefix)

    monkeypatch.setattr(owner, "calibration_prefix_identity", measured)
    with _open(holder, prepared) as borrowed:
        compiler = borrowed.views.prefix_compiler
        owned = borrowed.views.calibration[200]
        assert len(calls) == len(compiler._prefixes)
        calls.clear()
        for _ in range(3):
            assert compiler.assert_original_prefix(owned) is owned
            for group in owned.complete_groups:
                compiler.before_group(owned, group)
        assert not calls
        foreign = replace(owned)
        assert foreign is not owned
        assert compiler.assert_original_prefix(foreign) is owned
        assert len(calls) == 1 and calls[0] is foreign
        values = owned.raw.samples.copy()
        covered = np.zeros(len(values), dtype=bool)
        for group in owned.complete_groups:
            covered[slice(*group.view.raw_scope)] = True
        outside = np.flatnonzero(~covered)
        assert len(outside)
        values[outside[0], 0] += 1
        changed = replace(owned, raw=replace(owned.raw, samples=values))
        with pytest.raises(ValueError, match="original source prefix owner"):
            compiler.assert_original_prefix(changed)
        assert len(calls) == 2 and calls[-1] is changed
    holder.close()


def _owned_array(prefix, name):
    data = next(iter(prefix.trial_datasets.values()))
    group = prefix.complete_groups[0]
    arrays = {"raw_samples": prefix.raw.samples, "raw_onsets": prefix.raw.onsets,
        "raw_labels": prefix.raw.target_mask, "trial_X": data.X, "trial_y": data.y,
        "trial_mask": data.trial_channel_mask, "timeline_onsets": data.event_timeline.onset_samples,
        "group_samples": group.view.samples, "group_onsets": group.view.onsets,
        "membership": group.view.decision.membership, "stimulus_indices": group.view.decision.stimulus_indices,
        "group_labels": group.labels}
    return arrays[name]


@pytest.mark.parametrize("name", ["raw_samples", "raw_onsets", "raw_labels", "trial_X", "trial_y",
                                  "timeline_onsets", "group_samples", "group_onsets",
                                  "membership", "stimulus_indices", "group_labels"])
@pytest.mark.parametrize("mutation", ["shape", "dtype", "strides"])
@pytest.mark.filterwarnings("ignore:Setting the (shape|dtype|strides) on a NumPy array:DeprecationWarning")
def test_owned_header_mutation_is_rejected_before_any_numeric_hash(prepared, monkeypatch, name, mutation):
    import transfer.ensemble_views as owner

    holder = _holder(prepared)
    with _open(holder, prepared) as borrowed:
        prefix = borrowed.views.calibration[200]
        array = _owned_array(prefix, name)
        if mutation == "shape":
            array.shape = (array.size,) if array.ndim != 1 else (1, array.size)
        elif mutation == "dtype":
            array.dtype = np.uint8 if array.dtype != np.dtype(np.uint8) else np.int8
        else:
            array.strides = (0,) * array.ndim

        def no_hash(prefix):
            raise AssertionError("changed owned geometry must be rejected before scanning numeric payloads")

        monkeypatch.setattr(owner, "calibration_prefix_identity", no_hash)
        with pytest.raises(ValueError, match="array header or immutable backing changed"):
            borrowed.views.prefix_compiler.assert_original_prefix(prefix)
    holder.close()


def test_owned_absent_trial_mask_cannot_be_replaced_by_a_self_signed_array(prepared):
    holder = _holder(prepared)
    with _open(holder, prepared) as borrowed:
        prefix = borrowed.views.calibration[200]
        data = next(iter(prefix.trial_datasets.values()))
        assert data.trial_channel_mask is None
        mask = immutable_input_array(np.ones((data.n_epochs, data.n_channels), dtype=bool), bool)
        object.__setattr__(data, "trial_channel_mask", mask)
        with pytest.raises(ValueError, match="physical or label authority changed"):
            borrowed.views.prefix_compiler.assert_original_prefix(prefix)
    holder.close()


@pytest.mark.parametrize("mutation", ["shape", "dtype", "strides"])
@pytest.mark.filterwarnings("ignore:Setting the (shape|dtype|strides) on a NumPy array:DeprecationWarning")
def test_explicit_mask_software_prefix_header_mutations_are_rejected(prepared, mutation):
    """A current software fixture; its changed mask is not an old producer receipt."""
    from dataclasses import fields

    from data.epochs import EpochDataset
    from transfer.ensemble_views import CalibrationPrefix, _OwnedCalibrationPrefix
    from transfer.noise_model import array_digest

    original = prepared[0].calibration[200]
    key, data = next(iter(original.trial_datasets.items()))
    values = {item.name: getattr(data, item.name) for item in fields(EpochDataset) if item.init}
    values["trial_channel_mask"] = np.ones((data.n_epochs, data.n_channels), dtype=bool)
    masked = EpochDataset(**values)
    use = json.loads(json.dumps(original.label_use))
    use["trial"][key]["numerical_input_digest"] = array_digest(masked.X, masked.y, masked.trial_channel_mask)
    prefix = CalibrationPrefix(original.budget, original.raw, {key: masked}, original.complete_groups,
                               original.target_contract, use)
    ownership = _OwnedCalibrationPrefix(prefix, calibration_prefix_identity(prefix))
    ownership.assert_unchanged(prefix)
    mask = prefix.trial_datasets[key].trial_channel_mask
    if mutation == "shape":
        mask.shape = (mask.size,)
    elif mutation == "dtype":
        mask.dtype = np.uint8
    else:
        mask.strides = (0, 0)
    with pytest.raises(ValueError, match="array header or immutable backing changed"):
        ownership.assert_unchanged(prefix)


@pytest.mark.parametrize("scope", ["physical", "label_use", "group_truth", "capability"])
def test_owned_frozen_record_changes_are_rejected(prepared, scope):
    holder = _holder(prepared)
    with _open(holder, prepared) as borrowed:
        prefix = borrowed.views.calibration[200]
        if scope == "physical":
            dict.__setitem__(prefix.target_contract, "available_sample", -1)
        elif scope == "label_use":
            dict.__setitem__(prefix.label_use["raw"], "used_labels", 0)
        elif scope == "group_truth":
            object.__setattr__(prefix.complete_groups[0], "target_candidate", "changed_truth")
        else:
            capability = next(iter(prefix._prepared_capabilities.values()))
            dict.__setitem__(capability, "producer_source_sha256", "f" * 64)
        with pytest.raises(ValueError, match="physical or label authority changed"):
            borrowed.views.prefix_compiler.assert_original_prefix(prefix)
    holder.close()


def test_readonly_arrays_cannot_self_sign_owned_geometry_and_foreign_values_are_copied(prepared, tmp_path):
    holder = _holder(prepared)
    with _open(holder, prepared) as borrowed:
        prefix = borrowed.views.calibration[200]
        ordinary = prefix.raw.samples.copy()
        ordinary.setflags(write=False)
        with pytest.raises(ValueError, match="terminal bytes or verified kernel seals"):
            immutable_input_array_geometry(ordinary)
        foreign = replace(prefix, raw=replace(prefix.raw, samples=ordinary))
        assert not np.shares_memory(foreign.raw.samples, ordinary)
        assert borrowed.views.prefix_compiler.assert_original_prefix(foreign) is prefix
        path = tmp_path / "ordinary-readonly.bin"
        path.write_bytes(prefix.raw.samples.tobytes())
        with path.open("rb") as stream:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            readonly = np.ndarray(prefix.raw.samples.shape, dtype=prefix.raw.samples.dtype, buffer=mapping)
            with pytest.raises(ValueError, match="terminal bytes or verified kernel seals"):
                immutable_input_array_geometry(readonly)
            foreign = replace(prefix, raw=replace(prefix.raw, samples=readonly))
            assert not np.shares_memory(foreign.raw.samples, readonly)
            assert borrowed.views.prefix_compiler.assert_original_prefix(foreign) is prefix
            del readonly
            mapping.close()
    holder.close()


def _entrypoint_compiler(prepared, tmp_path, origin):
    from data.epochs import parse_preprocessing_spec
    from tests.support.prefix_replay_inputs import prefix_input_request
    from transfer.ensemble_views import prepare_bnci_independent_prefixes

    _, profiles, request, original_descriptor = prepared
    source = Path(original_descriptor["path"]).parent.parent / request["source_relative_path"]
    views = prepare_bnci_independent_prefixes(source, source_sha256=request["original_source_sha256"],
        budgets=(0, 400), profiles=profiles,
        authorization_preprocessing=parse_preprocessing_spec(request["authorization_preprocessing"]))
    if origin == "original":
        return views.prefix_compiler, None, None
    current_request = prefix_input_request(views, parse_preprocessing_spec(request["authorization_preprocessing"]), profiles)
    descriptor = export_prepared_views(views, profiles, tmp_path / "entrypoint_inputs", current_request)
    holder = PrefixReplayInputHolder.from_prepared(descriptor, request_sha=semantic_sha256(current_request),
        prep_outcome=_outcome(descriptor, current_request), admission=Admission())
    borrowed = open_prepared_inputs(holder.descriptor, expected_request_sha=semantic_sha256(current_request),
                                    expected_capsule_sha=descriptor["sha256"])
    return borrowed.views.prefix_compiler, borrowed, holder


@pytest.mark.parametrize("origin", ["original", "prepared"])
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_direct_cached_compile_rejects_changed_owned_header(prepared, tmp_path, origin):
    compiler, borrowed, holder = _entrypoint_compiler(prepared, tmp_path, origin)
    try:
        prefix = compiler.compile(120)
        prefix.raw.samples.shape = (prefix.raw.samples.size,)
        with pytest.raises(ValueError, match="array header or immutable backing changed"):
            compiler.compile(120)
    finally:
        if borrowed is not None:
            borrowed.close()
            holder.close()


@pytest.mark.parametrize("origin", ["original", "prepared"])
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_before_group_rejects_changed_earlier_owner_header(prepared, tmp_path, origin):
    compiler, borrowed, holder = _entrypoint_compiler(prepared, tmp_path, origin)
    try:
        parent = compiler.compile(400)
        held = parent.complete_groups[-1]
        earlier = compiler.before_group(parent, held)
        assert 0 < earlier.budget < parent.budget
        earlier.raw.samples.shape = (earlier.raw.samples.size,)
        with pytest.raises(ValueError, match="array header or immutable backing changed"):
            compiler.before_group(parent, held)
    finally:
        if borrowed is not None:
            borrowed.close()
            holder.close()


@pytest.mark.parametrize("origin", ["original", "prepared"])
def test_owned_group_matching_reuses_admitted_complete_records(prepared, tmp_path, monkeypatch, origin):
    import transfer.ensemble_views as owner

    compiler, borrowed, holder = _entrypoint_compiler(prepared, tmp_path, origin)
    try:
        parent = compiler.compile(400)
        earlier = [(group, compiler.before_group(parent, group)) for group in parent.complete_groups]
        assert len(earlier) == 3

        def no_record(group):
            raise AssertionError("unchanged owned groups must reuse their admitted complete records")

        monkeypatch.setattr(owner.CalibrationSelection, "record", no_record)
        for _ in range(3):
            for group, prefix in earlier:
                assert compiler.before_group(parent, group) is prefix
    finally:
        if borrowed is not None:
            borrowed.close()
            holder.close()


@pytest.mark.parametrize("origin", ["original", "prepared"])
def test_foreign_group_matching_hashes_one_complete_record_per_call(prepared, tmp_path, monkeypatch, origin):
    import transfer.ensemble_views as owner

    original_record = owner.CalibrationSelection.record
    compiler, borrowed, holder = _entrypoint_compiler(prepared, tmp_path, origin)
    try:
        parent = compiler.compile(400)
        held = parent.complete_groups[-1]
        earlier = compiler.before_group(parent, held)
        foreign = replace(held, view=replace(held.view))
        calls = []

        def complete_record(group):
            calls.append(group)
            return original_record(group)

        monkeypatch.setattr(owner.CalibrationSelection, "record", complete_record)
        for _ in range(2):
            assert compiler.before_group(parent, foreign) is earlier
        assert len(calls) == 2
        assert all(group is foreign for group in calls)
    finally:
        if borrowed is not None:
            borrowed.close()
            holder.close()
