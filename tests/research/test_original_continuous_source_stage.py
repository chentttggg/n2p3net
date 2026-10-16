"""Plan counterexamples only; the single real genesis is an archived programme."""
import copy

import pytest

from research.original_continuous_source_stage import PLAN_SCHEMA, validate_source_stage_plan


def plan():
    from data.contract import DEFAULT_P300_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract

    def original(name, digest, budget, generation_seed):
        return {"path": f"D:/software-fixture/{name}", "source_relative_path": name,
            "sha256": digest*64, "bytes": 1, "budget": budget, "input_generation_seed": generation_seed}

    return {"schema": PLAN_SCHEMA, "source_id": "original-source-stage-test",
        "source_evidence_scope": "software_correctness", "actual_source_seed": None, "native_threads": 1,
        "source_inputs": [original("A01.mat", "a", 120, 211)],
        "target_input": original("A02.mat", "b", 0, 212),
        "target_identity": {"participant_key": "n2p3-origin://BNCI2014-008/participant/A02",
            "source_dataset_id": "BNCI2014-008", "source_relative_path": "A02.mat", "source_sha256": "b"*64},
        "geometry": {"channels": ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
            "sfreq": 256., "unit": "V", "reference": "right earlobe",
            "operator_identity": "native_unfiltered_volts/1", "observation_rank": 8},
        "authorization_preprocessing": preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT).record(),
        "fit": {"basis": [[0.], [1.], [.5], [0.]], "ridge": 1., "noise_order": 0,
            "conditioning_samples": 64, "degrees_of_freedom": 4., "relative_noise_shape": .7,
            "response_seconds": 4/256, "pre_roll_seconds": 4/256},
        "exclusion_scope": "target_participant", "input_manifest": {"path": "D:/manifest.json", "bytes": 1, "sha256": "c"*64},
        "minimum_available_memory_bytes": 1024**3, "handshake_timeout_seconds": 120.}


def test_input_generation_seed_does_not_become_scientific_source_seed():
    declared = plan()
    assert validate_source_stage_plan(declared)["actual_source_seed"] is None
    assert declared["source_inputs"][0]["input_generation_seed"] == 211


@pytest.mark.parametrize("key,value", [
    ("source_evidence_scope", "public_development"), ("actual_source_seed", 211),
    ("native_threads", True), ("native_threads", 4), ("exclusion_scope", "whole_target_dataset"),
])
def test_scientific_scope_seed_and_native_contract_are_explicit(key, value):
    declared = plan()
    declared[key] = value
    with pytest.raises(ValueError):
        validate_source_stage_plan(declared)


def test_renamed_same_target_bytes_cannot_enter_source_cohort():
    declared = plan()
    declared["source_inputs"][0]["sha256"] = declared["target_input"]["sha256"]
    with pytest.raises(ValueError, match="exclude"):
        validate_source_stage_plan(declared)


def test_source_manifest_must_not_repeat_original_bytes_under_another_name():
    declared = plan()
    duplicate = copy.deepcopy(declared["source_inputs"][0])
    duplicate["source_relative_path"] = "A03.mat"
    declared["source_inputs"].append(duplicate)
    with pytest.raises(ValueError, match="exclude"):
        validate_source_stage_plan(declared)


def test_target_identity_cannot_be_rebound_to_another_declared_mat():
    declared = plan()
    declared["target_identity"]["source_relative_path"] = "A03.mat"
    with pytest.raises(ValueError, match="target exclusion"):
        validate_source_stage_plan(declared)


@pytest.mark.parametrize("wait_state,expected", [(0, 259), (258, None)])
def test_windows_exit_259_requires_the_real_handle_signal(wait_state, expected):
    from types import SimpleNamespace

    from research.original_continuous_source_stage import _ScientificExit

    def exit_code(_handle, pointer):
        pointer._obj.value = 259
        return 1

    observer = object.__new__(_ScientificExit)
    observer.handle = 17
    observer.kernel = SimpleNamespace(WaitForSingleObject=lambda *args: wait_state,
        GetExitCodeProcess=exit_code)
    assert observer.poll() == expected


def test_uncontained_bootstrap_freezes_and_discovers_shim_child_before_parent_kill(monkeypatch):
    from types import SimpleNamespace

    import psutil

    import research.original_continuous_source_stage as stage

    operations = []
    processes = {}

    def process(pid):
        def suspend():
            operations.append(("suspend", pid))
        def kill():
            operations.append(("kill", pid))
            processes.pop(pid)
        return SimpleNamespace(pid=pid, suspend=suspend, kill=kill)

    processes[10], processes[11] = process(10), process(11)
    owned = {10: {"pid": 10, "create_time": 1.}}
    monkeypatch.setattr(stage, "_same_process", lambda identity: processes.get(identity["pid"]))
    monkeypatch.setattr(psutil, "wait_procs", lambda *args, **kwargs: ([], []))

    def remember():
        assert ("suspend", 10) in operations
        operations.append(("discover", 11))
        owned[11] = {"pid": 11, "create_time": 2.}

    stage._retire_uncontained(owned, remember, 10)
    assert set(owned) == {10, 11} and not processes
    assert operations.index(("suspend", 10)) < operations.index(("discover", 11))
    assert operations.index(("suspend", 11)) < operations.index(("kill", 10))
    assert operations.index(("kill", 11)) < operations.index(("kill", 10))
