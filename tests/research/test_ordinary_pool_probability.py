"""Every ordinary mixture owns its exact roster, weighted OOF and epsilon."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from research.contracts import semantic_sha256
from tests.research.test_pool_probability_prefix import case as pool_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.pool_calibration import POOL_STRATEGY, TEMPERATURE_POOL_STRATEGY


def test_ordinary_distinct_weight_pools_and_fixed_control_reach_saved_parent(tmp_path, tmp_path_factory, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from transfer import ensemble_fit as fit_owner

    _, path, preprocessing, specs, declared, _ = pool_case.__wrapped__(tmp_path_factory)
    alternate = EnsembleProgram("reverse-weight-pool", declared.expert_ids, (.75, .25))
    control = EnsembleProgram.equal_weights("fixed-control", (specs[1].expert_id,))
    mixtures = (declared, alternate, control)
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "budgets": [200, 800], "seed_pairs": [[0, 21]], "experts": [spec.record() for spec in specs],
        "mixtures": [programme.record() for programme in mixtures], "baseline_mixture": declared.name,
        "source_checkpoints": {}, "source_seed_catalog": [],
        "calibration_strategies": ["fixed", POOL_STRATEGY, TEMPERATURE_POOL_STRATEGY],
        "temperature_expert_ids": [specs[0].expert_id],
        "authorization_preprocessing": preprocessing.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    actual_fit, fit_calls = fit_owner._fit_expert, []

    def captured_fit(*args, **kwargs):
        fitted = actual_fit(*args, **kwargs)
        fit_calls.append({"expert_id": args[0].expert_id, "labels": args[1].n_epochs})
        return fitted

    monkeypatch.setattr(fit_owner, "_fit_expert", captured_fit)
    with cpu_thread_budget(1):
        result = runner.run_study(plan_path, tmp_path / "run", execution="inline")
    assert result["status"] == "completed"
    participant = result["participants"][0]
    requested = len(participant["requested"])
    directory = tmp_path / "run/participant_000"
    ledger = runner._read_sealed_record(directory / "request_ledger.json")
    assert len(participant["predictions"]) == 2 * requested * (3 + 3 + 2)
    assert len(ledger["fit_requests"]) == 2 * (1 + 3 + 2)
    learned, packages, epsilon = {}, {}, {}
    for programme in mixtures:
        strategies = (POOL_STRATEGY,) if programme == control else (POOL_STRATEGY, TEMPERATURE_POOL_STRATEGY)
        for strategy in strategies:
            rows = [row for row in participant["predictions"] if row["mixture"] == programme.name
                    and row["calibration_strategy"] == strategy and row["K"] == 800]
            assert len(rows) == requested and all(row["saved_replay_exact"] for row in rows)
            state = rows[0]["pool_calibration"]
            assert state["program_sha256"] == programme.digest()
            assert state["requested_rows"] == 6 and state["candidate_count"] == 36
            assert all(row["pool_calibration"] == state for row in rows)
            package = rows[0]["probability_package"]
            assert all(row["probability_package"] == package for row in rows)
            manifest = json.loads((directory / package / "ensemble.json").read_bytes())["record"]
            assert manifest["program"] == programme.record()
            oof = manifest["pool_OOF_probability"]
            assert oof["OOF_program"] == programme.record()
            assert oof["captured_expert_ids"] == list(programme.expert_ids)
            assert all(set(row["expert_evidence"]) == set(programme.expert_ids) for row in oof["rows"])
            assert oof["complete_group_labels"] == 720 and oof["actual_K"] == 800
            key = (programme.name, strategy)
            learned[str(key)] = state
            packages[key] = package
            epsilon[key] = state["epsilon"]
            warmup = [row for row in participant["predictions"] if row["mixture"] == programme.name
                      and row["calibration_strategy"] == strategy and row["K"] == 200]
            assert len(warmup) == requested and all(row["pool_calibration"] is None for row in warmup)
            assert all(row["evidence"]["log_probability"] is None for row in warmup)
            assert all(row["temperature_calibration"]["failure"]["error_code"] == "insufficient_calibration_groups"
                       for row in warmup)
    assert len(set(packages.values())) == 5
    assert epsilon[(declared.name, POOL_STRATEGY)] != epsilon[(alternate.name, POOL_STRATEGY)]
    assert not any(row["mixture"] == control.name and row["calibration_strategy"] == TEMPERATURE_POOL_STRATEGY
                   for row in participant["predictions"])
    normalized = json.loads((tmp_path / "run/plan.json").read_bytes())
    source = json.loads((tmp_path / "run/executed_source.json").read_bytes())
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))

    def no_fit(*args, **kwargs):
        raise AssertionError("ordinary saved-parent admission must not refit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        admitted = runner._admit_candidate_participant(task, outcome, normalized, source)
    assert admitted["status"] == "completed" and len(admitted["predictions"]) == len(participant["predictions"])
    (tmp_path / "ordinary_pool_readout.json").write_text(json.dumps({"status": admitted["status"],
        "requested": requested, "predictions": len(admitted["predictions"]), "actual_software_expert_fit_calls": len(fit_calls),
        "actual_fit_inputs": fit_calls, "packages": {str(key): value for key, value in packages.items()},
        "pool_states": learned, "source_digest": semantic_sha256(source)}, indent=2), encoding="utf-8")
    receipt = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if runner._read_sealed_record(path)["mixture"] == declared.name
        and runner._read_sealed_record(path)["K"] == 800
        and runner._read_sealed_record(path)["calibration_strategy"] == POOL_STRATEGY)
    original = receipt.read_bytes()
    changed = runner._read_sealed_record(receipt)
    changed["pool_calibration"] = learned[str((alternate.name, POOL_STRATEGY))]
    runner._sealed_record(receipt, changed)
    with cpu_thread_budget(1), pytest.raises(ValueError, match="own admitted mixture"):
        runner._admit_candidate_participant(task, outcome, normalized, source)
    receipt.write_bytes(original)
