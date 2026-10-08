"""Actual source epoch selection/refit admission, with coherent shell tampering."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.contracts import semantic_sha256
from research.source_selection import fit_source_trajectory, validate_source_procedure
from tests.support.source_programs import source_early_stopping_config, source_early_stopping_data


@pytest.fixture(scope="module")
def source_programs():
    data, cfg = source_early_stopping_data(), source_early_stopping_config()
    common = dict(source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"], device=torch.device("cpu"), evidence_scope="development")
    _, selected = fit_source_trajectory(data, np.arange(32), config=cfg, **common)
    _, fixed = fit_source_trajectory(data, np.arange(32), config=replace(cfg,
        global_early_stopping=None, fixed_updates=3, sampling="domain_participant"), **common)
    return selected[2], fixed[3]


def admit(payload, kind):
    return validate_source_procedure(payload, source_seed=7, outer_target_study="outer",
        evidence_scope="software_correctness", kind=kind)


def bind_shell(payload):
    if "source_epoch_selection" in payload:
        payload["training_contract"]["validation"]["early_stopping"] = copy.deepcopy(payload["source_epoch_selection"])
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    return payload


def test_actual_es_producer_and_fixed_balanced_updates_have_distinct_admission(source_programs):
    selected, fixed = source_programs
    receipt = admit(selected, "selection_refit")
    assert receipt["selected_epochs"] == 1 and receipt["actual_inner_updates"] == 3
    assert receipt["actual_refit_updates"] == 2
    for scope in ("public_development", "public_calibration_development"):
        shared = validate_source_procedure(selected, source_seed=7, outer_target_study="outer",
            evidence_scope=scope, kind="selection_refit")
        assert shared["source_evidence_scope"] == "development"
        assert shared["consumer_evidence_scope"] == scope
    assert admit(fixed, "fixed_recipe")["actual_refit_updates"] == 3
    with pytest.raises(ValueError, match="cannot hide"):
        admit(selected, "fixed_recipe")
    changed = copy.deepcopy(fixed)
    changed["training_contract"]["optimizer"]["config"]["seed"] = 8
    with pytest.raises(ValueError, match="same-seed"):
        admit(bind_shell(changed), "fixed_recipe")


@pytest.mark.parametrize("change", ["chosen_epoch", "validation_partition", "source_snapshot", "refit_budget"])
def test_source_procedure_rejects_rehashed_unrelated_selection_or_refit(source_programs, change):
    payload = copy.deepcopy(source_programs[0])
    selection = payload["source_epoch_selection"]
    if change == "chosen_epoch":
        selection["selected_epochs"] = selection["inner_history"]["early_stopping"]["selected_epochs"] = 2
    elif change == "validation_partition":
        selection["validation_rows"][0] = selection["training_rows"][0]
    elif change == "source_snapshot":
        selection["source_snapshot_sha256"] = "c" * 64
    else:
        payload["source_fixed_updates"] = 4
        payload["training_contract"]["optimizer"]["actual_updates"] = 4
        payload["training_contract"]["optimizer"]["config"]["fixed_updates"] = 4
        selection["refit_updates"] = 4
    with pytest.raises(ValueError):
        admit(bind_shell(payload), "selection_refit")
