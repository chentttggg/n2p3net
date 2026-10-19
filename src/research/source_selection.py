"""Source-only training-chain folds, dose manifests, and fixed-cohort selection."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from math import ceil
from pathlib import Path

import numpy as np
import torch

from baselines.deep import DeepConfig, resolve_source_risk_weights
from baselines.n2p3net import N2P3NetBaseline
from baselines.validation import domain_stratified_group_validation_split
from data.domain import source_domain_ids
from data.epochs import (
    EpochDataset,
    _epoch_row_identity_digest,
    materialize_dataset_lineage,
    preprocessing_execution_binding,
)
from data.identity import DatasetIdentityTable, training_identity_ledger_from_rows
from data.study import checkpoint_study_membership, dataset_study_membership, study_ids_for_identity
from models.n2p3net import (
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from research.contracts import TrainingRunContract, semantic_sha256
from research.domain_folds import domain_lodo_folds
from transfer.checkpoint import CHECKPOINT_SCHEMA, checkpoint_state_digest

SOURCE_UPDATE_CANDIDATES = (500, 1000, 2000, 4000, 8000)
SOURCE_CHAIN_LAYOUTS = {"six": ("Fz", "P3", "Pz", "P4", "PO8", "Oz"),
                        "common_three": ("P3", "Pz", "P4")}


def _independent_source_config_parameters(value):
    if not isinstance(value, Mapping) or set(value) != {"pos_weight", "val_groups_min", "val_groups_max"}:
        raise ValueError("independent source_config must declare pos_weight and validation group bounds only")
    weight, minimum, maximum = (value[key] for key in ("pos_weight", "val_groups_min", "val_groups_max"))
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool) or not 0 < weight < float("inf")
            or type(minimum) is not int or minimum < 1
            or maximum is not None and (type(maximum) is not int or maximum < minimum)):
        raise ValueError("independent source_config requires positive class weight and valid group bounds")
    return {"pos_weight": float(weight), "val_groups_min": minimum, "val_groups_max": maximum}


def resolve_independent_source_config(declared, identity, study_membership):
    """Bind the source-only risk and fraction cap to actual participant authority."""
    declared = _independent_source_config_parameters(declared)
    participants = {}
    for record in identity.records:
        studies = study_ids_for_identity(DatasetIdentityTable((record,)), study_membership)
        if len(studies) != 1:
            raise ValueError("independent source participant must belong to one actual source study")
        participants.setdefault(studies[0], set()).add(record.authority_key())
    counts = {study: len(people) for study, people in sorted(participants.items())}
    maximum = max(counts.values()) if declared["val_groups_max"] is None else declared["val_groups_max"]
    return {"declared": declared, "resolved": {**declared, "val_groups_max": maximum},
            "source_participant_counts": counts,
            "val_groups_max_basis": ("maximum_actual_source_participants_per_study"
                                     if declared["val_groups_max"] is None else "explicit_source_cap")}


def resolve_source_program(plan, stages):
    """Separate the independent per-seed procedure from historical programs."""
    from data.epochs import parse_preprocessing_spec
    from research.source_catalog import paired_pipeline_seeds
    from train.early_stopping import GlobalEarlyStoppingPolicy

    evidence_scope = plan.get("evidence_scope", "development")
    if evidence_scope not in {"development", "software_correctness"}:
        raise ValueError("source chain supports only development or software_correctness evidence")
    if any(key.startswith("correctness_") for key in plan) and evidence_scope != "software_correctness":
        raise ValueError("correctness budgets cannot be used under public development evidence")

    expected = "unified" if "unified_source" in stages else "declared_legacy_stages"
    program = plan.get("source_program", expected)
    if program == expected:
        return {"name": program}
    if program != "independent_selection_refit":
        raise ValueError("source_program differs from the requested source stages")
    outers = plan["outer_folds"]
    if not outers or len({outer.get("target_study") for outer in outers}) != len(outers):
        raise ValueError("independent source requires nonempty unique outer target study folds")
    if (set(stages) != {"P2_source"} or plan.get("source_layouts") != ["six"]
            or plan.get("source_objectives") != {"ce": {}}
            or plan.get("paired_source_target_seeds") is not True
            or not isinstance(plan.get("source_preprocessing"), dict)
            or not isinstance(plan.get("global_early_stopping"), dict)
            or any(plan.get(field) is not None for field in (
                "unified_training_budget", "source_prior_selection", "source_budget"))
            or any(outer.get("source_prior_selection") is not None for outer in plan["outer_folds"])):
        raise ValueError("independent source requires plain CE, six channels, explicit preprocessing, paired seeds and its own patience/refit program")
    spec = parse_preprocessing_spec(plan["source_preprocessing"])
    if spec.filter_phase != "zero":
        raise ValueError("independent source requires an explicit finite zero-phase profile")
    if not {"patience", "min_delta", "source_max_epochs", "source_validation_fraction"} <= set(plan["global_early_stopping"]):
        raise ValueError("independent source requires its four consumed stopping parameters explicitly")
    source_config = _independent_source_config_parameters(plan.get("source_config"))
    policy = GlobalEarlyStoppingPolicy(**plan["global_early_stopping"])
    source_seeds, target_seeds = plan.get("source_seeds", ()), plan.get("target_seeds", ())
    pairs = paired_pipeline_seeds(plan.get("seed_pairs", list(zip(source_seeds, target_seeds, strict=True))))
    if (set(source_seeds) != {pair[0] for pair in pairs}
            or set(target_seeds) != {pair[1] for pair in pairs}):
        raise ValueError("independent source pairs must cover the declared source and target seeds exactly")
    for outer in plan["outer_folds"]:
        if (outer.get("role", "evaluation") != "evaluation"
                or outer.get("target_study") not in outer["excluded_source_datasets"]
                or "source_studies" not in outer):
            raise ValueError("independent source folds require explicit full study membership and whole-target-study exclusion")
    return {"name": program, "preprocessing": spec.record(), "stopping": policy.record(),
            "source_config": source_config,
            "seed_pairs": [list(pair) for pair in pairs], "procedure": "selection_refit"}


def validate_independent_source_input(dataset, preprocessing):
    """Admit the declared operator and its actual measured-Cz execution once."""
    from data.channel import canonical_channel_name
    from data.epochs import parse_preprocessing_spec

    declared = parse_preprocessing_spec(preprocessing)
    if (declared.filter_phase != "zero" or dataset.preprocessing != declared
            or tuple(map(canonical_channel_name, dataset.channel_names)) != tuple(map(canonical_channel_name, SOURCE_CHAIN_LAYOUTS["six"]))
            or dataset.provenance.get("source_reference") != "measured_channel:CZ"):
        raise ValueError("independent source cache differs from its declared finite profile or measured-Cz six-channel input")
    binding = preprocessing_execution_binding(dataset.provenance.get("preprocessing_execution"), declared)
    identity = binding.get("execution_identity")
    if identity is None:
        raise ValueError("independent source requires complete actual preprocessing execution evidence")
    return {"preprocessing": declared.record(), "execution_identity": identity,
            "execution_provenance_sha256": binding["execution_provenance_sha256"],
            "channel_names": list(dataset.channel_names), "n_times": dataset.n_times,
            "source_reference": dataset.provenance["source_reference"]}


def source_identity_axes(dataset: EpochDataset) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve actual person/library identities and reject relabelled duplicate domains."""
    if dataset.identity_table is None:
        raise ValueError("source selection requires a production identity table")
    table = dataset.identity_table
    table.authority_keys()
    domains = source_domain_ids(dataset)
    study_membership = dataset_study_membership(dataset)
    local_records = {record.local_subject_id: record for record in table.records}
    keys = {local: record.authority_key() for local, record in local_records.items()}
    local_ids, first_rows, inverse = np.unique(np.asarray(dataset.subject_ids).astype(str, copy=False),
                                               return_index=True, return_inverse=True)
    domains_by_local = domains[first_rows]
    if not np.array_equal(domains, domains_by_local[inverse]):
        raise ValueError("one actual participant appears in multiple source domains")
    people = np.asarray([keys[str(local)] for local in local_ids])[inverse]
    libraries: dict[str, set[str]] = {}
    library_domains: dict[str, str] = {}
    for local, domain in zip(local_ids, domains_by_local, strict=True):
        record = local_records[str(local)]
        for library in study_ids_for_identity(DatasetIdentityTable((record,)), study_membership):
            if library_domains.setdefault(library, domain) != domain:
                raise ValueError("one actual source library appears under multiple domain names")
            libraries.setdefault(domain, set()).add(library)
    if set(libraries) != set(domains) or any(len(values) != 1 for values in libraries.values()):
        raise ValueError("each source domain must identify exactly one actual library")
    return domains, people, {key: next(iter(value)) for key, value in libraries.items()}


def source_fit_ledger(
    dataset: EpochDataset, rows: np.ndarray, *, excluded_source_datasets: Sequence[str],
    forbidden_participant_keys: Sequence[str] = (),
) -> dict:
    """Prove library/person exclusions and observed coverage on actual fit rows."""
    domains, people, libraries = source_identity_axes(dataset)
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
        raise ValueError("source fit rows must be a nonempty integer vector")
    if np.any(rows < 0) or np.any(rows >= dataset.n_epochs) or len(np.unique(rows)) != len(rows):
        raise ValueError("source fit rows must be unique in-range observations")
    retained_domains = sorted(set(domains[rows]))
    forbidden = set(excluded_source_datasets)
    if any(libraries[domain] in forbidden for domain in retained_domains):
        raise ValueError("source fit includes an excluded outer source library")
    if set(people[rows]) & set(forbidden_participant_keys):
        raise ValueError("source fit includes a held-out participant authority key")
    mask = (np.broadcast_to(dataset.channel_mask, (len(rows), dataset.n_channels))
            if dataset.trial_channel_mask is None else dataset.trial_channel_mask[rows])
    if not np.all(mask.any(axis=0)):
        raise ValueError("source fit has a channel with no real observations after exclusion")
    fit_mask = np.zeros(dataset.n_epochs, bool)
    fit_mask[rows] = True
    identity = training_identity_ledger_from_rows(dataset.identity_table, dataset.subject_ids, fit_mask)
    input_digest = hashlib.sha256()
    input_digest.update(str((len(rows), *dataset.X.shape[1:])).encode())
    input_digest.update(str(dataset.X.dtype).encode())
    for start in range(0, len(rows), 512):
        input_digest.update(np.ascontiguousarray(dataset.X[rows[start:start + 512]]).tobytes())
    input_digest.update(np.ascontiguousarray(np.asarray(dataset.y)[rows], dtype="<i8").tobytes())
    input_digest.update(np.ascontiguousarray(mask).tobytes())
    return {
        "fit_rows": rows.astype(int).tolist(), "fit_rows_sha256": semantic_sha256(rows.tolist()),
        # source_identity_axes admitted this dataset earlier in this same
        # read-only call; the ordered digest retains its own mapping check.
        "cache_row_identity_sha256": _epoch_row_identity_digest(dataset),
        "fit_input_label_mask_sha256": input_digest.hexdigest(),
        "input_channel_names": list(dataset.channel_names),
        "input_lineage_digest": materialize_dataset_lineage(dataset).payload()["lineage_digest"],
        "training_participant_keys": sorted(set(people[rows])),
        "training_identity_ledger": identity.payload(),
        "training_identity_ledger_digest": identity.digest(),
        "source_domains": retained_domains,
        "actual_source_libraries": [libraries[domain] for domain in retained_domains],
        "excluded_source_datasets": sorted(forbidden),
        "channels": [{"channel": name, "observed_rows": int(mask[:, index].sum()),
                      "observed_domains": sorted(set(domains[rows][mask[:, index]])),
                      "observed_participants": sorted(set(people[rows][mask[:, index]]))}
                     for index, name in enumerate(dataset.channel_names)],
    }


def source_inner_folds(dataset: EpochDataset, *, seed: int, excluded_source_datasets: Sequence[str]) -> list[dict]:
    """Build the shared P00/P01 trajectory and all shared P10/P11 trajectories."""
    domains, people, _ = source_identity_axes(dataset)
    if len(np.unique(domains)) < 3:
        raise ValueError("E2 requires at least three outer-source libraries")
    split = domain_stratified_group_validation_split(
        people, domains, fraction=0.08, min_groups=2, max_groups=12, seed=seed,
    )
    raw = [("participant", "participant", np.flatnonzero(split.train_mask),
            np.flatnonzero(split.validation_mask))]
    raw.extend(("library", fold.held_out_domain, np.flatnonzero(fold.train),
                np.flatnonzero(fold.test)) for fold in domain_lodo_folds(people, domains))
    output = []
    for axis, name, train, validation in raw:
        ledger = source_fit_ledger(dataset, train, excluded_source_datasets=excluded_source_datasets,
                                   forbidden_participant_keys=np.unique(people[validation]).tolist())
        output.append({"axis": axis, "fold_id": name, "train_rows": train.tolist(),
                       "validation_rows": validation.tolist(),
                       "validation_participant_keys": sorted(set(people[validation])),
                       "ledger": ledger})
    return output


def source_prior_development_fold(dataset, *, seed, excluded_source_datasets, fraction=.08,
                                  people_per_domain=None, eligible_people=None, development_domains=None):
    """One declared participant split for a common source-prior strength across factors."""
    domains, people, _ = source_identity_axes(dataset)
    from research.source_catalog import study_names

    declared = (tuple(sorted(set(domains))) if development_domains is None else
                study_names(development_domains, field="development_domains"))
    if not set(declared) <= set(domains):
        raise ValueError("development_domains must be actual source domains")
    if development_domains is not None and people_per_domain is None:
        raise ValueError("explicit development_domains requires people_per_domain")
    if people_per_domain is None:
        split = domain_stratified_group_validation_split(people, domains, fraction=fraction,
                                                         min_groups=1, max_groups=12, seed=seed)
        training, validation = np.flatnonzero(split.train_mask), np.flatnonzero(split.validation_mask)
    else:
        if type(people_per_domain) is not int or people_per_domain < 1:
            raise ValueError("development people per domain must be positive")
        eligible = set(people if eligible_people is None else eligible_people)
        held = []
        for domain in sorted(declared):
            population = sorted(set(people[domains == domain]))
            candidates = [person for person in population if person in eligible]
            if len(candidates) < people_per_domain or len(population) - people_per_domain < 4:
                raise ValueError(f"source-prior development lacks eligible people or source validation support: {domain}")
            rng = np.random.default_rng(int(semantic_sha256({"seed": seed, "domain": domain})[:16], 16))
            held.extend(rng.permutation(candidates)[:people_per_domain])
        mask = np.isin(people, held)
        training, validation = np.flatnonzero(~mask), np.flatnonzero(mask)
    ledger = source_fit_ledger(dataset, training, excluded_source_datasets=excluded_source_datasets,
                               forbidden_participant_keys=np.unique(people[validation]).tolist())
    return {"train_rows": training.tolist(), "validation_rows": validation.tolist(),
            "validation_participant_keys": sorted(set(people[validation])), "fit_ledger": ledger,
            "split_seed": seed, "validation_fraction": fraction if people_per_domain is None else None,
            "people_per_domain": people_per_domain,
            "development_domains": sorted(declared),
            "participant_domains": {person: str(domains[np.flatnonzero(people == person)[0]])
                                    for person in sorted(set(people[validation]))}}


def select_source_prior_weight(outcomes, *, requested_cells, candidates, participant_domains=None):
    """Keep every requested person/seed; select one finite lambda by person-equal AUC."""
    candidates = tuple(candidates)
    if (not candidates or len(set(candidates)) != len(candidates)
            or any(isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in candidates)):
        raise ValueError("source-prior candidates must be unique nonnegative finite strengths")
    requested = set(tuple(cell) for cell in requested_cells)
    if not requested:
        return {"status": "pending", "reason": "no_legal_source_inner_K200", "selected_weight": None, "curves": []}
    curves = []
    for weight in candidates:
        matching = [row for row in outcomes if row["source_prior_weight"] == weight]
        cells = {(row["participant_key"], row["source_seed"], row["target_seed"]): row for row in matching}
        if len(cells) != len(matching) or set(cells) - requested:
            raise ValueError("source-prior results contain duplicate or undeclared participant/seed cells")
        failed = sorted(cell for cell in requested if cell not in cells or cells[cell].get("status") != "completed"
                        or not isinstance(cells[cell].get("auc"), (int, float))
                        or not np.isfinite(cells[cell]["auc"]) or not 0 <= cells[cell]["auc"] <= 1)
        people = sorted({cell[0] for cell in requested})
        score = None if failed else float(np.mean([
            np.mean([row["auc"] for cell, row in cells.items() if cell[0] == person]) for person in people]))
        if not failed and participant_domains is not None:
            if set(participant_domains) != set(people):
                raise ValueError("source-prior domain weights must cover exactly the requested people")
            score = float(np.mean([np.mean([
                np.mean([row["auc"] for cell, row in cells.items() if cell[0] == person])
                for person in people if participant_domains[person] == domain])
                for domain in sorted(set(participant_domains.values()))]))
        curves.append({"source_prior_weight": weight, "participant_macro_auc": score,
                       "failed_cells": [list(cell) for cell in failed]})
    valid = [row for row in curves if row["participant_macro_auc"] is not None]
    best = max((row["participant_macro_auc"] for row in valid), default=None)
    chosen = None if best is None else min(row["source_prior_weight"] for row in valid
                                          if row["participant_macro_auc"] >= best - 1e-6)
    return {"status": "selected" if chosen is not None else "pending", "selected_weight": chosen,
            "curves": curves, "requested_cells": [list(cell) for cell in sorted(requested)],
            "score": ("seed_mean_then_domain_participant_equal_auc" if participant_domains is not None
                      else "seed_mean_then_participant_equal_auc"), "failure_policy": "all_requested_cells_required",
            "tie_policy": "within_1e-6_choose_smaller_weight", "outer_target_used": False}


def source_fixed_dose(
    dataset: EpochDataset, *, domain_ids: Sequence[str], trials_per_participant: int,
    seed: int, chronological_rows: Mapping[str, Sequence[int]], participants_per_domain: int = 8,
) -> dict:
    """Select people once, then one unique row per equal chronological interval."""
    domains, people, _ = source_identity_axes(dataset)
    if trials_per_participant < 1 or participants_per_domain < 1:
        raise ValueError("fixed source dose counts must be positive")
    if len(set(domain_ids)) != len(domain_ids) or not set(domain_ids) <= set(domains):
        raise ValueError("dose domains must be unique actual source domains")
    selected, requests, failures, preflight_exclusions = [], [], [], []
    for domain in sorted(domain_ids):
        eligible = []
        for person in sorted(set(people[domains == domain])):
            if set(np.asarray(dataset.y)[people == person].tolist()) == {0, 1}:
                eligible.append(person)
            else:
                preflight_exclusions.append({"domain": domain, "participant_key": person,
                                             "reason": "participant_missing_class_before_sampling"})
        if len(eligible) < participants_per_domain:
            failures.append({"domain": domain, "reason": "too_few_participants"})
            continue
        # Stable per-domain seed keeps the same eight people in single/multi-source arms.
        domain_seed = int(semantic_sha256({"seed": seed, "domain": domain})[:16], 16)
        rng = np.random.default_rng(domain_seed)
        chosen = rng.choice(eligible, size=participants_per_domain, replace=False).tolist()
        for person in chosen:
            rows = np.asarray(chronological_rows.get(person, ()), dtype=np.int64)
            actual = np.flatnonzero(people == person)
            request = {"domain": domain, "participant_key": person,
                       "requested_trials": trials_per_participant, "available_trials": len(actual)}
            requests.append(request)
            if not np.array_equal(np.sort(rows), actual):
                failures.append({**request, "reason": "missing_or_invalid_chronological_rows"})
                continue
            if len(rows) < trials_per_participant:
                failures.append({**request, "reason": "insufficient_unique_rows"})
                continue
            boundaries = np.linspace(0, len(rows), trials_per_participant + 1, dtype=int)
            chosen_rows = rows[[rng.integers(left, right) for left, right in
                                zip(boundaries[:-1], boundaries[1:], strict=True)]]
            if set(np.asarray(dataset.y)[chosen_rows].tolist()) != {0, 1}:
                failures.append({**request, "reason": "selected_rows_missing_class"})
                continue
            selected.extend(chosen_rows.tolist())
            request["selected_rows"] = chosen_rows.tolist()
    return {"status": "invalid_design" if failures else "ready", "requests": requests,
            "preflight_exclusions": preflight_exclusions,
            "failures": failures, "selected_rows": selected, "unique_trials": len(set(selected)),
            "seed": seed, "trials_per_participant": trials_per_participant,
            "participants_per_domain": participants_per_domain,
            "selection": "one_uniform_row_per_equal_chronological_interval_no_label_resampling"}


def select_source_updates(
    outcomes: Sequence[Mapping], *, requested_cells: Sequence[tuple[str, str, int]],
    candidates: Sequence[int] = SOURCE_UPDATE_CANDIDATES,
) -> dict:
    """Select CE/calibration/zero budgets on one frozen cohort, retaining failures."""
    requested = set(requested_cells)
    if len(requested) != len(requested_cells) or not requested:
        raise ValueError("source selection cohort must have unique requested domain/person/K cells")
    lookup = {}
    for row in outcomes:
        key = (int(row["updates"]), str(row["domain"]), str(row["participant_key"]), int(row["actual_K"]))
        if key in lookup or key[0] not in candidates or key[1:] not in requested:
            raise ValueError("duplicate or unrequested source candidate outcome")
        lookup[key] = row

    def score(update, field, positive):
        cohort = [key for key in requested if (key[2] > 0) == positive]
        people_scores = {}
        failures = []
        for domain, person, budget in sorted(cohort):
            row = lookup.get((update, domain, person, budget))
            value = None if row is None else row.get(field)
            if row is None or row.get("status") != "completed" or value is None or not np.isfinite(value):
                failures.append({"domain": domain, "participant_key": person, "actual_K": budget})
            else:
                people_scores.setdefault((domain, person), []).append(float(value))
        if failures or not cohort:
            return None, failures
        domain_scores = {}
        for (domain, _), values in people_scores.items():
            domain_scores.setdefault(domain, []).append(float(np.mean(values)))
        return float(np.mean([np.mean(values) for values in domain_scores.values()])), []

    curves = []
    for update in candidates:
        ce, ce_failures = score(update, "source_ce", False)
        calibrated, calibration_failures = score(update, "auc", True)
        zero, zero_failures = score(update, "auc", False)
        curves.append({"updates": int(update), "source_ce": ce,
                       "calibration_budget_policy_score": calibrated, "zero_auc": zero,
                       "ce_failures": ce_failures, "calibration_failures": calibration_failures,
                       "zero_failures": zero_failures})
    chosen = {}
    for name, field, direction in (("U_ce", "source_ce", 1),
                                  ("U_cal", "calibration_budget_policy_score", -1),
                                  ("U_zero", "zero_auc", -1)):
        valid = [row for row in curves if row[field] is not None]
        if not valid:
            chosen[name] = None
            continue
        optimum = min(direction * row[field] for row in valid)
        chosen[name] = min(row["updates"] for row in valid if direction * row[field] <= optimum + 1e-6)
    return {**chosen, "curves": curves, "requested_cells": [list(cell) for cell in sorted(requested)],
            "failure_policy": "any_requested_cell_failure_invalidates_candidate_score",
            "estimand": "actual_positive_K_within_person_then_person_within_library_then_library_equal"}


def source_architecture_record(dataset: EpochDataset, architecture: N2P3ArchitectureConfig) -> dict:
    """Resolve the source model identity without constructing weights or fitting data."""
    return N2P3Net.default_architecture_record(
        pooling_mode=DEFAULT_N2P3_POOLING_MODE,
        tmin_s=dataset.preprocessing.tmin_ms / 1000,
        sfreq=dataset.preprocessing.sfreq, n_times=dataset.n_times, architecture=architecture)


def validate_source_procedure(payload, *, source_seed, outer_target_study, evidence_scope, kind,
                              expected_source_config=None):
    """Admit the procedure that produced these weights, not a catalog status flag.

    This owner supports its fixed-update and participant-disjoint epoch/refit
    programs. Reference-exposure and external hyperparameter selectors require
    their own complete per-seed producer binding before full-pipeline admission.
    """
    from train.early_stopping import GlobalEarlyStoppingPolicy, PatienceStopper
    from transfer.checkpoint import (
        checkpoint_training_contract,
        checkpoint_training_identity_ledger,
    )

    contract = checkpoint_training_contract(payload)
    identity = checkpoint_training_identity_ledger(payload)
    studies = study_ids_for_identity(identity, checkpoint_study_membership(payload))
    if outer_target_study in studies:
        raise ValueError("source catalog checkpoint must exclude the complete outer target study")
    if contract.seed != source_seed:
        raise ValueError("source procedure seed differs from actual checkpoint seed")
    config = contract.optimizer.get("config", {})
    program_config = contract.objective.get("source_program_config")
    declared_config = expected_source_config
    if declared_config is None and program_config is not None:
        declared_config = program_config["declared"]
    expected = None
    if declared_config is not None:
        expected = resolve_independent_source_config(declared_config, identity, checkpoint_study_membership(payload))
        if ({key: config.get(key) for key in expected["resolved"]} != expected["resolved"]
                or payload.get("source_calibration", {}).get("pos_weight") != expected["resolved"]["pos_weight"]
                or program_config is not None and program_config != expected):
            raise ValueError("source actual producer differs from the declared independent source_config")
    selection = payload.get("source_epoch_selection")
    known_selection = (selection is not None or payload.get("global_early_stopping") is not None
        or payload.get("source_budget") is not None or config.get("source_budget") is not None
        or config.get("global_early_stopping") is not None
        or contract.validation.get("early_stopping") is not None
        or contract.validation.get("strategy") == "source_internal_early_stopping_then_full_refit")
    external_selection = any(payload.get(key) is not None for key in (
        "source_prior_selection", "source_residual_selection", "source_prior_weight_selection"))
    fixture = contract.optimizer.get("kind") == "synthetic_test_fixture"
    source_scope = "software_correctness" if fixture else payload.get("evidence_scope")
    if evidence_scope not in {"software_correctness", "public_development", "public_calibration_development"}:
        raise ValueError("source procedure consumer evidence scope is unsupported")
    if fixture and evidence_scope != "software_correctness":
        raise ValueError("software source fixtures cannot establish public full-source pipeline replication")
    if kind not in {"fixed_recipe", "selection_refit"}:
        raise ValueError("source procedure must declare fixed recipe or actual selection/refit")
    if external_selection or (kind == "fixed_recipe" and known_selection):
        raise ValueError("fixed source recipe cannot hide a selection procedure")
    if fixture and kind == "fixed_recipe":
        return {"kind": kind, "selection": "not_applicable", "source_seed": source_seed,
                "source_evidence_scope": source_scope, "consumer_evidence_scope": evidence_scope,
                "source_studies": sorted(studies)}
    if source_scope == "software_correctness" and evidence_scope != "software_correctness":
        raise ValueError("software source fixtures cannot establish public full-source pipeline replication")
    if (source_scope not in {"development", "public_development", "public_calibration_development", "software_correctness"}
            or config.get("seed") != source_seed
            or contract.validation.get("evidence_scope") != source_scope
            or payload.get("source_cache_sha256") != contract.source_cache_sha256
            or payload.get("source_snapshot_sha256") != contract.source_snapshot_sha256
            or payload.get("source_full_refit") is not True):
        raise ValueError("source procedure requires actual same-seed producer evidence and executed input/source binding")
    ledger = payload.get("source_fit_ledger")
    if not isinstance(ledger, dict) or contract.validation.get("fit_ledger") != ledger:
        raise ValueError("source full refit ledger differs from its training contract")
    full_rows = np.asarray(ledger["fit_rows"])
    if (full_rows.ndim != 1 or full_rows.dtype.kind not in "iu" or not len(full_rows)
            or np.any(full_rows < 0) or len(np.unique(full_rows)) != len(full_rows)
            or ledger["fit_rows_sha256"] != semantic_sha256(full_rows.tolist())
            or ledger["training_identity_ledger"] != identity.payload()
            or ledger["training_identity_ledger_digest"] != identity.digest()
            or tuple(ledger["training_participant_keys"]) != contract.training_participant_keys
            or outer_target_study not in ledger["excluded_source_datasets"]
            or set(ledger["actual_source_libraries"]) != set(studies)):
        raise ValueError("source refit rows/identities/exclusion differ from actual fit authority")
    updates = contract.optimizer.get("actual_updates")
    if (type(updates) is not int or updates < 1 or payload.get("source_fixed_updates") != updates
            or config.get("fixed_updates") != updates
            or type(contract.optimizer.get("trial_draws")) is not int or contract.optimizer["trial_draws"] < updates
            or not contract.optimizer.get("batch_sequence_sha256")):
        raise ValueError("source refit budget differs from actual optimizer exposure")
    if kind == "fixed_recipe":
        if contract.validation.get("strategy") != "fixed_source_updates":
            raise ValueError("fixed source recipe requires a recorded no-selection producer program")
        return {"kind": kind, "selection": "not_applicable", "source_seed": source_seed,
                "actual_refit_updates": updates, "fit_ledger_digest": semantic_sha256(ledger),
                "source_evidence_scope": source_scope, "consumer_evidence_scope": evidence_scope,
                "source_studies": sorted(studies)}
    if (selection is None or contract.validation.get("strategy") != "source_internal_early_stopping_then_full_refit"
            or contract.validation.get("early_stopping") != selection
            or payload.get("source_budget") is not None or config.get("source_budget") is not None
            or selection.get("source_seed") != source_seed
            or selection.get("source_cache_sha256") != contract.source_cache_sha256
            or selection.get("source_snapshot_sha256") != contract.source_snapshot_sha256
            or selection.get("source_identity_digest") != contract.source_identity_digest
            or selection.get("full_refit_ledger_digest") != semantic_sha256(ledger)
            or selection.get("refit_initialization") != "fresh_same_seed"
            or selection.get("outer_target_used_for_selection") is not False
            or selection.get("epoch_unit") != "ceil(inner_rows/batch_size)_source_updates"
            or selection.get("validation_unit") != "source_participant_disjoint_within_each_domain"
            or selection.get("monitor") != "source_validation_hierarchical_weighted_cross_entropy"):
        raise ValueError("source selection/refit requires actual same-seed producer evidence")
    policy = GlobalEarlyStoppingPolicy(**selection["policy"])
    if payload.get("global_early_stopping") != policy.record():
        raise ValueError("source selected policy differs from actual producer declaration")
    training, validation = (np.asarray(selection[key]) for key in ("training_rows", "validation_rows"))
    inner, held = selection["inner_fit_ledger"], selection["validation_fit_ledger"]
    if (any(value.ndim != 1 or value.dtype.kind not in "iu" or not len(value)
            or len(np.unique(value)) != len(value) for value in (training, validation))
            or np.intersect1d(training, validation).size
            or not np.array_equal(np.sort(np.r_[training, validation]), np.sort(full_rows))
            or inner["fit_rows"] != training.tolist() or held["fit_rows"] != validation.tolist()
            or set(inner["training_participant_keys"]) & set(held["training_participant_keys"])
            or set(inner["training_participant_keys"]) | set(held["training_participant_keys"]) != set(ledger["training_participant_keys"])):
        raise ValueError("source inner selection and full refit must share the actual source-only row partition")
    for part in (inner, held):
        if (part["cache_row_identity_sha256"] != ledger["cache_row_identity_sha256"]
                or part["input_lineage_digest"] != ledger["input_lineage_digest"]
                or part["excluded_source_datasets"] != ledger["excluded_source_datasets"]
                or not set(part["actual_source_libraries"]) <= set(ledger["actual_source_libraries"])
                or part["fit_rows_sha256"] != semantic_sha256(part["fit_rows"])):
            raise ValueError("source inner selection rows differ from refit cache/source authority")
    if expected is not None:
        from baselines.validation import group_disjoint_validation_split

        held_identity = DatasetIdentityTable.from_payload(held["training_identity_ledger"])
        held_counts = resolve_independent_source_config(declared_config, held_identity,
                                                       checkpoint_study_membership(payload))["source_participant_counts"]
        expected_counts = {study: group_disjoint_validation_split(np.arange(count),
            fraction=policy.source_validation_fraction, min_groups=config["val_groups_min"],
            max_groups=config["val_groups_max"]).n_validation_groups
            for study, count in expected["source_participant_counts"].items()}
        if held_counts != expected_counts:
            raise ValueError("source selected validation participants differ from the declared fraction/group rule")
    check_every = ceil(len(training) / config["batch_size"])
    history = selection["inner_history"]
    early = history["early_stopping"]
    if (selection["check_every_updates"] != check_every or early != {
            key: selection[key] for key in early} or selection["maximum_epochs"] != policy.source_max_epochs):
        raise ValueError("source selection curve differs from recorded inner fit")
    stopper = PatienceStopper(policy.patience, policy.min_delta)
    for index, point in enumerate(selection["validation_curve"], start=1):
        if (stopper.exhausted or index > policy.source_max_epochs
                or point["epoch"] != index or point["updates"] != index * check_every):
            raise ValueError("source selection validation curve differs from producer update checks")
        stopper.observe(point["validation_loss"])
    if (not stopper.checks or any(selection[key] != value for key, value in stopper.record().items())
            or selection["selected_epochs"] != stopper.best_check
            or selection["selected_updates"] != stopper.best_check * check_every
            or history["optimizer_updates"] != stopper.checks * check_every
            or (not stopper.exhausted and stopper.checks != policy.source_max_epochs)
            or selection["refit_updates"] != updates
            or updates != stopper.best_check * ceil(len(full_rows) / config["batch_size"])
            or selection["total_optimizer_updates"] != history["optimizer_updates"] + updates
            or selection["total_trial_draws"] != history["trial_draws"] + contract.optimizer["trial_draws"]):
        raise ValueError("source selected epoch and actual full refit budget are inconsistent")
    return {"kind": kind, "selection": "per_source_seed", "source_seed": source_seed,
            "selected_epochs": stopper.best_check, "actual_inner_updates": history["optimizer_updates"],
            "actual_refit_updates": updates, "selection_digest": semantic_sha256(selection),
            "fit_ledger_digest": semantic_sha256(ledger), "source_evidence_scope": source_scope,
            "consumer_evidence_scope": evidence_scope,
            **({"source_program_config": program_config} if program_config is not None else {}),
            "source_studies": sorted(studies)}


def fit_source_trajectory(
    dataset: EpochDataset, rows: np.ndarray, *, config: DeepConfig,
    source_cache_sha256: str, source_snapshot_sha256: str,
    excluded_source_datasets: Sequence[str], device: torch.device,
    evidence_scope: str = "development",
    architecture: N2P3ArchitectureConfig = DEFAULT_N2P3_ARCHITECTURE,
    runtime_preload_budget_bytes: int | None = None,
    runtime_input_staging_gate=None,
    runtime_host_process_budget_bytes=None,
    runtime_progress_directory=None,
    source_program_config=None,
) -> tuple[N2P3NetBaseline, dict[int, dict]]:
    """Fit actual rows once and return loadable, source-bound candidate checkpoints."""
    if config.global_early_stopping is not None:
        return _fit_source_with_early_stopping(
            dataset, rows, config=config, source_cache_sha256=source_cache_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            excluded_source_datasets=excluded_source_datasets, device=device, evidence_scope=evidence_scope,
            architecture=architecture,
            runtime_preload_budget_bytes=runtime_preload_budget_bytes,
            runtime_input_staging_gate=runtime_input_staging_gate,
            runtime_host_process_budget_bytes=runtime_host_process_budget_bytes,
            runtime_progress_directory=runtime_progress_directory,
            source_program_config=source_program_config,
        )
    if config.fixed_updates is None:
        raise ValueError("source trajectory requires explicit fixed updates")
    if config.source_budget is not None:
        epochs = config.source_budget.get("selected_epochs")
        if (config.source_budget.get("mode") != "reference_selected_exposure"
                or type(epochs) is not int or epochs < 1
                or config.fixed_updates != epochs * ceil(len(rows) / config.batch_size)):
            raise ValueError("source updates differ from the selected reference exposure budget")
    ledger = source_fit_ledger(dataset, rows, excluded_source_datasets=excluded_source_datasets)
    study_membership = dataset_study_membership(dataset)
    domains, people, _ = source_identity_axes(dataset)
    rows = np.asarray(rows, np.int64)
    observed = (np.broadcast_to(dataset.channel_mask, (len(rows), dataset.n_channels)).copy()
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask[rows])
    unique_domains = sorted(set(domains[rows]))
    model = N2P3NetBaseline(dataset.n_channels, dataset.n_times, dataset.preprocessing.sfreq,
                           config=config, device=device, channel_mask=dataset.channel_mask,
                           architecture=architecture,
                           tmin_s=dataset.preprocessing.tmin_ms / 1000)
    model.runtime.configure_resident_memory_budget(runtime_preload_budget_bytes)
    model.runtime.input_staging_gate = runtime_input_staging_gate
    model.runtime.configure_host_process_resident_budget(runtime_host_process_budget_bytes)
    if runtime_progress_directory is not None:
        model.configure_epoch_progress(Path(runtime_progress_directory) / "fixed")
        model.configure_evaluation_fold(0)
    from train.source_rows import SourceRows

    model.fit(SourceRows(dataset.X, rows), np.asarray(dataset.y)[rows], trial_channel_mask=observed,
              source_domain_ids=domains[rows], risk_unit_ids=people[rows],
              source_domain_mass={domain: 1 / len(unique_domains) for domain in unique_domains})
    snapshots = dict(model.candidate_snapshots_)
    input_mean, input_std = model.fitted_input_statistics()
    snapshots.setdefault(config.fixed_updates, {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.model_.state_dict().items()},
        "input_mean": input_mean, "input_std": input_std,
        "optimizer_updates": config.fixed_updates,
        "trial_draws": model.last_history["trial_draws"],
        "batch_sequence_sha256": model.last_history["batch_sequence_sha256"],
        **({"source_objective": model.last_history["source_objective"]}
           if "source_objective" in model.last_history else {}),
    })
    payloads = {}
    for updates, snapshot in snapshots.items():
        contract = TrainingRunContract(
            source_cache_sha256=source_cache_sha256,
            source_identity_digest=dataset.identity_table.digest(),
            source_snapshot_sha256=source_snapshot_sha256, architecture=model.architecture_record(),
            preprocessing={**preprocessing_execution_binding(dataset.provenance.get("preprocessing_execution"), dataset.preprocessing),
                           "epoch": dataset.preprocessing.record(), "channel_names": list(dataset.channel_names),
                           "source_reference": dataset.provenance.get("source_reference")},
            optimizer={"name": "torch.optim.Adam", "config": asdict(config), "actual_updates": updates,
                       "trial_draws": snapshot["trial_draws"], "batch_sequence_sha256": snapshot["batch_sequence_sha256"]},
            validation={"strategy": "fixed_source_updates", "fit_ledger": ledger, "evidence_scope": evidence_scope},
            objective={"name": "hierarchical_weighted_CE", "sampling": model.sampling_ledger_,
                       **({"source_program_config": source_program_config} if source_program_config is not None else {}),
                       **({"source_auxiliary": snapshot["source_objective"]}
                          if "source_objective" in snapshot else {}),
                       "input_statistics": model.source_statistics_, "source_study_membership": study_membership}, seed=config.seed,
            training_participant_keys=tuple(ledger["training_participant_keys"]), holdout_participant_keys=(),
        )
        payloads[updates] = {
            "schema": CHECKPOINT_SCHEMA, "trunk_state_dict": snapshot["model_state_dict"],
            "trunk_state_dict_sha256": checkpoint_state_digest(snapshot["model_state_dict"]),
            "architecture": model.architecture_record(), "n_channels": dataset.n_channels,
            "n_times": dataset.n_times, "input_sample_rate_hz": dataset.preprocessing.sfreq,
            "input_tmin_s": dataset.preprocessing.tmin_ms / 1000,
            "input_preprocessing": dataset.preprocessing.record(), "input_channel_names": list(dataset.channel_names),
            "input_preprocessing_execution": dataset.provenance.get("preprocessing_execution"),
            "input_source_reference": dataset.provenance.get("source_reference"),
            "input_mean": snapshot["input_mean"].reshape(-1).tolist(),
            "input_std": snapshot["input_std"].reshape(-1).tolist(),
            "classifier_trained": True,
            "source_calibration": {"pos_weight": config.pos_weight, "train_prior": model.training_prior_,
                                   "temperature": 1.0, "source": "source_fixed_updates_weighted_ce_analytic"},
            "training_identity_ledger": ledger["training_identity_ledger"],
            "training_identity_ledger_digest": ledger["training_identity_ledger_digest"],
            "training_contract": contract.record(), "training_contract_digest": contract.digest(),
            "source_cache_sha256": source_cache_sha256, "source_snapshot_sha256": source_snapshot_sha256,
            "source_fixed_updates": updates, "source_full_refit": True,
            **({"source_budget": config.source_budget} if config.source_budget is not None else {}),
            "source_fit_ledger": ledger, "source_statistics": model.source_statistics_,
            "source_sampling": model.sampling_ledger_, "runtime": model.last_runtime,
            "source_study_membership": study_membership,
            "evidence_scope": evidence_scope,
        }
    return model, payloads


def _fit_source_with_early_stopping(
    dataset, rows, *, config, source_cache_sha256, source_snapshot_sha256,
    excluded_source_datasets, device, evidence_scope, architecture, runtime_preload_budget_bytes=None,
    runtime_input_staging_gate=None,
    runtime_host_process_budget_bytes=None,
    runtime_progress_directory=None,
    source_program_config=None,
):
    """Hold out source participants, select epochs, then independently refit all source rows."""
    from transfer.checkpoint import checkpoint_training_contract

    policy = config.global_early_stopping
    started = time.perf_counter()
    rows = np.asarray(rows, dtype=np.int64)
    source_fit_ledger(dataset, rows, excluded_source_datasets=excluded_source_datasets)
    domains, people, _ = source_identity_axes(dataset)
    split = domain_stratified_group_validation_split(
        people[rows], domains[rows], fraction=policy.source_validation_fraction,
        min_groups=config.val_groups_min, max_groups=config.val_groups_max, seed=config.seed,
    )
    training, validation = rows[split.train_mask], rows[split.validation_mask]
    inner_ledger = source_fit_ledger(dataset, training, excluded_source_datasets=excluded_source_datasets,
        forbidden_participant_keys=people[validation])
    observed = (np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask)
    check_every = ceil(len(training) / config.batch_size)
    inner_config = replace(config, global_early_stopping=None, val_group_frac=None,
                           fixed_updates=policy.source_max_epochs * check_every)
    inner = N2P3NetBaseline(dataset.n_channels, dataset.n_times, dataset.preprocessing.sfreq,
        config=inner_config, device=device, channel_mask=dataset.channel_mask,
        architecture=architecture,
        tmin_s=dataset.preprocessing.tmin_ms / 1000)
    domain_mass = {domain: 1 / len(set(domains[rows])) for domain in sorted(set(domains[rows]))}
    vy = np.asarray(dataset.y)[validation]
    multipliers, _ = resolve_source_risk_weights(vy, domains[validation], people[validation],
        domain_mass=domain_mass, within_domain_unit="participant", pos_weight=config.pos_weight)
    weights = multipliers * np.where(vy == 1, config.pos_weight, 1.0)
    weights = weights / weights.sum()
    inner.runtime.configure_resident_memory_budget(runtime_preload_budget_bytes)
    inner.runtime.input_staging_gate = runtime_input_staging_gate
    inner.runtime.configure_host_process_resident_budget(runtime_host_process_budget_bytes)
    if runtime_progress_directory is not None:
        inner.configure_epoch_progress(Path(runtime_progress_directory) / "selection")
        inner.configure_evaluation_fold(0)
    from train.source_rows import SourceRows

    inner.fit(SourceRows(dataset.X, training), np.asarray(dataset.y)[training], trial_channel_mask=observed[training],
        source_domain_ids=domains[training], risk_unit_ids=people[training], source_domain_mass=domain_mass,
        early_stopping_validation={"X": SourceRows(dataset.X, validation), "y": vy,
            "channel_mask": observed[validation], "weights": weights,
            "check_every_updates": check_every, "policy": policy})
    history = inner.last_history
    selection = {**history["early_stopping"], "policy": policy.record(),
        "source_seed": config.seed, "source_cache_sha256": source_cache_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_identity_digest": dataset.identity_table.digest(),
        "monitor": "source_validation_hierarchical_weighted_cross_entropy",
        "validation_unit": "source_participant_disjoint_within_each_domain",
        "epoch_unit": "ceil(inner_rows/batch_size)_source_updates",
        "check_every_updates": check_every,
        "training_rows": training.tolist(), "validation_rows": validation.tolist(),
        "inner_fit_ledger": inner_ledger,
        "validation_fit_ledger": source_fit_ledger(dataset, validation,
            excluded_source_datasets=excluded_source_datasets, forbidden_participant_keys=people[training]),
        "inner_history": history, "inner_runtime": inner.last_runtime,
        "outer_target_used_for_selection": False,
        "refit_initialization": "fresh_same_seed", "maximum_epochs": policy.source_max_epochs}
    selection["reached_epoch_limit"] = selection["validation_checks"] >= policy.source_max_epochs
    del inner
    refit_updates = selection["selected_epochs"] * ceil(len(rows) / config.batch_size)
    model, payloads = fit_source_trajectory(dataset, rows,
        config=replace(config, global_early_stopping=None, val_group_frac=None, fixed_updates=refit_updates),
        source_cache_sha256=source_cache_sha256, source_snapshot_sha256=source_snapshot_sha256,
        excluded_source_datasets=excluded_source_datasets, device=device, evidence_scope=evidence_scope,
        architecture=architecture, runtime_preload_budget_bytes=runtime_preload_budget_bytes,
        runtime_input_staging_gate=runtime_input_staging_gate,
        runtime_host_process_budget_bytes=runtime_host_process_budget_bytes,
        runtime_progress_directory=(Path(runtime_progress_directory) / "refit" if runtime_progress_directory is not None else None),
        source_program_config=source_program_config)
    selection.update(refit_updates=refit_updates,
        total_optimizer_updates=history["optimizer_updates"] + model.last_history["optimizer_updates"],
        total_trial_draws=history["trial_draws"] + model.last_history["trial_draws"],
        refit_runtime=dict(model.last_runtime), total_wall_seconds=time.perf_counter() - started)
    selection["full_refit_ledger_digest"] = semantic_sha256(payloads[refit_updates]["source_fit_ledger"])
    model.last_runtime = {**model.last_runtime, "fit_seconds": selection["total_wall_seconds"],
                          "includes_inner_and_full_refit": True}
    for payload in payloads.values():
        original = checkpoint_training_contract(payload)
        contract = replace(original, validation={**original.validation,
            "strategy": "source_internal_early_stopping_then_full_refit", "early_stopping": selection})
        payload.update(training_contract=contract.record(), training_contract_digest=contract.digest(),
                       global_early_stopping=policy.record(), source_epoch_selection=selection,
                       runtime=model.last_runtime)
    model.last_history = {**model.last_history, "source_epoch_selection": selection,
                          "optimizer_updates": selection["total_optimizer_updates"],
                          "trial_draws": selection["total_trial_draws"]}
    return model, payloads
