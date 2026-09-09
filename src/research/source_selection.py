"""Source-only training-chain folds, dose manifests, and fixed-cohort selection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from baselines.validation import domain_stratified_group_validation_split
from data.domain import source_domain_ids
from data.epochs import EpochDataset, epoch_row_identity_digest, materialize_dataset_lineage
from data.identity import DatasetIdentityTable, training_identity_ledger_from_rows
from data.study import dataset_study_membership, study_ids_for_identity
from research.contracts import TrainingRunContract, semantic_sha256
from research.domain_folds import domain_lodo_folds
from transfer.checkpoint import CHECKPOINT_SCHEMA

SOURCE_UPDATE_CANDIDATES = (500, 1000, 2000, 4000, 8000)
SOURCE_CHAIN_LAYOUTS = {"six": ("Fz", "P3", "Pz", "P4", "PO8", "Oz"),
                        "common_three": ("P3", "Pz", "P4")}


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
    people = np.asarray([keys[str(local)] for local in dataset.subject_ids])
    libraries: dict[str, set[str]] = {}
    library_domains: dict[str, str] = {}
    for local in np.unique(dataset.subject_ids):
        local_domains = np.unique(domains[dataset.subject_ids == local])
        if len(local_domains) != 1:
            raise ValueError("one actual participant appears in multiple source domains")
        domain = local_domains[0]
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
        "cache_row_identity_sha256": epoch_row_identity_digest(dataset),
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


def fit_source_trajectory(
    dataset: EpochDataset, rows: np.ndarray, *, config: DeepConfig,
    source_cache_sha256: str, source_snapshot_sha256: str,
    excluded_source_datasets: Sequence[str], device: torch.device,
    evidence_scope: str = "development",
) -> tuple[N2P3NetBaseline, dict[int, dict]]:
    """Fit actual rows once and return loadable, source-bound candidate checkpoints."""
    if config.fixed_updates is None:
        raise ValueError("source trajectory requires explicit fixed updates")
    ledger = source_fit_ledger(dataset, rows, excluded_source_datasets=excluded_source_datasets)
    study_membership = dataset_study_membership(dataset)
    domains, people, _ = source_identity_axes(dataset)
    rows = np.asarray(rows, np.int64)
    observed = (np.broadcast_to(dataset.channel_mask, (len(rows), dataset.n_channels)).copy()
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask[rows])
    unique_domains = sorted(set(domains[rows]))
    model = N2P3NetBaseline(dataset.n_channels, dataset.n_times, dataset.preprocessing.sfreq,
                           config=config, device=device, channel_mask=dataset.channel_mask,
                           tmin_s=dataset.preprocessing.tmin_ms / 1000)
    model.fit(dataset.X[rows], np.asarray(dataset.y)[rows], trial_channel_mask=observed,
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
    })
    payloads = {}
    for updates, snapshot in snapshots.items():
        contract = TrainingRunContract(
            source_cache_sha256=source_cache_sha256,
            source_identity_digest=dataset.identity_table.digest(),
            source_snapshot_sha256=source_snapshot_sha256, architecture=model.architecture_record(),
            preprocessing={"epoch": dataset.preprocessing.record(), "channel_names": list(dataset.channel_names),
                           "source_reference": dataset.provenance.get("source_reference")},
            optimizer={"name": "torch.optim.Adam", "config": asdict(config), "actual_updates": updates,
                       "trial_draws": snapshot["trial_draws"], "batch_sequence_sha256": snapshot["batch_sequence_sha256"]},
            validation={"strategy": "fixed_source_updates", "fit_ledger": ledger, "evidence_scope": evidence_scope},
            objective={"name": "hierarchical_weighted_CE", "sampling": model.sampling_ledger_,
                       "input_statistics": model.source_statistics_, "source_study_membership": study_membership}, seed=config.seed,
            training_participant_keys=tuple(ledger["training_participant_keys"]), holdout_participant_keys=(),
        )
        payloads[updates] = {
            "schema": CHECKPOINT_SCHEMA, "trunk_state_dict": snapshot["model_state_dict"],
            "architecture": model.architecture_record(), "n_channels": dataset.n_channels,
            "n_times": dataset.n_times, "input_sample_rate_hz": dataset.preprocessing.sfreq,
            "input_tmin_s": dataset.preprocessing.tmin_ms / 1000,
            "input_preprocessing": dataset.preprocessing.record(), "input_channel_names": list(dataset.channel_names),
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
            "source_fit_ledger": ledger, "source_statistics": model.source_statistics_,
            "source_sampling": model.sampling_ledger_, "runtime": model.last_runtime,
            "source_study_membership": study_membership,
            "evidence_scope": evidence_scope,
        }
    return model, payloads
