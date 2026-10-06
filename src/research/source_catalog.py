"""Explicit source composition, exclusions and dataset-count-independent folds.

A pool is an experiment declaration, never evidence that its local data are
ready. Prepared artifacts are validated by data.source_admission separately.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from data.identity import DatasetIdentityTable, source_dataset_ids_from_identity
from research.contracts import semantic_sha256
from transfer.checkpoint import load_checkpoint_payload

CORE_STUDIES = ("BI2014a", "BI2015a", "BNCI2014-008", "Huebner2017")
ADDED_STUDIES = ("BI2013a", "EPFLP300", "Chailloux2020", "Speier2017")
DEFAULT_POOLS = {
    "B0": CORE_STUDIES,
    "B1": (*CORE_STUDIES, "GTN"),
    **{f"B{i + 2}": (*CORE_STUDIES, study) for i, study in enumerate(ADDED_STUDIES)},
    "B6": (*CORE_STUDIES, *ADDED_STUDIES),
    "B7": (*CORE_STUDIES, *ADDED_STUDIES, "GTN"),
}


def study_names(values, *, field, allow_empty=False):
    """Require explicit unique physical study names; never expand aliases."""
    if not isinstance(values, (list, tuple)) or (not values and not allow_empty):
        raise ValueError(f"{field} requires an explicit list of study names")
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values):
        raise ValueError(f"{field} contains an invalid study name")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate study names")
    return tuple(values)


class SourceCatalog:
    """Resolve a declared pool after exclusions and check its actual inputs.

    Default B0--B7 memberships remain the already executed historical design.
    New compositions use explicit new names; existing names cannot change.
    """

    def __init__(self, pools: Mapping[str, Sequence[str]] | None = None):
        self._pools = dict(DEFAULT_POOLS)
        if pools is not None:
            if not isinstance(pools, Mapping):
                raise ValueError("source_pools must map pool names to study lists")
            for name, values in pools.items():
                if not isinstance(name, str) or not name.strip() or name != name.strip():
                    raise ValueError("source pool requires a nonempty name")
                studies = study_names(values, field=f"source pool {name}")
                if name in self._pools and studies != self._pools[name]:
                    raise ValueError(f"source pool {name} already has a frozen definition; use a new name")
                self._pools[name] = studies

    def studies(self, pool, *, excluded=()):
        if pool not in self._pools:
            raise ValueError(f"unknown source pool {pool!r}")
        exclusions = study_names(excluded, field="source exclusions", allow_empty=True)
        members = tuple(study for study in self._pools[pool] if study not in exclusions)
        if not members:
            raise ValueError(f"source pool {pool!r} is empty after exclusions")
        return members

    def validate_fold(self, outer, actual_domains):
        """Require exact declared membership, retaining legacy fixed-count plans."""
        role = outer.get("role", "evaluation")
        if role not in {"evaluation", "deployment"}:
            raise ValueError("source group role must be evaluation or deployment")
        deployment = role == "deployment"
        excluded = study_names(outer["excluded_source_datasets"], field="outer exclusions", allow_empty=deployment)
        if deployment and (excluded or any(outer.get(field) for field in
                ("targets", "target_study", "target_cache", "target_prefix_manifest", "target_protocol", "family_exclusion"))):
            raise ValueError("deployment source group cannot declare exclusions or outer evaluation")
        if deployment and not ("source_pool" in outer or "source_studies" in outer):
            raise ValueError("deployment requires explicit complete source membership")
        actual = set(actual_domains)
        if not actual or actual.intersection(excluded):
            raise ValueError("source fold is empty or contains an excluded target study")
        expected = None
        if "source_pool" in outer:
            expected = self.studies(outer["source_pool"], excluded=excluded)
        if "source_studies" in outer:
            explicit = study_names(outer["source_studies"], field="outer source_studies")
            if expected is not None and set(explicit) != set(expected):
                raise ValueError("source_studies differs from the declared pool after exclusions")
            expected = explicit
        if expected is None:
            if len(actual) != (2 if outer.get("family_exclusion", False) else 3):
                raise ValueError("expanded source folds require explicit source_studies or source_pool")
            expected = tuple(sorted(actual))
        if set(expected).intersection(excluded) or actual != set(expected):
            raise ValueError(f"source fold membership differs: expected={list(expected)}, actual={sorted(actual)}")
        targets = [target["study"] for target in outer.get("targets", [])]
        if outer.get("target_study") is not None:
            targets.append(outer["target_study"])
        if not set(targets) <= set(excluded):
            raise ValueError("all outer target studies must be excluded from source training")
        return expected


DEFAULT_SOURCE_CATALOG = SourceCatalog()


def paired_pipeline_seeds(values):
    """Three declared complete pipeline pairs, never a Cartesian product."""
    pairs = tuple(tuple(pair) for pair in values)
    if (len(pairs) != 3 or any(len(pair) != 2 or any(type(seed) is not int or seed < 0 for seed in pair)
            for pair in pairs) or len({pair[0] for pair in pairs}) != 3 or len({pair[1] for pair in pairs}) != 3):
        raise ValueError("complete pipeline replication requires three distinct paired source/target seeds")
    return pairs


class SourceSeedCatalog:
    """Admit exact source bytes by outer study and actual complete source seed.

    Selection/refit needs a per-seed producer record with measured outcomes and
    a bound final refit. A declaration of a shared development selection is not
    evidence of three complete source procedures. Fixtures retain their scope.
    """

    def __init__(self, entries, *, base_directory, evidence_scope):
        self._entries = {}
        self._records = []
        for entry in entries:
            key = (entry["outer_target_study"], entry["source_seed"])
            if key in self._entries or type(key[1]) is not int or key[1] < 0:
                raise ValueError("source catalog requires unique outer-study/source-seed keys")
            path = Path(base_directory) / entry["path"]
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("source catalog checkpoint bytes differ from declaration")
            payload = load_checkpoint_payload(io.BytesIO(content))
            contract = payload["training_contract"]
            if contract["seed"] != key[1]:
                raise ValueError("source catalog key differs from actual checkpoint seed")
            studies = set(source_dataset_ids_from_identity(DatasetIdentityTable.from_payload(payload["training_identity_ledger"])))
            if key[0] in studies:
                raise ValueError("source catalog checkpoint must exclude the complete outer target study")
            procedure = entry["procedure"]
            config = contract["optimizer"].get("config", {})
            selects = any(config.get(name) is not None for name in ("source_budget", "global_early_stopping"))
            if procedure["kind"] == "fixed_recipe":
                if procedure["selection"] != "not_applicable" or selects:
                    raise ValueError("fixed source recipe cannot hide a selection procedure")
                producer = None
            elif procedure["kind"] == "selection_refit":
                producer_path = Path(base_directory) / procedure["record_path"]
                producer_bytes = producer_path.read_bytes()
                if hashlib.sha256(producer_bytes).hexdigest() != procedure["record_sha256"]:
                    raise ValueError("source procedure producer bytes changed")
                producer = json.loads(producer_bytes)
                selection, refit = producer["selection"], producer["refit"]
                if (producer["source_seed"] != key[1] or selection["source_seed"] != key[1]
                        or refit["source_seed"] != key[1] or producer["outer_target_study"] != key[0]
                        or producer["evidence_scope"] != evidence_scope
                        or selection["status"] != "selected" or not selection["outcomes"]
                        or refit["status"] != "completed" or refit["checkpoint_sha256"] != digest
                        or refit["training_contract_digest"] != payload["training_contract_digest"]
                        or selection["source_snapshot_sha256"] != contract["source_snapshot_sha256"]
                        or not refit["physical_optimizer_updates"]):
                    raise ValueError("source selection/refit requires actual same-seed producer evidence")
            else:
                raise ValueError("source procedure must declare fixed recipe or actual selection/refit")
            fixture = contract["optimizer"].get("kind") == "synthetic_test_fixture"
            if fixture and evidence_scope != "software_correctness":
                raise ValueError("software source fixtures cannot establish public full-source pipeline replication")
            self._entries[key] = content
            self._records.append({"outer_target_study": key[0], "source_seed": key[1], "checkpoint_sha256": digest,
                "training_contract_digest": payload["training_contract_digest"], "source_studies": sorted(studies),
                "procedure": procedure, "producer_digest": None if producer is None else semantic_sha256(producer),
                "evidence_scope": evidence_scope, "source_selection": "not_applicable" if producer is None else "per_source_seed"})

    def content(self, outer_target_study, source_seed):
        return self._entries[(outer_target_study, source_seed)]

    def record(self):
        return self._records
