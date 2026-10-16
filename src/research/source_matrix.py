"""Explicit study pools and source-only matched-dose allocation."""
from __future__ import annotations

import numpy as np

from research.contracts import semantic_sha256
from research.source_catalog import ADDED_STUDIES as ADDED_STUDIES
from research.source_catalog import CORE_STUDIES as CORE_STUDIES
from research.source_catalog import DEFAULT_POOLS, SourceCatalog, study_names

SOURCE_POOLS = dict(DEFAULT_POOLS)
MATRIX_SEEDS = (20260909, 20260910, 20260911)


def unified_source_conditions(*, supcon_weight=.05, mmd_weight=.05, temperature=.1,
                              min_cell_count=2, feature_residual=None):
    """The compact design fixes R and compares CE with the joint source objective."""
    from dataclasses import asdict

    from models.n2p3net import N2P3ArchitectureConfig
    from transfer.source_objective import SourceObjectiveConfig

    SourceObjectiveConfig(supcon_weight=supcon_weight, mmd_weight=mmd_weight,
                          temperature=temperature, min_cell_count=min_cell_count)
    if min(supcon_weight, mmd_weight) <= 0:
        raise ValueError("unified source enabled factors require positive auxiliary weights")
    residual = (N2P3ArchitectureConfig.unified().feature_residual
                if feature_residual is None else dict(feature_residual))
    conditions = []
    for r, s, m in ((1, 0, 0), (1, 1, 1)):
        architecture = N2P3ArchitectureConfig(observed_channel_norm=True, feature_residual=residual if r else None)
        objective = SourceObjectiveConfig(supcon_weight=supcon_weight if s else 0.,
            mmd_weight=mmd_weight if m else 0., temperature=temperature,
            min_cell_count=min_cell_count, feature_location="post_residual" if r else "post_mst")
        conditions.append({"source_arm": f"unified_r{r}s{s}m{m}", "source_factors": {"R": r, "S": s, "M": m},
            "architecture_configuration": asdict(architecture),
            "source_objective": asdict(objective) if objective.enabled else None})
    return tuple(conditions)


def validate_unified_source_input(dataset, preprocessing):
    """Require the actual declared operator and complete execution receipts."""
    from data.epochs import parse_preprocessing_spec, preprocessing_execution_identity

    declared = parse_preprocessing_spec(preprocessing)
    if dataset.preprocessing != declared:
        raise ValueError("unified source cache differs from its declared preprocessing profile")
    if preprocessing_execution_identity(dataset) is None:
        raise ValueError("unified source cache requires complete actual preprocessing execution evidence")


def validate_unified_source_factors(payload, factors):
    """Compare treatment declarations with the executed source checkpoint."""
    from transfer.checkpoint import checkpoint_training_contract
    from transfer.source_objective import SourceObjectiveConfig

    if (not isinstance(factors, dict) or set(factors) != {"R", "S", "M"}
            or any(type(value) is not int or value not in (0, 1) for value in factors.values())):
        raise ValueError("unified source factors require explicit binary R, S and M")
    contract = checkpoint_training_contract(payload)
    architecture = contract.architecture
    if architecture.get("channel_masking") != "observed_temporal_bn/1":
        raise ValueError("unified source requires observed-channel temporal normalization")
    raw = contract.optimizer.get("config", {}).get("source_objective")
    objective = SourceObjectiveConfig(**raw) if raw is not None else SourceObjectiveConfig()
    actual = {"R": int(architecture.get("feature_residual") is not None),
              "S": int(objective.supcon_weight > 0), "M": int(objective.mmd_weight > 0)}
    if factors != actual:
        raise ValueError("unified source factors differ from the executed checkpoint")
    if objective.enabled and objective.feature_location != ("post_residual" if actual["R"] else "post_mst"):
        raise ValueError("unified source objective location differs from the executed architecture")
    return dict(actual)


def source_matrix_tasks(plan=None):
    """Compile explicit pools, target exclusions and seeds into training tasks.

    Omission preserves all 153 historical tasks. A new JSON plan supplies its
    entire design; it never inherits unrelated target or standalone jobs.
    """
    if plan is None:
        plan = {"schema": "n2p3_source_pool_plan/1", "source_pools": SOURCE_POOLS,
                "seeds": MATRIX_SEEDS, "targets": {s: [s] for s in CORE_STUDIES},
                "matched_pools": ["B0", "B1", "B6", "B7"], "product_pools": ["B0", "B7"],
                "standalone": {"GTN": {"studies": ["GTN"], "channels": ["Fz", "Pz"]}}}
    allowed = {"schema", "source_pools", "seeds", "targets", "matched_pools", "product_pools", "standalone"}
    if not isinstance(plan, dict) or plan.get("schema") != "n2p3_source_pool_plan/1" or set(plan) - allowed:
        raise ValueError("unsupported source pool plan schema or fields")
    pools = plan.get("source_pools", {})
    catalog = SourceCatalog(pools)
    seeds = plan.get("seeds", ())
    if (not isinstance(seeds, (tuple, list)) or not seeds or
            any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("source matrix seeds must be unique nonnegative integers")
    matched = study_names(plan.get("matched_pools", []), field="matched_pools", allow_empty=True)
    products = study_names(plan.get("product_pools", []), field="product_pools", allow_empty=True)
    if not set((*matched, *products)) <= set(pools):
        raise ValueError("matched/product pools must belong to the declared source_pools")
    targets = plan.get("targets", {})
    standalone = plan.get("standalone", {})
    if not isinstance(targets, dict) or not isinstance(standalone, dict):
        raise ValueError("targets and standalone must be mappings")
    for target, excluded in targets.items():
        if target not in study_names(excluded, field=f"target {target} exclusions"):
            raise ValueError("each target must be in its whole-study exclusion set")
    tasks = []
    for seed in seeds:
        for name, declaration in standalone.items():
            if set(declaration) != {"studies", "channels"}:
                raise ValueError("standalone source requires explicit studies and observed channels")
            studies = study_names(declaration["studies"], field="standalone studies")
            channels = study_names(declaration["channels"], field="standalone channels")
            tasks.append(dict(id=f"A_{name}_s{seed}", family="A", pool=name, studies=list(studies),
                              excluded=[], channels=list(channels), dose="full", seed=seed))
        for pool in products:
            tasks.append(dict(id=f"Bproduct_{pool}_s{seed}", family="Bproduct", pool=pool,
                studies=list(catalog.studies(pool)), excluded=[], channels=None, dose="full", seed=seed))
        for target, excluded in targets.items():
            for pool in pools:
                for dose in (("full", "matched") if pool in matched else ("full",)):
                    tasks.append(dict(id=f"Bpublic_{pool}_{target}_{dose}_s{seed}", family="Bpublic",
                        pool=pool, target=target, studies=list(catalog.studies(pool, excluded=excluded)),
                        excluded=list(excluded), channels=None, dose=dose, seed=seed))
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("source matrix requires nonempty unique task identities")
    if any(any(c in t["id"] for c in '/\\:') or '..' in t["id"] for t in tasks):
        raise ValueError("source task identities must be plain directory names")
    return tasks


def select_source_matrix_tasks(*, selected_ids=None, plan=None):
    """Select requested jobs without changing the declared comparison design."""
    all_tasks = source_matrix_tasks(plan)
    if selected_ids is not None:
        selected = set(selected_ids)
        unknown = selected - {task["id"] for task in all_tasks}
        if unknown:
            raise ValueError(f"unknown selected source task: {sorted(unknown)}")
        tasks = [task for task in all_tasks if task["id"] in selected]
    else:
        tasks = list(all_tasks)
    if not tasks:
        raise ValueError("source matrix has no executable tasks")
    return tuple(tasks)


def matched_dose_tasks(selected, declared):
    """Include all comparators/seeds for selected folds, so probes keep dose."""
    targets = {t["target"] for t in selected if t["dose"] == "matched"}
    return tuple(t for t in declared if t["dose"] == "matched" and t["target"] in targets)


def choose_dose_participants(available, studies, count, seed):
    """Stable per-study permutations with capped round-robin domain allocation."""
    if count < len(studies) or count > sum(len(available[s]) for s in studies):
        raise ValueError("matched dose cannot represent every requested source study")
    ordered = {}
    for study in sorted(studies):
        values = sorted(available[study])
        rng = np.random.default_rng(int(semantic_sha256(dict(seed=seed, study=study))[:16], 16))
        ordered[study] = rng.permutation(values).tolist()
    chosen = []
    while len(chosen) < count:
        for study, people in ordered.items():
            if people and len(chosen) < count:
                chosen.append((study, people.pop()))
    return chosen


def freeze_matched_dose(counts, *, tasks=None):
    """Derive P and class quotas exclusively from admitted source participant counts."""
    tasks = [task for task in (source_matrix_tasks() if tasks is None else tasks)
             if task["dose"] == "matched"]
    required = {s for t in tasks for s in t["studies"]}
    if not required <= set(counts):
        raise ValueError(f"matched source dose lacks study counts: {sorted(required - set(counts))}")
    for study in required:
        for counts_by_class in counts[study].values():
            if len(counts_by_class) != 2 or any(type(n) is not int or n < 0 for n in counts_by_class):
                raise ValueError("source dose requires nonnegative binary class counts")
    available = {s: {p for p, n in counts[s].items() if min(n) > 0} for s in required}
    folds, selections = {}, {}
    for target in dict.fromkeys(t["target"] for t in tasks):
        fold_tasks = [t for t in tasks if t["target"] == target]
        if any(set(t["studies"]) & set(t["excluded"]) or target not in t["excluded"] for t in fold_tasks):
            raise ValueError("matched dose contains a held-out target study")
        fold_studies = {s for t in fold_tasks for s in t["studies"]}
        people = min(sum(len(available[s]) for s in task["studies"]) for task in fold_tasks)
        chosen = {task["id"]: choose_dose_participants(available, task["studies"], people, task["seed"])
                  for task in fold_tasks}
        class_quota = [min(counts[s][p][c] for selection in chosen.values() for s, p in selection)
                       for c in (0, 1)]
        if min(class_quota) < 1:
            raise ValueError("matched source dose lacks both classes")
        folds[target] = dict(participants=people, per_participant_class_counts=class_quota,
            unique_trials=people * sum(class_quota), class_counts=[people * n for n in class_quota],
            source_counts_sha256=semantic_sha256({s: counts[s] for s in sorted(fold_studies)}))
        selections.update(chosen)
    return dict(schema="n2p3_matched_source_dose/2", folds=folds, selections=selections,
                selection="source-only capped round-robin participants; seeded class-stratified unique rows",
                counts_sha256=semantic_sha256(counts))


def matched_dose_rows(dataset, task, dose, people, domains):
    rows = []
    labels = np.asarray(dataset.y)
    for study, person in dose["selections"][task["id"]]:
        rng = np.random.default_rng(int(semantic_sha256(dict(seed=task["seed"], study=study, person=person))[:16], 16))
        for label, count in enumerate(dose["folds"][task["target"]]["per_participant_class_counts"]):
            choices = np.flatnonzero((people == person) & (domains == study) & (labels == label))
            if len(choices) < count:
                raise ValueError("admitted source dose changed")
            rows.extend(rng.choice(choices, count, replace=False).tolist())
    return np.sort(np.asarray(rows, dtype=np.int64))
