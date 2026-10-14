"""Matched ERP, original waveform-LDA and source/scratch NN comparisons.

Plans declare evidence scope, participants, budgets, seeds, arms and allowed
input changes. Unavailable budgets remain in the requested ledger. Only verified
independent raw C/T prefixes are accepted; public runs require an executed freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.structured_erp import ERP_FAMILIES, StructuredERP  # noqa: E402
from baselines.temporal_transfer import TemporalBasis, resolve_temporal_basis  # noqa: E402
from data.digests import compact_array_digest  # noqa: E402
from data.epochs import concatenate_epoch_datasets, select_epoch_channels  # noqa: E402
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord  # noqa: E402
from data.source_records import (  # noqa: E402
    file_sha256,
    load_target_prefixes,
    prepared_calibration_requests,
)
from data.study import (  # noqa: E402
    checkpoint_study_membership,
    dataset_study_membership,
    study_ids_for_identity,
)
from models.n2p3net import N2P3Net  # noqa: E402
from research.b12_contract import validate_b12_execution_manifest  # noqa: E402
from research.binary_probability import (  # noqa: E402
    binary_probability_metrics,
    export_binary_pools,
)
from research.cloud_execution import worker_compute_phase  # noqa: E402
from research.comparison import (  # noqa: E402
    ContrastPlan,
    evaluation_input_identity,
    validate_input_identity,
)
from research.contracts import semantic_sha256  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_catalog import SourceSeedCatalog, paired_pipeline_seeds  # noqa: E402
from stats.transfer import (  # noqa: E402
    ANALYSIS_SEED,
    analyze_transfer_contrasts,
    resolve_seed_pairs,
)
from transfer.binary import (  # noqa: E402
    matched_input_digests,
    matched_split_digests,
)
from transfer.calibration_candidates import (  # noqa: E402
    fit_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.calibration_state import FrozenTrialPredictor  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    PreparedTransferCheckpoint,
    ScopedSourceCheckpointPreparation,
    load_checkpoint_payload,
)
from transfer.recipes import TargetRecipe, erp_target_recipe  # noqa: E402

NN_FAMILIES = frozenset({"full_fine", "scratch"})
FAMILIES = ERP_FAMILIES | NN_FAMILIES | {"waveform_lda"}


def _executing_tree():
    paths = sorted([*ROOT.joinpath("src").rglob("*.py"), Path(__file__)])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _write_binary_json(path, report):
    from research.source_metadata_store import compact_source_semantics

    return atomic_write_json(path, compact_source_semantics(report))


def load_binary_report(path, *, metadata_store=None):
    """Independent ingress restores complete probability origins from original bytes."""
    from research.source_metadata_store import _SourceMetadataAdmission

    return _load_binary_report(path, _SourceMetadataAdmission(store=metadata_store))


def _load_binary_report(path, admission):
    from research.source_metadata_store import expand_source_semantics

    report = json.loads(Path(path).read_bytes())
    if report["schema"] not in {"n2p3_erp_comparator_result/2", "n2p3_erp_comparator_result/3"}:
        raise ValueError("unsupported binary scientific report schema")
    return expand_source_semantics(report, admission)


def _resolve(plan_path, value):
    path = Path(value)
    return path if path.is_absolute() else Path(plan_path).resolve().parent / path


def _checkpoint_bytes(arms, plan_path):
    """Seed/digest derive from the exact immutable bytes consumed by preparation."""
    snapshots, by_arm = {}, {}
    for arm in arms:
        if arm["family"] not in NN_FAMILIES:
            if "checkpoint" in arm or "checkpoints" in arm:
                raise ValueError("only NN anchors consume a source checkpoint")
            continue
        declared = arm.get("checkpoints")
        if declared is None:
            if "checkpoint" not in arm:
                raise ValueError("source and same-architecture scratch require an explicit checkpoint")
            declared = {None: arm["checkpoint"]}
        if not isinstance(declared, dict) or not declared:
            raise ValueError("checkpoints must map source seeds to paths")
        mapping = {}
        for declared_seed, value in declared.items():
            path = _resolve(plan_path, value)
            if path not in snapshots:
                content = path.read_bytes()
                payload = load_checkpoint_payload(io.BytesIO(content))
                snapshots[path] = (content, hashlib.sha256(content).hexdigest(), int(payload["training_contract"]["seed"]))
            content, digest, seed = snapshots[path]
            if declared_seed is not None and str(seed) != str(declared_seed):
                raise ValueError("source checkpoint seed differs from plan")
            if seed in mapping:
                raise ValueError("duplicate source checkpoint seed")
            mapping[seed] = (content, digest)
        by_arm[arm["name"]] = mapping
    seeds = sorted({seed for values in by_arm.values() for seed in values}) or [0]
    if any(set(values) != set(seeds) for values in by_arm.values()):
        raise ValueError("NN anchors must cover the same source-seed grid")
    return by_arm, seeds


def _recipe(arm, seed):
    family, batch = arm["family"], arm.get("batch_size", 64)
    if family in NN_FAMILIES:
        if type(arm.get("epochs")) is not int or arm["epochs"] < 1:
            raise ValueError("NN anchors require a predeclared positive fixed epoch budget")
        return TargetRecipe(arm["name"], initialization="source" if family == "full_fine" else "scratch",
            weights="non_bn", bn_buffers=arm.get("bn_buffers", "calibration"),
            bn_affine=arm.get("bn_affine", "trainable"), dropout=arm.get("dropout", "off"),
            input_statistics=arm.get("input_statistics", "target_prefix"),
            epochs=arm["epochs"], batch_size=batch, seed=seed,
            lr=arm.get("lr", .001), weight_decay=arm.get("weight_decay", .0001),
            trial_loss_reduction=arm.get("trial_loss_reduction", "batch_class_mass"))
    if family == "waveform_lda":
        return waveform_lda_recipe(batch_size=batch, seed=seed)
    return erp_target_recipe(family, bin_width=arm.get("bin_width", 1), seed=seed, batch_size=batch,
        temporal_basis_kind=arm.get("temporal_basis_kind"), temporal_rank=arm.get("temporal_rank"))


def _basis(arm, plan_path):
    value = arm.get("source_temporal_basis")
    if value is None:
        return None
    if arm["family"] != "temporal_lda":
        raise ValueError("only temporal_lda consumes a source temporal basis")
    return TemporalBasis.from_record(json.loads(_resolve(plan_path, value).read_bytes()))


def _combined(entry, subject, budget, arm):
    calibration = None if budget == 0 else entry["calibration"][budget]
    test = entry["test"]
    if arm.get("channel_names") is not None:
        if calibration is not None:
            calibration = select_epoch_channels(calibration, arm["channel_names"])
        test = select_epoch_channels(test, arm["channel_names"])
    shared = {}
    for key in ("source_reference", "event_contract", "candidate_task_contract", "candidate_vocabulary",
                "ocular_artifact_processing", "continuous_artifact_processing"):
        if calibration is not None and calibration.provenance.get(key) != test.provenance.get(key):
            raise ValueError(f"calibration/test input declarations differ: {key}")
        if key in test.provenance:
            shared[key] = test.provenance[key]
    if calibration is None:
        data, train, score = test, np.empty(0, dtype=np.int64), np.arange(test.n_epochs)
    else:
        data = concatenate_epoch_datasets((calibration, test), name=f"{subject}_K{budget}", provenance=shared)
        train, score = np.arange(calibration.n_epochs), np.arange(calibration.n_epochs, data.n_epochs)
    observed = np.broadcast_to(data.channel_mask, data.X.shape[:2]).copy()
    if data.trial_channel_mask is not None:
        observed &= data.trial_channel_mask
    return data, train, score, observed


def _participant_entries(arms, plan_path, subject):
    """Decompress each distinct collection once for this participant."""
    collections, entries = {}, {}
    for arm in arms:
        path = _resolve(plan_path, arm["prefix_collection"]).resolve()
        if path not in collections:
            collections[path] = load_target_prefixes(path, subjects=[subject], load_inner=False)[subject]
        entries[arm["name"]] = collections[path]
    return entries


def _prefix_bindings(plan, plan_path):
    paths = {str(_resolve(plan_path, arm["prefix_collection"]).resolve()) for arm in plan["arms"]}
    return {path: file_sha256(Path(path)) for path in sorted(paths)}


def _checkpoint_bindings(plan, plan_path):
    paths = set()
    for arm in plan["arms"]:
        if "checkpoint" in arm:
            paths.add(str(_resolve(plan_path, arm["checkpoint"]).resolve()))
        paths.update(str(_resolve(plan_path, path).resolve()) for path in arm.get("checkpoints", {}).values())
    return {path: file_sha256(Path(path)) for path in sorted(paths)}


def _participant_studies(plan, plan_path, subjects):
    studies = set()
    for path in _prefix_bindings(plan, plan_path):
        collection = json.loads(Path(path).read_bytes())
        for entry in collection["records"]:
            records = entry["records"] if entry["protocol"]["kind"] == "last_session_holdout" else [entry["record"]]
            if str(records[0]["subject"]) in subjects:
                studies.update(record["study"] for record in records)
    return studies


def _required_catalog_entries(plan, plan_path, subjects):
    """Only the participant group's actual outer keys enter its source heap."""
    studies = _participant_studies(plan, plan_path, subjects)
    return [entry for entry in plan["source_seed_catalog"] if entry["outer_target_study"] in studies]


def _source_snapshot(checkpoints, catalog_snapshots, arm, study, source_seed):
    if catalog_snapshots is None:
        return checkpoints[arm["name"]][source_seed]
    return catalog_snapshots[(study, source_seed)]


def _nn_anchor(snapshot, data, subject, preparation=None):
    content, digest = snapshot
    if preparation is None:
        prepared = PreparedTransferCheckpoint(io.BytesIO(content), data)
        training = DatasetIdentityTable.from_payload(prepared.payload["training_identity_ledger"])
        if set(study_ids_for_identity(training, checkpoint_study_membership(prepared.payload))) & set(
            study_ids_for_identity(data.identity_table, dataset_study_membership(data))
        ):
            raise ValueError("NN anchor source must exclude the complete target study")
    else:
        prepared = preparation.prepare(content, data, subject)
        prepared.assert_library_exclusion(target_membership=dataset_study_membership(data))
    trunk = prepared.trunk_for_target(subject)
    if type(trunk) is not N2P3Net or trunk.splitbn_config is not None:
        raise ValueError("NN comparator anchors require a plain source N2P3Net")
    origin = prepared.model_origin(digest)
    origin["source_snapshot_sha256"] = prepared.payload["training_contract"]["source_snapshot_sha256"]
    return prepared, trunk, origin


def _saved_replay(path, predictor, data, rows, observed, device):
    predictor.save(path)
    return FrozenTrialPredictor.load(path).predict_logit(data.X[rows], observed[rows], device=device)


def _control_matches(source, control):
    if source["actual_K"] is None or control["actual_K"] is None:
        return False
    fields = ("participant_key", "K", "source_seed", "target_seed")
    if any(source[key] != control[key] for key in fields):
        return False
    try:
        ContrastPlan().validate_pair(source["comparison_identity"], control["comparison_identity"])
    except ValueError:
        return False
    if control["family"] == "waveform_lda":
        return True
    if control["family"] != "scratch" or source["architecture"] != control["architecture"]:
        return False
    left, right = (dict(row["recipe"]) for row in (source, control))
    for recipe in (left, right):
        recipe.pop("name")
        recipe.pop("initialization")
    # A scratch control fits its target normalizer. Source normalization, when
    # explicitly requested, remains part of the transferred treatment's role.
    if right["input_statistics"] != "target_prefix":
        return False
    if left["input_statistics"] == "source":
        left["input_statistics"] = "target_prefix"
    return left == right


def _resolved_peers(source, cells):
    """Controls must own the same actual fit budget and pipeline request."""
    fields = ("participant_key", "K", "actual_K", "source_seed", "target_seed")
    key = tuple(source[field] for field in fields)
    return [(index, row) for index, row in enumerate(cells) if row["actual_K"] is not None
            and tuple(row[field] for field in fields) == key]


def _bind_matched_controls(cells):
    """The source effect is identified only by same-input/architecture/recipe controls."""
    mappings = []
    for index, source in enumerate(cells):
        if source["family"] != "full_fine" or source["actual_K"] in {None, 0}:
            continue
        peers = _resolved_peers(source, cells)
        matched = {}
        for family in ("scratch", "waveform_lda"):
            for peer_index, control in peers:
                if control["family"] != family:
                    continue
                if not _control_matches(source, control):
                    continue
                matched[family] = {"cell_index": peer_index, "arm": control["arm"]}
                break
            if family not in matched:
                raise ValueError(f"{source['arm']}/{source['participant_key']}/K{source['K']}: "
                    f"missing matched {family} control with identical actual input, architecture and fitting budget")
        source["matched_controls"] = matched
        mappings.append({"source_cell_index": index, "source_arm": source["arm"],
                         "participant_key": source["participant_key"], "K": source["K"],
                         "source_seed": source["source_seed"], "target_seed": source["target_seed"],
                         "controls": matched})
    return mappings


def _validate_transfer_contrast(name, weights, cells):
    """A legal control elsewhere in the matrix cannot certify a different contrast."""
    for source in cells:
        coefficients = weights.get("by_study", {}).get(source["study"], {}) if "by_study" in weights else weights
        if source["family"] != "full_fine" or source["arm"] not in coefficients or source["actual_K"] in {None, 0}:
            continue
        for _, control in _resolved_peers(source, cells):
            if control["family"] != "scratch" or control["arm"] not in coefficients:
                continue
            if not _control_matches(source, control):
                raise ValueError(f"contrast {name} uses an unmatched scratch control for {source['arm']}; "
                                 "an exploratory budget/architecture difference is not a transfer effect")


def _run_comparison_serial(plan_path: Path, output: Path, *, selected_subjects=None, finalize=True,
                           metadata_directory=None):
    plan = json.loads(Path(plan_path).read_bytes())
    authority_plan_sha256 = semantic_sha256(plan)
    if plan.get("schema") != "n2p3_erp_comparator_plan/1":
        raise ValueError("unsupported ERP comparator plan schema")
    scope = plan.get("evidence_scope")
    if scope not in {"public_development", "software_correctness"}:
        raise ValueError("plan requires explicit public_development or software_correctness evidence_scope")
    if selected_subjects is not None:
        if not set(selected_subjects) <= set(plan["subjects"]):
            raise ValueError("participant shard differs from the requested authority roster")
        plan = {**plan, "subjects": list(selected_subjects)}
    subjects, budgets, arms = plan["subjects"], plan["budgets"], plan["arms"]
    if not subjects or len(set(subjects)) != len(subjects) or any(not isinstance(v, str) or not v for v in subjects):
        raise ValueError("subjects must be unique explicit identifiers")
    if not budgets or len(set(budgets)) != len(budgets) or any(type(k) is not int or k < 0 for k in budgets):
        raise ValueError("comparison budgets must be unique nonnegative label budgets")
    if not arms or len({a["name"] for a in arms}) != len(arms) or any(a["family"] not in FAMILIES for a in arms):
        raise ValueError("comparator arms must be uniquely named supported families")
    paired = "seed_pairs" in plan
    pairs = paired_pipeline_seeds(plan["seed_pairs"]) if paired else None
    seeds = [pair[1] for pair in pairs] if paired else plan.get("target_seeds", [0])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("target seeds must be unique explicit integers")
    tree, freeze = _executing_tree(), None
    if scope == "public_development":
        if not plan.get("source_snapshot"):
            raise ValueError("public comparator requires an executed source_snapshot")
        freeze = verified_executed_snapshot(_resolve(plan_path, plan["source_snapshot"]), root=ROOT)
    catalog, catalog_snapshots = None, None
    if paired:
        if "source_seed_catalog" not in plan:
            raise ValueError("paired source pipelines require the actual outer-study/source-seed catalog")
        if any("checkpoint" in arm or "checkpoints" in arm for arm in arms):
            raise ValueError("paired source pipelines consume the source catalog exclusively")
        catalog = SourceSeedCatalog(_required_catalog_entries(plan, plan_path, subjects), base_directory=plan_path.parent,
                                    evidence_scope=scope, expected_source_config=plan.get("expected_source_config"))
        catalog_snapshots = {(entry["outer_target_study"], entry["source_seed"]): (
            catalog.content(entry["outer_target_study"], entry["source_seed"]), entry["checkpoint_sha256"]
        ) for entry in catalog.record()}
        checkpoints, source_seeds = {}, [pair[0] for pair in pairs]
    else:
        if "source_seed_catalog" in plan:
            raise ValueError("source seed catalog requires explicit complete pipeline seed pairs")
        checkpoints, source_seeds = _checkpoint_bytes(arms, plan_path)
        pairs = resolve_seed_pairs(source_seeds, seeds)
    if "source_seeds" in plan and list(plan["source_seeds"]) != source_seeds:
        raise ValueError("declared source seeds differ from actual checkpoint seeds")
    bases = {a["name"]: _basis(a, plan_path) for a in arms}
    contrast = ContrastPlan(tuple(plan["allowed_input_axes"]))
    contrasts = plan.get("contrasts", {f"{a['name']}_minus_{arms[0]['name']}":
        {a["name"]: 1., arms[0]["name"]: -1.} for a in arms[1:]})
    contrast_plans = {name: contrast.record() for name in contrasts} | plan.get("contrast_plans", {})
    from research.source_metadata_store import SourceMetadataStore

    metadata_store = SourceMetadataStore(metadata_directory or Path(output) / "source_metadata")
    source_preparations = {subject: ScopedSourceCheckpointPreparation() for subject in subjects}
    cells, requests, identities, authority = [], [], {}, {}
    # Keep only small preflight receipts. EEG memory is bounded to one arm and
    # participant; independent prefix loader owns per-K raw support validation.
    for subject in subjects:
        entries = _participant_entries(arms, plan_path, subject)
        for arm in arms:
            entry = entries[arm["name"]]
            test = entry["test"]
            study, = study_ids_for_identity(test.identity_table, dataset_study_membership(test))
            participant = test.identity_table.record_for(subject).authority_key()
            receipt = {"subject": subject, "study": study, "participant_key": participant,
                       "identity_record": test.identity_table.record_for(subject).payload()}
            if subject in authority and authority[subject] != receipt:
                raise ValueError("comparison arms disagree on the participant authority")
            authority[subject] = receipt
            resolved = prepared_calibration_requests(entry["manifest"]) if paired else {}
            actual_cells = {}
            for requested_budget in budgets:
                budget = resolved.get(requested_budget, {}).get("actual_K", requested_budget)
                adjustment = resolved.get(requested_budget, {}).get("adjustment")
                input_available = (not paired or requested_budget in resolved) and budget in entry["calibration"] and (
                    budget == 0 or entry["calibration"][budget] is not None)
                available = input_available and (budget > 0 or arm["family"] == "full_fine")
                if input_available and budget in actual_cells:
                    for index in actual_cells[budget]:
                        cell = cells[index]
                        cell["requested_Ks"].append(requested_budget)
                        requests.append({"study": study, "participant_key": participant, "requested_K": requested_budget,
                            "requested_Ks": [requested_budget],
                            "actual_K": cell["actual_K"], "arm": arm["name"], "source_seed": cell["source_seed"],
                            "target_seed": cell["target_seed"], "budget_adjustment": adjustment,
                            "cell_index": index, "status": "requested" if available else "protocol_excluded"})
                    continue
                if input_available:
                    data, train, score, observed = _combined(entry, subject, budget, arm)
                    identity = evaluation_input_identity(data, train, score)
                    if not identity["raw_bytes_bound"]:
                        raise ValueError("comparator requires original raw byte bindings")
                    key = (subject, budget)
                    if key in identities:
                        contrast.validate_pair(identities[key], identity)
                    else:
                        identities[key] = identity
                    digests = {**matched_split_digests(train, score),
                        **matched_input_digests(data.X, data.y, score, observed, data.preprocessing)}
                else:
                    identity, digests = None, {}
                for source_seed, seed in pairs:
                    origin, architecture = None, None
                    if available and arm["family"] in NN_FAMILIES:
                        prepared, trunk, origin = _nn_anchor(_source_snapshot(checkpoints, catalog_snapshots, arm, study, source_seed), data, subject,
                                                           source_preparations[subject])
                        architecture = trunk.architecture_record()
                        del prepared, trunk
                    recipe = _recipe(arm, seed) if budget > 0 else None
                    if available and arm["family"] in ERP_FAMILIES:
                        temporal = (resolve_temporal_basis(data, kind=recipe.temporal_basis_kind,
                            rank=recipe.temporal_rank, source_basis=bases[arm["name"]])
                            if arm["family"] == "temporal_lda" else None)
                        StructuredERP(recipe.initialization, bin_width=recipe.erp_bin_width,
                            baseline_mode=data.preprocessing.baseline_mode, temporal_basis=temporal)
                    cell = {"arm": arm["name"], "family": arm["family"], "subject": subject, "study": study,
                        "participant_key": participant, "K": budget, "requested_K": requested_budget,
                        "requested_Ks": [requested_budget], "budget_adjustments": entry["manifest"]["adjustments"] if paired else {},
                        "actual_K": budget if input_available else None, "source_seed": source_seed,
                        "target_seed": seed, "recipe": None if recipe is None else recipe.record(),
                        "prediction_program": "source_only" if budget == 0 and arm["family"] == "full_fine" else "target_calibration",
                        "status": "planned" if available else "protocol_unavailable",
                        "evidence_scope": scope, "comparison_identity": identity, **digests,
                        "available_prepared_budgets": sorted(entry["calibration"]),
                        "test_budget": entry["manifest"]["test_budget"], "checkpoint_origin": origin,
                        "architecture": architecture,
                        "source_pretraining_used": arm["family"] == "full_fine" or bases[arm["name"]] is not None,
                        "source_seed_axis_role": "checkpoint_fit_seed" if arm["family"] == "full_fine" else
                            "architecture_checkpoint_seed" if arm["family"] == "scratch" else "deterministic_control_alias",
                        "source_temporal_basis_digest": None if bases[arm["name"]] is None else bases[arm["name"]].record()["digest"]}
                    if not available:
                        cell.update(auc=None, error="K0 requires a frozen source classifier; this arm requires target calibration"
                            if input_available else "requested budget absent; prepared fallback is not substituted")
                    index = len(cells)
                    cells.append(cell)
                    if input_available:
                        actual_cells.setdefault(budget, []).append(index)
                    requests.append({"study": study, "participant_key": participant, "requested_K": requested_budget,
                        "requested_Ks": [requested_budget],
                        "actual_K": budget if input_available else None, "arm": arm["name"],
                        "source_seed": source_seed, "target_seed": seed,
                        "budget_adjustment": adjustment, "cell_index": index,
                        "status": "requested" if available else "protocol_excluded"})
                if input_available:
                    del data, train, score, observed
            del entry, test
        del entries
    matched_controls = _bind_matched_controls(cells)
    omitted_contrasts = {}
    for name in list(contrasts):
        try:
            _validate_transfer_contrast(name, contrasts[name], cells)
        except ValueError as error:
            if "contrasts" in plan:
                raise
            omitted_contrasts[name] = str(error)
            del contrasts[name]
            contrast_plans.pop(name, None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "execution_source.json", {"schema": "n2p3_executing_python_files/1",
        "files": tree, "digest": semantic_sha256(tree), "verified_snapshot": freeze})
    report = {"schema": "n2p3_erp_comparator_result/3", "plan": plan, "plan_sha256": semantic_sha256(plan),
        "source_metadata": {"schema": "n2p3_complete_source_store/1", "directory": str(metadata_store.directory)},
        "authority_plan_sha256": authority_plan_sha256, "participant_authority": authority,
        "input_manifests": _prefix_bindings(plan, plan_path),
        "source_sha256": semantic_sha256(tree), "execution_snapshot": freeze, "evidence_scope": scope,
        "reuse": "deterministic_arm_predictions_with_bound_origin", "source_seeds": source_seeds, "target_seeds": seeds,
        "seed_pairs": [list(pair) for pair in pairs], "seed_pair_design": "paired" if paired else "crossed",
        "source_seed_catalog": None if catalog is None else catalog.record(), "physical_target_fits": 0,
        "budget_program": "owner_resolved_requests" if paired else "exact_prepared_budget",
        "requested_cells": len(requests), "actual_fit_score_cells": len(cells),
        "requested_participants": requests, "cells": cells}
    report["matched_controls"] = matched_controls
    report["omitted_unmatched_default_contrasts"] = omitted_contrasts
    _write_binary_json(output / "result.json", report)
    for subject in subjects:
        entries = _participant_entries(arms, plan_path, subject)
        deterministic_results = {}
        for arm in arms:
            entry = entries[arm["name"]]
            for budget in sorted({cell["K"] for cell in cells if cell["subject"] == subject and cell["arm"] == arm["name"]}):
                pending = [(i, c) for i, c in enumerate(cells) if c["subject"] == subject
                    and c["arm"] == arm["name"] and c["K"] == budget and c["status"] == "planned"]
                if not pending:
                    continue
                data, train, test, observed = _combined(entry, subject, budget, arm)
                if evaluation_input_identity(data, train, test) != pending[0][1]["comparison_identity"]:
                    raise ValueError("prepared input changed after preflight")
                for index, cell in pending:
                    deterministic_key = (arm["name"], budget)
                    if arm["family"] not in NN_FAMILIES and deterministic_key in deterministic_results:
                        origin_index = deterministic_results[deterministic_key]
                        origin = cells[origin_index]
                        cell.update({key: origin[key] for key in ("status", "auc", "fit_history", "saved_replay_exact",
                            "saved_replay_max_abs", "saved_replay_tolerance", "model_artifact", "prediction_artifact",
                            "model_state_digest", "target_fit_rows")})
                        cell.update({key: origin[key] for key in ("probability_status", "probability_semantics", "nll", "brier",
                            "probability_scored_rows", "probability_failure", "probability_auc") if key in origin})
                        cell.update(reused_prediction_cell=origin_index, physical_target_fit=False)
                        _write_binary_json(output / "result.json", report)
                        continue
                    try:
                        source, stats = None, None
                        device = plan.get("device", "cpu")
                        if arm["family"] in NN_FAMILIES:
                            prepared, source, _ = _nn_anchor(_source_snapshot(checkpoints, catalog_snapshots, arm,
                                cell["study"], cell["source_seed"]), data, subject, source_preparations[subject])
                            stats = prepared.input_statistics if arm["family"] == "full_fine" else None
                        if budget == 0:
                            predictor = FrozenTrialPredictor.from_source(prepared, subject)
                            history, tolerance = {"program": "source_only", "target_fit_rows": 0,
                                "physical_optimizer_updates": 0}, 1e-6
                            cell.update(physical_target_fit=False, physical_optimizer_updates=0)
                        else:
                            mask = np.zeros(data.n_epochs, dtype=bool)
                            mask[train] = True
                            report["physical_target_fits"] += 1
                            cell.update(physical_target_fit=None, physical_target_fit_state="invocation_pending")
                            _write_binary_json(output / "result.json", report)
                            with worker_compute_phase(device if arm["family"] in NN_FAMILIES else "cpu"):
                                fitted = fit_calibration_candidate(data, TargetRecipe(**cell["recipe"]), mask,
                                    source_trunk=source, source_input_stats=stats, device=device,
                                    source_temporal_basis=bases[arm["name"]])
                            predictor = FrozenTrialPredictor.from_fitted(fitted)
                            history, tolerance = fitted.fit_history, 1e-6 if fitted.adapter is not None else 0.
                            cell.update(physical_target_fit=True, physical_target_fit_state="returned", fit_history=history)
                            _write_binary_json(output / "result.json", report)
                        with worker_compute_phase(device if arm["family"] in NN_FAMILIES else "cpu"):
                            probability = (predictor.probability_scores(data.X[test], observed[test], device=device)
                                           if predictor.posterior_available else None)
                        if plan.get("require_probability_output") and probability is None:
                            raise ValueError("declared proper probability output requires its actual fitted posterior programme")
                        with worker_compute_phase(device if arm["family"] in NN_FAMILIES else "cpu"):
                            scores = probability["llr"] if probability is not None else predictor.predict_logit(
                                data.X[test], observed[test], device=device)
                        if not np.isfinite(scores).all() or set(np.unique(data.y[test])) != {0, 1}:
                            raise ValueError("primary AUC requires all requested test trials and both classes")
                        model_path, prediction_path = output / f"{index:05d}.pt", output / f"{index:05d}.npz"
                        predictor.save(model_path, metadata_store=metadata_store)
                        restored = FrozenTrialPredictor.load(model_path, metadata_store=metadata_store)
                        cell.update(model_artifact={"path": model_path.name,
                            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()},
                            model_state_digest=restored.digest(), fit_history=history)
                        _write_binary_json(output / "result.json", report)
                        with worker_compute_phase(device if arm["family"] in NN_FAMILIES else "cpu"):
                            probability_replay = (restored.probability_scores(data.X[test], observed[test], device=device)
                                                  if probability is not None else None)
                            replay = probability_replay["llr"] if probability is not None else restored.predict_logit(
                                data.X[test], observed[test], device=device)
                        np.testing.assert_allclose(scores, replay, atol=tolerance, rtol=tolerance)
                        if roc_auc_score(data.y[test], scores) != roc_auc_score(data.y[test], replay):
                            raise ValueError("saved replay changed scored ranking/AUC")
                        probability_payload = {}
                        if probability is not None:
                            for key in ("raw_logits", "llr", "posterior_logits", "probability"):
                                np.testing.assert_allclose(probability[key], probability_replay[key], atol=tolerance, rtol=tolerance)
                                probability_payload[key] = probability[key]
                            metrics = binary_probability_metrics(data.y[test], probability["posterior_logits"])
                            cell.update(probability_status="completed", probability_semantics=probability["semantics"], **metrics)
                            if predictor.source_contract_reference is not None:
                                cell["source_contract_reference"] = predictor.source_contract_reference
                        else:
                            cell.update(probability_status="unavailable", probability_failure="no_actual_posterior_programme")
                        np.savez_compressed(prediction_path, calibration_rows=train, test_rows=test, y=data.y[test],
                                            logits=scores, **probability_payload)
                        cell.update(status="completed", auc=float(roc_auc_score(data.y[test], scores)),
                            fit_history=history, saved_replay_exact=bool(np.array_equal(scores, replay)),
                            saved_replay_max_abs=float(np.max(np.abs(scores-replay))), saved_replay_tolerance=tolerance,
                            model_artifact={"path": model_path.name, "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()},
                            prediction_artifact={"path": prediction_path.name, "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()},
                            target_fit_rows=len(train), reused_prediction_cell=None)
                        if arm["family"] not in NN_FAMILIES:
                            deterministic_results[deterministic_key] = index
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, AssertionError) as error:
                        if cell.get("physical_target_fit_state") == "invocation_pending":
                            cell.update(physical_target_fit=True, physical_target_fit_state="raised")
                        cell.update(status="failed", auc=None, error=str(error))
                    _write_binary_json(output / "result.json", report)
                del data, train, test, observed
            del entry
        del entries, deterministic_results
    if not finalize:
        report["execution_source_status"] = "unchanged" if _executing_tree() == tree else "changed_during_execution"
        _write_binary_json(output / "result.json", report)
        if report["execution_source_status"] != "unchanged":
            raise RuntimeError("executing source files changed during comparator")
        atomic_write_json(output / "participant_terminal.json", {"schema": "n2p3_binary_participant_terminal/1",
            "result_sha256": hashlib.sha256((output / "result.json").read_bytes()).hexdigest(),
            "authority_plan_sha256": authority_plan_sha256, "source_sha256": semantic_sha256(tree),
            "subjects": subjects})
        return report
    return _finalize_comparison(report, plan, output, tree)


def _finalize_comparison(report, plan, output, tree):
    cells = report["cells"]
    budgets, arms = plan["budgets"], plan["arms"]
    requests = report["requested_participants"]
    matched_controls = report["matched_controls"]
    pairs = tuple(tuple(pair) for pair in report["seed_pairs"])
    source_seeds, seeds = report["source_seeds"], report["target_seeds"]
    scope = report["evidence_scope"]
    contrasts = dict(plan.get("contrasts", {f"{arm['name']}_minus_{arms[0]['name']}":
        {arm["name"]: 1., arms[0]["name"]: -1.} for arm in arms[1:]}))
    for name in report["omitted_unmatched_default_contrasts"]:
        contrasts.pop(name, None)
    contrast = ContrastPlan(tuple(plan["allowed_input_axes"]))
    contrast_plans = {name: contrast.record() for name in contrasts} | plan.get("contrast_plans", {})
    report["probability_pools"] = export_binary_pools(cells, plan.get("probability_pools", []), output)
    if _executing_tree() != tree:
        report["execution_source_status"] = "changed_during_execution"
        _write_binary_json(output / "result.json", report)
        raise RuntimeError("executing source files changed during comparator")
    report["execution_source_status"] = "unchanged"
    report["matched_transfer_effects"] = []
    for mapping in matched_controls:
        source = cells[mapping["source_cell_index"]]
        scratch = cells[mapping["controls"]["scratch"]["cell_index"]]
        complete = source["status"] == scratch["status"] == "completed"
        report["matched_transfer_effects"].append({**mapping,
            "status": "complete_pair" if complete else "incomplete_pair",
            "delta_auc": source["auc"] - scratch["auc"] if complete else None,
            "negative_transfer": source["auc"] < scratch["auc"] if complete else None})
    report["requested_budget_coverage"] = [{"requested_K": budget,
        "requested_cells": sum(row["requested_K"] == budget for row in requests),
        "completed_cells": sum(row["requested_K"] == budget and cells[row["cell_index"]]["status"] == "completed" for row in requests),
        "protocol_unavailable_cells": sum(row["requested_K"] == budget and cells[row["cell_index"]]["status"] == "protocol_unavailable" for row in requests),
        "failed_cells": sum(row["requested_K"] == budget and cells[row["cell_index"]]["status"] == "failed" for row in requests)} for budget in budgets]
    positive_budgets = [budget for budget in budgets if budget > 0]
    unresolved = [row for row in requests if row.get("participant_key") is None or row.get("study") is None]
    if unresolved:
        report["statistics_status"] = "unresolved_participant_authority"
        report["unresolved_statistical_requests"] = unresolved
        report["declared_contrasts"] = contrasts
    elif contrasts and positive_budgets:
        stats = analyze_transfer_contrasts([c for c in cells if c["K"] > 0 and c["comparison_identity"] is not None], contrasts,
            expected_studies=sorted({c["study"] for c in cells}), source_seeds=source_seeds, target_seeds=seeds,
            standard_budgets=positive_budgets, requested_participants=requests, evidence_scope=scope, seed_pairs=pairs,
            contrast_plans=contrast_plans, iterations=plan.get("bootstrap_iterations", 1000),
            seed=plan.get("analysis_seed", ANALYSIS_SEED))
        report["statistics"] = stats
        stats["matched_transfer_effects"] = report["matched_transfer_effects"]
        # Generic requested contrasts can still compare budget/algorithm
        # alternatives; only these explicitly bound controls identify transfer.
        stats["transfer_effect_interpretation"] = "only matched_transfer_effects identify source initialization effects"
        stats["requested_budget_coverage"] = report["requested_budget_coverage"]
        _write_binary_json(output / "statistics.json", stats)
    if "seed_pairs" in plan:
        report["b12_execution_binding"] = validate_b12_execution_manifest(report, plan)
    _write_binary_json(output / "result.json", report)
    return report


def comparison_participant_adapter(arguments):
    from contextlib import nullcontext

    from research.source_execution import current_input_admission

    plan_path, output = Path(arguments["plan"]), Path(arguments["output"])
    plan = json.loads(plan_path.read_bytes())
    if semantic_sha256(plan) != arguments["plan_sha256"]:
        raise ValueError("participant plan changed before execution")
    if semantic_sha256(_executing_tree()) != arguments["source_sha256"]:
        raise ValueError("participant source changed before execution")
    if _prefix_bindings(plan, plan_path) != arguments["input_manifests"]:
        raise ValueError("participant prefix declarations changed before execution")
    if _checkpoint_bindings(plan, plan_path) != arguments["checkpoint_manifests"]:
        raise ValueError("participant checkpoint bytes changed before execution")
    admission = current_input_admission()
    writer = (nullcontext() if admission is None else admission.artifact_directory(
        output, _participant_output_estimate(plan, plan_path, arguments["subject"])))
    with writer:
        _run_comparison_serial(plan_path, output, selected_subjects=[arguments["subject"]], finalize=False,
                               metadata_directory=arguments.get("metadata_directory"))
    return str(output / "result.json")


def _participant_output_estimate(plan, plan_path, subject):
    """Estimate named results from small input receipts, then settle actual bytes."""
    maximum_model, prediction_bytes = 0, 0
    for path in _prefix_bindings(plan, plan_path):
        collection = json.loads(Path(path).read_bytes())
        for entry in collection["records"]:
            records = entry["records"] if entry["protocol"]["kind"] == "last_session_holdout" else [entry["record"]]
            if str(records[0]["subject"]) != subject:
                continue
            views = entry["test"] if isinstance(entry["test"], list) else [entry["test"]]
            maximum_model = max(maximum_model, max(len(view["channels"]) for view in views) * collection["preprocessing"]["n_times"] * 64)
            prediction_bytes = max(prediction_bytes, sum(view["n_epochs"] for view in views) * 8 * 10)
    checkpoint_paths = set(_checkpoint_bindings(plan, plan_path))
    if "source_seed_catalog" in plan:
        checkpoint_paths.update(str(_resolve(plan_path, entry["path"])) for entry in _required_catalog_entries(plan, plan_path, [subject]))
    maximum_model = max([maximum_model, *(Path(path).stat().st_size for path in checkpoint_paths)])
    cells = len(plan["arms"]) * len(plan["budgets"]) * len(_declared_pairs(plan))
    return (1 << 20) + cells * ((1 << 16) + maximum_model + prediction_bytes)


def comparison_tasks(plan_path, output, resources, *, dependencies=(), task_prefix="binary", metadata_directory=None):
    from research.cloud_execution import CloudTask, target_phase_resources

    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    binding = {"plan_sha256": semantic_sha256(plan), "source_sha256": semantic_sha256(_executing_tree()),
               "input_manifests": _prefix_bindings(plan, plan_path), "checkpoint_manifests": _checkpoint_bindings(plan, plan_path)}
    return [CloudTask(f"{task_prefix}:{index:03d}",
        "experiments.run_erp_comparator:comparison_participant_adapter",
        {"plan": str(plan_path), "output": str(output / "participants" / f"participant_{index:03d}"), "subject": subject,
         "metadata_directory": str(metadata_directory or output / "source_metadata"), **binding},
        str(output / "participants" / f"participant_{index:03d}"),
        target_phase_resources(resources, plan.get("device", "cpu")), tuple(dependencies))
        for index, subject in enumerate(plan["subjects"])]


def _declared_pairs(plan):
    if "seed_pairs" in plan:
        return [list(pair) for pair in paired_pipeline_seeds(plan["seed_pairs"])]
    targets = plan.get("target_seeds", [0])
    if "source_seeds" in plan:
        sources = plan["source_seeds"]
    else:
        anchors = [arm for arm in plan["arms"] if arm["family"] in NN_FAMILIES]
        if not anchors:
            sources = [0]
        else:
            declared = anchors[0].get("checkpoints")
            sources = [int(seed) for seed in declared] if declared is not None else [None]
    return [[source, target] for source in sources for target in targets]


def _validate_declaration(plan, plan_path, *, require_source_binding=False):
    if plan.get("schema") != "n2p3_erp_comparator_plan/1":
        raise ValueError("unsupported ERP comparator plan schema")
    if plan.get("evidence_scope") not in {"public_development", "software_correctness"}:
        raise ValueError("plan requires explicit public_development or software_correctness evidence_scope")
    subjects, budgets, arms = plan["subjects"], plan["budgets"], plan["arms"]
    if not subjects or len(set(subjects)) != len(subjects) or any(not isinstance(v, str) or not v for v in subjects):
        raise ValueError("subjects must be unique explicit identifiers")
    if not budgets or len(set(budgets)) != len(budgets) or any(type(k) is not int or k < 0 for k in budgets):
        raise ValueError("comparison budgets must be unique nonnegative label budgets")
    if not arms or len({a["name"] for a in arms}) != len(arms) or any(a["family"] not in FAMILIES for a in arms):
        raise ValueError("comparator arms must be uniquely named supported families")
    if require_source_binding and plan["evidence_scope"] == "public_development":
        if not plan.get("source_snapshot"):
            raise ValueError("public comparator requires an executed source_snapshot")
        verified_executed_snapshot(_resolve(plan_path, plan["source_snapshot"]), root=ROOT)
    _declared_pairs(plan)


def _logical_requests(plan):
    return [{"subject": subject, "arm": arm["name"], "requested_K": budget,
             "source_seed": source, "target_seed": target, "actual_K": None,
             "participant_key": None, "study": None, "status": "execution_unresolved"}
            for subject in plan["subjects"] for arm in plan["arms"] for budget in plan["budgets"]
            for source, target in _declared_pairs(plan)]


def _artifact_path(directory, artifact, verified):
    path = (directory / artifact["path"]).resolve()
    if Path(artifact["path"]).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("binary artifact escapes its participant output directory")
    if path not in verified:
        verified[path] = file_sha256(path)
    if verified[path] != artifact["sha256"]:
        raise ValueError("binary artifact SHA differs from its actual scientific bytes")
    return path


def _admit_binary_participant(task, outcome, plan, tree, *, _metadata_admission=None):
    """Completed and partial artifacts cross the same scientific boundary."""
    directory = Path(task.output).resolve()
    path = directory / "result.json"
    if outcome.status == "completed" and Path(outcome.value).resolve() != path:
        raise ValueError("participant result path differs from its declared scientific output")
    if not path.exists():
        if outcome.status == "completed":
            raise ValueError("completed participant has no scientific result")
        return None
    from research.source_metadata_store import SourceMetadataStore, _SourceMetadataAdmission

    store = SourceMetadataStore(task.arguments.get("metadata_directory", directory / "source_metadata"))
    metadata_admission = _metadata_admission or _SourceMetadataAdmission(store=store)
    report = _load_binary_report(path, metadata_admission)
    if report["schema"] == "n2p3_erp_comparator_result/3" and report.get("source_metadata") != {
        "schema": "n2p3_complete_source_store/1", "directory": str(store.directory)
    }:
        raise ValueError("binary report source store differs from its actual declared package")
    subject = task.arguments["subject"]
    if (report["schema"] not in {"n2p3_erp_comparator_result/2", "n2p3_erp_comparator_result/3"}
            or report["plan"] != {**plan, "subjects": [subject]}
            or report["plan_sha256"] != semantic_sha256(report["plan"])
            or report["authority_plan_sha256"] != semantic_sha256(plan)):
        raise ValueError("participant plan changed the declared scientific programme or subject")
    execution = json.loads((directory / "execution_source.json").read_bytes())
    if (report["source_sha256"] != semantic_sha256(tree) or execution["files"] != tree
            or execution["digest"] != semantic_sha256(tree)
            or execution.get("verified_snapshot") != report["execution_snapshot"]):
        raise ValueError("participant executed source differs from the comparison execution source")
    if report["input_manifests"] != task.arguments["input_manifests"]:
        raise ValueError("participant prefix declarations differ from the admitted inputs")
    if _prefix_bindings(plan, Path(task.arguments["plan"])) != report["input_manifests"]:
        raise ValueError("participant prefix declarations changed during execution")
    if outcome.status == "completed":
        terminal = json.loads((directory / "participant_terminal.json").read_bytes())
        if (terminal["result_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
                or terminal["subjects"] != [subject] or terminal["source_sha256"] != report["source_sha256"]
                or terminal["authority_plan_sha256"] != report["authority_plan_sha256"]
                or report.get("execution_source_status") != "unchanged"):
            raise ValueError("participant terminal receipt differs from the scientific result")
    receipt = report["participant_authority"][subject]
    authority = ParticipantIdentityRecord.from_payload(receipt["identity_record"])
    if authority.local_subject_id != subject or authority.authority_key() != receipt["participant_key"]:
        raise ValueError("participant authority differs from its actual input identity record")
    if set(report["participant_authority"]) != {subject} or receipt["subject"] != subject:
        raise ValueError("participant result contains a different authority roster")
    arms = {arm["name"]: arm for arm in plan["arms"]}
    actual_pairs = [list(pair) for pair in report["seed_pairs"]]
    declared_pairs = _declared_pairs(plan)
    if any(source is None for source, _ in declared_pairs):
        sources = report["source_seeds"]
        if len(sources) != 1 or type(sources[0]) is not int:
            raise ValueError("participant unresolved checkpoint seed did not resolve to one actual source")
        declared_pairs = [[sources[0], target] for _, target in declared_pairs]
    if len(actual_pairs) != len(declared_pairs) or sorted(actual_pairs) != sorted(declared_pairs):
        raise ValueError("participant seed grid differs from the requested pipelines")
    if "seed_pairs" in plan:
        expected_catalog = {(entry["outer_target_study"], entry["source_seed"]): entry
                            for entry in plan["source_seed_catalog"] if entry["outer_target_study"] == receipt["study"]}
        actual_catalog = {(entry["outer_target_study"], entry["source_seed"]): entry for entry in report["source_seed_catalog"]}
        if set(actual_catalog) != {(receipt["study"], source) for source, _ in actual_pairs}:
            raise ValueError("binary source producer catalogue differs from its actual outer/seed scope")
        for key, entry in actual_catalog.items():
            expected = expected_catalog[key]
            if entry["checkpoint_sha256"] != expected["sha256"] or entry["procedure"] != expected["procedure"]:
                raise ValueError("binary source producer differs from its admitted checkpoint/procedure")
    requested_keys = []
    for request in report["requested_participants"]:
        index = request["cell_index"]
        if type(index) is not int or not 0 <= index < len(report["cells"]):
            raise ValueError("binary request references no actual cell")
        cell = report["cells"][index]
        for key in ("participant_key", "study", "actual_K", "arm", "source_seed", "target_seed"):
            if request[key] != cell[key]:
                raise ValueError("binary request differs from its resolved scientific cell")
        if request["requested_K"] not in cell["requested_Ks"]:
            raise ValueError("binary budget alias differs from its actual cell")
        requested_keys.append((request["arm"], request["requested_K"], request["source_seed"], request["target_seed"]))
    expected_keys = [(arm["name"], budget, source, target) for arm in plan["arms"] for budget in plan["budgets"]
                     for source, target in actual_pairs]
    if sorted(requested_keys) != sorted(expected_keys):
        raise ValueError("participant result changed the complete requested denominator")
    if report["requested_cells"] != len(expected_keys) or report["actual_fit_score_cells"] != len(report["cells"]):
        raise ValueError("participant result counts differ from its actual ledger")
    models, verified = {}, {}
    for index, cell in enumerate(report["cells"]):
        if (cell["subject"] != subject or cell["participant_key"] != receipt["participant_key"]
                or cell["study"] != receipt["study"] or cell["arm"] not in arms
                or cell["family"] != arms[cell["arm"]]["family"]):
            raise ValueError("binary cell differs from its unique expected participant/arm")
        recipe = _recipe(arms[cell["arm"]], cell["target_seed"]).record() if cell["K"] > 0 else None
        if cell["recipe"] != recipe:
            raise ValueError("binary cell changed its scientific fitting recipe")
        if cell.get("checkpoint_origin") is not None:
            if "seed_pairs" in plan:
                expected_digest = expected_catalog[(cell["study"], cell["source_seed"])]["sha256"]
            else:
                arm = arms[cell["arm"]]
                checkpoint = arm.get("checkpoint") or arm["checkpoints"][str(cell["source_seed"])]
                expected_digest = task.arguments["checkpoint_manifests"][str(Path(checkpoint).resolve())]
            if cell["checkpoint_origin"]["checkpoint_sha256"] != expected_digest:
                raise ValueError("binary cell source origin differs from its actual admitted checkpoint")
        if cell["comparison_identity"] is not None:
            validate_input_identity(cell["comparison_identity"])
            if not cell["comparison_identity"]["raw_bytes_bound"]:
                raise ValueError("binary result lost its physical raw-byte binding")
        for kind in ("model_artifact", "prediction_artifact"):
            if kind in cell:
                _artifact_path(directory, cell[kind], verified)
        if "model_artifact" in cell:
            model_path = _artifact_path(directory, cell["model_artifact"], verified)
            if model_path not in models:
                models[model_path] = FrozenTrialPredictor._load_with_admission(model_path, metadata_admission)
            predictor = models[model_path]
            if predictor.digest() != cell["model_state_digest"]:
                raise ValueError("binary model numeric state differs from its saved receipt")
            reference = predictor.source_contract_reference
            if reference is not None:
                from research.frozen_metadata import plain_metadata

                target = plain_metadata(reference["target_binding"])
                target_identity = DatasetIdentityTable.from_payload(target["identity"])
                if (semantic_sha256(cell.get("source_contract_reference")) != semantic_sha256(reference)
                        or target["subject"] != subject
                        or target_identity.record_for(subject).payload() != receipt["identity_record"]
                        or study_ids_for_identity(target_identity, target["study_membership"]) != (cell["study"],)
                        or reference["checkpoint_sha256"] != cell["checkpoint_origin"]["checkpoint_sha256"]
                        or predictor.record["class_calibration"]["fit_contract"]["training_contract"]["seed"] != cell["source_seed"]):
                    raise ValueError("binary source reference differs from its actual cell and target authority")
        if cell["status"] != "completed":
            continue
        if not {"model_artifact", "prediction_artifact"} <= cell.keys():
            raise ValueError("completed binary cell lacks saved model or prediction bytes")
        tolerance = 1e-6 if cell["family"] in NN_FAMILIES else 0.
        if (cell["saved_replay_tolerance"] != tolerance or not np.isfinite(cell["saved_replay_max_abs"])
                or cell["saved_replay_max_abs"] < 0 or cell["saved_replay_exact"] != (cell["saved_replay_max_abs"] == 0)):
            raise ValueError("binary saved replay receipt differs from its scientific programme")
        prediction_path = _artifact_path(directory, cell["prediction_artifact"], verified)
        with np.load(prediction_path, allow_pickle=False) as saved:
            identity = cell["comparison_identity"]
            if cell["saved_replay_max_abs"] > tolerance * (1 + np.max(np.abs(saved["logits"]))) / (1 - tolerance):
                raise ValueError("binary saved replay error exceeds its declared scoring tolerance")
            if any(cell[key] != digest for key, digest in matched_split_digests(saved["calibration_rows"], saved["test_rows"]).items()):
                raise ValueError("binary artifact rows differ from its scored input")
            if (len(saved["y"]) != identity["score"]["n_rows"] or not np.isfinite(saved["logits"]).all()
                    or compact_array_digest(saved["y"]) != cell["test_labels_sha256"]
                    or float(roc_auc_score(saved["y"], saved["logits"])) != cell["auc"]):
                raise ValueError("binary artifact labels/scores differ from the recorded result")
            if cell["probability_status"] == "completed":
                from scipy.special import expit

                from transfer.class_calibration import semantic_trial_logits

                if semantic_sha256(cell["probability_semantics"]) != semantic_sha256(predictor.record["class_calibration"]):
                    raise ValueError("binary probability origin differs from its fitted numeric model")
                llr, posterior = semantic_trial_logits(saved["raw_logits"], cell["probability_semantics"])
                np.testing.assert_allclose(saved["llr"], llr, atol=0, rtol=0)
                np.testing.assert_allclose(saved["posterior_logits"], posterior, atol=0, rtol=0)
                np.testing.assert_allclose(saved["probability"], expit(posterior), atol=0, rtol=0)
                np.testing.assert_array_equal(saved["logits"], saved["llr"])
                if any(cell[key] != value for key, value in binary_probability_metrics(saved["y"], posterior).items()):
                    raise ValueError("binary probability metrics differ from the complete saved scores")
            elif plan.get("require_probability_output"):
                raise ValueError("binary cell omitted its declared proper probability output")
        alias = cell.get("reused_prediction_cell")
        if alias is not None:
            if type(alias) is not int or not 0 <= alias < index:
                raise ValueError("binary deterministic alias references no earlier physical fit")
            origin = report["cells"][alias]
            if (cell["family"] in NN_FAMILIES
                    or any(cell[key] != origin[key] for key in ("arm", "K", "comparison_identity", "model_artifact", "prediction_artifact"))
                    or cell["physical_target_fit"] is not False):
                raise ValueError("binary deterministic alias changed its bound physical fit")
    return report


class BinaryCloudProgramme:
    """Scientific adapter hooks for a caller-owned shared CloudExecutor."""

    def __init__(self, plan_path, output, resources, *, dependencies=(), task_prefix="binary", input_admission=None,
                 metadata_directory=None):
        self.original_plan_path = Path(plan_path).resolve()
        self.plan = json.loads(self.original_plan_path.read_bytes())
        _validate_declaration(self.plan, self.original_plan_path)
        self.output, self.resources = Path(output).resolve(), resources
        self.metadata_directory = Path(metadata_directory or self.output / "source_metadata").resolve()
        self.dependencies, self.task_prefix = tuple(dependencies), task_prefix
        self.input_admission = input_admission
        self.tree, self.admitted, self.admission_failures = _executing_tree(), {}, {}
        self.admitted_files = {}
        from research.source_metadata_store import SourceMetadataStore, _SourceMetadataAdmission

        self._metadata_admission = _SourceMetadataAdmission(store=SourceMetadataStore(self.metadata_directory))
        # The execution host compiles its own locators; no platform aliases.
        for arm in self.plan["arms"]:
            for key in ("prefix_collection", "checkpoint", "source_temporal_basis"):
                if key in arm:
                    arm[key] = str(_resolve(self.original_plan_path, arm[key]).resolve())
            if "checkpoints" in arm:
                arm["checkpoints"] = {seed: str(_resolve(self.original_plan_path, path).resolve())
                                       for seed, path in arm["checkpoints"].items()}
        if self.plan.get("source_snapshot"):
            self.plan["source_snapshot"] = str(_resolve(self.original_plan_path, self.plan["source_snapshot"]).resolve())
        for entry in self.plan.get("source_seed_catalog", []):
            entry["path"] = str(_resolve(self.original_plan_path, entry["path"]).resolve())
        self.output.mkdir(parents=True, exist_ok=False)
        self.plan_path = self.output / "consumed_plan.json"
        self.requests = _logical_requests(self.plan)
        self._write_json("requested.json", {"schema": "n2p3_binary_cloud_requests/1",
            "authority_plan_sha256": semantic_sha256(self.plan), "requests": self.requests,
            "source_binding_status": "declared" if self.plan.get("source_seed_catalog") else "unresolved"})
        self._write_json("execution_source.json", {"schema": "n2p3_executing_python_files/1",
            "files": self.tree, "digest": semantic_sha256(self.tree)})
        self._write_plan()

    def _write_json(self, name, payload):
        return atomic_write_json(self.output / name, payload, input_admission=self.input_admission)

    def _write_plan(self):
        self._write_json(self.plan_path.name, self.plan)
        self.tasks = comparison_tasks(self.plan_path, self.output, self.resources,
                                      dependencies=self.dependencies, task_prefix=self.task_prefix,
                                      metadata_directory=self.metadata_directory)
        self.by_id = {task.task_id: task for task in self.tasks}

    def bind_source(self, catalog, snapshot):
        if self.admitted or self.admission_failures:
            raise ValueError("binary source binding must precede participant execution")
        if any(not Path(entry["path"]).is_absolute() for entry in catalog):
            raise ValueError("shared binary source catalog needs execution-host absolute checkpoint paths")
        studies = _participant_studies(self.plan, self.plan_path, self.plan["subjects"])
        keys = [(entry["outer_target_study"], entry["source_seed"]) for entry in catalog]
        expected = {(study, seed) for study in studies for seed, _ in _declared_pairs(self.plan)}
        if not expected or len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("binary source catalog must exactly cover the declared outer-study/source-seed grid")
        self.plan["source_seed_catalog"], self.plan["source_snapshot"] = catalog, str(Path(snapshot).resolve())
        self._write_plan()
        self._write_json("source_binding.json", {"schema": "n2p3_binary_source_binding/1",
            "plan_sha256": semantic_sha256(self.plan), "catalog": catalog, "source_snapshot": self.plan["source_snapshot"]})
        return self.tasks

    def on_finished(self, outcome):
        task = self.by_id[outcome.task_id]
        try:
            report = _admit_binary_participant(task, outcome, self.plan, self.tree,
                                              _metadata_admission=self._metadata_admission)
        except Exception as error:
            self.admission_failures[outcome.task_id] = {"type": type(error).__name__, "message": str(error)}
            raise
        if report is not None:
            self.admitted[outcome.task_id] = report
            self.admitted_files[str(Path(task.output) / "result.json")] = file_sha256(Path(task.output) / "result.json")
            for cell in report["cells"]:
                for kind in ("model_artifact", "prediction_artifact"):
                    if kind in cell:
                        self.admitted_files[str(Path(task.output) / cell[kind]["path"])] = cell[kind]["sha256"]
                reference = cell.get("source_contract_reference")
                if reference is not None:
                    source_path = Path(task.arguments["metadata_directory"]) / (reference["checkpoint_sha256"] + ".pt")
                    self.admitted_files[str(source_path)] = reference["checkpoint_sha256"]
        self._write_json("participant_admission.json", {"schema": "n2p3_binary_admission/1",
            "accepted": {key: hashlib.sha256((Path(self.by_id[key].output) / "result.json").read_bytes()).hexdigest()
                         for key in self.admitted}, "rejected": self.admission_failures})

    def finalize(self, results, *, input_admission=None):
        import copy

        for path, digest in self.admitted_files.items():
            if file_sha256(Path(path)) != digest:
                raise ValueError("admitted binary scientific bytes changed before programme finalization")

        outcomes = {task.task_id: results[task.task_id] for task in self.tasks}
        pairs = _declared_pairs(self.plan)
        resolved = next(iter(self.admitted.values()), None)
        if resolved is not None:
            pairs = resolved["seed_pairs"]
        report = {"schema": "n2p3_erp_comparator_result/3", "plan": self.plan,
            "source_metadata": {"schema": "n2p3_complete_source_store/1", "directory": str(
                getattr(self, "metadata_directory", self.output / "source_metadata"))},
            "plan_sha256": semantic_sha256(self.plan), "source_sha256": semantic_sha256(self.tree),
            "authority_plan_sha256": semantic_sha256(self.plan),
            "execution_snapshot": None if resolved is None else resolved["execution_snapshot"],
            "evidence_scope": self.plan["evidence_scope"], "seed_pairs": pairs,
            "source_seeds": list(dict.fromkeys(source for source, _ in pairs)),
            "target_seeds": list(dict.fromkeys(target for _, target in pairs)),
            "seed_pair_design": "paired" if "seed_pairs" in self.plan else "crossed",
            "declared_pipeline_replicates": len(pairs) if "seed_pairs" in self.plan else None,
            "budget_program": "owner_resolved_requests" if "seed_pairs" in self.plan else "exact_prepared_budget",
            "reuse": "deterministic_arm_predictions_with_bound_origin", "cells": [], "requested_participants": [],
            "participant_authority": {}, "source_seed_catalog": [], "matched_controls": [],
            "input_manifests": _prefix_bindings(self.plan, self.plan_path),
            "omitted_unmatched_default_contrasts": {}, "execution_failures": [], "physical_target_fits_lower_bound": 0,
            "physical_target_fits_unknown": [], "actual_fit_score_cells": 0}
        catalog = {}
        for task in self.tasks:
            outcome = outcomes[task.task_id]
            participant = self.admitted.get(task.task_id)
            if outcome.status != "completed":
                report["execution_failures"].append({"task_id": task.task_id, "status": outcome.status,
                    "failure": outcome.failure, "supplementary_failures": outcome.supplementary_failures,
                    "scientific_admission_failure": self.admission_failures.get(task.task_id)})
            offset = len(report["cells"])
            if participant is None:
                started = outcome.worker is not None
                if started:
                    report["physical_target_fits_unknown"].append({"task_id": task.task_id, "scope": "participant"})
                for request in self.requests:
                    if request["subject"] != task.arguments["subject"]:
                        continue
                    index = len(report["cells"])
                    arm = next(arm for arm in self.plan["arms"] if arm["name"] == request["arm"])
                    report["cells"].append({**request, "K": request["requested_K"], "requested_Ks": [request["requested_K"]],
                        "family": arm["family"], "comparison_identity": None, "status": "failed", "auc": None,
                        "physical_target_fit": None if started else False, "fit_observation": "unknown" if started else "not_started",
                        "error": "participant scientific output was not admitted"})
                    report["requested_participants"].append({**request, "cell_index": index, "status": "execution_unreachable"})
                continue
            cells = copy.deepcopy(participant["cells"])
            for local_index, cell in enumerate(cells):
                if cell["status"] == "planned":
                    if cell.get("physical_target_fit_state") == "invocation_pending":
                        report["physical_target_fits_unknown"].append({"task_id": task.task_id, "cell_index": offset + local_index})
                    elif "physical_target_fit" not in cell:
                        cell.update(physical_target_fit=False, fit_observation="not_started")
                    cell.update(status="failed", auc=None, error="participant execution did not complete")
                alias = cell.get("reused_prediction_cell")
                if alias is not None:
                    cell["reused_prediction_cell"] = alias + offset
                for kind in ("model_artifact", "prediction_artifact"):
                    if kind in cell:
                        cell[kind]["path"] = str((Path(task.output) / cell[kind]["path"]).resolve().relative_to(self.output))
            requests = [{**request, "subject": task.arguments["subject"], "cell_index": request["cell_index"] + offset}
                        for request in participant["requested_participants"]]
            report["cells"].extend(cells)
            report["requested_participants"].extend(requests)
            report["actual_fit_score_cells"] += len(cells)
            report["physical_target_fits_lower_bound"] += sum(cell.get("physical_target_fit") is True for cell in cells)
            report["participant_authority"].update(participant["participant_authority"])
            for entry in participant.get("source_seed_catalog") or []:
                key = (entry["outer_target_study"], entry["source_seed"])
                if key in catalog and semantic_sha256(catalog[key]) != semantic_sha256(entry):
                    raise ValueError("binary participants name conflicting source producers for one study/seed")
                catalog[key] = entry
        report["source_seed_catalog"] = list(catalog.values()) or None
        report["physical_target_fits"] = (None if report["physical_target_fits_unknown"] else report["physical_target_fits_lower_bound"])
        report["requested_cells"] = len(report["requested_participants"])
        report["status"] = "failed" if report["execution_failures"] else "completed"
        report["matched_controls"] = _bind_matched_controls(report["cells"])
        arms = self.plan["arms"]
        contrasts = self.plan.get("contrasts", {f"{arm['name']}_minus_{arms[0]['name']}":
            {arm["name"]: 1., arms[0]["name"]: -1.} for arm in arms[1:]})
        for name, weights in contrasts.items():
            try:
                _validate_transfer_contrast(name, weights, report["cells"])
            except ValueError as error:
                if "contrasts" in self.plan:
                    raise
                report["omitted_unmatched_default_contrasts"][name] = str(error)
        input_admission = input_admission or self.input_admission
        if input_admission is None:
            return _finalize_comparison(report, self.plan, self.output, self.tree)
        estimate = (1 << 20) + sum(Path(path).stat().st_size for path in self.admitted_files if path.endswith(".npz")) * max(
            1, len(self.plan.get("probability_pools", [])))
        from research.source_execution import current_input_admission, install_input_admission

        previous = current_input_admission()
        install_input_admission(input_admission)
        try:
            with input_admission.artifact_directory(self.output, estimate):
                return _finalize_comparison(report, self.plan, self.output, self.tree)
        finally:
            install_input_admission(previous)


def build_binary_cloud_tasks(plan_path, output, resources, *, source_seed_catalog=None, source_snapshot=None,
                             dependencies=(), task_prefix="binary", input_admission=None, metadata_directory=None):
    programme = BinaryCloudProgramme(plan_path, output, resources, dependencies=dependencies,
                                    task_prefix=task_prefix, input_admission=input_admission,
                                    metadata_directory=metadata_directory)
    if source_seed_catalog is not None:
        programme.bind_source(source_seed_catalog, source_snapshot)
    return programme


def run_comparison(plan_path: Path, output: Path):
    from multiprocessing import get_context

    from research.cloud_execution import CloudExecutor
    from research.source_execution import (
        HostMemoryAdmission,
        SourceInputAdmission,
        resolve_source_resources,
        with_cloud_resource_floor,
    )
    from research.source_metadata_store import compact_source_semantics

    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    _validate_declaration(plan, plan_path, require_source_binding=True)
    resource_plan = {"source_worker_memory_gib": plan.get("target_worker_memory_gib", .25 if plan["evidence_scope"] == "software_correctness" else 4.),
                     "target_worker_memory_gib": .25 if plan["evidence_scope"] == "software_correctness" else 4., **plan}
    resources = resolve_source_resources(with_cloud_resource_floor(resource_plan, plan.get("device", "cpu")),
                                         len(plan["subjects"]), plan.get("device", "cpu"))
    programme = build_binary_cloud_tasks(plan_path, output, resources)
    with get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"],
                                      load_workers=plan.get("source_load_workers", resources["source_workers"]))
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        run = CloudExecutor(admission, max_workers=resources["target_workers"], diagnostic_directory=output / "worker_diagnostics",
                            identity={"actual_source_root": str(ROOT), "plan_sha256": semantic_sha256(plan)}).run(
                                programme.tasks, on_finished=programme.on_finished)
        report = programme.finalize(run.results, input_admission=inputs)
        report["execution"] = {"status": run.status, "resources": resources, "input_admission": inputs.record(),
                               "diagnostics": run.diagnostics, "observation_failures": run.observation_failures}
        atomic_write_json(output / "result.json", compact_source_semantics(report), input_admission=inputs)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_comparison(args.plan, args.output)


if __name__ == "__main__":
    main()
