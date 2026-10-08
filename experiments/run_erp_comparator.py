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
from data.epochs import concatenate_epoch_datasets, select_epoch_channels  # noqa: E402
from data.identity import DatasetIdentityTable  # noqa: E402
from data.source_records import load_target_prefixes, prepared_calibration_requests  # noqa: E402
from data.study import (  # noqa: E402
    checkpoint_study_membership,
    dataset_study_membership,
    study_ids_for_identity,
)
from models.n2p3net import N2P3Net  # noqa: E402
from research.comparison import ContrastPlan, evaluation_input_identity  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.source_catalog import SourceSeedCatalog, paired_pipeline_seeds  # noqa: E402
from stats.transfer import analyze_transfer_contrasts, resolve_seed_pairs  # noqa: E402
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
    load_checkpoint_payload,
)
from transfer.recipes import TargetRecipe, erp_target_recipe  # noqa: E402

NN_FAMILIES = frozenset({"full_fine", "scratch"})
FAMILIES = ERP_FAMILIES | NN_FAMILIES | {"waveform_lda"}


def _executing_tree():
    paths = sorted([*ROOT.joinpath("src").rglob("*.py"), Path(__file__)])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


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
            lr=arm.get("lr", .001), weight_decay=arm.get("weight_decay", .0001))
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


def _source_snapshot(checkpoints, catalog_snapshots, arm, study, source_seed):
    if catalog_snapshots is None:
        return checkpoints[arm["name"]][source_seed]
    return catalog_snapshots[(study, source_seed)]


def _nn_anchor(snapshot, data, subject):
    content, digest = snapshot
    prepared = PreparedTransferCheckpoint(io.BytesIO(content), data)
    training = DatasetIdentityTable.from_payload(prepared.payload["training_identity_ledger"])
    if set(study_ids_for_identity(training, checkpoint_study_membership(prepared.payload))) & set(
        study_ids_for_identity(data.identity_table, dataset_study_membership(data))
    ):
        raise ValueError("NN anchor source must exclude the complete target study")
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


def run_comparison(plan_path: Path, output: Path):
    plan = json.loads(Path(plan_path).read_bytes())
    if plan.get("schema") != "n2p3_erp_comparator_plan/1":
        raise ValueError("unsupported ERP comparator plan schema")
    scope = plan.get("evidence_scope")
    if scope not in {"public_development", "software_correctness"}:
        raise ValueError("plan requires explicit public_development or software_correctness evidence_scope")
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
        catalog = SourceSeedCatalog(plan["source_seed_catalog"], base_directory=plan_path.parent,
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
    cells, requests, identities = [], [], {}
    # Keep only small preflight receipts. EEG memory is bounded to one arm and
    # participant; independent prefix loader owns per-K raw support validation.
    for subject in subjects:
        entries = _participant_entries(arms, plan_path, subject)
        for arm in arms:
            entry = entries[arm["name"]]
            test = entry["test"]
            study, = study_ids_for_identity(test.identity_table, dataset_study_membership(test))
            participant = test.identity_table.record_for(subject).authority_key()
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
                        prepared, trunk, origin = _nn_anchor(_source_snapshot(checkpoints, catalog_snapshots, arm, study, source_seed), data, subject)
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
    report = {"schema": "n2p3_erp_comparator_result/2", "plan": plan, "plan_sha256": semantic_sha256(plan),
        "source_sha256": semantic_sha256(tree), "execution_snapshot": freeze, "evidence_scope": scope,
        "reuse": "deterministic_arm_predictions_with_bound_origin", "source_seeds": source_seeds, "target_seeds": seeds,
        "seed_pairs": [list(pair) for pair in pairs], "seed_pair_design": "paired" if paired else "crossed",
        "source_seed_catalog": None if catalog is None else catalog.record(), "physical_target_fits": 0,
        "budget_program": "owner_resolved_requests" if paired else "exact_prepared_budget",
        "requested_cells": len(requests), "actual_fit_score_cells": len(cells),
        "requested_participants": requests, "cells": cells}
    report["matched_controls"] = matched_controls
    report["omitted_unmatched_default_contrasts"] = omitted_contrasts
    atomic_write_json(output / "result.json", report)
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
                            "target_fit_rows")})
                        cell.update(reused_prediction_cell=origin_index, physical_target_fit=False)
                        atomic_write_json(output / "result.json", report)
                        continue
                    try:
                        source, stats = None, None
                        device = plan.get("device", "cpu")
                        if arm["family"] in NN_FAMILIES:
                            prepared, source, _ = _nn_anchor(_source_snapshot(checkpoints, catalog_snapshots, arm,
                                cell["study"], cell["source_seed"]), data, subject)
                            stats = prepared.input_statistics if arm["family"] == "full_fine" else None
                        if budget == 0:
                            predictor = FrozenTrialPredictor.from_source(prepared, subject)
                            scores = predictor.predict_logit(data.X[test], observed[test], device=device)
                            history, tolerance = {"program": "source_only", "target_fit_rows": 0,
                                "physical_optimizer_updates": 0}, 1e-6
                            cell.update(physical_target_fit=False, physical_optimizer_updates=0)
                        else:
                            mask = np.zeros(data.n_epochs, dtype=bool)
                            mask[train] = True
                            report["physical_target_fits"] += 1
                            fitted = fit_calibration_candidate(data, TargetRecipe(**cell["recipe"]), mask,
                                source_trunk=source, source_input_stats=stats, device=device,
                                source_temporal_basis=bases[arm["name"]])
                            scores = fitted.predict_logit(data.X[test], trial_channel_mask=observed[test])
                            predictor = FrozenTrialPredictor.from_fitted(fitted)
                            history, tolerance = fitted.fit_history, 1e-6 if fitted.adapter is not None else 0.
                            cell["physical_target_fit"] = True
                        if not np.isfinite(scores).all() or set(np.unique(data.y[test])) != {0, 1}:
                            raise ValueError("primary AUC requires all requested test trials and both classes")
                        model_path, prediction_path = output / f"{index:05d}.pt", output / f"{index:05d}.npz"
                        replay = _saved_replay(model_path, predictor, data, test, observed, device)
                        np.testing.assert_allclose(scores, replay, atol=tolerance, rtol=tolerance)
                        if roc_auc_score(data.y[test], scores) != roc_auc_score(data.y[test], replay):
                            raise ValueError("saved replay changed scored ranking/AUC")
                        np.savez_compressed(prediction_path, calibration_rows=train, test_rows=test, y=data.y[test], logits=scores)
                        cell.update(status="completed", auc=float(roc_auc_score(data.y[test], scores)),
                            fit_history=history, saved_replay_exact=bool(np.array_equal(scores, replay)),
                            saved_replay_max_abs=float(np.max(np.abs(scores-replay))), saved_replay_tolerance=tolerance,
                            model_artifact={"path": model_path.name, "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()},
                            prediction_artifact={"path": prediction_path.name, "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()},
                            target_fit_rows=len(train), reused_prediction_cell=None)
                        if arm["family"] not in NN_FAMILIES:
                            deterministic_results[deterministic_key] = index
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, AssertionError) as error:
                        cell.update(status="failed", auc=None, error=str(error))
                    atomic_write_json(output / "result.json", report)
                del data, train, test, observed
            del entry
        del entries, deterministic_results
    if _executing_tree() != tree:
        report["execution_source_status"] = "changed_during_execution"
        atomic_write_json(output / "result.json", report)
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
    if contrasts and positive_budgets:
        stats = analyze_transfer_contrasts([c for c in cells if c["K"] > 0 and c["comparison_identity"] is not None], contrasts,
            expected_studies=sorted({c["study"] for c in cells}), source_seeds=source_seeds, target_seeds=seeds,
            standard_budgets=positive_budgets, requested_participants=requests, evidence_scope=scope, seed_pairs=pairs,
            contrast_plans=contrast_plans, iterations=plan.get("bootstrap_iterations", 1000))
        report["statistics"] = stats
        stats["matched_transfer_effects"] = report["matched_transfer_effects"]
        # Generic requested contrasts can still compare budget/algorithm
        # alternatives; only these explicitly bound controls identify transfer.
        stats["transfer_effect_interpretation"] = "only matched_transfer_effects identify source initialization effects"
        stats["requested_budget_coverage"] = report["requested_budget_coverage"]
        atomic_write_json(output / "statistics.json", stats)
    atomic_write_json(output / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_comparison(args.plan, args.output)


if __name__ == "__main__":
    main()
