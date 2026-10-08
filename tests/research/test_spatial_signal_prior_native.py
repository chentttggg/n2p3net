"""Explicit same-C native mechanism panels through the real spatial CLI.

Normal pytest collection never fits EEG. Execution requires an actual root-owned
source freeze plus the predeclared plan, and records every one of its six cells.
The sole baseline noise fit reads the first two authorized complete groups.
Each cell consumes that same frozen native baseline through the existing CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from experiments import audit_spatial_signal_prior as audit  # noqa: E402
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity  # noqa: E402
from research.candidate_metrics import (  # noqa: E402
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import DecisionPlanContract, DecisionPlanEntry  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import UnidentifiableFitError, atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel  # noqa: E402
from transfer.ensemble_views import prepare_bnci_calibration_groups  # noqa: E402
from transfer.noise_model import array_digest, metadata_digest  # noqa: E402
from transfer.outcomes import CandidateCoverage, DecisionKey  # noqa: E402
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec  # noqa: E402
from transfer.spatial_signal_prior import SpatialSignalPosterior  # noqa: E402

ENTRY_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_plan(path, subject=None):
    payload = path.read_bytes()
    document = json.loads(payload)
    matrix = document["schema"] == "n2p3_B09_spatial_attribution_matrix/1"
    if matrix:
        inputs = document["inputs"]
        if ([row["subject"] for row in inputs] != [f"A{i:02d}" for i in range(1, 9)]
                or document["requested_grid"]["cells"] != 48 or subject is None):
            raise ValueError("attribution matrix requires the complete predeclared eight-participant roster")
        selected = next((row for row in inputs if row["subject"] == subject), None)
        if selected is None:
            raise ValueError("participant is outside the complete requested matrix")
        plan = {**document["protocol"], "schema": document["schema"],
                "input": {**document["input_contract"], **selected}}
        if subject == document["reuse"]["participant"]:
            plan["reuse"] = document["reuse"]
    else:
        plan = document
    observations = ["native", "cz"] if matrix else ["native", "cz", "car"]
    priors = ["noise_tied", "matched_diagonal_I", "predeclared_Pz_ROI"] if matrix else ["noise_tied", "predeclared_Pz_ROI"]
    support, baseline, arms = plan["support"], plan["baseline"], plan["arms"]
    if (plan["schema"] not in {"n2p3_B09_native_mechanism_plan/1", "n2p3_B09_spatial_attribution_matrix/1"}
            or support["authorized_labels_request"] != 800 or support["first_complete_fit_groups"] != 2
            or support["held_complete_group_ordinal"] != 3 or support["conditioning_samples"] != 32
            or baseline["noise_order"] != 0 or baseline["noise_estimator"] != "conditional_profile"
            or baseline["ridge"] != 1. or baseline["dictionary"] != {
                "family": "gaussian", "response_seconds": 1.2, "n_basis": 8, "endpoint": "free"}
            or arms["observations"] != observations
            or arms["signal_priors"] != priors
            or arms["predeclared_Pz_ROI"]["Pz_variance_ratio"] != 4. or arms["temperature"] != 1.
            or plan["resources"]["cpu_threads"] != 1 or arms["requested_cells"] != 6):
        raise ValueError("native B09 panel requires its complete agreed predeclared protocol")
    return plan, hashlib.sha256(payload).hexdigest()


def _files(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def _evaluation(prediction, view, truth, *, cell, model_sha256, fit_sha256, noise_sha256, failure=None,
                failure_kind="fit_failure"):
    vocabulary = view.decision.candidate_vocabulary
    counts = tuple(map(int, view.decision.membership.sum(axis=0)))
    identity = CandidateExpertIdentity(cell, view.raw_input_digest, model_sha256, noise_sha256,
        fit_sha256, "frozen_noise_general_Gaussian_conditional_tail", False)
    evidence = CandidateEvidence(vocabulary, None if failure is not None else np.asarray(prediction["log_probability"]),
        view.participant_key, view.decision_id, view.physical_support_digest, (identity,),
        view.decision.repetitions, counts, (0,) * len(counts) if failure is not None else counts,
        failure_reason=failure, failure_kind=failure_kind if failure is not None else None)
    decision_plan = DecisionPlanContract(view.source_sha256,
        metadata_digest({"participant": view.participant_key, "physical_support": view.physical_support_digest}),
        (view.participant_key,), (DecisionPlanEntry(view.participant_key, view.decision_id, truth),))
    coverage = CandidateCoverage.from_mappings(dict(zip(vocabulary, counts, strict=True)), {})
    metrics = evaluate_candidate_evidence({20260918: (evidence,)}, decision_plan=decision_plan,
        seeds=(20260918,), candidate_vocabulary=vocabulary, evidence_level=view.decision.repetitions,
        requested_coverage={DecisionKey(view.participant_key, view.decision_id): coverage})
    return metrics, evidence


def _spatial_energy(observation, signal, noise):
    mapping = observation.normalized_support_basis.T @ observation.normalized_matrix
    projected_noise, projected_signal = mapping @ noise @ mapping.T, mapping @ signal @ mapping.T
    return float(np.trace(np.linalg.solve(projected_noise, projected_signal)))


def _admit_original_cell(reuse, cell, state, view, signal, noise, baseline, dictionary):
    """Reuse an already scored A01 cell only after actual data/state relationships."""
    original_root = Path(reuse["cell_results_path"])
    panel_path = original_root / "panel_receipt.json"
    acceptance_path = Path(reuse["result_acceptance_path"])
    if (hashlib.sha256(panel_path.read_bytes()).hexdigest() != reuse["cell_panel_receipt_sha256"]
            or hashlib.sha256(acceptance_path.read_bytes()).hexdigest() != reuse["result_acceptance_sha256"]):
        raise ValueError("original completed A01 result differs from its predeclared reuse authority")
    original = json.loads(panel_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (acceptance["status"] != "passed" or original["status"] != "completed"
            or original["actual_source_archive_sha256"] != reuse["actual_original_spatial_source_archive_sha256"]
            or original["actual_calibration"] != state["actual_calibration"]
            or original["held_view"] != view.record()
            or original["baseline_model_digest"] != baseline.digest()):
        raise ValueError("original result source, same-C fit or actual held support differs")
    old_cell = next(row for row in original["cells"] if row["cell"] == cell["cell"])
    directory = original_root / cell["cell"]
    for name, digest in old_cell["saved_package_files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("original scored spatial package bytes changed")
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    serialized = json.loads((directory / "spatial_model.json").read_text(encoding="utf-8"))
    model = SpatialSignalPosterior.load(directory / "spatial_model.json",
                                       expected_artifact_sha256=serialized["artifact_sha256"])
    if (receipt["status"] != "completed" or not receipt["exact_saved_replay"]
            or receipt["prediction"]["input_sha256"] != view.raw_input_digest
            or model.prior.design_binding["dictionary_sha256"] != dictionary.digest()
            or not np.array_equal(model.prior.mean, np.zeros((16, 8)))
            or not np.array_equal(model.prior.temporal_covariance, np.eye(16))
            or not np.array_equal(model.prior.signal_covariance, signal)
            or not np.array_equal(model.native_noise_covariance, noise)):
        raise ValueError("original spatial state or numerical held input differs from the new declared cell")
    expected_observation = audit._observation(audit._layout_from_view(view), cell["observation"], "Cz")
    if receipt["observation"] != expected_observation.record():
        raise ValueError("original reference projection differs from its actual declared observation")
    producer = {"actual_source_archive_sha256": original["actual_source_archive_sha256"],
        "actual_source_manifest_sha256": original["snapshot_manifest_sha256"],
        "actual_original_plan_sha256": original["plan_sha256"],
        "model_artifact_sha256": serialized["artifact_sha256"],
        "actual_score_input_sha256": view.raw_input_digest,
        "actual_prior_authority_sha256": model.prior.source["authority_sha256"],
        "fit_or_score_repeated": False, "old_artifact_resigned": False,
        "original_baseline_equivalence": {key: value for key, value in old_cell.items()
                                           if key.startswith("baseline_")}}
    return directory, receipt, producer


def _baseline_equivalence(cell, spatial, baseline, score, receipt, plan):
    """Compare a newly fitted native MN special case to the existing baseline."""
    cell["baseline_posterior_mean_max_abs_error"] = float(np.max(np.abs(spatial.mean - baseline.kernel_mean)))
    covariance = np.kron(baseline.innovation_covariance, baseline.kernel_row_covariance)
    cell["baseline_posterior_covariance_relative_error"] = float(np.linalg.norm(spatial.covariance - covariance) / np.linalg.norm(covariance))
    root = spatial.precision_root
    cell["baseline_posterior_precision_unit_mean_max_abs_error"] = float(np.max(np.abs(
        root @ (spatial.mean - baseline.kernel_mean).flatten(order="F"))))
    cell["baseline_posterior_precision_unit_covariance_max_abs_error"] = float(np.max(np.abs(
        root @ covariance @ root.T - np.eye(len(root)))))
    gates = plan["evaluation"]["numerical_identity_gates"]
    posterior_passed = (
        cell["baseline_posterior_precision_unit_mean_max_abs_error"] <= gates["posterior_precision_unit_mean_max_abs_tolerance"]
        and cell["baseline_posterior_precision_unit_covariance_max_abs_error"] <= gates["posterior_precision_unit_covariance_identity_max_abs_tolerance"])
    cell["baseline_posterior_equivalence_passed"] = posterior_passed
    if score is None:
        cell.update(baseline_equivalence_passed=None, baseline_candidate_BF_validation="unavailable_reference_probability",
                    baseline_candidate_log_BF_max_abs_error=None)
        return
    reference_bf = np.asarray(score.log_evidence) - score.log_evidence[0]
    actual = np.asarray(receipt["prediction"]["log_evidence"])
    cell["baseline_candidate_log_BF_max_abs_error"] = float(np.max(np.abs(actual - actual[0] - reference_bf)))
    cell["baseline_equivalence_passed"] = posterior_passed and (
        cell["baseline_candidate_log_BF_max_abs_error"] <= gates["candidate_log_Bayes_factor_max_abs_tolerance"])


def _reference_score(baseline, view):
    """Reference-only numerical failure cannot cancel requested spatial arms."""
    try:
        result = baseline.score(view.samples, view.onsets, view.decision.membership,
            sfreq=view.sample_rate_hz, channels=view.channel_names, candidate_ids=view.decision.candidate_vocabulary,
            operator_identity=baseline.operator_identity, record_id=view.record_id)
        return result, None
    except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
        return None, {"stage": "baseline_reference_held_score", "type": type(error).__name__,
                      "detail": str(error), "status": "reference_probability_unavailable",
                      "fallback_model_or_prior": False}


def run_native_panel(*, plan_path, snapshot, output, reuse_baseline=None, baseline_origin_receipt=None,
                     subject=None):
    """Execute one declared six-cell panel; do not select priors from held output."""
    plan, plan_sha256 = _read_plan(plan_path, subject)
    reuse = plan.get("reuse")
    if reuse is not None:
        reuse_baseline = Path(reuse["baseline_path"])
        baseline_origin_receipt = Path(reuse["baseline_origin_receipt"])
        if (hashlib.sha256(reuse_baseline.read_bytes()).hexdigest() != reuse["baseline_sha256"]
                or hashlib.sha256(baseline_origin_receipt.read_bytes()).hexdigest() != reuse["baseline_origin_receipt_sha256"]):
            raise ValueError("original baseline or producing receipt changed from declared reuse authority")
    snapshot_state = verified_executed_snapshot(snapshot, root=ROOT,
        required_files=(Path(__file__).relative_to(ROOT).as_posix(),))
    input_path = Path(plan["input"]["path"])
    source_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if source_sha256 != plan["input"]["sha256"]:
        raise ValueError("actual A01 bytes differ from the predeclared scientific input")
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    start, cpu_start = perf_counter(), sum(process.cpu_times()[:2])
    resources = {"cpu_threads": 1, "gpu_workers": 0, "workers": 1,
                 "memory_available_before_bytes": psutil.virtual_memory().available,
                 "process_rss_before_bytes": process.memory_info().rss,
                 "host_cpu_before_percent": psutil.cpu_percent(interval=.1),
                 "cpu_affinity_logical": len(process.cpu_affinity())}
    cells = [{"cell": mode + "__" + prior, "observation": mode, "prior": prior, "status": "not_started"}
             for mode in plan["arms"]["observations"] for prior in plan["arms"]["signal_priors"]]
    state = {"schema": "n2p3_B09_native_panel/1", "status": "running", "plan_sha256": plan_sha256,
        "snapshot_manifest_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "actual_source_archive_sha256": snapshot_state["archive_sha256"], "entry_sha256": ENTRY_SHA256,
        "original_MAT_sha256": source_sha256, "requested_cells": 6, "cells": cells,
        "resources": resources, "evidence_scope": "single_participant_development_mechanism_panel",
        "public_population_efficacy_claimed": False, "adult_confirmation_claimed": False,
        "new_joint_AR_noise_MLE_claimed": False, "parameters_selected_from_held": False}
    receipt_path = output / "panel_receipt.json"
    atomic_write_json(receipt_path, state)
    with cpu_thread_budget(1):
        groups = prepare_bnci_calibration_groups(input_path, source_sha256=source_sha256,
            authorized_labels=800, group_count=3, response_seconds=1.2, pre_roll_seconds=.2)
        calibration, held = groups.groups[:2], groups.groups[2]
        view = held.view
        if (groups.ledger["used_label_count"] != 360 or sum(len(g.labels) for g in calibration) != 240
                or len(view.onsets) != 120 or len(view.decision.candidate_vocabulary) != 36
                or groups.ledger["response_samples"] != 307 or groups.ledger["preroll_samples"] != 51
                or tuple(view.channel_names) != tuple(plan["input"]["native_channels"])
                or view.reference != plan["input"]["native_reference"] or view.sample_rate_hz != 256.
                or view.signal_unit != "V" or any(kind != "EEG" for kind in view.channel_types)):
            raise ValueError("actual complete C/held scopes differ from the predeclared physical panel")
        state.update(actual_group_ledger=groups.ledger,
            actual_calibration=[g.record() for g in calibration], held_view=view.record(),
            actual_fit_label_count=240, actual_held_label_count_for_external_evaluation=120,
            unused_requested_fit_labels=800 - 240,
            unused_group_loader_authorized_labels=groups.ledger["unused_authorized_labels"],
            held_raw_disjoint_from_calibration=all(g.view.raw_scope[1] <= view.raw_scope[0] for g in calibration),
            evaluation_truth={"participant_key": view.participant_key, "decision_id": view.decision_id,
                              "target_candidate": held.target_candidate},
            candidate_vocabulary=list(view.decision.candidate_vocabulary),
            evidence_level=view.decision.repetitions,
            candidate_event_counts=list(map(int, view.decision.membership.sum(axis=0))))
        if not state["held_raw_disjoint_from_calibration"]:
            raise ValueError("later held raw observations overlap the actual C fit scopes")
        atomic_write_json(receipt_path, state)
        records = [CalibrationRecord(g.view.record_id, g.view.samples, g.view.onsets, g.labels) for g in calibration]
        dictionary = ResponseDictionary(ResponseDictionarySpec(**plan["baseline"]["dictionary"]), view.sample_rate_hz)
        baseline_path = output / "native_AR0_noise_model.npz" if reuse_baseline is None else reuse_baseline.resolve()
        begin = perf_counter()
        try:
            if reuse_baseline is None:
                if baseline_origin_receipt is not None:
                    raise ValueError("origin receipt is meaningful only for an explicit reused baseline")
                baseline = ContinuousDecisionModel.fit(records, sfreq=view.sample_rate_hz, channels=view.channel_names,
                    candidate_ids=view.decision.candidate_vocabulary,
                    operator_identity="B09_actual_native_unfiltered_reference_and_grid/1:" + metadata_digest({
                        "channels": view.channel_names, "reference": view.reference, "unit": view.signal_unit,
                        "source_sha256": source_sha256, "sample_rate_hz": view.sample_rate_hz}),
                    dictionary=dictionary, ridge=1., noise_order=0, conditioning_samples=32,
                    noise_estimator="conditional_profile", fit_scope="target_calibration")
                baseline.save(baseline_path)
            else:
                if baseline_origin_receipt is None:
                    raise ValueError("same-C baseline reuse requires its actual producer execution receipt")
                origin = json.loads(baseline_origin_receipt.read_text(encoding="utf-8"))
                original_plan_sha256 = plan_sha256 if reuse is None else reuse["prior_plan_sha256"]
                if (origin["plan_sha256"] != original_plan_sha256 or origin["original_MAT_sha256"] != source_sha256
                        or origin["actual_calibration"] != state["actual_calibration"]
                        or origin["prior_programs_frozen_before_first_held_score"][baseline_path.name]
                        != hashlib.sha256(baseline_path.read_bytes()).hexdigest()):
                    raise ValueError("reused baseline differs from its original same-C plan/source/producer")
                state["baseline_reuse"] = {"path": str(baseline_path),
                    "origin_receipt_sha256": hashlib.sha256(baseline_origin_receipt.read_bytes()).hexdigest(),
                    "original_source_archive_sha256": origin["actual_source_archive_sha256"],
                    "original_source_manifest_sha256": origin["snapshot_manifest_sha256"],
                    "original_fit_attempts": origin["baseline_fit_attempts"],
                    "current_fit_attempts": 0, "original_model_digest": origin["baseline_model_digest"]}
            baseline_file_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            baseline = ContinuousDecisionModel.load(baseline_path)
            if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_file_digest:
                raise ValueError("read-only baseline load altered its original model artifact")
            if reuse_baseline is not None and baseline.digest() != state["baseline_reuse"]["original_model_digest"]:
                raise ValueError("reused baseline state differs from its actual original producer")
        except UnidentifiableFitError as error:
            for cell in cells:
                cell.update(status="typed_baseline_failure", failure={"type": type(error).__name__, "detail": str(error)},
                            requested_decisions=1, scored_decisions=0)
                metrics, evidence = _evaluation(None, view, held.target_candidate, cell=cell["cell"],
                    model_sha256=None, fit_sha256=metadata_digest(state["actual_calibration"]),
                    noise_sha256=None, failure="typed_baseline_failure:" + str(error))
                cell["probability_metrics"] = metrics.record()
                cell["evidence"] = evidence.record()
            state.update(status="completed_with_typed_baseline_failure", baseline_fit_attempts=1,
                         baseline_fit_seconds=perf_counter() - begin)
            verified_executed_snapshot(snapshot, root=ROOT)
            if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_sha256:
                raise ValueError("original MAT changed during the typed baseline attempt") from error
            resources.update(wall_seconds=perf_counter() - start,
                process_cpu_seconds=sum(process.cpu_times()[:2]) - cpu_start,
                process_rss_after_bytes=process.memory_info().rss,
                memory_available_after_bytes=psutil.virtual_memory().available)
            atomic_write_json(receipt_path, state)
            return state
        if (baseline.sfreq != 256. or baseline.noise_order != 0 or baseline.conditioning_samples != 32
                or baseline.noise.provenance["fit_scope"] != "target_calibration"
                or baseline.dictionary.digest() != dictionary.digest()):
            raise ValueError("actual frozen baseline state differs from requested AR0/native/dictionary program")
        sigma = baseline.innovation_covariance
        shape = np.eye(len(view.channel_names))
        shape[view.channel_names.index("Pz"), view.channel_names.index("Pz")] = 4.
        roi_scale = len(sigma) / float(np.trace(np.linalg.solve(sigma, shape)))
        signals = {"noise_tied": sigma.copy(), "predeclared_Pz_ROI": roi_scale * shape}
        if "matched_diagonal_I" in plan["arms"]["signal_priors"]:
            independent_scale = len(sigma) / float(np.trace(np.linalg.solve(sigma, np.eye(len(sigma)))))
            signals["matched_diagonal_I"] = independent_scale * np.eye(len(sigma))
        priors = {}
        for name, signal in signals.items():
            prior_path = output / (name + "_prior_program.json")
            atomic_write_json(prior_path, {"channels": list(view.channel_names), "acquisition_reference": view.reference,
                "signal_unit": view.signal_unit, "mean": np.zeros((16, 8)).tolist(),
                "temporal_covariance": np.eye(16).tolist(), "signal_covariance": signal.tolist()})
            priors[name] = prior_path
        immutable_inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (baseline_path, *priors.values())}
        state.update(baseline_fit_attempts=1 if reuse_baseline is None else 0,
            baseline_fit_seconds=perf_counter() - begin,
            baseline_model_digest=baseline.digest(), baseline_noise_digest=baseline.noise.record()["artifact_sha256"],
            baseline_innovation_covariance_sha256=array_digest(sigma),
            baseline_noise_schema=baseline.noise.record()["schema"],
            baseline_dictionary=dictionary.metadata(),
            prior_programs_frozen_before_first_held_score=immutable_inputs,
            ROI_native_voltage_squared_scale=roi_scale, native_prior_spatial_energies={
                name: float(np.trace(np.linalg.solve(sigma, signal))) for name, signal in signals.items()})
        if "matched_diagonal_I" in signals:
            precision = np.linalg.solve(sigma, np.eye(len(sigma)))
            share = float(precision[view.channel_names.index("Pz"), view.channel_names.index("Pz")] / np.trace(precision))
            state["native_diagonal_shape_attribution"] = {"Pz_noise_precision_fraction": share,
                "a_I": independent_scale, "a_Pz4": roi_scale,
                "a_Pz4_over_a_I": roi_scale / independent_scale,
                "predeclared_scale_ratio_formula": 1 / (1 + 3 * share),
                "Pz_variance_relative_to_I": 4 * roi_scale / independent_scale,
                "other_variance_relative_to_I": roi_scale / independent_scale,
                "projection_rematched": False}
        atomic_write_json(receipt_path, state)
        # Prior arrays and C-fitted noise are frozen before reading any held score.
        baseline_score = None
        if reuse is not None:
            original = json.loads((Path(reuse["cell_results_path"]) / "panel_receipt.json").read_text(encoding="utf-8"))
            state["baseline_probability_metrics"] = original["baseline_probability_metrics"]
            state["baseline_score_reused_without_rescore"] = True
        else:
            baseline_score, reference_failure = _reference_score(baseline, view)
            state["baseline_reference_failure"] = reference_failure
            baseline_prediction = (None if baseline_score is None
                                   else {"log_probability": baseline_score.log_probability.tolist()})
            baseline_metrics, _ = _evaluation(baseline_prediction, view, held.target_candidate, cell="native_AR0_reference",
                model_sha256=baseline.digest(), fit_sha256=metadata_digest(state["actual_calibration"]),
                noise_sha256=immutable_inputs[baseline_path.name], failure_kind="not_available",
                failure=None if reference_failure is None else "reference_probability_unavailable:" + reference_failure["detail"])
            state["baseline_probability_metrics"] = baseline_metrics.record()
        panel_metrics = {}
        for cell in cells:
            cell_path = output / cell["cell"]
            begin = perf_counter()
            original_producer = None
            if reuse is not None and cell["cell"] in reuse["cells"]:
                cell_path, cell_receipt, original_producer = _admit_original_cell(
                    reuse, cell, state, view, signals[cell["prior"]], sigma, baseline, dictionary)
                exit_code = cell_receipt["status"] != "completed"
            else:
                cell.update(status="running", execution="new_fit_score_reload", spatial_fit_attempts=1)
                atomic_write_json(receipt_path, state)
                exit_code = audit.main(["--input", str(input_path), "--noise-model", str(baseline_path),
                    "--prior-program", str(priors[cell["prior"]]), "--snapshot", str(snapshot),
                    "--output", str(cell_path), "--authorized-labels", "800", "--fit-groups", "2",
                    "--observation", cell["observation"], "--reference-channel", "Cz"])
                cell_receipt = json.loads((cell_path / "receipt.json").read_text(encoding="utf-8"))
            if original_producer is None and cell_receipt["snapshot_sha256"] != state["snapshot_manifest_sha256"]:
                raise ValueError("a spatial cell consumed a different scientific source snapshot")
            if cell_receipt["identities"]["noise_model"] != immutable_inputs[baseline_path.name]:
                raise ValueError("a spatial cell used different fitted native noise")
            cell.update(status=cell_receipt["status"], cli_exit_code=exit_code, seconds=perf_counter() - begin,
                requested_decisions=1, scored_decisions=int(cell_receipt["status"] == "completed"),
                receipt_sha256=hashlib.sha256((cell_path / "receipt.json").read_bytes()).hexdigest(),
                observation_rank=cell_receipt["observation"]["rank_ledger"]["structural_rank"],
                result_directory=str(cell_path))
            if original_producer is not None:
                cell.update(execution="original_fit_and_score_reused_without_rescore", spatial_fit_attempts=0,
                            original_producer=original_producer)
            if cell["observation_rank"] != plan["arms"]["expected_observation_ranks"][cell["observation"]]:
                raise ValueError("actual observation rank differs from the declared measured information")
            observation = audit._observation(audit._layout_from_view(view), cell["observation"], "Cz")
            cell["projected_signal_prior_spatial_energy"] = _spatial_energy(observation, signals[cell["prior"]], sigma)
            model_digest = None
            failure = None
            if cell["status"] == "completed":
                spatial = SpatialSignalPosterior.load(cell_path / "spatial_model.json")
                model_digest = spatial.record()["artifact_sha256"]
                if cell["cell"] == "native__noise_tied":
                    if original_producer is not None:
                        cell.update(original_producer["original_baseline_equivalence"])
                        if cell.get("baseline_equivalence_passed") is not True:
                            raise ValueError("reused native MN equivalence was not verified by its original producer")
                    else:
                        _baseline_equivalence(cell, spatial, baseline, baseline_score, cell_receipt, plan)
            else:
                failure = json.dumps(cell_receipt["failure"], sort_keys=True)
                cell["failure"] = cell_receipt["failure"]
            cell["saved_package_files"] = _files(cell_path)
            metrics, evidence = _evaluation(cell_receipt.get("prediction"), view, held.target_candidate,
                cell=cell["cell"], model_sha256=model_digest, fit_sha256=metadata_digest(state["actual_calibration"]),
                noise_sha256=immutable_inputs[baseline_path.name], failure=failure)
            panel_metrics[cell["cell"]] = metrics
            cell["probability_metrics"] = metrics.record()
            cell["evidence"] = evidence.record()
            if evidence.probability_available:
                cell["winner_confidence"] = float(np.max(evidence.probability))
                cell["wrong_winner_confidence"] = (cell["winner_confidence"] if evidence.predicted is not None
                                                    and evidence.predicted != held.target_candidate else None)
            atomic_write_json(receipt_path, state)
        contrasts = plan["evaluation"].get("within_observation_contrasts", [["noise_tied", "predeclared_Pz_ROI"]])
        state["same_observation_prior_comparisons"] = {
            mode + ":" + first + "->" + second: compare_candidate_probability_metrics(
                panel_metrics[mode + "__" + first], panel_metrics[mode + "__" + second]).record()
            for mode in plan["arms"]["observations"] for first, second in contrasts}
        quotient_checks = {}
        for name in plan["arms"]["signal_priors"]:
            if "car" not in plan["arms"]["observations"]:
                break
            first = next(c for c in cells if c["cell"] == "cz__" + name)
            second = next(c for c in cells if c["cell"] == "car__" + name)
            if first["status"] == second["status"] == "completed":
                cz = json.loads((output / first["cell"] / "receipt.json").read_text(encoding="utf-8"))["prediction"]
                car = json.loads((output / second["cell"] / "receipt.json").read_text(encoding="utf-8"))["prediction"]
                d0, d1 = np.asarray(cz["log_evidence"]), np.asarray(car["log_evidence"])
                quotient_checks[name] = {"candidate_log_BF_max_abs_error": float(np.max(np.abs(d0 - d0[0] - d1 + d1[0]))),
                    "log_probability_max_abs_error": float(np.max(np.abs(np.asarray(cz["log_probability"]) - car["log_probability"]))) }
                gates = plan["evaluation"]["numerical_identity_gates"]
                quotient_checks[name]["passed"] = (
                    quotient_checks[name]["candidate_log_BF_max_abs_error"] <= gates["candidate_log_Bayes_factor_max_abs_tolerance"]
                    and quotient_checks[name]["log_probability_max_abs_error"] <= gates["normalized_log_probability_max_abs_tolerance"])
            else:
                quotient_checks[name] = {"status": "unavailable_due_to_requested_failure"}
        state["same_quotient_Cz_CAR_checks"] = quotient_checks
        state["status"] = "completed" if all(c["status"] == "completed" for c in cells) else "completed_with_typed_failures"
        if any(not value["passed"] for value in quotient_checks.values() if "passed" in value) or any(
            c.get("baseline_equivalence_passed") is False or c.get("baseline_posterior_equivalence_passed") is False
            for c in cells):
            state["status"] = "mechanism_validation_failed"
        state["equivalence_validation_quality"] = ("unavailable_reference_probability" if
            state.get("baseline_reference_failure") is not None or any(
                c.get("baseline_candidate_BF_validation") == "unavailable_reference_probability" for c in cells)
            else "available")
    verified_executed_snapshot(snapshot, root=ROOT, required_files=(Path(__file__).relative_to(ROOT).as_posix(),))
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_sha256 or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
        raise ValueError("scientific input or predeclared plan changed during execution")
    for name, digest in immutable_inputs.items():
        actual_path = baseline_path if name == baseline_path.name else output / name
        if hashlib.sha256(actual_path.read_bytes()).hexdigest() != digest:
            raise ValueError("a fixed baseline or prior input was modified by the panel")
    resources.update(wall_seconds=perf_counter() - start, process_cpu_seconds=sum(process.cpu_times()[:2]) - cpu_start,
        process_rss_after_bytes=process.memory_info().rss, memory_available_after_bytes=psutil.virtual_memory().available,
        process_peak_wset_bytes=process.memory_info()._asdict().get("peak_wset"))
    state["files_before_final_receipt"] = _files(output)
    atomic_write_json(receipt_path, state)
    return state


def _matrix_metrics(subjects, plan_sha256, observations, priors):
    """The shared requested-participant owner receives every subject, including failures."""
    output = {}
    scopes = {"all_eight_development": subjects,
              "A02_A08_variant_prospective_development": subjects[1:], "A01_seen_development": subjects[:1]}
    for scope, panels in scopes.items():
        roster = tuple(panel["evaluation_truth"]["participant_key"] for panel in panels)
        entries = tuple(DecisionPlanEntry(**panel["evaluation_truth"]) for panel in panels)
        identity = metadata_digest([{key: panel[key] for key in ("original_MAT_sha256", "held_view")}
                                    for panel in panels])
        decision_plan = DecisionPlanContract(plan_sha256, identity, roster, entries)
        vocabulary = tuple(panels[0]["candidate_vocabulary"])
        level = panels[0]["evidence_level"]
        coverage = {DecisionKey(panel["evaluation_truth"]["participant_key"], panel["evaluation_truth"]["decision_id"]):
            CandidateCoverage.from_mappings(dict(zip(vocabulary, panel["candidate_event_counts"], strict=True)), {})
            for panel in panels}
        metrics, records = {}, {}
        for mode in observations:
            for prior in priors:
                arm = mode + "__" + prior
                rows = [CandidateEvidence.from_record(next(c for c in panel["cells"] if c["cell"] == arm)["evidence"])
                        for panel in panels]
                result = evaluate_candidate_evidence({20260918: rows}, decision_plan=decision_plan,
                    seeds=(20260918,), candidate_vocabulary=vocabulary, evidence_level=level,
                    requested_coverage=coverage)
                metrics[arm] = result
                records[arm] = result.record()
        contrasts = {mode + ":" + first + "->" + second: compare_candidate_probability_metrics(
            metrics[mode + "__" + first], metrics[mode + "__" + second]).record()
            for mode in observations for first, second in (("noise_tied", "matched_diagonal_I"),
                                                           ("matched_diagonal_I", "predeclared_Pz_ROI"))}
        reference = {prior: compare_candidate_probability_metrics(metrics["native__" + prior],
            metrics["cz__" + prior]).record() for prior in priors}
        wrong_confidence = {mode + "__" + prior: [
            {"participant": panel["evaluation_truth"]["participant_key"],
             "confidence": next(c for c in panel["cells"] if c["cell"] == mode + "__" + prior).get("wrong_winner_confidence")}
            for panel in panels] for mode in observations for prior in priors}
        output[scope] = {"requested_participants": len(panels), "arms": records,
                        "within_observation_prior_contrasts": contrasts,
                        "reference_information_and_prior_interaction_diagnostics": reference,
                        "wrong_winner_confidence_by_requested_participant": wrong_confidence,
                        "default_or_population_promotion": False}
    return output


def run_attribution_matrix(*, plan_path, snapshot, output):
    """Sequentially complete all predeclared participants through the existing panel seam."""
    plan_bytes = plan_path.read_bytes()
    document = json.loads(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if document["schema"] != "n2p3_B09_spatial_attribution_matrix/1":
        raise ValueError("an attribution matrix requires the explicit 48-cell plan")
    source = verified_executed_snapshot(snapshot, root=ROOT,
        required_files=(Path(__file__).relative_to(ROOT).as_posix(),))
    inputs = document["inputs"]
    for item in inputs:
        _read_plan(plan_path, item["subject"])
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("requested original participant bytes differ from the complete frozen roster")
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema": "n2p3_B09_spatial_attribution_matrix_result/1", "status": "running",
        "plan_sha256": plan_sha256, "current_source_archive_sha256": source["archive_sha256"],
        "current_source_manifest_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "requested_participants": 8, "requested_cells": 48, "completed_subjects": [], "cells": [],
        "participant_roster": [{**item, "status": "not_started"} for item in inputs],
        "new_baseline_fits": 0, "new_spatial_fit_attempts": 0, "old_spatial_results_reused": 0,
        "seen_A01_kept_in_requested_development_roster": True,
        "CAR_repeated": False, "public_confirmatory_or_adult_claim": False,
        "whole_matrix_prior_selection_from_held": False}
    receipt = output / "matrix_receipt.json"
    atomic_write_json(receipt, state)
    panels = []
    start = perf_counter()
    try:
        for item in inputs:
            panel = run_native_panel(plan_path=plan_path, snapshot=snapshot,
                output=output / item["subject"], subject=item["subject"])
            panels.append(panel)
            state["completed_subjects"].append(item["subject"])
            for row in state["participant_roster"]:
                if row["subject"] == item["subject"]:
                    row["status"] = panel["status"]
            state["new_baseline_fits"] += panel["baseline_fit_attempts"]
            state["new_spatial_fit_attempts"] += sum(c.get("spatial_fit_attempts", 0) for c in panel["cells"])
            state["old_spatial_results_reused"] += sum(c.get("execution") == "original_fit_and_score_reused_without_rescore"
                                                       for c in panel["cells"])
            state["cells"] += [{"participant": item["subject"], **c} for c in panel["cells"]]
            state["current_subject_receipts"] = {i["subject"]: hashlib.sha256(
                (output / i["subject"] / "panel_receipt.json").read_bytes()).hexdigest()
                for i in inputs if i["subject"] in state["completed_subjects"]}
            atomic_write_json(receipt, state)
        state["participant_paired_metrics"] = _matrix_metrics(panels, plan_sha256,
            document["protocol"]["arms"]["observations"], document["protocol"]["arms"]["signal_priors"])
        if len(state["cells"]) != 48 or len(panels) != 8:
            raise ValueError("actual results do not cover every declared participant/arm")
        state["status"] = ("mechanism_validation_failed" if any(p["status"] == "mechanism_validation_failed" for p in panels)
            else "completed" if all(c["status"] == "completed" for c in state["cells"])
            else "completed_with_typed_failures")
        state["equivalence_reference_unavailable_participants"] = [item["subject"]
            for item, panel in zip(inputs, panels, strict=True)
            if panel.get("equivalence_validation_quality") == "unavailable_reference_probability"]
        state["wall_seconds"] = perf_counter() - start
        verified_executed_snapshot(snapshot, root=ROOT)
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
            raise ValueError("predeclared matrix plan changed during execution")
        atomic_write_json(receipt, state)
        return state
    except Exception as error:
        state.update(status="aborted", fatal_error={"type": type(error).__name__, "detail": str(error)})
        atomic_write_json(receipt, state)
        raise


def run_declared_program(*, plan_path, snapshot, output, **options):
    schema = json.loads(plan_path.read_text(encoding="utf-8"))["schema"]
    if schema == "n2p3_B09_spatial_attribution_matrix/1":
        if any(value is not None for value in options.values()):
            raise ValueError("matrix reuse authority belongs to the full declared plan")
        return run_attribution_matrix(plan_path=plan_path, snapshot=snapshot, output=output)
    return run_native_panel(plan_path=plan_path, snapshot=snapshot, output=output, **options)


def test_native_panel_only_runs_on_explicit_frozen_inputs():
    """Optional actual-data integration; no default pytest fit or duplicate cases."""
    import pytest
    keys = ("B09_NATIVE_PLAN", "B09_NATIVE_SNAPSHOT", "B09_NATIVE_OUTPUT")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("real A01 panel requires explicitly frozen plan/snapshot/output")
    result = run_declared_program(plan_path=Path(os.environ[keys[0]]), snapshot=Path(os.environ[keys[1]]),
                              output=Path(os.environ[keys[2]]))
    assert len(result["cells"]) in {6, 48} and all(c["status"] != "not_started" for c in result["cells"])


def test_attribution_metrics_keep_all_requested_failed_participants():
    """One state/metric seam regression; no native data, baseline or spatial fit."""
    panels = []
    identity = CandidateExpertIdentity("synthetic", "a" * 64, "f" * 64, None, "b" * 64,
                                      "synthetic_probability_metric_fixture", False)
    for index in range(8):
        participant = "synthetic_A" + str(index + 1)
        cells = []
        for mode in ("native", "cz"):
            for prior in ("noise_tied", "matched_diagonal_I", "predeclared_Pz_ROI"):
                failure = index == 1 and prior == "noise_tied"
                evidence = CandidateEvidence(("a", "b"), None if failure else np.log([.8, .2]),
                    participant, "held", "c" * 64, (identity,), 1, (10, 10), (0, 0) if failure else (10, 10),
                    failure_reason="synthetic_fit_failure" if failure else None,
                    failure_kind="fit_failure" if failure else None)
                cells.append({"cell": mode + "__" + prior, "evidence": evidence.record()})
        panels.append({"evaluation_truth": {"participant_key": participant, "decision_id": "held", "target_candidate": "a"},
            "original_MAT_sha256": "d" * 64, "held_view": {"physical_support_digest": "c" * 64},
            "candidate_vocabulary": ["a", "b"], "evidence_level": 1,
            "candidate_event_counts": [10, 10], "cells": cells})
    metrics = _matrix_metrics(panels, "e" * 64, ["native", "cz"],
        ["noise_tied", "matched_diagonal_I", "predeclared_Pz_ROI"])
    all_eight = metrics["all_eight_development"]
    failed_arm = all_eight["arms"]["native__noise_tied"]
    assert failed_arm["requested_participant_count"] == 8
    assert failed_arm["prediction_coverage"] == 7 / 8
    assert failed_arm["requested_participant_operational_hit_rate"] == 7 / 8
    assert len(failed_arm["participants"]) == 8
    contrast = all_eight["within_observation_prior_contrasts"]["native:noise_tied->matched_diagonal_I"]
    assert len(contrast["missing_probability_ledger"]) == 1
    assert contrast["development_gate_passed"] is False
    assert metrics["A02_A08_variant_prospective_development"]["requested_participants"] == 7


def test_reference_only_numeric_fault_is_recorded_without_hiding_contract_failure():
    """Actual helper fault path, zero source/EEG fits and no resource/public claim."""
    import pytest
    view = SimpleNamespace(samples=None, onsets=None, sample_rate_hz=256., channel_names=("Fz",),
        record_id="synthetic-held", decision=SimpleNamespace(membership=None, candidate_vocabulary=("a", "b")))
    for failure_type in (UnidentifiableFitError, np.linalg.LinAlgError):
        def failed_score(*args, _failure_type=failure_type, **kwargs):
            raise _failure_type("synthetic reference-only numerical fault")
        reference = SimpleNamespace(score=failed_score, operator_identity="synthetic_reference_only")
        result, failure = _reference_score(reference, view)
        assert result is None
        assert failure["stage"] == "baseline_reference_held_score"
        assert failure["status"] == "reference_probability_unavailable"
        assert failure["fallback_model_or_prior"] is False
    def bad_contract(*args, **kwargs):
        raise ValueError("synthetic source/physical contract failure")
    with pytest.raises(ValueError, match="source/physical contract"):
        _reference_score(SimpleNamespace(score=bad_contract, operator_identity="synthetic"), view)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-baseline", type=Path)
    parser.add_argument("--baseline-origin-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_declared_program(plan_path=args.plan, snapshot=args.snapshot, output=args.output,
            reuse_baseline=args.reuse_baseline, baseline_origin_receipt=args.baseline_origin_receipt)
    except Exception as error:
        receipt_path = args.output / "panel_receipt.json"
        if receipt_path.is_file():
            record = json.loads(receipt_path.read_text(encoding="utf-8"))
            record.update(status="aborted", fatal_error={"type": type(error).__name__, "detail": str(error)})
            atomic_write_json(receipt_path, record)
        raise
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]), "output": str(args.output)}))
    return 1 if result["status"] == "mechanism_validation_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
