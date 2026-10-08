"""One explicitly invoked A01 mechanism panel through the real spatial CLI.

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


def _read_plan(path):
    payload = path.read_bytes()
    plan = json.loads(payload)
    support, baseline, arms = plan["support"], plan["baseline"], plan["arms"]
    if (plan["schema"] != "n2p3_B09_native_mechanism_plan/1"
            or support["authorized_labels_request"] != 800 or support["first_complete_fit_groups"] != 2
            or support["held_complete_group_ordinal"] != 3 or support["conditioning_samples"] != 32
            or baseline["noise_order"] != 0 or baseline["noise_estimator"] != "conditional_profile"
            or baseline["ridge"] != 1. or baseline["dictionary"] != {
                "family": "gaussian", "response_seconds": 1.2, "n_basis": 8, "endpoint": "free"}
            or arms["observations"] != ["native", "cz", "car"]
            or arms["signal_priors"] != ["noise_tied", "predeclared_Pz_ROI"]
            or arms["predeclared_Pz_ROI"]["Pz_variance_ratio"] != 4. or arms["temperature"] != 1.
            or plan["resources"]["cpu_threads"] != 1 or arms["requested_cells"] != 6):
        raise ValueError("native B09 panel requires its complete agreed predeclared protocol")
    return plan, hashlib.sha256(payload).hexdigest()


def _files(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def _evaluation(prediction, view, truth, *, cell, model_sha256, fit_sha256, noise_sha256, failure=None):
    vocabulary = view.decision.candidate_vocabulary
    counts = tuple(map(int, view.decision.membership.sum(axis=0)))
    identity = CandidateExpertIdentity(cell, view.raw_input_digest, model_sha256, noise_sha256,
        fit_sha256, "frozen_noise_general_Gaussian_conditional_tail", False)
    evidence = CandidateEvidence(vocabulary, None if failure is not None else np.asarray(prediction["log_probability"]),
        view.participant_key, view.decision_id, view.physical_support_digest, (identity,),
        view.decision.repetitions, counts, (0,) * len(counts) if failure is not None else counts,
        failure_reason=failure, failure_kind="fit_failure" if failure is not None else None)
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


def run_native_panel(*, plan_path, snapshot, output):
    """Execute one declared six-cell panel; do not select priors from held output."""
    plan, plan_sha256 = _read_plan(plan_path)
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
        "resources": resources, "evidence_scope": "single_A01_development_mechanism_panel",
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
            held_raw_disjoint_from_calibration=all(g.view.raw_scope[1] <= view.raw_scope[0] for g in calibration))
        if not state["held_raw_disjoint_from_calibration"]:
            raise ValueError("later held raw observations overlap the actual C fit scopes")
        atomic_write_json(receipt_path, state)
        records = [CalibrationRecord(g.view.record_id, g.view.samples, g.view.onsets, g.labels) for g in calibration]
        dictionary = ResponseDictionary(ResponseDictionarySpec(**plan["baseline"]["dictionary"]), view.sample_rate_hz)
        baseline_path = output / "native_AR0_noise_model.npz"
        begin = perf_counter()
        try:
            baseline = ContinuousDecisionModel.fit(records, sfreq=view.sample_rate_hz, channels=view.channel_names,
                candidate_ids=view.decision.candidate_vocabulary,
                operator_identity="B09_actual_native_unfiltered_reference_and_grid/1:" + metadata_digest({
                    "channels": view.channel_names, "reference": view.reference, "unit": view.signal_unit,
                    "source_sha256": source_sha256, "sample_rate_hz": view.sample_rate_hz}),
                dictionary=dictionary, ridge=1., noise_order=0, conditioning_samples=32,
                noise_estimator="conditional_profile", fit_scope="target_calibration")
            baseline.save(baseline_path)
            baseline_file_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            baseline = ContinuousDecisionModel.load(baseline_path)
            if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_file_digest:
                raise ValueError("read-only baseline load altered its original model artifact")
        except UnidentifiableFitError as error:
            for cell in cells:
                cell.update(status="typed_baseline_failure", failure={"type": type(error).__name__, "detail": str(error)},
                            requested_decisions=1, scored_decisions=0)
                metrics, _ = _evaluation(None, view, held.target_candidate, cell=cell["cell"],
                    model_sha256=None, fit_sha256=metadata_digest(state["actual_calibration"]),
                    noise_sha256=None, failure="typed_baseline_failure:" + str(error))
                cell["probability_metrics"] = metrics.record()
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
        priors = {}
        for name, signal in signals.items():
            prior_path = output / (name + "_prior_program.json")
            atomic_write_json(prior_path, {"channels": list(view.channel_names), "acquisition_reference": view.reference,
                "signal_unit": view.signal_unit, "mean": np.zeros((16, 8)).tolist(),
                "temporal_covariance": np.eye(16).tolist(), "signal_covariance": signal.tolist()})
            priors[name] = prior_path
        immutable_inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (baseline_path, *priors.values())}
        state.update(baseline_fit_attempts=1, baseline_fit_seconds=perf_counter() - begin,
            baseline_model_digest=baseline.digest(), baseline_noise_digest=baseline.noise.record()["artifact_sha256"],
            baseline_innovation_covariance_sha256=array_digest(sigma),
            baseline_noise_schema=baseline.noise.record()["schema"],
            baseline_dictionary=dictionary.metadata(),
            prior_programs_frozen_before_first_held_score=immutable_inputs,
            ROI_native_voltage_squared_scale=roi_scale, native_prior_spatial_energies={
                name: float(np.trace(np.linalg.solve(sigma, signal))) for name, signal in signals.items()})
        atomic_write_json(receipt_path, state)
        # Prior arrays and C-fitted noise are frozen before reading any held score.
        baseline_score = baseline.score(view.samples, view.onsets, view.decision.membership,
            sfreq=view.sample_rate_hz, channels=view.channel_names, candidate_ids=view.decision.candidate_vocabulary,
            operator_identity=baseline.operator_identity, record_id=view.record_id)
        baseline_prediction = {"log_probability": baseline_score.log_probability.tolist()}
        baseline_metrics, _ = _evaluation(baseline_prediction, view, held.target_candidate, cell="native_AR0_reference",
            model_sha256=baseline.digest(), fit_sha256=metadata_digest(state["actual_calibration"]),
            noise_sha256=immutable_inputs[baseline_path.name])
        state["baseline_probability_metrics"] = baseline_metrics.record()
        panel_metrics = {}
        for cell in cells:
            cell_path = output / cell["cell"]
            begin = perf_counter()
            exit_code = audit.main(["--input", str(input_path), "--noise-model", str(baseline_path),
                "--prior-program", str(priors[cell["prior"]]), "--snapshot", str(snapshot),
                "--output", str(cell_path), "--authorized-labels", "800", "--fit-groups", "2",
                "--observation", cell["observation"], "--reference-channel", "Cz"])
            cell_receipt = json.loads((cell_path / "receipt.json").read_text(encoding="utf-8"))
            if cell_receipt["snapshot_sha256"] != state["snapshot_manifest_sha256"]:
                raise ValueError("a spatial cell consumed a different scientific source snapshot")
            if cell_receipt["identities"]["noise_model"] != immutable_inputs[baseline_path.name]:
                raise ValueError("a spatial cell used different fitted native noise")
            cell.update(status=cell_receipt["status"], cli_exit_code=exit_code, seconds=perf_counter() - begin,
                requested_decisions=1, scored_decisions=int(cell_receipt["status"] == "completed"),
                receipt_sha256=hashlib.sha256((cell_path / "receipt.json").read_bytes()).hexdigest(),
                observation_rank=cell_receipt["observation"]["rank_ledger"]["structural_rank"])
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
                    cell["baseline_posterior_mean_max_abs_error"] = float(np.max(np.abs(spatial.mean - baseline.kernel_mean)))
                    covariance = np.kron(sigma, baseline.kernel_row_covariance)
                    cell["baseline_posterior_covariance_relative_error"] = float(np.linalg.norm(spatial.covariance - covariance) / np.linalg.norm(covariance))
                    root = spatial.precision_root
                    cell["baseline_posterior_precision_unit_mean_max_abs_error"] = float(np.max(np.abs(
                        root @ (spatial.mean - baseline.kernel_mean).flatten(order="F"))))
                    cell["baseline_posterior_precision_unit_covariance_max_abs_error"] = float(np.max(np.abs(
                        root @ covariance @ root.T - np.eye(len(root)))))
                    reference_bf = np.asarray(baseline_score.log_evidence) - baseline_score.log_evidence[0]
                    actual = np.asarray(cell_receipt["prediction"]["log_evidence"])
                    cell["baseline_candidate_log_BF_max_abs_error"] = float(np.max(np.abs(actual - actual[0] - reference_bf)))
                    gates = plan["evaluation"]["numerical_identity_gates"]
                    cell["baseline_equivalence_passed"] = (
                        cell["baseline_posterior_precision_unit_mean_max_abs_error"] <= gates["posterior_precision_unit_mean_max_abs_tolerance"]
                        and cell["baseline_posterior_precision_unit_covariance_max_abs_error"] <= gates["posterior_precision_unit_covariance_identity_max_abs_tolerance"]
                        and cell["baseline_candidate_log_BF_max_abs_error"] <= gates["candidate_log_Bayes_factor_max_abs_tolerance"])
            else:
                failure = json.dumps(cell_receipt["failure"], sort_keys=True)
                cell["failure"] = cell_receipt["failure"]
            cell["saved_package_files"] = _files(cell_path)
            metrics, evidence = _evaluation(cell_receipt.get("prediction"), view, held.target_candidate,
                cell=cell["cell"], model_sha256=model_digest, fit_sha256=metadata_digest(state["actual_calibration"]),
                noise_sha256=immutable_inputs[baseline_path.name], failure=failure)
            panel_metrics[cell["cell"]] = metrics
            cell["probability_metrics"] = metrics.record()
            if evidence.probability_available:
                cell["winner_confidence"] = float(np.max(evidence.probability))
                cell["wrong_winner_confidence"] = (cell["winner_confidence"] if evidence.predicted is not None
                                                    and evidence.predicted != held.target_candidate else None)
            atomic_write_json(receipt_path, state)
        state["same_observation_prior_comparisons"] = {
            mode: compare_candidate_probability_metrics(panel_metrics[mode + "__noise_tied"],
                panel_metrics[mode + "__predeclared_Pz_ROI"]).record() for mode in plan["arms"]["observations"]}
        quotient_checks = {}
        for name in plan["arms"]["signal_priors"]:
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
            c.get("baseline_equivalence_passed") is False for c in cells):
            state["status"] = "mechanism_validation_failed"
    verified_executed_snapshot(snapshot, root=ROOT, required_files=(Path(__file__).relative_to(ROOT).as_posix(),))
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_sha256 or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
        raise ValueError("scientific input or predeclared plan changed during execution")
    for name, digest in immutable_inputs.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError("a fixed baseline or prior input was modified by the panel")
    resources.update(wall_seconds=perf_counter() - start, process_cpu_seconds=sum(process.cpu_times()[:2]) - cpu_start,
        process_rss_after_bytes=process.memory_info().rss, memory_available_after_bytes=psutil.virtual_memory().available,
        process_peak_wset_bytes=process.memory_info()._asdict().get("peak_wset"))
    state["files_before_final_receipt"] = _files(output)
    atomic_write_json(receipt_path, state)
    return state


def test_native_panel_only_runs_on_explicit_frozen_inputs():
    """Optional actual-data integration; no default pytest fit or duplicate cases."""
    import pytest
    keys = ("B09_NATIVE_PLAN", "B09_NATIVE_SNAPSHOT", "B09_NATIVE_OUTPUT")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("real A01 panel requires explicitly frozen plan/snapshot/output")
    result = run_native_panel(plan_path=Path(os.environ[keys[0]]), snapshot=Path(os.environ[keys[1]]),
                              output=Path(os.environ[keys[2]]))
    assert len(result["cells"]) == 6 and all(c["status"] != "not_started" for c in result["cells"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_native_panel(plan_path=args.plan, snapshot=args.snapshot, output=args.output)
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
