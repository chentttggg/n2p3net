"""Run an attested BNCI continuous diagnostic with all requested decisions.

Plan n2p3_bnci_continuous_study/1 declares records [{subject,path,sha256}],
budgets, continuous_input (ContinuousInputPlan fields), evidence_scope and
cpu_threads. Original MAT paths are resolved relative to the plan. This is
public development, not a claim of superiority over an epoch classifier.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.continuous_decision import ContinuousDecisionModel, smooth_erp_basis  # noqa: E402
from transfer.outcomes import (  # noqa: E402
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)
from transfer.raw_continuous import (  # noqa: E402
    CANDIDATES,
    SFREQ,
    ContinuousInputPlan,
    prepare_bnci_continuous,
)
from transfer.result_reuse import seal_result_record  # noqa: E402


def _source_bytes():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name is not None:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT / "src"):
                paths.add(path)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): path.read_bytes() for path in sorted(paths)}


_EXECUTING_BYTES = _source_bytes()


def _assert_source_unchanged():
    if any((ROOT / name).read_bytes() != content for name, content in _EXECUTING_BYTES.items()):
        raise RuntimeError("loaded implementation files changed; start a new frozen execution")


def _predict(model, view, operator_identity, channels):
    return model.score(view.samples, view.onsets, view.membership,
        operator_identity=operator_identity, sfreq=SFREQ, channels=channels,
        candidate_ids=CANDIDATES, record_id=view.decision_id)


def _cell(views, budget, plan, directory, source_digest, *, evidence_scope):
    started = time.perf_counter()
    directory.mkdir()
    operator = views.ledger["operator_identity"]
    channels = tuple(plan.record()["channels"])
    model = restored = None
    fit_failure = None
    model_path = directory / "model.npz"
    try:
        model = ContinuousDecisionModel.fit([views.calibration[budget]], sfreq=SFREQ,
            channels=channels, candidate_ids=CANDIDATES, operator_identity=operator,
            basis=smooth_erp_basis(SFREQ, plan.response_seconds, plan.n_basis), ridge=plan.ridge)
        model.save(model_path)
        restored = ContinuousDecisionModel.load(model_path)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        fit_failure = str(exc)
    outcomes, predictions, requested = [], [], []
    for view in views.decisions:
        key = DecisionKey(views.subject, view.decision_id)
        requested.append(key)
        truth = views.scoring_truth[view.decision_id]
        winner, nll, probabilities, log_evidence, scored_input = None, None, None, None, None
        reason = view.failure_reason
        if reason is not None:
            status = DecisionStatus.INCOMPLETE
        elif fit_failure is not None:
            status, reason = DecisionStatus.FIT_FAILURE, fit_failure
        else:
            try:
                predicted = _predict(model, view, operator, channels)
                replayed = _predict(restored, view, operator, channels)
                np.testing.assert_array_equal(predicted.log_evidence, replayed.log_evidence)
                if predicted.winner != replayed.winner:
                    raise AssertionError("frozen reload changed the candidate winner")
                winner = predicted.winner
                status = (DecisionStatus.TIE if winner is None else
                          DecisionStatus.CORRECT if winner == truth else DecisionStatus.INCORRECT)
                nll = float(logsumexp(predicted.log_evidence) - predicted.log_evidence[CANDIDATES.index(truth)])
                probabilities = predicted.probabilities.tolist()
                log_evidence = predicted.log_evidence.tolist()
                scored_input = predicted.input_sha256
                reason = predicted.reason
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                status, reason = DecisionStatus.ABSTAIN, str(exc)
        outcomes.append(DecisionOutcome(key, 10, status, view.coverage, truth, winner, reason,
            view.onset_start_sample / SFREQ, view.onset_end_sample / SFREQ, view.available_sample / SFREQ))
        predictions.append({"decision_id": view.decision_id, "prediction": winner,
            "probabilities": probabilities, "candidate_ids": list(CANDIDATES), "nll": nll,
            "log_evidence": log_evidence, "scored_input_sha256": scored_input,
            "nll_available": nll is not None, "status": status.value, "reason": reason,
            "sample_scope": [view.start_sample, view.stop_sample],
            "modeled_response_ready_s": view.stop_sample / SFREQ,
            "actual_offline_filter_ready_s": view.available_sample / SFREQ,
            "prediction_received_target_label": False,
            "saved_reload_scores_exact": nll is not None})
    accounting = build_decision_outcome_accounting(outcomes, requested_decisions=requested, evidence_levels=[10]).to_record()
    counts = accounting["by_evidence_level"]["10"]
    nlls = [row["nll"] for row in predictions if row["nll"] is not None]
    prediction_path = directory / "predictions.json"
    atomic_write_json(prediction_path, {"predictions": predictions, "outcomes": [o.to_record() for o in outcomes]})
    cell = {"schema": "n2p3_bnci_continuous_cell/1", "subject": views.subject, "K": budget,
        "status": "fit_failed" if fit_failure is not None else "completed", "fit_failure": fit_failure,
        "execution_source_sha256": source_digest, "source_raw_sha256": views.ledger["source_sha256"],
        "evidence_scope": evidence_scope,
        "scope_ledger_sha256": semantic_sha256(views.ledger), "operator_identity": operator,
        "fit": None if model is None else dict(model.provenance),
        "authorized_label_count": budget, "retained_label_count": len(views.calibration[budget].target_mask),
        "decision_accounting": accounting, "requested_hit_all": counts["operational_hit_rate"],
        "coverage": counts["valid"] / counts["requested"],
        "scored_nll_mean": float(np.mean(nlls)) if nlls else None,
        "nll_scored_decisions": len(nlls), "nll_missing_decisions": len(predictions) - len(nlls),
        "requested_nll_with_explicit_uniform_abstention": float(
            (sum(nlls) + (len(predictions) - len(nlls)) * np.log(len(CANDIDATES))) / len(predictions)),
        "uniform_abstention_is_reporting_policy_not_model_prediction": True,
        "mean_actual_offline_latency_s": float(np.mean([
            (view.available_sample - view.onset_start_sample) / SFREQ for view in views.decisions])),
        "mean_finite_response_latency_s": float(np.mean([
            (view.stop_sample - view.onset_start_sample) / SFREQ for view in views.decisions])),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "new_continuous_native_input_route_diagnostic_not_matched_algorithm_improvement"}
    files = {"predictions": prediction_path}
    if model_path.exists():
        files["model"] = model_path
    cell = seal_result_record(cell, files=files)
    atomic_write_json(directory / "record.json", cell)
    return cell


def _run_study(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan_bytes = plan_path.read_bytes()
    declared = json.loads(plan_bytes)
    if declared.get("schema") != "n2p3_bnci_continuous_study/1":
        raise ValueError("unsupported BNCI continuous study plan")
    records, budgets = declared["records"], tuple(declared["budgets"])
    if not records or len({row["subject"] for row in records}) != len(records):
        raise ValueError("explicit unique source records are required")
    if any(row["subject"] not in {f"A{i:02d}" for i in range(1, 9)} for row in records):
        raise ValueError("BNCI study has unknown participant")
    if not budgets or any(type(k) is not int or k <= 0 for k in budgets) or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("positive increasing calibration budgets are required")
    evidence_scope = declared["evidence_scope"]
    if evidence_scope not in {"public_development", "software_correctness"}:
        raise ValueError("research route is not a confirmation experiment")
    threads = declared["cpu_threads"]
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("explicit bounded CPU threads are required")
    plan = ContinuousInputPlan(**declared["continuous_input"])
    _assert_source_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "plan.json", declared)
    source_records = {}
    for name, content in _EXECUTING_BYTES.items():
        path = output / "executed_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source_records[name] = hashlib.sha256(content).hexdigest()
    source_digest = semantic_sha256(source_records)
    atomic_write_json(output / "execution_source.json", {"files": source_records, "sha256": source_digest})
    report = {"schema": "n2p3_bnci_continuous_study_result/1", "status": "running",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "execution_source_sha256": source_digest,
        "evidence_scope": evidence_scope, "product_confirmation": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "mne": mne.__version__, "cpu_threads": threads},
        "requested_subjects": [row["subject"] for row in records], "requested_Ks": list(budgets),
        "requested_cell_grid": [{"subject": row["subject"], "K": k} for row in records for k in budgets],
        "cells": [], "subject_failures": [], "claim": "diagnostic_route_only_no_comparator_superiority_claim"}
    atomic_write_json(output / "result.json", report)
    with cpu_thread_budget(threads):
        for row in records:
            _assert_source_unchanged()
            source = Path(row["path"])
            source = source if source.is_absolute() else plan_path.resolve().parent / source
            if source.name != row["subject"] + ".mat":
                raise ValueError("subject and physical MAT name disagree")
            directory = output / row["subject"]
            directory.mkdir()
            # Integrity errors abort: a changed source must never be recast as a weak participant.
            views = prepare_bnci_continuous(source, source_sha256=row["sha256"], budgets=budgets, plan=plan)
            atomic_write_json(directory / "scopes.json", views.ledger)
            for budget in sorted(set(budgets) | set(views.calibration)):
                if budget not in views.calibration:
                    report["cells"].append({"subject": row["subject"], "K": budget,
                        "status": "protocol_unavailable", "reason": views.ledger["adjustments"].get(str(budget)),
                        "is_exact_requested_K": True, "fallback_Ks": sorted(set(views.calibration) - set(budgets))})
                else:
                    cell = _cell(views, budget, plan, directory / f"K{budget}", source_digest,
                                 evidence_scope=evidence_scope)
                    cell["is_exact_requested_K"] = budget in budgets
                    # Cell is already sealed; the study annotation is separate from the physical cell file.
                    cell = seal_result_record(cell, files={role: directory / f"K{budget}" / item["name"]
                        for role, item in cell["result_integrity"]["files"].items()})
                    atomic_write_json(directory / f"K{budget}" / "record.json", cell)
                    report["cells"].append(cell)
                atomic_write_json(output / "result.json", report)
                print(json.dumps({"subject": row["subject"], "K": budget, "status": report["cells"][-1]["status"],
                                  "hit": report["cells"][-1].get("requested_hit_all")}), flush=True)
            del views
            gc.collect()
    _assert_source_unchanged()
    summaries = {}
    for budget in budgets:
        cells = [cell for cell in report["cells"] if cell["K"] == budget]
        completed = [cell for cell in cells if "requested_hit_all" in cell]
        summaries[str(budget)] = {"requested_subjects": len(records), "accounted_subjects": len(cells),
            "scored_subjects": len(completed), "participant_macro_hit_all": float(np.mean([
                cell.get("requested_hit_all", 0.) for cell in cells])),
            "participant_macro_coverage": float(np.mean([cell.get("coverage", 0.) for cell in cells])),
            "conditional_participant_macro_nll": (float(np.mean([cell["scored_nll_mean"] for cell in completed
                if cell["scored_nll_mean"] is not None])) if any(cell["scored_nll_mean"] is not None for cell in completed) else None)}
    report.update(status="completed", by_K=summaries, implementation_unchanged=True,
                  requested_cell_outcomes=_request_outcomes(report))
    report = seal_result_record(report, files={"execution_source": output / "execution_source.json", "plan": output / "plan.json"})
    atomic_write_json(output / "result.json", report)
    return report


def _request_outcomes(report):
    observed = {(cell["subject"], cell["K"]): cell for cell in report["cells"]}
    return [{**row, "status": observed[(row["subject"], row["K"])]["status"]
             if (row["subject"], row["K"]) in observed else "not_executed_due_to_run_failure"}
            for row in report["requested_cell_grid"]]


def run_study(plan_path, output):
    """Close interrupted/failed runs while preserving every requested cell."""
    output = Path(output)
    existed = output.exists()
    try:
        return _run_study(plan_path, output)
    except BaseException as error:
        result = output / "result.json"
        # Never overwrite a prior run rejected by exist_ok=False.
        if not existed and result.is_file():
            report = json.loads(result.read_text(encoding="utf-8"))
            if report.get("status") == "running":
                report.update(status="failed", requested_cell_outcomes=_request_outcomes(report),
                    run_failure={"type": type(error).__name__, "message": str(error),
                                 "unresolved_decision_counts": "not_invented_when_raw_admission_fails"})
                report = seal_result_record(report, files={"execution_source": output / "execution_source.json",
                                                         "plan": output / "plan.json"})
                atomic_write_json(result, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_study(args.plan, args.output)


if __name__ == "__main__":
    main()
