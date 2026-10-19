"""E2: AR order vs kernel prior scale — innovation_relative vs marginal_relative.

Runs a fixed declared matrix on the same eight BNCI participants and 48 folds
used in operator_research_20261004. Four arms: AR1/AR8 × innovation/marginal
prior scale, plus the declared Toeplitz and xDAWN ERP anchors. Each arm runs
the full ridge_relative grid; the primary cell summary uses ridge=1 and the
complete grid is retained in `ridge_grid_results`.

Operator declaration (deliberate, not shared): the AR arms consume the raw
distributed MAT (`filter=false`), matching the Phase 4 "raw AR8" baseline whose
NLL overconfidence this study is trying to explain. The Toeplitz/xDAWN anchors
consume the project bidirectional filter. The AR-vs-AR contrast is therefore a
controlled comparison (same operator across AR arms); the AR-vs-anchor contrast
differs in BOTH operator and method and is reported as a reference only, never
as a controlled effect.

The held selection never touches parameter selection; no winner is chosen from
held data. Gauge and unit counter-examples are verified before any data is
read so a failed math check aborts the run immediately.

Output directory schema: n2p3_e2_prior_scale_study/1
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "doc" / "assets" / "operator_research_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ASSETS))

import mne  # noqa: E402
import numpy as np  # noqa: E402
from continuous_model import OperatorGLS  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from selection_erp_model import SelectionERP  # noqa: E402

from data.bnci2014_008_candidate import (  # noqa: E402
    BNCI2014_008_CHANNELS,
    read_bnci2014_008_candidate_record,
)
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT  # noqa: E402
from data.epochs import preprocessing_spec_from_contract  # noqa: E402
from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from data.preprocess import design_continuous_sos  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.candidate_decision import row_column_target  # noqa: E402
from transfer.continuous_decision import CalibrationRecord, smooth_erp_basis  # noqa: E402
from transfer.noise_model import array_digest  # noqa: E402
from transfer.raw_continuous import CANDIDATES, SFREQ, _membership  # noqa: E402
from transfer.result_reuse import seal_result_record  # noqa: E402

# Import the boundary research math module for gauge/unit verification.
_RESEARCH_PATH = ROOT / "experiments" / "boundary_state_research.py"
_spec = importlib.util.spec_from_file_location("boundary_state_research", _RESEARCH_PATH)
_boundary = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _boundary
_spec.loader.exec_module(_boundary)
ScaleInvariantContinuousModel = _boundary.ScaleInvariantContinuousModel


# ---------------------------------------------------------------------------
# Math verification — must pass before any data is read
# ---------------------------------------------------------------------------

def _verify_gauge_and_unit_invariance():
    """Reproduce the gauge and unit counter-examples from the design record.

    Gauge:  K -> alpha*K, Sigma -> Sigma/alpha requires lambda -> lambda/alpha.
            Without that sync, log BF can shift by ~216.
    Unit:   Y -> s*Y, Sigma -> s^2*Sigma must leave lambda unchanged.
            Naive noise-scaled ridge shifts log BF by ~480.

    A failure here means the mathematical contract is broken and E2 must not run.
    """
    from scipy.linalg import toeplitz

    rng = np.random.default_rng(548734)
    n, m, rank = 60, 40, 4
    design = rng.normal(size=(n, rank))
    truth = rng.normal(size=(rank, 1))
    temporal = toeplitz(.7 ** np.arange(n))
    held_temporal = toeplitz(.7 ** np.arange(m))
    y = design @ truth + np.sqrt(.4) * np.linalg.cholesky(temporal) @ rng.normal(size=(n, 1))
    candidates = rng.normal(size=(3, m, rank))
    x = candidates[1] @ truth + np.sqrt(.4) * np.linalg.cholesky(held_temporal) @ rng.normal(size=(m, 1))
    spatial = np.array([[.4]])

    def bf(evidence):
        return evidence - evidence[0]

    # --- Gauge invariance (correct sync: lambda/alpha) ---
    ordinary = ScaleInvariantContinuousModel.fit_scale_invariant(design, y, temporal, spatial)
    gauged = ScaleInvariantContinuousModel.fit_scale_invariant(design, y, temporal, spatial, covariance_factor=200.)
    diff_correct = float(np.max(np.abs(bf(ordinary.log_evidence(x, candidates, held_temporal))
                                       - bf(gauged.log_evidence(x, candidates, held_temporal)))))
    if diff_correct > 1e-8:
        raise RuntimeError(f"Gauge invariance FAILED: max BF diff = {diff_correct:.6g} (must be < 1e-8)")

    # --- Negative control: wrong sync (lambda unchanged) must shift BF > 1 ---
    wrong_gauge = ScaleInvariantContinuousModel.fit_scale_invariant(
        design, y, temporal, spatial, ridge_relative=200., covariance_factor=200.)
    diff_wrong_gauge = float(np.max(np.abs(bf(wrong_gauge.log_evidence(x, candidates, held_temporal))
                                           - bf(ordinary.log_evidence(x, candidates, held_temporal)))))
    if diff_wrong_gauge <= 1:
        raise RuntimeError(f"Gauge negative control FAILED: max BF diff = {diff_wrong_gauge:.6g} (must be > 1)")

    # --- Voltage unit invariance (lambda must NOT scale with s^2) ---
    s = 1e6
    converted = ScaleInvariantContinuousModel.fit_scale_invariant(design, s * y, temporal, s**2 * spatial)
    diff_unit = float(np.max(np.abs(bf(converted.log_evidence(s * x, candidates, held_temporal))
                                    - bf(ordinary.log_evidence(x, candidates, held_temporal)))))
    if diff_unit > 1e-8:
        raise RuntimeError(f"Unit invariance FAILED: max BF diff = {diff_unit:.6g} (must be < 1e-8)")

    # --- Negative control: naive noise-scaled ridge must shift BF > 1 ---
    bad_unit = ScaleInvariantContinuousModel.fit_scale_invariant(
        design, s * y, temporal, s**2 * spatial, ridge_relative=s**2)
    diff_bad_unit = float(np.max(np.abs(bf(bad_unit.log_evidence(s * x, candidates, held_temporal))
                                        - bf(ordinary.log_evidence(x, candidates, held_temporal)))))
    if diff_bad_unit <= 1:
        raise RuntimeError(f"Unit negative control FAILED: max BF diff = {diff_bad_unit:.6g} (must be > 1)")

    return {
        "gauge_correct_max_bf_diff": diff_correct,
        "gauge_wrong_sync_max_bf_diff": diff_wrong_gauge,
        "unit_correct_max_bf_diff": diff_unit,
        "unit_naive_noise_scaled_max_bf_diff": diff_bad_unit,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Source snapshot
# ---------------------------------------------------------------------------

def _executing_files():
    paths = [
        *ROOT.joinpath("src").rglob("*.py"),
        Path(__file__),
        ASSETS / "continuous_model.py",
        ASSETS / "selection_erp_model.py",
        _RESEARCH_PATH,
    ]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


# ---------------------------------------------------------------------------
# Data loading — identical contract to run_calibration_study.selection_views
# ---------------------------------------------------------------------------

def _selection_views(row, plan):
    """Rebuild the six C800 groups from original MAT bytes."""
    payload = Path(row["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("original raw bytes changed")
    raw = read_bnci2014_008_candidate_record(
        io.BytesIO(payload),
        source_relative_path=row["subject"] + ".mat",
        source_sha256=row["sha256"],
    )
    del payload
    width = round(plan["response_seconds"] * SFREQ)
    pre = round(plan["pre_roll_seconds"] * SFREQ)
    authorized_stop = int(raw.flash_sample[plan["authorized_labels_per_participant"]])
    groups, receipt = [], []
    for group in np.unique(raw.selection_index):
        indices = np.flatnonzero(raw.selection_index == group)
        if len(indices) != 120 or int(indices[-1]) >= 720:
            continue
        start = int(raw.flash_sample[indices[0]]) - pre
        stop = int(raw.flash_sample[indices[-1]]) + width
        if start < 0 or stop > authorized_stop:
            raise ValueError("group outside legal C800 support")
        if indices[0] > 0 and raw.flash_sample[indices[0] - 1] + width > start:
            raise ValueError("unknown previous response overlaps group")
        if indices[-1] + 1 < len(raw.flash_sample) and raw.flash_sample[indices[-1] + 1] < stop:
            raise ValueError("next unknown stimulus overlaps group")
        codes = raw.candidate_id[indices]
        if not np.array_equal(np.bincount(codes, minlength=12), np.full(12, 10)):
            raise ValueError("incomplete physical selection")
        record_id = f"{row['subject']}:selection{int(group)}:{row['sha256']}"
        samples = np.asarray(raw.eeg_uv[start:stop], dtype=np.float64) * 1e-6
        onsets = np.array(raw.flash_sample[indices] - start, dtype=np.int64)
        y = np.array(raw.raw_is_target[indices], dtype=bool)
        member = _membership(codes)
        truth = row_column_target(int(raw.target_row[indices[0]]), int(raw.target_col[indices[0]]))
        if not np.array_equal(y, member[:, CANDIDATES.index(truth)]):
            raise ValueError("physical labels and scoring truth disagree")
        groups.append((CalibrationRecord(record_id, samples, onsets, y), member, truth))
        receipt.append({
            "record_id": record_id,
            "raw_scope": [start, stop],
            "label_indices": indices.tolist(),
            "input_sha256": array_digest(samples, onsets, y),
            "prediction_signal_sha256": array_digest(samples, onsets, member),
        })
    if len(groups) != 6:
        raise ValueError("all six predeclared complete groups are required")
    for left, right in zip(receipt[:-1], receipt[1:], strict=True):
        if left["raw_scope"][1] > right["raw_scope"][0]:
            raise ValueError("group scopes overlap")
    return groups, {
        "raw_sha256": row["sha256"],
        "C800_stop_sample": authorized_stop,
        "used_label_count": 720,
        "unused_authorized_labels": 80,
        "groups": receipt,
    }


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

def _run_subject(row, plan, directory, expected_source):
    directory = Path(directory)
    directory.mkdir()
    if _executing_files() != expected_source:
        raise RuntimeError("implementation differs from frozen execution snapshot")

    groups, ledger = _selection_views(row, plan)
    atomic_write_json(directory / "scopes.json", ledger)

    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    sos = design_continuous_sos(
        SFREQ, l_freq=spec.l_freq, h_freq=spec.h_freq, order=spec.filter_order,
        notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor,
    )
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), pad)
    basis = smooth_erp_basis(SFREQ, plan["response_seconds"], plan["basis_rank"])

    cells = []
    continuous_impl = hashlib.sha256((ASSETS / "continuous_model.py").read_bytes()).hexdigest()

    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for fold in range(6):
            held, member, truth = groups[fold]
            training = [record for idx, (record, _, _) in enumerate(groups) if idx != fold]

            for arm in plan["arms"]:
                began = time.perf_counter()
                cell_dir = directory / f"fold{fold}_{arm['name']}"
                cell_dir.mkdir()
                cell = {
                    "subject": row["subject"], "fold": fold, "arm": arm["name"],
                    "status": "running",
                    "family": arm.get("family", "continuous_gls"),
                    "evidence_scope": plan["evidence_scope"],
                    "training_groups": 5, "training_labels": 600,
                    "authorized_C_labels": 800, "scored_physical_decisions": 1,
                    "K_is_not_800_fit": True,
                    "held_record_id": held.record_id,
                    "scope_ledger_sha256": semantic_sha256(ledger),
                    "requested": 1, "correct": False, "nll": None, "error": None,
                    # E2-specific reporting fields
                    "ar_order": arm.get("order"),
                    "kernel_prior_scale": arm.get("kernel_prior_scale"),
                    "filter_declared": bool(arm.get("filter", False)),
                    "ridge_grid": list(plan["ridge_relative_grid"]),
                }
                files = {}
                try:
                    if arm.get("family") is not None:
                        # Anchor: SelectionERP (Toeplitz or xDAWN)
                        model = SelectionERP.fit(
                            training, family=arm["family"],
                            baseline_mode=arm["baseline_mode"],
                            sfreq=SFREQ, channels=BNCI2014_008_CHANNELS,
                            candidate_ids=CANDIDATES, operator=operator,
                            expected_label_count=600,
                        )
                        fit_seconds = time.perf_counter() - began
                        files["model"] = cell_dir / "model.npz"
                        model.save(files["model"])
                        restored = type(model).load(files["model"])
                        options = dict(record_id=held.record_id, sfreq=SFREQ,
                                       channels=BNCI2014_008_CHANNELS, candidate_ids=CANDIDATES)
                        result = model.score(held.samples, held.onsets, member, **options)
                        replay = restored.score(held.samples, held.onsets, member, **options)
                        np.testing.assert_array_equal(result.log_evidence, replay.log_evidence)
                        nll = float(logsumexp(result.log_evidence)
                                    - result.log_evidence[CANDIDATES.index(truth)])
                        prediction = {**result.record(), "truth_scorer_only": truth, "nll": nll,
                                      "operator": model.operator.record() if model.operator is not None else None}
                        files["prediction"] = cell_dir / "prediction.json"
                        atomic_write_json(files["prediction"], prediction)
                        cell.update(status="completed", correct=result.winner == truth, nll=nll,
                                    winner=result.winner, saved_reload_exact=True,
                                    fit_seconds=fit_seconds, input_sha256=result.input_sha256)
                    else:
                        # E2 main arms: AR1/AR8 × innovation/marginal, full ridge grid
                        use_operator = operator if arm.get("filter", False) else None
                        ridge_results = []
                        for ridge in plan["ridge_relative_grid"]:
                            t0 = time.perf_counter()
                            m = OperatorGLS.fit(
                                training, basis=basis, sfreq=SFREQ,
                                channels=BNCI2014_008_CHANNELS,
                                candidate_ids=CANDIDATES,
                                order=arm["order"],
                                noise_method=arm.get("noise_method", "burg"),
                                operator=use_operator,
                                filter_design=arm.get("filter_design", True),
                                ridge=ridge,
                                feasible_iterations=arm.get("feasible_iterations", 1),
                                channel_weighted_noise=arm.get("channel_weighted_noise", False),
                                spatial_shrinkage=arm.get("spatial_shrinkage", "pooled_lw"),
                                kernel_prior_scale=arm["kernel_prior_scale"],
                                startup_mode=arm.get("startup_mode", "stationary"),
                                implementation_sha256=continuous_impl,
                            )
                            fit_t = time.perf_counter() - t0
                            options = dict(record_id=held.record_id, sfreq=SFREQ,
                                           channels=BNCI2014_008_CHANNELS, candidate_ids=CANDIDATES)
                            res = m.score(held.samples, held.onsets, member, **options)
                            # Reload verification
                            mp = cell_dir / f"model_ridge{ridge:.4g}.npz"
                            m.save(mp)
                            rld = OperatorGLS.load(mp)
                            replay = rld.score(held.samples, held.onsets, member, **options)
                            np.testing.assert_array_equal(res.log_evidence, replay.log_evidence)
                            nll_r = float(logsumexp(res.log_evidence)
                                         - res.log_evidence[CANDIDATES.index(truth)])
                            # Diagnostic fields from TemporalNoise
                            noise_rec = m.noise.record() if hasattr(m.noise, "record") else {}
                            last_iteration = m.provenance["feasible_GLS_iterations"][-1]
                            ridge_results.append({
                                "ridge": ridge,
                                "correct": res.winner == truth,
                                "nll": nll_r,
                                "winner": res.winner,
                                "fit_seconds": fit_t,
                                "operator_declared": "project_bidirectional" if use_operator is not None else "raw_distributed_MAT",
                                "marginal_variance_per_unit_innovation":
                                    float(m.noise.marginal_variance_per_unit_innovation),
                                "effective_ridge": last_iteration["effective_ridge"],
                                "ar_coefficients": m.noise.coefficients.tolist(),
                                "noise_record": noise_rec,
                                "train_residual_diagnostics":
                                    last_iteration.get("residual_diagnostics", {}),
                                "saved_reload_exact": True,
                                "input_sha256": res.input_sha256,
                                "max_log_evidence_gap":
                                    float(np.max(res.log_evidence) - np.min(res.log_evidence)),
                                "max_probability": float(res.probabilities.max()),
                            })
                        # Use ridge=1 result as the primary cell summary (grid reported in full)
                        primary = next((r for r in ridge_results if r["ridge"] == 1.0), None)
                        if primary is None:
                            raise ValueError("declared ridge grid does not contain the primary ridge 1.0")
                        cell.update(
                            status="completed",
                            correct=primary["correct"],
                            nll=primary["nll"],
                            winner=primary.get("winner"),
                            fit_seconds=primary["fit_seconds"],
                            effective_ridge=primary["effective_ridge"],
                            ridge_grid_results=ridge_results,
                            saved_reload_exact=True,
                            input_sha256=primary["input_sha256"],
                        )
                        files["prediction"] = cell_dir / "primary_prediction.json"
                        atomic_write_json(files["prediction"], {"ridge_grid_results": ridge_results,
                                                                "truth_scorer_only": truth, "primary_ridge": 1.0})
                except (ValueError, RuntimeError, np.linalg.LinAlgError, AssertionError) as exc:
                    cell.update(status="failed", error=str(exc))
                cell["elapsed_seconds"] = time.perf_counter() - began
                try:
                    cell = seal_result_record(cell, files=files)
                except (OSError, TypeError, ValueError) as exc:
                    cell.update(status="failed", error=f"seal:{exc!r}")
                atomic_write_json(cell_dir / "record.json", cell)
                cells.append(cell)
                atomic_write_json(directory / "cells.json", cells)
                print(json.dumps({
                    k: cell[k] for k in ("subject", "fold", "arm", "status", "nll")
                }), flush=True)

    if _executing_files() != expected_source:
        raise RuntimeError("implementation changed during subject execution")
    del groups
    gc.collect()
    return json.loads(json.dumps(cells))


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(plan_path, output_path):
    plan_path, output_path = Path(plan_path), Path(output_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != "n2p3_e2_prior_scale_study/1":
        raise ValueError("plan schema must be n2p3_e2_prior_scale_study/1")

    output_path.mkdir(parents=True, exist_ok=False)

    # Math verification before touching any data
    print("Verifying gauge and unit invariance...", flush=True)
    math_result = _verify_gauge_and_unit_invariance()
    atomic_write_json(output_path / "math_verification.json", math_result)
    print(json.dumps({"math_check": "passed", **math_result}), flush=True)

    source = _executing_files()
    for rel_path, digest in source.items():
        data = (ROOT / rel_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"source changed during snapshot: {rel_path}")
        target = output_path / "executed_source" / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    atomic_write_json(output_path / "source.json",
                      {"files": source, "digest": semantic_sha256(source)})
    (output_path / "plan.json").write_bytes(plan_bytes)

    n_arms = len(plan["arms"])
    n_cells = len(plan["records"]) * 6 * n_arms
    report = {
        "schema": "n2p3_e2_prior_scale_study/1",
        "status": "running",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "source_sha256": semantic_sha256(source),
        "requested_cells": n_cells,
        "cells": [], "subject_failures": [],
        "evidence_scope": plan["evidence_scope"],
        "math_verification": math_result,
    }
    atomic_write_json(output_path / "result.json", report)

    try:
        with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
            futures = {
                pool.submit(_run_subject, row, plan,
                            output_path / row["subject"], source): row
                for row in plan["records"]
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    report["cells"].extend(future.result())
                except Exception as exc:
                    report["subject_failures"].append({"subject": row["subject"], "error": repr(exc)})
                    partial_path = output_path / row["subject"] / "cells.json"
                    partial = json.loads(partial_path.read_text()) if partial_path.exists() else []
                    done = {(c["fold"], c["arm"]) for c in partial}
                    report["cells"].extend(partial)
                    for fold in range(6):
                        for arm in plan["arms"]:
                            if (fold, arm["name"]) not in done:
                                report["cells"].append({
                                    "subject": row["subject"], "fold": fold,
                                    "arm": arm["name"], "status": "not_executed",
                                    "error": repr(exc), "correct": False, "nll": None,
                                })
                atomic_write_json(output_path / "result.json", report)
    except Exception as exc:
        report["subject_failures"].append({"scope": "executor", "error": repr(exc)})
        atomic_write_json(output_path / "result.json", report)

    if _executing_files() != source:
        report["subject_failures"].append({"error": "source_changed_during_run"})

    report["status"] = "failed" if report["subject_failures"] else "completed"
    report["cells"].sort(key=lambda c: (c["subject"], c["fold"], c["arm"]))
    report = seal_result_record(
        report,
        files={"source": output_path / "source.json", "plan": output_path / "plan.json"},
    )
    atomic_write_json(output_path / "result.json", report)
    return report


def _summarise(result):
    """Print participant-macro hit and mean NLL per arm to stdout."""
    from collections import defaultdict
    arm_hits = defaultdict(list)
    arm_nlls = defaultdict(list)
    for cell in result.get("cells", []):
        if cell["status"] != "completed":
            continue
        arm = cell["arm"]
        arm_hits[arm].append(int(cell["correct"]))
        if cell["nll"] is not None:
            arm_nlls[arm].append(cell["nll"])
    print("\n=== E2 summary (development, BNCI eight-participant C800) ===")
    for arm in sorted(arm_hits):
        hits = arm_hits[arm]
        nlls = arm_nlls[arm]
        print(f"  {arm:40s}  hit={sum(hits)/len(hits):.4f} ({sum(hits)}/{len(hits)})  "
              f"NLL={sum(nlls)/len(nlls):.4f}  n_nll={len(nlls)}")
    print("=== end summary ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    _summarise(result)
    if result["status"] != "completed":
        raise SystemExit(1)
