"""Execute the fixed full calibration-only operator study, preserving every cell."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def executing_files():
    paths = [*ROOT.joinpath("src").rglob("*.py"), Path(__file__),
             Path(__file__).with_name("continuous_model.py"), Path(__file__).with_name("selection_erp_model.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def selection_views(row, plan):
    payload = Path(row["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("original raw bytes changed")
    raw = read_bnci2014_008_candidate_record(io.BytesIO(payload),
        source_relative_path=row["subject"] + ".mat", source_sha256=row["sha256"])
    del payload
    width = round(plan["response_seconds"] * SFREQ)
    pre = round(plan["pre_roll_seconds"] * SFREQ)
    authorized_stop = int(raw.flash_sample[plan["authorized_labels_per_participant"]])
    groups, receipt = [], []
    for group in np.unique(raw.selection_index):
        indices = np.flatnonzero(raw.selection_index == group)
        if len(indices) != 120 or int(indices[-1]) >= 720:
            continue
        start, stop = int(raw.flash_sample[indices[0]]) - pre, int(raw.flash_sample[indices[-1]]) + width
        if start < 0 or stop > authorized_stop:
            raise ValueError("group outside legal C800 support")
        if indices[0] > 0 and raw.flash_sample[indices[0]-1] + width > start:
            raise ValueError("unknown previous response overlaps group")
        if indices[-1] + 1 < len(raw.flash_sample) and raw.flash_sample[indices[-1]+1] < stop:
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
        receipt.append({"record_id": record_id, "raw_scope": [start, stop], "label_indices": indices.tolist(),
            "input_sha256": array_digest(samples, onsets, y), "prediction_signal_sha256": array_digest(samples, onsets, member)})
    if len(groups) != 6:
        raise ValueError("all six predeclared complete groups are required")
    for left, right in zip(receipt[:-1], receipt[1:], strict=True):
        if left["raw_scope"][1] > right["raw_scope"][0]:
            raise ValueError("group scopes overlap")
    return groups, {"raw_sha256": row["sha256"], "C800_stop_sample": authorized_stop,
        "used_label_count": 720, "unused_authorized_labels": 80, "groups": receipt,
        "filtering_scope": "each_independent_selection", "held_labels_enter_fit": False,
        "K200_independent_group_CV": "not_estimable_one_complete_selection"}


def run_subject(row, plan, directory, expected_source):
    directory = Path(directory)
    directory.mkdir()
    if executing_files() != expected_source:
        raise RuntimeError("implementation differs from frozen execution")
    groups, ledger = selection_views(row, plan)
    atomic_write_json(directory / "scopes.json", ledger)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    sos = design_continuous_sos(SFREQ, l_freq=spec.l_freq, h_freq=spec.h_freq, order=spec.filter_order,
        notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), pad)
    basis = smooth_erp_basis(SFREQ, plan["response_seconds"], plan["basis_rank"])
    cells = []
    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for fold in range(6):
            held, member, truth = groups[fold]
            training = [record for index, (record, _, _) in enumerate(groups) if index != fold]
            for arm in plan["arms"]:
                began = time.perf_counter()
                cell_dir = directory / f"fold{fold}_{arm['name']}"
                cell_dir.mkdir()
                cell = {"subject": row["subject"], "fold": fold, "arm": arm["name"], "status": "running",
                    "family": arm.get("family", "continuous_gls"),
                    "evidence_scope": plan["evidence_scope"], "training_groups": 5, "training_labels": 600,
                    "authorized_C_labels": 800, "scored_physical_decisions": 1, "K_is_not_800_fit": True,
                    "held_record_id": held.record_id, "scope_ledger_sha256": semantic_sha256(ledger),
                    "requested": 1, "correct": False, "nll": None, "error": None}
                files = {}
                try:
                    fitted_basis = basis
                    if arm.get("basis_boundary") == "smooth_zero_endpoints":
                        window = np.sin(np.linspace(0, np.pi, len(basis)))**2
                        window[[0, -1]] = 0
                        fitted_basis = basis * window[:, None]
                        fitted_basis *= np.linalg.norm(basis, axis=0) / np.linalg.norm(fitted_basis, axis=0)
                    if arm.get("family") is None:
                        model = OperatorGLS.fit(training, basis=fitted_basis, sfreq=SFREQ, channels=BNCI2014_008_CHANNELS,
                            candidate_ids=CANDIDATES, order=arm["order"], noise_method=arm["noise_method"],
                            operator=operator if arm["filter"] else None, filter_design=arm["filter_design"], ridge=plan["ridge"],
                            feasible_iterations=arm.get("feasible_iterations", 1),
                            channel_weighted_noise=arm.get("channel_weighted_noise", False),
                            spatial_shrinkage=arm.get("spatial_shrinkage", "pooled_lw"),
                            kernel_prior_scale=arm.get("kernel_prior_scale", "innovation"),
                            startup_mode=arm.get("startup_mode", "stationary"),
                            implementation_sha256=expected_source["doc/assets/operator_research_20261004/continuous_model.py"])
                    else:
                        model = SelectionERP.fit(training, family=arm["family"], baseline_mode=arm["baseline_mode"],
                            sfreq=SFREQ, channels=BNCI2014_008_CHANNELS, candidate_ids=CANDIDATES,
                            operator=operator, expected_label_count=600)
                    fit_seconds = time.perf_counter() - began
                    files["model"] = cell_dir / "model.npz"
                    model.save(files["model"])
                    restored = type(model).load(files["model"])
                    options = dict(record_id=held.record_id, sfreq=SFREQ, channels=BNCI2014_008_CHANNELS, candidate_ids=CANDIDATES)
                    result = model.score(held.samples, held.onsets, member, **options)
                    replay = restored.score(held.samples, held.onsets, member, **options)
                    np.testing.assert_array_equal(result.log_evidence, replay.log_evidence)
                    # A truth switch is a scorer operation; prediction only receives samples/schedule.
                    nll = float(logsumexp(result.log_evidence) - result.log_evidence[CANDIDATES.index(truth)])
                    prediction = {**result.record(), "truth_scorer_only": truth, "nll": nll,
                                  "operator": None if model.operator is None else model.operator.record()}
                    files["prediction"] = cell_dir / "prediction.json"
                    atomic_write_json(files["prediction"], prediction)
                    cell.update(status="completed", correct=result.winner == truth, nll=nll,
                        winner=result.winner, saved_reload_exact=True, fit_seconds=fit_seconds,
                        noise=model.noise.record() if isinstance(model, OperatorGLS) else None,
                        fit=dict(model.provenance),
                        input_sha256=result.input_sha256)
                except (ValueError, RuntimeError, np.linalg.LinAlgError, AssertionError) as exc:
                    cell.update(status="failed", error=str(exc))
                cell["elapsed_seconds"] = time.perf_counter() - began
                cell = seal_result_record(cell, files=files)
                atomic_write_json(cell_dir / "record.json", cell)
                cells.append(cell)
                atomic_write_json(directory / "cells.json", cells)
                print(json.dumps({key: cell[key] for key in ("subject", "fold", "arm", "status", "nll")}), flush=True)
    if executing_files() != expected_source:
        raise RuntimeError("implementation changed during subject execution")
    del groups
    gc.collect()
    # Frozen in-process provenance is intentionally not pickle-mutable.
    # Cross-process result transport carries the same already-serialized JSON.
    return json.loads(json.dumps(cells))


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan["schema"] != "n2p3_operator_calibration_group_study/1":
        raise ValueError("unsupported fixed study")
    output.mkdir(parents=True, exist_ok=False)
    source = executing_files()
    for path, digest in source.items():
        data = (ROOT / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("source changed during snapshot")
        target = output / "executed_source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    atomic_write_json(output / "source.json", {"files": source, "digest": semantic_sha256(source)})
    (output / "plan.json").write_bytes(plan_bytes)
    report = {"status": "running", "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "source_sha256": semantic_sha256(source), "requested_cells": len(plan["records"])*6*len(plan["arms"]),
        "cells": [], "subject_failures": [], "evidence_scope": plan["evidence_scope"]}
    atomic_write_json(output / "result.json", report)
    try:
        with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
            futures = {pool.submit(run_subject, row, plan, output / row["subject"], source): row for row in plan["records"]}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    report["cells"].extend(future.result())
                except Exception as exc:
                    report["subject_failures"].append({"subject": row["subject"], "error": repr(exc)})
                    partial_path = output / row["subject"] / "cells.json"
                    partial = json.loads(partial_path.read_text()) if partial_path.exists() else []
                    done = {(cell["fold"], cell["arm"]) for cell in partial}
                    report["cells"].extend(partial)
                    for fold in range(6):
                        for arm in plan["arms"]:
                            if (fold, arm["name"]) not in done:
                                report["cells"].append({"subject": row["subject"], "fold": fold, "arm": arm["name"],
                                    "status": "not_executed", "error": repr(exc), "correct": False, "nll": None})
                atomic_write_json(output / "result.json", report)
    except Exception as exc:
        report["subject_failures"].append({"scope": "executor_start_or_shutdown", "error": repr(exc)})
        done = {(cell["subject"], cell["fold"], cell["arm"]) for cell in report["cells"]}
        for row in plan["records"]:
            for fold in range(6):
                for arm in plan["arms"]:
                    key = (row["subject"], fold, arm["name"])
                    if key not in done:
                        report["cells"].append({"subject": row["subject"], "fold": fold, "arm": arm["name"],
                            "status": "not_executed", "error": repr(exc), "correct": False, "nll": None})
        atomic_write_json(output / "result.json", report)
    if executing_files() != source:
        report["subject_failures"].append({"error": "source_changed"})
    report["status"] = "failed" if report["subject_failures"] else "completed"
    report["cells"].sort(key=lambda cell: (cell["subject"], cell["fold"], cell["arm"]))
    report = seal_result_record(report, files={"source": output / "source.json", "plan": output / "plan.json"})
    atomic_write_json(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    result = run(options.plan, options.output)
    if result["status"] != "completed":
        raise SystemExit(1)
