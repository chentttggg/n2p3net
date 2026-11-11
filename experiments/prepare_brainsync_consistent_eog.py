"""Prepare fresh matched basic/QC calibration doses and fixed BrainSync tests."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.brainsync import IndependentRecordingScope, load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.eog_preprocessing import EOGPreprocessingPolicy  # noqa: E402
from data.epochs import load_preprocessing_spec, save_epoch_dataset  # noqa: E402
from data.ocular import PRESERVE_ACQUIRED_EOG, save_eog_regression  # noqa: E402
from data.source_qc import continuous_parameters  # noqa: E402
from data.source_records import SIX, file_sha256  # noqa: E402
from experiments.fit_brainsync_continuous_qc import fit_brainsync_continuous_manifest  # noqa: E402
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def fit_target_preprocessing(roots, *, policy, preprocessing, output):
    """Only C files enter any EOG coefficient, threshold or state-envelope fit."""
    with ExitStack() as leases:
        sessions = []
        for root in roots:
            session = validate_brainsync_bids_session(root)
            leases.callback(session.bids.close)
            sessions.append(session)
        if len({semantic_sha256(dict(s.bids.eog_derivation)) for s in sessions}) != 1:
            raise ValueError("calibration EOG declarations differ; September placement/reference must be resolved explicitly")
        rates = {float(s.bids.sample_rate_hz) for s in sessions}
        if len(rates) != 1:
            raise ValueError("target continuous QC requires one native sample rate")
        spec, recovery = continuous_parameters(policy, next(iter(rates)))
        intervals = {s.session_id: [[0, int(s.manifest["recording"].get("clock", {}).get(
            "acquired_samples", round(s.bids.duration_seconds * s.bids.sample_rate_hz)))]] for s in sessions}
        manifest = {"schema": "brainsync_continuous_qc_calibration/1", "preprocessing": preprocessing.record(),
            "spec": asdict(spec), "epoch_recovery": recovery,
            "recovery_warmup_rule": policy.record()["continuous_qc"]["recovery_warmup_rule"],
            "threshold_aggregation": policy.record()["continuous_qc"]["threshold_aggregation"],
            "sessions": [{"session_dir": str(s.root), "clean_intervals_samples": intervals[s.session_id],
                "clean_interval_evidence": "complete predeclared C recording; label-blind robust thresholds; original BIDS exclusions retained"}
                         for s in sessions]}
        manifest_path = output / "continuous_manifest.json"
        atomic_write_json(manifest_path, manifest)
        continuous = fit_brainsync_continuous_manifest(manifest_path)
        atomic_write_json(output / "continuous.json", {"model": continuous.record(), "digest": semantic_sha256(continuous.record())})
        ocular = fit_brainsync_eog_sessions(roots, reference_channel="Cz", continuous_qc_model=continuous,
            eog_reference_policy=PRESERVE_ACQUIRED_EOG, fit_intervals_by_session=intervals, preprocessing=preprocessing,
            fit_domain=policy.fit_domain)
        save_eog_regression(output / "ocular.json", ocular)
        return continuous, ocular


def prepare(*, previous_plan, quality_manifest, output, doses, pipelines):
    original = json.loads(Path(previous_plan).read_text(encoding="utf-8"))
    quality = json.loads(Path(quality_manifest).read_text(encoding="utf-8"))
    by_id = {r["directory_id"]: Path(r["input_root"]) for r in quality["sessions"]}
    entries = {r["session_id"]: r for r in original["rows"]}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = load_preprocessing_spec(ROOT / "configs/brainsync_bidirectional_preprocessing.json")
    plans = {"schema": "n2p3_consistent_EOG_target_preparation/1", "status": "running", "cells": {}}
    for dose in doses:
        selected = original["calibration_order"][:dose]
        if len(selected) != dose or set(selected) & set(original["test_sessions"]):
            raise ValueError("invalid fixed calibration/test roster")
        sessions = selected + original["test_sessions"]
        scope = IndependentRecordingScope(tuple(selected), tuple(original["test_sessions"]))
        for pipeline in pipelines:
            key = f"{pipeline}_K{dose}"
            folder = output / key
            folder.mkdir()
            policy = (EOGPreprocessingPolicy.corrected() if pipeline == "QC" else
                EOGPreprocessingPolicy("basic_eog_ols_v1", False, "record_unavailable"))
            cell = {"status": "preparing", "policy": policy.record(), "calibration_sessions": selected,
                "test_sessions": original["test_sessions"], "rows": []}
            try:
                continuous = ocular = None
                if pipeline == "QC":
                    continuous, ocular = fit_target_preprocessing([by_id[s] for s in selected], policy=policy,
                        preprocessing=spec, output=folder)
                for session in sessions:
                    data = load_brainsync_session(by_id[session], preprocessing=spec,
                        channels=SIX if pipeline == "QC" else None, ocular_model=ocular,
                        continuous_qc_model=continuous, recording_scope=scope if pipeline == "QC" else None)
                    if pipeline == "basic":
                        data = adapt_observed_montage(data, SIX, reference_channel="Cz")
                    else:
                        data = adapt_observed_montage(data, SIX, preserve_reference_label=True)
                    data.provenance["artifact_preprocessing_policy"] = policy.record()
                    path = save_epoch_dataset(folder / (session + ".npz"), data, compressed=False, exclusive=True)
                    cell["rows"].append({"session_id": session, "target": entries[session]["target"],
                        "path": str(path), "sha256": file_sha256(path), "n_epochs": data.n_epochs,
                        "scheduled": data.event_timeline.n_events,
                        "role": "C" if session in selected else "fixed_future_test"})
                cell["status"] = "prepared"
            except Exception as error:
                cell.update(status="unavailable_or_failed", failure={"type": type(error).__name__, "message": str(error)})
            plans["cells"][key] = cell
            atomic_write_json(output / "prepared.json", plans)
            print(key, cell["status"], cell.get("failure", {}).get("message", ""), flush=True)
    plans["status"] = "prepared" if all(c["status"] == "prepared" for c in plans["cells"].values()) else "partial_inputs"
    atomic_write_json(output / "prepared.json", plans)
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doses", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--pipelines", nargs="+", choices=["basic", "QC"], default=["basic", "QC"])
    args = parser.parse_args(argv)
    prepare(previous_plan=args.previous_plan, quality_manifest=args.quality_manifest,
        output=args.output, doses=args.doses, pipelines=args.pipelines)


if __name__ == "__main__":
    main()
