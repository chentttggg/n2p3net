"""Read-only standards review; all deliberate changes target review-local copies."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.signal import butter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
PROTO = ROOT / "doc/assets/operator_research_20261004"
sys.path.insert(0, str(PROTO))

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from transfer.continuous_decision import CalibrationRecord, smooth_erp_basis  # noqa: E402
from transfer.noise_model import array_digest, metadata_digest  # noqa: E402


def digest(data):
    return hashlib.sha256(data).hexdigest()


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rewrite_model(source, destination, edit_meta=None, edit_state=None):
    with np.load(source, allow_pickle=False) as z:
        state = {key: z[key] for key in z.files if key != "metadata"}
        metadata = json.loads(str(z["metadata"]))
    if edit_meta:
        edit_meta(metadata)
    if edit_state:
        edit_state(state)
        metadata["state_sha256"] = array_digest(*(state[key] for key in
            ("basis", "mean", "row_covariance", "channel_covariance")))
    metadata.pop("artifact_sha256")
    metadata["artifact_sha256"] = metadata_digest(metadata)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **state, metadata=np.asarray(json.dumps(metadata)))


def run():
    scope = ["src/data/finite_filter_operator.py", "src/transfer/temporal_noise.py",
             "tests/data/test_finite_filter_operator.py", "tests/transfer/test_temporal_noise.py",
             *["doc/assets/operator_research_20261004/" + name for name in
               ("continuous_model.py", "selection_erp_model.py", "run_calibration_study.py",
                "analyze_calibration_study.py", "calibration_group_plan_stage3.json")]]
    identity = {name: digest((ROOT / name).read_bytes()) for name in scope}
    baseline = ROOT / "frozen/system_implementation_20261004_after"
    manifest = json.loads((baseline / "source.manifest.json").read_text())
    archive = baseline / manifest["archive"]
    assert digest(archive.read_bytes()) == manifest["archive_sha256"]
    with tarfile.open(archive, "r:gz") as frozen:
        members = set(frozen.getnames())
    results = {"scope_source_sha256": identity, "baseline_archive_sha256": manifest["archive_sha256"],
               "files_absent_from_baseline": [name for name in scope if name not in members],
               "evidence_scope": "software_correctness_independent_review"}
    copy = OUT / "continuous_model_copy.py"
    original = (PROTO / "continuous_model.py").read_bytes()
    copy.write_bytes(original)
    module = import_path("standards_review_continuous_model", copy)
    rng = np.random.default_rng(144)
    onsets = np.array([15, 55, 95, 135, 175])
    labels = np.array([False, True, False, True, False])
    records = [CalibrationRecord(f"C{i}", rng.normal(size=(240, 2)), onsets, labels) for i in range(2)]
    basis = smooth_erp_basis(100., response_seconds=.2, n_basis=2)
    op = FiniteFilterOperator(tuple(map(tuple, butter(2, .3, output="sos"))), 12)
    options = dict(basis=basis, sfreq=100., channels=("C3", "C4"), candidate_ids=("a", "b"),
                   order=1, operator=op)
    with threadpool_limits(1):
        model = module.OperatorGLS.fit(records, **options)
        pristine = OUT / "pristine.npz"
        model.save(pristine)
        member = np.column_stack([labels, ~labels])
        score_options = dict(record_id="C0", sfreq=100., channels=("C3", "C4"), candidate_ids=("a", "b"))
        try:
            model.score(records[0].samples, onsets, member, **score_options)
        except ValueError:
            original_guard = True
        else:
            original_guard = False
        mutated = OUT / "empty_training_ledger.npz"
        rewrite_model(pristine, mutated, lambda meta: meta["provenance"].update(records=[]))
        try:
            loaded = module.OperatorGLS.load(mutated)
        except ValueError as exc:
            results["missing_training_ledger_closed"] = {
                "original_training_record_rejected": original_guard,
                "load_rejected": str(exc)}
        else:
            replay = loaded.score(records[0].samples, onsets, member, **score_options)
            results["missing_training_ledger_accepted"] = {
                "original_training_record_rejected": original_guard,
                "loaded_records": list(loaded.provenance["records"]),
                "same_training_signal_scored": replay.log_evidence.tolist()}
        mutated = OUT / "wrong_operator_semantics.npz"
        rewrite_model(pristine, mutated, lambda meta: meta["operator"].update(edge_rule="zero_extension"))
        try:
            loaded = module.OperatorGLS.load(mutated)
        except ValueError as exc:
            results["operator_declaration_closed"] = {
                "saved_edge_rule": "zero_extension", "load_rejected": str(exc)}
        else:
            results["operator_declaration_silently_rewritten"] = {
                "saved_edge_rule": "zero_extension", "loaded_edge_rule": loaded.operator.record()["edge_rule"]}
        mutated = OUT / "asymmetric_covariance.npz"
        def asymmetry(state):
            state["channel_covariance"][0, 1] += 10.
        rewrite_model(pristine, mutated, edit_state=asymmetry)
        try:
            loaded = module.OperatorGLS.load(mutated)
        except ValueError as exc:
            results["asymmetric_covariance_closed"] = {
                "load_rejected": str(exc), "mutated_upper_minus_lower": 10.0}
        else:
            score = loaded.score(rng.normal(size=(240, 2)), onsets, member,
                                 **{**score_options, "record_id": "held"})
            results["asymmetric_covariance_accepted"] = {
                "asymmetry": float(np.max(np.abs(loaded.channel_covariance - loaded.channel_covariance.T))),
                "prediction_finite": bool(np.isfinite(score.log_evidence).all())}

        # Never alter the live source. The loaded copy still executes old fit,
        # but its recorded provenance hashes an explicitly non-executed version.
        changed = original.decode().replace("        return cls(basis, mean",
            '        raise RuntimeError("UPDATED_FILE_MUST_NOT_EXECUTE")\n        return cls(basis, mean', 1).encode()
        assert changed != original
        copy.write_bytes(changed)
        stale = module.OperatorGLS.fit(records, **options)
        results["loaded_source_vs_disk_identity"] = {
            "loaded_sha256": digest(original), "changed_disk_sha256": digest(changed),
            "claimed_sha256": stale.provenance["implementation_sha256"],
            "changed_disk_fit_would_raise": True, "loaded_old_fit_succeeded": True}

    # Completed ERP models are read-only. Exercise precisely the analyzer's
    # unconditional continuous-model load dispatch on a real mature artifact.
    stage3 = ROOT / "experiments/runs/operator_research_20261004/calibration_groups_stage3"
    for record_path in sorted(stage3.glob("*/fold*_waveform_baseline/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != "completed":
            continue
        path = record_path.with_name("model.npz")
        payload = path.read_bytes()
        try:
            module.OperatorGLS.load(io.BytesIO(payload))
        except Exception as exc:
            results["wrong_loader_negative_control"] = {"model": str(path.relative_to(ROOT)),
                "family": record["family"], "sha256": digest(payload),
                "exception_type": type(exc).__name__, "message": str(exc)}
        break

    # Inject executor setup failure in this review process only; creates no
    # workers and writes only the review-local output directory.
    runner = import_path("standards_review_runner", PROTO / "run_calibration_study.py")
    class BrokenExecutor:
        def __init__(self, **kwargs):
            raise OSError("review-only executor startup failure")
    runner.ProcessPoolExecutor = BrokenExecutor
    failed_output = OUT / "executor_failure"
    try:
        runner.run(PROTO / "calibration_group_plan_stage3.json", failed_output)
    except OSError as exc:
        failure = json.loads((failed_output / "result.json").read_text(encoding="utf-8"))
        results["executor_setup_failure_state"] = {
            "raised": str(exc), "persisted_status": failure["status"],
            "requested_cells": failure["requested_cells"], "materialized_cells": len(failure["cells"])}
    assert identity == {name: digest((ROOT / name).read_bytes()) for name in scope}
    results["live_scope_unchanged_during_probes"] = True
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
