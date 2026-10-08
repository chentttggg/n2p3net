"""Audit B06 on an authorized BNCI native slice without fitting or test scoring.

The plan fixes physical sample scopes, actual extra SOS operators, independent
ERP challenge shapes and optional frozen correction bytes. The MAT adapter is
the authority for channels/rate/reference and events. All long context is a
retrospective diagnostic teacher; a deployed T still reads only its short crop.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from data.bnci2014_008_candidate import (  # noqa: E402
    BNCI2014_008_CHANNELS,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    read_bnci2014_008_candidate_record,
)
from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from data.preprocess import RawSampleScope  # noqa: E402
from research.boundary_observability import (  # noqa: E402
    BoundaryIdentifiabilityError,
    NativeAffineBoundaryMap,
    NativeBoundaryGrid,
    NativeBoundaryObservability,
)
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.noise_model import array_digest, require_sha256  # noqa: E402


def _load_plan(path):
    payload = path.read_bytes()
    plan = json.loads(payload)
    required = {"schema", "source", "calibration_scope", "selection_index", "context_scope", "short_scope", "output_scope",
                "observed_scopes", "available_at_sample", "short_operator", "context_operator",
                "injection_design_identity", "injections", "byte_budget"}
    if (not isinstance(plan, dict) or plan.get("schema") != "n2p3_boundary_observability_audit_plan/1"
            or not required <= set(plan) or set(plan) - required - {"frozen_correction"}):
        raise ValueError("audit plan has an unsupported or incomplete explicit B06 contract")
    source = plan["source"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueError("source must bind the actual MAT path and bytes")
    require_sha256(source["sha256"], "actual MAT source")
    RawSampleScope.from_record(plan["calibration_scope"])
    if type(plan["selection_index"]) is not int or not 0 <= plan["selection_index"] < 34:
        raise ValueError("audit needs an actual selection with a following stimulus boundary")
    return plan, hashlib.sha256(payload).hexdigest()


def _correction(plan, shape, base):
    specification = plan.get("frozen_correction")
    if specification is None:
        return np.zeros(shape), None, "predeclared_finite_short_operator_no_correction_anchor"
    if (not isinstance(specification, dict)
            or set(specification) != {"path", "sha256", "parameter_identity"}):
        raise ValueError("frozen correction requires exact artifact bytes and its producer identity")
    require_sha256(specification["sha256"], "frozen correction artifact")
    path = (base / specification["path"]).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != specification["sha256"]:
        raise ValueError("frozen correction bytes differ from the predeclared plan")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) not in ({"correction_matrix"}, {"correction_matrix", "correction_offset"}):
            raise ValueError("correction archive must contain only K and optional b; labels/oracles are forbidden")
        K = archive["correction_matrix"].copy()
        b = archive["correction_offset"].copy() if "correction_offset" in archive else None
    return K, b, specification["parameter_identity"]


def _injections(plan, grid):
    designs = plan["injections"]
    if not isinstance(designs, list) or not designs:
        raise ValueError("predeclared independent ERP morphology challenges are required")
    time = np.arange(*grid.context_scope) / grid.sample_rate_hz
    output = {}
    required = {"identity", "peak_sample", "width_seconds", "amplitude_volts", "channel_weights"}
    for design in designs:
        if not isinstance(design, dict) or set(design) != required:
            raise ValueError("ERP challenges must declare morphology on the actual native physical grid")
        name, peak = design["identity"], design["peak_sample"]
        width, amplitude = design["width_seconds"], design["amplitude_volts"]
        if (not isinstance(name, str) or not name.strip() or name in output
                or type(peak) is not int or not grid.context_scope[0] <= peak < grid.context_scope[1]
                or not np.isfinite([width, amplitude]).all() or width <= 0 or amplitude == 0):
            raise ValueError("independent challenge requires unique identity, actual peak and nonzero morphology")
        weights = design["channel_weights"]
        if (not isinstance(weights, dict) or set(weights) != set(grid.channel_names)
                or not np.isfinite(list(weights.values())).all() or not any(weights.values())):
            raise ValueError("challenge must declare weights for each authoritative physical channel")
        shape = np.exp(-.5 * ((time - peak / grid.sample_rate_hz) / width)**2)
        output[name] = amplitude * shape[:, None] * np.array([weights[channel] for channel in grid.channel_names])
    return output


def _validate_calibration_events(scope, flash_samples, selection_rows, n_samples):
    if (scope.stop_sample > n_samples or scope.event_indices[-1] >= len(flash_samples)
            or not set(selection_rows.tolist()) <= set(scope.event_indices)):
        raise ValueError("actual selection rows must belong to the declared calibration contract")
    authorized_times = flash_samples[list(scope.event_indices)]
    if np.any(authorized_times < scope.start_sample) or np.any(authorized_times >= scope.stop_sample):
        raise ValueError("declared calibration events lie outside their actual native sample scope")


def _run(plan, plan_path, destination, snapshot_sha256, plan_sha256,
         *, snapshot_manifest_sha256):
    require_sha256(snapshot_sha256, "actual executing source archive")
    require_sha256(snapshot_manifest_sha256, "actual executing source manifest")
    source_path = (plan_path.parent / plan["source"]["path"]).resolve()
    payload = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != plan["source"]["sha256"]:
        raise ValueError("actual MAT bytes differ from the audit plan")
    record = read_bnci2014_008_candidate_record(
        io.BytesIO(payload), source_relative_path=source_path.name, source_sha256=actual_sha256,
        source_provenance={"stage": "public_processed_dataset_development", "physical_adc_raw": False})
    selection = plan["selection_index"]
    selection_rows = np.flatnonzero(record.selection_index == selection)
    next_rows = np.flatnonzero(record.selection_index == selection + 1)
    # This slice uses only the event-bounded interior. It does not infer label
    # permission from tail EEG or ask the MAT's target identity for a challenge.
    selection_interior = (int(record.flash_sample[selection_rows[0]]), int(record.flash_sample[next_rows[0]]))
    calibration_scope = RawSampleScope.from_record(plan["calibration_scope"])
    _validate_calibration_events(calibration_scope, record.flash_sample, selection_rows, len(record.eeg_uv))
    grid = NativeBoundaryGrid(tuple(plan["context_scope"]), BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE, actual_sha256,
        f"{record.subject_id}:{str(record.selection_id[selection_rows[0]])}:distributed_processed_MAT")
    if (not selection_interior[0] <= grid.context_scope[0] < grid.context_scope[1] <= selection_interior[1]
            or not calibration_scope.start_sample <= grid.context_scope[0]
            or grid.context_scope[1] > calibration_scope.stop_sample):
        raise ValueError("diagnostic context exceeds the actual requested selection interior")
    read_flash_rows = np.flatnonzero((record.flash_sample >= grid.context_scope[0])
                                   & (record.flash_sample < grid.context_scope[1]))
    if not set(read_flash_rows.tolist()) <= set(calibration_scope.event_indices):
        raise ValueError("diagnostic EEG includes an undeclared stimulus inside calibration support")
    context = record.eeg_uv[slice(*grid.context_scope)] * 1e-6
    problem = NativeBoundaryObservability(grid, FiniteFilterOperator.from_record(plan["short_operator"]),
        FiniteFilterOperator.from_record(plan["context_operator"]), tuple(plan["short_scope"]),
        tuple(plan["output_scope"]), tuple(map(tuple, plan["observed_scopes"])),
        plan["available_at_sample"], byte_budget=plan["byte_budget"])
    channels = len(grid.channel_names)
    shape = ((problem.output_scope[1] - problem.output_scope[0]) * channels,
             (problem.short_scope[1] - problem.short_scope[0]) * channels)
    K, b, producer = _correction(plan, shape, plan_path.parent)
    mapping = NativeAffineBoundaryMap(problem, K, producer, b, byte_budget=plan["byte_budget"])
    injections = _injections(plan, grid)
    observability = problem.report()
    # Correct rejection is a completed diagnostic, not an execution failure.
    witness = None
    if observability["status"] == "structurally_unidentifiable":
        witness, delta = problem.counterexample(context)
        injections["explicit_unobserved_exterior_impulse"] = delta
        try:
            problem.recover(problem.observe(context))
        except BoundaryIdentifiabilityError as error:
            recovery = {"status": "rejected", "type": type(error).__name__,
                        "reason": error.report["status"]}
        else:
            raise AssertionError("unidentified native target was silently reconstructed")
    else:
        recovered = problem.recover(problem.observe(context))
        target = problem.context_operator.apply(context)[
            problem.output_scope[0] - grid.context_scope[0]:problem.output_scope[1] - grid.context_scope[0]]
        recovery = {"status": "identified_noiseless_map", "recovered_sha256": array_digest(recovered),
                    "direct_operator_roundoff_norm": float(np.linalg.norm(recovered - target))}
    a, b = (sample - grid.context_scope[0] for sample in problem.short_scope)
    short = context[a:b]
    frozen_before = mapping.digest
    measurement = mapping.measure(context, injections, injection_design_identity=plan["injection_design_identity"])
    if mapping.digest != frozen_before:
        raise AssertionError("diagnostic execution mutated frozen T")
    padding = problem.padding_state_report(short)
    # Numerical artifacts contain maps/increments, never copied MAT, labels or
    # inferred physiological state. Source bytes/scope ledger bind actual reads.
    np.savez_compressed(destination / "numeric_maps.npz", baseline_matrix=mapping.baseline_matrix,
        complete_matrix=mapping.complete_matrix, correction_matrix=mapping.correction_matrix,
        correction_offset=mapping.correction_offset, **{f"injection_{i:03d}": h for i, h in enumerate(injections.values())})
    receipt = {"schema": "n2p3_boundary_observability_audit/1", "status": "completed",
        "source_snapshot_sha256": snapshot_sha256, "plan_sha256": plan_sha256,
        "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "source_sha256": actual_sha256, "native_source_grid": grid.record(),
        "selection_index": selection, "selection_id": str(record.selection_id[selection_rows[0]]),
        "declared_calibration_scope": calibration_scope.record(),
        "scope_authorization_issued_by_audit": False,
        "event_bounded_selection_interior": list(selection_interior),
        "actual_flash_rows_in_diagnostic_context": read_flash_rows.tolist(),
        "diagnostic_actual_read_scope": list(grid.context_scope),
        "legal_predictor_observation_scopes": [list(scope) for scope in problem.observed_scopes],
        "diagnostic_teacher_available_at_sample": grid.context_scope[1],
        "predictor_available_at_sample": problem.available_at_sample,
        "target_labels_used_for_fit_challenge_or_scoring": False, "training_executed": False,
        "upstream_bandpass_implementation": "unknown_not_reconstructed",
        "additional_operator_scope": "only_the_explicit_current_finite_SOS",
        "observability": observability, "recovery": recovery, "counterexample": witness,
        "padding_state": padding, "complete_T": measurement,
        "numeric_maps_sha256": hashlib.sha256((destination / "numeric_maps.npz").read_bytes()).hexdigest(),
        "evidence_scope": "public_processed_input_mathematical_software_diagnostic",
        "unknown_state_or_performance_confirmation": False}
    atomic_write_json(destination / "receipt.json", receipt)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != actual_sha256:
        raise ValueError("actual MAT bytes changed during the read-only audit")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, plan_sha256 = _load_plan(args.plan)
    snapshot = verified_executed_snapshot(args.snapshot, root=ROOT)
    snapshot_sha256 = snapshot["archive_sha256"]
    snapshot_manifest_sha256 = snapshot["manifest_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"schema": "n2p3_boundary_observability_execution/1", "status": "running",
              "source_snapshot_sha256": snapshot_sha256, "plan_sha256": plan_sha256}
    status["source_snapshot_manifest_sha256"] = snapshot_manifest_sha256
    atomic_write_json(args.output / "execution.json", status)
    try:
        receipt = _run(plan, args.plan, args.output, snapshot_sha256, plan_sha256,
                       snapshot_manifest_sha256=snapshot_manifest_sha256)
        if verified_executed_snapshot(args.snapshot, root=ROOT) != snapshot:
            raise ValueError("actual executing source snapshot changed during the boundary audit")
    except BaseException as error:
        atomic_write_json(args.output / "execution.json", {**status, "status": "failed",
                          "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    atomic_write_json(args.output / "execution.json", {**status, "status": "completed"})
    print(json.dumps({"status": receipt["status"], "observability": receipt["observability"]["status"],
                      "recovery": receipt["recovery"]["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
