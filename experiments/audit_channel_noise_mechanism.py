"""Audit a hash-bound real residual/ERP manifest with shared/private noise controls.

Input is n2p3_channel_noise_audit_input/1, restricted to public calibration
development. Each NPZ is verified before allow_pickle=False loading. Mean/noise
crosses retain every requested decision; truth enters reporting and residual
diagnostics after unlabelled score, never a fit, temperature or winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research.channel_noise_mechanism import (  # noqa: E402
    FittedChannelNoise,
    FrozenERPResponse,
    NoiseGeometry,
    ResidualRecord,
)
from research.execution import UnidentifiableFitError, atomic_write_json  # noqa: E402
from transfer.continuous_decision import _event_design  # noqa: E402
from transfer.noise_model import array_digest, require_identifiers, require_sha256  # noqa: E402


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BoundInput:
    """Verify the physical file once and return copied numeric arrays, no pickle."""

    def __init__(self, manifest):
        self.root = manifest.parent.resolve()
        self.files = {}
        self.cache = {}

    def file(self, descriptor):
        if not isinstance(descriptor, dict) or not {"path", "sha256"} <= descriptor.keys():
            raise ValueError("input file descriptor requires a path and actual SHA-256")
        require_sha256(descriptor["sha256"], "input file")
        path = (self.root / descriptor["path"]).resolve()
        if path not in self.files:
            self.files[path] = _sha(path)
        if self.files[path] != descriptor["sha256"]:
            raise ValueError(f"input byte identity differs: {path}")
        return path

    def arrays(self, descriptor, *names):
        path = self.file(descriptor)
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as archive:
                self.cache[path] = {key: np.asarray(archive[key]).copy() for key in archive.files}
        values = []
        for name in names:
            key = descriptor[f"{name}_key"]
            value = self.cache[path][key]
            if value.dtype.kind not in "fiu":
                raise ValueError(f"NPZ array {key} must contain actual real numeric values")
            values.append(value)
        return tuple(values)

    def verify_unchanged(self):
        for path, digest in self.files.items():
            if _sha(path) != digest:
                raise ValueError(f"study input changed during execution: {path}")


def _response(row, inputs, geometry):
    source = inputs.file(row["source_model"])
    mean, factor = inputs.arrays(row["arrays"], "mean", "coefficient_factor")
    original_inputs = []
    if row["source_model"].get("format") == "continuous_model_npz":
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        if (tuple(metadata["channels"]) != geometry.channels or metadata["sfreq"] != geometry.sfreq
                or metadata["operator_identity"] != geometry.operator_identity
                or metadata["provenance"]["fit_scope"] != row["fit_scope"]
                or {r["record_id"] for r in metadata["provenance"]["records"]}
                   != {r["record_id"] for r in row["training_records"]}):
            raise ValueError("exported response geometry or fit scope differs from actual original model")
        original_inputs = metadata["provenance"]["records"]
        source_arrays = {"path": str(source), "sha256": row["source_model"]["sha256"],
                         "mean_key": "kernel_mean", "row_covariance_key": "kernel_row_covariance",
                         "channel_covariance_key": "innovation_covariance"}
        original_mean, row_covariance, channel_covariance = inputs.arrays(source_arrays,
            "mean", "row_covariance", "channel_covariance")
        original_factor = np.kron(np.linalg.cholesky(row_covariance), np.linalg.cholesky(channel_covariance))
        normalized = solve_triangular(original_factor, factor, lower=True)
        if (not np.array_equal(mean, original_mean) or factor.shape[0] != len(original_factor)
                or np.linalg.norm(normalized @ normalized.T - np.eye(len(original_factor)))
                   > 1e-8 * np.sqrt(len(original_factor))):
            raise ValueError("exported ERP state differs from the actual original model arrays")
    return FrozenERPResponse(mean, factor, geometry, _sha(source), row["response_id"],
                             {"fit_scope": row["fit_scope"],
                              "source_model_path": str(source),
                              "exporter_identity": row["exporter_identity"],
                              "records": row.get("training_records", []),
                              "original_model_input_records": original_inputs})


def _residual(row, inputs, geometry, response, role):
    descriptor = row.get("arrays_by_response", {}).get(response.response_id, row.get("arrays"))
    samples, design, nuisance = inputs.arrays(descriptor, "samples", "design", "nuisance")
    originals = response.provenance.get("original_model_input_records", ())
    if role != "held_diagnostic" and originals:
        if "onsets_key" not in descriptor or "labels_key" not in descriptor:
            raise ValueError("actual original model residuals require their original stimulus/label operands")
        onsets, labels = inputs.arrays(descriptor, "onsets", "labels")
        matched = [original for original in originals if original["record_id"] == row["record_id"]]
        if (len(matched) != 1 or matched[0]["input_sha256"] != array_digest(samples, onsets, labels)
                or matched[0]["n_samples"] != len(samples)):
            raise ValueError("training residual operands differ from actual original model fit input")
        (basis,) = inputs.arrays({"path": response.provenance["source_model_path"],
            "sha256": response.source_model_sha256, "basis_key": "basis"}, "basis")
        if not np.array_equal(design, _event_design(onsets, labels, len(samples), basis)):
            raise ValueError("training residual design differs from actual original basis and events")
    return ResidualRecord.from_arrays(samples=samples, design=design, kernel_mean=response.mean,
        nuisance=nuisance, record_id=row["record_id"], participant_key=row["participant_key"],
        recording_id=row["recording_id"], sample_start=row["sample_start"], geometry=geometry,
        role=role, source_model_sha256=response.source_model_sha256,
        source_input_sha256=descriptor["sha256"])


def _metric(cells):
    successful = [cell for cell in cells if cell["status"] == "scored"]
    risks = []
    for cell in successful:
        score = cell["score"]
        truth = score["candidate_ids"].index(cell["target_candidate"])
        probabilities = np.asarray(score["probabilities"])
        risks.append((-score["log_probability"][truth],
                      float(np.sum((probabilities - np.eye(len(probabilities))[truth]) ** 2))))
    return {"requested_decisions": len(cells), "scored_decisions": len(successful),
            "fit_or_score_failures": len(cells) - len(successful),
            "hit_at_all": sum(cell["score"]["winner"] == cell["target_candidate"]
                              for cell in successful) / len(cells),
            "probability_coverage": len(successful) / len(cells),
            "available_probability_NLL": None if not risks else float(np.mean([r[0] for r in risks])),
            "available_probability_Brier": None if not risks else float(np.mean([r[1] for r in risks])),
            "risk_denominator": "scored_only_always_reported_with_full_requested_failure_denominator"}


def _optional_diagnostic(noise, records, *, max_lag):
    """Undefined residual spectra cannot invalidate an actual fit or prediction."""
    try:
        return {"status": "diagnosed", **noise.diagnose(records, max_lag=max_lag)}
    except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
        return {"status": "unavailable", "error_type": type(error).__name__, "detail": str(error),
                "role": "residual_noise_diagnostic_only", "statistics_fabricated": False,
                "fit_and_prediction_status_unchanged": True,
                "record_ids": [record.record_id for record in records]}


def audit(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    payload = json.loads(manifest_path.read_bytes())
    if (payload.get("schema") != "n2p3_channel_noise_audit_input/1"
            or payload.get("evidence_scope") != "public_calibration_development"):
        raise ValueError("this audit is public C development, not adult or formal confirmation")
    geometry = NoiseGeometry(**payload["geometry"])
    inputs = BoundInput(manifest_path)
    require_identifiers([row["response_id"] for row in payload["responses"]], "response controls")
    require_identifiers([row["arm_id"] for row in payload["noise_arms"]], "noise controls")
    require_identifiers([row["record_id"] for row in payload["held"]], "requested held decisions")
    if not payload["training"] or not payload["held"]:
        raise ValueError("the actual train and held cohorts must both be nonempty")
    diagnostic_lag = payload.get("max_diagnostic_lag", 32)
    if type(diagnostic_lag) is not int or diagnostic_lag < 1:
        raise ValueError("max diagnostic lag must be an explicit positive integer")
    responses = {row["response_id"]: _response(row, inputs, geometry) for row in payload["responses"]}
    # Validate all held truth identities before any fit, then pass none to score.
    for row in payload["held"]:
        require_identifiers(row["candidate_ids"], "complete held candidate identities")
        if row["target_candidate"] not in row["candidate_ids"]:
            raise ValueError("held report truth is outside its complete frozen vocabulary")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "n2p3_channel_noise_audit/1", "status": "running",
              "input_manifest_sha256": _sha(manifest_path), "geometry": geometry.record(),
              "evidence_scope": "public_calibration_development", "noise_arms": [], "cells": [],
              "claim": "attributable_frozen_response_noise_controls_no_production_promotion",
              "source_files": {str(Path(__file__).resolve()): _sha(Path(__file__)),
                **{str(Path(__file__).resolve().parents[1] / "src" / name):
                   _sha(Path(__file__).resolve().parents[1] / "src" / name)
                   for name in ("research/channel_noise_mechanism.py", "transfer/temporal_noise.py",
                                "transfer/noise_model.py", "transfer/continuous_decision.py")}}}
    atomic_write_json(output / "report.json", report)
    try:
        for arm_index, arm in enumerate(payload["noise_arms"]):
            noise, failure = None, None
            training_response = responses[arm["residual_response_id"]]
            training = tuple(_residual(row, inputs, geometry, training_response, payload["fit_scope"])
                             for row in payload["training"])
            try:
                if "frozen_noise" in arm:
                    noise = FittedChannelNoise.load(inputs.file(arm["frozen_noise"]))
                    if noise.geometry != geometry:
                        raise ValueError("frozen producer geometry differs from this comparison")
                elif "frozen_parameters" in arm:
                    frozen = arm["frozen_parameters"]
                    coefficients, reflections, covariance, prefix_root = inputs.arrays(frozen["arrays"],
                        "coefficients", "reflections", "covariance", "prefix_root")
                    source_model = inputs.file(frozen["source_model"])
                    with np.load(source_model, allow_pickle=False) as source_archive:
                        original_covariance = np.asarray(source_archive["innovation_covariance"])
                        original_metadata = json.loads(str(source_archive["metadata"]))
                    if (not np.array_equal(covariance, original_covariance)
                            or original_metadata["noise"] != frozen["actual_temporal_record"]
                            or original_metadata["conditioning_samples"] != payload["conditioning_samples"]
                            or tuple(original_metadata["channels"]) != geometry.channels
                            or original_metadata["sfreq"] != geometry.sfreq
                            or original_metadata["operator_identity"] != geometry.operator_identity):
                        raise ValueError("exported noise arrays differ from actual original model state")
                    noise = FittedChannelNoise.from_frozen_shared_arrays(coefficients, reflections,
                        covariance, prefix_root, geometry=geometry,
                        conditioning_samples=payload["conditioning_samples"],
                        source_model_sha256=frozen["source_model"]["sha256"],
                        actual_temporal_record=frozen["actual_temporal_record"], training_records=training)
                else:
                    noise = FittedChannelNoise.fit(training, family=arm["family"], order=arm["order"],
                        conditioning_samples=payload["conditioning_samples"],
                        minimum_samples_per_parameter=arm.get("minimum_samples_per_parameter", 20))
                if noise.conditioning_samples != payload["conditioning_samples"]:
                    raise ValueError("noise arms differ in actual fixed prefix support")
                model_path = output / f"noise_{arm_index:03d}.json"
                noise.save(model_path)
                reloaded = FittedChannelNoise.load(model_path)
                if reloaded.record() != noise.record():
                    raise ValueError("saved noise state did not replay exactly")
                training_diagnostic = _optional_diagnostic(noise, training,
                    max_lag=payload.get("max_diagnostic_lag", 32))
                report["noise_arms"].append({"arm_id": arm["arm_id"], "status": "fitted_or_frozen",
                    "residual_response_id": training_response.response_id,
                    "state_path": model_path.name, "state_file_sha256": _sha(model_path),
                    "parameter_budget": noise.record()["parameter_budget"],
                    "training_diagnostic": training_diagnostic})
            except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                failure = {"error_type": type(error).__name__, "detail": str(error)}
                report["noise_arms"].append({"arm_id": arm["arm_id"], "status": "fit_failure", **failure})
            for response_id, response in responses.items():
                for row in payload["held"]:
                    cell = {"arm_id": arm["arm_id"], "response_id": response_id,
                            "participant_key": row["participant_key"], "record_id": row["record_id"],
                            "target_candidate": row["target_candidate"], "status": "fit_failure"}
                    if failure is not None:
                        cell["failure"] = failure
                    else:
                        samples, designs = inputs.arrays(row["arrays"], "samples", "candidate_designs")
                        if "onsets_key" in row["arrays"] and "membership_key" in row["arrays"]:
                            onsets, membership = inputs.arrays(row["arrays"], "onsets", "membership")
                            if array_digest(samples, onsets, membership) != row["original_score_input_sha256"]:
                                raise ValueError("held raw/schedule differs from actual original read-only score")
                            (basis,) = inputs.arrays({"path": response.provenance["source_model_path"],
                                "sha256": response.source_model_sha256, "basis_key": "basis"}, "basis")
                            expected_designs = np.asarray([_event_design(onsets, membership[:, candidate],
                                len(samples), basis) for candidate in range(len(row["candidate_ids"]))])
                            if not np.array_equal(designs, expected_designs):
                                raise ValueError("held candidate designs differ from actual public schedule/basis")
                        try:
                            kwargs = {"candidate_ids": row["candidate_ids"], "geometry": geometry,
                                      "record_id": row["record_id"], "recording_id": row["recording_id"],
                                      "sample_start": row["sample_start"]}
                            prediction = noise.score(samples, designs, response, **kwargs)
                            replay = reloaded.score(samples, designs, response, **kwargs)
                            if replay != prediction:
                                raise ValueError("same saved noise/ERP state changed actual held score")
                            cell.update(status="scored", score=prediction, saved_noise_score_replay_exact=True)
                            descriptor = row.get("arrays_by_response", {}).get(response_id, row["arrays"])
                            if "design_key" in descriptor and "nuisance_key" in descriptor:
                                held_residual = _residual(row, inputs, geometry, response, "held_diagnostic")
                                cell["truth_residual_diagnostic"] = _optional_diagnostic(noise, (held_residual,),
                                    max_lag=payload.get("max_diagnostic_lag", 32))
                            reference = row.get("source_replays", {}).get(f"{arm['arm_id']}::{response_id}")
                            if reference is not None:
                                expected = np.asarray(reference["log_density"])
                                if expected.shape != (len(row["candidate_ids"]),):
                                    raise ValueError("frozen source replay differs in complete candidate support")
                                cell["frozen_source_replay_comparison"] = {
                                    "original_source_model_sha256": reference["source_model_sha256"],
                                    "max_absolute_log_density_difference": float(np.max(np.abs(
                                        np.asarray(prediction["log_density"]) - expected))),
                                    "max_absolute_probability_difference": float(np.max(np.abs(
                                        np.asarray(prediction["probabilities"]) - np.asarray(reference["probabilities"])))),
                                    "original_winner": reference["winner"],
                                    "winner_matches": prediction["winner"] == reference["winner"],
                                    "role": "numerical_relation_report_no_automatic_equivalence_claim"}
                        except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                            cell.update(status="score_failure", failure={"error_type": type(error).__name__,
                                                                         "detail": str(error)})
                    report["cells"].append(cell)
            atomic_write_json(output / "report.json", report)
        report["metrics"] = {}
        for arm in payload["noise_arms"]:
            for response_id in responses:
                cells = [cell for cell in report["cells"]
                         if cell["arm_id"] == arm["arm_id"] and cell["response_id"] == response_id]
                participants = sorted({cell["participant_key"] for cell in cells})
                metrics = {participant: _metric([cell for cell in cells
                            if cell["participant_key"] == participant]) for participant in participants}
                report["metrics"][f"{arm['arm_id']}::{response_id}"] = {
                    "participants": metrics, "fixed_participant_count": len(participants),
                    "participant_macro_hit_at_all": float(np.mean([value["hit_at_all"] for value in metrics.values()])),
                    "interpretation": "finite_public_C_development_single_frozen_response_program"}
        inputs.verify_unchanged()
        report.update(status="completed", input_files={str(path): digest for path, digest in inputs.files.items()},
                      no_held_labels_in_fit_or_score=True, no_temperature_or_model_selection=True)
    except BaseException as error:
        report.update(status="contract_or_execution_failed",
                      failure={"error_type": type(error).__name__, "detail": str(error)})
        atomic_write_json(output / "report.json", report)
        raise
    atomic_write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input_manifest, args.output)
    print(json.dumps({"status": report["status"], "requested_cells": len(report["cells"]),
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
