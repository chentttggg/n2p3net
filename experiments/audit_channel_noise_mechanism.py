"""Audit a hash-bound real residual/ERP manifest with shared/private noise controls.

Version 1 retains the original frozen-response diagnostic crosses. Version 2
binds actual C/held owner views, modern producer M/R/Sigma and one requested
DecisionPlan to shared/private noise controls and common paired metrics. Public
v2 views are rebuilt from original MATs; software fixtures retain software scope.
Truth enters the external evaluator, never the noise fit or label-free scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity  # noqa: E402
from research.candidate_metrics import (  # noqa: E402
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.channel_noise_audit_inputs import (  # noqa: E402
    BoundInput,
    _sha,
    paired_input_digests,
    public_calibration_authority,
)
from research.channel_noise_mechanism import (  # noqa: E402
    FittedChannelNoise,
    FrozenERPResponse,
    NoiseGeometry,
    ResidualRecord,
)
from research.contracts import DecisionPlanContract, semantic_sha256  # noqa: E402
from research.execution import UnidentifiableFitError, atomic_write_json  # noqa: E402
from transfer.continuous_decision import ContinuousDecisionModel, _event_design  # noqa: E402
from transfer.ensemble_views import (  # noqa: E402
    CalibrationSelection,
    PredictionDecisionView,
)
from transfer.noise_model import array_digest, require_identifiers  # noqa: E402
from transfer.outcomes import CandidateCoverage, DecisionKey  # noqa: E402

PAIRED_PROGRAM = "frozen_physical_response_noise_paired/1"
WINNER_PROGRAM = "CandidateEvidence_normalized_log_probability/1"


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
    if payload.get("schema") == "n2p3_channel_noise_audit_input/2":
        return _audit_paired(payload, manifest_path, output)
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
                   for name in ("research/channel_noise_mechanism.py", "research/stationary_noise_artifact.py",
                                "transfer/dictionary_artifact.py", "transfer/temporal_noise.py",
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


def _executing_source():
    root = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve(), root / "experiments/build_channel_noise_audit_manifest.py"}
    paths.update((root / "src").rglob("*.py"))
    return {str(path.relative_to(root)).replace("\\", "/"): _sha(path)
            for path in sorted(paths) if path.is_file()}


def _paired_view(row, inputs, authoritative):
    (samples,) = inputs.arrays(row["arrays"], "samples")
    receipt = row["view"]
    if receipt.get("unavailable_reason") is not None:
        if samples.size:
            raise ValueError("unavailable view cannot hide supplied observations")
        samples = None
    view = PredictionDecisionView.from_record(receipt, samples=samples)
    if authoritative is not None:
        original = authoritative.get(view.record_id)
        if (original is None or original.view.physical_support_digest != view.physical_support_digest
                or original.view.raw_input_digest != view.raw_input_digest):
            raise ValueError("paired view differs from the actual original MAT owner")
    return view


def _audit_paired(payload, manifest_path, output):
    scope = payload.get("evidence_scope")
    if scope not in {"software_correctness", "public_calibration_development"}:
        raise ValueError("paired noise audit supports software or public calibration development only")
    program = payload["program"]
    if (program.get("id") != PAIRED_PROGRAM or program.get("winner_program") != WINNER_PROGRAM
            or program.get("conditioning_samples") != 32 or program.get("noise_order") != 8
            or program.get("residual_nuisance") != "explicit_zero_C_nuisance/1"
            or type(program.get("seed")) is not int or program["seed"] < 0):
        raise ValueError("paired audit requires its frozen order8/prefix32/winner/residual program")
    arms = payload["noise_arms"]
    if arms != [{"arm_id": "shared8", "family": "shared", "order": 8},
                {"arm_id": "private8", "family": "private", "order": 8}]:
        raise ValueError("paired noise roster differs from the predeclared shared/private controls")
    plan = DecisionPlanContract.from_record(payload["decision_plan"])
    inputs, authority, source_truth = BoundInput(manifest_path), None, {}
    if scope == "public_calibration_development":
        authority = public_calibration_authority(payload.get("source_records"), inputs)
        source_people = {group.view.participant_key for group in authority.values()}
        if set(plan.requested_participant_keys) != source_people:
            raise ValueError("public requested cohort differs from actual original source participants")
        source_truth = {DecisionKey(group.view.participant_key, group.view.decision_id): group.target_candidate
                        for group in authority.values()}
    source_before = _executing_source()
    prepared, all_views, held_keys, participant_keys = [], [], set(), set()
    for participant in payload["participants"]:
        geometry = NoiseGeometry(**participant["geometry"])
        if geometry.operator_identity != "native_unfiltered_volts/1":
            raise ValueError("paired physical views require their actual native unfiltered operator")
        training = [(_paired_view(row, inputs, authority), row) for row in participant["training"]]
        held = [(_paired_view(row, inputs, authority), row) for row in participant["held"]]
        if not training or not held:
            raise ValueError("paired participant needs complete actual C and held views")
        person = training[0][0].participant_key
        if person in participant_keys or any(view.participant_key != person for view, _ in training + held):
            raise ValueError("each fixed response/noise procedure needs one unique participant")
        participant_keys.add(person)
        descriptor = participant["response"]["source_model"]
        if descriptor.get("format") != "modern_continuous_model_npz":
            raise ValueError("paired modern response cannot fall back to a legacy H programme")
        content = inputs.file(descriptor).read_bytes()
        if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
            raise ValueError("modern source bytes changed before producer loading")
        model = ContinuousDecisionModel.load(io.BytesIO(content))
        if model.conditioning_samples != 32 or model.noise is None or model.noise.order != 8:
            raise ValueError("paired response anchor differs from its order8/prefix32 program")
        records, residuals = [], []
        for view, row in training:
            samples, onsets, labels, design, nuisance = inputs.arrays(row["arrays"],
                "samples", "onsets", "labels", "design", "nuisance")
            if (view.samples is None or not np.array_equal(samples, view.samples)
                    or not np.array_equal(onsets, view.onsets) or np.any(nuisance != 0)
                    or not np.array_equal(design, _event_design(onsets, labels, len(samples), model.basis))):
                raise ValueError("paired C samples/events/design or declared zero nuisance differ")
            CalibrationSelection(view, np.asarray(labels, dtype=bool), row["target_candidate"])
            if authority is not None:
                actual = authority[view.record_id]
                if row["target_candidate"] != actual.target_candidate or not np.array_equal(labels, actual.labels):
                    raise ValueError("paired C labels/target differ from actual original source authorization")
            receipt = {"record_id": view.record_id, "participant_key": person,
                "recording_id": view.source_sha256, "read_support": list(view.raw_scope),
                "fit_scope": "target_calibration", "source_sha256": view.source_sha256,
                "input_sha256": array_digest(samples, onsets, labels), "n_samples": len(samples),
                "original_view": view.record()}
            records.append(receipt)
        response = FrozenERPResponse.from_continuous_model(model, geometry=geometry,
            source_model_sha256=descriptor["sha256"], response_id=participant["response"]["response_id"],
            training_records=records)
        for (view, row), receipt in zip(training, records, strict=True):
            samples, design, nuisance = inputs.arrays(row["arrays"], "samples", "design", "nuisance")
            residuals.append(ResidualRecord.from_arrays(samples=samples, design=design, kernel_mean=response.mean,
                nuisance=nuisance, record_id=view.record_id, participant_key=person,
                recording_id=receipt["recording_id"], sample_start=view.raw_scope[0], geometry=geometry,
                role="target_calibration", source_model_sha256=descriptor["sha256"], source_input_sha256=row["arrays"]["sha256"]))
        actual_held = []
        for view, row in held:
            key = DecisionKey(view.participant_key, view.decision_id)
            if key in held_keys:
                raise ValueError("paired requested physical decision is duplicated")
            held_keys.add(key)
            if tuple(view.channel_names) != geometry.channels or view.reference != geometry.reference or view.sample_rate_hz != geometry.sfreq:
                raise ValueError("paired held physical geometry differs from frozen C")
            designs = None
            if view.samples is not None:
                (designs,) = inputs.arrays(row["arrays"], "candidate_designs")
                expected = np.asarray([_event_design(view.onsets, view.decision.membership[:, q], len(view.samples), model.basis)
                    for q in range(len(view.decision.candidate_vocabulary))])
                if not np.array_equal(designs, expected):
                    raise ValueError("paired candidate designs differ from actual membership/onsets/basis")
            actual_held.append((view, designs))
        all_views.extend(view for view, _ in training + held)
        prepared.append((person, geometry, response, tuple(residuals), actual_held))
    expected_keys = {DecisionKey(entry.participant_key, entry.decision_id) for entry in plan.entries}
    if held_keys != expected_keys or participant_keys != set(plan.participant_keys):
        raise ValueError("paired physical held roster differs from the complete requested plan")
    if authority is not None and any(source_truth.get(DecisionKey(entry.participant_key, entry.decision_id)) != entry.target_candidate
                                     for entry in plan.entries):
        raise ValueError("paired external truth differs from its original public source owner")
    cache, identity = paired_input_digests(all_views)
    if cache != plan.target_cache_sha256 or identity != plan.target_identity_digest:
        raise ValueError("paired plan is not bound to its actual C/held inputs")
    vocabulary, level, seed = tuple(payload["candidate_vocabulary"]), payload["evidence_level"], program["seed"]
    requested_coverage = {}
    for _, _, _, _, held in prepared:
        for view, _ in held:
            if view.decision.candidate_vocabulary != vocabulary or view.decision.repetitions != level:
                raise ValueError("paired physical vocabulary/repetitions differ from requested task")
            expected = dict(zip(vocabulary, map(int, view.decision.membership.sum(axis=0)), strict=True))
            observed = dict(zip(vocabulary, map(int, view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0)), strict=True))
            requested_coverage[DecisionKey(view.participant_key, view.decision_id)] = CandidateCoverage.from_mappings(expected, observed)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "n2p3_channel_noise_audit/2", "status": "running", "program": program,
        "winner_program": WINNER_PROGRAM, "evidence_scope": scope, "decision_plan": plan.record(),
        "input_manifest_sha256": _sha(manifest_path), "source_files": source_before,
        "cells": [], "noise_arms": [], "responses": [response.record() for _, _, response, _, _ in prepared]}
    atomic_write_json(output / "report.json", report)
    evidence_by_arm = {arm["arm_id"]: [] for arm in arms}
    try:
        for index, (person, geometry, response, training, held) in enumerate(prepared):
            for arm in arms:
                noise, reloaded, failure = None, None, None
                try:
                    noise = FittedChannelNoise.fit(training, family=arm["family"], order=8, conditioning_samples=32)
                    path = output / f"noise_{index:03d}_{arm['arm_id']}.json"
                    noise.save(path)
                    reloaded = FittedChannelNoise.load(path)
                    if noise.record() != reloaded.record():
                        raise ValueError("paired saved noise state did not replay exactly")
                    report["noise_arms"].append({"participant_key": person, "arm_id": arm["arm_id"],
                        "status": "fitted", "state_path": path.name, "state_sha256": _sha(path)})
                except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                    failure = str(error)
                    report["noise_arms"].append({"participant_key": person, "arm_id": arm["arm_id"], "status": "fit_failure", "reason": failure})
                model_digest = None if noise is None else semantic_sha256({"noise": noise.record(), "response": response.record()})
                fit_digest = semantic_sha256({"C": [row.ledger() for row in training], "program": program, "arm": arm})
                for view, designs in held:
                    counts = requested_coverage[DecisionKey(person, view.decision_id)]
                    identity = CandidateExpertIdentity(arm["arm_id"], view.raw_input_digest, model_digest,
                        None, fit_digest, "frozen_response_conditional_tail_log_probability/1", False)
                    common = dict(candidate_vocabulary=vocabulary, participant_key=person, decision_id=view.decision_id,
                        physical_support_digest=view.physical_support_digest, expert_identities=(identity,), evidence_level=level,
                        expected_event_counts=tuple(dict(counts.expected_event_counts)[candidate] for candidate in vocabulary),
                        observed_event_counts=tuple(dict(counts.observed_event_counts)[candidate] for candidate in vocabulary))
                    cell = {"participant_key": person, "decision_id": view.decision_id, "arm_id": arm["arm_id"]}
                    reason, kind = failure, "fit_failure"
                    if failure is None and (view.samples is None or not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0)):
                        reason, kind = view.unavailable_reason or "incomplete_physical_decision", "incomplete"
                    if reason is None:
                        try:
                            kwargs = dict(candidate_ids=vocabulary, geometry=geometry, record_id=view.record_id,
                                recording_id=view.source_sha256, sample_start=view.raw_scope[0])
                            score = noise.score(view.samples, designs, response, **kwargs)
                            replay = reloaded.score(view.samples, designs, response, **kwargs)
                            if score != replay:
                                raise ValueError("paired saved noise/response changed actual held score")
                            value = paired_candidate_evidence(score, **common)
                            cell.update(status="scored", score=score, legacy_score_winner=score["winner"],
                                saved_noise_score_replay_exact=True)
                        except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                            reason, kind = str(error), "not_available"
                    if reason is not None:
                        value = CandidateEvidence(log_probability=None, failure_kind=kind, failure_reason=reason, **common)
                        cell.update(status=kind, failure_reason=reason)
                    cell["evidence"] = value.record()
                    evidence_by_arm[arm["arm_id"]].append(value)
                    report["cells"].append(cell)
                atomic_write_json(output / "report.json", report)
        metrics = {name: evaluate_candidate_evidence({seed: values}, decision_plan=plan, seeds=(seed,),
            candidate_vocabulary=vocabulary, evidence_level=level, requested_coverage=requested_coverage)
            for name, values in evidence_by_arm.items()}
        report["metrics"] = {name: value.record() for name, value in metrics.items()}
        report["contrasts"] = {"private8_minus_shared8": compare_candidate_probability_metrics(metrics["shared8"], metrics["private8"]).record()}
        inputs.verify_unchanged()
        if _executing_source() != source_before:
            raise RuntimeError("paired executing source changed during execution")
        report.update(status="completed", source_unchanged=True, input_files={str(p): digest for p, digest in inputs.files.items()},
            no_held_labels_in_fit_or_score=True, no_temperature_or_model_selection=True,
            claim="fixed_response_noise_scoring_program_development_no_automatic_promotion")
    except BaseException as error:
        report.update(status="contract_or_execution_failed", failure={"type": type(error).__name__, "detail": str(error)})
        atomic_write_json(output / "report.json", report)
        raise
    atomic_write_json(output / "report.json", report)
    return report


def paired_candidate_evidence(score, **identity):
    """New panel winner owner; retain the old score winner only as a diagnostic.

    The raw scorer already normalized its fixed candidate prior once. Do not
    add that prior again or reaggregate a joint continuous density by flashes.
    """
    return CandidateEvidence(log_probability=score["log_probability"], **identity)


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
