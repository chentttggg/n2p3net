"""Fixed-noise spatial-signal prior audit on actual independent BNCI C groups.

Consumes an already fitted native ContinuousDecisionModel on exactly the
declared C groups. It freezes that AR/noise state and refits only the response
Gaussian prior. This is an attributable signal-prior program, not joint MLE.
The held score path reads only PredictionDecisionView, never its target label.
Cloud execution belongs to the shared resource owner; this CLI does not launch
training workers or invent a new source/participant authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.continuous_decision import (  # noqa: E402
    CalibrationRecord,
    ContinuousDecisionModel,
    _event_design,
    _validate_record,
)
from transfer.ensemble_views import prepare_bnci_calibration_groups  # noqa: E402
from transfer.noise_model import array_digest, metadata_digest  # noqa: E402
from transfer.spatial_signal_prior import (  # noqa: E402
    IMPLEMENTATION_SHA256,
    SpatialLayout,
    SpatialObservation,
    SpatialPriorIdentificationError,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)


def _layout_from_view(view):
    if view.signal_unit != "V" or any(kind != "EEG" for kind in view.channel_types):
        raise ValueError("spatial research consumes actual measured native EEG volts")
    authority = {"channels": view.channel_names, "channel_types": view.channel_types,
                 "reference": view.reference, "unit": view.signal_unit,
                 "source_sha256": view.source_sha256, "source_dataset_id": view.source_dataset_id}
    return SpatialLayout(view.channel_names, view.reference, metadata_digest(authority),
                         view.channel_types, view.signal_unit)


def _observation(layout, mode, reference):
    if mode == "native":
        return SpatialObservation.native(layout)
    if mode == "cz":
        return SpatialObservation.measured_reference(layout, reference)
    return SpatialObservation.average_reference(layout)


def _temporal_logdet(noise, rows):
    if noise.startup_mode == "stationary_innovations":
        diagonal = np.diag(noise.startup_whitening_factor)[:min(noise.order, rows)]
        return float(np.log(diagonal).sum())
    if noise.startup_mode == "stationary":
        return -.5 * float(np.linalg.slogdet(noise.startup_covariance[:min(noise.order, rows), :min(noise.order, rows)])[1])
    raise ValueError("spatial full/prefix audit requires a complete invertible temporal observation map")


def _response_block(view, design, observation, baseline, *, scope, dictionary_sha256):
    noise = baseline.noise
    prefix = baseline.conditioning_samples
    times = np.arange(len(view.samples))
    trend = ((2 * times - (prefix - 1)) / (prefix - 1) if prefix
             else np.linspace(-1, 1, len(times)))
    nuisance = np.column_stack((np.ones(len(times)), trend))
    native = np.column_stack((view.samples, design, nuisance))
    white = noise.whiten(native)
    channels, coordinates = len(view.channel_names), design.shape[1]
    authorization = {"record_id": view.record_id, "fit_scope": scope,
                     "source_sha256": view.source_sha256, "raw_sample_support": list(view.raw_scope),
                     "dictionary_sha256": dictionary_sha256,
                     "temporal_program_sha256": noise.record()["artifact_sha256"],
                     "original_view": view.record()}
    return SpatialResponseBlock(observation.project(white[:, :channels]),
        white[:, channels:channels + coordinates], observation, authorization,
        white[:, channels + coordinates:], _temporal_logdet(noise, len(white)))


def _prefix(block, count, noise):
    authorization = dict(block.authorization)
    start = authorization["raw_sample_support"][0]
    authorization["raw_sample_support"] = [start, start + count]
    return SpatialResponseBlock(block.samples[:count], block.design[:count], block.observation,
                                authorization, block.nuisance[:count], _temporal_logdet(noise, count))


def _score_view(model, view, observation, baseline, dictionary_sha256):
    """Only unlabeled public membership generates held candidate designs."""
    evidence = []
    for labels in view.decision.membership.T:
        design = _event_design(view.onsets, labels, len(view.samples), baseline.basis)
        block = _response_block(view, design, observation, baseline, scope="read_only_prediction",
                                dictionary_sha256=dictionary_sha256)
        score = (model.conditional_log_density(block, _prefix(block, baseline.conditioning_samples, baseline.noise))
                 if baseline.conditioning_samples else model.log_density(block))
        evidence.append(score)
    evidence = np.asarray(evidence)
    log_probability = evidence - logsumexp(evidence)
    return {"candidate_ids": list(view.decision.candidate_vocabulary),
            "log_evidence": evidence.tolist(), "log_probability": log_probability.tolist(),
            "probabilities": np.exp(log_probability).tolist(),
            "held_truth_used": False, "input_sha256": view.raw_input_digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--noise-model", type=Path, required=True)
    parser.add_argument("--prior-program", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-labels", type=int, default=800)
    parser.add_argument("--fit-groups", type=int, default=2)
    parser.add_argument("--observation", choices=("native", "cz", "car"), default="native")
    parser.add_argument("--reference-channel", default="Cz")
    parser.add_argument("--output-row-gains", type=float, nargs="+",
                        help="explicit nonzero output coordinate gains; information is unchanged")
    args = parser.parse_args(argv)
    verified_executed_snapshot(args.snapshot, root=ROOT)
    if args.fit_groups < 1:
        raise ValueError("spatial audit requires at least one authorized fit group")
    args.output.mkdir(parents=True, exist_ok=False)
    identities = {"input": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "noise_model": hashlib.sha256(args.noise_model.read_bytes()).hexdigest(),
                  "prior_program": hashlib.sha256(args.prior_program.read_bytes()).hexdigest()}
    baseline = ContinuousDecisionModel.load(args.noise_model)
    if baseline.noise is None or baseline.noise.startup_mode == "conditional_prefix":
        raise ValueError("baseline requires an actual full stationary temporal whitening state")
    groups = prepare_bnci_calibration_groups(args.input, source_sha256=identities["input"],
        authorized_labels=args.authorized_labels, group_count=args.fit_groups + 1).groups
    calibration, held_view = groups[:args.fit_groups], groups[-1].view
    if any(group.view.sample_rate_hz != baseline.sfreq for group in groups):
        raise ValueError("fixed noise and physical response grid differ from actual native C/held sampling")
    geometry = _layout_from_view(calibration[0].view)
    if (tuple(baseline.channels) != geometry.channels
            or any(_layout_from_view(group.view).record() != geometry.record() for group in groups)):
        raise ValueError("fixed native noise and actual C/held measured layouts differ")
    records = [CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
               for group in calibration]
    actual_records = baseline.provenance["records"]
    if len(actual_records) != len(records):
        raise ValueError("frozen noise must have fitted exactly the declared C groups")
    for actual, record in zip(actual_records, records, strict=True):
        _validate_record(record.samples, record.onsets, len(geometry.channels), len(baseline.basis))
        if (actual["record_id"] != record.record_id
                or actual["input_sha256"] != array_digest(record.samples, record.onsets, record.target_mask)):
            raise ValueError("fixed noise fitted C differs from the actual authorized C input")
    specification = json.loads(args.prior_program.read_text(encoding="utf-8"))
    if (specification["channels"] != list(geometry.channels)
            or specification["acquisition_reference"] != geometry.acquisition_reference
            or specification["signal_unit"] != geometry.signal_unit):
        raise ValueError("prior program differs from actual native voltage/reference coordinates")
    dictionary_identity = (baseline.dictionary.digest() if baseline.dictionary is not None
                           else array_digest(baseline.basis))
    prior = SpatialSignalPrior(geometry, specification["mean"], specification["temporal_covariance"],
        specification["signal_covariance"],
        {"dictionary_sha256": dictionary_identity,
         "vectorization": "column_major_response_by_native_EEG",
         "actual_basis_sha256": array_digest(baseline.basis), "sample_rate_hz": baseline.sfreq,
         "operator_identity": baseline.operator_identity,
         "coefficient_groups": ["common", "target_increment"]},
        {"role": "predeclared", "authority_sha256": identities["prior_program"]})
    observation = _observation(geometry, args.observation, args.reference_channel)
    if args.output_row_gains is not None:
        observation = observation.with_output_gains(args.output_row_gains)
    blocks = [_response_block(group.view,
        _event_design(record.onsets, record.target_mask, len(record.samples), baseline.basis),
        observation, baseline, scope="target_calibration", dictionary_sha256=dictionary_identity)
        for group, record in zip(calibration, records, strict=True)]
    with cpu_thread_budget(1):
        try:
            fitted = SpatialSignalPosterior.fit(prior, blocks,
                native_noise_covariance=baseline.innovation_covariance,
                noise_binding={"authority_sha256": baseline.digest(),
                               "actual_noise_model_file_sha256": identities["noise_model"],
                               "program": "freeze_exact_C_fitted_noise_refit_signal_only"})
            prediction = _score_view(fitted, held_view, observation, baseline, dictionary_identity)
            fitted.save(args.output / "spatial_model.json")
            reloaded = SpatialSignalPosterior.load(args.output / "spatial_model.json",
                expected_artifact_sha256=fitted.record()["artifact_sha256"])
            replay = _score_view(reloaded, held_view, observation, baseline, dictionary_identity)
            if prediction != replay:
                raise ValueError("spatial signal posterior replay differs from saved state")
            result = {"status": "completed", "prediction": prediction, "exact_saved_replay": True}
        except SpatialPriorIdentificationError as error:
            result = {"status": "typed_mechanism_failure", "failure": error.record(), "scored_decisions": 0}
    verified_executed_snapshot(args.snapshot, root=ROOT)
    for name, path in (("input", args.input), ("noise_model", args.noise_model), ("prior_program", args.prior_program)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != identities[name]:
            raise ValueError("spatial audit changed or consumed changed input state")
    atomic_write_json(args.output / "receipt.json", {
        "schema": "n2p3_spatial_signal_audit/3", **result, "identities": identities,
        "observation": observation.record(), "prior": prior.record(),
        "calibration": [group.record() for group in calibration], "held_view": held_view.record(),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "implementation_sha256": IMPLEMENTATION_SHA256, "fixed_noise_for_signal_prior_comparison": True,
        "evidence_scope": "public_C_signal_prior_mechanism_audit", "joint_noise_MLE_claimed": False,
        "public_performance_claimed": False, "adult_confirmation_claimed": False, "cpu_threads": 1})
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
