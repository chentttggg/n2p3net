"""Complete seven-EEG noise operands using the original numerical noise owner.

The C bridge reuses the existing frozen-mean residual plugin programme. It
receives canonical raw/C capabilities and a real fixed-mean callback, then calls
FittedChannelNoise.fit once. Response-prior updating remains a separate stage.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.brainsync_raw_authority import (
    allowed_candidates,
    require_calibration_labels,
    require_raw_capability,
)
from data.contract import BRAIN_SYNC_ACQUISITION_REFERENCE, BRAIN_SYNC_EEG_CHANNELS
from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry, ResidualRecord
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.structured_response_frontier import StoredNoiseCarrier
from transfer.noise_model import array_digest, metadata_digest, require_sha256

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEGMENT_LAW = "independent_noise_segments_original_stationary_startups/1"
NATIVE_OPERATOR = "original_native_acquisition_voltage_no_filter_reference_change_or_regression"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target_geometry(geometry):
    if (not isinstance(geometry, NoiseGeometry)
            or geometry.channels != tuple(BRAIN_SYNC_EEG_CHANNELS)
            or geometry.sfreq != 250. or geometry.unit != "V"
            or geometry.reference != BRAIN_SYNC_ACQUISITION_REFERENCE
            or geometry.observation_rank != 7 or geometry.operator_identity != NATIVE_OPERATOR):
        raise ValueError("target noise requires the actual native seven EEG/250Hz/V/left-earlobe geometry")
    return geometry


def geometry_from_capability(capability):
    cap = require_raw_capability(capability)
    b = cap.public_binding
    return _target_geometry(NoiseGeometry(tuple(b["eeg_channels"]), b["sample_rate_hz"], b["signal_unit"],
        b["reference"], b["operator"], len(b["eeg_channels"])))


@dataclass(frozen=True)
class CalibrationNoiseRecord:
    """One original C record, its actual segment caps and granted event prefix."""

    raw_capabilities: tuple
    labels: object


def _callback_source(callback):
    target = callback if inspect.isfunction(callback) else type(callback)
    path = Path(inspect.getsourcefile(target)).resolve()
    return {"module": target.__module__, "qualname": target.__qualname__,
            "source_path": str(path), "source_sha256": _sha(path)}


def _compile_c_residuals(records, mean_callback):
    residuals, bindings = [], []
    for record in records:
        caps = tuple(require_raw_capability(cap) for cap in record.raw_capabilities)
        labels = require_calibration_labels(record.labels)
        constraints = allowed_candidates(caps, labels)
        label_ids = {label.event_id for label in labels.labels}
        for cap in caps:
            b = cap.public_binding
            if b["role"] != "calibration":
                raise ValueError("noise fitting consumes original calibration records only")
            geometry = geometry_from_capability(cap)
            history = b["public_physical_history"]["exposures"]
            considered = {e["event_id"] for e in history if e["sample"] < b["full_stop"]}
            completed = {e["event_id"] for e in history if e["sample"] < b["full_stop"]
                         and e["exposure_role"] == "completed_protocol_stimulus"}
            if not completed <= label_ids:
                raise ValueError("C noise raw must end before an unlabelled completed exposure")
            # The actual callback receives public/authorized operands only.
            values, error, binding = mean_callback(cap, labels)
            values, error = np.asarray(values), np.asarray(error)
            if values.ndim == 3:
                values, error = values[:, :, 0], error[:, :, 0]
            if (values.shape != cap.eeg.shape or error.shape != values.shape
                    or not np.isfinite(values).all() or not np.isfinite(error).all() or np.any(error < 0)):
                raise ValueError("fixed C mean callback must return its actual complete seven-channel image/error")
            require_sha256(binding.get("source_response_model_identity"), "fixed C mean model")
            if (binding.get("target_response_C_updates_before_noise_fit") != 0
                    or binding.get("mean_role") not in {
                        "original_source_functional_prior_mean", "caller_declared_analytic_fixed_mean"}
                    or set(binding["used_label_event_ids"]) != completed
                    or set(binding["considered_exposure_ids"]) != considered):
                raise ValueError("fixed C mean must bind original source/declared law and exactly authorized exposure labels")
            unknown = {e["event_id"] for e in history if e["event_id"] in considered and e["membership"] is None}
            if unknown and not binding.get("unknown_exposure_law"):
                raise ValueError("unknown C exposures need the actual callback's explicit joint mean law")
            # The compiled mean image itself is the design; I7 is its exact
            # algebraic coefficient chart. No response or nuisance fit occurs.
            residual = ResidualRecord.from_arrays(samples=cap.eeg, design=values,
                kernel_mean=np.eye(7), nuisance=np.zeros_like(values),
                record_id=b["record_segment_id"], participant_key=b["participant_id"],
                recording_id=b["record_id"], sample_start=b["sample_start"], geometry=geometry,
                role="target_calibration", source_model_sha256=binding["source_response_model_identity"],
                source_input_sha256=cap.identity_sha256)
            residuals.append(residual)
            bindings.append({"raw_capability_identity": cap.identity_sha256,
                "label_capability_identity": labels.identity_sha256, "C_constraints": constraints,
                "public_C_scope": {key: b[key] for key in
                    ("participant_id", "record_id", "independent_recording_scope_sha256")},
                "mean_callback": plain_metadata(binding), "mean_image_sha256": array_digest(values),
                "mean_error_sha256": array_digest(error), "residual": residual.ledger(),
                "mean_image_factorization": "D=compiled_mean_image_B=identity7_nuisance=zero",
                "drift_in_residual": "unremoved_as_in_original_frozen_external_mean_C_plugin"})
    return tuple(residuals), bindings


def _calibration_scope(binding):
    participants = {row["public_C_scope"]["participant_id"] for row in binding}
    scopes = {row["public_C_scope"]["independent_recording_scope_sha256"] for row in binding}
    if len(participants) != 1 or len(scopes) != 1:
        raise ValueError("one target C-noise programme uses one actual participant and declared recording split")
    return {"participant_id": next(iter(participants)),
            "independent_recording_scope_sha256": next(iter(scopes)),
            "calibration_record_ids": sorted({row["public_C_scope"]["record_id"] for row in binding})}


@dataclass(frozen=True)
class TargetNoiseOperand:
    noise: FittedChannelNoise
    eta: float
    programme: dict

    def __post_init__(self):
        _target_geometry(self.noise.geometry)
        self.noise.assert_integrity()
        if not np.isfinite(self.eta) or self.eta <= 0 or self.programme["segment_law"] != SEGMENT_LAW:
            raise ValueError("target noise needs an explicit positive variance scale and segment law")
        object.__setattr__(self, "eta", float(self.eta))
        object.__setattr__(self, "programme", freeze_metadata(self.programme))

    @property
    def identity(self):
        return metadata_digest({"noise_state": self.noise.record(), "eta": self.eta, "programme": self.programme})

    def carrier(self):
        return StoredNoiseCarrier(self.noise, eta=self.eta)

    def segment_binding(self, capability, *, noise_segment_id):
        cap = require_raw_capability(capability)
        if geometry_from_capability(cap) != self.noise.geometry:
            raise ValueError("target noise and original observation geometry differ")
        if self.programme["origin"] == "original_frozen_mean_C_residual_plugin":
            scope = self.programme.get("C_scope")
            if scope is None:
                raise ValueError("C-plugin application requires its actual parent recording-scope binding")
            if any(cap.public_binding[key] != scope[key]
                   for key in ("participant_id", "independent_recording_scope_sha256")):
                raise ValueError("target C noise belongs to another participant or original recording split")
        return {"law": SEGMENT_LAW, "noise_segment_id": noise_segment_id,
                "acquisition_segment_id": cap.public_binding["segment_id"], "owner": self.identity}

    def save(self, metadata_path, *, existing_noise_payload=None):
        path = Path(metadata_path)
        if existing_noise_payload is None:
            noise_path = path.with_name(path.stem + ".noise.json")
            self.noise.save(noise_path)
        else:
            noise_path = Path(existing_noise_payload)
            original = FittedChannelNoise.load(noise_path)
            if (noise_path.resolve().parent != path.resolve().parent
                    or metadata_digest(original.record()) != metadata_digest(self.noise.record())):
                raise ValueError("rebound metadata must reuse its actual complete original local noise payload")
        record = {"schema": "n2p3_complete_target_noise_operand/1", "identity": self.identity,
                  "eta": self.eta, "programme": plain_metadata(self.programme),
                  "noise_payload": {"path": noise_path.name, "sha256": _sha(noise_path)},
                  "noise_record_sha256": metadata_digest(self.noise.record()),
                  "adapter_producer_sha256": IMPLEMENTATION_SHA256,
                  "model_only_replay_grants_raw_or_C_authority": False}
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return _sha(path)

    @classmethod
    def load(cls, metadata_path, *, expected_sha256):
        path = Path(metadata_path)
        if _sha(path) != expected_sha256:
            raise ValueError("target noise operand differs from its admitted saved bytes")
        record = json.loads(path.read_bytes())
        if record.get("schema") != "n2p3_complete_target_noise_operand/1":
            raise ValueError("unknown complete target noise operand schema")
        noise_path = path.parent / record["noise_payload"]["path"]
        if _sha(noise_path) != record["noise_payload"]["sha256"]:
            raise ValueError("original complete noise payload differs from its admitted bytes")
        # Preserve the original startup root, compiled temporal maps, covariance
        # and their original producer. Never reconstruct from rho/AR fields.
        result = cls(FittedChannelNoise.load(noise_path), record["eta"], record["programme"])
        if result.identity != record["identity"] or metadata_digest(result.noise.record()) != record["noise_record_sha256"]:
            raise ValueError("complete target noise numerical state changed on cold load")
        return result


def declare_complete_target_noise(noise, *, eta, declaration, segment_law):
    """C0: actual complete original numerical owner and a separate declaration."""
    if not isinstance(noise, FittedChannelNoise) or noise.provenance["fit_scope"] != "analytic_reference":
        raise ValueError("C0 declaration consumes an actual complete declared law, not a fabricated fit receipt")
    if not declaration.get("declaration_id") or not declaration.get("source"):
        raise ValueError("complete C0 law needs its own attributable caller declaration")
    return TargetNoiseOperand(noise, eta, {"origin": "caller_declared_complete_proper_target_noise",
        "declaration": declaration, "segment_law": segment_law, "noise_C_fit_calls": 0,
        "response_C_updates_here": 0, "relative_scale_role": "fixed_caller_eta_variance_once",
        "source_response_prior_rescaled": False, "adapter_producer_sha256": IMPLEMENTATION_SHA256})


def fit_c_plugin_noise(records, *, mean_callback, family, order, conditioning_samples,
                       minimum_samples_per_parameter, eta, segment_law):
    """Original C residual fitter; the next response update uses the original prior."""
    if segment_law != SEGMENT_LAW or not np.isfinite(eta) or eta <= 0:
        raise ValueError("declare the supported independent noise-segment law and positive fixed eta before fitting")
    residuals, binding = _compile_c_residuals(tuple(records), mean_callback)
    scope, callback_source = _calibration_scope(binding), _callback_source(mean_callback)
    noise = FittedChannelNoise.fit(residuals, family=family, order=order,
        conditioning_samples=conditioning_samples, minimum_samples_per_parameter=minimum_samples_per_parameter)
    return TargetNoiseOperand(noise, eta, {"origin": "original_frozen_mean_C_residual_plugin",
        "segment_law": segment_law, "C_inputs": binding, "C_scope": scope,
        "mean_callback_source": callback_source,
        "noise_C_fit_calls": 1, "response_C_updates_here": 0,
        "complete_learner_programme": "C_residual_noise_then_one_response_C_update_given_original_source_prior/1",
        "noise_stage_C_use": "authorized_raw_and_bits_define_fixed_mean_residual_then_Burg_AR_and_tail_LedoitWolf_Sigma",
        "response_stage": "next_owner_updates_original_source_prior_with_authorized_C_once_given_frozen_plugin_noise",
        "response_posterior_reused_as_prior": False, "source_response_prior_rescaled": False,
        "relative_scale_role": "fixed_caller_eta_variance_once", "adapter_producer_sha256": IMPLEMENTATION_SHA256})


def replay_c_residual_inputs(operand, records, *, mean_callback):
    """Parent raw/C replay of actual residual operands, with zero refits."""
    _, binding = _compile_c_residuals(tuple(records), mean_callback)
    if metadata_digest(binding) != metadata_digest(operand.programme["C_inputs"]):
        raise ValueError("C noise parent differs from original authorized mean/raw/label operands")
    return {"C_input_replayed": True, "noise_fits": 0, "response_updates": 0,
            "target_noise_identity": operand.identity}


def rebind_c_scope_from_original_inputs(operand, records, *, mean_callback):
    """Upgrade an original parameter artifact from actual C inputs, zero refits.

    This joins original fitted residual ledgers to canonical C/raw issuance.
    It grants no raw/C capability and preserves every original noise operand.
    """
    residuals, binding = _compile_c_residuals(tuple(records), mean_callback)
    scope = _calibration_scope(binding)
    original = operand.programme["C_inputs"]
    comparable = [{key: row[key] for key in old} for row, old in zip(binding, original, strict=True)]
    if (metadata_digest(comparable) != metadata_digest(original)
            or metadata_digest([r.ledger() for r in residuals]) != metadata_digest(operand.noise.provenance["records"])):
        raise ValueError("original fitted noise residuals differ from actual canonical parent C inputs")
    programme = {**plain_metadata(operand.programme), "C_inputs": binding, "C_scope": scope,
        "scope_rebinding": {"original_parameter_operand_identity": operand.identity,
            "scope_consumer_source_sha256": IMPLEMENTATION_SHA256,
            "actual_mean_callback_source": _callback_source(mean_callback), "noise_parameter_refits": 0}}
    return TargetNoiseOperand(operand.noise, operand.eta, programme)
