"""Source-informed finite selection shapes with one physical record drift.

The original source posterior is used once as the target prior. Gaussian noise
uses its saved stationary map and original relative-noise scale, with lambda
fixed at one. This module grants no access to future target labels.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import weakref
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.channel_noise_mechanism import (
    GaussianQRPosteriorState,
    GaussianResponseImages,
    StructuredGaussianMarginalProgram,
)
from research.finite_response_law import FiniteResponseLaw
from research.frozen_metadata import plain_metadata
from research.history_authority import HistoryUnavailableError
from research.latent_response_reference import ResponseState, selection_design
from research.multichannel_c_latent import _assert_original
from research.response_history_identification import NeighborEventPrefix, finite_conditional_mixture
from transfer.joint_flat_programme import FrozenJointFlatSource, FrozenJointFlatSourcePreparation
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_sha256,
)
from transfer.response_dictionary import ResponseDictionary

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_source_informed_response_model/1"
_ARMS = {"nominal", "mean_shape", "gaussian_moment", "finite_shape"}
_LAWS = {"whole_observed_current_q", "fixed_current_prior_conditional_tail"}


def _bytes(record):
    return json.dumps(plain_metadata(record), sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf8")


@dataclass(frozen=True)
class SourceInformedResponseParameters:
    source: FrozenJointFlatSource
    dictionary: object
    binding: dict

    @classmethod
    def from_source(cls, actual_source, *, target_authority, continuation, continuation_role):
        from transfer.ensemble_views import CalibrationPrefix

        if not isinstance(actual_source, FrozenJointFlatSource) or not isinstance(target_authority, CalibrationPrefix):
            raise TypeError("source response parameters require the original source and actual C authority")
        target_authority.validate()
        target = target_authority.target_contract
        actual_source.assert_target(target)
        if (not isinstance(continuation, (NativeResponseDictionary, ResponseDictionary))
                or continuation.sample_rate_hz != actual_source.parameters.noise.geometry.sfreq
                or array_digest(continuation.matrix) != array_digest(actual_source.parameters.basis)):
            raise ValueError("response continuation must retain the original source native coefficient chart")
        if continuation_role == "new_declared_continuation_of_source_native_arrays":
            if not isinstance(continuation, NativeResponseDictionary):
                raise ValueError("new native continuation requires its independent explicit function owner")
        elif continuation_role == "original_saved_continuous_function":
            from transfer.continuous_decision import ContinuousDecisionModel

            original = ContinuousDecisionModel.load(io.BytesIO(actual_source.original_model_content))
            if (not isinstance(continuation, ResponseDictionary) or original.dictionary is None
                    or original.dictionary.digest() != continuation.digest()):
                raise ValueError("historical function authority must be present in the original source model")
        else:
            raise ValueError("response continuation role must be explicitly declared")
        binding = {"schema": "n2p3_source_informed_response_parameters/1", "source_sha256": actual_source.digest,
            "original_source_binding": actual_source.parameters.binding,
            "canonical_factor_binding": actual_source.canonical_factor_binding,
            "prior_mean_factor_sha256": array_digest(actual_source.parameters.prior.mean,
                                                      actual_source.parameters.prior.root),
            "factor_role": "direct_original_saved_prior_root", "additional_factor_source_error": 0.,
            "dictionary_digest": continuation.digest(), "continuation_role": continuation_role,
            "noise_law": "source_gaussian_lambda_one", "lambda": 1.,
            "noise_scale": actual_source.parameters.hyperparameters.relative_noise_shape,
            "original_gamma_degrees_role": "provenance_only_no_scale_integration",
            "target_contract": target, "parameter_fits": 0, "noise_recompilations": 0,
            "physical_flat_measure": "per_channel_intercept_volts_slope_volts_per_second"}
        return cls(actual_source, continuation, freeze_metadata(binding))

    def digest(self):
        return metadata_digest(self.binding)

    def record(self):
        return {"binding": plain_metadata(self.binding), "source_base64": base64.b64encode(self.source.content).decode("ascii"),
                "dictionary": self.dictionary.record()}

    @classmethod
    def from_record(cls, record, *, source_preparation=None):
        from transfer.continuous_decision import ContinuousDecisionModel

        binding = record["binding"]
        content = base64.b64decode(record["source_base64"], validate=True)
        if source_preparation is not None and not isinstance(source_preparation, FrozenJointFlatSourcePreparation):
            raise TypeError("source response loading requires its actual scoped source preparation owner")
        source = (FrozenJointFlatSource.from_bytes(content, expected_sha256=binding["source_sha256"])
            if source_preparation is None else source_preparation.prepare(content, expected_sha256=binding["source_sha256"]))
        role = binding["continuation_role"]
        if role not in {"new_declared_continuation_of_source_native_arrays", "original_saved_continuous_function"}:
            raise ValueError("saved source response changed its explicit continuation role")
        dictionary = (NativeResponseDictionary.from_record(record["dictionary"])
            if role == "new_declared_continuation_of_source_native_arrays"
            else ResponseDictionary.from_record(record["dictionary"]))
        source.assert_target(binding["target_contract"])
        if role == "original_saved_continuous_function":
            original = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content))
            if original.dictionary is None or original.dictionary.digest() != dictionary.digest():
                raise ValueError("saved historical function is absent from the actual original source")
        if (binding["schema"] != "n2p3_source_informed_response_parameters/1"
                or binding["source_sha256"] != source.digest
                or binding["dictionary_digest"] != dictionary.digest()
                or array_digest(source.parameters.basis) != array_digest(dictionary.matrix)
                or dictionary.sample_rate_hz != source.parameters.noise.geometry.sfreq
                or binding["noise_scale"] != source.parameters.hyperparameters.relative_noise_shape
                or binding["prior_mean_factor_sha256"] != array_digest(source.parameters.prior.mean, source.parameters.prior.root)
                or metadata_digest(binding["original_source_binding"]) != metadata_digest(source.parameters.binding)
                or metadata_digest(binding["canonical_factor_binding"]) != metadata_digest(source.canonical_factor_binding)
                or binding["factor_role"] != "direct_original_saved_prior_root"
                or binding["noise_law"] != "source_gaussian_lambda_one" or binding["lambda"] != 1.
                or binding["additional_factor_source_error"] != 0. or binding["parameter_fits"] != 0
                or binding["noise_recompilations"] != 0
                or binding["original_gamma_degrees_role"] != "provenance_only_no_scale_integration"
                or binding["physical_flat_measure"] != "per_channel_intercept_volts_slope_volts_per_second"):
            raise ValueError("saved source response parameters changed their original numerical law")
        return cls(source, dictionary, freeze_metadata(binding))

    def save(self, path):
        Path(path).write_bytes(_bytes(self.record()))
        return Path(path)

    @classmethod
    def load(cls, path, *, expected_sha256, source_preparation=None):
        return cls.from_bytes(Path(path).read_bytes(), expected_sha256=expected_sha256, source_preparation=source_preparation)

    @classmethod
    def from_bytes(cls, content, *, expected_sha256, source_preparation=None):
        require_sha256(expected_sha256, "source response parameters")
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("source response parameter bytes differ from external digest")
        return cls.from_record(json.loads(content), source_preparation=source_preparation)


@dataclass(frozen=True)
class SourceInformedFiniteResponseProgram:
    law: FiniteResponseLaw
    control_kind: str
    scoring_law: str
    maximum_calibration_components: int
    maximum_prediction_states: int
    maximum_working_bytes: int
    conditioning_offset_samples: int

    def __post_init__(self):
        if (not isinstance(self.law, FiniteResponseLaw) or self.control_kind not in _ARMS
                or self.scoring_law not in _LAWS):
            raise ValueError("source response programme must declare the law, control and scoring measure")
        for name in ("maximum_calibration_components", "maximum_prediction_states", "maximum_working_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("complete state and working-memory budgets must be explicitly positive")
        if type(self.conditioning_offset_samples) is not int:
            raise ValueError("conditioning policy must explicitly declare its native offset from current first onset")

    def record(self):
        return {"law": self.law.record(), "control_kind": self.control_kind, "scoring_law": self.scoring_law,
            **{name: getattr(self, name) for name in ("maximum_calibration_components", "maximum_prediction_states",
                                                    "maximum_working_bytes", "conditioning_offset_samples")}}

    def digest(self):
        return metadata_digest(self.record())

    @classmethod
    def from_record(cls, record):
        return cls(**{**record, "law": FiniteResponseLaw.from_record(record["law"])})


def _physical_nuisance(samples, channels, rate):
    z = np.column_stack((np.ones(samples), np.arange(samples)/rate))
    return np.einsum("nj,cd->ncjd", z, np.eye(channels)).reshape(samples, channels, 2*channels)


def _selections(compiler, prefix, stop):
    """C labels constrain q only where the original issuer authorized them."""
    from data.bnci2014_008_candidate import (
        BNCI2014_008_GRID_SHAPE,
        BNCI2014_008_REPETITIONS_PER_SELECTION,
    )
    from data.candidate_task import MembershipDecision

    raw, target = compiler._record, compiler._arguments["target"]
    occurred = np.flatnonzero(raw.flash_sample < stop)
    selections, allowed = [], {}
    for name in dict.fromkeys(raw.selection_id[occurred]):
        rows = occurred[raw.selection_id[occurred] == name]
        member = MembershipDecision.row_column(raw.candidate_id[rows], n_rows=BNCI2014_008_GRID_SHAPE[0],
            n_columns=BNCI2014_008_GRID_SHAPE[1], repetitions=BNCI2014_008_REPETITIONS_PER_SELECTION,
            event_ids=tuple(f"{target['source_dataset_id']}:{raw.subject_id}:{int(r)}" for r in rows),
            repetition_indices=raw.repetition_index[rows])
        selections.append(NeighborEventPrefix(str(name), member.candidate_vocabulary,
                                              raw.flash_sample[rows], member.membership))
        known = rows < prefix.budget
        possible = (np.all(member.membership[known] == prefix.raw.target_mask[rows[known], None], axis=0)
                    if np.any(known) else np.ones(len(member.candidate_vocabulary), dtype=bool))
        if not possible.any():
            raise ValueError("authorized C binary labels contradict the actual public membership")
        allowed[str(name)] = tuple(map(int, np.flatnonzero(possible)))
    return tuple(selections), allowed


def _selection_record(selection, allowed):
    return {"selection_id": selection.selection_id, "candidate_ids": list(selection.candidate_ids),
        "onsets": selection.onsets.tolist(), "membership": selection.membership.tolist(),
        "allowed_target_indices": list(allowed)}


def _history_selections(capability):
    """Join issuer public events to its authorized C labels by native onset."""
    labels = {}
    for raw in capability.calibration_records:
        if capability.binding["C_record_to_original_record"][raw.record_id] != capability.record_id:
            raise ValueError("source response C labels must belong to this single original physical record")
        if array_digest(raw.samples, raw.onsets, raw.target_mask) != capability.binding["C_record_input_sha256"][raw.record_id]:
            raise ValueError("source response original history changed authorized C arrays")
        for onset, label in zip(raw.onsets, raw.target_mask, strict=True):
            if int(onset) in labels:
                raise ValueError("source response authorized C native clock is duplicated")
            labels[int(onset)] = bool(label)
    selections, allowed = [], {}
    for row in capability.binding["public_selections"]:
        selection = NeighborEventPrefix(row["selection_id"], tuple(row["candidate_vocabulary"]),
            np.asarray(row["onsets"], dtype=np.int64), np.asarray(row["membership"]))
        known = [(index, labels[int(onset)]) for index, onset in enumerate(selection.onsets) if int(onset) in labels]
        possible = np.ones(len(selection.candidate_ids), dtype=bool)
        for index, label in known:
            possible &= selection.membership[index] == label
        if not possible.any():
            raise ValueError("original history C labels contradict actual public membership")
        selections.append(selection)
        allowed[selection.selection_id] = tuple(map(int, np.flatnonzero(possible)))
    return tuple(selections), allowed


def _posterior_record(state):
    if state is None:
        return None
    return {name: (getattr(state, name).tolist() if isinstance(getattr(state, name), np.ndarray)
                   else plain_metadata(getattr(state, name))) for name in state.__dataclass_fields__}


def _one_design(dictionary, selection, target, state, samples):
    matrix, ledger = selection_design(dictionary, selection.onsets, selection.membership[:, target],
        origin_sample=0, observed_samples=samples, state=state)
    error = np.zeros_like(matrix)
    absolute = np.zeros_like(matrix)
    k = matrix.shape[1]//2
    for event in ledger["events"]:
        evaluation = event["response_evaluation"]
        if evaluation is not None:
            start, stop = event["response_sample_centers_half_open"]
            first, last = max(0, start), min(samples, stop)
            basis = (np.asarray(evaluation["matrix"]) if "matrix" in evaluation else
                     dictionary.matrix[first-event["onset_sample"]:last-event["onset_sample"]])
            e = np.asarray(evaluation["error"]) if "error" in evaluation else np.zeros_like(basis)
            error[first:last, :k] += e
            error[first:last, k:] += event["target_indicator"]*e
            absolute[first:last, :k] += np.abs(basis)
            absolute[first:last, k:] += event["target_indicator"]*np.abs(basis)
            event["response_evaluation"] = {"evaluation_sha256": metadata_digest(evaluation),
                "program": evaluation.get("program"), "matrix_sha256": array_digest(basis),
                "error_sha256": array_digest(e), "native_direct_copy": "error" not in evaluation}
    # Account for the actual additions of overlapping event responses.
    eps = np.finfo(float).eps
    gamma = 2*len(selection.onsets)*eps/(1-2*len(selection.onsets)*eps)
    error = np.nextafter((error+gamma*absolute)/(1-gamma), np.inf)
    return matrix, error, ledger


def _design(parameters, program, selections, targets, response, samples):
    width = 2*parameters.dictionary.matrix.shape[1]
    result, error = np.zeros((samples, width)), np.zeros((samples, width))
    ledger = []
    for selection, target, state_index in zip(selections, targets, response, strict=True):
        if program.control_kind == "mean_shape":
            pieces = [_one_design(parameters.dictionary, selection, target, state, samples) for state in program.law.states]
            mean = program.law.mean_design(np.stack([p[0] for p in pieces]),
                                           design_errors=np.stack([p[1] for p in pieces]))
            d, e = mean.matrix, mean.error
            ledger.append({"selection_id": selection.selection_id, "mean_binding": mean.binding,
                           "states": [p[2] for p in pieces]})
        else:
            state = (program.law.states[state_index] if program.control_kind == "finite_shape"
                     else ResponseState(1., 1., 0., 1., 1.))
            d, e, detail = _one_design(parameters.dictionary, selection, target, state, samples)
            ledger.append({"selection_id": selection.selection_id, "state_index": state_index, "design": detail})
        error += e+np.finfo(float).eps*(np.abs(result)+np.abs(d))
        result += d
    return result, error, ledger


def _primitive(parameters, samples, *, transform=None, coefficient_mean=None, coefficient_factor=None,
               factor_error=None, response_images=None):
    original = parameters.source.parameters
    channels = samples.shape[1]
    return StructuredGaussianMarginalProgram(original.noise, samples,
        original.prior.mean.reshape(-1, channels) if coefficient_mean is None else coefficient_mean,
        original.prior.root if coefficient_factor is None else coefficient_factor,
        _physical_nuisance(len(samples), channels, parameters.dictionary.sample_rate_hz),
        intercept_chart=True, factor_source_error=factor_error, flat_coordinate_transform=transform,
        noise_scale=parameters.binding["noise_scale"], response_images=response_images)


def _moment_images(parameters, program, selections, targets, samples):
    original = parameters.source.parameters
    channels, rank = len(original.noise.geometry.channels), original.prior.root.shape[1]
    components = program.law.moment_components(original.prior.mean, original.prior.root)
    width = components.local_factor.shape[1]
    mean, mean_error = np.zeros((samples, channels)), np.zeros((samples, channels))
    image = np.zeros((samples, channels, rank+len(selections)*width))
    error = np.zeros_like(image)
    ledgers, bindings = [], []
    eps = np.finfo(float).eps
    for index, (selection, target) in enumerate(zip(selections, targets, strict=True)):
        pieces = [_one_design(parameters.dictionary, selection, target, state, samples) for state in program.law.states]
        contracted = components.contract_designs(np.stack([p[0] for p in pieces]), channels=channels,
                                                design_errors=np.stack([p[1] for p in pieces]))
        mean_error += contracted.mean_error+eps*(np.abs(mean)+np.abs(contracted.mean))
        mean += contracted.mean
        error[:, :, :rank] += contracted.shared_image_error+eps*(np.abs(image[:, :, :rank])+np.abs(contracted.shared_image))
        image[:, :, :rank] += contracted.shared_image
        local = slice(rank+index*width, rank+(index+1)*width)
        image[:, :, local], error[:, :, local] = contracted.local_image, contracted.local_image_error
        bindings.append(contracted.binding)
        ledgers.append({"selection_id": selection.selection_id, "states": [p[2] for p in pieces]})
    ids = tuple(f"global_source:{i}" for i in range(rank))+tuple(
        f"selection_local:{s.selection_id}:{i}" for s in selections for i in range(width))
    return GaussianResponseImages(mean, image, mean_error, error, ids,
        {"source_parameters_digest": parameters.digest(), "law_digest": program.law.digest(),
         "components": components.metadata(), "contracted_selections": bindings}), ledgers


def _proper_dimensions(parameters, program, selection_count):
    rank = parameters.source.parameters.prior.root.shape[1]
    return (rank+selection_count*len(program.law.states)*(rank+1)
            if program.control_kind == "gaussian_moment" else rank)


def _axes(program, selections, allowed):
    shapes = range(len(program.law.states)) if program.control_kind == "finite_shape" else (0,)
    return [tuple(product(allowed[s.selection_id], shapes)) for s in selections]


def _state_count(program, selections, allowed):
    shapes = len(program.law.states) if program.control_kind == "finite_shape" else 1
    return prod(len(allowed[s.selection_id])*shapes for s in selections)


def _log_prior(program, selections, allowed, response):
    value = sum(-np.log(len(allowed[s.selection_id])) for s in selections)
    error = sum(2*np.finfo(float).eps*abs(float(np.log(len(allowed[s.selection_id])))) for s in selections)
    if program.control_kind == "finite_shape":
        value += sum(program.law.log_probability[i] for i in response)
        error += sum(program.law.log_probability_error[i] for i in response)
    error += (2*len(selections)+2)*np.finfo(float).eps*(1+abs(value))
    return float(value), float(error)


def _budget(program, count, *, prediction, samples, channels, proper):
    maximum = program.maximum_prediction_states if prediction else program.maximum_calibration_components
    if count > maximum:
        raise HistoryUnavailableError(f"complete_positive_response_state_budget:{count}>{maximum}")
    # One component is streamed; saved C roots and input/output QR workspace remain live.
    required = 8*(samples*channels*(3*(proper+2*channels)+8)+5*(proper+2*channels)**2)
    if not prediction:
        required += 8*count*(proper+2*channels)*(proper+2*channels+1)
    if required > program.maximum_working_bytes:
        raise HistoryUnavailableError(f"source_response_working_memory_budget:{required}>{program.maximum_working_bytes}")


@dataclass(frozen=True)
class SourceInformedResponseModel:
    parameters: SourceInformedResponseParameters
    program: SourceInformedFiniteResponseProgram
    posteriors: tuple
    components: tuple
    binding: dict
    _actual_c_verifications: dict = field(default_factory=dict, init=False, compare=False, repr=False)
    _operational_counts: dict = field(default_factory=lambda: {"C_QR_reconstruction_attempts": 0, "C_QR_reconstructions": 0,
        "C_verification_cache_hits": 0}, init=False, compare=False, repr=False)

    def operational_receipt(self):
        return freeze_metadata(dict(self._operational_counts))

    @property
    def provenance(self):
        return self.binding

    def with_fit_binding(self, binding):
        return replace(self, binding=freeze_metadata({**self.binding, "ensemble_fit_binding": binding})).validate()

    def _verify_actual_c(self, capability):
        """Reconstruct actual C numerics once; preserve every saved state byte."""
        prefix = capability._prefix
        key = metadata_digest({"model_digest": self.digest(), "parameters_digest": self.parameters.digest(),
            "prefix_identity": self.binding["prefix_identity"], "C_input_sha256": self.binding["C_input_sha256"],
            "consumer_sha256": IMPLEMENTATION_SHA256, "verification": "actual_C_QR_reconstruction_compare_preserve/1"})
        if key in self._actual_c_verifications:
            self._operational_counts["C_verification_cache_hits"] += 1
            return self._actual_c_verifications[key], 0
        stop = self.calibration_samples
        if not stop:
            record = freeze_metadata({"method": "source_prior_only_no_empty_C_drift_posterior", "C_QR_reconstructions": 0,
                                      "saved_state_replacements": 0, "source_parameter_fits": 0, "target_parameter_fits": 0})
            self._actual_c_verifications[key] = record
            return record, 0
        if (prefix.raw is None or len(prefix.raw.samples) != stop
                or array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask) != self.binding["C_input_sha256"]):
            raise ValueError("saved source response C posterior differs from actual original C arrays")
        selections, allowed = _history_selections(capability._original_scope)
        actual = []
        for s in selections:
            occurred = s.onsets < stop
            if occurred.any():
                actual.append(NeighborEventPrefix(s.selection_id, s.candidate_ids, s.onsets[occurred], s.membership[occurred]))
        selections = tuple(actual)
        if metadata_digest([_selection_record(s, allowed[s.selection_id]) for s in selections]) != metadata_digest(self.binding["C_selections"]):
            raise ValueError("saved source response C posterior changed actual public membership or legal C targets")
        primitive = None if self.program.control_kind == "gaussian_moment" else _primitive(self.parameters, prefix.raw.samples)
        channels = prefix.raw.samples.shape[1]
        comparisons = []
        for component, saved in zip(self.components, self.posteriors, strict=True):
            targets, response = component["target_indices"], component["response_state_indices"]
            if self.program.control_kind == "gaussian_moment":
                images, _ = _moment_images(self.parameters, self.program, selections, targets, stop)
                primitive = _primitive(self.parameters, prefix.raw.samples, response_images=images)
                design = np.zeros((stop, len(self.parameters.source.parameters.prior.mean)//channels))
                self._operational_counts["C_QR_reconstruction_attempts"] += 1
                rebuilt = primitive.posterior_state(design)
            else:
                design, error, _ = _design(self.parameters, self.program, selections, targets, response, stop)
                self._operational_counts["C_QR_reconstruction_attempts"] += 1
                rebuilt = primitive.posterior_state(design, design_source_error=error)
            self._operational_counts["C_QR_reconstructions"] += 1
            names = ("precision_root", "precision_response", "flat_transform")
            if (any(not np.array_equal(getattr(saved, name), getattr(rebuilt, name)) for name in names)
                    or saved.log_marginal != rebuilt.log_marginal
                    or saved.raw_prefix_digest != rebuilt.raw_prefix_digest or saved.chart_digest != rebuilt.chart_digest
                    or saved.noise_digest != rebuilt.noise_digest or saved.proper_dimensions != rebuilt.proper_dimensions):
                raise ValueError("saved source response posterior differs from deterministic actual C QR reconstruction")
            if saved.resolution["C"]["absolute_log_density_error_estimate"] < rebuilt.resolution["C"]["absolute_log_density_error_estimate"]:
                raise ValueError("saved source response C error understates actual C QR reconstruction")
            comparisons.append({"saved_arrays_sha256": array_digest(*(getattr(saved, name) for name in names)),
                                "reconstruction_arrays_sha256": array_digest(*(getattr(rebuilt, name) for name in names))})
        record = freeze_metadata({"method": "actual_C_QR_reconstruction_compare_preserve/1", "C_QR_reconstructions": len(comparisons),
            "saved_state_replacements": 0, "source_parameter_fits": 0, "target_parameter_fits": 0,
            "comparisons": comparisons, "verification_key": key})
        self._actual_c_verifications[key] = record
        return record, len(comparisons)

    @property
    def calibration_samples(self):
        return self.binding["calibration_samples"]

    def record(self):
        return {"schema": _SCHEMA, "parameters_digest": self.parameters.digest(), "program": self.program.record(),
            "posteriors": [_posterior_record(s) for s in self.posteriors],
            "components": plain_metadata(self.components), "binding": plain_metadata(self.binding)}

    def digest(self):
        return metadata_digest(self.record())

    def save(self, path):
        Path(path).write_bytes(_bytes({**self.record(), "parameters": self.parameters.record()}))
        return Path(path)

    @classmethod
    def load(cls, path, *, expected_sha256, source_preparation=None):
        return cls.from_bytes(Path(path).read_bytes(), expected_sha256=expected_sha256, source_preparation=source_preparation)

    @classmethod
    def from_bytes(cls, content, *, expected_sha256, source_preparation=None):
        require_sha256(expected_sha256, "source response model")
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("source response model bytes differ from external digest")
        record = json.loads(content)
        parameters = SourceInformedResponseParameters.from_record(record["parameters"], source_preparation=source_preparation)
        if record["schema"] != _SCHEMA or record["parameters_digest"] != parameters.digest():
            raise ValueError("source response model changed its source/parameter binding")
        model = cls(parameters, SourceInformedFiniteResponseProgram.from_record(record["program"]),
            tuple(None if s is None else GaussianQRPosteriorState(**s) for s in record["posteriors"]),
            tuple(freeze_metadata(s) for s in record["components"]), freeze_metadata(record["binding"]))
        model.validate()
        return model

    def validate(self):
        if (self.binding["schema"] != _SCHEMA or self.binding["parameters_digest"] != self.parameters.digest()
                or self.binding["program_digest"] != self.program.digest()
                or len(self.posteriors) != len(self.components) or not self.components):
            raise ValueError("source response model changed its complete posterior law")
        empty = self.calibration_samples == 0
        if any((s is None) != empty for s in self.posteriors):
            raise ValueError("empty C retains source prior and flat measure, never a normalized drift posterior")
        expected = tuple(product(*_axes(self.program,
            tuple(NeighborEventPrefix(s["selection_id"], tuple(s["candidate_ids"]), np.asarray(s["onsets"], dtype=int),
                                      np.asarray(s["membership"])) for s in self.binding["C_selections"]),
            {s["selection_id"]: s["allowed_target_indices"] for s in self.binding["C_selections"]})))
        actual = tuple(tuple(zip(c["target_indices"], c["response_state_indices"], strict=True)) for c in self.components)
        if actual != expected:
            raise ValueError("source response model removed or changed a positive C component")
        priors, potentials, errors = [], [], []
        for c, state in zip(self.components, self.posteriors, strict=True):
            prior = sum(-np.log(len(s["allowed_target_indices"])) for s in self.binding["C_selections"])
            if self.program.control_kind == "finite_shape":
                prior += sum(self.program.law.log_probability[i] for i in c["response_state_indices"])
            if c["log_prior"] != prior:
                raise ValueError("source response C component changed its original effective mass")
            priors.append(prior)
            potentials.append(0. if state is None else state.log_marginal)
            allowed = {s["selection_id"]: s["allowed_target_indices"] for s in self.binding["C_selections"]}
            lightweight = tuple(NeighborEventPrefix(s["selection_id"], tuple(s["candidate_ids"]),
                np.asarray(s["onsets"], dtype=int), np.asarray(s["membership"])) for s in self.binding["C_selections"])
            _, prior_error = _log_prior(self.program, lightweight, allowed, c["response_state_indices"])
            error = prior_error+(0. if state is None else state.resolution["C"]["absolute_log_density_error_estimate"])
            if not np.isfinite(error) or error < 0:
                raise ValueError("saved source response C component error is invalid")
            errors.append(error)
        logs = np.asarray(priors)+potentials
        normalization_error = (len(logs)+8)*np.finfo(float).eps*(1+max(map(abs, logs)))
        for c, e in zip(self.components, errors, strict=True):
            if not np.isfinite(c["log_posterior_weight_error"]) or c["log_posterior_weight_error"] < e+max(errors)+normalization_error:
                raise ValueError("saved source response C log-weight error understates its actual numerical law")
        logs -= np.max(logs)
        expected_weights = logs-logsumexp(logs)
        if not np.array_equal(expected_weights, [c["log_posterior_weight"] for c in self.components]):
            raise ValueError("source response saved component weights differ from actual C potentials")
        return self

    def history_scope(self, compiler, view, *, prefix, conditioning_stop):
        from transfer.ensemble_views import calibration_prefix_identity

        prefix = compiler.assert_original_prefix(prefix)
        if not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0):
            raise HistoryUnavailableError("source_response_current_schedule_incomplete", failure_kind="incomplete")
        if calibration_prefix_identity(prefix) != self.binding["prefix_identity"]:
            raise ValueError("source response history must retain its actual original C prefix")
        first_onset = int(view.raw_scope[0])+int(view.onsets[0])
        if conditioning_stop != first_onset+self.program.conditioning_offset_samples:
            raise ValueError("source response P cutoff differs from the explicit programme conditioning policy")
        cap = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=conditioning_stop)
        selections, allowed = _history_selections(cap)
        result = object.__new__(SourceResponseHistory)
        for key, value in {"model_digest": self.digest(), "view": view, "samples": cap.samples,
            "selections": selections, "allowed": freeze_metadata(allowed),
            "conditioning_samples": conditioning_stop, "binding": cap.binding,
            "_compiler": compiler, "_prefix": prefix, "_original_scope": cap}.items():
            object.__setattr__(result, key, value)
        if not hasattr(compiler, "_source_response_history_scopes"):
            compiler._source_response_history_scopes = {}
        registry, identifier = compiler._source_response_history_scopes, id(result)
        def release(reference):
            current = registry.get(identifier)
            if current is not None and current[0] is reference:
                registry.pop(identifier, None)
        reference = weakref.ref(result, release)
        registry[identifier] = (reference, cap,
            metadata_digest([_selection_record(s, allowed[s.selection_id]) for s in selections]),
            metadata_digest(cap.binding), conditioning_stop)
        return result

    def score_history(self, view, capability):
        return score_source_response_history(self, view, capability)


@dataclass(frozen=True, init=False)
class SourceResponseHistory:
    model_digest: str
    view: object
    samples: np.ndarray
    selections: tuple
    allowed: dict
    conditioning_samples: int
    binding: dict
    _compiler: object
    _prefix: object
    _original_scope: object


def _assert_history_scope(model, view, capability):
    from transfer.ensemble_views import _record_history_key, calibration_prefix_identity

    if not isinstance(capability, SourceResponseHistory):
        raise TypeError("source response score requires its live original-issuer capability")
    compiler = capability._compiler
    registered = getattr(compiler, "_source_response_history_scopes", {}).get(id(capability))
    if (registered is None or registered[0]() is not capability or registered[1] is not capability._original_scope
            or registered[3] != metadata_digest(capability.binding)
            or registered[4] != capability.conditioning_samples):
        raise ValueError("source response history was not registered unchanged by its actual original issuer")
    prefix = compiler.assert_original_prefix(capability._prefix)
    if (calibration_prefix_identity(prefix) != model.binding["prefix_identity"]
            or capability.model_digest != model.digest()
            or metadata_digest(view.record()) != metadata_digest(capability.view.record())
            or capability.samples is not capability._original_scope.samples
            or array_digest(capability.samples) != capability.binding["samples_sha256"]):
        raise ValueError("source response scoring differs from its actual C, current view or original samples")
    original = getattr(compiler, "_record_history_scopes", {}).get(_record_history_key(capability._original_scope.binding))
    if original is None or metadata_digest(original.binding) != registered[3]:
        raise ValueError("source response original history readset is no longer registered by the actual issuer")
    selections, allowed = _history_selections(capability._original_scope)
    expected = metadata_digest([_selection_record(s, allowed[s.selection_id]) for s in selections])
    actual = metadata_digest([_selection_record(s, capability.allowed[s.selection_id]) for s in capability.selections])
    if (expected != registered[2] or actual != expected
            or metadata_digest(capability.allowed) != metadata_digest(allowed)
            or capability.binding["conditioning_sample_support"] != (0, capability.conditioning_samples)):
        raise ValueError("source response history changed actual public q membership or C/P/T conditioning")


def fit_source_informed_response(prefix, *, compiler, parameters, program):
    from transfer.ensemble_views import CalibrationPrefixCompiler, calibration_prefix_identity

    if type(compiler) is not CalibrationPrefixCompiler:
        raise TypeError("first source response producer requires the original BNCI single-record compiler")
    prefix = _assert_original(compiler, prefix)
    if not isinstance(parameters, SourceInformedResponseParameters) or not isinstance(program, SourceInformedFiniteResponseProgram):
        raise TypeError("source response fit needs actual parameter and programme owners")
    parameters.source.assert_target(prefix.target_contract)
    if metadata_digest(parameters.binding["target_contract"]) != metadata_digest(prefix.target_contract):
        # K changes raw_scope/label budget; physical target authority is invariant.
        keys = ("participant_key", "source_sha256", "channel_names", "reference", "sample_rate_hz")
        if any(parameters.binding["target_contract"][k] != prefix.target_contract[k] for k in keys):
            raise ValueError("source response fit target differs from its original parameter authority")
    stop = 0 if prefix.raw is None else len(prefix.raw.samples)
    selections, allowed = _selections(compiler, prefix, stop)
    count = _state_count(program, selections, allowed)
    channels = len(parameters.source.parameters.noise.geometry.channels)
    _budget(program, count, prediction=False, samples=stop, channels=channels,
            proper=_proper_dimensions(parameters, program, len(selections)))
    axes = _axes(program, selections, allowed)
    if stop and stop <= parameters.source.parameters.noise.conditioning_samples:
        raise HistoryUnavailableError("actual_C_cannot_identify_source_noise_startup_and_physical_drift")
    primitive = None if not stop or program.control_kind == "gaussian_moment" else _primitive(parameters, prefix.raw.samples)
    posteriors, components, logs, log_errors = [], [], [], []
    for configuration in product(*axes):
        targets, response = tuple(x[0] for x in configuration), tuple(x[1] for x in configuration)
        prior, prior_error = _log_prior(program, selections, allowed, response)
        if stop:
            if program.control_kind == "gaussian_moment":
                images, ledger = _moment_images(parameters, program, selections, targets, stop)
                design = np.zeros((stop, len(parameters.source.parameters.prior.mean)//channels))
                posterior = _primitive(parameters, prefix.raw.samples, response_images=images).posterior_state(design)
            else:
                design, error, ledger = _design(parameters, program, selections, targets, response, stop)
                posterior = primitive.posterior_state(design, design_source_error=error)
        else:
            posterior, ledger = None, []
        posteriors.append(posterior)
        logs.append(prior+(0. if posterior is None else posterior.log_marginal))
        log_errors.append(prior_error+(0. if posterior is None else
                          posterior.resolution["C"]["absolute_log_density_error_estimate"]))
        components.append({"target_indices": targets, "response_state_indices": response,
                           "log_prior": prior, "design_ledger": ledger})
    centered = np.asarray(logs)-max(logs)
    weights = centered-logsumexp(centered)
    normalization_error = (len(logs)+8)*np.finfo(float).eps*(1+max(map(abs, logs)))
    components = tuple(freeze_metadata({**c, "log_posterior_weight": float(w),
        "log_posterior_weight_error": float(e+max(log_errors)+normalization_error)})
        for c, w, e in zip(components, weights, log_errors, strict=True))
    binding = {"schema": _SCHEMA, "parameters_digest": parameters.digest(), "program_digest": program.digest(),
        "producer_sha256": IMPLEMENTATION_SHA256, "prefix_identity": calibration_prefix_identity(prefix),
        "calibration_samples": stop, "C_selections": [_selection_record(s, allowed[s.selection_id]) for s in selections],
        "C_input_sha256": None if not stop else array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask),
        "source_prior_applications": 1, "parameter_fits": 0, "noise_fits": 0, "lambda_integrations": 0,
        "C_state": "source_prior_only_common_flat_measure" if not stop else "normalized_joint_C_posterior",
        "positive_components": count, "fit_label_event_support": [0, prefix.budget], "fit_read_support": [0, stop]}
    return SourceInformedResponseModel(parameters, program, tuple(posteriors), components, freeze_metadata(binding)).validate()


def score_source_response_history(model, view, capability):
    _assert_history_scope(model, view, capability)
    if (model.calibration_samples == 0 and model.program.scoring_law == "fixed_current_prior_conditional_tail"
            and capability.conditioning_samples < 2):
        raise HistoryUnavailableError("source_only_conditional_prefix_cannot_identify_physical_flat", failure_kind="numerical",
            evidence={"conditioning_samples": capability.conditioning_samples,
                "physical_flat_columns": 2*capability.samples.shape[1],
                "prefix_scalar_rows": capability.conditioning_samples*capability.samples.shape[1]})
    c_verification, _ = model._verify_actual_c(capability)
    names = tuple(s["selection_id"] for s in model.binding["C_selections"])
    c_selections = tuple(next(s for s in capability.selections if s.selection_id == name) for name in names)
    future = tuple(s for s in capability.selections if s.selection_id not in names)
    if view.decision_id not in {s.selection_id for s in future}:
        raise ValueError("current held selection cannot also own C labels")
    current_axis = tuple(s.selection_id for s in future).index(view.decision_id)
    count = len(model.components)*_state_count(model.program, future, capability.allowed)
    _budget(model.program, count, prediction=True, samples=len(capability.samples), channels=capability.samples.shape[1],
            proper=_proper_dimensions(model.parameters, model.program, len(capability.selections)))
    axes = _axes(model.program, future, capability.allowed)
    transform = None if model.posteriors[0] is None else model.posteriors[0].flat_transform
    primitive = (None if model.program.control_kind == "gaussian_moment"
                 else _primitive(model.parameters, capability.samples, transform=transform))
    indices, current, logs, pairs, c_errors, whole_values, whole_errors, whole_certificates = [], [], [], [], [], [], [], []
    whole_primitives = []
    ledgers = []
    for component_id, (component, posterior) in enumerate(zip(model.components, model.posteriors, strict=True)):
        for configuration in product(*axes):
            targets, response = tuple(x[0] for x in configuration), tuple(x[1] for x in configuration)
            all_targets = (*component["target_indices"], *targets)
            all_response = (*component["response_state_indices"], *response)
            if model.program.control_kind == "gaussian_moment":
                images, ledger = _moment_images(model.parameters, model.program, (*c_selections, *future),
                                                all_targets, len(capability.samples))
                primitive = _primitive(model.parameters, capability.samples, transform=transform, response_images=images)
                design = np.zeros((len(capability.samples), primitive.mean.shape[0]))
                error = None
                continued = None if posterior is None else primitive.extend_posterior(design, posterior)
            else:
                design, error, ledger = _design(model.parameters, model.program, (*c_selections, *future),
                                                all_targets, all_response, len(capability.samples))
                continued = posterior
            future_prior, prior_error = _log_prior(model.program, future, capability.allowed, response)
            prior = component["log_posterior_weight"]+future_prior
            prior_error += component["log_posterior_weight_error"]
            if model.program.scoring_law == "whole_observed_current_q":
                if posterior is None:
                    # Empty C carries the common physical flat measure. Whole
                    # probabilities require full identification, independent of P.
                    full = primitive.posterior_state(design, design_source_error=error)
                    value, certificate = full.log_marginal, full.resolution["C"]
                else:
                    full = primitive.whole_increment(design, continued, design_source_error=error)
                    value, certificate = full.log_marginal, full.resolution["full"]
                whole_values.append(value)
                whole_errors.append(certificate["absolute_log_density_error_estimate"]+prior_error)
                whole_certificates.append({**certificate, "source_sha256": full.resolution["source_sha256"]})
                whole_primitives.append({"kind": ("common_flat_full_potential" if posterior is None
                    else "normalized_C_full_increment"), "log_marginal": value,
                    "resolution": full.resolution})
            else:
                pair = (primitive.evaluate(design, design_source_error=error,
                            conditioning_samples=capability.conditioning_samples) if posterior is None
                        else primitive.continue_posterior(design, continued,
                            conditioning_samples=capability.conditioning_samples, design_source_error=error))
                pairs.append(pair)
                c_errors.append(prior_error)
            indices.append((component_id, *targets, *response))
            current.append(targets[current_axis])
            logs.append(prior)
            ledgers.append(ledger)
    if whole_values:
        prior_masses = np.array([logsumexp(np.asarray(logs)[np.asarray(current) == q])
            for q in range(len(view.decision.candidate_vocabulary))])
        if not np.allclose(prior_masses, -np.log(len(prior_masses)), rtol=0, atol=128*np.finfo(float).eps):
            raise ValueError("whole response states must retain their declared uniform current q prior")
        terms = np.asarray(logs)+whole_values
        anchor = float(np.max(terms))
        terms -= anchor
        by_q = np.array([logsumexp(terms[np.asarray(current) == q]) for q in range(len(view.decision.candidate_vocabulary))])
        absolute_by_q = by_q+anchor
        by_q -= np.max(by_q)
        log_probability = by_q-logsumexp(by_q)
        operations = (len(logs)+16)*np.finfo(float).eps
        if operations >= 1:
            raise HistoryUnavailableError("complete_whole_response_reduction_unrepresentable", failure_kind="numerical")
        arithmetic_error = operations/(1-operations)*(1+abs(anchor)+4*max(map(abs, logs))
            +4*max(map(abs, whole_values))+2*np.log(len(logs)))
        probability_error = 2*(max(whole_errors)+arithmetic_error)
        increment_rows = (len(capability.samples)-model.calibration_samples)*capability.samples.shape[1]
        budget = increment_rows*np.sqrt(np.finfo(float).eps)
        if not np.isfinite(probability_error) or probability_error > budget:
            raise HistoryUnavailableError("whole_response_probability_numerically_unresolved", failure_kind="numerical",
                evidence={"normalized_log_probability_error_estimate": probability_error,
                          "absolute_log_density_error_budget": budget, "component_errors": whole_errors})
        evidence = WholeResponseEvidence(frozen_array(log_probability), frozen_array(absolute_by_q),
            frozen_array(terms-logsumexp(terms)), freeze_metadata({"component_errors": whole_errors,
                "component_full_certificates": whole_certificates,
                "component_primitives": whole_primitives,
                "normalized_log_probability_error_estimate": probability_error,
                "absolute_log_density_error_budget": budget,
                "actual_increment_scalar_rows": increment_rows,
                "normalization_arithmetic_error_estimate": arithmetic_error,
                "measure": ("common_physical_flat_relative_full_potential" if model.calibration_samples == 0
                    else "normalized_actual_C_posterior_ordinary_full_increment_density"),
                "P_identification_required": False, "conditional_ratio_computed": False}))
    else:
        evidence = finite_conditional_mixture(view.decision.candidate_vocabulary, indices, current, logs, pairs,
            tail_scalar_rows=(len(capability.samples)-capability.conditioning_samples)*capability.samples.shape[1],
            c_logmass_errors=c_errors, scoring_law=model.program.scoring_law)
    binding = freeze_metadata({"schema": "n2p3_source_response_prediction/1", "model_digest": model.digest(),
        "history_digest": metadata_digest(capability.binding), "consumer_sha256": IMPLEMENTATION_SHA256,
        "scoring_law": model.program.scoring_law, "control_kind": model.program.control_kind,
        "evaluated_positive_states": len(indices), "state_indices": indices, "state_design_ledgers": ledgers,
        "log_probability": evidence.log_candidate_probability.tolist(), "C_refits": 0, "noise_fits": 0,
        "actual_C_verification": c_verification,
        "numerical_resolution": (evidence.resolution if isinstance(evidence, WholeResponseEvidence)
                                  else evidence.mixture_resolution),
        "source_prior_reapplications": 0, "noise_recompilations": 0, "lambda_integrations": 0,
        "read_sample_support": capability.binding["read_sample_support"],
        "conditioning_sample_support": capability.binding["conditioning_sample_support"],
        "score_sample_support": capability.binding["score_sample_support"], "future_truth_consumed": False})
    return evidence, binding


@dataclass(frozen=True)
class WholeResponseEvidence:
    log_candidate_probability: np.ndarray
    log_evidence: np.ndarray
    state_log_weights: np.ndarray
    resolution: dict


SourceOnlyWholeEvidence = WholeResponseEvidence
