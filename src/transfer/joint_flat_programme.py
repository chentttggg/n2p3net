"""Typed frozen source and sampling programmes for the joint-history expert.

Original producer bytes stay separate from consumer noise degrees of freedom,
sampling and q scoring law. Source loading changes neither the saved producer
nor its covariance. These declarations grant no target sample/label access.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import (
    FittedChannelNoise,
    NoiseGeometry,
    _response_resolved_root,
)
from research.joint_flat_authority import JointFlatParameters
from research.joint_response_scale import GlobalGaussianPrior, _degrees
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_sha256,
)
from transfer.temporal_noise import TemporalNoise


def joint_flat_numeric_producer_identity():
    """Bind each new C/predictive state to the numerical code that made it."""
    root = Path(__file__).resolve().parents[1]
    names = ("research/joint_flat_chain.py", "research/joint_flat_factor_history.py",
        "research/joint_flat_chart_precision.py",
        "research/joint_flat_history_builder.py", "research/joint_flat_collapsed_predictive.py",
        "research/joint_flat_sparse.py", "research/channel_noise_mechanism.py",
        "research/joint_response_scale.py", "transfer/joint_flat_programme.py", "transfer/joint_flat_model.py")
    return {"schema": "n2p3_joint_flat_numeric_producer/1", "modules_sha256": {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}


class FrozenJointFlatSourcePreparation:
    """One ingress owns one mathematical decode of each original source codec."""

    def __init__(self):
        self._sources = {}

    def prepare(self, payload, *, expected_sha256):
        require_sha256(expected_sha256, "actual joint source checkpoint")
        if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("joint source checkpoint differs from original declared bytes")
        if expected_sha256 not in self._sources:
            self._sources[expected_sha256] = FrozenJointFlatSource.from_bytes(
                payload, expected_sha256=expected_sha256)
        return self._sources[expected_sha256]


@dataclass(frozen=True)
class JointFlatSamplingProgram:
    degrees_of_freedom: float
    burnin: int
    draws_per_chain: int
    chain_seeds: tuple[int, ...]
    sweeps_per_draw: int
    batch_size: int
    start_scale: float
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    max_log_probability_mcse: float
    leaf_rows: int = 2048
    qr_cache_bytes: int = 67108864
    kernel: str = "original_shared_source_beta_drift_blocks_Gibbs"
    predictive_integration: str = "collapsed_beta_all_drift_current_q"

    def __post_init__(self):
        nu = _degrees(self.degrees_of_freedom)
        seeds = tuple(self.chain_seeds)
        if (not seeds or len(seeds) < 2 or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or type(self.burnin) is not int or self.burnin < 0
                or type(self.draws_per_chain) is not int or self.draws_per_chain < 8
                or type(self.sweeps_per_draw) is not int or self.sweeps_per_draw < 1
                or type(self.batch_size) is not int or self.batch_size < 2
                or self.draws_per_chain%self.batch_size or type(self.leaf_rows) is not int or self.leaf_rows < 1
                or type(self.qr_cache_bytes) is not int or self.qr_cache_bytes < 0
                or self.kernel != "original_shared_source_beta_drift_blocks_Gibbs"
                or self.predictive_integration not in {"collapsed_beta_all_drift_current_q",
                                                       "current_q_Rao_Blackwell_at_retained_coefficients"}
                or not np.isfinite([self.start_scale, self.max_rhat, self.min_bulk_ess,
                    self.min_tail_ess, self.max_log_probability_mcse]).all()
                or self.start_scale <= 0 or self.max_rhat <= 0 or min(self.min_bulk_ess, self.min_tail_ess) < 0
                or self.max_log_probability_mcse <= 0):
            raise ValueError("explicit overdispersed chains, finite MC precision policy and QR resource programme are required")
        object.__setattr__(self, "degrees_of_freedom", nu)
        object.__setattr__(self, "chain_seeds", seeds)

    def record(self):
        return {"schema": "n2p3_joint_flat_sampling_programme/1", "degrees_of_freedom": str(self.degrees_of_freedom),
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "degrees_of_freedom"},
            "method": "joint_explicit_flat_drift_discrete_targets_Gibbs_prefix_only",
            "noise_scale_scope": "one_Gamma_per_original_true_record",
            "response_prior_scope": "one_global_proper_beta_prior",
            "random_stream_derivation": "SeedSequence_root_spawn_Cinit_Ctransition_Pinit_Ptransition/1",
            "tail_read_for_posterior_fit": False}

    def stream_seeds(self, phase):
        phases = {"C_initialisation": 0, "C_transition": 1, "P_initialisation": 2, "P_transition": 3}
        result = []
        for seed in self.chain_seeds:
            words = np.random.SeedSequence(seed).spawn(4)[phases[phase]].generate_state(2, dtype=np.uint32)
            result.append((int(words[0]) << 32) | int(words[1]))
        return tuple(result)

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        if (values.pop("schema") != "n2p3_joint_flat_sampling_programme/1"
                or values.pop("method") != "joint_explicit_flat_drift_discrete_targets_Gibbs_prefix_only"
                or values.pop("noise_scale_scope") != "one_Gamma_per_original_true_record"
                or values.pop("response_prior_scope") != "one_global_proper_beta_prior"
                or values.pop("random_stream_derivation") != "SeedSequence_root_spawn_Cinit_Ctransition_Pinit_Ptransition/1"
                or values.pop("tail_read_for_posterior_fit") is not False):
            raise ValueError("saved joint sampler changed its original mathematical programme")
        values["degrees_of_freedom"] = float(values["degrees_of_freedom"])
        return cls(**values)


@dataclass(frozen=True)
class JointFlatProgram:
    sampler: JointFlatSamplingProgram
    conditioning_samples: int
    scoring_law: str
    target_prior: str = "independent_uniform_targets_conditioned_on_actual_C_labels"
    source_exclusion_scope: str = "target_participant"
    conditioning_anchor: str = "first_current_event_native_clock"

    def __post_init__(self):
        if (not isinstance(self.sampler, JointFlatSamplingProgram)
                or type(self.conditioning_samples) is not int or self.conditioning_samples < 2
                or self.scoring_law not in {"whole_observed_current_q", "fixed_current_prior_conditional_tail"}
                or self.target_prior != "independent_uniform_targets_conditioned_on_actual_C_labels"
                or self.source_exclusion_scope not in {"target_participant", "whole_target_dataset"}
                or self.conditioning_anchor != "first_current_event_native_clock"):
            raise ValueError("typed joint sampler, identified observed prefix and complete public target law are required")

    def record(self):
        return {"schema": "n2p3_joint_flat_candidate_programme/1", "sampler": self.sampler.record(),
            "conditioning_samples": self.conditioning_samples, "scoring_law": self.scoring_law,
            "target_prior": self.target_prior, "physical_drift_chart": "per_true_record_intercept_V_slope_V_per_second",
            "source_exclusion_scope": self.source_exclusion_scope, "conditioning_anchor": self.conditioning_anchor,
            "source_parameters": "original_cached_checkpoint_proper_response_and_fixed_noise_no_refit",
            "calibration_OOF": "original_earlier_prefix_prequential_programme"}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        if (values.pop("schema") != "n2p3_joint_flat_candidate_programme/1"
                or values.pop("physical_drift_chart") != "per_true_record_intercept_V_slope_V_per_second"
                or values.pop("source_parameters") != "original_cached_checkpoint_proper_response_and_fixed_noise_no_refit"
                or values.pop("calibration_OOF") != "original_earlier_prefix_prequential_programme"):
            raise ValueError("saved joint-history expert changed its source, flat measure or OOF semantics")
        values["sampler"] = JointFlatSamplingProgram.from_record(values["sampler"])
        return cls(**values)


def _noise_from_original_record(record):
    """Use the same original stored-state owner constructor as its JSON codec."""
    if record["schema"] != "n2p3_channel_noise_mechanism/3":
        raise ValueError("source noise requires its original compiled-state schema")
    actual = tuple(TemporalNoise.from_record(value) for value in record["compiled_temporal_records"])
    channels = len(record["geometry"]["channels"])
    temporal = actual*channels if record["family"] == "shared" else actual
    noise = FittedChannelNoise(temporal, record["innovation_covariance"], NoiseGeometry(**record["geometry"]),
        record["conditioning_samples"], record["family"], record["provenance"], record["compiled_prefix_root"])
    if metadata_digest(noise.record()) != metadata_digest(record):
        raise ValueError("source noise constructor differs from its original saved numerical producer")
    return noise


@dataclass(frozen=True, init=False)
class FrozenJointFlatSource:
    """Original source checkpoint and actual admitted cohort/parameter owner."""

    content: bytes = field(repr=False)
    parameters: JointFlatParameters = field(repr=False)
    original_model_content: bytes = field(repr=False)
    temporal_covariance_factor: np.ndarray = field(repr=False)
    spatial_covariance_factor: np.ndarray = field(repr=False)
    temporal_factor_errors: np.ndarray = field(repr=False)
    canonical_factor_binding: dict = field(repr=False)

    @cached_property
    def consumer_prior_chart_certificate(self):
        from research.joint_flat_chart_precision import saved_source_chart_certificate

        return freeze_metadata(saved_source_chart_certificate(self.temporal_covariance_factor,
            self.spatial_covariance_factor, self.parameters.prior.root))

    @property
    def digest(self):
        return hashlib.sha256(self.content).hexdigest()

    @classmethod
    def from_path(cls, path):
        payload = Path(path).read_bytes()
        return cls.from_bytes(payload, expected_sha256=hashlib.sha256(payload).hexdigest())

    @classmethod
    def from_parameters(cls, parameters, *, original_model_content):
        if (not isinstance(parameters, JointFlatParameters) or not isinstance(original_model_content, bytes)
                or parameters.binding["program"] != "actual_source_response_and_noise"
                or hashlib.sha256(original_model_content).hexdigest() != parameters.binding["source_model_sha256"]):
            raise ValueError("joint source export must retain its actual original source checkpoint bytes")
        model = ContinuousDecisionModel.load(io.BytesIO(original_model_content))
        if model.provenance["fit_scope"] != "source_training" or model.digest() != parameters.binding["source_model_digest"]:
            raise ValueError("source response and noise must come from the original actual source producer")
        if model.posterior_state is None:
            raise ValueError("joint source requires its original posterior precision factor")
        precision = model.posterior_state["precision_root"]
        temporal_factor = solve_triangular(precision, np.eye(len(precision)), lower=False)
        spatial_factor = np.linalg.cholesky(model.innovation_covariance)
        state = {"schema": "n2p3_frozen_joint_flat_source/1", "source_binding": parameters.binding,
            "source_binding_sha256": parameters.digest, "prior_mean": parameters.prior.mean.tolist(),
            "prior_root": parameters.prior.root.tolist(), "prior_program": parameters.prior.program,
            "basis": parameters.basis.tolist(), "noise": parameters.noise.record(),
            "canonical_source_temporal_factor": temporal_factor.tolist(),
            "canonical_source_spatial_factor": spatial_factor.tolist(),
            "original_model_base64": base64.b64encode(original_model_content).decode("ascii")}
        payload = json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf8")
        return cls.from_bytes(payload, expected_sha256=hashlib.sha256(payload).hexdigest())

    @classmethod
    def from_bytes(cls, payload, *, expected_sha256):
        from research.joint_flat_authority import _parameters
        from research.joint_response_scale import FixedHyperparameters

        require_sha256(expected_sha256, "actual joint source checkpoint")
        if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("joint source checkpoint differs from original declared bytes")
        state = json.loads(payload)
        if (state["schema"] != "n2p3_frozen_joint_flat_source/1"
                or metadata_digest(state["source_binding"]) != state["source_binding_sha256"]
                or state["source_binding"]["program"] != "actual_source_response_and_noise"):
            raise ValueError("joint source checkpoint lacks its actual source parameter/cohort binding")
        original = base64.b64decode(state["original_model_base64"], validate=True)
        binding = state["source_binding"]
        if hashlib.sha256(original).hexdigest() != binding["source_model_sha256"]:
            raise ValueError("joint source changed the original response/noise producer bytes")
        model = ContinuousDecisionModel.load(io.BytesIO(original))
        prior = GlobalGaussianPrior(state["prior_mean"], state["prior_root"], state["prior_program"])
        noise = _noise_from_original_record(state["noise"])
        basis = frozen_array(state["basis"])
        if (model.provenance["fit_scope"] != "source_training" or model.digest() != binding["source_model_digest"]
                or not np.array_equal(basis, model.basis)
                or array_digest(prior.mean, prior.root) != binding["beta_prior_sha256"]
                or array_digest(basis) != binding["basis_sha256"]
                or metadata_digest(noise.record()) != metadata_digest(binding["noise"])):
            raise ValueError("joint source numerical arrays differ from actual source model/parameter authority")
        if (model.posterior_state is None or not np.array_equal(prior.mean, model.kernel_mean.ravel())
                or not np.array_equal(noise.innovation_covariance, model.innovation_covariance)
                or any(metadata_digest(item.record()) != metadata_digest(model.noise.record()) for item in noise.temporal)
                or noise.provenance.get("source_model_sha256") != binding["source_model_sha256"]):
            raise ValueError("joint source mean/noise changed its original fitted source law")
        source_precision = model.posterior_state["precision_root"]
        temporal_factor = np.asarray(state["canonical_source_temporal_factor"], dtype=float)
        spatial_factor = np.asarray(state["canonical_source_spatial_factor"], dtype=float)
        if (temporal_factor.shape != source_precision.shape or spatial_factor.shape != model.innovation_covariance.shape
                or not np.isfinite(temporal_factor).all() or not np.isfinite(spatial_factor).all()
                or np.any(np.tril(temporal_factor, -1) != 0) or np.any(np.triu(spatial_factor, 1) != 0)
                or np.any(np.diag(temporal_factor) == 0) or np.any(np.diag(spatial_factor) <= 0)):
            raise ValueError("saved source factors changed the original temporal/spatial chart")
        eps = np.finfo(float).eps
        gamma = len(source_precision)*eps/(1-len(source_precision)*eps)
        inverse_defect = float(np.linalg.norm(np.eye(len(source_precision))-source_precision@temporal_factor, ord=2))
        inverse_defect += gamma*float(np.linalg.norm(np.abs(source_precision)@np.abs(temporal_factor), ord=2))
        if not np.isfinite(inverse_defect) or inverse_defect >= np.sqrt(eps):
            raise ValueError("saved source temporal factor is unresolved against the original precision root")
        factor_error = inverse_defect*float(np.linalg.norm(temporal_factor, ord=2))/(1-inverse_defect)
        spatial_inverse = solve_triangular(spatial_factor, np.eye(len(spatial_factor)), lower=True)
        relative_sigma = spatial_inverse@model.innovation_covariance@spatial_inverse.T
        spatial_defect = float(np.linalg.norm(relative_sigma-np.eye(len(spatial_factor)), ord=2))
        spatial_tolerance = 256*eps*len(spatial_factor)*max(1., np.linalg.cond(
            spatial_factor/np.max(np.abs(spatial_factor), axis=1)[:, None]))
        if not np.isfinite(spatial_tolerance) or spatial_tolerance > np.sqrt(eps) or spatial_defect > spatial_tolerance:
            raise ValueError("saved source spatial factor differs from its original covariance in channel units")
        factor = np.kron(temporal_factor, spatial_factor)
        _response_resolved_root(factor, "original source coefficient covariance factor")
        row_scales = np.max(np.abs(factor), axis=1)
        equilibrated = factor/row_scales[:, None]
        scaled_prior = prior.root/row_scales[:, None]
        relative_chart = np.linalg.solve(equilibrated, scaled_prior)
        defect = np.linalg.norm(relative_chart@relative_chart.T-np.eye(len(prior.mean)), ord=2)
        gamma_chart = len(prior.mean)*eps/(1-len(prior.mean)*eps)
        inverse_chart = np.linalg.solve(equilibrated, np.eye(len(prior.mean)))
        inverse_chart_defect = float(np.linalg.norm(np.eye(len(prior.mean))-equilibrated@inverse_chart, ord=2))
        inverse_chart_defect += gamma_chart*float(np.linalg.norm(np.abs(equilibrated)@np.abs(inverse_chart), ord=2))
        if not np.isfinite(inverse_chart_defect) or inverse_chart_defect >= 1:
            raise ValueError("source unit-equilibrated factor inverse is unresolved")
        chart_error = float(np.linalg.norm(scaled_prior-equilibrated@relative_chart, ord=2))
        chart_error += gamma_chart*float(np.linalg.norm(np.abs(equilibrated)@np.abs(relative_chart), ord=2))
        chart_error += eps/(1-eps)*float(np.linalg.norm(
            np.abs(scaled_prior)+np.abs(equilibrated)@np.abs(relative_chart), ord=2))
        chart_error *= float(np.linalg.norm(inverse_chart, ord=2))/(1-inverse_chart_defect)
        defect += gamma_chart*float(np.linalg.norm(np.abs(relative_chart)@np.abs(relative_chart.T), ord=2))
        defect += 2*float(np.linalg.norm(relative_chart, ord=2))*chart_error+chart_error*chart_error
        tolerance = 256*np.finfo(float).eps*len(prior.mean)*max(1., np.linalg.cond(equilibrated))
        if not np.isfinite(tolerance) or tolerance > np.sqrt(np.finfo(float).eps):
            raise ValueError("original source coefficient chart is unresolved after physical unit equilibration")
        if not np.isfinite(defect) or defect > tolerance:
            raise ValueError("joint source prior covariance differs from the original saved source factor")
        hyper = FixedHyperparameters(float(binding["degrees_of_freedom"]), binding["relative_noise_shape"], "externally_fixed")
        result = object.__new__(cls)
        object.__setattr__(result, "content", payload)
        object.__setattr__(result, "original_model_content", original)
        object.__setattr__(result, "temporal_covariance_factor", frozen_array(temporal_factor))
        object.__setattr__(result, "spatial_covariance_factor", frozen_array(spatial_factor))
        object.__setattr__(result, "temporal_factor_errors", frozen_array(np.full_like(temporal_factor, factor_error)))
        object.__setattr__(result, "canonical_factor_binding", freeze_metadata({
            "original_source_precision_root_sha256": array_digest(source_precision),
            "original_source_model_sha256": binding["source_model_sha256"],
            "temporal_factor_sha256": array_digest(temporal_factor),
            "spatial_factor_sha256": array_digest(spatial_factor),
            "saved_prior_chart_relation_error": float(defect),
            "saved_prior_chart_solve_error_estimate": chart_error,
            "temporal_inverse_defect_estimate": inverse_defect,
            "temporal_inverse_error_estimate": factor_error,
            "spatial_covariance_chart_relation_error": spatial_defect,
            "original_prior_root_preserved": True, "source_noise_refits": 0}))
        object.__setattr__(result, "parameters", _parameters(prior, hyper, basis, noise, freeze_metadata(binding)))
        return result

    def assert_target_exclusion(self, target, *, scope):
        """Check the actual cohort before an explicitly transported observation.

        The target contract comes from its original input owner; physical
        geometry is checked by the native or transported caller separately.
        """
        if scope not in {"target_participant", "whole_target_dataset"}:
            raise ValueError("joint source requires an explicit target exclusion scope")
        participant = target["participant_key"]
        rows = self.parameters.binding["source_prefixes"]
        target_sources = {target["source_sha256"]} if target.get("source_sha256") else set()
        target_sources.update(scope["source_sha256"] for scope in target.get("record_scopes", {}).values())
        target_sources.update(row["source_sha256"] for row in target.get("source_record_roster", {}).values())
        source_inputs = {row["target_contract"]["source_sha256"] for row in rows}
        source_inputs.update(scope["source_sha256"] for row in rows
                             for scope in row["target_contract"].get("record_scopes", {}).values())
        if (participant in {row["target_contract"]["participant_key"] for row in rows}
                or target_sources & source_inputs):
            raise ValueError("joint source cohort consumed the target participant or original input bytes")
        if scope == "whole_target_dataset" and any(
                row["target_contract"]["source_dataset_id"] == target["source_dataset_id"] for row in rows):
            raise ValueError("joint source whole-study exclusion includes the actual target study")

    def assert_target(self, target, *, exclusion_scope="target_participant"):
        self.assert_target_exclusion(target, scope=exclusion_scope)
        geometry = self.parameters.noise.geometry
        if (tuple(target["channel_names"]) != geometry.channels or target["sample_rate_hz"] != geometry.sfreq
                or target["reference"] != geometry.reference or target["signal_unit"] != geometry.unit):
            raise ValueError("joint source and target require an actual matching or transported physical observation")
