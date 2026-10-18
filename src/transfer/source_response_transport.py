"""Typed B09 transport of frozen response priors into a proper target chart.

This module consumes the original source and geometry owners;
it estimates no parameter and grants no observation or label permission.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.channel_noise_mechanism import NoiseGeometry
from research.conditional_response_graph import (
    ExactConditionalResponseGraph,
)
from research.conditional_response_graph import (
    conditional as graph_conditional,
)
from research.conditional_response_graph import (
    kronecker as graph_kronecker,
)
from research.conditional_response_graph import (
    product as graph_product,
)
from research.conditional_response_graph import (
    proper as graph_proper,
)
from research.conditional_response_graph import (
    rows as graph_rows,
)
from research.conditional_response_graph import (
    transpose as graph_transpose,
)
from transfer.joint_flat_programme import FrozenJointFlatSource
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
)
from transfer.spatial_signal_prior import (
    IMPLEMENTATION_SHA256 as B09_IMPLEMENTATION_SHA256,
)
from transfer.spatial_signal_prior import (
    ERPPhysicalTransport,
    SpatialPriorIdentificationError,
    SpatialSignalPrior,
    _ExactRowSpace,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TransportedResponsePrior = ExactConditionalResponseGraph

def _exact_rows(values):
    return tuple(tuple(value if isinstance(value, Fraction) else Fraction.from_float(float(value))
                       for value in row) for row in values)


def _exact_image(mapping, values, *, rounding_enclosure=False):
    """Correctly round each exact rational image of saved binary64 operands."""
    rows = _exact_rows(mapping)
    values = np.asarray(values, dtype=float)
    vector = values.ndim == 1
    matrix = values[:, None] if vector else values
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or any(len(row) != len(matrix) for row in rows):
        raise ValueError("exact image requires finite operands in the declared coordinate frame")
    result = np.empty((len(rows), matrix.shape[1]))
    error_bound = np.zeros_like(result)
    exact_values = [[Fraction.from_float(float(value)) for value in row] for row in matrix]
    for index, row in enumerate(rows):
        support = [(j, value) for j, value in enumerate(row) if value]
        for column in range(matrix.shape[1]):
            exact = sum((weight * exact_values[j][column] for j, weight in support), Fraction())
            try:
                rounded = float(exact)
            except OverflowError as error:
                raise SpatialPriorIdentificationError("transport_image_not_representable") from error
            if not np.isfinite(rounded) or (exact and rounded == 0):
                raise SpatialPriorIdentificationError("transport_image_not_representable")
            result[index, column] = rounded
            if rounding_enclosure:
                error = abs(Fraction.from_float(rounded) - exact)
                upper = float(error)
                if Fraction.from_float(upper) < error:
                    upper = np.nextafter(upper, np.inf)
                error_bound[index, column] = upper
    result = result[:, 0] if vector else result
    return (result, error_bound[:, 0] if vector else error_bound) if rounding_enclosure else result


def _exact_inverse(matrix):
    width = len(matrix)
    rows = [list(row) + [Fraction(i == j) for j in range(width)] for i, row in enumerate(matrix)]
    determinant = Fraction(1)
    for column in range(width):
        pivot = next((i for i in range(column, width) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("common/completion chart is singular")
        if pivot != column:
            rows[pivot], rows[column] = rows[column], rows[pivot]
            determinant = -determinant
        scale = rows[column][column]
        determinant *= scale
        rows[column] = [value / scale for value in rows[column]]
        for i in range(width):
            if i != column and rows[i][column]:
                weight = rows[i][column]
                rows[i] = [a - weight * b for a, b in zip(rows[i], rows[column], strict=True)]
    return tuple(tuple(row[width:]) for row in rows), determinant


def _coefficient_map(spatial, coefficients):
    return tuple(tuple(value if block == coefficient else Fraction()
                       for block in range(coefficients) for value in row)
                 for coefficient in range(coefficients) for row in spatial)


@dataclass(frozen=True)
class ResponseCoordinateChart:
    """Exact reversible common-first spatial coordinates, applied per coefficient."""

    common_map: tuple[tuple[Fraction, ...], ...]
    exact_matrix: tuple = field(init=False, repr=False)
    exact_inverse: tuple = field(init=False, repr=False)
    determinant: Fraction = field(init=False)
    diagnostic_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    diagnostic_inverse: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        common = _exact_rows(self.common_map)
        if not common or not common[0] or any(len(row) != len(common[0]) for row in common):
            raise ValueError("chart requires a nonempty rectangular common spatial map")
        width = len(common[0])
        space = _ExactRowSpace(width)
        if not all(space.add(row) for row in common):
            raise ValueError("declared common spatial rows must be exactly independent")
        free = [j for j in range(width) if j not in space.pivots]
        exact = common + tuple(tuple(Fraction(i == j) for j in range(width)) for i in free)
        inverse, determinant = _exact_inverse(exact)
        object.__setattr__(self, "common_map", common)
        object.__setattr__(self, "exact_matrix", exact)
        object.__setattr__(self, "exact_inverse", inverse)
        object.__setattr__(self, "determinant", determinant)
        object.__setattr__(self, "diagnostic_matrix", frozen_array([[float(v) for v in row] for row in exact]))
        object.__setattr__(self, "diagnostic_inverse", frozen_array([[float(v) for v in row] for row in inverse]))

    @property
    def common_dimension(self):
        return len(self.common_map)

    @property
    def native_dimension(self):
        return len(self.exact_matrix)

    def record(self):
        body = {"schema": "n2p3_reversible_response_spatial_chart/1",
                "common_map": [[str(v) for v in row] for row in self.common_map],
                "matrix": [[str(v) for v in row] for row in self.exact_matrix],
                "inverse": [[str(v) for v in row] for row in self.exact_inverse],
                "determinant": str(self.determinant),
                "completion_rule": "unit_selectors_at_exact_common_RREF_free_native_columns"}
        return {**body, "artifact_sha256": metadata_digest(body)}

    @classmethod
    def from_record(cls, record):
        chart = cls(tuple(tuple(Fraction(v) for v in row) for row in record["common_map"]))
        if chart.record() != record:
            raise ValueError("saved chart differs from its exact common/completion relation")
        return chart

    def exact_observations_to_chart(self, native_observations):
        values = graph_rows(native_observations)
        if not values or any(len(row) != self.native_dimension for row in values):
            raise ValueError("native observations must be sample by declared native EEG")
        return graph_transpose(graph_product(self.exact_matrix, graph_transpose(values)))

    def exact_spatial_noise_factor_to_chart(self, native_factor):
        values = graph_rows(native_factor)
        if len(values) != self.native_dimension:
            raise ValueError("noise factor needs the same complete native spatial rows")
        return graph_product(self.exact_matrix, values)

    def exact_native_observation_map_to_chart(self, native_map):
        values = graph_rows(native_map)
        if not values or any(len(row) != self.native_dimension for row in values):
            raise ValueError("observation map differs from the native coefficient chart")
        return graph_product(values, self.exact_inverse)

    def diagnostic_observations_to_chart(self, native_observations):
        values = finite_real_array(native_observations, name="native target observations")
        if values.ndim != 2 or values.shape[1] != self.native_dimension:
            raise ValueError("native observations must be sample by declared native EEG")
        return _exact_image(self.exact_matrix, values.T).T

    def diagnostic_spatial_noise_factor_to_chart(self, native_factor):
        factor = finite_real_array(native_factor, name="native spatial noise factor")
        if factor.ndim != 2 or factor.shape[0] != self.native_dimension:
            raise ValueError("noise factor needs the same complete native spatial rows")
        return _exact_image(self.exact_matrix, factor)

    def diagnostic_native_observation_map_to_chart(self, native_map):
        mapping = finite_real_array(native_map, name="native observation map")
        if mapping.ndim != 2 or mapping.shape[1] != self.native_dimension:
            raise ValueError("observation map differs from the native coefficient chart")
        return _exact_image(tuple(zip(*self.exact_inverse, strict=True)), mapping.T).T

    def native_log_density_adjustment(self, *, observation_rows, transformed_affine_drift_groups=0):
        if (type(observation_rows) is not int or observation_rows < 0
                or type(transformed_affine_drift_groups) is not int or transformed_affine_drift_groups < 0):
            raise ValueError("volume counts require actual observations and distinct affine drift groups")
        # Each group's intercept and slope are both spatially transformed.
        # Shared groups across segments are counted once by the caller.
        determinant = abs(self.determinant)
        logdet = math.log(determinant.numerator) - math.log(determinant.denominator)
        return (observation_rows - 2 * transformed_affine_drift_groups) * logdet


def _complete_gaussian(source_mean, source_factor, source_map, target_mean, target_factor, target_map):
    """Independent general-covariance mathematical entry; returns an exact graph."""
    chart = ResponseCoordinateChart(_exact_rows(target_map))
    reference = graph_product(graph_rows(target_factor), graph_transpose(graph_rows(target_factor)))
    reference_chart = graph_product(graph_product(chart.exact_matrix, reference), graph_transpose(chart.exact_matrix))
    gain, covariance = graph_conditional(reference_chart, chart.common_dimension)
    mean = graph_product(chart.exact_matrix, tuple((Fraction.from_float(float(v)),) for v in target_mean))
    return TransportedResponsePrior(source_mean, source_factor, _exact_rows(source_map),
        tuple(row[0] for row in mean), gain, covariance, chart, 1,
        {"role": "independent_mathematical_parameters_no_raw_or_label_authority"})


def _source_spatial_relation(source, transport):
    if not isinstance(source, FrozenJointFlatSource) or not isinstance(transport, ERPPhysicalTransport):
        raise TypeError("transport requires the original FrozenJointFlatSource and B09 ERPPhysicalTransport")
    geometry, layout = source.parameters.noise.geometry, transport.source_effect.layout
    if (layout.channels != geometry.channels or layout.acquisition_reference != geometry.reference
            or layout.signal_unit != geometry.unit):
        raise ValueError("B09 source layout differs from the actual original source geometry")
    return transport.local_effect.spatial_map, transport.destination_effect.spatial_map


def transport_response_prior(source, *, spatial_transport, completion_prior, continuation,
                             target_observation_geometry, target_participant_key):
    """Preserve the complete common prior and caller-owned residual law.

    No completion variance is implicit. Function anchors remain source-native;
    the target observation clock is a separate input to the eventual caller.
    """
    source_spatial, target_spatial = _source_spatial_relation(source, spatial_transport)
    if (not isinstance(completion_prior, SpatialSignalPrior)
            or not isinstance(continuation, NativeResponseDictionary)
            or not isinstance(target_observation_geometry, NoiseGeometry)):
        raise TypeError("a proper B09 completion prior, source continuation, and target geometry are required")
    source_geometry = source.parameters.noise.geometry
    target_layout = completion_prior.layout
    spatial_transport.destination_effect.validate_prior(completion_prior)
    dictionary_digest = continuation.digest()
    if (not np.array_equal(continuation.matrix, source.parameters.basis)
            or continuation.sample_rate_hz != source_geometry.sfreq
            or not np.array_equal(spatial_transport.source_effect.basis, continuation.matrix)
            or spatial_transport.source_effect.design_binding["dictionary_sha256"] != dictionary_digest
            or spatial_transport.source_effect.design_binding["sample_rate_hz"] != continuation.sample_rate_hz
            or target_observation_geometry.channels != target_layout.channels
            or target_observation_geometry.reference != target_layout.acquisition_reference
            or target_observation_geometry.unit != target_layout.signal_unit):
        raise ValueError("source function chart, B09 effect, completion, or target observation geometry changed")
    participants = tuple(row["target_contract"]["participant_key"] for row in source.parameters.binding["source_prefixes"])
    if not isinstance(target_participant_key, str) or not target_participant_key or target_participant_key in participants:
        raise ValueError("transport requires an independent explicitly declared target participant")
    p = 2 * source.parameters.basis.shape[1]
    if completion_prior.mean.shape != (p, len(target_layout.channels)):
        raise ValueError("completion coefficient chart differs from the source common/increment columns")
    exact_source_spatial = tuple(tuple(Fraction(value) for value in row)
        for row in spatial_transport.binding["local_structural_spatial_map"])
    chart = ResponseCoordinateChart(spatial_transport.destination_effect.structural_spatial_map)
    source_map = _coefficient_map(exact_source_spatial, p)
    target_map = _coefficient_map(chart.common_map, p)
    h, spatial = graph_rows(completion_prior.temporal_covariance), graph_rows(completion_prior.signal_covariance)
    graph_proper(h)
    graph_proper(spatial)
    spatial_chart_covariance = graph_product(graph_product(chart.exact_matrix, spatial),
                                            graph_transpose(chart.exact_matrix))
    spatial_gain, spatial_residual = graph_conditional(spatial_chart_covariance, chart.common_dimension)
    gain = _coefficient_map(spatial_gain, p)
    residual_covariance = graph_kronecker(h, spatial_residual)
    reference_chart_mean = graph_product(_coefficient_map(chart.exact_matrix, p),
        tuple((Fraction.from_float(float(value)),) for value in completion_prior.mean.ravel(order="C")))
    relation = {"method": "exact_named_source_map_and_separable_reference_conditional_graph",
        "canonical_reference_law": "exact_rational_saved_B09_H_tensor_S_covariance",
        "common_coordinates": len(target_map), "target_coordinates": p * chart.native_dimension,
        "source_latent_columns": source.parameters.prior.root.shape[1],
        "proper_conditional_dimensions": p * (chart.native_dimension - chart.common_dimension),
        "conditional_spatial_gain": [[str(v) for v in row] for row in spatial_gain],
        "conditional_spatial_covariance": [[str(v) for v in row] for row in spatial_residual],
        "properness": "exact_rational_positive_LDL_pivots_H_S_and_conditional_S",
        "common_map_or_conditional_source_factor_materialized": False,
        "spectral_truncation": False, "all_target_coordinates_retained": True}
    binding = {
        "schema": "n2p3_frozen_source_prior_transport_binding/3",
        "source_codec_sha256": source.digest,
        "source_model_sha256": source.parameters.binding["source_model_sha256"],
        "source_model_digest": source.parameters.binding["source_model_digest"],
        "source_prior_mean_factor_sha256": array_digest(source.parameters.prior.mean, source.parameters.prior.root),
        "source_geometry": source_geometry.record(),
        "original_excluded_target_participant": source.parameters.binding["excluded_target_participant"],
        "target_participant_key": target_participant_key,
        "source_participant_keys": list(participants),
        "spatial_transport": spatial_transport.record(),
        "completion_prior_artifact_sha256": completion_prior.record()["artifact_sha256"],
        "completion_prior_source": completion_prior.source,
        "conditional_law": "preserve_reference_theta_given_common_replace_complete_common_marginal",
        "function_dictionary_sha256": dictionary_digest,
        "function_source_clock_hz": continuation.sample_rate_hz,
        "function_continuation_role": continuation.metadata()["function_role"],
        "target_observation_geometry": target_observation_geometry.record(),
        "vectorization": "row_major_common_then_target_increment_by_reversible_spatial_chart",
        "coefficient_chart_sha256": chart.record()["artifact_sha256"],
        "native_target_EEG_dimensions_retained": len(target_layout.channels),
        "cross_participant_forecast": "declared_programme_source_common_posterior_as_target_common_prior",
        "between_participant_covariance_added": False,
        "target_native_reconstruction_role": "exact_named_linear_map_consumer_requests_actual_functionals",
        "source_fits": 0, "target_parameter_fits": 0,
        "prior_coordinate_QR_calls": 0,
        "sample_or_label_authority_granted": False,
        "source_response_and_target_noise_laws_separate": True,
        "exact_graph_generation": relation,
        "numeric_producer_sha256": IMPLEMENTATION_SHA256,
        "B09_owner_implementation_sha256": B09_IMPLEMENTATION_SHA256,
    }
    return TransportedResponsePrior(source.parameters.prior.mean, source.parameters.prior.root, source_map,
        tuple(row[0] for row in reference_chart_mean), gain, residual_covariance, chart, p, binding)


def _ratio(value):
    return [str(value.numerator), str(value.denominator)]


@dataclass(frozen=True)
class ExactSharedAR1Push:
    source_rho: Fraction
    target_rho: Fraction
    innovation_covariance: tuple[tuple[Fraction, ...], ...]
    binding: dict

    def __post_init__(self):
        if (not isinstance(self.source_rho, Fraction) or not isinstance(self.target_rho, Fraction)
                or not -1 < self.source_rho < 1 or self.target_rho != self.source_rho**4):
            raise ValueError("exact shared AR1 push requires the declared stable fourth-power transition")
        covariance = tuple(tuple(Fraction(v) for v in row) for row in self.innovation_covariance)
        width = len(covariance)
        if (not width or any(len(row) != width for row in covariance)
                or any(covariance[i][j] != covariance[j][i] for i in range(width) for j in range(width))):
            raise ValueError("exact shared AR1 innovation requires a symmetric complete common covariance")
        # Exact LDL signs establish properness without squaring tiny units.
        work = [list(row) for row in covariance]
        for k in range(width):
            pivot = work[k][k]
            if pivot <= 0:
                raise SpatialPriorIdentificationError("exact_common_noise_covariance_not_proper")
            for i in range(k + 1, width):
                for j in range(k + 1, width):
                    work[i][j] -= work[i][k] * work[k][j] / pivot
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "binding", freeze_metadata(self.binding))

    def record(self):
        return {"schema": "n2p3_shared_AR1_spatial_decimation_exact_real/1",
                "source_rho": _ratio(self.source_rho), "target_rho": _ratio(self.target_rho),
                "target_innovation_covariance": [[_ratio(value) for value in row] for row in self.innovation_covariance],
                "binding": self.binding,
                "role": "exact_real_source_law_push_not_a_rounded_FittedChannelNoise_or_target_fit"}

    @classmethod
    def from_record(cls, record):
        def fraction(pair):
            if len(pair) != 2 or any(not isinstance(v, str) for v in pair):
                raise ValueError("exact saved operands require integer numerator/denominator strings")
            return Fraction(int(pair[0]), int(pair[1]))
        if record.get("schema") != "n2p3_shared_AR1_spatial_decimation_exact_real/1":
            raise ValueError("unknown exact shared AR1 parameter schema")
        result = cls(fraction(record["source_rho"]), fraction(record["target_rho"]),
            tuple(tuple(fraction(v) for v in row) for row in record["target_innovation_covariance"]),
            record["binding"])
        if metadata_digest(result.record()) != metadata_digest(record):
            raise ValueError("exact common-noise parameter record changed")
        return result

    def validate_origin(self, source, **factory_arguments):
        if self.record() != push_shared_ar1_noise(source, **factory_arguments).record():
            raise ValueError("exact common noise differs from actual source/spatial/clock owners")
        return self

    def stationary_covariance_at_lag(self, lag):
        if type(lag) is not int:
            raise TypeError("target lag is an integer number of target observations")
        scale = self.target_rho**abs(lag) / (1 - self.target_rho**2)
        return tuple(tuple(scale * value for value in row) for row in self.innovation_covariance)


def _shared_ar1_rho(noise):
    if noise.family != "shared" or noise.order != 1:
        raise SpatialPriorIdentificationError("projected_private_or_general_AR_requires_explicit_state_space_owner")
    rho = Fraction.from_float(float(noise.temporal[0].coefficients[0]))
    if (not -1 < rho < 1 or any(Fraction.from_float(float(item.coefficients[0])) != rho for item in noise.temporal)):
        raise ValueError("source is not the same stable AR1 law on every actual source channel")
    return rho


def push_shared_ar1_noise(source, *, spatial_transport, target_clock_hz, decimation_phase_source_samples):
    """Exact-real named law for a genuine shared AR1 and four-fold selection.

    The output covers only the admitted common contrasts. General private AR
    requires the projected/state-space owner; no approximate family is fitted.
    """
    source_spatial, _target_spatial = _source_spatial_relation(source, spatial_transport)
    noise = source.parameters.noise
    rho = _shared_ar1_rho(noise)
    if Fraction.from_float(float(noise.geometry.sfreq)) != 4 * Fraction.from_float(float(target_clock_hz)):
        raise ValueError("this exact-real push requires a declared four-fold observation selection")
    if type(decimation_phase_source_samples) is not int or decimation_phase_source_samples not in range(4):
        raise ValueError("the actual four-fold selection phase must be explicit")
    exact_map = tuple(tuple(Fraction(value) for value in row)
                      for row in spatial_transport.binding["local_structural_spatial_map"])
    eta = Fraction.from_float(float(source.parameters.hyperparameters.relative_noise_shape))
    multiplier = sum((rho**(2 * index) for index in range(4)), Fraction())
    support = [tuple((index, value) for index, value in enumerate(row) if value) for row in exact_map]
    covariance = tuple(tuple(eta * multiplier * sum((a * b * Fraction.from_float(float(noise.innovation_covariance[i, j]))
        for i, a in left for j, b in right), Fraction()) for right in support) for left in support)
    binding = freeze_metadata({"source_codec_sha256": source.digest,
        "source_noise_state_sha256": noise._runtime_state_digest(), "source_geometry": noise.geometry.record(),
        "spatial_transport_sha256": spatial_transport.record()["artifact_sha256"],
        "common_spatial_map_sha256": array_digest(source_spatial),
        "exact_structural_common_map": [[str(value) for value in row] for row in exact_map],
        "target_clock_hz": float(target_clock_hz), "decimation_factor": 4,
        "decimation_phase_source_samples": decimation_phase_source_samples,
        "innovation_multiplier": _ratio(multiplier), "relative_noise_shape": _ratio(eta),
        "stationary_startup_law": "target_Gamma_k=rho4_abs_k_times_target_innovation_over_1_minus_rho8",
        "base_noise_law": "stationary_Gaussian_covariance_at_lambda_one",
        "Gamma_marginal_law": "not_defined_by_this_parameter_push",
        "observed_common_dimensions": len(source_spatial), "full_native_target_noise_completed": False,
        "target_parameter_fits": 0, "source_fits": 0,
        "numeric_producer_sha256": IMPLEMENTATION_SHA256,
        "arithmetic": "exact_rational_operations_on_stored_binary64_source_operands"})
    return ExactSharedAR1Push(rho, rho**4, covariance, binding)
