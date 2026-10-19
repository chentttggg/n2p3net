"""Source-only stationary moment projection to one shared AR1 reference.

The reference matches the equally weighted source zero-lag covariance. Its
scalar AR1 coefficient is the metric-optimal projection of the averaged lag1
covariance. It initializes target noise and defines Kref; it is no new noise
fit or claim that population recordings share this reference noise law.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.execution import UnidentifiableFitError
from research.stationary_noise_artifact import _CONTEXT, _cholesky, _law_covariance
from transfer.dictionary_artifact import _binary64
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest
from transfer.temporal_noise import TemporalNoise

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceStationaryMoments:
    participant_key: str
    geometry: NoiseGeometry
    zero_root: np.ndarray
    lag_factor: np.ndarray
    dependency: dict

    def __post_init__(self):
        zero, lag = np.asarray(self.zero_root, dtype=float), np.asarray(self.lag_factor, dtype=float)
        size = len(self.geometry.channels)
        if (zero.shape != (size, size) or lag.shape != zero.shape or not np.isfinite(zero).all()
                or not np.isfinite(lag).all() or np.linalg.slogdet(zero)[0] == 0 or not self.participant_key):
            raise ValueError("complete finite source zero/lag moment factors are required")
        object.__setattr__(self, "zero_root", frozen_array(zero))
        object.__setattr__(self, "lag_factor", frozen_array(lag))
        object.__setattr__(self, "dependency", freeze_metadata(self.dependency))

    @property
    def identity(self):
        return metadata_digest({"participant": self.participant_key, "geometry": self.geometry.record(),
            "factors": array_digest(self.zero_root, self.lag_factor), "dependency": self.dependency})

    def record(self):
        return {"schema": "source_stationary_zero_lag1_moment_factors/1", "participant_key": self.participant_key,
            "geometry": self.geometry.record(), "zero_root": self.zero_root.tolist(),
            "lag_factor": self.lag_factor.tolist(), "dependency": self.dependency, "identity": self.identity,
            "S0": "zero_root @ zero_root.T", "S1": "lag_factor @ zero_root.T",
            "raw_or_label_authority_granted": False}

    @classmethod
    def from_record(cls, record):
        if (record["schema"] != "source_stationary_zero_lag1_moment_factors/1"
                or record["raw_or_label_authority_granted"] is not False):
            raise ValueError("the source moment role differs from its saved numeric-only relation")
        value = cls(record["participant_key"], NoiseGeometry(**record["geometry"]), record["zero_root"],
            record["lag_factor"], record["dependency"])
        if value.identity != record["identity"]:
            raise ValueError("the original source moment factors changed")
        return value


def _parameter_moments(reflections, covariance):
    values = np.asarray(reflections, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("the declared reference projection consumes actual stationary AR1 source laws")
    covariance = np.asarray(covariance, dtype=float)
    with localcontext(_CONTEXT):
        zero = _cholesky(_law_covariance(values, covariance, 1))
    lag = np.empty_like(zero)
    for channel, row in enumerate(zero):
        coefficient = -Fraction.from_float(float(values[channel, 0]))
        for j, value in enumerate(row):
            product = coefficient*Fraction.from_float(float(value))
            lag[channel, j] = _binary64(product.numerator, product.denominator)
    return zero, lag


def moments_from_fitted_noise(noise, *, participant_key, dependency):
    """Use original stored PARCOR/full Q; never form a TC-by-TC covariance."""
    if type(noise) is not FittedChannelNoise or noise.order != 1:
        raise TypeError("the actual complete shared/private AR1 source noise owner is required")
    noise.assert_integrity()
    zero, lag = _parameter_moments([item.reflection_coefficients for item in noise.temporal], noise.innovation_covariance)
    return SourceStationaryMoments(participant_key, noise.geometry, zero, lag,
        {**dependency, "original_noise_record_sha256": metadata_digest(noise.record()),
         "moment_producer_sha256": IMPLEMENTATION_SHA256,
         "moment_program": "exact_original_PARCOR_pairwise_YW_small_Decimal96_root_and_exact_AR1_row_product"})


def moments_from_original_spatial_model(handle, *, likelihood_reference):
    """Extract once per same-fit source person, for all later fold references."""
    model = handle.model
    baseline, layout = model.baseline, model.posterior.prior.layout
    if baseline.noise.order != 1 or baseline.noise.startup_mode == "conditional_prefix":
        raise ValueError("the fixed source moment programme requires a stationary AR1 source")
    people = {row["authorization"]["original_view"]["participant_key"] for row in model.posterior.fit_records}
    if len(people) != 1:
        raise ValueError("one source moment package retains exactly one original participant")
    return moments_from_source_baseline(baseline, layout=layout, participant_key=next(iter(people)), dependency=
        {"original_source_file_sha256": handle.source_file_sha256,
         "original_source_model_sha256": model.digest(), "source_likelihood": likelihood_reference.record(),
        })


def moments_from_source_baseline(baseline, *, layout, participant_key, dependency):
    """Numerical moments from the original noise owner, without a posterior.

    Source publication authority belongs to the calling same-fit issuer;
    this transformation itself grants no data or live-fit permission.
    """
    from transfer.continuous_decision import ContinuousDecisionModel

    if (type(baseline) is not ContinuousDecisionModel or baseline.noise is None
            or baseline.noise.order != 1 or baseline.noise.startup_mode == "conditional_prefix"):
        raise TypeError("original complete stationary AR1 baseline parameters are required")
    baseline.assert_integrity()
    if tuple(baseline.channels) != layout.channels:
        raise ValueError("source moment layout differs from its actual baseline channels")
    geometry = NoiseGeometry(tuple(baseline.channels), baseline.sfreq, layout.signal_unit,
        layout.acquisition_reference, baseline.operator_identity, len(baseline.channels))
    reflections = [baseline.noise.reflection_coefficients]*len(baseline.channels)
    zero, lag = _parameter_moments(reflections, baseline.innovation_covariance)
    return SourceStationaryMoments(participant_key, geometry, zero, lag,
        {**dependency, "original_temporal_noise_sha256": baseline.noise.record()["artifact_sha256"],
         "original_full_Q_sha256": array_digest(baseline.innovation_covariance),
         "moment_producer_sha256": IMPLEMENTATION_SHA256,
         "moment_program": "exact_original_shared_PARCOR_pairwise_YW_small_Decimal96_root_and_exact_AR1_row_product"})


def _average_covariance(factors, scalar):
    count, size = len(factors), len(factors[0])
    exact = [[[Fraction.from_float(float(v)) for v in row] for row in factor] for factor in factors]
    output = np.empty((size, size))
    for i in range(size):
        for j in range(i+1):
            covariance = sum((sum((matrix[i][k]*matrix[j][k] for k in range(size)), Fraction())
                              for matrix in exact), Fraction())/count
            value = scalar*covariance
            output[i, j] = output[j, i] = _binary64(value.numerator, value.denominator)
    return output


def build_fold_noise_reference(moments, *, fold, conditioning_samples):
    """a=tr(S0bar^-1 S1bar)/C; Qref=(1-a^2) S0bar, source-only.

    QR/triangular solves act on small channel factors and preserve weak units.
    No a clipping, covariance jitter or target-derived scale enters this law.
    """
    moments = tuple(moments)
    by_person = {value.participant_key: value for value in moments}
    if len(by_person) != len(moments) or set(by_person) != set(fold.source_participant_keys):
        raise ValueError("the reference must retain exactly the fold's source participants")
    values = tuple(by_person[key] for key in fold.source_participant_keys)
    geometry = values[0].geometry
    if any(value.geometry.record() != geometry.record() for value in values):
        raise ValueError("source moment references require the same declared native physical coordinates")
    if set(by_person) & set(fold.target_identity_keys):
        raise ValueError("the target participant entered the source-only reference")
    count, channels = len(values), len(geometry.channels)
    divisor = np.sqrt(float(count))
    stack = np.concatenate([value.zero_root for value in values], axis=1)/divisor
    upper = np.linalg.qr(stack.T, mode="r")
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    lower = (signs[:, None]*upper).T
    if lower.shape != (channels, channels) or np.any(np.diag(lower) <= 0):
        raise UnidentifiableFitError("the complete source zero-lag reference root is unresolved")
    terms = []
    for value in values:
        zero = solve_triangular(lower, value.zero_root/divisor, lower=True)
        lag = solve_triangular(lower, value.lag_factor/divisor, lower=True)
        terms.extend(float(a)*float(b) for a, b in zip(zero.flat, lag.flat, strict=True))
    coefficient = math.fsum(terms)/channels
    if not np.isfinite(coefficient) or not -1 < coefficient < 1:
        raise UnidentifiableFitError("the metric-optimal shared AR1 reference coefficient is unresolved")
    exact_coefficient = Fraction.from_float(coefficient)
    covariance = _average_covariance([value.zero_root for value in values], 1-exact_coefficient**2)
    diagnostic_variance = math.fsum(float(covariance[i, i])/channels for i in range(channels))
    temporal = TemporalNoise.from_reflections([-coefficient], innovation_variance=diagnostic_variance,
        reference_id="source_only_equal_person_stationary_moment_projection")
    binding = {"schema": "equal_person_stationary_moment_shared_AR1_reference/1", "origin": "fold_source_only",
        "source_fold": fold.record(), "source_moment_identities": [value.identity for value in values],
        "source_dependencies": [value.dependency for value in values], "equal_person_weights": [1/count]*count,
        "coefficient_formula": "trace(S0bar_inverse_S1bar)/observed_channels",
        "covariance_formula": "(1-a^2)*S0bar", "shared_AR1_coefficient": coefficient,
        "source_only_reference_not_population_true_noise": True, "new_noise_fits": 0,
        "target_data_used": False, "source_posterior_factor_used": False,
        "coefficient_clipped": False, "covariance_jitter_added": False,
        "producer_sha256": IMPLEMENTATION_SHA256}
    return FittedChannelNoise.from_markov_stationary_factors((temporal,)*channels, covariance,
        geometry=geometry, conditioning_samples=conditioning_samples, family="shared",
        provenance={"fit_scope": "analytic_reference", "reference_id": "fold_source_stationary_moment_projection",
                    "records": [], "source_reference_binding": binding})


def publish_fold_noise_reference(moments, *, fold, conditioning_samples, output_path):
    """Save once, then bind Kref to the original saved-noise consumer identity."""
    value = build_fold_noise_reference(moments, fold=fold, conditioning_samples=conditioning_samples)
    value.save(output_path)
    content = Path(output_path).read_bytes()
    reference = {"path": str(Path(output_path).resolve()), "bytes": len(content),
                 "sha256": hashlib.sha256(content).hexdigest()}
    restored = FittedChannelNoise.load(output_path)
    if metadata_digest(json.loads(content)["state"]) != metadata_digest(restored.record()):
        raise ValueError("published source reference differs from its original saved noise law")
    return restored, reference
