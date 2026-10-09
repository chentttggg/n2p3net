"""Frozen-response controls for shared spectra and diagonal-lag VAR noise.

This research module never substitutes a private spectrum into a matrix-normal
row posterior. It scores the general vector Gaussian model, conditions the
frozen ERP uncertainty and diffuse drift on the actual prefix, and keeps noise
fit, response fit and diagnostic labels separately attributable. Residuals can
contain unmodelled ERP; better innovation ACF is not a promotion criterion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_triangular
from scipy.signal import welch
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from research.execution import UnidentifiableFitError
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)
from transfer.temporal_noise import TemporalNoise

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_FIT_SCOPES = {"source_training", "target_calibration"}
_SCHEMA = "n2p3_channel_noise_mechanism/2"


def _array(value, name, ndim):
    result = finite_real_array(value, name=name)
    if result.ndim != ndim:
        raise ValueError(f"{name} requires {ndim} dimensions")
    return result


def _positive_covariance(value, name):
    covariance = _array(value, name, 2)
    if not covariance.shape[0] or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be nonempty and square")
    scale = max(float(np.linalg.norm(covariance)), np.finfo(float).tiny)
    if np.linalg.norm(covariance - covariance.T) > 1e-11 * scale:
        raise ValueError(f"{name} must be symmetric")
    covariance = (covariance + covariance.T) / 2
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise UnidentifiableFitError(f"{name} requires the full identified observed support") from error
    return covariance, root


def _validate_training_ledger(ledger):
    if not isinstance(ledger, (list, tuple)) or not ledger:
        raise ValueError("a fitted producer requires actual training recording support")
    require_identifiers([row["record_id"] for row in ledger], "producer training record IDs")
    for row in ledger:
        support = row.get("read_support")
        if (not isinstance(row.get("recording_id"), str) or not row["recording_id"].strip()
                or not isinstance(row.get("participant_key"), str) or not row["participant_key"].strip()
                or not isinstance(support, (list, tuple)) or len(support) != 2
                or any(type(value) is not int for value in support) or not 0 <= support[0] < support[1]):
            raise ValueError("producer training ledger needs exact physical recording intervals")


@dataclass(frozen=True)
class NoiseGeometry:
    """Explicit physical coordinates; rank-deficient references need a new basis."""

    channels: tuple[str, ...]
    sfreq: float
    unit: str
    reference: str
    operator_identity: str
    observation_rank: int

    def __post_init__(self):
        require_identifiers(self.channels, "observed noise channels")
        if (isinstance(self.sfreq, bool) or not np.isfinite(self.sfreq) or self.sfreq <= 0
                or self.unit not in {"V", "uV"}
                or not isinstance(self.reference, str) or not self.reference.strip()
                or not isinstance(self.operator_identity, str) or not self.operator_identity.strip()):
            raise ValueError("noise geometry requires physical units, rate, reference and operator")
        if type(self.observation_rank) is not int or self.observation_rank != len(self.channels):
            raise ValueError("supply identified full-rank coordinates; no CAR null-space inflation")
        object.__setattr__(self, "channels", tuple(self.channels))

    def record(self):
        return {"channels": list(self.channels), "sfreq": float(self.sfreq), "unit": self.unit,
                "reference": self.reference, "operator_identity": self.operator_identity,
                "observation_rank": self.observation_rank}


@dataclass(frozen=True)
class ResidualRecord:
    """Actual authorized residual and identities of every subtraction operand."""

    record_id: str
    participant_key: str
    recording_id: str
    sample_start: int
    residual: np.ndarray
    geometry: NoiseGeometry
    role: str
    provenance: dict

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip()
               for x in (self.record_id, self.participant_key, self.recording_id)):
            raise ValueError("residual records need explicit record, participant and recording identities")
        values = _array(self.residual, "actual residual", 2)
        if (type(self.sample_start) is not int or self.sample_start < 0
                or len(values) < 3 or values.shape[1] != len(self.geometry.channels)
                or self.role not in _FIT_SCOPES | {"held_diagnostic"}):
            raise ValueError("residual support, observed channels or fit/diagnostic role is invalid")
        required = {"samples_sha256", "design_sha256", "kernel_mean_sha256", "nuisance_sha256",
                    "source_model_sha256", "source_input_sha256", "residualization"}
        if not isinstance(self.provenance, dict) or set(self.provenance) != required:
            raise ValueError("residual requires the actual subtraction and source binding")
        for key in required - {"residualization"}:
            require_sha256(self.provenance[key], key)
        if self.provenance["residualization"] != "actual_Y_minus_D_B_minus_declared_nuisance":
            raise ValueError("unknown residualization; do not treat arbitrary arrays as ERP-free noise")
        object.__setattr__(self, "residual", frozen_array(values))
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))

    @classmethod
    def from_arrays(cls, *, samples, design, kernel_mean, nuisance, record_id, participant_key,
                    recording_id, sample_start, geometry, role, source_model_sha256,
                    source_input_sha256):
        y, d, b, z = (_array(value, name, 2) for value, name in
                      ((samples, "samples"), (design, "design"),
                       (kernel_mean, "kernel mean"), (nuisance, "nuisance mean")))
        if d.shape[0] != len(y) or d.shape[1] != b.shape[0] or b.shape[1] != y.shape[1] or z.shape != y.shape:
            raise ValueError("actual residual operands have inconsistent shapes")
        return cls(record_id, participant_key, recording_id, sample_start, y - d @ b - z,
                   geometry, role,
                   {"samples_sha256": array_digest(y), "design_sha256": array_digest(d),
                    "kernel_mean_sha256": array_digest(b), "nuisance_sha256": array_digest(z),
                    "source_model_sha256": source_model_sha256,
                    "source_input_sha256": source_input_sha256,
                    "residualization": "actual_Y_minus_D_B_minus_declared_nuisance"})

    def ledger(self):
        return {"record_id": self.record_id, "participant_key": self.participant_key,
                "recording_id": self.recording_id,
                "read_support": [self.sample_start, self.sample_start + len(self.residual)],
                "residual_sha256": array_digest(self.residual),
                "residual_channel_sha256": [array_digest(self.residual[:, [channel]])
                    for channel in range(self.residual.shape[1])], "role": self.role,
                "provenance": self.provenance}


@dataclass(frozen=True)
class FrozenERPResponse:
    """Physical vec(B) uncertainty in row-major coefficient/channel order.

    The factor may have zero columns for the explicitly known-mean control.
    Keeping this state fixed across noise arms isolates noise weighting. Its
    training source is external to this residual-only noise fitter.
    """

    mean: np.ndarray
    coefficient_factor: np.ndarray
    geometry: NoiseGeometry
    source_model_sha256: str
    response_id: str
    provenance: dict

    def __post_init__(self):
        mean = _array(self.mean, "frozen ERP mean", 2)
        factor = _array(self.coefficient_factor, "physical coefficient covariance factor", 2)
        require_sha256(self.source_model_sha256, "frozen ERP source model")
        if (mean.shape[0] < 1 or mean.shape[1] != len(self.geometry.channels)
                or factor.shape[0] != mean.size or factor.shape[1] > mean.size
                or not isinstance(self.response_id, str)
                or not self.response_id.strip() or not isinstance(self.provenance, dict)):
            raise ValueError("frozen ERP response has inconsistent physical coordinates or source")
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if provenance.get("fit_scope") not in _FIT_SCOPES | {"analytic_reference"}:
            raise ValueError("ERP response must come from source/C or an explicitly known oracle")
        if provenance["fit_scope"] in _FIT_SCOPES:
            _validate_training_ledger(provenance.get("records"))
        object.__setattr__(self, "mean", frozen_array(mean))
        object.__setattr__(self, "coefficient_factor", frozen_array(factor))
        object.__setattr__(self, "provenance", freeze_metadata(provenance))

    @classmethod
    def from_matrix_normal(cls, mean, row_covariance, channel_covariance, **identity):
        """Convert a frozen MN posterior; no assertion about the new noise family."""
        _, left = _positive_covariance(row_covariance, "frozen kernel row covariance")
        _, right = _positive_covariance(channel_covariance, "frozen kernel channel covariance")
        return cls(mean, np.kron(left, right), **identity)

    @classmethod
    def from_spatial_posterior(cls, posterior, *, geometry, source_model_sha256,
                               response_id, training_records):
        """Convert an admitted physical B09 state without fitting response or noise.

        The caller supplies the actual participant/recording training authority;
        a numerical posterior does not issue it. The saved B09 precision factor
        determines coefficient uncertainty, independently of score-time noise.
        """
        from transfer.spatial_signal_prior import SpatialSignalPosterior

        if not isinstance(posterior, SpatialSignalPosterior):
            raise TypeError("spatial response conversion requires an admitted posterior")
        layout, binding = posterior.prior.layout, posterior.prior.design_binding
        if (geometry.channels != layout.channels or geometry.unit != layout.signal_unit
                or geometry.reference != layout.acquisition_reference
                or binding.get("sample_rate_hz") != geometry.sfreq
                or binding.get("operator_identity") != geometry.operator_identity):
            raise ValueError("spatial response physical geometry, sampling or operator differs")
        require_sha256(binding.get("actual_basis_sha256"), "actual physical response basis")
        records = json.loads(json.dumps(list(training_records), allow_nan=False))
        _validate_training_ledger(records)
        by_id = {record["record_id"]: record for record in records}
        actual = [record["authorization"] for record in posterior.fit_records]
        scopes = {record["fit_scope"] for record in actual}
        if (len(scopes) != 1 or not scopes <= _FIT_SCOPES
                or set(by_id) != {record["record_id"] for record in actual}):
            raise ValueError("spatial posterior and training producer scope/record identities differ")
        scope = next(iter(scopes))
        ordered = []
        for authority in actual:
            record = by_id[authority["record_id"]]
            require_sha256(record.get("source_sha256"), "training original source")
            if (record.get("fit_scope") != scope
                    or record["source_sha256"] != authority["source_sha256"]
                    or record["read_support"] != list(authority["raw_sample_support"])):
                raise ValueError("spatial posterior training source, scope or actual support differs")
            original = authority.get("original_view")
            if original is not None and (not isinstance(original, dict)
                    or original.get("participant_key") != record["participant_key"]):
                raise ValueError("training participant differs from actual spatial producer view")
            ordered.append(record)
        if scope == "target_calibration" and len({r["participant_key"] for r in ordered}) != 1:
            raise ValueError("target calibration response needs one actual training participant")
        p, channels = posterior.mean.shape
        # R_F.T R_F is precision, so L_F=R_F^-1 is a covariance factor.
        # Only coefficient rows change order; factor columns stay latent draws.
        factor_f = solve_triangular(posterior.precision_root, np.eye(p * channels))
        if not np.isfinite(factor_f).all():
            raise UnidentifiableFitError("frozen spatial response factor is not representable")
        row_order = np.arange(p * channels).reshape(p, channels, order="F").ravel()
        original = posterior.record()
        conversion = {"posterior_schema": original["schema"],
            "posterior_artifact_sha256": original["artifact_sha256"],
            "source_order": "column_major_response_by_native_EEG",
            "destination_order": "row_major_coefficient_then_observed_channel",
            "source_precision_root_sha256": array_digest(posterior.precision_root),
            "actual_design_binding": dict(binding), "noise_refitted": False,
            "posterior_recomputed_for_score_noise": False}
        return cls(posterior.mean, factor_f[row_order], geometry, source_model_sha256,
                   response_id, {"fit_scope": scope, "records": ordered,
                                 "spatial_posterior_conversion": conversion})

    def record(self):
        return {"response_id": self.response_id, "geometry": self.geometry.record(),
                "source_model_sha256": self.source_model_sha256,
                "mean_sha256": array_digest(self.mean),
                "coefficient_factor_sha256": array_digest(self.coefficient_factor),
                "coefficient_count": self.mean.size,
                "uncertainty_factor_columns": self.coefficient_factor.shape[1],
                "coefficient_order": "row_major_coefficient_then_observed_channel",
                "provenance": self.provenance}


def _records(records, geometry, *, training=False):
    values = tuple(records)
    if not values:
        raise ValueError("at least one actual independent residual record is required")
    require_identifiers([r.record_id for r in values], "residual record IDs")
    if any(r.geometry != geometry for r in values):
        raise ValueError("residual geometry differs in units, reference, channels or operator")
    if training and (len({r.role for r in values}) != 1 or values[0].role not in _FIT_SCOPES):
        raise ValueError("fit cannot consume held diagnostics or mix source and calibration scopes")
    return values


def _overlap(recording_id, start, stop, ledger):
    return any(row["recording_id"] == recording_id
               and max(start, row["read_support"][0]) < min(stop, row["read_support"][1])
               for row in ledger)


def _prefix_root(temporal, covariance, prefix, family):
    channels = len(temporal)
    spatial = np.linalg.cholesky(covariance)
    if family == "shared":
        whitening = temporal[0].whiten(np.eye(prefix))
        time_root = solve_triangular(whitening, np.eye(prefix), lower=True)
        return np.kron(time_root, spatial)
    order = temporal[0].order
    if not order:
        return np.kron(np.eye(prefix), spatial)
    # General diagonal-lag VAR, including correlated innovations. Solving a
    # stationary state is confined to C*p; no N*C covariance is materialized.
    transition = np.zeros((channels * order, channels * order))
    for lag in range(order):
        transition[:channels, lag * channels:(lag + 1) * channels] = np.diag(
            [noise.coefficients[lag] for noise in temporal])
    transition[channels:, :-channels] = np.eye(channels * (order - 1))
    driving = np.zeros_like(transition)
    driving[:channels, :channels] = covariance
    stationary = solve_discrete_lyapunov(transition, driving)
    stationary = (stationary + stationary.T) / 2
    scale = max(float(np.linalg.norm(stationary)), np.finfo(float).tiny)
    residual = stationary - transition @ stationary @ transition.T - driving
    if not np.isfinite(stationary).all() or np.linalg.norm(residual) > 1e-8 * scale:
        raise UnidentifiableFitError("private VAR stationary state failed its Lyapunov backward check")
    lags = []
    power = np.eye(len(transition))
    for _ in range(prefix):
        lags.append((power @ stationary)[:channels, :channels])
        power = transition @ power
    matrix = np.empty((prefix * channels, prefix * channels))
    for row in range(prefix):
        for column in range(prefix):
            block = lags[row - column] if row >= column else lags[column - row].T
            matrix[row * channels:(row + 1) * channels,
                   column * channels:(column + 1) * channels] = block
    _, root = _positive_covariance(matrix, "private stationary prefix covariance")
    return root


def _posterior_qr(design, response, prior_root):
    width = design.shape[1]
    q, root = np.linalg.qr(np.vstack((design, prior_root)), mode="reduced")
    singular = np.linalg.svd(root, compute_uv=False)
    if len(singular) != width or singular[-1] <= np.finfo(float).eps * max(design.shape) * singular[0]:
        raise UnidentifiableFitError("actual prefix cannot identify the declared diffuse nuisance")
    rhs = q.T @ np.r_[response, np.zeros(len(prior_root))]
    return rhs, root, q, singular[-1]


@dataclass(frozen=True)
class FittedChannelNoise:
    """Immutable noise fit plus a general conditional Gaussian scoring seam."""

    temporal: tuple[TemporalNoise, ...]
    innovation_covariance: np.ndarray
    geometry: NoiseGeometry
    conditioning_samples: int
    family: str
    provenance: dict
    compiled_prefix_root: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        temporal = tuple(self.temporal)
        covariance, spatial = _positive_covariance(self.innovation_covariance, "innovation covariance")
        channels = len(self.geometry.channels)
        if (self.family not in {"shared", "private"} or len(temporal) != channels
                or covariance.shape != (channels, channels) or not temporal
                or len({noise.order for noise in temporal}) != 1
                or type(self.conditioning_samples) is not int
                or self.conditioning_samples < max(2, temporal[0].order)
                or not isinstance(self.provenance, dict)):
            raise ValueError("noise family, observed coordinates and actual prefix are inconsistent")
        for noise in temporal:
            noise.assert_integrity()
            if noise.startup_mode != "stationary_innovations":
                raise ValueError("fixed-prefix mechanism requires a stable stationary innovation lattice")
        if self.family == "shared" and any(
                not np.array_equal(temporal[0].reflection_coefficients, noise.reflection_coefficients)
                for noise in temporal[1:]):
            raise ValueError("shared spectrum cannot carry private reflection parameters")
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if provenance.get("fit_scope") not in _FIT_SCOPES | {"analytic_reference", "frozen_production"}:
            raise ValueError("noise requires an attributable fit, frozen producer or explicit oracle")
        if provenance["fit_scope"] != "analytic_reference":
            _validate_training_ledger(provenance.get("records"))
        if provenance["fit_scope"] in _FIT_SCOPES:
            fitted_shapes = provenance.get("shape_fits")
            actual_shapes = [item.record() for item in temporal[:1 if self.family == "shared" else channels]]
            if fitted_shapes != json.loads(json.dumps(actual_shapes)):
                raise ValueError("noise parameters differ from their actual fitted temporal producer records")
            ledger_by_id = {row["record_id"]: row for row in provenance["records"]}
            for channel, item in enumerate(temporal[:1 if self.family == "shared" else channels]):
                shape_provenance = item.provenance
                if (shape_provenance["fit_scope"] != provenance["fit_scope"]
                        or shape_provenance["n_columns"] != (channels if self.family == "shared" else 1)
                        or {row["record_id"] for row in shape_provenance["records"]} != set(ledger_by_id)):
                    raise ValueError("fitted noise shape and residual scope/observed channels differ")
                for row in shape_provenance["records"]:
                    source = ledger_by_id[row["record_id"]]
                    digest = (source["residual_sha256"] if self.family == "shared"
                              else source["residual_channel_sha256"][channel])
                    if (row["input_sha256"] != digest
                            or row["n_samples"] != source["read_support"][1] - source["read_support"][0]):
                        raise ValueError("fitted temporal noise input differs from actual residual bytes/support")
        prefix = self.conditioning_samples
        root = self.compiled_prefix_root
        supplied_root = root is not None
        if root is None:
            root = _prefix_root(temporal, covariance, prefix, self.family)
        root = _array(root, "compiled stationary prefix root", 2)
        if (root.shape != (prefix * channels, prefix * channels)
                or np.any(np.diag(root) <= 0) or np.any(np.triu(root, 1) != 0)):
            raise ValueError("compiled prefix root requires a positive lower-triangular factor")
        if supplied_root:
            expected_root = _prefix_root(temporal, covariance, prefix, self.family)
            normalized = solve_triangular(expected_root, root, lower=True)
            discrepancy = normalized @ normalized.T - np.eye(len(root))
            if np.linalg.norm(discrepancy) > 1e-8 * np.sqrt(len(root)):
                raise ValueError("compiled prefix state does not obey the declared stationary noise model")
        object.__setattr__(self, "temporal", temporal)
        object.__setattr__(self, "innovation_covariance", frozen_array(covariance))
        object.__setattr__(self, "compiled_prefix_root", frozen_array(root))
        object.__setattr__(self, "_spatial_root", frozen_array(spatial))
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity", metadata_digest(self.record()))
        object.__setattr__(self, "_runtime_guard", self._runtime_state_digest())

    @property
    def order(self):
        return self.temporal[0].order

    @classmethod
    def fit(cls, records, *, family, order, conditioning_samples=32,
            minimum_samples_per_parameter=20):
        values = tuple(records)
        if not values:
            raise ValueError("noise fit needs actual training residual records")
        geometry = values[0].geometry
        values = _records(values, geometry, training=True)
        if family not in {"shared", "private"}:
            raise ValueError("noise fit family must be shared or private")
        if type(order) is not int or not 0 <= order <= 128:
            raise ValueError("AR order must be an integer in 0..128")
        if type(conditioning_samples) is not int or conditioning_samples < max(2, order):
            raise ValueError("all arms need the same prefix at least as long as their AR order")
        if any(len(r.residual) <= conditioning_samples for r in values):
            raise UnidentifiableFitError("actual training prefix leaves no common tail")
        ledger = [r.ledger() for r in values]
        for index, row in enumerate(ledger):
            if _overlap(row["recording_id"], *row["read_support"], ledger[:index]):
                raise ValueError("training residual records overlap physical raw support")
        arrays, ids = [r.residual for r in values], [r.record_id for r in values]
        arguments = {"order": order, "record_ids": ids, "fit_scope": values[0].role,
                     "startup_mode": "stationary_innovations",
                     "minimum_samples_per_parameter": minimum_samples_per_parameter}
        if family == "shared":
            actual = TemporalNoise.fit(arrays, **arguments)
            temporal = (actual,) * len(geometry.channels)
        else:
            temporal = tuple(TemporalNoise.fit([a[:, [c]] for a in arrays], **arguments)
                             for c in range(len(geometry.channels)))
        # Burg shape and spatial covariance are two finite estimation steps.
        # Innovation variance in TemporalNoise is diagnostic, never divided out.
        innovations = np.vstack([np.column_stack(
            [noise.whiten(a[:, [c]])[conditioning_samples:, 0]
             for c, noise in enumerate(temporal)]) for a in arrays])
        singular = np.linalg.svd(innovations, compute_uv=False)
        if singular[-1] <= np.finfo(float).eps * max(innovations.shape) * singular[0]:
            raise UnidentifiableFitError("training innovations do not identify observed channel support")
        estimator = LedoitWolf(assume_centered=True).fit(innovations)
        provenance = {"fit_scope": values[0].role, "method": "segmented_Burg_then_common_tail_LedoitWolf",
                      "estimator_is_joint_MLE": False, "records": ledger,
                      "shape_training_support": "full_authorized_residual_records",
                      "covariance_training_rows": len(innovations),
                      "covariance_training_support": "same_fixed_prefix_tail",
                      "covariance_shrinkage": float(estimator.shrinkage_),
                      "shape_fits": [noise.record() for noise in temporal[:1 if family == "shared" else len(temporal)]],
                      "implementation_sha256": _IMPLEMENTATION_SHA256}
        return cls(temporal, estimator.covariance_, geometry, conditioning_samples, family, provenance)

    @classmethod
    def from_parameters(cls, coefficients, covariance, *, geometry, conditioning_samples,
                        reference_id, family="private"):
        """Known-noise oracle. Never records a simulated or external input as a fit."""
        coefficients = _array(coefficients, "analytic AR coefficients", 2)
        if coefficients.shape[0] != len(geometry.channels):
            raise ValueError("analytic AR coefficients must match observed channels")
        temporal = tuple(TemporalNoise.from_parameters(row, innovation_variance=1,
                         reference_id=reference_id, startup_mode="stationary_innovations")
                         for row in coefficients)
        return cls(temporal, covariance, geometry, conditioning_samples, family,
                   {"fit_scope": "analytic_reference", "reference_id": reference_id, "records": []})

    @classmethod
    def from_temporal_noise(cls, temporal, covariance, *, geometry, conditioning_samples,
                            source_model_sha256, training_records):
        """Consume the actual frozen shared production object without refitting."""
        require_sha256(source_model_sha256, "frozen production noise source")
        temporal.assert_integrity()
        training = _records(training_records, geometry, training=True)
        actual_ids = {row["record_id"] for row in temporal.provenance.get("records", ())}
        if {record.record_id for record in training} != actual_ids:
            raise ValueError("frozen production noise differs from its actual training record identities")
        return cls((temporal,) * len(geometry.channels), covariance, geometry,
                   conditioning_samples, "shared",
                   {"fit_scope": "frozen_production", "source_model_sha256": source_model_sha256,
                    "actual_temporal_record": temporal.record(),
                    "records": [record.ledger() for record in training]})

    @classmethod
    def from_frozen_shared_arrays(cls, coefficients, reflections, covariance, prefix_root, *,
                                 geometry, conditioning_samples, source_model_sha256,
                                 actual_temporal_record, training_records):
        """A diagnostic consumer of arrays exported by the original verified loader.

        This does not load a production checkpoint or declare it current. The
        original temporal record and model-file SHA remain producer identities;
        the new scorer validates the stable shape and compiled covariance.
        """
        require_sha256(source_model_sha256, "frozen production noise source")
        coefficients = _array(coefficients, "frozen shared AR coefficients", 2)
        reflections = _array(reflections, "frozen shared PARCOR", 2)
        old = json.loads(json.dumps(actual_temporal_record, allow_nan=False))
        expected = old.pop("artifact_sha256", None)
        if expected != metadata_digest(old):
            raise ValueError("original temporal record digest differs from its frozen payload")
        channels = len(geometry.channels)
        if (coefficients.shape != reflections.shape or coefficients.shape[0] != channels
                or any(not np.array_equal(row, old["coefficients"]) for row in coefficients)
                or any(not np.array_equal(row, old["reflection_coefficients"]) for row in reflections)
                or old.get("startup_mode") != "stationary_innovations"):
            raise ValueError("exported shared parameters differ from actual frozen producer state")
        training = _records(training_records, geometry, training=True)
        if {r.record_id for r in training} != {r["record_id"] for r in old["provenance"]["records"]}:
            raise ValueError("exported producer training identities differ from actual frozen noise")
        if old["schema"] == "n2p3_temporal_noise/5":
            temporal = TemporalNoise.from_record(actual_temporal_record)
            role = "frozen_compiled_temporal_state_consumer_preserves_original_producer"
        else:
            temporal = TemporalNoise(coefficients[0], 1,
                {"fit_scope": "analytic_reference", "reference_id": "legacy_diagnostic_shared_noise_consumer",
                 "records": []}, startup_mode="stationary_innovations", reflection_state=reflections[0])
            role = "legacy_diagnostic_array_consumer_not_portable_production_load"
        return cls((temporal,) * channels, covariance, geometry, conditioning_samples, "shared",
                   {"fit_scope": "frozen_production", "source_model_sha256": source_model_sha256,
                    "actual_temporal_record": actual_temporal_record,
                    "records": [r.ledger() for r in training],
                    "role": role}, prefix_root)

    def assert_integrity(self):
        for temporal in {id(item): item for item in self.temporal}.values():
            temporal.assert_integrity()
        if self._runtime_state_digest() != self._runtime_guard:
            raise ValueError("frozen channel noise state changed")

    def _runtime_state_digest(self):
        # Hash compiled numeric bytes directly instead of serializing a C*h
        # prefix factor to JSON at every matrix map in a candidate loop.
        return metadata_digest({"family": self.family, "geometry": self.geometry.record(),
            "prefix": self.conditioning_samples, "provenance": self.provenance,
            "numeric_state_sha256": array_digest(self.innovation_covariance, self.compiled_prefix_root,
                np.asarray([item.coefficients for item in self.temporal]),
                np.asarray([item.reflection_coefficients for item in self.temporal]))})

    def record(self):
        channels = len(self.geometry.channels)
        return {"schema": _SCHEMA, "geometry": self.geometry.record(), "family": self.family,
                "conditioning_samples": self.conditioning_samples,
                "coefficients": [noise.coefficients.tolist() for noise in self.temporal],
                "reflections": [noise.reflection_coefficients.tolist() for noise in self.temporal],
                "compiled_temporal_records": [noise.record() for noise in
                    self.temporal[:1 if self.family == "shared" else len(self.temporal)]],
                "innovation_covariance": self.innovation_covariance.tolist(),
                "compiled_prefix_root": self.compiled_prefix_root.tolist(),
                "provenance": self.provenance,
                "parameter_budget": {"AR_shape": self.order * (1 if self.family == "shared" else channels),
                                     "innovation_covariance": channels * (channels + 1) // 2,
                                     "diffuse_drift_per_independent_record": 2 * channels},
                "prediction_family": "general_vec_Gaussian_not_private_MN_row_posterior"}

    def save(self, path):
        self.assert_integrity()
        target = Path(path)
        # Immutable study outputs do not silently replace previous fits.
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"state": self.record(), "state_sha256": self._integrity}, stream,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_bytes())
        if set(payload) != {"state", "state_sha256"} or metadata_digest(payload["state"]) != payload["state_sha256"]:
            raise ValueError("channel noise artifact digest differs from its actual payload")
        state = payload["state"]
        if state.get("schema") != _SCHEMA:
            raise ValueError("unknown channel noise artifact schema")
        temporal_records = state["compiled_temporal_records"]
        expected_count = 1 if state["family"] == "shared" else len(state["geometry"]["channels"])
        if len(temporal_records) != expected_count:
            raise ValueError("saved compiled temporal state count differs from the noise family")
        actual = tuple(TemporalNoise.from_record(record) for record in temporal_records)
        temporal = actual * len(state["geometry"]["channels"]) if state["family"] == "shared" else actual
        if (not np.array_equal(state["coefficients"], np.asarray([item.coefficients for item in temporal]))
                or not np.array_equal(state["reflections"], np.asarray([item.reflection_coefficients for item in temporal]))):
            raise ValueError("noise arrays disagree with their frozen compiled temporal records")
        fitted = cls(temporal, state["innovation_covariance"], NoiseGeometry(**state["geometry"]),
                     state["conditioning_samples"], state["family"], state["provenance"],
                     state["compiled_prefix_root"])
        if fitted._integrity != payload["state_sha256"]:
            raise ValueError("noise artifact contains unrecognized or inconsistent fields")
        return fitted

    def whiten(self, values):
        """Same full-record map for Y, D B, covariance factors and nuisance.

        values are N x C or N x C x F. Prefix stationary whitening is joint;
        each tail innovation uses actual within-record history, then Sigma.
        """
        self.assert_integrity()
        return self._whiten(values)

    def whitening_logabsdet(self, native_samples: int) -> float:
        """Log determinant of the actual frozen square causal whitening map.

        A short prefix uses the leading block of the *saved* joint prefix root.
        Every subsequent native row contributes one spatial innovation root.
        This is a numerical getter; it grants no observation or fit authority.
        """
        self.assert_integrity()
        if type(native_samples) is not int or native_samples < 1:
            raise ValueError("whitening determinant needs a positive native sample count")
        leading = min(native_samples, self.conditioning_samples)*len(self.geometry.channels)
        return float(-np.log(np.diag(self.compiled_prefix_root)[:leading]).sum()
                     -max(native_samples-self.conditioning_samples, 0)
                     *np.log(np.diag(self._spatial_root)).sum())

    def _whiten(self, values):
        value = finite_real_array(values, name="observation/design map")
        vector = value.ndim == 2
        if vector:
            value = value[:, :, None]
        prefix, channels = self.conditioning_samples, len(self.geometry.channels)
        if value.ndim != 3 or value.shape[1] != channels or len(value) <= prefix:
            raise ValueError("map requires full actual prefix plus a positive tail in observed channels")
        result = np.empty_like(value)
        result[:prefix] = solve_triangular(self.compiled_prefix_root,
            value[:prefix].reshape(prefix * channels, -1), lower=True).reshape(value[:prefix].shape)
        tail = np.empty_like(value[prefix:])
        for channel, noise in enumerate(self.temporal):
            tail[:, channel] = noise.whiten(value[:, channel])[prefix:]
        result[prefix:] = solve_triangular(self._spatial_root,
            tail.transpose(1, 0, 2).reshape(channels, -1), lower=True).reshape(
                channels, len(tail), -1).transpose(1, 0, 2)
        return result[:, :, 0] if vector else result

    def _held_support(self, *, record_id, recording_id, sample_start, n_samples, geometry):
        if geometry != self.geometry:
            raise ValueError("held units, reference, channels or operator differ from frozen noise")
        if (not isinstance(record_id, str) or not record_id.strip()
                or not isinstance(recording_id, str) or not recording_id.strip()
                or type(sample_start) is not int or sample_start < 0):
            raise ValueError("held scoring requires exact record identity and absolute raw support")
        ledger = self.provenance.get("records", ())
        if any(row["record_id"] == record_id for row in ledger) or _overlap(
                recording_id, sample_start, sample_start + n_samples, ledger):
            raise ValueError("held scoring overlaps the noise fit recording support")

    def score(self, samples, candidate_designs, response, *, candidate_ids, geometry,
              record_id, recording_id, sample_start, candidate_prior=None):
        """No truth argument. Prefix updates uncertainty, never frozen C state."""
        self.assert_integrity()
        y = _array(samples, "held samples", 2)
        designs = _array(candidate_designs, "all candidate designs", 3)
        ids = tuple(candidate_ids)
        require_identifiers(ids, "frozen candidate identities")
        self._held_support(record_id=record_id, recording_id=recording_id, sample_start=sample_start,
                           n_samples=len(y), geometry=geometry)
        if (response.geometry != geometry or y.shape[1] != len(geometry.channels)
                or designs.shape != (len(ids), len(y), response.mean.shape[0])):
            raise ValueError("candidate design, ERP state and observed channels do not match")
        response_ledger = response.provenance.get("records", ())
        if any(row["record_id"] == record_id for row in response_ledger) or _overlap(
                recording_id, sample_start, sample_start + len(y), response_ledger):
            raise ValueError("held scoring overlaps the frozen ERP producer training support")
        prior = np.ones(len(ids)) / len(ids) if candidate_prior is None else _array(candidate_prior, "candidate prior", 1)
        if prior.shape != (len(ids),) or np.any(prior <= 0) or not np.isclose(prior.sum(), 1, rtol=1e-12, atol=0):
            raise ValueError("candidate prior must be a positive frozen complete mass")
        prefix, channels = self.conditioning_samples, y.shape[1]
        factor = response.coefficient_factor
        rank = factor.shape[1]
        coordinate = (2 * np.arange(len(y)) - (prefix - 1)) / (prefix - 1)
        drift = np.column_stack((np.ones(len(y)), coordinate))
        # Diffuse nuisance coordinate scaling cancels in the conditional ratio.
        # It keeps volt/microvolt changes from changing the QR rank decision.
        scale = np.sqrt(np.diag(self.innovation_covariance))
        nuisance = np.einsum("nt,cd->nctd", drift, np.diag(scale)).reshape(len(y), channels, 2 * channels)
        white_nuisance = self._whiten(nuisance).reshape(len(y) * channels, 2 * channels)
        proper_root = np.zeros((rank, rank + 2 * channels))
        proper_root[:, :rank] = np.eye(rank)
        prefix_rows = prefix * channels
        nuisance_q, nuisance_root, nuisance_scales = _response_nuisance_qr(
            white_nuisance, require_prefix=prefix_rows)
        projection_geometry = _diffuse_qr_geometry(white_nuisance, nuisance_q, nuisance_root, nuisance_scales)
        densities, diagnostics, mean_contrasts, white_means = [], [], [], []
        # Both marginal potentials use a*=a-y[0]/scale in their common
        # intercept coordinates. This Lebesgue translation has Jacobian one;
        # conditional on the prefix, the raw-tail translation also has unit
        # Jacobian. The original y remains the source/read-support identity.
        chart_y = y-y[0]
        if not np.isfinite(chart_y).all():
            raise DiffuseProjectionResolutionError("raw intercept chart is not representable")
        same_sign = np.signbit(y) == np.signbit(y[0])
        exact_subtraction = ((same_sign & (np.abs(y)/2 <= np.abs(y[0]))
                              & (np.abs(y[0])/2 <= np.abs(y))) | (y == 0) | (y[0] == 0))
        chart_error = _roundoff_gamma(1)*float(np.linalg.norm(np.where(exact_subtraction, 0., chart_y)))
        white_y = self._whiten(chart_y)
        map_geometry = _score_whitening_geometry(self)
        white_y_error = _score_whitening_error(self, chart_y, white_y, chart_error, map_geometry)
        nuisance_error = _score_whitening_error(self, nuisance, white_nuisance.reshape(nuisance.shape),
            _roundoff_gamma(3)*float(np.linalg.norm(nuisance)), map_geometry)
        for design in designs:
            mean = design @ response.mean
            mean_contrasts.append(mean)
            white_mean = self._whiten(mean)
            mean_error = _roundoff_gamma(design.shape[1])*float(
                np.linalg.norm(np.abs(design)@np.abs(response.mean)))
            white_mean_error = _score_whitening_error(self, mean, white_mean, mean_error, map_geometry)
            white_means.append(white_mean.ravel())
            residual = (white_y - white_mean).ravel()
            if rank:
                uncertainty = np.einsum("nk,kcr->ncr", design,
                    factor.reshape(response.mean.shape[0], channels, rank))
                white_uncertainty = self._whiten(uncertainty).reshape(len(y) * channels, rank)
                uncertainty_error = _roundoff_gamma(design.shape[1])*float(np.linalg.norm(
                    np.einsum("nk,kcr->ncr", np.abs(design),
                        np.abs(factor).reshape(response.mean.shape[0], channels, rank))))
                white_uncertainty_error = _score_whitening_error(self, uncertainty,
                    white_uncertainty.reshape(uncertainty.shape), uncertainty_error, map_geometry)
            else:
                white_uncertainty = np.empty((len(y) * channels, 0))
                white_uncertainty_error = 0.
            # The actual frozen covariance image is already the full proper
            # latent prior chart. Reuse record Z-QR only for numerical admission;
            # the raw prefix/tail conditional posterior below is unchanged.
            centering_error = _roundoff_gamma(1)*float(
                _column_norms((np.abs(white_y)+np.abs(white_mean)).reshape(-1, 1))[0])
            _, _, resolution = _project_diffuse_arrays(white_uncertainty, residual, white_nuisance,
                nuisance_q, nuisance_root, nuisance_scales, geometry=projection_geometry,
                response_source_error=centering_error)
            matrix = np.column_stack((white_uncertainty, white_nuisance))
            prefix_rhs, prefix_root, prefix_q, prefix_smallest = _posterior_qr(
                matrix[:prefix_rows], residual[:prefix_rows], proper_root)
            tail_design, tail_y = matrix[prefix_rows:], residual[prefix_rows:]
            full_design = np.vstack((tail_design, prefix_root))
            full_y = np.r_[tail_y, prefix_rhs]
            q, full_root = np.linalg.qr(full_design, mode="reduced")
            full_rhs = q.T@full_y
            full_mean = solve_triangular(full_root, full_rhs)
            remaining = tail_y - tail_design @ full_mean
            # c_p=R_p mu_p encodes the same raw-prefix proper+flat posterior.
            # Avoid subtracting two large nuisance means before applying R_p.
            shift = prefix_root@full_mean-prefix_rhs
            quadratic = float(remaining @ remaining + shift @ shift)
            conditional_resolution = _conditional_score_resolution(
                matrix, residual, proper_root, prefix_rows, prefix_q, prefix_root,
                prefix_smallest, full_design, full_y, q, full_root, full_rhs, full_mean,
                remaining, shift, quadratic,
                white_uncertainty_error+nuisance_error,
                white_y_error+white_mean_error+centering_error)
            logdet = float(2 * (np.log(np.abs(np.diag(full_root))).sum()
                               - np.log(np.abs(np.diag(prefix_root))).sum()))
            noise_logdet = 2 * np.log(np.diag(self._spatial_root)).sum()
            density = -0.5 * (quadratic + logdet + (len(y) - prefix) *
                              (noise_logdet + channels * np.log(2 * np.pi)))
            if not np.isfinite(density):
                raise UnidentifiableFitError("conditional candidate likelihood is nonfinite")
            densities.append(float(density))
            diagnostics.append({"quadratic": quadratic, "predictive_logdet": logdet,
                                "prefix_information_condition": float(np.linalg.cond(prefix_root)),
                                "kernel_uncertainty_columns": rank,
                                "diffuse_projection_resolution": resolution,
                                "conditional_numerical_resolution": conditional_resolution})
        evidence = np.asarray(densities) + np.log(prior)
        log_probability = evidence - logsumexp(evidence)
        probability = np.exp(log_probability)
        maxima = np.flatnonzero(np.isclose(evidence, evidence.max(), rtol=0, atol=1e-10))
        pairs = []
        white_means = np.asarray(white_means)
        projected_means = white_means - (white_means @ nuisance_q) @ nuisance_q.T
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                delta = mean_contrasts[left] - mean_contrasts[right]
                whitened = white_means[left] - white_means[right]
                projected = projected_means[left] - projected_means[right]
                energy = float(whitened @ whitened)
                retained = float(projected @ projected)
                pairs.append({"candidates": [ids[left], ids[right]],
                              "raw_mean_contrast_squared": float(np.sum(delta ** 2)),
                              "fitted_noise_mean_contrast_squared": energy,
                              "after_full_diffuse_projection_squared": retained,
                              "diffuse_retention_fraction": None if energy == 0 else retained / energy})
        return {"schema": "n2p3_channel_noise_candidate_score/1", "candidate_ids": list(ids),
                "log_density": densities, "log_probability": log_probability.tolist(),
                "probabilities": probability.tolist(),
                "winner": ids[int(maxima[0])] if len(maxima) == 1 else None,
                "reason": "ok" if len(maxima) == 1 else "tie",
                "tie_log_evidence_absolute_tolerance": 1e-10,
                "read_support": [sample_start, sample_start + len(y)],
                "conditioning_support": [sample_start, sample_start + prefix],
                "score_support": [sample_start + prefix, sample_start + len(y)],
                "score_estimand": "conditional_tail_frozen_prior_prefix_updates_kernel_and_diffuse_drift",
                "samples_sha256": array_digest(y), "candidate_designs_sha256": array_digest(designs),
                "noise_sha256": self._integrity, "response": response.record(),
                "candidate_diagnostics": diagnostics, "signal_geometry": pairs,
                "signal_geometry_role": "full_record_nuisance_overlap_diagnostic_not_oracle_information",
                "truth_used": False, "parameters_updated": False}

    def fit_response_given_frozen_noise(self, prior, records, *, response_id):
        """Fit authorized C with a fixed physical prior and this exact noise state."""
        return _fit_response_given_frozen_noise(self, prior, records, response_id=response_id)

    def diagnose(self, records, *, max_lag=32, psd_samples=256):
        """Truth-known residual-only diagnostics; never selection or a predictor."""
        self.assert_integrity()
        values = _records(records, self.geometry)
        for record in values:
            if record.role == "held_diagnostic":
                self._held_support(record_id=record.record_id, recording_id=record.recording_id,
                                   sample_start=record.sample_start, n_samples=len(record.residual),
                                   geometry=record.geometry)
        prefix = self.conditioning_samples
        if type(max_lag) is not int or max_lag < 1 or type(psd_samples) is not int or psd_samples < 2:
            raise ValueError("diagnostics require positive lag and PSD sample declarations")
        if any(len(r.residual) <= prefix + max_lag for r in values):
            raise UnidentifiableFitError("residual diagnostic tail cannot support the requested lag/PSD statistic")
        innovations = [self._whiten(r.residual)[prefix:] for r in values]
        return {"residual": _sequence_stats([r.residual[prefix:] for r in values],
                                             self.geometry.sfreq, max_lag, psd_samples),
                "standardized_innovation": _sequence_stats(innovations,
                                             self.geometry.sfreq, max_lag, psd_samples),
                "records": [r.ledger() for r in values],
                "role": "truth_known_residual_noise_diagnostic_no_kernel_predictive_whitening",
                "innovation_covariance_condition": float(np.linalg.cond(self.innovation_covariance)),
                "noise_sha256": self._integrity, "selection_allowed": False}


def _sequence_stats(records, sfreq, max_lag, psd_samples):
    nperseg = min(psd_samples, min(map(len, records)))
    acfs, spectra, means, second_moments = [], [], [], []
    for value in records:
        centered = value - value.mean(axis=0)
        energy = np.sum(centered ** 2, axis=0)
        if np.any(energy <= 0):
            raise UnidentifiableFitError("ACF/PSD diagnostic has a zero-energy observed channel")
        acfs.append(np.asarray([np.sum(centered[:len(value) - lag] * centered[lag:], axis=0) / energy
                                for lag in range(max_lag + 1)]))
        frequencies, spectrum = welch(centered, fs=sfreq, axis=0, nperseg=nperseg,
                                      detrend=False)
        spectra.append(spectrum)
        means.append(value.mean(axis=0))
        second_moments.append(value.T @ value / len(value))
    psd = np.mean(spectra, axis=0)
    mass = psd.sum(axis=0)
    if np.any(mass <= 0):
        raise UnidentifiableFitError("normalized spectrum has no positive observed mass")
    normalized = psd / mass
    distances = np.linalg.norm(normalized[:, :, None] - normalized[:, None, :], axis=0)
    return {"record_lengths": list(map(len, records)), "record_channel_means": np.asarray(means).tolist(),
            "record_second_moments": np.asarray(second_moments).tolist(),
            "acf_lags": list(range(max_lag + 1)), "acf_record_equal": np.mean(acfs, axis=0).tolist(),
            "psd_frequency_hz": frequencies.tolist(), "psd_record_equal": psd.tolist(),
            "normalized_spectrum_distance": distances.tolist(),
            "weighting": "equal_independent_records_no_joined_lag_pairs",
            "centering": "record_local_for_ACF_PSD_only_no_score_change"}


def known_covariance_discriminant_efficiency(delta, true_covariance, assumed_covariance):
    """Equal-covariance oracle efficiency; unavailable from real residuals alone."""
    contrast = _array(delta, "known mean contrast", 1)
    truth, _ = _positive_covariance(true_covariance, "known true covariance")
    assumed, _ = _positive_covariance(assumed_covariance, "assumed covariance")
    if truth.shape != assumed.shape or truth.shape != (len(contrast), len(contrast)):
        raise ValueError("oracle contrast/covariance coordinates differ")
    weight = np.linalg.solve(assumed, contrast)
    oracle = float(contrast @ np.linalg.solve(truth, contrast))
    variance = float(weight @ truth @ weight)
    if oracle <= 0 or variance <= 0:
        raise UnidentifiableFitError("oracle contrast has no identifiable signal")
    return {"oracle_mahalanobis_squared": oracle,
            "assumed_discriminant_snr_squared": float((weight @ contrast) ** 2 / variance),
            "efficiency": float((weight @ contrast) ** 2 / (variance * oracle)),
            "role": "known_equal_covariance_oracle_only"}


def _response_resolved_root(root, name):
    scales = np.max(np.abs(root), axis=0)
    if np.any(scales == 0) or not np.isfinite(scales).all():
        raise UnidentifiableFitError(f"{name} has unresolved coefficient coordinates")
    normalized = root/scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    if (len(singular) != root.shape[1]
            or singular[-1] <= 64*np.finfo(float).eps*max(root.shape)*singular[0]):
        raise UnidentifiableFitError(f"{name} is numerically unresolved in its declared coordinates")
    return scales


@dataclass(frozen=True)
class PhysicalCoefficientPrior:
    mean: np.ndarray
    precision_root: np.ndarray
    basis: np.ndarray
    geometry: NoiseGeometry
    prior_id: str
    provenance: dict

    def __post_init__(self):
        mean, root, basis = (_array(value, name, 2) for value, name in (
            (self.mean, "physical prior mean"), (self.precision_root, "physical prior precision"),
            (self.basis, "actual response basis")))
        if (not isinstance(self.geometry, NoiseGeometry) or min(basis.shape) == 0
                or mean.shape != (2*basis.shape[1], len(self.geometry.channels))
                or root.shape != (mean.size, mean.size) or not isinstance(self.prior_id, str)
                or not self.prior_id.strip() or not isinstance(self.provenance, dict)):
            raise ValueError("physical prior requires its actual common/increment basis and full observed geometry")
        _response_resolved_root(root, "proper physical prior")
        if self.provenance.get("role") not in {"fixed_source_prior", "C_derived_plugin_reference", "analytic_reference"}:
            raise ValueError("physical prior requires an explicit source/C-plugin/oracle role")
        require_sha256(self.provenance.get("source_sha256"), "physical prior source")
        for key, value in (("mean", mean), ("precision_root", root), ("basis", basis)):
            object.__setattr__(self, key, frozen_array(value))
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))

    def record(self):
        return {"prior_id": self.prior_id, "geometry": self.geometry.record(), "provenance": self.provenance,
                "mean_sha256": array_digest(self.mean), "precision_root_sha256": array_digest(self.precision_root),
                "actual_basis_sha256": array_digest(self.basis),
                "coefficient_order": "row_major_coefficient_then_observed_channel",
                "precision_units": "inverse_"+self.geometry.unit,
                "noise_arm_rescaling": False}


@dataclass(frozen=True)
class ResponseFitRecord:
    samples: np.ndarray
    design: np.ndarray
    nuisance: np.ndarray
    known_labels: np.ndarray
    basis_sha256: str
    geometry: NoiseGeometry
    authorization: dict

    def __post_init__(self):
        y, design = _array(self.samples, "actual C volts", 2), _array(self.design, "actual C response design", 2)
        z = finite_real_array(self.nuisance, name="actual C common diffuse coordinates")
        labels = np.asarray(self.known_labels)
        if (not isinstance(self.geometry, NoiseGeometry) or len(y) == 0 or y.shape[1] != len(self.geometry.channels)
                or design.shape[0] != len(y) or not design.shape[1] or z.ndim != 3 or z.shape[:2] != y.shape
                or labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "bifu"
                or not np.isin(labels, [0, 1]).all()):
            raise ValueError("authorized C samples/design/nuisance/labels do not align")
        require_sha256(self.basis_sha256, "actual C basis")
        authority = self.authorization
        if not isinstance(authority, dict) or authority.get("fit_scope") not in {"source_training", "target_calibration"}:
            raise ValueError("response fitting accepts declared source/C labels, never held diagnostics")
        if any(not isinstance(authority.get(key), str) or not authority[key].strip()
               for key in ("record_id", "participant_key", "recording_id")):
            raise ValueError("actual C requires record/participant/recording identities")
        support, label_support = authority.get("read_support"), authority.get("label_support")
        if (not isinstance(support, (list, tuple)) or len(support) != 2
                or any(type(v) is not int for v in support) or support[0] < 0 or support[1]-support[0] != len(y)
                or not isinstance(label_support, (list, tuple)) or len(label_support) != 2
                or any(type(v) is not int for v in label_support)
                or not support[0] <= label_support[0] < label_support[1] <= support[1]):
            raise ValueError("actual C read and known-label support are inconsistent")
        require_sha256(authority.get("source_sha256"), "actual C source")
        require_sha256(authority.get("source_input_sha256"), "actual C original input")
        if authority.get("design_role") != "actual_known_C_event_design":
            raise ValueError("response fit design must come from actual authorized C event labels")
        for key, value in (("samples", y), ("design", design), ("nuisance", z), ("known_labels", labels)):
            object.__setattr__(self, key, frozen_array(value))
        object.__setattr__(self, "authorization", freeze_metadata(authority))

    def ledger(self):
        return {**self.authorization, "geometry": self.geometry.record(), "basis_sha256": self.basis_sha256,
                "samples_sha256": array_digest(self.samples), "design_sha256": array_digest(self.design),
                "nuisance_sha256": array_digest(self.nuisance), "known_labels_sha256": array_digest(self.known_labels)}


@dataclass(frozen=True)
class FittedResponseDrift:
    record_id: str
    mean: np.ndarray
    conditional_factor: np.ndarray
    response_draw_factor: np.ndarray
    response_to_drift: np.ndarray


@dataclass(frozen=True)
class FixedNoiseResponseFit:
    response: FrozenERPResponse
    precision_root: np.ndarray
    precision_response: np.ndarray
    information_roots: tuple[np.ndarray, ...]
    information_responses: tuple[np.ndarray, ...]
    augmented_root: np.ndarray
    drifts: tuple[FittedResponseDrift, ...]
    binding: dict

    def joint_coefficient_drift_factor(self):
        """One shared response draw plus independent conditional drift draws."""
        width = self.response.mean.size
        total = width+sum(len(item.mean) for item in self.drifts)
        factor = np.zeros((total, total))
        factor[:width, :width] = self.response.coefficient_factor
        offset = width
        for item in self.drifts:
            stop = offset+len(item.mean)
            factor[offset:stop, :width] = item.response_draw_factor
            factor[offset:stop, offset:stop] = item.conditional_factor
            offset = stop
        return frozen_array(factor)


def _response_nuisance_qr(z, *, require_prefix=None):
    if z.shape[1] == 0:
        return np.empty((len(z), 0)), np.empty((0, 0)), np.empty(0)
    scales = np.max(np.abs(z), axis=0)
    if np.any(scales == 0):
        raise UnidentifiableFitError("actual common diffuse coordinates are not identified")
    q, root = np.linalg.qr(z/scales, mode="reduced")
    _response_resolved_root(root, "actual common diffuse nuisance")
    if require_prefix is not None:
        _response_nuisance_qr(z[:require_prefix])
    return q, root, scales


class DiffuseProjectionResolutionError(UnidentifiableFitError):
    """The supplied finite diffuse quotient cannot resolve the likelihood."""


@dataclass(frozen=True)
class DiffuseProjection:
    design: np.ndarray
    response: np.ndarray
    nuisance_q: np.ndarray
    nuisance_root: np.ndarray
    nuisance_scales: np.ndarray
    resolution: dict


def _roundoff_gamma(operations):
    product = operations*np.finfo(float).eps
    if product >= 1:
        raise DiffuseProjectionResolutionError("diffuse projection arithmetic has no finite error bound")
    return product/(1-product)


def _score_whitening_geometry(noise):
    """Fixed numerical-map error geometry, computed once for the held record."""
    roots = (noise.compiled_prefix_root, noise._spatial_root)
    result = []
    for root in roots:
        inverse = solve_triangular(root, np.eye(len(root)), lower=True)
        inverse_norm = float(np.linalg.norm(np.abs(inverse), ord=2))
        root_norm = float(np.linalg.norm(np.abs(root), ord=2))
        perturbation = _roundoff_gamma(len(root))*inverse_norm*root_norm
        if not np.isfinite(perturbation) or perturbation >= 1:
            raise DiffuseProjectionResolutionError("conditional whitening solve is numerically unresolved")
        result.append((inverse_norm/(1-perturbation), root_norm))
    # Each PARCOR forward/backward pair has norm <=1+|k|. Initial
    # duplicated history has norm <=sqrt(2); discarded rows only contract it.
    lattice_gain = max(float(np.sqrt(2)*np.prod(1+np.abs(item.reflection_coefficients)))
                       if item.order else 1. for item in noise.temporal)
    return (*result, lattice_gain)


def _score_whitening_error(noise, raw, white, source_error, geometry):
    """Normwise source/lattice/triangular-solve roundoff estimate, not a new map."""
    (prefix_inverse, prefix_root), (spatial_inverse, spatial_root), gain = geometry
    cut = noise.conditioning_samples
    prefix_error = prefix_inverse*(source_error+_roundoff_gamma(cut*len(noise.temporal))*
                                  prefix_root*float(np.linalg.norm(white[:cut])))
    lattice_error = gain*(source_error+_roundoff_gamma(2*noise.order)*float(np.linalg.norm(raw)))
    tail_error = spatial_inverse*(lattice_error+_roundoff_gamma(len(noise.temporal))*
                                 spatial_root*float(np.linalg.norm(white[cut:])))
    return prefix_error+tail_error


def _conditional_score_resolution(matrix, response, prior_root, cut, prefix_q, prefix_root,
                                  prefix_smallest, full_design, full_y, full_q, full_root,
                                  full_rhs, full_mean, remaining, shift, quadratic,
                                  matrix_source_error, response_source_error):
    """Estimate absolute raw conditional log-density error in its actual chart.

    QR reconstruction and source-scale errors perturb the prefix R,c_p, then
    the augmented tail least-squares problem. If ||delta A||<sigma_min(A),
    any perturbed minimizer has norm <=(||b||+||delta b||)/(sigma_min-||delta A||).
    Thus its residual norm moves by at most e=||delta b||+||delta A||*max||theta||,
    and |delta q| <=2||r||e+e^2, plus actual residual/dot arithmetic.
    Determinants use the corresponding singular-value perturbation bound.

    The explicit policy is sqrt(eps) nats per actual tail scalar row. These
    normwise floating-point estimates cover this chart/map/QR program under
    ordinary arithmetic assumptions; they are not an all-input interval
    certificate, a posterior accuracy claim, or the full quotient admission.
    """
    width = matrix.shape[1]
    prefix_design = np.vstack((matrix[:cut], prior_root))
    prefix_y = np.r_[response[:cut], np.zeros(len(prior_root))]
    prefix_error = matrix_source_error+float(np.linalg.norm(prefix_design-prefix_q@prefix_root))
    prefix_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(prefix_q)@np.abs(prefix_root)))
    prefix_relative = prefix_error/prefix_smallest
    if not np.isfinite(prefix_relative) or prefix_relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional prefix posterior is numerically unresolved")
    q_error = (2*prefix_relative/(1-prefix_relative)
               +float(np.linalg.norm(prefix_q.T@prefix_q-np.eye(width), ord=2))
               +_roundoff_gamma(len(prefix_q))*float(np.linalg.norm(np.abs(prefix_q).T@np.abs(prefix_q), ord=2)))
    root_error = prefix_error+q_error*float(np.linalg.norm(prefix_design))
    prefix_root_relative = root_error/prefix_smallest
    if not np.isfinite(prefix_root_relative) or prefix_root_relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional compressed prefix root is numerically unresolved")
    rhs_error = response_source_error+(q_error+_roundoff_gamma(len(prefix_q)))*float(np.linalg.norm(prefix_y))
    design_error = matrix_source_error+root_error+float(np.linalg.norm(full_design-full_q@full_root))
    design_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(full_q)@np.abs(full_root)))
    full_orthogonality = (float(np.linalg.norm(full_q.T@full_q-np.eye(width), ord=2))
        +_roundoff_gamma(len(full_q))*float(np.linalg.norm(np.abs(full_q).T@np.abs(full_q), ord=2)))
    if not np.isfinite(full_orthogonality) or full_orthogonality >= 1:
        raise DiffuseProjectionResolutionError("raw conditional tail QR is numerically unresolved")
    projected_error = (_roundoff_gamma(len(full_y))*float(np.linalg.norm(full_y))
        +float(np.linalg.norm(full_root@full_mean-full_rhs))
        +_roundoff_gamma(width)*float(np.linalg.norm(np.abs(full_root)@np.abs(full_mean))))
    # Q.T Q=S: the true least-squares R theta is S^-1 Q.T b, whereas
    # the computed solve uses Q.T b. Account for this optimality defect.
    optimality_error = np.sqrt(1+full_orthogonality)/(1-full_orthogonality)*(
        projected_error+full_orthogonality*float(np.linalg.norm(full_root@full_mean)))
    data_error = response_source_error+rhs_error+optimality_error
    smallest = float(np.linalg.svd(full_root, compute_uv=False)[-1])*np.sqrt(1-full_orthogonality)
    relative = design_error/smallest
    if not np.isfinite(relative) or relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional tail posterior is numerically unresolved")
    coefficient_norm = max(float(np.linalg.norm(full_mean)),
                           (float(np.linalg.norm(full_y))+data_error)/(smallest-design_error))
    residual_error = data_error+design_error*coefficient_norm
    residual_error += _roundoff_gamma(width+1)*(
        float(np.linalg.norm(np.abs(full_y)+np.abs(full_design)@np.abs(full_mean))))
    quadratic_error = (2*np.sqrt(quadratic)*residual_error+residual_error**2
                       +_roundoff_gamma(len(remaining)+len(shift))*quadratic)
    logdet_error = 2*width*(-np.log1p(-relative)-np.log1p(-prefix_root_relative))
    logdet_error -= width*np.log1p(-full_orthogonality)
    logdet_error += _roundoff_gamma(width+1)*float(
        np.sum(np.abs(np.log(np.abs(np.diag(prefix_root)))))
        +np.sum(np.abs(np.log(np.abs(np.diag(full_root))))))
    tail_rows = len(response)-cut
    error = float(.5*(quadratic_error+logdet_error)+_roundoff_gamma(3)*(quadratic+tail_rows))
    target = float(tail_rows*np.sqrt(np.finfo(float).eps))
    if not np.isfinite(error) or error > target:
        raise DiffuseProjectionResolutionError(
            "raw conditional log-density is numerically unresolved: "
            f"absolute error estimate {error:.6g} exceeds {tail_rows}-row budget {target:.6g}")
    return {"method": "raw_intercept_chart_prefix_QR_conditional_error_estimate",
            "absolute_log_density_error_estimate": error, "absolute_log_density_error_budget": target,
            "actual_tail_scalar_rows": tail_rows, "quadratic_error_estimate": float(quadratic_error),
            "prefix_matrix_relative_error_estimate": float(prefix_relative),
            "compressed_prefix_root_relative_error_estimate": float(prefix_root_relative),
            "tail_matrix_relative_error_estimate": float(relative),
            "full_QR_orthogonality_error_estimate": float(full_orthogonality),
            "original_raw_identity_preserved": True, "all_input_certificate": False}


def _column_norms(values):
    """Scaled Euclidean norms, including zero columns, without squaring volts."""
    maxima = np.max(np.abs(values), axis=0, initial=0.)
    normalized = values/np.where(maxima == 0, 1., maxima)
    return maxima*np.sqrt(np.sum(normalized*normalized, axis=0))


def project_diffuse_nuisance(design, response, nuisance, *, prior_factor,
                             prior_mean=None, require_prefix=None, response_reference_scale=1.):
    """Project one whitened record and assess its original numerical scale.

    ``prior_factor @ prior_factor.T`` must be the complete proper coefficient
    covariance in the supplied design coordinates. For an already standardized
    prior chart it is I. The response is in noise units; a scalar noise shape
    must therefore already have been applied to both design and response.
    ``response_reference_scale`` is 1 for a fixed Gaussian likelihood. A
    Gamma(nu/2,nu/2) scale mixture uses min(1,sqrt(nu)): the rate reference must
    retain both its small-nu rate and the Gaussian noise-unit limit at large nu.

    Small output residuals do not certify accuracy. We estimate source-scale
    QR/subspace and multiplication errors, then bound their effect relative to
    [diag(I,response_reference_scale); projected (D F, y-D m)]. This retains
    the proper prior chart and the declared response rate/noise reference.
    The sqrt(eps) resolution target retains half the binary64
    significant digits in this augmented norm; pure-null roundoff then has at
    most eps squared-likelihood information. Nothing is clipped or rank-dropped.
    This is a conservative floating-point admission estimate, not an interval
    arithmetic certificate for all underlying whitening operations.
    """
    d, y, z = (_array(value, name, ndim) for value, name, ndim in (
        (design, "whitened response design", 2), (response, "whitened response", 1),
        (nuisance, "whitened diffuse design", 2)))
    factor = _array(prior_factor, "complete proper prior covariance factor", 2)
    width = d.shape[1]
    mean = np.zeros(width) if prior_mean is None else _array(prior_mean, "proper prior mean", 1)
    if (len(d) != len(y) or len(z) != len(y) or factor.shape != (width, width)
            or mean.shape != (width,) or z.shape[1] > len(z)):
        raise ValueError("diffuse projection requires consistent complete proper-prior coordinates")
    if (isinstance(response_reference_scale, (bool, np.bool_))
            or not np.isscalar(response_reference_scale)
            or not np.isfinite(response_reference_scale) or not 0 < response_reference_scale <= 1):
        raise ValueError("diffuse projection response rate reference must be positive and retain one noise unit")
    if width:
        _response_resolved_root(factor, "complete proper prior covariance factor")
    q, root, scales = _response_nuisance_qr(z, require_prefix=require_prefix)
    projected_d, projected_y, resolution = _project_diffuse_arrays(d, y, z, q, root, scales,
        factor=factor, mean=mean, response_reference_scale=response_reference_scale)
    return DiffuseProjection(frozen_array(projected_d), frozen_array(projected_y),
        frozen_array(q), frozen_array(root), frozen_array(scales), freeze_metadata(resolution))


def _diffuse_qr_geometry(z, q, root, scales):
    """One source-scale nuisance geometry, reusable across actual candidates."""
    if z.shape[1] == 0 or z.shape[1] == len(z):
        return None
    normalized_z = z/scales
    reconstruction = float(np.linalg.norm(normalized_z-q@root, ord=2))
    reconstruction += _roundoff_gamma(len(scales)+1)*float(
        np.linalg.norm(np.abs(q)@np.abs(root), ord=2))
    smallest = float(np.linalg.svd(root, compute_uv=False)[-1])
    subspace = reconstruction/smallest
    orthogonality = float(np.linalg.norm(q.T@q-np.eye(len(scales)), ord=2))
    absolute_q_squared_norm = float(np.linalg.norm(np.abs(q).T@np.abs(q), ord=2))
    orthogonality += _roundoff_gamma(len(z))*absolute_q_squared_norm
    if not np.isfinite(subspace) or subspace >= 1:
        raise DiffuseProjectionResolutionError("diffuse nuisance subspace is numerically unresolved")
    projector_error = (2*subspace+subspace*subspace)/(1-subspace)+orthogonality
    arithmetic_error = _roundoff_gamma(2*len(z)+2*len(scales)+4)*(1+absolute_q_squared_norm)
    return subspace, orthogonality, projector_error, arithmetic_error


def _project_diffuse_arrays(d, y, z, q, root, scales, *, factor=None, mean=None,
                            geometry=None, response_reference_scale=1., response_source_error=0.):
    """Owned, validated arrays only; factor=None denotes the actual prior chart.

    Scoring reuses its one record QR and geometry. Admission never substitutes
    this quotient for the conditional raw-tail proper+diffuse posterior.
    """
    width = d.shape[1]
    if z.shape[1] == len(z):
        # Full observed nuisance span: the quotient is structurally zero.
        projected_d, projected_y = np.zeros_like(d), np.zeros_like(y)
        resolution = {"method": "full_observed_nuisance_span_structural_zero",
                      "normalized_error_bound": 0., "structural_zero": True}
    elif z.shape[1] == 0:
        projected_d, projected_y = d.copy(), y.copy()
        resolution = {"method": "no_diffuse_projection", "normalized_error_bound": 0.,
                      "structural_zero": False}
    else:
        projected_d, projected_y = d-q@(q.T@d), y-q@(q.T@y)
        if factor is None:
            chart, projected = np.column_stack((d, y)), np.column_stack((projected_d, projected_y))
        else:
            chart = np.column_stack((d@factor, y-d@mean))
            projected = np.column_stack((projected_d@factor, projected_y-projected_d@mean))
        subspace, orthogonality, projector_error, arithmetic_error = (
            _diffuse_qr_geometry(z, q, root, scales) if geometry is None else geometry)
        source_norms = _column_norms(chart)
        errors = projector_error*source_norms
        # Full nonorthogonal prior factors and centered-response operands are
        # included; column maxima of a precision root are not prior deviations.
        if factor is not None:
            absolute_chart_design = np.abs(d)@np.abs(factor)
            absolute_chart_response = np.abs(y)+np.abs(d)@np.abs(mean)
            # Projecting raw D before the full F pullback can amplify roundoff
            # hidden by cancellation in D F. Bound arithmetic in its actual
            # original operands, rather than treating those operations as exact.
            errors[:width] += arithmetic_error*_column_norms(absolute_chart_design)
            errors[-1] += arithmetic_error*float(_column_norms(absolute_chart_response[:, None])[0])
            multiply_error = _roundoff_gamma(width+1)
            errors[:width] += multiply_error*(_column_norms(absolute_chart_design)
                +_column_norms(np.abs(projected_d)@np.abs(factor)))
            errors[-1] += multiply_error*(float(_column_norms(y[:, None])[0])
                +float(_column_norms((np.abs(d)@np.abs(mean))[:, None])[0])
                +float(_column_norms(projected_y[:, None])[0])
                +float(_column_norms((np.abs(projected_d)@np.abs(mean))[:, None])[0]))
        else:
            errors += arithmetic_error*source_norms
        errors[-1] += response_source_error
        if not np.isfinite(projected).all() or not np.isfinite(errors).all():
            raise DiffuseProjectionResolutionError("diffuse prior/noise chart is not representable")
        target = float(np.sqrt(np.finfo(float).eps))
        floor_errors = errors.copy()
        floor_errors[-1] /= response_reference_scale
        normalized_error = float(np.linalg.norm(floor_errors))
        reference_method = "proper_prior_and_noise_floor_bound"
        if normalized_error > target:
            # Strong, genuinely retained information can resolve large input
            # amplitudes. Only this case needs another augmented QR.
            reference_floor = np.eye(width+1)
            reference_floor[-1, -1] = response_reference_scale
            reference = np.linalg.qr(np.vstack((reference_floor, projected)), mode="r")
            inverse = solve_triangular(reference, np.eye(width+1))
            normalized_error = float(np.linalg.norm(errors@np.abs(inverse)))
            reference_method = "projected_augmented_reference_QR"
        if not np.isfinite(normalized_error) or normalized_error > target:
            raise DiffuseProjectionResolutionError(
                "diffuse projection is unresolved in the complete proper-prior/noise chart: "
                f"source-scale error bound {normalized_error:.6g} exceeds binary64 resolution {target:.6g}")
        resolution = {"method": "source_scale_thin_QR_complete_prior_noise_chart",
                      "normalized_error_bound": normalized_error, "resolution_target": target,
                      "source_chart_column_norms": source_norms.tolist(),
                      "chart_column_error_bounds": errors.tolist(),
                      "response_reference_scale": float(response_reference_scale),
                      "reference_method": reference_method,
                      "subspace_error_reference": subspace,
                      "orthogonality_error_reference": orthogonality, "structural_zero": False}
    return projected_d, projected_y, resolution


@dataclass(frozen=True)
class _ResponseLikelihoodRecord:
    raw: ResponseFitRecord
    white_y: np.ndarray
    white_design: np.ndarray
    white_nuisance: np.ndarray
    map_binding: dict


def _compile_current_response_likelihood(noise, row):
    n, channels = row.samples.shape
    width = row.design.shape[1]*channels
    mapping = np.kron(row.design, np.eye(channels)).reshape(n, channels, width)
    combined = np.concatenate((row.samples[:, :, None], mapping, row.nuisance), axis=2)
    white = noise.whiten(combined).reshape(n*channels, -1)
    y, design, z = white[:, 0], white[:, 1:1+width], white[:, 1+width:]
    return _ResponseLikelihoodRecord(row, frozen_array(y), frozen_array(design), frozen_array(z),
        freeze_metadata({"program": "current_actual_B05_whitened_C",
            "noise_sha256": metadata_digest(noise.record()),
            "actual_raw_record": row.ledger(), "white_map_sha256": array_digest(y, design, z)}))


def _fit_response_given_frozen_noise(noise, prior, records, *, response_id):
    """Compile canonical current-noise maps and consume the shared QR core."""
    if not isinstance(noise, FittedChannelNoise) or not isinstance(prior, PhysicalCoefficientPrior):
        raise TypeError("fit needs an admitted B05 frozen noise and physical coefficient prior")
    noise.assert_integrity()
    records = tuple(records)
    if not records or any(not isinstance(row, ResponseFitRecord) for row in records):
        raise ValueError("proper physical response fit needs explicit authorized C records")
    compiled = (_compile_current_response_likelihood(noise, row) for row in records)
    return _fit_response_likelihood_core(prior, records, compiled, geometry=noise.geometry,
        conditioning_samples=noise.conditioning_samples, noise_family=noise.family,
        noise_binding=noise.record(), response_id=response_id)


def _fit_response_likelihood_core(prior, records, likelihood_records, *, geometry, conditioning_samples,
                                  noise_family, noise_binding, response_id):
    """All authorized C Gaussian fit with proper physical prior and fixed noise.

    Same full-record whitening acts on Y, vec response design and diffuse Z.
    Prefix identifies the same nuisance measure used for later conditional score.
    Record roots are compressed incrementally; no full-C covariance or Gram is
    formed. External caller authority remains external to this numerical fit.
    """
    # Record source/role/geometry checks are shared by both map producers.
    records = tuple(records)
    if not records or any(not isinstance(row, ResponseFitRecord) for row in records):
        raise ValueError("response fit requires actual authorized C records")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("fitted physical response needs an identity")
    ids = [row.authorization["record_id"] for row in records]
    require_identifiers(ids, "response C record identities")
    scopes = {row.authorization["fit_scope"] for row in records}
    if len(scopes) != 1 or geometry != prior.geometry:
        raise ValueError("fixed noise/prior/C physical geometry or fit scopes differ")
    if next(iter(scopes)) == "target_calibration" and len({row.authorization["participant_key"] for row in records}) != 1:
        raise ValueError("target response C requires one actual participant")
    for i, row in enumerate(records):
        if (row.geometry != prior.geometry or row.basis_sha256 != array_digest(prior.basis)
                or row.design.shape[1] != prior.mean.shape[0] or len(row.samples) <= conditioning_samples):
            raise ValueError("C design/basis/geometry/prefix differs from fixed physical prior/noise")
        for other in records[:i]:
            left, right = row.authorization["read_support"], other.authorization["read_support"]
            if (row.authorization["recording_id"] == other.authorization["recording_id"]
                    and max(left[0], right[0]) < min(left[1], right[1])):
                raise ValueError("response C records duplicate actual physical observation support")
    width, channels = prior.mean.size, len(prior.geometry.channels)
    prior_factor = np.linalg.solve(prior.precision_root, np.eye(width))
    scales = np.max(np.abs(prior.precision_root), axis=0)
    augmented = np.column_stack((prior.precision_root/scales,
                                  prior.precision_root@prior.mean.ravel()))
    compressed = np.linalg.qr(augmented, mode="r")
    drift_parts, compression, information_roots, information_responses = [], [], [], []
    for row, compiled in zip(records, likelihood_records, strict=True):
        if not isinstance(compiled, _ResponseLikelihoodRecord) or compiled.raw is not row:
            raise ValueError("compiled likelihood producer and actual C records differ")
        y, design, z = compiled.white_y, compiled.white_design, compiled.white_nuisance
        if (y.shape != (row.samples.size,) or design.shape != (row.samples.size, width)
                or z.shape != (row.samples.size, row.nuisance.shape[2])
                or compiled.map_binding["actual_raw_record"] != freeze_metadata(row.ledger())
                or compiled.map_binding["white_map_sha256"] != array_digest(y, design, z)):
            raise ValueError("compiled C maps changed after their actual producer validation")
        quotient = project_diffuse_nuisance(design, y, z, prior_factor=prior_factor,
            prior_mean=prior.mean.ravel(), require_prefix=conditioning_samples*channels)
        qz, rz, zscales = quotient.nuisance_q, quotient.nuisance_root, quotient.nuisance_scales
        zy, zd = qz.T@y, qz.T@design
        projected_y, projected_d = quotient.response, quotient.design
        update = np.column_stack((projected_d/scales, projected_y))
        record_root = np.linalg.qr(update, mode="r")
        information = record_root[:, :width]*scales
        record_response = record_root[:, width]
        information_roots.append(frozen_array(information))
        information_responses.append(frozen_array(record_response))
        compressed = np.linalg.qr(np.vstack((compressed, record_root)), mode="r")
        drift_parts.append((row.authorization["record_id"], rz, zscales, zy, zd))
        compression.append({"record_id": row.authorization["record_id"],
                            "projected_design_sha256": array_digest(projected_d),
                            "projected_response_sha256": array_digest(projected_y),
                            "information_root_sha256": array_digest(information),
                            "information_response_sha256": array_digest(record_response),
                            "compressed_augmented_root_sha256": array_digest(compressed),
                            "nuisance_coordinates": z.shape[1],
                            "diffuse_projection_resolution": quotient.resolution,
                            "map_producer": compiled.map_binding})
    normalized_root, rhs = compressed[:width, :width], compressed[:width, width]
    _response_resolved_root(normalized_root, "given-noise physical posterior")
    vector = solve_triangular(normalized_root, rhs)/scales
    factor = solve_triangular(normalized_root, np.eye(width))/scales[:, None]
    root = normalized_root*scales
    if not (np.isfinite(vector).all() and np.isfinite(factor).all()):
        raise UnidentifiableFitError("given-noise physical response posterior is not representable")
    drifts = []
    for record_id, rz, zscales, zy, zd in drift_parts:
        count = len(zscales)
        if count:
            transform = solve_triangular(rz, zd)/zscales[:, None]
            mean = solve_triangular(rz, zy)/zscales-transform@vector
            conditional = solve_triangular(rz, np.eye(count))/zscales[:, None]
        else:
            transform, mean, conditional = np.empty((0, width)), np.empty(0), np.empty((0, 0))
        drifts.append(FittedResponseDrift(record_id, frozen_array(mean), frozen_array(conditional),
            frozen_array(-transform@factor), frozen_array(transform)))
    mean = vector.reshape(prior.mean.shape)
    provenance = {"fit_scope": next(iter(scopes)), "records": [row.ledger() for row in records],
                  "method": "all_C_given_frozen_noise_streamed_diffuse_QR_fixed_physical_prior",
                  "prior": prior.record(), "actual_noise": noise_binding,
                  "compression": compression, "final_augmented_root_sha256": array_digest(compressed),
                  "prior_column_scales": scales.tolist(), "implementation_sha256": _IMPLEMENTATION_SHA256,
                  "producer_identity_kind": "actual_in_memory_fit_state_not_model_file_or_legacy_relabel",
                  "drift_state": [{"record_id": item.record_id,
                      "actual_arrays_sha256": array_digest(item.mean, item.conditional_factor,
                                                          item.response_draw_factor, item.response_to_drift)}
                      for item in drifts],
                  "hyperparameter_uncertainty_integrated": False, "noise_refitted": False}
    fit_identity = metadata_digest({"provenance": provenance, "mean_sha256": array_digest(mean),
                                   "root_and_rhs_sha256": array_digest(root, rhs), "factor_sha256": array_digest(factor)})
    response = FrozenERPResponse(mean, factor, prior.geometry, fit_identity, response_id, provenance)
    binding = freeze_metadata({"fit_sha256": fit_identity, "prior": prior.record(), "noise_sha256": metadata_digest(noise_binding),
                               "implementation_sha256": _IMPLEMENTATION_SHA256, "fit_scope": next(iter(scopes)),
                               "coefficient_order": "row_major_coefficient_then_observed_channel",
                               "noise_family": noise_family, "conditioning_samples": conditioning_samples,
                               "posterior_scope": "all_authorized_C_prefix_and_tail_given_frozen_parameters",
                               "measurement_authority": "external_caller", "matrix_normal_claimed": False})
    return FixedNoiseResponseFit(response, frozen_array(root), frozen_array(rhs), tuple(information_roots),
                                tuple(information_responses), frozen_array(compressed), tuple(drifts), binding)
