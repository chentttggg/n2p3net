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
import math
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
    def from_continuous_model(cls, model, *, geometry, source_model_sha256,
                              response_id, training_records):
        """Export the saved modern M/R/Sigma physical posterior, without refitting.

        R is the upper row precision factor. Coefficient rows are common then
        increment, each followed by observed channels. The original Sigma stays
        in this factor when score-time noise changes. No legacy-H fallback.
        """
        from transfer.continuous_decision import ContinuousDecisionModel

        if not isinstance(model, ContinuousDecisionModel):
            raise TypeError("continuous conversion requires an admitted actual model")
        model.assert_integrity()
        if (model.posterior_state is None or model.provenance.get("fit_scope") not in _FIT_SCOPES
                or geometry.channels != model.channels or geometry.sfreq != model.sfreq
                or geometry.unit != "V" or geometry.operator_identity != model.operator_identity):
            raise ValueError("modern response requires its actual fitted R and physical geometry")
        records = json.loads(json.dumps(list(training_records), allow_nan=False))
        _validate_training_ledger(records)
        by_id = {row["record_id"]: row for row in records}
        actual = model.provenance["records"]
        if set(by_id) != {row["record_id"] for row in actual}:
            raise ValueError("response export must preserve every actual producer fit record")
        ordered = []
        for original in actual:
            row = by_id[original["record_id"]]
            view = row.get("original_view")
            if (row.get("fit_scope") != model.provenance["fit_scope"]
                    or row.get("input_sha256") != original["input_sha256"]
                    or row.get("n_samples") != original["n_samples"]
                    or not isinstance(view, dict)
                    or row["record_id"] != view.get("record_id")
                    or row["participant_key"] != view.get("participant_key")
                    or row["read_support"] != view.get("raw_scope")
                    or row["source_sha256"] != view.get("source_sha256")
                    or view.get("reference") != geometry.reference
                    or tuple(view.get("channel_names", ())) != geometry.channels
                    or view.get("sample_rate_hz") != geometry.sfreq or view.get("signal_unit") != geometry.unit
                    or row["read_support"][1] - row["read_support"][0] != original["n_samples"]):
                raise ValueError("response export differs from actual fit inputs or physical support")
            ordered.append(row)
        if model.provenance["fit_scope"] == "target_calibration" and len({r["participant_key"] for r in ordered}) != 1:
            raise ValueError("one target response requires one actual participant")
        mean = model.kernel_mean
        p, channels = mean.shape
        root = model.posterior_state["precision_root"]
        if p != 2 * model.basis.shape[1] or root.shape != (p, p):
            raise ValueError("response posterior is not the actual common/increment coefficient chart")
        inverse = solve_triangular(root, np.eye(p), lower=False)
        spatial = np.linalg.cholesky(model.innovation_covariance)
        factor = np.kron(inverse, spatial)
        if not np.isfinite(factor).all():
            raise UnidentifiableFitError("modern physical coefficient factor is not representable")
        conversion = {"program": "saved_continuous_M_R_original_Sigma_to_physical_beta/1",
            "source_precision_root_sha256": array_digest(root),
            "source_mean_sha256": array_digest(mean),
            "source_spatial_covariance_sha256": array_digest(model.innovation_covariance),
            "actual_basis_sha256": array_digest(model.basis),
            "source_order": "common_then_increment_coefficients_by_observed_channel",
            "destination_order": "row_major_coefficient_then_observed_channel",
            "covariance_factor": "kron(R_inverse,original_spatial_Cholesky)",
            "posterior_recomputed": False, "score_noise_rescaling": False}
        return cls(mean, factor, geometry, source_model_sha256, response_id,
            {"fit_scope": model.provenance["fit_scope"], "records": ordered,
             "original_model_input_records": list(actual), "continuous_posterior_conversion": conversion})

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
    rhs = _row_dot(q, np.r_[response, np.zeros(len(prior_root))])
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
        chart_error = _roundoff_gamma(1)*np.abs(np.where(exact_subtraction, 0., chart_y))
        white_y = self._whiten(chart_y)
        map_geometry = _score_whitening_geometry(self)
        white_y_error = _score_whitening_error(self, chart_y, white_y, chart_error, map_geometry)
        nuisance_error = _score_whitening_error(self, nuisance, white_nuisance.reshape(nuisance.shape),
            _roundoff_gamma(3)*np.abs(nuisance), map_geometry)
        # Only the flat coordinate block changes. The same stored transform
        # defines full/prefix da and is shared by all candidates in this record.
        score_nuisance, nuisance_transform, nuisance_logabsdet = _prefix_nuisance_chart(
            white_nuisance, prefix_rows)
        nuisance_error = _nuisance_chart_error(nuisance_error, white_nuisance,
                                               nuisance_transform, prefix_rows)
        for design in designs:
            mean = design @ response.mean
            mean_contrasts.append(mean)
            white_mean = self._whiten(mean)
            mean_error = _roundoff_gamma(design.shape[1])*(np.abs(design)@np.abs(response.mean))
            white_mean_error = _score_whitening_error(self, mean, white_mean, mean_error, map_geometry)
            white_means.append(white_mean.ravel())
            residual = (white_y - white_mean).ravel()
            if rank:
                uncertainty = np.einsum("nk,kcr->ncr", design,
                    factor.reshape(response.mean.shape[0], channels, rank))
                white_uncertainty = self._whiten(uncertainty).reshape(len(y) * channels, rank)
                uncertainty_error = _roundoff_gamma(design.shape[1])*np.einsum("nk,kcr->ncr", np.abs(design),
                        np.abs(factor).reshape(response.mean.shape[0], channels, rank))
                white_uncertainty_error = _score_whitening_error(self, uncertainty,
                    white_uncertainty.reshape(uncertainty.shape), uncertainty_error, map_geometry)
            else:
                white_uncertainty = np.empty((len(y) * channels, 0))
                white_uncertainty_error = _SegmentError(0., 0., np.empty(0), np.empty(0))
            # The actual frozen covariance image is already the full proper
            # latent prior chart. Reuse record Z-QR only for numerical admission;
            # the raw prefix/tail conditional posterior below is unchanged.
            centering_bounds = _roundoff_gamma(1)*(np.abs(white_y)+np.abs(white_mean))
            centering_error = _SegmentError(float(np.linalg.norm(centering_bounds[:prefix])),
                                            float(np.linalg.norm(centering_bounds[prefix:])))
            _, _, resolution = _project_diffuse_arrays(white_uncertainty, residual, white_nuisance,
                nuisance_q, nuisance_root, nuisance_scales, geometry=projection_geometry,
                response_source_error=centering_error.prefix+centering_error.tail)
            matrix = np.column_stack((white_uncertainty, score_nuisance))
            prefix_rhs, prefix_root, prefix_q, prefix_smallest = _posterior_qr(
                matrix[:prefix_rows], residual[:prefix_rows], proper_root)
            tail_design, tail_y = matrix[prefix_rows:], residual[prefix_rows:]
            full_design = np.vstack((tail_design, prefix_root))
            full_y = np.r_[tail_y, prefix_rhs]
            q, full_root = np.linalg.qr(full_design, mode="reduced")
            full_rhs = _row_dot(q, full_y)
            full_mean = solve_triangular(full_root, full_rhs)
            remaining = tail_y - tail_design @ full_mean
            # c_p=R_p mu_p encodes the same raw-prefix proper+flat posterior.
            # Avoid subtracting two large nuisance means before applying R_p.
            shift = prefix_root@full_mean-prefix_rhs
            quadratic = float(remaining @ remaining + shift @ shift)
            noise_logs = 2*np.log(np.diag(self._spatial_root))
            tail_normalizer = (len(y)-prefix)*(float(noise_logs.sum())+channels*np.log(2*np.pi))
            normalizer_error = .5*(len(y)-prefix)*_roundoff_gamma(channels+5)*(
                float(np.abs(noise_logs).sum())+channels*abs(float(np.log(2*np.pi))))
            conditional_resolution = _conditional_score_resolution(
                matrix, residual, proper_root, prefix_rows, prefix_q, prefix_root,
                prefix_smallest, full_design, full_y, q, full_root, full_rhs, full_mean,
                remaining, shift, quadratic,
                white_uncertainty_error.join_columns(nuisance_error),
                white_y_error+white_mean_error+centering_error,
                normalizer_error=normalizer_error, normalizer_magnitude=abs(tail_normalizer))
            logdet = float(2 * (np.log(np.abs(np.diag(full_root))).sum()
                               - np.log(np.abs(np.diag(prefix_root))).sum()))
            density = -0.5 * (quadratic + logdet + tail_normalizer)
            if not np.isfinite(density):
                raise UnidentifiableFitError("conditional candidate likelihood is nonfinite")
            densities.append(float(density))
            diagnostics.append({"quadratic": quadratic, "predictive_logdet": logdet,
                                "prefix_information_condition": float(np.linalg.cond(prefix_root)),
                                "kernel_uncertainty_columns": rank,
                                "diffuse_projection_resolution": resolution,
                                "conditional_numerical_resolution": conditional_resolution,
                                "flat_coordinate_transform_sha256": array_digest(nuisance_transform),
                                "flat_coordinate_logabsdet": nuisance_logabsdet,
                                "flat_coordinate_measure": "same_full_prefix_da_Jacobian_cancels"})
        evidence = np.asarray(densities) + np.log(prior)
        # The raw V-density can have a common offset above 1e6 nats. Remove
        # that exact common gauge before logsumexp to retain probability mass.
        centered_evidence = evidence-evidence.max()
        log_probability = centered_evidence-logsumexp(centered_evidence)
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
                "scoring_implementation_sha256": _IMPLEMENTATION_SHA256,
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
class _SegmentError:
    """Norm bounds on disjoint native prefix and tail operands."""

    prefix: float
    tail: float
    prefix_columns: np.ndarray | None = field(default=None, repr=False)
    tail_columns: np.ndarray | None = field(default=None, repr=False)

    def __add__(self, other):
        return _SegmentError(float(_projection_upper_add(self.prefix, other.prefix)),
                             float(_projection_upper_add(self.tail, other.tail)))

    def join_columns(self, other):
        prefix = np.r_[self.prefix_columns, other.prefix_columns]
        tail = np.r_[self.tail_columns, other.tail_columns]
        return _SegmentError(float(np.linalg.norm(prefix)), float(np.linalg.norm(tail)),
                             prefix, tail)


_DOT_LEAF_ROWS = 256


def _row_dot(left, right):
    """left.T @ right with fixed short dots and a balanced addition tree.

    The numerical program has at most leaf_rows product/add operations and
    ceil(log2(number_of_leaves)) further additions on every scalar path.
    _row_dot_gamma describes this actual tree, never a full-length BLAS dot.
    """
    leaves = [left[start:start+_DOT_LEAF_ROWS].T @ right[start:start+_DOT_LEAF_ROWS]
              for start in range(0, len(left), _DOT_LEAF_ROWS)]
    if not leaves:
        return left.T @ right
    while len(leaves) > 1:
        leaves = [leaves[index]+leaves[index+1] if index+1 < len(leaves) else leaves[index]
                  for index in range(0, len(leaves), 2)]
    return leaves[0]


def _row_dot_gamma(rows):
    leaves = max(1, (rows+_DOT_LEAF_ROWS-1)//_DOT_LEAF_ROWS)
    depth = (leaves-1).bit_length()
    return _projection_gamma_upper(min(rows, _DOT_LEAF_ROWS)+depth)


def _row_dot_error(left, response):
    """Norm dot error with the actual |Q|.T |b| image, including cancellation.

    Orthogonality controls ||Q||, not ||abs(Q)||. The positive image is bounded
    by its own ordinary reduction enclosure; actual signed dots use the short
    tree. Neither a small Q.T b nor another QR error pays for this dot term.
    """
    image = _projection_upper_matmul(np.abs(left).T, np.abs(response)[:, None])
    return float(_projection_upper_product(_row_dot_gamma(len(left)),
        _projection_norm_upper(image)[0]))


def _prefix_nuisance_chart(nuisance, cut):
    """One invertible flat-only chart, fixed by Z/noise and the public cut.

    Zp = Q R diag(scales), T = diag(scales)^-1 R^-1, a=T a'.
    The same stored T is used on all full/prefix candidates: da=|det T| da'
    cancels in the conditional ratio. T's stored finite values define the
    exact coordinate change; only applying Z @ T adds multiplication error.
    This grants neither extra prefix rank nor a proper prior on flat a'.
    """
    _, root, scales = _response_nuisance_qr(nuisance[:cut])
    transform = solve_triangular(root, np.eye(len(root)))/scales[:, None]
    _response_resolved_root(transform, "prefix flat nuisance coordinate transform")
    chart = nuisance @ transform
    if not np.isfinite(chart).all():
        raise DiffuseProjectionResolutionError("prefix nuisance coordinate image is not representable")
    logabsdet = float(np.log(np.abs(np.diag(transform))).sum())
    return chart, transform, logabsdet


def _nuisance_chart_error(error, original, transform, cut):
    """Pull original white-Z errors through the exact stored invertible T."""
    multiplication = _roundoff_gamma(len(transform))*(np.abs(original)@np.abs(transform))
    # Intercept and slope have very different absolute source error and T
    # scales. A single ||T||*||error|| bound would apply the intercept gain to
    # the huge slope operands and recreate the full-record false rejection.
    prefix = error.prefix_columns@np.abs(transform)+_column_norms(multiplication[:cut])
    tail = error.tail_columns@np.abs(transform)+_column_norms(multiplication[cut:])
    return _SegmentError(float(np.linalg.norm(prefix)), float(np.linalg.norm(tail)), prefix, tail)


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
    """Fixed absolute inverse images, computed once for the held record."""
    roots = (noise.compiled_prefix_root, noise._spatial_root)
    result = []
    for root in roots:
        inverse_bound, _ = _inverse_root_geometry(root, lower=True)
        result.append((inverse_bound, root))
    return tuple(result)


def _inverse_root_geometry(root, *, lower=False):
    """Bound the exact real inverse, including the computed solve's defect.

    If X is the stored approximate inverse and E=I-R X, then
    R^-1=X (I-E)^-1. ||E||<1 gives both a norm inverse bound and
    ||R^-1-X|| <= ||X|| ||E||/(1-||E||), also an entrywise error bound.
    Thus no floating X is silently assumed to be the exact root inverse.
    """
    inverse = solve_triangular(root, np.eye(len(root)), lower=lower)
    inverse_norm = float(np.linalg.norm(inverse, ord=2))
    perturbation = float(np.linalg.norm(np.eye(len(root))-root@inverse, ord=2))
    perturbation += _roundoff_gamma(len(root)+1)*float(np.linalg.norm(
        np.abs(root)@np.abs(inverse), ord=2))
    if not np.isfinite(perturbation) or perturbation >= 1:
        raise DiffuseProjectionResolutionError("conditional root inverse is numerically unresolved")
    inverse_error = perturbation*inverse_norm/(1-perturbation)
    return np.nextafter(np.abs(inverse)+inverse_error, np.inf), inverse_norm/(1-perturbation)


def _triangular_residual(root, values, image):
    """Accurate stored-input residual using two-product and an fsum reduction.

    A normal product has an exactly representable FMA (or Dekker) remainder.
    fsum reduces both limbs and rhs together: roundoff follows the small
    residual, not the cancelling L*x operands. This is a residual certificate,
    not a new whitening map. Nonnormal/overflowing operands remain unresolved.
    """
    residual = np.empty_like(values)
    error = np.empty_like(values)
    fma = getattr(math, "fma", None)
    splitter = 134217729.
    for row in range(len(root)):
        left = root[row, :row+1, None]
        right = image[:row+1]
        product = left*right
        if not np.isfinite(product).all():
            raise DiffuseProjectionResolutionError("whitening residual products are not representable")
        if fma is None:
            split_left, split_right = splitter*left, splitter*right
            if not np.isfinite(split_left).all() or not np.isfinite(split_right).all():
                raise DiffuseProjectionResolutionError("whitening residual splitting is not representable")
            high_left = split_left-(split_left-left)
            high_right = split_right-(split_right-right)
            low_left, low_right = left-high_left, right-high_right
            with np.errstate(over="ignore", invalid="ignore"):
                remainder = ((high_left*high_right-product)+high_left*low_right
                             +low_left*high_right)+low_left*low_right
        else:
            remainder = np.asarray([[fma(float(a), float(b), -float(c))
                for b, c in zip(right[index], product[index], strict=True)]
                for index, a in enumerate(left[:, 0])])
        if not np.isfinite(remainder).all():
            raise DiffuseProjectionResolutionError("whitening residual remainder is not representable")
        try:
            for column in range(values.shape[1]):
                residual[row, column] = math.fsum([float(values[row, column]),
                    *map(float, -product[:, column]), *map(float, -remainder[:, column])])
        except OverflowError as error:
            raise DiffuseProjectionResolutionError("whitening residual sum is not representable") from error
        # Include possible two-product underflow, independent of cancellation.
        # tiny is deliberately a normal upper bound on each subnormal loss.
        underflow = ((np.abs(product) < np.finfo(float).tiny/np.finfo(float).eps**2) & (left != 0) & (right != 0)
                     | ((remainder != 0) & (np.abs(remainder) < np.finfo(float).tiny)))
        error[row] = (2*np.finfo(float).eps*np.abs(residual[row])
                      +np.sum(underflow, axis=0)*np.finfo(float).tiny)
        if not np.isfinite(residual[row]).all() or not np.isfinite(error[row]).all():
            raise DiffuseProjectionResolutionError("whitening residual bound is not representable")
    return residual, error


def _triangular_image_error(root, raw, white, source_error, inverse_bound, *, accurate=False):
    """Bound stored L^-1 input errors separately from solve arithmetic.

    Prefix uses accurate residual/refinement because its stationary L can be
    ill conditioned. Tail spatial L is small: the standard componentwise solve
    bound suffices and leaves the full native PARCOR history unchanged.
    """
    if not np.any(raw) and not np.any(white) and not np.any(source_error):
        return np.zeros_like(white)
    if accurate:
        residual, residual_error = _triangular_residual(root, raw, white)
        correction = solve_triangular(root, residual, lower=True)
        remainder, remainder_error = _triangular_residual(root, residual, correction)
        local = inverse_bound @ (np.abs(remainder)+remainder_error+residual_error+source_error)
        bound = np.abs(correction)+local
        return np.where(bound == 0, 0., np.nextafter(bound, np.inf))
    operands = np.abs(root) @ np.abs(white)
    residual = np.abs(raw-root@white)
    bound = source_error+residual+_roundoff_gamma(len(root)+1)*(operands+np.abs(raw))
    bound = inverse_bound@bound
    return np.where(bound == 0, 0., np.nextafter(bound, np.inf))


def _score_whitening_error(noise, raw, white, source_error, geometry):
    """Local prefix/tail bounds with actual full-record PARCOR error history.

    source_error is componentwise in the original raw coordinates. Prefix
    triangular-solve error is not fed back as a raw tail-history operand.
    The tail recursion carries raw prefix history and its source error itself.
    """
    (prefix_inverse, prefix_root), (spatial_inverse, spatial_root) = geometry
    cut = noise.conditioning_samples
    native, channels = raw.shape[:2]
    values = raw.reshape(native, channels, -1)
    image = white.reshape(native, channels, -1)
    errors = source_error.reshape(values.shape)
    prefix = _triangular_image_error(prefix_root, values[:cut].reshape(cut*channels, -1),
        image[:cut].reshape(cut*channels, -1), errors[:cut].reshape(cut*channels, -1),
        prefix_inverse, accurate=True)
    tail = np.empty_like(values[cut:])
    tail_errors = np.empty_like(tail)
    for channel, temporal in enumerate(noise.temporal):
        x, e = values[:, channel], errors[:, channel]
        if temporal.order:
            forward, backward = x[1:].copy(), x[:-1].copy()
            ef, eb = e[1:].copy(), e[:-1].copy()
            for index, reflection in enumerate(temporal.reflection_coefficients, 1):
                nf, nb = forward+reflection*backward, backward+reflection*forward
                ne_f = ef+abs(reflection)*eb+_roundoff_gamma(2)*(np.abs(forward)+abs(reflection)*np.abs(backward))
                ne_b = eb+abs(reflection)*ef+_roundoff_gamma(2)*(np.abs(backward)+abs(reflection)*np.abs(forward))
                if index < temporal.order:
                    forward, backward = nf[1:], nb[:-1]
                    ef, eb = ne_f[1:], ne_b[:-1]
                else:
                    tail[:, channel] = nf[cut-temporal.order:]
                    tail_errors[:, channel] = ne_f[cut-temporal.order:]
        else:
            tail[:, channel], tail_errors[:, channel] = x[cut:], e[cut:]
    rows = native-cut
    error = _triangular_image_error(spatial_root, tail.transpose(1, 0, 2).reshape(channels, -1),
        image[cut:].transpose(1, 0, 2).reshape(channels, -1),
        tail_errors.transpose(1, 0, 2).reshape(channels, -1), spatial_inverse)
    prefix_columns = _column_norms(prefix)
    tail_columns = _column_norms(error.reshape(channels, rows, -1).reshape(channels*rows, -1))
    return _SegmentError(float(np.linalg.norm(prefix_columns)), float(np.linalg.norm(tail_columns)),
                         prefix_columns, tail_columns)


def _conditional_minimizer_norm(full_mean, matrix_norm, response_norm, smallest,
                                design_error, data_error, optimality_error):
    """Local minimizer perturbation bound, given the caller's resolved estimates.

    For exact current minimizer theta and r=b-A theta, A.T r=0. The
    perturbed minimizer obeys A'.T A' delta=A'.T(f-E theta)+E.T r.
    Normwise input/singular-value estimates remain caller-owned; positive
    arithmetic for this local relation is outward under the existing model.
    No observed residual is substituted for a residual or optimality bound.
    """
    add, multiply = _projection_upper_add, _projection_upper_product

    def ratio(numerator, denominator):
        denominator = np.nextafter(denominator, -np.inf)
        if not np.isfinite(denominator) or denominator <= 0:
            raise DiffuseProjectionResolutionError("conditional minimizer separation is unresolved")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            quotient = numerator/denominator
        # A positive rounded zero is not structural zero. A rounded tiny can
        # also conceal an exact subnormal quotient at the normal boundary.
        if numerator > 0 and quotient <= np.finfo(float).tiny:
            raise DiffuseProjectionResolutionError("conditional minimizer quotient may underflow")
        return float(_projection_outward(quotient))

    observed = float(_projection_norm_upper(np.abs(full_mean)[:, None])[0])
    current = float(add(observed, ratio(optimality_error, smallest)))
    # The caller's matrix differs from Q R by at most design_error. Include
    # that remainder also in the reference norm before perturbing it again.
    reference_norm = float(add(matrix_norm, design_error))
    residual = float(add(response_norm, multiply(reference_norm, current)))
    input_error = add(data_error, multiply(design_error, current))
    numerator = add(multiply(add(reference_norm, design_error), input_error),
                    multiply(design_error, residual))
    separation = np.nextafter(smallest-design_error, -np.inf)
    if separation <= 0 or not np.isfinite(separation):
        raise DiffuseProjectionResolutionError("conditional minimizer separation is unresolved")
    denominator = np.nextafter(separation*separation, -np.inf)
    displacement = ratio(numerator, denominator)
    local = float(add(current, displacement))
    global_bound = ratio(add(response_norm, data_error), separation)
    bound = max(observed, min(local, global_bound))
    return bound, {"observed": observed, "current": current, "displacement": displacement,
                   "local": local, "global": global_bound}


def _conditional_score_resolution(matrix, response, prior_root, cut, prefix_q, prefix_root,
                                  prefix_smallest, full_design, full_y, full_q, full_root,
                                  full_rhs, full_mean, remaining, shift, quadratic,
                                  matrix_source_error, response_source_error, *,
                                  normalizer_error=0., normalizer_magnitude=0.):
    """Estimate absolute raw conditional log-density error in its actual chart.

    QR reconstruction and source-scale errors perturb the prefix R,c_p, then
    the augmented tail least-squares problem. A normal-equation perturbation
    relation bounds the perturbed minimizer around the current resolved one;
    the QR optimality defect is included, rather than assuming measured theta
    itself is an exact minimizer. The older ||b||/sigma bound remains a valid
    independent cap; neither measured residual nor theta replaces an error bound.
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
    prefix_error = matrix_source_error.prefix+float(np.linalg.norm(prefix_design-prefix_q@prefix_root))
    prefix_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(prefix_q)@np.abs(prefix_root)))
    prefix_relative = prefix_error/prefix_smallest
    if not np.isfinite(prefix_relative) or prefix_relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional prefix posterior is numerically unresolved")
    q_error = (2*prefix_relative/(1-prefix_relative)
               +float(np.linalg.norm(_row_dot(prefix_q, prefix_q)-np.eye(width), ord=2))
               +_row_dot_gamma(len(prefix_q))*float(np.linalg.norm(_row_dot(np.abs(prefix_q), np.abs(prefix_q)), ord=2)))
    root_error = prefix_error+q_error*float(np.linalg.norm(prefix_design))
    prefix_root_relative = root_error/prefix_smallest
    if not np.isfinite(prefix_root_relative) or prefix_root_relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional compressed prefix root is numerically unresolved")
    rhs_error = response_source_error.prefix+q_error*float(np.linalg.norm(prefix_y))
    rhs_dot_error = _row_dot_error(prefix_q, prefix_y)
    rhs_error += rhs_dot_error
    reconstruction_error = float(np.linalg.norm(full_design-full_q@full_root))
    reconstruction_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(full_q)@np.abs(full_root)))
    full_orthogonality = (float(np.linalg.norm(_row_dot(full_q, full_q)-np.eye(width), ord=2))
        +_row_dot_gamma(len(full_q))*float(np.linalg.norm(_row_dot(np.abs(full_q), np.abs(full_q)), ord=2)))
    if not np.isfinite(full_orthogonality) or full_orthogonality >= 1:
        raise DiffuseProjectionResolutionError("raw conditional tail QR is numerically unresolved")
    full_rhs_dot_error = _row_dot_error(full_q, full_y)
    projected_error = (full_rhs_dot_error
        +float(np.linalg.norm(full_root@full_mean-full_rhs))
        +_roundoff_gamma(width)*float(np.linalg.norm(np.abs(full_root)@np.abs(full_mean))))
    # Q.T Q=S: the true least-squares R theta is S^-1 Q.T b, whereas
    # the computed solve uses Q.T b. Account for this optimality defect.
    optimality_error = np.sqrt(1+full_orthogonality)/(1-full_orthogonality)*(
        projected_error+full_orthogonality*float(np.linalg.norm(full_root@full_mean)))
    data_error = response_source_error.tail+rhs_error+optimality_error
    # Evaluate input perturbations in the full-posterior metric v=R_full theta.
    # This is a fixed exact coordinate change for the error proof, not a
    # replacement proper prior, a different Z span or a new score program.
    # A strong nuisance column cannot spend its absolute error on an unrelated
    # weak proper direction. Column norm bounds pull back through full R^-1.
    inverse, inverse_norm = _inverse_root_geometry(full_root)
    if matrix_source_error.tail_columns is None:
        source_design_error = matrix_source_error.tail*inverse_norm
    else:
        source_design_error = float(np.linalg.norm(matrix_source_error.tail_columns@inverse))
    design_error = source_design_error+(root_error+reconstruction_error)*inverse_norm
    metric_reconstruction = reconstruction_error*inverse_norm
    smallest = np.sqrt(1-full_orthogonality)-metric_reconstruction
    relative = design_error/smallest
    if not np.isfinite(relative) or relative >= 1:
        raise DiffuseProjectionResolutionError("raw conditional tail posterior is numerically unresolved")
    # full_design R^-1 = Q + (full_design-Q R) R^-1. The Frobenius
    # bound below includes that reconstruction remainder in this same metric.
    matrix_norm = np.sqrt(width*(1+full_orthogonality))+metric_reconstruction
    response_norm = float(_projection_norm_upper(np.abs(full_y)[:, None])[0])
    metric_mean = full_root@full_mean
    coefficient_norm, minimizer_terms = _conditional_minimizer_norm(metric_mean,
        matrix_norm, response_norm, smallest, design_error, data_error, optimality_error)
    # The local perturbation theorem also bounds displacement from the
    # current minimizer. Use original operand-column bounds at the observed
    # theta and bound only its displacement in the posterior metric. This
    # retains error on weak directions while not spending a strong slope's
    # full absolute error on an unrelated large response coefficient.
    if matrix_source_error.tail_columns is None:
        directional_design_error = matrix_source_error.tail*float(np.linalg.norm(full_mean))
    else:
        directional_design_error = float(matrix_source_error.tail_columns@np.abs(full_mean))
    directional_design_error += (root_error+reconstruction_error)*float(np.linalg.norm(full_mean))
    displacement = minimizer_terms["displacement"]+optimality_error/smallest
    residual_error = data_error+directional_design_error+design_error*displacement
    residual_error += _roundoff_gamma(width+1)*(
        float(np.linalg.norm(np.abs(full_y)+np.abs(full_design)@np.abs(full_mean))))
    quadratic_error = (2*np.sqrt(quadratic)*residual_error+residual_error**2
                       +_roundoff_gamma(len(remaining)+len(shift))*quadratic)
    logdet_error = 2*width*(-np.log1p(-relative)-np.log1p(-prefix_root_relative))
    logdet_error -= width*np.log1p(-full_orthogonality)
    absolute_root_logs = float(
        np.sum(np.abs(np.log(np.abs(np.diag(prefix_root)))))
        +np.sum(np.abs(np.log(np.abs(np.diag(full_root))))))
    # logdet is twice the difference of the two root-log reductions.
    # Its factor two must remain in the error before density multiplies .5.
    logdet_arithmetic_error = 2*_roundoff_gamma(width+1)*absolute_root_logs
    logdet_error += logdet_arithmetic_error
    tail_rows = len(response)-cut
    output_error = _roundoff_gamma(3)*(quadratic+2*absolute_root_logs+normalizer_magnitude+tail_rows)
    error = float(.5*(quadratic_error+logdet_error)+normalizer_error+output_error)
    target = float(tail_rows*np.sqrt(np.finfo(float).eps))
    if not np.isfinite(error) or error > target:
        raise DiffuseProjectionResolutionError(
            "raw conditional log-density is numerically unresolved: "
            f"absolute error estimate {error:.6g} exceeds {tail_rows}-row budget {target:.6g}")
    return {"method": "raw_intercept_chart_prefix_QR_local_minimizer_error_estimate",
            "absolute_log_density_error_estimate": error, "absolute_log_density_error_budget": target,
            "actual_tail_scalar_rows": tail_rows, "quadratic_error_estimate": float(quadratic_error),
            "raw_noise_normalizer_error_estimate": normalizer_error,
            "output_density_roundoff_error_estimate": output_error,
            "predictive_logdet_arithmetic_error_estimate": logdet_arithmetic_error,
            "prefix_matrix_relative_error_estimate": float(prefix_relative),
            "compressed_prefix_root_relative_error_estimate": float(prefix_root_relative),
            "tail_matrix_relative_error_estimate": float(relative),
            "full_QR_orthogonality_error_estimate": float(full_orthogonality),
            "prefix_rhs_dot_error_estimate": rhs_dot_error,
            "full_rhs_dot_error_estimate": full_rhs_dot_error,
            "observed_coefficient_norm_upper": minimizer_terms["observed"],
            "current_minimizer_norm_estimate": minimizer_terms["current"],
            "perturbed_minimizer_displacement_estimate": minimizer_terms["displacement"],
            "local_coefficient_norm_estimate": minimizer_terms["local"],
            "global_coefficient_norm_estimate": minimizer_terms["global"],
            "directional_source_design_error_estimate": directional_design_error,
            "posterior_metric_minimizer_displacement_estimate": displacement,
            "prefix_source_design_error_estimate": matrix_source_error.prefix,
            "tail_source_design_error_estimate": matrix_source_error.tail,
            "tail_source_design_posterior_metric_error_estimate": source_design_error,
            "conditional_error_coordinate": "fixed_actual_full_posterior_root_metric",
            "prefix_source_response_error_estimate": response_source_error.prefix,
            "tail_source_response_error_estimate": response_source_error.tail,
            "dot_program": "fixed_256_row_leaves_balanced_addition_tree",
            "original_raw_identity_preserved": True, "all_input_certificate": False}


def _column_norms(values):
    """Scaled Euclidean norms, including zero columns, without squaring volts."""
    maxima = np.max(np.abs(values), axis=0, initial=0.)
    normalized = values/np.where(maxima == 0, 1., maxima)
    return maxima*np.sqrt(np.sum(normalized*normalized, axis=0))


def _projection_outward(values):
    """Round positive normal results upward; retain an exact zero as zero."""
    values = np.asarray(values, dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any((values != 0) & (values < np.finfo(float).tiny))):
        raise DiffuseProjectionResolutionError("projection error bound has nonnormal arithmetic")
    result = np.where(values == 0, 0., np.nextafter(values, np.inf))
    if not np.isfinite(result).all():
        raise DiffuseProjectionResolutionError("projection error bound is not representable")
    return result


def _projection_gamma_upper(count):
    # Shape counts below 2**53 have exact integer-to-binary64 conversion.
    if not 0 <= count < 2**53:
        raise DiffuseProjectionResolutionError("projection operation count is not representable")
    if count == 0:
        return 0.
    product = float(_projection_outward(count*np.finfo(float).eps))
    denominator = np.nextafter(1-product, -np.inf)
    if denominator <= 0:
        raise DiffuseProjectionResolutionError("projection arithmetic has no finite error bound")
    return float(_projection_outward(product/denominator))


def _projection_upper_product(left, right):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw = np.multiply(left, right)
    if np.any((np.asarray(left) > 0) & (np.asarray(right) > 0) & (raw == 0)):
        raise DiffuseProjectionResolutionError("projection bound product underflowed")
    return _projection_outward(raw)


def _projection_upper_add(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _projection_outward(np.add(left, right))


def _projection_reduction_upper(raw, count):
    denominator = np.nextafter(1-_projection_gamma_upper(count), -np.inf)
    if denominator <= 0:
        raise DiffuseProjectionResolutionError("projection positive reduction is unresolved")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return _projection_outward(raw/denominator)


def _projection_require_normal_products(left, right):
    # A global minimum is conservative even with disjoint zero supports.
    # This rejects possible hidden underflow before a positive dot reduction.
    smallest_left = np.min(left, where=left > 0, initial=np.inf)
    smallest_right = np.min(right, where=right > 0, initial=np.inf)
    if np.isfinite(smallest_left) and np.isfinite(smallest_right):
        with np.errstate(over="ignore", under="ignore"):
            smallest_product = smallest_left*smallest_right
        # A rounded tiny can conceal an exact subnormal product at the tie.
        if smallest_product <= np.finfo(float).tiny:
            raise DiffuseProjectionResolutionError("projection dot products may underflow")


def _projection_upper_matmul(left, right):
    _projection_require_normal_products(left, right)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw = left@right
    return _projection_reduction_upper(raw, left.shape[-1])


def _projection_scaled_columns(values):
    maxima = np.max(values, axis=0, initial=0.)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = values/np.where(maxima == 0, 1., maxima)
    if np.any((values > 0) & (normalized == 0)):
        raise DiffuseProjectionResolutionError("projection column normalization underflowed")
    return _projection_outward(normalized), maxima


def _projection_norm_upper(values):
    """Vectorized positive column norms with an outward reduction bound."""
    normalized, maxima = _projection_scaled_columns(values)
    squares = _projection_upper_product(normalized, normalized)
    total = _projection_reduction_upper(np.sum(squares, axis=0), len(values))
    return _projection_upper_product(maxima, _projection_outward(np.sqrt(total)))


def _projection_arithmetic_columns(values, q, *, coefficients=None):
    """Bound fl(X-Q fl(Q.T X))-exact(X-Q Q.T X), in stored Q coordinates.

    Both positive matrix products used to compute the bound are rounded upward
    under the same standard binary64 dot model as the existing gamma estimate.
    Upward column normalization keeps 1e+/-200 operands away from squared overflow.
    Nonnormal products that this model cannot certify fail explicitly.
    """
    absolute_values = np.abs(values)
    absq = np.abs(q)
    _projection_require_normal_products(absq, absolute_values)
    coefficients = _row_dot(q, values) if coefficients is None else coefficients
    if not np.isfinite(coefficients).all():
        raise DiffuseProjectionResolutionError("projection coefficients are not representable")
    _projection_require_normal_products(absq, np.abs(coefficients))
    absolute, scales = _projection_scaled_columns(absolute_values)
    rows, width = q.shape
    g_rows, g_width = _row_dot_gamma(rows), _projection_gamma_upper(width)
    projected_dot = _projection_upper_matmul(absq.T, absolute)
    dot_error = _projection_upper_product(g_rows, projected_dot)
    dot_bound = _projection_upper_add(projected_dot, dot_error)
    energy = np.zeros(values.shape[1])
    # The positive bound image is blocked; actual coefficient dots use the
    # fixed short-leaf tree above. Avoid several
    # simultaneous N x response-width bound arrays for an actual long record.
    for start in range(0, rows, 4096):
        block_q, block_x = absq[start:start+4096], absolute[start:start+4096]
        image_bound = _projection_upper_matmul(block_q, dot_bound)
        image_error = _projection_upper_add(_projection_upper_matmul(block_q, dot_error),
            _projection_upper_product(g_width, image_bound))
        image_bound = _projection_upper_product(_projection_upper_add(1., g_width), image_bound)
        subtraction_error = _projection_upper_product(_projection_gamma_upper(1),
            _projection_upper_add(block_x, image_bound))
        total = _projection_upper_add(image_error, subtraction_error)
        block_norm = _projection_norm_upper(total)
        energy = _projection_upper_add(energy, _projection_upper_product(block_norm, block_norm))
    return _projection_upper_product(scales, _projection_outward(np.sqrt(energy)))


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
    orthogonality = float(np.linalg.norm(_row_dot(q, q)-np.eye(len(scales)), ord=2))
    absolute_q_squared_norm = float(np.linalg.norm(_row_dot(np.abs(q), np.abs(q)), ord=2))
    orthogonality += _row_dot_gamma(len(z))*absolute_q_squared_norm
    if not np.isfinite(subspace) or subspace >= 1:
        raise DiffuseProjectionResolutionError("diffuse nuisance subspace is numerically unresolved")
    projector_error = (2*subspace+subspace*subspace)/(1-subspace)+orthogonality
    arithmetic_error = (2*_row_dot_gamma(len(z))+_roundoff_gamma(2*len(scales)+4))*(1+absolute_q_squared_norm)
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
        coefficients_d, coefficients_y = _row_dot(q, d), _row_dot(q, y)
        projected_d, projected_y = d-q@coefficients_d, y-q@coefficients_y
        if factor is None:
            chart, projected = np.column_stack((d, y)), np.column_stack((projected_d, projected_y))
        else:
            chart = np.column_stack((d@factor, y-d@mean))
            projected = np.column_stack((projected_d@factor, projected_y-projected_d@mean))
        subspace, orthogonality, projector_error, _ = (
            _diffuse_qr_geometry(z, q, root, scales) if geometry is None else geometry)
        source_norms = _projection_norm_upper(np.abs(chart))
        errors = _projection_upper_product(projector_error, source_norms)
        arithmetic = _projection_arithmetic_columns(np.column_stack((d, y)), q,
            coefficients=np.column_stack((coefficients_d, coefficients_y)))
        # Full nonorthogonal prior factors and centered-response operands are
        # included; column maxima of a precision root are not prior deviations.
        if factor is not None:
            absolute_chart_design = _projection_upper_matmul(np.abs(d), np.abs(factor))
            absolute_mean = _projection_upper_matmul(np.abs(d), np.abs(mean)[:, None])
            # Projecting raw D before the full F pullback can amplify roundoff
            # hidden by cancellation in D F. Bound arithmetic in its actual
            # original operands, rather than treating those operations as exact.
            factor_arithmetic = _projection_upper_matmul(arithmetic[:width][None, :], np.abs(factor))[0]
            response_arithmetic = _projection_upper_add(arithmetic[-1],
                _projection_upper_matmul(arithmetic[:width][None, :], np.abs(mean)[:, None])[0, 0])
            errors[:width] = _projection_upper_add(errors[:width], factor_arithmetic)
            errors[-1] = _projection_upper_add(errors[-1], response_arithmetic)
            multiply_error = _projection_gamma_upper(width+1)
            design_operands = _projection_upper_add(_projection_norm_upper(absolute_chart_design),
                _projection_norm_upper(_projection_upper_matmul(np.abs(projected_d), np.abs(factor))))
            errors[:width] = _projection_upper_add(errors[:width],
                _projection_upper_product(multiply_error, design_operands))
            response_operands = _projection_upper_add(_projection_norm_upper(np.abs(y)[:, None])[0],
                _projection_norm_upper(absolute_mean)[0])
            response_operands = _projection_upper_add(response_operands,
                _projection_norm_upper(np.abs(projected_y)[:, None])[0])
            response_operands = _projection_upper_add(response_operands,
                _projection_norm_upper(_projection_upper_matmul(np.abs(projected_d), np.abs(mean)[:, None]))[0])
            errors[-1] = _projection_upper_add(errors[-1],
                _projection_upper_product(multiply_error, response_operands))
        else:
            errors = _projection_upper_add(errors, arithmetic)
        errors[-1] = _projection_upper_add(errors[-1], response_source_error)
        if not np.isfinite(projected).all() or not np.isfinite(errors).all():
            raise DiffuseProjectionResolutionError("diffuse prior/noise chart is not representable")
        target = float(np.sqrt(np.finfo(float).eps))
        floor_errors = errors.copy()
        floor_errors[-1] = _projection_outward(floor_errors[-1]/response_reference_scale)
        normalized_error = float(_projection_norm_upper(floor_errors[:, None])[0])
        reference_method = "proper_prior_and_noise_floor_bound"
        if normalized_error > target:
            # Strong, genuinely retained information can resolve large input
            # amplitudes. Only this case needs another augmented QR.
            reference_floor = np.eye(width+1)
            reference_floor[-1, -1] = response_reference_scale
            reference = np.linalg.qr(np.vstack((reference_floor, projected)), mode="r")
            inverse = solve_triangular(reference, np.eye(width+1))
            pulled_error = _projection_upper_matmul(errors[None, :], np.abs(inverse))[0]
            normalized_error = float(_projection_norm_upper(pulled_error[:, None])[0])
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
                       "arithmetic_error_method": "operation_wise_outward_positive_dot_bounds",
                      "dot_program": "fixed_256_row_leaves_balanced_addition_tree",
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
        zy, zd = _row_dot(qz, y), _row_dot(qz, design)
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
