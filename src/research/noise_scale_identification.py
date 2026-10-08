"""Small-matrix projected noise-scale identification reference.

Noise shape K0 and nuisance/state column spaces are fixed before these inputs.
Each complete decision has one independent Gamma scale. All shared ERP and
declared state columns are removed by proper error contrasts; this is a partial
observation likelihood, not the full ERP/noise likelihood or an unknown-state fit.
Caller-owned scopes, complete row maps, fixed shape producer and real-noise
independence must be established externally; declared roles do not grant access.

Exact rational rank costs O(n*p*min(n,p)) operations with growing rational bit
lengths. It is a small reference, not a default large-record admission algorithm.
Finite-df formulas are a moderate-df numerical reference, not a precision
certificate. Nonrepresentable inputs, intermediates or results fail explicitly;
the module supplies no default df, artificial df cap or production entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import fsum

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import expit, gammaln, polygamma

from transfer.noise_model import finite_real_array, frozen_array


class UnidentifiedContrastError(ValueError):
    """No positive finite scale fit exists in the represented contrast model."""


class ContrastNumericalError(ValueError):
    """This numerical reference cannot represent the requested calculation."""


def _degrees(value):
    if isinstance(value, (bool, complex)):
        raise ValueError("nu must be positive or the explicitly requested Gaussian limit")
    nu = float(value)
    if not (np.isposinf(nu) or np.isfinite(nu) and nu > 0):
        raise ValueError("nu must be positive or the explicitly requested Gaussian limit")
    return nu


def _energy(projected):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        energy = float(projected @ projected)
    if not np.isfinite(energy) or energy == 0 and np.any(projected != 0):
        raise ContrastNumericalError("contrast energy is not representable")
    return energy


def _finite_sum(values, name):
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as error:
        raise ContrastNumericalError(f"{name} is not representable") from error
    if not np.isfinite(result):
        raise ContrastNumericalError(f"{name} is not representable")
    return result


def _exact_stored_rank(matrix):
    """Small-matrix rank of actual binary64 operands, before whitening roundoff.

This is only an admission check; it never creates a tiny numerical inverse or
defines away a weak, nonzero nuisance direction. No coefficient bound is assumed.
"""
    rows = [[Fraction.from_float(float(v)) for v in row] for row in matrix]
    rank = 0
    for column in range(matrix.shape[1]):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        leading = rows[rank][column]
        rows[rank] = [value / leading for value in rows[rank]]
        for index in range(rank + 1, len(rows)):
            weight = rows[index][column]
            if weight:
                rows[index] = [left - weight * right
                               for left, right in zip(rows[index], rows[rank], strict=True)]
        rank += 1
        if rank == len(rows):
            break
    return rank


@dataclass(frozen=True)
class CalibrationContrast:
    decision_id: str
    recording_id: str
    support: tuple[int, int]
    role: str
    noise_shape_producer: str
    projected: np.ndarray
    projection: np.ndarray
    nuisance_rank: int

    @property
    def dimension(self):
        return len(self.projected)


def calibration_contrast(samples, nuisance_design, noise_root, *, decision_id,
                         recording_id, read_support, role, noise_shape_producer):
    """Consume caller-declared C support; no authorization or independence grant.

The fixed root L0 is in actual observation units and K0=L0 L0'. Q' L0^-1
annihilates the complete supplied nuisance/state span, including shared ERP.
Original stored rank must agree with resolved whitened SVD rank. ``read_support``
is an explicit half-open native interval; observation-vector rows are not guessed
to be native samples or measured channels. Physical maps remain caller-owned.
"""
    if role != "target_calibration":
        raise ValueError("only declared calibration inputs are supported by this reference")
    y, design, root = (finite_real_array(v, name=name) for v, name in
        ((samples, "observed decision"), (nuisance_design, "complete nuisance design"),
         (noise_root, "fixed producer noise root")))
    if (y.ndim != 1 or design.ndim != 2 or root.shape != (len(y), len(y))
            or design.shape[0] != len(y) or not np.isfinite(y).all()
            or not np.isfinite(design).all() or not np.isfinite(root).all()):
        raise ValueError("finite arrays must share one nonempty observation row grid")
    if len(y) == 0 or np.any(np.triu(root, 1) != 0) or np.any(np.diag(root) <= 0):
        raise ValueError("a full-rank lower noise shape root is required")
    if (not isinstance(read_support, (tuple, list)) or len(read_support) != 2
            or any(type(v) is not int for v in read_support)
            or not 0 <= read_support[0] < read_support[1]
            or any(not isinstance(v, str) or not v for v in
                   (decision_id, recording_id, noise_shape_producer))):
        raise ValueError("decision, recording, relative-scale producer and absolute support are required")
    actual_rank = _exact_stored_rank(design)
    if actual_rank == len(y):
        raise UnidentifiedContrastError("actual stored nuisance/state span leaves no noise contrast")
    white_design = solve_triangular(root, design, lower=True)
    if not np.isfinite(white_design).all():
        raise ContrastNumericalError("whitened nuisance design is not representable")
    scales = np.max(np.abs(white_design), axis=0) if design.shape[1] else np.empty(0)
    white_design = white_design[:, scales > 0] / scales[scales > 0]
    basis, singular, _ = np.linalg.svd(white_design, full_matrices=True)
    rank = int(np.sum(singular > 1e-12 * singular[0])) if len(singular) else 0
    if rank != actual_rank:
        raise ContrastNumericalError("nonzero nuisance directions are unresolved at numerical resolution")
    if rank == len(y):
        raise UnidentifiedContrastError("declared ERP/state span leaves no identifying noise contrast")
    complement = basis[:, rank:]
    projection = solve_triangular(root.T, complement, lower=False).T
    with np.errstate(over="ignore", invalid="ignore"):
        projected = projection @ y
    if not np.isfinite(projection).all() or not np.isfinite(projected).all():
        raise ContrastNumericalError("noise projection or projected observation is not representable")
    return CalibrationContrast(decision_id, recording_id, tuple(read_support), role,
                               noise_shape_producer, frozen_array(projected), frozen_array(projection), rank)


def _ledger(contrasts):
    blocks = tuple(contrasts)
    if not blocks or any(v.role != "target_calibration" for v in blocks):
        raise ValueError("fitting and inner scoring both require declared nonempty calibration blocks")
    if len({v.decision_id for v in blocks}) != len(blocks):
        raise ValueError("one complete decision is one Gamma draw; prefixes cannot duplicate it")
    if len({v.noise_shape_producer for v in blocks}) != 1:
        raise ValueError("the spatial/temporal K0 producer must stay fixed across scale fits")
    for i, left in enumerate(blocks):
        for right in blocks[i + 1:]:
            if (left.recording_id == right.recording_id
                    and max(left.support[0], right.support[0]) < min(left.support[1], right.support[1])):
                raise ValueError("complete decisions overlap their declared read support")
    # Disjoint support is necessary, NOT sufficient for independent real noise.
    # Actual use requires an existing scope owner and a justified independent
    # noise/reset protocol. Producer identity is a declaration, not attestation.
    return blocks


def contrast_log_density(projected, *, degrees_of_freedom, log_shape):
    """Proper marginal t density with the SAME prior df after projection."""
    z = finite_real_array(projected, name="projected observation")
    if z.ndim != 1 or not len(z):
        raise ValueError("a nonempty projected decision vector is required")
    r = len(z)
    q = _energy(z)
    nu = _degrees(degrees_of_freedom)
    if isinstance(log_shape, (bool, complex)) or not np.isfinite(log_shape):
        raise ValueError("log noise shape must be explicitly finite and real")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        standardized_energy = q * np.exp(-log_shape) if q else 0.
    if not np.isfinite(standardized_energy):
        raise ContrastNumericalError("standardized contrast energy is not representable")
    if np.isposinf(nu):
        density = -0.5 * (r * (np.log(2 * np.pi) + log_shape) + standardized_energy)
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            prior_shape, updated_shape = nu / 2, (nu + r) / 2
            log_prior, log_updated = gammaln(prior_shape), gammaln(updated_shape)
            if (not np.isfinite([log_prior, log_updated]).all()
                    or updated_shape <= prior_shape
                    or prior_shape >= 2 and log_updated <= log_prior):
                raise ContrastNumericalError("finite-df Gamma normalization is not numerically resolved")
            density = (log_updated - log_prior
                       - r / 2 * (np.log(nu) + np.log(np.pi) + log_shape)
                       - (nu + r) / 2 * np.log1p(standardized_energy / nu))
    if not np.isfinite(density):
        raise ContrastNumericalError("contrast log density is not representable")
    return float(density)


def fit_contrast_shape(contrasts, *, degrees_of_freedom):
    """One-dimensional exact score root for tau^2 relative to the frozen K0.

The curvature is of the observed projected likelihood in log(tau^2). It is
not an independent-participant count, uncertainty interval or integration of
    noise-shape/df/ERP estimates. Finite-df use is a moderate-df reference.
"""
    blocks = _ledger(contrasts)
    nu = _degrees(degrees_of_freedom)
    dimensions = np.asarray([v.dimension for v in blocks], dtype=float)
    energies = np.asarray([_energy(v.projected) for v in blocks], dtype=float)
    total_dimension = float(dimensions.sum())
    nonzero = energies > 0
    if not nonzero.any():
        raise UnidentifiedContrastError("zero residual has unbounded likelihood at tau^2 -> 0")
    if np.isinf(nu):
        shape = _finite_sum(energies, "Gaussian scale scatter") / total_dimension
        eta = float(np.log(shape))
        curvature = total_dimension / 2
        score = 0.
    else:
        if nu * int(nonzero.sum()) <= dimensions[~nonzero].sum():
            raise UnidentifiedContrastError("finite-df scale optimum lies at zero boundary, not an interior MLE")
        with np.errstate(divide="ignore"):
            log_ratios = np.log(energies) - np.log(nu)

        def derivative(log_shape):
            with np.errstate(over="ignore", invalid="ignore"):
                value = -total_dimension / 2 + .5 * np.sum(
                    (nu + dimensions) * expit(log_ratios - log_shape))
            if not np.isfinite(value):
                raise ContrastNumericalError("noise shape score is not representable")
            return float(value)

        lower, upper = float(np.min(log_ratios[nonzero]) - 40), float(np.max(log_ratios[nonzero]) + 40)
        eta = float(brentq(derivative, lower, upper, xtol=2e-12))
        with np.errstate(over="ignore", under="ignore"):
            shape = float(np.exp(eta))
        fraction = expit(log_ratios - eta)
        curvature = float(.5 * np.sum((nu + dimensions) * fraction * (1 - fraction)))
        score = derivative(eta)
    if not np.isfinite(shape) or shape <= 0 or not np.isfinite(curvature) or curvature <= 0:
        raise ContrastNumericalError("no finite positive shape and observed curvature")
    log_likelihood = _finite_sum((contrast_log_density(v.projected, degrees_of_freedom=nu,
                                log_shape=eta) for v in blocks), "contrast log likelihood")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        covariance = (shape if np.isinf(nu) else nu / (nu - 2) * shape if nu > 2 else None)
    if covariance is not None and not np.isfinite(covariance):
        raise ContrastNumericalError("finite covariance multiplier is not representable")
    return {"degrees_of_freedom": None if np.isinf(nu) else nu,
            "gaussian_limit": bool(np.isinf(nu)), "relative_noise_shape": shape,
            "noise_covariance_multiplier": covariance,
            "log_shape": eta, "log_likelihood": float(log_likelihood),
            "log_shape_score": score, "observed_log_shape_curvature": curvature,
            "independent_scale_draws": len(blocks), "contrast_rows": int(total_dimension),
            "noise_shape_producer": blocks[0].noise_shape_producer,
            "fit_scope": "declared_target_calibration_reference",
            "fit_identity": [v.decision_id for v in blocks],
            "full_joint_MLE": False, "free_spatial_covariance_fitted": False,
            "unknown_state_fitted": False, "parameter_uncertainty_integrated": False}


def projected_t_fisher_information(dimension, degrees_of_freedom):
    """Population information for (log tau^2, nu) in ONE projected decision.

    This supports moderate finite df, not the nonregular Gaussian
boundary. It does not measure a participant population or identify K0/state.
"""
    r, nu = dimension, _degrees(degrees_of_freedom)
    if type(r) is not int or r <= 0 or not np.isfinite(nu) or nu <= 0:
        raise ValueError("a positive contrast dimension and finite df are required")
    total = nu + r
    shape = r * nu / (2 * (total + 2))
    cross = -r / (total * (total + 2))
    degrees = float(.25 * (polygamma(1, nu / 2) - polygamma(1, total / 2))
                    - r * (total + 4) / (2 * nu * total * (total + 2)))
    if not np.isfinite([shape, cross, degrees]).all() or shape <= 0 or degrees <= 0:
        raise ContrastNumericalError("finite-df information is unresolved at numerical precision")
    efficient_degrees = degrees - cross ** 2 / shape
    if not np.isfinite(efficient_degrees) or efficient_degrees <= 0:
        raise ContrastNumericalError("finite-df information is unresolved at numerical precision")
    return {"log_shape_information": float(shape), "cross_information": float(cross),
            "nu_information": degrees, "nu_information_after_profile_shape": efficient_degrees,
            "independent_scale_draws": 1}


def calibration_group_cv(contrasts, *, degrees_grid, folds):
    """Predeclared complete-decision folds on caller-declared C data.

This compares proper contrast densities, NOT held task hit/NLL. All arms use
the same projected dimensions, physical C blocks, frozen K0 and Gamma Eλ=1.
The Gaussian control independently fits its own relative shape on each fold.
"""
    blocks = _ledger(contrasts)
    assignments = tuple(folds)
    if len(assignments) != len(blocks) or len(set(assignments)) < 2:
        raise ValueError("inner folds must assign every complete calibration decision")
    degrees_grid = tuple(_degrees(v) for v in degrees_grid)
    if len(set(degrees_grid)) != len(degrees_grid) or not any(np.isposinf(v) for v in degrees_grid):
        raise ValueError("predeclared unique df grid must retain the Gaussian control")
    results = []
    for nu in degrees_grid:
        terms, fits, matched_terms = [], [], []
        for fold in sorted(set(assignments)):
            train = [v for v, a in zip(blocks, assignments, strict=True) if a != fold]
            score_blocks = [v for v, a in zip(blocks, assignments, strict=True) if a == fold]
            fitted = fit_contrast_shape(train, degrees_of_freedom=nu)
            matched_shape = None
            if np.isfinite(nu) and nu > 2:
                gaussian = fit_contrast_shape(train, degrees_of_freedom=np.inf)
                matched_shape = gaussian["relative_noise_shape"] * ((nu - 2) / nu)
                if not np.isfinite(matched_shape) or matched_shape <= 0:
                    raise ContrastNumericalError("covariance-matched noise shape is not representable")
            fits.append({"fold": fold, "training_decisions": fitted["fit_identity"],
                         "scored_decisions": [v.decision_id for v in score_blocks],
                         "relative_noise_shape": fitted["relative_noise_shape"]})
            terms.extend(contrast_log_density(v.projected, degrees_of_freedom=nu,
                         log_shape=fitted["log_shape"]) for v in score_blocks)
            if matched_shape is not None:
                matched_terms.extend(contrast_log_density(v.projected, degrees_of_freedom=nu,
                    log_shape=np.log(matched_shape)) for v in score_blocks)
        results.append({"degrees_of_freedom": None if np.isinf(nu) else float(nu),
                        "gaussian_limit": bool(np.isinf(nu)),
                        "mean_complete_decision_log_density": _finite_sum(terms, "C mean score") / len(terms),
                        "gaussian_covariance_matched_control_mean_log_density": (
                            _finite_sum(matched_terms, "C matched mean score") / len(matched_terms)
                            if matched_terms else None),
                        "scored_independent_decisions": len(terms), "fold_fits": fits})
    return results
