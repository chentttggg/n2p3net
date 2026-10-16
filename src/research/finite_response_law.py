"""One positive finite shape measure and its source-prior moment controls.

Raw state masses are normalized once, as exact rationals of their stored
binary64 values. All inference consumers use the saved effective log masses.
The controls retain the source beta factor, including singular local images.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction
from numbers import Real
from pathlib import Path

import numpy as np

from research.frozen_metadata import freeze_metadata, plain_metadata
from research.latent_response_reference import ResponseState
from transfer.dictionary_artifact import _binary64
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    frozen_array,
    metadata_digest,
    require_sha256,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LAW_SCHEMA = "n2p3_normalized_finite_response_law/1"
LAW_PROGRAM = "positive_binary64_mass_exact_rational_normalization_decimal96_log/1"
_LOG_CONTEXT = Context(prec=96, rounding=ROUND_HALF_EVEN)
_EPS = np.finfo(float).eps
_TINY = np.nextafter(0., 1.)


class FiniteResponseResolutionError(ArithmeticError):
    """A positive law or its coefficient arithmetic exceeds binary64 support."""


def _rounded_fraction(value):
    return _binary64(value.numerator, value.denominator)


def _fraction_error_upper(value):
    if value == 0:
        return 0.
    rounded = _rounded_fraction(value)
    return math.nextafter(rounded, math.inf) if Fraction.from_float(rounded) < value else rounded


def _upper(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise FiniteResponseResolutionError("finite response coefficient error is unrepresentable")
    inflated = values/(1-8*_EPS)
    if not np.isfinite(inflated).all():
        raise FiniteResponseResolutionError("finite response coefficient error is unrepresentable")
    result = np.where(values == 0, 0., np.nextafter(inflated, np.inf))
    if not np.isfinite(result).all():
        raise FiniteResponseResolutionError("finite response coefficient error is unrepresentable")
    return result


def _operation_roundoff(value, left_nonzero=True, right_nonzero=True):
    return _upper(_EPS/(1-_EPS)*np.abs(value)+_TINY*(left_nonzero & right_nonzero))


def _positive_contraction(expression, *operands, terms):
    """Upper-bound a positive dot, including its own reduction arithmetic.

    There are at most one product and one addition per term. The extra eight
    operations include supplied sums such as |F|+eF and final scalar inflation.
    Error operands are bounds, so their positive dot must itself be rounded up;
    the signed central-image dot error cannot cover this separate reduction.
    """
    value = np.einsum(expression, *operands)
    gamma = (2*terms+8)*_EPS/(1-(2*terms+8)*_EPS)
    gradual = 2*terms*_TINY
    return _upper((value+gradual)/(1-gamma))


def _state(value):
    if isinstance(value, Mapping):
        value = ResponseState(**value)
    if not isinstance(value, ResponseState):
        raise TypeError("finite response law requires ResponseState atoms or their records")
    fields = tuple(asdict(value).values())
    if any(not isinstance(x, Real) or isinstance(x, (bool, np.bool_)) for x in fields):
        raise ValueError("finite response parameters must be real physical scalars")
    try:
        numbers = tuple(float(x) for x in fields)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("finite response parameters must be real physical scalars") from error
    if (not all(map(math.isfinite, numbers)) or numbers[:2] != (1., 1.)
            or numbers[2] < 0 or numbers[3] <= 0 or numbers[4] <= 0):
        raise ValueError("this law requires unit amplitudes, nonnegative delay and positive width/mass")
    return ResponseState(*numbers)


@dataclass(frozen=True, init=False)
class FiniteResponseLaw:
    states: tuple[ResponseState, ...]
    log_probability: np.ndarray
    log_probability_error: np.ndarray
    probability_error: np.ndarray
    _probability: np.ndarray
    _metadata: dict
    _identity: str

    @classmethod
    def from_states(cls, states):
        return cls._compile(states, IMPLEMENTATION_SHA256)

    @classmethod
    def _compile(cls, states, producer):
        require_sha256(producer, "finite response law producer")
        states = tuple(_state(value) for value in states)
        if not states:
            raise ValueError("finite response law needs at least one positive atom")
        masses = tuple(Fraction.from_float(value.probability) for value in states)
        total = sum(masses, Fraction(0))
        exact = tuple(value/total for value in masses)
        probability, probability_error, log_probability, log_error = [], [], [], []
        with localcontext(_LOG_CONTEXT):
            for value in exact:
                rounded = _rounded_fraction(value)
                probability.append(rounded)
                probability_error.append(_fraction_error_upper(abs(Fraction.from_float(rounded)-value)))
                logarithm = (Decimal(value.numerator)/Decimal(value.denominator)).ln()
                stored_log = _rounded_fraction(Fraction(logarithm))
                log_probability.append(stored_log)
                # Decimal96 division/log error is far below one binary64 ULP;
                # retain an explicit scalar estimate as well as IEEE rounding.
                log_error.append(0. if value == 1 else math.ulp(stored_log)+1e-88*(1+abs(stored_log)))
        metadata = {"schema": LAW_SCHEMA, "program": LAW_PROGRAM, "implementation_sha256": producer,
            "raw_state_mass_role": "input_weights_only_inference_uses_effective_log_probability",
            "states": [asdict(value) for value in states],
            "raw_mass_total_exact_ratio": [total.numerator, total.denominator],
            "effective_probability_exact_ratio": [[value.numerator, value.denominator] for value in exact],
            "probability": probability, "probability_error": probability_error,
            "log_probability": log_probability, "log_probability_error": log_error,
            "state_unit": "one_shape_per_selection_shared_by_flashes_and_channels",
            "latency_role": "nonnegative_extra_delay", "width_role": "voltage_height_preserving",
            "normalization_owner_count": 1}
        return cls._install(states, metadata)

    @classmethod
    def _install(cls, states, record):
        result = object.__new__(cls)
        for name in ("log_probability", "log_probability_error", "probability_error"):
            object.__setattr__(result, name, frozen_array(record[name]))
        object.__setattr__(result, "_probability", frozen_array(record["probability"]))
        object.__setattr__(result, "states", tuple(states))
        object.__setattr__(result, "_metadata", freeze_metadata(record))
        object.__setattr__(result, "_identity", metadata_digest(record))
        return result

    @property
    def probability(self):
        if np.any(self._probability <= 0):
            raise FiniteResponseResolutionError("positive moment weights require an unrepresentable binary64 mass")
        return self._probability

    def record(self):
        return plain_metadata(self._metadata)

    def digest(self):
        return self._identity

    @property
    def identity_digest(self):
        return self._identity

    @classmethod
    def from_record(cls, record):
        if (not isinstance(record, Mapping) or record.get("schema") != LAW_SCHEMA
                or record.get("program") != LAW_PROGRAM):
            raise ValueError("unsupported finite response law")
        expected = cls._compile(record["states"], record["implementation_sha256"])
        if metadata_digest(record) != expected.digest():
            raise ValueError("saved effective response masses differ from their original positive measure")
        # Keep the original producer and saved numbers after relation admission.
        return cls._install(expected.states, plain_metadata(record))

    def save(self, path):
        from research.execution import atomic_write_json

        atomic_write_json(Path(path), self.record())
        return Path(path)

    @classmethod
    def load(cls, path):
        return cls.from_record(json.loads(Path(path).read_bytes()))

    def mean_design(self, designs, *, design_errors=None):
        designs = finite_real_array(designs, name="finite response designs")
        if designs.ndim != 3 or designs.shape[0] != len(self.states):
            raise ValueError("mean design requires every state on one common observed grid")
        errors = _source_error(design_errors, designs.shape)
        probability = self.probability
        if len(probability) == 1:
            matrix, bound = designs[0].copy(), errors[0].copy()
        else:
            matrix = np.einsum("h,hnk->nk", probability, designs)
            terms = len(probability)
            absolute = _positive_contraction("h,hnk->nk", probability, np.abs(designs), terms=terms)
            bound = _positive_contraction("h,hnk->nk", self.probability_error, np.abs(designs), terms=terms)
            bound += _positive_contraction("h,hnk->nk", probability+self.probability_error, errors, terms=terms)
            gamma = (2*len(probability))*_EPS/(1-2*len(probability)*_EPS)
            bound = _upper(bound+gamma*absolute+len(probability)*_TINY)
        if not np.isfinite(matrix).all():
            raise FiniteResponseResolutionError("finite response mean design is unrepresentable")
        return MeanResponseDesign(frozen_array(matrix), frozen_array(bound), freeze_metadata({
            "schema": "n2p3_finite_response_mean_design/1", "law_digest": self.digest(),
            "design_sha256": array_digest(designs), "design_error_sha256": array_digest(errors),
            "compiler_sha256": IMPLEMENTATION_SHA256, "operation": "one_effective_law_weighted_design_sum"}))

    def moment_components(self, mean, factor, *, factor_source_error=None):
        return moment_components(self, mean, factor, factor_source_error=factor_source_error)


def _source_error(value, shape):
    result = np.zeros(shape) if value is None else finite_real_array(value, name="response source error")
    if result.shape != shape or np.any(result < 0):
        raise ValueError("response source errors must match their coefficient image")
    return result


@dataclass(frozen=True)
class MeanResponseDesign:
    matrix: np.ndarray
    error: np.ndarray
    binding: dict

    def record(self):
        return {"binding": plain_metadata(self.binding), "matrix": self.matrix.tolist(), "error": self.error.tolist()}

    @classmethod
    def from_record(cls, record, *, law, designs, design_errors=None):
        expected = law.mean_design(designs, design_errors=design_errors).record()
        require_sha256(record["binding"]["compiler_sha256"], "original mean-design compiler")
        expected["binding"]["compiler_sha256"] = record["binding"]["compiler_sha256"]
        if metadata_digest(record) != metadata_digest(expected):
            raise ValueError("saved mean design differs from its original law or source designs")
        return cls(frozen_array(record["matrix"]), frozen_array(record["error"]), freeze_metadata(record["binding"]))


@dataclass(frozen=True)
class ResponseMomentComponents:
    state_weights: np.ndarray
    mean: np.ndarray
    shared_factor: np.ndarray
    local_factor: np.ndarray
    mean_error: np.ndarray
    shared_factor_error: np.ndarray
    local_factor_error: np.ndarray
    binding: dict

    def metadata(self):
        return {**plain_metadata(self.binding), "array_sha256": {
            name: array_digest(getattr(self, name)) for name in _MOMENT_ARRAYS}}

    def record(self):
        return {"binding": self.metadata(), **{name: getattr(self, name).tolist() for name in _MOMENT_ARRAYS}}

    @classmethod
    def from_record(cls, record, *, law, mean, factor, factor_source_error=None):
        expected = moment_components(law, mean, factor, factor_source_error=factor_source_error).record()
        require_sha256(record["binding"]["compiler_sha256"], "original moment compiler")
        expected["binding"]["compiler_sha256"] = record["binding"]["compiler_sha256"]
        if metadata_digest(record) != metadata_digest(expected):
            raise ValueError("saved moment image differs from its original law or source beta")
        binding = {key: value for key, value in record["binding"].items() if key != "array_sha256"}
        return cls(*(frozen_array(record[name]) for name in _MOMENT_ARRAYS), freeze_metadata(binding))

    def bank(self, selection_count):
        """Small explicit oracle/fixture bank; long-record callers use blocks."""
        if type(selection_count) is not int or selection_count < 1:
            raise ValueError("a moment bank needs a positive selection count")
        rows, local_width = self.local_factor.shape
        shared_width = self.shared_factor.shape[1]
        factor = np.zeros((selection_count*rows, shared_width+selection_count*local_width))
        error = np.zeros_like(factor)
        factor[:, :shared_width] = np.tile(self.shared_factor, (selection_count, 1))
        error[:, :shared_width] = np.tile(self.shared_factor_error, (selection_count, 1))
        for selection in range(selection_count):
            target = np.s_[selection*rows:(selection+1)*rows,
                shared_width+selection*local_width:shared_width+(selection+1)*local_width]
            factor[target], error[target] = self.local_factor, self.local_factor_error
        return {"mean": frozen_array(np.tile(self.mean, selection_count)), "factor": frozen_array(factor),
            "mean_error": frozen_array(np.tile(self.mean_error, selection_count)), "factor_error": frozen_array(error),
            "component_binding": self.metadata(), "selection_count": selection_count}

    def contract_designs(self, designs, *, channels, design_errors=None):
        """Contract one selection's state designs without a coefficient bank."""
        designs = finite_real_array(designs, name="moment response state designs")
        states = len(self.state_weights)
        if (type(channels) is not int or channels < 1 or designs.ndim != 3
                or designs.shape[0] != states or not designs.shape[1]
                or self.mean.size != states*designs.shape[2]*channels):
            raise ValueError("moment state designs differ from original beta/channel coordinates")
        errors = _source_error(design_errors, designs.shape)
        coefficients = (self.mean[:, None], self.shared_factor, self.local_factor)
        coefficient_errors = (self.mean_error[:, None], self.shared_factor_error, self.local_factor_error)
        outputs = []
        terms = states*designs.shape[2]
        gamma = 2*terms*_EPS/(1-2*terms*_EPS)
        for coefficient, error in zip(coefficients, coefficient_errors, strict=True):
            shape = (states, designs.shape[2], channels, coefficient.shape[1])
            values, bounds = coefficient.reshape(shape), error.reshape(shape)
            image = np.einsum("hnk,hkcr->ncr", designs, values)
            absolute = _positive_contraction("hnk,hkcr->ncr", np.abs(designs), np.abs(values), terms=terms)
            source = _positive_contraction("hnk,hkcr->ncr", np.abs(designs), bounds, terms=terms)
            source += _positive_contraction("hnk,hkcr->ncr", errors, np.abs(values)+bounds, terms=terms)
            image_error = _upper(source+gamma*absolute+terms*_TINY)
            if not np.isfinite(image).all():
                raise FiniteResponseResolutionError("contracted source-prior response is unrepresentable")
            outputs.extend((frozen_array(image), frozen_array(image_error)))
        mean, mean_error, shared, shared_error, local, local_error = outputs
        return ContractedResponseImage(frozen_array(mean[..., 0]), shared, local,
            frozen_array(mean_error[..., 0]), shared_error, local_error, freeze_metadata({
                "schema": "n2p3_contracted_finite_response_moment_image/1", "components": self.metadata(),
                "design_sha256": array_digest(designs), "design_error_sha256": array_digest(errors),
                "channels": channels, "compiler_sha256": IMPLEMENTATION_SHA256,
                "operation": "state_coefficient_sum_then_selection_global_sum_and_local_append"}))


_MOMENT_ARRAYS = ("state_weights", "mean", "shared_factor", "local_factor", "mean_error",
                  "shared_factor_error", "local_factor_error")


@dataclass(frozen=True)
class ContractedResponseImage:
    mean: np.ndarray
    shared_image: np.ndarray
    local_image: np.ndarray
    mean_error: np.ndarray
    shared_image_error: np.ndarray
    local_image_error: np.ndarray
    binding: dict

    def record(self):
        names = ("mean", "shared_image", "local_image", "mean_error", "shared_image_error", "local_image_error")
        return {"binding": plain_metadata(self.binding), **{name: getattr(self, name).tolist() for name in names}}


def moment_components(law, mean, factor, *, factor_source_error=None):
    if not isinstance(law, FiniteResponseLaw):
        raise TypeError("moment controls require the one admitted finite response law")
    mean = finite_real_array(mean, name="original source beta mean")
    factor = finite_real_array(factor, name="original source beta factor")
    if mean.ndim != 1 or factor.ndim != 2 or not len(mean) or factor.shape[0] != len(mean) or not factor.shape[1]:
        raise ValueError("source beta mean and original rectangular factor have different coordinates")
    source_error = _source_error(factor_source_error, factor.shape)
    probability = law.probability
    h, p = len(probability), len(mean)
    if h == 1:
        bank_mean, mean_error = mean.copy(), np.zeros_like(mean)
        shared, shared_error = factor.copy(), source_error.copy()
        local = np.zeros((p, factor.shape[1]+1))
        local_error = np.zeros_like(local)
        categorical_column_sum = 0.
    else:
        weight_error = law.probability_error
        square_root = np.sqrt(probability)
        # For the original rational pi, |sqrt_hat-sqrt(pi)| is bounded by
        # |sqrt_hat**2-pi|/sqrt_hat. This includes sqrt and pi rounding once.
        exact = law.record()["effective_probability_exact_ratio"]
        root_error = np.array([_fraction_error_upper(abs(Fraction.from_float(float(value))**2-Fraction(*ratio))
            /Fraction.from_float(float(value))) for value, ratio in zip(square_root, exact, strict=True)])
        product = np.outer(probability, square_root)
        categorical = np.diag(square_root)-product
        cat_error = weight_error[:, None]*np.abs(square_root)[None, :]
        cat_error += (probability+weight_error)[:, None]*root_error[None, :]
        cat_error += np.diag(root_error)
        cat_error += _operation_roundoff(product, probability[:, None] != 0, square_root[None, :] != 0)
        cat_error += _operation_roundoff(categorical)
        cat_error = _upper(cat_error)
        categorical_column_sum = float(np.max(np.abs(categorical.sum(axis=0))))
        g = np.column_stack((factor, mean))
        g_error = np.column_stack((source_error, np.zeros(p)))
        bank_mean = (probability[:, None]*mean).reshape(h*p)
        mean_error = _upper(weight_error[:, None]*np.abs(mean)
            +_operation_roundoff(bank_mean.reshape(h, p), probability[:, None] != 0, mean != 0)).reshape(h*p)
        shared = np.kron(probability[:, None], factor)
        shared_error = np.kron(weight_error[:, None], np.abs(factor))
        shared_error += np.kron((probability+weight_error)[:, None], source_error)
        shared_error = _upper(shared_error+_operation_roundoff(shared,
            np.kron(probability[:, None] != 0, factor != 0), True))
        local = np.kron(categorical, g)
        local_error = np.kron(cat_error, np.abs(g))+np.kron(np.abs(categorical)+cat_error, g_error)
        local_error = _upper(local_error+_operation_roundoff(local, np.kron(categorical != 0, g != 0), True))
    if not all(np.isfinite(array).all() for array in (bank_mean, shared, local)):
        raise FiniteResponseResolutionError("source-prior moment factor is unrepresentable")
    binding = {"schema": "n2p3_source_finite_response_moment_components/1", "law_digest": law.digest(),
        "source_mean_sha256": array_digest(mean), "source_factor_sha256": array_digest(factor),
        "source_factor_error_sha256": array_digest(source_error), "compiler_sha256": IMPLEMENTATION_SHA256,
        "program": "original_global_beta_and_independent_Gaussian_categorical_residuals/1",
        "coefficient_order": "state_then_original_beta_coordinate", "global_beta_prior_occurrences": 1,
        "local_latent_order": "state_contrast_then_original_source_columns_and_mean_column",
        "categorical_column_sum_roundoff": categorical_column_sum,
        "zero_or_weak_directions_removed": False, "ridge_added": False,
        "error_role": "original_source_and_actual_probability_sqrt_outer_subtract_product_roundoff",
        "process_extension": "share_global_beta_append_only_new_selection_local_prior"}
    return ResponseMomentComponents(*(frozen_array(array) for array in (probability, bank_mean, shared, local,
        mean_error, shared_error, local_error)), freeze_metadata(binding))
