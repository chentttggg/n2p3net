"""A declared linear continuation of original source-native basis columns.

Native sites retain the original binary64 rows. Between those sites the new
function is exact rational linear interpolation, with an explicit knot at L.
This role carries no historical analytic-family or clock authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.dictionary_artifact import (
    COMPILER_SOURCE_SHA256,
    DictionaryArtifactError,
    _binary64,
    exact_query,
)
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    frozen_array,
    metadata_digest,
    require_sha256,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NATIVE_DICTIONARY_SCHEMA = "n2p3_source_native_anchored_dictionary/1"
NATIVE_EVALUATION_SCHEMA = "n2p3_source_native_anchored_evaluation/1"
NATIVE_PROGRAM = "original_native_rows_exact_rational_linear_explicit_terminal/1"


def _rounded(value):
    return _binary64(value.numerator, value.denominator)


def _error_upper(value):
    if value == 0:
        return 0.
    rounded = _rounded(value)
    return math.nextafter(rounded, math.inf) if Fraction.from_float(rounded) < value else rounded


@dataclass(frozen=True)
class NativeAnchoredDictionarySpec:
    response_seconds: float
    n_basis: int
    family: str = field(default="source_native_anchored_linear", init=False)
    endpoint: str = field(default="explicit_terminal_knot", init=False)

    def record(self):
        return asdict(self)


@dataclass(frozen=True, init=False)
class NativeResponseDictionary:
    matrix: np.ndarray
    terminal_values: np.ndarray
    sample_rate_hz: float
    spec: NativeAnchoredDictionarySpec
    _metadata: dict
    _identity: str

    @classmethod
    def from_source_matrix(cls, matrix, *, sample_rate_hz, source_basis_sha256, terminal_values):
        return cls._build(matrix, sample_rate_hz=sample_rate_hz, source_basis_sha256=source_basis_sha256,
            terminal_values=terminal_values, producer=IMPLEMENTATION_SHA256, arithmetic=COMPILER_SOURCE_SHA256)

    @classmethod
    def _build(cls, matrix, *, sample_rate_hz, source_basis_sha256, terminal_values, producer, arithmetic):
        matrix = finite_real_array(matrix, name="original source-native basis")
        terminal = finite_real_array(terminal_values, name="explicit source continuation terminal knot")
        require_sha256(source_basis_sha256, "original source-native basis")
        require_sha256(producer, "native continuation producer")
        require_sha256(arithmetic, "native continuation binary64 arithmetic")
        if (matrix.ndim != 2 or not min(matrix.shape) or terminal.shape != (matrix.shape[1],)
                or array_digest(matrix) != source_basis_sha256):
            raise DictionaryArtifactError("source_native_matrix_or_terminal_binding_differs")
        if (isinstance(sample_rate_hz, (bool, np.bool_))
                or not isinstance(sample_rate_hz, (int, float, np.integer, np.floating))
                or not math.isfinite(float(sample_rate_hz)) or sample_rate_hz <= 0):
            raise DictionaryArtifactError("invalid_native_sampling_rate")
        rate = float(sample_rate_hz)
        duration = Fraction(len(matrix))/Fraction.from_float(rate)
        spec = NativeAnchoredDictionarySpec(_rounded(duration), matrix.shape[1])
        metadata = {"schema": NATIVE_DICTIONARY_SCHEMA, "program": NATIVE_PROGRAM,
            "implementation_sha256": producer, "arithmetic_source_sha256": arithmetic,
            "spec": spec.record(), "sample_rate_hz": rate,
            "source_basis_sha256": source_basis_sha256,
            "native_matrix_sha256": array_digest(matrix), "terminal_values_sha256": array_digest(terminal),
            "support_samples": [0, len(matrix)], "duration_seconds_exact_ratio": [duration.numerator, duration.denominator],
            "coefficient_chart": "original_source_actual_native_columns",
            "native_evaluation": "direct_original_stored_row_copy",
            "off_grid_evaluation": "exact_rational_linear_then_one_binary64_round",
            "terminal_knot": "explicit_stored_row_at_exact_L",
            "analytic_support": "closed_anchor_interval_zero_extension",
            "causal_support": "half_open_0_L_terminal_excluded",
            "function_role": "new_declared_continuation_of_source_native_arrays"}
        return cls._install(matrix, terminal, spec, metadata)

    @classmethod
    def _install(cls, matrix, terminal, spec, metadata):
        result = object.__new__(cls)
        for name, value in (("matrix", frozen_array(matrix)), ("terminal_values", frozen_array(terminal)),
                ("sample_rate_hz", metadata["sample_rate_hz"]), ("spec", spec),
                ("_metadata", freeze_metadata(metadata))):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_identity", metadata_digest(result._record()))
        return result

    @property
    def actual_seconds(self):
        return self.spec.response_seconds

    @property
    def identity_digest(self):
        return self._identity

    def metadata(self):
        return plain_metadata(self._metadata)

    def _record(self):
        return {**self.metadata(), "matrix": self.matrix.tolist(), "terminal_values": self.terminal_values.tolist()}

    def digest(self):
        if metadata_digest(self._record()) != self._identity:
            raise DictionaryArtifactError("immutable_source_native_dictionary_changed")
        return self._identity

    def record(self):
        self.digest()
        return self._record()

    @classmethod
    def from_record(cls, record):
        if (not isinstance(record, Mapping) or record.get("schema") != NATIVE_DICTIONARY_SCHEMA
                or record.get("program") != NATIVE_PROGRAM):
            raise DictionaryArtifactError("unsupported_source_native_dictionary_schema")
        expected = cls._build(record["matrix"], sample_rate_hz=record["sample_rate_hz"],
            source_basis_sha256=record["source_basis_sha256"], terminal_values=record["terminal_values"],
            producer=record["implementation_sha256"], arithmetic=record["arithmetic_source_sha256"])
        if metadata_digest(record) != expected.identity_digest:
            raise DictionaryArtifactError("saved_source_native_dictionary_relation_differs")
        metadata = {key: value for key, value in record.items() if key not in {"matrix", "terminal_values"}}
        return cls._install(record["matrix"], record["terminal_values"], expected.spec, metadata)

    def evaluate(self, query, *, coordinate="seconds", support="analytic"):
        return self._evaluate(query, coordinate=coordinate, support=support, consumer=IMPLEMENTATION_SHA256)

    def _evaluate(self, query, *, coordinate, support, consumer):
        self.digest()
        require_sha256(consumer, "native continuation evaluator")
        if coordinate not in {"seconds", "sample"} or support not in {"analytic", "half_open"}:
            raise DictionaryArtifactError("unknown_query_coordinate_or_support")
        shape, ratios = exact_query(query)
        rate = Fraction.from_float(self.sample_rate_hz)
        samples = tuple(value*rate if coordinate == "seconds" else value for value in ratios)
        length, width = self.matrix.shape
        values, errors = np.zeros((len(samples), width)), np.zeros((len(samples), width))
        mask = np.array([0 <= value < length for value in samples], dtype=bool).reshape(shape)
        row_cache = {}
        for index, site in enumerate(samples):
            if (support == "half_open" and not 0 <= site < length) or not 0 <= site <= length:
                continue
            if site.denominator == 1:
                values[index] = self.matrix[int(site)] if site < length else self.terminal_values
                continue
            if site not in row_cache:
                lower = site.numerator//site.denominator
                fraction = site-lower
                right = self.matrix[lower+1] if lower+1 < length else self.terminal_values
                row, bound = [], []
                for left, next_value in zip(self.matrix[lower], right, strict=True):
                    exact = (1-fraction)*Fraction.from_float(float(left))+fraction*Fraction.from_float(float(next_value))
                    rounded = _rounded(exact)
                    row.append(rounded)
                    bound.append(_error_upper(abs(Fraction.from_float(rounded)-exact)))
                row_cache[site] = row, bound
            values[index], errors[index] = row_cache[site]
        arrays = {"matrix": frozen_array(values.reshape((*shape, width))),
            "error": frozen_array(errors.reshape((*shape, width))),
            "support_mask": np.frombuffer(mask.tobytes(), dtype=bool).reshape(shape)}
        metadata = {"schema": NATIVE_EVALUATION_SCHEMA, "program": NATIVE_PROGRAM,
            "dictionary_identity_digest": self.identity_digest,
            "dictionary_producer_sha256": self._metadata["implementation_sha256"],
            "consumer_implementation_sha256": consumer, "query_shape": list(shape),
            "query_coordinate": coordinate, "query_exact_ratio": [[value.numerator, value.denominator] for value in ratios],
            "sample_coordinates_exact_ratio": [[value.numerator, value.denominator] for value in samples],
            "support_program": support, "support_exact_sample_interval": [0, length],
            "rounding_error_role": "absolute_exact_interpolation_to_stored_binary64_difference",
            "state_sha256": array_digest(arrays["matrix"], arrays["error"], arrays["support_mask"])}
        return NativeResponseEvaluation._install(self, arrays, metadata)

    def save(self, path):
        from research.execution import atomic_write_json

        atomic_write_json(Path(path), self.record())
        return Path(path)

    @classmethod
    def load(cls, path):
        return cls.from_record(json.loads(Path(path).read_bytes()))


@dataclass(frozen=True, init=False)
class NativeResponseEvaluation:
    dictionary: NativeResponseDictionary
    matrix: np.ndarray
    error: np.ndarray
    support_mask: np.ndarray
    _metadata: dict
    _identity: str

    @classmethod
    def _install(cls, dictionary, arrays, metadata):
        result = object.__new__(cls)
        object.__setattr__(result, "dictionary", dictionary)
        for name, value in arrays.items():
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_metadata", freeze_metadata(metadata))
        object.__setattr__(result, "_identity", metadata_digest(result._record()))
        return result

    def _record(self):
        return {**self.metadata(), "matrix": self.matrix.tolist(), "error": self.error.tolist(),
            "support_mask": self.support_mask.tolist()}

    @property
    def identity_digest(self):
        return self._identity

    def metadata(self):
        return plain_metadata(self._metadata)

    def digest(self):
        if (self.dictionary.digest() != self._metadata["dictionary_identity_digest"]
                or metadata_digest(self._record()) != self._identity):
            raise DictionaryArtifactError("immutable_source_native_evaluation_changed")
        return self._identity

    def record(self):
        self.digest()
        return self._record()

    @classmethod
    def from_record(cls, record, *, dictionary):
        if (not isinstance(dictionary, NativeResponseDictionary) or not isinstance(record, Mapping)
                or record.get("schema") != NATIVE_EVALUATION_SCHEMA):
            raise DictionaryArtifactError("evaluation_requires_original_source_native_dictionary")
        query = np.array([Fraction(*value) for value in record["query_exact_ratio"]], dtype=object)
        query = query.reshape(record["query_shape"])
        expected = dictionary._evaluate(query, coordinate=record["query_coordinate"], support=record["support_program"],
            consumer=record["consumer_implementation_sha256"])
        if metadata_digest(record) != expected.identity_digest:
            raise DictionaryArtifactError("saved_source_native_evaluation_relation_differs")
        arrays = {name: frozen_array(record[name]) for name in ("matrix", "error")}
        mask = np.asarray(record["support_mask"], dtype=bool)
        arrays["support_mask"] = np.frombuffer(mask.tobytes(), dtype=bool).reshape(mask.shape)
        metadata = {key: value for key, value in record.items() if key not in arrays}
        return cls._install(dictionary, arrays, metadata)

    def save(self, path):
        from research.execution import atomic_write_json

        atomic_write_json(Path(path), self.record())
        return Path(path)

    @classmethod
    def load(cls, path, *, dictionary):
        return cls.from_record(json.loads(Path(path).read_bytes()), dictionary=dictionary)
