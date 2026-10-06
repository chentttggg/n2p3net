"""Physical ERP function dictionaries, with explicit endpoint restrictions.

One owner generates the numeric basis and its physical grid. Hard endpoints
restrict the same ambient cosine space through an orthonormal nullspace map;
they do not silently substitute another family or add latent random effects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResponseDictionarySpec:
    family: str
    response_seconds: float
    n_basis: int
    endpoint: str = "free"

    def __post_init__(self):
        if self.family not in {"gaussian", "cosine"} or self.endpoint not in {"free", "hard_zero"}:
            raise ValueError("unsupported physical dictionary family or endpoint program")
        if (isinstance(self.response_seconds, bool) or not isinstance(self.response_seconds, (int, float))
                or not np.isfinite(self.response_seconds) or self.response_seconds <= 0
                or type(self.n_basis) is not int or self.n_basis < 1):
            raise ValueError("dictionary duration and ambient dimension must be positive finite values")
        object.__setattr__(self, "response_seconds", float(self.response_seconds))

    def record(self):
        return {"family": self.family, "response_seconds": self.response_seconds,
                "n_basis": self.n_basis, "endpoint": self.endpoint}

    @classmethod
    def from_record(cls, record):
        return cls(**record)


def _endpoint_nullspace(width):
    columns = []
    for parity in (0, 1):
        indices = np.arange(parity, width, 2)
        for j in range(1, len(indices)):
            value = np.zeros(width)
            denominator = np.sqrt(j*(j+1))
            value[indices[:j]] = 1/denominator
            value[indices[j]] = -j/denominator
            columns.append(value)
    return np.column_stack(columns)


@dataclass(frozen=True)
class ResponseDictionary:
    spec: ResponseDictionarySpec
    sample_rate_hz: float
    times: np.ndarray = field(init=False, repr=False, compare=False)
    ambient_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    matrix: np.ndarray = field(init=False, repr=False, compare=False)
    endpoint_rows: np.ndarray = field(init=False, repr=False, compare=False)
    constraint_map: np.ndarray = field(init=False, repr=False, compare=False)
    _generation: dict = field(init=False, repr=False, compare=False)
    _state_sha256: str = field(init=False, repr=False, compare=False)
    _integrity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.spec, ResponseDictionarySpec) or isinstance(self.sample_rate_hz, bool)
                or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0):
            raise ValueError("dictionary compilation needs an owned spec and native sampling rate")
        spec, rate = self.spec, float(self.sample_rate_hz)
        length = round(rate*spec.response_seconds)
        if length < 3 or spec.n_basis > length:
            raise ValueError("dictionary ambient dimension must fit its actual response sample support")
        grid = np.arange(length, dtype=float)
        if spec.family == "gaussian":
            if spec.endpoint != "free":
                raise ValueError("hard endpoint Gaussian is outside this dictionary slice")
            centers = np.linspace(0, length-1, spec.n_basis)
            width = max(1., length/max(spec.n_basis-1, 1))
            ambient = np.exp(-.5*((grid[:, None]-centers)/width)**2)
            endpoint = np.exp(-.5*((np.array([0., length])[:, None]-centers)/width)**2)
            generation = {"centers_seconds": (centers/rate).tolist(), "width_seconds": width/rate,
                          "evaluation": "native_sample_coordinate_equivalent_to_physical_seconds"}
        else:
            frequencies = np.arange(spec.n_basis)
            ambient = np.cos(np.pi*grid[:, None]*frequencies/length)
            endpoint = np.stack((np.ones(spec.n_basis), (-1.)**frequencies))
            generation = {"angular_frequencies_per_second": (np.pi*frequencies/(length/rate)).tolist(),
                          "evaluation": "cosine_on_actual_half_open_physical_grid"}
        if spec.endpoint == "hard_zero":
            if spec.n_basis < 3:
                raise ValueError("hard endpoint constraints leave an empty function space")
            embedding = _endpoint_nullspace(spec.n_basis)
        else:
            embedding = np.eye(spec.n_basis)
        matrix = ambient@embedding if spec.endpoint == "hard_zero" else ambient
        for name, value in (("times", grid/rate), ("ambient_matrix", ambient), ("matrix", matrix),
                            ("endpoint_rows", endpoint), ("constraint_map", embedding)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "_generation", freeze_metadata(generation))
        object.__setattr__(self, "_state_sha256", array_digest(*self._arrays().values()))
        object.__setattr__(self, "_integrity", metadata_digest(self._record()))

    @classmethod
    def compile(cls, spec, *, sample_rate_hz):
        return cls(spec, sample_rate_hz)

    @property
    def actual_seconds(self):
        return len(self.times)/self.sample_rate_hz

    def _arrays(self):
        return {"times": self.times, "ambient_matrix": self.ambient_matrix, "matrix": self.matrix,
                "endpoint_rows": self.endpoint_rows, "constraint_map": self.constraint_map}

    def _record(self):
        return {**self.metadata(), **{key: value.tolist() for key, value in self._arrays().items()}}

    @property
    def identity_digest(self):
        """Already validated immutable identity for trusted internal consumers."""
        return self._integrity

    def metadata(self):
        return {"schema": "n2p3_response_dictionary/1", "spec": self.spec.record(), "sample_rate_hz": self.sample_rate_hz,
                "actual_seconds": self.actual_seconds, "support_samples": [0, len(self.times)],
                "last_sample_seconds": float(self.times[-1]), "endpoint_evaluation": "u=0_and_actual_L_over_sample_rate",
                "column_units": "dimensionless", "coefficient_units": "volts", "generation": self._generation,
                "ambient_dimension": self.spec.n_basis, "function_dimension": self.matrix.shape[1],
                "implementation_sha256": IMPLEMENTATION_SHA256,
                "state_sha256": self._state_sha256}

    def digest(self):
        if metadata_digest(self._record()) != self._integrity:
            raise ValueError("immutable response dictionary numerical state changed")
        return self._integrity

    def record(self):
        self.digest()
        return json.loads(json.dumps(self._record(), allow_nan=False))

    @classmethod
    def from_record(cls, record):
        result = cls.compile(ResponseDictionarySpec.from_record(record["spec"]), sample_rate_hz=record["sample_rate_hz"])
        if metadata_digest(result._record()) != metadata_digest(record):
            raise ValueError("dictionary numerical record, grid, constraints or producer differ from its actual function program")
        return result
