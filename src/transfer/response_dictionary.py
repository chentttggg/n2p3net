"""Physical ERP dictionaries: owned specs and immutable portable compiled state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transfer.dictionary_artifact import (
    ARRAY_KEYS,
    DICTIONARY_SCHEMA,
    DictionaryArtifactError,
    admit_record,
    compile_state,
)
from transfer.noise_model import freeze_metadata, metadata_digest

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


@dataclass(frozen=True)
class ResponseDictionary:
    spec: ResponseDictionarySpec
    sample_rate_hz: float
    times: np.ndarray = field(init=False, repr=False, compare=False)
    ambient_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    matrix: np.ndarray = field(init=False, repr=False, compare=False)
    endpoint_rows: np.ndarray = field(init=False, repr=False, compare=False)
    constraint_map: np.ndarray = field(init=False, repr=False, compare=False)
    _metadata: dict = field(init=False, repr=False, compare=False)
    _integrity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.spec, ResponseDictionarySpec):
            raise DictionaryArtifactError("unowned_physical_spec")
        arrays, metadata = compile_state(self.spec.record(), self.sample_rate_hz,
                                         implementation_sha256=IMPLEMENTATION_SHA256)
        self._install(arrays, metadata)

    def _install(self, arrays, metadata):
        for key in ARRAY_KEYS:
            object.__setattr__(self, key, arrays[key])
        object.__setattr__(self, "sample_rate_hz", metadata["sample_rate_hz"])
        object.__setattr__(self, "_metadata", freeze_metadata(metadata))
        object.__setattr__(self, "_integrity", metadata_digest(self._record()))

    @classmethod
    def compile(cls, spec, *, sample_rate_hz):
        return cls(spec, sample_rate_hz)

    @property
    def actual_seconds(self):
        return self._metadata["actual_seconds"]

    def _arrays(self):
        return {key: getattr(self, key) for key in ARRAY_KEYS}

    def _record(self):
        return {**self.metadata(), **{key: value.tolist() for key, value in self._arrays().items()}}

    @property
    def identity_digest(self):
        """Identity of state admitted once; internal consumers need no recompile/hash."""
        return self._integrity

    def metadata(self):
        """Retain the artifact's actual compiler/producer, including after load."""
        return json.loads(json.dumps(self._metadata, allow_nan=False))

    def digest(self):
        if (self.spec.record() != self._metadata["spec"]
                or self.sample_rate_hz != self._metadata["sample_rate_hz"]
                or metadata_digest(self._record()) != self._integrity):
            raise DictionaryArtifactError("immutable_numerical_state_changed")
        return self._integrity

    def record(self):
        self.digest()
        return self._record()

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or record.get("schema") != DICTIONARY_SCHEMA:
            raise DictionaryArtifactError("unsupported_dictionary_schema", field="schema")
        try:
            spec = ResponseDictionarySpec.from_record(record["spec"])
        except (KeyError, TypeError, ValueError) as error:
            raise DictionaryArtifactError("invalid_physical_spec", field="spec") from error
        arrays, metadata = admit_record(record, spec.record())
        result = object.__new__(cls)
        object.__setattr__(result, "spec", spec)
        result._install(arrays, metadata)
        return result
