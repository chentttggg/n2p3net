"""A fixed finite SOS operator and its exact two-ended state decomposition.

Padding states computed from observed samples are deterministic. Their basis
does not imply an extra random covariance; a random external-state model must
be separately identified. Dense matrices are bounded diagnostic exports only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt

from data.preprocess import filter_bidirectional_segment
from research.contracts import semantic_sha256


@dataclass(frozen=True)
class FiniteFilterOperator:
    sos: tuple[tuple[float, ...], ...]
    padding_limit: int

    def __post_init__(self):
        sos = np.asarray(self.sos, dtype=float)
        if (sos.ndim != 2 or sos.shape[1] != 6 or not len(sos) or not np.isfinite(sos).all()
                or not np.all(sos[:, 3] == 1)):
            raise ValueError("finite operator requires finite normalized SOS sections")
        if type(self.padding_limit) is not int or self.padding_limit < 0:
            raise ValueError("finite operator padding limit must be nonnegative")
        if any(np.max(np.abs(np.roots(row[3:]))) >= 1 for row in sos):
            raise ValueError("finite operator requires stable sections")
        object.__setattr__(self, "sos", tuple(tuple(float(v) for v in row) for row in sos))

    def record(self):
        return {"schema": "n2p3_finite_filter_operator/1", "sos": [list(row) for row in self.sos],
                "padding_limit": self.padding_limit, "edge_rule": "observed_odd_reflection",
                "state_role": "deterministic_function_of_observed_segment"}

    @classmethod
    def from_record(cls, record):
        if (not isinstance(record, dict) or record.get("schema") != "n2p3_finite_filter_operator/1"
                or record.get("edge_rule") != "observed_odd_reflection"
                or record.get("state_role") != "deterministic_function_of_observed_segment"):
            raise ValueError("finite filter operator record has an unknown edge or state contract")
        return cls(tuple(tuple(float(value) for value in row) for row in record["sos"]), record["padding_limit"])

    @property
    def digest(self):
        return semantic_sha256(self.record())

    @staticmethod
    def _values(values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("operator input must be finite samples by columns")
        return values

    def apply(self, values):
        values = self._values(values)
        return filter_bidirectional_segment(np.asarray(self.sos), values.T,
                                            padding_limit=self.padding_limit).data.T

    def zero_state(self, values):
        """Q x = H.T H x on this exact finite grid, without padding states."""
        values = self._values(values)
        sos = np.asarray(self.sos)
        forward = sosfilt(sos, values, axis=0)
        return sosfilt(sos, forward[::-1], axis=0)[::-1]

    def state_basis(self, n_samples):
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("state basis requires positive integer samples")
        order = 2 * len(self.sos)
        states = np.eye(order).reshape(order, len(self.sos), 2).transpose(1, 0, 2).copy()
        response = sosfilt(np.asarray(self.sos), np.zeros((order, n_samples)), axis=1, zi=states)[0].T
        left = sosfilt(np.asarray(self.sos), response[::-1], axis=0)[::-1]
        return np.column_stack((left, response[::-1]))

    def decompose(self, values):
        """Return Qx, B, actual a such that F(x)=Qx+B a; a is not an oracle."""
        values = self._values(values)
        executed = filter_bidirectional_segment(np.asarray(self.sos), values.T,
                                               padding_limit=self.padding_limit)
        states = np.concatenate((executed.left_state.transpose(0, 2, 1).reshape(-1, values.shape[1]),
                                 executed.right_state.transpose(0, 2, 1).reshape(-1, values.shape[1])))
        return self.zero_state(values), self.state_basis(len(values)), states

    def dense_matrix(self, n_samples, *, byte_budget=8 * 1024 * 1024):
        if type(n_samples) is not int or n_samples < 1 or 8 * n_samples**2 > byte_budget:
            raise ValueError("dense operator exceeds the explicit small diagnostic budget")
        return self.apply(np.eye(n_samples))
