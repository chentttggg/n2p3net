"""Frozen ERP classifiers behind one fit/predict/numeric-state interface.

ToeplitzLDA follows Sosulski et al. (JNE 2022): pooled class residuals,
standardized shrinkage, block lag averaging, then linear taper. It requires
un-baselined equally spaced input; channel-prime vectorization is internal.
The shrinkage implementation is adapted from jsosulski/toeplitzlda (BSD);
see structured_erp.LICENSE. No test observations enter fitted covariance.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler

from baselines.classic import Baseline
from baselines.temporal_transfer import TemporalBasis

ERP_FAMILIES = frozenset({"xdawn_erp", "toeplitz_lda", "pooled_lda", "temporal_lda"})
ERP_SCHEMA = "n2p3_structured_erp/1"


def _shrinkage(residuals):
    """Official pooled-residual shrinkage, including n-1 normalization."""
    scaler = StandardScaler().fit(residuals)
    Z = scaler.transform(residuals).T
    Z -= Z.mean(axis=1, keepdims=True)
    p, n = Z.shape
    S = Z @ Z.T
    target = np.trace(S) / p * np.eye(p)
    V = ((Z * Z) @ (Z * Z).T - S * S / n) / (n - 1)
    denominator = np.sum((S - target) ** 2)
    gamma = 1.0 if denominator <= 0 else float(np.clip(n * np.sum(V) / denominator, 0, 1))
    covariance = (gamma * target + (1 - gamma) * S) / (n - 1)
    return covariance * scaler.scale_[:, None] * scaler.scale_[None, :], gamma


def _tapered_toeplitz(covariance, channels, times):
    # Biased lag sum / times equals diagonal mean times (1-lag/times).
    # This is the covariance of a zero-extended stationary sliding process:
    # unlike untapered diagonal averaging it preserves positive semidefiniteness.
    blocks = covariance.reshape(times, channels, times, channels)
    output = np.empty_like(blocks)
    for lag in range(times):
        value = sum(blocks[t, :, t + lag, :] for t in range(times - lag)) / times
        for t in range(times - lag):
            output[t, :, t + lag, :] = value
            output[t + lag, :, t, :] = value.T
    return output.reshape(covariance.shape)


class StructuredERP(Baseline):
    """Target-only ERP family with an explicit, pickle-free fitted schema."""

    def __init__(self, family, *, bin_width=1, baseline_mode="none", nfilter=4, temporal_basis=None):
        if family not in ERP_FAMILIES:
            raise ValueError("unknown ERP family")
        if isinstance(bin_width, bool) or not isinstance(bin_width, int) or bin_width < 1:
            raise ValueError("ERP bin_width must be a positive integer")
        if family == "toeplitz_lda" and baseline_mode != "none":
            raise ValueError("standard ToeplitzLDA requires baseline_mode=none")
        if isinstance(nfilter, bool) or not isinstance(nfilter, int) or nfilter < 1:
            raise ValueError("ERP nfilter must be a positive integer")
        self.family, self.bin_width, self.baseline_mode, self.nfilter = family, bin_width, baseline_mode, nfilter
        if family == "temporal_lda":
            if not isinstance(temporal_basis, TemporalBasis) or bin_width != 1:
                raise ValueError("temporal LDA requires a frozen TemporalBasis and bin_width=1")
            self.temporal_basis = TemporalBasis.from_record(temporal_basis.record())
            if self.temporal_basis.record()["preprocessing"]["baseline_mode"] != baseline_mode:
                raise ValueError("temporal basis and ERP classifier baseline contracts differ")
        elif temporal_basis is not None:
            raise ValueError("frozen temporal basis belongs only to temporal_lda")
        else:
            self.temporal_basis = None
        self._state = None

    def _input(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[1] < 1 or X.shape[2] < self.bin_width or not np.isfinite(X).all():
            raise ValueError("ERP input must be finite (N,C,T) with complete observations")
        if self._state is not None and X.shape[1:] != self.input_shape_:
            raise ValueError("ERP channel/time geometry differs from fitted input")
        bins = X.shape[2] // self.bin_width
        # Every retained interval has identical width. The unused tail is fixed
        # and recorded; it is never averaged as a shorter final bin.
        return X[:, :, :bins * self.bin_width].reshape(len(X), X.shape[1], bins, self.bin_width).mean(axis=-1)

    def fit(self, X, y):
        self._state = None
        A = self._input(X)
        y = np.asarray(y)
        if y.shape != (len(A),) or set(np.unique(y)) != {0, 1} or min(np.bincount(y.astype(int))) < 2:
            raise ValueError("ERP fitting requires at least two trials in each binary class")
        self.input_shape_ = tuple(np.asarray(X).shape[1:])
        if self.temporal_basis is not None:
            A = self.temporal_basis.project(A)
        if self.family == "xdawn_erp":
            from baselines.riemann import XdawnRiemann
            model = XdawnRiemann(nfilter=self.nfilter).fit(A, y)
            self.effective_nfilter_ = model.effective_nfilter_
            self.observed_rank_ = model.observed_rank_
            self._state = {"filters": model.xd_.filters_, "prototypes": model.prototypes_,
                           "reference": model.ts_.reference_, "coefficient": model.lda_.coef_[0],
                           "intercept": model.lda_.intercept_[0]}
        else:
            features = A.transpose(0, 2, 1).reshape(len(A), -1)
            means = np.stack([features[y == label].mean(axis=0) for label in (0, 1)])
            residuals = features - means[y.astype(int)]
            covariance, self.shrinkage_ = _shrinkage(residuals)
            if self.family == "toeplitz_lda":
                covariance = _tapered_toeplitz(covariance, A.shape[1], A.shape[2])
            try:
                factor = cho_factor(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("ERP covariance is not positive definite; inspect constant/missing channels") from exc
            coefficient = cho_solve(factor, means[1] - means[0])
            prior_log_odds = np.log(np.mean(y == 1) / np.mean(y == 0))
            intercept = -0.5 * (means[1] + means[0]) @ coefficient + prior_log_odds
            self.covariance_ = covariance
            self._state = {"coefficient": coefficient, "intercept": intercept}
            if self.temporal_basis is not None:
                self._state["temporal_basis"] = self.temporal_basis.matrix
        self._state = {key: np.asarray(value, dtype=np.float64).copy() for key, value in self._state.items()}
        return self

    def predict_logit(self, X):
        if self._state is None:
            raise RuntimeError("ERP classifier must be fitted first")
        A = self._input(X)
        if self.temporal_basis is not None:
            A = self.temporal_basis.project(A)
        if not len(A):
            return np.empty(0, dtype=np.float64)
        if self.family == "xdawn_erp":
            from pyriemann.tangentspace import TangentSpace

            from baselines.riemann import _oas_covariances_vectorized
            transformed = np.einsum("vc,nct->nvt", self._state["filters"], A)
            P = self._state["prototypes"]
            augmented = np.concatenate((np.broadcast_to(P, (len(A), *P.shape)), transformed), axis=1)
            tangent = TangentSpace(metric="riemann")
            tangent.reference_ = self._state["reference"]
            features = tangent.transform(_oas_covariances_vectorized(augmented))
        else:
            features = A.transpose(0, 2, 1).reshape(len(A), -1)
        return features @ self._state["coefficient"] + self._state["intercept"]

    def export_state(self):
        if self._state is None:
            raise RuntimeError("ERP classifier must be fitted first")
        record = {"schema": ERP_SCHEMA, "family": self.family, "bin_width": self.bin_width,
                  "baseline_mode": self.baseline_mode, "nfilter": self.nfilter,
                  "input_shape": list(self.input_shape_), "feature_order": "channel_prime",
                  "discarded_tail_samples": self.input_shape_[1] % self.bin_width}
        if self.temporal_basis is not None:
            record["temporal_basis"] = self.temporal_basis.record()
        if self.family == "xdawn_erp":
            record.update(effective_nfilter=self.effective_nfilter_, observed_rank=self.observed_rank_)
        return record, {key: value.copy() for key, value in self._state.items()}

    @classmethod
    def from_state(cls, record, state):
        if record.get("schema") != ERP_SCHEMA or record.get("feature_order") != "channel_prime":
            raise ValueError("unsupported ERP numeric artifact schema or order")
        temporal = (None if "temporal_basis" not in record else TemporalBasis.from_record(record["temporal_basis"]))
        model = cls(record["family"], bin_width=record["bin_width"], baseline_mode=record["baseline_mode"], nfilter=record["nfilter"], temporal_basis=temporal)
        model.input_shape_ = tuple(record["input_shape"])
        if len(model.input_shape_) != 2 or any(type(v) is not int or v < 1 for v in model.input_shape_):
            raise ValueError("ERP state input geometry is invalid")
        c, t = model.input_shape_
        b = t // model.bin_width
        if b < 1 or record.get("discarded_tail_samples") != t % model.bin_width:
            raise ValueError("ERP state binning differs")
        expected = {"coefficient", "intercept"}
        if model.family == "xdawn_erp":
            expected |= {"filters", "prototypes", "reference"}
        if temporal is not None:
            expected.add("temporal_basis")
        if set(state) != expected:
            raise ValueError("ERP state keys differ from the frozen family")
        model._state = {key: np.asarray(value, dtype=np.float64).copy() for key, value in state.items()}
        if any(not np.isfinite(value).all() for value in model._state.values()):
            raise ValueError("ERP state contains nonfinite values")
        if model.family == "xdawn_erp":
            rank, effective = record.get("observed_rank"), record.get("effective_nfilter")
            if type(rank) is not int or not 1 <= rank <= c or type(effective) is not int or effective != min(model.nfilter, rank):
                raise ValueError("ERP artifact requires actual observed rank and effective filter count")
            model.observed_rank_, model.effective_nfilter_ = rank, effective
            v = 2 * effective
            shapes = {"filters": (v, c), "prototypes": (v, b), "reference": (2*v, 2*v),
                      "coefficient": ((2*v)*(2*v+1)//2,), "intercept": ()}
            reference = model._state["reference"]
            if reference.shape == shapes["reference"] and (
                not np.allclose(reference, reference.T) or np.linalg.eigvalsh(reference).min() <= 0
            ):
                raise ValueError("ERP tangent reference must be symmetric positive definite")
        else:
            shapes = {"coefficient": (c*(b if temporal is None else temporal.matrix.shape[1]),), "intercept": ()}
            if temporal is not None:
                shapes["temporal_basis"] = temporal.matrix.shape
                if temporal.matrix.shape[0] != t or not np.array_equal(model._state["temporal_basis"], temporal.matrix):
                    raise ValueError("temporal basis state differs from its declared artifact")
        if any(model._state[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("ERP state shape differs from the frozen geometry")
        return model
