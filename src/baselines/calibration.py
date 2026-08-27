"""Inductive binary threshold and log-likelihood-ratio calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve


@dataclass(frozen=True)
class LogitCalibration:
    threshold: float
    threshold_balanced_acc: float
    llr_slope: float
    llr_intercept: float
    source: str
    n_samples: int

    def to_llr(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        return self.llr_slope * values + self.llr_intercept


@dataclass(frozen=True)
class WeightedLogitTemperatureCalibration:
    """Explicit weighted-BCE offset correction with validation temperature."""

    pos_weight: float
    train_prior: float
    temperature: float
    validation_nll: float
    source: str
    n_samples: int

    @property
    def offset(self) -> float:
        prior_log_odds = np.log(self.train_prior / (1.0 - self.train_prior))
        return float(np.log(self.pos_weight) + prior_log_odds)

    def to_llr(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        return (values - self.offset) / self.temperature


def fit_weighted_logit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    *,
    pos_weight: float,
    train_prior: float,
    source: str,
) -> WeightedLogitTemperatureCalibration:
    """Fit temperature on group-disjoint validation after exact offset removal."""

    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(logits) == 0 or len(logits) != len(y):
        raise ValueError("temperature calibration logits/y must be non-empty and aligned.")
    if not np.isfinite(logits).all() or not np.isfinite(y).all():
        raise ValueError("temperature calibration inputs contain NaN/inf.")
    if len(np.unique(y)) != 2:
        raise ValueError("temperature calibration requires both binary classes.")
    if pos_weight <= 0.0 or not 0.0 < train_prior < 1.0:
        raise ValueError("pos_weight/train_prior are invalid for weighted-logit correction.")
    prior_log_odds = float(np.log(train_prior / (1.0 - train_prior)))
    offset = float(np.log(pos_weight) + prior_log_odds)
    corrected = logits - offset

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        posterior_log_odds = corrected / temperature + prior_log_odds
        return float(np.mean(np.logaddexp(0.0, posterior_log_odds) - y * posterior_log_odds))

    optimum = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError("inner-validation temperature optimization failed.")
    return WeightedLogitTemperatureCalibration(
        pos_weight=float(pos_weight),
        train_prior=float(train_prior),
        temperature=float(np.exp(optimum.x)),
        validation_nll=float(optimum.fun),
        source=str(source),
        n_samples=int(len(y)),
    )


def learn_balanced_threshold(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Select a threshold maximizing Youden's J on non-test observations."""

    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    if len(logits) != len(y) or len(logits) == 0:
        raise ValueError("calibration logits/y must be non-empty and aligned.")
    if not np.isfinite(logits).all():
        raise ValueError("calibration logits contain NaN/inf.")
    if len(np.unique(y)) != 2:
        raise ValueError("threshold calibration requires both binary classes.")

    fpr, tpr, thresholds = roc_curve(y, logits, drop_intermediate=False)
    objective = tpr - fpr
    finite = np.isfinite(thresholds)
    best = np.flatnonzero(finite & np.isclose(objective, np.max(objective[finite])))
    # A deterministic conservative tie break: require the higher score when J
    # is identical. This avoids dependence on sklearn's internal threshold order.
    threshold = float(np.max(thresholds[best]))
    score = float(balanced_accuracy_score(y, logits >= threshold))
    return threshold, score


def fit_logit_calibration(
    logits: np.ndarray,
    y: np.ndarray,
    *,
    source: str,
) -> LogitCalibration:
    """Fit threshold and a prior-corrected Platt map from score to LLR."""

    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    threshold, bacc = learn_balanced_threshold(logits, y)

    # Platt's decision function estimates posterior log-odds. Subtracting the
    # empirical class-prior log-odds yields an evidence LLR suitable for sums.
    platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
    platt.fit(logits[:, None], y)
    prior = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    prior_log_odds = float(np.log(prior / (1.0 - prior)))
    return LogitCalibration(
        threshold=threshold,
        threshold_balanced_acc=bacc,
        llr_slope=float(platt.coef_[0, 0]),
        llr_intercept=float(platt.intercept_[0] - prior_log_odds),
        source=str(source),
        n_samples=int(len(y)),
    )


def calibration_data_from_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Require group-disjoint validation scores exposed by trainable adapters."""

    logits = getattr(model, "calibration_logits_", None)
    labels = getattr(model, "calibration_labels_", None)
    source = getattr(model, "calibration_source_", None)
    if logits is not None and labels is not None and len(np.unique(labels)) == 2:
        return np.asarray(logits), np.asarray(labels), str(source or "model_validation")
    raise ValueError(
        "Subject-disjoint validation calibration is unavailable; "
        "training-set resubstitution calibration is forbidden."
    )
