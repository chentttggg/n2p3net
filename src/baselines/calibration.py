"""Inductive binary threshold and log-likelihood-ratio calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score, roc_curve


@dataclass(frozen=True)
class LogitCalibration:
    threshold: float
    threshold_balanced_acc: float
    llr_slope: float
    llr_intercept: float
    source: str
    n_samples: int
    mode: str = "monotone_platt_llr"
    temperature: float | None = None
    offset: float | None = None

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
    pos_weight: float | None = None,
    train_prior: float | None = None,
) -> LogitCalibration:
    """Fit an order-preserving score-to-LLR map on non-test observations.

    Weighted-cross-entropy models have a known analytic offset, so only a
    strictly positive temperature is fitted. Other estimators use a monotone
    Platt map whose slope is parameterized as ``exp(log_slope)``. A noisy inner
    validation set can therefore attenuate an unhelpful score toward zero, but
    can never reverse candidate rankings.
    """

    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    threshold, bacc = learn_balanced_threshold(logits, y)
    if (pos_weight is None) != (train_prior is None):
        raise ValueError("pos_weight and train_prior must be supplied together.")
    if pos_weight is not None and train_prior is not None:
        weighted = fit_weighted_logit_temperature(
            logits,
            y,
            pos_weight=float(pos_weight),
            train_prior=float(train_prior),
            source=source,
        )
        slope = 1.0 / weighted.temperature
        intercept = -weighted.offset / weighted.temperature
        return LogitCalibration(
            threshold=threshold,
            threshold_balanced_acc=bacc,
            llr_slope=float(slope),
            llr_intercept=float(intercept),
            source=str(source),
            n_samples=int(len(y)),
            mode="weighted_ce_positive_temperature_llr",
            temperature=float(weighted.temperature),
            offset=float(weighted.offset),
        )

    prior = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    prior_log_odds = float(np.log(prior / (1.0 - prior)))

    score_center = float(np.mean(logits))
    score_scale = max(float(np.std(logits)), 1e-6)
    normalized_logits = (logits - score_center) / score_scale

    def objective(parameters: np.ndarray) -> float:
        normalized_slope = float(np.exp(parameters[0]))
        posterior_log_odds = (
            normalized_slope * normalized_logits + float(parameters[1])
        )
        nll = np.mean(
            np.logaddexp(0.0, posterior_log_odds) - y * posterior_log_odds
        )
        # Match the former C=1 Platt fit's finite-sample slope regularization
        # while leaving the intercept unpenalized.
        return float(nll + 0.5 * normalized_slope**2 / len(y))

    optimum = minimize(
        objective,
        x0=np.asarray([0.0, prior_log_odds], dtype=float),
        method="L-BFGS-B",
        bounds=((np.log(1e-6), np.log(1e3)), (-50.0, 50.0)),
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError("monotone Platt calibration optimization failed.")
    slope = float(np.exp(optimum.x[0]) / score_scale)
    posterior_intercept = float(optimum.x[1] - slope * score_center)
    llr_intercept = posterior_intercept - prior_log_odds
    temperature = 1.0 / slope
    offset = -llr_intercept / slope
    return LogitCalibration(
        threshold=threshold,
        threshold_balanced_acc=bacc,
        llr_slope=slope,
        llr_intercept=llr_intercept,
        source=str(source),
        n_samples=int(len(y)),
        mode="monotone_platt_llr",
        temperature=float(temperature),
        offset=float(offset),
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
