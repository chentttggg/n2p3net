"""Transparent Closure stage for the P300 PEC audit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import MetricSpec
from .types import AuditInputError, FeatureTable, P300AuditData, P300Split


@dataclass(frozen=True)
class ClosureConfig:
    C: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    denominator_epsilon: float = 1e-12
    random_seed: int = 4311

    def __post_init__(self) -> None:
        if self.C <= 0 or self.max_iter < 100 or self.denominator_epsilon <= 0:
            raise AuditInputError(
                "Closure C and denominator_epsilon must be positive; max_iter must be >=100."
            )
        if self.class_weight not in (None, "balanced"):
            raise AuditInputError("Closure class_weight must be None or 'balanced'.")


@dataclass(frozen=True)
class ClosureResult:
    metric: str
    feature_count: int
    selected_features: tuple[str, ...]
    selected_families: tuple[str, ...]
    transparent_metric: float
    random_metric: float
    model_metric: float
    closure_ratio: float
    defined: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class P300ClosureAuditor:
    """Train transparent logistic classifiers on confirmed P300 features."""

    def __init__(self, config: ClosureConfig | None = None):
        self.config = config or ClosureConfig()

    def run(
        self,
        data: P300AuditData,
        split: P300Split,
        features: FeatureTable,
        selected_features: Sequence[str],
        model_test_score: np.ndarray,
        *,
        metrics: Sequence[MetricSpec] | None = None,
    ) -> tuple[ClosureResult, ...]:
        split.validate_against(data)
        if features.n_trials != data.n_trials:
            raise AuditInputError("feature rows must match the audit data rows.")
        model_score = np.asarray(model_test_score, dtype=float).reshape(-1)
        if model_score.shape[0] != split.test.shape[0] or not np.all(np.isfinite(model_score)):
            raise AuditInputError("model_test_score must be finite and aligned to split.test.")
        selected = tuple(selected_features)
        if len(set(selected)) != len(selected):
            raise AuditInputError("selected_features must not contain duplicates.")
        indices = []
        for name in selected:
            if name not in features.names:
                raise AuditInputError(f"Unknown closure feature {name!r}.")
            indices.append(features.names.index(name))
        if not indices:
            raise AuditInputError("Closure requires at least one confirmed feature.")
        feature_matrix = features.values[:, indices]
        if not np.all(np.isfinite(feature_matrix)):
            raise AuditInputError("selected closure features contain NaN or infinite values.")
        train_features = feature_matrix[split.train]
        test_features = feature_matrix[split.test]
        train_y = data.target[split.train]
        if np.unique(train_y).size < 2:
            raise AuditInputError("Closure training split must contain both target classes.")
        transparent_score = self._fit_score(train_features, train_y, test_features)

        rng = np.random.default_rng(np.random.SeedSequence([self.config.random_seed, len(indices)]))
        random_features = rng.standard_normal(feature_matrix.shape)
        random_score = self._fit_score(
            random_features[split.train], train_y, random_features[split.test]
        )
        test_data = data.subset(split.test)
        selected_families = tuple(features.families[index] for index in indices)
        selected_metrics = (
            tuple(metrics)
            if metrics is not None
            else (
                MetricSpec("binary_auc"),
                *((MetricSpec("digit_hit"),) if data.has_digit_labels else ()),
            )
        )
        results: list[ClosureResult] = []
        for metric in selected_metrics:
            transparent_metric = metric.evaluate(
                test_data.target,
                transparent_score,
                digits=test_data.digits,
                thought_numbers=test_data.thought_numbers,
                subjects=test_data.subjects,
            )
            random_metric = metric.evaluate(
                test_data.target,
                random_score,
                digits=test_data.digits,
                thought_numbers=test_data.thought_numbers,
                subjects=test_data.subjects,
            )
            model_metric = metric.evaluate(
                test_data.target,
                model_score,
                digits=test_data.digits,
                thought_numbers=test_data.thought_numbers,
                subjects=test_data.subjects,
            )
            denominator = model_metric - random_metric
            defined = (
                np.isfinite(denominator) and abs(denominator) > self.config.denominator_epsilon
            )
            ratio = (
                float((transparent_metric - random_metric) / denominator)
                if defined
                else float("nan")
            )
            results.append(
                ClosureResult(
                    metric=metric.name,
                    feature_count=len(indices),
                    selected_features=selected,
                    selected_families=selected_families,
                    transparent_metric=float(transparent_metric),
                    random_metric=float(random_metric),
                    model_metric=float(model_metric),
                    closure_ratio=ratio,
                    defined=bool(defined),
                )
            )
        return tuple(results)

    def _fit_score(
        self, train_features: np.ndarray, train_y: np.ndarray, test_features: np.ndarray
    ) -> np.ndarray:
        scaler = StandardScaler().fit(train_features)
        classifier = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            solver="lbfgs",
            random_state=self.config.random_seed,
        )
        classifier.fit(scaler.transform(train_features), train_y)
        score = classifier.decision_function(scaler.transform(test_features))
        result = np.asarray(score, dtype=float).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise AuditInputError("transparent classifier produced NaN or infinite scores.")
        return result
