"""Interventional Erase stage for the P300 PEC audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .adapters import LayerIntervention, P300ModelAdapter
from .metrics import (
    BootstrapSummary,
    MetricSpec,
    benjamini_hochberg,
    paired_bootstrap_drop,
)
from .probe import ProbeResult
from .types import AuditInputError, FeatureTable, P300AuditData, P300Split


@dataclass(frozen=True)
class EraseConfig:
    covariance_threshold: float = 1e-4
    n_bootstrap: int = 128
    fdr_q: float = 0.05
    residual_r2_max: float = 0.35
    require_control_ci: bool = True
    random_seed: int = 4311

    def __post_init__(self) -> None:
        if self.covariance_threshold <= 0 or self.n_bootstrap < 16:
            raise AuditInputError(
                "Erase threshold must be positive and bootstrap count must be >=16."
            )
        if not 0 < self.fdr_q <= 1 or not 0 <= self.residual_r2_max <= 1:
            raise AuditInputError("Erase fdr_q must be in (0,1] and residual_r2_max in [0,1].")


@dataclass(frozen=True)
class CrossCovarianceEraser:
    """Euclidean cross-covariance subspace eraser.

    The fitted object stores the training activation mean and orthonormal
    directions.  It is immutable after fitting and safe to reuse across
    validation/test batches.
    """

    mean: np.ndarray
    directions: np.ndarray
    singular_values: np.ndarray

    @classmethod
    def fit(
        cls,
        activation: np.ndarray,
        target: np.ndarray,
        *,
        covariance_threshold: float = 1e-4,
    ) -> CrossCovarianceEraser:
        activation_array = np.asarray(activation, dtype=float)
        target_array = np.asarray(target, dtype=float)
        if (
            activation_array.ndim != 2
            or target_array.ndim not in (1, 2)
            or activation_array.shape[0] != target_array.shape[0]
        ):
            raise AuditInputError(
                "eraser activation must be (N,D) and target must align as (N,) or (N,P)."
            )
        if (
            activation_array.shape[0] < 2
            or activation_array.shape[1] == 0
            or not np.all(np.isfinite(activation_array))
            or not np.all(np.isfinite(target_array))
        ):
            raise AuditInputError("eraser inputs must be finite and contain at least two rows.")
        if np.std(target_array, axis=0).max() <= np.finfo(float).eps:
            raise AuditInputError("eraser target is constant in the training split.")
        if target_array.ndim == 1:
            target_array = target_array[:, None]
        activation_mean = activation_array.mean(axis=0)
        target_centered = target_array - target_array.mean(axis=0, keepdims=True)
        activation_centered = activation_array - activation_mean
        covariance = activation_centered.T @ target_centered / max(activation_array.shape[0], 1)
        left_vectors, singular_values, _ = np.linalg.svd(covariance, full_matrices=False)
        threshold = float(covariance_threshold) * max(float(singular_values.max(initial=0.0)), 1.0)
        keep = singular_values > threshold
        if not np.any(keep):
            keep[: min(activation_array.shape[1], target_centered.shape[1])] = True
        directions = left_vectors[:, keep]
        if directions.size == 0:
            raise AuditInputError("eraser could not retain a non-empty target-correlated subspace.")
        return cls(
            mean=activation_mean.copy(),
            directions=directions.copy(),
            singular_values=singular_values[keep].copy(),
        )

    @property
    def rank(self) -> int:
        return int(self.directions.shape[1])

    def transform(self, activation: np.ndarray) -> np.ndarray:
        activation_array = np.asarray(activation, dtype=float)
        if (
            activation_array.ndim != 2
            or activation_array.shape[1] != self.mean.shape[0]
            or not np.all(np.isfinite(activation_array))
        ):
            raise AuditInputError("activation has incompatible shape for this eraser.")
        centered = activation_array - self.mean
        return activation_array - centered @ self.directions @ self.directions.T


@dataclass(frozen=True)
class EraseResult:
    feature: str
    family: str
    layer: str | None
    status: str
    eraser_rank: int
    baseline_metric: float
    erased_metric: float
    random_metric: float
    shuffled_metric: float
    gaussian_metric: float
    real_drop: BootstrapSummary | None
    control_drop: BootstrapSummary | None
    residual_probe_r2: float
    fdr_q_value: float
    representation_causal: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("real_drop", "control_drop"):
            value = payload[key]
            if isinstance(value, BootstrapSummary):
                payload[key] = asdict(value)
        return payload


class P300EraseAuditor:
    """Run feature-specific interventions with null controls and FDR."""

    def __init__(self, config: EraseConfig | None = None):
        self.config = config or EraseConfig()

    def run(
        self,
        adapter: P300ModelAdapter,
        data: P300AuditData,
        split: P300Split,
        features: FeatureTable,
        activations: dict[str, np.ndarray],
        probes: tuple[ProbeResult, ...],
        *,
        metric: MetricSpec | None = None,
        batch_size: int = 256,
    ) -> tuple[EraseResult, ...]:
        metric = metric or MetricSpec("binary_auc")
        split.validate_against(data)
        if len(probes) != features.n_features:
            raise AuditInputError("one ProbeResult is required for every feature column.")
        for layer, value in activations.items():
            array = np.asarray(value, dtype=float)
            if (
                array.ndim != 2
                or array.shape[0] != data.n_trials
                or array.shape[1] == 0
                or not np.all(np.isfinite(array))
            ):
                raise AuditInputError(f"activation layer {layer!r} has incompatible shape.")
        test_data = data.subset(split.test)
        baseline_all = adapter.predict_scores(data.X, batch_size=batch_size)
        baseline_score = baseline_all[split.test]
        baseline_metric = metric.evaluate(
            test_data.target,
            baseline_score,
            digits=test_data.digits,
            thought_numbers=test_data.thought_numbers,
            subjects=test_data.subjects,
        )

        partial: list[dict[str, object]] = []
        causal_indices: list[int] = []
        for feature_index, probe in enumerate(probes):
            if (
                probe.feature != features.names[feature_index]
                or probe.family != features.families[feature_index]
            ):
                raise AuditInputError(
                    "ProbeResult order or feature metadata does not match features."
                )
            feature = features.values[:, feature_index]
            if not probe.selection_encoded:
                partial.append(
                    {
                        "feature": probe.feature,
                        "family": probe.family,
                        "layer": probe.peak_layer,
                        "status": "encoded_gate_failed",
                        "eraser_rank": 0,
                        "baseline_metric": baseline_metric,
                        "erased_metric": float("nan"),
                        "random_metric": float("nan"),
                        "shuffled_metric": float("nan"),
                        "gaussian_metric": float("nan"),
                        "real_drop": None,
                        "control_drop": None,
                        "residual_probe_r2": float("nan"),
                        "feature_index": feature_index,
                    }
                )
                continue
            if probe.peak_layer not in activations:
                raise AuditInputError(
                    f"Probe selected unknown activation layer {probe.peak_layer!r}."
                )
            train_activation = activations[probe.peak_layer][split.train]
            target_train = feature[split.train]
            real_eraser = CrossCovarianceEraser.fit(
                train_activation,
                target_train,
                covariance_threshold=self.config.covariance_threshold,
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([self.config.random_seed, feature_index])
            )
            random_eraser = self._random_eraser(train_activation, rng)
            shuffled_eraser = CrossCovarianceEraser.fit(
                train_activation,
                rng.permutation(target_train),
                covariance_threshold=self.config.covariance_threshold,
            )
            gaussian_eraser = CrossCovarianceEraser.fit(
                train_activation,
                rng.standard_normal(target_train.shape[0]),
                covariance_threshold=self.config.covariance_threshold,
            )
            interventions = [
                LayerIntervention(probe.peak_layer, real_eraser.transform, "feature"),
                LayerIntervention(probe.peak_layer, random_eraser.transform, "random_subspace"),
                LayerIntervention(probe.peak_layer, shuffled_eraser.transform, "shuffled_target"),
                LayerIntervention(probe.peak_layer, gaussian_eraser.transform, "gaussian_target"),
            ]
            scores = [
                adapter.predict_scores(data.X, intervention=intervention, batch_size=batch_size)[
                    split.test
                ]
                for intervention in interventions
            ]
            real_drop = paired_bootstrap_drop(
                metric,
                test_data.target,
                baseline_score,
                scores[0],
                subjects=test_data.subjects,
                digits=test_data.digits,
                thought_numbers=test_data.thought_numbers,
                n_bootstrap=self.config.n_bootstrap,
                seed=self.config.random_seed + feature_index * 17,
            )
            control_drop = paired_bootstrap_drop(
                metric,
                test_data.target,
                scores[1],
                scores[0],
                subjects=test_data.subjects,
                digits=test_data.digits,
                thought_numbers=test_data.thought_numbers,
                n_bootstrap=self.config.n_bootstrap,
                seed=self.config.random_seed + feature_index * 17 + 1,
            )
            residual_r2 = self._residual_probe_r2(
                real_eraser.transform(train_activation),
                target_train,
                real_eraser.transform(activations[probe.peak_layer][split.validation]),
                feature[split.validation],
            )
            partial.append(
                {
                    "feature": probe.feature,
                    "family": probe.family,
                    "layer": probe.peak_layer,
                    "status": "erased",
                    "eraser_rank": real_eraser.rank,
                    "baseline_metric": baseline_metric,
                    "erased_metric": metric.evaluate(
                        test_data.target,
                        scores[0],
                        digits=test_data.digits,
                        thought_numbers=test_data.thought_numbers,
                        subjects=test_data.subjects,
                    ),
                    "random_metric": metric.evaluate(
                        test_data.target,
                        scores[1],
                        digits=test_data.digits,
                        thought_numbers=test_data.thought_numbers,
                        subjects=test_data.subjects,
                    ),
                    "shuffled_metric": metric.evaluate(
                        test_data.target,
                        scores[2],
                        digits=test_data.digits,
                        thought_numbers=test_data.thought_numbers,
                        subjects=test_data.subjects,
                    ),
                    "gaussian_metric": metric.evaluate(
                        test_data.target,
                        scores[3],
                        digits=test_data.digits,
                        thought_numbers=test_data.thought_numbers,
                        subjects=test_data.subjects,
                    ),
                    "real_drop": real_drop,
                    "control_drop": control_drop,
                    "residual_probe_r2": residual_r2,
                    "feature_index": feature_index,
                }
            )
            causal_indices.append(len(partial) - 1)

        p_values = np.full(len(partial), np.nan, dtype=float)
        for index in causal_indices:
            real_drop = partial[index]["real_drop"]
            assert isinstance(real_drop, BootstrapSummary)
            p_values[index] = real_drop.p_value
        q_values = benjamini_hochberg(p_values)

        results: list[EraseResult] = []
        for index, row in enumerate(partial):
            real_drop = row["real_drop"]
            control_drop = row["control_drop"]
            if not isinstance(real_drop, BootstrapSummary) or not isinstance(
                control_drop, BootstrapSummary
            ):
                results.append(
                    EraseResult(
                        feature=str(row["feature"]),
                        family=str(row["family"]),
                        layer=row["layer"],
                        status=str(row["status"]),
                        eraser_rank=int(row["eraser_rank"]),
                        baseline_metric=float(row["baseline_metric"]),
                        erased_metric=float(row["erased_metric"]),
                        random_metric=float(row["random_metric"]),
                        shuffled_metric=float(row["shuffled_metric"]),
                        gaussian_metric=float(row["gaussian_metric"]),
                        real_drop=None,
                        control_drop=None,
                        residual_probe_r2=float(row["residual_probe_r2"]),
                        fdr_q_value=float("nan"),
                        representation_causal=False,
                    )
                )
                continue
            control_pass = control_drop.estimate > 0 and (
                control_drop.lower > 0 if self.config.require_control_ci else True
            )
            causal = (
                q_values[index] <= self.config.fdr_q
                and real_drop.lower > 0
                and control_pass
                and float(row["residual_probe_r2"]) <= self.config.residual_r2_max
            )
            results.append(
                EraseResult(
                    feature=str(row["feature"]),
                    family=str(row["family"]),
                    layer=row["layer"],
                    status=str(row["status"]),
                    eraser_rank=int(row["eraser_rank"]),
                    baseline_metric=float(row["baseline_metric"]),
                    erased_metric=float(row["erased_metric"]),
                    random_metric=float(row["random_metric"]),
                    shuffled_metric=float(row["shuffled_metric"]),
                    gaussian_metric=float(row["gaussian_metric"]),
                    real_drop=real_drop,
                    control_drop=control_drop,
                    residual_probe_r2=float(row["residual_probe_r2"]),
                    fdr_q_value=float(q_values[index]),
                    representation_causal=bool(causal),
                )
            )
        return tuple(results)

    @staticmethod
    def _random_eraser(activation: np.ndarray, rng: np.random.Generator) -> CrossCovarianceEraser:
        dimension = activation.shape[1]
        random_matrix = rng.standard_normal((dimension, 1))
        directions, _ = np.linalg.qr(random_matrix, mode="reduced")
        return CrossCovarianceEraser(
            mean=activation.mean(axis=0).copy(),
            directions=directions[:, :1].copy(),
            singular_values=np.ones(1, dtype=float),
        )

    @staticmethod
    def _residual_probe_r2(
        train_activation: np.ndarray,
        train_y: np.ndarray,
        validation_activation: np.ndarray,
        validation_y: np.ndarray,
    ) -> float:
        if np.std(train_y) <= np.finfo(float).eps:
            return 0.0
        x_scaler = StandardScaler().fit(train_activation)
        y_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
        model = Ridge(alpha=1.0).fit(
            x_scaler.transform(train_activation),
            y_scaler.transform(train_y.reshape(-1, 1)).ravel(),
        )
        prediction = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(validation_activation)).reshape(-1, 1)
        ).reshape(-1)
        value = float(r2_score(validation_y, prediction))
        return max(0.0, value) if np.isfinite(value) else 0.0
