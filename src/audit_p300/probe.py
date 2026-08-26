"""Layer-wise Probe stage of the P300 PEC audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .types import AuditInputError, FeatureTable, P300AuditData, P300Split


@dataclass(frozen=True)
class ProbeConfig:
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    encoding_r2_threshold: float = 0.04
    control_margin: float = 0.01
    peak_margin: float = 0.002
    random_seed: int = 4311

    def __post_init__(self) -> None:
        if not self.alphas or any(float(alpha) <= 0 for alpha in self.alphas):
            raise AuditInputError("Probe alphas must be a non-empty tuple of positive values.")
        if self.encoding_r2_threshold < 0 or self.control_margin < 0 or self.peak_margin < 0:
            raise AuditInputError("Probe thresholds and margins must be non-negative.")


@dataclass(frozen=True)
class ProbeResult:
    feature: str
    family: str
    peak_layer: str
    peak_validation_r2: float
    second_best_validation_r2: float
    second_best_test_r2: float
    test_r2: float
    selected_alpha: float
    validation_shuffled_r2: float
    validation_gaussian_r2: float
    test_shuffled_r2: float
    test_gaussian_r2: float
    selection_encoded: bool
    test_encoded: bool
    layer_validation_r2: dict[str, float]
    layer_test_r2: dict[str, float]

    @property
    def encoded(self) -> bool:
        return self.selection_encoded

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _FittedProbe:
    model: Ridge
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    @property
    def alpha(self) -> float:
        return float(self.model.alpha)

    def predict(self, activation: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(self.x_scaler.transform(activation)).reshape(-1, 1)
        return self.y_scaler.inverse_transform(prediction).reshape(-1)


class LayerProbeAuditor:
    """Fit leakage-safe ridge probes at every declared representation layer."""

    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()

    def run(
        self,
        activations: dict[str, np.ndarray],
        features: FeatureTable,
        split: P300Split,
        data: P300AuditData,
    ) -> tuple[ProbeResult, ...]:
        split.validate_against(data)
        if features.n_trials != data.n_trials:
            raise AuditInputError("feature rows must match the audit data rows.")
        if not activations:
            raise AuditInputError("at least one activation layer is required.")
        normalized: dict[str, np.ndarray] = {}
        for name, value in activations.items():
            array = np.asarray(value, dtype=float)
            if array.ndim != 2 or array.shape[0] != data.n_trials:
                raise AuditInputError(
                    f"layer {name!r} must have shape (N,D) with N={data.n_trials}."
                )
            if array.shape[1] == 0 or not np.all(np.isfinite(array)):
                raise AuditInputError(f"layer {name!r} must have finite non-empty features.")
            normalized[name] = array

        results: list[ProbeResult] = []
        for feature_index, (feature_name, family) in enumerate(
            zip(features.names, features.families, strict=True)
        ):
            target = features.values[:, feature_index]
            if not np.all(np.isfinite(target)):
                raise AuditInputError(f"feature {feature_name!r} contains NaN or infinite values.")
            result = self._run_one(
                feature_index,
                feature_name,
                family,
                target,
                normalized,
                split,
            )
            results.append(result)
        return tuple(results)

    def _run_one(
        self,
        feature_index: int,
        feature_name: str,
        family: str,
        target: np.ndarray,
        activations: dict[str, np.ndarray],
        split: P300Split,
    ) -> ProbeResult:
        train_target = target[split.train]
        val_target = target[split.validation]
        test_target = target[split.test]
        if np.std(train_target) <= np.finfo(float).eps:
            raise AuditInputError(f"feature {feature_name!r} is constant in the training split.")

        layer_val: dict[str, float] = {}
        layer_test: dict[str, float] = {}
        layer_models: dict[str, _FittedProbe] = {}
        for layer, activation in activations.items():
            model = self._fit_best_probe(
                activation[split.train], train_target, activation[split.validation], val_target
            )
            val_prediction = model.predict(activation[split.validation])
            test_prediction = model.predict(activation[split.test])
            layer_val[layer] = self._r2(val_target, val_prediction)
            layer_test[layer] = self._r2(test_target, test_prediction)
            layer_models[layer] = model

        ordered = sorted(layer_val.items(), key=lambda item: (-item[1], item[0]))
        peak_layer, peak_val = ordered[0]
        second_best_validation = ordered[1][1] if len(ordered) > 1 else float("-inf")
        ordered_test = sorted(layer_test.values(), reverse=True)
        second_best_test = ordered_test[1] if len(ordered_test) > 1 else float("-inf")
        peak_model = layer_models[peak_layer]
        selected_alpha = peak_model.alpha
        peak_activation = activations[peak_layer]
        test_prediction = peak_model.predict(peak_activation[split.test])

        rng = np.random.default_rng(
            np.random.SeedSequence([self.config.random_seed, feature_index])
        )
        shuffled_train = rng.permutation(train_target)
        gaussian_train = rng.standard_normal(train_target.shape[0])
        shuffled_model = self._fit_at_alpha(
            peak_activation[split.train], shuffled_train, selected_alpha
        )
        gaussian_model = self._fit_at_alpha(
            peak_activation[split.train], gaussian_train, selected_alpha
        )
        shuffled_val = shuffled_model.predict(peak_activation[split.validation])
        shuffled_test = shuffled_model.predict(peak_activation[split.test])
        gaussian_val = gaussian_model.predict(peak_activation[split.validation])
        gaussian_test = gaussian_model.predict(peak_activation[split.test])
        val_shuffled_r2 = self._r2(val_target, shuffled_val)
        val_gaussian_r2 = self._r2(val_target, gaussian_val)
        test_shuffled_r2 = self._r2(test_target, shuffled_test)
        test_gaussian_r2 = self._r2(test_target, gaussian_test)

        selection_encoded = (
            peak_val >= self.config.encoding_r2_threshold
            and peak_val - val_shuffled_r2 >= self.config.control_margin
            and peak_val - val_gaussian_r2 >= self.config.control_margin
            and peak_val - second_best_validation >= self.config.peak_margin
        )
        test_encoded = (
            layer_test[peak_layer] >= self.config.encoding_r2_threshold
            and layer_test[peak_layer] - test_shuffled_r2 >= self.config.control_margin
            and layer_test[peak_layer] - test_gaussian_r2 >= self.config.control_margin
            and layer_test[peak_layer] - second_best_test >= self.config.peak_margin
        )
        return ProbeResult(
            feature=feature_name,
            family=family,
            peak_layer=peak_layer,
            peak_validation_r2=float(peak_val),
            second_best_validation_r2=float(second_best_validation),
            second_best_test_r2=float(second_best_test),
            test_r2=float(layer_test[peak_layer]),
            selected_alpha=selected_alpha,
            validation_shuffled_r2=float(val_shuffled_r2),
            validation_gaussian_r2=float(val_gaussian_r2),
            test_shuffled_r2=float(test_shuffled_r2),
            test_gaussian_r2=float(test_gaussian_r2),
            selection_encoded=bool(selection_encoded),
            test_encoded=bool(test_encoded),
            layer_validation_r2={k: float(v) for k, v in layer_val.items()},
            layer_test_r2={k: float(v) for k, v in layer_test.items()},
        )

    def _fit_best_probe(
        self,
        train_activation: np.ndarray,
        train_y: np.ndarray,
        validation_activation: np.ndarray,
        validation_y: np.ndarray,
    ) -> _FittedProbe:
        best: tuple[float, _FittedProbe] | None = None
        for alpha in self.config.alphas:
            model = self._fit_at_alpha(train_activation, train_y, float(alpha))
            prediction = model.predict(validation_activation)
            score = self._r2(validation_y, prediction)
            candidate = (score, model)
            if best is None or score > best[0]:
                best = candidate
        assert best is not None
        return best[1]

    @staticmethod
    def _fit_at_alpha(activation: np.ndarray, target: np.ndarray, alpha: float) -> _FittedProbe:
        x_scaler = StandardScaler().fit(activation)
        scaled_activation = x_scaler.transform(activation)
        target_scaler = StandardScaler().fit(np.asarray(target).reshape(-1, 1))
        model = Ridge(alpha=alpha)
        model.fit(
            scaled_activation,
            target_scaler.transform(np.asarray(target).reshape(-1, 1)).ravel(),
        )

        return _FittedProbe(model=model, x_scaler=x_scaler, y_scaler=target_scaler)

    @staticmethod
    def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
        try:
            value = float(r2_score(y_true, prediction))
        except ValueError:
            return 0.0
        return max(0.0, value) if np.isfinite(value) else 0.0
