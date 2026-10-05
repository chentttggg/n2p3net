"""One run-level stopping policy; each fit owns an independent patience counter."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

DEFAULT_EARLY_STOP_PATIENCE = 6
DEFAULT_EARLY_STOP_MIN_DELTA = 1e-6


@dataclass(frozen=True)
class UnifiedTrainingBudget:
    """One source reference selects exposure; all personal fits use a fixed budget."""

    reference_patience: int = 3
    reference_min_delta: float = 1e-4
    reference_max_epochs: int = 12
    source_validation_fraction: float = 0.2
    target_epochs: int = 90
    development_people_per_domain: int = 2
    development_seed: int = 20260906

    def __post_init__(self):
        self.reference_policy()
        for name in ("target_epochs", "development_people_per_domain"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.development_seed) is not int or self.development_seed < 0:
            raise ValueError("development_seed must be a nonnegative integer")

    def record(self):
        return asdict(self)

    def reference_policy(self):
        return GlobalEarlyStoppingPolicy(patience=self.reference_patience,
            min_delta=self.reference_min_delta, source_max_epochs=self.reference_max_epochs,
            target_max_epochs=1, source_validation_fraction=self.source_validation_fraction)


@dataclass(frozen=True)
class GlobalEarlyStoppingPolicy:
    """Freeze once per experiment, never select settings per person or outer test."""

    patience: int = DEFAULT_EARLY_STOP_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOP_MIN_DELTA
    source_max_epochs: int = 32
    target_max_epochs: int = 200
    source_validation_fraction: float = 0.2

    def __post_init__(self):
        PatienceStopper(self.patience, self.min_delta)
        for name in ("source_max_epochs", "target_max_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        fraction = self.source_validation_fraction
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("source_validation_fraction must lie in (0, 1)")

    def record(self):
        return asdict(self)

    def target_selection(self):
        return {"mode": "patience", "patience": self.patience,
                "min_delta": self.min_delta, "require_validation": True}


class PatienceStopper:
    """Consume finite validation losses; patience counts unsuccessful checks."""

    def __init__(self, patience=DEFAULT_EARLY_STOP_PATIENCE,
                 min_delta=DEFAULT_EARLY_STOP_MIN_DELTA):
        if type(patience) is not int or patience < 1:
            raise ValueError("early stopping patience must be a positive integer")
        if isinstance(min_delta, bool) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("early stopping min_delta must be finite and nonnegative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.best_check = None
        self.checks = 0
        self.bad_checks = 0

    def observe(self, loss):
        if not math.isfinite(loss):
            raise ValueError("early stopping requires finite validation loss")
        self.checks += 1
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = float(loss)
            self.best_check = self.checks
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return improved

    @property
    def exhausted(self):
        return self.bad_checks >= self.patience

    def record(self, *, enabled=True, stop_at_patience=True):
        return {"enabled": enabled, "patience": self.patience, "min_delta": self.min_delta,
                "validation_checks": self.checks, "best_check": self.best_check,
                "best_validation_loss": self.best_loss if self.best_check is not None else None,
                "bad_checks": self.bad_checks,
                "stop_reason": ("patience" if self.exhausted and stop_at_patience else "epoch_limit")
                if enabled else "fixed_budget"}
