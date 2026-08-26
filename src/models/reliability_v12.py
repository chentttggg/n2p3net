"""v12 Reliability object: two independently falsifiable estimands.

* ``FidelityEstimator`` maps nuisance features to a scalar fidelity score.
  Its training objective is a margin/rank objective over synthetic corruption
  severity; the score is a mixture weight, never a probability claim.
* ``CleanProbabilityEstimator`` outputs ``P(clean | q; prior)`` only when it is
  trained on an explicit binary pollution generator with hard clean/artifact
  labels. Deployment prior shifts use an odds conversion; group-specific
  calibration patches are never applied at deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

_EPS = 1e-12


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(probabilities)
    p = np.asarray(probabilities, dtype=np.float64)[order]
    y = np.asarray(labels, dtype=np.float64)[order]
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    count = 0
    for index in range(bins):
        mask = (p > edges[index]) & (p <= edges[index + 1])
        if mask.sum():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
            count += mask.sum()
    return float(total / max(count, 1))


class FidelityEstimator(nn.Module):
    """Monotone-risk scalar quality score trained by margin/rank loss."""

    def __init__(self, n_features: int, hidden_size: int = 16) -> None:
        super().__init__()
        if n_features < 1 or hidden_size < 1:
            raise ValueError("n_features and hidden_size must be positive.")
        self.n_features = int(n_features)
        self.net = nn.Sequential(
            nn.Linear(self.n_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("center", torch.zeros(self.n_features))
        self.register_buffer("scale", torch.ones(self.n_features))

    @torch.no_grad()
    def fit_normalizer(self, quality: torch.Tensor) -> None:
        values = quality.detach().float().reshape(-1, self.n_features)
        if values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
            raise ValueError("fidelity normalizer needs at least two finite rows.")
        center = values.median(dim=0).values
        q25 = torch.quantile(values, 0.25, dim=0)
        q75 = torch.quantile(values, 0.75, dim=0)
        scale = ((q75 - q25) / 1.349).clamp_min(1e-3)
        self.center.copy_(center.to(self.center))
        self.scale.copy_(scale.to(self.scale))

    def normalize(self, quality: torch.Tensor) -> torch.Tensor:
        return ((quality - self.center) / self.scale).clamp(-8.0, 8.0)

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        if quality.shape[-1] != self.n_features:
            raise ValueError(f"quality needs {self.n_features} features.")
        return self.net(self.normalize(quality)).squeeze(-1)

    @staticmethod
    def margin_rank_loss(clean_scores: torch.Tensor, corrupt_scores: torch.Tensor) -> torch.Tensor:
        clean_scores = clean_scores.reshape(-1)
        corrupt_scores = corrupt_scores.reshape(-1)
        return F.relu(1.0 - (clean_scores[:, None] - corrupt_scores[None, :])).mean()

    def fidelity_margin_rank_loss(self, quality: torch.Tensor) -> torch.Tensor:
        """Class-blind feature-space corruption margin objective.

        This replaces the legacy soft-BCE(0.9/0.1) probability anchor.
        """
        normalized = self.normalize(quality).detach()
        clean_mask = normalized.abs().amax(dim=-1) < 3.0
        if not bool(clean_mask.any()):
            return quality.sum() * 0.0
        corrupted = normalized.clone()
        rows = torch.arange(corrupted.shape[0], device=corrupted.device)
        feature = rows.remainder(self.n_features)
        corrupted[rows, feature] = torch.maximum(
            corrupted[rows, feature], corrupted.new_full((len(rows),), 6.0)
        )
        clean_scores = self.forward(quality[clean_mask])
        corrupt_scores = self.net(corrupted).squeeze(-1)
        return self.margin_rank_loss(clean_scores, corrupt_scores)


@dataclass(frozen=True)
class FidelityGateReport:
    passed: bool
    unseen_type_auc: dict[str, float]
    subject_auc: dict[str, float]
    min_unseen_auc: float
    min_subject_auc: float


def evaluate_fidelity_gate(
    estimator: FidelityEstimator,
    quality: torch.Tensor,
    labels: np.ndarray,
    corruption_type: np.ndarray,
    subject_ids: np.ndarray,
    *,
    unseen_types: set[str],
    unseen_auc_min: float = 0.85,
    subject_auc_min: float = 0.80,
) -> FidelityGateReport:
    """Evaluate fidelity on unseen corruption types and unseen subjects.

    ``labels`` is 1 for clean, 0 for corrupted; ``corruption_type`` is the
    string identifier of each corrupted trial and is empty for clean trials.
    """
    with torch.inference_mode():
        scores = estimator(quality).detach().cpu().numpy().astype(np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    corruption_type = np.asarray(corruption_type)
    subject_ids = np.asarray(subject_ids)
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary.")
    if set(np.unique(corruption_type[labels == 0])).issubset(unseen_types) is False:
        raise ValueError("corrupt trials must all carry an unseen corruption type.")

    unseen_type_auc: dict[str, float] = {}
    subject_auc: dict[str, float] = {}
    for corruption in unseen_types:
        rows = np.flatnonzero((labels == 0) & (corruption_type == corruption))
        if not len(rows):
            unseen_type_auc[corruption] = float("nan")
            continue
        clean_rows = np.flatnonzero(labels == 1)
        y = np.concatenate((np.ones(len(clean_rows)), np.zeros(len(rows))))
        score = np.concatenate((scores[clean_rows], scores[rows]))
        unseen_type_auc[corruption] = float(roc_auc_score(y, score))
    for subject in np.unique(subject_ids):
        rows = np.flatnonzero(subject_ids == subject)
        if np.unique(labels[rows]).size != 2:
            subject_auc[str(subject)] = float("nan")
            continue
        subject_auc[str(subject)] = float(roc_auc_score(labels[rows], scores[rows]))
    finite_type = [value for value in unseen_type_auc.values() if np.isfinite(value)]
    finite_subject = [value for value in subject_auc.values() if np.isfinite(value)]
    min_unseen_auc = float(min(finite_type)) if finite_type else float("nan")
    min_subject_auc = float(min(finite_subject)) if finite_subject else float("nan")
    passed = bool(
        finite_type
        and finite_subject
        and min_unseen_auc >= unseen_auc_min
        and min_subject_auc >= subject_auc_min
    )
    return FidelityGateReport(
        passed=passed,
        unseen_type_auc=unseen_type_auc,
        subject_auc=subject_auc,
        min_unseen_auc=min_unseen_auc,
        min_subject_auc=min_subject_auc,
    )


class CleanProbabilityEstimator:
    """Binary clean/artifact probability from an explicit pollution generator.

    The estimator is intentionally a shallow logistic model plus a monotone
    recalibrator. It has no hidden state and can be refit per fold.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1.0, max_iter=2000)
        self.calibrator: IsotonicRegression | None = None
        self.calibration_prior: float | None = None
        self._fitted = False

    def fit(self, quality: np.ndarray, labels: np.ndarray) -> CleanProbabilityEstimator:
        q = np.asarray(quality, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if q.ndim != 2 or q.shape[0] != len(y):
            raise ValueError("quality and labels must align.")
        if not ((y == 0) | (y == 1)).all() or np.unique(y).size != 2:
            raise ValueError("clean_probability requires hard binary labels with both classes.")
        self.model.fit(q, y)
        self._fitted = True
        return self

    def predict_proba(self, quality: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CleanProbabilityEstimator must be fit first.")
        return self.model.predict_proba(np.asarray(quality, dtype=np.float64))[:, 1]

    def fit_calibrator(
        self, quality: np.ndarray, labels: np.ndarray, *, calibration_prior: float
    ) -> None:
        """Fit a monotone calibrator on an independent hard-label pool."""
        if not 0.0 < calibration_prior < 1.0:
            raise ValueError("calibration_prior must lie strictly inside (0,1).")
        probabilities = self.predict_proba(quality)
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            probabilities, labels
        )
        self.calibration_prior = float(calibration_prior)

    def predict_calibrated(self, quality: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(quality)
        if self.calibrator is None:
            return probabilities
        return self.calibrator.predict(probabilities)


def convert_prior_odds(
    probabilities: np.ndarray, *, calibration_prior: float, deployment_prior: float
) -> np.ndarray:
    """Convert posterior probabilities from one clean prevalence to another."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    odds = p / (1.0 - p)
    odds = odds * (deployment_prior / (1.0 - deployment_prior)) * (
        (1.0 - calibration_prior) / calibration_prior
    )
    return odds / (1.0 + odds)



@dataclass(frozen=True)
class CleanProbabilityChainGateReport:
    passed: bool
    n_sequences: int
    n_candidates: int
    uniform_nll: float
    selection_nll: float
    selection_brier: float
    selection_ece: float
    decision_hit_rate: float
    top1_margin_mean: float
    decision_reverified: bool
    decision_agreement_rate: float | None
    failure: str | None = None


def digit_chain_scores_from_components(
    clean_probability: np.ndarray,
    clean_log_prob: np.ndarray,
    artifact_log_prob: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Build digit-chain candidate scores from calibrated clean probabilities.

    The only chain semantics sanctioned by blueprint 4.2 is the per-flash
    mixture ``log(rho * exp(A) + (1-rho) * exp(B))``, accumulated over the
    complete candidate path. This is the same object used by S0-3
    ``s0_monotone_rho_chain_rank``.
    """
    rho = np.asarray(clean_probability, dtype=np.float64)
    clean = np.asarray(clean_log_prob, dtype=np.float64)
    artifact = np.asarray(artifact_log_prob, dtype=np.float64)
    if rho.ndim != 2:
        raise ValueError("clean_probability must be (N,T) for digit-chain scoring.")
    if clean.ndim != 3 or artifact.shape != clean.shape:
        raise ValueError("clean_log_prob/artifact_log_prob must share shape (N,T,C).")
    n_sequences, n_steps = rho.shape
    if clean.shape[:2] != (n_sequences, n_steps):
        raise ValueError("rho and density components must align on (N,T).")
    if not bool(((rho >= 0.0) & (rho <= 1.0)).all()):
        raise ValueError("clean_probability must lie in [0,1].")
    if lengths is None:
        lengths = np.full(n_sequences, n_steps, dtype=np.int64)
    else:
        lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if lengths.shape != (n_sequences,) or not bool(
            ((lengths >= 1) & (lengths <= n_steps)).all()
        ):
            raise ValueError("lengths must contain one valid length per sequence.")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rho = np.where(rho > 0.0, np.log(rho), -np.inf)
        log_one_minus_rho = np.where(rho < 1.0, np.log1p(-rho), -np.inf)
    mixture = np.logaddexp(
        log_rho[:, :, None] + clean,
        log_one_minus_rho[:, :, None] + artifact,
    )
    active = np.arange(n_steps)[None, :] < lengths[:, None]
    return np.where(active[:, :, None], mixture, 0.0).sum(axis=1)


def evaluate_clean_probability_chain_gate(
    chain_scores: np.ndarray,
    true_candidates: np.ndarray,
    *,
    reference_chain_scores: np.ndarray | None = None,
    nll_uniform_floor: bool = True,
) -> CleanProbabilityChainGateReport:
    """Fail-closed digit-chain NLL and decision re-verification.

    The rho Brier/ECE/AUC gate is deliberately insufficient: a monotone rho
    recalibration can preserve rho AUC while flipping chain ordering. This
    gate therefore recomputes the complete digit-chain softmax, its selection
    NLL and the resulting 9-choice decisions on held-out true candidates.
    Callers must keep the chain sequences subject-disjoint from the pools used
    to fit ``clean_probability`` and its monotone calibrator.
    """
    scores = np.asarray(chain_scores, dtype=np.float64)
    true = np.asarray(true_candidates, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("chain_scores must be a non-empty (N,C) array.")
    n_sequences, n_candidates = scores.shape
    if true.shape != (n_sequences,):
        raise ValueError("true_candidates must have one value per chain sequence.")
    if not bool(np.isfinite(scores).all()):
        raise ValueError("chain_scores must be finite.")
    if not bool(((true >= 0) & (true < n_candidates)).all()):
        raise ValueError("true_candidates must be valid candidate indices.")
    if n_candidates < 2:
        raise ValueError("digit-chain re-verification requires at least two candidates.")

    agreement: float | None = None
    if reference_chain_scores is not None:
        reference = np.asarray(reference_chain_scores, dtype=np.float64)
        if reference.shape != scores.shape or not bool(np.isfinite(reference).all()):
            raise ValueError("reference_chain_scores must be finite with the same shape.")
        agreement = float((reference.argmax(axis=1) == scores.argmax(axis=1)).mean())

    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rows = np.arange(n_sequences)
    selection_nll = float(-np.log(probabilities[rows, true].clip(1e-12, 1.0)).mean())
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, true] = 1.0
    selection_brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == true
    selection_ece = _ece(confidence, correct.astype(np.float64))
    decision_hit_rate = float(correct.mean())
    top_two = np.sort(np.partition(scores, -2, axis=1)[:, -2:], axis=1)
    top1_margin_mean = float((top_two[:, 1] - top_two[:, 0]).mean())
    uniform_nll = float(np.log(n_candidates))

    failure: str | None = None
    decision_reverified = bool(
        np.isfinite(selection_nll)
        and np.isfinite(selection_brier)
        and np.isfinite(selection_ece)
        and np.isfinite(decision_hit_rate)
    )
    if not decision_reverified:
        failure = "digit_chain_metrics_nonfinite"
    elif nll_uniform_floor and selection_nll + 1e-9 >= uniform_nll:
        failure = "digit_chain_nll_worse_than_uniform"
    return CleanProbabilityChainGateReport(
        passed=bool(decision_reverified and failure is None),
        n_sequences=int(n_sequences),
        n_candidates=int(n_candidates),
        uniform_nll=uniform_nll,
        selection_nll=selection_nll,
        selection_brier=selection_brier,
        selection_ece=selection_ece,
        decision_hit_rate=decision_hit_rate,
        top1_margin_mean=top1_margin_mean,
        decision_reverified=decision_reverified,
        decision_agreement_rate=agreement,
        failure=failure,
    )


def evaluate_clean_probability_gate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    brier_max: float = 0.15,
    ece_max: float = 0.20,
    chain_scores: np.ndarray | None = None,
    true_candidates: np.ndarray | None = None,
    reference_chain_scores: np.ndarray | None = None,
    chain_required: bool = True,
    nll_uniform_floor: bool = True,
) -> dict[str, object]:
    """Evaluate clean_probability on held-out hard labels plus digit-chain NLL.

    ``chain_scores``/``true_candidates`` are required by default: a monotone
    rho recalibration can preserve Brier/ECE/AUC while flipping chain order,
    so chain selection NLL and decision re-verification are part of the gate.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("labels must be binary.")
    brier = float(np.mean((p - y) ** 2))
    ece = _ece(p, y)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    rho_passed = bool(brier <= brier_max and ece <= ece_max)
    if (chain_scores is None) != (true_candidates is None):
        return {
            "passed": False,
            "brier": brier,
            "ece": ece,
            "auc": auc,
            "failure": "chain_scores_and_true_candidates_must_be_paired",
            "chain": {
                "passed": False,
                "failure": "chain_scores_and_true_candidates_must_be_paired",
            },
        }
    if chain_required and (chain_scores is None or true_candidates is None):
        return {
            "passed": False,
            "brier": brier,
            "ece": ece,
            "auc": auc,
            "failure": "missing_digit_chain_nll_or_true_candidates",
            "chain": {
                "passed": False,
                "failure": "missing_digit_chain_nll_or_true_candidates",
            },
        }
    if chain_scores is not None and true_candidates is not None:
        chain_report = evaluate_clean_probability_chain_gate(
            chain_scores,
            true_candidates,
            reference_chain_scores=reference_chain_scores,
            nll_uniform_floor=nll_uniform_floor,
        ).__dict__
    else:
        chain_report = {"passed": True, "available": False}
    passed = bool(rho_passed and chain_report["passed"])
    return {
        "passed": passed,
        "rho_passed": rho_passed,
        "brier": brier,
        "ece": ece,
        "auc": auc,
        "chain": chain_report,
        "failure": (
            None
            if passed
            else "rho_brier_or_ece"
            if not rho_passed
            else chain_report.get("failure") or "digit_chain_gate_failed"
        ),
    }
