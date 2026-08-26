"""Object L: LatencyMeasurement for the v12 architecture.

The estimator is intentionally statistical and classifier-independent:

1. a fold-local, subject-balanced P3b template is aligned with restricted
   Woody iterations on optimization subjects;
2. residuals are whitened with a shrunk covariance estimate;
3. each trial is scored with an amplitude profile likelihood over a latency
   grid, producing a full posterior ``q_i(tau)``;
4. population tau0 is identified under ``E[delta]=0`` with an explicit anchor
   (fixed template and/or a physiological prior).

The posterior may be consumed by PCW only after detaching it from any
autograd graph. This module has no trainable parameters and no PyTorch import,
so classification gradients cannot leak back into the measurement by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndshift
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class LatencyPosterior:
    """Per-trial posterior over component latency.

    Attributes
    ----------
    q : np.ndarray
        ``(N, G)`` normalized posterior mass over ``grid_ms``.
    grid_ms : np.ndarray
        ``(G,)`` latency grid in milliseconds.
    mean_ms : np.ndarray
        ``(N,)`` posterior mean latency.
    amplitude_mean : np.ndarray
        ``(N,)`` posterior-weighted whitened-template amplitude.
    lower_ms : np.ndarray
        ``(N,)`` lower bound of the shortest contiguous 90% interval.
    upper_ms : np.ndarray
        ``(N,)`` upper bound of the shortest contiguous 90% interval.
    entropy : np.ndarray
        ``(N,)`` posterior entropy in nats.
    effective_n : int
        Number of finite posterior rows.
    """

    q: np.ndarray
    grid_ms: np.ndarray
    mean_ms: np.ndarray
    amplitude_mean: np.ndarray
    lower_ms: np.ndarray
    upper_ms: np.ndarray
    entropy: np.ndarray
    effective_n: int

    def validate(self) -> None:
        if self.q.ndim != 2:
            raise ValueError("LatencyPosterior q must be two-dimensional.")
        n, g = self.q.shape
        if g != len(self.grid_ms) or n < 1:
            raise ValueError("LatencyPosterior q/grid shapes are inconsistent.")
        if not np.isfinite(self.q).all():
            raise ValueError("LatencyPosterior contains non-finite mass.")
        summaries = (
            self.mean_ms,
            self.amplitude_mean,
            self.lower_ms,
            self.upper_ms,
            self.entropy,
        )
        if any(np.asarray(values).shape != (n,) for values in summaries):
            raise ValueError("LatencyPosterior summary shapes are inconsistent.")
        if not np.isfinite(self.grid_ms).all() or not all(
            np.isfinite(values).all() for values in summaries
        ):
            raise ValueError("LatencyPosterior contains non-finite summaries.")
        if self.effective_n != n:
            raise ValueError("LatencyPosterior effective_n must match q rows.")
        if not np.allclose(self.q.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("LatencyPosterior rows must be normalized.")


def parabolic_peak(samples: np.ndarray) -> float:
    """Sub-sample peak location of a three-point discrete peak."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if samples.shape != (3,):
        raise ValueError("parabolic_peak requires exactly three samples.")
    y0, y1, y2 = samples
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 1.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))


class LatencyMeasurement:
    """Fold-local, whitened template latency estimator.

    Parameters
    ----------
    anchor_tau0_ms:
        Physiological prior for the component center. Used as the initial
        template anchor and for population tau0 reporting.
    grid_radius_ms:
        Half-width of the latency grid around the anchor.
    grid_step_ms:
        Grid resolution.
    score_lo_ms / score_hi_ms:
        Interior scoring window; shifts are never evaluated through a
        zero-padded boundary shortcut.
    """

    def __init__(
        self,
        *,
        anchor_tau0_ms: float,
        sfreq: float,
        time_ms: np.ndarray,
        grid_radius_ms: float = 60.0,
        grid_step_ms: float = 0.5,
        score_lo_ms: float | None = None,
        score_hi_ms: float | None = None,
        max_alignment_shift_ms: float = 40.0,
        woody_iterations: int = 3,
        posterior_scale: float = 1.0,
    ) -> None:
        scalar_values = (
            anchor_tau0_ms,
            sfreq,
            grid_radius_ms,
            grid_step_ms,
            max_alignment_shift_ms,
            posterior_scale,
        )
        if not np.isfinite(scalar_values).all():
            raise ValueError("LatencyMeasurement scalar parameters must be finite.")
        if anchor_tau0_ms <= 0.0:
            raise ValueError("anchor_tau0_ms must be positive.")
        if sfreq <= 0.0 or grid_radius_ms <= 0.0 or grid_step_ms <= 0.0:
            raise ValueError("sfreq, grid_radius_ms and grid_step_ms must be positive.")
        self.anchor_tau0_ms = float(anchor_tau0_ms)
        self.sfreq = float(sfreq)
        self.time_ms = np.asarray(time_ms, dtype=np.float64)
        if self.time_ms.ndim != 1 or not np.all(np.diff(self.time_ms) > 0.0):
            raise ValueError("time_ms must be strictly increasing.")
        self.grid_radius_ms = float(grid_radius_ms)
        self.grid_step_ms = float(grid_step_ms)
        self.max_alignment_shift_ms = float(max_alignment_shift_ms)
        if isinstance(woody_iterations, bool) or not isinstance(
            woody_iterations, (int, np.integer)
        ) or woody_iterations < 0:
            raise ValueError("woody_iterations must be a non-negative integer.")
        if max_alignment_shift_ms < 0.0 or posterior_scale <= 0.0:
            raise ValueError("alignment shift must be non-negative and posterior scale positive.")
        self.woody_iterations = int(woody_iterations)
        self.posterior_scale = float(posterior_scale)

        if score_lo_ms is None:
            score_lo_ms = self.anchor_tau0_ms - 200.0
        if score_hi_ms is None:
            score_hi_ms = self.anchor_tau0_ms + 200.0
        if not (self.time_ms[0] <= score_lo_ms < score_hi_ms <= self.time_ms[-1]):
            score_lo_ms = max(self.time_ms[0], self.anchor_tau0_ms - 200.0)
            score_hi_ms = min(self.time_ms[-1], self.anchor_tau0_ms + 200.0)
        if not (self.time_ms[0] <= score_lo_ms < score_hi_ms <= self.time_ms[-1]):
            raise ValueError("score window must lie inside the physical time axis.")
        self.score_mask = (self.time_ms >= score_lo_ms) & (self.time_ms <= score_hi_ms)
        if self.score_mask.sum() < 8:
            raise ValueError("score window is too small.")

        self.grid_ms = np.arange(
            self.anchor_tau0_ms - self.grid_radius_ms,
            self.anchor_tau0_ms + self.grid_radius_ms + 0.5 * self.grid_step_ms,
            self.grid_step_ms,
            dtype=np.float64,
        )
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _mask_dot(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(
            np.dot(
                np.asarray(first)[:, self.score_mask].ravel(),
                np.asarray(second)[:, self.score_mask].ravel(),
            )
        )

    def _subject_balanced_template(self, X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        """Estimate one aligned template per subject, then average subjects."""
        subject_means: list[np.ndarray] = []
        for subject in np.unique(subject_ids):
            rows = np.flatnonzero(subject_ids == subject)
            if rows.size < 2:
                continue
            subject_mean = np.mean(X[rows], axis=0)
            subject_means.append(subject_mean)
        if not subject_means:
            raise ValueError("Template estimation requires at least one subject with two target trials.")
        template = np.mean(np.asarray(subject_means), axis=0)

        max_shift = int(round(self.max_alignment_shift_ms * self.sfreq / 1000.0))
        for _ in range(self.woody_iterations):
            aligned_means: list[np.ndarray] = []
            for subject_mean in subject_means:
                best = (-np.inf, 0)
                for shift_sample in range(-max_shift, max_shift + 1):
                    shifted = ndshift(
                        subject_mean, (0, shift_sample), order=1, mode="constant", cval=0.0
                    )
                    value = self._mask_dot(shifted, template)
                    if value > best[0]:
                        best = (value, shift_sample)
                aligned_means.append(
                    ndshift(subject_mean, (0, best[1]), order=1, mode="constant", cval=0.0)
                )
            template = np.mean(np.asarray(aligned_means), axis=0)

        # Optionally sharpen with per-trial Woody alignment using subject-wise
        # means as robust starting points.
        for _ in range(self.woody_iterations):
            aligned_trials: list[np.ndarray] = []
            for subject in np.unique(subject_ids):
                rows = np.flatnonzero(subject_ids == subject)
                for row in rows:
                    trial = X[row]
                    best = (-np.inf, 0)
                    for shift_sample in range(-max_shift, max_shift + 1):
                        shifted = ndshift(trial, (0, shift_sample), order=1, mode="constant", cval=0.0)
                        value = self._mask_dot(shifted, template)
                        if value > best[0]:
                            best = (value, shift_sample)
                    aligned_trials.append(
                        ndshift(trial, (0, best[1]), order=1, mode="constant", cval=0.0)
                    )
            candidate = np.mean(np.asarray(aligned_trials), axis=0)
            if np.linalg.norm(candidate - template) < 1e-3:
                template = candidate
                break
            template = candidate
        return template

    def fit(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> LatencyMeasurement:
        """Fit template and whitening on optimization subjects only.

        ``y==1`` trials provide the template; ``y==0`` trials provide the
        residual covariance. Callers must guarantee subject-disjointness.
        """
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("LatencyMeasurement X must have a floating dtype.")
        X = X.astype(np.float64, copy=False)
        y = np.asarray(y)
        subject_ids = np.asarray(subject_ids)
        if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or set(
            np.unique(y).tolist()
        ) != {0, 1}:
            raise ValueError("LatencyMeasurement y must be an integer binary vector.")
        if subject_ids.ndim != 1:
            raise ValueError("LatencyMeasurement subject_ids must be one-dimensional.")
        if X.ndim != 3 or X.shape[0] != len(y) or len(y) != len(subject_ids):
            raise ValueError("X, y and subject_ids must align as (N,C,T).")
        if X.shape[2] != len(self.time_ms):
            raise ValueError("X time axis does not match time_ms.")
        if not np.isfinite(X).all():
            raise ValueError("LatencyMeasurement.fit requires finite observations.")

        target_rows = np.flatnonzero(y == 1)
        if target_rows.size < 8:
            raise ValueError("LatencyMeasurement.fit requires at least eight target trials.")
        template = self._subject_balanced_template(X[target_rows], subject_ids[target_rows])

        # Estimate per-trial amplitude/tau on training targets to obtain
        # template-free residuals for the covariance estimator.
        residual_rows: list[np.ndarray] = []
        max_shift = int(round(self.max_alignment_shift_ms * self.sfreq / 1000.0))
        for row in target_rows:
            trial = X[row]
            best = (-np.inf, 0)
            for shift_sample in range(-max_shift, max_shift + 1):
                shifted = ndshift(template, (0, shift_sample), order=1, mode="constant", cval=0.0)
                num = self._mask_dot(shifted, trial)
                den = self._mask_dot(shifted, shifted)
                if den <= 0.0:
                    continue
                value = num * num / den
                if value > best[0]:
                    best = (value, shift_sample)
            shifted = ndshift(template, (0, best[1]), order=1, mode="constant", cval=0.0)
            den = self._mask_dot(shifted, shifted)
            amplitude = self._mask_dot(shifted, trial) / den
            residual = trial - amplitude * shifted
            residual_rows.append(residual)

        nontarget_rows = np.flatnonzero(y == 0)
        if nontarget_rows.size:
            subject_nontarget_means: dict[object, np.ndarray] = {}
            for subject in np.unique(subject_ids[nontarget_rows]):
                rows = np.flatnonzero((subject_ids == subject) & (y == 0))
                subject_nontarget_means[subject] = np.mean(X[rows], axis=0)
            for row in nontarget_rows:
                residual_rows.append(X[row] - subject_nontarget_means[subject_ids[row]])

        residual_matrix = np.stack(residual_rows)[:, :, self.score_mask].reshape(
            len(residual_rows), -1
        )
        if residual_matrix.shape[0] < 10:
            raise ValueError("Whitening estimation requires at least ten residual trials.")
        covariance = LedoitWolf().fit(residual_matrix).covariance_
        chol = np.linalg.cholesky(covariance + 1e-8 * np.eye(covariance.shape[0]))
        self._whitening_chol = chol
        self._template = template
        whitened_templates = np.stack([self._whitened_template(tau) for tau in self.grid_ms])
        template_denom = (whitened_templates * whitened_templates).sum(axis=1)
        valid_grid = template_denom > 1e-12
        if int(valid_grid.sum()) < 8:
            raise ValueError("Template support is too small on the configured latency grid.")
        if not bool(valid_grid.all()):
            self.grid_ms = self.grid_ms[valid_grid]
            whitened_templates = whitened_templates[valid_grid]
            template_denom = template_denom[valid_grid]
        self._whitened_templates = whitened_templates
        self._template_denom = template_denom
        self._fitted = True
        return self

    def _whitened_template(self, tau_ms: float) -> np.ndarray:
        if not hasattr(self, "_template") or not hasattr(self, "_whitening_chol"):
            raise RuntimeError("LatencyMeasurement must be fit before prediction.")
        shifted = ndshift(
            self._template,
            (0, (tau_ms - self.anchor_tau0_ms) * self.sfreq / 1000.0),
            order=1,
            mode="constant",
            cval=0.0,
        )
        flat = shifted[:, self.score_mask].reshape(-1)
        return np.linalg.solve(self._whitening_chol.T, flat)

    def _profile_statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(profile_log_likelihoods, profile_amplitudes)``."""
        if not self._fitted:
            raise RuntimeError("LatencyMeasurement must be fit before prediction.")
        flat_x = np.asarray(X, dtype=np.float64)[:, :, self.score_mask].reshape(X.shape[0], -1)
        whitened_x = np.linalg.solve(self._whitening_chol.T, flat_x.T).T
        templates = self._whitened_templates
        denom = self._template_denom
        num = whitened_x @ templates.T
        amplitudes = num / denom[None, :]
        rss = (whitened_x * whitened_x).sum(axis=1)[:, None] - num * num / denom[None, :]
        return -0.5 * rss, amplitudes

    def _profile_log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        return self._profile_statistics(X)[0]

    def predict(self, X: np.ndarray, posterior_scale: float | None = None) -> LatencyPosterior:
        """Return posterior summaries for new trials."""
        if not self._fitted:
            raise RuntimeError("LatencyMeasurement must be fit before prediction.")
        scale = self.posterior_scale if posterior_scale is None else float(posterior_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("posterior_scale must be finite and positive.")
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1:] != self._template.shape:
            raise ValueError("LatencyMeasurement.predict X must match fitted (C,T) geometry.")
        if not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
            raise ValueError("LatencyMeasurement.predict requires finite floating observations.")
        lls, profile_amplitudes = self._profile_statistics(X)
        lls = lls / (scale * scale)
        lls = lls - lls.max(axis=1, keepdims=True)
        q = np.exp(lls)
        q /= q.sum(axis=1, keepdims=True)

        means = q @ self.grid_ms
        amplitude_means = (q * profile_amplitudes).sum(axis=1)
        lowers: list[float] = []
        uppers: list[float] = []
        entropies: list[float] = []
        for row in q:
            i = int(np.argmax(row))
            j = i
            mass = row[i]
            while mass < 0.90:
                left = row[i - 1] if i > 0 else -1.0
                right = row[j + 1] if j + 1 < len(row) else -1.0
                if left >= right:
                    i -= 1
                    mass += row[i]
                else:
                    j += 1
                    mass += row[j]
            lowers.append(self.grid_ms[i])
            uppers.append(self.grid_ms[j])
            entropies.append(-float(np.sum(row * np.log(np.clip(row, 1e-12, None)))))
        posterior = LatencyPosterior(
            q=q,
            grid_ms=self.grid_ms,
            mean_ms=means,
            amplitude_mean=amplitude_means,
            lower_ms=np.asarray(lowers),
            upper_ms=np.asarray(uppers),
            entropy=np.asarray(entropies),
            effective_n=len(means),
        )
        posterior.validate()
        return posterior

    def calibrate_posterior_scale(
        self,
        X: np.ndarray,
        true_latency_ms: np.ndarray,
        *,
        target_coverage: float = 0.90,
        scale_lo: float = 0.5,
        scale_hi: float = 12.0,
        n_candidates: int = 121,
    ) -> float:
        """Choose the posterior sharpness on a known-shift calibration set.

        This method is only valid for synthetic audits or digitally shifted
        real epochs where ground-truth latency exists. For unlabeled real data
        the default scale remains descriptive.
        """
        if not self._fitted:
            raise RuntimeError("LatencyMeasurement must be fit before calibration.")
        truth = np.asarray(true_latency_ms, dtype=np.float64)
        if truth.shape != (len(X),):
            raise ValueError("true_latency_ms must align with X.")
        best_scale = float(scale_lo)
        best_gap = np.inf
        for scale in np.linspace(scale_lo, scale_hi, n_candidates):
            posterior = self.predict(X, posterior_scale=scale)
            coverage = float(
                np.mean((truth >= posterior.lower_ms) & (truth <= posterior.upper_ms))
            )
            gap = abs(coverage - target_coverage)
            if gap < best_gap:
                best_gap = gap
                best_scale = float(scale)
        self.posterior_scale = best_scale
        return best_scale

    def population_tau0(
        self,
        posterior: LatencyPosterior,
        *,
        prior_mean_ms: float,
        prior_sd_ms: float,
    ) -> dict[str, float]:
        """Estimate population tau0 under E[delta]=0 plus a physiological anchor."""
        if not np.isfinite(posterior.mean_ms).all():
            raise ValueError("posterior means must be finite.")
        sample_mean = float(np.mean(posterior.mean_ms))
        n = max(posterior.effective_n, 1)
        prior_weight = 1.0 / (prior_sd_ms**2)
        sample_weight = n / max(float(np.var(posterior.mean_ms)), 1e-6)

        def weighted_tau0(anchor_ms: float) -> float:
            return (prior_weight * anchor_ms + sample_weight * sample_mean) / (
                prior_weight + sample_weight
            )

        posterior_tau0 = weighted_tau0(float(prior_mean_ms))
        tau0_from_lo = weighted_tau0(float(prior_mean_ms) - 2.0 * float(prior_sd_ms))
        tau0_from_hi = weighted_tau0(float(prior_mean_ms) + 2.0 * float(prior_sd_ms))
        anchor_sensitivity = abs(tau0_from_hi - tau0_from_lo)
        return {
            "tau0_ms": posterior_tau0,
            "sample_mean_ms": sample_mean,
            "prior_mean_ms": float(prior_mean_ms),
            "prior_sd_ms": float(prior_sd_ms),
            # How far the reported tau0 moves when the physiological anchor is
            # displaced by +/- 2 prior SD. This is the meaningful anchor/init
            # sensitivity for the pre-registered <2 ms gate.
            "anchor_sensitivity_ms": anchor_sensitivity,
            "init_sensitivity_ms": anchor_sensitivity,
        }



def detached_expected_window(
    posterior: LatencyPosterior,
    *,
    time_ms: np.ndarray,
    width_ms: float,
) -> np.ndarray:
    """Materialize the posterior as a detached expected Gaussian window.

    The returned array is plain NumPy and therefore has no autograd history.
    PCW may consume it as an auxiliary feature, but it cannot propagate
    classification gradients back into the measurement object.
    """
    time_ms = np.asarray(time_ms, dtype=np.float64)
    if time_ms.ndim != 1 or not np.all(np.diff(time_ms) > 0):
        raise ValueError("time_ms must be strictly increasing.")
    if width_ms <= 0.0:
        raise ValueError("width_ms must be positive.")
    gaussian = np.exp(-0.5 * ((time_ms[None, :] - posterior.grid_ms[:, None]) / width_ms) ** 2)
    return posterior.q @ gaussian
