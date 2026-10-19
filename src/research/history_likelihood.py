"""Stored native noise and the existing augmented-QR owner for history marginals.

Y/Z are whitened once on the actual full block. Every state design uses that
same record startup; prefix takes leading rows. Work arrays are n by (k+c+2),
never n by n. The Gaussian coefficient integral and common flat nuisance
measure are the same ones as the continuous scorer's _joint_posterior.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from research.response_history_identification import HistoryNuisanceResolutionError
from transfer.continuous_decision import _joint_posterior


class NativeHistoryResolutionError(HistoryNuisanceResolutionError):
    def record(self):
        return {"kind": "native_history_numerical_resolution_failure", "detail": str(self)}


class NativeHistoryMarginals:
    def __init__(self, model, observation):
        self.model, self.observation = model, observation
        self.channels, self.kernel_width = model.kernel_mean.shape[1], model.kernel_mean.shape[0]
        self.nuisance_width = observation.nuisance.shape[1]
        self.white = model.noise.whiten(np.column_stack((observation.samples, observation.nuisance)))
        if self.white.shape != (len(observation.samples), self.channels+self.nuisance_width):
            raise ValueError("history needs a square stored full-record stationary whitening program")
        self.spatial = np.linalg.cholesky(model.innovation_covariance)
        self.logdet_spatial = 2*float(np.log(np.diag(self.spatial)).sum())
        self.root = model._posterior_precision_root
        self.logdet_prior = 2*float(np.log(np.abs(np.diag(self.root))).sum())
        for stop in (observation.conditioning_samples, len(observation.samples)):
            z = self.white[:stop, self.channels:]
            if self.nuisance_width:
                scales = np.max(np.abs(z), axis=0)
                if np.any(scales == 0):
                    raise HistoryNuisanceResolutionError("prefix cannot identify the common nuisance measure")
                singular = np.linalg.svd(z/scales, compute_uv=False)
                if (len(singular) < self.nuisance_width or singular[-1]
                        <= np.finfo(float).eps*max(z.shape)*singular[0]):
                    raise HistoryNuisanceResolutionError("prefix nuisance measure is numerically unresolved")

    def _log_whitening_determinant(self, stop):
        noise = self.model.noise
        startup = min(stop, noise.order)
        if noise.startup_mode == "stationary_innovations":
            return float(np.log(noise._startup_scales[:startup]).sum())
        if noise.startup_mode == "stationary":
            return -float(np.log(np.diag(noise._startup_root)[:startup]).sum())
        raise ValueError("conditional-prefix noise has no declared joint native history density")

    def _marginal(self, design, stop):
        y = self.white[:stop, :self.channels]
        matrix = np.column_stack((design[:stop], self.white[:stop, self.channels:]))
        prior = np.zeros((self.kernel_width, matrix.shape[1]))
        prior[:, :self.kernel_width] = self.root
        augmented = np.vstack((matrix, prior))
        scales = np.max(np.abs(augmented), axis=0)
        if np.any(scales == 0) or not np.isfinite(augmented).all():
            raise NativeHistoryResolutionError("history augmented coordinates are not finitely representable")
        singular = np.linalg.svd(augmented/scales, compute_uv=False)
        if singular[-1] <= np.finfo(float).eps*max(augmented.shape)*singular[0]:
            raise NativeHistoryResolutionError("history augmented coefficient/nuisance chart is unresolved")
        mean, root = _joint_posterior(matrix, y, self.model._predictive_kernel_mean, self.root)
        if not np.isfinite(root).all() or np.any(np.diag(root) == 0):
            raise NativeHistoryResolutionError("history augmented QR is unresolved")
        residual = y-matrix@mean
        parameter = self.root@(mean[:self.kernel_width]-self.model._predictive_kernel_mean)
        spatial_residual = solve_triangular(self.spatial, np.vstack((residual, parameter)).T, lower=True)
        logdet = 2*float(np.log(np.abs(np.diag(root))).sum())-self.logdet_prior
        logdet -= 2*self._log_whitening_determinant(stop)
        n_effective = stop-self.nuisance_width
        value = -.5*(float(np.sum(spatial_residual**2))+self.channels*logdet
                     + n_effective*(self.channels*np.log(2*np.pi)+self.logdet_spatial))
        if not np.isfinite(value):
            raise NativeHistoryResolutionError("native history marginal is not finitely representable")
        return float(value)

    def __call__(self, design):
        white_design = self.model.noise.whiten(design)
        return (self._marginal(white_design, len(design)),
                self._marginal(white_design, self.observation.conditioning_samples))
