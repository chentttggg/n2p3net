"""Uncertainty-preserving projection from arbitrary EEG layouts to canonical sensors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CanonicalProjection:
    """GP posterior moments for one batch.

    ``mean`` has shape ``(B,Q,...)`` and ``covariance`` has shape
    ``(B,Q,Q)``. The covariance is never inferred from or reduced by the
    signal values: it only reflects sensor geometry, observation noise, and
    the effective channel mask.
    """

    mean: torch.Tensor
    covariance: torch.Tensor

    @property
    def variance(self) -> torch.Tensor:
        return torch.diagonal(self.covariance, dim1=-2, dim2=-1)


def _registered_rows(positions: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite coordinates.")
    norm = np.linalg.norm(value, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError(f"{name} contains a coordinate at the head origin.")
    if np.any(norm > 0.5):
        raise ValueError(
            f"{name} must be registered head-frame coordinates in metres; "
            "values above 0.5 m usually indicate unit-sphere or millimetre input."
        )
    return value


def _matern32(
    left: np.ndarray, right: np.ndarray, length_scale: float, variance: float
) -> np.ndarray:
    # Chordal distance is the Euclidean distance between registered 3D points.
    # This is not a scalp-geodesic or manifold kernel.
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    scaled = math.sqrt(3.0) * distance / length_scale
    return variance * (1.0 + scaled) * np.exp(-scaled)


def _tuple_rows(value: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(x) for x in row) for row in value)


@lru_cache(maxsize=256)
def _cached_posterior(
    observed_positions: tuple[tuple[float, float, float], ...],
    query_positions: tuple[tuple[float, float, float], ...],
    noise_variance: tuple[float, ...],
    observed_mask: tuple[bool, ...],
    length_scale: float,
    kernel_variance: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_positions, dtype=np.float64)
    query = np.asarray(query_positions, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    k_qq = _matern32(query, query, length_scale, kernel_variance)
    if not bool(mask.any()):
        weights = np.zeros((len(query), len(observed)), dtype=np.float64)
        covariance = k_qq
    else:
        active = observed[mask]
        k_pp = _matern32(active, active, length_scale, kernel_variance)
        k_qp = _matern32(query, active, length_scale, kernel_variance)
        noise = np.asarray(noise_variance, dtype=np.float64)[mask]
        system = k_pp + np.diag(noise + jitter)
        chol = np.linalg.cholesky(system)
        # K_QP (K_PP + Sigma_n)^-1, evaluated by two triangular solves.
        active_weights = np.linalg.solve(chol.T, np.linalg.solve(chol, k_qp.T)).T
        weights = np.zeros((len(query), len(observed)), dtype=np.float64)
        weights[:, mask] = active_weights
        covariance = k_qq - active_weights @ k_qp.T

    covariance = 0.5 * (covariance + covariance.T)
    # Round-off can leave tiny negative eigenvalues after the Schur complement.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return weights.astype(np.float32), covariance.astype(np.float32)


class RegisteredCoordinateGPProjector(nn.Module):
    """Exact GP/kriging on registered, metre-scale head coordinates.

    The Matern-3/2 kernel uses Euclidean chordal distance between real 3D
    sensor locations. It does not claim scalp-surface geodesics. A manifold
    Matern requires a registered scalp mesh and a Laplace-Beltrami eigensystem;
    requesting that geometry without those objects fails closed.
    """

    def __init__(
        self,
        observed_positions: np.ndarray | torch.Tensor,
        query_positions: np.ndarray | torch.Tensor,
        *,
        noise_variance: float | Sequence[float] = 0.05,
        length_scale: float = 0.055,
        kernel_variance: float = 1.0,
        jitter: float = 1e-6,
        kernel_geometry: str = "chordal_3d",
    ) -> None:
        super().__init__()
        if length_scale <= 0.0 or kernel_variance <= 0.0 or jitter <= 0.0:
            raise ValueError("GP length_scale, kernel_variance, and jitter must be positive.")
        if kernel_geometry != "chordal_3d":
            if kernel_geometry in {"manifold", "manifold_matern", "laplace_beltrami"}:
                raise ValueError(
                    "Manifold Matern requires a registered scalp mesh plus "
                    "Laplace-Beltrami eigenvalues/eigenfunctions; chordal coordinates alone "
                    "cannot implement or substantiate scalp-surface distance."
                )
            raise ValueError(f"Unsupported GP kernel_geometry {kernel_geometry!r}.")
        observed = _registered_rows(np.asarray(observed_positions), name="observed_positions")
        query = _registered_rows(np.asarray(query_positions), name="query_positions")
        if np.isscalar(noise_variance):
            noise = np.full(len(observed), float(noise_variance), dtype=np.float64)
        else:
            noise = np.asarray(tuple(noise_variance), dtype=np.float64)
        if noise.shape != (len(observed),) or not np.isfinite(noise).all() or np.any(noise < 0.0):
            raise ValueError(
                "noise_variance must be non-negative with one value per observed sensor."
            )

        self.length_scale = float(length_scale)
        self.kernel_variance = float(kernel_variance)
        self.jitter = float(jitter)
        self.kernel_geometry = kernel_geometry
        self.coordinate_frame = "head"
        self.coordinate_units = "m"
        self.register_buffer("observed_positions", torch.from_numpy(observed.astype(np.float32)))
        self.register_buffer("query_positions", torch.from_numpy(query.astype(np.float32)))
        self.register_buffer("noise_variance", torch.from_numpy(noise.astype(np.float32)))

    @property
    def n_observed(self) -> int:
        return int(self.observed_positions.shape[0])

    @property
    def n_query(self) -> int:
        return int(self.query_positions.shape[0])

    def kernel_spec(self) -> dict[str, object]:
        return {
            "family": "matern_3_2",
            "geometry": self.kernel_geometry,
            "distance": "euclidean_chordal",
            "coordinate_frame": self.coordinate_frame,
            "coordinate_units": self.coordinate_units,
            "length_scale_m": self.length_scale,
            "variance": self.kernel_variance,
            "jitter": self.jitter,
            "observation_noise_variance": [
                float(value) for value in self.noise_variance.detach().cpu().tolist()
            ],
        }

    @staticmethod
    def cache_info():
        return _cached_posterior.cache_info()

    @staticmethod
    def clear_cache() -> None:
        _cached_posterior.cache_clear()

    def _moments(
        self, mask: tuple[bool, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, covariance = _cached_posterior(
            _tuple_rows(self.observed_positions.detach().cpu().numpy()),
            _tuple_rows(self.query_positions.detach().cpu().numpy()),
            tuple(float(v) for v in self.noise_variance.detach().cpu().tolist()),
            mask,
            self.length_scale,
            self.kernel_variance,
            self.jitter,
        )
        return (
            torch.from_numpy(weights).to(device=device),
            torch.from_numpy(covariance).to(device=device),
        )

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> CanonicalProjection:
        if values.dim() < 2 or values.shape[1] != self.n_observed:
            raise ValueError(
                f"values must be (B,{self.n_observed},...), got {tuple(values.shape)}."
            )
        batch_size = values.shape[0]
        if channel_mask is None:
            masks = torch.ones(batch_size, self.n_observed, device=values.device, dtype=torch.bool)
        else:
            masks = channel_mask.to(device=values.device, dtype=torch.bool)
            if masks.shape == (self.n_observed,):
                masks = masks[None].expand(batch_size, -1)
            if masks.shape != (batch_size, self.n_observed):
                raise ValueError(
                    "channel_mask must be (P,) or (B,P), got "
                    f"{tuple(masks.shape)} for B={batch_size}, P={self.n_observed}."
                )

        flattened = values.reshape(batch_size, self.n_observed, -1)
        mean = torch.empty(
            batch_size,
            self.n_query,
            flattened.shape[-1],
            device=values.device,
            dtype=values.dtype,
        )
        covariance = torch.empty(
            batch_size,
            self.n_query,
            self.n_query,
            device=values.device,
            dtype=torch.float32,
        )
        unique_masks, inverse = torch.unique(masks, dim=0, return_inverse=True)
        for index, unique_mask in enumerate(unique_masks):
            rows = inverse == index
            key = tuple(bool(v) for v in unique_mask.detach().cpu().tolist())
            weights, cov = self._moments(key, values.device)
            mean[rows] = torch.einsum(
                "qp,bpr->bqr", weights.to(dtype=values.dtype), flattened[rows]
            )
            covariance[rows] = cov
        return CanonicalProjection(
            mean=mean.reshape(batch_size, self.n_query, *values.shape[2:]),
            covariance=covariance,
        )


# Compatibility import for older experiment code. The implementation no longer
# projects sensors to a unit sphere and must not be described as a spherical GP.
SphericalGPProjector = RegisteredCoordinateGPProjector


class CoordinateResidualAttention(nn.Module):
    """Bounded coordinate cross-attention correction applied after GP mean."""

    def __init__(
        self,
        observed_positions: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        attention_dim: int = 16,
        max_residual_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        if attention_dim < 1 or not 0.0 <= max_residual_fraction <= 1.0:
            raise ValueError("Invalid coordinate attention dimension or residual bound.")
        self.query = nn.Linear(3, attention_dim, bias=False)
        self.key = nn.Linear(3, attention_dim, bias=False)
        self.gate_raw = nn.Parameter(torch.zeros(query_positions.shape[0]))
        self.max_residual_fraction = float(max_residual_fraction)
        self.register_buffer("observed_positions", observed_positions.detach().clone())
        self.register_buffer("query_positions", query_positions.detach().clone())

    def forward(
        self,
        observed: torch.Tensor,
        gp_mean: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observed.dim() < 3 or gp_mean.dim() != observed.dim():
            raise ValueError(
                "observed and gp_mean must be matching batched sensor feature tensors."
            )
        batch_size, n_observed = observed.shape[:2]
        query = self.query(self.query_positions)
        key = self.key(self.observed_positions)
        logits = query @ key.T / math.sqrt(float(query.shape[-1]))
        if channel_mask is None:
            mask = torch.ones(batch_size, n_observed, device=observed.device, dtype=torch.bool)
        else:
            mask = channel_mask.to(device=observed.device, dtype=torch.bool)
            if mask.shape == (n_observed,):
                mask = mask[None].expand(batch_size, -1)
            if mask.shape != (batch_size, n_observed):
                raise ValueError("channel_mask must be (P,) or (B,P) for coordinate attention.")
        scores = logits[None].expand(batch_size, -1, -1)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        weights = scores.exp() * mask[:, None].to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attention_mean = torch.einsum(
            "bqp,bpr->bqr",
            weights.to(dtype=observed.dtype),
            observed.reshape(batch_size, n_observed, -1),
        ).reshape_as(gp_mean)
        gate = self.max_residual_fraction * torch.tanh(self.gate_raw)
        shape = (1, len(gate)) + (1,) * (gp_mean.dim() - 2)
        return gp_mean + gate.view(shape).to(dtype=gp_mean.dtype) * (attention_mean - gp_mean)
