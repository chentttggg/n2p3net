"""Exact CPU/float64 gradients for the calibration conditional AR objective.

Scientific ingress owns authorization, physical coordinates and finite float64
samples/designs. This numerical producer consumes only that calibration data.
One shared zero-mean kernel prior has a frozen precision root; every record has
its own diffuse intercept/trend. The same prefix-centered drift coordinates
are used for prefix and tail, so their change of measure cancels in the
conditional density. A general kernel-basis change also needs a transformed
prior. Calibration-g roots are compiled from the authorized C design once.

PARCOR lattice whitening preserves stationary startup and record boundaries.
Prefix and tail augmented QR avoid squaring the condition of a low-rank Gram
matrix; scatter is computed as a sum of squares, without subtracting nearly
equal full and prefix quadratics. All constant tensors are owned copies. The
only per-evaluation ingress is the small real finite eta vector.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from research.execution import UnidentifiableFitError
from transfer.kernel_prior import CompiledKernelPrior, resolve_kernel_prior
from transfer.noise_model import array_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _whiten_record(raw, reflections):
    """Record-local lattice with unit innovation variance."""
    order = len(reflections)
    if not order:
        return raw
    log_variances = -torch.flip(
        torch.cumsum(torch.log1p(-torch.flip(reflections, (0,)) ** 2), 0), (0,)
    )
    rows = [raw[:1] * torch.exp(-.5 * log_variances[0])]
    forward, backward = raw[1:], raw[:-1]
    for stage, reflection in enumerate(reflections, 1):
        next_forward = forward + reflection * backward
        if stage < order:
            rows.append(next_forward[:1] * torch.exp(-.5 * log_variances[stage]))
            next_backward = backward + reflection * forward
            forward, backward = next_forward[1:], next_backward[:-1]
        else:
            rows.append(next_forward)
    return torch.cat(rows)


class ConditionalGaussianObjective:
    """Fit-only exact profile objective and its eta gradient, per scored row.

    ``samples`` and ``designs`` are aligned, already validated float64 matrices
    for independent records. Each record must have more than
    ``conditioning_samples >= 2`` rows. Eta length is the AR order and may not
    exceed the common prefix. Execution thread budgets belong to the caller;
    this class does not alter Torch/global thread or device defaults.
    """

    __slots__ = (
        "_raw", "_prefix", "_channels", "_kernel_width", "_objective_rows",
        "_prior", "_prior_rhs", "_drift_before", "_drift_after",
    )

    def __init__(self, samples, designs, *, conditioning_samples, ridge=None, kernel_prior=None):
        if type(conditioning_samples) is not int or conditioning_samples < 2:
            raise ValueError("conditioning_samples must be an integer of at least two")
        if kernel_prior is not None:
            if not isinstance(kernel_prior, CompiledKernelPrior) or ridge is not None:
                raise ValueError("compiled kernel prior and scalar ridge are mutually exclusive")
            kernel_prior.digest()
        else:
            ridge = resolve_kernel_prior(None, ridge).precision
        pairs = tuple(zip(samples, designs, strict=True))
        if not pairs or any(len(y) <= conditioning_samples for y, _ in pairs):
            raise ValueError("every calibration record needs a prefix and a nonempty scored tail")
        lengths = np.array([len(y) for y, _ in pairs], dtype=np.int64)
        self._prefix = conditioning_samples
        self._channels = pairs[0][0].shape[1]
        self._kernel_width = pairs[0][1].shape[1]
        if kernel_prior is not None:
            if kernel_prior.precision_root.shape != (self._kernel_width, self._kernel_width):
                raise ValueError("compiled prior root differs from the objective kernel coordinates")
            if kernel_prior.spec.uses_calibration_reference and [
                array_digest(d) for _, d in pairs
            ] != [row["design_sha256"] for row in kernel_prior.provenance["records"]]:
                raise ValueError("gradient calibration designs differ from the frozen C prior")
        self._objective_rows = int(np.sum(lengths - conditioning_samples))
        record_count = len(pairs)
        width = self._kernel_width + 2 * record_count

        center = (conditioning_samples - 1) / 2
        raw = []
        for y, design in pairs:
            n = len(y)
            # column_stack owns its data, including caller samples/designs.
            raw.append(torch.from_numpy(np.column_stack((
                y, design, np.ones(n), (np.arange(n) - center) / center,
            ))))
        self._raw = tuple(raw)
        # Constant zero drift blocks are constructed once. Processing each
        # record separately also keeps lattice working sets small and removes
        # padding/index masks; measured packed batches were slower on CPU.
        self._drift_before = tuple(
            torch.zeros((n, 2 * index), dtype=torch.float64, device="cpu")
            for index, n in enumerate(lengths)
        )
        self._drift_after = tuple(
            torch.zeros((n, 2 * (record_count - index - 1)), dtype=torch.float64, device="cpu")
            for index, n in enumerate(lengths)
        )
        prior = np.zeros((self._kernel_width, width))
        prior[:, :self._kernel_width] = (np.sqrt(ridge)*np.eye(self._kernel_width)
                                         if kernel_prior is None else kernel_prior.precision_root)
        self._prior = torch.tensor(prior, dtype=torch.float64, device="cpu")
        self._prior_rhs = torch.zeros(
            (self._kernel_width, self._channels), dtype=torch.float64, device="cpu"
        )

    @property
    def objective_rows(self):
        return self._objective_rows

    def _conditional_terms(self, eta):
        reflections = torch.tanh(eta)
        prefix_a, prefix_y, tail_a, tail_y = [], [], [], []
        for raw, before, after in zip(self._raw, self._drift_before, self._drift_after, strict=True):
            transformed = _whiten_record(raw, reflections)
            matrix = torch.cat((transformed[:, self._channels:-2], before, transformed[:, -2:], after), dim=1)
            prefix_a.append(matrix[:self._prefix])
            tail_a.append(matrix[self._prefix:])
            prefix_y.append(transformed[:self._prefix, :self._channels])
            tail_y.append(transformed[self._prefix:, :self._channels])
        prefix_a, prefix_y, tail_a, tail_y = map(torch.cat, (prefix_a, prefix_y, tail_a, tail_y))
        prefix_q, prefix_root = torch.linalg.qr(torch.cat((prefix_a, self._prior)))
        prefix_mean = torch.linalg.solve_triangular(
            prefix_root, prefix_q.T @ torch.cat((prefix_y, self._prior_rhs)), upper=True
        )
        full_q, full_root = torch.linalg.qr(torch.cat((tail_a, prefix_root)))
        full_mean = torch.linalg.solve_triangular(
            full_root, full_q.T @ torch.cat((tail_y, prefix_root @ prefix_mean)), upper=True
        )
        residual = tail_y - tail_a @ full_mean
        correction = prefix_root @ (full_mean - prefix_mean)
        scatter = residual.T @ residual + correction.T @ correction
        logdet = 2 * (
            torch.log(torch.abs(torch.diag(full_root))).sum()
            - torch.log(torch.abs(torch.diag(prefix_root))).sum()
        )
        return scatter, logdet

    def value_gradient(self, eta):
        """Return raw profile objective / nTail and exact d(objective/nTail)/deta."""
        raw_eta = np.asarray(eta)
        if (raw_eta.ndim != 1 or raw_eta.dtype.kind not in "fiu"
                or len(raw_eta) > self._prefix or not np.isfinite(raw_eta).all()):
            raise ValueError("eta must be a finite real vector with order at most the prefix")
        if np.any(np.abs(np.tanh(raw_eta)) >= 1.):
            raise ValueError("eta must produce strictly stationary PARCOR coefficients")
        with torch.enable_grad():
            point = torch.tensor(raw_eta, dtype=torch.float64, device="cpu", requires_grad=True)
            try:
                scatter, logdet = self._conditional_terms(point)
                root = torch.linalg.cholesky((scatter + scatter.T) / (2 * self._objective_rows))
                objective = torch.log(torch.diag(root)).sum() + .5 * self._channels * logdet / self._objective_rows
                gradient = (torch.autograd.grad(objective, point)[0] if len(point)
                            else point.new_empty((0,)))
            except torch.linalg.LinAlgError as error:
                raise UnidentifiableFitError("conditional innovation covariance is not identifiable") from error
        value = float(objective.detach())
        result = gradient.detach().numpy().copy()
        if not np.isfinite(value) or not np.isfinite(result).all():
            raise UnidentifiableFitError("conditional Gaussian value or gradient is nonfinite")
        return value, result
