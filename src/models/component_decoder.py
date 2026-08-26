"""Constrained sparse morphology dictionary for probabilistic ERP decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ERPComponentOutput:
    """Moment-matched ERP decomposition and auditable morphology state."""

    reconstruction: torch.Tensor
    component_waveforms: torch.Tensor
    amplitude_mean: torch.Tensor
    amplitude_variance: torch.Tensor
    null_variance: torch.Tensor
    waveform_variance: torch.Tensor
    morphology_basis: torch.Tensor
    morphology_parameters: torch.Tensor
    morphology_variance: torch.Tensor
    atom_coefficients: torch.Tensor
    atom_gates: torch.Tensor
    expected_l0: torch.Tensor
    shape_uncertainty: torch.Tensor
    epistemic_variance: torch.Tensor
    anchor_latency_ms: torch.Tensor
    component_peak_latency_ms: torch.Tensor
    waveform_peak_latency_ms: torch.Tensor


class ERPComponentDecoder(nn.Module):
    """Decode PCW states with a constrained, sparse ERP morphology dictionary.

    Each component has a unit-coefficient asymmetric generalized-Gaussian main
    peak plus optional derivative-edge, signed secondary-peak and smooth gamma
    tail atoms. Optional atoms use global Hard-Concrete gates, while their
    trial-specific coefficients remain bounded.

    The mean is a signed channel amplitude times a constrained morphology basis.
    There is deliberately no multiplicative existence probability: class-
    contrast supervision cannot identify existence and amplitude separately,
    and their product admits a stable zero shortcut. The variance path consumes
    ``stopgrad(H)`` and uses a diagonal delta-method approximation
    ``J_theta V_theta J_theta^T`` for morphology uncertainty.
    """

    _MORPHOLOGY_PARAMETER_NAMES = (
        "tau_ms",
        "sigma_up_ms",
        "sigma_down_ms",
        "beta",
        "secondary_delay_ms",
        "tail_decay_ms",
    )

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        n_components: int = 3,
        *,
        tmin_ms: float = -200.0,
        sfreq: float = 256.0,
        n_time: int = 256,
        log_variance_bounds: tuple[float, float] = (-4.0, 2.0),
        variance_floor: float = 1e-4,
        beta_bounds: tuple[float, float] = (1.0, 4.0),
        secondary_delay_bounds_ms: tuple[float, float] = (40.0, 200.0),
        tail_decay_bounds_ms: tuple[float, float] = (40.0, 300.0),
        coefficient_bounds: tuple[float, float, float] = (0.5, 0.7, 0.5),
        hard_concrete_temperature: float = 2.0 / 3.0,
        hard_concrete_stretch: tuple[float, float] = (-0.1, 1.1),
    ):
        super().__init__()
        lo, hi = (float(v) for v in log_variance_bounds)
        if not lo < hi or variance_floor <= 0.0:
            raise ValueError("Invalid amplitude variance bounds/floor.")
        if int(n_time) < 2 or float(sfreq) <= 0.0:
            raise ValueError("n_time and sfreq must define at least two samples.")
        if not beta_bounds[0] < beta_bounds[1]:
            raise ValueError("beta_bounds must have positive width.")
        if not secondary_delay_bounds_ms[0] < secondary_delay_bounds_ms[1]:
            raise ValueError("secondary_delay_bounds_ms must have positive width.")
        if not tail_decay_bounds_ms[0] < tail_decay_bounds_ms[1]:
            raise ValueError("tail_decay_bounds_ms must have positive width.")
        if any(float(v) <= 0.0 for v in coefficient_bounds):
            raise ValueError("All optional-atom coefficient bounds must be positive.")
        gamma, zeta = (float(v) for v in hard_concrete_stretch)
        if not gamma < 0.0 < 1.0 < zeta or hard_concrete_temperature <= 0.0:
            raise ValueError("Invalid Hard-Concrete temperature/stretch.")

        self.n_channels = int(n_channels)
        self.n_components = int(n_components)
        self.n_time = int(n_time)
        self.sfreq = float(sfreq)
        self.log_variance_bounds = (lo, hi)
        self.variance_floor = float(variance_floor)
        self.beta_bounds = tuple(float(v) for v in beta_bounds)
        self.secondary_delay_bounds_ms = tuple(float(v) for v in secondary_delay_bounds_ms)
        self.tail_decay_bounds_ms = tuple(float(v) for v in tail_decay_bounds_ms)
        self.coefficient_bounds = tuple(float(v) for v in coefficient_bounds)
        self.hard_concrete_temperature = float(hard_concrete_temperature)
        self.hard_concrete_gamma = gamma
        self.hard_concrete_zeta = zeta

        times_ms = float(tmin_ms) + torch.arange(self.n_time, dtype=torch.float32) * (
            1000.0 / self.sfreq
        )
        self.register_buffer("times_ms", times_ms)
        self.register_buffer(
            "jacobian_steps",
            torch.tensor((1.0, 1.0, 1.0, 0.02, 1.0, 1.0), dtype=torch.float32),
        )
        # Parameter-specific scales retain the physical units in J V J^T.
        std_lo = torch.tensor((0.5, 0.5, 0.5, 0.01, 0.5, 0.5), dtype=torch.float32)
        std_hi = torch.tensor((80.0, 50.0, 50.0, 1.0, 80.0, 100.0), dtype=torch.float32)
        self.register_buffer("morph_log_variance_lo", 2.0 * std_lo.log())
        self.register_buffer("morph_log_variance_hi", 2.0 * std_hi.log())

        self.amplitude_heads = nn.ModuleList(
            nn.Linear(d_model, self.n_channels) for _ in range(self.n_components)
        )
        self.variance_heads = nn.ModuleList(
            nn.Linear(d_model, self.n_channels) for _ in range(self.n_components)
        )
        # beta, secondary delay, tail decay, k_edge, k_second, k_tail.
        self.morphology_heads = nn.ModuleList(
            nn.Linear(d_model, 6) for _ in range(self.n_components)
        )
        self.morphology_variance_heads = nn.ModuleList(
            nn.Linear(d_model, 6) for _ in range(self.n_components)
        )

        self.atom_gate_logits = nn.Parameter(torch.full((self.n_components, 3), -1.5))
        self.null_log_variance_raw = nn.Parameter(torch.full((self.n_channels,), -1.0))
        for head in self.variance_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -1.0)
        for head in self.morphology_variance_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -1.0)
        for head in self.morphology_heads:
            nn.init.zeros_(head.bias)

    @property
    def morphology_parameter_names(self) -> tuple[str, ...]:
        return self._MORPHOLOGY_PARAMETER_NAMES

    def expected_l0(self) -> torch.Tensor:
        """Probability that each optional Hard-Concrete gate is non-zero."""

        offset = self.hard_concrete_temperature * math.log(
            -self.hard_concrete_gamma / self.hard_concrete_zeta
        )
        return torch.sigmoid(self.atom_gate_logits - offset)

    def _hard_concrete_gates(self, sample: bool) -> torch.Tensor:
        logits = self.atom_gate_logits
        if sample:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            relaxed = torch.sigmoid(
                (logits + uniform.log() - torch.log1p(-uniform)) / self.hard_concrete_temperature
            )
        else:
            relaxed = torch.sigmoid(logits)
        stretched = (
            relaxed * (self.hard_concrete_zeta - self.hard_concrete_gamma)
            + self.hard_concrete_gamma
        )
        return stretched.clamp(0.0, 1.0)

    @staticmethod
    def _normalize_atom(atom: torch.Tensor) -> torch.Tensor:
        scale = atom.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return atom / scale

    def _generalized_peak(
        self,
        times: torch.Tensor,
        tau: torch.Tensor,
        sigma_up: torch.Tensor,
        sigma_down: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        diff = times[None, None, :] - tau[..., None]
        smooth_width = (0.2 * (sigma_up + sigma_down)).clamp_min(2.0)
        side = torch.sigmoid(diff / smooth_width[..., None])
        sigma_t = sigma_up[..., None] + (sigma_down - sigma_up)[..., None] * side
        normalized = diff.abs() / sigma_t.clamp_min(1e-3)
        return torch.exp(-0.5 * normalized.clamp_min(1e-8).pow(beta[..., None]))

    def _edge_atom(self, peak: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dt = 1000.0 / self.sfreq
        derivative = torch.empty_like(peak)
        derivative[..., 1:-1] = (peak[..., 2:] - peak[..., :-2]) / (2.0 * dt)
        derivative[..., 0] = (peak[..., 1] - peak[..., 0]) / dt
        derivative[..., -1] = (peak[..., -1] - peak[..., -2]) / dt
        return self._normalize_atom(sigma[..., None] * derivative)

    def _tail_atom(
        self, times: torch.Tensor, tau: torch.Tensor, tail_decay: torch.Tensor
    ) -> torch.Tensor:
        diff = times[None, None, :] - tau[..., None]
        onset_width = (0.15 * tail_decay).clamp_min(4.0)
        positive_time = torch.nn.functional.softplus(diff / onset_width[..., None])
        positive_time = positive_time * onset_width[..., None]
        scaled = positive_time / tail_decay[..., None].clamp_min(1e-3)
        gamma_shape = 2.5
        tail = scaled.clamp_min(1e-8).pow(gamma_shape - 1.0) * torch.exp(-scaled)
        tail = tail * torch.sigmoid(diff / onset_width[..., None])
        return self._normalize_atom(tail)

    def _morphology_basis(
        self,
        theta: torch.Tensor,
        coefficients: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        times = self.times_ms.to(device=theta.device, dtype=theta.dtype)
        tau, sigma_up, sigma_down, beta, delay, tail_decay = theta.unbind(dim=-1)
        peak = self._generalized_peak(times, tau, sigma_up, sigma_down, beta)
        edge = self._edge_atom(peak, 0.5 * (sigma_up + sigma_down))
        second = self._generalized_peak(times, tau + delay, sigma_up, sigma_down, beta)
        tail = self._tail_atom(times, tau, tail_decay)
        optional = torch.stack((edge, second, tail), dim=-2)
        optional_weight = coefficients[..., :, None] * gates[None, ..., :, None]
        return peak + (optional_weight * optional).sum(dim=-2)

    def _decode_morphology_mean(
        self, H: torch.Tensor, tau: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.stack(
            [head(H[:, idx]) for idx, head in enumerate(self.morphology_heads)], dim=1
        )
        beta_lo, beta_hi = self.beta_bounds
        delay_lo, delay_hi = self.secondary_delay_bounds_ms
        tail_lo, tail_hi = self.tail_decay_bounds_ms
        beta = beta_lo + (beta_hi - beta_lo) * torch.sigmoid(raw[..., 0])
        delay = delay_lo + (delay_hi - delay_lo) * torch.sigmoid(raw[..., 1])
        tail_decay = tail_lo + (tail_hi - tail_lo) * torch.sigmoid(raw[..., 2])
        bounds = raw.new_tensor(self.coefficient_bounds)
        coefficients = torch.tanh(raw[..., 3:]) * bounds[None, None]
        theta = torch.stack(
            (
                tau,
                sigma[None, :, 0].expand_as(tau),
                sigma[None, :, 1].expand_as(tau),
                beta,
                delay,
                tail_decay,
            ),
            dim=-1,
        )
        return theta, coefficients

    def _decode_morphology_variance(self, H: torch.Tensor) -> torch.Tensor:
        raw = torch.stack(
            [head(H.detach()[:, idx]) for idx, head in enumerate(self.morphology_variance_heads)],
            dim=1,
        )
        lo = self.morph_log_variance_lo.to(device=H.device, dtype=H.dtype)
        hi = self.morph_log_variance_hi.to(device=H.device, dtype=H.dtype)
        log_variance = lo[None, None] + (hi - lo)[None, None] * torch.sigmoid(raw)
        return log_variance.exp()

    def _shape_delta_variance(
        self,
        theta: torch.Tensor,
        coefficients: torch.Tensor,
        gates: torch.Tensor,
        theta_variance: torch.Tensor,
    ) -> torch.Tensor:
        theta_const = theta.detach()
        coefficient_const = coefficients.detach()
        gate_const = gates.detach()
        steps = self.jacobian_steps.to(device=theta.device, dtype=theta.dtype)
        total = torch.zeros(
            theta.shape[0],
            theta.shape[1],
            self.n_time,
            device=theta.device,
            dtype=theta.dtype,
        )
        for index in range(theta.shape[-1]):
            delta = torch.zeros_like(theta_const)
            delta[..., index] = steps[index]
            plus = theta_const + delta
            minus = theta_const - delta
            plus[..., 1:3].clamp_(min=1.0)
            minus[..., 1:3].clamp_(min=1.0)
            plus[..., 3].clamp_(*self.beta_bounds)
            minus[..., 3].clamp_(*self.beta_bounds)
            plus[..., 4].clamp_(*self.secondary_delay_bounds_ms)
            minus[..., 4].clamp_(*self.secondary_delay_bounds_ms)
            plus[..., 5].clamp_(*self.tail_decay_bounds_ms)
            minus[..., 5].clamp_(*self.tail_decay_bounds_ms)
            derivative = (
                self._morphology_basis(plus, coefficient_const, gate_const)
                - self._morphology_basis(minus, coefficient_const, gate_const)
            ) / (2.0 * steps[index])
            total = total + derivative.square() * theta_variance[..., index, None]
        return total

    def forward(
        self,
        H: torch.Tensor,
        tau: torch.Tensor,
        sigma: torch.Tensor,
        *,
        epistemic_variance: torch.Tensor | None = None,
        sample_gates: bool | None = None,
    ) -> ERPComponentOutput:
        if H.dim() != 3 or H.shape[1] != self.n_components:
            raise ValueError(f"H must be (B,{self.n_components},D), got {tuple(H.shape)}.")
        if tau.shape != H.shape[:2]:
            raise ValueError(f"tau must be {tuple(H.shape[:2])}, got {tuple(tau.shape)}.")
        if sigma.shape != (self.n_components, 2):
            raise ValueError(f"sigma must be ({self.n_components},2), got {tuple(sigma.shape)}.")

        amplitude_mean = torch.stack(
            [head(H[:, i]) for i, head in enumerate(self.amplitude_heads)], dim=1
        )
        raw_variance = torch.stack(
            [head(H.detach()[:, i]) for i, head in enumerate(self.variance_heads)], dim=1
        )
        lo, hi = self.log_variance_bounds
        log_variance = lo + (hi - lo) * torch.sigmoid(raw_variance)
        amplitude_variance = log_variance.exp() + self.variance_floor
        theta, coefficients = self._decode_morphology_mean(H, tau, sigma)
        theta_variance = self._decode_morphology_variance(H)
        gates = self._hard_concrete_gates(
            self.training if sample_gates is None else bool(sample_gates)
        )
        basis = self._morphology_basis(theta, coefficients, gates)
        component_waveforms = amplitude_mean[:, :, :, None] * basis[:, :, None, :]
        reconstruction = component_waveforms.sum(dim=1)

        # Faithful variance path: only variance-head/null parameters receive
        # gradients from a variance objective.
        amplitude_const = amplitude_mean.detach()
        basis_const = basis.detach()
        amplitude_uncertainty = (
            amplitude_variance[:, :, :, None] * basis_const[:, :, None].square()
        ).sum(dim=1)
        basis_shape_variance = self._shape_delta_variance(
            theta, coefficients, gates, theta_variance
        )
        shape_uncertainty = (
            amplitude_const.square()[..., None] * basis_shape_variance[:, :, None]
        ).sum(dim=1)

        null_log_variance = lo + (hi - lo) * torch.sigmoid(self.null_log_variance_raw)
        null_variance = null_log_variance.exp() + self.variance_floor
        if epistemic_variance is None:
            epistemic = torch.zeros_like(reconstruction)
        else:
            if epistemic_variance.shape != reconstruction.shape:
                raise ValueError(
                    "epistemic_variance must match reconstruction shape; got "
                    f"{tuple(epistemic_variance.shape)} vs {tuple(reconstruction.shape)}."
                )
            epistemic = (
                epistemic_variance.detach()
                .to(device=reconstruction.device, dtype=reconstruction.dtype)
                .clamp_min(0.0)
            )
        waveform_variance = (
            null_variance[None, :, None] + amplitude_uncertainty + shape_uncertainty + epistemic
        )

        times = self.times_ms.to(device=H.device, dtype=H.dtype)
        component_peak = times[basis.argmax(dim=-1)]
        waveform_peak = times[reconstruction.argmax(dim=-1)]
        return ERPComponentOutput(
            reconstruction=reconstruction,
            component_waveforms=component_waveforms,
            amplitude_mean=amplitude_mean,
            amplitude_variance=amplitude_variance,
            null_variance=null_variance,
            waveform_variance=waveform_variance,
            morphology_basis=basis,
            morphology_parameters=theta,
            morphology_variance=theta_variance,
            atom_coefficients=coefficients,
            atom_gates=gates,
            expected_l0=self.expected_l0(),
            shape_uncertainty=shape_uncertainty,
            epistemic_variance=epistemic,
            anchor_latency_ms=tau,
            component_peak_latency_ms=component_peak,
            waveform_peak_latency_ms=waveform_peak,
        )
