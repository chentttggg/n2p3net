"""Deterministic time masking for short P300 epochs.

SpellerSSL masks long time segments; FAME-style channel masking is deliberately
not implemented because the project montages are 3 or 8 channels and removing
one channel destroys most spatial evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskingConfig:
    mask_fraction: float = 0.5
    min_block_samples: int = 12
    max_block_samples: int = 32

    def validate(self, n_times: int) -> None:
        if not 0.0 < self.mask_fraction < 1.0:
            raise ValueError("mask_fraction must be in (0, 1).")
        if not 1 <= self.min_block_samples <= self.max_block_samples < n_times:
            raise ValueError(
                "mask block bounds must satisfy 1 <= min <= max < n_times."
            )


def make_temporal_mask(
    n_times: int,
    *,
    config: MaskingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a boolean mask of shape ``(n_times,)``.

    The mask is deterministic for a given ``generator``. Blocks are contiguous
    in time, which is the regime relevant to ERP morphology; independent sample
    masks would be unrealistically easy to reconstruct and would not test
    temporal context.
    """

    config.validate(n_times)
    target = int(round(config.mask_fraction * n_times))
    mask = torch.zeros(n_times, dtype=torch.bool)
    attempts = 0
    while int(mask.sum()) < target and attempts < 1000:
        block = int(
            torch.randint(
                config.min_block_samples,
                config.max_block_samples + 1,
                (1,),
                generator=generator,
            ).item()
        )
        block = min(block, target - int(mask.sum()), n_times - int(mask.sum()))
        start = int(
            torch.randint(0, n_times - block + 1, (1,), generator=generator).item()
        )
        mask[start : start + block] = True
        attempts += 1
    if int(mask.sum()) == 0:
        raise RuntimeError("mask generation produced an empty mask.")
    return mask


def apply_time_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero masked samples and return ``(masked_x, float_mask)``.

    ``x`` may be ``(B,C,T)`` or ``(C,T)``. The returned float mask has the same
    trailing time dimension so it can be pooled or embedded by a decoder.
    """

    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean.")
    if mask.ndim != 1 or x.shape[-1] != mask.shape[0]:
        raise ValueError("mask must be one-dimensional and match x.shape[-1].")
    keep = ~mask
    masked = x * keep.to(device=x.device, dtype=x.dtype)
    return masked, keep.to(device=x.device, dtype=x.dtype)
