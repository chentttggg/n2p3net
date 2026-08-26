"""Single-source training batch configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingBatchConfig:
    """Describe the physical micro-batch and its accumulated effective batch.

    ``physical_batch_size`` is the largest tensor held by one forward/backward
    pass. ``effective_batch_size`` is the optimizer batch for trial losses and
    must be an exact multiple so ``accumulation_steps`` is deterministic.
    """

    physical_batch_size: int = 256
    effective_batch_size: int = 256

    def __post_init__(self) -> None:
        physical = int(self.physical_batch_size)
        effective = int(self.effective_batch_size)
        if physical < 1:
            raise ValueError("physical_batch_size must be positive.")
        if effective < physical:
            raise ValueError("effective_batch_size must be at least physical_batch_size.")
        if effective % physical:
            raise ValueError(
                "effective_batch_size must be an exact multiple of physical_batch_size."
            )
        object.__setattr__(self, "physical_batch_size", physical)
        object.__setattr__(self, "effective_batch_size", effective)

    @property
    def accumulation_steps(self) -> int:
        """Number of physical batches represented by one optimizer batch."""

        return self.effective_batch_size // self.physical_batch_size

    def trainer_overrides(self) -> dict[str, int]:
        """Return the legacy TrainerConfig fields from this batch contract."""

        return {
            "batch_size": self.physical_batch_size,
            "accum_steps": self.accumulation_steps,
        }

    def record(self) -> dict[str, int]:
        """Return an audit-friendly serialized batch contract."""

        return {
            **asdict(self),
            "accumulation_steps": self.accumulation_steps,
        }

    @classmethod
    def from_cli(
        cls,
        *,
        physical_batch_size: int,
        effective_batch_size: int | None = None,
        accum_steps: int | None = None,
        default: TrainingBatchConfig | None = None,
    ) -> TrainingBatchConfig:
        """Build a contract from the public pair of knobs.

        ``accum_steps`` remains a compatibility-only escape hatch for older
        commands. New callers should set ``effective_batch_size`` instead.
        """

        fallback = default or cls()
        if effective_batch_size is not None and accum_steps is not None:
            raise ValueError("set effective_batch_size or accum_steps, not both.")
        if accum_steps is not None:
            if int(accum_steps) < 1:
                raise ValueError("accum_steps must be positive.")
            effective_batch_size = int(physical_batch_size) * int(accum_steps)
        if effective_batch_size is None:
            effective_batch_size = fallback.effective_batch_size
        return cls(
            physical_batch_size=physical_batch_size,
            effective_batch_size=effective_batch_size,
        )


DEFAULT_TRAINING_BATCH = TrainingBatchConfig()
