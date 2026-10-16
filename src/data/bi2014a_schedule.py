"""Dependency-light BI2014a flash-code and raw-label contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BIFlashCodeFamily:
    axis: str
    is_target: bool
    first_code: int
    last_code: int

    def __post_init__(self) -> None:
        if self.axis not in {"row", "column"}:
            raise ValueError(f"unsupported BI flash axis {self.axis!r}.")
        if self.last_code < self.first_code:
            raise ValueError("BI flash code family is empty.")

    def contains(self, code: int) -> bool:
        return self.first_code <= code <= self.last_code


@dataclass(frozen=True)
class BIFlashEvent:
    flash_code: int
    axis: str
    candidate_index: int
    is_target: bool

    @property
    def candidate_key(self) -> str:
        return f"{self.axis}:{self.candidate_index}"


@dataclass(frozen=True)
class BIFlashLabelMismatch:
    event_index: int
    flash_sample: int
    flash_code: int
    raw_label: int
    expected_raw_label: int


@dataclass(frozen=True)
class BIFlashLabelAudit:
    stage: str
    n_events: int
    n_mismatches: int
    mismatch_examples: tuple[BIFlashLabelMismatch, ...]
    observed_code_label_counts: tuple[tuple[str, int], ...]

    def to_record(self, contract: BIFlashScheduleContract) -> dict[str, object]:
        return {
            "schema": contract.schema,
            "stage": self.stage,
            "n_events": self.n_events,
            "n_mismatches": self.n_mismatches,
            "mismatch_examples": [
                {
                    "event_index": item.event_index,
                    "flash_sample": item.flash_sample,
                    "flash_code": item.flash_code,
                    "raw_label": item.raw_label,
                    "expected_raw_label": item.expected_raw_label,
                }
                for item in self.mismatch_examples
            ],
            "raw_label_codebook": {
                str(label): meaning for label, meaning in contract.raw_label_codebook
            },
            "flash_codebook": contract.codebook_record(),
            "observed_code_label_counts": dict(self.observed_code_label_counts),
        }


class BIFlashLabelContractError(ValueError):
    def __init__(self, audit: BIFlashLabelAudit, contract: BIFlashScheduleContract):
        self.audit = audit
        examples = ", ".join(
            f"sample={item.flash_sample} code={item.flash_code} "
            f"raw={item.raw_label} expected={item.expected_raw_label}"
            for item in audit.mismatch_examples
        )
        super().__init__(
            f"BI flash/raw-label mismatch at {audit.stage}: "
            f"{audit.n_mismatches}/{audit.n_events} events; {examples}"
        )
        self.contract = contract


@dataclass(frozen=True)
class BIFlashScheduleContract:
    """Single decoder for BI2014a row/column flash semantics."""

    schema: str = "bi2014a_flash_schedule/1"
    grid_size: int = 6
    families: tuple[BIFlashCodeFamily, ...] = (
        BIFlashCodeFamily("row", False, 20, 25),
        BIFlashCodeFamily("column", False, 40, 45),
        BIFlashCodeFamily("row", True, 60, 65),
        BIFlashCodeFamily("column", True, 80, 85),
    )
    raw_label_codebook: tuple[tuple[int, str], ...] = (
        (1, "non_target"),
        (2, "target"),
    )

    def decode(self, flash_code: int) -> BIFlashEvent:
        try:
            code = int(flash_code)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"flash_code {flash_code!r} is not an integer code.") from error
        if code != flash_code:
            raise ValueError(f"flash_code {flash_code!r} is not an integer code.")
        for family in self.families:
            if family.contains(code):
                candidate_index = code - family.first_code
                if not 0 <= candidate_index < self.grid_size:
                    raise ValueError(
                        f"flash_code {code} decodes outside the {self.grid_size}x"
                        f"{self.grid_size} candidate grid."
                    )
                return BIFlashEvent(
                    flash_code=code,
                    axis=family.axis,
                    candidate_index=candidate_index,
                    is_target=family.is_target,
                )
        raise ValueError(f"flash_code {code} is outside the BI2014a schedule.")

    def codebook_record(self) -> list[dict[str, object]]:
        return [
            {
                "axis": family.axis,
                "is_target": family.is_target,
                "first_code": family.first_code,
                "last_code": family.last_code,
            }
            for family in self.families
        ]

    def record(self) -> dict[str, object]:
        """Return the single semantic record shared by BI producers and consumers."""

        return {
            "schema": self.schema,
            "grid_size": self.grid_size,
            "codebook": self.codebook_record(),
            "raw_label_codebook": [list(item) for item in self.raw_label_codebook],
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.record(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def target_mask(self, flash_codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(flash_codes)
        if codes.ndim != 1:
            raise ValueError("BI flash codes must be one-dimensional.")
        return np.asarray([self.decode(code).is_target for code in codes], dtype=bool)

    def audit_raw_labels(
        self,
        flash_codes: np.ndarray,
        raw_labels: np.ndarray,
        *,
        flash_samples: np.ndarray,
        stage: str,
    ) -> BIFlashLabelAudit:
        codes = np.asarray(flash_codes)
        labels = np.asarray(raw_labels)
        samples = np.asarray(flash_samples)
        if not (codes.ndim == labels.ndim == samples.ndim == 1):
            raise ValueError("BI flash audit arrays must be one-dimensional.")
        if not (len(codes) == len(labels) == len(samples)):
            raise ValueError("BI flash audit arrays must be aligned.")
        mismatches: list[BIFlashLabelMismatch] = []
        observed: dict[str, int] = {}
        for event_index, (code, raw_label, sample) in enumerate(
            zip(codes, labels, samples, strict=True)
        ):
            decoded = self.decode(code)
            if not np.isfinite(raw_label) or int(raw_label) != raw_label:
                raw_label_int = -1
            else:
                raw_label_int = int(raw_label)
            expected = 2 if decoded.is_target else 1
            observed_key = f"code={decoded.flash_code},raw_label={raw_label!s}"
            observed[observed_key] = observed.get(observed_key, 0) + 1
            if raw_label_int != expected:
                mismatches.append(
                    BIFlashLabelMismatch(
                        event_index=event_index,
                        flash_sample=int(sample),
                        flash_code=decoded.flash_code,
                        raw_label=raw_label_int,
                        expected_raw_label=expected,
                    )
                )
        audit = BIFlashLabelAudit(
            stage=str(stage),
            n_events=len(codes),
            n_mismatches=len(mismatches),
            mismatch_examples=tuple(mismatches[:16]),
            observed_code_label_counts=tuple(sorted(observed.items())),
        )
        if mismatches:
            raise BIFlashLabelContractError(audit, self)
        return audit


BI2014A_FLASH_SCHEDULE = BIFlashScheduleContract()


__all__ = [
    "BI2014A_FLASH_SCHEDULE",
    "BIFlashCodeFamily",
    "BIFlashEvent",
    "BIFlashLabelAudit",
    "BIFlashLabelContractError",
    "BIFlashLabelMismatch",
    "BIFlashScheduleContract",
]
