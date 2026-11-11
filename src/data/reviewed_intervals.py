"""Immutable, content-bound annotations from an explicit recording review."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path


@dataclass(frozen=True)
class ReviewedInterval:
    start_sample: int
    stop_sample: int
    channels: tuple[str, ...]
    reason: str

    def __post_init__(self):
        if (not isinstance(self.start_sample, Integral) or isinstance(self.start_sample, bool)
                or not isinstance(self.stop_sample, Integral) or isinstance(self.stop_sample, bool)
                or not 0 <= self.start_sample < self.stop_sample or not self.channels
                or len(set(self.channels)) != len(self.channels)
                or any(not isinstance(c, str) or not c.strip() for c in self.channels)
                or not isinstance(self.reason, str) or not self.reason.startswith("reviewed_")):
            raise ValueError("invalid reviewed interval support, channels or reason")
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "start_sample", int(self.start_sample))
        object.__setattr__(self, "stop_sample", int(self.stop_sample))


@dataclass(frozen=True)
class ReviewedRecording:
    raw_sha256: str
    n_samples: int
    session_id: str
    authority: str
    intervals: tuple[ReviewedInterval, ...]

    def __post_init__(self):
        if (not isinstance(self.raw_sha256, str) or len(self.raw_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.raw_sha256)
                or not isinstance(self.n_samples, Integral) or isinstance(self.n_samples, bool) or self.n_samples < 1
                or not isinstance(self.session_id, str) or not self.session_id.strip()
                or not isinstance(self.authority, str) or not self.authority.strip()
                or any(not isinstance(r, ReviewedInterval) or r.stop_sample > self.n_samples for r in self.intervals)):
            raise ValueError("invalid reviewed recording identity or annotation authority")
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "n_samples", int(self.n_samples))

    def record(self):
        record = asdict(self)
        record["intervals"] = [{**asdict(r), "channels": list(r.channels)} for r in self.intervals]
        return record

    @classmethod
    def from_record(cls, record):
        return cls(**{**record, "intervals": tuple(ReviewedInterval(**r) for r in record["intervals"])})


def match_reviewed_recording(raw, recordings):
    if not recordings:
        return None
    paths = raw.filenames
    if len(paths) != 1 or paths[0] is None:
        raise ValueError("reviewed annotations require the original single source file identity")
    with Path(paths[0]).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    matched = next((r for r in recordings if r.raw_sha256 == digest), None)
    if matched is not None and matched.n_samples != raw.n_times:
        raise ValueError("reviewed source file sample support differs from the recording")
    return matched
