"""Map software event times onto explicitly discontinuous stored samples."""

from __future__ import annotations

import math

SEGMENTED_TIME_BASE = "segmented_recording"
SAMPLE_CLOCK_SCHEMA = "brainsync-sample-clock/1"


class SampleClockLedger:
    """Bind received packet evidence to stored rows without claiming ADC time."""

    def __init__(self, sample_rate_hz: float, *, counter_modulus: int = 65536) -> None:
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0 or counter_modulus < 2:
            raise ValueError("Invalid sample clock geometry.")
        self.sample_rate_hz = float(sample_rate_hz)
        self.counter_modulus = int(counter_modulus)
        self.received_samples = 0
        self.stored_samples = 0
        self.batches: list[dict] = []
        self.segments: list[dict] = []
        self._last_sequence: int | None = None
        self._pending: dict | None = None
        self._active_segment: dict | None = None

    def observe_batch(self, sequence: list, delta_time: list, receive_monotonic_ns: int) -> dict:
        if self._pending is not None:
            raise ValueError("Previous clock batch has not been committed.")
        if not sequence or len(sequence) != len(delta_time) or receive_monotonic_ns < 0:
            raise ValueError("Clock batch geometry is invalid.")
        statuses = []
        missing = 0
        for value in sequence:
            if value is None:
                statuses.append("unavailable")
                self._last_sequence = None
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.counter_modulus:
                statuses.append("invalid_counter")
                self._last_sequence = None
                continue
            if self._last_sequence is None:
                statuses.append("first")
            else:
                advance = (value - self._last_sequence) % self.counter_modulus
                if advance == 1:
                    statuses.append("contiguous")
                elif advance == 0:
                    statuses.append("duplicate")
                elif advance < self.counter_modulus // 2:
                    statuses.append("known_gap")
                    missing += advance - 1
                else:
                    statuses.append("out_of_order_or_reset")
            self._last_sequence = value
        record = {
            "batch_id": len(self.batches), "received_start": self.received_samples,
            "n_received": len(sequence), "seq_num": list(sequence),
            "delta_time_raw": list(delta_time), "delta_time_unit": "10us_sdk_declared",
            "delta_time_semantics_verified": False,
            "host_receive_monotonic_ns": int(receive_monotonic_ns),
            "sequence_status": statuses,
            "sequence_valid": all(value in {"first", "contiguous", "unavailable"} for value in statuses),
            "sequence_observed": all(value != "unavailable" for value in statuses),
            "known_missing_samples": missing,
        }
        self.received_samples += len(sequence)
        self._pending = record
        return record

    def commit_batch(self, batch: dict, *, stored_start: int | None) -> None:
        if batch is not self._pending:
            raise ValueError("Clock batch is not the pending observation.")
        if stored_start is not None:
            if not batch["sequence_valid"] or stored_start != self.stored_samples:
                raise ValueError("Discontinuous or misaligned batch cannot be stored as contiguous EEG.")
            if self._active_segment is None:
                first_ns = batch["host_receive_monotonic_ns"] - round(
                    (batch["n_received"] - 1) * 1e9 / self.sample_rate_hz
                )
                self._active_segment = {
                    "segment_id": len(self.segments), "stored_start": stored_start,
                    "stored_stop": stored_start, "first_sample_estimate_monotonic_ns": first_ns,
                    "anchor_source": "first_batch_delivery_minus_nominal_span",
                    "relative_rate_assumption": "nominal_sample_rate",
                    "delivery_residual_min_seconds": 0.0, "delivery_residual_max_seconds": 0.0,
                }
                self.segments.append(self._active_segment)
            self.stored_samples += batch["n_received"]
            self._active_segment["stored_stop"] = self.stored_samples
            estimate = self._active_segment["first_sample_estimate_monotonic_ns"] / 1e9 + (
                self.stored_samples - 1 - self._active_segment["stored_start"]
            ) / self.sample_rate_hz
            residual = batch["host_receive_monotonic_ns"] / 1e9 - estimate
            for name, function in (("min", min), ("max", max)):
                key = f"delivery_residual_{name}_seconds"
                self._active_segment[key] = function(self._active_segment[key], residual)
            batch["segment_id"] = self._active_segment["segment_id"]
        else:
            batch["segment_id"] = None
        batch["stored_start"] = stored_start
        batch["n_stored"] = 0 if stored_start is None else batch["n_received"]
        self.batches.append(dict(batch))
        self._pending = None

    def close_segment(self, wall_monotonic: float) -> None:
        if self._active_segment is not None:
            self._active_segment["closed_monotonic_ns"] = round(wall_monotonic * 1e9)
            self._active_segment = None

    @property
    def first_sample_monotonic(self) -> float | None:
        if not self.segments:
            return None
        return self.segments[0]["first_sample_estimate_monotonic_ns"] / 1e9

    def source_sample(self, wall_monotonic: float) -> int:
        return self.source_position(wall_monotonic)[0]

    def source_position(self, wall_monotonic: float) -> tuple[int, int]:
        if not math.isfinite(wall_monotonic) or not self.segments:
            raise ValueError("No received sample anchor is available.")
        stamp = round(wall_monotonic * 1e9)
        segment = self.segments[0]
        for item in self.segments:
            if item["first_sample_estimate_monotonic_ns"] <= stamp:
                segment = item
        estimate = segment["stored_start"] + round(
            (stamp - segment["first_sample_estimate_monotonic_ns"]) * self.sample_rate_hz / 1e9
        )
        if "closed_monotonic_ns" in segment:
            estimate = min(estimate, segment["stored_stop"])
        return max(segment["stored_start"], estimate), segment["segment_id"]

    def record(self) -> dict:
        observed = bool(self.batches) and all(item["sequence_observed"] for item in self.batches)
        return {
            "schema": SAMPLE_CLOCK_SCHEMA, "sample_rate_hz": self.sample_rate_hz,
            "counter_modulus": self.counter_modulus, "received_samples": self.received_samples,
            "stored_samples": self.stored_samples, "batch_count": len(self.batches),
            "alignment_level": "delivery_anchored", "absolute_offset_bound_seconds": None,
            "sequence_observed": observed,
            "sequence_continuity_verified": observed and all(item["sequence_valid"] for item in self.batches),
            "known_missing_samples": sum(item["known_missing_samples"] for item in self.batches),
            "segments": [dict(segment) for segment in self.segments],
        }


def stored_time(wall_seconds: float, pauses: list[dict], sample_rate: float) -> float:
    wall_anchor = 0.0
    stored_anchor = 0.0
    for pause in pauses:
        boundary = float(pause["boundary_sample"]) / sample_rate
        if wall_seconds <= pause["start_wall_seconds"]:
            return min(boundary, stored_anchor + max(0.0, wall_seconds - wall_anchor))
        if pause.get("end_wall_seconds") is None or wall_seconds <= pause["end_wall_seconds"]:
            return boundary
        wall_anchor = float(pause["end_wall_seconds"])
        stored_anchor = boundary
    return stored_anchor + max(0.0, wall_seconds - wall_anchor)
