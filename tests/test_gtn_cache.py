from __future__ import annotations

import numpy as np
import pytest

from data.events import ScheduledEventTimeline
from experiments.run_gtn_baseline import _load_gtn_cache, _save_gtn_cache


def _cache_timeline() -> ScheduledEventTimeline:
    return ScheduledEventTimeline(
        event_ids=np.asarray(["e0", "e1", "e2"]),
        group_ids=np.asarray(["s", "s", "s"]),
        subject_ids=np.asarray(["s", "s", "s"]),
        stimulus_ids=np.asarray([1, 2, 3]),
        onset_samples=np.asarray([0, 10, 20]),
        onset_times_s=np.asarray([0.0, 0.1, 0.2]),
        evidence_available_times_s=np.asarray([0.8, np.nan, 1.0]),
        evidence_indices=np.asarray([0, -1, 1]),
        statuses=np.asarray(["available", "artifact_rejected", "available"]),
        status_details=np.asarray(["", "threshold", ""]),
        dataset_ids=np.repeat("gtn", 3),
        session_ids=np.repeat("", 3),
        run_ids=np.repeat("run", 3),
        selection_ids=np.repeat("s", 3),
        complete=True,
        online_causal=False,
        timing_source="synthetic_gtn",
    ).validate(n_epochs=2)


def test_gtn_v2_cache_round_trip_preserves_complete_timeline(tmp_path) -> None:
    path = tmp_path / "gtn.npz"
    X = np.zeros((2, 3, 8), dtype=np.float32)
    y = np.asarray([1, 0])
    digits = np.asarray([1, 3])
    subjects = np.asarray(["s", "s"])
    timeline = _cache_timeline()

    _save_gtn_cache(path, X, y, digits, subjects, {"s": 1}, ["quality-note"], timeline)
    loaded = _load_gtn_cache(path)

    assert np.array_equal(loaded[0], X)
    assert np.array_equal(loaded[2], digits)
    assert loaded[4] == {"s": 1}
    assert loaded[5] == ["quality-note"]
    assert loaded[6].n_events == 3 and loaded[6].n_available == 2
    assert loaded[6].fingerprint(truth={"s": 1}) == timeline.fingerprint(truth={"s": 1})
    assert not path.with_suffix(".tmp.npz").exists()


def test_gtn_loader_rejects_legacy_cache(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, X=np.zeros((1, 3, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="Unsupported GTN cache schema"):
        _load_gtn_cache(path)
