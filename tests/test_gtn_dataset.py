from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from data.gtn_dataset import _prepare_gtn_experiment
from experiments.prepare_gtn_dataset import _derive_n_times


def test_gtn_n_times_uses_the_shared_exclusive_right_edge_rule() -> None:
    assert _derive_n_times(256.0, -200.0, 800.0) == 256
    assert _derive_n_times(128.0, 0.0, 1000.0) == 128


def test_gtn_preparation_preserves_the_complete_nine_choice_ledger(tmp_path) -> None:
    experiment = tmp_path / "Experiment_001_P3_Numbers"
    experiment.mkdir()
    scheduled_digits = np.tile(np.arange(1, 10, dtype=np.int64), 2)
    result = SimpleNamespace(
        data=np.zeros((len(scheduled_digits), 3, 256), dtype=np.float32),
        channel_names=("Fz", "Cz", "Pz"),
        channel_positions_m=np.ones((3, 3), dtype=np.float32),
        channel_mask=np.ones(3, dtype=bool),
        event_samples=np.arange(len(scheduled_digits), dtype=np.int64),
        event_times_s=np.arange(len(scheduled_digits), dtype=float),
        evidence_available_times_s=np.arange(len(scheduled_digits), dtype=float) + 0.8,
        event_evidence_indices=np.arange(len(scheduled_digits), dtype=np.int64),
        event_statuses=np.repeat("available", len(scheduled_digits)),
        event_status_details=np.repeat("", len(scheduled_digits)),
        event_indices=np.arange(len(scheduled_digits), dtype=np.int64),
        online_causal=False,
    )
    gtn = SimpleNamespace(
        events=np.column_stack(
            [np.arange(len(scheduled_digits), dtype=np.int64), np.zeros(len(scheduled_digits)), scheduled_digits]
        ).astype(np.int64),
        thought_number=4,
        subject_id="participant-1",
    )

    prepared = _prepare_gtn_experiment(experiment, gtn, result)

    assert prepared.timeline.supports_full_candidate_chain is True
    assert prepared.y.sum() == 2
    assert set(prepared.metadata["candidate_id"]) == {str(value) for value in range(1, 10)}
    repetitions = prepared.timeline.repetition_indices.reshape(2, 9)
    assert np.array_equal(repetitions[0], np.zeros(9, dtype=np.int64))
    assert np.array_equal(repetitions[1], np.ones(9, dtype=np.int64))
