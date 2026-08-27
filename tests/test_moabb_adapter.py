from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.epochs import PreprocessingSpec
from data.moabb import prepare_moabb_p300


def test_moabb_adapter_rejects_retired_fixed_artifact_threshold(monkeypatch) -> None:
    from data import moabb as adapter_module

    data = np.zeros((3, 2, 10), dtype=np.float64)
    data[1, 0, 3] = 2.0
    events = np.column_stack(
        [np.array([100, 200, 300]), np.zeros(3, dtype=int), np.array([1, 2, 1])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "Target"]),
                pd.DataFrame({"subject": [1, 1, 2]}),
            )

    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: SimpleNamespace(subject_list=[1, 2]))
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=1.0,
    )

    with pytest.raises(ValueError, match="Fixed absolute-voltage artifact rejection is retired"):
        prepare_moabb_p300("FakeP300", preprocessing=profile)


def test_moabb_adapter_rejects_retired_filter_before_subject_coverage(monkeypatch) -> None:
    from data import moabb as adapter_module

    data = np.zeros((3, 2, 10), dtype=np.float64)
    data[2, 0, 3] = 2.0
    events = np.column_stack(
        [np.array([100, 200, 300]), np.zeros(3, dtype=int), np.array([1, 2, 1])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            pass

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "NonTarget"]),
                pd.DataFrame({"subject": [1, 1, 2]}),
            )

    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: SimpleNamespace(subject_list=[1, 2]),
    )
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_subject_loss",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=1.0,
    )

    with pytest.raises(ValueError, match="Fixed absolute-voltage artifact rejection is retired"):
        prepare_moabb_p300("FakeP300", preprocessing=profile)


def test_moabb_adapter_uses_explicit_candidates_not_binary_event_codes(monkeypatch) -> None:
    from data import moabb as adapter_module

    data = np.zeros((4, 2, 10), dtype=np.float64)
    events = np.column_stack(
        [np.arange(100, 500, 100), np.zeros(4, dtype=int), np.array([7, 8, 7, 8])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "NonTarget", "Target"]),
                pd.DataFrame(
                    {
                        "subject": [1] * 4,
                        "selection_id": ["choice-1"] * 4,
                        "candidate_id": ["left", "right", "left", "right"],
                        "target_candidate_id": ["right"] * 4,
                        "repetition_index": np.array([0, 0, 1, 1], dtype=np.int64),
                    }
                ),
            )

    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: SimpleNamespace(subject_list=[1]),
    )
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_candidates",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=None,
    )

    dataset = prepare_moabb_p300("FakeCandidateP300", preprocessing=profile)

    timeline = dataset.event_timeline
    assert timeline.stimulus_ids.tolist() == [7, 8, 7, 8]
    assert timeline.candidate_ids.tolist() == ["left", "right", "left", "right"]
    assert timeline.has_repetition_structure is True
    assert timeline.supports_full_candidate_chain is False
