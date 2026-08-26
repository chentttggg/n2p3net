from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.epochs import PreprocessingSpec
from data.moabb import prepare_moabb_p300


def test_moabb_adapter_applies_declared_artifact_threshold(monkeypatch) -> None:
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

    dataset = prepare_moabb_p300("FakeP300", preprocessing=profile)

    assert dataset.X.shape == (2, 2, 10)
    assert dataset.y.tolist() == [0, 1]
    assert dataset.subject_ids.tolist() == ["1", "2"]
    assert dataset.provenance["artifact_rejection"] == {
        "threshold_v": 1.0,
        "n_before": 3,
        "n_rejected": 1,
    }
