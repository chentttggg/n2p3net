from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.epochs import PreprocessingSpec
from data.moabb import prepare_moabb_p300


def _fake_dataset(subjects=(1, 2)):
    acquisition = SimpleNamespace(
        sampling_rate=100.0,
        reference="right earlobe",
        ground="Fz",
    )
    return SimpleNamespace(
        subject_list=list(subjects),
        metadata=SimpleNamespace(acquisition=acquisition),
    )
def test_moabb_adapter_builds_causal_cache_from_raw_runs(monkeypatch) -> None:
    from data import moabb as adapter_module

    info = mne.create_info(["Fz", "Cz"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info, verbose=False)
    raw.set_montage("standard_1005")
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 4.0],
            duration=[0.0] * 4,
            description=["NonTarget", "Target", "NonTarget", "Target"],
        )
    )
    dataset = _fake_dataset((1,))
    dataset.event_id = {"NonTarget": 1, "Target": 2}
    dataset.get_data = lambda subjects: {1: {"0": {"0": raw}}}
    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: dataset)
    profile = PreprocessingSpec(
        name="causal_mock",
        sfreq=100.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=100,
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )

    prepared = prepare_moabb_p300("FakeP300", preprocessing=profile)

    assert prepared.X.shape == (4, 2, 100)
    assert prepared.preprocessing.filter_phase == "forward"
    assert prepared.event_timeline.online_causal is True





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
        times = np.arange(10, dtype=float) / 100.0

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

    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: _fake_dataset())
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


def test_moabb_adapter_executes_declared_mean_baseline(monkeypatch) -> None:
    from data import moabb as adapter_module

    data = np.zeros((4, 2, 30), dtype=np.float64)
    data[:, 0] += np.arange(4, dtype=float)[:, None] + 5.0
    data[:, 1] += np.arange(4, dtype=float)[:, None] - 3.0
    data[:, :, 20:] += 2.0
    events = np.column_stack(
        [np.arange(100, 500, 100), np.zeros(4, dtype=int), np.array([1, 2, 1, 2])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = -0.2 + np.arange(30, dtype=float) / 100.0

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
                pd.DataFrame({"subject": [1, 1, 2, 2]}),
            )

    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: _fake_dataset(),
    )
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_executed_baseline",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=100.0,
        n_times=30,
        baseline_mode="mean_only",
        reject_threshold_v=None,
    )

    dataset = prepare_moabb_p300("FakeP300", preprocessing=profile)

    np.testing.assert_allclose(dataset.X[:, :, :20].mean(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(dataset.X[:, :, 20:], 2.0, atol=1e-12)
    assert dataset.provenance["epoch_baseline"]["mode"] == "mean_only"


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
        times = np.arange(10, dtype=float) / 100.0

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
        lambda name: _fake_dataset(),
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
        times = np.arange(10, dtype=float) / 100.0

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
        lambda name: _fake_dataset((1,)),
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
