"""Finite MNE signals and in-bounds events for ingestion tests."""

import mne
import numpy as np

from data.channel import STANDARD_CHANNELS


def make_raw(sfreq=512.0, n_seconds=20.0, ch_names=STANDARD_CHANNELS, amp=10e-6, dc=0.0, seed=0):
    """Construct synthetic finite EEG samples."""
    rng = np.random.default_rng(seed)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((len(ch_names), n_times)) * amp + dc
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_events(sfreq=512.0, n_seconds=20.0, first=2.0, step=1.0):
    """构造刺激 events（单一类型 id=1），保证每个 epoch 都落在数据范围内。"""
    times = np.arange(first, n_seconds - 1.0, step)  # final event still leaves >1.2 s
    samples = np.round(times * sfreq).astype(int)
    return np.column_stack(
        [samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)]
    )
