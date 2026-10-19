"""Synthetic official-shape BNCI MAT fields for parser and continuous tests."""
import numpy as np

from data.bnci2014_008_candidate import (
    BNCI2014_008_CHANNELS,
    BNCI2014_008_SAMPLE_COUNT,
    BNCI2014_008_SELECTION_COUNT,
)

FIRST_TRIAL_SAMPLE = 1800
TRIAL_STRIDE = 9728
FLASH_STRIDE = 64
FLASH_DURATION = 32


def bnci_mat_fields():
    X = np.zeros((BNCI2014_008_SAMPLE_COUNT, 8), dtype=np.float32)
    y = np.zeros((BNCI2014_008_SAMPLE_COUNT, 1), dtype=np.float32)
    y_stim = np.zeros((BNCI2014_008_SAMPLE_COUNT, 1), dtype=np.int16)
    trial_starts = FIRST_TRIAL_SAMPLE + np.arange(BNCI2014_008_SELECTION_COUNT) * TRIAL_STRIDE
    trial = (trial_starts + 1).reshape(1, -1).astype(np.float64)
    for selection, start in enumerate(trial_starts):
        target_row, target_col = selection % 6, (selection + 2) % 6
        for repetition in range(10):
            order = np.roll(np.arange(12), repetition)
            for position, candidate in enumerate(order):
                onset = int(start + (repetition * 12 + position) * FLASH_STRIDE)
                stop = onset + FLASH_DURATION
                y_stim[onset:stop, 0] = int(candidate + 1)
                is_target = candidate == target_row or candidate == 6 + target_col
                y[onset:stop, 0] = 2.0 if is_target else 1.0
    return {
        "channels": np.asarray(BNCI2014_008_CHANNELS, dtype=object).reshape(1, -1),
        "X": X, "y": y, "y_stim": y_stim, "trial": trial,
        "classes": np.asarray(["NonTarget", "Target"], dtype=object).reshape(1, -1),
        "classes_stim": np.asarray([*(f"Row{i}" for i in range(1, 7)), *(f"Col{i}" for i in range(1, 7))],
                                    dtype=object).reshape(1, -1),
        "gender": "synthetic", "age": "0", "ALSfrs": "0", "onsetALS": "synthetic",
    }
