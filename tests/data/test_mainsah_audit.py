from __future__ import annotations

import numpy as np
import pytest

from experiments.audit_mainsah_s2 import (
    compare_waveforms,
    parse_original_identity,
    validate_events,
)


def test_original_roles_come_from_paths_including_nonsequential_file_numbers():
    identity = parse_original_identity("S2_04/SE001/Train/House/S2_04_SE001_House_Train04.edf")
    assert identity == {
        "subject": 4,
        "session": "SE001",
        "role": "Train",
        "condition": "House",
        "original_record_number": 4,
    }
    with pytest.raises(ValueError, match="Inconsistent"):
        parse_original_identity("S2_04/SE001/Test/House/S2_04_SE001_House_Train04.edf")


def test_one_sample_event_shift_and_label_flip_are_rejected():
    begin = np.array([0, 1, 1, 0, 1, 0])
    types = np.array([0, 0, 0, 0, 1, 0])
    rows = [
        {"sample": "1", "onset": str(1 / 256), "trial_type": "NonTarget", "value": "1"},
        {"sample": "4", "onset": str(4 / 256), "trial_type": "Target", "value": "2"},
    ]
    assert validate_events(begin, types, rows, 256, 6)["event_samples_and_labels_exact"]
    shifted = [dict(row) for row in rows]
    shifted[0]["sample"] = "2"
    with pytest.raises(ValueError, match="sample indices"):
        validate_events(begin, types, shifted, 256, 6)
    flipped = [dict(row) for row in rows]
    flipped[0]["trial_type"] = "Target"
    with pytest.raises(ValueError, match="labels/values"):
        validate_events(begin, types, flipped, 256, 6)


def test_quantization_bound_rejects_gain_change_even_with_perfect_correlation():
    original = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]) * 1e-6
    quantum = np.array([1e-11, 1e-11])
    assert compare_waveforms(original, original + 0.4e-11, quantum)[
        "all_channels_within_half_quantum"
    ]
    with pytest.raises(ValueError, match="quantization bound"):
        compare_waveforms(original, original * 1.01, quantum)
    with pytest.raises(ValueError, match="quantization bound"):
        compare_waveforms(original, original[::-1], quantum)
