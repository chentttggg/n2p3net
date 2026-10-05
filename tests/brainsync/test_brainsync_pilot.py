from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.run_brainsync_pilot import validate_pilot_scopes


def _scope(sessions, times, available):
    return SimpleNamespace(
        X=np.zeros((len(sessions), 2, 3)), y=np.arange(len(sessions)) % 2,
        subject_ids=np.repeat("person", len(sessions)),
        metadata=pd.DataFrame({"session": sessions, "session_start_timestamp_s": times,
                               "calibration_available_timestamp_s": available,
                               "trial_id": [f"{session}-0" for session in sessions]}),
    )


def test_pilot_rejects_session_leakage_late_labels_and_changed_refit_rows():
    calibration = _scope(["a", "b"], [0, 10], [5, 15])
    test = _scope(["c"], [20], [25])
    full = _scope(["a", "b", "c"], [0, 10, 20], [5, 15, 25])
    validate_pilot_scopes(calibration, test, full)
    with pytest.raises(ValueError, match="overlap"):
        validate_pilot_scopes(calibration, calibration, full)
    calibration.metadata.loc[1, "calibration_available_timestamp_s"] = 21
    with pytest.raises(ValueError, match="before"):
        validate_pilot_scopes(calibration, test, full)
    calibration.metadata.loc[1, "calibration_available_timestamp_s"] = 15
    full.y[2] = 1
    with pytest.raises(ValueError, match="rows differ"):
        validate_pilot_scopes(calibration, test, full)
