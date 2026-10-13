"""Training artifacts retain target scope even when metadata is removed."""

import json

import numpy as np
import pytest

from train.source_sampling import load_pds_selection_record, select_pds_rows


@pytest.mark.parametrize("target_record", [None, [], "calibration", {}])
def test_training_artifact_rejects_absent_or_nonrecord_target_scope(tmp_path, target_record):
    selection = select_pds_rows(
        np.arange(8, dtype=float).reshape(4, 2), np.ones((2, 2)),
        n_select=2, selection_scope="target_calibration", seed=7,
    )
    record = selection.record()
    if target_record is not None:
        record["target_feature_record"] = target_record
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="target-calibration scope"):
        load_pds_selection_record(path)


def test_training_artifact_rejects_explicit_null_target_record(tmp_path):
    selection = select_pds_rows(
        np.arange(8, dtype=float).reshape(4, 2), np.ones((2, 2)),
        n_select=2, selection_scope="target_calibration", seed=7,
    )
    record = {**selection.record(), "target_feature_record": None}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="target-calibration scope"):
        load_pds_selection_record(path)
