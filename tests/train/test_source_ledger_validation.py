"""Source ledgers validate each current dataset once, without cross-call caching."""

from dataclasses import replace

import numpy as np
import pytest

from data.epochs import epoch_row_identity_digest
from research.source_selection import source_fit_ledger
from tests.support.source_programs import source_early_stopping_data


@pytest.mark.parametrize("order", ["full", "subset", "reversed"])
def test_ledger_admits_once_and_public_row_digest_still_validates(monkeypatch, order):
    data = source_early_stopping_data()
    rows = np.arange(data.n_epochs)
    if order == "subset":
        rows = rows[::2]
    elif order == "reversed":
        rows = rows[::-1]
    actual_validate = data.validate
    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        return actual_validate(**kwargs)

    monkeypatch.setattr(data, "validate", validate)
    first = source_fit_ledger(data, rows, excluded_source_datasets=())
    assert len(calls) == 1
    assert epoch_row_identity_digest(data) == first["cache_row_identity_sha256"]
    assert len(calls) == 2
    data.X[rows[0], 0, 0] *= 2
    second = source_fit_ledger(data, rows, excluded_source_datasets=())
    assert len(calls) == 3
    assert second["fit_rows"] == rows.tolist()
    assert second["cache_row_identity_sha256"] == first["cache_row_identity_sha256"]
    assert second["fit_input_label_mask_sha256"] != first["fit_input_label_mask_sha256"]


@pytest.mark.parametrize("corruption", ["nan", "missing_event", "boolean_labels", "unobserved_channel"])
def test_invalid_current_dataset_is_rejected_after_an_earlier_good_ledger(corruption):
    data = source_early_stopping_data()
    rows = np.arange(data.n_epochs)
    source_fit_ledger(data, rows, excluded_source_datasets=())
    if corruption == "nan":
        data.X[0, 0, 0] = np.nan
    elif corruption == "missing_event":
        indices = data.event_timeline.evidence_indices.copy()
        indices[0] = -1
        data.event_timeline = replace(data.event_timeline, evidence_indices=indices)
    elif corruption == "boolean_labels":
        data.y = np.asarray(data.y, dtype=bool)
    else:
        data.X[:, 0, :] = 0
        data.channel_mask = np.asarray([False, True, True])
    with pytest.raises(ValueError):
        source_fit_ledger(data, rows, excluded_source_datasets=())
