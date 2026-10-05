from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.epochs import (
    TARGET_CALIBRATION_SCOPE_SCHEMA,
    load_epoch_dataset,
    save_epoch_dataset,
)
from experiments.prepare_target_calibration_cache import main
from tests.support.candidate import candidate_dataset
from tests.support.preparation_cli import cli_options
from transfer.candidate_within_subject import candidate_calibration_decision_split


@pytest.mark.parametrize(
    "n_selections,calibration_selections,test_repetitions,extra",
    [(7, 3, 3, {}), (5, 2, 2, {"selection_repetition_floor": 2})],
)
def test_target_calibration_cache_keeps_only_calibration_rows(
    tmp_path: Path, n_selections, calibration_selections, test_repetitions, extra
) -> None:
    dataset = candidate_dataset(n_selections=n_selections)
    source = tmp_path / "target.npz"
    output = tmp_path / "target-cal.npz"
    save_epoch_dataset(source, dataset)

    main(
        cli_options(
            dataset_cache=source,
            output=output,
            calibration_selections=calibration_selections,
            test_repetitions=test_repetitions,
            **extra,
        )
    )

    split = candidate_calibration_decision_split(
        dataset,
        calibration_selections=calibration_selections,
        test_repetitions=test_repetitions,
        **extra,
    )
    expected_rows = int(np.asarray(split.calibration_mask).sum())
    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert adapted.n_epochs == expected_rows == calibration_selections * 12 * 3
    assert (
        adapted.provenance["target_calibration_scope"]["schema"] == TARGET_CALIBRATION_SCOPE_SCHEMA
    )
    assert (
        adapted.provenance["target_calibration_scope"]["calibration_selections"]
        == calibration_selections
    )
    assert set(adapted.metadata["selection_id"].unique()) == {
        f"s1:selection{i}" for i in range(calibration_selections)
    }
    np.testing.assert_array_equal(adapted.y, dataset.y[split.calibration_mask])
    assert adapted.identity_table.local_subject_ids == ("s1",)
    assert adapted.event_timeline.n_events == adapted.n_epochs


def test_source_selection_vectors_cli_writes_attested_vectors(tmp_path: Path) -> None:
    dataset = candidate_dataset()
    source = tmp_path / "source.npz"
    labels_path = tmp_path / "labels.npz"
    groups_path = tmp_path / "groups.npz"
    save_epoch_dataset(source, dataset)

    from experiments.prepare_source_selection_vectors import main as vectors_main

    vectors_main(
        cli_options(dataset_cache=source, output_labels=labels_path, output_groups=groups_path)
    )

    with np.load(labels_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["labels"], dataset.y)
    with np.load(groups_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["groups"], np.asarray(dataset.subject_ids, dtype=str))
