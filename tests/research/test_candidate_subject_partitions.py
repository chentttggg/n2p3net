from __future__ import annotations

import numpy as np
import pytest

from experiments.prepare_candidate_subject_partitions import (
    balanced_subject_partitions,
)


def test_balanced_subject_partitions_are_complete_equal_and_deterministic() -> None:
    subject_ids = np.concatenate(
        [np.repeat(f"s{index}", count) for index, count in enumerate(range(1, 9), 1)]
    )

    first = balanced_subject_partitions(subject_ids, n_partitions=4)
    second = balanced_subject_partitions(subject_ids[::-1], n_partitions=4)

    assert first == second
    assert all(len(block) == 2 for block in first)
    assert sorted(subject for block in first for subject in block) == [
        f"s{index}" for index in range(1, 9)
    ]


def test_partition_cli_binds_each_file_and_refuses_a_used_directory(tmp_path, monkeypatch) -> None:
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    import experiments.prepare_candidate_subject_partitions as module

    dataset = SimpleNamespace(subject_ids=np.repeat(["s1", "s2", "s3", "s4"], 2), provenance={})
    task = SimpleNamespace(dataset_id="d", task_id="t", record=lambda: {"task": "t"})
    monkeypatch.setattr(module, "load_epoch_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, "loaded_epoch_cache_attestation", lambda data: {"sha256": "a" * 64, "byte_size": 1})
    monkeypatch.setattr(module, "candidate_task_contract_from_provenance", lambda provenance: task)
    output = tmp_path / "partitions"
    monkeypatch.setattr(sys, "argv", ["prepare", "--dataset-cache", str(tmp_path / "cache"),
                                      "--output-dir", str(output), "--partitions", "2"])
    module.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["partitions"]) == 2
    for record in manifest["partitions"]:
        assert record["sha256"] == hashlib.sha256((output / record["file"]).read_bytes()).hexdigest()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="not empty"):
        module.main()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_balanced_subject_partitions_require_equal_capacity() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        balanced_subject_partitions(np.asarray(["s1", "s2", "s3"]), n_partitions=2)
