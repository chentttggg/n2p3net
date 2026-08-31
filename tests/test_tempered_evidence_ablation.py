from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
import torch

from experiments.run_tempered_evidence_ablation import (
    _holm_adjust,
    fit_learned_tempered_evidence,
    load_ledger_evidence,
    score_subject,
)


def test_score_subject_count_power_interpolates_mean_and_sum() -> None:
    record = {
        "truth": "1",
        "sums": {"0": 1.5, "1": 1.0},
        "counts": {"0": 3, "1": 1},
    }

    mean_prediction, _ = score_subject(record, ("0", "1"), count_power=0.0)
    tempered_prediction, _ = score_subject(record, ("0", "1"), count_power=0.5)
    sum_prediction, _ = score_subject(record, ("0", "1"), count_power=1.0)

    assert mean_prediction == tempered_prediction == "1"
    assert sum_prediction == "0"


def test_load_ledger_evidence_requires_identical_counts_across_arms(tmp_path) -> None:
    block_sizes = (62, 61, 61, 61)
    subjects_by_block = {}
    offset = 0
    for block, size in enumerate(block_sizes):
        subjects = tuple(f"s{index:03d}" for index in range(offset, offset + size))
        subjects_by_block[block] = subjects
        offset += size

    for kernel in (35, 65):
        for seed in (20260828, 20260829, 20260830):
            for block in range(4):
                path = tmp_path / f"k{kernel}_seed{seed}_blk{block}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    for subject in subjects_by_block[block]:
                        for candidate in range(9):
                            stream.write(
                                json.dumps(
                                    {
                                        "kernel": kernel,
                                        "seed": seed,
                                        "block": block,
                                        "subject": subject,
                                        "candidate": str(candidate),
                                        "target": "4",
                                        "llr_score": 1.0 if candidate == 4 else 0.0,
                                        "available_occurrence_index": 0,
                                    }
                                )
                                + "\n"
                            )

    evidence, candidates, subject_blocks = load_ledger_evidence(
        tmp_path,
        kernels=(35, 65),
        seeds=(20260828, 20260829, 20260830),
        blocks=(0, 1, 2, 3),
        subjects_by_block=subjects_by_block,
    )

    assert len(evidence) == 2 * 3 * 245
    assert candidates == tuple(str(value) for value in range(9))
    assert len(subject_blocks) == 245
    assert evidence[(35, 20260828, "s000")]["counts"]["4"] == 1


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.2})


def test_learned_tempered_evidence_optimizes_listwise_loss() -> None:
    values = torch.tensor(
        [
            [[2.0, 1.5], [-1.0, -0.5]],
            [[-1.0, -0.5], [2.0, 1.5]],
        ]
        * 4
    )
    tensors = {
        "values": values,
        "mask": torch.ones_like(values),
        "occurrence": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]] * 8),
        "targets": torch.tensor([0, 1] * 4),
        "blocks": np.repeat(np.arange(4), 2),
    }

    _, record = fit_learned_tempered_evidence(
        tensors,
        train_blocks=(0, 1, 2),
        epochs=50,
        learning_rate=0.03,
        weight_decay=1e-3,
    )

    assert record["final_loss"] < record["initial_loss"]
    assert 0.0 <= record["parameters"]["count_power"] <= 1.0
