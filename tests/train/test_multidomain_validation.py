from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.deep import DeepBaseline, DeepConfig
from baselines.validation import domain_stratified_group_validation_split
from data.exclusions import select_allowed_subjects, source_participant_excluded


def test_every_domain_is_held_out_and_unrelated_domains_do_not_change_the_split():
    groups = np.repeat([f"A{i}" for i in range(20)] + [f"B{i}" for i in range(4)], 4)
    domains = np.repeat(["A"] * 20 + ["B"] * 4, 4)
    split = domain_stratified_group_validation_split(groups, domains, fraction=0.2, seed=9)
    assert set(domains[split.validation_mask]) == {"A", "B"}
    assert not set(groups[split.train_mask]) & set(groups[split.validation_mask])
    only_b = domains == "B"
    other = domain_stratified_group_validation_split(
        groups[only_b], domains[only_b], fraction=0.2, seed=9
    )
    np.testing.assert_array_equal(split.validation_mask[only_b], other.validation_mask)


def test_group_identity_cannot_cross_domain_boundaries():
    with pytest.raises(ValueError, match="exactly one"):
        domain_stratified_group_validation_split(
            np.array(["same", "same"]), np.array(["A", "B"]), fraction=0.2
        )


def test_all_domain_validation_preserves_declared_risk_mass():
    rng = np.random.default_rng(22)
    groups = np.repeat([f"A{i}" for i in range(6)] + [f"B{i}" for i in range(4)], 8)
    domains = np.repeat(["A"] * 6 + ["B"] * 4, 8)
    X = rng.normal(size=(80, 3, 128)).astype(np.float32)
    y = np.tile([0, 0, 0, 1, 0, 0, 0, 1], 10)
    model = DeepBaseline(
        "eegnet",
        n_chans=3,
        n_times=128,
        config=DeepConfig(
            epochs=1,
            batch_size=32,
            source_validation_all_domains=True,
            compile_mode=None,
            fused_adam=False,
        ),
        device=torch.device("cpu"),
    )
    model.fit(
        X,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        source_domain_mass={"A": 0.5, "B": 0.5},
        risk_unit_ids=groups,
    )
    for phase in ("training", "validation"):
        record = model.last_source_risk[phase]
        assert {d["domain_id"] for d in record["domains"]} == {"A", "B"}
        assert all(
            d["achieved_class_weighted_mass"] == pytest.approx(0.5) for d in record["domains"]
        )


def test_bi2015a_author_rejected_subjects_are_not_active():
    assert select_allowed_subjects("BI2015a", list(range(1, 44)), explicit=False) == [
        i for i in range(1, 44) if i not in (1, 27)
    ]
    assert source_participant_excluded("BrainInvaders2015a", "subject_27")
    assert not source_participant_excluded("Huebner2017", 1)
    with pytest.raises(ValueError, match="excluded subjects"):
        select_allowed_subjects("BI2015a", [1, 2], explicit=True)
