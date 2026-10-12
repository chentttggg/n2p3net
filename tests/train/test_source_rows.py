"""Source indexed rows preserve exact statistics, normalizers and row order."""
from __future__ import annotations

import numpy as np
import pytest

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from train.source_rows import SourceRows
from train.source_sampling import observed_source_statistics


def _data():
    X = np.random.default_rng(857).normal(0., 2e-6, (5033, 6, 179)).astype(np.float32)
    rows = np.r_[np.arange(5032, -1, -1), [17, 17, 29]]
    observed = np.ones((len(rows), 6), dtype=bool)
    observed[::7, 3] = False
    domains = np.where(np.arange(len(rows)) % 3 == 0, "A", "B")
    people = np.array([f"{domain}-{index % 9}" for index, domain in enumerate(domains)])
    return X, rows, observed, domains, people


def test_source_rows_preserve_order_duplicate_rows_and_bound_array_conversion():
    X, rows, *_ = _data()
    view = SourceRows(X, rows)
    np.testing.assert_array_equal(view[3:13], X[rows][3:13])
    np.testing.assert_array_equal(view[2040:4099, 1:5:2, 13:19], X[rows][2040:4099, 1:5:2, 13:19])
    np.testing.assert_array_equal(view[np.array([5, 2, 5]), 4], X[rows][np.array([5, 2, 5]), 4])
    with pytest.raises(ValueError, match="bounded row blocks"):
        np.asarray(view)
    rows[:] = 0
    assert not np.array_equal(view.rows, rows)
    assert not view.rows.flags.writeable and not view.base.flags.writeable
    with pytest.raises(ValueError, match="row-first"):
        view[..., 0]


@pytest.mark.parametrize("weighting", ["observed_rows", "observed_domain_participant"])
def test_source_rows_statistics_are_exact_for_masked_and_partial_scope(weighting):
    X, rows, observed, domains, people = _data()
    allowed = np.arange(len(rows)) % 5 != 0
    channels = np.array([0, 2, 3, 5])
    dense = X[rows][allowed][:, channels]
    lazy = SourceRows(X, rows).select(allowed, channels)
    expected = observed_source_statistics(dense, observed[allowed][:, channels], domain_ids=domains[allowed],
        participant_keys=people[allowed], weighting=weighting)
    actual = observed_source_statistics(lazy, observed[allowed][:, channels], domain_ids=domains[allowed],
        participant_keys=people[allowed], weighting=weighting)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


@pytest.mark.parametrize("weighting", ["observed_rows", "observed_domain_participant"])
def test_source_row_moments_remain_exact_across_32768_row_reduction_blocks(weighting):
    X = np.random.default_rng(774).normal(0., 2e-6, (33011, 3, 17)).astype(np.float32)
    rows = np.r_[np.arange(len(X) - 1, -1, -1), [17, 17]]
    observed = np.ones((len(rows), 3), bool)
    observed[::113, 1] = False
    domains = np.where(np.arange(len(rows)) % 3, "A", "B")
    people = np.asarray([f"{domain}-{index % 7}" for index, domain in enumerate(domains)])
    arguments = dict(domain_ids=domains, participant_keys=people, weighting=weighting)
    expected = observed_source_statistics(X[rows], observed, **arguments)
    actual = observed_source_statistics(SourceRows(X, rows), observed, **arguments)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_source_row_normalizer_is_exact_for_training_and_validation():
    X, rows, observed, domains, people = _data()
    mean, std, _ = observed_source_statistics(X[rows], observed, domain_ids=domains, participant_keys=people)
    model = N2P3NetBaseline(6, 179, 128., config=DeepConfig(epochs=1), device="cpu")
    model._set_input_statistics(mean, std)
    expected = model._prepare_input(X[rows], observed)
    actual = model._prepare_input(SourceRows(X, rows), observed)
    np.testing.assert_array_equal(actual, expected)
    held = rows[::-1]
    np.testing.assert_array_equal(model._prepare_input(SourceRows(X, held), observed[::-1]),
                                  model._prepare_input(X[held], observed[::-1]))


def test_lazy_source_rows_execute_exact_fixed_updates_with_cpu_resident_transfer(tmp_path):
    import json
    import multiprocessing

    import psutil
    import torch

    from research.source_execution import SourceInputAdmission

    X, rows, observed, domains, people = _data()
    y = np.arange(len(rows), dtype=np.int64) % 2
    cfg = DeepConfig(fixed_updates=2, batch_size=32, val_group_frac=None,
        sampling="domain_participant", pos_weight=1., compile_mode=None, seed=853)
    arguments = dict(trial_channel_mask=observed, source_domain_ids=domains,
        risk_unit_ids=people, source_domain_mass={"A": .5, "B": .5})
    dense = N2P3NetBaseline(6, 179, 128., config=cfg, device="cpu")
    dense.fit(X[rows], y, **arguments)
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=40 * 2**20, load_workers=1)
        lazy = N2P3NetBaseline(6, 179, 128., config=cfg, device="cpu")
        lazy.runtime.input_staging_gate = owner
        # Whole-process admission includes allocator state retained by earlier
        # suite cases, plus this fixture's bounded source/normalization work.
        baseline_rss = psutil.Process().memory_info().rss
        fixture_headroom = 8 * len(rows) * 6 * 179 * 4
        lazy.runtime.configure_host_process_resident_budget(baseline_rss + fixture_headroom)
        lazy.configure_epoch_progress(tmp_path / "progress")
        lazy.configure_evaluation_fold(0)
        lazy.fit(SourceRows(X, rows), y, **arguments)
        assert owner.record()["free_staging_bytes"] == 40 * 2**20
        assert lazy.last_runtime["normalized_host_matrix_retained"]
        assert lazy.last_runtime["normalized_host_resident_bytes"] == len(rows) * 6 * 179 * 4
    events = [json.loads(line) for line in (tmp_path / "progress/fold_0.jsonl").read_text().splitlines()]
    assert len(events) == 1 and events[0]["optimizer_steps"] == 2
    assert events[0]["trial_draws"] == 64 and events[0]["phase"] == "source_fixed_updates"
    assert dense.last_history == lazy.last_history
    for key, value in dense.model_.state_dict().items():
        assert torch.equal(value, lazy.model_.state_dict()[key]), key


def test_cpu_fallback_over_resident_budget_fails_before_updates_and_releases_staging(monkeypatch):
    import multiprocessing

    import torch

    from research.source_execution import SourceInputAdmission

    X, rows, observed, domains, people = _data()
    steps = []
    original_step = torch.optim.Adam.step

    def step(*arguments, **options):
        steps.append(True)
        return original_step(*arguments, **options)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    model = N2P3NetBaseline(6, 179, 128., config=DeepConfig(fixed_updates=2,
        batch_size=32, val_group_frac=None, sampling="domain_participant", compile_mode=None), device="cpu")
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=40 * 2**20, load_workers=1)
        model.runtime.input_staging_gate = owner
        model.runtime.configure_host_process_resident_budget(64 * 2**20)
        with pytest.raises(RuntimeError, match="whole-process resident budget"):
            model.fit(SourceRows(X, rows), np.arange(len(rows), dtype=np.int64) % 2,
                trial_channel_mask=observed, source_domain_ids=domains, risk_unit_ids=people,
                source_domain_mass={"A": .5, "B": .5})
        assert owner.record()["free_staging_bytes"] == 40 * 2**20
        assert owner.record()["nn_active"] == 0
    assert not steps and not model._fitted
