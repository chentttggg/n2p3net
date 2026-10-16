"""Live AS-MMD source-stream path and target-calibration buffer installation."""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from models.n2p3net import N2P3Net
from models.splitbn import (
    SplitBN1d,
    SplitBN2d,
    install_splitbn_target_calibration_buffers,
)
from tests.support.checkpoints import initialized_splitbn_core
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def _live_stream_data(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source_X = rng.standard_normal((32, 3, 64)).astype(np.float32)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 16)
    domains = np.tile(np.asarray([0, 1], dtype=np.int64), 16)
    target_X = rng.standard_normal((32, 3, 64)).astype(np.float32)
    target_X[labels == 1] += 0.25
    return source_X, labels, domains, target_X, labels


def _live_config(*, epochs: int = 2, batch_size: int = 8) -> SubjectAdapterConfig:
    return SubjectAdapterConfig(
        head_kind="full_fine",
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        val_group_fraction=None,
        refit_full_prefix=False,
        freeze_batchnorm_stats=True,
        input_statistics="source",
        as_mmd_lambda0=0.4,
        as_mmd_warmup_epochs=1,
        input_mean=np.zeros((1, 3, 1), dtype=np.float32),
        input_std=np.ones((1, 3, 1), dtype=np.float32),
    )


def _splitbn_modules(core: nn.Module) -> list[nn.Module]:
    return [module for module in core.modules() if isinstance(module, (SplitBN1d, SplitBN2d))]


def _buffer_state(core: nn.Module, *, target: bool) -> dict[str, torch.Tensor]:
    return {
        f"{path}.{name}": value.detach().clone()
        for path, module in core.named_modules()
        if isinstance(module, (SplitBN1d, SplitBN2d))
        for name, value in module.named_buffers(recurse=False)
        if name.startswith(("pooled_", "target_")) == target
    }


def test_live_asmmd_requires_exactly_one_source_input_mode() -> None:
    core = initialized_splitbn_core("pooled_source")
    _, _, _, target_X, target_y = _live_stream_data()
    source_X, _, domains = _live_stream_data(1)[:3]
    frozen_logits = np.zeros((8, 2), dtype=np.float32)
    frozen_labels = np.asarray([0, 1] * 4, dtype=np.int64)

    adapter = SubjectAdapter(core, config=_live_config())
    with pytest.raises(ValueError, match="exactly one AS-MMD input mode"):
        adapter.fit(target_X, target_y)
    with pytest.raises(ValueError, match="exactly one AS-MMD input mode"):
        adapter.fit(
            target_X,
            target_y,
            source_logits=frozen_logits,
            source_labels=frozen_labels,
            source_X=source_X,
            source_domain_indices=domains,
        )
    # A malformed live payload (no labels) is rejected by the same gate before
    # any optimizer state is constructed.
    with pytest.raises(ValueError, match="exactly one AS-MMD input mode"):
        adapter.fit(target_X, target_y, source_X=source_X, source_domain_indices=domains)


@pytest.mark.parametrize(
    "make_core,head,match",
    [
        (
            partial(N2P3Net, 3, n_times=64, sfreq=128.0, tmin_s=-0.2),
            "full_fine",
            "requires a SplitBN checkpoint",
        ),
        (partial(initialized_splitbn_core, "pooled_source"), "linear", "head_kind='full_fine'"),
    ],
)
def test_live_asmmd_requires_splitbn_and_full_fine(make_core, head, match) -> None:
    source_X, source_labels, domains, target_X, target_y = _live_stream_data(2)
    adapter = SubjectAdapter(make_core(), config=replace(_live_config(), head_kind=head))
    with pytest.raises(ValueError, match=match):
        adapter.fit(
            target_X,
            target_y,
            source_X=source_X,
            source_labels=source_labels,
            source_domain_indices=domains,
        )


def test_live_asmmd_validates_source_domain_encoding() -> None:
    core = initialized_splitbn_core("pooled_source")
    source_X, source_labels, _, target_X, target_y = _live_stream_data(3)
    adapter = SubjectAdapter(core, config=_live_config())
    bad_domains = np.asarray([0, 1] * 16, dtype=np.int32)
    with pytest.raises(ValueError, match="int64"):
        adapter.fit(
            target_X,
            target_y,
            source_X=source_X,
            source_labels=source_labels,
            source_domain_indices=bad_domains,
        )


def test_live_asmmd_pooled_source_updates_source_buffers_only() -> None:
    core = initialized_splitbn_core("pooled_source")
    source_before = _buffer_state(core, target=False)
    target_before = _buffer_state(core, target=True)
    source_X, source_labels, domains, target_X, target_y = _live_stream_data(4)

    adapter = SubjectAdapter(core, config=_live_config(epochs=3))
    adapter.fit(
        target_X,
        target_y,
        source_X=source_X,
        source_labels=source_labels,
        source_domain_indices=domains,
    )

    source_after = _buffer_state(core, target=False)
    target_after = _buffer_state(core, target=True)
    assert source_before.keys() == source_after.keys()
    assert any(not torch.equal(source_before[key], source_after[key]) for key in source_before)
    assert target_before.keys() == target_after.keys()
    assert all(torch.equal(target_before[key], target_after[key]) for key in target_before)

    history = adapter.last_history
    assert history["as_mmd_input_mode"] == "source_stream"
    assert history["source_stream_count"] == len(source_X)
    assert history["source_domain_count"] == 2
    assert (
        history["splitbn_source_buffer_digest_before"]
        != (history["splitbn_source_buffer_digest_after"])
    )
    assert (
        history["splitbn_target_buffer_digest_before"]
        == (history["splitbn_target_buffer_digest_after"])
    )
    assert history["train_losses"] and all(np.isfinite(value) for value in history["train_losses"])
    assert history["as_mmd_losses"]
    assert all(
        np.isfinite(diag["source_ce"])
        and np.isfinite(diag["target_ce"])
        and np.isfinite(diag["mmd_squared"])
        for diag in history["as_mmd_losses"]
    )
    assert np.isfinite(adapter.predict_logit(target_X[:4])).all()


def test_live_asmmd_installs_target_calibration_buffer_and_keeps_it_frozen() -> None:
    core = initialized_splitbn_core("target_calibration")
    assert all(not bool(module.target_buffer_initialized) for module in _splitbn_modules(core))
    source_X, source_labels, domains, target_X, target_y = _live_stream_data(5)

    adapter = SubjectAdapter(core, config=_live_config(epochs=2))
    adapter.fit(
        target_X,
        target_y,
        source_X=source_X,
        source_labels=source_labels,
        source_domain_indices=domains,
    )

    assert all(bool(module.target_buffer_initialized) for module in _splitbn_modules(core))
    history = adapter.last_history
    assert history["splitbn_target_buffer_digest_before"] is not None
    assert (
        history["splitbn_target_buffer_digest_before"]
        == (history["splitbn_target_buffer_digest_after"])
    )
    assert (
        history["splitbn_source_buffer_digest_before"]
        != (history["splitbn_source_buffer_digest_after"])
    )
    assert np.isfinite(adapter.predict_logit(target_X[:4])).all()


def test_frozen_asmmd_path_remains_available() -> None:
    core = N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)
    rng = np.random.default_rng(6)
    target_X = rng.standard_normal((24, 3, 64)).astype(np.float32)
    target_y = np.tile(np.asarray([0, 1], dtype=np.int64), 12)
    source_logits = rng.standard_normal((12, 2)).astype(np.float32)
    source_labels = np.tile(np.asarray([0, 1], dtype=np.int64), 6)
    adapter = SubjectAdapter(
        core,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=1,
            batch_size=8,
            val_group_fraction=None,
            refit_full_prefix=False,
            input_statistics="target_prefix",
            as_mmd_lambda0=0.2,
        ),
    )
    adapter.fit(
        target_X,
        target_y,
        source_logits=source_logits,
        source_labels=source_labels,
    )
    assert adapter.last_history["as_mmd_input_mode"] == "frozen_logits"
    assert adapter.last_history["as_mmd_losses"]
    assert np.isfinite(adapter.predict_logit(target_X[:4])).all()


@pytest.mark.parametrize(
    "make_core,head,source_keys",
    [
        pytest.param(
            partial(N2P3Net, 3, n_times=64, sfreq=128.0, tmin_s=-0.2),
            "linear",
            ("source_logits", "source_labels"),
            id="frozen-logits",
        ),
        pytest.param(
            partial(initialized_splitbn_core, "pooled_source"),
            "full_fine",
            ("source_X", "source_labels", "source_domain_indices"),
            id="pooled-source",
        ),
        pytest.param(
            partial(initialized_splitbn_core, "target_calibration"),
            "full_fine",
            ("source_X", "source_labels", "source_domain_indices"),
            id="target-calibration",
        ),
    ],
)
@pytest.mark.parametrize("prefix_count", [24, 25])
def test_asmmd_refit_matches_fixed_training_on_authorized_prefix(
    make_core, head, source_keys, prefix_count: int
) -> None:
    source_X, source_labels, domains, target_X, target_y = _live_stream_data(26)
    initial = make_core()
    config = replace(
        _live_config(epochs=2),
        head_kind=head,
        refit_full_prefix=True,
    )
    available_source_inputs = {
        "source_logits": source_X[:, 0, :2],
        "source_X": source_X,
        "source_labels": source_labels,
        "source_domain_indices": domains,
    }
    source = {key: available_source_inputs[key] for key in source_keys}
    rows = np.arange(len(target_y))
    prefix = rows < prefix_count
    train = rows < prefix_count - 8
    validation = prefix & ~train
    with_context = target_X.copy()
    with_context[~prefix] += 500.0
    adapter = SubjectAdapter(copy.deepcopy(initial), config=config)
    adapter.fit(
        with_context,
        target_y,
        prefix_mask=prefix,
        training_mask=train,
        validation_mask=validation,
        **source,
    )
    history = adapter.last_history
    assert history["refit_full_prefix"]
    refit_epochs = int(history["refit_epochs"])
    direct = SubjectAdapter(
        copy.deepcopy(initial),
        config=replace(config, epochs=refit_epochs, refit_full_prefix=False),
    )
    direct.fit(target_X[prefix], target_y[prefix], **source)

    for scope, state in adapter._state_dict().items():
        for name, value in state.items():
            assert torch.equal(value, direct._state_dict()[scope][name]), (scope, name)
    np.testing.assert_array_equal(
        adapter.predict_logit(target_X[~prefix]), direct.predict_logit(target_X[~prefix])
    )
    assert history["refit_as_mmd_losses"] == direct.last_history["as_mmd_losses"]
    assert history["refit_train_losses"] == direct.last_history["train_losses"]
    for diagnostic in history["refit_as_mmd_losses"]:
        assert diagnostic["target_count"] == prefix_count
        assert diagnostic["batch_target_count"] >= 2
    assert sum(d["batch_target_count"] for d in history["refit_as_mmd_losses"]) == (
        prefix_count * refit_epochs
    )


def test_asmmd_rejects_single_sample_batch_configuration() -> None:
    with pytest.raises(ValueError, match="AS-MMD.*batch_size"):
        SubjectAdapterConfig(as_mmd_lambda0=0.2, batch_size=1)


class _SingleSplitBNProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = SplitBN1d(
            4,
            domain_count=2,
            target_buffer_source="target_calibration",
            momentum=1.0,
        )

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
    ) -> torch.Tensor:
        return self.norm(
            x,
            domain_indices=domain_indices,
            eval_buffer_source=eval_buffer_source,
            eval_buffer_index=eval_buffer_index,
        )


def test_install_target_calibration_buffers_records_and_installs_statistics() -> None:
    model = _SingleSplitBNProbe()
    model.train()
    with torch.no_grad():
        model.forward_features(
            torch.randn(8, 4, 9),
            domain_indices=torch.full((8,), 0, dtype=torch.long),
        )
    model.eval()
    calibration = torch.randn(13, 4, 9)

    record = install_splitbn_target_calibration_buffers(
        model,
        [calibration[:6], calibration[6:]],
        device=torch.device("cpu"),
        batch_size=4,
        domain_index=0,
    )

    assert set(record) == {"norm"}
    assert record["norm"]["shape"] == [4]
    assert record["norm"]["sample_count"] == 13
    assert record["norm"]["observation_count"] == 13 * 9
    assert len(str(record["norm"]["mean_digest"])) == 64
    assert len(str(record["norm"]["variance_digest"])) == 64
    expected_var, expected_mean = torch.var_mean(calibration.double(), dim=(0, 2), unbiased=False)
    torch.testing.assert_close(model.norm.target_running_mean, expected_mean.float())
    torch.testing.assert_close(model.norm.target_running_var, expected_var.float())
    assert bool(model.norm.target_buffer_initialized)
    assert len(model.norm._forward_pre_hooks) == 0

    # Already-installed buffers are diagnostic only and are never overwritten.
    installed_mean = model.norm.target_running_mean.clone()
    installed_var = model.norm.target_running_var.clone()
    install_splitbn_target_calibration_buffers(
        model,
        calibration,
        device=torch.device("cpu"),
        eval_buffer_source="target_calibration",
    )
    assert torch.equal(model.norm.target_running_mean, installed_mean)
    assert torch.equal(model.norm.target_running_var, installed_var)


def test_install_target_calibration_buffers_rejects_wrong_contract() -> None:
    pooled = SplitBN1d(4, domain_count=2, target_buffer_source="pooled_source")
    with pytest.raises(ValueError, match="every SplitBN module"):
        install_splitbn_target_calibration_buffers(
            pooled,
            torch.randn(4, 4, 9),
            device=torch.device("cpu"),
        )
    ordinary = nn.BatchNorm1d(4)
    with pytest.raises(ValueError, match="no SplitBN"):
        install_splitbn_target_calibration_buffers(
            ordinary,
            torch.randn(4, 4, 9),
            device=torch.device("cpu"),
        )
