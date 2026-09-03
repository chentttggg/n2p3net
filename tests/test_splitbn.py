"""Counterexamples for the explicit SplitBN architecture contract."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from baselines.n2p3net import N2P3NetBaseline
from models.adapters import FeatureTransferConfig, source_domain_axis_record
from models.n2p3net import N2P3Net
from models.splitbn import SplitBN1d, SplitBN2d, SplitBNConfig
from research.contracts import semantic_sha256
from transfer.checkpoint import checkpoint_architecture_record, predict_n2p3_checkpoint


def test_splitbn_mixed_domain_training_isolates_statistics_and_shares_affine() -> None:
    torch.manual_seed(3)
    layer = SplitBN1d(
        2,
        domain_count=2,
        target_buffer_source="pooled_source",
        momentum=1.0,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([2.0, 0.5]))
        layer.bias.copy_(torch.tensor([-1.0, 3.0]))
    first = torch.randn(6, 2, 11)
    second = 7.0 + 3.0 * torch.randn(6, 2, 11)
    inputs = torch.cat((first, second))
    domains = torch.tensor([0] * len(first) + [1] * len(second))

    outputs = layer(inputs, domain_indices=domains)

    torch.testing.assert_close(
        outputs[: len(first)].mean(dim=(0, 2)), layer.bias, atol=2e-6, rtol=0.0
    )
    torch.testing.assert_close(
        outputs[len(first) :].mean(dim=(0, 2)), layer.bias, atol=2e-6, rtol=0.0
    )
    torch.testing.assert_close(layer.running_mean[0], first.mean(dim=(0, 2)))
    torch.testing.assert_close(layer.running_mean[1], second.mean(dim=(0, 2)))
    torch.testing.assert_close(layer.pooled_running_mean, inputs.mean(dim=(0, 2)))
    assert set(dict(layer.named_parameters())) == {"weight", "bias"}


def test_same_distribution_splitbn_matches_pooled_training_normalization() -> None:
    torch.manual_seed(5)
    base = torch.randn(8, 3, 19)
    inputs = torch.cat((base, base))
    domains = torch.tensor([0] * len(base) + [1] * len(base))
    pooled = nn.BatchNorm1d(3, momentum=1.0)
    split = SplitBN1d(
        3,
        domain_count=2,
        target_buffer_source="pooled_source",
        momentum=1.0,
    )
    with torch.no_grad():
        pooled.weight.copy_(torch.tensor([0.5, 1.5, 2.0]))
        pooled.bias.copy_(torch.tensor([-0.3, 0.2, 1.1]))
        split.weight.copy_(pooled.weight)
        split.bias.copy_(pooled.bias)

    expected = pooled(inputs)
    observed = split(inputs, domain_indices=domains)

    torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(split.pooled_running_mean, pooled.running_mean)
    torch.testing.assert_close(split.pooled_running_var, pooled.running_var)


def test_splitbn_requires_explicit_training_domains_and_eval_buffer_index() -> None:
    layer = SplitBN1d(2, domain_count=2, target_buffer_source="pooled_source")
    inputs = torch.randn(6, 2, 9)
    domains = torch.tensor([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="explicit domain_indices"):
        layer(inputs)
    with pytest.raises(ValueError, match="out-of-contract"):
        layer(inputs, domain_indices=torch.tensor([0, 1, 0, 1, 0, 2]))
    layer(inputs, domain_indices=domains)
    layer.eval()

    with pytest.raises(ValueError, match="explicit eval_buffer_source"):
        layer(inputs)
    with pytest.raises(ValueError, match="requires domain_indices or eval_buffer_index"):
        layer(inputs, eval_buffer_source="source_domain")
    by_rows = layer(
        inputs,
        domain_indices=domains,
        eval_buffer_source="source_domain",
    )
    by_index = layer(
        inputs[:3],
        eval_buffer_source="source_domain",
        eval_buffer_index=0,
    )
    assert by_rows.shape == inputs.shape
    assert by_index.shape == inputs[:3].shape


def test_splitbn_rejected_training_batch_does_not_mutate_running_state() -> None:
    """A singleton domain or empty batch must fail before any buffer update."""

    for inputs, domains, message in (
        (
            torch.randn(2, 2),
            torch.tensor([0, 1]),
            "more than one value per channel",
        ),
        (
            torch.empty(0, 2, 3),
            torch.empty(0, dtype=torch.long),
            "non-empty batch",
        ),
    ):
        layer = SplitBN1d(2, domain_count=2, target_buffer_source="pooled_source")
        before = deepcopy(layer.state_dict())
        with pytest.raises(ValueError, match=message):
            layer(inputs, domain_indices=domains)
        for name, value in before.items():
            assert torch.equal(value, layer.state_dict()[name]), name


@pytest.mark.parametrize(
    "weights",
    [None, (0.5,), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 1.0)],
)
def test_source_weighted_requires_declared_probability_weights(
    weights: tuple[float, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="source_weight"):
        SplitBNConfig(2, "source_weighted", weights)


def test_source_weighted_uses_total_variance_not_mean_within_variance() -> None:
    layer = SplitBN1d(
        1,
        domain_count=2,
        target_buffer_source="source_weighted",
        source_weights=(0.25, 0.75),
    )
    with torch.no_grad():
        layer.running_mean.copy_(torch.tensor([[0.0], [10.0]]))
        layer.running_var.copy_(torch.tensor([[1.0], [9.0]]))
        layer.num_batches_tracked.fill_(1)
    layer.eval()

    observed = layer(torch.zeros(2, 1, 3), eval_buffer_source="source_weighted")
    mixture_mean = 0.25 * 0.0 + 0.75 * 10.0
    mixture_var = 0.25 * (1.0 + 0.0**2) + 0.75 * (9.0 + 10.0**2)
    mixture_var -= mixture_mean**2
    expected = (0.0 - mixture_mean) / (mixture_var + layer.eps) ** 0.5

    torch.testing.assert_close(observed, torch.full_like(observed, expected))


def test_target_calibration_buffer_must_be_explicitly_installed_and_frozen() -> None:
    layer = SplitBN2d(
        2,
        domain_count=2,
        target_buffer_source="target_calibration",
    )
    inputs = torch.randn(4, 2, 3, 5)
    layer(inputs, domain_indices=torch.tensor([0, 0, 1, 1]))
    layer.eval()
    state_before = deepcopy(layer.state_dict())

    with pytest.raises(RuntimeError, match="not explicitly installed"):
        layer(inputs, eval_buffer_source="target_calibration")
    with pytest.raises(ValueError, match="explicitly frozen"):
        layer.install_target_calibration_buffer(
            torch.tensor([2.0, -1.0]), torch.tensor([4.0, 9.0]), frozen=False
        )
    layer.install_target_calibration_buffer(
        torch.tensor([2.0, -1.0]), torch.tensor([4.0, 9.0]), frozen=True
    )
    with pytest.raises(RuntimeError, match="already frozen"):
        layer.install_target_calibration_buffer(
            torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), frozen=True
        )
    installed = deepcopy(layer.state_dict())

    observed = layer(inputs, eval_buffer_source="target_calibration")

    assert torch.isfinite(observed).all()
    assert not bool(state_before["target_buffer_initialized"])
    assert bool(installed["target_buffer_initialized"])
    for name, value in installed.items():
        assert torch.equal(value, layer.state_dict()[name]), name

    pooled = SplitBN1d(2, domain_count=2, target_buffer_source="pooled_source")
    with pytest.raises(RuntimeError, match="outside this SplitBN buffer contract"):
        pooled.install_target_calibration_buffer(torch.zeros(2), torch.ones(2), frozen=True)


@pytest.mark.parametrize(
    "source, weights",
    [
        ("pooled_source", None),
        ("source_weighted", (0.5, 0.5)),
        ("target_calibration", None),
    ],
)
def test_splitbn_eval_never_mutates_declared_source_or_target_buffers(
    source: str,
    weights: tuple[float, ...] | None,
) -> None:
    layer = SplitBN1d(
        2,
        domain_count=2,
        target_buffer_source=source,
        source_weights=weights,
    )
    layer(torch.randn(6, 2, 5), domain_indices=torch.tensor([0, 0, 0, 1, 1, 1]))
    layer.eval()
    if source == "target_calibration":
        layer.install_target_calibration_buffer(
            torch.tensor([1.0, -2.0]), torch.tensor([2.0, 3.0]), frozen=True
        )
    before = deepcopy(layer.state_dict())

    output = layer(torch.randn(4, 2, 5), eval_buffer_source=source)

    assert torch.isfinite(output).all()
    for name, value in before.items():
        assert torch.equal(value, layer.state_dict()[name]), name


def test_n2p3net_replaces_all_three_norm_sites_and_records_estimand() -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=64,
        splitbn_domain_count=2,
        splitbn_target_buffer_source="pooled_source",
    )
    assert isinstance(model.st_temporal_norm, SplitBN2d)
    assert isinstance(model.st_spatial_norm, SplitBN2d)
    assert all(isinstance(branch.norm, SplitBN1d) for branch in model.mst_branches)
    inputs = torch.randn(8, 3, 64)
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="explicit domain_indices"):
        model(inputs)
    assert model(inputs, domain_indices=domains).shape == (8, 2)
    model.eval()
    with pytest.raises(ValueError, match="explicit eval_buffer_source"):
        model(inputs)
    assert model(inputs, eval_buffer_source="pooled_source").shape == (8, 2)
    assert model.architecture_record()["normalization"] == {
        "schema": "n2p3_normalization/1",
        "mode": "splitbn",
        "domain_count": 2,
        "buffer_source": "pooled_source",
        "buffer_index": None,
        "source_weights": None,
        "estimand": "source_mixture",
    }


def test_default_n2p3net_retains_ordinary_batchnorm_and_rejects_split_selectors() -> None:
    model = N2P3Net(n_channels=3, n_times=64)
    assert isinstance(model.st_temporal_norm, nn.BatchNorm2d)
    assert isinstance(model.st_spatial_norm, nn.BatchNorm2d)
    assert all(isinstance(branch.norm, nn.BatchNorm1d) for branch in model.mst_branches)
    inputs = torch.randn(4, 3, 64)
    assert model(inputs).shape == (4, 2)
    with pytest.raises(ValueError, match="ordinary BatchNorm"):
        model(inputs, domain_indices=torch.zeros(4, dtype=torch.long))
    assert model.architecture_record()["normalization"]["mode"] == "batchnorm"


def test_baseline_propagates_splitbn_configuration_to_model_and_record() -> None:
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        device=torch.device("cpu"),
        source_domain_vocabulary=("dataset-a", "dataset-b"),
        splitbn_domain_count=2,
        splitbn_target_buffer_source="source_weighted",
        splitbn_source_weights={"dataset-a": 0.3, "dataset-b": 0.7},
    )

    model = baseline._make_model()
    record = baseline.architecture_record()["normalization"]

    assert isinstance(model, N2P3Net)
    assert isinstance(model.st_temporal_norm, SplitBN2d)
    assert record["mode"] == "splitbn"
    assert record["source_weights"] == [0.3, 0.7]
    assert record["estimand"] == "declared_source_composition"
    assert baseline.source_domain_axis_record() == {
        "schema": "n2p3_model_source_domain_axis/1",
        "vocabulary": ["dataset-a", "dataset-b"],
        "domain_to_index": {"dataset-a": 0, "dataset-b": 1},
        "source_weights_by_domain": {"dataset-a": 0.3, "dataset-b": 0.7},
    }
    assert "source_domain_axis_digest" in baseline.architecture_record()


def test_baseline_rejects_indexed_splitbn_without_named_domain_authority() -> None:
    with pytest.raises(ValueError, match="require source_domain_vocabulary"):
        N2P3NetBaseline(
            3,
            64,
            128.0,
            device=torch.device("cpu"),
            splitbn_domain_count=2,
            splitbn_target_buffer_source="pooled_source",
        )


def test_checkpoint_normalization_contract_rejects_old_or_forged_records() -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=64,
        splitbn_domain_count=2,
        splitbn_target_buffer_source="source_weighted",
        splitbn_source_weights=(0.4, 0.6),
    )
    axis_record = source_domain_axis_record(
        ("dataset-a", "dataset-b"),
        {"dataset-a": 0.4, "dataset-b": 0.6},
    )
    axis_digest = semantic_sha256(axis_record)
    architecture = model.architecture_record()
    architecture["source_domain_axis_digest"] = axis_digest
    payload = {
        "architecture": architecture,
        "input_tmin_s": model.tmin_s,
        "input_sample_rate_hz": model.sfreq,
        "n_channels": model.n_channels,
        "n_times": model.n_times,
        "source_domain_axis": axis_record,
        "source_domain_axis_digest": axis_digest,
    }
    assert (
        checkpoint_architecture_record(payload)["normalization"]
        == payload["architecture"]["normalization"]
    )

    old_payload = deepcopy(payload)
    old_payload["architecture"].pop("normalization")
    with pytest.raises(ValueError, match="normalization must be a mapping"):
        checkpoint_architecture_record(old_payload)

    forged_payload = deepcopy(payload)
    forged_payload["architecture"]["normalization"]["estimand"] = "target_test"
    with pytest.raises(ValueError, match="estimand or buffer contract is inconsistent"):
        checkpoint_architecture_record(forged_payload)


def test_checkpoint_binds_weighted_buffers_to_the_named_domain_order() -> None:
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        device=torch.device("cpu"),
        source_domain_vocabulary=("dataset-a", "dataset-b"),
        splitbn_domain_count=2,
        splitbn_target_buffer_source="source_weighted",
        splitbn_source_weights={"dataset-a": 0.4, "dataset-b": 0.6},
    )
    axis_record = baseline.source_domain_axis_record()
    assert axis_record is not None
    payload = {
        "architecture": baseline.architecture_record(),
        "input_tmin_s": baseline.tmin_s,
        "input_sample_rate_hz": baseline.sfreq,
        "n_channels": baseline.n_chans,
        "n_times": baseline.n_times,
        "source_domain_axis": axis_record,
        "source_domain_axis_digest": semantic_sha256(axis_record),
    }
    checkpoint_architecture_record(payload)

    missing = deepcopy(payload)
    missing.pop("source_domain_axis")
    with pytest.raises(ValueError, match="lacks a source_domain_axis"):
        checkpoint_architecture_record(missing)

    reversed_axis = source_domain_axis_record(
        ("dataset-b", "dataset-a"),
        {"dataset-a": 0.4, "dataset-b": 0.6},
    )
    reversed_digest = semantic_sha256(reversed_axis)
    mismatched = deepcopy(payload)
    mismatched["source_domain_axis"] = reversed_axis
    mismatched["source_domain_axis_digest"] = reversed_digest
    mismatched["architecture"]["source_domain_axis_digest"] = reversed_digest
    with pytest.raises(ValueError, match="keyed source weights disagree"):
        checkpoint_architecture_record(mismatched)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "legacy_schema"])
def test_checkpoint_rejects_incomplete_or_legacy_feature_transfer_records(
    mutation: str,
) -> None:
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        device=torch.device("cpu"),
        feature_transfer=FeatureTransferConfig(),
    )
    architecture = baseline.architecture_record()
    transfer = architecture["feature_transfer"]
    if mutation == "missing":
        transfer.pop("source_drop_mode")
    elif mutation == "unknown":
        transfer["source_domain_alias"] = "retired"
    else:
        transfer["schema"] = "n2p3_feature_domain_transfer/1"
    payload = {
        "architecture": architecture,
        "input_tmin_s": baseline.tmin_s,
        "input_sample_rate_hz": baseline.sfreq,
        "n_times": baseline.n_times,
    }

    with pytest.raises(ValueError, match="feature_transfer"):
        checkpoint_architecture_record(payload)


def test_checkpoint_prediction_uses_the_predeclared_target_buffer() -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=64,
        splitbn_domain_count=2,
        splitbn_target_buffer_source="pooled_source",
    )
    model.train()
    model(
        torch.randn(8, 3, 64),
        domain_indices=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    values = np.random.default_rng(11).normal(size=(3, 3, 64)).astype(np.float32)

    logits = predict_n2p3_checkpoint(
        model,
        values,
        input_stats=(np.zeros((1, 3, 1)), np.ones((1, 3, 1))),
        device=torch.device("cpu"),
    )

    assert logits.shape == (3,)
    assert np.isfinite(logits).all()
    with pytest.raises(ValueError, match="cannot override"):
        predict_n2p3_checkpoint(
            model,
            values,
            input_stats=(np.zeros((1, 3, 1)), np.ones((1, 3, 1))),
            device=torch.device("cpu"),
            eval_buffer_source="target_calibration",
        )
