"""Focused counterexamples for the active feature-domain transfer contract."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.adapters import (
    MODEL_SOURCE_DOMAIN_AXIS_SCHEMA,
    SOURCE_RESIDUAL_EPS,
    SOURCE_RESIDUAL_LEDGER_SCHEMA,
    SOURCE_RESIDUAL_R_MAX,
    FeatureDomainTransfer,
    FeatureTransferConfig,
    SourceResidualContractError,
    feature_transfer_config_from_record,
    source_domain_axis_from_record,
    source_domain_axis_record,
    source_residual_loss_ledger,
)
from models.n2p3net import N2P3Net
from transfer.alignment import conditional_mean_alignment_loss
from transfer.checkpoint import checkpoint_architecture_record
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

CONFIG = FeatureTransferConfig(
    invariant_bottleneck_channels=2,
    private_channels=3,
    target_bottleneck_channels=2,
    target_kernel_size=5,
    max_invariant_residual=0.2,
    max_target_residual=0.3,
    private_domain_count=2,
)


def _core() -> N2P3Net:
    return N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)


def _wrapper() -> FeatureDomainTransfer:
    return FeatureDomainTransfer(_core(), config=CONFIG)


def _source_wrapper(
    *,
    source_drop_p: float = 0.3,
    source_drop_mode: str = "sample",
) -> FeatureDomainTransfer:
    return FeatureDomainTransfer(
        _core(),
        config=replace(
            CONFIG,
            source_domain_count=2,
            source_drop_p=source_drop_p,
            source_drop_mode=source_drop_mode,
        ),
    )


def _set_constant_source_residuals(
    wrapped: FeatureDomainTransfer,
    *,
    output_biases: tuple[float, float] = (1.0, 2.0),
) -> None:
    with torch.no_grad():
        for adapter, output_bias in zip(
            wrapped.source_domain_residuals,
            output_biases,
            strict=True,
        ):
            adapter.depthwise.weight.zero_()
            adapter.down.weight.zero_()
            adapter.down.bias.fill_(1.0)
            adapter.up.weight.zero_()
            adapter.up.bias.fill_(output_bias)
            adapter.gate.fill_(0.5)


def test_wrapper_is_exact_source_identity_before_target_adaptation() -> None:
    torch.manual_seed(7)
    plain = _core()
    torch.manual_seed(7)
    wrapped = FeatureDomainTransfer(_core(), config=CONFIG)
    plain.eval()
    wrapped.eval()
    x = torch.randn(6, 3, 64)
    with torch.no_grad():
        expected = plain(x)
        observed = wrapped(x)
    assert torch.equal(expected, observed)


def test_target_residual_scale_is_zero_and_bounded() -> None:
    adapter = _wrapper().target_adapter
    features = torch.randn(4, adapter.channels, 16)
    assert adapter.scale_value() == 0.0
    with torch.no_grad():
        adapter.gate.fill_(50.0)
    assert adapter.scale_value() == pytest.approx(CONFIG.max_target_residual)
    with torch.no_grad():
        residual = adapter.residual(features)
        output = adapter(features)
    assert torch.all(
        (output - features).abs() <= CONFIG.max_target_residual * residual.abs() + 1e-6
    )


def test_wrapper_has_only_explicit_source_training_routing_surface() -> None:
    wrapped = _source_wrapper()
    assert wrapped.source_residual_enabled is True
    assert not hasattr(wrapped, "domain_adapters")
    assert not hasattr(wrapped.core, "domain_adapters")
    with pytest.raises(ValueError, match="explicit domain_indices"):
        wrapped.source_domain_view(
            torch.ones(2, wrapped.feature_channels, 8),
            None,
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"private_domain_count": 1}, "zero or at least two"),
        ({"source_domain_count": -1}, "zero or at least two"),
        ({"source_domain_count": 1}, "zero or at least two"),
        ({"source_drop_p": -0.1}, "preregistered"),
        ({"source_drop_p": 0.1}, "preregistered"),
        ({"source_drop_p": 0.6}, "preregistered"),
        ({"source_drop_p": 1.0}, "preregistered"),
        ({"source_drop_p": float("nan")}, "preregistered"),
        ({"source_drop_mode": "channel"}, "source_drop_mode"),
        (
            {"private_domain_count": 2, "source_domain_count": 3},
            "must match",
        ),
    ],
)
def test_source_residual_config_rejects_ambiguous_contracts(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(CONFIG, **changes)


def test_source_residual_config_and_ratio_contract_are_recorded() -> None:
    wrapped = _source_wrapper(source_drop_p=0.5, source_drop_mode="batch")
    record = wrapped.feature_transfer_record()
    assert record["schema"] == "n2p3_feature_domain_transfer/2"
    assert record["source_domain_count"] == 2
    assert record["source_drop_p"] == 0.5
    assert record["source_drop_mode"] == "batch"
    assert record["source_residual_ledger_schema"] == SOURCE_RESIDUAL_LEDGER_SCHEMA
    ratio = record["source_residual_ratio_contract"]
    assert ratio["eps_res"] == SOURCE_RESIDUAL_EPS
    assert ratio["r_max"] == SOURCE_RESIDUAL_R_MAX
    assert ratio["source_gate"] == "counterfactual_m=1"
    assert feature_transfer_config_from_record(record) == wrapped.config
    incomplete = dict(record)
    incomplete.pop("source_drop_mode")
    with pytest.raises(ValueError, match="fields mismatch"):
        feature_transfer_config_from_record(incomplete)
    forged = {**record, "source_domain_alias": "legacy"}
    with pytest.raises(ValueError, match="unknown=.*source_domain_alias"):
        feature_transfer_config_from_record(forged)


def test_model_source_domain_axis_freezes_order_and_keyed_weights() -> None:
    record = source_domain_axis_record(
        ("BNCI", "BI"),
        {"BI": 0.3, "BNCI": 0.7},
    )
    assert record == {
        "schema": MODEL_SOURCE_DOMAIN_AXIS_SCHEMA,
        "vocabulary": ["BNCI", "BI"],
        "domain_to_index": {"BNCI": 0, "BI": 1},
        "source_weights_by_domain": {"BNCI": 0.7, "BI": 0.3},
    }
    vocabulary, weights = source_domain_axis_from_record(record)
    assert vocabulary == ("BNCI", "BI")
    assert weights == {"BNCI": 0.7, "BI": 0.3}


@pytest.mark.parametrize(
    "record",
    [
        {
            "schema": MODEL_SOURCE_DOMAIN_AXIS_SCHEMA,
            "vocabulary": ["A", "B"],
            "domain_to_index": {"A": 1, "B": 0},
            "source_weights_by_domain": None,
        },
        {
            "schema": MODEL_SOURCE_DOMAIN_AXIS_SCHEMA,
            "vocabulary": ["A", "B"],
            "domain_to_index": {"A": 0, "B": 1},
            "source_weights_by_domain": None,
            "domain_alias": "legacy",
        },
    ],
)
def test_model_source_domain_axis_parser_rejects_forged_records(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="mismatch|non-canonical"):
        source_domain_axis_from_record(record)


@pytest.mark.parametrize(
    ("vocabulary", "weights"),
    [
        (("A", "A"), None),
        ((" A", "B"), None),
        (("A", "B"), {"A": 1.0}),
        (("A", "B"), {"A": 0.5, "B": float("nan")}),
        (("A", "B"), {"A": 0.2, "B": 0.2}),
    ],
)
def test_model_source_domain_axis_rejects_ambiguous_vocabulary_or_weights(
    vocabulary: tuple[str, ...],
    weights: dict[str, float] | None,
) -> None:
    with pytest.raises(ValueError):
        source_domain_axis_record(vocabulary, weights)


def test_source_residual_zero_init_is_bitwise_identity_without_double_add() -> None:
    wrapped = _source_wrapper(source_drop_p=0.0)
    wrapped.train()
    features = torch.randn(6, wrapped.feature_channels, 11)
    domains = torch.tensor([0, 1, 0, 1, 0, 1])
    observed = wrapped.source_domain_view(
        features,
        domains,
        source_residual_mask=torch.ones(6),
    )
    assert torch.equal(observed, features)

    _set_constant_source_residuals(wrapped)
    mask = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float32)[:, None, None]
    delta = wrapped.source_residual_delta(features, domains)
    observed = wrapped.source_domain_view(
        features,
        domains,
        source_residual_mask=mask,
    )
    assert torch.equal(observed, features + mask * delta)


def test_all_zero_mask_and_eval_are_exact_consensus_paths() -> None:
    wrapped = _source_wrapper()
    _set_constant_source_residuals(wrapped)
    features = torch.ones(8, wrapped.feature_channels, 9)
    domains = torch.tensor([0, 1] * 4)

    wrapped.train()
    dropped = wrapped.source_domain_view(
        features,
        domains,
        source_residual_mask=torch.zeros(8),
    )
    assert torch.equal(dropped, features)

    wrapped.eval()
    eval_view = wrapped.source_domain_view(
        features,
        domains,
        source_residual_mask=torch.ones(8),
    )
    assert torch.equal(eval_view, features)
    x = torch.randn(8, 3, 64)
    with torch.no_grad():
        consensus = wrapped.forward_features(x)
        requested_source = wrapped.forward_features(
            x,
            use_source_residual=True,
        )
    assert torch.equal(requested_source, consensus)


@pytest.mark.parametrize("drop_p", [0.0, 0.3, 0.5])
def test_sample_drop_mask_has_bernoulli_one_minus_p_semantics(drop_p: float) -> None:
    wrapped = _source_wrapper(source_drop_p=drop_p)
    features = torch.ones(50_000, wrapped.feature_channels, 1)
    torch.manual_seed(123)
    mask = wrapped.sample_source_residual_mask(features)
    keep_rate = float(mask[:, 0, 0].mean())
    assert keep_rate == pytest.approx(1.0 - drop_p, abs=0.01)


def test_batch_drop_mode_samples_one_shared_mask() -> None:
    wrapped = _source_wrapper(source_drop_p=0.3, source_drop_mode="batch")
    mask = wrapped.sample_source_residual_mask(torch.ones(128, wrapped.feature_channels, 1))
    assert torch.unique(mask).numel() == 1


def test_source_ratio_zero_denominator_and_counterfactual_gate() -> None:
    wrapped = _source_wrapper(source_drop_p=0.5)
    features = torch.zeros(4, wrapped.feature_channels, 7)
    domains = torch.tensor([0, 1, 0, 1])
    ratios = wrapped.source_residual_ratios(features, domains)
    assert torch.equal(ratios, torch.zeros_like(ratios))

    _set_constant_source_residuals(
        wrapped,
        output_biases=(1e-8, 1e-8),
    )
    torch.manual_seed(0)
    sampled_mask = wrapped.sample_source_residual_mask(features)
    assert bool((sampled_mask == 0).any())
    _, _, record = wrapped.source_residual_counterfactual_views(
        features,
        domains,
        fail_closed=False,
    )
    assert record["source_gate"] == "counterfactual_m=1"
    assert record["status"] == "evaluation_failed"
    assert record["max"] is None
    assert record["zero_denominator_nonzero_count"] == len(features)
    json.dumps(record, allow_nan=False)
    with pytest.raises(SourceResidualContractError, match="zero feature denominator"):
        wrapped.source_residual_counterfactual_views(features, domains)


def test_source_ratio_reports_all_statistics_and_fails_on_any_excess() -> None:
    wrapped = _source_wrapper()
    _set_constant_source_residuals(wrapped, output_biases=(1.0, 10.0))
    features = torch.ones(4, wrapped.feature_channels, 7)
    domains = torch.tensor([0, 0, 0, 1])
    _, _, record = wrapped.source_residual_counterfactual_views(
        features,
        domains,
        fail_closed=False,
    )
    for key in ("mean", "q50", "q95", "q99", "max"):
        assert isinstance(record[key], float)
    assert record["valid"] is False
    with pytest.raises(SourceResidualContractError):
        wrapped.source_residual_counterfactual_views(features, domains)


def test_counterfactual_loss_ledger_preserves_both_paths() -> None:
    wrapped = _source_wrapper()
    features = torch.ones(4, wrapped.feature_channels, 7)
    domains = torch.tensor([0, 1, 0, 1])
    _, _, ratio_record = wrapped.source_residual_counterfactual_views(
        features,
        domains,
    )
    ledger = source_residual_loss_ledger(
        torch.tensor(0.4),
        torch.tensor(0.7),
        ratio_record=ratio_record,
    )
    assert ledger["schema"] == SOURCE_RESIDUAL_LEDGER_SCHEMA
    assert ledger["loss_m0"] == pytest.approx(0.4)
    assert ledger["loss_m1"] == pytest.approx(0.7)
    assert ledger["loss_delta_m1_minus_m0"] == pytest.approx(0.3)
    assert ledger["ratio"]["source_gate"] == "counterfactual_m=1"


def test_conditional_alignment_has_active_gradient_and_sparse_cell_guard() -> None:
    rng = np.random.default_rng(3)
    features = torch.tensor(
        rng.standard_normal((32, 4, 10)), dtype=torch.float32, requires_grad=True
    )
    labels = torch.tensor(([0, 1] * 8) * 2)
    domains = torch.tensor([0] * 16 + [1] * 16)
    loss, diagnostics = conditional_mean_alignment_loss(
        features, labels, domains, 2, min_cell_count=2
    )
    assert diagnostics["active_terms"] > 0
    assert loss.item() > 0.0
    loss.backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0.0
    sparse, sparse_diagnostics = conditional_mean_alignment_loss(
        features.detach(), labels, domains, 2, min_cell_count=20
    )
    assert sparse.item() == 0.0
    assert sparse_diagnostics["active_terms"] == 0.0


def test_subject_adapter_updates_only_target_adapter() -> None:
    torch.manual_seed(17)
    wrapped = _source_wrapper()
    rng = np.random.default_rng(23)
    x = rng.standard_normal((60, 3, 64)).astype(np.float32)
    y = (rng.random(60) < 0.5).astype(np.int64)
    x[y == 1] += 0.8
    before = {key: value.detach().clone() for key, value in wrapped.state_dict().items()}
    adapter = SubjectAdapter(
        wrapped,
        config=SubjectAdapterConfig(
            head_kind="adapter",
            epochs=3,
            batch_size=16,
            val_group_fraction=None,
            refit_full_prefix=False,
            input_statistics="target_prefix",
        ),
        device=torch.device("cpu"),
    )
    assert wrapped._target_adapter_enabled is True
    adapter.fit(x, y, validation_mask=np.r_[np.zeros(50, dtype=bool), np.ones(10, dtype=bool)])
    after = wrapped.state_dict()
    target_keys = {key for key in after if key.startswith("target_adapter.")}
    assert target_keys
    assert any(not torch.equal(after[key], before[key]) for key in target_keys)
    assert all(torch.equal(after[key], before[key]) for key in after if key not in target_keys)


def test_subject_adapter_candidate_loss_updates_adapter_and_records_diagnostics() -> None:
    from data.candidate_task import MembershipDecision

    torch.manual_seed(19)
    wrapped = _wrapper()
    rng = np.random.default_rng(29)
    n_rows = 8
    x = rng.standard_normal((n_rows, 3, 64)).astype(np.float32)
    y = np.asarray([1, 0, 0, 1, 0, 1, 1, 0], dtype=np.int64)
    before = {
        key: value.detach().clone() for key, value in wrapped.target_adapter.state_dict().items()
    }
    adapter = SubjectAdapter(
        wrapped,
        config=SubjectAdapterConfig(
            head_kind="adapter",
            epochs=2,
            batch_size=4,
            val_group_fraction=None,
            refit_full_prefix=False,
            input_statistics="target_prefix",
            candidate_loss_weight=0.5,
            candidate_loss_temperature=1.0,
        ),
        device=torch.device("cpu"),
    )
    adapter.fit(
        x,
        y,
        decision_groups=tuple(MembershipDecision.row_column(np.arange(4), n_rows=2, n_columns=2,
            event_ids=tuple(f"d{index}:event{event}" for event in range(4)),
            repetition_indices=np.zeros(4, dtype=np.int64), repetitions=1,
            evidence_indices=np.arange(index * 4, (index + 1) * 4)) for index in range(2)),
        decision_targets=(1, 2),
    )
    after = wrapped.target_adapter.state_dict()
    assert any(not torch.equal(after[key], before[key]) for key in before)
    assert adapter.last_history["candidate_loss_active"] is True
    assert len(adapter.last_history["candidate_losses"]) == 4
    assert all(record["eligible"] == 1 for record in adapter.last_history["candidate_losses"])
    assert adapter.last_history["fit_budget"]["updates"] == 4


def test_subject_adapter_rejects_plain_core_for_target_mode() -> None:
    with pytest.raises(ValueError, match="FeatureDomainTransfer"):
        SubjectAdapter(_core(), config=SubjectAdapterConfig(head_kind="adapter"))


def _pretraining_baseline(*, seed=0, **model_options):
    config = DeepConfig(
        epochs=1,
        batch_size=16,
        val_group_frac=0.25,
        val_groups_min=2,
        val_groups_max=4,
        compile_mode=None,
        fused_adam=False,
        seed=seed,
    )
    return N2P3NetBaseline(
        3,
        64,
        128.0,
        config=config,
        device=torch.device("cpu"),
        tmin_s=-0.2,
        **model_options,
    )


def _source_data(seed, n_rows=64):
    x = np.random.default_rng(seed).standard_normal((n_rows, 3, 64)).astype(np.float32)
    y = np.tile([0, 1], n_rows // 2).astype(np.int64)
    domains = np.repeat(["a", "b"], n_rows // 2)
    groups = np.array([f"{domain}-{i // 4}" for i, domain in enumerate(domains)])
    return x, y, domains, groups


def test_pretraining_wrapper_records_new_schema_and_private_losses() -> None:
    baseline = _pretraining_baseline(
        feature_transfer=CONFIG,
        source_domain_vocabulary=("b", "a"),
    )
    x, y, domains, groups = _source_data(11, n_rows=64)
    baseline.fit(
        x,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        feature_alignment_weight=0.1,
        feature_private_domain_weight=0.1,
        feature_orthogonality_weight=0.01,
    )
    record = baseline.last_history["feature_alignment"]
    assert record["alignment_active"] is True
    assert record["private_domain_active"] is True
    assert "routing_active" not in record
    assert len(record["epoch_private_domain_loss"]) == 1
    architecture = baseline.model_.architecture_record()
    assert architecture["feature_transfer"]["schema"] == "n2p3_feature_domain_transfer/2"


def test_pretraining_wrapper_supports_conditional_rbf_mmd_alignment() -> None:
    baseline = _pretraining_baseline(
        feature_transfer=CONFIG,
        source_domain_vocabulary=("a", "b"),
    )
    x, y, domains, groups = _source_data(23, n_rows=96)

    baseline.fit(
        x,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        feature_alignment_weight=0.1,
        feature_alignment_mode="conditional_rbf_mmd",
        feature_alignment_min_cell_count=2,
    )

    record = baseline.last_history["feature_alignment"]
    assert record["mode"] == "conditional_rbf_mmd"
    assert record["alignment_active"] is True
    assert len(record["epoch_mmd_squared"]) == 1


def test_pretraining_routes_source_residual_and_records_counterfactual_losses() -> None:
    baseline = _pretraining_baseline(
        feature_transfer=replace(CONFIG, source_domain_count=2, source_drop_p=0.3),
        source_domain_vocabulary=("b", "a"),
        splitbn_domain_count=2,
        splitbn_target_buffer_source="pooled_source",
    )
    x, y, domains, groups = _source_data(101, n_rows=64)
    baseline.fit(x, y, group_ids=groups, source_domain_ids=domains)

    ledger = baseline.last_history["source_residual"]
    assert ledger["schema"] == SOURCE_RESIDUAL_LEDGER_SCHEMA
    assert ledger["status"] == "valid"
    assert ledger["source_drop_p"] == 0.3
    assert ledger["source_drop_mode"] == "sample"
    assert ledger["domain_vocabulary"] == ["b", "a"]
    assert len(ledger["epoch_loss_m0"]) == 1
    assert len(ledger["epoch_loss_m1"]) == 1
    assert len(ledger["epoch_loss_delta_m1_minus_m0"]) == 1
    assert ledger["epoch_ratio_forwards"][0]
    assert len(ledger["final_domain_scales"]) == 2


def test_splitbn_source_validation_uses_per_row_source_buffers() -> None:
    baseline = _pretraining_baseline(
        seed=3,
        source_domain_vocabulary=("a", "b"),
        splitbn_domain_count=2,
        # This target buffer is deliberately uninitialized. Source validation
        # can succeed only by selecting per-row source-domain buffers.
        splitbn_target_buffer_source="target_calibration",
    )
    x, y, domains, groups = _source_data(103, n_rows=64)

    baseline.fit(x, y, group_ids=groups, source_domain_ids=domains)

    assert baseline.calibration_source_ == "group_disjoint_validation"
    assert baseline.calibration_logits_ is not None
    assert np.isfinite(baseline.calibration_logits_).all()


def test_pretraining_source_residual_requires_exact_domain_contract() -> None:
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=DeepConfig(epochs=1, compile_mode=None, fused_adam=False),
        device=torch.device("cpu"),
        tmin_s=-0.2,
        feature_transfer=replace(CONFIG, source_domain_count=2),
        source_domain_vocabulary=("a", "b"),
    )
    rng = np.random.default_rng(107)
    x = rng.standard_normal((32, 3, 64)).astype(np.float32)
    y = np.tile([0, 1], 16).astype(np.int64)
    with pytest.raises(ValueError, match="explicit source_domain_ids"):
        baseline.fit(x, y)
    with pytest.raises(ValueError, match="exactly match"):
        baseline.fit(x, y, source_domain_ids=np.repeat("only", len(x)))


def test_checkpoint_record_rejects_retired_domain_bank() -> None:
    plain = _core()
    payload = {
        "architecture": {**plain.architecture_record(), "domain_adapters": {}},
        "input_tmin_s": -0.2,
        "input_sample_rate_hz": 128.0,
        "n_times": 64,
    }
    with pytest.raises(ValueError, match="retired source-domain adapter bank"):
        checkpoint_architecture_record(payload)


def test_wrapper_state_roundtrip_preserves_all_transfer_paths() -> None:
    torch.manual_seed(41)
    first = _wrapper()
    with torch.no_grad():
        first.invariant_adapter.gate.fill_(0.4)
        first.target_adapter.gate.fill_(-0.6)
        first.private_projection.bias.fill_(0.2)
    second = FeatureDomainTransfer(_core(), config=CONFIG)
    second.load_state_dict(first.state_dict())
    x = torch.randn(5, 3, 64)
    first.eval()
    second.eval()
    first.set_target_adapter_enabled(True)
    second.set_target_adapter_enabled(True)
    with torch.no_grad():
        expected = first(x)
        observed = second(x)
    assert torch.equal(expected, observed)


def test_pretraining_fail_closed_when_conditional_mmd_is_inactive() -> None:
    baseline = _pretraining_baseline(
        feature_transfer=CONFIG,
        source_domain_vocabulary=("a", "b"),
    )
    x, y, domains, groups = _source_data(31, n_rows=64)
    with pytest.raises(RuntimeError, match="feature alignment became inactive"):
        baseline.fit(
            x,
            y,
            group_ids=groups,
            source_domain_ids=domains,
            feature_alignment_weight=0.1,
            feature_alignment_mode="conditional_rbf_mmd",
            feature_alignment_min_cell_count=100,
            feature_alignment_inactive_policy="fail_closed",
        )
