"""Focused counterexamples for the active feature-domain transfer contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.adapters import FeatureDomainTransfer, FeatureTransferConfig
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
    assert torch.all((output - features).abs() <= CONFIG.max_target_residual * residual.abs() + 1e-6)


def test_wrapper_has_no_source_domain_routing_surface() -> None:
    wrapped = _wrapper()
    x = torch.randn(2, 3, 64)
    with pytest.raises(TypeError):
        wrapped(x, domain_ids=torch.zeros(2, dtype=torch.long))  # type: ignore[call-arg]
    assert not hasattr(wrapped, "domain_adapters")
    assert not hasattr(wrapped.core, "domain_adapters")


def test_conditional_alignment_has_active_gradient_and_sparse_cell_guard() -> None:
    rng = np.random.default_rng(3)
    features = torch.tensor(rng.standard_normal((32, 4, 10)), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor(([0, 1] * 8) * 2)
    domains = torch.tensor([0] * 16 + [1] * 16)
    loss, diagnostics = conditional_mean_alignment_loss(features, labels, domains, 2, min_cell_count=2)
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
    wrapped = _wrapper()
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


def test_subject_adapter_rejects_plain_core_for_target_mode() -> None:
    with pytest.raises(ValueError, match="FeatureDomainTransfer"):
        SubjectAdapter(_core(), config=SubjectAdapterConfig(head_kind="adapter"))


def test_pretraining_wrapper_records_new_schema_and_private_losses() -> None:
    config = DeepConfig(
        epochs=1,
        batch_size=16,
        val_group_frac=0.25,
        val_groups_min=2,
        val_groups_max=4,
        compile_mode=None,
        fused_adam=False,
        seed=0,
    )
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=config,
        device=torch.device("cpu"),
        tmin_s=-0.2,
        feature_transfer=CONFIG,
    )
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, 3, 64)).astype(np.float32)
    y = np.tile([0, 1], 32).astype(np.int64)
    domains = np.repeat(["a", "b"], 32)
    groups = np.array([f"{domain}-{i // 4}" for i, domain in enumerate(domains)])
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
    assert architecture["feature_transfer"]["schema"] == "n2p3_feature_domain_transfer/1"


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
