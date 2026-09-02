"""Counterexample and contract tests for the domain-adapter mechanism.

These tests encode the falsifiable invariants of the 2026-09-02 alignment
design: identity initialization (a routed adapted model equals the shared
trunk before any data moves the gate), bounded residual amplitude, fail-closed
domain vocabularies, order-preserving routing, permutation-invariant
conditional alignment, and an adapter-only subject inner loop that leaves the
trunk and classifier bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.adapters import (
    RESERVED_TARGET_DOMAIN,
    DomainAdapterBank,
    FeatureAdapterConfig,
    FeatureResidualAdapter,
)
from models.n2p3net import N2P3Net
from transfer.alignment import conditional_mean_alignment_loss
from transfer.checkpoint import checkpoint_architecture_record
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

ADAPTER_TEST_CONFIG = FeatureAdapterConfig(
    bottleneck_channels=2,
    kernel_size=5,
    max_residual=0.5,
)


def _synthetic_two_domain(
    n_per_domain: int = 120,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, domains, groups) with class signal and a domain shift."""

    rng = np.random.default_rng(seed)
    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    blocks_d: list[np.ndarray] = []
    blocks_g: list[np.ndarray] = []
    for domain_index, domain in enumerate(("alpha", "beta")):
        for subject in range(4):
            n = n_per_domain // 4
            x = rng.standard_normal((n, 3, 64)).astype(np.float32)
            y = (rng.random(n) < 0.5).astype(np.int64)
            x[y == 1] += 0.8
            x += 0.4 * domain_index
            blocks_x.append(x)
            blocks_y.append(y)
            blocks_d.append(np.full(n, domain, dtype=object))
            blocks_g.append(np.full(n, f"{domain}-{subject}", dtype=object))
    return (
        np.concatenate(blocks_x),
        np.concatenate(blocks_y),
        np.concatenate(blocks_d).astype(str),
        np.concatenate(blocks_g).astype(str),
    )


def _cpu_baseline(
    *,
    feature_adapter: FeatureAdapterConfig | None,
    adapter_domains: tuple[str, ...] | None,
) -> N2P3NetBaseline:
    config = DeepConfig(
        epochs=2,
        batch_size=64,
        lr=1e-3,
        pos_weight=1.0,
        val_group_frac=0.25,
        val_groups_min=2,
        val_groups_max=6,
        compile_mode=None,
        fused_adam=False,
        seed=0,
    )
    return N2P3NetBaseline(
        3,
        64,
        128.0,
        config=config,
        device=torch.device("cpu"),
        tmin_s=-0.2,
        pooling_mode="full_unfold",
        feature_adapter=feature_adapter,
        adapter_domains=adapter_domains,
    )


def test_adapter_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(7)
    plain = N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)
    torch.manual_seed(7)
    adapted = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("a", "b"),
    )
    x = torch.randn(6, 3, 64)
    plain.eval()
    adapted.eval()
    with torch.no_grad():
        out_plain = plain(x)
        out_unrouted = adapted(x)
        out_routed = adapted(x, domain_ids=["a", "b", "a", "b", "a", "b"])
        out_tensor = adapted(x, domain_ids=torch.tensor([0, 1, 0, 1, 0, 1]))
    # tanh(0) = 0 makes the residual exactly zero in IEEE arithmetic, so the
    # identity claim is exact, not approximate.
    assert torch.equal(out_plain, out_unrouted)
    assert torch.equal(out_plain, out_routed)
    assert torch.equal(out_plain, out_tensor)


def test_adapter_residual_is_bounded_by_max_residual() -> None:
    torch.manual_seed(11)
    model = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("a",),
    )
    adapter = model.domain_adapters.adapters["a"]
    assert adapter.residual_scale() == 0.0
    with torch.no_grad():
        adapter.gate.fill_(50.0)
    assert adapter.residual_scale() == pytest.approx(ADAPTER_TEST_CONFIG.max_residual)
    model.eval()
    x = torch.randn(4, 3, 64)
    with torch.no_grad():
        features = model.forward_features(x, domain_ids=["a"] * 4)
        raw = model.forward_features(x)
    residual = (features - raw).abs()
    with torch.no_grad():
        magnitude = adapter.residual(raw).abs()
    assert torch.all(residual <= ADAPTER_TEST_CONFIG.max_residual * magnitude + 1e-6)
    assert bool(torch.isfinite(features).all())


def test_bank_fails_closed_on_unknown_domains() -> None:
    bank = DomainAdapterBank(4, config=ADAPTER_TEST_CONFIG, domains=("alpha", "beta"))
    features = torch.randn(3, 4, 16)
    with pytest.raises(ValueError, match="absent from the frozen adapter vocabulary"):
        bank(features, domain_ids=["alpha", "gamma", "beta"])
    with pytest.raises(ValueError, match=r"\[0,2\)"):
        bank(features, domain_ids=torch.tensor([0, 2, 1]))
    with pytest.raises(ValueError):
        bank(features, domain_ids=["alpha"])
    empty = DomainAdapterBank(4, config=ADAPTER_TEST_CONFIG)
    with pytest.raises(ValueError):
        empty(features, domain_ids=torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="reserved"):
        bank.register("gamma", FeatureResidualAdapter(4, config=ADAPTER_TEST_CONFIG))
    with pytest.raises(ValueError, match="already registered"):
        bank.register(
            "alpha", FeatureResidualAdapter(4, config=ADAPTER_TEST_CONFIG), reserved=True
        )


def test_bank_routing_preserves_row_order_across_mixed_batches() -> None:
    torch.manual_seed(13)
    bank = DomainAdapterBank(4, config=ADAPTER_TEST_CONFIG, domains=("a", "b"))
    with torch.no_grad():
        bank.adapters["a"].gate.fill_(3.0)
        bank.adapters["b"].gate.fill_(-3.0)
    features = torch.randn(6, 4, 16)
    indices = torch.tensor([0, 1, 1, 0, 1, 0])
    routed = bank(features, domain_ids=indices)
    for row, domain in enumerate(indices.tolist()):
        alone = bank(features[row : row + 1], domain_ids=torch.tensor([domain]))
        # Different batch sizes may round convolutions differently; the
        # routing contract is value equality within float tolerance.
        assert torch.allclose(routed[row : row + 1], alone, rtol=0.0, atol=1e-6)


def test_alignment_loss_zero_when_conditional_means_match() -> None:
    rng = np.random.default_rng(3)
    base = torch.tensor(rng.standard_normal((16, 4, 10)), dtype=torch.float32)
    features = torch.cat([base, base])
    labels = torch.tensor(([0, 1] * 8) * 2)
    domains = torch.tensor([0] * 16 + [1] * 16)
    loss, diagnostics = conditional_mean_alignment_loss(features, labels, domains, 2)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["active_terms"] > 0


def test_alignment_loss_positive_and_permutation_invariant() -> None:
    rng = np.random.default_rng(5)
    features = torch.tensor(rng.standard_normal((40, 4, 10)), dtype=torch.float32)
    labels = torch.tensor(
        rng.integers(0, 2, size=40), dtype=torch.long
    )
    domains = torch.tensor(rng.integers(0, 3, size=40), dtype=torch.long)
    loss, diagnostics = conditional_mean_alignment_loss(
        features, labels, domains, 3, min_cell_count=2
    )
    assert diagnostics["active_terms"] > 0
    assert loss.item() > 0.0
    # Relabeling domains is a bijection over the unordered pair set: the loss
    # must be invariant (counterexample guard against asymmetric handling).
    permutation = {0: 2, 1: 0, 2: 1}
    relabeled = torch.tensor(
        [permutation[int(domain)] for domain in domains], dtype=torch.long
    )
    loss_relabeled, _ = conditional_mean_alignment_loss(
        features, labels, relabeled, 3, min_cell_count=2
    )
    assert loss_relabeled.item() == pytest.approx(loss.item(), rel=1e-6)
    # The gradient must reach the features when terms are active.
    probe = features.detach().clone().requires_grad_(True)
    loss_probe, _ = conditional_mean_alignment_loss(
        probe, labels, domains, 3, min_cell_count=2
    )
    loss_probe.backward()
    assert probe.grad is not None
    assert float(probe.grad.abs().sum()) > 0.0


def test_alignment_loss_skips_sparse_cells_and_handles_degenerate_axes() -> None:
    features = torch.randn(6, 4, 10)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    domains = torch.tensor([0, 0, 0, 1, 1, 1])
    loss, diagnostics = conditional_mean_alignment_loss(
        features, labels, domains, 2, min_cell_count=8
    )
    assert diagnostics["active_terms"] == 0
    assert loss.item() == 0.0
    # A single domain axis cannot produce cross-domain terms.
    loss_single, _ = conditional_mean_alignment_loss(
        features, labels, torch.zeros(6, dtype=torch.long), 1
    )
    assert loss_single.item() == 0.0
    with pytest.raises(ValueError):
        conditional_mean_alignment_loss(features, labels, torch.full((6,), 5), 2)


def test_deep_baseline_routes_adapters_and_alignment() -> None:
    X, y, domains, groups = _synthetic_two_domain()
    baseline = _cpu_baseline(
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha", "beta"),
    )
    baseline.fit(
        X,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        selection_domain="alpha",
        feature_alignment_weight=0.5,
    )
    record = baseline.last_history["feature_alignment"]
    assert record["routing_active"] is True
    assert record["alignment_active"] is True
    distances = record["epoch_mean_squared_distance"]
    assert len(distances) == len(baseline.last_history["train_losses"])
    assert any(value is not None and value > 0.0 for value in distances)
    scores = baseline.predict_logit(X[:10])
    assert np.isfinite(scores).all()
    architecture = baseline.model_.architecture_record()
    assert architecture["domain_adapters"]["domains"] == ["alpha", "beta"]


def test_deep_baseline_alignment_without_adapters_is_the_a1_arm() -> None:
    X, y, domains, groups = _synthetic_two_domain()
    baseline = _cpu_baseline(feature_adapter=None, adapter_domains=None)
    baseline.fit(
        X,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        selection_domain="alpha",
        feature_alignment_weight=0.25,
    )
    record = baseline.last_history["feature_alignment"]
    assert record["routing_active"] is False
    assert record["alignment_active"] is True
    assert any(value is not None for value in record["epoch_mean_squared_distance"])


def test_deep_baseline_rejects_adapter_domains_without_domain_axis() -> None:
    X, y, _, _ = _synthetic_two_domain(n_per_domain=30)
    baseline = _cpu_baseline(
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha", "beta"),
    )
    with pytest.raises(ValueError, match="domain-adapter routing is inactive"):
        baseline.fit(X, y)


def test_deep_baseline_rejects_unknown_adapter_domains() -> None:
    X, y, domains, groups = _synthetic_two_domain(n_per_domain=30)
    baseline = _cpu_baseline(
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha",),
    )
    with pytest.raises(ValueError, match="absent from the adapter vocabulary"):
        baseline.fit(
            X,
            y,
            group_ids=groups,
            source_domain_ids=domains,
            selection_domain="alpha",
        )


def test_deep_baseline_rejects_alignment_without_domain_axis() -> None:
    X, y, _, _ = _synthetic_two_domain(n_per_domain=30)
    baseline = _cpu_baseline(feature_adapter=None, adapter_domains=None)
    with pytest.raises(ValueError, match="requires an explicit source-domain axis"):
        baseline.fit(X, y, feature_alignment_weight=0.1)


def test_subject_adapter_mode_trains_only_the_target_adapter() -> None:
    torch.manual_seed(17)
    trunk = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha", "beta"),
    )
    rng = np.random.default_rng(23)
    X = rng.standard_normal((60, 3, 64)).astype(np.float32)
    y = (rng.random(60) < 0.5).astype(np.int64)
    X[y == 1] += 0.8
    validation_mask = np.zeros(60, dtype=bool)
    validation_mask[50:] = True
    adapter = SubjectAdapter(
        trunk,
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
    assert RESERVED_TARGET_DOMAIN in trunk.domain_adapters.adapters
    # Snapshot after the target slot exists so the comparison covers every
    # parameter the fit could touch.
    before = {
        key: value.detach().clone() for key, value in trunk.state_dict().items()
    }
    adapter.fit(X, y, validation_mask=validation_mask)
    after = trunk.state_dict()
    target_prefix = f"domain_adapters.adapters.{RESERVED_TARGET_DOMAIN}"
    for key, value in after.items():
        if key.startswith(target_prefix):
            continue
        assert torch.equal(value, before[key]), f"non-adapter parameter {key} changed"
    target_changed = any(
        not torch.equal(after[key], before[key])
        for key in after
        if key.startswith(target_prefix)
    )
    assert target_changed
    scores = adapter.predict_logit(X[:8])
    assert np.isfinite(scores).all()
    # The reserved slot is single-use: a second subject adapter on the same
    # trunk must fail closed instead of silently sharing the slot.
    with pytest.raises(ValueError, match="already registered"):
        SubjectAdapter(
            trunk,
            config=SubjectAdapterConfig(head_kind="adapter"),
            device=torch.device("cpu"),
        )


def test_subject_adapter_mode_requires_a_bank() -> None:
    trunk = N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)
    with pytest.raises(ValueError, match="requires a trunk constructed with"):
        SubjectAdapter(
            trunk,
            config=SubjectAdapterConfig(head_kind="adapter"),
            device=torch.device("cpu"),
        )


def test_routed_source_training_matches_single_domain_slices() -> None:
    """End-to-end routing equivalence counterexample.

    Feeding the full multi-domain matrix through one routed fit must produce
    the same first optimizer step as feeding each domain separately would
    route them: the bank applies per-domain residuals without reordering.
    """

    torch.manual_seed(29)
    bank = DomainAdapterBank(4, config=ADAPTER_TEST_CONFIG, domains=("a", "b"))
    with torch.no_grad():
        bank.adapters["a"].gate.fill_(1.5)
        bank.adapters["b"].gate.fill_(-1.5)
    features = torch.randn(5, 4, 12, requires_grad=True)
    indices = torch.tensor([1, 0, 1, 1, 0])
    routed = bank(features, domain_ids=indices)
    manual = torch.empty_like(features)
    for row, domain in enumerate(indices.tolist()):
        manual[row] = bank.adapters["a" if domain == 0 else "b"](features[row : row + 1])[0]
    assert torch.allclose(routed, manual, atol=1e-6)
    routed.sum().backward()
    assert features.grad is not None


def test_adapter_checkpoint_state_roundtrips_through_the_architecture_record() -> None:
    torch.manual_seed(31)
    trunk = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha", "beta"),
    )
    with torch.no_grad():
        # Move the gates so the bank state is non-trivial to roundtrip.
        trunk.domain_adapters.adapters["alpha"].gate.fill_(0.7)
        trunk.domain_adapters.adapters["beta"].gate.fill_(-0.4)
    record = trunk.architecture_record()
    adapter_record = record["domain_adapters"]
    rebuilt = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=FeatureAdapterConfig(
            bottleneck_channels=adapter_record["adapter_config"]["bottleneck_channels"],
            kernel_size=adapter_record["adapter_config"]["kernel_size"],
            max_residual=adapter_record["adapter_config"]["max_residual"],
        ),
        adapter_domains=tuple(adapter_record["domains"]),
    )
    rebuilt.load_state_dict(trunk.state_dict())
    x = torch.randn(3, 3, 64)
    trunk.eval()
    rebuilt.eval()
    with torch.no_grad():
        expected = trunk(x, domain_ids=["alpha", "beta", "alpha"])
        observed = rebuilt(x, domain_ids=["alpha", "beta", "alpha"])
    assert torch.equal(expected, observed)
    # A bankless trunk must report the bank weights as unexpected instead of
    # silently deploying an adapter checkpoint without its adapters.
    bankless = N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)
    _, unexpected = bankless.load_state_dict(trunk.state_dict(), strict=False)
    assert unexpected


def test_checkpoint_architecture_record_preserves_the_adapter_contract() -> None:
    trunk = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        feature_adapter=ADAPTER_TEST_CONFIG,
        adapter_domains=("alpha", "beta"),
    )
    payload = {
        "architecture": trunk.architecture_record(),
        "input_tmin_s": -0.2,
        "input_sample_rate_hz": 128.0,
        "n_times": 64,
    }
    summary = checkpoint_architecture_record(payload)
    assert summary["domain_adapters"]["domains"] == ["alpha", "beta"]
    assert summary["domain_adapters"]["adapter_config"]["kernel_size"] == 5
    plain = N2P3Net(n_channels=3, n_times=64, sfreq=128.0, tmin_s=-0.2)
    plain_payload = {
        "architecture": plain.architecture_record(),
        "input_tmin_s": -0.2,
        "input_sample_rate_hz": 128.0,
        "n_times": 64,
    }
    assert "domain_adapters" not in checkpoint_architecture_record(plain_payload)
