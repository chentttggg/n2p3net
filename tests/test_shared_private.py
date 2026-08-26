from __future__ import annotations

import pytest
import torch

from data.channel import STANDARD_CHANNELS, build_channel_identity
from models.n2p3net import N2P3Net
from models.transfer import (
    LowRankDatasetAdapter,
    SharedPrivateEncoder,
    gradient_reverse,
)
from train.losses import active_domain_cross_entropy, compute_losses


def test_low_rank_dataset_adapter_is_exact_identity_at_initialization() -> None:
    adapter = LowRankDatasetAdapter(d_model=16, n_datasets=3, rank=4)
    features = torch.randn(6, 10, 16)
    dataset_id = torch.tensor([0, 1, 2, 0, 1, 2])
    assert torch.equal(adapter(features, dataset_id), features)
    assert adapter.rank == 4


def test_low_rank_dataset_adapter_learns_dataset_specific_residuals() -> None:
    adapter = LowRankDatasetAdapter(d_model=8, n_datasets=2, rank=4)
    with torch.no_grad():
        adapter.up[1].fill_(0.1)
    features = torch.ones(2, 3, 8)
    output = adapter(features, torch.tensor([0, 1]))
    assert torch.equal(output[0], features[0])
    assert not torch.equal(output[1], features[1])


def test_gradient_reversal_changes_only_backward_direction() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    reversed_value = gradient_reverse(value, scale=0.25)
    assert torch.equal(reversed_value, value)
    reversed_value.sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, -0.25))


def test_shared_private_encoder_shapes_and_main_sequence_contract() -> None:
    encoder = SharedPrivateEncoder(d_model=16, n_datasets=3, private_dim=8)
    output = encoder(torch.randn(5, 12, 16))
    assert output.shared_sequence.shape == (5, 12, 16)
    assert output.shared.shape == (5, 16)
    assert output.private.shape == (5, 8)
    assert output.domain_logits.shape == output.dataset_logits.shape == (5, 3)


def test_n2p3net_canonical_shared_private_end_to_end() -> None:
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=256,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        component_decoder=False,
        n_domains=2,
        canonical_channel_names=STANDARD_CHANNELS,
        dataset_adapter_rank=4,
        shared_private=True,
        private_dim=8,
    )
    identity = torch.from_numpy(build_channel_identity(("Fz", "Cz", "Pz")).embedding)
    domain_id = torch.tensor([0, 0, 1, 1])
    output = model(
        torch.randn(4, 3, 256),
        E_chn=identity,
        channel_mask=torch.ones(3, dtype=torch.bool),
        domain_id=domain_id,
    )

    assert output.canonical_covariance is not None
    assert output.canonical_covariance.shape == (4, 8, 8)
    assert output.shared_features.shape == (4, 16)
    assert output.private_features.shape == (4, 8)
    assert output.domain_logits.shape == output.dataset_logits.shape == (4, 2)
    assert output.heads.logit_target.shape == (4, 1)


def test_shared_private_losses_are_finite_and_backpropagate() -> None:
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=256,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        component_decoder=False,
        n_domains=2,
        canonical_channel_names=STANDARD_CHANNELS,
        dataset_adapter_rank=4,
        shared_private=True,
        private_dim=8,
    )
    domain_id = torch.tensor([0, 0, 1, 1, 0, 1])
    output = model(torch.randn(6, 3, 256), domain_id=domain_id)
    losses = compute_losses(
        output,
        model.component_window.tau0_bounded,
        torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [1.0]]),
        lambda2=0.0,
        lambda3=0.0,
        domain_ids=domain_id,
        lambda_orth=0.1,
        lambda_adv=0.2,
        lambda_private=0.2,
    )
    assert all(
        torch.isfinite(value)
        for value in (losses.total, losses.orth, losses.domain, losses.private)
    )
    assert losses.domain.item() > 0.0
    assert losses.private.item() > 0.0
    losses.total.backward()
    assert model.shared_private_encoder.private_encoder[1].weight.grad is not None


def test_held_out_domain_is_absent_from_softmax_and_gets_zero_gradient() -> None:
    logits = torch.tensor(
        [[2.0, 1000.0, -1.0], [-1.0, -1000.0, 2.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 2])
    loss = active_domain_cross_entropy(logits, labels, active_domain_indices=(0, 2))
    reference = torch.nn.functional.cross_entropy(logits[:, [0, 2]], torch.tensor([0, 1]))
    assert torch.equal(loss, reference)
    loss.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert torch.count_nonzero(logits.grad[:, [0, 2]]) > 0


def test_active_domain_cross_entropy_rejects_out_of_vocabulary_labels() -> None:
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        active_domain_cross_entropy(logits, torch.tensor([0, -1]), (0, 2))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        active_domain_cross_entropy(logits, torch.tensor([0, 3]), (0, 2))


def test_auxiliary_domain_reconstruction_is_not_silently_masked() -> None:
    from train.losses import estimate_reconstruction_profile

    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=64,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=64.0,
        baseline_n=12,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        component_decoder=True,
        innovation_d_model=8,
        innovation_dilations=(1,),
        n_domains=2,
        canonical_channel_names=STANDARD_CHANNELS,
        dataset_adapter_rank=4,
        shared_private=True,
        private_dim=8,
    )
    X = torch.randn(8, 3, 64)
    y = torch.tensor([[0.0], [1.0]] * 4)
    profile = estimate_reconstruction_profile(
        X,
        y,
        sfreq=64.0,
        tmin_ms=-200.0,
        bands_hz=((1.0, 12.0),),
        prior_weights=(1.0,),
        bootstrap_samples=2,
        split_half_repeats=0,
    )
    domain_id = torch.ones(8, dtype=torch.long)
    output = model(X, domain_id=domain_id, return_attention=True)
    losses = compute_losses(
        output,
        model.component_window.tau0_bounded,
        y,
        lambda2=0.0,
        lambda3=0.0,
        lambda_recon=1.0,
        reconstruction_profile=profile,
        domain_ids=domain_id,
        main_domain=0,
        reconstruct_all_domains=True,
    )
    assert losses.target.item() == 0.0
    assert losses.recon.item() > 0.0


def test_transfer_task_head_exposes_no_native_sensor_bypass() -> None:
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=64,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=64.0,
        baseline_n=12,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        innovation_d_model=8,
        innovation_dilations=(1,),
        n_domains=2,
        canonical_channel_names=STANDARD_CHANNELS,
        dataset_adapter_rank=4,
        shared_private=True,
        private_dim=8,
        task_head_shared_only=True,
    ).eval()
    X = torch.randn(4, 3, 64)
    domain_id = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        output = model(X, domain_id=domain_id)
    assert torch.equal(output.heads.logit_target, output.heads.logit_pcw)
    assert not hasattr(output.heads, "logit_residual")
    assert not hasattr(output, "residual_features")
