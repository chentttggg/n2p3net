from __future__ import annotations

from unittest import mock

import torch

from data.channel import STANDARD_CHANNELS
from models.multidataset import MontageBranchSpec, MultiMontageN2P3Net
from train.contracts import TrialContext
from train.multidataset import MultiDatasetSchedule, MultiDatasetTrainer
from train.preloaded import PreloadedDataLoader
from train.trainer import TrainerConfig


def _model() -> MultiMontageN2P3Net:
    return MultiMontageN2P3Net(
        {
            "gtn": MontageBranchSpec(("Fz", "Cz", "Pz")),
            "dry8": MontageBranchSpec(STANDARD_CHANNELS),
        },
        canonical_channel_names=STANDARD_CHANNELS,
        model_kwargs={
            "n_time": 64,
            "tmin_ms": -200.0,
            "tmax_ms": 800.0,
            "sfreq": 64.0,
            "d_model": 16,
            "temporal_kernels": (13,),
            "filters_per_scale": 2,
            "encoder_depth": 1,
            "component_decoder": False,
            "dataset_adapter_rank": 4,
            "shared_private": True,
            "private_dim": 8,
        },
    )


def _loader(channels: int, batches: int, *, seed: int) -> PreloadedDataLoader:
    generator = torch.Generator().manual_seed(seed)
    n_trials = batches * 2
    epochs = torch.randn(n_trials, channels, 64, generator=generator)
    y = torch.tensor([0.0, 1.0] * batches)
    return PreloadedDataLoader(epochs, y, batch_size=2, shuffle=False)


def test_multidataset_trainer_uses_one_optimizer_and_balances_domains() -> None:
    model = _model()
    trainer = MultiDatasetTrainer(
        model,
        TrainerConfig(
            epochs=1,
            batch_size=2,
            augment=False,
            lambda2=0.0,
            lambda3=0.0,
            lambda_orth=0.01,
            lambda_adv=0.1,
            lambda_private=0.1,
            early_stop_patience=1,
        ),
        schedule=MultiDatasetSchedule(sampling="balanced"),
        device=torch.device("cpu"),
    )
    assert all(branch.optimizer is trainer.optimizer for branch in trainer.branch_trainers.values())

    before = model.branch("gtn").dataset_adapter.up.detach().clone()
    history = trainer.fit(
        {
            "gtn": _loader(3, 2, seed=1),
            "dry8": _loader(8, 1, seed=2),
        }
    )
    after = model.branch("gtn").dataset_adapter.up.detach()

    assert len(history["train_losses"]) == 1
    assert set(history["train_losses_by_domain"]) == {"gtn", "dry8"}
    assert not torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])


def test_multidataset_trainer_honors_shared_cosine_schedule() -> None:
    trainer = MultiDatasetTrainer(
        _model(),
        TrainerConfig(
            epochs=2,
            batch_size=2,
            augment=False,
            lambda2=0.0,
            lambda3=0.0,
            lr=1e-3,
            lr_schedule="cosine",
            lr_warmup_fraction=0.0,
            min_lr_ratio=0.1,
        ),
        schedule=MultiDatasetSchedule(sampling="balanced"),
        device=torch.device("cpu"),
    )

    trainer.fit({"gtn": _loader(3, 2, seed=1), "dry8": _loader(8, 1, seed=2)})

    main = trainer.branch_trainers[trainer.main_domain]
    assert main.planned_optimizer_steps == 8
    assert main.lr_scheduler is not None
    assert all(
        branch.lr_scheduler is main.lr_scheduler
        for branch in trainer.branch_trainers.values()
    )
    assert trainer.optimizer.param_groups[0]["lr"] < 1e-3


def test_multidataset_trainer_rejects_marginal_mmd_for_homogeneous_batches() -> None:
    import pytest

    with pytest.raises(ValueError, match="MMD"):
        MultiDatasetTrainer(
            _model(),
            TrainerConfig(epochs=1, lambda4=0.1),
            device=torch.device("cpu"),
        )


def test_multidataset_trainer_preserves_trial_masks_through_gp_projection() -> None:
    model = _model()
    static_mask = torch.tensor([True, True, False])
    trainer = MultiDatasetTrainer(
        model,
        TrainerConfig(
            epochs=1,
            batch_size=2,
            augment=False,
            lambda2=0.0,
            lambda3=0.0,
            lambda_amp=0.0,
        ),
        channel_masks={"gtn": static_mask, "dry8": torch.ones(8, dtype=torch.bool)},
        device=torch.device("cpu"),
    )
    dynamic_mask = torch.tensor([[True, False, True], [False, True, True]])
    X = torch.randn(2, 3, 64) * dynamic_mask[:, :, None]
    context = TrialContext(X, torch.tensor([0.0, 1.0]), channel_mask=dynamic_mask)
    projector = model.branch("gtn").tokenizer.canonical_projector
    seen: list[torch.Tensor] = []
    original_forward = projector.forward

    def spy_forward(values, channel_mask=None):
        seen.append(channel_mask.detach().clone())
        return original_forward(values, channel_mask=channel_mask)

    with mock.patch.object(projector, "forward", side_effect=spy_forward):
        trainer.branch_trainers["gtn"]._train_step(
            trainer._with_domain("gtn", context),
            step=0,
        )

    expected = dynamic_mask & static_mask
    assert torch.equal(seen[0], expected)
    assert torch.equal(trainer._with_domain("gtn", context).channel_mask, dynamic_mask)


def test_multidataset_early_stopping_selects_main_task_not_transfer_objective() -> None:
    trainer = MultiDatasetTrainer(
        _model(),
        TrainerConfig(
            epochs=2,
            batch_size=2,
            augment=False,
            lambda2=0.0,
            lambda3=0.0,
            early_stop_patience=2,
        ),
        device=torch.device("cpu"),
    )
    loaders = {"gtn": _loader(3, 1, seed=1), "dry8": _loader(8, 1, seed=2)}
    zero = torch.tensor(0.0)
    gtn = trainer.branch_trainers["gtn"]
    dry8 = trainer.branch_trainers["dry8"]

    with (
        mock.patch.object(gtn, "_train_step", return_value=zero),
        mock.patch.object(dry8, "_train_step", return_value=zero),
        mock.patch.object(gtn, "_evaluate", side_effect=[0.1, 1.0]),
        mock.patch.object(dry8, "_evaluate", side_effect=[0.2, 0.2]),
        mock.patch.object(gtn, "_evaluate_task", side_effect=[1.0, 0.5]),
        mock.patch.object(dry8, "_evaluate_task", side_effect=[0.8, 0.8]),
    ):
        history = trainer.fit(loaders, loaders)

    assert history["best_epoch"] == 1
    assert history["val_losses"] == [1.0, 0.5]
    assert history["val_losses_by_domain"]["gtn"] == [0.1, 1.0]


def test_multidataset_trainer_can_exclude_held_out_dataset_from_all_loaders() -> None:
    model = _model()
    trainer = MultiDatasetTrainer(
        model,
        TrainerConfig(
            epochs=1,
            batch_size=2,
            augment=False,
            lambda2=0.0,
            lambda3=0.0,
            weight_decay=1.0,
        ),
        active_domains=("gtn",),
        device=torch.device("cpu"),
    )
    adapter_down = model.branch("gtn").dataset_adapter.down[1].detach().clone()
    domain_scale = model.branch("gtn").encoder.blocks[0].dom_scale[1].detach().clone()
    domain_classifier = (
        model.branch("gtn").shared_private_encoder.domain_classifier.weight[1].detach().clone()
    )
    private_classifier = (
        model.branch("gtn").shared_private_encoder.dataset_classifier.weight[1].detach().clone()
    )

    history = trainer.fit({"gtn": _loader(3, 1, seed=1)})

    assert history["active_domains"] == ["gtn"]
    assert set(history["train_losses_by_domain"]) == {"gtn"}
    assert torch.equal(model.branch("gtn").dataset_adapter.down[1], adapter_down)
    assert torch.equal(model.branch("gtn").encoder.blocks[0].dom_scale[1], domain_scale)
    assert torch.equal(
        model.branch("gtn").shared_private_encoder.domain_classifier.weight[1],
        domain_classifier,
    )
    assert torch.equal(
        model.branch("gtn").shared_private_encoder.dataset_classifier.weight[1],
        private_classifier,
    )
