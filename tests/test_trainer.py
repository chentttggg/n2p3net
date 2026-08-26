"""模块 #14 测试：训练循环（冒烟）。

在 CPU 上运行（显式 device="cpu"）以保证沙箱内稳定。
冒烟：get_device 返回 device、单步训练更新权重、fit 跑通（含/不含 val）、early stop 保存最佳权重。
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from models.n2p3net import N2P3Net
from train.contracts import TrialContext
from train.trainer import Trainer, TrainerConfig, get_device

C = 8
D = 64
T = 256  # 与模型 1 秒 @256 Hz 的物理时间轴契约一致


class _DummyDataset(Dataset):
    def __init__(self, n, C=C, T=T):
        g = torch.Generator().manual_seed(0)
        self.X = torch.randn(n, C, T, generator=g)
        self.y = torch.randint(0, 2, (n, 1), generator=g).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_trainer(**cfg_kw):
    torch.manual_seed(0)
    needs_erp_decoder = (
        max(
            float(cfg_kw.get("lambda_recon", 0.0)),
            float(cfg_kw.get("lambda_morphology_l0", 0.0)),
        )
        > 0.0
    )
    model = N2P3Net(
        component_decoder=needs_erp_decoder,
        use_innovation_likelihood=float(cfg_kw.get("lambda_innovation", 0.0)) > 0.0,
    )
    cfg = TrainerConfig(augment=False, **cfg_kw)
    return Trainer(model, cfg, device=torch.device("cpu")), model


# ---------------- 冒烟测试 ----------------


def test_get_device_returns_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_jit_default_off():
    """方案 B：自监督 jitter 一致性默认关闭（失败诊断：与 P300 时间局域判别冲突）。"""
    cfg = TrainerConfig()
    assert cfg.lambda_jit == 0.0
    assert cfg.jit_prob == 0.0


def test_compile_mode_is_explicit_and_uses_in_place_module_compile():
    with mock.patch.object(N2P3Net, "compile", autospec=True) as compile_mock:
        trainer, model = make_trainer(compile_mode="reduce-overhead")
    compile_mock.assert_called_once_with(model, mode="reduce-overhead")
    assert trainer.model is model


def test_eager_mode_does_not_compile_and_invalid_mode_fails():
    with mock.patch.object(N2P3Net, "compile", autospec=True) as compile_mock:
        make_trainer()
    compile_mock.assert_not_called()
    with pytest.raises(ValueError, match="compile_mode"):
        make_trainer(compile_mode="fastest")


def test_optimizer_step_cosine_schedule_and_audit_history():
    trainer, _ = make_trainer(
        lr=1e-3,
        lr_schedule="cosine",
        lr_warmup_fraction=0.5,
        min_lr_ratio=0.1,
    )
    trainer._configure_lr_scheduler(total_steps=4)
    for _ in range(4):
        trainer._optimizer_step()

    assert trainer.optimizer_steps == 4
    assert trainer.planned_optimizer_steps == 4
    assert trainer._lr_history[0] == pytest.approx(5e-4)
    assert trainer._lr_history[1] == pytest.approx(1e-3)
    assert trainer._lr_history[-1] == pytest.approx(1e-4)


def test_interleaved_update_sources_are_proportional_and_complete():
    sources = list(Trainer._interleaved_update_sources(7, 3))
    assert sources == [
        "trial",
        "trial",
        "trial",
        "set",
        "trial",
        "trial",
        "set",
        "trial",
        "trial",
        "set",
    ]


def test_batch_norm_recalibration_uses_only_supplied_training_batches():
    model = N2P3Net(encoder_depth=1, encoder_norm="bn")
    trainer = Trainer(
        model,
        TrainerConfig(epochs=1, augment=False, recalibrate_batch_norm=True),
        device=torch.device("cpu"),
    )
    loader = DataLoader(_DummyDataset(8), batch_size=4, shuffle=False)
    batch_norm = model.encoder.blocks[0].ln
    batch_norm.running_mean.fill_(123.0)

    calibrated_batches = trainer._recalibrate_batch_norm(loader)

    assert calibrated_batches == 2
    assert int(batch_norm.num_batches_tracked) == 2
    assert not torch.allclose(batch_norm.running_mean, torch.full_like(batch_norm.running_mean, 123.0))


def test_erp_loss_requires_explicit_decoder_opt_in():
    model = N2P3Net(component_decoder=False)
    with pytest.raises(ValueError, match="component_decoder"):
        Trainer(
            model,
            TrainerConfig(lambda_recon=1.0, augment=False),
            device=torch.device("cpu"),
        )


def test_faithful_variance_warmup_and_linear_ramp():
    trainer, _ = make_trainer(
        lambda_recon=0.01,
        recon_nll_weight=0.12,
        variance_warmup_epochs=2,
        variance_ramp_epochs=3,
    )
    values = [trainer._variance_nll_weight_for_epoch(epoch) for epoch in range(6)]
    assert torch.allclose(torch.tensor(values), torch.tensor([0.0, 0.0, 0.04, 0.08, 0.12, 0.12]))
    progress = [trainer._variance_progress_for_epoch(epoch) for epoch in range(6)]
    assert torch.allclose(torch.tensor(progress), torch.tensor([0.0, 0.0, 1 / 3, 2 / 3, 1, 1]))


def test_audit_result_cannot_extend_training_budget():
    trainer, _ = make_trainer(epochs=6, lambda_recon=0.01)
    assert trainer._required_epoch_count() == 6


def test_train_step_updates_weights():
    trainer, model = make_trainer()
    w0 = model.tokenizer.pointwise.weight.detach().clone()
    X = torch.randn(8, C, T)
    y = torch.randint(0, 2, (8, 1)).float()
    trainer._train_step(TrialContext(X, y), step=0)
    assert not torch.allclose(w0, model.tokenizer.pointwise.weight), "单步训练后权重应更新"


def test_fit_smoke():
    trainer, _ = make_trainer(epochs=2, batch_size=8)
    loader = DataLoader(_DummyDataset(16), batch_size=8)
    history = trainer.fit(loader)
    assert len(history["train_losses"]) == 2
    assert all(not (v != v) for v in history["train_losses"]), "train_loss 不应为 NaN"


def test_fit_with_val_and_early_stop():
    """有 val_loader：跑通并保存最佳权重（early stop 逻辑）。"""
    trainer, _ = make_trainer(epochs=3, batch_size=8, early_stop_patience=1)
    train_loader = DataLoader(_DummyDataset(16), batch_size=8)
    val_loader = DataLoader(_DummyDataset(8), batch_size=8)
    events = []
    history = trainer.fit(train_loader, val_loader=val_loader, on_epoch_end=events.append)
    assert len(history["val_losses"]) >= 1
    assert len(history["task_val_aucs"]) == len(history["val_losses"])
    assert all(auc is not None and 0.0 <= auc <= 1.0 for auc in history["task_val_aucs"])
    assert len(events) == len(history["val_losses"])
    assert all("task_val_auc" in event for event in events)
    assert all(event["task_val_auc"] is not None for event in events)
    # 若触发了 early stop 或正常跑完，best_state 都应已保存（val_loss 曾改善）
    assert trainer.best_state is not None, "val 评估后应保存最佳权重"


def test_early_stopping_checkpoint_selection_starts_after_variance_ramp():
    trainer, _ = make_trainer(
        epochs=5,
        batch_size=8,
        early_stop_patience=1,
        lambda_recon=0.01,
        recon_bootstrap_samples=2,
        recon_split_half_repeats=0,
        variance_warmup_epochs=2,
        variance_ramp_epochs=2,
    )
    train_loader = DataLoader(_DummyDataset(8), batch_size=8)
    val_loader = DataLoader(_DummyDataset(8), batch_size=8)
    profile_data = _DummyDataset(8)
    events = []
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        reconstruction_context=TrialContext(profile_data.X, profile_data.y),
        on_epoch_end=events.append,
    )
    assert history["phases"] == [
        "mean_warmup",
        "mean_warmup",
        "variance_ramp",
        "joint",
        "joint",
    ]
    assert history["selection_epochs"] == [3, 4]
    assert events[0]["best_epoch"] is None
    assert events[0]["selection_active"] is False


def test_combined_validation_matches_separate_objective_and_task_evaluation():
    trainer, _ = make_trainer(epochs=1, batch_size=4)
    loader = DataLoader(_DummyDataset(8), batch_size=4)

    objective, task, density, auc = trainer._evaluate_with_task(loader)

    assert objective == pytest.approx(trainer._evaluate(loader))
    assert task == pytest.approx(trainer._evaluate_task(loader))
    assert density == 0.0
    assert auc is not None
    assert 0.0 <= auc <= 1.0


def test_validation_auc_is_none_for_single_class_loader():
    trainer, _ = make_trainer(epochs=1, batch_size=4)
    loader = [(torch.randn(4, C, T), torch.zeros(4, 1))]

    _, _, _, auc = trainer._evaluate_with_task(loader)

    assert auc is None


def test_strict_past_runs_full_budget_and_merges_task_and_density_checkpoints():
    trainer, model = make_trainer(
        epochs=4,
        batch_size=1,
        early_stop_patience=1,
        lambda_innovation=1.0,
    )
    task_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("innovation_encoder.", "innovation_decoder."))
    )
    density_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("innovation_encoder.")
    )
    epoch_number = 0

    def fake_train_step(context, step, *, record_gradient_diagnostics=False):
        nonlocal epoch_number
        epoch_number += 1
        with torch.no_grad():
            task_parameter.fill_(float(epoch_number))
            density_parameter.fill_(float(epoch_number))
        return torch.zeros((), device=trainer.device)

    task_losses = iter((1.0, 2.0, 3.0, 4.0))
    density_losses = iter((4.0, 3.0, 2.0, 1.0))
    trainer._train_step = fake_train_step
    trainer._prepare_reconstruction_profile = mock.Mock()
    trainer._evaluate_with_task = mock.Mock(
        side_effect=lambda loader, compute_auc=True: (10.0, next(task_losses), next(density_losses), None)
    )
    loader = [(torch.zeros(1, C, T), torch.zeros(1, 1))]

    history = trainer.fit(loader, val_loader=loader)

    assert len(history["train_losses"]) == 4
    assert history["best_epoch"] == 0
    assert history["best_task_epoch"] == 0
    assert history["best_density_epoch"] == 3
    assert history["best_task_val_loss"] == 1.0
    assert history["best_density_nll"] == 1.0
    assert history["val_innovation_nlls"] == [4.0, 3.0, 2.0, 1.0]
    assert history["task_patience_exhausted"] is True
    assert torch.all(task_parameter == 1.0)
    assert torch.all(density_parameter == 4.0)


def test_epoch_trajectory_captures_every_raw_epoch_before_best_restore():
    trainer, model = make_trainer(
        epochs=3,
        batch_size=1,
        early_stop_patience=3,
        epoch_trajectory_audit=True,
    )
    tracked_parameter = next(model.parameters())
    epoch_number = 0

    def fake_train_step(context, step, *, record_gradient_diagnostics=False):
        nonlocal epoch_number
        epoch_number += 1
        with torch.no_grad():
            tracked_parameter.fill_(float(epoch_number))
        return torch.zeros((), device=trainer.device)

    task_losses = iter((3.0, 1.0, 2.0))
    checkpoints = []
    trainer._train_step = fake_train_step
    trainer._evaluate_with_task = mock.Mock(
        side_effect=lambda loader, compute_auc=True: (10.0, next(task_losses), 0.0, None)
    )
    loader = [(torch.zeros(1, C, T), torch.zeros(1, 1))]

    history = trainer.fit(
        loader,
        val_loader=loader,
        on_epoch_checkpoint=lambda event, state: checkpoints.append((event, state)),
    )

    parameter_name = next(iter(model.state_dict()))
    assert [event["epoch"] for event, _ in checkpoints] == [1, 2, 3]
    assert [state[parameter_name].flatten()[0].item() for _, state in checkpoints] == [1, 2, 3]
    assert history["best_task_epoch"] == 1
    assert history["epoch_trajectory_audit"] is True
    assert checkpoints[1][0]["task_val_loss"] == 1.0
    assert torch.all(tracked_parameter == 2.0)


def test_pcw_only_still_stops_when_task_patience_is_exhausted():
    trainer, _ = make_trainer(epochs=4, batch_size=1, early_stop_patience=1)
    trainer._train_step = mock.Mock(return_value=torch.zeros((), device=trainer.device))
    task_losses = iter((1.0, 2.0, 3.0, 4.0))
    trainer._evaluate_with_task = mock.Mock(
        side_effect=lambda loader, compute_auc=True: (10.0, next(task_losses), 0.0, None)
    )
    loader = [(torch.zeros(1, C, T), torch.zeros(1, 1))]

    history = trainer.fit(loader, val_loader=loader)

    assert len(history["train_losses"]) == 2
    assert history["best_epoch"] == 0
    assert history["best_density_epoch"] is None
    assert history["task_patience_exhausted"] is True


def test_gradient_diagnostics_are_recorded_once_per_epoch():
    trainer, _ = make_trainer(epochs=2, batch_size=4, track_pcw_gradients=True)
    loader = DataLoader(_DummyDataset(8), batch_size=4)

    diagnostics = trainer.fit(loader)["pcw_gradient_diagnostics"]

    assert len(diagnostics["tau_gradient_norms"]) == 2
    assert len(diagnostics["head_gradient_norms"]) == 2
    assert len(diagnostics["pcw_classifier_gradient_norms"]) == 2
    assert len(diagnostics["pcw_path_gradient_norms"]) == 2
    assert len(diagnostics["innovation_path_gradient_norms"]) == 2


def test_pcw_only_checkpoint_selection_starts_immediately():
    trainer, _ = make_trainer(
        epochs=3,
        batch_size=8,
        early_stop_patience=1,
        variance_warmup_epochs=5,
        variance_ramp_epochs=10,
    )
    train_loader = DataLoader(_DummyDataset(8), batch_size=8)
    val_loader = DataLoader(_DummyDataset(8), batch_size=8)
    history = trainer.fit(train_loader, val_loader=val_loader)
    assert set(history["phases"]) == {"joint"}
    assert history["selection_epochs"][0] == 0


def test_accum_steps():
    """梯度累积：accum_steps=2 时每 2 步才 step。"""
    trainer, model = make_trainer(accum_steps=2)
    w0 = model.tokenizer.pointwise.weight.detach().clone()
    X = torch.randn(8, C, T)
    y = torch.randint(0, 2, (8, 1)).float()
    trainer._train_step(TrialContext(X, y), step=0)  # 第 1 步：只累积，不 step
    assert torch.allclose(w0, model.tokenizer.pointwise.weight), (
        "accum_steps=2 时第 1 步不应更新权重"
    )
    trainer._train_step(TrialContext(X, y), step=1)  # 第 2 步：step
    assert not torch.allclose(w0, model.tokenizer.pointwise.weight), "第 2 步应更新权重"


def test_channel_mask_forwarded_and_augment_safe():
    """review v6 P0-2：Trainer 必须把 channel_mask 传给 forward，且增强后缺失通道恒 0。"""
    torch.manual_seed(0)
    model = N2P3Net()
    mask = torch.tensor([True, True, True, False, False, False, False, False])
    cfg = TrainerConfig(augment=True)
    trainer = Trainer(model, cfg, channel_mask=mask, device=torch.device("cpu"))

    X = torch.randn(8, C, T)
    X[:, ~mask, :] = 0.0
    y = torch.randint(0, 2, (8, 1)).float()

    seen = {}
    orig_forward = model.forward

    def spy_forward(
        X,
        E_chn=None,
        E_sub=None,
        channel_mask=None,
        domain_id=None,
        return_attention=False,
        return_heads=True,
        return_likelihood=True,
        likelihood_input=None,
        likelihood_channel_mask=None,
        likelihood_class_means=None,
    ):
        seen["channel_mask"] = channel_mask
        seen["missing_is_zero"] = bool((X[:, ~mask, :] == 0.0).all())
        return orig_forward(
            X,
            E_chn,
            E_sub,
            channel_mask=channel_mask,
            domain_id=domain_id,
            return_attention=return_attention,
            return_heads=return_heads,
            return_likelihood=return_likelihood,
            likelihood_input=likelihood_input,
            likelihood_channel_mask=likelihood_channel_mask,
            likelihood_class_means=likelihood_class_means,
        )

    with mock.patch.object(model, "forward", side_effect=spy_forward):
        trainer._train_step(TrialContext(X, y), step=0)

    assert seen.get("channel_mask") is not None, "Trainer 未把 channel_mask 传给 forward"
    assert torch.equal(seen["channel_mask"], mask), "forward 收到的 channel_mask 不一致"
    assert seen["missing_is_zero"], "增强后缺失通道在进入 forward 前必须仍为 0"


def test_trainer_resolves_bi2014a_pz_by_name_not_channel_count():
    channels = (
        "Fp1",
        "Fp2",
        "F5",
        "AFz",
        "F6",
        "T7",
        "Cz",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "Oz",
        "O2",
    )
    model = N2P3Net(n_channels=16, channel_names=channels)
    trainer = Trainer(
        model,
        TrainerConfig(epochs=1, augment=False),
        device=torch.device("cpu"),
    )
    assert trainer.pz_channel == 10


def test_all_aux_tail_batch_does_not_drop_accumulated_gradients():
    """audit P1：accum 边界遇到全辅助零损失 batch 时，前序 GTN 梯度仍须 step。"""
    torch.manual_seed(0)
    model = N2P3Net()
    cfg = TrainerConfig(
        augment=False,
        lambda_amp=0.0,
        lambda_jit=0.0,
        lambda2=0.0,
        lambda3=0.0,
        epochs=1,
        accum_steps=2,
    )
    trainer = Trainer(model, cfg, device=torch.device("cpu"))
    w0 = model.tokenizer.pointwise.weight.detach().clone()

    class D(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            if i == 0:
                return torch.randn(C, T), torch.tensor([0.0]), torch.tensor(0)
            return torch.randn(C, T), torch.tensor([1.0]), torch.tensor(1)

    loader = DataLoader(D(), batch_size=1)
    trainer.fit(loader)
    assert not torch.allclose(w0, model.tokenizer.pointwise.weight), (
        "全辅助尾批不应清空前序 GTN 梯度而不 step"
    )


def test_preloaded_dataloader_batches_and_domain():
    """GTN-N2P3Net 性能项：预上传 loader 每 epoch 只在设备端切片。"""
    from train.preloaded import PreloadedDataLoader

    X = torch.randn(5, C, T)
    y = torch.randint(0, 2, (5, 1)).float()
    d = torch.tensor([0, 0, 1, 1, 0])
    loader = PreloadedDataLoader(X, y, d, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert all(isinstance(batch, TrialContext) for batch in batches)
    assert torch.equal(batches[-1].domain_id, torch.tensor([0]))


def test_preloaded_dataloader_keeps_per_trial_masks_aligned() -> None:
    from train.preloaded import PreloadedDataLoader

    X = torch.zeros(5, C, T)
    X[:, 0, 0] = torch.arange(5)
    y = (torch.arange(5) % 2).float()
    mask = torch.zeros(5, C, dtype=torch.bool)
    mask[torch.arange(5), torch.arange(5)] = True
    loader = PreloadedDataLoader(X, y, channel_mask=mask, batch_size=2, shuffle=True, seed=7)

    for batch in loader:
        selected_channel = batch.channel_mask.to(torch.long).argmax(dim=1)
        assert torch.equal(selected_channel, batch.X[:, 0, 0].to(torch.long))


def test_preloaded_dataloader_rejects_integer_channel_mask() -> None:
    from train.preloaded import PreloadedDataLoader

    with pytest.raises(ValueError, match="boolean dtype"):
        PreloadedDataLoader(
            torch.zeros(2, C, T),
            torch.tensor([0.0, 1.0]),
            channel_mask=torch.ones(2, C, dtype=torch.int64),
        )


def test_trainer_rejects_nonfinite_config_and_integer_static_mask() -> None:
    with pytest.raises(ValueError, match="lr must be finite"):
        Trainer(
            N2P3Net(encoder_depth=1),
            TrainerConfig(epochs=1, lr=float("nan")),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        Trainer(
            N2P3Net(encoder_depth=1),
            TrainerConfig(epochs=1),
            channel_mask=torch.ones(C, dtype=torch.int64),
            device=torch.device("cpu"),
        )


def test_preloaded_dataloader_validates_finite_once_and_marks_contexts():
    from train.preloaded import PreloadedDataLoader

    X = torch.randn(4, C, T)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = PreloadedDataLoader(X, y, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert loader.finite_validated is True
    assert batch.prevalidated is True
    batch.validate()

    bad_X = X.clone()
    bad_X[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PreloadedDataLoader(bad_X, y, batch_size=2, shuffle=False)

    unchecked = PreloadedDataLoader(
        bad_X, y, batch_size=2, shuffle=False, validate_finite=False
    )
    assert unchecked.finite_validated is False
    assert next(iter(unchecked)).prevalidated is False


def test_trainer_disables_forward_finite_check_for_prevalidated_loaders():
    from train.preloaded import PreloadedDataLoader

    model = N2P3Net(encoder_depth=1)
    trainer = Trainer(
        model,
        TrainerConfig(epochs=1, batch_size=4, augment=False),
        device=torch.device("cpu"),
    )
    X = torch.randn(4, C, T)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = PreloadedDataLoader(X, y, batch_size=4, shuffle=False)

    assert model.validate_input_finite is True
    trainer.fit(loader)
    assert model.validate_input_finite is False


def test_prevalidated_context_skips_value_checks_but_keeps_shape_checks():
    X = torch.zeros(2, C, T)
    y = torch.tensor([0.0, 1.0])
    empty_mask = torch.zeros(2, C, dtype=torch.bool)

    prevalidated = TrialContext(X, y, channel_mask=empty_mask, prevalidated=True)
    prevalidated.validate()

    with pytest.raises(ValueError, match="observed channel"):
        TrialContext(X, y, channel_mask=empty_mask).validate()
    with pytest.raises(ValueError, match="channel_mask"):
        TrialContext(
            X, y, channel_mask=torch.zeros(3, C, dtype=torch.bool), prevalidated=True
        ).validate()


def test_prevalidated_set_metadata_skips_value_checks_only():
    from train.contracts import SetMetadata

    metadata = SetMetadata(
        stimulus_digits=torch.zeros(3, dtype=torch.long),
        group_ids=torch.zeros(3, dtype=torch.long),
        repetition_ranks=-torch.ones(3, dtype=torch.long),
        sequence_ranks=-torch.ones(3, dtype=torch.long),
        prevalidated=True,
    )
    metadata.validate(3)

    with pytest.raises(ValueError, match="positive"):
        SetMetadata(
            stimulus_digits=torch.zeros(3, dtype=torch.long),
            group_ids=torch.zeros(3, dtype=torch.long),
            repetition_ranks=torch.zeros(3, dtype=torch.long),
            sequence_ranks=torch.zeros(3, dtype=torch.long),
        ).validate(3)
    with pytest.raises(ValueError, match="match the batch"):
        metadata.validate(2)


def test_trial_context_moves_per_trial_mask_to_device() -> None:
    X = torch.zeros(2, C, T)
    mask = torch.tensor(
        [
            [True, True, False, False, False, False, False, False],
            [False, True, True, False, False, False, False, False],
        ]
    )
    moved = TrialContext(X, torch.tensor([0.0, 1.0]), channel_mask=mask).to(torch.device("cpu"))

    assert moved.channel_mask.dtype == torch.bool
    assert torch.equal(moved.channel_mask, mask)


def test_reconstruction_is_skipped_when_augmentation_hides_profile_channel() -> None:
    trainer, _ = make_trainer(lambda_recon=0.5)
    trainer.reconstruction_profile = mock.Mock(channel_mask=torch.tensor([True] * C))
    complete = torch.ones(2, C, dtype=torch.bool)
    missing = complete.clone()
    missing[0, 0] = False

    assert trainer._batch_reconstruction_weight(complete, batch_size=2, channels=C) == 0.5
    assert trainer._batch_reconstruction_weight(missing, batch_size=2, channels=C) == 0.0


def test_gtn_set_loader_emits_complete_fixed_k_groups():
    from train.preloaded import GTNSetDataLoader

    k = 2
    groups = torch.arange(3).repeat_interleave(9 * k)
    digits = torch.arange(1, 10).repeat_interleave(k).repeat(3)
    y = (
        ((groups == 0) & (digits == 2))
        | ((groups == 1) & (digits == 5))
        | ((groups == 2) & (digits == 8))
    )
    X = torch.randn(len(groups), 3, T)
    loader = GTNSetDataLoader(
        X,
        y.float(),
        digits,
        groups,
        evidence_k=k,
        batch_size=36,
        shuffle=False,
        seed=0,
    )
    assert loader.n_groups_eligible == 3
    assert loader.n_sets_per_epoch == 3
    for batch in loader:
        batch_groups = batch.set_metadata.group_ids
        batch_digits = batch.set_metadata.stimulus_digits
        batch_ranks = batch.set_metadata.repetition_ranks
        for group in torch.unique(batch_groups):
            counts = torch.stack(
                [
                    ((batch_groups == group) & (batch_digits == digit)).sum()
                    for digit in range(1, 10)
                ]
            )
            assert torch.equal(counts, torch.full((9,), k))
            for digit in range(1, 10):
                ranks = batch_ranks[(batch_groups == group) & (batch_digits == digit)]
                assert torch.equal(torch.sort(ranks).values, torch.arange(k))


def test_gtn_set_loader_keeps_ragged_k_coverage_and_explicit_order():
    from train.preloaded import GTNSetDataLoader

    digits = torch.cat((torch.arange(1, 10).repeat(2), torch.arange(1, 10)))
    groups = torch.cat((torch.zeros(18, dtype=torch.long), torch.ones(9, dtype=torch.long)))
    acquisition = torch.arange(len(digits), dtype=torch.long)
    X = acquisition.float().view(-1, 1, 1).expand(-1, 3, T).clone()
    y = ((groups == 0) & (digits == 4)) | ((groups == 1) & (digits == 7))
    permutation = torch.randperm(len(digits), generator=torch.Generator().manual_seed(3))
    loader = GTNSetDataLoader(
        X[permutation],
        y[permutation].float(),
        digits[permutation],
        groups[permutation],
        acquisition_indices=acquisition[permutation],
        evidence_ks=(1, 2),
        batch_size=54,
        shuffle=False,
    )
    assert loader.n_groups_eligible == 2
    assert loader.coverage_by_k == {1: 1.0, 2: 0.5}
    batch = next(iter(loader))
    assert batch.prevalidated is True
    assert batch.set_metadata.prevalidated is True
    # Batch-wide guarantee is the minimum covered K, not the loader's max K:
    # group 0 reaches K2 while group 1 only reaches K1.
    assert batch.set_metadata.prevalidated_kmax == 1
    for group in torch.unique(batch.set_metadata.group_ids):
        rows = batch.set_metadata.group_ids == group
        order = torch.argsort(batch.set_metadata.sequence_ranks[rows])
        acquisition_values = batch.X[rows, 0, 0][order]
        assert torch.equal(acquisition_values, torch.sort(acquisition_values).values)


def test_trainer_records_active_pcw_and_digit_losses():
    from train.preloaded import GTNSetDataLoader, PreloadedDataLoader

    groups = torch.arange(2).repeat_interleave(9)
    digits = torch.arange(1, 10).repeat(2)
    y = ((groups == 0) & (digits == 3)) | ((groups == 1) & (digits == 7))
    loader = GTNSetDataLoader(
        torch.randn(18, 3, T),
        y.float(),
        digits,
        groups,
        evidence_k=1,
        batch_size=18,
        shuffle=False,
    )
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    trainer = Trainer(
        model,
        TrainerConfig(
            epochs=1,
            batch_size=18,
            augment=False,
            lambda_pcw=0.3,
            lambda_digit=0.2,
            digit_evidence_ks=(1,),
            digit_evidence_weights=(1.0,),
            lambda_amp=0.0,
        ),
        device=torch.device("cpu"),
    )
    trial_loader = PreloadedDataLoader(
        loader.X, loader.y, batch_size=18, shuffle=False, device=torch.device("cpu")
    )
    history = trainer.fit(
        trial_loader,
        val_loader=trial_loader,
        train_set_loader=loader,
        val_set_loader=loader,
    )
    components = history["train_loss_components"][0]
    assert components["pcw"] > 0.0
    assert components["digit"] > 0.0
    assert history["epoch_update_counts"] == [{"trial": 1, "set": 1, "total": 2}]
    assert history["optimizer_steps"] == 2


def test_fold_local_reconstruction_profile_and_loss_smoke():
    from train.preloaded import PreloadedDataLoader

    trainer, _ = make_trainer(
        epochs=1,
        batch_size=8,
        lambda_recon=0.01,
        recon_profile_max_trials=16,
    )
    X = torch.randn(16, C, T)
    y = torch.tensor([0, 1] * 8).float()
    loader = PreloadedDataLoader(X, y, batch_size=8, shuffle=False, device=torch.device("cpu"))
    history = trainer.fit(loader, reconstruction_context=loader.full_context)
    profile = history["reconstruction_profile"]
    assert profile["scope"] == "optimization_train_only"
    assert len(profile["weights"]) == 5
    assert abs(sum(profile["weights"]) - 1.0) < 1e-5
    assert profile["bootstrap_samples"] == 64
    assert profile["split_half_repeats"] == 16
    assert -1.0 <= profile["split_half_correlation"] <= 1.0
    assert profile["target_variance_mean"] >= 0.0


def test_generative_profile_fit_stays_on_cpu(monkeypatch):
    """Fold-level AR statistics must not allocate lagged matrices on CUDA."""
    import train.trainer as trainer_module

    trainer, _ = make_trainer(
        epochs=1,
        batch_size=8,
        lambda_innovation=1.0,
        innovation_ar_order=4,
        recon_profile_max_trials=16,
    )
    X = torch.randn(16, C, T)
    y = torch.tensor([0, 1] * 8).float()
    context = TrialContext(X, y)
    observed_devices: list[torch.device] = []
    estimate = trainer_module.estimate_generative_profile

    def spy(observation, labels, **kwargs):
        observed_devices.append(observation.device)
        return estimate(observation, labels, **kwargs)

    monkeypatch.setattr(trainer_module, "estimate_generative_profile", spy)
    trainer._prepare_reconstruction_profile(context)

    assert observed_devices == [torch.device("cpu")]
    assert trainer.generative_profile is not None
