"""模块 #14 测试：训练循环（冒烟）。

在 CPU 上运行（显式 device="cpu"）以保证沙箱内稳定。
冒烟：get_device 返回 device、单步训练更新权重、fit 跑通（含/不含 val）、early stop 保存最佳权重。
"""

from __future__ import annotations

from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

from models.n2p3net import N2P3Net
from train.trainer import Trainer, TrainerConfig, get_device

C = 8
D = 64
T = 128  # 最小可用（tokenizer 最大核长 129，padding 64 < 128）


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
    model = N2P3Net()
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


def test_train_step_updates_weights():
    trainer, model = make_trainer()
    w0 = model.tokenizer.pointwise.weight.detach().clone()
    X = torch.randn(8, C, T)
    y = torch.randint(0, 2, (8, 1)).float()
    trainer._train_step(X, y, step=0)
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
    history = trainer.fit(train_loader, val_loader=val_loader)
    assert len(history["val_losses"]) >= 1
    # 若触发了 early stop 或正常跑完，best_state 都应已保存（val_loss 曾改善）
    assert trainer.best_state is not None, "val 评估后应保存最佳权重"


def test_accum_steps():
    """梯度累积：accum_steps=2 时每 2 步才 step。"""
    trainer, model = make_trainer(accum_steps=2)
    w0 = model.tokenizer.pointwise.weight.detach().clone()
    X = torch.randn(8, C, T)
    y = torch.randint(0, 2, (8, 1)).float()
    trainer._train_step(X, y, step=0)  # 第 1 步：只累积，不 step
    assert torch.allclose(w0, model.tokenizer.pointwise.weight), "accum_steps=2 时第 1 步不应更新权重"
    trainer._train_step(X, y, step=1)  # 第 2 步：step
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

    def spy_forward(X, E_chn=None, E_sub=None, channel_mask=None, domain_id=None,
                    return_attention=False, return_heads=True):
        seen["channel_mask"] = channel_mask
        seen["missing_is_zero"] = bool((X[:, ~mask, :] == 0.0).all())
        return orig_forward(X, E_chn, E_sub, channel_mask=channel_mask,
                            domain_id=domain_id, return_attention=return_attention,
                            return_heads=return_heads)

    with mock.patch.object(model, "forward", side_effect=spy_forward):
        trainer._train_step(X, y, step=0)

    assert seen.get("channel_mask") is not None, "Trainer 未把 channel_mask 传给 forward"
    assert torch.equal(seen["channel_mask"], mask), "forward 收到的 channel_mask 不一致"
    assert seen["missing_is_zero"], "增强后缺失通道在进入 forward 前必须仍为 0"



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


def test_n2p3net_domain_baseline_smoke():
    """T3（transfer_policy 方式 B）：辅助域只进 L_MMD，GTN 标签仍驱动主头。"""
    from baselines.n2p3net import N2P3NetDomainBaseline

    torch.manual_seed(0)
    rng = __import__("numpy").random.default_rng(0)
    X = rng.standard_normal((16, C, T)).astype("float32")
    y = rng.integers(0, 2, 16).astype("int64")
    aux_X = rng.standard_normal((16, C, T)).astype("float32")
    aux_y = rng.integers(0, 2, 16).astype("int64")
    adapter = N2P3NetDomainBaseline(
        model_kwargs={},
        trainer_kwargs=dict(epochs=1, batch_size=8, augment=False, lambda_jit=0.0, lambda4=0.0),
        aux_X=aux_X,
        aux_y=aux_y,
        lambda4=0.1,
        mmd_bandwidth=5.0,
        device=torch.device("cpu"),
    )
    adapter.fit(X, y)
    logits = adapter.predict_logit(X)
    assert logits.shape == (16,)
    assert (logits == logits).all()
    assert adapter.model_.encoder.blocks[0].dom_scale.shape == (2, D)
    assert adapter.model_.encoder.blocks[0].dom_shift.shape == (2, D)


def test_preloaded_dataloader_batches_and_domain():
    """GTN-N2P3Net 性能项：预上传 loader 每 epoch 只在设备端切片。"""
    from train.preloaded import PreloadedDataLoader

    X = torch.randn(5, C, T)
    y = torch.randint(0, 2, (5, 1)).float()
    d = torch.tensor([0, 0, 1, 1, 0])
    loader = PreloadedDataLoader(X, y, d, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert all(len(b) == 3 for b in batches)
    assert torch.equal(batches[-1][2], torch.tensor([0]))