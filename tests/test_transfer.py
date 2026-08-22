"""P9 / transfer_policy 能力测试：辅助预训练加载、冻结、checkpoint、域隔离、辅助数据加载。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from baselines.deep import DeepBaseline, DeepConfig
from data.auxiliary import load_auxiliary, select_channels
from models.heads import HeadsOutput
from models.n2p3net import N2P3Net, N2P3NetOutput
from train.losses import (
    Losses,
    _bce_with_pos_weight,
    compute_losses,
    tau_regularization,
)
from train.trainer import Trainer, TrainerConfig


def make_output(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logit_target = torch.randn(B, 1, generator=g)
    logit_early = torch.randn(B, 1, generator=g)
    amplitude = torch.randn(B, 1, generator=g)
    return N2P3NetOutput(
        heads=HeadsOutput(
            logit_target=logit_target,
            logit_early=logit_early,
            amplitude=amplitude,
        ),
        tau=torch.randn(B, 3, generator=g),
        sigma=torch.randn(3, 2),
        H=torch.randn(B, 3, 64, generator=g),
        attention=None,
        features=torch.randn(B, 4, 64, generator=g),
    )


# ---------------- DeepBaseline P9 接口 ----------------


def test_deep_pretrained_same_weights_and_report():
    source = DeepBaseline("eegnet", n_chans=3, n_times=256, sfreq=256.0, device=torch.device("cpu"))
    state = source._make_model().state_dict()

    clf = DeepBaseline(
        "eegnet",
        n_chans=3,
        n_times=256,
        sfreq=256.0,
        config=DeepConfig(epochs=0, batch_size=16),
        device=torch.device("cpu"),
        pretrained_state_dict=state,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 3, 256)).astype(np.float32)
    y = rng.integers(0, 2, 16).astype(np.int64)
    clf.fit(X, y)

    assert any(e["event"] == "loaded" for e in clf.load_report)
    assert not any(e["event"] == "shape_mismatch" for e in clf.load_report)
    for key in ("conv_temporal.weight", "final_layer.conv_classifier.weight"):
        assert torch.equal(clf.model_.state_dict()[key], state[key])


def test_deep_pretrained_shape_mismatch_skip_and_strict_raises():
    source = DeepBaseline("eegnet", n_chans=4, n_times=256, sfreq=256.0, device=torch.device("cpu"))
    state = source._make_model().state_dict()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3, 256)).astype(np.float32)
    y = rng.integers(0, 2, 8).astype(np.int64)

    clf = DeepBaseline(
        "eegnet",
        n_chans=3,
        n_times=256,
        sfreq=256.0,
        config=DeepConfig(epochs=0),
        device=torch.device("cpu"),
        pretrained_state_dict=state,
        strict_load=False,
    )
    clf.fit(X, y)
    assert any(e["event"] == "shape_mismatch" for e in clf.load_report)

    strict = DeepBaseline(
        "eegnet",
        n_chans=3,
        n_times=256,
        sfreq=256.0,
        config=DeepConfig(epochs=0),
        device=torch.device("cpu"),
        pretrained_state_dict=state,
        strict_load=True,
    )
    with pytest.raises(ValueError, match="strict_load"):
        strict.fit(X, y)


def test_deep_freeze_prefixes():
    clf = DeepBaseline(
        "eegnet",
        n_chans=3,
        n_times=256,
        sfreq=256.0,
        config=DeepConfig(epochs=0),
        device=torch.device("cpu"),
        freeze_prefixes=("final_layer",),
    )
    rng = np.random.default_rng(0)
    clf.fit(rng.standard_normal((8, 3, 256)).astype(np.float32), rng.integers(0, 2, 8))
    for name, param in clf.model_.named_parameters():
        if name.startswith("final_layer"):
            assert not param.requires_grad
        elif name == "conv_temporal.weight":
            assert param.requires_grad


def test_deep_checkpoint_roundtrip(tmp_path: Path):
    clf = DeepBaseline("eegnet", n_chans=3, n_times=256, sfreq=256.0, device=torch.device("cpu"))
    model = clf._make_model()
    clf.model_ = model
    path = clf.save_checkpoint(tmp_path / "eegnet_pretrain.pt")
    loaded = DeepBaseline.load_state_dict_file(path)
    assert loaded.keys() == model.state_dict().keys()


# ---------------- compute_losses P9 域隔离 ----------------


def test_losses_all_aux_batch_has_zero_main_supervision():
    out = make_output(B=4)
    y = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    domain_ids = torch.tensor([1, 1, 1, 1])
    losses = compute_losses(
        out, torch.tensor([220.0, 300.0, 350.0]), y,
        lambda2=0.3, lambda_amp=0.1, domain_ids=domain_ids, main_domain=0, aux_domain=1,
    )
    assert losses.target.item() == 0.0
    assert losses.early.item() == 0.0
    assert losses.amp.item() == 0.0
    # P9：L_tau/L_jit 也不能从 aux 样本回传（aux 只允许进 L_MMD）。
    assert losses.tau.item() == 0.0
    assert losses.jit.item() == 0.0
    assert losses.total.item() == 0.0


def test_losses_mixed_domain_only_main_labels_used():
    out = make_output(B=4)
    y = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    domain_ids = torch.tensor([0, 0, 1, 1])
    losses = compute_losses(
        out, torch.tensor([220.0, 300.0, 350.0]), y,
        lambda2=0.0, lambda_amp=0.0, domain_ids=domain_ids,
    )
    expected = _bce_with_pos_weight(out.heads.logit_target[:2], y[:2], 8.0)
    assert torch.allclose(losses.target, expected)


def test_losses_tau_and_jit_main_domain_only():
    out = make_output(B=4)
    y = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    domain_ids = torch.tensor([0, 0, 1, 1])
    tau0 = torch.tensor([220.0, 300.0, 350.0])
    tau_shift = out.tau.detach().clone()
    shift_ms = torch.tensor([10.0, -8.0, 5.0, -3.0])
    losses = compute_losses(
        out, tau0, y,
        lambda2=0.0, lambda_amp=0.0, lambda3=0.2, lambda_jit=0.1,
        tau_shift=tau_shift, shift_ms=shift_ms,
        domain_ids=domain_ids, main_domain=0, aux_domain=1,
    )
    assert torch.allclose(losses.tau, tau_regularization(out.tau[:2], tau0, 50.0))
    # 混合域中 L_jit 也只作用于主域样本
    expected_jit = ((tau_shift[:2] - out.tau[:2] - shift_ms[:2, None]) ** 2).mean() / 2500.0
    assert torch.allclose(losses.jit, expected_jit, atol=1e-6)


def test_losses_all_aux_batch_tau_and_jit_zero():
    out = make_output(B=4)
    y = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    domain_ids = torch.tensor([1, 1, 1, 1])
    losses = compute_losses(
        out, torch.tensor([220.0, 300.0, 350.0]), y,
        lambda2=0.0, lambda_amp=0.0, lambda3=0.2, lambda_jit=0.1,
        tau_shift=out.tau.detach().clone(), shift_ms=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        domain_ids=domain_ids, main_domain=0, aux_domain=1, lambda4=0.0,
    )
    assert losses.tau.item() == 0.0
    assert losses.jit.item() == 0.0
    assert losses.total.item() == 0.0


def test_losses_domain_length_mismatch_raises():
    out = make_output(B=4)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    with pytest.raises(ValueError, match="domain_ids"):
        compute_losses(out, torch.tensor([220.0, 300.0, 350.0]), y, domain_ids=torch.tensor([0, 0]))


def test_trainer_accepts_domain_id_batches():
    torch.manual_seed(0)
    model = N2P3Net()
    cfg = TrainerConfig(augment=False, lambda_amp=0.0, lambda2=0.0, lambda3=0.0, epochs=1)
    trainer = Trainer(model, cfg, device=torch.device("cpu"))

    class D(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return torch.randn(8, 256), torch.tensor([i % 2], dtype=torch.float32), torch.tensor(i % 2)

    loader = DataLoader(D(), batch_size=4)
    trainer.fit(loader)
    assert len(trainer.fit(loader)["train_losses"]) == 1


def test_trainer_all_aux_batch_zero_loss_does_not_crash():
    """P9：全辅助域 batch 总损失为 0 且无计算图时，Trainer 必须跳过 backward 正常完成。"""
    torch.manual_seed(0)
    model = N2P3Net()
    cfg = TrainerConfig(
        augment=False, lambda_amp=0.0, lambda2=0.0, lambda3=0.1,
        lambda_jit=0.1, lambda4=0.0, epochs=1,
    )
    trainer = Trainer(model, cfg, device=torch.device("cpu"))

    class D(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return torch.randn(8, 256), torch.tensor([1.0]), torch.tensor(1)

    loader = DataLoader(D(), batch_size=4)
    history = trainer.fit(loader)
    assert len(history["train_losses"]) == 1
    assert history["train_losses"][0] == pytest.approx(0.0, abs=1e-6)


# ---------------- 辅助数据加载器 ----------------


def test_auxiliary_load_and_channel_select(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3, 257)).astype(np.float32)
    y = np.array(["Target", "NonTarget", "Target", "NonTarget", "Target"], dtype=object)
    np.savez(tmp_path / "bnci008.npz", X=X, y=y, channel_names=np.array(["Fz", "Cz", "Pz"], dtype=object))
    pd.DataFrame({"subject": [1, 2, 3, 4, 5]}).to_csv(tmp_path / "bnci008_metadata.csv", index=False)

    aux = load_auxiliary("bnci008", tmp_path, target_channels=("Fz", "Cz", "Pz"), n_times=256)
    assert aux.X.shape == (5, 3, 256)
    assert aux.y.tolist() == [1, 0, 1, 0, 1]
    assert aux.channel_names == ("Fz", "Cz", "Pz")
    assert aux.subject_ids.tolist() == ["1", "2", "3", "4", "5"]


def test_auxiliary_old_cache_fallback_channels(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 8, 256)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    np.savez(tmp_path / "bnci008.npz", X=X, y=y)
    aux = load_auxiliary("bnci008", tmp_path, target_channels=("Fz", "Cz", "Pz"))
    assert aux.X.shape == (4, 3, 256)
    assert aux.channel_names == ("Fz", "Cz", "Pz")


def test_auxiliary_strict_channel_error(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 16, 256)).astype(np.float32)
    np.savez(tmp_path / "bi2014a.npz", X=X, y=np.array([0, 1]))
    with pytest.raises(ValueError, match="Fz"):
        load_auxiliary("bi2014a", tmp_path, target_channels=("Fz", "Cz", "Pz"), strict_channels=True)
    aux = load_auxiliary("bi2014a", tmp_path, target_channels=("Fz", "Cz", "Pz"), strict_channels=False)
    assert aux.channel_names == ("Cz", "Pz")
