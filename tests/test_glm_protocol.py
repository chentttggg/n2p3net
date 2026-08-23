"""GLM 协议测试（2026-08-22）：被试级验证早停 + TCN BN 消融 + σ 界透传。

语义：
    - _split_val_subjects：train/val 无被试交集、确定性（同 seed 同划分）、
      clamp 到 [min, max]、被试数 <4 时回退整集训练。
    - fit(subject_ids=...) 端到端：val 曲线存在、早停后 best_state 恢复。
    - evaluate() 只对声明 fit_accepts_subject_ids 的模型传 subject_ids。
    - encoder norm="bn"：可构造、前向形状不变、反向传播非 NaN。
    - N2P3Net sigma_bounds 透传到 ComponentWindow。
"""

from __future__ import annotations

import numpy as np
import torch

from models.n2p3net import N2P3Net
from train.preloaded import PreloadedDataLoader

C = 3
D = 64
T = 128


def make_adapter(**kw):
    from baselines.n2p3net import N2P3NetBaseline

    torch.manual_seed(0)
    trainer_kwargs = dict(epochs=2, batch_size=8, augment=False, lambda_jit=0.0, seed=0)
    trainer_kwargs.update(kw.pop("trainer_kwargs", {}))
    return N2P3NetBaseline(
        model_kwargs=dict(n_channels=3, channel_names=("Fz", "Cz", "Pz"), encoder_depth=1),
        trainer_kwargs=trainer_kwargs,
        device=torch.device("cpu"),
        **kw,
    )


def make_subject_data(n_subjects=8, trials_per_subject=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_subjects * trials_per_subject, C, T)).astype("float32")
    y = rng.integers(0, 2, n_subjects * trials_per_subject).astype("int64")
    s = np.repeat(np.arange(n_subjects), trials_per_subject)
    return X, y, s


# ---------------- _split_val_subjects 语义 ----------------


def test_val_split_no_subject_overlap_and_deterministic():
    adapter = make_adapter(val_subject_frac=0.25, val_subjects_min=2, val_subjects_max=4)
    X, y, s = make_subject_data(n_subjects=8)
    X1, y1, Xv, yv, k = adapter._split_val_subjects(X, y, s)
    assert k == 2  # clamp(0.25*8=2, min=2, max=4)
    assert len(X1) + len(Xv) == len(X)
    assert len(y1) + len(yv) == len(y)
    # 验证侧行数应恰为整数个被试的试次数（6/被试），即切分按被试整体、无被试跨界
    assert len(Xv) % 6 == 0 and len(Xv) == 2 * 6
    # 同 seed 复跑应得到完全一致的划分
    X1b, y1b, Xvb, yvb, kb = adapter._split_val_subjects(X, y, s)
    assert kb == k
    assert np.array_equal(X1, X1b) and np.array_equal(Xv, Xvb)
    # 被试级切分：用相同 rng 逻辑复现 val 被试集合，验证整组被试试次进 val（无跨集泄漏）
    cfg_seed = int(adapter.trainer_kwargs.get("seed", 0))
    rng = np.random.default_rng(cfg_seed)
    val_subj = set(rng.choice(np.unique(s), size=k, replace=False).tolist())
    val_mask = np.isin(s, list(val_subj))
    assert len(Xv) == int(val_mask.sum())
    assert np.array_equal(Xv, X[val_mask])
    assert np.array_equal(X1, X[~val_mask])


def test_val_split_clamps_to_bounds():
    adapter = make_adapter(val_subject_frac=0.5, val_subjects_min=2, val_subjects_max=3)
    X, y, s = make_subject_data(n_subjects=20)
    _, _, _, _, k = adapter._split_val_subjects(X, y, s)
    assert k == 3  # 0.5*20=10 → clamp 到 max=3


def test_val_split_disabled_or_too_few_subjects():
    X, y, s = make_subject_data(n_subjects=3)
    adapter = make_adapter(val_subject_frac=0.5)
    X1, y1, Xv, yv, k = adapter._split_val_subjects(X, y, s)
    assert k == 0 and len(Xv) == 0 and len(X1) == len(X)  # <4 被试回退整集
    adapter_off = make_adapter(val_subject_frac=None)
    X1, y1, Xv, yv, k = adapter_off._split_val_subjects(X, y, s)
    assert k == 0 and len(Xv) == 0


def test_fit_with_subject_ids_runs_val_early_stop():
    """端到端：fit(subject_ids=...) 产生 val 曲线且 last_val_subjects 被记录。"""
    X, y, s = make_subject_data(n_subjects=8, trials_per_subject=8)
    adapter = make_adapter(
        val_subject_frac=0.25,
        trainer_kwargs=dict(epochs=3, early_stop_patience=2),
    )
    adapter.fit(X, y, subject_ids=s)
    assert adapter.last_val_subjects == 2
    assert len(adapter.last_history["val_losses"]) >= 1
    logits = adapter.predict_logit(X[:4])
    assert logits.shape == (4,)
    assert np.isfinite(logits).all()


def test_fit_without_subject_ids_keeps_old_path():
    """不传 subject_ids（旧调用方）：无验证、行为与旧版一致。"""
    X, y, s = make_subject_data(n_subjects=8, trials_per_subject=8)
    adapter = make_adapter()
    adapter.fit(X, y)
    assert adapter.last_val_subjects is None
    assert adapter.last_history["val_losses"] == []


def test_evaluate_passes_subject_ids_only_to_supporting_models():
    """evaluate()：声明 fit_accepts_subject_ids 的模型收到 subject_ids，其余走旧契约。"""
    from baselines.evaluate import evaluate, loso_folds

    X, y, s = make_subject_data(n_subjects=6, trials_per_subject=9)
    y = (np.arange(len(y)) % 3 == 0).astype("int64")  # 两类均衡
    digits = np.tile(np.arange(1, 10), len(y) // 9 + 1)[: len(y)]
    true_digits = {int(sub): 1 for sub in np.unique(s)}

    calls = {}

    class SpyWithSubjects:
        fit_accepts_subject_ids = True

        def fit(self, X_, y_, subject_ids=None):
            calls["with_subjects"] = (len(X_), len(subject_ids))
            return self

        def predict_logit(self, X_):
            return np.zeros(len(X_))

    class SpyPlain:
        def fit(self, X_, y_):
            calls["plain"] = (len(X_), len(y_))
            return self

        def predict_logit(self, X_):
            return np.zeros(len(X_))

    folds = loso_folds(s)
    evaluate(SpyWithSubjects(), X, y, digits, s, true_digits, folds, n_jobs=1)
    assert "with_subjects" in calls and calls["with_subjects"][0] == calls["with_subjects"][1]
    evaluate(SpyPlain(), X, y, digits, s, true_digits, folds, n_jobs=1)
    assert "plain" in calls


# ---------------- encoder BN 消融 + σ 界透传 ----------------


def test_encoder_norm_bn_forward_and_backward():
    torch.manual_seed(0)
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        encoder_type="tcn",
        encoder_depth=2,
        encoder_norm="bn",
    )
    X = torch.randn(4, 3, T)
    out = model(X)
    assert out.heads.logit_target.shape == (4, 1)
    assert not torch.isnan(out.heads.logit_target).any()
    # BN 生效校验：blocks 内是 BatchNorm1d
    assert isinstance(model.encoder.blocks[0].ln, torch.nn.BatchNorm1d)
    loss = out.heads.logit_target.sum()
    loss.backward()
    assert model.encoder.blocks[0].depthwise.weight.grad is not None
    assert not torch.isnan(model.encoder.blocks[0].depthwise.weight.grad).any()


def test_encoder_norm_invalid_raises():
    from models.encoder import Stage2Encoder

    try:
        Stage2Encoder(encoder_type="tcn", depth=1, norm="gn")
    except ValueError:
        return
    raise AssertionError("norm='gn' 应抛 ValueError")


def test_sigma_bounds_forwarded_to_component_window():
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 150.0)),
    )
    assert model.component_window.sigma_hi[2].item() == 150.0
    assert model.component_window.sigma_lo[2].item() == 20.0


def test_glm_default_model_budget_within_50k():
    """GLM 默认（3 导 + separable_pool 旁路）参数预算仍 ≤50k（E4）。"""
    torch.manual_seed(0)
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    assert model.num_parameters() <= 50000


def test_glm_mini_presets_budget():
    """GLM 容量预设：mini ≈7.3k / mini_a ≈2.5k 参数（容量非瓶颈的实证载体）。"""
    torch.manual_seed(0)
    mini = N2P3Net(
        n_channels=3, channel_names=("Fz", "Cz", "Pz"),
        d_model=32, filters_per_scale=4, temporal_kernels=(33, 65), encoder_depth=1,
    )
    mini_a = N2P3Net(
        n_channels=3, channel_names=("Fz", "Cz", "Pz"),
        d_model=16, filters_per_scale=2, temporal_kernels=(65,), encoder_depth=0,
    )
    assert mini.num_parameters() < 10000
    assert mini_a.num_parameters() < 5000
    X = torch.randn(2, 3, T)
    assert mini(X).heads.logit_target.shape == (2, 1)
    assert mini_a(X).heads.logit_target.shape == (2, 1)


def test_glm_noref_forward():
    """GLM 前端修复：use_rereference=False 时 reference 层不存在（鼻参考数据默认）。"""
    torch.manual_seed(0)
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"), use_rereference=False)
    assert model.reference is None
    X = torch.randn(2, 3, T)
    out = model(X)
    assert not torch.isnan(out.heads.logit_target).any()
