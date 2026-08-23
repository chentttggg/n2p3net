"""models.reference.py 的两级测试（冒烟 + 语义）——GLM v2 门控参考层版。

语义变化（2026-08-23，D-glm-gate）：默认 init 从「强制 CAR」改为「精确恒等」
（gate=0）；CAR 行为在 gate=1 时可达。旧版在鼻参考少导数据上销毁 P3b
（GTN 实测信号损失 4.36×，见 failure_diagnosis §12），v2 用自由线性门修复。
"""

import torch
import pytest

from models.reference import WeightedRereference


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_shape_dtype():
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(4, 8, 256)
    out = ref(X)
    assert out.shape == (4, 8, 256)
    assert out.dtype == X.dtype
    assert out.requires_grad  # 参数可学习、输出可回传梯度


def test_invalid_input():
    ref = WeightedRereference(n_channels=8)
    with pytest.raises(ValueError):
        ref(torch.randn(8, 256))  # 非 (B,C,T)


# --------------------------------------------------------------------------- #
# 语义测试（GLM v2）
# --------------------------------------------------------------------------- #
def test_default_is_identity():
    """GLM v2 核心语义：gate=0 初始化 → 输出精确等于输入（保留记录参考）。

    旧版默认 CAR 在 <32 导时是无效参考（Junghöfer 2001/Luck 2014），且 softmax
    约束使恒等不可表达——这是 v1 的结构缺陷，本测试守护修复。
    """
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    assert torch.equal(ref(X), X)


def test_gate_open_is_car():
    """gate=1 + w 均匀 → 输出 = X − 通道均值 = CAR（旧行为在门全开时可达）。"""
    ref = WeightedRereference(n_channels=8)
    with torch.no_grad():
        ref.gate_raw.fill_(1.0)
    X = torch.randn(2, 8, 16)
    out = ref(X)
    car = X - X.mean(dim=1, keepdim=True)
    assert torch.allclose(out, car, atol=1e-6)


def test_gate_gradient_flows_at_identity_init():
    """反梯度饥饿（对照 tau0/旧 sigmoid 门教训）：恒等初始化下 gate 仍有健康梯度。

    sigmoid 门在 0 附近有 g(1−g) 因子（≈0），会重蹈梯度饥饿；自由线性门
    ∂out/∂g = −m(t)，与门值无关。
    """
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    ref(X).sum().backward()
    assert ref.gate_raw.grad is not None
    assert ref.gate_raw.grad.abs().sum() > 0, "恒等初始化下 gate 梯度不应为零"


def test_w_normalized():
    """w = softmax(logits)，Σw=1 且各分量 >= 0。"""
    ref = WeightedRereference(n_channels=8)
    w = ref.w
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    assert (w >= 0).all()


def test_r_matrix_correctness():
    """R = I − diag(g)·1·wᵀ，out == R @ X（外积方向 v3 修正 + v2 门）。"""
    ref = WeightedRereference(n_channels=8)
    with torch.no_grad():
        ref.gate_raw.copy_(torch.tensor([0.0, 0.5, 1.0, 0.2, 0.8, 0.0, 1.0, 0.3]))
    X = torch.randn(2, 8, 16)
    w = ref.w.detach()  # (C,)
    g = ref.gate_raw.detach()  # (C,)
    R = torch.eye(8) - torch.diag(g) @ (torch.ones(8, 1) @ w.unsqueeze(0))
    expected = R @ X
    assert torch.allclose(ref(X), expected, atol=1e-6)


def test_reference_invariance_to_common_offset_when_gate_open():
    """参考无关性（门全开时）：所有通道加同一常数偏移，输出严格不变（D-ref-invariance）。

    GLM v2：门部分开启时残留 offset·(1−g)，由下游基线段标准化（μ_b 减除）兜底；
    恒等（g=0）时输出随输入平移——这正是「保留记录参考」的语义。
    """
    torch.manual_seed(0)
    ref = WeightedRereference(n_channels=8)
    with torch.no_grad():
        ref.gate_raw.fill_(1.0)
    X = torch.randn(4, 8, 256)
    offset = 3.7  # 模拟参考电极带来的 DC 偏移
    out1 = ref(X)
    out2 = ref(X + offset)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_effective_reference_readable():
    """可解释性读数：effective_reference = g·w（训练后报告每通道有效参考权重）。"""
    ref = WeightedRereference(n_channels=8)
    with torch.no_grad():
        ref.gate_raw.fill_(0.5)
    eff = ref.effective_reference()  # (C,)
    assert eff.shape == (8,)
    assert torch.allclose(eff, 0.5 * ref.w.detach())


def test_gain():
    """use_gain=True 时输出 = (X − g ⊙ 1·wᵀX) * gain。"""
    ref = WeightedRereference(n_channels=8, use_gain=True)
    with torch.no_grad():
        ref.gate_raw.fill_(1.0)
    X = torch.randn(2, 8, 16)
    w = ref.w.detach()
    m = torch.einsum("c,bct->bt", w, X)
    expected = (X - m.unsqueeze(1)) * ref.gain.detach().view(1, -1, 1)
    assert torch.allclose(ref(X), expected, atol=1e-6)


def test_mask_weight_renormalized():
    """mask 重归一化：加权均值只由存在通道计算（w 置 0 后 renorm）。"""
    ref = WeightedRereference(n_channels=8)
    with torch.no_grad():
        ref.gate_raw.fill_(1.0)  # 打开门以检验参考路径
    X = torch.randn(2, 8, 16)
    mask = torch.tensor([True, True, False, True, False, False, False, False])
    out = ref(X, channel_mask=mask)
    # 手工：存在通道上的均匀均值（w 初始均匀 → renorm 后存在通道各 1/3）
    m_manual = X[:, mask, :].mean(dim=1)  # (B, T)
    expected_present = X[:, mask, :] - m_manual.unsqueeze(1)
    assert torch.allclose(out[:, mask, :], expected_present, atol=1e-6)


def test_mask_identity_keeps_missing_zero():
    """GLM v2：恒等初始化 + mask → 缺失通道恒 0、存在通道不变。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    mask = torch.tensor([True, True, False, True, False, False, False, False])
    X[:, ~mask, :] = 0.0
    out = ref(X, channel_mask=mask)
    assert torch.equal(out, X)  # 恒等：整体不变


def test_bf16_direct_input_keeps_dtype():
    """review v6 P1：不开 autocast、直接 bf16 输入时，参数须对齐到 X.dtype 且输出保持 bf16。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    out = ref(X)
    assert out.dtype == torch.bfloat16


def test_mask_no_phantom_channel():
    """幻象通道回归（v4）：缺失通道出口恒 0——若对所有通道减 m，缺失通道会变 −m(t)，
    被下游基线段标准化放大成 std≈1 的幻象（逐试次变化、坐标上冒充枕/顶区地形）。"""
    torch.manual_seed(0)
    ref = WeightedRereference(n_channels=8, use_gain=True)
    with torch.no_grad():
        ref.gate_raw.fill_(1.0)  # 门全开检验最坏情形
    X = torch.randn(4, 8, 256)
    mask = torch.tensor([True, True, False, True, False, False, False, False])
    X[:, ~mask, :] = 0.0  # 零填充（nan_to_num 后）
    out = ref(X, channel_mask=mask)
    assert (out[:, ~mask, :] == 0.0).all(), "缺失通道必须恒 0（含 gain 之后）"
    # 存在通道确实被减了均值（非恒等）
    assert out[:, mask, :].abs().sum() > 0


# ---------------- GLM v2：按域条件化（Phase 3 参考无关通路） ----------------


def test_per_domain_reference():
    """n_domains=2：不同 domain_id 使用不同 w/g；domain_id 缺失时回退主域（P9）。"""
    ref = WeightedRereference(n_channels=4, n_domains=2)
    with torch.no_grad():
        ref.gate_raw[0].fill_(1.0)  # 域 0：门全开（CAR 方向）
        ref.gate_raw[1].fill_(0.0)  # 域 1：恒等
        ref.w_logits[1, 0] = 10.0   # 域 1 的 w 无所谓（门关）
    X = torch.randn(3, 4, 8)
    domain = torch.tensor([0, 1, 0])
    out = ref(X, domain_id=domain)
    # 域 0（batch 0/2）：CAR
    car = X[0] - X[0].mean(dim=0, keepdim=True)
    assert torch.allclose(out[0], car, atol=1e-6)
    # 域 1（batch 1）：恒等
    assert torch.equal(out[1], X[1])
    # domain_id 缺失 → 回退主域（域 0）行为（P9：推理只吃主域）
    out_nodomain = ref(X)
    assert torch.allclose(out_nodomain[0], car, atol=1e-6)


def test_effective_reference_per_domain_shape():
    ref = WeightedRereference(n_channels=4, n_domains=3)
    eff = ref.effective_reference()
    assert eff.shape == (3, 4)
