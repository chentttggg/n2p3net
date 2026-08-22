"""模块 #7 测试：Stage 2 序列编码。

冒烟：depth 0/1/2/3、conformer/tcn、形状不变、无 NaN、异常。
语义：
    - depth=0 恒等（透传，不改变 tokenizer 输出）。
    - depth>0 序列编码生效（输出 ≠ 输入，跨时间信息混合）。
    - 域条件仿射生效（domain_id 改变输出）。
    - 参数预算透明（depth=1 conformer 在预期区间、TCN 更轻、depth=0 零参数）。
"""

from __future__ import annotations

import pytest
import torch

from models.encoder import Stage2Encoder

D = 64
T = 128


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_input(B=4, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


# ---------------- 冒烟测试 ----------------


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_forward_conformer_shapes(depth):
    enc = Stage2Encoder(depth=depth, encoder_type="conformer")
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_forward_tcn_shapes(depth):
    enc = Stage2Encoder(depth=depth, encoder_type="tcn")
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)



def test_tcn_causal_shape():
    """audit P2：TCN 提供 causal padding 选项，离线整段编码时 T 仍保持不变。"""
    enc = Stage2Encoder(depth=3, encoder_type="tcn", tcn_causal=True)
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)
    assert not torch.isnan(out).any()

def test_forward_no_nan():
    for typ in ("conformer", "tcn"):
        enc = Stage2Encoder(depth=2, encoder_type=typ)
        Z = make_input(B=4)
        out = enc(Z)
        assert not torch.isnan(out).any(), typ
        assert not torch.isinf(out).any(), typ


def test_invalid_encoder_type_raises():
    with pytest.raises(ValueError):
        Stage2Encoder(encoder_type="rnn")


def test_negative_depth_raises():
    with pytest.raises(ValueError):
        Stage2Encoder(depth=-1)


def test_tcn_depth_exceeding_dilations_raises():
    """review v6 P1：TCN depth 超过膨胀系数长度应显式报错，而不是静默截断。"""
    with pytest.raises(ValueError):
        Stage2Encoder(depth=4, encoder_type="tcn", tcn_dilations=(1, 4, 16))


def test_n_heads_divisibility_raises():
    """n_heads 须整除 d_model，否则提前报清晰错误（而非 nn.MultiheadAttention 的底层报错）。"""
    with pytest.raises(ValueError):
        Stage2Encoder(d_model=64, n_heads=3, encoder_type="conformer")


# ---------------- 语义测试 ----------------


def test_depth0_is_identity():
    """depth=0 恒等：输出严格等于输入（透传，零开销地板）。"""
    enc = Stage2Encoder(depth=0)
    Z = make_input(B=3)
    out = enc(Z)
    assert torch.equal(out, Z)


def test_depth_positive_changes_output():
    """depth>0 序列编码生效：输出 ≠ 输入（跨时间信息被混合）。"""
    for typ in ("conformer", "tcn"):
        enc = Stage2Encoder(depth=1, encoder_type=typ).eval()
        Z = make_input(B=3)
        out = enc(Z)
        assert not torch.allclose(out, Z), f"{typ} depth=1 应改变输入"


def test_domain_affine_changes_output():
    """域条件仿射生效：不同 domain_id 产生不同输出。

    域仿射初始化为恒等（scale=1, shift=0），故初始状态下不同 domain 输出相同；
    此处手动注入非恒等 scale 以验证「per-domain scale/shift」机制真实存在、且会被训练激活。
    """
    enc = Stage2Encoder(depth=1, encoder_type="conformer", n_domains=3).eval()
    with torch.no_grad():
        # 让 domain 0/1/2 的 scale 分别为 1/2/3（shift 保持 0）
        enc.blocks[0].dom_scale.copy_(torch.tensor([[1.0], [2.0], [3.0]]).expand(3, D))
        enc.blocks[0].dom_shift.copy_(torch.zeros(3, D))
    Z = make_input(B=3)
    d0 = torch.zeros(3, dtype=torch.long)
    d1 = torch.ones(3, dtype=torch.long)
    out0 = enc(Z, d0)
    out1 = enc(Z, d1)
    assert not torch.allclose(out0, out1), "不同 domain 应产生不同输出"


def test_domain_affine_disabled_without_n_domains():
    """n_domains=None 时不启用域仿射（Phase 2 零开销），传 domain_id 也应无副作用。"""
    enc = Stage2Encoder(depth=1, encoder_type="conformer", n_domains=None).eval()
    Z = make_input(B=3)
    d = torch.zeros(3, dtype=torch.long)
    out_with = enc(Z, d)
    out_without = enc(Z)
    assert torch.allclose(out_with, out_without), "n_domains=None 时 domain_id 不应影响输出"


def test_parameter_budget():
    """参数预算透明（D-budget）：depth=0 零参数、TCN 轻于 Conformer、depth=1 在预期区间。"""
    enc0 = Stage2Encoder(depth=0)
    assert count_params(enc0) == 0, "depth=0 应为恒等零参数"

    conf1 = count_params(Stage2Encoder(depth=1, encoder_type="conformer"))
    tcn3 = count_params(Stage2Encoder(depth=3, encoder_type="tcn"))
    assert 20000 <= conf1 <= 40000, f"depth=1 conformer 参数 {conf1} 应在 20k–40k 区间"
    assert tcn3 < conf1, f"TCN({tcn3}) 应轻于 Conformer({conf1})（降容备选）"
