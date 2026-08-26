"""模块 #7 测试：Stage 2 序列编码。

冒烟：depth 0/1/2/3/4/5/6、conformer/tcn、形状不变、无 NaN、异常。
语义：
    - depth=0 恒等（透传，不改变 tokenizer 输出）。
    - depth>0 序列编码生效（输出 ≠ 输入，跨时间信息混合）。
    - 域条件仿射生效（domain_id 改变输出）。
    - 参数预算透明（depth=1 conformer 在预期区间、TCN 更轻、depth=0 零参数）。
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from models.encoder import DEFAULT_ENCODER_DEPTH, Stage2Encoder, default_tcn_dilations

D = 64
T = 128


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_input(B=4, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


# ---------------- 冒烟测试 ----------------


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4, 5, 6])
def test_forward_conformer_shapes(depth):
    enc = Stage2Encoder(depth=depth, encoder_type="conformer")
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_forward_tcn_shapes(depth):
    enc = Stage2Encoder(depth=depth, encoder_type="tcn")
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)


def test_tcn_causal_shape():
    """audit P2：TCN 提供 causal padding 选项，离线整段编码时 T 仍保持不变。"""
    enc = Stage2Encoder(depth=4, encoder_type="tcn", tcn_causal=True)
    Z = make_input(B=4)
    out = enc(Z)
    assert out.shape == (4, T, D)
    assert not torch.isnan(out).any()


def test_tcn_pointwise_linear_matches_legacy_conv1d_and_checkpoint_shape():
    """B*T Linear 调度须与旧 1x1 Conv1d 等价，且不得改变状态字典。"""

    enc = Stage2Encoder(
        depth=1,
        encoder_type="tcn",
        dropout=0.0,
        tcn_pointwise_execution="linear",
    ).eval()
    block = enc.blocks[0]
    Z = make_input(B=3, T=47)
    with torch.no_grad():
        h = block._apply_norm(Z).transpose(1, 2)
        h = block.depthwise(h)[..., : Z.shape[1]]
        legacy = Z + block.pointwise(block.act(h)).transpose(1, 2)
        scheduled = block(Z)

    assert block.pointwise.weight.shape == (D, D, 1)
    assert "blocks.0.pointwise.weight" in enc.state_dict()
    assert torch.allclose(scheduled, legacy, atol=1e-6, rtol=1e-5)


def test_tcn_pointwise_execution_validation_and_default():
    enc = Stage2Encoder(depth=1, encoder_type="tcn")
    assert enc.blocks[0].pointwise_execution == "linear"
    with pytest.raises(ValueError, match="tcn_pointwise_execution"):
        Stage2Encoder(tcn_pointwise_execution="matmul")


def test_tcn_bn_channel_first_fast_path_matches_original_layout_formula():
    enc = Stage2Encoder(
        depth=1,
        encoder_type="tcn",
        dropout=0.0,
        n_domains=2,
        norm="bn",
        tcn_pointwise_execution="linear",
        bn_momentum=0.026,
    ).eval()
    block = enc.blocks[0]
    Z = make_input(B=3, T=47)
    domain_id = torch.tensor([0, 1, 0])
    with torch.no_grad():
        normalized = block._apply_norm(Z)
        legacy = normalized * block.dom_scale[domain_id].unsqueeze(1)
        legacy = legacy + block.dom_shift[domain_id].unsqueeze(1)
        legacy = block.depthwise(legacy.transpose(1, 2))[..., : Z.shape[1]]
        legacy = F.linear(
            block.act(legacy).transpose(1, 2),
            block.pointwise.weight.squeeze(-1),
            block.pointwise.bias,
        )
        scheduled = block(Z, domain_id)

    assert block.ln.momentum == pytest.approx(0.026)
    assert torch.allclose(scheduled, Z + legacy, atol=1e-6, rtol=1e-5)


def test_tcn_bn_momentum_validation():
    with pytest.raises(ValueError, match="bn_momentum"):
        Stage2Encoder(depth=1, encoder_type="tcn", norm="bn", bn_momentum=0.0)


def test_tcn_conv1d_stays_channel_first_across_blocks_and_matches_legacy_formula():
    enc = Stage2Encoder(
        depth=4,
        encoder_type="tcn",
        dropout=0.0,
        norm="bn",
        tcn_pointwise_execution="conv1d",
    ).eval()
    Z = make_input(B=3, T=47)
    with torch.no_grad():
        legacy = Z
        for block in enc.blocks:
            legacy = block(legacy)
        legacy = enc.final_norm(legacy)
        channel_first = enc(Z)

    assert torch.allclose(channel_first, legacy, atol=1e-6, rtol=1e-5)


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


def test_depth_is_the_single_tcn_capacity_control():
    """dep alone selects the number of blocks and the canonical dilation prefix."""
    for depth, expected in ((5, [1, 4, 16, 32, 64]), (6, [1, 4, 16, 32, 64, 128])):
        enc = Stage2Encoder(depth=depth, encoder_type="tcn")
        assert len(enc.blocks) == depth
        assert enc.tcn_dilations == tuple(expected)
        assert [int(block.depthwise.dilation[0]) for block in enc.blocks] == expected


def test_default_dilation_interface_extends_from_depth():
    assert DEFAULT_ENCODER_DEPTH == 4
    assert default_tcn_dilations(0) == ()
    assert default_tcn_dilations(4) == (1, 4, 16, 32)
    assert default_tcn_dilations(6) == (1, 4, 16, 32, 64, 128)
    assert default_tcn_dilations(7)[-1] == 256


def test_default_tcn_dilations_cover_depth4():
    """The legacy dep4 dilation prefix remains unchanged."""
    enc = Stage2Encoder(depth=4, encoder_type="tcn")
    assert len(enc.blocks) == 4
    assert [int(block.depthwise.dilation[0]) for block in enc.blocks] == [1, 4, 16, 32]


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
    tcn4 = count_params(Stage2Encoder(depth=4, encoder_type="tcn"))
    assert 20000 <= conf1 <= 40000, f"depth=1 conformer 参数 {conf1} 应在 20k–40k 区间"
    assert 15000 <= tcn4 <= 20000, f"depth=4 TCN 参数 {tcn4} 应在 15k–20k 区间"
    assert tcn4 < conf1, f"TCN({tcn4}) 应轻于 Conformer({conf1})（降容备选）"
