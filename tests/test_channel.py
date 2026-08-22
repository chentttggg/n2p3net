"""data/channel.py 的两级测试（冒烟 + 语义）。"""

import numpy as np
import pytest

from data.channel import (
    HEAD_RADIUS,
    ChannelIdentity,
    build_channel_identity,
    channel_coords,
    sinusoidal_embedding,
    standard_coords,
)
from data.preprocess import STANDARD_CHANNELS


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_standard_coords_shape():
    c = standard_coords()
    assert c.shape == (8, 3)
    assert c.dtype == np.float64


def test_sinusoidal_embedding_shape_dtype():
    coords = standard_coords()
    e = sinusoidal_embedding(coords)  # 默认 n_freqs=8 → 6*8=48 维
    assert e.shape == (8, 48)
    assert e.dtype == np.float32


def test_build_channel_identity_default():
    ident = build_channel_identity()
    assert isinstance(ident, ChannelIdentity)
    assert ident.n_channels == 8
    assert ident.dim == 48
    assert ident.embedding.shape == (8, 48)
    assert ident.embedding.dtype == np.float32
    assert ident.mask.all()  # 默认全部存在
    assert ident.coords.shape == (8, 3)


# --------------------------------------------------------------------------- #
# 语义测试
# --------------------------------------------------------------------------- #
def test_standard_coords_topology():
    """坐标拓扑：Fz 前、Oz 后、P3 左、P4 右、Cz 顶。"""
    c = standard_coords()
    idx = {ch: i for i, ch in enumerate(STANDARD_CHANNELS)}

    assert c[idx["Fz"], 1] > 0     # y 前（+）
    assert c[idx["Oz"], 1] < 0     # y 后（-）
    assert c[idx["P3"], 0] < 0     # x 左（-）
    assert c[idx["P4"], 0] > 0     # x 右（+）
    assert c[idx["Cz"], 2] == pytest.approx(c[:, 2].max(), rel=1e-3)  # Cz 顶（z 最大）


def test_standard_coords_near_unit_sphere():
    """归一化：所有坐标范数接近 1（头半径量级），且保留了径向差异。"""
    c = standard_coords()
    r = np.linalg.norm(c, axis=1)
    assert r.min() > 0.8   # Fz 等前额电极范数 ~0.885（真实头非完美球，更靠内）
    assert r.max() < 1.3   # Oz 等枕骨电极略超 1（真实头非完美球）


def test_channel_coords_canonical():
    """通道名规范化：'pz'、'Pz-A2' 均映射到 Pz 坐标。"""
    coords, mask = channel_coords(["pz", "Pz-A2", "PO7"])
    assert mask.all()

    std = standard_coords()
    pz = std[STANDARD_CHANNELS.index("Pz")]
    po7 = std[STANDARD_CHANNELS.index("PO7")]
    assert np.allclose(coords[0], pz)
    assert np.allclose(coords[1], pz)
    assert np.allclose(coords[2], po7)


def test_channel_coords_unknown_raises():
    with pytest.raises(ValueError):
        channel_coords(["Fz", "NOT_A_CHANNEL"])


def test_channel_coords_allow_missing():
    """allow_missing=True：未知通道记为缺失（坐标 0、mask False）。"""
    coords, mask = channel_coords(["Fz", "X99"], allow_missing=True)
    assert bool(mask[0]) and not bool(mask[1])
    assert np.all(coords[1] == 0.0)


def test_sinusoidal_embedding_deterministic():
    coords = standard_coords()
    e1 = sinusoidal_embedding(coords)
    e2 = sinusoidal_embedding(coords)
    assert np.array_equal(e1, e2)


def test_sinusoidal_embedding_distinguishes_coords():
    """不同坐标 → 不同嵌入（P3 与 P4、Fz 与 Cz 均不同）。"""
    coords = standard_coords()
    e = sinusoidal_embedding(coords)
    idx = {ch: i for i, ch in enumerate(STANDARD_CHANNELS)}
    assert not np.allclose(e[idx["P3"]], e[idx["P4"]])
    assert not np.allclose(e[idx["Fz"]], e[idx["Cz"]])


def test_sinusoidal_embedding_adjacent_similarity():
    """相邻电极（坐标近）嵌入应更相似——频率封顶后保留平滑性（P0② 回归测试）。"""
    coords = standard_coords()
    e = sinusoidal_embedding(coords)
    idx = {ch: i for i, ch in enumerate(STANDARD_CHANNELS)}

    # Pz 与 P3/P4 是相邻电极，与 Oz 相距远 → 相似度应单调
    cos_pz_p3 = _cosine(e[idx["Pz"]], e[idx["P3"]])
    cos_pz_p4 = _cosine(e[idx["Pz"]], e[idx["P4"]])
    cos_pz_oz = _cosine(e[idx["Pz"]], e[idx["Oz"]])
    assert cos_pz_p3 > cos_pz_oz
    assert cos_pz_p4 > cos_pz_oz


def test_sinusoidal_embedding_invalid_coords():
    coords = standard_coords()
    with pytest.raises(ValueError):
        sinusoidal_embedding(np.zeros((8, 2)))  # 非 (C,3)


def test_build_channel_identity_mask():
    """缺失通道：mask 标记 False，嵌入置全 0，存在通道嵌入非 0。"""
    mask = np.array([True, True, False, True, False, True, False, True])  # 缺 P3/P4/PO8
    ident = build_channel_identity(channel_mask=mask)

    assert np.array_equal(ident.mask, mask)
    for idx in (2, 4, 6):  # 缺失
        assert np.all(ident.embedding[idx] == 0)
    for idx in (0, 1, 3, 5, 7):  # 存在
        assert np.any(ident.embedding[idx] != 0)


def test_build_channel_identity_merges_mask():
    """显式 mask 与坐标可查性取「与」：传未知通道 + 显式 mask，两者共同决定。"""
    ident = build_channel_identity(["Fz", "X99"], channel_mask=[True, True])
    # X99 不在标准蒙太奇 → 即使显式 mask 为 True，最终仍为 False
    assert not ident.mask[1]
    assert np.all(ident.embedding[1] == 0)
