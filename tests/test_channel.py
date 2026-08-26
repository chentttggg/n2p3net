"""data/channel.py 的两级测试（冒烟 + 语义）。"""

import mne
import numpy as np
import pytest

from data.channel import (
    STANDARD_CHANNELS,
    ChannelIdentity,
    build_channel_identity,
    channel_coords,
    fiducial_head_transform,
    register_sensor_positions,
    resolve_channel_layout,
    rigid_icp,
    sinusoidal_embedding,
    standard_coords,
)


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

    assert c[idx["Fz"], 1] > 0  # y 前（+）
    assert c[idx["Oz"], 1] < 0  # y 后（-）
    assert c[idx["P3"], 0] < 0  # x 左（-）
    assert c[idx["P4"], 0] > 0  # x 右（+）
    assert c[idx["Cz"], 2] == pytest.approx(c[:, 2].max(), rel=1e-3)  # Cz 顶（z 最大）


def test_standard_coords_are_registered_metres_not_unit_rows():
    """Coordinates preserve physical head scale and radial differences."""
    c = standard_coords()
    r = np.linalg.norm(c, axis=1)
    assert r.min() > 0.07
    assert r.max() < 0.16
    assert not np.allclose(r, r[0])


def test_resolve_channel_layout_accepts_tuple_coordinate_matrix():
    positions = ((-0.03, 0.02, 0.08), (0.0, 0.0, 0.10), (0.03, -0.02, 0.08))
    layout = resolve_channel_layout(("X1", "X2", "X3"), positions_m=positions, montage=None)
    assert np.allclose(layout.positions_m, positions)
    assert layout.registration.method == "identity_head"


def test_fiducial_registration_preserves_scale_and_builds_head_frame():
    shift = np.array([0.4, -0.2, 0.3])
    head_positions = np.array([[-0.03, 0.02, 0.08], [0.03, -0.02, 0.08]])
    fiducials = {
        "lpa": np.array([-0.07, 0.0, 0.0]) + shift,
        "rpa": np.array([0.07, 0.0, 0.0]) + shift,
        "nasion": np.array([0.0, 0.10, 0.0]) + shift,
    }
    registered, spec = register_sensor_positions(
        head_positions + shift,
        source="individual_digitization",
        coordinate_frame="digitizer",
        fiducials_m=fiducials,
    )
    assert np.allclose(registered, head_positions, atol=1e-7)
    assert spec.method == "fiducial_rigid"
    assert spec.fiducials_used == ("lpa", "rpa", "nasion")
    assert not spec.spherical_fallback


def test_fiducial_registration_matches_mne_for_asymmetric_auricular_points():
    """The head origin is the nasion projection, not the LPA/RPA midpoint."""
    fiducials = {
        "lpa": np.array([-0.073, -0.004, 0.002]),
        "rpa": np.array([0.081, 0.006, -0.003]),
        "nasion": np.array([0.012, 0.102, 0.009]),
    }
    expected = mne.transforms.get_ras_to_neuromag_trans(
        fiducials["nasion"],
        fiducials["lpa"],
        fiducials["rpa"],
    )
    actual = fiducial_head_transform(fiducials)
    assert np.allclose(actual, expected, atol=1e-12)


def test_rigid_icp_recovers_small_frame_misalignment():
    target = np.array(
        [[-0.04, 0.02, 0.08], [0.04, 0.02, 0.08], [0.0, -0.04, 0.09], [0.0, 0.0, 0.11]]
    )
    source = target + np.array([0.002, -0.001, 0.001])
    registered, _, iterations, rmse = rigid_icp(source, target)
    assert np.allclose(registered, target, atol=1e-6)
    assert iterations >= 1
    assert rmse < 1e-6


def test_unit_sphere_is_only_an_explicit_fallback():
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    with pytest.raises(ValueError, match="Unit-sphere fallback is disabled"):
        register_sensor_positions(
            positions,
            source="legacy",
            coordinate_frame="unknown",
        )
    registered, spec = register_sensor_positions(
        positions,
        source="legacy",
        coordinate_frame="unknown",
        allow_spherical_fallback=True,
    )
    assert np.allclose(np.linalg.norm(registered, axis=1), 0.1)
    assert spec.spherical_fallback


def test_channel_coords_canonical():
    """通道名规范化：参考后缀不改变电极的物理位置。"""
    coords, mask = channel_coords(["Pz-A2", "PO7"])
    assert mask.all()

    std = standard_coords()
    pz = std[STANDARD_CHANNELS.index("Pz")]
    po7 = std[STANDARD_CHANNELS.index("PO7")]
    assert np.allclose(coords[0], pz)
    assert np.allclose(coords[1], po7)


def test_channel_coords_rejects_duplicate_physical_sensor():
    """同一物理电极的两个标签不能伪装成两个独立传感器。"""
    with pytest.raises(ValueError, match="not unique"):
        channel_coords(["pz", "Pz-A2", "PO7"])


def test_channel_coords_unknown_raises():
    with pytest.raises(ValueError):
        channel_coords(["Fz", "NOT_A_CHANNEL"])


def test_standard_1005_resolves_arbitrary_16_channel_layout():
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
    identity = build_channel_identity(channels, allow_missing_positions=False)
    assert identity.embedding.shape == (16, 48)
    assert identity.mask.all()
    names = {name: i for i, name in enumerate(identity.names)}
    assert identity.coords[names["P3"], 0] < 0.0
    assert identity.coords[names["P4"], 0] > 0.0
    assert identity.coords[names["AFZ"], 1] > identity.coords[names["PZ"], 1]


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
