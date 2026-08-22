"""data/metadata.py 的两级测试（冒烟 + 语义）。"""

import numpy as np
import pytest

from data.metadata import (
    SubjectEmbedding,
    build_subject_embedding,
    build_subject_embeddings,
    encode_age,
    encode_sex,
    normalize_sex,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_encode_age_shape():
    e = encode_age(25.0)  # 默认 n_freqs=8 → 16 维
    assert e.shape == (16,)
    assert e.dtype == np.float32


def test_encode_sex_shape():
    e = encode_sex("M")
    assert e.shape == (3,)
    assert e.dtype == np.float32


def test_build_subject_embedding_shape():
    s = build_subject_embedding(25.0, "F")
    assert isinstance(s, SubjectEmbedding)
    assert s.embedding.shape == (19,)  # 16 年龄 + 3 性别
    assert s.embedding.dtype == np.float32
    assert s.dim == 19


# --------------------------------------------------------------------------- #
# 语义测试
# --------------------------------------------------------------------------- #
def test_normalize_sex_variants():
    assert normalize_sex("M") == "M"
    assert normalize_sex("male") == "M"
    assert normalize_sex("男") == "M"
    assert normalize_sex("f") == "F"
    assert normalize_sex("FEMALE") == "F"
    assert normalize_sex("女") == "F"
    assert normalize_sex(None) == "unknown"
    assert normalize_sex("xyz") == "unknown"


def test_normalize_sex_mne_numeric_code():
    """MNE 惯例：0=unknown, 1=male, 2=female（D-sex-code）。"""
    assert normalize_sex(1) == "M"
    assert normalize_sex(2) == "F"
    assert normalize_sex(0) == "unknown"
    # 纯数字字符串同样遵循
    assert normalize_sex("1") == "M"
    assert normalize_sex("2") == "F"
    assert normalize_sex("0") == "unknown"


def test_encode_age_none_is_zero():
    e = encode_age(None)
    assert np.all(e == 0.0)


def test_encode_age_distinguishes_values():
    e1 = encode_age(20.0)
    e2 = encode_age(60.0)
    assert not np.allclose(e1, e2)


def test_encode_age_adjacent_similarity():
    """相邻年龄的编码应相似——频率封顶后保留平滑性（P0② 回归测试）。"""
    e20 = encode_age(20.0)
    e21 = encode_age(21.0)
    e60 = encode_age(60.0)

    cos_20_21 = _cosine(e20, e21)
    cos_20_60 = _cosine(e20, e60)
    assert cos_20_21 > cos_20_60   # 单调性：年龄近 → 编码更相似
    assert cos_20_21 > 0.3         # 非近似正交（修复前频率爆炸时余弦≈0）


def test_encode_sex_onehot():
    assert np.array_equal(encode_sex("M"), [1, 0, 0])
    assert np.array_equal(encode_sex("F"), [0, 1, 0])
    assert np.array_equal(encode_sex(None), [0, 0, 1])


def test_build_subject_embedding_concat():
    """E_sub = 年龄正弦编码(前 16 维) + 性别 one-hot(末 3 维)。"""
    s = build_subject_embedding(25.0, "M")
    assert np.array_equal(s.embedding[-3:], [1, 0, 0])
    assert np.allclose(s.embedding[:-3], encode_age(25.0))
    assert s.age_known is True
    assert s.sex == "M"


def test_build_subject_embedding_missing():
    """年龄/性别均缺失 → 年龄段全 0 + unknown one-hot。"""
    s = build_subject_embedding(None, None)
    assert np.all(s.embedding[:-3] == 0.0)
    assert np.array_equal(s.embedding[-3:], [0, 0, 1])
    assert s.age_known is False
    assert s.sex == "unknown"


def test_build_subject_embedding_nan_treated_as_missing():
    """CSV/DataFrame 缺失值常见为 NaN：不得产生 NaN 嵌入或 age_known=True（audit P1-2）。"""
    s = build_subject_embedding(float("nan"), float("nan"))
    assert np.all(np.isfinite(s.embedding))
    assert np.all(s.embedding[:-3] == 0.0)
    assert np.array_equal(s.embedding[-3:], [0, 0, 1])
    assert s.age_known is False
    assert s.sex == "unknown"


def test_build_subject_embedding_deterministic():
    s1 = build_subject_embedding(30.0, "F")
    s2 = build_subject_embedding(30.0, "F")
    assert np.array_equal(s1.embedding, s2.embedding)


def test_build_subject_embeddings_batch():
    emb = build_subject_embeddings([20.0, None, 55.0], ["M", "F", None])
    assert emb.shape == (3, 19)
    # 逐行验证与单被试版本一致
    assert np.allclose(emb[0], build_subject_embedding(20.0, "M").embedding)
    assert np.allclose(emb[1], build_subject_embedding(None, "F").embedding)
    assert np.allclose(emb[2], build_subject_embedding(55.0, None).embedding)


def test_build_subject_embeddings_length_mismatch():
    with pytest.raises(ValueError):
        build_subject_embeddings([20.0, 30.0], ["M"])
