"""模块 #3：subject metadata 嵌入（年龄/性别 → E_sub）。

职责（blueprint Stage 0 的 0.6「subject metadata 嵌入」）：
    把被试元数据（年龄、性别）编码为确定性嵌入 E_sub，与通道坐标嵌入 E_chn 并列输入。

明确「不做」（留给 models 层）：
    - 可学习元数据投影到隐藏维 D（与 channel 的「学习」分支同理，归 models 层）
    - 年龄对 P300 潜伏期的可学习调制（models 层 Stage 0 的职责）

三思决策记录（供后续会话追溯）：
    D-age-sin    年龄是连续标量，用与通道坐标一致的「正弦编码」映射到 2*n_freqs 维（确定性、无学习）；
                 复用 channel.sinusoidal_encode_1d，保证两类嵌入的编码风格一致。
    D-freq-cap   频率封顶（P0②，见 channel.D-freq-cap）：避免高频段在 float32 下退化为数值噪声，
                 导致相邻年龄编码近似正交、「年龄平滑调制 P300 潜伏期」归纳偏置丧失。
    D-age-norm   年龄归一化到 [0,1]（除以 MAX_AGE=100），与固定头尺度归一化后的坐标编码同量级。
    D-sex-onehot 性别是类别，用 3 维 one-hot（男/女/未知）；未知性别显式编码，不丢信息。
    D-missing    年龄缺失（None）→ 年龄编码全 0；性别缺失 → unknown one-hot [0,0,1]。缺失被显式保留，
                 models 层可据此学习「缺失」语义。
    D-sex-code   数字编码遵循 MNE 惯例（0=unknown, 1=male, 2=female），字符串另行规范化。

契约（输入 → 输出）：
    输入：age（float | None）、sex（str | int | None，接受 M/F/male/female/男/女 与 0/1/2）
    输出：SubjectEmbedding
        - embedding : (2*n_freqs + 3,) float32（前段年龄正弦，末 3 维性别 one-hot）
        - age_known : bool
        - sex       : 'M' / 'F' / 'unknown'

依赖的决策：blueprint 0.6（subject metadata 嵌入）、channel.sinusoidal_encode_1d（正弦编码复用）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data.channel import DEFAULT_N_FREQS, sinusoidal_encode_1d

# 年龄归一化参考上限（岁）。成人为主（18–80 → [0.18, 0.8]），超 100 岁会略超 1（正弦编码仍可处理）。
MAX_AGE: float = 100.0

# 性别 one-hot 的槽位顺序
_SEX_SLOTS: tuple[str, ...] = ("M", "F", "unknown")


@dataclass
class SubjectEmbedding:
    """单个被试的元数据嵌入。

    Attributes
    ----------
    embedding : np.ndarray
        (2*n_freqs + 3,) float32；前 2*n_freqs 维为年龄正弦编码，末 3 维为性别 one-hot。
    age_known : bool
        年龄是否已知（False 时年龄段为全 0）。
    sex : str
        规范化性别：'M' / 'F' / 'unknown'。
    """

    embedding: np.ndarray
    age_known: bool
    sex: str

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[0])


def normalize_sex(sex: str | int | float | None) -> str:
    """规范化性别 → 'M' / 'F' / 'unknown'。

    数字编码遵循 MNE 惯例（D-sex-code）：0=unknown, 1=male, 2=female。
    字符串接受大小写/中英文（M/F/male/female/男/女）。
    """
    if sex is None:
        return "unknown"

    # 数字（或纯数字字符串）按 MNE 惯例；NaN/inf 视为缺失（audit P1-2）。
    if isinstance(sex, (int, float)) and not isinstance(sex, bool):
        if not np.isfinite(float(sex)):
            return "unknown"
        return {1: "M", 2: "F"}.get(int(sex), "unknown")

    s = str(sex).strip().lower()
    if s in ("m", "male", "man", "男"):
        return "M"
    if s in ("f", "female", "woman", "女"):
        return "F"
    # 纯数字字符串（如 "1"、"2"）同样按 MNE 惯例
    if s.isdigit():
        return {1: "M", 2: "F"}.get(int(s), "unknown")
    return "unknown"


def encode_age(age: float | None, n_freqs: int = DEFAULT_N_FREQS) -> np.ndarray:
    """年龄 → (2*n_freqs,) 正弦编码；None/NaN/inf → 全 0（D-missing，audit P1-2）。"""
    if age is None or not np.isfinite(float(age)):
        return np.zeros(2 * n_freqs, dtype=np.float32)
    a = float(age) / MAX_AGE  # 归一化 [0,1]
    return sinusoidal_encode_1d(np.array([a], dtype=float), n_freqs)[0]


def encode_sex(sex: str | int | float | None) -> np.ndarray:
    """性别 → (3,) one-hot [男, 女, 未知]（D-sex-onehot）。"""
    onehot = np.zeros(3, dtype=np.float32)
    onehot[_SEX_SLOTS.index(normalize_sex(sex))] = 1.0
    return onehot


def build_subject_embedding(
    age: float | None,
    sex: str | int | float | None,
    *,
    n_freqs: int = DEFAULT_N_FREQS,
) -> SubjectEmbedding:
    """年龄 + 性别 → E_sub ∈ R^{2*n_freqs+3}（确定性、无学习）。

    n_freqs 控制年龄正弦编码的频段数（封顶，见 D-freq-cap）；末 3 维固定给性别 one-hot。
    """
    e_age = encode_age(age, n_freqs)
    e_sex = encode_sex(sex)
    emb = np.concatenate([e_age, e_sex]).astype(np.float32)

    age_known = bool(age is not None and np.isfinite(float(age)))
    return SubjectEmbedding(embedding=emb, age_known=age_known, sex=normalize_sex(sex))


def build_subject_embeddings(
    ages: Sequence[float | None],
    sexes: Sequence[str | int | float | None],
    *,
    n_freqs: int = DEFAULT_N_FREQS,
) -> np.ndarray:
    """批量：多个被试的 (年龄, 性别) → (N, 2*n_freqs+3) 嵌入。"""
    if len(ages) != len(sexes):
        raise ValueError(f"ages 与 sexes 长度须一致，得到 {len(ages)} vs {len(sexes)}。")
    return np.stack(
        [
            build_subject_embedding(a, s, n_freqs=n_freqs).embedding
            for a, s in zip(ages, sexes, strict=True)
        ]
    )
