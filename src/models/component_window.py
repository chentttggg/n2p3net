"""模块 #8：参数化成分窗（Parameterized Component Window, PCW）—— 项目核心模块。

职责（blueprint Stage 2 的 2.2 + 2.3）：
    对序列表示 Z' ∈ R^{B×T×D} 用「参数化成分窗」显式读取三个 ERP 成分（N2 / P3a / P3b）的
    成分表示 H_c，并输出每个成分的潜伏期 τ_c 与窗宽 σ_c。这是「成分感知优先于黑盒」（P3）的
    核心落地，也是「可解释性」（P4）的证据来源。

核心机制（因果反转 D8）：
    先估计潜伏期 τ_c = τ0_c + Δτ_c，再用 τ_c 生成参数化软窗 A_c(t)（不对称高斯，D9），
    软对齐读取 H_c = Σ_t A_c(t)·Z'(t)。τ 是 A 的「生成参数」而非「事后统计量」，故被分类
    经 H→A→τ 直接、单调监督，逐试次 latency jitter 可被 τ 自然学到。

明确「不做」（留给下游模块）：
    - 多任务头（Stage 3，heads.py）：Head-A 分类 / Head-B 早期证据 / Head-D 幅值。
      Head-D 幅值 Â = Σ A_P3b(t)·X_Pz(t) 需要原始信号 X_Pz，而本模块只吃 Z'，故幅值读取
      由 heads.py 用本模块输出的 A（窗分布）+ 额外传入的 X 完成。

三思决策记录（供后续会话追溯）：
    D-causal-reverse 因果反转（blueprint D8）：τ 是 A 的生成参数，被分类直接监督（∂L/∂τ 经
                     H→A→τ 单调且方向正确）。旧方案「free attention → soft-argmax」已被推翻。
    D-tau-param      τ_c = τ0_c + Δτ_c。τ0 是可学习 Parameter（数据驱动，初始先验 220/300/350ms）；
                     Δτ 逐试次，由 MLP(global_pool(Z')) 预测。两者语义不同（群体平均 vs 试次抖动），
                     不耦合。
      D-tau0-bounds    τ0 有生理界约束（默认 N2 180–280、P3a 250–380、P3b 280–500ms）：
                       forward 使用 clamp 后的有效 τ0，Trainer 在 optimizer 后调用 clamp_tau0_()
                       并让 τ0 不参与 AdamW weight decay，防其被缓慢拉向 0（review v6 P1）。
    D-asym-bounds    Δτ 缩放界不对称（v3 P1 落地，防 P3a/P3b 互换坍缩，E7）：
                     P3a 只前移 Δτ∈[−30,0]ms、P3b 只后移 Δτ∈[0,30]ms、N2 双向 Δτ∈[−30,30]ms。
                     实现：Δτ = lo + (hi−lo)·sigmoid(δ)，δ 为 raw 参数。**注意**：blueprint §5 Head-C
                     的「τ0 + 50·tanh(Δτ)」是 v3 修订前的旧对称写法，已被 §4 2.2 的不对称界取代。
                     初始 δ=0 → Δτ = (lo+hi)/2（界的中间，非 0），故初始 τ 略偏离 τ0（P3a −15ms、
                     P3b +15ms）——这是「界中心初始化」的固有性质，非 bug，训练会调整。
    D-asym-gauss     不对称高斯窗（blueprint D9）：左右独立宽度 σ_up（上升沿）/σ_down（下降沿）。
                     σ_c(t) = σ_up + (σ_down−σ_up)·sigmoid((t−τ)/w)，w≈10ms 平滑过渡（处处可微）。
                     A_c(t) = softmax(−(t−τ)²/(2σ_c(t)²))。峰值严格落在 t=τ（因 (t−τ)² 在 t=τ 为 0、
                     logit 最大），符合 ERP「峰值潜伏期」惯例；σ 经 sigmoid 软映射到 [20,80]ms
                     （有界无 clamp），防退化（→0 脉冲 / →∞ 均匀）。
    D-global-sigma   σ 是全局可学习参数（blueprint §5 Head-C 的 σ_raw，非逐试次），(3,2) 即每成分
                     up/down；只有 τ 逐试次。窗宽在试次间变异小，逐试次 σ 是过度工程（P1）。
    D-global-pool-e5 Δτ 经 global_pool(Z') 估计（对 T 维均值），功能上豁免 E5（读出路径 H_c=ΣA·Z'
                     未池化、保时间分辨率），但平均池化洗平时间信息会压低 τ 定位上限——Phase 2 的
                     MAE<40ms 诊断若失败，须先排除此摘要瓶颈（换 attention-pool/首 token 读出）再
                     归因 PCW 机制（v3 P1 豁免说明）。
    D-mlp-init       dtau_mlp 最后一层「小随机初始化」（weight std=0.01、bias=0）而非完全置 0：
                     完全置 0 会让初始 dtau_raw≡0（Δτ 精确=界中间），但 weight_last=0 会**截断第一层
                     的梯度**（∂dtau_raw/∂weight_first = weight_last^T·… = 0），第一层须等 weight_last
                     脱离零才开始学（审查实测：零初始化时 dtau_mlp[0].weight.grad 全 0）。小随机让
                     初始 dtau_raw≈0（Δτ≈界中间、从先验出发）且两层梯度从第一步就非零。与 tokenizer
                     坐标调制（单层，无截断）不同——多层 + 零初始化最后一层才会踩中此坑。
    D-dtype-align    参数/buffer 在 forward 内显式对齐 Z.dtype（AMP bf16 下 float32 Parameter/buffer
                     与 bf16 token 流相加会 promote 到 float32——tokenizer 时间 PE 的教训，已实测）。
    D-device         纯 nn.Module，不硬编码设备（DP1）。

契约（输入 → 输出）：
    Z' ∈ R^{B×T×D} → (H, tau, sigma)：
        H     ∈ R^{B×3×D}  成分表示（软对齐读取，逐成分）
        tau   ∈ R^{B×3}    潜伏期（ms，逐试次）
        sigma ∈ R^{3×2}    窗宽（ms，全局；[:,0]=σ_up 上升沿、[:,1]=σ_down 下降沿）
    return_attention=True 时额外返回 A ∈ R^{B×3×T}（软窗分布，可解释性 / Head-D 用）。

依赖的决策：blueprint §4 2.2 / §5 Head-C、D8 / D9、constitution P3/P4/E2/E3/E5/E7、
    tokenizer.D-time-pe（dtype 对齐教训）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

# 三个成分的固定顺序（索引 0/1/2），供下游与可解释性输出引用。
COMPONENT_NAMES: tuple[str, ...] = ("N2", "P3a", "P3b")

# Δτ 读出路径（review v6 Phase 2 诊断后的消融轴）：
#   global_pool            当前基线：T 维平均（丢失潜伏期）
#   maxmean                峰值 + 平均，保留峰值幅度但无位置
#   attention              单 query 时间注意力池化，保留关键时间上下文
#   attention_softargmax   逐成分 attention + 显式时间质心 + 峰值特征（推荐融合）
#   attention_direct       逐成分 attention 的 soft-argmax 时间质心直接生成 Δτ（不经过 MLP）
DTAU_READOUTS: tuple[str, ...] = (
    "global_pool",
    "maxmean",
    "attention",
    "attention_softargmax",
    "attention_direct",
)


class ComponentWindow(nn.Module):
    """参数化成分窗：Z' (B,T,D) → H (B,3,D) + tau (B,3) + sigma (3,2)。

    Parameters
    ----------
    d_model : int
        隐藏维 D（默认 64）。
    tmin / tmax : float
        时间窗物理范围（ms），用于生成时间轴（默认 -200 ~ +800）。
    tau0_ms : Sequence[float]
        三个成分的先验中心潜伏期（ms），顺序 N2/P3a/P3b（默认 220/300/350）。
    tau0_bounds : Sequence[tuple[float, float]]
        每成分 τ0 的生理界 [lo, hi]（ms）。默认 N2 [180,280]、P3a [250,380]、
        P3b [280,500]；forward 使用 clamp 后的有效 τ0，且 Trainer 在 optimizer 后
        会调用 clamp_tau0_() 保持参数本身不漂移（review v6 P1）。
    sigma_bounds : Sequence[tuple[float, float]]
        每成分 σ 的 [下界, 上界]（ms）。N2 单独收窄 [20,50]，P3a/P3b [20,80]（v3 P1 / Head-B）。
    dtau_bounds : Sequence[tuple[float, float]]
        每成分 Δτ 的不对称界 [lo, hi]（ms）：N2 [−30,30]、P3a [−30,0]（只前移）、
          P3b [−50,150]（Phase 2 默认放宽，覆盖真实 P3b 300–500ms；review v6 诊断后修订）。
    smooth_w_ms : float
        σ 在 τ 处平滑过渡的宽度 w（ms，默认 10，D9）。
    dtau_readout : str
        Δτ 的读出路径（默认 "attention_direct"）：见模块常量 DTAU_READOUTS。
    """

    def __init__(
        self,
        d_model: int = 64,
        tmin: float = -200.0,
        tmax: float = 800.0,
        tau0_ms: Sequence[float] = (220.0, 300.0, 350.0),
        tau0_bounds: Sequence[tuple[float, float]] = (
            (180.0, 280.0),
            (250.0, 380.0),
            (280.0, 500.0),
        ),
        sigma_bounds: Sequence[tuple[float, float]] = ((20.0, 50.0), (20.0, 80.0), (20.0, 80.0)),
        dtau_bounds: Sequence[tuple[float, float]] = ((-30.0, 30.0), (-30.0, 0.0), (-50.0, 150.0)),
        smooth_w_ms: float = 10.0,
        dtau_readout: str = "attention_direct",
    ):
        super().__init__()
        self.d_model = d_model
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.smooth_w_ms = float(smooth_w_ms)
        if dtau_readout not in DTAU_READOUTS:
            raise ValueError(f"dtau_readout 须为 {DTAU_READOUTS}，得到 {dtau_readout!r}。")
        self.dtau_readout = dtau_readout

        # τ0 可学习（数据驱动初始化，D-tau-param）
        self.tau0 = nn.Parameter(torch.tensor(tuple(tau0_ms), dtype=torch.float32))
        self.tau0_bounds = tuple(tau0_bounds)

        # σ 全局参数（σ_up_raw / σ_down_raw），初始 0 → σ = 界中间（D-global-sigma）
        self.sigma_raw = nn.Parameter(torch.zeros(3, 2))

        # Δτ 读出：输入维度随 dtau_readout 变化。
        hidden = max(d_model // 4, 4)
        self.dtau_mlp = None
        if dtau_readout == "global_pool":
            dtau_in = d_model
            self.dtau_attn_query = None
        elif dtau_readout == "maxmean":
            dtau_in = 2 * d_model
            self.dtau_attn_query = None
        elif dtau_readout == "attention":
            dtau_in = d_model
            self.dtau_attn_query = nn.Parameter(torch.randn(d_model) * 0.1)
        elif dtau_readout == "attention_softargmax":
            dtau_in = 3 * d_model + 2 * d_model + 3
            # 每个成分一个 query，用于定位各自的 ERP 时间窗口。
            self.dtau_attn_query = nn.Parameter(torch.randn(3, d_model) * 0.1)
        else:  # attention_direct：soft-argmax 时间质心直接映射到 Δτ，不建 MLP
            self.dtau_attn_query = nn.Parameter(torch.randn(3, d_model) * 0.1)
            self.dtau_attn_temp = nn.Parameter(torch.ones(3))
            self.dtau_gain = nn.Parameter(torch.ones(3))
            self.dtau_bias = nn.Parameter(torch.zeros(3))
        if dtau_readout != "attention_direct":
            self.dtau_mlp = nn.Sequential(
                nn.Linear(dtau_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )
            nn.init.normal_(self.dtau_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.dtau_mlp[-1].bias)

        # 界（非参数，buffer）：σ 与 Δτ 的上下界
        self.register_buffer(
            "sigma_lo", torch.tensor([b[0] for b in sigma_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "sigma_hi", torch.tensor([b[1] for b in sigma_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "dtau_lo", torch.tensor([b[0] for b in dtau_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "dtau_hi", torch.tensor([b[1] for b in dtau_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "tau0_lo", torch.tensor([b[0] for b in tau0_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "tau0_hi", torch.tensor([b[1] for b in tau0_bounds], dtype=torch.float32)
        )

    @property
    def tau0_bounded(self) -> torch.Tensor:
        """生理界内的有效 τ0（forward 与 L_tau 使用；raw tau0 仍可学习）。"""
        return self.tau0.clamp(min=self.tau0_lo, max=self.tau0_hi)

    def clamp_tau0_(self) -> "ComponentWindow":
        """把 tau0 参数本身拉回生理界（review v6 P1：防 AdamW decay 拉向 0）。"""
        with torch.no_grad():
            self.tau0.data.clamp_(min=self.tau0_lo, max=self.tau0_hi)
        return self

    def _dtau_features(self, Z: torch.Tensor, tau0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """按 dtau_readout 生成 Δτ MLP 输入（见 DTAU_READOUTS）。"""
        B, T, D = Z.shape
        dtype = Z.dtype

        if self.dtau_readout == "global_pool":
            return Z.mean(dim=1)

        if self.dtau_readout == "maxmean":
            g_max = Z.max(dim=1).values  # (B,D)
            g_mean = Z.mean(dim=1)
            return torch.cat([g_max, g_mean], dim=-1)

        if self.dtau_readout == "attention":
            q = self.dtau_attn_query.to(dtype=dtype)  # (D,)
            scores = torch.einsum("btd,d->bt", Z, q) / max(float(D) ** 0.5, 1.0)
            a = torch.softmax(scores, dim=1)  # (B,T)
            return torch.einsum("bt,btd->bd", a, Z)

        # attention_softargmax：逐成分 query attention + 显式时间质心 + max/mean 峰值
        q = self.dtau_attn_query.to(dtype=dtype)  # (3,D)
        scores = torch.einsum("btd,cd->bct", Z, q) / max(float(D) ** 0.5, 1.0)
        a = torch.softmax(scores, dim=2)  # (B,3,T)
        g_attn = torch.einsum("bct,btd->bcd", a, Z).reshape(B, 3 * D)  # (B,3D)
        g_max = Z.max(dim=1).values
        g_mean = Z.mean(dim=1)
        g_peak = torch.cat([g_max, g_mean], dim=-1)  # (B,2D)
        t_attn = torch.einsum("bct,t->bc", a, t)  # (B,3) 物理 ms 时间质心
        # detach：L_tau 只应正则逐试次偏移，不监督群体先验 tau0（D-tau0-not-supervised）。
        t_norm = (t_attn - tau0.detach()[None, :]) / 50.0  # 相对先验中心，50ms 尺度
        return torch.cat([g_attn, g_peak, t_norm], dim=-1)

    def _attention_time_centroid(
        self, Z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """逐成分 attention 的 soft-argmax 时间质心 (B,3) ms。"""
        q = self.dtau_attn_query.to(dtype=Z.dtype)  # (3,D)
        scores = torch.einsum("btd,cd->bct", Z, q) / max(float(Z.shape[-1]) ** 0.5, 1.0)
        temp = self.dtau_attn_temp.to(dtype=Z.dtype)
        scores = scores * temp[None, :, None]
        a = torch.softmax(scores, dim=2)  # (B,3,T)
        return torch.einsum("bct,t->bc", a, t)

    def forward(
        self,
        Z: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Z (B,T,D) → (H, tau, sigma[, A])。

        return_attention=True 时额外返回软窗分布 A (B,3,T)。
        """
        if Z.dim() != 3:
            raise ValueError(f"Z 须为 (B,T,D)，得到 {Z.shape}。")
        B, T, _ = Z.shape
        dtype = Z.dtype
        dev = Z.device

        # 参数/buffer 对齐 dtype（D-dtype-align，AMP 兼容）；τ0 使用生理界内有效值
        tau0 = self.tau0_bounded.to(dtype)
        sigma_lo = self.sigma_lo.to(dtype)
        sigma_hi = self.sigma_hi.to(dtype)
        dtau_lo = self.dtau_lo.to(dtype)
        dtau_hi = self.dtau_hi.to(dtype)
        sigma_raw = self.sigma_raw.to(dtype)

        # 时间轴（物理时间 ms，与 tokenizer 时间 PE 同构）
        # review P2 时间轴修正：arange 采样间隔 (tmax-tmin)/T（真实采样），非 linspace 的 /(T-1)
        #（后者末点虚高到 tmax、造成 τ 读数 0~3.9ms 系统拉伸）
        t = self.tmin + torch.arange(T, device=dev, dtype=dtype) * (self.tmax - self.tmin) / T

        # σ：sigmoid 软映射到 [lo, hi]（D-asym-gauss / D-global-sigma）
        sigma = sigma_lo[:, None] + (sigma_hi[:, None] - sigma_lo[:, None]) * torch.sigmoid(
            sigma_raw
        )  # (3, 2)
        sigma_up = sigma[:, 0]  # (3,)
        sigma_down = sigma[:, 1]  # (3,)

        # Δτ：读出特征 → MLP（或 attention_direct 的 soft-argmax 时间质心）→ sigmoid 界。
        if self.dtau_readout == "attention_direct":
            t_attn = self._attention_time_centroid(Z, t)  # (B,3)
            # detach：保证 L_tau 对 tau0 的梯度只来自 τ 显式项并精确抵消（D-tau0-not-supervised）。
            mid = tau0.detach()[None, :] + 0.5 * (dtau_lo[None, :] + dtau_hi[None, :])
            halfspan = 0.5 * (dtau_hi[None, :] - dtau_lo[None, :])
            gain = self.dtau_gain.to(dtype=dtype)
            bias = self.dtau_bias.to(dtype=dtype)
            z = gain[None, :] * (t_attn - mid) / halfspan.clamp(min=1e-6) + bias[None, :]
            # tanh 软映射：中心附近斜率 ≈ gain（初始 1），远处平滑饱和到界。
            dtau = 0.5 * (dtau_lo[None, :] + dtau_hi[None, :]) + halfspan * torch.tanh(z)
        else:
            g = self._dtau_features(Z, tau0, t)
            dtau_raw = self.dtau_mlp(g)  # (B, 3)
            dtau = dtau_lo[None, :] + (dtau_hi[None, :] - dtau_lo[None, :]) * torch.sigmoid(
                dtau_raw
            )  # (B, 3)

        # τ = τ0 + Δτ（D-tau-param）
        tau = tau0[None, :] + dtau  # (B, 3)

        # 不对称高斯窗（D9）：σ_c(t) 左右独立宽度、τ 处平滑过渡
        diff = t[None, None, :] - tau[:, :, None]  # (B, 3, T) = t − τ_c
        sigma_t = sigma_up[None, :, None] + (
            sigma_down[None, :, None] - sigma_up[None, :, None]
        ) * torch.sigmoid(diff / self.smooth_w_ms)  # (B, 3, T)
        logits = -(diff ** 2) / (2.0 * sigma_t ** 2)  # (B, 3, T)
        A = torch.softmax(logits, dim=-1)  # (B, 3, T) 归一化窗

        # 软对齐读取（D-causal-reverse 的读出端，未池化、保时间分辨率）
        H = torch.einsum("bct,btd->bcd", A, Z)  # (B, 3, D)

        if return_attention:
            return H, tau, sigma, A
        return H, tau, sigma
