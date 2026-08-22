"""Phase 2 诊断 1：冻结 tokenizer/encoder，只训练 PCW + heads 时，τ 是否被分类损失拉回真值。

用法：
    .venv/Scripts/python.exe experiments/diagnose_tau_identifiability.py \
        --mode frozen_n2p3net --epochs 60 --batch-size 64

两种模式：
    frozen_n2p3net  默认 N2P3Net，冻结 tokenizer/encoder，只训 component_window + heads。
    identity_pcw    把原始 C 通道当作 token 特征直接进 PCW（等价于“恒等 tokenizer/encoder”），
                    用于排除随机冻结特征是否洗平时间信息的混淆。

判定：
    - 若分类 AUC 高但 τ 与 true_latency 不协变 -> D8 的“分类直接监督 τ”风险坐实。
    - 若 identity_pcw 能恢复而 frozen_n2p3net 不能 -> 说明瓶颈在随机冻结特征，不是 PCW 机制。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.component_window import ComponentWindow  # noqa: E402
from models.heads import MultiTaskHeads  # noqa: E402
from models.n2p3net import N2P3Net, N2P3NetOutput  # noqa: E402
from train.trainer import Trainer, TrainerConfig  # noqa: E402

SFREQ = 256.0
TMIN = -200.0
T = 256
PZ_IDX = 3


class PCWOnlyProbe(nn.Module):
    """恒等 tokenizer/encoder：X0 (B,C,T) → Z (B,T,C) → PCW + heads。"""

    def __init__(self, n_channels: int = 8, baseline_n: int = 51,
                 dtau_readout: str = "global_pool", dtau_bounds=None):
        super().__init__()
        self.n_channels = n_channels
        self.baseline_n = baseline_n
        self.sfreq = SFREQ
        self.component_window = ComponentWindow(
            d_model=n_channels, dtau_readout=dtau_readout, dtau_bounds=dtau_bounds
        )
        self.heads = MultiTaskHeads(d_model=n_channels)

    def _baseline_standardize(self, X: torch.Tensor) -> torch.Tensor:
        b = X[:, :, : self.baseline_n]
        mu = b.mean(dim=2, keepdim=True)
        std = b.std(dim=2, keepdim=True).clamp(min=1e-6)
        return (X - mu) / std

    def forward(
        self,
        X: torch.Tensor,
        E_chn=None,
        E_sub=None,
        channel_mask=None,
        domain_id=None,
        return_attention: bool = False,
        return_heads: bool = True,
    ) -> N2P3NetOutput:
        X = torch.nan_to_num(X, nan=0.0)
        X0 = self._baseline_standardize(X)
        Z = X0.transpose(1, 2)  # (B,T,C)
        if return_attention:
            H, tau, sigma, A = self.component_window(Z, return_attention=True)
        else:
            H, tau, sigma = self.component_window(Z)
            A = None
        return N2P3NetOutput(
            heads=self.heads(H) if return_heads else None,
            tau=tau,
            sigma=sigma,
            H=H,
            attention=A,
            features=Z,
        )


def make_data_jitter(
    n_target=600,
    n_nontarget=600,
    lat_min=280.0,
    lat_max=480.0,
    seed=0,
    amp=4.0,
    width_ms=31.0,
    noise_std=1.0,
):
    """每个 target 试次随机 latency jitter：诊断 2 用。"""
    rng = np.random.default_rng(seed)
    t_ms = TMIN + np.arange(T) * 1000.0 / SFREQ
    Xs, ys, true_lats = [], [], []
    for _ in range(n_target):
        lat = rng.uniform(lat_min, lat_max)
        x = (rng.standard_normal((8, T)) * noise_std).astype(np.float32)
        g = amp * np.exp(-0.5 * ((t_ms - lat) / width_ms) ** 2)
        x[PZ_IDX, :] += g.astype(np.float32)
        Xs.append(x)
        ys.append(1)
        true_lats.append(lat)
    for _ in range(n_nontarget):
        Xs.append((rng.standard_normal((8, T)) * noise_std).astype(np.float32))
        ys.append(0)
        true_lats.append(np.nan)
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    true_lats = np.array(true_lats, dtype=float)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], true_lats[idx]


def evaluate_tau_jitter(model, device, X, y, true_lats, batch_size=256, n_bins=4):
    """逐试次 jitter 诊断：Pearson r + 分 bin 的 MAE。"""
    model.eval()
    logits_list, tau_list = [], []
    Xt = torch.from_numpy(X).to(device)
    with torch.inference_mode():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i : i + batch_size]
            out = model(xb, return_attention=False)
            logits_list.append(out.heads.logit_target.cpu())
            tau_list.append(out.tau.cpu())
    logits = torch.cat(logits_list).squeeze(-1).numpy()
    tau = torch.cat(tau_list).numpy()
    auc = roc_auc_score(y, logits) if len(np.unique(y)) == 2 else float("nan")
    bacc = balanced_accuracy_score(y, ((logits - logits.mean()) > 0).astype(int))

    mask = (y == 1) & ~np.isnan(true_lats)
    true = true_lats[mask]
    pred = tau[mask, 2]
    r = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else float("nan")

    order = np.argsort(true)
    true_sorted = true[order]
    pred_sorted = pred[order]
    edges = np.linspace(0, len(true_sorted), n_bins + 1).astype(int)
    bin_rows = []
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        if sl.stop <= sl.start:
            continue
        t = true_sorted[sl]
        p = pred_sorted[sl]
        bin_rows.append((float(t.mean()), float(p.mean()), float(p.std()), int(len(t))))
    mae = float(np.mean([
        abs(row[0] - row[1]) for row in bin_rows
    ])) if bin_rows else float("nan")
    return {
        "auc": auc,
        "bacc": bacc,
        "tau0_p3b": float(model.component_window.tau0_bounded.detach().cpu()[2]),
        "pearson_r": r,
        "mae_bin_ms": mae,
        "bins": bin_rows,
    }


def make_data(
    latencies_ms=(250.0, 300.0, 400.0, 450.0, 500.0),
    n_target_per_latency=120,
    n_nontarget=600,
    seed=0,
    amp=4.0,
    width_ms=31.0,
    noise_std=1.0,
):
    rng = np.random.default_rng(seed)
    t_ms = TMIN + np.arange(T) * 1000.0 / SFREQ
    Xs, ys, true_lats = [], [], []
    for lat in latencies_ms:
        for _ in range(n_target_per_latency):
            x = (rng.standard_normal((8, T)) * noise_std).astype(np.float32)
            g = amp * np.exp(-0.5 * ((t_ms - lat) / width_ms) ** 2)
            x[PZ_IDX, :] += g.astype(np.float32)
            Xs.append(x)
            ys.append(1)
            true_lats.append(lat)
    for _ in range(n_nontarget):
        x = (rng.standard_normal((8, T)) * noise_std).astype(np.float32)
        Xs.append(x)
        ys.append(0)
        true_lats.append(np.nan)

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    true_lats = np.array(true_lats, dtype=float)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], true_lats[idx]


def evaluate_tau(model, device, X, y, true_lats, batch_size=256):
    """返回 dict：AUC/bacc、各条件 mean tau、MAE(条件均值 vs 真值)、Pearson r。"""
    model.eval()
    logits_list, tau_list = [], []
    Xt = torch.from_numpy(X).to(device)
    with torch.inference_mode():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i : i + batch_size]
            out = model(xb, return_attention=False)
            logits_list.append(out.heads.logit_target.cpu())
            tau_list.append(out.tau.cpu())
    logits = torch.cat(logits_list).squeeze(-1).numpy()
    tau = torch.cat(tau_list).numpy()  # (N,3)

    auc = roc_auc_score(y, logits) if len(np.unique(y)) == 2 else float("nan")
    bacc = balanced_accuracy_score(y, ((logits - logits.mean()) > 0).astype(int))

    rows = []
    pred_means = []
    for lat in sorted(set(true_lats[~np.isnan(true_lats)])):
        mask = (y == 1) & (true_lats == lat)
        mean_tau = float(tau[mask, 2].mean())
        std_tau = float(tau[mask, 2].std())
        rows.append((lat, mean_tau, std_tau, int(mask.sum())))
        pred_means.append(mean_tau)

    true_means = [lat for lat, _, _, _ in rows]
    mae = float(np.mean(np.abs(np.array(pred_means) - np.array(true_means))))
    r = float(np.corrcoef(true_means, pred_means)[0, 1]) if len(true_means) > 1 else float("nan")

    return {
        "auc": auc,
        "bacc": bacc,
        "tau0_p3b": float(model.component_window.tau0_bounded.detach().cpu()[2]),
        "per_condition": rows,
        "mae_ms": mae,
        "pearson_r": r,
    }


def make_model(mode: str, device, dtau_readout: str = "global_pool", dtau_bounds=None):
    if mode == "frozen_n2p3net":
        model = N2P3Net(dtau_readout=dtau_readout, dtau_bounds=dtau_bounds).to(device)
        for name, param in model.named_parameters():
            if name.startswith("tokenizer.") or name.startswith("encoder."):
                param.requires_grad = False
        return model
    if mode == "identity_pcw":
        return PCWOnlyProbe(dtau_readout=dtau_readout, dtau_bounds=dtau_bounds).to(device)
    if mode == "full_model_jitter":
        return N2P3Net(dtau_readout=dtau_readout, dtau_bounds=dtau_bounds).to(device)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("frozen_n2p3net", "identity_pcw", "full_model_jitter"),
                    default="frozen_n2p3net")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda2", type=float, default=0.0)
    ap.add_argument("--lambda3", type=float, default=1e-2)
    ap.add_argument("--lambda-amp", type=float, default=0.0)
    ap.add_argument("--lambda-jit", type=float, default=0.05,
                    help="自监督 jitter 一致性损失权重（Phase 2 τ 尺度锚定）")
    ap.add_argument("--jit-max-ms", type=float, default=40.0,
                    help="已知时间偏移的最大绝对值（ms）")
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--dtau-readout", default="attention_direct",
                    choices=("global_pool", "maxmean", "attention",
                             "attention_softargmax", "attention_direct"))
    ap.add_argument("--wide-bounds", action="store_true",
                    help="诊断专用：放宽 P3b Δτ 到 [-100,150]，让 τ 可覆盖 250-500ms")
    ap.add_argument("--attn-temp", type=float, default=1.0,
                    help="attention_direct 的 soft-argmax 温度（越大 attention 越尖锐）")
    ap.add_argument("--amp", type=float, default=4.0, help="P3b 高斯幅值（低 SNR 压力测试用）")
    ap.add_argument("--noise-std", type=float, default=1.0, help="背景噪声标准差（低 SNR 压力测试用）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.mode == "full_model_jitter":
        X, y, true_lats = make_data_jitter(
            seed=args.seed, amp=args.amp, noise_std=args.noise_std
        )
    else:
        X, y, true_lats = make_data(
            seed=args.seed, amp=args.amp, noise_std=args.noise_std
        )

    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    Xte, yte, latte = X[n_train + n_val :], y[n_train + n_val :], true_lats[n_train + n_val :]

    # 训练前模型先放 CPU，再交给 Trainer 自动迁移；这里直接预取设备会与 Trainer 内部一致。
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    dtau_bounds = ((-30.0, 30.0), (-30.0, 0.0), (-100.0, 150.0)) if args.wide_bounds else None
    model = make_model(
        args.mode, device, dtau_readout=args.dtau_readout, dtau_bounds=dtau_bounds
    )
    if args.dtau_readout == "attention_direct":
        with torch.no_grad():
            model.component_window.dtau_attn_temp.data.fill_(args.attn_temp)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[diagnose] mode={args.mode} dtau_readout={args.dtau_readout} "
          f"trainable={n_trainable}/{n_total} device={device} X={X.shape}", flush=True)

    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda_amp=args.lambda_amp,
        lambda_jit=args.lambda_jit,
        jit_max_ms=args.jit_max_ms,
        pos_weight=args.pos_weight,
        augment=False,
        seed=args.seed,
    )
    trainer = Trainer(model, cfg, device=device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).float().unsqueeze(1)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    trainer.fit(train_loader)

    if args.mode == "full_model_jitter":
        metrics = evaluate_tau_jitter(model, device, Xte, yte, latte, batch_size=256)
        print("\n[diagnose] full_model_jitter test metrics", flush=True)
        print(f"AUC={metrics['auc']:.4f} bacc={metrics['bacc']:.4f}", flush=True)
        print(f"tau0_P3b={metrics['tau0_p3b']:.2f} ms", flush=True)
        print(f"Pearson r(true_latency, tau_P3b)={metrics['pearson_r']:.4f}", flush=True)
        print(f"MAE(bin mean tau vs true)={metrics['mae_bin_ms']:.2f} ms", flush=True)
        print("bins true_mean pred_mean pred_std n:", flush=True)
        for true_mean, pred_mean, pred_std, n in metrics["bins"]:
            print(f"  {true_mean:6.1f}  {pred_mean:7.2f}  {pred_std:6.2f}  {n}", flush=True)
    else:
        metrics = evaluate_tau(model, device, Xte, yte, latte, batch_size=256)
        print("\n[diagnose] test metrics", flush=True)
        print(f"AUC={metrics['auc']:.4f} bacc={metrics['bacc']:.4f}", flush=True)
        print(f"tau0_P3b={metrics['tau0_p3b']:.2f} ms", flush=True)
        print("per_condition true_ms mean_tau std_tau n:", flush=True)
        for true_ms, mean_tau, std_tau, n in metrics["per_condition"]:
            print(f"  {true_ms:6.1f}  {mean_tau:7.2f}  {std_tau:6.2f}  {n}", flush=True)
        print(f"MAE(condition mean tau vs true)={metrics['mae_ms']:.2f} ms", flush=True)
        print(f"Pearson r(true, condition mean tau)={metrics['pearson_r']:.4f}", flush=True)


if __name__ == "__main__":
    main()
