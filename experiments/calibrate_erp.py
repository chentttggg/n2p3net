"""自适应 ERP 成分窗校准：新数据集接入的前置步骤（免人工拍超参）。

动机（2026-08-24）：τ0/σ 先验依赖数据集一致性——GTN 实测 P3b 峰值 460-490ms（旧先验
350ms 曾致 3+pt AUC 损失）；BNCI-008 用成人 380ms 拍脑袋先验后与 EEGNet 差 6.6pt。
本工具用数据本身确定成分窗，输出 JSON 直接对接 runner 的 --erp-calib。

方法（梯度/半高宽法，无人工参数）：
    1. 逐被试基线校正 → grand-average 差值曲线 d_c(t) = mean(target) − mean(nontarget)
    2. 成分峰检测（按生理搜索窗 + 显著性检验）：
       P3b：[250, 700]ms 内 d̄(t)（跨通道均值）最大值 → τ_P3b = argmax
       N2 ：[100, 350]ms 内 d̄(t) 最小值，且 |min| ≥ 0.4·|P3b 峰|（显著性守卫）
       P3a：[180, 400]ms 内 N2 与 P3b 之间的局部极大（存在性可选）
    3. σ（窗宽）：峰附近半高全宽 FWHM → σ = FWHM / 2.355（高斯等效），clamp [40, 160]
    4. τ 边界：τ ± max(80, 1.5σ) ms；σ 边界 [max(30, σ/2), min(180, 2σ)]

用法：
    .venv/Scripts/python.exe experiments/calibrate_erp.py --dataset gtn
    .venv/Scripts/python.exe experiments/calibrate_erp.py --dataset bnci008
    .venv/Scripts/python.exe experiments/run_bnci008_loso.py --models n2p3net8 \
        --erp-calib experiments/cache/erp_calib_bnci008.json

输出：experiments/cache/erp_calib_<dataset>.json（tau0_ms / tau0_bounds / sigma_bounds /
     每成分证据：峰潜伏期、幅值、FWHM、信噪比）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# 生理搜索窗（ms）：只限定搜索范围，不预设峰位（峰位由数据决定）
SEARCH = {
    "N2": (100.0, 350.0),
    "P3a": (180.0, 420.0),
    "P3b": (250.0, 700.0),
}
FWHM_FACTOR = 2.355  # 高斯等效：FWHM = 2.355σ


def _load_dataset(name: str, cache_dir: Path):
    """返回 (X (N,C,T) float32, y (N,) bool/int, subject_ids (N,) str, t_ms (T,), ch_names)"""
    if name == "gtn":
        from experiments.run_gtn_baseline import _load_gtn_cache

        X, y, digits, subject_ids, _, _ = _load_gtn_cache(
            cache_dir / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz"
        )
        t_ms = -200.0 + np.arange(X.shape[2]) * 1000.0 / 256.0
        return X.astype(np.float32), y.astype(bool), subject_ids.astype(str), t_ms, ("Fz", "Cz", "Pz")
    from data.auxiliary import load_auxiliary

    aux = load_auxiliary(name, str(cache_dir), n_times=256)
    X = aux.X.astype(np.float32)
    y = aux.y.astype(bool)
    t_ms = np.arange(X.shape[2]) * 1000.0 / 256.0  # aux epoch 0~800ms
    names = getattr(aux, "channel_names", None) or ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")
    return X, y, aux.subject_ids.astype(str), t_ms, tuple(names)


def _baseline_correct(X: np.ndarray, subject_ids: np.ndarray, n_baseline: int) -> np.ndarray:
    """逐被试基线校正（与模型前端一致；aux 数据从 0ms 起时 n_baseline 用前若干点）。"""
    X = X.astype(np.float64).copy()
    for s in np.unique(subject_ids):
        m = subject_ids == s
        b = X[m, :, :n_baseline].mean(axis=2, keepdims=True)
        X[m] -= b
    return X


def _fwhm(d: np.ndarray, t: np.ndarray, i_peak: int, amp: float) -> float:
    """峰附近半高全宽（对负峰 amp<0 亦适用）。"""
    half = amp / 2.0
    left = i_peak
    while left > 0 and (d[left] - half) * np.sign(amp) > 0:
        left -= 1
    right = i_peak
    while right < len(d) - 1 and (d[right] - half) * np.sign(amp) > 0:
        right += 1
    return float(t[right] - t[left])


def calibrate(name: str, cache_dir: Path) -> dict:
    X, y, subj, t_ms, ch_names = _load_dataset(name, cache_dir)
    n_baseline = 51 if name == "gtn" else 4  # GTN: -200~0ms；aux: 0ms 起取前 4 点
    X = _baseline_correct(X, subj, n_baseline)
    uv = 1e6 if name == "gtn" else 1.0  # GTN 缓存单位是伏特，统一转 μV 展示
    d = (X[y].mean(axis=0) - X[~y].mean(axis=0)) * uv  # (C, T) 差值曲线（μV）
    d_mean = d.mean(axis=0)  # 跨通道均值（P3b 宽正波为主部）
    noise = float(np.median([X[~y][:, :, -100:].std()] or [1.0])) * uv  # 末 400ms 静息段噪声估计（μV）

    def in_win(lo, hi):
        return np.where((t_ms >= lo) & (t_ms <= hi))[0]

    # P3b：正峰
    idx = in_win(*SEARCH["P3b"])
    i_p3b = idx[np.argmax(d_mean[idx])]
    p3b_amp = float(d_mean[i_p3b])
    p3b_tau = float(t_ms[i_p3b])
    p3b_fwhm = _fwhm(d_mean, t_ms, i_p3b, p3b_amp)
    p3b_sigma = float(np.clip(p3b_fwhm / FWHM_FACTOR, 40.0, 160.0))

    # N2：负峰（显著性守卫：幅值 ≥ 0.4×P3b）
    idx = in_win(*SEARCH["N2"])
    i_n2 = idx[np.argmin(d_mean[idx])]
    n2_amp = float(d_mean[i_n2])
    has_n2 = abs(n2_amp) >= 0.4 * abs(p3b_amp)
    n2_tau = float(t_ms[i_n2]) if has_n2 else 220.0
    n2_sigma = float(np.clip(_fwhm(d_mean, t_ms, i_n2, n2_amp) / FWHM_FACTOR, 30.0, 80.0)) if has_n2 else 40.0

    # P3a：N2 与 P3b 之间的局部极大（可选；须距 P3b 峰 ≥60ms，排除肩峰伪检）
    i_p3a, p3a_tau, p3a_sigma, has_p3a = None, 300.0, 60.0, False
    if has_n2 and i_n2 < i_p3b and t_ms[i_p3b] - t_ms[i_n2] > 150:
        seg = d_mean[i_n2:i_p3b]
        if len(seg) > 8:
            i_local = i_n2 + int(np.argmax(seg))
            if (t_ms[i_local] <= p3b_tau - 60.0
                    and (d_mean[i_local] - max(d_mean[i_n2], 0.0)) > 0.3 * p3b_amp):
                i_p3a = i_local
                p3a_tau = float(t_ms[i_local])
                p3a_sigma = float(np.clip(_fwhm(d_mean, t_ms, i_local, d_mean[i_local]) / FWHM_FACTOR, 30.0, 100.0))
                has_p3a = True

    def tau_bounds(tau, sigma):
        m = max(80.0, 1.5 * sigma)
        return (tau - m, tau + m)

    def sigma_bounds(sigma):
        return (max(30.0, sigma / 2.0), min(180.0, 2.0 * sigma))

    tau0_ms = [n2_tau, p3a_tau, p3b_tau]
    tau0_bounds = [tau_bounds(n2_tau, n2_sigma), tau_bounds(p3a_tau, p3a_sigma), tau_bounds(p3b_tau, p3b_sigma)]
    sigma_bounds = [sigma_bounds(s) for s in (n2_sigma, p3a_sigma, p3b_sigma)]

    # 每通道峰潜伏期（证据：跨通道一致性）
    per_ch = {}
    for c, ch in enumerate(ch_names):
        i_c = in_win(*SEARCH["P3b"])
        per_ch[ch] = {"p3b_peak_ms": float(t_ms[i_c[np.argmax(d[c][i_c])]]),
                      "p3b_peak_uv": float(d[c][i_c].max())}

    return {
        "dataset": name,
        "n_trials": int(len(y)),
        "n_subjects": int(len(np.unique(subj))),
        "target_ratio": float(y.mean()),
        "tau0_ms": tau0_ms,
        "tau0_bounds": [tuple(b) for b in tau0_bounds],
        "sigma_bounds": [tuple(b) for b in sigma_bounds],
        "evidence": {
            "p3b": {"peak_ms": p3b_tau, "amp_uv": p3b_amp, "fwhm_ms": p3b_fwhm,
                    "sigma_ms": p3b_sigma, "snr": p3b_amp / max(noise, 1e-9)},
            "n2": {"detected": has_n2, "peak_ms": n2_tau if has_n2 else None, "amp_uv": n2_amp},
            "p3a": {"detected": has_p3a, "peak_ms": p3a_tau if has_p3a else None},
            "per_channel": per_ch,
            "noise_uv": noise,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser(description="自适应 ERP 成分窗校准（新数据集接入前置步骤）")
    ap.add_argument("--dataset", required=True, choices=("gtn", "bnci008", "erpcore", "bi2014a"))
    ap.add_argument("--cache-dir", default="experiments/cache")
    args = ap.parse_args()

    calib = calibrate(args.dataset, Path(args.cache_dir))
    out = Path(args.cache_dir) / f"erp_calib_{args.dataset}.json"
    out.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")

    ev = calib["evidence"]
    print(f"[calib:{args.dataset}] trials={calib['n_trials']} subjects={calib['n_subjects']}")
    print(f"[calib:{args.dataset}] P3b 峰={ev['p3b']['peak_ms']:.0f}ms 幅值={ev['p3b']['amp_uv']:.1f}μV "
          f"FWHM={ev['p3b']['fwhm_ms']:.0f}ms → σ={ev['p3b']['sigma_ms']:.0f}ms SNR={ev['p3b']['snr']:.1f}")
    print(f"[calib:{args.dataset}] N2 检出={ev['n2']['detected']} "
          f"峰={ev['n2']['peak_ms']}ms；P3a 检出={ev['p3a']['detected']} 峰={ev['p3a']['peak_ms']}ms")
    print(f"[calib:{args.dataset}] tau0_ms={[round(v) for v in calib['tau0_ms']]}")
    print(f"[calib:{args.dataset}] 已写入 {out}（runner 加 --erp-calib {out} 接入）")


if __name__ == "__main__":
    main()
