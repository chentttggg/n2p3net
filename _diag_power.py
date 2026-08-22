"""Paired-outcome power diagnostics for the small-N pilots (no GPU, no new training)."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments" / "runs"


def load_hits(path):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    return {str(r["subject"]): int(r["hit"]) for r in p["records"]}


def mcnemar_and_power(hits_a, hits_b, total=242, alpha=0.05, power=0.8):
    common = sorted(set(hits_a) & set(hits_b))
    a = [hits_a[s] for s in common]
    b = [hits_b[s] for s in common]
    b_ab = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # A hit, B miss
    c_ab = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # A miss, B hit
    n = len(common)
    q = b_ab + c_ab
    p_hat = b_ab / q if q else 0.5
    p_exact = binomtest(b_ab, q, 0.5).pvalue if q else 1.0
    q_rate = q / n if n else 0.0
    q_242 = q_rate * total
    # projected power if discordant rate is unchanged at 242 subjects
    z_a = norm.ppf(1 - alpha / 2)
    z_p = norm.ppf(power)
    if 0.0 < p_hat < 1.0:
        # normal approx: q_needed = (z_a*sqrt(.25)+z_p*sqrt(p(1-p)))^2/(p-.5)^2
        q_needed = (z_a * 0.5 + z_p * math.sqrt(p_hat * (1 - p_hat))) ** 2 / (p_hat - 0.5) ** 2
        proj_power_242 = None
        # numerical normal approx power at q_242
        if q_242 > 0:
            se = math.sqrt(p_hat * (1 - p_hat) / q_242)
            z_obs = abs(p_hat - 0.5) / se
            proj_power_242 = 1 - 0.5 * math.erfc((z_obs - z_a) / math.sqrt(2))
    else:
        q_needed = float("inf")
        proj_power_242 = None
    return {
        "n": n, "discordant": q, "A_hit_B_miss": b_ab, "A_miss_B_hit": c_ab,
        "p_hat_A|discordant": p_hat, "discordant_rate": q_rate,
        "mcnemar_p": float(p_exact), "projected_discordant_242": round(q_242, 1),
        "q_needed_80pct": round(q_needed, 1),
        "projected_power_242": None if proj_power_242 is None else round(float(proj_power_242), 3),
    }


def paired_bootstrap_ci(x, y, n_boot=20000, seed=0):
    """Bootstrap CI for mean(x-y) with subject-level paired resampling."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        stats.append(d[idx].mean())
    stats = np.array(stats)
    return float(d.mean()), float(d.std(ddof=1)), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def load_folds(record_path):
    with open(record_path, encoding="utf-8") as f:
        r = json.load(f)
    folds = r["results"]["per_fold"]
    return np.array([x["hit_rate"] for x in folds]), np.array([x["balanced_acc"] for x in folds]), np.array([x["auc"] for x in folds])


pairs = [
    ("planB new30 vs old30", RUNS / "diag_planb_new30/scores/n2p3net.json", RUNS / "diag_planb_old30/scores/n2p3net.json"),
    ("planB new60 vs old60", RUNS / "planb_new60/scores/n2p3net.json", RUNS / "planb_old60/scores/n2p3net.json"),
    ("T1 erpcore vs T0", RUNS / "d_transfer/t1_erpcore/eegnet.json", RUNS / "d_transfer/t0/eegnet.json"),
    ("T1 bnci008 vs T0", RUNS / "d_transfer/t1_bnci/eegnet.json", RUNS / "d_transfer/t0/eegnet.json"),
    ("T1 bi2014a vs T0", RUNS / "d_transfer/t1_bi/eegnet.json", RUNS / "d_transfer/t0/eegnet.json"),
    ("T2 bi2014a vs T0", RUNS / "d_transfer/t2_bi/eegnet.json", RUNS / "d_transfer/t0/eegnet.json"),
    ("T3 erpcore60 vs T0-new60", RUNS / "d2_t3_erpcore60/scores/n2p3net.json", RUNS / "planb_new60/scores/n2p3net.json"),
    ("T3 bnci60 vs T0-new60", RUNS / "d2_t3_bnci60/scores/n2p3net.json", RUNS / "planb_new60/scores/n2p3net.json"),
    ("T3 erpcore12 vs T0-new12", RUNS / "d2_t3_bench12/scores/n2p3net.json", RUNS / "d2_t0_12/scores/n2p3net.json"),
]
print("== paired hit power at current n and projected 242 ==")
for name, pa, pb in pairs:
    r = mcnemar_and_power(load_hits(pa), load_hits(pb), total=242)
    print(f"{name:28s} {r}")

print("\n== paired per-fold bacc/AUC bootstrap (N2P3Net records) ==")
for name, ra, rb in [
    ("new30 vs old30", RUNS / "diag_planb_new30/record.json", RUNS / "diag_planb_old30/record.json"),
    ("new60 vs old60", RUNS / "planb_new60/record.json", RUNS / "planb_old60/record.json"),
    ("T3-erpcore60 vs new60", RUNS / "d2_t3_erpcore60/record.json", RUNS / "planb_new60/record.json"),
]:
    ha, ba, aa = load_folds(ra)
    hb, bb, ab = load_folds(rb)
    assert len(ha) == len(hb)
    for metric, x, y in [("hit", ha, hb), ("bacc", ba, bb), ("auc", aa, ab)]:
        m, sd, lo, hi = paired_bootstrap_ci(x, y)
        print(f"{name:18s} {metric}: mean={m:+.4f} SE={sd:.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]")
