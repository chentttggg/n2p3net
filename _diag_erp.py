"""GTN ERP-level diagnostics: grand-average P3, per-subject latency, single-trial signal."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

CACHE = ROOT / "experiments" / "cache" / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz"
SFREQ = 256.0
T0 = int(round(0.2 * SFREQ))  # sample index at 0 ms
BASE_N = int(round(0.2 * SFREQ))

z = np.load(CACHE, allow_pickle=True)
X = z["X"].astype(np.float32)
y = z["y"].astype(np.int64)
subjects = z["subject_ids"].astype(str)
uniq = np.unique(subjects)
n_subs = len(uniq)
print("N", X.shape, "subjects", n_subs, "targets", y.sum(), "target_rate", y.mean())

# per-subject baseline correction (same as N2P3Net model front-end)
Xb = np.empty_like(X)
for s in uniq:
    m = subjects == s
    b = X[m, :, :BASE_N].mean(axis=2, keepdims=True)
    Xb[m] = X[m] - b

# grand averages (each subject equal weight)
subj_ga_t = np.zeros((n_subs, 3, X.shape[2]))
subj_ga_n = np.zeros((n_subs, 3, X.shape[2]))
for i, s in enumerate(uniq):
    m = subjects == s
    subj_ga_t[i] = Xb[m & (y == 1)].mean(axis=0)
    subj_ga_n[i] = Xb[m & (y == 0)].mean(axis=0)
ga_t = subj_ga_t.mean(axis=0)
ga_n = subj_ga_n.mean(axis=0)
t_ms = -200 + np.arange(X.shape[2]) * 1000.0 / SFREQ
peak_region = (t_ms >= 200) & (t_ms <= 600)
for c, name in enumerate(["Fz", "Cz", "Pz"]):
    diff = ga_t[c] - ga_n[c]
    j = np.argmax(np.abs(diff[peak_region]))
    pk = t_ms[peak_region][j]
    amp = diff[peak_region][j]
    jmax = np.argmax(diff[peak_region])
    print(f"{name}: target peak {t_ms[np.argmax(ga_t[c][peak_region]) + np.where(peak_region)[0][0]]:.1f}ms "
          f"target-nontarget maxdiff {pk:.1f}ms {amp*1e6:.3f}uV")
    # mean amplitude windows
    for lo, hi in [(250, 450), (300, 500), (350, 550), (450, 650)]:
        sel = (t_ms >= lo) & (t_ms <= hi)
        print(f"  {lo}-{hi}ms diff mean {diff[sel].mean()*1e6:.4f}uV")

# per-subject Pz peak latency (target) and single-trial window AUC
lats = np.full(n_subs, np.nan)
amps = np.full(n_subs, np.nan)
aucs = np.full(n_subs, np.nan)
for i, s in enumerate(uniq):
    m = subjects == s
    if y[m].sum() == 0 or (1 - y[m]).sum() == 0:
        continue
    diff = subj_ga_t[i, 2] - subj_ga_n[i, 2]
    j = np.argmax(diff[peak_region])
    lats[i] = t_ms[peak_region][j]
    amps[i] = diff[peak_region][j]
    win = (t_ms >= 250) & (t_ms <= 500)
    feat = Xb[m, 2, :][:, win].mean(axis=1)
    if len(np.unique(y[m])) > 1:
        aucs[i] = roc_auc_score(y[m], feat)
print("\nper-subject Pz target peak latency: mean=%.1f sd=%.1f q10/50/90=%s" % (
    np.nanmean(lats), np.nanstd(lats), np.nanpercentile(lats, [10, 50, 90])))
print("per-subject Pz 250-500ms mean-amp single-trial AUC: mean=%.3f sd=%.3f median=%.3f" % (
    np.nanmean(aucs), np.nanstd(aucs), np.nanmedian(aucs)))
print("latency vs AUC spearman", spearmanr(lats, aucs, nan_policy="omit"))
print("amp vs AUC spearman", spearmanr(amps, aucs, nan_policy="omit"))

# join with N2P3Net per-fold bacc and baseline EEGNet hit
rec = json.load(open(ROOT / "experiments" / "runs" / "n2p3net_gtn_20260822_011817Z" / "record.json", encoding="utf-8"))
folds = rec["results"]["per_fold"]
bacc = np.array([f["balanced_acc"] for f in folds])
auc_model = np.array([f["auc"] for f in folds])
print("\nper-subject ERP signal vs N2P3Net metrics:")
for name, arr in [("peak_latency", lats), ("peak_amp", amps), ("window_AUC", aucs)]:
    for name2, arr2 in [("bacc", bacc), ("auc", auc_model)]:
        r = spearmanr(arr, arr2, nan_policy="omit")
        print(f"  {name} vs {name2}: r={r.statistic:.3f} p={r.pvalue:.3f}")
