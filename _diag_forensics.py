"""N2P3Net failure forensics (CPU-only, reads saved results + GTN cache)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results"
RUN10 = ROOT / "experiments" / "runs" / "n2p3net_gtn_20260822_011817Z"
RUN30 = ROOT / "experiments" / "runs" / "n2p3net_gtn_20260822_061620Z"

MODELS = ["inception", "conformer", "eegnet", "swlda", "template", "xdawn", "windowlr"]


def load(path):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    return p


def load_hits(path):
    p = load(path)
    return {str(r["subject"]): int(r["hit"]) for r in p["records"]}


def load_meta(path):
    p = load(path)
    return {str(r["subject"]): (int(r["true"]), int(r["predicted"])) for r in p["records"]}


n10 = RUN10 / "scores" / "n2p3net.json"
n30 = RUN30 / "scores" / "n2p3net.json"
h10 = load_hits(n10)
h30 = load_hits(n30)
p10 = load(n10)
p30 = load(n30)

hits = {"n2p3net10": h10}
for m in MODELS:
    hits[m] = load_hits(RESULTS / f"{m}.json")

subjects = sorted(h10)
print("== summary ==")
for name, h in hits.items():
    vals = [h[s] for s in subjects if s in h]
    print(f"{name:12s} n={len(vals):3d} hit={np.mean(vals):.4f}")

print("\n== first-60 subcohort ==")
sub60 = sorted(subjects)[:60]
for name, h in hits.items():
    vals = [h[s] for s in sub60]
    print(f"{name:12s} n={len(vals):3d} hit={np.mean(vals):.4f}")

print("\n== per-model bacc/AUC means (from score JSON) ==")
for path in [RESULTS / f"{m}.json" for m in MODELS] + [n10, n30]:
    p = load(path)
    print(f"{p['model']:12s} hit={p['hit_rate_mean']:.4f} bacc={p['balanced_acc_mean']:.4f} auc={p['auc_mean']:.4f}")

print("\n== failure cross-tab (242 subjects) ==")
print("subjects where N2P3Net10 misses:")
for m in MODELS:
    hm = hits[m]
    n10_miss = {s for s in subjects if h10[s] == 0}
    m_miss = {s for s in subjects if s in hm and hm[s] == 0}
    both = n10_miss & m_miss
    only_n10 = n10_miss - m_miss
    only_m = m_miss - n10_miss
    print(f"{m:12s} both_miss={len(both):3d} only_n2p3_miss={len(only_n10):3d} only_{m}_miss={len(only_m):3d}")

print("\nall 8 models miss (hard subjects):")
all_miss = set(subjects)
for h in hits.values():
    all_miss &= {s for s in subjects if h[s] == 0}
print(len(all_miss), sorted(all_miss)[:20])
print("N2P3Net10 misses:", sum(1 for s in subjects if h10[s] == 0))

print("\n== digit prediction statistics ==")
for path in [n10, n30] + [RESULTS / f"{m}.json" for m in MODELS]:
    p = load(path)
    preds = Counter(r["predicted"] for r in p["records"])
    trues = Counter(r["true"] for r in p["records"])
    bias = max(preds.values()) / len(p["records"])
    print(f"{p['model']:12s} pred_dist={[preds[i] for i in range(1,10)]} top_pred_bias={bias:.3f} true_min={min(trues.values())}")

print("\n== confusion: N2P3Net10 per true digit accuracy ==")
acc = defaultdict(lambda: [0, 0])
for s in subjects:
    t, pr = load_meta(n10)[s]
    acc[t][0] += h10[s]
    acc[t][1] += 1
for d in range(1, 10):
    a, n = acc[d]
    print(f"true={d}: hit {a}/{n} = {a/n:.3f}")

print("\n== N2P3Net10 errors: predicted digit distribution when wrong ==")
wrong_preds = Counter()
for s in subjects:
    if h10[s] == 0:
        t, pr = load_meta(n10)[s]
        wrong_preds[pr] += 1
print([wrong_preds[i] for i in range(1, 10)])

print("\n== tie to trial counts / target counts (from cache) ==")
cache = np.load(ROOT / "experiments" / "cache" / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz", allow_pickle=True)
y = cache["y"]
subject_ids = cache["subject_ids"].astype(str)
n_trials = Counter(subject_ids)
n_target = Counter()
for s_, y_ in zip(subject_ids, y):
    if y_ == 1:
        n_target[s_] += 1
rows = []
for s in subjects:
    rows.append((h10[s], n_trials.get(s, 0), n_target.get(s, 0)))
rows = np.array(rows)
from scipy.stats import pointbiserialr, spearmanr  # noqa: E402
r1 = pointbiserialr(rows[:, 0], rows[:, 1])
r2 = pointbiserialr(rows[:, 0], rows[:, 2])
r3 = pointbiserialr(rows[:, 0], rows[:, 2] / np.maximum(rows[:, 1], 1))
print("hit vs n_trials: r=%.3f p=%.3f" % (r1.correlation, r1.pvalue))
print("hit vs n_target: r=%.3f p=%.3f" % (r2.correlation, r2.pvalue))
print("hit vs target_rate: r=%.3f p=%.3f" % (r3.correlation, r3.pvalue))
miss_rows = rows[rows[:, 0] == 0]
hit_rows = rows[rows[:, 0] == 1]
print(f"miss subjects: n_trials mean={miss_rows[:,1].mean():.1f} target mean={miss_rows[:,2].mean():.1f}")
print(f"hit subjects : n_trials mean={hit_rows[:,1].mean():.1f} target mean={hit_rows[:,2].mean():.1f}")

print("\n== per-fold metrics vs trial/target counts ==")
r10 = load(RUN10 / "record.json")
folds = r10["results"]["per_fold"]
assert len(folds) == len(subjects) == 242
hits_f = np.array([f["hit_rate"] for f in folds])
bacc = np.array([f["balanced_acc"] for f in folds])
auc = np.array([f["auc"] for f in folds])
nt = np.array([f["n_test_trials"] for f in folds])
ntgt = np.array([n_target[s] for s in subjects])
for name_, arr in [("hit", hits_f), ("bacc", bacc), ("auc", auc)]:
    print(f"{name_} vs n_trials spearman {spearmanr(nt, arr).statistic:.3f} p={spearmanr(nt, arr).pvalue:.3f}; vs n_target {spearmanr(ntgt, arr).statistic:.3f} p={spearmanr(ntgt, arr).pvalue:.3f}")
print("bacc quartiles by n_trials:")
q = np.quantile(nt, [0, .25, .5, .75, 1])
for lo, hi in zip(q[:-1], q[1:]):
    sel = (nt >= lo) & (nt <= hi)
    print(f"  {lo:.0f}-{hi:.0f}: n={sel.sum()} hit={hits_f[sel].mean():.3f} bacc={bacc[sel].mean():.3f} auc={auc[sel].mean():.3f}")

print("\n== cross-model per-subject hit correlation with N2P3Net10 ==")
for m in MODELS:
    hm = hits[m]
    xs = [h10[s] for s in subjects]
    ys = [hm[s] for s in subjects]
    print(f"{m:12s} agreement={np.mean([a==b for a,b in zip(xs,ys)]):.3f} corr={np.corrcoef(xs,ys)[0,1]:.3f}")
