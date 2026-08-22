"""Second small-N LOSO sweep: tau0 priors and dtau readout variants."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from baselines.evaluate import evaluate, loso_folds
from data.channel import build_channel_identity
from data.preprocess import STANDARD_CHANNELS
from experiments.run_gtn_baseline import _load_gtn_cache
from _diag_loso_sweep import CACHE, MASK, DiagAdapter

CACHE = Path(CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    args = ap.parse_args()

    X3, y, digits, subject_ids, true_digits, _ = _load_gtn_cache(CACHE)
    keep_subj = np.unique(subject_ids)[: args.subjects]
    keep = np.isin(subject_ids, keep_subj)
    X3, y, digits, subject_ids = X3[keep], y[keep], digits[keep], subject_ids[keep]
    true_digits = {k: v for k, v in true_digits.items() if k in set(keep_subj.tolist())}
    X8 = np.zeros((X3.shape[0], 8, X3.shape[2]), dtype=np.float32)
    X8[:, :3, :] = X3
    mask = torch.tensor(MASK, dtype=torch.bool)
    identity = build_channel_identity(ch_names=list(STANDARD_CHANNELS), channel_mask=MASK)
    E_chn = torch.from_numpy(identity.embedding)
    folds = loso_folds(subject_ids)

    base_trainer = dict(epochs=args.epochs, batch_size=512, lr=1e-3, lambda2=0.3,
                        lambda3=0.01, lambda_amp=0.01, lambda_jit=0.05, jit_prob=0.5,
                        augment=False, seed=0)
    variants = [
        ("tau0450soft", "attention_softargmax", 450.0),
        ("tau0460soft", "attention_softargmax", 460.0),
        ("tau0480", "attention_direct", 480.0),
        ("tau0500", "attention_direct", 500.0),
        ("tau0450direct", "attention_direct", 450.0),
        ("attn_mlp", "attention", None),
        ("maxmean", "maxmean", None),
        ("global_pool", "global_pool", None),
    ]
    if args.only:
        allowed = {x.strip() for x in args.only.split(",") if x.strip()}
        variants = [v for v in variants if v[0] in allowed]
    for name, readout, tau0_init in variants:
        trainer_kwargs = dict(base_trainer)
        model_kwargs = dict(dtau_readout=readout, encoder_depth=3, encoder_type="tcn")
        adapter = DiagAdapter(model_kwargs=model_kwargs, trainer_kwargs=trainer_kwargs,
                              E_chn=E_chn, channel_mask=mask,
                              p3b_tau0_init=tau0_init, p3b_tau0_hi=600.0 if tau0_init else None)
        t0 = time.perf_counter()
        summary = evaluate(adapter, X8, y, digits, subject_ids, true_digits, folds, n_jobs=1)
        wall = time.perf_counter() - t0
        print(f"[{name}] hit={summary.hit_rate_mean:.4f} bacc={summary.balanced_acc_mean:.4f} "
              f"auc={summary.auc_mean:.4f} wall={wall:.1f}s "
              f"fold_mean={sum(adapter.fit_durations)/len(adapter.fit_durations):.2f}s", flush=True)


if __name__ == "__main__":
    main()
