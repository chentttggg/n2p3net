"""Small-N LOSO diagnostic sweep for N2P3Net failure diagnosis.

Runs several config variants sequentially on the first N GTN subjects and prints
hit/bacc/AUC + timing so larger pilots can be extrapolated.
"""
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
from baselines.n2p3net import N2P3NetBaseline
from data.channel import build_channel_identity
from data.preprocess import STANDARD_CHANNELS
from experiments.run_gtn_baseline import _load_gtn_cache
from models.n2p3net import N2P3Net
from train.preloaded import PreloadedDataLoader
from train.trainer import Trainer, TrainerConfig

CACHE = ROOT / "experiments" / "cache" / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz"
MASK = (True, True, True, False, False, False, False, False)


class DiagAdapter(N2P3NetBaseline):
    """Same as N2P3NetBaseline but applies diagnostic model surgery after construction."""

    def __init__(self, *args, p3b_tau0_init=None, p3b_tau0_hi=None, freeze_tau=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.p3b_tau0_init = p3b_tau0_init
        self.p3b_tau0_hi = p3b_tau0_hi
        self.freeze_tau = freeze_tau

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        model = N2P3Net(**self.model_kwargs)
        if self.p3b_tau0_init is not None:
            with torch.no_grad():
                model.component_window.tau0.data[2] = self.p3b_tau0_init
        if self.p3b_tau0_hi is not None:
            with torch.no_grad():
                model.component_window.tau0_hi[2] = self.p3b_tau0_hi
        if self.freeze_tau:
            for p in model.component_window.parameters():
                p.requires_grad_(False)
        cfg = TrainerConfig(**self.trainer_kwargs)
        trainer = Trainer(model, cfg, E_chn=self.E_chn, channel_mask=self.channel_mask, device=self.device)
        loader = PreloadedDataLoader(
            torch.from_numpy(X), torch.from_numpy(y), batch_size=cfg.batch_size,
            shuffle=True, seed=cfg.seed, device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        history = trainer.fit(loader)
        self.fit_durations.append(time.perf_counter() - t0)
        if self.device.type == "cuda":
            self.fit_peak_memory_mb.append(torch.cuda.max_memory_allocated() / 1e6)
        else:
            self.fit_peak_memory_mb.append(float("nan"))
        self.last_history = history
        self.model_ = model
        self._fitted = True
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
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
    print(f"subjects={args.subjects} trials={len(X8)} folds={len(folds)}")

    base_trainer = dict(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                        lambda2=0.3, lambda3=0.01, lambda_amp=0.01, lambda_jit=0.05,
                        jit_prob=0.5, augment=False, seed=0)
    variants = [
        ("baseline", dict(), {}),
        ("nojit", dict(lambda_jit=0.0), {}),
        ("noamp", dict(lambda_amp=0.0), {}),
        ("noearly", dict(lambda2=0.0), {}),
        ("notau", dict(lambda3=0.0), {}),
        ("jit1", dict(jit_prob=1.0), {}),
        ("augment", dict(augment=True), {}),
        ("tau0450", dict(), dict(p3b_tau0_init=450.0, p3b_tau0_hi=600.0)),
        ("freezetau", dict(), dict(freeze_tau=True)),
        ("softargmax", dict(), dict()),
    ]
    for name, tk, diag in variants:
        tk = {**base_trainer, **tk}
        model_kwargs = dict(dtau_readout="attention_direct", encoder_depth=3, encoder_type="tcn")
        if name == "softargmax":
            model_kwargs["dtau_readout"] = "attention_softargmax"
        adapter = DiagAdapter(model_kwargs=model_kwargs, trainer_kwargs=tk, E_chn=E_chn,
                              channel_mask=mask, **diag)
        t0 = time.perf_counter()
        summary = evaluate(adapter, X8, y, digits, subject_ids, true_digits, folds, n_jobs=1)
        wall = time.perf_counter() - t0
        print(f"[{name}] hit={summary.hit_rate_mean:.4f} bacc={summary.balanced_acc_mean:.4f} "
              f"auc={summary.auc_mean:.4f} wall={wall:.1f}s fold_mean={sum(adapter.fit_durations)/len(adapter.fit_durations):.2f}s")


if __name__ == "__main__":
    main()
