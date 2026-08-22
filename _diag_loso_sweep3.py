"""Third small-N LOSO sweep: replace PCW with mean pooling (architecture ablation)."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

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


class MeanReadout(nn.Module):
    """PCW replacement: mean-pool over time, broadcast to 3 component slots."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.tau0 = nn.Parameter(torch.tensor([220.0, 300.0, 350.0]))
        self.register_buffer("tau0_lo", torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer("tau0_hi", torch.tensor([1000.0, 1000.0, 1000.0]))

    @property
    def tau0_bounded(self):
        return self.tau0

    def clamp_tau0_(self):
        return self

    def forward(self, Z, return_attention=False):
        B, T, D = Z.shape
        H = Z.mean(dim=1).unsqueeze(1).expand(B, 3, D)
        tau = self.tau0.to(Z.dtype).unsqueeze(0).expand(B, 3)
        sigma = torch.full((3, 2), 50.0, dtype=Z.dtype, device=Z.device)
        if return_attention:
            A = torch.full((B, 3, T), 1.0 / T, dtype=Z.dtype, device=Z.device)
            return H, tau, sigma, A
        return H, tau, sigma


class MeanAdapter(N2P3NetBaseline):
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        model = N2P3Net(**self.model_kwargs)
        model.component_window = MeanReadout()
        cfg = TrainerConfig(**self.trainer_kwargs)
        trainer = Trainer(model, cfg, E_chn=self.E_chn, channel_mask=self.channel_mask, device=self.device)
        loader = PreloadedDataLoader(torch.from_numpy(X), torch.from_numpy(y),
                                     batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed,
                                     device=self.device)
        t0 = time.perf_counter()
        history = trainer.fit(loader)
        self.fit_durations.append(time.perf_counter() - t0)
        self.last_history = history
        self.model_ = model
        self._fitted = True
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
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
                        lambda3=0.0, lambda_amp=0.0, lambda_jit=0.0, jit_prob=0.0,
                        augment=False, seed=0)
    variants = [
        ("meanpool_heads", dict(encoder_depth=3, encoder_type="tcn")),
        ("meanpool_enc0", dict(encoder_depth=0, encoder_type="tcn")),
    ]
    for name, model_kwargs in variants:
        adapter = MeanAdapter(model_kwargs=model_kwargs, trainer_kwargs=dict(base_trainer),
                              E_chn=E_chn, channel_mask=mask)
        t0 = time.perf_counter()
        summary = evaluate(adapter, X8, y, digits, subject_ids, true_digits, folds, n_jobs=1)
        wall = time.perf_counter() - t0
        print(f"[{name}] hit={summary.hit_rate_mean:.4f} bacc={summary.balanced_acc_mean:.4f} "
              f"auc={summary.auc_mean:.4f} wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
