"""Gradient health diagnostic: are PCW tau parameters receiving supervision?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from data.channel import build_channel_identity
from data.preprocess import STANDARD_CHANNELS
from models.n2p3net import N2P3Net
from train.preloaded import PreloadedDataLoader
from train.trainer import Trainer, TrainerConfig

CACHE = ROOT / "experiments" / "cache" / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz"
MASK = (True, True, True, False, False, False, False, False)


def main():
    z = np.load(CACHE, allow_pickle=True)
    X3 = z["X"][:10000].astype(np.float32)
    y = z["y"][:10000].astype(np.int64)
    X8 = np.zeros((len(X3), 8, X3.shape[2]), dtype=np.float32)
    X8[:, :3, :] = X3
    mask = torch.tensor(MASK, dtype=torch.bool)
    identity = build_channel_identity(ch_names=list(STANDARD_CHANNELS), channel_mask=MASK)
    E_chn = torch.from_numpy(identity.embedding)
    model = N2P3Net()
    cfg = TrainerConfig(epochs=10, batch_size=512, accum_steps=2, lr=1e-3, lambda2=0.3, lambda3=0.01,
                        lambda_amp=0.01, lambda_jit=0.05, jit_prob=0.5, augment=False, seed=0)
    trainer = Trainer(model, cfg, E_chn=E_chn, channel_mask=mask, device=torch.device("cuda"))
    loader = PreloadedDataLoader(torch.from_numpy(X8), torch.from_numpy(y), batch_size=512,
                                 shuffle=True, seed=0, device=trainer.device)

    def grads():
        out = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                out[n] = float(p.grad.detach().float().norm())
        return out

    for epoch in range(5):
        model.train()
        gsum = {}
        for step, batch in enumerate(loader):
            trainer._train_step(batch[0], batch[1], step)
            g = grads()
            for k, v in g.items():
                gsum[k] = gsum.get(k, 0.0) + v
            if step == 0:
                first = g
            break  # only first batch per epoch for speed
        t0 = model.component_window.tau0.detach().cpu().numpy()
        print(f"epoch {epoch+1}: tau0={t0.round(3)}")
        for k in ["component_window.tau0", "component_window.dtau_attn_query",
                  "component_window.dtau_attn_temp", "component_window.dtau_gain",
                  "component_window.sigma_raw", "heads.head_a.0.weight",
                  "tokenizer.pointwise.weight", "encoder.blocks.0.pointwise.0.weight"]:
            print(f"  {k:44s} grad={first.get(k, float('nan')):.3e}")
    # final averaged
    print("avg grad over epochs:", {k: round(v, 5) for k, v in gsum.items() if "tau" in k or "dtau" in k or "sigma" in k})


if __name__ == "__main__":
    main()
