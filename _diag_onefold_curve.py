"""One-fold instrumentation: loss components + per-epoch held-out metrics (60-subject subset)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data.channel import build_channel_identity
from data.preprocess import STANDARD_CHANNELS
from models.decision import decide
from models.n2p3net import N2P3Net
from train.losses import compute_losses
from train.preloaded import PreloadedDataLoader
from train.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "cache" / "gtn_3ch_sf256_lf0.1_tm-0.2_tx0.8_nall.npz"
GTN_MASK = (True, True, True, False, False, False, False, False)
BATCH = 512
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lambda2", type=float, default=0.3)
    ap.add_argument("--lambda3", type=float, default=0.01)
    ap.add_argument("--lambda-amp", type=float, default=0.01)
    ap.add_argument("--lambda-jit", type=float, default=0.05)
    ap.add_argument("--jit-prob", type=float, default=0.5)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--p3b-tau0-init", type=float, default=None)
    ap.add_argument("--p3b-tau0-hi", type=float, default=None)
    ap.add_argument("--freeze-tau", action="store_true")
    args = ap.parse_args()
    z = np.load(CACHE, allow_pickle=True)
    X3 = z["X"]
    y = z["y"].astype(np.int64)
    digits = z["digits"].astype(np.int64)
    subject_ids = z["subject_ids"].astype(str)
    true_keys = z["true_keys"].astype(str)
    true_values = z["true_values"].astype(np.int64)
    subjects = np.unique(subject_ids)
    keep_subjects = subjects[:60]
    keep = np.isin(subject_ids, keep_subjects)
    X3, y, digits, subject_ids = X3[keep], y[keep], digits[keep], subject_ids[keep]
    keep_true = np.isin(true_keys, keep_subjects)
    true_keys, true_values = true_keys[keep_true], true_values[keep_true]
    subjects = np.unique(subject_ids)
    print("data", X3.shape, len(subjects), "target_rate", y.mean())

    X8 = np.zeros((X3.shape[0], 8, X3.shape[2]), dtype=np.float32)
    X8[:, :3, :] = X3
    mask = torch.tensor(GTN_MASK, dtype=torch.bool)
    identity = build_channel_identity(ch_names=list(STANDARD_CHANNELS), channel_mask=GTN_MASK)
    E_chn = torch.from_numpy(identity.embedding)

    test_subj = subjects[0]
    test_mask = subject_ids == test_subj
    train_mask = ~test_mask
    Xtr, ytr, Xte, yte, dte = X8[train_mask], y[train_mask], X8[test_mask], y[test_mask], digits[test_mask]
    print("train", Xtr.shape, "test", Xte.shape, "test targets", yte.sum())

    model = N2P3Net()
    if args.p3b_tau0_init is not None:
        with torch.no_grad():
            model.component_window.tau0.data[2] = args.p3b_tau0_init
    if args.p3b_tau0_hi is not None:
        with torch.no_grad():
            model.component_window.tau0_hi[2] = args.p3b_tau0_hi
    if args.freeze_tau:
        for p in model.component_window.parameters():
            p.requires_grad_(False)
    print("params", model.num_parameters())
    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=BATCH,
        lr=1e-3,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda_amp=args.lambda_amp,
        lambda_jit=args.lambda_jit,
        jit_prob=args.jit_prob,
        jit_max_ms=40.0,
        augment=args.augment,
        seed=SEED,
    )
    print("CONFIG", args)
    trainer = Trainer(model, cfg, E_chn=E_chn, channel_mask=mask, device=torch.device("cuda"))
    loader = PreloadedDataLoader(
        torch.from_numpy(Xtr), torch.from_numpy(ytr), batch_size=BATCH, shuffle=True, seed=SEED, device=trainer.device
    )
    print(f"{'ep':>3} {'train_total':>11} {'L_tgt':>7} {'L_early':>7} {'L_tau':>7} {'L_amp':>7} "
          f"{'val_bacc':>9} {'val_auc':>7} {'hit_sum':>7} {'hit_mean':>8} "
          f"{'tau0_P3b':>8} {'tauT_P3b':>8}")

    @torch.inference_mode()
    def evaluate_epoch(epoch_label):
        model_eval = trainer.model
        model_eval.eval()
        # component losses on train set (no jitter)
        comp = {"target": 0.0, "early": 0.0, "tau": 0.0, "amp": 0.0, "total": 0.0}
        n = 0
        for Xb, yb in loader:
            with trainer._autocast_ctx():
                out = model_eval(Xb, trainer.E_chn, None, channel_mask=trainer.channel_mask, return_attention=True)
                losses = compute_losses(
                    out, model_eval.component_window.tau0_bounded, yb,
                    lambda2=cfg.lambda2, lambda3=cfg.lambda3, lambda_amp=cfg.lambda_amp,
                    lambda_jit=0.0, pos_weight=cfg.pos_weight, tau_scale_ms=cfg.tau_scale_ms,
                    X=Xb, pz_channel=2 if Xb.shape[1] == 3 else 3,
                )
            b = Xb.shape[0]
            comp["target"] += float(losses.target.detach().float()) * b
            comp["early"] += float(losses.early.detach().float()) * b
            comp["tau"] += float(losses.tau.detach().float()) * b
            comp["amp"] += float(losses.amp.detach().float()) * b
            comp["total"] += float(losses.total.detach().float()) * b
            n += b
        for k in comp:
            comp[k] /= max(n, 1)
        # test predictions
        logits, tau, sigma = [], [], []
        for i in range(0, Xte.shape[0], 256):
            xb = torch.from_numpy(Xte[i:i+256]).to(trainer.device)
            out = model_eval(xb, trainer.E_chn, None, channel_mask=trainer.channel_mask)
            logits.append(out.heads.logit_target.float().cpu())
            tau.append(out.tau.float().cpu())
            sigma.append(out.sigma.float().cpu())
        lg = torch.cat(logits).squeeze(-1).numpy()
        tau_all = torch.cat(tau).numpy()
        bacc = balanced_accuracy_score(yte, (lg - lg.mean()) > 0)
        auc = roc_auc_score(yte, lg) if len(np.unique(yte)) > 1 else float("nan")
        res_sum = decide(lg, dte, np.array([test_subj] * len(dte)), aggregation="sum")
        res_mean = decide(lg, dte, np.array([test_subj] * len(dte)), aggregation="mean")
        true_d = int(true_values[np.where(true_keys == test_subj)[0][0]])
        hit_sum = int(res_sum.predicted[0] == true_d) if res_sum.predicted[0] is not None else 0
        hit_mean = int(res_mean.predicted[0] == true_d) if res_mean.predicted[0] is not None else 0
        model_eval.train()
        return comp, bacc, auc, hit_sum, hit_mean, tau_all

    for epoch in range(args.epochs):
        trainer.model.train()
        total = 0.0
        nb = 0
        for step, batch in enumerate(loader):
            loss_tensor = trainer._train_step(batch[0], batch[1], step)
            total += float(loss_tensor)
            nb += 1
        train_total = total / max(nb, 1)
        comp, bacc, auc, h_sum, h_mean, tau_all = evaluate_epoch(epoch + 1)
        tau0 = trainer.model.component_window.tau0_bounded.detach().cpu().numpy()
        tau_t = tau_all[yte == 1, 2].mean() if (yte == 1).any() else float("nan")
        print(f"{epoch+1:3d} {train_total:11.4f} {comp['target']:7.4f} {comp['early']:7.4f} "
              f"{comp['tau']:7.4f} {comp['amp']:7.4f} {bacc:9.4f} {auc:7.4f} {h_sum:7d} {h_mean:8d} "
              f"{tau0[2]:8.2f} {tau_t:8.2f}")


if __name__ == "__main__":
    main()
