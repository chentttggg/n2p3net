"""Empirical information-theoretic bound for BNCI P300 candidate identification.

Turns the abstract Q-ary bound in ``transfer.information_bound`` into a concrete
per-participant number: given the calibration ERP epochs actually recorded, what
is the best hit rate an optimal Bayesian decoder could reach, and how much of
that is the current pipeline capturing?

This is a diagnostic on previously observed public calibration data. It reads no
test truth, fits no model, and promotes nothing. It answers one question: is the
empirical ~0.77 near the information ceiling for this signal, or is there a large
unexploited gap that a better decoder could close?

Scope note: d^2 is estimated from ALL six calibration selections pooled (the
ceiling question, not a held-out performance claim). Held-out prediction is the
job of a model comparison, not of this bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_e2_prior_scale import _selection_views  # noqa: E402

from research.execution import atomic_write_json  # noqa: E402
from transfer.information_bound import (  # noqa: E402
    mahalanobis_snr,
    spectral_discriminability,
    theoretical_hit_rate_bound,
)
from transfer.raw_continuous import CANDIDATES, SFREQ  # noqa: E402

N_FLASHES_PER_SELECTION = 120  # one complete BNCI C800 selection


def _extract_epochs(record, start, stop):
    """Per-event windows [onset+start, onset+stop) with their binary labels."""
    x = np.asarray(record.samples, dtype=float)
    epochs, labels = [], []
    for onset, label in zip(record.onsets, record.target_mask, strict=True):
        lo, hi = int(onset) + start, int(onset) + stop
        if lo < 0 or hi > len(x):
            continue
        epochs.append(x[lo:hi])
        labels.append(bool(label))
    if not epochs:
        raise ValueError("no complete epochs inside the requested window")
    return np.stack(epochs), np.asarray(labels, dtype=bool)


def analyse_record(row, plan, start, stop, *, n_components=8, empirical_hit=None):
    """Compute the Mahalanobis SNR, the Q-ary bound and the spectral split."""
    groups, ledger = _selection_views(row, plan)
    target_parts, nontarget_parts = [], []
    per_group = []
    for record, _member, _truth in groups:
        epochs, labels = _extract_epochs(record, start, stop)
        target_parts.append(epochs[labels])
        nontarget_parts.append(epochs[~labels])
        tgt, non = epochs[labels], epochs[~labels]
        group_snr = mahalanobis_snr(tgt, non)
        per_group.append({
            "record_id": record.record_id,
            "n_target": int(labels.sum()),
            "n_nontarget": int((~labels).sum()),
            "d2_mahalanobis": group_snr["d2_mahalanobis"],
        })
    target = np.concatenate(target_parts)
    nontarget = np.concatenate(nontarget_parts)

    snr = mahalanobis_snr(target, nontarget)
    bound = theoretical_hit_rate_bound(
        snr["d2_mahalanobis"], len(CANDIDATES), N_FLASHES_PER_SELECTION
    )
    spectral = spectral_discriminability(
        target, nontarget, SFREQ, n_components=n_components
    )

    result = {
        "subject": row["subject"],
        "raw_sha256": row["sha256"],
        "window_samples": [start, stop],
        "n_target_epochs": int(len(target)),
        "n_nontarget_epochs": int(len(nontarget)),
        "snr": snr,
        "bound": {k: v for k, v in bound.items() if k != "required_d2_for_targets"},
        "required_d2_for_targets": bound["required_d2_for_targets"],
        "spectral_components": spectral["components"],
        "per_group": per_group,
    }
    if empirical_hit is not None:
        chance = 1.0 / len(CANDIDATES)
        achievable = float(bound["p_hit_monte_carlo"])
        result["efficiency"] = {
            "empirical_hit": float(empirical_hit),
            "achievable_hit_mc": achievable,
            "chance_hit": chance,
            "absolute_gap": achievable - float(empirical_hit),
            "fraction_of_room_used": max(
                0.0,
                (float(empirical_hit) - chance)
                / max(achievable - chance, np.finfo(float).tiny),
            ),
        }
    return result, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="BNCI plan JSON carrying the eight MAT records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-ms", default="0,1200",
                        help="response window in ms relative to each flash onset")
    parser.add_argument("--spectral-components", type=int, default=8)
    parser.add_argument("--empirical-hit", type=float, default=None,
                        help="optional observed hit rate to compute an efficiency gap")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if len(plan.get("records", [])) < 1:
        raise ValueError("plan must carry at least one MAT record")
    start_ms, stop_ms = (float(v) for v in args.window_ms.split(","))
    start = round(start_ms * SFREQ / 1000.0)
    stop = round(stop_ms * SFREQ / 1000.0)
    if stop <= start:
        raise ValueError("window stop must exceed start")

    args.output.mkdir(parents=True, exist_ok=False)
    per_subject, failures = [], []
    for row in plan["records"]:
        try:
            result, ledger = analyse_record(
                row, plan, start, stop,
                n_components=args.spectral_components,
                empirical_hit=args.empirical_hit,
            )
            per_subject.append(result)
            subject_dir = args.output / row["subject"]
            subject_dir.mkdir()
            atomic_write_json(subject_dir / "bound.json", result)
            print(json.dumps({
                "subject": row["subject"],
                "d2": round(result["snr"]["d2_mahalanobis"], 4),
                "achievable_hit_mc": round(result["bound"]["p_hit_monte_carlo"], 4),
                "union_lower": round(result["bound"]["p_hit_lower_bound"], 4),
            }), flush=True)
        except Exception as exc:  # noqa: BLE001 - record failures, never drop them
            failures.append({"subject": row["subject"], "error": repr(exc)})
            print(json.dumps({"subject": row["subject"], "failed": repr(exc)}), flush=True)

    summary = {
        "schema": "n2p3_information_bound_driver/1",
        "evidence_scope": "public_calibration_development_diagnostic",
        "plan_sha256": None,
        "window_ms": [start_ms, stop_ms],
        "n_flashes_per_selection": N_FLASHES_PER_SELECTION,
        "n_candidates": len(CANDIDATES),
        "empirical_hit_input": args.empirical_hit,
        "participants": per_subject,
        "failures": failures,
    }
    if per_subject:
        d2 = np.array([p["snr"]["d2_mahalanobis"] for p in per_subject])
        achievable = np.array([p["bound"]["p_hit_monte_carlo"] for p in per_subject])
        summary["participant_macro"] = {
            "mean_d2": float(d2.mean()),
            "median_d2": float(np.median(d2)),
            "min_d2": float(d2.min()),
            "max_d2": float(d2.max()),
            "mean_achievable_hit_mc": float(achievable.mean()),
            "min_achievable_hit_mc": float(achievable.min()),
        }
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": "complete", "n_participants": len(per_subject),
                      "n_failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
