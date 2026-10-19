"""Reproduce the 2026-09-29 exploratory audit from saved participant summaries.

Run with the repository virtual environment. This reads no model weights, launches
no training, and does not regenerate predictions. Default output is beside this
script; --check compares recomputed numerical sections without writing a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/runs/unified12_20260918/results"
ITERATIONS = 20_000
ANALYSIS_SEED = 20260929
ARMS = ["A0Q0", "A0Q1", "A1Q0", "A1Q1"]
WEIGHTS = {
    "A1Q1_minus_A0Q0": [-1, 0, 0, 1],
    "A1_minus_A0_at_Q0": [-1, 0, 1, 0],
    "A1_minus_A0_at_Q1": [0, -1, 0, 1],
    "Q1_minus_Q0_at_A0": [-1, 1, 0, 0],
    "Q1_minus_Q0_at_A1": [0, 0, -1, 1],
    "A_by_Q_interaction": [1, -1, -1, 1],
    "A_marginal_effect": [-0.5, -0.5, 0.5, 0.5],
    "Q_marginal_effect": [-0.5, 0.5, -0.5, 0.5],
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_auc(group, budget):
    rows = {}
    for arm in ARMS:
        for item in group["arm_descriptives"][arm][str(budget)]["participant_rows"]:
            assert item["status"] == "complete_pair"
            rows.setdefault((item["study"], item["participant_key"]), {})[arm] = item[
                "difference"
            ]
    assert all(set(item) == set(ARMS) for item in rows.values())
    return {key: np.array([item[arm] for arm in ARMS]) for key, item in rows.items()}


def summarize(rows):
    """Keep participant pairing and give each fixed study equal weight."""
    studies = sorted({key[0] for key in rows})
    rng = np.random.default_rng(ANALYSIS_SEED)
    samples = {
        study: np.array([value for (name, _), value in rows.items() if name == study])
        for study in studies
    }
    replicates = {
        study: values[rng.integers(0, len(values), size=(ITERATIONS, len(values)))].mean(
            axis=1
        )
        for study, values in samples.items()
    }
    means = np.mean([values.mean(axis=0) for values in samples.values()], axis=0)
    bootstrap = np.mean(list(replicates.values()), axis=0)
    result = {
        "participants": len(rows),
        "counts": {study: len(values) for study, values in samples.items()},
        "arm_means": dict(zip(ARMS, map(float, means), strict=True)),
        "contrasts": {},
    }
    for contrast, weights in WEIGHTS.items():
        values = {study: array @ weights for study, array in samples.items()}
        result["contrasts"][contrast] = {
            "mean": float(means @ weights),
            "ci95": list(map(float, np.quantile(bootstrap @ weights, [0.025, 0.975]))),
            "study_means": {study: float(array.mean()) for study, array in values.items()},
            "negative_participant_fraction_study_equal": float(
                np.mean([np.mean(array < 0) for array in values.values()])
            ),
        }
    return result


def summarize_hit(saved):
    result = {}
    for budget in [200, 800]:
        items = [row for row in saved["rows"] if row["K"] == budget]
        by_arm = {
            arm: {row["participant_key"]: row for row in items if row["arm"] == arm}
            for arm in sorted({row["arm"] for row in items})
        }
        keys = sorted(by_arm["A0Q0"])
        rng = np.random.default_rng(ANALYSIS_SEED)
        indices = rng.integers(0, len(keys), size=(ITERATIONS, len(keys)))
        record = {"participants": len(keys), "arms": {}, "paired_differences": {}}
        for arm, rows in by_arm.items():
            assert set(rows) == set(keys)
            values = np.array([rows[key]["hit_operational"] for key in keys])
            record["arms"][arm] = {
                "mean": float(values.mean()),
                "ci95": list(
                    map(float, np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))
                ),
                **{
                    f"{field}_sum": sum(rows[key][field] for key in keys)
                    for field in ["requested", "valid", "correct", "complete"]
                },
                "coverage": float(np.mean([rows[key]["coverage"] for key in keys])),
            }
        hit = np.array(
            [[by_arm[arm][key]["hit_operational"] for arm in ARMS] for key in keys]
        )
        for contrast, weights in WEIGHTS.items():
            differences = hit @ weights
            record["paired_differences"][contrast] = {
                "mean": float(differences.mean()),
                "ci95": list(
                    map(float, np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]))
                ),
            }
        result[str(budget)] = record
    return result


def recompute():
    ordinary = read_json(RESULTS / "ordinary_contrasts.json")["groups"][0]
    family = read_json(RESULTS / "family_contrasts.json")["groups"][0]
    bnci = read_json(RESULTS / "bnci_hit_all.json")
    decisions = read_json(RESULTS / "bnci_decisions.json")
    output = {
        "status": "exploratory_read_only_reanalysis_no_training",
        "input_artifacts": {
            str(path.relative_to(ROOT)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in [
                RESULTS / "ordinary_contrasts.json",
                RESULTS / "family_contrasts.json",
                RESULTS / "bnci_hit_all.json",
                RESULTS / "bnci_decisions.json",
            ]
        },
        "method": {
            "metric": "AUC",
            "arm_order": ARMS,
            "weights": WEIGHTS,
            "sampling": "paired participants within each fixed study, studies weighted equally",
            "bootstrap_iterations": ITERATIONS,
            "analysis_seed": ANALYSIS_SEED,
            "training_seed": 20260918,
            "scope": "conditional on frozen shared source and target models, fixed studies and source selection",
            "limitations": [
                "saved participant-level AUC values, not replay from original predictions",
                "does not cover training randomness, source participant resampling or unseen studies",
                "all added contrasts and intervals exploratory, no multiple-comparison adjustment",
                "two exact-K cohorts differ; common cohort provided separately",
                "distinct physical calibration selections are not guaranteed statistically independent",
            ],
        },
        "panels": {},
        "verification": {},
    }
    for budget in [200, 800]:
        rows = paired_auc(ordinary, budget)
        subsets = [
            ("ordinary_all", rows),
            ("binary_only", {key: value for key, value in rows.items() if key[0] != "BNCI2014-008"}),
            ("bnci", {key: value for key, value in rows.items() if key[0] == "BNCI2014-008"}),
        ]
        for label, subset in subsets:
            output["panels"][f"{label}_K{budget}"] = summarize(subset)
        output["panels"][f"BI_family_K{budget}"] = summarize(paired_auc(family, budget))

    low = paired_auc(ordinary, 200)
    high = paired_auc(ordinary, 800)
    common = set(low) & set(high)
    for budget, rows in [(200, low), (800, high)]:
        output["panels"][f"common_K800_at_K{budget}"] = summarize(
            {key: rows[key] for key in low if key in common}
        )
    output["panels"]["common_K800_change_800_minus_200"] = summarize(
        {key: high[key] - low[key] for key in low if key in common}
    )
    for budget in ["200", "800"]:
        rows = paired_auc(ordinary, budget)
        saved = ordinary["contrasts"]["A_by_Q_interaction"]["exact_K"][budget]["participant_rows"]
        error = max(
            abs(
                float(rows[(item["study"], item["participant_key"])] @ WEIGHTS["A_by_Q_interaction"])
                - item["difference"]
            )
            for item in saved
        )
        assert error < 1e-12
        output["verification"][f"interaction_K{budget}_max_abs_error"] = error
    output["bnci_hit_all"] = summarize_hit(bnci)
    output["bnci_calibration"] = {}
    for budget in [200, 800]:
        items = [
            row for row in decisions["rows"]
            if row["K"] == budget and row["R"] == 10 and row["arm"] in ["A0Q1", "A1Q1"]
        ]
        output["bnci_calibration"][str(budget)] = {
            "unique_authorized_decisions": sorted(
                {row["calibration_decisions"]["authorized_groups"] for row in items}
            ),
            "unique_eligible_draws": sorted(
                {row["calibration_decisions"]["eligible_draws"] for row in items}
            ),
            "unique_candidate_trial_draws": sorted({row["epoch_candidate_draws"] for row in items}),
            "participants": len({row["participant_key"] for row in items}),
        }
    output["bnci_native_repetitions"] = sorted(
        {count for item in bnci["physical_groups"].values() for count in item["native_repetitions"]}
    )
    output["original_frozen_analysis_interaction_CI"] = {
        budget: ordinary["contrasts"]["A_by_Q_interaction"]["exact_K"][budget]["bootstrap"]
        for budget in ["200", "800"]
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evidence_audit.json"))
    parser.add_argument("--check", action="store_true", help="Compare numerical sections without writing.")
    args = parser.parse_args()
    result = recompute()
    if args.check:
        saved = read_json(args.output)
        for section in ["panels", "verification", "bnci_hit_all", "bnci_calibration", "bnci_native_repetitions"]:
            assert result[section] == saved[section], f"Mismatch in {section}"
        print("All numerical sections reproduce exactly.")
        return
    # Retain separately audited provenance notes when refreshing an existing report.
    previous = read_json(args.output) if args.output.exists() else {}
    previous.update(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
