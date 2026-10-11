"""Phase-4 development study: flexible prior, spatial precision and ERP hybrid.

All choices are fixed before held-selection scoring. Hyperprior averaging is a
predeclared categorical mixture over kernel scales, while spatial weighting is
an explicit covariance treatment estimated from the five training selections.
The equal AR/Toeplitz blend is a robust model mixture, not a learned test gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODULE_DIR))

from run_calibration_study import executing_files, selection_views  # noqa: E402
from selection_erp_model import SelectionERP  # noqa: E402

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS  # noqa: E402
from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.continuous_decision import smooth_erp_basis  # noqa: E402
from transfer.noise_model import freeze_metadata  # noqa: E402


def load_gls():
    spec = importlib.util.spec_from_file_location("phase4_continuous_model", MODULE_DIR / "continuous_model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OperatorGLS


def fixed_spatial_weights():
    # BNCI order Fz,Cz,Pz,Oz,P3,P4,PO7,PO8. This is a declared anatomical
    # prior, with Pz emphasized, and is normalized to mean one.
    values = np.array([.75, 1.0, 1.5, 1.2, 1.0, 1.0, .9, .9], dtype=float)
    return values / values.mean()


def empirical_spatial_weights(records):
    values = []
    for record in records:
        x, y = record.samples, record.target_mask
        windows = []
        for onset, label in zip(record.onsets, y, strict=True):
            left, right = int(onset + round(.25 * 256)), int(onset + round(.6 * 256))
            if right <= len(x):
                windows.append((x[left:right].mean(axis=0), int(label)))
        if windows:
            values.extend(windows)
    targets = np.asarray([v for v, label in values if label == 1])
    non_targets = np.asarray([v for v, label in values if label == 0])
    if len(targets) < 2 or len(non_targets) < 2:
        raise ValueError("training selections cannot estimate spatial ERP weights")
    effect = np.abs(targets.mean(axis=0) - non_targets.mean(axis=0))
    pooled = np.sqrt((targets.var(axis=0) + non_targets.var(axis=0)) / 2)
    dprime = effect / np.maximum(pooled, np.finfo(float).tiny)
    dprime = np.clip(dprime, np.quantile(dprime, .1), np.quantile(dprime, .9))
    weights = dprime / max(float(np.mean(dprime)), np.finfo(float).tiny)
    return .5 * weights + .5 * fixed_spatial_weights()


def spatial_variant(model, weights, rule):
    OperatorGLS = type(model)
    precision = np.sqrt(np.asarray(weights, dtype=float))
    inv = np.diag(1 / precision)
    covariance = inv @ np.asarray(model.channel_covariance) @ inv
    provenance = dict(model.provenance)
    provenance["spatial_weight_rule"] = rule
    provenance["spatial_weights"] = list(map(float, weights))
    provenance["spatial_weight_fit_scope"] = "training_selections_only"
    return OperatorGLS(model.basis, model.mean, model.row_covariance, covariance, model.noise,
                       model.operator, model.filter_design, model.sfreq, model.channels,
                       model.candidate_ids, freeze_metadata(provenance))


def score_record(model, record, membership, truth):
    result = model.score(record.samples, record.onsets, membership,
                         record_id=record.record_id, sfreq=256., channels=BNCI2014_008_CHANNELS,
                         candidate_ids=tuple(f"row:{r}|column:{c}" for r in range(6) for c in range(6)))
    truth_index = list(result.candidate_ids).index(truth)
    nll = float(logsumexp(result.log_evidence) - result.log_evidence[truth_index])
    return result, nll


def mixture(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    value = np.mean(probabilities, axis=0)
    winner = int(np.argmax(value))
    return value, winner


def run(plan_path, output):
    plan = json.loads(Path(plan_path).read_bytes())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = executing_files()
    for relative, digest in source.items():
        target = output / "executed_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source snapshot changed during phase4 setup")
    (output / "source.json").write_text(json.dumps({"files": source, "digest": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()}, indent=2), encoding="utf-8")
    rows = []
    OperatorGLS = load_gls()
    basis = smooth_erp_basis(256., 1.2, 8)
    candidate_ids = tuple(f"row:{r}|column:{c}" for r in range(6) for c in range(6))
    for row in plan["records"]:
        groups, ledger = selection_views(row, plan)
        sos = plan["operator_sos"]
        operator = FiniteFilterOperator(tuple(tuple(v for v in section) for section in sos), plan["padding_limit"])
        for fold in range(6):
            held, membership, truth = groups[fold]
            training = [record for i, (record, _, _) in enumerate(groups) if i != fold]
            with cpu_thread_budget(2):
                models = {}
                for lam in plan["kernel_prior_grid"]:
                    models[f"raw_ar8_lambda_{lam:g}"] = OperatorGLS.fit(training, basis=basis,
                        sfreq=256., channels=BNCI2014_008_CHANNELS, candidate_ids=candidate_ids, order=8,
                        noise_method="burg", operator=None, filter_design=True, ridge=float(lam),
                        feasible_iterations=3, kernel_prior_scale="innovation", startup_mode="stationary_innovations",
                        implementation_sha256=source["doc/assets/operator_research_20261004/continuous_model.py"])
                base = models["raw_ar8_lambda_1"]
                models["raw_ar8_pz_fixed"] = spatial_variant(base, fixed_spatial_weights(), "fixed_anatomical_Pz_priority")
                models["raw_ar8_spatial_training"] = spatial_variant(base, empirical_spatial_weights(training), "training_dprime_shrink_to_Pz_prior")
                models["toeplitz"] = SelectionERP.fit(training, family="toeplitz_lda", baseline_mode="none",
                    sfreq=256., channels=BNCI2014_008_CHANNELS, candidate_ids=candidate_ids, operator=operator,
                    expected_label_count=600)
                models["xdawn"] = SelectionERP.fit(training, family="xdawn_erp", baseline_mode="mean_only",
                    sfreq=256., channels=BNCI2014_008_CHANNELS, candidate_ids=candidate_ids, operator=operator,
                    expected_label_count=600)
                outputs = {}
                for name, model in models.items():
                    result, nll = score_record(model, held, membership, truth)
                    outputs[name] = {"probabilities": result.probabilities.tolist(), "nll": nll,
                                     "winner": result.winner, "correct": result.winner == truth}
                probs = [np.asarray(outputs[name]["probabilities"]) for name in models if name.startswith("raw_ar8_lambda_")]
                mixed, winner = mixture(probs)
                truth_index = candidate_ids.index(truth)
                outputs["raw_ar8_hyperprior_mixture"] = {"probabilities": mixed.tolist(),
                    "nll": float(-np.log(max(mixed[truth_index], np.finfo(float).tiny))),
                    "winner": candidate_ids[winner], "correct": candidate_ids[winner] == truth,
                    "components": [name for name in models if name.startswith("raw_ar8_lambda_")],
                    "component_weights": [1/len(probs)]*len(probs)}
                blend = .5 * np.asarray(outputs["raw_ar8_lambda_1"]["probabilities"]) + .5 * np.asarray(outputs["toeplitz"]["probabilities"])
                blend_winner = int(np.argmax(blend))
                outputs["equal_raw_toeplitz_mixture"] = {"probabilities": blend.tolist(),
                    "nll": float(-np.log(max(blend[truth_index], np.finfo(float).tiny))),
                    "winner": candidate_ids[blend_winner], "correct": candidate_ids[blend_winner] == truth,
                    "components": ["raw_ar8_lambda_1", "toeplitz"], "component_weights": [.5, .5]}
                blend_three = np.mean([np.asarray(outputs[name]["probabilities"])
                                       for name in ("raw_ar8_lambda_1", "toeplitz", "xdawn")], axis=0)
                three_winner = int(np.argmax(blend_three))
                outputs["equal_raw_toeplitz_xdawn_mixture"] = {"probabilities": blend_three.tolist(),
                    "nll": float(-np.log(max(blend_three[truth_index], np.finfo(float).tiny))),
                    "winner": candidate_ids[three_winner], "correct": candidate_ids[three_winner] == truth,
                    "components": ["raw_ar8_lambda_1", "toeplitz", "xdawn"], "component_weights": [1/3]*3}
                for name, value in outputs.items():
                    rows.append({"subject": row["subject"], "fold": fold, "arm": name, "truth": truth, **value})
        del groups
    result = {"schema": "n2p3_phase4_flexible_spatial_hybrid/1", "plan": plan,
              "source": source, "rows": rows, "requested": len(rows),
              "statistical_unit": "eight_participants_with_six_dependent_CV_folds",
              "held_selection_used_for_parameter_selection": False,
              "interpretation": "predeclared_training_group_mixture_and_spatial_prior_development"}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
