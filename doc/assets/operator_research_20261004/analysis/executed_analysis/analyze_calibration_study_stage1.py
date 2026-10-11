"""Independent sealed-logit and legal-scope audit of the complete fixed study.

The residual readout uses held calibration truth only after predictions were
sealed. It never refits, chooses an arm, or consumes post-C800 target evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def digest_json(value):
    return digest_bytes(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False).encode("ascii"))


def sealed(record, directory):
    integrity = record["result_integrity"]
    assert integrity["schema"] == "n2p3_result_integrity/1"
    payload = {key: value for key, value in record.items() if key != "result_integrity"}
    assert digest_json({"record": payload, "files": integrity["files"]}) == integrity["record_sha256"]
    contents = {}
    for role, item in integrity["files"].items():
        assert Path(item["name"]).name == item["name"]
        content = (directory/item["name"]).read_bytes()
        assert len(content) == item["byte_size"]
        assert digest_bytes(content) == item["sha256"]
        contents[role] = content
    return contents


def import_frozen(run):
    frozen = run/"executed_source"
    sys.path.insert(0, str(frozen/"src"))
    path = frozen/"doc/assets/operator_research_20261004/continuous_model.py"
    spec = importlib.util.spec_from_file_location("calibration_audit_frozen_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rebuild_scopes(row, plan, original):
    from data.bnci2014_008_candidate import read_bnci2014_008_candidate_record
    from transfer.noise_model import array_digest

    source = Path(row["path"]).read_bytes()
    assert digest_bytes(source) == row["sha256"]
    raw = read_bnci2014_008_candidate_record(io.BytesIO(source), source_relative_path=row["subject"]+".mat",
                                           source_sha256=row["sha256"])
    del source
    width = round(plan["response_seconds"]*plan["sample_rate_hz"])
    pre = round(plan["pre_roll_seconds"]*plan["sample_rate_hz"])
    authorized_stop = int(raw.flash_sample[800])
    candidates = tuple(f"row:{r}|column:{c}" for r in range(6) for c in range(6))
    groups = []
    for fold in range(6):
        indices = np.arange(120*fold, 120*(fold+1))
        assert np.all(raw.selection_index[indices] == fold)
        start = int(raw.flash_sample[indices[0]])-pre
        stop = int(raw.flash_sample[indices[-1]])+width
        assert 0 <= start < stop <= authorized_stop
        if indices[0] > 0:
            assert int(raw.flash_sample[indices[0]-1])+width <= start
        assert int(raw.flash_sample[indices[-1]+1]) >= stop
        if groups:
            assert groups[-1]["scope"][1] <= start
        x = np.array(raw.eeg_uv[start:stop], dtype=np.float64)*1e-6
        onset = np.array(raw.flash_sample[indices]-start, dtype=np.int64)
        y = np.array(raw.raw_is_target[indices], dtype=bool)
        codes = np.array(raw.candidate_id[indices], dtype=int)
        assert np.array_equal(np.bincount(codes, minlength=12), np.full(12, 10))
        member = ((codes[:, None] == np.repeat(np.arange(6), 6)[None])
                  | (codes[:, None] == (6+np.tile(np.arange(6), 6))[None]))
        truth = f"row:{int(raw.target_row[indices[0]])}|column:{int(raw.target_col[indices[0]])}"
        assert np.array_equal(y, member[:, candidates.index(truth)])
        identity = f"{row['subject']}:selection{fold}:{row['sha256']}"
        receipt = original["groups"][fold]
        assert receipt["record_id"] == identity and receipt["raw_scope"] == [start, stop]
        assert receipt["label_indices"] == indices.tolist()
        assert receipt["input_sha256"] == array_digest(x, onset, y)
        assert receipt["prediction_signal_sha256"] == array_digest(x, onset, member)
        groups.append({"samples": x, "onsets": onset, "labels": y, "membership": member, "truth": truth,
                       "record_id": identity, "scope": [start, stop], "input_sha256": array_digest(x, onset, y),
                       "signal_sha256": array_digest(x, onset), "prediction_sha256": array_digest(x, onset, member)})
    assert original["C800_stop_sample"] == authorized_stop
    assert original["used_label_count"] == 720 and original["unused_authorized_labels"] == 80
    return groups, candidates


def event_design(group, basis):
    n = len(group["samples"])
    impulse, targets = np.zeros(n), np.zeros(n)
    impulse[group["onsets"]] = 1
    targets[group["onsets"]] = group["labels"]
    return np.column_stack([np.convolve(events, basis[:, column])[:n]
                            for events in (impulse, targets) for column in range(basis.shape[1])])


def residual(model, group):
    x = group["samples"]
    design = event_design(group, model.basis)
    if model.operator is not None:
        x = model.operator.apply(x)
        if model.filter_design:
            design = model.operator.apply(design)
    # Independently apply the shared nuisance projector to innovations.
    nuisance = np.column_stack([np.ones(len(x)), np.linspace(-1, 1, len(x))])
    q, _ = np.linalg.qr(model.noise.whiten(nuisance), mode="reduced")
    observed, d = model.noise.whiten(x), model.noise.whiten(design)
    observed -= q@(q.T@observed)
    d -= q@(q.T@d)
    return observed-d@model.mean, d


def acf_records(records):
    centered = [record-record.mean(axis=0) for record in records]
    denom = sum(np.sum(record**2, axis=0) for record in centered)
    assert np.all(denom > 0)
    result = {}
    for lag in [1, 2, 8, 32, 128]:
        numerator = sum(np.sum(record[lag:]*record[:-lag], axis=0) for record in centered if len(record) > lag)
        result[str(lag)] = (numerator/denom).tolist()
    return result


def residual_readout(model, groups, fold):
    training = [residual(model, group)[0] for i, group in enumerate(groups) if i != fold]
    held, d = residual(model, groups[fold])
    train_mse = sum(float(np.sum(record**2)) for record in training)/sum(record.size for record in training)
    held_mse = float(np.mean(held**2))
    old_train = model.provenance["train_residual_diagnostics"]
    np.testing.assert_allclose(train_mse, old_train["mean_squared_innovation"], rtol=1e-10, atol=1e-30)
    train_acf, held_acf = acf_records(training), acf_records([held])
    for lag in train_acf:
        np.testing.assert_allclose(np.mean(train_acf[lag]), old_train["mean_channel_acf"][lag], atol=2e-11)
    z = solve_triangular(np.linalg.cholesky(model.channel_covariance), held.T, lower=True).T
    u = d@np.linalg.cholesky(model.row_covariance)
    correction = solve_triangular(np.linalg.cholesky(np.eye(u.shape[1])+u.T@u), u.T@z, lower=True)
    predictive_q = float(np.sum(z**2)-np.sum(correction**2))
    residual_dof = (len(held)-2)*held.shape[1]
    return {"held_truth_role": "post_prediction_diagnostic_only", "cross_record_lag_pairs": 0,
            "train_temporal_innovation_mse": train_mse, "held_temporal_innovation_mse": held_mse,
            "held_train_variance_ratio": held_mse/train_mse, "held_train_std_ratio": np.sqrt(held_mse/train_mse),
            "held_spatially_whitened_mean_squared": float(np.mean(z**2)),
            "held_posterior_predictive_quadratic_per_projected_dimension": predictive_q/residual_dof,
            "train_acf_per_channel": train_acf, "held_acf_per_channel": held_acf,
            "held_acf_mean": {key: float(np.mean(value)) for key, value in held_acf.items()},
            "held_acf_max_abs": {key: float(np.max(np.abs(value))) for key, value in held_acf.items()}}


def paired_summary(delta, seed):
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(values))))
    randomization = (signs*values).mean(axis=1)
    return {"n_participants": len(values), "participant_differences": values.tolist(),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "bootstrap_participant_95pct_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "exact_paired_sign_flip_two_sided_p": float(np.mean(np.abs(randomization) >= abs(values.mean())-1e-14)),
            "positive_participants": int(np.sum(values > 0)), "negative_participants": int(np.sum(values < 0)),
            "zero_participants": int(np.sum(values == 0)), "interpretation": "descriptive_development_uncertainty_eight_participants"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    result_bytes = (run/"result.json").read_bytes()
    report = json.loads(result_bytes)
    assert report["status"] == "completed" and not report["subject_failures"]
    main_files = sealed(report, run)
    plan, source = json.loads(main_files["plan"]), json.loads(main_files["source"])
    assert digest_bytes(main_files["plan"]) == report["plan_sha256"]
    assert digest_json(source["files"]) == source["digest"] == report["source_sha256"]
    for path, expected in source["files"].items():
        assert digest_bytes((run/"executed_source"/path).read_bytes()) == expected
    module = import_frozen(run)
    arms = [item["name"] for item in plan["arms"]]
    subjects = [item["subject"] for item in plan["records"]]
    expected_keys = {(subject, fold, arm) for subject in subjects for fold in range(6) for arm in arms}
    actual_keys = [(cell["subject"], cell["fold"], cell["arm"]) for cell in report["cells"]]
    assert len(actual_keys) == len(set(actual_keys)) == report["requested_cells"] == 240
    assert set(actual_keys) == expected_keys
    cells = {(cell["subject"], cell["fold"], cell["arm"]): cell for cell in report["cells"]}
    independent = []
    model_hashes, prediction_hashes = {}, {}
    with threadpool_limits(limits=2):
        for row in plan["records"]:
            subject = row["subject"]
            scope = json.loads((run/subject/"scopes.json").read_bytes())
            groups, candidates = rebuild_scopes(row, plan, scope)
            for fold in range(6):
                for arm in arms:
                    cell = cells[(subject, fold, arm)]
                    directory = run/subject/f"fold{fold}_{arm}"
                    assert json.loads((directory/"record.json").read_bytes()) == cell
                    content = sealed(cell, directory)
                    assert cell["scope_ledger_sha256"] == digest_json(scope)
                    assert cell["training_labels"] == 600 and cell["authorized_C_labels"] == 800
                    assert cell["training_groups"] == 5 and cell["requested"] == 1
                    audit = {"subject": subject, "fold": fold, "arm": arm, "status": cell["status"],
                             "correct": False, "conditional_nll": None, "uniform_abstention_nll": float(np.log(36)),
                             "requested": 1, "coverage": 0}
                    if cell["status"] != "completed":
                        independent.append(audit)
                        continue
                    predicted = json.loads(content["prediction"])
                    assert tuple(predicted["candidate_ids"]) == candidates
                    logits = np.asarray(predicted["log_evidence"], dtype=float)
                    assert logits.shape == (36,) and np.isfinite(logits).all()
                    truth = groups[fold]["truth"]
                    assert predicted["truth_scorer_only"] == truth
                    nll = float(np.logaddexp.reduce(logits)-logits[candidates.index(truth)])
                    probability = np.exp(logits-logits.max())
                    probability /= probability.sum()
                    np.testing.assert_allclose(probability, predicted["probabilities"], rtol=1e-10, atol=1e-13)
                    np.testing.assert_allclose(nll, predicted["nll"], rtol=1e-10, atol=1e-9)
                    np.testing.assert_allclose(nll, cell["nll"], rtol=1e-10, atol=1e-9)
                    best = int(np.argmax(logits))
                    tied = np.count_nonzero(np.abs(logits-logits[best]) <= 1e-10) > 1
                    winner = None if tied else candidates[best]
                    correct = winner == truth
                    assert winner == predicted["winner"] == cell["winner"] and correct == cell["correct"]
                    assert predicted["input_sha256"] == cell["input_sha256"] == groups[fold]["prediction_sha256"]
                    model = module.OperatorGLS.load(io.BytesIO(content["model"]))
                    assert model.provenance["implementation_sha256"] == source["files"]["doc/assets/operator_research_20261004/continuous_model.py"]
                    expected_training = {group["record_id"]: group for i, group in enumerate(groups) if i != fold}
                    assert len(model.provenance["records"]) == 5
                    assert {r["record_id"] for r in model.provenance["records"]} == set(expected_training)
                    for ledger in model.provenance["records"]:
                        reference = expected_training[ledger["record_id"]]
                        assert ledger["input_sha256"] == reference["input_sha256"] and ledger["signal_sha256"] == reference["signal_sha256"]
                        assert ledger["events"] == 120 and ledger["target_count"] == 20
                    noise_records = model.noise.provenance["records"]
                    if model.noise.provenance["fit_scope"] != "analytic_reference":
                        assert {r["record_id"] for r in noise_records} == set(expected_training)
                        assert model.noise.provenance["implementation_sha256"] == source["files"]["src/transfer/temporal_noise.py"]
                    reread = model.score(groups[fold]["samples"], groups[fold]["onsets"], groups[fold]["membership"],
                        record_id=groups[fold]["record_id"], sfreq=256., channels=model.channels, candidate_ids=candidates)
                    np.testing.assert_array_equal(reread.log_evidence, logits)
                    truth_vector = np.zeros(36)
                    truth_vector[candidates.index(truth)] = 1
                    audit.update(correct=correct, conditional_nll=nll, uniform_abstention_nll=nll, coverage=1,
                        max_probability=float(probability.max()), truth_probability=float(probability[candidates.index(truth)]),
                        entropy=float(-np.sum(probability*np.log(np.maximum(probability, np.finfo(float).tiny)))),
                        brier=float(np.sum((probability-truth_vector)**2)), winner=winner, truth=truth,
                        fit_seconds=cell["fit_seconds"], exact_model_replay=True,
                        residual=residual_readout(model, groups, fold))
                    key = f"{subject}/{fold}/{arm}"
                    model_hashes[key], prediction_hashes[key] = digest_bytes(content["model"]), digest_bytes(content["prediction"])
                    independent.append(audit)
            print(json.dumps({"subject": subject, "audited_cells": 30}), flush=True)
            del groups
    participant = {}
    for subject in subjects:
        participant[subject] = {}
        for arm in arms:
            selected = [c for c in independent if c["subject"] == subject and c["arm"] == arm]
            valid = [c for c in selected if c["coverage"]]
            assert len(selected) == 6
            participant[subject][arm] = {"requested": 6, "coverage": len(valid)/6,
                "requested_hit": sum(c["correct"] for c in selected)/6,
                "conditional_nll": float(np.mean([c["conditional_nll"] for c in valid])) if valid else None,
                "uniform_abstention_nll": float(np.mean([c["uniform_abstention_nll"] for c in selected]))}
    aggregate = {}
    for arm in arms:
        valid = [c for c in independent if c["arm"] == arm and c["coverage"]]
        aggregate[arm] = {metric: float(np.mean([participant[s][arm][metric] for s in subjects]))
                          for metric in ("coverage", "requested_hit", "conditional_nll", "uniform_abstention_nll")}
        aggregate[arm].update(requested=48, scored=len(valid), correct=sum(c["correct"] for c in valid),
            median_max_probability=float(np.median([c["max_probability"] for c in valid])),
            mean_max_probability=float(np.mean([c["max_probability"] for c in valid])),
            mean_brier=float(np.mean([c["brier"] for c in valid])),
            mean_entropy=float(np.mean([c["entropy"] for c in valid])),
            nll_max=float(np.max([c["conditional_nll"] for c in valid])),
            nll_median=float(np.median([c["conditional_nll"] for c in valid])),
            held_acf1_mean=float(np.mean([c["residual"]["held_acf_mean"]["1"] for c in valid])),
            held_acf1_median_abs=float(np.median([abs(c["residual"]["held_acf_mean"]["1"]) for c in valid])),
            held_train_variance_ratio_median=float(np.median([c["residual"]["held_train_variance_ratio"] for c in valid])),
            held_train_variance_ratio_range=[float(np.min([c["residual"]["held_train_variance_ratio"] for c in valid])),
                                            float(np.max([c["residual"]["held_train_variance_ratio"] for c in valid]))],
            held_predictive_quadratic_ratio_median=float(np.median([c["residual"]["held_posterior_predictive_quadratic_per_projected_dimension"] for c in valid])))
    contrasts = {}
    control = "old_filtered_y"
    for index, arm in enumerate(arms):
        if arm == control:
            continue
        contrasts[arm] = {}
        for metric in ("conditional_nll", "requested_hit"):
            delta = [participant[s][arm][metric]-participant[s][control][metric] for s in subjects]
            contrasts[arm][metric] = paired_summary(delta, 20261004+index)
    summary = {"schema": "n2p3_independent_calibration_analysis/1", "run": str(run),
        "result_sha256": digest_bytes(result_bytes), "source_sha256": source["digest"],
        "analysis_script_sha256": digest_bytes(Path(__file__).read_bytes()),
        "evidence_scope": "public_calibration_development", "prediction_integrity_checked": 240,
        "source_snapshot_members_checked": len(source["files"]), "status_counts": dict(Counter(c["status"] for c in independent)),
        "participants": subjects, "arms": arms, "participant_metrics": participant, "aggregate": aggregate,
        "paired_contrasts": contrasts, "baseline_chance_nll": float(np.log(36)),
        "fit_budget": {"labels_available": 800, "labels_fit_per_fold": 600, "held_labels": 120, "unused_labels": 80,
                       "K200_group_validation": "not_estimable"},
        "statistical_unit": "eight_participants_each_contributing_six_dependent_CV_folds",
        "diagnostic_truth_role": "after_sealed_predictions_no_refit_no_T_use",
        "no_cross_record_lags": True, "all_240_frozen_scores_replayed_exactly": True,
        "elapsed_seconds": time.perf_counter()-began,
        "model_sha256": model_hashes, "prediction_sha256": prediction_hashes}
    assert digest_bytes((run/"result.json").read_bytes()) == summary["result_sha256"]
    (output/"analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    (output/"diagnostic_cells.json").write_text(json.dumps(independent, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "elapsed_seconds": summary["elapsed_seconds"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
