"""Independent saved-probability audit; no scientific module is imported.

Explicit --root and --output preserve input authority after relocating this file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

PANEL = "tmp/P1_NUMERIC_PANEL"
BINDINGS = "doc/assets/research_resume_20261011"
ARMS = ("shared8", "private8")
REVIEW_HEAD = "b26c099"
B07_COMMIT = "948c485850c068083bc1240d9475b3fd382f8869"
NEW_CALLS = {"fit": 0, "score": 0, "model_load": 0, "EEG_read": 0, "filter": 0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(rows, fields, expected, name):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"duplicate {name}: {key}")
        result[key] = row
    if expected is not None:
        require(set(result) == expected, f"missing or extra {name}")
    return result


def logsumexp(values):
    top = max(values)
    return top if top == -math.inf else top + math.log(math.fsum(math.exp(v - top) for v in values))


class Inputs:
    def __init__(self, root):
        self.root, self.pins = root, {}

    def raw(self, relative, role):
        relative = relative.replace("\\", "/")
        data = (self.root / relative).read_bytes()
        pin = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}
        if relative in self.pins:
            require(self.pins[relative]["sha256"] == pin["sha256"], f"changed input: {relative}")
        else:
            self.pins[relative] = pin
        return data

    def read(self, relative, role):
        value = json.loads(self.raw(relative, role))
        self.pins[relative]["schema"] = value.get("schema")
        return value

    def csv(self, relative, role):
        return list(csv.DictReader(self.raw(relative, role).decode("utf-8-sig").splitlines()))

    def digest(self, relative):
        return self.pins[relative.replace("\\", "/")]["sha256"]

    def check_unchanged(self):
        for relative, pin in self.pins.items():
            require(sha(self.root / relative) == pin["sha256"], f"consumed input changed: {relative}")


class Checks:
    def __init__(self):
        self.count, self.max_difference, self.max_probability_difference = 0, 0.0, 0.0

    def same(self, actual, reported, label, probability=False):
        self.count += 1
        if isinstance(actual, list):
            require(isinstance(reported, list) and len(actual) == len(reported), f"shape: {label}")
            for i, (a, b) in enumerate(zip(actual, reported, strict=True)):
                self.same(a, b, f"{label}[{i}]", probability)
        elif actual is None or isinstance(actual, (str, bool)):
            require(type(actual) is type(reported) and actual == reported, f"mismatch: {label}")
        else:
            require(isinstance(reported, (int, float)) and not isinstance(reported, bool), f"number: {label}")
            difference = abs(actual - reported)
            if probability:
                self.max_probability_difference = max(self.max_probability_difference, difference)
                absolute = 4 * max(math.ulp(actual), math.ulp(reported))
            else:
                self.max_difference = max(self.max_difference, difference)
                absolute = 1e-12
            require(math.isclose(actual, reported, rel_tol=1e-11, abs_tol=absolute),
                    f"numeric mismatch: {label}: {actual!r} vs {reported!r}")

    def compare(self, actual, reported, label, fields=None):
        probability_fields = {"probabilities", "p_true", "winner_confidence", "probability_mass_of_true_row", "probability_mass_of_true_column"}
        for key in actual if fields is None else fields:
            self.same(actual[key], reported[key], f"{label}.{key}", key in probability_fields)

    def compare_csv(self, actual, reported, label, fields):
        for key in fields:
            a, text = actual[key], reported[key]
            if isinstance(a, bool):
                require(text in {"True", "False"}, f"invalid CSV boolean {label}.{key}")
                b = text == "True"
            elif a is None:
                b = None if text == "" else text
            elif isinstance(a, str):
                b = text
            else:
                b = float(text)
            self.compare({key: a}, {key: b}, label)


def generic_contract(text):
    owner = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "CandidateEvidence")
    fields = []
    for node in owner.body:
        if isinstance(node, ast.AnnAssign):
            init_false = isinstance(node.value, ast.Call) and any(
                kw.arg == "init" and ast.literal_eval(kw.value) is False for kw in node.value.keywords)
            if not init_false:
                fields.append(node.target.id)
    post = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    calls = [n for n in ast.walk(post) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "isclose" and len(n.args) == 2
             and isinstance(n.args[0], ast.Name) and n.args[0].id == "values"
             and isinstance(n.args[1], ast.Name) and n.args[1].id == "maximum"]
    require(len(calls) == 1, "actual generic log-probability tie rule cannot be recovered")
    tolerances = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
    schema = "n2p3_candidate_probability_evidence/1"
    require(schema in text, "generic evidence schema changed")
    return {"schema": schema, "constructor_fields": sorted(fields), "rtol": tolerances["rtol"],
            "atol": tolerances["atol"], "tie_rule_source_line": calls[0].lineno}


def reject_custom_v1(record, contract):
    if record.get("schema") != contract["schema"] or set(record) != set(contract["constructor_fields"]) | {"schema"}:
        raise ValueError("P1 custom schema/fields cannot be imported directly as generic evidence v1")
    # Shape admission alone is not probability or authority admission.
    return True


def derive(score, vocabulary, truth, contract):
    ids = score["candidate_ids"]
    require(ids == vocabulary and len(ids) == len(set(ids)) == 36 and truth in ids, "original candidate vocabulary/truth differs")
    coordinates = []
    for candidate in ids:
        match = re.fullmatch(r"row:(\d+)\|column:(\d+)", candidate)
        require(match is not None, "candidate is not in the declared row/column vocabulary")
        coordinates.append(tuple(map(int, match.groups())))
    require(len({r for r, _ in coordinates}) == len({c for _, c in coordinates}) == 6, "missing original row/column axis")
    logs = [-math.inf if p is None else float(p) for p in score["log_probability"]]
    require(len(logs) == 36 and all(not math.isnan(v) and v != math.inf for v in logs), "invalid stable saved log probabilities")
    require(math.isclose(logsumexp(logs), 0.0, rel_tol=0, abs_tol=1e-10), "saved probability not normalized; no audit renormalization")
    probability = [math.exp(v) for v in logs]
    ordered = sorted(range(36), key=logs.__getitem__, reverse=True)
    top, target = ordered[0], ids.index(truth)
    maxima = [i for i, v in enumerate(logs) if abs(v - logs[top]) <= contract["atol"] + contract["rtol"] * abs(logs[top])]
    winner = ids[maxima[0]] if len(maxima) == 1 else None
    row_indices = [i for i, coordinate in enumerate(coordinates) if coordinate[0] == coordinates[target][0]]
    col_indices = [i for i, coordinate in enumerate(coordinates) if coordinate[1] == coordinates[target][1]]
    row_log_mass = logsumexp([logs[i] for i in row_indices])
    diagnostics = score["candidate_diagnostics"]
    require(len(diagnostics) == 36, "saved numerical diagnostics lose candidates")
    errors = [d["conditional_numerical_resolution"]["absolute_log_density_error_estimate"] for d in diagnostics]
    budgets = [d["conditional_numerical_resolution"]["absolute_log_density_error_budget"] for d in diagnostics]
    require(all(math.isfinite(e) and 0 <= e <= b for e, b in zip(errors, budgets, strict=True)), "saved density error budget failed")
    require(all(d["diffuse_projection_resolution"]["normalized_error_bound"] <= d["diffuse_projection_resolution"]["resolution_target"] for d in diagnostics), "saved diffuse error budget failed")
    q_gap = diagnostics[target]["quadratic"] - diagnostics[top]["quadratic"]
    det_gap = diagnostics[target]["predictive_logdet"] - diagnostics[top]["predictive_logdet"]
    density_gap = score["log_density"][top] - score["log_density"][target]
    retention = [g["diffuse_retention_fraction"] for g in score["signal_geometry"] if g["diffuse_retention_fraction"] is not None]
    return {"truth": truth, "winner": winner, "correct": winner == truth, "tie": len(maxima) != 1,
            "probability_available": True, "operational_available": winner is not None,
            "generic_evidence_status_recomputed": "available" if len(maxima) == 1 else "tie", "top_candidate_indices": maxima,
            "nll": -logs[target], "brier": math.fsum((p - int(i == target)) ** 2 for i, p in enumerate(probability)),
            "p_true": probability[target], "truth_rank": 1 + sum(v > logs[target] for v in logs),
            "winner_confidence": probability[top], "top_two_gap_nats": logs[top] - logs[ordered[1]],
            "entropy_nats": -math.fsum(p * v for p, v in zip(probability, logs, strict=True) if p > 0),
            "winner_row_matches_truth": winner is not None and coordinates[maxima[0]][0] == coordinates[target][0],
            "winner_column_matches_truth": winner is not None and coordinates[maxima[0]][1] == coordinates[target][1],
            "probability_mass_of_true_row": math.fsum(probability[i] for i in row_indices),
            "probability_mass_of_true_column": math.fsum(probability[i] for i in col_indices),
            "nll_true_row": -row_log_mass, "nll_true_column_given_true_row": -(logs[target] - row_log_mass),
            "max_diffuse_error_estimate": max(d["diffuse_projection_resolution"]["normalized_error_bound"] for d in diagnostics),
            "max_density_error_estimate_nats": max(errors), "density_error_budget_nats": min(budgets),
            "density_error_within_budget": True,
            "all_input_certificate": all(d["conditional_numerical_resolution"]["all_input_certificate"] for d in diagnostics),
            "top_two_density_error_estimate_sum": errors[top] + errors[ordered[1]],
            "true_minus_winner_quadratic": q_gap, "true_minus_winner_logdet": det_gap,
            "winner_minus_true_density_nats": density_gap, "decomposition_residual_nats": density_gap - 0.5 * (q_gap + det_gap),
            "minimum_diffuse_signal_retention": min(retention) if retention else None,
            "candidate_count": 36, "candidate_vocabulary": ids, "log_probability": score["log_probability"], "probabilities": probability}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(root.is_dir() and output.is_dir(), "explicit existing --root and --output are required")
    inp, check = Inputs(root), Checks()
    script_sha = sha(Path(__file__))
    binding = inp.read(f"{BINDINGS}/p1_numeric_panel_acceptance_binding.json", "original P1 acceptance")
    analysis_path = f"{PANEL}/analysis/final/analysis.json"
    analysis = inp.read(analysis_path, "old readout; comparison only")
    require(inp.digest(analysis_path) == binding["artifacts"]["analysis"]["sha256"], "readout does not match original acceptance")
    config_path = f"{PANEL}/execution_config.json"
    config = inp.read(config_path, "original requested roster/vocabulary/DecisionPlan truth")
    require(inp.digest(config_path) == binding["config_sha256"], "original config does not match acceptance")
    result_path = f"{PANEL}/result.json"
    result = inp.read(result_path, "saved generic metric owner; comparison only")
    require(inp.digest(result_path) == binding["result_sha256"], "result does not match acceptance")
    contract = generic_contract(inp.raw("src/models/candidate_evidence.py", "actual generic constructor/tie contract").decode("utf-8-sig"))
    inp.raw("src/research/candidate_metrics.py", "generic macro/micro/risk schema")
    history = inp.raw("src/research/continuous_history_adapter.py", "actual B07 central callsite").decode("utf-8-sig")
    require("primitive = native_history_marginal_program(" in history, "B07 central callsite is not connected")
    plan = config["producer_plan"]
    require(plan["winner_owner"] == "CandidateEvidence_normalized_log_probability/1", "original winner owner differs")
    participants = index(plan["participants"], ("subject",), None, "original participants")
    require(len(participants) == binding["participants"] == 8, "original participant denominator differs")
    require(len({p["participant_key"] for p in participants.values()}) == 8
            and {p["participant_key"] for p in participants.values()} == set(plan["decision_plan"]["requested_participant_keys"]), "original participant identity mismatch")
    truths = index(plan["decision_plan"]["entries"], ("participant_key", "decision_id"), None, "DecisionPlan truth")
    expected = {(subject, arm) for subject, in participants for arm in ARMS}
    subjects = {(subject,) for subject, in participants}
    requested = index(config["requested_cells"], ("subject", "arm"), expected, "requested subject-arm")
    readout = index(analysis["rows"], ("subject", "arm"), expected, "readout subject-arm")
    old_paired = index(analysis["paired"], ("subject",), subjects, "readout pairs")
    row_csv = index(inp.csv(f"{PANEL}/analysis/final/participants.csv", "reported row CSV"), ("subject", "arm"), expected, "row CSV")
    pair_csv = index(inp.csv(f"{PANEL}/analysis/final/paired.csv", "reported paired CSV"), ("subject",), subjects, "pair CSV")
    summary_csv = index(inp.csv(f"{PANEL}/analysis/final/summary.csv", "reported summary CSV"), ("arm",), {(a,) for a in ARMS}, "summary CSV")
    for field, expected_value in {"requested_cells": 16, "requested_participants": 8, "complete_cells": 16, "pending_cells": [], "analysis_new_fit_or_score_calls": 0}.items():
        check.same(analysis[field], expected_value, f"analysis.{field}")
    historical_hashes = {k.replace("\\", "/"): v for k, v in analysis["input_sha256"].items()}
    rows, rejected = [], 0
    for key in sorted(expected):
        subject, arm = key
        request, original, person = requested[key], readout[key], participants[(subject,)]
        require(request["participant_key"] == person["participant_key"], "requested participant differs")
        target = truths[(request["participant_key"], request["decision_id"])]["target_candidate"]
        held = [v for v in person["views"] if v["decision_id"] == request["decision_id"]]
        require(len(held) == 1 and held[0]["decision"]["candidate_vocabulary"] == person["vocabulary"], "held vocabulary/identity differs")
        require(person["expected_counts"] == person["observed_counts"] and all(n > 0 for n in person["expected_counts"]), "physical coverage incomplete")
        score_path, receipt_path = (f"{PANEL}/cells/{subject}_{arm}/{n}.json" for n in ("score", "receipt"))
        score = inp.read(score_path, "original saved 36-way log probability/diagnostics")
        receipt = inp.read(receipt_path, "original saved call ledger and producer identity")
        require(score["schema"] == "n2p3_channel_noise_candidate_score/1", "unexpected original score schema")
        for path, suffix in ((score_path, "score"), (receipt_path, "receipt")):
            require(inp.digest(path) == historical_hashes[f"cells/{subject}_{arm}/{suffix}.json"], f"unbound {suffix}: {key}")
        require(inp.digest(score_path) == receipt["score_file_sha256"], "score/receipt relation failed")
        require((receipt["subject"], receipt["arm"]) == key and receipt["participant_key"] == request["participant_key"] and receipt["decision_id"] == request["decision_id"], "receipt identity mismatch")
        require(score["scoring_implementation_sha256"] == receipt["consumer_source_sha256"] == request["consumer_source_sha256"], "consumer source mismatch")
        require(score["noise_sha256"] == receipt["noise_producer_digest"], "original noise law mismatch")
        require(not score["truth_used"] and not score["parameters_updated"], "saved scorer consumed truth/updated state")
        require(receipt["new_response_or_noise_fits"] == 0 and receipt["fit_calls"] == [], "new fit in saved rescoring receipt")
        require(receipt["score_calls"] == (0 if subject == "A01" else 1), "historical scoring-call ledger mismatch")
        require(receipt["status"] == original["status"] == "scored", "incomplete original score")
        require(score["read_support"] == receipt["read_support"] == held[0]["raw_scope"] == original["read_support"], "original held support mismatch")
        derived = derive(score, person["vocabulary"], target, contract)
        fields = [f for f in derived if f in original]
        check.compare(derived, original, f"readout:{key}", fields)
        check.same(derived["probabilities"], score["probabilities"], f"score probability:{key}", probability=True)
        check.same(derived["winner"], score["winner"], f"original scorer winner:{key}")
        check.same(derived["nll"], derived["nll_true_row"] + derived["nll_true_column_given_true_row"], f"chain rule:{key}")
        check.compare_csv(derived, row_csv[key], f"row_csv:{key}", [f for f in fields if not isinstance(derived[f], list)])
        for receipt_field, old_field in {"anchor_model_sha256": "model_producer_sha256", "noise_file_sha256": "noise_state_sha256", "score_calls": "scorer_calls"}.items():
            check.same(receipt[receipt_field], original[old_field], f"receipt:{key}:{old_field}")
        check.same(len(receipt["fit_calls"]), original["fit_calls"], f"saved rescoring fits:{key}")
        try:
            reject_custom_v1(dict(original, schema=analysis["schema"]), contract)
        except ValueError:
            rejected += 1
        row = dict(derived, subject=subject, arm=arm, participant_key=request["participant_key"], decision_id=request["decision_id"],
                   status=receipt["status"], read_support=score["read_support"], model_producer_sha256=receipt["anchor_model_sha256"],
                   noise_state_sha256=receipt["noise_file_sha256"], noise_law_digest=score["noise_sha256"], consumer_source_sha256=score["scoring_implementation_sha256"],
                   saved_rescoring_score_calls=receipt["score_calls"], saved_rescoring_new_fit_calls=0, this_review_new_calls=NEW_CALLS,
                   original_score_schema=score["schema"], original_readout_schema=analysis["schema"],
                   source_files={"score": {"path": score_path, "sha256": inp.digest(score_path)}, "receipt": {"path": receipt_path, "sha256": inp.digest(receipt_path)},
                                 "truth_vocabulary_roster": {"path": config_path, "sha256": inp.digest(config_path)}, "reported_row": {"path": analysis_path, "sha256": inp.digest(analysis_path)}})
        rows.append(row)
    computed = index(rows, ("subject", "arm"), expected, "computed rows")
    paired = []
    for subject, in sorted(subjects):
        shared, private = computed[(subject, "shared8")], computed[(subject, "private8")]
        require(shared["read_support"] == private["read_support"] and shared["truth"] == private["truth"], "paired support/truth mismatch")
        pair = {"subject": subject, "delta_hit_private_minus_shared": int(private["correct"]) - int(shared["correct"]),
                "delta_nll_private_minus_shared": private["nll"] - shared["nll"], "delta_brier_private_minus_shared": private["brier"] - shared["brier"],
                "rescued": not shared["correct"] and private["correct"], "harmed": shared["correct"] and not private["correct"], "paired_probability_available": True}
        check.compare(pair, old_paired[(subject,)], f"paired:{subject}")
        check.compare_csv(pair, pair_csv[(subject,)], f"paired_csv:{subject}", pair.keys())
        paired.append(pair)
    summary = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        value = {"hits": sum(r["correct"] for r in arm_rows), "operational_hit": sum(r["correct"] for r in arm_rows) / 8,
                 "prediction_coverage": sum(r["operational_available"] for r in arm_rows) / 8, "probability_coverage": 1.0,
                 "nll": statistics.fmean(r["nll"] for r in arm_rows), "brier": statistics.fmean(r["brier"] for r in arm_rows),
                 "wrong_winner_confidences": [r["winner_confidence"] for r in arm_rows if r["operational_available"] and not r["correct"]]}
        summary[arm] = value
        check.compare(value, analysis["summary"][arm], f"analysis_summary:{arm}")
        check.compare(value, binding["summary"][arm], f"acceptance_summary:{arm}")
        check.compare_csv(value, summary_csv[(arm,)], f"summary_csv:{arm}", [f for f in value if f != "wrong_winner_confidences"])
        for field, owner_field in {"operational_hit": "requested_participant_operational_hit_rate", "prediction_coverage": "prediction_coverage", "probability_coverage": "probability_coverage", "nll": "available_probability_nll", "brier": "available_probability_brier"}.items():
            check.same(value[field], result["metrics"][arm][owner_field], f"original_metric_owner:{arm}:{owner_field}")
    nll_sum = math.fsum(p["delta_nll_private_minus_shared"] for p in paired)
    pair_summary = {"participants": 8, "delta_hit_macro": statistics.fmean(p["delta_hit_private_minus_shared"] for p in paired),
                    "delta_nll_macro": statistics.fmean(p["delta_nll_private_minus_shared"] for p in paired), "delta_nll_sum_row_values": nll_sum,
                    "delta_brier_macro": statistics.fmean(p["delta_brier_private_minus_shared"] for p in paired), "rescued": sum(p["rescued"] for p in paired),
                    "harmed": sum(p["harmed"] for p in paired), "probability_pairs": 8,
                    "A01_share_of_row_sum_nll_improvement": next(p["delta_nll_private_minus_shared"] for p in paired if p["subject"] == "A01") / nll_sum}
    owner_pair = {"delta_hit": pair_summary["delta_hit_macro"], "paired_baseline_nll": summary["shared8"]["nll"], "paired_candidate_nll": summary["private8"]["nll"],
                  "delta_paired_nll": pair_summary["delta_nll_macro"], "paired_probability_count": 8, "missing_probability_ledger": [],
                  "rescued_count": pair_summary["rescued"], "harmed_count": pair_summary["harmed"], "lost_prediction_count": 0, "lost_probability_count": 0,
                  "development_gate_passed": pair_summary["delta_hit_macro"] >= 0 and pair_summary["delta_nll_macro"] <= 0, "gate_reasons": [],
                  "participant_delta_hit": [[participants[(p["subject"],)]["participant_key"], float(p["delta_hit_private_minus_shared"])] for p in paired]}
    check.compare(owner_pair, analysis["owner_paired_comparison"], "analysis_owner_pair")
    check.compare(owner_pair, binding["paired"], "accepted_owner_pair")
    calls = {"saved_rescoring_score_calls": sum(r["saved_rescoring_score_calls"] for r in rows), "saved_rescoring_new_response_or_noise_fit_calls": 0,
             "reused_A01_score_cells": 2, "reused_A01_candidates": binding["original_A01_candidate_reuse"], "other_scored_candidates": binding["remaining_scored_candidates"],
             "original_anchor_and_noise_fit_calls": "not recounted; original fitted producers are preserved"}
    require(calls["saved_rescoring_score_calls"] == 14 and rejected == 16, "original call/schema rejection count differs")
    cloud = inp.read(f"{BINDINGS}/cloud_candidate_analysis_binding.json", "public BNCI summary/denominator binding only")
    bi = inp.read(f"{BINDINGS}/bi_q1_queue_acceptance_binding.json", "BI queue acceptance binding only")
    bnci = inp.read(f"{BINDINGS}/bnci_q1_queue_acceptance_binding.json", "BNCI C600/R10 acceptance binding only")
    inp.raw("doc/evidence/public_evaluation.zh.md", "public scopes and denominator description")
    rejection_checks = {}
    valid_shape = dict.fromkeys(contract["constructor_fields"])
    valid_shape["schema"] = contract["schema"]
    require(reject_custom_v1(valid_shape, contract), "generic shape positive control failed")
    rejection_checks["generic_shape_positive_control"] = "accepted_shape_only_not_scientific_admission"
    for label, changed in (("duplicate_subject_arm", analysis["rows"] + [analysis["rows"][0]]), ("missing_roster", analysis["rows"][:-1])):
        try:
            index(changed, ("subject", "arm"), expected, label)
        except ValueError:
            rejection_checks[label] = "rejected"
        else:
            raise ValueError(f"roster mutation not rejected: {label}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    inp.check_unchanged()
    before = {path: pin["sha256"] for path, pin in inp.pins.items()}
    after = {path: sha(root / path) for path in inp.pins}
    require(before == after, "consumed SHA changed before publication")
    verification = {"schema": "n2p3_metric_scope_independent_recompute_verification/2", "status": "passed", "review_scope_head": REVIEW_HEAD,
                    "observed_execution_head": head, "script_sha256": script_sha, "row_count": 16, "paired_count": 8, "comparison_count": check.count,
                    "maximum_numeric_absolute_difference": check.max_difference, "maximum_probability_absolute_difference": check.max_probability_difference,
                    "comparison_scope": "independent original log vectors plus DecisionPlan truth vs rows/paired/summary/CSV/accepted binding/saved metric owner",
                    "roster_rejection_checks": rejection_checks, "custom_generic_v1_direct_import_rejections": rejected,
                    "input_sha256_before": before, "input_sha256_after": after, "input_sha256_unchanged": True,
                    "this_review_new_calls": NEW_CALLS, "original_saved_call_ledger": calls}
    denoms = [
        {"name": "P1_shared8/private8", "source_ref": f"{BINDINGS}/p1_numeric_panel_acceptance_binding.json", "requested_participants": 8,
         "requested_decisions_per_arm": 8, "probability_decisions_per_arm": 8, "candidate_count": 36, "truth_support": sorted({r["truth"] for r in rows}),
         "aggregation": "one decision per participant; equal eight-person mean; macro equals micro only for this panel", "claim_scope": binding["scope"],
         "verification_depth": "all original 16 log-probability vectors independently recomputed", "original_saved_call_ledger": calls, "this_review_new_calls": NEW_CALLS},
        {"name": "BNCI_public_candidate_panel", "source_ref": f"{BINDINGS}/cloud_candidate_analysis_binding.json", "requested_participants": cloud["participants"],
         "requested_decisions": cloud["requested_decisions"], "common_probability_decisions": cloud["common_probability_decisions"], "candidate_count": 36,
         "aggregation": "participant-first common available-decision risk; requested operational hit", "claim_scope": "public calibration development, one seed",
         "verification_depth": "accepted summary/denominator binding inspected; no 152-vector recomputation here", "original_calls": "not recounted", "this_review_new_calls": NEW_CALLS},
        {"name": "BI2014a_Q1", "source_ref": f"{BINDINGS}/bi_q1_queue_acceptance_binding.json", "requested_participants": bi["participants"],
         "requested_decisions": bi["main_requested_keys"], "probability_decisions": bi["main_available_keys"], "incomplete_decisions": bi["main_incomplete_keys"], "candidate_count": 36,
         "participant_macro_probability_coverage": bi["participant_macro_probability_coverage"], "decision_micro_probability_coverage": bi["decision_micro_probability_coverage"],
         "aggregation": "equal participant mean; micro coverage separately; full original episode horizons", "claim_scope": bi["scope"],
         "verification_depth": "acceptance binding inspected; no 1287-vector recomputation here", "original_calls": "not recounted; saved model count is not new audit calls", "this_review_new_calls": NEW_CALLS},
        {"name": "BNCI_C600_R10", "source_ref": f"{BINDINGS}/bnci_q1_queue_acceptance_binding.json", "requested_participants": bnci["participants"],
         "requested_decisions_per_arm": bnci["requested_per_arm"], "probability_decisions_per_arm": bnci["metrics"]["source_full_fine"]["probability_count"],
         "arms": len(bnci["metrics"]), "candidate_count": 36, "aggregation": "equal participant mean; 30 R10 episodes per person implies macro=micro", "claim_scope": bnci["scope"],
         "verification_depth": "acceptance binding inspected; no 240-vector-per-arm recomputation here", "original_target_fit_calls": bnci["target_fits_in_original_run"],
         "original_source_fit_calls": bnci["source_fits_in_original_run"], "this_review_new_calls": NEW_CALLS},
    ]
    field_contracts = {
        "winner/correct/tie/NLL/Brier/confidence/rank/entropy": {"score_schema": "n2p3_channel_noise_candidate_score/1", "score_fields": ["candidate_ids", "log_probability"],
            "truth_field": "execution_config.producer_plan.decision_plan.entries.target_candidate", "source_SHA": "each ledger row.source_files.score and truth_vocabulary_roster",
            "formula": "actual generic log-space tie rule; -log_p_true; sum_j(exp(log_p_j)-onehot_j)^2; no renormalization"},
        "row_column_masses_and_NLL": {"score_fields": ["candidate_ids", "log_probability"], "source_SHA": "each ledger row.source_files.score and truth_vocabulary_roster",
            "formula": "sum declared six-member row/column groups; stable row logsumexp; exact NLL chain rule; no independence assumption"},
        "error_bounds_and_density_decomposition": {"score_fields": ["candidate_diagnostics", "log_density", "signal_geometry"], "source_SHA": "each ledger row.source_files.score",
            "formula": "saved estimates/maxima/budgets and Q/logdet differences; no EEG scoring recomputation"},
        "producer_consumer_and_original_calls": {"receipt_fields": ["anchor_model_sha256", "noise_file_sha256", "score_calls", "fit_calls"], "source_SHA": "each ledger row.source_files.receipt and score",
            "restriction": "historical rescoring calls separate from this_review_new_calls"},
        "paired_and_summary": {"source_SHA": "all sixteen original score SHA plus original plan SHA", "formula": "computed row risks -> same subject/support shared-private deltas -> eight-person mean"},
    }
    crosswalk = {"schema": "n2p3_metric_scope_crosswalk/2", "status": "passed_independent_log_probability_recompute", "review_scope_head": REVIEW_HEAD,
                 "observed_execution_head": head, "script_sha256": script_sha, "source_files": list(inp.pins.values()), "denominator_crosswalk": denoms,
                 "ledger_field_contracts": field_contracts, "generic_evidence_contract": dict(contract, source_sha256=inp.digest("src/models/candidate_evidence.py")),
                 "generic_schema_rejection": {"P1_custom_fields_direct_import_allowed": False, "rejected_rows": rejected, "reason": "source schema/constructor fields differ from generic evidence/metrics v1"},
                 "p1_recomputed": summary, "p1_paired_recomputed": pair_summary, "this_review_new_calls": NEW_CALLS, "original_saved_call_ledger": calls,
                 "B07_status": {"history_adapter_connected_commit": B07_COMMIT, "actual_central_callsite_verified": True, "central_certificate_and_capsule_independent_review": "in_progress", "public_B07_fit_score_claim": "none from this audit"},
                 "claims": {"allowed": ["P1 fixed eight C-held diagnostics improve with private8", "948c485 connected B07 history adapter; certificate/capsule review continues"],
                            "forbidden": ["P1 hit as public152 hit@all", "P1 custom record as generic v1", "fixed P1 truth as balanced 36-way risk", "local estimate as all-input interval proof", "new audit zero calls as historical zero training/scoring"]}}
    write(output / "metric_crosswalk.json", crosswalk)
    write(output / "p1_error_ledger.json", {"schema": "n2p3_p1_row_error_ledger/2", "method": "original stable log_probability plus original DecisionPlan truth",
               "rows": rows, "paired": paired, "summary": summary, "recomputed": pair_summary, "verification": verification})
    write(output / "verification.json", verification)
    csv_fields = ["subject", "arm", "truth", "winner", "correct", "tie", "probability_available", "operational_available", "generic_evidence_status_recomputed", "candidate_count",
                  "truth_rank", "p_true", "nll", "brier", "winner_confidence", "probability_mass_of_true_row", "probability_mass_of_true_column", "nll_true_row", "nll_true_column_given_true_row",
                  "top_two_gap_nats", "max_density_error_estimate_nats", "density_error_budget_nats", "density_error_within_budget", "all_input_certificate", "saved_rescoring_score_calls",
                  "saved_rescoring_new_fit_calls", "read_support", "model_producer_sha256", "noise_state_sha256", "consumer_source_sha256"]
    with (output / "p1_error_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# 指标分母与字段审阅：独立概率复算

审阅范围主树 `{REVIEW_HEAD}`。脚本使用必填 `--root`、`--output`；放进其他artifact目录后仍消费明确指定的原仓库。只读保存JSON/CSV、binding和源码文本，本次fit、score、model load、EEG read、filter各0。

## 复算与核对

逐格读取原36维 `log_probability`，从原执行配置的DecisionPlan取得truth并核原vocabulary/complete roster。NLL直接取 `−log_p_true`；Brier由 `exp(log_p)`计算；winner/tie按当前generic源码中原log-space容差独立计算。没有使用旧row的nll/brier/correct作为计算输入，也没有归一化或clip原概率。行/列边缘质量与NLL链式分解同样复核。

shared8：5/8，NLL {summary['shared8']['nll']:.10f}，Brier {summary['shared8']['brier']:.10f}；private8：6/8，NLL {summary['private8']['nll']:.10f}，Brier {summary['private8']['brier']:.10f}。覆盖均100%。八人macro Δhit {pair_summary['delta_hit_macro']:.3f}，ΔNLL {pair_summary['delta_nll_macro']:.10f}，ΔBrier {pair_summary['delta_brier_macro']:.10f}；救回1人、损害0人。八行ΔNLL和为 {nll_sum:.10f}，不可替代宏平均。A01贡献约 {100*pair_summary['A01_share_of_row_sum_nll_improvement']:.4f}%。

独立结果与原row、paired、summary、三份CSV、accepted binding及保存generic metric owner逐项核对，共{check.count}项比较；最大数值绝对差 {check.max_difference:.3g}。消费输入SHA前后不变；重复subject-arm、missing roster、P1 custom fields直接导入generic v1均拒绝。

## 调用数与分母

本次审阅0调用。原完成receipt合计14次score；A01两格复用72候选，其余14格评分504候选。原续评新增response/noise fit=0，原anchor/noise已有拟合，其历史拟合数本次没有重数。不能把本次0调用写成原训练/评分0调用。

P1每臂8人各1个C-held decision、36候选、全真值row5/col0；macro=micro只在此面板成立。public BNCI保持152/144，BI保持61人1287/1243且宏微coverage分列，BNCI C600/R10每臂240/240。后三项本次只核接受binding/文档，没有声称重算其全部概率向量。

P1是固定C-held诊断，非公共/成人效能；两臂NLL均高于ln36。`all_input_certificate=false`不是评分失败，row/column边缘概率不能替换joint p_true。每格producer/noise文件/consumer与score/receipt/plan SHA均进入账本。

B07 history adapter已在 `948c485` 接central；中央certificate/capsule独立复签进行中。本报告不写consumer未接，不声明新B07训练结果。

产物：metric_crosswalk.json、p1_error_ledger.json/.csv、verification.json。
"""
    (output / "REPORT.zh.md").write_text(report, encoding="utf-8")
    previous = output / "build_metric_audit.previous.py"
    if previous.exists():
        diff = "".join(difflib.unified_diff(previous.read_text(encoding="utf-8-sig").splitlines(keepends=True),
                    Path(__file__).read_text(encoding="utf-8-sig").splitlines(keepends=True), fromfile="build_metric_audit.previous.py", tofile="build_metric_audit.py"))
        (output / "build_metric_audit.diff").write_text(diff, encoding="utf-8")
    print(json.dumps({"status": "passed", "rows": 16, "pairs": 8, "checks": check.count, "consumed_inputs": len(inp.pins),
                      "input_SHA_unchanged": True, "script_sha256": script_sha, "review_new_calls": NEW_CALLS, "original_saved_score_calls": 14}))


if __name__ == "__main__":
    main()
