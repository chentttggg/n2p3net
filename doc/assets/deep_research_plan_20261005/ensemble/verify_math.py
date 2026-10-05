"""Deterministic candidate-pooling counterexamples; no fitting or training.

Run from the isolated checkout with the repository virtual environment.
Independent enumeration oracles are compared with the live membership scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import data.candidate_task as candidate_task  # noqa: E402
import models.candidate_evidence as candidate_evidence  # noqa: E402
import transfer.decision_loss as decision_loss  # noqa: E402
from data.candidate_task import MembershipDecision  # noqa: E402
from models.candidate_evidence import membership_decision_scores, membership_evidence  # noqa: E402
from transfer.decision_loss import membership_decision_loss  # noqa: E402


def softmax(scores):
    values = np.asarray(scores, dtype=float)
    unnormalized = np.exp(values - values.max())
    return unnormalized / unnormalized.sum()


def bernoulli_oracle(log_odds, membership, prior):
    """Enumerate complete candidate hypotheses using Bernoulli probabilities."""
    likelihoods = []
    target = 1 / (1 + np.exp(-np.asarray(log_odds, dtype=float)))
    for column in np.asarray(membership, dtype=bool).T:
        likelihood = 1.0
        for label, probability in zip(column, target, strict=True):
            likelihood *= probability if label else 1 - probability
        likelihoods.append(likelihood)
    values = np.asarray(likelihoods) * prior
    return values / values.sum()


def direct_decision():
    return MembershipDecision.direct(
        ["A", "B", "C", "B", "C", "A"], ["A", "B", "C"],
        event_ids=[f"flash:{index}" for index in range(6)],
        repetition_indices=np.repeat(np.arange(2), 3), repetitions=2,
    )


def verify():
    records = {}
    group = direct_decision()
    values = np.array([0.4, -0.1, 0.3, 0.6, -0.2, 0.7])
    prior = np.array([0.2, 0.5, 0.3])
    temperature, offset = 1.7, 0.25
    result = membership_decision_scores(values, group, prior=prior,
                                        temperature=temperature, logit_offset=offset)
    oracle = bernoulli_oracle((values - offset) / temperature, group.membership, prior)
    np.testing.assert_allclose(result.posterior, oracle, rtol=1e-13, atol=1e-13)
    records["nn_mapping_bernoulli_oracle"] = {
        "pass": True, "posterior": result.posterior.tolist(),
        "oracle_max_error": float(np.max(np.abs(result.posterior - oracle))),
        "assumptions": "conditional independence; corrected trial odds; frozen prior",
    }

    raw_pi, positive_weight = 0.2, 4.0
    likelihood_ratio = np.array([0.4, 1.7, 3.0])
    posterior = raw_pi * likelihood_ratio / (raw_pi * likelihood_ratio + 1 - raw_pi)
    weighted_posterior = positive_weight * posterior / (
        positive_weight * posterior + 1 - posterior)
    logits = np.log(weighted_posterior / (1 - weighted_posterior))
    rho = math.log(raw_pi / (1 - raw_pi)) + math.log(positive_weight)
    np.testing.assert_allclose(logits - rho, np.log(likelihood_ratio), atol=1e-13)
    unweighted_logits = np.log(posterior / (1 - posterior))
    np.testing.assert_allclose(unweighted_logits - math.log(raw_pi / (1 - raw_pi)),
                               np.log(likelihood_ratio), atol=1e-13)
    records["weighted_ce_prior_ratio"] = {
        "pass": True, "effective_trial_prior": raw_pi, "positive_weight": positive_weight,
        "rho": rho, "balanced_target_rho_is_zero": True,
        "limitation": "not a guarantee of empirical calibration or target class-conditionals",
    }

    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    differentiable = membership_decision_loss(tensor, group, 1, temperature=temperature,
                                             prior=prior, logit_offset=offset)
    differentiable.loss.backward()
    expected = group.membership @ (oracle - np.eye(3)[1]) / temperature
    np.testing.assert_allclose(tensor.grad.numpy(), expected, atol=1e-13)
    records["membership_gradient"] = {
        "pass": True, "autograd": tensor.grad.tolist(), "oracle": expected.tolist(),
    }

    permutation = np.array([2, 0, 1])
    permuted = replace(group,
                       candidate_vocabulary=tuple(group.candidate_vocabulary[i] for i in permutation),
                       membership=group.membership[:, permutation])
    permuted_result = membership_decision_scores(values, permuted, prior=prior[permutation],
                                                 temperature=temperature, logit_offset=offset)
    np.testing.assert_allclose(permuted_result.posterior, oracle[permutation], atol=1e-13)
    assert permuted_result.predicted == result.predicted
    unavailable = np.ones(6, dtype=bool)
    unavailable[0] = False
    incomplete = membership_decision_scores(values, group, available=unavailable)
    assert incomplete.predicted is None and not incomplete.eligible
    assert incomplete.reason == "incomplete_candidate_coverage"
    assert len(incomplete.posterior) == 3
    all_missing = membership_decision_scores(np.full(6, np.nan), group,
                                             available=np.zeros(6, dtype=bool), prior=prior)
    assert all_missing.predicted is None and all_missing.reason == "no_usable_evidence"
    invalid_loss = membership_decision_loss(torch.tensor(values, dtype=torch.float64), group, 1,
                                           available=unavailable)
    assert invalid_loss.loss is None
    tie = membership_decision_loss(torch.zeros(6, dtype=torch.float64), group, 1)
    assert tie.status == "tie" and tie.loss is not None and tie.argmax_candidate is None
    records["membership_permutation_and_failure"] = {
        "pass": True, "incomplete_reason": incomplete.reason,
        "missing_reason": all_missing.reason, "frozen_candidates": list(incomplete.candidate_vocabulary),
        "prior_only_not_operational_prediction": True, "tie_trains_but_abstains": True,
    }

    row_column = MembershipDecision.row_column(
        np.tile(np.arange(4), 2), n_rows=2, n_columns=2,
        event_ids=[f"rowcolumn:{index}" for index in range(8)],
        repetition_indices=np.repeat(np.arange(2), 4), repetitions=2,
    )
    row_values = np.array([0.1, 0.5, -0.6, 0.2, 0.3, -0.1, 0.8, 0.4])
    row_posterior = membership_decision_scores(row_values, row_column).posterior
    np.testing.assert_allclose(row_posterior,
                               bernoulli_oracle(row_values, row_column.membership, np.full(4, .25)),
                               atol=1e-13)
    np.testing.assert_allclose(row_posterior,
                               membership_decision_scores(row_values + 17.0, row_column).posterior,
                               atol=1e-13)
    records["row_column_not_stimulus_softmax"] = {
        "pass": True, "n_stimulus_types": 4, "n_physical_choices": 4,
        "candidate_posterior": row_posterior.tolist(), "equal_exposure_offset_invariant": True,
    }

    unbalanced = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=bool)
    unbalanced_values = np.array([0.6, 0.7, 0.8, 0.1])
    scores_before = membership_evidence(unbalanced_values, unbalanced)
    scores_after = membership_evidence(unbalanced_values, unbalanced, logit_offset=2.0)
    assert np.argmax(scores_before) == 0 and np.argmax(scores_after) == 2
    records["unequal_exposure_offset_changes_winner"] = {
        "pass": True, "counts": unbalanced.sum(axis=0).tolist(),
        "scores_before": scores_before.tolist(), "scores_after": scores_after.tolist(),
    }

    anchor = np.array([0.51, 0.49])
    new_expert = np.array([0.01, 0.99])
    mixed = (anchor + new_expert) / 2
    assert np.argmax(anchor) == 0 and np.argmax(mixed) == 1
    assert -math.log(mixed[0]) <= (-math.log(anchor[0]) - math.log(new_expert[0])) / 2
    anchor_all = np.array([[.51, .49], [1e-8, 1 - 1e-8]])
    alternative_all = np.array([[.01, .99], [.49, .51]])
    mixed_all = (anchor_all + alternative_all) / 2
    anchor_hit = float(np.mean(np.argmax(anchor_all, axis=1) == 0))
    mixed_hit = float(np.mean(np.argmax(mixed_all, axis=1) == 0))
    anchor_nll, mixed_nll = -np.log(anchor_all[:, 0]).mean(), -np.log(mixed_all[:, 0]).mean()
    assert mixed_hit < anchor_hit and mixed_nll < anchor_nll
    records["mixture_hit_is_not_monotonic"] = {
        "pass": True, "anchor": anchor.tolist(), "added_expert": new_expert.tolist(),
        "mixed": mixed.tolist(), "anchor_hit_two_decisions": anchor_hit,
        "mixed_hit_two_decisions": mixed_hit, "anchor_nll": float(anchor_nll),
        "mixed_nll": float(mixed_nll), "jensen_bound_is_to_average_nll_not_best_expert": True,
    }

    first, second = np.array([.8, .15, .05]), np.array([.05, .3, .65])
    probability_pool = (first + second) / 2
    log_pool = softmax((np.log(first) + np.log(second)) / 2)
    assert np.argmax(probability_pool) == 0 and np.argmax(log_pool) == 1
    confident = softmax([2.2, 0.0])
    softened = softmax(np.array([2.2, 0.0]) / 10)
    rival = np.array([.1, .9])
    assert np.argmax(confident) == np.argmax(softened) == 0
    assert np.argmax((confident + rival) / 2) == 0
    assert np.argmax((softened + rival) / 2) == 1
    records["pooling_and_temperature_are_treatments"] = {
        "pass": True, "probability_pool": probability_pool.tolist(),
        "log_pool": log_pool.tolist(), "single_expert_temperature_preserves_hit": True,
        "fusion_temperature_changes_hit": True,
    }

    expert_probabilities = torch.tensor([[.6, .3, .1], [.1, .2, .7]], dtype=torch.float64)
    gate_logits = torch.tensor([.4, -.1], dtype=torch.float64, requires_grad=True)
    weights = torch.softmax(gate_logits, dim=0)
    pooled = weights @ expert_probabilities
    loss = -torch.log(pooled[0])
    loss.backward()
    w = weights.detach().numpy()
    expected_gate = w * (1 - expert_probabilities.numpy()[:, 0] / float(pooled[0].detach()))
    np.testing.assert_allclose(gate_logits.grad.numpy(), expected_gate, atol=1e-13)
    logits = torch.tensor([[.2, -.3, .8], [.7, -.2, .1]], dtype=torch.float64, requires_grad=True)
    per_expert = torch.softmax(logits, dim=1)
    frozen_weights = torch.tensor([.4, .6], dtype=torch.float64)
    neural_pool = frozen_weights @ per_expert
    (-torch.log(neural_pool[1])).backward()
    expert_p = per_expert.detach().numpy()
    responsibility = frozen_weights.numpy() * expert_p[:, 1] / float(neural_pool[1].detach())
    expected_neural = responsibility[:, None] * (expert_p - np.eye(3)[1])
    np.testing.assert_allclose(logits.grad.numpy(), expected_neural, atol=1e-13)
    records["gate_and_expert_gradients"] = {
        "pass": True, "gate_gradient": gate_logits.grad.tolist(),
        "expert_responsibilities": responsibility.tolist(), "expert_gradient": logits.grad.tolist(),
    }

    inverse_temperature = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    evidence = torch.tensor(group.membership.T @ values, dtype=torch.float64)
    temperature_posterior = torch.softmax(
        inverse_temperature * evidence + torch.tensor(np.log(prior)), dim=0)
    (-torch.log(temperature_posterior[1])).backward()
    beta_oracle = float(np.dot(temperature_posterior.detach().numpy(), evidence.numpy())
                        - evidence.numpy()[1])
    np.testing.assert_allclose(float(inverse_temperature.grad), beta_oracle, atol=1e-13)
    records["inverse_temperature_gradient"] = {
        "pass": True, "autograd": float(inverse_temperature.grad), "oracle": beta_oracle,
    }

    # A rounded probability of zero can still represent a finite log probability.
    log_pool_true = float(np.logaddexp(-1000 + math.log(.5), -1100 + math.log(.5)))
    assert math.exp(log_pool_true) == 0.0 and math.isfinite(log_pool_true)
    with localcontext() as context:
        context.prec = 70
        decimal_probability = (Decimal(-1000).exp() + Decimal(-1100).exp()) / 2
        decimal_nll = float(-decimal_probability.ln())
    np.testing.assert_allclose(-log_pool_true, decimal_nll, atol=1e-12)
    records["log_pool_underflow_is_not_true_zero_mass"] = {
        "pass": True, "rounded_probability": 0.0, "stable_nll": -log_pool_true,
        "decimal_oracle_nll": decimal_nll, "probability_clip_not_used": True,
    }

    participant_hits = np.array([9 / 10, 0 / 1])
    macro_hit, micro_hit = float(participant_hits.mean()), 9 / 11
    assert macro_hit != micro_hit
    per_seed_hits = np.array([[.8, .6, .7], [.1, .2, .3]])
    seed_averaged_macro = float(per_seed_hits.mean(axis=1).mean())
    observed_nll = -math.log(.8)
    requested_uniform_fill_nll = (observed_nll + math.log(3)) / 2
    assert requested_uniform_fill_nll > observed_nll
    records["hit_nll_failure_and_seed_accounting"] = {
        "pass": True, "requested_participant_macro": macro_hit, "micro": micro_hit,
        "seed_averaged_macro": seed_averaged_macro, "independent_participants": 2,
        "seed_count_is_not_independent_participant_count": True,
        "conditional_nll_one_prediction": observed_nll,
        "uniform_abstention_convention_nll_two_requests": requested_uniform_fill_nll,
        "all_abstained_conditional_nll": None,
        "nll_better_cannot_override_hit_loss": mixed_nll < anchor_nll and mixed_hit < anchor_hit,
    }

    # Exact expectation for two indistinguishable methods, each with two dev decisions.
    maxima = []
    for bits in itertools.product([0, 1], repeat=4):
        maxima.append(max(sum(bits[:2]), sum(bits[2:])) / 2)
    expected_selected_dev = float(np.mean(maxima))
    assert expected_selected_dev == .6875 and expected_selected_dev > .5
    records["winners_curse_exact_enumeration"] = {
        "pass": True, "methods": 2, "dev_decisions_per_method": 2,
        "true_accuracy_each_method": .5, "expected_maximum_dev_accuracy": expected_selected_dev,
        "independent_test_expectation": .5, "enumerated_states": 16,
    }
    return records


def audit_phase4(original_root):
    original_root = Path(original_root).resolve()
    directory = original_root / "doc/assets/operator_research_20261004"
    report_file = directory / "phase4_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    result_file = original_root / report["run"] / "result.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    rows = payload["rows"]
    selected = [row for row in rows if row["arm"] == "equal_raw_toeplitz_xdawn_mixture"]
    subjects = sorted({row["subject"] for row in selected})
    assert len(subjects) == 8 and len(selected) == 48
    candidate_ids = tuple(f"row:{r}|column:{c}" for r in range(6) for c in range(6))
    paired = {(row["subject"], row["fold"], row["arm"]): row for row in rows}
    metrics = []
    gains = losses = tied_maxima = 0
    for subject in subjects:
        choices = [row for row in selected if row["subject"] == subject]
        assert len(choices) == 6
        correct, losses_nll = [], []
        for row in choices:
            components = [paired[(subject, row["fold"], arm)] for arm in
                          ("raw_ar8_lambda_1", "toeplitz", "xdawn")]
            probs = np.asarray(row["probabilities"])
            np.testing.assert_allclose(probs, np.mean([v["probabilities"] for v in components], axis=0),
                                       atol=1e-15, rtol=1e-13)
            assert len(probs) == 36 and np.isclose(probs.sum(), 1)
            winners = np.flatnonzero(np.isclose(probs, probs.max(), atol=1e-12, rtol=1e-12))
            tied_maxima += int(len(winners) > 1)
            winner = candidate_ids[int(probs.argmax())]
            hit = winner == row["truth"]
            nll = -math.log(max(float(probs[candidate_ids.index(row["truth"])]),
                                np.finfo(float).tiny))
            assert hit == row["correct"] and winner == row["winner"]
            np.testing.assert_allclose(nll, row["nll"], atol=1e-12)
            correct.append(hit)
            losses_nll.append(nll)
            toeplitz_hit = components[1]["correct"]
            gains += int(hit and not toeplitz_hit)
            losses += int(toeplitz_hit and not hit)
        metrics.append({"participant": subject, "hit": float(np.mean(correct)),
                        "nll": float(np.mean(losses_nll))})
    hit = float(np.mean([row["hit"] for row in metrics]))
    nll = float(np.mean([row["nll"] for row in metrics]))
    np.testing.assert_allclose(hit, report["arms"]["equal_raw_toeplitz_xdawn_mixture"]["hit_mean"])
    np.testing.assert_allclose(nll, report["arms"]["equal_raw_toeplitz_xdawn_mixture"]["nll_mean"])
    entry = "doc/assets/operator_research_20261004/phase4_hybrid_study.py"
    provenance_paths = (
        "phase4_hybrid_study.py", "run_calibration_study.py", "phase4_plan.json",
        "phase4_manifest.json", "continuous_model.py", "selection_erp_model.py",
    )
    provenance = {}
    for name in provenance_paths:
        path = directory / name
        relative = path.relative_to(original_root).as_posix()
        provenance[relative] = {
            "current_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime_source_sha256": payload["source"].get(relative),
        }
    return {
        "pass": True, "result_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
        "rows": len(rows), "mixture_requested_decisions": len(selected),
        "participant_macro_hit": hit, "participant_macro_nll": nll,
        "participant_metrics": metrics, "gained_vs_toeplitz": gains, "lost_vs_toeplitz": losses,
        "probability_ties_at_1e12_tolerance": tied_maxima,
        "raw_component": "raw_ar8_lambda_1",
        "entrypoint_in_runtime_source_map": entry in payload["source"],
        "provenance_files": provenance,
        "mixtures_list_in_saved_plan": payload["plan"]["mixtures"],
        "inference_limit": "prediction arithmetic verified; current entrypoint manifest is not execution proof",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("math_results.json"))
    parser.add_argument("--phase4-root", type=Path)
    args = parser.parse_args()
    for module in (candidate_task, candidate_evidence, decision_loss):
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "src"), module.__file__
    records = verify()
    report = {
        "schema": "n2p3_ensemble_math_verification/1", "status": "passed",
        "scope": "deterministic mathematics and read-only historical prediction audit; no new model fits",
        "isolated_src_first": sys.path[0], "module_files": {
            module.__name__: str(Path(module.__file__).resolve())
            for module in (candidate_task, candidate_evidence, decision_loss)
        },
        "module_sha256": {
            module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (candidate_task, candidate_evidence, decision_loss)
        },
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": records,
    }
    if args.phase4_root is not None:
        report["phase4_read_only_audit"] = audit_phase4(args.phase4_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "checks": len(records),
                      "phase4_audited": "phase4_read_only_audit" in report,
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
