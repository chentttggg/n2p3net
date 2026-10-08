"""Exact finite-distribution check of the source risk; no model or Torch import."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OWNER = Path(r"C:\Users\chent\.codex\worktrees\calibration-g-prior\n2p3-net\src\baselines\deep.py")

# P_du(x,y), one library and two equally weighted people with heterogeneous priors.
PEOPLE = (
    {("a", 1): F(72, 100), ("a", 0): F(4, 100), ("b", 1): F(8, 100), ("b", 0): F(16, 100)},
    {("a", 1): F(4, 100), ("a", 0): F(16, 100), ("b", 1): F(16, 100), ("b", 0): F(64, 100)},
)


def mixture(beta):
    z = [sum(F(beta if y else 1) * value for (x, y), value in person.items()) for person in PEOPLE]
    # Removing the global class tilt leaves a/Z person weights, rather than a.
    unnormalized = {(x, y): sum(person[(x, y)] / normalizer for person, normalizer in zip(PEOPLE, z)) / 2
                    for x in ("a", "b") for y in (0, 1)}
    total = sum(unnormalized.values())
    corrected = {key: value / total for key, value in unnormalized.items()}
    weighted = {key: value * (beta if key[1] else 1) for key, value in unnormalized.items()}
    weighted_total = sum(weighted.values())
    assert weighted_total == 1
    return z, corrected, weighted


def posterior(distribution, x):
    return distribution[(x, 1)] / (distribution[(x, 0)] + distribution[(x, 1)])


def expected_nll(prior, prediction):
    return -prior * math.log(prediction) - (1 - prior) * math.log1p(-prediction)


def main():
    proper = {key: sum(person[key] for person in PEOPLE) / 2 for key in PEOPLE[0]}
    z1, beta1, _ = mixture(1)
    z8, corrected8, weighted8 = mixture(8)
    assert z1 == [F(1), F(1)] and beta1 == proper
    assert posterior(corrected8, "a") != posterior(proper, "a")
    q = posterior(weighted8, "a")
    logit_q = math.log(float(q / (1 - q)))
    corrected_probability = 1 / (1 + math.exp(-(logit_q - math.log(8))))
    assert math.isclose(corrected_probability, float(posterior(corrected8, "a")), abs_tol=1e-15)
    assert not math.isclose(corrected_probability, float(posterior(proper, "a")), abs_tol=1e-5)
    # A valid class LLR is not a proper posterior logit without its prior.
    prior = .2
    posterior_logit = math.log(prior / (1 - prior))
    llr = posterior_logit - math.log(prior / (1 - prior))
    wrong_posterior = 1 / (1 + math.exp(-llr))
    assert wrong_posterior == .5
    nll_optimal, nll_llr_as_posterior = expected_nll(prior, prior), expected_nll(prior, wrong_posterior)
    brier_optimal = prior * (1 - prior)
    brier_llr_as_posterior = prior * (1 - wrong_posterior) ** 2 + (1 - prior) * wrong_posterior ** 2
    assert nll_llr_as_posterior > nll_optimal and brier_llr_as_posterior > brier_optimal
    receipt = {"schema": "source_risk_beta_counterexample/1", "status": "passed_exact_distribution_checks",
        "source_owner": str(OWNER), "source_owner_sha256": hashlib.sha256(OWNER.read_bytes()).hexdigest(),
        "formula": "r_i = alpha_d/N_d * beta**y_i/(n_du * (1+(beta-1)*pi_du))",
        "population_limit": "Unrestricted binary CE minimizer under the declared empirical hierarchical measure; finite NN need not attain it.",
        "person_priors": ["4/5", "1/5"], "beta8_Z": list(map(str, z8)),
        "proper_equal_person_posterior_at_a": str(posterior(proper, "a")),
        "beta8_minus_log_beta_posterior_at_a": str(posterior(corrected8, "a")),
        "beta1_recovers_equal_person_measure": beta1 == proper,
        "llr_is_not_binary_posterior_without_prior": {"prior": prior, "llr": llr,
            "sigmoid_llr": wrong_posterior, "optimal_NLL": nll_optimal, "LLR_as_posterior_NLL": nll_llr_as_posterior,
            "optimal_Brier": brier_optimal, "LLR_as_posterior_Brier": brier_llr_as_posterior},
        "torch_imported": False, "model_fit": False}
    (HERE / "beta_counterexample.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
