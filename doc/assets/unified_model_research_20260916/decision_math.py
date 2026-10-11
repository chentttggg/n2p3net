from __future__ import annotations

import json
import math
from pathlib import Path

import torch

F = torch.nn.functional
torch.set_default_dtype(torch.float64)
torch.manual_seed(16)
checks = []


def note(name, **kw):
    checks.append({"check": name, "status": "PASS", **kw})


def matrix_rowcol(nr, nc, reps):
    return torch.tensor(
        [
            [
                int((s < nr and r == s) or (s >= nr and c == s - nr))
                for r in range(nr)
                for c in range(nc)
            ]
            for _ in range(reps)
            for s in range(nr + nc)
        ],
        dtype=torch.float64,
    )


for name, A in [("direct9", torch.eye(9).repeat(3, 1)), ("rowcol2x3", matrix_rowcol(2, 3, 3))]:
    e = torch.randn(len(A), requires_grad=True)
    logprior = torch.full((A.shape[1],), -math.log(A.shape[1]))
    p1 = e.sigmoid()
    p0 = 1 - p1
    joint = (A * p1[:, None] + (1 - A) * p0[:, None]).prod(0) * logprior.exp()
    q = joint / joint.sum()
    qmat = (logprior + A.T @ e).softmax(0)
    torch.testing.assert_close(q, qmat, rtol=1e-12, atol=1e-12)
    y = 1
    loss = F.cross_entropy((A.T @ e)[None], torch.tensor([y]))
    actual = torch.autograd.grad(loss, e)[0]
    expected = A @ (qmat - F.one_hot(torch.tensor(y), A.shape[1]))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    col = torch.randperm(A.shape[1])
    row = torch.randperm(len(A))
    torch.testing.assert_close((A[:, col].T @ e).softmax(0), qmat[col], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(A[row].T @ e[row], A.T @ e, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((A.T @ (e + 2.8)).softmax(0), qmat, rtol=1e-12, atol=1e-12)
    note(name + "_bayes_gradient_permutations_equal_coverage")
A = matrix_rowcol(2, 3, 3)
e = torch.randn(len(A), requires_grad=True)
flash = e.reshape(3, 5).sum(0)
full = F.cross_entropy((A.T @ e)[None], torch.tensor([4]))
separate = F.cross_entropy(flash[:2][None], torch.tensor([1])) + F.cross_entropy(
    flash[2:][None], torch.tensor([1])
)
torch.testing.assert_close(full, separate, rtol=1e-12, atol=1e-12)
gf = torch.autograd.grad(full, e, retain_graph=True)[0]
gs = torch.autograd.grad(separate, e)[0]
torch.testing.assert_close(gf, gs, rtol=1e-12, atol=1e-12)
note("row_column_joint_nll_equals_two_axes", nll=full.item())
A = torch.eye(9).repeat(5, 1)
e = torch.linspace(-0.2, 0.8, 45)
s = A.T @ e
mean = s / 5
assert s.argmax() == mean.argmax()
assert not torch.allclose(s.softmax(0), mean.softmax(0))
note(
    "sum_mean_same_argmax_different_probability",
    sum_max_probability=s.softmax(0).max().item(),
    mean_max_probability=mean.softmax(0).max().item(),
)
A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
f = torch.tensor([0.2, -0.2, -0.2])
offset = math.log((1 / 9) / (8 / 9))
uncorrected = A.T @ f
corrected = A.T @ (f - offset)
assert uncorrected.argmax().item() == 0 and corrected.argmax().item() == 1
note(
    "unequal_coverage_prior_offset_changes_winner",
    raw_scores=uncorrected.tolist(),
    llr_scores=corrected.tolist(),
)
s = torch.tensor([1.0, 0.0])
assert (2 * s).softmax(0).max() > s.softmax(0).max()
note(
    "duplicate_evidence_artificially_increases_confidence",
    once=s.softmax(0).max().item(),
    twice=(2 * s).softmax(0).max().item(),
)
# Only a flash of candidate zero was rejected: selection itself carries target information.
rejection_likelihood = torch.tensor([0.8, 0.1])
posterior = rejection_likelihood / rejection_likelihood.sum()
assert not torch.allclose(posterior, torch.tensor([0.5, 0.5]))
note(
    "missing_not_at_random_has_nonzero_evidence",
    posterior_from_missingness=posterior.tolist(),
    neutral_mask_posterior=[0.5, 0.5],
)
losses = torch.tensor([0.1, 5.0])
penalty = math.log(9)
full = losses.mean()
rejected = (losses[0] + penalty) / 2
assert rejected < full
note(
    "fixed_failure_penalty_alone_does_not_prevent_learned_reject_escape",
    all_included=full.item(),
    learned_drop_hard_case=rejected.item(),
)
result = {"torch": torch.__version__, "checks": checks, "count": len(checks)}
serialized = json.dumps(result, indent=2) + "\n"
Path(__file__).with_suffix(".json").write_text(serialized, encoding="utf-8")
print(serialized, end="")
