# N2P3-Net v11 blueprint: formal and strict-past routes

> Source of truth for Issue 2. Historical residual-subtraction designs are rejected
> and documented only in `issue2_strict_past_rewrite.md`.
> Route selection and operator-facing boundaries are in `routes.md`.

## 1. Claims and route boundary

The formal model has one enabled evidence mechanism:

1. A PCW-constrained discriminative trial classifier.

A parameter-disjoint class-conditional prequential density belongs only to the
registered strict-past research route. It is disabled in the canonical recipe because
locked outer-fold experiments did not establish complementary decision value.

The route switch is explicit: `lambda_innovation == 0` selects
`neural_ride_v11_pcw_fail_closed`; `lambda_innovation > 0` selects
`neural_ride_v11_strict_past_research`. Other ablations do not silently promote or
replace either route.

There is no residual classifier, global classification bypass, subtraction gate,
residual alpha, residual ramp, or identifiability hold.

The strict-past claim is `preprocessed_epoch_strict_past`. It does not claim that
upstream continuous filtering is online-causal.

## 2. Data partitions

Every outer LOSO training fold is split by complete subjects:

```text
outer train
  +-- optimization subjects: gradients, ERP class means, VAR and covariance profile
  +-- audit subjects: likelihood structure eligibility and variant selection
  +-- validation subjects: epoch selection, calibration and fusion coefficient
outer test subject: final metrics only
```

No learned transform, template, normalizer, covariance parameter, candidate choice,
threshold or coefficient may use the outer test subject.

The audit and validation sets are disjoint. Audit NLL is aggregated once per subject
before candidates are compared.

## 3. Fixed observation coordinates

Likelihood input is the physical sensor epoch after NaN/missing-channel handling and
the configured fixed baseline transform. Learned rereferencing, PCW tokens and ERP
decoder output are excluded.

For trial baseline standardization, the complete pre-stimulus interval is used. The
likelihood score mask must therefore begin at or after 0 ms. Negative-time scoring is
a contract error.

All missing channels are masked in templates, VAR coefficients, covariance factors,
NLL and diagnostics.

## 4. Formal route: PCW classifier

The tokenizer and encoder preserve the physical time axis. The component window
produces `H_N2`, `H_P3a`, `H_P3b`, latency `tau` and width `sigma`.

The production PCW logit is

```text
logit_pcw = head(concat(H_N2, H_P3a, H_P3b))
```

and `logit_target == logit_pcw` inside the neural model. No unrestricted global or
sensor-space feature can enter this head.

The PCW classifier is an epoch-level discriminative model and is not required to be
strict-past. It may use the observed epoch in the ordinary discriminative sense.

## 5. Strict-past route: hypothesis-conditioned likelihood

Optimization subjects define fixed class means

```math
mu_y(t) = E[x(t) | y],  y in {0,1}.
```

For an unlabeled trial both hypotheses are evaluated:

```math
h_y(t) = x(t) - mu_y(t).
```

The model never first chooses a class and never performs label-free ERP subtraction.
It propagates both histories to two conditional densities. For the explicit per-trial
observation set `S`, the active graph is the plug-in joint conditional model

```math
q_y(x_{S,0:T}\mid b,m)=\prod_t \mathcal N\left(
x_{S,t};\mu_{y,S,t}+f_y((x-\mu_y)_{S,<t},m),
D_{y,S,t}+U_{y,S,t}U_{y,S,t}^\top\right).
```

Joint normalization does not require the fixed class template and neural correction
to be optimized simultaneously. This is a frozen-profile plug-in density, not a
claim that a latent physiological ERP and background were identified separately.

### 5.1 Mean baseline

A ridge VAR(`p=32` by default) is fit on optimization subjects. At deployment it
uses positive lags only. A per-trial adaptive AR(1) correction estimates its
coefficient from innovation pairs ending before the predicted sample.

### 5.2 Neural correction

The observation encoder is a five-layer causal depthwise-separable TCN:

- kernel: 9
- dilations: 1, 2, 4, 8, 16
- time-local channel LayerNorm
- default hidden width: 28
- receptive field: `1 + 8(1+2+4+8+16) = 249` samples

Encoder state at `t` contains the sample at `t`. The decoder shifts that state by
one sample internally. Mean, diagonal variance and low-rank factor at `t` are
therefore functions of `x_<t` only. The all-zero state predicts the first sample.

The two hypotheses have separate residual histories. Sharing one history between
class hypotheses is forbidden. Runtime mask and fold-profile mask are intersected
before subtracting `mu_y`; missing channels therefore cannot enter VAR, AR(1), TCN,
covariance or scoring histories. A non-finite value in an observed channel is an
error, not an implicit zero observation.

## 6. Strict-past route: covariance and NLL

The covariance family is

```math
Sigma_t = D_t + U_t U_t^T,  rank(U) in {1,2}.
```

For `M = I + U^T D^-1 U`:

```math
log|Sigma| = sum_j log D_j + log|M|
r^T Sigma^-1 r = r^T D^-1 r
  - (U^T D^-1 r)^T M^-1 (U^T D^-1 r).
```

`M` is solved by Cholesky. No full learned `C x C` SPD matrix or explicit inverse
is constructed. For missing channels the scorer evaluates the exact Gaussian
marginal `D_S + U_S U_S^T`. The empty marginal has log probability zero and hence
zero LLR, but it is invalid as an optimization/audit score.

Training uses a nested composite score:

```text
mean score: fixed-covariance neural mean NLL
covariance score: dynamic diagonal + optional low-rank NLL
loss = normalized positive sum(mean score, covariance score)
```

The scorer exposes two quantities with different contracts:

- `nll_per_observed_scalar`: training and subject-balanced density audit;
- `nll_sum` and `llr_sum`: unnormalized additive evidence over scored samples.

The score-time mask is binary. A time average is never labeled an LLR.

The fixed-covariance term remains present throughout training so a flexible variance
head cannot hide mean error.

Class-dependent power is allowed. Phase-locked class means are represented by
`mu_y`; induced power differences may remain in `D_y/U_y`.

## 7. Strict-past route: audit selection

Every registered candidate is scored on untouched audit subjects. A candidate is
eligible only if all checks pass:

- every audit subject contains both target classes;
- every audit trial uses the fold-supported observation pattern and has at least one
  observed channel;
- finite NLL;
- minimum relative NLL gain over both the static class-template model and the
  candidate's direct nested parent;
- strict-majority wins over both references on audit subjects;
- maximum absolute standardized mean <= 0.25;
- subject-balanced class-conditional mean-waveform RMS <= 0.25;
- subject-balanced target-minus-nontarget complex mean-spectrum RMS <= 0.35;
- standardized covariance error <= 0.50;
- maximum lag-1..10 temporal autocorrelation <= 0.25.

The lowest subject-balanced NLL is recorded as the density winner; exact ties prefer
the simpler declared candidate. If none is eligible, the prequential branch is
disabled. The density winner is not automatically the fusion winner: density fit and
incremental discrimination are different claims.

Only six candidates remain: `m0` (template/diagonal), `linear_ar`, `m1` (TCN mean),
`m2_diag`, `m2_low_rank`, and `m3_low_rank_dynamic`. The final candidate has two
direct parents and must beat both `m2_diag` and `m2_low_rank`; all other candidates
must beat their single direct parent. The redundant parallel FIR and covariance-only
zero/AR combinations are deleted.

Dynamic diagonal and low-rank covariance are optional results of this audit, not
mandatory architecture claims.

The two class-conditional mean checks prevent opposite ERP leakage in the two
classes from cancelling in a pooled mean. They operate on complex mean, not spectral
power, so class-dependent induced power remains permitted in `D_y/U_y`.

## 8. Strict-past route: fusion

For the fixed audit-eligible candidate set:

1. Compute its LLR on validation subjects.
2. Give every validation subject equal loss weight.
3. Leave one validation subject out, apply positive RMS scaling only, and fit one
   non-negative coefficient on the remaining subjects, then score the held-out subject.
4. Require at least four validation subjects, both target classes in every
   predetermined validation subject, lower cross-fitted subject-balanced BCE, and
   improvement on a strict majority of held-out subjects.
5. Among passing candidates, choose the lowest cross-fitted subject-balanced BCE.
6. Only after that gate, refit one coefficient and its positive scale on all validation
   subjects. If no audit-eligible candidate passes, set the coefficient to zero.

```text
logit_final = logit_pcw + temperature * prequential_llr_sum
```

No LLR center or additive offset is fit inside this fusion. Prior/intercept correction
belongs to the separately reported validation-only Platt calibration.

The outer test fold never changes the coefficient or selected structure. In-sample
validation BCE is recorded for reproducibility but is not the fusion claim gate.
After all locked outer folds finish, a read-only claim gate requires at least five
complete folds, active cross-fitted fusion in a strict majority, and strict-majority
plus mean improvement for both AUC and Brier. It may reject a claim but never refit a
model or change a fold prediction.

## 9. Shared interpretation boundary

The learned ERP waveform decoder is outside the conditional-density graph and
disabled in the canonical v11 recipe. Its previous waveform/spectrum NRMSE near 1.0
did not support using it as a training requirement or classification input.

The fold-local analytic class means remain the identifiable ERP representation used
by the likelihood. PCW `tau/sigma` remain available for localization audits.

If the optional ERP decoder is enabled, its reconstruction profile must still be fit
on optimization subjects only. Its output may not be subtracted into a classifier.

## 10. Route-specific training recipes

Canonical single-dataset recipe: `neural_ride_v11_pcw_fail_closed`.
Registered research recipe: `neural_ride_v11_strict_past_research`.

Key defaults:

| parameter | value |
|---|---:|
| PCW width | 64 |
| PCW encoder depth | 3 |
| innovation width | 28 |
| innovation AR order | 32 |
| innovation rank ceiling | 2 |
| lambda_pcw | 0.3 |
| lambda_innovation | 0.0 production / 1.0 research |
| innovation likelihood modules | false production / true research |
| lambda_recon | 0.0 |
| component_decoder | false |
| variance warmup | 5 epochs |
| variance ramp | 10 epochs |
| early-stop patience | 6 |

For research runs, checkpoint selection starts at the first epoch whose covariance
weight is 1.0. PCW-only canonical runs enter joint selection at epoch zero; disabled
variance objectives cannot delay production early stopping. Audit outcomes never
extend the fixed epoch budget.

GTN repetition evidence is a separate deployment-level mechanism. Its calibration
and readiness gates do not alter the strict-past density contract. The rho/reliability
gate decides only whether a fold's chain predictions are formally claim-eligible;
gate-failed folds remain descriptive and stay in the ITT denominator. The
run-level `primary_metric_gate` is non-blocking and records `checks`,
`failed_checks`, and `claim_eligible` instead of deleting the summary.
Primary claim coverage and efficiency-curve coverage use separate configuration
values. Descriptive per-subject chain records are retained independently from
formal unavailable records, and efficiency summaries must identify the same
aggregation and budget semantics as the registered primary metric.

## 11. Required evidence by route

Code acceptance for the formal route requires:

- PCW-only outputs satisfy the formal classifier and repeated-evidence contracts;
- the default formal recipe does not instantiate strict-past likelihood modules;
- `component_decoder=False` in the canonical recipe;
- optimization, validation and test subjects are disjoint;
- the full test suite and static checks pass.

Additional strict-past route acceptance requires current-sample intervention tests,
distinct hypothesis histories, strict-past adaptive AR, dense-Gaussian agreement for
Woodbury NLL, direct-parent fail-closed gates, class-conditional ERP neutrality,
single-class audit exclusion, and disjoint optimization/audit/validation/test subjects.

Empirical acceptance for enabling the research branch requires 5-8 locked outer
folds reporting, per fold:

- selected variant and every candidate diagnostic;
- validation fusion coefficient and validation BCE change;
- PCW, prequential contribution and final metrics;
- paired outer-fold decision changes;
- calibration, coverage and repetition efficiency;
- failures and zero-coefficient folds without omission.

The 60-subject development folds enabled fusion in only 1/5 folds. A subsequent
locked five-fold block enabled it in 0/5 folds, so the outer claim gate failed and the
canonical recipe remains PCW-only. After adding the mandatory direct-parent gate, a
new unseen five-fold block (folds 16-20) produced no eligible density in 5/5 folds and
therefore kept fusion at zero in 5/5 folds. After adding the class-conditional
time/complex-mean neutrality checks, a final unseen block (folds 21-25) again produced
no eligible density and zero fusion in 5/5 folds.

Passing unit tests or one audit fold is not a SOTA result.
