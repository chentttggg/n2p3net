# Issue 2 strict-past rewrite: research and decision record

> This is the research/decision record, not the operator-facing route guide. Use
> [`routes.md`](routes.md) to choose between the canonical PCW route and the
> opt-in `neural_ride_v11_strict_past_research` route. The strict-past branch is
> fail-closed and is not a production default.

Date: 2026-08-25

## Conclusion

The failed residual classifier was not primarily under-parameterized. It asked an
unidentified question: subtract a class-specific ERP from a single unlabeled trial,
then classify the remainder. At deployment the class is unknown, and a label-free
single-trial ERP estimate cannot in general decide which class template should be
removed. A more expressive decoder can lower training reconstruction error without
resolving that circular dependence.

The mathematically valid research replacement is hypothesis-conditioned prequential
classification:

1. Estimate fixed target and non-target sensor templates on optimization subjects.
2. For each trial, form two histories, one under each class hypothesis.
3. Predict every moment at time `t` only from samples before `t`.
4. Score both conditional densities and take their log-likelihood ratio.
5. Retain the likelihood contribution only after untouched audit subjects pass
   proper-score gains over both `m0` and the direct nested parent, strict-majority
   subject wins, pooled and class-conditional time/complex-mean neutrality,
   identity-covariance and temporal-whiteness checks.
6. Fit one non-negative fusion coefficient on a separate validation-subject split.
7. Use the outer test subject only once for final evaluation.

The tightened scorer treats missingness and evidence magnitude as part of the
probability contract. Runtime and profile masks are intersected before hypothesis
centering, the exact observed-channel Gaussian marginal is scored, and all-missing
trials return zero LLR but no valid training/audit score. Optimization uses NLL per
observed scalar; reported prequential evidence is the uncentered sum over time.

The fixed class template plus strict-past correction already defines a normalized
plug-in joint conditional density. The learned morphology ERP decoder is not part of
that graph and will not be reattached to manufacture an end-to-end latent
ERP/background interpretation.

This integrates ERP and background dynamics without requiring a preliminary class
decision. However, validity is not utility: in 60-subject experiments fusion was
enabled in only 1/5 development folds and 0/5 subsequent locked folds. The canonical
production result is therefore PCW-only, while the strict-past system remains an
explicit research recipe. The PCW classifier may use the complete epoch; strict-past
is a probability contract for the conditional density, not a universal restriction
on every discriminative representation.

## Why the original system was introduced

The original motivation was reasonable: preserve phase-locked ERP evidence in an
interpretable component path while allowing induced or non-phase-locked activity to
provide complementary evidence. The implementation failed because it turned that
conceptual decomposition into a learned single-trial subtraction before the class
was known. Its identifiability gate then became a training controller, adding a large
state machine around a premise that the data did not support.

Real-data smoke results exposed the failure rather than a mere optimization problem:
ERP waveform and spectrum NRMSE stayed around 1.0 and a residual template probe stayed
around 0.86-0.87. The correct behavior was `alpha_residual=0`. Increasing decoder or
TCN capacity would only make the same unidentified factorization easier to fit.

## Strict-past scope

The implemented claim is `preprocessed_epoch_strict_past`, not raw online EEG:

- moments at scored post-stimulus sample `t` depend only on baseline-standardized
  samples before `t`;
- both hypotheses have separate histories;
- the TCN state is shifted internally by one sample;
- the fold-fixed VAR and trial-adaptive AR(1) use positive lags only;
- class templates, covariance baselines and all normalizers are fit only on inner
  optimization subjects;
- scoring before stimulus onset is rejected because trial baseline normalization
  uses the complete pre-stimulus segment;
- continuous filtering or resampling upstream may be acausal, so this is not yet a
  raw-streaming latency guarantee.

Applying strict-past to the PCW classifier would answer a different question. A
discriminative epoch classifier is allowed to use `x_t` when producing an epoch-level
label. Applying it to the ERP decoder is also unnecessary because that decoder is an
offline interpretation. The invalid operation was using `x_t` to parameterize the
density assigned to that same `x_t`.

## Research architecture retained

- PCW-only trial classifier. There is no global or residual classification bypass.
- Fixed-coordinate class templates and fold-local VAR(32).
- Five-layer causal depthwise-separable TCN, kernel 9 and dilations 1/2/4/8/16.
- One-sample decoder shift, giving strict-past moments with a 249-sample history.
- Strict-past online AR(1) adaptation per trial.
- Diagonal-plus-low-rank covariance with rank 1 or 2 and Woodbury/Cholesky scoring.
- Nested mean score plus covariance score, so variance cannot hide mean error.
- Subject-balanced audit selection with mandatory direct-parent improvement and
  fail-closed fusion. An all-zero learned correction cannot inherit its parent's
  predictive gain.
- Six registered density candidates only. The final dynamic low-rank candidate must
  improve over both its dynamic-diagonal and static-low-rank direct parents. The
  redundant learned FIR and covariance-only zero/AR combinations were removed.
- Subject-balanced class-conditional waveform and complex-spectrum mean checks.
  Opposite ERP leakage cannot cancel in a pooled mean, while class-dependent power
  remains allowed.
- Audit subjects must contain both classes. A single-class subject remains available
  to optimization but cannot be selected for audit; if any predetermined validation
  subject lacks one class, fusion fails closed instead of dropping that subject.

Neither the likelihood graph nor the learned offline ERP waveform decoder is enabled
by the canonical recipe. Both remain explicit opt-in research modules. The ERP decoder
is never a classification input. In the strict-past research recipe, identifiable
fold-local ERP class means enter the likelihood hypotheses directly, avoiding decoder
underfit and preventing a failed reconstruction loss from pulling the PCW backbone
away from the classification objective.

The fail-closed default also applies to the raw `N2P3Net()` constructor, so callers
cannot accidentally bypass the recipe and instantiate either research graph. Under
the registered three-channel GTN contract, canonical and strict-past research models
contain 40,132 and 50,280 parameters respectively; production removes 10,148
parameters (20.2%). New GTN runs compute and serialize the exact count from their
effective `model_kwargs` rather than relying on stale preset labels.

## Literature synthesis

1. Dawid (1984), [The Prequential Approach](https://doi.org/10.2307/2981683).
   Sequential predictive statements must be assessed using information available
   before the outcome. This is the core reason for shifting all density heads.
2. Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and
   Estimation](https://doi.org/10.1198/016214506000001437). NLL is meaningful only
   when the predictive distribution cannot condition on its realized outcome.
3. van den Oord et al. (2016), [WaveNet](https://arxiv.org/abs/1609.03499).
   Autoregressive sample likelihoods condition on preceding samples, not the current
   sample. Causal convolution alone is insufficient unless the output alignment is
   also shifted.
4. Bai, Kolter and Koltun (2018), [An Empirical Evaluation of Generic Convolutional
   and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271).
   Dilated residual TCNs are a defensible finite-history sequence model; depth is not
   evidence of identifiability.
5. Salinas et al. (2020), [DeepAR](https://doi.org/10.1016/j.ijforecast.2019.07.001).
   Neural autoregressive density models work best when their probabilistic semantics
   and scaling are explicit. The current model starts from a strong linear baseline
   instead of relearning physical scale.
6. Blankertz et al. (2011), [Single-trial analysis and classification of ERP
   components](https://doi.org/10.1016/j.neuroimage.2010.06.048). ERP decoding needs
   supervised spatial-temporal structure; the averaged ERP is not a directly observed
   single-trial latent waveform.
7. Rivet et al. (2009), [xDAWN algorithm to enhance evoked
   potentials](https://doi.org/10.1109/TBME.2009.2012869). Class-locked evoked
   structure can be estimated reliably at the training-set level, supporting fixed
   fold-local templates rather than per-trial blind subtraction.
8. Kappenman et al. (2021), [ERP CORE](https://doi.org/10.1016/j.neuroimage.2020.117465).
   ERP component measurement depends on explicit preprocessing, time windows and
   scoring conventions. The physical time axis is therefore a checked model input.
9. Polich (2007), [Updating P300](https://doi.org/10.1016/j.clinph.2007.04.019), and
   Donchin and Coles (1988), [context updating](https://doi.org/10.1017/S0140525X00058027).
   P3a/P3b latency and morphology are variable and task-dependent; fixed analytic
   templates are hypotheses, not single-trial ground truth.
10. Kriegeskorte et al. (2009), [Circular analysis in systems
    neuroscience](https://doi.org/10.1038/nn.2303). Reusing labels to define a feature
    and then validate it on the same subjects creates circular evidence.
11. Varma and Simon (2006), [Bias in error estimation when using cross-validation for
    model selection](https://doi.org/10.1186/1471-2105-7-91), and Cawley and Talbot
    (2010), [On Over-fitting in Model Selection](https://www.jmlr.org/papers/v11/cawley10a.html).
    Structure selection, coefficient calibration and final evaluation need distinct
    subject-level partitions.
12. Ledoit and Wolf (2004), [A well-conditioned estimator for large-dimensional
    covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4). Full sample
    covariance is unstable in limited data. A diagonal baseline plus audited rank-1/2
    correction is a conservative alternative.
13. Lawhern et al. (2018), [EEGNet](https://doi.org/10.1088/1741-2552/aace8c), and
    Schirrmeister et al. (2017), [Deep ConvNets for EEG
    decoding](https://doi.org/10.1002/hbm.23730). Compact depthwise/separable EEG
    models are strong comparators; architecture size alone does not establish a new
    mechanism or SOTA result.

## Evidence and remaining risks

Five non-overlapping outer-fold blocks were used during this rewrite:

1. `issue2_v11_60subj_5fold_24ep_seed0_crossfit`, folds 1-5. The density audit passed
   5/5. Neural mean NLL improved about 8.5-9.5% over the best static mean baseline,
   but validation-subject cross-fit enabled fusion in only 1/5 folds. Its outer AUC
   change was negative, so the claim failed.
2. `issue2_v11_60subj_folds06to10_24ep_seed0_candidate_crossfit`, locked folds 6-10.
   The density audit passed 3/5; the other two folds had no eligible family and
   correctly fell back to `m0`. In all three passing folds, every audit-eligible
   candidate failed validation-subject cross-fit. Fusion was therefore active in
   0/5 and final equaled PCW exactly. The read-only outer claim gate failed. PCW
   `sum@3` was 3/5, all-trial sum was 4/5 and single-trial AUC was 0.743; these are
   five-subject diagnostics, not SOTA evidence.
3. `issue2_v11_canonical_pcw_60subj_folds11to15_24ep_seed0`, locked folds 11-15.
   This exercised the final PCW-only recipe with no likelihood modules. Every epoch
   was in the joint PCW selection phase; `sum@3` was 3/5, all-trial sum was 4/5,
   balanced accuracy was 0.585 and AUC was 0.674.
4. `issue2_v11_strict_parent_gate_60subj_folds16to20_24ep_seed0`, unseen folds
   16-20 after the nested-parent audit fix. Every candidate had to improve over
   both `m0` and its direct parent on a strict majority of audit subjects. No
   density family was eligible in 5/5 folds, so fusion remained zero and final
   equaled PCW in every fold. PCW `sum@3` was 4/5, all-trial sum was 5/5,
   balanced accuracy was 0.709 and AUC was 0.804. These are again five-subject
   diagnostics, not a SOTA result.
5. `issue2_v11_final_strict_gate_60subj_folds21to25_24ep_seed0`, unseen folds
   21-25 under the final direct-parent plus class-conditional complex-mean gate.
   No density family satisfied every necessary check in 5/5 folds. Fusion remained
   zero and final equaled PCW in every fold; the outer claim gate failed. PCW
   `sum@3` and all-trial sum were both 5/5, balanced accuracy was 0.752 and AUC was
   0.849. These are five outer subjects and cannot support a SOTA claim.

The blocks contain different outer subjects, so their headline metrics are not a
paired architecture comparison. They establish protocol behavior and the absence of
validated complementary likelihood evidence. They do not establish SOTA.

Remaining risks are explicit:

- Four audit subjects are enough for fail-closed development but still produce noisy
  structure selection. Any future re-enablement requires a newly locked outer block,
  not reuse of the folds above.
- The standardized class-conditional RMS margins (0.25 time-domain and 0.35 complex
  mean-spectrum) are conservative engineering equivalence margins, not independently
  calibrated physiological bounds. They may reject a useful research density, but
  they cannot enable one; future relaxation requires frozen negative-control or
  simulation calibration before examining another outer block.
- Gaussian innovations are a working score, not a claim that EEG artifacts are
  Gaussian. Heavy-tailed likelihoods remain a future registered comparison.
- Fixed class templates may drift by subject. They may be adapted only with unlabeled,
  class-symmetric rules; test labels must never update them.
- Candidate selection still compares several nested families. Each family now has a
  declared direct parent and must improve over both that parent and `m0`, but reports
  must still include every candidate diagnostic rather than only the winner.
- A positive prequential gate does not prove added decision value. Fusion must first
  improve subject-balanced leave-one-validation-subject-out BCE on a strict majority
  of at least four validation subjects. The resulting coefficient must be non-zero
  and outer-fold paired metrics must improve.
- No SOTA claim is permitted from the development-exposed GTN cohort. A new-subject
  or untouched external confirmatory cohort must beat registered EEGNet/xDAWN/
  Riemannian comparators under the frozen complete-ITT, multi-seed protocol with
  paired uncertainty and coverage reporting.
