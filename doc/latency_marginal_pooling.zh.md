# N2P3-Net: MS-EEGNet Trunk With Latency-Marginal Contrast Pooling

## Scope and Evidence Boundary

The production N2P3-Net candidate is now a lightweight MS-EEGNet-style
spatio-temporal trunk followed by latency-marginal contrast pooling (LMBC).
It is not the earlier four-scale, fully mixed, dilated-TCN architecture.

The trunk is derived from Borra, Fantozzi, and Magosso (2021), "A Lightweight
Multi-Scale Convolutional Neural Network for P300 Decoding," DOI
10.3389/fnhum.2021.655840. The paper's baseline has an EEGNet-style
spatio-temporal (ST) block, two compressed multi-scale temporal (MST) branches,
and a flatten-plus-FC head. This implementation reproduces the published
baseline parameter counts for its reported input geometries before replacing
only the final aggregation with LMBC.

This document establishes an architectural and mathematical contract. It does
not claim that the new candidate has improved held-out performance. Earlier
GTN/BI2014a results produced by the retired TCN trunk are not evidence for this
architecture and must not be compared or reported as its results.

## What Is Fixed and What Learns

| Quantity | Status | Rationale |
|---|---|---|
| ST temporal filter coefficients | learned | EEG spectra and morphology vary by corpus and subject. |
| Spatial projection within each temporal filter | learned, grouped and max-norm bounded | A P300 topography is conditional on temporal/spectral content. |
| MST short/long filter coefficients | learned | The data decide the waveform summary within fixed short and long scales. |
| Two branch readout weights | learned by the final linear classifier | The model can weight short and long evidence without a high-dimensional gate. |
| Latency posterior within each scale | learned | Trial-to-trial latency is latent, but must remain in a plausible interval. |
| P300/reference windows and latency bank | fixed and recorded | These are physiological/protocol constraints, not labels to overfit. |
| Number of scales, compressed width, and depth | fixed by default | The reference paper found no stable benefit from extra branches, width, or MST depth. |
| Subject IDs, test-fold QC thresholds, unrestricted attention | never learned | They create leakage or allow alignment to arbitrary artifacts. |

## Mathematical Contract

Let `X in R^(C x T)` be one standardized EEG epoch. The ST block learns eight
temporal filters and two spatial projections per temporal filter:

```text
U_f(c, t) = (h_f * X_c)(t),                         f = 1,...,8
V_(f,d)(t) = ELU(BN(sum_c a_(f,d,c) U_f(c, t))),    d = 1,2
```

`a_(f,d,:)` is a spatial depthwise filter whose effective L2 norm is bounded by
one. Average pooling by four gives 16 feature maps at approximately one quarter
of the input sampling frequency.

The MST block has two depthwise-separable branches `s in {short, long}`:

```text
H_(s,k)(t) = ELU(BN(sum_(f,d) p_(s,k,f,d)
                    (g_s * V_(f,d))(t))),          k = 1,2
g_short: kernel 5   (about 150 ms after ST pooling)
g_long:  kernel 17  (about 500 ms after ST pooling)
```

Each branch is deliberately compressed from 16 input maps to two output maps.
The paper's original `ms_flatten` head applies a further average pool by eight,
flattens both branches, and learns a binary FC classifier. It remains available
as an explicit paper-style ablation.

The default LMBC head instead preserves the ST-pooled physical time coordinate.
For each branch feature `H_(s,k)`, a fixed reference `R=[-200,0)` ms and latent
P300 candidates `W_delta=[250,600)+delta` ms are used:

```text
c_(s,delta,k) = mean_(t in W_delta) H_(s,k)(t) - mean_(t in R) H_(s,k)(t)
alpha_(s,delta) = softmax_delta(tanh(q_s)^T c_(s,delta) / (theta sqrt(2)))
z_(s,k) = sum_delta alpha_(s,delta) c_(s,delta,k)
logits = A [z_short ; z_long] + b
```

`q_s`, temporal filters, spatial filters, MST filters, and `A` are learned only
from the training fold. The reference interval, P300 interval, and finite
latency offsets are visible configuration. The final linear layer learns the
relative importance of the two scales.

## Counterexamples and Failure Boundaries

1. **Global-mean position loss.** An early impulse and a late impulse with the
   same mean produce the same global average. `ms_flatten` produces different
   pooled coordinates, and LMBC only admits the impulse when it lies in a
   declared P300 candidate. This is covered by a tensor counterexample test.
2. **Common feature offset.** If `H'(t)=H(t)+b`, then every LMBC contrast is
   unchanged: `c'_(s,delta,k)=c_(s,delta,k)`. This is exact at the encoded
   feature level.
3. **Out-of-window peak.** A large peak outside both `R` and every `W_delta`
   has zero LMBC weight. Global averaging changes; LMBC does not.
4. **In-window artifact.** LMBC cannot distinguish an artifact that occurs
   inside a valid P300 candidate. Fold-local QC and held-out evaluation remain
   necessary.
5. **Out-of-bank latency.** A genuine response beyond the candidate union can
   be missed. Expanding the bank is an explicit, training-fold-tested ablation,
   never an unconstrained attention fallback.
6. **Cross-scale overfitting.** A fully connected spatial mixer can represent
   arbitrary correlations between 32 raw branch maps. The grouped ST projection
   intentionally forbids this before the low-dimensional MST recombination.

## Code Chain

```text
EpochDataset.preprocessing(sfreq, tmin_ms, n_times)
  -> train.factory.build_binary_model(..., tmin_s)
  -> N2P3NetBaseline
  -> N2P3Net ST block -> two MST branches
  -> ms_flatten | global_average | LMBC head
  -> DeepBaseline weighted CE, subject-disjoint validation, calibration
  -> decision.py calibrated candidate aggregation when metadata exist
```

The manifest records `trunk=ms_eegnet_style`, ST/MST dimensions, pooling mode,
and the LMBC physical windows. `global_average` is the matched head ablation;
`ms_flatten` is the paper-style MS-EEGNet head; `latency_marginal_contrast` is
the production candidate.

## Verification Requirements

1. Parameter tests reproduce the paper baseline counts: 1,154 for `C=8,T=140`
   and 1,210 for `C=12,T=113` using `ms_flatten`.
2. Unit tests verify grouped spatial filters, effective max norm, physical
   reference fail-closed behavior, latency selection, and the global-mean
   counterexample.
3. Rerun MS-EEGNet (`ms_flatten`), N2P3 global average, N2P3 LMBC, and EEGNet
   under identical grouped folds, QC, epochs, seed, and calibration.
4. Promote LMBC only if it improves the complete held-out Pareto comparison;
   prior results from the retired TCN trunk are archived historical evidence.
