import N2P3.CandidateJointLaw
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.Tactic.Linarith

/-!
# The accepted source prediction and conditional zero-one error

The actual source picks the first maximum of evidence, except that it abstains
if another candidate lies within its absolute log-evidence tolerance. Derive
the order of its normalized probabilities from the source exp expression.
Compute zero-one error as an actual integral under a probability measure,
then connect it to a candidate posterior measure in the downstream theorem.

Abstention has no declared loss/cost in this source interface. No optimality
claim about the abstention action, real EEG calibration, or IEEE tolerances
is inferred. A near-tie bound concerns the error of choosing a close candidate,
not the error of the source's separate abstention outcome.
-/

namespace N2P3.CandidateDecisionRisk

open MeasureTheory ProbabilityTheory
open N2P3.CandidateProbability N2P3.CandidateTie

noncomputable section

variable {κ : Type*} [Fintype κ]

theorem probability_le_iff_score_le (e : κ → ℝ) (i j : κ) :
    probability e i ≤ probability e j ↔ e i ≤ e j := by
  simp only [probability, Real.exp_le_exp, sub_le_sub_iff_right]

theorem probability_lt_iff_score_lt (e : κ → ℝ) (i j : κ) :
    probability e i < probability e j ↔ e i < e j := by
  simp only [probability, Real.exp_lt_exp, sub_lt_sub_iff_right]

theorem probability_ratio (e : κ → ℝ) (i j : κ) :
    probability e i / probability e j = Real.exp (e i - e j) := by
  rw [probability, probability, ← Real.exp_sub]
  congr 1
  ring

def zeroOneLoss [DecidableEq κ] (prediction truth : κ) : ℝ :=
  if truth = prediction then 0 else 1

def conditionalError [MeasurableSpace κ] [DecidableEq κ] (μ : Measure κ)
    (prediction : κ) : ℝ := ∫ truth, zeroOneLoss prediction truth ∂μ

section MeasureError

variable [MeasurableSpace κ] [MeasurableSingletonClass κ] [DecidableEq κ]

omit [Fintype κ] in
theorem conditionalError_eq_one_sub_mass (μ : Measure κ) [IsProbabilityMeasure μ]
    (prediction : κ) : conditionalError μ prediction = 1 - μ.real {prediction} := by
  have heq : zeroOneLoss prediction = ({prediction}ᶜ : Set κ).indicator (1 : κ → ℝ) := by
    funext truth
    by_cases h : truth = prediction <;> simp [zeroOneLoss, h]
  rw [conditionalError, heq, integral_indicator_one (measurableSet_singleton prediction).compl,
    measureReal_compl (measurableSet_singleton prediction), measureReal_univ_eq_one]

end MeasureError

def scoreError (e : κ → ℝ) (prediction : κ) : ℝ := 1 - probability e prediction

theorem scoreError_minimal_iff_maximum (e : κ → ℝ) (b : κ) :
    (∀ j, scoreError e b ≤ scoreError e j) ↔ IsMaximum e b := by
  constructor
  · intro h j
    have hp : probability e j ≤ probability e b := by
      have he := h j
      unfold scoreError at he
      linarith
    exact (probability_le_iff_score_le e j b).mp hp
  · intro h j
    have hp := (probability_le_iff_score_le e j b).mpr (h j)
    unfold scoreError
    linarith

theorem accepted_winner_minimizes_scoreError [Nonempty κ] [LinearOrder κ]
    (e : κ → ℝ) (b : κ) {tol : ℝ} (htol : 0 ≤ tol)
    (hwin : winner e tol = some b) : ∀ j, scoreError e b ≤ scoreError e j := by
  apply (scoreError_minimal_iff_maximum e b).mpr
  exact ((winner_some_iff_margin e b htol).mp hwin).1

theorem near_maximum_probability_gap [Nonempty κ] (e : κ → ℝ) (b j : κ)
    {tol : ℝ} (htol : 0 ≤ tol) (hb : IsMaximum e b) (hgap : e b - e j ≤ tol) :
    0 ≤ probability e b - probability e j ∧
      probability e b - probability e j ≤ 1 - Real.exp (-tol) := by
  have hporder := (probability_le_iff_score_le e j b).mpr (hb j)
  have hscaled : probability e b * Real.exp (-tol) ≤ probability e j := by
    rw [probability, ← Real.exp_add, probability]
    apply Real.exp_le_exp.mpr
    linarith
  have hexp := Real.exp_le_one_iff.mpr (neg_nonpos.mpr htol)
  have hpbound := probability_le_one e b
  have hmul := mul_le_mul_of_nonneg_right hpbound (sub_nonneg.mpr hexp)
  constructor
  · linarith
  · nlinarith

theorem close_candidate_error_bound [Nonempty κ] (e : κ → ℝ) (b j : κ)
    {tol : ℝ} (htol : 0 ≤ tol) (hb : IsMaximum e b)
    (hj : j ∈ closeCandidates e b tol) :
    0 ≤ scoreError e j - scoreError e b ∧
      scoreError e j - scoreError e b ≤ 1 - Real.exp (-tol) := by
  have h := near_maximum_probability_gap e b j htol hb
    ((close_at_maximum_iff e b j tol hb).mp hj)
  unfold scoreError
  constructor <;> linarith [h.1, h.2]

section ActualPosterior

open Matrix N2P3.MatrixNormalCoordinates
open scoped Matrix

variable {n q m p c nc qc mc : Type*}
  [Fintype n] [Fintype q] [Fintype m] [Fintype p] [Fintype c]
  [Fintype nc] [Fintype qc] [Fintype mc]
  [DecidableEq n] [DecidableEq m] [DecidableEq p] [DecidableEq c]
  [DecidableEq nc] [DecidableEq mc] [LinearOrder p]
  [MeasurableSpace κ] [MeasurableSingletonClass κ] [Nonempty κ] [DecidableEq κ]

variable (M : GaussianCandidates.Model (n := n) (q := q) (m := m) (p := p) (c := c)
  (nc := nc) (qc := qc) (mc := mc) (κ := κ))

/-- Actual zero-one error under the proved conditional law of the same
candidate-observation joint, using the source's uniform candidate prior. -/
def sourcePosteriorError (prediction : κ) : ℝ :=
  conditionalError
    (CandidateJointLaw.posteriorKernel M uniformPrior (vectorize M.Y)) prediction

theorem sourcePosteriorError_eq_scoreError (prediction : κ) :
    sourcePosteriorError M prediction = scoreError M.evidence prediction := by
  haveI := CandidateJointLaw.posteriorKernel_markov M uniformPrior
    (uniformPrior_isPrior (κ := κ))
  rw [sourcePosteriorError, conditionalError_eq_one_sub_mass]
  change 1 - (CandidateJointLaw.posteriorKernel M uniformPrior (vectorize M.Y)
    {prediction}).toReal = 1 - probability M.evidence prediction
  rw [← CandidateJointLaw.source_uniform_probability_eq_posterior_singleton M prediction,
    ENNReal.toReal_ofReal (probability_positive M.evidence prediction).le]

omit [DecidableEq κ] in
theorem source_nll_is_posterior_log_loss (truth : κ) :
    nll M.evidence truth = -Real.log
      ((CandidateJointLaw.posteriorKernel M uniformPrior (vectorize M.Y) {truth}).toReal) := by
  rw [← CandidateJointLaw.source_uniform_probability_eq_posterior_singleton M truth,
    ENNReal.toReal_ofReal (probability_positive M.evidence truth).le]
  exact nll_eq_neg_log_probability M.evidence truth

/-- An accepted candidate minimizes actual conditional classification error.
The source abstention branch has a distinct, unspecified loss. -/
theorem accepted_source_winner_minimizes_posterior_error [LinearOrder κ]
    (b : κ) {tol : ℝ} (htol : 0 ≤ tol)
    (hwin : winner M.evidence tol = some b) :
    ∀ j, sourcePosteriorError M b ≤ sourcePosteriorError M j := by
  intro j
  rw [sourcePosteriorError_eq_scoreError M b, sourcePosteriorError_eq_scoreError M j]
  exact accepted_winner_minimizes_scoreError M.evidence b htol hwin j

theorem close_candidate_posterior_error_bound (b j : κ) {tol : ℝ} (htol : 0 ≤ tol)
    (hb : IsMaximum M.evidence b) (hj : j ∈ closeCandidates M.evidence b tol) :
    0 ≤ sourcePosteriorError M j - sourcePosteriorError M b ∧
      sourcePosteriorError M j - sourcePosteriorError M b ≤ 1 - Real.exp (-tol) := by
  rw [sourcePosteriorError_eq_scoreError M j, sourcePosteriorError_eq_scoreError M b]
  exact close_candidate_error_bound M.evidence b j htol hb hj

end ActualPosterior

def equalScores : Fin 2 → ℝ := fun _ => 0

theorem equalScores_error_half : scoreError equalScores 0 = 1 / 2 := by
  rw [scoreError, probability_eq_exp_div_partition]
  norm_num [equalScores, partition, Fin.sum_univ_succ]

theorem nearTie_errors_differ :
    scoreError nearTieScores 0 ≠ scoreError nearTieScores 1 := by
  have hp : probability nearTieScores 0 < probability nearTieScores 1 := by
    apply (probability_lt_iff_score_lt nearTieScores 0 1).mpr
    norm_num [nearTieScores]
  unfold scoreError
  linarith

end

end N2P3.CandidateDecisionRisk
