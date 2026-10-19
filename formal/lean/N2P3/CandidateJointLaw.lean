import N2P3.GaussianCandidates
import Mathlib.MeasureTheory.Integral.Lebesgue.Countable

/-!
# A genuine finite-candidate / observation joint and its posterior kernel

The candidate observations are the already normalized retained Gaussian laws,
with the actual posterior-predictive covariance factors. The candidate prior is
separate from the calibration coefficient prior and may assign zero mass.
No density integral, marginal identity, Markov property or disintegration
identity is an input hypothesis. We derive these from the actual law and the
explicit finite nonnegative prior with total mass one.

The joint is ordered (observation, candidate). The conditional kernel is a
globally measurable version: every Gaussian density is strictly positive in
its retained-coordinate formula, including the chosen extension away from the
observation support. The joint only determines this version almost everywhere
under its observation marginal. Statistical and numerical scopes are inherited
from GaussianCandidates.Model; no IEEE or physical EEG claim is added here.
-/

namespace N2P3.CandidateJointLaw

open Matrix MeasureTheory ProbabilityTheory
open N2P3.MatrixNormalCoordinates N2P3.RetainedMeasure N2P3.RetainedMatrixMeasure
open scoped Matrix BigOperators ENNReal ComplexOrder ProbabilityTheory

noncomputable section

abbrev Obs (n c : Type*) := (n × c) → ℝ

variable {n q m p c nc qc mc κ : Type*}
  [Fintype n] [Fintype q] [Fintype m] [Fintype p] [Fintype c]
  [Fintype nc] [Fintype qc] [Fintype mc]
  [DecidableEq n] [DecidableEq m] [DecidableEq p] [DecidableEq c]
  [DecidableEq nc] [DecidableEq mc] [LinearOrder p]

variable (M : GaussianCandidates.Model (n := n) (q := q) (m := m) (p := p) (c := c)
  (nc := nc) (qc := qc) (mc := mc) (κ := κ))

def baseMeasure : Measure (Obs n c) := coordinateVolume (channelLift M.B)

def candidatePDF (i : κ) (y : Obs n c) : ℝ :=
  RetainedGaussianLaw.candidatePDF M.B (M.observationFactor i) M.L
    (M.D i * M.sourceCenter) y

def candidateNN (i : κ) (y : Obs n c) : ℝ≥0∞ := ENNReal.ofReal (candidatePDF M i y)

def candidateMeasure (i : κ) : Measure (Obs n c) :=
  RetainedGaussianLaw.candidateLaw M.B (M.observationFactor i) M.L
    (M.D i * M.sourceCenter)

theorem candidatePDF_continuous (i : κ) : Continuous (candidatePDF M i) := by
  exact (RetainedGaussianLaw.retainedPDF_continuous M.B (M.observationFactor i) M.L).comp
    (continuous_id.sub continuous_const)

theorem candidatePDF_positive (i : κ) (y : Obs n c) : 0 < candidatePDF M i y :=
  RetainedGaussianLaw.retainedPDF_positive M.B (M.D i * M.C) (M.observationFactor i)
    M.L M.Sigma (M.observationFactor_cov i) M.hL M.hSigma _

theorem candidateNN_measurable (i : κ) : Measurable (candidateNN M i) :=
  (candidatePDF_continuous M i).measurable.ennreal_ofReal

theorem candidate_measure_probability (i : κ) : IsProbabilityMeasure (candidateMeasure M i) :=
  RetainedGaussianLaw.candidateLaw_probability M.B (M.D i * M.C) (M.observationFactor i)
    M.L M.Sigma (M.D i * M.sourceCenter) (M.observationFactor_cov i) M.hL M.hSigma

theorem candidate_measure_withDensity (i : κ) :
    candidateMeasure M i = (baseMeasure M).withDensity (candidateNN M i) :=
  RetainedGaussianLaw.candidateLaw_withDensity M.B M.hB (M.observationFactor i) M.L _

/-- This is a proved property of the actual retained law, not a density premise. -/
theorem candidateNN_mass_one (i : κ) : ∫⁻ y, candidateNN M i y ∂baseMeasure M = 1 := by
  haveI := candidate_measure_probability M i
  have h := measure_univ (μ := candidateMeasure M i)
  rw [candidate_measure_withDensity M i, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ] at h
  exact h

theorem candidatePDF_at_source (i : κ) : candidatePDF M i (vectorize M.Y) = M.density i :=
  (M.density_eq_candidatePDF i).symm

section FiniteCandidates

variable [Fintype κ]

def jointNN (prior : κ → ℝ) (z : Obs n c × κ) : ℝ≥0∞ :=
  ENNReal.ofReal (prior z.2) * candidateNN M z.2 z.1

def mixtureNN (prior : κ → ℝ) (y : Obs n c) : ℝ≥0∞ := ∑ i, jointNN M prior (y, i)

theorem mixtureNN_measurable (prior : κ → ℝ) : Measurable (mixtureNN M prior) := by
  apply Finset.measurable_sum
  intro i _
  change Measurable (fun y => ENNReal.ofReal (prior i) * candidateNN M i y)
  exact measurable_const.mul (candidateNN_measurable M i)

theorem mixtureNN_ne_top (prior : κ → ℝ) (y : Obs n c) : mixtureNN M prior y ≠ ∞ := by
  apply ENNReal.sum_ne_top.mpr
  intro i _
  exact ENNReal.mul_ne_top ENNReal.ofReal_ne_top ENNReal.ofReal_ne_top

theorem mixtureNN_positive (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior)
    (y : Obs n c) : 0 < mixtureNN M prior y := by
  obtain ⟨i, _, hi⟩ := (Finset.sum_pos_iff_of_nonneg (fun j _ => hprior.1 j)).mp
    (show (0 : ℝ) < ∑ j, prior j by rw [hprior.2]; norm_num)
  have hterm : 0 < jointNN M prior (y, i) :=
    ENNReal.mul_pos (ENNReal.ofReal_pos.mpr hi).ne'
      (ENNReal.ofReal_pos.mpr (candidatePDF_positive M i y)).ne'
  exact hterm.trans_le (Finset.single_le_sum
    (fun j _ => show (0 : ℝ≥0∞) ≤ jointNN M prior (y, j) from bot_le) (Finset.mem_univ i))

theorem mixtureNN_eq_ofReal (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior)
    (y : Obs n c) :
    mixtureNN M prior y = ENNReal.ofReal (∑ i, prior i * candidatePDF M i y) := by
  rw [ENNReal.ofReal_sum_of_nonneg (fun i _ =>
    mul_nonneg (hprior.1 i) (candidatePDF_positive M i y).le)]
  simp only [mixtureNN, jointNN, candidateNN, ENNReal.ofReal_mul (hprior.1 _)]

theorem mixtureNN_mass_one (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    ∫⁻ y, mixtureNN M prior y ∂baseMeasure M = 1 := by
  change (∫⁻ y, ∑ i, jointNN M prior (y, i) ∂baseMeasure M) = 1
  rw [lintegral_finset_sum]
  · simp only [jointNN]
    simp_rw [lintegral_const_mul _ (candidateNN_measurable M _), candidateNN_mass_one, mul_one]
    rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => hprior.1 i), hprior.2,
      ENNReal.ofReal_one]
  · intro i _
    change Measurable (fun y => ENNReal.ofReal (prior i) * candidateNN M i y)
    exact measurable_const.mul (candidateNN_measurable M i)

variable [MeasurableSpace κ] [MeasurableSingletonClass κ]

theorem jointNN_measurable (prior : κ → ℝ) : Measurable (jointNN M prior) := by
  apply measurable_from_prod_countable_left
  intro i
  change Measurable (fun y => ENNReal.ofReal (prior i) * candidateNN M i y)
  exact measurable_const.mul (candidateNN_measurable M i)

def jointMeasure (prior : κ → ℝ) : Measure (Obs n c × κ) :=
  ((baseMeasure M).prod (Measure.count : Measure κ)).withDensity (jointNN M prior)

def observationMeasure (prior : κ → ℝ) : Measure (Obs n c) :=
  (baseMeasure M).withDensity (mixtureNN M prior)

theorem joint_mass_one (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    jointMeasure M prior Set.univ = 1 := by
  rw [jointMeasure, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    lintegral_prod _ (jointNN_measurable M prior).aemeasurable]
  simp_rw [lintegral_fintype, Measure.count_singleton, mul_one]
  exact mixtureNN_mass_one M prior hprior

theorem joint_probability (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    IsProbabilityMeasure (jointMeasure M prior) := ⟨joint_mass_one M prior hprior⟩

omit [MeasurableSpace κ] [MeasurableSingletonClass κ] in
theorem observation_probability (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    IsProbabilityMeasure (observationMeasure M prior) := by
  constructor
  rw [observationMeasure, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ]
  exact mixtureNN_mass_one M prior hprior

theorem joint_rectangle (prior : κ → ℝ) (A : Set (Obs n c)) (S : Set κ)
    (hA : MeasurableSet A) (hS : MeasurableSet S) :
    jointMeasure M prior (A ×ˢ S) =
      ∫⁻ y in A, ∫⁻ i in S, jointNN M prior (y, i) ∂Measure.count ∂baseMeasure M := by
  haveI : SFinite (baseMeasure M) := by unfold baseMeasure coordinateVolume; infer_instance
  rw [jointMeasure, withDensity_apply _ (hA.prod hS),
    setLIntegral_prod _ (jointNN_measurable M prior).aemeasurable.restrict]

/-- The joint is tied to the actual candidate observation laws before any
posterior kernel is constructed. -/
theorem joint_candidate_slice (prior : κ → ℝ) (i : κ) (A : Set (Obs n c))
    (hA : MeasurableSet A) :
    jointMeasure M prior (A ×ˢ ({i} : Set κ)) =
      ENNReal.ofReal (prior i) * candidateMeasure M i A := by
  rw [joint_rectangle M prior A {i} hA (measurableSet_singleton i)]
  simp only [lintegral_singleton, Measure.count_singleton, mul_one, jointNN]
  rw [lintegral_const_mul _ (candidateNN_measurable M i), candidate_measure_withDensity M i,
    withDensity_apply _ hA]

theorem candidate_marginal_singleton (prior : κ → ℝ) (i : κ) :
    (jointMeasure M prior).snd {i} = ENNReal.ofReal (prior i) := by
  haveI := candidate_measure_probability M i
  rw [Measure.snd_apply (measurableSet_singleton i), ← Set.univ_prod,
    joint_candidate_slice M prior i Set.univ MeasurableSet.univ, measure_univ, mul_one]

theorem observation_marginal (prior : κ → ℝ) :
    (jointMeasure M prior).fst = observationMeasure M prior := by
  ext A hA
  rw [Measure.fst_apply hA, ← Set.prod_univ, joint_rectangle M prior A Set.univ hA MeasurableSet.univ,
    observationMeasure, withDensity_apply _ hA]
  simp only [Measure.restrict_univ, lintegral_fintype, Measure.count_singleton, mul_one, mixtureNN]

def posteriorNN (prior : κ → ℝ) (y : Obs n c) (i : κ) : ℝ≥0∞ :=
  jointNN M prior (y, i) / mixtureNN M prior y

theorem posteriorNN_measurable (prior : κ → ℝ) :
    Measurable (Function.uncurry (posteriorNN M prior)) := by
  apply measurable_from_prod_countable_left
  intro i
  change Measurable (fun y => ENNReal.ofReal (prior i) * candidateNN M i y / mixtureNN M prior y)
  exact (measurable_const.mul (candidateNN_measurable M i)).div (mixtureNN_measurable M prior)

omit [MeasurableSpace κ] [MeasurableSingletonClass κ] in
theorem posteriorNN_sum_one (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior)
    (y : Obs n c) : ∑ i, posteriorNN M prior y i = 1 := by
  simp only [posteriorNN, div_eq_mul_inv]
  rw [← Finset.sum_mul, ← mixtureNN,
    ENNReal.mul_inv_cancel (mixtureNN_positive M prior hprior y).ne'
      (mixtureNN_ne_top M prior y)]

def posteriorKernel (prior : κ → ℝ) : Kernel (Obs n c) κ :=
  (Kernel.const (Obs n c) (Measure.count : Measure κ)).withDensity (posteriorNN M prior)

theorem posteriorKernel_apply (prior : κ → ℝ) (y : Obs n c) :
    posteriorKernel M prior y = Measure.count.withDensity (posteriorNN M prior y) := by
  rw [posteriorKernel, Kernel.withDensity_apply _ (posteriorNN_measurable M prior)]
  rfl

theorem posteriorKernel_markov (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    IsMarkovKernel (posteriorKernel M prior) := by
  constructor
  intro y
  constructor
  rw [posteriorKernel_apply M prior y, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ]
  simp only [lintegral_fintype, Measure.count_singleton, mul_one]
  exact posteriorNN_sum_one M prior hprior y

theorem posteriorKernel_singleton (prior : κ → ℝ) (y : Obs n c) (i : κ) :
    posteriorKernel M prior y {i} = posteriorNN M prior y i := by
  rw [posteriorKernel_apply M prior y, withDensity_apply _ (measurableSet_singleton i)]
  simp

theorem zero_prior_posterior_singleton (prior : κ → ℝ) (y : Obs n c) (i : κ)
    (hi : prior i = 0) : posteriorKernel M prior y {i} = 0 := by
  rw [posteriorKernel_singleton]
  simp [posteriorNN, jointNN, hi]

omit [MeasurableSpace κ] [MeasurableSingletonClass κ] in
theorem jointNN_eq_marginal_mul_posterior (prior : κ → ℝ)
    (hprior : CandidateProbability.IsPrior prior) (y : Obs n c) (i : κ) :
    jointNN M prior (y, i) = mixtureNN M prior y * posteriorNN M prior y i := by
  rw [posteriorNN, ENNReal.mul_div_cancel (mixtureNN_positive M prior hprior y).ne'
    (mixtureNN_ne_top M prior y)]

theorem posterior_rectangle_disintegration (prior : κ → ℝ)
    (hprior : CandidateProbability.IsPrior prior) (A : Set (Obs n c)) (S : Set κ)
    (hA : MeasurableSet A) (hS : MeasurableSet S) :
    jointMeasure M prior (A ×ˢ S) = ∫⁻ y in A, posteriorKernel M prior y S
      ∂observationMeasure M prior := by
  rw [joint_rectangle M prior A S hA hS, observationMeasure,
    setLIntegral_withDensity_eq_setLIntegral_mul _ (mixtureNN_measurable M prior)
      (Kernel.measurable_coe _ hS) hA]
  apply lintegral_congr
  intro y
  simp only [Pi.mul_apply]
  rw [posteriorKernel_apply M prior y, withDensity_apply _ hS]
  simp_rw [jointNN_eq_marginal_mul_posterior M prior hprior]
  exact lintegral_const_mul _ ((posteriorNN_measurable M prior).of_uncurry_left (x := y))

/-- This is the conditional-kernel disintegration equation for this same joint.
Together with the separately proved Markov property it identifies a genuine
posterior, rather than merely assigning that name to normalized weights. -/
theorem joint_equals_marginal_compProd_posterior (prior : κ → ℝ)
    (hprior : CandidateProbability.IsPrior prior) :
    jointMeasure M prior = (jointMeasure M prior).fst ⊗ₘ posteriorKernel M prior := by
  haveI := joint_probability M prior hprior
  haveI := posteriorKernel_markov M prior hprior
  rw [observation_marginal M prior]
  haveI := observation_probability M prior hprior
  apply Measure.ext_prod
  intro A S hA hS
  rw [Measure.compProd_apply_prod hA hS]
  exact posterior_rectangle_disintegration M prior hprior A S hA hS

theorem posterior_nonnegative_test_disintegration (prior : κ → ℝ)
    (hprior : CandidateProbability.IsPrior prior) (f : Obs n c × κ → ℝ≥0∞)
    (hf : Measurable f) :
    (∫⁻ z, f z ∂jointMeasure M prior) =
      ∫⁻ y, ∫⁻ i, f (y, i) ∂posteriorKernel M prior y ∂(jointMeasure M prior).fst := by
  haveI := joint_probability M prior hprior
  haveI := posteriorKernel_markov M prior hprior
  calc
    _ = ∫⁻ z, f z ∂((jointMeasure M prior).fst ⊗ₘ posteriorKernel M prior) := by
      congr 1
      exact joint_equals_marginal_compProd_posterior M prior hprior
    _ = _ := Measure.lintegral_compProd hf

omit [MeasurableSpace κ] [MeasurableSingletonClass κ] in
theorem posteriorNN_eq_ofReal_ratio (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior)
    (y : Obs n c) (i : κ) :
    posteriorNN M prior y i =
      ENNReal.ofReal (prior i * candidatePDF M i y / ∑ j, prior j * candidatePDF M j y) := by
  have hsum : (0 : ℝ) < ∑ j, prior j * candidatePDF M j y := by
    apply ENNReal.ofReal_pos.mp
    rw [← mixtureNN_eq_ofReal M prior hprior y]
    exact mixtureNN_positive M prior hprior y
  rw [posteriorNN, mixtureNN_eq_ofReal M prior hprior y, jointNN, candidateNN,
    ← ENNReal.ofReal_mul (hprior.1 i), ENNReal.ofReal_div_of_pos hsum]

theorem source_explicit_prior_eq_posterior_singleton (prior : κ → ℝ)
    (hprior : CandidateProbability.IsPrior prior) (i : κ) :
    ENNReal.ofReal (CandidateProbability.posterior prior M.evidence i) =
      posteriorKernel M prior (vectorize M.Y) {i} := by
  rw [posteriorKernel_singleton, posteriorNN_eq_ofReal_ratio M prior hprior]
  simp_rw [candidatePDF_at_source]
  rw [M.explicit_candidate_prior prior i]

theorem source_uniform_probability_eq_posterior_singleton [Nonempty κ] (i : κ) :
    ENNReal.ofReal (CandidateProbability.probability M.evidence i) =
      posteriorKernel M CandidateProbability.uniformPrior (vectorize M.Y) {i} := by
  rw [← CandidateProbability.uniform_posterior_eq_probability M.evidence i]
  exact source_explicit_prior_eq_posterior_singleton M CandidateProbability.uniformPrior
    CandidateProbability.uniformPrior_isPrior i

end FiniteCandidates

end

end N2P3.CandidateJointLaw
