import N2P3.BayesDensity
import Mathlib.Probability.Kernel.WithDensity
import Mathlib.Probability.Kernel.Composition.MeasureCompProd

/-!
# The actual fixed-parameter Gaussian joint law and its posterior kernel

The observation and coefficient coordinate spaces are the nonredundant finite
row-major product coordinates. The density is the independently normalized
likelihood times the independently normalized prior from BayesDensity. All
parameters/factors are fixed while observations vary; the posterior row factor
R depends on D,K,lambda, not on Y.

Tonelli proves joint mass one. The observation marginal is derived as the
evidence integral. BayesDensity then identifies a jointly measurable Markov
posterior kernel and proves that composing this marginal with the kernel gives
the same joint measure (hence every measurable rectangle and nonnegative test
function has the disintegration identity). No desired conditional law,
integral value, marginal normalization or independence conclusion is a premise.

This is an exact declared Gaussian model on nonredundant coordinates. It does
not establish the physical held/calibration cross-noise premise, integrate
hyperparameter estimation, or verify NumPy numerical factorizations.
-/

namespace N2P3.GaussianJointLaw

open Matrix MeasureTheory ProbabilityTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning N2P3.MatrixNormalCoordinates N2P3.BayesDensity
open scoped Matrix BigOperators ENNReal ComplexOrder ProbabilityTheory

noncomputable section

variable {n p c : Type*} [Fintype n] [Fintype p] [Fintype c]
  [DecidableEq n] [DecidableEq p] [DecidableEq c]

abbrev Coeff (p c : Type*) := (p × c) → ℝ
abbrev Obs (n c : Type*) := (n × c) → ℝ

section Continuity

variable {Z : Type*} [TopologicalSpace Z]

omit [Fintype p] [Fintype c] [DecidableEq p] [DecidableEq c] in
theorem vectorize_continuous {F : Z → Matrix p c ℝ} (hF : Continuous F) :
    Continuous (fun z => vectorize (F z)) := by
  apply continuous_pi
  intro i
  exact (continuous_apply i.2).comp ((continuous_apply i.1).comp hF)

omit [Fintype p] [Fintype c] [DecidableEq p] [DecidableEq c] in
theorem matricize_continuous {f : Z → Coeff p c} (hf : Continuous f) :
    Continuous (fun z => matricize (f z)) := by
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  exact (continuous_apply (i, j)).comp hf

theorem matrixPDF_continuous_composed
    (R : Matrix p p ℝ) (L : Matrix c c ℝ) {F : Z → Matrix p c ℝ} (hF : Continuous F)
    {f : Z → Coeff p c} (hf : Continuous f) :
    Continuous (fun z => matrixPDF R L (F z) (f z)) := by
  unfold matrixPDF GaussianMeasure.affinePDF
  apply continuous_const.mul
  apply (GaussianMeasure.productPDF_continuous (1 / 2)).comp
  exact (Matrix.toLin' (whiteningMatrix R L)).continuous_on_pi.comp
    (hf.sub (vectorize_continuous hF))

omit [Fintype c] [DecidableEq c] in
theorem mean_continuous_composed
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    {Y : Z → Matrix n c ℝ} (hY : Continuous Y) :
    Continuous (fun z => mean D (Y z) K lam) := by
  unfold mean
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  simp only [Matrix.mul_apply]
  fun_prop

end Continuity

def priorNN (lam : ℝ) (L : Matrix c c ℝ) (x : Coeff p c) : ℝ≥0∞ :=
  ENNReal.ofReal (priorDensity lam L x)

def likelihoodNN (D : Matrix n p ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ)
    (x : Coeff p c) (y : Obs n c) : ℝ≥0∞ :=
  ENNReal.ofReal (likelihoodDensity D (matricize x) H L y)

def jointNN (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ)
    (z : Obs n c × Coeff p c) : ℝ≥0∞ := likelihoodNN D H L z.2 z.1 * priorNN lam L z.2

theorem priorNN_measurable (lam : ℝ) (L : Matrix c c ℝ) :
    Measurable (priorNN (p := p) lam L) :=
  (matrixPDF_continuous_composed (priorRowFactor lam) L continuous_const continuous_id).measurable.ennreal_ofReal

omit [DecidableEq p] in
theorem likelihoodNN_measurable (D : Matrix n p ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    Measurable (fun z : Coeff p c × Obs n c => likelihoodNN D H L z.1 z.2) := by
  have hcont : Continuous (fun z : Coeff p c × Obs n c =>
      likelihoodDensity D (matricize z.1) H L z.2) := by
    unfold likelihoodDensity
    apply matrixPDF_continuous_composed H⁻¹ L _ continuous_snd
    apply continuous_pi
    intro i
    apply continuous_pi
    intro j
    simp only [Matrix.mul_apply, matricize]
    fun_prop
  exact hcont.measurable.ennreal_ofReal

theorem jointNN_measurable (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    Measurable (jointNN D lam H L) :=
  ((likelihoodNN_measurable D H L).comp measurable_swap).mul
    ((priorNN_measurable lam L).comp measurable_snd)

theorem priorNN_mass_one (lam : ℝ) (L Sigma : Matrix c c ℝ)
    (hlam : 0 < lam) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    ∫⁻ x, priorNN (p := p) lam L x = 1 := by
  simp only [priorNN]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (prior_density_integrable lam L Sigma hlam hL hSigma)
    (Filter.Eventually.of_forall (fun x => matrixPDF_nonnegative (priorRowFactor lam) L 0 x)),
    prior_density_integral_one lam L Sigma hlam hL hSigma, ENNReal.ofReal_one]

omit [DecidableEq p] in
theorem likelihoodNN_mass_one
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hSigma : Sigma.PosDef)
    (x : Coeff p c) : ∫⁻ y, likelihoodNN D H L x y = 1 := by
  simp only [likelihoodNN]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (likelihood_density_integrable D (matricize x) K H L Sigma hH hL hK hSigma)
    (Filter.Eventually.of_forall (fun y => matrixPDF_nonnegative H⁻¹ L (D * matricize x) y)),
    likelihood_density_integral_one D (matricize x) K H L Sigma hH hL hK hSigma,
    ENNReal.ofReal_one]

/-- The joint is ordered (observation, coefficient), so its first marginal is
the actual data distribution and its second conditional kernel is the posterior. -/
def jointMeasure (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    Measure (Obs n c × Coeff p c) :=
  ((volume : Measure (Obs n c)).prod (volume : Measure (Coeff p c))).withDensity (jointNN D lam H L)

theorem joint_mass_one
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) : jointMeasure D lam H L Set.univ = 1 := by
  unfold jointMeasure
  rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    lintegral_prod_symm _ (jointNN_measurable D lam H L).aemeasurable]
  simp only [jointNN]
  have hinner (x : Coeff p c) :
      (∫⁻ y, likelihoodNN D H L x y * priorNN lam L x) = priorNN lam L x := by
    rw [lintegral_mul_const' (priorNN lam L x) (fun y => likelihoodNN D H L x y)
        (show priorNN lam L x ≠ ∞ from ENNReal.ofReal_ne_top),
      likelihoodNN_mass_one D K H L Sigma hH hL hK hSigma x, one_mul]
  simp_rw [hinner]
  exact priorNN_mass_one lam L Sigma hlam hL hSigma

theorem joint_probability
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) : IsProbabilityMeasure (jointMeasure D lam H L) :=
  ⟨joint_mass_one D K H L Sigma lam hH hL hK hlam hSigma⟩

/-- Marginal evidence is defined by the actual nonnegative joint integral. -/
def evidenceNN (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ)
    (y : Obs n c) : ℝ≥0∞ := ∫⁻ x, jointNN D lam H L (y, x)

theorem evidenceNN_measurable (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    Measurable (evidenceNN D lam H L) := (jointNN_measurable D lam H L).lintegral_prod_right'

def evidenceMeasure (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    Measure (Obs n c) := volume.withDensity (evidenceNN D lam H L)

theorem joint_rectangle_as_evidence_integral
    (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ)
    (A : Set (Obs n c)) (B : Set (Coeff p c)) (hA : MeasurableSet A) (hB : MeasurableSet B) :
    jointMeasure D lam H L (A ×ˢ B) = ∫⁻ y in A, ∫⁻ x in B, jointNN D lam H L (y, x) := by
  rw [jointMeasure, withDensity_apply _ (hA.prod hB),
    setLIntegral_prod _ (jointNN_measurable D lam H L).aemeasurable.restrict]

theorem observation_marginal_is_evidence
    (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ) :
    (jointMeasure D lam H L).fst = evidenceMeasure D lam H L := by
  ext A hA
  rw [Measure.fst_apply hA, ← Set.prod_univ,
    joint_rectangle_as_evidence_integral D lam H L A Set.univ hA MeasurableSet.univ,
    evidenceMeasure, withDensity_apply _ hA]
  simp only [Measure.restrict_univ, evidenceNN]

theorem evidence_probability
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) : IsProbabilityMeasure (evidenceMeasure D lam H L) := by
  haveI := joint_probability D K H L Sigma lam hH hL hK hlam hSigma
  rw [← observation_marginal_is_evidence D lam H L]
  infer_instance

theorem jointNN_as_properJointAtData
    (D : Matrix n p ℝ) (lam : ℝ) (H : Matrix n n ℝ) (L : Matrix c c ℝ)
    (y : Obs n c) (x : Coeff p c) :
    jointNN D lam H L (y, x) = ENNReal.ofReal (properJointAtData D (matricize y) lam H L x) := by
  unfold jointNN likelihoodNN priorNN properJointAtData
  change ENNReal.ofReal (likelihoodDensity D (matricize x) H L y) *
    ENNReal.ofReal (priorDensity lam L x) =
      ENNReal.ofReal (likelihoodDensity D (matricize x) H L y * priorDensity lam L x)
  exact (ENNReal.ofReal_mul (matrixPDF_nonnegative H⁻¹ L (D * matricize x) y)).symm

/-- Evidence value, positivity and finiteness come from the already derived
proper-prior/likelihood integral; no marginal normalizer is assumed. -/
theorem evidenceNN_as_explicit_normalizer
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (y : Obs n c) :
    evidenceNN D lam H L y = ENNReal.ofReal
      ((jacobianNormalizer H⁻¹ L * jacobianNormalizer (priorRowFactor (p := p) lam) L) *
        (dataFactor D (matricize y) K Sigma lam / jacobianNormalizer R L)) := by
  unfold evidenceNN
  simp_rw [jointNN_as_properJointAtData]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (proper_joint_integrable D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma)
    (Filter.Eventually.of_forall (fun x => mul_nonneg
      (matrixPDF_nonnegative H⁻¹ L (D * matricize x) y)
      (matrixPDF_nonnegative (priorRowFactor lam) L 0 x))),
    proper_joint_integral D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma]

theorem evidenceNN_positive_finite
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (y : Obs n c) :
    0 < evidenceNN D lam H L y ∧ evidenceNN D lam H L y < ∞ := by
  rw [evidenceNN_as_explicit_normalizer D K H L Sigma lam R hR hH hL hK hlam hSigma y]
  have hp := proper_joint_integral_positive D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma
  rw [proper_joint_integral D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma] at hp
  exact ⟨ENNReal.ofReal_pos.mpr hp, ENNReal.ofReal_lt_top⟩

def posteriorNN (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L : Matrix c c ℝ) (y : Obs n c) (x : Coeff p c) : ℝ≥0∞ :=
  ENNReal.ofReal (matrixPDF R L (mean D (matricize y) K lam) x)

theorem posteriorNN_measurable (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L : Matrix c c ℝ) :
    Measurable (Function.uncurry (posteriorNN D K lam R L)) := by
  have hcont : Continuous (fun z : Obs n c × Coeff p c =>
      matrixPDF R L (mean D (matricize z.1) K lam) z.2) :=
    matrixPDF_continuous_composed R L
      (mean_continuous_composed D K lam (matricize_continuous continuous_fst)) continuous_snd
  exact hcont.measurable.ennreal_ofReal

def posteriorKernel (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L : Matrix c c ℝ) : Kernel (Obs n c) (Coeff p c) :=
  (Kernel.const (Obs n c) volume).withDensity (posteriorNN D K lam R L)

theorem posteriorKernel_apply (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L : Matrix c c ℝ) (y : Obs n c) :
    posteriorKernel D K lam R L y = matrixKernelMeasure R L (mean D (matricize y) K lam) := by
  rw [posteriorKernel, Kernel.withDensity_apply _ (posteriorNN_measurable D K lam R L)]
  rfl

theorem posteriorKernel_markov
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    IsMarkovKernel (posteriorKernel D K lam R L) := by
  constructor
  intro y
  rw [posteriorKernel_apply]
  exact source_matrixKernelMeasure_probability D (matricize y) K lam R L Sigma hR hL hK hlam hSigma

/-- The Markov posterior is exactly the normalization of the proper prior
times likelihood, using its actual coefficient integral at the supplied Y. -/
theorem posteriorKernel_as_normalized_Bayes_density
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (y : Obs n c) :
    posteriorKernel D K lam R L y = volume.withDensity
      (fun x => ENNReal.ofReal (properBayesPDF D (matricize y) lam H L x)) := by
  simp_rw [proper_bayes_equals_matrixPDF D (matricize y) K H L Sigma lam R
    hR hH hL hK hlam hSigma]
  exact posteriorKernel_apply D K lam R L y

theorem jointNN_as_evidence_mul_posterior
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (y : Obs n c) (x : Coeff p c) :
    jointNN D lam H L (y, x) = evidenceNN D lam H L y * posteriorNN D K lam R L y x := by
  rw [jointNN_as_properJointAtData]
  have hproper := proper_bayes_equals_matrixPDF D (matricize y) K H L Sigma lam R
    hR hH hL hK hlam hSigma x
  unfold properBayesPDF at hproper
  have hpos := proper_joint_integral_positive D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma
  have hfactor : properJointAtData D (matricize y) lam H L x =
      (∫ z, properJointAtData D (matricize y) lam H L z) *
        matrixPDF R L (mean D (matricize y) K lam) x := by
    have h := (div_eq_iff (ne_of_gt hpos)).mp hproper
    simpa only [mul_comm] using h
  rw [hfactor, ENNReal.ofReal_mul hpos.le,
    proper_joint_integral D (matricize y) K H L Sigma lam R hR hH hL hK hlam hSigma,
    evidenceNN_as_explicit_normalizer D K H L Sigma lam R hR hH hL hK hlam hSigma y]
  rfl

theorem posterior_rectangle_disintegration
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (A : Set (Obs n c)) (B : Set (Coeff p c))
    (hA : MeasurableSet A) (hB : MeasurableSet B) :
    jointMeasure D lam H L (A ×ˢ B) = ∫⁻ y in A, posteriorKernel D K lam R L y B
      ∂evidenceMeasure D lam H L := by
  rw [joint_rectangle_as_evidence_integral D lam H L A B hA hB,
    evidenceMeasure, setLIntegral_withDensity_eq_setLIntegral_mul _
      (evidenceNN_measurable D lam H L) (Kernel.measurable_coe _ hB) hA]
  apply lintegral_congr
  intro y
  simp only [Pi.mul_apply]
  rw [posteriorKernel, Kernel.withDensity_apply' _ (posteriorNN_measurable D K lam R L)]
  simp_rw [jointNN_as_evidence_mul_posterior D K H L Sigma lam R hR hH hL hK hlam hSigma]
  exact lintegral_const_mul _ ((posteriorNN_measurable D K lam R L).of_uncurry_left (x := y))

/-- An actual probability-kernel disintegration of the same joint, not a
separate family of normalized densities without a joint-law identification. -/
theorem joint_equals_marginal_compProd_posterior
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) :
    jointMeasure D lam H L = (jointMeasure D lam H L).fst ⊗ₘ posteriorKernel D K lam R L := by
  haveI := joint_probability D K H L Sigma lam hH hL hK hlam hSigma
  haveI := posteriorKernel_markov D K L Sigma lam R hR hL hK hlam hSigma
  rw [observation_marginal_is_evidence]
  haveI := evidence_probability D K H L Sigma lam hH hL hK hlam hSigma
  apply Measure.ext_prod
  intro A B hA hB
  rw [Measure.compProd_apply_prod hA hB]
  exact posterior_rectangle_disintegration D K H L Sigma lam R hR hH hL hK hlam hSigma A B hA hB

theorem posterior_nonnegative_test_disintegration
    (D : Matrix n p ℝ) (K H : Matrix n n ℝ) (L Sigma : Matrix c c ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (hK : K.PosDef) (hlam : 0 < lam)
    (hSigma : Sigma.PosDef) (f : Obs n c × Coeff p c → ℝ≥0∞) (hf : Measurable f) :
    (∫⁻ z, f z ∂jointMeasure D lam H L) =
      ∫⁻ y, ∫⁻ x, f (y, x) ∂posteriorKernel D K lam R L y ∂(jointMeasure D lam H L).fst := by
  haveI := joint_probability D K H L Sigma lam hH hL hK hlam hSigma
  haveI := posteriorKernel_markov D K L Sigma lam R hR hL hK hlam hSigma
  calc
    _ = ∫⁻ z, f z ∂((jointMeasure D lam H L).fst ⊗ₘ posteriorKernel D K lam R L) := by
      congr 1
      exact joint_equals_marginal_compProd_posterior D K H L Sigma lam R hR hH hL hK hlam hSigma
    _ = _ := Measure.lintegral_compProd hf

end

end N2P3.GaussianJointLaw
