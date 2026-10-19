import N2P3.MatrixNormalCoordinates
import Mathlib.Tactic.FieldSimp

/-!
# Normalizing the actual fixed-covariance Bayesian coefficient kernel

For fixed D,Y,K,Sigma,lambda, complete the square using the already proved
normal equations and integrate the resulting coefficient kernel using the
independently normalized matrix density. This proves a positive, finite
coefficient integral, its explicit value, and the normalized density identity.
No integral value is a premise. The coefficient coordinates are the actual
row-major product coordinates, not a differently ordered abstract vector.

Parameters and exact factor equations are frozen. Numerical QR/Cholesky
refinement, estimation uncertainty, a joint-law disintegration theorem and a
predictive probability law are separate obligations.
-/

namespace N2P3.BayesDensity

open Matrix MeasureTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning N2P3.MatrixNormalCoordinates
open scoped Matrix BigOperators ENNReal ComplexOrder

noncomputable section

variable {n p c : Type*} [Fintype n] [Fintype p] [Fintype c]
  [DecidableEq n] [DecidableEq p] [DecidableEq c]

/-- The actual coefficient-independent factor obtained by completing the square. -/
def dataFactor (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma (constantEnergy D Y K lam))

/-- The existing source joint kernel, now on its explicit integration coordinates.
Y stays fixed here; this is not yet a joint measure over both Y and W. -/
def jointCoordinates (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (x : (p × c) → ℝ) : ℝ :=
  jointKernel D Y K Sigma lam (matricize x)

def kernelNormalizer (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ) : ℝ :=
  dataFactor D Y K Sigma lam / jacobianNormalizer R L

theorem dataFactor_positive (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) : 0 < dataFactor D Y K Sigma lam := Real.exp_pos _

omit [Fintype n] [DecidableEq n] in
theorem jacobianNormalizer_positive (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : IsUnit R.det) (hL : IsUnit L.det) : 0 < jacobianNormalizer R L := by
  unfold jacobianNormalizer
  exact mul_pos (abs_pos.mpr (whiteningMatrix_det_ne_zero R L hR hL))
    (pow_pos (inv_pos.mpr (Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos))) _)

theorem source_jacobianNormalizer_positive
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    0 < jacobianNormalizer R L :=
  jacobianNormalizer_positive R L (normalFactor_unit D K lam R hR hK hlam)
    (whiteningFactor_unit L Sigma hL hSigma)

theorem kernelNormalizer_positive
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    0 < kernelNormalizer D Y K Sigma lam R L :=
  div_pos (dataFactor_positive D Y K Sigma lam)
    (source_jacobianNormalizer_positive D K lam R L Sigma hR hL hK hlam hSigma)

theorem source_pdf_coordinates_as_centered_kernel
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma) (x : (p × c) → ℝ) :
    matrixPDF R L (mean D Y K lam) x = jacobianNormalizer R L *
      centeredKernel D Y K Sigma lam (matricize x) :=
  source_matrixPDF_as_centeredKernel D Y K lam R L Sigma (matricize x) hR hL

/-- The actual W-dependent kernel equals a fully normalized matrix density
times a derived scalar. This is not an assumption about its integral. -/
theorem jointCoordinates_as_normalizer_mul_pdf
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) (x : (p × c) → ℝ) :
    jointCoordinates D Y K Sigma lam x =
      kernelNormalizer D Y K Sigma lam R L * matrixPDF R L (mean D Y K lam) x := by
  have hJ := ne_of_gt (source_jacobianNormalizer_positive D K lam R L Sigma hR hL hK hlam hSigma)
  have hc : centeredKernel D Y K Sigma lam (matricize x) =
      (jacobianNormalizer R L)⁻¹ * matrixPDF R L (mean D Y K lam) x := by
    rw [source_pdf_coordinates_as_centered_kernel D Y K Sigma lam R L hR hL x,
      ← mul_assoc, inv_mul_cancel₀ hJ, one_mul]
  unfold jointCoordinates
  rw [exponential_kernel_factorization D Y K Sigma lam _ hK hlam, hc]
  exact (mul_assoc _ _ _).symm

/-- Integrability is proved separately, before taking a totalized Bochner integral. -/
theorem jointCoordinates_integrable
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    Integrable (jointCoordinates D Y K Sigma lam) := by
  change Integrable (fun x => jointCoordinates D Y K Sigma lam x)
  simp_rw [jointCoordinates_as_normalizer_mul_pdf D Y K Sigma lam R L hR hL hK hlam hSigma]
  exact (matrixPDF_integrable R L (mean D Y K lam)
    (normalFactor_unit D K lam R hR hK hlam) (whiteningFactor_unit L Sigma hL hSigma)).const_mul _

/-- The explicit positive finite coefficient integral is c/J, derived using
the independently proved integral of matrixPDF. -/
theorem jointCoordinates_integral
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    (∫ x, jointCoordinates D Y K Sigma lam x) =
      dataFactor D Y K Sigma lam / jacobianNormalizer R L := by
  simp_rw [jointCoordinates_as_normalizer_mul_pdf D Y K Sigma lam R L hR hL hK hlam hSigma]
  rw [integral_const_mul, matrixPDF_integral_one R L (mean D Y K lam)
    (normalFactor_unit D K lam R hR hK hlam) (whiteningFactor_unit L Sigma hL hSigma), mul_one]
  rfl

theorem jointCoordinates_integral_positive
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    0 < ∫ x, jointCoordinates D Y K Sigma lam x := by
  rw [jointCoordinates_integral D Y K Sigma lam R L hR hL hK hlam hSigma]
  exact kernelNormalizer_positive D Y K Sigma lam R L hR hL hK hlam hSigma

/-- The nonnegative integral is explicitly finite and nonzero as well. -/
theorem jointCoordinates_lintegral
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    (∫⁻ x, ENNReal.ofReal (jointCoordinates D Y K Sigma lam x)) =
      ENNReal.ofReal (dataFactor D Y K Sigma lam / jacobianNormalizer R L) := by
  rw [← ofReal_integral_eq_lintegral_ofReal
    (jointCoordinates_integrable D Y K Sigma lam R L hR hL hK hlam hSigma)
    (Filter.Eventually.of_forall (fun x => (jointKernel_positive D Y K Sigma lam (matricize x)).le)),
    jointCoordinates_integral D Y K Sigma lam R L hR hL hK hlam hSigma]

theorem jointCoordinates_lintegral_positive_finite
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    0 < (∫⁻ x, ENNReal.ofReal (jointCoordinates D Y K Sigma lam x)) ∧
      (∫⁻ x, ENNReal.ofReal (jointCoordinates D Y K Sigma lam x)) < ∞ := by
  rw [jointCoordinates_lintegral D Y K Sigma lam R L hR hL hK hlam hSigma]
  exact ⟨ENNReal.ofReal_pos.mpr (kernelNormalizer_positive D Y K Sigma lam R L hR hL hK hlam hSigma),
    ENNReal.ofReal_lt_top⟩

def normalizedJointPDF (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (x : (p × c) → ℝ) : ℝ :=
  jointCoordinates D Y K Sigma lam x / ∫ z, jointCoordinates D Y K Sigma lam z

/-- Divide by the actual derived integral, rather than the kernel's value at
its mode. The result is exactly the normalized matrix density. -/
theorem normalized_joint_equals_matrixPDF
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) (x : (p × c) → ℝ) :
    normalizedJointPDF D Y K Sigma lam x = matrixPDF R L (mean D Y K lam) x := by
  unfold normalizedJointPDF
  rw [jointCoordinates_as_normalizer_mul_pdf D Y K Sigma lam R L hR hL hK hlam hSigma x,
    jointCoordinates_integral D Y K Sigma lam R L hR hL hK hlam hSigma]
  exact mul_div_cancel_left₀ _ (ne_of_gt
    (kernelNormalizer_positive D Y K Sigma lam R L hR hL hK hlam hSigma))

theorem normalized_joint_integral_one
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    ∫ x, normalizedJointPDF D Y K Sigma lam x = 1 := by
  simp_rw [normalized_joint_equals_matrixPDF D Y K Sigma lam R L hR hL hK hlam hSigma]
  exact matrixPDF_integral_one R L (mean D Y K lam)
    (normalFactor_unit D K lam R hR hK hlam) (whiteningFactor_unit L Sigma hL hSigma)

def normalizedJointMeasure (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) : Measure ((p × c) → ℝ) :=
  volume.withDensity (fun x => ENNReal.ofReal (normalizedJointPDF D Y K Sigma lam x))

theorem normalized_joint_probability
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (normalizedJointMeasure D Y K Sigma lam) := by
  have heq : normalizedJointPDF D Y K Sigma lam = matrixPDF R L (mean D Y K lam) := by
    funext x
    exact normalized_joint_equals_matrixPDF D Y K Sigma lam R L hR hL hK hlam hSigma x
  unfold normalizedJointMeasure
  rw [heq]
  exact source_matrixKernelMeasure_probability D Y K lam R L Sigma hR hL hK hlam hSigma

/-! ## Proper prior and likelihood densities

The observation coordinate type n below denotes ordinary nonredundant model
coordinates (or an explicitly declared identity-extension model), not an
unlabelled ambient density for a singular nuisance projection.
-/

def priorRowFactor (lam : ℝ) : Matrix p p ℝ := Real.sqrt lam • 1

omit [Fintype n] [Fintype c] [DecidableEq n] [DecidableEq c] in
theorem prior_row_factor_gram (lam : ℝ) (hlam : 0 < lam) :
    (priorRowFactor (p := p) lam)ᵀ * priorRowFactor lam = lam • 1 := by
  calc
    _ = (Real.sqrt lam * Real.sqrt lam) • (1 : Matrix p p ℝ) := by
      simp only [priorRowFactor, Matrix.transpose_smul, Matrix.transpose_one,
        Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, smul_smul]
    _ = lam • 1 := by rw [← pow_two, Real.sq_sqrt hlam.le]

omit [Fintype n] [Fintype c] [DecidableEq n] [DecidableEq c] in
theorem prior_row_factor_unit (lam : ℝ) (hlam : 0 < lam) :
    IsUnit (priorRowFactor (p := p) lam).det :=
  positive_definite_det_unit (Matrix.PosDef.one.smul (Real.sqrt_pos.mpr hlam))

def priorDensity (lam : ℝ) (L : Matrix c c ℝ) (x : (p × c) → ℝ) : ℝ :=
  matrixPDF (priorRowFactor lam) L 0 x

theorem prior_density_as_kernel (lam : ℝ) (L Sigma : Matrix c c ℝ)
    (hlam : 0 < lam) (hL : L * Lᵀ = Sigma) (x : (p × c) → ℝ) :
    priorDensity lam L x = jacobianNormalizer (priorRowFactor (p := p) lam) L *
      priorKernel Sigma lam (matricize x) := by
  unfold priorDensity
  change matrixPDF (priorRowFactor lam) L 0 (vectorize (matricize x)) = _
  rw [matrixPDF_as_trace_kernel (priorRowFactor lam) (lam • 1) L Sigma
    0 (matricize x) (prior_row_factor_gram lam hlam) hL]
  unfold priorKernel weightedEnergy
  congr 2
  simp only [sub_zero, Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul,
    Matrix.trace_smul, smul_eq_mul]
  ring

theorem prior_density_integrable (lam : ℝ) (L Sigma : Matrix c c ℝ)
    (hlam : 0 < lam) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    Integrable (priorDensity (p := p) lam L) :=
  matrixPDF_integrable (priorRowFactor lam) L 0 (prior_row_factor_unit lam hlam)
    (whiteningFactor_unit L Sigma hL hSigma)

theorem prior_density_integral_one (lam : ℝ) (L Sigma : Matrix c c ℝ)
    (hlam : 0 < lam) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    ∫ x, priorDensity (p := p) lam L x = 1 :=
  matrixPDF_integral_one (priorRowFactor lam) L 0 (prior_row_factor_unit lam hlam)
    (whiteningFactor_unit L Sigma hL hSigma)

def priorProbabilityMeasure (lam : ℝ) (L : Matrix c c ℝ) : Measure ((p × c) → ℝ) :=
  matrixKernelMeasure (priorRowFactor lam) L 0

theorem prior_probability (lam : ℝ) (L Sigma : Matrix c c ℝ)
    (hlam : 0 < lam) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (priorProbabilityMeasure (p := p) lam L) :=
  matrixKernelMeasure_probability (priorRowFactor lam) L 0 (prior_row_factor_unit lam hlam)
    (whiteningFactor_unit L Sigma hL hSigma)

omit [Fintype p] [Fintype c] [DecidableEq p] [DecidableEq c] in
theorem likelihood_row_factor_gram (H K : Matrix n n ℝ) (hH : H * Hᵀ = K) :
    (H⁻¹)ᵀ * H⁻¹ = K⁻¹ := by
  rw [← hH, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv]

def likelihoodDensity (D : Matrix n p ℝ) (W : Matrix p c ℝ)
    (H : Matrix n n ℝ) (L : Matrix c c ℝ) (y : (n × c) → ℝ) : ℝ :=
  matrixPDF H⁻¹ L (D * W) y

omit [DecidableEq p] in
theorem likelihood_density_as_kernel
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (W : Matrix p c ℝ)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) :
    likelihoodDensity D W H L (vectorize Y) =
      jacobianNormalizer H⁻¹ L * likelihoodKernel D Y K Sigma W :=
  matrixPDF_as_trace_kernel H⁻¹ K⁻¹ L Sigma (D * W) Y (likelihood_row_factor_gram H K hH) hL

omit [DecidableEq p] in
theorem likelihood_density_integrable
    (D : Matrix n p ℝ) (W : Matrix p c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hSigma : Sigma.PosDef) :
    Integrable (likelihoodDensity D W H L) :=
  matrixPDF_integrable H⁻¹ L (D * W)
    (Matrix.isUnit_nonsing_inv_det H (whiteningFactor_unit H K hH hK))
    (whiteningFactor_unit L Sigma hL hSigma)

omit [DecidableEq p] in
theorem likelihood_density_integral_one
    (D : Matrix n p ℝ) (W : Matrix p c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hSigma : Sigma.PosDef) :
    ∫ y, likelihoodDensity D W H L y = 1 :=
  matrixPDF_integral_one H⁻¹ L (D * W)
    (Matrix.isUnit_nonsing_inv_det H (whiteningFactor_unit H K hH hK))
    (whiteningFactor_unit L Sigma hL hSigma)

def likelihoodProbabilityMeasure (D : Matrix n p ℝ) (W : Matrix p c ℝ)
    (H : Matrix n n ℝ) (L : Matrix c c ℝ) : Measure ((n × c) → ℝ) :=
  matrixKernelMeasure H⁻¹ L (D * W)

omit [DecidableEq p] in
theorem likelihood_probability
    (D : Matrix n p ℝ) (W : Matrix p c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (likelihoodProbabilityMeasure D W H L) :=
  matrixKernelMeasure_probability H⁻¹ L (D * W)
    (Matrix.isUnit_nonsing_inv_det H (whiteningFactor_unit H K hH hK))
    (whiteningFactor_unit L Sigma hL hSigma)

/-- Product of the actual independently normalized likelihood and prior,
evaluated at the fixed observed Y. This is integrated over coefficient x. -/
def properJointAtData (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lam : ℝ)
    (H : Matrix n n ℝ) (L : Matrix c c ℝ) (x : (p × c) → ℝ) : ℝ :=
  likelihoodDensity D (matricize x) H L (vectorize Y) * priorDensity lam L x

theorem proper_joint_as_kernel
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma) (x : (p × c) → ℝ) :
    properJointAtData D Y lam H L x =
      (jacobianNormalizer H⁻¹ L * jacobianNormalizer (priorRowFactor (p := p) lam) L) *
        jointCoordinates D Y K Sigma lam x := by
  unfold properJointAtData
  rw [likelihood_density_as_kernel D Y K H L Sigma (matricize x) hH hL,
    prior_density_as_kernel lam L Sigma hlam hL x,
    mul_mul_mul_comm, likelihood_prior_kernel_product]
  rfl

theorem proper_joint_integrable
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ)
    (hR : Rᵀ * R = precision D K lam) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    Integrable (properJointAtData D Y lam H L) := by
  change Integrable (fun x => properJointAtData D Y lam H L x)
  simp_rw [proper_joint_as_kernel D Y K H L Sigma lam hlam hH hL]
  exact (jointCoordinates_integrable D Y K Sigma lam R L hR hL hK hlam hSigma).const_mul _

/-- The actual proper-prior/likelihood evidence integral, with both independent
normalizers retained and the posterior kernel normalizer derived from measure. -/
theorem proper_joint_integral
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ)
    (hR : Rᵀ * R = precision D K lam) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    (∫ x, properJointAtData D Y lam H L x) =
      (jacobianNormalizer H⁻¹ L * jacobianNormalizer (priorRowFactor (p := p) lam) L) *
        (dataFactor D Y K Sigma lam / jacobianNormalizer R L) := by
  simp_rw [proper_joint_as_kernel D Y K H L Sigma lam hlam hH hL]
  rw [integral_const_mul, jointCoordinates_integral D Y K Sigma lam R L hR hL hK hlam hSigma]

theorem proper_joint_integral_positive
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ)
    (hR : Rᵀ * R = precision D K lam) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    0 < ∫ x, properJointAtData D Y lam H L x := by
  rw [proper_joint_integral D Y K H L Sigma lam R hR hH hL hK hlam hSigma]
  exact mul_pos (mul_pos
    (jacobianNormalizer_positive H⁻¹ L
      (Matrix.isUnit_nonsing_inv_det H (whiteningFactor_unit H K hH hK))
      (whiteningFactor_unit L Sigma hL hSigma))
    (jacobianNormalizer_positive (priorRowFactor lam) L (prior_row_factor_unit lam hlam)
      (whiteningFactor_unit L Sigma hL hSigma)))
    (kernelNormalizer_positive D Y K Sigma lam R L hR hL hK hlam hSigma)

def properBayesPDF (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lam : ℝ)
    (H : Matrix n n ℝ) (L : Matrix c c ℝ) (x : (p × c) → ℝ) : ℝ :=
  properJointAtData D Y lam H L x / ∫ z, properJointAtData D Y lam H L z

/-- Bayes normalization of the actual normalized prior times likelihood gives
the same source-derived posterior matrix density; it is not a mode ratio. -/
theorem proper_bayes_equals_matrixPDF
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ)
    (hR : Rᵀ * R = precision D K lam) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) (x : (p × c) → ℝ) :
    properBayesPDF D Y lam H L x = matrixPDF R L (mean D Y K lam) x := by
  have hz := proper_joint_integral_positive D Y K H L Sigma lam R hR hH hL hK hlam hSigma
  rw [proper_joint_integral D Y K H L Sigma lam R hR hH hL hK hlam hSigma] at hz
  unfold properBayesPDF
  rw [proper_joint_as_kernel D Y K H L Sigma lam hlam hH hL x,
    jointCoordinates_as_normalizer_mul_pdf D Y K Sigma lam R L hR hL hK hlam hSigma x,
    proper_joint_integral D Y K H L Sigma lam R hR hH hL hK hlam hSigma,
    ← mul_assoc]
  exact mul_div_cancel_left₀ _ (ne_of_gt hz)

theorem proper_bayes_integral_one
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K H : Matrix n n ℝ)
    (L Sigma : Matrix c c ℝ) (lam : ℝ) (R : Matrix p p ℝ)
    (hR : Rᵀ * R = precision D K lam) (hH : H * Hᵀ = K) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    ∫ x, properBayesPDF D Y lam H L x = 1 := by
  simp_rw [proper_bayes_equals_matrixPDF D Y K H L Sigma lam R hR hH hL hK hlam hSigma]
  exact matrixPDF_integral_one R L (mean D Y K lam)
    (normalFactor_unit D K lam R hR hK hlam) (whiteningFactor_unit L Sigma hL hSigma)

end

end N2P3.BayesDensity
