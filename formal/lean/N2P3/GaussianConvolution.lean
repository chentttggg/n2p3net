import N2P3.BayesDensity
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.Tactic.Abel

/-!
# Integrating an independent matrix-normal observation against a general prior

All covariance parameters are fixed. The prior row precision A and observation
row covariance K are positive definite, with a shared positive definite spatial
covariance Sigma. Exact factors are solver specifications, not density or
integral hypotheses. The prior and conditional observation densities are the
independently normalized row-major densities from MatrixNormalCoordinates.

The product likelihood encodes the independent observation-noise generative
model. Cross-block C=0 in a real caller is a separate source premise. This file
does not infer independence from disjoint sample scopes or from zero covariance.
-/

namespace N2P3.GaussianConvolution

open Matrix MeasureTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning N2P3.MatrixNormalCoordinates
open N2P3.BayesDensity
open scoped BigOperators Matrix ComplexOrder ENNReal Kronecker

noncomputable section

variable {n p c : Type*} [Fintype n] [Fintype p] [Fintype c]
  [DecidableEq n] [DecidableEq p] [DecidableEq c]

def updatedPrecision (D : Matrix n p ℝ) (A : Matrix p p ℝ) (K : Matrix n n ℝ) :
    Matrix p p ℝ := A + Dᵀ * K⁻¹ * D

def predictiveCovariance (D : Matrix n p ℝ) (A : Matrix p p ℝ) (K : Matrix n n ℝ) :
    Matrix n n ℝ := K + D * A⁻¹ * Dᵀ

omit [Fintype c] [DecidableEq p] [DecidableEq c] in
theorem updatedPrecision_posDef (D : Matrix n p ℝ) (A : Matrix p p ℝ)
    (K : Matrix n n ℝ) (hA : A.PosDef) (hK : K.PosDef) :
    (updatedPrecision D A K).PosDef := by
  apply hA.add_posSemidef
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    hK.inv.posSemidef.conjTranspose_mul_mul_same D

omit [Fintype c] [DecidableEq n] [DecidableEq c] in
theorem predictiveCovariance_posDef (D : Matrix n p ℝ) (A : Matrix p p ℝ)
    (K : Matrix n n ℝ) (hA : A.PosDef) (hK : K.PosDef) :
    (predictiveCovariance D A K).PosDef := by
  apply hK.add_posSemidef
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    hA.inv.posSemidef.mul_mul_conjTranspose_same D

omit [Fintype c] [DecidableEq c] in
theorem predictive_inverse (D : Matrix n p ℝ) (A : Matrix p p ℝ)
    (K : Matrix n n ℝ) (hA : A.PosDef) (hK : K.PosDef) :
    (predictiveCovariance D A K)⁻¹ =
      K⁻¹ - K⁻¹ * D * (updatedPrecision D A K)⁻¹ * Dᵀ * K⁻¹ := by
  have hu : IsUnit (A⁻¹⁻¹ + Dᵀ * K⁻¹ * D) := by
    rw [Matrix.nonsing_inv_nonsing_inv A (positive_definite_det_unit hA)]
    exact (updatedPrecision_posDef D A K hA hK).isUnit
  simpa only [predictiveCovariance, updatedPrecision,
    Matrix.nonsing_inv_nonsing_inv A (positive_definite_det_unit hA)] using
    Matrix.add_mul_mul_inv_eq_sub K D A⁻¹ Dᵀ hK.isUnit hA.inv.isUnit hu

omit [Fintype c] [DecidableEq c] in
/-- The determinant identity follows from Sylvester's theorem and actual
invertibility of A and K. -/
theorem predictive_determinant (D : Matrix n p ℝ) (A : Matrix p p ℝ)
    (K : Matrix n n ℝ) (hA : A.PosDef) (hK : K.PosDef) :
    (predictiveCovariance D A K).det * A.det =
      K.det * (updatedPrecision D A K).det := by
  have hid : A * (1 + A⁻¹ * Dᵀ * K⁻¹ * D) = updatedPrecision D A K := by
    simp only [Matrix.mul_add, Matrix.mul_one, updatedPrecision]
    rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, ← Matrix.mul_assoc,
      Matrix.mul_nonsing_inv A (positive_definite_det_unit hA), Matrix.one_mul]
  have hd := congrArg Matrix.det hid
  rw [Matrix.det_mul] at hd
  have ht : (predictiveCovariance D A K).det =
      K.det * (1 + A⁻¹ * Dᵀ * K⁻¹ * D).det := by
    simpa only [predictiveCovariance, Matrix.mul_assoc] using
      Matrix.det_add_mul D (A⁻¹ * Dᵀ) (positive_definite_det_unit hK)
  rw [ht, ← hd]
  ring

omit [Fintype n] [Fintype c] [DecidableEq n] [DecidableEq c] in
theorem precision_factor_unit (R A : Matrix p p ℝ) (hR : Rᵀ * R = A)
    (hA : A.PosDef) : IsUnit R.det := by
  have hp : IsUnit (R.det * R.det) := by
    simpa only [← hR, Matrix.det_mul, Matrix.det_transpose] using
      positive_definite_det_unit hA
  exact isUnit_of_mul_isUnit_left hp

/-- Fixed exact factors for arbitrary SPD prior and observation covariances.
No integral or predictive-law statement occurs among these premises. -/
structure Model where
  D : Matrix n p ℝ
  A : Matrix p p ℝ
  K : Matrix n n ℝ
  Sigma : Matrix c c ℝ
  mean : Matrix p c ℝ
  R : Matrix p p ℝ
  B : Matrix p p ℝ
  H : Matrix n n ℝ
  G : Matrix n n ℝ
  L : Matrix c c ℝ
  hA : A.PosDef
  hK : K.PosDef
  hSigma : Sigma.PosDef
  hR : Rᵀ * R = A
  hB : Bᵀ * B = updatedPrecision D A K
  hH : H * Hᵀ = K
  hG : G * Gᵀ = predictiveCovariance D A K
  hL : L * Lᵀ = Sigma

namespace Model

variable (M : Model (n := n) (p := p) (c := c))

def residual (Y : Matrix n c ℝ) : Matrix n c ℝ := Y - M.D * M.mean

def shift (Y : Matrix n c ℝ) : Matrix p c ℝ :=
  (updatedPrecision M.D M.A M.K)⁻¹ * M.Dᵀ * M.K⁻¹ * M.residual Y

def coefficientCenter (Y : Matrix n c ℝ) : Matrix p c ℝ := M.mean + M.shift Y

def fullEnergy (Y : Matrix n c ℝ) (W : Matrix p c ℝ) : Matrix c c ℝ :=
  (Y - M.D * W)ᵀ * M.K⁻¹ * (Y - M.D * W) +
    (W - M.mean)ᵀ * M.A * (W - M.mean)

def coefficientEnergy (Y : Matrix n c ℝ) (W : Matrix p c ℝ) : Matrix c c ℝ :=
  (W - M.coefficientCenter Y)ᵀ * updatedPrecision M.D M.A M.K *
    (W - M.coefficientCenter Y)

def predictiveEnergy (Y : Matrix n c ℝ) : Matrix c c ℝ :=
  (M.residual Y)ᵀ * (predictiveCovariance M.D M.A M.K)⁻¹ * M.residual Y

omit [DecidableEq c] in
theorem shift_normal_equation (Y : Matrix n c ℝ) :
    updatedPrecision M.D M.A M.K * M.shift Y = M.Dᵀ * M.K⁻¹ * M.residual Y := by
  simpa only [shift, Matrix.mul_assoc] using
    Matrix.mul_nonsing_inv_cancel_left (updatedPrecision M.D M.A M.K)
      (M.Dᵀ * M.K⁻¹ * M.residual Y)
      (positive_definite_det_unit (updatedPrecision_posDef _ _ _ M.hA M.hK))

omit [DecidableEq c] in
theorem shift_transpose_normal_equation (Y : Matrix n c ℝ) :
    (M.shift Y)ᵀ * updatedPrecision M.D M.A M.K = (M.residual Y)ᵀ * M.K⁻¹ * M.D := by
  have ht := congrArg Matrix.transpose (M.shift_normal_equation Y)
  have hs : (updatedPrecision M.D M.A M.K)ᵀ = updatedPrecision M.D M.A M.K := by
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
      (updatedPrecision_posDef _ _ _ M.hA M.hK).isHermitian.eq
  simpa only [Matrix.transpose_mul, Matrix.transpose_transpose, hs,
    inverse_symmetric _ M.hK, Matrix.mul_assoc] using ht

omit [DecidableEq c] in
theorem shifted_constant_energy (Y : Matrix n c ℝ) :
    M.predictiveEnergy Y =
      (M.residual Y)ᵀ * M.K⁻¹ * M.residual Y -
        (M.shift Y)ᵀ * updatedPrecision M.D M.A M.K * M.shift Y := by
  rw [predictiveEnergy, predictive_inverse _ _ _ M.hA M.hK,
    Matrix.mul_sub, Matrix.sub_mul, M.shift_transpose_normal_equation]
  simp only [shift, Matrix.mul_assoc]

omit [DecidableEq c] in
theorem general_completing_square (Y : Matrix n c ℝ) (W : Matrix p c ℝ) :
    M.fullEnergy Y W = M.coefficientEnergy Y W + M.predictiveEnergy Y := by
  have hy : Y - M.D * W = M.residual Y - M.D * (W - M.mean) := by
    simp only [residual, Matrix.mul_sub]
    abel
  have hw : W - M.coefficientCenter Y = (W - M.mean) - M.shift Y := by
    simp only [coefficientCenter]
    abel
  rw [fullEnergy, hy, coefficientEnergy, hw, M.shifted_constant_energy]
  simp only [Matrix.transpose_sub, Matrix.transpose_mul,
    Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_assoc]
  simp only [← Matrix.mul_assoc, M.shift_transpose_normal_equation]
  simp only [Matrix.mul_assoc, M.shift_normal_equation]
  simp only [updatedPrecision, Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc]
  abel

theorem weighted_completing_square (Y : Matrix n c ℝ) (W : Matrix p c ℝ) :
    weightedEnergy M.Sigma (M.fullEnergy Y W) =
      weightedEnergy M.Sigma (M.coefficientEnergy Y W) +
        weightedEnergy M.Sigma (M.predictiveEnergy Y) := by
  rw [M.general_completing_square]
  simp only [weightedEnergy, Matrix.mul_add, Matrix.trace_add]

omit [DecidableEq c] in
theorem row_determinant_factorization :
    |M.H⁻¹.det| * |M.R.det| = |M.G⁻¹.det| * |M.B.det| := by
  have hd := predictive_determinant M.D M.A M.K M.hA M.hK
  have hR := congrArg Matrix.det M.hR
  have hB := congrArg Matrix.det M.hB
  have hH := congrArg Matrix.det M.hH
  have hG := congrArg Matrix.det M.hG
  simp only [Matrix.det_mul, Matrix.det_transpose] at hR hB hH hG
  rw [← hR, ← hB, ← hH, ← hG] at hd
  have hh : M.H.det ≠ 0 := isUnit_iff_ne_zero.mp (whiteningFactor_unit _ _ M.hH M.hK)
  have hg : M.G.det ≠ 0 := isUnit_iff_ne_zero.mp
    (whiteningFactor_unit _ _ M.hG (predictiveCovariance_posDef _ _ _ M.hA M.hK))
  rw [Matrix.det_nonsing_inv, Matrix.det_nonsing_inv, Ring.inverse_eq_inv,
    Ring.inverse_eq_inv, abs_inv, abs_inv]
  have hs : (|M.H.det|⁻¹ * |M.R.det|) ^ 2 = (|M.G.det|⁻¹ * |M.B.det|) ^ 2 := by
    rw [mul_pow, mul_pow, inv_pow, inv_pow, sq_abs, sq_abs, sq_abs, sq_abs]
    field_simp
    nlinarith [hd]
  nlinarith [mul_nonneg (inv_nonneg.mpr (abs_nonneg M.H.det)) (abs_nonneg M.R.det),
    mul_nonneg (inv_nonneg.mpr (abs_nonneg M.G.det)) (abs_nonneg M.B.det)]

theorem jacobian_normalizers_factorization :
    jacobianNormalizer M.H⁻¹ M.L * jacobianNormalizer M.R M.L =
      jacobianNormalizer M.G⁻¹ M.L * jacobianNormalizer M.B M.L := by
  have hp := congrArg (fun x : ℝ => x ^ Fintype.card c) M.row_determinant_factorization
  simp only [mul_pow] at hp
  simp only [jacobianNormalizer, whiteningMatrix_abs_det]
  calc
    _ = (|M.H⁻¹.det| ^ Fintype.card c * |M.R.det| ^ Fintype.card c) *
        (|M.L⁻¹.det| ^ Fintype.card n * |M.L⁻¹.det| ^ Fintype.card p *
          (Real.sqrt (2 * Real.pi))⁻¹ ^ Fintype.card (n × c) *
          (Real.sqrt (2 * Real.pi))⁻¹ ^ Fintype.card (p × c)) := by ring
    _ = _ := by rw [hp]; ring

def priorPDF (x : (p × c) → ℝ) : ℝ := matrixPDF M.R M.L M.mean x

def jointAtData (Y : Matrix n c ℝ) (x : (p × c) → ℝ) : ℝ :=
  likelihoodDensity M.D (matricize x) M.H M.L (vectorize Y) * M.priorPDF x

def posteriorPDF (Y : Matrix n c ℝ) (x : (p × c) → ℝ) : ℝ :=
  matrixPDF M.B M.L (M.coefficientCenter Y) x

def predictivePDF (y : (n × c) → ℝ) : ℝ := matrixPDF M.G⁻¹ M.L (M.D * M.mean) y

theorem predictive_whitening_inverse_gram :
    ((whiteningMatrix M.G⁻¹ M.L)ᵀ * whiteningMatrix M.G⁻¹ M.L)⁻¹ =
      predictiveCovariance M.D M.A M.K ⊗ₖ M.Sigma := by
  rw [whitening_gram_inverse M.G⁻¹ (predictiveCovariance M.D M.A M.K)⁻¹ M.L M.Sigma
    (likelihood_row_factor_gram _ _ M.hG) M.hL M.hSigma,
    Matrix.nonsing_inv_nonsing_inv _
      (positive_definite_det_unit (predictiveCovariance_posDef _ _ _ M.hA M.hK))]

theorem joint_density_as_energy (Y : Matrix n c ℝ) (x : (p × c) → ℝ) :
    M.jointAtData Y x =
      (jacobianNormalizer M.H⁻¹ M.L * jacobianNormalizer M.R M.L) *
        Real.exp (-(1 / 2 : ℝ) * weightedEnergy M.Sigma (M.fullEnergy Y (matricize x))) := by
  unfold jointAtData priorPDF
  rw [likelihood_density_as_kernel M.D Y M.K M.H M.L M.Sigma (matricize x) M.hH M.hL]
  change jacobianNormalizer M.H⁻¹ M.L * likelihoodKernel M.D Y M.K M.Sigma (matricize x) *
    matrixPDF M.R M.L M.mean (vectorize (matricize x)) = _
  rw [matrixPDF_as_trace_kernel M.R M.A M.L M.Sigma M.mean (matricize x) M.hR M.hL]
  unfold likelihoodKernel fullEnergy weightedEnergy
  rw [Matrix.mul_add, Matrix.trace_add, mul_mul_mul_comm, ← Real.exp_add]
  congr 2
  ring

theorem posterior_pdf_as_kernel (Y : Matrix n c ℝ) (x : (p × c) → ℝ) :
    M.posteriorPDF Y x = jacobianNormalizer M.B M.L *
      Real.exp (-(1 / 2 : ℝ) * weightedEnergy M.Sigma (M.coefficientEnergy Y (matricize x))) :=
  matrixPDF_as_trace_kernel M.B (updatedPrecision M.D M.A M.K) M.L M.Sigma
    (M.coefficientCenter Y) (matricize x) M.hB M.hL

theorem predictive_pdf_as_kernel (Y : Matrix n c ℝ) :
    M.predictivePDF (vectorize Y) = jacobianNormalizer M.G⁻¹ M.L *
      Real.exp (-(1 / 2 : ℝ) * weightedEnergy M.Sigma (M.predictiveEnergy Y)) :=
  matrixPDF_as_trace_kernel M.G⁻¹ (predictiveCovariance M.D M.A M.K)⁻¹ M.L M.Sigma
    (M.D * M.mean) Y (likelihood_row_factor_gram _ _ M.hG) M.hL

/-- Exact factorization of independently normalized prior times likelihood.
The coefficient center, predictive energy and all normalizers have been derived
above; this identity assumes no desired marginal integral. -/
theorem joint_density_factorization (Y : Matrix n c ℝ) (x : (p × c) → ℝ) :
    M.jointAtData Y x = M.predictivePDF (vectorize Y) * M.posteriorPDF Y x := by
  rw [M.joint_density_as_energy, M.weighted_completing_square, mul_add, Real.exp_add,
    M.jacobian_normalizers_factorization, M.predictive_pdf_as_kernel,
    M.posterior_pdf_as_kernel]
  ring

theorem prior_integrable : Integrable M.priorPDF :=
  matrixPDF_integrable _ _ _ (precision_factor_unit _ _ M.hR M.hA)
    (whiteningFactor_unit _ _ M.hL M.hSigma)

theorem prior_integral_one : ∫ x, M.priorPDF x = 1 :=
  matrixPDF_integral_one _ _ _ (precision_factor_unit _ _ M.hR M.hA)
    (whiteningFactor_unit _ _ M.hL M.hSigma)

theorem posterior_integrable (Y : Matrix n c ℝ) : Integrable (M.posteriorPDF Y) :=
  matrixPDF_integrable _ _ _
    (precision_factor_unit _ _ M.hB (updatedPrecision_posDef _ _ _ M.hA M.hK))
    (whiteningFactor_unit _ _ M.hL M.hSigma)

theorem posterior_integral_one (Y : Matrix n c ℝ) : ∫ x, M.posteriorPDF Y x = 1 :=
  matrixPDF_integral_one _ _ _
    (precision_factor_unit _ _ M.hB (updatedPrecision_posDef _ _ _ M.hA M.hK))
    (whiteningFactor_unit _ _ M.hL M.hSigma)

/-- Proved independently of the Bochner integral equality, so a nonintegrable
function's totalized integral cannot fake the marginal calculation. -/
theorem coefficient_integrand_integrable (Y : Matrix n c ℝ) :
    Integrable (M.jointAtData Y) := by
  have heq : M.jointAtData Y = fun x => M.predictivePDF (vectorize Y) * M.posteriorPDF Y x :=
    funext (M.joint_density_factorization Y)
  rw [heq]
  exact (M.posterior_integrable Y).const_mul _

/-- The actual predictive marginal integral over the row-major coefficient
Lebesgue measure equals the normalized matrix-normal density with mean D m
and row covariance K + D A^-1 Dᵀ. -/
theorem predictive_marginal_integral (Y : Matrix n c ℝ) :
    (∫ x, M.jointAtData Y x) = M.predictivePDF (vectorize Y) := by
  simp_rw [M.joint_density_factorization]
  rw [integral_const_mul, M.posterior_integral_one, mul_one]

theorem predictive_marginal_explicit (Y : Matrix n c ℝ) :
    (∫ x, likelihoodDensity M.D (matricize x) M.H M.L (vectorize Y) *
      matrixPDF M.R M.L M.mean x) = matrixPDF M.G⁻¹ M.L (M.D * M.mean) (vectorize Y) :=
  M.predictive_marginal_integral Y

theorem predictive_integrable : Integrable M.predictivePDF :=
  matrixPDF_integrable _ _ _
    (Matrix.isUnit_nonsing_inv_det M.G (whiteningFactor_unit _ _ M.hG
      (predictiveCovariance_posDef _ _ _ M.hA M.hK)))
    (whiteningFactor_unit _ _ M.hL M.hSigma)

theorem predictive_integral_one : ∫ y, M.predictivePDF y = 1 :=
  matrixPDF_integral_one _ _ _
    (Matrix.isUnit_nonsing_inv_det M.G (whiteningFactor_unit _ _ M.hG
      (predictiveCovariance_posDef _ _ _ M.hA M.hK)))
    (whiteningFactor_unit _ _ M.hL M.hSigma)

theorem predictive_nonnegative (y : (n × c) → ℝ) : 0 ≤ M.predictivePDF y :=
  matrixPDF_nonnegative _ _ _ _

theorem predictive_positive (Y : Matrix n c ℝ) : 0 < M.predictivePDF (vectorize Y) := by
  rw [M.predictive_pdf_as_kernel]
  exact mul_pos (jacobianNormalizer_positive _ _
    (Matrix.isUnit_nonsing_inv_det M.G (whiteningFactor_unit _ _ M.hG
      (predictiveCovariance_posDef _ _ _ M.hA M.hK)))
    (whiteningFactor_unit _ _ M.hL M.hSigma)) (Real.exp_pos _)

theorem predictive_integral_positive (Y : Matrix n c ℝ) : 0 < ∫ x, M.jointAtData Y x := by
  rw [M.predictive_marginal_integral]
  exact M.predictive_positive Y

theorem predictive_lintegral_one : ∫⁻ y, ENNReal.ofReal (M.predictivePDF y) = 1 := by
  rw [← ofReal_integral_eq_lintegral_ofReal M.predictive_integrable
    (Filter.Eventually.of_forall M.predictive_nonnegative), M.predictive_integral_one]
  exact ENNReal.ofReal_one

def predictiveMeasure : Measure ((n × c) → ℝ) :=
  volume.withDensity (fun y => ENNReal.ofReal (M.predictivePDF y))

theorem predictive_probability : IsProbabilityMeasure M.predictiveMeasure :=
  matrixKernelMeasure_probability _ _ _
    (Matrix.isUnit_nonsing_inv_det M.G (whiteningFactor_unit _ _ M.hG
      (predictiveCovariance_posDef _ _ _ M.hA M.hK)))
    (whiteningFactor_unit _ _ M.hL M.hSigma)

end Model

abbrev One := Fin 1

def sqrtTwoFactor : Matrix One One ℝ := Real.sqrt 2 • 1

theorem sqrtTwoFactor_gram : sqrtTwoFactorᵀ * sqrtTwoFactor = (2 : ℝ) • 1 := by
  unfold sqrtTwoFactor
  simp only [Matrix.transpose_smul, Matrix.transpose_one, Matrix.smul_mul,
    Matrix.mul_smul, Matrix.one_mul, smul_smul]
  rw [← pow_two, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

theorem sqrtTwoFactor_cov : sqrtTwoFactor * sqrtTwoFactorᵀ = (2 : ℝ) • 1 := by
  simpa only [sqrtTwoFactor, Matrix.transpose_smul, Matrix.transpose_one] using sqrtTwoFactor_gram

/-- A concrete legal one-row, one-channel model with prior variance one and
independent observation variance one, hence predictive variance two. -/
def scalarModel : Model (n := One) (p := One) (c := One) where
  D := 1
  A := 1
  K := 1
  Sigma := 1
  mean := 0
  R := 1
  B := sqrtTwoFactor
  H := 1
  G := sqrtTwoFactor
  L := 1
  hA := Matrix.PosDef.one
  hK := Matrix.PosDef.one
  hSigma := Matrix.PosDef.one
  hR := by simp
  hB := by
    rw [sqrtTwoFactor_gram]
    ext i j
    fin_cases i
    fin_cases j
    norm_num [updatedPrecision, Matrix.mul_apply, Fin.sum_univ_one]
  hH := by simp
  hG := by
    rw [sqrtTwoFactor_cov]
    ext i j
    fin_cases i
    fin_cases j
    norm_num [predictiveCovariance, Matrix.mul_apply, Fin.sum_univ_one]
  hL := by simp

/-- Integrating coefficient uncertainty reduces the centered peak by sqrt(2).
This is an actual integral witness, not just a covariance-matrix inequality. -/
theorem scalar_integrated_peak_ratio :
    Real.sqrt 2 * (∫ x, scalarModel.jointAtData 0 x) = scalarModel.priorPDF 0 := by
  rw [scalarModel.predictive_marginal_integral, scalarModel.predictive_pdf_as_kernel]
  have he : scalarModel.predictiveEnergy 0 = 0 := by
    simp [Model.predictiveEnergy, Model.residual, scalarModel]
  rw [he]
  have hp : scalarModel.priorPDF 0 =
      jacobianNormalizer (1 : Matrix One One ℝ) (1 : Matrix One One ℝ) := by
    have h := matrixPDF_as_trace_kernel (1 : Matrix One One ℝ) 1
      (1 : Matrix One One ℝ) 1 0 0
      (by simp) (by simp)
    simpa [Model.priorPDF, scalarModel, weightedEnergy] using h
  rw [hp]
  have hs : sqrtTwoFactor⁻¹ = (Real.sqrt 2)⁻¹ • (1 : Matrix One One ℝ) := by
    simpa only [sqrtTwoFactor, inv_one] using
      inverse_smul (1 : Matrix One One ℝ) (Real.sqrt 2)
      (ne_of_gt (Real.sqrt_pos.mpr (by norm_num))) (by rw [Matrix.det_one]; exact isUnit_one)
  change Real.sqrt 2 * (jacobianNormalizer sqrtTwoFactor⁻¹ (1 : Matrix One One ℝ) *
    Real.exp (-(1 / 2 : ℝ) * weightedEnergy (1 : Matrix One One ℝ) 0)) = _
  rw [hs]
  simp [jacobianNormalizer, whiteningMatrix_abs_det, weightedEnergy,
    abs_of_pos (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2))]

theorem scalar_plug_in_peak_wrong :
    (∫ x, scalarModel.jointAtData 0 x) ≠ scalarModel.priorPDF 0 := by
  intro h
  have hp := scalarModel.predictive_integral_positive (0 : Matrix One One ℝ)
  have hr := scalar_integrated_peak_ratio
  rw [← h] at hr
  have hs : 1 < Real.sqrt 2 := by
    have hh := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
    nlinarith [Real.sqrt_nonneg (2 : ℝ)]
  nlinarith

end

end N2P3.GaussianConvolution
