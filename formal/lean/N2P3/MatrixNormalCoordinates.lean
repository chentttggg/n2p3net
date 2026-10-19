import N2P3.GaussianConditioning
import N2P3.GaussianMeasure
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Row-major coordinates for the actual matrix-normal kernel

Rows precede channels, matching NumPy's default reshape(-1). If the row
precision is A=RᵀR and the channel covariance is Sigma=LLᵀ, the whitening
matrix is R ⊗ L^-1. Its action is vec(R E L^-T), not a column-major convention.
No posterior density or covariance identity is used as an assumption.
-/

namespace N2P3.MatrixNormalCoordinates

open Matrix
open MeasureTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning
open scoped BigOperators Matrix ComplexOrder Kronecker

noncomputable section

variable {p c n : Type*}

def vectorize (E : Matrix p c ℝ) : (p × c) → ℝ := fun j => E j.1 j.2

def matricize (v : (p × c) → ℝ) : Matrix p c ℝ := fun i j => v (i, j)

def vectorizeEquiv : Matrix p c ℝ ≃ₗ[ℝ] ((p × c) → ℝ) where
  toFun := vectorize
  invFun := matricize
  left_inv _ := rfl
  right_inv _ := rfl
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem vectorize_sub (E F : Matrix p c ℝ) : vectorize (E - F) = vectorize E - vectorize F := rfl

/-- Explicit C-order storage for a finite rows-by-channels array. -/
def rowMajorVector {rows channels : ℕ} (E : Matrix (Fin rows) (Fin channels) ℝ) :
    Fin (rows * channels) → ℝ := fun j => vectorize E (finProdFinEquiv.symm j)

theorem rowMajorIndex_value {rows channels : ℕ} (i : Fin rows) (j : Fin channels) :
    (finProdFinEquiv (i, j)).val = channels * i.val + j.val := by
  simp [finProdFinEquiv]
  omega

theorem rowMajorVector_at {rows channels : ℕ} (E : Matrix (Fin rows) (Fin channels) ℝ)
    (i : Fin rows) (j : Fin channels) :
    rowMajorVector E (finProdFinEquiv (i, j)) = E i j := by
  simp [rowMajorVector, vectorize]

variable [Fintype p] [Fintype c] [Fintype n]
variable [DecidableEq p] [DecidableEq c] [DecidableEq n]

omit [DecidableEq p] [DecidableEq c] in
/-- General scalar Kronecker identity in the declared row-major convention. -/
theorem kronecker_vectorize (R : Matrix p p ℝ) (S : Matrix c c ℝ) (E : Matrix p c ℝ) :
    (R ⊗ₖ S) *ᵥ vectorize E = vectorize (R * E * Sᵀ) := by
  funext j
  rcases j with ⟨i, a⟩
  simp only [Matrix.mulVec, dotProduct, vectorize, Matrix.kronecker_apply,
    Matrix.mul_apply, Matrix.transpose_apply, Fintype.sum_prod_type, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro k _
  ring

omit [DecidableEq p] [DecidableEq c] in
theorem vectorize_energy_trace (E : Matrix p c ℝ) :
    vectorize E ⬝ᵥ vectorize E = Matrix.trace (Eᵀ * E) := by
  simp only [dotProduct, vectorize, Fintype.sum_prod_type, Matrix.trace,
    Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  rw [Finset.sum_comm]

omit [DecidableEq p] [DecidableEq c] in
theorem vectorize_squares_trace (E : Matrix p c ℝ) :
    (∑ j, (vectorize E j) ^ 2) = Matrix.trace (Eᵀ * E) := by
  simpa only [pow_two, dotProduct] using vectorize_energy_trace E

def whiteningMatrix (R : Matrix p p ℝ) (L : Matrix c c ℝ) :
    Matrix (p × c) (p × c) ℝ := R ⊗ₖ L⁻¹

def whitenResidual (R : Matrix p p ℝ) (L : Matrix c c ℝ) (E : Matrix p c ℝ) :
    Matrix p c ℝ := R * E * (L⁻¹)ᵀ

omit [DecidableEq p] in
theorem whitening_coordinates (R : Matrix p p ℝ) (L : Matrix c c ℝ) (E : Matrix p c ℝ) :
    whiteningMatrix R L *ᵥ vectorize E = vectorize (whitenResidual R L E) :=
  kronecker_vectorize R L⁻¹ E

omit [DecidableEq p] in
theorem whitening_trace (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (E : Matrix p c ℝ) (hA : Rᵀ * R = A) (hL : L * Lᵀ = Sigma) :
    Matrix.trace ((whitenResidual R L E)ᵀ * whitenResidual R L E) =
      weightedEnergy Sigma (Eᵀ * A * E) := by
  rw [weightedEnergy_as_congruence Sigma L (Eᵀ * A * E) hL, ← hA]
  simp only [whitenResidual, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]

omit [DecidableEq p] in
/-- The actual full-channel trace exponent is the squared norm of M vec(E). -/
theorem whitening_squares (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (E : Matrix p c ℝ) (hA : Rᵀ * R = A) (hL : L * Lᵀ = Sigma) :
    (∑ j, ((whiteningMatrix R L *ᵥ vectorize E) j) ^ 2) =
      weightedEnergy Sigma (Eᵀ * A * E) := by
  rw [whitening_coordinates, vectorize_squares_trace, whitening_trace R A L Sigma E hA hL]

theorem whiteningMatrix_det (R : Matrix p p ℝ) (L : Matrix c c ℝ) :
    (whiteningMatrix R L).det = R.det ^ Fintype.card c * (L⁻¹).det ^ Fintype.card p := by
  exact Matrix.det_kronecker R L⁻¹

/-- The affine Jacobian uses absolute determinant, allowing QR sign choices. -/
theorem whiteningMatrix_abs_det (R : Matrix p p ℝ) (L : Matrix c c ℝ) :
    |(whiteningMatrix R L).det| = |R.det| ^ Fintype.card c * |(L⁻¹).det| ^ Fintype.card p := by
  rw [whiteningMatrix_det, abs_mul, abs_pow, abs_pow]

theorem whiteningMatrix_det_ne_zero (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (hR : IsUnit R.det) (hL : IsUnit L.det) : (whiteningMatrix R L).det ≠ 0 := by
  rw [whiteningMatrix_det]
  exact mul_ne_zero (pow_ne_zero _ (isUnit_iff_ne_zero.mp hR))
    (pow_ne_zero _ (isUnit_iff_ne_zero.mp (Matrix.isUnit_nonsing_inv_det L hL)))

/-- Legal source K, positive ridge, exact QR and spatial Cholesky factors prove
invertibility of the actual vectorized whitening matrix. -/
theorem source_whiteningMatrix_det_ne_zero
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    (whiteningMatrix R L).det ≠ 0 :=
  whiteningMatrix_det_ne_zero R L (normalFactor_unit D K lam R hR hK hlam)
    (whiteningFactor_unit L Sigma hL hSigma)

/-- Completion-of-square's exact centered matrix energy has the same coordinates
as the independently constructed affine Gaussian whitening map. -/
theorem source_centered_whitening_squares
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ) (W : Matrix p c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma) :
    (∑ j, ((whiteningMatrix R L *ᵥ (vectorize W - vectorize (mean D Y K lam))) j) ^ 2) =
      weightedEnergy Sigma (centeredEnergy D Y K lam W) := by
  rw [← vectorize_sub]
  exact whitening_squares R (precision D K lam) L Sigma (W - mean D Y K lam) hR hL

omit [DecidableEq p] in
/-- Vectorized precision is derived from the actual two factor equations. -/
theorem whitening_gram (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hA : Rᵀ * R = A) (hL : L * Lᵀ = Sigma) :
    (whiteningMatrix R L)ᵀ * whiteningMatrix R L = A ⊗ₖ Sigma⁻¹ := by
  have hinv : Sigma⁻¹ = (L⁻¹)ᵀ * L⁻¹ := by
    rw [← hL, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv]
  unfold whiteningMatrix
  rw [← Matrix.kroneckerMap_transpose, ← Matrix.mul_kronecker_mul, hA, ← hinv]

/-- The inverse Gram is the expected row/channel covariance parameter. This
is an inverse identity, not an assumed probability-moment formula. -/
theorem whitening_gram_inverse (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hA : Rᵀ * R = A) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    ((whiteningMatrix R L)ᵀ * whiteningMatrix R L)⁻¹ = A⁻¹ ⊗ₖ Sigma := by
  rw [whitening_gram R A L Sigma hA hL, Matrix.inv_kronecker,
    Matrix.nonsing_inv_nonsing_inv Sigma (positive_definite_det_unit hSigma)]

theorem source_whitening_gram_inverse
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    ((whiteningMatrix R L)ᵀ * whiteningMatrix R L)⁻¹ = rowCov D K lam ⊗ₖ Sigma :=
  whitening_gram_inverse R (precision D K lam) L Sigma hR hL hSigma

/-- Density on explicitly row-major coefficient coordinates. -/
def matrixPDF (R : Matrix p p ℝ) (L : Matrix c c ℝ) (center : Matrix p c ℝ)
    (x : (p × c) → ℝ) : ℝ :=
  N2P3.GaussianMeasure.affinePDF (1 / 2) (whiteningMatrix R L) (vectorize center) x

def jacobianNormalizer (R : Matrix p p ℝ) (L : Matrix c c ℝ) : ℝ :=
  |(whiteningMatrix R L).det| * (Real.sqrt (2 * Real.pi))⁻¹ ^ Fintype.card (p × c)

/-- The normalizer contains the actual absolute Jacobian, with no QR sign
assumption. The matrix trace exponent is derived by finite coordinate algebra. -/
theorem matrixPDF_as_trace_kernel (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (center W : Matrix p c ℝ) (hA : Rᵀ * R = A) (hL : L * Lᵀ = Sigma) :
    matrixPDF R L center (vectorize W) = jacobianNormalizer R L *
      Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma ((W - center)ᵀ * A * (W - center))) := by
  rw [matrixPDF, N2P3.GaussianMeasure.affinePDF_as_quadratic_kernel,
    ← vectorize_sub, whitening_squares R A L Sigma (W - center) hA hL]
  have hpi : Real.pi / (1 / 2 : ℝ) = 2 * Real.pi := by ring
  rw [hpi]
  rfl

theorem matrixPDF_nonnegative (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (center : Matrix p c ℝ) (x : (p × c) → ℝ) : 0 ≤ matrixPDF R L center x :=
  N2P3.GaussianMeasure.affinePDF_nonnegative _ _ _ _

theorem matrixPDF_integrable (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (center : Matrix p c ℝ) (hR : IsUnit R.det) (hL : IsUnit L.det) :
    Integrable (matrixPDF R L center) :=
  N2P3.GaussianMeasure.affinePDF_integrable (1 / 2) (by norm_num) (whiteningMatrix R L)
    (whiteningMatrix_det_ne_zero R L hR hL) (vectorize center)

theorem matrixPDF_integral_one (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (center : Matrix p c ℝ) (hR : IsUnit R.det) (hL : IsUnit L.det) :
    ∫ x, matrixPDF R L center x = 1 :=
  N2P3.GaussianMeasure.affinePDF_integral_one (1 / 2) (by norm_num) (whiteningMatrix R L)
    (whiteningMatrix_det_ne_zero R L hR hL) (vectorize center)

def matrixKernelMeasure (R : Matrix p p ℝ) (L : Matrix c c ℝ) (center : Matrix p c ℝ) :
    Measure ((p × c) → ℝ) :=
  N2P3.GaussianMeasure.affineGaussianMeasure (1 / 2) (whiteningMatrix R L) (vectorize center)

theorem matrixKernelMeasure_probability (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (center : Matrix p c ℝ) (hR : IsUnit R.det) (hL : IsUnit L.det) :
    IsProbabilityMeasure (matrixKernelMeasure R L center) :=
  N2P3.GaussianMeasure.affineGaussianMeasure_isProbabilityMeasure
    (1 / 2) (by norm_num) (whiteningMatrix R L)
      (whiteningMatrix_det_ne_zero R L hR hL) (vectorize center)

/-- The source-derived center/precision now instantiate the normalized density,
and its shape is exactly the completed conditional kernel. -/
theorem source_matrixPDF_as_centeredKernel
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ) (W : Matrix p c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma) :
    matrixPDF R L (mean D Y K lam) (vectorize W) =
      jacobianNormalizer R L * centeredKernel D Y K Sigma lam W :=
  matrixPDF_as_trace_kernel R (precision D K lam) L Sigma (mean D Y K lam) W hR hL

/-- Probability mass one follows from legal source K/ridge/spatial covariance
and exact source factor equations; density normalization is not assumed. -/
theorem source_matrixKernelMeasure_probability
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (matrixKernelMeasure R L (mean D Y K lam)) :=
  matrixKernelMeasure_probability R L (mean D Y K lam)
    (normalFactor_unit D K lam R hR hK hlam) (whiteningFactor_unit L Sigma hL hSigma)

end

end N2P3.MatrixNormalCoordinates
