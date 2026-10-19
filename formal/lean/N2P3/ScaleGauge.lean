import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Scale/gauge identities of the dense matrix-normal reference

Exact real matrix normal equations from
`experiments.boundary_state_research.ScaleInvariantContinuousModel`.
Positive definiteness and positive ridge establish actual invertibility before
inverse identities are used. No distributional posterior theorem, fitted-noise
validity, or floating point/QR/Cholesky refinement is asserted here.
-/

namespace N2P3.ScaleGauge

open Matrix
open scoped BigOperators Matrix ComplexOrder Kronecker

noncomputable section

variable {n p m c : Type*}
variable [Fintype n] [Fintype p] [Fintype m] [Fintype c]
variable [DecidableEq n] [DecidableEq p] [DecidableEq m] [DecidableEq c]

def precision (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ) : Matrix p p ℝ :=
  Dᵀ * K⁻¹ * D + lam • 1

def rowCov (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ) : Matrix p p ℝ :=
  (precision D K lam)⁻¹

def mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ) (lam : ℝ) :
    Matrix p c ℝ := rowCov D K lam * Dᵀ * K⁻¹ * Y

def temporalPredictive (A : Matrix m p ℝ) (Kt : Matrix m m ℝ) (P : Matrix p p ℝ) :
    Matrix m m ℝ := Kt + A * P * Aᵀ

/-- Time-major row/channel product indices, exactly `np.kron(T, Sigma)`. -/
def predictiveCov (T : Matrix m m ℝ) (Sigma : Matrix c c ℝ) :
    Matrix (m × c) (m × c) ℝ := T ⊗ₖ Sigma

theorem positive_definite_det_unit {K : Matrix n n ℝ} (hK : K.PosDef) :
    IsUnit K.det :=
  (Matrix.isUnit_iff_isUnit_det K).mp hK.isUnit

theorem precision_positive_definite (D : Matrix n p ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) : (precision D K lam).PosDef := by
  have hgram : (Dᵀ * K⁻¹ * D).PosSemidef := by
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
      hK.inv.posSemidef.conjTranspose_mul_mul_same D
  exact Matrix.PosDef.posSemidef_add hgram (Matrix.PosDef.one.smul hlam)

theorem rowCov_positive_definite (D : Matrix n p ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) : (rowCov D K lam).PosDef :=
  (precision_positive_definite D K lam hK hlam).inv

/-- Inverse scaling follows from a verified two-sided inverse, not a desired
covariance equality. The determinant unit premise is discharged by SPD above. -/
theorem inverse_smul (K : Matrix n n ℝ) (a : ℝ) (ha : a ≠ 0)
    (hK : IsUnit K.det) : (a • K)⁻¹ = a⁻¹ • K⁻¹ := by
  apply Matrix.inv_eq_left_inv
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul,
    inv_mul_cancel₀ ha, one_smul, Matrix.nonsing_inv_mul _ hK]

omit [Fintype p] in
theorem precision_gauge (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam a : ℝ)
    (ha : 0 < a) (hK : K.PosDef) :
    precision D (a • K) (lam / a) = a⁻¹ • precision D K lam := by
  unfold precision
  rw [inverse_smul K a (ne_of_gt ha) (positive_definite_det_unit hK)]
  simp only [Matrix.mul_smul, Matrix.smul_mul, smul_add, smul_smul]
  rw [div_eq_mul_inv, mul_comm lam a⁻¹]

theorem rowCov_gauge (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam a : ℝ)
    (ha : 0 < a) (hK : K.PosDef) (hlam : 0 < lam) :
    rowCov D (a • K) (lam / a) = a • rowCov D K lam := by
  unfold rowCov
  rw [precision_gauge D K lam a ha hK,
    inverse_smul (precision D K lam) a⁻¹ (inv_ne_zero (ne_of_gt ha))
      (positive_definite_det_unit (precision_positive_definite D K lam hK hlam)), inv_inv]

omit [Fintype c] [DecidableEq c] in
theorem mean_gauge (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam a : ℝ)
    (ha : 0 < a) (hK : K.PosDef) (hlam : 0 < lam) :
    mean D Y (a • K) (lam / a) = mean D Y K lam := by
  unfold mean
  rw [rowCov_gauge D K lam a ha hK hlam,
    inverse_smul K a (ne_of_gt ha) (positive_definite_det_unit hK)]
  simp only [Matrix.mul_smul, Matrix.smul_mul, smul_smul,
    inv_mul_cancel₀ (ne_of_gt ha), one_smul]

omit [Fintype m] [DecidableEq p] [DecidableEq m] in
theorem temporalPredictive_gauge (A : Matrix m p ℝ) (Kt : Matrix m m ℝ)
    (P : Matrix p p ℝ) (a : ℝ) :
    temporalPredictive A (a • Kt) (a • P) = a • temporalPredictive A Kt P := by
  simp only [temporalPredictive, Matrix.mul_smul, Matrix.smul_mul, smul_add]

omit [Fintype m] [Fintype c] [DecidableEq m] [DecidableEq c] in
theorem kronecker_gauge (T : Matrix m m ℝ) (Sigma : Matrix c c ℝ)
    (a : ℝ) (ha : a ≠ 0) :
    predictiveCov (a • T) (a⁻¹ • Sigma) = predictiveCov T Sigma := by
  simp only [predictiveCov, Matrix.smul_kronecker, Matrix.kronecker_smul,
    smul_smul, inv_mul_cancel₀ ha, one_smul]

omit [Fintype m] [Fintype c] [DecidableEq m] [DecidableEq c] in
/-- Full predictive covariance, not just candidate score differences, is gauge
invariant when lambda is divided by the same temporal covariance factor. -/
theorem predictiveCov_gauge (D : Matrix n p ℝ) (A : Matrix m p ℝ)
    (K : Matrix n n ℝ) (Kt : Matrix m m ℝ) (Sigma : Matrix c c ℝ)
    (lam a : ℝ) (ha : 0 < a) (hK : K.PosDef) (hlam : 0 < lam) :
    predictiveCov (temporalPredictive A (a • Kt)
      (rowCov D (a • K) (lam / a))) (a⁻¹ • Sigma) =
    predictiveCov (temporalPredictive A Kt (rowCov D K lam)) Sigma := by
  rw [rowCov_gauge D K lam a ha hK hlam, temporalPredictive_gauge,
    kronecker_gauge _ _ a (ne_of_gt ha)]

omit [Fintype c] [DecidableEq c] in
theorem mean_voltage (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam s : ℝ) :
    mean D (s • Y) K lam = s • mean D Y K lam := by
  simp only [mean, Matrix.mul_smul]

omit [Fintype m] [Fintype c] [DecidableEq m] [DecidableEq c] in
theorem predictiveCov_voltage (T : Matrix m m ℝ) (Sigma : Matrix c c ℝ) (s : ℝ) :
    predictiveCov T ((s ^ 2) • Sigma) = (s ^ 2) • predictiveCov T Sigma := by
  rw [predictiveCov, Matrix.kronecker_smul]
  rfl

omit [Fintype m] [Fintype c] [DecidableEq m] [DecidableEq c] in
theorem residual_voltage (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (A : Matrix m p ℝ) (Yt : Matrix m c ℝ) (K : Matrix n n ℝ) (lam s : ℝ) :
    s • Yt - A * mean D (s • Y) K lam = s • (Yt - A * mean D Y K lam) := by
  rw [mean_voltage, Matrix.mul_smul, smul_sub]

omit [DecidableEq p] [DecidableEq m] in
theorem temporalPredictive_positive_definite (A : Matrix m p ℝ)
    (Kt : Matrix m m ℝ) (P : Matrix p p ℝ) (hKt : Kt.PosDef) (hP : P.PosSemidef) :
    (temporalPredictive A Kt P).PosDef := by
  apply hKt.add_posSemidef
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    hP.mul_mul_conjTranspose_same A

theorem predictiveCov_det_positive (T : Matrix m m ℝ) (Sigma : Matrix c c ℝ)
    (hT : T.PosDef) (hSigma : Sigma.PosDef) : 0 < (predictiveCov T Sigma).det := by
  rw [predictiveCov, Matrix.det_kronecker]
  exact mul_pos (pow_pos hT.det_pos _) (pow_pos hSigma.det_pos _)

def quadratic (C : Matrix n n ℝ) (r : n → ℝ) : ℝ := r ⬝ᵥ (C⁻¹ *ᵥ r)

/-- Source score with the common Gaussian normalizing constant omitted. -/
def logEvidence (C : Matrix n n ℝ) (r : n → ℝ) : ℝ :=
  -(1 / 2 : ℝ) * (quadratic C r + Real.log C.det)

theorem quadratic_voltage (C : Matrix n n ℝ) (r : n → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hC : IsUnit C.det) :
    quadratic ((s ^ 2) • C) (s • r) = quadratic C r := by
  unfold quadratic
  rw [inverse_smul C (s ^ 2) (pow_ne_zero _ hs) hC]
  simp only [Matrix.smul_mulVec, Matrix.mulVec_smul, smul_dotProduct,
    dotProduct_smul, smul_eq_mul]
  have hc : s * ((s ^ 2)⁻¹ * s) = 1 := by field_simp
  calc
    s * ((s ^ 2)⁻¹ * (s * (r ⬝ᵥ (C⁻¹ *ᵥ r)))) =
        (s * ((s ^ 2)⁻¹ * s)) * (r ⬝ᵥ (C⁻¹ *ᵥ r)) := by ring
    _ = r ⬝ᵥ (C⁻¹ *ᵥ r) := by rw [hc, one_mul]

theorem logDet_voltage (C : Matrix n n ℝ) (s : ℝ)
    (hs : s ≠ 0) (hC : IsUnit C.det) :
    Real.log (((s ^ 2) • C).det) =
      (Fintype.card n : ℝ) * Real.log (s ^ 2) + Real.log C.det := by
  rw [Matrix.det_smul, Real.log_mul (pow_ne_zero _ (pow_ne_zero 2 hs))
    (isUnit_iff_ne_zero.mp hC), Real.log_pow]

theorem logEvidence_voltage (C : Matrix n n ℝ) (r : n → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hC : IsUnit C.det) :
    logEvidence ((s ^ 2) • C) (s • r) = logEvidence C r -
      (Fintype.card n : ℝ) / 2 * Real.log (s ^ 2) := by
  unfold logEvidence
  rw [quadratic_voltage C r s hs hC, logDet_voltage C s hs hC]
  ring

/-- Candidate dimensions agree, so unit conversion adds one common constant. -/
theorem logEvidence_voltage_difference (C D : Matrix n n ℝ) (r t : n → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hC : IsUnit C.det) (hD : IsUnit D.det) :
    logEvidence ((s ^ 2) • C) (s • r) - logEvidence ((s ^ 2) • D) (s • t) =
      logEvidence C r - logEvidence D t := by
  rw [logEvidence_voltage C r s hs hC, logEvidence_voltage D t s hs hD]
  ring

/-- The whole dense candidate score uses the same row/channel coordinate order
as its Kronecker covariance. -/
def candidateEvidence (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (A : Matrix m p ℝ) (Yt : Matrix m c ℝ) (Kt : Matrix m m ℝ)
    (Sigma : Matrix c c ℝ) : ℝ :=
  logEvidence (predictiveCov (temporalPredictive A Kt (rowCov D K lam)) Sigma)
    (fun j => (Yt - A * mean D Y K lam) j.1 j.2)

theorem candidateEvidence_gauge (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (A : Matrix m p ℝ)
    (Yt : Matrix m c ℝ) (Kt : Matrix m m ℝ) (Sigma : Matrix c c ℝ)
    (a : ℝ) (ha : 0 < a) (hK : K.PosDef) (hlam : 0 < lam) :
    candidateEvidence D Y (a • K) (lam / a) A Yt (a • Kt) (a⁻¹ • Sigma) =
      candidateEvidence D Y K lam A Yt Kt Sigma := by
  unfold candidateEvidence
  rw [predictiveCov_gauge D A K Kt Sigma lam a ha hK hlam,
    mean_gauge D Y K lam a ha hK hlam]

theorem candidateEvidence_voltage (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (A : Matrix m p ℝ)
    (Yt : Matrix m c ℝ) (Kt : Matrix m m ℝ) (Sigma : Matrix c c ℝ)
    (s : ℝ) (hs : s ≠ 0) (hK : K.PosDef) (hlam : 0 < lam)
    (hKt : Kt.PosDef) (hSigma : Sigma.PosDef) :
    candidateEvidence D (s • Y) K lam A (s • Yt) Kt ((s ^ 2) • Sigma) =
      candidateEvidence D Y K lam A Yt Kt Sigma -
        (Fintype.card (m × c) : ℝ) / 2 * Real.log (s ^ 2) := by
  have hT := temporalPredictive_positive_definite A Kt (rowCov D K lam) hKt
    (rowCov_positive_definite D K lam hK hlam).posSemidef
  have hC : IsUnit (predictiveCov (temporalPredictive A Kt (rowCov D K lam)) Sigma).det :=
    isUnit_iff_ne_zero.mpr (ne_of_gt (predictiveCov_det_positive _ _ hT hSigma))
  unfold candidateEvidence
  rw [predictiveCov_voltage]
  simp_rw [residual_voltage]
  exact logEvidence_voltage _ _ s hs hC

omit [Fintype p] [Fintype c] [DecidableEq c] in
/-- The physical matrix-normal prior covariance is unchanged by the gauge. -/
theorem physicalPrior_gauge (Sigma : Matrix c c ℝ) (lam a : ℝ) (ha : a ≠ 0) :
    predictiveCov (((lam / a)⁻¹) • (1 : Matrix p p ℝ)) (a⁻¹ • Sigma) =
      predictiveCov (lam⁻¹ • (1 : Matrix p p ℝ)) Sigma := by
  have hscale : (lam / a)⁻¹ = a * lam⁻¹ := by rw [inv_div, div_eq_mul_inv]
  rw [hscale, MulAction.mul_smul]
  exact kronecker_gauge _ _ a ha

omit [Fintype p] [Fintype c] [DecidableEq p] [DecidableEq c] in
/-- Exact Cholesky whitening gives the normal-equation cross product. The
factor equation is a solver specification, not a floating-point assertion. -/
theorem whitening_cross_product (L K : Matrix n n ℝ) (D : Matrix n p ℝ)
    (Y : Matrix n c ℝ) (hL : L * Lᵀ = K) :
    (L⁻¹ * D)ᵀ * (L⁻¹ * Y) = Dᵀ * K⁻¹ * Y := by
  rw [← hL, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv,
    Matrix.transpose_mul]
  simp only [Matrix.mul_assoc]

/-- `inverse @ inverse.T` equals the normal-equation covariance whenever the
exact augmented QR factor has Gram matrix equal to the proved precision. -/
theorem normalFactor_rowCov (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam) :
    R⁻¹ * (R⁻¹)ᵀ = rowCov D K lam := by
  unfold rowCov
  rw [← hR, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv]

theorem normalFactor_unit (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = precision D K lam)
    (hK : K.PosDef) (hlam : 0 < lam) : IsUnit R.det := by
  have hM := positive_definite_det_unit (precision_positive_definite D K lam hK hlam)
  have hprod : IsUnit (R.det * R.det) := by
    simpa only [← hR, Matrix.det_mul, Matrix.det_transpose] using hM
  exact isUnit_of_mul_isUnit_left hprod

/-- Vertical concatenation, with data rows before ridge rows. -/
def verticalStack (A : Matrix n p ℝ) (B : Matrix m p ℝ) : Matrix (n ⊕ m) p ℝ :=
  fun i j => Sum.elim (fun k => A k j) (fun k => B k j) i

omit [Fintype p] [DecidableEq n] [DecidableEq p] [DecidableEq m] in
theorem verticalStack_gram (A : Matrix n p ℝ) (B : Matrix m p ℝ) :
    (verticalStack A B)ᵀ * verticalStack A B = Aᵀ * A + Bᵀ * B := by
  ext i j
  simp [Matrix.mul_apply, verticalStack, Fintype.sum_sum_type]

/-- The actual augmented ridge design has Gram `dᵀd + lambda I`. -/
def augmentedDesign (d : Matrix n p ℝ) (lam : ℝ) : Matrix (n ⊕ p) p ℝ :=
  verticalStack d (Real.sqrt lam • (1 : Matrix p p ℝ))

omit [DecidableEq n] in
theorem augmentedDesign_gram (d : Matrix n p ℝ) (lam : ℝ) (hlam : 0 ≤ lam) :
    (augmentedDesign d lam)ᵀ * augmentedDesign d lam = dᵀ * d + lam • 1 := by
  unfold augmentedDesign
  rw [verticalStack_gram, Matrix.transpose_smul, Matrix.transpose_one, Matrix.smul_mul,
    Matrix.mul_smul, smul_smul, Matrix.one_mul, Real.mul_self_sqrt hlam]

/-- Orthogonality of the exact reduced QR factor gives its normal precision.
This derives the normal equations from the source's actual augmented design. -/
theorem augmentedQR_precision (L K : Matrix n n ℝ) (D : Matrix n p ℝ) (lam : ℝ)
    (Q : Matrix (n ⊕ p) p ℝ) (R : Matrix p p ℝ)
    (hL : L * Lᵀ = K) (hlam : 0 ≤ lam)
    (hQR : augmentedDesign (L⁻¹ * D) lam = Q * R) (hQ : Qᵀ * Q = 1) :
    Rᵀ * R = precision D K lam := by
  have hgram : (augmentedDesign (L⁻¹ * D) lam)ᵀ *
      augmentedDesign (L⁻¹ * D) lam = Rᵀ * R := by
    rw [hQR, Matrix.transpose_mul]
    calc
      (Rᵀ * Qᵀ) * (Q * R) = Rᵀ * (Qᵀ * Q) * R := by simp [Matrix.mul_assoc]
      _ = Rᵀ * R := by rw [hQ, Matrix.mul_one]
  rw [← hgram, augmentedDesign_gram _ lam hlam,
    whitening_cross_product L K D D hL]
  rfl

theorem augmentedQR_rowCov (L K : Matrix n n ℝ) (D : Matrix n p ℝ) (lam : ℝ)
    (Q : Matrix (n ⊕ p) p ℝ) (R : Matrix p p ℝ)
    (hL : L * Lᵀ = K) (hlam : 0 ≤ lam)
    (hQR : augmentedDesign (L⁻¹ * D) lam = Q * R) (hQ : Qᵀ * Q = 1) :
    R⁻¹ * (R⁻¹)ᵀ = rowCov D K lam :=
  normalFactor_rowCov D K lam R (augmentedQR_precision L K D lam Q R hL hlam hQR hQ)

omit [Fintype c] [DecidableEq c] in
/-- The exact source calculation `row_cov @ d.T @ observed` equals the normal
equation mean after the declared Cholesky and augmented QR factorizations. -/
theorem augmentedQR_mean (L K : Matrix n n ℝ) (D : Matrix n p ℝ)
    (Y : Matrix n c ℝ) (lam : ℝ) (Q : Matrix (n ⊕ p) p ℝ) (R : Matrix p p ℝ)
    (hL : L * Lᵀ = K) (hlam : 0 ≤ lam)
    (hQR : augmentedDesign (L⁻¹ * D) lam = Q * R) (hQ : Qᵀ * Q = 1) :
    (R⁻¹ * (R⁻¹)ᵀ) * (L⁻¹ * D)ᵀ * (L⁻¹ * Y) = mean D Y K lam := by
  rw [augmentedQR_rowCov L K D lam Q R hL hlam hQR hQ,
    Matrix.mul_assoc, whitening_cross_product L K D Y hL]
  simp only [mean, Matrix.mul_assoc]

omit [DecidableEq n] [DecidableEq c] in
theorem gauge_preserves_legal_covariances (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ)
    (lam a : ℝ) (hK : K.PosDef) (hSigma : Sigma.PosDef) (hlam : 0 < lam) (ha : 0 < a) :
    (a • K).PosDef ∧ (a⁻¹ • Sigma).PosDef ∧ 0 < lam / a :=
  ⟨hK.smul ha, hSigma.smul (inv_pos.mpr ha), div_pos hlam ha⟩

/-- The Cholesky solve used for whitening has a genuine inverse for legal K. -/
theorem whiteningFactor_unit (L K : Matrix n n ℝ)
    (hL : L * Lᵀ = K) (hK : K.PosDef) : IsUnit L.det := by
  have hprod : IsUnit (L.det * L.det) := by
    simpa only [← hL, Matrix.det_mul, Matrix.det_transpose] using
      positive_definite_det_unit hK
  exact isUnit_of_mul_isUnit_left hprod

/-- The augmented upper QR solve is nonsingular even if D has dependent columns,
because the positive ridge term was included before the factorization. -/
theorem augmentedQR_unit (L K : Matrix n n ℝ) (D : Matrix n p ℝ) (lam : ℝ)
    (Q : Matrix (n ⊕ p) p ℝ) (R : Matrix p p ℝ)
    (hL : L * Lᵀ = K) (hQR : augmentedDesign (L⁻¹ * D) lam = Q * R)
    (hQ : Qᵀ * Q = 1) (hK : K.PosDef) (hlam : 0 < lam) : IsUnit R.det :=
  normalFactor_unit D K lam R
    (augmentedQR_precision L K D lam Q R hL hlam.le hQR hQ) hK hlam

/-- Exact Cholesky whitening computes the quadratic term used by the source. -/
theorem cholesky_quadratic (L C : Matrix n n ℝ) (r : n → ℝ) (hL : L * Lᵀ = C) :
    quadratic C r = (L⁻¹ *ᵥ r) ⬝ᵥ (L⁻¹ *ᵥ r) := by
  unfold quadratic
  rw [← hL, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv,
    ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]

/-- A lower-triangular Cholesky factor with positive diagonal gives exactly
the `2 * log(diag(root)).sum()` expression used in `log_evidence`. -/
theorem cholesky_logDet [LinearOrder n] (L C : Matrix n n ℝ)
    (hL : L * Lᵀ = C) (htri : L.BlockTriangular OrderDual.toDual)
    (hdiag : ∀ i, 0 < L i i) :
    Real.log C.det = 2 * ∑ i, Real.log (L i i) := by
  have hd : L.det ≠ 0 := by
    rw [Matrix.det_of_lowerTriangular L htri]
    exact Finset.prod_ne_zero_iff.mpr (fun i _ => ne_of_gt (hdiag i))
  rw [← hL, Matrix.det_mul, Matrix.det_transpose, Real.log_mul hd hd,
    Matrix.det_of_lowerTriangular L htri,
    Real.log_prod Finset.univ (fun i => L i i) (fun i _ => ne_of_gt (hdiag i))]
  ring

/-- The actual exact-arithmetic dense Cholesky score equals the normal-equation
score whose gauge and voltage rules were proved above. -/
theorem cholesky_logEvidence [LinearOrder n] (L C : Matrix n n ℝ) (r : n → ℝ)
    (hL : L * Lᵀ = C) (htri : L.BlockTriangular OrderDual.toDual)
    (hdiag : ∀ i, 0 < L i i) :
    -(1 / 2 : ℝ) * ((L⁻¹ *ᵥ r) ⬝ᵥ (L⁻¹ *ᵥ r) +
      2 * ∑ i, Real.log (L i i)) = logEvidence C r := by
  rw [logEvidence, cholesky_quadratic L C r hL, cholesky_logDet L C hL htri hdiag]

end

end N2P3.ScaleGauge
