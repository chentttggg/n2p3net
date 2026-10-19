import N2P3.ScaleGauge
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Abel

/-!
# Exact conditional Gaussian kernel algebra

Complete the square in the actual multichannel regularized residual energy.
The mean and row precision are derived from the fixed-covariance normal
equations proved in ScaleGauge. Trace weighting by the frozen channel precision
then gives an exponential factor independent of the kernel coefficients.

These are unnormalized kernel identities. Probability normalization, a joint
measure construction and conditional-probability/disintegration remain separate
obligations. The pinned Mathlib snapshot provides real Gaussian densities and
the Banach-space `IsGaussian` predicate, but no multivariate density constructor.
-/

namespace N2P3.GaussianConditioning

open Matrix
open N2P3.ScaleGauge
open scoped BigOperators Matrix ComplexOrder

noncomputable section

variable {n p c : Type*} [Fintype n] [Fintype p] [Fintype c]
variable [DecidableEq n] [DecidableEq p] [DecidableEq c]

/-- Actual row/channel matrix energy before channel precision weighting. -/
def energy (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (W : Matrix p c ℝ) : Matrix c c ℝ :=
  (Y - D * W)ᵀ * K⁻¹ * (Y - D * W) + lam • (Wᵀ * W)

def centeredEnergy (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (W : Matrix p c ℝ) : Matrix c c ℝ :=
  (W - mean D Y K lam)ᵀ * precision D K lam * (W - mean D Y K lam)

def constantEnergy (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) : Matrix c c ℝ :=
  Yᵀ * K⁻¹ * Y - (mean D Y K lam)ᵀ * precision D K lam * mean D Y K lam

theorem precision_symmetric (D : Matrix n p ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) :
    (precision D K lam)ᵀ = precision D K lam := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    (precision_positive_definite D K lam hK hlam).isHermitian.eq

theorem inverse_symmetric (K : Matrix n n ℝ) (hK : K.PosDef) : K⁻¹ᵀ = K⁻¹ := by
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using hK.inv.isHermitian.eq

omit [Fintype c] [DecidableEq c] in
/-- The defined mean solves the actual normal equations; invertibility follows
from positive K and positive ridge, including rank-deficient designs. -/
theorem mean_normal_equation (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) :
    precision D K lam * mean D Y K lam = Dᵀ * K⁻¹ * Y := by
  have hM := positive_definite_det_unit (precision_positive_definite D K lam hK hlam)
  simpa only [mean, rowCov, Matrix.mul_assoc] using
    Matrix.mul_nonsing_inv_cancel_left (precision D K lam) (Dᵀ * K⁻¹ * Y) hM

omit [Fintype c] [DecidableEq c] in
theorem mean_transpose_normal_equation (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) :
    (mean D Y K lam)ᵀ * precision D K lam = Yᵀ * K⁻¹ * D := by
  have ht := congrArg Matrix.transpose (mean_normal_equation D Y K lam hK hlam)
  simpa only [Matrix.transpose_mul, Matrix.transpose_transpose,
    precision_symmetric D K lam hK hlam, inverse_symmetric K hK, Matrix.mul_assoc] using ht

omit [Fintype c] [DecidableEq c] in
theorem energy_expansion (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (lam : ℝ) (W : Matrix p c ℝ) :
    energy D Y K lam W = Yᵀ * K⁻¹ * Y - Yᵀ * K⁻¹ * D * W -
      Wᵀ * Dᵀ * K⁻¹ * Y + Wᵀ * precision D K lam * W := by
  simp only [energy, precision, Matrix.transpose_sub, Matrix.transpose_mul,
    Matrix.sub_mul, Matrix.mul_sub, Matrix.add_mul, Matrix.mul_add,
    Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_one, Matrix.mul_assoc]
  abel

omit [Fintype c] [DecidableEq c] in
theorem centeredEnergy_expansion (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    centeredEnergy D Y K lam W = Wᵀ * precision D K lam * W - Yᵀ * K⁻¹ * D * W -
      Wᵀ * Dᵀ * K⁻¹ * Y +
        (mean D Y K lam)ᵀ * precision D K lam * mean D Y K lam := by
  simp only [centeredEnergy, Matrix.transpose_sub, Matrix.sub_mul, Matrix.mul_sub]
  simp only [mean_transpose_normal_equation D Y K lam hK hlam, Matrix.mul_assoc,
    mean_normal_equation D Y K lam hK hlam]
  abel

omit [Fintype c] [DecidableEq c] in
/-- Matrix completion of the square, before taking traces. -/
theorem completing_square (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    energy D Y K lam W = centeredEnergy D Y K lam W + constantEnergy D Y K lam := by
  rw [energy_expansion, centeredEnergy_expansion D Y K lam W hK hlam]
  unfold constantEnergy
  abel

omit [Fintype c] [DecidableEq c] in
/-- Expanded statement in the notation of the source's Gaussian kernel. -/
theorem completing_square_explicit (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    (Y - D * W)ᵀ * K⁻¹ * (Y - D * W) + lam • (Wᵀ * W) =
      (W - mean D Y K lam)ᵀ * precision D K lam * (W - mean D Y K lam) +
        Yᵀ * K⁻¹ * Y -
          (mean D Y K lam)ᵀ * precision D K lam * mean D Y K lam := by
  have h := completing_square D Y K lam W hK hlam
  simpa only [energy, centeredEnergy, constantEnergy, sub_eq_add_neg, add_assoc] using h

def weightedEnergy (Sigma : Matrix c c ℝ) (E : Matrix c c ℝ) : ℝ :=
  Matrix.trace (Sigma⁻¹ * E)

theorem trace_completing_square (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    weightedEnergy Sigma (energy D Y K lam W) =
      weightedEnergy Sigma (centeredEnergy D Y K lam W) +
        weightedEnergy Sigma (constantEnergy D Y K lam) := by
  rw [completing_square D Y K lam W hK hlam]
  simp only [weightedEnergy, Matrix.mul_add, Matrix.trace_add]

def jointKernel (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma (energy D Y K lam W))

def centeredKernel (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma (centeredEnergy D Y K lam W))

/-- The W-independent factor is derived from completing the square. It is not
assumed to be an evidence integral or probability normalizer. -/
theorem exponential_kernel_factorization (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    jointKernel D Y K Sigma lam W =
      Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma (constantEnergy D Y K lam)) *
        centeredKernel D Y K Sigma lam W := by
  unfold jointKernel centeredKernel
  rw [trace_completing_square D Y K Sigma lam W hK hlam, mul_add, Real.exp_add]
  exact mul_comm _ _

omit [Fintype c] [DecidableEq c] in
theorem centeredEnergy_at_mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) :
    centeredEnergy D Y K lam (mean D Y K lam) = 0 := by
  simp only [centeredEnergy, sub_self, Matrix.transpose_zero, Matrix.zero_mul, Matrix.mul_zero]

omit [Fintype c] [DecidableEq c] in
theorem constantEnergy_is_energy_at_mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (hK : K.PosDef) (hlam : 0 < lam) :
    constantEnergy D Y K lam = energy D Y K lam (mean D Y K lam) := by
  rw [completing_square D Y K lam _ hK hlam, centeredEnergy_at_mean, zero_add]

theorem centeredKernel_at_mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) :
    centeredKernel D Y K Sigma lam (mean D Y K lam) = 1 := by
  simp only [centeredKernel, centeredEnergy_at_mean, weightedEnergy,
    Matrix.trace_zero, mul_zero, Real.exp_zero]

omit [DecidableEq p] in
theorem jointKernel_positive (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ) :
    0 < jointKernel D Y K Sigma lam W := Real.exp_pos _

/-- Ratio of kernel values at W and at the derived mean. -/
theorem relative_kernel (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    jointKernel D Y K Sigma lam W / jointKernel D Y K Sigma lam (mean D Y K lam) =
      centeredKernel D Y K Sigma lam W := by
  rw [exponential_kernel_factorization D Y K Sigma lam W hK hlam,
    exponential_kernel_factorization D Y K Sigma lam (mean D Y K lam) hK hlam,
    centeredKernel_at_mean, mul_one]
  exact mul_div_cancel_left₀ _ (Real.exp_ne_zero _)

omit [DecidableEq c] in
theorem centeredEnergy_positive_semidefinite (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) : (centeredEnergy D Y K lam W).PosSemidef := by
  simpa only [centeredEnergy, Matrix.conjTranspose_eq_transpose_of_trivial] using
    (precision_positive_definite D K lam hK hlam).posSemidef.conjTranspose_mul_mul_same
      (W - mean D Y K lam)

omit [Fintype c] [DecidableEq c] in
/-- The center is the unique solution of the proved normal equations. -/
theorem normal_solution_unique (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam)
    (hW : precision D K lam * W = Dᵀ * K⁻¹ * Y) : W = mean D Y K lam := by
  have hM := positive_definite_det_unit (precision_positive_definite D K lam hK hlam)
  calc
    W = (precision D K lam)⁻¹ * (precision D K lam * W) :=
      (Matrix.nonsing_inv_mul_cancel_left _ W hM).symm
    _ = (precision D K lam)⁻¹ * (Dᵀ * K⁻¹ * Y) := by rw [hW]
    _ = mean D Y K lam := by simp only [mean, rowCov, Matrix.mul_assoc]

def likelihoodKernel (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma : Matrix c c ℝ) (W : Matrix p c ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma ((Y - D * W)ᵀ * K⁻¹ * (Y - D * W)))

def priorKernel (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * lam * weightedEnergy Sigma (Wᵀ * W))

omit [DecidableEq p] in
/-- Multiplying the likelihood and ridge-prior kernels gives exactly the
joint energy used by the completing-square theorem. -/
theorem likelihood_prior_kernel_product (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ) :
    likelihoodKernel D Y K Sigma W * priorKernel Sigma lam W =
      jointKernel D Y K Sigma lam W := by
  unfold likelihoodKernel priorKernel jointKernel energy weightedEnergy
  rw [Matrix.mul_add, Matrix.trace_add, Matrix.mul_smul, Matrix.trace_smul,
    smul_eq_mul, ← Real.exp_add]
  congr 1
  ring

/-- Algebraic Bayesian kernel update: the prior precision is lambda I and the
likelihood row precision is K^-1. Normalized probabilities are not assumed. -/
theorem likelihood_prior_completed_kernel (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (Sigma : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) :
    likelihoodKernel D Y K Sigma W * priorKernel Sigma lam W =
      Real.exp (-(1 / 2 : ℝ) * weightedEnergy Sigma (constantEnergy D Y K lam)) *
        centeredKernel D Y K Sigma lam W := by
  rw [likelihood_prior_kernel_product, exponential_kernel_factorization D Y K Sigma lam W hK hlam]

/-- Spatial Cholesky whitening turns the weighted trace into a PSD congruence.
The factor equation is the exact solver specification used by the source. -/
theorem weightedEnergy_as_congruence (Sigma S E : Matrix c c ℝ) (hS : S * Sᵀ = Sigma) :
    weightedEnergy Sigma E = Matrix.trace (S⁻¹ * E * (S⁻¹)ᵀ) := by
  unfold weightedEnergy
  rw [← hS, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv]
  simpa only [Matrix.mul_assoc] using Matrix.trace_mul_comm (S⁻¹)ᵀ (S⁻¹ * E)

theorem weightedEnergy_nonnegative_of_factor (Sigma S E : Matrix c c ℝ)
    (hS : S * Sᵀ = Sigma) (hE : E.PosSemidef) : 0 ≤ weightedEnergy Sigma E := by
  rw [weightedEnergy_as_congruence Sigma S E hS]
  have h := hE.mul_mul_conjTranspose_same S⁻¹
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using h.trace_nonneg

theorem centered_weightedEnergy_nonnegative_of_factor
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma S : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) (hS : S * Sᵀ = Sigma) :
    0 ≤ weightedEnergy Sigma (centeredEnergy D Y K lam W) :=
  weightedEnergy_nonnegative_of_factor Sigma S _ hS
    (centeredEnergy_positive_semidefinite D Y K lam W hK hlam)

/-- The derived center minimizes the exact weighted residual-plus-prior energy. -/
theorem weightedEnergy_minimized_at_mean_of_factor
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma S : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) (hS : S * Sᵀ = Sigma) :
    weightedEnergy Sigma (energy D Y K lam (mean D Y K lam)) ≤
      weightedEnergy Sigma (energy D Y K lam W) := by
  rw [← constantEnergy_is_energy_at_mean D Y K lam hK hlam,
    trace_completing_square D Y K Sigma lam W hK hlam]
  exact le_add_of_nonneg_left (centered_weightedEnergy_nonnegative_of_factor
    D Y K Sigma S lam W hK hlam hS)

theorem centeredKernel_le_one_of_factor
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma S : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) (hS : S * Sᵀ = Sigma) :
    centeredKernel D Y K Sigma lam W ≤ 1 := by
  unfold centeredKernel
  apply Real.exp_le_one_iff.mpr
  have h := centered_weightedEnergy_nonnegative_of_factor D Y K Sigma S lam W hK hlam hS
  linarith

/-- A kernel mode follows without calling the unnormalized kernel a measure. -/
theorem jointKernel_maximized_at_mean_of_factor
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (K : Matrix n n ℝ)
    (Sigma S : Matrix c c ℝ) (lam : ℝ) (W : Matrix p c ℝ)
    (hK : K.PosDef) (hlam : 0 < lam) (hS : S * Sᵀ = Sigma) :
    jointKernel D Y K Sigma lam W ≤ jointKernel D Y K Sigma lam (mean D Y K lam) := by
  unfold jointKernel
  apply Real.exp_le_exp.mpr
  have h := weightedEnergy_minimized_at_mean_of_factor D Y K Sigma S lam W hK hlam hS
  linarith

end

end N2P3.GaussianConditioning
