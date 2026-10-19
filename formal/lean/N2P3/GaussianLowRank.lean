import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Ring

/-!
# Exact low-rank algebra for the continuous candidate scorer (V03.a prerequisite)

The source `ContinuousDecisionModel.score` uses U = D @ kernel_root and
S = I + U.T @ U. After spatial whitening, C = I + U @ U.T is the identity
extension of row covariance on the nuisance-projected observation subspace.
The small Cholesky solve subtracts a squared projection from
the residual sum of squares and uses `channels * 2 * sum(log(diag(root)))`.

The results below identify this computation with the full covariance inverse
and determinant over exact real arithmetic, for arbitrary finite dimensions.
No invertibility is guessed: I + U Uᵀ and I + Uᵀ U are proved strictly positive
definite even if U is rank deficient. A supplied Cholesky factor must satisfy
its factor equation and, for the diagonal logarithm expression, be lower
triangular with strictly positive diagonal.

This is algebra conditional on fixed, legally fitted covariance, aligned
whitened observations, and an exact factor/solve. It does not yet derive a
Gaussian probability density or verify whitening, the rank-two nuisance
projection/support/Jacobian, Bayesian conditioning,
NumPy/SciPy floating-point Cholesky, the numerical negative-energy tolerance,
or whole-project implementation refinement. The retained channel count and
candidate-dependent log determinant are explicit rather than discarded.
-/

namespace N2P3.GaussianLowRank

open Matrix
open scoped BigOperators Matrix ComplexOrder

variable {n r c : Type*} [Fintype n] [Fintype r] [Fintype c]
  [DecidableEq n] [DecidableEq r]

def fullCovariance (U : Matrix n r ℝ) : Matrix n n ℝ := 1 + U * Uᵀ

def smallCovariance (U : Matrix n r ℝ) : Matrix r r ℝ := 1 + Uᵀ * U

omit [Fintype c] [DecidableEq r] in
theorem fullCovariance_posDef (U : Matrix n r ℝ) : (fullCovariance U).PosDef := by
  exact Matrix.PosDef.one.add_posSemidef (by
    simpa only [conjTranspose_eq_transpose_of_trivial] using
      Matrix.posSemidef_self_mul_conjTranspose U)

omit [Fintype c] [DecidableEq n] in
theorem smallCovariance_posDef (U : Matrix n r ℝ) : (smallCovariance U).PosDef := by
  exact Matrix.PosDef.one.add_posSemidef (by
    simpa only [conjTranspose_eq_transpose_of_trivial] using
      Matrix.posSemidef_conjTranspose_mul_self U)

omit [Fintype c] [DecidableEq r] in
theorem fullCovariance_det_pos (U : Matrix n r ℝ) : 0 < (fullCovariance U).det :=
  (fullCovariance_posDef U).det_pos

omit [Fintype c] [DecidableEq n] in
theorem smallCovariance_det_pos (U : Matrix n r ℝ) : 0 < (smallCovariance U).det :=
  (smallCovariance_posDef U).det_pos

omit [Fintype c] in
/-- Exact Woodbury identity; the small inverse is derived from SPD, not assumed. -/
theorem full_inverse_equals_low_rank (U : Matrix n r ℝ) :
    (fullCovariance U)⁻¹ = 1 - U * (smallCovariance U)⁻¹ * Uᵀ := by
  have hs : IsUnit ((1 : Matrix r r ℝ)⁻¹ + Uᵀ * (1 : Matrix n n ℝ)⁻¹ * U) := by
    simpa [smallCovariance] using (smallCovariance_posDef U).isUnit
  simpa [fullCovariance, smallCovariance] using
    Matrix.add_mul_mul_inv_eq_sub (1 : Matrix n n ℝ) U (1 : Matrix r r ℝ) Uᵀ
      isUnit_one isUnit_one hs

omit [Fintype c] in
/-- Sylvester's determinant identity preserves the candidate-dependent term. -/
theorem full_det_equals_small_det (U : Matrix n r ℝ) :
    (fullCovariance U).det = (smallCovariance U).det :=
  Matrix.det_one_add_mul_comm U Uᵀ

omit [Fintype c] in
/-- One observed channel's full Mahalanobis quadratic equals the small solve. -/
theorem full_quadratic_equals_low_rank (U : Matrix n r ℝ) (x : n → ℝ) :
    x ⬝ᵥ ((fullCovariance U)⁻¹ *ᵥ x) =
      x ⬝ᵥ x - (Uᵀ *ᵥ x) ⬝ᵥ ((smallCovariance U)⁻¹ *ᵥ (Uᵀ *ᵥ x)) := by
  rw [full_inverse_equals_low_rank, Matrix.sub_mulVec, Matrix.one_mulVec, dotProduct_sub]
  congr 1
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, dotProduct_mulVec,
    ← Matrix.mulVec_transpose]

omit [Fintype c] in
theorem exact_mahalanobis_nonnegative (U : Matrix n r ℝ) (x : n → ℝ) :
    0 ≤ x ⬝ᵥ x - (Uᵀ *ᵥ x) ⬝ᵥ ((smallCovariance U)⁻¹ *ᵥ (Uᵀ *ᵥ x)) := by
  rw [← full_quadratic_equals_low_rank]
  simpa only [star_trivial] using (fullCovariance_posDef U).inv.posSemidef.2 x

omit [Fintype c] in
/-- Clamping exact arithmetic at zero has no effect; floating-point stability is separate. -/
theorem exact_mahalanobis_clamp_is_identity (U : Matrix n r ℝ) (x : n → ℝ) :
    max 0 (x ⬝ᵥ x - (Uᵀ *ᵥ x) ⬝ᵥ ((smallCovariance U)⁻¹ *ᵥ (Uᵀ *ᵥ x))) =
      x ⬝ᵥ x - (Uᵀ *ᵥ x) ⬝ᵥ ((smallCovariance U)⁻¹ *ᵥ (Uᵀ *ᵥ x)) :=
  max_eq_right (exact_mahalanobis_nonnegative U x)

omit [Fintype c] [DecidableEq n] in
/-- Any exact factor of the SPD small matrix is nonsingular. -/
theorem exact_small_factor_isUnit (U : Matrix n r ℝ) (L : Matrix r r ℝ)
    (hfactor : L * Lᵀ = smallCovariance U) : IsUnit L := by
  apply (Matrix.isUnit_iff_isUnit_det L).mpr
  apply isUnit_iff_ne_zero.mpr
  intro hzero
  have hd := smallCovariance_det_pos U
  rw [← hfactor, Matrix.det_mul, Matrix.det_transpose, hzero, zero_mul] at hd
  exact lt_irrefl 0 hd

omit [Fintype c] [DecidableEq n] in
/-- The Euclidean energy of an exact triangular solve equals the small quadratic.
Triangularity is not needed for this identity; the factor equation suffices. -/
theorem factor_solve_energy (U : Matrix n r ℝ) (L : Matrix r r ℝ)
    (hfactor : L * Lᵀ = smallCovariance U) (z : r → ℝ) :
    (L⁻¹ *ᵥ z) ⬝ᵥ (L⁻¹ *ᵥ z) = z ⬝ᵥ ((smallCovariance U)⁻¹ *ᵥ z) := by
  symm
  rw [← hfactor, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv,
    ← Matrix.mulVec_mulVec, dotProduct_mulVec z (L⁻¹)ᵀ (L⁻¹ *ᵥ z),
    Matrix.vecMul_transpose]

omit [Fintype c] in
theorem full_quadratic_equals_factor_projection (U : Matrix n r ℝ)
    (L : Matrix r r ℝ) (hfactor : L * Lᵀ = smallCovariance U) (x : n → ℝ) :
    x ⬝ᵥ ((fullCovariance U)⁻¹ *ᵥ x) =
      x ⬝ᵥ x - (L⁻¹ *ᵥ (Uᵀ *ᵥ x)) ⬝ᵥ (L⁻¹ *ᵥ (Uᵀ *ᵥ x)) := by
  rw [factor_solve_energy U L hfactor, full_quadratic_equals_low_rank]

/-- Source residual/projected matrices include every observed channel column. -/
theorem full_multichannel_energy_equals_projection (U : Matrix n r ℝ)
    (L : Matrix r r ℝ) (hfactor : L * Lᵀ = smallCovariance U) (R : Matrix n c ℝ) :
    (∑ channel, (fun i => R i channel) ⬝ᵥ
      ((fullCovariance U)⁻¹ *ᵥ (fun i => R i channel))) =
    (∑ channel, (fun i => R i channel) ⬝ᵥ (fun i => R i channel)) -
      ∑ channel, (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel))) ⬝ᵥ
        (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel))) := by
  simp_rw [full_quadratic_equals_factor_projection U L hfactor]
  rw [Finset.sum_sub_distrib]

theorem multichannel_projection_energy_nonnegative (U : Matrix n r ℝ)
    (L : Matrix r r ℝ) (hfactor : L * Lᵀ = smallCovariance U) (R : Matrix n c ℝ) :
    0 ≤ (∑ channel, (fun i => R i channel) ⬝ᵥ (fun i => R i channel)) -
      ∑ channel, (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel))) ⬝ᵥ
        (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel))) := by
  rw [← full_multichannel_energy_equals_projection U L hfactor R]
  apply Finset.sum_nonneg
  intro channel _
  simpa only [star_trivial] using
    (fullCovariance_posDef U).inv.posSemidef.2 (fun i => R i channel)

omit [Fintype c] in
theorem full_channel_logdet_equals_small (U : Matrix n r ℝ) (channels : ℕ) :
    Real.log ((fullCovariance U).det ^ channels) =
      (channels : ℝ) * Real.log (smallCovariance U).det := by
  rw [full_det_equals_small_det, Real.log_pow]

omit [Fintype c] [DecidableEq n] in
/-- The factor-product determinant identity is independent of triangularity. -/
theorem small_det_equals_factor_det_squared (U : Matrix n r ℝ) (L : Matrix r r ℝ)
    (hfactor : L * Lᵀ = smallCovariance U) :
    (smallCovariance U).det = L.det ^ 2 := by
  rw [← hfactor, Matrix.det_mul, Matrix.det_transpose, pow_two]

omit [Fintype c] in
theorem full_logdet_equals_factor_logdet (U : Matrix n r ℝ) (L : Matrix r r ℝ)
    (hfactor : L * Lᵀ = smallCovariance U) :
    Real.log (fullCovariance U).det = 2 * Real.log L.det := by
  rw [full_det_equals_small_det, small_det_equals_factor_det_squared U L hfactor,
    Real.log_pow]
  norm_num

omit [Fintype c] in
/-- Exact match for the source's sum of log Cholesky diagonal entries. -/
theorem factor_logdet_equals_diagonal_sum [LinearOrder r]
    (L : Matrix r r ℝ) (htriangular : L.BlockTriangular OrderDual.toDual)
    (hdiag : ∀ i, 0 < L i i) : Real.log L.det = ∑ i, Real.log (L i i) := by
  rw [Matrix.det_of_lowerTriangular L htriangular]
  exact Real.log_prod Finset.univ (fun i => L i i) (fun i _ => ne_of_gt (hdiag i))

omit [Fintype c] in
theorem source_channel_logdet_equals_full [LinearOrder r]
    (U : Matrix n r ℝ) (L : Matrix r r ℝ)
    (hfactor : L * Lᵀ = smallCovariance U)
    (htriangular : L.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < L i i)
    (channels : ℕ) :
    (channels : ℝ) * 2 * (∑ i, Real.log (L i i)) =
      Real.log ((fullCovariance U).det ^ channels) := by
  rw [Real.log_pow, full_logdet_equals_factor_logdet U L hfactor,
    factor_logdet_equals_diagonal_sum L htriangular hdiag]
  ring

/-- The declared exact-real candidate score with every actual residual channel. -/
noncomputable def fullCandidateScore (U : Matrix n r ℝ) (R : Matrix n c ℝ) : ℝ :=
  -(1 / 2 : ℝ) * ((∑ channel, (fun i => R i channel) ⬝ᵥ
    ((fullCovariance U)⁻¹ *ᵥ (fun i => R i channel))) +
      (Fintype.card c : ℝ) * Real.log (fullCovariance U).det)

/-- Exact counterpart of residual**2 minus projected**2 and the Cholesky logdet. -/
noncomputable def sourceCandidateScore (U : Matrix n r ℝ) (L : Matrix r r ℝ) (R : Matrix n c ℝ) : ℝ :=
  -(1 / 2 : ℝ) * (max 0 ((∑ channel, (fun i => R i channel) ⬝ᵥ (fun i => R i channel)) -
    ∑ channel, (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel))) ⬝ᵥ
      (L⁻¹ *ᵥ (Uᵀ *ᵥ (fun i => R i channel)))) +
        (Fintype.card c : ℝ) * 2 * (∑ i, Real.log (L i i)))

/-- The channel multiplier is tied to the actual column type, not a free scalar.
This is an algebraic score identity on the declared identity extension. -/
theorem source_candidate_score_equals_full [LinearOrder r]
    (U : Matrix n r ℝ) (L : Matrix r r ℝ) (R : Matrix n c ℝ)
    (hfactor : L * Lᵀ = smallCovariance U)
    (htriangular : L.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < L i i) :
    sourceCandidateScore U L R = fullCandidateScore U R := by
  have hlog := source_channel_logdet_equals_full U L hfactor htriangular hdiag (Fintype.card c)
  rw [Real.log_pow] at hlog
  unfold sourceCandidateScore fullCandidateScore
  rw [max_eq_right (multichannel_projection_energy_nonnegative U L hfactor R),
    hlog, full_multichannel_energy_equals_projection U L hfactor R]

omit [Fintype c] [Fintype n] [DecidableEq r] in
/-- The ambient SPD matrix replaces the projected covariance's missing
nuisance block with identity. This alone does not claim ambient density. -/
theorem full_is_identity_extension (U : Matrix n r ℝ) (P : Matrix n n ℝ) :
    (P + U * Uᵀ) + (1 - P) = fullCovariance U := by
  unfold fullCovariance
  abel

omit [Fintype c] [DecidableEq r] in
/-- On vectors actually retained by the nuisance projector, both covariance
operators act identically. No orthogonal-complement observations are invented. -/
theorem identity_extension_agrees_on_retained_vectors
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (x : n → ℝ) (hx : P *ᵥ x = x) :
    (P + U * Uᵀ) *ᵥ x = fullCovariance U *ᵥ x := by
  simp only [fullCovariance, Matrix.add_mulVec, Matrix.one_mulVec, hx]

omit [Fintype c] in
/-- The low-rank inverse preserves the retained range when the design and
residual belong to that range. This is the key solve-support obligation. -/
theorem inverse_preserves_projected_range
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hU : P * U = U)
    (x : n → ℝ) (hx : P *ᵥ x = x) :
    P *ᵥ ((fullCovariance U)⁻¹ *ᵥ x) = (fullCovariance U)⁻¹ *ᵥ x := by
  rw [full_inverse_equals_low_rank]
  simp only [Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.mulVec_sub]
  rw [hx]
  congr 1
  simp only [← Matrix.mulVec_mulVec]
  rw [Matrix.mulVec_mulVec, hU]

omit [Fintype c] in
/-- The ambient inverse therefore solves the singular projected equation for
retained right-hand sides; it does not invert that singular matrix globally. -/
theorem identity_extension_inverse_solves_projected_equation
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hU : P * U = U)
    (x : n → ℝ) (hx : P *ᵥ x = x) :
    (P + U * Uᵀ) *ᵥ ((fullCovariance U)⁻¹ *ᵥ x) = x := by
  rw [identity_extension_agrees_on_retained_vectors U P _
    (inverse_preserves_projected_range U P hU x hx)]
  rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv]
  · exact Matrix.one_mulVec x
  · exact isUnit_iff_ne_zero.mpr (ne_of_gt (fullCovariance_det_pos U))

omit [Fintype c] [DecidableEq r] in
/-- The added nuisance block has eigenvalue one, provided P is a projection
and the design lies in its range. Symmetry supplies the right-side statement. -/
theorem identity_extension_on_nuisance_complement
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hU : P * U = U) :
    (1 - P) * fullCovariance U = 1 - P := by
  have hzero : (1 - P) * U = 0 := by simp [Matrix.sub_mul, hU]
  simp only [fullCovariance, Matrix.mul_add, Matrix.mul_one, ← Matrix.mul_assoc,
    hzero, Matrix.zero_mul, add_zero]

omit [Fintype c] [Fintype r] [DecidableEq n] [DecidableEq r] in
theorem transpose_design_stays_in_symmetric_projection
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hP : Pᵀ = P) (hU : P * U = U) :
    Uᵀ * P = Uᵀ := by
  have h := congrArg Matrix.transpose hU
  simpa only [Matrix.transpose_mul, hP] using h

omit [Fintype c] [DecidableEq r] in
/-- The identity extension contributes exactly unit covariance on the discarded
orthogonal complement; both cross-blocks vanish for an orthogonal projector. -/
theorem identity_extension_right_nuisance_complement
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hP : Pᵀ = P) (hU : P * U = U) :
    fullCovariance U * (1 - P) = 1 - P := by
  have hzero : Uᵀ * (1 - P) = 0 := by
    simp [Matrix.mul_sub, transpose_design_stays_in_symmetric_projection U P hP hU]
  simp only [fullCovariance, Matrix.add_mul, Matrix.one_mul, Matrix.mul_assoc,
    hzero, Matrix.mul_zero, add_zero]

omit [Fintype c] [DecidableEq r] in
/-- Compression to the actual observed subspace yields P+UUᵀ, not I+UUᵀ.
The latter's determinant is therefore an identity extension determinant. -/
theorem projected_covariance_is_compressed_full
    (U : Matrix n r ℝ) (P : Matrix n n ℝ) (hP : Pᵀ = P)
    (hidempotent : P * P = P) (hU : P * U = U) :
    P * fullCovariance U * P = P + U * Uᵀ := by
  simp only [fullCovariance, Matrix.mul_add, Matrix.mul_one, ← Matrix.mul_assoc, hU,
    Matrix.add_mul, hidempotent]
  rw [Matrix.mul_assoc, transpose_design_stays_in_symmetric_projection U P hP hU]

omit [Fintype c] in
/-- Candidate-dependent covariance cannot be removed as a common score term:
even a zero residual distinguishes determinants 1 and 2. -/
theorem removing_logdet_can_create_false_tie :
    (- (1 / 2 : ℝ) * (0 + Real.log 1) = 0) ∧
      (- (1 / 2 : ℝ) * (0 + Real.log 2) < 0) := by
  constructor
  · norm_num
  · have h : 0 < Real.log 2 := Real.log_pos (by norm_num)
    exact mul_neg_of_neg_of_pos (by norm_num) (by simpa using h)

end N2P3.GaussianLowRank
