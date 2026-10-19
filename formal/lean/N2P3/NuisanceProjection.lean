import N2P3.GaussianLowRank
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega

/-!
# Thin-QR nuisance projection and the retained observation coordinates

`continuous_decision._whiten_and_project` returns W X - Q(Qᵀ W X), preserving N
array rows while deleting the whitened constant/trend subspace. The first part
of this file derives the projector from the exact thin-QR equation QᵀQ=I; it
does not assume that the projected covariance is ambient full rank. Exact QR
factorization/orthonormality are algorithmic input contracts here, not a proof
about NumPy's numerical QR implementation.

An orthonormal complementary frame B identifies the actual retained coordinates.
We prove equality of the identity-extension determinant, quadratic and candidate
score with their reduced-coordinate versions. Existence/computation of B and
measure/density/Jacobian assertions are separate obligations.
-/

namespace N2P3.NuisanceProjection

open Matrix
open scoped Matrix BigOperators ComplexOrder

section ThinQR

variable {n q c r : Type*} [Fintype n] [Fintype q] [Fintype c] [Fintype r]
  [DecidableEq n] [DecidableEq q]

/-- Orthogonal-complement candidate constructed from the actual thin QR columns. -/
def projection (Q : Matrix n q ℝ) : Matrix n n ℝ := 1 - Q * Qᵀ

omit [Fintype c] [Fintype r] [Fintype n] [DecidableEq q] in
theorem projection_symmetric (Q : Matrix n q ℝ) : (projection Q)ᵀ = projection Q := by
  simp [projection, Matrix.transpose_sub, Matrix.transpose_mul]

omit [Fintype c] [Fintype r] [DecidableEq n] in
theorem thin_qr_range_idempotent (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    (Q * Qᵀ) * (Q * Qᵀ) = Q * Qᵀ := by
  calc
    _ = Q * (Qᵀ * Q) * Qᵀ := by simp only [Matrix.mul_assoc]
    _ = Q * Qᵀ := by rw [hQ, Matrix.mul_one]

omit [Fintype c] [Fintype r] in
theorem projection_idempotent (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    projection Q * projection Q = projection Q := by
  have hh := thin_qr_range_idempotent Q hQ
  simp [projection, Matrix.sub_mul, Matrix.mul_sub, hh]

omit [Fintype c] [Fintype r] in
theorem projection_annihilates_qr_columns (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    projection Q * Q = 0 := by
  simp [projection, Matrix.sub_mul, Matrix.mul_assoc, hQ]

omit [Fintype c] [Fintype r] in
theorem qr_columns_annihilate_projection (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    Qᵀ * projection Q = 0 := by
  simp [projection, Matrix.mul_sub, ← Matrix.mul_assoc, hQ]

omit [Fintype r] [Fintype c] [DecidableEq q] in
/-- Exact lowering of the expression returned by `_whiten_and_project`. -/
theorem whiten_project_expression (W : Matrix n n ℝ) (Q : Matrix n q ℝ)
    (X : Matrix n c ℝ) :
    W * X - Q * (Qᵀ * (W * X)) = projection Q * W * X := by
  simp [projection, Matrix.sub_mul, Matrix.mul_assoc]

omit [Fintype r] [Fintype c] in
theorem projected_input_is_retained (W : Matrix n n ℝ) (Q : Matrix n q ℝ)
    (hQ : Qᵀ * Q = 1) (X : Matrix n c ℝ) :
    projection Q * (projection Q * W * X) = projection Q * W * X := by
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, projection_idempotent Q hQ]

omit [Fintype c] in
/-- Both projected data and projected design enter the residual, so support is
preserved after subtracting a fitted ERP mean and multiplying a kernel root. -/
theorem residual_and_lowrank_design_are_retained
    (W : Matrix n n ℝ) (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (X : Matrix n c ℝ) (D : Matrix n r ℝ) (H : Matrix r c ℝ) (L : Matrix r r ℝ) :
    projection Q * (projection Q * W * X - (projection Q * W * D) * H) =
      projection Q * W * X - (projection Q * W * D) * H ∧
    projection Q * ((projection Q * W * D) * L) = (projection Q * W * D) * L := by
  constructor
  · rw [Matrix.mul_sub, projected_input_is_retained W Q hQ X]
    rw [← Matrix.mul_assoc, projected_input_is_retained W Q hQ D]
  · rw [← Matrix.mul_assoc, projected_input_is_retained W Q hQ D]

omit [Fintype r] in
/-- An exact QR factorization supplies the column span to remove. -/
theorem projection_removes_whitened_nuisance
    (W : Matrix n n ℝ) (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (N : Matrix n q ℝ) (T : Matrix q q ℝ) (hQR : W * N = Q * T) :
    projection Q * W * N = 0 := by
  rw [Matrix.mul_assoc, hQR, ← Matrix.mul_assoc,
    projection_annihilates_qr_columns Q hQ, Matrix.zero_mul]

omit [Fintype r] [Fintype c] in
/-- Changing only the fitted nuisance coefficients leaves the projected input
unchanged; the conclusion follows from the QR equations, not an invariance premise. -/
theorem nuisance_addition_does_not_change_projected_input
    (W : Matrix n n ℝ) (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (N : Matrix n q ℝ) (T : Matrix q q ℝ) (hQR : W * N = Q * T)
    (X : Matrix n c ℝ) (coefficients : Matrix q c ℝ) :
    projection Q * W * (X + N * coefficients) = projection Q * W * X := by
  rw [Matrix.mul_add, ← Matrix.mul_assoc,
    projection_removes_whitened_nuisance W Q hQ N T hQR, Matrix.zero_mul, add_zero]

omit [Fintype c] [Fintype r] in
theorem retained_design_orthogonal_to_nuisance
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) (U : Matrix n r ℝ)
    (hU : projection Q * U = U) : Uᵀ * Q = 0 := by
  have ht : Uᵀ * projection Q = Uᵀ :=
    GaussianLowRank.transpose_design_stays_in_symmetric_projection U (projection Q)
      (projection_symmetric Q) hU
  calc
    _ = (Uᵀ * projection Q) * Q := by rw [ht]
    _ = 0 := by rw [Matrix.mul_assoc, projection_annihilates_qr_columns Q hQ,
      Matrix.mul_zero]

omit [Fintype c] in
theorem projected_covariance_annihilates_nuisance
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) (U : Matrix n r ℝ)
    (hU : projection Q * U = U) : (projection Q + U * Uᵀ) * Q = 0 := by
  rw [Matrix.add_mul, projection_annihilates_qr_columns Q hQ,
    Matrix.mul_assoc, retained_design_orthogonal_to_nuisance Q hQ U hU,
    Matrix.mul_zero, add_zero]

omit [Fintype c] in
/-- With a nonempty nuisance coordinate, the projected covariance is singular
in ambient coordinates despite the strictly positive identity extension. -/
theorem projected_covariance_not_ambient_posDef
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) (index : q) (U : Matrix n r ℝ)
    (hU : projection Q * U = U) : ¬(projection Q + U * Uᵀ).PosDef := by
  let e : q → ℝ := Pi.single index 1
  have he : e ≠ 0 := by
    intro h
    have hvalue := congrFun h index
    simp [e] at hvalue
  have hleft : Qᵀ *ᵥ (Q *ᵥ e) = e := by
    rw [Matrix.mulVec_mulVec, hQ, Matrix.one_mulVec]
  have hvector : Q *ᵥ e ≠ 0 := by
    intro hz
    rw [hz, Matrix.mulVec_zero] at hleft
    exact he hleft.symm
  intro hpd
  have hpositive := hpd.2 (Q *ᵥ e) hvector
  rw [Matrix.mulVec_mulVec, projected_covariance_annihilates_nuisance Q hQ U hU,
    Matrix.zero_mulVec, dotProduct_zero] at hpositive
  exact lt_irrefl 0 hpositive

end ThinQR

section RetainedCoordinates

variable {n q m r c : Type*} [Fintype n] [Fintype q] [Fintype m] [Fintype r]
  [Fintype c] [DecidableEq n] [DecidableEq m] [DecidableEq r]

omit [Fintype r] [Fintype c] [Fintype n] [DecidableEq m] [DecidableEq r] in
/-- Completeness of an orthonormal frame [B Q] identifies the retained projector. -/
theorem complete_frame_identifies_projection (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hcomplete : B * Bᵀ + Q * Qᵀ = 1) : B * Bᵀ = projection Q := by
  unfold projection
  exact eq_sub_of_add_eq hcomplete

omit [Fintype r] [Fintype c] [DecidableEq r] in
/-- A complete nonredundant frame fixes the actual retained dimension. For
the source's two nuisance columns this gives m=N-2, rather than ambient N. -/
theorem complete_orthonormal_frame_dimension
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) [DecidableEq q]
    (hQ : Qᵀ * Q = 1) (hB : Bᵀ * B = 1) (hcomplete : B * Bᵀ + Q * Qᵀ = 1) :
    Fintype.card n = Fintype.card m + Fintype.card q := by
  have ht := congrArg Matrix.trace hcomplete
  rw [Matrix.trace_add, Matrix.trace_mul_comm B Bᵀ, Matrix.trace_mul_comm Q Qᵀ,
    hB, hQ] at ht
  simp only [Matrix.trace_one] at ht
  exact_mod_cast ht.symm

omit [Fintype q] [Fintype r] [Fintype c] [DecidableEq n] [DecidableEq r] in
theorem orthonormal_frame_reconstructs_coordinates (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (x : m → ℝ) : Bᵀ *ᵥ (B *ᵥ x) = x := by
  rw [Matrix.mulVec_mulVec, hB, Matrix.one_mulVec]

omit [Fintype r] [Fintype c] [DecidableEq m] [DecidableEq r] in
theorem retained_coordinates_reconstruct_vector (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hP : B * Bᵀ = projection Q) (x : n → ℝ) (hx : projection Q *ᵥ x = x) :
    B *ᵥ (Bᵀ *ᵥ x) = x := by
  rw [Matrix.mulVec_mulVec, hP, hx]

omit [Fintype r] [Fintype c] [DecidableEq m] [DecidableEq r] in
theorem retained_coordinate_dotProduct (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hP : B * Bᵀ = projection Q) (x y : n → ℝ) (hx : projection Q *ᵥ x = x) :
    (Bᵀ *ᵥ x) ⬝ᵥ (Bᵀ *ᵥ y) = x ⬝ᵥ y := by
  rw [dotProduct_mulVec, Matrix.vecMul_transpose,
    retained_coordinates_reconstruct_vector Q B hP x hx]

omit [Fintype c] [Fintype r] [DecidableEq m] [DecidableEq r] in
theorem retained_kernel_gram_equals_original (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) (hU : projection Q * U = U) :
    (Bᵀ * U)ᵀ * (Bᵀ * U) = Uᵀ * U := by
  calc
    _ = Uᵀ * ((B * Bᵀ) * U) := by simp only [Matrix.transpose_mul,
      Matrix.transpose_transpose, Matrix.mul_assoc]
    _ = Uᵀ * U := by rw [hP, hU]

omit [Fintype q] [Fintype c] [Fintype m] [DecidableEq r] in
theorem reduced_covariance_is_compression (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (U : Matrix n r ℝ) :
    Bᵀ * GaussianLowRank.fullCovariance U * B = GaussianLowRank.fullCovariance (Bᵀ * U) := by
  simp only [GaussianLowRank.fullCovariance, Matrix.mul_add, Matrix.add_mul,
    Matrix.mul_one, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  rw [hB]

omit [Fintype c] in
/-- A Parseval frame suffices for this identity-extension determinant equality.
A nonredundant observation basis additionally needs BᵀB=I, imposed below. -/
theorem identity_extension_det_equals_frame_extension_det
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hP : B * Bᵀ = projection Q)
    (U : Matrix n r ℝ) (hU : projection Q * U = U) :
    (GaussianLowRank.fullCovariance U).det =
      (GaussianLowRank.fullCovariance (Bᵀ * U)).det := by
  rw [GaussianLowRank.full_det_equals_small_det,
    GaussianLowRank.full_det_equals_small_det]
  unfold GaussianLowRank.smallCovariance
  rw [retained_kernel_gram_equals_original Q B hP U hU]

omit [Fintype c] [Fintype r] [DecidableEq m] [DecidableEq r] in
theorem retained_kernel_coordinates_preserved
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hP : B * Bᵀ = projection Q)
    (U : Matrix n r ℝ) (x : n → ℝ) (hx : projection Q *ᵥ x = x) :
    (Bᵀ * U)ᵀ *ᵥ (Bᵀ *ᵥ x) = Uᵀ *ᵥ x := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose, ← Matrix.mulVec_mulVec,
    retained_coordinates_reconstruct_vector Q B hP x hx]

omit [Fintype c] in
theorem identity_extension_quadratic_equals_frame_extension_quadratic
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hP : B * Bᵀ = projection Q)
    (U : Matrix n r ℝ) (hU : projection Q * U = U)
    (x : n → ℝ) (hx : projection Q *ᵥ x = x) :
    x ⬝ᵥ ((GaussianLowRank.fullCovariance U)⁻¹ *ᵥ x) =
      (Bᵀ *ᵥ x) ⬝ᵥ ((GaussianLowRank.fullCovariance (Bᵀ * U))⁻¹ *ᵥ (Bᵀ *ᵥ x)) := by
  rw [GaussianLowRank.full_quadratic_equals_low_rank,
    GaussianLowRank.full_quadratic_equals_low_rank,
    retained_coordinate_dotProduct Q B hP x x hx,
    retained_kernel_coordinates_preserved Q B hP U x hx]
  unfold GaussianLowRank.smallCovariance
  rw [retained_kernel_gram_equals_original Q B hP U hU]

omit [Fintype r] [Fintype c] [DecidableEq m] [DecidableEq r] in
theorem retained_column_of_retained_matrix (Q : Matrix n q ℝ) (X : Matrix n c ℝ)
    (hX : projection Q * X = X) (channel : c) :
    projection Q *ᵥ (fun i => X i channel) = (fun i => X i channel) := by
  funext i
  exact congrArg (fun A : Matrix n c ℝ => A i channel) hX

/-- Algebra for a possibly redundant Parseval frame. This is an extension
score identity and does not identify a nonsingular m-dimensional observation. -/
theorem identity_extension_score_equals_frame_extension_score
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hP : B * Bᵀ = projection Q)
    (U : Matrix n r ℝ) (hU : projection Q * U = U)
    (R : Matrix n c ℝ) (hR : projection Q * R = R) :
    GaussianLowRank.fullCandidateScore U R =
      GaussianLowRank.fullCandidateScore (Bᵀ * U) (Bᵀ * R) := by
  unfold GaussianLowRank.fullCandidateScore
  rw [identity_extension_det_equals_frame_extension_det Q B hP U hU]
  congr 2
  apply Finset.sum_congr rfl
  intro channel _
  exact identity_extension_quadratic_equals_frame_extension_quadratic Q B hP U hU
    (fun i => R i channel) (retained_column_of_retained_matrix Q R hR channel)

/-- Covariance of the actual Bᵀ observation coordinates, before any claim that
those coordinates constitute a nonredundant orthonormal basis. -/
def reducedObservationCovariance (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (U : Matrix n r ℝ) :
    Matrix m m ℝ := Bᵀ * (projection Q + U * Uᵀ) * B

omit [Fintype c] [DecidableEq r] in
/-- Both BBᵀ=P and BᵀB=I matter: the latter excludes redundant Parseval frames
and makes this the actual nonsingular retained covariance. -/
theorem nonredundant_retained_covariance_equals_extension
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) :
    reducedObservationCovariance Q B U = GaussianLowRank.fullCovariance (Bᵀ * U) := by
  have hPB : projection Q * B = B := by
    rw [← hP, Matrix.mul_assoc, hB, Matrix.mul_one]
  unfold reducedObservationCovariance
  rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc Bᵀ (projection Q) B, hPB, hB]
  simp only [GaussianLowRank.fullCovariance, Matrix.transpose_mul,
    Matrix.transpose_transpose, Matrix.mul_assoc]

omit [Fintype c] [DecidableEq r] in
theorem nonredundant_retained_covariance_posDef
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) :
    (reducedObservationCovariance Q B U).PosDef := by
  rw [nonredundant_retained_covariance_equals_extension Q B hB hP U]
  exact GaussianLowRank.fullCovariance_posDef (Bᵀ * U)

omit [Fintype c] in
theorem identity_extension_det_equals_retained_det
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) (hU : projection Q * U = U) :
    (GaussianLowRank.fullCovariance U).det = (reducedObservationCovariance Q B U).det := by
  rw [nonredundant_retained_covariance_equals_extension Q B hB hP U]
  exact identity_extension_det_equals_frame_extension_det Q B hP U hU

omit [Fintype c] in
theorem identity_extension_quadratic_equals_retained_quadratic
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) (hU : projection Q * U = U)
    (x : n → ℝ) (hx : projection Q *ᵥ x = x) :
    x ⬝ᵥ ((GaussianLowRank.fullCovariance U)⁻¹ *ᵥ x) =
      (Bᵀ *ᵥ x) ⬝ᵥ ((reducedObservationCovariance Q B U)⁻¹ *ᵥ (Bᵀ *ᵥ x)) := by
  rw [nonredundant_retained_covariance_equals_extension Q B hB hP U]
  exact identity_extension_quadratic_equals_frame_extension_quadratic Q B hP U hU x hx

noncomputable def reducedCandidateScore
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (U : Matrix n r ℝ) (R : Matrix n c ℝ) : ℝ :=
  -(1 / 2 : ℝ) * ((∑ channel, (fun i => (Bᵀ * R) i channel) ⬝ᵥ
    ((reducedObservationCovariance Q B U)⁻¹ *ᵥ (fun i => (Bᵀ * R) i channel))) +
      (Fintype.card c : ℝ) * Real.log (reducedObservationCovariance Q B U).det)

/-- This final score refers to actual nonredundant observation coordinates.
The ordinary reduced Gaussian normalizer/Jacobian still needs its measure proof. -/
theorem identity_extension_score_equals_retained_score
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hP : B * Bᵀ = projection Q) (U : Matrix n r ℝ) (hU : projection Q * U = U)
    (R : Matrix n c ℝ) (hR : projection Q * R = R) :
    GaussianLowRank.fullCandidateScore U R = reducedCandidateScore Q B U R := by
  unfold reducedCandidateScore
  rw [nonredundant_retained_covariance_equals_extension Q B hB hP U]
  exact identity_extension_score_equals_frame_extension_score Q B hP U hU R hR

end RetainedCoordinates

/-- BBᵀ=P alone may describe a redundant Parseval frame, not nonredundant
observation coordinates. The final covariance wrappers explicitly require BᵀB=I. -/
theorem parseval_frame_can_be_redundant :
    ∃ B : Matrix (Fin 1) (Fin 2) ℝ, B * Bᵀ = 1 ∧ Bᵀ * B ≠ 1 := by
  let B : Matrix (Fin 1) (Fin 2) ℝ := fun _ j => if j = 0 then 1 else 0
  refine ⟨B, ?_, ?_⟩
  · ext i j
    have hi : i = 0 := Subsingleton.elim _ _
    have hj : j = 0 := Subsingleton.elim _ _
    subst i
    subst j
    norm_num [B, Matrix.mul_apply, Fin.sum_univ_succ]
  · intro h
    have hv := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 1 1) h
    norm_num [B, Matrix.mul_apply, Fin.sum_univ_succ] at hv

section StationaryAR1

/-- The literal natural-number distance `(i-j)+(j-i)` equals |i-j|. -/
noncomputable def ar1Entry (phi : ℝ) (i j : ℕ) : ℝ :=
  phi ^ ((i - j) + (j - i)) / (1 - phi ^ 2)

theorem ar1_entry_symmetric (phi : ℝ) (i j : ℕ) : ar1Entry phi i j = ar1Entry phi j i := by
  simp only [ar1Entry, Nat.add_comm]

theorem stationary_ar1_denominator_positive (phi : ℝ) (hphi : |phi| < 1) :
    0 < 1 - phi ^ 2 := by
  obtain ⟨hlower, hupper⟩ := abs_lt.mp hphi
  nlinarith

theorem ar1_first_row_geometric (phi : ℝ) (j : ℕ) :
    ar1Entry phi 0 (j + 1) = phi * ar1Entry phi 0 j := by
  simp only [ar1Entry, Nat.zero_sub, zero_add, Nat.sub_zero, pow_succ]
  ring

theorem ar1_interior_covariance_offdiagonal (phi : ℝ) (i j : ℕ) (hij : i < j) :
    ar1Entry phi (i + 1) (j + 1) - phi * ar1Entry phi i (j + 1) -
      phi * ar1Entry phi (i + 1) j + phi ^ 2 * ar1Entry phi i j = 0 := by
  let d := j - i - 1
  have hj : j = i + d + 1 := by dsimp [d]; omega
  have hA : (i + 1 - (j + 1)) + (j + 1 - (i + 1)) = d + 1 := by omega
  have hB : (i - (j + 1)) + (j + 1 - i) = (d + 1) + 1 := by omega
  have hC : (i + 1 - j) + (j - (i + 1)) = d := by omega
  have hD : (i - j) + (j - i) = d + 1 := by omega
  simp only [ar1Entry, hA, hB, hC, hD, pow_succ]
  ring

theorem ar1_interior_covariance_diagonal (phi : ℝ) (hphi : |phi| < 1) (i : ℕ) :
    ar1Entry phi (i + 1) (i + 1) - phi * ar1Entry phi i (i + 1) -
      phi * ar1Entry phi (i + 1) i + phi ^ 2 * ar1Entry phi i i = 1 := by
  have hone : i + 1 - i = 1 := by omega
  have hzero : i - (i + 1) = 0 := by omega
  simp only [ar1Entry, Nat.sub_self, zero_add, pow_zero, hone, hzero, add_zero, pow_one]
  calc
    _ = (1 - phi ^ 2) / (1 - phi ^ 2) := by ring
    _ = 1 := div_self (ne_of_gt (stationary_ar1_denominator_positive phi hphi))

theorem ar1_interior_covariance (phi : ℝ) (hphi : |phi| < 1) (i j : ℕ) :
    ar1Entry phi (i + 1) (j + 1) - phi * ar1Entry phi i (j + 1) -
      phi * ar1Entry phi (i + 1) j + phi ^ 2 * ar1Entry phi i j =
        if i = j then 1 else 0 := by
  by_cases heq : i = j
  · subst j
    simpa using ar1_interior_covariance_diagonal phi hphi i
  · rw [if_neg heq]
    rcases lt_or_gt_of_ne heq with hlt | hgt
    · exact ar1_interior_covariance_offdiagonal phi i j hlt
    · calc
        _ = ar1Entry phi (j + 1) (i + 1) - phi * ar1Entry phi j (i + 1) -
            phi * ar1Entry phi (j + 1) i + phi ^ 2 * ar1Entry phi j i := by
              rw [ar1_entry_symmetric phi (i + 1) (j + 1),
                ar1_entry_symmetric phi i (j + 1), ar1_entry_symmetric phi (i + 1) j,
                ar1_entry_symmetric phi i j]
              ring
        _ = 0 := ar1_interior_covariance_offdiagonal phi j i hgt

/-- Exact source whitening: first sample sqrt(1-phi²)*x0; subsequent samples
xi-phi*x(i-1). The delta rows expose the actual sparse sample dependence. -/
noncomputable def ar1Whitening (phi : ℝ) (N : ℕ) : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  fun i => Fin.cases (Pi.single 0 (Real.sqrt (1 - phi ^ 2)))
    (fun k => Pi.single k.succ 1 - Pi.single k.castSucc phi) i

noncomputable def stationaryAR1Covariance (phi : ℝ) (N : ℕ) : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  fun i j => ar1Entry phi i.val j.val

theorem ar1_covariance_symmetric (phi : ℝ) (N : ℕ) :
    (stationaryAR1Covariance phi N)ᵀ = stationaryAR1Covariance phi N := by
  ext i j
  exact ar1_entry_symmetric phi j.val i.val

theorem ar1_whitening_left_first {c : Type*} (phi : ℝ) (N : ℕ)
    (X : Matrix (Fin (N + 1)) c ℝ) (channel : c) :
    (ar1Whitening phi N * X) 0 channel = Real.sqrt (1 - phi ^ 2) * X 0 channel := by
  change (Pi.single 0 (Real.sqrt (1 - phi ^ 2))) ⬝ᵥ (fun k => X k channel) = _
  rw [single_dotProduct]

theorem ar1_whitening_left_successor {c : Type*} (phi : ℝ) (N : ℕ)
    (X : Matrix (Fin (N + 1)) c ℝ) (i : Fin N) (channel : c) :
    (ar1Whitening phi N * X) i.succ channel = X i.succ channel - phi * X i.castSucc channel := by
  change (Pi.single i.succ 1 - Pi.single i.castSucc phi) ⬝ᵥ (fun k => X k channel) = _
  rw [sub_dotProduct, single_dotProduct, single_dotProduct, one_mul]

theorem ar1_whitening_right_first {c : Type*} (phi : ℝ) (N : ℕ)
    (X : Matrix c (Fin (N + 1)) ℝ) (channel : c) :
    (X * (ar1Whitening phi N)ᵀ) channel 0 = X channel 0 * Real.sqrt (1 - phi ^ 2) := by
  change (fun k => X channel k) ⬝ᵥ (Pi.single 0 (Real.sqrt (1 - phi ^ 2))) = _
  rw [dotProduct_single]

theorem ar1_whitening_right_successor {c : Type*} (phi : ℝ) (N : ℕ)
    (X : Matrix c (Fin (N + 1)) ℝ) (i : Fin N) (channel : c) :
    (X * (ar1Whitening phi N)ᵀ) channel i.succ = X channel i.succ - X channel i.castSucc * phi := by
  change (fun k => X channel k) ⬝ᵥ (Pi.single i.succ 1 - Pi.single i.castSucc phi) = _
  rw [dotProduct_sub, dotProduct_single, dotProduct_single, mul_one]

/-- Entrywise verification of the actual stationary AR(1) covariance whitening,
including the initial stationary variance. White covariance is a conclusion. -/
theorem stationary_ar1_whitening_is_identity (phi : ℝ) (hphi : |phi| < 1) (N : ℕ) :
    ar1Whitening phi N * stationaryAR1Covariance phi N * (ar1Whitening phi N)ᵀ = 1 := by
  have ha := stationary_ar1_denominator_positive phi hphi
  ext i j
  refine Fin.cases ?_ (fun i' => ?_) i
  · refine Fin.cases ?_ (fun j' => ?_) j
    · rw [ar1_whitening_right_first, ar1_whitening_left_first]
      simp only [stationaryAR1Covariance, ar1Entry, Nat.sub_self, zero_add, pow_zero,
        Matrix.one_apply_eq]
      calc
        _ = Real.sqrt (1 - phi ^ 2) ^ 2 / (1 - phi ^ 2) := by ring
        _ = 1 := by rw [Real.sq_sqrt ha.le, div_self (ne_of_gt ha)]
    · rw [ar1_whitening_right_successor, ar1_whitening_left_first,
        ar1_whitening_left_first]
      change Real.sqrt (1 - phi ^ 2) * ar1Entry phi 0 (j'.val + 1) -
        Real.sqrt (1 - phi ^ 2) * ar1Entry phi 0 j'.val * phi = _
      rw [ar1_first_row_geometric]
      have hz : (0 : Fin (N + 1)) ≠ j'.succ := fun h => Fin.succ_ne_zero j' h.symm
      simp only [Matrix.one_apply, if_neg hz]
      ring
  · refine Fin.cases ?_ (fun j' => ?_) j
    · rw [ar1_whitening_right_first, ar1_whitening_left_successor]
      change (ar1Entry phi (i'.val + 1) 0 - phi * ar1Entry phi i'.val 0) *
        Real.sqrt (1 - phi ^ 2) = _
      rw [ar1_entry_symmetric phi (i'.val + 1) 0, ar1_entry_symmetric phi i'.val 0,
        ar1_first_row_geometric]
      simp
    · rw [ar1_whitening_right_successor, ar1_whitening_left_successor,
        ar1_whitening_left_successor]
      change ar1Entry phi (i'.val + 1) (j'.val + 1) - phi * ar1Entry phi i'.val (j'.val + 1) -
        (ar1Entry phi (i'.val + 1) j'.val - phi * ar1Entry phi i'.val j'.val) * phi = _
      calc
        _ = ar1Entry phi (i'.val + 1) (j'.val + 1) - phi * ar1Entry phi i'.val (j'.val + 1) -
          phi * ar1Entry phi (i'.val + 1) j'.val + phi ^ 2 * ar1Entry phi i'.val j'.val := by ring
        _ = if i'.val = j'.val then 1 else 0 := ar1_interior_covariance phi hphi _ _
        _ = _ := by simp [Matrix.one_apply, Fin.ext_iff]

theorem stationary_ar1_whitening_isUnit (phi : ℝ) (hphi : |phi| < 1) (N : ℕ) :
    IsUnit (ar1Whitening phi N) := by
  let W := ar1Whitening phi N
  let K := stationaryAR1Covariance phi N
  have hw : W * K * Wᵀ = 1 := stationary_ar1_whitening_is_identity phi hphi N
  have hdet : W.det * (K * Wᵀ).det = 1 := by
    rw [← Matrix.det_mul, ← Matrix.mul_assoc, hw, Matrix.det_one]
  apply (Matrix.isUnit_iff_isUnit_det W).mpr
  apply isUnit_iff_ne_zero.mpr
  intro hzero
  rw [hzero, zero_mul] at hdet
  exact zero_ne_one hdet

/-- Stationary covariance positivity follows from the actual whitening identity
and nonsingularity; it is not introduced as an unverified covariance premise. -/
theorem stationary_ar1_covariance_posDef (phi : ℝ) (hphi : |phi| < 1) (N : ℕ) :
    (stationaryAR1Covariance phi N).PosDef := by
  let W := ar1Whitening phi N
  let K := stationaryAR1Covariance phi N
  have hW : IsUnit W := stationary_ar1_whitening_isUnit phi hphi N
  have hWdet : IsUnit W.det := (Matrix.isUnit_iff_isUnit_det W).mp hW
  have hWTdet : IsUnit Wᵀ.det := by rw [Matrix.det_transpose]; exact hWdet
  have hw : W * K * Wᵀ = 1 := stationary_ar1_whitening_is_identity phi hphi N
  have hKW : K * Wᵀ = W⁻¹ := by
    apply Eq.symm
    apply Matrix.inv_eq_right_inv
    rw [← Matrix.mul_assoc, hw]
  have hK : K = W⁻¹ * (W⁻¹)ᵀ := by
    calc
      K = (K * Wᵀ) * (Wᵀ)⁻¹ := (Matrix.mul_nonsing_inv_cancel_right Wᵀ K hWTdet).symm
      _ = W⁻¹ * (W⁻¹)ᵀ := by rw [hKW, Matrix.transpose_nonsing_inv]
  change K.PosDef
  rw [hK]
  have hinj : Function.Injective (W⁻¹).vecMul :=
    Matrix.vecMul_injective_iff_isUnit.mpr (Matrix.isUnit_nonsing_inv_iff.mpr hW)
  simpa only [conjTranspose_eq_transpose_of_trivial] using
    Matrix.PosDef.mul_conjTranspose_self (W⁻¹) hinj

/-- Initial AR(1) observation variance is stationary variance, not innovation
variance: using unit first-row scaling fails already at phi=1/2. -/
theorem omitting_stationary_first_scale_is_wrong : ar1Entry (1 / 2) 0 0 ≠ 1 := by
  norm_num [ar1Entry]

/-- Exact mathematical [ones, linspace(-1,1,N+1)] source nuisance dictionary. -/
noncomputable def sourceNuisance (N : ℕ) : Matrix (Fin (N + 1)) (Fin 2) ℝ :=
  fun i j => if j = 0 then 1 else -1 + 2 * (i.val : ℝ) / (N : ℝ)

/-- The actual constant/trend dictionary has two independent columns when the
record has at least two samples. No QR rank is guessed from an array shape. -/
theorem source_nuisance_columns_independent (N : ℕ) (hN : 0 < N) :
    Function.Injective (sourceNuisance N).mulVec := by
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hN
  have hlast : -1 + 2 * (N : ℝ) / (N : ℝ) = 1 := by
    rw [mul_div_cancel_right₀ 2 hn]
    norm_num
  intro a b hab
  have hfirst := congrFun hab 0
  have hfinal := congrFun hab (Fin.last N)
  simp only [Matrix.mulVec, dotProduct, Fin.sum_univ_two, sourceNuisance,
    ite_true, Fin.val_zero, Nat.cast_zero, mul_zero, zero_div, add_zero,
    one_ne_zero, if_false, one_mul, neg_mul, Fin.val_last, hlast] at hfirst hfinal
  have ha : a 0 = b 0 := by linarith
  have hb : a 1 = b 1 := by linarith
  funext i
  by_cases hi : i = 0
  · subst i; exact ha
  · have hi' : i = 1 := by apply Fin.ext; have := i.isLt; omega
    subst i
    exact hb

/-- Exact stationary AR whitening is invertible, so it preserves the dictionary
rank before thin QR. Source QR must still deliver its exact factor equations. -/
theorem whitened_source_nuisance_columns_independent
    (phi : ℝ) (hphi : |phi| < 1) (N : ℕ) (hN : 0 < N) :
    Function.Injective (ar1Whitening phi N * sourceNuisance N).mulVec := by
  intro a b hab
  apply source_nuisance_columns_independent N hN
  apply Matrix.mulVec_injective_of_isUnit (stationary_ar1_whitening_isUnit phi hphi N)
  simpa only [Matrix.mulVec_mulVec] using hab

end StationaryAR1

end N2P3.NuisanceProjection
