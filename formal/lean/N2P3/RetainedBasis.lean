import N2P3.NuisanceProjection
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# A genuine orthonormal complement of the actual thin QR columns

Derive an orthonormal basis extension from `Qᵀ Q = I`, preserving each given
column of Q exactly and in its original index order. Its remaining columns
form B. Neither B's existence nor completeness is assumed.

This uses the pinned Mathlib finite-dimensional orthonormal-extension theorem.
The construction is mathematical and noncomputable; a numerical QR completion
algorithm and its error bounds remain separate implementation obligations.
-/

namespace N2P3.RetainedBasis

open Matrix
open scoped BigOperators Matrix

noncomputable section

variable {n q : Type*} [Fintype n] [Fintype q] [DecidableEq n] [DecidableEq q]

/-- Matrix columns in the Euclidean inner-product space, rather than the
default supremum-norm function space. -/
def euclideanColumns (Q : Matrix n q ℝ) : q → EuclideanSpace ℝ n :=
  fun j => WithLp.toLp 2 (fun i => Q i j)

omit [Fintype q] [DecidableEq n] in
theorem thin_qr_columns_orthonormal (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    Orthonormal ℝ (euclideanColumns Q) := by
  rw [orthonormal_iff_ite]
  intro i j
  have h := congrArg (fun A : Matrix q q ℝ => A i j) hQ
  simpa [euclideanColumns, EuclideanSpace.inner_eq_star_dotProduct, Matrix.mul_apply,
    Matrix.transpose_apply, Matrix.one_apply, dotProduct, mul_comm] using h

omit [DecidableEq n] in
/-- The given orthonormal columns really fit in the ambient dimension. -/
theorem thin_qr_dimension_le (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    Fintype.card q ≤ Fintype.card n := by
  simpa using (thin_qr_columns_orthonormal Q hQ).linearIndependent.fintype_card_le_finrank

/-- Coordinate matrix of an arbitrary Euclidean orthonormal basis. -/
def basisMatrix {ι : Type*} [Fintype ι]
    (b : OrthonormalBasis ι ℝ (EuclideanSpace ℝ n)) : Matrix n ι ℝ :=
  fun i j => b j i

omit [DecidableEq n] in
theorem basisMatrix_columns_gram {ι : Type*} [Fintype ι] [DecidableEq ι]
    (b : OrthonormalBasis ι ℝ (EuclideanSpace ℝ n)) :
    (basisMatrix b)ᵀ * basisMatrix b = 1 := by
  ext i j
  have h := b.inner_eq_ite i j
  simpa [basisMatrix, EuclideanSpace.inner_eq_star_dotProduct, Matrix.mul_apply,
    Matrix.transpose_apply, Matrix.one_apply, dotProduct, mul_comm] using h

/-- Completeness of a genuine orthonormal basis gives the other Gram identity,
including when the basis uses a different finite index type from n. -/
theorem basisMatrix_rows_complete {ι : Type*} [Fintype ι]
    (b : OrthonormalBasis ι ℝ (EuclideanSpace ℝ n)) :
    basisMatrix b * (basisMatrix b)ᵀ = 1 := by
  ext i j
  have h := b.sum_inner_mul_inner (EuclideanSpace.single i 1) (EuclideanSpace.single j 1)
  simpa [basisMatrix, Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
    EuclideanSpace.inner_single_left, EuclideanSpace.inner_single_right,
    EuclideanSpace.single_apply, eq_comm] using h

omit [DecidableEq n] in
/-- Extend into q plus exactly n-q new coordinates. The left block agrees with
the supplied Q, not merely with a basis of the same nuisance span. -/
theorem exists_orthonormal_extension (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    ∃ b : OrthonormalBasis (q ⊕ Fin (Fintype.card n - Fintype.card q)) ℝ
        (EuclideanSpace ℝ n),
      ∀ j, b (Sum.inl j) = euclideanColumns Q j := by
  let m := Fintype.card n - Fintype.card q
  let ι := q ⊕ Fin m
  let v : ι → EuclideanSpace ℝ n := Sum.elim (euclideanColumns Q) (fun _ => 0)
  let s : Set ι := Set.range (Sum.inl : q → ι)
  have hdim : Module.finrank ℝ (EuclideanSpace ℝ n) = Fintype.card ι := by
    have hle := thin_qr_dimension_le Q hQ
    dsimp only [ι]
    simp only [finrank_euclideanSpace, Fintype.card_sum, Fintype.card_fin]
    dsimp [m]
    omega
  have hv : Orthonormal ℝ (s.restrict v) := by
    rw [orthonormal_iff_ite]
    intro i j
    rcases i with ⟨i, hi⟩
    rcases j with ⟨j, hj⟩
    obtain ⟨i, rfl⟩ := hi
    obtain ⟨j, rfl⟩ := hj
    have hcond : ((Sum.inl i : ι) = Sum.inl j) ↔ i = j :=
      ⟨Sum.inl.inj, congrArg Sum.inl⟩
    simpa only [Set.restrict_apply, v, Sum.elim_inl, Subtype.mk.injEq, hcond] using
      (orthonormal_iff_ite.mp (thin_qr_columns_orthonormal Q hQ) i j)
  obtain ⟨b, hb⟩ := hv.exists_orthonormalBasis_extension_of_card_eq hdim
  refine ⟨b, ?_⟩
  intro j
  exact hb (Sum.inl j) ⟨j, rfl⟩

/-- Nonredundant retained coordinates exist for every exact thin QR matrix.
All three matrix identities are derived from the actual basis extension. -/
theorem exists_retained_basis (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    ∃ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ + Q * Qᵀ = 1 ∧
        B * Bᵀ = NuisanceProjection.projection Q := by
  obtain ⟨b, hb⟩ := exists_orthonormal_extension Q hQ
  let B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ :=
    fun i j => b (Sum.inr j) i
  have hleft (i : n) (j : q) : b (Sum.inl j) i = Q i j := by
    exact congrArg (fun x : EuclideanSpace ℝ n => x i) (hb j)
  have hB : Bᵀ * B = 1 := by
    ext i j
    have h := b.inner_eq_ite (Sum.inr i) (Sum.inr j)
    simpa [B, EuclideanSpace.inner_eq_star_dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Matrix.one_apply, dotProduct, mul_comm] using h
  have hcomplete : B * Bᵀ + Q * Qᵀ = 1 := by
    have h := basisMatrix_rows_complete b
    ext i j
    have hij := congrArg (fun A : Matrix n n ℝ => A i j) h
    simpa [basisMatrix, Matrix.mul_apply, Matrix.transpose_apply, Matrix.add_apply,
      Fintype.sum_sum_type, B, hleft, add_comm] using hij
  exact ⟨B, hB, hcomplete,
    NuisanceProjection.complete_frame_identifies_projection Q B hcomplete⟩

omit [DecidableEq n] in
/-- The complement dimension is forced by QᵀQ=I; it is not a free basis size. -/
theorem retained_dimension (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    Fintype.card n = (Fintype.card n - Fintype.card q) + Fintype.card q := by
  have hle := thin_qr_dimension_le Q hQ
  omega

/-- This specializes the existence theorem to the source's two QR columns. -/
theorem exists_rank_two_retained_basis (Q : Matrix n (Fin 2) ℝ) (hQ : Qᵀ * Q = 1) :
    ∃ B : Matrix n (Fin (Fintype.card n - 2)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ = NuisanceProjection.projection Q := by
  obtain ⟨B, hB, _, hP⟩ := exists_retained_basis Q hQ
  exact ⟨B, hB, hP⟩

/-- The accepted retained-space scoring theorem now has an actual basis witness
derived from Q; no existence/completeness premise is passed in by the caller. -/
theorem exists_actual_retained_score_coordinates
    {r c : Type*} [Fintype r] [Fintype c] [DecidableEq r]
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (R : Matrix n c ℝ) (hR : NuisanceProjection.projection Q * R = R) :
    ∃ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ = NuisanceProjection.projection Q ∧
        (NuisanceProjection.reducedObservationCovariance Q B U).PosDef ∧
          GaussianLowRank.fullCandidateScore U R =
            NuisanceProjection.reducedCandidateScore Q B U R := by
  obtain ⟨B, hB, _, hP⟩ := exists_retained_basis Q hQ
  exact ⟨B, hB, hP,
    NuisanceProjection.nonredundant_retained_covariance_posDef Q B hB hP U,
    NuisanceProjection.identity_extension_score_equals_retained_score Q B hB hP U hU R hR⟩

end

end N2P3.RetainedBasis
