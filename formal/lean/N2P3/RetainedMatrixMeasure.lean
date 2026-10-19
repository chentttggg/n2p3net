import N2P3.RetainedMeasure
import N2P3.MatrixNormalCoordinates

/-!
# Retained volume for the actual rows-by-channels observation coordinates

Lift every time-row operator with the channel identity. The resulting product
coordinates are the row-major convention already connected to NumPy storage.
The nonzero volume is concentrated on P E = E and independent of the chosen
orthonormal time basis. It is not the ambient Lebesgue volume of a singular
projected observation, and no conditional probability law is claimed here.
-/

namespace N2P3.RetainedMatrixMeasure

open Matrix MeasureTheory
open N2P3.RetainedMeasure N2P3.MatrixNormalCoordinates
open scoped Matrix BigOperators Kronecker

noncomputable section

variable {n q m c : Type*} [Fintype n] [Fintype q] [Fintype m] [Fintype c]
variable [DecidableEq n] [DecidableEq q] [DecidableEq m] [DecidableEq c]

def channelLift {a b : Type*} (A : Matrix a b ℝ) : Matrix (a × c) (b × c) ℝ :=
  A ⊗ₖ (1 : Matrix c c ℝ)

omit [Fintype n] [Fintype c] in
theorem channelLift_one : channelLift (1 : Matrix n n ℝ) = (1 : Matrix (n × c) (n × c) ℝ) :=
  Matrix.one_kronecker_one

omit [Fintype n] [Fintype m] [Fintype c] [DecidableEq n] [DecidableEq m] in
theorem channelLift_transpose (A : Matrix n m ℝ) :
    (channelLift (c := c) A)ᵀ = channelLift Aᵀ := by
  unfold channelLift
  rw [← Matrix.kroneckerMap_transpose, Matrix.transpose_one]

omit [Fintype n] [Fintype q] [DecidableEq n] [DecidableEq q] [DecidableEq m] in
theorem channelLift_mul (A : Matrix n m ℝ) (B : Matrix m q ℝ) :
    channelLift (c := c) (A * B) = channelLift A * channelLift B := by
  unfold channelLift
  rw [← Matrix.mul_kronecker_mul, Matrix.one_mul]

omit [Fintype n] [Fintype c] [DecidableEq n] in
theorem channelLift_sub (A B : Matrix n n ℝ) :
    channelLift (c := c) (A - B) = channelLift A - channelLift B := by
  ext ⟨i, a⟩ ⟨j, b⟩
  simp only [channelLift, Matrix.kronecker_apply, Matrix.sub_apply]
  ring

omit [Fintype n] [DecidableEq n] [DecidableEq m] in
/-- This action uses the exact row-major matrix coordinate map. -/
theorem channelLift_vectorize (B : Matrix n m ℝ) (E : Matrix m c ℝ) :
    channelLift B *ᵥ vectorize E = vectorize (B * E) := by
  ext ⟨i, a⟩
  simp [channelLift, Matrix.mulVec, dotProduct, vectorize,
    Matrix.mul_apply, Matrix.one_apply, Fintype.sum_prod_type]

omit [Fintype m] [DecidableEq n] in
theorem channelLift_gram (B : Matrix n m ℝ) (hB : Bᵀ * B = 1) :
    (channelLift (c := c) B)ᵀ * channelLift (c := c) B =
      (1 : Matrix (m × c) (m × c) ℝ) := by
  rw [channelLift_transpose, ← channelLift_mul, hB, channelLift_one]

omit [Fintype n] [DecidableEq q] in
/-- The lifted nuisance projection is the actual P acting on each channel. -/
theorem channelLift_projection (Q : Matrix n q ℝ) :
    NuisanceProjection.projection (channelLift (c := c) Q) =
      channelLift (NuisanceProjection.projection Q) := by
  unfold NuisanceProjection.projection
  rw [channelLift_transpose, ← channelLift_mul, channelLift_sub, channelLift_one]

omit [Fintype n] [DecidableEq q] [DecidableEq m] in
theorem channelLift_complete (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q) :
    channelLift (c := c) B * (channelLift B)ᵀ =
      NuisanceProjection.projection (channelLift Q) := by
  rw [channelLift_transpose, ← channelLift_mul, hPB, channelLift_projection]

omit [DecidableEq q] in
theorem matrix_retained_support_iff (Q : Matrix n q ℝ) (E : Matrix n c ℝ) :
    vectorize E ∈ retainedSupport (channelLift Q) ↔
      NuisanceProjection.projection Q * E = E := by
  change NuisanceProjection.projection (channelLift Q) *ᵥ vectorize E = vectorize E ↔ _
  rw [channelLift_projection, channelLift_vectorize]
  constructor
  · intro h
    exact congrArg matricize h
  · intro h
    exact congrArg vectorize h

omit [DecidableEq q] in
/-- The actual multichannel coordinate volume is independent of time-basis
choice, including nontrivial channel covariance models using these coordinates. -/
theorem matrixCoordinateVolume_basis_independent
    (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hC : Cᵀ * C = 1)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q) :
    coordinateVolume (channelLift (c := c) B) = coordinateVolume (channelLift C) :=
  coordinateVolume_basis_independent (channelLift Q) (channelLift B) (channelLift C)
    (channelLift_gram B hB) (channelLift_gram C hC)
    (channelLift_complete Q B hPB) (channelLift_complete Q C hPC)

omit [DecidableEq q] in
theorem matrixCoordinateVolume_supported
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q) :
    coordinateVolume (channelLift (c := c) B)
      {x | NuisanceProjection.projection Q * matricize x ≠ matricize x} = 0 := by
  have hset : {x : (n × c) → ℝ |
      NuisanceProjection.projection Q * matricize x ≠ matricize x} =
      (retainedSupport (channelLift (c := c) Q))ᶜ := by
    ext x
    simpa only [Set.mem_setOf_eq, Set.mem_compl_iff] using
      not_congr (matrix_retained_support_iff Q (matricize x)).symm
  rw [hset]
  exact coordinateVolume_supported (channelLift Q) (channelLift B)
    (channelLift_gram B hB) (channelLift_complete Q B hPB)

/-- An actual multichannel retained-volume witness is obtained from Q alone.
No existence of the time basis or multichannel invariance is assumed. -/
theorem exists_intrinsic_matrix_retained_volume
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    ∃ ν : Measure ((n × c) → ℝ), ν ≠ 0 ∧
      ν {x | NuisanceProjection.projection Q * matricize x ≠ matricize x} = 0 ∧
      ∀ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
        Bᵀ * B = 1 → B * Bᵀ = NuisanceProjection.projection Q →
          coordinateVolume (channelLift B) = ν := by
  obtain ⟨B, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Q hQ
  refine ⟨coordinateVolume (channelLift B), coordinateVolume_ne_zero _,
    matrixCoordinateVolume_supported Q B hB hPB, ?_⟩
  intro C hC hPC
  exact matrixCoordinateVolume_basis_independent Q C B hC hB hPC hPB

end

end N2P3.RetainedMatrixMeasure
