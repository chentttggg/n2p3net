import N2P3.RetainedBasis
import N2P3.GaussianMeasure

/-!
# Basis-independent volume on the actual retained observations

Two nonredundant orthonormal bases of the same retained space differ by an
orthogonal coordinate matrix. Its determinant has absolute value one, so the
actual Lebesgue matrix-transformation theorem proves that their pushforward
coordinate volumes agree. A basis is then supplied by RetainedBasis from Q.

This constructs a nonzero measure concentrated on P x=x. It does not assign
ambient full-dimensional Lebesgue density to the singular projected data.
Gaussian conditional/disintegration and numerical-QR claims remain separate.
-/

namespace N2P3.RetainedMeasure

open Matrix MeasureTheory
open scoped Matrix BigOperators ENNReal

noncomputable section

variable {n q m : Type*} [Fintype n] [Fintype q] [Fintype m]
variable [DecidableEq n] [DecidableEq m]

def basisChange (B C : Matrix n m ℝ) : Matrix m m ℝ := Cᵀ * B

theorem basis_range_projection (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hP : B * Bᵀ = NuisanceProjection.projection Q) :
    NuisanceProjection.projection Q * B = B := by
  rw [← hP, Matrix.mul_assoc, hB, Matrix.mul_one]

/-- The coordinate change follows from the complete frames, without assuming
that the two bases are already related by a desired change matrix. -/
theorem basis_change_reconstructs (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q) :
    C * basisChange B C = B := by
  rw [basisChange, ← Matrix.mul_assoc, hPC, basis_range_projection Q B hB hPB]

theorem basis_change_orthogonal (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hC : Cᵀ * C = 1)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q) :
    (basisChange B C)ᵀ * basisChange B C = 1 ∧
      basisChange B C * (basisChange B C)ᵀ = 1 := by
  constructor
  · calc
      _ = Bᵀ * (C * Cᵀ) * B := by
        simp only [basisChange, Matrix.transpose_mul, Matrix.transpose_transpose,
          Matrix.mul_assoc]
      _ = Bᵀ * (NuisanceProjection.projection Q * B) := by
        rw [hPC, Matrix.mul_assoc]
      _ = 1 := by rw [basis_range_projection Q B hB hPB, hB]
  · calc
      _ = Cᵀ * (B * Bᵀ) * C := by
        simp only [basisChange, Matrix.transpose_mul, Matrix.transpose_transpose,
          Matrix.mul_assoc]
      _ = Cᵀ * (NuisanceProjection.projection Q * C) := by
        rw [hPB, Matrix.mul_assoc]
      _ = 1 := by rw [basis_range_projection Q C hC hPC, hC]

omit [DecidableEq m] in
theorem retained_coordinates_change (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (x : n → ℝ) (hx : NuisanceProjection.projection Q *ᵥ x = x) :
    basisChange B C *ᵥ (Bᵀ *ᵥ x) = Cᵀ *ᵥ x := by
  have hmatrix : basisChange B C * Bᵀ = Cᵀ * NuisanceProjection.projection Q := by
    rw [basisChange, Matrix.mul_assoc, hPB]
  rw [Matrix.mulVec_mulVec, hmatrix, ← Matrix.mulVec_mulVec, hx]

/-- Orthogonality gives an actual unit-volume Jacobian. -/
theorem orthogonal_abs_det_one (O : Matrix m m ℝ) (hO : Oᵀ * O = 1) : |O.det| = 1 := by
  have h := congrArg Matrix.det hO
  rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
  have hs : O.det ^ 2 = 1 := by nlinarith [h]
  have ha : |O.det| ^ 2 = 1 := by simpa only [sq_abs] using hs
  nlinarith [abs_nonneg O.det]

omit [DecidableEq n] in
theorem matrix_embedding_measurable (B : Matrix n m ℝ) : Measurable (Matrix.toLin' B) :=
  (Matrix.toLin' B).continuous_on_pi.measurable

/-- This is the Lebesgue pushforward equation, rather than a formal determinant
identity with no connection to the observation measure. -/
theorem orthogonal_change_preserves_volume (O : Matrix m m ℝ) (hO : Oᵀ * O = 1) :
    Measure.map (Matrix.toLin' O) volume = (volume : Measure (m → ℝ)) := by
  have ha := orthogonal_abs_det_one O hO
  have hdet : O.det ≠ 0 := by
    intro hz
    rw [hz, abs_zero] at ha
    exact zero_ne_one ha
  have h := Real.map_matrix_volume_pi_eq_smul_volume_pi hdet
  simpa only [abs_inv, ha, inv_one, ENNReal.ofReal_one, one_smul] using h

/-- Coordinate volume embedded in the original N-row array space. It is a
measure on the retained subspace, not the ambient volume measure. -/
def coordinateVolume (B : Matrix n m ℝ) : Measure (n → ℝ) :=
  Measure.map (Matrix.toLin' B) (volume : Measure (m → ℝ))

omit [DecidableEq n] in
theorem coordinateVolume_ne_zero (B : Matrix n m ℝ) : coordinateVolume B ≠ 0 := by
  intro hzero
  have hz : (volume : Measure (m → ℝ)) = 0 :=
    (Measure.map_eq_zero_iff
      ((matrix_embedding_measurable B).aemeasurable :
        AEMeasurable (Matrix.toLin' B) (volume : Measure (m → ℝ)))).mp hzero
  exact (NeZero.ne (volume : Measure (m → ℝ))) hz

/-- All nonredundant orthonormal bases of one retained space induce the same
actual embedded volume. No measure-invariance hypothesis is assumed. -/
theorem coordinateVolume_basis_independent (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hC : Cᵀ * C = 1)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q) :
    coordinateVolume B = coordinateVolume C := by
  have hO := (basis_change_orthogonal Q B C hB hC hPB hPC).1
  have hmap := orthogonal_change_preserves_volume (basisChange B C) hO
  have hcompose : Matrix.toLin' B =
      (Matrix.toLin' C) ∘ (Matrix.toLin' (basisChange B C)) := by
    funext x
    simp only [Function.comp_apply, Matrix.toLin'_apply, Matrix.mulVec_mulVec,
      basis_change_reconstructs Q B C hB hPB hPC]
  unfold coordinateVolume
  rw [hcompose, ← Measure.map_map (matrix_embedding_measurable C)
    (matrix_embedding_measurable (basisChange B C)), hmap]

def retainedSupport (Q : Matrix n q ℝ) : Set (n → ℝ) :=
  {x | NuisanceProjection.projection Q *ᵥ x = x}

theorem retainedSupport_measurable (Q : Matrix n q ℝ) : MeasurableSet (retainedSupport Q) := by
  exact (isClosed_eq (Matrix.toLin' (NuisanceProjection.projection Q)).continuous_on_pi
    continuous_id).measurableSet

/-- The embedded volume puts no mass outside the actual retained support. -/
theorem coordinateVolume_supported (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q) :
    coordinateVolume B (retainedSupport Q)ᶜ = 0 := by
  have hPB' := basis_range_projection Q B hB hPB
  have hpre : (Matrix.toLin' B) ⁻¹' (retainedSupport Q)ᶜ = ∅ := by
    ext x
    simp [retainedSupport, Matrix.toLin'_apply, Matrix.mulVec_mulVec, hPB']
  rw [coordinateVolume, Measure.map_apply (matrix_embedding_measurable B)
    (retainedSupport_measurable Q).compl, hpre, measure_empty]

/-- Q alone supplies a genuine, nonzero, basis-independent retained volume:
the basis existence theorem is used to construct its witness. -/
theorem exists_intrinsic_retained_volume [DecidableEq q]
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1) :
    ∃ ν : Measure (n → ℝ), ν ≠ 0 ∧ ν (retainedSupport Q)ᶜ = 0 ∧
      ∀ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
        Bᵀ * B = 1 → B * Bᵀ = NuisanceProjection.projection Q → coordinateVolume B = ν := by
  obtain ⟨B, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Q hQ
  refine ⟨coordinateVolume B, coordinateVolume_ne_zero B,
    coordinateVolume_supported Q B hB hPB, ?_⟩
  intro C hC hPC
  exact coordinateVolume_basis_independent Q C B hC hB hPC hPB

end

end N2P3.RetainedMeasure
