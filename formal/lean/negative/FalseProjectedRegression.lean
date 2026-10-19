import N2P3.ProjectedRegression
import Mathlib.Tactic.FinCases

/-!
Two false source-fit invariance controls. The frames themselves are genuine
orthonormal complete complements; the source input deliberately violates
P D=D (and P Y=Y). This isolates why fit invariance requires actual pre-fit
projection rather than dropping coordinates from arbitrary ambient data.
Each example must end with exactly one False goal and no auxiliary error.
-/

namespace N2P3.ProjectedRegression.Negative

open Matrix
open N2P3.NuisanceProjection
open scoped Matrix BigOperators ComplexOrder

private def Q : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 1 else 0
private def B : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 1 then 1 else 0

private theorem actual_frames_legal : Qᵀ * Q = 1 ∧ Bᵀ * B = 1 ∧ B * Bᵀ = projection Q := by
  constructor
  · ext i j
    fin_cases i
    fin_cases j
    norm_num [Q, Matrix.mul_apply, Fin.sum_univ_succ]
  constructor
  · ext i j
    fin_cases i
    fin_cases j
    norm_num [B, Matrix.mul_apply, Fin.sum_univ_succ]
  · ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [B, Q, projection, Matrix.mul_apply, Fin.sum_univ_succ]

private theorem discarded_design_not_retained : projection Q * Q ≠ Q := by
  rw [projection_annihilates_qr_columns Q actual_frames_legal.1]
  intro h
  have hv := congrArg (fun D : Matrix (Fin 2) (Fin 1) ℝ => D 0 0) h
  norm_num [Q] at hv

private theorem original_rowCov_value : sourceRowCov Q 1 = ((1 / 2 : ℝ) • 1) := by
  unfold sourceRowCov sourcePrecision
  rw [actual_frames_legal.1]
  ext i j
  fin_cases i
  fin_cases j
  norm_num [Matrix.inv_subsingleton, Matrix.diagonal, Matrix.add_apply, Matrix.one_apply,
    Matrix.ofNat_apply, Ring.inverse_eq_inv]

private theorem compressed_design_zero : Bᵀ * Q = 0 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [B, Q, Matrix.mul_apply, Fin.sum_univ_succ]

private theorem compressed_rowCov_value : sourceRowCov (Bᵀ * Q) 1 = 1 := by
  rw [compressed_design_zero]
  simp [sourceRowCov, sourcePrecision]

private theorem original_mean_value : sourceMean Q Q 1 = ((1 / 2 : ℝ) • 1) := by
  unfold sourceMean
  rw [original_rowCov_value, Matrix.mul_assoc, actual_frames_legal.1, Matrix.mul_one]

private theorem compressed_mean_value : sourceMean (Bᵀ * Q) (Bᵀ * Q) 1 = 0 := by
  rw [compressed_design_zero]
  simp [sourceMean]

/-- False: nonredundant coordinates alone preserve a fit made on unprojected D. -/
example : sourceRowCov (Bᵀ * Q) 1 = sourceRowCov Q 1 := by
  rw [compressed_rowCov_value, original_rowCov_value]
  have hne : (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ (1 / 2 : ℝ) • 1 := by
    intro h
    have hv := congrArg (fun A : Matrix (Fin 1) (Fin 1) ℝ => A 0 0) h
    norm_num at hv
  simp only [hne]

/-- False: the ambient ridge mean is unchanged after deleting unprojected signal. -/
example : sourceMean (Bᵀ * Q) (Bᵀ * Q) 1 = sourceMean Q Q 1 := by
  rw [compressed_mean_value, original_mean_value]
  have hne : (0 : Matrix (Fin 1) (Fin 1) ℝ) ≠ (1 / 2 : ℝ) • 1 := by
    intro h
    have hv := congrArg (fun A : Matrix (Fin 1) (Fin 1) ℝ => A 0 0) h
    norm_num at hv
  simp only [hne]

end N2P3.ProjectedRegression.Negative
