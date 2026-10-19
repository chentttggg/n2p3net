import N2P3.MatrixNormalCoordinates
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-! Deliberately false row-major whitening identities. Both examples must
reduce to an unsolved False goal; import or type errors do not count. -/

namespace N2P3.MatrixNormalCoordinates.Negative

open Matrix
open scoped BigOperators Matrix Kronecker

noncomputable section

private def upper : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]
private def lower : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 1, 1]
private def lowerInv : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; -1, 1]
private def impulse : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 0]

private theorem inverse_lower : lower⁻¹ = lowerInv := by
  apply Matrix.inv_eq_left_inv
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [lower, lowerInv, Matrix.mul_apply, Fin.sum_univ_two]

private def squared (M : Matrix (Fin 2 × Fin 2) (Fin 2 × Fin 2) ℝ) : ℝ :=
  ∑ j, ((M *ᵥ vectorize impulse) j) ^ 2

/-- False: transposing the non-symmetric row factor changes energy 2 to 4. -/
example : squared (upperᵀ ⊗ₖ lower⁻¹) = squared (whiteningMatrix upper lower) := by
  simp only [whiteningMatrix, inverse_lower]
  norm_num [squared, upper, lowerInv, impulse, vectorize, Matrix.mulVec, dotProduct,
    Fintype.sum_prod_type, Fin.sum_univ_two, Matrix.kroneckerMap_apply, Matrix.transpose_apply]

/-- False: the channel Kronecker factor is L^-1; using L^-T changes 2 to 1. -/
example : squared (upper ⊗ₖ (lower⁻¹)ᵀ) = squared (whiteningMatrix upper lower) := by
  simp only [whiteningMatrix, inverse_lower]
  norm_num [squared, upper, lowerInv, impulse, vectorize, Matrix.mulVec, dotProduct,
    Fintype.sum_prod_type, Fin.sum_univ_two, Matrix.kroneckerMap_apply, Matrix.transpose_apply]

end

end N2P3.MatrixNormalCoordinates.Negative
