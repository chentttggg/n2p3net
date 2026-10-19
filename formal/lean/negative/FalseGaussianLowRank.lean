import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.NormNum

/-! Two deliberately false claims from the scorer's covariance/evidence formulas
specialized to one row, one kernel coordinate and one channel. These algebraic
specializations do not assert admission as a complete raw EEG recording.
This file must be rejected; the accepted GaussianLowRank module is unchanged. -/

namespace N2P3.GaussianLowRank.Negative

open Matrix
open scoped Matrix BigOperators

private def unitDesign : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => 1

private def zeroDesign : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => 0

/-- False: U=[1] has full row covariance 2, so kernel uncertainty is not identity noise. -/
example : (1 + unitDesign * unitDesignᵀ).det = (1 : Matrix (Fin 1) (Fin 1) ℝ).det := by
  norm_num [unitDesign, Matrix.det_fin_one, Matrix.mul_apply, Fin.sum_univ_succ]

/-- Exact full scorer algebra for one observed channel and residual x=0.
The Mahalanobis term is zero for either candidate; row logdet still varies. -/
private noncomputable def zeroResidualEvidence (U : Matrix (Fin 1) (Fin 1) ℝ) : ℝ :=
  -(1 / 2 : ℝ) * (max 0 (0 : ℝ) + 1 * Real.log (1 + U * Uᵀ).det)

/-- False: dropping logdet creates a tie between U=[0] and U=[1]. -/
example : zeroResidualEvidence zeroDesign = zeroResidualEvidence unitDesign := by
  have hlog : Real.log 2 ≠ 0 :=
    Real.log_ne_zero_of_pos_of_ne_one (by norm_num) (by norm_num)
  norm_num [zeroResidualEvidence, zeroDesign, unitDesign, Matrix.det_fin_one,
    Matrix.mul_apply, Fin.sum_univ_succ, hlog]

end N2P3.GaussianLowRank.Negative
