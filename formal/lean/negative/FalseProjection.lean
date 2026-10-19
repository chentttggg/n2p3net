import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Tactic.NormNum

/-! Orthogonal-coordinate/scalar counterexamples to false ambient-rank and AR1
startup rules. They do not pretend to be complete raw EEG recordings. -/

open Matrix
open scoped Matrix BigOperators

private def nuisanceQ : Matrix (Fin 3) (Fin 2) ℝ :=
  fun i j => if i.val = j.val then 1 else 0

/-- False: removing two orthonormal nuisance coordinates leaves identity noise. -/
example : ((1 : Matrix (Fin 3) (Fin 3) ℝ) - nuisanceQ * nuisanceQᵀ) 0 0 = 1 := by
  norm_num [nuisanceQ, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

/-- False: a stationary AR1 first sample already has innovation variance 1. -/
example : (1 : ℝ) / (1 - (1 / 2 : ℝ) ^ 2) = 1 := by
  norm_num
