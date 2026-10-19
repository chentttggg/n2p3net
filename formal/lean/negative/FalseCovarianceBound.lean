import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum

/-! False numerical claims: a negative input direction can be amplified, and
adding a ridge equal to the negative bound is not strictly positive. -/

example : (-(1 / 10 : ℝ)) * 3 ^ 2 ≥ -(1 / 10 : ℝ) := by
  norm_num

example : (0 : ℝ) < -(9 / 10 : ℝ) + 9 / 10 := by
  norm_num
