import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum

/-!
Deliberately false scalar instances of the actual matrix normal equations.
These examples must each be rejected with an unsolved `False` goal.
-/

namespace N2P3.ScaleGauge.Negative

/-- D=Y=K=lambda=1. Scaling K by 2 while incorrectly fixing effective lambda
changes the mean from 1/2 to 1/3, so representation invariance is false. -/
example : (((1 : ℝ) / 2 + 1)⁻¹ * (1 / 2)) = (1 + 1)⁻¹ := by
  norm_num

/-- Voltage factor s=2 with Y'=2, Sigma'=4*Sigma: incorrectly changing lambda
to 4 gives mean 2/5 rather than the required 2*(1/2)=1. -/
example : (((1 : ℝ) + 4)⁻¹ * 2) = 2 * (1 + 1)⁻¹ := by
  norm_num

end N2P3.ScaleGauge.Negative
