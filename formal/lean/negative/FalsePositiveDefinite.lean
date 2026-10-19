import N2P3Pilot

open Matrix
open scoped Matrix ComplexOrder

-- Deliberately false: a nonzero rank-one projection is singular.
example : (diagonal ![1, 0] : Matrix (Fin 2) (Fin 2) ℝ).PosDef := by
  apply Matrix.PosDef.diagonal
  intro i
  fin_cases i <;> norm_num
