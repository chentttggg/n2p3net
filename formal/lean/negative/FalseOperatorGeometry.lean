import N2P3.OperatorGeometry

namespace N2P3.OperatorGeometry.Negative

open Matrix
open scoped BigOperators Matrix

/-- False: the strict source eigenvalue threshold includes equality. -/
example : (1 : ℝ) * eigenTolerance < eigenTolerance := by
  simp only [one_mul, lt_self_iff_false]

/-- False: source support ridge makes every ambient direction positive definite. -/
example : (regularizedCovariance axisBasis axisValues 2).PosDef := by
  have hn : ¬(regularizedCovariance axisBasis axisValues 2).PosDef := by
    intro h
    have hx : discardedAxis ≠ 0 := by
      intro he
      have hv := congrFun he (1 : Fin 2)
      norm_num [discardedAxis] at hv
    have hp := h.2 discardedAxis hx
    rw [ridge_leaves_discarded_axis_zero] at hp
    simp at hp
  simp only [hn]

/-- False: the real source contrast tolerance proves exact retained support. -/
example : contrastAccepted axisBasis nearContrast → projector axisBasis *ᵥ nearContrast = nearContrast := by
  intro _
  simp only [near_contrast_is_not_exactly_supported]

/-- False: source weights solve the full tolerated contrast, including its null component. -/
example : regularizedCovariance axisBasis axisValues 2 *ᵥ weights axisBasis axisValues 2 nearContrast = nearContrast := by
  simp only [actual_weights_solve_only_projected_near_contrast]

/-- False: appending an in-support row cannot change the admission of a fixed null row. -/
example : batchAccepted axisBasis dilutedNullRow → batchAccepted axisBasis nullRow := by
  intro _
  simp only [isolated_null_row_refused]

/-- False: the two class means' source intercept can be omitted. -/
example : logit (fun _ : Fin 1 => 1) (fun _ => 2)
    (intercept (fun _ : Fin 1 => 0) (fun _ => 2) (fun _ => 2)) =
    (fun _ : Fin 1 => (1 : ℝ)) ⬝ᵥ (fun _ => 2) := by
  norm_num [logit, intercept, dotProduct, Fin.sum_univ_one]

end N2P3.OperatorGeometry.Negative
