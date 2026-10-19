import N2P3.ObservedStatistics
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases

/-! Source-fixed spatial_maxnorm_contract_01.json. Each output filter row is
flattened independently; clamp_min precedes division and clamp_max follows it.
The source passes a temporary effective tensor to functional convolution.
This is exact finite-real mathematics, not a certificate of native tensor
ownership, rounding, gradients, whole-forward purity or model serialization. -/
namespace N2P3.SpatialEffectiveMaxNorm
open scoped BigOperators
noncomputable section
variable {I : Type*} [Fintype I]

def squares (row : I → ℝ) : ℝ := ∑ i, row i ^ 2
def rowNorm (row : I → ℝ) : ℝ := Real.sqrt (squares row)
def denominator (epsilon : ℝ) (row : I → ℝ) : ℝ := max (rowNorm row) epsilon
def scale (limit epsilon : ℝ) (row : I → ℝ) : ℝ := min (limit / denominator epsilon row) 1
def effective (limit epsilon : ℝ) (row : I → ℝ) : I → ℝ := fun i => scale limit epsilon row * row i

theorem squares_nonnegative (row : I → ℝ) : 0 ≤ squares row :=
  Finset.sum_nonneg (fun _ _ => sq_nonneg _)

theorem row_norm_nonnegative (row : I → ℝ) : 0 ≤ rowNorm row := Real.sqrt_nonneg _

theorem source_denominator_is_positive (row : I → ℝ) (epsilon : ℝ) (positive : 0 < epsilon) :
    0 < denominator epsilon row := lt_of_lt_of_le positive (le_max_right _ _)

theorem source_scale_is_positive_and_at_most_one (row : I → ℝ) (limit epsilon : ℝ)
    (limitPositive : 0 < limit) (floorPositive : 0 < epsilon) :
    0 < scale limit epsilon row ∧ scale limit epsilon row ≤ 1 := by
  exact ⟨lt_min (div_pos limitPositive (source_denominator_is_positive row epsilon floorPositive)) zero_lt_one,
    min_le_right _ _⟩

theorem squares_scale_by_the_square_of_the_amplitude (row : I → ℝ) (amplitude : ℝ) :
    squares (fun i => amplitude * row i) = amplitude ^ 2 * squares row := by
  simp only [squares, mul_pow, Finset.mul_sum]

theorem norm_scales_by_a_nonnegative_amplitude (row : I → ℝ) (amplitude : ℝ) (nonnegative : 0 ≤ amplitude) :
    rowNorm (fun i => amplitude * row i) = amplitude * rowNorm row := by
  unfold rowNorm
  rw [squares_scale_by_the_square_of_the_amplitude, Real.sqrt_mul (sq_nonneg amplitude), Real.sqrt_sq nonnegative]

theorem actual_effective_norm_is_scale_times_raw_norm (row : I → ℝ) (limit epsilon : ℝ)
    (limitPositive : 0 < limit) (floorPositive : 0 < epsilon) :
    rowNorm (effective limit epsilon row) = scale limit epsilon row * rowNorm row :=
  norm_scales_by_a_nonnegative_amplitude row _
    (source_scale_is_positive_and_at_most_one row limit epsilon limitPositive floorPositive).1.le

theorem source_effective_norm_is_bounded_by_limit_and_raw_norm (row : I → ℝ) (limit epsilon : ℝ)
    (limitPositive : 0 < limit) (floorPositive : 0 < epsilon) :
    rowNorm (effective limit epsilon row) ≤ limit ∧ rowNorm (effective limit epsilon row) ≤ rowNorm row := by
  have denominatorPositive := source_denominator_is_positive row epsilon floorPositive
  have scaleBounds := source_scale_is_positive_and_at_most_one row limit epsilon limitPositive floorPositive
  have multiplied : scale limit epsilon row * denominator epsilon row ≤ limit :=
    (le_div_iff₀ denominatorPositive).mp (min_le_left _ _)
  rw [actual_effective_norm_is_scale_times_raw_norm row limit epsilon limitPositive floorPositive]
  exact ⟨le_trans (mul_le_mul_of_nonneg_left (le_max_left _ _) scaleBounds.1.le) multiplied,
    by simpa using mul_le_mul_of_nonneg_right scaleBounds.2 (row_norm_nonnegative row)⟩

theorem denominator_inside_limit_gives_the_actual_identity (row : I → ℝ) (limit epsilon : ℝ)
    (floorPositive : 0 < epsilon) (inside : denominator epsilon row ≤ limit) :
    effective limit epsilon row = row := by
  have positive := source_denominator_is_positive row epsilon floorPositive
  have quotient : 1 ≤ limit / denominator epsilon row := (le_div_iff₀ positive).mpr (by simpa using inside)
  funext i
  simp [effective, scale, min_eq_right quotient]

theorem ordinary_inside_ball_identity_requires_the_source_floor_condition (row : I → ℝ) (limit epsilon : ℝ)
    (floorPositive : 0 < epsilon) (normInside : rowNorm row ≤ limit) (floorInside : epsilon ≤ limit) :
    effective limit epsilon row = row :=
  denominator_inside_limit_gives_the_actual_identity row limit epsilon floorPositive (max_le normInside floorInside)

theorem above_both_floor_and_limit_gives_exact_boundary_norm (row : I → ℝ) (limit epsilon : ℝ)
    (limitPositive : 0 < limit) (floorPositive : 0 < epsilon)
    (floorBelow : epsilon ≤ rowNorm row) (limitBelow : limit ≤ rowNorm row) :
    rowNorm (effective limit epsilon row) = limit := by
  have normPositive : 0 < rowNorm row := lt_of_lt_of_le floorPositive floorBelow
  rw [actual_effective_norm_is_scale_times_raw_norm row limit epsilon limitPositive floorPositive]
  unfold scale denominator
  rw [max_eq_left floorBelow, min_eq_left ((div_le_one normPositive).mpr limitBelow)]
  exact div_mul_cancel₀ _ normPositive.ne'

theorem zero_raw_row_gives_zero_effective_row (limit epsilon : ℝ) :
    effective limit epsilon (fun _ : I => 0) = fun _ => 0 := by
  funext i
  simp [effective]

theorem fixed_three_four_raw_row_has_norm_five : rowNorm ![3, 4] = 5 := by
  apply (Real.sqrt_eq_iff_eq_sq (squares_nonnegative _) (by norm_num)).mpr
  norm_num [squares, Fin.sum_univ_succ]

theorem fixed_three_four_source_uses_row_scaling :
    effective 1 (1 / 1000000000000) ![3, 4] = ![3 / 5, 4 / 5] := by
  funext i
  fin_cases i <;> norm_num [effective, scale, denominator, fixed_three_four_raw_row_has_norm_five]

theorem singleton_nonnegative_row_norm (value : ℝ) (nonnegative : 0 ≤ value) :
    rowNorm (fun _ : Fin 1 => value) = value := by
  simp [rowNorm, squares, Real.sqrt_sq nonnegative]

theorem tiny_floor_fixture_shrinks_a_row_already_inside_the_ball (epsilon : ℝ) (positive : 0 < epsilon) :
    rowNorm (fun _ : Fin 1 => epsilon / 100) < epsilon / 10 ∧
      effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100) = fun _ => epsilon / 1000 := by
  have rowNonnegative : 0 ≤ epsilon / 100 := by positivity
  have normValue := singleton_nonnegative_row_norm (epsilon / 100) rowNonnegative
  have floorDominates : epsilon / 100 ≤ epsilon := by linarith
  have ratio : (epsilon / 10) / epsilon = (1 / 10 : ℝ) := by field_simp
  constructor
  · rw [normValue]
    linarith
  · funext i
    simp only [effective, scale, denominator, normValue, max_eq_right floorDominates, ratio]
    norm_num
    ring

theorem tiny_floor_fixture_is_not_idempotent (epsilon : ℝ) (positive : 0 < epsilon) :
    effective (epsilon / 10) epsilon
        (effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100)) ≠
      effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100) := by
  rw [(tiny_floor_fixture_shrinks_a_row_already_inside_the_ball epsilon positive).2]
  have normValue := singleton_nonnegative_row_norm (epsilon / 1000) (by positivity : 0 ≤ epsilon / 1000)
  have floorDominates : epsilon / 1000 ≤ epsilon := by linarith
  have ratio : (epsilon / 10) / epsilon = (1 / 10 : ℝ) := by field_simp
  intro same
  have cell := congrFun same 0
  simp only [effective, scale, denominator, normValue, max_eq_right floorDominates, ratio] at cell
  norm_num at cell
  linarith

end
end N2P3.SpatialEffectiveMaxNorm
