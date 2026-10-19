import N2P3.CandidateTie
import Mathlib.Tactic.FieldSimp

/-! Literal NumPy relative tie band at the maximum. Source-first contract:
lean_v03b_20261018/slice_contract_01.json; actual API falsifier nonfit_probe_02.
The three callers compare scores, not softmax probabilities. Unlike the
continuous rtol=0 rule, this rule is not invariant under common score shifts. -/
namespace N2P3.RelativeCandidateTie
open N2P3.CandidateTie
noncomputable section
variable {C : Type*} [Fintype C] [Nonempty C] [LinearOrder C]

def threshold (scores : C → ℝ) (atol rtol : ℝ) : ℝ :=
  atol + rtol * |scores (firstArgmax scores)|

def prediction (scores : C → ℝ) (atol rtol : ℝ) : Option C :=
  winner scores (threshold scores atol rtol)

theorem threshold_nonnegative (scores : C → ℝ) (atol rtol : ℝ)
    (ha : 0 ≤ atol) (hr : 0 ≤ rtol) : 0 ≤ threshold scores atol rtol :=
  add_nonneg ha (mul_nonneg hr (abs_nonneg _))

theorem threshold_at_any_maximum (scores : C → ℝ) (atol rtol : ℝ) (b : C)
    (hb : IsMaximum scores b) : threshold scores atol rtol = atol + rtol * |scores b| := by
  unfold threshold
  rw [maximum_values_equal scores (firstArgmax scores) b (firstArgmax_isMaximum scores) hb]

omit [Fintype C] [Nonempty C] [LinearOrder C] in
theorem actual_relative_closeness_at_maximum (scores : C → ℝ) (b j : C)
    (atol rtol : ℝ) (hb : IsMaximum scores b) :
    |scores j - scores b| ≤ atol + rtol * |scores b| ↔
      scores b - scores j ≤ atol + rtol * |scores b| := by
  rw [abs_of_nonpos (sub_nonpos.mpr (hb j))]
  constructor <;> intro h <;> linarith

theorem prediction_some_iff_actual_relative_margin (scores : C → ℝ) (b : C)
    (atol rtol : ℝ) (ha : 0 ≤ atol) (hr : 0 ≤ rtol) :
    prediction scores atol rtol = some b ↔ IsMaximum scores b ∧
      ∀ j, j ≠ b → atol + rtol * |scores b| < scores b - scores j := by
  rw [prediction, winner_some_iff_margin scores b (threshold_nonnegative scores atol rtol ha hr)]
  constructor
  · rintro ⟨hb,margin⟩
    exact ⟨hb,by simpa only [threshold_at_any_maximum scores atol rtol b hb] using margin⟩
  · rintro ⟨hb,margin⟩
    exact ⟨hb,by simpa only [threshold_at_any_maximum scores atol rtol b hb] using margin⟩

theorem prediction_none_iff_relative_competitor (scores : C → ℝ) (atol rtol : ℝ)
    (ha : 0 ≤ atol) (hr : 0 ≤ rtol) :
    prediction scores atol rtol = none ↔
      ∃ j, j ≠ firstArgmax scores ∧ scores (firstArgmax scores)-scores j ≤ threshold scores atol rtol := by
  rw [prediction, winner_none_iff, tied_iff_close_competitor scores _
    (threshold_nonnegative scores atol rtol ha hr) (firstArgmax_isMaximum scores)]

theorem zero_relative_tolerance_recovers_absolute_rule (scores : C → ℝ) (atol : ℝ) :
    prediction scores atol 0 = winner scores atol := by simp [prediction,threshold]

theorem shifted_threshold_uses_shifted_maximum (scores : C → ℝ) (atol rtol shift : ℝ) :
    threshold (fun j => scores j + shift) atol rtol =
      atol + rtol * |scores (firstArgmax scores) + shift| := by
  simp only [threshold, firstArgmax_add_constant]

def sourceAtol : ℝ := 1/1000000000000
def sourceRtol : ℝ := 1/1000000000000
def fixture : Fin 2 → ℝ := fun j => if j=0 then 0 else 4/1000000000000

theorem fixture_strict_maximum (j : Fin 2) (hj : j ≠ 1) : fixture j < fixture 1 := by
  fin_cases j <;> norm_num [fixture] at *

theorem fixture_actual_maximum : IsMaximum fixture 1 := by
  intro j
  fin_cases j <;> norm_num [fixture]

theorem fixture_before_shift_predicts : prediction fixture sourceAtol sourceRtol = some 1 := by
  apply (prediction_some_iff_actual_relative_margin fixture 1 sourceAtol sourceRtol
    (by norm_num [sourceAtol]) (by norm_num [sourceRtol])).mpr
  refine ⟨fixture_actual_maximum,?_⟩
  intro j hj
  fin_cases j <;> norm_num [fixture,sourceAtol,sourceRtol] at *

theorem fixture_after_shift_abstains :
    prediction (fun j => fixture j + 10) sourceAtol sourceRtol = none := by
  rw [prediction, winner_none_iff]
  have maximum : IsMaximum (fun j => fixture j+10) 1 :=
    (maximum_add_constant_iff fixture 1 10).mpr fixture_actual_maximum
  apply (tie_independent_of_argmax (fun j => fixture j+10)
    (firstArgmax (fun j => fixture j+10)) 1 _ (firstArgmax_isMaximum _) maximum).mpr
  apply (tied_iff_close_competitor _ 1
    (threshold_nonnegative _ sourceAtol sourceRtol (by norm_num [sourceAtol]) (by norm_num [sourceRtol])) maximum).mpr
  refine ⟨0,by decide,?_⟩
  rw [threshold_at_any_maximum _ sourceAtol sourceRtol 1 maximum]
  norm_num [fixture,sourceAtol,sourceRtol]

theorem common_shift_preserves_strict_argmax_but_changes_operational_prediction :
    (∀ j : Fin 2, j ≠ 1 → fixture j < fixture 1 ∧ fixture j+10 < fixture 1+10) ∧
      prediction fixture sourceAtol sourceRtol ≠
        prediction (fun j => fixture j+10) sourceAtol sourceRtol := by
  constructor
  · intro j hj
    exact ⟨fixture_strict_maximum j hj,add_lt_add_right (fixture_strict_maximum j hj) 10⟩
  · rw [fixture_before_shift_predicts,fixture_after_shift_abstains]
    simp

end
end N2P3.RelativeCandidateTie
