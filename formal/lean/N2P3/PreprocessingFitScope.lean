import N2P3.OcularPointwise
import Mathlib.Data.Set.Disjoint
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Linarith

/-!
# Calibration support, physical identity authorization and overlap weighting

Source-fixed V08 contract: continuous_qc.py:558-580 requires both difference
endpoints within selected clean support and removes segment crossings;
ea_artifact.py:415-453 resolves authorized physical IDs and covers all output
domains; harmonic.py:174-197 normalizes positive overlap weights.
This is the exact finite support/weight mathematics used by those sources.
Caller scientific authority, hash security, numerical fitting and downstream
model efficacy are separate obligations and are not assumed as conclusions.
-/
namespace N2P3.PreprocessingFitScope
open scoped BigOperators
noncomputable section

def eligibleDifference (samples : ℕ) (fit clean segmentStart : ℕ → Bool) (t : ℕ) : Prop :=
  t+1 < samples ∧ fit t=true ∧ fit (t+1)=true ∧ clean t=true ∧ clean (t+1)=true ∧ segmentStart (t+1)=false

def difference (x : ℕ → ℝ) (t : ℕ) : ℝ := x (t+1)-x t

theorem changing_unselected_samples_preserves_every_eligible_difference
    (samples : ℕ) (fit clean starts : ℕ → Bool) (x y : ℕ → ℝ)
    (unchanged : ∀ t, fit t=true → x t=y t) (t : ℕ)
    (eligible : eligibleDifference samples fit clean starts t) : difference x t=difference y t := by
  rcases eligible with ⟨_,left,right,_,_,_⟩
  simp only [difference,unchanged t left,unchanged (t+1) right]

theorem missing_either_calibration_endpoint_excludes_the_difference
    (samples : ℕ) (fit clean starts : ℕ → Bool) (t : ℕ)
    (missing : fit t=false ∨ fit (t+1)=false) : ¬eligibleDifference samples fit clean starts t := by
  rintro ⟨_,left,right,_,_,_⟩
  rcases missing with missing | missing
  · rw [missing] at left
    cases left
  · rw [missing] at right
    cases right

theorem an_actual_segment_start_excludes_the_crossing_difference
    (samples : ℕ) (fit clean starts : ℕ → Bool) (t : ℕ) (boundary : starts (t+1)=true) :
    ¬eligibleDifference samples fit clean starts t := by
  rintro ⟨_,_,_,_,_,safe⟩
  rw [boundary] at safe
  cases safe

theorem all_selected_differences_and_any_frozen_summary_are_unchanged
    {S : Type*} (samples : ℕ) (fit clean starts : ℕ → Bool) (x y : ℕ → ℝ)
    (unchanged : ∀ t, fit t=true → x t=y t) (pairs : List ℕ)
    (legal : ∀ t ∈ pairs, eligibleDifference samples fit clean starts t)
    (summary : List ℝ → S) : summary (pairs.map (difference x))=summary (pairs.map (difference y)) := by
  congr 1
  apply List.map_congr_left
  intro t member
  exact changing_unselected_samples_preserves_every_eligible_difference samples fit clean starts x y unchanged t (legal t member)

theorem resolved_physical_identity_order_excludes_heldout_fit_rows
    {Row ID : Type*} (identity : Row → ID) (rows : List Row) (authorized : List ID) (heldout : Set ID)
    (resolution : rows.map identity=authorized) (disjoint : ∀ id ∈ authorized, id ∉ heldout) :
    ∀ row ∈ rows, identity row ∉ heldout := by
  intro row member
  apply disjoint (identity row)
  rw [← resolution]
  exact List.mem_map.mpr ⟨row,member,rfl⟩

theorem actual_domain_coverage_supplies_a_fit_row_for_every_output_domain
    {Row Domain : Type*} [DecidableEq Domain] (domain : Row → Domain) (fitRows outputRows : List Row)
    (coverage : (fitRows.map domain).toFinset=(outputRows.map domain).toFinset) :
    ∀ row ∈ outputRows, ∃ fitted ∈ fitRows, domain fitted=domain row := by
  intro row member
  have output : domain row ∈ (outputRows.map domain).toFinset := by
    simpa only [List.mem_toFinset] using List.mem_map.mpr ⟨row,member,rfl⟩
  rw [← coverage] at output
  simpa only [List.mem_toFinset,List.mem_map] using output

section Overlap
variable {I : Type*} [Fintype I]

def overlapCorrection (weights estimates : I → ℝ) : ℝ :=
  (∑ i, weights i*estimates i)/(∑ i, weights i)

theorem one_positive_blend_and_nonnegative_others_give_a_positive_denominator
    (weights : I → ℝ) (nonnegative : ∀ i, 0 ≤ weights i) (i : I) (positive : 0 < weights i) :
    0 < ∑ j, weights j := by
  exact lt_of_lt_of_le positive (Finset.single_le_sum (fun j _ => nonnegative j) (Finset.mem_univ i))

theorem normalized_overlap_weights_form_a_partition_of_unity
    (weights : I → ℝ) (denominator : (∑ i, weights i) ≠ 0) :
    (∑ i, weights i/(∑ j, weights j))=1 := by
  rw [← Finset.sum_div]
  exact div_self denominator

theorem identical_local_corrections_are_preserved_by_normalized_overlap
    (weights : I → ℝ) (value : ℝ) (denominator : (∑ i, weights i) ≠ 0) :
    overlapCorrection weights (fun _ => value)=value := by
  unfold overlapCorrection
  rw [← Finset.sum_mul,mul_div_cancel_left₀ value denominator]

theorem overlap_correction_stays_above_a_common_lower_bound
    (weights estimates : I → ℝ) (lower : ℝ)
    (nonnegative : ∀ i, 0 ≤ weights i) (positive : 0 < ∑ i, weights i)
    (bound : ∀ i, lower ≤ estimates i) : lower ≤ overlapCorrection weights estimates := by
  apply (le_div_iff₀ positive).2
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  simpa only [mul_comm lower] using mul_le_mul_of_nonneg_left (bound i) (nonnegative i)

theorem overlap_correction_stays_below_a_common_upper_bound
    (weights estimates : I → ℝ) (upper : ℝ)
    (nonnegative : ∀ i, 0 ≤ weights i) (positive : 0 < ∑ i, weights i)
    (bound : ∀ i, estimates i ≤ upper) : overlapCorrection weights estimates ≤ upper := by
  apply (div_le_iff₀ positive).2
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  simpa only [mul_comm upper] using mul_le_mul_of_nonneg_left (bound i) (nonnegative i)

theorem omitting_overlap_normalization_doubles_equal_corrections_in_a_two_window_overlap :
    overlapCorrection (fun _ : Fin 2 => 1) (fun _ => 3)=3 ∧
    (∑ _ : Fin 2, (1 : ℝ)*3)=6 := by
  norm_num [overlapCorrection,Fin.sum_univ_succ]

end Overlap
end
end N2P3.PreprocessingFitScope
