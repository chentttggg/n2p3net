import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# Source mathematics of the frozen pointwise ocular correction

Source-fixed before this module: lean_v08_fit_20261018/science_contract_01.json.
ocular.py:157-197 fixes the calibration center and coefficients at inference;
_reference_raw precedes this subtraction. continuous_qc.py:513-540 represents
the resulting native-channel dependencies, deliberately excluding DC terms.
This is exact real algebra with finite EEG/EOG roles. It does not claim that
fitted nuisance is brain-independent, or refine NumPy/MNE/IEEE execution.
-/
namespace N2P3.OcularPointwise
open scoped BigOperators
noncomputable section
variable {E T U : Type*} [Fintype E]

def correct (B mean : E → ℝ) (strength x : ℝ) (z : E → ℝ) : ℝ :=
  x-strength*∑ e, B e*(z e-mean e)

def apply (B mean : E → ℝ) (strength : ℝ) (x : T → ℝ) (z : E → T → ℝ) (t : T) : ℝ :=
  correct B mean strength (x t) (fun e => z e t)

theorem pointwise_output_only_reads_the_current_samples
    (B mean : E → ℝ) (strength : ℝ) (x y : T → ℝ) (z w : E → T → ℝ) (t : T)
    (eeg : x t=y t) (eog : ∀ e, z e t=w e t) :
    apply B mean strength x z t=apply B mean strength y w t := by
  simp only [apply,eeg,eog]

theorem sample_restriction_commutes_with_the_same_frozen_model
    (B mean : E → ℝ) (strength : ℝ) (x : T → ℝ) (z : E → T → ℝ) (select : U → T) :
    apply B mean strength (x ∘ select) (fun e => z e ∘ select)=
      apply B mean strength x z ∘ select := rfl

theorem zero_strength_preserves_the_input_to_regression (B mean z : E → ℝ) (x : ℝ) :
    correct B mean 0 x z=x := by simp [correct]

theorem zero_effective_coefficients_preserve_the_input (mean z : E → ℝ) (strength x : ℝ) :
    correct (fun _ => 0) mean strength x z=x := by simp [correct]

theorem calibration_center_samples_have_zero_correction (B mean : E → ℝ) (strength x : ℝ) :
    correct B mean strength x mean=x := by simp [correct]

theorem frozen_center_is_an_affine_DC_term (B mean z : E → ℝ) (strength x : ℝ) :
    correct B mean strength x z=
      x-strength*(∑ e, B e*z e)+strength*(∑ e, B e*mean e) := by
  simp only [correct,mul_sub,Finset.sum_sub_distrib]
  ring

theorem two_outputs_cancel_the_frozen_center (B mean z w : E → ℝ) (strength x y : ℝ) :
    correct B mean strength x z-correct B mean strength y w=
      (x-y)-strength*(∑ e, B e*(z e-w e)) := by
  simp only [frozen_center_is_an_affine_DC_term,mul_sub,Finset.sum_sub_distrib]
  ring

theorem changing_the_center_adds_an_explicit_offset (B mean other z : E → ℝ) (strength x : ℝ) :
    correct B other strength x z-correct B mean strength x z=
      strength*(∑ e, B e*(other e-mean e)) := by
  simp only [frozen_center_is_an_affine_DC_term,mul_sub,Finset.sum_sub_distrib]
  ring

def referenceEOG (shared : Bool) (r : ℝ) (z : E → ℝ) : E → ℝ :=
  fun e => z e-(if shared then r else 0)

def referenced (B mean : E → ℝ) (strength : ℝ) (shared : Bool) (x r : ℝ) (z : E → ℝ) : ℝ :=
  correct B mean strength (x-r) (referenceEOG shared r z)

def dependency (B : E → ℝ) (strength : ℝ) (shared : Bool) (x r : ℝ) (z : E → ℝ) : ℝ :=
  (x-r)-strength*(∑ e, B e*(referenceEOG shared r z e))

theorem native_dependency_excludes_only_the_frozen_DC_center
    (B mean z : E → ℝ) (strength x r : ℝ) (shared : Bool) :
    referenced B mean strength shared x r z=
      dependency B strength shared x r z+strength*(∑ e, B e*mean e) := by
  simp only [referenced,dependency,frozen_center_is_an_affine_DC_term]

theorem dependency_map_matches_differences_of_actual_corrected_outputs
    (B mean z w : E → ℝ) (strength x y r s : ℝ) (shared : Bool) :
    referenced B mean strength shared x r z-referenced B mean strength shared y s w=
      dependency B strength shared x r z-dependency B strength shared y s w := by
  simp only [native_dependency_excludes_only_the_frozen_DC_center]
  ring

theorem zero_strength_preserves_reference_order (B mean z : E → ℝ) (x r : ℝ) (shared : Bool) :
    referenced B mean 0 shared x r z=x-r := by
  simp [referenced,zero_strength_preserves_the_input_to_regression]

theorem preserved_EOG_reference_has_anchor_coefficient_minus_one
    (B mean z : E → ℝ) (strength x r : ℝ) :
    referenced B mean strength false x r z=
      correct B mean strength x z-r := by
  simp only [referenced,referenceEOG,Bool.false_eq_true,ite_false,sub_zero,correct]
  ring

theorem shared_EOG_reference_has_the_sum_of_effective_coefficients
    (B mean z : E → ℝ) (strength x r : ℝ) :
    referenced B mean strength true x r z=
      correct B mean strength x z+(-1+strength*(∑ e, B e))*r := by
  have shifted : (∑ e, B e*(z e-r-mean e))=
      (∑ e, B e*(z e-mean e))-(∑ e, B e)*r := by
    calc
      _ = ∑ e, (B e*(z e-mean e)-B e*r) := by
        apply Finset.sum_congr rfl
        intro e _
        ring
      _ = _ := by rw [Finset.sum_sub_distrib,Finset.sum_mul]
  simp only [referenced,referenceEOG,ite_true,correct,shifted]
  ring

theorem reference_policies_differ_by_the_shared_anchor_contribution
    (B mean z : E → ℝ) (strength x r : ℝ) :
    referenced B mean strength true x r z-referenced B mean strength false x r z=
      strength*(∑ e, B e)*r := by
  rw [shared_EOG_reference_has_the_sum_of_effective_coefficients,
    preserved_EOG_reference_has_anchor_coefficient_minus_one]
  ring

theorem changing_a_preserved_EOG_anchor_has_coefficient_minus_one
    (B mean z : E → ℝ) (strength x r delta : ℝ) :
    referenced B mean strength false x (r+delta) z-referenced B mean strength false x r z= -delta := by
  simp only [preserved_EOG_reference_has_anchor_coefficient_minus_one]
  ring

theorem changing_a_shared_EOG_anchor_has_the_declared_dependency
    (B mean z : E → ℝ) (strength x r delta : ℝ) :
    referenced B mean strength true x (r+delta) z-referenced B mean strength true x r z=
      (-1+strength*(∑ e, B e))*delta := by
  simp only [shared_EOG_reference_has_the_sum_of_effective_coefficients]
  ring

theorem replacing_the_frozen_center_can_change_the_same_samples :
    correct (fun _ : Fin 1 => 1) (fun _ => 0) 1 2 (fun _ => 3)= -1 ∧
    correct (fun _ : Fin 1 => 1) (fun _ => 1) 1 2 (fun _ => 3)=0 := by
  norm_num [correct,Fin.sum_univ_succ]

theorem source_fixed_reference_policy_counterexample :
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 true 2 1 (fun _ => 3)= -3 ∧
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 false 2 1 (fun _ => 3)= -5 ∧
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 0 true 2 1 (fun _ => 3)=1 := by
  norm_num [referenced,referenceEOG,correct,Fin.sum_univ_succ]

end
end N2P3.OcularPointwise
