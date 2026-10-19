import N2P3.OcularCalibrationOLS
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Actual selected calibration matrices and the frozen applied coefficient image

Source-first lean_ocular_chain_20261018/slice_contract_02.json. The source
selects the actual mask after reference preprocessing, centers both arrays,
stores sklearn.coef_ in EEG-by-EOG order and stores only the EOG mean.
Exact OLS uses independent design columns; native rank_ and solver arithmetic
are the separately named numerical interface, not an exact-rank certificate.
The selected mask is fixed here; recomputing a QC mask is a different operation.
-/
namespace N2P3.OcularCalibrationChain
open Matrix
open scoped Matrix BigOperators
open N2P3.OcularCalibrationOLS
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

@[ext] structure Frozen (C E : Type*) where
  coefficient : C → E → ℝ
  center : E → ℝ
  strength : ℝ

def selectedMatrix {T A : Type*} (fit : Finset T) (values : T → A → ℝ) : Matrix fit A ℝ :=
  fun sample axis => values sample.val axis

def selectedCenter {T E : Type*} (fit : Finset T) (eye : T → E → ℝ) (e : E) : ℝ :=
  sampleMean (fun sample : fit => eye sample.val e)

def design {T E : Type*} (fit : Finset T) (eye : T → E → ℝ) : Matrix fit E ℝ :=
  centered (selectedMatrix fit eye)

def response {T C : Type*} (fit : Finset T) (eeg : T → C → ℝ) : Matrix fit C ℝ :=
  centered (selectedMatrix fit eeg)

def fitFrozen {T E C : Type*} [Fintype E] (fit : Finset T)
    (eye : T → E → ℝ) (eeg : T → C → ℝ) (strength : ℝ) : Frozen C E :=
  ⟨fun c e => coefficients (design fit eye) (response fit eeg) e c,
   selectedCenter fit eye,strength⟩

theorem actual_source_fit_fields_use_the_selected_center_and_transposed_OLS_image
    {T E C : Type*} [Fintype E] (fit : Finset T)
    (eye : T → E → ℝ) (eeg : T → C → ℝ) (strength : ℝ) :
    (fitFrozen fit eye eeg strength).center=selectedCenter fit eye ∧
      (fitFrozen fit eye eeg strength).coefficient=
        fun c e => coefficients (design fit eye) (response fit eeg) e c := ⟨rfl,rfl⟩

theorem selected_arrays_read_only_actual_selected_rows
    {T A : Type*} (fit : Finset T) (left right : T → A → ℝ)
    (same : ∀ sample ∈ fit, left sample=right sample) :
    selectedMatrix fit left=selectedMatrix fit right := by
  funext sample axis
  exact congrFun (same sample.val sample.property) axis

theorem selected_center_and_centered_design_ignore_unselected_rows
    {T E : Type*} (fit : Finset T) (left right : T → E → ℝ)
    (same : ∀ sample ∈ fit, left sample=right sample) :
    selectedCenter fit left=selectedCenter fit right ∧ design fit left=design fit right := by
  have selected := selected_arrays_read_only_actual_selected_rows fit left right same
  refine ⟨?_,?_⟩
  · funext e
    unfold selectedCenter
    congr 1
    funext sample
    exact congrFun (same sample.val sample.property) e
  · exact congrArg centered selected

theorem arbitrary_unselected_EEG_and_EOG_edits_preserve_the_complete_frozen_numeric_fit
    {T E C : Type*} [Fintype E] (fit : Finset T)
    (eye otherEye : T → E → ℝ) (eeg otherEEG : T → C → ℝ) (strength : ℝ)
    (sameEye : ∀ sample ∈ fit, eye sample=otherEye sample)
    (sameEEG : ∀ sample ∈ fit, eeg sample=otherEEG sample) :
    fitFrozen fit eye eeg strength=fitFrozen fit otherEye otherEEG strength := by
  have eyes := selected_center_and_centered_design_ignore_unselected_rows fit eye otherEye sameEye
  have signals : response fit eeg=response fit otherEEG :=
    congrArg centered (selected_arrays_read_only_actual_selected_rows fit eeg otherEEG sameEEG)
  unfold fitFrozen
  rw [eyes.1,eyes.2,signals]

theorem actual_selected_centered_EOG_and_EEG_columns_have_zero_sum
    {T E C : Type*} (fit : Finset T) [Nonempty fit]
    (eye : T → E → ℝ) (eeg : T → C → ℝ) (e : E) (c : C) :
    (∑ sample : fit, design fit eye sample e)=0 ∧
      (∑ sample : fit, response fit eeg sample c)=0 :=
  ⟨centered_calibration_columns_have_zero_sum _ e,
   centered_calibration_columns_have_zero_sum _ c⟩

theorem actual_full_rank_selected_design_has_positive_Gram
    {T E : Type*} [Fintype E] (fit : Finset T) (eye : T → E → ℝ)
    (rank : Function.Injective (design fit eye).mulVec) : (gram (design fit eye)).PosDef :=
  independent_full_column_rank_makes_the_actual_Gram_positive_definite _ rank

theorem actual_frozen_transpose_image_satisfies_the_selected_centered_normal_equations
    {T E C : Type*} [Fintype E] (fit : Finset T)
    (eye : T → E → ℝ) (eeg : T → C → ℝ) (strength : ℝ)
    (rank : Function.Injective (design fit eye).mulVec) :
    ∀ e c, (∑ f, gram (design fit eye) e f *
      (fitFrozen fit eye eeg strength).coefficient c f)=
      ((design fit eye)ᵀ*response fit eeg) e c := by
  intro e c
  exact congrFun (congrFun
    (explicit_OLS_coefficients_solve_the_normal_equations _ _ rank) e) c

theorem actual_frozen_selected_residual_is_orthogonal_to_selected_EOG
    {T E C : Type*} [Fintype E] (fit : Finset T)
    (eye : T → E → ℝ) (eeg : T → C → ℝ)
    (rank : Function.Injective (design fit eye).mulVec) :
    (design fit eye)ᵀ*(response fit eeg-design fit eye*coefficients (design fit eye) (response fit eeg))=0 :=
  calibration_residual_is_orthogonal_to_the_selected_EOG_design _ _ rank

theorem full_selected_EOG_rank_supplies_positive_centered_variation_for_every_EOG
    {T E : Type*} [Fintype E] (fit : Finset T) (eye : T → E → ℝ)
    (rank : Function.Injective (design fit eye).mulVec) (e : E) :
    0 < ∑ sample : fit, (design fit eye sample e)^2 := by
  have nonzero : ∃ sample : fit, design fit eye sample e ≠ 0 := by
    by_contra absent
    push_neg at absent
    have same : (design fit eye).mulVec (Pi.single e 1)=(design fit eye).mulVec 0 := by
      funext sample
      simp [Matrix.mulVec,dotProduct,Pi.single_apply,mul_ite,absent]
    have impossible := congrFun (rank same) e
    simp at impossible
  obtain ⟨sample,positive⟩ := nonzero
  exact Finset.sum_pos' (fun _ _ => sq_nonneg _) ⟨sample,Finset.mem_univ _,sq_pos_of_ne_zero positive⟩

theorem a_constant_calibration_EEG_response_has_zero_centered_response
    {T C : Type*} (fit : Finset T) [Nonempty fit] (constant : C → ℝ) :
    response fit (fun _ => constant)=0 := by
  have size : (fit.card : ℝ) ≠ 0 := by
    simpa only [Fintype.card_coe] using
      (Nat.cast_ne_zero.mpr Fintype.card_ne_zero : (Fintype.card fit : ℝ) ≠ 0)
  ext sample c
  simp [response,centered,selectedMatrix,sampleMean]
  field_simp
  ring

theorem a_constant_EEG_response_is_a_legal_zero_coefficient_image
    {T E C : Type*} [Fintype E] (fit : Finset T) [Nonempty fit]
    (eye : T → E → ℝ) (constant : C → ℝ) (strength : ℝ) :
    (fitFrozen fit eye (fun _ => constant) strength).coefficient=fun _ _ => 0 := by
  unfold fitFrozen
  rw [a_constant_calibration_EEG_response_has_zero_centered_response]
  simp [coefficients]

def applyFrozen {C E T : Type*} [Fintype E] (model : Frozen C E)
    (eeg : C → T → ℝ) (eye : E → T → ℝ) (c : C) (t : T) : ℝ :=
  OcularPointwise.correct (model.coefficient c) model.center model.strength (eeg c t) (fun e => eye e t)

theorem actual_fit_then_new_recording_application_uses_only_the_frozen_calibration_center
    {S E C T : Type*} [Fintype E] (fit : Finset S)
    (calEye : S → E → ℝ) (calEEG : S → C → ℝ) (strength : ℝ)
    (newEEG : C → T → ℝ) (newEye : E → T → ℝ) (c : C) (t : T) :
    applyFrozen (fitFrozen fit calEye calEEG strength) newEEG newEye c t=
      newEEG c t-strength*∑ e, coefficients (design fit calEye) (response fit calEEG) e c*
        (newEye e t-selectedCenter fit calEye e) := rfl

theorem frozen_application_only_reads_current_inference_samples
    {C E T : Type*} [Fintype E] (model : Frozen C E)
    (eeg otherEEG : C → T → ℝ) (eye otherEye : E → T → ℝ) (c : C) (t : T)
    (sameEEG : eeg c t=otherEEG c t) (sameEye : ∀ e, eye e t=otherEye e t) :
    applyFrozen model eeg eye c t=applyFrozen model otherEEG otherEye c t := by
  exact OcularPointwise.pointwise_output_only_reads_the_current_samples _ _ _
    (eeg c) (otherEEG c) eye otherEye t sameEEG sameEye

theorem frozen_application_restriction_uses_the_same_stored_model
    {C E T U : Type*} [Fintype E] (model : Frozen C E)
    (eeg : C → T → ℝ) (eye : E → T → ℝ) (select : U → T) :
    applyFrozen model (fun c => eeg c ∘ select) (fun e => eye e ∘ select)=
      fun c => applyFrozen model eeg eye c ∘ select := rfl

theorem zero_strength_preserves_the_already_referenced_EEG_input
    {C E T : Type*} [Fintype E] (model : Frozen C E) (zero : model.strength=0)
    (eeg : C → T → ℝ) (eye : E → T → ℝ) (c : C) (t : T) :
    applyFrozen model eeg eye c t=eeg c t := by simp [applyFrozen,OcularPointwise.correct,zero]

end
end N2P3.OcularCalibrationChain
