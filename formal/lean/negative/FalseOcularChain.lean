import N2P3.OcularChainWitnesses

namespace N2P3.NegativeOcularChain
open N2P3.OcularChainWitnesses N2P3.OcularCalibrationChain N2P3.OcularCalibrationOLS
open N2P3.OcularReferenceRecord N2P3.OcularCallerAuthorization
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

-- O01: all raw rows are incorrectly used to center selected EOG.
example : selectedCenter fitRows calibrationEye 0=104/3 := by
  rw [selected_EOG_center_differs_from_the_all_raw_center.1]
  norm_num

-- O02: only EOG is centered, with raw EEG5 used as the regression response.
example : response fitRows calibrationEEG ⟨0,by simp [fitRows]⟩ 0=5 := by
  rw [source_response_is_centered_instead_of_using_the_native_EEG_value.1]
  norm_num

-- O03: an identically flat EOG column incorrectly supplies positive variation.
example : 0 < ∑ row : fitRows,
    (design fitRows (fun (_ : Fin 3) (_ : Fin 1) => (0 : ℝ)) row 0)^2 := by
  norm_num [design,centered,selectedMatrix,sampleMean]

-- O04: inference silently replaces the frozen center by new-recording mean5.
example : applyFrozen frozenOne (fun _ _ : Fin 1 => 10) (fun _ _ => 5) 0 0=10 := by
  rw [new_recording_center_reestimation_changes_the_frozen_output.1]
  norm_num

-- O05: strength0 incorrectly returns untouched native EEG10.
example : OcularPointwise.correct (fun _ : Fin 1 => 2) (fun _ => 3) 0
    (referenceValue .shared .eeg 10 1) (fun _ => 2)=10 := by
  rw [strength0_is_referenced_EEG_rather_than_untouched_native_EEG]
  norm_num

-- O06: preserved EOG is incorrectly rereferenced, giving shared output13.
example : OcularPointwise.referenced (fun _ : Fin 1 => 2) (fun _ => 3) 1 false 10 1 (fun _ => 2)=13 := by
  rw [shared_and_preserved_acquired_EOG_give_distinct_correct_outputs.2]
  norm_num

def roleCode : Role → ℕ | .eeg => 0 | .eog => 1 | .misc => 2

-- O07: retained EOG is incorrectly included as a classifier EEG coordinate.
example : roleCode (afterReferenceRole roleFixture (some 0) 2)=0 := by
  rw [anchor_is_misc_while_retained_EOG_is_still_outside_the_classifier_EEG_domain.2.1]
  norm_num [roleCode]

-- O08: legacy raw equality is incorrectly transplanted to prospective labels.
example : prospectiveGate ({0} : Finset ℕ) 1 10 10 := by
  norm_num [prospectiveGate]

-- O09: unknown participant is incorrectly authorized by an empty budget.
example : appliedBudget true true (fun _ : ℕ => (∅ : Finset ℕ)) (⟨0,{0}⟩ : Ledger ℕ ℕ) := by
  norm_num [appliedBudget,budgetAuthorized]

-- O10: schema5 with default strength1 is incorrectly accepted on restoration.
example : decode defaultInSchema5=some (proposed defaultInSchema5) := by
  rw [explicit_default_strength_cannot_be_smuggled_into_schema5]
  simp

-- O11: shared measured EOG reference is incorrectly invented without derivation.
example : sharedDerivationGate false true
    (fun _ : Fin 1 => ⟨.bipolar,"eye","other"⟩) "ear" := by
  simp [sharedDerivationGate]

-- O12: EEG-only CAR is incorrectly described as independent of EOG values.
example : commonAverage (fun c : Fin 2 => OcularPointwise.correct
    (fun _ : Fin 1 => if c=0 then 2 else 0) (fun _ => 0) 1 0 (fun _ => 1)) 0=0 := by
  rw [CAR_EEG_coordinates_can_still_depend_on_the_retained_EOG_signal.1]
  norm_num

end
end N2P3.NegativeOcularChain
