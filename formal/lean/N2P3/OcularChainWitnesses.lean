import N2P3.OcularCallerAuthorization
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
Independent wrong alternatives fixed in slice_contract_02, before these
fixtures. Tiny exact examples distinguish wrong support, centering, target
re-estimation, reference/order, classifier domain, time and record semantics.
The separate native string-payload boundary is measured only by the predeclared
non-fit probe; these typed real records never pretend to normalize Python.
-/
namespace N2P3.OcularChainWitnesses
open N2P3.OcularCalibrationChain N2P3.OcularCalibrationOLS
open N2P3.OcularReferenceRecord N2P3.OcularCallerAuthorization
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def fitRows : Finset (Fin 3) := {0,1}
def calibrationEye (row : Fin 3) (_ : Fin 1) : ℝ := (![1,3,100] : Fin 3 → ℝ) row
def calibrationEEG (row : Fin 3) (_ : Fin 1) : ℝ := (![5,7,30] : Fin 3 → ℝ) row

theorem selected_EOG_center_differs_from_the_all_raw_center :
    selectedCenter fitRows calibrationEye 0=2 ∧
      selectedCenter (Finset.univ : Finset (Fin 3)) calibrationEye 0=104/3 := by
  constructor
  · change (∑ row ∈ fitRows.attach, calibrationEye row.val 0)/(fitRows.card : ℝ)=2
    rw [Finset.sum_attach fitRows (fun row => calibrationEye row 0)]
    norm_num [fitRows,calibrationEye]
  · norm_num [selectedCenter,sampleMean,calibrationEye,Finset.sum_attach,Fin.sum_univ_succ]

theorem source_response_is_centered_instead_of_using_the_native_EEG_value :
    response fitRows calibrationEEG ⟨0,by simp [fitRows]⟩ 0= -1 ∧ calibrationEEG 0 0=5 := by
  constructor
  · change 5-(∑ row ∈ fitRows.attach, calibrationEEG row.val 0)/(fitRows.card : ℝ)= -1
    rw [Finset.sum_attach fitRows (fun row => calibrationEEG row 0)]
    norm_num [fitRows,calibrationEEG]
  · norm_num [calibrationEEG]

theorem flat_selected_EOG_cannot_supply_exact_full_column_rank :
    ¬ Function.Injective (design fitRows (fun (_ : Fin 3) (_ : Fin 1) => (0 : ℝ))).mulVec := by
  intro injective
  have equalImage :
      (design fitRows (fun (_ : Fin 3) (_ : Fin 1) => (0 : ℝ))).mulVec (fun _ => 0)=
      (design fitRows (fun (_ : Fin 3) (_ : Fin 1) => (0 : ℝ))).mulVec (fun _ => 1) := by
    funext sample
    simp [design,centered,selectedMatrix,sampleMean,Matrix.mulVec,dotProduct]
  have impossible := congrFun (injective equalImage) 0
  norm_num at impossible

def frozenOne : Frozen (Fin 1) (Fin 1) := ⟨fun _ _ => 2,fun _ => 3,1⟩

theorem new_recording_center_reestimation_changes_the_frozen_output :
    applyFrozen frozenOne (fun _ _ : Fin 1 => 10) (fun _ _ => 5) 0 0=6 ∧
      OcularPointwise.correct (fun _ : Fin 1 => 2) (fun _ => 5) 1 10 (fun _ => 5)=10 := by
  norm_num [applyFrozen,frozenOne,OcularPointwise.correct,Fin.sum_univ_succ]

theorem strength0_is_referenced_EEG_rather_than_untouched_native_EEG :
    OcularPointwise.correct (fun _ : Fin 1 => 2) (fun _ => 3) 0
      (referenceValue .shared .eeg 10 1) (fun _ => 2)=9 := by
  norm_num [referenceValue,OcularPointwise.correct]

theorem shared_and_preserved_acquired_EOG_give_distinct_correct_outputs :
    OcularPointwise.referenced (fun _ : Fin 1 => 2) (fun _ => 3) 1 true 10 1 (fun _ => 2)=13 ∧
      OcularPointwise.referenced (fun _ : Fin 1 => 2) (fun _ => 3) 1 false 10 1 (fun _ => 2)=11 := by
  norm_num [OcularPointwise.referenced,OcularPointwise.referenceEOG,OcularPointwise.correct,Fin.sum_univ_succ]

def roleFixture (name : Fin 3) : Role := if name=2 then .eog else .eeg

theorem anchor_is_misc_while_retained_EOG_is_still_outside_the_classifier_EEG_domain :
    afterReferenceRole roleFixture (some 0) 0=.misc ∧
      afterReferenceRole roleFixture (some 0) 2=.eog ∧
      (2 : Fin 3) ∉ roleOrder [0,1,2] (afterReferenceRole roleFixture (some 0)) .eeg := by
  norm_num [afterReferenceRole,roleFixture,roleOrder,Fin.ext_iff]
  intro impossible
  cases impossible

theorem the_three_actual_caller_time_policies_are_distinct :
    legacyRawGate (⟨0,{0}⟩ : Ledger ℕ ℕ) 0 1 10 10 ∧
      ¬ prospectiveGate ({0} : Finset ℕ) 1 10 10 ∧
      independentGate ({0} : Finset ℕ) {1} {0} 1 := by
  norm_num [legacyRawGate,prospectiveGate,independentGate,Finset.disjoint_left]

theorem unknown_participant_budget_refuses_the_nonempty_fit_ledger :
    ¬ appliedBudget true true (fun _ : ℕ => (∅ : Finset ℕ)) (⟨0,{0}⟩ : Ledger ℕ ℕ) := by
  exact unknown_participant_with_empty_allowed_support_cannot_authorize_applied_processing _ _ rfl _

def numericPayload : Payload (Fin 1) (Fin 1) :=
  ⟨fun _ _ => 2,fun _ => 3,["FZ"],["HEOG"],"P","measured_channel:CZ",some "CZ",["cal"],[],"ear"⟩

def defaultInSchema5 : Record (Fin 1) (Fin 1) :=
  ⟨numericPayload,5,some .shared,some 1,true,true,true,[]⟩

theorem explicit_default_strength_cannot_be_smuggled_into_schema5 : decode defaultInSchema5=none := by
  simp [decode,proposed,encode,defaultInSchema5]

theorem shared_measured_reference_cannot_invent_missing_EOG_derivation :
    ¬ sharedDerivationGate false true (fun _ : Fin 1 => ⟨.bipolar,"eye","other"⟩) "ear" := by
  simp [sharedDerivationGate]

theorem CAR_EEG_coordinates_can_still_depend_on_the_retained_EOG_signal :
    commonAverage (fun c : Fin 2 => OcularPointwise.correct
      (fun _ : Fin 1 => if c=0 then 2 else 0) (fun _ => 0) 1 0 (fun _ => 1)) 0= -1 ∧
    commonAverage (fun c : Fin 2 => OcularPointwise.correct
      (fun _ : Fin 1 => if c=0 then 2 else 0) (fun _ => 0) 1 0 (fun _ => 0)) 0=0 := by
  norm_num [commonAverage,OcularPointwise.correct,Fin.sum_univ_succ]

end
end N2P3.OcularChainWitnesses
