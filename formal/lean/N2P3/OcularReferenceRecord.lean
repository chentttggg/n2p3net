import N2P3.OcularCalibrationChain
import Mathlib.Data.List.Defs

/-!
Source-first EO2 composition. Names here are already canonical declared names.
Role/order facts concern BIDS/raw metadata, not physical electrode authenticity.
The record model covers canonical numeric source-produced fields; Python object
coercion, native JSON parsing and cryptographic authenticity remain interfaces.
-/
namespace N2P3.OcularReferenceRecord
open scoped BigOperators
open N2P3.OcularCalibrationChain
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

inductive Role | eeg | eog | misc deriving DecidableEq
inductive Policy | shared | preserve deriving DecidableEq
inductive DerivationKind | singleEnded | bipolar deriving DecidableEq

structure Derivation where
  kind : DerivationKind
  placement : String
  reference : String

def sharedDerivationGate {E : Type*} (specified rereference : Bool)
    (derivation : E → Derivation) (acquisition : String) : Prop :=
  (specified=false → rereference=false) ∧
  (specified=true → ∀ e, (derivation e).placement ≠ "" ∧ (derivation e).reference ≠ "" ∧
    (rereference=true → (derivation e).kind=.singleEnded ∧ (derivation e).reference=acquisition))

theorem shared_measured_rereference_requires_explicit_matching_single_ended_derivation
    {E : Type*} (specified : Bool) (derivation : E → Derivation) (acquisition : String)
    (accepted : sharedDerivationGate specified true derivation acquisition) (e : E) :
    specified=true ∧ (derivation e).kind=.singleEnded ∧ (derivation e).reference=acquisition := by
  have explicit : specified=true := by
    cases specified with
    | false => have impossible := accepted.1 rfl; contradiction
    | true => rfl
  exact ⟨explicit,(accepted.2 explicit e).2.2 rfl⟩

theorem absent_preserved_derivation_does_not_invent_acquisition_wiring
    {E : Type*} (derivation : E → Derivation) (acquisition : String) :
    sharedDerivationGate false false derivation acquisition := by simp [sharedDerivationGate]

def roleOrderGate {N : Type*} (storedEEG storedEOG actualEEG actualEOG : List N) : Prop :=
  actualEEG=storedEEG ∧ actualEOG=storedEOG

def guardedCorrection {N E : Type*} [Fintype E]
    (storedEEG storedEOG actualEEG actualEOG : List N)
    (B center eye : E → ℝ) (strength x : ℝ) : Option ℝ :=
  if roleOrderGate storedEEG storedEOG actualEEG actualEOG then
    some (OcularPointwise.correct B center strength x eye) else none

theorem zero_strength_cannot_bypass_the_actual_per_role_order_gate
    {N E : Type*} [Fintype E] (storedEEG storedEOG actualEEG actualEOG : List N)
    (B center eye : E → ℝ) (x : ℝ) (mismatch : actualEEG ≠ storedEEG ∨ actualEOG ≠ storedEOG) :
    guardedCorrection storedEEG storedEOG actualEEG actualEOG B center eye 0 x=none := by
  have refused : ¬ roleOrderGate storedEEG storedEOG actualEEG actualEOG := by
    intro same
    rcases mismatch with eeg | eog
    · exact eeg same.1
    · exact eog same.2
  simp [guardedCorrection,refused]

def afterReferenceRole {N : Type*} (role : N → Role) (anchor : Option N) (name : N) : Role :=
  if anchor=some name then .misc else role name

def roleOrder {N : Type*} (names : List N) (role : N → Role) (wanted : Role) : List N :=
  names.filter (fun name => decide (role name=wanted))

theorem measured_anchor_is_removed_from_the_postreference_EEG_domain
    {N : Type*} (role : N → Role) (names : List N) (anchor : N) :
    anchor ∉ roleOrder names (afterReferenceRole role (some anchor)) .eeg := by
  simp [roleOrder,afterReferenceRole]

theorem retagging_a_declared_EEG_anchor_preserves_every_declared_EOG_role
    {N : Type*} (role : N → Role) (anchor name : N)
    (anchorEEG : role anchor=.eeg) (eye : role name=.eog) :
    afterReferenceRole role (some anchor) name=.eog := by
  have different : anchor ≠ name := by intro same; subst name; simp_all
  simp [afterReferenceRole,different,eye]

theorem postreference_EOG_order_is_exactly_the_original_raw_EOG_order
    {N : Type*} (role : N → Role) (names : List N) (anchor : N)
    (anchorEEG : role anchor=.eeg) :
    roleOrder names (afterReferenceRole role (some anchor)) .eog=roleOrder names role .eog := by
  unfold roleOrder
  apply List.filter_congr
  intro name _
  by_cases same : anchor=name
  · subst name; simp [afterReferenceRole,anchorEEG]
  · simp [afterReferenceRole,same]

theorem requested_postreference_EEG_names_are_neither_EOG_nor_the_anchor
    {N : Type*} (role : N → Role) (anchor name : N)
    (selected : afterReferenceRole role (some anchor) name=.eeg) :
    role name ≠ .eog ∧ name ≠ anchor := by
  by_cases same : anchor=name
  · subst name; simp [afterReferenceRole] at selected
  · simp [afterReferenceRole,same] at selected
    exact ⟨by rw [selected]; decide,Ne.symm same⟩

def referenceValue (policy : Policy) (kind : Role) (x anchor : ℝ) : ℝ :=
  if kind=.eeg ∨ (kind=.eog ∧ policy=.shared) then x-anchor else x

theorem measured_reference_subtracts_the_anchor_from_every_EEG
    (policy : Policy) (x anchor : ℝ) : referenceValue policy .eeg x anchor=x-anchor := by
  simp [referenceValue]

theorem preserved_EOG_is_not_subtracted_by_the_EEG_anchor (x anchor : ℝ) :
    referenceValue .preserve .eog x anchor=x := by simp [referenceValue]

theorem shared_EOG_is_subtracted_by_the_same_measured_anchor (x anchor : ℝ) :
    referenceValue .shared .eog x anchor=x-anchor := by simp [referenceValue]

theorem zero_strength_still_outputs_measured_referenced_EEG
    {E : Type*} [Fintype E] (B center eye : E → ℝ) (x anchor : ℝ) :
    OcularPointwise.correct B center 0 (referenceValue .shared .eeg x anchor) eye=x-anchor := by
  simp [OcularPointwise.correct,referenceValue]

@[ext] structure Payload (C E : Type*) where
  coefficients : C → E → ℝ
  center : E → ℝ
  eegNames : List String
  eogNames : List String
  participant : String
  sourceReference : String
  referenceAnchor : Option String
  provenance : List String
  derivation : List String
  acquisitionReference : String

@[ext] structure Model (C E : Type*) where
  payload : Payload C E
  policy : Policy
  strength : ℝ

@[ext] structure Record (C E : Type*) where
  payload : Payload C E
  schema : ℕ
  policy : Option Policy
  strength : Option ℝ
  referenceChanged : Bool
  frozenApplication : Bool
  volts : Bool
  unknown : List String

def encode {C E : Type*} (model : Model C E) : Record C E :=
  if model.strength=1 then
    if model.policy=.shared then ⟨model.payload,3,none,none,model.payload.referenceAnchor.isSome,true,true,[]⟩
    else ⟨model.payload,4,some .preserve,none,model.payload.referenceAnchor.isSome,true,true,[]⟩
  else ⟨model.payload,5,some model.policy,some model.strength,model.payload.referenceAnchor.isSome,true,true,[]⟩

def proposed {C E : Type*} (record : Record C E) : Model C E :=
  ⟨record.payload,record.policy.getD .shared,record.strength.getD 1⟩

def decode {C E : Type*} (record : Record C E) : Option (Model C E) :=
  if 0 ≤ (proposed record).strength ∧ (proposed record).strength ≤ 1 ∧
      encode (proposed record)=record then some (proposed record) else none

theorem every_canonical_numeric_source_model_roundtrips
    {C E : Type*} (model : Model C E) (valid : 0 ≤ model.strength ∧ model.strength ≤ 1) :
    decode (encode model)=some model := by
  have recover : proposed (encode model)=model := by
    cases model with
    | mk payload policy strength =>
      cases policy <;> by_cases default : strength=1 <;>
        simp_all [encode,proposed]
  simp [decode,recover,valid]

theorem accepted_restore_keeps_all_stored_numeric_and_identity_fields
    {C E : Type*} (record : Record C E) (model : Model C E)
    (accepted : decode record=some model) :
    model.payload=record.payload ∧ encode model=record := by
  unfold decode at accepted
  split_ifs at accepted with valid
  · cases Option.some.inj accepted
    exact ⟨rfl,valid.2.2⟩

theorem schema3_has_shared_policy_and_omits_both_optional_fields
    {C E : Type*} (model : Model C E) (shared : model.policy=.shared) (default : model.strength=1) :
    (encode model).schema=3 ∧ (encode model).policy=none ∧ (encode model).strength=none := by
  simp [encode,shared,default]

theorem schema4_has_explicit_preserved_policy_and_default_strength
    {C E : Type*} (model : Model C E) (preserve : model.policy=.preserve) (default : model.strength=1) :
    (encode model).schema=4 ∧ (encode model).policy=some .preserve ∧ (encode model).strength=none := by
  simp [encode,preserve,default]

theorem schema5_has_explicit_policy_and_nondefault_strength
    {C E : Type*} (model : Model C E) (nondefault : model.strength ≠ 1) :
    (encode model).schema=5 ∧ (encode model).policy=some model.policy ∧
      (encode model).strength=some model.strength := by simp [encode,nondefault]

theorem an_unknown_record_field_cannot_survive_canonical_restoration
    {C E : Type*} (record : Record C E) (extra : record.unknown ≠ []) : decode record=none := by
  unfold decode
  split_ifs with valid
  · have fields := congrArg Record.unknown valid.2.2
    have canonical : (encode (proposed record)).unknown=[] := by
      simp only [encode]; split_ifs <;> rfl
    exact False.elim (extra (canonical.symm.trans fields).symm)
  · rfl

def restoredFrozen {C E : Type*} (model : Model C E) : Frozen C E :=
  ⟨model.payload.coefficients,model.payload.center,model.strength⟩

theorem accepted_restore_applies_the_same_frozen_numeric_fields
    {C E T : Type*} [Fintype E] (record : Record C E) (model : Model C E)
    (accepted : decode record=some model) (eeg : C → T → ℝ) (eye : E → T → ℝ) (c : C) (t : T) :
    applyFrozen (restoredFrozen model) eeg eye c t=
      eeg c t-model.strength*∑ e, record.payload.coefficients c e*(eye e t-record.payload.center e) := by
  have payload := (accepted_restore_keeps_all_stored_numeric_and_identity_fields record model accepted).1
  simp only [applyFrozen,restoredFrozen,OcularPointwise.correct,payload]

def replayBoundRecord {C E : Type*} (savedRequest contractModel : Option (Record C E)) : Option (Model C E) :=
  if savedRequest=contractModel then savedRequest.bind decode else none

theorem replay_refuses_a_saved_ocular_request_different_from_the_applied_input_contract
    {C E : Type*} (savedRequest contractModel : Option (Record C E)) (different : savedRequest ≠ contractModel) :
    replayBoundRecord savedRequest contractModel=none := by simp [replayBoundRecord,different]

theorem replay_of_the_matching_canonical_record_recovers_the_frozen_model
    {C E : Type*} (model : Model C E) (valid : 0 ≤ model.strength ∧ model.strength ≤ 1) :
    replayBoundRecord (some (encode model)) (some (encode model))=some model := by
  simp [replayBoundRecord,every_canonical_numeric_source_model_roundtrips model valid]

def commonAverage {C : Type*} [Fintype C] (x : C → ℝ) (c : C) : ℝ :=
  x c-(∑ channel, x channel)/(Fintype.card C : ℝ)

theorem CAR_of_corrected_EEG_uses_only_the_selected_EEG_coordinate_average
    {C E : Type*} [Fintype C] [Fintype E] (B : C → E → ℝ)
    (center eye : E → ℝ) (eeg : C → ℝ) (strength : ℝ) (c : C) :
    commonAverage (fun channel => OcularPointwise.correct (B channel) center strength (eeg channel) eye) c=
      commonAverage eeg c-strength*∑ e,
        (B c e-(∑ channel, B channel e)/(Fintype.card C : ℝ))*(eye e-center e) := by
  have swap : (∑ channel, ∑ e, B channel e*(eye e-center e))=
      ∑ e, (∑ channel, B channel e)*(eye e-center e) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro e _
    rw [Finset.sum_mul]
  have centeredCoefficient :
      (∑ e, (B c e-(∑ channel, B channel e)/(Fintype.card C : ℝ))*(eye e-center e))=
      (∑ e, B c e*(eye e-center e))-
        (∑ channel, ∑ e, B channel e*(eye e-center e))/(Fintype.card C : ℝ) := by
    simp only [sub_mul,Finset.sum_sub_distrib]
    congr 1
    rw [swap,Finset.sum_div]
    apply Finset.sum_congr rfl
    intro e _
    rw [div_mul_eq_mul_div₀]
  rw [centeredCoefficient]
  simp only [commonAverage,OcularPointwise.correct,Finset.sum_sub_distrib,← Finset.mul_sum]
  ring

end
end N2P3.OcularReferenceRecord
