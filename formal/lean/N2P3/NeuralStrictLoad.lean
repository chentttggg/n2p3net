import N2P3.NeuralRegisteredState
import N2P3.PersonalState

/-! NN03 success/refusal semantics over the actual registered slot table.
Ordinary BN's metadata-version-None counter completion and scalar [1] -> []
compatibility precede the per-slot copy. Cast replies are per numeric scalar,
and may be nonfinite; there is no whole-load success or value-preservation
oracle. This is not a model of partial native mutation/error-message ordering
on a refused load, hooks, tensor subclasses, meta/lazy allocation or swap mode. -/
namespace N2P3.NeuralStrictLoad

open N2P3.NeuralRegisteredState N2P3.PersonalState N2P3.FrozenState
open N2P3.StateLedger (numericScalar)
open N2P3.PersonalRecipePython (Number)
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map

inductive Failure where
  | missing (key : String) | unexpected | shape (key : String) | scalar | cast
  deriving DecidableEq

structure RuntimeTensor where
  dtype : String
  shape : List ℕ
  values : List Number

abbrev RuntimeState := List (String × RuntimeTensor)
abbrev Cast := String → ℝ → Except Failure Number

def zeroCounter : Tensor := ⟨"torch.int64",[],[.integer 0]⟩

def sourceForSlot (slot : Slot) (source : NamedState) : Option Tensor :=
  match lookupTensor slot.name source with
  | some tensor => some tensor
  | none => if slot.legacyCounter then some zeroCounter else none

def scalarCompatibility (slot : Slot) (source : Tensor) : Tensor :=
  if slot.shape=[] ∧ source.shape=[1] then {source with shape:=[]} else source

def copyValues (cast : Cast) (dtype : String) : List Value → Except Failure (List Number)
  | [] => .ok []
  | head::tail => do
    let value ← match numericScalar head with | none => .error .scalar | some value => .ok value
    let copied ← cast dtype value
    let rest ← copyValues cast dtype tail
    .ok (copied::rest)

def loadOne (cast : Cast) (slot : Slot) (source : NamedState) : Except Failure RuntimeTensor := do
  let tensor ← match sourceForSlot slot source with | none => .error (.missing slot.name) | some tensor => .ok tensor
  let prepared := scalarCompatibility slot tensor
  if prepared.shape≠slot.shape then .error (.shape slot.name)
  else
    let values ← copyValues cast slot.dtype prepared.values
    .ok ⟨slot.dtype,slot.shape,values⟩

def loadEntries (cast : Cast) (source : NamedState) : List Slot → Except Failure RuntimeState
  | [] => .ok []
  | slot::tail => do
    let tensor ← loadOne cast slot source
    let rest ← loadEntries cast source tail
    .ok ((slot.name,tensor)::rest)

def noUnexpected (expected : List Slot) (source : NamedState) : Prop :=
  ∀ entry ∈ source,entry.1 ∈ expected.map (·.name)

def strictLoad (cast : Cast) (expected : List Slot) (source : NamedState) : Except Failure RuntimeState :=
  if noUnexpected expected source then loadEntries cast source expected else .error .unexpected

def isFinite : Number → Bool
  | .finite _ => true
  | _ => false

def runtimeFinite (tensor : RuntimeTensor) : Prop := ∀ value ∈ tensor.values,isFinite value=true

def positiveSaved (value : Value) : Prop := ∃ scalar,numericScalar value=some scalar ∧ 0<scalar

def validInputStats (channels : ℕ) (mean std : Tensor) : Prop :=
  mean.shape=[1,channels,1] ∧ std.shape=mean.shape ∧ ∀ value ∈ std.values,positiveSaved value

def inputStatsGuard (channels : ℕ) (source : NamedState) : Bool :=
  match lookupTensor "input_mean" source,lookupTensor "input_std" source with
  | some mean,some std => decide (validInputStats channels mean std)
  | _,_ => false

theorem existing_BN_counter_is_never_overwritten_by_legacy_completion
    (slot : Slot) (source : NamedState) (tensor : Tensor)
    (present : lookupTensor slot.name source=some tensor) : sourceForSlot slot source=some tensor := by
  simp [sourceForSlot,present]

theorem missing_ordinary_BN_counter_is_completed_from_the_actual_zero_destination
    (slot : Slot) (source : NamedState) (counter : slot.legacyCounter=true)
    (absent : lookupTensor slot.name source=none) : sourceForSlot slot source=some zeroCounter := by
  simp [sourceForSlot,absent,counter]

theorem missing_noncounter_is_not_synthesized (slot : Slot) (source : NamedState)
    (notCounter : slot.legacyCounter=false) (absent : lookupTensor slot.name source=none) :
    sourceForSlot slot source=none := by simp [sourceForSlot,absent,notCounter]

theorem scalar_backward_compatibility_keeps_the_single_value_and_source_dtype
    (slot : Slot) (source : Tensor) (destination : slot.shape=[]) (one : source.shape=[1]) :
    (scalarCompatibility slot source).shape=[] ∧
      (scalarCompatibility slot source).values=source.values ∧
      (scalarCompatibility slot source).dtype=source.dtype := by
  simp [scalarCompatibility,destination,one]

theorem nonscalar_compatibility_does_not_invent_a_reshape (slot : Slot) (source : Tensor)
    (notScalar : slot.shape≠[]) : scalarCompatibility slot source=source := by
  simp [scalarCompatibility,notScalar]

theorem missing_noncounter_refuses_its_slot_before_any_value_copy
    (cast : Cast) (slot : Slot) (source : NamedState)
    (notCounter : slot.legacyCounter=false) (absent : lookupTensor slot.name source=none) :
    loadOne cast slot source=.error (.missing slot.name) := by
  simp [loadOne,missing_noncounter_is_not_synthesized slot source notCounter absent]

theorem successful_slot_copy_keeps_destination_dtype_and_shape
    (cast : Cast) (slot : Slot) (source : NamedState) (loaded : RuntimeTensor)
    (success : loadOne cast slot source=.ok loaded) : loaded.dtype=slot.dtype ∧ loaded.shape=slot.shape := by
  unfold loadOne at success
  cases found : sourceForSlot slot source with
  | none => simp [found] at success
  | some tensor =>
    by_cases wrongShape : (scalarCompatibility slot tensor).shape≠slot.shape
    · simp [found,wrongShape] at success
    · cases converted : copyValues cast slot.dtype (scalarCompatibility slot tensor).values with
      | error fault => simp [found,wrongShape,converted] at success
      | ok values =>
        simp [found,wrongShape,converted] at success
        subst loaded
        exact ⟨rfl,rfl⟩

theorem successful_complete_load_has_every_registered_name_in_source_order
    (cast : Cast) (source : NamedState) (expected : List Slot) (loaded : RuntimeState)
    (success : loadEntries cast source expected=.ok loaded) : loaded.map Prod.fst=expected.map (·.name) := by
  induction expected generalizing loaded with
  | nil => simp [loadEntries] at success; subst loaded; rfl
  | cons slot tail ih =>
    cases first : loadOne cast slot source with
    | error fault => simp [loadEntries,first] at success
    | ok tensor =>
      cases rest : loadEntries cast source tail with
      | error fault => simp [loadEntries,first,rest] at success
      | ok remaining =>
        simp [loadEntries,first,rest] at success
        subst loaded
        simp [ih remaining rest]

theorem strict_load_keeps_the_complete_registered_parameter_and_buffer_count
    (cast : Cast) (source : NamedState) (expected : List Slot) (loaded : RuntimeState)
    (success : strictLoad cast expected source=.ok loaded) : loaded.length=expected.length := by
  unfold strictLoad at success
  split_ifs at success with allowed
  · have names := successful_complete_load_has_every_registered_name_in_source_order cast source expected loaded success
    simpa using congrArg List.length names

theorem unexpected_projected_keys_refuse_strict_load
    (cast : Cast) (expected : List Slot) (source : NamedState)
    (extra : ¬noUnexpected expected source) : strictLoad cast expected source=.error .unexpected := by
  simp [strictLoad,extra]

def preservingCast : Cast := fun _ value => .ok (.finite value)

theorem individual_numeric_and_cast_replies_derive_the_whole_value_copy
    (cast : Cast) (dtype : String) (values : List Value) (output : Value → Number)
    (atoms : ∀ value ∈ values,∃ scalar,numericScalar value=some scalar ∧ cast dtype scalar=.ok (output value)) :
    copyValues cast dtype values=.ok (values.map output) := by
  induction values with
  | nil => rfl
  | cons head tail ih =>
    obtain ⟨scalar,found,copied⟩ := atoms head (by simp)
    have rest : ∀ value ∈ tail,∃ scalar,numericScalar value=some scalar ∧ cast dtype scalar=.ok (output value) :=
      fun value member => atoms value (by simp [member])
    simp [copyValues,found,copied,ih rest]

theorem independent_shape_and_scalar_operations_derive_one_registered_tensor
    (cast : Cast) (slot : Slot) (source : NamedState) (tensor : Tensor) (output : Value → Number)
    (found : sourceForSlot slot source=some tensor)
    (shape : (scalarCompatibility slot tensor).shape=slot.shape)
    (atoms : ∀ value ∈ (scalarCompatibility slot tensor).values,
      ∃ scalar,numericScalar value=some scalar ∧ cast slot.dtype scalar=.ok (output value)) :
    loadOne cast slot source=.ok ⟨slot.dtype,slot.shape,(scalarCompatibility slot tensor).values.map output⟩ := by
  have copied := individual_numeric_and_cast_replies_derive_the_whole_value_copy cast slot.dtype _ output atoms
  simp [loadOne,found,shape,copied]

theorem complete_registered_load_is_derived_from_each_tensor_copy
    (cast : Cast) (source : NamedState) (expected : List Slot) (values : Slot → RuntimeTensor)
    (each : ∀ slot ∈ expected,loadOne cast slot source=.ok (values slot)) :
    loadEntries cast source expected=.ok (expected.map (fun slot => (slot.name,values slot))) := by
  induction expected with
  | nil => rfl
  | cons slot tail ih =>
    have first := each slot (by simp)
    have rest : ∀ entry ∈ tail,loadOne cast entry source=.ok (values entry) :=
      fun entry member => each entry (by simp [member])
    simp [loadEntries,first,ih rest]

theorem preserving_individual_scalar_casts_give_exact_value_copy
    (dtype : String) (values : List Value) (numeric : ∀ value ∈ values,(numericScalar value).isSome=true) :
    copyValues preservingCast dtype values=.ok (values.map (fun value => Number.finite ((numericScalar value).getD 0))) := by
  induction values with
  | nil => rfl
  | cons head tail ih =>
    have headNumeric := numeric head (by simp)
    have tailNumeric : ∀ value ∈ tail,(numericScalar value).isSome=true := fun value member => numeric value (by simp [member])
    cases found : numericScalar head with
    | none => simp [found] at headNumeric
    | some scalar => simp [copyValues,found,preservingCast,ih tailNumeric]

def finiteSavedTensor : Tensor := ⟨"torch.float64",[1],[.number 1]⟩
def weightSlot : Slot := parameter "torch.float32" "weight" [1]
def infinityCast : Cast := fun _ _ => .ok .positiveInfinity

theorem finite_saved_tensor_does_not_constrain_an_unspecified_cast_reply :
    loadOne infinityCast weightSlot [("weight",finiteSavedTensor)]=
      .ok ⟨"torch.float32",[1],[.positiveInfinity]⟩ ∧
      ¬runtimeFinite ⟨"torch.float32",[1],[.positiveInfinity]⟩ := by
  simp [loadOne,sourceForSlot,lookupTensor,weightSlot,parameter,scalarCompatibility,
    finiteSavedTensor,copyValues,numericScalar,infinityCast,runtimeFinite,isFinite]

end
end N2P3.NeuralStrictLoad
