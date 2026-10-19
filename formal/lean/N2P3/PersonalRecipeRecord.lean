import N2P3.PersonalFullRecipeOutcome

/-!
# Actual dataclass record and conditional default omission

recipes.py:100-117 calls asdict BEFORE default equality tests, recursively
includes the ten FeatureTransferConfig constructor fields, and conditionally
removes six groups of keys. The full21-key kwargs image is not the saved
record. This module keeps each NumPy scalar deepcopy and each equality test
as its own operation. No whole record/asdict/constructor-success oracle is
used. Mutated typed objects, custom classes and unrepresented nonstring-key
dictionary contents remain explicit native/representation obligations.
-/
namespace N2P3.PersonalRecipeRecord
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe
noncomputable section
set_option maxHeartbeats 800000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Fault where
  | deepcopy (field : String) (exception : Exception)
  | operation (fault : PersonalRecipePython.Fault)
  deriving DecidableEq

abbrev RecordResult (α : Type) := Except Fault α

structure CopyOperations where
  numpyScalar : Value → Except Exception Unit

def copyScalar (p : CopyOperations) (field : String) (value : Value) : RecordResult Value := do
  (p.numpyScalar value).mapError (Fault.deepcopy field)
  pure value

mutual
def copyValue (p : CopyOperations) (field : String) : Value → RecordResult Value
  | .null => .ok .null
  | .boolean value => .ok (.boolean value)
  | .integer value => .ok (.integer value)
  | .floating value => .ok (.floating value)
  | .text value => .ok (.text value)
  | .numpyInteger value => copyScalar p field (.numpyInteger value)
  | .numpyFloat64 value => copyScalar p field (.numpyFloat64 value)
  | .numpyFloat32 value => copyScalar p field (.numpyFloat32 value)
  | .numpyBoolean value => copyScalar p field (.numpyBoolean value)
  | .list values => .list <$> copyValues p field values
  | .tuple values => .tuple <$> copyValues p field values
  | .object fields => .object <$> copyFields p fields
  | .nonstringKeyDict => .error (.deepcopy field (.unrefined "asdict nonstring-key dictionary contents"))

def copyValues (p : CopyOperations) (field : String) : Values → RecordResult Values
  | .nil => .ok .nil
  | .cons value tail => do
    let value ← copyValue p field value
    let tail ← copyValues p field tail
    pure (.cons value tail)

def copyFields (p : CopyOperations) : Fields → RecordResult Fields
  | .nil => .ok .nil
  | .cons field value tail => do
    let value ← copyValue p field value
    let tail ← copyFields p tail
    pure (.cons field value tail)
end

mutual
inductive CopyValueFacts (p : CopyOperations) : Value → Prop where
  | null : CopyValueFacts p .null
  | boolean (value : Bool) : CopyValueFacts p (.boolean value)
  | integer (value : ℤ) : CopyValueFacts p (.integer value)
  | floating (value : Number) : CopyValueFacts p (.floating value)
  | text (value : String) : CopyValueFacts p (.text value)
  | numpyInteger (value : ℤ) (atom : p.numpyScalar (.numpyInteger value) = .ok ()) : CopyValueFacts p (.numpyInteger value)
  | numpyFloat64 (value : Number) (atom : p.numpyScalar (.numpyFloat64 value) = .ok ()) : CopyValueFacts p (.numpyFloat64 value)
  | numpyFloat32 (value : Number) (atom : p.numpyScalar (.numpyFloat32 value) = .ok ()) : CopyValueFacts p (.numpyFloat32 value)
  | numpyBoolean (value : Bool) (atom : p.numpyScalar (.numpyBoolean value) = .ok ()) : CopyValueFacts p (.numpyBoolean value)
  | list (values : Values) (facts : CopyValuesFacts p values) : CopyValueFacts p (.list values)
  | tuple (values : Values) (facts : CopyValuesFacts p values) : CopyValueFacts p (.tuple values)
  | object (fields : Fields) (facts : CopyFieldsFacts p fields) : CopyValueFacts p (.object fields)
inductive CopyValuesFacts (p : CopyOperations) : Values → Prop where
  | nil : CopyValuesFacts p .nil
  | cons (value : Value) (tail : Values) (head : CopyValueFacts p value) (rest : CopyValuesFacts p tail) :
      CopyValuesFacts p (.cons value tail)
inductive CopyFieldsFacts (p : CopyOperations) : Fields → Prop where
  | nil : CopyFieldsFacts p .nil
  | cons (field : String) (value : Value) (tail : Fields)
      (head : CopyValueFacts p value) (rest : CopyFieldsFacts p tail) : CopyFieldsFacts p (.cons field value tail)
end

mutual
theorem value_copy_from_each_leaf (p : CopyOperations) (field : String) :
    ∀ {value}, CopyValueFacts p value → copyValue p field value = .ok value
  | _, .null => rfl
  | _, .boolean _ => rfl
  | _, .integer _ => rfl
  | _, .floating _ => rfl
  | _, .text _ => rfl
  | _, .numpyInteger _ atom => by simp [copyValue, copyScalar, atom]
  | _, .numpyFloat64 _ atom => by simp [copyValue, copyScalar, atom]
  | _, .numpyFloat32 _ atom => by simp [copyValue, copyScalar, atom]
  | _, .numpyBoolean _ atom => by simp [copyValue, copyScalar, atom]
  | _, .list _ facts => by simp [copyValue, values_copy_from_each_leaf p field facts]
  | _, .tuple _ facts => by simp [copyValue, values_copy_from_each_leaf p field facts]
  | _, .object _ facts => by simp [copyValue, fields_copy_from_each_leaf p facts]

theorem values_copy_from_each_leaf (p : CopyOperations) (field : String) :
    ∀ {values}, CopyValuesFacts p values → copyValues p field values = .ok values
  | _, .nil => rfl
  | _, .cons _ _ head rest => by
    simp [copyValues, value_copy_from_each_leaf p field head, values_copy_from_each_leaf p field rest]

theorem fields_copy_from_each_leaf (p : CopyOperations) :
    ∀ {fields}, CopyFieldsFacts p fields → copyFields p fields = .ok fields
  | _, .nil => rfl
  | _, .cons field _ _ head rest => by
    simp [copyFields, value_copy_from_each_leaf p field head, fields_copy_from_each_leaf p rest]
end

def erase (fields : Fields) (key : String) : Fields :=
  match fields with
  | .nil => .nil
  | .cons name value tail => if name = key then erase tail key else .cons name value (erase tail key)

def equal (p : Primitives) (field : String) (left right : Value) : RecordResult Bool :=
  (atom field (.compare .eq) (p.compare .eq left right)).mapError Fault.operation

def record (copy : CopyOperations) (p : Primitives) (r : Raw) : RecordResult Fields := do
  let fields ← copyFields copy (allRecordFields r)
  let fields := if r.adapter = .raw .null then erase fields "target_adapter_config" else fields
  let fields ← if r.lossWeight = .null then do
    let fields := erase fields "candidate_loss_weight"
    if ← equal p "record temperature default" r.temperature (.floating (.finite 1)) then
      pure (erase fields "candidate_loss_temperature")
    else pure fields
  else pure fields
  let fields ← if ← equal p "record source prior default" r.prior (.integer 0) then
    pure (erase fields "source_prior_weight") else pure fields
  let fields ← if ← equal p "record bin width default" r.binWidth (.integer 1) then
    pure (erase fields "erp_bin_width") else pure fields
  let fields := if r.temporalKind = .null then erase (erase fields "temporal_basis_kind") "temporal_rank" else fields
  pure fields

def image (r : Raw) (temperature prior width : Bool) : Fields :=
  let fields := allRecordFields r
  let fields := if r.adapter = .raw .null then erase fields "target_adapter_config" else fields
  let fields := if r.lossWeight = .null then
    let fields := erase fields "candidate_loss_weight"
    if temperature then erase fields "candidate_loss_temperature" else fields
    else fields
  let fields := if prior then erase fields "source_prior_weight" else fields
  let fields := if width then erase fields "erp_bin_width" else fields
  if r.temporalKind = .null then erase (erase fields "temporal_basis_kind") "temporal_rank" else fields

structure EqualityAtoms (p : Primitives) (r : Raw) (temperature prior width : Bool) : Prop where
  temperature : r.lossWeight = .null → p.compare .eq r.temperature (.floating (.finite 1)) = .ok temperature
  prior : p.compare .eq r.prior (.integer 0) = .ok prior
  width : p.compare .eq r.binWidth (.integer 1) = .ok width

theorem record_image_from_individual_copy_and_equality_atoms (copy : CopyOperations) (p : Primitives)
    (r : Raw) (temperature prior width : Bool) (hc : CopyFieldsFacts copy (allRecordFields r))
    (he : EqualityAtoms p r temperature prior width) : record copy p r = .ok (image r temperature prior width) := by
  rw [record, fields_copy_from_each_leaf copy hc]
  by_cases hl : r.lossWeight = .null
  · cases temperature <;> cases prior <;> cases width <;>
      simp [hl, equal, atom, he.temperature hl, he.prior, he.width, image]
  · cases temperature <;> cases prior <;> cases width <;>
      simp [hl, equal, atom, he.prior, he.width, image]

def explicitImage (r : Raw) (temperature prior width : Bool) : Fields := Fields.ofList (
  [("name", r.name), ("initialization", r.initialization), ("input_statistics", r.inputStatistics),
    ("bn_buffers", r.buffers), ("bn_affine", r.affine), ("weights", r.weights), ("dropout", r.dropout),
    ("batch_size", r.batchSize), ("lr", r.lr), ("weight_decay", r.weightDecay), ("epochs", r.epochs),
    ("max_updates", r.maxUpdates), ("seed", r.seed), ("refit_full_prefix", r.refit)] ++
  (if r.adapter = .raw .null then [] else [("target_adapter_config", adapterRecordValue r.adapter)]) ++
  (if r.lossWeight = .null then [] else [("candidate_loss_weight", r.lossWeight)]) ++
  (if r.lossWeight = .null ∧ temperature = true then [] else [("candidate_loss_temperature", r.temperature)]) ++
  (if prior then [] else [("source_prior_weight", r.prior)]) ++
  (if width then [] else [("erp_bin_width", r.binWidth)]) ++
  (if r.temporalKind = .null then [] else [("temporal_basis_kind", r.temporalKind), ("temporal_rank", r.temporalRank)]))

theorem conditional_pop_image_is_the_actual_saved_field_list (r : Raw) (temperature prior width : Bool) :
    image r temperature prior width = explicitImage r temperature prior width := by
  by_cases ha : r.adapter = .raw .null <;> by_cases hl : r.lossWeight = .null <;>
    by_cases hk : r.temporalKind = .null <;>
      cases temperature <;> cases prior <;> cases width <;>
        simp +decide [image, explicitImage, allRecordFields, erase, Fields.ofList, ha, hl, hk]

/-- Equality-based omission restores constructor defaults, which may change
integer/NumPy numeric tags into builtin float tags. The nested adapter snapshot
becomes its literal ten-field dictionary and is reconstructed independently. -/
def restored (r : Raw) (temperature prior width : Bool) : Raw :=
  { r with
    adapter := if r.adapter = .raw .null then .raw .null else .raw (adapterRecordValue r.adapter)
    temperature := if r.lossWeight = .null ∧ temperature = true then .floating (.finite 1) else r.temperature
    prior := if prior then .floating (.finite 0) else r.prior
    binWidth := if width then .integer 1 else r.binWidth
    temporalRank := if r.temporalKind = .null then .null else r.temporalRank }

theorem omitted_defaults_reconstruct_the_explicit_normalized_raw (r : Raw) (temperature prior width : Bool) :
    readArgs (image r temperature prior width) = .ok (restored r temperature prior width) := by
  rw [conditional_pop_image_is_the_actual_saved_field_list]
  by_cases ha : r.adapter = .raw .null <;> by_cases hl : r.lossWeight = .null <;>
    by_cases hk : r.temporalKind = .null <;>
      cases temperature <;> cases prior <;> cases width <;>
        simp +decide [explicitImage, restored, readArgs, keywordNames, allRecordFields,
          Fields.ofList, Fields.keys, Fields.get, getDefault, ha, hl, hk]

theorem numpy_copy_failure_is_before_default_comparisons (copy : CopyOperations)
    (field : String) (value : ℤ) (tail : Fields) (exception : Exception)
    (h : copy.numpyScalar (.numpyInteger value) = .error exception) :
    copyFields copy (.cons field (.numpyInteger value) tail) = .error (.deepcopy field exception) := by
  simp [copyFields, copyValue, copyScalar, h]

end
end N2P3.PersonalRecipeRecord
