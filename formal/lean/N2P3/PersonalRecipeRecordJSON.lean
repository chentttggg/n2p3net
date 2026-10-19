import N2P3.PersonalRecipeRecord

/-! Fieldwise JSON normalization of the actual conditionally omitted record.
Only individual integer codec replies are premises. Finite builtin/NumPy
float64 values, tuple/list recursion and literal JSON errors stay explicit.
The pipeline is the recipe fragment of metadata JSON, not a certificate of
the other physical/QC/metadata ingress fields or native codec implementations.
-/
namespace N2P3.PersonalRecipeRecordJSON
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalRecipeRecord
noncomputable section
set_option maxHeartbeats 1000000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

mutual
def normalValue : Value → Value
  | .numpyFloat64 value => .floating value
  | .list values | .tuple values => .list (normalValues values)
  | .object fields => .object (normalFields fields)
  | value => value
def normalValues : Values → Values
  | .nil => .nil
  | .cons value tail => .cons (normalValue value) (normalValues tail)
def normalFields : Fields → Fields
  | .nil => .nil
  | .cons field value tail => .cons field (normalValue value) (normalFields tail)
end

mutual
def JSONLeafFacts (p : Primitives) : Value → Prop
  | .null | .boolean _ | .text _ | .floating (.finite _) | .numpyFloat64 (.finite _) => True
  | .integer value => p.jsonInteger value = .ok ()
  | .list values | .tuple values => JSONValuesFacts p values
  | .object fields => JSONFieldsFacts p fields
  | _ => False
def JSONValuesFacts (p : Primitives) : Values → Prop
  | .nil => True
  | .cons value tail => JSONLeafFacts p value ∧ JSONValuesFacts p tail
def JSONFieldsFacts (p : Primitives) : Fields → Prop
  | .nil => True
  | .cons _ value tail => JSONLeafFacts p value ∧ JSONFieldsFacts p tail
end

mutual
theorem normalized_value_from_individual_leaves (p : Primitives) (field : String) :
    ∀ value, JSONLeafFacts p value → PersonalRecipeJSON.normalize p field value = .ok (normalValue value)
  | .null, _ => rfl
  | .boolean _, _ => rfl
  | .text _, _ => rfl
  | .integer value, h => by
    simp only [JSONLeafFacts] at h
    simp [PersonalRecipeJSON.normalize, normalValue, atom, h]
  | .floating (.finite _), _ => rfl
  | .floating .nan, h => False.elim h
  | .floating .positiveInfinity, h => False.elim h
  | .floating .negativeInfinity, h => False.elim h
  | .numpyFloat64 (.finite _), _ => rfl
  | .numpyFloat64 .nan, h => False.elim h
  | .numpyFloat64 .positiveInfinity, h => False.elim h
  | .numpyFloat64 .negativeInfinity, h => False.elim h
  | .numpyInteger _, h => False.elim h
  | .numpyFloat32 _, h => False.elim h
  | .numpyBoolean _, h => False.elim h
  | .nonstringKeyDict, h => False.elim h
  | .list values, h => by
    simp [PersonalRecipeJSON.normalize, normalValue, normalized_values_from_individual_leaves p field values h]
  | .tuple values, h => by
    simp [PersonalRecipeJSON.normalize, normalValue, normalized_values_from_individual_leaves p field values h]
  | .object fields, h => by
    simp [PersonalRecipeJSON.normalize, normalValue, normalized_fields_from_individual_leaves p fields h]

theorem normalized_values_from_individual_leaves (p : Primitives) (field : String) :
    ∀ values, JSONValuesFacts p values → PersonalRecipeJSON.normalizeValues p field values = .ok (normalValues values)
  | .nil, _ => rfl
  | .cons value tail, h => by
    simp [PersonalRecipeJSON.normalizeValues, normalValues,
      normalized_value_from_individual_leaves p field value h.1,
      normalized_values_from_individual_leaves p field tail h.2]

theorem normalized_fields_from_individual_leaves (p : Primitives) :
    ∀ fields, JSONFieldsFacts p fields → PersonalRecipeJSON.normalizeFields p fields = .ok (normalFields fields)
  | .nil, _ => rfl
  | .cons field value tail, h => by
    simp [PersonalRecipeJSON.normalizeFields, normalFields,
      normalized_value_from_individual_leaves p field value h.1,
      normalized_fields_from_individual_leaves p tail h.2]
end

theorem omission_preserves_individual_JSON_leaf_facts (p : Primitives) (key : String) :
    ∀ fields, JSONFieldsFacts p fields → JSONFieldsFacts p (erase fields key)
  | .nil, _ => trivial
  | .cons name value tail, h => by
    have ih := omission_preserves_individual_JSON_leaf_facts p key tail h.2
    by_cases hn : name = key
    · simpa [erase, hn] using ih
    · simpa [erase, hn, JSONFieldsFacts] using And.intro h.1 ih

theorem actual_record_image_JSON_facts (p : Primitives) (r : Raw) (temperature prior width : Bool)
    (h : JSONFieldsFacts p (allRecordFields r)) : JSONFieldsFacts p (image r temperature prior width) := by
  unfold image
  split_ifs <;> (repeat' apply omission_preserves_individual_JSON_leaf_facts) <;> exact h

theorem normal_fields_commute_with_actual_key_removal (key : String) :
    ∀ fields, normalFields (erase fields key) = erase (normalFields fields) key
  | .nil => rfl
  | .cons name value tail => by
    have ih := normal_fields_commute_with_actual_key_removal key tail
    by_cases hn : name = key <;> simp [erase, normalFields, hn, ih]

def normalRaw (r : Raw) : Raw := {
  name := normalValue r.name, initialization := normalValue r.initialization,
  inputStatistics := normalValue r.inputStatistics, buffers := normalValue r.buffers,
  affine := normalValue r.affine, weights := normalValue r.weights, dropout := normalValue r.dropout,
  batchSize := normalValue r.batchSize, lr := normalValue r.lr, weightDecay := normalValue r.weightDecay,
  epochs := normalValue r.epochs, maxUpdates := normalValue r.maxUpdates, seed := normalValue r.seed,
  refit := normalValue r.refit, adapter := .raw (normalValue (adapterRecordValue r.adapter)),
  lossWeight := normalValue r.lossWeight, temperature := normalValue r.temperature, prior := normalValue r.prior,
  binWidth := normalValue r.binWidth, temporalKind := normalValue r.temporalKind, temporalRank := normalValue r.temporalRank }

theorem fieldwise_JSON_normalization_is_not_a_whole_record_oracle (r : Raw) :
    normalFields (allRecordFields r) = allRecordFields (normalRaw r) := by
  simp [allRecordFields, normalRaw, normalFields, Fields.ofList, adapterRecordValue]

theorem JSON_normalization_preserves_None_identity (value : Value) : normalValue value = .null ↔ value = .null := by
  cases value <;> simp only [normalValue] <;> first
  | exact Iff.rfl
  | constructor <;> intro h <;> cases h

theorem nested_adapter_None_identity_after_JSON (r : Raw) :
    (normalRaw r).adapter = .raw .null ↔ r.adapter = .raw .null := by
  change AdapterArgument.raw (normalValue (adapterRecordValue r.adapter)) = .raw .null ↔ _
  rw [AdapterArgument.raw.injEq, JSON_normalization_preserves_None_identity]
  cases r.adapter with
  | raw value => simp only [adapterRecordValue, AdapterArgument.raw.injEq]
  | existing config => constructor <;> intro h <;> cases h

theorem actual_omission_and_JSON_normalization_commute (r : Raw) (temperature prior width : Bool) :
    normalFields (image r temperature prior width) = image (normalRaw r) temperature prior width := by
  have hl' : (normalRaw r).lossWeight = .null ↔ r.lossWeight = .null := JSON_normalization_preserves_None_identity _
  have hk' : (normalRaw r).temporalKind = .null ↔ r.temporalKind = .null := JSON_normalization_preserves_None_identity _
  by_cases ha : r.adapter = .raw .null <;> by_cases hl : r.lossWeight = .null <;>
    by_cases hk : r.temporalKind = .null <;> cases temperature <;> cases prior <;> cases width <;>
      simp [image, nested_adapter_None_identity_after_JSON, hl', hk',
        normal_fields_commute_with_actual_key_removal,
        fieldwise_JSON_normalization_is_not_a_whole_record_oracle, ha, hl, hk]

def recordJSON (copy : CopyOperations) (p : Primitives) (r : Raw) : RecordResult Fields := do
  let fields ← record copy p r
  (PersonalRecipeJSON.normalizeFields p fields).mapError PersonalRecipeRecord.Fault.operation

def saveThenReconstruct (copy : CopyOperations) (p : Primitives) (r : Raw) : RecordResult Recipe := do
  let fields ← recordJSON copy p r
  (fromKwargs p fields).mapError PersonalRecipeRecord.Fault.operation

theorem saved_record_JSON_image_from_individual_operations (copy : CopyOperations) (p : Primitives)
    (r : Raw) (temperature prior width : Bool) (hc : CopyFieldsFacts copy (allRecordFields r))
    (he : EqualityAtoms p r temperature prior width) (hj : JSONFieldsFacts p (allRecordFields r)) :
    recordJSON copy p r = .ok (image (normalRaw r) temperature prior width) := by
  rw [recordJSON, record_image_from_individual_copy_and_equality_atoms copy p r temperature prior width hc he]
  simp [normalized_fields_from_individual_leaves p (image r temperature prior width)
    (actual_record_image_JSON_facts p r temperature prior width hj), actual_omission_and_JSON_normalization_commute]

theorem actual_record_to_reconstructed_constructor_formula (copy : CopyOperations) (p : Primitives)
    (r : Raw) (temperature prior width : Bool) (hc : CopyFieldsFacts copy (allRecordFields r))
    (he : EqualityAtoms p r temperature prior width) (hj : JSONFieldsFacts p (allRecordFields r)) :
    saveThenReconstruct copy p r =
      (validate p (restored (normalRaw r) temperature prior width)).mapError PersonalRecipeRecord.Fault.operation := by
  rw [saveThenReconstruct, saved_record_JSON_image_from_individual_operations copy p r temperature prior width hc he hj]
  simp [fromKwargs, omitted_defaults_reconstruct_the_explicit_normalized_raw]

theorem integer_default_omission_changes_temperature_and_prior_tags (r : Raw)
    (ht : r.temperature = .integer 1) (hp : r.prior = .integer 0) (hl : r.lossWeight = .null) :
    r.temperature = .integer 1 ∧ r.prior = .integer 0 ∧
    (restored (normalRaw r) true true true).temperature = .floating (.finite 1) ∧
    (restored (normalRaw r) true true true).prior = .floating (.finite 0) := by
  refine ⟨ht, hp, ?_⟩
  simp [restored, normalRaw, normalValue, hl]

theorem numpy_float64_JSON_changes_retained_numeric_tag (value : ℝ) :
    normalValue (.numpyFloat64 (.finite value)) = .floating (.finite value) := rfl

end
end N2P3.PersonalRecipeRecordJSON
