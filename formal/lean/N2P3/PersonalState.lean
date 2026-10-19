import N2P3.StateLedger

/-!
# Literal decoded personal-predictor packets and preconstructor copy/restore

Source: calibrated.py:523-550,1194-1232 and calibration_candidates.py:310-332.
Metadata is the ENTIRE recursive value tree. Each named tensor retains its
name, dtype, shape and every ordered decoded finite scalar. Unknown metadata
fields and extra tensor names are retained: the source has no global exact-key
gate for the personal state. Object/state keys correspond to unique-key parsed
mappings; dictionary byte order, Torch storage/devices/strides, json/pickle/.pt
bytes, IEEE/nonfinite scalars, normalization/casts and physical I/O are outside
this immutable semantic domain.

The loader result is explicitly an input to CalibratedPredictor.__init__, after
the actual schema, metadata seal, state seal and legacy-QC gates and the first
recursive copy. It is NOT successful completion of every numeric constructor.
No constructor-success, whole-roundtrip or score-equality premise is used.
The five nonempty canonical packet witnesses exercise this preconstructor
layer; in particular their schematic neural tensors are not claimed to be a
complete executable N2P3Net state_dict. Neural forward/shape/BN/adapter gates,
ERP constructors, participant authority and QC fitting require later layers.

Hashes are arbitrary functions with no injectivity/authentication axiom. Their
actual implementations may canonicalize metadata or sort tensor names. Copy
preservation here is structural value preservation, not a proof that Python's
public tensors or dictionaries cannot be modified after construction.
-/

namespace N2P3.PersonalState

open N2P3.FrozenState N2P3.StateLedger

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

/-- Ordered decoded contents retain scalar tags; no IEEE byte codec is inferred. -/
structure Tensor where
  dtype : String
  shape : List ℕ
  values : List Value

def Tensor.numel (t : Tensor) : ℕ := t.shape.foldr (· * ·) 1

def Tensor.WellFormed (t : Tensor) : Prop :=
  t.values.length = t.numel ∧ ∀ v ∈ t.values, (numericScalar v).isSome

abbrev NamedState := List (String × Tensor)

def lookupTensor (name : String) : NamedState → Option Tensor
  | [] => none
  | (key, value) :: tail => if name = key then some value else lookupTensor name tail

def UniqueNames (state : NamedState) : Prop := (state.map Prod.fst).Nodup

def copyTensor (t : Tensor) : Tensor :=
  ⟨t.dtype, t.shape.map id, t.values.map copyValue⟩

def copyNamedState : NamedState → NamedState
  | [] => []
  | (name, tensor) :: tail => (name, copyTensor tensor) :: copyNamedState tail

theorem copyTensor_preserves (t : Tensor) : copyTensor t = t := by
  cases t with
  | mk dtype shape values =>
      have hv : values.map copyValue = values := by
        induction values with
        | nil => rfl
        | cons v tail ih => simp [copyValue_preserves, ih]
      simp [copyTensor, hv]

theorem copyNamedState_preserves (state : NamedState) : copyNamedState state = state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
      rcases entry with ⟨name, tensor⟩
      simp [copyNamedState, copyTensor_preserves, ih]

theorem copy_lookup_preserves_all_fields (state : NamedState) (name : String) :
    lookupTensor name (copyNamedState state) = lookupTensor name state := by
  rw [copyNamedState_preserves]

def encodeTensor (t : Tensor) : Value := .object (Fields.ofList
  [("dtype", .text t.dtype), ("shape", .sequence (encodeNats t.shape)),
   ("values", .sequence (valuesOfList t.values))])

def decodeTensor : Value → Option Tensor
  | .object fs => do
      let dtype ← (← fs.get "dtype").readText
      let shape ← (← fs.get "shape").readSequence
      let shape ← decodeNats shape
      let values ← (← fs.get "values").readSequence
      pure ⟨dtype, shape, valuesToList values⟩
  | _ => none

def encodeNamedFields : NamedState → Fields
  | [] => .nil
  | (name, tensor) :: tail => .cons name (encodeTensor tensor) (encodeNamedFields tail)

def decodeNamedFields : Fields → Option NamedState
  | .nil => some []
  | .cons name value tail => do
      let tensor ← decodeTensor value
      let rest ← decodeNamedFields tail
      pure ((name, tensor) :: rest)

def encodeState (state : NamedState) : Value := .object (encodeNamedFields state)

def decodeState : Value → Option NamedState
  | .object fs => decodeNamedFields fs
  | _ => none

theorem decode_encode_tensor (t : Tensor) : decodeTensor (encodeTensor t) = some t := by
  cases t
  simp [decodeTensor, encodeTensor, Fields.ofList, Fields.get, Value.readText,
    Value.readSequence, decode_encode_nats, values_list_roundtrip]

theorem decode_encode_named_fields (state : NamedState) :
    decodeNamedFields (encodeNamedFields state) = some state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
      rcases entry with ⟨name, tensor⟩
      simp [encodeNamedFields, decodeNamedFields, decode_encode_tensor, ih]

theorem decode_encode_state (state : NamedState) : decodeState (encodeState state) = some state :=
  decode_encode_named_fields state

theorem decoded_lookup_exact (state : NamedState) (name : String) :
    (decodeState (encodeState state)).bind (lookupTensor name) = lookupTensor name state := by
  rw [decode_encode_state]
  rfl

theorem tensor_encoding_injective : Function.Injective encodeTensor := by
  intro a b h
  have hd := congrArg decodeTensor h
  simpa only [decode_encode_tensor, Option.some.injEq] using hd

theorem named_state_encoding_injective : Function.Injective encodeState := by
  intro a b h
  have hd := congrArg decodeState h
  simpa only [decode_encode_state, Option.some.injEq] using hd

structure ConstructorInput where
  metadata : Value
  state : NamedState

def copyInput (input : ConstructorInput) : ConstructorInput :=
  ⟨copyValue input.metadata, copyNamedState input.state⟩

theorem copyInput_preserves (input : ConstructorInput) : copyInput input = input := by
  cases input
  simp [copyInput, copyValue_preserves, copyNamedState_preserves]

abbrev MetadataHash := Value → String
abbrev StateHash := NamedState → String

def currentSchema : String := "n2p3_calibrated_predictor/4"

def SupportedSchema (schema : Value) : Prop :=
  schema = .text currentSchema ∨ schema = .text "n2p3_calibrated_predictor/3" ∨
    schema = .text "n2p3_calibrated_predictor/1"

/-- Literal five outer payload fields. Full metadata/state are never projected away. -/
def payloadView (schema metadata state metadataSeal stateSeal : Value) : Value :=
  .object (Fields.ofList [("schema", schema), ("metadata", metadata),
    ("metadata_digest", metadataSeal), ("state", state), ("state_digest", stateSeal)])

def sealedView (mh : MetadataHash) (sh : StateHash) (schema : Value) (input : ConstructorInput) : Value :=
  payloadView schema input.metadata (encodeState input.state)
    (.text (mh input.metadata)) (.text (sh input.state))

/-- Current source saves /4; legacy schemas are readers, not alternate writer defaults. -/
def saveView (mh : MetadataHash) (sh : StateHash) (input : ConstructorInput) : Value :=
  sealedView mh sh (.text currentSchema) input

def objectFields (value : Value) (site : String) : Check Fields :=
  match value with
  | .object fs => .ok fs
  | _ => .error (.fieldFailure site)

def readState (value : Value) : Check NamedState :=
  match decodeState value with
  | some state => .ok state
  | none => .error (.fieldFailure "decoded tensor state")

/-- `is not None`, not Python truthiness: even false quality enters the legacy branch. -/
def checkLegacyQC (schema : Value) (metadata : Fields) : Check Unit := do
  let quality := fieldDefault metadata "epoch_quality"
  if schema = .text currentSchema ∨ quality = .null then pure ()
  else
    let q ← objectFields quality "epoch_quality[model]"
    let model ← required q "model"
    let modelFields ← objectFields model "epoch_quality.model.get"
    ensure (fieldDefault modelFields "method" = .text "fold_local_ptp_cv") "legacy epoch quality"

/-- This result only reaches the copied constructor INPUT, not a returned predictor. -/
def loadInput (mh : MetadataHash) (sh : StateHash) (raw : Value) : Check ConstructorInput := do
  let payload ← match raw with
    | .object fs => .ok fs
    | _ => .error (.sourceReject "schema")
  let schema := fieldDefault payload "schema"
  ensure (SupportedSchema schema) "schema"
  let metadata ← required payload "metadata"
  ensure (fieldDefault payload "metadata_digest" = .text (mh metadata)) "metadata digest"
  let state ← readState (← required payload "state")
  ensure (fieldDefault payload "state_digest" = .text (sh state)) "state digest"
  let metadataFields ← objectFields metadata "metadata.get(epoch_quality)"
  checkLegacyQC schema metadataFields
  pure (copyInput ⟨metadata, state⟩)

theorem current_skips_only_legacy_gate (metadata : Fields) :
    checkLegacyQC (.text currentSchema) metadata = .ok () := by
  simp [checkLegacyQC]

theorem current_source_copy_restore (mh : MetadataHash) (sh : StateHash)
    (metadata : Fields) (state : NamedState) :
    loadInput mh sh (saveView mh sh ⟨.object metadata, state⟩) = .ok ⟨.object metadata, state⟩ := by
  simp [loadInput, saveView, sealedView, payloadView, SupportedSchema, objectFields,
    readState, decode_encode_state, fieldDefault, required, Fields.ofList, Fields.get,
    ensure, current_skips_only_legacy_gate, copyInput_preserves]

theorem schema_rejected_first (mh : MetadataHash) (sh : StateHash)
    (schema metadata state metadataSeal stateSeal : Value) (h : ¬SupportedSchema schema) :
    loadInput mh sh (payloadView schema metadata state metadataSeal stateSeal) =
      .error (.sourceReject "schema") := by
  simp [loadInput, payloadView, fieldDefault, Fields.ofList, Fields.get, ensure, h]

theorem metadata_seal_before_state (mh : MetadataHash) (sh : StateHash)
    (schema metadata state metadataSeal stateSeal : Value)
    (hs : SupportedSchema schema) (hm : metadataSeal ≠ .text (mh metadata)) :
    loadInput mh sh (payloadView schema metadata state metadataSeal stateSeal) =
      .error (.sourceReject "metadata digest") := by
  simp [loadInput, payloadView, fieldDefault, Fields.ofList, Fields.get, required, ensure, hs, hm]

theorem state_seal_before_legacy_and_constructor (mh : MetadataHash) (sh : StateHash)
    (schema metadata : Value) (state : NamedState) (stateSeal : Value)
    (hs : SupportedSchema schema) (hst : stateSeal ≠ .text (sh state)) :
    loadInput mh sh (payloadView schema metadata (encodeState state)
      (.text (mh metadata)) stateSeal) = .error (.sourceReject "state digest") := by
  simp [loadInput, payloadView, fieldDefault, Fields.ofList, Fields.get, required, ensure,
    hs, readState, decode_encode_state, hst]

def legacyQuality (method : Value) : Value :=
  .object (Fields.ofList [("model", .object (Fields.ofList [("method", method)]))])

theorem legacy_quality_exact_method (schema method : Value) (h : schema ≠ .text currentSchema) :
    checkLegacyQC schema (Fields.ofList [("epoch_quality", legacyQuality method)]) =
      ensure (method = .text "fold_local_ptp_cv") "legacy epoch quality" := by
  simp [checkLegacyQC, fieldDefault, legacyQuality, Fields.ofList, Fields.get,
    objectFields, required, h]

theorem false_quality_is_not_missing (schema : Value) (h : schema ≠ .text currentSchema) :
    checkLegacyQC schema (Fields.ofList [("epoch_quality", .boolean false)]) =
      .error (.fieldFailure "epoch_quality[model]") := by
  simp [checkLegacyQC, fieldDefault, Fields.ofList, Fields.get, objectFields, h]

theorem no_quality_legacy_copy_restore (mh : MetadataHash) (sh : StateHash)
    (schema : Value) (hs : SupportedSchema schema) (state : NamedState) :
    loadInput mh sh (sealedView mh sh schema ⟨.object .nil, state⟩) = .ok ⟨.object .nil, state⟩ := by
  simp [loadInput, sealedView, payloadView, fieldDefault, Fields.ofList, Fields.get,
    required, ensure, hs, readState, decode_encode_state, objectFields, checkLegacyQC,
    copyInput_preserves]

inductive Kind where
  | sourceInitialized
  | scratch
  | targetERP
  | sourceTemporalERP
  | targetLDA
  deriving DecidableEq

def Kind.text : Kind → String
  | .sourceInitialized => "source_initialized_n2p3"
  | .scratch => "scratch_n2p3"
  | .targetERP => "target_only_structured_erp"
  | .sourceTemporalERP => "source_temporal_target_erp"
  | .targetLDA => "target_only_shrinkage_lda"

def decodeKind (value : Value) : Option Kind :=
  if value = .text Kind.sourceInitialized.text then some .sourceInitialized
  else if value = .text Kind.scratch.text then some .scratch
  else if value = .text Kind.targetERP.text then some .targetERP
  else if value = .text Kind.sourceTemporalERP.text then some .sourceTemporalERP
  else if value = .text Kind.targetLDA.text then some .targetLDA else none

theorem kind_text_roundtrip (kind : Kind) : decodeKind (.text kind.text) = some kind := by
  cases kind <;> simp [decodeKind, Kind.text]

/-- An observational accessor, not a premature model-kind gate in the loader. -/
def inputKind (input : ConstructorInput) : Option Kind :=
  match input.metadata with
  | .object fs => (fs.get "model_kind").bind decodeKind
  | _ => none

def prefixState (tag : String) (state : NamedState) : NamedState :=
  state.map (fun entry => (tag ++ entry.1, entry.2))

theorem prefixed_tensor_values_preserved (tag name : String) (tensor : Tensor) (tail : NamedState) :
    prefixState tag ((name, tensor) :: tail) =
      (tag ++ name, tensor) :: prefixState tag tail := rfl

def scalarTensor (dtype : String) (value : Value) : Tensor := ⟨dtype, [], [value]⟩
def vectorTensor (dtype : String) (values : List Value) : Tensor := ⟨dtype, [values.length], values⟩

def canonicalState : Kind → NamedState
  | .sourceInitialized | .scratch =>
      [("trunk.classifier.bias", vectorTensor "torch.float32" [.number 0, .number 0]),
       ("input_mean", ⟨"torch.float32", [1, 1, 1], [.number 0]⟩),
       ("input_std", ⟨"torch.float32", [1, 1, 1], [.number 1]⟩)]
  | .targetERP | .sourceTemporalERP =>
      [("erp.coefficient", vectorTensor "torch.float64" [.number 1]),
       ("erp.intercept", scalarTensor "torch.float64" (.number 0)),
       ("waveform_channels", vectorTensor "torch.int64" [.integer 0])]
  | .targetLDA =>
      [("feature_mean", vectorTensor "torch.float64" [.number 0]),
       ("feature_scale", vectorTensor "torch.float64" [.number 1]),
       ("coefficient", vectorTensor "torch.float64" [.number 1]),
       ("intercept", scalarTensor "torch.float64" (.number 0)),
       ("waveform_channels", vectorTensor "torch.int64" [.integer 0])]

def canonicalMetadata (kind : Kind) : Fields := Fields.ofList
  [("model_kind", .text kind.text), ("participant_key", .text "BrainSync:P0"),
   ("subject_id", .text "P0"), ("calibration_available_timestamp_s", .number 10),
   ("calibration_sessions", .sequence (valuesOfList
     [.object (Fields.ofList [("session_id", .text "calibration-0"),
       ("calibration_available_timestamp_s", .number 10)])])),
   ("input_contract", .object (Fields.ofList [("n_channels", .integer 1), ("n_times", .integer 1),
     ("channel_names", .sequence (encodeTexts ["Cz"])), ("source_reference", .text "left earlobe")])),
   ("input_request", .object (Fields.ofList [("channel_names", .sequence (encodeTexts ["Cz"]))])),
   ("recipe", .object (Fields.ofList [("name", .text "canonical-copy-fixture")])),
   ("provenance_extension", .object (Fields.ofList [("nested", .sequence (valuesOfList
     [.null, .boolean false, .integer 7, .number (3 / 2), .text "unchanged"]))]))]

def canonicalInput (kind : Kind) : ConstructorInput := ⟨.object (canonicalMetadata kind), canonicalState kind⟩

theorem canonical_five_kind_copy_restore (mh : MetadataHash) (sh : StateHash) (kind : Kind) :
    loadInput mh sh (saveView mh sh (canonicalInput kind)) = .ok (canonicalInput kind) :=
  current_source_copy_restore mh sh (canonicalMetadata kind) (canonicalState kind)

theorem canonical_five_kinds_identified (kind : Kind) : inputKind (canonicalInput kind) = some kind := by
  simp [canonicalInput, canonicalMetadata, inputKind, Fields.ofList, Fields.get, kind_text_roundtrip]

theorem canonical_five_states_nonempty (kind : Kind) : (canonicalState kind).length > 0 := by
  cases kind <;> simp [canonicalState]

theorem canonical_five_states_unique (kind : Kind) : UniqueNames (canonicalState kind) := by
  cases kind <;> simp [UniqueNames, canonicalState]

theorem canonical_five_tensor_shapes (kind : Kind) :
    ∀ entry ∈ canonicalState kind, entry.2.values.length = entry.2.numel := by
  cases kind <;> simp [canonicalState, scalarTensor, vectorTensor, Tensor.numel]

/-- Full recursive metadata is copied, including fields no current classifier reads. -/
theorem arbitrary_metadata_extension_preserved (mh : MetadataHash) (sh : StateHash)
    (key : String) (value : Value) (tail : Fields) (state : NamedState) :
    loadInput mh sh (saveView mh sh ⟨.object (.cons key value tail), state⟩) =
      .ok ⟨.object (.cons key value tail), state⟩ :=
  current_source_copy_restore mh sh _ state

theorem extra_named_tensor_preserved (mh : MetadataHash) (sh : StateHash)
    (metadata : Fields) (name : String) (tensor : Tensor) (tail : NamedState) :
    loadInput mh sh (saveView mh sh ⟨.object metadata, (name, tensor) :: tail⟩) =
      .ok ⟨.object metadata, (name, tensor) :: tail⟩ :=
  current_source_copy_restore mh sh metadata _

def zeroHashMetadata : MetadataHash := fun _ => "seal"
def zeroHashState : StateHash := fun _ => "seal"

def legacyNewQC : ConstructorInput :=
  ⟨.object (Fields.ofList [("epoch_quality", legacyQuality (.text "personal_epoch_signal_evidence"))]),
   canonicalState .targetLDA⟩

theorem legacy_new_quality_rejected :
    loadInput zeroHashMetadata zeroHashState
      (sealedView zeroHashMetadata zeroHashState (.text "n2p3_calibrated_predictor/1") legacyNewQC) =
      .error (.sourceReject "legacy epoch quality") := by
  simp [loadInput, sealedView, payloadView, legacyNewQC, fieldDefault, Fields.ofList, Fields.get,
    required, ensure, SupportedSchema, currentSchema, readState, decode_encode_state,
    objectFields, checkLegacyQC, legacyQuality]

theorem current_new_quality_reaches_constructor_input :
    loadInput zeroHashMetadata zeroHashState (saveView zeroHashMetadata zeroHashState legacyNewQC) =
      .ok legacyNewQC := current_source_copy_restore _ _ _ _

def changedDtype : ConstructorInput :=
  ⟨.object (canonicalMetadata .targetLDA),
   [("coefficient", vectorTensor "torch.int64" [.integer 1]) ]⟩

theorem same_seals_do_not_identify_tensor_state :
    zeroHashState (canonicalInput .targetLDA).state = zeroHashState changedDtype.state ∧
      (canonicalInput .targetLDA).state ≠ changedDtype.state := by
  constructor
  · rfl
  · intro h
    have hv := congrArg List.length h
    norm_num [canonicalInput, canonicalState, changedDtype] at hv

theorem changed_packet_still_has_own_copy_restore :
    loadInput zeroHashMetadata zeroHashState (saveView zeroHashMetadata zeroHashState changedDtype) =
      .ok changedDtype := current_source_copy_restore _ _ _ _

end

end N2P3.PersonalState
