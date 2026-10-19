import N2P3.ERPTemporalSnapshot

/-!
# Raw NPZ member access and literal attested-loader prefix

Source: epochs.py:1324-1454. Each readMember is a separate NpzFile access:
members other than retained X are deliberately read again at source sites.
The array image is raw, with arbitrary rank/dtype/nonfinite atoms; no decoded
Dataset or complete validation result can be returned by any primitive.

This prefix does not call dataset.validate. Identity/timeline/lineage
constructors, dataset.record(validate=False), attested comparison and physical
return gates belong to the following loader stages. Scalar string/bool, dtype
casts, UTF8/JSON and pandas table codecs remain individual operation leaves.
-/

namespace N2P3.ERPTemporalNPZ

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot

noncomputable section

inductive Kind where
  | signed | unsigned | uint8 | boolean | floating | complex | unicode | bytes | object | other
  deriving DecidableEq

def Kind.integer : Kind → Bool
  | .signed | .unsigned | .uint8 => true
  | _ => false

def Kind.numeric : Kind → Bool
  | .signed | .unsigned | .uint8 | .floating | .complex => true
  | _ => false

def Kind.coarse : Kind → DType
  | .signed | .unsigned | .uint8 => .integer
  | .boolean => .boolean
  | .floating => .floating
  | .complex => .numericOther
  | _ => .other

inductive Cell where
  | atom (value : Atom)
  | text (value : String)
  | bytes (value : List UInt8)

def Cell.matches (kind : Kind) (cell : Cell) : Prop :=
  match kind with
  | .unicode => ∃ value, cell=.text value
  | .boolean => ∃ value, cell=.atom (.boolean value)
  | .signed | .unsigned | .uint8 => ∃ value, cell=.atom (.integer value)
  | .floating => ∃ value, cell=.atom value ∧
      ((∃ real, value=.real real) ∨ value=.nan ∨ value=.positiveInfinity ∨ value=.negativeInfinity)
  | .complex => ∃ value, cell=.atom value
  | _ => True

structure RawArray where
  kind : Kind
  dtype : String
  shape : List ℕ
  cells : List Cell
  storageGeometry : cells.length=shape.prod
  dtypeContents : ∀ cell ∈ cells, cell.matches kind
  cOrderBytes : Option (List UInt8) := none

inductive NativeValue where
  | scalar (value : Cell)
  | array (value : RawArray)

inductive Cast where
  | float32 | float64 | int64 | boolean | string
  deriving DecidableEq

def Cast.kind : Cast → Kind
  | .float32 | .float64 => .floating
  | .int64 => .signed
  | .boolean => .boolean
  | .string => .unicode

abbrev CastResult (target : Cast) (original : RawArray) :=
  {image : RawArray // image.kind=target.kind ∧ image.shape=original.shape}

structure TableImage where
  rows : ℕ
  columns : ℕ
  jsonTable : Value

structure Operations (World : Type) where
  snapshot : ERPTemporalSnapshot.Operations World
  openNPZNoPickle : ℕ → EStateM Fault World ℕ
  closeNPZ : ℕ → EStateM Fault World Unit
  archiveFiles : ℕ → EStateM Fault World (List String)
  readMember : ℕ → String → EStateM Fault World RawArray
  asArray : (original : RawArray) → EStateM Fault World {image : RawArray // image=original}
  item : RawArray → EStateM Fault World Cell
  pythonString : NativeValue → EStateM Fault World String
  pythonTruth : NativeValue → EStateM Fault World Bool
  cast : (target : Cast) → (original : RawArray) → EStateM Fault World (CastResult target original)
  copyAndReadOnly : (target : Cast) → (original : RawArray) → EStateM Fault World (CastResult target original)
  repeatEmpty : ℕ → EStateM Fault World RawArray
  fillMinusOne : ℕ → EStateM Fault World RawArray
  iterateFirstAxis : RawArray → EStateM Fault World (List NativeValue)
  decodeUTF8Bytes : RawArray → EStateM Fault World String
  readTableJSON : String → EStateM Fault World TableImage
  strip : String → String

def requiredMembers : List String :=
  ["schema","name","X","subject_ids","channel_names","channel_positions_m","channel_mask",
   "preprocessing_json","metadata_json","provenance_json","identity_json","lineage_json","has_labels","y",
   "event_schema","event_ids","event_group_ids","event_subject_ids","event_stimulus_ids","event_onset_samples",
   "event_onset_times_s","event_evidence_available_times_s","event_evidence_indices","event_statuses",
   "event_status_details","event_dataset_ids","event_session_ids","event_run_ids","event_selection_ids",
   "event_complete","event_online_causal","event_timing_source"]

def candidateMembers : List String :=
  ["event_candidate_ids","event_target_candidate_ids","event_repetition_indices"]

def qualityMembers : List String :=
  ["qc_feature_schema","qc_relative_ptp","qc_channel_std_v","qc_epoch_scale_v","qc_observed_mask"]

def readArray {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World RawArray := do
  let raw ← small (p.readMember archive name)
  let array ← small (p.asArray raw)
  pure array.1

def readItem {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World Cell := do
  small (p.item (← readArray p archive name))

def readString {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World String := do
  small (p.pythonString (.scalar (← readItem p archive name)))

def readBool {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World Bool := do
  small (p.pythonTruth (.scalar (← readItem p archive name)))

def readJSON {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World Value := do
  small (p.snapshot.decodeJSON (← readString p archive name))

def readCast {World : Type} (p : Operations World) (archive : ℕ) (name : String)
    (kind : Cast) : TraceM World {image : RawArray // image.kind=kind.kind} := do
  let raw ← small (p.readMember archive name)
  let array ← small (p.cast kind raw)
  pure ⟨array.1,array.2.1⟩

def arrayLength (array : RawArray) : Check ℕ :=
  match array.shape with
  | [] => .error (.fieldFailure "len() of unsized ndarray")
  | first :: _ => .ok first

def checkIntegerMembers {World : Type} (p : Operations World) (archive : ℕ) :
    List String → TraceM World Unit
  | [] => pure ()
  | name :: tail => do
    let array ← readArray p archive name
    checked (ensure (array.kind.integer=true) ("cache integer field " ++ name))
    checkIntegerMembers p archive tail

def checkBooleanMembers {World : Type} (p : Operations World) (archive : ℕ) :
    List String → TraceM World Unit
  | [] => pure ()
  | name :: tail => do
    let array ← readArray p archive name
    checked (ensure (array.kind=.boolean) ("cache strict Boolean field " ++ name))
    checkBooleanMembers p archive tail

def checkFloatingMembers {World : Type} (p : Operations World) (archive : ℕ) :
    List String → TraceM World Unit
  | [] => pure ()
  | name :: tail => do
    let array ← readArray p archive name
    checked (ensure (array.kind=.floating) ("cache floating field " ++ name))
    checkFloatingMembers p archive tail

def metadataText {World : Type} (p : Operations World) (image : RawArray) : TraceM World String := do
  if image.kind=.uint8 ∧ image.shape.length=1 then small (p.decodeUTF8Bytes image)
  else if image.shape.length=0 ∧ image.kind=.unicode then do
    small (p.pythonString (.scalar (← small (p.item image))))
  else checked (.error (.sourceReject "metadata_json must be UTF8 bytes or scalar Unicode"))

def readStoredDecodedDigest {World : Type} (p : Operations World) (archive : ℕ)
    (members : List String) : TraceM World (Option String) := do
  if "decoded_dataset_sha256" ∈ members then
    let array ← readArray p archive "decoded_dataset_sha256"
    checked (ensure (array.shape.length=0) "stored decoded digest scalar")
    let value ← small (p.pythonString (.scalar (← small (p.item array))))
    let value := p.strip value
    checked (ensure (LowerHex64 value) "stored decoded digest lowercase SHA256")
    pure (some value)
  else pure none

structure ArrayPrefix where
  members : List String
  retainedX : RawArray
  storedDecodedDigest : Option String

structure Prefix where
  members : List String
  retainedX : RawArray
  preprocessing : Fields
  hasLabels : Bool
  metadataJSON : String
  storedDecodedDigest : Option String

def readArrayPrefix {World : Type} (p : Operations World) (archive : ℕ) : TraceM World ArrayPrefix := do
  let members ← small (p.archiveFiles archive)
  checked (ensure (requiredMembers ⊆ members) "required EpochDataset cache fields")
  let schema ← readString p archive "schema"
  checked (ensure (schema="n2p3net_epoch_dataset/5") "EpochDataset cache schema")
  let eventSchema ← readString p archive "event_schema"
  checked (ensure (eventSchema="n2p3net_scheduled_events/2") "cache event schema")
  checked (ensure (candidateMembers ⊆ members) "current candidate cache fields")
  checked (ensure (qualityMembers ⊆ members) "current quality cache fields")
  let hasLabels ← readBool p archive "has_labels"
  let integerFields := ["event_stimulus_ids","event_onset_samples","event_evidence_indices"] ++
    (if "event_repetition_indices" ∈ members then ["event_repetition_indices"] else []) ++
    (if hasLabels then ["y"] else [])
  checkIntegerMembers p archive integerFields
  checkBooleanMembers p archive ["event_complete","event_online_causal","has_labels"]
  let storedDecodedDigest ← readStoredDecodedDigest p archive members
  let X ← readArray p archive "X"
  checked (ensure (X.kind=.floating) "cache X floating dtype")
  checkFloatingMembers p archive ["channel_positions_m"]
  checkBooleanMembers p archive ["channel_mask"]
  if "trial_channel_mask" ∈ members then checkBooleanMembers p archive ["trial_channel_mask"]
  let schema ← readString p archive "qc_feature_schema"
  checked (ensure (schema="n2p3net_epoch_qc_features/1") "cache QC schema")
  checkFloatingMembers p archive ["qc_relative_ptp","qc_channel_std_v","qc_epoch_scale_v"]
  checkBooleanMembers p archive ["qc_observed_mask"]
  pure ⟨members,X,storedDecodedDigest⟩

def readPrefix {World : Type} (p : Operations World) (archive : ℕ) : TraceM World Prefix := do
  let arrays ← readArrayPrefix p archive
  let preprocessing ← checked (objectAt (← readJSON p archive "preprocessing_json") "PreprocessingSpec kwargs")
  let preprocessing ← checked (readPreprocessingArgs preprocessing)
  let hasLabels ← readBool p archive "has_labels"
  let metadataJSON ← metadataText p (← readArray p archive "metadata_json")
  pure ⟨arrays.members,arrays.retainedX,preprocessing,hasLabels,metadataJSON,arrays.storedDecodedDigest⟩

def firstEventArguments : List (String × Cast) :=
  [("event_ids",.string),("event_group_ids",.string),("event_subject_ids",.string),
   ("event_stimulus_ids",.int64),("event_onset_samples",.int64),("event_onset_times_s",.float64),
   ("event_evidence_available_times_s",.float64),("event_evidence_indices",.int64),
   ("event_statuses",.string),("event_status_details",.string),("event_dataset_ids",.string),
   ("event_session_ids",.string),("event_run_ids",.string),("event_selection_ids",.string)]

def lastEventArguments : List (String × Cast) :=
  [("event_candidate_ids",.string),("event_target_candidate_ids",.string),("event_repetition_indices",.int64)]

def eventCast (index : Fin 17) : Cast :=
  if index.val ∈ [3,4,7,16] then .int64
  else if index.val ∈ [5,6] then .float64
  else .string

def readArguments {World : Type} (p : Operations World) (archive : ℕ) :
    (arguments : List (String × Cast)) → TraceM World {arrays : List RawArray // arrays.length=arguments.length}
  | [] => pure ⟨[],rfl⟩
  | (name,kind) :: tail => do
    let array ← readCast p archive name kind
    let rest ← readArguments p archive tail
    pure ⟨array.1 :: rest.1,by simp [rest.2]⟩

structure TimelineKeywords where
  arrays : Fin 17 → RawArray
  complete : Bool
  onlineCausal : Bool
  timingSource : String

def readTimelineKeywords {World : Type} (p : Operations World) (archive : ℕ) : TraceM World TimelineKeywords := do
  let first ← readArguments p archive firstEventArguments
  let complete ← readBool p archive "event_complete"
  let online ← readBool p archive "event_online_causal"
  let timing ← readString p archive "event_timing_source"
  let last ← readArguments p archive lastEventArguments
  let images := first.1 ++ last.1
  have length : images.length=17 := by
    simp [images,first.2,last.2,firstEventArguments,lastEventArguments]
  let arrays := fun i : Fin 17 => images.get ⟨i.val,by simpa only [length] using i.isLt⟩
  pure ⟨arrays,complete,online,timing⟩

def arrayMap {World : Type} {α β : Type} (action : α → TraceM World β) :
    (values : List α) → TraceM World {images : List β // images.length=values.length}
  | [] => pure ⟨[],rfl⟩
  | value :: tail => do
    let image ← action value
    let rest ← arrayMap action tail
    pure ⟨image :: rest.1,by simp [rest.2]⟩

structure Timeline where
  count : ℕ
  arrays : Fin 17 → RawArray
  ranks : ∀ i, (arrays i).shape.length=1
  commonLength : ∀ i, (arrays i).cells.length=count
  kinds : ∀ i, (arrays i).kind=(eventCast i).kind
  complete : Bool
  onlineCausal : Bool
  timingSource : String

def copyFamily {World : Type} (p : Operations World) : (n : ℕ) → (images : Fin n → RawArray) →
    (casts : Fin n → Cast) → TraceM World ((i : Fin n) → CastResult (casts i) (images i))
  | 0,_,_ => pure (fun i => Fin.elim0 i)
  | n+1,images,casts => do
    let first ← small (p.copyAndReadOnly (casts 0) (images 0))
    let rest ← copyFamily p n (fun i => images i.succ) (fun i => casts i.succ)
    pure (fun i => Fin.cases first (fun i => rest i) i)

def constructTimeline {World : Type} (p : Operations World) (input : TimelineKeywords) : TraceM World Timeline := do
  let first ← small (p.asArray (input.arrays 0))
  let count ← checked (arrayLength first.1)
  let _ ← small (p.repeatEmpty count)
  let _ ← small (p.repeatEmpty count)
  let _ ← small (p.fillMinusOne count)
  let _ ← arrayMap (fun i => do
    let image ← small (p.asArray (input.arrays i))
    pure image.1) (List.finRange 17)
  if rank : ∀ i, (input.arrays i).shape.length=1 then
    checked (ensure (∀ i ∈ ([3,4,7,16] : List (Fin 17)), (input.arrays i).kind.integer=true)
    "scheduled integer arrays forbid Boolean")
    checked (ensure (∀ i ∈ ([5,6] : List (Fin 17)), (input.arrays i).kind.numeric=true)
    "scheduled numeric time arrays")
    let converted ← copyFamily p 17 input.arrays eventCast
    let arrays := fun i => (converted i).1
    have ranks : ∀ i, (arrays i).shape.length=1 := fun i => by
      simpa only [arrays,(converted i).2.2] using rank i
    let complete ← small (p.pythonTruth (.scalar (.atom (.boolean input.complete))))
    let online ← small (p.pythonTruth (.scalar (.atom (.boolean input.onlineCausal))))
    let timing ← small (p.pythonString (.scalar (.text input.timingSource)))
    let _ ← arrayMap (fun i => checked (arrayLength (arrays i))) (List.finRange 17)
    if aligned : ∀ i, (arrays i).cells.length=(arrays 0).cells.length then
      checked (ensure (p.strip timing ≠ "") "nonblank scheduled timing source")
      pure ⟨(arrays 0).cells.length,arrays,ranks,aligned,(fun i => (converted i).2.1),complete,online,timing⟩
    else checked (.error (.sourceReject "scheduled fields equal converted length"))
  else checked (.error (.sourceReject "every scheduled field one dimensional"))

theorem boolean_array_is_not_integer_dtype : Kind.integer .boolean=false := rfl

theorem unsigned_array_is_integer_dtype : Kind.integer .unsigned=true := rfl

theorem complex_array_is_not_floating_dtype : Kind.complex ≠ Kind.floating := by decide

theorem different_native_byte_images_are_not_the_same_array (a b : RawArray)
    (different : a.cOrderBytes ≠ b.cOrderBytes) : a ≠ b := by
  intro equal
  exact different (congrArg RawArray.cOrderBytes equal)

theorem rank_one_length_is_actual_cell_count (array : RawArray) (rank : array.shape.length=1) :
    arrayLength array=.ok array.cells.length := by
  rcases List.length_eq_one_iff.mp rank with ⟨length,shape⟩
  have cells : array.cells.length=length := by simpa [shape] using array.storageGeometry
  simp [arrayLength,shape,cells]

theorem converted_length_set_singleton_iff_all_equal_first (arrays : Fin 17 → RawArray) :
    (Finset.univ.image (fun i => (arrays i).cells.length)).card=1 ↔
      ∀ i, (arrays i).cells.length=(arrays 0).cells.length := by
  constructor
  · intro h
    obtain ⟨length,eq⟩ := Finset.card_eq_one.mp h
    have each : ∀ i, (arrays i).cells.length=length := by
      intro i
      have member : (arrays i).cells.length ∈ Finset.univ.image (fun i => (arrays i).cells.length) :=
        Finset.mem_image.mpr ⟨i,Finset.mem_univ i,rfl⟩
      simpa only [eq,Finset.mem_singleton] using member
    intro i
    exact (each i).trans (each 0).symm
  · intro h
    have eq : Finset.univ.image (fun i => (arrays i).cells.length) = {(arrays 0).cells.length} := by
      ext length
      simp [h,eq_comm]
    simp [eq]

end
end N2P3.ERPTemporalNPZ
