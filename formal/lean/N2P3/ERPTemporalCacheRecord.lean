import N2P3.ERPTemporalRawCacheDecode
import N2P3.ERPTemporalArrayJSON
import N2P3.PersonalERPArrayIngress

/-!
# Computed EpochDataset.record(validate=False), without full validation

Source: epochs.py:481-520,706-713; events.py:205-241,488-523.
The raw Dataset can still have wrong X/subject/mask shapes. The existing-table
materialize branch performs actual astype/tolist/set coverage first. Only the
timeline's OWN earlier constructor supplied M views. fingerprint calls the
explicit timeline validator with expected epochs equal to available count,
matching validate(n_epochs=None), not X.shape[0].

JSON dump/load, standard per-array operations, scalar float conversion and
hash initialization/update/finalization are local leaves. No leaf returns an
expected record, decoded-dataset hash, admitted Dataset, basis or fitted model.
Native scalar conversion retains negative zero; Real coordinates alone do
not supply the bytes used by the later fingerprint.
-/

namespace N2P3.ERPTemporalCacheRecord

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ
open N2P3.ERPTemporalJSON N2P3.ERPTemporalArrayViews

noncomputable section

attribute [local instance] Classical.propDecidable

structure HashOperations (World : Type) where
  HashState : Type
  startHash : List UInt8 → EStateM Fault World HashState
  update : HashState → List UInt8 → EStateM Fault World HashState
  finish : HashState → EStateM Fault World String
  asciiArrayHeader : JValue → EStateM Fault World (List UInt8)
  fingerprintUTF8 : JValue → EStateM Fault World (List UInt8)
  strictSemanticASCII : JValue → EStateM Fault World (List UInt8)
  asContiguous : (original : RawArray) → EStateM Fault World
    {image : RawArray // image.kind=original.kind ∧ image.dtype=original.dtype ∧
      image.cells=original.cells ∧ image.cOrderBytes=original.cOrderBytes ∧
      image.shape=(if original.shape=[] then [1] else original.shape)}
  memoryBytes : (array : RawArray) → EStateM Fault World
    {bytes : List UInt8 // ∀ actual, array.cOrderBytes=some actual → bytes=actual}

/-- Logical atom correspondence of Python float(np.float64(...)). Native
negative zero remains a distinct JSON value while both zeros have Real 0.
The separate native byte image is still needed for sign/byte refinement. -/
def scalarFloatCell : JValue → Option Cell
  | .floating (.finite value) => some (.atom (.real value))
  | .floating .nan => some (.atom .nan)
  | .floating .positiveInfinity => some (.atom .positiveInfinity)
  | .floating .negativeInfinity => some (.atom .negativeInfinity)
  | .negativeZero => some (.atom (.real 0))
  | _ => none

structure Operations (World : Type) where
  decode : ERPTemporalRawCacheDecode.Operations World
  timeline : PersonalERPArrayIngress.Operations
  timelineInput : timeline.input.data.input.original=decode.input
  hash : HashOperations World
  astype : (target : Cast) → (original : RawArray) → EStateM Fault World (CastResult target original)
  uniqueStrings : {array : RawArray // array.kind=.unicode} → EStateM Fault World
    {array : RawArray // array.kind=.unicode ∧ array.shape.length=1}
  meanBoolean : {array : RawArray // array.kind=.boolean} → EStateM Fault World ERPTemporalMultiScope.FloatAtom
  floatAt : (array : RawArray) → (index : Fin array.cells.length) → EStateM Fault World
    {value : JValue // scalarFloatCell value=some (array.cells.get index)}
  dumpDefaultJSON : JValue → EStateM Fault World String

def fields (items : List (String × JValue)) : JValue := .object (JFields.ofList items)

def texts (values : List String) : JValue := .sequence (JValues.ofList (values.map JValue.text))

def integers (values : List ℕ) : JValue := .sequence (JValues.ofList (values.map (fun value => .integer value)))

def ofFloat : ERPTemporalMultiScope.FloatAtom → JValue := JValue.floating

def comparisonTag : PersonalERPScalar.Comparison → String
  | .ltZero => "ltZero" | .leZero => "leZero" | .ltOne => "ltOne"
  | .zeroLe => "zeroLe" | .zeroLt => "zeroLt" | .neZero => "neZero"

def atomicOperationTag : PersonalERPScalar.Operation → String
  | .numpyFinite => "numpyFinite"
  | .mathFinite => "mathFinite"
  | .compare comparison => "compare(" ++ comparisonTag comparison ++ ")"
  | .setMembership => "setMembership"

def atomicExceptionTag : PersonalERPScalar.AtomicException → String
  | .typeError => "TypeError"
  | .overflowError => "OverflowError"
  | .other name => "other(" ++ reprStr name ++ ")"
  | .unrefined boundary => "unrefined(" ++ reprStr boundary ++ ")"

/-- Preserve the runtime field/operation/exception distinction in the older
cache fault carrier, without modifying its frozen accepted declaration. -/
def runtimeFault : PersonalERPScalar.Fault → ERPTemporalCache.Fault
  | .inherited fault => .check fault
  | .sourceValueError field => .check (.sourceReject field)
  | .atomic field operation exception => .other ("atomic(" ++ reprStr field ++ "," ++
      atomicOperationTag operation ++ "," ++ atomicExceptionTag exception ++ ")")
  | .unrefined boundary => .other ("unrefined(" ++ reprStr boundary ++ ")")

def runtimeChecked {World α : Type} (result : PersonalERPScalar.Result α) : TraceM World α :=
  small fun world => match result with
    | .ok value => .ok value world
    | .error fault => .error (runtimeFault fault) world

def sourceHash {World : Type} (p : HashOperations World) (bytes : List UInt8) : TraceM World String := do
  let state ← small (p.startHash bytes)
  small (p.finish state)

def hashChunks {World : Type} (p : HashOperations World) :
    List (List UInt8) → p.HashState → TraceM World p.HashState
  | [],state => pure state
  | bytes :: tail,state => do
    let state ← small (p.update state bytes)
    hashChunks p tail state

def arrayHash {World : Type} (p : Operations World) (original : RawArray) : TraceM World String := do
  let array ← small (p.decode.npz.asArray original)
  let array ← small (p.hash.asContiguous array.1)
  let state ← small (p.hash.startHash [])
  let header ← small (p.hash.asciiArrayHeader (fields [
    ("dtype",.text array.1.dtype),("shape",integers array.1.shape)]))
  let state ← small (p.hash.update state header)
  let content ← small (p.hash.memoryBytes array.1)
  let block := 16*1024*1024
  let offsets := (List.range ((content.1.length+block-1)/block)).map (fun index => index*block)
  let chunks := offsets.map (fun offset => (content.1.drop offset).take block)
  let state ← hashChunks p.hash chunks state
  small (p.hash.finish state)

def hasCandidates (strip : String → String) {M : ℕ} (image : TimelineImage M) : Bool :=
  !(List.finRange M).isEmpty && (List.finRange M).all (fun i => strip (image.candidates i) ≠ "")

def hasTargets (strip : String → String) {M : ℕ} (image : TimelineImage M) : Bool :=
  hasCandidates strip image && !(List.finRange M).isEmpty &&
    (List.finRange M).all (fun i => strip (image.targets i) ≠ "")

def hasRepetitions (strip : String → String) {M : ℕ} (image : TimelineImage M) : Bool :=
  hasTargets strip image && !(List.finRange M).isEmpty &&
    (List.finRange M).all (fun i => decide (0 ≤ image.repetitions i))

def vocabulary {M : ℕ} (image : TimelineImage M) (group : String) : List String :=
  ((groupRows image group).map image.candidates).dedup.mergeSort (· ≤ ·)

def fullCandidateChain (strip : String → String) {M : ℕ} (image : TimelineImage M) : Bool :=
  if image.complete && hasRepetitions strip image then
    let groups := ((List.ofFn image.groups).dedup).mergeSort (· ≤ ·)
    let vocabularies := (groups.map (vocabulary image)).dedup
    groups.all (fun group => (vocabulary image group).all (fun candidate =>
      let rows := (groupRows image group).filter (fun row => image.candidates row=candidate)
      (rows.map image.repetitions)=(List.range rows.length).map Int.ofNat)) &&
      decide (vocabularies.length=1) && !((vocabularies.headD []).isEmpty)
  else false

def floatIndex (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17) (i : Fin timeline.count) :
    Fin (timeline.arrays field).cells.length :=
  ⟨i.val,by simpa only [timeline.commonLength field] using i.isLt⟩

def fingerprintRow {World : Type} (p : Operations World) (timeline : ERPTemporalNPZ.Timeline)
    (i : Fin timeline.count) : TraceM World JValue := do
  let image := timelineImage timeline
  let onset ← small (p.floatAt (timeline.arrays 5) (floatIndex timeline 5 i))
  let available ← if finiteNumber (image.availableAt i) then
      (fun value => value.1) <$> small (p.floatAt (timeline.arrays 6) (floatIndex timeline 6 i))
    else pure JValue.null
  pure (.sequence (JValues.ofList [
    .text (image.eventIds i),.text (image.groups i),.text (image.subjects i),
    .integer (image.stimulus i),.integer (image.onsetSamples i),onset.1,available,
    .integer (image.evidence i),.text (image.statuses i),.text (image.details i),
    .text (image.datasets i),.text (image.sessions i),.text (image.runs i),.text (image.selections i),
    .text (image.candidates i),.text (image.targets i),.integer (image.repetitions i)]))

def fingerprintRows {World : Type} (p : Operations World) (timeline : ERPTemporalNPZ.Timeline) :
    List (Fin timeline.count) → TraceM World (List JValue)
  | [] => pure []
  | row :: tail => do
    let row ← fingerprintRow p timeline row
    let tail ← fingerprintRows p timeline tail
    pure (row :: tail)

def timelineFingerprint {World : Type} (p : Operations World) (timeline : ERPTemporalNPZ.Timeline) :
    TraceM World String := do
  -- The no-N branch computes its expected count only AFTER native status
  -- casts, Boolean masking and int64 conversion. Group checks remain in
  -- sorted-group source order. This reuses the explicit raw validator;
  -- it does not inherit the old frozen aggregate's error-priority claim.
  let raw : PersonalERPArrayIngress.RawTimeline := ⟨timeline.arrays,timeline.complete,timeline.onlineCausal⟩
  let _ ← runtimeChecked (PersonalERPArrayIngress.timelineValidateOptional p.timeline raw none)
  let rows ← fingerprintRows p timeline (List.finRange timeline.count)
  let payload := fields [("schema",.text "n2p3net_scheduled_events/2"),
    ("complete",.boolean timeline.complete),("online_causal",.boolean timeline.onlineCausal),
    ("timing_source",.text timeline.timingSource),("events",.sequence (JValues.ofList rows)),
    ("truth",.sequence .nil)]
  sourceHash p.hash (← small (p.hash.fingerprintUTF8 payload))

def identityPayload {World : Type} (p : Operations World) (identities : List PersonalERPInput.IdentityRecord) :
    TraceM World JValue := do
  let payload := PersonalERPInput.identityCanonicalPayload identities
  let digest ← checked (p.decode.input.execution.semanticHash (.object payload))
  pure (fromValue (.object (.cons "digest" (.text digest) payload)))

def checkStoredIdentityCoverage {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) : TraceM World Unit := do
  let subjects ← small (p.decode.npz.asArray dataset.subjects.1)
  let subjects ← small (p.astype .string subjects.1)
  let localSet ← checked (ERPTemporalArrayJSON.pythonSubjectSet
    (ERPTemporalArrayJSON.strings ⟨subjects.1,subjects.2.1⟩))
  checked (ensure ((dataset.identity.map PersonalERPInput.IdentityRecord.localSubject).toFinset=localSet.toFinset)
    "identity table exact dataset subject coverage")

def record {World : Type} (p : Operations World) (dataset : ERPTemporalRawCacheDecode.Dataset) : TraceM World JValue := do
  checkStoredIdentityCoverage p dataset
  let channelMask ← small (p.astype .boolean dataset.staticMask.1)
  let maskJSON := ERPTemporalArrayJSON.booleans ⟨channelMask.1,channelMask.2.1⟩
  let mean ← match dataset.trialMask with
    | none => small (p.meanBoolean dataset.staticMask)
    | some trial => do
      let trial ← small (p.decode.npz.cast .boolean trial.1)
      small (p.meanBoolean ⟨trial.1,trial.2.1⟩)
  let preprocessing ← checked (ERPTemporalRawFields.preprocessingRecord dataset.preprocessing)
  let subjects ← small (p.decode.npz.asArray dataset.subjects.1)
  let unique ← small (p.uniqueStrings ⟨subjects.1,by simpa only [subjects.2] using dataset.subjects.2⟩)
  let subjectCount ← checked (arrayLength unique.1)
  let identity ← identityPayload p dataset.identity
  let lineage ← checked (ERPTemporalLineage.lineagePayload
    (ERPTemporalRawLineage.finitePrimitives (ERPTemporalRawCacheDecode.lineageOperations p.decode)) dataset.lineage)
  let image := timelineImage dataset.timeline
  let available := (List.finRange dataset.timeline.count).filter (fun i => decide (image.statuses i="available"))
  let candidate := hasCandidates p.decode.input.execution.strip image
  let target := hasTargets p.decode.input.execution.strip image
  let repetition := hasRepetitions p.decode.input.execution.strip image
  let full := fullCandidateChain p.decode.input.execution.strip image
  let fingerprint ← timelineFingerprint p dataset.timeline
  let events := fields [("schema",.text "n2p3net_scheduled_events/2"),
    ("n_scheduled",.integer dataset.timeline.count),("n_available",.integer available.length),
    ("complete",.boolean dataset.timeline.complete),("online_causal",.boolean dataset.timeline.onlineCausal),
    ("has_candidate_ids",.boolean candidate),("has_candidate_sets",.boolean target),
    ("has_repetition_structure",.boolean repetition),("supports_full_candidate_chain",.boolean full),
    ("fingerprint",.text fingerprint)]
  let quality := fields [("schema",.text dataset.quality.schema),
    ("relative_ptp",.text "dimensionless_within_epoch_observed_channel_median"),
    ("channel_std_v",.text "within_epoch_standard_deviation_v"),
    ("epoch_scale_v",.text "within_epoch_observed_channel_median_raw_ptp_v"),
    ("observed_mask",.text "effective_channel_availability")]
  pure (fields [("schema",.text "n2p3net_epoch_dataset/5"),("name",.text dataset.name),
    ("shape",integers dataset.X.1.shape),("channel_names",texts dataset.channelNames),
    ("channel_mask",maskJSON),("has_trial_channel_mask",.boolean dataset.trialMask.isSome),
    ("mean_observed_channel_fraction",ofFloat mean),("preprocessing",.object preprocessing),
    ("n_subjects",.integer subjectCount),("identity",identity),("lineage",fromValue (.object lineage)),
    ("events",events),("qc_features",quality),("provenance",dataset.provenance)])

def roundtripRecord {World : Type} (p : Operations World) (dataset : ERPTemporalRawCacheDecode.Dataset) :
    TraceM World JValue := do
  let raw ← record p dataset
  let text ← small (p.dumpDefaultJSON raw)
  small (p.decode.decodeJSON text)

end
end N2P3.ERPTemporalCacheRecord
