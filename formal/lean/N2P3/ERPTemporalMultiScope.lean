import N2P3.ERPTemporalCLI
import N2P3.PersonalERPInput

/-!
# Stateful source-temporal computation across actual dataset references

Source: temporal_transfer.py:146-225. A request contains a dataset REFERENCE
and a requested-row object. The heap stores the decoded Data/Execution/Input
fields; repeated references share the materialized identity cache. EStateM
retains writes even when a later operation fails. No dataset.validate,
load_epoch_dataset, fit_source or reader-success callback is supplied.

The input checker below is the explicit imported Input/Data/Execution program.
Its source-order corrections and fresh dependency identity must be accepted
before this connection is accepted. This module is new P12 work, not a claim
that the complete CLI, cache decoder or all numerical/IO refinements passed.

The numerical interface exposes only individual array/scalar operations.
Results can be nonfinite and operations can fail. No operation can return an
authority, receipt, model, fitted artifact, admitted scope or complete fit.
The existing exact-real C-H components remain under their own identities.
-/

namespace N2P3.ERPTemporalMultiScope

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPExecution
open N2P3.ERPTemporalCLI

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Dataset where
  epochs : ℕ
  channels : ℕ
  times : ℕ
  events : ℕ
  context : PersonalERPInput.Context epochs channels times events
  labelDtype : String
  nativeFloat64Dtype : String

structure Request where
  reference : ℕ
  requested : RowObject

inductive FloatAtom where
  | finite (value : ℝ)
  | nan | positiveInfinity | negativeInfinity

def FloatAtom.toAtom : FloatAtom → Atom
  | .finite value => .real value
  | .nan => .nan
  | .positiveInfinity => .positiveInfinity
  | .negativeInfinity => .negativeInfinity

def FloatAtom.isFinite : FloatAtom → Bool
  | .finite _ => true
  | _ => false

structure MatrixArray (rows columns : ℕ) where
  dtype : String
  values : Fin rows → Fin columns → FloatAtom

structure BatchArray (rows channels times : ℕ) where
  dtype : DType
  values : Fin rows → Fin channels → Fin times → Atom

structure FloatBatchArray (rows channels times : ℕ) where
  dtype : String
  values : Fin rows → Fin channels → Fin times → FloatAtom

structure PackedSquare where
  times : ℕ
  array : MatrixArray times times

structure EigenArrays (times : ℕ) where
  values : Fin times → FloatAtom
  vectors : MatrixArray times times

structure BasisArray where
  times : ℕ
  rank : ℕ
  array : MatrixArray times rank

structure HashImage where
  dtype : String
  shape : List ℕ
  values : List Atom

structure HashOperations where
  HashState : Type
  asContiguous : HashImage → Check HashImage
  dtypeShapeHeader : String → List ℕ → Check (List UInt8)
  startHash : List UInt8 → Check HashState
  toBytes : HashImage → Check (List UInt8)
  update : HashState → List UInt8 → Check HashState
  finish : HashState → Check String

def numericHash (p : HashOperations) (image : HashImage) : Check String := do
  let image ← p.asContiguous image
  let header ← p.dtypeShapeHeader image.dtype image.shape
  let digest ← p.startHash header
  let bytes ← p.toBytes image
  let digest ← p.update digest bytes
  p.finish digest

structure Numerics where
  castFloat64 : (N C T : ℕ) → BatchArray N C T → Check (FloatBatchArray N C T)
  classMean : (N C T : ℕ) → FloatBatchArray N C T → (Fin N → Bool) → Check (MatrixArray C T)
  subtract : (C T : ℕ) → MatrixArray C T → MatrixArray C T → Check (MatrixArray C T)
  gram : (C T : ℕ) → MatrixArray C T → Check (MatrixArray T T)
  trace : (T : ℕ) → MatrixArray T T → Check FloatAtom
  divide : (T : ℕ) → MatrixArray T T → FloatAtom → Check (MatrixArray T T)
  meanMatrices : (first : PackedSquare) → List PackedSquare → Check (MatrixArray first.times first.times)
  eigh : (T : ℕ) → MatrixArray T T → Check (EigenArrays T)
  multiply : FloatAtom → FloatAtom → Check FloatAtom
  lessEqual : FloatAtom → FloatAtom → Check Bool
  lessThan : FloatAtom → FloatAtom → Check Bool
  absoluteColumn : (T : ℕ) → (Fin T → FloatAtom) → Check (Fin T → FloatAtom)
  argmax : (T : ℕ) → (Fin T → FloatAtom) → Check (Fin T)
  negateColumn : (T : ℕ) → (Fin T → FloatAtom) → Check (Fin T → FloatAtom)

structure Operations where
  input : PersonalERPInput.Primitives
  rows : ArrayCoercion
  broadcastCopy : (N C : ℕ) → (Fin C → Bool) → Check (Fin N → Fin C → Bool)
  intersectTrial : (N C : ℕ) → (Fin N → Fin C → Bool) →
    (Fin N → Fin C → Bool) → Check (Fin N → Fin C → Bool)
  emptyInverse : (N E : ℕ) → Check (Fin N → Option (Fin E))
  numeric : Numerics
  hash : HashOperations
  temporal : TemporalPrimitives

structure FitHeader where
  preprocessing : Fields
  times : ℕ
  rank : ℕ

structure Accumulator where
  header : Option FitHeader := none
  participants : List String := []
  matrices : List PackedSquare := []
  receipts : List Fields := []
  studies : List String := []

structure State where
  heap : ℕ → Option Dataset
  accumulator : Accumulator := {}

abbrev SourceM := EStateM LedgerFault State

def attempt {α : Type} (operation : Check α) : SourceM α := fun state =>
  match operation with
  | .ok value => .ok value state
  | .error fault => .error fault state

def datasetAt (reference : ℕ) : SourceM Dataset := fun state =>
  match state.heap reference with
  | some dataset => .ok dataset state
  | none => .error (.fieldFailure "dataset reference at source validate") state

def storeDataset (reference : ℕ) (dataset : Dataset) : SourceM Unit :=
  fun state => .ok () {state with heap := fun key =>
    if key = reference then (some dataset) else (state.heap key)}

def readRank (raw : Value) (times : ℕ) : Check ℕ := do
  match raw with
  | .integer rank =>
    ensure (1 ≤ rank ∧ rank ≤ (times : ℤ)) "source temporal positive integer rank"
    pure rank.toNat
  | _ => .error (.sourceReject "source temporal positive integer rank")

def sharePreprocessing (rawRank : Value) (dataset : Dataset) : SourceM Unit := do
  let args ← attempt (readPreprocessingArgs dataset.context.preprocessingArgs)
  let state ← get
  match state.accumulator.header with
  | none =>
    let rank ← attempt (readRank rawRank dataset.times)
    modify fun state => {state with accumulator := {state.accumulator with
      header := some ⟨args,dataset.times,rank⟩}}
  | some header =>
    attempt (ensure (PersonalAdmission.PyEq (.object args) (.object header.preprocessing))
      "source temporal shared full preprocessing")

def evidenceEvents (dataset : Dataset) : List (Fin dataset.events) :=
  (List.finRange dataset.events).filter (fun event =>
    @decide (0 ≤ dataset.context.data.timeline.evidence event) (Int.decLe _ _))

def evidenceGate (dataset : Dataset) : Prop :=
  ((evidenceEvents dataset).map dataset.context.data.timeline.evidence).mergeSort
    ERPTemporalProducer.integerLE = (List.range dataset.epochs).map Int.ofNat

structure Frame where
  reference : ℕ
  dataset : Dataset
  identity : List PersonalERPInput.IdentityRecord
  rows : SelectedRows dataset.epochs
  observed : Fin dataset.epochs → Fin dataset.channels → Bool
  inverse : Fin dataset.epochs → Option (Fin dataset.events)

/-- This is the source order within ONE iteration. A global caller must run
the participant body before advancing to the next scope. It must not validate
all scopes first and thereby move a later error ahead of earlier numerical work.
Identity cache writes are retained before event/participant operations fail. -/
def scopePrefix (p : Operations) (rawRank : Value) (request : Request) : SourceM Frame := do
  let dataset ← datasetAt request.reference
  attempt (PersonalERPInput.checkDataset p.input dataset.context true)
  sharePreprocessing rawRank dataset
  let image ← attempt (p.rows request.requested)
  let rows ← attempt (checkProducerRows dataset.epochs image)
  let materialized ← attempt (PersonalERPInput.materializeIdentity p.input dataset.context)
  let dataset := {dataset with context := materialized.2}
  storeDataset request.reference dataset
  let observed ← attempt (p.broadcastCopy dataset.epochs dataset.channels dataset.context.data.static)
  let observed ← match dataset.context.data.trial with
    | none => pure observed
    | some trial => attempt (p.intersectTrial dataset.epochs dataset.channels observed trial)
  attempt (ensure (evidenceGate dataset) "source temporal physical event per input row")
  let initial ← attempt (p.emptyInverse dataset.epochs dataset.events)
  let inverse := ERPTemporalProducer.writeInverse dataset.context.data.timeline.evidence
    (evidenceEvents dataset) initial
  pure ⟨request.reference,dataset,materialized.1,rows,observed,inverse⟩

def selectedRows (frame : Frame) (subject : String) : List (Fin frame.dataset.epochs) :=
  frame.rows.indices.filter (fun row => frame.dataset.context.data.subjects row = subject)

def participantSubjects (frame : Frame) : List String :=
  ((frame.rows.indices.map frame.dataset.context.data.subjects).dedup).mergeSort (· ≤ ·)

def commonChannels (frame : Frame) (rows : List (Fin frame.dataset.epochs)) :
    List (Fin frame.dataset.channels) :=
  (List.finRange frame.dataset.channels).filter (fun channel => rows.all (fun row => frame.observed row channel))

def labelCoordinates (frame : Frame) : Check (Fin frame.dataset.epochs → ℤ) :=
  match frame.dataset.context.data.labelValues with
  | some labels => .ok labels
  | none => .error (.fieldFailure "dataset.y[selected]")

def classGate {N : ℕ} (labels : Fin N → ℤ) (rows : List (Fin N)) : Prop :=
  (∀ row ∈ rows, labels row = 0 ∨ labels row = 1) ∧
    2 ≤ (rows.filter (fun row => labels row = 0)).length ∧
    2 ≤ (rows.filter (fun row => labels row = 1)).length

def gatheredX (frame : Frame) (rows : List (Fin frame.dataset.epochs))
    (channels : List (Fin frame.dataset.channels)) : BatchArray rows.length channels.length frame.dataset.times :=
  ⟨frame.dataset.context.data.x.dtype,
    fun i c t => frame.dataset.context.data.values (rows.get i) (channels.get c) t⟩

def floatValuesImage {N C T : ℕ} (batch : FloatBatchArray N C T) : HashImage :=
  ⟨batch.dtype,[N,C,T],(List.ofFn fun i =>
    (List.ofFn fun c => (List.ofFn (batch.values i c)).map FloatAtom.toAtom).flatten).flatten⟩

def labelsImage (frame : Frame) (labels : Fin frame.dataset.epochs → ℤ)
    (rows : List (Fin frame.dataset.epochs)) : HashImage :=
  ⟨frame.dataset.labelDtype,[rows.length],rows.map (fun row => .integer (labels row))⟩

def observedImage (frame : Frame) (rows : List (Fin frame.dataset.epochs)) : HashImage :=
  ⟨"|b1",[rows.length,frame.dataset.channels],rows.flatMap (fun row =>
    (List.ofFn (frame.observed row)).map Atom.boolean)⟩

def originalEventIds (frame : Frame) : List (Fin frame.dataset.epochs) → Check (List String)
  | [] => .ok []
  | row :: tail => do
    let event ← match frame.inverse row with
      | some event => .ok event
      | none => .error (.fieldFailure "event_ids[inverse[selected]]")
    let rest ← originalEventIds frame tail
    pure (frame.dataset.context.data.timeline.eventIds event :: rest)

/-- The accessor revalidates stored fields and returns its six-key canonical
dictionary. Extra attached fields are dropped; X is not rehashed here. -/
def storedParent (frame : Frame) : Check Value := do
  match frame.dataset.context.attestation with
  | none => pure .null
  | some raw =>
    let fields ← objectAt raw "loaded cache attestation mapping"
    checkVerifiedCacheFields fields
    pure (.object (Fields.ofList [
      ("schema",.text "n2p3_verified_epoch_cache_load/1"),
      ("cache_attestation_schema",fieldDefault fields "cache_attestation_schema"),
      ("full_contract_validated",.boolean true),("sha256",fieldDefault fields "sha256"),
      ("byte_size",fieldDefault fields "byte_size"),
      ("decoded_dataset_sha256",fieldDefault fields "decoded_dataset_sha256")]))

def receipt (p : Operations) (frame : Frame) (identity : PersonalERPInput.IdentityRecord)
    (key : String) (labels : Fin frame.dataset.epochs → ℤ)
    (rows : List (Fin frame.dataset.epochs)) (channels : List (Fin frame.dataset.channels))
    (batch : FloatBatchArray rows.length channels.length frame.dataset.times) : Check Fields := do
  let events ← originalEventIds frame rows
  let names ← channels.mapM (fun channel => match frame.dataset.context.data.channelNames[channel.val]? with
    | some name => .ok name
    | none => .error (.fieldFailure "channel_names[c]"))
  let valuesSha ← numericHash p.hash (floatValuesImage batch)
  let labelsSha ← numericHash p.hash (labelsImage frame labels rows)
  let observedSha ← numericHash p.hash (observedImage frame rows)
  let parent ← storedParent frame
  pure (Fields.ofList [
    ("participant_key",.text key),("origin_keys",.sequence (encodeTexts identity.origins)),
    ("global_keys",.sequence (encodeTexts identity.globals)),
    ("rows",.sequence (encodeNats (rows.map Fin.val))),
    ("event_ids",.sequence (encodeTexts events)),("observed_channels",.sequence (encodeTexts names)),
    ("values_sha256",.text valuesSha),("labels_sha256",.text labelsSha),("observed_sha256",.text observedSha),
    ("parent_cache_attestation",parent),("identity",.object (PersonalERPInput.identityRecordFields identity))])

def participantBody (p : Operations) (frame : Frame) (subject : String) : SourceM Unit := do
  let rows := selectedRows frame subject
  let identity ← attempt (PersonalERPInput.recordFor p.input frame.identity subject)
  let key ← attempt (PersonalERPInput.authorityKey identity)
  let state ← get
  attempt (ensure (key ∉ state.accumulator.participants) "source temporal duplicate participant")
  modify fun state => {state with accumulator := {state.accumulator with
    participants := state.accumulator.participants ++ [key]}}
  let labels ← attempt (labelCoordinates frame)
  attempt (ensure (classGate labels rows) "source temporal two observations per binary class")
  let channels := commonChannels frame rows
  attempt (ensure (channels ≠ []) "source temporal participant observed channels")
  let batch ← attempt (p.numeric.castFloat64 rows.length channels.length frame.dataset.times
    (gatheredX frame rows channels))
  attempt (ensure (∀ i c t, (batch.values i c t).isFinite=true) "source temporal finite observations")
  let target ← attempt (p.numeric.classMean rows.length channels.length frame.dataset.times batch
    (fun i => decide (labels (rows.get i) = 1)))
  let nontarget ← attempt (p.numeric.classMean rows.length channels.length frame.dataset.times batch
    (fun i => decide (labels (rows.get i) = 0)))
  let contrast ← attempt (p.numeric.subtract channels.length frame.dataset.times target nontarget)
  let gram ← attempt (p.numeric.gram channels.length frame.dataset.times contrast)
  let energy ← attempt (p.numeric.trace frame.dataset.times gram)
  attempt (ensure (energy.isFinite=true) "source temporal identifiable energy")
  let nonpositive ← attempt (p.numeric.lessEqual energy (.finite 0))
  attempt (ensure (nonpositive=false) "source temporal identifiable energy")
  let normalized ← attempt (p.numeric.divide frame.dataset.times gram energy)
  modify fun state => {state with accumulator := {state.accumulator with
    matrices := state.accumulator.matrices ++ [⟨frame.dataset.times,normalized⟩]}}
  let item ← attempt (receipt p frame identity key labels rows channels batch)
  modify fun state => {state with accumulator := {state.accumulator with
    receipts := state.accumulator.receipts ++ [item]}}
  let selected ← attempt (subsetIdentity p.input frame.identity [subject])
  let studies ← attempt (PersonalERPInput.allIdentityStudies p.input selected)
  modify fun state => {state with accumulator := {state.accumulator with
    studies := (state.accumulator.studies ++ studies).dedup}}

def collectSubjects (p : Operations) (frame : Frame) : List String → SourceM Unit
  | [] => pure ()
  | subject :: tail => do
    participantBody p frame subject
    collectSubjects p frame tail

def collectScopes (p : Operations) (rank : Value) : List Request → SourceM Unit
  | [] => pure ()
  | request :: tail => do
    let frame ← scopePrefix p rank request
    collectSubjects p frame (participantSubjects frame)
    collectScopes p rank tail

def flipColumns (p : Numerics) {T R : ℕ} (basis : MatrixArray T R) :
    List (Fin R) → Check (MatrixArray T R)
  | [] => .ok basis
  | column :: tail => do
    let current := fun i => basis.values i column
    let magnitude ← p.absoluteColumn T current
    let pivot ← p.argmax T magnitude
    let negative ← p.lessThan (current pivot) (.finite 0)
    let basis ← if negative then do
        let changed ← p.negateColumn T current
        pure {basis with values := fun i c => if c=column then changed i else basis.values i c}
      else pure basis
    flipColumns p basis tail

/-- These bounds express valid negative array indexing, not a new producer
rank gate. The first nonempty scope has already called the exact-int gate. -/
def selectBasis (p : Numerics) (T R : ℕ) (eigen : EigenArrays T) : Check BasisArray := do
  if bounds : 0 < R ∧ R ≤ T then
    let cutoff : Fin T := ⟨T-R,by omega⟩
    let top : Fin T := ⟨T-1,by omega⟩
    let threshold ← p.multiply (eigen.values top) (.finite (1/10000000000))
    let tooSmall ← p.lessEqual (eigen.values cutoff) threshold
    ensure (tooSmall=false) "source temporal identifiable rank"
    let raw : MatrixArray T R := ⟨eigen.vectors.dtype,fun i c =>
      eigen.vectors.values i ⟨T-1-c.val,by have := c.isLt; omega⟩⟩
    let signed ← flipColumns p raw (List.finRange R)
    pure ⟨T,R,signed⟩
  else .error (.fieldFailure "source temporal eigenvalue negative indexing")

/-- Conversion rejects IEEE nonfinite cells at the finite canonical-JSON
boundary used by semantic_sha256. It does not assert the eigensolver is exact
or that a successful contrast must produce finite eigenvectors. -/
def basisValue (basis : BasisArray) : Check Value := do
  let rows ← (List.finRange basis.times).mapM (fun i => do
    let row ← PersonalERPInput.atomJSONList
      ((List.ofFn (basis.array.values i)).map FloatAtom.toAtom)
    pure (.sequence (valuesOfList row)))
  pure (.sequence (valuesOfList rows))

def computedMetadata (header : FitHeader) (accumulator : Accumulator)
    (basis : BasisArray) : Check Fields := do
  let matrix ← basisValue basis
  pure (Fields.ofList [
    ("schema",.text "n2p3_temporal_basis/1"),("fit_role",.text "source_supervised"),
    ("preprocessing",.object (preprocessingRecord header.preprocessing)),
    ("rank",.integer basis.rank),("basis",matrix),
    ("fit_receipts",.sequence (valuesOfList (accumulator.receipts.map Value.object))),
    ("source_studies",.sequence (encodeTexts (accumulator.studies.mergeSort (· ≤ ·)))),
    ("risk",.text "equal_participant_normalized_ERP_contrast")])

def finishSource (p : Operations) : SourceM TemporalModel := do
  let state ← get
  match state.accumulator.header,state.accumulator.matrices with
  | some header,first :: rest =>
    let mean ← attempt (p.numeric.meanMatrices first rest)
    let eigen ← attempt (p.numeric.eigh first.times mean)
    let basis ← attempt (selectBasis p.numeric first.times header.rank eigen)
    let metadata ← attempt (computedMetadata header state.accumulator basis)
    let record := Fields.cons "digest" (.text (p.temporal.metadataHash metadata)) metadata
    attempt (readTemporal p.temporal (.object record))
  | _,_ => throw (.sourceReject "source temporal nonempty scopes")

/-- Only fit-local accumulators reset; the shared dataset reference heap and
materialized identity caches persist across each actual replay fit. -/
def fitSource (p : Operations) (scopes : List Request) (rank : Value) : SourceM TemporalModel := do
  modify fun state => {state with accumulator := {}}
  collectScopes p rank scopes
  finishSource p

/-- The list is reiterable, as the CLI builds it. This theorem/program does
not turn an arbitrary consumed Python generator into a replayable scope. -/
def verifySource (p : Operations) (saved : TemporalModel) (scopes : List Request) : SourceM Unit := do
  attempt (ensure (fieldDefault saved.record "fit_role"=.text "source_supervised")
    "fixed temporal control has no source fitting to verify")
  let matrix ← attempt (p.temporal.metadataArray (← attempt (required saved.record "basis")))
  let rank ← match matrix.shape[1]? with
    | some width => pure width
    | none => throw (.fieldFailure "source temporal matrix.shape[1]")
  let repeated ← fitSource p scopes (.integer rank)
  attempt (ensure (ERPTemporalProducer.replayRecordEqual repeated.record saved.record)
    "temporal source replay differs")

theorem Boolean_rank_is_not_an_integer (rank : Bool) (times : ℕ) :
    readRank (.boolean rank) times = .error (.sourceReject "source temporal positive integer rank") := rfl

theorem float_rank_is_not_an_integer (rank : ℝ) (times : ℕ) :
    readRank (.number rank) times = .error (.sourceReject "source temporal positive integer rank") := rfl

theorem gathered_coordinate_retains_original_row_channel (frame : Frame)
    (rows : List (Fin frame.dataset.epochs)) (channels : List (Fin frame.dataset.channels))
    (i : Fin rows.length) (c : Fin channels.length) (t : Fin frame.dataset.times) :
    (gatheredX frame rows channels).values i c t=
      frame.dataset.context.data.values (rows.get i) (channels.get c) t := rfl

theorem observed_digest_keeps_full_original_width (frame : Frame)
    (rows : List (Fin frame.dataset.epochs)) :
    (observedImage frame rows).shape=[rows.length,frame.dataset.channels] := rfl

theorem labels_digest_keeps_original_dtype (frame : Frame)
    (labels : Fin frame.dataset.epochs → ℤ) (rows : List (Fin frame.dataset.epochs)) :
    (labelsImage frame labels rows).dtype=frame.dataset.labelDtype := rfl

theorem common_channel_requires_every_selected_row (frame : Frame)
    (rows : List (Fin frame.dataset.epochs)) (channel : Fin frame.dataset.channels) :
    channel ∈ commonChannels frame rows ↔ ∀ row ∈ rows, frame.observed row channel=true := by
  simp [commonChannels,List.all_eq_true]

theorem selected_participants_are_unique (frame : Frame) :
    (participantSubjects frame).Nodup := by
  simpa only [participantSubjects,List.nodup_mergeSort] using
    List.nodup_dedup (frame.rows.indices.map frame.dataset.context.data.subjects)

theorem hash_contiguous_failure_precedes_header (p : HashOperations) (image : HashImage)
    (fault : LedgerFault) (failed : p.asContiguous image=.error fault) :
    numericHash p image=.error fault := by
  simp [numericHash,failed]

theorem hash_initialization_failure_precedes_bytes (p : HashOperations) (image contiguous : HashImage)
    (header : List UInt8) (fault : LedgerFault)
    (coerced : p.asContiguous image=.ok contiguous)
    (encoded : p.dtypeShapeHeader contiguous.dtype contiguous.shape=.ok header)
    (failed : p.startHash header=.error fault) : numericHash p image=.error fault := by
  simp [numericHash,coerced,encoded,failed]

theorem heap_write_survives_later_failure (reference : ℕ) (dataset : Dataset)
    (fault : LedgerFault) (state : State) :
    ((storeDataset reference dataset >>= fun _ => attempt (.error fault : Check Unit)) state)=
      .error fault {state with heap := fun key =>
        if key=reference then some dataset else state.heap key} := by
  rfl

theorem materialize_identity_preserves_sample_fields (p : PersonalERPInput.Primitives)
    {N C T M : ℕ} (context after : PersonalERPInput.Context N C T M)
    (rows : List PersonalERPInput.IdentityRecord)
    (success : PersonalERPInput.materializeIdentity p context=.ok (rows,after)) :
    after.data=context.data := by
  cases cache : context.identityCache with
  | some identities =>
    simp only [PersonalERPInput.materializeIdentity,cache] at success
    cases checked : ensure
      ((identities.map PersonalERPInput.IdentityRecord.localSubject).toFinset=
        (PersonalERPInput.subjects context).toFinset)
      "materialized identity exact original subject coverage" <;>
      simp_all
  | none =>
    simp only [PersonalERPInput.materializeIdentity,cache] at success
    cases checked : checkTimeline p.execution.strip context.data.timeline N <;>
      simp_all
    case ok checkedValue =>
      cases paired : PersonalERPInput.sourceRowPairs p context (List.finRange N) <;>
        simp_all
      case ok pairs =>
        cases built : PersonalERPInput.constructSourceRows pairs <;> simp_all
        exact (congrArg PersonalERPInput.Context.data success.2).symm

theorem store_preserves_other_dataset_references (reference other : ℕ) (dataset : Dataset)
    (state : State) (different : other ≠ reference) :
    (match storeDataset reference dataset state with
      | .ok _ after => after.heap other
      | .error _ after => after.heap other) = state.heap other := by
  simp [storeDataset,different]

theorem repeated_reference_reads_written_identity_context (reference : ℕ) (dataset : Dataset)
    (state : State) :
    ((storeDataset reference dataset >>= fun _ => datasetAt reference) state) =
      .ok dataset {state with heap := fun key =>
        if key = reference then some dataset else state.heap key} := by
  simp [storeDataset,datasetAt,EStateM.bind]

end
end N2P3.ERPTemporalMultiScope
