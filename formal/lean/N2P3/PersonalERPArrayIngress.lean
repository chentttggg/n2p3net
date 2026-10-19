import N2P3.ERPTemporalNPZ
import N2P3.PersonalERPInputRuntime

/-!
# Unvalidated public ndarray ingress, in literal guard order

The raw arrays below have independent ranks, lengths, dtypes and cells. Only
the ndarray storage/dtype representation is intrinsic; no N/C/M domain or
dataset-success predicate is supplied by an input. Finite views are created
only after their corresponding source shape/dtype tests. In particular label
decoding cannot exclude a float or wrong-length label before the timeline.

Sources: epochs.py:325-471; events.py:283-432; qc_features.py:44-85.
Every primitive is one dtype conversion, ndarray row norm, scalar comparison
or subtraction. There is no complete timeline/dataset validator primitive.
Inherited indexing faults retain the stated non-refinement of Python classes.
The arbitrary raw timeline validator does not invent the common-length guard
that exists in __post_init__ but is absent from validate(). Context export
therefore separately needs actual constructor-derived field correspondence.
-/

namespace N2P3.PersonalERPArrayIngress
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime N2P3.ERPTemporalNPZ

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Operations where
  input : PersonalERPExecutionRuntime.Primitives
  cast : (target : Cast) → (raw : RawArray) → R (CastResult target raw)
  less : Cell → Cell → R Bool
  lessEqual : Cell → Cell → R Bool
  greaterEqual : Cell → Cell → R Bool
  equal : Cell → Cell → R Bool
  subtract : String → Cell → Cell → R Cell
  addInt64 : Cell → Cell → R Cell
  normRow : String → List Cell → R Cell
  finiteAll : RawArray → R Bool
  finiteRows : String → List (List Cell) → R Bool
  int64Cells : Kind → String → (cells : List Cell) → R {values : List ℤ // values.length=cells.length}

def atomCell (a : Atom) : Cell := .atom a
def integerCell (z : ℤ) : Cell := .atom (.integer z)
def zeroCell : Cell := integerCell 0

def cellAtom : Cell → Atom
  | .atom value => value
  | _ => .other

def cellText : Cell → String
  | .text value => value
  | _ => ""

def cellBool : Cell → Bool
  | .atom (.boolean b) => b
  | _ => false

def cellInteger : Cell → ℤ
  | .atom (.integer z) => z
  | _ => 0

def rawShape (raw : RawArray) (shape : List ℕ) (site : String) : R (PLift (raw.shape=shape)) :=
  if h : raw.shape=shape then .ok ⟨h⟩ else .error (.sourceValueError site)

def arrayLength (raw : RawArray) : R ℕ := liftCheck (ERPTemporalNPZ.arrayLength raw)

def nthCell (cells : List α) (index : ℕ) (site : String) : R α :=
  match cells[index]? with
  | some cell => .ok cell
  | none => .error (.inherited (.fieldFailure (site ++ " ndarray index out of bounds")))

/-- One native one-dimensional Boolean-index operation. Its shape fault occurs
at the operation, rather than at an invented common-length ingress gate. -/
def masked (raw : RawArray) (mask : List Bool) (site : String) : R (List Cell) := do
  if raw.shape=[mask.length] then
    pure (((raw.cells.zip mask).filter (fun row => row.2)).map Prod.fst)
  else
    .error (.inherited (.fieldFailure (site ++ " boolean index shape mismatch")))

def nativeRows (raw : RawArray) (rows : List ℕ) (site : String) : R (List Cell) :=
  rows.mapM (fun i => nthCell raw.cells i site)

def vector (cells : List α) (N : ℕ) (length : cells.length=N) : Fin N → α :=
  fun i => cells.get (Fin.cast length.symm i)

theorem vector_decoding (cells : List α) (N : ℕ) (length : cells.length=N) :
    List.ofFn (vector cells N length)=cells := by
  subst N
  change List.ofFn cells.get=cells
  exact List.ofFn_get cells

structure TensorHeader (raw : RawArray) where
  N : ℕ
  C : ℕ
  T : ℕ
  shape : raw.shape=[N,C,T]
  floating : raw.kind=.floating

def tensorHeader (raw : RawArray) (nTimes : ℤ) : R (TensorHeader raw) := do
  match hs : raw.shape with
  | [N,C,T] =>
    if hd : raw.kind=.floating then
      require (0 < N ∧ 0 < C) "positive epoch and channel counts"
      require ((T : ℤ)=nTimes) "dataset sample count equals preprocessing n_times"
      pure ⟨N,C,T,hs,hd⟩
    else .error (.sourceValueError "floating EEG tensor dtype")
  | _ => .error (.sourceValueError "EEG tensor rank three")

structure MaskHeader (N C : ℕ) where
  staticRaw : RawArray
  staticShape : staticRaw.shape=[C]
  staticDType : staticRaw.kind=.boolean
  trialRaw : Option RawArray
  trialShape : ∀ raw, trialRaw=some raw → raw.shape=[N,C]
  trialDType : ∀ raw, trialRaw=some raw → raw.kind=.boolean

def staticAt {N C : ℕ} (masks : MaskHeader N C) (c : Fin C) : Bool :=
  cellBool (masks.staticRaw.cells[c.val]?.getD (.atom .other))

def trialAt {N C : ℕ} (masks : MaskHeader N C) (i : Fin N) (c : Fin C) : Bool :=
  match masks.trialRaw with
  | none => staticAt masks c
  | some raw => cellBool (raw.cells[i.val*C+c.val]?.getD (.atom .other))

def maskHeader (N C : ℕ) (static : RawArray) (trial : Option RawArray) :
    R {masks : MaskHeader N C // masks.staticRaw=static ∧ masks.trialRaw=trial} := do
  if hd : static.kind=.boolean then
    let hs ← rawShape static [C] "static channel mask array shape"
    require (true ∈ static.cells.map cellBool) "at least one static observed channel"
    match trial with
    | none => pure ⟨⟨static,hs.down,hd,none,by simp,by simp⟩,rfl,rfl⟩
    | some raw =>
      if htd : raw.kind=.boolean then
        let hts ← rawShape raw [N,C] "trial channel mask array shape"
        let result : MaskHeader N C := ⟨static,hs.down,hd,some raw,
          by intro other h; cases h; exact hts.down,
          by intro other h; cases h; exact htd⟩
        require (∀ i, ∃ c, trialAt result i c=true) "at least one observed channel per trial"
        require (∀ i c, trialAt result i c=true → staticAt result c=true) "trial cannot enable static absence"
        pure ⟨result,rfl,rfl⟩
      else .error (.sourceValueError "trial channel mask boolean dtype")
  else .error (.sourceValueError "static channel mask boolean dtype")

structure RawTimeline where
  arrays : Fin 17 → RawArray
  complete : Bool
  onlineCausal : Bool

structure RawQuality where
  schema : Value
  relativePTP : RawArray
  std : RawArray
  scale : RawArray
  mask : RawArray

structure RawDataset where
  historical : Bool
  coordinates : Value
  name : Value
  x : RawArray
  channelNames : List String
  positions : RawArray
  staticMask : RawArray
  trialMask : Option RawArray
  subjects : RawArray
  timeline : RawTimeline
  labels : Option RawArray
  quality : Option RawQuality
  metadataIsDataFrame : Bool
  metadataRows : ℕ
  metadataColumns : ℕ
  preprocessingArgs : Fields
  provenance : Fields
  provenanceIsMapping : Bool := true
  provenanceIsDict : Bool := true
  dictImpliesMapping : provenanceIsDict=true → provenanceIsMapping=true
  identityCache : Option (List PersonalERPInput.IdentityRecord)
  lineage : LineageImage
  attestation : Option Value

def toTexts (p : Operations) (raw : RawArray) : R RawArray := do
  let converted ← p.cast .string raw
  pure converted.1

def toFloats (p : Operations) (raw : RawArray) : R RawArray := do
  let converted ← p.cast .float64 raw
  pure converted.1

def comparedZero (p : Operations) (reverse : Bool) (raw : RawArray) : R (List Bool) :=
  raw.cells.mapM (fun cell => if reverse then p.less cell zeroCell else p.less zeroCell cell)

def assertFinite (p : Operations) (raw : RawArray) (site : String) : R Unit := do
  let finite ← p.finiteAll raw
  require (finite=true) site

def observedCoordinates (p : Operations) {N C : ℕ} (raw : RawArray) (masks : MaskHeader N C) : R Unit := do
  let rows := ((List.finRange C).filter (staticAt masks)).map
    (fun c => (List.range 3).map (fun axis => raw.cells[c.val*3+axis]?.getD (.atom .other)))
  require ((← p.finiteRows raw.dtype rows)=true) "finite observed coordinates"
  let norms ← rows.mapM (p.normRow raw.dtype)
  let nonpositive ← norms.mapM (fun value => p.lessEqual value zeroCell)
  require (true ∉ nonpositive) "nonzero observed coordinate norms"
  -- Source calls np.linalg.norm a second time after the first guard.
  let norms ← rows.mapM (p.normRow raw.dtype)
  let tooLarge ← norms.mapM (fun value => p.less (.atom (.real (1/2))) value)
  require (true ∉ tooLarge) "registered head coordinates within half metre"

def groupRows (groups : List String) (group : String) : List ℕ :=
  (List.range groups.length).filter (fun i => decide (groups[i]?=some group))

def adjacent (p : Operations) (dtype : String) (cells : List Cell) (strict : Bool) (site : String) : R Unit := do
  let differences ← (cells.zip (cells.drop 1)).mapM (fun pair => p.subtract dtype pair.2 pair.1)
  let bad ← differences.mapM (fun delta => if strict then p.lessEqual delta zeroCell else p.less delta zeroCell)
  require (true ∉ bad) site

def homogeneous (p : Operations) (cells : List Cell) (site : String) : R Unit := do
  match cells with
  | [] => .error (.inherited (.fieldFailure (site ++ " empty indexed group")))
  | first :: _ =>
    let same ← cells.mapM (p.equal first)
    require (false ∉ same) site

def timelineGroups (p : Operations) (raw : RawTimeline) (arrays : Fin 17 → RawArray)
    (allTargets allRepetitions : Bool) : List String → R Unit
  | [] => .ok ()
  | group :: tail => do
    let rows := groupRows ((arrays 1).cells.map cellText) group
    let onset ← nativeRows (arrays 5) rows "group onset"
    adjacent p (arrays 5).dtype onset false "chronological event groups"
    homogeneous p (← nativeRows (arrays 2) rows "group subject") "one subject per group"
    let _ ← ([10,11,12,13] : List (Fin 17)).mapM (fun j => do
      homogeneous p (← nativeRows (arrays j) rows "group identity") "homogeneous event group identities")
    let candidates ← nativeRows (arrays 14) rows "group candidate"
    if allTargets then
      let targets ← nativeRows (arrays 15) rows "group target"
      homogeneous p targets "one target per group"
      require (2 ≤ (candidates.map cellText).dedup.length) "at least two candidates per target group"
      let target ← nthCell targets 0 "target membership"
      require (cellText target ∈ candidates.map cellText) "target belongs to candidate group"
    if allRepetitions then
      let _ ← (candidates.map cellText).dedup.mergeSort (· ≤ ·) |>.mapM (fun candidate => do
        let selected := rows.filter (fun i => decide ((arrays 14).cells[i]?.map cellText=some candidate))
        let repetitions ← nativeRows (arrays 16) selected "candidate repetitions"
        adjacent p (arrays 16).dtype repetitions true "increasing candidate repetitions"
        if raw.complete then
          let first ← nthCell repetitions 0 "first repetition"
          let expected ← (List.range repetitions.length).mapM (fun i => p.addInt64 (integerCell i) first)
          let same ← (repetitions.zip expected).mapM (fun pair => p.equal pair.1 pair.2)
          require (false ∉ same) "complete contiguous original repetitions")
      pure ()
    timelineGroups p raw arrays allTargets allRepetitions tail

/-- Literal validate() sequence. Each mask/index operation can fail before
later checks when independent raw lengths differ. No equal-length gate is
inserted here. actual __post_init__ supplies that fact on constructor routes. -/
def decodeTimelineArrays (p : Operations) (raw : RawTimeline) : R (Fin 17 → RawArray) := do
  let event ← toTexts p (raw.arrays 0)
  let groups ← toTexts p (raw.arrays 1)
  let subjects ← toTexts p (raw.arrays 2)
  let statuses ← toTexts p (raw.arrays 8)
  let details ← toTexts p (raw.arrays 9)
  let onset ← toFloats p (raw.arrays 5)
  let availableAt ← toFloats p (raw.arrays 6)
  let candidates ← toTexts p (raw.arrays 14)
  let targets ← toTexts p (raw.arrays 15)
  let arrays := fun j : Fin 17 =>
    if j=0 then event else if j=1 then groups else if j=2 then subjects else
    if j=5 then onset else if j=6 then availableAt else if j=8 then statuses else
    if j=9 then details else if j=14 then candidates else if j=15 then targets else raw.arrays j
  pure arrays

def validateDecodedTimeline (p : Operations) (raw : RawTimeline) (arrays : Fin 17 → RawArray)
    (N : Option ℕ) : R (Fin 17 → RawArray) := do
  let event := arrays 0
  let groups := arrays 1
  let subjects := arrays 2
  let statuses := arrays 8
  let onset := arrays 5
  let availableAt := arrays 6
  let candidates := arrays 14
  let targets := arrays 15
  let M ← arrayLength (raw.arrays 0)
  require (0 < M) "nonempty scheduled event timeline"
  require (∀ j, (arrays j).shape.length=1) "one dimensional scheduled event fields"
  require ((event.cells.map cellText).Nodup ∧ "" ∉ event.cells.map cellText) "unique nonempty event ids"
  require ("" ∉ groups.cells.map cellText ∧ "" ∉ subjects.cells.map cellText) "event group and participant ids"
  require (∀ status ∈ statuses.cells.map cellText, status ∈ timelineStatuses) "known scheduled event status"
  require ((arrays 3).kind.integer=true) "integer event stimulus ids"
  require ((arrays 4).kind.integer=true) "integer event onset samples"
  require (true ∉ (← comparedZero p true (arrays 4))) "nonnegative event onset samples"
  assertFinite p onset "finite event onset"
  require ((arrays 7).kind.integer=true) "integer event evidence indices"
  let strip := p.input.data.input.original.execution.strip
  let candidatePresent := candidates.cells.map (fun c => decide (strip (cellText c) ≠ ""))
  let targetPresent := targets.cells.map (fun c => decide (strip (cellText c) ≠ ""))
  require (true ∉ candidatePresent ∨ false ∉ candidatePresent) "uniform candidate presence"
  require (true ∉ targetPresent ∨ false ∉ targetPresent) "uniform target presence"
  require (true ∉ targetPresent ∨ false ∉ candidatePresent) "targets require candidates"
  let repetitionPresent ← (arrays 16).cells.mapM (fun c => p.greaterEqual c zeroCell)
  require (true ∉ repetitionPresent ∨ false ∉ repetitionPresent) "uniform repetition presence"
  let tooNegative ← (arrays 16).cells.mapM (fun c => p.less c (integerCell (-1)))
  require (true ∉ tooNegative) "repetitions at least minus one"
  require (true ∉ repetitionPresent ∨ false ∉ candidatePresent) "repetitions require candidates"
  let available := statuses.cells.map (fun c => decide (cellText c="available"))
  let availableEvidence ← masked (arrays 7) available "available evidence"
  let bad ← availableEvidence.mapM (fun c => p.less c zeroCell)
  require (true ∉ bad) "available evidence nonnegative"
  let unavailableEvidence ← masked (arrays 7) (available.map not) "unavailable evidence"
  let equals ← unavailableEvidence.mapM (fun c => p.equal c (integerCell (-1)))
  require (false ∉ equals) "unavailable evidence minus one"
  let finite ← (← masked availableAt available "available time").mapM (fun c => pure (finiteNumber (cellAtom c)))
  require (false ∉ finite) "finite availability on available events"
  let finite ← (← masked availableAt (available.map not) "unavailable time").mapM (fun c => pure (finiteNumber (cellAtom c)))
  require (true ∉ finite) "nonfinite availability on unavailable events"
  let availableOnset ← masked onset available "available onset"
  let times ← masked availableAt available "available ordering time"
  let early ← (times.zip availableOnset).mapM (fun pair => p.less pair.1 pair.2)
  require (true ∉ early) "event evidence not before onset"
  let mapped ← p.int64Cells (arrays 7).kind (arrays 7).dtype availableEvidence
  let expected := N.getD mapped.1.length
  require (mapped.1.Perm ((List.range expected).map Int.ofNat)) "event evidence row bijection"
  require (mapped.1.length=expected) "event availability denominator"
  let datasets ← toTexts p (raw.arrays 10)
  let sessions ← toTexts p (raw.arrays 11)
  let runs ← toTexts p (raw.arrays 12)
  let selections ← toTexts p (raw.arrays 13)
  let arrays := fun j : Fin 17 => if j=10 then datasets else if j=11 then sessions else
    if j=12 then runs else if j=13 then selections else arrays j
  -- np.r_[0, ..., len(groups)] has one start/stop pair even when unique
  -- groups is empty; the source's strict three-way zip then raises here.
  require (groups.cells ≠ []) "empty unique groups versus boundary strict zip"
  timelineGroups p raw arrays (decide (false ∉ targetPresent)) (decide (false ∉ repetitionPresent))
    ((groups.cells.map cellText).dedup.mergeSort (· ≤ ·))
  pure arrays

def timelineValidateOptional (p : Operations) (raw : RawTimeline) (N : Option ℕ) : R (Fin 17 → RawArray) := do
  let arrays ← decodeTimelineArrays p raw
  validateDecodedTimeline p raw arrays N

def timelineValidate (p : Operations) (raw : RawTimeline) (N : ℕ) : R (Fin 17 → RawArray) :=
  timelineValidateOptional p raw (some N)

structure LabelHeader (N : ℕ) where
  raw : RawArray
  shape : raw.shape=[N]
  dtype : raw.kind.integer=true

def labelChecks (p : Operations) (raw : Option RawArray) (N : ℕ)
    (timeline : Fin 17 → RawArray) (requireLabels : Bool) :
    R {labels : Option (LabelHeader N) // labels.map LabelHeader.raw=raw} := do
  let labels ← (show R {labels : Option (LabelHeader N) // labels.map LabelHeader.raw=raw} from do
  match raw with
  | none => pure ⟨none,rfl⟩
  | some labels =>
    let shape ← rawShape labels [N] "one dimensional labels aligned to epochs"
    let dtype ← if hd : labels.kind.integer=true then pure (PLift.up hd)
      else .error (.sourceValueError "nonbool integer labels")
    let strip := p.input.data.input.original.execution.strip
    let candidate := (timeline 14).cells.map cellText
    let target := (timeline 15).cells.map cellText
    if candidate ≠ [] ∧ target ≠ [] ∧ (∀ c ∈ candidate, strip c ≠ "") ∧ (∀ t ∈ target, strip t ≠ "") then
      let evidence ← p.cast .int64 (timeline 7)
      let indices := evidence.1.cells.map cellInteger
      let available := indices.map (fun i => decide (0 ≤ i))
      let cs ← masked (timeline 14) available "candidate labels"
      let ts ← masked (timeline 15) available "target labels"
      let expected := (cs.zip ts).map (fun row => if cellText row.1=cellText row.2 then (1 : ℤ) else 0)
      let kept := (indices.zip available).filterMap (fun row => if row.2 then some row.1 else none)
      let cast ← p.cast .int64 labels
      let values := cast.1.cells.map cellInteger
      require (∀ pair ∈ kept.zip expected, values[pair.1.toNat]?=some pair.2) "binary labels agree with candidate targets"
    pure ⟨some ⟨labels,shape.down,dtype.down⟩,rfl⟩)
  require (requireLabels=false ∨ raw.isSome=true) "required labels present"
  pure labels

def scheduledExclusions (p : Operations) (timeline : RawTimeline) : R Unit := do
  let datasets ← toTexts p (timeline.arrays 10)
  let subjects ← toTexts p (timeline.arrays 2)
  let datasetLength ← match datasets.shape with
    | [] => .error (.inherited (.fieldFailure "scheduled exclusion iteration over zero dimensional dataset array"))
    | first :: _ => pure first
  let subjectLength ← match subjects.shape with
    | [] => .error (.inherited (.fieldFailure "scheduled exclusion iteration over zero dimensional subject array"))
    | first :: _ => pure first
  -- Only an actually yielded tuple is hashed. Higher-rank arrays with a
  -- zero first dimension yield no tuple and can pass this source stage.
  -- With nonempty iterands, hashing fails before a later strict-zip length
  -- mismatch could be discovered.
  if 0 < datasetLength ∧ 0 < subjectLength then
    if datasets.shape.length ≠ 1 ∨ subjects.shape.length ≠ 1 then
      .error (.inherited (.fieldFailure "scheduled exclusion unhashable ndarray tuple element"))
  require (datasetLength=subjectLength) "scheduled exclusion strict zip lengths"
  require (∀ pair ∈ datasets.cells.zip subjects.cells,
    PersonalERPInput.sourceParticipantExcluded p.input.data.input.original (cellText pair.1) (cellText pair.2)=false)
    "scheduled source participants author admitted"

def subjectAlignment (p : Operations) (timeline : RawTimeline) (subjects : RawArray) (N : ℕ) : R Unit := do
  let eventSubjects ← toTexts p (timeline.arrays 2)
  let eventEvidence ← p.cast .int64 (timeline.arrays 7)
  let indices := eventEvidence.1.cells.map cellInteger
  let available := indices.map (fun z => decide (0 ≤ z))
  let keptSubjects ← masked eventSubjects available "subject-alignment event subjects"
  let kept : List ℤ := (indices.zip available).filterMap (fun row => if row.2 then some row.1 else none)
  let actualSubjects ← toTexts p subjects
  require (∀ pair ∈ kept.zip keptSubjects, pair.1 < (N : ℤ) ∧ actualSubjects.cells[pair.1.toNat]?=some pair.2)
    "event evidence agrees with subject ids"

def placeholders (p : Operations) (raw : RawArray) {N C : ℕ} (T : ℕ) (masks : MaskHeader N C) : R Unit := do
  if ∃ c, staticAt masks c=false then
    let hidden := (List.finRange N).flatMap (fun i => ((List.finRange C).filter
      (fun c => !(staticAt masks c))).flatMap (fun c => (List.range T).map
        (fun t => raw.cells[(i.val*C+c.val)*T+t]?.getD (.atom .other))))
    let equals ← hidden.mapM (fun c => p.equal c (.atom (.real 0)))
    require (false ∉ equals) "static masked placeholders are zero"
  match masks.trialRaw with
  | none => pure ()
  | some _ =>
    let hidden := (List.finRange N).flatMap (fun i => ((List.finRange C).filter
      (fun c => !(trialAt masks i c))).flatMap (fun c => (List.range T).map
        (fun t => raw.cells[(i.val*C+c.val)*T+t]?.getD (.atom .other))))
    let equals ← hidden.mapM (fun c => p.equal c (.atom (.real 0)))
    require (false ∉ equals) "trial masked placeholders are zero"

structure QualityHeader (N C : ℕ) where
  raw : RawQuality
  relativeShape : raw.relativePTP.shape=[N,C]
  stdShape : raw.std.shape=[N,C]
  scaleShape : raw.scale.shape=[N]
  maskShape : raw.mask.shape=[N,C]
  relativeDType : raw.relativePTP.kind=.floating
  stdDType : raw.std.kind=.floating
  scaleDType : raw.scale.kind=.floating
  maskDType : raw.mask.kind=.boolean

def kindGuard (raw : RawArray) (kind : Kind) (site : String) : R (PLift (raw.kind=kind)) :=
  if h : raw.kind=kind then .ok ⟨h⟩ else .error (.sourceValueError site)

def qualityChecks (p : Operations) {N C : ℕ} (qc : RawQuality) (masks : MaskHeader N C) :
    R {header : QualityHeader N C // header.raw=qc} := do
  require (qc.schema=.text "n2p3net_epoch_qc_features/1") "QC feature schema"
  let rs ← rawShape qc.relativePTP [N,C] "QC relative PTP shape"
  let ss ← rawShape qc.std [N,C] "QC standard deviation shape"
  let ms ← rawShape qc.mask [N,C] "QC observed mask shape"
  let es ← rawShape qc.scale [N] "QC epoch scale shape"
  let rd ← kindGuard qc.relativePTP .floating "QC relative PTP floating dtype"
  let sd ← kindGuard qc.std .floating "QC standard deviation floating dtype"
  let ed ← kindGuard qc.scale .floating "QC epoch scale floating dtype"
  let md ← kindGuard qc.mask .boolean "QC observed mask boolean dtype"
  let observed := fun (i : Fin N) (c : Fin C) =>
    cellBool (qc.mask.cells[i.val*C+c.val]?.getD (.atom .other))
  require (∀ i, ∃ c, observed i c=true) "QC observed channel per epoch"
  let _ ← [qc.relativePTP,qc.std,qc.scale].mapM (fun array => do
    assertFinite p array "finite QC feature"
    require (true ∉ (← comparedZero p true array)) "nonnegative QC feature")
  require (∀ i c, observed i c=trialAt masks i c) "QC actual effective observed mask"
  pure ⟨⟨qc,rs.down,ss.down,es.down,ms.down,rd.down,sd.down,ed.down,md.down⟩,rfl⟩

structure Validated (raw : RawDataset) where
  tensor : TensorHeader raw.x
  masks : MaskHeader tensor.N tensor.C
  staticCorrespondence : masks.staticRaw=raw.staticMask
  trialCorrespondence : masks.trialRaw=raw.trialMask
  positionShape : raw.positions.shape=[tensor.C,3]
  positionDType : raw.positions.kind.numeric=true
  subjectShape : raw.subjects.shape=[tensor.N]
  subjectCast : CastResult .string raw.subjects
  convertedTimeline : Fin 17 → RawArray
  labels : Option (LabelHeader tensor.N)
  labelCorrespondence : labels.map LabelHeader.raw=raw.labels
  quality : Option (QualityHeader tensor.N tensor.C)
  qualityCorrespondence : quality.map QualityHeader.raw=raw.quality
  provenanceDict : raw.provenanceIsDict=true

/-- Public raw-array validator. A malformed label cannot disappear before
physical/exclusion/preprocessing/mask/subject/timeline checks have executed. -/
def validate (p : Operations) (raw : RawDataset) (requireLabels : Bool) : R (Validated raw) := do
  let input := p.input.data.input.original
  liftCheck (checkPhysicalCoordinates raw.historical raw.coordinates
    (if raw.provenanceIsMapping then raw.provenance.get "data_alignment" else none))
  match raw.identityCache with
  | none => pure ()
  | some rows => liftCheck (PersonalERPInput.checkIdentityExclusions input rows)
  scheduledExclusions p raw.timeline
  let args ← liftCheck (readPreprocessingArgs raw.preprocessingArgs)
  let declaration ← PersonalERPDataRuntime.checkPreprocessing p.input.data args
  require (raw.timeline.onlineCausal=decide (fieldDefault args "filter_phase"=.text "forward"))
    "event causal flag matches preprocessing"
  let name ← PersonalERPModelRuntime.text raw.name "dataset name string type"
  require (input.execution.strip name ≠ "") "nonempty dataset name"
  let tensor ← tensorHeader raw.x declaration.nTimes
  let canonical ← liftCheck (canonicalNames input.canonicalChannel raw.channelNames)
  require (canonical.length=tensor.C ∧ canonical.Nodup) "distinct aligned physical electrodes"
  let positions ← rawShape raw.positions [tensor.C,3] "physical channel positions geometry"
  let positionDType ← if hd : raw.positions.kind.numeric=true then pure (PLift.up hd)
    else .error (.sourceValueError "numeric physical channel positions")
  let masks ← maskHeader tensor.N tensor.C raw.staticMask raw.trialMask
  observedCoordinates p raw.positions masks.1
  let subjectShape ← rawShape raw.subjects [tensor.N] "one dimensional subject ids aligned to epochs"
  let subjects ← p.cast .string raw.subjects
  require (∀ value ∈ subjects.1.cells, input.execution.strip (cellText value) ≠ "") "nonempty subject ids"
  let timeline ← timelineValidate p raw.timeline tensor.N
  subjectAlignment p raw.timeline raw.subjects tensor.N
  let labels ← labelChecks p raw.labels tensor.N timeline requireLabels
  assertFinite p raw.x "finite EEG tensor values"
  placeholders p raw.x tensor.T masks.1
  let quality ← (show R {qc : Option (QualityHeader tensor.N tensor.C) // qc.map QualityHeader.raw=raw.quality} from do
    match raw.quality with
    | none => pure ⟨none,rfl⟩
    | some qc =>
      let checked ← qualityChecks p qc masks.1
      pure ⟨some checked.1,by simp only [Option.map_some,checked.2]⟩)
  require (raw.metadataIsDataFrame=true) "metadata DataFrame type"
  require (raw.metadataRows=0 ∨ raw.metadataColumns=0 ∨ raw.metadataRows=tensor.N) "aligned metadata rows"
  let provenanceType ← if h : raw.provenanceIsDict=true then pure (PLift.up h)
    else .error (.sourceValueError "provenance dictionary type")
  let _ ← PersonalERPExecutionRuntime.executionBinding p.input args (raw.provenance.get "preprocessing_execution")
  liftCheck (checkDatasetTail (subjects.1.cells.map cellText)
    (raw.identityCache.map (fun rows => rows.map PersonalERPInput.IdentityRecord.localSubject)) raw.lineage raw.attestation)
  pure ⟨tensor,masks.1,masks.2.1,masks.2.2,positions.down,positionDType.down,subjectShape.down,subjects,
    timeline,labels.1,labels.2,quality.1,quality.2,provenanceType.down⟩

theorem tensor_rank_fault_precedes_dtype (raw : RawArray) (nTimes : ℤ)
    (h : ∀ N C T, raw.shape ≠ [N,C,T]) :
    tensorHeader raw nTimes=.error (.sourceValueError "EEG tensor rank three") := by
  unfold tensorHeader
  split <;> rename_i hs
  · exact False.elim (h _ _ _ hs)
  · rfl

theorem static_dtype_fault_precedes_trial_shape (N C : ℕ) (static : RawArray) (trial : Option RawArray)
    (h : static.kind ≠ .boolean) :
    maskHeader N C static trial=.error (.sourceValueError "static channel mask boolean dtype") := by
  simp [maskHeader,h]

end
end N2P3.PersonalERPArrayIngress
