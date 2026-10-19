import N2P3.FrozenState
import N2P3.StateLedger
import N2P3.PersonalState
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Sort
import Mathlib.Tactic.Linarith

/-!
# Actual personal inference admission and the raw callback boundary

Source: calibrated.py:464-520,661-763,811-821; brainsync.py:645-668;
brainsync_model_task.py:122-134,405-437. Objects below are already decoded;
physical participant resolution, datetime parsing, channel-layout resolution,
EpochDataset validation, preprocessing/QC/EOG and execution/basis checks remain
explicit external steps. We do not assume that their successful result implies
the desired whole-route authorization or that every source object is immutable.

The scope helper uses pandas groupby(dropna=True): missing session identities
are skipped and a column-complete empty table has no groups. These weak source
paths are preserved. Strong per-row conclusions require the actual row to have
a present session ID; the raw official callback checks the recording before QC
even when processing later reports no usable evidence. No physiological clock
authenticity or neural-forward correctness is inferred.
Present group keys here are already strings, as emitted by the official raw
builder. Arbitrary pandas objects (e.g. integer 1 versus text "1") are not
merged before grouping: their separate Python grouping/stringification is an
unrefined boundary. Input materialization and participant resolution retain
their independent failure steps rather than assuming successful extraction.
-/

namespace N2P3.PersonalAdmission

open N2P3.FrozenState

noncomputable section

attribute [local instance] Classical.propDecidable

inductive Fault where
  | reusedSession | beforeAvailable | participant | simulation
  | acquisitionReference | geometry | requestChannels | requestOcular | requestQC
  | rawReference
  | materializedReference
  | inputContract | execution | basis | dataset | missingTimingColumns
  | inconsistentFiniteTime (session : String)
  | noUsableEvidence | external (operation : String)
  deriving DecidableEq

inductive Time where
  | finite (value : ℝ)
  | nan | positiveInfinity | negativeInfinity

/-- Python finite JSON equality identifies bool/int/real numeric values.
Objects are key/value maps; their order is immaterial. Parsed keys are unique.
The source's identity checks (is None/is False) stay separate from this rule. -/
inductive PyValue where
  | null
  | number (value : ℝ)
  | text (value : String)
  | sequence (values : List PyValue)
  | object (fields : String → Option PyValue)

mutual
def normalizeValue : Value → PyValue
  | .null => .null
  | .boolean b => .number (if b then 1 else 0)
  | .integer z => .number z
  | .number r => .number r
  | .text s => .text s
  | .sequence xs => .sequence (normalizeValues xs)
  | .object fs => .object (normalizeFields fs)
def normalizeValues : Values → List PyValue
  | .nil => []
  | .cons v tail => normalizeValue v :: normalizeValues tail
def normalizeFields : Fields → String → Option PyValue
  | .nil => fun _ => none
  | .cons k v tail => fun key => if key = k then some (normalizeValue v) else normalizeFields tail key
end

def PyEq (a b : Value) : Prop := normalizeValue a = normalizeValue b

theorem python_true_equals_numeric_one : PyEq (.boolean true) (.number 1) := rfl

theorem python_false_equals_integer_zero : PyEq (.boolean false) (.integer 0) := by
  simp [PyEq, normalizeValue]

theorem python_integer_equals_real (z : ℤ) : PyEq (.integer z) (.number z) := rfl

theorem python_object_field_order_irrelevant (a b : String) (x y : Value) (h : a ≠ b) :
    PyEq (.object (Fields.ofList [(a, x), (b, y)]))
      (.object (Fields.ofList [(b, y), (a, x)])) := by
  unfold PyEq normalizeValue
  congr 1
  funext key
  simp only [Fields.ofList, normalizeFields]
  by_cases ha : key = a <;> by_cases hb : key = b <;> simp_all

def Time.notLater : Time → ℝ → Prop
  | .finite t, cutoff => t ≤ cutoff
  | .nan, _ => False
  | .positiveInfinity, _ => False
  | .negativeInfinity, _ => True

structure ModelScope where
  participant : String
  calibrationSessions : List String
  available : ℝ

def calibrationSessionRows : List String → Values
  | [] => .nil
  | id :: tail => .cons (.object (Fields.ofList [("session_id", .text id)])) (calibrationSessionRows tail)

def readCalibrationSessionIds : Values → Option (List String)
  | .nil => some []
  | .cons (.object fs) tail => do
      let id ← (fs.get "session_id").bind Value.readText
      let rest ← readCalibrationSessionIds tail
      pure (id :: rest)
  | _ => none

def metadataFields : Value → Option Fields
  | .object fs => some fs
  | _ => none

/-- A projection on canonical producer metadata, not a replacement constructor
validator. The real direct helper does not impose these parsing/type guards. -/
def scopeFromMetadata (metadata : Value) : Option ModelScope := do
  let fs ← metadataFields metadata
  let participant ← (fs.get "participant_key").bind Value.readText
  let sessions ← (fs.get "calibration_sessions").bind Value.readSequence
  let ids ← readCalibrationSessionIds sessions
  let available ← (fs.get "calibration_available_timestamp_s").bind StateLedger.numericScalar
  pure ⟨participant, ids, available⟩

def scopeMetadata (m : ModelScope) (extra : Fields := .nil) : Value :=
  .object (.cons "participant_key" (.text m.participant)
    (.cons "calibration_sessions" (.sequence (calibrationSessionRows m.calibrationSessions))
      (.cons "calibration_available_timestamp_s" (.number m.available) extra)))

theorem read_canonical_calibration_ids (ids : List String) :
    readCalibrationSessionIds (calibrationSessionRows ids) = some ids := by
  induction ids with
  | nil => rfl
  | cons id tail ih => simp [readCalibrationSessionIds, calibrationSessionRows, Fields.ofList, Fields.get, Value.readText, ih]

theorem canonical_scope_metadata_projection (m : ModelScope) (extra : Fields) :
    scopeFromMetadata (scopeMetadata m extra) = some m := by
  cases m
  simp [scopeFromMetadata, scopeMetadata, metadataFields, Value.readText, Value.readSequence,
    Fields.get, read_canonical_calibration_ids, StateLedger.numericScalar]

def loadedScope (result : StateLedger.Check PersonalState.ConstructorInput) : Option ModelScope :=
  match result with
  | .error _ => none
  | .ok input => scopeFromMetadata input.metadata

/-- Full saved metadata and named tensors feed the actual copied constructor
input first; only then are the three source scope fields projected. -/
theorem sealed_packet_recovers_scope_fields (mh : PersonalState.MetadataHash)
    (sh : PersonalState.StateHash) (m : ModelScope) (extra : Fields)
    (state : PersonalState.NamedState) :
    loadedScope (PersonalState.loadInput mh sh
      (PersonalState.saveView mh sh ⟨scopeMetadata m extra, state⟩)) = some m := by
  unfold scopeMetadata
  rw [PersonalState.current_source_copy_restore]
  exact canonical_scope_metadata_projection m extra

def sessionReason (m : ModelScope) (session : String) (started : Time) : Option Fault :=
  if session ∈ m.calibrationSessions then some .reusedSession
  else if started.notLater m.available then some .beforeAvailable else none

theorem finite_session_success_iff (m : ModelScope) (id : String) (t : ℝ) :
    sessionReason m id (.finite t) = none ↔
      id ∉ m.calibrationSessions ∧ m.available < t := by
  unfold sessionReason Time.notLater
  split_ifs <;> simp_all [not_le]

theorem reused_session_has_priority (m : ModelScope) (id : String) (t : Time)
    (h : id ∈ m.calibrationSessions) : sessionReason m id t = some .reusedSession := by
  simp [sessionReason, h]

theorem equality_is_not_prospective (m : ModelScope) (id : String)
    (h : id ∉ m.calibrationSessions) :
    sessionReason m id (.finite m.available) = some .beforeAvailable := by
  simp [sessionReason, Time.notLater, h]

/-- Direct helper comparison alone does not reject NaN. Official grouping
and the validated-record boundary provide the finite input separately. -/
theorem direct_nan_comparison_passes (m : ModelScope) (id : String)
    (h : id ∉ m.calibrationSessions) : sessionReason m id .nan = none := by
  simp [sessionReason, Time.notLater, h]

structure InputContract where
  channelNames : List String
  channelPositions : Value
  preprocessing : Value
  sourceReference : String
  nChannels : ℕ
  nTimes : ℕ
  simulation : Value
  ocularProcessing : Value
  continuousProcessing : Value

def InputContract.encode (c : InputContract) : Value := .object (Fields.ofList
  [("channel_names", .sequence (encodeTexts c.channelNames)),
    ("channel_positions_m", c.channelPositions), ("preprocessing", c.preprocessing),
    ("source_reference", .text c.sourceReference), ("n_channels", .integer c.nChannels),
    ("n_times", .integer c.nTimes), ("simulation_status", c.simulation),
    ("ocular_artifact_processing", c.ocularProcessing),
    ("continuous_artifact_processing", c.continuousProcessing)])

structure RawRequest where
  channels : List String
  ocular : Value
  continuousQC : Value

def processingModel (processing : Value) : Except Fault Value :=
  match processing with
  | .object fs => if fs.get "status" = some (.text "applied")
      then match fs.get "model" with
        | some model => .ok model
        | none => .error (.external "processing_model_key")
      else .ok .null
  | _ => .error (.external "processing_get")

def requestReason (request : RawRequest) (contract : InputContract) : Option Fault :=
  if request.channels ≠ contract.channelNames then some .requestChannels
  else match processingModel contract.ocularProcessing with
    | .error fault => some fault
    | .ok ocular => if ¬PyEq request.ocular ocular then some .requestOcular
      else match processingModel contract.continuousProcessing with
        | .error fault => some fault
        | .ok qc => if ¬PyEq request.continuousQC qc then some .requestQC else none

theorem request_success_iff (request : RawRequest) (contract : InputContract) :
    requestReason request contract = none ↔ request.channels = contract.channelNames ∧
      (∃ ocular, processingModel contract.ocularProcessing = .ok ocular ∧ PyEq request.ocular ocular) ∧
      (∃ qc, processingModel contract.continuousProcessing = .ok qc ∧ PyEq request.continuousQC qc) := by
  unfold requestReason
  cases processingModel contract.ocularProcessing <;> cases processingModel contract.continuousProcessing <;>
    split_ifs <;> simp_all
  all_goals split_ifs <;> simp_all

structure AcquisitionReference where
  label : String
  details : Value

structure RawRecording where
  participant : String
  session : String
  started : ℝ
  transport : Option String
  acquisitionReference : AcquisitionReference
  rawPositions : Value

def recordingSimulation (r : RawRecording) : Value :=
  .text (if r.transport = some "mock" then "declared_mock" else "not_declared")

def rawReason (m : ModelScope) (contract : InputContract)
    (references : List AcquisitionReference) (canonicalPositions : Value)
    (postReference : Bool) (r : RawRecording) : Option Fault :=
  if r.participant ≠ m.participant then some .participant
  else match sessionReason m r.session (.finite r.started) with
    | some fault => some fault
    | none => if recordingSimulation r ≠ contract.simulation then some .simulation
      else if references ≠ [] ∧ ¬(∃ ref ∈ references,
          ref.label = r.acquisitionReference.label ∧ PyEq ref.details r.acquisitionReference.details)
        then some .acquisitionReference
      else if ¬PyEq (if postReference then canonicalPositions else r.rawPositions) contract.channelPositions
        then some .geometry else none

theorem raw_success_iff (m : ModelScope) (contract : InputContract)
    (references : List AcquisitionReference) (canonicalPositions : Value)
    (postReference : Bool) (r : RawRecording) :
    rawReason m contract references canonicalPositions postReference r = none ↔
      r.participant = m.participant ∧ r.session ∉ m.calibrationSessions ∧ m.available < r.started ∧
      recordingSimulation r = contract.simulation ∧
      (references = [] ∨ ∃ ref ∈ references,
        ref.label = r.acquisitionReference.label ∧ PyEq ref.details r.acquisitionReference.details) ∧
      PyEq (if postReference then canonicalPositions else r.rawPositions) contract.channelPositions := by
  unfold rawReason
  cases hs : sessionReason m r.session (.finite r.started) with
  | some fault =>
      have hn : ¬(r.session ∉ m.calibrationSessions ∧ m.available < r.started) := by
        intro h
        have he := (finite_session_success_iff m r.session r.started).mpr h
        rw [hs] at he
        cases he
      split_ifs <;> simp_all
  | none =>
      have hh := (finite_session_success_iff m r.session r.started).mp hs
      split_ifs <;> simp_all <;> tauto

theorem wrong_participant_before_every_later_raw_guard (m : ModelScope) (contract : InputContract)
    (refs : List AcquisitionReference) (positions : Value) (post : Bool) (r : RawRecording)
    (h : r.participant ≠ m.participant) :
    rawReason m contract refs positions post r = some .participant := by
  simp [rawReason, h]

theorem empty_acquisition_reference_list_does_not_require_match
    (m : ModelScope) (contract : InputContract) (positions : Value) (post : Bool) (r : RawRecording)
    (hp : r.participant = m.participant) (hs : sessionReason m r.session (.finite r.started) = none)
    (hsi : recordingSimulation r = contract.simulation)
    (hg : PyEq (if post then positions else r.rawPositions) contract.channelPositions) :
    rawReason m contract [] positions post r = none := by
  simp [rawReason, hp, hs, hsi, hg]

structure TrialRow where
  session : Option String
  started : Time

structure Session where
  id : String
  started : ℝ

def insertFresh (acc : List String) (id : String) : List String :=
  if id ∈ acc then acc else acc ++ [id]

theorem mem_insertFresh (x id : String) (acc : List String) :
    x ∈ insertFresh acc id ↔ x ∈ acc ∨ x = id := by
  unfold insertFresh
  split_ifs with h
  · constructor
    · exact Or.inl
    · rintro (hx | hx)
      · exact hx
      · exact hx ▸ h
  · simp [List.mem_append]

theorem mem_fold_insertFresh (x : String) (ids acc : List String) :
    x ∈ ids.foldl insertFresh acc ↔ x ∈ acc ∨ x ∈ ids := by
  induction ids generalizing acc with
  | nil => simp
  | cons id tail ih => simp only [List.foldl_cons, ih, mem_insertFresh, List.mem_cons, or_assoc]

/-- First-seen nonmissing pandas group identities, before summary-time sorting. -/
def sessionIds (rows : List TrialRow) : List String :=
  (rows.filterMap TrialRow.session).foldl insertFresh []

theorem present_row_id_in_groups (rows : List TrialRow) (row : TrialRow)
    (hr : row ∈ rows) (id : String) (hi : row.session = some id) : id ∈ sessionIds rows := by
  unfold sessionIds
  rw [mem_fold_insertFresh]
  right
  exact List.mem_filterMap.mpr ⟨row, hr, hi⟩

def groupRows (rows : List TrialRow) (id : String) : List TrialRow :=
  rows.filter (fun row => row.session = some id)

def readGroup (rows : List TrialRow) (id : String) : Except Fault Session :=
  match groupRows rows id with
  | [] => .error (.inconsistentFiniteTime id)
  | first :: tail => match first.started with
    | .finite t => if ∀ row ∈ tail, row.started = .finite t
        then .ok ⟨id, t⟩ else .error (.inconsistentFiniteTime id)
    | _ => .error (.inconsistentFiniteTime id)

theorem read_group_success_recovers_every_group_row (rows : List TrialRow) (id : String)
    (s : Session) (h : readGroup rows id = .ok s) :
    s.id = id ∧ ∀ row ∈ rows, row.session = some id → row.started = .finite s.started := by
  cases hg : groupRows rows id with
  | nil => simp [readGroup, hg] at h
  | cons first tail =>
      cases ht : first.started with
      | nan => simp [readGroup, hg, ht] at h
      | positiveInfinity => simp [readGroup, hg, ht] at h
      | negativeInfinity => simp [readGroup, hg, ht] at h
      | finite t =>
          by_cases hc : ∀ row ∈ tail, row.started = .finite t
          · simp only [readGroup, hg, ht, if_pos hc, Except.ok.injEq] at h
            subst s
            refine ⟨rfl, ?_⟩
            intro row hr hi
            have hm : row ∈ first :: tail := by
              rw [← hg]
              exact List.mem_filter.mpr ⟨hr, by simpa using hi⟩
            rcases List.mem_cons.mp hm with he | he
            · subst row
              exact ht
            · exact hc row he
          · simp [readGroup, hg, ht, hc] at h

def buildGroups (rows : List TrialRow) : List String → Except Fault (List Session)
  | [] => .ok []
  | id :: tail => match readGroup rows id with
    | .error fault => .error fault
    | .ok first => match buildGroups rows tail with
      | .error fault => .error fault
      | .ok rest => .ok (first :: rest)

theorem built_groups_contain_each_requested_identity (rows : List TrialRow) (ids : List String)
    (sessions : List Session) (h : buildGroups rows ids = .ok sessions)
    (id : String) (hi : id ∈ ids) :
    ∃ s ∈ sessions, s.id = id ∧ readGroup rows id = .ok s := by
  induction ids generalizing sessions with
  | nil => simp at hi
  | cons first tail ih =>
      cases hf : readGroup rows first with
      | error fault => simp [buildGroups, hf] at h
      | ok group =>
          cases ht : buildGroups rows tail with
          | error fault => simp [buildGroups, hf, ht] at h
          | ok groups =>
              simp only [buildGroups, hf, ht, Except.ok.injEq] at h
              subst sessions
              rcases List.mem_cons.mp hi with he | he
              · subst id
                exact ⟨group, by simp, (read_group_success_recovers_every_group_row _ _ _ hf).1, hf⟩
              · obtain ⟨s, hs, hid, hr⟩ := ih groups ht he
                exact ⟨s, by simp [hs], hid, hr⟩

def sortedSessions (sessions : List Session) : List Session :=
  sessions.mergeSort (fun a b => decide (a.started < b.started ∨ a.started = b.started ∧ a.id ≤ b.id))

theorem sorted_sessions_preserve_membership (sessions : List Session) (s : Session) :
    s ∈ sortedSessions sessions ↔ s ∈ sessions :=
  (List.mergeSort_perm sessions _).mem_iff

def checkSessions (m : ModelScope) : List Session → Option Fault
  | [] => none
  | first :: tail => match sessionReason m first.id (.finite first.started) with
    | some fault => some fault
    | none => checkSessions m tail

theorem check_sessions_success_iff (m : ModelScope) (sessions : List Session) :
    checkSessions m sessions = none ↔
      ∀ s ∈ sessions, s.id ∉ m.calibrationSessions ∧ m.available < s.started := by
  induction sessions with
  | nil => simp [checkSessions]
  | cons first tail ih =>
      cases h : sessionReason m first.id (.finite first.started) with
      | none => simp [checkSessions, h, ih, (finite_session_success_iff _ _ _).mp h]
      | some fault =>
          have hn : ¬(first.id ∉ m.calibrationSessions ∧ m.available < first.started) := by
            intro hp
            have he := (finite_session_success_iff _ _ _).mpr hp
            rw [h] at he
            cases he
          simp [checkSessions, h, hn]

structure TimingTable where
  hasSessionColumn : Bool
  hasStartColumn : Bool
  rows : List TrialRow

def scopeReason (m : ModelScope) (table : TimingTable) : Option Fault :=
  if table.hasSessionColumn = false ∨ table.hasStartColumn = false then some .missingTimingColumns
  else match buildGroups table.rows (sessionIds table.rows) with
    | .error fault => some fault
    | .ok sessions => checkSessions m (sortedSessions sessions)

theorem empty_timing_table_passes (m : ModelScope) :
    scopeReason m ⟨true, true, []⟩ = none := by
  simp [scopeReason, sessionIds, buildGroups, sortedSessions, checkSessions]

theorem missing_ids_are_not_groups (rows : List TrialRow)
    (h : ∀ row ∈ rows, row.session = none) : sessionIds rows = [] := by
  have hf : rows.filterMap TrialRow.session = [] := by
    apply List.filterMap_eq_nil_iff.mpr
    exact h
  simp [sessionIds, hf]

theorem all_missing_ids_pass_scope (m : ModelScope) (rows : List TrialRow)
    (h : ∀ row ∈ rows, row.session = none) :
    scopeReason m ⟨true, true, rows⟩ = none := by
  simp [scopeReason, missing_ids_are_not_groups rows h, buildGroups, sortedSessions, checkSessions]

def exampleScope : ModelScope := ⟨"participant", ["calibration"], 100⟩

def exampleFutureRow : TrialRow := ⟨some "later", .finite 101⟩

/-- A nonempty four-epoch producer-style timing table passes the real group,
finite/unique stamp, chronological sort and prospective session checks. -/
theorem nonempty_finite_scope_example :
    scopeReason exampleScope ⟨true, true,
      [exampleFutureRow, exampleFutureRow, exampleFutureRow, exampleFutureRow]⟩ = none := by
  have hne : ("later" : String) ≠ "calibration" := by decide
  norm_num [scopeReason, exampleScope, exampleFutureRow, sessionIds, insertFresh,
    List.filterMap_cons, List.filterMap_nil, List.foldl_cons, List.foldl_nil,
    buildGroups, readGroup, groupRows, sortedSessions, checkSessions, sessionReason, Time.notLater, hne]

/-- The implication is limited to present session IDs, as in pandas groupby.
No absent ID or empty metadata is silently promoted to a checked epoch. -/
theorem successful_scope_checks_each_present_row (m : ModelScope) (table : TimingTable)
    (h : scopeReason m table = none) (row : TrialRow) (hr : row ∈ table.rows)
    (id : String) (hi : row.session = some id) :
    ∃ t : ℝ, row.started = .finite t ∧ id ∉ m.calibrationSessions ∧ m.available < t := by
  unfold scopeReason at h
  split at h <;> try { cases h }
  cases hb : buildGroups table.rows (sessionIds table.rows) with
  | error fault => simp [hb] at h
  | ok sessions =>
      have hcheck : checkSessions m (sortedSessions sessions) = none := by simpa [hb] using h
      obtain ⟨s, hs, hid, hread⟩ := built_groups_contain_each_requested_identity table.rows
        (sessionIds table.rows) sessions hb id (present_row_id_in_groups _ _ hr _ hi)
      have checked := (check_sessions_success_iff m _).mp hcheck s
        ((sorted_sessions_preserve_membership sessions s).mpr hs)
      have htime := (read_group_success_recovers_every_group_row _ _ _ hread).2 row hr hi
      exact ⟨s.started, htime, by simpa [hid] using checked⟩

def recordingRows (r : RawRecording) (n : ℕ) : List TrialRow :=
  List.replicate n ⟨some r.session, .finite r.started⟩

theorem official_recording_rows_have_present_finite_identity (r : RawRecording) (n : ℕ)
    (row : TrialRow) (h : row ∈ recordingRows r n) :
    row.session = some r.session ∧ row.started = .finite r.started := by
  have he := (List.mem_replicate.mp h).2
  subst row
  exact ⟨rfl, rfl⟩

structure Dataset where
  materializedContract : Except Fault Value
  participantResolution : Except Fault String
  executionIdentity : Value
  timing : TimingTable
  datasetValidation : Option Fault
  temporalBasisValidation : Option Fault

def inputReason (m : ModelScope) (contract : InputContract) (expectedExecution : Value)
    (hasTemporalBasis : Bool) (dataset : Dataset) : Option Fault :=
  match dataset.datasetValidation with
  | some fault => some fault
  | none => match dataset.materializedContract with
    | .error fault => some fault
    | .ok actual => if ¬PyEq actual contract.encode then some .inputContract
      else if expectedExecution ≠ .null ∧ ¬PyEq dataset.executionIdentity expectedExecution then some .execution
      else match (if hasTemporalBasis then dataset.temporalBasisValidation else none) with
        | some fault => some fault
        | none => match dataset.participantResolution with
          | .error fault => some fault
          | .ok participant => if participant ≠ m.participant then some .participant else none

theorem input_success_iff (m : ModelScope) (contract : InputContract) (expectedExecution : Value)
    (basis : Bool) (dataset : Dataset) :
    inputReason m contract expectedExecution basis dataset = none ↔
      dataset.datasetValidation = none ∧
      (∃ actual, dataset.materializedContract = .ok actual ∧ PyEq actual contract.encode) ∧
      (expectedExecution = .null ∨ PyEq dataset.executionIdentity expectedExecution) ∧
      (basis = false ∨ dataset.temporalBasisValidation = none) ∧
      dataset.participantResolution = .ok m.participant := by
  unfold inputReason
  cases dataset.datasetValidation <;> cases dataset.materializedContract <;>
    cases dataset.participantResolution <;> cases dataset.temporalBasisValidation <;>
    cases basis <;> simp_all
  all_goals split_ifs <;> simp_all <;> tauto

/-- Typed string-reference subdomain of _input_contract before JSON copying.
Nontext references and decoding errors enter materializedContract as errors. -/
def materializeContract (c : InputContract) : Except Fault Value :=
  if StateLedger.Nonblank c.sourceReference then .ok c.encode else .error .materializedReference

theorem nonblank_reference_materializes_literal_contract (c : InputContract)
    (h : StateLedger.Nonblank c.sourceReference) : materializeContract c = .ok c.encode := by
  simp [materializeContract, h]

theorem blank_reference_fails_materialization (c : InputContract)
    (h : ¬StateLedger.Nonblank c.sourceReference) : materializeContract c = .error .materializedReference := by
  simp [materializeContract, h]

def referenceReason (contract : InputContract) (reference : Option String) : Option Fault :=
  match reference with
  | some ref => if contract.sourceReference ≠ "measured_channel:" ++ ref then some .rawReference else none
  | none => if contract.sourceReference.startsWith "measured_channel:" then some .rawReference else none

def loadRoute (m : ModelScope) (contract : InputContract) (request : RawRequest)
    (references : List AcquisitionReference) (canonicalPositions expectedExecution : Value)
    (postReference hasTemporalBasis : Bool) (r : RawRecording)
    (prepare : Except Fault Unit) (reference : Option String)
    (process : RawRecording → Except Fault Dataset) : Except Fault Dataset :=
  match requestReason request contract with
  | some fault => .error fault
  | none => match prepare with
    | .error fault => .error fault
    | .ok () => match referenceReason contract reference with
      | some fault => .error fault
      | none => match rawReason m contract references canonicalPositions postReference r with
        | some fault => .error fault
        | none => match process r with
          | .error fault => .error fault
          | .ok dataset => match inputReason m contract expectedExecution hasTemporalBasis dataset with
            | some fault => .error fault
            | none => match scopeReason m dataset.timing with
              | some fault => .error fault
              | none => .ok dataset

theorem raw_refusal_precedes_and_is_independent_of_processing (m : ModelScope)
    (contract : InputContract) (request : RawRequest) (refs : List AcquisitionReference)
    (positions expectedExecution : Value) (post basis : Bool) (recording : RawRecording)
    (process : RawRecording → Except Fault Dataset) (fault : Fault) (reference : Option String)
    (hr : requestReason request contract = none)
    (href : referenceReason contract reference = none)
    (hf : rawReason m contract refs positions post recording = some fault) :
    loadRoute m contract request refs positions expectedExecution post basis recording (.ok ()) reference process = .error fault := by
  simp [loadRoute, hr, href, hf]

theorem loaded_route_checks_actual_recording (m : ModelScope) (contract : InputContract)
    (request : RawRequest) (refs : List AcquisitionReference) (positions expectedExecution : Value)
    (post basis : Bool) (recording : RawRecording) (process : RawRecording → Except Fault Dataset)
    (prepare : Except Fault Unit) (reference : Option String)
    (dataset : Dataset)
    (h : loadRoute m contract request refs positions expectedExecution post basis recording prepare reference process = .ok dataset) :
    rawReason m contract refs positions post recording = none ∧
      inputReason m contract expectedExecution basis dataset = none ∧ scopeReason m dataset.timing = none := by
  unfold loadRoute at h
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  cases h
  exact ⟨by assumption, by assumption, by assumption⟩

theorem loaded_route_is_same_participant_and_prospective (m : ModelScope) (contract : InputContract)
    (request : RawRequest) (refs : List AcquisitionReference) (positions expectedExecution : Value)
    (post basis : Bool) (recording : RawRecording) (process : RawRecording → Except Fault Dataset)
    (prepare : Except Fault Unit) (reference : Option String) (dataset : Dataset)
    (h : loadRoute m contract request refs positions expectedExecution post basis recording prepare reference process = .ok dataset) :
    recording.participant = m.participant ∧ recording.session ∉ m.calibrationSessions ∧
      m.available < recording.started := by
  have hr := (loaded_route_checks_actual_recording m contract request refs positions expectedExecution
    post basis recording process prepare reference dataset h).1
  have ha := (raw_success_iff m contract refs positions post recording).mp hr
  exact ⟨ha.1, ha.2.1, ha.2.2.1⟩

structure TrialCounts where
  scheduled : ℕ
  inputEpochs : ℕ
  evidence : ℕ
  preprocessingUnavailable : ℕ
  modelUnavailable : ℕ
  unavailable : ℕ

def trialCounts (scheduled inputEpochs evidence : ℕ) : Option TrialCounts :=
  if evidence ≤ inputEpochs ∧ inputEpochs ≤ scheduled then
    some ⟨scheduled, inputEpochs, evidence, scheduled - inputEpochs, inputEpochs - evidence,
      scheduled - evidence⟩
  else none

theorem accepted_trial_counts_partition (scheduled inputEpochs evidence : ℕ) (counts : TrialCounts)
    (h : trialCounts scheduled inputEpochs evidence = some counts) :
    counts.evidence + counts.unavailable = counts.scheduled ∧
      counts.preprocessingUnavailable + counts.modelUnavailable = counts.unavailable := by
  unfold trialCounts at h
  split_ifs at h with hb
  · have he := Option.some.inj h
    subst counts
    simp only
    omega

theorem no_evidence_keeps_complete_scheduled_denominator (scheduled : ℕ) :
    trialCounts scheduled 0 0 = some ⟨scheduled, 0, 0, scheduled, 0, scheduled⟩ := by
  simp [trialCounts]

inductive OfficialOutcome (P : Type*) where
  | evaluated (value : P)
  | abstainedNoEvidence (counts : TrialCounts)

/-- Source CLI catches the raw no-evidence exception as an abstention before
inference_record; other errors remain errors. A returned dataset is checked
again for session scope, and predict_logit repeats input validation. -/
def inferOutcome {P : Type*} (m : ModelScope) (contract : InputContract)
    (expectedExecution : Value) (basis : Bool) (scheduled : ℕ)
    (loaded : Except Fault Dataset) (predict : Dataset → Except Fault P) :
    Except Fault (OfficialOutcome P) :=
  match loaded with
  | .error .noUsableEvidence => .ok (.abstainedNoEvidence ⟨scheduled, 0, 0, scheduled, 0, scheduled⟩)
  | .error fault => .error fault
  | .ok dataset => match scopeReason m dataset.timing with
    | some fault => .error fault
    | none => match inputReason m contract expectedExecution basis dataset with
      | some fault => .error fault
      | none => match predict dataset with
        | .error fault => .error fault
        | .ok value => .ok (.evaluated value)

theorem official_no_evidence_abstains_without_prediction {P : Type*} (m : ModelScope)
    (contract : InputContract) (execution : Value) (basis : Bool) (scheduled : ℕ)
    (predict : Dataset → Except Fault P) :
    inferOutcome m contract execution basis scheduled (.error .noUsableEvidence) predict =
      .ok (.abstainedNoEvidence ⟨scheduled, 0, 0, scheduled, 0, scheduled⟩) := rfl

theorem actual_load_rejects_unauthorized_recording_before_any_no_evidence
    (m : ModelScope) (contract : InputContract) (request : RawRequest)
    (refs : List AcquisitionReference) (positions execution : Value) (post basis : Bool)
    (recording : RawRecording) (reference : Option String)
    (hrequest : requestReason request contract = none)
    (href : referenceReason contract reference = none)
    (hparticipant : recording.participant ≠ m.participant) :
    loadRoute m contract request refs positions execution post basis recording (.ok ()) reference
      (fun _ => .error .noUsableEvidence) = .error .participant := by
  exact raw_refusal_precedes_and_is_independent_of_processing m contract request refs positions execution
    post basis recording (fun _ => .error .noUsableEvidence) .participant reference hrequest href
      (wrong_participant_before_every_later_raw_guard _ _ _ _ _ _ hparticipant)

theorem scope_refusal_precedes_direct_prediction {P : Type*} (m : ModelScope)
    (contract : InputContract) (execution : Value) (basis : Bool) (scheduled : ℕ)
    (dataset : Dataset) (predict : Dataset → Except Fault P) (fault : Fault)
    (h : scopeReason m dataset.timing = some fault) :
    inferOutcome m contract execution basis scheduled (.ok dataset) predict = .error fault := by
  simp [inferOutcome, h]

end

end N2P3.PersonalAdmission
