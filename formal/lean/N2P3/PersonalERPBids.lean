import N2P3.PersonalERPModels

/-!
# Target-free BrainSync BIDS field validation

Source: bids_eeg.py:107-197,228-482; the fixed contract is
brainsync_contract.py:36-49. IO, CSV parsing, pathlib operations and each scalar
floating operation are named boundaries. File containment, table headers and
rows, reference/ocular declarations, geometry and all event/schedule guards
are executable field checks; there is no whole BIDS-validity primitive.
This module is P06 WIP and does not establish physical storage authenticity.
-/

namespace N2P3.PersonalERPBids

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure PathImage where
  absolute : Bool
  parts : List String
  resolved : String
  contained : Bool
  file : Bool
  suffix : String

structure CSVRow where
  fields : Fields
  extraUnnamed : Bool
  missingCell : Bool

structure TSV where
  header : List String
  rows : List CSVRow

structure Primitives where
  models : RecoveryPrimitives
  path : String → String → Check PathImage
  json : String → Check Value
  tsv : String → Check TSV
  casefold : String → String
  upper : String → String
  isDigits : String → Bool
  decimal : String → Check ℤ
  integerText : ℤ → String
  close : ℝ → ℝ → ℝ → Bool
  add : ℝ → ℝ → Check ℝ
  sub : ℝ → ℝ → Check ℝ
  multiply : ℝ → ℝ → Check ℝ
  divide : ℝ → ℝ → Check ℝ
  squareRoot : ℝ → Check ℝ

def pathKeys : List String := ["dataset_description","recording","eeg_json","channels_tsv",
  "events_tsv","electrodes_tsv","coordsystem_json"]

def containedPaths (p : Primitives) (root : String) (files : Fields) : List String → Check (List (String × String))
  | [] => .ok []
  | key :: tail => do
    let value ← required files key
    let relative ← text value "BIDS path string"
    ensure (p.models.input.execution.strip relative ≠ "") "nonempty BIDS relative path"
    let path ← p.path root relative
    ensure (path.absolute=false ∧ ".." ∉ path.parts) "BIDS lexical relative containment"
    ensure (path.contained=true) "BIDS resolved path containment"
    ensure (path.file=true) "BIDS contained file exists"
    let rest ← containedPaths p root files tail
    pure ((key,path.resolved) :: rest)

def getPath (paths : List (String × String)) (key : String) : Check String :=
  match paths.lookup key with
  | none => .error (.fieldFailure "missing contained BIDS path")
  | some value => .ok value

def readObject (p : Primitives) (path : String) : Check Fields := do
  let value ← p.json path
  objectAt value "BIDS readable JSON object"

def checkRows : List CSVRow → Check (List Fields)
  | [] => .ok []
  | row :: tail => do
    ensure (row.extraUnnamed=false ∧ row.missingCell=false) "TSV row agrees with header width"
    let rest ← checkRows tail
    pure (row.fields :: rest)

def readTable (p : Primitives) (path : String) : Check (List String × List Fields) := do
  let parsed ← p.tsv path
  ensure (parsed.header ≠ [] ∧ ∀ field ∈ parsed.header, field ≠ "") "nonempty valid TSV header"
  ensure (parsed.header.Nodup) "unique TSV header columns"
  let rows ← checkRows parsed.rows
  ensure (rows ≠ []) "nonempty TSV rows"
  pure (parsed.header,rows)

def rowText (p : Primitives) (row : Fields) (key : String) : Check String := do
  let value ← required row key
  let value ← text value "TSV text cell"
  pure (p.models.input.execution.strip value)

def finiteNumberAt (p : Primitives) (value : Value) (minimum : Option ℝ) : Check ℝ := do
  ensure (∀ b, value ≠ .boolean b) "BIDS nonbool numeric conversion"
  let atom ← p.models.floatScalar value
  let number ← strictReal atom
  ensure (∀ bound ∈ minimum, bound ≤ number) "BIDS finite number minimum"
  pure number

def integerCell (p : Primitives) (row : Fields) (key : String) (minimum : ℤ) : Check ℤ := do
  let cell ← required row key
  let cell ← p.models.input.pythonString cell
  let cell := p.models.input.execution.strip cell
  let value ← p.decimal cell
  ensure (p.integerText value=cell ∧ minimum ≤ value) "canonical integer TSV text and minimum"
  pure value

def readVersion (p : Primitives) (value : Value) : Check (ℤ × ℤ × ℤ) := do
  let version ← text value "BIDS semantic version string"
  let parts := (p.models.input.execution.strip version).splitOn "."
  ensure (parts.length=3 ∧ ∀ part ∈ parts, p.isDigits part=true) "BIDS three digit version components"
  let values ← parts.mapM p.decimal
  match values with
  | [major,minor,patch] => pure (major,minor,patch)
  | _ => .error (.fieldFailure "BIDS version component geometry")

structure Channel where
  name : String
  kind : String
  status : String
  row : Fields

def channelFields (p : Primitives) : List Fields → Check (List Channel)
  | [] => .ok []
  | row :: tail => do
    let name ← rowText p row "name"
    let kind ← rowText p row "type"
    let status ← rowText p row "status"
    let rest ← channelFields p tail
    pure (⟨name,p.upper kind,p.casefold status,row⟩ :: rest)

def checkChannelRates (p : Primitives) (rate : ℝ) : List Channel → Check Unit
  | [] => .ok ()
  | channel :: tail => do
    let value ← required channel.row "sampling_frequency"
    let rowRate ← finiteNumberAt p value (some (1/1000000000000))
    ensure (p.close rowRate rate (1/1000000)=true) "BIDS channel sidecar sampling rates agree"
    checkChannelRates p rate tail

def electrodeNames (p : Primitives) : List Fields → Check (List (String × Fields))
  | [] => .ok []
  | row :: tail => do
    let name ← rowText p row "name"
    let rest ← electrodeNames p tail
    pure ((name,row)::rest)

def coordinateCell (p : Primitives) (row : Fields) (key : String) : Check ℝ := do
  let value ← required row key
  finiteNumberAt p value none

def readPositions (p : Primitives) (electrodes : List (String × Fields)) :
    List Channel → Check (List (Option (ℝ × ℝ × ℝ)))
  | [] => .ok []
  | channel :: tail => do
    let row := electrodes.lookup channel.name
    let position ← if channel.kind ≠ "EEG" then do
      match row with
      | none => .ok none
      | some fields => do
        let x ← rowText p fields "x"
        let y ← rowText p fields "y"
        let z ← rowText p fields "z"
        if x="n/a" ∧ y="n/a" ∧ z="n/a" then pure none
        else do
          let x ← coordinateCell p fields "x"
          let y ← coordinateCell p fields "y"
          let z ← coordinateCell p fields "z"
          pure (some (x,y,z))
      else do
        let fields ← match row with
          | none => .error (.sourceReject "EEG channel electrode coordinates present")
          | some fields => .ok fields
        let x ← coordinateCell p fields "x"
        let y ← coordinateCell p fields "y"
        let z ← coordinateCell p fields "z"
        pure (some (x,y,z))
    match position with
    | none => pure ()
    | some (x,y,z) => do
      let xx ← p.multiply x x
      let yy ← p.multiply y y
      let zz ← p.multiply z z
      let xy ← p.add xx yy
      let sum ← p.add xy zz
      let norm ← p.squareRoot sum
      ensure (0 < norm ∧ max (abs x) (max (abs y) (abs z)) ≤ 1/2) "nonzero metre head coordinates"
    let rest ← readPositions p electrodes tail
    pure (position :: rest)

structure Stimulus where
  onset : ℝ
  duration : ℝ
  sample : ℤ
  candidate : String
  trial : String
  block : ℤ
  index : ℤ
  rowNumber : ℕ

structure Interval where
  onset : ℝ
  duration : ℝ
  sample : ℤ
  id : Value
  block : ℤ
  rowNumber : ℕ
  stop : ℝ

structure EventState where
  previousOnset : Option ℝ := none
  stimuli : List Stimulus := []
  rests : List Interval := []
  boundaries : List ℝ := []
  excluded : List Interval := []
  trialIds : List String := []
  scheduleIds : List (ℤ × ℤ) := []
  previousSchedule : Option (ℤ × ℤ) := none
  restIds : List String := []

def digits : List String := ["1","2","3","4","5","6","7","8","9"]

def readEvents (p : Primitives) (rate duration : ℝ) : List Fields → ℕ → EventState → Check EventState
  | [],_,state => .ok state
  | row :: tail,rowNumber,state => do
    let onsetValue ← required row "onset"
    let onset ← finiteNumberAt p onsetValue (some 0)
    let durationValue ← required row "duration"
    let width ← finiteNumberAt p durationValue (some 0)
    let sample ← integerCell p row "sample" 0
    ensure (∀ previous ∈ state.previousOnset, previous ≤ onset) "chronological BIDS event rows"
    ensure (onset < duration) "BIDS event onset inside duration"
    let stop ← p.add onset width
    let totalTolerance ← p.add duration (1/1000000000)
    ensure (stop ≤ totalTolerance) "BIDS event interval inside duration"
    let sampleTime ← p.divide sample rate
    let error ← p.sub sampleTime onset
    let halfSample ← p.divide (1/2) rate
    let tolerance ← p.add halfSample (1/1000000000)
    ensure (abs error ≤ tolerance) "BIDS onset sample half sample agreement"
    let kind ← rowText p row "trial_type"
    let block ← integerCell p row "block_id" 1
    let state := {state with previousOnset := some onset}
    let state ← if kind="stimulus" then do
      let trial ← rowText p row "trial_id"
      let index ← integerCell p row "trial_index" 1
      let candidate ← rowText p row "digit"
      ensure (trial ≠ "" ∧ trial ≠ "n/a" ∧ candidate ∈ digits) "BIDS stimulus trial candidate identity"
      ensure (trial ∉ state.trialIds ∧ (block,index) ∉ state.scheduleIds) "unique BIDS stimulus trial schedule"
      ensure (∀ previous ∈ state.previousSchedule, previous < (block,index)) "strict BIDS schedule order"
      let event : Stimulus := ⟨onset,width,sample,candidate,trial,block,index,rowNumber⟩
      pure { state with
        stimuli := state.stimuli ++ [event]
        trialIds := state.trialIds ++ [trial]
        scheduleIds := state.scheduleIds ++ [(block,index)]
        previousSchedule := some (block,index) }
      else if kind="acquisition_boundary" then do
        ensure (width=0 ∧ onset ∉ state.boundaries) "unique zero duration acquisition boundaries"
        pure {state with boundaries := state.boundaries ++ [onset]}
      else if kind="interrupted_stimulus" then do
        ensure (0 < width) "positive excluded event interval"
        let trial ← required row "trial_id"
        pure {state with excluded := state.excluded ++ [⟨onset,width,sample,trial,block,rowNumber,stop⟩]}
      else if kind="rest" then do
        let id ← rowText p row "rest_segment_id"
        ensure (id ≠ "" ∧ id ≠ "n/a" ∧ id ∉ state.restIds ∧ 0 < width) "unique positive BIDS rest intervals"
        pure { state with
          rests := state.rests ++ [⟨onset,width,sample,.text id,block,rowNumber,stop⟩]
          restIds := state.restIds ++ [id] }
      else .error (.sourceReject "supported BIDS event kind")
    readEvents p rate duration tail (rowNumber+1) state

structure Recording where
  root : String
  paths : List (String × String)
  sidecar : Fields
  channels : List Channel
  positions : List (Option (ℝ × ℝ × ℝ))
  rate : ℝ
  duration : ℝ
  reference : String
  referenceDetails : Fields
  eogDerivation : Value
  events : EventState

def validate (p : Primitives) (root : String) (files : Fields) : Check Recording := do
  ensure (pathKeys ⊆ fieldKeys files) "BIDS required file manifest keys"
  let paths ← containedPaths p root files pathKeys
  let rawPath ← getPath paths "recording"
  let rawImage ← p.path root rawPath
  ensure (p.casefold rawImage.suffix ∈ [".edf",".bdf",".vhdr",".set"]) "supported BIDS raw extension"
  let descriptionPath ← getPath paths "dataset_description"
  let description ← readObject p descriptionPath
  let version ← readVersion p (fieldDefault description "BIDSVersion")
  ensure (version.1=1 ∧ (1,11,0) ≤ version) "minimum BIDS version one eleven zero"
  ensure ((description.get "DatasetType").getD (.text "raw")=.text "raw") "BIDS raw dataset type"
  let sidecarPath ← getPath paths "eeg_json"
  let sidecar ← readObject p sidecarPath
  ensure (fieldDefault sidecar "TaskName"=.text "gtn") "BIDS gtn task"
  let rate ← finiteNumberAt p (fieldDefault sidecar "SamplingFrequency") (some (1/1000000000000))
  let duration ← finiteNumberAt p (fieldDefault sidecar "RecordingDuration") (some (1/1000000000000))
  let reference ← text (fieldDefault sidecar "EEGReference") "BIDS reference string"
  let reference := p.models.input.execution.strip reference
  ensure (reference ≠ "") "BIDS explicit nonempty reference"
  let details ← objectAt ((sidecar.get "EEGReferenceDetails").getD (.object .nil)) "BIDS reference details dictionary"
  ensure (details=.nil ∨ fieldDefault details "label"=.text reference) "BIDS reference label details agreement"
  let channelPath ← getPath paths "channels_tsv"
  let table ← readTable p channelPath
  ensure (["name","type","units","sampling_frequency","status"] ⊆ table.1) "BIDS required channel columns"
  let channels ← channelFields p table.2
  ensure (∀ channel ∈ channels, channel.name ≠ "") "nonempty BIDS channel names"
  ensure ((channels.map (fun channel => p.casefold channel.name)).Nodup) "casefold unique BIDS channel names"
  let eog := channels.filter (fun channel => decide (channel.kind="EOG"))
  let derivation := (sidecar.get "EOGDerivation").getD (.object .nil)
  checkEOGDerivation p.models.input derivation (eog.map (fun channel => .text channel.name)) (.text reference) false
  ensure (∀ channel ∈ channels, channel.kind ∈ ["EEG","EOG"]) "BIDS allowed EEG EOG types"
  ensure (∃ channel ∈ channels, channel.kind="EEG") "BIDS contains EEG channel"
  checkChannelRates p rate channels
  ensure (∀ channel ∈ channels, channel.status ∈ ["good","bad"]) "BIDS channel status good or bad"
  let coordinatePath ← getPath paths "coordsystem_json"
  let coordinates ← readObject p coordinatePath
  ensure (fieldDefault coordinates "EEGCoordinateUnits"=.text "m") "BIDS metre coordinate units"
  let system ← text (fieldDefault coordinates "EEGCoordinateSystem") "BIDS coordinate system string"
  ensure (p.models.input.execution.strip system ≠ "") "BIDS declared coordinate system"
  if system="Other" then
    let description ← p.models.input.pythonString ((coordinates.get "EEGCoordinateSystemDescription").getD (.text ""))
    ensure (p.models.input.execution.strip description ≠ "") "BIDS Other coordinate system description"
  let electrodePath ← getPath paths "electrodes_tsv"
  let electrodes ← readTable p electrodePath
  ensure (["name","x","y","z"] ⊆ electrodes.1) "BIDS required electrode columns"
  let electrodes ← electrodeNames p electrodes.2
  ensure ((electrodes.map Prod.fst).Nodup ∧ (electrodes.map Prod.fst) ⊆ channels.map Channel.name)
    "unique electrode names identify recorded channels"
  let positions ← readPositions p electrodes channels
  let eventPath ← getPath paths "events_tsv"
  let events ← readTable p eventPath
  ensure (["onset","duration","sample","trial_type","trial_id","block_id","trial_index","digit","rest_segment_id"] ⊆ events.1)
    "BIDS required event columns"
  let events ← readEvents p rate duration events.2 2 {}
  ensure (events.stimuli ≠ []) "BIDS nonempty stimuli"
  let rests := events.rests.mergeSort (fun a b => a.onset ≤ b.onset)
  ensure (rests.Pairwise (fun a b => a.stop ≤ b.onset)) "nonoverlapping BIDS rest intervals"
  ensure (∀ stimulus ∈ events.stimuli, ∀ rest ∈ rests, ¬(rest.onset ≤ stimulus.onset ∧ stimulus.onset < rest.stop))
    "BIDS stimulus outside rest intervals"
  pure ⟨root,paths,sidecar,channels,positions,rate,duration,reference,details,derivation,{events with rests:=rests}⟩

theorem bool_is_not_BIDS_numeric (p : Primitives) (b : Bool) (minimum : Option ℝ) :
    finiteNumberAt p (.boolean b) minimum=.error (.sourceReject "BIDS nonbool numeric conversion") := by
  simp [finiteNumberAt,ensure]

theorem equal_schedule_is_not_strictly_later (block index : ℤ) : ¬(block,index) < (block,index) := lt_irrefl _

end

end N2P3.PersonalERPBids
