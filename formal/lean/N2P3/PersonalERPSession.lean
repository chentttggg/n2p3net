import N2P3.PersonalERPClock

/-!
# Target-free BrainSync recording/session ingress

Source: brainsync_contract.py:150-303,311-497. This expands the actual
require_target=False branch, including its second sidecar/table reads.
ISO parsing, timestamp conversion and path/IO operations remain distinct
standard-library boundaries. Target-label fields are never accessed by this
path. No recording-validity primitive or default recording is provided.
This is P06 work in progress.
-/

namespace N2P3.PersonalERPSession

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure ISOImage where
  timezone : Bool
  epochSeconds : Check Atom

structure Primitives where
  clock : PersonalERPClock.Primitives
  rootPath : String → Check String
  fileExists : String → Check Bool
  join : String → String → String
  isoParse : String → Check ISOImage

def nonempty (p : Primitives) (value : Value) : Check String := do
  let value ← text value "BrainSync nonempty string"
  let value := p.clock.bids.models.input.execution.strip value
  ensure (value ≠ "") "BrainSync nonempty stripped string"
  pure value

def timestamp (p : Primitives) (value : Value) : Check (String × ℝ) := do
  let value ← nonempty p value
  let parsed ← p.isoParse (value.replace "Z" "+00:00")
  ensure (parsed.timezone=true) "BrainSync ISO timestamp has timezone"
  let atom ← parsed.epochSeconds
  let seconds ← strictReal atom
  pure (value,seconds)

def positiveInteger (value : Value) : Check ℤ := do
  let value ← integer value "BrainSync positive nonbool integer"
  ensure (0 < value) "BrainSync positive integer"
  pure value

structure Protocol where
  blocks : ℤ
  perBlock : ℤ
  repetitions : ℤ

def protocol (experiment : Fields) : Check Protocol := do
  ensure (fieldDefault experiment "sequence_policy"=.text "balanced_random_permutation_cycles")
    "BrainSync balanced random permutation sequence policy"
  let blocks ← positiveInteger (fieldDefault experiment "blocks")
  let perBlock ← positiveInteger (fieldDefault experiment "digits_per_block")
  ensure (perBlock % 9=0) "BrainSync complete nine candidate cycles"
  let repetitions ← positiveInteger (fieldDefault experiment "repetitions_per_digit")
  ensure (repetitions=perBlock/9) "BrainSync repetitions agree with digits per block"
  pure ⟨blocks,perBlock,repetitions⟩

def checkBalanced (experiment : Fields) (bids : PersonalERPBids.Recording) : Check Unit := do
  let declared ← protocol experiment
  ensure ((bids.events.stimuli.length : ℤ)=declared.blocks*declared.perBlock) "BrainSync scheduled stimulus count"
  ensure (∀ block ∈ List.range declared.blocks.toNat,
    let events := bids.events.stimuli.filter (fun event => decide (event.block=(block : ℤ)+1))
    events.map PersonalERPBids.Stimulus.index=(List.range declared.perBlock.toNat).map (fun i => (i : ℤ)+1))
    "BrainSync consecutive per block trial indices"
  ensure (∀ block ∈ List.range declared.blocks.toNat, ∀ candidate ∈ PersonalERPBids.digits,
    ((bids.events.stimuli.filter (fun event => decide (event.block=(block : ℤ)+1 ∧ event.candidate=candidate))).length : ℤ)=
      declared.repetitions) "BrainSync exact per candidate block counts"
  ensure (∀ block ∈ List.range declared.blocks.toNat, ∀ cycle ∈ List.range declared.repetitions.toNat,
    let events := bids.events.stimuli.filter (fun event => decide (event.block=(block : ℤ)+1))
    ((events.drop (cycle*9)).take 9).map PersonalERPBids.Stimulus.candidate |>.toFinset = PersonalERPBids.digits.toFinset)
    "BrainSync every cycle is candidate permutation"

def referenceRecord : Fields := Fields.ofList [("schema",.text "brainsync-acquisition-reference/1"),
  ("label",.text "left earlobe"),("side",.text "left"),("site",.text "earlobe"),("electrode",.text "ear_clip")]

def checkReferenceRows (p : Primitives) (reference : String) : List PersonalERPBids.CSVRow → Check Unit
  | [] => .ok ()
  | row :: tail => do
    let kind ← required row.fields "type"
    let kind ← text kind "reread channel type string"
    if p.clock.bids.upper kind="EEG" then
      ensure ((row.fields.get "reference").getD (.text reference)=.text reference) "BrainSync common acquisition reference"
    checkReferenceRows p reference tail

def checkReference (p : Primitives) (manifest : Fields) (bids : PersonalERPBids.Recording) : Check Unit := do
  let montage ← objectAt ((manifest.get "montage").getD (.object .nil)) "BrainSync montage object"
  let declared := fieldDefault montage "ref_label"
  ensure (declared=.null ∨ declared=.text bids.reference) "BrainSync montage BIDS reference label"
  let details := (montage.get "reference_details").getD (.object .nil)
  ensure (PersonalAdmission.PyEq details (.object bids.referenceDetails)) "BrainSync cross record reference details"
  ensure (¬PyTruthy details ∨ PersonalAdmission.PyEq details (.object referenceRecord)) "BrainSync left earlobe details"
  let path ← PersonalERPBids.getPath bids.paths "channels_tsv"
  let reread ← p.clock.bids.tsv path
  checkReferenceRows p bids.reference reread.rows

def nativeNames : List String := ["Fz","Cz","P3","Pz","P4","PO8","EOG","Oz"]
def nativeTypes : List String := ["EEG","EEG","EEG","EEG","EEG","EEG","EOG","EEG"]

def checkNativeLayout (p : Primitives) (bids : PersonalERPBids.Recording) : Check Unit := do
  ensure (bids.channels.length=8) "BrainSync eight acquisition channels"
  let actual ← (bids.channels.map PersonalERPBids.Channel.name).mapM p.clock.bids.models.input.canonicalChannel
  let expected ← nativeNames.mapM p.clock.bids.models.input.canonicalChannel
  ensure (actual=expected ∧ bids.channels.map PersonalERPBids.Channel.kind=nativeTypes)
    "BrainSync seven EEG independent EOG native order"

def readRests (p : Primitives) : List Value → Check (List (ℝ × ℝ))
  | [] => .ok []
  | value :: tail => do
    let row ← objectAt value "BrainSync timeline rest segment mapping"
    let start ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault row "start_seconds") (some 0)
    let stop ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault row "end_seconds") (some 0)
    let duration ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault row "duration_seconds") (some 0)
    let actual ← p.clock.bids.sub stop start
    ensure (start < stop ∧ p.clock.bids.close actual duration (1/1000000)=true) "BrainSync rest bounds duration"
    let rest ← readRests p tail
    pure ((start,stop)::rest)

def checkRests (p : Primitives) (timeline : Fields) (bids : PersonalERPBids.Recording) : Check Unit := do
  let rows ← listAt (fieldDefault timeline "rest_segments") "BrainSync timeline rest array"
  let actual ← readRests p rows
  let expected := bids.events.rests.map (fun rest => (rest.onset,rest.stop))
  ensure (actual.length=expected.length) "BrainSync BIDS manifest rest count"
  ensure (∀ pair ∈ actual.zip expected,
    p.clock.bids.close pair.1.1 pair.2.1 (1/1000000)=true ∧
    p.clock.bids.close pair.1.2 pair.2.2 (1/1000000)=true) "BrainSync BIDS manifest rest endpoints"

structure Pause where
  fields : Fields
  start : ℝ
  stop : ℝ
  sample : ℤ

structure PauseState where
  previousWallEnd : ℝ := 0
  previousSample : ℤ := -1
  boundaries : List ℝ := []
  pauses : List Pause := []

def readPauses (p : Primitives) (rate : ℝ) : List Value → PauseState → Check PauseState
  | [],state => .ok state
  | value :: tail,state => do
    let fields ← objectAt value "BrainSync acquisition pause mapping"
    let start ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault fields "start_wall_seconds") (some 0)
    let stop ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault fields "end_wall_seconds") (some 0)
    let sample ← integer (fieldDefault fields "boundary_sample") "BrainSync exact integer pause sample"
    ensure (0 ≤ sample ∧ state.previousSample ≤ sample ∧ state.previousWallEnd ≤ start ∧ start ≤ stop)
      "BrainSync ordered pause wall intervals sample boundaries"
    let boundaries ← if sample=state.previousSample then pure state.boundaries else do
      let time ← p.clock.bids.divide sample rate
      pure (state.boundaries ++ [time])
    let pause : Pause := ⟨fields,start,stop,sample⟩
    readPauses p rate tail ⟨stop,sample,boundaries,state.pauses ++ [pause]⟩

def checkPauses (p : Primitives) (timeline recording : Fields) (bids : PersonalERPBids.Recording)
    (clock : PersonalERPClock.Result) : Check (List Value) := do
  let values ← listAt ((timeline.get "acquisition_pauses").getD (.sequence .nil)) "BrainSync acquisition pause array"
  let state ← readPauses p bids.rate values {}
  ensure (state.boundaries.length=bids.events.boundaries.length ∧
    ∀ pair ∈ state.boundaries.zip bids.events.boundaries,
      p.clock.bids.close pair.1 pair.2 (1/100000000)=true) "BrainSync pause BIDS boundary agreement"
  if fieldDefault recording "clock" ≠ .null then
    let counts ← match clock.counts with
      | none => .error (.fieldFailure "present recording clock acquired count")
      | some counts => .ok counts
    ensure (∀ pause ∈ state.pauses, pause.sample ≤ counts.acquired) "BrainSync pause boundary within acquired samples"
    let actual := (clock.segments.drop 1).map PersonalERPClock.Segment.start
    let expected := ((state.pauses.filter (fun pause => decide (0 < pause.sample ∧ pause.sample < counts.acquired))).map Pause.sample).dedup.mergeSort (· ≤ ·)
    ensure (actual=expected) "BrainSync clock pause acquisition segment boundaries"
  let segmented := decide (fieldDefault timeline "time_base"=.text "segmented_recording")
  ensure (segmented= !values.isEmpty ∧ (segmented=true → fieldDefault timeline "pause_samples_retained"=.boolean false))
    "BrainSync segmented time base omits pause samples"
  let sidecarPath ← PersonalERPBids.getPath bids.paths "eeg_json"
  let sidecar ← PersonalERPBids.readObject p.clock.bids sidecarPath
  if segmented then
    ensure (fieldDefault sidecar "RecordingType"=.text "discontinuous" ∧
      PersonalAdmission.PyEq (fieldDefault sidecar "AcquisitionPauses") (.sequence (valuesOfList values)))
      "BrainSync reread BIDS segmented timeline agreement"
  pure values

structure Recording where
  root : String
  manifest : Fields
  session : String
  subject : String
  startedText : String
  started : ℝ
  endedText : String
  ended : ℝ
  age : Option ℝ
  sex : Value
  bids : PersonalERPBids.Recording
  clock : PersonalERPClock.Result

def readChronology (p : Primitives) (manifest : Fields) : Check ((String × ℝ) × (String × ℝ)) := do
  let started ← timestamp p (fieldDefault manifest "started_utc")
  let ended ← timestamp p (fieldDefault manifest "ended_utc")
  ensure (started.2 ≤ ended.2) "BrainSync start no later than end"
  pure (started,ended)

def validateManifest (p : Primitives) (root : String) (manifest : Fields) : Check Recording := do
  ensure (fieldDefault manifest "schema"=.text "brainsync-gtn-session/4") "BrainSync session schema four"
  ensure (fieldDefault manifest "status"=.text "completed") "BrainSync completed session input status"
  let recording ← objectAt (fieldDefault manifest "recording") "BrainSync recording object"
  ensure (fieldDefault recording "stage"=.text "bids_raw") "BrainSync BIDS raw stage"
  ensure (fieldDefault recording "preprocessing_status"=.text "pending") "BrainSync pending raw preprocessing status"
  let timeline ← objectAt (fieldDefault recording "timeline") "BrainSync timeline object"
  ensure (fieldDefault timeline "status"=.text "finalized") "BrainSync finalized timeline"
  ensure (fieldDefault timeline "time_base"=.text "continuous_recording" ∨
    fieldDefault timeline "time_base"=.text "segmented_recording") "BrainSync supported raw time base"
  ensure (fieldDefault timeline "rest_segments_retained"=.boolean true) "BrainSync strict retained rest segments"
  let files ← objectAt (fieldDefault recording "bids") "BrainSync BIDS file manifest"
  let continuity := match fieldDefault manifest "quality" with
    | .object quality => fieldDefault quality "eeg_continuity"
    | _ => .null
  let continuity ← objectAt continuity "BrainSync EEG continuity quality mapping"
  ensure (fieldDefault continuity "passed"=.boolean true) "BrainSync EEG continuity passed before preprocessing"
  let experiment ← objectAt (fieldDefault manifest "experiment") "BrainSync experiment object"
  let subject ← nonempty p (fieldDefault experiment "subject_id")
  let _ ← protocol experiment
  let session ← nonempty p (fieldDefault manifest "session_id")
  let chronology ← readChronology p manifest
  let bids ← PersonalERPBids.validate p.clock.bids root files
  checkReference p manifest bids
  checkBalanced experiment bids
  let recordingPath ← text (fieldDefault recording "path") "BrainSync recording path string"
  let recordingPath ← p.clock.bids.path root recordingPath
  let rawPath ← PersonalERPBids.getPath bids.paths "recording"
  ensure (recordingPath.resolved=rawPath) "BrainSync recording path identifies BIDS raw"
  checkNativeLayout p bids
  let recordingRate ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault recording "source_sample_rate_hz") (some 0)
  let clock ← PersonalERPClock.validate p.clock root recording bids.rate bids.duration
  let timelineRate ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault timeline "source_sample_rate_hz") (some 0)
  let timelineDuration ← PersonalERPBids.finiteNumberAt p.clock.bids (fieldDefault timeline "output_duration_seconds") (some 0)
  ensure (p.clock.bids.close recordingRate bids.rate (1/1000000)=true ∧
    p.clock.bids.close timelineRate bids.rate (1/1000000)=true) "BrainSync manifest BIDS sample rate consistency"
  ensure (p.clock.bids.close timelineDuration bids.duration (1/1000000)=true) "BrainSync manifest BIDS duration consistency"
  checkRests p timeline bids
  let _ ← checkPauses p timeline recording bids clock
  let lifetime ← p.clock.bids.sub chronology.2.2 chronology.1.2
  ensure (bids.duration ≤ lifetime) "BrainSync recording duration within session lifetime"
  let ageValue := fieldDefault experiment "age"
  let age ← if ageValue=.null ∨ ageValue=.text "" then pure none else do
    let age ← PersonalERPBids.finiteNumberAt p.clock.bids ageValue (some 0)
    ensure (age ≠ 0) "BrainSync optional age positive when declared"
    pure (some age)
  pure ⟨root,manifest,session,subject,chronology.1.1,chronology.1.2,chronology.2.1,chronology.2.2,
    age,fieldDefault experiment "sex",bids,clock⟩

def validate (p : Primitives) (directory : String) : Check Recording := do
  let root ← p.rootPath directory
  let pending ← p.fileExists (p.join root ".target-confirmation.pending.json")
  ensure (pending=false) "BrainSync target confirmation commit not pending"
  let manifest ← PersonalERPBids.readObject p.clock.bids (p.join root "session.json")
  validateManifest p root manifest

def callbackFields (recording : Recording) : RecordingCallbackFields :=
  ⟨recording.subject,recording.session,.finite recording.started,recording.manifest,
   recording.bids.reference,recording.bids.referenceDetails⟩

theorem chronology_success_requires_start_no_later_than_end (p : Primitives) (manifest : Fields)
    (chronology : (String × ℝ) × (String × ℝ)) (h : readChronology p manifest=.ok chronology) :
    chronology.1.2 ≤ chronology.2.2 := by
  unfold readChronology at h
  rcases (bind_success _ _ _).mp h with ⟨started,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨ended,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hg,h⟩
  have he : (started,ended)=chronology := Except.ok.inj h
  rw [←he]
  exact ensure_success _ _ completion hg

theorem pending_commit_refused_before_manifest_io (p : Primitives) (directory root : String)
    (hr : p.rootPath directory=.ok root)
    (hp : p.fileExists (p.join root ".target-confirmation.pending.json")=.ok true) :
    validate p directory=.error (.sourceReject "BrainSync target confirmation commit not pending") := by
  simp [validate,hr,hp,ensure]

end

end N2P3.PersonalERPSession
