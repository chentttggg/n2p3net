import N2P3.PersonalERPSession

/-!
# Raw selection, independent scope and frozen calibration/recovery gates

Source: brainsync.py:59-86,176-216,281-313,634-642;
continuous_qc.py:93-122; recovery.py:56-78. All membership, selection, authority,
time and certificate checks are explicit. Membership retains Python container
semantics, including substring membership for a malformed but source-accepted
string session container. Per-array numerical transforms are separate
operations and do not establish calibration-envelope coverage of raw signals.
-/

namespace N2P3.PersonalERPRawScope

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure IndependentScope where
  calibration : List String
  evaluation : List String

def constructIndependentScope (p : PersonalERPInput.Primitives) (calibration evaluation : Value) : Check IndependentScope := do
  let calibration ← tupleValue calibration
  let evaluation ← tupleValue evaluation
  let calibration ← calibration.mapM (fun value => text value "independent calibration session string")
  let evaluation ← evaluation.mapM (fun value => text value "independent evaluation session string")
  ensure (calibration ≠ [] ∧ (∀ id ∈ calibration, p.execution.strip id ≠ "") ∧ calibration.Nodup)
    "independent unique nonempty calibration sessions"
  ensure (evaluation ≠ [] ∧ (∀ id ∈ evaluation, p.execution.strip id ≠ "") ∧ evaluation.Nodup)
    "independent unique nonempty evaluation sessions"
  ensure (calibration.toFinset ∩ evaluation.toFinset=∅) "independent disjoint calibration evaluation sessions"
  pure ⟨calibration,evaluation⟩

def independentApplication (scope : IndependentScope) (session : String) (fitSessions : Value) : Check Unit := do
  ensure (session ∈ scope.calibration ++ scope.evaluation) "recording belongs to independent scope"
  let fit ← readHashableTuple fitSessions
  ensure (∀ id ∈ fit, ∃ saved ∈ scope.calibration, PersonalAdmission.PyEq id (.text saved))
    "raw fit independent calibration session subset"

structure Primitives where
  modal : ModalPrimitives
  contains : Value → Value → Check Bool
  transformBasisChunk : Fields → ℝ → ℕ → ℕ → ℕ → Check (ArrayImage Atom)
  concatenateRows : List (ArrayImage Atom) → Check (ArrayImage Atom)
  absolute : ArrayImage Atom → Check (ArrayImage Atom)
  sumAxisZero : ArrayImage Atom → Check (ArrayImage Atom)
  maximum : ArrayImage Atom → Check Atom

def checkRawFitScope (p : Primitives) (provenance : Fields) (session : String) (started : ℝ)
    (independent : Option IndependentScope) : Check Unit := do
  let sessions := (provenance.get "calibration_session_ids").getD (.sequence .nil)
  ensure (PyTruthy sessions) "raw fit explicit calibration sessions"
  let ended ← numeric (fieldDefault provenance "calibration_end_timestamp_s") "raw fit numeric finite calibration end"
  match independent with
  | some scope => independentApplication scope session sessions
  | none =>
    let reused ← p.contains sessions (.text session)
    if !reused then ensure (ended ≤ started) "new raw recording not earlier than calibration end"

def selectChannels (p : PersonalERPInput.Primitives) (bids : PersonalERPBids.Recording)
    (requested : Option (List String)) : Check (List String) := do
  let eeg := (bids.channels.filter (fun channel => decide (channel.kind="EEG"))).map PersonalERPBids.Channel.name
  let selected := requested.getD eeg
  ensure (selected ≠ []) "nonempty selected BrainSync EEG channels"
  let selectedKeys ← selected.mapM p.canonicalChannel
  let eegKeys ← eeg.mapM p.canonicalChannel
  ensure (selectedKeys ⊆ eegKeys) "model selected channels belong to measured EEG never EOG"
  let mapping := eegKeys.zip eeg
  let selected ← selectedKeys.mapM fun key => match (mapping.reverse.lookup key) with
    | none => .error (.fieldFailure "canonical selected EEG channel lookup")
    | some name => .ok name
  ensure (selected.Nodup) "unique selected raw EEG channels"
  pure selected

def projectionChunks (p : Primitives) (args : Fields) (rate : ℝ) (samples chunk : ℕ) :
    List ℕ → Check (List (ArrayImage Atom))
  | [] => .ok []
  | start :: tail => do
    let count := min chunk (samples-start)
    let transformed ← p.transformBasisChunk args rate samples start count
    let rest ← projectionChunks p args rate samples chunk tail
    pure (transformed :: rest)

def epochOperatorGain (p : Primitives) (args : Fields) (rate : ℝ) : Check Atom := do
  let _ ← checkPreprocessing p.modal.recovery.input.execution.strip p.modal.recovery.input.execution.floorSamples
    p.modal.recovery.input.execution.windowSamples (rawPreprocessing args)
  ensure (0 < rate) "finite positive projection source rate"
  let right ← p.modal.roundEndpoint (fieldDefault args "tmax_ms") rate
  let left ← p.modal.roundEndpoint (fieldDefault args "tmin_ms") rate
  let samples := right-left+1
  checkEmptyLinearEpoch p.modal.recovery args rate samples
  ensure (samples*8 ≤ 8388608) "bounded epoch projection source basis byte budget"
  let chunk := min samples.toNat (8388608 / (samples.toNat*8))
  ensure (0 < chunk) "positive bounded projection chunk"
  let starts := (List.range samples.toNat).filter (fun start => decide (start % chunk=0))
  let chunks ← projectionChunks p args rate samples.toNat chunk starts
  let projection ← p.concatenateRows chunks
  let absolute ← p.absolute projection
  let sums ← p.sumAxisZero absolute
  p.maximum sums

def gainUnderstated (bound : ℝ) : Atom → Check Bool
  | .real value => .ok (decide (bound < value))
  | .integer value => .ok (decide (bound < (value : ℝ)))
  | .boolean value => .ok (decide (bound < if value then (1 : ℝ) else 0))
  | .positiveInfinity => .ok true
  | .negativeInfinity | .nan => .ok false
  | _ => .error (.fieldFailure "executed epoch gain scalar comparison")

def checkRecoveryContract (p : Primitives) (model : ContinuousQC) (args : Fields)
    (sourceReference : String) (ocular : Option EOGModel) : Check Unit := do
  match model.epochRecovery with
  | some empirical =>
    let actual ← p.modal.recovery.input.execution.semanticHash (.object args)
    let expected ← p.modal.recovery.input.execution.semanticHash (.object empirical.preprocessing)
    ensure (actual=expected) "empirical recovery asdict epoch operator identity"
  | none =>
    ensure (fieldDefault args "filter_phase" ≠ .text "zero") "zero phase raw QC requires fitted empirical boundary recovery"
    match model.recovery with
    | none => pure ()
    | some box =>
      ensure (fieldDefault args "filter_method"=.text "iir" ∧ fieldDefault args "filter_phase"=.text "forward")
        "SOS certificate forward IIR execution"
      let expected ← expectedRecoverySOS p.modal.recovery args model.rate false
      ensure (box.sos.shape=expected.shape) "SOS certificate executed coefficient geometry"
      let close ← p.modal.recovery.arrayAllclose box.sos expected (1/1000000000000) 0
      ensure (close=true) "SOS certificate executed coefficient agreement"
      let gain ← epochOperatorGain p args model.rate
      let understated ← gainUnderstated box.epochGain gain
      ensure (understated=false) "SOS certificate does not understate epoch gain"
      let reference := match ocular with | none => sourceReference | some ocular => ocular.sourceReference
      let digest ← match ocular with
        | none => .ok Value.null
        | some ocular => do
          let hash ← p.modal.recovery.input.execution.semanticHash (.object (eogRecord ocular))
          pure (.text hash)
      ensure (PersonalAdmission.PyEq box.inputReference (.text reference) ∧ PersonalAdmission.PyEq box.ocularDigest digest)
        "SOS recovery actual reference ocular identity"

def checkQCBeforeDetection (p : Primitives) (model : ContinuousQC) (args : Fields)
    (recording : PersonalERPSession.Recording) (ocular : Option EOGModel) (scope : Option IndependentScope) : Check Unit := do
  checkRawFitScope p model.provenance recording.session recording.started scope
  ensure (fieldDefault model.provenance "source_reference"=.text recording.bids.reference)
    "raw QC native acquisition reference"
  let participant ← recordingAuthority p.modal.recovery.input recording.subject
  ensure (fieldDefault model.provenance "participant_key"=.text participant) "raw QC authoritative participant"
  checkRecoveryContract p model args recording.bids.reference ocular

def checkOcularBeforeApplication (p : Primitives) (model : EOGModel)
    (recording : PersonalERPSession.Recording) (scope : Option IndependentScope) : Check Unit := do
  let participant ← recordingAuthority p.modal.recovery.input recording.subject
  ensure (model.participant=participant) "raw ocular authoritative participant"
  checkRawFitScope p model.fitProvenance recording.session recording.started scope

theorem raw_fit_time_gate_allows_equality (time : ℝ) :
    ensure (time ≤ time) "new raw recording not earlier than calibration end"=.ok () := by simp [ensure]

theorem raw_fit_time_gate_rejects_future_fit (started ended : ℝ) (h : started < ended) :
    ensure (ended ≤ started) "new raw recording not earlier than calibration end"=
      .error (.sourceReject "new raw recording not earlier than calibration end") := by simp [ensure,not_le.mpr h]

theorem epoch_recovery_branch_does_not_claim_reference_binding (p : Primitives) (model : ContinuousQC)
    (empirical : EpochRecovery) (args : Fields) (first second : String) (ocularA ocularB : Option EOGModel)
    (h : model.epochRecovery=some empirical) :
    checkRecoveryContract p model args first ocularA=checkRecoveryContract p model args second ocularB := by
  simp [checkRecoveryContract,h]

end

end N2P3.PersonalERPRawScope
