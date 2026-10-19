import N2P3.PersonalERPRawScope
import N2P3.PersonalERPGeometry

/-!
# Native raw ingress, actual EOG application and dependency construction

Source: brainsync.py:176-340; ocular.py:43-76,155-201;
continuous_qc.py:513-540. Every authorization/reference/role gate precedes its
actual raw array operation. Temporary float validation of saved EOG values is
not reused as the stored numeric dtype: mean subtraction and coefficient
multiplication perform fresh dtype-preserving asarray calls, in source order.
MNE IO, copying, annotation and individual NumPy arithmetic remain explicitly
unrefined library operations. No complete raw builder/validator is a primitive.
-/

namespace N2P3.PersonalERPRawOperators

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Raw where
  names : List String
  kinds : List String
  bads : List String
  rate : Atom
  samples : ℕ
  firstSample : ℤ
  payload : Value

structure Span where
  start : ℕ
  stop : ℕ

structure Primitives where
  scope : PersonalERPRawScope.Primitives
  recordingIO : PersonalERPSession.Primitives
  readRaw : String → Check Raw
  annotate : Raw → ℝ → ℝ → String → Check Raw
  setTypes : Raw → List (String × String) → Check Raw
  copyLoad : Raw → Check Raw
  getData : Raw → List String → Check (ArrayImage Atom)
  setEEGReference : Raw → String → Check Raw
  setMisc : Raw → String → Check Raw
  subtractSignal : Raw → List String → ArrayImage Atom → Check Raw
  columnBroadcast : ArrayImage Atom → Check (ArrayImage Atom)
  subtractArrays : ArrayImage Atom → ArrayImage Atom → Check (ArrayImage Atom)
  scalarMultiply : ℝ → ArrayImage Atom → Check (ArrayImage Atom)
  matrixMultiply : ArrayImage Atom → ArrayImage Atom → Check (ArrayImage Atom)
  lower : String → String
  scalarSubtract : Atom → Atom → Check Atom
  scalarAdd : Atom → Atom → Check Atom
  matrixFromRows : List (List Atom) → Check (ArrayImage Atom)

def roles (raw : Raw) : Check (List String × List String) := do
  ensure (raw.names.length=raw.kinds.length) "strict raw channel name type zip"
  let channels := raw.names.zip raw.kinds
  let eeg := (channels.filter (fun row => decide (row.2="eeg"))).map Prod.fst
  let eog := (channels.filter (fun row => decide (row.2="eog"))).map Prod.fst
  ensure (eeg ≠ [] ∧ eog ≠ []) "independently typed raw EEG and EOG channels"
  ensure (eog.toFinset ∩ raw.bads.toFinset=∅) "bad EOG cannot supply artifact correction"
  pure (eeg,eog)

def referenceRaw (p : Primitives) (raw : Raw) (reference : Option String) (policy : Value) : Check Raw := do
  ensure (policy=.text "shared_measured_reference" ∨ policy=.text "preserve_acquired_eog") "raw ocular supported EOG policy"
  let output ← p.copyLoad raw
  match reference with
  | none => pure output
  | some anchor =>
    let (eeg,eog) ← roles output
    let canonical ← eeg.mapM p.scope.modal.recovery.input.canonicalChannel
    let matched := ((eeg.zip canonical).filter (fun row => decide (row.2=anchor))).map Prod.fst
    let measured ← match matched with
      | [name] => .ok name
      | _ => .error (.sourceReject "one usable measured ocular reference")
    ensure (measured ∉ output.bads) "measured ocular reference not bad"
    let waveform ← p.getData output [measured]
    let output ← p.setEEGReference output measured
    let output ← if policy=.text "shared_measured_reference" then p.subtractSignal output eog waveform else pure output
    p.setMisc output measured

def applyEOG (p : Primitives) (model : EOGModel) (raw : Raw) (sourceReference : String)
    (actualDerivation : Value) : Check Raw := do
  if model.referencePolicy=.text "preserve_acquired_eog" then
    let acquisition ← required model.fitProvenance "acquisition_source_reference"
    ensure (acquisition=.text sourceReference) "preserved EOG actual acquisition reference"
    let actual := if PyTruthy actualDerivation then actualDerivation else .object .nil
    let actual ← objectAt actual "preserved EOG actual derivation mapping"
    ensure (PersonalAdmission.PyEq (.object actual) (.object model.derivation)) "preserved EOG frozen derivation agreement"
  if PyTruthy (.object model.derivation) then
    ensure (PersonalAdmission.PyEq actualDerivation (.object model.derivation)) "raw actual EOG derivation agreement"
  checkEOGDerivation p.scope.modal.recovery.input (.object model.derivation) model.eogChannels (.text sourceReference)
    (decide (model.reference.isSome ∧ model.referencePolicy=.text "shared_measured_reference"))
  let output ← referenceRaw p raw model.reference model.referencePolicy
  let (eeg,eog) ← roles output
  let actualEEG ← eeg.mapM p.scope.modal.recovery.input.canonicalChannel
  let actualEOG ← eog.mapM p.scope.modal.recovery.input.canonicalChannel
  let modelEEG ← canonicalRoles p.scope.modal.recovery.input model.eegChannels
  let modelEOG ← canonicalRoles p.scope.modal.recovery.input model.eogChannels
  ensure (actualEEG=modelEEG ∧ actualEOG=modelEOG) "ocular exact raw EEG EOG channel order"
  let reference := match model.reference with
    | none => sourceReference
    | some anchor => "measured_channel:" ++ anchor
  ensure (reference=model.sourceReference) "ocular effective raw reference"
  if model.correctionStrength=0 then pure output
  else
    let artifact ← p.getData output eog
    ensure (finiteArray artifact) "ocular finite EOG application samples"
    -- Source stores the original tuple values; neither operation below uses
    -- the temporary dtype=float arrays produced by its constructor.
    let mean ← p.scope.modal.recovery.genericArray model.mean
    let mean ← p.columnBroadcast mean
    let artifact ← p.subtractArrays artifact mean
    let coefficients ← p.scope.modal.recovery.genericArray model.coefficients
    let coefficients ← p.scalarMultiply model.correctionStrength coefficients
    let correction ← p.matrixMultiply coefficients artifact
    p.subtractSignal output eeg correction

structure Native where
  raw : Raw
  rate : ℝ
  acquired : ℕ
  selected : List String
  sourceReference : String

def annotateBoundaries (p : Primitives) (rate : ℝ) (raw : Raw) : List ℝ → Check Raw
  | [] => .ok raw
  | onset :: tail => do
    let raw ← p.annotate raw onset 0 "BAD boundary"
    let raw ← p.annotate raw onset 0 "EDGE boundary"
    annotateBoundaries p rate raw tail

def prepareNative (p : Primitives) (recording : PersonalERPSession.Recording) (selected : Option (List String))
    (scope : Option PersonalERPRawScope.IndependentScope) : Check Native := do
  match scope with
  | none => pure ()
  | some scope => PersonalERPRawScope.independentApplication scope recording.session (.sequence .nil)
  let selected ← PersonalERPRawScope.selectChannels p.scope.modal.recovery.input recording.bids selected
  let path ← PersonalERPBids.getPath recording.bids.paths "recording"
  let raw ← p.readRaw path
  let rate ← strictReal raw.rate
  let acquired ← match recording.clock.counts with
    | none => .ok raw.samples
    | some counts => do
      ensure (0 ≤ counts.acquired) "nonnegative finalized raw acquired samples"
      pure counts.acquired.toNat
  let raw ← if acquired < raw.samples then do
    let onset ← p.recordingIO.clock.bids.divide acquired rate
    let duration ← p.recordingIO.clock.bids.divide (raw.samples-acquired) rate
    p.annotate raw onset duration "BAD_acq_skip_edf_padding"
    else pure raw
  ensure (p.recordingIO.clock.bids.close recording.bids.rate rate (1/1000000)=true) "raw BIDS source sample rate agreement"
  let rawNames ← raw.names.mapM p.scope.modal.recovery.input.canonicalChannel
  let expected ← (recording.bids.channels.map PersonalERPBids.Channel.name).mapM p.scope.modal.recovery.input.canonicalChannel
  ensure (rawNames=expected) "raw BIDS exact channel order"
  ensure (raw.names.length=recording.bids.channels.length) "strict raw BIDS type status zip"
  let types := raw.names.zip (recording.bids.channels.map (fun channel => p.lower channel.kind))
  let raw ← p.setTypes raw types
  let bads := ((raw.names.zip recording.bids.channels).filter (fun row => decide (row.2.status="bad"))).map Prod.fst
  let raw := {raw with bads := bads}
  let raw ← annotateBoundaries p rate raw recording.bids.events.boundaries
  pure ⟨raw,rate,acquired,selected,recording.bids.reference⟩

def setCell (row : List Atom) (index : ℕ) (value : Atom) : List Atom := row.set index value

def modifyCell (p : Primitives) (row : List Atom) (index : ℕ) (coefficient : Atom) (add : Bool) : Check (List Atom) := do
  let value ← match row[index]? with
    | none => .error (.fieldFailure "QC dependency matrix column index")
    | some value => .ok value
  let updated ← if add then p.scalarAdd value coefficient else p.scalarSubtract value coefficient
  pure (setCell row index updated)

def channelIndex (mapping : List (String × ℕ)) (name : String) : Check ℕ :=
  match mapping.reverse.lookup name with
  | none => .error (.sourceReject "QC includes each raw signal dependency")
  | some index => .ok index

def dependencyEOG (p : Primitives) (mapping : List (String × ℕ)) (anchor : Option String)
    (policy : Value) : List Value → List Atom → List Atom → Check (List Atom)
  | [],[],row => .ok row
  | eye :: eyes,coefficient :: coefficients,row => do
    if zeroNumber coefficient=true then dependencyEOG p mapping anchor policy eyes coefficients row
    else
      let eye ← p.scope.modal.recovery.input.pythonString eye
      let eye ← p.scope.modal.recovery.input.canonicalChannel eye
      let index ← channelIndex mapping eye
      let row ← modifyCell p row index coefficient false
      let row ← match anchor with
        | none => .ok row
        | some name => do
          if policy=.text "shared_measured_reference" then
            let index ← channelIndex mapping name
            modifyCell p row index coefficient true
          else pure row
      dependencyEOG p mapping anchor policy eyes coefficients row
  | _,_,_ => .error (.fieldFailure "strict EOG coefficient dependency zip")

def indexOfChannel (name : String) : List String → ℕ → Check ℕ
  | [],_ => .error (.fieldFailure "frozen EOG output channel index")
  | head :: tail,index => if name=head then .ok index else indexOfChannel name tail (index+1)

def dependencyRows (p : Primitives) (mapping : List (String × ℕ)) (nativeCount : ℕ)
    (anchor : Option String) (ocular : Option EOGModel) : List String → Check (List (List Atom))
  | [] => .ok []
  | output :: tail => do
    let name ← p.scope.modal.recovery.input.canonicalChannel output
    match anchor with
    | none => pure ()
    | some anchor =>
      let _ ← channelIndex mapping anchor
      pure ()
    let row ← match mapping.reverse.lookup name with
      | none => pure (List.replicate nativeCount (Atom.real 0))
      | some index => do
        let row := setCell (List.replicate nativeCount (Atom.real 0)) index (.real 1)
        let row ← match anchor with
          | none => .ok row
          | some name => do
            let index ← channelIndex mapping name
            modifyCell p row index (.real 1) false
        match ocular with
        | none => pure row
        | some model => do
          let eeg ← canonicalRoles p.scope.modal.recovery.input model.eegChannels
          let rowIndex ← indexOfChannel name eeg 0
          let coefficients ← p.scope.modal.recovery.genericArray model.coefficients
          let effective ← p.scalarMultiply model.correctionStrength coefficients
          let width ← match effective.shape with
            | [_,width] => .ok width
            | _ => .error (.fieldFailure "effective ocular coefficient rank")
          let coefficients := (effective.values.drop (rowIndex*width)).take width
          dependencyEOG p mapping anchor model.referencePolicy model.eogChannels coefficients row
    let rest ← dependencyRows p mapping nativeCount anchor ocular tail
    pure (row :: rest)

def signalDependencyMatrix (p : Primitives) (native output : List String) (reference : Option String)
    (ocular : Option EOGModel) : Check (ArrayImage Atom) := do
  let names ← native.mapM p.scope.modal.recovery.input.canonicalChannel
  let mapping := names.zip (List.range native.length)
  let anchor := match ocular with | none => reference | some model => model.reference
  let anchor ← anchor.mapM p.scope.modal.recovery.input.canonicalChannel
  let rows ← dependencyRows p mapping native.length anchor ocular output
  p.matrixFromRows rows

theorem zero_strength_ignores_regression_product (p : Primitives) (model : EOGModel) (raw : Raw)
    (source : String) (derivation : Value)
    (multiply : ℝ → ArrayImage Atom → Check (ArrayImage Atom))
    (matmul : ArrayImage Atom → ArrayImage Atom → Check (ArrayImage Atom))
    (h : model.correctionStrength=0) :
    applyEOG {p with scalarMultiply:=multiply, matrixMultiply:=matmul} model raw source derivation=
      applyEOG p model raw source derivation := by simp [applyEOG,referenceRaw,h]

end

end N2P3.PersonalERPRawOperators
