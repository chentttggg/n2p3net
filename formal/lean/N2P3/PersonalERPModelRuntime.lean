import N2P3.PersonalERPQuality
import N2P3.PersonalERPScalar

/-!
# Scalar and enum operation refinement of the frozen raw-model field readers

This new layer preserves the reviewed decoded modules byte-for-byte. Each
NumPy finite expression with boolean interpretation and actual set membership runs at its source position,
with typed atomic exceptions. Tuple policy membership remains tuple membership.
Array finite-all is one named NumPy expression; dtype conversion is separate.
Inherited structural/IO/conversion faults keep their explicit old boundary;
this file does not claim exact Python exception classes at those boundaries.
No complete validator, default-valid model or prechecked integer range is used.
P06-NUMERIC-01 and P06-SET-01 are addressed only after current checks/review.
-/

namespace N2P3.PersonalERPModelRuntime

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels
open N2P3.PersonalERPScalar

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev R (α : Type) := PersonalERPScalar.Result α

/-- The source finite expression followed by Python boolean interpretation.
For a JSON sequence/object this is not assumed to be a scalar ufunc: ndarray
truth-value ValueError, TypeError, OverflowError and unrefined cases are kept. -/
structure Finite where
  numpyFiniteTruth : Value → Except AtomicException Bool

structure Arrays where
  finiteAll : ArrayImage Atom → Except AtomicException Bool

structure Execution where
  original : PersonalERPExecution.Primitives

def Execution.strip (p : Execution) : String → String := p.original.strip
def Execution.floatArray (p : Execution) (value : Value) : R (ArrayImage Atom) := liftCheck (p.original.floatArray value)
def Execution.semanticHash (p : Execution) (value : Value) : R String := liftCheck (p.original.semanticHash value)

structure Input where
  original : PersonalERPInput.Primitives
  finite : Finite

def Input.execution (p : Input) : Execution := ⟨p.original.execution⟩
def Input.pythonString (p : Input) (value : Value) : R String := liftCheck (p.original.pythonString value)
def Input.canonicalChannel (p : Input) (value : String) : R String := liftCheck (p.original.canonicalChannel value)

structure Recovery where
  original : RecoveryPrimitives
  finite : Finite
  arrays : Arrays

def Recovery.input (p : Recovery) : Input := ⟨p.original.input,p.finite⟩
def Recovery.genericArray (p : Recovery) (value : Value) : R (ArrayImage Atom) := liftCheck (p.original.genericArray value)
def Recovery.floatScalar (p : Recovery) (value : Value) : R Atom := liftCheck (p.original.floatScalar value)

structure Modal where
  original : ModalPrimitives
  finite : Finite
  arrays : Arrays

def Modal.recovery (p : Modal) : Recovery := ⟨p.original.recovery,p.finite,p.arrays⟩
def Modal.arrayJSON (p : Modal) (image : ArrayImage Atom) : R Value := liftCheck (p.original.arrayJSON image)

def objectAt (value : Value) (site : String) : R Fields := liftCheck (ERPState.objectAt value site)
def text (value : Value) (site : String) : R String := liftCheck (PersonalERPExecution.text value site)
def integer (value : Value) (site : String) : R ℤ := liftCheck (PersonalERPExecution.integer value site)
def exactBool (value : Value) (site : String) : R Bool := liftCheck (PersonalERPData.exactBool value site)
def required (fields : Fields) (key : String) : R Value := liftCheck (StateLedger.required fields key)
def numeric (value : Value) (site : String) : R ℝ := liftCheck (PersonalERPModels.numeric value site)
def strictReal (value : Atom) : R ℝ := liftCheck (PersonalERPData.strictReal value)
def tupleValue (value : Value) : R (List Value) := liftCheck (PersonalERPRaw.tupleValue value)
def tupleJSON (value : Value) : R Value := liftCheck (PersonalERPModels.tupleJSON value)
def matrixTupleJSON (value : Value) : R Value := liftCheck (PersonalERPModels.matrixTupleJSON value)
def readQCSpecArgs (fields : Fields) : R Fields := liftCheck (PersonalERPModels.readQCSpecArgs fields)
def checkQCVersionHeader (fields : Fields) : R Fields := liftCheck (PersonalERPModels.checkQCVersionHeader fields)
def readHashableTuple (value : Value) : R (List Value) := liftCheck (PersonalERPModels.readHashableTuple value)
def epochRecoveryFromRecord (p : Modal) (value : Value) : R EpochRecovery := liftCheck (PersonalERPModels.epochRecoveryFromRecord p.original value)
def continuousQCRecord (p : Modal) (model : ContinuousQC) : R Fields := liftCheck (PersonalERPModels.continuousQCRecord p.original model)

def numpyNumeric (p : Finite) (value : Value) (site : String) : R ℝ := do
  let finite ← (p.numpyFiniteTruth value).mapError (Fault.atomic site .numpyFinite)
  require (finite=true) site
  numeric value site

def builtinNumeric : Value → Bool
  | .boolean _ | .integer _ | .number _ => true
  | _ => false

def typedFiniteNumeric (p : Finite) (value : Value) (site : String) : R ℝ := do
  require (builtinNumeric value=true) site
  numpyNumeric p value site

def arrayFinite (p : Arrays) (image : ArrayImage Atom) (site : String) : R Unit := do
  let finite ← (p.finiteAll image).mapError (Fault.atomic site .numpyFinite)
  require (finite=true) site

def numpyScalar : Value → Bool
  | .text _ | .boolean _ | .integer _ | .number _ => true
  | _ => false

def canonicalReference (p : Input) (value : Value) : R (Option String) := do
  if value=.null then pure none
  else
    let rendered ← p.pythonString value
    let channel ← p.canonicalChannel rendered
    pure (some channel)

def canonicalRoles (p : Input) : List Value → R (List String)
  | [] => .ok []
  | value :: tail => do
    let rendered ← p.pythonString value
    let channel ← p.canonicalChannel rendered
    let channels ← canonicalRoles p tail
    pure (channel :: channels)

def checkDerivationRecords (p : Input) (source : Value) (rereference : Bool) :
    List (String × Value) → R Unit
  | [] => .ok ()
  | (_,value) :: tail => do
    let row ← objectAt value "EOG derivation channel record mapping"
    setOption "EOG derivation kind" [.text "single_ended",.text "bipolar"] (fieldDefault row "kind")
    let placement ← p.pythonString ((row.get "placement").getD (.text ""))
    let reference ← p.pythonString ((row.get "reference").getD (.text ""))
    require (p.execution.strip placement ≠ "" ∧ p.execution.strip reference ≠ "")
      "explicit EOG placement and reference"
    if rereference then
      require (fieldDefault row "kind"=.text "single_ended" ∧
        PersonalAdmission.PyEq (fieldDefault row "reference") source) "shared EOG acquisition reference"
    checkDerivationRecords p source rereference tail

def checkEOGDerivation (p : Input) (value : Value) (channels : List Value)
    (source : Value) (rereference : Bool) : R Unit := do
  if ¬PyTruthy value then
    require (rereference=false) "rereferencing requires explicit EOG derivation"
  else
    let fields ← objectAt value "EOG derivation mapping"
    require (∀ channel ∈ channels, hashableJSON channel) "hashable EOG channel keys"
    require (((fieldKeys fields).map Value.text).toFinset = channels.toFinset) "EOG derivation identifies typed EOG channels"
    checkDerivationRecords p source rereference (fieldPairs fields)

def constructEOGFromRecord (p : Input) (value : Value) : R EOGModel := do
  let fields ← objectAt value "frozen ocular model mapping"
  let schema := fieldDefault fields "schema"
  require (schema=.text "n2p3_calibration_eog_regression/3" ∨ schema=.text "n2p3_calibration_eog_regression/4" ∨
    schema=.text "n2p3_calibration_eog_regression/5") "frozen ocular model schema"
  require (schema ≠ .text "n2p3_calibration_eog_regression/3" ∨ fields.get "eog_reference_policy"=none)
    "ocular schema three has no EOG policy field"
  require (schema ≠ .text "n2p3_calibration_eog_regression/4" ∨
    fieldDefault fields "eog_reference_policy"=.text "preserve_acquired_eog") "ocular schema four preserved EOG"
  require (schema ≠ .text "n2p3_calibration_eog_regression/5" ∨
    (fields.get "correction_strength").isSome ∧ (fields.get "eog_reference_policy").isSome)
    "ocular schema five explicit strength and policy"
  let eegValue ← required fields "eeg_channels"
  let eeg ← tupleValue eegValue
  let eogValue ← required fields "eog_channels"
  let eog ← tupleValue eogValue
  let coefficients ← required fields "coefficients"
  let coefficients ← matrixTupleJSON coefficients
  let mean ← required fields "calibration_eog_mean_v"
  let mean ← tupleJSON mean
  let participantValue ← required fields "participant_key"
  let sourceValue ← required fields "source_reference"
  let provenanceValue ← required fields "fit_provenance"
  let referenceValue ← required fields "reference_channel"
  let derivationValue ← required fields "eog_derivation"
  let policy := (fields.get "eog_reference_policy").getD (.text "shared_measured_reference")
  let strengthValue := (fields.get "correction_strength").getD (.number 1)
  require ((∀ b, strengthValue ≠ .boolean b) ∧ numpyScalar strengthValue=true) "ocular scalar nonbool strength"
  let strength ← numpyNumeric p.finite strengthValue "ocular strength finite test"
  require (0 ≤ strength ∧ strength ≤ 1) "finite ocular strength between zero and one"
  require (policy=.text "shared_measured_reference" ∨ policy=.text "preserve_acquired_eog") "ocular reference policy"
  let provenance ← objectAt provenanceValue "ocular fit provenance mapping"
  if policy=.text "preserve_acquired_eog" then
    let acquisition ← text (fieldDefault provenance "acquisition_source_reference") "preserved EOG acquisition reference"
    require (p.execution.strip acquisition ≠ "") "explicit preserved EOG acquisition reference"
  let acquisition := (provenance.get "acquisition_source_reference").getD sourceValue
  checkEOGDerivation p derivationValue eog acquisition
    (decide (referenceValue ≠ .null ∧ policy=.text "shared_measured_reference"))
  let reference ← canonicalReference p referenceValue
  match reference with
  | none => pure ()
  | some channel => require (sourceValue=.text ("measured_channel:" ++ channel)) "ocular final measured reference"
  let roles ← canonicalRoles p (eeg ++ eog)
  require (eeg ≠ [] ∧ eog ≠ [] ∧ roles.Nodup) "nonempty unique disjoint ocular roles"
  let coefficientArray ← p.execution.floatArray coefficients
  let meanArray ← p.execution.floatArray mean
  require (coefficientArray.shape=[eeg.length,eog.length]) "ocular coefficient channel role geometry"
  require (meanArray.shape=[eog.length] ∧ (∀ value ∈ meanArray.values, finiteNumber value=true) ∧
    (∀ value ∈ coefficientArray.values, finiteNumber value=true)) "finite aligned ocular model values"
  let participant ← text participantValue "ocular participant string"
  let source ← text sourceValue "ocular source reference string"
  require (p.execution.strip participant ≠ "" ∧ p.execution.strip source ≠ "" ∧ PyTruthy provenanceValue)
    "ocular participant reference and fit provenance"
  let derivation ← objectAt derivationValue "ocular record derivation dictionary"
  let model : EOGModel := ⟨fields,eeg,eog,coefficients,mean,participant,source,provenance,reference,
    derivation,policy,strength⟩
  pure model

def readEOGModel (p : Input) (value : Value) : R EOGModel := do
  let model ← constructEOGFromRecord p value
  require (PersonalAdmission.PyEq (.object (eogRecord model)) (.object model.original)) "exact reconstructed ocular record"
  pure model

def checkQCSpec (p : Input) (fields : Fields) : R QCSpec := do
  let fields ← readQCSpecArgs fields
  let value := fieldDefault fields
  setOption "continuous range action" [.text "reject",.text "diagnostic"] (value "range_action")
  setOption "continuous boundary action" [.text "all_edges",.text "artifact_edges"] (value "boundary_action")
  let difference ← numpyNumeric p.finite (value "difference_z") "finite numeric difference z"
  require (0 < difference) "positive continuous difference z"
  let floor ← numpyNumeric p.finite (value "difference_floor_v") "finite numeric difference floor"
  require (0 < floor) "positive continuous difference floor"
  let settle ← integer (value "settle_samples") "continuous exact integer settle samples"
  require (1 ≤ settle) "positive continuous settle samples"
  let localFlag ← exactBool (value "channel_local") "continuous strict local Boolean"
  let includeEOG ← exactBool (value "include_eog") "continuous strict EOG Boolean"
  let guard ← integer (value "artifact_pre_guard_samples") "continuous exact integer pre guard"
  require (0 ≤ guard ∧ (guard ≠ 0 → localFlag=true)) "nonnegative local continuous pre guard"
  let fraction ← numeric (value "minimum_channel_fraction") "continuous numeric minimum fraction"
  require (0 < fraction ∧ fraction ≤ 1) "continuous minimum channel fraction"
  require (value "reference_channel"=.null ∨ localFlag=true) "QC owned reference requires local mode"
  let reference ← canonicalReference p (value "reference_channel")
  let fields := match reference with
    | none => fields
    | some channel => .cons "reference_channel" (.text channel) (fields.erase "reference_channel")
  let range ← if value "range_window_samples"=.null then .ok none else do
    require (localFlag=true) "range monitoring requires local mode"
    let samples ← integer (value "range_window_samples") "continuous exact integer range window"
    require (2 ≤ samples) "continuous range at least two samples"
    let quantile ← numeric (value "range_quantile") "continuous numeric range quantile"
    require (1/2 < quantile ∧ quantile < 1) "continuous range quantile interval"
    let inflation ← numpyNumeric p.finite (value "range_inflation") "continuous finite range inflation"
    require (1 ≤ inflation) "continuous range inflation at least one"
    let rangeFloor ← numpyNumeric p.finite (value "range_floor_v") "continuous finite range floor"
    require (0 < rangeFloor) "positive continuous range floor"
    pure (some samples)
  let duration ← if value "flatline_min_duration_s"=.null then .ok none else do
    require (localFlag=true) "flatline detection requires local mode"
    require ((∀ b, value "flatline_min_duration_s" ≠ .boolean b) ∧
      numericNonBool (value "flatline_min_duration_s")=true) "raw flatline duration numeric nonbool type"
    let duration ← numpyNumeric p.finite (value "flatline_min_duration_s") "raw flatline duration finite test"
    require (0 < duration) "positive finite nonbool raw flatline duration"
    pure (some duration)
  pure ⟨fields,localFlag,includeEOG,reference,fraction,guard,range,duration⟩

def readStateBox (p : Recovery) (value : Value) : R StateBox := do
  let fields ← objectAt value "state recovery kwargs mapping"
  require (fieldKeys fields ⊆ stateBoxKeywords) "state recovery known kwargs"
  let sosValue ← required fields "sos"
  let sosValue ← matrixTupleJSON sosValue
  let boundsValue ← required fields "state_error_bound_v"
  let boundsValue ← matrixTupleJSON boundsValue
  let toleranceValue ← required fields "tolerance_v"
  let gainValue ← required fields "epoch_operator_gain"
  let evidenceValue ← required fields "evidence"
  let evidence ← text evidenceValue "state bound evidence string"
  require (p.input.execution.strip evidence ≠ "") "explicit state bound evidence"
  let sos ← p.genericArray sosValue
  let bounds ← p.genericArray boundsValue
  let sections ← match sos.shape with
    | [sections,6] => .ok sections
    | _ => .error (.inherited (.fieldFailure "SOS rank two with six columns"))
  require (bounds.shape=[sections,2]) "state box geometry"
  arrayFinite p.arrays sos "state box SOS finite all"
  arrayFinite p.arrays bounds "state box bounds finite all"
  require (nonnegativeArray bounds) "state box nonnegative bounds"
  require (∀ sosRow ∈ List.range sections, (sos.values[sosRow*6+3]?).bind atomNumber=some 1)
    "normalized state box SOS denominator"
  let tolerance ← numpyNumeric p.finite toleranceValue "finite numeric state tolerance"
  require (0 < tolerance) "positive state tolerance"
  let gain ← numpyNumeric p.finite gainValue "finite numeric epoch operator gain"
  require (1 ≤ gain) "epoch gain at least one"
  let reference := (fields.get "input_reference").getD .null
  let ocular := (fields.get "ocular_model_digest").getD .null
  let record := Fields.ofList
    [("sos",sosValue),("state_error_bound_v",boundsValue),("tolerance_v",toleranceValue),
     ("epoch_operator_gain",gainValue),("evidence",.text evidence),
     ("input_reference",reference),("ocular_model_digest",ocular)]
  pure ⟨record,sos,bounds,tolerance,gain,evidence,reference,ocular⟩

def checkNestedRecovery (finite : Finite) (parent child : Fields) : R Unit := do
  require (PyTruthy (fieldDefault parent "participant_key") ∧
    PersonalAdmission.PyEq (fieldDefault child "participant_key") (fieldDefault parent "participant_key"))
    "nested recovery participant authority"
  require (PyTruthy (fieldDefault parent "source_reference") ∧
    PersonalAdmission.PyEq (fieldDefault child "source_reference") (fieldDefault parent "source_reference"))
    "nested recovery source reference"
  let childSessionsValue := fieldDefault child "calibration_session_ids"
  let parentSessionsValue := fieldDefault parent "calibration_session_ids"
  require (PyTruthy childSessionsValue ∧ PyTruthy parentSessionsValue) "nested recovery nonempty calibration sessions"
  let childSessions ← readHashableTuple childSessionsValue
  let parentSessions ← readHashableTuple parentSessionsValue
  require (∀ value ∈ childSessions, ∃ saved ∈ parentSessions, PersonalAdmission.PyEq value saved)
    "nested recovery calibration session subset"
  let childEnd ← typedFiniteNumeric finite (fieldDefault child "calibration_end_timestamp_s") "nested recovery child calibration end"
  let parentEnd ← typedFiniteNumeric finite (fieldDefault parent "calibration_end_timestamp_s") "nested recovery parent calibration end"
  require (childEnd ≤ parentEnd) "nested recovery excludes future calibration"

def constructContinuousQC (p : Modal) (value : Value) : R ContinuousQC := do
  let fields ← objectAt value "continuous QC record mapping"
  let specFields ← checkQCVersionHeader fields
  let recovery ← if fieldDefault fields "recovery"=.null then .ok none else do
    let model ← readStateBox p.recovery (fieldDefault fields "recovery")
    pure (some model)
  let spec ← checkQCSpec p.recovery.input specFields
  let channelsValue ← required fields "channel_names"
  let channels ← tupleValue channelsValue
  let rateValue ← required fields "sample_rate_hz"
  let centerValue ← required fields "difference_center_v"
  let center ← tupleJSON centerValue
  let scaleValue ← required fields "difference_scale_v"
  let scale ← tupleJSON scaleValue
  let provenanceValue ← required fields "fit_provenance"
  let epochValue ← required fields "epoch_recovery"
  let epoch ← if epochValue=.null then .ok none else do
    let model ← epochRecoveryFromRecord p epochValue
    pure (some model)
  let rangesValue ← required fields "range_threshold_v"
  let ranges ← if rangesValue=.null then .ok none else do
    let tuple ← tupleJSON rangesValue
    pure (some tuple)
  require (channels ≠ []) "nonempty continuous QC channels"
  require (∀ channel ∈ channels, hashableJSON channel) "hashable continuous QC channels"
  require ((channels.map PersonalAdmission.normalizeValue).Nodup) "unique continuous QC channels"
  let centerImage ← p.recovery.genericArray center
  let scaleImage ← p.recovery.genericArray scale
  require (centerImage.shape=[channels.length] ∧ scaleImage.shape=[channels.length]) "continuous QC center scale geometry"
  arrayFinite p.arrays centerImage "continuous QC center finite all"
  arrayFinite p.arrays scaleImage "continuous QC scale finite all"
  require (∀ value ∈ scaleImage.values, ∃ number, atomNumber value=some number ∧ 0 < number)
    "positive continuous QC scale"
  let rate ← numpyNumeric p.finite rateValue "continuous QC finite sample rate"
  require (0 < rate ∧ PyTruthy provenanceValue) "positive continuous QC rate and fit provenance"
  let provenance ← objectAt provenanceValue "continuous QC fit provenance mapping"
  match epoch with
  | none => pure ()
  | some child => do
    require (spec.channelLocal=true ∧ recovery=none ∧
      PersonalAdmission.PyEq (.sequence (valuesOfList child.channels)) (.sequence (valuesOfList channels)) ∧ child.rate=rate)
      "exclusive matching native epoch recovery"
    checkNestedRecovery p.finite provenance child.provenance
  if spec.rangeWindow.isSome then
    let image ← p.recovery.input.execution.floatArray (ranges.getD .null)
    require (image.shape=[channels.length] ∧ finiteArray image ∧
      ∀ value ∈ image.values, ∃ number, atomNumber value=some number ∧ 0 < number)
      "continuous QC positive fitted range thresholds"
  pure ⟨spec,channels,rateValue,rate,center,scale,provenance,recovery,epoch,ranges⟩

def continuousQCFromRecord (p : Modal) (value : Value) : R ContinuousQC := do
  let model ← constructContinuousQC p value
  let record ← continuousQCRecord p model
  let actualHash ← p.recovery.input.execution.semanticHash (.object record)
  let savedHash ← p.recovery.input.execution.semanticHash value
  require (actualHash=savedHash) "continuous QC canonical semantic digest"
  pure model


theorem numpy_exception_stops_before_scalar_comparison (p : Finite)
    (value : Value) (site : String) (exception : AtomicException)
    (h : p.numpyFiniteTruth value=.error exception) :
    numpyNumeric p value site=.error (.atomic site .numpyFinite exception) := by
  simp [numpyNumeric,h]

theorem tuple_EOG_policy_is_not_a_set_hashability_gate :
    require (Value.sequence .nil=.text "shared_measured_reference" ∨
      Value.sequence .nil=.text "preserve_acquired_eog") "ocular reference policy"=
      Except.error (Fault.sourceValueError "ocular reference policy") := by simp [require]

end
end N2P3.PersonalERPModelRuntime
