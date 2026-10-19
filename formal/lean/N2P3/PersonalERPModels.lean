import N2P3.PersonalERPRaw

/-!
# Explicit frozen raw-model readers used by the personal ERP loader

Source: ocular.py:31-147,291-326; continuous_qc.py:124-183,605-629.
The reader checks constructor fields and reconstructs the actual exported
record before comparison. Array conversion is per NumPy call; an entire
fitted-model validator, QC fit, raw scope or physical admission is never an
input primitive. Reader success alone does not guarantee numeric application
success. In particular numeric strings can pass the source's float-array
validation while the original stored tuple retains strings.

Continuous recovery/model and epoch-quality branches are subsequent explicit
components, not assumed success cases of this EOG/spec layer. All P06 rows
remain pending until the complete actual callers are connected and reviewed.
-/

namespace N2P3.PersonalERPModels

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def numeric (value : Value) (site : String) : Check ℝ :=
  match numericScalar value with
  | some number => .ok number
  | none => .error (.fieldFailure site)

def tupleJSON (value : Value) : Check Value := do
  let values ← tupleValue value
  pure (.sequence (valuesOfList values))

def matrixTuples : List Value → Check (List Value)
  | [] => .ok []
  | row :: tail => do
    let row ← tupleJSON row
    let rows ← matrixTuples tail
    pure (row :: rows)

def matrixTupleJSON (value : Value) : Check Value := do
  let rows ← tupleValue value
  let rows ← matrixTuples rows
  pure (.sequence (valuesOfList rows))

def canonicalRoles (p : PersonalERPInput.Primitives) : List Value → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let rendered ← p.pythonString value
    let channel ← p.canonicalChannel rendered
    let channels ← canonicalRoles p tail
    pure (channel :: channels)

def checkDerivationRecords (p : PersonalERPInput.Primitives) (source : Value) (rereference : Bool) :
    List (String × Value) → Check Unit
  | [] => .ok ()
  | (_,value) :: tail => do
    let row ← objectAt value "EOG derivation channel record mapping"
    ensure (fieldDefault row "kind"=.text "single_ended" ∨ fieldDefault row "kind"=.text "bipolar")
      "EOG derivation kind"
    let placement ← p.pythonString ((row.get "placement").getD (.text ""))
    let reference ← p.pythonString ((row.get "reference").getD (.text ""))
    ensure (p.execution.strip placement ≠ "" ∧ p.execution.strip reference ≠ "")
      "explicit EOG placement and reference"
    if rereference then
      ensure (fieldDefault row "kind"=.text "single_ended" ∧
        PersonalAdmission.PyEq (fieldDefault row "reference") source) "shared EOG acquisition reference"
    checkDerivationRecords p source rereference tail

def fieldPairs : Fields → List (String × Value)
  | .nil => []
  | .cons key value tail => (key,value) :: fieldPairs tail

def checkEOGDerivation (p : PersonalERPInput.Primitives) (value : Value) (channels : List Value)
    (source : Value) (rereference : Bool) : Check Unit := do
  if ¬PyTruthy value then
    ensure (rereference=false) "rereferencing requires explicit EOG derivation"
  else
    let fields ← objectAt value "EOG derivation mapping"
    ensure (∀ channel ∈ channels, hashableJSON channel) "hashable EOG channel keys"
    ensure (((fieldKeys fields).map Value.text).toFinset = channels.toFinset) "EOG derivation identifies typed EOG channels"
    checkDerivationRecords p source rereference (fieldPairs fields)

structure EOGModel where
  original : Fields
  eegChannels : List Value
  eogChannels : List Value
  coefficients : Value
  mean : Value
  participant : String
  sourceReference : String
  fitProvenance : Fields
  reference : Option String
  derivation : Fields
  referencePolicy : Value
  correctionStrength : ℝ

def eogRecord (model : EOGModel) : Fields :=
  let schema := if model.referencePolicy=.text "shared_measured_reference" then
    "n2p3_calibration_eog_regression/3" else "n2p3_calibration_eog_regression/4"
  let base := Fields.ofList
    [("schema",.text schema),("eeg_channels",.sequence (valuesOfList model.eegChannels)),
     ("eog_channels",.sequence (valuesOfList model.eogChannels)),("coefficients",model.coefficients),
     ("calibration_eog_mean_v",model.mean),("participant_key",.text model.participant),
     ("source_reference",.text model.sourceReference),("fit_provenance",.object model.fitProvenance),
     ("application",.text "pointwise_subtraction_with_frozen_calibration_center"),("signal_unit",.text "V"),
     ("reference_changed",.boolean model.reference.isSome),
     ("reference_channel",(model.reference.map Value.text).getD .null),("eog_derivation",.object model.derivation)]
  let withPolicy := if model.referencePolicy=.text "shared_measured_reference" then base
    else .cons "eog_reference_policy" model.referencePolicy base
  if model.correctionStrength ≠ 1 then
    .cons "schema" (.text "n2p3_calibration_eog_regression/5")
      (.cons "correction_strength" (.number model.correctionStrength)
        (.cons "eog_reference_policy" model.referencePolicy
          ((withPolicy.erase "schema").erase "eog_reference_policy")))
  else withPolicy

def constructEOGFromRecord (p : PersonalERPInput.Primitives) (value : Value) : Check EOGModel := do
  let fields ← objectAt value "frozen ocular model mapping"
  let schema := fieldDefault fields "schema"
  ensure (schema=.text "n2p3_calibration_eog_regression/3" ∨ schema=.text "n2p3_calibration_eog_regression/4" ∨
    schema=.text "n2p3_calibration_eog_regression/5") "frozen ocular model schema"
  ensure (schema ≠ .text "n2p3_calibration_eog_regression/3" ∨ fields.get "eog_reference_policy"=none)
    "ocular schema three has no EOG policy field"
  ensure (schema ≠ .text "n2p3_calibration_eog_regression/4" ∨
    fieldDefault fields "eog_reference_policy"=.text "preserve_acquired_eog") "ocular schema four preserved EOG"
  ensure (schema ≠ .text "n2p3_calibration_eog_regression/5" ∨
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
  let strength ← strictReal (atomFromValue strengthValue)
  ensure (0 ≤ strength ∧ strength ≤ 1) "finite ocular strength between zero and one"
  ensure (policy=.text "shared_measured_reference" ∨ policy=.text "preserve_acquired_eog") "ocular reference policy"
  let provenance ← objectAt provenanceValue "ocular fit provenance mapping"
  if policy=.text "preserve_acquired_eog" then
    let acquisition ← text (fieldDefault provenance "acquisition_source_reference") "preserved EOG acquisition reference"
    ensure (p.execution.strip acquisition ≠ "") "explicit preserved EOG acquisition reference"
  let acquisition := (provenance.get "acquisition_source_reference").getD sourceValue
  checkEOGDerivation p derivationValue eog acquisition
    (decide (referenceValue ≠ .null ∧ policy=.text "shared_measured_reference"))
  let reference ← canonicalReference p referenceValue
  match reference with
  | none => pure ()
  | some channel => ensure (sourceValue=.text ("measured_channel:" ++ channel)) "ocular final measured reference"
  let roles ← canonicalRoles p (eeg ++ eog)
  ensure (eeg ≠ [] ∧ eog ≠ [] ∧ roles.Nodup) "nonempty unique disjoint ocular roles"
  let coefficientArray ← p.execution.floatArray coefficients
  let meanArray ← p.execution.floatArray mean
  ensure (coefficientArray.shape=[eeg.length,eog.length]) "ocular coefficient channel role geometry"
  ensure (meanArray.shape=[eog.length] ∧ (∀ value ∈ meanArray.values, finiteNumber value=true) ∧
    (∀ value ∈ coefficientArray.values, finiteNumber value=true)) "finite aligned ocular model values"
  let participant ← text participantValue "ocular participant string"
  let source ← text sourceValue "ocular source reference string"
  ensure (p.execution.strip participant ≠ "" ∧ p.execution.strip source ≠ "" ∧ PyTruthy provenanceValue)
    "ocular participant reference and fit provenance"
  let derivation ← objectAt derivationValue "ocular record derivation dictionary"
  let model : EOGModel := ⟨fields,eeg,eog,coefficients,mean,participant,source,provenance,reference,
    derivation,policy,strength⟩
  pure model

def readEOGModel (p : PersonalERPInput.Primitives) (value : Value) : Check EOGModel := do
  let model ← constructEOGFromRecord p value
  ensure (PersonalAdmission.PyEq (.object (eogRecord model)) (.object model.original)) "exact reconstructed ocular record"
  pure model

def eogReferenceView (model : EOGModel) : OcularReferenceView := ⟨model.reference⟩

theorem shared_policy_strength_one_retains_schema_three (model : EOGModel)
    (hp : model.referencePolicy=.text "shared_measured_reference") (hs : model.correctionStrength=1) :
    (eogRecord model).get "schema"=some (.text "n2p3_calibration_eog_regression/3") := by
  simp [eogRecord,hp,hs,Fields.ofList,Fields.get]

theorem nonunit_strength_export_is_schema_five (model : EOGModel) (h : model.correctionStrength ≠ 1) :
    (eogRecord model).get "schema"=some (.text "n2p3_calibration_eog_regression/5") := by
  simp [eogRecord,h,Fields.get]

theorem bool_strength_rejected_by_source_scalar_gate (b : Bool) :
    strictReal (atomFromValue (.boolean b))=.error (.fieldFailure "finite nonbool Real") := rfl

theorem accepted_ocular_reader_reconstructs_original_fields (p : PersonalERPInput.Primitives)
    (value : Value) (model : EOGModel) (h : readEOGModel p value=.ok model) :
    PersonalAdmission.PyEq (.object (eogRecord model)) (.object model.original) := by
  unfold readEOGModel at h
  rcases (bind_success _ _ _).mp h with ⟨built,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hg,h⟩
  have he : built=model := Except.ok.inj h
  rw [←he]
  exact ensure_success _ _ completion hg

def qcSpecDefaults : Fields := Fields.ofList
  [("channel_local",.boolean false),("reference_channel",.null),("include_eog",.boolean false),
   ("minimum_channel_fraction",.number 1),("range_window_samples",.null),("range_quantile",.number (995/1000)),
   ("range_inflation",.number 2),("range_floor_v",.number (50/1000000)),("artifact_pre_guard_samples",.integer 0),
   ("flatline_min_duration_s",.null),("range_action",.text "reject"),("boundary_action",.text "all_edges")]

def qcSpecKeywords : List String := ["difference_z","difference_floor_v","settle_samples"] ++ fieldKeys qcSpecDefaults

def readQCSpecArgs (fields : Fields) : Check Fields := do
  ensure (fieldKeys fields ⊆ qcSpecKeywords) "continuous QC spec known kwargs"
  let _ ← required fields "difference_z"
  let _ ← required fields "difference_floor_v"
  let _ ← required fields "settle_samples"
  pure (Fields.ofList (qcSpecKeywords.map fun key =>
    (key,(fields.get key).getD (fieldDefault qcSpecDefaults key))))

structure QCSpec where
  normalized : Fields
  channelLocal : Bool
  includeEOG : Bool
  reference : Option String
  minimumFraction : ℝ
  preGuard : ℤ
  rangeWindow : Option ℤ
  flatlineDuration : Option ℝ

def checkQCSpec (p : PersonalERPInput.Primitives) (fields : Fields) : Check QCSpec := do
  let fields ← readQCSpecArgs fields
  let value := fieldDefault fields
  ensure (value "range_action"=.text "reject" ∨ value "range_action"=.text "diagnostic") "continuous range action"
  ensure (value "boundary_action"=.text "all_edges" ∨ value "boundary_action"=.text "artifact_edges")
    "continuous boundary action"
  let difference ← numeric (value "difference_z") "finite numeric difference z"
  ensure (0 < difference) "positive continuous difference z"
  let floor ← numeric (value "difference_floor_v") "finite numeric difference floor"
  ensure (0 < floor) "positive continuous difference floor"
  let settle ← integer (value "settle_samples") "continuous exact integer settle samples"
  ensure (1 ≤ settle) "positive continuous settle samples"
  let localFlag ← exactBool (value "channel_local") "continuous strict local Boolean"
  let includeEOG ← exactBool (value "include_eog") "continuous strict EOG Boolean"
  let guard ← integer (value "artifact_pre_guard_samples") "continuous exact integer pre guard"
  ensure (0 ≤ guard ∧ (guard ≠ 0 → localFlag=true)) "nonnegative local continuous pre guard"
  let fraction ← numeric (value "minimum_channel_fraction") "continuous numeric minimum fraction"
  ensure (0 < fraction ∧ fraction ≤ 1) "continuous minimum channel fraction"
  ensure (value "reference_channel"=.null ∨ localFlag=true) "QC owned reference requires local mode"
  let reference ← canonicalReference p (value "reference_channel")
  let fields := match reference with
    | none => fields
    | some channel => .cons "reference_channel" (.text channel) (fields.erase "reference_channel")
  let range ← if value "range_window_samples"=.null then .ok none else do
    ensure (localFlag=true) "range monitoring requires local mode"
    let samples ← integer (value "range_window_samples") "continuous exact integer range window"
    ensure (2 ≤ samples) "continuous range at least two samples"
    let quantile ← numeric (value "range_quantile") "continuous numeric range quantile"
    ensure (1/2 < quantile ∧ quantile < 1) "continuous range quantile interval"
    let inflation ← numeric (value "range_inflation") "continuous finite range inflation"
    ensure (1 ≤ inflation) "continuous range inflation at least one"
    let rangeFloor ← numeric (value "range_floor_v") "continuous finite range floor"
    ensure (0 < rangeFloor) "positive continuous range floor"
    pure (some samples)
  let duration ← if value "flatline_min_duration_s"=.null then .ok none else do
    ensure (localFlag=true) "flatline detection requires local mode"
    let duration ← strictReal (atomFromValue (value "flatline_min_duration_s"))
    ensure (0 < duration) "positive finite nonbool raw flatline duration"
    pure (some duration)
  pure ⟨fields,localFlag,includeEOG,reference,fraction,guard,range,duration⟩

def qcSpecRecord (spec : QCSpec) : Fields :=
  let noRange := if fieldDefault spec.normalized "range_action"=.text "reject" then
    spec.normalized.erase "range_action" else spec.normalized
  let noBoundary := if fieldDefault spec.normalized "boundary_action"=.text "all_edges" then
    noRange.erase "boundary_action" else noRange
  if spec.flatlineDuration=none then noBoundary.erase "flatline_min_duration_s" else noBoundary

def qcSpecSchema (spec : QCSpec) : String :=
  if spec.flatlineDuration=none then "n2p3_continuous_qc/3" else "n2p3_continuous_qc/4"

def checkQCVersionHeader (fields : Fields) : Check Fields := do
  let schema := fieldDefault fields "schema"
  ensure (schema=.text "n2p3_continuous_qc/3" ∨ schema=.text "n2p3_continuous_qc/4") "continuous QC schema"
  let spec ← objectAt (fieldDefault fields "spec") "continuous QC spec mapping"
  ensure (schema ≠ .text "n2p3_continuous_qc/3" ∨ spec.get "flatline_min_duration_s"=none)
    "continuous schema three cannot declare raw flatline detection"
  ensure (schema ≠ .text "n2p3_continuous_qc/4" ∨ fieldDefault spec "flatline_min_duration_s" ≠ .null)
    "continuous schema four requires raw flatline duration"
  pure spec

theorem numeric_bool_thresholds_have_source_numeric_behavior :
    numeric (.boolean true) "threshold"=.ok 1 := rfl

theorem continuous_switch_still_needs_exact_bool :
    exactBool (.integer 1) "continuous strict local Boolean"=
      .error (.fieldFailure "continuous strict local Boolean") := rfl

theorem flatline_duration_has_no_bool_alias (b : Bool) :
    strictReal (atomFromValue (.boolean b))=.error (.fieldFailure "finite nonbool Real") := rfl

structure RecoveryPrimitives where
  input : PersonalERPInput.Primitives
  genericArray : Value → Check (ArrayImage Atom)
  floatScalar : Value → Check Atom
  arrayAllclose : ArrayImage Atom → ArrayImage Atom → ℝ → ℝ → Check Bool
  designSOS : ℝ → Fields → Check (ArrayImage Atom)
  baselineIndex : ℝ → ℝ → ℝ → Check ℤ
  multiply : ℝ → ℝ → Check ℝ

def nonnegativeArray (array : ArrayImage Atom) : Prop :=
  ∀ value ∈ array.values, ∃ r, atomNumber value=some r ∧ 0 ≤ r

def finiteArray (array : ArrayImage Atom) : Prop := ∀ value ∈ array.values, finiteNumber value=true

def checkNormalizedSOS (array : ArrayImage Atom) : Check ℕ := do
  let sections ← match array.shape with
    | [sections,6] => .ok sections
    | _ => .error (.fieldFailure "SOS rank two with six columns")
  ensure (finiteArray array) "finite SOS coefficients"
  ensure (∀ sosRow ∈ List.range sections, (array.values[sosRow*6+3]?).bind atomNumber=some 1)
    "normalized SOS denominator"
  pure sections

structure StateBox where
  record : Fields
  sos : ArrayImage Atom
  bounds : ArrayImage Atom
  tolerance : ℝ
  epochGain : ℝ
  evidence : String
  inputReference : Value
  ocularDigest : Value

def stateBoxKeywords : List String :=
  ["sos","state_error_bound_v","tolerance_v","epoch_operator_gain","evidence","input_reference","ocular_model_digest"]

def readStateBox (p : RecoveryPrimitives) (value : Value) : Check StateBox := do
  let fields ← objectAt value "state recovery kwargs mapping"
  ensure (fieldKeys fields ⊆ stateBoxKeywords) "state recovery known kwargs"
  let sosValue ← required fields "sos"
  let sosValue ← matrixTupleJSON sosValue
  let boundsValue ← required fields "state_error_bound_v"
  let boundsValue ← matrixTupleJSON boundsValue
  let toleranceValue ← required fields "tolerance_v"
  let gainValue ← required fields "epoch_operator_gain"
  let evidenceValue ← required fields "evidence"
  let evidence ← text evidenceValue "state bound evidence string"
  ensure (p.input.execution.strip evidence ≠ "") "explicit state bound evidence"
  let sos ← p.genericArray sosValue
  let bounds ← p.genericArray boundsValue
  let sections ← match sos.shape with
    | [sections,6] => .ok sections
    | _ => .error (.fieldFailure "SOS rank two with six columns")
  ensure (bounds.shape=[sections,2]) "state box geometry"
  ensure (finiteArray sos ∧ finiteArray bounds ∧ nonnegativeArray bounds)
    "finite nonnegative state box coefficients"
  ensure (∀ sosRow ∈ List.range sections, (sos.values[sosRow*6+3]?).bind atomNumber=some 1)
    "normalized state box SOS denominator"
  let tolerance ← numeric toleranceValue "finite numeric state tolerance"
  let gain ← numeric gainValue "finite numeric epoch operator gain"
  ensure (0 < tolerance ∧ 1 ≤ gain) "positive tolerance and epoch gain at least one"
  let reference := (fields.get "input_reference").getD .null
  let ocular := (fields.get "ocular_model_digest").getD .null
  let record := Fields.ofList
    [("sos",sosValue),("state_error_bound_v",boundsValue),("tolerance_v",toleranceValue),
     ("epoch_operator_gain",gainValue),("evidence",.text evidence),
     ("input_reference",reference),("ocular_model_digest",ocular)]
  pure ⟨record,sos,bounds,tolerance,gain,evidence,reference,ocular⟩

def checkActualBaselineGrid (p : RecoveryPrimitives) (args : Fields) (declaration : DeclarationScalars)
    (sourceRate : ℝ) : Check Unit := do
  let raw := rawPreprocessing args
  if raw.baseline=.text "none" then pure ()
  else
    ensure (raw.baseline=.text "mean_only" ∨ raw.baseline=.text "trial_reference") "executable center only baseline"
    ensure (raw.referenceCenter=.text "mean" ∨ raw.referenceCenter=.text "median") "executable baseline center"
    ensure (raw.referenceScale=.text "none") "baseline preserves physical volts"
    let tmin ← p.input.execution.divide declaration.tmin 1000
    let effectiveRate := if p.input.execution.closeRates sourceRate declaration.sfreq then sourceRate else declaration.sfreq
    let window ← if raw.baseline=.text "mean_only" then do
      ensure (tmin < 0) "actual prestimulus baseline time"
      let start ← p.multiply tmin 1000
      pure (start,(0 : ℝ))
      else match raw.referenceWindow with
      | .sequence [start,stop] => do
        let start ← strictReal start
        let stop ← strictReal stop
        pure (start,stop)
      | _ => .error (.fieldFailure "actual trial reference window pair")
    let start ← p.baselineIndex window.1 tmin effectiveRate
    let stop ← p.baselineIndex window.2 tmin effectiveRate
    ensure (0 ≤ start ∧ stop ≤ declaration.nTimes ∧ 2 ≤ stop-start) "actual half open baseline grid"

/-- Explicit validation path for transform_epoch_samples(empty((0,samples)),
require_linear=True). The empty tensor returns before the numerical FFT, but
after every execution/baseline/capability and source-grid guard. -/
def checkEmptyLinearEpoch (p : RecoveryPrimitives) (args : Fields) (sourceRate : ℝ) (samples : ℤ) : Check Unit := do
  let raw := rawPreprocessing args
  let declaration ← checkPreprocessing p.input.execution.strip p.input.execution.floorSamples
    p.input.execution.windowSamples raw
  ensure (0 < sourceRate) "finite positive epoch source rate"
  ensure (raw.rejectThreshold=.null) "retired absolute voltage epoch rejection"
  ensure (∀ frequency ∈ declaration.lFreq, frequency < sourceRate/2) "epoch high pass below source Nyquist"
  ensure (∀ frequency ∈ declaration.hFreq, frequency < sourceRate/2) "epoch low pass below source Nyquist"
  let _ ← readNotches raw.notchFrequencies raw.notchQuality (some sourceRate)
  checkActualBaselineGrid p args declaration sourceRate
  ensure (raw.baseline ≠ .text "trial_reference" ∨ raw.referenceCenter=.text "mean") "linear epoch cannot use median reference"
  ensure (0 < samples) "nonempty source epoch time axis"
  let resampled ← if p.input.execution.closeRates sourceRate declaration.sfreq then .ok samples
    else p.input.execution.roundResampled samples declaration.sfreq sourceRate
  ensure (0 < resampled ∧ declaration.nTimes ≤ resampled) "empty epoch transform output sample coverage"

def expectedRecoverySOS (p : RecoveryPrimitives) (args : Fields) (rate : ℝ) (bidirectional : Bool) :
    Check (ArrayImage Atom) := do
  let raw := rawPreprocessing args
  let _ ← checkPreprocessing p.input.execution.strip p.input.execution.floorSamples p.input.execution.windowSamples raw
  if bidirectional then
    ensure (raw.phase=.text "zero" ∧ raw.edgePolicy=.text "mne_reflect_limited") "bidirectional finite record SOS declaration"
  else ensure (raw.filterMethod=.text "iir" ∧ raw.phase=.text "forward") "forward state recovery IIR declaration"
  p.designSOS rate args

theorem state_box_evidence_is_not_optional (p : RecoveryPrimitives) (evidence : String)
    (h : p.input.execution.strip evidence="") :
    ensure (p.input.execution.strip evidence ≠ "") "explicit state bound evidence"=
      .error (.sourceReject "explicit state bound evidence") := by simp [ensure,h]

theorem scalar_bool_gain_has_source_numeric_behavior : numeric (.boolean true) "gain"=.ok 1 := rfl

/-- Each field names one numerical/library operation. No fitted recovery
constructor or complete modal-system validator is supplied as a primitive. -/
structure ModalPrimitives where
  recovery : RecoveryPrimitives
  zeroResponse : ArrayImage Atom → ℕ → Check (List Atom × Atom)
  columnStack : List (List Atom) → Check (ArrayImage Atom)
  eigen : ArrayImage Atom → Check (ArrayImage Atom × ArrayImage Atom)
  lexsortRealImag : ArrayImage Atom → Check (List ℕ)
  takeVector : ArrayImage Atom → List ℕ → Check (ArrayImage Atom)
  takeColumns : ArrayImage Atom → List ℕ → Check (ArrayImage Atom)
  normalizeColumns : ArrayImage Atom → Check (ArrayImage Atom)
  condition : ArrayImage Atom → Check Atom
  maxAbsolute : ArrayImage Atom → Check Atom
  inverse : ArrayImage Atom → Check (ArrayImage Atom)
  realImagColumns : ArrayImage Atom → Check (ArrayImage Atom)
  arrayJSON : ArrayImage Atom → Check Value
  roundEndpoint : Value → ℝ → Check ℤ
  ceilProduct : ℝ → ℝ → Check ℤ

def modalResponses (p : ModalPrimitives) (sos : ArrayImage Atom) :
    List ℕ → Check (List (List Atom))
  | [] => .ok []
  | index :: rest => do
    let response ← p.zeroResponse sos index
    let tail ← modalResponses p sos rest
    pure (response.1 :: tail)

def checkedModalIdentity (p : ModalPrimitives) (sos : ArrayImage Atom) : Check (ArrayImage Atom) := do
  let sections ← match sos.shape with
    | [sections,6] => .ok sections
    | _ => .error (.fieldFailure "modal SOS geometry")
  let columns ← modalResponses p sos (List.range (2*sections))
  let transition ← p.columnStack columns
  let (poles,vectors) ← p.eigen transition
  let order ← p.lexsortRealImag poles
  let poles ← p.takeVector poles order
  let vectors ← p.takeColumns vectors order
  let vectors ← p.normalizeColumns vectors
  let condition ← p.condition vectors
  let condition ← strictReal condition
  ensure (condition ≤ 100000000) "well conditioned modal system"
  let radius ← p.maxAbsolute poles
  let radius ← strictReal radius
  ensure (radius < 1) "strictly stable modal system"
  let _ ← p.inverse vectors
  p.realImagColumns poles

structure EpochRecovery where
  bidirectional : Bool
  sos : ArrayImage Atom
  rate : ℝ
  preprocessing : Fields
  channels : List Value
  envelope : ArrayImage Atom
  tolerance : ℝ
  provenance : Fields
  serializedIdentity : Option Value

def readFiniteFloat (p : RecoveryPrimitives) (value : Value) : Check ℝ := do
  let converted ← p.floatScalar value
  strictReal converted

def readEpochRecovery (p : ModalPrimitives) (value : Value) : Check EpochRecovery := do
  let fields ← objectAt value "epoch recovery record mapping"
  let bidirectional := decide (fieldDefault fields "schema"=.text "n2p3_bidirectional_state_recovery/1")
  let argsValue ← required fields "preprocessing"
  let args ← objectAt argsValue "epoch recovery preprocessing kwargs"
  let args ← readPreprocessingArgs args
  let sosValue ← required fields "sos"
  let sos ← p.recovery.input.execution.floatArray sosValue
  let rateValue ← required fields "sample_rate_hz"
  let rateAtom ← p.recovery.floatScalar rateValue
  let channelsValue ← required fields "channel_names"
  let channels ← tupleValue channelsValue
  let envelopeValue ← required fields "modal_envelope"
  let envelope ← p.recovery.input.execution.floatArray envelopeValue
  let toleranceValue ← required fields "tolerance_v"
  let toleranceAtom ← p.recovery.floatScalar toleranceValue
  let provenanceValue ← required fields "fit_provenance"
  let provenance ← objectAt provenanceValue "epoch recovery provenance mapping"
  let identityValue := fieldDefault fields "mode_identity"
  let identity ← if identityValue=.null then .ok none else do
    let value ← matrixTupleJSON identityValue
    pure (some value)
  if bidirectional then
    let _ ← checkPreprocessing p.recovery.input.execution.strip p.recovery.input.execution.floorSamples
      p.recovery.input.execution.windowSamples (rawPreprocessing args)
    ensure (fieldDefault args "filter_phase"=.text "zero" ∧
      fieldDefault args "zero_phase_edge_policy"=.text "mne_reflect_limited")
      "bidirectional reflect limited declaration"
  let rate ← strictReal rateAtom
  let tolerance ← strictReal toleranceAtom
  ensure (0 < rate ∧ 0 < tolerance ∧ PyTruthy provenanceValue ∧ channels ≠ [])
    "positive recovery rate tolerance provenance channels"
  ensure (∀ channel ∈ channels, hashableJSON channel) "hashable recovery channel names"
  ensure ((channels.map PersonalAdmission.normalizeValue).Nodup) "unique recovery channel names"
  let expected ← expectedRecoverySOS p.recovery args rate bidirectional
  ensure (sos.shape=expected.shape) "recovery SOS matches expected geometry"
  let close ← p.recovery.arrayAllclose sos expected (1/1000000000000) 0
  ensure (close=true) "recovery SOS matches executed design"
  let sections ← match sos.shape with
    | [sections,6] => .ok sections
    | _ => .error (.fieldFailure "designed recovery SOS geometry")
  let width := if bidirectional then 4*sections else 2*sections
  ensure (envelope.shape=[channels.length,width] ∧ finiteArray envelope ∧ nonnegativeArray envelope)
    "nonnegative finite modal envelope geometry"
  let right ← p.roundEndpoint (fieldDefault args "tmax_ms") rate
  let left ← p.roundEndpoint (fieldDefault args "tmin_ms") rate
  checkEmptyLinearEpoch p.recovery args rate (right-left+1)
  -- Bidirectional construction always computes the modal system; the forward
  -- constructor computes it only for an explicitly serialized identity.
  let _ ← if bidirectional ∨ identity.isSome then do
    let expectedIdentity ← checkedModalIdentity p sos
    match identity with
    | none => pure ()
    | some value => do
      let image ← if bidirectional then p.recovery.genericArray value
        else p.recovery.input.execution.floatArray value
      ensure (image.shape=expectedIdentity.shape) "serialized mode identity geometry"
      if !bidirectional then ensure (finiteArray image) "finite serialized forward mode identity"
      let close ← p.recovery.arrayAllclose image expectedIdentity 0 (1/1000000000000)
      ensure (close=true) "serialized mode identity matches frozen filter"
    else pure ()
  pure ⟨bidirectional,sos,rate,args,channels,envelope,tolerance,provenance,identity⟩

def epochRecoveryRecord (p : ModalPrimitives) (model : EpochRecovery) : Check Fields := do
  -- The bidirectional record always evaluates modal_system, even when a
  -- serialized identity is retained. The forward record has the lazy branch.
  let expectedIdentity ← if model.bidirectional ∨ model.serializedIdentity=none then do
    let image ← checkedModalIdentity p model.sos
    let value ← p.arrayJSON image
    pure (some value)
    else pure none
  let identity ← match model.serializedIdentity,expectedIdentity with
    | some value,_ => .ok value
    | none,some value => .ok value
    | none,none => .error (.fieldFailure "recovery identity unavailable")
  let sos ← p.arrayJSON model.sos
  let envelope ← p.arrayJSON model.envelope
  let base := Fields.ofList
    [("schema",.text (if model.bidirectional then "n2p3_bidirectional_state_recovery/1"
       else "n2p3_epoch_state_recovery/2")),("sos",sos),("sample_rate_hz",.number model.rate),
     ("preprocessing",.object (preprocessingRecord model.preprocessing)),
     ("channel_names",.sequence (valuesOfList model.channels)),("modal_envelope",envelope),
     ("tolerance_v",.number model.tolerance),("fit_provenance",.object model.provenance),
     ("mode_identity",identity),("mode_order",.text (if model.bidirectional then
       "left_then_right_eigenvalue_real_then_imaginary_unit_vector_norm"
       else "eigenvalue_real_then_imaginary_unit_vector_norm")),
     ("interpretation",.text (if model.bidirectional then
       "conditional_on_calibration_empirical_effective_boundary_state_envelope"
       else "conditional_on_calibration_empirical_state_envelope"))]
  pure (if model.bidirectional then base else .cons "operator"
    (.text "actual_epoch_resample_crop_mean_baseline_modal_envelope") base)

def epochRecoveryFromRecord (p : ModalPrimitives) (value : Value) : Check EpochRecovery := do
  let model ← readEpochRecovery p value
  let record ← epochRecoveryRecord p model
  let actualHash ← p.recovery.input.execution.semanticHash (.object record)
  let savedHash ← p.recovery.input.execution.semanticHash value
  ensure (actualHash=savedHash) "recovery canonical semantic digest"
  pure model

structure ContinuousQC where
  spec : QCSpec
  channels : List Value
  rateValue : Value
  rate : ℝ
  center : Value
  scale : Value
  provenance : Fields
  recovery : Option StateBox
  epochRecovery : Option EpochRecovery
  ranges : Option Value

def readHashableTuple (value : Value) : Check (List Value) := do
  let values ← tupleValue value
  ensure (∀ value ∈ values, hashableJSON value) "hashable tuple members"
  pure values

def checkNestedRecovery (parent child : Fields) : Check Unit := do
  ensure (PyTruthy (fieldDefault parent "participant_key") ∧
    PersonalAdmission.PyEq (fieldDefault child "participant_key") (fieldDefault parent "participant_key"))
    "nested recovery participant authority"
  ensure (PyTruthy (fieldDefault parent "source_reference") ∧
    PersonalAdmission.PyEq (fieldDefault child "source_reference") (fieldDefault parent "source_reference"))
    "nested recovery source reference"
  let childSessionsValue := fieldDefault child "calibration_session_ids"
  let parentSessionsValue := fieldDefault parent "calibration_session_ids"
  ensure (PyTruthy childSessionsValue ∧ PyTruthy parentSessionsValue) "nested recovery nonempty calibration sessions"
  let childSessions ← readHashableTuple childSessionsValue
  let parentSessions ← readHashableTuple parentSessionsValue
  ensure (∀ value ∈ childSessions, ∃ saved ∈ parentSessions, PersonalAdmission.PyEq value saved)
    "nested recovery calibration session subset"
  let childEnd ← numeric (fieldDefault child "calibration_end_timestamp_s") "nested recovery child calibration end"
  let parentEnd ← numeric (fieldDefault parent "calibration_end_timestamp_s") "nested recovery parent calibration end"
  ensure (childEnd ≤ parentEnd) "nested recovery excludes future calibration"

def constructContinuousQC (p : ModalPrimitives) (value : Value) : Check ContinuousQC := do
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
  ensure (channels ≠ []) "nonempty continuous QC channels"
  ensure (∀ channel ∈ channels, hashableJSON channel) "hashable continuous QC channels"
  ensure ((channels.map PersonalAdmission.normalizeValue).Nodup) "unique continuous QC channels"
  let centerImage ← p.recovery.genericArray center
  let scaleImage ← p.recovery.genericArray scale
  ensure (centerImage.shape=[channels.length] ∧ scaleImage.shape=[channels.length]) "continuous QC center scale geometry"
  ensure (finiteArray centerImage ∧ finiteArray scaleImage ∧
    ∀ value ∈ scaleImage.values, ∃ number, atomNumber value=some number ∧ 0 < number)
    "finite continuous QC center and positive scale"
  let rate ← numeric rateValue "continuous QC finite sample rate"
  ensure (0 < rate ∧ PyTruthy provenanceValue) "positive continuous QC rate and fit provenance"
  let provenance ← objectAt provenanceValue "continuous QC fit provenance mapping"
  match epoch with
  | none => pure ()
  | some child => do
    ensure (spec.channelLocal=true ∧ recovery=none ∧
      PersonalAdmission.PyEq (.sequence (valuesOfList child.channels)) (.sequence (valuesOfList channels)) ∧ child.rate=rate)
      "exclusive matching native epoch recovery"
    checkNestedRecovery provenance child.provenance
  if spec.rangeWindow.isSome then
    let image ← p.recovery.input.execution.floatArray (ranges.getD .null)
    ensure (image.shape=[channels.length] ∧ finiteArray image ∧
      ∀ value ∈ image.values, ∃ number, atomNumber value=some number ∧ 0 < number)
      "continuous QC positive fitted range thresholds"
  pure ⟨spec,channels,rateValue,rate,center,scale,provenance,recovery,epoch,ranges⟩

def continuousLookahead (p : ModalPrimitives) (model : ContinuousQC) : Check ℤ := do
  let previous := (match model.spec.rangeWindow with
    | none => 0
    | some window => if fieldDefault model.spec.normalized "range_action"=.text "diagnostic" then 0 else window-1)
    + model.spec.preGuard
  let flatline ← match model.spec.flatlineDuration with
    | none => .ok 0
    | some duration => p.ceilProduct duration model.rate
  pure (max previous flatline)

def continuousQCRecord (p : ModalPrimitives) (model : ContinuousQC) : Check Fields := do
  let epochValue ← match model.epochRecovery with
    | none => .ok Value.null
    | some model => do
      let fields ← epochRecoveryRecord p model
      pure (.object fields)
  let lookahead ← continuousLookahead p model
  let policy := match model.epochRecovery with
    | some child => if fieldDefault child.preprocessing "filter_phase"=.text "zero" then
      "calibration_empirical_bidirectional_boundary_envelope" else "calibration_empirical_epoch_modal_envelope"
    | none => if model.recovery=none then "reject_unbounded_state_error" else "declared_state_box_modal_tail_bound"
  pure (Fields.ofList [("schema",.text (qcSpecSchema model.spec)),("spec",.object (qcSpecRecord model.spec)),
    ("channel_names",.sequence (valuesOfList model.channels)),("sample_rate_hz",model.rateValue),
    ("difference_center_v",model.center),("difference_scale_v",model.scale),
    ("fit_provenance",.object model.provenance),("recovery_policy",.text policy),
    ("recovery",match model.recovery with | none => .null | some box => .object box.record),
    ("epoch_recovery",epochValue),("range_threshold_v",model.ranges.getD .null),
    ("detection_lookahead_samples",.integer lookahead),("signal_unit",.text "V")])

def continuousQCFromRecord (p : ModalPrimitives) (value : Value) : Check ContinuousQC := do
  let model ← constructContinuousQC p value
  let record ← continuousQCRecord p model
  let actualHash ← p.recovery.input.execution.semanticHash (.object record)
  let savedHash ← p.recovery.input.execution.semanticHash value
  ensure (actualHash=savedHash) "continuous QC canonical semantic digest"
  pure model

theorem nested_recovery_future_end_rejected (childEnd parentEnd : ℝ) (h : parentEnd < childEnd) :
    ensure (childEnd ≤ parentEnd) "nested recovery excludes future calibration"=
      .error (.sourceReject "nested recovery excludes future calibration") := by
  simp [ensure,not_le.mpr h]

theorem modal_instability_rejected (radius : ℝ) (h : 1 ≤ radius) :
    ensure (radius < 1) "strictly stable modal system"=
      .error (.sourceReject "strictly stable modal system") := by
  simp [ensure,not_lt.mpr h]

end

end N2P3.PersonalERPModels
