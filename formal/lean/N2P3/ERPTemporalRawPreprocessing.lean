import N2P3.ERPTemporalRawFields
import N2P3.PersonalERPDataRuntime

/-!
# PreprocessingSpec.validate on original decoded JSON fields

Source: epochs.py:100-211; contract.py:34-61. The input is the complete
field image of an already constructed Spec. This is the declaration layer,
not a competing full Dataset validator. Every original field is returned
unchanged, including an unused nonfinite reference window. It is not finite
projected before the source reaches its actual guards or later JSON hash.

Single math/NumPy finite tests, float conversion and the literal numerical
expressions remain atomic boundaries with their own exceptions. Incoherent
native-success results outside the finite numerical representation are
explicitly unrefined; they never become a new early source rejection.
-/

namespace N2P3.ERPTemporalRawPreprocessing

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.PersonalERPScalar N2P3.ERPTemporalJSON

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev R := PersonalERPScalar.Result

structure Operations where
  strip : String → String
  mathFinite : JValue → Except AtomicException Bool
  numpyFinite : JValue → Except AtomicException Bool
  numpyTupleFiniteAll : List JValue → Except AtomicException Bool
  numpyFloatFiniteAll : List ERPTemporalMultiScope.FloatAtom → Except AtomicException Bool
  floatScalar : JValue → R ERPTemporalMultiScope.FloatAtom
  divide : JValue → JValue → R ℝ
  floorSamples : JValue → JValue → JValue → R ℤ
  windowSamples : ℝ → ℝ → JValue → R Bool

def numericNonBool : JValue → Bool
  | .integer _ | .floating _ | .negativeZero => true
  | _ => false

def numeric (value : JValue) (site : String) : R ℝ :=
  match value with
  | .integer value => .ok value
  | .floating (.finite value) => .ok value
  | .negativeZero => .ok 0
  | _ => .error (.unrefined (site ++ " finite native scalar representation"))

def realType (value : JValue) (site : String) : R Unit :=
  require (numericNonBool value=true) site

def mathFinite (p : Operations) (value : JValue) (site : String) : R Unit := do
  let finite ← (p.mathFinite value).mapError (Fault.atomic site .mathFinite)
  require (finite=true) site

def numpyNumber (p : Operations) (value : JValue) (site : String) : R ℝ := do
  let finite ← (p.numpyFinite value).mapError (Fault.atomic site .numpyFinite)
  require (finite=true) site
  numeric value site

def setOption (site : String) (allowed : List JValue) (value : JValue) : R Unit := do
  match value with
  | .sequence _ | .object _ => .error (.atomic site .setMembership .typeError)
  | _ => require (∃ option ∈ allowed, pythonEqual value option) site

def notchValues (p : Operations) : List JValue → R (List ℝ)
  | [] => .ok []
  | value :: tail => do
    realType value "notch frequency numeric nonbool type"
    mathFinite p value "notch frequency math finite"
    let value ← numeric value "notch frequency comparison"
    require (0 < value) "positive notch frequency"
    let rest ← notchValues p tail
    pure (value :: rest)

def checkNotches (p : Operations) (frequencies quality : JValue) : R (List ℝ × ℝ) := do
  let values ← match frequencies with
    | .sequence values => pure values.toList
    | _ => .error (.sourceValueError "notch frequencies list or tuple")
  let numbers ← notchValues p values
  require ((values.map ERPTemporalJSON.normalize).Nodup) "distinct notch frequencies"
  realType quality "notch quality numeric nonbool type"
  mathFinite p quality "notch quality math finite"
  let quality ← numeric quality "notch quality comparison"
  require (0 < quality) "positive notch quality"
  pure (numbers,quality)

def checkEdge (policy method phase : JValue) : R Unit := do
  require (∃ text, policy=.text text) "edge policy string type"
  setOption "zero phase edge policy" [.text "legacy",.text "mne_reflect_limited"] policy
  require (pythonEqual policy (.text "legacy") ∨
    (pythonEqual method (.text "iir") ∧ pythonEqual phase (.text "zero"))) "edge policy requires zero IIR"

def sequenceLength (value : JValue) : R ℕ :=
  match value with
  | .sequence values => .ok values.toList.length
  | .object fields => .ok fields.keys.length
  | .text value => .ok value.length
  | _ => .error (.inherited (.fieldFailure "len() of reference window"))

def iterableWindow (value : JValue) : R (List JValue) :=
  match value with
  | .sequence values => .ok values.toList
  | .object fields => .ok (fields.keys.map JValue.text)
  | .text value => .ok (value.toList.map (fun char => .text (String.singleton char)))
  | _ => .error (.inherited (.fieldFailure "iterate reference window"))

def floatNumber : ERPTemporalMultiScope.FloatAtom → R ℝ
  | .finite value => .ok value
  | _ => .error (.unrefined "finite converted reference endpoint representation")

def checkWindow (p : Operations) (fields : JFields) (tmin tmax : ℝ) : R Unit := do
  let value := ERPTemporalRawFields.fieldDefault fields
  setOption "reference center" [.text "mean",.text "median"] (value "trial_reference_center")
  setOption "reference scale" [.text "none",.text "std",.text "mad"] (value "trial_reference_scale")
  if pythonEqual (value "baseline_mode") (.text "trial_reference") then
    let window := value "trial_reference_window_ms"
    require (window ≠ .null) "reference window required"
    require ((← sequenceLength window)=2) "reference window length two"
    let endpoints ← iterableWindow window
    let _ ← endpoints.mapM (fun endpoint => realType endpoint "reference window numeric nonbool type")
    let endpoints ← match endpoints with
      | [start,stop] => pure (start,stop)
      | _ => .error (.unrefined "reference length versus iteration correspondence")
    let start ← p.floatScalar endpoints.1
    let stop ← p.floatScalar endpoints.2
    let finite ← (p.numpyFloatFiniteAll [start,stop]).mapError
      (Fault.atomic "converted reference window finite all" .numpyFinite)
    require (finite=true) "converted reference window finite all"
    let start ← floatNumber start
    let stop ← floatNumber stop
    require (start < stop) "increasing converted reference window"
    require (tmin ≤ start ∧ stop ≤ tmax) "converted reference window inside physical epoch"
    require ((← p.windowSamples start stop (value "sfreq"))=true) "at least two actual reference samples"

def optionalType (fields : JFields) (key : String) : R Unit :=
  let value := ERPTemporalRawFields.fieldDefault fields key
  require (value=.null ∨ numericNonBool value=true) (key ++ " numeric or None")

def optionalPositive (p : Operations) (fields : JFields) (key : String) : R (Option ℝ) := do
  let value := ERPTemporalRawFields.fieldDefault fields key
  if value=.null then pure none
  else
    let number ← numpyNumber p value (key ++ " NumPy finite")
    require (0 < number) (key ++ " positive")
    pure (some number)

structure Declaration where
  original : JFields
  scalars : DeclarationScalars

def validateScalars (p : Operations) (fields : JFields) : R DeclarationScalars := do
  let value := ERPTemporalRawFields.fieldDefault fields
  let notches ← checkNotches p (value "notch_freqs_hz") (value "notch_quality_factor")
  checkEdge (value "zero_phase_edge_policy") (value "filter_method") (value "filter_phase")
  let name ← match value "name" with
    | .text name => pure name
    | _ => .error (.sourceValueError "preprocessing name string type")
  require (p.strip name ≠ "") "nonempty preprocessing name"
  realType (value "sfreq") "sfreq numeric nonbool type"
  realType (value "tmin_ms") "tmin numeric nonbool type"
  realType (value "tmax_ms") "tmax numeric nonbool type"
  let finite ← (p.numpyTupleFiniteAll [value "sfreq",value "tmin_ms",value "tmax_ms"]).mapError
    (Fault.atomic "physical time tuple NumPy finite all" .numpyFinite)
  require (finite=true) "physical time tuple NumPy finite all"
  let sfreq ← numeric (value "sfreq") "sfreq comparison"
  let tmin ← numeric (value "tmin_ms") "tmin comparison"
  let tmax ← numeric (value "tmax_ms") "tmax comparison"
  let nTimes ← match value "n_times" with
    | .integer times => pure times
    | _ => .error (.sourceValueError "nonbool integer preprocessing sample count")
  require (0 < sfreq ∧ 0 < nTimes) "positive preprocessing rate and sample count"
  require (tmin < tmax) "increasing epoch endpoints"
  require (nTimes=(← p.floorSamples (value "tmin_ms") (value "tmax_ms") (value "sfreq")))
    "exclusive right physical endpoint sample count"
  setOption "baseline mode" [.text "trial",.text "mean_only",.text "none",.text "trial_reference"] (value "baseline_mode")
  require (pythonEqual (value "signal_unit") (.text "V")) "physical voltage unit"
  require (pythonEqual (value "filter_method") (.text "iir") ∧
    pythonEqual (value "filter_order") (.integer 4)) "fourth order IIR method order"
  setOption "filter phase" [.text "zero",.text "forward"] (value "filter_phase")
  if pythonEqual (value "filter_phase") (.text "forward") then
    require (pythonEqual (value "causal_iir_initial_state") (.text "steady_state_first_sample")) "forward initial state"
  else require (pythonEqual (value "causal_iir_initial_state") (.text "not_applicable")) "zero phase initial state"
  require (pythonEqual (value "resample_domain") (.text "epoched")) "epoched resampling domain"
  require (pythonEqual (value "resample_method") (.text "fft") ∧
    pythonEqual (value "resample_npad") (.text "auto") ∧ pythonEqual (value "resample_window") (.text "auto") ∧
    pythonEqual (value "resample_pad") (.text "edge")) "FFT resampling declaration"
  require (¬pythonEqual (value "baseline_mode") (.text "trial") ∧
    (¬pythonEqual (value "baseline_mode") (.text "trial_reference") ∨
      pythonEqual (value "trial_reference_scale") (.text "none"))) "no per trial voltage scaling"
  require ((pythonEqual (value "baseline_mode") (.text "trial") ∨
    pythonEqual (value "baseline_mode") (.text "mean_only")) → tmin < 0) "prestimulus baseline"
  checkWindow p fields tmin tmax
  optionalType fields "l_freq"
  optionalType fields "h_freq"
  optionalType fields "reject_threshold_v"
  let low ← optionalPositive p fields "l_freq"
  let high ← optionalPositive p fields "h_freq"
  require (∀ l ∈ low, ∀ h ∈ high, l < h) "ordered passband"
  let nyquist ← p.divide (value "sfreq") (.floating (.finite 2))
  require (∀ l ∈ low, l < nyquist) "high pass below actual model Nyquist"
  require (∀ h ∈ high, h < nyquist) "low pass below actual model Nyquist"
  let reject ← optionalPositive p fields "reject_threshold_v"
  pure ⟨sfreq,tmin,tmax,nTimes,low,high,reject,notches.1,notches.2⟩

def validate (p : Operations) (fields : JFields) : R Declaration :=
  (validateScalars p fields).map (fun scalars => ⟨fields,scalars⟩)

theorem decoded_nan_is_numeric_before_its_finite_guard :
    numericNonBool (.floating .nan)=true := rfl

theorem nonreference_mode_ignores_unused_window (p : Operations) (fields : JFields)
    (tmin tmax : ℝ)
    (center : ERPTemporalRawFields.fieldDefault fields "trial_reference_center"=.text "mean")
    (scale : ERPTemporalRawFields.fieldDefault fields "trial_reference_scale"=.text "none")
    (baseline : ¬pythonEqual (ERPTemporalRawFields.fieldDefault fields "baseline_mode") (.text "trial_reference")) :
    checkWindow p fields tmin tmax=.ok () := by
  have centerOK : setOption "reference center" [.text "mean",.text "median"] (.text "mean")=.ok () := by
    simp [setOption,require,pythonEqual,scalarNaN]
  have scaleOK : setOption "reference scale" [.text "none",.text "std",.text "mad"] (.text "none")=.ok () := by
    simp [setOption,require,pythonEqual,scalarNaN]
  simp [checkWindow,center,scale,centerOK,scaleOK,baseline]

theorem raw_optional_nan_passes_the_type_guard (fields : JFields) (key : String)
    (nan : ERPTemporalRawFields.fieldDefault fields key=.floating .nan) :
    optionalType fields key=.ok () := by
  simp [optionalType,nan,numericNonBool,require]

theorem raw_optional_nan_native_failure_keeps_its_operation (p : Operations) (fields : JFields)
    (key : String) (exception : AtomicException)
    (nan : ERPTemporalRawFields.fieldDefault fields key=.floating .nan)
    (failed : p.numpyFinite (.floating .nan)=.error exception) :
    optionalPositive p fields key=.error (.atomic (key ++ " NumPy finite") .numpyFinite exception) := by
  simp [optionalPositive,nan,numpyNumber,failed]

theorem Boolean_notch_refused_before_quality (p : Operations) (quality : JValue) (flag : Bool) :
    checkNotches p (.sequence (.cons (.boolean flag) .nil)) quality=
      .error (.sourceValueError "notch frequency numeric nonbool type") := by
  simp [checkNotches,JValues.toList,notchValues,realType,numericNonBool,require]

theorem successful_declaration_keeps_every_original_field (p : Operations) (fields : JFields)
    (declaration : Declaration) (success : validate p fields=.ok declaration) :
    declaration.original=fields := by
  unfold validate at success
  cases checked : validateScalars p fields <;> simp_all
  exact (congrArg Declaration.original success).symm

end
end N2P3.ERPTemporalRawPreprocessing
