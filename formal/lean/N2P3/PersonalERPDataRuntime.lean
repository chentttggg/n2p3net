import N2P3.PersonalERPModelRuntime

/-!
# Actual preprocessing scalar/type/set order before the physical dataset tail

Source: epochs.py:100-238; contract.py:34-61. This does not insert separate
scalar isfinite calls where source executes one tuple/list isfinite(...).all().
Notch checks use math.isfinite, not NumPy. Window floats are converted before
their joint finite test. All three optional-band type guards precede the
first optional finite test. The remaining dataset field checks use their
explicit decoded continuation, never an assumed successful dataset validator.
IO/array representations and inherited faults remain separately qualified.
-/

namespace N2P3.PersonalERPDataRuntime

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPModels N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  input : PersonalERPModelRuntime.Input
  mathFinite : Value → Except AtomicException Bool
  numpyTupleFiniteAll : List Value → Except AtomicException Bool
  numpyFloatFiniteAll : List Atom → Except AtomicException Bool
  floatScalar : Value → R Atom
  divide : Value → Value → R ℝ
  floorSamples : Value → Value → Value → R ℤ
  windowSamples : ℝ → ℝ → Value → R Bool

def realType (value : Value) (site : String) : R Unit := require (numericNonBool value=true) site

def mathematicalFinite (p : Primitives) (value : Value) (site : String) : R Unit := do
  let finite ← (p.mathFinite value).mapError (Fault.atomic site .mathFinite)
  require (finite=true) site

def tupleFinite (p : Primitives) (values : List Value) (site : String) : R Unit := do
  let finite ← (p.numpyTupleFiniteAll values).mapError (Fault.atomic site .numpyFinite)
  require (finite=true) site

def floatTupleFinite (p : Primitives) (values : List Atom) (site : String) : R Unit := do
  let finite ← (p.numpyFloatFiniteAll values).mapError (Fault.atomic site .numpyFinite)
  require (finite=true) site

def notchValues (p : Primitives) : List Value → R (List ℝ)
  | [] => .ok []
  | value :: tail => do
    realType value "notch frequency numeric nonbool type"
    mathematicalFinite p value "notch frequency math finite"
    let number ← PersonalERPModelRuntime.numeric value "notch frequency comparison representation"
    require (0 < number) "positive notch frequency"
    let rest ← notchValues p tail
    pure (number::rest)

def checkNotches (p : Primitives) (frequencies quality : Value) (rate : Option Value) : R (List ℝ × ℝ) := do
  let values ← match frequencies with
    | .sequence values => .ok (valuesToList values)
    | _ => .error (.sourceValueError "notch frequencies list or tuple")
  let numbers ← notchValues p values
  require ((values.map PersonalAdmission.normalizeValue).Nodup) "distinct notch frequencies"
  realType quality "notch quality numeric nonbool type"
  mathematicalFinite p quality "notch quality math finite"
  let qualityNumber ← PersonalERPModelRuntime.numeric quality "notch quality comparison representation"
  require (0 < qualityNumber) "positive notch quality"
  match rate with
  | none => pure ()
  | some rate =>
    mathematicalFinite p rate "notch source rate math finite"
    let rateNumber ← PersonalERPModelRuntime.numeric rate "notch source rate comparison representation"
    require (0 < rateNumber) "positive notch source rate"
    let nyquist ← p.divide rate (.integer 2)
    require (∀ frequency ∈ numbers, frequency < nyquist) "notches below actual source Nyquist"
  pure (numbers,qualityNumber)

def checkEdge (policy method phase : Value) : R Unit := do
  require (∃ text, policy=.text text) "edge policy string type"
  setOption "zero phase edge policy" [.text "legacy",.text "mne_reflect_limited"] policy
  require (policy=.text "legacy" ∨ (method=.text "iir" ∧ phase=.text "zero")) "edge policy requires zero IIR"

def checkWindow (p : Primitives) (fields : Fields) (tmin tmax : ℝ) : R Unit := do
  setOption "reference center" [.text "mean",.text "median"] (fieldDefault fields "trial_reference_center")
  setOption "reference scale" [.text "none",.text "std",.text "mad"] (fieldDefault fields "trial_reference_scale")
  if fieldDefault fields "baseline_mode"=.text "trial_reference" then
    let value := fieldDefault fields "trial_reference_window_ms"
    require (value ≠ .null) "reference window required"
    let values ← PersonalERPModelRuntime.tupleValue value
    require (values.length=2) "reference window length two"
    let endpoints ← match values with
      | [start,stop] => .ok (start,stop)
      | _ => .error (.sourceValueError "reference window length two")
    realType endpoints.1 "reference window first numeric nonbool type"
    realType endpoints.2 "reference window second numeric nonbool type"
    let start ← p.floatScalar endpoints.1
    let stop ← p.floatScalar endpoints.2
    floatTupleFinite p [start,stop] "converted reference window finite all"
    let start ← PersonalERPModelRuntime.strictReal start
    let stop ← PersonalERPModelRuntime.strictReal stop
    require (start < stop) "increasing converted reference window"
    require (tmin ≤ start ∧ stop ≤ tmax) "converted reference window inside physical epoch"
    let enough ← p.windowSamples start stop (fieldDefault fields "sfreq")
    require (enough=true) "at least two actual reference window samples"

def optionalType (fields : Fields) (key : String) : R Unit :=
  require (fieldDefault fields key=.null ∨ numericNonBool (fieldDefault fields key)=true) (key ++ " numeric or None")

def optionalPositive (p : Primitives) (fields : Fields) (key : String) : R (Option ℝ) := do
  let value := fieldDefault fields key
  if value=.null then pure none
  else
    let number ← numpyNumeric p.input.finite value (key ++ " NumPy finite")
    require (0 < number) (key ++ " positive")
    pure (some number)

def checkPreprocessing (p : Primitives) (fields : Fields) : R DeclarationScalars := do
  let fields ← liftCheck (readPreprocessingArgs fields)
  let value := fieldDefault fields
  let notches ← checkNotches p (value "notch_freqs_hz") (value "notch_quality_factor") none
  checkEdge (value "zero_phase_edge_policy") (value "filter_method") (value "filter_phase")
  let name ← match value "name" with
    | .text name => .ok name
    | _ => .error (.sourceValueError "preprocessing name string type")
  require (p.input.original.execution.strip name ≠ "") "nonempty preprocessing name"
  realType (value "sfreq") "sfreq numeric nonbool type"
  realType (value "tmin_ms") "tmin numeric nonbool type"
  realType (value "tmax_ms") "tmax numeric nonbool type"
  tupleFinite p [value "sfreq",value "tmin_ms",value "tmax_ms"] "physical time tuple NumPy finite all"
  let sfreq ← PersonalERPModelRuntime.numeric (value "sfreq") "sfreq comparison representation"
  let tmin ← PersonalERPModelRuntime.numeric (value "tmin_ms") "tmin comparison representation"
  let tmax ← PersonalERPModelRuntime.numeric (value "tmax_ms") "tmax comparison representation"
  let nTimes ← PersonalERPModelRuntime.integer (value "n_times") "nonbool integer preprocessing sample count"
  require (0 < sfreq ∧ 0 < nTimes) "positive preprocessing rate and sample count"
  require (tmin < tmax) "increasing epoch endpoints"
  let expected ← p.floorSamples (value "tmin_ms") (value "tmax_ms") (value "sfreq")
  require (nTimes=expected) "exclusive right physical endpoint sample count"
  setOption "baseline mode" [.text "trial",.text "mean_only",.text "none",.text "trial_reference"] (value "baseline_mode")
  require (value "signal_unit"=.text "V") "physical voltage unit"
  require (value "filter_method"=.text "iir" ∧ PersonalAdmission.PyEq (value "filter_order") (.integer 4)) "fourth order IIR method order"
  setOption "filter phase" [.text "zero",.text "forward"] (value "filter_phase")
  if value "filter_phase"=.text "forward" then
    require (value "causal_iir_initial_state"=.text "steady_state_first_sample") "forward initial state"
  else require (value "causal_iir_initial_state"=.text "not_applicable") "zero phase initial state"
  require (value "resample_domain"=.text "epoched") "epoched resampling domain"
  require (value "resample_method"=.text "fft" ∧ value "resample_npad"=.text "auto" ∧
    value "resample_window"=.text "auto" ∧ value "resample_pad"=.text "edge") "FFT resampling declaration"
  require (value "baseline_mode" ≠ .text "trial" ∧
    (value "baseline_mode" ≠ .text "trial_reference" ∨ value "trial_reference_scale"=.text "none")) "no per trial voltage scaling"
  require ((value "baseline_mode"=.text "trial" ∨ value "baseline_mode"=.text "mean_only") → tmin < 0) "prestimulus baseline"
  checkWindow p fields tmin tmax
  optionalType fields "l_freq"
  optionalType fields "h_freq"
  optionalType fields "reject_threshold_v"
  let low ← optionalPositive p fields "l_freq"
  let high ← optionalPositive p fields "h_freq"
  require (∀ l ∈ low, ∀ h ∈ high, l < h) "ordered passband"
  let nyquist ← p.divide (value "sfreq") (.number 2)
  require (∀ l ∈ low, l < nyquist) "high pass below actual model Nyquist"
  require (∀ h ∈ high, h < nyquist) "low pass below actual model Nyquist"
  let reject ← optionalPositive p fields "reject_threshold_v"
  pure ⟨sfreq,tmin,tmax,nTimes,low,high,reject,notches.1,notches.2⟩

def checkDatasetBody (p : Primitives) (fields : Fields) {N C T M : ℕ} (data : DatasetImage N C T M)
    (requireLabels : Bool) : R DeclarationScalars := do
  let fields ← liftCheck (readPreprocessingArgs fields)
  let declaration ← checkPreprocessing p fields
  liftCheck (checkDatasetBodyAfterDeclaration p.input.original.execution.strip p.input.original.canonicalChannel
    p.input.original.coordinateNorm (rawPreprocessing fields) declaration data requireLabels)
  pure declaration

theorem physical_tuple_failure_preserves_the_single_array_operation (p : Primitives) (values : List Value)
    (site : String) (error : AtomicException) (h : p.numpyTupleFiniteAll values=.error error) :
    tupleFinite p values site=.error (.atomic site .numpyFinite error) := by simp [tupleFinite,h]

theorem notch_math_failure_is_not_a_numpy_test (p : Primitives) (value : Value) (site : String)
    (error : AtomicException) (h : p.mathFinite value=.error error) :
    mathematicalFinite p value site=.error (.atomic site .mathFinite error) := by simp [mathematicalFinite,h]

end
end N2P3.PersonalERPDataRuntime
