import N2P3.ERPState
import N2P3.PersonalAdmission
import Mathlib.Data.List.FinRange

/-!
# Explicit physical declarations at personal ERP ingress

Source: epochs.py:56-246,325-471,1550-1561; contract.py:34-61.
This file refines individual fields rather than calling a supplied dataset or
preprocessing validator. Numeric atoms preserve nonfinite and bool/type cases.
The two scalar operations below are the literal Python/NumPy floating
arithmetic expressions; their implementation is not replaced by real floor.
The field layer does not certify physical producer honesty or actual bytes.
Full P06 remains pending until the dataset, execution, raw and caller layers
are connected. No default-valid personal model is constructed here.
-/

namespace N2P3.PersonalERPData

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Atom where
  | boolean (value : Bool)
  | integer (value : ℤ)
  | real (value : ℝ)
  | complex (real imag : ℝ)
  | nan | positiveInfinity | negativeInfinity | null | other
  deriving DecidableEq

def atomFromValue : Value → Atom
  | .boolean b => .boolean b
  | .integer z => .integer z
  | .number r => .real r
  | .null => .null
  | _ => .other

def strictReal : Atom → Check ℝ
  | .integer z => .ok z
  | .real r => .ok r
  | _ => .error (.fieldFailure "finite nonbool Real")

def strictInteger : Atom → Check ℤ
  | .integer z => .ok z
  | _ => .error (.fieldFailure "nonbool integer")

def optionalReal : Atom → Check (Option ℝ)
  | .null => .ok none
  | a => (strictReal a).map some

theorem bool_is_not_declared_real (b : Bool) :
    strictReal (.boolean b)=.error (.fieldFailure "finite nonbool Real") := rfl

theorem real_integer_is_not_integer_type (r : ℝ) :
    strictInteger (.real r)=.error (.fieldFailure "nonbool integer") := rfl

theorem nonfinite_scalar_is_not_valid :
    strictReal .nan=.error (.fieldFailure "finite nonbool Real") ∧
    strictReal .positiveInfinity=.error (.fieldFailure "finite nonbool Real") ∧
    strictReal .negativeInfinity=.error (.fieldFailure "finite nonbool Real") := ⟨rfl,rfl,rfl⟩

def readRealList : List Atom → Check (List ℝ)
  | [] => .ok []
  | a :: tail => do
    let r ← strictReal a
    let rs ← readRealList tail
    pure (r :: rs)

def readNotches (frequencies : Option (List Atom)) (quality : Atom) (sourceRate : Option ℝ) :
    Check (List ℝ × ℝ) := do
  let atoms ← match frequencies with
    | some atoms => .ok atoms
    | none => .error (.fieldFailure "notch list or tuple")
  let values ← readRealList atoms
  ensure (values.Nodup ∧ ∀ value ∈ values, 0 < value) "distinct positive finite notch frequencies"
  let q ← strictReal quality
  ensure (0 < q) "positive finite notch quality"
  match sourceRate with
  | none => pure (values,q)
  | some rate =>
    ensure (0 < rate) "positive finite notch source rate"
    ensure (∀ value ∈ values, value < rate/2) "notch source Nyquist"
    pure (values,q)

def checkEdgePolicy (policy method phase : Value) : Check Unit := do
  ensure (policy=.text "legacy" ∨ policy=.text "mne_reflect_limited") "zero phase edge policy"
  ensure (policy=.text "legacy" ∨ (method=.text "iir" ∧ phase=.text "zero")) "edge policy requires zero IIR"

inductive WindowInput where
  | null
  | sequence (values : List Atom)
  | other (value : Value)

structure RawPreprocessing where
  name : Value
  sfreq : Atom
  lFreq : Atom
  hFreq : Atom
  tmin : Atom
  tmax : Atom
  nTimes : Atom
  baseline : Value
  unit : Value
  filterMethod : Value
  filterOrder : Value
  phase : Value
  initialState : Value
  notchFrequencies : Option (List Atom)
  notchQuality : Atom
  edgePolicy : Value
  resampleDomain : Value
  resampleMethod : Value
  resampleNpad : Value
  resampleWindow : Value
  resamplePad : Value
  referenceWindow : WindowInput
  referenceCenter : Value
  referenceScale : Value
  rejectThreshold : Atom

/-- Exactly floor(((tmax-tmin)*sfreq/1000)+1e-9), with Python floating
subtraction/multiplication/division/addition, np.floor and int conversion. -/
abbrev FloorSampleCount := ℝ → ℝ → ℝ → Check ℤ

/-- Exactly the source guard not ((end-start)*sfreq/1000 < 2), including
floating nonfinite comparison behavior, not an assumed finite real result. -/
abbrev WindowSampleCount := ℝ → ℝ → ℝ → Check Bool

structure DeclarationScalars where
  sfreq : ℝ
  tmin : ℝ
  tmax : ℝ
  nTimes : ℤ
  lFreq : Option ℝ
  hFreq : Option ℝ
  rejectThreshold : Option ℝ
  notches : List ℝ
  notchQuality : ℝ

def preprocessingDefaults : Fields := Fields.ofList
  [("name",.text "p300_ms_eegnet_input_v3"),("sfreq",.number 128),("l_freq",.number (1/10)),
   ("h_freq",.number 30),("tmin_ms",.number (-200)),("tmax_ms",.number 1200),("n_times",.integer 179),
   ("baseline_mode",.text "mean_only"),("signal_unit",.text "V"),("filter_method",.text "iir"),
   ("filter_order",.integer 4),("filter_phase",.text "zero"),("causal_iir_initial_state",.text "not_applicable"),
   ("notch_freqs_hz",.sequence .nil),("notch_quality_factor",.number 30),("zero_phase_edge_policy",.text "legacy"),
   ("resample_domain",.text "epoched"),("resample_method",.text "fft"),("resample_npad",.text "auto"),
   ("resample_window",.text "auto"),("resample_pad",.text "edge"),("trial_reference_window_ms",.null),
   ("trial_reference_center",.text "mean"),("trial_reference_scale",.text "none"),("reject_threshold_v",.null)]

def preprocessingKeywords : List String := fieldKeys preprocessingDefaults

/-- Dataclass kwargs reconstruction alone does not call validate(). In
particular the constructor with absent execution evidence must not acquire
new validation that the source does not perform. JSON lists already encode
the tuple/list normalization relevant to later JSON comparison. -/
def readPreprocessingArgs (fields : Fields) : Check Fields := do
  ensure (fieldKeys fields ⊆ preprocessingKeywords) "unknown preprocessing keyword"
  pure (Fields.ofList (preprocessingKeywords.map fun key =>
    (key,(fields.get key).getD (fieldDefault preprocessingDefaults key))))

def rawPreprocessing (fields : Fields) : RawPreprocessing :=
  let value := fieldDefault fields
  { name := value "name", sfreq := atomFromValue (value "sfreq"), lFreq := atomFromValue (value "l_freq"),
    hFreq := atomFromValue (value "h_freq"), tmin := atomFromValue (value "tmin_ms"),
    tmax := atomFromValue (value "tmax_ms"), nTimes := atomFromValue (value "n_times"),
    baseline := value "baseline_mode", unit := value "signal_unit", filterMethod := value "filter_method",
    filterOrder := value "filter_order", phase := value "filter_phase", initialState := value "causal_iir_initial_state",
    notchFrequencies := match value "notch_freqs_hz" with
      | .sequence values => some ((valuesToList values).map atomFromValue)
      | _ => none,
    notchQuality := atomFromValue (value "notch_quality_factor"), edgePolicy := value "zero_phase_edge_policy",
    resampleDomain := value "resample_domain", resampleMethod := value "resample_method",
    resampleNpad := value "resample_npad", resampleWindow := value "resample_window", resamplePad := value "resample_pad",
    referenceWindow := match value "trial_reference_window_ms" with
      | .null => .null
      | .sequence values => .sequence ((valuesToList values).map atomFromValue)
      | other => .other other,
    referenceCenter := value "trial_reference_center", referenceScale := value "trial_reference_scale",
    rejectThreshold := atomFromValue (value "reject_threshold_v") }

def preprocessingRecord (normalizedArgs : Fields) : Fields :=
  let withoutEdge := if fieldDefault normalizedArgs "zero_phase_edge_policy"=.text "legacy"
    then normalizedArgs.erase "zero_phase_edge_policy" else normalizedArgs
  if fieldDefault normalizedArgs "notch_freqs_hz"=.sequence .nil ∧
      PersonalAdmission.PyEq (fieldDefault normalizedArgs "notch_quality_factor") (.number 30) then
    (withoutEdge.erase "notch_freqs_hz").erase "notch_quality_factor"
  else withoutEdge

def completePreprocessingKeys : List String :=
  preprocessingKeywords.filter fun key => key ≠ "notch_freqs_hz" && key ≠ "notch_quality_factor" &&
    key ≠ "zero_phase_edge_policy"

def readCompletePreprocessingArgs (fields : Fields) : Check Fields := do
  ensure (completePreprocessingKeys ⊆ fieldKeys fields) "complete preprocessing declaration"
  ensure ((fields.get "notch_freqs_hz").isSome=(fields.get "notch_quality_factor").isSome)
    "explicit notch frequencies and quality pair"
  readPreprocessingArgs fields

theorem omitted_notches_preserve_historical_none :
    (rawPreprocessing preprocessingDefaults).notchFrequencies=some [] ∧
    (rawPreprocessing preprocessingDefaults).edgePolicy=.text "legacy" := ⟨rfl,rfl⟩

theorem default_declaration_is_not_the_product_notch_profile :
    (rawPreprocessing preprocessingDefaults).notchFrequencies ≠ some [.real 50] := by
  change (some [] : Option (List Atom)) ≠ some [.real 50]
  intro h
  cases h

def checkReferenceWindow (samples : WindowSampleCount) (raw : RawPreprocessing)
    (sfreq tmin tmax : ℝ) : Check Unit := do
  ensure (raw.referenceCenter=.text "mean" ∨ raw.referenceCenter=.text "median") "reference center"
  ensure (raw.referenceScale=.text "none" ∨ raw.referenceScale=.text "std" ∨
    raw.referenceScale=.text "mad") "reference scale"
  if raw.baseline=.text "trial_reference" then
    let atoms ← match raw.referenceWindow with
      | .sequence [start,stop] => .ok (start,stop)
      | _ => .error (.fieldFailure "two reference window endpoints")
    let start ← strictReal atoms.1
    let stop ← strictReal atoms.2
    ensure (start < stop) "increasing reference window"
    ensure (tmin ≤ start ∧ stop ≤ tmax) "reference window inside epoch"
    let enough ← samples start stop sfreq
    ensure (enough=true) "at least two reference samples"
  else pure ()

def checkPreprocessing (strip : String → String) (floorSamples : FloorSampleCount) (windowSamples : WindowSampleCount)
    (raw : RawPreprocessing) : Check DeclarationScalars := do
  let notches ← readNotches raw.notchFrequencies raw.notchQuality none
  checkEdgePolicy raw.edgePolicy raw.filterMethod raw.phase
  let name ← match raw.name with
    | .text name => .ok name
    | _ => .error (.fieldFailure "preprocessing name")
  ensure (strip name ≠ "") "nonempty preprocessing name"
  let sfreq ← strictReal raw.sfreq
  let tmin ← strictReal raw.tmin
  let tmax ← strictReal raw.tmax
  let nTimes ← strictInteger raw.nTimes
  ensure (0 < sfreq ∧ 0 < nTimes) "positive preprocessing rate and sample count"
  ensure (tmin < tmax) "increasing epoch endpoints"
  let expected ← floorSamples tmin tmax sfreq
  ensure (nTimes=expected) "exclusive right endpoint sample count"
  ensure (raw.baseline=.text "trial" ∨ raw.baseline=.text "mean_only" ∨
    raw.baseline=.text "none" ∨ raw.baseline=.text "trial_reference") "baseline mode"
  ensure (raw.unit=.text "V") "physical voltage unit"
  ensure (raw.filterMethod=.text "iir" ∧ PersonalAdmission.PyEq raw.filterOrder (.integer 4) ∧
    (raw.phase=.text "zero" ∨ raw.phase=.text "forward")) "fourth order IIR"
  if raw.phase=.text "forward" then
    ensure (raw.initialState=.text "steady_state_first_sample") "forward initial state"
  else ensure (raw.initialState=.text "not_applicable") "zero phase initial state"
  ensure (raw.resampleDomain=.text "epoched") "epoched resampling domain"
  ensure (raw.resampleMethod=.text "fft" ∧ raw.resampleNpad=.text "auto" ∧
    raw.resampleWindow=.text "auto" ∧ raw.resamplePad=.text "edge") "FFT resampling declaration"
  ensure (raw.baseline ≠ .text "trial" ∧
    (raw.baseline ≠ .text "trial_reference" ∨ raw.referenceScale=.text "none")) "no per trial voltage scaling"
  ensure ((raw.baseline=.text "trial" ∨ raw.baseline=.text "mean_only") → tmin < 0) "prestimulus baseline"
  checkReferenceWindow windowSamples raw sfreq tmin tmax
  let lFreq ← optionalReal raw.lFreq
  let hFreq ← optionalReal raw.hFreq
  let reject ← optionalReal raw.rejectThreshold
  ensure (∀ value ∈ lFreq, 0 < value) "positive optional high pass"
  ensure (∀ value ∈ hFreq, 0 < value) "positive optional low pass"
  ensure (∀ low ∈ lFreq, ∀ high ∈ hFreq, low < high) "ordered passband"
  ensure (∀ value ∈ lFreq, value < sfreq/2) "high pass below model Nyquist"
  ensure (∀ value ∈ hFreq, value < sfreq/2) "low pass below model Nyquist"
  ensure (∀ value ∈ reject, 0 < value) "positive optional rejection threshold"
  pure ⟨sfreq,tmin,tmax,nTimes,lFreq,hFreq,reject,notches.1,notches.2⟩

theorem wrong_unit_fails_at_voltage_gate (raw : RawPreprocessing) (h : raw.unit ≠ .text "V") :
    ensure (raw.unit=.text "V") "physical voltage unit"=
      .error (.sourceReject "physical voltage unit") := by simp [ensure,h]

theorem per_trial_scaling_is_rejected (scale : Value) :
    ensure ((.text "trial" : Value) ≠ .text "trial" ∧
      ((.text "trial" : Value) ≠ .text "trial_reference" ∨ scale=.text "none"))
      "no per trial voltage scaling"=.error (.sourceReject "no per trial voltage scaling") := by
  simp [ensure]

theorem invalid_notch_container_refused (quality : Atom) (rate : Option ℝ) :
    readNotches none quality rate=.error (.fieldFailure "notch list or tuple") := rfl

theorem nonlegacy_edge_needs_zero_iir (method phase : Value)
    (h : ¬(method=.text "iir" ∧ phase=.text "zero")) :
    checkEdgePolicy (.text "mne_reflect_limited") method phase=
      .error (.sourceReject "edge policy requires zero IIR") := by simp [checkEdgePolicy,ensure,h]

inductive DType where
  | floating | integer | boolean | numericOther | other
  deriving DecidableEq

structure ArrayImage (α : Type*) where
  dtype : DType
  shape : List ℕ
  values : List α
  storageGeometry : values.length=shape.prod

def finiteNumber : Atom → Bool
  | .real _ | .integer _ | .boolean _ | .complex _ _ => true
  | _ => false

def zeroNumber : Atom → Bool
  | .real r => decide (r=0)
  | .integer z => decide (z=0)
  | .boolean b => !b
  | .complex real imag => decide (real=0 ∧ imag=0)
  | _ => false

def checkPhysicalCoordinates (historical : Bool) (coordinates : Value) (alignment : Option Value) : Check Unit := do
  ensure (historical=false) "historical dataset cannot enter physical model"
  ensure (coordinates=.text "physical_electrode_voltages") "physical electrode voltage coordinates"
  ensure (alignment=none ∨ alignment=some .null) "no physical EA data alignment"

theorem historical_even_finite_refused (coordinates : Value) (alignment : Option Value) :
    checkPhysicalCoordinates true coordinates alignment=
      .error (.sourceReject "historical dataset cannot enter physical model") := by
  simp [checkPhysicalCoordinates,ensure]

theorem dimensionless_coordinates_refused (alignment : Option Value) :
    checkPhysicalCoordinates false (.text "dimensionless") alignment=
      .error (.sourceReject "physical electrode voltage coordinates") := by
  simp [checkPhysicalCoordinates,ensure]

def checkTensorGeometry (x : ArrayImage Atom) (nChannels nTimes : ℕ) : Check ℕ := do
  let n ← match x.shape with
    | [n,c,t] => do
      ensure (x.dtype=.floating) "floating EEG tensor dtype"
      ensure (0 < n ∧ 0 < c) "positive epoch and channel counts"
      ensure (c=nChannels ∧ t=nTimes) "EEG tensor physical geometry"
      pure n
    | _ => .error (.fieldFailure "EEG tensor rank three")
  pure n

def checkFiniteTensor (x : ArrayImage Atom) : Check Unit :=
  ensure (∀ value ∈ x.values, finiteNumber value=true) "finite EEG tensor values"

theorem finite_tensor_excludes_nan (x : ArrayImage Atom)
    (h : checkFiniteTensor x=.ok ()) : Atom.nan ∉ x.values := by
  intro hn
  have ha : ∀ value ∈ x.values, finiteNumber value=true := by
    simpa [checkFiniteTensor,ensure] using h
  have hx := ha .nan hn
  cases hx

def checkMaskedPlaceholders {N C T : ℕ} (values : Fin N → Fin C → Fin T → Atom)
    (static : Fin C → Bool) (trial : Option (Fin N → Fin C → Bool)) : Check Unit := do
  ensure (∀ i c t, static c=false → zeroNumber (values i c t)=true) "static masked placeholders are zero"
  match trial with
  | none => pure ()
  | some mask =>
    ensure (∀ i c t, mask i c=false → zeroNumber (values i c t)=true) "trial masked placeholders are zero"

def checkObservedMasks {N C : ℕ} (static : Fin C → Bool) (trial : Option (Fin N → Fin C → Bool)) :
    Check Unit := do
  ensure (∃ c, static c=true) "at least one static observed channel"
  match trial with
  | none => pure ()
  | some mask =>
    ensure (∀ i, ∃ c, mask i c=true) "at least one observed channel per trial"
    ensure (∀ i c, mask i c=true → static c=true) "trial cannot enable static absence"

theorem trial_mask_cannot_restore_absent {N C : ℕ} (static : Fin C → Bool)
    (mask : Fin N → Fin C → Bool) (i : Fin N) (c : Fin C)
    (hm : mask i c=true) (hs : static c=false) :
    ensure (∀ j d, mask j d=true → static d=true) "trial cannot enable static absence"=
      .error (.sourceReject "trial cannot enable static absence") := by
  have hn : ¬∀ j d, mask j d=true → static d=true := by
    intro h
    have he := h i c hm
    simp [hs] at he
  simp [ensure,hn]

theorem observed_masks_without_trial_exact {N C : ℕ} (static : Fin C → Bool) :
    checkObservedMasks (N:=N) static none=.ok () ↔ ∃ c, static c=true := by
  by_cases h : ∃ c, static c=true <;> simp [checkObservedMasks,ensure,h]

theorem observed_masks_with_trial_exact {N C : ℕ} (static : Fin C → Bool)
    (trial : Fin N → Fin C → Bool) :
    checkObservedMasks static (some trial)=.ok () ↔
      (∃ c, static c=true) ∧ (∀ i, ∃ c, trial i c=true) ∧
      (∀ i c, trial i c=true → static c=true) := by
  simp only [checkObservedMasks,ensure]
  split_ifs <;> simp_all
  all_goals assumption

theorem placeholders_without_trial_exact {N C T : ℕ} (values : Fin N → Fin C → Fin T → Atom)
    (static : Fin C → Bool) :
    checkMaskedPlaceholders values static none=.ok () ↔
      ∀ i c t, static c=false → zeroNumber (values i c t)=true := by
  by_cases h : ∀ i c t, static c=false → zeroNumber (values i c t)=true
  · simp only [checkMaskedPlaceholders,ensure,if_pos h]
    exact ⟨fun _ => h,fun _ => rfl⟩
  · simp only [checkMaskedPlaceholders,ensure,if_neg h]
    constructor
    · intro he
      cases he
    · intro he
      exact False.elim (h he)

theorem placeholders_with_trial_exact {N C T : ℕ} (values : Fin N → Fin C → Fin T → Atom)
    (static : Fin C → Bool) (trial : Fin N → Fin C → Bool) :
    checkMaskedPlaceholders values static (some trial)=.ok () ↔
      (∀ i c t, static c=false → zeroNumber (values i c t)=true) ∧
      (∀ i c t, trial i c=false → zeroNumber (values i c t)=true) := by
  simp only [checkMaskedPlaceholders,ensure]
  split_ifs <;> simp_all

theorem tensor_geometry_from_independent_fields (x : ArrayImage Atom) (N C T : ℕ)
    (hd : x.dtype=.floating) (hs : x.shape=[N,C,T]) (hn : 0 < N) (hc : 0 < C) :
    checkTensorGeometry x C T=.ok N := by
  simp [checkTensorGeometry,ensure,hd,hs,hn,hc]

theorem finite_tensor_exact (x : ArrayImage Atom) :
    checkFiniteTensor x=.ok () ↔ ∀ value ∈ x.values, finiteNumber value=true := by
  simp [checkFiniteTensor,ensure]

theorem physical_coordinates_success_exact (historical : Bool) (coordinates : Value) (alignment : Option Value) :
    checkPhysicalCoordinates historical coordinates alignment=.ok () ↔
      historical=false ∧ coordinates=.text "physical_electrode_voltages" ∧
      (alignment=none ∨ alignment=some .null) := by
  simp only [checkPhysicalCoordinates,ensure]
  split_ifs <;> simp_all

theorem nonzero_hidden_static_value_refuses {N C T : ℕ}
    (values : Fin N → Fin C → Fin T → Atom) (static : Fin C → Bool)
    (trial : Option (Fin N → Fin C → Bool)) (i : Fin N) (c : Fin C) (t : Fin T)
    (hs : static c=false) (hv : zeroNumber (values i c t)=false) :
    checkMaskedPlaceholders values static trial=
      .error (.sourceReject "static masked placeholders are zero") := by
  have hn : ¬∀ j d u, static d=false → zeroNumber (values j d u)=true := by
    intro h
    have he := h i c t hs
    simp [hv] at he
  simp [checkMaskedPlaceholders,ensure,hn]

def atomNumber : Atom → Option ℝ
  | .boolean b => some (if b then 1 else 0)
  | .integer z => some z
  | .real r => some r
  | _ => none

def numberLE (left right : Atom) : Prop :=
  ∃ a b, atomNumber left=some a ∧ atomNumber right=some b ∧ a ≤ b

def timelineStatuses : List String :=
  ["available","artifact_rejected","boundary_dropped","missing","acquisition_rejected"]

structure TimelineConstructorInput (M : ℕ) where
  rawShapes : Fin 17 → List ℕ
  rawDTypes : Fin 17 → DType
  complete : Value
  onlineCausal : Value
  timingSource : Value
  convertedLengths : Fin 17 → ℕ

structure TimelineConstructorFlags where
  complete : Bool
  onlineCausal : Bool
  timingSource : String

def exactBool (value : Value) (site : String) : Check Bool :=
  match value with
  | .boolean b => .ok b
  | _ => .error (.fieldFailure site)

/-- Actual ScheduledEventTimeline.__post_init__ guards, before the typed
converted field image below. Individual np.array string/int64/float64 copy
operations remain local array conversion boundaries, never a constructor
success primitive. Field positions follow the literal 17-field source list. -/
def checkTimelineConstructor (strip : String → String) {M : ℕ} (input : TimelineConstructorInput M) :
    Check TimelineConstructorFlags := do
  ensure (∀ j, (input.rawShapes j).length=1) "scheduled raw array ranks"
  ensure (∀ j ∈ ([3,4,7,16] : List (Fin 17)), input.rawDTypes j=.integer)
    "scheduled integer arrays forbid bool coercion"
  ensure (∀ j ∈ ([5,6] : List (Fin 17)), input.rawDTypes j=.floating ∨
    input.rawDTypes j=.integer ∨ input.rawDTypes j=.numericOther) "scheduled numeric time arrays"
  let complete ← exactBool input.complete "strict complete Boolean"
  let online ← exactBool input.onlineCausal "strict online Boolean"
  let timing ← match input.timingSource with
    | .text timing => .ok timing
    | _ => .error (.fieldFailure "timing source string")
  ensure (∀ j, input.convertedLengths j=M) "scheduled arrays common converted length"
  ensure (strip timing ≠ "") "nonblank acquisition timing source"
  pure ⟨complete,online,timing⟩

theorem strict_timeline_flags_do_not_accept_numeric_one (site : String) :
    exactBool (.integer 1) site=.error (.fieldFailure site) := rfl

/-- The image is after each scheduled-array conversion in __post_init__.
`shapes` retains every actual field rank; the common finite index domain is
the separately checked common length, not an assumed valid event timeline. -/
structure TimelineImage (M : ℕ) where
  shapes : Fin 17 → List ℕ
  eventIds : Fin M → String
  groups : Fin M → String
  subjects : Fin M → String
  stimulusDType : DType
  onsetSampleDType : DType
  evidenceDType : DType
  stimulus : Fin M → ℤ
  onsetSamples : Fin M → ℤ
  onset : Fin M → Atom
  availableAt : Fin M → Atom
  evidence : Fin M → ℤ
  statuses : Fin M → String
  details : Fin M → String
  datasets : Fin M → String
  sessions : Fin M → String
  runs : Fin M → String
  selections : Fin M → String
  candidates : Fin M → String
  targets : Fin M → String
  repetitions : Fin M → ℤ
  complete : Bool
  onlineCausal : Bool

def availableEventRows {M : ℕ} (timeline : TimelineImage M) : List (Fin M) :=
  (List.finRange M).filter (fun i => decide (timeline.statuses i="available"))

def mappedEvidence {M : ℕ} (timeline : TimelineImage M) : List ℤ :=
  (availableEventRows timeline).map timeline.evidence

def groupRows {M : ℕ} (timeline : TimelineImage M) (group : String) : List (Fin M) :=
  (List.finRange M).filter (fun i => decide (timeline.groups i=group))

def candidateRows {M : ℕ} (timeline : TimelineImage M) (group candidate : String) : List (Fin M) :=
  (groupRows timeline group).filter (fun i => decide (timeline.candidates i=candidate))

def checkTimeline (strip : String → String) {M : ℕ} (timeline : TimelineImage M) (N : ℕ) : Check Unit := do
  ensure (0 < M) "nonempty scheduled event timeline"
  ensure (∀ j, timeline.shapes j=[M]) "one dimensional aligned event fields"
  ensure ((List.ofFn timeline.eventIds).Nodup ∧ ∀ i, timeline.eventIds i ≠ "") "unique nonempty event ids"
  ensure (∀ i, timeline.groups i ≠ "" ∧ timeline.subjects i ≠ "") "event group and participant ids"
  ensure (∀ i, timeline.statuses i ∈ timelineStatuses) "known scheduled event status"
  ensure (timeline.stimulusDType=.integer) "integer event stimulus ids"
  ensure (timeline.onsetSampleDType=.integer ∧ ∀ i, 0 ≤ timeline.onsetSamples i) "nonnegative integer event sample"
  ensure (∀ i, (atomNumber (timeline.onset i)).isSome) "finite event onset"
  ensure (timeline.evidenceDType=.integer) "integer event evidence indices"
  ensure ((∃ i, strip (timeline.candidates i) ≠ "") → ∀ i, strip (timeline.candidates i) ≠ "")
    "uniform candidate presence"
  ensure ((∃ i, strip (timeline.targets i) ≠ "") → ∀ i, strip (timeline.targets i) ≠ "")
    "uniform target presence"
  ensure ((∃ i, strip (timeline.targets i) ≠ "") → ∀ i, strip (timeline.candidates i) ≠ "")
    "targets require candidates"
  ensure ((∃ i, 0 ≤ timeline.repetitions i) → ∀ i, 0 ≤ timeline.repetitions i) "uniform repetition presence"
  ensure (∀ i, -1 ≤ timeline.repetitions i) "valid event repetition values"
  ensure ((∃ i, 0 ≤ timeline.repetitions i) → ∀ i, strip (timeline.candidates i) ≠ "")
    "repetitions require candidates"
  ensure (∀ i, (timeline.statuses i="available" → 0 ≤ timeline.evidence i) ∧
    (timeline.statuses i ≠ "available" → timeline.evidence i = -1)) "available-only evidence indices"
  ensure (∀ i, (atomNumber (timeline.availableAt i)).isSome ↔ timeline.statuses i="available")
    "finite availability exactly on available events"
  ensure (∀ i, timeline.statuses i="available" → numberLE (timeline.onset i) (timeline.availableAt i))
    "event evidence not before onset"
  ensure ((mappedEvidence timeline).Perm ((List.range N).map Int.ofNat)) "event evidence row bijection"
  ensure ((mappedEvidence timeline).length=N) "event availability denominator"
  ensure (∀ i j, timeline.groups i=timeline.groups j → i.val < j.val →
    numberLE (timeline.onset i) (timeline.onset j)) "chronological event groups"
  ensure (∀ i j, timeline.groups i=timeline.groups j →
    timeline.subjects i=timeline.subjects j ∧ timeline.datasets i=timeline.datasets j ∧
    timeline.sessions i=timeline.sessions j ∧ timeline.runs i=timeline.runs j ∧
    timeline.selections i=timeline.selections j) "homogeneous event group identities"
  if ∀ i, strip (timeline.targets i) ≠ "" then
    ensure (∀ i j, timeline.groups i=timeline.groups j → timeline.targets i=timeline.targets j)
      "one target per event group"
    ensure (∀ i, 2 ≤ (((groupRows timeline (timeline.groups i)).map timeline.candidates).toFinset.card))
      "at least two candidates per target group"
    ensure (∀ i, ∃ j, timeline.groups j=timeline.groups i ∧ timeline.candidates j=timeline.targets i)
      "target belongs to group candidates"
  if ∀ i, 0 ≤ timeline.repetitions i then
    ensure (∀ i j, timeline.groups i=timeline.groups j → timeline.candidates i=timeline.candidates j →
      i.val < j.val → timeline.repetitions i < timeline.repetitions j) "increasing candidate repetitions"
    if timeline.complete then
      ensure (∀ i, let repetitions := ((candidateRows timeline (timeline.groups i) (timeline.candidates i)).map
          timeline.repetitions)
        repetitions=(List.range repetitions.length).map (fun k => (k : ℤ)+repetitions.headD 0))
        "complete contiguous candidate repetitions"
  pure ()

def checkEvidenceBijection {M : ℕ} (timeline : TimelineImage M) (N : ℕ) : Check Unit :=
  ensure ((mappedEvidence timeline).Perm ((List.range N).map Int.ofNat)) "event evidence row bijection"

def checkConstructedTimeline (strip : String → String) {M : ℕ}
    (input : TimelineConstructorInput M) (timeline : TimelineImage M) (N : ℕ) : Check Unit := do
  let flags ← checkTimelineConstructor strip input
  ensure (timeline.complete=flags.complete ∧ timeline.onlineCausal=flags.onlineCausal)
    "actual converted timeline flags"
  checkTimeline strip timeline N

theorem evidence_bijection_preserves_denominator {M : ℕ} (timeline : TimelineImage M) (N : ℕ)
    (h : checkEvidenceBijection timeline N=.ok ()) : (mappedEvidence timeline).length=N := by
  have hp : (mappedEvidence timeline).Perm ((List.range N).map Int.ofNat) := by
    simpa [checkEvidenceBijection,ensure] using h
  simpa using hp.length_eq

theorem available_event_membership {M : ℕ} (timeline : TimelineImage M) (event : Fin M) :
    event ∈ availableEventRows timeline ↔ timeline.statuses event="available" := by
  simp [availableEventRows]

theorem mapped_evidence_not_compressed_to_scheduled_count {M : ℕ}
    (timeline : TimelineImage M) (N : ℕ) (h : checkEvidenceBijection timeline N=.ok ()) :
    (availableEventRows timeline).length=N := by
  simpa [mappedEvidence] using evidence_bijection_preserves_denominator timeline N h

structure QCFeatureImage (N C : ℕ) where
  schema : Value
  relativePTP : ArrayImage Atom
  std : ArrayImage Atom
  scale : ArrayImage Atom
  mask : ArrayImage Bool
  observed : Fin N → Fin C → Bool
  maskDecoding : mask.values=(List.ofFn fun i => List.ofFn (observed i)).flatten

def nonnegativeAtom (value : Atom) : Prop := ∃ r, atomNumber value=some r ∧ 0 ≤ r

def effectiveObserved {N C : ℕ} (static : Fin C → Bool) (trial : Option (Fin N → Fin C → Bool)) :
    Fin N → Fin C → Bool := fun i c => static c && (trial.map (fun mask => mask i c)).getD true

def checkQCFeatures {N C : ℕ} (qc : QCFeatureImage N C) (static : Fin C → Bool)
    (trial : Option (Fin N → Fin C → Bool)) : Check Unit := do
  ensure (qc.schema=.text "n2p3net_epoch_qc_features/1") "QC feature schema"
  ensure (qc.relativePTP.shape=[N,C] ∧ qc.std.shape=[N,C] ∧ qc.mask.shape=[N,C] ∧ qc.scale.shape=[N])
    "QC feature geometry"
  ensure (qc.relativePTP.dtype=.floating ∧ qc.std.dtype=.floating ∧ qc.scale.dtype=.floating ∧
    qc.mask.dtype=.boolean) "QC feature array dtypes"
  ensure (∀ i, ∃ c, qc.observed i c=true) "QC features retain a channel per trial"
  ensure ((∀ value ∈ qc.relativePTP.values, nonnegativeAtom value) ∧
    (∀ value ∈ qc.std.values, nonnegativeAtom value) ∧
    (∀ value ∈ qc.scale.values, nonnegativeAtom value)) "finite nonnegative QC feature values"
  ensure (qc.observed=effectiveObserved static trial) "QC features use actual observed mask"

theorem QC_observation_gate_is_actual_mask {N C : ℕ} (qc : QCFeatureImage N C)
    (static : Fin C → Bool) (trial : Option (Fin N → Fin C → Bool)) :
    ensure (qc.observed=effectiveObserved static trial) "QC features use actual observed mask"=.ok () ↔
      ∀ i c, qc.observed i c=(static c && (trial.map (fun mask => mask i c)).getD true) := by
  simp only [ensure]
  split_ifs with h
  · constructor
    · intro _ i c
      exact congrFun (congrFun h i) c
    · intro _
      rfl
  · constructor
    · intro he
      cases he
    · intro he
      exfalso
      apply h
      funext i c
      exact he i c

/-- NumPy norm of one decoded physical coordinate row. Numerical precision,
complex numeric arrays and actual registered head-frame provenance are local
representation boundaries; a name never supplies the missing coordinates. -/
abbrev CoordinateNorm := (Fin 3 → Atom) → Check ℝ

structure DatasetImage (N C T M : ℕ) where
  historical : Bool
  coordinateSystem : Value
  dataAlignment : Option Value
  name : Value
  x : ArrayImage Atom
  values : Fin N → Fin C → Fin T → Atom
  xDecoding : x.values=(List.ofFn fun i => (List.ofFn fun c => List.ofFn (values i c)).flatten).flatten
  channelNames : List String
  positions : ArrayImage Atom
  coordinates : Fin C → Fin 3 → Atom
  positionsDecoding : positions.values=(List.ofFn fun c => List.ofFn (coordinates c)).flatten
  staticImage : ArrayImage Bool
  static : Fin C → Bool
  staticDecoding : staticImage.values=List.ofFn static
  trialImage : Option (ArrayImage Bool)
  trial : Option (Fin N → Fin C → Bool)
  trialDecoding : match trialImage,trial with
    | none,none => True
    | some image,some mask => image.values=(List.ofFn fun i => List.ofFn (mask i)).flatten
    | _,_ => False
  subjectImage : ArrayImage String
  subjects : Fin N → String
  subjectDecoding : subjectImage.values=List.ofFn subjects
  timeline : TimelineImage M
  labels : Option (ArrayImage Atom)
  labelValues : Option (Fin N → ℤ)
  labelDecoding : match labels,labelValues with
    | none,none => True
    | some image,some y => image.values=List.ofFn (fun i => Atom.integer (y i))
    | _,_ => False
  qc : Option (QCFeatureImage N C)
  metadataIsDataFrame : Bool
  metadataRows : ℕ
  metadataColumns : ℕ
  provenanceIsDict : Bool

def checkArrayMasks {N C T M : ℕ} (data : DatasetImage N C T M) : Check Unit := do
  ensure (data.staticImage.dtype=.boolean ∧ data.staticImage.shape=[C]) "static channel mask array"
  ensure (∃ c, data.static c=true) "at least one static observed channel"
  match data.trialImage with
  | none => pure ()
  | some image => ensure (image.dtype=.boolean ∧ image.shape=[N,C]) "trial channel mask array"
  checkObservedMasks data.static data.trial

def checkPositionHeader {N C T M : ℕ} (data : DatasetImage N C T M) : Check Unit := do
  ensure (data.positions.shape=[C,3]) "physical channel positions geometry"
  ensure (data.positions.dtype=.floating ∨ data.positions.dtype=.integer ∨ data.positions.dtype=.numericOther)
    "numeric physical channel positions"

def checkObservedPositions (norm : CoordinateNorm) {N C T M : ℕ} (data : DatasetImage N C T M) : Check Unit := do
  let checked := (List.finRange C).filter data.static
  ensure (∀ c ∈ checked, ∀ coordinate, finiteNumber (data.coordinates c coordinate)=true)
    "finite observed coordinates"
  let lengths ← checked.mapM (fun c => norm (data.coordinates c))
  ensure (∀ length ∈ lengths, 0 < length) "nonzero observed coordinates"
  let lengths ← checked.mapM (fun c => norm (data.coordinates c))
  ensure (∀ length ∈ lengths, length ≤ 1/2) "registered metre head frame"

def checkChannelPositions (norm : CoordinateNorm) {N C T M : ℕ} (data : DatasetImage N C T M) : Check Unit := do
  checkPositionHeader data
  checkObservedPositions norm data

def checkPositionAndMasks (norm : CoordinateNorm) {N C T M : ℕ} (data : DatasetImage N C T M) : Check Unit := do
  checkPositionHeader data
  checkArrayMasks data
  checkObservedPositions norm data

theorem rank_and_dtype_fault_reports_rank_first :
    checkTensorGeometry ⟨.integer,[0],[],by rfl⟩ 0 0=
      .error (.fieldFailure "EEG tensor rank three") := rfl

theorem position_header_fault_precedes_mask_fault (norm : CoordinateNorm) {N C T M : ℕ}
    (data : DatasetImage N C T M)
    (h : data.positions.shape ≠ [C,3] ∧ data.staticImage.dtype ≠ .boolean) :
    checkPositionAndMasks norm data=.error (.sourceReject "physical channel positions geometry") := by
  simp [checkPositionAndMasks,checkPositionHeader,ensure,h.1]

theorem static_absence_precedes_trial_array_checks {N C T M : ℕ} (data : DatasetImage N C T M)
    (h : data.staticImage.dtype=.boolean ∧ data.staticImage.shape=[C] ∧ ¬∃ c, data.static c=true) :
    checkArrayMasks data=.error (.sourceReject "at least one static observed channel") := by
  simp [checkArrayMasks,ensure,h.1,h.2.1,h.2.2]

theorem all_observed_finite_checks_precede_any_norm (norm : CoordinateNorm) {N C T M : ℕ}
    (data : DatasetImage N C T M)
    (h : ¬∀ c ∈ (List.finRange C).filter data.static, ∀ coordinate,
      finiteNumber (data.coordinates c coordinate)=true) :
    checkObservedPositions norm data=.error (.sourceReject "finite observed coordinates") := by
  simp only [checkObservedPositions,ensure,if_neg h]
  rfl

def checkDatasetLabels (strip : String → String) {N C T M : ℕ}
    (data : DatasetImage N C T M) (requireLabels : Bool) : Check Unit := do
  match data.labels,data.labelValues with
  | none,none => ensure (requireLabels=false) "required labels present"
  | some image,some labels =>
    ensure (image.shape=[N] ∧ image.dtype=.integer) "aligned nonbool integer labels"
    if ∀ event, strip (data.timeline.candidates event) ≠ "" ∧ strip (data.timeline.targets event) ≠ "" then
      ensure (∀ event, data.timeline.statuses event="available" →
        ∀ row : Fin N, data.timeline.evidence event=(row.val : ℤ) →
          labels row=(if data.timeline.candidates event=data.timeline.targets event then 1 else 0))
        "binary labels agree with scheduled candidate targets"
  | _,_ => .error (.fieldFailure "label array decoding")

theorem optional_absent_labels_are_not_required (strip : String → String) {N C T M : ℕ}
    (data : DatasetImage N C T M) (hl : data.labels=none) (hy : data.labelValues=none) :
    checkDatasetLabels strip data false=.ok () := by simp [checkDatasetLabels,hl,hy,ensure]

theorem required_absent_labels_refuse (strip : String → String) {N C T M : ℕ}
    (data : DatasetImage N C T M) (hl : data.labels=none) (hy : data.labelValues=none) :
    checkDatasetLabels strip data true=.error (.sourceReject "required labels present") := by
  simp [checkDatasetLabels,hl,hy,ensure]

/-- Body of EpochDataset.validate after physical/identity exclusion and before
execution, attached-identity coverage, lineage and cache-attestation checks.
Those continuations are separate explicit layers, not assumed here. -/
def canonicalNames (canonicalChannel : String → Check String) : List String → Check (List String)
  | [] => .ok []
  | name :: tail => do
    let canonical ← canonicalChannel name
    let rest ← canonicalNames canonicalChannel tail
    pure (canonical :: rest)

def checkDatasetBodyAfterDeclaration (strip : String → String) (canonicalChannel : String → Check String)
    (norm : CoordinateNorm) (raw : RawPreprocessing) (declaration : DeclarationScalars)
    {N C T M : ℕ} (data : DatasetImage N C T M) (requireLabels : Bool) : Check Unit := do
  ensure (data.timeline.onlineCausal=decide (raw.phase=.text "forward")) "event causal flag matches preprocessing"
  let name ← match data.name with
    | .text name => .ok name
    | _ => .error (.fieldFailure "dataset name")
  ensure (strip name ≠ "") "nonempty dataset name"
  let actualN ← checkTensorGeometry data.x C T
  ensure (actualN=N ∧ declaration.nTimes=(T : ℤ)) "dataset tensor matches physical time axis"
  let canonical ← canonicalNames canonicalChannel data.channelNames
  ensure (canonical.length=C ∧ canonical.Nodup)
    "distinct aligned physical electrodes"
  checkPositionAndMasks norm data
  ensure (data.subjectImage.shape=[N] ∧ ∀ i, strip (data.subjects i) ≠ "") "aligned nonempty subject ids"
  checkTimeline strip data.timeline N
  ensure (∀ event, data.timeline.statuses event="available" → ∀ row : Fin N,
    data.timeline.evidence event=(row.val : ℤ) → data.timeline.subjects event=data.subjects row)
    "event evidence agrees with subject ids"
  checkDatasetLabels strip data requireLabels
  checkFiniteTensor data.x
  checkMaskedPlaceholders data.values data.static data.trial
  match data.qc with
  | none => pure ()
  | some qc => checkQCFeatures qc data.static data.trial
  ensure (data.metadataIsDataFrame=true) "metadata DataFrame type"
  ensure (data.metadataRows=0 ∨ data.metadataColumns=0 ∨ data.metadataRows=N) "aligned metadata rows"
  ensure (data.provenanceIsDict=true) "provenance dictionary type"
  pure ()

def checkDatasetBody (strip : String → String) (canonicalChannel : String → Check String)
    (floorSamples : FloorSampleCount) (windowSamples : WindowSampleCount) (norm : CoordinateNorm)
    (raw : RawPreprocessing) {N C T M : ℕ} (data : DatasetImage N C T M) (requireLabels : Bool) :
    Check DeclarationScalars := do
  let declaration ← checkPreprocessing strip floorSamples windowSamples raw
  checkDatasetBodyAfterDeclaration strip canonicalChannel norm raw declaration data requireLabels
  pure declaration

def checkCacheAttestationFields (fields : Fields) : Check Unit := do
  ensure (fieldDefault fields "schema"=.text "n2p3net_epoch_cache_attestation/1") "cache attestation schema"
  ensure (fieldDefault fields "full_contract_validated"=.boolean true) "cache full validation strict True"
  let size ← positiveInt (fieldDefault fields "byte_size") "positive integer cache byte size"
  ensure (0 < size) "positive cache bytes"
  let digest ← match fieldDefault fields "sha256" with
    | .text digest => .ok digest
    | _ => .error (.fieldFailure "cache digest string")
  ensure (LowerHex64 digest) "lowercase cache SHA256"

def checkVerifiedCacheFields (fields : Fields) : Check Unit := do
  ensure (fieldDefault fields "schema"=.text "n2p3_verified_epoch_cache_load/1") "verified cache schema"
  checkCacheAttestationFields (Fields.ofList
    [("schema",fieldDefault fields "cache_attestation_schema"),
     ("full_contract_validated",fieldDefault fields "full_contract_validated"),
     ("sha256",fieldDefault fields "sha256"),("byte_size",fieldDefault fields "byte_size")])
  let digest ← match fieldDefault fields "decoded_dataset_sha256" with
    | .text digest => .ok digest
    | _ => .error (.fieldFailure "decoded dataset digest string")
  ensure (LowerHex64 digest) "lowercase decoded dataset SHA256"

inductive LineageImage where
  | absent | dataLineage | other
  deriving DecidableEq

def checkDatasetTail (subjects : List String) (identitySubjects : Option (List String))
    (lineageTag : LineageImage) (attestation : Option Value) : Check Unit := do
  match identitySubjects with
  | none => pure ()
  | some keys => ensure (keys.toFinset=subjects.toFinset) "exact attached identity subject coverage"
  ensure (lineageTag=.absent ∨ lineageTag=.dataLineage) "optional DataLineage runtime type"
  match attestation with
  | none => pure ()
  | some value =>
    let fields ← objectAt value "verified cache mapping"
    checkVerifiedCacheFields fields

theorem extra_attached_identity_subject_refused (subjects keys : List String)
    (lineageTag : LineageImage) (attestation : Option Value) (h : keys.toFinset ≠ subjects.toFinset) :
    checkDatasetTail subjects (some keys) lineageTag attestation=
      .error (.sourceReject "exact attached identity subject coverage") := by
  simp [checkDatasetTail,ensure,h]

end

end N2P3.PersonalERPData
