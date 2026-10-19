import N2P3.TemporalWhitening
import N2P3.StateLedger
import N2P3.PersonalAdmission

/-!
# Full TemporalNoise constructor and canonical state replay

Source: temporal_noise.py:50-115,157-235,375-442. The program below contains
the source's guards and every retained scalar/array/metadata field. Numerical
boundaries are inverse PARCOR, root radius, Lyapunov solve/residual, condition
number and Cholesky, with explicit success/failure results. There is no opaque
whole constructor and no desired replay/whitening equation in the model.
`lyapunovCovariance` returns the source solve followed by (C+C.T)/2;
`inverseReflections` is the source checked step-down procedure. Success of an
arbitrary numerical primitive is not a proof of its mathematical correctness.

The domain is finite decoded real scalars, unique-key recursive JSON mappings
and float-normalized rank-one coefficients. String-to-float parsing is an
independent scalar boundary. IEEE finite checks/overflow, NumPy coercion and
SciPy numerical correctness require refinement; reals here contain no NaN or
infinity. Hashes are arbitrary functions, never injectivity/authentication
axioms. Accepted raw records can normalize coefficient tags: only canonical
numeric writer records are claimed to recover the complete constructor state.
Raw parsing below is restricted to complete required keys and coefficient /
optional reflection vectors that successfully decode to finite floats. Early
Python subscription/coercion failure order outside that domain is not refined.

Successful construction is the legal-model domain, derived from this literal
program. It is not a premise that loading succeeds or scores agree. Full
provenance is retained, including actual analytic falsy-record values. A
declared training ledger does not establish physical fitting authority.
-/

namespace N2P3.TemporalState

open N2P3.FrozenState N2P3.StateLedger
open N2P3.PersonalAdmission (PyEq)

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev Mode := N2P3.TemporalWhitening.Mode

def modeText : Mode → String
  | .stationary => "stationary"
  | .conditionalPrefix => "conditional_prefix"
  | .stationaryInnovations => "stationary_innovations"

def modeSchema : Mode → String
  | .stationary => "n2p3_temporal_noise/1"
  | .conditionalPrefix => "n2p3_temporal_noise/2"
  | .stationaryInnovations => "n2p3_temporal_noise/3"

def modeAssumption : Mode → String
  | .conditionalPrefix => "conditional_on_observed_first_p_samples"
  | _ => "stationary_AR_latent_initial_state"

def scaleConvention := "unit_innovation_temporal_shape_preserve_data_units"

def objectFields (v : Value) (site : String) : Check Fields :=
  match v with
  | .object f => .ok f
  | _ => .error (.fieldFailure site)

def temporalRecordKeys := ["record_id", "n_samples", "input_sha256"]
def temporalTrainingKeys := ["fit_scope", "records", "method", "assume_centered",
  "available_lag_rows", "minimum_lag_rows", "minimum_samples_per_parameter", "n_columns",
  "implementation_sha256"]

def readExactInt (v : Value) (site : String) : Check ℤ :=
  match v with
  | .integer n => .ok n
  | _ => .error (.sourceReject site)

def checkTemporalRecord (order : ℕ) (record : Value) : Check ℤ := do
  let f ← objectFields record "temporal record mapping"
  ensure (ExactKeys f temporalRecordKeys) "temporal fit record"
  let ns ← readExactInt (← required f "n_samples") "temporal fit record"
  ensure (max (3 : ℤ) ((order : ℤ)+2) ≤ ns) "temporal fit record"
  requireDigest (← required f "input_sha256") "temporal fit input"
  pure (ns-order)

def checkTemporalRecords (order : ℕ) : List Value → Check ℤ
  | [] => .ok 0
  | record::tail => do
      let rows ← checkTemporalRecord order record
      let rest ← checkTemporalRecords order tail
      pure (rows+rest)

/-- Exact source guard order, including ID collection before per-record keys. -/
def checkProvenance (implementation : String) (order : ℕ) (provenance : Value) : Check Unit := do
  let f ← objectFields provenance "temporal provenance.get"
  if fieldDefault f "fit_scope" = .text "analytic_reference" then
    ensure (ExactKeys f analyticKeys) "temporal analytic reference"
    let reference ← required f "reference_id"
    match reference with
    | .text s => ensure (Nonblank s) "temporal analytic reference"
    | _ => .error (.sourceReject "temporal analytic reference")
    ensure (¬PyTruthy (fieldDefault f "records")) "temporal analytic reference"
  else
    ensure (ExactKeys f temporalTrainingKeys) "temporal scope and records"
    allowedFitScope (fieldDefault f "fit_scope") "temporal scope and records"
    ensure (fieldDefault f "method" = .text "segmented_multichannel_burg_no_pole_clipping" ∧
      fieldDefault f "assume_centered" = .boolean true ∧ PyTruthy (fieldDefault f "records"))
      "temporal scope and records"
    let records ← trainingRecords (fieldDefault f "records")
    let ids ← collectRecordIds records
    let _ ← requireIdentifiers (.sequence (valuesOfList ids)) "temporal fit record IDs"
    let columns ← readExactInt (← required f "n_columns") "temporal positive columns"
    ensure (1 ≤ columns) "temporal positive columns"
    let actual ← checkTemporalRecords order records
    let perParameter ← readExactInt (← required f "minimum_samples_per_parameter") "temporal per parameter"
    ensure (2 ≤ perParameter) "temporal per parameter"
    let minimum := max (32 : ℤ) (perParameter * max 1 (order : ℤ))
    let available ← readExactInt (← required f "available_lag_rows") "temporal lag rows"
    let declaredMinimum ← readExactInt (← required f "minimum_lag_rows") "temporal lag rows"
    ensure (available = actual ∧ declaredMinimum = minimum ∧ minimum ≤ actual) "temporal lag rows"
    let implementationField ← required f "implementation_sha256"
    requireDigest implementationField "temporal implementation"
    ensure (implementationField = .text implementation) "temporal implementation identity"

theorem temporal_analytic_requires_nonblank_text (implementation : String) (order : ℕ)
    (reference records : Value) :
    checkProvenance implementation order (analyticLedger reference records) = .ok () ↔
      (∃ s, reference = .text s ∧ Nonblank s) ∧ ¬PyTruthy records := by
  cases reference <;>
    simp [checkProvenance, analyticLedger, objectFields, fieldDefault, required,
      Fields.ofList, Fields.get, ExactKeys, fieldKeys, analyticKeys, ensure]
  split_ifs <;> simp_all

theorem false_record_value_is_analytic_falsy (implementation : String) (order : ℕ)
    (s : String) (h : Nonblank s) :
    checkProvenance implementation order (analyticLedger (.text s) (.boolean false)) = .ok () := by
  rw [temporal_analytic_requires_nonblank_text]
  exact ⟨⟨s,rfl,h⟩, by simp [PyTruthy]⟩

theorem boolean_reference_rejected (implementation : String) (order : ℕ) :
    checkProvenance implementation order (analyticLedger (.boolean true) (.sequence .nil)) =
      .error (.sourceReject "temporal analytic reference") := by
  simp [checkProvenance, analyticLedger, objectFields, fieldDefault, required,
    Fields.ofList, Fields.get, ExactKeys, fieldKeys, analyticKeys, ensure]

def emptyArray : ArrayState := ⟨[0,0], []⟩

structure Primitives where
  parseFloatText : String → Option ℝ
  inverseReflections : List ℝ → Check (List ℝ)
  poleRadius : List ℝ → Check ℝ
  lyapunovCovariance : List ℝ → Check ArrayState
  startupResidualOK : List ℝ → ArrayState → Prop
  conditionNumber : ArrayState → Check ℝ
  cholesky : ArrayState → Check ArrayState
  arrayHash : ArrayState → String
  metadataHash : Value → String
  implementation : String

def floatScalar (p : Primitives) : Value → Option ℝ
  | .number r => some r
  | .integer z => some z
  | .boolean b => some (if b then 1 else 0)
  | .text s => p.parseFloatText s
  | _ => none

def floatValues (p : Primitives) : Values → Option (List ℝ)
  | .nil => some []
  | .cons v tail => do
      let x ← floatScalar p v
      let xs ← floatValues p tail
      pure (x::xs)

def floatVector (p : Primitives) : Value → Check (List ℝ)
  | .sequence xs => match floatValues p xs with
    | some values => .ok values
    | none => .error (.fieldFailure "finite float coefficient vector")
  | _ => .error (.fieldFailure "rank one coefficient vector")

theorem floatValues_encode (p : Primitives) (xs : List ℝ) :
    floatValues p (encodeReals xs) = some xs := by
  induction xs with
  | nil => rfl
  | cons x tail ih => simp [encodeReals, floatValues, floatScalar, ih]

theorem floatVector_encode (p : Primitives) (xs : List ℝ) :
    floatVector p (.sequence (encodeReals xs)) = .ok xs := by
  simp [floatVector, floatValues_encode]

def readVariance (v : Value) : Check ℝ := do
  let r ← match v with
    | .number r => .ok r
    | .integer z => .ok (z : ℝ)
    | .boolean _ => .error (.sourceReject "positive innovation variance")
    | _ => .error (.fieldFailure "innovation variance scalar")
  ensure (0 < r) "positive innovation variance"
  pure r

theorem readVariance_number (r : ℝ) (h : 0 < r) : readVariance (.number r) = .ok r := by
  simp [readVariance, ensure, h]

theorem readVariance_result_positive (v : Value) (r : ℝ) (h : readVariance v = .ok r) : 0 < r := by
  cases v <;> simp [readVariance,ensure] at h
  all_goals (split_ifs at h; simp_all)
  subst r
  exact_mod_cast ‹(0 : ℤ) < _›

def parcorStep (polynomial : List ℝ) (k : ℝ) : List ℝ :=
  List.zipWith (fun a b => a+k*b) polynomial polynomial.reverse ++ [k]

def parcorCoefficients (reflections : List ℝ) : List ℝ :=
  (reflections.foldl parcorStep []).map (- ·)

def allClose (coefficients expected : List ℝ) : Prop :=
  coefficients.length = expected.length ∧ ∀ i < coefficients.length,
    |coefficients.getD i 0 - expected.getD i 0| ≤
      (2/100000000000000 : ℝ) + (2/100000000000000 : ℝ) * |expected.getD i 0|

def parcorMargin : ℝ := 1-1/1000000000000

structure Startup where
  covariance : ArrayState
  radius : ℝ
  condition : Option ℝ

def startup (p : Primitives) (coefficients : List ℝ) (conditional : Bool) : Check Startup := do
  if coefficients = [] then
    pure ⟨emptyArray,0,if conditional then none else some 1⟩
  else
    let radius ← p.poleRadius coefficients
    ensure (radius < parcorMargin) "strict AR pole stability"
    if conditional then pure ⟨emptyArray,radius,none⟩
    else
      let covariance ← p.lyapunovCovariance coefficients
      ensure (p.startupResidualOK coefficients covariance) "startup covariance solve"
      let condition ← p.conditionNumber covariance
      ensure (condition ≤ 1000000000000) "startup covariance condition"
      let _ ← p.cholesky covariance
      pure ⟨covariance,radius,some condition⟩

def factorArray (reflections : List ℝ) : ArrayState :=
  ⟨[reflections.length,reflections.length], (List.range reflections.length).flatMap (fun t =>
    (List.range reflections.length).map (fun j =>
      let scale := N2P3.TemporalWhitening.prefixScale reflections t
      if j=t then scale
      else if j<t then
        (reflections.take t |>.foldl parcorStep []).getD (t-1-j) 0 * scale
      else 0))⟩

structure Factor where
  array : ArrayState
  logs : List ℝ
  condition : ℝ

def innovationFactor (p : Primitives) (reflections : List ℝ) : Check Factor := do
  let array := factorArray reflections
  let logs := N2P3.TemporalWhitening.suffixLogs reflections
  ensure (∀ t < reflections.length, 0 < N2P3.TemporalWhitening.prefixScale reflections t)
    "invertible stationary innovations factor"
  let condition ← if reflections = [] then .ok 1 else p.conditionNumber array
  pure ⟨array,logs,condition⟩

structure ConstructorInput where
  coefficients : List ℝ
  variance : Value
  provenance : Value
  mode : Mode
  reflectionState : Option (List ℝ)

structure NumericFields where
  reflections : List ℝ
  covariance : ArrayState
  radius : ℝ
  startupCondition : Option ℝ
  factor : ArrayState
  logVariances : List ℝ
  factorCondition : Option ℝ

def selectReflections (p : Primitives) (input : ConstructorInput) : Check (List ℝ) := do
  if input.mode = .stationaryInnovations then
    match input.reflectionState with
      | some ks => .ok ks
      | none => do
          let provenance ← objectFields input.provenance "temporal provenance.get"
          ensure (fieldDefault provenance "fit_scope" = .text "analytic_reference") "actual fitted PARCOR required"
          p.inverseReflections input.coefficients
  else
    ensure (input.reflectionState = none) "PARCOR only belongs to innovations"
    p.inverseReflections input.coefficients

def finishNumeric (p : Primitives) (input : ConstructorInput) (reflection : List ℝ) : Check NumericFields := do
  if input.mode = .stationaryInnovations then
    ensure (reflection.length = input.coefficients.length ∧ ∀ k ∈ reflection, |k| < parcorMargin)
      "actual PARCOR strict stability"
    ensure (allClose input.coefficients (parcorCoefficients reflection)) "AR and actual PARCOR agreement"
    let factor ← innovationFactor p reflection
    let radius ← if input.coefficients = [] then .ok 0 else p.poleRadius input.coefficients
    pure ⟨reflection,emptyArray,radius,none,factor.array,factor.logs,some factor.condition⟩
  else
    let initial ← startup p input.coefficients (input.mode = .conditionalPrefix)
    pure ⟨reflection,initial.covariance,initial.radius,initial.condition,emptyArray,[],none⟩

def numericFields (p : Primitives) (input : ConstructorInput) : Check NumericFields := do
  let reflection ← selectReflections p input
  finishNumeric p input reflection

structure CoreModel where
  coefficients : List ℝ
  variance : ℝ
  provenance : Value
  mode : Mode
  numeric : NumericFields
  startupRoot : ArrayState
  startupInnovationVariances : List ℝ

def marginalVariance (m : CoreModel) : ℝ :=
  if m.coefficients = [] then 1
  else if m.mode = .stationary then m.numeric.covariance.values.headD 0
  else Real.exp (N2P3.TemporalWhitening.logVariance m.numeric.reflections)

def constructCore (p : Primitives) (input : ConstructorInput) : Check CoreModel := do
  ensure (input.coefficients.length ≤ 128) "AR order at most 128"
  let variance ← readVariance input.variance
  let numeric ← numericFields p input
  checkProvenance p.implementation input.coefficients.length input.provenance
  let root ← p.cholesky numeric.covariance
  let model : CoreModel := ⟨input.coefficients,variance,copyValue input.provenance,input.mode,
    numeric,root,numeric.logVariances.map Real.exp⟩
  ensure (0 < marginalVariance model) "positive marginal variance scalar"
  pure model

def canonicalInput (m : CoreModel) : ConstructorInput :=
  ⟨m.coefficients,.number m.variance,m.provenance,m.mode,
    if m.mode = .stationaryInnovations then some m.numeric.reflections else none⟩

theorem bind_success {A B : Type} {first : Check A} {next : A → Check B} {result : B}
    (h : (first >>= next) = .ok result) :
    ∃ x, first = .ok x ∧ next x = .ok result := by
  cases first with
  | error e => simp at h
  | ok x => exact ⟨x,rfl,h⟩

theorem finishNumeric_retains_reflections (p : Primitives) (input : ConstructorInput)
    (ks : List ℝ) (numeric : NumericFields) (h : finishNumeric p input ks = .ok numeric) :
    numeric.reflections = ks := by
  by_cases hm : input.mode = .stationaryInnovations
  · simp only [finishNumeric,if_pos hm] at h
    obtain ⟨u,_,h⟩ := bind_success h
    obtain ⟨v,_,h⟩ := bind_success h
    obtain ⟨factor,_,h⟩ := bind_success h
    by_cases hc : input.coefficients=[]
    · simp only [if_pos hc] at h
      have he := Except.ok.inj h
      exact congrArg NumericFields.reflections he.symm
    · simp only [if_neg hc] at h
      obtain ⟨radius,_,h⟩ := bind_success h
      have he := Except.ok.inj h
      exact congrArg NumericFields.reflections he.symm
  · simp only [finishNumeric,if_neg hm] at h
    obtain ⟨initial,_,h⟩ := bind_success h
    have he := Except.ok.inj h
    exact congrArg NumericFields.reflections he.symm

theorem numericFields_replay (p : Primitives) (input : ConstructorInput) (numeric : NumericFields)
    (h : numericFields p input = .ok numeric) :
    numericFields p {input with reflectionState :=
      if input.mode = .stationaryInnovations then some numeric.reflections else none} = .ok numeric := by
  cases hs : selectReflections p input with
  | error e => simp [numericFields, hs] at h
  | ok ks =>
    have hf : finishNumeric p input ks = .ok numeric := by simpa [numericFields,hs] using h
    have hk := finishNumeric_retains_reflections p input ks numeric hf
    by_cases hm : input.mode = .stationaryInnovations
    · have he : selectReflections p {input with reflectionState := some numeric.reflections} =
          .ok ks := by simp [selectReflections,hm,hk]
      have he' := he
      simp only [hm] at he'
      simp only [hm, ite_true,numericFields]
      rw [he']
      simpa only [finishNumeric,hm] using hf
    · have hn : input.reflectionState = none := by
        simp [selectReflections,hm,ensure] at hs
        (split_ifs at hs; simp_all)
      have he : {input with reflectionState := none} = input := by cases input; simp_all
      simpa only [if_neg hm,he] using h

theorem constructCore_components (p : Primitives) (input : ConstructorInput) (m : CoreModel)
    (h : constructCore p input = .ok m) :
    input.coefficients.length ≤ 128 ∧ ∃ variance numeric root,
      readVariance input.variance = .ok variance ∧ numericFields p input = .ok numeric ∧
      checkProvenance p.implementation input.coefficients.length input.provenance = .ok () ∧
      p.cholesky numeric.covariance = .ok root ∧
      m = ⟨input.coefficients,variance,input.provenance,input.mode,numeric,root,
        numeric.logVariances.map Real.exp⟩ ∧ 0 < marginalVariance m := by
  simp only [constructCore] at h
  obtain ⟨u,ho,h⟩ := bind_success h
  cases u
  have ho' := (ensure_ok_iff _ _).mp ho
  obtain ⟨variance,hv,h⟩ := bind_success h
  obtain ⟨numeric,hn,h⟩ := bind_success h
  obtain ⟨v,hp,h⟩ := bind_success h
  cases v
  obtain ⟨root,hr,h⟩ := bind_success h
  obtain ⟨u,hm,h⟩ := bind_success h
  cases u
  have hm' := (ensure_ok_iff _ _).mp hm
  have he := Except.ok.inj h
  simp only [copyValue_preserves] at he hm'
  exact ⟨ho',variance,numeric,root,hv,hn,hp,hr,he.symm,he ▸ hm'⟩

theorem constructCore_canonical_replay (p : Primitives) (input : ConstructorInput) (m : CoreModel)
    (h : constructCore p input = .ok m) : constructCore p (canonicalInput m) = .ok m := by
  obtain ⟨ho,variance,numeric,root,hv,hn,hp,hr,hm,hpositive⟩ := constructCore_components p input m h
  have hvpos := readVariance_result_positive input.variance variance hv
  have hnr := numericFields_replay p input numeric hn
  subst m
  have he : numericFields p
      ⟨input.coefficients,.number variance,input.provenance,input.mode,
        if input.mode=.stationaryInnovations then some numeric.reflections else none⟩ = .ok numeric := by
    simpa [numericFields,selectReflections,finishNumeric] using hnr
  simp [constructCore,canonicalInput,ensure,ho,readVariance_number _ hvpos,he,hp,hr,
    copyValue_preserves,hpositive]

def optionalReal : Option ℝ → Value
  | none => .null
  | some r => .number r

def arrayVector (xs : List ℝ) : ArrayState := ⟨[xs.length],xs⟩

def commonStateList (p : Primitives) (m : CoreModel) : List (String × Value) :=
  [("schema",.text (modeSchema m.mode)),
   ("coefficients",.sequence (encodeReals m.coefficients)),
   ("innovation_variance",.number m.variance),("provenance",m.provenance),
   ("scale_convention",.text scaleConvention),("startup_assumption",.text (modeAssumption m.mode)),
   ("startup_covariance_sha256",.text (p.arrayHash m.numeric.covariance)),
   ("startup_root_sha256",.text (p.arrayHash m.startupRoot)),
   ("reflection_coefficients",.sequence (encodeReals m.numeric.reflections)),
   ("pole_radius",.number m.numeric.radius),("startup_condition",optionalReal m.numeric.startupCondition)]

def extraStateList (p : Primitives) (m : CoreModel) : List (String × Value) :=
  match m.mode with
  | .stationary => []
  | .conditionalPrefix =>
    [("startup_mode",.text (modeText m.mode)),("conditioning_prefix_samples",.integer m.coefficients.length),
     ("marginal_variance_per_unit_innovation",.number (marginalVariance m))]
  | .stationaryInnovations =>
    [("startup_mode",.text (modeText m.mode)),("marginal_variance_per_unit_innovation",.number (marginalVariance m)),
     ("startup_whitening_factor_sha256",.text (p.arrayHash m.numeric.factor)),
     ("startup_innovation_variances",.sequence (encodeReals m.startupInnovationVariances)),
     ("startup_log_variances_sha256",.text (p.arrayHash (arrayVector m.numeric.logVariances))),
     ("startup_factor_condition",optionalReal m.numeric.factorCondition),
     ("floating_precision_condition_indicator",optionalReal (m.numeric.factorCondition.map
        (fun r => (1/4503599627370496 : ℝ)*r))),
     ("stability_certificate",.text "actual_PARCOR_strictly_inside_unit_interval"),
     ("reflection_origin",.text (if (match m.provenance with
       | .object f => fieldDefault f "fit_scope" | _ => .null) = .text "analytic_reference"
       then "analytic_reference" else "actual_Burg_recursion")),
     ("expanded_polynomial_radius_role",.text "diagnostic_only_not_lattice_stability_test")]

def stateFields (p : Primitives) (m : CoreModel) : Fields :=
  Fields.ofList (commonStateList p m ++ extraStateList p m)

structure Model where
  core : CoreModel
  integrity : String

def sealModel (p : Primitives) (m : CoreModel) : Model :=
  ⟨m,p.metadataHash (.object (stateFields p m))⟩

def construct (p : Primitives) (input : ConstructorInput) : Check Model := do
  let core ← constructCore p input
  pure (sealModel p core)

def record (p : Primitives) (model : Model) : Check Value := do
  ensure (model.integrity = p.metadataHash (.object (stateFields p model.core))) "temporal integrity"
  pure (.object (.cons "artifact_sha256"
    (.text (p.metadataHash (.object (stateFields p model.core)))) (stateFields p model.core)))

def canonicalRecord (p : Primitives) (m : CoreModel) : Value :=
  .object (.cons "artifact_sha256" (.text (p.metadataHash (.object (stateFields p m)))) (stateFields p m))

theorem sealed_record (p : Primitives) (m : CoreModel) :
    record p (sealModel p m) = .ok (canonicalRecord p m) := by simp [record,sealModel,ensure,canonicalRecord]

def readMode (fields : Fields) : Check Mode := do
  let schema := fieldDefault fields "schema"
  let rawMode := if schema = .text "n2p3_temporal_noise/1" then .text "stationary"
    else fieldDefault fields "startup_mode"
  let mode ← if rawMode = .text "stationary" then .ok N2P3.TemporalWhitening.Mode.stationary
    else if rawMode = .text "conditional_prefix" then .ok N2P3.TemporalWhitening.Mode.conditionalPrefix
    else if rawMode = .text "stationary_innovations" then .ok N2P3.TemporalWhitening.Mode.stationaryInnovations
    else .error (.sourceReject "temporal contract")
  ensure ((schema = .text "n2p3_temporal_noise/1" ∨ schema = .text "n2p3_temporal_noise/2" ∨
    schema = .text "n2p3_temporal_noise/3") ∧
    fieldDefault fields "scale_convention" = .text scaleConvention ∧
    fieldDefault fields "startup_assumption" = .text (modeAssumption mode) ∧
    (schema = .text "n2p3_temporal_noise/2" → mode=.conditionalPrefix) ∧
    (schema = .text "n2p3_temporal_noise/3" → mode=.stationaryInnovations)) "temporal contract"
  pure mode

def readInput (p : Primitives) (fields : Fields) (mode : Mode) : Check ConstructorInput := do
  let coefficients ← floatVector p (← required fields "coefficients")
  let variance ← required fields "innovation_variance"
  let provenance ← required fields "provenance"
  let reflection ← if mode = .stationaryInnovations then do
    let ks ← floatVector p (← required fields "reflection_coefficients")
    pure (some ks)
    else pure none
  pure ⟨coefficients,variance,provenance,mode,reflection⟩

def commonDiagnosticKeys := ["reflection_coefficients","startup_covariance_sha256","startup_root_sha256",
  "pole_radius","startup_condition"]
def conditionalDiagnosticKeys := ["conditioning_prefix_samples","marginal_variance_per_unit_innovation"]
def innovationsDiagnosticKeys := ["marginal_variance_per_unit_innovation","startup_whitening_factor_sha256",
  "startup_innovation_variances","startup_log_variances_sha256","startup_factor_condition",
  "floating_precision_condition_indicator","stability_certificate","reflection_origin","expanded_polynomial_radius_role"]

def diagnosticsEqual (keys : List String) (a b : Fields) : Prop :=
  ∀ key ∈ keys, PyEq (fieldDefault a key) (fieldDefault b key)

def checkReplay (p : Primitives) (fields : Fields) (m : CoreModel) : Check Unit := do
  let actual := stateFields p m
  ensure (ExactKeys fields (fieldKeys actual) ∧ diagnosticsEqual commonDiagnosticKeys actual fields)
    "temporal numeric replay"
  if m.mode = .conditionalPrefix then
    ensure (diagnosticsEqual conditionalDiagnosticKeys actual fields) "conditional numeric replay"
  else if m.mode = .stationaryInnovations then
    ensure (diagnosticsEqual innovationsDiagnosticKeys actual fields) "innovations numeric replay"
  else pure ()

def fromRecord (p : Primitives) (raw : Value) : Check Model := do
  let fields ← objectFields raw "temporal record mapping"
  let payload := fields.erase "artifact_sha256"
  ensure (fieldDefault fields "artifact_sha256" = .text (p.metadataHash (.object payload))) "temporal artifact digest"
  let mode ← readMode payload
  let input ← readInput p payload mode
  let m ← constructCore p input
  checkReplay p payload m
  pure (sealModel p m)

theorem readMode_stateFields (p : Primitives) (m : CoreModel) :
    readMode (stateFields p m) = .ok m.mode := by
  cases hm : m.mode <;>
    simp [readMode,stateFields,commonStateList,extraStateList,hm,modeSchema,modeText,modeAssumption,
      fieldDefault,Fields.ofList,Fields.get,ensure]

theorem readInput_stateFields (p : Primitives) (m : CoreModel) :
    readInput p (stateFields p m) m.mode = .ok (canonicalInput m) := by
  cases hm : m.mode <;>
    simp [readInput,stateFields,commonStateList,extraStateList,hm,canonicalInput,
      fieldDefault,required,Fields.ofList,Fields.get,floatVector_encode]

theorem erase_artifact_stateFields (p : Primitives) (m : CoreModel) :
    (stateFields p m).erase "artifact_sha256" = stateFields p m := by
  cases hm : m.mode <;> simp [stateFields,commonStateList,extraStateList,hm,Fields.ofList,Fields.erase]

theorem checkReplay_self (p : Primitives) (m : CoreModel) :
    checkReplay p (stateFields p m) m = .ok () := by
  have hk : ExactKeys (stateFields p m) (fieldKeys (stateFields p m)) := ⟨fun _ h => h,fun _ h => h⟩
  have hd : ∀ keys, diagnosticsEqual keys (stateFields p m) (stateFields p m) :=
    fun _ _ _ => rfl
  simp [checkReplay,ensure,hk,hd]

theorem canonical_full_state_replay (p : Primitives) (input : ConstructorInput) (m : CoreModel)
    (h : constructCore p input = .ok m) :
    fromRecord p (canonicalRecord p m) = .ok (sealModel p m) := by
  have hc := constructCore_canonical_replay p input m h
  simp [fromRecord,canonicalRecord,objectFields,fieldDefault,Fields.get,Fields.erase,
    erase_artifact_stateFields,ensure,readMode_stateFields,readInput_stateFields,hc,checkReplay_self]

structure RecordData where
  id : String
  samples : ℤ
  inputHash : String

def encodeRecord (r : RecordData) : Value := .object (Fields.ofList
  [("record_id",.text r.id),("n_samples",.integer r.samples),("input_sha256",.text r.inputHash)])

def RecordBounds (order : ℕ) (r : RecordData) : Prop :=
  max (3 : ℤ) ((order : ℤ)+2) ≤ r.samples ∧ LowerHex64 r.inputHash

theorem encoded_record_check_exact (order : ℕ) (r : RecordData) :
    checkTemporalRecord order (encodeRecord r) = .ok (r.samples-order) ↔ RecordBounds order r := by
  simp [checkTemporalRecord,encodeRecord,objectFields,Fields.ofList,Fields.get,required,
    ExactKeys,fieldKeys,temporalRecordKeys,readExactInt,requireDigest,ensure,RecordBounds]
  split_ifs <;> simp_all

def actualLagRows (order : ℕ) (records : List RecordData) : ℤ :=
  (records.map (fun r => r.samples-(order : ℤ))).sum

theorem encoded_records_sum (order : ℕ) (records : List RecordData)
    (h : ∀ r ∈ records, RecordBounds order r) :
    checkTemporalRecords order (records.map encodeRecord) = .ok (actualLagRows order records) := by
  induction records with
  | nil => rfl
  | cons r tail ih =>
    have hr := (encoded_record_check_exact order r).mpr (h r (by simp))
    have ht := ih (fun a ha => h a (by simp [ha]))
    simp [checkTemporalRecords,hr,ht,actualLagRows]

theorem encoded_record_ids (records : List RecordData) :
    collectRecordIds (records.map encodeRecord) = .ok ((records.map RecordData.id).map Value.text) := by
  induction records with
  | nil => rfl
  | cons r tail ih => simp [collectRecordIds,encodeRecord,required,Fields.ofList,Fields.get,ih]

def trainingLedger (scope : FitScope) (records : List RecordData) (columns perParameter order : ℕ)
    (implementation : String) : Value := .object (Fields.ofList
  [("fit_scope",.text scope.text),("records",.sequence (valuesOfList (records.map encodeRecord))),
   ("method",.text "segmented_multichannel_burg_no_pole_clipping"),("assume_centered",.boolean true),
   ("available_lag_rows",.integer (actualLagRows order records)),
   ("minimum_lag_rows",.integer (max (32 : ℤ) ((perParameter : ℤ)*max 1 (order : ℤ)))),
   ("minimum_samples_per_parameter",.integer perParameter),("n_columns",.integer columns),
   ("implementation_sha256",.text implementation)])

theorem produced_training_ledger_accepted (implementation : String) (scope : FitScope)
    (records : List RecordData) (columns perParameter order : ℕ)
    (hi : Identifiers (records.map RecordData.id))
    (hr : ∀ r ∈ records, RecordBounds order r) (hc : 1 ≤ columns) (hp : 2 ≤ perParameter)
    (hl : max (32 : ℤ) ((perParameter : ℤ)*max 1 (order : ℤ)) ≤ actualLagRows order records)
    (himpl : LowerHex64 implementation) :
    checkProvenance implementation order (trainingLedger scope records columns perParameter order implementation) = .ok () := by
  have hne : records ≠ [] := fun h => hi.1 (by simp [h])
  have hs := encoded_records_sum order records hr
  have hid := (identifiers_ok_iff (.sequence (encodeTexts (records.map RecordData.id)))
    "temporal fit record IDs" (records.map RecordData.id)).mpr ⟨rfl,hi⟩
  have hc' : (1 : ℤ) ≤ columns := by omega
  have hp' : (2 : ℤ) ≤ perParameter := by omega
  have he : valuesOfList (records.map (Value.text ∘ RecordData.id)) =
      encodeTexts (records.map RecordData.id) := by rw [← List.map_map,text_list_encoding]
  cases scope <;> simp [checkProvenance,trainingLedger,objectFields,fieldDefault,required,
    Fields.ofList,Fields.get,ExactKeys,fieldKeys,temporalTrainingKeys,allowedFitScope,FitScope.text,
    ensure,sequence_truthy_iff_nonempty,hne,trainingRecords,values_list_roundtrip,
    encoded_record_ids,he,hid,readExactInt,hc',hp',hs,requireDigest,himpl,hl]

theorem allClose_self (xs : List ℝ) : allClose xs xs := by
  refine ⟨rfl,?_⟩
  intro i _
  simp only [sub_self,abs_zero]
  positivity

theorem constructCore_order_refusal_first (p : Primitives) (input : ConstructorInput)
    (h : 128 < input.coefficients.length) :
    constructCore p input = .error (.sourceReject "AR order at most 128") := by
  simp [constructCore,ensure,show ¬input.coefficients.length≤128 by omega]

theorem boolean_variance_rejected (b : Bool) :
    readVariance (.boolean b) = .error (.sourceReject "positive innovation variance") := rfl

theorem model_primary_and_derived_fields (p : Primitives) (input : ConstructorInput) (m : CoreModel)
    (h : constructCore p input = .ok m) :
    m.coefficients=input.coefficients ∧ m.provenance=input.provenance ∧ m.mode=input.mode ∧
    0 < m.variance ∧ p.cholesky m.numeric.covariance=.ok m.startupRoot ∧
    m.startupInnovationVariances=m.numeric.logVariances.map Real.exp := by
  obtain ⟨_,variance,numeric,root,hv,_,_,hr,hm,_⟩ := constructCore_components p input m h
  have hp := readVariance_result_positive input.variance variance hv
  subst m
  exact ⟨rfl,rfl,rfl,hp,hr,rfl⟩

theorem every_cached_field_recovered (p : Primitives) (input : ConstructorInput) (m : CoreModel)
    (h : constructCore p input=.ok m) :
    (fromRecord p (canonicalRecord p m)).map (fun loaded =>
      (loaded.core.coefficients,loaded.core.variance,loaded.core.provenance,loaded.core.mode,
       loaded.core.numeric.reflections,loaded.core.numeric.covariance,loaded.core.numeric.radius,
       loaded.core.numeric.startupCondition,loaded.core.numeric.factor,loaded.core.numeric.logVariances,
       loaded.core.numeric.factorCondition,loaded.core.startupRoot,loaded.core.startupInnovationVariances,
       loaded.integrity)) = .ok
      (m.coefficients,m.variance,m.provenance,m.mode,m.numeric.reflections,m.numeric.covariance,
       m.numeric.radius,m.numeric.startupCondition,m.numeric.factor,m.numeric.logVariances,
       m.numeric.factorCondition,m.startupRoot,m.startupInnovationVariances,
       p.metadataHash (.object (stateFields p m))) := by
  rw [canonical_full_state_replay p input m h]
  rfl

theorem bad_record_seal_precedes_contract (p : Primitives) (payload : Fields) (savedSeal : Value)
    (he : payload.erase "artifact_sha256"=payload)
    (h : savedSeal ≠ .text (p.metadataHash (.object payload))) :
    fromRecord p (.object (.cons "artifact_sha256" savedSeal payload)) =
      .error (.sourceReject "temporal artifact digest") := by
  simp [fromRecord,objectFields,Fields.erase,fieldDefault,Fields.get,he,ensure,h]

theorem state_schema_and_coefficients (p : Primitives) (m : CoreModel) :
    (stateFields p m).get "schema"=some (.text (modeSchema m.mode)) ∧
    (stateFields p m).get "coefficients"=some (.sequence (encodeReals m.coefficients)) := by
  simp [stateFields,commonStateList,Fields.ofList,Fields.get]

theorem stationary_schema_omits_mode (p : Primitives) (m : CoreModel) (h : m.mode=.stationary) :
    (stateFields p m).get "startup_mode"=none := by
  simp [stateFields,commonStateList,extraStateList,h,Fields.ofList,Fields.get]

theorem stationary_ignores_extra_mode_until_replay (p : Primitives) (m : CoreModel)
    (h : m.mode=.stationary) (extra : Value) :
    readMode (.cons "startup_mode" extra (stateFields p m)) = .ok .stationary := by
  simp [readMode,stateFields,commonStateList,extraStateList,h,modeSchema,modeAssumption,
    fieldDefault,Fields.ofList,Fields.get,ensure]

def implementationDigest := "d5e2383cfdabd4991765aa57f6ae6ba2d42da948968e39375e9b377b96a8ea05"

/-- Actual one-dimensional analytic primitives, with an explicit failure outside
the AR(0)/AR(1) witness domain. No fit is claimed by these synthetic references. -/
def fixturePrimitives : Primitives where
  parseFloatText s := if s="0.6" then some (3/5) else none
  inverseReflections a := match a with
    | [] => .ok []
    | [x] => if |x| < parcorMargin then .ok [-x] else .error (.sourceReject "unstable reflection")
    | _ => .error (.fieldFailure "analytic primitive outside order one")
  poleRadius a := match a with
    | [] => .ok 0
    | [x] => .ok |x|
    | _ => .error (.fieldFailure "analytic primitive outside order one")
  lyapunovCovariance a := match a with
    | [x] => .ok ⟨[1,1],[1/(1-x^2)]⟩
    | _ => .error (.fieldFailure "analytic primitive outside order one")
  startupResidualOK a covariance := match a,covariance.values with
    | [x],[v] => v*(1-x^2)=1
    | _,_ => False
  conditionNumber a := if a.shape=[1,1] then .ok 1 else .error (.fieldFailure "one dimensional condition")
  cholesky a := if a.shape=[0,0] ∨ a.shape=[1,1] then .ok ⟨a.shape,a.values.map Real.sqrt⟩
    else .error (.fieldFailure "one dimensional Cholesky")
  arrayHash _ := "array-seal"
  metadataHash _ := "metadata-seal"
  implementation := implementationDigest

def fixtureProvenance : Value := analyticLedger (.text "analytic-AR1") (.sequence .nil)

def fixtureInput (mode : Mode) : ConstructorInput := ⟨[3/5],.number 2,fixtureProvenance,mode,none⟩

def fixtureNumeric : Mode → NumericFields
  | .stationary => ⟨[-3/5],⟨[1,1],[25/16]⟩,3/5,some 1,emptyArray,[],none⟩
  | .conditionalPrefix => ⟨[-3/5],emptyArray,3/5,none,emptyArray,[],none⟩
  | .stationaryInnovations => ⟨[-3/5],emptyArray,3/5,none,factorArray [-3/5],
      N2P3.TemporalWhitening.suffixLogs [-3/5],some 1⟩

def fixtureRoot (mode : Mode) : ArrayState :=
  match mode with
  | .stationary => ⟨[1,1],[Real.sqrt (25/16)]⟩
  | _ => emptyArray

def fixtureCore (mode : Mode) : CoreModel :=
  ⟨[3/5],2,fixtureProvenance,mode,fixtureNumeric mode,fixtureRoot mode,
    (fixtureNumeric mode).logVariances.map Real.exp⟩

theorem fixture_inverse_reflections :
    fixturePrimitives.inverseReflections [3/5]=.ok [-3/5] := by
  norm_num [fixturePrimitives,parcorMargin]

theorem fixture_actual_parcor_coefficients : parcorCoefficients [-3/5]=[3/5] := by
  norm_num [parcorCoefficients,parcorStep]

theorem fixture_provenance_accepted (order : ℕ) :
    checkProvenance implementationDigest order fixtureProvenance = .ok () := by
  unfold fixtureProvenance
  rw [temporal_analytic_requires_nonblank_text]
  exact ⟨⟨"analytic-AR1",rfl,by rfl⟩,by simp [PyTruthy]⟩

theorem fixture_innovation_factor : innovationFactor fixturePrimitives [-3/5] =
    .ok ⟨factorArray [-3/5],N2P3.TemporalWhitening.suffixLogs [-3/5],1⟩ := by
  simp [innovationFactor,ensure,N2P3.TemporalWhitening.prefixScale_positive,fixturePrimitives,factorArray]

theorem fixture_numeric_fields (mode : Mode) :
    numericFields fixturePrimitives (fixtureInput mode) = .ok (fixtureNumeric mode) := by
  have hp : allClose [3/5] (parcorCoefficients [-3/5]) := by
    rw [fixture_actual_parcor_coefficients]
    exact allClose_self _
  have hselect : ∀ m, selectReflections fixturePrimitives (fixtureInput m)=.ok [-3/5] := by
    intro m
    cases m <;> simp [selectReflections,fixtureInput,ensure,fixture_inverse_reflections,
      fixtureProvenance,analyticLedger,objectFields,fieldDefault,Fields.ofList,Fields.get]
  have hradius : fixturePrimitives.poleRadius [3/5]=.ok (3/5) := by norm_num [fixturePrimitives]
  have hstart : ∀ b, startup fixturePrimitives [3/5] b = .ok
      (if b then ⟨emptyArray,3/5,none⟩ else ⟨⟨[1,1],[25/16]⟩,3/5,some 1⟩) := by
    intro b
    cases b <;> norm_num [startup,fixturePrimitives,ensure,parcorMargin]
  have hstable : ([-3/5] : List ℝ).length=([3/5] : List ℝ).length ∧
      ∀ k ∈ ([-3/5] : List ℝ), |k|<parcorMargin := by norm_num [parcorMargin]
  simp only [numericFields,hselect,Except.bind,Bind.bind]
  cases mode <;>
    simp [finishNumeric,fixtureInput,fixtureNumeric,ensure,hstable,
      hp,fixture_innovation_factor,hradius,hstart]

theorem fixture_constructs_all_three_modes (mode : Mode) :
    constructCore fixturePrimitives (fixtureInput mode) = .ok (fixtureCore mode) := by
  have hn := fixture_numeric_fields mode
  have hp : checkProvenance fixturePrimitives.implementation (fixtureInput mode).coefficients.length
      (fixtureInput mode).provenance=.ok () := fixture_provenance_accepted 1
  simp only [constructCore,hn,hp]
  cases mode <;> simp [fixtureInput,fixtureCore,fixtureNumeric,fixtureRoot,fixturePrimitives,
    emptyArray,readVariance,ensure,copyValue_preserves,marginalVariance,Real.exp_pos]

theorem fixture_three_schema_full_replay (mode : Mode) :
    fromRecord fixturePrimitives (canonicalRecord fixturePrimitives (fixtureCore mode)) =
      .ok (sealModel fixturePrimitives (fixtureCore mode)) :=
  canonical_full_state_replay _ (fixtureInput mode) _ (fixture_constructs_all_three_modes mode)

def legalTrainingRecord : RecordData := ⟨"calibration-C0",33,zeroDigest⟩
def legalTrainingLedger : Value :=
  trainingLedger .targetCalibration [legalTrainingRecord] 1 2 1 implementationDigest

theorem implementation_digest_format : LowerHex64 implementationDigest := by constructor <;> rfl

theorem legal_training_records_meet_explicit_lag_floor : actualLagRows 1 [legalTrainingRecord]=32 := by
  norm_num [actualLagRows,legalTrainingRecord]

theorem legal_training_ledger_accepted : checkProvenance implementationDigest 1 legalTrainingLedger=.ok () := by
  apply produced_training_ledger_accepted
  · refine ⟨by simp,?_,by simp⟩
    intro s hs
    have he : s="calibration-C0" := by simpa [legalTrainingRecord] using hs
    subst s
    rfl
  · intro r hr
    have he := List.mem_singleton.mp hr
    subst r
    norm_num [RecordBounds,legalTrainingRecord,zero_digest_has_lowercase_hex_format]
  · decide
  · decide
  · norm_num [legal_training_records_meet_explicit_lag_floor]
  · exact implementation_digest_format

theorem fitted_innovations_requires_actual_reflections :
    selectReflections fixturePrimitives {fixtureInput .stationaryInnovations with
      provenance:=legalTrainingLedger} = .error (.sourceReject "actual fitted PARCOR required") := by
  simp [selectReflections,fixtureInput,legalTrainingLedger,trainingLedger,FitScope.text,
    objectFields,fieldDefault,Fields.ofList,Fields.get,ensure]

def extraModeState (p : Primitives) (m : CoreModel) : Fields :=
  .cons "startup_mode" (.text "conditional_prefix") (stateFields p m)

theorem schema_one_extra_mode_fails_exact_keys :
    checkReplay fixturePrimitives (extraModeState fixturePrimitives (fixtureCore .stationary))
      (fixtureCore .stationary) = .error (.sourceReject "temporal numeric replay") := by
  have hk : ¬ExactKeys (extraModeState fixturePrimitives (fixtureCore .stationary))
      (fieldKeys (stateFields fixturePrimitives (fixtureCore .stationary))) := by
    intro h
    have hm := h.1 (show "startup_mode" ∈ fieldKeys
      (extraModeState fixturePrimitives (fixtureCore .stationary)) by simp [extraModeState,fieldKeys])
    simp [stateFields,commonStateList,extraStateList,fixtureCore,fieldKeys,Fields.ofList] at hm
  simp [checkReplay,ensure,hk]

def changedRadiusState (p : Primitives) (m : CoreModel) : Fields :=
  .cons "pole_radius" (.number 999) ((stateFields p m).erase "pole_radius")

theorem changed_radius_fails_recomputed_diagnostic :
    checkReplay fixturePrimitives (changedRadiusState fixturePrimitives (fixtureCore .stationary))
      (fixtureCore .stationary) = .error (.sourceReject "temporal numeric replay") := by
  have hd : ¬diagnosticsEqual commonDiagnosticKeys
      (stateFields fixturePrimitives (fixtureCore .stationary))
      (changedRadiusState fixturePrimitives (fixtureCore .stationary)) := by
    intro h
    have hr := h "pole_radius" (by simp [commonDiagnosticKeys])
    simp [changedRadiusState,stateFields,commonStateList,extraStateList,
      fixtureCore,fixtureNumeric,fieldDefault,Fields.ofList,Fields.get,PyEq,
      N2P3.PersonalAdmission.normalizeValue] at hr
    norm_num at hr
  simp [checkReplay,ensure,hd]

theorem finite_string_coefficients_normalize :
    floatVector fixturePrimitives (.sequence (valuesOfList [.text "0.6"])) = .ok [3/5] := by
  simp [floatVector,floatValues,floatScalar,fixturePrimitives,valuesOfList]

theorem raw_string_coefficients_not_numeric_encoding :
    (.sequence (valuesOfList [.text "0.6"]) : Value) ≠ .sequence (encodeReals [3/5]) := by
  intro h
  cases h

def foreignTrainingLedger : Value := trainingLedger .targetCalibration [legalTrainingRecord] 1 2 1 zeroDigest

theorem valid_foreign_ledger_fails_implementation_identity :
    checkProvenance implementationDigest 1 foreignTrainingLedger =
      .error (.sourceReject "temporal implementation identity") := by
  have hi : Identifiers ["calibration-C0"] :=
    ⟨by simp,by intro s hs; have he:=List.mem_singleton.mp hs; subst s; rfl,by simp⟩
  have hid := (identifiers_ok_iff (.sequence (encodeTexts ["calibration-C0"]))
    "temporal fit record IDs" ["calibration-C0"]).mpr ⟨rfl,hi⟩
  have hid' : requireIdentifiers (.sequence (.cons (.text "calibration-C0") .nil))
      "temporal fit record IDs"=.ok ["calibration-C0"] := hid
  have hneq : zeroDigest ≠ implementationDigest := by simp [zeroDigest,implementationDigest]
  simp [checkProvenance,foreignTrainingLedger,trainingLedger,objectFields,fieldDefault,
    required,Fields.ofList,Fields.get,ExactKeys,fieldKeys,temporalTrainingKeys,FitScope.text,
    allowedFitScope,ensure,PyTruthy,trainingRecords,valuesToList,valuesOfList,
    collectRecordIds,encodeRecord,legalTrainingRecord,hid',
    readExactInt,checkTemporalRecords,checkTemporalRecord,
    temporalRecordKeys,requireDigest,zero_digest_has_lowercase_hex_format,
    actualLagRows,hneq]

end

end N2P3.TemporalState
