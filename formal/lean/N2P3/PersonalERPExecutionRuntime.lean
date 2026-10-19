import N2P3.PersonalERPDataRuntime

/-!
# Source-ordered receipt/bundle refinement with actual preprocessing atoms

Source: epochs.py:523-687. Declaration parsing uses DataRuntime. Source rate
type/NumPy tests precede source-grid integers. All support span lists are read
before boundary metadata is touched, in the dictionaries' actual order.
Output names retain arbitrary builtin hashable JSON tags until the source's
support-name equality rejects mismatches; no premature string guard is added.
Padding compatibility precedes its exact-integer gate. Existing small hash,
span/causes and NumPy conversion boundaries retain inherited fault identity.
No whole execution-validation or binding-success primitive is supplied.
-/

namespace N2P3.PersonalERPExecutionRuntime
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPModels N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  data : PersonalERPDataRuntime.Primitives
  arrays : PersonalERPModelRuntime.Arrays
  roundOffset : Value → Value → R ℤ
  closeRates : Value → Value → R Bool
  roundResampled : ℤ → Value → Value → R ℤ

def execution (p : Primitives) : PersonalERPExecution.Primitives := p.data.input.original.execution

def listValue (value : Value) (site : String) : R (List Value) := do
  match value with
  | .sequence values => pure (valuesToList values)
  | _ => .error (.sourceValueError site)

def sourceGrid (p : Primitives) (fields : Fields) : R PersonalERPExecution.SourceGrid := do
  let value := fieldDefault fields "sample_rate_hz"
  PersonalERPDataRuntime.realType value "source grid rate numeric nonbool type"
  let rate ← numpyNumeric p.data.input.finite value "source grid NumPy finite rate"
  require (0 < rate) "source grid positive rate"
  let first ← PersonalERPModelRuntime.integer (fieldDefault fields "first_samp") "source first exact integer"
  let length ← PersonalERPModelRuntime.integer (fieldDefault fields "n_times") "source length exact integer"
  require (0 < length) "source positive sample count"
  pure ⟨rate,first,length⟩

def sourceBands (p : Primitives) (args : Fields) (declaration : DeclarationScalars)
    (context : Fields) : R Unit := do
  match declaration.lFreq with
  | none => pure ()
  | some frequency =>
    let nyquist ← p.data.divide (fieldDefault context "sample_rate_hz") (.integer 2)
    require (frequency < nyquist) "source high pass below actual Nyquist"
  match declaration.hFreq with
  | none => pure ()
  | some frequency =>
    let nyquist ← p.data.divide (fieldDefault context "sample_rate_hz") (.integer 2)
    require (frequency < nyquist) "source low pass below actual Nyquist"
  let _ ← PersonalERPDataRuntime.checkNotches p.data (fieldDefault args "notch_freqs_hz")
    (fieldDefault args "notch_quality_factor") (some (fieldDefault context "sample_rate_hz"))
  pure ()

structure Grid where
  fields : Fields
  resampled : ℤ

def expectedGrid (p : Primitives) (expectedArgs context : Fields) (declaration : DeclarationScalars) : R Grid := do
  let rate := fieldDefault context "sample_rate_hz"
  let modelRate := fieldDefault expectedArgs "sfreq"
  let start ← p.roundOffset (fieldDefault expectedArgs "tmin_ms") rate
  let stop ← p.roundOffset (fieldDefault expectedArgs "tmax_ms") rate
  let samples := stop-start+1
  let close ← p.closeRates rate modelRate
  let resample := !close
  let resampled ← if resample then p.roundResampled samples modelRate rate else pure samples
  let requested ← p.data.divide (fieldDefault expectedArgs "tmin_ms") (.integer 1000)
  let actual ← p.data.divide (.integer start) rate
  pure ⟨Fields.ofList [("start_offset_samples",.integer start),("end_offset_samples",.integer stop),
    ("source_n_times",.integer samples),("resampled_n_times",.integer resampled),
    ("output_n_times",.integer declaration.nTimes),("requested_tmin_s",.number requested),
    ("actual_source_tmin_s",.number actual),("resampled",.boolean resample)],resampled⟩

structure Output where
  names : List Value
  rows : ℤ
  channels : ℤ
  times : ℤ
  scheduled : ℤ

def outputGeometry (fields : Fields) (declaration : DeclarationScalars) : R Output := do
  let names ← listValue (fieldDefault fields "channel_names") "output channel names list"
  require (names ≠ []) "output channel names nonempty"
  require (∀ name ∈ names, builtinHashable name=true) "output channel names builtin hashable"
  require ((names.map PersonalAdmission.normalizeValue).Nodup) "output channel names Python unique"
  let shape ← listValue (fieldDefault fields "shape") "output shape list"
  require (shape.length=3) "output shape rank three"
  let shape ← shape.mapM (fun value => do
    let dimension ← PersonalERPModelRuntime.integer value "output dimension exact integer"
    require (0 ≤ dimension) "output dimension nonnegative"
    pure dimension)
  let dimensions ← match shape with
    | [rows,channels,times] => .ok (rows,channels,times)
    | _ => .error (.sourceValueError "output shape rank three")
  require (dimensions.2.1=(names.length : ℤ) ∧ dimensions.2.2=declaration.nTimes ∧ fieldDefault fields "dtype"=.text "float32")
    "output physical shape dtype"
  let scheduled ← PersonalERPModelRuntime.integer (fieldDefault fields "scheduled_events") "output scheduled exact integer"
  require (dimensions.1 ≤ scheduled) "output scheduled denominator"
  let _ ← liftCheck (PersonalERPExecution.hexDigest (fieldDefault fields "event_samples_sha256") "source event sample digest")
  let _ ← liftCheck (PersonalERPExecution.hexDigest (fieldDefault fields "selected_event_indices_sha256") "selected event index digest")
  match fields.get "scheduled_event_indices_sha256" with
  | none => pure ()
  | some value =>
    let _ ← liftCheck (PersonalERPExecution.hexDigest value "scheduled event index digest")
    pure ()
  pure ⟨names,dimensions.1,dimensions.2.1,dimensions.2.2,scheduled⟩

def rawScopeContext (p : Primitives) (context output : Fields) (source : PersonalERPExecution.SourceGrid)
    (geometry : Output) : R Unit := do
  match context.get "raw_scope" with
  | none => pure ()
  | some value =>
    let record ← PersonalERPModelRuntime.objectAt value "raw scope context mapping"
    let scope ← liftCheck (PersonalERPExecution.readRawScope (execution p).strip (fieldDefault record "declaration"))
    let first ← PersonalERPModelRuntime.integer (fieldDefault record "parent_first_samp") "raw scope parent first"
    let length ← PersonalERPModelRuntime.integer (fieldDefault record "parent_n_times") "raw scope parent length"
    let events ← PersonalERPModelRuntime.integer (fieldDefault record "parent_event_count") "raw scope parent event count"
    require (0 < length ∧ 0 < events) "positive raw scope parent support"
    require (first ≤ scope.start ∧ scope.stop ≤ first+length ∧ source.first=scope.start ∧ source.length=scope.stop-scope.start)
      "raw scope actual crop support"
    require (scope.eventIndices.getLastD 0 < events ∧ (scope.eventIndices.length : ℤ)=geometry.scheduled) "raw scope actual event denominator"
    let digest ← p.data.input.execution.semanticHash (.sequence (valuesOfList (scope.eventIndices.map Value.integer)))
    require (fieldDefault output "scheduled_event_indices_sha256"=.text digest ∧
      fieldDefault record "operator_order"=.text "crop_original_raw_before_filtering_and_epoching") "raw scope actual event digest operator order"

def supportLists (sourceLength : ℤ) : List (String × Value) → R (List (String × List PersonalERPExecution.Span))
  | [] => .ok []
  | (name,value) :: tail => do
    let values ← listValue value "filter support span list"
    let spans ← liftCheck (PersonalERPExecution.readSpans sourceLength 0 values)
    let rest ← supportLists sourceLength tail
    pure ((name,spans)::rest)

def boundaryLists (sourceLength : ℤ) (supports : List (String × List PersonalERPExecution.Span)) :
    List (String × Value) → R Unit
  | [] => .ok ()
  | (name,value) :: tail => do
    let rows ← listValue value "filter boundary row list"
    let spans ← match supports.lookup name with
      | none => .error (.inherited (.fieldFailure "boundary corresponding segment lookup"))
      | some spans => .ok spans
    liftCheck (PersonalERPExecution.checkBoundaryRows sourceLength rows spans)
    boundaryLists sourceLength supports tail

def namesMatchKeys (fields : Fields) (names : List Value) : Prop :=
  ((fieldKeys fields).map (fun name => PersonalAdmission.normalizeValue (.text name))).toFinset =
    (names.map PersonalAdmission.normalizeValue).toFinset

def filterSupports (context : Fields) (source : PersonalERPExecution.SourceGrid) (geometry : Output) : R Unit := do
  let segments ← PersonalERPModelRuntime.objectAt (fieldDefault context "filter_segments") "filter support mapping"
  require (namesMatchKeys segments geometry.names) "exact per channel filter supports"
  let supports ← supportLists source.length (fieldPairs segments)
  match context.get "filter_segment_boundaries" with
  | none => pure ()
  | some value =>
    let boundaries ← PersonalERPModelRuntime.objectAt value "filter boundary mapping"
    require (namesMatchKeys boundaries geometry.names) "exact filter boundary channels"
    boundaryLists source.length supports (fieldPairs boundaries)

def filterPolicy (p : Primitives) (filtering : Fields) (raw : RawPreprocessing) (declaration : DeclarationScalars) : R Unit := do
  let sos ← p.data.input.execution.floatArray (fieldDefault filtering "sos")
  if sos.values ≠ [] then
    require (∃ rows, sos.shape=[rows,6]) "nonempty SOS rank two six columns"
    arrayFinite p.arrays sos "execution SOS finite all"
    require (∀ row ∈ List.range (sos.shape.headD 0), (sos.values[row*6+3]?).bind atomNumber=some 1) "execution SOS normalized denominator"
  let active := declaration.lFreq.isSome || declaration.hFreq.isSome || !declaration.notches.isEmpty
  let backend := PersonalERPExecution.expectedFilterBackend raw declaration
  require (decide (sos.values ≠ [])=active ∧ fieldDefault filtering "backend"=.text backend) "execution filter active backend"
  if backend="identity" ∨ backend="scipy.sosfilt" then
    require (PersonalAdmission.PyEq (fieldDefault filtering "padding_limit_samples") (.integer 0)) "identity forward zero padding compatibility"
  let padding ← PersonalERPModelRuntime.integer (fieldDefault filtering "padding_limit_samples") "execution padding exact integer"
  require (0 ≤ padding ∧ fieldDefault filtering "initial_state"=raw.initialState) "execution padding initial state"

structure Receipt where
  original : Fields
  declaration : Fields
  source : PersonalERPExecution.SourceGrid
  output : Output

def checkIndividual (p : Primitives) (expectedArgs fields : Fields) : R Receipt := do
  require (fieldDefault fields "schema"=.text "n2p3_preprocessing_execution/1") "individual receipt schema"
  let digest ← p.data.input.execution.semanticHash (.object (fields.erase "receipt_sha256"))
  require (fieldDefault fields "receipt_sha256"=.text digest) "receipt actual semantic digest"
  let declared ← PersonalERPModelRuntime.objectAt (fieldDefault fields "declaration") "receipt preprocessing declaration mapping"
  let declared ← liftCheck (readCompletePreprocessingArgs declared)
  let declaration ← PersonalERPDataRuntime.checkPreprocessing p.data declared
  require (PersonalAdmission.PyEq (.object (expectedArgs.erase "name")) (.object (declared.erase "name"))) "receipt executed asdict physical declaration"
  require (PersonalAdmission.PyEq (fieldDefault fields "semantic_identity") PersonalERPExecution.operatorIdentity) "receipt supported operator identity"
  liftCheck (PersonalERPExecution.checkImplementation fields)
  let context ← PersonalERPModelRuntime.objectAt (fieldDefault fields "source_context") "receipt source context"
  let grid ← PersonalERPModelRuntime.objectAt (fieldDefault fields "epoch_grid") "receipt epoch grid"
  let output ← PersonalERPModelRuntime.objectAt (fieldDefault fields "output") "receipt output mapping"
  let filtering ← PersonalERPModelRuntime.objectAt (fieldDefault fields "filter") "receipt filter mapping"
  let source ← sourceGrid p context
  sourceBands p declared declaration context
  let expected ← expectedGrid p expectedArgs context declaration
  require (PersonalAdmission.PyEq (.object grid) (.object expected.fields)) "actual source epoch grid equality"
  require (declaration.nTimes ≤ expected.resampled) "resampled output coverage"
  let geometry ← outputGeometry output declaration
  rawScopeContext p context output source geometry
  filterSupports context source geometry
  filterPolicy p filtering (rawPreprocessing declared) declaration
  let _ ← liftCheck (PersonalERPExecution.nonemptyText (fieldDefault fields "timing_source") "execution nonempty timing source")
  pure ⟨fields,declared,source,geometry⟩

def children (p : Primitives) (expectedArgs : Fields) : List Value → R (List Receipt)
  | [] => .ok []
  | value :: tail => do
    let fields ← PersonalERPModelRuntime.objectAt value "bundle child dictionary"
    require (fieldDefault fields "schema"=.text "n2p3_preprocessing_execution/1") "bundle contains individual receipts only"
    let child ← checkIndividual p expectedArgs fields
    let rest ← children p expectedArgs tail
    pure (child::rest)

structure CheckedRecord where
  original : Fields
  receipts : List Receipt
  unknown : Value

def checkRecord (p : Primitives) (expectedArgs : Fields) (value : Value) : R CheckedRecord := do
  let fields ← PersonalERPModelRuntime.objectAt value "execution receipt dictionary"
  if fieldDefault fields "schema"=.text "n2p3_preprocessing_execution_bundle/1" then
    let values ← listValue (fieldDefault fields "recordings") "execution bundle recording list"
    let unknown ← PersonalERPModelRuntime.integer (fieldDefault fields "unverified_inputs") "execution bundle unknown exact integer"
    require (0 ≤ unknown ∧ (values ≠ [] ∨ unknown ≠ 0)) "nonnegative unknown nonempty bundle"
    let receipts ← children p expectedArgs values
    pure ⟨fields,receipts,.integer unknown⟩
  else
    let receipt ← checkIndividual p expectedArgs fields
    pure ⟨fields,[receipt],(fields.get "unverified_inputs").getD (.integer 0)⟩

def executionBinding (p : Primitives) (expectedArgs : Fields) (value : Option Value) : R (Option PersonalERPExecution.Binding) := do
  match value with
  | none | some .null => pure none
  | some value =>
    let record ← checkRecord p expectedArgs value
    let digest ← p.data.input.execution.semanticHash (.object record.original)
    let identity := if PyTruthy record.unknown ∨ record.receipts=[] then none else some PersonalERPExecution.operatorIdentity
    pure (some ⟨identity,digest⟩)

theorem absent_execution_does_not_run_preprocessing (p : Primitives) (args : Fields) :
    executionBinding p args none=.ok none := rfl

end
end N2P3.PersonalERPExecutionRuntime
